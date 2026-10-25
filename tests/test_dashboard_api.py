"""Integration tests for the dashboard API, run against the real saved model in outputs/."""
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import pytest

from dashboard import auth, config, db, service

ROOT = Path(__file__).resolve().parents[1]
PW = "Sturdy-pass-123"


def make_user(client, name="alice", role="user", password=PW):
    r = client.post("/api/auth/register", json={"username": name, "password": password})
    assert r.status_code == 201, r.text
    if role == "admin":
        db.execute("UPDATE users SET role='admin' WHERE username=?", (name,))
    return login(client, name, password)


def login(client, name, password=PW, admin=False):
    r = client.post("/api/auth/admin/login" if admin else "/api/auth/login", json={"username": name, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode()


def original() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "kharghar_period_totals.csv")


def upload(client, headers, frame=None, name="data.csv", raw=None):
    raw = raw if raw is not None else csv_bytes(frame)
    return client.post("/api/datasets/upload", headers=headers, files={"file": (name, raw, "text/csv")})


# ------------------------------------------------------------ original app preserved
def test_original_chatbot_endpoints_still_work(client):
    assert client.get("/api/health").json()["status"] == "ok"
    r = client.post("/api/chat", json={"message": "help"})
    assert r.status_code == 200 and r.json()["reply"]


# ------------------------------------------------------------ auth + authorization
def test_protected_routes_require_login(client):
    for path in ("/api/dashboard/overview", "/api/models/catalog", "/api/datasets", "/api/predictions", "/api/admin/users"):
        assert client.get(path).status_code == 401, path


def test_password_is_hashed_and_policy_enforced(client):
    assert client.post("/api/auth/register", json={"username": "bob", "password": "short1"}).status_code == 422
    assert client.post("/api/auth/register", json={"username": "bob", "password": "onlyletterslong"}).status_code == 422
    make_user(client, "bob")
    stored = db.one("SELECT password_hash FROM users WHERE username='bob'")["password_hash"]
    assert PW not in stored and stored.startswith("$2")


def test_public_registration_never_grants_admin(client):
    r = client.post("/api/auth/register", json={"username": "mallory", "password": PW, "role": "admin"})
    assert r.status_code == 201 and r.json()["user"]["role"] == "user"
    h = login(client, "mallory")
    assert client.get("/api/admin/overview", headers=h).status_code == 403


def test_admin_login_rejects_normal_user_and_accepts_admin(client):
    make_user(client, "carol")
    assert client.post("/api/auth/admin/login", json={"username": "carol", "password": PW}).status_code == 401
    make_user(client, "root", role="admin")
    h = login(client, "root", admin=True)
    assert client.get("/api/admin/overview", headers=h).status_code == 200


def test_bad_login_is_audited_and_rate_limited(client):
    make_user(client, "dave")
    for _ in range(5):
        assert client.post("/api/auth/login", json={"username": "dave", "password": "wrong-pass-1"}).status_code == 401
    assert client.post("/api/auth/login", json={"username": "dave", "password": PW}).status_code == 429
    assert db.one("SELECT COUNT(*) n FROM audit_log WHERE action='login' AND outcome='failed'")["n"] == 5


def test_tampered_or_expired_token_rejected(client):
    make_user(client, "erin")
    assert client.get("/api/auth/me", headers={"Authorization": "Bearer not.a.token"}).status_code == 401


def test_deactivated_user_loses_access_immediately(client):
    h_user = make_user(client, "frank")
    make_user(client, "boss", role="admin")
    h_admin = login(client, "boss", admin=True)
    uid = client.get("/api/auth/me", headers=h_user).json()["user"]["id"]
    assert client.patch(f"/api/admin/users/{uid}", headers=h_admin, json={"active": False}).status_code == 200
    assert client.get("/api/auth/me", headers=h_user).status_code == 401


def test_cannot_demote_or_deactivate_last_admin(client):
    make_user(client, "solo", role="admin")
    h = login(client, "solo", admin=True)
    uid = client.get("/api/auth/me", headers=h).json()["user"]["id"]
    assert client.patch(f"/api/admin/users/{uid}", headers=h, json={"role": "user"}).status_code == 400
    assert client.patch(f"/api/admin/users/{uid}", headers=h, json={"active": False}).status_code == 400


def test_audit_log_never_contains_passwords_or_tokens(client):
    h = make_user(client, "gina")
    client.post("/api/auth/login", json={"username": "gina", "password": "Wrong-password-9"})
    blob = json.dumps(db.rows("SELECT * FROM audit_log"))
    assert PW not in blob and "Wrong-password-9" not in blob and h["Authorization"].split()[1] not in blob


# ------------------------------------------------------------ evaluation views use real artifacts
def test_model_catalog_marks_weights_and_matches_saved_metrics(client):
    h = make_user(client, "hank")
    cat = client.get("/api/models/catalog?metric=rmse", headers=h).json()
    by = {m["key"]: m for m in cat["models"]}
    assert set(by) == {"cnn_lstm_attention", "cnn_lstm", "lstm_attention", "lstm", "naive_last_value", "seasonal_naive_6"}
    assert [k for k, m in by.items() if m["has_saved_weights"]] == ["cnn_lstm_attention"]
    run = service.resolve_run_dir()
    saved = pd.read_csv(run / "metrics.csv").set_index("model")
    for k, m in by.items():
        assert m["rmse"] == pytest.approx(saved.loc[k, "rmse"], rel=1e-9)
        assert m["mae"] == pytest.approx(saved.loc[k, "mae"], rel=1e-9)
        assert m["r2"] == pytest.approx(saved.loc[k, "r2"], rel=1e-9)
        assert m["mse"] == pytest.approx(saved.loc[k, "rmse"] ** 2, rel=1e-9)
    assert cat["best"]["key"] == min(by, key=lambda k: by[k]["rmse"])
    assert client.get("/api/models/catalog?metric=r2", headers=h).json()["best"]["key"] == max(by, key=lambda k: by[k]["r2"])


def test_training_view_does_not_fabricate_loss_curves(client):
    h = make_user(client, "ivy")
    t = client.get("/api/models/training", headers=h).json()
    assert t["loss_history"]["recorded"] is False
    assert t["validation"]["loss_recorded"] is False
    assert "epochs" in json.dumps(t["training"]["final_training_epochs"]) or t["training"]["final_training_epochs"] == 22


def test_validation_panel_has_honest_statuses(client):
    h = make_user(client, "jack")
    p = client.get("/api/models/validation", headers=h).json()
    st = {c["name"]: c["status"] for c in p["checks"]}
    assert st["Training/validation loss history"] == "not_evaluated"
    assert st["Saved forecast is reproduced"] == "passed"
    assert st["R-squared on test bills"] == "warning"       # real R2 is negative
    assert p["overall"] == "warning"


def test_confusion_matrix_accuracy_is_diagonal_over_total(client):
    h = make_user(client, "kim")
    c = client.get("/api/models/confusion?model=cnn_lstm_attention", headers=h).json()
    m = c["matrix"]
    assert sum(map(sum, m)) == c["n"] == 8
    assert c["accuracy"] == pytest.approx((m[0][0] + m[1][1] + m[2][2]) / c["n"])
    assert c["kind"] == "derived_category_classification"
    assert c["majority_class_accuracy"] == pytest.approx(max(map(sum, m)) / c["n"])
    assert c["accuracy_informative"] is False and any("single class" in w for w in c["warnings"])
    run = service.resolve_run_dir()
    hist = pd.read_csv(run / "cleaned_period_totals.csv")["total_qty"].to_numpy()[:24]
    import numpy as np
    assert c["thresholds"]["low_below"] == pytest.approx(float(np.quantile(hist, 1 / 3)))  # train-only thresholds
    assert client.get("/api/models/confusion?model=nope", headers=h).status_code == 404
    assert client.get("/api/models/confusion?q_low=0.8&q_high=0.5", headers=h).status_code == 422


def test_walk_forward_residuals_are_actual_minus_predicted(client):
    h = make_user(client, "lee")
    w = client.get("/api/models/walk-forward", headers=h).json()
    k = "cnn_lstm_attention"
    assert w["residuals"][k][0] == pytest.approx(w["actual"][0] - w["predictions"][k][0])


# ------------------------------------------------------------ upload, validation, prediction
def test_valid_upload_confirm_predict_export_flow(client):
    h = make_user(client, "mona")
    frame = original().iloc[:14]
    r = upload(client, h, frame)
    assert r.status_code == 201 and r.json()["status"] == "pending_confirmation"
    did = r.json()["dataset_id"]
    assert r.json()["validation"]["status"] in ("passed", "warning")
    # prediction before confirmation is refused
    assert client.post("/api/predictions", headers=h, json={"dataset_id": did, "horizon": 2}).status_code == 409
    assert client.post(f"/api/datasets/{did}/confirm", headers=h).json()["status"] == "ready"
    pv = client.get(f"/api/datasets/{did}/preview?limit=5", headers=h).json()
    assert pv["total_rows"] == 14 and len(pv["rows"]) == 5
    p = client.post("/api/predictions", headers=h, json={"dataset_id": did, "horizon": 3})
    assert p.status_code == 201, p.text
    body = p.json()
    assert len(body["forecast"]) == 3 and body["forecast"][0]["method"] == "one-step forecast"
    assert "autoregressive" in body["forecast"][1]["method"]
    assert len(body["backtest"]["rows"]) == 14 - 6
    assert body["backtest"]["in_sample"] is True          # these periods were in the training data
    assert all(f["lower"] is not None for f in body["forecast"])     # horizon <= 3 has measured errors
    csv_text = client.get(f"/api/predictions/{body['prediction_id']}/csv", headers=h).text
    assert "forecast" in csv_text and body["model"]["variant"] in csv_text
    hist = client.get("/api/predictions", headers=h).json()["predictions"]
    assert len(hist) == 1 and hist[0]["dataset_id"] == did


def test_forecast_matches_saved_original_forecast(client):
    h = make_user(client, "nina")
    p = client.post("/api/predictions", headers=h, json={"dataset_id": 0, "horizon": 1}).json()
    saved = pd.read_csv(service.resolve_run_dir() / "forecast_next_period.csv")
    assert p["forecast"][0]["forecast"] == pytest.approx(float(saved["forecast_total_qty"].iloc[0]), abs=0.01)
    assert p["forecast"][0]["label"] == saved["billing_period"].iloc[0]
    assert p["backtest"]["in_sample"] is True and "NOT a test" in p["backtest"]["scope"]


def test_horizon_beyond_measured_errors_has_no_range(client):
    h = make_user(client, "omar")
    p = client.post("/api/predictions", headers=h, json={"dataset_id": 0, "horizon": 5}).json()
    assert p["forecast"][3]["lower"] is None and p["forecast"][4]["upper"] is None
    assert client.post("/api/predictions", headers=h, json={"dataset_id": 0, "horizon": 7}).status_code == 422


@pytest.mark.parametrize("mutate,needle", [
    (lambda f: f.drop(columns=["total_qty"]), "Missing required column"),
    (lambda f: f.iloc[:4], "model reads 6"),
    (lambda f: f.drop(index=[3]).reset_index(drop=True), "not continuous"),
    (lambda f: f.assign(total_qty=f["total_qty"].where(f.index != 2, -5)), "negative"),
    (lambda f: f.assign(total_qty=f["total_qty"].astype(object).where(f.index != 2, "abc")), "not numeric"),
    (lambda f: f.assign(period_start=f["period_start"].where(f.index != 1, "not-a-date")), "valid dates"),
    (lambda f: pd.concat([f, f.iloc[[2]]], ignore_index=True), "exactly once"),
])
def test_incompatible_files_are_rejected_with_reason_and_not_stored(client, mutate, needle):
    h = make_user(client, "pat")
    r = upload(client, h, mutate(original().iloc[:12].copy()))
    assert r.status_code == 422, r.text
    errs = " ".join(r.json()["validation"]["errors"])
    assert needle in errs
    assert not list(config.upload_dir().rglob("*.csv"))
    row = db.one("SELECT status, stored_path FROM datasets WHERE id=?", (r.json()["dataset_id"],))
    assert row["status"] == "rejected" and row["stored_path"] is None


def test_bad_extension_empty_oversize_and_fake_xlsx_rejected(client, monkeypatch):
    h = make_user(client, "quin")
    assert upload(client, h, raw=b"x", name="evil.exe").status_code == 422
    assert upload(client, h, raw=b"", name="empty.csv").status_code == 422
    assert upload(client, h, raw=b"not a zip", name="fake.xlsx").status_code == 422
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 100)
    assert upload(client, h, raw=b"a,b\n" * 100, name="big.csv").status_code == 422
    assert not list(config.upload_dir().rglob("*.*"))


def test_xlsx_upload_works(client):
    h = make_user(client, "rosa")
    buf = io.BytesIO()
    original().iloc[:12].to_excel(buf, index=False)
    r = upload(client, h, raw=buf.getvalue(), name="d.xlsx")
    assert r.status_code == 201, r.text


def test_out_of_range_values_warn_but_are_not_modified(client):
    h = make_user(client, "sam")
    frame = original().iloc[:12].copy()
    frame.loc[5, "total_qty"] = 99_000_000
    r = upload(client, h, frame)
    assert r.status_code == 201 and r.json()["validation"]["status"] == "warning"
    assert any("outside the range" in w for w in r.json()["validation"]["warnings"])


def test_other_users_cannot_see_or_use_a_dataset_but_admin_can(client):
    ha = make_user(client, "tina")
    hb = make_user(client, "uma")
    did = upload(client, ha, original().iloc[:12]).json()["dataset_id"]
    assert client.get(f"/api/datasets/{did}/preview", headers=hb).status_code == 404
    assert client.delete(f"/api/datasets/{did}", headers=hb).status_code == 404
    assert did not in [d["id"] for d in client.get("/api/datasets", headers=hb).json()["datasets"]]
    make_user(client, "vic", role="admin")
    hadm = login(client, "vic", admin=True)
    assert client.get(f"/api/datasets/{did}/preview", headers=hadm).status_code == 200


def test_original_dataset_is_protected_and_unchanged(client):
    h = make_user(client, "walt")
    before = hashlib.sha256((ROOT / "data" / "kharghar_period_totals.csv").read_bytes()).hexdigest()
    assert client.delete("/api/datasets/0", headers=h).status_code == 403
    client.post("/api/predictions", headers=h, json={"dataset_id": 0, "horizon": 2})
    assert hashlib.sha256((ROOT / "data" / "kharghar_period_totals.csv").read_bytes()).hexdigest() == before


def test_delete_removes_file_and_is_audited(client):
    h = make_user(client, "xena")
    did = upload(client, h, original().iloc[:12]).json()["dataset_id"]
    path = Path(db.one("SELECT stored_path FROM datasets WHERE id=?", (did,))["stored_path"])
    assert path.exists()
    assert client.delete(f"/api/datasets/{did}", headers=h).status_code == 200
    assert not path.exists()
    assert db.one("SELECT COUNT(*) n FROM audit_log WHERE action='dataset_delete'")["n"] == 1


def test_uploads_and_predictions_never_touch_model_files(client):
    run = service.resolve_run_dir()
    files = [run / "model.pt", run / "scaler.json", run / "model_bundle.json"]
    before = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    h = make_user(client, "yara")
    did = upload(client, h, original().iloc[:20]).json()["dataset_id"]
    client.post(f"/api/datasets/{did}/confirm", headers=h)
    client.post("/api/predictions", headers=h, json={"dataset_id": did, "horizon": 3})
    after = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    assert before == after


# ------------------------------------------------------------ admin
def test_admin_overview_models_audit_and_reports(client):
    make_user(client, "zed", role="admin")
    h = login(client, "zed", admin=True)
    hu = make_user(client, "user1")
    upload(client, hu, original().iloc[:2])       # rejected -> shows in recent errors
    client.post("/api/predictions", headers=hu, json={"dataset_id": 0, "horizon": 1})
    ov = client.get("/api/admin/overview", headers=h).json()
    assert ov["counts"]["predictions"] == 1 and ov["counts"]["datasets_rejected"] == 1
    assert ov["counts"]["trained_models_with_weights"] == 1
    assert any(e["action"] == "dataset_upload" for e in ov["recent_errors"])
    models = client.get("/api/admin/models", headers=h).json()
    assert models["loaded"] and models["retraining"]["supported"] is False
    ev = client.post("/api/admin/models/evaluate", headers=h).json()
    assert len(ev["models"]) == 6
    audit = client.get("/api/admin/audit?action=model_evaluation", headers=h).json()["entries"]
    assert len(audit) == 1
    assert client.get("/api/admin/predictions", headers=h).json()["predictions"][0]["username"] == "user1"
    assert "id,user_id" in client.get("/api/admin/predictions.csv", headers=h).text
    assert client.get("/api/admin/audit.csv", headers=h).status_code == 200
    # normal users are blocked server-side from every admin route
    for path in ("/api/admin/overview", "/api/admin/models", "/api/admin/audit", "/api/admin/users", "/api/admin/datasets"):
        assert client.get(path, headers=hu).status_code == 403, path
    assert client.post("/api/admin/models/evaluate", headers=hu).status_code == 403


# ------------------------------------------------------------ failure handling
def test_missing_model_files_give_clear_503_not_fake_results(client, tmp_path, monkeypatch):
    h = make_user(client, "ann")
    empty = tmp_path / "no_outputs"
    empty.mkdir()
    monkeypatch.setenv("WATER_OUTPUTS_DIR", str(empty))
    service.clear_cache()
    r = client.get("/api/models/catalog", headers=h)
    assert r.status_code == 503 and "Model unavailable" in r.json()["detail"]
    assert client.post("/api/predictions", headers=h, json={"dataset_id": 0, "horizon": 1}).status_code == 503


def test_csv_export_neutralises_formula_injection(client):
    from dashboard.routes import csv_response
    body = csv_response("x.csv", ["a"], [["=HYPERLINK(\"http://evil\")"], ["ok"]]).body.decode()
    assert "'=HYPERLINK" in body and "\nok" in body


def test_analytics_filters_by_date_range(client):
    h = make_user(client, "bea")
    full = client.get("/api/dashboard/analytics", headers=h).json()
    part = client.get("/api/dashboard/analytics?start=2024-01-01&end=2025-12-31", headers=h).json()
    assert len(part["labels"]) < len(full["labels"]) and part["stats"]["records"] == len(part["labels"])
    empty = client.get("/api/dashboard/analytics?start=2030-01-01", headers=h).json()
    assert empty["empty"] is True
    assert client.get("/api/dashboard/analytics?start=garbage", headers=h).status_code == 422


def test_overview_cards_come_from_real_data(client):
    h = make_user(client, "cyrus")
    o = client.get("/api/dashboard/overview", headers=h).json()
    df = original()
    assert o["stats"]["records"] == len(df) and o["stats"]["total"] == pytest.approx(df["total_qty"].sum())
    assert o["stats"]["max"] == df["total_qty"].max() and o["stats"]["latest_label"] == df["billing_period"].iloc[-1]
    assert o["best_model"]["label"] == "CNN-LSTM + Attention"
