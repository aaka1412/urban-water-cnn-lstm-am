"""HTTP API for the dashboard. Every route except /auth/register, /auth/login and /auth/admin/login
requires a valid token; /admin/* additionally requires the admin role (checked server-side)."""
from __future__ import annotations

import csv
import io
import json
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from . import auth, config, db, service, validation
from .service import ModelUnavailable, py

router = APIRouter(prefix="/api")
ORIGINAL_ID = 0


# ------------------------------------------------------------------ helpers
def csv_response(filename: str, header: list[str], rows: list[list]) -> Response:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    for row in rows:
        # neutralise spreadsheet formula injection in text cells
        writer.writerow([("'" + c) if isinstance(c, str) and c[:1] in "=+-@\t\r" else c for c in row])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


def sequence_length() -> int:
    return int(service.get_bundle()[2]["sequence_length"])


def dataset_summary(d: dict) -> dict:
    return {"id": d["id"], "name": d["name"], "original_filename": d["original_filename"], "size_bytes": d["size_bytes"],
            "n_rows": d["n_rows"], "n_cols": d["n_cols"], "status": d["status"],
            "validation_status": d["validation_status"], "uploaded_at": d["uploaded_at"], "owner_id": d["owner_id"],
            "is_original": False}


def original_summary() -> dict:
    p = config.original_dataset()
    n = len(service.load_original_frame()) if p.exists() else None
    return {"id": ORIGINAL_ID, "name": "Original project dataset (Kharghar billing-period totals)",
            "original_filename": p.name, "size_bytes": p.stat().st_size if p.exists() else None, "n_rows": n,
            "n_cols": 7, "status": "ready", "validation_status": "passed", "uploaded_at": None, "owner_id": None,
            "is_original": True, "protected": True}


def get_dataset(user: dict, dataset_id: int) -> dict:
    if dataset_id == ORIGINAL_ID:
        return original_summary()
    d = db.one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    if not d or d["status"] == "deleted" or (d["owner_id"] != user["id"] and user["role"] != "admin"):
        raise HTTPException(404, "Dataset not found.")
    return d


def load_frame(d: dict) -> pd.DataFrame:
    if d["id"] == ORIGINAL_ID:
        return service.load_original_frame()
    if d["status"] not in ("ready", "pending_confirmation") or not d.get("stored_path"):
        raise HTTPException(409, "This dataset was rejected or has no stored file.")
    path = Path(d["stored_path"])
    if config.upload_dir() not in path.resolve().parents:
        raise HTTPException(500, "Stored file path is outside the upload folder.")
    raw = path.read_bytes()
    return validation.to_billing_frame(validation.read_table(raw, d["original_filename"]))


def need_model() -> None:
    service.get_bundle()


# ------------------------------------------------------------------ auth
class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=1, max_length=200)


class RegisterBody(Credentials):
    email: str | None = Field(default=None, max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


def _login(body: Credentials, request: Request, admin_only: bool) -> dict:
    ip = auth.client_ip(request)
    key = f"{body.username.lower()}|{ip}"
    auth.check_rate_limit(key)
    user = db.one("SELECT * FROM users WHERE username = ?", (body.username,))
    ok = bool(user and user["active"] and auth.verify_password(body.password, user["password_hash"])
              and (user["role"] == "admin" or not admin_only))
    if not ok:
        auth.record_failure(key)
        db.audit("admin_login" if admin_only else "login", "failed", username=body.username, ip=ip,
                 detail="invalid credentials or not permitted")
        raise HTTPException(401, "Invalid username or password.")
    auth.clear_failures(key)
    db.execute("UPDATE users SET last_login = ? WHERE id = ?", (db.now(), user["id"]))
    db.audit("admin_login" if admin_only else "login", "ok", user, ip=ip)
    return {"token": auth.make_token(user), "user": auth.public_user(user)}


@router.post("/auth/register", status_code=201)
def register(body: RegisterBody, request: Request):
    """Public sign-up always creates a normal 'user'. Admins are created only with the CLI or by another admin."""
    if not auth.USERNAME_RE.match(body.username):
        raise HTTPException(422, "Username must be 3-32 characters: letters, digits, _ . -")
    auth.validate_password(body.password)
    if db.one("SELECT id FROM users WHERE username = ?", (body.username,)):
        raise HTTPException(409, "That username is taken.")
    uid = db.execute("INSERT INTO users (username,email,password_hash,role,active,created_at) VALUES (?,?,?,?,1,?)",
                     (body.username, body.email, auth.hash_password(body.password), "user", db.now()))
    user = db.one("SELECT * FROM users WHERE id = ?", (uid,))
    db.audit("register", "ok", user, ip=auth.client_ip(request))
    return {"user": auth.public_user(user)}


@router.post("/auth/login")
def login(body: Credentials, request: Request):
    return _login(body, request, admin_only=False)


@router.post("/auth/admin/login")
def admin_login(body: Credentials, request: Request):
    return _login(body, request, admin_only=True)


@router.get("/auth/me")
def me(user: dict = Depends(auth.current_user)):
    return {"user": auth.public_user(user)}


@router.post("/auth/logout")
def logout(request: Request, user: dict = Depends(auth.current_user)):
    db.audit("logout", "ok", user, ip=auth.client_ip(request))  # tokens are stateless; the client discards it
    return {"ok": True}


@router.post("/auth/change-password")
def change_password(body: PasswordChange, request: Request, user: dict = Depends(auth.current_user)):
    if not auth.verify_password(body.current_password, user["password_hash"]):
        db.audit("password_change", "failed", user, ip=auth.client_ip(request))
        raise HTTPException(403, "Current password is incorrect.")
    auth.validate_password(body.new_password)
    db.execute("UPDATE users SET password_hash = ? WHERE id = ?", (auth.hash_password(body.new_password), user["id"]))
    db.audit("password_change", "ok", user, ip=auth.client_ip(request))
    return {"ok": True}


# ------------------------------------------------------------------ dataset management
@router.get("/datasets")
def list_datasets(user: dict = Depends(auth.current_user)):
    rows = db.rows("SELECT * FROM datasets WHERE owner_id = ? AND status != 'deleted' ORDER BY id DESC", (user["id"],))
    return {"original": original_summary(), "datasets": [dataset_summary(r) for r in rows]}


@router.post("/datasets/upload", status_code=201)
async def upload_dataset(request: Request, file: UploadFile = File(...), user: dict = Depends(auth.current_user)):
    ip = auth.client_ip(request)
    filename = Path(file.filename or "upload").name[:120]
    ext = Path(filename).suffix.lower()
    raw = await file.read(config.MAX_UPLOAD_BYTES + 1)

    def reject(report: dict, size: int):
        did = db.execute(
            "INSERT INTO datasets (owner_id,name,original_filename,stored_path,size_bytes,n_rows,n_cols,status,"
            "validation_status,validation_json,uploaded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (user["id"], filename, filename, None, size, report.get("n_rows"), report.get("n_cols"), "rejected",
             "failed", json.dumps(py(report)), db.now()))
        db.audit("dataset_upload", "rejected", user, detail=f"{filename}: {(report['errors'] or ['invalid'])[0]}", ip=ip)
        return JSONResponse(status_code=422, content={"dataset_id": did, "detail": "Dataset rejected.", "validation": py(report)})

    base = {"n_rows": None, "n_cols": None, "checks": [], "warnings": [], "columns": [], "preview": [],
            "status": "failed", "compatible": False, "missing_values": 0, "duplicate_records": 0,
            "invalid_dates": 0, "invalid_values": 0}
    if ext not in config.ALLOWED_EXTENSIONS:
        return reject({**base, "errors": ["Only .csv and .xlsx files are accepted."]}, len(raw))
    if len(raw) > config.MAX_UPLOAD_BYTES:
        return reject({**base, "errors": [f"File is larger than {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB."]}, len(raw))
    if not raw:
        return reject({**base, "errors": ["The file is empty."]}, 0)
    try:
        frame = validation.read_table(raw, filename)
    except Exception as exc:
        return reject({**base, "errors": [f"The file could not be read as a table ({type(exc).__name__})."]}, len(raw))
    need_model()
    lo, hi = service.training_range()
    report = validation.validate_frame(frame, sequence_length(), lo, hi)
    if not report["compatible"]:
        return reject(report, len(raw))
    owner_dir = config.upload_dir() / str(user["id"])
    owner_dir.mkdir(parents=True, exist_ok=True)
    stored = owner_dir / f"{uuid.uuid4().hex}{ext}"
    stored.write_bytes(raw)
    did = db.execute(
        "INSERT INTO datasets (owner_id,name,original_filename,stored_path,size_bytes,n_rows,n_cols,status,"
        "validation_status,validation_json,uploaded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (user["id"], filename, filename, str(stored), len(raw), report["n_rows"], report["n_cols"],
         "pending_confirmation", report["status"], json.dumps(py(report)), db.now()))
    db.audit("dataset_upload", "ok", user, detail=f"{filename} ({report['n_rows']} rows, {report['status']})", ip=ip)
    return {"dataset_id": did, "status": "pending_confirmation", "validation": py(report)}


@router.post("/datasets/{dataset_id}/confirm")
def confirm_dataset(dataset_id: int, request: Request, user: dict = Depends(auth.current_user)):
    if dataset_id == ORIGINAL_ID:
        raise HTTPException(400, "The original dataset is always available.")
    d = get_dataset(user, dataset_id)
    if d["status"] != "pending_confirmation":
        raise HTTPException(409, f"Dataset is '{d['status']}', not waiting for confirmation.")
    db.execute("UPDATE datasets SET status='ready' WHERE id=?", (dataset_id,))
    db.audit("dataset_confirm", "ok", user, detail=d["name"], ip=auth.client_ip(request))
    return {"dataset_id": dataset_id, "status": "ready"}


@router.get("/datasets/{dataset_id}/preview")
def preview_dataset(dataset_id: int, limit: int = Query(25, ge=1, le=200), user: dict = Depends(auth.current_user)):
    d = get_dataset(user, dataset_id)
    frame = load_frame(d)
    head = frame.head(limit).copy()
    for c in ("period_start", "period_end"):
        head[c] = head[c].dt.strftime("%Y-%m-%d")
    return {"dataset": {"id": d["id"], "name": d["name"]}, "total_rows": int(len(frame)),
            "columns": list(frame.columns), "rows": py(head.to_dict("records"))}


@router.get("/datasets/{dataset_id}/validation")
def dataset_validation(dataset_id: int, user: dict = Depends(auth.current_user)):
    d = get_dataset(user, dataset_id)
    if d["id"] == ORIGINAL_ID:
        need_model()
        lo, hi = service.training_range()
        frame = pd.read_csv(config.original_dataset())
        return {"dataset_id": 0, "validation": py(validation.validate_frame(frame, sequence_length(), lo, hi))}
    return {"dataset_id": dataset_id, "status": d["status"], "validation": db.loads(d["validation_json"])}


@router.get("/datasets/{dataset_id}/validation-report.csv")
def validation_report(dataset_id: int, user: dict = Depends(auth.current_user)):
    d = get_dataset(user, dataset_id)
    rep = db.loads(d["validation_json"]) if d["id"] != ORIGINAL_ID else \
        py(validation.validate_frame(pd.read_csv(config.original_dataset()), sequence_length(), *service.training_range()))
    rows = [[c["name"], c["status"], c["detail"]] for c in rep["checks"]]
    rows += [["error", "failed", e] for e in rep["errors"]] + [["warning", "warning", w] for w in rep["warnings"]]
    rows += [["column: " + c["name"], c["dtype"], f"missing={c['missing']}"] for c in rep["columns"]]
    return csv_response(f"validation_{dataset_id}.csv", ["check", "status", "detail"], rows)


@router.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: int, request: Request, user: dict = Depends(auth.current_user)):
    if dataset_id == ORIGINAL_ID:
        raise HTTPException(403, "The original project dataset is protected and cannot be deleted.")
    d = get_dataset(user, dataset_id)
    if d.get("stored_path"):
        Path(d["stored_path"]).unlink(missing_ok=True)
    db.execute("UPDATE datasets SET status='deleted', stored_path=NULL WHERE id=?", (dataset_id,))
    db.audit("dataset_delete", "ok", user, detail=f"{d['name']} (id {dataset_id})", ip=auth.client_ip(request))
    return {"deleted": dataset_id}


# ------------------------------------------------------------------ dashboard views
@router.get("/dashboard/overview")
def overview(dataset_id: int = ORIGINAL_ID, user: dict = Depends(auth.current_user)):
    d = get_dataset(user, dataset_id)
    frame = load_frame(d)
    need_model()
    cat = service.catalog()
    best = service.best_model(cat["models"], "rmse")
    forecast = None
    if len(frame) >= sequence_length():
        forecast = service.run_prediction(frame, 1, d["id"] == ORIGINAL_ID)["forecast"][0]
    last = db.one("SELECT MAX(uploaded_at) AS t FROM datasets WHERE owner_id=? AND status IN ('ready','pending_confirmation')",
                  (user["id"],))
    panel = service.validation_panel()
    return py({"dataset": {"id": d["id"], "name": d["name"], "is_original": d["id"] == ORIGINAL_ID},
               "unit": config.UNIT_LABEL, "unit_note": config.UNIT_NOTE, "stats": service.series_stats(frame),
               "forecast_next": forecast,
               "best_model": {"metric": "rmse", "label": best["label"], "key": best["key"], "rmse": best["rmse"],
                              "mae": best["mae"], "r2": best["r2"], "basis": f"walk-forward test, {best['n_test_points']} bills"},
               "dataset_last_uploaded": last["t"] if last else None,
               "validation_status": panel["overall"], "run_id": cat["run_id"],
               "history": {"labels": frame["billing_period"].tolist(), "values": frame["total_qty"].tolist()}})


@router.get("/dashboard/analytics")
def analytics(dataset_id: int = ORIGINAL_ID, start: str | None = None, end: str | None = None,
              user: dict = Depends(auth.current_user)):
    for v in (start, end):
        if v:
            try:
                pd.to_datetime(v)
            except Exception:
                raise HTTPException(422, "start/end must be dates (YYYY-MM-DD).")
    return service.analytics(load_frame(get_dataset(user, dataset_id)), start, end)


# ------------------------------------------------------------------ models + evaluation
@router.get("/models/catalog")
def models_catalog(metric: str = Query("rmse", pattern="^(rmse|mae|mse|r2)$"), user: dict = Depends(auth.current_user)):
    cat = service.catalog()
    best = service.best_model(cat["models"], metric)
    cat["best"] = {"metric": metric, "key": best["key"] if best else None,
                   "rule": "higher is better" if metric == "r2" else "lower is better"}
    return cat


@router.get("/models/walk-forward")
def models_walk_forward(user: dict = Depends(auth.current_user)):
    return service.walk_forward_view()


@router.get("/models/horizon")
def models_horizon(user: dict = Depends(auth.current_user)):
    return {"rows": service.horizon_table()}


@router.get("/models/training")
def models_training(user: dict = Depends(auth.current_user)):
    return service.training_view()


@router.get("/models/validation")
def models_validation(user: dict = Depends(auth.current_user)):
    return service.validation_panel()


@router.get("/models/confusion")
def models_confusion(model: str = "cnn_lstm_attention", q_low: float = Query(1 / 3, gt=0, lt=1),
                     q_high: float = Query(2 / 3, gt=0, lt=1), user: dict = Depends(auth.current_user)):
    try:
        return service.confusion(model, q_low, q_high)
    except KeyError:
        raise HTTPException(404, f"Unknown model '{model}'.")
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.get("/reports/model-metrics.csv")
def report_metrics(user: dict = Depends(auth.current_user)):
    cat = service.catalog()
    rows = [[m["key"], m["label"], m["type"], m["has_saved_weights"], m["rmse"], m["mae"], m["mse"], m["r2"], m["mape"],
             m["n_test_points"], cat["run_id"]] for m in cat["models"]]
    return csv_response("model_metrics.csv", ["model", "label", "type", "has_saved_weights", "rmse", "mae", "mse", "r2",
                                              "mape_percent", "n_test_points", "run_id"], rows)


@router.get("/reports/validation.csv")
def report_validation(user: dict = Depends(auth.current_user)):
    p = service.validation_panel()
    return csv_response("model_validation.csv", ["check", "status", "detail"],
                        [[c["name"], c["status"], c["detail"]] for c in p["checks"]])


@router.get("/reports/walk-forward.csv")
def report_walk_forward(user: dict = Depends(auth.current_user)):
    wf = service.walk_forward_view()
    keys = list(wf["predictions"])
    rows = [[wf["labels"][i], wf["actual"][i]] + [wf["predictions"][k][i] for k in keys] for i in range(len(wf["labels"]))]
    return csv_response("walk_forward_predictions.csv", ["billing_period", "actual"] + keys, rows)


# ------------------------------------------------------------------ predictions
class PredictBody(BaseModel):
    dataset_id: int = ORIGINAL_ID
    horizon: int = Field(1, ge=1, le=6)


def _prediction_row(r: dict, full: bool = False) -> dict:
    out = {"id": r["id"], "user_id": r["user_id"], "dataset_id": r["dataset_id"], "dataset_name": r["dataset_name"],
           "run_id": r["run_id"], "model_variant": r["model_variant"], "horizon": r["horizon"],
           "created_at": r["created_at"]}
    if full:
        out["result"] = db.loads(r["result_json"])
    else:
        res = db.loads(r["result_json"])
        out["first_forecast"] = res["forecast"][0] if res.get("forecast") else None
    return out


@router.post("/predictions", status_code=201)
def create_prediction(body: PredictBody, request: Request, user: dict = Depends(auth.current_user)):
    d = get_dataset(user, body.dataset_id)
    if d["id"] != ORIGINAL_ID and d["status"] != "ready":
        raise HTTPException(409, "Confirm the dataset after reviewing its validation before running a prediction.")
    frame = load_frame(d)
    try:
        result = service.run_prediction(frame, body.horizon, d["id"] == ORIGINAL_ID)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    result["dataset"] = {"id": d["id"], "name": d["name"]}
    result["created_at"] = db.now()
    pid = db.execute("INSERT INTO predictions (user_id,dataset_id,dataset_name,run_id,model_variant,horizon,created_at,result_json)"
                     " VALUES (?,?,?,?,?,?,?,?)", (user["id"], d["id"], d["name"], result["model"]["run_id"],
                                                   result["model"]["variant"], body.horizon, result["created_at"],
                                                   json.dumps(result)))
    db.audit("prediction", "ok", user, detail=f"dataset {d['id']} horizon {body.horizon}", ip=auth.client_ip(request))
    return {"prediction_id": pid, **result}


def _filtered(sql_user: str | None, user_id: int | None, dataset_id: int | None, model: str | None,
              date_from: str | None, date_to: str | None, limit: int) -> list[dict]:
    where, params = [], []
    if user_id is not None:
        where.append("user_id = ?"); params.append(user_id)
    if dataset_id is not None:
        where.append("dataset_id = ?"); params.append(dataset_id)
    if model:
        where.append("model_variant = ?"); params.append(model)
    if date_from:
        where.append("created_at >= ?"); params.append(date_from)
    if date_to:
        where.append("created_at <= ?"); params.append(date_to + "T23:59:59")
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    return db.rows(f"SELECT * FROM predictions {clause} ORDER BY id DESC LIMIT ?", (*params, limit))


@router.get("/predictions")
def list_predictions(dataset_id: int | None = None, model: str | None = None, date_from: str | None = None,
                     date_to: str | None = None, limit: int = Query(100, ge=1, le=500),
                     user: dict = Depends(auth.current_user)):
    return {"predictions": [_prediction_row(r) for r in _filtered(None, user["id"], dataset_id, model, date_from, date_to, limit)]}


@router.get("/predictions/{prediction_id}")
def get_prediction(prediction_id: int, user: dict = Depends(auth.current_user)):
    r = db.one("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
    if not r or (r["user_id"] != user["id"] and user["role"] != "admin"):
        raise HTTPException(404, "Prediction not found.")
    return _prediction_row(r, full=True)


@router.get("/predictions/{prediction_id}/csv")
def prediction_csv(prediction_id: int, request: Request, user: dict = Depends(auth.current_user)):
    r = db.one("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
    if not r or (r["user_id"] != user["id"] and user["role"] != "admin"):
        raise HTTPException(404, "Prediction not found.")
    res = db.loads(r["result_json"])
    meta = [["# model", res["model"]["variant"], "run", res["model"]["run_id"], "created_at", r["created_at"],
             "unit", res["unit"] + " (" + res["unit_note"] + ")", ""]]
    rows = meta + [["forecast", f["label"], f["period_start"], f["period_end"], f["forecast"], f["lower"], f["upper"], f["method"]]
                   for f in res["forecast"]]
    rows += [["backtest_one_step", b["label"], "", "", b["predicted"], b["actual"], b["residual"],
              "in-sample" if res["backtest"]["in_sample"] else "held-out"] for b in res["backtest"]["rows"]]
    db.audit("export_prediction", "ok", user, detail=f"prediction {prediction_id}", ip=auth.client_ip(request))
    return csv_response(f"prediction_{prediction_id}.csv",
                        ["type", "period", "period_start", "period_end", "predicted_or_forecast", "lower_or_actual",
                         "upper_or_residual", "note"], rows)


# ------------------------------------------------------------------ admin
class UserPatch(BaseModel):
    role: str | None = Field(default=None, pattern="^(user|admin)$")
    active: bool | None = None


@router.get("/admin/overview")
def admin_overview(admin: dict = Depends(auth.admin_user)):
    count = lambda sql: db.one(sql)["n"]  # noqa: E731
    try:
        models = len(service.catalog()["models"]); latest = service.validation_panel()["overall"]
        cat = service.catalog(); best = service.best_model(cat["models"], "rmse")
    except ModelUnavailable as exc:
        models, latest, best = 0, f"unavailable: {exc}", None
    return py({
        "counts": {"datasets_uploaded": count("SELECT COUNT(*) n FROM datasets WHERE status IN ('ready','pending_confirmation')"),
                   "datasets_rejected": count("SELECT COUNT(*) n FROM datasets WHERE status='rejected'"),
                   "users": count("SELECT COUNT(*) n FROM users"), "predictions": count("SELECT COUNT(*) n FROM predictions"),
                   "trained_models_with_weights": 1 if service.model_info().get("loaded") else 0,
                   "models_with_evaluation": models},
        "latest_evaluation": {"validation_overall": latest, "best_model_by_rmse": best and {"label": best["label"], "rmse": best["rmse"]},
                              "last_recomputed": (db.one("SELECT ts FROM evaluations ORDER BY id DESC LIMIT 1") or {}).get("ts")},
        "recent_datasets": db.rows("SELECT id,name,status,validation_status,uploaded_at,owner_id FROM datasets ORDER BY id DESC LIMIT 5"),
        "recent_predictions": db.rows("SELECT id,user_id,dataset_name,model_variant,horizon,created_at FROM predictions ORDER BY id DESC LIMIT 5"),
        "recent_errors": db.rows("SELECT ts,username,action,outcome,detail FROM audit_log WHERE outcome != 'ok' ORDER BY id DESC LIMIT 8"),
    })


@router.get("/admin/datasets")
def admin_datasets(admin: dict = Depends(auth.admin_user)):
    rows = db.rows("SELECT d.*, u.username AS owner FROM datasets d JOIN users u ON u.id = d.owner_id ORDER BY d.id DESC")
    return {"original": original_summary(), "datasets": [{**dataset_summary(r), "owner": r["owner"]} for r in rows]}


@router.get("/admin/models")
def admin_models(admin: dict = Depends(auth.admin_user)):
    info = service.model_info()
    info["catalog"] = service.catalog()["models"] if info.get("loaded") else []
    info["evaluations"] = db.rows("SELECT id,ts,run_id FROM evaluations ORDER BY id DESC LIMIT 10")
    return info


@router.post("/admin/models/evaluate")
def admin_evaluate(request: Request, admin: dict = Depends(auth.admin_user)):
    result = service.recompute_evaluation()
    db.execute("INSERT INTO evaluations (ts,user_id,run_id,result_json) VALUES (?,?,?,?)",
               (db.now(), admin["id"], result["run_id"], json.dumps(py(result))))
    db.audit("model_evaluation", "ok", admin, detail=f"recomputed from saved predictions, run {result['run_id']}",
             ip=auth.client_ip(request))
    return result


@router.get("/admin/predictions")
def admin_predictions(user_id: int | None = None, dataset_id: int | None = None, model: str | None = None,
                      date_from: str | None = None, date_to: str | None = None, limit: int = Query(200, ge=1, le=1000),
                      admin: dict = Depends(auth.admin_user)):
    rows = _filtered(None, user_id, dataset_id, model, date_from, date_to, limit)
    names = {u["id"]: u["username"] for u in db.rows("SELECT id, username FROM users")}
    return {"predictions": [{**_prediction_row(r), "username": names.get(r["user_id"])} for r in rows]}


@router.get("/admin/predictions.csv")
def admin_predictions_csv(request: Request, user_id: int | None = None, dataset_id: int | None = None,
                          model: str | None = None, date_from: str | None = None, date_to: str | None = None,
                          admin: dict = Depends(auth.admin_user)):
    rows = _filtered(None, user_id, dataset_id, model, date_from, date_to, 1000)
    out = []
    for r in rows:
        f = (db.loads(r["result_json"])["forecast"] or [{}])[0]
        out.append([r["id"], r["user_id"], r["dataset_name"], r["model_variant"], r["horizon"], r["created_at"],
                    f.get("label"), f.get("forecast")])
    db.audit("export_predictions", "ok", admin, detail=f"{len(out)} rows", ip=auth.client_ip(request))
    return csv_response("predictions_export.csv", ["id", "user_id", "dataset", "model", "horizon", "created_at",
                                                   "first_period", "first_forecast"], out)


@router.get("/admin/users")
def admin_users(admin: dict = Depends(auth.admin_user)):
    return {"users": [auth.public_user(u) for u in db.rows("SELECT * FROM users ORDER BY id")]}


@router.patch("/admin/users/{user_id}")
def admin_patch_user(user_id: int, body: UserPatch, request: Request, admin: dict = Depends(auth.admin_user)):
    target = db.one("SELECT * FROM users WHERE id = ?", (user_id,))
    if not target:
        raise HTTPException(404, "User not found.")
    new_role = body.role or target["role"]
    new_active = target["active"] if body.active is None else int(body.active)
    if target["id"] == admin["id"] and (new_role != "admin" or not new_active):
        raise HTTPException(400, "You cannot demote or deactivate your own account.")
    if target["role"] == "admin" and (new_role != "admin" or not new_active):
        if db.one("SELECT COUNT(*) n FROM users WHERE role='admin' AND active=1")["n"] <= 1:
            raise HTTPException(400, "At least one active administrator must remain.")
    db.execute("UPDATE users SET role=?, active=? WHERE id=?", (new_role, new_active, user_id))
    db.audit("user_update", "ok", admin, detail=f"user {target['username']}: role={new_role}, active={bool(new_active)}",
             ip=auth.client_ip(request))
    return {"user": auth.public_user(db.one("SELECT * FROM users WHERE id=?", (user_id,)))}


@router.get("/admin/audit")
def admin_audit(action: str | None = None, outcome: str | None = None, limit: int = Query(200, ge=1, le=1000),
                admin: dict = Depends(auth.admin_user)):
    where, params = [], []
    if action:
        where.append("action = ?"); params.append(action)
    if outcome:
        where.append("outcome = ?"); params.append(outcome)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    return {"entries": db.rows(f"SELECT * FROM audit_log {clause} ORDER BY id DESC LIMIT ?", (*params, limit))}


@router.get("/admin/audit.csv")
def admin_audit_csv(admin: dict = Depends(auth.admin_user)):
    rows = db.rows("SELECT * FROM audit_log ORDER BY id DESC LIMIT 5000")
    return csv_response("audit_log.csv", ["id", "ts", "user_id", "username", "action", "outcome", "detail", "ip"],
                        [[r["id"], r["ts"], r["user_id"], r["username"], r["action"], r["outcome"], r["detail"], r["ip"]] for r in rows])


async def model_unavailable_handler(request: Request, exc: ModelUnavailable):
    return JSONResponse(status_code=503, content={"detail": f"Model unavailable: {exc}"})
