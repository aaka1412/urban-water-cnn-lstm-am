import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Fresh data store (DB, uploads, signing key) per test; real outputs/ model artifacts."""
    monkeypatch.setenv("WATER_DATA_STORE", str(tmp_path / "store"))
    monkeypatch.delenv("WATER_OUTPUTS_DIR", raising=False)
    monkeypatch.delenv("WATER_RUN_ID", raising=False)
    from fastapi.testclient import TestClient
    from api.dashboard_app import app
    from dashboard import auth, db, service
    db.init_db()
    auth._FAILS.clear()
    service.clear_cache()
    with TestClient(app) as c:
        yield c
