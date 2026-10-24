"""Environment-driven configuration. Nothing here is hard-coded to a machine path."""
from __future__ import annotations

import os
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MAX_UPLOAD_BYTES = int(float(os.getenv("WATER_MAX_UPLOAD_MB", "5")) * 1024 * 1024)
MAX_UPLOAD_ROWS = int(os.getenv("WATER_MAX_UPLOAD_ROWS", "5000"))
TOKEN_HOURS = float(os.getenv("WATER_TOKEN_HOURS", "8"))
ALLOWED_EXTENSIONS = {".csv", ".xlsx"}
REQUIRED_COLUMNS = ["period_start", "period_end", "billing_period", "total_qty"]
UNIT_LABEL = "QTY"  # the project itself marks this unit as unconfirmed
UNIT_NOTE = "QTY unit not confirmed in the source data"


def data_store() -> Path:
    """Folder for the SQLite DB, uploads and the signing key (default: <project>/instance)."""
    path = Path(os.getenv("WATER_DATA_STORE", ROOT / "instance"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def db_path() -> Path:
    return data_store() / "dashboard.sqlite3"


def upload_dir() -> Path:
    path = data_store() / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def outputs_dir() -> Path:
    return Path(os.getenv("WATER_OUTPUTS_DIR", ROOT / "outputs"))


def original_dataset() -> Path:
    return Path(os.getenv("WATER_ORIGINAL_DATASET", ROOT / "data" / "kharghar_period_totals.csv"))


def run_id_override() -> str | None:
    return os.getenv("WATER_RUN_ID") or None


def jwt_secret() -> str:
    """WATER_JWT_SECRET if set; otherwise a random key generated once and kept in the data store."""
    env = os.getenv("WATER_JWT_SECRET")
    if env:
        return env
    key_file = data_store() / "secret.key"
    if not key_file.exists():
        key_file.write_text(secrets.token_hex(32))
        try:
            key_file.chmod(0o600)
        except OSError:
            pass
    return key_file.read_text().strip()
