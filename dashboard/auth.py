"""Password hashing (bcrypt), signed tokens (JWT) and server-side role checks."""
from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import config, db

_bearer = HTTPBearer(auto_error=False)
MIN_PASSWORD = 10
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
_FAILS: dict[str, list[float]] = {}
MAX_FAILS, WINDOW_S = 5, 15 * 60


def validate_password(password: str) -> None:
    if len(password) < MIN_PASSWORD:
        raise HTTPException(422, f"Password must be at least {MIN_PASSWORD} characters.")
    if len(password.encode()) > 72:
        raise HTTPException(422, "Password must be at most 72 bytes (bcrypt limit).")
    if not (re.search(r"[A-Za-z]", password) and re.search(r"\d", password)):
        raise HTTPException(422, "Password must contain at least one letter and one digit.")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode()[:72], hashed.encode())
    except ValueError:
        return False


def make_token(user: dict) -> str:
    exp = datetime.now(timezone.utc) + timedelta(hours=config.TOKEN_HOURS)
    return jwt.encode({"sub": str(user["id"]), "role": user["role"], "exp": exp}, config.jwt_secret(), "HS256")


def client_ip(request: Request) -> str | None:
    return request.client.host if request.client else None


def check_rate_limit(key: str) -> None:
    now = time.time()
    recent = [t for t in _FAILS.get(key, []) if now - t < WINDOW_S]
    _FAILS[key] = recent
    if len(recent) >= MAX_FAILS:
        raise HTTPException(429, "Too many failed sign-in attempts. Try again in 15 minutes.")


def record_failure(key: str) -> None:
    _FAILS.setdefault(key, []).append(time.time())


def clear_failures(key: str) -> None:
    _FAILS.pop(key, None)


def public_user(user: dict) -> dict:
    return {k: user[k] for k in ("id", "username", "email", "role", "active", "created_at", "last_login")}


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> dict:
    """Valid token AND an existing, active account (so deactivation takes effect immediately)."""
    if creds is None:
        raise HTTPException(401, "Sign in required.", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(creds.credentials, config.jwt_secret(), algorithms=["HS256"])
        user_id = int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise HTTPException(401, "Session expired or invalid. Please sign in again.",
                            headers={"WWW-Authenticate": "Bearer"})
    user = db.one("SELECT * FROM users WHERE id = ?", (user_id,))
    if not user or not user["active"]:
        raise HTTPException(401, "Account not found or deactivated.")
    return user  # role comes from the DB, not from the token


def admin_user(user: dict = Depends(current_user)) -> dict:
    if user["role"] != "admin":
        db.audit("admin_access_denied", "denied", user, detail="non-admin requested admin endpoint")
        raise HTTPException(403, "Administrator access required.")
    return user
