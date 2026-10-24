"""Create (or reset) an administrator account from the command line.

  python -m dashboard.create_admin --username admin
The password is read from the WATER_ADMIN_PASSWORD environment variable or prompted for; it is never
stored in plain text and never accepted on the command line.
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys

from fastapi import HTTPException

from . import auth, db


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or reset an administrator account.")
    parser.add_argument("--username", required=True)
    parser.add_argument("--email", default=None)
    parser.add_argument("--reset-password", action="store_true", help="Reset the password of an existing admin.")
    args = parser.parse_args()
    if not auth.USERNAME_RE.match(args.username):
        print("Username must be 3-32 characters: letters, digits, _ . -", file=sys.stderr)
        return 2
    password = os.getenv("WATER_ADMIN_PASSWORD") or getpass.getpass("Admin password: ")
    try:
        auth.validate_password(password)
    except HTTPException as exc:
        print(exc.detail, file=sys.stderr)
        return 2
    db.init_db()
    existing = db.one("SELECT * FROM users WHERE username = ?", (args.username,))
    if existing and not args.reset_password:
        print("That username already exists. Use --reset-password to change an admin's password.", file=sys.stderr)
        return 1
    hashed = auth.hash_password(password)
    if existing:
        db.execute("UPDATE users SET password_hash=?, role='admin', active=1 WHERE id=?", (hashed, existing["id"]))
        db.audit("admin_password_reset", detail=f"CLI reset for {args.username}", username="cli")
        print(f"Password reset for admin '{args.username}'.")
    else:
        db.execute("INSERT INTO users (username,email,password_hash,role,active,created_at) VALUES (?,?,?,?,1,?)",
                   (args.username, args.email, hashed, "admin", db.now()))
        db.audit("admin_created", detail=f"CLI created admin {args.username}", username="cli")
        print(f"Admin '{args.username}' created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
