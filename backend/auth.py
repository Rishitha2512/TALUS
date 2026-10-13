"""Offline-capable password hashing, signed access tokens, and role enforcement."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any, Callable

from fastapi import Depends, Header, HTTPException

from backend.database import get_db_connection, init_db

ROLES = {"Viewer", "Engineer", "Admin"}
TOKEN_LIFETIME_SECONDS = 8 * 60 * 60
PBKDF2_ROUNDS = 310_000


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _secret() -> bytes:
    secret = os.getenv("TALUS_AUTH_SECRET", "")
    if len(secret) < 32:
        raise RuntimeError("TALUS_AUTH_SECRET must contain at least 32 characters when auth is enabled.")
    return secret.encode("utf-8")


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    """Hash a password using PBKDF2-HMAC-SHA256 with a random salt."""
    if len(password) < 12:
        raise ValueError("Passwords must contain at least 12 characters.")
    salt_bytes = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${PBKDF2_ROUNDS}${_b64(salt_bytes)}${_b64(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, rounds, salt_b64, digest_b64 = encoded.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.urlsafe_b64decode(salt_b64 + "=" * (-len(salt_b64) % 4))
        expected = base64.urlsafe_b64decode(digest_b64 + "=" * (-len(digest_b64) % 4))
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(rounds))
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def create_access_token(username: str, role: str) -> str:
    if role not in ROLES:
        raise ValueError("Unknown TALUS role.")
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    now = int(time.time())
    payload = _b64(json.dumps(
        {"sub": username, "role": role, "iat": now, "exp": now + TOKEN_LIFETIME_SECONDS},
        separators=(",", ":"),
    ).encode())
    signed = f"{header}.{payload}".encode("ascii")
    signature = _b64(hmac.new(_secret(), signed, hashlib.sha256).digest())
    return f"{header}.{payload}.{signature}"


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        header, payload, signature = token.split(".")
        signed = f"{header}.{payload}".encode("ascii")
        expected = _b64(hmac.new(_secret(), signed, hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            raise ValueError("Invalid token signature.")
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        if claims["exp"] <= int(time.time()) or claims["role"] not in ROLES:
            raise ValueError("Expired or invalid token.")
        return {"username": str(claims["sub"]), "role": str(claims["role"])}
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid or expired access token.") from exc


def get_current_user(authorization: str | None = Header(default=None)) -> dict[str, str]:
    if os.getenv("TALUS_AUTH_ENABLED", "false").strip().lower() not in {"1", "true", "yes"}:
        return {"username": "local-demo", "role": "Admin"}
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer access token required.")
    try:
        claims = decode_access_token(authorization[7:])
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return claims


def require_roles(*roles: str) -> Callable[..., dict[str, str]]:
    if not roles or not set(roles).issubset(ROLES):
        raise ValueError("A supported role is required.")

    def check_role(user: dict[str, str] = Depends(get_current_user)) -> dict[str, str]:
        if user["role"] not in roles:
            raise HTTPException(status_code=403, detail="Insufficient role for this action.")
        return user

    return check_role


def ensure_admin_user() -> None:
    """Create the first admin account from required deployment environment values."""
    if os.getenv("TALUS_AUTH_ENABLED", "false").strip().lower() not in {"1", "true", "yes"}:
        return
    _secret()
    username = os.getenv("TALUS_ADMIN_USERNAME", "").strip()
    password = os.getenv("TALUS_ADMIN_PASSWORD", "")
    if not username or len(password) < 12:
        raise RuntimeError(
            "Set TALUS_ADMIN_USERNAME and a TALUS_ADMIN_PASSWORD of at least 12 characters."
        )
    conn = get_db_connection()
    try:
        exists = conn.execute("SELECT id FROM talus_users WHERE username=?", (username,)).fetchone()
        if exists is None:
            conn.execute(
                "INSERT INTO talus_users(username,password_hash,role) VALUES(?,?,'Admin')",
                (username, hash_password(password)),
            )
        conn.commit()
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> dict[str, str] | None:
    init_db()
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT username,password_hash,role FROM talus_users WHERE username=?",
            (username,),
        ).fetchone()
        if row is None or not verify_password(password, row["password_hash"]):
            return None
        user = {"username": row["username"], "role": row["role"]}
        conn.execute(
            "INSERT INTO audit_log(username,action,resource) VALUES(?,?,?)",
            (username, "login", "session"),
        )
        conn.commit()
        return user
    finally:
        conn.close()


def create_user(username: str, password: str, role: str, created_by: str) -> None:
    if role not in ROLES:
        raise ValueError("Role must be Viewer, Engineer, or Admin.")
    if not username.strip() or len(username) > 100:
        raise ValueError("Username must contain 1 to 100 characters.")
    init_db()
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO talus_users(username,password_hash,role) VALUES(?,?,?)",
            (username.strip(), hash_password(password), role),
        )
        conn.execute(
            "INSERT INTO audit_log(username,action,resource) VALUES(?,?,?)",
            (created_by, "create_user", username.strip()),
        )
        conn.commit()
    finally:
        conn.close()


def record_audit_event(username: str, action: str, resource: str) -> None:
    """Persist a backend user/action record used by the audit trail."""
    if not username or not action or not resource:
        raise ValueError("Audit user, action, and resource are required.")
    init_db()
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO audit_log(username,action,resource) VALUES(?,?,?)",
            (username, action, resource),
        )
        conn.commit()
    finally:
        conn.close()
