# app/auth_store.py
"""
AuthStore: a small repository-style wrapper around the auth SQLite DB.

Goals:
- Same "store class" shape as ArtifactIndex (init-once + methods).
- Works with FastAPI via dependency factories (no global singletons required).
- Safe to call ensure_init() multiple times (idempotent).
"""

from __future__ import annotations

import hmac
import hashlib
import json
import secrets
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from fastapi import Depends, HTTPException, Request

from app.settings import settings

ROLE_PRESET_SCOPES: Dict[str, List[str]] = {
    "client": ["extract:run"],
    "debugger": ["extract:run", "debug:run"],
    "admin": ["extract:run", "debug:run", "debug:read_raw"],
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ApiPrincipal:
    api_key_id: int
    key_id: str
    role: str
    scopes: Set[str]


class AuthStore:
    """
    SQLite-backed API key store.

    Notes:
    - If settings.AUTH_ENABLED is True, pepper must be set (settings.API_KEY_PEPPER),
      otherwise RuntimeError is raised (fail-fast).
    - The store is safe to use from multiple threads: it opens a new SQLite connection
      per call and guards one-time init with a lock.
    """

    def __init__(self, db_path: str = settings.AUTH_DB_PATH) -> None:
        self._db_path = db_path
        self._init_lock = threading.Lock()
        self._initialized = False

    # ---- internals ----

    @staticmethod
    def _require_pepper() -> str:
        if not settings.API_KEY_PEPPER:
            raise RuntimeError(
                "settings.API_KEY_PEPPER is required (set env var) when auth is enabled."
            )
        return settings.API_KEY_PEPPER

    @staticmethod
    def compute_key_hash(api_key: str, pepper: str) -> str:
        # HMAC-SHA256(secret=pepper, msg=api_key)
        return hmac.new(
            pepper.encode("utf-8"),
            api_key.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _ensure_db_dir(self) -> None:
        Path(self._db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        self._ensure_db_dir()
        conn = sqlite3.connect(self._db_path, timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ---- init ----

    def init(self) -> None:
        """Create tables/indexes if missing. Safe to call on every startup."""
        if settings.AUTH_ENABLED:
            self._require_pepper()

        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    api_key_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_id       TEXT NOT NULL UNIQUE,
                    key_hash     TEXT NOT NULL UNIQUE,
                    role         TEXT NOT NULL,
                    scopes       TEXT NOT NULL, -- JSON array or CSV
                    is_active    INTEGER NOT NULL DEFAULT 1,
                    created_at   TEXT NOT NULL,
                    last_used_at TEXT,
                    revoked_at   TEXT
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_keyid ON api_keys(key_id);")
            conn.commit()

    def ensure_init(self) -> None:
        """One-time init (lazy)."""
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            self.init()
            self._initialized = True

    # ---- request parsing / dependencies ----

    @staticmethod
    def extract_api_key_from_request(request: Request) -> Optional[str]:
        """
        Priority:
          1) Authorization: Bearer <key>
          2) X-API-Key: <key>
        """
        auth = request.headers.get("authorization")
        if auth:
            parts = auth.split(None, 1)
            if len(parts) == 2 and parts[0].lower() == "bearer":
                return parts[1].strip() or None
            raise HTTPException(
                status_code=401,
                detail="Invalid Authorization header. Use 'Bearer <api_key>'.",
            )

        xkey = request.headers.get("x-api-key")
        if xkey:
            return xkey.strip() or None

        return None

    def require_api_key_dep(self):
        """
        Returns a FastAPI dependency function that:
        - If AUTH is disabled -> returns a synthetic principal
        - If AUTH is enabled -> validates key + attaches request.state.principal
        """
        def _dep(request: Request) -> ApiPrincipal:
            if not settings.AUTH_ENABLED:
                principal = ApiPrincipal(api_key_id=0, key_id="anonymous", role="anonymous", scopes=set())
                request.state.principal = principal
                return principal

            api_key = self.extract_api_key_from_request(request)
            if not api_key:
                raise HTTPException(
                    status_code=401,
                    detail="Missing API key (Authorization: Bearer ... or X-API-Key).",
                )

            principal = self.lookup_principal(api_key)
            if principal is None:
                raise HTTPException(status_code=403, detail="Invalid or inactive API key.")

            request.state.principal = principal
            return principal

        return _dep

    def require_scopes_dep(self, required_scopes: Sequence[str]):
        """Dependency factory that enforces scopes when AUTH is enabled."""
        required = [s.strip() for s in required_scopes if s and str(s).strip()]
        require_api_key = self.require_api_key_dep()

        def _dep(principal: ApiPrincipal = Depends(require_api_key)) -> ApiPrincipal:
            if not settings.AUTH_ENABLED:
                return principal
            missing = [s for s in required if s not in principal.scopes]
            if missing:
                raise HTTPException(
                    status_code=403,
                    detail=f"Missing required scope(s): {', '.join(missing)}",
                )
            return principal

        return _dep

    # ---- core operations ----

    def lookup_principal(self, api_key: str) -> Optional[ApiPrincipal]:
        """
        Returns principal if key exists and active, otherwise None.
        Also updates last_used_at (best-effort).
        """
        pepper = self._require_pepper()
        self.ensure_init()

        kh = self.compute_key_hash(api_key, pepper)

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT api_key_id, key_id, role, scopes, is_active
                FROM api_keys
                WHERE key_hash = ?
                LIMIT 1;
                """,
                (kh,),
            ).fetchone()

            if row is None:
                return None
            if int(row["is_active"]) != 1:
                return None

            # Update last_used_at (best-effort)
            try:
                conn.execute(
                    "UPDATE api_keys SET last_used_at = ? WHERE api_key_id = ?;",
                    (_utc_now_iso(), int(row["api_key_id"])),
                )
                conn.commit()
            except Exception:
                pass

            scopes_raw = str(row["scopes"] or "").strip()
            scopes_set: Set[str] = set()
            if scopes_raw:
                try:
                    parsed = json.loads(scopes_raw)
                    if isinstance(parsed, list):
                        scopes_set = {str(s).strip() for s in parsed if str(s).strip()}
                    else:
                        scopes_set = {s.strip() for s in scopes_raw.split(",") if s.strip()}
                except Exception:
                    scopes_set = {s.strip() for s in scopes_raw.split(",") if s.strip()}

            return ApiPrincipal(
                api_key_id=int(row["api_key_id"]),
                key_id=str(row["key_id"]),
                role=str(row["role"]),
                scopes=scopes_set,
            )

    @staticmethod
    def role_default_scopes(role: str) -> List[str]:
        return ROLE_PRESET_SCOPES.get(role, [])

    def create_api_key(
        self,
        key_id: str,
        role: str,
        scopes: Optional[Sequence[str]] = None,
    ) -> Tuple[str, ApiPrincipal]:
        """
        Creates a new API key.
        Returns: (raw_api_key, principal).
        """
        pepper = self._require_pepper()
        self.ensure_init()

        role = role.strip()
        if role not in ROLE_PRESET_SCOPES:
            raise ValueError(f"Unknown role: {role}. Allowed: {sorted(ROLE_PRESET_SCOPES.keys())}")

        if scopes is None:
            scopes_list = self.role_default_scopes(role)
        else:
            scopes_list = [s.strip() for s in scopes if s and str(s).strip()]

        raw_key = secrets.token_urlsafe(32)  # ~43 chars
        kh = self.compute_key_hash(raw_key, pepper)

        with self._connect() as conn:
            created_at = _utc_now_iso()
            scopes_json = json.dumps(scopes_list, ensure_ascii=False)

            try:
                cur = conn.execute(
                    """
                    INSERT INTO api_keys (key_id, key_hash, role, scopes, is_active, created_at)
                    VALUES (?, ?, ?, ?, 1, ?);
                    """,
                    (key_id, kh, role, scopes_json, created_at),
                )
                conn.commit()
            except sqlite3.IntegrityError as e:
                raise ValueError(
                    f"Failed to create key. Probably duplicate key_id. Details: {e}"
                ) from e

            api_key_id = int(cur.lastrowid)

        principal = ApiPrincipal(
            api_key_id=api_key_id,
            key_id=key_id,
            role=role,
            scopes=set(scopes_list),
        )
        return raw_key, principal

    def revoke_api_key(self, key_id: str) -> bool:
        """Revokes key by key_id. Returns True if updated."""
        self.ensure_init()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE api_keys
                SET is_active = 0, revoked_at = ?
                WHERE key_id = ? AND is_active = 1;
                """,
                (_utc_now_iso(), key_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def list_api_keys(self) -> List[Dict[str, Any]]:
        """Lists all keys (for admin/CLI tooling)."""
        self.ensure_init()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT api_key_id, key_id, role, scopes, is_active, created_at, last_used_at, revoked_at
                FROM api_keys
                ORDER BY api_key_id ASC;
                """
            ).fetchall()

        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "api_key_id": int(r["api_key_id"]),
                    "key_id": str(r["key_id"]),
                    "role": str(r["role"]),
                    "scopes": str(r["scopes"]),
                    "is_active": bool(int(r["is_active"])),
                    "created_at": r["created_at"],
                    "last_used_at": r["last_used_at"],
                    "revoked_at": r["revoked_at"],
                }
            )
        return out
