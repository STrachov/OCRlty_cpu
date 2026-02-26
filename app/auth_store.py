# app/auth_store.py
"""
AuthStore: PostgreSQL-backed API key repository.

Goals:
- Same "store class" shape as before (init-once + methods).
- Works with FastAPI via dependency factories.
- Fail-fast if migrations were not applied.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from fastapi import Depends, HTTPException, Request

from app.settings import settings

ROLE_PRESET_SCOPES: Dict[str, List[str]] = {
    "client": ["extract:run"],
    "debugger": ["extract:run", "debug:run"],
    "admin": ["extract:run", "debug:run", "debug:read_raw"],
}


try:
    import psycopg
    from psycopg.rows import dict_row
except Exception:  # pragma: no cover - optional dependency
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]


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
    PostgreSQL-backed API key store.

    Notes:
    - If settings.AUTH_ENABLED is True, pepper must be set (settings.API_KEY_PEPPER),
      otherwise RuntimeError is raised (fail-fast).
    - The store opens a new DB connection per call and guards one-time init with a lock.
    """

    def __init__(self, database_url: str = settings.DATABASE_URL) -> None:
        self._database_url = str(database_url or "").strip()
        if not (
            self._database_url.startswith("postgresql://")
            or self._database_url.startswith("postgres://")
            or self._database_url.startswith("postgresql+psycopg://")
        ):
            raise RuntimeError(
                "AuthStore requires PostgreSQL DATABASE_URL "
                "(expected postgresql://..., postgres://... or postgresql+psycopg://...)."
            )
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

    def _connect(self) -> Any:
        if psycopg is None:
            raise RuntimeError(
                "AuthStore requires psycopg. Install dependency: psycopg[binary]>=3.2,<4"
            )
        dsn = self._database_url
        if dsn.startswith("postgresql+psycopg://"):
            dsn = "postgresql://" + dsn[len("postgresql+psycopg://") :]
        return psycopg.connect(  # type: ignore[union-attr]
            dsn,
            connect_timeout=10,
            row_factory=dict_row,
        )

    def _execute(self, conn: Any, query: str, args: Sequence[Any] = ()) -> Any:
        return conn.execute(query, tuple(args))

    # ---- init ----

    def init(self) -> None:
        """Verify schema. Migrations are expected to be applied via Alembic."""
        if settings.AUTH_ENABLED:
            self._require_pepper()

        with self._connect() as conn:
            cur = self._execute(
                conn,
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = current_schema() AND table_name = 'api_keys';",
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(
                    "Table 'api_keys' is missing. Run migrations first: alembic upgrade head"
                )

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
            row = self._execute(
                conn,
                """
                SELECT api_key_id, key_id, role, scopes, is_active
                FROM api_keys
                WHERE key_hash = %s
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
                self._execute(
                    conn,
                    "UPDATE api_keys SET last_used_at = %s WHERE api_key_id = %s;",
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
                row = self._execute(
                    conn,
                    """
                    INSERT INTO api_keys (key_id, key_hash, role, scopes, is_active, created_at)
                    VALUES (%s, %s, %s, %s, 1, %s)
                    RETURNING api_key_id;
                    """,
                    (key_id, kh, role, scopes_json, created_at),
                ).fetchone()
                conn.commit()
            except Exception as e:
                # PostgreSQL unique violation SQLSTATE
                sqlstate = getattr(e, "sqlstate", None) or getattr(getattr(e, "__cause__", None), "sqlstate", None)
                if sqlstate == "23505":
                    raise ValueError(
                        "Failed to create key. Probably duplicate key_id or key hash."
                    ) from e
                raise

            if row is None:
                raise RuntimeError("Failed to insert API key row.")
            api_key_id = int(row["api_key_id"])

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
            cur = self._execute(
                conn,
                """
                UPDATE api_keys
                SET is_active = 0, revoked_at = %s
                WHERE key_id = %s AND is_active = 1;
                """,
                (_utc_now_iso(), key_id),
            )
            conn.commit()
            return (cur.rowcount or 0) > 0

    def list_api_keys(self) -> List[Dict[str, Any]]:
        """Lists all keys (for admin/CLI tooling)."""
        self.ensure_init()
        with self._connect() as conn:
            rows = self._execute(
                conn,
                """
                SELECT api_key_id, key_id, role, scopes, is_active, created_at, last_used_at, revoked_at
                FROM api_keys
                ORDER BY api_key_id ASC;
                """,
            ).fetchall()

        out: List[Dict[str, Any]] = []
        for r in list(rows or []):
            rr: Mapping[str, Any] = r
            out.append(
                {
                    "api_key_id": int(rr["api_key_id"]),
                    "key_id": str(rr["key_id"]),
                    "role": str(rr["role"]),
                    "scopes": str(rr["scopes"]),
                    "is_active": bool(int(rr["is_active"])),
                    "created_at": rr["created_at"],
                    "last_used_at": rr["last_used_at"],
                    "revoked_at": rr["revoked_at"],
                }
            )
        return out
