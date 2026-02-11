# app/auth.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from fastapi import Request

from app.settings import settings
from app.auth_store import AuthStore, ApiPrincipal, ROLE_PRESET_SCOPES


# --- Store registry (keeps compatibility with db_path=... params) ---

_STORES: Dict[str, AuthStore] = {}


def _get_store(db_path: str) -> AuthStore:
    store = _STORES.get(db_path)
    if store is None:
        store = AuthStore(db_path=db_path)
        _STORES[db_path] = store
    return store


_DEFAULT_STORE = _get_store(settings.AUTH_DB_PATH)
_DEFAULT_REQUIRE_API_KEY = _DEFAULT_STORE.require_api_key_dep()


# --- Public API (backwards-compatible) ---


def init_auth_db(db_path: str = settings.AUTH_DB_PATH) -> None:
    """Creates tables if not exist. Safe to call on every startup."""
    _get_store(db_path).ensure_init()


def extract_api_key_from_request(request: Request) -> Optional[str]:
    return AuthStore.extract_api_key_from_request(request)


def compute_key_hash(api_key: str, pepper: str) -> str:
    return AuthStore.compute_key_hash(api_key, pepper)


def lookup_principal(api_key: str, db_path: str = settings.AUTH_DB_PATH) -> Optional[ApiPrincipal]:
    return _get_store(db_path).lookup_principal(api_key)


def require_api_key(request: Request) -> ApiPrincipal:
    """
    FastAPI dependency:
      - If settings.AUTH_ENABLED=0 -> returns a synthetic principal (for local/dev), without checks.
      - If settings.AUTH_ENABLED=1 -> requires valid active key.
    """
    return _DEFAULT_REQUIRE_API_KEY(request)


def require_scopes(required_scopes: Sequence[str]):
    """Dependency factory that enforces scopes when AUTH is enabled."""
    return _DEFAULT_STORE.require_scopes_dep(required_scopes)


# --- CLI utilities (used by app/auth_cli.py) ---


def role_default_scopes(role: str) -> List[str]:
    return AuthStore.role_default_scopes(role)


def create_api_key(
    key_id: str,
    role: str,
    scopes: Optional[Sequence[str]] = None,
    db_path: str = settings.AUTH_DB_PATH,
) -> Tuple[str, ApiPrincipal]:
    """Creates a new API key. Returns (raw_api_key, principal)."""
    return _get_store(db_path).create_
