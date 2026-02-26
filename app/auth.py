# app/auth.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from fastapi import Request

from app.settings import settings
from app.auth_store import AuthStore, ApiPrincipal, ROLE_PRESET_SCOPES


# --- Store registry ---

_STORES: Dict[str, AuthStore] = {}


def _get_store(database_url: str) -> AuthStore:
    store = _STORES.get(database_url)
    if store is None:
        store = AuthStore(database_url=database_url)
        _STORES[database_url] = store
    return store


_DEFAULT_STORE = _get_store(settings.DATABASE_URL)
_DEFAULT_REQUIRE_API_KEY = _DEFAULT_STORE.require_api_key_dep()


# --- Public API (backwards-compatible) ---


def init_auth_db(database_url: str = settings.DATABASE_URL) -> None:
    """Verifies auth schema is present (migrations must be applied beforehand)."""
    _get_store(database_url).ensure_init()


def extract_api_key_from_request(request: Request) -> Optional[str]:
    return AuthStore.extract_api_key_from_request(request)


def compute_key_hash(api_key: str, pepper: str) -> str:
    return AuthStore.compute_key_hash(api_key, pepper)


def lookup_principal(api_key: str, database_url: str = settings.DATABASE_URL) -> Optional[ApiPrincipal]:
    return _get_store(database_url).lookup_principal(api_key)


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
    database_url: str = settings.DATABASE_URL,
) -> Tuple[str, ApiPrincipal]:
    """Creates a new API key. Returns (raw_api_key, principal)."""
    return _get_store(database_url).create_api_key(key_id, role, scopes)


def revoke_api_key(
    key_id: str,
    database_url: str = settings.DATABASE_URL,
) -> bool:
    """Revokes key by key_id. Returns True if updated."""
    return _get_store(database_url).revoke_api_key(key_id)

def list_api_keys(database_url: str = settings.DATABASE_URL) -> List[Dict[str, Any]]:
    """Lists all keys (for admin/CLI tooling)."""
    return _get_store(database_url).list_api_keys()
#create_api_key / revoke_api_key / list_api_keys
