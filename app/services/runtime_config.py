from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.settings import settings
from app.services.runtime_settings_store import RuntimeSettingRecord, RuntimeSettingsStore

_TTL_S = 3.0
_LOCK = threading.Lock()
_CACHE_AT = 0.0
_CACHE: Dict[str, RuntimeSettingRecord] = {}

_META: Dict[str, Dict[str, Any]] = {
    "VLLM_BASE_URL": {"type": "str", "secret": False},
    "VLLM_API_KEY": {"type": "str", "secret": True},
    "VLLM_MODEL": {"type": "str", "secret": False},
    "INFERENCE_BACKEND": {"type": "str", "secret": False, "choices": {"mock", "vllm"}},
    "DEBUG_MODE": {"type": "bool", "secret": False},
    "S3_PRESIGN_TTL_S": {"type": "int", "secret": False, "min": 1, "max": 86400},
}

_STORE = RuntimeSettingsStore(settings.DATABASE_URL)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def allowed_keys() -> list[str]:
    return sorted(_META.keys())


def is_secret(key: str) -> bool:
    return bool((_META.get(key) or {}).get("secret"))


def _load_cache(force: bool = False) -> None:
    global _CACHE_AT, _CACHE
    now = time.monotonic()
    if not force and (now - _CACHE_AT) < _TTL_S:
        return
    with _LOCK:
        now = time.monotonic()
        if not force and (now - _CACHE_AT) < _TTL_S:
            return
        try:
            _CACHE = {row.key: row for row in _STORE.list()}
        except Exception:
            _CACHE = {}
        _CACHE_AT = now


def invalidate_cache() -> None:
    global _CACHE_AT
    with _LOCK:
        _CACHE_AT = 0.0


def get_record(key: str) -> Optional[RuntimeSettingRecord]:
    _load_cache()
    return _CACHE.get(key)


def _parse_bool(raw: str) -> bool:
    s = str(raw or "").strip().lower()
    return s in {"1", "true", "yes", "on"}


def _coerce_from_text(key: str, value_text: str) -> Any:
    meta = _META.get(key)
    if meta is None:
        raise KeyError(key)
    typ = str(meta.get("type") or "str")
    if typ == "bool":
        return _parse_bool(value_text)
    if typ == "int":
        return int(str(value_text).strip())
    return str(value_text)


def _validate_for_store(key: str, value: str) -> str:
    meta = _META.get(key)
    if meta is None:
        raise ValueError(f"Unsupported runtime setting: {key}")
    raw = str(value)
    typ = str(meta.get("type") or "str")
    if typ == "str":
        out = raw.strip()
    elif typ == "bool":
        out = "1" if _parse_bool(raw) else "0"
    elif typ == "int":
        iv = int(raw.strip())
        if "min" in meta:
            iv = max(int(meta["min"]), iv)
        if "max" in meta:
            iv = min(int(meta["max"]), iv)
        out = str(iv)
    else:
        raise ValueError(f"Unsupported runtime setting type: {typ}")
    choices = meta.get("choices")
    if choices and out not in set(choices):
        raise ValueError(f"Invalid value for {key}: must be one of {sorted(set(choices))}")
    if typ == "str" and not out and key != "VLLM_API_KEY":
        raise ValueError(f"Value for {key} must not be empty.")
    return out


def get_value(key: str, default: Any) -> Any:
    rec = get_record(key)
    if rec is None:
        return default
    try:
        return _coerce_from_text(key, rec.value_text)
    except Exception:
        return default


def get_str(key: str, default: str) -> str:
    return str(get_value(key, default))


def get_bool(key: str, default: bool) -> bool:
    return bool(get_value(key, default))


def get_int(key: str, default: int) -> int:
    try:
        return int(get_value(key, default))
    except Exception:
        return int(default)


def set_value(*, key: str, value: str, updated_by_key_id: Optional[str]) -> RuntimeSettingRecord:
    value_text = _validate_for_store(key, value)
    rec = _STORE.upsert(
        key=key,
        value_text=value_text,
        updated_at=_utc_now_iso(),
        updated_by_key_id=updated_by_key_id,
    )
    invalidate_cache()
    return rec


def reset_value(*, key: str) -> None:
    if key not in _META:
        raise KeyError(key)
    _STORE.delete(key)
    invalidate_cache()


def list_effective_settings() -> list[dict[str, Any]]:
    _load_cache()
    items: list[dict[str, Any]] = []
    for key in allowed_keys():
        rec = _CACHE.get(key)
        env_value = getattr(settings, key)
        effective = get_value(key, env_value)
        if is_secret(key):
            value_out = "***" if rec is not None and rec.value_text else None
            effective_out = "***" if effective else None
        else:
            value_out = rec.value_text if rec is not None else None
            effective_out = effective
        items.append(
            {
                "key": key,
                "value": value_out,
                "effective_value": effective_out,
                "source": ("db" if rec is not None else "env"),
                "updated_at": (rec.updated_at if rec is not None else None),
                "is_secret": is_secret(key),
                "is_set": bool(rec.value_text) if rec is not None else bool(env_value),
            }
        )
    return items
