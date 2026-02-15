"""
app/services/artifacts.py

Artifact layer:
- Stores extract/batch/eval artifacts either locally or in S3/R2.
- Preserves existing helper function names used in the codebase.
- Adds a best-effort SQLite index for fast lookup by request_id / run_id.

S3 primitives are implemented in app/services/s3_service.py and imported here.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from fastapi import HTTPException

from app.settings import settings
from app.services.s3_service import (
    s3_enabled,
    s3_key,
    s3_put_json,
    s3_get_json,
    s3_exists,
    s3_list_common_prefixes,
)
# Used only for converting full S3 key -> relative artifact_rel (strip prefix).
_S3_PREFIX = (settings.S3_PREFIX or "ocrlty").strip().strip("/")

# Optional SQLite index (safe fallback if module isn't present)
try:
    from app.artifact_index import ArtifactIndex
except Exception:  # pragma: no cover
    ArtifactIndex = None  # type: ignore


ARTIFACTS_DIR = Path(settings.ARTIFACTS_DIR).resolve()
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ---------------------------
# Local filesystem helpers
# ---------------------------

def _ensure_artifacts_dir() -> None:
    try:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        # best-effort
        return


def _write_json_file(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json_file(path: Union[str, Path]) -> Any:
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read JSON: {e}")


# ---------------------------
# Artifact key/paths
# ---------------------------

def _artifact_key(kind: str, day: str, name: str) -> str:
    # kind: extracts|batches|evals
    return s3_key(f"{kind}/{day}/{name}")


def _extract_artifact_ref(request_id: str, *, day: Optional[str] = None) -> Union[str, Path]:
    d = day or _utc_day()
    if s3_enabled():
        return _artifact_key("extracts", d, f"{request_id}.json")
    return (ARTIFACTS_DIR / "extracts" / d / f"{request_id}.json").resolve()


def _batch_artifact_ref(run_id: str, *, day: Optional[str] = None) -> Union[str, Path]:
    d = day or _utc_day()
    if s3_enabled():
        return _artifact_key("batches", d, f"{run_id}.json")
    return (ARTIFACTS_DIR / "batches" / d / f"{run_id}.json").resolve()


def _eval_artifact_ref(eval_id: str, *, day: Optional[str] = None) -> Union[str, Path]:
    d = day or _utc_day()
    if s3_enabled():
        return _artifact_key("evals", d, f"{eval_id}.json")
    return (ARTIFACTS_DIR / "evals" / d / f"{eval_id}.json").resolve()


def _job_result_artifact_ref(job_id: str, *, day: Optional[str] = None) -> Union[str, Path]:
    d = day or _utc_day()
    if s3_enabled():
        return _artifact_key("jobs", d, f"{job_id}.json")
    return (ARTIFACTS_DIR / "jobs" / d / f"{job_id}.json").resolve()


def _job_error_artifact_ref(job_id: str, *, day: Optional[str] = None) -> Union[str, Path]:
    d = day or _utc_day()
    if s3_enabled():
        return _artifact_key("job_errors", d, f"{job_id}.json")
    return (ARTIFACTS_DIR / "job_errors" / d / f"{job_id}.json").resolve()


def to_artifact_rel(ref: Union[str, Path]) -> str:
    """
    Convert full reference into artifact_rel that can be stored in responses/artifacts.
    - S3: strip prefix
    - Local: path relative to ARTIFACTS_DIR
    """
    if s3_enabled():
        key = str(ref)
        if _S3_PREFIX and key.startswith(_S3_PREFIX + "/"):
            return key[len(_S3_PREFIX) + 1 :]
        return key
    try:
        p = Path(ref).resolve()
        return str(p.relative_to(ARTIFACTS_DIR))
    except Exception:
        return str(ref)


def artifact_date_from_path(ref: Union[str, Path]) -> Optional[str]:
    s = str(ref)
    m = re.search(r"/(extracts|batches|evals)/(\d{4}-\d{2}-\d{2})/", s)
    return m.group(2) if m else None


# ---------------------------
# SQLite index (best-effort)
# ---------------------------

def _index_enabled() -> bool:
    v = os.getenv("ARTIFACT_INDEX_ENABLED", "1").strip().lower()
    return v not in {"0", "false", "no", "off"}


def _index_db_path() -> Path:
    p = getattr(settings, "ARTIFACT_INDEX_DB_PATH", None) or os.getenv("ARTIFACT_INDEX_DB_PATH") or "/data/db/artifacts.db"
    return Path(p).expanduser().resolve()


_index = None
if ArtifactIndex is not None and _index_enabled():
    try:
        _index = ArtifactIndex(
            enabled=True,
            db_path=_index_db_path(),
            log_event=lambda level, event, **fields: logging.getLogger("ocrlty").log(level, "%s %s", event, fields),
            is_s3_enabled=s3_enabled,
        )
    except Exception:
        _index = None


def artifact_index_init() -> None:
    if _index is not None:
        _index.init()


def _index_upsert(kind: str, artifact_id: str, ref: Union[str, Path], owner_key_id: Optional[str] = None) -> None:
    if _index is None:
        return
    try:
        _index.upsert(
            kind=kind,
            artifact_id=artifact_id,
            full_ref=str(ref),
            rel_ref=to_artifact_rel(ref),
            day=artifact_date_from_path(ref) or _utc_day(),
            owner_key_id=owner_key_id,
        )
    except Exception:
        pass


def _index_get(kind: str, artifact_id: str) -> Optional[str]:
    if _index is None:
        return None
    try:
        rec = _index.get(kind=kind, artifact_id=artifact_id)
        if not rec:
            return None
        full = rec.get("full_ref")
        return str(full) if full else None
    except Exception:
        return None


def _index_delete(kind: str, artifact_id: str) -> None:
    if _index is None:
        return
    try:
        _index.delete(kind=kind, artifact_id=artifact_id)
    except Exception:
        pass


# ---------------------------
# Save/load artifacts
# ---------------------------

def save_artifact(request_id: str, payload: Dict[str, Any], owner_key_id : Optional[str] = None) -> Union[str, Path]:
    _ensure_artifacts_dir()
    ref = _extract_artifact_ref(request_id)
    if s3_enabled():
        s3_put_json(str(ref), payload)
    else:
        _write_json_file(Path(ref), payload)
    _index_upsert("extract", request_id, ref, owner_key_id)
    return ref


def save_batch_artifact(run_id: str, payload: Dict[str, Any], owner_key_id : Optional[str] = None) -> Union[str, Path]:
    _ensure_artifacts_dir()
    ref = _batch_artifact_ref(run_id)
    if s3_enabled():
        s3_put_json(str(ref), payload)
    else:
        _write_json_file(Path(ref), payload)
    _index_upsert("batch", run_id, ref, owner_key_id)
    return ref


def save_eval_artifact(payload: Dict[str, Any], owner_key_id : Optional[str] = None) -> Union[str, Path]:
    _ensure_artifacts_dir()
    eval_id = str(payload.get("eval_id") or "")
    if not eval_id:
        raise HTTPException(status_code=500, detail="eval_id is missing.")
    ref = _eval_artifact_ref(eval_id)
    if s3_enabled():
        s3_put_json(str(ref), payload)
    else:
        _write_json_file(Path(ref), payload)
    _index_upsert("eval", eval_id, ref, owner_key_id)
    return ref


def save_job_result_artifact(job_id: str, payload: Dict[str, Any], owner_key_id : Optional[str] = None) -> Union[str, Path]:
    """Store full (potentially large) job result JSON under artifacts and return ref."""
    _ensure_artifacts_dir()
    ref = _job_result_artifact_ref(job_id)
    if s3_enabled():
        s3_put_json(str(ref), payload)
    else:
        _write_json_file(Path(ref), payload)
    _index_upsert("job_result", job_id, ref, owner_key_id)
    return ref


def save_job_error_artifact(job_id: str, payload: Dict[str, Any], owner_key_id : Optional[str] = None) -> Union[str, Path]:
    """Store job error details (including traceback) under artifacts and return ref."""
    _ensure_artifacts_dir()
    ref = _job_error_artifact_ref(job_id)
    if s3_enabled():
        s3_put_json(str(ref), payload)
    else:
        _write_json_file(Path(ref), payload)
    _index_upsert("job_error", job_id, ref, owner_key_id)
    return ref



def read_artifact_json(ref: Union[str, Path]) -> Dict[str, Any]:
    if s3_enabled() and isinstance(ref, str) and not ref.startswith("/"):
        return s3_get_json(ref)
    obj = read_json_file(ref)
    if not isinstance(obj, dict):
        raise HTTPException(status_code=500, detail="Invalid artifact JSON (expected object).")
    return obj


# ---------------------------
# Listing / search helpers
# ---------------------------

def iter_artifact_days_desc(kind: str = "extracts") -> List[str]:
    # kind is directory name: extracts|batches|evals
    if s3_enabled():
        base = s3_key(f"{kind}/")
        days: List[str] = []
        for pfx in s3_list_common_prefixes(base):
            m = re.search(rf"/{re.escape(kind)}/(\d{{4}}-\d{{2}}-\d{{2}})/$", pfx)
            if m:
                days.append(m.group(1))
        days.sort(reverse=True)
        return days

    base = ARTIFACTS_DIR / kind
    if not base.exists():
        return []
    days = [p.name for p in base.iterdir() if p.is_dir() and DATE_RE.match(p.name)]
    days.sort(reverse=True)
    return days


def find_artifact_path(request_id: str, *, date: Optional[str] = None, max_days: int = 30) -> Optional[Union[str, Path]]:
    # 1) direct date
    if date is not None:
        if not DATE_RE.match(date):
            raise HTTPException(status_code=400, detail="date must be in YYYY-MM-DD format.")
        ref = _extract_artifact_ref(request_id, day=date)
        if s3_enabled():
            return ref if s3_exists(str(ref)) else None
        return ref if Path(ref).exists() else None

    # 2) index
    full = _index_get("extract", request_id)
    if full:
        if s3_enabled() and not str(full).startswith("/"):
            if s3_exists(full):
                return full
        else:
            if Path(full).exists():
                return Path(full)
        _index_delete("extract", request_id)

    # 3) scan days
    for i, day in enumerate(iter_artifact_days_desc("extracts")):
        if i >= int(max_days):
            break
        ref = _extract_artifact_ref(request_id, day=day)
        if s3_enabled():
            if s3_exists(str(ref)):
                _index_upsert("extract", request_id, ref)
                return ref
        else:
            if Path(ref).exists():
                _index_upsert("extract", request_id, ref)
                return Path(ref)
    return None


def find_batch_artifact_path(run_id: str, *, date: Optional[str] = None, max_days: int = 90) -> Optional[Union[str, Path]]:
    if date is not None:
        if not DATE_RE.match(date):
            raise HTTPException(status_code=400, detail="date must be in YYYY-MM-DD format.")
        ref = _batch_artifact_ref(run_id, day=date)
        if s3_enabled():
            return ref if s3_exists(str(ref)) else None
        return ref if Path(ref).exists() else None

    full = _index_get("batch", run_id)
    if full:
        if s3_enabled() and not str(full).startswith("/"):
            if s3_exists(full):
                return full
        else:
            if Path(full).exists():
                return Path(full)
        _index_delete("batch", run_id)

    for i, day in enumerate(iter_artifact_days_desc("batches")):
        if i >= int(max_days):
            break
        ref = _batch_artifact_ref(run_id, day=day)
        if s3_enabled():
            if s3_exists(str(ref)):
                _index_upsert("batch", run_id, ref)
                return ref
        else:
            if Path(ref).exists():
                _index_upsert("batch", run_id, ref)
                return Path(ref)
    return None
