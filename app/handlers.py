# app/handlers.py
from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import sys
import unicodedata
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.auth import ApiPrincipal, require_scopes
from app.settings import settings
from app.vllm_client import VLLMClient

from app.services.artifacts import (
    DATE_RE,
    artifact_date_from_path,
    read_artifact_json,
    read_json_file,
    save_eval_artifact,
    find_batch_artifact_path,
)
from app.services.s3_service import s3_enabled


# Load .env for local development convenience.
# In production (Docker/K8s/systemd), env vars should be provided by the runtime.
load_dotenv(override=False)

APP_ROOT = Path(__file__).resolve().parent
SCHEMAS_DIR = APP_ROOT / "schemas"
PROMPTS_DIR = APP_ROOT / "prompts"

# Data root for server-side batch runs.
# By convention in this repo: <repo_root>/data
DATA_ROOT = APP_ROOT.parent.resolve()
DEFAULT_IMAGES_DIR = str((DATA_ROOT / "data").resolve())

ARTIFACTS_DIR = Path(settings.ARTIFACTS_DIR).resolve()
MODEL_ID = settings.VLLM_MODEL


# ---------------------------
# Minimal JSON logging (request_id correlation)
# ---------------------------
REQUEST_ID_CTX: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


def _utc_ts_ms() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_log_value(v: Any, *, max_len: int = 1000) -> Any:
    if v is None or isinstance(v, (int, float, bool)):
        return v
    if isinstance(v, str):
        return v if len(v) <= max_len else (v[: max_len - 3] + "...")
    try:
        s = json.dumps(v, ensure_ascii=False)
    except Exception:
        s = repr(v)
    return s if len(s) <= max_len else (s[: max_len - 3] + "...")


class _JsonFormatter(logging.Formatter):
    _skip = {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": _utc_ts_ms(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid = REQUEST_ID_CTX.get() or ""
        if rid:
            payload["request_id"] = rid

        ev = getattr(record, "event", None)
        if ev:
            payload["event"] = ev

        for k, v in record.__dict__.items():
            if k in self._skip or k.startswith("_") or k in payload:
                continue
            payload[k] = _safe_log_value(v)

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def setup_logging() -> None:
    level_s = str(getattr(settings, "LOG_LEVEL", "INFO")).upper().strip()
    fmt = str(getattr(settings, "LOG_FORMAT", "json")).lower().strip()
    level = getattr(logging, level_s, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(logging.StreamHandler(stream=sys.stdout))

    for h in root.handlers:
        if fmt == "text":
            h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s %(message)s"))
        else:
            h.setFormatter(_JsonFormatter())

    logging.getLogger("botocore").setLevel(max(level, logging.INFO))
    logging.getLogger("boto3").setLevel(max(level, logging.INFO))


LOG = logging.getLogger("ocrlty")


def _set_request_id(rid: str):
    return REQUEST_ID_CTX.set(rid or "")


def _reset_request_id(token):
    try:
        REQUEST_ID_CTX.reset(token)
    except Exception:
        pass


def _ctx_request_id() -> str:
    return REQUEST_ID_CTX.get() or ""


def _log_event(level: int, event: str, **fields: Any) -> None:
    LOG.log(level, event, extra={"event": event, **fields})


# ---------------------------
# Tasks registry
# ---------------------------
TASK_SPECS = [
    {"task_id": "receipt_fields_v1", "prompt_file": "receipt_fields_v1.txt", "schema_file": "receipt_fields_v1.schema.json"},
]
_TASKS: Dict[str, Dict[str, Any]] = {}
_TASK_LOAD_ERRORS: List[str] = []


def load_tasks() -> None:
    global _TASKS, _TASK_LOAD_ERRORS
    _TASKS = {}
    _TASK_LOAD_ERRORS = []

    for spec in TASK_SPECS:
        task_id = spec["task_id"]
        try:
            prompt_path = (PROMPTS_DIR / spec["prompt_file"]).resolve()
            schema_path = (SCHEMAS_DIR / spec["schema_file"]).resolve()
            prompt_value = prompt_path.read_text(encoding="utf-8")
            schema_value = json.loads(schema_path.read_text(encoding="utf-8"))
            if not isinstance(schema_value, dict):
                raise ValueError("schema JSON must be an object")
            _TASKS[task_id] = {
                "task_id": task_id,
                "prompt_path": str(prompt_path),
                "schema_path": str(schema_path),
                "prompt_value": prompt_value,
                "schema_value": schema_value,
            }
        except Exception as e:
            _TASK_LOAD_ERRORS.append(f"{task_id}: {e}")


def _get_task(task_id: str) -> Dict[str, Any]:
    if not _TASKS:
        load_tasks()
    task = _TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id!r}. Available: {sorted(_TASKS.keys())}")
    if _TASK_LOAD_ERRORS:
        # Still allow requests, but expose the errors for easier debugging.
        _log_event(logging.WARNING, "task_load_errors", errors=_TASK_LOAD_ERRORS)
    return task


# ---------------------------
# Shared API models used by debug service
# ---------------------------
class ArtifactListItem(BaseModel):
    kind: str
    request_id: str
    rel_path: str
    created_at: str


class ArtifactListResponse(BaseModel):
    items: List[ArtifactListItem]


# ---------------------------
# Request IDs + path safety
# ---------------------------
_REQUEST_ID_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def make_request_id(user_request_id: Optional[str]) -> str:
    return user_request_id or uuid.uuid4().hex


def _validate_request_id_for_fs(request_id: str) -> None:
    if not _REQUEST_ID_SAFE.match(request_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid request_id (allowed: A-Za-z0-9_.-, max 128 chars).",
        )


def _validate_under_data_root(p: Path) -> Path:
    try:
        p_rel = p.resolve().relative_to(DATA_ROOT)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Path must be under data root {str(DATA_ROOT)!r}. Got: {str(p)!r}",
        )
    return DATA_ROOT / p_rel


def _validate_under_any_root(p: Path, *, roots: List[Path]) -> Path:
    p_res = p.expanduser().resolve()
    for r in roots:
        r_res = r.expanduser().resolve()
        try:
            rel = p_res.relative_to(r_res)
            return r_res / rel
        except Exception:
            continue
    raise HTTPException(
        status_code=400,
        detail=f"Path must be under one of: {[str(r) for r in roots]}. Got: {str(p)!r}",
    )


# ---------------------------
# vLLM client DI helper (used by routers)
# ---------------------------
def get_vllm_client(request: Request) -> VLLMClient:
    vc = getattr(request.app.state, "vllm_client", None)
    if vc is None:
        raise HTTPException(status_code=500, detail="vLLM client is not initialized")
    return vc


# ---------------------------
# Auth: who am I
# ---------------------------
async def me(principal: ApiPrincipal = Depends(require_scopes([]))):
    return {"key_id": principal.key_id, "role": principal.role, "scopes": sorted(list(principal.scopes))}


# ---------------------------
# Eval: batch vs GT (debug tool)
# ---------------------------
class EvalBatchVsGTRequest(BaseModel):
    run_id: str = Field(..., description="Batch run_id to load from artifacts/batches/<day>/<run_id>.json")
    batch_date: Optional[str] = Field(default=None, description="Optional day (YYYY-MM-DD) to avoid searching over days.")
    max_days: int = Field(default=30, ge=1, le=365, description="How many recent days to search for batch artifact if batch_date is not provided.")
    gt_path: str = Field(..., description="Path to GT JSON under DATA_ROOT or ARTIFACTS_DIR.")
    gt_image_key: str = Field(default="image", description="Which GT key contains image filename or path (basename is used).")
    gt_record_key: Optional[str] = Field(default=None, description="Optional: nested key in GT record containing the payload object to compare.")
    include_worst: bool = Field(default=False, description="If true, include worst_samples in eval artifact.")
    limit: Optional[int] = Field(default=None, ge=1, le=100000, description="Limit number of batch items to evaluate.")
    str_mode: str = Field(default="exact", description="String comparison mode: exact | strip | strip_collapse")


class EvalBatchVsGTResponse(BaseModel):
    eval_id: str
    created_at: str
    run_id: str
    gt_path: str
    batch_artifact_path: str
    eval_artifact_path: str


class _Missing:
    pass


_MISSING = _Missing()


def _json_type(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return "unknown"


def canon_x_string(v: Any, *, str_mode: str = "exact") -> Tuple[bool, str, Optional[str]]:
    if v is None:
        return True, "", None
    if isinstance(v, (int, float, bool)):
        s = str(v)
    else:
        s = str(v)

    if str_mode == "strip":
        s = s.strip()
    elif str_mode == "strip_collapse":
        s = " ".join(s.strip().split())

    return True, s, None


def _get_decimal_sep() -> str:
    # used in money/count canonicalization (can be configured)
    return str(getattr(settings, "DECIMAL_SEP", ".") or ".").strip()[:1] or "."


def canon_x_money(v: Any, *, decimal_sep: Optional[str] = None) -> Tuple[bool, Optional[int], Optional[str]]:
    if v is None:
        return True, None, None
    s = str(v).strip()
    if not s:
        return True, None, None

    dec = (decimal_sep or _get_decimal_sep()).strip()[:1] or "."
    s = s.replace(" ", "").replace("\u00a0", "")

    # normalize decimal separator to "."
    if dec != ".":
        s = s.replace(dec, ".")
    # common cases: "1,23" or "1.234,56"
    if "," in s and "." in s:
        # assume "," is decimal sep, "." thousands
        s = s.replace(".", "").replace(",", ".")
    elif "," in s and "." not in s:
        s = s.replace(",", ".")

    # remove currency symbols
    s = re.sub(r"[^0-9.\-]+", "", s)
    if s in {"", "-", ".", "-."}:
        return False, None, "x-money: cannot parse"

    try:
        # store as integer cents
        val = float(s)
        cents = int(round(val * 100))
        return True, cents, None
    except Exception:
        return False, None, "x-money: cannot parse"


def canon_x_count(v: Any, *, decimal_sep: Optional[str] = None) -> Tuple[bool, Optional[int], Optional[str]]:
    if v is None:
        return True, None, None
    s = str(v).strip()
    if not s:
        return True, None, None

    dec = (decimal_sep or _get_decimal_sep()).strip()[:1] or "."
    s = s.replace(" ", "").replace("\u00a0", "")

    if dec != ".":
        s = s.replace(dec, ".")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s and "." not in s:
        s = s.replace(",", ".")

    s = re.sub(r"[^0-9.\-]+", "", s)
    if s in {"", "-", ".", "-."}:
        return False, None, "x-count: cannot parse"

    try:
        val = float(s)
        # store counts as int (rounded)
        return True, int(round(val)), None
    except Exception:
        return False, None, "x-count: cannot parse"


def _deref_schema(schema: Dict[str, Any], *, root_schema: Dict[str, Any]) -> Dict[str, Any]:
    # Minimal local $ref resolver for "#/$defs/..." and "#/definitions/..."
    if not isinstance(schema, dict):
        return schema
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return schema
    parts = ref[2:].split("/")
    cur: Any = root_schema
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return schema
        cur = cur[p]
    return cur if isinstance(cur, dict) else schema


def _schema_leaf_paths(schema: Dict[str, Any]) -> List[str]:
    out: List[str] = []

    def walk(s: Dict[str, Any], prefix: Tuple[str, ...]):
        s = _deref_schema(s, root_schema=schema)
        t = s.get("type")

        if t == "object":
            props = s.get("properties") if isinstance(s.get("properties"), dict) else {}
            for k, v in props.items():
                if not isinstance(v, dict):
                    continue
                walk(v, prefix + (k,))
            return

        if t == "array":
            items = s.get("items")
            if isinstance(items, dict):
                walk(items, prefix + ("[]",))
            return

        # leaf
        out.append(".".join(prefix))

    walk(schema, ())
    return out


def _walk_schema(
    schema_node: Dict[str, Any],
    pred: Any,
    gt: Any,
    *,
    root_schema: Dict[str, Any],
    path: Tuple[str, ...],
    metrics: Dict[str, Dict[str, int]],
    mismatches: List[Dict[str, Any]],
    str_mode: str,
    mismatch_limit_total: int,
    ctx: Dict[str, int],
):
    schema_node = _deref_schema(schema_node, root_schema=root_schema)
    t = schema_node.get("type")

    def bump(p: str, key: str, inc: int = 1):
        metrics.setdefault(p, defaultdict(int))
        metrics[p][key] += inc

    p_str = ".".join(path) if path else "$"

    if t == "object":
        if not isinstance(pred, dict):
            bump(p_str, "pred_type_mismatch")
            if len(mismatches) < mismatch_limit_total:
                mismatches.append({"path": p_str, "reason": "pred_type_mismatch", "pred_type": _json_type(pred), "gt_type": "object"})
            return
        if not isinstance(gt, dict):
            bump(p_str, "gt_type_mismatch")
            return

        props = schema_node.get("properties") if isinstance(schema_node.get("properties"), dict) else {}
        for k, s_child in props.items():
            if not isinstance(s_child, dict):
                continue
            _walk_schema(
                s_child,
                pred.get(k, _MISSING),
                gt.get(k, _MISSING),
                root_schema=root_schema,
                path=path + (k,),
                metrics=metrics,
                mismatches=mismatches,
                str_mode=str_mode,
                mismatch_limit_total=mismatch_limit_total,
                ctx=ctx,
            )
        return

    if t == "array":
        if not isinstance(pred, list):
            bump(p_str, "pred_type_mismatch")
            return
        if not isinstance(gt, list):
            bump(p_str, "gt_type_mismatch")
            return
        items_schema = schema_node.get("items")
        if not isinstance(items_schema, dict):
            return
        n = min(len(pred), len(gt))
        for i in range(n):
            _walk_schema(
                items_schema,
                pred[i],
                gt[i],
                root_schema=root_schema,
                path=path + ("[]",),
                metrics=metrics,
                mismatches=mismatches,
                str_mode=str_mode,
                mismatch_limit_total=mismatch_limit_total,
                ctx=ctx,
            )
        if len(pred) != len(gt):
            bump(p_str, "len_mismatch")
        return

    # leaf comparison
    # handle missing
    if pred is _MISSING:
        bump(p_str, "pred_missing")
        return
    if gt is _MISSING:
        bump(p_str, "gt_missing")
        return

    fmt = schema_node.get("format")
    if fmt == "x-money":
        ok_p, p_c, p_err = canon_x_money(pred)
        ok_g, g_c, g_err = canon_x_money(gt)
        if not ok_p or not ok_g:
            bump(p_str, "canon_error")
            if len(mismatches) < mismatch_limit_total:
                mismatches.append({"path": p_str, "reason": "canon_error", "pred_err": p_err, "gt_err": g_err})
            return
        if p_c == g_c:
            bump(p_str, "matched")
        else:
            bump(p_str, "mismatch")
            if len(mismatches) < mismatch_limit_total:
                mismatches.append({"path": p_str, "reason": "mismatch", "pred": pred, "gt": gt, "pred_canon": p_c, "gt_canon": g_c})
        return

    if fmt == "x-count":
        ok_p, p_c, p_err = canon_x_count(pred)
        ok_g, g_c, g_err = canon_x_count(gt)
        if not ok_p or not ok_g:
            bump(p_str, "canon_error")
            if len(mismatches) < mismatch_limit_total:
                mismatches.append({"path": p_str, "reason": "canon_error", "pred_err": p_err, "gt_err": g_err})
            return
        if p_c == g_c:
            bump(p_str, "matched")
        else:
            bump(p_str, "mismatch")
            if len(mismatches) < mismatch_limit_total:
                mismatches.append({"path": p_str, "reason": "mismatch", "pred": pred, "gt": gt, "pred_canon": p_c, "gt_canon": g_c})
        return

    # default string-ish compare for everything else
    ok_p, p_s, _ = canon_x_string(pred, str_mode=str_mode)
    ok_g, g_s, _ = canon_x_string(gt, str_mode=str_mode)
    if not ok_p or not ok_g:
        bump(p_str, "canon_error")
        return
    if p_s == g_s:
        bump(p_str, "matched")
    else:
        bump(p_str, "mismatch")
        if len(mismatches) < mismatch_limit_total:
            mismatches.append({"path": p_str, "reason": "mismatch", "pred": pred, "gt": gt, "pred_canon": p_s, "gt_canon": g_s})


async def eval_batch_vs_gt(
    req: EvalBatchVsGTRequest,
) -> EvalBatchVsGTResponse:
    if req.str_mode not in {"exact", "strip", "strip_collapse"}:
        raise HTTPException(status_code=400, detail="str_mode must be one of: exact | strip | strip_collapse")

    batch_path = find_batch_artifact_path(req.run_id, date=req.batch_date, max_days=req.max_days)
    if batch_path is None:
        raise HTTPException(status_code=404, detail="Batch artifact not found (check run_id and batch_date/max_days).")

    # IMPORTANT: batch_path may be a local Path or an S3 key ref.
    batch_obj = read_artifact_json(batch_path)
    if not isinstance(batch_obj, dict):
        raise HTTPException(status_code=500, detail="Invalid batch artifact JSON (expected object).")

    batch_date = artifact_date_from_path(batch_path) or (req.batch_date or "")
    task_id = str(batch_obj.get("task_id") or "")
    if not task_id:
        raise HTTPException(status_code=500, detail="Batch artifact missing task_id.")
    task = _get_task(task_id)

    schema_json = task.get("schema_value")
    if not isinstance(schema_json, dict):
        raise HTTPException(status_code=500, detail=f"Task {task_id!r} has no schema_value (dict).")

    # Load GT
    gt_p = _validate_under_any_root(Path(req.gt_path), roots=[DATA_ROOT, ARTIFACTS_DIR])
    gt_obj = read_json_file(gt_p)

    if isinstance(gt_obj, dict):
        gt_records = list(gt_obj.values())
    elif isinstance(gt_obj, list):
        gt_records = gt_obj
    else:
        raise HTTPException(status_code=400, detail="GT JSON must be an object or an array of objects.")

    gt_index: Dict[str, Dict[str, Any]] = {}
    for rec in gt_records:
        if not isinstance(rec, dict):
            continue
        img_val = rec.get(req.gt_image_key)
        if img_val is None and req.gt_image_key != "image_file":
            img_val = rec.get("image_file")
        if img_val is None:
            continue
        base = Path(str(img_val)).name
        if base:
            gt_index[base] = rec

    items = batch_obj.get("items") if isinstance(batch_obj.get("items"), list) else []
    if req.limit is not None:
        items = items[: int(req.limit)]

    metrics: Dict[str, Dict[str, int]] = {}
    sample_summaries: List[Dict[str, Any]] = []

    gt_found = 0
    gt_missing = 0
    pred_found = 0
    pred_missing = 0

    mismatch_limit_total = 5000  # cap per sample

    for it in items:
        if not isinstance(it, dict):
            continue

        image = it.get("file") or it.get("image") or it.get("image_file")
        image = str(image) if image is not None else ""
        image_base = Path(image).name if image else ""

        request_id = str(it.get("request_id") or "")
        artifact_rel = it.get("artifact_rel")

        gt_rec = gt_index.get(image_base) if image_base else None
        if gt_rec is not None:
            gt_found += 1
        else:
            gt_missing += 1

        # Resolve pred artifact (supports local + S3)
        pred_obj: Any = _MISSING
        artifact_ref: Optional[Union[str, Path]] = None

        if isinstance(artifact_rel, str) and artifact_rel:
            rel = Path(artifact_rel)
            if rel.is_absolute():
                artifact_ref = _validate_under_any_root(rel, roots=[ARTIFACTS_DIR, DATA_ROOT])
            else:
                # Old batch format might omit "extracts/" prefix.
                if len(rel.parts) >= 1 and DATE_RE.match(rel.parts[0]) and rel.parts[0].count("-") == 2:
                    rel = Path("extracts") / rel
                if s3_enabled():
                    artifact_ref = str(rel).replace("\\", "/")
                else:
                    artifact_ref = (ARTIFACTS_DIR / rel).resolve()
        elif request_id:
            # Fallback for older batch artifacts without artifact_rel: try resolving by request_id.
            # This works for both local and S3 (via the artifacts index / S3 existence check).
            try:
                from app.services.artifacts import find_artifact_path
                artifact_ref = find_artifact_path(request_id)
            except Exception:
                artifact_ref = None

        if artifact_ref is not None:
            try:
                art = read_artifact_json(artifact_ref)
                pred_obj = (
                    art.get("parsed")
                    or art.get("result")
                    or art.get("data")
                    or art.get("output")
                    or art.get("extracted")
                    or _MISSING
                )
                if pred_obj is not _MISSING:
                    pred_found += 1
                else:
                    pred_missing += 1
            except Exception:
                pred_missing += 1
        else:
            pred_missing += 1

        gt_payload: Any = gt_rec if gt_rec is not None else _MISSING
        if gt_payload is not _MISSING and req.gt_record_key:
            if isinstance(gt_payload, dict) and req.gt_record_key in gt_payload:
                gt_payload = gt_payload.get(req.gt_record_key)
            else:
                gt_payload = _MISSING

        mismatches: List[Dict[str, Any]] = []
        if gt_payload is not _MISSING and pred_obj is not _MISSING:
            _walk_schema(
                schema_json,
                pred_obj,
                gt_payload,
                root_schema=schema_json,
                path=(),
                metrics=metrics,
                mismatches=mismatches,
                str_mode=req.str_mode,
                mismatch_limit_total=mismatch_limit_total,
                ctx={"n": 0},
            )

        sample_summaries.append(
            {
                "image_base": image_base,
                "request_id": request_id,
                "gt_ok": gt_rec is not None,
                "pred_ok": pred_obj is not _MISSING,
                "mismatches_count": len(mismatches),
                "mismatches": mismatches[:50],  # keep inline small; full mismatches in eval artifact
            }
        )

    # Build rows
    field_rows: List[Dict[str, Any]] = []
    for path, m in metrics.items():
        matched = int(m.get("matched", 0))
        mismatch = int(m.get("mismatch", 0))
        pred_miss = int(m.get("pred_missing", 0))
        gt_miss = int(m.get("gt_missing", 0))
        canon_err = int(m.get("canon_error", 0))
        total = matched + mismatch + pred_miss + gt_miss + canon_err

        denom = max(1, total - canon_err)
        accuracy = matched / denom

        field_rows.append(
            {
                "path": path,
                "matched": matched,
                "mismatch": mismatch,
                "pred_missing": pred_miss,
                "gt_missing": gt_miss,
                "canon_error": canon_err,
                "total": total,
                "accuracy": accuracy,
            }
        )
    field_rows.sort(key=lambda r: (r["accuracy"], r["total"]), reverse=False)

    # Prepare eval artifact payload
    eval_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    payload = {
        "eval_id": eval_id,
        "created_at": created_at,
        "run_id": req.run_id,
        "batch_date": batch_date,
        "task_id": task_id,
        "gt_path": str(gt_p),
        "batch_artifact_path": str(batch_path),
        "summary": {
            "items": len(items),
            "gt_found": gt_found,
            "gt_missing": gt_missing,
            "pred_found": pred_found,
            "pred_missing": pred_missing,
            "str_mode": req.str_mode,
            "decimal_sep": _get_decimal_sep(),
        },
        "fields": field_rows,
        "samples": sample_summaries,
    }

    owner = (batch_obj.get("auth") or {}).get("key_id")
    eval_artifact_path = await save_eval_artifact(payload, owner)

    return EvalBatchVsGTResponse(
        eval_id=eval_id,
        created_at=created_at,
        run_id=req.run_id,
        gt_path=str(gt_p),
        batch_artifact_path=str(batch_path),
        eval_artifact_path=eval_artifact_path,
    )
