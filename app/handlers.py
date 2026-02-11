# app/handlers.py
import asyncio
import base64
import math
from io import BytesIO

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None
import hashlib
import json
import mimetypes
import os
import logging
import sys
import contextvars
import re
import unicodedata
import time
import uuid
import tarfile
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, List, Callable, Awaitable
from collections import defaultdict
from dataclasses import dataclass
import boto3
import botocore
from botocore.config import Config as BotoConfig

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request,  File, UploadFile, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from app.auth import ApiPrincipal, init_auth_db, require_scopes
from app.settings import settings
# Artifact helpers moved to app.services.artifacts (legacy names preserved)
S3_BUCKET = settings.S3_BUCKET or ""
from app.services.s3_service import (
    s3_enabled,
    s3_key,
    s3_get_bytes,
    s3_put_bytes,
)
# Artifact helpers moved to app.services.artifacts (legacy names preserved)
ARTIFACTS_DIR=Path(settings.ARTIFACTS_DIR).resolve()
from app.services.artifacts import (
    DATE_RE,
    to_artifact_rel,
    artifact_date_from_path,
    read_artifact_json,
    read_json_file,
    save_artifact,
    save_batch_artifact,
    save_eval_artifact,
    find_artifact_path,
    find_batch_artifact_path,
)

from dotenv import load_dotenv  
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)

from app.vllm_client import VLLMClient
vllm = VLLMClient()

APP_ROOT = Path(__file__).resolve().parent
SCHEMAS_DIR = APP_ROOT / "schemas"
PROMPTS_DIR = APP_ROOT / "prompts"

# Data root for server-side batch runs.
# By convention in this repo: /workspace/src/data
DATA_ROOT = APP_ROOT.parent.resolve()
DEFAULT_IMAGES_DIR = str((DATA_ROOT / "data").resolve())

ARTIFACTS_DIR = Path(settings.ARTIFACTS_DIR).resolve()
MODEL_ID = settings.VLLM_MODEL




# ---------------------------
# Minimal JSON logging (request_id correlation)
# ---------------------------
# - JSON by default (one line per event), good for Docker + log collectors
# - request_id is taken from X-Request-ID header (if provided) otherwise generated per request
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
        "args","asctime","created","exc_info","exc_text","filename","funcName","levelname","levelno","lineno",
        "module","msecs","message","msg","name","pathname","process","processName","relativeCreated",
        "stack_info","thread","threadName",
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
# ---------------------------
# Context-length safety / smart image retry
# ---------------------------
# We don't have exact multimodal tokenization locally. Instead, on "decoder prompt too long" errors,
# we parse the server-provided token counts and compute a one-shot downscale factor for the image.
_CTX_SAFETY_TOKENS = 256          # keep some slack for chat template / special tokens
_SCALE_SAFETY = 0.90              # extra shrink for safety
_MIN_SCALE = 0.20                 # do not shrink below this in one retry
_MAX_SCALE = 0.95                 # avoid pointless re-encode
_QWEN_ALIGN = 32                  # Qwen-VL family typically aligns vision sizes to multiples of 32
_TEXT_CHARS_PER_TOKEN = 3.0       # conservative heuristic for "JSON-y" text


# ---------------------------
# Tasks (p.4)
# ---------------------------

TASK_SPECS: Dict[str, Dict[str, str]] = {
    # Existing default task
    "receipt_fields_v1": {
        "schema_file": "receipt_fields_v1.schema.json",
        "schema_name": "default_schema_receipt_fields_v1",
        "prompt_file": "receipt_fields_v1.txt",
        "prompt_name": "default_prompt_receipt_fields_v1",
    },
    # New tasks (add files under app/schemas and app/prompts)
    "payment_v1": {
        "schema_file": "payment_v1.schema.json",
        "schema_name": "payment_v1",
        "prompt_file": "payment_v1.txt",
        "prompt_name": "payment_v1",
    },
    "items_light_v1": {
        "schema_file": "items_light_v1.schema.json",
        "schema_name": "items_light_v1",
        "prompt_file": "items_light_v1.txt",
        "prompt_name": "items_light_v1",
    },
}

# task_id -> runtime payload
_TASKS: Dict[str, Dict[str, Any]] = {}
_TASK_LOAD_ERRORS: Dict[str, str] = {}


def _load_task_or_error(task_id: str) -> None:
    spec = TASK_SPECS[task_id]
    schema_path = SCHEMAS_DIR / spec["schema_file"]
    prompt_path = PROMPTS_DIR / spec["prompt_file"]
    try:
        schema_value = json.loads(schema_path.read_text(encoding="utf-8"))
        prompt_value = prompt_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        _TASK_LOAD_ERRORS[task_id] = f"{e.__class__.__name__}: {e}"
        return

    _TASKS[task_id] = {
        "task_id": task_id,
        "schema_value": schema_value,
        "schema_name": spec["schema_name"],
        "prompt_value": prompt_value,
        "prompt_name": spec["prompt_name"],
    }


def load_tasks() -> None:
    _TASKS.clear()
    _TASK_LOAD_ERRORS.clear()

    for task_id in TASK_SPECS.keys():
        _load_task_or_error(task_id)

    # Default task must exist; otherwise the service isn't usable.
    if "receipt_fields_v1" in _TASK_LOAD_ERRORS:
        raise RuntimeError(
            "Failed to load default task receipt_fields_v1: "
            f"{_TASK_LOAD_ERRORS['receipt_fields_v1']}"
        )


def _get_task(task_id: str) -> Dict[str, Any]:
    task_id = (task_id or "").strip()
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is empty.")
    if task_id not in TASK_SPECS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id!r}. Allowed: {sorted(TASK_SPECS.keys())}")
    if task_id in _TASK_LOAD_ERRORS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Task files are missing or invalid for task_id={task_id!r}: "
                f"{_TASK_LOAD_ERRORS[task_id]}"
            ),
        )
    return _TASKS[task_id]


# ---------------------------
# Models
# ---------------------------

class DebugOptions(BaseModel):
    # Any debug usage requires DEBUG_MODE=1 (global env switch).
    prompt_override: Optional[str] = Field(default=None, description="Override the base prompt (DEBUG only).")
    schema_override: Optional[str] = Field(
        default=None,
        description="Override schema (JSON string), or 'none'/'null' to disable schema enforcement (DEBUG only).",
    )
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Override temperature (DEBUG only).")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Override max_tokens (DEBUG only).")

    # Mock backend controls (DEBUG only).
    mock_mode: Optional[str] = Field(
        default=None,
        description="For INFERENCE_BACKEND=mock: 'ok' | 'fail_json' | 'fail_schema'. "
                    "If omitted, may be inferred from request_id prefix.",
    )
    mock_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="For INFERENCE_BACKEND=mock: provide a JSON object to return as the model output (DEBUG only).",
    )


class ExtractRequest(BaseModel):
    task_id: str = Field(default="receipt_fields_v1", description="Which extraction task to run.")
    # Provide either image_url OR image_base64.
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    image_ref: Optional[str] = Field(default=None, description="Optional reference to the source image (e.g., relative path). Stored in artifacts only.")
    mime_type: str = Field(default="image/png", description="Used only with image_base64.")
    request_id: Optional[str] = None

    # Debug block. Presence of this block requires DEBUG_MODE=1 AND debug:run scope.
    debug: Optional[DebugOptions] = None


class ExtractResponse(BaseModel):
    request_id: str
    task_id: str
    model: str

    schema_ok: bool
    result: Dict[str, Any]
    timings_ms: Dict[str, float]
    schema_errors: Optional[list[str]] = None

    # Populated when we had to downscale the image due to context-length constraints.
    image_resize: Optional[Dict[str, Any]] = None

    # System-level errors/retries recorded during processing (empty for a clean successful run).
    error_history: List[Dict[str, Any]] = Field(default_factory=list)

    # Returned only when DEBUG_MODE=1 and caller has debug:read_raw (or AUTH is disabled).
    raw_model_text: Optional[str] = None

    # Relative artifact pointer (S3 key relative to prefix, or path relative to ARTIFACTS_DIR)
    artifact_rel: Optional[str] = None


class ArtifactListItem(BaseModel):
    request_id: str
    date: str
    task_id: Optional[str] = None
    schema_ok: Optional[bool] = None


class ArtifactListResponse(BaseModel):
    items: List[ArtifactListItem]




# ---------------------------
# Helpers
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


def _list_image_files(images_dir: Path, glob_pattern: str, exts: List[str], limit: Optional[int]) -> List[Path]:
    if not images_dir.exists() or not images_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"images_dir does not exist or is not a directory: {str(images_dir)!r}")

    allowed = {("." + e.lower().lstrip(".")) for e in (exts or [])}
    files: List[Path] = []
    for p in images_dir.glob(glob_pattern or "**/*"):
        if not p.is_file():
            continue
        if allowed and p.suffix.lower() not in allowed:
            continue
        files.append(p)

    files.sort(key=lambda x: str(x))
    if limit is not None:
        files = files[: int(limit)]
    return files


def _default_run_id() -> str:
    # Must match request_id regex and remain short.
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]


def _make_batch_request_id(run_id: str, rel_path: str) -> str:
    # Keep it stable and short: <run_id>__<base>_<hash8>
    base = os.path.splitext(os.path.basename(rel_path))[0]
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base)[:40] or "img"
    h = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:8]
    req_id = f"{run_id}__{base}_{h}"
    if len(req_id) > 128:
        req_id = req_id[:128]
    _validate_request_id_for_fs(req_id)
    return req_id


def _guess_mime_type(path: Path) -> str:
    mt, _ = mimetypes.guess_type(str(path))
    return mt or "image/png"


def _image_content(image_url: Optional[str], image_base64: Optional[str], mime_type: str) -> Dict[str, Any]:
    if image_url and image_base64:
        raise HTTPException(status_code=400, detail="Provide either image_url or image_base64 (not both).")
    if image_url:
        return {"type": "image_url", "image_url": {"url": image_url}}
    if image_base64:
        data_url = f"data:{mime_type};base64,{image_base64}"
        return {"type": "image_url", "image_url": {"url": data_url}}
    raise HTTPException(status_code=400, detail="Provide either image_url or image_base64.")



def _sanitize_input_for_artifact(req: "ExtractRequest", image_resize: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # Avoid storing large blobs (base64) in artifacts.
    return {
        "task_id": req.task_id,
        "user_request_id": req.request_id,
        "image_ref": req.image_ref,
        "image_url": req.image_url,
        "has_image_base64": bool(req.image_base64),
        "image_base64_chars": len(req.image_base64) if req.image_base64 else 0,
        "mime_type": req.mime_type,
        "image_resize": image_resize,
        "has_debug": bool(req.debug),
    }

def _debug_enabled() -> None:
    if not settings.DEBUG_MODE:
        raise HTTPException(status_code=403, detail="Debug features are disabled (DEBUG_MODE=0).")


def _debug_allowed(principal: ApiPrincipal) -> None:
    _debug_enabled()
    if settings.AUTH_ENABLED and ("debug:run" not in principal.scopes):
        raise HTTPException(status_code=403, detail="Missing required scope: debug:run")


def _raw_allowed(principal: ApiPrincipal) -> bool:
    if not settings.DEBUG_MODE:
        return False
    if not settings.AUTH_ENABLED:
        return True
    return "debug:read_raw" in principal.scopes


def _clamp_max_tokens(v: int) -> int:
    return max(1, min(int(v), int(settings.MAX_TOKENS_CAP)))


def _apply_debug_overrides(
    principal: ApiPrincipal,
    debug: Optional[DebugOptions],
    *,
    base_prompt_text: str,
    base_schema_json: dict[str, Any],
) -> Tuple[str, float, int, Union[dict[str, Any], str, None]]:
    prompt_text = base_prompt_text
    schema_json: Union[dict[str, Any], str, None] = base_schema_json
    temperature = 0.0
    max_tokens = 128

    if debug is None:
        return prompt_text, temperature, max_tokens, schema_json

    _debug_allowed(principal)

    if debug.prompt_override is not None:
        p = debug.prompt_override.strip()
        if len(p) > int(settings.MAX_PROMPT_CHARS):
            raise HTTPException(status_code=400, detail=f"prompt_override too long (>{settings.MAX_PROMPT_CHARS} chars).")
        if p:
            prompt_text = p

    if debug.schema_override is not None:
        s = debug.schema_override.strip()
        if s:
            schema_json = s

    if debug.temperature is not None:
        temperature = float(debug.temperature)

    if debug.max_tokens is not None:
        max_tokens = int(debug.max_tokens)

    max_tokens = _clamp_max_tokens(max_tokens)
    return prompt_text, temperature, max_tokens, schema_json


def _resolve_ref(ref: str, root_schema: Dict[str, Any]) -> Dict[str, Any]:
    if not ref.startswith("#/"):
        return {}
    parts = ref[2:].split("/")
    cur: Any = root_schema
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return {}
    return cur if isinstance(cur, dict) else {}


def _sample_from_schema(schema: Dict[str, Any], root_schema: Dict[str, Any]) -> Any:
    if "$ref" in schema and isinstance(schema["$ref"], str):
        resolved = _resolve_ref(schema["$ref"], root_schema)
        if resolved:
            return _sample_from_schema(resolved, root_schema)

    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema and isinstance(schema[key], list) and schema[key]:
            return _sample_from_schema(schema[key][0], root_schema)

    t = schema.get("type")
    if isinstance(t, list) and t:
        t = t[0]

    if t == "object" or ("properties" in schema):
        props = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
        required = schema.get("required", []) if isinstance(schema.get("required"), list) else []
        out: Dict[str, Any] = {}
        for k in required:
            if k in props and isinstance(props[k], dict):
                out[k] = _sample_from_schema(props[k], root_schema)
            else:
                out[k] = None
        added = 0
        for k, v in props.items():
            if k in out:
                continue
            if isinstance(v, dict):
                out[k] = _sample_from_schema(v, root_schema)
                added += 1
            if added >= 2:
                break
        return out

    if t == "array":
        items = schema.get("items", {})
        min_items = int(schema.get("minItems", 0) or 0)
        if min_items > 0 and isinstance(items, dict):
            return [_sample_from_schema(items, root_schema)]
        return []

    if t == "string":
        if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
            return str(schema["enum"][0])
        fmt = schema.get("format")
        if fmt == "date":
            return "2026-01-24"
        if fmt == "date-time":
            return "2026-01-24T12:00:00Z"
        return "mock"

    if t == "integer":
        return 0
    if t == "number":
        return 0.0
    if t == "boolean":
        return False

    return None


def _mock_chat_completions(
    request_id: str,
    principal: ApiPrincipal,
    debug: Optional[DebugOptions],
    schema_dict: Optional[dict[str, Any]],
) -> Dict[str, Any]:
    mode = "ok"
    if request_id.startswith("fail_json"):
        mode = "fail_json"
    elif request_id.startswith("fail_schema"):
        mode = "fail_schema"

    if debug is not None:
        _debug_allowed(principal)
        if debug.mock_mode:
            mode = debug.mock_mode.strip().lower()
        if debug.mock_result is not None:
            content = json.dumps(debug.mock_result, ensure_ascii=False)
            return {"choices": [{"message": {"content": content}}]}

    if mode == "fail_json":
        content = "{"
    elif mode == "fail_schema":
        content = "{}"
    else:
        if schema_dict is None:
            content = json.dumps({"mock": True, "schema_used": False}, ensure_ascii=False)
        else:
            sample = _sample_from_schema(schema_dict, schema_dict)
            content = json.dumps(sample, ensure_ascii=False)

    return {"choices": [{"message": {"content": content}}]}


def _build_messages(prompt_text: str, req: ExtractRequest) -> Any:
    return [
        {
            "role": "user",
            "content": [
                _image_content(req.image_url, req.image_base64, req.mime_type),
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


# ---------------------------
# Smart retry for context-length errors (decoder prompt too long)
# ---------------------------
# Fragment of the error: The decoder prompt (length 9254) is longer than the maximum model length of 4096. Make sure that `max_model_len` is no smaller ...
_DECODER_TOO_LONG_RE = re.compile(
    r"(?:decoder\s+)?prompt\s*\(length\s+(\d+)\)\s+is\s+longer\s+than\s+the\s+maximum\s+model\s+length\s+of\s+(\d+)",
    re.IGNORECASE,
)


def _parse_decoder_prompt_too_long(err: str) -> Optional[Tuple[int, int]]:
    """Return (t_resp, t_max) if error looks like a context-length error."""
    if not err:
        return None
    m = _DECODER_TOO_LONG_RE.search(err)
    if not m:
        return None
    try:
        return int(m.group(1)), int(m.group(2))
    except Exception:
        return None


def _estimate_prompt_tokens(prompt_text: str, schema_dict: Optional[dict[str, Any]]) -> int:
    """Heuristic token count for the *text* part of the request (prompt + schema).

    We cannot reproduce exact multimodal tokenization locally. This estimate is used only to compute
    an image downscale factor after the server returns an exact total length (t_resp).
    """
    schema_s = ""
    if schema_dict is not None:
        try:
            schema_s = json.dumps(schema_dict, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            schema_s = str(schema_dict)

    chars = len(prompt_text) + len(schema_s)
    # Add slack for chat template, role wrappers, and structured output glue.
    chars += 1000
    return int(math.ceil(chars / _TEXT_CHARS_PER_TOKEN))


def _align_down(x: int, m: int) -> int:
    if x <= 0:
        return m
    return max(m, (x // m) * m)


def _resize_base64_image_by_scale(image_base64: str, mime_type: str, scale: float) -> Tuple[str, Dict[str, Any]]:
    if Image is None:
        raise RuntimeError("PIL (Pillow) is required for image resizing but is not available.")
    data = base64.b64decode(image_base64)
    im = Image.open(BytesIO(data))
    im.load()
    w0, h0 = im.size

    # Compute new size (align to 32 for Qwen-VL style models).
    new_w = _align_down(int(w0 * scale), _QWEN_ALIGN)
    new_h = _align_down(int(h0 * scale), _QWEN_ALIGN)

    # Avoid degenerate sizes.
    new_w = max(_QWEN_ALIGN, new_w)
    new_h = max(_QWEN_ALIGN, new_h)

    # If scaling doesn't change anything, return original.
    if new_w >= w0 and new_h >= h0:
        return image_base64, {
            "applied": False,
            "reason": "scale_noop",
            "scale": float(scale),
            "orig_w": int(w0),
            "orig_h": int(h0),
            "new_w": int(w0),
            "new_h": int(h0),
        }

    im = im.convert("RGB").resize((new_w, new_h), Image.Resampling.LANCZOS)

    out = BytesIO()
    mt = (mime_type or "").lower().strip()

    # Preserve PNG when explicitly requested; otherwise JPEG is fine for receipts.
    if mt == "image/png":
        im.save(out, format="PNG", optimize=True)
        out_mime = "image/png"
    else:
        im.save(out, format="JPEG", quality=92, optimize=True, progressive=True)
        out_mime = "image/jpeg"

    b64_new = base64.b64encode(out.getvalue()).decode("ascii")
    return b64_new, {
        "applied": True,
        "reason": "context_len_retry",
        "scale": float(scale),
        "orig_w": int(w0),
        "orig_h": int(h0),
        "new_w": int(new_w),
        "new_h": int(new_h),
        "orig_b64_chars": int(len(image_base64)),
        "new_b64_chars": int(len(b64_new)),
        "sent_mime_type": out_mime,
    }


def _compute_retry_scale(t_resp: int, t_max: int, t_prompt_est: int) -> Optional[float]:
    """Compute linear image downscale factor from server-provided lengths.

    Let:
      t_resp = total decoder prompt length from the server error (text + image tokens)
      t_prompt_est = estimated text-only tokens (prompt + schema + wrappers)
      t_max = server max context length

    We want: t_prompt_est + t_img_new <= t_max - safety
    and assume image tokens scale ~ area ~ (scale^2).
    """
    budget = int(t_max) - int(_CTX_SAFETY_TOKENS)
    if t_prompt_est >= budget:
        return None

    t_img_now = max(1, int(t_resp) - int(t_prompt_est))
    t_img_budget = max(1, int(budget) - int(t_prompt_est))

    # token reduction ratio needed
    r_tokens = t_img_budget / float(t_img_now)
    # linear scale ~ sqrt(token_ratio)
    scale = math.sqrt(r_tokens) * float(_SCALE_SAFETY)
    scale = max(float(_MIN_SCALE), min(float(_MAX_SCALE), float(scale)))
    return scale


def _now_iso_ms() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace('+00:00', 'Z')


def _mk_err(stage: str, message: str, **extra: Any) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "ts": _now_iso_ms(),
        "stage": stage,
        "message": message,
    }
    if extra:
        d["extra"] = extra
    return d


def _try_load_error_history_for_request(request_id: str) -> Optional[list[Dict[str, Any]]]:
    art_p = find_artifact_path(request_id, max_days=30)
    if art_p is None:
        return None
    try:
        obj = json.loads(Path(art_p).read_text(encoding="utf-8"))
        eh = obj.get("error_history")
        return eh if isinstance(eh, list) else None
    except Exception:
        return None


def _normalize_schema(schema: Union[dict[str, Any], str, None], default_schema: dict[str, Any]) -> Optional[dict[str, Any]]:
    if schema is None:
        return default_schema

    if isinstance(schema, dict):
        return schema

    if isinstance(schema, str):
        s = schema.strip().lower()
        if s in {"none", "null"}:
            return None

        try:
            parsed = json.loads(schema)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail=f"schema is a string but not valid JSON: {schema!r}")

        if not isinstance(parsed, dict):
            raise HTTPException(status_code=400, detail=f"schema JSON must decode to dict, got {type(parsed).__name__}")

        return parsed

    raise HTTPException(
        status_code=400,
        detail=f"schema must be dict | json-str | 'none' | None, got {type(schema).__name__}",
    )


def _build_response_format(schema_dict: dict[str, Any], schema_name: str) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": schema_dict,
            "strict": True,
        },
    }


def _make_validator_or_raise(schema_dict: dict[str, Any], *, is_custom_schema: bool) -> Draft202012Validator:
    try:
        Draft202012Validator.check_schema(schema_dict)
    except SchemaError as e:
        msg = getattr(e, "message", str(e))
        if is_custom_schema:
            raise HTTPException(status_code=400, detail=f"Invalid JSON schema in schema_override: {msg}")
        raise HTTPException(status_code=500, detail=f"Server misconfiguration: invalid default schema: {msg}")

    try:
        return Draft202012Validator(schema_dict)
    except Exception as e:
        if is_custom_schema:
            raise HTTPException(status_code=400, detail=f"Invalid JSON schema in schema_override: {e}")
        raise HTTPException(status_code=500, detail=f"Server misconfiguration: invalid default schema: {e}")


# ---------------------------
# Artifacts 
# ---------------------------

def _validate_under_any_root(p: Path, roots: List[Path]) -> Path:
    pp = p.expanduser().resolve()
    for r in roots:
        try:
            rel = pp.relative_to(r.resolve())
            return r.resolve() / rel
        except Exception:
            continue
    raise HTTPException(
        status_code=400,
        detail=f"Path must be under one of allowed roots: {[str(r) for r in roots]}. Got: {str(p)!r}",
    )


class _Missing:
    pass


_MISSING = _Missing()


def _json_type(v: Any) -> str:
    if v is _MISSING:
        return "missing"
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int) and not isinstance(v, bool):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    # Fallback for odd types (e.g. Decimal) - treat as string repr
    return "unknown"


def _schema_types(schema: Dict[str, Any]) -> List[str]:
    t = schema.get("type")
    if isinstance(t, str):
        return [t]
    if isinstance(t, list):
        return [x for x in t if isinstance(x, str)]
    # Infer from keywords
    if isinstance(schema.get("properties"), dict):
        return ["object"]
    if "items" in schema:
        return ["array"]
    return []


def _schema_allows_type(schema: Dict[str, Any], vtype: str, root_schema: Dict[str, Any]) -> bool:
    schema = _deref_schema(schema, root_schema)
    for key in ("anyOf", "oneOf"):
        if key in schema and isinstance(schema[key], list) and schema[key]:
            return any(_schema_allows_type(s, vtype, root_schema) for s in schema[key] if isinstance(s, dict))
    if "allOf" in schema and isinstance(schema["allOf"], list) and schema["allOf"]:
        # allOf: value must satisfy all; use a weak check (any) to avoid false negatives
        return any(_schema_allows_type(s, vtype, root_schema) for s in schema["allOf"] if isinstance(s, dict))

    types = _schema_types(schema)
    if not types:
        return True  # no type constraint
    if vtype == "integer" and "number" in types:
        return True
    return vtype in types


def _pick_variant(schema: Dict[str, Any], value: Any, root_schema: Dict[str, Any]) -> Dict[str, Any]:
    schema = _deref_schema(schema, root_schema)

    for key in ("anyOf", "oneOf"):
        variants = schema.get(key)
        if isinstance(variants, list) and variants:
            vtype = _json_type(value)
            # Try match by GT type first; fall back to first
            for s in variants:
                if isinstance(s, dict) and _schema_allows_type(s, vtype, root_schema):
                    return _deref_schema(s, root_schema)
            # If gt is missing/null but pred has value, try pred type
            if value in (_MISSING, None):
                return _deref_schema(variants[0], root_schema) if isinstance(variants[0], dict) else schema
            return _deref_schema(variants[0], root_schema) if isinstance(variants[0], dict) else schema

    # allOf: choose first (we don't attempt true merging here)
    if "allOf" in schema and isinstance(schema["allOf"], list) and schema["allOf"]:
        first = schema["allOf"][0]
        return _deref_schema(first, root_schema) if isinstance(first, dict) else schema

    return schema


def _deref_schema(schema: Dict[str, Any], root_schema: Dict[str, Any]) -> Dict[str, Any]:
    if "$ref" in schema and isinstance(schema["$ref"], str):
        resolved = _resolve_ref(schema["$ref"], root_schema)
        if resolved:
            return _deref_schema(resolved, root_schema)
    return schema


def _safe_scalar(v: Any, max_len: int = 200) -> Any:
    # Keep response JSON small and stable.
    if v is _MISSING:
        return "<MISSING>"
    if isinstance(v, (str, int, float, bool)) or v is None:
        if isinstance(v, str) and len(v) > max_len:
            return v[: max_len - 3] + "..."
        return v
    # objects/arrays: truncated JSON
    try:
        s = json.dumps(v, ensure_ascii=False)
    except Exception:
        s = repr(v)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _get_prop(obj: Any, key: str, *, key_fallbacks: bool = False) -> Any:
    if not isinstance(obj, dict):
        return _MISSING
    if key in obj:
        return obj[key]
    if key_fallbacks and key.endswith("_raw"):
        k2 = key[:-4]
        if k2 in obj:
            return obj[k2]
    return _MISSING


def _get_decimal_sep() -> str:
    """Decimal separator preference for numeric parsing.

    OCRLTY_DECIMAL_SEP:
      - "." (default)    : dot is decimal separator
      - ","              : comma is decimal separator
      - "auto"           : infer from the string

    """
    v = (os.getenv("OCRLTY_DECIMAL_SEP", ".") or ".").strip().lower()
    if v in {".", ","}:
        return v
    return "auto"


def canon_x_string(v: Any, *, mode: str = "strip_collapse") -> Tuple[bool, Optional[str], Optional[str]]:
    """x-string canonicalization.

    - None -> ok, None
    - other -> stringified + NFKC normalized
    - mode: exact | strip | strip_collapse
    - empty string -> None (treat as null-like noise)
    """
    if v is None:
        return True, None, None

    try:
        s = v if isinstance(v, str) else str(v)
    except Exception:
        return False, None, "to_string_failed"

    s = unicodedata.normalize("NFKC", s)

    if mode == "exact":
        out = s
    else:
        out = s.strip()
        if mode == "strip_collapse":
            out = re.sub(r"\s+", " ", out)

    if out == "":
        return True, None, None

    return True, out, None


def canon_x_count(v: Any) -> Tuple[bool, Optional[int], Optional[str]]:
    """x-count canonicalization (quantity / count).

    Accepts:
      - int
      - float (only if int-like)
      - string (extracts first integer anywhere: "1 X" -> 1, "x2" -> 2)
      - None -> ok, None
    """
    if v is None:
        return True, None, None

    if isinstance(v, bool):
        return False, None, "bool_not_allowed"

    if isinstance(v, int) and not isinstance(v, bool):
        return True, int(v), None

    if isinstance(v, float):
        if abs(v - round(v)) <= 1e-9:
            return True, int(round(v)), None
        return False, None, "float_not_intlike"

    if not isinstance(v, str):
        try:
            v = str(v)
        except Exception:
            return False, None, "to_string_failed"

    s = v.strip()
    if s == "":
        return True, None, None

    m = re.search(r"[+-]?\d+", s)
    if not m:
        return False, None, "no_int_found"

    try:
        return True, int(m.group(0)), None
    except Exception:
        return False, None, "int_parse_failed"


def canon_x_money(v: Any) -> Tuple[bool, Optional[float], Optional[str]]:
    """x-money canonicalization.

    Goal: turn things like "261,333", "Rp 73.450", "1 234,56" into a number.

    - None -> ok, None
    - int/float -> float
    - string -> parse with OCRLTY_DECIMAL_SEP (auto/. /,)
    """
    if v is None:
        return True, None, None

    if isinstance(v, bool):
        return False, None, "bool_not_allowed"

    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return True, float(v), None

    if not isinstance(v, str):
        try:
            v = str(v)
        except Exception:
            return False, None, "to_string_failed"

    s = v.strip()
    if s == "":
        return True, None, None

    # Keep digits, separators, sign; drop currency/letters/spaces.
    s = re.sub(r"[^0-9,\.\-+]", "", s)
    if not re.search(r"\d", s):
        return False, None, "no_digits"

    dec_pref = _get_decimal_sep()

    def _collapse_dots(x: str) -> str:
        # Collapse multiple dots to a single decimal dot (keep last as decimal).
        if x.count(".") <= 1:
            return x
        parts = x.split(".")
        return "".join(parts[:-1]) + "." + parts[-1]

    def parse_with_decimal_sep(raw: str, dec: str) -> Optional[float]:
        other = "," if dec == "." else "."
        x = raw.replace(other, "")
        if dec == ",":
            x = x.replace(",", ".")
        x = _collapse_dots(x)
        try:
            return float(x)
        except Exception:
            return None

    def parse_auto(raw: str) -> Optional[float]:
        x = raw
        has_comma = "," in x
        has_dot = "." in x

        if has_comma and has_dot:
            # right-most separator is decimal, the other is thousands
            dec = "." if x.rfind(".") > x.rfind(",") else ","
            return parse_with_decimal_sep(x, dec)

        if has_comma:
            left, right = x.rsplit(",", 1)
            # if 1-2 digits after comma -> likely decimals
            if re.fullmatch(r"\d{1,2}", right or ""):
                return parse_with_decimal_sep(x, ",")
            # if exactly 3 digits -> likely thousands
            if re.fullmatch(r"\d{3}", right or ""):
                try:
                    return float((left + right).replace(",", ""))
                except Exception:
                    return None
            # fallback: treat commas as thousands
            try:
                return float(x.replace(",", ""))
            except Exception:
                return None

        if has_dot:
            # multiple dots: likely thousands separators
            if x.count(".") > 1:
                try:
                    return float(x.replace(".", ""))
                except Exception:
                    return None
            left, right = x.rsplit(".", 1)
            # if exactly 3 digits -> thousands
            if re.fullmatch(r"\d{3}", right or ""):
                try:
                    return float(left + right)
                except Exception:
                    return None
            return parse_with_decimal_sep(x, ".")

        try:
            return float(x)
        except Exception:
            return None

    out: Optional[float]
    if dec_pref in {".", ","}:
        out = parse_with_decimal_sep(s, dec_pref)
    else:
        out = parse_auto(s)

    if out is None:
        return False, None, "parse_fail"
    return True, out, None


def _canon_value(v: Any, *, fmt: Optional[str], str_mode: str) -> Tuple[bool, Any, Optional[str]]:
    """Canonicalize a scalar value given a schema format hint."""
    if v is _MISSING:
        return False, None, "missing"

    f = (fmt or "").strip().lower()
    if f in {"x-string", "x-str", "string"}:
        return canon_x_string(v, mode=str_mode)
    if f in {"x-count", "x-int", "x-integer", "count"}:
        return canon_x_count(v)
    if f in {"x-money", "x-number", "money", "number"}:
        return canon_x_money(v)

    # No explicit format -> conservative fallback.
    if isinstance(v, str) or v is None:
        return canon_x_string(v, mode=str_mode)

    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return True, float(v), None

    # last resort: stringify
    return canon_x_string(v, mode=str_mode)


def _compare_canon(
    pred_raw: Any,
    gt_raw: Any,
    *,
    fmt: Optional[str],
    str_mode: str,
) -> Tuple[bool, str, Any, Any]:
    """Compare leaf scalar values with canonicalization.

    Returns:
      (matched, reason, pred_value_canon, gt_value_canon)
    """
    if gt_raw is _MISSING:
        return False, "gt_missing", None, None
    if pred_raw is _MISSING:
        return False, "pred_missing", None, None

    ok_gt, gt_v, gt_err = _canon_value(gt_raw, fmt=fmt, str_mode=str_mode)
    ok_pred, pred_v, pred_err = _canon_value(pred_raw, fmt=fmt, str_mode=str_mode)

    if not ok_gt and not ok_pred:
        return False, "canon_errors", pred_v, gt_v
    if not ok_gt:
        return False, "gt_canon_error", pred_v, gt_v
    if not ok_pred:
        return False, "pred_canon_error", pred_v, gt_v

    if gt_v is None and pred_v is None:
        return True, "ok", pred_v, gt_v
    if (gt_v is None) != (pred_v is None):
        return False, "value_mismatch", pred_v, gt_v

    f = (fmt or "").strip().lower()
    if f in {"x-money", "x-number", "money", "number"} and isinstance(gt_v, (int, float)) and isinstance(pred_v, (int, float)):
        if math.isclose(float(gt_v), float(pred_v), rel_tol=0.0, abs_tol=1e-6):
            return True, "ok", pred_v, gt_v
        return False, "value_mismatch", pred_v, gt_v

    if gt_v == pred_v:
        return True, "ok", pred_v, gt_v
    return False, "value_mismatch", pred_v, gt_v


def _walk_schema(
    schema: Dict[str, Any],
    pred: Any,
    gt: Any,
    *,
    root_schema: Dict[str, Any],
    path: Tuple[Union[str, int, None], ...],
    metrics: Dict[str, Dict[str, int]],
    mismatches: List[Dict[str, Any]],
    str_mode: str,
    key_fallbacks: bool,
    mismatch_limit: int,
) -> None:
    schema_eff = _pick_variant(schema, gt, root_schema)
    schema_eff = _deref_schema(schema_eff, root_schema)

    # Determine node type
    types = _schema_types(schema_eff)
    node_type = types[0] if types else _json_type(gt if gt is not _MISSING else pred)
    if node_type == "integer" and isinstance(gt, float):
        node_type = "number"

    if node_type == "object":
        props = schema_eff.get("properties", {})
        if not isinstance(props, dict):
            props = {}
        pred_obj = pred if isinstance(pred, dict) else {}
        gt_obj = gt if isinstance(gt, dict) else {}

        for k, subschema in props.items():
            if not isinstance(subschema, dict):
                continue
            pred_v = _get_prop(pred_obj, k, key_fallbacks=key_fallbacks)
            gt_v = _get_prop(gt_obj, k, key_fallbacks=key_fallbacks)
            _walk_schema(
                subschema,
                pred_v,
                gt_v,
                root_schema=root_schema,
                path=path + (k,),
                metrics=metrics,
                mismatches=mismatches,
                str_mode=str_mode,
                key_fallbacks=key_fallbacks,
                mismatch_limit=mismatch_limit,
            )
        return

    if node_type == "array":
        items_schema = schema_eff.get("items", {})
        if not isinstance(items_schema, dict):
            items_schema = {}
        pred_list = pred if isinstance(pred, list) else []
        gt_list = gt if isinstance(gt, list) else []
        n = max(len(pred_list), len(gt_list))
        for i in range(n):
            pred_v = pred_list[i] if i < len(pred_list) else _MISSING
            gt_v = gt_list[i] if i < len(gt_list) else _MISSING
            _walk_schema(
                items_schema,
                pred_v,
                gt_v,
                root_schema=root_schema,
                path=path + (None,),  # wildcard index
                metrics=metrics,
                mismatches=mismatches,
                str_mode=str_mode,
                key_fallbacks=key_fallbacks,
                mismatch_limit=mismatch_limit,
            )
        return

    # Scalar leaf
    def _fmt_path(pth: Tuple[Union[str, int, None], ...]) -> str:
        out = ""
        for seg in pth:
            if seg is None:
                out += "[*]"
            elif isinstance(seg, int):
                out += f"[{seg}]"
            else:
                out += ("." if out else "") + str(seg)
        return out or "<root>"

    path_str = _fmt_path(path)
    m = metrics.setdefault(path_str, defaultdict(int))
    m["total"] += 1

    if gt is _MISSING:
        m["gt_missing"] += 1
        return

    m["comparable"] += 1

    if pred is _MISSING:
        m["pred_missing"] += 1
        m["mismatched"] += 1
        if len(mismatches) < mismatch_limit:
            mismatches.append(
                {
                    "path": path_str,
                    "gt_raw": _safe_scalar(gt),
                    "pred_raw": _safe_scalar(pred),
                    "gt_value": None,
                    "pred_value": None,
                    "reason": "pred_missing",
                }
            )
        return

    fmt = schema_eff.get("format") if isinstance(schema_eff.get("format"), str) else None

    matched, reason, pred_v, gt_v = _compare_canon(pred, gt, fmt=fmt, str_mode=str_mode)

    if matched:
        m["matched"] += 1
        return

    m["mismatched"] += 1
    if reason == "gt_canon_error":
        m["gt_canon_error"] += 1
    elif reason == "pred_canon_error":
        m["pred_canon_error"] += 1
    elif reason == "canon_errors":
        m["canon_errors"] += 1
    else:
        m["value_mismatch"] += 1

    if len(mismatches) < mismatch_limit:
        mismatches.append(
            {
                "path": path_str,
                "gt_raw": _safe_scalar(gt),
                "pred_raw": _safe_scalar(pred),
                "gt_value": _safe_scalar(gt_v),
                "pred_value": _safe_scalar(pred_v),
                "reason": reason,
                "format": fmt,
            }
        )


def _schema_leaf_paths(schema: Dict[str, Any], root_schema: Dict[str, Any]=None, *, path: Tuple[Union[str, None], ...] = ()) -> List[str]:
    """Return leaf (scalar) paths derived from schema, mostly for reporting."""
    if root_schema is None:
        root_schema = schema
    schema_eff = _deref_schema(_pick_variant(schema, _MISSING, root_schema), root_schema)
    types = _schema_types(schema_eff)
    node_type = types[0] if types else None

    if node_type == "object" or isinstance(schema_eff.get("properties"), dict):
        out: List[str] = []
        props = schema_eff.get("properties", {})
        if isinstance(props, dict):
            for k, s in props.items():
                if isinstance(s, dict):
                    out.extend(_schema_leaf_paths(s, root_schema, path=path + (k,)))
        return out

    if node_type == "array" or "items" in schema_eff:
        items_schema = schema_eff.get("items", {})
        if isinstance(items_schema, dict):
            return _schema_leaf_paths(items_schema, root_schema, path=path + (None,))
        return []

    # Scalar
    out = ""
    for seg in path:
        if seg is None:
            out += "[*]"
        else:
            out += ("." if out else "") + str(seg)
    return [out or "<root>"]



def _sanitize_ext_from_name(filename: str, mime_type: str) -> str:
    fn = (filename or "").strip()
    ext = Path(fn).suffix.lower()
    if ext and re.fullmatch(r"\.[a-z0-9]{1,8}", ext or ""):
        return ext
    mt = (mime_type or "").split(";", 1)[0].strip().lower()
    guessed = mimetypes.guess_extension(mt) if mt else None
    if guessed and re.fullmatch(r"\.[a-z0-9]{1,8}", guessed):
        return guessed
    return ".bin"


def _inputs_rel_for_hash(key_id: str, sha256_hex: str, *, filename: str, mime_type: str) -> str:
    ext = _sanitize_ext_from_name(filename, mime_type)
    kid = re.sub(r"[^A-Za-z0-9_.-]+", "_", (key_id or "anon"))[:64] or "anon"
    return f"inputs/{kid}/sha256/{sha256_hex}{ext}"


def _persist_input_bytes(
    principal: ApiPrincipal,
    *,
    data: bytes,
    filename: str,
    mime_type: str,
) -> Tuple[str, str, bool]:
    """Persist input bytes under a content-addressed key. Returns (sha256_hex, input_rel, existed)."""
    sha = hashlib.sha256(data or b"").hexdigest()
    rel = _inputs_rel_for_hash(principal.key_id, sha, filename=filename, mime_type=mime_type)

    if s3_enabled():
        key = s3_key(rel)
        existed = s3_put_bytes(key=key, data=data, content_type=mime_type)
        return sha, rel, existed

    # Local mode: ARTIFACTS_DIR/inputs/<key_id>/sha256/<sha>.<ext>
    path = (ARTIFACTS_DIR / rel).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return sha, rel, True
    path.write_bytes(data)
    return sha, rel, False

async def _load_input_bytes_by_rel(input_rel: str) -> bytes:
    if not input_rel:
        raise HTTPException(status_code=400, detail="input_rel is empty.")

    if s3_enabled():
        key = s3_key(input_rel)
        return await asyncio.to_thread(s3_get_bytes, key)

    path = (ARTIFACTS_DIR / input_rel).resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail="Input file not found.")
    return await asyncio.to_thread(path.read_bytes)

# ---------------------------
# Routes
# ---------------------------


async def health() -> Dict[str, Any]:
    backend = settings.INFERENCE_BACKEND.strip().lower()
    out: Dict[str, Any] = {"ok": True, "inference_backend": backend}

    # show tasks status
    out["tasks_available"] = sorted(list(_TASKS.keys()))
    out["tasks_missing"] = dict(_TASK_LOAD_ERRORS)

    if backend == "mock":
        out["vllm_ok"] = None
        return out

    try:
        models = await vllm.models()
        out["vllm_ok"] = True
        out["models"] = [m.get("id") for m in models.get("data", [])]
        return out
    except Exception as e:
        out["vllm_ok"] = False
        out["error"] = str(e)
        return out



async def me(principal: ApiPrincipal = Depends(require_scopes([]))):
    return {"key_id": principal.key_id, "role": principal.role, "scopes": sorted(list(principal.scopes))}


class EvalBatchVsGTRequest(BaseModel):
    run_id: str = Field(..., description="Batch run_id to load from artifacts/batches/<day>/<run_id>.json")
    batch_date: Optional[str] = Field(default=None, description="Optional day (YYYY-MM-DD) to avoid searching over days.")
    max_days: int = Field(default=30, ge=1, le=365, description="How many recent days to search for batch artifact if batch_date is not provided.")
    gt_path: str = Field(..., description="Path to GT JSON file (must be under DATA_ROOT or ARTIFACTS_DIR).")
    gt_image_key: str = Field(default="file", description="Key inside each GT record that contains the image path/name (basename is used). Fallback: image_file.")
    gt_record_key: Optional[str] = Field(default=None, description="Optional key inside each GT record to use as the actual GT object (e.g. 'gt' or 'fields').")
    limit: Optional[int] = Field(default=None, ge=1, description="Optional max number of batch items to evaluate.")

    # Output size controls (keep endpoint ergonomic).
    include_worst: bool = Field(default=False, description="Include worst_samples only in the saved eval artifact instead of all samples.")
    samples_limit: int = Field(default=10, ge=0, le=200, description="How many worst samples to include (0 disables).")
    mismatches_per_sample: int = Field(default=20, ge=0, le=200, description="Max mismatches to include per sample.")

    # Canonicalization controls
    str_mode: str = Field(default="strip_collapse", description="x-string mode: exact | strip | strip_collapse")


class EvalFieldMetric(BaseModel):
    path: str
    total: int
    comparable: int
    matched: int
    mismatched: int
    accuracy: Optional[float] = None

    pred_missing: int = 0
    gt_missing: int = 0

    gt_canon_error: int = 0
    pred_canon_error: int = 0
    canon_errors: int = 0
    value_mismatch: int = 0


class EvalSampleSummary(BaseModel):
    image: str
    request_id: str
    ok: bool
    schema_ok: bool
    artifact_rel: Optional[str] = None
    gt_found: bool = False
    pred_found: bool = False
    mismatches: int = 0
    mismatches_preview: List[Dict[str, Any]] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class EvalBatchVsGTResult(BaseModel):
    eval_id: str
    run_id: str
    batch_date: Optional[str] = None
    task_id: Optional[str] = None
    model: Optional[str] = None
    inference_backend: Optional[str] = None

    batch_summary: Dict[str, Any]
    gt_summary: Dict[str, Any]
    eval_summary: Dict[str, Any]

    fields: List[EvalFieldMetric]
    worst_samples: Optional[List[EvalSampleSummary]] = None


class EvalBatchVsGTResponse(BaseModel):
    eval_id: str #eval_id,
    created_at: str #datetime.now(timezone.utc).isoformat(),
    run_id: str #run_id,
    gt_path: str #req.gt_path,
    batch_artifact_path: str
    eval_artifact_path: str #eval_artifact_path,


async def eval_batch_vs_gt(
    req: EvalBatchVsGTRequest,
) -> EvalBatchVsGTResponse:
    # Validate
    if req.str_mode not in {"exact", "strip", "strip_collapse"}:
        raise HTTPException(status_code=400, detail="str_mode must be one of: exact | strip | strip_collapse")

    # Locate batch artifact
    batch_path = find_batch_artifact_path(req.run_id, date=req.batch_date, max_days=req.max_days)
    if batch_path is None:
        raise HTTPException(status_code=404, detail="Batch artifact not found (check run_id and batch_date/max_days).")

    batch_obj = read_json_file(batch_path)
    if not isinstance(batch_obj, dict):
        raise HTTPException(status_code=500, detail="Invalid batch artifact JSON (expected object).")

    batch_date = artifact_date_from_path(batch_path) or (req.batch_date or "")
    task_id = str(batch_obj.get("task_id") or "")
    if not task_id:
        raise HTTPException(status_code=500, detail="Batch artifact missing task_id.")
    task = _get_task(task_id)

    # Load schema JSON for the task
    schema_json = task.get("schema_value")
    if not isinstance(schema_json, dict):
        raise HTTPException(status_code=500, detail=f"Task {task_id!r} has no schema_value (dict).")

    leaf_paths = _schema_leaf_paths(schema_json)

    # Load GT
    gt_p = _validate_under_any_root(Path(req.gt_path), roots=[DATA_ROOT, ARTIFACTS_DIR])
    gt_obj = read_json_file(gt_p)

    # GT may be {id: rec} or [rec, ...]
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

    # Evaluate
    items = batch_obj.get("items") if isinstance(batch_obj.get("items"), list) else []
    if req.limit is not None:
        items = items[: int(req.limit)]

    metrics: Dict[str, Dict[str, int]] = {}
    sample_summaries: List[Dict[str, Any]] = []

    gt_found = 0
    gt_missing = 0
    pred_found = 0
    pred_missing = 0

    mismatch_limit_total = 5000  # hard cap per sample to keep eval artifacts bounded

    for it in items:
        if not isinstance(it, dict):
            continue

        image = it.get("file") or it.get("image") or it.get("image_file")
        image = str(image) if image is not None else ""
        image_base = Path(image).name if image else ""

        request_id = str(it.get("request_id") or "")
        artifact_rel = it.get("artifact_rel")
        ok = bool(it.get("ok"))
        schema_ok = bool(it.get("schema_ok"))

        gt_rec = gt_index.get(image_base) if image_base else None
        gt_ok = gt_rec is not None
        if gt_ok:
            gt_found += 1
        else:
            gt_missing += 1

        # Resolve pred artifact
        pred_obj: Any = _MISSING
        artifact_abs: Optional[Path] = None
        if isinstance(artifact_rel, str) and artifact_rel:
            rel = Path(artifact_rel)
            if rel.is_absolute():
                artifact_abs = _validate_under_any_root(rel, roots=[ARTIFACTS_DIR, DATA_ROOT])
            else:
                # Old batch format might omit "extracts/" prefix.
                if len(rel.parts) >= 1 and DATE_RE.match(rel.parts[0]) and rel.parts[0].count("-") == 2:
                    rel = Path("extracts") / rel
                artifact_abs = (ARTIFACTS_DIR / rel).resolve()
        if artifact_abs is not None and artifact_abs.exists():
            art = read_artifact_json(artifact_abs)
            # Common locations for model output
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
        else:
            pred_missing += 1

        # Choose GT payload object
        gt_payload: Any = gt_rec if gt_rec is not None else _MISSING
        if gt_payload is not _MISSING and req.gt_record_key:
            if isinstance(gt_payload, dict) and req.gt_record_key in gt_payload:
                gt_payload = gt_payload.get(req.gt_record_key)
            else:
                # If requested key is missing, treat as missing GT for scoring.
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
                key_fallbacks=False,
                mismatch_limit=mismatch_limit_total,
            )
        elif gt_payload is not _MISSING and pred_obj is _MISSING:
            # Count missing pred for every leaf path in schema
            for pth in leaf_paths:
                m = metrics.setdefault(pth, defaultdict(int))
                m["total"] += 1
                m["comparable"] += 1
                m["pred_missing"] += 1
                m["mismatched"] += 1

        # Build sample summary
        mismatches_preview = mismatches[: int(req.mismatches_per_sample)] if req.mismatches_per_sample > 0 else []

        sample_summaries.append(
            {
                "image": image_base or image,
                "request_id": request_id,
                "ok": ok,
                "schema_ok": schema_ok,
                "artifact_rel": str(artifact_rel) if artifact_rel is not None else None,
                "gt_found": gt_ok,
                "pred_found": pred_obj is not _MISSING,
                "mismatches": len(mismatches),
                "mismatches_preview": mismatches_preview,
                "notes": [],
            }
        )

    # Build per-field metrics (ordered by schema leaf paths if possible)
    order = {p: i for i, p in enumerate(leaf_paths)}
    field_rows: List[EvalFieldMetric] = []
    for path_str, m in metrics.items():
        total = int(m.get("total", 0))
        comparable = int(m.get("comparable", 0))
        matched = int(m.get("matched", 0))
        mismatched = int(m.get("mismatched", 0))
        acc = (matched / comparable) if comparable else None

        field_rows.append(
            EvalFieldMetric(
                path=path_str,
                total=total,
                comparable=comparable,
                matched=matched,
                mismatched=mismatched,
                accuracy=acc,
                pred_missing=int(m.get("pred_missing", 0)),
                gt_missing=int(m.get("gt_missing", 0)),
                gt_canon_error=int(m.get("gt_canon_error", 0)),
                pred_canon_error=int(m.get("pred_canon_error", 0)),
                canon_errors=int(m.get("canon_errors", 0)),
                value_mismatch=int(m.get("value_mismatch", 0)),
            )
        )

    field_rows.sort(key=lambda r: (order.get(r.path, 10**9), r.path))

    comparable_total = sum(r.comparable for r in field_rows)
    matched_total = sum(r.matched for r in field_rows)
    overall_accuracy = (matched_total / comparable_total) if comparable_total else None

    # Pick worst samples (by mismatches)
    worst_samples: Optional[List[Dict[str, Any]]] = None
    if req.include_worst and req.samples_limit > 0:
        worst = [s for s in sample_summaries if int(s.get("mismatches") or 0) > 0]
        worst.sort(key=lambda s: int(s.get("mismatches") or 0), reverse=True)
        worst_samples = worst[: int(req.samples_limit)]
        sample_summaries = []

    eval_id = f"{req.run_id}__eval_{uuid.uuid4().hex[:8]}"
    created_at = datetime.now(timezone.utc).isoformat()

    payload: Dict[str, Any] = {
        "eval_id": eval_id,
        "created_at": created_at,
        "run_id": req.run_id,
        "gt_path": str(gt_p),
        "batch_artifact_path": str(batch_path),
        "batch_date": batch_date,
        "task_id": task_id,
        "model": batch_obj.get("model"),
        "inference_backend": batch_obj.get("inference_backend"),
        "batch_summary": batch_obj.get("summary", {}),
        "gt_summary": {
            "gt_path": str(gt_p),
            "gt_records": len(gt_index),
            "gt_image_key": req.gt_image_key,
            "gt_found_for_batch_items": gt_found,
            "gt_missing_for_batch_items": gt_missing,
        },
        "eval_summary": {
            "items_considered": len(items),
            "pred_found": pred_found,
            "pred_missing": pred_missing,
            "leaf_paths_in_schema": len(leaf_paths),
            "comparable_total": comparable_total,
            "matched_total": matched_total,
            "overall_accuracy": overall_accuracy,
            "str_mode": req.str_mode,
            "decimal_sep": _get_decimal_sep(),
        },
        "fields": [r.model_dump() for r in field_rows],
        "worst_samples": worst_samples,
        "all_samples": sample_summaries,
    }

    eval_artifact_path = await asyncio.to_thread(save_eval_artifact, payload)

    return EvalBatchVsGTResponse(
        eval_id=eval_id,
        created_at=created_at,
        run_id=req.run_id,
        gt_path=str(gt_p),
        batch_artifact_path=str(batch_path),
        eval_artifact_path=eval_artifact_path,
    )


async def extract(
    req: ExtractRequest,
    principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
) -> ExtractResponse:
    task = _get_task(req.task_id)

    request_id = make_request_id(req.request_id or _ctx_request_id())
    _validate_request_id_for_fs(request_id)

    # image_url is allowed only as a DEBUG feature (DEBUG_MODE=1 and debug:run scope).
    if req.image_url:
        _debug_allowed(principal)

    t0 = time.perf_counter()

    prompt_text, temperature, max_tokens, schema_json = _apply_debug_overrides(
        principal,
        req.debug,
        base_prompt_text=task["prompt_value"],
        base_schema_json=task["schema_value"],
    )

    messages = _build_messages(prompt_text, req)

    schema_dict = _normalize_schema(schema_json, default_schema=task["schema_value"])
    is_custom_schema = bool(req.debug and req.debug.schema_override and req.debug.schema_override.strip())
    schema_name = "custom_schema" if is_custom_schema else task["schema_name"]
    response_format = _build_response_format(schema_dict, schema_name=schema_name) if schema_dict is not None else None

    backend = settings.INFERENCE_BACKEND.strip().lower()

    # Record system-level errors/retries in artifacts (and optionally return them).
    error_history: List[Dict[str, Any]] = []
    image_resize: Optional[Dict[str, Any]] = None

    messages_cur = messages
    mime_cur = req.mime_type

    t_inf0 = time.perf_counter()
    t_inf1 = t_inf0

    retried = False
    attempt = 0

    prompt_sha256 = hashlib.sha256(task["prompt_value"].encode("utf-8")).hexdigest()
    schema_obj = task["schema_value"]  
    schema_canon = json.dumps(schema_obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    schema_sha256 = hashlib.sha256(schema_canon.encode("utf-8")).hexdigest()

    artifact_base = {
    "request_id": request_id,
    "task_id": req.task_id,
    "model": MODEL_ID,
    "inference_backend": backend,
    "input": _sanitize_input_for_artifact(req, image_resize=image_resize),
    "auth": {"key_id": principal.key_id, "role": principal.role},
    "image_resize": image_resize,
    "prompt_sha256": prompt_sha256,
    "schema_sha256": schema_sha256,
    }

    while True:
        try:
            t_inf0 = time.perf_counter()
            if backend == "mock":
                raw = _mock_chat_completions(request_id, principal, req.debug, schema_dict=schema_dict)
            else:
                _log_event(logging.INFO, 'vllm_call_start', backend=backend, model=MODEL_ID, attempt=attempt, retried=retried)
                raw = await vllm.chat_completions(
                    model=MODEL_ID,
                    messages=messages_cur,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    request_id=request_id,
                )
            t_inf1 = time.perf_counter()
            _log_event(logging.INFO, 'vllm_call_end', backend=backend, model=MODEL_ID, attempt=attempt, duration_ms=(t_inf1 - t_inf0) * 1000.0)
            break

        except Exception as e:
            err = str(e)
            _log_event(logging.WARNING, 'vllm_call_error', backend=backend, model=MODEL_ID, attempt=attempt, error_type=e.__class__.__name__)
            error_history.append(_mk_err("inference_request" if attempt == 0 else "retry_request", err))

            # One-shot smart retry: if server says the decoder prompt is too long, downscale the image
            # by a factor derived from server-provided token counts.
            if attempt == 0 and backend != "mock":
                ctx = _parse_decoder_prompt_too_long(err)
                if ctx and req.image_base64 and (not req.image_url):
                    t_resp, t_max = ctx
                    t_prompt_est = _estimate_prompt_tokens(prompt_text, schema_dict)
                    scale = _compute_retry_scale(t_resp=t_resp, t_max=t_max, t_prompt_est=t_prompt_est)
                    if scale is None:
                        error_history.append(
                            _mk_err(
                                "retry_skipped",
                                "Context-length error but estimated prompt is already too large; image downscale cannot help.",
                                t_resp=t_resp,
                                t_max=t_max,
                                t_prompt_est=t_prompt_est,
                            )
                        )
                    else:
                        try:
                            b64_new, resize_info = _resize_base64_image_by_scale(req.image_base64, req.mime_type, scale)
                            if resize_info.get("applied"):
                                image_resize = {
                                    **resize_info,
                                    "t_resp": int(t_resp),
                                    "t_max": int(t_max),
                                    "t_prompt_est": int(t_prompt_est),
                                }

                                mime_cur = image_resize.get("sent_mime_type") or req.mime_type
                                messages_cur = [
                                    {
                                        "role": "user",
                                        "content": [
                                            _image_content(req.image_url, b64_new, mime_cur),
                                            {"type": "text", "text": prompt_text},
                                        ],
                                    }
                                ]

                                retried = True
                                attempt = 1
                                continue

                            error_history.append(
                                _mk_err(
                                    "retry_skipped",
                                    "Computed resize scale resulted in no size change; skipping retry.",
                                    scale=float(scale),
                                )
                            )
                        except Exception as e2:
                            error_history.append(_mk_err("retry_failed", str(e2)))
            

            # Inference failed (and retry either didn't happen or also failed).
            artifact_path = await asyncio.to_thread(save_artifact,
                request_id,
                {
                    **artifact_base,
                    "error_history": error_history,
                    "error_stage": "inference_request",
                    "vllm_request_meta": {
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "has_response_format": response_format is not None,
                        "retried": retried,
                    },
                    "timings_ms": {
                        "total_ms": (time.perf_counter() - t0) * 1000.0,
                    },
                },
            )
            detail = f"Inference request failed ({backend}): {err}"
            if _raw_allowed(principal):
                detail += f" | artifact_rel={to_artifact_rel(artifact_path)}"
            raise HTTPException(status_code=502, detail=detail)

    try:
        raw_text = raw["choices"][0]["message"]["content"]
    except Exception as e:
        error_history.append(_mk_err("unexpected_response_format", str(e)))

        artifact_path = await asyncio.to_thread(save_artifact,
            request_id,
            {
                **artifact_base,
                "error_stage": "unexpected_response_format",
                "error_history": error_history,
                "raw_response": raw,
                "timings_ms": {
                    "inference_ms": (t_inf1 - t_inf0) * 1000.0,
                    "total_ms": (time.perf_counter() - t0) * 1000.0,
                },
            },
        )
        detail = f"Unexpected {backend} response format."
        if _raw_allowed(principal):
            detail += f" | artifact_rel={to_artifact_rel(artifact_path)}"
        raise HTTPException(status_code=502, detail=detail)

    # Parse JSON produced by the model.
    try:
        parsed = json.loads(raw_text)
    except Exception as e:
        error_history.append(_mk_err("parse_json", str(e)))

        # Сохраняем артефакт, иначе реально “некуда посмотреть”
        artifact_path = await asyncio.to_thread(save_artifact,
            request_id,
            {
                **artifact_base,
                "error_stage": "parse_json",
                "error_history": error_history,
                "raw_model_text": raw_text,
                "raw_response": raw,
                "vllm_request_meta": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "has_response_format": response_format is not None,
                },
                "timings_ms": {
                    "inference_ms": (t_inf1 - t_inf0) * 1000.0,
                    "total_ms": (time.perf_counter() - t0) * 1000.0,
                },
            },
        )
        detail = f"Model output is not valid JSON ({backend}): {e}"
        if _raw_allowed(principal):
            detail += f" | artifact_rel={to_artifact_rel(artifact_path)}"
        raise HTTPException(status_code=502, detail=detail)


    if schema_dict is None:
        schema_ok = True
        errors: list[str] = []
    else:
        schema_json_validator = _make_validator_or_raise(schema_dict, is_custom_schema=is_custom_schema)
        errors = [e.message for e in schema_json_validator.iter_errors(parsed)]
        schema_ok = len(errors) == 0

    artifact_key = await asyncio.to_thread(save_artifact,
        request_id,
        {
            **artifact_base,
            "error_history": error_history,
            "schema": None if schema_dict is None else json.dumps(schema_dict, ensure_ascii=False),
            "schema_ok": schema_ok,
            "schema_errors": errors,
            "raw_model_text": raw_text,
            "parsed": parsed,
            "raw_response": raw,
            "timings_ms": {
                "inference_ms": (t_inf1 - t_inf0) * 1000.0,
                "total_ms": (time.perf_counter() - t0) * 1000.0,
            },
        },
    )

    raw_out = raw_text if _raw_allowed(principal) else None

    return ExtractResponse(
        request_id=request_id,
        task_id=req.task_id,
        model=MODEL_ID,
        schema_ok=schema_ok,
        result=parsed if isinstance(parsed, dict) else {"_parsed": parsed},
        raw_model_text=raw_out,
        image_resize=image_resize,
        error_history=error_history,
        artifact_rel=to_artifact_rel(artifact_key),
        timings_ms={
            "inference_ms": (t_inf1 - t_inf0) * 1000.0,
            "total_ms": (time.perf_counter() - t0) * 1000.0,
        },
        schema_errors=errors if not schema_ok else None,
    )


@dataclass
class _BatchInput:
    image_ref: str
    request_id: str
    mime_type: str
    load_bytes: Callable[[], Awaitable[bytes]]

    # Optional persistence metadata (used by upload/rerun flows)
    orig_name: Optional[str] = None
    sha256: Optional[str] = None
    input_rel: Optional[str] = None
    input_existed: Optional[bool] = None
    bytes_len: Optional[int] = None


async def _run_batch_extract_core(
    *,
    run_id: str,
    task_id: str,
    task: Dict[str, Any],
    inputs: List[_BatchInput],
    concurrency: int,
    principal: ApiPrincipal,
    source: Dict[str, Any],
    batch_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Shared batch runner: prepares ExtractRequest and calls /v1/extract for each prepared input."""
    t0 = time.perf_counter()
    sem = asyncio.Semaphore(int(concurrency))

    async def _process_one(inp: _BatchInput) -> Dict[str, Any]:
        async with sem:
            request_id = inp.request_id
            try:
                data = await inp.load_bytes()
                b64 = base64.b64encode(data).decode("ascii")

                sub_req = ExtractRequest(
                    task_id=task_id,
                    image_base64=b64,
                    mime_type=inp.mime_type or "image/png",
                    request_id=request_id,
                    image_ref=inp.image_ref,
                    debug=None,
                )

                token_item = _set_request_id(request_id)
                try:
                    resp = await extract(sub_req, principal)
                finally:
                    _reset_request_id(token_item)

                item = {
                    "image": inp.image_ref,
                    "request_id": resp.request_id,
                    "artifact_rel": resp.artifact_rel,
                    "ok": True,
                    "schema_ok": resp.schema_ok,
                    "timings_ms": resp.timings_ms,
                    "image_resize": resp.image_resize,
                    "error_history": resp.error_history,
                }
                if inp.sha256:
                    item["sha256"] = inp.sha256
                if inp.input_rel:
                    item["input_rel"] = inp.input_rel
                if inp.input_existed is not None:
                    item["input_existed"] = bool(inp.input_existed)
                if inp.bytes_len is not None:
                    item["bytes"] = int(inp.bytes_len)
                return item

            except HTTPException as he:
                art_p = find_artifact_path(request_id, max_days=30)
                artifact_rel = to_artifact_rel(art_p) if art_p is not None else None
                err_hist = await asyncio.to_thread(_try_load_error_history_for_request, request_id) or [
                    _mk_err("http_error", f"HTTP {he.status_code}: {he.detail}")
                ]
                item = {
                    "image": inp.image_ref,
                    "request_id": request_id,
                    "artifact_rel": artifact_rel,
                    "ok": False,
                    "schema_ok": False,
                    "error_history": err_hist,
                }
                if inp.sha256:
                    item["sha256"] = inp.sha256
                if inp.input_rel:
                    item["input_rel"] = inp.input_rel
                if inp.input_existed is not None:
                    item["input_existed"] = bool(inp.input_existed)
                if inp.bytes_len is not None:
                    item["bytes"] = int(inp.bytes_len)
                return item

            except Exception as e:
                art_p = find_artifact_path(request_id, max_days=30)
                artifact_rel = to_artifact_rel(art_p) if art_p is not None else None
                err_hist = await asyncio.to_thread(_try_load_error_history_for_request,request_id) or [
                    _mk_err("exception", f"{e.__class__.__name__}: {e}")
                ]
                item = {
                    "image": inp.image_ref,
                    "request_id": request_id,
                    "artifact_rel": artifact_rel,
                    "ok": False,
                    "schema_ok": False,
                    "error_history": err_hist,
                }
                if inp.sha256:
                    item["sha256"] = inp.sha256
                if inp.input_rel:
                    item["input_rel"] = inp.input_rel
                if inp.input_existed is not None:
                    item["input_existed"] = bool(inp.input_existed)
                if inp.bytes_len is not None:
                    item["bytes"] = int(inp.bytes_len)
                return item

    items = list(await asyncio.gather(*[_process_one(i) for i in inputs]))

    total = len(items)
    ok = sum(1 for it in items if it.get("ok"))
    failed = total - ok
    schema_ok = sum(1 for it in items if it.get("ok") and it.get("schema_ok"))
    schema_failed = sum(1 for it in items if it.get("ok") and (not it.get("schema_ok")))

    t1 = time.perf_counter()

    prompt_sha256 = hashlib.sha256(task["prompt_value"].encode("utf-8")).hexdigest()
    schema_obj = task["schema_value"]
    schema_canon = json.dumps(schema_obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    schema_sha256 = hashlib.sha256(schema_canon.encode("utf-8")).hexdigest()

    inputs_index: List[Dict[str, Any]] = []
    for inp in inputs:
        if inp.sha256 or inp.input_rel:
            inputs_index.append(
                {
                    "image_ref": inp.image_ref,
                    "orig_name": inp.orig_name or inp.image_ref,
                    "sha256": inp.sha256,
                    "input_rel": inp.input_rel,
                    "input_existed": inp.input_existed,
                    "bytes": inp.bytes_len,
                    "mime_type": inp.mime_type,
                }
            )

    batch_payload: Dict[str, Any] = {
        "run_id": run_id,
        "task_id": task_id,
        "source": source,
        "task_snapshot": {
            "prompt_text": task["prompt_value"],
            "schema_json": task["schema_value"],
            "prompt_sha256": prompt_sha256,
            "schema_sha256": schema_sha256,
        },
        "concurrency": int(concurrency),
        "model": MODEL_ID,
        "inference_backend": settings.INFERENCE_BACKEND.strip().lower(),
        "auth": {"key_id": principal.key_id, "role": principal.role},
        "summary": {
            "total": total,
            "ok": ok,
            "failed": failed,
            "schema_ok": schema_ok,
            "schema_failed": schema_failed,
        },
        "timings_ms": {
            "total_ms": (t1 - t0) * 1000.0,
        },
        "items": items,
    }

    if batch_fields:
        batch_payload.update(batch_fields)

    if inputs_index:
        batch_payload["inputs_index"] = inputs_index

    batch_artifact_path = await asyncio.to_thread(save_batch_artifact,run_id, batch_payload)

    return {
        "run_id": run_id,
        "batch_artifact_path": batch_artifact_path,
        "items": items,
        "summary": batch_payload["summary"],
        "timings_ms": batch_payload["timings_ms"],
    }


class BatchExtractResponse(BaseModel):
    run_id: str
    task_id: str
    images_dir: Optional[str] = Field(default=None, description="Optional image source directory (if provided).")

    total: int
    ok: int
    failed: int
    schema_ok: int
    schema_failed: int

    timings_ms: Dict[str, float]
    batch_artifact_path: str

    eval: Optional[Dict[str, Any]] = Field(default=None, description="Optional eval result (if gt_path was provided).")
    

async def _run_eval_batch_vs_gt(
    *,
    gt_path: Optional[str],
    gt_image_key: str,
    run_id: str,
    batch_artifact_path: str,
    principal: ApiPrincipal,
) -> Optional[Dict[str, Any]]:
    if not gt_path:
        return None

    _debug_enabled()
    if settings.AUTH_ENABLED and ("debug:read_raw" not in principal.scopes):
        raise HTTPException(status_code=403, detail="Missing required scope: debug:read_raw")

    _validate_under_any_root(Path(gt_path), roots=[DATA_ROOT, ARTIFACTS_DIR])

    eval_req = EvalBatchVsGTRequest(
        run_id=run_id,
        batch_date=Path(str(batch_artifact_path)).parent.name,
        gt_path=gt_path,
        gt_image_key=gt_image_key or "image",
        include_worst=False,
        str_mode="exact",
    )
    eval_resp = await eval_batch_vs_gt(eval_req)
    return eval_resp.model_dump()


async def batch_extract(
    task_id: str = Form(default="receipt_fields_v1", description="Which extraction task to run for all images."),
    images_dir: str = Form(default=DEFAULT_IMAGES_DIR, description="Server-side directory with images (must be under /workspace/src)."),
    glob: str = Form(default="**/*", description="Glob pattern relative to images_dir."),
    exts: List[str] = Form(default_factory=lambda: ["png", "jpg", "jpeg", "webp"], description="Allowed file extensions (without dot)."),
    limit: Optional[int] = Form(default=None, ge=1, description="Optional max number of images to process."),
    concurrency: int = Form(default=4, ge=1, le=32, description="Number of concurrent requests within this process."),
    run_id: Optional[str] = Form(default=None, description="Optional run id (used to build per-image request_id)."),

    gt_path: Optional[str] = Form(default=None, description=("If provided, the server will run batch eval vs GT after batch_extract.For safety this requires DEBUG_MODE=1 and scope debug:run. Path must be under DATA_ROOT or ARTIFACTS_DIR.")),
    gt_image_key: str = Form(default="file",description="Key in each GT record that contains image path/name (basename is used). Fallback: image_file."),

    principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
) -> BatchExtractResponse:
    task = _get_task(task_id)

    run_id = (run_id or _default_run_id()).strip()
    _validate_request_id_for_fs(run_id)

    images_dir = _validate_under_data_root(Path(images_dir).expanduser())
    files = _list_image_files(images_dir, glob, exts, limit)

    inputs: List[_BatchInput] = []
    for p in files:
        rel_path = str(p.relative_to(images_dir)).replace("\\", "/")
        request_id = _make_batch_request_id(run_id, rel_path)
        mime = _guess_mime_type(p)

        async def _lb(pp=p) -> bytes:
            return await asyncio.to_thread(pp.read_bytes)

        inputs.append(_BatchInput(image_ref=rel_path, request_id=request_id, mime_type=mime, load_bytes=_lb))

    res = await _run_batch_extract_core(
        run_id=run_id,
        task_id=task_id,
        task=task,
        inputs=inputs,
        concurrency=concurrency,
        principal=principal,
        source={
            "type": "server_dir",
            "images_dir": str(images_dir),
            "glob": glob,
            "exts": exts,
            "limit": limit,
        },
        batch_fields={
            "images_dir": str(images_dir),
            "glob": glob,
            "exts": exts,
            "limit": limit,
        },
    )

    # --- AUTO-EVAL (optional) ---
    eval_obj = await _run_eval_batch_vs_gt(
        gt_path=gt_path,
        gt_image_key=gt_image_key,
        run_id=run_id,
        batch_artifact_path=str(res["batch_artifact_path"]),
        principal=principal,
    )
    summ = res["summary"]
    return BatchExtractResponse(
        run_id=run_id,
        task_id=task_id,
        images_dir=str(images_dir),
        total=int(summ["total"]),
        ok=int(summ["ok"]),
        failed=int(summ["failed"]),
        schema_ok=int(summ["schema_ok"]),
        schema_failed=int(summ["schema_failed"]),
        timings_ms=res["timings_ms"],
        batch_artifact_path=str(res["batch_artifact_path"]),
        eval=eval_obj,
    )


async def batch_extract_upload(
    task_id: str = Form(...),
    persist_inputs: bool = Form(False),
    concurrency: int = Form(4),
    run_id: Optional[str] = Form(None),
    
    files: List[UploadFile] = File(...),

    gt_path: Optional[str] = Form(default=None, description=("If provided, the server will run batch eval vs GT after batch_extract.For safety this requires DEBUG_MODE=1 and scope debug:run. Path must be under DATA_ROOT or ARTIFACTS_DIR.")),
    gt_image_key: str = Form(default="file",description="Key in each GT record that contains image path/name (basename is used). Fallback: image_file."),

    principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
) -> BatchExtractResponse:
    task = _get_task(task_id)

    run_id = (run_id or _default_run_id()).strip()
    _validate_request_id_for_fs(run_id)

    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    inputs: List[_BatchInput] = []
    max_files = int(os.getenv("BATCH_UPLOAD_MAX_FILES", "500"))
    if len(files) > max_files:
        raise HTTPException(status_code=400, detail=f"Too many files (>{max_files}).")

    for idx, uf in enumerate(files):
        name = (uf.filename or f"file_{idx}").strip() or f"file_{idx}"
        data = await uf.read()
        mt = (uf.content_type or mimetypes.guess_type(name)[0] or "image/png").strip()

        sha: Optional[str] = None
        input_rel: Optional[str] = None
        existed: Optional[bool] = None

        if persist_inputs:
            sha, input_rel, existed = _persist_input_bytes(principal, data=data, filename=name, mime_type=mt)

        req_id = _make_batch_request_id(run_id, name)

        async def _lb(d=data) -> bytes:
            return d

        inputs.append(
            _BatchInput(
                image_ref=name,
                request_id=req_id,
                mime_type=mt,
                load_bytes=_lb,
                orig_name=name,
                sha256=sha,
                input_rel=input_rel,
                input_existed=existed,
                bytes_len=len(data) if data is not None else None,
            )
        )

    res = await _run_batch_extract_core(
        run_id=run_id,
        task_id=task_id,
        task=task,
        inputs=inputs,
        concurrency=concurrency,
        principal=principal,
        source={
            "type": "upload",
            "persist_inputs": bool(persist_inputs),
            "count": len(inputs),
        },
        batch_fields={
            "persist_inputs": bool(persist_inputs),
            "upload_count": len(inputs),
        },
    )
    # --- AUTO-EVAL (optional) ---
    
    eval_obj = await _run_eval_batch_vs_gt(
        gt_path=gt_path,
        gt_image_key=gt_image_key,
        run_id=run_id,
        batch_artifact_path=str(res["batch_artifact_path"]),
        principal=principal,
    )


    summ = res["summary"]
    return BatchExtractResponse(
        run_id=run_id,
        task_id=task_id,
        total=int(summ["total"]),
        ok=int(summ["ok"]),
        failed=int(summ["failed"]),
        schema_ok=int(summ["schema_ok"]),
        schema_failed=int(summ["schema_failed"]),
        timings_ms=res["timings_ms"],
        batch_artifact_path=str(res["batch_artifact_path"]),
        eval=eval_obj,
    )


async def batch_extract_rerun(
    source_run_id: str = Form(..., description="Existing batch run_id to rerun."),
    source_batch_date: Optional[str] = Form(default=None, description="Optional day (YYYY-MM-DD) to avoid searching over days."),
    max_days: int = Form(default=90, ge=1, le=365, description="How many recent days to search if source_batch_date is not provided."),
    
    task_id: str = Form(default="receipt_fields_v1", description="Task to run for the rerun."),
    concurrency: int = Form(default=4, ge=1, le=32),
    run_id: Optional[str] = Form(default=None, description="Optional new run_id (if omitted, generated)."),

    gt_path: Optional[str] = Form(default=None, description=("If provided, the server will run batch eval vs GT after batch_extract.For safety this requires DEBUG_MODE=1 and scope debug:run. Path must be under DATA_ROOT or ARTIFACTS_DIR.")),
    gt_image_key: str = Form(default="file",description="Key in each GT record that contains image path/name (basename is used). Fallback: image_file."),

    principal: ApiPrincipal = Depends(require_scopes(["extract:run"])),
) -> BatchExtractResponse:
    task = _get_task(task_id)
    
    src_path = find_batch_artifact_path(source_run_id, date=source_batch_date, max_days=max_days)
    if src_path is None:
        raise HTTPException(status_code=404, detail="Source batch artifact not found.")

    src_obj = read_json_file(src_path)
    if not isinstance(src_obj, dict):
        raise HTTPException(status_code=500, detail="Invalid source batch artifact JSON (expected object).")

    src_auth = src_obj.get("auth") if isinstance(src_obj.get("auth"), dict) else {}
    src_key_id = str(src_auth.get("key_id") or "")
    if settings.AUTH_ENABLED and principal.role != "admin":
        if not src_key_id or src_key_id != principal.key_id:
            raise HTTPException(status_code=403, detail="You may only rerun your own batches.")

    inputs_index = src_obj.get("inputs_index")
    if not isinstance(inputs_index, list) or not inputs_index:
        raise HTTPException(
            status_code=400,
            detail="Source batch has no persisted inputs (inputs_index missing/empty). Re-run is not possible.",
        )

    run_id = (run_id or _default_run_id()).strip()
    _validate_request_id_for_fs(run_id)

    inputs: List[_BatchInput] = []
    for idx, ix in enumerate(inputs_index):
        if not isinstance(ix, dict):
            continue
        input_rel = str(ix.get("input_rel") or "")
        if not input_rel:
            continue

        orig_name = str(ix.get("orig_name") or ix.get("image_ref") or f"file_{idx}")
        mt = str(ix.get("mime_type") or mimetypes.guess_type(orig_name)[0] or "image/png")
        sha = str(ix.get("sha256") or "") or None

        req_id = _make_batch_request_id(run_id, orig_name)

        async def _lb(rel=input_rel) -> bytes:
            return await _load_input_bytes_by_rel(rel)

        inputs.append(
            _BatchInput(
                image_ref=orig_name,
                request_id=req_id,
                mime_type=mt,
                load_bytes=_lb,
                orig_name=orig_name,
                sha256=sha,
                input_rel=input_rel,
                input_existed=True,
                bytes_len=int(ix.get("bytes")) if isinstance(ix.get("bytes"), int) else None,
            )
        )

    if not inputs:
        raise HTTPException(status_code=400, detail="No usable persisted inputs found in inputs_index.")

    res = await _run_batch_extract_core(
        run_id=run_id,
        task_id=task_id,
        task=task,
        inputs=inputs,
        concurrency=concurrency,
        principal=principal,
        source={
            "type": "rerun",
            "source_run_id": source_run_id,
            "source_batch_date": artifact_date_from_path(src_path),
            "source_batch_artifact": to_artifact_rel(src_path) if src_path is not None else None,
        },
        batch_fields={
            "source_run_id": source_run_id,
            "source_batch_date": artifact_date_from_path(src_path),
        },
    )
    # --- AUTO-EVAL (optional) ---
    eval_obj = await _run_eval_batch_vs_gt(
        gt_path=gt_path,
        gt_image_key=gt_image_key,
        run_id=run_id,
        batch_artifact_path=str(res["batch_artifact_path"]),
        principal=principal,
    )

    summ = res["summary"]
    return BatchExtractResponse(
        run_id=run_id,
        task_id=task_id,
        total=int(summ["total"]),
        ok=int(summ["ok"]),
        failed=int(summ["failed"]),
        schema_ok=int(summ["schema_ok"]),
        schema_failed=int(summ["schema_failed"]),
        timings_ms=res["timings_ms"],
        batch_artifact_path=str(res["batch_artifact_path"]),
        eval=eval_obj,
    )
