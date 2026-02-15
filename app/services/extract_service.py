# app/services/extract_service.py
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import mimetypes
import random
import re
import time
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from fastapi import HTTPException, UploadFile
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel, Field

from app.auth import ApiPrincipal
from app.settings import settings
from app.vllm_client import VLLMClient, VLLMHTTPError
from app.services.artifacts import (
    artifact_date_from_path,
    read_artifact_json,
    save_artifact,
    save_batch_artifact,
    find_artifact_path,
    find_batch_artifact_path,
)
from app.services.s3_service import s3_enabled, s3_get_bytes, s3_key, s3_put_bytes, s3_put_json

from app.handlers import (
    _ctx_request_id,
    _get_task,
    _log_event,
    _reset_request_id,
    _set_request_id,
    _validate_request_id_for_fs,
    _validate_under_any_root,
    _validate_under_data_root,
    ARTIFACTS_DIR,
    DATA_ROOT,
    DEFAULT_IMAGES_DIR,
    MODEL_ID,
    make_request_id,
)
from app.handlers import EvalBatchVsGTRequest, eval_batch_vs_gt

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None


LOG = logging.getLogger("ocrlty.extract")


# -----------------------
# Public request/response models
# -----------------------
class DebugOptions(BaseModel):
    prompt_override: Optional[str] = Field(default=None, description="Override prompt (debug only).")
    schema_override: Optional[Dict[str, Any]] = Field(default=None, description="Override schema (debug only).")
    schema_override_json: Optional[str] = Field(default=None, description="Override schema as JSON string (debug only).")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Override temperature (debug only).")
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192, description="Override max_tokens (debug only).")


class ExtractRequest(BaseModel):
    task_id: str = Field(..., description="Task registry id.")
    request_id: Optional[str] = Field(default=None, description="Optional request id. If omitted - generated.")
    image_url: Optional[str] = Field(default=None, description="DEBUG ONLY. Remote image URL.")
    image_base64: Optional[str] = Field(default=None, description="Base64-encoded bytes of image.")
    image_ref: Optional[str] = Field(default=None, description="Artifact input ref (returned by batch_upload with persist_inputs).")
    mime_type: Optional[str] = Field(default=None, description="Mime type for base64/ref image. e.g. image/jpeg")
    debug: Optional[DebugOptions] = Field(default=None, description="Debug overrides.")


class ExtractResponse(BaseModel):
    request_id: str
    created_at: str
    task_id: str
    artifact_path: str

    model_id: str
    prompt_sha256: str
    schema_sha256: str

    raw: Optional[Any] = None
    parsed: Optional[Dict[str, Any]] = None

    schema_valid: bool = True
    schema_errors: Optional[List[Dict[str, Any]]] = None

    error: Optional[Dict[str, Any]] = None
    error_history: Optional[List[Dict[str, Any]]] = None

    meta: Dict[str, Any] = Field(default_factory=dict)


class BatchExtractRequest(BaseModel):
    task_id: str = Field(..., description="Task registry id.")
    images_dir: str = Field(default=DEFAULT_IMAGES_DIR, description="Directory under DATA_ROOT containing images.")
    glob: str = Field(default="**/*", description="Glob for images_dir scan.")
    concurrency: int = Field(default=4, ge=1, le=64)
    run_id: Optional[str] = Field(default=None, description="Optional run id. If omitted - generated.")
    gt_path: Optional[str] = Field(default=None, description="Optional GT JSON path to auto-run eval (debug only).")
    gt_image_key: str = Field(default="image", description="GT key with image filename/path.")
    gt_record_key: Optional[str] = Field(default=None, description="Optional: nested key in GT record with payload to compare.")
    include_worst: bool = Field(default=False)
    limit: Optional[int] = Field(default=None, ge=1, le=100000)



class BatchInputRef(BaseModel):
    file: str
    input_ref: str
    mime_type: Optional[str] = None


class BatchExtractInputsRequest(BaseModel):
    task_id: str = Field(..., description="Task registry id.")
    concurrency: int = Field(default=4, ge=1, le=64)
    run_id: Optional[str] = Field(default=None, description="Optional run id. If omitted - generated.")
    inputs: List[BatchInputRef] = Field(..., description="List of persisted input refs to process.")
    gt_path: Optional[str] = Field(default=None, description="Optional GT JSON path to auto-run eval (debug only).")
    gt_image_key: str = Field(default="image", description="GT key with image filename/path.")
    gt_record_key: Optional[str] = Field(default=None, description="Optional: nested key in GT record with payload to compare.")
    include_worst: bool = Field(default=False)
    limit: Optional[int] = Field(default=None, ge=1, le=100000)

class BatchItemResult(BaseModel):
    file: str
    request_id: str
    input_ref: Optional[str] = None
    artifact_rel: Optional[str] = None
    ok: bool = True
    error: Optional[Dict[str, Any]] = None
    error_history: Optional[List[Dict[str, Any]]] = None


class BatchExtractResponse(BaseModel):
    run_id: str
    created_at: str
    task_id: str
    images_dir: str
    glob: str
    concurrency: int
    item_count: int
    ok_count: int
    error_count: int
    artifact_path: str
    items: List[BatchItemResult]
    eval: Optional[Dict[str, Any]] = None


class BatchRerunRequest(BaseModel):
    task_id: str = Field(..., description="Task registry id.")
    run_id: str = Field(..., description="Old run_id to rerun from artifacts.")
    batch_date: Optional[str] = Field(default=None, description="Optional day (YYYY-MM-DD).")
    max_days: int = Field(default=30, ge=1, le=365)
    concurrency: int = Field(default=4, ge=1, le=64)
    new_run_id: Optional[str] = Field(default=None, description="New run_id. If omitted - generated.")
    gt_path: Optional[str] = Field(default=None, description="Optional GT JSON path to auto-run eval (debug only).")
    gt_image_key: str = Field(default="image", description="GT key with image filename/path.")
    gt_record_key: Optional[str] = Field(default=None, description="Optional: nested key in GT record with payload to compare.")
    include_worst: bool = Field(default=False)
    limit: Optional[int] = Field(default=None, ge=1, le=100000)


# -----------------------
# Internal helpers
# -----------------------
def _now_utc_s() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _mk_err(code: str, message: str, *, detail: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = {"code": code, "message": message}
    if detail:
        out["detail"] = detail
    return out


def _debug_enabled() -> bool:
    return bool(getattr(settings, "DEBUG_MODE", False))


def _debug_allowed(principal: ApiPrincipal) -> None:
    if not _debug_enabled():
        # keep endpoints obscure in prod
        raise HTTPException(status_code=404, detail="Not found")
    if getattr(settings, "AUTH_ENABLED", True):
        if "debug:run" not in principal.scopes:
            raise HTTPException(status_code=403, detail="debug:run scope required")


def _raw_allowed(principal: ApiPrincipal) -> bool:
    if not _debug_enabled():
        return False
    if getattr(settings, "AUTH_ENABLED", True) and ("debug:raw" not in principal.scopes):
        return False
    return True


def _clamp_max_tokens(max_tokens: int) -> int:
    lim = int(getattr(settings, "VLLM_MAX_TOKENS_LIMIT", 8192) or 8192)
    return max(1, min(int(max_tokens), lim))


def _sanitize_input_for_artifact(req: ExtractRequest) -> Dict[str, Any]:
    data = req.model_dump()
    if data.get("image_base64"):
        s = str(data["image_base64"])
        data["image_base64"] = s[:256] + ("..." if len(s) > 256 else "")
    return data


def _normalize_schema(schema: Any) -> Dict[str, Any]:
    if schema is None:
        raise HTTPException(status_code=400, detail="Schema is required")
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except Exception:
            raise HTTPException(status_code=400, detail="schema_override_json must be valid JSON")
    if not isinstance(schema, dict):
        raise HTTPException(status_code=400, detail="Schema must be a JSON object")
    return schema


def _make_validator_or_raise(schema: Dict[str, Any]) -> Draft202012Validator:
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON Schema: {e.message}")
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _build_response_format(schema: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "schema": schema,
            "strict": True,
        },
    }


def _image_content(*, image_base64: str, mime_type: str) -> Dict[str, Any]:
    # vLLM/OpenAI-style: {"type":"image_url","image_url":{"url":"data:..."}}
    url = f"data:{mime_type};base64,{image_base64}"
    return {"type": "image_url", "image_url": {"url": url}}


def _build_messages(*, prompt_text: str, image_base64: str, mime_type: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                _image_content(image_base64=image_base64, mime_type=mime_type),
            ],
        }
    ]


def _estimate_prompt_tokens(text: str) -> int:
    # rough estimate to decide how aggressively to resize
    # (works better than counting bytes; safe upper bound)
    return max(1, int(len(text) / 3.5))


_TOO_LONG_PAT = re.compile(r"decoder prompt too long", re.IGNORECASE)


def _parse_decoder_prompt_too_long(err_text: str) -> Optional[int]:
    # Example errors differ by backend; try to extract max tokens if present.
    m = re.search(r"max\\s*(\\d+)", err_text, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _compute_retry_scale(*, attempt: int, max_attempts: int) -> float:
    # progressively shrink
    if max_attempts <= 1:
        return 1.0
    # 1.0 -> 0.75 -> 0.6 -> 0.5 ...
    return max(0.25, 1.0 - (attempt - 1) * (0.5 / max(1, max_attempts - 1)))


def _resize_base64_image_by_scale(image_base64: str, *, scale: float) -> str:
    if Image is None:
        raise RuntimeError("Pillow is not installed; cannot resize images.")
    raw = base64.b64decode(image_base64)
    with Image.open(BytesIO(raw)) as im:
        w, h = im.size
        new_w = max(64, int(w * math.sqrt(scale)))
        new_h = max(64, int(h * math.sqrt(scale)))
        im2 = im.resize((new_w, new_h))
        out = BytesIO()
        im2.save(out, format="JPEG", quality=85)
        return base64.b64encode(out.getvalue()).decode("ascii")


# -----------------------
# vLLM call with retries
# -----------------------
_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}
_RETRY_EXC = (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError, httpx.ReadError)


def _retry_after_from_headers(headers: Dict[str, Any]) -> Optional[float]:
    ra = headers.get("Retry-After") if headers else None
    if ra is None:
        return None
    try:
        return float(str(ra).strip())
    except Exception:
        return None


def _retry_delay_s(*, attempt: int, base: float = 0.8, cap: float = 10.0) -> float:
    # exp backoff + jitter
    raw = min(cap, base * (2 ** max(0, attempt - 1)))
    return raw * (0.75 + random.random() * 0.5)


async def _vllm_call_with_retries(
    *,
    vllm_client: VLLMClient,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    response_format: Dict[str, Any],
) -> Dict[str, Any]:
    max_attempts = int(getattr(settings, "VLLM_RETRY_MAX_ATTEMPTS", 3) or 3)

    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await vllm_client.chat_completions(
                model=MODEL_ID,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        except VLLMHTTPError as e:
            last_err = e
            retry_after = _retry_after_from_headers(getattr(e, "headers", {}) or {})
            if e.status_code in _RETRY_STATUS and attempt < max_attempts:
                await asyncio.sleep(retry_after if retry_after is not None else _retry_delay_s(attempt=attempt))
                continue
            raise
        except _RETRY_EXC as e:
            last_err = e
            if attempt < max_attempts:
                await asyncio.sleep(_retry_delay_s(attempt=attempt))
                continue
            raise

    if last_err:
        raise last_err
    raise RuntimeError("Unexpected vLLM retry loop state")


# -----------------------
# Inputs persistence for batch rerun / auditability
# -----------------------
_INPUTS_DIR = ARTIFACTS_DIR / "inputs"


def _sanitize_ext_from_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return ""
    # keep only last extension and sanitize it
    ext = Path(name).suffix.lower()
    if not ext:
        return ""
    if len(ext) > 10:
        ext = ext[:10]
    if not re.fullmatch(r"\\.[a-z0-9]+", ext):
        return ""
    return ext


def _inputs_rel_for_hash(*, sha256_hex: str, name_hint: str) -> str:
    ext = _sanitize_ext_from_name(name_hint)
    p = Path("inputs") / sha256_hex[:2] / sha256_hex[2:4] / (sha256_hex + ext)
    return str(p).replace("\\", "/")


def _persist_input_bytes(*, data: bytes, name_hint: str) -> str:
    h = hashlib.sha256(data).hexdigest()
    rel = _inputs_rel_for_hash(sha256_hex=h, name_hint=name_hint)
    if s3_enabled():
        key = s3_key(rel)
        s3_put_bytes(key=key, data=data, content_type=_guess_mime_type(name_hint) or "application/octet-stream")
        return rel
    abs_p = _INPUTS_DIR / Path(rel).relative_to("inputs")
    abs_p.parent.mkdir(parents=True, exist_ok=True)
    abs_p.write_bytes(data)
    return rel


def _load_input_bytes_by_rel(rel: str) -> bytes:
    rel_norm = str(rel).replace("\\", "/").lstrip("/")
    if not rel_norm.startswith("inputs/"):
        # allow caller to pass just the tail
        rel_norm = "inputs/" + rel_norm.lstrip("/")
    if s3_enabled():
        key = s3_key(rel_norm)
        return s3_get_bytes(key)
    abs_p = ARTIFACTS_DIR / rel_norm
    if not abs_p.exists():
        raise HTTPException(status_code=404, detail=f"Input ref not found: {rel}")
    return abs_p.read_bytes()

def persist_input_bytes(*, data: bytes, name_hint: str) -> str:
    """Persist bytes into artifacts inputs/.. and return input_ref (relative path).

    This is a small public wrapper used by async endpoints to avoid storing base64 in JobsStore.
    """
    return _persist_input_bytes(data=data, name_hint=name_hint)


async def persist_upload_files(files: List[UploadFile]) -> List[Dict[str, Any]]:
    """Persist uploaded files and return list of dicts: {file, input_ref, mime_type}."""
    out: List[Dict[str, Any]] = []
    for uf in files:
        data = await uf.read()
        if not data:
            continue
        mime = (uf.content_type or _guess_mime_type(uf.filename or "file")).split(";")[0].strip() or "application/octet-stream"
        ref = await asyncio.to_thread(_persist_input_bytes, data=data, name_hint=uf.filename or "file")
        out.append({"file": uf.filename or "file", "input_ref": ref, "mime_type": mime})
    return out



# -----------------------
# Extract (single)
# -----------------------
def _apply_debug_overrides(
    principal: ApiPrincipal,
    *,
    base_prompt_text: str,
    base_schema_json: Dict[str, Any],
    debug: Optional[DebugOptions],
) -> Tuple[str, float, int, Dict[str, Any]]:
    prompt_text = base_prompt_text
    schema_json: Any = base_schema_json
    temperature = float(getattr(settings, "VLLM_TEMPERATURE", 0.0) or 0.0)
    max_tokens = _clamp_max_tokens(int(getattr(settings, "VLLM_MAX_TOKENS", 1024) or 1024))

    if not debug:
        return prompt_text, temperature, max_tokens, _normalize_schema(schema_json)

    _debug_allowed(principal)

    if debug.prompt_override is not None:
        prompt_text = debug.prompt_override
    if debug.temperature is not None:
        temperature = float(debug.temperature)
    if debug.max_tokens is not None:
        max_tokens = _clamp_max_tokens(int(debug.max_tokens))

    if debug.schema_override is not None:
        schema_json = debug.schema_override
    if debug.schema_override_json is not None:
        schema_json = debug.schema_override_json

    schema_json = _normalize_schema(schema_json)
    return prompt_text, temperature, max_tokens, schema_json


async def extract(
    req: ExtractRequest,
    *,
    principal: ApiPrincipal,
    vllm_client: VLLMClient,
) -> ExtractResponse:
    task = _get_task(req.task_id)

    request_id = make_request_id(req.request_id)
    _validate_request_id_for_fs(request_id)

    if req.image_url:
        _debug_allowed(principal)

    if req.image_base64 and req.image_ref:
        raise HTTPException(status_code=400, detail="Provide only one of: image_base64, image_ref, image_url")

    prompt_text, temperature, max_tokens, schema_json = _apply_debug_overrides(
        principal,
        base_prompt_text=task["prompt_value"],
        base_schema_json=task["schema_value"],
        debug=req.debug,
    )

    prompt_sha = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    schema_sha = hashlib.sha256(json.dumps(schema_json, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    # Resolve image input
    mime_type = (req.mime_type or "image/jpeg").strip() or "image/jpeg"

    if req.image_ref:
        img_bytes = await asyncio.to_thread(_load_input_bytes_by_rel, req.image_ref)
        req_image_base64 = base64.b64encode(img_bytes).decode("ascii")
    elif req.image_base64:
        req_image_base64 = req.image_base64
    elif req.image_url:
        # vLLM accepts remote URLs in image_url, but we intentionally restrict to debug only.
        # Here we fetch and embed to make downstream deterministic.
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(req.image_url)
            r.raise_for_status()
            mime_type = (r.headers.get("Content-Type") or mime_type).split(";")[0].strip() or mime_type
            req_image_base64 = base64.b64encode(r.content).decode("ascii")
    else:
        raise HTTPException(status_code=400, detail="One of image_base64, image_ref, image_url is required.")

    messages = _build_messages(prompt_text=prompt_text, image_base64=req_image_base64, mime_type=mime_type)
    response_format = _build_response_format(schema_json)
    validator = _make_validator_or_raise(schema_json)

    created_at = _now_utc_s()
    error_history: List[Dict[str, Any]] = []

    artifact_base: Dict[str, Any] = {
        "request_id": request_id,
        "created_at": created_at,
        "task_id": req.task_id,
        "auth": {"key_id": principal.key_id, "role": principal.role, "scopes": sorted(list(principal.scopes))},
        "input": _sanitize_input_for_artifact(req),
        "model_id": MODEL_ID,
        "prompt_sha256": prompt_sha,
        "schema_sha256": schema_sha,
    }

    attempt = 1
    max_attempts = int(getattr(settings, "VLLM_CONTEXT_MAX_ATTEMPTS", 2) or 2)
    resize_scale = 1.0
    current_b64 = req_image_base64

    while True:
        try:
            _log_event(
                logging.INFO,
                "vllm_call_start",
                request_id=request_id,
                attempt=attempt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if getattr(settings, "MOCK_VLLM", False):
                resp = _mock_chat_completions(schema_json)
            else:
                resp = await _vllm_call_with_retries(
                    vllm_client=vllm_client,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )

            content = resp["choices"][0]["message"]["content"]
            raw = content if _raw_allowed(principal) else None

            try:
                parsed = json.loads(content) if isinstance(content, str) else content
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Model returned non-JSON content: {e}")

            errors = sorted(validator.iter_errors(parsed), key=lambda e: str(getattr(e, "path", "")))
            if errors:
                schema_errors = [{"path": list(e.path), "message": e.message} for e in errors[:50]]
                raise HTTPException(status_code=502, detail={"code": "schema_validation_failed", "errors": schema_errors})

            artifact = {
                **artifact_base,
                "raw": raw,
                "parsed": parsed,
                "schema_valid": True,
                "schema_errors": None,
                "error": None,
                "error_history": error_history or None,
                "meta": {
                    "attempts": attempt,
                    "resize_scale": resize_scale,
                    "image_tokens_est": _estimate_prompt_tokens(prompt_text),
                },
            }

            artifact_path = await asyncio.to_thread(save_artifact, request_id, artifact, principal.key_id)

            return ExtractResponse(
                request_id=request_id,
                created_at=created_at,
                task_id=req.task_id,
                artifact_path=str(artifact_path),
                model_id=MODEL_ID,
                prompt_sha256=prompt_sha,
                schema_sha256=schema_sha,
                raw=raw,
                parsed=parsed,
                schema_valid=True,
                schema_errors=None,
                error=None,
                error_history=error_history or None,
                meta=artifact["meta"],
            )

        except HTTPException as e:
            # context too long -> try resize
            detail_s = str(e.detail) if not isinstance(e.detail, str) else e.detail
            is_too_long = (e.status_code in {400, 413, 422, 500, 502}) and (
                _TOO_LONG_PAT.search(detail_s) is not None
            )

            error_history.append({"attempt": attempt, "http_status": e.status_code, "detail": e.detail})

            if is_too_long and attempt < max_attempts and Image is not None:
                attempt += 1
                resize_scale *= _compute_retry_scale(attempt=attempt, max_attempts=max_attempts)
                try:
                    current_b64 = await asyncio.to_thread(_resize_base64_image_by_scale, current_b64, scale=resize_scale)
                    messages = _build_messages(prompt_text=prompt_text, image_base64=current_b64, mime_type=mime_type)
                    continue
                except Exception as resize_e:
                    error_history.append({"attempt": attempt, "resize_error": str(resize_e)})
                    # fallthrough to persist error
            # Persist error artifact
            artifact = {
                **artifact_base,
                "raw": None,
                "parsed": None,
                "schema_valid": False,
                "schema_errors": None,
                "error": _mk_err("extract_failed", "Extraction failed", detail={"status": e.status_code, "detail": e.detail}),
                "error_history": error_history or None,
                "meta": {"attempts": attempt, "resize_scale": resize_scale},
            }
            artifact_path = await asyncio.to_thread(save_artifact, request_id, artifact, principal.key_id)
            return ExtractResponse(
                request_id=request_id,
                created_at=created_at,
                task_id=req.task_id,
                artifact_path=str(artifact_path),
                model_id=MODEL_ID,
                prompt_sha256=prompt_sha,
                schema_sha256=schema_sha,
                raw=None,
                parsed=None,
                schema_valid=False,
                schema_errors=None,
                error=artifact["error"],
                error_history=error_history or None,
                meta=artifact["meta"],
            )
        except Exception as e:
            error_history.append({"attempt": attempt, "error": str(e)})
            artifact = {
                **artifact_base,
                "raw": None,
                "parsed": None,
                "schema_valid": False,
                "schema_errors": None,
                "error": _mk_err("extract_failed", "Unhandled extraction error", detail={"error": str(e)}),
                "error_history": error_history or None,
                "meta": {"attempts": attempt, "resize_scale": resize_scale},
            }
            artifact_path = await asyncio.to_thread(save_artifact, request_id, artifact, principal.key_id)
            return ExtractResponse(
                request_id=request_id,
                created_at=created_at,
                task_id=req.task_id,
                artifact_path=str(artifact_path),
                model_id=MODEL_ID,
                prompt_sha256=prompt_sha,
                schema_sha256=schema_sha,
                raw=None,
                parsed=None,
                schema_valid=False,
                schema_errors=None,
                error=artifact["error"],
                error_history=error_history or None,
                meta=artifact["meta"],
            )


def _mock_chat_completions(schema: Dict[str, Any]) -> Dict[str, Any]:
    # Deterministic tiny mock (for CI / local w/o GPU)
    example = schema.get("examples", [{}])
    if isinstance(example, list) and example:
        payload = example[0]
    else:
        payload = {}
    return {"choices": [{"message": {"content": json.dumps(payload)}}]}


# -----------------------
# Batch extract core
# -----------------------
def _default_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime()) + "_" + uuid.uuid4().hex[:8]


def _make_batch_request_id(task_id: str, run_id: str, image_file: str) -> str:
    base = Path(image_file).name
    key = f"{task_id}|{run_id}|{base}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()[:32]


def _guess_mime_type(name: str) -> str:
    mt, _ = mimetypes.guess_type(name)
    return mt or "application/octet-stream"


def _list_image_files(root: Path, pattern: str) -> List[str]:
    if "**" in pattern:
        files = sorted([str(p) for p in root.glob(pattern) if p.is_file()])
    else:
        files = sorted([str(p) for p in root.rglob(pattern) if p.is_file()])
    # basic image-ish filter
    out = []
    for f in files:
        ext = Path(f).suffix.lower()
        if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
            out.append(f)
    return out


def _try_load_error_history_for_request(request_id: str) -> Optional[List[Dict[str, Any]]]:
    art_ref = find_artifact_path(request_id)
    if not art_ref:
        return None
    try:
        obj = read_artifact_json(art_ref)
        eh = obj.get("error_history")
        return eh if isinstance(eh, list) else None
    except Exception:
        return None


@dataclass
class _BatchInput:
    file: str
    bytes: Optional[bytes] = None
    mime_type: Optional[str] = None
    input_ref: Optional[str] = None


async def _run_eval_batch_vs_gt(
    *,
    principal: ApiPrincipal,
    batch_path: Union[str, Path],
    run_id: str,
    gt_path: str,
    gt_image_key: str,
    gt_record_key: Optional[str],
    include_worst: bool,
    limit: Optional[int],
) -> Dict[str, Any]:
    _debug_allowed(principal)

    # validate gt_path is a local file under allowed roots
    gt_p = _validate_under_any_root(Path(gt_path), roots=[DATA_ROOT, ARTIFACTS_DIR])

    eval_req = EvalBatchVsGTRequest(
        run_id=run_id,
        batch_date=artifact_date_from_path(batch_path),
        gt_path=str(gt_p),
        gt_image_key=gt_image_key,
        gt_record_key=gt_record_key,
        include_worst=include_worst,
        limit=limit,
    )

    eval_resp = await eval_batch_vs_gt(eval_req)
    eval_dict = eval_resp.model_dump()

    # attach eval to batch artifact (supports local + S3)
    try:
        batch_obj = read_artifact_json(batch_path)
        if isinstance(batch_obj, dict):
            batch_obj["eval"] = eval_dict
            if isinstance(batch_path, Path):
                batch_path.write_text(json.dumps(batch_obj, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                s3_put_json(batch_path, batch_obj)  # batch_path is S3 key ref
    except Exception as e:
        _log_event(logging.WARNING, "batch_eval_attach_failed", run_id=run_id, error=str(e))

    return eval_dict


async def _run_batch_extract_core(
    *,
    task_id: str,
    run_id: str,
    inputs: List[_BatchInput],
    concurrency: int,
    principal: ApiPrincipal,
    vllm_client: VLLMClient,
    gt_path: Optional[str] = None,
    gt_image_key: str = "image",
    gt_record_key: Optional[str] = None,
    include_worst: bool = False,
    limit: Optional[int] = None,
) -> BatchExtractResponse:
    # Optional limit for huge batches
    if limit is not None:
        inputs = inputs[: int(limit)]

    sem = asyncio.Semaphore(int(concurrency))
    items: List[BatchItemResult] = []

    ok_count = 0
    err_count = 0

    async def one(inp: _BatchInput) -> None:
        nonlocal ok_count, err_count
        async with sem:
            request_id = _make_batch_request_id(task_id, run_id, inp.file)
            token = _set_request_id(request_id)
            try:
                if inp.bytes is None and inp.input_ref:
                    img_bytes = await asyncio.to_thread(_load_input_bytes_by_rel, inp.input_ref)
                    b64 = base64.b64encode(img_bytes).decode("ascii")
                    mime = inp.mime_type or "application/octet-stream"
                else:
                    img_bytes = inp.bytes or b""
                    b64 = base64.b64encode(img_bytes).decode("ascii")
                    mime = inp.mime_type or _guess_mime_type(inp.file)

                sub_req = ExtractRequest(task_id=task_id, request_id=request_id, image_base64=b64, mime_type=mime)
                resp = await extract(sub_req, principal=principal, vllm_client=vllm_client)
                items.append(
                    BatchItemResult(
                        file=inp.file,
                        request_id=request_id,
                        input_ref=inp.input_ref,
                        artifact_rel=resp.artifact_path,
                        ok=resp.error is None,
                        error=resp.error,
                        error_history=resp.error_history,
                    )
                )
                if resp.error is None:
                    ok_count += 1
                else:
                    err_count += 1
            except Exception as e:
                err_count += 1
                eh = await asyncio.to_thread(_try_load_error_history_for_request, request_id)
                art_ref = find_artifact_path(request_id)
                items.append(
                    BatchItemResult(
                        file=inp.file,
                        request_id=request_id,
                        input_ref=inp.input_ref,
                        artifact_rel=str(art_ref) if art_ref else None,
                        ok=False,
                        error=_mk_err("batch_item_failed", str(e)),
                        error_history=eh,
                    )
                )
            finally:
                _reset_request_id(token)

    await asyncio.gather(*(one(i) for i in inputs))

    created_at = _now_utc_s()
    batch_payload: Dict[str, Any] = {
        "run_id": run_id,
        "created_at": created_at,
        "task_id": task_id,
        "images_dir": None,
        "glob": None,
        "concurrency": concurrency,
        "item_count": len(items),
        "ok_count": ok_count,
        "error_count": err_count,
        "items": [it.model_dump() for it in items],
        "auth": {"key_id": principal.key_id, "role": principal.role, "scopes": sorted(list(principal.scopes))},
    }

    batch_path = await asyncio.to_thread(save_batch_artifact, run_id, batch_payload, principal.key_id)

    eval_info: Optional[Dict[str, Any]] = None
    if gt_path:
        try:
            eval_info = await _run_eval_batch_vs_gt(
                principal=principal,
                batch_path=batch_path,
                run_id=run_id,
                gt_path=gt_path,
                gt_image_key=gt_image_key,
                gt_record_key=gt_record_key,
                include_worst=include_worst,
                limit=limit,
            )
        except Exception as e:
            _log_event(logging.WARNING, "batch_eval_failed", run_id=run_id, error=str(e))

    return BatchExtractResponse(
        run_id=run_id,
        created_at=created_at,
        task_id=task_id,
        images_dir="",
        glob="",
        concurrency=concurrency,
        item_count=len(items),
        ok_count=ok_count,
        error_count=err_count,
        artifact_path=str(batch_path),
        items=items,
        eval=eval_info,
    )


async def batch_extract(
    req: BatchExtractRequest,
    *,
    principal: ApiPrincipal,
    vllm_client: VLLMClient,
) -> BatchExtractResponse:
    _get_task(req.task_id)  # validate

    images_dir = _validate_under_data_root(Path(req.images_dir))
    files = _list_image_files(images_dir, req.glob)
    if not files:
        raise HTTPException(status_code=404, detail="No image files found in images_dir with given glob.")

    run_id = req.run_id or _default_run_id()

    inputs: List[_BatchInput] = []
    for f in files:
        b = await asyncio.to_thread(Path(f).read_bytes)
        inputs.append(_BatchInput(file=f, bytes=b, mime_type=_guess_mime_type(f)))

    resp = await _run_batch_extract_core(
        task_id=req.task_id,
        run_id=run_id,
        inputs=inputs,
        concurrency=req.concurrency,
        principal=principal,
        vllm_client=vllm_client,
        gt_path=req.gt_path,
        gt_image_key=req.gt_image_key,
        gt_record_key=req.gt_record_key,
        include_worst=req.include_worst,
        limit=req.limit,
    )

    # fill response fields
    resp.images_dir = str(images_dir)
    resp.glob = req.glob
    return resp


async def batch_extract_inputs(
    req: BatchExtractInputsRequest,
    *,
    principal: ApiPrincipal,
    vllm_client: VLLMClient,
) -> BatchExtractResponse:
    _get_task(req.task_id)  # validate
    run_id2 = req.run_id or _default_run_id()

    inputs: List[_BatchInput] = []
    for it in req.inputs:
        # it can be dict-like if called from job executor
        if isinstance(it, dict):
            file = str(it.get("file") or "file")
            input_ref = str(it.get("input_ref") or "")
            mime_type = it.get("mime_type")
        else:
            file = str(it.file or "file")
            input_ref = str(it.input_ref or "")
            mime_type = it.mime_type

        if not input_ref:
            continue
        inputs.append(_BatchInput(file=file, bytes=None, mime_type=mime_type, input_ref=input_ref))

    if not inputs:
        raise HTTPException(status_code=400, detail="No inputs provided (input_ref list is empty).")

    resp = await _run_batch_extract_core(
        task_id=req.task_id,
        run_id=run_id2,
        inputs=inputs,
        concurrency=req.concurrency,
        principal=principal,
        vllm_client=vllm_client,
        gt_path=req.gt_path,
        gt_image_key=req.gt_image_key,
        gt_record_key=req.gt_record_key,
        include_worst=req.include_worst,
        limit=req.limit,
    )
    resp.images_dir = "(upload_refs)"
    resp.glob = "(upload_refs)"
    return resp


async def batch_extract_upload(
    *,
    task_id: str,
    persist_inputs: bool,
    concurrency: int,
    run_id: Optional[str],
    files: List[UploadFile],
    gt_path: Optional[str],
    gt_image_key: str,
    gt_record_key: Optional[str],
    include_worst: bool,
    limit: Optional[int],
    principal: ApiPrincipal,
    vllm_client: VLLMClient,
) -> BatchExtractResponse:
    _get_task(task_id)  # validate

    run_id2 = run_id or _default_run_id()

    inputs: List[_BatchInput] = []
    for uf in files:
        data = await uf.read()
        if not data:
            continue
        mime = (uf.content_type or _guess_mime_type(uf.filename or "file")).split(";")[0].strip()
        ref = None
        if persist_inputs:
            ref = await asyncio.to_thread(_persist_input_bytes, data=data, name_hint=uf.filename or "file")
        inputs.append(_BatchInput(file=uf.filename or "file", bytes=data, mime_type=mime, input_ref=ref))

    resp = await _run_batch_extract_core(
        task_id=task_id,
        run_id=run_id2,
        inputs=inputs,
        concurrency=concurrency,
        principal=principal,
        vllm_client=vllm_client,
        gt_path=gt_path,
        gt_image_key=gt_image_key,
        gt_record_key=gt_record_key,
        include_worst=include_worst,
        limit=limit,
    )
    resp.images_dir = "(upload)"
    resp.glob = "(upload)"
    return resp


async def batch_extract_rerun(
    req: BatchRerunRequest,
    *,
    principal: ApiPrincipal,
    vllm_client: VLLMClient,
) -> BatchExtractResponse:
    _get_task(req.task_id)  # validate

    src_path = find_batch_artifact_path(req.run_id, date=req.batch_date, max_days=req.max_days)
    if src_path is None:
        raise HTTPException(status_code=404, detail="Source batch artifact not found.")

    src_obj = read_artifact_json(src_path)
    if not isinstance(src_obj, dict):
        raise HTTPException(status_code=500, detail="Invalid source batch JSON")

    src_items = src_obj.get("items") if isinstance(src_obj.get("items"), list) else []
    if req.limit is not None:
        src_items = src_items[: int(req.limit)]

    inputs: List[_BatchInput] = []
    for it in src_items:
        if not isinstance(it, dict):
            continue
        f = str(it.get("file") or "")
        inp_ref = it.get("input_ref") or it.get("inputRel") or it.get("input_rel")
        if isinstance(inp_ref, str) and inp_ref:
            inputs.append(_BatchInput(file=f or "file", bytes=None, mime_type=None, input_ref=inp_ref))
            continue
        # fallback: re-read image from disk (if it was server-side run and file path still exists)
        try:
            p = Path(f)
            if p.exists():
                b = p.read_bytes()
                inputs.append(_BatchInput(file=f, bytes=b, mime_type=_guess_mime_type(f)))
        except Exception:
            continue

    new_run_id = req.new_run_id or _default_run_id()

    resp = await _run_batch_extract_core(
        task_id=req.task_id,
        run_id=new_run_id,
        inputs=inputs,
        concurrency=req.concurrency,
        principal=principal,
        vllm_client=vllm_client,
        gt_path=req.gt_path,
        gt_image_key=req.gt_image_key,
        gt_record_key=req.gt_record_key,
        include_worst=req.include_worst,
        limit=req.limit,
    )
    resp.images_dir = "(rerun)"
    resp.glob = "(rerun)"
    return resp
