"""
S3/R2 service layer (boto3 client + retries + basic operations).

This module is intentionally generic and reusable across the project (not only artifacts):
- persisted inputs
- future datasets / caches
- any debug/admin operations

NOTE: boto3 is synchronous. In async endpoints, wrap calls with:
    await asyncio.to_thread(...)
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import boto3
import botocore
from botocore.config import Config as BotoConfig
from fastapi import HTTPException

from app.settings import settings


# --- S3 settings (R2 via S3 API) ---
_S3_BUCKET = (settings.S3_BUCKET or "").strip()
_S3_ENDPOINT_URL = settings.S3_ENDPOINT_URL
_S3_REGION = (settings.S3_REGION or "auto").strip()
_S3_PREFIX = (settings.S3_PREFIX or "ocrlty").strip().strip("/")
_S3_ALLOW_OVERWRITE = bool(settings.S3_ALLOW_OVERWRITE)
_S3_FORCE_PATH_STYLE = bool(getattr(settings, "S3_FORCE_PATH_STYLE", False))

_s3_client_singleton = None


def s3_enabled() -> bool:
    return bool(_S3_BUCKET)


def s3_client():
    global _s3_client_singleton
    if _s3_client_singleton is not None:
        return _s3_client_singleton

    access_key = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("S3_ACCESS_KEY_ID") or ""
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("S3_SECRET_ACCESS_KEY") or ""

    kwargs: Dict[str, Any] = {
        "service_name": "s3",
        "region_name": _S3_REGION or None,
        "endpoint_url": _S3_ENDPOINT_URL,
        "config": BotoConfig(
            retries={"max_attempts": 8, "mode": "standard"},
            s3={"addressing_style": ("path" if _S3_FORCE_PATH_STYLE else "virtual")},
        ),
    }
    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key

    _s3_client_singleton = boto3.client(**kwargs)
    return _s3_client_singleton


def _s3_key(rel: str) -> str:
    rel = (rel or "").lstrip("/")
    if _S3_PREFIX:
        return f"{_S3_PREFIX}/{rel}" if rel else _S3_PREFIX
    return rel


def _s3_is_retryable_client_error(e: Exception) -> bool:
    if not isinstance(e, botocore.exceptions.ClientError):
        return False
    code = (e.response or {}).get("Error", {}).get("Code", "") or ""
    status = int((e.response or {}).get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 0)
    if status >= 500:
        return True
    if code in {"RequestTimeout", "Throttling", "ThrottlingException", "SlowDown", "InternalError"}:
        return True
    return False


def s3_call_with_retries(fn, *, op: str, max_attempts: int = 5):
    delay_s = 0.25
    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            retryable = _s3_is_retryable_client_error(e) or isinstance(
                e,
                (
                    botocore.exceptions.EndpointConnectionError,
                    botocore.exceptions.ConnectionClosedError,
                    botocore.exceptions.ReadTimeoutError,
                    botocore.exceptions.ConnectTimeoutError,
                ),
            )
            if not retryable or attempt == max_attempts:
                raise
            time.sleep(delay_s)
            delay_s = min(delay_s * 2.0, 2.0)
    raise last_err  # pragma: no cover


def _s3_put_json(key: str, payload: Dict[str, Any]) -> None:
    body = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    def _do():
        kwargs: Dict[str, Any] = {
            "Bucket": _S3_BUCKET,
            "Key": key,
            "Body": body,
            "ContentType": "application/json; charset=utf-8",
            "CacheControl": "no-store",
        }
        if not _S3_ALLOW_OVERWRITE:
            kwargs["IfNoneMatch"] = "*"
        return s3_client().put_object(**kwargs)

    try:
        s3_call_with_retries(_do, op="put_object")
    except botocore.exceptions.ClientError as e:
        code = (e.response or {}).get("Error", {}).get("Code", "") or ""
        if code in {"PreconditionFailed"} and not _S3_ALLOW_OVERWRITE:
            return
        raise


def _s3_get_text(key: str) -> str:
    def _do():
        return s3_client().get_object(Bucket=_S3_BUCKET, Key=key)

    try:
        obj = s3_call_with_retries(_do, op="get_object")
    except botocore.exceptions.ClientError as e:
        code = (e.response or {}).get("Error", {}).get("Code", "") or ""
        if code in {"NoSuchKey", "404"}:
            raise HTTPException(status_code=404, detail="Artifact not found.")
        raise HTTPException(status_code=502, detail=f"S3 get_object failed: {code}")

    body = obj["Body"].read()
    return body.decode("utf-8", errors="replace")


def _s3_get_json(key: str) -> Dict[str, Any]:
    try:
        return json.loads(_s3_get_text(key))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse artifact JSON: {e}")


def _s3_exists(key: str) -> bool:
    def _do():
        return s3_client().head_object(Bucket=_S3_BUCKET, Key=key)

    try:
        s3_call_with_retries(_do, op="head_object", max_attempts=3)
        return True
    except botocore.exceptions.ClientError as e:
        code = (e.response or {}).get("Error", {}).get("Code", "") or ""
        if code in {"NoSuchKey", "404"}:
            return False
        status = int((e.response or {}).get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 0)
        if status == 404:
            return False
        raise


def _s3_list_common_prefixes(prefix: str) -> List[str]:
    out: List[str] = []
    token: Optional[str] = None
    while True:
        def _do():
            kwargs: Dict[str, Any] = {"Bucket": _S3_BUCKET, "Prefix": prefix, "Delimiter": "/"}
            if token:
                kwargs["ContinuationToken"] = token
            return s3_client().list_objects_v2(**kwargs)

        resp = s3_call_with_retries(_do, op="list_objects_v2")
        for cp in resp.get("CommonPrefixes") or []:
            pfx = cp.get("Prefix") or ""
            if pfx:
                out.append(pfx)
        token = resp.get("NextContinuationToken")
        if not token:
            break
    return out


def _s3_list_objects(prefix: str, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    token: Optional[str] = None
    while True:
        def _do():
            kwargs: Dict[str, Any] = {"Bucket": _S3_BUCKET, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            return s3_client().list_objects_v2(**kwargs)

        resp = s3_call_with_retries(_do, op="list_objects_v2")
        for it in resp.get("Contents") or []:
            out.append(it)
            if limit is not None and len(out) >= limit:
                return out
        token = resp.get("NextContinuationToken")
        if not token:
            break
    return out




# ---------------------------
# Public API (stable names)
# ---------------------------

def s3_key(rel: str) -> str:
    return _s3_key(rel)

def s3_put_json(key: str, payload: Dict[str, Any]) -> None:
    return _s3_put_json(key, payload)

def s3_put_bytes(*, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    body = data or b""

    def _do():
        kwargs: Dict[str, Any] = {
            "Bucket": _S3_BUCKET,
            "Key": key,
            "Body": body,
            "ContentType": content_type or "application/octet-stream",
            "CacheControl": "no-store",
        }
        if not _S3_ALLOW_OVERWRITE:
            kwargs["IfNoneMatch"] = "*"
        return s3_client().put_object(**kwargs)

    try:
        s3_call_with_retries(_do, op="put_object")
    except botocore.exceptions.ClientError as e:
        code = (e.response or {}).get("Error", {}).get("Code", "") or ""
        if code in {"PreconditionFailed"} and not _S3_ALLOW_OVERWRITE:
            return
        raise

def s3_get_text(key: str) -> str:
    return _s3_get_text(key)

def s3_get_json(key: str) -> Dict[str, Any]:
    return _s3_get_json(key)

def s3_get_bytes(key: str) -> bytes:
    def _do():
        return s3_client().get_object(Bucket=_S3_BUCKET, Key=key)
    try:
        obj = s3_call_with_retries(_do, op="get_object")
    except botocore.exceptions.ClientError as e:
        code = (e.response or {}).get("Error", {}).get("Code", "") or ""
        if code in {"NoSuchKey", "404"}:
            raise HTTPException(status_code=404, detail="Object not found.")
        raise HTTPException(status_code=502, detail=f"S3 get_object failed: {code}")
    return obj["Body"].read()

def s3_exists(key: str) -> bool:
    return _s3_exists(key)

def s3_list_common_prefixes(prefix: str) -> List[str]:
    return _s3_list_common_prefixes(prefix)

def s3_list_objects(prefix: str, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    return _s3_list_objects(prefix, limit=limit)
