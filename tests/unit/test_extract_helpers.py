import asyncio
import base64
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException

from app.auth_store import ApiPrincipal
from app.services import extract_service
from app.vllm_client import VLLMHTTPError


def test_sanitize_ext_from_name_keeps_valid_extension():
    assert extract_service._sanitize_ext_from_name("photo.jpg") == ".jpg"


def test_sanitize_ext_from_name_rejects_invalid_extension():
    assert extract_service._sanitize_ext_from_name("photo.jp*g") == ""


def test_apply_debug_overrides_without_debug_uses_base(principal):
    prompt, temperature, max_tokens, schema = extract_service._apply_debug_overrides(
        principal,
        base_prompt_text="base-prompt",
        base_schema_json={"type": "object", "properties": {}},
        debug=None,
    )

    assert prompt == "base-prompt"
    assert isinstance(temperature, float)
    assert isinstance(max_tokens, int)
    assert schema["type"] == "object"


def test_apply_debug_overrides_requires_scope(monkeypatch, principal):
    monkeypatch.setattr(extract_service.settings, "DEBUG_MODE", True)
    monkeypatch.setattr(extract_service.settings, "AUTH_ENABLED", True)

    with pytest.raises(HTTPException) as exc:
        extract_service._apply_debug_overrides(
            principal,
            base_prompt_text="p",
            base_schema_json={"type": "object"},
            debug=extract_service.DebugOptions(prompt_override="x"),
        )

    assert exc.value.status_code == 403


def test_is_context_too_long_error_matches_vllm_message():
    assert extract_service._is_context_too_long_error(
        status_code=400,
        detail="The decoder prompt (length 9591) is longer than the maximum model length of 4096",
    )


def test_extract_context_lengths_parses_vllm_message():
    assert extract_service._extract_context_lengths(
        "{'error': {'message': 'The decoder prompt (length 9597) is longer than the maximum model length of 4096.'}}"
    ) == (9597, 4096)


def test_compute_retry_scale_uses_adaptive_ratio_from_error(monkeypatch):
    monkeypatch.setattr(extract_service.settings, "VLLM_MAX_MODEL_LEN", 8192)
    monkeypatch.setattr(extract_service.settings, "VLLM_CONTEXT_RESPONSE_RESERVE_TOKENS", 512)
    monkeypatch.setattr(extract_service.settings, "VLLM_CONTEXT_SAFETY_MARGIN_TOKENS", 256)
    scale = extract_service._compute_retry_scale(
        attempt=2,
        max_attempts=3,
        detail="The decoder prompt (length 9597) is longer than the maximum model length of 4096",
        max_tokens=1024,
    )
    assert 0.10 < scale < 0.50


def test_compute_context_prompt_budget_reserves_completion_and_margin(monkeypatch):
    monkeypatch.setattr(extract_service.settings, "VLLM_CONTEXT_RESPONSE_RESERVE_TOKENS", 512)
    monkeypatch.setattr(extract_service.settings, "VLLM_CONTEXT_SAFETY_MARGIN_TOKENS", 256)
    assert extract_service._compute_context_prompt_budget(model_len=8192, max_tokens=1024) == 6912


def test_compute_retry_max_tokens_reduces_budget_on_retry():
    assert extract_service._compute_retry_max_tokens(current_max_tokens=1024, attempt=2) < 1024
    assert extract_service._compute_retry_max_tokens(current_max_tokens=1024, attempt=2) >= 256


@pytest.mark.asyncio
async def test_vllm_call_with_retries_retries_then_success(monkeypatch):
    client = AsyncMock()
    client.chat_completions.side_effect = [
        VLLMHTTPError(503, "busy", {}),
        {"choices": [{"message": {"content": "{}"}}]},
    ]

    async def _no_sleep(_: float):
        return None

    monkeypatch.setattr(extract_service, "_retry_delay_s", lambda **_: 0.0)
    monkeypatch.setattr(extract_service.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(extract_service.settings, "VLLM_RETRY_MAX_ATTEMPTS", 3)

    out = await extract_service._vllm_call_with_retries(
        vllm_client=client,
        model_id="model-x",
        messages=[],
        temperature=0.0,
        max_tokens=16,
        response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}},
    )

    assert "choices" in out
    assert client.chat_completions.call_count == 2


@pytest.mark.asyncio
async def test_vllm_call_with_retries_raises_after_max_attempts(monkeypatch):
    client = AsyncMock()
    client.chat_completions.side_effect = httpx.ReadTimeout("timeout")

    async def _no_sleep(_: float):
        return None

    monkeypatch.setattr(extract_service, "_retry_delay_s", lambda **_: 0.0)
    monkeypatch.setattr(extract_service.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(extract_service.settings, "VLLM_RETRY_MAX_ATTEMPTS", 2)

    with pytest.raises(httpx.ReadTimeout):
        await extract_service._vllm_call_with_retries(
            vllm_client=client,
            model_id="model-x",
            messages=[],
            temperature=0.0,
            max_tokens=16,
            response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}},
        )

    assert client.chat_completions.call_count == 2


@pytest.mark.asyncio
async def test_vllm_call_with_retries_uses_passed_model_id():
    client = AsyncMock()
    client.chat_completions.return_value = {"choices": [{"message": {"content": "{}"}}]}

    await extract_service._vllm_call_with_retries(
        vllm_client=client,
        model_id="Qwen/Qwen3-VL-32B-Instruct-FP8",
        messages=[],
        temperature=0.0,
        max_tokens=16,
        response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}},
    )

    assert client.chat_completions.await_args.kwargs["model"] == "Qwen/Qwen3-VL-32B-Instruct-FP8"


@pytest.mark.asyncio
async def test_extract_persists_last_raw_on_non_json_failure(monkeypatch):
    principal = ApiPrincipal(
        api_key_id=1,
        key_id="admin-k1",
        role="admin",
        scopes={"extract:run", "debug:run", "debug:read_raw"},
    )
    captured = {}

    monkeypatch.setattr(extract_service.settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(extract_service.settings, "DEBUG_MODE", True)
    monkeypatch.setattr(
        extract_service,
        "get_str",
        lambda key, default=None: "vllm" if key == "INFERENCE_BACKEND" else default,
    )
    monkeypatch.setattr(
        extract_service,
        "_get_task",
        lambda task_id: {
            "task_id": task_id,
            "prompt_value": "return json",
            "schema_value": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
        },
    )
    monkeypatch.setattr(extract_service, "make_request_id", lambda value: value or "req-1")
    monkeypatch.setattr(extract_service, "_validate_request_id_for_fs", lambda _: None)
    monkeypatch.setattr(extract_service, "_log_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(extract_service, "to_artifact_rel", lambda ref: f"rel/{ref}")

    def _save_artifact(request_id, payload, owner_key_id=None):
        captured["request_id"] = request_id
        captured["payload"] = payload
        captured["owner_key_id"] = owner_key_id
        return "artifact-1"

    monkeypatch.setattr(extract_service, "save_artifact", _save_artifact)

    vllm_client = AsyncMock()
    vllm_client.chat_completions.return_value = {
        "choices": [{"message": {"content": '{\n  "ok": true,\n}\ninvalid'}}]
    }

    req = extract_service.ExtractRequest(
        task_id="receipt_fields_v1",
        request_id="req-1",
        image_base64=base64.b64encode(b"fake-image").decode("ascii"),
        mime_type="image/jpeg",
    )

    resp = await extract_service.extract(req=req, principal=principal, vllm_client=vllm_client)

    assert resp.error is not None
    assert resp.error["code"] == "extract_failed"
    assert captured["request_id"] == "req-1"
    assert captured["owner_key_id"] == "admin-k1"
    assert captured["payload"]["raw"] == '{\n  "ok": true,\n}\ninvalid'
    assert captured["payload"]["error"]["detail"]["status"] == 502


@pytest.mark.asyncio
async def test_run_batch_extract_core_prefers_input_ref_for_extract(monkeypatch, principal):
    captured = {}

    async def _fake_extract(req, *, principal, vllm_client):
        captured["req"] = req
        return extract_service.ExtractResponse(
            request_id=req.request_id or "req-1",
            created_at="2026-03-11T10:00:00Z",
            task_id=req.task_id,
            artifact_rel="extracts/2026-03-11/req-1.json",
            model_id="model-x",
            prompt_sha256="p",
            schema_sha256="s",
            timings_ms={"total": 1},
            raw=None,
            parsed={"ok": True},
            schema_valid=True,
            schema_errors=None,
            error=None,
            error_history=None,
            meta={},
        )

    monkeypatch.setattr(extract_service, "extract", _fake_extract)
    monkeypatch.setattr(extract_service, "save_batch_artifact", lambda run_id, payload, owner_key_id=None: "batch-ref")
    monkeypatch.setattr(extract_service, "_now_utc_s", lambda: "2026-03-11T10:00:00Z")

    resp = await extract_service._run_batch_extract_core(
        task_id="receipt_fields_v1",
        run_id="run-1",
        inputs=[
            extract_service._BatchInput(
                file="receipt.jpg",
                bytes=b"fake-image",
                mime_type="image/jpeg",
                input_ref="inputs/1/receipt.jpg",
            )
        ],
        concurrency=1,
        principal=principal,
        vllm_client=AsyncMock(),
    )

    req = captured["req"]
    assert req.image_ref == "inputs/1/receipt.jpg"
    assert req.image_base64 is None
    assert resp.items[0].input_ref == "inputs/1/receipt.jpg"


@pytest.mark.asyncio
async def test_extract_persists_non_empty_detail_for_empty_vllm_http_error(monkeypatch, principal):
    captured = {}

    monkeypatch.setattr(extract_service.settings, "DEBUG_MODE", True)
    monkeypatch.setattr(
        extract_service,
        "get_str",
        lambda key, default=None: "vllm" if key == "INFERENCE_BACKEND" else default,
    )
    monkeypatch.setattr(
        extract_service,
        "_get_task",
        lambda task_id: {
            "task_id": task_id,
            "prompt_value": "return json",
            "schema_value": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
        },
    )
    monkeypatch.setattr(extract_service, "make_request_id", lambda value: value or "req-404")
    monkeypatch.setattr(extract_service, "_validate_request_id_for_fs", lambda _: None)
    monkeypatch.setattr(extract_service, "_log_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(extract_service, "to_artifact_rel", lambda ref: f"rel/{ref}")

    def _save_artifact(request_id, payload, owner_key_id=None):
        captured["payload"] = payload
        return "artifact-404"

    monkeypatch.setattr(extract_service, "save_artifact", _save_artifact)

    vllm_client = AsyncMock()
    vllm_client.chat_completions.side_effect = VLLMHTTPError(404, "", {"x-request-id": "up-404"})

    req = extract_service.ExtractRequest(
        task_id="receipt_fields_v1",
        request_id="req-404",
        image_base64=base64.b64encode(b"fake-image").decode("ascii"),
        mime_type="image/jpeg",
    )

    resp = await extract_service.extract(req=req, principal=principal, vllm_client=vllm_client)

    assert resp.error is not None
    assert resp.error["detail"]["status"] == 404
    assert resp.error["detail"]["detail"]
    assert "up-404" in resp.error["detail"]["detail"]
    assert captured["payload"]["error_history"][0]["detail"]
