import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException

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


def test_compute_retry_scale_uses_adaptive_ratio_from_error():
    scale = extract_service._compute_retry_scale(
        attempt=2,
        max_attempts=3,
        detail="The decoder prompt (length 9597) is longer than the maximum model length of 4096",
    )
    assert 0.10 < scale < 0.50


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
            messages=[],
            temperature=0.0,
            max_tokens=16,
            response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}},
        )

    assert client.chat_completions.call_count == 2
