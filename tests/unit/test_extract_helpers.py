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
