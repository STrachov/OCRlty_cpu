# app/vllm_client.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import httpx


def _trim(s: str, n: int = 4000) -> str:
    s = s or ""
    return s if len(s) <= n else (s[:n] + "…<trimmed>")

class VLLMHTTPError(RuntimeError):
    def __init__(self, status_code: int, body_text: str, headers: dict[str, str] | None = None):
        super().__init__(f"vLLM HTTP {status_code}: {body_text}")
        self.status_code = int(status_code)
        self.body_text = body_text
        self.headers = headers or {}

class VLLMClient:
    """Minimal async client for vLLM OpenAI-compatible API (reuses one httpx.AsyncClient)."""

    def __init__(self, http: httpx.AsyncClient, *, base_url: str, api_key: str = "") -> None:
        self._http = http
        self._base_url = base_url
        self._api_key = api_key or ""

    def _headers(self, request_id: Optional[str] = None) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if request_id:
            headers["X-Request-ID"] = request_id
        return headers

    async def chat_completions(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 256,
        response_format: Optional[Dict[str, Any]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        r = await self._http.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers(request_id=request_id),
            json=payload,
        )
        if r.status_code >= 400:
            try:
                body_s = _trim(str(r.json()))
            except Exception:
                body_s = _trim(r.text)
            #raise RuntimeError(f"vLLM HTTP {r.status_code}: {body_s}")
            raise VLLMHTTPError(r.status_code, body_s, dict(r.headers))
        return r.json()

    async def models(self) -> Dict[str, Any]:
        r = await self._http.get(f"{self._base_url}/models", headers=self._headers())
        if r.status_code >= 400:
            try:
                body_s = _trim(str(r.json()))
            except Exception:
                body_s = _trim(r.text)
            raise RuntimeError(f"vLLM HTTP {r.status_code}: {body_s}")
        return r.json()
