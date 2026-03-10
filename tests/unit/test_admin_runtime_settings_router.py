from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.auth_store import ApiPrincipal
from app.routers import admin_runtime_settings


def _principal(*, role: str) -> ApiPrincipal:
    return ApiPrincipal(api_key_id=1, key_id="k1", role=role, scopes={"extract:run"})


def _request_with_store(store: object):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(runtime_settings_store=store)))


@pytest.mark.asyncio
async def test_list_runtime_settings_requires_admin():
    with pytest.raises(HTTPException) as exc:
        await admin_runtime_settings.list_runtime_settings(
            request=_request_with_store(object()),
            principal=_principal(role="client"),
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_update_runtime_setting_returns_updated_value(monkeypatch):
    monkeypatch.setattr(
        admin_runtime_settings.runtime_config,
        "set_value",
        lambda **_: None,
    )
    monkeypatch.setattr(
        admin_runtime_settings.runtime_config,
        "list_effective_settings",
        lambda: [
            {
                "key": "DEBUG_MODE",
                "value": "1",
                "effective_value": True,
                "source": "db",
                "updated_at": "2026-03-10T10:00:00Z",
                "is_secret": False,
                "is_set": True,
            }
        ],
    )

    resp = await admin_runtime_settings.update_runtime_setting(
        key="DEBUG_MODE",
        body=admin_runtime_settings.RuntimeSettingUpdateRequest(value="true"),
        request=_request_with_store(object()),
        principal=_principal(role="admin"),
    )
    assert resp.key == "DEBUG_MODE"
    assert resp.effective_value is True
    assert resp.source == "db"
