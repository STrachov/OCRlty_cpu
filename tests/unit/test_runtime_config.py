from types import SimpleNamespace

from app.services import runtime_config
from app.services.runtime_settings_store import RuntimeSettingRecord


def test_runtime_config_db_override_and_fallback(monkeypatch):
    monkeypatch.setattr(
        runtime_config,
        "_STORE",
        SimpleNamespace(
            list=lambda: [
                RuntimeSettingRecord(
                    key="DEBUG_MODE",
                    value_text="1",
                    updated_at="2026-03-10T10:00:00Z",
                    updated_by_key_id="admin",
                ),
                RuntimeSettingRecord(
                    key="S3_PRESIGN_TTL_S",
                    value_text="123",
                    updated_at="2026-03-10T10:00:00Z",
                    updated_by_key_id="admin",
                ),
            ]
        ),
    )
    runtime_config.invalidate_cache()

    assert runtime_config.get_bool("DEBUG_MODE", False) is True
    assert runtime_config.get_int("S3_PRESIGN_TTL_S", 3600) == 123
    assert runtime_config.get_str("VLLM_MODEL", "fallback-model") == "fallback-model"


def test_runtime_config_set_value_normalizes_bool(monkeypatch):
    calls = {}

    def _upsert(**kwargs):
        calls.update(kwargs)
        return RuntimeSettingRecord(
            key=kwargs["key"],
            value_text=kwargs["value_text"],
            updated_at=kwargs["updated_at"],
            updated_by_key_id=kwargs["updated_by_key_id"],
        )

    monkeypatch.setattr(runtime_config, "_STORE", SimpleNamespace(upsert=_upsert))
    runtime_config.invalidate_cache()

    rec = runtime_config.set_value(key="DEBUG_MODE", value="true", updated_by_key_id="admin")
    assert rec.value_text == "1"
    assert calls["key"] == "DEBUG_MODE"
    assert calls["updated_by_key_id"] == "admin"
