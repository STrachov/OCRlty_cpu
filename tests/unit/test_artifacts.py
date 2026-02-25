from pathlib import Path

import pytest
from fastapi import HTTPException

from app.services import artifacts


def test_save_and_find_extract_artifact_local(monkeypatch, tmp_path):
    monkeypatch.setattr(artifacts, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(artifacts, "s3_enabled", lambda: False)

    request_id = "abc123"
    payload = {"ok": True}

    ref = artifacts.save_artifact(request_id, payload, owner_key_id="k1")
    assert isinstance(ref, Path)
    assert ref.exists()

    found = artifacts.find_artifact_path(request_id, date=artifacts._utc_day(), max_days=1)
    assert found is not None

    obj = artifacts.read_artifact_json(found)
    assert obj["ok"] is True


def test_to_artifact_rel_local(monkeypatch, tmp_path):
    monkeypatch.setattr(artifacts, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(artifacts, "s3_enabled", lambda: False)

    p = tmp_path / "extracts" / "2026-02-25" / "x.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"a":1}', encoding="utf-8")

    rel = artifacts.to_artifact_rel(p)
    assert rel == "extracts\\2026-02-25\\x.json" or rel == "extracts/2026-02-25/x.json"


def test_find_artifact_path_rejects_invalid_date(monkeypatch):
    monkeypatch.setattr(artifacts, "s3_enabled", lambda: False)

    with pytest.raises(HTTPException) as exc:
        artifacts.find_artifact_path("rid", date="20260225")

    assert exc.value.status_code == 400
