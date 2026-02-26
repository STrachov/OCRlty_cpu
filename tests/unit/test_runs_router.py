import pytest
from fastapi import HTTPException

from app.auth_store import ApiPrincipal
from app.routers import runs


def _principal(*, role: str = "client", key_id: str = "k1") -> ApiPrincipal:
    return ApiPrincipal(api_key_id=1, key_id=key_id, role=role, scopes={"extract:run"})


def test_decode_cursor_invalid():
    with pytest.raises(HTTPException):
        runs._decode_cursor("bad-cursor")


@pytest.mark.asyncio
async def test_list_runs_uses_cursor_and_returns_next(monkeypatch):
    monkeypatch.setattr(
        runs,
        "list_batch_artifacts",
        lambda **_: (
            [
                {
                    "artifact_id": "run2",
                    "full_ref": "/x/run2.json",
                    "rel_ref": "batches/2026-02-26/run2.json",
                },
            ],
            ("2026-02-26T10:00:00Z", "run2"),
        ),
    )
    monkeypatch.setattr(
        runs,
        "read_artifact_json",
        lambda _: {
            "created_at": "2026-02-26T10:00:00Z",
            "task_id": "receipt_fields_v1",
            "item_count": 5,
            "ok_count": 4,
            "error_count": 1,
        },
    )

    resp = await runs.list_runs(principal=_principal(), limit=10, cursor=None)
    assert len(resp.items) == 1
    assert resp.items[0].run_id == "run2"
    assert resp.items[0].artifact_rel == "batches/2026-02-26/run2.json"
    assert resp.next_cursor == "2026-02-26T10:00:00Z|run2"


@pytest.mark.asyncio
async def test_get_run_forbidden_for_non_owner(monkeypatch):
    monkeypatch.setattr(runs, "find_batch_artifact_path", lambda *_args, **_kwargs: "/x/run1.json")
    monkeypatch.setattr(runs, "read_artifact_json", lambda _: {"auth": {"key_id": "owner-1"}})

    with pytest.raises(HTTPException) as exc:
        await runs.get_run("run1", principal=_principal(role="client", key_id="other"))
    assert exc.value.status_code == 403
