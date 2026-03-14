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
async def test_list_runs_builds_eval_summary_with_fields(monkeypatch):
    monkeypatch.setattr(
        runs,
        "list_batch_artifacts",
        lambda **_: (
            [
                {
                    "artifact_id": "run-eval",
                    "full_ref": "/x/run-eval.json",
                    "rel_ref": "batches/2026-03-14/run-eval.json",
                },
            ],
            None,
        ),
    )
    monkeypatch.setattr(
        runs,
        "read_artifact_json",
        lambda _: {
            "created_at": "2026-03-14T10:00:00Z",
            "task_id": "receipt_fields_v1",
            "item_count": 3,
            "ok_count": 3,
            "error_count": 0,
            "eval": {
                "summary": {
                    "items": 3,
                    "gt_found": 3,
                    "pred_found": 3,
                },
                "fields": [
                    {"path": "tax", "mismatch": 2, "matched": 1, "accuracy": 0.333},
                    {"path": "total", "mismatch": 1, "matched": 2, "accuracy": 0.666},
                ],
                "by_request_id": {
                    "r1": {
                        "mismatches_count": 2,
                    },
                    "r2": {
                        "mismatches_count": 1,
                    },
                    "r3": {
                        "mismatches_count": 0,
                    },
                },
            },
        },
    )

    resp = await runs.list_runs(principal=_principal(), limit=10, cursor=None)

    assert len(resp.items) == 1
    assert resp.items[0].eval_summary is not None
    assert resp.items[0].eval_summary.mismatched == 2
    assert resp.items[0].eval_summary.fields == [
        {"path": "tax", "mismatch": 2, "matched": 1, "accuracy": 0.333},
        {"path": "total", "mismatch": 1, "matched": 2, "accuracy": 0.666},
    ]


@pytest.mark.asyncio
async def test_get_run_forbidden_for_non_owner(monkeypatch):
    monkeypatch.setattr(runs, "find_batch_artifact_path", lambda *_args, **_kwargs: "/x/run1.json")
    monkeypatch.setattr(runs, "read_artifact_json", lambda _: {"auth": {"key_id": "owner-1"}})

    with pytest.raises(HTTPException) as exc:
        await runs.get_run("run1", principal=_principal(role="client", key_id="other"))
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_get_eval_artifact_reads_eval_by_rel_and_allows_owner(monkeypatch):
    def _read_artifact_json(ref):
        ref_str = str(ref).replace("\\", "/")
        if ref_str.endswith("evals/2026-03-12/e1.json"):
            return {
                "eval_id": "e1",
                "batch_artifact_rel": "batches/2026-03-12/run1.json",
                "samples": [{"request_id": "r1"}],
            }
        if ref_str.endswith("batches/2026-03-12/run1.json"):
            return {"auth": {"key_id": "k1"}}
        raise AssertionError(f"unexpected ref: {ref}")

    monkeypatch.setattr(runs, "read_artifact_json", _read_artifact_json)

    resp = await runs.get_eval_artifact(
        artifact_rel="evals/2026-03-12/e1.json",
        principal=_principal(role="client", key_id="k1"),
    )
    assert resp["eval_id"] == "e1"


@pytest.mark.asyncio
async def test_get_eval_artifact_forbidden_for_non_owner(monkeypatch):
    def _read_artifact_json(ref):
        ref_str = str(ref).replace("\\", "/")
        if ref_str.endswith("evals/2026-03-12/e1.json"):
            return {
                "eval_id": "e1",
                "batch_artifact_rel": "batches/2026-03-12/run1.json",
            }
        if ref_str.endswith("batches/2026-03-12/run1.json"):
            return {"auth": {"key_id": "owner-1"}}
        raise AssertionError(f"unexpected ref: {ref}")

    monkeypatch.setattr(runs, "read_artifact_json", _read_artifact_json)

    with pytest.raises(HTTPException) as exc:
        await runs.get_eval_artifact(
            artifact_rel="evals/2026-03-12/e1.json",
            principal=_principal(role="client", key_id="other"),
        )
    assert exc.value.status_code == 403
