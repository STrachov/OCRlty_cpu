import pytest

from app import handlers
from app.auth_store import ApiPrincipal
from app.services.ground_truths_store import GroundTruthRecord


def _principal() -> ApiPrincipal:
    return ApiPrincipal(api_key_id=1, key_id="k1", role="client", scopes={"debug:run", "extract:run"})


def test_canon_x_money_normalizes_without_minor_units(monkeypatch):
    monkeypatch.setattr(handlers, "_get_decimal_sep", lambda: ".")

    assert handlers.canon_x_money("580965") == (True, "580965", None)
    assert handlers.canon_x_money("580965.00") == (True, "580965", None)
    assert handlers.canon_x_money("1,234,567.89") == (True, "1234567.89", None)
    assert handlers.canon_x_money("52,815") == (True, "52815", None)


def test_canon_x_money_rejects_invalid_or_ambiguous_grouping(monkeypatch):
    monkeypatch.setattr(handlers, "_get_decimal_sep", lambda: ".")

    ok, value, err = handlers.canon_x_money("580,96")
    assert ok is False
    assert value is None
    assert "invalid grouping" in str(err)

    ok, value, err = handlers.canon_x_money("1.234,56")
    assert ok is False
    assert value is None


def test_canon_x_count_requires_integer_semantics(monkeypatch):
    monkeypatch.setattr(handlers, "_get_decimal_sep", lambda: ".")

    assert handlers.canon_x_count("1,234") == (True, "1234", None)
    assert handlers.canon_x_count("0012") == (True, "12", None)

    ok, value, err = handlers.canon_x_count("12.50")
    assert ok is False
    assert value is None
    assert "fractional part is not allowed" in str(err)


@pytest.mark.asyncio
async def test_eval_batch_vs_gt_uses_gt_id(monkeypatch):
    captured = {}

    monkeypatch.setattr(handlers, "find_batch_artifact_path", lambda *_args, **_kwargs: "/tmp/batches/2026-03-06/run1.json")
    monkeypatch.setattr(
        handlers,
        "read_artifact_json",
        lambda _ref: {
            "task_id": "receipt_fields_v1",
            "items": [
                {
                    "file": "a.jpg",
                    "request_id": "r1",
                    "parsed": {"total_raw": "100"},
                }
            ],
            "auth": {"key_id": "k1"},
        },
    )
    monkeypatch.setattr(
        handlers,
        "_get_task",
        lambda _task_id: {
            "schema_value": {
                "type": "object",
                "properties": {
                    "total_raw": {"type": "string"},
                },
            }
        },
    )
    monkeypatch.setattr(
        handlers,
        "load_ground_truth_json",
        lambda **_: (
            GroundTruthRecord(
                gt_id="gt_1",
                owner_key_id="k1",
                name="gt.json",
                s3_rel="ground_truths/gt_1.json",
                created_at="2026-03-06T10:00:00Z",
            ),
            [{"file": "a.jpg", "total_raw": "100"}],
        ),
    )

    def _save_eval_artifact(payload, owner_key_id=None):
        captured["payload"] = payload
        captured["owner_key_id"] = owner_key_id
        return "/tmp/evals/2026-03-06/e1.json"

    monkeypatch.setattr(handlers, "save_eval_artifact", _save_eval_artifact)

    resp = await handlers.eval_batch_vs_gt(
        handlers.EvalBatchVsGTRequest(run_id="run1", gt_id="gt_1"),
        principal=_principal(),
    )

    assert resp.gt_id == "gt_1"
    assert resp.gt_name == "gt.json"
    assert resp.summary is not None
    assert resp.summary["gt_found"] == 1
    assert resp.by_request_id is not None
    assert resp.by_request_id["r1"]["mismatches_count"] == 0
    assert captured["owner_key_id"] == "k1"
    assert captured["payload"]["samples"][0]["pred"] == {"total_raw": "100"}
    assert captured["payload"]["samples"][0]["gt"] == {"file": "a.jpg", "total_raw": "100"}
