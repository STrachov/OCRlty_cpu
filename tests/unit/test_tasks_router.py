import pytest

from app.auth_store import ApiPrincipal
from app.routers import tasks


def _principal() -> ApiPrincipal:
    return ApiPrincipal(api_key_id=1, key_id="k1", role="client", scopes={"extract:run"})


@pytest.mark.asyncio
async def test_list_tasks_returns_sorted_task_ids_with_descriptions(monkeypatch):
    monkeypatch.setattr(
        tasks,
        "list_task_summaries",
        lambda: [
            {"task_id": "invoice_v1", "description": "Invoice totals extractor."},
            {"task_id": "receipt_fields_v1", "description": "Receipt totals extractor."},
        ],
    )

    resp = await tasks.list_tasks(_principal=_principal())
    assert [item.task_id for item in resp.items] == ["invoice_v1", "receipt_fields_v1"]
    assert [item.description for item in resp.items] == ["Invoice totals extractor.", "Receipt totals extractor."]
