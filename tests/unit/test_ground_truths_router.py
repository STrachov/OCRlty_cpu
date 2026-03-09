from types import SimpleNamespace

import pytest

from app.auth_store import ApiPrincipal
from app.routers import ground_truths
from app.services.ground_truths_store import GroundTruthRecord


def _principal(*, role: str = "client", key_id: str = "k1") -> ApiPrincipal:
    return ApiPrincipal(api_key_id=1, key_id=key_id, role=role, scopes={"debug:run"})


def _request_with_store(store: object):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(ground_truths_store=store)))


@pytest.mark.asyncio
async def test_list_ground_truths_returns_items(monkeypatch):
    monkeypatch.setattr(
        ground_truths,
        "list_ground_truths_for_principal",
        lambda **_: [
            GroundTruthRecord(
                gt_id="gt_1",
                owner_key_id="k1",
                name="dataset.json",
                s3_rel="ground_truths/gt_1.json",
                created_at="2026-03-06T10:00:00Z",
            )
        ],
    )

    resp = await ground_truths.list_ground_truths(
        request=_request_with_store(object()),
        principal=_principal(),
        limit=10,
        offset=0,
    )
    assert len(resp.items) == 1
    assert resp.items[0].gt_id == "gt_1"
    assert resp.items[0].name == "dataset.json"
