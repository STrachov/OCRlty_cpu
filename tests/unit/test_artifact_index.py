from pathlib import Path

from app.artifact_index import ArtifactIndex


def _mk_index(tmp_path: Path) -> ArtifactIndex:
    idx = ArtifactIndex(
        enabled=True,
        db_path=tmp_path / "artifact_index.db",
        log_event=lambda *_args, **_kwargs: None,
        is_s3_enabled=lambda: False,
    )
    idx.init()
    return idx


def test_artifact_index_list_cursor(tmp_path):
    idx = _mk_index(tmp_path)
    idx.upsert(
        kind="batch",
        artifact_id="run_a",
        full_ref="/tmp/a.json",
        rel_ref="batches/2026-02-26/run_a.json",
        day="2026-02-26",
        owner_key_id="k1",
    )
    idx.upsert(
        kind="batch",
        artifact_id="run_b",
        full_ref="/tmp/b.json",
        rel_ref="batches/2026-02-26/run_b.json",
        day="2026-02-26",
        owner_key_id="k1",
    )

    page1 = idx.list(kind="batch", limit=1, owner_key_id="k1")
    assert len(page1) == 1
    cur_created = page1[0]["created_at"]
    cur_id = page1[0]["artifact_id"]

    page2 = idx.list(
        kind="batch",
        limit=10,
        owner_key_id="k1",
        cursor_created_at=cur_created,
        cursor_artifact_id=cur_id,
    )
    assert len(page2) == 1
    assert page2[0]["artifact_id"] != cur_id
