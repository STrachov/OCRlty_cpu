from app.services.jobs_runner import _thin_result_meta


def test_thin_result_meta_extract_shape():
    meta = _thin_result_meta(
        "extract",
        {
            "task_id": "receipt_fields_v1",
            "request_id": "rid1",
            "artifact_path": "extracts/2026-02-25/rid1.json",
            "schema_valid": True,
            "model_id": "m1",
            "prompt_sha256": "a",
            "schema_sha256": "b",
        },
    )

    assert meta["type"] == "extract"
    assert meta["request_id"] == "rid1"
    assert "artifact_path" in meta


def test_thin_result_meta_batch_shape():
    meta = _thin_result_meta(
        "batch_extract",
        {
            "task_id": "receipt_fields_v1",
            "run_id": "run1",
            "artifact_path": "batches/2026-02-25/run1.json",
            "concurrency": 4,
            "item_count": 10,
            "ok_count": 9,
            "error_count": 1,
            "images_dir": "d",
            "glob": "**/*",
            "eval": {"eval_artifact_path": "evals/2026-02-25/e1.json"},
        },
    )

    assert meta["type"] == "batch"
    assert meta["run_id"] == "run1"
    assert meta["eval_artifact_path"].endswith("e1.json")
