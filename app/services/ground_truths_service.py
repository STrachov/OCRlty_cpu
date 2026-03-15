from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from fastapi import HTTPException

from app.auth_store import ApiPrincipal
from app.services.artifacts import artifact_ref_from_rel, find_artifact_path, find_batch_artifact_path, read_artifact_json
from app.services.ground_truths_store import GroundTruthRecord, GroundTruthsStore
from app.services.schema_canonicalization import canonicalize_value_for_schema
from app.services.s3_service import (
    s3_delete_object,
    s3_enabled,
    s3_get_json,
    s3_key,
    s3_put_bytes,
    s3_put_bytes_overwrite,
)
from app.services.task_registry import get_task


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _gt_s3_rel(gt_id: str) -> str:
    return f"ground_truths/{gt_id}.json"


def _normalize_name(name: str | None) -> str:
    raw = str(name or "").strip()
    return raw or "ground_truth.json"


def _enforce_owner_or_admin(principal: ApiPrincipal, rec: GroundTruthRecord) -> None:
    if principal.role == "admin":
        return
    if rec.owner_key_id != principal.key_id:
        raise HTTPException(status_code=404, detail="Ground truth file not found.")


def _parse_gt_json(data: bytes) -> Any:
    try:
        text = (data or b"").decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="Ground truth file must be valid UTF-8 JSON.")
    try:
        obj = json.loads(text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ground truth file must be valid JSON: {e}")
    if not isinstance(obj, (dict, list)):
        raise HTTPException(status_code=400, detail="Ground truth JSON must be an object or an array.")
    return obj


def _parse_gt_obj(obj: Any) -> Any:
    if not isinstance(obj, (dict, list)):
        raise HTTPException(status_code=400, detail="Ground truth JSON must be an object or an array.")
    return obj


def create_ground_truth_from_upload(
    *,
    store: GroundTruthsStore,
    principal: ApiPrincipal,
    filename: str | None,
    data: bytes,
) -> GroundTruthRecord:
    if not s3_enabled():
        raise HTTPException(status_code=501, detail="S3 is not enabled.")
    _parse_gt_json(data)
    gt_id = uuid.uuid4().hex
    created_at = _utc_now_iso()
    name = _normalize_name(filename)
    s3_rel = _gt_s3_rel(gt_id)
    ok = s3_put_bytes(
        key=s3_key(s3_rel),
        data=data,
        content_type="application/json; charset=utf-8",
    )
    if not ok:
        raise HTTPException(status_code=409, detail="Ground truth object already exists.")
    return store.create_ground_truth(
        gt_id=gt_id,
        owner_key_id=str(principal.key_id),
        name=name,
        s3_rel=s3_rel,
        created_at=created_at,
        source_type="upload",
        updated_at=created_at,
    )


def create_ground_truth_from_content(
    *,
    store: GroundTruthsStore,
    principal: ApiPrincipal,
    name: str,
    payload: Any,
) -> GroundTruthRecord:
    if not s3_enabled():
        raise HTTPException(status_code=501, detail="S3 is not enabled.")
    obj = _parse_gt_obj(payload)
    gt_id = uuid.uuid4().hex
    created_at = _utc_now_iso()
    normalized_name = _normalize_name(name)
    s3_rel = _gt_s3_rel(gt_id)

    if isinstance(obj, dict):
        obj["gt_id"] = gt_id
        obj["name"] = normalized_name
        obj["created_at"] = obj.get("created_at") or created_at
        obj["updated_at"] = created_at

    task_id = str(obj.get("task_id") or "").strip() if isinstance(obj, dict) else ""
    source_type = str(obj.get("source_type") or "").strip() if isinstance(obj, dict) else ""
    source_run_id = str(obj.get("source_run_id") or "").strip() if isinstance(obj, dict) else ""
    body = (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    ok = s3_put_bytes(
        key=s3_key(s3_rel),
        data=body,
        content_type="application/json; charset=utf-8",
    )
    if not ok:
        raise HTTPException(status_code=409, detail="Ground truth object already exists.")

    return store.create_ground_truth(
        gt_id=gt_id,
        owner_key_id=str(principal.key_id),
        name=normalized_name,
        s3_rel=s3_rel,
        created_at=created_at,
        task_id=task_id or None,
        source_type=source_type or None,
        source_run_id=source_run_id or None,
        updated_at=created_at,
    )


def _build_gt_records_from_run(batch_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    task_id = str(batch_obj.get("task_id") or "").strip()
    task = get_task(task_id) if task_id else None
    root_schema = task.get("schema_value") if isinstance(task, dict) else None
    items = batch_obj.get("items") if isinstance(batch_obj.get("items"), list) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        file_name = str(item.get("file") or "").strip()
        parsed = item.get("parsed")
        if not isinstance(parsed, dict):
            artifact_rel = str(item.get("artifact_rel") or "").strip()
            request_id = str(item.get("request_id") or "").strip()
            try:
                if artifact_rel:
                    extract_obj = read_artifact_json(artifact_ref_from_rel(artifact_rel))
                elif request_id:
                    extract_ref = find_artifact_path(request_id, max_days=3650)
                    extract_obj = read_artifact_json(extract_ref) if extract_ref is not None else {}
                else:
                    extract_obj = {}
            except Exception:
                extract_obj = {}
            parsed = extract_obj.get("parsed") if isinstance(extract_obj, dict) else None
        if not file_name or not isinstance(parsed, dict):
            continue
        normalized_payload = (
            canonicalize_value_for_schema(parsed, root_schema, root_schema=root_schema)
            if isinstance(root_schema, dict)
            else parsed
        )
        if not isinstance(normalized_payload, dict):
            normalized_payload = parsed
        records.append({
            "file": file_name,
            **normalized_payload,
        })
    if not records:
        raise HTTPException(status_code=400, detail="Run has no parsed items suitable for ground truth creation.")
    return records


def build_ground_truth_draft_from_run(*, batch_obj: Dict[str, Any], run_id: str, name: str | None) -> Dict[str, Any]:
    task_id = str(batch_obj.get("task_id") or "").strip() or None
    created_at = _utc_now_iso()
    normalized_name = _normalize_name(name or f"{run_id}.json")
    return {
        "name": normalized_name,
        "created_at": created_at,
        "updated_at": created_at,
        "task_id": task_id,
        "source_type": "run_bootstrap",
        "source_run_id": run_id,
        "meta": {
            "model_id": batch_obj.get("model_id"),
            "prompt_sha256": batch_obj.get("prompt_sha256"),
            "schema_sha256": batch_obj.get("schema_sha256"),
            "code_version": batch_obj.get("code_version"),
        },
        "items": _build_gt_records_from_run(batch_obj),
    }


def create_ground_truth_from_run(
    *,
    store: GroundTruthsStore,
    principal: ApiPrincipal,
    run_id: str,
    name: str | None,
) -> GroundTruthRecord:
    if not s3_enabled():
        raise HTTPException(status_code=501, detail="S3 is not enabled.")

    batch_ref = find_batch_artifact_path(run_id, date=None, max_days=3650)
    if batch_ref is None:
        raise HTTPException(status_code=404, detail="Run not found.")

    batch_obj = read_artifact_json(batch_ref)
    if not isinstance(batch_obj, dict):
        raise HTTPException(status_code=500, detail="Invalid run JSON.")

    auth = batch_obj.get("auth") if isinstance(batch_obj.get("auth"), dict) else {}
    owner_key_id = str(auth.get("key_id") or "")
    if principal.role != "admin" and owner_key_id != principal.key_id:
        raise HTTPException(status_code=404, detail="Run not found.")

    gt_id = uuid.uuid4().hex
    draft = build_ground_truth_draft_from_run(batch_obj=batch_obj, run_id=run_id, name=name)
    created_at = str(draft["created_at"])
    normalized_name = str(draft["name"])
    task_id = draft.get("task_id")
    s3_rel = _gt_s3_rel(gt_id)

    payload: Dict[str, Any] = {
        "gt_id": gt_id,
        **draft,
    }
    body = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    ok = s3_put_bytes(
        key=s3_key(s3_rel),
        data=body,
        content_type="application/json; charset=utf-8",
    )
    if not ok:
        raise HTTPException(status_code=409, detail="Ground truth object already exists.")

    return store.create_ground_truth(
        gt_id=gt_id,
        owner_key_id=str(principal.key_id),
        name=normalized_name,
        s3_rel=s3_rel,
        created_at=created_at,
        task_id=task_id,
        source_type="run_bootstrap",
        source_run_id=run_id,
        updated_at=created_at,
    )


def load_run_for_ground_truth(
    *,
    principal: ApiPrincipal,
    run_id: str,
) -> Dict[str, Any]:
    batch_ref = find_batch_artifact_path(run_id, date=None, max_days=3650)
    if batch_ref is None:
        raise HTTPException(status_code=404, detail="Run not found.")

    batch_obj = read_artifact_json(batch_ref)
    if not isinstance(batch_obj, dict):
        raise HTTPException(status_code=500, detail="Invalid run JSON.")

    auth = batch_obj.get("auth") if isinstance(batch_obj.get("auth"), dict) else {}
    owner_key_id = str(auth.get("key_id") or "")
    if principal.role != "admin" and owner_key_id != principal.key_id:
        raise HTTPException(status_code=404, detail="Run not found.")
    return batch_obj


def build_ground_truth_draft_for_principal(
    *,
    principal: ApiPrincipal,
    run_id: str,
    name: str | None,
) -> Dict[str, Any]:
    batch_obj = load_run_for_ground_truth(principal=principal, run_id=run_id)
    return build_ground_truth_draft_from_run(batch_obj=batch_obj, run_id=run_id, name=name)


def update_ground_truth_content(
    *,
    store: GroundTruthsStore,
    principal: ApiPrincipal,
    gt_id: str,
    name: str,
    payload: Any,
) -> GroundTruthRecord:
    if not s3_enabled():
        raise HTTPException(status_code=501, detail="S3 is not enabled.")
    rec = get_ground_truth_for_principal(store=store, principal=principal, gt_id=gt_id)
    normalized_name = _normalize_name(name)
    obj = _parse_gt_obj(payload)

    if isinstance(obj, dict):
        obj["gt_id"] = gt_id
        obj["name"] = normalized_name
        obj["updated_at"] = _utc_now_iso()

    body = (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    s3_put_bytes_overwrite(
        key=s3_key(rec.s3_rel),
        data=body,
        content_type="application/json; charset=utf-8",
    )
    return store.update_ground_truth(gt_id=gt_id, name=normalized_name, updated_at=_utc_now_iso())


def get_ground_truth_for_principal(
    *,
    store: GroundTruthsStore,
    principal: ApiPrincipal,
    gt_id: str,
) -> GroundTruthRecord:
    rec = store.get_ground_truth(gt_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Ground truth file not found.")
    _enforce_owner_or_admin(principal, rec)
    return rec


def list_ground_truths_for_principal(
    *,
    store: GroundTruthsStore,
    principal: ApiPrincipal,
    limit: int,
    offset: int,
) -> List[GroundTruthRecord]:
    owner_key_id = None if principal.role == "admin" else principal.key_id
    return store.list_ground_truths(owner_key_id=owner_key_id, limit=limit, offset=offset)


def load_ground_truth_json(
    *,
    store: GroundTruthsStore,
    principal: ApiPrincipal,
    gt_id: str,
) -> Tuple[GroundTruthRecord, Any]:
    rec = get_ground_truth_for_principal(store=store, principal=principal, gt_id=gt_id)
    obj = s3_get_json(s3_key(rec.s3_rel))
    if not isinstance(obj, (dict, list)):
        raise HTTPException(status_code=500, detail="Ground truth JSON must be an object or an array.")
    return rec, obj


def delete_ground_truth_for_principal(
    *,
    store: GroundTruthsStore,
    principal: ApiPrincipal,
    gt_id: str,
) -> GroundTruthRecord:
    if not s3_enabled():
        raise HTTPException(status_code=501, detail="S3 is not enabled.")
    rec = get_ground_truth_for_principal(store=store, principal=principal, gt_id=gt_id)
    s3_delete_object(s3_key(rec.s3_rel))
    store.delete_ground_truth(gt_id)
    return rec
