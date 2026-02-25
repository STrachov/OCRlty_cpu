"""
Debug service used by the debug router.

Goal: keep routers thin and avoid app.state.
"""
from __future__ import annotations

import os
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from app.services import artifacts
from app.services import s3_service
from app.settings import settings
from app import handlers


ARTIFACTS_DIR = Path(settings.ARTIFACTS_DIR).resolve() 

def _require_debug_enabled() -> None:
    if not settings.DEBUG_MODE:
        # Hide existence
        raise HTTPException(status_code=404, detail="Not found")


class DebugService:
    def list_artifacts(self, *, date: Optional[str], limit: int) -> handlers.ArtifactListResponse:
        _require_debug_enabled()
        items: list[handlers.ArtifactListItem] = []

        if date is not None:
            if not artifacts.DATE_RE.match(date):
                raise HTTPException(status_code=400, detail="date must be in YYYY-MM-DD format.")

            if s3_service.s3_enabled():
                prefix = s3_service._s3_key(f"extracts/{date}/")
                objs = s3_service._s3_list_objects(prefix, limit=limit * 3)
                objs.sort(
                    key=lambda o: o.get("LastModified") or datetime.fromtimestamp(0, tz=timezone.utc),
                    reverse=True,
                )
                for o in objs[:limit]:
                    key = str(o.get("Key") or "")
                    if not key.endswith(".json"):
                        continue
                    rid = Path(key).stem
                    items.append(handlers.ArtifactListItem(
                        request_id=rid, 
                        rel_path=artifacts.to_artifact_rel(key),
                        created_at=datetime.now(timezone.utc).isoformat(),kind="extract"
                        ))
                return handlers.ArtifactListResponse(items=items)

            d = ARTIFACTS_DIR / "extracts" / date
            if not d.exists():
                return handlers.ArtifactListResponse(items=[])
            files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            for p in files[:limit]:
                items.append(handlers.ArtifactListItem(
                    request_id=p.stem, 
                    rel_path=artifacts.to_artifact_rel(p),
                    created_at=datetime.now(timezone.utc).isoformat(),kind="extract"
                    ))
            return handlers.ArtifactListResponse(items=items)

        # No date: walk days desc
        for day in artifacts.iter_artifact_days_desc("extracts"):
            if len(items) >= limit:
                break
            part = self.list_artifacts(date=day, limit=min(50, limit - len(items)))
            items.extend(part.items)
        return handlers.ArtifactListResponse(items=items[:limit])

    def backup_tar_gz(self, background_tasks: BackgroundTasks) -> FileResponse:
        _require_debug_enabled()

        tmp_dir = Path(tempfile.mkdtemp(prefix="ocrlty_artifacts_backup_"))
        tar_path = tmp_dir / "artifacts_backup.tar.gz"

        # Keep it bounded (env override)
        limit = int(os.getenv("DEBUG_BACKUP_LIMIT", "1000"))
        items = self.list_artifacts(date=None, limit=min(limit, 2000)).items

        with tarfile.open(tar_path, mode="w:gz") as tf:
            for it in items:
                rid = it.request_id
                ref = artifacts.find_artifact_path(rid, date=None, max_days=30)
                if ref is None:
                    continue
                day = artifacts.artifact_date_from_path(ref) or "unknown"
                arcname = f"extracts/{day}/{rid}.json"
                if s3_service.s3_enabled() and isinstance(ref, str) and not ref.startswith("/"):
                    data = s3_service._s3_get_text(ref).encode("utf-8", errors="replace")
                else:
                    data = Path(ref).read_bytes()
                info = tarfile.TarInfo(name=arcname)
                info.size = len(data)
                info.mtime = int(datetime.now(timezone.utc).timestamp())
                import io
                tf.addfile(info, fileobj=io.BytesIO(data))

        # Cleanup after response is sent
        background_tasks.add_task(lambda: (tar_path.unlink(missing_ok=True), tmp_dir.rmdir()))
        return FileResponse(
            str(tar_path),
            media_type="application/gzip",
            filename="artifacts_backup.tar.gz",
            headers={"Cache-Control": "no-store"},
        )

    def read_artifact(self, request_id: str, *, date: Optional[str]) -> Dict[str, Any]:
        _require_debug_enabled()
        ref = artifacts.find_artifact_path(request_id, date=date, max_days=30)
        if ref is None:
            raise HTTPException(status_code=404, detail="Artifact not found.")
        data = artifacts.read_artifact_json(ref)
        return {"request_id": request_id, "date": artifacts.artifact_date_from_path(ref), "artifact": data}

    def read_artifact_raw(self, request_id: str, *, date: Optional[str]) -> Dict[str, Any]:
        _require_debug_enabled()
        ref = artifacts.find_artifact_path(request_id, date=date, max_days=30)
        if ref is None:
            raise HTTPException(status_code=404, detail="Artifact not found.")
        if s3_service.s3_enabled() and isinstance(ref, str) and not ref.startswith("/"):
            raw = s3_service._s3_get_text(ref)
        else:
            raw = Path(ref).read_text(encoding="utf-8", errors="replace")
        return {"request_id": request_id, "date": artifacts.artifact_date_from_path(ref), "raw": raw}

    async def eval_batch_vs_gt(self, req: handlers.EvalBatchVsGTRequest) -> handlers.EvalBatchVsGTResponse:
        _require_debug_enabled()
        # Safest: reuse existing implementation from handlers (logic-heavy, depends on many helpers)
        return await handlers.eval_batch_vs_gt(req)
