# app/services/jobs_store.py
from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from app.auth_store import ApiPrincipal

logger = logging.getLogger(__name__)

try:
    import psycopg
    from psycopg.rows import dict_row
except Exception:  # pragma: no cover - optional dependency
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]


# ---- status model ----

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_CANCELED = "canceled"

ALLOWED_STATUSES: Sequence[str] = (
    STATUS_QUEUED,
    STATUS_RUNNING,
    STATUS_SUCCEEDED,
    STATUS_FAILED,
    STATUS_CANCELED,
)

TERMINAL_STATUSES: Sequence[str] = (
    STATUS_SUCCEEDED,
    STATUS_FAILED,
    STATUS_CANCELED,
)


# ---- utils ----

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def _json_loads(s: Optional[str]) -> Any:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        logger.debug("Failed to json.loads(%r)", s, exc_info=True)
        return None


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    kind: str
    status: str
    created_at: str
    updated_at: str
    started_at: Optional[str]
    finished_at: Optional[str]
    owner_key_id: Optional[str]
    owner_role: Optional[str]
    owner_scopes: List[str]
    cancel_requested: bool

    request: Dict[str, Any]
    progress: Optional[Dict[str, Any]]

    # Thin result:
    result_ref: Optional[str]
    result_meta: Optional[Dict[str, Any]]
    result_bytes: Optional[int]
    result_sha256: Optional[str]

    # Backward compatible alias: points to result_meta (never the full "thick" JSON).
    result: Optional[Dict[str, Any]]

    error_ref: Optional[str]
    error: Optional[Dict[str, Any]]


class JobsStore:
    """
    PostgreSQL-backed jobs store.

    Notes:
    - Expects PostgreSQL DSN in constructor.
    - Uses one connection per call.
    """

    def __init__(self, database_url: str) -> None:
        self._database_url = str(database_url or "").strip()
        if not (
            self._database_url.startswith("postgresql://")
            or self._database_url.startswith("postgres://")
            or self._database_url.startswith("postgresql+psycopg://")
        ):
            raise RuntimeError(
                "JobsStore requires PostgreSQL DATABASE_URL "
                "(expected postgresql://..., postgres://... or postgresql+psycopg://...)."
            )
        self._init_lock = threading.Lock()
        self._initialized = False

    def _connect(self) -> Any:
        if psycopg is None:
            raise RuntimeError(
                "JobsStore requires psycopg. Install dependency: psycopg[binary]>=3.2,<4"
            )
        dsn = self._database_url
        if dsn.startswith("postgresql+psycopg://"):
            dsn = "postgresql://" + dsn[len("postgresql+psycopg://") :]
        return psycopg.connect(  # type: ignore[union-attr]
            dsn,
            connect_timeout=10,
            row_factory=dict_row,
        )

    def _execute(self, conn: Any, query: str, args: Sequence[Any] = ()) -> Any:
        return conn.execute(query, tuple(args))

    def _fetchone(self, conn: Any, query: str, args: Sequence[Any] = ()) -> Optional[Mapping[str, Any]]:
        return self._execute(conn, query, args).fetchone()

    def _fetchall(self, conn: Any, query: str, args: Sequence[Any] = ()) -> List[Mapping[str, Any]]:
        rows = self._execute(conn, query, args).fetchall()
        return list(rows or [])

    def init(self) -> None:
        # Runtime init is intentionally minimal; Alembic should own schema lifecycle.
        with self._connect() as conn:
            cur = self._execute(
                conn,
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = current_schema() AND table_name = 'jobs';",
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(
                    "Table 'jobs' is missing. Run migrations first: alembic upgrade head"
                )

    def ensure_init(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            self.init()
            self._initialized = True

    # ---- helpers ----

    def _row_to_record(self, r: Mapping[str, Any]) -> JobRecord:
        keys = set(r.keys())
        result_ref = str(r["result_ref"]) if ("result_ref" in keys and r["result_ref"]) else None
        error_ref = str(r["error_ref"]) if ("error_ref" in keys and r["error_ref"]) else None

        result_meta = _json_loads(r["result_meta_json"]) if ("result_meta_json" in keys and r["result_meta_json"]) else None
        result_bytes = int(r["result_bytes"]) if ("result_bytes" in keys and r["result_bytes"] is not None) else None
        result_sha256 = str(r["result_sha256"]) if ("result_sha256" in keys and r["result_sha256"]) else None

        return JobRecord(
            job_id=str(r["job_id"]),
            kind=str(r["kind"]),
            status=str(r["status"]),
            created_at=str(r["created_at"]),
            updated_at=str(r["updated_at"]),
            started_at=str(r["started_at"]) if r["started_at"] else None,
            finished_at=str(r["finished_at"]) if r["finished_at"] else None,
            owner_key_id=str(r["owner_key_id"]) if r["owner_key_id"] else None,
            owner_role=str(r["owner_role"]) if r["owner_role"] else None,
            owner_scopes=list(_json_loads(r["owner_scopes"]) or []),
            cancel_requested=bool(int(r["cancel_requested"] or 0)),
            request=_json_loads(r["request_json"]) or {},
            progress=_json_loads(r["progress_json"]),
            result_ref=result_ref,
            result_meta=result_meta,
            result_bytes=result_bytes,
            result_sha256=result_sha256,
            result=result_meta,
            error_ref=error_ref,
            error=_json_loads(r["error_json"]),
        )

    # ---- public API ----

    def create_job(self, *, kind: str, request: Dict[str, Any], principal: ApiPrincipal) -> JobRecord:
        self.ensure_init()
        job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:10]
        now = _utc_now_iso()

        with self._connect() as conn:
            self._execute(
                conn,
                """
                INSERT INTO jobs (
                    job_id, kind, status,
                    created_at, updated_at,
                    owner_key_id, owner_role, owner_scopes,
                    cancel_requested,
                    request_json, progress_json, error_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 0, %s, NULL, NULL);
                """,
                (
                    job_id,
                    kind,
                    STATUS_QUEUED,
                    now,
                    now,
                    principal.key_id,
                    principal.role,
                    _json_dumps(sorted(list(principal.scopes))),
                    _json_dumps(request or {}),
                ),
            )
            conn.commit()

            row = self._fetchone(conn, "SELECT * FROM jobs WHERE job_id = %s LIMIT 1;", (job_id,))
            assert row is not None
            return self._row_to_record(row)

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        self.ensure_init()
        with self._connect() as conn:
            row = self._fetchone(conn, "SELECT * FROM jobs WHERE job_id = %s LIMIT 1;", (job_id,))
            return self._row_to_record(row) if row else None

    def list_jobs(
        self,
        *,
        owner_key_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[JobRecord]:
        self.ensure_init()
        q = "SELECT * FROM jobs"
        conds: List[str] = []
        args: List[Any] = []

        if owner_key_id:
            conds.append("owner_key_id = %s")
            args.append(owner_key_id)
        if status:
            conds.append("status = %s")
            args.append(status)

        if conds:
            q += " WHERE " + " AND ".join(conds)

        q += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        args.extend([int(limit), int(offset)])

        with self._connect() as conn:
            rows = self._fetchall(conn, q, tuple(args))
            return [self._row_to_record(r) for r in rows]

    def mark_running(self, job_id: str) -> None:
        self.ensure_init()
        now = _utc_now_iso()
        with self._connect() as conn:
            self._execute(
                conn,
                """
                UPDATE jobs
                SET status = %s,
                    updated_at = %s,
                    finished_at = COALESCE(finished_at, %s),
                    cancel_requested = 1
                WHERE job_id = %s
                  AND status = %s
                  AND cancel_requested = 1;
                """,
                (STATUS_CANCELED, now, now, job_id, STATUS_QUEUED),
            )
            self._execute(
                conn,
                """
                UPDATE jobs
                SET status = %s,
                    updated_at = %s,
                    started_at = COALESCE(started_at, %s)
                WHERE job_id = %s
                  AND status = %s
                  AND cancel_requested = 0;
                """,
                (STATUS_RUNNING, now, now, job_id, STATUS_QUEUED),
            )
            conn.commit()

    def mark_succeeded(
        self,
        job_id: str,
        *,
        result_ref: str,
        result_meta: Dict[str, Any],
        result_bytes: Optional[int] = None,
        result_sha256: Optional[str] = None,
    ) -> None:
        self.ensure_init()
        now = _utc_now_iso()
        with self._connect() as conn:
            self._execute(
                conn,
                """
                UPDATE jobs
                SET status = %s,
                    updated_at = %s,
                    finished_at = COALESCE(finished_at, %s),
                    cancel_requested = 1,
                    result_ref = NULL,
                    result_meta_json = NULL,
                    result_bytes = NULL,
                    result_sha256 = NULL,
                    error_json = NULL,
                    error_ref = NULL
                WHERE job_id = %s
                  AND status = %s
                  AND cancel_requested = 1;
                """,
                (STATUS_CANCELED, now, now, job_id, STATUS_RUNNING),
            )

            self._execute(
                conn,
                """
                UPDATE jobs
                SET status = %s,
                    updated_at = %s,
                    finished_at = COALESCE(finished_at, %s),
                    result_ref = %s,
                    result_meta_json = %s,
                    result_bytes = %s,
                    result_sha256 = %s,
                    error_json = NULL,
                    error_ref = NULL
                WHERE job_id = %s
                  AND status = %s
                  AND cancel_requested = 0;
                """,
                (
                    STATUS_SUCCEEDED,
                    now,
                    now,
                    str(result_ref),
                    _json_dumps(result_meta or {}),
                    int(result_bytes) if result_bytes is not None else None,
                    str(result_sha256) if result_sha256 else None,
                    job_id,
                    STATUS_RUNNING,
                ),
            )
            conn.commit()

    def mark_failed(self, job_id: str, error: Dict[str, Any], *, error_ref: Optional[str] = None) -> None:
        self.ensure_init()
        now = _utc_now_iso()
        with self._connect() as conn:
            self._execute(
                conn,
                """
                UPDATE jobs
                SET status = %s,
                    updated_at = %s,
                    finished_at = COALESCE(finished_at, %s),
                    cancel_requested = 1,
                    result_ref = NULL,
                    result_meta_json = NULL,
                    result_bytes = NULL,
                    result_sha256 = NULL,
                    error_json = NULL,
                    error_ref = NULL
                WHERE job_id = %s
                  AND status IN (%s,%s)
                  AND cancel_requested = 1;
                """,
                (STATUS_CANCELED, now, now, job_id, STATUS_QUEUED, STATUS_RUNNING),
            )

            self._execute(
                conn,
                """
                UPDATE jobs
                SET status = %s,
                    updated_at = %s,
                    finished_at = COALESCE(finished_at, %s),
                    result_ref = NULL,
                    result_meta_json = NULL,
                    result_bytes = NULL,
                    result_sha256 = NULL,
                    error_json = %s,
                    error_ref = %s
                WHERE job_id = %s
                  AND status IN (%s,%s)
                  AND cancel_requested = 0;
                """,
                (
                    STATUS_FAILED,
                    now,
                    now,
                    _json_dumps(error or {}),
                    str(error_ref) if error_ref else None,
                    job_id,
                    STATUS_QUEUED,
                    STATUS_RUNNING,
                ),
            )
            conn.commit()

    def mark_canceled(self, job_id: str) -> None:
        self.ensure_init()
        now = _utc_now_iso()
        with self._connect() as conn:
            self._execute(
                conn,
                """
                UPDATE jobs
                SET status = %s,
                    updated_at = %s,
                    finished_at = COALESCE(finished_at, %s),
                    cancel_requested = 1
                WHERE job_id = %s
                  AND status IN (%s, %s);
                """,
                (STATUS_CANCELED, now, now, job_id, STATUS_QUEUED, STATUS_RUNNING),
            )
            conn.commit()

    def set_progress(self, job_id: str, progress: Dict[str, Any]) -> None:
        self.ensure_init()
        now = _utc_now_iso()
        with self._connect() as conn:
            self._execute(
                conn,
                """
                UPDATE jobs
                SET updated_at = %s,
                    progress_json = %s
                WHERE job_id = %s
                  AND status IN (%s, %s)
                  AND finished_at IS NULL;
                """,
                (now, _json_dumps(progress or {}), job_id, STATUS_QUEUED, STATUS_RUNNING),
            )
            conn.commit()

    def request_cancel(self, job_id: str) -> None:
        self.ensure_init()
        now = _utc_now_iso()
        with self._connect() as conn:
            self._execute(
                conn,
                """
                UPDATE jobs
                SET cancel_requested = 1,
                    updated_at = %s
                WHERE job_id = %s
                  AND status IN (%s, %s);
                """,
                (now, job_id, STATUS_QUEUED, STATUS_RUNNING),
            )
            conn.commit()

    def mark_stale_running_as_failed(
        self,
        *,
        include_queued: bool = False,
        reason: str = "stale_after_restart",
    ) -> int:
        self.ensure_init()
        now = _utc_now_iso()

        stale_error = {
            "type": reason,
            "message": "Job was running when the process restarted; execution state was lost.",
            "at": now,
        }

        statuses: List[str] = [STATUS_RUNNING]
        if include_queued:
            statuses.append(STATUS_QUEUED)

        q_marks = ",".join(["%s"] * len(statuses))

        with self._connect() as conn:
            cur1 = self._execute(
                conn,
                f"""
                UPDATE jobs
                SET status = %s,
                    updated_at = %s,
                    finished_at = COALESCE(finished_at, %s),
                    cancel_requested = 1
                WHERE status IN ({q_marks})
                  AND cancel_requested = 1
                  AND status NOT IN ('{STATUS_SUCCEEDED}','{STATUS_FAILED}','{STATUS_CANCELED}');
                """,
                (STATUS_CANCELED, now, now, *statuses),
            )

            cur2 = self._execute(
                conn,
                f"""
                UPDATE jobs
                SET status = %s,
                    updated_at = %s,
                    finished_at = COALESCE(finished_at, %s),
                    error_json = %s
                WHERE status IN ({q_marks})
                  AND cancel_requested = 0
                  AND status NOT IN ('{STATUS_SUCCEEDED}','{STATUS_FAILED}','{STATUS_CANCELED}');
                """,
                (STATUS_FAILED, now, now, _json_dumps(stale_error), *statuses),
            )

            conn.commit()
            return int((cur1.rowcount or 0) + (cur2.rowcount or 0))
