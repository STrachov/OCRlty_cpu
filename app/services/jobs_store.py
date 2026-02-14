# app/services/jobs_store.py
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.auth_store import ApiPrincipal


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
    result: Optional[Dict[str, Any]]
    error: Optional[Dict[str, Any]]


class JobsStore:
    """
    Minimal SQLite job store.
    - One connection per call (thread-safe enough for FastAPI + background tasks).
    - Best-effort; designed to be swappable with other backends later.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = str(db_path)
        self._init_lock = threading.Lock()
        self._initialized = False

    def _ensure_db_dir(self) -> None:
        Path(self._db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        self._ensure_db_dir()
        conn = sqlite3.connect(self._db_path, timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Safer defaults for concurrent read/write
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id           TEXT PRIMARY KEY,
                    kind             TEXT NOT NULL,
                    status           TEXT NOT NULL,

                    created_at       TEXT NOT NULL,
                    updated_at       TEXT NOT NULL,
                    started_at       TEXT,
                    finished_at      TEXT,

                    owner_key_id     TEXT,
                    owner_role       TEXT,
                    owner_scopes     TEXT NOT NULL,  -- JSON array

                    cancel_requested INTEGER NOT NULL DEFAULT 0,

                    request_json     TEXT NOT NULL,  -- JSON object
                    progress_json    TEXT,           -- JSON object
                    result_json      TEXT,           -- JSON object
                    error_json       TEXT            -- JSON object
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_owner ON jobs(owner_key_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);")
            conn.commit()

    def ensure_init(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            self.init()
            self._initialized = True

    # ---- helpers ----

    def _row_to_record(self, r: sqlite3.Row) -> JobRecord:
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
            result=_json_loads(r["result_json"]),
            error=_json_loads(r["error_json"]),
        )

    # ---- public API ----

    def create_job(self, *, kind: str, request: Dict[str, Any], principal: ApiPrincipal) -> JobRecord:
        self.ensure_init()
        job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:10]
        now = _utc_now_iso()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, kind, status,
                    created_at, updated_at,
                    owner_key_id, owner_role, owner_scopes,
                    cancel_requested,
                    request_json, progress_json, result_json, error_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, NULL, NULL, NULL);
                """,
                (
                    job_id,
                    kind,
                    "queued",
                    now,
                    now,
                    principal.key_id,
                    principal.role,
                    _json_dumps(sorted(list(principal.scopes))),
                    _json_dumps(request or {}),
                ),
            )
            conn.commit()

            row = conn.execute("SELECT * FROM jobs WHERE job_id = ? LIMIT 1;", (job_id,)).fetchone()
            assert row is not None
            return self._row_to_record(row)

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        self.ensure_init()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ? LIMIT 1;", (job_id,)).fetchone()
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
            conds.append("owner_key_id = ?")
            args.append(owner_key_id)
        if status:
            conds.append("status = ?")
            args.append(status)

        if conds:
            q += " WHERE " + " AND ".join(conds)

        q += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        args.extend([int(limit), int(offset)])

        with self._connect() as conn:
            rows = conn.execute(q, tuple(args)).fetchall()
            return [self._row_to_record(r) for r in rows]

    def mark_running(self, job_id: str) -> None:
        self.ensure_init()
        now = _utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'running',
                    updated_at = ?,
                    started_at = COALESCE(started_at, ?)
                WHERE job_id = ? AND status = 'queued';
                """,
                (now, now, job_id),
            )
            conn.commit()

    def mark_succeeded(self, job_id: str, result: Dict[str, Any]) -> None:
        self.ensure_init()
        now = _utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'succeeded',
                    updated_at = ?,
                    finished_at = ?,
                    result_json = ?,
                    error_json = NULL
                WHERE job_id = ?;
                """,
                (now, now, _json_dumps(result or {}), job_id),
            )
            conn.commit()

    def mark_failed(self, job_id: str, error: Dict[str, Any]) -> None:
        self.ensure_init()
        now = _utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    updated_at = ?,
                    finished_at = ?,
                    error_json = ?,
                    result_json = NULL
                WHERE job_id = ?;
                """,
                (now, now, _json_dumps(error or {}), job_id),
            )
            conn.commit()

    def mark_canceled(self, job_id: str) -> None:
        self.ensure_init()
        now = _utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'canceled',
                    updated_at = ?,
                    finished_at = COALESCE(finished_at, ?),
                    cancel_requested = 1
                WHERE job_id = ? AND status IN ('queued','running');
                """,
                (now, now, job_id),
            )
            conn.commit()

    def set_progress(self, job_id: str, progress: Dict[str, Any]) -> None:
        self.ensure_init()
        now = _utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET updated_at = ?,
                    progress_json = ?
                WHERE job_id = ?;
                """,
                (now, _json_dumps(progress or {}), job_id),
            )
            conn.commit()

    def request_cancel(self, job_id: str) -> None:
        self.ensure_init()
        now = _utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET cancel_requested = 1,
                    updated_at = ?
                WHERE job_id = ?;
                """,
                (now, job_id),
            )
            conn.commit()


    def mark_stale_running_as_failed(
        self,
        *,
        include_queued: bool = False,
        reason: str = "stale_after_restart",
    ) -> int:
        """
        Mark jobs left in 'running' (and optionally 'queued') after process restart.
        Returns number of affected rows.

        Recommended: call this on startup ONLY for local/in-process runner.
        For Celery backend you typically do NOT want to mark queued/running as stale.
        """
        self.ensure_init()
        now = _utc_now_iso()

        stale_error = {
            "type": reason,
            "message": "Job was running when the process restarted; execution state was lost.",
            "at": now,
        }

        statuses = ["running"]
        if include_queued:
            statuses.append("queued")

        q_marks = ",".join(["?"] * len(statuses))

        with self._connect() as conn:
            # 1) running/queued + cancel_requested -> canceled
            cur1 = conn.execute(
                f"""
                UPDATE jobs
                SET status = 'canceled',
                    updated_at = ?,
                    finished_at = COALESCE(finished_at, ?),
                    cancel_requested = 1,
                    result_json = NULL
                WHERE status IN ({q_marks}) AND cancel_requested = 1;
                """,
                (now, now, *statuses),
            )

            # 2) remaining running/queued -> failed (stale)
            cur2 = conn.execute(
                f"""
                UPDATE jobs
                SET status = 'failed',
                    updated_at = ?,
                    finished_at = COALESCE(finished_at, ?),
                    error_json = ?,
                    result_json = NULL
                WHERE status IN ({q_marks}) AND cancel_requested = 0;
                """,
                (now, now, _json_dumps(stale_error), *statuses),
            )

            conn.commit()
            return int((cur1.rowcount or 0) + (cur2.rowcount or 0))
