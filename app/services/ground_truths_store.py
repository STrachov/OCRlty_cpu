from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from app.settings import settings

try:
    import psycopg
    from psycopg.rows import dict_row
except Exception:  # pragma: no cover - optional dependency
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]


@dataclass(frozen=True)
class GroundTruthRecord:
    gt_id: str
    owner_key_id: str
    name: str
    s3_rel: str
    created_at: str


class GroundTruthsStore:
    def __init__(self, database_url: str = settings.DATABASE_URL) -> None:
        self._database_url = str(database_url or "").strip()
        if not (
            self._database_url.startswith("postgresql://")
            or self._database_url.startswith("postgres://")
            or self._database_url.startswith("postgresql+psycopg://")
        ):
            raise RuntimeError(
                "GroundTruthsStore requires PostgreSQL DATABASE_URL "
                "(expected postgresql://..., postgres://... or postgresql+psycopg://...)."
            )
        self._init_lock = threading.Lock()
        self._initialized = False

    def _connect(self) -> Any:
        if psycopg is None:
            raise RuntimeError(
                "GroundTruthsStore requires psycopg. Install dependency: psycopg[binary]>=3.2,<4"
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
        with self._connect() as conn:
            cur = self._execute(
                conn,
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = current_schema() AND table_name = 'ground_truth_files';",
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(
                    "Table 'ground_truth_files' is missing. Run migrations first: alembic upgrade head"
                )

    def ensure_init(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            self.init()
            self._initialized = True

    def _row_to_record(self, row: Mapping[str, Any]) -> GroundTruthRecord:
        return GroundTruthRecord(
            gt_id=str(row["gt_id"]),
            owner_key_id=str(row["owner_key_id"]),
            name=str(row["name"]),
            s3_rel=str(row["s3_rel"]),
            created_at=str(row["created_at"]),
        )

    def create_ground_truth(
        self,
        *,
        gt_id: str,
        owner_key_id: str,
        name: str,
        s3_rel: str,
        created_at: str,
    ) -> GroundTruthRecord:
        self.ensure_init()
        with self._connect() as conn:
            self._execute(
                conn,
                """
                INSERT INTO ground_truth_files (gt_id, owner_key_id, name, s3_rel, created_at)
                VALUES (%s, %s, %s, %s, %s);
                """,
                (gt_id, owner_key_id, name, s3_rel, created_at),
            )
            conn.commit()
            row = self._fetchone(conn, "SELECT * FROM ground_truth_files WHERE gt_id = %s LIMIT 1;", (gt_id,))
            assert row is not None
            return self._row_to_record(row)

    def get_ground_truth(self, gt_id: str) -> Optional[GroundTruthRecord]:
        self.ensure_init()
        with self._connect() as conn:
            row = self._fetchone(conn, "SELECT * FROM ground_truth_files WHERE gt_id = %s LIMIT 1;", (gt_id,))
            return self._row_to_record(row) if row else None

    def list_ground_truths(
        self,
        *,
        owner_key_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[GroundTruthRecord]:
        self.ensure_init()
        q = "SELECT * FROM ground_truth_files"
        args: List[Any] = []
        if owner_key_id is not None:
            q += " WHERE owner_key_id = %s"
            args.append(owner_key_id)
        q += " ORDER BY created_at DESC, gt_id DESC LIMIT %s OFFSET %s"
        args.extend([int(limit), int(offset)])
        with self._connect() as conn:
            rows = self._fetchall(conn, q, tuple(args))
            return [self._row_to_record(row) for row in rows]
