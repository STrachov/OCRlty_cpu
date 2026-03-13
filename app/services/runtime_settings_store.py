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
class RuntimeSettingRecord:
    key: str
    value_text: str
    updated_at: str
    updated_by_key_id: Optional[str]


class RuntimeSettingsStore:
    def __init__(self, database_url: str = settings.DATABASE_URL) -> None:
        self._database_url = str(database_url or "").strip()
        if not (
            self._database_url.startswith("postgresql://")
            or self._database_url.startswith("postgres://")
            or self._database_url.startswith("postgresql+psycopg://")
        ):
            raise RuntimeError(
                "RuntimeSettingsStore requires PostgreSQL DATABASE_URL "
                "(expected postgresql://..., postgres://... or postgresql+psycopg://...)."
            )
        self._init_lock = threading.Lock()
        self._initialized = False

    def _connect(self) -> Any:
        if psycopg is None:
            raise RuntimeError(
                "RuntimeSettingsStore requires psycopg. Install dependency: psycopg[binary]>=3.2,<4"
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
                "WHERE table_schema = current_schema() AND table_name = 'runtime_settings';",
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(
                    "Table 'runtime_settings' is missing. Run migrations first: alembic upgrade head"
                )

    def ensure_init(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            self.init()
            self._initialized = True

    def _row_to_record(self, row: Mapping[str, Any]) -> RuntimeSettingRecord:
        return RuntimeSettingRecord(
            key=str(row["key"]),
            value_text=str(row["value_text"]),
            updated_at=str(row["updated_at"]),
            updated_by_key_id=str(row["updated_by_key_id"]) if row.get("updated_by_key_id") else None,
        )

    def get(self, key: str) -> Optional[RuntimeSettingRecord]:
        self.ensure_init()
        with self._connect() as conn:
            row = self._fetchone(conn, "SELECT * FROM runtime_settings WHERE key = %s LIMIT 1;", (key,))
            return self._row_to_record(row) if row else None

    def list(self) -> List[RuntimeSettingRecord]:
        self.ensure_init()
        with self._connect() as conn:
            rows = self._fetchall(conn, "SELECT * FROM runtime_settings ORDER BY key ASC;")
            return [self._row_to_record(row) for row in rows]

    def upsert(
        self,
        *,
        key: str,
        value_text: str,
        updated_at: str,
        updated_by_key_id: Optional[str],
    ) -> RuntimeSettingRecord:
        self.ensure_init()
        with self._connect() as conn:
            self._execute(
                conn,
                """
                INSERT INTO runtime_settings (key, value_text, updated_at, updated_by_key_id)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (key) DO UPDATE SET
                    value_text = EXCLUDED.value_text,
                    updated_at = EXCLUDED.updated_at,
                    updated_by_key_id = EXCLUDED.updated_by_key_id;
                """,
                (key, value_text, updated_at, updated_by_key_id),
            )
            conn.commit()
            row = self._fetchone(conn, "SELECT * FROM runtime_settings WHERE key = %s LIMIT 1;", (key,))
            assert row is not None
            return self._row_to_record(row)

    def delete(self, key: str) -> None:
        self.ensure_init()
        with self._connect() as conn:
            self._execute(conn, "DELETE FROM runtime_settings WHERE key = %s;", (key,))
            conn.commit()
