"""
Artifact index (PostgreSQL) for fast lookup of artifacts by id without scanning date prefixes.

- Stores only metadata (no blobs). The artifacts themselves remain in filesystem/S3.
- Best-effort: indexing failures must never break request handling.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional


try:
    import psycopg
    from psycopg.rows import dict_row
except Exception:  # pragma: no cover - optional dependency
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]


class ArtifactIndex:
    def __init__(
        self,
        *,
        enabled: bool,
        database_url: str,
        log_event: Callable[..., None],
        is_s3_enabled: Callable[[], bool] | None = None,
    ) -> None:
        self._initialized = False
        self._enabled = bool(enabled)
        self._database_url = str(database_url or "").strip()
        self._log_event = log_event
        self._is_s3_enabled = is_s3_enabled or (lambda: False)
        self._lock = threading.Lock()

        if self._enabled and not (
            self._database_url.startswith("postgresql://")
            or self._database_url.startswith("postgres://")
            or self._database_url.startswith("postgresql+psycopg://")
        ):
            raise RuntimeError(
                "ArtifactIndex requires PostgreSQL DATABASE_URL "
                "(expected postgresql://..., postgres://... or postgresql+psycopg://...)."
            )

    def enabled(self) -> bool:
        return bool(self._enabled)

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        self.init()
        self._initialized = True

    def _connect(self) -> Any:
        if psycopg is None:
            raise RuntimeError(
                "ArtifactIndex requires psycopg. Install dependency: psycopg[binary]>=3.2,<4"
            )
        dsn = self._database_url
        if dsn.startswith("postgresql+psycopg://"):
            dsn = "postgresql://" + dsn[len("postgresql+psycopg://") :]
        return psycopg.connect(  # type: ignore[union-attr]
            dsn,
            connect_timeout=10,
            row_factory=dict_row,
        )

    def _execute(self, conn: Any, query: str, args: tuple[Any, ...] = ()) -> Any:
        return conn.execute(query, args)

    def init(self) -> None:
        """Validate schema. Migrations are expected to be applied via Alembic."""
        if not self.enabled():
            return
        try:
            with self._lock:
                with self._connect() as conn:
                    cur = self._execute(
                        conn,
                        "SELECT 1 FROM information_schema.tables "
                        "WHERE table_schema = current_schema() AND table_name = 'artifact_index';",
                    )
                    row = cur.fetchone()
                    if row is None:
                        raise RuntimeError(
                            "Table 'artifact_index' is missing. Run migrations first: alembic upgrade head"
                        )
        except Exception as e:
            try:
                self._log_event(
                    logging.WARNING,
                    "artifact_index_init_failed",
                    error=str(e),
                )
            except Exception:
                pass

    def upsert(
        self,
        *,
        kind: str,
        artifact_id: str,
        full_ref: str,
        rel_ref: str,
        day: Optional[str],
        owner_key_id: Optional[str],
    ) -> None:
        if not self.enabled():
            return
        try:
            self._ensure_init()
            created_at = datetime.now(timezone.utc).isoformat()
            # In S3 mode we store keys (no leading "/"). In local mode we store absolute paths.
            storage = "s3" if (self._is_s3_enabled() and full_ref and not str(full_ref).startswith("/")) else "local"
            with self._lock:
                with self._connect() as conn:
                    self._execute(
                        conn,
                        """
                        INSERT INTO artifact_index (kind, artifact_id, owner_key_id, day, storage, full_ref, rel_ref, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT(kind, artifact_id) DO UPDATE SET
                            owner_key_id=COALESCE(EXCLUDED.owner_key_id, artifact_index.owner_key_id),
                            day=EXCLUDED.day,
                            storage=EXCLUDED.storage,
                            full_ref=EXCLUDED.full_ref,
                            rel_ref=EXCLUDED.rel_ref,
                            created_at=EXCLUDED.created_at
                        """,
                        (kind, artifact_id, owner_key_id, day, storage, full_ref, rel_ref, created_at),
                    )
                    conn.commit()
        except Exception as e:
            try:
                self._log_event(
                    logging.WARNING,
                    "artifact_index_upsert_failed",
                    kind=kind,
                    artifact_id=artifact_id,
                    error=str(e),
                )
            except Exception:
                pass

    def get(self, *, kind: str, artifact_id: str) -> Optional[Dict[str, str]]:
        if not self.enabled():
            return None
        try:
            self._ensure_init()
            with self._lock:
                with self._connect() as conn:
                    row = self._execute(
                        conn,
                        "SELECT storage, full_ref, rel_ref, day, owner_key_id "
                        "FROM artifact_index WHERE kind=%s AND artifact_id=%s",
                        (kind, artifact_id),
                    ).fetchone()
            if not row:
                return None
            rr: Mapping[str, Any] = row
            return {
                "storage": str(rr.get("storage") or ""),
                "full_ref": str(rr.get("full_ref") or ""),
                "rel_ref": str(rr.get("rel_ref") or ""),
                "day": str(rr.get("day") or ""),
                "owner_key_id": str(rr.get("owner_key_id") or ""),
            }
        except Exception as e:
            try:
                self._log_event(
                    logging.WARNING,
                    "artifact_index_get_failed",
                    kind=kind,
                    artifact_id=artifact_id,
                    error=str(e),
                )
            except Exception:
                pass
            return None

    def delete(self, *, kind: str, artifact_id: str) -> None:
        if not self.enabled():
            return
        try:
            self._ensure_init()
            with self._lock:
                with self._connect() as conn:
                    self._execute(
                        conn,
                        "DELETE FROM artifact_index WHERE kind=%s AND artifact_id=%s",
                        (kind, artifact_id),
                    )
                    conn.commit()
        except Exception as e:
            try:
                self._log_event(
                    logging.WARNING,
                    "artifact_index_delete_failed",
                    kind=kind,
                    artifact_id=artifact_id,
                    error=str(e),
                )
            except Exception:
                pass

    def list(
        self,
        *,
        kind: str,
        limit: int,
        owner_key_id: Optional[str] = None,
        cursor_created_at: Optional[str] = None,
        cursor_artifact_id: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        if not self.enabled():
            return []
        try:
            self._ensure_init()
            lim = max(1, int(limit))

            q = (
                "SELECT kind, artifact_id, owner_key_id, day, storage, full_ref, rel_ref, created_at "
                "FROM artifact_index WHERE kind = %s"
            )
            args: List[Any] = [kind]

            if owner_key_id is not None:
                q += " AND owner_key_id = %s"
                args.append(owner_key_id)

            if cursor_created_at and cursor_artifact_id:
                q += " AND (created_at < %s OR (created_at = %s AND artifact_id < %s))"
                args.extend([cursor_created_at, cursor_created_at, cursor_artifact_id])

            q += " ORDER BY created_at DESC, artifact_id DESC LIMIT %s"
            args.append(lim)

            with self._lock:
                with self._connect() as conn:
                    rows = self._execute(conn, q, tuple(args)).fetchall()

            out: List[Dict[str, str]] = []
            for row in list(rows or []):
                rr: Mapping[str, Any] = row
                out.append(
                    {
                        "kind": str(rr.get("kind") or ""),
                        "artifact_id": str(rr.get("artifact_id") or ""),
                        "owner_key_id": str(rr.get("owner_key_id") or ""),
                        "day": str(rr.get("day") or ""),
                        "storage": str(rr.get("storage") or ""),
                        "full_ref": str(rr.get("full_ref") or ""),
                        "rel_ref": str(rr.get("rel_ref") or ""),
                        "created_at": str(rr.get("created_at") or ""),
                    }
                )
            return out
        except Exception as e:
            try:
                self._log_event(
                    logging.WARNING,
                    "artifact_index_list_failed",
                    kind=kind,
                    owner_key_id=owner_key_id,
                    error=str(e),
                )
            except Exception:
                pass
            return []
