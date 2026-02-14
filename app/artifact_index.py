"""
Artifact index (SQLite) for fast lookup of artifacts by id without scanning date prefixes.

- Stores only metadata (no blobs). The artifacts themselves remain in filesystem/S3.
- Best-effort: indexing failures must never break request handling.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class ArtifactIndex:
    def __init__(
        self,
        *,
        enabled: bool,
        db_path: Path,
        log_event: Callable[..., None],
        is_s3_enabled: Callable[[], bool] | None = None,
    ) -> None:
        self._initialized = False
        self._enabled = bool(enabled)
        self._db_path = Path(db_path)
        self._log_event = log_event
        self._is_s3_enabled = is_s3_enabled or (lambda: False)
        self._lock = threading.Lock()

    def enabled(self) -> bool:
        return bool(self._enabled)
    
    def _ensure_init(self) -> None:
        if self._initialized:
            return
        self.init()
        self._initialized = True

    @property
    def db_path(self) -> Path:
        return self._db_path

    def init(self) -> None:
        """Initialize DB schema. Safe to call multiple times."""
        if not self.enabled():
            return
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                conn = sqlite3.connect(str(self._db_path), timeout=30.0)
                try:
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute("PRAGMA synchronous=NORMAL;")
                    conn.execute("PRAGMA busy_timeout=30000;")
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS artifact_index (
                            kind TEXT NOT NULL,
                            artifact_id TEXT NOT NULL,
                            owner_key_id TEXT,
                            day TEXT,
                            storage TEXT NOT NULL,
                            full_ref TEXT NOT NULL,
                            rel_ref TEXT NOT NULL,
                            created_at TEXT NOT NULL,
                            PRIMARY KEY (kind, artifact_id)
                        );
                        """
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_artifact_index_owner_kind_day "
                        "ON artifact_index(owner_key_id, kind, day);"
                    )
                    conn.commit()
                finally:
                    conn.close()
        except Exception as e:
            # Best-effort: do not fail app startup.
            try:
                self._log_event(
                    logging.WARNING,
                    "artifact_index_init_failed",
                    error=str(e),
                    db_path=str(self._db_path),
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
                conn = sqlite3.connect(str(self._db_path), timeout=30.0)
                try:
                    conn.execute("PRAGMA busy_timeout=30000;")
                    conn.execute(
                        """
                        INSERT INTO artifact_index (kind, artifact_id, owner_key_id, day, storage, full_ref, rel_ref, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(kind, artifact_id) DO UPDATE SET
                            owner_key_id=COALESCE(excluded.owner_key_id, artifact_index.owner_key_id),
                            day=excluded.day,
                            storage=excluded.storage,
                            full_ref=excluded.full_ref,
                            rel_ref=excluded.rel_ref,
                            created_at=excluded.created_at
                        """,
                        (kind, artifact_id, owner_key_id, day, storage, full_ref, rel_ref, created_at),
                    )
                    conn.commit()
                finally:
                    conn.close()
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
                conn = sqlite3.connect(str(self._db_path), timeout=30.0)
                try:
                    conn.execute("PRAGMA busy_timeout=30000;")
                    row = conn.execute(
                        "SELECT storage, full_ref, rel_ref, day, owner_key_id "
                        "FROM artifact_index WHERE kind=? AND artifact_id=?",
                        (kind, artifact_id),
                    ).fetchone()
                finally:
                    conn.close()
            if not row:
                return None
            storage, full_ref, rel_ref, day, owner_key_id = row
            return {
                "storage": str(storage or ""),
                "full_ref": str(full_ref or ""),
                "rel_ref": str(rel_ref or ""),
                "day": str(day or ""),
                "owner_key_id": str(owner_key_id or ""),
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
                conn = sqlite3.connect(str(self._db_path), timeout=30.0)
                try:
                    conn.execute("PRAGMA busy_timeout=30000;")
                    conn.execute("DELETE FROM artifact_index WHERE kind=? AND artifact_id=?", (kind, artifact_id))
                    conn.commit()
                finally:
                    conn.close()
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
