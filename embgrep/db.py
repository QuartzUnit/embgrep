"""SQLite storage for embgrep — files and chunks tables."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS indexed_files (
    id INTEGER PRIMARY KEY,
    file_path TEXT UNIQUE NOT NULL,
    file_hash TEXT NOT NULL,
    indexed_at TEXT DEFAULT CURRENT_TIMESTAMP,
    chunk_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    FOREIGN KEY (file_id) REFERENCES indexed_files(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_files_path ON indexed_files(file_path);
"""

DEFAULT_DB_PATH = "~/.local/share/embgrep/embgrep.db"


class Database:
    """SQLite database wrapper for embgrep index storage."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = os.path.expanduser(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # --- indexed_files operations ---

    def insert_file(self, file_path: str, file_hash: str) -> int:
        """Insert a file record and return its id."""
        cur = self._conn.execute(
            "INSERT INTO indexed_files (file_path, file_hash, chunk_count) VALUES (?, ?, 0)",
            (file_path, file_hash),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_file(self, file_path: str) -> tuple[int, str, str, str, int] | None:
        """Get file record by path. Returns (id, file_path, file_hash, indexed_at, chunk_count) or None."""
        cur = self._conn.execute("SELECT id, file_path, file_hash, indexed_at, chunk_count FROM indexed_files WHERE file_path = ?", (file_path,))
        return cur.fetchone()

    def get_all_files(self) -> list[tuple[int, str, str, str, int]]:
        """Get all file records."""
        cur = self._conn.execute("SELECT id, file_path, file_hash, indexed_at, chunk_count FROM indexed_files")
        return cur.fetchall()

    def update_file_hash(self, file_id: int, file_hash: str) -> None:
        """Update file hash and reset indexed_at."""
        self._conn.execute(
            "UPDATE indexed_files SET file_hash = ?, indexed_at = CURRENT_TIMESTAMP WHERE id = ?",
            (file_hash, file_id),
        )
        self._conn.commit()

    def update_chunk_count(self, file_id: int, count: int) -> None:
        """Update the chunk count for a file."""
        self._conn.execute("UPDATE indexed_files SET chunk_count = ? WHERE id = ?", (count, file_id))
        self._conn.commit()

    def delete_file(self, file_id: int) -> None:
        """Delete a file and its chunks (cascading)."""
        self._conn.execute("DELETE FROM indexed_files WHERE id = ?", (file_id,))
        self._conn.commit()

    def file_count(self) -> int:
        """Get total number of indexed files."""
        cur = self._conn.execute("SELECT COUNT(*) FROM indexed_files")
        return cur.fetchone()[0]

    # --- chunks operations ---

    def insert_chunks(self, file_id: int, chunks: list[tuple[str, int, int, bytes]]) -> int:
        """Insert multiple chunks for a file. Each chunk is (text, line_start, line_end, embedding_blob).

        Returns the number of chunks inserted.
        """
        self._conn.executemany(
            "INSERT INTO chunks (file_id, chunk_text, line_start, line_end, embedding) VALUES (?, ?, ?, ?, ?)",
            [(file_id, text, ls, le, emb) for text, ls, le, emb in chunks],
        )
        self._conn.commit()
        return len(chunks)

    def delete_chunks_for_file(self, file_id: int) -> None:
        """Delete all chunks for a file."""
        self._conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
        self._conn.commit()

    def get_chunks(
        self, path_filter: str | None = None
    ) -> list[tuple[int, str, str, int, int, bytes]]:
        """Get chunks with optional path filter.

        Returns list of (chunk_id, file_path, chunk_text, line_start, line_end, embedding_blob).
        """
        if path_filter:
            cur = self._conn.execute(
                """
                SELECT c.id, f.file_path, c.chunk_text, c.line_start, c.line_end, c.embedding
                FROM chunks c
                JOIN indexed_files f ON c.file_id = f.id
                WHERE f.file_path LIKE ?
                """,
                (path_filter,),
            )
        else:
            cur = self._conn.execute(
                """
                SELECT c.id, f.file_path, c.chunk_text, c.line_start, c.line_end, c.embedding
                FROM chunks c
                JOIN indexed_files f ON c.file_id = f.id
                """
            )
        return cur.fetchall()

    def chunk_count(self) -> int:
        """Get total number of chunks."""
        cur = self._conn.execute("SELECT COUNT(*) FROM chunks")
        return cur.fetchone()[0]

    # --- statistics ---

    def last_updated(self) -> str | None:
        """Get the most recent indexed_at timestamp."""
        cur = self._conn.execute("SELECT MAX(indexed_at) FROM indexed_files")
        row = cur.fetchone()
        return row[0] if row else None

    def db_size_mb(self) -> float:
        """Get database file size in MB."""
        try:
            return os.path.getsize(self.db_path) / (1024 * 1024)
        except OSError:
            return 0.0
