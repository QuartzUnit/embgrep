"""Tests for embgrep.db — SQLite storage layer."""

from __future__ import annotations

import os
import sqlite3

import numpy as np
import pytest

from embgrep.db import Database


@pytest.fixture()
def db(tmp_path):
    """Create a temporary database."""
    db_path = str(tmp_path / "test.db")
    database = Database(db_path)
    yield database
    database.close()


class TestFileOperations:
    def test_insert_file(self, db):
        file_id = db.insert_file("/tmp/test.py", "abc123")
        assert isinstance(file_id, int)
        assert file_id > 0

    def test_get_file(self, db):
        db.insert_file("/tmp/test.py", "abc123")
        row = db.get_file("/tmp/test.py")
        assert row is not None
        assert row[1] == "/tmp/test.py"
        assert row[2] == "abc123"
        assert row[4] == 0  # chunk_count

    def test_get_file_not_found(self, db):
        result = db.get_file("/nonexistent/file.py")
        assert result is None

    def test_insert_duplicate_file_raises(self, db):
        db.insert_file("/tmp/test.py", "abc123")
        with pytest.raises(sqlite3.IntegrityError):
            db.insert_file("/tmp/test.py", "def456")

    def test_update_file_hash(self, db):
        file_id = db.insert_file("/tmp/test.py", "abc123")
        db.update_file_hash(file_id, "new_hash")
        row = db.get_file("/tmp/test.py")
        assert row[2] == "new_hash"

    def test_delete_file(self, db):
        file_id = db.insert_file("/tmp/test.py", "abc123")
        db.delete_file(file_id)
        assert db.get_file("/tmp/test.py") is None

    def test_file_count(self, db):
        assert db.file_count() == 0
        db.insert_file("/tmp/a.py", "h1")
        db.insert_file("/tmp/b.py", "h2")
        assert db.file_count() == 2

    def test_get_all_files(self, db):
        db.insert_file("/tmp/a.py", "h1")
        db.insert_file("/tmp/b.py", "h2")
        files = db.get_all_files()
        assert len(files) == 2
        paths = {f[1] for f in files}
        assert paths == {"/tmp/a.py", "/tmp/b.py"}

    def test_update_chunk_count(self, db):
        file_id = db.insert_file("/tmp/test.py", "abc123")
        db.update_chunk_count(file_id, 42)
        row = db.get_file("/tmp/test.py")
        assert row[4] == 42


class TestChunkOperations:
    def test_insert_and_get_chunks(self, db):
        file_id = db.insert_file("/tmp/test.py", "abc123")
        emb = np.random.rand(384).astype(np.float32).tobytes()
        count = db.insert_chunks(file_id, [("chunk text", 1, 10, emb)])
        assert count == 1

        chunks = db.get_chunks()
        assert len(chunks) == 1
        assert chunks[0][2] == "chunk text"  # chunk_text
        assert chunks[0][3] == 1  # line_start
        assert chunks[0][4] == 10  # line_end

    def test_chunk_count(self, db):
        assert db.chunk_count() == 0
        file_id = db.insert_file("/tmp/test.py", "abc123")
        emb = np.random.rand(384).astype(np.float32).tobytes()
        db.insert_chunks(file_id, [
            ("chunk1", 1, 5, emb),
            ("chunk2", 6, 10, emb),
        ])
        assert db.chunk_count() == 2

    def test_cascade_delete_chunks(self, db):
        """Deleting a file should cascade-delete its chunks."""
        file_id = db.insert_file("/tmp/test.py", "abc123")
        emb = np.random.rand(384).astype(np.float32).tobytes()
        db.insert_chunks(file_id, [("chunk1", 1, 5, emb)])
        assert db.chunk_count() == 1

        db.delete_file(file_id)
        assert db.chunk_count() == 0

    def test_delete_chunks_for_file(self, db):
        file_id = db.insert_file("/tmp/test.py", "abc123")
        emb = np.random.rand(384).astype(np.float32).tobytes()
        db.insert_chunks(file_id, [("chunk1", 1, 5, emb)])
        assert db.chunk_count() == 1

        db.delete_chunks_for_file(file_id)
        assert db.chunk_count() == 0
        # File record should still exist
        assert db.get_file("/tmp/test.py") is not None

    def test_get_chunks_with_path_filter(self, db):
        fid1 = db.insert_file("/tmp/test.py", "h1")
        fid2 = db.insert_file("/tmp/readme.md", "h2")
        emb = np.random.rand(384).astype(np.float32).tobytes()
        db.insert_chunks(fid1, [("py chunk", 1, 5, emb)])
        db.insert_chunks(fid2, [("md chunk", 1, 5, emb)])

        py_chunks = db.get_chunks(path_filter="%.py")
        assert len(py_chunks) == 1
        assert py_chunks[0][2] == "py chunk"

        md_chunks = db.get_chunks(path_filter="%.md")
        assert len(md_chunks) == 1
        assert md_chunks[0][2] == "md chunk"

    def test_embedding_roundtrip(self, db):
        """Verify numpy embedding survives serialize/deserialize via SQLite BLOB."""
        file_id = db.insert_file("/tmp/test.py", "abc123")
        original = np.random.rand(384).astype(np.float32)
        blob = original.tobytes()
        db.insert_chunks(file_id, [("text", 1, 1, blob)])

        chunks = db.get_chunks()
        loaded = np.frombuffer(chunks[0][5], dtype=np.float32)
        np.testing.assert_array_almost_equal(original, loaded)


class TestStatistics:
    def test_last_updated_empty(self, db):
        assert db.last_updated() is None

    def test_last_updated_after_insert(self, db):
        db.insert_file("/tmp/test.py", "abc123")
        ts = db.last_updated()
        assert ts is not None
        assert len(ts) > 0

    def test_db_size_mb(self, db):
        size = db.db_size_mb()
        assert isinstance(size, float)
        assert size >= 0


class TestDatabaseCreation:
    def test_creates_parent_directory(self, tmp_path):
        db_path = str(tmp_path / "subdir" / "deep" / "test.db")
        database = Database(db_path)
        try:
            assert os.path.isfile(db_path)
        finally:
            database.close()
