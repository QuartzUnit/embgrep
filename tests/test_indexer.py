"""Tests for embgrep.indexer — EmbGrep orchestrator (with mocked embedder)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from embgrep.embedder import EMBEDDING_DIM
from embgrep.indexer import EmbGrep, IndexStatus, SearchResult, _cosine_similarity, _file_hash


def _fake_embed(texts):
    """Return deterministic fake embeddings."""
    return [np.random.RandomState(hash(t) % 2**31).rand(EMBEDDING_DIM).astype(np.float32) for t in texts]


def _fake_embed_query(query):
    """Return deterministic fake query embedding."""
    return np.random.RandomState(hash(query) % 2**31).rand(EMBEDDING_DIM).astype(np.float32)


@pytest.fixture()
def mock_embedder():
    """Mock the Embedder class."""
    with patch("embgrep.indexer.Embedder") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.embed = MagicMock(side_effect=_fake_embed)
        mock_instance.embed_query = MagicMock(side_effect=_fake_embed_query)
        mock_cls.return_value = mock_instance
        yield mock_instance


@pytest.fixture()
def embgrep(tmp_path, mock_embedder):
    """Create an EmbGrep instance with temp DB and mocked embedder."""
    db_path = str(tmp_path / "test.db")
    eg = EmbGrep(db_path=db_path)
    yield eg
    eg.close()


@pytest.fixture()
def sample_dir(tmp_path):
    """Create a sample directory with files to index."""
    d = tmp_path / "project"
    d.mkdir()

    (d / "main.py").write_text("def hello():\n    print('hello')\n\ndef world():\n    print('world')\n")
    (d / "readme.md").write_text("# My Project\n\nA great project.\n\n## Usage\n\nRun it.\n")
    (d / "config.toml").write_text("[settings]\nkey = 'value'\n")
    (d / "binary.dat").write_bytes(b"\x00\x01\x02\x03")  # unsupported extension

    return str(d)


class TestIndex:
    def test_index_directory(self, embgrep, sample_dir):
        result = embgrep.index(sample_dir)
        assert result["files_indexed"] >= 2  # at least .py and .md
        assert result["chunks_created"] > 0
        assert "index_size_mb" in result

    def test_index_with_patterns(self, embgrep, sample_dir):
        result = embgrep.index(sample_dir, patterns=["*.py"])
        assert result["files_indexed"] == 1

    def test_index_skips_unchanged_files(self, embgrep, sample_dir):
        embgrep.index(sample_dir)
        result2 = embgrep.index(sample_dir)
        # Second run should skip all files (unchanged)
        assert result2["files_indexed"] == 0
        assert result2["chunks_created"] == 0

    def test_index_reindexes_changed_files(self, embgrep, sample_dir):
        embgrep.index(sample_dir)
        # Modify a file
        with open(os.path.join(sample_dir, "main.py"), "a") as f:
            f.write("\ndef new_func():\n    pass\n")
        result = embgrep.index(sample_dir)
        assert result["files_indexed"] == 1

    def test_index_nonexistent_dir_raises(self, embgrep):
        with pytest.raises(FileNotFoundError):
            embgrep.index("/nonexistent/dir")


class TestSearch:
    def test_search_returns_results(self, embgrep, sample_dir):
        embgrep.index(sample_dir)
        results = embgrep.search("hello function")
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_result_fields(self, embgrep, sample_dir):
        embgrep.index(sample_dir)
        results = embgrep.search("hello")
        if results:
            r = results[0]
            assert isinstance(r.file_path, str)
            assert isinstance(r.chunk_text, str)
            assert isinstance(r.score, float)
            assert isinstance(r.line_start, int)
            assert isinstance(r.line_end, int)

    def test_search_top_k(self, embgrep, sample_dir):
        embgrep.index(sample_dir)
        results = embgrep.search("test", top_k=2)
        assert len(results) <= 2

    def test_search_with_path_filter(self, embgrep, sample_dir):
        embgrep.index(sample_dir)
        results = embgrep.search("test", path_filter="%.py")
        for r in results:
            assert r.file_path.endswith(".py")

    def test_search_empty_index(self, embgrep):
        results = embgrep.search("anything")
        assert results == []

    def test_search_results_sorted_by_score(self, embgrep, sample_dir):
        embgrep.index(sample_dir)
        results = embgrep.search("function definition", top_k=10)
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score


class TestUpdate:
    def test_update_detects_changed_file(self, embgrep, sample_dir):
        embgrep.index(sample_dir)
        # Modify a file
        with open(os.path.join(sample_dir, "main.py"), "w") as f:
            f.write("def changed():\n    pass\n")
        result = embgrep.update()
        assert result["updated_files"] >= 1

    def test_update_detects_deleted_file(self, embgrep, sample_dir):
        embgrep.index(sample_dir)
        os.remove(os.path.join(sample_dir, "main.py"))
        result = embgrep.update()
        assert result["removed_files"] >= 1

    def test_update_no_changes(self, embgrep, sample_dir):
        embgrep.index(sample_dir)
        result = embgrep.update()
        assert result["updated_files"] == 0
        assert result["removed_files"] == 0


class TestStatus:
    def test_status_empty_index(self, embgrep):
        st = embgrep.status()
        assert isinstance(st, IndexStatus)
        assert st.total_files == 0
        assert st.total_chunks == 0
        assert st.last_updated == "never"

    def test_status_after_indexing(self, embgrep, sample_dir):
        embgrep.index(sample_dir)
        st = embgrep.status()
        assert st.total_files >= 2
        assert st.total_chunks > 0
        assert st.last_updated != "never"
        assert st.index_size_mb >= 0


class TestHelpers:
    def test_file_hash_consistency(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h1 = _file_hash(str(f))
        h2 = _file_hash(str(f))
        assert h1 == h2

    def test_file_hash_changes_on_content_change(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        h1 = _file_hash(str(f))
        f.write_text("world")
        h2 = _file_hash(str(f))
        assert h1 != h2

    def test_cosine_similarity_identical(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert abs(_cosine_similarity(a, a) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_cosine_similarity_zero_vector(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        z = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        assert _cosine_similarity(a, z) == 0.0
