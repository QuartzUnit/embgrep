"""Tests for embgrep.embedder — fastembed wrapper (fully mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from embgrep.embedder import EMBEDDING_DIM, Embedder


def _fake_embed(texts):
    """Generate fake embeddings matching fastembed's API."""
    for _text in texts:
        yield np.random.rand(EMBEDDING_DIM).astype(np.float32)


@pytest.fixture()
def mock_fastembed():
    """Mock fastembed.TextEmbedding entirely."""
    mock_model = MagicMock()
    mock_model.embed = _fake_embed
    mock_cls = MagicMock(return_value=mock_model)
    with patch.dict("sys.modules", {"fastembed": MagicMock(TextEmbedding=mock_cls)}):
        yield mock_cls


class TestEmbedder:
    def test_lazy_init(self):
        """Model should not be loaded at construction time."""
        embedder = Embedder()
        assert embedder._model is None

    def test_embed_loads_model(self, mock_fastembed):
        embedder = Embedder()
        result = embedder.embed(["hello world"])
        assert embedder._model is not None
        assert len(result) == 1
        assert result[0].shape == (EMBEDDING_DIM,)
        assert result[0].dtype == np.float32

    def test_embed_multiple_texts(self, mock_fastembed):
        embedder = Embedder()
        texts = ["text one", "text two", "text three"]
        result = embedder.embed(texts)
        assert len(result) == 3
        for emb in result:
            assert emb.shape == (EMBEDDING_DIM,)
            assert emb.dtype == np.float32

    def test_embed_empty_list(self, mock_fastembed):
        embedder = Embedder()
        result = embedder.embed([])
        assert result == []
        # Model should not be loaded for empty input
        assert embedder._model is None

    def test_embed_query(self, mock_fastembed):
        embedder = Embedder()
        result = embedder.embed_query("search query")
        assert result.shape == (EMBEDDING_DIM,)
        assert result.dtype == np.float32

    def test_custom_model_name(self, mock_fastembed):
        embedder = Embedder(model="custom/model-name")
        embedder.embed(["test"])
        mock_fastembed.assert_called_once_with("custom/model-name")
