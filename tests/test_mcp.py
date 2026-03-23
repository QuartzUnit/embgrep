"""Tests for embgrep.mcp_server — FastMCP tool registration (mocked)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from embgrep.embedder import EMBEDDING_DIM


def _fake_embed(texts):
    return [np.random.rand(EMBEDDING_DIM).astype(np.float32) for _t in texts]


def _fake_embed_query(query):
    return np.random.rand(EMBEDDING_DIM).astype(np.float32)


@pytest.fixture()
def mock_deps():
    """Mock EmbGrep at the source (embgrep.indexer) so mcp_server picks it up."""
    with patch("embgrep.indexer.EmbGrep") as mock_eg_cls:
        mock_eg = MagicMock()
        mock_eg.index.return_value = {"files_indexed": 5, "chunks_created": 20, "index_size_mb": 0.1}
        mock_eg.search.return_value = []
        mock_eg.status.return_value = MagicMock(
            total_files=5, total_chunks=20, last_updated="2026-01-01", index_size_mb=0.1
        )
        mock_eg.update.return_value = {"updated_files": 1, "new_chunks": 5, "removed_files": 0}
        mock_eg_cls.return_value = mock_eg

        yield mock_eg_cls, mock_eg


def _list_tools(server):
    """Helper to call async list_tools synchronously."""
    return asyncio.run(server.list_tools())


class TestMCPServerCreation:
    def test_server_creates_successfully(self):
        """Server should be importable and creatable."""
        with patch("embgrep.indexer.EmbGrep"):
            try:
                from embgrep.mcp_server import _create_server

                server = _create_server()
                assert server is not None
                assert server.name == "embgrep"
            except ImportError:
                pytest.skip("fastmcp not installed")


class TestMCPTools:
    def test_index_directory_tool(self, mock_deps):
        """Test index_directory tool is registered."""
        try:
            from embgrep.mcp_server import _create_server

            server = _create_server()
            tools = {t.name: t for t in _list_tools(server)}
            assert "index_directory" in tools
        except ImportError:
            pytest.skip("fastmcp not installed")

    def test_semantic_search_tool(self, mock_deps):
        """Test semantic_search tool exists."""
        try:
            from embgrep.mcp_server import _create_server

            server = _create_server()
            tools = {t.name: t for t in _list_tools(server)}
            assert "semantic_search" in tools
        except ImportError:
            pytest.skip("fastmcp not installed")

    def test_index_status_tool(self, mock_deps):
        """Test index_status tool exists."""
        try:
            from embgrep.mcp_server import _create_server

            server = _create_server()
            tools = {t.name: t for t in _list_tools(server)}
            assert "index_status" in tools
        except ImportError:
            pytest.skip("fastmcp not installed")

    def test_update_index_tool(self, mock_deps):
        """Test update_index tool exists."""
        try:
            from embgrep.mcp_server import _create_server

            server = _create_server()
            tools = {t.name: t for t in _list_tools(server)}
            assert "update_index" in tools
        except ImportError:
            pytest.skip("fastmcp not installed")

    def test_all_four_tools_registered(self, mock_deps):
        """Verify exactly 4 tools are registered."""
        try:
            from embgrep.mcp_server import _create_server

            server = _create_server()
            tools = _list_tools(server)
            assert len(tools) == 4
            names = {t.name for t in tools}
            assert names == {"index_directory", "semantic_search", "index_status", "update_index"}
        except ImportError:
            pytest.skip("fastmcp not installed")
