"""FastMCP server for embgrep — 4 semantic search tools."""

from __future__ import annotations

import json


def _create_server():
    """Create and configure the FastMCP server."""
    try:
        from fastmcp import FastMCP
    except ImportError:
        msg = "MCP server requires extra dependencies: pip install embgrep[mcp]"
        raise ImportError(msg)  # noqa: B904

    from embgrep.indexer import EmbGrep

    mcp = FastMCP("embgrep", instructions="Local semantic search — embedding-powered grep for files.")

    def _get_embgrep() -> EmbGrep:
        return EmbGrep()

    @mcp.tool()
    def index_directory(path: str, patterns: str = "*.md,*.py,*.txt") -> str:
        """Index files in a directory for semantic search.

        Args:
            path: Directory path to index.
            patterns: Comma-separated glob patterns (default: "*.md,*.py,*.txt").

        Returns:
            JSON string with indexing results.
        """
        pattern_list = [p.strip() for p in patterns.split(",")]
        eg = _get_embgrep()
        try:
            result = eg.index(path, patterns=pattern_list)
            return json.dumps(result, indent=2)
        finally:
            eg.close()

    @mcp.tool()
    def semantic_search(query: str, top_k: int = 5, path_filter: str | None = None) -> str:
        """Search indexed files using natural language.

        Args:
            query: Natural language search query.
            top_k: Number of results to return (default: 5).
            path_filter: Optional SQL LIKE pattern to filter by file path.

        Returns:
            JSON string with search results.
        """
        eg = _get_embgrep()
        try:
            results = eg.search(query, top_k=top_k, path_filter=path_filter)
            return json.dumps(
                [
                    {
                        "file_path": r.file_path,
                        "score": round(r.score, 4),
                        "line_start": r.line_start,
                        "line_end": r.line_end,
                        "chunk_text": r.chunk_text[:500],
                    }
                    for r in results
                ],
                indent=2,
            )
        finally:
            eg.close()

    @mcp.tool()
    def index_status() -> str:
        """Get current index statistics.

        Returns:
            JSON string with index status information.
        """
        eg = _get_embgrep()
        try:
            st = eg.status()
            return json.dumps(
                {
                    "total_files": st.total_files,
                    "total_chunks": st.total_chunks,
                    "last_updated": st.last_updated,
                    "index_size_mb": st.index_size_mb,
                },
                indent=2,
            )
        finally:
            eg.close()

    @mcp.tool()
    def update_index() -> str:
        """Incremental update — re-index changed files only (hash comparison).

        Returns:
            JSON string with update results.
        """
        eg = _get_embgrep()
        try:
            result = eg.update()
            return json.dumps(result, indent=2)
        finally:
            eg.close()

    return mcp


def main() -> None:
    """Run the MCP server."""
    server = _create_server()
    server.run()


if __name__ == "__main__":
    main()
