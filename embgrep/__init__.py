"""embgrep — Local semantic search, embedding-powered grep for files."""

from __future__ import annotations

from embgrep.indexer import EmbGrep, IndexStatus, SearchResult

__all__ = ["EmbGrep", "IndexStatus", "SearchResult", "index", "search", "status", "update"]
__version__ = "0.1.0"


def index(directory: str, patterns: list[str] | None = None, db_path: str | None = None) -> dict:
    """Index files in a directory.

    Args:
        directory: Path to the directory to index.
        patterns: Optional list of glob patterns to filter files.
        db_path: Optional path to the SQLite database.

    Returns:
        Dictionary with files_indexed, chunks_created, index_size_mb.
    """
    eg = EmbGrep(db_path=db_path) if db_path else EmbGrep()
    try:
        return eg.index(directory, patterns=patterns)
    finally:
        eg.close()


def search(
    query: str, top_k: int = 5, path_filter: str | None = None, db_path: str | None = None
) -> list[SearchResult]:
    """Semantic search across indexed files.

    Args:
        query: Natural language search query.
        top_k: Number of results to return.
        path_filter: Optional LIKE pattern to filter by file path.
        db_path: Optional path to the SQLite database.

    Returns:
        List of SearchResult sorted by similarity score.
    """
    eg = EmbGrep(db_path=db_path) if db_path else EmbGrep()
    try:
        return eg.search(query, top_k=top_k, path_filter=path_filter)
    finally:
        eg.close()


def status(db_path: str | None = None) -> IndexStatus:
    """Get index statistics.

    Args:
        db_path: Optional path to the SQLite database.

    Returns:
        IndexStatus with total_files, total_chunks, last_updated, index_size_mb.
    """
    eg = EmbGrep(db_path=db_path) if db_path else EmbGrep()
    try:
        return eg.status()
    finally:
        eg.close()


def update(db_path: str | None = None) -> dict:
    """Incremental update — re-index changed files only.

    Args:
        db_path: Optional path to the SQLite database.

    Returns:
        Dictionary with updated_files, new_chunks, removed_files.
    """
    eg = EmbGrep(db_path=db_path) if db_path else EmbGrep()
    try:
        return eg.update()
    finally:
        eg.close()
