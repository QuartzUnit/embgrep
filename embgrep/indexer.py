"""EmbGrep — main orchestrator for indexing and semantic search."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from embgrep.chunker import SUPPORTED_EXTENSIONS, chunk_file
from embgrep.db import DEFAULT_DB_PATH, Database
from embgrep.embedder import EMBEDDING_DIM, Embedder


@dataclass
class SearchResult:
    """A single search result with file location and similarity score."""

    file_path: str
    chunk_text: str
    score: float
    line_start: int
    line_end: int


@dataclass
class IndexStatus:
    """Statistics about the current embgrep index."""

    total_files: int
    total_chunks: int
    last_updated: str
    index_size_mb: float


def _file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of file contents."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_files(directory: str, patterns: list[str] | None = None) -> list[str]:
    """Collect files from directory matching patterns or supported extensions."""
    root = Path(directory).resolve()
    if not root.is_dir():
        msg = f"Directory not found: {directory}"
        raise FileNotFoundError(msg)

    if patterns:
        files: list[str] = []
        for pattern in patterns:
            files.extend(str(p) for p in root.rglob(pattern) if p.is_file())
        return sorted(set(files))

    # Default: collect files with supported extensions
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(str(p))
    return sorted(files)


class EmbGrep:
    """Main orchestrator for embedding-powered file search.

    Args:
        db_path: Path to the SQLite database file.
        model: Name of the fastembed model to use.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        model: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self._db = Database(db_path)
        self._embedder = Embedder(model)

    def index(self, directory: str, patterns: list[str] | None = None) -> dict:
        """Index files in a directory.

        Args:
            directory: Path to the directory to index.
            patterns: Optional list of glob patterns to filter files (e.g., ["*.py", "*.md"]).

        Returns:
            Dict with keys: files_indexed, chunks_created, index_size_mb.
        """
        files = _collect_files(directory, patterns)

        files_indexed = 0
        chunks_created = 0

        for file_path in files:
            fhash = _file_hash(file_path)
            existing = self._db.get_file(file_path)

            if existing and existing[2] == fhash:
                # File unchanged, skip
                continue

            if existing:
                # File changed, remove old data
                self._db.delete_chunks_for_file(existing[0])
                self._db.delete_file(existing[0])

            # Index the file
            n_chunks = self._index_single_file(file_path, fhash)
            if n_chunks > 0:
                files_indexed += 1
                chunks_created += n_chunks

        return {
            "files_indexed": files_indexed,
            "chunks_created": chunks_created,
            "index_size_mb": round(self._db.db_size_mb(), 2),
        }

    def _index_single_file(self, file_path: str, fhash: str) -> int:
        """Index a single file: chunk, embed, store. Returns number of chunks created."""
        chunks = chunk_file(file_path)
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        embeddings = self._embedder.embed(texts)

        file_id = self._db.insert_file(file_path, fhash)

        chunk_records = []
        for chunk_data, emb in zip(chunks, embeddings, strict=True):
            blob = emb.astype(np.float32).tobytes()
            chunk_records.append((chunk_data["text"], chunk_data["line_start"], chunk_data["line_end"], blob))

        self._db.insert_chunks(file_id, chunk_records)
        self._db.update_chunk_count(file_id, len(chunk_records))

        return len(chunk_records)

    def search(self, query: str, top_k: int = 5, path_filter: str | None = None) -> list[SearchResult]:
        """Semantic search across indexed chunks.

        Args:
            query: Natural language search query.
            top_k: Number of top results to return.
            path_filter: Optional SQL LIKE pattern to filter by file path (e.g., "%.py").

        Returns:
            List of SearchResult sorted by descending similarity score.
        """
        query_emb = self._embedder.embed_query(query)
        db_chunks = self._db.get_chunks(path_filter=path_filter)

        if not db_chunks:
            return []

        results: list[SearchResult] = []

        for _chunk_id, file_path, chunk_text, line_start, line_end, emb_blob in db_chunks:
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            if emb.shape[0] != EMBEDDING_DIM:
                continue
            score = _cosine_similarity(query_emb, emb)
            results.append(SearchResult(
                file_path=file_path,
                chunk_text=chunk_text,
                score=float(score),
                line_start=line_start,
                line_end=line_end,
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def update(self) -> dict:
        """Incremental update — re-index changed files, remove deleted files.

        Returns:
            Dict with keys: updated_files, new_chunks, removed_files.
        """
        all_files = self._db.get_all_files()
        updated_files = 0
        new_chunks = 0
        removed_files = 0

        for file_id, file_path, stored_hash, _indexed_at, _chunk_count in all_files:
            if not os.path.isfile(file_path):
                # File was deleted
                self._db.delete_file(file_id)
                removed_files += 1
                continue

            current_hash = _file_hash(file_path)
            if current_hash != stored_hash:
                # File changed, re-index
                self._db.delete_chunks_for_file(file_id)
                self._db.delete_file(file_id)
                n_chunks = self._index_single_file(file_path, current_hash)
                updated_files += 1
                new_chunks += n_chunks

        return {
            "updated_files": updated_files,
            "new_chunks": new_chunks,
            "removed_files": removed_files,
        }

    def status(self) -> IndexStatus:
        """Get current index statistics.

        Returns:
            IndexStatus dataclass with summary information.
        """
        return IndexStatus(
            total_files=self._db.file_count(),
            total_chunks=self._db.chunk_count(),
            last_updated=self._db.last_updated() or "never",
            index_size_mb=round(self._db.db_size_mb(), 2),
        )

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
