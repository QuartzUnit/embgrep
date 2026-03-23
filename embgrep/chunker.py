"""File chunking for embgrep — split files into semantically meaningful chunks."""

from __future__ import annotations

import re
from pathlib import Path

# File extensions grouped by chunking strategy
_CODE_EXTENSIONS = {".py", ".js", ".ts", ".java", ".go", ".rs"}
_DOC_EXTENSIONS = {".md", ".txt"}
_CONFIG_EXTENSIONS = {".yaml", ".yml", ".json", ".toml", ".cfg", ".ini"}
_SHELL_EXTENSIONS = {".sh", ".bash"}

SUPPORTED_EXTENSIONS = _CODE_EXTENSIONS | _DOC_EXTENSIONS | _CONFIG_EXTENSIONS | _SHELL_EXTENSIONS

# Regex patterns for detecting function/class boundaries per language
_CODE_BOUNDARY_PATTERNS: dict[str, re.Pattern[str]] = {
    ".py": re.compile(r"^(def |class |async def )", re.MULTILINE),
    ".js": re.compile(r"^(function |class |export |const \w+ = )", re.MULTILINE),
    ".ts": re.compile(r"^(function |class |export |const \w+ = )", re.MULTILINE),
    ".java": re.compile(r"^(public |private |class )", re.MULTILINE),
    ".go": re.compile(r"^(func |type )", re.MULTILINE),
    ".rs": re.compile(r"^(fn |pub fn |pub struct |struct |impl |pub impl )", re.MULTILINE),
}

# Markdown heading pattern
_MD_HEADING = re.compile(r"^#{1,6}\s", re.MULTILINE)


def chunk_file(file_path: str, max_chunk_size: int = 1000) -> list[dict]:
    """Split a file into chunks for embedding.

    Args:
        file_path: Path to the file to chunk.
        max_chunk_size: Maximum number of characters per chunk for fixed-size fallback.

    Returns:
        List of dicts with keys: text, line_start, line_end.

    Strategy:
        - .py/.js/.ts/.java/.go/.rs: split by function/class definitions (regex)
        - .md/.txt: split by headings (## or blank line groups)
        - Others: split by max_chunk_size characters
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except UnicodeDecodeError:
        try:
            content = path.read_text(encoding="latin-1")
        except Exception:
            return []

    if not content.strip():
        return []

    lines = content.splitlines(keepends=True)

    if suffix in _CODE_EXTENSIONS:
        chunks = _chunk_code(lines, suffix, max_chunk_size)
    elif suffix in _DOC_EXTENSIONS:
        chunks = _chunk_docs(lines, suffix, max_chunk_size)
    else:
        chunks = _chunk_fixed(lines, max_chunk_size)

    # Filter out empty chunks
    return [c for c in chunks if c["text"].strip()]


def _chunk_code(lines: list[str], suffix: str, max_chunk_size: int) -> list[dict]:
    """Split code files by function/class boundaries."""
    pattern = _CODE_BOUNDARY_PATTERNS.get(suffix)
    if pattern is None:
        return _chunk_fixed(lines, max_chunk_size)

    # Find boundary line numbers
    boundaries: list[int] = []
    for i, line in enumerate(lines):
        if pattern.match(line):
            boundaries.append(i)

    if not boundaries:
        return _chunk_fixed(lines, max_chunk_size)

    chunks: list[dict] = []

    # Lines before the first boundary (imports, module docstring, etc.)
    if boundaries[0] > 0:
        text = "".join(lines[: boundaries[0]])
        if text.strip():
            chunks.append({"text": text, "line_start": 1, "line_end": boundaries[0]})

    # Each boundary to the next
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(lines)
        text = "".join(lines[start:end])
        if text.strip():
            chunks.append({"text": text, "line_start": start + 1, "line_end": end})

    return chunks


def _chunk_docs(lines: list[str], suffix: str, max_chunk_size: int) -> list[dict]:
    """Split document files by headings or blank-line groups."""
    if suffix == ".md":
        return _chunk_markdown(lines, max_chunk_size)
    return _chunk_by_blank_lines(lines, max_chunk_size)


def _chunk_markdown(lines: list[str], max_chunk_size: int) -> list[dict]:
    """Split markdown by heading boundaries."""
    boundaries: list[int] = []
    for i, line in enumerate(lines):
        if _MD_HEADING.match(line):
            boundaries.append(i)

    if not boundaries:
        return _chunk_by_blank_lines(lines, max_chunk_size)

    chunks: list[dict] = []

    # Lines before the first heading
    if boundaries[0] > 0:
        text = "".join(lines[: boundaries[0]])
        if text.strip():
            chunks.append({"text": text, "line_start": 1, "line_end": boundaries[0]})

    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(lines)
        text = "".join(lines[start:end])
        if text.strip():
            chunks.append({"text": text, "line_start": start + 1, "line_end": end})

    return chunks


def _chunk_by_blank_lines(lines: list[str], max_chunk_size: int) -> list[dict]:
    """Split text by groups of blank lines."""
    chunks: list[dict] = []
    current_lines: list[str] = []
    start_line = 0

    for i, line in enumerate(lines):
        if not line.strip() and current_lines:
            # End of a paragraph
            text = "".join(current_lines)
            if len(text) > max_chunk_size:
                # Split large paragraphs
                sub_chunks = _split_text_fixed(current_lines, start_line, max_chunk_size)
                chunks.extend(sub_chunks)
            else:
                chunks.append({"text": text, "line_start": start_line + 1, "line_end": i})
            current_lines = []
            start_line = i + 1
        else:
            if not current_lines:
                start_line = i
            current_lines.append(line)

    # Remaining lines
    if current_lines:
        text = "".join(current_lines)
        if text.strip():
            chunks.append({"text": text, "line_start": start_line + 1, "line_end": len(lines)})

    return chunks


def _chunk_fixed(lines: list[str], max_chunk_size: int) -> list[dict]:
    """Split by fixed character size."""
    return _split_text_fixed(lines, 0, max_chunk_size)


def _split_text_fixed(lines: list[str], offset: int, max_chunk_size: int) -> list[dict]:
    """Split a list of lines into chunks of approximately max_chunk_size characters."""
    chunks: list[dict] = []
    current_lines: list[str] = []
    current_size = 0
    start_line = offset

    for i, line in enumerate(lines):
        if current_size + len(line) > max_chunk_size and current_lines:
            text = "".join(current_lines)
            chunks.append({"text": text, "line_start": start_line + 1, "line_end": offset + i})
            current_lines = [line]
            current_size = len(line)
            start_line = offset + i
        else:
            current_lines.append(line)
            current_size += len(line)

    if current_lines:
        text = "".join(current_lines)
        if text.strip():
            chunks.append({
                "text": text,
                "line_start": start_line + 1,
                "line_end": offset + len(lines),
            })

    return chunks
