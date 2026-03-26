# embgrep

> [한국어 문서](README.ko.md)

**Local semantic search — embedding-powered grep for files, zero external services.**

[![PyPI](https://img.shields.io/pypi/v/embgrep)](https://pypi.org/project/embgrep/)
[![Python](https://img.shields.io/pypi/pyversions/embgrep)](https://pypi.org/project/embgrep/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Search your codebase and documentation by *meaning*, not just keywords. embgrep indexes files into local embeddings and lets you run semantic queries — no API keys, no cloud services, no vector database servers.

## Features

- **Local embeddings** — Uses [fastembed](https://github.com/qdrant/fastembed) (ONNX Runtime), no API keys needed
- **SQLite storage** — Single-file index, no external vector DB
- **Incremental indexing** — Only re-indexes changed files (SHA-256 hash comparison)
- **Smart chunking** — Function-level splitting for code, heading-level for docs
- **MCP native** — 4-tool FastMCP server for LLM agent integration
- **15+ file types** — `.py`, `.js`, `.ts`, `.java`, `.go`, `.rs`, `.md`, `.txt`, `.yaml`, `.json`, `.toml`, and more

## Install

```bash
pip install embgrep              # core (fastembed + numpy)
pip install embgrep[cli]         # + click/rich CLI
pip install embgrep[mcp]         # + FastMCP server
pip install embgrep[all]         # everything
```

## Quick Start

### Python API

```python
from embgrep import EmbGrep

eg = EmbGrep()

# Index a directory
eg.index("./my-project", patterns=["*.py", "*.md"])

# Semantic search
results = eg.search("database connection pooling", top_k=5)
for r in results:
    print(f"{r.file_path}:{r.line_start}-{r.line_end} (score: {r.score:.4f})")
    print(f"  {r.chunk_text[:80]}...")

# Incremental update (only changed files)
eg.update()

# Index statistics
status = eg.status()
print(f"{status.total_files} files, {status.total_chunks} chunks, {status.index_size_mb} MB")

eg.close()
```

### CLI

```bash
# Index a project
embgrep index ./my-project --patterns "*.py,*.md"

# Search
embgrep search "error handling patterns"

# Filter by file type
embgrep search "async database query" --path-filter "%.py"

# Check status
embgrep status

# Update changed files
embgrep update
```

### Convenience functions

```python
import embgrep

embgrep.index("./src")
results = embgrep.search("authentication middleware")
status = embgrep.status()
embgrep.update()
```

## MCP Server

Add to your Claude Desktop / MCP client configuration:

```json
{
  "mcpServers": {
    "embgrep": {
      "command": "embgrep-mcp"
    }
  }
}
```

Or with uvx:

```json
{
  "mcpServers": {
    "embgrep": {
      "command": "uvx",
      "args": ["--from", "embgrep[mcp]", "embgrep-mcp"]
    }
  }
}
```

### MCP Tools

| Tool | Description |
|------|-------------|
| `index_directory` | Index files in a directory for semantic search |
| `semantic_search` | Search indexed files using natural language |
| `index_status` | Get current index statistics |
| `update_index` | Incremental update — re-index changed files only |

## How It Works

1. **Chunking** — Files are split into semantically meaningful chunks:
   - Code files (`.py`, `.js`, `.ts`, etc.): split by function/class boundaries
   - Documents (`.md`, `.txt`): split by headings or paragraph breaks
   - Config files: fixed-size chunking

2. **Embedding** — Each chunk is converted to a 384-dimensional vector using [BGE-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) via ONNX Runtime (no PyTorch needed)

3. **Storage** — Embeddings are stored as BLOBs in a local SQLite database

4. **Search** — Query text is embedded and compared against all chunks using cosine similarity

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `db_path` | `~/.local/share/embgrep/embgrep.db` | SQLite database location |
| `model` | `BAAI/bge-small-en-v1.5` | fastembed model name |
| `max_chunk_size` | 1000 chars | Maximum chunk size for fixed-size splitting |
| `top_k` | 5 | Number of search results |

## QuartzUnit Ecosystem

| Package | Description |
|---------|-------------|
| [markgrab](https://github.com/QuartzUnit/markgrab) | HTML/YouTube/PDF/DOCX to LLM-ready markdown |
| [snapgrab](https://github.com/QuartzUnit/snapgrab) | URL to screenshot + metadata |
| [docpick](https://github.com/QuartzUnit/docpick) | OCR + LLM document structure extraction |
| [browsegrab](https://github.com/QuartzUnit/browsegrab) | Local LLM browser agent |
| [feedkit](https://github.com/QuartzUnit/feedkit) | RSS feed collection + MCP |
| **embgrep** | **Local semantic search for files** |

## Used in

- [newswatch](https://github.com/QuartzUnit/newswatch) — RSS news monitoring pipeline (feedkit → markgrab → embgrep → diffgrab)

## License

MIT

<!-- mcp-name: io.github.QuartzUnit/embgrep -->
