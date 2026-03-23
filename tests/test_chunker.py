"""Tests for embgrep.chunker — file chunking strategies."""

from __future__ import annotations

import pytest

from embgrep.chunker import SUPPORTED_EXTENSIONS, chunk_file


@pytest.fixture()
def _write_file(tmp_path):
    """Helper to write a file with given content and extension."""
    def _inner(content: str, ext: str = ".py") -> str:
        path = str(tmp_path / f"test_file{ext}")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path
    return _inner


class TestPythonChunking:
    def test_split_by_functions(self, _write_file):
        content = '''import os

def hello():
    print("hello")

def world():
    print("world")

class Foo:
    pass
'''
        chunks = chunk_file(_write_file(content, ".py"))
        assert len(chunks) >= 3  # imports + hello + world + Foo (world+Foo may merge)
        # First chunk should contain import
        assert "import" in chunks[0]["text"]

    def test_function_boundaries_have_line_numbers(self, _write_file):
        content = '''def foo():
    pass

def bar():
    pass
'''
        chunks = chunk_file(_write_file(content, ".py"))
        assert len(chunks) == 2
        assert chunks[0]["line_start"] == 1
        assert chunks[0]["line_end"] == 3
        assert chunks[1]["line_start"] == 4
        assert chunks[1]["line_end"] == 5

    def test_async_def_detected(self, _write_file):
        content = '''async def handler():
    await something()

def sync_func():
    pass
'''
        chunks = chunk_file(_write_file(content, ".py"))
        assert len(chunks) == 2
        assert "async def" in chunks[0]["text"]

    def test_no_functions_fallback(self, _write_file):
        content = "# just a comment\nx = 1\ny = 2\n"
        chunks = chunk_file(_write_file(content, ".py"))
        assert len(chunks) >= 1


class TestMarkdownChunking:
    def test_split_by_headings(self, _write_file):
        content = """# Title

Some intro text.

## Section 1

Content 1.

## Section 2

Content 2.
"""
        chunks = chunk_file(_write_file(content, ".md"))
        assert len(chunks) == 3  # Title + Section 1 + Section 2
        assert "# Title" in chunks[0]["text"]
        assert "## Section 1" in chunks[1]["text"]
        assert "## Section 2" in chunks[2]["text"]

    def test_no_headings_fallback(self, _write_file):
        content = "Just some text.\nMore text.\nEven more.\n"
        chunks = chunk_file(_write_file(content, ".md"))
        assert len(chunks) >= 1

    def test_heading_line_numbers(self, _write_file):
        content = """# First

Text.

# Second

More text.
"""
        chunks = chunk_file(_write_file(content, ".md"))
        assert chunks[0]["line_start"] == 1
        assert chunks[1]["line_start"] == 5


class TestTxtChunking:
    def test_split_by_blank_lines(self, _write_file):
        content = """Paragraph one line one.
Paragraph one line two.

Paragraph two line one.
Paragraph two line two.

Paragraph three.
"""
        chunks = chunk_file(_write_file(content, ".txt"))
        assert len(chunks) >= 2

    def test_single_paragraph(self, _write_file):
        content = "A single paragraph with no blank lines.\n"
        chunks = chunk_file(_write_file(content, ".txt"))
        assert len(chunks) == 1


class TestJavaScriptChunking:
    def test_js_function_split(self, _write_file):
        content = """function hello() {
  console.log("hello");
}

function world() {
  console.log("world");
}
"""
        chunks = chunk_file(_write_file(content, ".js"))
        assert len(chunks) == 2

    def test_ts_export_split(self, _write_file):
        content = """export function main() {
  return 1;
}

export const helper = () => {
  return 2;
};
"""
        chunks = chunk_file(_write_file(content, ".ts"))
        assert len(chunks) == 2


class TestGoRustChunking:
    def test_go_func_split(self, _write_file):
        content = """func main() {
    fmt.Println("hello")
}

func helper() int {
    return 42
}
"""
        chunks = chunk_file(_write_file(content, ".go"))
        assert len(chunks) == 2

    def test_rust_fn_split(self, _write_file):
        content = """fn main() {
    println!("hello");
}

pub fn helper() -> i32 {
    42
}
"""
        chunks = chunk_file(_write_file(content, ".rs"))
        assert len(chunks) == 2


class TestFixedSizeChunking:
    def test_config_file_chunked(self, _write_file):
        content = "[section1]\nkey1 = value1\n\n[section2]\nkey2 = value2\n"
        chunks = chunk_file(_write_file(content, ".toml"))
        assert len(chunks) >= 1

    def test_shell_file_chunked(self, _write_file):
        content = "#!/bin/bash\necho hello\necho world\n"
        chunks = chunk_file(_write_file(content, ".sh"))
        assert len(chunks) >= 1

    def test_max_chunk_size_respected(self, _write_file):
        # Create content that exceeds max_chunk_size
        content = "\n".join([f"line {i}: {'x' * 50}" for i in range(100)])
        chunks = chunk_file(_write_file(content, ".cfg"), max_chunk_size=200)
        assert len(chunks) > 1
        for chunk in chunks:
            # Each chunk text should be roughly within max_chunk_size
            assert len(chunk["text"]) < 500  # generous bound


class TestEdgeCases:
    def test_empty_file(self, _write_file):
        chunks = chunk_file(_write_file("", ".py"))
        assert chunks == []

    def test_whitespace_only_file(self, _write_file):
        chunks = chunk_file(_write_file("   \n\n  \n", ".py"))
        assert chunks == []

    def test_nonexistent_file(self):
        chunks = chunk_file("/nonexistent/file.py")
        assert chunks == []

    def test_supported_extensions_complete(self):
        expected = {
            ".md", ".txt", ".py", ".js", ".ts", ".java", ".go", ".rs",
            ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".sh", ".bash",
        }
        assert expected == SUPPORTED_EXTENSIONS

    def test_chunk_has_required_keys(self, _write_file):
        content = "def foo():\n    pass\n"
        chunks = chunk_file(_write_file(content, ".py"))
        assert len(chunks) >= 1
        for chunk in chunks:
            assert "text" in chunk
            assert "line_start" in chunk
            assert "line_end" in chunk
            assert isinstance(chunk["text"], str)
            assert isinstance(chunk["line_start"], int)
            assert isinstance(chunk["line_end"], int)

    def test_line_numbers_are_1_based(self, _write_file):
        content = "def foo():\n    pass\n"
        chunks = chunk_file(_write_file(content, ".py"))
        assert chunks[0]["line_start"] >= 1
