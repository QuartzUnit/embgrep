"""CLI entry point for embgrep — embedding-powered grep for files."""

from __future__ import annotations

import sys


def main() -> None:
    """Main CLI entry point."""
    try:
        import click
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        print("CLI requires extra dependencies: pip install embgrep[cli]")
        sys.exit(1)

    console = Console()

    @click.group()
    @click.version_option(package_name="embgrep")
    def cli() -> None:
        """embgrep — Local semantic search, embedding-powered grep for files."""

    @cli.command()
    @click.argument("path", type=click.Path(exists=True))
    @click.option("--patterns", "-p", default=None, help="Comma-separated glob patterns (e.g., '*.md,*.py').")
    @click.option("--db-path", default=None, help="Path to SQLite database.")
    @click.option("--model", default="BAAI/bge-small-en-v1.5", help="Embedding model name.")
    def index(path: str, patterns: str | None, db_path: str | None, model: str) -> None:
        """Index files in PATH for semantic search."""
        from embgrep.indexer import EmbGrep

        pattern_list = [p.strip() for p in patterns.split(",")] if patterns else None

        kwargs: dict = {"model": model}
        if db_path:
            kwargs["db_path"] = db_path

        eg = EmbGrep(**kwargs)
        try:
            with console.status("[bold green]Indexing files..."):
                result = eg.index(path, patterns=pattern_list)
            console.print(f"[green]Indexed {result['files_indexed']} files, {result['chunks_created']} chunks[/green]")
            console.print(f"Index size: {result['index_size_mb']} MB")
        finally:
            eg.close()

    @cli.command()
    @click.argument("query")
    @click.option("--top-k", "-k", default=5, help="Number of results to return.")
    @click.option("--path-filter", "-f", default=None, help="SQL LIKE pattern for file path filter.")
    @click.option("--db-path", default=None, help="Path to SQLite database.")
    @click.option("--model", default="BAAI/bge-small-en-v1.5", help="Embedding model name.")
    def search(query: str, top_k: int, path_filter: str | None, db_path: str | None, model: str) -> None:
        """Semantic search across indexed files."""
        from embgrep.indexer import EmbGrep

        kwargs: dict = {"model": model}
        if db_path:
            kwargs["db_path"] = db_path

        eg = EmbGrep(**kwargs)
        try:
            with console.status("[bold green]Searching..."):
                results = eg.search(query, top_k=top_k, path_filter=path_filter)

            if not results:
                console.print("[yellow]No results found.[/yellow]")
                return

            table = Table(title=f"Search: {query!r}", show_lines=True)
            table.add_column("#", style="dim", width=3)
            table.add_column("Score", style="cyan", width=7)
            table.add_column("File", style="green")
            table.add_column("Lines", style="yellow", width=10)
            table.add_column("Preview", max_width=60)

            for i, r in enumerate(results, 1):
                preview = r.chunk_text[:120].replace("\n", " ").strip()
                if len(r.chunk_text) > 120:
                    preview += "..."
                table.add_row(
                    str(i),
                    f"{r.score:.4f}",
                    r.file_path,
                    f"{r.line_start}-{r.line_end}",
                    preview,
                )

            console.print(table)
        finally:
            eg.close()

    @cli.command()
    @click.option("--db-path", default=None, help="Path to SQLite database.")
    def status(db_path: str | None) -> None:
        """Show index statistics."""
        from embgrep.indexer import EmbGrep

        kwargs: dict = {}
        if db_path:
            kwargs["db_path"] = db_path

        eg = EmbGrep(**kwargs)
        try:
            st = eg.status()
            console.print("[bold]embgrep Index Status[/bold]")
            console.print(f"  Files:        {st.total_files}")
            console.print(f"  Chunks:       {st.total_chunks}")
            console.print(f"  Last updated: {st.last_updated}")
            console.print(f"  Index size:   {st.index_size_mb} MB")
        finally:
            eg.close()

    @cli.command()
    @click.option("--db-path", default=None, help="Path to SQLite database.")
    @click.option("--model", default="BAAI/bge-small-en-v1.5", help="Embedding model name.")
    def update(db_path: str | None, model: str) -> None:
        """Incremental update — re-index changed files only."""
        from embgrep.indexer import EmbGrep

        kwargs: dict = {"model": model}
        if db_path:
            kwargs["db_path"] = db_path

        eg = EmbGrep(**kwargs)
        try:
            with console.status("[bold green]Updating index..."):
                result = eg.update()
            console.print(f"[green]Updated {result['updated_files']} files, {result['new_chunks']} new chunks[/green]")
            if result["removed_files"]:
                console.print(f"[yellow]Removed {result['removed_files']} deleted files[/yellow]")
        finally:
            eg.close()

    cli()


if __name__ == "__main__":
    main()
