"""MemoryAtlas CLI — manage memories from the terminal."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="memory-atlas",
    help="Game-engine inspired memory management for AI agents.",
    no_args_is_help=True,
)


def _get_engine(storage: str):
    from memory_atlas.engine import MemoryEngine
    return MemoryEngine(storage_path=storage, embedding_model="local")


@app.command()
def init(
    storage: str = typer.Option("./memory_atlas_data", "--storage", "-s", help="Storage path"),
):
    """Initialize a new memory store."""
    path = Path(storage)
    if (path / "index.duckdb").exists():
        typer.echo(f"Store already exists at {storage}")
        raise typer.Exit(1)
    engine = _get_engine(storage)
    engine.close()
    typer.echo(f"Initialized memory store at {storage}")


@app.command()
def ingest(
    content: str = typer.Argument(..., help="Content to ingest"),
    storage: str = typer.Option("./memory_atlas_data", "--storage", "-s"),
    session: str = typer.Option("", "--session", help="Session ID"),
):
    """Ingest content into memory."""
    engine = _get_engine(storage)
    ids = engine.ingest(content, session_id=session)
    engine.close()
    typer.echo(f"Ingested {len(ids)} memories: {', '.join(ids)}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    storage: str = typer.Option("./memory_atlas_data", "--storage", "-s"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
):
    """Search memories."""
    engine = _get_engine(storage)
    results = engine.retrieve(query, top_k=top_k)
    engine.close()
    if not results:
        typer.echo("No memories found.")
        return
    for mem in results:
        typer.echo(f"  [{mem.lod}] {mem.id}: {mem.display_text[:100]}")


@app.command()
def stats(
    storage: str = typer.Option("./memory_atlas_data", "--storage", "-s"),
):
    """Show memory store statistics."""
    engine = _get_engine(storage)
    s = engine.stats()
    engine.close()
    for k, v in s.items():
        typer.echo(f"  {k}: {v}")


@app.command()
def forget(
    storage: str = typer.Option("./memory_atlas_data", "--storage", "-s"),
    limit: int = typer.Option(500, "--limit", "-l"),
):
    """Run forgetting cycle: compress/archive low-activity memories."""
    engine = _get_engine(storage)
    result = engine.forget(limit=limit)
    engine.close()
    typer.echo(
        f"Scanned {result.scanned}: kept {result.kept}, "
        f"compressed {result.compressed}, archived {result.archived}"
    )


@app.command(name="export")
def export_cmd(
    output: str = typer.Argument(..., help="Output JSON file path"),
    storage: str = typer.Option("./memory_atlas_data", "--storage", "-s"),
    session: Optional[str] = typer.Option(None, "--session", help="Filter by session"),
):
    """Export memories to JSON."""
    engine = _get_engine(storage)
    result = engine.export_memories(output, session_id=session)
    engine.close()
    typer.echo(f"Exported {result['memories_exported']} memories to {output}")


@app.command(name="import")
def import_cmd(
    input_file: str = typer.Argument(..., help="Input JSON file path"),
    storage: str = typer.Option("./memory_atlas_data", "--storage", "-s"),
    mode: str = typer.Option("merge", "--mode", "-m", help="merge or overwrite"),
):
    """Import memories from JSON."""
    engine = _get_engine(storage)
    result = engine.import_memories(input_file, mode=mode)
    engine.close()
    typer.echo(
        f"Imported {result['imported']}, skipped {result['skipped']}, "
        f"overwritten {result['overwritten']}"
    )


@app.command()
def clusters(
    storage: str = typer.Option("./memory_atlas_data", "--storage", "-s"),
):
    """List all memory clusters."""
    engine = _get_engine(storage)
    cls_list = engine.cluster_mgr.list_clusters()
    engine.close()
    if not cls_list:
        typer.echo("No clusters.")
        return
    for c in cls_list:
        typer.echo(f"  {c.id}: {c.name} ({len(c.memory_ids)} memories) tags={c.entity_tags}")


if __name__ == "__main__":
    app()
