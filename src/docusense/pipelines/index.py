"""Indexing component entrypoint.

Reads text files from ``--corpus``, chunks them, computes embeddings,
and pushes them to Azure AI Search. Local variant writes a chunks.jsonl
manifest so CI can inspect it without touching Azure.
"""

from __future__ import annotations

from pathlib import Path

import typer

from docusense.classifier.embeddings import build_default_provider
from docusense.config import get_settings
from docusense.retrieval.chunker import chunk_documents
from docusense.retrieval.indexer import (
    dump_chunks,
    emit_manifest,
    ensure_search_index,
    read_corpus,
    upsert_to_ai_search,
)

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    corpus: Path = typer.Argument(...),
    output: Path = typer.Argument(...),
    push_to_search: str = typer.Argument("false"),
) -> None:
    """Index the corpus."""
    _push = push_to_search.lower() in ("true", "1", "yes")
    if not corpus.exists():
        raise typer.BadParameter(f"Corpus path does not exist: {corpus}")
    output.mkdir(parents=True, exist_ok=True)
    docs = read_corpus(corpus)
    chunks = chunk_documents(docs)
    dump_chunks(chunks, output / "chunks.jsonl")
    emit_manifest(output, chunks)

    if _push:
        settings = get_settings()
        if not (settings.azure_search_endpoint and settings.azure_search_key):
            raise typer.BadParameter("Azure Search endpoint/key not configured")
        embedding = build_default_provider()
        ensure_search_index(
            endpoint=settings.azure_search_endpoint,
            index_name=settings.azure_search_index,
            key=settings.azure_search_key,
            embedding_dim=embedding.dim,
        )
        n = upsert_to_ai_search(
            chunks=chunks,
            embedding=embedding,
            endpoint=settings.azure_search_endpoint,
            index_name=settings.azure_search_index,
            key=settings.azure_search_key,
        )
        typer.echo(f"uploaded {n} chunks to index {settings.azure_search_index}")

    typer.echo(f"wrote {len(chunks)} chunks under {output}")


if __name__ == "__main__":
    app()
