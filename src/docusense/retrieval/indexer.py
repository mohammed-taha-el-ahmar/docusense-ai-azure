"""Index-building utilities — used both locally and from the AML pipeline.

The local path builds an in-memory retriever and pickles it, letting
CI and the local endpoint work without Azure. The Azure path upserts
chunks + embeddings into Azure AI Search.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from docusense.classifier.embeddings import EmbeddingProvider
from docusense.retrieval.chunker import chunk_documents
from docusense.retrieval.search import InMemoryHybridRetriever
from docusense.schemas.document import DocumentChunk, DocumentPayload


def build_in_memory_retriever(
    docs: Iterable[DocumentPayload],
    embedding: EmbeddingProvider,
) -> InMemoryHybridRetriever:
    chunks = chunk_documents(docs)
    return InMemoryHybridRetriever(embedding=embedding, chunks=chunks)


def dump_chunks(chunks: Iterable[DocumentChunk], path: Path) -> None:
    """Persist chunks as JSONL — used by both index paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")


def load_chunks(path: Path) -> list[DocumentChunk]:
    return [
        DocumentChunk.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def ensure_search_index(
    endpoint: str,
    index_name: str,
    key: str,
    embedding_dim: int = 3072,
) -> None:  # pragma: no cover — network call
    """Create the Azure AI Search index if it does not already exist."""
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import ResourceNotFoundError
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration,
        SearchableField,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SimpleField,
        VectorSearch,
        VectorSearchProfile,
    )

    client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    try:
        client.get_index(index_name)
        return  # already exists
    except ResourceNotFoundError:
        pass

    fields = [
        SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True, filterable=True),
        SimpleField(name="doc_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32),
        SearchableField(name="text", type=SearchFieldDataType.String),
        SimpleField(name="n_tokens", type=SearchFieldDataType.Int32),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=embedding_dim,
            vector_search_profile_name="default-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="default-hnsw")],
        profiles=[VectorSearchProfile(name="default-profile", algorithm_configuration_name="default-hnsw")],
    )

    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
    client.create_index(index)


def upsert_to_ai_search(
    chunks: list[DocumentChunk],
    embedding: EmbeddingProvider,
    endpoint: str,
    index_name: str,
    key: str,
    batch_size: int = 100,
) -> int:  # pragma: no cover — network call
    """Push chunks into Azure AI Search. Returns count uploaded."""
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient

    client = SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(key),
    )

    n = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embedding.embed([c.text for c in batch])
        records = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "n_tokens": c.n_tokens,
                "embedding": vectors[i].tolist(),
            }
            for i, c in enumerate(batch)
        ]
        client.upload_documents(records)
        n += len(records)
    return n


def read_corpus(directory: Path) -> list[DocumentPayload]:
    return [
        DocumentPayload(doc_id=p.stem, text=p.read_text(), source="local")
        for p in sorted(directory.glob("*.txt"))
    ]


def emit_manifest(directory: Path, chunks: list[DocumentChunk]) -> Path:
    """Write a small manifest that CI checks against for sanity."""
    manifest = {
        "chunk_count": len(chunks),
        "documents": sorted({c.doc_id for c in chunks}),
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path
