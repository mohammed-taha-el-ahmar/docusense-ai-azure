"""Retriever tests — hybrid blend, empty index."""

from __future__ import annotations

from docusense.classifier.embeddings import HashedBagOfWordsEmbedding
from docusense.retrieval.search import InMemoryHybridRetriever


def test_returns_top_k_ordered(in_memory_retriever: InMemoryHybridRetriever) -> None:
    results = in_memory_retriever.search("net thirty days invoice", top_k=3)
    assert 1 <= len(results) <= 3
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_empty_index_returns_nothing() -> None:
    retriever = InMemoryHybridRetriever(embedding=HashedBagOfWordsEmbedding(dim=32), chunks=[])
    assert retriever.search("anything", top_k=5) == []
