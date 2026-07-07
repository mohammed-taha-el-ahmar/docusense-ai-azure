"""Retrieval protocol + a fully in-memory implementation used by tests.

The protocol is what lets the serving code stay ignorant of whether it's
talking to Azure AI Search or the in-memory implementation — same
interface, different backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from docusense.classifier.embeddings import EmbeddingProvider
from docusense.schemas.document import DocumentChunk


@dataclass
class RetrievedPassage:
    chunk: DocumentChunk
    score: float


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[RetrievedPassage]: ...


class InMemoryHybridRetriever:
    """Vector cosine similarity + simple keyword overlap, then linear blend.

    Not a substitute for Azure AI Search's semantic ranker in production,
    but adequate for unit tests and local dev — and, importantly, a fair
    baseline against which the AI-Search-backed retriever can be compared.
    """

    def __init__(
        self,
        embedding: EmbeddingProvider,
        chunks: list[DocumentChunk],
        vector_weight: float = 0.7,
    ) -> None:
        if not 0.0 <= vector_weight <= 1.0:
            raise ValueError("vector_weight must be in [0, 1]")
        self.embedding = embedding
        self.chunks = chunks
        self.vector_weight = vector_weight
        self._matrix = (
            embedding.embed([c.text for c in chunks])
            if chunks
            else np.zeros((0, embedding.dim), dtype=np.float32)
        )

    def search(self, query: str, top_k: int = 5) -> list[RetrievedPassage]:
        if not self.chunks:
            return []
        q_vec = self.embedding.embed([query])[0]
        # Cosine — inputs already normalised in the hashed provider.
        vector_scores = self._matrix @ q_vec
        keyword_scores = self._keyword_overlap(query)
        blended = self.vector_weight * vector_scores + (1 - self.vector_weight) * keyword_scores
        top_idx = np.argsort(-blended)[:top_k]
        return [RetrievedPassage(chunk=self.chunks[i], score=float(blended[i])) for i in top_idx]

    def _keyword_overlap(self, query: str) -> np.ndarray:
        query_tokens = {t.lower() for t in query.split() if len(t) > 2}
        if not query_tokens:
            return np.zeros(len(self.chunks), dtype=np.float32)
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for i, chunk in enumerate(self.chunks):
            chunk_tokens = {t.lower() for t in chunk.text.split() if len(t) > 2}
            if not chunk_tokens:
                continue
            scores[i] = len(query_tokens & chunk_tokens) / len(query_tokens)
        return scores


class AzureAISearchRetriever:
    """Hybrid retriever backed by Azure AI Search's vector + semantic ranker.

    Lazy client construction so the module can be imported without Azure
    credentials — useful for CI and for tests that swap in the in-memory
    variant instead.
    """

    def __init__(
        self,
        endpoint: str,
        index_name: str,
        key: str,
        embedding: EmbeddingProvider,
        use_semantic_ranker: bool = False,
    ) -> None:
        self._endpoint = endpoint
        self._index_name = index_name
        self._key = key
        self.embedding = embedding
        self._use_semantic_ranker = use_semantic_ranker
        self._client = None

    def _lazy_client(self):  # pragma: no cover — needs Azure
        if self._client is None:
            from azure.core.credentials import AzureKeyCredential
            from azure.search.documents import SearchClient

            self._client = SearchClient(
                endpoint=self._endpoint,
                index_name=self._index_name,
                credential=AzureKeyCredential(self._key),
            )
        return self._client

    def search(self, query: str, top_k: int = 5) -> list[RetrievedPassage]:  # pragma: no cover
        from azure.search.documents.models import VectorizedQuery

        q_vec = self.embedding.embed([query])[0].tolist()
        search_kwargs: dict = {
            "search_text": query,
            "vector_queries": [
                VectorizedQuery(vector=q_vec, k_nearest_neighbors=top_k * 2, fields="embedding")
            ],
            "top": top_k,
        }
        if self._use_semantic_ranker:
            search_kwargs["query_type"] = "semantic"
            search_kwargs["semantic_configuration_name"] = "default"

        results = self._lazy_client().search(**search_kwargs)
        out: list[RetrievedPassage] = []
        for r in results:
            out.append(
                RetrievedPassage(
                    chunk=DocumentChunk(
                        chunk_id=r["chunk_id"],
                        doc_id=r["doc_id"],
                        chunk_index=int(r["chunk_index"]),
                        text=r["text"],
                        n_tokens=int(r.get("n_tokens", 0) or 1),
                    ),
                    score=float(
                        r.get("@search.reranker_score", r.get("@search.score", 0.0)) or 0.0
                    ),
                )
            )
        return out
