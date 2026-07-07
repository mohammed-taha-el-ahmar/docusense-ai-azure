"""Shared fixtures.

The fixtures here are what make the whole test suite run in seconds
without network access:

- ``sample_document``: a small in-memory contract-like document.
- ``in_memory_retriever``: retriever built from a couple of chunks.
- ``fake_llm_factory``: builds a ``ScriptedFakeLLM`` from a list of dicts.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from docusense.classifier.embeddings import HashedBagOfWordsEmbedding
from docusense.llm.client import LLMResponse, ScriptedFakeLLM, ToolCall
from docusense.retrieval.chunker import chunk_document
from docusense.retrieval.search import InMemoryHybridRetriever
from docusense.schemas.document import DocumentPayload

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_document() -> DocumentPayload:
    return DocumentPayload(
        doc_id="msa-01",
        text=(
            "MASTER SERVICES AGREEMENT\n\n"
            "This Master Services Agreement is made effective as of July 1, 2024, "
            'between ACME Corp ("Customer") and Vandelay Consulting ("Provider"). '
            "Provider shall provide professional services under one or more Statements "
            "of Work. Payment terms are net thirty (30) days from a correct invoice. "
            "Either party may terminate for convenience upon thirty (30) days' written "
            "notice."
        ),
        source="unit-test",
        received_at=datetime(2024, 7, 1, tzinfo=UTC),
    )


@pytest.fixture
def in_memory_retriever(sample_document: DocumentPayload) -> InMemoryHybridRetriever:
    embedding = HashedBagOfWordsEmbedding(dim=128)
    chunks = chunk_document(sample_document, max_tokens=80, overlap_tokens=10)
    return InMemoryHybridRetriever(embedding=embedding, chunks=chunks)


@pytest.fixture
def build_llm_response():
    """Factory building an ``LLMResponse`` from kwargs — used by many tests."""

    def _build(
        content: str | dict | None = None,
        tool_calls: list[ToolCall] | None = None,
        tokens_in: int = 100,
        tokens_out: int = 50,
        model: str = "gpt-4o",
        finish_reason: str = "stop",
    ) -> LLMResponse:
        if isinstance(content, dict):
            content = json.dumps(content)
        return LLMResponse(
            content=content,
            tool_calls=tool_calls or [],
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=model,
            finish_reason=finish_reason,
        )

    return _build


@pytest.fixture
def fake_llm_factory(build_llm_response):
    def _build(script: list[dict]) -> ScriptedFakeLLM:
        responses = [build_llm_response(**item) for item in script]
        return ScriptedFakeLLM(script=responses)

    return _build
