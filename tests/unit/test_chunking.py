"""Chunker tests."""

from __future__ import annotations

import pytest

from docusense.retrieval.chunker import chunk_document, count_tokens
from docusense.schemas.document import DocumentPayload


def test_chunk_ids_are_stable(sample_document: DocumentPayload) -> None:
    chunks = chunk_document(sample_document, max_tokens=60, overlap_tokens=10)
    assert all(c.chunk_id.startswith(sample_document.doc_id + "#") for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunks_have_overlap(sample_document: DocumentPayload) -> None:
    chunks = chunk_document(sample_document, max_tokens=20, overlap_tokens=5)
    assert len(chunks) >= 2
    # neighbouring chunks should share text substrings around the overlap
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert any(word in b.text for word in a.text.split()[-3:])


def test_reject_overlap_greater_than_max(sample_document: DocumentPayload) -> None:
    with pytest.raises(ValueError):
        chunk_document(sample_document, max_tokens=20, overlap_tokens=20)


def test_short_document_gives_single_chunk() -> None:
    doc = DocumentPayload(doc_id="d1", text="Just a short document.")
    chunks = chunk_document(doc, max_tokens=500)
    assert len(chunks) == 1
    assert chunks[0].n_tokens == count_tokens(doc.text)
