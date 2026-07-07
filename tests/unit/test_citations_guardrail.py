"""Citation guardrail tests."""

from __future__ import annotations

import pytest

from docusense.guardrails.citations import CitationViolation, enforce_citations
from docusense.retrieval.chunker import chunk_document
from docusense.retrieval.search import RetrievedPassage
from docusense.schemas.classification import IntentLabel
from docusense.schemas.reasoning import Citation, ReasoningResponse


def _passages(sample_document):
    chunks = chunk_document(sample_document, max_tokens=80)
    return [RetrievedPassage(chunk=c, score=1.0) for c in chunks]


def test_valid_citation_passes(sample_document) -> None:
    passages = _passages(sample_document)
    quote = "Payment terms are net thirty"
    response = ReasoningResponse(
        doc_id=sample_document.doc_id,
        intent=IntentLabel.MSA,
        confidence=0.9,
        reasoning="MSA identified by services + payment terms language.",
        citations=[Citation(chunk_id=passages[0].chunk.chunk_id, quote=quote)],
    )
    enforce_citations(response, passages)  # no exception


def test_unknown_chunk_id_fails(sample_document) -> None:
    passages = _passages(sample_document)
    response = ReasoningResponse(
        doc_id=sample_document.doc_id,
        intent=IntentLabel.MSA,
        confidence=0.9,
        reasoning="MSA identified but citation is unsupported.",
        citations=[Citation(chunk_id="ghost#0000", quote="Payment terms are net thirty")],
    )
    with pytest.raises(CitationViolation):
        enforce_citations(response, passages)


def test_hallucinated_quote_fails(sample_document) -> None:
    passages = _passages(sample_document)
    response = ReasoningResponse(
        doc_id=sample_document.doc_id,
        intent=IntentLabel.MSA,
        confidence=0.9,
        reasoning="MSA identified but citation is unsupported.",
        citations=[
            Citation(chunk_id=passages[0].chunk.chunk_id, quote="This quote never appears anywhere")
        ],
    )
    with pytest.raises(CitationViolation):
        enforce_citations(response, passages)


def test_no_citations_ok_for_other(sample_document) -> None:
    passages = _passages(sample_document)
    response = ReasoningResponse(
        doc_id=sample_document.doc_id,
        intent=IntentLabel.OTHER,
        confidence=0.3,
        reasoning="insufficient signal",
        citations=[],
    )
    enforce_citations(response, passages)


def test_no_citations_fails_for_classified(sample_document) -> None:
    passages = _passages(sample_document)
    response = ReasoningResponse(
        doc_id=sample_document.doc_id,
        intent=IntentLabel.MSA,
        confidence=0.9,
        reasoning="this is a services agreement",
        citations=[],
    )
    with pytest.raises(CitationViolation):
        enforce_citations(response, passages)
