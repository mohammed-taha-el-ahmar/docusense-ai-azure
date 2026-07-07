"""Schema tests — the contracts must reject invalid inputs."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from docusense.schemas.classification import ClassifyResponse, IntentLabel
from docusense.schemas.document import DocumentPayload
from docusense.schemas.reasoning import Citation, ExtractedField, ReasoningResponse


def test_document_payload_requires_non_empty_text() -> None:
    with pytest.raises(ValidationError):
        DocumentPayload(doc_id="d1", text="")


def test_intent_label_enumerates_known_intents() -> None:
    values = {i.value for i in IntentLabel}
    assert {"nda", "msa", "purchase_order", "rfp", "termination", "price_change", "other"} <= values


def test_classify_response_bounds_confidence() -> None:
    with pytest.raises(ValidationError):
        ClassifyResponse(
            doc_id="d1",
            intent=IntentLabel.NDA,
            confidence=1.5,
            route="fast",
            latency_ms=10.0,
        )


def test_reasoning_response_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ReasoningResponse(
            doc_id="d1",
            intent=IntentLabel.OTHER,
            confidence=0.5,
            reasoning="short reason here",
            unexpected="not allowed",  # type: ignore[call-arg]
        )


def test_extracted_field_requires_at_least_one_citation() -> None:
    with pytest.raises(ValidationError):
        ExtractedField(name="effective_date", value="2024-07-01", citations=[])


def test_citation_short_quote_ok() -> None:
    Citation(chunk_id="c#0000", quote="short quote")


def test_scoring_response_serialises_cleanly() -> None:
    response = ReasoningResponse(
        doc_id="d1",
        intent=IntentLabel.OTHER,
        confidence=0.5,
        reasoning="short reason here",
    )
    assert "unexpected" not in response.model_dump()
    _ = datetime.now(tz=UTC)  # keep import
