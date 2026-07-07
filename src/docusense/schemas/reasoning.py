"""Reasoning-path schemas.

The ``ReasoningResponse`` is doubly important: it's both the shape of the
API response *and* the source of the JSON schema handed to Azure OpenAI's
``response_format`` parameter. That contract is what guarantees the LLM
output is parseable — no regex, no string-scraping.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from docusense.schemas.classification import IntentLabel
from docusense.schemas.document import DocumentPayload


class ReasoningRequest(BaseModel):
    document: DocumentPayload
    # Optional caller hint — the fast classifier's guess. The LLM is free
    # to override it, but the hint tightens retrieval and prompt priming.
    hinted_intent: IntentLabel | None = None


class Citation(BaseModel):
    """Pointer back to a retrieved chunk that supports a claim."""

    chunk_id: str = Field(..., description="Identifier of the retrieved chunk.")
    quote: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description="Verbatim snippet from the chunk supporting the claim.",
    )


class ExtractedField(BaseModel):
    """One structured field the LLM extracted from the document."""

    name: str = Field(..., min_length=1, description="e.g. 'effective_date'")
    value: str = Field(..., description="String value; caller parses further.")
    citations: list[Citation] = Field(..., min_length=1)


class ReasoningResponse(BaseModel):
    """The exact shape returned to the caller, and the JSON schema the LLM fills."""

    # We turn off arbitrary types etc. — the schema needs to be pure JSON.
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    intent: IntentLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(
        ...,
        min_length=10,
        description="Short natural-language justification, grounded in citations.",
    )
    extracted_fields: list[ExtractedField] = Field(default_factory=list)
    citations: list[Citation] = Field(
        default_factory=list,
        description="Citations for the reasoning statement itself.",
    )
    tools_used: list[str] = Field(default_factory=list)
    model_version: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
