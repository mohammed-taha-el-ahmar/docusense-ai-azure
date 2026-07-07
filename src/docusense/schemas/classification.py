"""Fast-path classification schemas."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from docusense.schemas.document import DocumentPayload


class IntentLabel(str, Enum):
    """The enumeration of intents the fast classifier and the LLM both use.

    Keeping one enum in one file is what stops the two heads from silently
    disagreeing on the label space over time.
    """

    NDA = "nda"
    MSA = "msa"
    PURCHASE_ORDER = "purchase_order"
    RFP = "rfp"
    TERMINATION = "termination"
    PRICE_CHANGE = "price_change"
    OTHER = "other"


class ClassifyRequest(BaseModel):
    document: DocumentPayload


class ClassifyResponse(BaseModel):
    """Fast-path result — millisecond-scale, no LLM involvement."""

    doc_id: str
    intent: IntentLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    route: str = Field(..., description="'fast' or 'escalate_to_reason'")
    latency_ms: float = Field(..., ge=0.0)
    model_version: str = "local"
