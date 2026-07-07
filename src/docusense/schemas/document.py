"""Document-level schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DocumentPayload(BaseModel):
    """A single document coming in on the API — text plus optional metadata."""

    doc_id: str = Field(..., description="Stable identifier assigned by the caller.")
    text: str = Field(..., min_length=1, description="Raw document text.")
    source: str | None = Field(default=None, description="e.g. 'email', 'sharepoint'.")
    received_at: datetime | None = None


class DocumentChunk(BaseModel):
    """A chunk of a document — the unit of retrieval."""

    chunk_id: str = Field(..., description="Stable id: doc_id + '#' + chunk_index.")
    doc_id: str
    chunk_index: int = Field(..., ge=0)
    text: str = Field(..., min_length=1)
    n_tokens: int = Field(..., gt=0)
