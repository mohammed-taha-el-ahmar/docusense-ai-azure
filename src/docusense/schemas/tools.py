"""Argument schemas for the two callable tools.

Pydantic models here are the single source of truth. ``llm.tools``
converts them into OpenAI tool declarations, and the local
implementations validate incoming arguments against them.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LookupClauseLibraryArgs(BaseModel):
    """Arguments for ``lookup_clause_library``.

    Returns similar clauses from the internal clause library, useful when
    the model wants to compare a piece of the current document against
    prior standard language.
    """

    query: str = Field(..., min_length=3, description="Free-text query.")
    top_k: int = Field(default=3, ge=1, le=10)


class CheckCounterpartyArgs(BaseModel):
    """Arguments for ``check_counterparty_history``.

    In production this would call the CRM. In the demo it hits a stub
    JSON file so the tool-calling flow is fully exercised end to end.
    """

    entity_name: str = Field(..., min_length=2)
