"""Citation guardrail.

Enforces two things about a ``ReasoningResponse``:

1. Every referenced ``chunk_id`` must exist in the retrieved passages.
2. Every quoted snippet must appear verbatim in the chunk it cites.

The second check is what stops the model from citing a real chunk id
while hallucinating a quote — a classic RAG failure mode.
"""

from __future__ import annotations

from docusense.retrieval.search import RetrievedPassage
from docusense.schemas.reasoning import Citation, ReasoningResponse


class CitationViolation(ValueError):
    """Raised when the response contains a citation not backed by retrieval."""


def enforce_citations(
    response: ReasoningResponse,
    passages: list[RetrievedPassage],
    max_snippet_slack: int = 20,
) -> None:
    """Validate response citations against the retrieved passages."""
    chunk_by_id = {p.chunk.chunk_id: p.chunk for p in passages}

    all_citations: list[Citation] = list(response.citations)
    for field in response.extracted_fields:
        all_citations.extend(field.citations)

    if response.intent.value != "other" and not all_citations:
        raise CitationViolation(
            "no citations provided for a non-'other' intent — every claim must be grounded"
        )

    for citation in all_citations:
        chunk = chunk_by_id.get(citation.chunk_id)
        if chunk is None:
            raise CitationViolation(
                f"citation references chunk_id {citation.chunk_id!r} not in retrieved passages"
            )
        if not _fuzzy_contains(chunk.text, citation.quote, max_snippet_slack):
            raise CitationViolation(
                f"quoted snippet not found in chunk {citation.chunk_id!r}: {citation.quote[:80]!r}"
            )


def _fuzzy_contains(haystack: str, needle: str, slack: int) -> bool:
    """Substring match, tolerant of collapsed whitespace.

    ``slack`` is not currently used for character-level fuzz — we're strict
    that the quote appears verbatim after normalising whitespace. The
    parameter is retained for a future edit-distance relaxation.
    """
    _ = slack
    normalise = lambda s: " ".join(s.split()).lower()  # noqa: E731
    return normalise(needle) in normalise(haystack)
