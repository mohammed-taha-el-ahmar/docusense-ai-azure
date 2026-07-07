"""OpenTelemetry span helpers.

Every LLM call and every retrieval call gets a span with the attributes
production dashboards need to slice on: model, tokens, cost, prompt_hash.

The default tracer provider is the no-op one; production wires up
Azure Monitor OpenTelemetry via ``configure_azure_monitor()`` in the
serving init path.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace

_tracer = trace.get_tracer("docusense")


def prompt_hash(text: str) -> str:
    """Short, stable hash of a rendered prompt — useful as a span attribute."""
    return hashlib.blake2b(text.encode(), digest_size=8).hexdigest()


@contextmanager
def llm_span(*, model: str, prompt: str, extra: dict[str, Any] | None = None):
    """Context manager wrapping an LLM call in an OTel span."""
    with _tracer.start_as_current_span("docusense.llm.complete") as span:
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.prompt_hash", prompt_hash(prompt))
        for k, v in (extra or {}).items():
            span.set_attribute(k, v)
        yield span


@contextmanager
def retrieval_span(*, query: str, top_k: int):
    with _tracer.start_as_current_span("docusense.retrieval.search") as span:
        span.set_attribute("retrieval.query_hash", prompt_hash(query))
        span.set_attribute("retrieval.top_k", top_k)
        yield span
