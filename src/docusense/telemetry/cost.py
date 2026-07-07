"""Per-request cost accounting.

Numbers here are illustrative — production would source them from the
Azure pricing API or a manually maintained YAML in ``infra/``. The point
is that every ``ReasoningResponse`` carries a cost estimate the endpoint
can log, alert on, and eventually enforce budgets against.
"""

from __future__ import annotations

# USD per 1K tokens, illustrative — check current Azure pricing.
_PRICING: dict[str, tuple[float, float]] = {
    # (input, output)
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4-turbo": (0.01, 0.03),
    "text-embedding-3-large": (0.00013, 0.0),
    "text-embedding-3-small": (0.00002, 0.0),
}

DEFAULT_MODEL = "gpt-4o"


def estimate_cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    """Return a USD cost estimate for a call.

    Falls back to the default model if ``model`` is unknown so unknown
    deployment names produce a comparable number rather than 0.
    """
    price_in, price_out = _PRICING.get(model, _PRICING[DEFAULT_MODEL])
    return (tokens_in / 1000.0) * price_in + (tokens_out / 1000.0) * price_out
