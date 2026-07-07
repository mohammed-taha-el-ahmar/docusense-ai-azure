"""Cost estimator tests."""

from __future__ import annotations

from docusense.telemetry.cost import DEFAULT_MODEL, estimate_cost_usd


def test_zero_tokens_zero_cost() -> None:
    assert estimate_cost_usd("gpt-4o", 0, 0) == 0.0


def test_output_more_expensive_than_input() -> None:
    in_only = estimate_cost_usd("gpt-4o", 1000, 0)
    out_only = estimate_cost_usd("gpt-4o", 0, 1000)
    assert out_only > in_only


def test_unknown_model_falls_back_to_default() -> None:
    known = estimate_cost_usd(DEFAULT_MODEL, 500, 500)
    unknown = estimate_cost_usd("mystery-model-x", 500, 500)
    assert known == unknown
