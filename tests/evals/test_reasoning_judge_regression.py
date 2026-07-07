"""LLM-as-judge regression test.

Uses ``ScriptedFakeLLM`` to simulate both the reasoning model and the
judge model. The point of the test is not to score a real LLM (that's
the ``prompt_regression`` workflow) but to verify the judge pipeline
itself works and returns a comparable score.
"""

from __future__ import annotations

import json

import pytest

from docusense.evals.llm_judge import Judge, JudgeScore
from docusense.llm.client import LLMResponse, ScriptedFakeLLM
from docusense.schemas.classification import IntentLabel
from docusense.schemas.reasoning import Citation, ReasoningResponse


@pytest.mark.evals
def test_judge_returns_structured_score() -> None:
    judge_output: dict = {
        "intent_correctness": 5,
        "citation_grounding": 4,
        "field_extraction_quality": 4,
        "conciseness_and_style": 5,
        "notes": "Solid answer; one field could be stronger.",
    }
    llm = ScriptedFakeLLM([LLMResponse(content=json.dumps(judge_output))])
    judge = Judge(llm=llm)

    response = ReasoningResponse(
        doc_id="msa-01",
        intent=IntentLabel.MSA,
        confidence=0.9,
        reasoning="Master Services Agreement identified.",
        citations=[Citation(chunk_id="msa-01#0000", quote="Master Services Agreement")],
    )
    result = judge.score(response)
    assert isinstance(result.score, JudgeScore)
    assert result.score.mean == pytest.approx((5 + 4 + 4 + 5) / 4)
