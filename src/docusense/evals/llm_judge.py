"""LLM-as-judge evaluator.

Scores a ``ReasoningResponse`` against a rubric, using an ``LLMClient`` —
which lets tests pass a ``ScriptedFakeLLM`` and lets production wire a
strong judge model in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from docusense.llm.client import LLMClient
from docusense.llm.prompting import load_prompt
from docusense.llm.structured import parse_response, pydantic_to_openai_schema
from docusense.schemas.reasoning import ReasoningResponse


class JudgeScore(BaseModel):
    """Structured output the judge model returns."""

    intent_correctness: int = Field(..., ge=1, le=5)
    citation_grounding: int = Field(..., ge=1, le=5)
    field_extraction_quality: int = Field(..., ge=1, le=5)
    conciseness_and_style: int = Field(..., ge=1, le=5)
    notes: str = Field(..., max_length=400)

    @property
    def mean(self) -> float:
        return (
            self.intent_correctness
            + self.citation_grounding
            + self.field_extraction_quality
            + self.conciseness_and_style
        ) / 4.0


@dataclass
class JudgeResult:
    doc_id: str
    score: JudgeScore


class Judge:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._rubric = load_prompt("judge_rubric")
        self._schema = pydantic_to_openai_schema(JudgeScore)

    def score(self, response: ReasoningResponse) -> JudgeResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._rubric},
            {
                "role": "user",
                "content": (
                    "Response to score:\n" + json.dumps(response.model_dump(mode="json"), indent=2)
                ),
            },
        ]
        llm_response = self._llm.complete(
            messages=messages,
            response_schema=self._schema,
            temperature=0.0,
        )
        parsed = parse_response(llm_response.content, JudgeScore)
        assert isinstance(parsed, JudgeScore)
        return JudgeResult(doc_id=response.doc_id, score=parsed)
