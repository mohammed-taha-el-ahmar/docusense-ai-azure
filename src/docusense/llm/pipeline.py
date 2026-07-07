"""End-to-end reasoning pipeline.

Runs on every ``/reason`` request:

1. PII scrub on the incoming document.
2. Retrieve top-K passages from the retriever.
3. Build the prompt (system + user with citations).
4. Call the LLM in a bounded tool-calling loop (max ``max_turns``).
5. Parse the structured response.
6. Enforce guardrails: every claim must have a valid citation; output
   passes Content Safety (or the local no-op fallback).
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from docusense.guardrails.citations import enforce_citations
from docusense.guardrails.pii import PIIRedactor
from docusense.guardrails.safety import OutputSafetyChecker
from docusense.llm.client import LLMClient, LLMResponse
from docusense.llm.prompting import load_prompt, render_reason_user
from docusense.llm.structured import parse_response, pydantic_to_openai_schema
from docusense.llm.tools import TOOLS_BY_NAME, openai_tool_declarations
from docusense.retrieval.search import Retriever
from docusense.schemas.reasoning import ReasoningRequest, ReasoningResponse
from docusense.telemetry.cost import estimate_cost_usd


@dataclass
class ReasoningPipelineConfig:
    top_k: int = 5
    max_turns: int = 4  # 1 initial + up to 3 tool round-trips
    temperature: float = 0.0


@dataclass
class ReasoningOutcome:
    response: ReasoningResponse
    tools_used: list[str]
    turns: int
    cost_usd: float


class ReasoningPipeline:
    """Composes retriever + LLM + tools + guardrails."""

    def __init__(
        self,
        llm: LLMClient,
        retriever: Retriever,
        pii: PIIRedactor | None = None,
        safety: OutputSafetyChecker | None = None,
        config: ReasoningPipelineConfig | None = None,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.pii = pii or PIIRedactor()
        self.safety = safety or OutputSafetyChecker()
        self.config = config or ReasoningPipelineConfig()
        self._schema = pydantic_to_openai_schema(ReasoningResponse)
        self._tool_declarations = openai_tool_declarations()
        self._system_prompt = load_prompt("reason_system")

    def run(self, request: ReasoningRequest) -> ReasoningOutcome:
        start = time.perf_counter()

        # 1. PII scrub
        scrubbed_text = self.pii.redact(request.document.text)
        scrubbed = request.model_copy(deep=True)
        scrubbed.document = scrubbed.document.model_copy(update={"text": scrubbed_text})

        # 2. Retrieve
        passages = self.retriever.search(
            query=scrubbed.document.text[:2000],
            top_k=self.config.top_k,
        )

        # 3+4. Prompt + tool-calling loop
        user_prompt = render_reason_user(scrubbed, passages)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        tools_used: list[str] = []
        tokens_in = 0
        tokens_out = 0
        model_name = ""

        response: LLMResponse | None = None
        for turn in range(self.config.max_turns):
            response = self.llm.complete(
                messages=messages,
                tools=self._tool_declarations,
                response_schema=self._schema,
                temperature=self.config.temperature,
            )
            tokens_in += response.tokens_in
            tokens_out += response.tokens_out
            model_name = model_name or response.model

            if not response.tool_calls:
                break

            # Model asked for tools — execute and feed results back.
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        }
                        for tc in response.tool_calls
                    ],
                }
            )
            for tc in response.tool_calls:
                tool = TOOLS_BY_NAME.get(tc.name)
                if tool is None:
                    result = {"error": f"unknown tool: {tc.name}"}
                else:
                    result = tool.invoke(tc.arguments)
                    tools_used.append(tc.name)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    }
                )
            # If we've hit the last turn without a stop, force a final answer.
            if turn == self.config.max_turns - 1:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "You have used the maximum number of tool calls. "
                            "Return the final JSON object per the schema now."
                        ),
                    }
                )

        assert response is not None  # loop always assigns
        parsed = parse_response(response.content, ReasoningResponse)
        assert isinstance(parsed, ReasoningResponse)

        # 5. Guardrails
        enforce_citations(parsed, passages)
        self.safety.check(parsed.reasoning)

        latency_ms = (time.perf_counter() - start) * 1000.0
        parsed = parsed.model_copy(
            update={
                "tools_used": tools_used,
                "model_version": model_name or "unknown",
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "latency_ms": latency_ms,
                "doc_id": request.document.doc_id,
            }
        )
        cost = estimate_cost_usd(model_name or "gpt-4o", tokens_in, tokens_out)
        return ReasoningOutcome(
            response=parsed,
            tools_used=tools_used,
            turns=self.llm_call_count(response, tools_used),
            cost_usd=cost,
        )

    def llm_call_count(self, _final: LLMResponse, tools_used: Sequence[str]) -> int:
        """Number of LLM calls used to produce the answer (1 + tool round trips)."""
        # Every tool-call turn adds one LLM call after tool results are fed back.
        # An answer with no tool calls used exactly 1 call.
        return 1 + len(tools_used)
