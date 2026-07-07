"""Reasoning pipeline tests — the tool-calling loop end-to-end.

Uses ``ScriptedFakeLLM`` for the LLM and the in-memory retriever, so the
whole pipeline runs in milliseconds. Two scenarios:

1. Model answers directly on the first turn, no tools.
2. Model asks for a tool, we execute it, model returns a final answer.
"""

from __future__ import annotations

import json

from docusense.llm.client import ToolCall
from docusense.llm.pipeline import ReasoningPipeline, ReasoningPipelineConfig
from docusense.schemas.classification import IntentLabel
from docusense.schemas.reasoning import ReasoningRequest


def _valid_answer(doc_id: str, chunk_id: str) -> dict:
    return {
        "doc_id": doc_id,
        "intent": "msa",
        "confidence": 0.88,
        "reasoning": "Master services agreement based on services + net-30 payment terms.",
        "extracted_fields": [
            {
                "name": "payment_terms",
                "value": "net 30",
                "citations": [{"chunk_id": chunk_id, "quote": "Payment terms are net thirty"}],
            }
        ],
        "citations": [{"chunk_id": chunk_id, "quote": "Master Services Agreement"}],
        "tools_used": [],
        "model_version": "",
        "tokens_in": 0,
        "tokens_out": 0,
        "latency_ms": 0.0,
    }


def test_pipeline_direct_answer(sample_document, in_memory_retriever, fake_llm_factory) -> None:
    chunk_id = in_memory_retriever.chunks[0].chunk_id
    llm = fake_llm_factory([{"content": _valid_answer(sample_document.doc_id, chunk_id)}])
    pipeline = ReasoningPipeline(llm=llm, retriever=in_memory_retriever)
    outcome = pipeline.run(ReasoningRequest(document=sample_document))
    assert outcome.response.intent == IntentLabel.MSA
    assert outcome.tools_used == []
    assert outcome.response.tokens_in >= 0
    assert outcome.response.latency_ms > 0


def test_pipeline_with_tool_call(sample_document, in_memory_retriever, fake_llm_factory) -> None:
    chunk_id = in_memory_retriever.chunks[0].chunk_id
    tool_call = ToolCall(
        id="call_1",
        name="lookup_clause_library",
        arguments=json.dumps({"query": "net 30 payment", "top_k": 2}),
    )
    llm = fake_llm_factory(
        [
            # first turn: model asks for a tool
            {"content": None, "tool_calls": [tool_call]},
            # second turn: model returns the final structured answer
            {"content": _valid_answer(sample_document.doc_id, chunk_id)},
        ]
    )
    pipeline = ReasoningPipeline(
        llm=llm,
        retriever=in_memory_retriever,
        config=ReasoningPipelineConfig(max_turns=3),
    )
    outcome = pipeline.run(ReasoningRequest(document=sample_document))
    assert outcome.response.intent == IntentLabel.MSA
    assert "lookup_clause_library" in outcome.tools_used
    assert llm.n_calls == 2
