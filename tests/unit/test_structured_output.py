"""JSON-schema generation + response parsing tests."""

from __future__ import annotations

from docusense.llm.structured import parse_response, pydantic_to_openai_schema
from docusense.schemas.reasoning import ReasoningResponse


def test_schema_is_strict() -> None:
    schema = pydantic_to_openai_schema(ReasoningResponse)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"].keys())


def test_schema_resolves_nested_refs() -> None:
    schema = pydantic_to_openai_schema(ReasoningResponse)
    # citations should have expanded — no dangling $ref
    citations = schema["properties"]["citations"]
    assert citations["type"] == "array"
    assert citations["items"]["type"] == "object"
    assert "chunk_id" in citations["items"]["properties"]
    assert "$ref" not in str(schema)


def test_parse_valid_response() -> None:
    payload = {
        "doc_id": "d1",
        "intent": "other",
        "confidence": 0.5,
        "reasoning": "insufficient signal to classify",
        "extracted_fields": [],
        "citations": [],
        "tools_used": [],
        "model_version": "",
        "tokens_in": 0,
        "tokens_out": 0,
        "latency_ms": 0.0,
    }
    import json

    parsed = parse_response(json.dumps(payload), ReasoningResponse)
    assert isinstance(parsed, ReasoningResponse)
    assert parsed.intent.value == "other"
