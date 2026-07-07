"""Tool-declaration and tool-execution tests."""

from __future__ import annotations

import json

from docusense.llm.tools import TOOLS_BY_NAME, openai_tool_declarations


def test_tools_are_declared_with_strict_schemas() -> None:
    decls = openai_tool_declarations()
    names = {d["function"]["name"] for d in decls}
    assert {"lookup_clause_library", "check_counterparty_history"} <= names
    for decl in decls:
        params = decl["function"]["parameters"]
        assert params["additionalProperties"] is False
        assert set(params["required"]) == set(params["properties"].keys())


def test_lookup_clause_library_returns_results() -> None:
    tool = TOOLS_BY_NAME["lookup_clause_library"]
    result = tool.invoke(json.dumps({"query": "termination for convenience", "top_k": 2}))
    assert "results" in result
    assert len(result["results"]) <= 2


def test_lookup_rejects_invalid_arguments() -> None:
    tool = TOOLS_BY_NAME["lookup_clause_library"]
    result = tool.invoke(json.dumps({"query": "a"}))  # too short
    assert "error" in result


def test_check_counterparty_known_and_unknown() -> None:
    tool = TOOLS_BY_NAME["check_counterparty_history"]
    known = tool.invoke(json.dumps({"entity_name": "ACME Corp"}))
    assert known["status"] == "known"
    unknown = tool.invoke(json.dumps({"entity_name": "Nonexistent Ltd"}))
    assert unknown["status"] == "unknown"
