"""Tool declarations + local implementations.

The tool schema comes straight from the Pydantic argument model — no
hand-written JSON schema drift. Every tool has a validated Python
implementation the runtime calls when the LLM emits a ``tool_calls``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from docusense.llm.structured import pydantic_to_openai_schema
from docusense.schemas.tools import CheckCounterpartyArgs, LookupClauseLibraryArgs

ToolFn = Callable[[BaseModel], dict[str, Any]]


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: ToolFn

    def as_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": pydantic_to_openai_schema(self.args_model),
            },
        }

    def invoke(self, raw_arguments: str) -> dict[str, Any]:
        try:
            args = self.args_model.model_validate_json(raw_arguments)
        except ValidationError as e:
            return {"error": f"invalid arguments: {e}"}
        return self.fn(args)


# ---- Tool implementations ---------------------------------------------------


def _lookup_clause_library(args: LookupClauseLibraryArgs) -> dict[str, Any]:
    """Stubbed clause-library lookup.

    In production, this would call an actual clause repository backed by
    AI Search over a curated corpus. For the demo, we serve a small
    hand-authored JSON so the tool-calling flow is fully exercised.
    """
    stub_path = Path(__file__).parent / "stubs" / "clause_library.json"
    library = json.loads(stub_path.read_text()) if stub_path.exists() else []
    scored = []
    q = args.query.lower()
    for entry in library:
        score = sum(1 for tok in q.split() if tok in entry["text"].lower())
        scored.append((score, entry))
    scored.sort(key=lambda x: -x[0])
    return {"results": [e for _, e in scored[: args.top_k]]}


def _check_counterparty(args: CheckCounterpartyArgs) -> dict[str, Any]:
    """Stubbed counterparty history lookup."""
    stub_path = Path(__file__).parent / "stubs" / "counterparties.json"
    directory = json.loads(stub_path.read_text()) if stub_path.exists() else {}
    match = directory.get(args.entity_name.lower())
    if match is None:
        return {"status": "unknown", "entity_name": args.entity_name}
    return {"status": "known", **match}


# ---- Registry ---------------------------------------------------------------


TOOLS: list[Tool] = [
    Tool(
        name="lookup_clause_library",
        description=(
            "Look up similar clauses from the internal clause library. Use when a "
            "clause in the current document should be compared to standard language."
        ),
        args_model=LookupClauseLibraryArgs,
        fn=_lookup_clause_library,
    ),
    Tool(
        name="check_counterparty_history",
        description=(
            "Return prior deal history for a counterparty. Use when the reasoning "
            "would benefit from knowing whether the counterparty is a new relationship."
        ),
        args_model=CheckCounterpartyArgs,
        fn=_check_counterparty,
    ),
]


TOOLS_BY_NAME: dict[str, Tool] = {t.name: t for t in TOOLS}


def openai_tool_declarations() -> list[dict[str, Any]]:
    return [t.as_openai() for t in TOOLS]
