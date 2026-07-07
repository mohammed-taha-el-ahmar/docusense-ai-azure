"""Structured-output helpers.

Turning a Pydantic model into a strict JSON schema for OpenAI's
``response_format`` is fiddly — the API rejects several things Pydantic
emits by default (unset ``additionalProperties``, ``$ref`` cycles,
``allOf`` wrappers). This module normalises the schema so the call
either succeeds or fails with a clear error, never silently drops fields.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from pydantic import BaseModel, ValidationError


def pydantic_to_openai_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return an OpenAI-compatible JSON schema for ``model``.

    OpenAI's ``strict`` mode requires:
    - every object to declare ``additionalProperties: false``
    - every property to appear in ``required``
    - no top-level ``$defs`` references left unresolved (we inline them)
    """
    raw = model.model_json_schema()
    return _normalise(raw)


def _normalise(schema: dict[str, Any]) -> dict[str, Any]:
    defs = schema.pop("$defs", {})
    schema = copy.deepcopy(schema)
    resolved = _resolve_refs(schema, defs)
    _tighten(resolved)
    return resolved


def _resolve_refs(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node and node["$ref"].startswith("#/$defs/"):
            name = node["$ref"].split("/")[-1]
            resolved = copy.deepcopy(defs[name])
            return _resolve_refs(resolved, defs)
        return {k: _resolve_refs(v, defs) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_refs(v, defs) for v in node]
    return node


def _tighten(schema: dict[str, Any]) -> None:
    """Force ``additionalProperties=false`` and full ``required`` lists in-place."""
    if not isinstance(schema, dict):
        return
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
        properties = schema.get("properties", {})
        schema["required"] = sorted(properties.keys())
        for value in properties.values():
            _tighten(value)
    if schema.get("type") == "array":
        _tighten(schema.get("items", {}))
    for value in schema.values():
        if isinstance(value, list):
            for item in value:
                _tighten(item)


def parse_response(content: str | None, model: type[BaseModel]) -> BaseModel:
    """Parse a structured response, raising ``ValidationError`` on failure.

    Called by ``guardrails.citations`` before it checks that every claim
    has at least one citation — validation is layered on top of parsing.
    """
    if not content:
        raise ValidationError.from_exception_data(model.__name__, [])
    return model.model_validate(json.loads(content))
