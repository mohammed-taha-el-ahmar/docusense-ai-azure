"""Eval dataset loaders.

Three golden files:
- ``golden_intents.jsonl`` — labelled intent per document.
- ``golden_extractions.jsonl`` — gold field extractions.
- ``judge_prompts.jsonl`` — reasoning-quality prompts for LLM-as-judge.

All three ship with the repo so evals run in CI without external data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from docusense.schemas.classification import IntentLabel


@dataclass
class IntentSample:
    doc_id: str
    intent: IntentLabel


@dataclass
class ExtractionSample:
    doc_id: str
    fields: dict[str, str]


@dataclass
class JudgePrompt:
    doc_id: str
    prompt: str
    reference_intent: IntentLabel


def load_intents(path: Path) -> list[IntentSample]:
    return [
        IntentSample(doc_id=r["doc_id"], intent=IntentLabel(r["intent"])) for r in _iter_jsonl(path)
    ]


def load_extractions(path: Path) -> list[ExtractionSample]:
    return [ExtractionSample(doc_id=r["doc_id"], fields=r["fields"]) for r in _iter_jsonl(path)]


def load_judge_prompts(path: Path) -> list[JudgePrompt]:
    return [
        JudgePrompt(
            doc_id=r["doc_id"],
            prompt=r["prompt"],
            reference_intent=IntentLabel(r["reference_intent"]),
        )
        for r in _iter_jsonl(path)
    ]


def _iter_jsonl(path: Path):
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)
