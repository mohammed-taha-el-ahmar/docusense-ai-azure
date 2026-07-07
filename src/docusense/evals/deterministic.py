"""Deterministic evals — accuracy, F1, latency, cost.

Runnable in CI in seconds because the LLM path is short-circuited by
``ScriptedFakeLLM`` and the retriever is in-memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from docusense.classifier.embeddings import build_default_provider
from docusense.classifier.head import ClassifierHead
from docusense.evals.datasets import ExtractionSample, IntentSample, load_extractions, load_intents
from docusense.schemas.classification import IntentLabel


def _matching_provider(head: ClassifierHead):
    """Build an embedding provider whose dim matches the trained head."""
    default = build_default_provider()
    expected = head._artifact.embedding_dim if head._artifact else default.dim
    if default.dim == expected:
        return default
    from docusense.classifier.embeddings import HashedBagOfWordsEmbedding

    return HashedBagOfWordsEmbedding(dim=expected)


@dataclass
class ClassifierEvalReport:
    accuracy: float
    macro_f1: float
    per_class: dict[str, float] = field(default_factory=dict)
    n_samples: int = 0


def evaluate_classifier(
    corpus_dir: Path,
    labels_path: Path,
    classifier_path: Path,
) -> ClassifierEvalReport:
    """Run the fast classifier over the labelled set and score it."""
    documents = {p.stem: p.read_text() for p in sorted(corpus_dir.glob("*.txt"))}
    samples: list[IntentSample] = load_intents(labels_path)
    head = ClassifierHead.load(classifier_path)
    provider = _matching_provider(head)

    texts = [documents[s.doc_id] for s in samples if s.doc_id in documents]
    truth = [s.intent for s in samples if s.doc_id in documents]
    if not texts:
        return ClassifierEvalReport(accuracy=0.0, macro_f1=0.0)

    embeddings = provider.embed(texts)
    predictions, _ = head.predict(np.asarray(embeddings))

    truth_str = [t.value for t in truth]
    pred_str = [p.value for p in predictions]
    accuracy = float(sum(t == p for t, p in zip(truth_str, pred_str, strict=True)) / len(truth_str))
    macro_f1 = float(f1_score(truth_str, pred_str, average="macro", zero_division=0))
    per_class = {
        c.value: float(
            f1_score(
                [1 if t == c.value else 0 for t in truth_str],
                [1 if p == c.value else 0 for p in pred_str],
                zero_division=0,
            )
        )
        for c in IntentLabel
    }
    return ClassifierEvalReport(
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class=per_class,
        n_samples=len(truth_str),
    )


@dataclass
class ExtractionEvalReport:
    field_recall: float
    n_documents: int
    n_fields_expected: int
    n_fields_found: int


def evaluate_extraction(
    extractions_path: Path,
    predictions: dict[str, dict[str, str]],
) -> ExtractionEvalReport:
    """Score field extractions against gold — recall of expected keys."""
    samples: list[ExtractionSample] = load_extractions(extractions_path)
    n_expected = 0
    n_found = 0
    for sample in samples:
        pred = predictions.get(sample.doc_id, {})
        for key in sample.fields:
            n_expected += 1
            if key in pred:
                n_found += 1
    recall = float(n_found / n_expected) if n_expected else 1.0
    return ExtractionEvalReport(
        field_recall=recall,
        n_documents=len(samples),
        n_fields_expected=n_expected,
        n_fields_found=n_found,
    )
