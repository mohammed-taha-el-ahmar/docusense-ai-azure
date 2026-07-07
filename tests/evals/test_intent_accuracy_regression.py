"""Deterministic intent-accuracy regression test.

Trains the fast head on the golden set and asserts a floor on macro F1.
Uses the shipped sample corpus + hashed embeddings, so it runs in CI
without touching Azure OpenAI.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from docusense.classifier.embeddings import HashedBagOfWordsEmbedding
from docusense.classifier.head import ClassifierHead, ClassifierHeadConfig
from docusense.evals.deterministic import evaluate_classifier
from docusense.schemas.classification import IntentLabel

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS = REPO_ROOT / "data" / "sample_contracts"
LABELS = REPO_ROOT / "data" / "golden_intents.jsonl"


@pytest.mark.evals
def test_fast_classifier_meets_accuracy_floor(tmp_path: Path) -> None:
    if not CORPUS.exists() or not LABELS.exists():
        pytest.skip("run scripts/generate_sample_corpus.py first")

    provider = HashedBagOfWordsEmbedding(dim=256)
    docs = {p.stem: p.read_text() for p in sorted(CORPUS.glob("*.txt"))}
    labels: dict[str, IntentLabel] = {}
    for line in LABELS.read_text().splitlines():
        record = json.loads(line)
        labels[record["doc_id"]] = IntentLabel(record["intent"])

    doc_ids = sorted(docs.keys())
    embeddings = np.asarray(provider.embed([docs[d] for d in doc_ids]))
    y = [labels[d] for d in doc_ids]

    head = ClassifierHead(config=ClassifierHeadConfig(n_estimators=100)).fit(embeddings, y)
    artifact = tmp_path / "artifact.joblib"
    head.save(artifact)

    report = evaluate_classifier(CORPUS, LABELS, artifact)
    # Sanity floor — the shipped sample is small and synthetic, but the
    # classifier should still separate the intent labels cleanly.
    assert report.n_samples > 0
    assert report.macro_f1 >= 0.5, f"macro_f1 too low: {report.macro_f1}"
