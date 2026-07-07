"""Classifier head tests — train tiny model, predict, round-trip persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from docusense.classifier.embeddings import HashedBagOfWordsEmbedding
from docusense.classifier.head import ClassifierHead, ClassifierHeadConfig
from docusense.schemas.classification import IntentLabel


def test_predict_round_trip(tmp_path: Path) -> None:
    provider = HashedBagOfWordsEmbedding(dim=64)
    texts = [
        "confidentiality nondisclosure recipient discloser",
        "confidential proprietary information party",
        "master services agreement provider customer",
        "professional services statements of work",
        "purchase order 500 units delivery",
        "buyer supplier line items delivery net-30",
    ]
    labels = [
        IntentLabel.NDA,
        IntentLabel.NDA,
        IntentLabel.MSA,
        IntentLabel.MSA,
        IntentLabel.PURCHASE_ORDER,
        IntentLabel.PURCHASE_ORDER,
    ]
    embeddings = provider.embed(texts)
    head = ClassifierHead(config=ClassifierHeadConfig(n_estimators=30)).fit(
        np.asarray(embeddings), labels
    )
    preds, confs = head.predict(np.asarray(embeddings))
    assert len(preds) == len(texts)
    assert all(0.0 <= c <= 1.0 for c in confs)

    path = tmp_path / "artifact.joblib"
    head.save(path)
    loaded = ClassifierHead.load(path)
    preds_loaded, _ = loaded.predict(np.asarray(embeddings))
    assert preds_loaded == preds
