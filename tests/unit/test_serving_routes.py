"""Serving route tests — /health, /classify, /reason with mocked LLM."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from docusense.classifier.embeddings import HashedBagOfWordsEmbedding
from docusense.classifier.head import ClassifierHead, ClassifierHeadConfig
from docusense.llm.client import ToolCall
from docusense.retrieval.chunker import chunk_document
from docusense.retrieval.search import InMemoryHybridRetriever
from docusense.schemas.classification import IntentLabel


def _train_head(tmp_path: Path) -> Path:
    provider = HashedBagOfWordsEmbedding(dim=64)
    texts = [
        "nondisclosure confidential recipient discloser",
        "confidential proprietary party term",
        "master services agreement provider statement of work",
        "professional services network net thirty payment",
        "purchase order units total delivery net-30",
        "buyer supplier delivery line items grand total",
    ]
    labels = [
        IntentLabel.NDA,
        IntentLabel.NDA,
        IntentLabel.MSA,
        IntentLabel.MSA,
        IntentLabel.PURCHASE_ORDER,
        IntentLabel.PURCHASE_ORDER,
    ]
    head = ClassifierHead(config=ClassifierHeadConfig(n_estimators=30)).fit(
        np.asarray(provider.embed(texts)), labels
    )
    path = tmp_path / "artifact.joblib"
    head.save(path)
    return path


def test_health(monkeypatch, tmp_path: Path) -> None:
    from docusense.serving import local_app

    monkeypatch.setattr(local_app, "CLASSIFIER_PATH", tmp_path / "no.joblib")
    monkeypatch.setattr(local_app, "CORPUS_PATH", tmp_path / "no_corpus")
    local_app._load_classifier.cache_clear()
    client = TestClient(local_app.app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["classifier_loaded"] is False


def test_classify_route(monkeypatch, tmp_path: Path, sample_document) -> None:
    from docusense.serving import local_app

    model_path = _train_head(tmp_path)
    monkeypatch.setattr(local_app, "CLASSIFIER_PATH", model_path)
    local_app._load_classifier.cache_clear()

    client = TestClient(local_app.app)
    response = client.post(
        "/classify",
        json={"document": sample_document.model_dump(mode="json")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["doc_id"] == sample_document.doc_id
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["route"] in {"fast", "escalate_to_reason"}


def test_reason_route_with_fake_llm(
    monkeypatch, tmp_path: Path, sample_document, fake_llm_factory
) -> None:
    from docusense.serving import local_app

    # classifier not strictly needed for /reason, but health check touches path
    model_path = _train_head(tmp_path)
    monkeypatch.setattr(local_app, "CLASSIFIER_PATH", model_path)
    local_app._load_classifier.cache_clear()

    # Build retriever with the sample document as its only corpus
    embedding = HashedBagOfWordsEmbedding(dim=64)
    chunks = chunk_document(sample_document, max_tokens=80, overlap_tokens=10)
    retriever = InMemoryHybridRetriever(embedding=embedding, chunks=chunks)
    local_app.set_retriever(retriever)

    chunk_id = chunks[0].chunk_id
    valid = {
        "doc_id": sample_document.doc_id,
        "intent": "msa",
        "confidence": 0.9,
        "reasoning": "Services agreement with net-30 payment terms.",
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
    local_app.set_llm_client(fake_llm_factory([{"content": valid}]))

    try:
        client = TestClient(local_app.app)
        response = client.post(
            "/reason",
            json={"document": sample_document.model_dump(mode="json")},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["intent"] == "msa"
        assert body["citations"][0]["chunk_id"] == chunk_id
    finally:
        local_app.set_llm_client(None)
        local_app.set_retriever(None)


def test_reason_route_rejects_hallucinated_citation(
    monkeypatch, tmp_path: Path, sample_document, fake_llm_factory
) -> None:
    from docusense.serving import local_app

    model_path = _train_head(tmp_path)
    monkeypatch.setattr(local_app, "CLASSIFIER_PATH", model_path)
    local_app._load_classifier.cache_clear()

    embedding = HashedBagOfWordsEmbedding(dim=64)
    chunks = chunk_document(sample_document, max_tokens=80, overlap_tokens=10)
    retriever = InMemoryHybridRetriever(embedding=embedding, chunks=chunks)
    local_app.set_retriever(retriever)

    bad = {
        "doc_id": sample_document.doc_id,
        "intent": "msa",
        "confidence": 0.9,
        "reasoning": "Services agreement with net-30 payment terms.",
        "extracted_fields": [],
        "citations": [{"chunk_id": "ghost#0000", "quote": "not in the doc"}],
        "tools_used": [],
        "model_version": "",
        "tokens_in": 0,
        "tokens_out": 0,
        "latency_ms": 0.0,
    }
    local_app.set_llm_client(fake_llm_factory([{"content": bad}]))
    try:
        client = TestClient(local_app.app)
        response = client.post(
            "/reason",
            json={"document": sample_document.model_dump(mode="json")},
        )
        assert response.status_code == 422
    finally:
        local_app.set_llm_client(None)
        local_app.set_retriever(None)


# unused-imports guard
_ = json
_ = ToolCall
