"""End-to-end integration test.

Exercises the classifier training entrypoint on the shipped corpus,
then the full ``/reason`` route through the FastAPI mirror using a
scripted fake LLM. This is the closest thing to a smoke test the CI
runs without touching Azure OpenAI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from docusense.classifier.train import app as train_app
from docusense.retrieval.chunker import chunk_document
from docusense.retrieval.search import InMemoryHybridRetriever
from docusense.schemas.document import DocumentPayload

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS = REPO_ROOT / "data" / "sample_contracts"
LABELS = REPO_ROOT / "data" / "golden_intents.jsonl"


@pytest.mark.integration
def test_train_then_serve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_llm_factory
) -> None:
    if not CORPUS.exists() or not LABELS.exists():
        pytest.skip("run scripts/generate_sample_corpus.py first")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")

    runner = CliRunner()
    output_dir = tmp_path / "classifier"
    result = runner.invoke(
        train_app,
        [
            "--corpus",
            str(CORPUS),
            "--labels",
            str(LABELS),
            "--output-dir",
            str(output_dir),
            "--tracking-uri",
            f"file:{tmp_path / 'mlruns'}",
        ],
    )
    assert result.exit_code == 0, result.output
    artifact = output_dir / "artifact.joblib"
    assert artifact.exists()

    from docusense.classifier.embeddings import HashedBagOfWordsEmbedding
    from docusense.serving import local_app

    monkeypatch.setattr(local_app, "CLASSIFIER_PATH", artifact)
    monkeypatch.setattr(local_app, "CORPUS_PATH", CORPUS)
    local_app._load_classifier.cache_clear()

    document = DocumentPayload(
        doc_id="msa-01",
        text=(CORPUS / "msa-01.txt").read_text(),
    )
    embedding = HashedBagOfWordsEmbedding(dim=128)
    chunks = chunk_document(document, max_tokens=80, overlap_tokens=10)
    retriever = InMemoryHybridRetriever(embedding=embedding, chunks=chunks)
    local_app.set_retriever(retriever)

    valid = {
        "doc_id": document.doc_id,
        "intent": "msa",
        "confidence": 0.9,
        "reasoning": "Services agreement with net-30 payment terms.",
        "extracted_fields": [],
        "citations": [{"chunk_id": chunks[0].chunk_id, "quote": "Master Services Agreement"}],
        "tools_used": [],
        "model_version": "",
        "tokens_in": 0,
        "tokens_out": 0,
        "latency_ms": 0.0,
    }
    local_app.set_llm_client(fake_llm_factory([{"content": valid}]))
    try:
        client = TestClient(local_app.app)
        classify = client.post("/classify", json={"document": document.model_dump(mode="json")})
        assert classify.status_code == 200
        body = classify.json()
        assert body["intent"] in {
            "nda",
            "msa",
            "purchase_order",
            "rfp",
            "termination",
            "price_change",
            "other",
        }

        reason = client.post("/reason", json={"document": document.model_dump(mode="json")})
        assert reason.status_code == 200, reason.text
        assert reason.json()["intent"] == "msa"
    finally:
        local_app.set_llm_client(None)
        local_app.set_retriever(None)

    _ = json  # silence unused
