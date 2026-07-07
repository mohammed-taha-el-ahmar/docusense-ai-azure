"""Local FastAPI mirror of the AML online endpoint.

Two routes, matching the deployed endpoint's contract:

- ``POST /classify`` — fast head only, milliseconds.
- ``POST /reason``   — full LLM pipeline with retrieval + tools.

Both use the same handler code as the AML entry script, so a request
that works locally works in AML.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from docusense.classifier.embeddings import build_default_provider
from docusense.classifier.head import ClassifierHead
from docusense.config import get_settings
from docusense.guardrails.citations import CitationViolation
from docusense.guardrails.pii import PIIRedactor
from docusense.guardrails.safety import OutputSafetyChecker, UnsafeOutputError
from docusense.llm.client import LLMClient, LLMClientError
from docusense.llm.pipeline import ReasoningPipeline
from docusense.retrieval.indexer import build_in_memory_retriever, read_corpus
from docusense.retrieval.search import Retriever
from docusense.schemas.classification import (
    ClassifyRequest,
    ClassifyResponse,
    IntentLabel,
)
from docusense.schemas.reasoning import ReasoningRequest, ReasoningResponse

CLASSIFIER_PATH = Path(os.environ.get("CLASSIFIER_PATH", "outputs/classifier/artifact.joblib"))
CORPUS_PATH = Path(os.environ.get("CORPUS_PATH", "data/sample_contracts"))

app = FastAPI(title="DocuSense local", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Overridable module-level state so tests can swap components without
# monkeypatching internals.
_llm_client: LLMClient | None = None
_retriever: Retriever | None = None


def set_llm_client(client: LLMClient | None) -> None:
    global _llm_client
    _llm_client = client


def set_retriever(retriever: Retriever | None) -> None:
    global _retriever
    _retriever = retriever


@lru_cache(maxsize=1)
def _load_classifier() -> tuple[ClassifierHead, object]:
    if not CLASSIFIER_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"classifier artifact missing at {CLASSIFIER_PATH}. "
                "Run `make train-classifier` first."
            ),
        )
    head = ClassifierHead.load(CLASSIFIER_PATH)
    provider = _build_matching_provider(head)
    return head, provider


def _build_matching_provider(head: ClassifierHead) -> object:
    """Build an embedding provider whose dimension matches the trained head.

    In local runs the head is trained against the hashed provider at a
    given dim; the endpoint must query at the same dim or the LightGBM
    predictor will reject the input shape. In Azure runs the head was
    trained against the Azure OpenAI embedding and both sides use the
    real provider, so this branch is a no-op there.
    """
    default = build_default_provider()
    expected = head._artifact.embedding_dim if head._artifact else default.dim
    if default.dim == expected:
        return default
    # Local training used a smaller hashed dim — rebuild one to match.
    from docusense.classifier.embeddings import HashedBagOfWordsEmbedding

    return HashedBagOfWordsEmbedding(dim=expected)


def _get_retriever() -> Retriever:
    if _retriever is not None:
        return _retriever
    if not CORPUS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"corpus not found at {CORPUS_PATH}. Run `make sample-corpus` first.",
        )
    docs = read_corpus(CORPUS_PATH)
    provider = build_default_provider()
    return build_in_memory_retriever(docs, provider)


def _get_llm() -> LLMClient:
    if _llm_client is not None:
        return _llm_client
    # Real Azure OpenAI — only used if env is configured.
    from docusense.llm.client import AzureOpenAIClient

    return AzureOpenAIClient()


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "classifier_loaded": CLASSIFIER_PATH.exists(),
        "corpus_present": CORPUS_PATH.exists(),
    }


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest) -> ClassifyResponse:
    head, provider = _load_classifier()
    settings = get_settings()
    start = time.perf_counter()
    embedding = provider.embed([request.document.text])
    labels, confidences = head.predict(np.asarray(embedding))
    label, confidence = labels[0], float(confidences[0])
    route = (
        "fast" if confidence >= settings.classifier_confidence_threshold else "escalate_to_reason"
    )
    return ClassifyResponse(
        doc_id=request.document.doc_id,
        intent=label,
        confidence=confidence,
        route=route,
        latency_ms=(time.perf_counter() - start) * 1000.0,
    )


@app.post("/reason", response_model=ReasoningResponse)
def reason(request: ReasoningRequest) -> ReasoningResponse:
    try:
        pipeline = ReasoningPipeline(
            llm=_get_llm(),
            retriever=_get_retriever(),
            pii=PIIRedactor(),
            safety=OutputSafetyChecker(),
        )
        outcome = pipeline.run(request)
    except LLMClientError as e:
        raise HTTPException(status_code=502, detail=f"LLM upstream failed: {e}") from e
    except CitationViolation as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except UnsafeOutputError as e:
        raise HTTPException(status_code=451, detail=str(e)) from e
    return outcome.response


# Convenience for demos: a single "auto" route that classifies first and
# only escalates to reasoning if the confidence is below threshold.
@app.post("/auto", response_model=ReasoningResponse | ClassifyResponse)
def auto(request: ReasoningRequest) -> ReasoningResponse | ClassifyResponse:
    fast = classify(ClassifyRequest(document=request.document))
    if fast.route == "fast":
        return fast
    return reason(request)


# ─── AML endpoint proxy (avoids browser CORS issues) ────────────────────────
import httpx  # noqa: E402
from pydantic import BaseModel as _BaseModel  # noqa: E402


class _ProxyRequest(_BaseModel):
    scoring_uri: str
    api_key: str
    body: dict


@app.post("/proxy/score")
def proxy_score(req: _ProxyRequest):
    """Forward a request to the Azure ML endpoint server-side, bypassing CORS."""
    try:
        resp = httpx.post(
            req.scoring_uri,
            json=req.body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {req.api_key}",
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text) from e
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


# Best-effort: preload the intent enum so route validation errors are
# helpful in dev.
_ALL_INTENTS = [i.value for i in IntentLabel]
