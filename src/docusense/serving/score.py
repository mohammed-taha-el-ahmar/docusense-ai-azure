"""Azure ML managed-endpoint scoring script.

AML calls ``init()`` once and ``run(raw_data)`` per request. We support
two routes on the same endpoint by dispatching on the ``route`` field of
the request payload — AML endpoints are single-URL, so multiplexing at
the payload level is the standard pattern.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from docusense.classifier.embeddings import build_default_provider
from docusense.classifier.head import ClassifierHead
from docusense.config import get_settings
from docusense.guardrails.citations import CitationViolation
from docusense.guardrails.pii import PIIRedactor
from docusense.guardrails.safety import OutputSafetyChecker, UnsafeOutputError
from docusense.llm.client import AzureOpenAIClient, LLMClientError
from docusense.llm.pipeline import ReasoningPipeline
from docusense.retrieval.search import AzureAISearchRetriever
from docusense.schemas.classification import ClassifyRequest, ClassifyResponse
from docusense.schemas.reasoning import ReasoningRequest

_log = logging.getLogger("docusense.serving")

_HEAD: ClassifierHead | None = None
_PIPELINE: ReasoningPipeline | None = None
_MODEL_VERSION: str = "unknown"


def init() -> None:  # pragma: no cover — AML runtime
    global _HEAD, _PIPELINE, _MODEL_VERSION
    model_dir = os.environ.get("AZUREML_MODEL_DIR")
    if model_dir is None:
        raise RuntimeError("AZUREML_MODEL_DIR not set — is this running in an AML endpoint?")

    # Model may be at <model_dir>/artifact.joblib or <model_dir>/classifier/artifact.joblib
    # depending on how it was registered (directory vs subdirectory).
    artifact_path = Path(model_dir) / "artifact.joblib"
    if not artifact_path.exists():
        artifact_path = Path(model_dir) / "classifier" / "artifact.joblib"
    _HEAD = ClassifierHead.load(artifact_path)
    _MODEL_VERSION = os.environ.get("AZUREML_MODEL_VERSION", "unknown")

    settings = get_settings()
    embedding = build_default_provider()
    retriever = AzureAISearchRetriever(
        endpoint=settings.azure_search_endpoint or "",
        index_name=settings.azure_search_index,
        key=settings.azure_search_key or "",
        embedding=embedding,
    )
    _PIPELINE = ReasoningPipeline(
        llm=AzureOpenAIClient(),
        retriever=retriever,
        pii=PIIRedactor(),
        safety=OutputSafetyChecker(),
    )
    _log.info("model loaded: version=%s", _MODEL_VERSION)


def run(raw_data: str) -> str:  # pragma: no cover — AML runtime
    if _HEAD is None or _PIPELINE is None:
        raise RuntimeError("init() has not run")
    payload = json.loads(raw_data)
    route = payload.pop("route", "auto")

    if route == "classify":
        return _handle_classify(payload)
    if route == "reason":
        return _handle_reason(payload)
    if route == "auto":
        fast_response = json.loads(_handle_classify(payload))
        if fast_response["route"] == "fast":
            return json.dumps(fast_response)
        return _handle_reason(payload)
    return json.dumps({"error": f"unknown route: {route}"})


def _handle_classify(payload: dict) -> str:
    assert _HEAD is not None
    request = ClassifyRequest.model_validate(payload)
    settings = get_settings()
    provider = build_default_provider()
    start = time.perf_counter()
    embedding = provider.embed([request.document.text])
    labels, confidences = _HEAD.predict(np.asarray(embedding))
    label, confidence = labels[0], float(confidences[0])
    resp = ClassifyResponse(
        doc_id=request.document.doc_id,
        intent=label,
        confidence=confidence,
        route="fast"
        if confidence >= settings.classifier_confidence_threshold
        else "escalate_to_reason",
        latency_ms=(time.perf_counter() - start) * 1000.0,
        model_version=_MODEL_VERSION,
    )
    return resp.model_dump_json()


def _handle_reason(payload: dict) -> str:
    assert _PIPELINE is not None
    request = ReasoningRequest.model_validate(payload)
    try:
        outcome = _PIPELINE.run(request)
    except (LLMClientError, CitationViolation, UnsafeOutputError) as e:
        return json.dumps({"error": type(e).__name__, "detail": str(e)})
    resp = outcome.response.model_copy(update={"model_version": _MODEL_VERSION})
    return resp.model_dump_json()
