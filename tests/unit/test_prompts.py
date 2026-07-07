"""Snapshot tests on rendered prompts.

Rendered prompts are what actually gets sent to the LLM. Snapshotting a
sample rendering here catches template drift before eval regressions do.
"""

from __future__ import annotations

from docusense.classifier.embeddings import HashedBagOfWordsEmbedding
from docusense.llm.prompting import load_prompt, render_reason_user
from docusense.retrieval.search import InMemoryHybridRetriever
from docusense.schemas.classification import IntentLabel
from docusense.schemas.reasoning import ReasoningRequest


def test_system_prompt_contains_intent_list() -> None:
    prompt = load_prompt("reason_system")
    for intent in ["nda", "msa", "purchase_order", "termination", "price_change"]:
        assert f"`{intent}`" in prompt


def test_system_prompt_enforces_citations() -> None:
    prompt = load_prompt("reason_system")
    assert "citation" in prompt.lower()
    assert "verbatim" in prompt.lower()


def test_user_prompt_includes_hinted_intent(sample_document) -> None:
    request = ReasoningRequest(document=sample_document, hinted_intent=IntentLabel.MSA)
    embedding = HashedBagOfWordsEmbedding(dim=64)
    retriever = InMemoryHybridRetriever(embedding=embedding, chunks=[])
    passages = retriever.search("msa", top_k=3)
    rendered = render_reason_user(request, passages)
    assert "msa" in rendered.lower()
    assert sample_document.doc_id in rendered


def test_user_prompt_truncates_long_documents() -> None:
    from docusense.schemas.document import DocumentPayload

    long_text = "word " * 5000
    request = ReasoningRequest(document=DocumentPayload(doc_id="long", text=long_text))
    embedding = HashedBagOfWordsEmbedding(dim=32)
    retriever = InMemoryHybridRetriever(embedding=embedding, chunks=[])
    rendered = render_reason_user(request, retriever.search("x", top_k=0), max_document_chars=200)
    assert "truncated" in rendered
