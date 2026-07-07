"""Text chunking.

Fixed-size chunks with overlap. This is the simplest approach that works
for contract-like text; recursive splitting on clause boundaries is a
natural next step and is called out in DEMO.md.

Encoding preference is ``tiktoken`` (BPE, ~= what the model actually
counts). When the tiktoken data files can't be fetched — offline sandbox,
firewalled build, hostile network — we fall back to a whitespace
tokeniser so tests and local dev keep working. Chunk *ids* and structure
are stable across backends; only the exact chunk boundaries shift.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from docusense.schemas.document import DocumentChunk, DocumentPayload


class _Encoder(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...


class _WhitespaceEncoder:
    """Deterministic offline fallback."""

    def __init__(self) -> None:
        self._word_list: list[str] = []
        self._index: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        tokens: list[int] = []
        for word in text.split(" "):
            if word not in self._index:
                self._index[word] = len(self._word_list)
                self._word_list.append(word)
            tokens.append(self._index[word])
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return " ".join(self._word_list[t] for t in tokens)


_cached_encoder: _Encoder | None = None


def _get_encoder(name: str = "cl100k_base") -> _Encoder:
    global _cached_encoder
    if _cached_encoder is not None:
        return _cached_encoder
    try:
        import tiktoken

        _cached_encoder = tiktoken.get_encoding(name)
    except Exception:
        _cached_encoder = _WhitespaceEncoder()
    return _cached_encoder


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    return len(_get_encoder(encoding_name).encode(text))


def chunk_document(
    doc: DocumentPayload,
    max_tokens: int = 400,
    overlap_tokens: int = 40,
    encoding_name: str = "cl100k_base",
) -> list[DocumentChunk]:
    """Split ``doc.text`` into overlapping token windows."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be in [0, max_tokens)")

    encoder = _get_encoder(encoding_name)
    tokens = encoder.encode(doc.text)
    if not tokens:
        return []

    step = max_tokens - overlap_tokens
    chunks: list[DocumentChunk] = []
    for i, start in enumerate(range(0, len(tokens), step)):
        window = tokens[start : start + max_tokens]
        text = encoder.decode(window)
        chunks.append(
            DocumentChunk(
                chunk_id=f"{doc.doc_id}_chunk_{i:04d}",
                doc_id=doc.doc_id,
                chunk_index=i,
                text=text,
                n_tokens=len(window),
            )
        )
        if start + max_tokens >= len(tokens):
            break
    return chunks


def chunk_documents(
    docs: Iterable[DocumentPayload],
    **kwargs,
) -> list[DocumentChunk]:
    result: list[DocumentChunk] = []
    for doc in docs:
        result.extend(chunk_document(doc, **kwargs))
    return result
