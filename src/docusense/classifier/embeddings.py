"""Embedding provider — the same interface for Azure OpenAI and a local fake.

The local provider is a deterministic hashed bag-of-words vector. It's
intentionally cheap and awful, but it gives training + serving something
consistent to work with in tests and CI, where we don't want to pay for
real embeddings on every commit.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")


class EmbeddingProvider(ABC):
    """Base class for embedding providers."""

    dim: int

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return a float32 array of shape ``(len(texts), dim)``."""


class HashedBagOfWordsEmbedding(EmbeddingProvider):
    """Deterministic, dependency-free embedding — used locally and in CI.

    Not competitive with real embeddings; it exists so that the fast head
    can be trained and the endpoint tested without network access.
    """

    def __init__(self, dim: int = 384) -> None:
        if dim < 32:
            raise ValueError("dim must be at least 32")
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = _TOKEN_RE.findall(text.lower())
            if not tokens:
                continue
            for token in tokens:
                h = hashlib.blake2b(token.encode(), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                sign = 1.0 if (h[4] & 1) else -1.0
                matrix[i, idx] += sign
            norm = float(np.linalg.norm(matrix[i]))
            if norm > 0:
                matrix[i] /= norm
        return matrix


class AzureOpenAIEmbedding(EmbeddingProvider):
    """Thin wrapper over the Azure OpenAI embeddings API.

    Instantiation is lazy — nothing calls Azure until ``embed`` is invoked,
    so tests can import this module freely.
    """

    dim = 3072  # text-embedding-3-large

    def __init__(self, deployment: str, endpoint: str, api_version: str, api_key: str) -> None:
        self._deployment = deployment
        self._endpoint = endpoint
        self._api_version = api_version
        self._api_key = api_key
        self._client = None

    def _lazy_client(self):  # pragma: no cover — hits network
        if self._client is None:
            from openai import AzureOpenAI

            self._client = AzureOpenAI(
                azure_endpoint=self._endpoint,
                api_version=self._api_version,
                api_key=self._api_key,
            )
        return self._client

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        response = self._lazy_client().embeddings.create(
            model=self._deployment,
            input=list(texts),
        )
        return np.asarray(
            [d.embedding for d in response.data],
            dtype=np.float32,
        )


def build_default_provider() -> EmbeddingProvider:
    """Choose provider based on runtime config — local by default."""
    from docusense.config import get_settings

    settings = get_settings()
    if (
        settings.docusense_env != "local"
        and settings.azure_openai_endpoint
        and settings.azure_openai_key
    ):
        return AzureOpenAIEmbedding(
            deployment=settings.azure_openai_embedding_deployment,
            endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            api_key=settings.azure_openai_key,
        )
    return HashedBagOfWordsEmbedding()
