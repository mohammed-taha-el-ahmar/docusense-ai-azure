"""Embedding provider tests — deterministic and unit-normalised."""

from __future__ import annotations

import numpy as np

from docusense.classifier.embeddings import HashedBagOfWordsEmbedding


def test_shape_and_dtype() -> None:
    provider = HashedBagOfWordsEmbedding(dim=64)
    vec = provider.embed(["hello world foo bar"])
    assert vec.shape == (1, 64)
    assert vec.dtype == np.float32


def test_deterministic() -> None:
    provider = HashedBagOfWordsEmbedding(dim=64)
    a = provider.embed(["repeatable input"])
    b = provider.embed(["repeatable input"])
    np.testing.assert_array_equal(a, b)


def test_unit_norm_when_non_empty() -> None:
    provider = HashedBagOfWordsEmbedding(dim=64)
    vec = provider.embed(["some meaningful tokens here"])[0]
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_empty_input_produces_zeros() -> None:
    provider = HashedBagOfWordsEmbedding(dim=64)
    vec = provider.embed([""])[0]
    assert np.allclose(vec, 0.0)
