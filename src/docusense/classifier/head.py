"""LightGBM head over embeddings — the fast path.

Design decisions worth defending in an interview:

- Head over embeddings, not fine-tuning. With ~12 documents per class,
  fine-tuning would overfit hard; a head over off-the-shelf embeddings
  is the honest baseline.
- LightGBM rather than logistic regression. Non-linear, cheap, and
  handles the interaction between hashed features well.
- Multi-class one-vs-rest via LightGBM's native multiclass mode.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

from docusense.schemas.classification import IntentLabel


@dataclass
class ClassifierHeadConfig:
    n_estimators: int = 200
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 5
    class_weight: str | None = "balanced"
    random_state: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClassifierArtifact:
    model: lgb.LGBMClassifier
    classes_: list[str] = field(default_factory=list)
    embedding_dim: int = 0


class ClassifierHead:
    """Multiclass head over an embedding provider's output."""

    def __init__(self, config: ClassifierHeadConfig | None = None) -> None:
        self.config = config or ClassifierHeadConfig()
        self._artifact: ClassifierArtifact | None = None

    def fit(self, embeddings: np.ndarray, labels: list[IntentLabel]) -> ClassifierHead:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if len(labels) != embeddings.shape[0]:
            raise ValueError("labels and embeddings length mismatch")
        model = lgb.LGBMClassifier(objective="multiclass", **self.config.to_dict())
        model.fit(embeddings, [label.value for label in labels])
        self._artifact = ClassifierArtifact(
            model=model,
            classes_=[str(c) for c in model.classes_],
            embedding_dim=embeddings.shape[1],
        )
        return self

    def predict(self, embeddings: np.ndarray) -> tuple[list[IntentLabel], np.ndarray]:
        """Return (predicted labels, confidence per prediction)."""
        artifact = self._require_fitted()
        proba = artifact.model.predict_proba(embeddings)
        idx = np.argmax(proba, axis=1)
        labels = [IntentLabel(artifact.classes_[i]) for i in idx]
        confidences = proba[np.arange(len(idx)), idx]
        return labels, confidences.astype(float)

    def save(self, path: str | Path) -> None:
        artifact = self._require_fitted()
        joblib.dump(
            {
                "config": self.config.to_dict(),
                "model": artifact.model,
                "classes_": artifact.classes_,
                "embedding_dim": artifact.embedding_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> ClassifierHead:
        payload = joblib.load(path)
        instance = cls(ClassifierHeadConfig(**payload["config"]))
        instance._artifact = ClassifierArtifact(
            model=payload["model"],
            classes_=payload["classes_"],
            embedding_dim=payload["embedding_dim"],
        )
        return instance

    def _require_fitted(self) -> ClassifierArtifact:
        if self._artifact is None:
            raise RuntimeError("classifier head is not fitted")
        return self._artifact
