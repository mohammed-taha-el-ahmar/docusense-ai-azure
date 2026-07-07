"""PII redaction.

Two backends:

- ``RegexPIIRedactor`` — dependency-free, catches obvious things (emails,
  phones, common credit-card patterns). Not exhaustive; that's the point.
- ``PresidioPIIRedactor`` — proper NER-backed redaction via Microsoft
  Presidio, selected automatically when it can be imported.

The class ``PIIRedactor`` is what the rest of the code depends on; it
picks the best backend it can find on first use.
"""

from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?:(?<=\D)|^)(?:\+?\d[\s\-()]?){9,14}\d(?=\D|$)")
_CC_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")


class RegexPIIRedactor:
    """Cheap PII masking used when Presidio isn't available."""

    def redact(self, text: str) -> str:
        text = _EMAIL_RE.sub("[EMAIL]", text)
        text = _CC_RE.sub("[CARD]", text)
        text = _PHONE_RE.sub("[PHONE]", text)
        return text


class PresidioPIIRedactor:
    """NER-backed redactor. Lazy import so tests without Presidio still work."""

    def __init__(self) -> None:
        self._analyzer = None
        self._anonymizer = None

    def _load(self) -> None:  # pragma: no cover — needs presidio
        if self._analyzer is not None:
            return
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()

    def redact(self, text: str) -> str:  # pragma: no cover
        self._load()
        assert self._analyzer is not None and self._anonymizer is not None
        results = self._analyzer.analyze(text=text, language="en")
        return self._anonymizer.anonymize(text=text, analyzer_results=results).text


class PIIRedactor:
    """Default redactor. Picks the best available backend on first use."""

    def __init__(self) -> None:
        self._impl: RegexPIIRedactor | PresidioPIIRedactor | None = None

    def redact(self, text: str) -> str:
        if self._impl is None:
            self._impl = _select_backend()
        return self._impl.redact(text)


def _select_backend() -> RegexPIIRedactor | PresidioPIIRedactor:
    try:  # pragma: no cover — presidio is optional
        import presidio_analyzer  # noqa: F401

        return PresidioPIIRedactor()
    except ImportError:
        return RegexPIIRedactor()
