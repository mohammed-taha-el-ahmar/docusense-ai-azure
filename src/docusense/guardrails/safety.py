"""Output-side safety check.

Wraps Azure Content Safety when configured; falls back to a no-op with
a keyword blocklist for the local dev path. The idea is: production
behaviour (block on Content Safety judgement) is preserved even without
the SDK, at a much crappier fidelity, so tests can exercise the branch.
"""

from __future__ import annotations

import logging
import re

from docusense.config import get_settings

_log = logging.getLogger("docusense.safety")

_BLOCK_KEYWORDS = ("kill ", "harm ", "destroy the ", "attack the ")


class UnsafeOutputError(RuntimeError):
    """Raised when a generated response fails a safety check."""


class OutputSafetyChecker:
    def __init__(self, endpoint: str | None = None, key: str | None = None) -> None:
        settings = get_settings()
        self._endpoint = endpoint or settings.content_safety_endpoint
        self._key = key or settings.content_safety_key
        self._client = None

    def _lazy_client(self):  # pragma: no cover — needs Azure
        if self._client is None and self._endpoint and self._key:
            from azure.ai.contentsafety import ContentSafetyClient
            from azure.core.credentials import AzureKeyCredential

            self._client = ContentSafetyClient(self._endpoint, AzureKeyCredential(self._key))
        return self._client

    def check(self, text: str) -> None:
        client = self._lazy_client()
        if client is None:
            self._local_check(text)
            return
        self._azure_check(client, text)  # pragma: no cover

    def _local_check(self, text: str) -> None:
        pattern = re.compile("|".join(re.escape(k) for k in _BLOCK_KEYWORDS), re.IGNORECASE)
        if pattern.search(text):
            raise UnsafeOutputError("local safety heuristic flagged the output")

    def _azure_check(self, client, text: str) -> None:  # pragma: no cover
        from azure.ai.contentsafety.models import AnalyzeTextOptions

        response = client.analyze_text(AnalyzeTextOptions(text=text))
        for category in response.categories_analysis:
            if getattr(category, "severity", 0) >= 4:
                raise UnsafeOutputError(
                    f"Content Safety flagged category {category.category} (severity={category.severity})"
                )
