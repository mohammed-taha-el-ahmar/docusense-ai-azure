"""Safety guardrail tests — local blocklist heuristic."""

from __future__ import annotations

import pytest

from docusense.guardrails.safety import OutputSafetyChecker, UnsafeOutputError


def test_local_check_allows_normal_text() -> None:
    checker = OutputSafetyChecker(endpoint=None, key=None)
    checker.check("This document is a services agreement between two parties.")


def test_local_check_blocks_known_keyword() -> None:
    checker = OutputSafetyChecker(endpoint=None, key=None)
    with pytest.raises(UnsafeOutputError):
        checker.check("The plan is to kill the deal today.")
