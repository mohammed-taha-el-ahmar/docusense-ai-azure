"""PII redaction tests — regex backend."""

from __future__ import annotations

from docusense.guardrails.pii import RegexPIIRedactor


def test_masks_email() -> None:
    redactor = RegexPIIRedactor()
    result = redactor.redact("Please contact alice.jones@example.com about the deal.")
    assert "alice.jones@example.com" not in result
    assert "[EMAIL]" in result


def test_masks_phone() -> None:
    redactor = RegexPIIRedactor()
    result = redactor.redact("Call +1 415 555 0132 to confirm.")
    assert "555 0132" not in result
    assert "[PHONE]" in result


def test_masks_credit_card_like() -> None:
    redactor = RegexPIIRedactor()
    result = redactor.redact("Card 4111 1111 1111 1111 was used.")
    assert "4111 1111 1111 1111" not in result
    assert "[CARD]" in result


def test_leaves_clean_text_unchanged() -> None:
    redactor = RegexPIIRedactor()
    text = "This is a normal contract paragraph without PII."
    assert redactor.redact(text) == text
