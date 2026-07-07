"""Generate a small, deterministic corpus of synthetic contract-like documents.

Twelve documents, roughly two per intent class (plus a couple of ``other``),
each written to a plain-text file. Alongside them we emit the golden intent
labels, gold extractions, and a small set of judge prompts.

None of this content is a real legal document — it's stylised text designed
to be easy for the fast classifier to separate and to give the LLM path
believable material to reason over.
"""

from __future__ import annotations

import json
from pathlib import Path

DOCUMENTS: list[dict[str, str]] = [
    # NDA
    {
        "doc_id": "nda-01",
        "intent": "nda",
        "text": (
            "MUTUAL NON-DISCLOSURE AGREEMENT\n\n"
            'This Mutual Non-Disclosure Agreement ("Agreement") is entered into as of '
            'March 14, 2024, by and between ACME Corp ("Discloser") and Globex Ltd '
            '("Recipient"). The parties intend to explore a potential business '
            "relationship and, in connection therewith, may disclose confidential "
            "information to each other. Recipient agrees to hold all Confidential "
            "Information in strict confidence for a period of three (3) years and "
            "shall not disclose it to any third party without prior written consent. "
            "This Agreement is governed by the laws of the State of Delaware."
        ),
    },
    {
        "doc_id": "nda-02",
        "intent": "nda",
        "text": (
            "CONFIDENTIALITY AGREEMENT\n\n"
            "Effective January 5, 2025, Initech LLC and Umbrella Systems agree that "
            "any proprietary technical information exchanged in the course of the "
            "collaboration shall be treated as confidential. The confidentiality "
            "obligations survive for five years following termination. Governing law: "
            "England and Wales."
        ),
    },
    # MSA
    {
        "doc_id": "msa-01",
        "intent": "msa",
        "text": (
            "MASTER SERVICES AGREEMENT\n\n"
            "This Master Services Agreement is made effective as of July 1, 2024, "
            'between ACME Corp ("Customer") and Vandelay Consulting ("Provider"). '
            "Provider shall provide professional services under one or more Statements "
            "of Work. Payment terms are net thirty (30) days from a correct invoice. "
            "Either party may terminate for convenience upon thirty (30) days' written "
            "notice. Aggregate liability is capped at fees paid in the preceding twelve "
            "months."
        ),
    },
    {
        "doc_id": "msa-02",
        "intent": "msa",
        "text": (
            "SERVICES FRAMEWORK AGREEMENT\n\n"
            "Between Globex Ltd (Buyer) and Stark Industries (Supplier), effective "
            "September 15, 2024. Supplier shall deliver engineering and integration "
            "services under separate work orders. Initial term is 24 months with "
            "auto-renewal for successive 12-month periods unless terminated. Net-45 "
            "payment. Governed by New York law."
        ),
    },
    # Purchase order
    {
        "doc_id": "po-01",
        "intent": "purchase_order",
        "text": (
            "PURCHASE ORDER 8842\n\n"
            "Buyer: ACME Corp\nSupplier: Wonka Industries\n"
            "Item: 500 units of Widget-A at $12.50 per unit\n"
            "Total: $6,250\n"
            "Delivery: on or before April 20, 2024, to the ACME Corp Chicago warehouse.\n"
            "Payment terms: net-30 upon receipt of goods."
        ),
    },
    {
        "doc_id": "po-02",
        "intent": "purchase_order",
        "text": (
            "PURCHASE ORDER 9013\n\n"
            "Buyer: Globex Ltd\nSupplier: Cyberdyne Systems\n"
            "Line 1: 12 x Model-T300 chassis at $9,500 each\n"
            "Line 2: 24 x Model-T400 controllers at $2,750 each\n"
            "Grand total: $180,000\n"
            "Delivery FOB destination on May 5, 2025."
        ),
    },
    # RFP
    {
        "doc_id": "rfp-01",
        "intent": "rfp",
        "text": (
            "REQUEST FOR PROPOSAL\n\n"
            "ACME Corp invites qualified vendors to submit proposals for the "
            "implementation of a new customer data platform. Proposals must be "
            "received by 5pm CET on August 1, 2024. The evaluation criteria include "
            "technical capability (40%), price (30%), delivery timeline (20%), and "
            "customer references (10%). Proposals should be submitted to procurement "
            "at ACME Corp."
        ),
    },
    {
        "doc_id": "rfp-02",
        "intent": "rfp",
        "text": (
            "REQUEST FOR TENDER — CLOUD MIGRATION\n\n"
            "Globex Ltd is soliciting proposals to migrate its on-premises workloads "
            "to a hyperscale cloud environment. Interested vendors must submit "
            "responses by October 3, 2024. The award criteria emphasise security "
            "(35%), cost of ownership (30%), migration approach (25%), and prior "
            "public-sector experience (10%)."
        ),
    },
    # Termination
    {
        "doc_id": "term-01",
        "intent": "termination",
        "text": (
            "NOTICE OF TERMINATION\n\n"
            "Pursuant to Section 12.2 of the Master Services Agreement dated "
            "July 1, 2024 between ACME Corp and Vandelay Consulting, this letter "
            "serves as formal notice that ACME Corp is terminating the Agreement "
            "for convenience effective March 15, 2025. Final invoices should be "
            "submitted no later than April 15, 2025."
        ),
    },
    {
        "doc_id": "term-02",
        "intent": "termination",
        "text": (
            "TERMINATION NOTICE\n\n"
            "Globex Ltd hereby provides ninety (90) days' notice of termination of "
            "the Framework Agreement dated September 15, 2024 with Stark Industries. "
            "The Agreement will end on June 30, 2025. All outstanding obligations "
            "shall be settled within thirty days of the termination effective date."
        ),
    },
    # Price change
    {
        "doc_id": "price-01",
        "intent": "price_change",
        "text": (
            "PRICE ADJUSTMENT NOTICE\n\n"
            "Vandelay Consulting will apply a 4.5% annual price adjustment across "
            "all rates under the Master Services Agreement with ACME Corp, effective "
            "January 1, 2025. This adjustment is within the Consumer Price Index cap "
            "specified in Section 6.3 of the Agreement. Please contact your account "
            "manager with any questions."
        ),
    },
    {
        "doc_id": "price-02",
        "intent": "price_change",
        "text": (
            "PRICING UPDATE\n\n"
            "Effective February 15, 2025, Stark Industries will increase unit prices "
            "on Model-T400 controllers by 6.0%. New unit price: $2,915. Existing "
            "purchase orders confirmed before this date will honour the previous "
            "pricing."
        ),
    },
    # Other
    {
        "doc_id": "other-01",
        "intent": "other",
        "text": (
            "Meeting minutes — Q3 kickoff\n\n"
            "Attendees: A. Chen, R. Patel, D. Ford.\n"
            "Discussed roadmap for the analytics platform, agreed to revisit "
            "prioritisation of the SSO integration item next sprint. Action items "
            "captured in the shared tracker."
        ),
    },
]


EXTRACTIONS: dict[str, dict[str, str]] = {
    "nda-01": {
        "effective_date": "March 14, 2024",
        "counterparty": "Globex Ltd",
        "governing_law": "Delaware",
    },
    "msa-01": {"effective_date": "July 1, 2024", "payment_terms": "net 30"},
    "po-01": {"total_value": "$6,250", "delivery_date": "April 20, 2024"},
    "term-01": {"effective_date": "March 15, 2025", "counterparty": "Vandelay Consulting"},
    "price-01": {"effective_date": "January 1, 2025", "adjustment": "4.5%"},
}


JUDGE_PROMPTS = [
    {
        "doc_id": "msa-01",
        "prompt": "Classify and reason over document msa-01.",
        "reference_intent": "msa",
    },
    {
        "doc_id": "term-01",
        "prompt": "Classify and reason over document term-01.",
        "reference_intent": "termination",
    },
    {
        "doc_id": "price-01",
        "prompt": "Classify and reason over document price-01.",
        "reference_intent": "price_change",
    },
]


def generate(root: Path) -> None:
    corpus_dir = root / "sample_contracts"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    intents_path = root / "golden_intents.jsonl"
    extractions_path = root / "golden_extractions.jsonl"
    judge_path = root / "judge_prompts.jsonl"

    for record in DOCUMENTS:
        (corpus_dir / f"{record['doc_id']}.txt").write_text(record["text"])

    with intents_path.open("w") as f:
        for record in DOCUMENTS:
            f.write(json.dumps({"doc_id": record["doc_id"], "intent": record["intent"]}) + "\n")

    with extractions_path.open("w") as f:
        for doc_id, fields in EXTRACTIONS.items():
            f.write(json.dumps({"doc_id": doc_id, "fields": fields}) + "\n")

    with judge_path.open("w") as f:
        for record in JUDGE_PROMPTS:
            f.write(json.dumps(record) + "\n")

    print(f"wrote {len(DOCUMENTS)} documents to {corpus_dir}")
    print(f"wrote {len(DOCUMENTS)} intent labels to {intents_path}")
    print(f"wrote {len(EXTRACTIONS)} gold extractions to {extractions_path}")
    print(f"wrote {len(JUDGE_PROMPTS)} judge prompts to {judge_path}")


if __name__ == "__main__":
    generate(Path("data"))
