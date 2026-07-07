"""Prompt loading and rendering.

Prompts are shipped inside the package (see ``pyproject.toml`` force-include)
so they are available whether the package is installed as a wheel or run
from source. Rendered prompts are what CI snapshots against, so this
module is deliberately small and testable.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from jinja2 import Environment

from docusense.retrieval.search import RetrievedPassage
from docusense.schemas.document import DocumentPayload
from docusense.schemas.reasoning import ReasoningRequest

_PROMPTS_DIR = Path(__file__).parent / "prompts"

_env = Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)


@lru_cache(maxsize=8)
def load_prompt(name: str) -> str:
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"prompt not found: {name} (looked in {path})")
    return path.read_text()


_USER_TEMPLATE = _env.from_string(
    """\
Document id: {{ doc_id }}
{% if hinted_intent %}Fast-classifier hint: {{ hinted_intent }}
{% endif %}
Document text:
---
{{ document_text }}
---

Retrieved passages:
{% for p in passages %}
[{{ p.chunk.chunk_id }}] score={{ '%.3f' % p.score }}
{{ p.chunk.text }}

{% endfor %}
Return the JSON object per the schema. Use the retrieved passages for citations.
"""
)


def render_reason_user(
    request: ReasoningRequest,
    passages: list[RetrievedPassage],
    max_document_chars: int = 8000,
) -> str:
    """Render the user message for the reasoning route."""
    doc: DocumentPayload = request.document
    text = (
        doc.text
        if len(doc.text) <= max_document_chars
        else doc.text[:max_document_chars] + "\n[…truncated…]"
    )
    return _USER_TEMPLATE.render(
        doc_id=doc.doc_id,
        document_text=text,
        hinted_intent=request.hinted_intent.value if request.hinted_intent else None,
        passages=passages,
    )
