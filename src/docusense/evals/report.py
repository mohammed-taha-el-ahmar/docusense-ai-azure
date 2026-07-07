"""Report generation for eval runs.

Emits a small HTML + JSON pair so that CI can upload both. The JSON is
the machine-consumable version (used by the PR-gate to compare against
baseline); the HTML is for humans looking at the workflow run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class RunSummary:
    """Combined summary of a single eval run."""

    classifier: dict[str, Any]
    extraction: dict[str, Any] | None = None
    judge: dict[str, Any] | None = None
    generated_at: str = ""


def build_summary(**parts: Any) -> RunSummary:
    def _normalise(obj: Any) -> Any:
        if obj is None:
            return None
        if is_dataclass(obj):
            return asdict(obj)
        return obj

    return RunSummary(
        classifier=_normalise(parts["classifier"]),
        extraction=_normalise(parts.get("extraction")),
        judge=_normalise(parts.get("judge")),
        generated_at=datetime.now(tz=UTC).isoformat(),
    )


def write_json(summary: RunSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(summary), indent=2))


def write_html(summary: RunSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = _render_html(summary)
    path.write_text(body)


def _render_html(summary: RunSummary) -> str:
    def section(title: str, payload: Any) -> str:
        if payload is None:
            return f"<h2>{title}</h2><p><em>skipped</em></p>"
        return f"<h2>{title}</h2><pre>{json.dumps(payload, indent=2)}</pre>"

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>DocuSense evals — {summary.generated_at}</title>
  <style>
    body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
    h1 {{ margin-bottom: 0; }}
    h2 {{ margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.2rem; }}
    pre {{ background: #f6f6f6; padding: 0.75rem; border-radius: 4px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>DocuSense evaluation report</h1>
  <p>Generated at {summary.generated_at}</p>
  {section("Fast classifier", summary.classifier)}
  {section("Field extraction", summary.extraction)}
  {section("LLM-as-judge (reasoning)", summary.judge)}
</body>
</html>
"""


def compare_to_baseline(
    current: dict[str, Any],
    baseline: dict[str, Any],
    tolerances: dict[str, float] | None = None,
) -> list[str]:
    """Return a list of human-readable regressions.

    Metrics compared are ``classifier.macro_f1`` and
    ``judge.mean_score`` when both are present. A regression is anything
    that drops by more than the tolerance for that metric.
    """
    tol = tolerances or {"classifier.macro_f1": 0.03, "judge.mean_score": 0.3}
    findings: list[str] = []
    for key, allowed_drop in tol.items():
        section, name = key.split(".", 1)
        cur = (current.get(section) or {}).get(name)
        base = (baseline.get(section) or {}).get(name)
        if cur is None or base is None:
            continue
        drop = float(base) - float(cur)
        if drop > allowed_drop:
            findings.append(f"{key}: {cur:.3f} < {base:.3f} (drop {drop:.3f} > {allowed_drop:.3f})")
    return findings
