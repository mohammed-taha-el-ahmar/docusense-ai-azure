"""Evaluate component entrypoint.

Runs the deterministic evals and writes ``metrics.json``, ``report.html``.
The judge suite is intentionally excluded here — it lives in a separate
CI workflow because it costs money.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer

from docusense.evals.deterministic import evaluate_classifier
from docusense.evals.report import build_summary, write_html, write_json

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    corpus: Path = typer.Argument(..., exists=True, file_okay=False),
    labels: Path = typer.Argument(..., exists=True),
    classifier: Path = typer.Argument(..., exists=True),
    output: Path = typer.Argument(...),
    min_macro_f1: float = typer.Argument(0.5),
) -> None:
    output.mkdir(parents=True, exist_ok=True)

    classifier_report = evaluate_classifier(
        corpus_dir=corpus,
        labels_path=labels,
        classifier_path=classifier,
    )
    summary = build_summary(classifier=classifier_report)
    write_json(summary, output / "metrics.json")
    write_html(summary, output / "report.html")

    typer.echo(json.dumps(asdict(classifier_report), indent=2))
    if classifier_report.macro_f1 < min_macro_f1:
        typer.echo(
            f"regression: macro_f1 {classifier_report.macro_f1:.3f} < {min_macro_f1:.3f}",
            err=True,
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
