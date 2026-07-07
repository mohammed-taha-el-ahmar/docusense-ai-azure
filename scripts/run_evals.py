"""Run the deterministic eval suite locally.

For the LLM-as-judge suite, see ``tests/evals/`` and the ``evals``
GitHub workflow — that one is intentionally not runnable via this script
so people don't accidentally rack up an OpenAI bill.
"""

from __future__ import annotations

from pathlib import Path

import typer

from docusense.evals.deterministic import evaluate_classifier
from docusense.evals.report import build_summary, write_html, write_json

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    corpus: Path = typer.Option(Path("data/sample_contracts"), exists=True),
    labels: Path = typer.Option(Path("data/golden_intents.jsonl"), exists=True),
    classifier: Path = typer.Option(Path("outputs/classifier/artifact.joblib"), exists=True),
    output: Path = typer.Option(Path("reports/evals")),
) -> None:
    classifier_report = evaluate_classifier(corpus, labels, classifier)
    summary = build_summary(classifier=classifier_report)
    write_json(summary, output / "metrics.json")
    write_html(summary, output / "report.html")
    typer.echo(
        f"macro_f1={classifier_report.macro_f1:.3f} accuracy={classifier_report.accuracy:.3f}"
    )


if __name__ == "__main__":
    app()
