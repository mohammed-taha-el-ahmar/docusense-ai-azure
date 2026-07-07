"""Training entrypoint for the fast classifier.

Reads text files from a corpus directory and a JSONL of labels, embeds
them via the configured provider, fits the head, and dumps everything
so the endpoint can load a single ``artifact.joblib`` at startup.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
import numpy as np
import typer
from sklearn.metrics import classification_report, f1_score

from docusense.classifier.embeddings import build_default_provider
from docusense.classifier.head import ClassifierHead, ClassifierHeadConfig
from docusense.schemas.classification import IntentLabel

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _read_labels(path: Path) -> dict[str, IntentLabel]:
    labels: dict[str, IntentLabel] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        labels[record["doc_id"]] = IntentLabel(record["intent"])
    return labels


def _read_corpus(corpus: Path) -> dict[str, str]:
    return {p.stem: p.read_text() for p in sorted(corpus.glob("*.txt"))}


@app.command()
def main(
    corpus: Path = typer.Argument(..., exists=True, file_okay=False),
    labels: Path = typer.Argument(..., exists=True),
    output_dir: Path = typer.Argument(Path("outputs/classifier")),
    tracking_uri: str = typer.Argument("file:./mlruns"),
    experiment_name: str = typer.Option("docusense-fast-classifier"),
) -> None:
    """Fit the fast head and persist it alongside its metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect if running inside an AML job (platform pre-creates the MLflow run).
    _in_aml = bool(os.environ.get("MLFLOW_RUN_ID") or os.environ.get("AZUREML_RUN_ID"))

    if not _in_aml:
        # Local run: set tracking URI and experiment explicitly.
        if tracking_uri and tracking_uri not in ("azureml://", "azureml:"):
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    documents = _read_corpus(corpus)
    label_map = _read_labels(labels)
    missing = set(documents) - set(label_map)
    if missing:
        raise typer.BadParameter(f"documents without labels: {sorted(missing)}")

    doc_ids = list(documents.keys())
    texts = [documents[d] for d in doc_ids]
    y = [label_map[d] for d in doc_ids]

    provider = build_default_provider()
    embeddings = provider.embed(texts)

    with mlflow.start_run() as run:
        config = ClassifierHeadConfig()
        mlflow.log_params(config.to_dict())
        mlflow.log_param("embedding_provider", type(provider).__name__)
        mlflow.log_param("embedding_dim", provider.dim)
        mlflow.log_param("n_documents", len(doc_ids))
        mlflow.log_param("n_classes", len({label.value for label in y}))

        head = ClassifierHead(config=config).fit(embeddings, y)
        preds, _ = head.predict(embeddings)
        macro_f1 = float(
            f1_score([lbl.value for lbl in y], [p.value for p in preds], average="macro")
        )
        mlflow.log_metric("train_macro_f1", macro_f1)

        model_path = output_dir / "artifact.joblib"
        head.save(model_path)
        report = classification_report(
            [lbl.value for lbl in y],
            [p.value for p in preds],
            zero_division=0,
        )
        report_path = output_dir / "report.txt"
        report_path.write_text(report)

        # Persist provider name so the endpoint can pick the matching one.
        (output_dir / "provider.json").write_text(
            json.dumps(
                {"provider": type(provider).__name__, "dim": provider.dim},
                indent=2,
            )
        )

        # In AML, outputs are captured via the pipeline output mount;
        # mlflow.log_artifact fails due to mlflow/mlflow-skinny version skew
        # in the curated environment, so skip it when running on the platform.
        if not _in_aml:
            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(report_path))
        typer.echo(f"run_id={run.info.run_id} macro_f1={macro_f1:.3f}")

    # Silence unused-variable warnings for embeddings/y in reduced paths.
    _ = np.asarray(embeddings)


if __name__ == "__main__":
    app()
