"""Smoke-test a deployed endpoint by exercising /classify and /reason.

Runs against the local FastAPI mirror by default; point ``--url`` at an
AML endpoint URL to exercise a real deployment. When ``--api-key`` is
given, sent as a Bearer token.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    url: str = typer.Option("http://localhost:8000"),
    api_key: str | None = typer.Option(None),
    doc: Path = typer.Option(Path("data/sample_contracts/msa-01.txt"), exists=True),
) -> None:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {"document": {"doc_id": doc.stem, "text": doc.read_text()}}
    with httpx.Client(timeout=30.0) as client:
        classify = client.post(f"{url}/classify", json=payload, headers=headers)
        classify.raise_for_status()
        typer.echo(f"classify: {json.dumps(classify.json(), indent=2)}")

        reason = client.post(f"{url}/reason", json=payload, headers=headers)
        if reason.status_code >= 400:
            typer.echo(f"reason FAILED: {reason.status_code} {reason.text}", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"reason: {json.dumps(reason.json(), indent=2)}")


if __name__ == "__main__":
    app()
