"""Build the retrieval index — local (JSONL) or push to Azure AI Search."""

from __future__ import annotations

from pathlib import Path

import typer

from docusense.pipelines.index import main as index_main

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def cli(
    corpus: Path = typer.Option(Path("data/sample_contracts"), exists=True),
    output: Path = typer.Option(Path("outputs/index")),
    push_to_search: bool = typer.Option(False),
) -> None:
    index_main(corpus=corpus, output=output, push_to_search=push_to_search)


if __name__ == "__main__":
    app()
