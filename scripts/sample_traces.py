"""Sample production traces from App Insights for the judge regression.

Pulls recent /reason responses from App Insights custom events and writes
them as JSONL for the judge to score. Falls back to the golden judge
prompts if App Insights is not configured or has no recent traces.

Usage:
    # From App Insights (needs APPINSIGHTS_APP_ID + APPINSIGHTS_API_KEY):
    uv run python scripts/sample_traces.py --output traces.jsonl --max-traces 20

    # Fallback to local golden data (always works):
    uv run python scripts/sample_traces.py --output traces.jsonl --local
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from docusense.config import get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _fetch_from_appinsights(
    app_id: str,
    api_key: str,
    max_traces: int,
    days_back: int,
) -> list[dict]:
    """Query App Insights REST API for recent /reason responses."""
    import httpx

    query = f"""
    customEvents
    | where timestamp > ago({days_back}d)
    | where name == "docusense.reason.response"
    | project timestamp, customDimensions
    | order by timestamp desc
    | take {max_traces}
    """

    resp = httpx.post(
        f"https://api.applicationinsights.io/v1/apps/{app_id}/query",
        headers={"x-api-key": api_key},
        json={"query": query},
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()

    traces = []
    rows = data.get("tables", [{}])[0].get("rows", [])
    columns = data.get("tables", [{}])[0].get("columns", [])
    col_names = [c["name"] for c in columns]

    for row in rows:
        row_dict = dict(zip(col_names, row, strict=False))
        dims = row_dict.get("customDimensions")
        if dims:
            if isinstance(dims, str):
                dims = json.loads(dims)
            # Expect the response payload to be stored in a known field
            response_json = dims.get("response") or dims.get("reasoning_response")
            if response_json:
                if isinstance(response_json, str):
                    response_json = json.loads(response_json)
                traces.append(response_json)

    return traces


def _load_local_fallback(
    corpus_dir: Path,
    judge_prompts_path: Path,
) -> list[dict]:
    """Build synthetic ReasoningResponse objects from golden data for local testing.

    This lets you run the full judge pipeline locally without App Insights,
    simulating what production traces would look like.
    """
    from docusense.evals.datasets import load_judge_prompts
    from docusense.schemas.classification import IntentLabel
    from docusense.schemas.reasoning import Citation, ReasoningResponse

    prompts = load_judge_prompts(judge_prompts_path)
    responses = []

    for jp in prompts:
        doc_path = corpus_dir / f"{jp.doc_id}.txt"
        text_snippet = ""
        if doc_path.exists():
            text_snippet = doc_path.read_text()[:200]

        # Build a plausible response structure that the judge can score
        response = ReasoningResponse(
            doc_id=jp.doc_id,
            intent=jp.reference_intent,
            confidence=0.92,
            reasoning=f"Document identified as {jp.reference_intent.value}. "
            f"Key indicators found in the opening section.",
            citations=[
                Citation(
                    chunk_id=f"{jp.doc_id}_chunk_0000",
                    quote=text_snippet[:100] if text_snippet else "Sample clause text.",
                )
            ],
            extracted_fields=[],
            tools_used=[],
            model_version="gpt-5.1",
            tokens_in=500,
            tokens_out=200,
            latency_ms=1200.0,
        )
        responses.append(response.model_dump(mode="json"))

    return responses


@app.command()
def main(
    output: Path = typer.Option(Path("reports/traces.jsonl")),
    max_traces: int = typer.Option(20, help="Max traces to sample from App Insights"),
    days_back: int = typer.Option(7, help="Look back N days in App Insights"),
    local: bool = typer.Option(False, help="Use local golden data instead of App Insights"),
    corpus_dir: Path = typer.Option(Path("data/sample_contracts"), exists=True),
    judge_prompts: Path = typer.Option(Path("data/judge_prompts.jsonl"), exists=True),
) -> None:
    """Sample traces for the prompt regression judge."""
    output.parent.mkdir(parents=True, exist_ok=True)

    if local:
        typer.echo("Using local fallback (golden data)...")
        traces = _load_local_fallback(corpus_dir, judge_prompts)
    else:
        settings = get_settings()
        app_id = settings.appinsights_app_id or ""
        api_key = settings.appinsights_api_key or ""
        if not (app_id and api_key):
            typer.echo(
                "⚠️  APPINSIGHTS_APP_ID / APPINSIGHTS_API_KEY not set. "
                "Falling back to local golden data."
            )
            traces = _load_local_fallback(corpus_dir, judge_prompts)
        else:
            typer.echo(f"Querying App Insights (last {days_back} days, max {max_traces})...")
            traces = _fetch_from_appinsights(app_id, api_key, max_traces, days_back)
            if not traces:
                typer.echo("⚠️  No traces found in App Insights. Falling back to local data.")
                traces = _load_local_fallback(corpus_dir, judge_prompts)

    output.write_text("\n".join(json.dumps(t) for t in traces) + "\n")
    typer.echo(f"✓ Wrote {len(traces)} traces to {output}")


if __name__ == "__main__":
    app()
