"""Run the prompt regression judge.

Scores sampled traces (from App Insights or local fallback) using the
LLM-as-judge pipeline, compares against a baseline, and fails if the
mean score drops below the regression threshold.

Usage:
    # Full local run (samples + judges + compares):
    uv run python scripts/run_prompt_regression.py --local

    # Against real traces (needs App Insights + Azure OpenAI):
    uv run python scripts/run_prompt_regression.py

    # With custom threshold:
    uv run python scripts/run_prompt_regression.py --local --threshold 0.3
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


@dataclass
class RegressionReport:
    """Output of a prompt regression run."""

    timestamp: str
    n_traces: int
    mean_score: float
    per_dimension: dict[str, float]
    baseline_mean: float | None
    delta: float | None
    threshold: float
    regressed: bool
    per_doc: list[dict[str, Any]]


def _load_baseline(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _build_judge(use_fake: bool):
    """Construct a Judge with either a real or fake LLM."""
    from docusense.evals.llm_judge import Judge

    if use_fake:
        from docusense.llm.client import LLMResponse, ScriptedFakeLLM

        def _fake_judge_response(idx: int) -> LLMResponse:
            """Simulate a judge that gives consistently good scores."""
            score = {
                "intent_correctness": 5,
                "citation_grounding": 4,
                "field_extraction_quality": 4,
                "conciseness_and_style": 4,
                "notes": f"Automated fake judge — trace {idx}.",
            }
            return LLMResponse(content=json.dumps(score))

        llm = ScriptedFakeLLM(_fake_judge_response)
    else:
        from docusense.llm.client import AzureOpenAIClient

        llm = AzureOpenAIClient()

    return Judge(llm=llm)


@app.command()
def main(
    traces_file: Path = typer.Option(
        Path("reports/traces.jsonl"),
        help="JSONL file with ReasoningResponse traces to score",
    ),
    baseline_file: Path = typer.Option(
        Path("data/judge_baseline.json"),
        help="Baseline scores to compare against",
    ),
    output: Path = typer.Option(
        Path("reports/regression"),
        help="Output directory for report files",
    ),
    threshold: float = typer.Option(
        0.5,
        help="Max allowed drop in mean score before flagging regression",
    ),
    local: bool = typer.Option(
        False,
        help="Run the full pipeline locally: sample from golden data + fake judge",
    ),
    fake_judge: bool = typer.Option(
        False,
        help="Use fake judge (scripted) instead of real Azure OpenAI",
    ),
    max_traces: int = typer.Option(20, help="Max traces to sample (if sampling)"),
    skip_sample: bool = typer.Option(
        False,
        help="Skip sampling — assume traces_file already exists",
    ),
) -> None:
    """Run the prompt regression pipeline: sample → judge → compare → report."""
    from docusense.evals.llm_judge import JudgeResult
    from docusense.schemas.reasoning import ReasoningResponse

    # ── Step 1: Sample traces ─────────────────────────────────────────
    if not skip_sample:
        typer.echo("── Step 1: Sampling traces ──")
        from subprocess import run as subprocess_run

        sample_cmd = [
            sys.executable,
            "scripts/sample_traces.py",
            "--output",
            str(traces_file),
            "--max-traces",
            str(max_traces),
        ]
        if local:
            sample_cmd.append("--local")
        result = subprocess_run(sample_cmd, capture_output=True, text=True)
        typer.echo(result.stdout.strip())
        if result.returncode != 0:
            typer.echo(f"ERROR: {result.stderr}", err=True)
            raise typer.Exit(code=1)
    else:
        typer.echo("── Step 1: Skipped (using existing traces file) ──")

    if not traces_file.exists():
        typer.echo(f"ERROR: Traces file not found: {traces_file}", err=True)
        raise typer.Exit(code=1)

    # ── Step 2: Load traces ───────────────────────────────────────────
    traces_raw = [
        json.loads(line)
        for line in traces_file.read_text().splitlines()
        if line.strip()
    ]
    if not traces_raw:
        typer.echo("ERROR: No traces to score.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"── Step 2: Loaded {len(traces_raw)} traces ──")

    # ── Step 3: Run the judge ─────────────────────────────────────────
    typer.echo("── Step 3: Running judge ──")
    use_fake = fake_judge or local
    judge = _build_judge(use_fake=use_fake)

    results: list[JudgeResult] = []
    for i, trace_dict in enumerate(traces_raw):
        try:
            response = ReasoningResponse(**trace_dict)
            result = judge.score(response)
            results.append(result)
            typer.echo(
                f"  [{i + 1}/{len(traces_raw)}] {result.doc_id}: "
                f"mean={result.score.mean:.2f}"
            )
        except Exception as e:
            typer.echo(f"  [{i + 1}/{len(traces_raw)}] ERROR: {e}", err=True)

    if not results:
        typer.echo("ERROR: No traces scored successfully.", err=True)
        raise typer.Exit(code=1)

    # ── Step 4: Compute aggregates ────────────────────────────────────
    typer.echo("── Step 4: Computing aggregates ──")
    mean_score = sum(r.score.mean for r in results) / len(results)
    per_dimension = {
        "intent_correctness": sum(r.score.intent_correctness for r in results) / len(results),
        "citation_grounding": sum(r.score.citation_grounding for r in results) / len(results),
        "field_extraction_quality": sum(
            r.score.field_extraction_quality for r in results
        ) / len(results),
        "conciseness_and_style": sum(r.score.conciseness_and_style for r in results) / len(results),
    }

    # ── Step 5: Compare against baseline ──────────────────────────────
    typer.echo("── Step 5: Comparing against baseline ──")
    baseline = _load_baseline(baseline_file)
    baseline_mean: float | None = None
    delta: float | None = None
    regressed = False

    if baseline is not None:
        baseline_mean = baseline.get("mean_score", 0.0)
        delta = mean_score - baseline_mean
        regressed = delta < -threshold
        typer.echo(
            f"  Baseline: {baseline_mean:.3f} | Current: {mean_score:.3f} | "
            f"Δ = {delta:+.3f} (threshold: -{threshold})"
        )
        if regressed:
            typer.echo("  ❌ REGRESSION DETECTED")
        else:
            typer.echo("  ✓ No regression")
    else:
        typer.echo("  ⚠️  No baseline found — saving current run as baseline.")
        baseline_file.parent.mkdir(parents=True, exist_ok=True)
        baseline_file.write_text(
            json.dumps(
                {
                    "mean_score": mean_score,
                    "per_dimension": per_dimension,
                    "generated_at": datetime.now(tz=UTC).isoformat(),
                    "n_traces": len(results),
                },
                indent=2,
            )
        )

    # ── Step 6: Write report ──────────────────────────────────────────
    report = RegressionReport(
        timestamp=datetime.now(tz=UTC).isoformat(),
        n_traces=len(results),
        mean_score=mean_score,
        per_dimension=per_dimension,
        baseline_mean=baseline_mean,
        delta=delta,
        threshold=threshold,
        regressed=regressed,
        per_doc=[
            {
                "doc_id": r.doc_id,
                "mean": r.score.mean,
                "intent_correctness": r.score.intent_correctness,
                "citation_grounding": r.score.citation_grounding,
                "field_extraction_quality": r.score.field_extraction_quality,
                "conciseness_and_style": r.score.conciseness_and_style,
                "notes": r.score.notes,
            }
            for r in results
        ],
    )

    output.mkdir(parents=True, exist_ok=True)
    report_json = output / "regression_report.json"
    report_json.write_text(json.dumps(asdict(report), indent=2))
    typer.echo(f"\n── Report written to {report_json} ──")

    # Summary
    typer.echo(f"\n{'=' * 50}")
    typer.echo(f"  Traces scored:  {report.n_traces}")
    typer.echo(f"  Mean score:     {report.mean_score:.3f} / 5.0")
    for dim, val in report.per_dimension.items():
        typer.echo(f"    {dim}: {val:.2f}")
    if report.delta is not None:
        typer.echo(f"  Delta vs baseline: {report.delta:+.3f}")
    typer.echo(f"  Regressed:      {'YES ❌' if report.regressed else 'NO ✓'}")
    typer.echo(f"{'=' * 50}")

    if regressed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
