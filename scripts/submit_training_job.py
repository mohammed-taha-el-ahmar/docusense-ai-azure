"""Submit the AML training pipeline for the fast classifier."""

from __future__ import annotations

from pathlib import Path

import typer

from docusense.config import get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    pipeline: Path = typer.Option(Path("aml/pipelines/training_pipeline.yml"), exists=True),
    wait: bool = typer.Option(True),
) -> None:  # pragma: no cover — talks to Azure
    from azure.ai.ml import MLClient, load_job
    from azure.identity import DefaultAzureCredential

    settings = get_settings()
    if not all(
        [settings.azure_subscription_id, settings.azure_resource_group, settings.aml_workspace_name]
    ):
        raise typer.BadParameter("Azure settings missing — check your .env")

    client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=settings.azure_subscription_id,
        resource_group_name=settings.azure_resource_group,
        workspace_name=settings.aml_workspace_name,
    )
    job = load_job(source=pipeline)

    # Inject Azure credentials into pipeline steps so containers can access
    # services (AI Search, OpenAI, Content Safety) without a .env file.
    _inject_env_vars(job, settings)

    submitted = client.jobs.create_or_update(job)
    typer.echo(f"submitted: {submitted.name}")
    typer.echo(f"studio url: {submitted.studio_url}")
    if wait:
        client.jobs.stream(submitted.name)
        final = client.jobs.get(submitted.name)
        typer.echo(f"final status: {final.status}")
        if final.status != "Completed":
            raise typer.Exit(code=1)


def _inject_env_vars(job, settings) -> None:
    """Inject Azure service credentials into each pipeline step."""
    env_vars = {
        k: v
        for k, v in {
            "AZURE_OPENAI_ENDPOINT": settings.azure_openai_endpoint,
            "AZURE_OPENAI_KEY": settings.azure_openai_key,
            "AZURE_OPENAI_CHAT_DEPLOYMENT": settings.azure_openai_chat_deployment,
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": settings.azure_openai_embedding_deployment,
            "AZURE_SEARCH_ENDPOINT": settings.azure_search_endpoint,
            "AZURE_SEARCH_KEY": settings.azure_search_key,
            "AZURE_SEARCH_INDEX": settings.azure_search_index,
            "CONTENT_SAFETY_ENDPOINT": settings.content_safety_endpoint,
            "CONTENT_SAFETY_KEY": settings.content_safety_key,
        }.items()
        if v  # skip None/empty
    }
    if not env_vars:
        return

    # Pipeline jobs have a .jobs dict of child steps
    if hasattr(job, "jobs") and job.jobs:
        for step_name in job.jobs:
            step = job.jobs[step_name]
            if not hasattr(step, "environment_variables") or step.environment_variables is None:
                step.environment_variables = {}
            step.environment_variables.update(env_vars)
        typer.echo(f"injected {len(env_vars)} env vars into {len(job.jobs)} pipeline step(s)")
    else:
        # Single command job
        if not hasattr(job, "environment_variables") or job.environment_variables is None:
            job.environment_variables = {}
        job.environment_variables.update(env_vars)
        typer.echo(f"injected {len(env_vars)} env vars into job")


if __name__ == "__main__":
    app()
