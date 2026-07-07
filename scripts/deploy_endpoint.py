"""Deploy or update the online endpoint hosting DocuSense."""

from __future__ import annotations

from pathlib import Path

import typer

from docusense.config import get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    endpoint_yaml: Path = typer.Option(Path("aml/endpoints/online_endpoint.yml"), exists=True),
    deployment_yaml: Path = typer.Option(Path("aml/endpoints/online_deployment.yml"), exists=True),
    deployment_name: str = typer.Option("blue"),
) -> None:  # pragma: no cover — talks to Azure
    from azure.ai.ml import MLClient, load_online_deployment, load_online_endpoint
    from azure.identity import DefaultAzureCredential

    settings = get_settings()
    client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=settings.azure_subscription_id,
        resource_group_name=settings.azure_resource_group,
        workspace_name=settings.aml_workspace_name,
    )

    endpoint = load_online_endpoint(source=endpoint_yaml)
    client.online_endpoints.begin_create_or_update(endpoint).result()
    typer.echo(f"endpoint ready: {endpoint.name}")

    deployment = load_online_deployment(source=deployment_yaml)
    deployment.name = deployment_name

    # Inject secrets from .env at deploy time — never hardcode in YAML.
    _inject_env_vars(deployment, settings)

    client.online_deployments.begin_create_or_update(deployment).result()
    typer.echo(f"deployment ready: {deployment.name}")

    # Route 100% traffic to the new deployment.
    endpoint.traffic = {deployment_name: 100}
    client.online_endpoints.begin_create_or_update(endpoint).result()
    typer.echo(f"traffic routed: {deployment_name}=100%")


def _inject_env_vars(deployment, settings) -> None:
    """Merge Azure service credentials into the deployment's environment_variables."""
    secrets = {
        "AZURE_OPENAI_ENDPOINT": settings.azure_openai_endpoint or "",
        "AZURE_OPENAI_KEY": settings.azure_openai_key or "",
        "AZURE_OPENAI_CHAT_DEPLOYMENT": settings.azure_openai_chat_deployment,
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": settings.azure_openai_embedding_deployment,
        "AZURE_OPENAI_API_VERSION": settings.azure_openai_api_version,
        "AZURE_SEARCH_ENDPOINT": settings.azure_search_endpoint or "",
        "AZURE_SEARCH_KEY": settings.azure_search_key or "",
        "AZURE_SEARCH_INDEX": settings.azure_search_index,
        "CONTENT_SAFETY_ENDPOINT": settings.content_safety_endpoint or "",
        "CONTENT_SAFETY_KEY": settings.content_safety_key or "",
    }
    if deployment.environment_variables is None:
        deployment.environment_variables = {}
    deployment.environment_variables.update(secrets)


if __name__ == "__main__":
    app()
