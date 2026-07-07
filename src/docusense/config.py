"""Runtime configuration.

Every environment-driven value lives here. Tests build ``Settings(...)``
directly rather than mutating ``os.environ``, which keeps them fast and
parallelisable.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    docusense_env: str = Field(default="local", description="local | dev | prod")

    # Azure core
    azure_subscription_id: str | None = None
    azure_resource_group: str | None = None
    azure_tenant_id: str | None = None

    # Azure OpenAI
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str = "2024-08-01-preview"
    azure_openai_key: str | None = None
    azure_openai_chat_deployment: str = "gpt-4o"
    azure_openai_embedding_deployment: str = "text-embedding-3-large"

    # Azure AI Search
    azure_search_endpoint: str | None = None
    azure_search_key: str | None = None
    azure_search_index: str = "docusense-clauses"

    # Azure ML
    aml_workspace_name: str | None = None
    aml_endpoint_name: str = "docusense-online"

    # Storage
    storage_account: str | None = None
    storage_container_raw: str = "raw"
    storage_container_evals: str = "evals"
    storage_container_traces: str = "traces"

    # Content Safety
    content_safety_endpoint: str | None = None
    content_safety_key: str | None = None

    # Monitoring
    appinsights_connection_string: str | None = None
    appinsights_app_id: str | None = None
    appinsights_api_key: str | None = None

    # Runtime knobs
    classifier_confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    llm_max_tokens: int = Field(default=1024, gt=0)
    llm_timeout_seconds: float = Field(default=15.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0)

    @property
    def is_local(self) -> bool:
        return self.docusense_env == "local"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
