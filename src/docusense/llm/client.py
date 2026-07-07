"""LLM client wrapper.

A single ``LLMClient`` protocol has two implementations:

- ``AzureOpenAIClient`` — real calls, with retries + timeout + cost tags.
- ``ScriptedFakeLLM`` — deterministic responses for unit and eval tests.

The protocol shape is small on purpose: one ``complete`` method taking
messages, tools, and a response schema. The reasoning pipeline calls it
in a bounded loop (max 4 turns) to handle tool calls.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from docusense.config import get_settings


@dataclass
class ToolCall:
    """A tool the model asked to invoke."""

    id: str
    name: str
    arguments: str  # raw JSON string emitted by the model


@dataclass
class LLMResponse:
    """Normalised response from any LLM backend."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""
    finish_reason: str = "stop"


class LLMClient(Protocol):
    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


class LLMClientError(Exception):
    """Raised when the LLM call fails after retries."""


class AzureOpenAIClient:
    """Azure OpenAI chat client with structured-output + tool-calling support."""

    def __init__(
        self,
        deployment: str | None = None,
        endpoint: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        settings = get_settings()
        self._deployment = deployment or settings.azure_openai_chat_deployment
        self._endpoint = endpoint or settings.azure_openai_endpoint
        self._api_version = api_version or settings.azure_openai_api_version
        self._api_key = api_key or settings.azure_openai_key
        self._timeout = timeout_seconds or settings.llm_timeout_seconds
        self._max_retries = max_retries if max_retries is not None else settings.llm_max_retries
        self._client = None

    def _lazy_client(self):  # pragma: no cover — needs SDK
        if self._client is None:
            from openai import AzureOpenAI

            if not (self._endpoint and self._api_key):
                raise LLMClientError("Azure OpenAI endpoint or key not configured.")
            self._client = AzureOpenAI(
                azure_endpoint=self._endpoint,
                api_version=self._api_version,
                api_key=self._api_key,
                timeout=self._timeout,
            )
        return self._client

    def complete(  # pragma: no cover — network call
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        @retry(
            reraise=True,
            stop=stop_after_attempt(max(1, self._max_retries)),
            wait=wait_random_exponential(min=0.5, max=8),
            retry=retry_if_exception_type(Exception),
        )
        def _call():
            kwargs: dict[str, Any] = {
                "model": self._deployment,
                "messages": list(messages),
                "temperature": temperature,
            }
            if tools:
                kwargs["tools"] = list(tools)
                kwargs["tool_choice"] = "auto"
            if response_schema is not None:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "docusense_reasoning",
                        "schema": response_schema,
                        "strict": True,
                    },
                }
            return self._lazy_client().chat.completions.create(**kwargs)

        try:
            response = _call()
        except Exception as e:
            raise LLMClientError(str(e)) from e

        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content,
            tool_calls=[
                ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
                for tc in (choice.message.tool_calls or [])
            ],
            tokens_in=response.usage.prompt_tokens,
            tokens_out=response.usage.completion_tokens,
            model=response.model,
            finish_reason=choice.finish_reason,
        )


class ScriptedFakeLLM:
    """Deterministic LLM used in tests.

    Given a queue of ``LLMResponse`` values, each call pops the head. Handy
    for testing the tool-calling loop (first response has tool_calls, next
    has the final content), for testing guardrails, and for evals against
    known-good outputs.
    """

    def __init__(self, script: Sequence[LLMResponse] | Callable[[int], LLMResponse]) -> None:
        if callable(script):
            self._callable = script
            self._responses: list[LLMResponse] = []
        else:
            self._callable = None
            self._responses = list(script)
        self._n_calls = 0

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        response_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        idx = self._n_calls
        self._n_calls += 1
        if self._callable is not None:
            return self._callable(idx)
        if not self._responses:
            raise LLMClientError("ScriptedFakeLLM exhausted")
        return self._responses.pop(0)

    @property
    def n_calls(self) -> int:
        return self._n_calls
