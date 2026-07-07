# Demo Coverage Matrix

Honest inventory of what this repository proves vs. what remains as TODO.

## What the scaffold proves

| Capability | Evidence |
| --- | --- |
| Reproducible env with `uv` and `ruff` | `pyproject.toml`, `Makefile`, `.python-version` |
| Ruff lint + format enforced in CI | `.github/workflows/ci.yml` |
| Prompts as versioned `.md` files | `src/docusense/llm/prompts/{reason_system,classify_system,judge_rubric}.md` |
| Snapshot tests on rendered prompts | `tests/unit/test_prompts.py` |
| Pydantic contracts at every API + LLM boundary | `src/docusense/schemas/` — `document`, `classification`, `reasoning`, `tools` |
| Fast head (LightGBM over embeddings) with MLflow tracking | `src/docusense/classifier/{embeddings,head,train}.py` |
| Deterministic hashed-BOW embedding for offline dev + CI | `HashedBagOfWordsEmbedding` in `embeddings.py` |
| Chunker with overlap + tiktoken counting | `src/docusense/retrieval/chunker.py` |
| Hybrid retrieval interface with in-memory + AI Search backends | `src/docusense/retrieval/search.py` |
| LLM client wrapper with retries + timeout | `src/docusense/llm/client.py` |
| `ScriptedFakeLLM` for tests and evals | same file |
| Tool declarations generated from Pydantic models | `src/docusense/llm/tools.py` |
| Structured output with strict JSON schema | `src/docusense/llm/structured.py`, tested |
| Bounded tool-calling loop | `src/docusense/llm/pipeline.py` — `max_turns` cap |
| PII scrub (regex fallback + Presidio when installed) | `src/docusense/guardrails/pii.py` |
| Citation guardrail with fuzzy verbatim check | `src/docusense/guardrails/citations.py` |
| Output safety with local heuristic + Azure Content Safety branch | `src/docusense/guardrails/safety.py` |
| OpenTelemetry spans around LLM + retrieval | `src/docusense/telemetry/traces.py` |
| Per-request cost estimator | `src/docusense/telemetry/cost.py` |
| Deterministic classifier + extraction evals | `src/docusense/evals/deterministic.py` |
| LLM-as-judge with rubric | `src/docusense/evals/llm_judge.py` |
| Eval report generator (JSON + HTML) | `src/docusense/evals/report.py` |
| FastAPI mirror of the AML scoring script | `src/docusense/serving/local_app.py` |
| AML endpoint entry with route multiplexing | `src/docusense/serving/score.py` |
| Terraform for AML + Azure OpenAI + AI Search + Content Safety + KV + storage + monitoring | `infra/` |
| CI: lint + unit + integration tests, all offline | `.github/workflows/ci.yml` |
| Terraform plan on PR, apply on push, OIDC to Azure | `.github/workflows/terraform.yml` |
| Manual-dispatch index+train pipeline | `.github/workflows/index_and_train.yml` |
| Tag-triggered blue/green deployment with smoke test | `.github/workflows/deploy_endpoint.yml` |
| Sample corpus + golden intents + gold extractions + judge prompts | `data/` |

## Scaffolded but requires live Azure to exercise

| Item | Status |
| --- | --- |
| Push chunks + embeddings to Azure AI Search | `scripts/build_index.py --push-to-search`; needs Search + OpenAI creds |
| Submit the AML training/indexing pipelines | `scripts/submit_training_job.py`; needs workspace |
| Real Azure OpenAI reasoning path | `AzureOpenAIClient` in `llm/client.py`; needs endpoint + key |
| Content Safety Azure call | `_azure_check` in `guardrails/safety.py` |
| Presidio-backed PII redaction | Auto-picked when the `azure` extra is installed |
| App Insights custom events + OTel exporter | Wired via config but tracer is no-op until `configure_azure_monitor` is called in serving init |
| Blue/green traffic shift | Deployment YAML present; workflow smoke-tests but does not include a ramp helper |

## Explicit TODOs — not attempted in this scaffold

- **Judge-on-real-LLM CI job.** `evals.yml` has the shape and the trigger, but the step body is a placeholder — the deterministic + fake-LLM regression is what runs today. Wiring a real judge model with a per-PR cost budget and a comparison against a baseline JSON is the next step.
- **Model card generation.** Each classifier version should emit a signed markdown card (dataset, metrics, intended use); not done here.
- **Cost budget enforcement.** `telemetry/cost.py` estimates per-request USD; the endpoint does not yet emit a `usage_over_budget` event or apply a per-tenant hard cap.
- **Recursive clause-boundary chunking.** Current chunker is fixed-token with overlap — good enough for the demo, but recursive splitting at section/clause boundaries would be a natural upgrade.
- **Reranker.** No cross-encoder or LLM reranker yet; hybrid retrieval + semantic-ranker is the current best.
- **Judge diversity.** The judge runs a single model. Rotating judges (or ensembling) would reduce same-family bias.
- **Networking hardening.** Workspace + OpenAI account run with public network access enabled for demo simplicity. A production variant would use Private Endpoint + Private DNS.

## Why this scoping

An AI Engineer portfolio project is more credible when it demonstrates the *engineering* around LLMs — evals, guardrails, tool calling, structured outputs, cost accounting — than when it stacks features. Every "does it work offline" box is ticked so a reviewer can `git clone` + `make install` + `make test` and see green in under a minute; every "would this survive production" box that isn't ticked is called out here rather than hidden behind a mock.
