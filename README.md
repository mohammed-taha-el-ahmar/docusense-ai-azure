# DocuSense — Document Intent Classifier + LLM Reasoning on Azure

[![CI](https://github.com/mohammed-taha-el-ahmar/docusense-ai-azure/actions/workflows/ci.yml/badge.svg)](https://github.com/mohammed-taha-el-ahmar/docusense-ai-azure/actions/workflows/ci.yml)


![Architecture](https://github.com/mohammed-taha-el-ahmar/docusense-ai-azure/blob/main/docs/img/docusense-ai-azure-demo.png)


AI Engineer portfolio project. Two-route document intelligence service on Azure ML managed endpoints: a millisecond-scale intent classifier for cheap traffic, and a GPT-4o-backed reasoning route with RAG, tool calling, structured outputs, and guardrails.

## What it demonstrates

- **Prompt engineering** — prompts as versioned files, snapshot-tested, judged in CI, hot-swappable without redeploying code.
- **Structured outputs** — Azure OpenAI `response_format=json_schema` with a Pydantic-derived schema; the response is *always* parseable.
- **Tool calling** — two tools (`lookup_clause_library`, `check_counterparty_history`), declared from Pydantic argument models, executed in a bounded loop with a hard turn budget.
- **RAG** — hybrid vector + keyword retrieval; Azure AI Search in production, in-memory in CI, same interface.
- **Guardrails** — Presidio PII scrub inbound, mandatory citation enforcement, Azure Content Safety on output.
- **LLM-as-judge evals** — golden reasoning prompts scored by a rubric-driven judge with a fake-LLM regression test in CI.
- **Fast + slow routing** — a LightGBM head over embeddings answers 80% of traffic without calling GPT-4o at all.
- **Infra as code** — Terraform for the whole footprint including Azure OpenAI + AI Search + Content Safety.
- **CI/CD** — Ruff + pytest + integration tests on PR, eval regression checks, terraform plan/apply, tag-triggered blue/green deployment.

## Architecture

```
                    ┌────────────────────────── AML Online Endpoint ─────────────────────────┐
POST /classify ────►│  fast head (LGBM over embeddings)  ──────► route: fast | escalate      │
                    │                                                                        │
POST /reason   ────►│  PII scrub → hybrid retrieval → GPT-4o (tools + json_schema)           │
                    │                                       ↓                                │
                    │                    guardrails: citations required, Content Safety      │
                    └────────────────────────────────────────────────────────────────────────┘
```

## Local quickstart

```bash
uv sync --extra dev
uv run python scripts/generate_sample_corpus.py
make train-classifier              # MLflow local, no Azure
make test                          # unit + integration
make evals                         # deterministic + judge (fake LLM)
make prompt-regression             # nightly judge regression (local, no API cost)
make score-local                   # FastAPI mirror; POST to /classify or /reason
make frontend                      # testing UI at http://localhost:3000
```

Local requests reach a scripted fake LLM — real Azure OpenAI is only used when `.env` has real credentials and `DOCUSENSE_ENV != local`.

## Azure quickstart

```bash
# 1. Pre-requisites: only subscription + tenant are needed before Terraform
cp .env.example .env
# Fill in AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_TENANT_ID

# 2. Provision infrastructure — this creates OpenAI, AI Search, Content Safety, etc.
#   terraform init \
#     -backend-config="resource_group_name=rg-tfstate" \
#     -backend-config="storage_account_name=sttfstatedocusense" \
#     -backend-config="container_name=tfstate" \
#     -backend-config="key=docusense-${env}.tfstate"

make tf-init && make tf-plan && make tf-apply

# 3. After apply: grab endpoints + keys from Terraform outputs and update .env
terraform -chdir=infra output                          # endpoints
az cognitiveservices account keys list \
  --name <openai-resource> --resource-group <rg> --query key1 -o tsv   # AZURE_OPENAI_KEY
az search admin-key show \
  --service-name <search-resource> --resource-group <rg> --query primaryKey -o tsv  # AZURE_SEARCH_KEY
az cognitiveservices account keys list \
  --name <cs-resource> --resource-group <rg> --query key1 -o tsv       # CONTENT_SAFETY_KEY

# App Insights (for prompt regression trace sampling)
az extension add --name application-insights             # one-time: install CLI extension
terraform -chdir=infra output -raw application_insights_connection_string  # APPINSIGHTS_CONNECTION_STRING
az resource show --resource-group <rg> \
  --resource-type "Microsoft.Insights/components" \
  --name <appi-resource> --query properties.AppId -o tsv               # APPINSIGHTS_APP_ID
az monitor app-insights api-key create \
  --app <appi-resource> -g <rg> \
  --api-key prompt-regression-reader \
  --read-properties ReadTelemetry --query apiKey -o tsv                # APPINSIGHTS_API_KEY

# # Example
# terraform -chdir=infra output                          # endpoints
# az cognitiveservices account keys list \
#   --name "aoai-docusense-dev-0vf41" --resource-group "rg-docusense-dev" --query key1 -o tsv   # AZURE_OPENAI_KEY
# az search admin-key show \
#   --service-name "srch-docusense-dev-0vf41" --resource-group "rg-docusense-dev" --query primaryKey -o tsv  # AZURE_SEARCH_KEY
# az cognitiveservices account keys list \
#   --name "cs-docusense-dev-0vf41" --resource-group "rg-docusense-dev" --query key1 -o tsv       # CONTENT_SAFETY_KEY
# az resource show --resource-group "rg-docusense-dev" \
#   --resource-type "Microsoft.Insights/components" \
#   --name "appi-docusense-dev" --query properties.AppId -o tsv       # APPINSIGHTS_APP_ID
# az monitor app-insights api-key create \
#   --app "appi-docusense-dev" -g "rg-docusense-dev" \
#   --api-key prompt-regression-reader \
#   --read-properties ReadTelemetry --query apiKey -o tsv             # APPINSIGHTS_API_KEY


# 4. Create AML environments, build index, upload training data, train, deploy
az ml environment create -f aml/environments/training_env.yml
az ml environment create -f aml/environments/scoring_env.yml

uv run python scripts/generate_sample_corpus.py
uv run python scripts/build_index.py --push-to-search   # push chunks + embeddings to AI Search

# 5. Register the 'raw' datastore in AML (required — not created automatically)
SA_KEY=$(az storage account keys list --account-name <storage-account> -g <rg> --query "[0].value" -o tsv)
az rest --method PUT \
  --url "https://management.azure.com/subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.MachineLearningServices/workspaces/<ws>/datastores/raw?api-version=2024-04-01" \
  --body "{\"properties\":{\"datastoreType\":\"AzureBlob\",\"accountName\":\"<storage-account>\",\"containerName\":\"raw\",\"credentials\":{\"credentialsType\":\"AccountKey\",\"secrets\":{\"key\":\"$SA_KEY\",\"secretsType\":\"AccountKey\"}}}}"

# 6. Upload training data to blob storage (required before AML pipeline)
az storage blob upload-batch --account-name <storage-account> \
  --destination raw --destination-path contracts --source data/sample_contracts/ --overwrite
az storage blob upload --account-name <storage-account> --container-name raw \
  --name labels/golden_intents.jsonl --file data/golden_intents.jsonl --overwrite

# 7. Submit training pipeline, register model, and deploy
uv pip install azure-ai-ml azure-identity       # one-time: install Azure ML SDK
uv run python scripts/submit_training_job.py    # train the fast head in AML

# 8. Register the model artifact (required before endpoint deployment)
az ml model create --name docusense-fast-classifier --version 1 \
  --path outputs/classifier/ --type custom_model

# 9. Deploy the online endpoint (injects secrets from .env automatically)
uv run python scripts/deploy_endpoint.py        # deploy + route traffic
uv run python scripts/smoke_endpoint.py         # smoke test
```

> **Note:** Azure OpenAI, AI Search, and Content Safety endpoints/keys are *outputs* of `terraform apply` — you do not need them beforehand.
> 
> **Security:** `deploy_endpoint.py` reads credentials from `.env` and injects them as environment variables at deploy time. No secrets are stored in the deployment YAML.

## AML pipeline architecture

The training pipeline (`aml/pipelines/training_pipeline.yml`) runs two steps on Azure ML compute:

```
┌─────────┐     ┌──────────┐
│  train  │────▶│ evaluate │
└─────────┘     └──────────┘
 corpus + labels   classifier artifact
 → model artifact  + labels → gate decision
```

**Key design decisions:**
- **Data must exist in blob storage before submission** — the pipeline inputs reference `azureml://datastores/raw/paths/contracts/` and `azureml://datastores/raw/paths/labels/golden_intents.jsonl`. Upload first!
- **Environments must be registered** — if you recreate the AML workspace (e.g. after purge), re-register with `az ml environment create`.
- **PYTHONPATH** — components reference `code: ../../` (the repo root) so `docusense` package is importable.
- **MLflow in AML** — when `tracking_uri == "azureml://"`, AML auto-configures via env vars; no explicit `mlflow.set_tracking_uri()` needed.
- **Azure SDKs required locally** — `submit_training_job.py` needs `azure-ai-ml` and `azure-identity`. Install with `uv pip install azure-ai-ml azure-identity`.

## Repository layout

```
docusense-ai-azure/
├── infra/                    Terraform: AML, OpenAI, AI Search, Content Safety, KV, storage, monitoring
├── frontend/                 Single-page testing UI (HTML + vanilla JS)
├── src/docusense/
│   ├── schemas/              Contracts at every API + LLM boundary
│   ├── classifier/           Embedding + LightGBM head + MLflow training
│   ├── retrieval/            Chunker + in-memory + AI-Search retrievers + indexer
│   ├── llm/                  Client, prompts (md), tools, structured-output helpers, pipeline
│   ├── guardrails/           PII scrub, citations, output safety
│   ├── serving/              AML score.py + local FastAPI mirror
│   ├── evals/                Deterministic + LLM-as-judge + report generator
│   ├── telemetry/            OpenTelemetry spans + cost estimator
│   └── pipelines/            AML component entrypoints (index, evaluate)
├── aml/                      YAML for environments, components, pipelines, endpoints
├── scripts/                  CLIs for corpus gen, index build, train, evals, deploy, smoke
├── tests/                    unit / evals / integration
├── data/                     sample corpus + golden intents + golden extractions + judge prompts
└── .github/workflows/        ci · evals · terraform · index_and_train · deploy_endpoint · prompt_regression
```

## GitHub workflows

| Workflow | Trigger | Purpose |
| --- | --- | --- |
| `ci.yml` | PR + push | Ruff + pytest + integration tests, all offline |
| `evals.yml` | PR touching prompts/LLM/classifier | Deterministic + judge regression (fake LLM); real-LLM job gated on prompt changes |
| `terraform.yml` | PR (plan) / push (apply) / dispatch | Provision the Azure footprint |
| `index_and_train.yml` | Manual dispatch | Submit the AML indexing + training pipeline |
| `deploy_endpoint.yml` | Tag `v*` / dispatch | Deploy new deployment + smoke test |
| `prompt_regression.yml` | Nightly cron / dispatch | Sample traces from App Insights, run LLM-as-judge, flag regressions |

### Running `index_and_train` manually via CLI

The GitHub workflow is just a thin wrapper around `scripts/submit_training_job.py`. You can run the same pipelines locally without GitHub Actions:

```bash
# Pre-requisites
az login
uv pip install azure-ai-ml azure-identity   # one-time

# 1. Run the INDEXING pipeline (chunks corpus → pushes to AI Search)
uv run python scripts/submit_training_job.py \
  --pipeline aml/pipelines/indexing_pipeline.yml --wait

# 2. Run the TRAINING pipeline (trains classifier + evaluates)
uv run python scripts/submit_training_job.py \
  --pipeline aml/pipelines/training_pipeline.yml --wait

# 3. (Optional) Run without blocking the terminal
uv run python scripts/submit_training_job.py \
  --pipeline aml/pipelines/training_pipeline.yml --no-wait
```

**Before submitting**, ensure:
- Training data is uploaded to blob storage (see step 6 in Azure quickstart)
- The `raw` datastore is registered in the AML workspace (see step 5)
- AML environments are registered (`az ml environment create -f aml/environments/training_env.yml`)
- A `cpu-cluster` compute exists in the workspace (auto-created by AML on first run, or create manually):
  ```bash
  az ml compute create --name cpu-cluster --type amlcompute \
    --size Standard_DS3_v2 --min-instances 0 --max-instances 2
  ```

You can also submit either pipeline directly with the `az ml` CLI (no Python script needed):

```bash
# Submit the training pipeline directly via az ml
az ml job create --file aml/pipelines/training_pipeline.yml \
  -g rg-docusense-dev -w aml-docusense-dev --stream

# Submit the indexing pipeline directly via az ml
az ml job create --file aml/pipelines/indexing_pipeline.yml \
  -g rg-docusense-dev -w aml-docusense-dev --stream
```

> **Tip:** Drop `--stream` to submit without waiting. Use `az ml job show --name <job-name>` to check status later.

## Prompt regression (LLM-as-judge)

A nightly pipeline that samples recent `/reason` responses and scores them against a rubric-driven judge. If quality drops below a threshold, it flags a regression.

```bash
# ─── Local (no API cost, fake judge + golden data) ────────────────────
make prompt-regression

# ─── Live (real Azure OpenAI judge, needs .env credentials) ───────────
make prompt-regression-live

# ─── Individual steps ─────────────────────────────────────────────────
# 1. Sample traces (from App Insights, or --local for golden data fallback)
uv run python scripts/sample_traces.py --output reports/traces.jsonl --local

# 2. Run the judge on sampled traces
uv run python scripts/run_prompt_regression.py \
  --traces-file reports/traces.jsonl \
  --baseline-file data/judge_baseline.json \
  --skip-sample

# 3. Run with custom regression threshold (default: 0.5)
uv run python scripts/run_prompt_regression.py --local --threshold 0.3
```

**How it works:**

1. `scripts/sample_traces.py` pulls the last N `/reason` responses from App Insights (or builds synthetic responses from `data/judge_prompts.jsonl` when `--local`)
2. `scripts/run_prompt_regression.py` feeds each trace through the `Judge` class, which scores on 4 dimensions (1–5 scale): intent correctness, citation grounding, field extraction quality, conciseness & style
3. The mean score is compared against `data/judge_baseline.json` — if it drops by more than `--threshold`, the run fails (exit code 1)
4. In CI (`prompt_regression.yml`), a failure auto-creates a GitHub issue labelled `regression`

**Updating the baseline:** After improving prompts, re-run locally and delete the baseline to regenerate it:

```bash
rm data/judge_baseline.json
make prompt-regression-live     # saves current scores as the new baseline
```

## Frontend testing UI

A single-page app (`frontend/index.html`) for interacting with either the **local FastAPI server** or the **Azure ML online endpoint**.

```bash
# Terminal 1 — start the local API (required for both Local and Azure ML modes)
make score-local          # http://localhost:8000

# Terminal 2 — serve the frontend
make frontend             # http://localhost:3000
```

> **Important:** The local server must be running even when targeting Azure ML — it acts as a CORS proxy (`/proxy/score`) because browsers block direct cross-origin requests to AML endpoints.

**Features:**
- Toggle between Local and Azure ML targets
- Azure ML mode prompts for scoring URI + Bearer key
- Pre-loaded sample documents (matching the indexed corpus)
- Classify (fast head) and Reason (LLM pipeline) buttons
- Syntax-highlighted JSON response with latency + status

**For Azure ML endpoint testing:**

```bash
# Get the scoring URI
az ml online-endpoint show --name docusense-online \
  -g rg-docusense-dev -w aml-docusense-dev --query scoring_uri -o tsv

# Get the API key
az ml online-endpoint get-credentials --name docusense-online \
  -g rg-docusense-dev -w aml-docusense-dev --query primaryKey -o tsv
```

Paste both into the frontend UI, switch to "Azure ML Endpoint" mode, and send a request.

## Troubleshooting

### Terraform: model deprecated or not supported

Azure OpenAI models are retired frequently. If `terraform apply` fails with:

```
ServiceModelDeprecated: The model 'Format:OpenAI,Name:gpt-4o,Version:...' has been deprecated
ServiceModelDeprecating: The model '...' is in deprecating state and cannot be used for new deployments
DeploymentModelNotSupported: The model '...' is not supported
```

**Fix:** List the models available in your OpenAI account and pick a current version:

```bash
az cognitiveservices account list-models \
  --name <openai-account-name> \
  --resource-group <rg> \
  -o json | python3 -c "
import json, sys
for m in json.load(sys.stdin):
    if 'gpt' in m.get('name',''):
        print(m['name'], m.get('version','?'), [s['name'] for s in m.get('skus',[])])
"
```

Then update `infra/modules/openai/main.tf` → `model { name = "..." version = "..." }`.

### Terraform: storage account HNS vs AML / versioning

Azure ML workspaces do **not** support Data Lake Gen2 storage (HNS). If you see:

```
Cannot use storage with HNS enabled
```

**Fix:** Set `is_hns_enabled = false` in `infra/modules/storage/main.tf`.

> Note: `versioning_enabled = true` is only compatible with `is_hns_enabled = false`.

### Terraform: resource group still contains resources on destroy

If `terraform destroy` fails with:

```
the Resource Group still contains Resources (e.g. Application Insights Smart Detection)
```

**Fix:** Add to the provider `features` block in `infra/providers.tf`:

```hcl
resource_group {
  prevent_deletion_if_contains_resources = false
}
```

### Terraform: stale state after partial destroy

If apply fails with `ResourceNotFound` for a resource that was deleted outside of Terraform:

```bash
cd infra
terraform state rm <resource_address>   # e.g. module.storage.azurerm_storage_account.this
```

Then re-run `make tf-apply`.

### Terraform: resource already exists (needs import)

If a previous apply partially succeeded and state is out of sync, you'll see:

```
A resource with the ID "..." already exists - to be managed via Terraform this resource needs to be imported into the State.
```

**Fix:** Import each existing resource back into state:

```bash
cd infra
terraform import -var-file=envs/dev.tfvars '<terraform_address>' '<azure_resource_id>'

# Examples:
terraform import -var-file=envs/dev.tfvars \
  'module.openai.azurerm_cognitive_account.openai' \
  '/subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.CognitiveServices/accounts/<name>'

terraform import -var-file=envs/dev.tfvars \
  'module.ai_search.azurerm_search_service.this' \
  '/subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.Search/searchServices/<name>'

terraform import -var-file=envs/dev.tfvars \
  'module.content_safety.azurerm_cognitive_account.content_safety' \
  '/subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.CognitiveServices/accounts/<name>'

terraform import -var-file=envs/dev.tfvars \
  'module.monitoring.azurerm_log_analytics_workspace.this' \
  '/subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.OperationalInsights/workspaces/<name>'

terraform import -var-file=envs/dev.tfvars \
  'module.storage.azurerm_storage_account.this' \
  '/subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.Storage/storageAccounts/<name>'
```

> Tip: Use `az resource list --resource-group <rg> -o table` to see what actually exists in Azure.

### Terraform: soft-deleted AML workspace blocking creation

If you see:

```
Soft-deleted workspace exists. Please purge or recover it.
```

**Fix:** Purge the soft-deleted workspace from the Azure Portal:
1. Go to **Azure Machine Learning** → **Recently deleted workspaces**
2. Select the workspace → **Purge**

Or via CLI (if supported in your CLI version):
```bash
az ml workspace delete --name <workspace> --resource-group <rg> --permanently-delete --yes
```

### Terraform: Search Service name conflict (409 ServiceDeleting)

If a previous destroy is still running in the background:

```
Cannot provision service named '...' because a background operation is still in progress
```

**Fix:** Wait a few minutes and retry. Azure Search deletions can take 5–10 minutes to fully release the name. If the service eventually reappears, import it:

```bash
terraform import -var-file=envs/dev.tfvars \
  'module.ai_search.azurerm_search_service.this' \
  '/subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.Search/searchServices/<name>'
```

### Azure CLI token expired

If you see `User '...' does not exist in MSAL token cache`:

```bash
az login
```

### Azure CLI: wrong resource group / workspace defaults

If `az ml` commands fail with `ResourceGroupNotFound` for a different project's resource group (e.g. `rg-signalflow-dev`), your Azure CLI defaults are stale:

```bash
# Check current defaults
cat ~/.azure/config

# Fix: set correct defaults
az configure --defaults group=rg-docusense-dev workspace=aml-docusense-dev
```

### Cognitive deployment: `sku` vs `scale` block

With `azurerm` provider v3.x, `azurerm_cognitive_deployment` uses a `scale` block (not `sku`):

```hcl
# ✗ Wrong
sku {
  name     = "Standard"
  capacity = 30
}

# ✓ Correct
scale {
  type     = "Standard"
  capacity = 30
}
```

### AI Search: invalid document key characters

Azure AI Search document keys only allow letters, digits, `_`, `-`, or `=`. If you see:

```
InvalidDocumentKey: Invalid document key: 'msa-01#0000'. Keys can only contain letters, digits, underscore (_), dash (-), or equal sign (=).
```

**Fix:** The chunk ID separator in `src/docusense/retrieval/chunker.py` must not use `#`. Use `_chunk_` instead:

```python
# ✗ Wrong
chunk_id=f"{doc.doc_id}#{i:04d}"

# ✓ Correct
chunk_id=f"{doc.doc_id}_chunk_{i:04d}"
```

### AI Search: service in failed/error state (timeout on all requests)

If `build_index.py` times out on the search service but the endpoint is correct:

```bash
# Check provisioning state
az search service show --name <search-service> --resource-group <rg> \
  --query "{status:status, provisioningState:provisioningState}" -o json
```

If `provisioningState: Failed`, the service was created during a name conflict. **Fix:**

```bash
az search service delete --name <search-service> --resource-group <rg> --yes
cd infra && terraform state rm 'module.ai_search.azurerm_search_service.this'
sleep 60  # wait for Azure to release the name
make tf-apply
```

### Cognitive account: soft-deleted blocking recreation (409)

If you see `FlagMustBeSetForRestore` or a 409 on the OpenAI/Content Safety account:

```bash
az cognitiveservices account purge \
  --name <account-name> --resource-group <rg> --location swedencentral
```

Then re-run `make tf-apply`.

### AML pipeline: `ModuleNotFoundError: No module named 'azure.ai'`

The submission script requires Azure ML SDK packages not included in the base dev deps:

```bash
uv pip install azure-ai-ml azure-identity
```

### AML pipeline: `ResourceNotFoundError` on environment resolution

If `submit_training_job.py` fails resolving `azureml:docusense-training@latest`:

```
ResourceNotFoundError: (UserError) System.Net.Http.HttpConnectionResponseContent
```

**Cause:** The AML workspace was recreated (purged + re-provisioned) but environments weren't re-registered.

**Fix:**
```bash
az ml environment create -f aml/environments/training_env.yml
az ml environment create -f aml/environments/scoring_env.yml
```

### AML pipeline: `Failed nodes: /train`

If the pipeline submits successfully but the `/train` step fails:

**Most common cause:** Training data not uploaded to blob storage. The pipeline references:
- `azureml://datastores/raw/paths/contracts/` (sample contract files)
- `azureml://datastores/raw/paths/labels/golden_intents.jsonl` (intent labels)

**Fix:** Upload the data before submission:
```bash
az storage blob upload-batch --account-name <storage-account> \
  --destination raw --destination-path contracts --source data/sample_contracts/ --overwrite
az storage blob upload --account-name <storage-account> --container-name raw \
  --name labels/golden_intents.jsonl --file data/golden_intents.jsonl --overwrite
```

**Other causes:**
- Workspace managed identity lacks blob access → grant `Storage Blob Data Contributor`:
  ```bash
  az role assignment create \
    --assignee-object-id <workspace_principal_id> \
    --assignee-principal-type ServicePrincipal \
    --role "Storage Blob Data Contributor" \
    --scope <storage_account_resource_id>
  ```
- Compute cluster quota exceeded → use a smaller SKU or reduce `max_node_count`

### AML pipeline: `Could not find datastore: raw`

The `raw` datastore isn't registered automatically. Normally it exists if the storage account was the workspace's default, but if not:

```bash
SA_KEY=$(az storage account keys list --account-name <storage-account> -g <rg> --query "[0].value" -o tsv)
az rest --method PUT \
  --url "https://management.azure.com/subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.MachineLearningServices/workspaces/<ws>/datastores/raw?api-version=2024-04-01" \
  --body "{
    \"properties\": {
      \"datastoreType\": \"AzureBlob\",
      \"accountName\": \"<storage-account>\",
      \"containerName\": \"raw\",
      \"credentials\": { \"credentialsType\": \"AccountKey\", \"secrets\": { \"key\": \"$SA_KEY\", \"secretsType\": \"AccountKey\" } }
    }
  }"
```

### AML pipeline: `Got unexpected extra arguments` (flag-stripping)

AML strips `--flag` names from the command string when they collide with declared input/output names (e.g. `--corpus`, `--labels`, `--output`). The component command ends up passing paths as positional args, but the script expects named options → error.

**Fix:** Convert CLI arguments from `typer.Option` to `typer.Argument` (positional) in the Python entrypoints, and remove all `--` prefixes from the component YAML command:

```python
# ✗ Wrong — AML strips these
corpus: Path = typer.Option(..., exists=True, file_okay=False)

# ✓ Correct — positional args survive AML
corpus: Path = typer.Argument(...)
```

> **Note:** Do not pass `exists=True` or `file_okay=False` to `typer.Argument()` — older Click/Typer versions in the AML curated environment throw `TypeError: TyperArgument.make_metavar() takes 1 positional argument but 2 were given`. Validate paths manually in your code instead.

```yaml
# ✗ Wrong
command: >-
  python -m docusense.classifier.train
  --corpus ${{inputs.corpus}} --labels ${{inputs.labels}}

# ✓ Correct
command: >-
  python -m docusense.classifier.train
  ${{inputs.corpus}} ${{inputs.labels}} ${{outputs.artifact}} azureml://
```

### AML pipeline: `TypeError: Secondary flag is not valid for non-boolean flag`

AML substitutes `${{inputs.push_to_search}}` → `True` (a string value). When Typer sees a `bool` parameter, it expects a flag (`--push-to-search / --no-push-to-search`) and can't accept a string value after it.

**Fix:** Accept the parameter as `str` and parse it manually:

```python
# ✗ Wrong — Typer treats bool as a flag, AML passes "True" as a value
push_to_search: bool = typer.Argument(False)

# ✓ Correct — accept string, parse manually
push_to_search: str = typer.Argument("false")
# then in the body:
_push = push_to_search.lower() in ("true", "1", "yes")
```

### AML pipeline: `BadParameter: Azure Search endpoint/key not configured`

The pipeline container doesn't have a `.env` file, so `get_settings()` returns all `None` for Azure credentials.

**Fix:** `submit_training_job.py` now injects credentials from your local `.env` as environment variables into each pipeline step before submission. This is automatic — just ensure your `.env` has the correct values and resubmit.

If you're submitting via `az ml job create` directly (without the Python script), pass env vars in the YAML:

```yaml
jobs:
  index:
    environment_variables:
      AZURE_SEARCH_ENDPOINT: "https://..."
      AZURE_SEARCH_KEY: "..."
```

### AML pipeline: `MlflowException: Cannot start run with ID ... experiment ID does not match`

AML pre-creates an MLflow run (via `MLFLOW_RUN_ID` env var) that belongs to the pipeline's auto-generated experiment. If your code calls `mlflow.set_experiment("custom-name")`, it creates a *different* experiment → when `start_run()` picks up the pre-existing run ID, the experiment IDs clash.

**Fix:** Detect AML and skip `set_tracking_uri` / `set_experiment`:

```python
import os
_in_aml = bool(os.environ.get("MLFLOW_RUN_ID") or os.environ.get("AZUREML_RUN_ID"))

if not _in_aml:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
```

### AML pipeline: `TypeError: azureml_artifacts_builder() got an unexpected keyword argument 'tracking_uri'`

The AML curated environment has a version mismatch between `mlflow` (2.14.1) and `mlflow-skinny` (3.14.0). The newer skinny package passes `tracking_uri` to the artifact builder, but the older `azureml-mlflow` plugin doesn't accept it.

**Fix:** Skip `mlflow.log_artifact()` when running in AML — the pipeline already captures outputs via the declared `${{outputs.artifact}}` mount:

```python
if not _in_aml:
    mlflow.log_artifact(str(model_path))
    mlflow.log_artifact(str(report_path))
```

Metrics and params are still logged successfully (they use the tracking API, not the artifact API).

### Endpoint deployment: `ModelNotFound: Model container with name: docusense-fast-classifier not found`

The deployment YAML references `azureml:docusense-fast-classifier@latest` but the model isn't registered.

**Fix:** Register the model from local outputs (after a successful local or pipeline training run):

```bash
az ml model create --name docusense-fast-classifier --version 1 \
  --path outputs/classifier/ --type custom_model
```

### Endpoint deployment: `ImageBuildFailure` — base image does not exist

If the scoring environment uses a non-existent MCR image (e.g. `mcr.microsoft.com/azureml/minimal-ubuntu22.04-py311-cpu-inference`):

**Fix:** Use the standard AML inference base image in `aml/environments/scoring_env.yml`:

```yaml
image: mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04
```

### Endpoint deployment: `ImageBuildFailure` — pip dependency conflict (pydantic)

`azureml-inference-server-http==1.0.0` requires `pydantic<1.11` which conflicts with our Pydantic v2 deps.

**Fix:** Use `azureml-inference-server-http>=1.3.0` (supports Pydantic v2) and relax pydantic pin to `>=2.7,<3`.

### Endpoint deployment: `OutOfQuota` — not enough CPU quota

If you see `Not enough subscription CPU quota. The amount of CPU quota requested is 8`:

**Fix:** Downgrade the instance type in `aml/endpoints/online_deployment.yml`:

```yaml
# Standard_DS3_v2 = 4 vCPUs (requested 8 with overhead)
# Standard_DS2_v2 = 2 vCPUs — fits in most dev subscriptions
instance_type: Standard_DS2_v2
```

Or request a quota increase via the Azure Portal → **Quotas**.

### Endpoint deployment: `FileNotFoundError: artifact.joblib` — model path mismatch

If the model was registered from a directory (e.g. `outputs/classifier/`), the artifact lives at `<model_dir>/classifier/artifact.joblib` rather than `<model_dir>/artifact.joblib`.

**Fix:** The score script should check both paths:

```python
artifact_path = Path(model_dir) / "artifact.joblib"
if not artifact_path.exists():
    artifact_path = Path(model_dir) / "classifier" / "artifact.joblib"
```

### Endpoint deployment: `Specified deployment [blue] failed during initial provisioning and is in an unrecoverable state`

A previously failed deployment cannot be updated in-place.

**Fix:** Delete and recreate:

```bash
az ml online-deployment delete --name blue --endpoint-name docusense-online \
  -g rg-docusense-dev -w aml-docusense-dev --yes
uv run python scripts/deploy_endpoint.py
```

### Endpoint: 404 — `No valid deployments to route to`

The endpoint exists but all deployments have 0% traffic.

**Cause:** After redeployment, traffic weight wasn't set.

**Fix:** The updated `deploy_endpoint.py` now sets `traffic: blue=100` automatically. If you hit this on an existing deployment:

```bash
az ml online-endpoint update --name docusense-online \
  -g rg-docusense-dev -w aml-docusense-dev --traffic "blue=100"
```

### Endpoint: 500 — `Invalid URL ... No scheme supplied` (missing env vars)

The `/reason` route fails because `AZURE_SEARCH_ENDPOINT` (or `AZURE_OPENAI_ENDPOINT`) is empty in the deployed container.

**Cause:** The deployment was created without the Azure service credentials as environment variables.

**Fix:** `deploy_endpoint.py` now injects all required secrets from `.env` at deploy time. Redeploy:

```bash
uv run python scripts/deploy_endpoint.py
```

The script reads your local `.env` and sets `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`, `AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_KEY`, `CONTENT_SAFETY_ENDPOINT`, `CONTENT_SAFETY_KEY`, etc. as deployment environment variables. No secrets are committed to Git.

### Endpoint: 500 — `semanticConfiguration` required (semantic ranker not configured)

The `/reason` route fails with:

```
This index must have valid semantic configurations defined before using the 'semanticConfiguration' query parameter.
```

**Cause:** The search index was created without a semantic configuration, but the retriever was requesting `query_type="semantic"`.

**Fix:** The retriever now defaults to `use_semantic_ranker=False` — pure vector + keyword hybrid search works without a paid semantic ranker. After updating `src/docusense/retrieval/search.py`, redeploy:

```bash
az ml online-deployment delete --name blue --endpoint-name docusense-online \
  -g rg-docusense-dev -w aml-docusense-dev --yes
uv run python scripts/deploy_endpoint.py
```

To enable semantic ranker in production, add a semantic configuration to the index via `scripts/build_index.py` and pass `use_semantic_ranker=True` when constructing the retriever. This requires the search service to be on the Standard tier (not Free/Basic).

### Endpoint: deleting deployment with non-zero traffic

If you see `Can't delete deployment [blue] with non-zero traffic weight`:

```bash
# First zero the traffic, then delete
az ml online-endpoint update --name docusense-online \
  -g rg-docusense-dev -w aml-docusense-dev --traffic "blue=0"
az ml online-deployment delete --name blue --endpoint-name docusense-online \
  -g rg-docusense-dev -w aml-docusense-dev --yes
```

### Local server: `ModuleNotFoundError: No module named 'openai'`

The `/reason` route requires the `openai` SDK. If you installed without the `azure` extra:

```bash
uv pip install openai
# or install all Azure extras:
uv sync --extra dev --extra azure
```

### Local server: `ModuleNotFoundError: No module named 'azure.ai.contentsafety'`

The output safety guardrail lazily imports `azure-ai-contentsafety` when checking LLM output:

```bash
uv pip install azure-ai-contentsafety
```

### Local server: 401 — `Access denied due to invalid subscription key`

The `.env` file has a stale or rotated Azure OpenAI API key.

**Fix:** Re-fetch the current key and update `.env`:

```bash
az cognitiveservices account keys list \
  --name aoai-docusense-dev-0vf41 -g rg-docusense-dev --query key1 -o tsv
```

Also ensure `AZURE_OPENAI_CHAT_DEPLOYMENT` matches the deployed model name (e.g. `gpt-5.1` not `gpt-4o` if you redeployed).

### Local server: 422 — `quoted snippet not found in chunk`

The citation guardrail rejects the response because the LLM quoted text from the user's prompt rather than the actual indexed chunk.

**Cause:** The document you submitted doesn't match the indexed corpus, so the retrieved chunk text differs from what the LLM cited.

**Fix:** Use documents that are actually in the search index (i.e. from `data/sample_contracts/`). The frontend's sample documents are pre-loaded from the corpus for this reason. If you submit freeform text, the citation guardrail may reject it if the LLM hallucinates quotes.

## Useful commands

```bash
# ─── Local development ───────────────────────────────────────────────
uv sync --extra dev                         # install all deps
make train-classifier                       # train LightGBM head locally (MLflow)
make test                                   # unit + integration tests
make evals                                  # deterministic + LLM-as-judge (fake LLM)
make prompt-regression                      # judge regression (local, fake LLM)
make prompt-regression-live                 # judge regression (real Azure OpenAI)
make score-local                            # start FastAPI server locally
make frontend                               # testing UI at http://localhost:3000

# ─── Terraform ───────────────────────────────────────────────────────
make tf-init                                # terraform init with backend config
make tf-plan                                # plan changes
make tf-apply                               # apply changes
terraform -chdir=infra destroy -var-file=envs/dev.tfvars   # tear down

terraform -chdir=infra output               # show all outputs (endpoints, names)
terraform -chdir=infra state list           # list managed resources
terraform -chdir=infra state rm <addr>      # remove stale resource from state
terraform -chdir=infra import -var-file=envs/dev.tfvars '<addr>' '<id>'  # import existing resource

# ─── Azure CLI: authentication ───────────────────────────────────────
az login                                    # re-authenticate when token expires
az account show                             # verify subscription + tenant

# ─── Azure CLI: retrieve keys after terraform apply ──────────────────
az cognitiveservices account keys list \
  --name <openai-resource> --resource-group <rg> --query key1 -o tsv

az search admin-key show \
  --service-name <search-resource> --resource-group <rg> --query primaryKey -o tsv

az cognitiveservices account keys list \
  --name <content-safety-resource> --resource-group <rg> --query key1 -o tsv

# ─── Azure CLI: App Insights (for prompt regression) ─────────────────
az extension add --name application-insights                # one-time: install extension
terraform -chdir=infra output -raw application_insights_connection_string
az resource show --resource-group <rg> \
  --resource-type "Microsoft.Insights/components" \
  --name <appi-resource> --query properties.AppId -o tsv    # APPINSIGHTS_APP_ID
az monitor app-insights api-key create \
  --app <appi-resource> -g <rg> \
  --api-key prompt-regression-reader \
  --read-properties ReadTelemetry --query apiKey -o tsv     # APPINSIGHTS_API_KEY (create once)

# ─── Azure CLI: inspect resources ────────────────────────────────────
az resource list --resource-group <rg> -o table             # what exists in Azure
az cognitiveservices account list-models \
  --name <openai-account> --resource-group <rg> -o table    # available models
az storage account delete --name <acct> --resource-group <rg> --yes  # force-delete storage

# ─── Azure ML ────────────────────────────────────────────────────────
az ml environment create -f aml/environments/training_env.yml
az ml environment create -f aml/environments/scoring_env.yml
az ml online-endpoint show --name docusense-online --query scoring_uri -o tsv
az ml workspace delete --name <ws> --resource-group <rg> --permanently-delete --yes  # purge

# ─── Azure ML: upload training data (required before pipeline) ───────
az storage blob upload-batch --account-name <storage-account> \
  --destination raw --destination-path contracts --source data/sample_contracts/ --overwrite
az storage blob upload --account-name <storage-account> --container-name raw \
  --name labels/golden_intents.jsonl --file data/golden_intents.jsonl --overwrite

# ─── Azure ML: pipeline jobs ─────────────────────────────────────────
uv pip install azure-ai-ml azure-identity       # one-time: install Azure ML SDK
uv run python scripts/submit_training_job.py    # submit training pipeline
uv run python scripts/submit_training_job.py --no-wait  # submit without blocking

# ─── Azure ML: monitor pipeline runs ─────────────────────────────────
az ml job show --name <job-name> --query "{status:status,display_name:display_name}" -o json
az ml job list --parent-job-name <pipeline-job> \
  --query "[].{name:name,display_name:display_name,status:status}" -o table  # child steps
az ml job stream --name <job-name>              # stream logs live
az ml job list --query "[?status=='Completed'].{name:name, status:status}" -o table

# ─── Azure ML: deployment ────────────────────────────────────────────
az ml model create --name docusense-fast-classifier --version 1 \
  --path outputs/classifier/ --type custom_model    # register model
uv run python scripts/deploy_endpoint.py        # deploy the online endpoint
uv run python scripts/smoke_endpoint.py         # smoke test the endpoint
az ml online-endpoint show --name docusense-online --query scoring_uri -o tsv
az ml online-deployment show --name blue --endpoint-name docusense-online \
  --query provisioning_state -o tsv             # check deployment status
az ml online-deployment get-logs --name blue --endpoint-name docusense-online \
  --lines 100                                   # container logs
az ml online-deployment delete --name blue --endpoint-name docusense-online --yes  # delete failed deployment
```

## Known gaps

See `DEMO.md`.
