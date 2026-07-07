.PHONY: help install fmt lint test evals prompt-regression sample-corpus train-classifier score-local build-index clean tf-init tf-plan tf-apply frontend

help:
	@echo "DocuSense — make targets"
	@echo "  install              uv sync (dev + azure)"
	@echo "  fmt                  ruff format"
	@echo "  lint                 ruff check + format check"
	@echo "  test                 pytest (unit only)"
	@echo "  evals                pytest -m evals (deterministic evals with fake LLM)"
	@echo "  prompt-regression    run judge regression locally (fake LLM + golden data)"
	@echo "  prompt-regression-live  run judge regression with real Azure OpenAI"
	@echo "  sample-corpus        generate synthetic contract corpus + labels"
	@echo "  train-classifier     train fast intent head on cached embeddings"
	@echo "  score-local          start local FastAPI mirror of the AML endpoint"
	@echo "  frontend             serve the testing UI (open http://localhost:3000)"
	@echo "  build-index          upsert chunks into AI Search (needs Azure creds)"
	@echo "  tf-init/plan/apply   terraform lifecycle (dev)"
	@echo "  clean                purge caches and outputs"

install:
	uv sync --extra dev --extra azure

fmt:
	uv run ruff format .

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -m "not integration and not evals"

evals:
	uv run pytest -m evals

prompt-regression:
	uv run python scripts/run_prompt_regression.py --local

prompt-regression-live:
	uv run python scripts/run_prompt_regression.py

sample-corpus:
	uv run python scripts/generate_sample_corpus.py

train-classifier:
	uv run python -m docusense.classifier.train \
		data/sample_contracts \
		data/golden_intents.jsonl \
		outputs/classifier \
		file:./mlruns

score-local:
	uv run uvicorn docusense.serving.local_app:app --reload --port 8000

frontend:
	@echo "Opening frontend at http://localhost:3000"
	python3 -m http.server 3000 --directory frontend

build-index:
	uv run python scripts/build_index.py

clean:
	rm -rf .pytest_cache .ruff_cache mlruns mlartifacts outputs reports
	find . -type d -name __pycache__ -exec rm -rf {} +

tf-init:
	cd infra && terraform init

tf-plan:
	cd infra && terraform plan -var-file=envs/dev.tfvars

tf-apply:
	cd infra && terraform apply -var-file=envs/dev.tfvars
