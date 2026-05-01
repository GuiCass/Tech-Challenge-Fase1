.PHONY: install lint test run-api mlflow

install:
	pip install -e ".[dev,notebook]"

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

test:
	pytest tests/ -v --tb=short

run-api:
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

mlflow:
	python -m mlflow ui --backend-store-uri sqlite:///notebooks/mlflow.db --port 5000
