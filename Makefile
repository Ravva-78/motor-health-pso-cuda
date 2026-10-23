.PHONY: help install test lint format demo run-daemon run-api run-ui benchmark docker-build clean

PYTHON ?= python
PIP ?= pip
PYTEST ?= pytest

help:
	@echo "AeroForge Developer Command Center"
	@echo "=================================="
	@echo "make install       Install all production and dev dependencies"
	@echo "make test          Run full test suite with coverage reporting"
	@echo "make lint          Run flake8 and code quality audits"
	@echo "make format        Format codebase with black"
	@echo "make demo          Execute 1-command zero-setup quickstart demo"
	@echo "make run-daemon    Start the background edge daemon (ZMQ + Ingestion + Retrain)"
	@echo "make run-api       Start the FastAPI REST gateway on port 8000"
	@echo "make run-ui        Start the Streamlit dark glassmorphic dashboard"
	@echo "make benchmark     Run standalone or live inference latency benchmark"
	@echo "make docker-build  Build multi-container docker images"
	@echo "make clean         Remove cache directories, logs, and build artifacts"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install black flake8 pytest-cov

test:
	$(PYTEST) tests/ -v --cov=backend --cov-report=term-missing

lint:
	flake8 --count --select=E9,F63,F7,F82 --show-source --statistics backend/ scripts/ tests/

format:
	black --line-length 120 backend/ scripts/ tests/

demo:
	$(PYTHON) scripts/quickstart_demo.py

run-daemon:
	$(PYTHON) -m backend.backend_daemon

run-api:
	uvicorn backend.api_server:app --port 8000 --reload

run-ui:
	streamlit run app.py

benchmark:
	$(PYTHON) scripts/benchmark.py

docker-build:
	docker compose build

clean:
	rm -rf .pytest_cache .coverage coverage.xml htmlcov build dist *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
