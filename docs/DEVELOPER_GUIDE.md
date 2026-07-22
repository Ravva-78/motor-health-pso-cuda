# Developer Guide

## Setup
1. Clone repository.
2. Ensure Python 3.11+.
3. `pip install -r requirements.txt`

## Running Tests
Run the test suite using pytest:
```bash
pytest tests/ -v --cov=backend
```
*Note: Test coverage currently stands at ~60%. GPU paths are generally skipped in standard CI unless hardware is present.*

## Architecture Principles
- **Fail Gracefully**: If GPU is unavailable, modules (Inference, PSO) must fallback to CPU gracefully.
- **Zero Downtime**: The hot-swap mechanism in `inference_service.py` atomic pointer swap ensures inference continues during model retraining.