# Contributing to AeroForge

Thank you for your interest in contributing to **AeroForge**! We welcome contributions from engineers, researchers, and open-source practitioners.

This document outlines the development workflow, code standards, and contribution guidelines to ensure high architectural integrity across edge AI, IoT messaging, and CUDA optimization modules.

---

## 🧭 Code of Conduct

All contributors and maintainers are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md) (Contributor Covenant v2.1).

---

## 🛠 Local Development Setup

### 1. Prerequisites
- **Python**: 3.10 or 3.11
- **Git**: 2.30+
- **NVIDIA GPU Driver & CUDA Toolkit 12.x** (optional; CPU fallback is fully supported)
- **Make** (optional; for developer shortcuts)

### 2. Clone and Setup Environment
```bash
git clone https://github.com/Ravva-78/motor-health-pso-cuda.git
cd motor-health-pso-cuda

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Install all dependencies and development tools
pip install --upgrade pip
pip install -r requirements.txt
pip install black flake8 pytest-cov
```

### 3. Verify Local Installation
Run the 1-command zero-setup pipeline verification:
```bash
python scripts/quickstart_demo.py --dry-run
```
Run the full production test suite:
```bash
pytest -v
```

---

## 🌿 Git Branching Strategy

We follow a structured branch naming convention:
- `feature/<feature-name>`: New architectural features, endpoints, or ML enhancements.
- `fix/<bug-description>`: Bug fixes or resilience hardening.
- `perf/<optimization-area>`: Inference latency, memory buffer, or CUDA kernel optimizations.
- `docs/<documentation-topic>`: Architecture documents, benchmarks, or README updates.

### Workflow:
1. Fork the repository and create your feature branch:
   ```bash
   git checkout -b feature/tensorrt-engine-optimization
   ```
2. Make modular, focused commits following [Conventional Commits](#-commit-conventions).
3. Ensure all existing and new tests pass locally:
   ```bash
   pytest tests/ -v
   ```
4. Push to your fork and submit a Pull Request to `main`.

---

## 📝 Commit Conventions

We enforce [Conventional Commits](https://www.conventionalcommits.org/):
- `feat(service)`: Add new functionality (e.g., `feat(inference): support dynamic batching in ONNX provider`)
- `fix(daemon)`: Resolve a bug or error (e.g., `fix(ipc): handle socket timeouts during high retrain load`)
- `perf(cuda)`: Performance improvement (e.g., `perf(pso): optimize warp memory alignment in numba kernel`)
- `docs(readme)`: Documentation update (e.g., `docs(architecture): add sequence flow diagram`)
- `test(contracts)`: Add or modify test coverage (e.g., `test(contracts): add shape validation for 5-feature tensor`)
- `ci(actions)`: CI/CD workflow changes (e.g., `ci(github): add dependency cache and demo smoke test`)

---

## 📐 Engineering & Code Standards

- **PEP-8 Compliance**: Adhere to PEP-8 standards. Keep line length reasonable (target 100-120 chars).
- **Type Annotations**: Provide Python type hints on all public interfaces and class methods.
- **Hardware Fallback**: Never assume native NVIDIA drivers are installed on the host. Always wrap GPU-specific routines in graceful CPU fallbacks.
- **Microservice Isolation**: Keep ZeroMQ IPC, MQTT networking, and ONNX Runtime inference logically separated into their respective modules under `backend/`.
- **Thread Safety**: Any data structure accessed concurrently (e.g., `RingBuffer`, model reload pointers) must implement thread locks or atomic pointer swaps.

---

## 🧪 Testing Guidelines

Every new PR must include corresponding test coverage under `tests/`:
- **Shape & Tensor Contracts**: If modifying model schemas or inference pipelines, update `tests/test_shape_contracts.py`.
- **Sensor Preprocessing**: Validate sensor bounds and physical transformations under `tests/test_sensor_transforms.py`.
- **Schema Contracts**: Validate configuration parameters under `tests/test_config_schema.py`.
- **Fast Execution**: Tests must run quickly in isolated memory without external broker dependencies.

---

## 🚀 Developer Shortcuts (`Makefile`)

If you have `make` installed:
```bash
make install       # Install dependencies
make test          # Run full pytest suite with coverage
make lint          # Run flake8 syntax checks
make demo          # Execute 1-command quickstart demo
make run-daemon    # Launch edge backend daemon
make run-api       # Launch FastAPI REST gateway
make run-ui        # Launch Streamlit dashboard
make benchmark     # Run inference latency benchmark
```

---

## 📬 Submitting a Pull Request

1. Fill out the [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md).
2. Clearly explain the motivation, architectural trade-offs, and test evidence.
3. Link any related GitHub issues.
4. Verify all GitHub Actions CI checks pass green.

Thank you for helping make **AeroForge** the premier open-source predictive maintenance platform!
