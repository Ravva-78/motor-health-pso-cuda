# Engineering Program Initialization (EPI)
**Project Name:** AeroForge – Autonomous Predictive Edge AI Platform
**Document Owner:** Director of Engineering
**Reference Base:** Vision Freeze, PRD, TRD, SAD, DED, ARB
**Status:** PROGRAM LAUNCHED

---

## PART 1: Repository Bootstrap

### Folder Initialization
The repository will be strictly initialized following the frozen directory structure defined in the Detailed Engineering Design (DED) Part 1. No developer is authorized to alter root-level folder domains without ARB approval.
*   `src/backend/` and `src/frontend/` will serve as the monorepo application roots.
*   `tests/`, `config/`, and `deploy/` are strictly separated from source code.

### Git Strategy
*   We operate on **GitFlow (Modified)**.
*   `main` is production-ready. Direct commits to `main` are disabled at the repository level.
*   `develop` is the default branch for all feature integrations.
*   Feature branches must follow the convention: `feat/JIRA-ID-short-description` (e.g., `feat/AF-101-ring-buffer`).

### Documentation
*   The `docs/architecture/` folder will be populated immediately with the final PRD, TRD, SAD, DED, and ARB sign-offs.
*   An `ADR/` (Architecture Decision Records) folder will be initialized to track any micro-level library decisions made during sprints.

---

## PART 2: Development Environment

### Technology Stack & Tooling
*   **Python:** Strictly `v3.10.x`.
*   **CUDA:** Developers must use `CUDA Toolkit 12.x`.
*   **Docker:** Docker Desktop or Rancher Desktop is required for local Mosquitto broker emulation.
*   **IDE:** VS Code is the mandated editor.
    *   *Required Extensions:* Python, Pylance, Black Formatter, Flake8, Docker.
    *   A `.vscode/settings.json` file will be committed to enforce format-on-save for the entire team.

### Dependency Management
*   **Tool:** `Poetry` is the mandated package manager. `requirements.txt` is banned to prevent dependency resolution conflicts.
*   *Initialization Task:* `poetry init` and lock core dependencies (`paho-mqtt`, `numpy`, `tensorrt`, `streamlit`, `scipy`, `numba`).

---

## PART 3: CI/CD Initialization (GitHub Actions)

### Quality Gates
*   **Linting:** `flake8` for style, `black` for formatting.
*   **Type Checking:** `mypy` running in strict mode.
*   **Testing:** `pytest` executed on every pull request.
*   **Dependency Checks:** Dependabot will be enabled for weekly automated security patching.
*   **Security Scanning:** `Trivy` will scan the `Dockerfile.edge` on every push to `main`.

### Docker Builds
*   The CI pipeline will build the multi-stage ARM64 Docker image and push to the GitHub Container Registry (GHCR) only upon successful merge to `main`.

---

## PART 4: Configuration Management

### Environments
*   **Development:** `config.dev.yaml` — Connects to `localhost:1883`, uses mock GPU memory limits.
*   **Testing (CI):** Uses temporary, ephemeral configurations spun up via PyTest fixtures.
*   **Production:** `config.prod.yaml` — Bound to physical Jetson hardware limits.

### Secrets Management
*   Strict enforcement: `.env` files are in `.gitignore`.
*   Developer setup requires copying `.env.example` to `.env`.
*   Production secrets (MQTT passwords) will be injected via Kubernetes (`k3s`) Secrets.

### Feature Flags
*   A `feature_flags` block in the YAML config will control optional UI elements (e.g., `enable_3d_twin: false`) to manage GPU load during early sprints.

---

## PART 5: Engineering Standards

### Commit Conventions
*   Conventional Commits are strictly enforced:
    *   `feat: add shared memory ring buffer`
    *   `fix: resolve memory leak in MQTT client`
    *   `docs: update API specification`

### Code Reviews
*   Minimum 1 Approving Review from a Senior Engineer.
*   PRs must pass all CI checks before the Merge button is enabled.
*   No PR may exceed 400 lines of code (to ensure review quality).

### Definition of Done (DoD)
1.  Code compiles and runs locally via `docker-compose`.
2.  Unit tests written and passing (Line coverage > 80%).
3.  Docstrings (Google format) present on all public methods.
4.  No regression in latency budgets.
5.  CI pipeline passes green.

---

## PART 6: Sprint Planning

*(Mapped from DED Engineering Backlog)*

### Sprint 1: Foundation & Fabric (Weeks 1-2)
*   **Critical Path:** The Ring Buffer is a dependency for all subsequent modules.
*   **Tasks:** Dockerfile setup, Poetry init, SharedMemory Ring Buffer implementation, MQTT Subscriber integration, basic Streamlit skeleton.
*   **Dependencies:** None.

### Sprint 2: Fast Path Inference (Weeks 3-4)
*   **Critical Path:** Real-time evaluation of data.
*   **Tasks:** ONNX to TensorRT compiler script, Inference Engine polling loop, GPU TreeSHAP execution, MQTT Alarm Publisher.
*   **Dependencies:** Requires Sprint 1 Ring Buffer.

### Sprint 3: Slow Path MLOps (Weeks 5-6)
*   **Critical Path:** Drift detection.
*   **Tasks:** SQLite historical aggregator, Kolmogorov-Smirnov drift detector, ZeroMQ IPC Trigger bus.
*   **Dependencies:** Requires Sprint 1 Ring Buffer.

### Sprint 4: Autonomous Optimization (Weeks 7-8)
*   **Critical Path:** Asynchronous CUDA.
*   **Tasks:** Numba CUDA streams implementation, ThreadPoolExecutor for velocity updates, RF Training, ONNX Export, ZeroMQ Hot-Swap signal.
*   **Dependencies:** Requires Sprint 3 ZMQ trigger.

---

## PART 7: Project Milestones

1.  **M1 (Bootstrap Complete):** Repository initialized, CI green, developer environments successfully running `docker-compose up`. *(End of Week 1)*
2.  **M2 (Data Flow Proven):** MQTT telemetry successfully traverses the network, enters the Ring Buffer, and displays on Streamlit. *(End of Sprint 1)*
3.  **M3 (Sub-100ms Inference Proven):** TensorRT and SHAP execute end-to-end within the 100ms latency budget. *(End of Sprint 2)*
4.  **M4 (Autonomy Proven):** A simulated data drift successfully triggers a background Numba PSO retraining loop without crashing the Fast Path. *(End of Sprint 4)*
5.  **M5 (Edge Deployment Ready):** The production ARM64 Docker container successfully boots and runs on the physical Nvidia Jetson Orin Nano hardware. *(Week 9)*

---

## PART 8: Architecture Compliance (ARB Translation)

The Architecture Review Board (ARB) approved the architecture with specific recommendations. I have translated these directly into implementation tasks. None of these require redesigning the architecture; they are standard engineering hardening tasks.

| ARB Category | ARB Recommendation | Engineering Task Implementation |
| :--- | :--- | :--- |
| **Product** | Alarm suppression cooldown | Add a `last_alarm_time` state in `Inference` to enforce a 60-second minimum gap between identical alarms. |
| **Software** | Robust lock-free Ring Buffer | Utilize Python's `multiprocessing.shared_memory` with atomic integer counters for Head/Tail. |
| **AI/ML** | Drift trigger debounce | Require KS-Test to fail `N` consecutive times before firing ZMQ trigger. (Configurable in YAML). |
| **CUDA** | Cap VRAM allocation | Implement `torch.cuda.set_per_process_memory_fraction` (or TRT equivalent) on module startup. |
| **IoT** | MQTT persistent sessions | Set `clean_session=False` in the Paho-MQTT Alarm Publisher client. |
| **Performance** | TensorRT Warm-up | Run one forward pass with a dummy zero-tensor immediately after compiling the ONNX model, *before* hot-swapping. |
| **Reliability** | SQLite WAL mode | Execute `PRAGMA journal_mode=WAL;` upon SQLite connection initialization. |

---
**END OF DOCUMENT.**
*(Approved by Director of Engineering. Program is officially launched.)*
