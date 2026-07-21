# Detailed Engineering Design (DED)
**Project Name:** AeroForge – Autonomous Predictive Edge AI Platform
**Document Owner:** Principal Engineering Manager
**Reference Documents:** Vision Freeze, PRD, TRD, SAD
**Status:** IMPLEMENTATION READY

---

## PART 1: Repository Structure (FROZEN)

```text
aeroforge/
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Lint, type-check, unit tests
│       └── cd.yml                 # Docker build, scan, registry push
├── config/
│   ├── config.prod.yaml           # Production environment configurations
│   └── config.dev.yaml            # Development overrides
├── data/
│   └── historical.db              # SQLite persistence for drift baseline
├── docker/
│   ├── Dockerfile.edge            # Multi-stage edge build (Jetson)
│   ├── docker-compose.yml         # Local dev orchestration
│   └── mosquitto.conf             # MQTT broker settings
├── docs/
│   └── architecture/              # PRD, TRD, SAD, DED, ADRs
├── src/
│   ├── backend/
│   │   ├── api/                   # FastAPI REST controllers (Admin)
│   │   ├── core/                  # Shared IPC, Ring Buffer definitions
│   │   ├── inference/             # TensorRT execution, GPU TreeSHAP
│   │   ├── ingestion/             # MQTT Client, Buffer writes
│   │   ├── mlops/                 # KS-Test Drift detector
│   │   └── optimization/          # Numba CUDA, PSO loop, RF training
│   ├── frontend/
│   │   ├── app.py                 # Streamlit entrypoint
│   │   ├── components/            # Reusable UI (Gauges, SHAP Waterfall)
│   │   └── views/                 # Multi-page layouts (Dashboard, Config)
├── tests/
│   ├── unit/                      # Isolated module tests
│   ├── integration/               # Cross-module IPC tests
│   └── e2e/                       # End-to-end telemetry -> alarm tests
├── .env.example                   # Secret templates (DO NOT COMMIT SECRETS)
├── .gitignore                     # Excludes virtualenvs, .onnx, .db
├── pyproject.toml                 # Poetry dependency & build management
└── README.md                      # Developer onboarding
```

---

## PART 2: Backend Design

### Module: `src/backend/ingestion`
*   **Responsibilities:** Subscribe to MQTT broker, parse JSON, validate schema, enqueue to Ring Buffer.
*   **Interfaces:** Paho-MQTT async subscriber; SharedMemory RingBuffer writer.
*   **Dependencies:** `paho-mqtt`, `pydantic` (Schema validation).

### Module: `src/backend/inference`
*   **Responsibilities:** Poll Ring Buffer at fixed Hz, run TensorRT forward pass, compute SHAP, publish MQTT Alarms.
*   **Interfaces:** SharedMemory RingBuffer reader; TensorRT context executor; Paho-MQTT async publisher.
*   **Dependencies:** `tensorrt`, `shap`, `numpy`.

### Module: `src/backend/optimization`
*   **Responsibilities:** Receive IPC ZMQ trigger, load historical dataset, execute Asynchronous CUDA PSO, train RF, export ONNX, signal IPC ZMQ ready.
*   **Interfaces:** ZeroMQ subscriber (`TRIGGER_TRAIN`); SQLite reader; ZeroMQ publisher (`MODEL_READY`).
*   **Dependencies:** `numba`, `scikit-learn`, `skl2onnx`, `pyzmq`.

### Module: `src/backend/mlops`
*   **Responsibilities:** Continuously test current Ring Buffer data against historical baseline for Kolmogorov-Smirnov statistical drift.
*   **Interfaces:** SharedMemory RingBuffer reader; SQLite reader; ZeroMQ publisher (`TRIGGER_TRAIN`).
*   **Dependencies:** `scipy.stats`.

---

## PART 3: Frontend Design (Streamlit)

### Pages
1.  `/views/dashboard.py`: Live gauge visualization, prediction status, active alarms.
2.  `/views/diagnostics.py`: Historical trends, SHAP waterfall charts for root cause analysis.
3.  `/views/system.py`: Edge device health (GPU VRAM, CPU utilization, Model drift status).

### Components
*   `components/gauge_card.py`: Reusable stylized dark-glassmorphism sensor gauge.
*   `components/shap_waterfall.py`: Custom wrapper integrating Plotly SHAP graphs.
*   `components/status_pill.py`: Dynamic CSS pill (Green/Yellow/Red) based on health state.

### State Management & Navigation
*   **Navigation:** Uses `st.navigation` and `st.Page` (Streamlit 1.36+ native multi-page routing).
*   **State Management:** `st.session_state` strictly holds UI toggle states (e.g., dark mode, selected motor). Live telemetry is NOT stored in session state; it is dynamically queried from the local MQTT subscriber on every render loop using `st_mqtt`.

---

## PART 4: MQTT Specification

### Topics
*   **Ingestion:** `factory/motor/{motor_id}/telemetry`
*   **Alarms:** `factory/motor/{motor_id}/alarms`
*   **System:** `aeroforge/system/status`
*   **MLOps:** `aeroforge/mlops/drift_events`

### Payloads
*   **Telemetry (JSON):** `{"ts": 1690000000, "ta": 298.5, "tp": 309.2, "sp": 1500, "tq": 40.1, "tw": 102}`
*   **Alarm (JSON):** `{"ts": 1690000005, "status": "CRITICAL", "shap_values": {"tq": +2.4, "tw": +1.1}, "confidence": 0.98}`

### QoS & Reliability
*   **Telemetry:** QoS 0 (At most once). Missing a single frame is acceptable; speed is paramount.
*   **Alarms:** QoS 1 (At least once). Alarms must be acknowledged by the broker to ensure delivery.
*   **Reconnect Strategy:** Exponential backoff (1s, 2s, 4s, 8s, max 60s).
*   **Error Topics:** Dead Letter Queue at `aeroforge/system/dlq` for malformed JSON payloads.

---

## PART 5: REST API Specification
*(Used strictly for Edge Device Administration/Debugging, NOT telemetry)*

### Base URL: `http://localhost:8000/api/v1`

| Endpoint | Method | Purpose | Response |
| :--- | :--- | :--- | :--- |
| `/health` | GET | Check system readiness | `200 OK: {"status": "healthy"}` |
| `/model/active` | GET | Get current ONNX model hash | `200 OK: {"hash": "a1b2c3d4", "epochs": 20}` |
| `/model/trigger` | POST | Manually trigger PSO retrain | `202 Accepted: {"job_id": "pso_99"}` |
| `/buffer/stats` | GET | Check Ring Buffer utilization | `200 OK: {"capacity": 10000, "used": 4500}` |

**Errors:**
*   `400 Bad Request` (Invalid manual trigger parameters)
*   `503 Service Unavailable` (Inference engine down)

---

## PART 6: Database Design (SQLite)
*(Used purely for baselining and historical context, NOT for live inference buffering)*

### Schema
**Table: `historical_aggregates`**
*   `id` (INTEGER, Primary Key)
*   `timestamp` (INTEGER, Index)
*   `motor_id` (TEXT, Index)
*   `ta_mean` (REAL), `tp_mean` (REAL), `sp_mean` (REAL), `tq_mean` (REAL), `tw_max` (REAL)
*   `health_status` (TEXT)

**Table: `drift_events`**
*   `id` (INTEGER)
*   `timestamp` (INTEGER)
*   `ks_statistic` (REAL)
*   `p_value` (REAL)
*   `triggered_retrain` (BOOLEAN)

### Retention & Backup
*   **Retention:** Data older than 30 days is purged daily via a background CRON thread to prevent Edge SSD wear and capacity limits.
*   **Backup:** None. Data is ephemeral to the Edge device. If the node dies, a new node rebuilds baselines from scratch over 24 hours.

---

## PART 7: Configuration Design

### YAML (`config.yaml`)
```yaml
system:
  motor_id: "MOTOR-AXIS-01"
  log_level: "INFO"
mqtt:
  broker_url: "localhost"
  port: 1883
  qos: 0
buffer:
  capacity: 10000
pso:
  n_particles: 20
  max_iter: 20
drift:
  p_value_threshold: 0.05
  window_size: 1000
```

### Environment Variables (Secrets)
*   `MQTT_USER`
*   `MQTT_PASS`
*   `API_ADMIN_TOKEN`

### Feature Flags (UI)
*   `ENABLE_3D_TWIN=false` (Toggle WebGL components if Jetson GPU is under heavy load).

---

## PART 8: AI/ML Design

### Pipeline Definitions
*   **Training Pipeline:** PSO generates hyperparameter bounds -> RF fits on historical SQLite baseline -> `skl2onnx` exports `.onnx`.
*   **Validation Pipeline:** K-Fold cross validation during PSO fitness evaluation.
*   **Inference Pipeline:** Ring Buffer batch -> MinMax Scaler -> TensorRT `execute_async_v2` -> Softmax output.
*   **Retraining Pipeline:** Triggered by MLOps KS-Test -> Locks historical DB for read -> Executes Training Pipeline -> Signals Inference Pipeline.

### Model Registry
*   Models are stored locally at `/var/opt/aeroforge/models/`.
*   Naming convention: `rf_model_epoch_{ts}_{hash}.onnx`.
*   Registry retains the last 3 models. Older models are purged to save disk space.

---

## PART 9: CUDA Design (Numba)

### Responsibilities
*   Calculate Random Forest prediction accuracy across 20 distinct particle hyperparameter sets simultaneously.

### Memory Layout
*   **Pinned Memory:** Particle positions (hyperparameters) `[20 x 4]`.
*   **Device Memory (Persistent):** Training Dataset X `[N x 5]`, Training Dataset Y `[N]`.
*   **Device Memory (Volatile):** Fitness array `[20]`.

### Synchronization & Streams
*   2 CUDA Streams instantiated.
*   `Stream 1`: Evaluates particles 1-10.
*   `Stream 2`: Evaluates particles 11-20.
*   CPU computes Velocity for Particle `N` while GPU evaluates Fitness for Particle `N+1`.
*   `cuda.synchronize()` explicitly called ONLY at the end of the PSO iteration.

### Performance Targets
*   Total PSO completion (20 particles, 20 iterations, 10k samples) < 3.0 seconds on Jetson Orin Nano.

---

## PART 10: Testing Strategy

### Suites
1.  **Unit Tests (PyTest):** Test PSO vector mathematics independently of CUDA. Test JSON schema validation.
2.  **Integration Tests:** Start an in-memory MQTT broker, publish mock telemetry, assert Ring Buffer tail advances.
3.  **System Tests:** Send known anomalous telemetry stream, assert `factory/motor/alarms` topic receives CRITICAL payload within 100ms.
4.  **Failure Tests:** Kill the `aeroforge-optimization` process midway through training. Assert Inference continues uninterrupted.
5.  **Performance Tests:** Flood MQTT with 1000Hz telemetry. Assert Ring Buffer drops oldest frames gracefully without memory leaks.

---

## PART 11: DevOps

### Docker Multi-Stage Build
*   **Stage 1 (Builder):** Uses full `nvcr.io/nvidia/l4t-ml` image. Installs compilers, builds Python C-extensions (Numba/TensorRT bindings).
*   **Stage 2 (Runtime):** Copies only compiled wheels and source code to a slim `l4t-base` image to minimize edge deployment size (< 1GB).

### CI/CD (GitHub Actions)
1.  **On Pull Request:** Run `flake8`, `mypy`, `pytest` (Unit/Integration).
2.  **On Push to `main`:** Build ARM64 Docker image using QEMU, run `trivy` security scan, push to GitHub Container Registry (GHCR).

### Git Strategy
*   `main` branch is protected. Requires 1 approving review.
*   Semantic Versioning tags trigger release builds (`v4.0.0`).

---

## PART 12: Engineering Backlog

*(To be imported into Jira/Linear)*

### Epic 1: Foundation & Data Fabric
*   **Feature 1.1:** Setup Multi-Stage Dockerfile for Jetson.
*   **Feature 1.2:** Implement SharedMemory Ring Buffer core module.
*   **Feature 1.3:** Implement MQTT Ingestion subscriber with JSON validation.
*   **Feature 1.4:** Build initial Streamlit layout (Empty views).

### Epic 2: Inference & Edge AI
*   **Feature 2.1:** Implement ONNX-to-TensorRT compilation script.
*   **Feature 2.2:** Implement Inference module polling Ring Buffer.
*   **Feature 2.3:** Integrate GPU TreeSHAP explainer.
*   **Feature 2.4:** Implement MQTT Alarm publisher.

### Epic 3: Autonomy & MLOps
*   **Feature 3.1:** Implement SQLite historical aggregator.
*   **Feature 3.2:** Implement Kolmogorov-Smirnov drift detector on sliding windows.
*   **Feature 3.3:** Implement ZeroMQ IPC event bus (TRIGGER_TRAIN).

### Epic 4: Asynchronous HPC Optimization
*   **Feature 4.1:** Rewrite Numba kernel for asynchronous memory streams.
*   **Feature 4.2:** Implement Python ThreadPoolExecutor for overlapping CPU velocity updates.
*   **Feature 4.3:** Implement RF training and ONNX export.
*   **Feature 4.4:** Implement ZeroMQ IPC event bus (MODEL_READY) and hot-swapping logic.

### Implementation Order
**Epic 1 -> Epic 2 -> Epic 3 -> Epic 4.**
*Dependency Rationale:* Inference (Epic 2) cannot be built until the Ring Buffer (Epic 1) exists. Optimization (Epic 4) cannot be built until the Drift Trigger (Epic 3) is capable of requesting it.

---
**END OF DOCUMENT.**
*(Approved by Principal Engineering Manager. Hand off to Development Teams.)*
