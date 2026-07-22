
# Revision 2 (v1.1 / Rev-B) - Final Engineering Audit
**Date**: July 2026
**Status**: Synchronized with Repository Truth (v1.0.0 RC-1)

## Revision History
* **Rev-A (Original)**: Initial architecture and requirements freeze.
* **Rev-B (v1.1)**: Final Engineering Audit synchronization. Reflects actual implementation truth over initial design assumptions.

## Repository Truth Alignment (Rev-B)
* **Docker/Deployment**: Scaffolded but pending full production validation.
* **Hardware Acceleration (TensorRT/CUDA)**: Implemented dynamically. Code attempts to bind to TensorRT and CUDA via ONNX/Numba, but falls back to CPU if drivers are absent. Hardware execution was not validated in the CI test suite.
* **ML Pipeline**: Fully implemented and validated via unit tests (Inference, SHAP, Drift, Retraining).
---

# Technical Requirements Document (TRD)
**Project Name:** AeroForge – Autonomous Predictive Edge AI Platform
**Authors:** Ravva Nagarjun

**Status:** APPROVED FOR ENGINEERING

---

## PART 1: Executive Technical Summary
AeroForge is an edge-native, GPU-accelerated autonomous predictive maintenance platform. This document serves as the absolute architectural source of truth for the engineering lifecycle. The core engineering mandate is to build a highly decoupled, asynchronous, and robust system that safely isolates I/O bound ingestion (MQTT) from heavy compute-bound heuristics (PSO/CUDA) and ultra-low latency inference (TensorRT). We are prioritizing deterministic execution, memory safety, and high-throughput over developer convenience.

---

## PART 2: System Constraints & Assumptions

### System Constraints
1.  **Hardware Boundaries:** The primary inference edge node is constrained to the memory limits of an Nvidia Jetson Orin Nano (8GB Unified Memory). 
2.  **Network Air-Gap:** The system must assume zero external internet connectivity. All dependencies, models, and drift logic must execute locally.
3.  **Latency Budget:** Total round-trip time from MQTT ingestion to MQTT alarm publication must not exceed 100 milliseconds.
4.  **No Dynamic Memory Allocation in Kernels:** All CUDA kernel memory must be pre-allocated by the host to prevent fragmentation.

### Engineering Assumptions
1.  Telemetry payload structures (JSON over MQTT) will remain stable during a single motor's lifecycle.
2.  The operating environment (Linux/Ubuntu) will support containerization (Docker/k3s) with GPU passthrough enabled.

### Technical Risks
1.  **GPU Context Switching:** Running TensorRT inference and Numba CUDA PSO optimization simultaneously on a single edge GPU could cause context-switching overhead and memory starvation.
2.  **Drift False Positives:** High-variance telemetry could trigger continuous retraining loops, degrading hardware and consuming power.

---

## PART 3: Technology Decision Records (TDR)

### TDR-01: Programming Language (Python)
*   **Decision:** Python 3.10+
*   **Alternatives:** C++, Rust, Go
*   **Pros:** De facto standard for AI/ML; rapid integration with Numba and TensorRT APIs.
*   **Cons:** GIL (Global Interpreter Lock) bottlenecks; higher memory overhead than C++.
*   **Tradeoffs:** Sacrificing bare-metal thread performance for vastly accelerated ML iteration speed.
*   **Final Justification:** The heavy compute is offloaded to CUDA and TensorRT; Python simply acts as the orchestration and C-binding layer.

### TDR-02: GPU Programming (CUDA via Numba)
*   **Decision:** Numba (JIT compilation)
*   **Alternatives:** PyCUDA, Raw C++/CUDA wrappers (pybind11).
*   **Pros:** Native Python integration; JIT optimization avoids pre-compiled binary hell across different Jetson architectures.
*   **Cons:** Less granular control over low-level PTX instructions compared to raw C++.
*   **Final Justification:** Numba provides 95% of the performance of C++ CUDA with 10% of the maintenance overhead, which is critical for an agile MLOps lifecycle.

### TDR-03: Machine Learning Model (Random Forest)
*   **Decision:** Random Forest (RF)
*   **Alternatives:** Deep Neural Networks (DNN), Support Vector Machines (SVM).
*   **Pros:** Deterministic inference latency; highly interpretable (SHAP compatibility); immune to feature scaling issues.
*   **Cons:** Large memory footprint for deep trees.
*   **Final Justification:** Industrial operators require explainability. RF provides a perfect balance of non-linear classification and feature importance without the black-box nature of DNNs.

### TDR-04: Optimization Heuristic (PSO)
*   **Decision:** Particle Swarm Optimization (PSO)
*   **Alternatives:** Genetic Algorithms (GA), Bayesian Optimization, Grid Search.
*   **Pros:** Memory-efficient (particles only store current state and pBest); continuous space optimization; highly parallelizable on GPUs.
*   **Cons:** Risk of premature convergence in non-convex spaces.
*   **Final Justification:** PSO translates perfectly to grid-stride CUDA kernels. The vector mathematics of velocity updates align seamlessly with SIMT architecture.

### TDR-05: Ingestion Protocol (MQTT)
*   **Decision:** MQTT (Mosquitto Broker)
*   **Alternatives:** Kafka, RabbitMQ, HTTP REST.
*   **Pros:** Extreme lightweight footprint; built for unstable edge networks; publish/subscribe decoupling.
*   **Cons:** Lacks the persistent distributed logging of Kafka.
*   **Final Justification:** Kafka requires a JVM and massive RAM overhead, rendering it unfit for Jetson Edge deployment. MQTT is the industry standard for OT/IT bridging.

### TDR-06: Inference Engine (TensorRT)
*   **Decision:** Nvidia TensorRT
*   **Alternatives:** ONNX Runtime, scikit-learn (CPU).
*   **Pros:** Hardware-specific kernel auto-tuning; INT8/FP16 precision quantization.
*   **Cons:** Hardware lock-in (models compiled on Jetson cannot run on a desktop RTX card).
*   **Final Justification:** Latency is critical. TensorRT provides up to 40x inference acceleration over CPU-based scikit-learn.

### TDR-07: Model Interoperability (ONNX)
*   **Decision:** Open Neural Network Exchange (ONNX)
*   **Alternatives:** Pickle, PMML.
*   **Pros:** Universal intermediate representation; bridge between scikit-learn (skl2onnx) and TensorRT.
*   **Cons:** Conversion quirks for certain scikit-learn estimators.
*   **Final Justification:** Pickle files represent a severe security vulnerability (arbitrary code execution). ONNX provides a secure, static execution graph.

### TDR-08: Explainability (SHAP)
*   **Decision:** TreeSHAP (GPU Accelerated)
*   **Alternatives:** LIME, Feature Importance (Gini).
*   **Pros:** Cryptographically solid game-theoretic explanations; consistent feature attribution.
*   **Cons:** Computationally expensive to calculate marginal contributions.
*   **Final Justification:** TreeSHAP is the only method that provides local accuracy and consistency, which is legally and operationally required for critical industrial alarms.

### TDR-09: User Interface (Streamlit)
*   **Decision:** Streamlit
*   **Alternatives:** React/Node.js, Grafana.
*   **Pros:** Pure Python; no dual-language context switching; native integration with Pandas/Plotly.
*   **Cons:** State management is complex; not designed for 60Hz real-time rendering.
*   **Final Justification:** The UI is strictly an observer. Streamlit allows rapid deployment of custom Python data visualizations (like SHAP waterfall charts) that Grafana struggles with.

---

## PART 4: Architecture Decision Records (ADR)

### ADR-01: Decoupling Inference and Optimization via IPC
**Context:** PSO optimization is a massive GPU compute load. Inference must happen every 100ms. If they share a process, the GIL or CUDA context lock will cause inference delays.
**Decision:** We will strictly separate the Inference Engine and the Optimization Engine into distinct OS processes communicating via IPC (Inter-Process Communication) or Redis.
**Consequences:** Increased architectural complexity; requires explicit model checkpointing (ONNX files) to hand off models between processes. Guarantees 100% inference uptime.

### ADR-02: Single Source of Truth via Circular Buffers
**Context:** Both Inference and Drift Detection require access to live telemetry.
**Decision:** Implement a lock-free circular buffer (Ring Buffer) in shared memory. The MQTT subscriber writes to the head; Inference and Drift detectors read from fixed-offset tails.
**Consequences:** Zero-copy reads prevent memory bloat and garbage collection pauses in Python.

---

## PART 5: Complete Modular Architecture

### Layered Architecture
1.  **I/O Layer (Ingestion & Egress):** MQTT Subscribers/Publishers.
2.  **Data Fabric Layer:** Shared Memory Ring Buffers.
3.  **Inference Layer (Fast Path):** TensorRT execution, SHAP generation.
4.  **Autonomy Layer (Slow Path):** Drift Detection, Asynchronous PSO, ONNX Compilation.
5.  **Presentation Layer:** Streamlit Dashboard.

### Responsibilities & Isolation
*   **Fast Path:** Must NEVER block. It reads data, infers, and publishes. If it crashes, the process supervisor (systemd/k3s) restarts it instantly.
*   **Slow Path:** Can be safely killed or paused without halting production monitoring.
*   **Communication:** Inter-module communication occurs strictly via event buses (MQTT or Redis). No direct function calls between Fast and Slow paths.

---

## PART 6: Module Specifications

### Module: `AeroForge.Ingestion`
*   **Purpose:** Subscribe to factory sensors and populate the ring buffer.
*   **Inputs:** MQTT JSON payloads.
*   **Outputs:** Shared memory writes.
*   **Dependencies:** `paho-mqtt`.
*   **Error Handling:** Reconnects with exponential backoff on network drop.
*   **Future Expansion:** Support for OPC-UA protocols.

### Module: `AeroForge.Inference`
*   **Purpose:** Execute TensorRT engine against telemetry.
*   **Inputs:** N-dimensional array from ring buffer.
*   **Outputs:** Health classification and SHAP array -> MQTT Alarm Topic.
*   **Dependencies:** `tensorrt`.
*   **Error Handling:** Falls back to CPU ONNX runtime if TensorRT context fails.

### Module: `AeroForge.Autonomy`
*   **Purpose:** Monitor data for drift and trigger self-healing.
*   **Inputs:** Historical baseline, live sliding window.
*   **Outputs:** Trigger signal to PSO module.
*   **Dependencies:** `scipy`.

### Module: `AeroForge.Optimization`
*   **Purpose:** Run Swarm intelligence to find optimal RF hyperparameters.
*   **Inputs:** Training dataset, Drift trigger.
*   **Outputs:** New `.onnx` model file.
*   **Dependencies:** `numba`, `scikit-learn`, `skl2onnx`.

---

## PART 7: Complete Data Flow

1.  **End-to-End Packet Flow:** Motor Sensor -> WiFi/Ethernet -> Jetson Network Interface -> Mosquitto Broker -> `AeroForge.Ingestion` -> Ring Buffer.
2.  **Inference Flow:** Ring Buffer (Read) -> Preprocessing (MinMax) -> TensorRT Engine (Execute) -> GPU_SHAP (Execute) -> JSON Alarm Construct -> Mosquitto Broker.
3.  **Optimization Flow:** Drift Trigger -> Query Local Historical DB -> Initialize PSO Swarm -> Transfer to GPU Memory -> Numba Kernel (Evaluate) -> CPU (Update) -> Iterate -> Best Params -> Train RF -> Export ONNX.
4.  **Retraining Handshake:** `Optimization` saves `v2.onnx`. It publishes an MQTT message `system/model_update`. `Inference` receives message, instantiates a secondary TensorRT context, swaps pointers, and destroys the old context (Hot Swap).

---

## PART 8: CUDA Architecture

### Memory Hierarchy
*   **Host Memory (Pinned):** Particle positions and velocities are stored in page-locked (pinned) memory for maximum PCIe transfer bandwidth.
*   **Device Memory:** The training dataset (X, Y) resides permanently in VRAM during the PSO lifecycle to avoid redundant transfers.
### Execution Model
*   **CUDA Streams:** Memory transfers (`cudaMemcpyAsync`) and kernel executions are placed in non-default CUDA streams. This allows CPU velocity calculations for Swarm A to overlap with GPU fitness evaluations for Swarm B.
*   **Kernel Synchronization:** `cudaStreamSynchronize` is used strictly at the end of a generation barrier.
### Performance Optimizations
*   **Grid-Stride Loops:** Kernels are written using grid-stride loops to ensure hardware abstraction and coalesced memory access across varying GPU SM counts.

---

## PART 9: Concurrency Architecture

### Asynchronous Execution
*   The system utilizes Python's `asyncio` for all I/O bound tasks (MQTT subscriptions).
*   Compute-bound tasks (Model Training) are offloaded to `ProcessPoolExecutor` to bypass the GIL.
### Message Queues
*   Internal IPC is handled via lightweight ZeroMQ (ZMQ) sockets to prevent dropping telemetry bursts during garbage collection pauses.
### Failure Recovery
*   If the Optimization process exceeds a memory limit (OOM killer), the Inference process remains untouched. A supervisor daemon restarts Optimization, which resumes from the last saved Checkpoint on disk.

---

## PART 10: Edge AI Architecture

### Hardware Deployment
*   **Primary Node:** Nvidia Jetson Orin Nano.
*   **Microcontroller Interface:** ESP32 microcontrollers gather physical analog signals and bridge them to WiFi/MQTT.
### Deployment Pipeline
*   The system is packaged as an OCI-compliant Docker container.
*   A cross-compilation pipeline builds the container on x86 CI/CD servers (GitHub Actions) utilizing QEMU for ARM64 architecture, pushing to a private registry.
*   The Jetson runs `k3s`, pulling the latest image and executing with `--gpus all`.

---

## PART 11: Repository Structure (FROZEN)

```text
aeroforge/
├── .github/workflows/       (CI/CD logic)
├── aeroforge/
│   ├── ingestion/           (MQTT & OPC-UA clients)
│   ├── inference/           (TensorRT & ONNX execution)
│   ├── optimization/        (PSO & Numba Kernels)
│   ├── autonomy/            (Drift & MLOps triggers)
│   ├── shared/              (Ring Buffers & IPC logic)
│   └── ui/                  (Streamlit views)
├── tests/                   (Unit and Integration suites)
├── deploy/                  (Dockerfiles, k3s manifests)
├── docs/                    (ADRs, TRD, API spec)
├── config/                  (YAML environment configs)
```

---

## PART 12: Coding Standards

*   **Logging:** Strict use of structured JSON logging (`structlog`). No `print()` statements. Logs must be easily ingested by ELK/Splunk.
*   **Testing:** 80% line coverage required. Integration tests must use `pytest-asyncio` and spun-up Mosquitto containers.
*   **Documentation:** Google-style docstrings enforced via `pylint`.
*   **Versioning:** Semantic Versioning (SemVer 2.0). APIs must be versioned (e.g., `topic: v1/telemetry`).
*   **Error Handling:** Exceptions must be typed and caught at the module boundary. Unhandled exceptions trigger a ZMQ fatal alert to the supervisor before exiting.

---

## PART 13: Future Scalability

*   **Version 2 (Multi-Motor):** Expanding the architecture from monitoring a single asset to a fleet of 50 motors per Edge Node using dynamically allocated CUDA streams.
*   **Version 3 (Digital Twin & Cloud):** Pushing aggregated anomaly metadata (not raw telemetry) to an AWS/Azure cloud backend for fleet-wide 3D visualization and cross-factory federated learning. 
*   **Version 4 (Hardware-in-the-Loop):** Closing the loop by sending PLC throttling commands (SCADA) to automatically reduce motor load when an imminent failure is detected.

---
**END OF DOCUMENT.**
*(Approved by Chief System Architect)*
