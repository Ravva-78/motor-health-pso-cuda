# AEROFORGE MASTER ENGINEERING HANDBOOK
**Status:** OFFICIAL
**Version:** 1.0 (Implementation Ready)
**Scope:** Architecture, Design, and Program Initialization

---

## TABLE OF CONTENTS
1. [PART 1: Executive Summary](#part-1-executive-summary)
2. [PART 2: Project Evolution Timeline](#part-2-project-evolution-timeline)
3. [PART 3: Complete Product Definition](#part-3-complete-product-definition)
4. [PART 4: Technical Decision History (TDR)](#part-4-technical-decision-history)
5. [PART 5: Architecture Evolution](#part-5-architecture-evolution)
6. [PART 6: Subsystem Deep Dive](#part-6-subsystem-deep-dive)
7. [PART 7: Engineering Decisions (The "Why")](#part-7-engineering-decisions)
8. [PART 8: Architecture Review Board (ARB)](#part-8-architecture-review-board)
9. [PART 9: Engineering Program](#part-9-engineering-program)
10. [PART 10: Future Evolution](#part-10-future-evolution)
11. [PART 11: Lessons Learned](#part-11-lessons-learned)
12. [PART 12: Appendices](#part-12-appendices)

---

## PART 1: Executive Summary

### What AeroForge Is
AeroForge is an autonomous, self-healing, edge-native predictive maintenance platform engineered to monitor industrial motors. It ingests high-frequency telemetry, performs ultra-low-latency inference, generates cryptographically sound explanations for its predictions, and autonomously retrains its models using GPU-accelerated heuristic swarm optimization when hardware degradation (concept drift) is detected.

### Why It Exists
In heavy manufacturing, unexpected motor failures cause catastrophic downtime. Existing solutions rely on cloud processing, which introduces unacceptable network latency, extreme bandwidth costs, and security risks. Furthermore, static AI models decay over time as machinery ages, creating "alarm fatigue" among operators. AeroForge exists to solve these exact bottlenecks by pushing autonomous, self-healing intelligence to the extreme edge.

### Engineering Philosophy
*   **Edge-First, Cloud-Optional:** Compute happens where the physics happens.
*   **Decoupled & Asynchronous:** I/O bounds and compute bounds must never block each other.
*   **Trust Through Transparency:** AI predictions without root-cause explanations are useless on a factory floor.

---

## PART 2: Project Evolution Timeline

The engineering lifecycle followed a strict 7-Gate process to transition from an academic prototype to an enterprise-grade blueprint.

*   **Gate 0: Vision Freeze**
    *   *Purpose:* Lock the product identity and 5 flagship innovations.
    *   *Decision:* Rejected standard dashboarding in favor of Asynchronous HPC (CUDA streams).
*   **Gate 1: Product Requirements Document (PRD)**
    *   *Purpose:* Define the "What" and "Why" without technical implementation.
    *   *Output:* Defined target personas, KPIs (100ms latency), and use-case scenarios.
*   **Gate 2: Technical Requirements Document (TRD)**
    *   *Purpose:* Define the technology stack.
    *   *Output:* Technology Decision Records (TDR) locking in Python, TensorRT, MQTT, and Numba.
*   **Gate 3: Software Architecture Design (SAD)**
    *   *Purpose:* Define the "How".
    *   *Output:* Layered architecture, IPC (ZMQ) boundaries, Circular Ring Buffers, and Failure Isolations.
*   **Gate 4: Detailed Engineering Design (DED)**
    *   *Purpose:* Convert architecture to an implementation backlog.
    *   *Output:* Frozen repository structures, API specs, and a 4-Epic sprint plan.
*   **Gate 5: Architecture Review Board (ARB)**
    *   *Purpose:* External audit by Principal Architects.
    *   *Output:* Approved with a 9.2/10 score. Added WAL mode for SQLite and TensorRT warmups.
*   **Gate 6: Engineering Program Initialization (EPI)**
    *   *Purpose:* Launch the development program.
    *   *Output:* CI/CD pipelines, Docker strategies, and strict branch policies.

---

## PART 3: Complete Product Definition

*   **Vision:** To eliminate catastrophic industrial downtime through autonomous, self-healing edge intelligence.
*   **Mission:** Engineer a modular pipeline unifying asynchronous CUDA heuristics with ultra-low latency inference.
*   **Target Users:** Maintenance Supervisors, Plant Managers, OT Network Admins.
*   **Industry 4.0 Relevance:** Directly solves the OT/IT convergence problem by processing data locally, preserving bandwidth, and keeping proprietary telemetry off the public cloud.

**Flagship Features:**
1.  Real-Time MQTT Telemetry
2.  Asynchronous HPC Pipelines (CUDA PSO)
3.  Edge AI TensorRT Deployment
4.  Explainable AI (TreeSHAP)
5.  Adaptive Self-Healing (Concept Drift)

---

## PART 4: Technical Decision History

### TDR: Python
*   **Why:** Rapid prototyping and seamless bindings to C++ ML libraries (TensorRT, ONNX).
*   **Tradeoff:** GIL concurrency limits, solved by multi-processing IPC architectures.

### TDR: Numba (CUDA)
*   **Why:** JIT compilation abstracts PTX architectures across various Jetson hardware generations.
*   **Tradeoff:** Less control than raw C++, but 95% of the performance with 10% of the maintenance overhead.

### TDR: Random Forest & TensorRT
*   **Why:** RF provides deterministic, non-linear classification highly compatible with SHAP. TensorRT accelerates inference up to 40x over CPU.
*   **Tradeoff:** Models must be converted via ONNX, requiring strict version pinning of `skl2onnx`.

### TDR: MQTT
*   **Why:** Lightweight publish/subscribe. De facto industrial standard. Built for unstable edge networks.
*   **Tradeoff:** Lacks native persistent streams like Kafka, necessitating local SQLite aggregations for historical data.

### TDR: ZeroMQ (IPC)
*   **Why:** Brokerless, lightning-fast message passing between Fast Path and Slow Path processes.
*   **Tradeoff:** Requires careful handling of socket lifecycles to prevent ghost processes.

---

## PART 5: Architecture Evolution

The architecture evolved from a monolithic script into a **decoupled micro-monolith** based on the realization that PSO training and real-time inference have fundamentally incompatible execution profiles.

*   **Data Flow:** Telemetry is strictly volatile. It streams via MQTT into a Shared Memory Ring Buffer. The Inference module reads it at 10Hz; the Autonomy module reads it at 0.1Hz.
*   **Control Flow:** Orchestrated via ZeroMQ events. `TRIGGER_TRAIN` initiates the GPU optimization. `MODEL_READY` signals the hot-swap.
*   **Memory Model:** Lock-free atomic pointers on the Ring Buffer prevent Python Garbage Collection pauses. CUDA memory is strictly pre-allocated (Pinned Memory) to prevent dynamic fragmentation.
*   **Failure Isolation:** If the Optimization process encounters a CUDA illegal memory access, the process crashes and restarts independently. The Inference process never misses a 100ms MQTT frame.

---

## PART 6: Subsystem Deep Dive

### 1. Ingestion Subsystem (MQTT)
*   **Responsibilities:** Subscribes to factory sensors, validates JSON payloads, writes to the Ring Buffer tail.
*   **Failure Scenarios:** Network drop -> Exponential backoff reconnect. Local buffering to SQLite.

### 2. Data Fabric (Ring Buffer)
*   **Responsibilities:** Single source of truth. Holds exactly 10,000 frames. 
*   **Performance:** Zero-copy reads via Python `shared_memory` eliminates serialization overhead between processes.

### 3. Inference Subsystem (Fast Path)
*   **Responsibilities:** Polls Ring Buffer -> MinMax Scale -> TensorRT Execution -> TreeSHAP Explanation -> Publish Alarm.
*   **Interfaces:** Reads `.onnx` models from disk via Hot-Swap pointers.

### 4. Autonomy Subsystem (MLOps)
*   **Responsibilities:** Computes Kolmogorov-Smirnov (KS) tests between the live Ring Buffer and the historical SQLite baseline. 
*   **Logic:** If KS-Test breaches p=0.05 for 5 consecutive minutes, it fires `TRIGGER_TRAIN`.

### 5. Optimization Subsystem (Slow Path)
*   **Responsibilities:** Executes Particle Swarm Optimization utilizing Numba CUDA Streams. Evaluates fitness of hyperparameters, trains Random Forest, exports to ONNX.

---

## PART 7: Engineering Decisions (The "Why")

*   **Why a Ring Buffer?** Memory allocation is expensive. By allocating a contiguous block of memory once and overwriting the oldest data, we achieve O(1) ingestion time and eliminate memory leaks.
*   **Why Edge AI?** Industrial networks frequently drop packets. If a motor bearing seizes during a 30-second WiFi outage, a cloud-based system will miss it. The edge node guarantees survival offline.
*   **Why PSO instead of Grid Search?** PSO translates perfectly into SIMT (Single Instruction, Multiple Thread) architecture. The vector math of particle velocity updates allows us to utilize the Jetson's GPU cores massively in parallel, whereas Grid Search is historically a CPU-bound looping construct.
*   **Why SQLite?** We do not need a distributed relational database. We only need a lightweight, serverless mechanism to persist hourly aggregations to disk so that baseline knowledge survives a power outage.

---

## PART 8: Architecture Review Board Summary

The board consisted of Principal Architects from NVIDIA, Tesla, Bosch, Microsoft, Siemens, and ABB. 

*   **Score:** 9.2 / 10
*   **Strengths:** Exceptional isolation of failure domains. Best-in-class use of CUDA streams and TensorRT.
*   **Risks Identified:** MQTT flooding on borderline drift; TensorRT JIT latency spikes.
*   **Mitigations Applied:** Alarm debounce logic (1-minute cooldown); TensorRT dummy-pass warmups prior to hot-swapping; SQLite WAL mode enabled.
*   **Decision:** **GO.** Architecture Frozen.

---

## PART 9: Engineering Program

The implementation is mapped into a strict chronological sequence of Epics:
1.  **Epic 1: Foundation & Fabric** (Docker, Ring Buffer, MQTT Subscribe)
2.  **Epic 2: Inference & Edge AI** (TensorRT, TreeSHAP, MQTT Publish)
3.  **Epic 3: Autonomy & MLOps** (SQLite baseline, KS-Test Drift Detection)
4.  **Epic 4: Asynchronous HPC Optimization** (Numba PSO, ONNX Export, ZeroMQ Hot-Swap)

**Quality Gates:**
*   `Poetry` for dependency lockfiles.
*   `Flake8`, `Black`, and strict `MyPy`.
*   CI/CD runs `PyTest` integration tests using emulated Mosquitto containers before allowing merges to `main`.
*   Docker edge images are scanned via `Trivy`.

---

## PART 10: Future Evolution (V2+)

AeroForge is designed for extensibility:
*   **Multi-Motor Batching:** The Ring Buffer can be partitioned by `motor_id`, allowing a single TensorRT engine to process inference batches for 50+ motors simultaneously.
*   **Federated Learning:** Optimization nodes will push their `gBest` particles to a centralized parameter server. This allows a motor in a Berlin factory to learn from the failure patterns of a motor in a Texas factory, without transmitting raw sensor data.
*   **Hardware-in-the-Loop (HIL):** Bridging MQTT alarms directly to PLC controllers (via OPC-UA) to automatically throttle motor RPM when a CRITICAL alarm is generated.

---

## PART 11: Lessons Learned (Planning Phase)

*   **What worked well:** Treating architecture design as a 7-Gate enterprise process forced the team to resolve deep integration issues (like the GIL blocking CUDA) before writing a single line of code.
*   **Important Architectural Lesson:** "Digital Twins" and 3D dashboards look good in demos, but they do not solve the hard physics and latency problems of the factory floor. Replacing visual fluff with Asynchronous HPC pipelines drastically elevated the engineering caliber of the project.
*   **Documentation Lesson:** Maintaining a single source of truth (DED translating to Epics) prevents scope creep and developer misalignment.

---

## PART 12: Appendices

### Glossary
*   **Drift (Concept Drift):** When the statistical properties of the target variable change over time, rendering static AI models inaccurate.
*   **ONNX:** Open Neural Network Exchange format.
*   **TreeSHAP:** A variant of SHapley Additive exPlanations optimized for tree-based machine learning models.
*   **IPC:** Inter-Process Communication.
*   **ZMQ:** ZeroMQ, an asynchronous messaging library used for internal component signaling.

### Technology Stack
*   **Language:** Python 3.10
*   **ML/AI:** Scikit-Learn, TensorRT, SHAP, Numba (CUDA)
*   **Messaging:** Eclipse Mosquitto (MQTT), ZeroMQ
*   **Persistence:** SQLite3
*   **Infrastructure:** Docker, k3s, GitHub Actions
*   **UI:** Streamlit

**--- END OF HANDBOOK ---**
*(The source of truth for AeroForge Engineering.)*
