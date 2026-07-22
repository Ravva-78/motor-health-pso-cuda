# AeroForge v1.0.0 — Autonomous Predictive Edge AI Platform

![AeroForge](https://img.shields.io/badge/AeroForge-v1.0.0-blue?style=for-the-badge)
![Python 3.11](https://img.shields.io/badge/Python-3.11-brightgreen?style=for-the-badge&logo=python)
![CUDA 12](https://img.shields.io/badge/CUDA-12.8-76B900?style=for-the-badge&logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

**AeroForge** is an enterprise-grade IoT predictive maintenance platform designed for high-frequency industrial environments. It ingests live motor telemetry via MQTT, performs sub-millisecond AI inference, provides real-time SHAP explainability, and features a zero-downtime autonomous retraining pipeline (via Particle Swarm Optimization) when data drift occurs.

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Architecture Highlights](#-architecture-highlights)
- [Key Features](#-key-features)
- [Performance & Fallback Capabilities](#-performance--fallback-capabilities)
- [Repository Structure](#-repository-structure)
- [Installation & Quick Start](#-installation--quick-start)
- [Documentation](#-documentation)
- [License](#-license)

---

## 🎯 Overview

Built as a robust, edge-ready application, AeroForge isolates ML workloads from networking overhead using a lock-free Ring Buffer and ZeroMQ IPC. It actively monitors incoming telemetry for data drift using PSI (Population Stability Index). When drift is detected, the system launches a background Particle Swarm Optimization (PSO) routine using `numba.cuda` to retrain a Random Forest model, which is then seamlessly hot-swapped into production via ONNX Runtime without dropping a single inference request.

## 🏗 Architecture Highlights

- **Data Ingestion**: High-throughput MQTT client fetching live sensor data.
- **IPC Fabric**: ZeroMQ orchestrates communication between the background daemon and the FastAPI REST server.
- **Inference Engine**: Hardware-accelerated ONNX Runtime inference natively targeting TensorRT/CUDA.
- **Explainability**: Real-time SHAP feature contribution analysis.
- **Dashboard**: Live interactive Streamlit dashboard rendering telemetry, drift alerts, and predictions simultaneously.

## ✨ Key Features

1. **Sub-Millisecond Inference**: Highly optimized ONNX predictions.
2. **Autonomous Healing**: Continuous PSI drift detection triggers background PSO retraining.
3. **Graceful Hot-Swapping**: Zero-downtime model updates in live production.
4. **Real-time Explainable AI**: Understand exactly *why* a motor is predicted to fail.
5. **Decoupled Architecture**: Ensures the REST API remains highly responsive even during massive background ML training tasks.

## 🚀 Performance & Fallback Capabilities

### Graceful CPU Fallback (ONNX ML Inference)
To maximize compatibility across development environments, the ONNX Inference engine checks for native NVIDIA `cuDNN` and `TensorRT` binaries. **If these C++ binaries are missing from your system, the daemon executes a seamless, graceful fallback to the `CPUExecutionProvider`.** The system will not crash and will remain fully operational.

*(To enable full TensorRT acceleration, ensure you have correctly installed cuDNN and TensorRT into your local CUDA toolkit directories).*

### CUDA Acceleration (PSO Retraining)
The Particle Swarm Optimization (PSO) component actively utilizes your GPU via `numba.cuda`. This relies solely on the base CUDA toolkit and is fully functional out of the box, massively accelerating the hyperparameter search space during retraining.

## 📂 Repository Structure

```text
motor-health-pso-cuda/
├── app.py                  # Streamlit Live Dashboard (Frontend)
├── backend/                # Core ML Daemon, ZeroMQ, MQTT, Drift, PSO
├── config.yaml             # System configuration parameters
├── data/                   # Telemetry datasets
├── docs/                   # Comprehensive Architecture & Engineering docs
├── scripts/                # Utility scripts (e.g., simulate_telemetry.py)
└── tests/                  # Unit and integration test suite
```

## ⚙️ Installation & Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/your-org/motor-health-pso-cuda.git
cd motor-health-pso-cuda
pip install -r requirements.txt
```

### 2. Start the Backend Daemon
The daemon handles MQTT ingestion, ZeroMQ routing, ML Inference, and Drift Detection.
```bash
python -m backend.backend_daemon
```

### 3. Start the REST API
In a new terminal window, boot the FastAPI gateway.
```bash
uvicorn backend.api_server:app --port 8000
```

### 4. Launch the Dashboard
In a third terminal window, spin up the Streamlit UI.
```bash
streamlit run app.py
```

### 5. Simulate Live Telemetry
Click "Start Simulation" directly within the Streamlit dashboard, or manually run the script:
```bash
python scripts/simulate_telemetry.py
```

## 📚 Documentation
For detailed insights into the architecture, deployment strategies, and API contracts, please review our comprehensive engineering handbooks located in the [`docs/`](docs/) directory.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.