# v1.0.0 — AeroForge Autonomous Predictive Edge AI Platform

We are extremely proud to announce the official **v1.0.0** release of AeroForge!

AeroForge is an enterprise-grade IoT predictive maintenance platform designed for high-frequency industrial environments. This release represents the culmination of extensive engineering to build a robust, fault-tolerant edge AI pipeline that isolates machine learning workloads from telemetry ingestion, providing real-time predictions and autonomous self-healing capabilities.

## ?? Major Features
- **Real-Time Telemetry Ingestion**: High-throughput MQTT pipeline capable of handling live, noisy sensor data with zero dropped frames.
- **Decoupled Architecture**: A lock-free Ring Buffer and ZeroMQ IPC fabric isolate the ML Daemon from the FastAPI REST gateway, ensuring sub-millisecond API response times even during heavy ML operations.
- **Autonomous Healing**: Continuous PSI (Population Stability Index) tracking detects data drift. When a motor fails or data drifts, a background Particle Swarm Optimization (PSO) routine is launched to retrain the model dynamically.
- **Graceful Hot-Swapping**: The newly optimized model is compiled to ONNX and hot-swapped into the live inference server without dropping a single incoming request.
- **Explainable AI (XAI)**: Every prediction is accompanied by a real-time SHAP analysis, providing human-readable feature contributions.
- **Live Streamlit Dashboard**: A beautiful, interactive frontend providing real-time dials, timeline charts, and drift warnings.

## ?? Performance & Research Highlights
- **Hardware Acceleration**: 
  - **PSO**: Natively accelerated using 
umba.cuda to heavily parallelize hyperparameter searches across the GPU.
  - **Inference**: Dynamically targets TensorRT/CUDA via ONNX Runtime. **(Graceful CPU Fallback Included)**: If native cuDNN or TensorRT binaries are missing on the host machine, the daemon gracefully catches the fatal exception and seamlessly falls back to the CPUExecutionProvider to prevent downtime.
- **Zero-Downtime Retraining**: The PSO optimizer leverages a detached subprocess thread, ensuring the live stream never stutters while cross-validating 400+ Random Forest configurations.

## ?? Known Limitations
- The current architecture is designed for a single massive edge node. Horizontal scaling via Kubernetes is planned for v1.1.0.
- Docker scaffolding is provided but has not been officially validated in production environments. Native deployment is recommended for this release.

## ?? Installation
Please refer to the updated [README.md](README.md) for full installation and quick-start instructions.

## ?? Acknowledgements
Special thanks to the engineering team and contributors who validated this release candidate!
