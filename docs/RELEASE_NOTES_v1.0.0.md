# Release Notes v1.0.0

AeroForge v1.0.0 RC-1 is the first major release candidate. It implements the core autonomous predictive maintenance loop.

**Features:**
- Real-time telemetry ingestion via MQTT.
- Thread-safe Ring Buffer for high-throughput buffering.
- ZeroMQ IPC bridging the Backend Daemon and the FastAPI Gateway.
- Dynamic hardware acceleration (TensorRT -> CUDA -> CPU).
- Autonomous PSI Drift Detection and PSO-based Random Forest retraining.
- SHAP TreeExplainer integration for real-time AI explainability.

**Status:** Ready for initial deployment and hardware validation.