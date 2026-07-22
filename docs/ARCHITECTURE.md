# AeroForge Architecture

## System Topology
AeroForge relies on a decoupled microservices architecture connected via IPC (ZeroMQ) and MQTT.

1. **MQTT Broker**: Ingests high-frequency telemetry.
2. **Backend Daemon (`backend_daemon.py`)**: 
   - Subscribes to MQTT.
   - Pushes data into a thread-safe `RingBuffer`.
   - Analyzes for data drift (`DriftDetector`).
   - Invokes PSO-based retraining if drift > threshold.
   - Provides an IPC Server via ZeroMQ to serve inference requests to the API.
3. **REST API (`api_server.py`)**:
   - FastAPI gateway for frontend communication.
   - Sends IPC requests to Daemon for inference.
4. **Dashboard (`app.py`)**:
   - Streamlit application displaying real-time predictions and SHAP explainability.

## Hardware Acceleration Status
- **ONNX Runtime**: Implemented. Attempts `TensorrtExecutionProvider`, then `CUDAExecutionProvider`, then `CPUExecutionProvider`.
- **PSO/Numba**: Implemented. `cuda_module.py` will use NVIDIA GPU for fitness evaluation if available, otherwise CPU.