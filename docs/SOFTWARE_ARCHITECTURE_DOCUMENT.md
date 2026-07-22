# Software Architecture Document (SAD)

*See AEROFORGE_SAD.md for the legacy/design phase SAD.*

## Current Architectural State (v1.0.0)
AeroForge is implemented as a multi-process architecture:
1. **Process 1**: `backend_daemon.py` (Heavy lifting, ML, IO).
2. **Process 2**: `api_server.py` (Lightweight asynchronous REST).
3. **Process 3**: `app.py` (Streamlit Dashboard).
4. **Process 4**: Eclipse Mosquitto (Message Broker).

This decoupling ensures that ML training spikes do not block incoming HTTP or MQTT requests.