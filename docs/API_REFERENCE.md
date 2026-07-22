# API Reference

## FastAPI Endpoints (`api_server.py`)

### `GET /health`
Returns the status of the API Server and the IPC connection to the Daemon.

### `POST /predict`
- **Body**: JSON array of features `[temperature_air, temperature_process, speed_rpm, torque, tool_wear]`
- **Returns**: JSON object with `label` (0 or 1), `confidence` (float), and `explanation` (SHAP values).