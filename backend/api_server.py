from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from typing import Optional
from backend.ipc.client import IPCClient
from backend.config import config

app = FastAPI(
    title="AeroForge REST API",
    description="REST Gateway to the AeroForge GPU Daemon via ZeroMQ IPC",
    version="1.0.0"
)

# Connect IPC Client to the Daemon
ipc_cfg = config.get_ipc_config()
client = IPCClient(address=ipc_cfg['address'], timeout_ms=ipc_cfg['timeout_ms'])

class PredictionRequest(BaseModel):
    temperature_air: float
    temperature_process: float
    speed_rpm: float
    torque: float
    tool_wear: float

def send_ipc(command: str, payload: dict = None):
    res = client.send_request(command, payload)
    if "error" in res:
        raise HTTPException(status_code=503, detail=res["error"])
    return res

@app.get("/health")
def health_check():
    """Check health of the backend daemon."""
    return send_ipc("health")

@app.get("/version")
def version():
    """Get the current version of the system."""
    return send_ipc("version")

@app.get("/metrics")
@app.get("/buffer")
def buffer_stats():
    """Retrieve Ring Buffer usage metrics."""
    return send_ipc("buffer_stats")

@app.get("/telemetry/latest")
def latest_telemetry():
    """Retrieve the single most recent telemetry point."""
    return send_ipc("latest_telemetry")

@app.post("/predict")
def predict(req: Optional[PredictionRequest] = None):
    """
    Run TensorRT prediction + SHAP explanation.
    If no payload is provided, it predicts on the latest telemetry in the buffer.
    """
    payload = req.dict() if req else None
    return send_ipc("predict", payload)

@app.get("/drift/status")
def drift_status():
    """Retrieve the latest drift detection report."""
    return send_ipc("drift_status")

@app.get("/retraining/status")
def retraining_status():
    """Retrieve the status of the background PSO retraining pipeline."""
    return send_ipc("retraining_status")

if __name__ == "__main__":
    uvicorn.run("backend.api_server:app", host="0.0.0.0", port=8000, reload=False)
