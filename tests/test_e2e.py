import pytest
import time
import threading
import json
from fastapi.testclient import TestClient

# Import components
from backend.backend_daemon import BackendDaemon
from backend.api_server import app
from backend.config import config

# We use FastAPI's TestClient to hit the REST endpoints
client = TestClient(app)

@pytest.fixture(scope="module")
def e2e_system():
    # Disable actual MQTT connection for test reliability
    daemon = BackendDaemon()
    daemon.mqtt_client.connect = lambda: None
    daemon.mqtt_client.disconnect = lambda: None
    
    # Start Daemon in background
    t = threading.Thread(target=daemon.start, daemon=True)
    t.start()
    
    # Wait for IPC Server to bind
    time.sleep(2)
    
    yield daemon
    
    # Teardown
    daemon.shutdown()
    t.join(timeout=2)

def test_rest_api_health(e2e_system):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_telemetry_pipeline(e2e_system):
    # Inject fake telemetry directly into TelemetryManager (simulating MQTT payload)
    fake_payload = {
        "temperature_air": 300.0,
        "temperature_process": 310.0,
        "speed_rpm": 1500.0,
        "torque": 40.0,
        "tool_wear": 10.0
    }
    
    class MockMsg:
        def __init__(self, payload_dict):
            self.payload = json.dumps(payload_dict).encode('utf-8')
            self.topic = "test/topic"
            
    msg = MockMsg(fake_payload)
    e2e_system.telemetry_manager.on_message(None, None, msg)
    time.sleep(0.5)
    
    # Validate it reached the Buffer via REST API
    response = client.get("/telemetry/latest")
    assert response.status_code == 200
    data = response.json().get("data")
    assert data is not None
    assert data["speed_rpm"] == 1500.0

def test_inference_pipeline(e2e_system):
    # Trigger prediction on the latest buffer data via REST API
    response = client.post("/predict")
    assert response.status_code == 200
    
    data = response.json()
    assert "label" in data
    assert "confidence" in data
    assert "feature_contributions" in data
    
def test_drift_and_retraining_status(e2e_system):
    # Check Drift
    d_res = client.get("/drift/status")
    assert d_res.status_code == 200
    assert "drift_detected" in d_res.json()
    
    # Check Retraining
    r_res = client.get("/retraining/status")
    assert r_res.status_code == 200
    assert "is_retraining" in r_res.json()
