import pytest
import time
import json
import threading
from unittest.mock import MagicMock
from backend.ipc.server import IPCServer
from backend.ipc.client import IPCClient

@pytest.fixture
def mock_ring_buffer():
    rb = MagicMock()
    rb.size.return_value = 5
    rb.capacity.return_value = 100
    rb.latest.return_value = {"sensor": 42}
    return rb

@pytest.fixture
def ipc_server(mock_ring_buffer):
    address = "tcp://127.0.0.1:5556"  # Use different port to avoid conflicts
    server = IPCServer(address, mock_ring_buffer)
    server.start()
    time.sleep(0.1) # Wait for bind
    yield server
    server.stop()

def test_ipc_health_command(ipc_server):
    client = IPCClient(address="tcp://127.0.0.1:5556", timeout_ms=1000)
    res = client.send_request("health")
    assert res.get("status") == "ok"
    client.close()

def test_ipc_buffer_stats_command(ipc_server, mock_ring_buffer):
    client = IPCClient(address="tcp://127.0.0.1:5556", timeout_ms=1000)
    res = client.send_request("buffer_stats")
    assert res.get("size") == 5
    assert res.get("capacity") == 100
    client.close()

def test_ipc_latest_telemetry_command(ipc_server):
    client = IPCClient(address="tcp://127.0.0.1:5556", timeout_ms=1000)
    res = client.send_request("latest_telemetry")
    assert res.get("data") == {"sensor": 42}
    client.close()

def test_ipc_unknown_command(ipc_server):
    client = IPCClient(address="tcp://127.0.0.1:5556", timeout_ms=1000)
    res = client.send_request("foo_bar")
    assert "error" in res
    client.close()

def test_ipc_timeout_retries():
    # Connect to a non-existent server port
    client = IPCClient(address="tcp://127.0.0.1:5557", timeout_ms=100)
    start_time = time.time()
    res = client.send_request("health", retries=2)
    duration = time.time() - start_time
    
    # Should take at least 200ms (2 retries * 100ms)
    assert duration >= 0.2
    assert "error" in res
    assert "timeout" in res["error"].lower()
    client.close()
