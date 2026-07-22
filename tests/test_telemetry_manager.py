import pytest
import json
from unittest.mock import MagicMock
from backend.telemetry_manager import TelemetryManager
from backend.ring_buffer import RingBuffer

@pytest.fixture
def mock_mqtt_client():
    client = MagicMock()
    return client

@pytest.fixture
def telemetry_manager(mock_mqtt_client):
    rb = RingBuffer(capacity=10)
    return TelemetryManager(mock_mqtt_client, rb)

def test_telemetry_manager_initialization(telemetry_manager, mock_mqtt_client):
    assert telemetry_manager.messages_received == 0
    assert telemetry_manager.messages_dropped == 0
    # ensure on_message is wired correctly
    assert mock_mqtt_client.client.on_message == telemetry_manager.on_message

def test_valid_payload_insertion(telemetry_manager):
    msg = MagicMock()
    msg.topic = "test/topic"
    msg.payload = json.dumps({"sensor": "val", "timestamp": 123456}).encode('utf-8')
    
    telemetry_manager.on_message(None, None, msg)
    
    assert telemetry_manager.messages_received == 1
    assert telemetry_manager.messages_dropped == 0
    assert telemetry_manager.ring_buffer.size() == 1
    latest = telemetry_manager.ring_buffer.latest()
    assert latest['sensor'] == "val"
    assert latest['timestamp'] == 123456

def test_valid_payload_adds_timestamp(telemetry_manager):
    msg = MagicMock()
    msg.topic = "test/topic"
    # No timestamp included
    msg.payload = json.dumps({"sensor": "val"}).encode('utf-8')
    
    telemetry_manager.on_message(None, None, msg)
    
    assert telemetry_manager.messages_received == 1
    assert telemetry_manager.messages_dropped == 0
    latest = telemetry_manager.ring_buffer.latest()
    assert 'timestamp' in latest

def test_invalid_json_dropped(telemetry_manager):
    msg = MagicMock()
    msg.topic = "test/topic"
    msg.payload = b"not valid json"
    
    telemetry_manager.on_message(None, None, msg)
    
    assert telemetry_manager.messages_received == 1
    assert telemetry_manager.messages_dropped == 1
    assert telemetry_manager.ring_buffer.size() == 0

def test_invalid_payload_type_dropped(telemetry_manager):
    msg = MagicMock()
    msg.topic = "test/topic"
    # Valid JSON, but not a dict (a list)
    msg.payload = json.dumps([1, 2, 3]).encode('utf-8')
    
    telemetry_manager.on_message(None, None, msg)
    
    assert telemetry_manager.messages_received == 1
    assert telemetry_manager.messages_dropped == 1
    assert telemetry_manager.ring_buffer.size() == 0
