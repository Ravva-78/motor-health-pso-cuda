import pytest
import json
from unittest.mock import MagicMock, patch
from backend.mqtt_client import MQTTTelemetryClient

@pytest.fixture
def mock_config():
    return {
        'broker': 'localhost',
        'port': 1883,
        'topics': ['test/topic'],
        'client_id': 'test_client'
    }

@pytest.fixture
def mqtt_client(mock_config):
    with patch('backend.mqtt_client.mqtt.Client') as mock_mqtt:
        client = MQTTTelemetryClient(mock_config)
        yield client

def test_mqtt_client_initialization(mock_config):
    client = MQTTTelemetryClient(mock_config)
    assert client.broker == 'localhost'
    assert client.port == 1883
    assert client.client_id == 'test_client'
    assert client.topics == ['test/topic']

def test_mqtt_connect_success(mqtt_client):
    mqtt_client.connect()
    mqtt_client.client.connect.assert_called_once_with('localhost', 1883, keepalive=60)
    mqtt_client.client.loop_start.assert_called_once()

def test_mqtt_disconnect(mqtt_client):
    mqtt_client.disconnect()
    mqtt_client.client.loop_stop.assert_called_once()
    mqtt_client.client.disconnect.assert_called_once()

def test_mqtt_publish(mqtt_client):
    payload = {"temp": 45.2}
    mqtt_client.publish('test/topic', payload)
    mqtt_client.client.publish.assert_called_once_with(
        'test/topic', json.dumps(payload), qos=1
    )

def test_on_connect_success(mqtt_client):
    # Simulate a successful connection (reason_code = 0)
    mqtt_client.on_connect(None, None, None, 0, None)
    mqtt_client.client.subscribe.assert_called_once_with('test/topic', qos=1)

def test_on_message_valid_json(mqtt_client, caplog):
    # Simulate an incoming MQTT message
    msg = MagicMock()
    msg.topic = "test/topic"
    msg.payload = b'{"temp": 50.0, "vibration": 1.2}'
    
    with caplog.at_level("INFO"):
        mqtt_client.on_message(None, None, msg)
    
    assert "Received valid JSON telemetry on topic test/topic" in caplog.text

def test_on_message_invalid_json_does_not_crash(mqtt_client, caplog):
    # Simulate an incoming MQTT message with malformed JSON
    msg = MagicMock()
    msg.topic = "test/topic"
    msg.payload = b'{malformed: json}'
    
    with caplog.at_level("ERROR"):
        mqtt_client.on_message(None, None, msg)
        
    assert "Invalid JSON received on topic test/topic" in caplog.text
    # Should safely catch the error and not raise
