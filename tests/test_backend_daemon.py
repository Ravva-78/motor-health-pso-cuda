import pytest
import threading
import time
import signal
from unittest.mock import MagicMock, patch
from backend.backend_daemon import BackendDaemon

@pytest.fixture
def mock_dependencies():
    with patch('backend.backend_daemon.RingBuffer') as mock_rb, \
         patch('backend.backend_daemon.MQTTTelemetryClient') as mock_mqtt, \
         patch('backend.backend_daemon.TelemetryManager') as mock_tm:
         
        yield mock_rb, mock_mqtt, mock_tm

def test_backend_daemon_initialization(mock_dependencies):
    daemon = BackendDaemon()
    assert daemon.running is False
    assert daemon.start_time is None

def test_backend_daemon_start_and_shutdown(mock_dependencies):
    mock_rb, mock_mqtt, mock_tm = mock_dependencies
    daemon = BackendDaemon()
    
    # Run the daemon in a separate thread so it doesn't block
    t = threading.Thread(target=daemon.start)
    t.start()
    
    # Wait for it to start
    time.sleep(0.1)
    assert daemon.running is True
    assert daemon.start_time is not None
    
    # Signal shutdown
    daemon.shutdown()
    t.join(timeout=1.0)
    
    assert daemon.running is False
    
    # Verify connections were managed
    mqtt_instance = mock_mqtt.return_value
    mqtt_instance.connect.assert_called_once()
    mqtt_instance.disconnect.assert_called_once()

def test_backend_daemon_signal_handling(mock_dependencies):
    daemon = BackendDaemon()
    daemon.running = True
    
    # Simulate a signal call
    daemon.handle_shutdown(signal.SIGTERM, None)
    
    assert daemon.running is False
