import time
import json
from backend.logger import get_logger

logger = get_logger(__name__)

class TelemetryManager:
    """
    Acts as the bridge between incoming MQTT messages and the Ring Buffer.
    Validates, normalizes, and appends payloads without ML processing.
    """
    def __init__(self, mqtt_client, ring_buffer):
        self.mqtt_client = mqtt_client
        self.ring_buffer = ring_buffer
        self.messages_received = 0
        self.messages_dropped = 0
        
        # Override the on_message callback from MQTT client
        self.mqtt_client.client.on_message = self.on_message

    def on_message(self, client, userdata, msg):
        """Callback to handle incoming MQTT messages."""
        self.messages_received += 1
        try:
            payload = msg.payload.decode('utf-8')
            parsed_data = json.loads(payload)
            
            # Basic validation: ensure it's a dict
            if not isinstance(parsed_data, dict):
                raise ValueError("Payload must be a JSON object")
                
            # Normalize schema: ensure timestamp exists
            if 'timestamp' not in parsed_data:
                parsed_data['timestamp'] = time.time()
                
            # Append to ring buffer
            self.ring_buffer.append(parsed_data)
            logger.debug(f"Telemetry appended to buffer. Topic: {msg.topic}")
            
        except json.JSONDecodeError as e:
            self.messages_dropped += 1
            logger.error(f"Invalid JSON dropped from {msg.topic}: {e}")
        except Exception as e:
            self.messages_dropped += 1
            logger.error(f"Error processing payload on {msg.topic}: {e}")
