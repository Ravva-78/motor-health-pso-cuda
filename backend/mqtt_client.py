import json
import time
import paho.mqtt.client as mqtt
from backend.logger import get_logger

logger = get_logger(__name__)

class MQTTTelemetryClient:
    def __init__(self, config: dict):
        self.broker = config.get('broker', 'localhost')
        self.port = config.get('port', 1883)
        self.client_id = config.get('client_id', 'aeroforge_backend')
        self.topics = config.get('topics', ['aeroforge/telemetry'])
        self.qos = config.get('qos', 1)
        self.username = config.get('username')
        self.password = config.get('password')
        self.min_reconnect_delay = config.get('min_reconnect_delay', 1)
        self.max_reconnect_delay = config.get('max_reconnect_delay', 120)

        # Initialize paho-mqtt client using recommended API for v2
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self.client_id)
        
        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)

        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        self.client.on_subscribe = self.on_subscribe

        self.client.reconnect_delay_set(
            min_delay=self.min_reconnect_delay, 
            max_delay=self.max_reconnect_delay
        )

    def connect(self):
        """Connect to the MQTT broker and start the background loop."""
        logger.info(f"Connecting to MQTT broker at {self.broker}:{self.port}")
        try:
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            raise

    def disconnect(self):
        """Gracefully disconnect from the broker."""
        logger.info("Disconnecting from MQTT broker...")
        self.client.loop_stop()
        self.client.disconnect()

    def publish(self, topic: str, payload: dict, qos: int = None):
        """Publish a JSON payload to a specific topic."""
        qos = qos if qos is not None else self.qos
        try:
            json_payload = json.dumps(payload)
            self.client.publish(topic, json_payload, qos=qos)
            logger.debug(f"Published to {topic}: {json_payload}")
        except Exception as e:
            logger.error(f"Failed to publish to {topic}: {e}")

    def on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            logger.info("Successfully connected to MQTT broker.")
            for topic in self.topics:
                logger.info(f"Subscribing to topic: {topic}")
                self.client.subscribe(topic, qos=self.qos)
        else:
            logger.error(f"Failed to connect. Reason code: {reason_code}")

    def on_disconnect(self, client, userdata, flags, reason_code, properties):
        logger.warning(f"Disconnected from MQTT broker. Reason code: {reason_code}")
        # loop_start() handles automatic reconnection via exponential backoff

    def on_subscribe(self, client, userdata, mid, reason_code_list, properties):
        logger.info(f"Subscription confirmed. Message ID: {mid}")

    def on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode('utf-8')
            parsed_data = json.loads(payload)
            logger.info(f"Received valid JSON telemetry on topic {msg.topic}")
            logger.debug(f"Payload content: {parsed_data}")
            # NOTE: Data is parsed but NOT processed here (as per AF-006 rules)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON received on topic {msg.topic}: {e}")
            logger.debug(f"Raw payload: {msg.payload}")
        except Exception as e:
            logger.error(f"Unexpected error processing message on {msg.topic}: {e}")
