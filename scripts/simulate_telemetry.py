import time
import json
import random
import paho.mqtt.client as mqtt

broker_address = "localhost"
port = 1883
topic = "aeroforge/telemetry"

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "AeroForgeSimulator")

try:
    print(f"Connecting to MQTT Broker at {broker_address}:{port}...")
    client.connect(broker_address, port)
    client.loop_start()
    
    print(f"Publishing simulated telemetry to '{topic}' every 2 seconds. Press Ctrl+C to stop.")
    
    while True:
        # Generate some semi-realistic noisy telemetry
        payload = {
            "temperature_air": round(random.uniform(295.0, 305.0), 1),
            "temperature_process": round(random.uniform(305.0, 315.0), 1),
            "speed_rpm": round(random.uniform(1450.0, 1550.0), 1),
            "torque": round(random.uniform(35.0, 45.0), 1),
            "tool_wear": round(random.uniform(10.0, 15.0), 1)
        }
        
        client.publish(topic, json.dumps(payload))
        print(f"Published: {payload}")
        time.sleep(2)

except KeyboardInterrupt:
    print("\nStopping simulation.")
    client.loop_stop()
    client.disconnect()
except Exception as e:
    print(f"Error: {e}")
