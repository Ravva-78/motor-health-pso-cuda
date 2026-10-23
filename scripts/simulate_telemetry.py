import time
import json
import random
import sys
import paho.mqtt.client as mqtt

broker_address = "localhost"
port = 1883
topic = "aeroforge/telemetry"

def generate_telemetry_sample():
    """Generates a realistic multi-variate motor telemetry payload."""
    return {
        "temperature_air": round(random.uniform(295.0, 305.0), 2),
        "temperature_process": round(random.uniform(305.0, 315.0), 2),
        "speed_rpm": round(random.uniform(1450.0, 1550.0), 2),
        "torque": round(random.uniform(35.0, 45.0), 2),
        "tool_wear": round(random.uniform(10.0, 15.0), 2)
    }

def run_simulation(max_samples=None, interval=2.0):
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "AeroForgeSimulator")
    connected = False
    
    try:
        print(f"Connecting to MQTT Broker at {broker_address}:{port}...")
        client.connect(broker_address, port, keepalive=10)
        client.loop_start()
        connected = True
        print(f"Connected successfully. Publishing simulated telemetry to '{topic}'. Press Ctrl+C to stop.")
    except Exception as e:
        print(f"Notice: Local MQTT broker not reachable ({e}). Running in standalone mock publisher mode.")

    count = 0
    try:
        while True:
            payload = generate_telemetry_sample()
            if connected:
                client.publish(topic, json.dumps(payload))
                print(f"[MQTT Published #{count+1} -> {topic}]: {payload}")
            else:
                print(f"[Mock Telemetry #{count+1}]: {payload}")
            
            count += 1
            if max_samples and count >= max_samples:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nSimulation halted by user.")
    finally:
        if connected:
            client.loop_stop()
            client.disconnect()

if __name__ == "__main__":
    max_count = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_simulation(max_samples=max_count)

