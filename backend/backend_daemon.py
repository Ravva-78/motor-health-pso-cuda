import sys
import time
import signal
import threading
from backend.config import config
from backend.logger import get_logger
from backend.mqtt_client import MQTTTelemetryClient
from backend.ring_buffer import RingBuffer
from backend.telemetry_manager import TelemetryManager
from backend.ipc.server import IPCServer

logger = get_logger("backend_daemon")

class BackendDaemon:
    def __init__(self):
        self.running = False
        self.start_time = None
        self.health_thread = None
        
        logger.info("Initializing Backend Daemon components...")
        
        # Initialize Core Components
        self.ring_buffer = RingBuffer(capacity=config.get_ring_buffer_config()['capacity'])
        self.mqtt_client = MQTTTelemetryClient(config.get_mqtt_config())
        self.telemetry_manager = TelemetryManager(self.mqtt_client, self.ring_buffer)
        
        ipc_config = config.get_ipc_config()
        self.ipc_server = IPCServer(ipc_config['address'], self.ring_buffer)
        
        # Setup signals
        try:
            signal.signal(signal.SIGINT, self.handle_shutdown)
            signal.signal(signal.SIGTERM, self.handle_shutdown)
        except ValueError:
            # When running in non-main threads (like pytest), signal handlers might fail. Safely ignore.
            pass

    def handle_shutdown(self, signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False

    def health_monitor(self):
        """Background thread to log health metrics every 30 seconds."""
        # Use 1-second sleeps to allow fast shutdown checking
        cycles = 0
        while self.running:
            time.sleep(1)
            cycles += 1
            
            if not self.running:
                break
                
            if cycles >= 30:
                cycles = 0
                uptime = time.time() - self.start_time
                size = self.ring_buffer.size()
                capacity = self.ring_buffer.capacity()
                received = self.telemetry_manager.messages_received
                dropped = self.telemetry_manager.messages_dropped
                
                mqtt_status = "Connected" if self.mqtt_client.client.is_connected() else "Disconnected"
                
                logger.info(
                    f"[HEALTH] Uptime: {uptime:.1f}s | "
                    f"Buffer: {size}/{capacity} | "
                    f"MQTT: {mqtt_status} | "
                    f"Msg Received: {received} | "
                    f"Msg Dropped: {dropped}"
                )

    def start(self):
        logger.info("Starting Backend Daemon...")
        self.running = True
        self.start_time = time.time()
        
        try:
            self.mqtt_client.connect()
            self.ipc_server.start()
            
            # Start health monitor thread
            self.health_thread = threading.Thread(target=self.health_monitor, daemon=True)
            self.health_thread.start()
            
            # Main event loop
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.handle_shutdown(signal.SIGINT, None)
        except Exception as e:
            logger.error(f"Daemon encountered a fatal error: {e}")
        finally:
            if not getattr(self, '_shutdown_called', False):
                self.shutdown()

    def shutdown(self):
        if getattr(self, '_shutdown_called', False):
            return
        self._shutdown_called = True
        logger.info("Shutting down Backend Daemon...")
        self.running = False
        
        try:
            self.ipc_server.stop()
        except Exception as e:
            logger.error(f"Error stopping IPC server: {e}")
            
        try:
            self.mqtt_client.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting MQTT: {e}")
        logger.info("Daemon stopped.")

if __name__ == "__main__":
    daemon = BackendDaemon()
    daemon.start()
