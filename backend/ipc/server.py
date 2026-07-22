import zmq
import json
import threading
import time
from backend.logger import get_logger

logger = get_logger(__name__)

class IPCServer:
    def __init__(self, address: str, ring_buffer):
        self.address = address
        self.ring_buffer = ring_buffer
        self.context = zmq.Context()
        self.socket = None
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info(f"IPC Server started on {self.address}")

    def _run(self):
        self.socket = self.context.socket(zmq.REP)
        
        # Use linger=0 so we don't block on exit if there are pending messages
        self.socket.setsockopt(zmq.LINGER, 0)
        # Receive timeout so we can periodically check self.running
        self.socket.setsockopt(zmq.RCVTIMEO, 500)
        
        try:
            self.socket.bind(self.address)
        except zmq.ZMQError as e:
            logger.error(f"IPC Server failed to bind to {self.address}: {e}")
            self.running = False
            return

        while self.running:
            try:
                # Wait for request
                message = self.socket.recv_string()
            except zmq.Again:
                continue
            except Exception as e:
                if not self.running:
                    break
                logger.error(f"IPC Server error receiving message: {e}")
                continue
                
            try:
                req = json.loads(message)
                command = req.get('command')
                
                # Route command
                if command == 'health':
                    response = {"status": "ok", "message": "Backend daemon is healthy"}
                elif command == 'version':
                    response = {"version": "3.1.0"}
                elif command == 'buffer_stats':
                    response = {
                        "size": self.ring_buffer.size(),
                        "capacity": self.ring_buffer.capacity()
                    }
                elif command == 'latest_telemetry':
                    response = {"data": self.ring_buffer.latest()}
                else:
                    response = {"error": f"Unknown command: {command}"}
                    
                self.socket.send_string(json.dumps(response))
                
            except json.JSONDecodeError:
                self.socket.send_string(json.dumps({"error": "Invalid JSON request format"}))
            except Exception as e:
                logger.error(f"IPC Server failed to process request: {e}")
                self.socket.send_string(json.dumps({"error": str(e)}))

    def stop(self):
        if not self.running:
            return
        logger.info("Stopping IPC Server...")
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.socket:
            self.socket.close()
        self.context.term()
        logger.info("IPC Server stopped.")
