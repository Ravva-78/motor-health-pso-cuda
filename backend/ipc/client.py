import zmq
import json
from backend.logger import get_logger

logger = get_logger(__name__)

class IPCClient:
    def __init__(self, address: str, timeout_ms: int = 2000):
        self.address = address
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self.socket = None
        self._connect()

    def _connect(self):
        if self.socket:
            self.socket.close()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(self.address)
        logger.debug(f"IPC Client connected to {self.address}")

    def send_request(self, command: str, payload: dict = None, retries: int = 3) -> dict:
        request = {"command": command}
        if payload:
            request["payload"] = payload
            
        req_str = json.dumps(request)
        
        for attempt in range(retries):
            try:
                self.socket.send_string(req_str)
                resp_str = self.socket.recv_string()
                return json.loads(resp_str)
            except zmq.Again:
                logger.warning(f"IPC request timeout on attempt {attempt + 1}/{retries}")
                self._connect()  # Recreate socket on timeout (REQ socket rule)
            except Exception as e:
                logger.error(f"IPC request error: {e}")
                self._connect()
                
        return {"error": "IPC request failed after retries due to timeout"}

    def close(self):
        if self.socket:
            self.socket.close()
        self.context.term()
