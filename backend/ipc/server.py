import zmq
import json
import threading
import time
from backend.logger import get_logger

logger = get_logger(__name__)

class IPCServer:
    def __init__(self, address: str, ring_buffer, inference_service=None, shap_service=None, drift_detector=None, retraining_service=None):
        self.address = address
        self.ring_buffer = ring_buffer
        self.inference_service = inference_service
        self.shap_service = shap_service
        self.drift_detector = drift_detector
        self.retraining_service = retraining_service
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
                elif command == 'drift_status':
                    if self.drift_detector:
                        # Fetch latest score from drift detector
                        response = self.drift_detector.get_latest_report()
                    else:
                        response = {"error": "Drift detector unavailable"}
                elif command == 'retraining_status':
                    if self.retraining_service:
                        response = {
                            "is_retraining": self.retraining_service.is_retraining(),
                            "last_retraining_time": self.retraining_service.get_last_retraining_time()
                        }
                    else:
                        response = {"error": "Retraining service unavailable"}
                elif command == 'predict':
                    if not self.inference_service or not self.shap_service:
                        response = {"error": "Inference/SHAP services unavailable"}
                    else:
                        payload = req.get('payload')
                        if not payload:
                            # If no payload, grab latest from buffer
                            payload = self.ring_buffer.latest()
                            
                        if not payload:
                            response = {"error": "No telemetry data available for prediction"}
                        else:
                            # Extract features
                            feature_keys = ['temperature_air', 'temperature_process', 'speed_rpm', 'torque', 'tool_wear']
                            try:
                                features = [float(payload[k]) for k in feature_keys]
                                
                                # 1. Predict (TensorRT)
                                pred_res = self.inference_service.predict(features)
                                
                                if "error" in pred_res:
                                    response = pred_res
                                else:
                                    # 2. Explain (SHAP)
                                    shap_res = self.shap_service.explain(features, pred_res['label'])
                                    
                                    if "error" in shap_res:
                                        response = pred_res # return prediction anyway
                                        response['explanation_error'] = shap_res['error']
                                    else:
                                        # 3. Combine
                                        response = {**pred_res, **shap_res}
                                        
                            except KeyError as e:
                                response = {"error": f"Missing feature in payload: {e}"}
                            except ValueError as e:
                                response = {"error": f"Invalid feature value: {e}"}
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
