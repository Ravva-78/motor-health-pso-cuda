import onnxruntime as rt
import numpy as np
from backend.logger import get_logger

logger = get_logger(__name__)

class InferenceService:
    def __init__(self, model_path: str = "backend/model.onnx"):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self._initialize_engine()

    def _initialize_engine(self):
        logger.info("Initializing ONNXRuntime Inference Engine...")
        
        # Configure TensorRT caching
        so = rt.SessionOptions()
        so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = [
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': 'backend/.trt_cache',
                    'trt_fp16_enable': True, # Use FP16 if supported, else FP32
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]
        
        try:
            self.session = rt.InferenceSession(self.model_path, so, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            active_providers = self.session.get_providers()
            logger.info(f"Inference Engine initialized successfully. Active providers: {active_providers}")
            
            # Warm-up inference
            logger.info("Running TensorRT warm-up inference...")
            dummy_input = np.zeros((1, 5), dtype=np.float32)
            self.session.run(None, {self.input_name: dummy_input})
            logger.info("Warm-up complete.")
            
        except Exception as e:
            logger.error(f"Failed to initialize Inference Engine: {e}")
            raise

    def reload_model(self):
        logger.info("Hot-swapping Inference Engine...")
        so = rt.SessionOptions()
        so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = [
            ('TensorrtExecutionProvider', {
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': 'backend/.trt_cache',
                'trt_fp16_enable': True,
            }),
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]
        try:
            # Build new session first to not interrupt current inference requests
            new_session = rt.InferenceSession(self.model_path, so, providers=providers)
            input_name = new_session.get_inputs()[0].name
            
            # Warm-up new engine
            logger.info("Running TensorRT warm-up inference for new engine...")
            dummy_input = np.zeros((1, 5), dtype=np.float32)
            new_session.run(None, {input_name: dummy_input})
            
            # Atomic pointer swap
            self.session = new_session
            self.input_name = input_name
            logger.info("Inference Engine hot-swap complete.")
            return True
        except Exception as e:
            logger.error(f"Failed to hot-swap Inference Engine: {e}")
            return False

    def predict(self, features: list) -> dict:
        """
        Executes a prediction using the active Execution Provider (TensorRT > CUDA > CPU).
        Returns a dict containing label, health string, and probabilities.
        """
        if not self.session:
            return {"error": "Inference engine not initialized"}
            
        try:
            # Ensure shape is (1, 5) and float32
            x = np.array([features], dtype=np.float32)
            
            # Run inference
            res = self.session.run(None, {self.input_name: x})
            label = int(res[0][0])
            
            # Extract probability for the predicted class
            prob_dict = res[1][0]
            confidence = prob_dict.get(label, 0.0)
            
            return {
                "label": label,
                "confidence": round(confidence, 4)
            }
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {"error": str(e)}
