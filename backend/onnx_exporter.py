import os
import onnx
import onnxruntime as rt
import numpy as np
import pickle
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from backend.logger import get_logger

logger = get_logger(__name__)

class ModelExporter:
    @staticmethod
    def export_to_onnx(model, save_path: str):
        """Convert a Scikit-Learn Random Forest to ONNX and save it."""
        try:
            # Random Forest takes 5 features
            initial_type = [('float_input', FloatTensorType([None, 5]))]
            onx = convert_sklearn(model, initial_types=initial_type)
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(onx.SerializeToString())
                
            logger.info(f"Model exported to ONNX successfully at {save_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export ONNX model: {e}")
            return False

    @staticmethod
    def validate_parity(sklearn_model, onnx_path: str, num_samples: int = 100) -> bool:
        """
        Validates that the ONNX model produces identical predictions 
        to the original Sklearn model (within floating point tolerance).
        """
        try:
            sess = rt.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            input_name = sess.get_inputs()[0].name
            
            # Generate random test data matching the feature shape (5 features)
            # Typically scaled data is roughly between -3 and 3
            X_test = np.random.uniform(-3, 3, size=(num_samples, 5)).astype(np.float32)
            
            # Predict using Pickle (Sklearn)
            sklearn_preds = sklearn_model.predict(X_test)
            sklearn_proba = sklearn_model.predict_proba(X_test)
            
            # Predict using ONNX
            onnx_res = sess.run(None, {input_name: X_test})
            onnx_preds = onnx_res[0]  # First output is class labels
            
            # Extract ONNX probabilities. The second output is a list of dictionaries mapping class -> prob
            onnx_proba = []
            for prob_dict in onnx_res[1]:
                # Assuming classes are 0, 1, 2...
                probs = [prob_dict.get(c, 0.0) for c in range(sklearn_proba.shape[1])]
                onnx_proba.append(probs)
            onnx_proba = np.array(onnx_proba)
            
            # Validate predictions
            labels_match = np.array_equal(sklearn_preds, onnx_preds)
            probs_match = np.allclose(sklearn_proba, onnx_proba, atol=1e-5)
            
            if labels_match and probs_match:
                logger.info(f"Prediction parity validated successfully on {num_samples} samples.")
                return True
            else:
                logger.error(f"Prediction parity failed! Labels match: {labels_match}, Probs match: {probs_match}")
                return False
                
        except Exception as e:
            logger.error(f"Exception during parity validation: {e}")
            return False
