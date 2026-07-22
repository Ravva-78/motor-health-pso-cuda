import shap
import pickle
import numpy as np
from backend.logger import get_logger

logger = get_logger(__name__)

class SHAPService:
    def __init__(self, pickle_path: str = "backend/saved_model.pkl"):
        self.pickle_path = pickle_path
        self.explainer = None
        self.feature_names = [
            'temperature_air', 'temperature_process',
            'speed_rpm', 'torque', 'tool_wear'
        ]
        self._initialize_explainer()

    def _initialize_explainer(self):
        logger.info("Initializing SHAP TreeExplainer from Pickle fallback...")
        try:
            with open(self.pickle_path, 'rb') as f:
                model = pickle.load(f)
                
            self.explainer = shap.TreeExplainer(model)
            logger.info("SHAP Explainer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            self.explainer = None

    def explain(self, features: list, predicted_label: int) -> dict:
        """
        Calculates SHAP values for the specific input.
        Returns a structured JSON mapping of feature contributions.
        """
        if not self.explainer:
            return {"error": "SHAP Explainer not initialized"}
            
        try:
            x = np.array([features], dtype=np.float32)
            
            # Get SHAP values. Shape: (n_samples, n_features, n_classes) for RF
            shap_values = self.explainer.shap_values(x)
            
            # Extract the SHAP values for the predicted class
            # Note: For some RFs, shap_values is a list of arrays (one per class).
            if isinstance(shap_values, list):
                class_shap = shap_values[predicted_label][0]
            else:
                # If it's a 3D array (n_samples, n_features, n_classes)
                if len(shap_values.shape) == 3:
                    class_shap = shap_values[0, :, predicted_label]
                else:
                    class_shap = shap_values[0]
                    
            # Map features to their contributions
            contributions = {}
            for name, val in zip(self.feature_names, class_shap):
                contributions[name] = float(round(val, 4))
                
            # Rank top features by absolute impact
            ranked = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
            top_features = [item[0] for item in ranked[:3]]
            
            return {
                "top_features": top_features,
                "feature_contributions": contributions
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation error: {e}")
            return {"error": str(e)}
