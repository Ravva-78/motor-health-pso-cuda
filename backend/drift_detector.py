import numpy as np
import pandas as pd
from backend.logger import get_logger
from backend.preprocessing import load_and_preprocess
from backend.config import config

logger = get_logger(__name__)

class DriftDetector:
    """
    Monitors live telemetry from the Ring Buffer against the statistical baseline of the training data.
    Uses Population Stability Index (PSI) to detect drift in individual features.
    """
    def __init__(self, data_path: str = "data/predictive_maintenance.csv"):
        self.drift_cfg = config.get_drift_config()
        self.enabled = self.drift_cfg['enabled']
        self.window_size = self.drift_cfg['window_size']
        self.psi_threshold = self.drift_cfg['psi_threshold']
        
        self.feature_names = ['temperature_air', 'temperature_process', 'speed_rpm', 'torque', 'tool_wear']
        self.baseline_hists = {}
        self.baseline_bins = {}
        
        if self.enabled:
            self._initialize_baseline(data_path)

    def _initialize_baseline(self, data_path: str):
        logger.info("Initializing DriftDetector baseline from training data...")
        try:
            # We only need the raw features before SMOTE, but load_and_preprocess returns scaled SMOTE data.
            # To be accurate to the incoming live telemetry, we should compare against the RAW, UNSCALED baseline.
            df = pd.read_csv(data_path)
            
            # Map columns just like preprocessing.py
            rename_map = {
                'Air temperature [K]': 'temperature_air',
                'Process temperature [K]': 'temperature_process',
                'Rotational speed [rpm]': 'speed_rpm',
                'Torque [Nm]': 'torque',
                'Tool wear [min]': 'tool_wear'
            }
            df = df.rename(columns=rename_map)
            
            # Compute histograms with 10 bins for PSI calculation
            for feature in self.feature_names:
                data = df[feature].values
                # Create 10 bins
                hist, bins = np.histogram(data, bins=10, density=False)
                # Normalize to probabilities (add small epsilon to avoid division by zero)
                hist = hist.astype(np.float64)
                hist = (hist + 1e-5) / (np.sum(hist) + 1e-5 * len(hist))
                
                self.baseline_hists[feature] = hist
                self.baseline_bins[feature] = bins
                
            logger.info("DriftDetector baseline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize DriftDetector: {e}")
            self.enabled = False

    def _calculate_psi(self, expected_hist, actual_data, bins):
        """Calculates Population Stability Index for a single feature."""
        # Bin the actual data using the expected bins
        actual_hist, _ = np.histogram(actual_data, bins=bins, density=False)
        
        # Normalize
        actual_hist = actual_hist.astype(np.float64)
        actual_hist = (actual_hist + 1e-5) / (np.sum(actual_hist) + 1e-5 * len(actual_hist))
        
        # PSI Formula: sum((Actual - Expected) * ln(Actual / Expected))
        psi = np.sum((actual_hist - expected_hist) * np.log(actual_hist / expected_hist))
        return psi

    def monitor(self, buffer_data: list) -> dict:
        """
        Takes a window of recent telemetry (list of dicts) and calculates drift.
        Returns a JSON payload if drift exceeds threshold.
        """
        if not self.enabled or len(buffer_data) < self.window_size:
            return {"drift_detected": False}
            
        try:
            # Convert list of dicts to DataFrame for easy column extraction
            df_live = pd.DataFrame(buffer_data)
            
            affected_features = []
            max_psi = 0.0
            
            # Check each feature
            for feature in self.feature_names:
                if feature not in df_live.columns:
                    return {
                        "drift_detected": True,
                        "score": 1.0,
                        "affected_features": [feature],
                        "reason": f"Missing feature anomaly: {feature}"
                    }
                    
                live_data = df_live[feature].values
                expected_hist = self.baseline_hists[feature]
                bins = self.baseline_bins[feature]
                
                psi = self._calculate_psi(expected_hist, live_data, bins)
                if psi > max_psi:
                    max_psi = psi
                    
                if psi > self.psi_threshold:
                    affected_features.append(feature)
                    
            if len(affected_features) > 0:
                logger.warning(f"[DRIFT DETECTED] PSI={max_psi:.4f} > {self.psi_threshold} on features {affected_features}")
                return {
                    "drift_detected": True,
                    "score": float(round(max_psi, 4)),
                    "affected_features": affected_features,
                    "reason": "Feature distribution drift (PSI threshold exceeded)"
                }
                
            return {"drift_detected": False}
            
        except Exception as e:
            logger.error(f"Error during drift monitoring: {e}")
            return {"drift_detected": False, "error": str(e)}
