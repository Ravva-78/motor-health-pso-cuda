import pytest
import numpy as np
import pandas as pd
from backend.drift_detector import DriftDetector
from backend.config import config

def test_drift_detector_initialization():
    detector = DriftDetector("data/predictive_maintenance.csv")
    assert detector.enabled == config.get_drift_config()['enabled']
    # If file exists, histograms should be populated
    if detector.enabled:
        assert len(detector.baseline_hists) == 5
        assert len(detector.baseline_bins) == 5

def test_drift_detector_no_drift():
    detector = DriftDetector("data/predictive_maintenance.csv")
    if not detector.enabled:
        pytest.skip("DriftDetector disabled or missing data")
        
    # Generate a sample matching the expected baseline somewhat
    # Instead of perfectly matching, we just use the original data as the buffer
    df = pd.read_csv("data/predictive_maintenance.csv")
    rename_map = {
        'Air temperature [K]': 'temperature_air',
        'Process temperature [K]': 'temperature_process',
        'Rotational speed [rpm]': 'speed_rpm',
        'Torque [Nm]': 'torque',
        'Tool wear [min]': 'tool_wear'
    }
    df = df.rename(columns=rename_map)
    
    # Sample randomly so the distribution matches the overall population
    buffer_data = df.sample(n=detector.window_size, random_state=42).to_dict('records')
    
    report = detector.monitor(buffer_data)
    # The first N rows of the dataset should not drift too significantly 
    # compared to the whole dataset.
    assert report.get("drift_detected") is False

def test_drift_detector_with_drift():
    detector = DriftDetector("data/predictive_maintenance.csv")
    if not detector.enabled:
        pytest.skip("DriftDetector disabled or missing data")
        
    # Generate severely shifted data
    shifted_data = []
    for _ in range(detector.window_size):
        shifted_data.append({
            'temperature_air': 9999.0, # Complete outlier
            'temperature_process': 9999.0,
            'speed_rpm': 0.0,
            'torque': 0.0,
            'tool_wear': 0.0
        })
        
    report = detector.monitor(shifted_data)
    assert report.get("drift_detected") is True
    assert "score" in report
    assert "affected_features" in report
    assert len(report["affected_features"]) > 0

def test_drift_detector_missing_features():
    detector = DriftDetector("data/predictive_maintenance.csv")
    if not detector.enabled:
        pytest.skip("DriftDetector disabled or missing data")
        
    # Missing columns entirely
    bad_data = [{'invalid_feature': 1.0} for _ in range(detector.window_size)]
    report = detector.monitor(bad_data)
    assert report.get("drift_detected") is True
    assert "Missing feature" in report.get("reason")
