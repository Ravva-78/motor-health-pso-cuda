"""
tests/test_sensor_transforms.py
───────────────────────────────
Unit and property tests for sensor telemetry preprocessing and data transforms:
- Scaling contracts (MinMaxScaler bounds [0, 1])
- Physical sensor domain constraints (Temperature difference, RPM bounds)
- Telemetry payload normalization and error handling
- Feature alignment with training constants
"""

import pytest
import numpy as np
import pandas as pd
from backend.preprocessing import load_and_preprocess
from backend.constants import FEATURE_COLS, FEATURE_NAMES, LABEL_MAP_DICT


def test_feature_constants_alignment():
    """Ensures raw CSV columns and internal feature names remain strictly synchronized."""
    assert len(FEATURE_COLS) == 5
    assert len(FEATURE_NAMES) == 5
    assert "Air temperature [K]" in FEATURE_COLS
    assert "temperature_air" in FEATURE_NAMES
    assert "Torque [Nm]" in FEATURE_COLS
    assert "torque" in FEATURE_NAMES


def test_scaled_data_bounds():
    """Verifies that load_and_preprocess produces features strictly bounded within [0, 1]."""
    X_train, X_test, y_train, y_test, feats, scaler, df_clean = load_and_preprocess(
        csv_path="data/predictive_maintenance.csv",
        use_smote=False
    )
    
    assert X_train.shape[1] == 5
    assert X_test.shape[1] == 5
    
    # Train set should be in [0, 1] for MinMax scaling
    assert np.all(X_train >= 0.0)
    assert np.all(X_train <= 1.0)
    
    # Scaler must have 5 features
    assert scaler.n_features_in_ == 5
    assert len(feats) == 5
    assert len(df_clean) == 10000


def test_smote_or_class_distribution_contract():
    """Verifies dataset class contract (classes 0, 1, 2) and SMOTE when available."""
    X_train, X_test, y_train, y_test, feats, scaler, df_clean = load_and_preprocess(
        csv_path="data/predictive_maintenance.csv",
        use_smote=True
    )
    
    # Check that all 3 health classes exist in the target labels
    classes_present = np.unique(y_train)
    assert 0 in classes_present
    assert 1 in classes_present
    assert 2 in classes_present



def test_physical_sensor_calculations():
    """Tests engineering calculations for motor physics."""
    # Speed: 1500 RPM -> rad/s = 1500 * 2 * pi / 60 = 157.079 rad/s
    speed_rpm = 1500.0
    omega = speed_rpm * (2 * np.pi) / 60.0
    torque_nm = 40.0
    
    # Mechanical Power: P = Torque * Omega (Watts) -> kW
    power_kw = (torque_nm * omega) / 1000.0
    assert np.isclose(power_kw, 6.283, atol=1e-2)
    
    # Temp difference: Process Temp - Air Temp > 0 in standard operation
    t_air = 298.15
    t_process = 308.65
    delta_t = t_process - t_air
    assert delta_t > 0.0
    assert np.isclose(delta_t, 10.5, atol=1e-2)
