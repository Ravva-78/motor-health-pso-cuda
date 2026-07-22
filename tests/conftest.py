import os
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def dummy_csv_path(tmp_path):
    """
    Creates a temporary CSV file with synthetic motor health data 
    isolated from production.
    """
    df = pd.DataFrame({
        'UDI': range(1, 11),
        'Product ID': [f'M{i:04d}' for i in range(1, 11)],
        'Type': ['M', 'L', 'L', 'M', 'H', 'L', 'L', 'M', 'H', 'M'],
        'Air temperature [K]': np.random.uniform(295.0, 304.0, 10),
        'Process temperature [K]': np.random.uniform(305.0, 314.0, 10),
        'Rotational speed [rpm]': np.random.uniform(1100, 2900, 10),
        'Torque [Nm]': np.random.uniform(3.0, 80.0, 10),
        'Tool wear [min]': np.random.uniform(0, 250, 10),
        'Target': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        'Failure Type': [
            'No Failure', 'No Failure', 'Power Failure', 'No Failure', 
            'No Failure', 'No Failure', 'No Failure', 'Tool Wear Failure', 
            'No Failure', 'No Failure'
        ]
    })
    
    file_path = tmp_path / "dummy_data.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)

@pytest.fixture
def dummy_config_path(tmp_path):
    import yaml
    cfg = {
        'model': {'cv_folds': 3},
        'pso': {'n_particles': 5, 'max_iter': 5}
    }
    file_path = tmp_path / "dummy_config.yaml"
    with open(file_path, "w") as f:
        yaml.dump(cfg, f)
    return str(file_path)
