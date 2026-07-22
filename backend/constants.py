"""
Project-wide constants to prevent magic strings and redundancy.
"""

# Feature definitions
FEATURE_COLS = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
]

FEATURE_NAMES = [
    'temperature_air', 
    'temperature_process',
    'speed_rpm', 
    'torque', 
    'tool_wear'
]

# Label definitions
LABEL_MAP_DICT = {0: 'Normal', 1: 'Warning', 2: 'Critical'}
LABEL_NAMES_LIST = ['Normal', 'Warning', 'Critical']

# PSO Default Bounds (used if config is missing)
PSO_DEFAULT_BOUNDS = {
    'n_estimators':      (10,  300),
    'max_depth':         (3,   25),
    'min_samples_split': (2,   15),
}
