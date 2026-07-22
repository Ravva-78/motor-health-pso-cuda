from backend.constants import FEATURE_COLS, FEATURE_NAMES, LABEL_MAP_DICT, LABEL_NAMES_LIST, PSO_DEFAULT_BOUNDS

def test_constants_feature_mapping():
    assert len(FEATURE_COLS) == len(FEATURE_NAMES)
    assert 'Air temperature [K]' in FEATURE_COLS
    assert 'temperature_air' in FEATURE_NAMES

def test_constants_labels():
    assert 0 in LABEL_MAP_DICT
    assert LABEL_MAP_DICT[0] == 'Normal'
    assert 'Normal' in LABEL_NAMES_LIST

def test_constants_pso_bounds():
    assert 'n_estimators' in PSO_DEFAULT_BOUNDS
    assert isinstance(PSO_DEFAULT_BOUNDS['n_estimators'], tuple)
