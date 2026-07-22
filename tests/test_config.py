import os
from backend.config import ConfigManager

def test_config_manager_defaults(tmp_path):
    # Pass a path that doesn't exist
    missing_path = str(tmp_path / "does_not_exist.yaml")
    config = ConfigManager(missing_path)
    
    # Assert sensible defaults are returned
    model_cfg = config.get_model_config()
    assert model_cfg['cv_folds'] == 5
    assert model_cfg['random_state'] == 42
    
    pso_cfg = config.get_pso_config()
    assert pso_cfg['n_particles'] == 20

def test_config_manager_loading(dummy_config_path):
    config = ConfigManager(dummy_config_path)
    
    model_cfg = config.get_model_config()
    assert model_cfg['cv_folds'] == 3
    
    pso_cfg = config.get_pso_config()
    assert pso_cfg['n_particles'] == 5

def test_config_manager_env_override(monkeypatch):
    monkeypatch.setenv("MODEL_CV_FOLDS", "10")
    monkeypatch.setenv("PSO_PARTICLES", "50")
    
    config = ConfigManager("nonexistent_so_defaults_used.yaml")
    
    model_cfg = config.get_model_config()
    assert model_cfg['cv_folds'] == 10
    
    pso_cfg = config.get_pso_config()
    assert pso_cfg['n_particles'] == 50
