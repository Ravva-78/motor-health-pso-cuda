"""
tests/test_config_schema.py
───────────────────────────
Tests schema validation, bounds enforcement, and fault-tolerant fallbacks for configuration.
"""

import pytest
import os
import yaml
from backend.config import ConfigManager, config


def test_config_file_exists():
    """Ensures primary configuration file config.yaml is present."""
    assert os.path.exists("config.yaml"), "config.yaml must exist at repository root"


def test_config_schema_completeness():
    """Validates that all essential microservice configuration blocks are present."""
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    required_sections = ["pso", "model", "cuda", "mqtt", "ring_buffer", "ipc", "paths", "drift"]
    for section in required_sections:
        assert section in cfg, f"Missing required configuration section: '{section}'"


def test_pso_bounds_schema():
    """Validates that PSO hyperparameter search spaces have valid lower < upper bounds."""
    pso_cfg = config.get_pso_config()
    assert pso_cfg["n_particles"] > 0
    assert pso_cfg["max_iter"] > 0
    
    raw_pso = config.config.get("pso", {})
    bounds = raw_pso.get("bounds", {})
    assert len(bounds) > 0, "PSO bounds must be defined"
    for param, val in bounds.items():
        low, high = val[0], val[1]
        assert low < high, f"Invalid bounds for {param}: lower ({low}) must be < upper ({high})"
        assert low > 0, f"Lower bound for {param} must be positive"



def test_network_and_buffer_schemas():
    """Validates network addresses, ports, and ring buffer sizing."""
    mqtt_cfg = config.get_mqtt_config()
    assert 1 <= mqtt_cfg["port"] <= 65535, f"Invalid MQTT port: {mqtt_cfg['port']}"
    assert isinstance(mqtt_cfg["topics"], list)
    assert len(mqtt_cfg["topics"]) > 0
    
    buf_cfg = config.get_ring_buffer_config()
    assert buf_cfg["capacity"] >= 64, "RingBuffer capacity must be at least 64"
    
    ipc_cfg = config.get_ipc_config()
    assert ipc_cfg["address"].startswith("tcp://"), "IPC address must use TCP protocol"
    assert ipc_cfg["timeout_ms"] > 0


def test_drift_schema_validation():
    """Validates statistical drift detector settings."""
    drift_cfg = config.get_drift_config()
    assert isinstance(drift_cfg["enabled"], bool)
    assert drift_cfg["window_size"] >= 10
    assert 0.0 < drift_cfg["psi_threshold"] < 1.0, "PSI threshold must be between 0 and 1"
