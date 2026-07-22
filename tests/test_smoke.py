"""
Smoke tests to verify the overall integrity of the application components.
These tests run fast and prove everything imports and initializes safely.
"""

def test_backend_imports():
    """Ensure all backend modules can be imported without cyclical dependencies or missing packages."""
    try:
        import backend.constants
        import backend.config
        import backend.logger
        import backend.preprocessing
        import backend.model
        import backend.pso_optimizer
        import backend.cuda_module
    except Exception as e:
        import pytest
        pytest.fail(f"Backend imports failed: {e}")

def test_config_instantiation():
    """Ensure the global config object is created successfully."""
    from backend.config import config
    paths = config.get_paths()
    assert isinstance(paths, dict)
    assert 'data' in paths

def test_logger_instantiation():
    """Ensure the global logger initializes."""
    from backend.logger import get_logger
    logger = get_logger("smoke_test")
    logger.info("Smoke test log entry")
    assert logger.name == "smoke_test"
