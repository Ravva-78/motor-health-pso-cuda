import logging
from backend.logger import get_logger

def test_get_logger_creation():
    logger = get_logger("test_logger_1")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger_1"
    
    # Assert handlers were added (console + file fallback)
    assert len(logger.handlers) >= 1
    
    # Assert singleton-like behavior (calling again doesn't duplicate handlers)
    handler_count = len(logger.handlers)
    logger_again = get_logger("test_logger_1")
    assert len(logger_again.handlers) == handler_count
