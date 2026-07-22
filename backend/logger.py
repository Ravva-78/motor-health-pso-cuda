import logging
import os
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Centralized logging module.
    Outputs to both console and a log file with structured formatting.
    """
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers if logger is already configured
    if logger.handlers:
        return logger

    # Ensure results directory exists for log file
    os.makedirs('results', exist_ok=True)
    
    # Configure logging level from environment (default: INFO)
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    formatter = logging.Formatter(
        '%(asctime)s  %(levelname)-8s  %(name)s — %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File Handler
    try:
        fh = logging.FileHandler('results/app.log', mode='a')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except IOError as e:
        # Fallback if file writing fails
        ch.error(f"Failed to initialize file logger: {e}")

    return logger
