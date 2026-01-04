import logging
import os
from logging.handlers import RotatingFileHandler
from .config import LOGS_DIR

def setup_logger(name='ai_trader', log_file='trading.log', level=logging.INFO):
    """
    Setup a logger with rotating file handler and console handler.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times if logger is reused
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File Handler (Rotating)
    file_path = os.path.join(LOGS_DIR, log_file)
    file_handler = RotatingFileHandler(file_path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

# Default logger instance
logger = setup_logger()
