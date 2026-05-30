"""
Logging utilities for unified testing pipeline
"""

import logging
import os
import sys
from datetime import datetime


def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Setup logger with file and console handlers

    Args:
        name: logger name
        log_file: log file path (optional)
        level: logging level

    Returns:
        logger: configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # clear existing handlers
    logger.handlers = []

    # formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # file handler (if log_file provided)
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_timestamp():
    """
    Get current timestamp string

    Returns:
        timestamp: formatted timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")
