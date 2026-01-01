"""
Logging Configuration

Provides structured logging setup for the application.
Currently configured for console output, but can be extended for file logging.
"""

import logging
import sys
from pathlib import Path

# Create logs directory if it doesn't exist
logs_dir = Path(__file__).parent.parent.parent / "logs"
logs_dir.mkdir(exist_ok=True)


def setup_logger(name: str = "ship_rag_ai", level: int = logging.INFO) -> logging.Logger:
    """
    Set up and configure application logger.
    
    Args:
        name: Logger name
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger


# Create default logger instance
logger = setup_logger()

