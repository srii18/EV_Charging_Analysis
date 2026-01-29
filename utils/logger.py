"""
Logging utilities for ChargeSmart India
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str = "ChargeSmart", log_level: str = "INFO") -> logging.Logger:
    """
    Set up a logger with both file and console handlers
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"chargesmart_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def log_data_info(logger: logging.Logger, df, dataset_name: str):
    """
    Log basic information about a dataset
    
    Args:
        logger: Logger instance
        df: DataFrame to log info about
        dataset_name: Name of the dataset
    """
    logger.info(f"=== {dataset_name} Dataset Info ===")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Missing values: {df.isnull().sum().sum()}")
    
    if 'State' in df.columns:
        logger.info(f"Unique states: {df['State'].nunique()}")
    
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info("=" * 50)
