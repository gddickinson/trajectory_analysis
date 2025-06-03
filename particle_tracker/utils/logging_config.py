#!/usr/bin/env python3
"""
Utility Modules
===============

Logging utilities.
"""

import os
import json
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import pickle
import zipfile

from PyQt6.QtCore import QObject, pyqtSignal, QSettings


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging(debug: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration for the application.

    Args:
        debug: Enable debug level logging
        log_file: Path to log file (optional)

    Returns:
        Configured logger instance
    """

    # Create logs directory
    log_dir = Path.home() / ".particle_tracker" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Determine log level
    level = logging.DEBUG if debug else logging.INFO

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"particle_tracker_{timestamp}.log"

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Application logger
    app_logger = logging.getLogger('particle_tracker')
    app_logger.info("Logging system initialized")

    return app_logger

