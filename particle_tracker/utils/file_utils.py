#!/usr/bin/env python3
"""
Utility Modules
===============

File utilities.
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
# UTILITY FUNCTIONS
# ============================================================================

def ensure_directory(path: str) -> Path:
    """Ensure directory exists, create if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_app_data_directory() -> Path:
    """Get application data directory."""
    return ensure_directory(Path.home() / ".particle_tracker")


def get_temp_directory() -> Path:
    """Get temporary directory for the application."""
    return ensure_directory(get_app_data_directory() / "temp")


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration in human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def backup_file(file_path: str, backup_dir: Optional[str] = None) -> str:
    """Create a backup of a file."""
    file_path = Path(file_path)

    if backup_dir is None:
        backup_dir = get_app_data_directory() / "backups"
    else:
        backup_dir = Path(backup_dir)

    ensure_directory(backup_dir)

    # Create backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
    backup_path = backup_dir / backup_name

    # Copy file
    import shutil
    shutil.copy2(file_path, backup_path)

    return str(backup_path)


class PerformanceMonitor:
    """Monitor application performance."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.start_time = None
        self.memory_usage = {}

    def start_timing(self, operation: str):
        """Start timing an operation."""
        self.start_time = time.time()
        self.logger.debug(f"Started timing: {operation}")

    def end_timing(self, operation: str):
        """End timing an operation."""
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.logger.info(f"{operation} completed in {format_duration(duration)}")
            self.start_time = None
            return duration
        return 0

    def log_memory_usage(self, label: str):
        """Log current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_usage[label] = memory_mb
            self.logger.debug(f"Memory usage ({label}): {memory_mb:.1f} MB")
        except ImportError:
            # psutil not available
            pass

    def get_memory_report(self) -> str:
        """Get memory usage report."""
        if not self.memory_usage:
            return "No memory data available"

        lines = ["Memory Usage Report:", "-" * 20]
        for label, usage in self.memory_usage.items():
            lines.append(f"{label}: {usage:.1f} MB")

        return "\n".join(lines)


# Initialize performance monitor
performance_monitor = PerformanceMonitor()


# ============================================================================
# ERROR HANDLING
# ============================================================================

class ParticleTrackerError(Exception):
    """Base exception for particle tracker application."""
    pass


class DataLoadError(ParticleTrackerError):
    """Error loading data."""
    pass


class AnalysisError(ParticleTrackerError):
    """Error during analysis."""
    pass


class ProjectError(ParticleTrackerError):
    """Error with project operations."""
    pass


def handle_exception(exc_type, exc_value, exc_traceback):
    """Global exception handler."""
    logger = logging.getLogger(__name__)

    if issubclass(exc_type, KeyboardInterrupt):
        # Allow keyboard interrupt to pass through
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # Log the exception
    logger.error(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback)
    )


# Set global exception handler
import sys
import time
sys.excepthook = handle_exception
