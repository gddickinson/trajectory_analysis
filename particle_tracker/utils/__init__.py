"""
Utilities Package
================

Contains utility functions and helper classes:
- ConfigManager: Application configuration management
- LoggingConfig: Logging setup and configuration
- FileUtils: File handling utilities
- PerformanceMonitor: Performance monitoring tools
"""

from .config_manager import ConfigManager, ApplicationConfig
from .logging_config import setup_logging
from .file_utils import (
    ensure_directory,
    get_app_data_directory,
    get_temp_directory,
    format_file_size,
    format_duration,
    backup_file,
    PerformanceMonitor,
    performance_monitor
)

# Import custom exceptions
from .file_utils import (
    ParticleTrackerError,
    DataLoadError,
    AnalysisError,
    ProjectError
)

__all__ = [
    # Configuration
    "ConfigManager",
    "ApplicationConfig",

    # Logging
    "setup_logging",

    # File utilities
    "ensure_directory",
    "get_app_data_directory",
    "get_temp_directory",
    "format_file_size",
    "format_duration",
    "backup_file",
    "PerformanceMonitor",
    "performance_monitor",

    # Exceptions
    "ParticleTrackerError",
    "DataLoadError",
    "AnalysisError",
    "ProjectError",
]

# Utils package metadata
__utils_version__ = "1.0.0"

def get_utils_info():
    """Get information about utility components."""
    return {
        "version": __utils_version__,
        "components": {
            "ConfigManager": "Application configuration management",
            "LoggingConfig": "Logging setup and configuration",
            "FileUtils": "File handling utilities",
            "PerformanceMonitor": "Performance monitoring tools"
        }
    }

def get_system_info():
    """Get system information for debugging."""
    import sys
    import platform

    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture(),
        "processor": platform.processor(),
    }

    # Add memory info if available
    try:
        import psutil
        memory = psutil.virtual_memory()
        info["memory_total_gb"] = memory.total / (1024**3)
        info["memory_available_gb"] = memory.available / (1024**3)
    except ImportError:
        info["memory_info"] = "psutil not available"

    return info