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
    ensure_directory, get_app_data_directory, get_temp_directory,
    format_file_size, format_duration, backup_file,
    performance_monitor, ParticleTrackerError, DataLoadError,
    AnalysisError, ProjectError
)

# Import path utilities with error handling
try:
    from .path_utils import (
        find_project_root, get_default_training_data_path,
        get_resources_directory, get_example_data_directory,
        get_training_data_directory
    )
except ImportError:
    # If path_utils doesn't exist yet, create placeholder functions
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("path_utils module not found - path utilities not available")

    def find_project_root():
        from pathlib import Path
        return Path.cwd()

    def get_default_training_data_path():
        return None

    def get_resources_directory():
        return find_project_root() / "particle_tracker" / "resources"

    def get_example_data_directory():
        return get_resources_directory() / "example_data"

    def get_training_data_directory():
        return get_resources_directory() / "training_data"

__all__ = [
    # Config management
    'ConfigManager', 'ApplicationConfig',

    # Logging
    'setup_logging',

    # File utilities
    'ensure_directory', 'get_app_data_directory', 'get_temp_directory',
    'format_file_size', 'format_duration', 'backup_file', 'performance_monitor',

    # Path utilities
    'find_project_root', 'get_default_training_data_path',
    'get_resources_directory', 'get_example_data_directory', 'get_training_data_directory',

    # Exceptions
    'ParticleTrackerError', 'DataLoadError', 'AnalysisError', 'ProjectError'
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
