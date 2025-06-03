#!/usr/bin/env python3
"""
Utility Modules
===============

Configuration management
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
# CONFIGURATION MANAGER
# ============================================================================

@dataclass
class ApplicationConfig:
    """Configuration settings for the application."""

    # File paths
    last_data_directory: str = ""
    last_project_directory: str = ""
    recent_files: List[str] = None
    recent_projects: List[str] = None

    # Analysis defaults
    default_pixel_size: float = 108.0
    default_frame_rate: float = 10.0
    default_detection_method: str = "threshold"
    default_linking_method: str = "trackpy"  # Updated default

    # SVM Training data
    default_svm_training_data: str = ""  # Will be auto-populated
    svm_auto_detect: bool = True  # Whether to auto-detect training data on startup

    # UI settings
    window_geometry: str = ""
    window_state: str = ""
    theme: str = "default"
    show_advanced_options: bool = False

    # Performance settings
    max_memory_usage_mb: int = 2048
    num_threads: int = 4
    enable_gpu: bool = False

    def __post_init__(self):
        if self.recent_files is None:
            self.recent_files = []
        if self.recent_projects is None:
            self.recent_projects = []



class ConfigManager(QObject):
    """Manages application configuration."""

    configChanged = pyqtSignal(str, object)  # key, value

    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(__name__)

        # Configuration file path
        self.config_dir = Path.home() / ".particle_tracker"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / "config.json"

        # Load configuration
        self.config = self._load_config()

        self.logger.info("Configuration manager initialized")

    def _load_config(self) -> ApplicationConfig:
        """Load configuration from file with auto-detection of training data."""

        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_dict = json.load(f)

                # Create config object
                config = ApplicationConfig(**config_dict)
                self.logger.info(f"Configuration loaded from {self.config_file}")

            except Exception as e:
                self.logger.warning(f"Error loading config: {e}, using defaults")
                config = ApplicationConfig()
        else:
            # Return default configuration
            config = ApplicationConfig()

        # Auto-detect training data if enabled and not already set
        if config.svm_auto_detect and not config.default_svm_training_data:
            try:
                from particle_tracker.utils.path_utils import get_default_training_data_path
                default_path = get_default_training_data_path()
                if default_path:
                    config.default_svm_training_data = default_path
                    self.logger.info(f"Auto-detected SVM training data: {default_path}")
                    # Save the updated config
                    self.config = config
                    self.save_config()
            except Exception as e:
                self.logger.debug(f"Could not auto-detect training data: {e}")

        return config


    def save_config(self):
        """Save current configuration to file."""

        try:
            config_dict = asdict(self.config)

            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)

            self.logger.info(f"Configuration saved to {self.config_file}")

        except Exception as e:
            self.logger.error(f"Error saving config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return getattr(self.config, key, default)

    def set(self, key: str, value: Any):
        """Set configuration value."""
        if hasattr(self.config, key):
            setattr(self.config, key, value)
            self.configChanged.emit(key, value)
            self.save_config()
        else:
            self.logger.warning(f"Unknown configuration key: {key}")

    def add_recent_file(self, file_path: str):
        """Add file to recent files list."""
        file_path = str(Path(file_path).absolute())

        # Remove if already exists
        if file_path in self.config.recent_files:
            self.config.recent_files.remove(file_path)

        # Add to beginning
        self.config.recent_files.insert(0, file_path)

        # Keep only last 10
        self.config.recent_files = self.config.recent_files[:10]

        self.save_config()

    def add_recent_project(self, project_path: str):
        """Add project to recent projects list."""
        project_path = str(Path(project_path).absolute())

        # Remove if already exists
        if project_path in self.config.recent_projects:
            self.config.recent_projects.remove(project_path)

        # Add to beginning
        self.config.recent_projects.insert(0, project_path)

        # Keep only last 10
        self.config.recent_projects = self.config.recent_projects[:10]

        self.save_config()

    def get_recent_files(self) -> List[str]:
        """Get list of recent files (existing only)."""
        return [f for f in self.config.recent_files if Path(f).exists()]

    def get_recent_projects(self) -> List[str]:
        """Get list of recent projects (existing only)."""
        return [p for p in self.config.recent_projects if Path(p).exists()]

    def get_default_svm_training_data(self) -> str:
        """Get the default SVM training data path."""
        return self.config.default_svm_training_data

    def set_default_svm_training_data(self, path: str):
        """Set the default SVM training data path."""
        self.config.default_svm_training_data = path
        self.save_config()
        self.configChanged.emit('default_svm_training_data', path)

    def refresh_training_data_path(self):
        """Refresh the training data path by re-running auto-detection."""
        try:
            from particle_tracker.utils.path_utils import get_default_training_data_path
            new_path = get_default_training_data_path()
            if new_path and new_path != self.config.default_svm_training_data:
                self.set_default_svm_training_data(new_path)
                self.logger.info(f"Updated SVM training data path: {new_path}")
                return new_path
        except Exception as e:
            self.logger.warning(f"Error refreshing training data path: {e}")
        return self.config.default_svm_training_data
