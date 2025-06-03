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
    default_linking_method: str = "nearest_neighbor"

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
        """Load configuration from file."""

        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_dict = json.load(f)

                # Create config object
                config = ApplicationConfig(**config_dict)
                self.logger.info(f"Configuration loaded from {self.config_file}")
                return config

            except Exception as e:
                self.logger.warning(f"Error loading config: {e}, using defaults")

        # Return default configuration
        return ApplicationConfig()

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

