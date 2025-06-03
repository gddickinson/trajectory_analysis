#!/usr/bin/env python3
"""
Main Application Module
=======================

Contains the main ParticleTrackingApp class that coordinates all components.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
import traceback

# PyQt imports with error handling
try:
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from PyQt6.QtCore import Qt, QTimer, QSettings
    from PyQt6.QtGui import QIcon, QPixmap
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("Warning: PyQt6 not available. GUI functionality disabled.")

# Scientific computing imports
import numpy as np
import pandas as pd

# Import our custom modules
from .core.data_manager import DataManager
from .core.analysis_engine import AnalysisEngine
from .core.project_manager import ProjectManager
from .utils.logging_config import setup_logging
from .utils.config_manager import ConfigManager

# Conditional GUI imports
if PYQT_AVAILABLE:
    from .gui.main_window import MainWindow


class ParticleTrackingApp:
    """Main application class for particle tracking analysis."""

    def __init__(self, argv=None, debug=False):
        """Initialize the application.

        Args:
            argv: Command line arguments (for QApplication)
            debug: Enable debug logging
        """
        if not PYQT_AVAILABLE:
            raise ImportError("PyQt6 is required for GUI functionality. Install with: pip install PyQt6")

        # Initialize Qt Application
        if argv is None:
            argv = sys.argv
        self.qt_app = QApplication(argv)

        # Application metadata
        self.qt_app.setApplicationName("Particle Tracking Analyzer")
        self.qt_app.setApplicationVersion("1.0.0")
        self.qt_app.setOrganizationName("Scientific Computing")

        # Initialize logging
        self.logger = setup_logging(debug=debug)
        self.logger.info("Starting Particle Tracking Application")

        # Initialize configuration
        self.config = ConfigManager()

        # Initialize core components
        self.data_manager = DataManager()
        self.analysis_engine = AnalysisEngine()
        self.project_manager = ProjectManager()

        # Main window (will be created in show())
        self.main_window = None

        # Setup application
        self._setup_application()

    def _setup_application(self):
        """Setup application-wide configurations."""
        # Set application icon
        self._set_app_icon()

        # Apply stylesheet
        self._apply_stylesheet()

        # Setup exception handling
        sys.excepthook = self._handle_exception

    def _set_app_icon(self):
        """Set application icon if available."""
        try:
            from .resources import get_icon_path
            icon_path = get_icon_path("app_icon.png")
            if icon_path.exists():
                self.qt_app.setWindowIcon(QIcon(str(icon_path)))
        except Exception:
            # Use default icon or no icon
            pass

    def _apply_stylesheet(self):
        """Apply modern stylesheet to the application."""
        style = """
        QMainWindow {
            background-color: #f5f5f5;
        }

        QTabWidget::pane {
            border: 1px solid #c0c0c0;
            background-color: white;
            border-radius: 4px;
        }

        QTabBar::tab {
            background-color: #e1e1e1;
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }

        QTabBar::tab:selected {
            background-color: white;
            border-bottom: 2px solid #0078d4;
        }

        QTabBar::tab:hover {
            background-color: #d1d1d1;
        }

        QPushButton {
            background-color: #0078d4;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }

        QPushButton:hover {
            background-color: #106ebe;
        }

        QPushButton:pressed {
            background-color: #005a9e;
        }

        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }

        QGroupBox {
            font-weight: bold;
            border: 2px solid #cccccc;
            border-radius: 5px;
            margin-top: 1ex;
            padding-top: 10px;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }

        QProgressBar {
            border: 2px solid #cccccc;
            border-radius: 5px;
            text-align: center;
            background-color: #f0f0f0;
        }

        QProgressBar::chunk {
            background-color: #0078d4;
            border-radius: 3px;
        }

        QTreeWidget, QTableView {
            alternate-background-color: #f9f9f9;
            selection-background-color: #0078d4;
            gridline-color: #d0d0d0;
        }

        QScrollBar:vertical {
            background: #f0f0f0;
            width: 12px;
            border-radius: 6px;
        }

        QScrollBar::handle:vertical {
            background: #c0c0c0;
            border-radius: 6px;
        }

        QScrollBar::handle:vertical:hover {
            background: #a0a0a0;
        }
        """
        self.qt_app.setStyleSheet(style)

    def _handle_exception(self, exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions."""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        self.logger.error(f"Uncaught exception: {error_msg}")

        # Show error dialog
        if hasattr(self, 'main_window') and self.main_window:
            QMessageBox.critical(
                self.main_window,
                "Application Error",
                f"An unexpected error occurred:\n\n{str(exc_value)}\n\nCheck the log for details."
            )
        else:
            print(f"Critical error: {exc_value}")

    def show(self):
        """Show the main window."""
        if not self.main_window:
            self.main_window = MainWindow(
                data_manager=self.data_manager,
                analysis_engine=self.analysis_engine,
                project_manager=self.project_manager,
                config=self.config
            )

        self.main_window.show()
        self.main_window.raise_()
        self.main_window.activateWindow()

    def exec(self):
        """Execute the application event loop."""
        # Show main window if not already shown
        if not self.main_window:
            self.show()

        # Run the application
        try:
            exit_code = self.qt_app.exec()
            self.logger.info("Application closed normally")
            return exit_code
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            return 1
        finally:
            self._cleanup()

    def _cleanup(self):
        """Cleanup resources on exit."""
        try:
            # Stop any running analysis
            if self.analysis_engine:
                self.analysis_engine.stop_analysis()

            # Save configuration
            if self.config:
                self.config.save_config()

            self.logger.info("Application cleanup completed")

        except Exception as e:
            print(f"Error during cleanup: {e}")

    def load_project(self, project_path: str):
        """Load a project file."""
        if self.project_manager.load_project(project_path):
            self.config.add_recent_project(project_path)
            return True
        return False

    def load_data(self, data_path: str):
        """Load a data file."""
        if self.data_manager.load_file(data_path):
            self.config.add_recent_file(data_path)
            return True
        return False


def main(argv=None):
    """Convenience function to run the application."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Particle Tracking Application")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--project", type=str, help="Project file to load")
    parser.add_argument("--data", type=str, help="Data file to load")

    if argv is None:
        argv = sys.argv

    # Only parse known args to avoid conflicts with Qt
    args, unknown = parser.parse_known_args(argv[1:])

    # Create application
    app = ParticleTrackingApp(argv=[argv[0]] + unknown, debug=args.debug)

    # Load project if specified
    if args.project and Path(args.project).exists():
        app.load_project(args.project)

    # Load data if specified
    if args.data and Path(args.data).exists():
        app.load_data(args.data)

    # Run application
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
