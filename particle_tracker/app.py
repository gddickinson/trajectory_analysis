#!/usr/bin/env python3
"""
Enhanced Main Application Module
================================

Enhanced particle tracking application with comprehensive analysis capabilities
including multi-radius density analysis, direction autocorrelation, advanced
shape metrics, scaled radius of gyration, and batch processing support.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
import traceback
import numpy as np
import pandas as pd

# PyQt imports with error handling
try:
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from PyQt6.QtCore import Qt, QTimer, QSettings
    from PyQt6.QtGui import QIcon, QPixmap
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("Warning: PyQt6 not available. GUI functionality disabled.")

# Import our enhanced modules
from .core.data_manager import EnhancedDataManager
from .core.analysis_engine import AnalysisEngine  # Use the enhanced one from core
from .core.project_manager import EnhancedProjectManager
from .core.batch_analysis import BatchAnalysisManager
from .utils.logging_config import setup_logging
from .utils.config_manager import ConfigManager

# Conditional GUI imports
if PYQT_AVAILABLE:
    from .gui.main_window import RedesignedEnhancedMainWindow


class EnhancedParticleTrackingApp:
    """Enhanced main application class for comprehensive particle tracking analysis."""

    def __init__(self, argv=None, debug=False):
        """Initialize the enhanced application.

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
        self.qt_app.setApplicationName("Enhanced Particle Tracking Analyzer")
        self.qt_app.setApplicationVersion("2.0.0")
        self.qt_app.setOrganizationName("Scientific Computing Lab")

        # Initialize logging
        self.logger = setup_logging(debug=debug)
        self.logger.info("Starting Enhanced Particle Tracking Application")

        # Initialize configuration
        self.config = ConfigManager()

        # Initialize core components
        self.data_manager = EnhancedDataManager()
        self.analysis_engine = AnalysisEngine()
        self.project_manager = EnhancedProjectManager()
        self.batch_manager = BatchAnalysisManager(self.analysis_engine, self.data_manager)

        # Main window (will be created in show())
        self.main_window = None

        # Setup application
        self._setup_application()

    def _setup_application(self):
        """Setup application-wide configurations."""
        # Set application icon
        self._set_app_icon()

        # Apply enhanced stylesheet
        self._apply_enhanced_stylesheet()

        # Setup exception handling
        sys.excepthook = self._handle_exception

        # Connect component signals
        self._connect_component_signals()

    def _connect_component_signals(self):
        """Connect signals between application components."""
        # Data manager to analysis engine
        self.data_manager.dataLoaded.connect(lambda data_name, data: self.analysis_engine.analysisStarted.emit([]))

        # Batch manager signals
        self.batch_manager.batchStarted.connect(self._on_batch_started)
        self.batch_manager.batchCompleted.connect(self._on_batch_completed)
        self.batch_manager.errorOccurred.connect(self._on_batch_error)

        # Set analysis engine reference in batch manager
        self.batch_manager.analysis_engine = self.analysis_engine

    def _set_app_icon(self):
        """Set application icon if available."""
        try:
            # Look for icon in resources
            icon_path = Path(__file__).parent / "resources" / "icons" / "app_icon.png"
            if icon_path.exists():
                self.qt_app.setWindowIcon(QIcon(str(icon_path)))
        except Exception:
            # Use default icon or no icon
            pass

    def _apply_enhanced_stylesheet(self):
        """Apply enhanced modern stylesheet to the application."""
        style = """
        QMainWindow {
            background-color: #f8f9fa;
            color: #212529;
        }

        /* Enhanced Tab Widget Styling */
        QTabWidget::pane {
            border: 1px solid #dee2e6;
            background-color: white;
            border-radius: 6px;
            margin-top: 2px;
        }

        QTabBar::tab {
            background-color: #e9ecef;
            padding: 10px 20px;
            margin-right: 2px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            font-weight: 500;
            color: #495057;
        }

        QTabBar::tab:selected {
            background-color: white;
            border-bottom: 3px solid #0d6efd;
            color: #0d6efd;
            font-weight: 600;
        }

        QTabBar::tab:hover:!selected {
            background-color: #f8f9fa;
            color: #0d6efd;
        }

        /* Enhanced Button Styling */
        QPushButton {
            background-color: #0d6efd;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 14px;
            min-height: 20px;
        }

        QPushButton:hover {
            background-color: #0b5ed7;
        }

        QPushButton:pressed {
            background-color: #0a58ca;
        }

        QPushButton:disabled {
            background-color: #adb5bd;
            color: #6c757d;
        }

        /* Enhanced Group Box */
        QGroupBox {
            font-weight: 600;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            margin-top: 1ex;
            padding-top: 15px;
            background-color: white;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 15px;
            padding: 0 10px 0 10px;
            background-color: white;
            color: #495057;
        }

        /* Enhanced Progress Bar */
        QProgressBar {
            border: 2px solid #dee2e6;
            border-radius: 8px;
            text-align: center;
            background-color: #f8f9fa;
            font-weight: 600;
        }

        QProgressBar::chunk {
            background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                      stop: 0 #0d6efd, stop: 1 #6610f2);
            border-radius: 6px;
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
        """Show the enhanced main window."""
        if not self.main_window:
            self.main_window = RedesignedEnhancedMainWindow(
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

            # Stop batch processing
            if self.batch_manager:
                self.batch_manager.stop_current_analysis()

            # Save configuration
            if self.config:
                self.config.save_config()

            self.logger.info("Application cleanup completed")

        except Exception as e:
            print(f"Error during cleanup: {e}")

    # Batch processing event handlers
    def _on_batch_started(self, experiment_name: str):
        """Handle batch processing start."""
        self.logger.info(f"Batch processing started for experiment: {experiment_name}")

    def _on_batch_completed(self, experiment_name: str):
        """Handle batch processing completion."""
        self.logger.info(f"Batch processing completed for experiment: {experiment_name}")

    def _on_batch_error(self, context: str, error_message: str):
        """Handle batch processing errors."""
        self.logger.error(f"Batch processing error in {context}: {error_message}")


# Legacy compatibility
ParticleTrackingApp = EnhancedParticleTrackingApp


def main(argv=None):
    """Convenience function to run the enhanced application."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Enhanced Particle Tracking Application")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--project", type=str, help="Project file to load")
    parser.add_argument("--data", type=str, help="Data file to load")

    if argv is None:
        argv = sys.argv

    # Only parse known args to avoid conflicts with Qt
    args, unknown = parser.parse_known_args(argv[1:])

    # Create enhanced application
    app = EnhancedParticleTrackingApp(argv=[argv[0]] + unknown, debug=args.debug)

    # Load project if specified
    if args.project and Path(args.project).exists():
        # Load project functionality would go here
        pass

    # Load data if specified
    if args.data and Path(args.data).exists():
        # Load data functionality would go here
        pass

    # Run application
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
