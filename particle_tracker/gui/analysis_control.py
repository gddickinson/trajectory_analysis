#!/usr/bin/env python3
"""
Additional GUI Components
========================

analysis control
"""

import logging
from typing import Optional, Dict, List, Any
from pathlib import Path

import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QTableView, QAbstractItemView, QPushButton, QLabel, QGroupBox,
    QProgressBar, QTextEdit, QComboBox, QCheckBox, QSplitter,
    QHeaderView, QMenu, QMessageBox, QFileDialog, QFrame
)
from PyQt6.QtCore import (
    Qt, QAbstractTableModel, QModelIndex, QVariant, pyqtSignal,
    QSortFilterProxyModel, QTimer
)
from PyQt6.QtGui import QFont, QAction, QIcon, QStandardItemModel, QStandardItem

from particle_tracker.core.data_manager import DataManager, DataType

# Import AnalysisEngine and AnalysisStep with error handling
try:
    from particle_tracker.core.analysis_engine import AnalysisEngine, AnalysisStep
except ImportError:
    # Create fallback classes
    from enum import Enum

    class AnalysisStep(Enum):
        """Enumeration of analysis steps."""
        DETECTION = "detection"
        LINKING = "linking"
        FEATURES = "features"
        CLASSIFICATION = "classification"
        NEAREST_NEIGHBORS = "nearest_neighbors"
        DIFFUSION = "diffusion"
        VELOCITY = "velocity"

    # AnalysisEngine will be passed in as a parameter, so we don't need to define it here


class AnalysisControlWidget(QWidget):
    """Widget for controlling analysis operations."""

    def __init__(self, analysis_engine, data_manager: DataManager, parameter_manager=None, parent=None):
        super().__init__(parent)

        self.logger = logging.getLogger(__name__)
        self.analysis_engine = analysis_engine
        self.data_manager = data_manager
        self.parameter_manager = parameter_manager  # Add parameter manager reference

        self._setup_ui()
        self._connect_signals()

    def set_parameter_manager(self, parameter_manager):
        """Set the parameter manager reference."""
        self.parameter_manager = parameter_manager

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)

        # Data selection
        data_group = QGroupBox("Input Data")
        data_layout = QVBoxLayout(data_group)

        self.data_combo = QComboBox()
        self.data_combo.setPlaceholderText("Select data...")
        data_layout.addWidget(QLabel("Data:"))
        data_layout.addWidget(self.data_combo)

        layout.addWidget(data_group)

        # Analysis steps
        steps_group = QGroupBox("Analysis Steps")
        steps_layout = QVBoxLayout(steps_group)

        self.detection_cb = QCheckBox("Particle Detection")
        self.detection_cb.setChecked(True)
        steps_layout.addWidget(self.detection_cb)

        self.linking_cb = QCheckBox("Trajectory Linking")
        self.linking_cb.setChecked(True)
        steps_layout.addWidget(self.linking_cb)

        self.features_cb = QCheckBox("Feature Calculation")
        self.features_cb.setChecked(True)
        steps_layout.addWidget(self.features_cb)

        self.classification_cb = QCheckBox("Classification")
        self.classification_cb.setChecked(True)
        steps_layout.addWidget(self.classification_cb)

        self.nn_cb = QCheckBox("Nearest Neighbors")
        self.nn_cb.setChecked(False)
        steps_layout.addWidget(self.nn_cb)

        self.diffusion_cb = QCheckBox("Diffusion Analysis")
        self.diffusion_cb.setChecked(False)
        steps_layout.addWidget(self.diffusion_cb)

        self.velocity_cb = QCheckBox("Velocity Analysis")
        self.velocity_cb.setChecked(False)
        steps_layout.addWidget(self.velocity_cb)

        layout.addWidget(steps_group)

        # Control buttons
        button_layout = QVBoxLayout()

        self.run_button = QPushButton("Run Analysis")
        self.run_button.clicked.connect(self._run_analysis)
        button_layout.addWidget(self.run_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop_analysis)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        # button to trigger suggestions
        self.suggest_button = QPushButton("Suggest Parameters")
        self.suggest_button.clicked.connect(self.suggest_parameter_improvements)
        button_layout.addWidget(self.suggest_button)

        layout.addLayout(button_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
        layout.addWidget(self.status_label)

        layout.addStretch()

    def _connect_signals(self):
        """Connect signals."""
        self.data_manager.dataLoaded.connect(self._update_data_list)
        self.data_manager.dataRemoved.connect(self._update_data_list)

        # Connect analysis engine signals if available
        if hasattr(self.analysis_engine, 'analysisStarted'):
            self.analysis_engine.analysisStarted.connect(self._on_analysis_started)
        if hasattr(self.analysis_engine, 'analysisCompleted'):
            self.analysis_engine.analysisCompleted.connect(self._on_analysis_completed)
        if hasattr(self.analysis_engine, 'progressUpdate'):
            self.analysis_engine.progressUpdate.connect(self._update_progress)
        if hasattr(self.analysis_engine, 'errorOccurred'):
            self.analysis_engine.errorOccurred.connect(self._on_analysis_error)

    def _update_data_list(self):
        """Update the data selection combo box."""
        current_text = self.data_combo.currentText()

        self.data_combo.clear()
        data_names = self.data_manager.get_data_names()
        self.data_combo.addItems(data_names)

        # Restore selection if possible
        if current_text in data_names:
            self.data_combo.setCurrentText(current_text)

    def _run_analysis(self):
        """Run the selected analysis steps with smart parameter optimization."""
        # Get selected data
        data_name = self.data_combo.currentText()
        if not data_name:
            QMessageBox.warning(self, "Warning", "Please select input data")
            return

        data = self.data_manager.get_data(data_name)
        if data is None:
            QMessageBox.warning(self, "Warning", "Selected data not found")
            return

        # Get selected steps
        steps = []
        if self.detection_cb.isChecked():
            steps.append(AnalysisStep.DETECTION)
        if self.linking_cb.isChecked():
            steps.append(AnalysisStep.LINKING)
        if self.features_cb.isChecked():
            steps.append(AnalysisStep.FEATURES)
        if self.classification_cb.isChecked():
            steps.append(AnalysisStep.CLASSIFICATION)
        if self.nn_cb.isChecked():
            steps.append(AnalysisStep.NEAREST_NEIGHBORS)
        if self.diffusion_cb.isChecked():
            steps.append(AnalysisStep.DIFFUSION)
        if self.velocity_cb.isChecked():
            steps.append(AnalysisStep.VELOCITY)

        if not steps:
            QMessageBox.warning(self, "Warning", "Please select at least one analysis step")
            return

        # Get parameters from parameter manager
        if self.parameter_manager is not None:
            try:
                parameters = self.parameter_manager.get_all_parameters()

                # Apply smart parameter optimization
                parameters = self._optimize_parameters(parameters, data, steps)

                self.logger.info("Using optimized parameters from parameter manager")
            except Exception as e:
                self.logger.warning(f"Error getting parameters from manager: {e}, using defaults")
                parameters = self._get_default_parameters()
        else:
            self.logger.warning("No parameter manager available, using defaults")
            parameters = self._get_default_parameters()

        # Start analysis
        if hasattr(self.analysis_engine, 'run_analysis_pipeline'):
            self.analysis_engine.run_analysis_pipeline(data, parameters, steps)
        else:
            QMessageBox.information(
                self, "Info",
                "Analysis engine not fully loaded. Please check the console for any import errors."
            )

    def _optimize_parameters(self, parameters: Dict[str, Any], data, steps) -> Dict[str, Any]:
        """Optimize parameters based on data characteristics."""

        optimized = parameters.copy()

        # If doing both detection and linking with trackpy detection
        if (AnalysisStep.DETECTION in steps and AnalysisStep.LINKING in steps):
            detection_method = parameters.get('detection_method', 'threshold')
            linking_method = parameters.get('linking_method', 'nearest_neighbor')

            # Auto-match linking method to detection method
            if detection_method == 'trackpy' and linking_method != 'trackpy':
                self.logger.info("Auto-switching to trackpy linking for trackpy detection")
                optimized['linking_method'] = 'trackpy'

                # Apply trackpy-optimized linking parameters
                current_distance = parameters.get('max_distance', 5.0)
                if current_distance > 3.0:
                    optimized['max_distance'] = 2.5  # More conservative for trackpy
                    self.logger.info(f"Reduced max_distance from {current_distance} to 2.5 for trackpy")

                # Reduce memory for dense data
                if isinstance(data, np.ndarray) and len(data.shape) >= 2:
                    # Estimate if data might be dense based on image size
                    total_pixels = np.prod(data.shape[-2:])  # Last 2 dims are spatial
                    if total_pixels < 50000:  # Small field of view = likely dense
                        optimized['max_gap_frames'] = min(1, parameters.get('max_gap_frames', 2))
                        self.logger.info("Reduced max_gap_frames for dense data")

            elif detection_method in ['threshold', 'log'] and linking_method == 'trackpy':
                self.logger.info("Consider using nearest_neighbor linking for threshold/LoG detection")

        # Optimize other parameters based on data type
        if isinstance(data, np.ndarray) and len(data.shape) == 3:
            n_frames = data.shape[0]
            if n_frames < 10:
                # Short time series - reduce memory and min track length
                optimized['max_gap_frames'] = min(1, parameters.get('max_gap_frames', 2))
                optimized['min_track_length'] = max(2, min(3, parameters.get('min_track_length', 3)))
                self.logger.info("Optimized parameters for short time series")

        return optimized

    # Also add this helper method to suggest parameter improvements in the GUI:

    def suggest_parameter_improvements(self):
        """Show suggestions for parameter improvements based on current settings."""
        if self.parameter_manager is None:
            return

        try:
            current_params = self.parameter_manager.get_all_parameters()

            suggestions = []

            # Check for common issues
            detection_method = current_params.get('detection_method', 'threshold')
            linking_method = current_params.get('linking_method', 'nearest_neighbor')
            max_distance = current_params.get('max_distance', 5.0)

            if detection_method == 'trackpy' and linking_method != 'trackpy':
                suggestions.append("💡 Consider using 'trackpy' linking with 'trackpy' detection for best results")

            if max_distance > 5.0:
                suggestions.append("⚠️ Max distance > 5 pixels may cause false linkages. Try 2-3 pixels first.")

            if current_params.get('min_track_length', 3) < 5:
                suggestions.append("🎯 Increase min track length to 5-10 to filter out noise")

            if suggestions:
                suggestion_text = "Parameter Suggestions:\n\n" + "\n\n".join(suggestions)
                QMessageBox.information(self, "Parameter Suggestions", suggestion_text)
            else:
                QMessageBox.information(self, "Parameters", "Current parameters look good! 👍")

        except Exception as e:
            self.logger.error(f"Error generating parameter suggestions: {e}")

    def _get_default_parameters(self):
        """Get default parameters as fallback."""
        try:
            from particle_tracker.core.analysis_engine import AnalysisParameters
            return AnalysisParameters()
        except ImportError:
            # Use a simple dict if AnalysisParameters not available
            return {
                'detection_method': 'threshold',
                'detection_sigma': 1.6,
                'detection_threshold': 3.0,
                'linking_method': 'nearest_neighbor',
                'max_distance': 5.0,
                'max_gap_frames': 2,
                'min_track_length': 3,
                'pixel_size': 108.0,
                'frame_rate': 10.0,
                'mobility_threshold': 2.11
            }

    def _stop_analysis(self):
        """Stop the current analysis."""
        if hasattr(self.analysis_engine, 'stop_analysis'):
            self.analysis_engine.stop_analysis()

    def _on_analysis_started(self, steps: List[str]):
        """Handle analysis started."""
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Running: {', '.join(steps)}")

    def _on_analysis_completed(self, result: Any):
        """Handle analysis completed."""
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Analysis completed")

    def _update_progress(self, message: str, percentage: int):
        """Update progress display."""
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)

    def _on_analysis_error(self, error_message: str):
        """Handle analysis error."""
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Analysis failed")

        QMessageBox.critical(self, "Analysis Error", error_message)
