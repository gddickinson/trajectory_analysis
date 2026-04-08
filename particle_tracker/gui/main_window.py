#!/usr/bin/env python3
"""
Redesigned Enhanced Main Window GUI Module
==========================================

Completely redesigned GUI with better tab organization, improved accessibility,
and cleaner layout while preserving all functionality.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any

from PyQt6.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QSplitter,
    QTabWidget, QMenuBar, QStatusBar, QProgressBar, QTextEdit,
    QLabel, QPushButton, QFileDialog, QMessageBox, QTreeView,
    QTableView, QGroupBox, QGridLayout, QDockWidget, QToolBar,
    QFrame, QScrollArea, QSlider, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QLineEdit, QFormLayout, QAbstractItemView,
    QHeaderView, QDialog, QDialogButtonBox, QInputDialog, QStackedWidget,
    QListWidget, QListWidgetItem
)
from PyQt6.QtCore import (
    Qt, QTimer, QSettings, pyqtSignal, QAbstractTableModel,
    QModelIndex, QVariant, QSortFilterProxyModel
)
from PyQt6.QtGui import QAction, QFont, QIcon, QStandardItemModel, QStandardItem

import pandas as pd
import numpy as np

# Import GUI components with fallbacks
try:
    from .visualization_widget import EnhancedVisualizationWidget
except ImportError:
    from .visualization_widget import VisualizationWidget as EnhancedVisualizationWidget

try:
    from .parameter_panels import ParameterPanelManager
except ImportError:
    class ParameterPanelManager(QWidget):
        def __init__(self, data_manager=None):
            super().__init__()
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("Parameter Manager - Loading..."))

        def get_all_parameters(self):
            return {}

        def set_all_parameters(self, params):
            pass

try:
    from .data_browser import EnhancedDataBrowserWidget
except ImportError:
    class EnhancedDataBrowserWidget(QWidget):
        def __init__(self, data_manager):
            super().__init__()
            self.data_manager = data_manager
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("Enhanced Data Browser - Loading..."))

        def get_selected_data_name(self):
            return None

try:
    from .analysis_control import EnhancedAnalysisControlWidget
except ImportError:
    class EnhancedAnalysisControlWidget(QWidget):
        def __init__(self, analysis_engine, data_manager):
            super().__init__()
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("Enhanced Analysis Control - Loading..."))

# Core imports with error handling
try:
    from ..core.analysis_engine import AnalysisParameters, AnalysisStep
except ImportError:
    # Create fallback classes
    from enum import Enum
    from dataclasses import dataclass

    class AnalysisStep(Enum):
        DETECTION = "detection"
        LINKING = "linking"
        FEATURES = "features"
        ENHANCED_FEATURES = "enhanced_features"
        CLASSIFICATION = "classification"

    @dataclass
    class AnalysisParameters:
        detection_method: str = "threshold"
        linking_method: str = "nearest_neighbor"

try:
    from ..core.batch_analysis import BatchAnalysisManager, BatchExperiment
except ImportError:
    class BatchAnalysisManager:
        def __init__(self, analysis_engine, data_manager):
            self.analysis_engine = analysis_engine
            self.data_manager = data_manager

    class BatchExperiment:
        pass

try:
    from ..analysis.autocorrelation_analysis import DirectionAutocorrelationAnalyzer
except ImportError:
    class DirectionAutocorrelationAnalyzer:
        def __init__(self):
            pass


class AnalysisPresetsTab(QWidget):
    """NEW - Dedicated tab for analysis presets only - clean and simple."""

    def __init__(self, analysis_engine, data_manager, parent=None):
        super().__init__(parent)
        self.analysis_engine = analysis_engine
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the analysis presets interface - focused and uncluttered."""
        layout = QVBoxLayout(self)
        layout.setSpacing(30)
        layout.setContentsMargins(50, 40, 50, 40)

        # Title and description
        title_label = QLabel("🚀 Analysis Presets")
        title_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #2c3e50; margin-bottom: 10px;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        description_label = QLabel("Choose a preset to quickly analyze your data with optimized parameters")
        description_label.setStyleSheet("font-size: 16px; color: #7f8c8d; margin-bottom: 30px;")
        description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(description_label)

        # Data selection section
        data_section = self._create_data_selection_section()
        layout.addWidget(data_section)

        # Preset buttons in a grid
        presets_section = self._create_presets_section()
        layout.addWidget(presets_section)

        # Progress and status
        progress_section = self._create_progress_section()
        layout.addWidget(progress_section)

        layout.addStretch()

    def _create_data_selection_section(self):
        """Create data selection section."""
        group = QGroupBox("📊 Select Data to Analyze")
        group.setStyleSheet("QGroupBox { font-size: 18px; font-weight: bold; padding-top: 15px; }")
        layout = QVBoxLayout(group)

        # Data combo with larger font
        self.data_combo = QComboBox()
        self.data_combo.setMinimumHeight(50)
        self.data_combo.setStyleSheet("font-size: 16px; padding: 12px;")
        layout.addWidget(self.data_combo)

        # Data info display
        self.data_info_label = QLabel("No data selected")
        self.data_info_label.setStyleSheet("color: #7f8c8d; margin-top: 15px; font-size: 14px;")
        layout.addWidget(self.data_info_label)

        return group

    def _create_presets_section(self):
        """Create analysis presets section with large, clear buttons."""
        group = QGroupBox("🎯 Analysis Presets")
        group.setStyleSheet("QGroupBox { font-size: 18px; font-weight: bold; padding-top: 15px; }")
        layout = QGridLayout(group)
        layout.setSpacing(20)

        # Define presets with descriptions
        presets = [
            {
                'name': '🚀 Quick Analysis',
                'description': 'Fast detection → linking → basic features\nBest for: Initial data exploration (2-5 minutes)',
                'color': '#3498db',
                'preset_type': 'quick'
            },
            {
                'name': '🔬 Comprehensive Analysis',
                'description': 'Full pipeline with all enhanced features\nBest for: Complete analysis (5-15 minutes)',
                'color': '#e74c3c',
                'preset_type': 'comprehensive'
            },
            {
                'name': '🏃 Mobility-Focused',
                'description': 'Enhanced mobility and diffusion metrics\nBest for: Mobility classification studies',
                'color': '#f39c12',
                'preset_type': 'mobility'
            },
            {
                'name': '📐 Shape-Focused',
                'description': 'Advanced shape and geometry analysis\nBest for: Morphology studies',
                'color': '#9b59b6',
                'preset_type': 'shape'
            }
        ]

        for i, preset in enumerate(presets):
            button = self._create_preset_button(preset)
            row = i // 2
            col = i % 2
            layout.addWidget(button, row, col)

        return group

    def _create_preset_button(self, preset_info):
        """Create a large, informative preset button."""
        button = QPushButton()
        button.setMinimumHeight(140)
        button.setMinimumWidth(350)

        # Simple text instead of HTML for better compatibility
        button.setText(f"{preset_info['name']}\n\n{preset_info['description']}")

        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {preset_info['color']};
                border: none;
                border-radius: 15px;
                color: white;
                font-weight: bold;
                font-size: 14px;
                text-align: center;
                padding: 15px;
            }}
            QPushButton:hover {{
                background-color: {self._darken_color(preset_info['color'])};
            }}
            QPushButton:pressed {{
                background-color: {self._darken_color(preset_info['color'])};
            }}
        """)

        # Store preset type for signal connection
        button.preset_type = preset_info['preset_type']

        return button

    def _darken_color(self, hex_color):
        """Darken a hex color for hover effects."""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        darkened = tuple(max(0, int(c * 0.8)) for c in rgb)
        return f"#{darkened[0]:02x}{darkened[1]:02x}{darkened[2]:02x}"

    def _create_progress_section(self):
        """Create progress and status section."""
        group = QGroupBox("📈 Analysis Status")
        group.setStyleSheet("QGroupBox { font-size: 18px; font-weight: bold; padding-top: 15px; }")
        layout = QVBoxLayout(group)

        # Status label
        self.status_label = QLabel("Ready to analyze")
        self.status_label.setStyleSheet("font-size: 16px; margin-bottom: 15px;")
        layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(30)
        layout.addWidget(self.progress_bar)

        # Control buttons
        button_layout = QHBoxLayout()

        self.stop_btn = QPushButton("⏹️ Stop Analysis")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setStyleSheet("background-color: #e74c3c; color: white; border: none; border-radius: 8px; font-weight: bold; font-size: 14px;")
        button_layout.addWidget(self.stop_btn)

        button_layout.addStretch()

        self.view_results_btn = QPushButton("👁️ View Results")
        self.view_results_btn.setMinimumHeight(40)
        self.view_results_btn.setStyleSheet("background-color: #27ae60; color: white; border: none; border-radius: 8px; font-weight: bold; font-size: 14px;")
        button_layout.addWidget(self.view_results_btn)

        layout.addLayout(button_layout)

        return group

    def _connect_signals(self):
        """Connect preset signals."""
        # Data selection
        self.data_combo.currentTextChanged.connect(self._on_data_selected)

        # Find and connect preset buttons
        for button in self.findChildren(QPushButton):
            if hasattr(button, 'preset_type'):
                button.clicked.connect(lambda checked, b=button: self._run_preset_analysis(b.preset_type))

        # Control buttons
        self.stop_btn.clicked.connect(self._stop_analysis)
        self.view_results_btn.clicked.connect(self._view_results)

        # Update data list when data manager changes
        if hasattr(self.data_manager, 'dataLoaded'):
            self.data_manager.dataLoaded.connect(self._update_data_list)
        if hasattr(self.data_manager, 'dataRemoved'):
            self.data_manager.dataRemoved.connect(self._update_data_list)

        # Connect analysis engine signals
        if hasattr(self.analysis_engine, 'analysisStarted'):
            self.analysis_engine.analysisStarted.connect(self._on_analysis_started)
        if hasattr(self.analysis_engine, 'analysisCompleted'):
            self.analysis_engine.analysisCompleted.connect(self._on_analysis_completed)
        if hasattr(self.analysis_engine, 'progressUpdate'):
            self.analysis_engine.progressUpdate.connect(self._update_progress)

        # Initial data list update
        self._update_data_list()

    def _update_data_list(self):
        """Update the data selection combo box."""
        try:
            current_selection = self.data_combo.currentText()
            self.data_combo.clear()

            if hasattr(self.data_manager, 'get_data_names'):
                data_names = self.data_manager.get_data_names()
                self.data_combo.addItems(data_names)

                if current_selection in data_names:
                    self.data_combo.setCurrentText(current_selection)

                self.logger.debug(f"Updated data list with {len(data_names)} items")
            else:
                self.logger.warning("Data manager does not have get_data_names method")

        except Exception as e:
            self.logger.error(f"Error updating data list: {e}")

    def _on_data_selected(self, data_name):
        """Handle data selection change."""
        if data_name:
            try:
                if hasattr(self.data_manager, 'get_data'):
                    data = self.data_manager.get_data(data_name)
                    if data is not None:
                        if hasattr(data, 'shape'):
                            info_text = f"Image stack: {data.shape} ({data.dtype})"
                        elif hasattr(data, '__len__'):
                            info_text = f"Data: {len(data)} rows"
                        else:
                            info_text = "Data loaded successfully"
                        self.data_info_label.setText(info_text)
                        self.logger.info(f"Selected data: {data_name}")
                    else:
                        self.data_info_label.setText("Could not load data")
                else:
                    self.data_info_label.setText("Data manager not available")
            except Exception as e:
                self.data_info_label.setText(f"Error: {e}")
                self.logger.error(f"Error selecting data: {e}")
        else:
            self.data_info_label.setText("No data selected")

    def _run_preset_analysis(self, preset_type):
        """Run analysis with the selected preset."""
        try:
            data_name = self.data_combo.currentText()
            if not data_name:
                QMessageBox.warning(self, "No Data", "Please select data to analyze first.")
                return

            if not hasattr(self.data_manager, 'get_data'):
                QMessageBox.warning(self, "Error", "Data manager not properly initialized.")
                return

            data = self.data_manager.get_data(data_name)
            if data is None:
                QMessageBox.warning(self, "No Data", f"Could not load data: {data_name}")
                return

            # Get analysis steps based on preset
            steps = self._get_preset_steps(preset_type)

            # Get parameters
            parameters = self._get_analysis_parameters()

            # Show confirmation
            preset_names = {
                'quick': 'Quick Analysis',
                'comprehensive': 'Comprehensive Analysis',
                'mobility': 'Mobility-Focused Analysis',
                'shape': 'Shape-Focused Analysis'
            }

            preset_name = preset_names.get(preset_type, preset_type)
            reply = QMessageBox.question(
                self, "Start Analysis",
                f"Start {preset_name} on {data_name}?\n\n"
                f"This will analyze your data using optimized parameters.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                if hasattr(self.analysis_engine, 'run_analysis_pipeline'):
                    self.analysis_engine.run_analysis_pipeline(data, parameters, steps)
                    self.logger.info(f"Started {preset_type} analysis on {data_name}")
                else:
                    QMessageBox.information(
                        self, "Analysis Started",
                        f"Started {preset_name} on {data_name}\n"
                        f"Analysis engine integration pending."
                    )

        except Exception as e:
            self.logger.error(f"Error running {preset_type} analysis: {e}")
            QMessageBox.critical(self, "Analysis Error", f"Failed to run analysis: {e}")

    def _get_preset_steps(self, preset_type):
        """Get analysis steps for a preset type."""
        # These would normally be imported from your analysis module
        # For now, using placeholder strings
        if preset_type == "quick":
            return ["DETECTION", "LINKING", "FEATURES", "CLASSIFICATION"]
        elif preset_type == "comprehensive":
            return ["DETECTION", "LINKING", "ENHANCED_FEATURES", "CLASSIFICATION"]
        elif preset_type == "mobility":
            return ["DETECTION", "LINKING", "ENHANCED_FEATURES"]
        elif preset_type == "shape":
            return ["DETECTION", "LINKING", "ENHANCED_FEATURES"]
        else:
            return ["DETECTION", "LINKING", "FEATURES"]

    def _get_analysis_parameters(self):
        """Get current analysis parameters."""
        try:
            # Return default parameters - this would normally get from parameter manager
            return {
                'detection_method': 'trackpy',
                'detection_threshold': 2.0,
                'min_intensity': 50,
                'max_intensity': 50000,
                'linking_method': 'trackpy',
                'max_distance': 5.0,
                'memory': 3,
                'min_track_length': 3
            }
        except Exception as e:
            self.logger.warning(f"Error getting parameters, using defaults: {e}")
            return {}

    def _stop_analysis(self):
        """Stop current analysis."""
        try:
            if hasattr(self.analysis_engine, 'stop_analysis'):
                self.analysis_engine.stop_analysis()
                self.logger.info("Analysis stopped by user")
            else:
                QMessageBox.information(self, "Stop Analysis", "Analysis stopping functionality pending.")
        except Exception as e:
            self.logger.error(f"Error stopping analysis: {e}")

    def _view_results(self):
        """Switch to results tab."""
        try:
            main_window = self.parent()
            if hasattr(main_window, 'main_tabs'):
                main_window.main_tabs.setCurrentIndex(3)  # Visualization tab
        except Exception as e:
            self.logger.error(f"Error switching to results: {e}")

    def _on_analysis_started(self, steps):
        """Handle analysis started."""
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Analysis running...")

        # Disable preset buttons during analysis
        for button in self.findChildren(QPushButton):
            if hasattr(button, 'preset_type'):
                button.setEnabled(False)

    def _on_analysis_completed(self, result):
        """Handle analysis completed."""
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Analysis completed successfully!")

        # Re-enable preset buttons
        for button in self.findChildren(QPushButton):
            if hasattr(button, 'preset_type'):
                button.setEnabled(True)

        QMessageBox.information(self, "Analysis Complete", "Analysis has completed successfully!\n\nClick 'View Results' to see your data.")

    def _update_progress(self, message, percentage):
        """Update analysis progress."""
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)

class AnalysisSetupTab(QWidget):
    """Analysis Setup tab - focused ONLY on parameters and custom analysis - NO PRESETS."""

    def __init__(self, analysis_engine, data_manager, parent=None):
        super().__init__(parent)
        self.analysis_engine = analysis_engine
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the analysis configuration interface - parameters only."""
        layout = QHBoxLayout(self)

        # Create splitter for parameters and custom analysis control
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left: Parameter Configuration (larger)
        param_panel = self._create_parameter_panel()
        splitter.addWidget(param_panel)

        # Right: Custom Analysis Control (smaller, NO PRESETS)
        control_panel = self._create_custom_analysis_panel()
        splitter.addWidget(control_panel)

        # Set proportions
        splitter.setSizes([700, 300])

    def _create_parameter_panel(self):
        """Create parameter configuration panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("⚙️ Analysis Parameters")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 15px;")
        layout.addWidget(title)

        # Parameter tabs for organization
        param_tabs = QTabWidget()

        # Detection Parameters Tab
        detection_tab = self._create_detection_params_tab()
        param_tabs.addTab(detection_tab, "🔍 Detection")

        # Linking Parameters Tab
        linking_tab = self._create_linking_params_tab()
        param_tabs.addTab(linking_tab, "🔗 Linking")

        # Feature Parameters Tab
        features_tab = self._create_features_params_tab()
        param_tabs.addTab(features_tab, "📊 Features")

        # Classification Parameters Tab
        classification_tab = self._create_classification_params_tab()
        param_tabs.addTab(classification_tab, "🎯 Classification")

        layout.addWidget(param_tabs)

        # Parameter management buttons
        button_layout = QHBoxLayout()

        self.load_params_btn = QPushButton("📂 Load Parameters")
        self.save_params_btn = QPushButton("💾 Save Parameters")
        self.reset_params_btn = QPushButton("🔄 Reset to Defaults")

        for btn in [self.load_params_btn, self.save_params_btn, self.reset_params_btn]:
            btn.setMinimumHeight(35)
            button_layout.addWidget(btn)

        layout.addLayout(button_layout)

        return panel

    def _create_detection_params_tab(self):
        """Create detection parameters tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Detection method
        method_group = QGroupBox("Detection Method")
        method_layout = QVBoxLayout(method_group)

        self.detection_method_combo = QComboBox()
        self.detection_method_combo.addItems(["trackpy", "threshold", "blob", "enhanced_trackpy"])
        method_layout.addWidget(self.detection_method_combo)

        layout.addWidget(method_group)

        # Basic parameters
        basic_group = QGroupBox("Basic Parameters")
        basic_layout = QFormLayout(basic_group)

        self.diameter_spin = QSpinBox()
        self.diameter_spin.setRange(3, 21)
        self.diameter_spin.setValue(7)
        basic_layout.addRow("Particle Diameter:", self.diameter_spin)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 10.0)
        self.threshold_spin.setValue(2.0)
        self.threshold_spin.setSingleStep(0.1)
        basic_layout.addRow("Detection Threshold:", self.threshold_spin)

        self.min_intensity_spin = QSpinBox()
        self.min_intensity_spin.setRange(1, 10000)
        self.min_intensity_spin.setValue(50)
        basic_layout.addRow("Min Intensity:", self.min_intensity_spin)

        self.max_intensity_spin = QSpinBox()
        self.max_intensity_spin.setRange(100, 100000)
        self.max_intensity_spin.setValue(50000)
        basic_layout.addRow("Max Intensity:", self.max_intensity_spin)

        layout.addWidget(basic_group)

        # Advanced parameters
        advanced_group = QGroupBox("Advanced Parameters")
        advanced_layout = QFormLayout(advanced_group)

        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.5, 5.0)
        self.sigma_spin.setValue(1.6)
        self.sigma_spin.setSingleStep(0.1)
        advanced_layout.addRow("Gaussian Sigma:", self.sigma_spin)

        self.preprocess_cb = QCheckBox("Enable Preprocessing")
        self.preprocess_cb.setChecked(True)
        advanced_layout.addRow(self.preprocess_cb)

        self.background_sub_cb = QCheckBox("Background Subtraction")
        self.background_sub_cb.setChecked(True)
        advanced_layout.addRow(self.background_sub_cb)

        layout.addWidget(advanced_group)
        layout.addStretch()

        return widget

    def _create_linking_params_tab(self):
        """Create linking parameters tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Linking method
        method_group = QGroupBox("Linking Method")
        method_layout = QVBoxLayout(method_group)

        self.linking_method_combo = QComboBox()
        self.linking_method_combo.addItems(["trackpy", "nearest_neighbor", "LAP"])
        method_layout.addWidget(self.linking_method_combo)

        layout.addWidget(method_group)

        # Distance parameters
        distance_group = QGroupBox("Distance Parameters")
        distance_layout = QFormLayout(distance_group)

        self.max_distance_spin = QDoubleSpinBox()
        self.max_distance_spin.setRange(1.0, 50.0)
        self.max_distance_spin.setValue(5.0)
        self.max_distance_spin.setSingleStep(0.5)
        distance_layout.addRow("Max Distance:", self.max_distance_spin)

        self.memory_spin = QSpinBox()
        self.memory_spin.setRange(0, 20)
        self.memory_spin.setValue(3)
        distance_layout.addRow("Memory (frames):", self.memory_spin)

        self.min_track_length_spin = QSpinBox()
        self.min_track_length_spin.setRange(2, 100)
        self.min_track_length_spin.setValue(3)
        distance_layout.addRow("Min Track Length:", self.min_track_length_spin)

        layout.addWidget(distance_group)

        # Advanced linking
        advanced_group = QGroupBox("Advanced Linking")
        advanced_layout = QFormLayout(advanced_group)

        self.adaptive_cb = QCheckBox("Adaptive Linking")
        advanced_layout.addRow(self.adaptive_cb)

        self.prediction_cb = QCheckBox("Velocity Prediction")
        advanced_layout.addRow(self.prediction_cb)

        layout.addWidget(advanced_group)
        layout.addStretch()

        return widget

    def _create_features_params_tab(self):
        """Create features parameters tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Feature selection
        features_group = QGroupBox("Feature Calculation")
        features_layout = QVBoxLayout(features_group)

        # Basic features
        self.calc_velocity_cb = QCheckBox("Velocity & Speed")
        self.calc_velocity_cb.setChecked(True)
        features_layout.addWidget(self.calc_velocity_cb)

        self.calc_rg_cb = QCheckBox("Radius of Gyration")
        self.calc_rg_cb.setChecked(True)
        features_layout.addWidget(self.calc_rg_cb)

        self.calc_scaled_rg_cb = QCheckBox("Scaled Radius of Gyration")
        self.calc_scaled_rg_cb.setChecked(True)
        features_layout.addWidget(self.calc_scaled_rg_cb)

        # Advanced features
        self.calc_density_cb = QCheckBox("Multi-Radius Density")
        self.calc_density_cb.setChecked(True)
        features_layout.addWidget(self.calc_density_cb)

        self.calc_shape_cb = QCheckBox("Advanced Shape Metrics")
        self.calc_shape_cb.setChecked(True)
        features_layout.addWidget(self.calc_shape_cb)

        self.calc_diffusion_cb = QCheckBox("Diffusion Analysis")
        self.calc_diffusion_cb.setChecked(True)
        features_layout.addWidget(self.calc_diffusion_cb)

        layout.addWidget(features_group)

        # Physical parameters
        physical_group = QGroupBox("Physical Parameters")
        physical_layout = QFormLayout(physical_group)

        self.pixel_size_spin = QDoubleSpinBox()
        self.pixel_size_spin.setRange(1.0, 1000.0)
        self.pixel_size_spin.setValue(108.0)
        self.pixel_size_spin.setSuffix(" nm")
        physical_layout.addRow("Pixel Size:", self.pixel_size_spin)

        self.frame_rate_spin = QDoubleSpinBox()
        self.frame_rate_spin.setRange(0.1, 1000.0)
        self.frame_rate_spin.setValue(10.0)
        self.frame_rate_spin.setSuffix(" Hz")
        physical_layout.addRow("Frame Rate:", self.frame_rate_spin)

        layout.addWidget(physical_group)
        layout.addStretch()

        return widget

    def _create_classification_params_tab(self):
        """Create classification parameters tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Classification method
        method_group = QGroupBox("Classification Method")
        method_layout = QVBoxLayout(method_group)

        self.classification_method_combo = QComboBox()
        self.classification_method_combo.addItems(["threshold", "svm", "kmeans", "none"])
        method_layout.addWidget(self.classification_method_combo)

        layout.addWidget(method_group)

        # Threshold parameters
        threshold_group = QGroupBox("Threshold Parameters")
        threshold_layout = QFormLayout(threshold_group)

        self.mobility_threshold_spin = QDoubleSpinBox()
        self.mobility_threshold_spin.setRange(0.1, 20.0)
        self.mobility_threshold_spin.setValue(2.11)
        self.mobility_threshold_spin.setSingleStep(0.1)
        threshold_layout.addRow("Mobility Threshold:", self.mobility_threshold_spin)

        self.linear_threshold_spin = QDoubleSpinBox()
        self.linear_threshold_spin.setRange(1.0, 100.0)
        self.linear_threshold_spin.setValue(20.0)
        threshold_layout.addRow("Linear Eigenvalue Threshold:", self.linear_threshold_spin)

        layout.addWidget(threshold_group)

        layout.addStretch()

        return widget

    def _create_custom_analysis_panel(self):
        """Create custom analysis control panel - NO PRESET BUTTONS."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("🔧 Custom Analysis")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 15px;")
        layout.addWidget(title)

        # Data selection
        data_group = QGroupBox("Data Selection")
        data_layout = QVBoxLayout(data_group)

        self.data_combo = QComboBox()
        self.data_combo.setMinimumHeight(30)
        data_layout.addWidget(self.data_combo)

        layout.addWidget(data_group)

        # Step selection
        steps_group = QGroupBox("Analysis Steps")
        steps_layout = QVBoxLayout(steps_group)

        self.detection_step_cb = QCheckBox("Detection")
        self.detection_step_cb.setChecked(True)
        steps_layout.addWidget(self.detection_step_cb)

        self.linking_step_cb = QCheckBox("Linking")
        self.linking_step_cb.setChecked(True)
        steps_layout.addWidget(self.linking_step_cb)

        self.features_step_cb = QCheckBox("Feature Calculation")
        self.features_step_cb.setChecked(True)
        steps_layout.addWidget(self.features_step_cb)

        self.classification_step_cb = QCheckBox("Classification")
        self.classification_step_cb.setChecked(True)
        steps_layout.addWidget(self.classification_step_cb)

        layout.addWidget(steps_group)

        # Execution control - CUSTOM ONLY
        execution_group = QGroupBox("Custom Execution")
        execution_layout = QVBoxLayout(execution_group)

        self.run_btn = QPushButton("▶️ Run Custom Analysis")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setStyleSheet("background-color: #3498db; color: white; font-weight: bold; border: none; border-radius: 5px;")
        execution_layout.addWidget(self.run_btn)

        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumHeight(35)
        self.stop_btn.setStyleSheet("background-color: #e74c3c; color: white; border: none; border-radius: 5px;")
        execution_layout.addWidget(self.stop_btn)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        execution_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-size: 12px; color: #7f8c8d;")
        execution_layout.addWidget(self.status_label)

        layout.addWidget(execution_group)

        layout.addStretch()
        return panel

    def _connect_signals(self):
        """Connect analysis setup signals - custom analysis only."""
        # Parameter management
        self.load_params_btn.clicked.connect(self._load_parameters)
        self.save_params_btn.clicked.connect(self._save_parameters)
        self.reset_params_btn.clicked.connect(self._reset_parameters)

        # Data selection
        self.data_combo.currentTextChanged.connect(self._on_data_selected)

        # Custom analysis execution only
        self.run_btn.clicked.connect(self._run_custom_analysis)
        self.stop_btn.clicked.connect(self._stop_analysis)

        # Update data list when data manager changes
        if hasattr(self.data_manager, 'dataLoaded'):
            self.data_manager.dataLoaded.connect(self._update_data_list)
        if hasattr(self.data_manager, 'dataRemoved'):
            self.data_manager.dataRemoved.connect(self._update_data_list)

        # Connect analysis engine signals
        if hasattr(self.analysis_engine, 'analysisStarted'):
            self.analysis_engine.analysisStarted.connect(self._on_analysis_started)
        if hasattr(self.analysis_engine, 'analysisCompleted'):
            self.analysis_engine.analysisCompleted.connect(self._on_analysis_completed)
        if hasattr(self.analysis_engine, 'progressUpdate'):
            self.analysis_engine.progressUpdate.connect(self._update_progress)

        # Initial data list update
        self._update_data_list()

    def _update_data_list(self):
        """Update the data selection combo box."""
        try:
            current_selection = self.data_combo.currentText()
            self.data_combo.clear()

            if hasattr(self.data_manager, 'get_data_names'):
                data_names = self.data_manager.get_data_names()
                self.data_combo.addItems(data_names)

                if current_selection in data_names:
                    self.data_combo.setCurrentText(current_selection)

                self.logger.debug(f"Updated data list with {len(data_names)} items")
            else:
                self.logger.warning("Data manager does not have get_data_names method")

        except Exception as e:
            self.logger.error(f"Error updating data list: {e}")

    def _on_data_selected(self, data_name):
        """Handle data selection change."""
        if data_name:
            self.logger.info(f"Selected data: {data_name}")

    def _load_parameters(self):
        """Load analysis parameters from file."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Parameters", "", "JSON Files (*.json);;All Files (*)"
            )
            if file_path:
                QMessageBox.information(self, "Load Parameters", "Parameter loading will be implemented soon.")
        except Exception as e:
            self.logger.error(f"Error loading parameters: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load parameters: {e}")

    def _save_parameters(self):
        """Save current analysis parameters to file."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Parameters", "", "JSON Files (*.json);;All Files (*)"
            )
            if file_path:
                QMessageBox.information(self, "Save Parameters", "Parameter saving will be implemented soon.")
        except Exception as e:
            self.logger.error(f"Error saving parameters: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save parameters: {e}")

    def _reset_parameters(self):
        """Reset parameters to defaults."""
        try:
            QMessageBox.information(self, "Reset Parameters", "Parameter reset will be implemented soon.")
        except Exception as e:
            self.logger.error(f"Error resetting parameters: {e}")

    def _run_custom_analysis(self):
        """Run custom analysis based on current settings."""
        try:
            data_name = self.data_combo.currentText()
            if not data_name:
                QMessageBox.warning(self, "No Data", "Please select data to analyze first.")
                return

            if not hasattr(self.data_manager, 'get_data'):
                QMessageBox.warning(self, "Error", "Data manager not properly initialized.")
                return

            data = self.data_manager.get_data(data_name)
            if data is None:
                QMessageBox.warning(self, "No Data", f"Could not load data: {data_name}")
                return

            # Get selected steps
            steps = []
            if self.detection_step_cb.isChecked():
                steps.append("DETECTION")
            if self.linking_step_cb.isChecked():
                steps.append("LINKING")
            if self.features_step_cb.isChecked():
                steps.append("ENHANCED_FEATURES")
            if self.classification_step_cb.isChecked():
                steps.append("CLASSIFICATION")

            if not steps:
                QMessageBox.warning(self, "No Steps", "Please select at least one analysis step.")
                return

            # Get parameters from UI
            parameters = self._get_ui_parameters()

            # Run analysis
            if hasattr(self.analysis_engine, 'run_analysis_pipeline'):
                self.analysis_engine.run_analysis_pipeline(data, parameters, steps)
                self.logger.info(f"Started custom analysis on {data_name}")
            else:
                QMessageBox.information(
                    self, "Analysis Started",
                    f"Custom analysis started on {data_name}\n"
                    f"Steps: {len(steps)} selected\n"
                    f"Analysis engine integration pending."
                )

        except Exception as e:
            self.logger.error(f"Error running custom analysis: {e}")
            QMessageBox.critical(self, "Analysis Error", f"Failed to run analysis: {e}")

    def _get_ui_parameters(self):
        """Get parameters from the UI controls."""
        try:
            params = {
                'detection_method': self.detection_method_combo.currentText(),
                'diameter': self.diameter_spin.value(),
                'detection_threshold': self.threshold_spin.value(),
                'min_intensity': self.min_intensity_spin.value(),
                'max_intensity': self.max_intensity_spin.value(),
                'detection_sigma': self.sigma_spin.value(),
                'preprocess': self.preprocess_cb.isChecked(),
                'background_subtraction': self.background_sub_cb.isChecked(),
                'linking_method': self.linking_method_combo.currentText(),
                'max_distance': self.max_distance_spin.value(),
                'memory': self.memory_spin.value(),
                'min_track_length': self.min_track_length_spin.value(),
                'pixel_size': self.pixel_size_spin.value(),
                'frame_rate': self.frame_rate_spin.value(),
                'calculate_velocity': self.calc_velocity_cb.isChecked(),
                'calculate_rg': self.calc_rg_cb.isChecked(),
                'calculate_scaled_rg': self.calc_scaled_rg_cb.isChecked(),
                'calculate_density': self.calc_density_cb.isChecked(),
                'calculate_advanced_shape': self.calc_shape_cb.isChecked(),
                'calculate_diffusion': self.calc_diffusion_cb.isChecked(),
                'classification_method': self.classification_method_combo.currentText(),
                'mobility_threshold': self.mobility_threshold_spin.value(),
                'linear_eigenvalue_threshold': self.linear_threshold_spin.value()
            }

            self.logger.info("Retrieved parameters from UI")
            return params

        except Exception as e:
            self.logger.error(f"Error getting UI parameters: {e}")
            return {}

    def _stop_analysis(self):
        """Stop current analysis."""
        try:
            if hasattr(self.analysis_engine, 'stop_analysis'):
                self.analysis_engine.stop_analysis()
                self.logger.info("Analysis stopped by user")
            else:
                QMessageBox.information(self, "Stop Analysis", "Analysis stopping functionality pending.")
        except Exception as e:
            self.logger.error(f"Error stopping analysis: {e}")

    def _on_analysis_started(self, steps):
        """Handle analysis started."""
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Analysis running...")

    def _on_analysis_completed(self, result):
        """Handle analysis completed."""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Analysis completed successfully")

        QMessageBox.information(self, "Analysis Complete", "Custom analysis has completed successfully!")

    def _update_progress(self, message, percentage):
        """Update analysis progress."""
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)

class DataManagementTab(QWidget):
    """Dedicated tab for all data management functions."""

    def __init__(self, data_manager, project_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.project_manager = project_manager
        self.logger = logging.getLogger(__name__)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the data management interface."""
        layout = QHBoxLayout(self)

        # Create splitter for data browser and project management
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left: Data Browser (larger)
        data_panel = self._create_data_browser_panel()
        splitter.addWidget(data_panel)

        # Right: Project & File Management
        management_panel = self._create_management_panel()
        splitter.addWidget(management_panel)

        # Set proportions: data browser gets more space
        splitter.setSizes([700, 300])

    def _connect_signals(self):
        """Connect button signals to actual functionality."""
        # File import buttons
        self.refresh_btn.clicked.connect(self._refresh_data)
        self.import_btn.clicked.connect(self._import_data_dialog)

        # Project management buttons
        self.new_project_btn.clicked.connect(self._new_project)
        self.open_project_btn.clicked.connect(self._open_project)
        self.save_project_btn.clicked.connect(self._save_project)
        self.save_as_btn.clicked.connect(self._save_project_as)

        # File operations
        self.import_images_btn.clicked.connect(self._import_images)
        self.import_trajectories_btn.clicked.connect(self._import_trajectories)
        self.import_folder_btn.clicked.connect(self._import_folder)

        # Utility buttons
        self.clear_recent_btn.clicked.connect(self._clear_recent_files)
        self.cleanup_btn.clicked.connect(self._cleanup_memory)

        # Update memory display periodically
        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self._update_memory_display)
        self.memory_timer.start(5000)  # Update every 5 seconds

    def _setup_ui(self):
        """Setup the data management interface."""
        layout = QHBoxLayout(self)

        # Create splitter for data browser and project management
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left: Data Browser (larger)
        data_panel = self._create_data_browser_panel()
        splitter.addWidget(data_panel)

        # Right: Project & File Management
        management_panel = self._create_management_panel()
        splitter.addWidget(management_panel)

        # Set proportions: data browser gets more space
        splitter.setSizes([700, 300])

    def _create_data_browser_panel(self) -> QWidget:
        """Create the enhanced data browser panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Header with title and controls
        header_layout = QHBoxLayout()

        title = QLabel("📁 Data Browser")
        title.setFont(QFont("", 14, QFont.Weight.Bold))
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Quick action buttons
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.setToolTip("Refresh data list")
        header_layout.addWidget(self.refresh_btn)

        self.import_btn = QPushButton("📂 Import Data...")
        self.import_btn.setToolTip("Import data files")
        header_layout.addWidget(self.import_btn)

        layout.addLayout(header_layout)

        # Data browser widget
        self.data_browser = EnhancedDataBrowserWidget(self.data_manager)
        layout.addWidget(self.data_browser)

        return panel

    def _create_management_panel(self) -> QWidget:
        """Create the project and file management panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Project Management Section
        project_group = QGroupBox("📋 Project Management")
        project_layout = QVBoxLayout(project_group)

        # Current project info
        self.project_label = QLabel("No project loaded")
        self.project_label.setStyleSheet("QLabel { font-weight: bold; color: #2E86AB; }")
        project_layout.addWidget(self.project_label)

        # Project controls
        project_controls = QGridLayout()

        self.new_project_btn = QPushButton("🆕 New Project")
        project_controls.addWidget(self.new_project_btn, 0, 0)

        self.open_project_btn = QPushButton("📂 Open Project")
        project_controls.addWidget(self.open_project_btn, 0, 1)

        self.save_project_btn = QPushButton("💾 Save Project")
        project_controls.addWidget(self.save_project_btn, 1, 0)

        self.save_as_btn = QPushButton("💾 Save As...")
        project_controls.addWidget(self.save_as_btn, 1, 1)

        project_layout.addLayout(project_controls)
        layout.addWidget(project_group)

        # File Operations Section
        file_group = QGroupBox("📄 File Operations")
        file_layout = QVBoxLayout(file_group)

        # File import options
        import_layout = QGridLayout()

        self.import_images_btn = QPushButton("🖼️ Import Images")
        self.import_images_btn.setToolTip("Import TIFF, ND2, or other image files")
        import_layout.addWidget(self.import_images_btn, 0, 0)

        self.import_trajectories_btn = QPushButton("📊 Import Trajectories")
        self.import_trajectories_btn.setToolTip("Import CSV or Excel trajectory data")
        import_layout.addWidget(self.import_trajectories_btn, 0, 1)

        self.import_folder_btn = QPushButton("📁 Import Folder")
        self.import_folder_btn.setToolTip("Import entire folder with multiple files")
        import_layout.addWidget(self.import_folder_btn, 1, 0, 1, 2)

        file_layout.addLayout(import_layout)
        layout.addWidget(file_group)

        # Recent Files Section
        recent_group = QGroupBox("🕒 Recent Files")
        recent_layout = QVBoxLayout(recent_group)

        self.recent_files_list = QListWidget()
        self.recent_files_list.setMaximumHeight(120)
        recent_layout.addWidget(self.recent_files_list)

        self.clear_recent_btn = QPushButton("Clear Recent")
        recent_layout.addWidget(self.clear_recent_btn)

        layout.addWidget(recent_group)

        # Memory Usage Section
        memory_group = QGroupBox("💾 Memory Usage")
        memory_layout = QVBoxLayout(memory_group)

        self.memory_label = QLabel("Memory: 0 MB")
        memory_layout.addWidget(self.memory_label)

        self.cleanup_btn = QPushButton("🧹 Clean Up Memory")
        memory_layout.addWidget(self.cleanup_btn)

        layout.addWidget(memory_group)

        layout.addStretch()
        return panel

    # Implementation methods for data management functionality
    def _refresh_data(self):
        """Refresh the data browser."""
        try:
            if hasattr(self.data_browser, '_refresh_data'):
                self.data_browser._refresh_data()
            self.logger.info("Data refreshed")
        except Exception as e:
            self.logger.error(f"Error refreshing data: {e}")

    def _import_data_dialog(self):
        """Show import data dialog with options."""
        from PyQt6.QtWidgets import QMessageBox

        msg = QMessageBox(self)
        msg.setWindowTitle("Import Data")
        msg.setText("What type of data would you like to import?")

        images_btn = msg.addButton("🖼️ Images (TIFF, ND2)", QMessageBox.ButtonRole.ActionRole)
        trajectories_btn = msg.addButton("📊 Trajectories (CSV)", QMessageBox.ButtonRole.ActionRole)
        folder_btn = msg.addButton("📁 Folder", QMessageBox.ButtonRole.ActionRole)
        msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)

        msg.exec()

        if msg.clickedButton() == images_btn:
            self._import_images()
        elif msg.clickedButton() == trajectories_btn:
            self._import_trajectories()
        elif msg.clickedButton() == folder_btn:
            self._import_folder()

    def _import_images(self):
        """Import image files."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Import Image Files", "",
            "Image Files (*.tif *.tiff *.nd2 *.lsm *.png *.jpg);;All Files (*)"
        )

        if file_paths:
            for file_path in file_paths:
                try:
                    success = self.data_manager.load_file(file_path)
                    if success:
                        self.logger.info(f"Successfully loaded image: {file_path}")
                        self._add_to_recent_files(file_path)
                    else:
                        self.logger.error(f"Failed to load image: {file_path}")
                except Exception as e:
                    self.logger.error(f"Error loading image {file_path}: {e}")

    def _import_trajectories(self):
        """Import trajectory data files."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Import Trajectory Files", "",
            "Data Files (*.csv *.xlsx *.txt *.json);;All Files (*)"
        )

        if file_paths:
            for file_path in file_paths:
                try:
                    success = self.data_manager.load_file(file_path)
                    if success:
                        self.logger.info(f"Successfully loaded trajectories: {file_path}")
                        self._add_to_recent_files(file_path)
                    else:
                        self.logger.error(f"Failed to load trajectories: {file_path}")
                except Exception as e:
                    self.logger.error(f"Error loading trajectories {file_path}: {e}")

    def _import_folder(self):
        """Import entire folder of data files."""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Data Folder")

        if folder_path:
            try:
                from pathlib import Path
                folder = Path(folder_path)

                # Find supported files
                patterns = ['*.tif', '*.tiff', '*.nd2', '*.csv', '*.xlsx']
                files_found = []

                for pattern in patterns:
                    files_found.extend(folder.glob(pattern))
                    files_found.extend(folder.glob(f"**/{pattern}"))  # Recursive

                if files_found:
                    loaded_count = 0
                    for file_path in files_found[:20]:  # Limit to first 20 files
                        try:
                            success = self.data_manager.load_file(str(file_path))
                            if success:
                                loaded_count += 1
                        except Exception as e:
                            self.logger.warning(f"Could not load {file_path}: {e}")

                    self.logger.info(f"Loaded {loaded_count} files from folder: {folder_path}")
                    QMessageBox.information(self, "Import Complete",
                                          f"Successfully loaded {loaded_count} files from the selected folder.")
                else:
                    QMessageBox.warning(self, "No Files Found",
                                      "No supported data files found in the selected folder.")

            except Exception as e:
                self.logger.error(f"Error importing folder {folder_path}: {e}")
                QMessageBox.critical(self, "Import Error", f"Error importing folder: {e}")

    def _add_to_recent_files(self, file_path):
        """Add file to recent files list."""
        try:
            # Add to recent files list widget
            item_text = f"{Path(file_path).name} - {Path(file_path).parent}"

            # Check if already in list
            for i in range(self.recent_files_list.count()):
                if self.recent_files_list.item(i).text() == item_text:
                    return  # Already in list

            # Add to top of list
            self.recent_files_list.insertItem(0, item_text)

            # Limit to 10 recent files
            while self.recent_files_list.count() > 10:
                self.recent_files_list.takeItem(self.recent_files_list.count() - 1)

        except Exception as e:
            self.logger.warning(f"Error updating recent files: {e}")

    def _clear_recent_files(self):
        """Clear recent files list."""
        self.recent_files_list.clear()

    def _cleanup_memory(self):
        """Clean up memory usage."""
        try:
            import gc
            gc.collect()
            self._update_memory_display()
            self.logger.info("Memory cleanup completed")
            QMessageBox.information(self, "Memory Cleanup", "Memory cleanup completed successfully.")
        except Exception as e:
            self.logger.error(f"Error during memory cleanup: {e}")

    def _update_memory_display(self):
        """Update memory usage display."""
        try:
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)
                self.memory_label.setText(f"Memory: {memory_mb:.1f} MB")
            except ImportError:
                # Fallback if psutil not available
                import resource
                memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # On Linux/Mac, this is in KB; on some systems it might be different
                memory_mb = memory_kb / 1024
                self.memory_label.setText(f"Memory: {memory_mb:.1f} MB")
        except Exception:
            self.memory_label.setText("Memory: N/A")

    def _new_project(self):
        """Create new project."""
        try:
            if hasattr(self.project_manager, 'new_project'):
                self.project_manager.new_project()
                self.logger.info("New project created")
                self.project_label.setText("New Project")
            else:
                QMessageBox.information(self, "New Project", "New project functionality will be implemented soon.")
        except Exception as e:
            self.logger.error(f"Error creating new project: {e}")

    def _open_project(self):
        """Open existing project."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Project", "", "Project Files (*.json *.xml);;All Files (*)"
            )
            if file_path and hasattr(self.project_manager, 'load_project'):
                self.project_manager.load_project(file_path)
                self.logger.info(f"Project opened: {file_path}")
                self.project_label.setText(f"Project: {Path(file_path).stem}")
            elif file_path:
                QMessageBox.information(self, "Open Project", "Project loading functionality will be implemented soon.")
        except Exception as e:
            self.logger.error(f"Error opening project: {e}")

    def _save_project(self):
        """Save current project."""
        try:
            if hasattr(self.project_manager, 'save_project'):
                self.project_manager.save_project()
                self.logger.info("Project saved")
            else:
                QMessageBox.information(self, "Save Project", "Project saving functionality will be implemented soon.")
        except Exception as e:
            self.logger.error(f"Error saving project: {e}")

    def _save_project_as(self):
        """Save project with new name."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Project As", "", "Project Files (*.json);;All Files (*)"
            )
            if file_path and hasattr(self.project_manager, 'save_project_as'):
                self.project_manager.save_project_as(file_path)
                self.logger.info(f"Project saved as: {file_path}")
                self.project_label.setText(f"Project: {Path(file_path).stem}")
            elif file_path:
                QMessageBox.information(self, "Save Project As", "Project saving functionality will be implemented soon.")
        except Exception as e:
            self.logger.error(f"Error saving project as: {e}")


class AnalysisPresetsTab(QWidget):
    """Dedicated tab for analysis presets - clean and simple."""

    def __init__(self, analysis_engine, data_manager, parent=None):
        super().__init__(parent)
        self.analysis_engine = analysis_engine
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the analysis presets interface - focused and uncluttered."""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)

        # Title and description
        title_label = QLabel("🚀 Analysis Presets")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #2c3e50; margin-bottom: 10px;")
        layout.addWidget(title_label)

        description_label = QLabel("Choose a preset to quickly analyze your data with optimized parameters")
        description_label.setStyleSheet("font-size: 14px; color: #7f8c8d; margin-bottom: 20px;")
        layout.addWidget(description_label)

        # Data selection
        data_section = self._create_data_selection_section()
        layout.addWidget(data_section)

        # Preset buttons in a grid
        presets_section = self._create_presets_section()
        layout.addWidget(presets_section)

        # Progress and status
        progress_section = self._create_progress_section()
        layout.addWidget(progress_section)

        layout.addStretch()

    def _create_data_selection_section(self):
        """Create data selection section."""
        group = QGroupBox("📊 Select Data to Analyze")
        group.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; }")
        layout = QVBoxLayout(group)

        # Data combo with larger font
        self.data_combo = QComboBox()
        self.data_combo.setMinimumHeight(40)
        self.data_combo.setStyleSheet("font-size: 14px; padding: 8px;")
        layout.addWidget(self.data_combo)

        # Data info display
        self.data_info_label = QLabel("No data selected")
        self.data_info_label.setStyleSheet("color: #7f8c8d; margin-top: 10px;")
        layout.addWidget(self.data_info_label)

        return group

    def _create_presets_section(self):
        """Create analysis presets section with large, clear buttons."""
        group = QGroupBox("🎯 Analysis Presets")
        group.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; }")
        layout = QGridLayout(group)
        layout.setSpacing(15)

        # Define presets with descriptions
        presets = [
            {
                'name': '🚀 Quick Analysis',
                'description': 'Fast detection → linking → basic features\nBest for: Initial data exploration (2-5 minutes)',
                'color': '#3498db'
            },
            {
                'name': '🔬 Comprehensive Analysis',
                'description': 'Full pipeline with all enhanced features\nBest for: Complete analysis (5-15 minutes)',
                'color': '#e74c3c'
            },
            {
                'name': '🏃 Mobility-Focused',
                'description': 'Enhanced mobility and diffusion metrics\nBest for: Mobility classification studies',
                'color': '#f39c12'
            },
            {
                'name': '📐 Shape-Focused',
                'description': 'Advanced shape and geometry analysis\nBest for: Morphology studies',
                'color': '#9b59b6'
            }
        ]

        for i, preset in enumerate(presets):
            button = self._create_preset_button(preset)
            row = i // 2
            col = i % 2
            layout.addWidget(button, row, col)

        return group

    def _create_preset_button(self, preset_info):
        """Create a large, informative preset button."""
        button = QPushButton()
        button.setMinimumHeight(120)
        button.setMinimumWidth(300)

        # Create rich text content
        content = f"""
        <div style='text-align: center; padding: 10px;'>
            <h3 style='margin: 0; color: white; font-size: 16px;'>{preset_info['name']}</h3>
            <p style='margin: 10px 0 0 0; color: white; font-size: 12px; line-height: 1.4;'>
                {preset_info['description']}
            </p>
        </div>
        """

        button.setText(content)
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {preset_info['color']};
                border: none;
                border-radius: 10px;
                color: white;
                font-weight: bold;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: {self._darken_color(preset_info['color'])};
                transform: scale(1.05);
            }}
            QPushButton:pressed {{
                background-color: {self._darken_color(preset_info['color'])};
            }}
        """)

        # Store preset type for signal connection
        preset_type = preset_info['name'].split(' ')[1].lower()  # quick, comprehensive, etc.
        button.preset_type = preset_type

        return button

    def _darken_color(self, hex_color):
        """Darken a hex color for hover effects."""
        # Simple darkening by reducing each RGB component
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        darkened = tuple(max(0, int(c * 0.8)) for c in rgb)
        return f"#{darkened[0]:02x}{darkened[1]:02x}{darkened[2]:02x}"

    def _create_progress_section(self):
        """Create progress and status section."""
        group = QGroupBox("📈 Analysis Status")
        group.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; }")
        layout = QVBoxLayout(group)

        # Status label
        self.status_label = QLabel("Ready to analyze")
        self.status_label.setStyleSheet("font-size: 14px; margin-bottom: 10px;")
        layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(25)
        layout.addWidget(self.progress_bar)

        # Control buttons
        button_layout = QHBoxLayout()

        self.stop_btn = QPushButton("⏹️ Stop Analysis")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumHeight(35)
        self.stop_btn.setStyleSheet("background-color: #e74c3c; color: white; border: none; border-radius: 5px; font-weight: bold;")
        button_layout.addWidget(self.stop_btn)

        button_layout.addStretch()

        self.view_results_btn = QPushButton("👁️ View Results")
        self.view_results_btn.setMinimumHeight(35)
        self.view_results_btn.setStyleSheet("background-color: #27ae60; color: white; border: none; border-radius: 5px; font-weight: bold;")
        button_layout.addWidget(self.view_results_btn)

        layout.addLayout(button_layout)

        return group

    def _connect_signals(self):
        """Connect preset signals."""
        # Data selection
        self.data_combo.currentTextChanged.connect(self._on_data_selected)

        # Find and connect preset buttons
        for button in self.findChildren(QPushButton):
            if hasattr(button, 'preset_type'):
                button.clicked.connect(lambda checked, b=button: self._run_preset_analysis(b.preset_type))

        # Control buttons
        self.stop_btn.clicked.connect(self._stop_analysis)
        self.view_results_btn.clicked.connect(self._view_results)

        # Update data list when data manager changes
        if hasattr(self.data_manager, 'dataLoaded'):
            self.data_manager.dataLoaded.connect(self._update_data_list)
        if hasattr(self.data_manager, 'dataRemoved'):
            self.data_manager.dataRemoved.connect(self._update_data_list)

        # Connect analysis engine signals
        if hasattr(self.analysis_engine, 'analysisStarted'):
            self.analysis_engine.analysisStarted.connect(self._on_analysis_started)
        if hasattr(self.analysis_engine, 'analysisCompleted'):
            self.analysis_engine.analysisCompleted.connect(self._on_analysis_completed)
        if hasattr(self.analysis_engine, 'progressUpdate'):
            self.analysis_engine.progressUpdate.connect(self._update_progress)

        # Initial data list update
        self._update_data_list()

    def _update_data_list(self):
        """Update the data selection combo box."""
        try:
            current_selection = self.data_combo.currentText()

            self.data_combo.clear()

            if hasattr(self.data_manager, 'get_data_names'):
                data_names = self.data_manager.get_data_names()
                self.data_combo.addItems(data_names)

                # Restore selection if possible
                if current_selection in data_names:
                    self.data_combo.setCurrentText(current_selection)

                self.logger.debug(f"Updated data list with {len(data_names)} items")
            else:
                self.logger.warning("Data manager does not have get_data_names method")

        except Exception as e:
            self.logger.error(f"Error updating data list: {e}")

    def _on_data_selected(self, data_name):
        """Handle data selection change."""
        if data_name:
            try:
                if hasattr(self.data_manager, 'get_data'):
                    data = self.data_manager.get_data(data_name)
                    if data is not None:
                        # Update info display
                        if hasattr(data, 'shape'):
                            info_text = f"Image stack: {data.shape} ({data.dtype})"
                        elif hasattr(data, '__len__'):
                            info_text = f"Data: {len(data)} rows"
                        else:
                            info_text = "Data loaded successfully"
                        self.data_info_label.setText(info_text)
                        self.logger.info(f"Selected data: {data_name}")
                    else:
                        self.data_info_label.setText("Could not load data")
                else:
                    self.data_info_label.setText("Data manager not available")
            except Exception as e:
                self.data_info_label.setText(f"Error: {e}")
                self.logger.error(f"Error selecting data: {e}")
        else:
            self.data_info_label.setText("No data selected")

    def _run_preset_analysis(self, preset_type):
        """Run analysis with the selected preset."""
        try:
            # Get selected data
            data_name = self.data_combo.currentText()
            if not data_name:
                QMessageBox.warning(self, "No Data", "Please select data to analyze first.")
                return

            if not hasattr(self.data_manager, 'get_data'):
                QMessageBox.warning(self, "Error", "Data manager not properly initialized.")
                return

            data = self.data_manager.get_data(data_name)
            if data is None:
                QMessageBox.warning(self, "No Data", f"Could not load data: {data_name}")
                return

            # Get analysis steps based on preset
            steps = self._get_preset_steps(preset_type)

            # Get parameters
            parameters = self._get_analysis_parameters()

            # Show confirmation
            preset_names = {
                'quick': 'Quick Analysis',
                'comprehensive': 'Comprehensive Analysis',
                'mobility': 'Mobility-Focused Analysis',
                'shape': 'Shape-Focused Analysis'
            }

            preset_name = preset_names.get(preset_type, preset_type)
            reply = QMessageBox.question(
                self, "Start Analysis",
                f"Start {preset_name} on {data_name}?\n\n"
                f"This will analyze your data using optimized parameters.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                # Run analysis
                if hasattr(self.analysis_engine, 'run_analysis_pipeline'):
                    self.analysis_engine.run_analysis_pipeline(data, parameters, steps)
                    self.logger.info(f"Started {preset_type} analysis on {data_name}")
                else:
                    # Fallback - show what would be analyzed
                    QMessageBox.information(
                        self, "Analysis Started",
                        f"Started {preset_name} on {data_name}\n"
                        f"Steps: {', '.join([step.value if hasattr(step, 'value') else str(step) for step in steps])}\n"
                        f"Analysis engine integration pending."
                    )

        except Exception as e:
            self.logger.error(f"Error running {preset_type} analysis: {e}")
            QMessageBox.critical(self, "Analysis Error", f"Failed to run analysis: {e}")

    def _get_preset_steps(self, preset_type):
        """Get analysis steps for a preset type."""
        if preset_type == "quick":
            return [AnalysisStep.DETECTION, AnalysisStep.LINKING, AnalysisStep.FEATURES, AnalysisStep.CLASSIFICATION]
        elif preset_type == "comprehensive":
            return [AnalysisStep.DETECTION, AnalysisStep.LINKING, AnalysisStep.ENHANCED_FEATURES, AnalysisStep.CLASSIFICATION]
        elif preset_type == "mobility":
            return [AnalysisStep.DETECTION, AnalysisStep.LINKING, AnalysisStep.ENHANCED_FEATURES]
        elif preset_type == "shape":
            return [AnalysisStep.DETECTION, AnalysisStep.LINKING, AnalysisStep.ENHANCED_FEATURES]
        else:
            return [AnalysisStep.DETECTION, AnalysisStep.LINKING, AnalysisStep.FEATURES]

    def _get_analysis_parameters(self):
        """Get current analysis parameters."""
        try:
            if hasattr(self.parameter_manager, 'get_all_parameters'):
                return self.parameter_manager.get_all_parameters()
            else:
                # Return default parameters
                return AnalysisParameters()
        except Exception as e:
            self.logger.warning(f"Error getting parameters, using defaults: {e}")
            return AnalysisParameters()

    def _stop_analysis(self):
        """Stop current analysis."""
        try:
            if hasattr(self.analysis_engine, 'stop_analysis'):
                self.analysis_engine.stop_analysis()
                self.logger.info("Analysis stopped by user")
            else:
                QMessageBox.information(self, "Stop Analysis", "Analysis stopping functionality pending.")
        except Exception as e:
            self.logger.error(f"Error stopping analysis: {e}")

    def _view_results(self):
        """Switch to results tab."""
        try:
            # Get the main window and switch to visualization tab
            main_window = self.parent()
            if hasattr(main_window, 'main_tabs'):
                main_window.main_tabs.setCurrentIndex(3)  # Visualization tab (index 3)
        except Exception as e:
            self.logger.error(f"Error switching to results: {e}")

    def _on_analysis_started(self, steps):
        """Handle analysis started."""
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Analysis running...")

        # Disable preset buttons during analysis
        for button in self.findChildren(QPushButton):
            if hasattr(button, 'preset_type'):
                button.setEnabled(False)

    def _on_analysis_completed(self, result):
        """Handle analysis completed."""
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Analysis completed successfully!")

        # Re-enable preset buttons
        for button in self.findChildren(QPushButton):
            if hasattr(button, 'preset_type'):
                button.setEnabled(True)

        QMessageBox.information(self, "Analysis Complete", "Analysis has completed successfully!\n\nClick 'View Results' to see your data.")

    def _update_progress(self, message, percentage):
        """Update analysis progress."""
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)



class VisualizationResultsTab(QWidget):
    """Dedicated tab for visualization and results viewing."""

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the visualization and results interface."""
        layout = QVBoxLayout(self)

        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(main_splitter)

        # Left: Visualization
        viz_panel = self._create_visualization_panel()
        main_splitter.addWidget(viz_panel)

        # Right: Results and Analysis
        results_panel = self._create_results_panel()
        main_splitter.addWidget(results_panel)

        # Set proportions: visualization gets more space
        main_splitter.setSizes([800, 400])

    def _connect_signals(self):
        """Connect visualization signals."""
        # View controls
        self.zoom_fit_btn.clicked.connect(self._zoom_fit)
        self.reset_view_btn.clicked.connect(self._reset_view)
        self.export_view_btn.clicked.connect(self._export_view)

        # Frame controls
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        self.play_btn.clicked.connect(self._toggle_playback)

        # Display options
        self.show_tracks_cb.toggled.connect(self._toggle_tracks)
        self.show_points_cb.toggled.connect(self._toggle_points)
        self.show_ids_cb.toggled.connect(self._toggle_ids)
        self.show_mobility_cb.toggled.connect(self._toggle_mobility)

        # Color and filtering
        self.color_by_combo.currentTextChanged.connect(self._change_color_scheme)
        self.min_length_spin.valueChanged.connect(self._change_min_length)

        # Results controls
        self.refresh_stats_btn.clicked.connect(self._refresh_statistics)
        self.detailed_stats_btn.clicked.connect(self._show_detailed_stats)
        self.export_stats_btn.clicked.connect(self._export_statistics)

        # Data table controls
        self.table_data_combo.currentTextChanged.connect(self._update_table_data)
        self.filter_edit.textChanged.connect(self._filter_table_data)

        # Plot controls
        self.generate_plot_btn.clicked.connect(self._generate_plot)

        # Connect to data manager
        if hasattr(self.data_manager, 'dataLoaded'):
            self.data_manager.dataLoaded.connect(self._on_data_loaded)

    def _setup_ui(self):
        """Setup the visualization and results interface."""
        layout = QVBoxLayout(self)

        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(main_splitter)

        # Left: Visualization
        viz_panel = self._create_visualization_panel()
        main_splitter.addWidget(viz_panel)

        # Right: Results and Analysis
        results_panel = self._create_results_panel()
        main_splitter.addWidget(results_panel)

        # Set proportions: visualization gets more space
        main_splitter.setSizes([800, 400])

    def _create_visualization_panel(self) -> QWidget:
        """Create the main visualization panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Header with controls
        header_layout = QHBoxLayout()

        title = QLabel("📊 Data Visualization")
        title.setFont(QFont("", 14, QFont.Weight.Bold))
        header_layout.addWidget(title)

        header_layout.addStretch()

        # View controls
        self.zoom_fit_btn = QPushButton("🔍 Fit")
        self.reset_view_btn = QPushButton("🏠 Reset")
        self.export_view_btn = QPushButton("📷 Export")

        header_layout.addWidget(self.zoom_fit_btn)
        header_layout.addWidget(self.reset_view_btn)
        header_layout.addWidget(self.export_view_btn)

        layout.addLayout(header_layout)

        # Main visualization widget
        self.visualization = EnhancedVisualizationWidget(self.data_manager)
        layout.addWidget(self.visualization)

        # Simplified control panel at bottom
        controls_panel = self._create_simplified_controls()
        layout.addWidget(controls_panel)

        return panel

    def _create_simplified_controls(self) -> QWidget:
        """Create simplified visualization controls."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        panel.setMaximumHeight(120)

        layout = QHBoxLayout(panel)

        # Frame control
        frame_group = QGroupBox("Frame")
        frame_layout = QVBoxLayout(frame_group)

        frame_control_layout = QHBoxLayout()

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)
        frame_control_layout.addWidget(self.frame_slider)

        self.frame_label = QLabel("0/0")
        self.frame_label.setMinimumWidth(50)
        frame_control_layout.addWidget(self.frame_label)

        self.play_btn = QPushButton("▶️")
        self.play_btn.setMaximumWidth(40)
        frame_control_layout.addWidget(self.play_btn)

        frame_layout.addLayout(frame_control_layout)
        layout.addWidget(frame_group)

        # Display options
        display_group = QGroupBox("Display")
        display_layout = QGridLayout(display_group)

        self.show_tracks_cb = QCheckBox("Tracks")
        self.show_tracks_cb.setChecked(True)
        display_layout.addWidget(self.show_tracks_cb, 0, 0)

        self.show_points_cb = QCheckBox("Points")
        self.show_points_cb.setChecked(True)
        display_layout.addWidget(self.show_points_cb, 0, 1)

        self.show_ids_cb = QCheckBox("IDs")
        display_layout.addWidget(self.show_ids_cb, 1, 0)

        self.show_mobility_cb = QCheckBox("Mobility")
        display_layout.addWidget(self.show_mobility_cb, 1, 1)

        layout.addWidget(display_group)

        # Color and filtering
        color_group = QGroupBox("Coloring & Filtering")
        color_layout = QFormLayout(color_group)

        self.color_by_combo = QComboBox()
        self.color_by_combo.addItems(["Track ID", "Mobility", "Velocity", "Scaled Rg"])
        color_layout.addRow("Color by:", self.color_by_combo)

        self.min_length_spin = QSpinBox()
        self.min_length_spin.setRange(1, 1000)
        self.min_length_spin.setValue(3)
        color_layout.addRow("Min length:", self.min_length_spin)

        layout.addWidget(color_group)

        return panel

    def _create_results_panel(self) -> QWidget:
        """Create the results and analysis panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Header
        title = QLabel("📈 Results & Analysis")
        title.setFont(QFont("", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        # Results tabs
        self.results_tabs = QTabWidget()

        # Statistics tab
        stats_tab = self._create_statistics_tab()
        self.results_tabs.addTab(stats_tab, "📊 Statistics")

        # Data table tab
        table_tab = self._create_data_table_tab()
        self.results_tabs.addTab(table_tab, "📋 Data")

        # Feature plots tab
        plots_tab = self._create_plots_tab()
        self.results_tabs.addTab(plots_tab, "📈 Plots")

        layout.addWidget(self.results_tabs)

        return panel

    def _create_statistics_tab(self) -> QWidget:
        """Create statistics display tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Statistics display
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(QFont("Courier", 10))
        layout.addWidget(self.stats_text)

        # Statistics controls
        controls_layout = QHBoxLayout()

        self.refresh_stats_btn = QPushButton("🔄 Refresh")
        controls_layout.addWidget(self.refresh_stats_btn)

        self.detailed_stats_btn = QPushButton("📊 Detailed")
        controls_layout.addWidget(self.detailed_stats_btn)

        self.export_stats_btn = QPushButton("📁 Export")
        controls_layout.addWidget(self.export_stats_btn)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        return tab

    def _create_data_table_tab(self) -> QWidget:
        """Create data table display tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Table controls
        controls_layout = QHBoxLayout()

        self.table_data_combo = QComboBox()
        controls_layout.addWidget(QLabel("Data:"))
        controls_layout.addWidget(self.table_data_combo)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter...")
        controls_layout.addWidget(self.filter_edit)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Results table
        self.results_table = QTableView()
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSortingEnabled(True)
        layout.addWidget(self.results_table)

        return tab

    def _create_plots_tab(self) -> QWidget:
        """Create feature plots tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Plot controls
        controls_layout = QHBoxLayout()

        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Histogram", "Scatter", "Box Plot", "Violin Plot"])
        controls_layout.addWidget(QLabel("Plot type:"))
        controls_layout.addWidget(self.plot_type_combo)

        self.x_feature_combo = QComboBox()
        controls_layout.addWidget(QLabel("X:"))
        controls_layout.addWidget(self.x_feature_combo)

        self.y_feature_combo = QComboBox()
        controls_layout.addWidget(QLabel("Y:"))
        controls_layout.addWidget(self.y_feature_combo)

        self.generate_plot_btn = QPushButton("📈 Generate Plot")
        controls_layout.addWidget(self.generate_plot_btn)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Plot area placeholder
        self.plot_area = QLabel("Select features and click 'Generate Plot' to create visualizations")
        self.plot_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plot_area.setStyleSheet("QLabel { border: 1px dashed gray; min-height: 300px; }")
        layout.addWidget(self.plot_area)

        return tab

    # Implementation methods for visualization functionality
    def _zoom_fit(self):
        """Zoom to fit all data."""
        try:
            if hasattr(self.visualization, 'zoom_fit'):
                self.visualization.zoom_fit()
                self.logger.debug("Zoomed to fit")
        except Exception as e:
            self.logger.error(f"Error zooming to fit: {e}")

    def _reset_view(self):
        """Reset view to default."""
        try:
            if hasattr(self.visualization, 'reset_view'):
                self.visualization.reset_view()
                self.logger.debug("View reset")
        except Exception as e:
            self.logger.error(f"Error resetting view: {e}")

    def _export_view(self):
        """Export current visualization."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Visualization", "",
                "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)"
            )
            if file_path:
                if hasattr(self.visualization, 'export_current_view'):
                    self.visualization.export_current_view(file_path)
                    self.logger.info(f"View exported to {file_path}")
                    QMessageBox.information(self, "Export Complete", f"View exported to {file_path}")
                else:
                    QMessageBox.information(self, "Export View", "Export functionality will be implemented soon.")
        except Exception as e:
            self.logger.error(f"Error exporting view: {e}")

    def _on_frame_changed(self, frame):
        """Handle frame slider change."""
        try:
            if hasattr(self.visualization, 'set_frame'):
                self.visualization.set_frame(frame)
                max_frame = self.frame_slider.maximum()
                self.frame_label.setText(f"{frame}/{max_frame}")
        except Exception as e:
            self.logger.error(f"Error changing frame: {e}")

    def _toggle_playback(self):
        """Toggle animation playback."""
        # Placeholder for playback functionality
        current_text = self.play_btn.text()
        if "▶️" in current_text:
            self.play_btn.setText("⏸️")
            self.logger.debug("Started playback")
        else:
            self.play_btn.setText("▶️")
            self.logger.debug("Paused playback")

    def _toggle_tracks(self, enabled):
        """Toggle track display."""
        try:
            if hasattr(self.visualization, 'show_tracks'):
                self.visualization.show_tracks = enabled
                if hasattr(self.visualization, '_update_tracking_display'):
                    self.visualization._update_tracking_display()
                self.logger.debug(f"Tracks display: {enabled}")
        except Exception as e:
            self.logger.error(f"Error toggling tracks: {e}")

    def _toggle_points(self, enabled):
        """Toggle point display."""
        try:
            if hasattr(self.visualization, 'show_localizations'):
                self.visualization.show_localizations = enabled
                if hasattr(self.visualization, '_update_tracking_display'):
                    self.visualization._update_tracking_display()
                self.logger.debug(f"Points display: {enabled}")
        except Exception as e:
            self.logger.error(f"Error toggling points: {e}")

    def _toggle_ids(self, enabled):
        """Toggle ID display."""
        try:
            if hasattr(self.visualization, 'show_track_ids'):
                self.visualization.show_track_ids = enabled
                if hasattr(self.visualization, '_update_tracking_display'):
                    self.visualization._update_tracking_display()
                self.logger.debug(f"IDs display: {enabled}")
        except Exception as e:
            self.logger.error(f"Error toggling IDs: {e}")

    def _toggle_mobility(self, enabled):
        """Toggle mobility overlay."""
        try:
            if hasattr(self.visualization, 'show_mobility_overlay'):
                self.visualization.show_mobility_overlay = enabled
                if hasattr(self.visualization, '_update_tracking_display'):
                    self.visualization._update_tracking_display()
                self.logger.debug(f"Mobility overlay: {enabled}")
        except Exception as e:
            self.logger.error(f"Error toggling mobility: {e}")

    def _change_color_scheme(self, scheme):
        """Change color scheme."""
        try:
            if hasattr(self.visualization, '_change_color_scheme'):
                self.visualization._change_color_scheme(scheme)
                self.logger.debug(f"Color scheme changed to: {scheme}")
        except Exception as e:
            self.logger.error(f"Error changing color scheme: {e}")

    def _change_min_length(self, min_length):
        """Change minimum track length filter."""
        try:
            if hasattr(self.visualization, 'min_track_length'):
                self.visualization.min_track_length = min_length
                if hasattr(self.visualization, '_update_tracking_display'):
                    self.visualization._update_tracking_display()
                self.logger.debug(f"Min track length: {min_length}")
        except Exception as e:
            self.logger.error(f"Error changing min length: {e}")

    def _on_data_loaded(self, data_name, data):
        """Handle new data being loaded."""
        try:
            self.logger.info(f"New data loaded: {data_name}")

            # Update data combo boxes
            self._update_data_combos()

            # Update visualization
            if isinstance(data, np.ndarray):
                if hasattr(self.visualization, 'set_image_data'):
                    self.visualization.set_image_data(data)

                    # Update frame slider for image stacks
                    if len(data.shape) == 3:
                        self.frame_slider.setMaximum(data.shape[0] - 1)
                        self.frame_slider.setValue(0)
                        self._on_frame_changed(0)

            elif isinstance(data, pd.DataFrame):
                if hasattr(self.visualization, 'set_tracking_data'):
                    self.visualization.set_tracking_data(data)

                # Update statistics display
                self._update_statistics_display(data)

        except Exception as e:
            self.logger.error(f"Error handling loaded data: {e}")

    def _update_data_combos(self):
        """Update data selection combo boxes."""
        try:
            if hasattr(self.data_manager, 'get_data_names'):
                data_names = self.data_manager.get_data_names()

                # Update table data combo
                current_table = self.table_data_combo.currentText()
                self.table_data_combo.clear()
                self.table_data_combo.addItems(data_names)
                if current_table in data_names:
                    self.table_data_combo.setCurrentText(current_table)

                # Update feature combo boxes
                for combo in [self.x_feature_combo, self.y_feature_combo]:
                    current_feature = combo.currentText()
                    combo.clear()
                    # Add common features
                    combo.addItems([
                        "x", "y", "frame", "track_number", "radius_gyration",
                        "velocity", "scaled_rg", "mobility_classification"
                    ])
                    if current_feature:
                        index = combo.findText(current_feature)
                        if index >= 0:
                            combo.setCurrentIndex(index)

        except Exception as e:
            self.logger.error(f"Error updating data combos: {e}")

    def _update_statistics_display(self, data):
        """Update statistics display with data info."""
        try:
            stats_text = f"Dataset Statistics\n{'='*50}\n\n"

            if isinstance(data, pd.DataFrame):
                stats_text += f"Total rows: {len(data)}\n"
                stats_text += f"Columns: {len(data.columns)}\n\n"

                if 'track_number' in data.columns:
                    n_tracks = data['track_number'].nunique()
                    track_lengths = data.groupby('track_number').size()
                    stats_text += f"Number of tracks: {n_tracks}\n"
                    stats_text += f"Mean track length: {track_lengths.mean():.1f}\n"
                    stats_text += f"Median track length: {track_lengths.median():.1f}\n"
                    stats_text += f"Track length range: {track_lengths.min()}-{track_lengths.max()}\n\n"

                if 'frame' in data.columns:
                    n_frames = data['frame'].nunique()
                    stats_text += f"Number of frames: {n_frames}\n\n"

                # Classification statistics
                for col in ['mobility_classification', 'linear_classification']:
                    if col in data.columns:
                        counts = data.groupby('track_number')[col].first().value_counts()
                        stats_text += f"{col.replace('_', ' ').title()}:\n"
                        for classification, count in counts.items():
                            pct = (count / len(counts)) * 100
                            stats_text += f"  {classification}: {count} ({pct:.1f}%)\n"
                        stats_text += "\n"

            elif isinstance(data, np.ndarray):
                stats_text += f"Array shape: {data.shape}\n"
                stats_text += f"Data type: {data.dtype}\n"
                stats_text += f"Min value: {data.min():.2f}\n"
                stats_text += f"Max value: {data.max():.2f}\n"
                stats_text += f"Mean value: {data.mean():.2f}\n"

            self.stats_text.setText(stats_text)

        except Exception as e:
            self.logger.error(f"Error updating statistics display: {e}")

    def _refresh_statistics(self):
        """Refresh statistics display."""
        try:
            data_name = self.table_data_combo.currentText()
            if data_name and hasattr(self.data_manager, 'get_data'):
                data = self.data_manager.get_data(data_name)
                if data is not None:
                    self._update_statistics_display(data)
                    self.logger.info("Statistics refreshed")
        except Exception as e:
            self.logger.error(f"Error refreshing statistics: {e}")

    def _show_detailed_stats(self):
        """Show detailed statistics in a separate window."""
        QMessageBox.information(self, "Detailed Statistics", "Detailed statistics view will be implemented soon.")

    def _export_statistics(self):
        """Export statistics to file."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Statistics", "", "Text Files (*.txt);;All Files (*)"
            )
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(self.stats_text.toPlainText())
                self.logger.info(f"Statistics exported to {file_path}")
                QMessageBox.information(self, "Export Complete", f"Statistics exported to {file_path}")
        except Exception as e:
            self.logger.error(f"Error exporting statistics: {e}")

    def _update_table_data(self, data_name):
        """Update table display with selected data."""
        try:
            if data_name and hasattr(self.data_manager, 'get_data'):
                data = self.data_manager.get_data(data_name)
                if isinstance(data, pd.DataFrame):
                    # Create a simple table model
                    from PyQt6.QtCore import QAbstractTableModel, QVariant

                    class PandasModel(QAbstractTableModel):
                        def __init__(self, data):
                            super().__init__()
                            self._data = data

                        def rowCount(self, parent=None):
                            return len(self._data)

                        def columnCount(self, parent=None):
                            return len(self._data.columns)

                        def data(self, index, role=Qt.ItemDataRole.DisplayRole):
                            if role == Qt.ItemDataRole.DisplayRole:
                                value = self._data.iloc[index.row(), index.column()]
                                return str(value)
                            return QVariant()

                        def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
                            if role == Qt.ItemDataRole.DisplayRole:
                                if orientation == Qt.Orientation.Horizontal:
                                    return str(self._data.columns[section])
                                else:
                                    return str(section)
                            return QVariant()

                    # Limit to first 1000 rows for performance
                    display_data = data.head(1000)
                    model = PandasModel(display_data)
                    self.results_table.setModel(model)

                    self.logger.debug(f"Table updated with {len(display_data)} rows")

        except Exception as e:
            self.logger.error(f"Error updating table data: {e}")

    def _filter_table_data(self, filter_text):
        """Filter table data based on text."""
        # Placeholder for table filtering
        self.logger.debug(f"Table filter: {filter_text}")

    def _generate_plot(self):
        """Generate feature plot."""
        QMessageBox.information(self, "Generate Plot", "Feature plotting will be implemented soon.")


class BatchProcessingTab(QWidget):
    """Dedicated tab for batch processing and automation."""

    def __init__(self, analysis_engine, data_manager, parent=None):
        super().__init__(parent)
        self.analysis_engine = analysis_engine
        self.data_manager = data_manager
        self.batch_manager = BatchAnalysisManager(analysis_engine, data_manager)
        self.logger = logging.getLogger(__name__)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the batch processing interface."""
        layout = QHBoxLayout(self)

        # Create splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left: Experiment Management
        exp_panel = self._create_experiment_panel()
        splitter.addWidget(exp_panel)

        # Right: Batch Execution and Results
        exec_panel = self._create_execution_panel()
        splitter.addWidget(exec_panel)

        # Set proportions
        splitter.setSizes([500, 500])

    def _create_experiment_panel(self) -> QWidget:
        """Create experiment management panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Header
        title = QLabel("📦 Batch Experiments")
        title.setFont(QFont("", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        # Experiment list
        exp_group = QGroupBox("Current Experiments")
        exp_layout = QVBoxLayout(exp_group)

        self.experiments_list = QComboBox()
        self.experiments_list.setMinimumHeight(35)
        exp_layout.addWidget(self.experiments_list)

        # Experiment management buttons
        exp_buttons = QGridLayout()

        self.new_exp_btn = QPushButton("🆕 New Experiment")
        self.new_exp_btn.setMinimumHeight(40)
        exp_buttons.addWidget(self.new_exp_btn, 0, 0)

        self.load_exp_btn = QPushButton("📂 Load Experiment")
        self.load_exp_btn.setMinimumHeight(40)
        exp_buttons.addWidget(self.load_exp_btn, 0, 1)

        self.save_exp_btn = QPushButton("💾 Save Experiment")
        self.save_exp_btn.setMinimumHeight(40)
        exp_buttons.addWidget(self.save_exp_btn, 1, 0)

        self.duplicate_exp_btn = QPushButton("📋 Duplicate")
        self.duplicate_exp_btn.setMinimumHeight(40)
        exp_buttons.addWidget(self.duplicate_exp_btn, 1, 1)

        exp_layout.addLayout(exp_buttons)
        layout.addWidget(exp_group)

        # File management
        files_group = QGroupBox("📁 Files & Conditions")
        files_layout = QVBoxLayout(files_group)

        # File list (simplified view)
        self.files_summary = QTextEdit()
        self.files_summary.setMaximumHeight(150)
        self.files_summary.setReadOnly(True)
        files_layout.addWidget(self.files_summary)

        # File management buttons
        file_buttons = QHBoxLayout()

        self.add_files_btn = QPushButton("➕ Add Files")
        file_buttons.addWidget(self.add_files_btn)

        self.add_folder_btn = QPushButton("📁 Add Folder")
        file_buttons.addWidget(self.add_folder_btn)

        self.manage_files_btn = QPushButton("⚙️ Manage")
        file_buttons.addWidget(self.manage_files_btn)

        files_layout.addLayout(file_buttons)
        layout.addWidget(files_group)

        # Parameter templates
        template_group = QGroupBox("🎯 Parameter Templates")
        template_layout = QVBoxLayout(template_group)

        self.template_combo = QComboBox()
        self.template_combo.addItems([
            "Quick Analysis Template",
            "Comprehensive Template",
            "Mobility-Focused Template",
            "Shape Analysis Template"
        ])
        template_layout.addWidget(self.template_combo)

        template_buttons = QHBoxLayout()

        self.apply_template_btn = QPushButton("Apply Template")
        template_buttons.addWidget(self.apply_template_btn)

        self.save_template_btn = QPushButton("Save Template")
        template_buttons.addWidget(self.save_template_btn)

        template_layout.addLayout(template_buttons)
        layout.addWidget(template_group)

        layout.addStretch()
        return panel

    def _create_execution_panel(self) -> QWidget:
        """Create batch execution and monitoring panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Header
        title = QLabel("🚀 Batch Execution")
        title.setFont(QFont("", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        # Execution controls
        exec_group = QGroupBox("Execution Control")
        exec_layout = QVBoxLayout(exec_group)

        # Main execution buttons
        main_buttons = QHBoxLayout()

        self.run_batch_btn = QPushButton("▶️ Run Batch Analysis")
        self.run_batch_btn.setMinimumHeight(50)
        self.run_batch_btn.setStyleSheet("QPushButton { background-color: #28a745; color: white; font-weight: bold; }")
        main_buttons.addWidget(self.run_batch_btn)

        self.stop_batch_btn = QPushButton("⏹️ Stop Batch")
        self.stop_batch_btn.setMinimumHeight(50)
        self.stop_batch_btn.setStyleSheet("QPushButton { background-color: #dc3545; color: white; }")
        self.stop_batch_btn.setEnabled(False)
        main_buttons.addWidget(self.stop_batch_btn)

        exec_layout.addLayout(main_buttons)

        # Progress monitoring
        self.batch_progress = QProgressBar()
        self.batch_progress.setVisible(False)
        exec_layout.addWidget(self.batch_progress)

        self.batch_status = QLabel("No batch analysis running")
        self.batch_status.setStyleSheet("QLabel { color: gray; font-size: 11px; }")
        exec_layout.addWidget(self.batch_status)

        layout.addWidget(exec_group)

        # Monitoring and results
        monitor_group = QGroupBox("📊 Progress Monitoring")
        monitor_layout = QVBoxLayout(monitor_group)

        # Progress details
        self.progress_details = QTextEdit()
        self.progress_details.setMaximumHeight(200)
        self.progress_details.setReadOnly(True)
        monitor_layout.addWidget(self.progress_details)

        # Quick actions
        quick_actions = QHBoxLayout()

        self.view_results_btn = QPushButton("👁️ View Results")
        quick_actions.addWidget(self.view_results_btn)

        self.export_summary_btn = QPushButton("📁 Export Summary")
        quick_actions.addWidget(self.export_summary_btn)

        self.open_output_btn = QPushButton("📂 Open Output Folder")
        quick_actions.addWidget(self.open_output_btn)

        monitor_layout.addLayout(quick_actions)
        layout.addWidget(monitor_group)

        # Special analysis tools
        tools_group = QGroupBox("🛠️ Special Analysis Tools")
        tools_layout = QVBoxLayout(tools_group)

        self.autocorr_btn = QPushButton("📈 Autocorrelation Analysis")
        self.autocorr_btn.setMinimumHeight(35)
        self.autocorr_btn.setToolTip("Run direction autocorrelation analysis")
        tools_layout.addWidget(self.autocorr_btn)

        self.comparison_btn = QPushButton("⚖️ Condition Comparison")
        self.comparison_btn.setMinimumHeight(35)
        self.comparison_btn.setToolTip("Statistical comparison between conditions")
        tools_layout.addWidget(self.comparison_btn)

        self.quality_check_btn = QPushButton("✅ Quality Assessment")
        self.quality_check_btn.setMinimumHeight(35)
        self.quality_check_btn.setToolTip("Assess data quality across the batch")
        tools_layout.addWidget(self.quality_check_btn)

        layout.addWidget(tools_group)

        layout.addStretch()
        return panel


class ExportReportsTab(QWidget):
    """Dedicated tab for export and report generation."""

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the export and reports interface."""
        layout = QHBoxLayout(self)

        # Create splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left: Export Options
        export_panel = self._create_export_panel()
        splitter.addWidget(export_panel)

        # Right: Report Generation
        report_panel = self._create_report_panel()
        splitter.addWidget(report_panel)

        # Set proportions
        splitter.setSizes([500, 500])

    def _create_export_panel(self) -> QWidget:
        """Create data export options panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Header
        title = QLabel("📁 Data Export")
        title.setFont(QFont("", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        # Data selection
        data_group = QGroupBox("Select Data to Export")
        data_layout = QVBoxLayout(data_group)

        self.export_data_combo = QComboBox()
        self.export_data_combo.setMinimumHeight(35)
        data_layout.addWidget(self.export_data_combo)

        # Data preview
        self.data_preview = QTextEdit()
        self.data_preview.setMaximumHeight(100)
        self.data_preview.setReadOnly(True)
        data_layout.addWidget(self.data_preview)

        layout.addWidget(data_group)

        # Export formats
        format_group = QGroupBox("📄 Export Formats")
        format_layout = QVBoxLayout(format_group)

        self.export_csv_cb = QCheckBox("📊 CSV Files (Data tables)")
        self.export_csv_cb.setChecked(True)
        format_layout.addWidget(self.export_csv_cb)

        self.export_excel_cb = QCheckBox("📈 Excel Workbook (Multiple sheets)")
        format_layout.addWidget(self.export_excel_cb)

        self.export_plots_cb = QCheckBox("📊 Analysis Plots (PNG/PDF)")
        format_layout.addWidget(self.export_plots_cb)

        self.export_summary_cb = QCheckBox("📋 Summary Statistics (TXT)")
        self.export_summary_cb.setChecked(True)
        format_layout.addWidget(self.export_summary_cb)

        layout.addWidget(format_group)

        # Export controls
        controls_group = QGroupBox("🎯 Export Controls")
        controls_layout = QVBoxLayout(controls_group)

        # Output directory
        output_layout = QHBoxLayout()

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select output directory...")
        output_layout.addWidget(self.output_dir_edit)

        self.browse_output_btn = QPushButton("📂 Browse")
        output_layout.addWidget(self.browse_output_btn)

        controls_layout.addLayout(output_layout)

        # Export buttons
        export_buttons = QGridLayout()

        self.export_current_btn = QPushButton("📁 Export Current Data")
        self.export_current_btn.setMinimumHeight(40)
        export_buttons.addWidget(self.export_current_btn, 0, 0)

        self.export_all_btn = QPushButton("📁 Export All Data")
        self.export_all_btn.setMinimumHeight(40)
        export_buttons.addWidget(self.export_all_btn, 0, 1)

        self.export_selected_btn = QPushButton("📁 Export Selected")
        self.export_selected_btn.setMinimumHeight(40)
        export_buttons.addWidget(self.export_selected_btn, 1, 0, 1, 2)

        controls_layout.addLayout(export_buttons)
        layout.addWidget(controls_group)

        layout.addStretch()
        return panel

    def _create_report_panel(self) -> QWidget:
        """Create report generation panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Header
        title = QLabel("📊 Report Generation")
        title.setFont(QFont("", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        # Report types
        report_group = QGroupBox("Report Types")
        report_layout = QVBoxLayout(report_group)

        self.summary_report_btn = QPushButton("📋 Summary Report")
        self.summary_report_btn.setMinimumHeight(40)
        self.summary_report_btn.setToolTip("Overview of all analysis results")
        report_layout.addWidget(self.summary_report_btn)

        self.detailed_report_btn = QPushButton("📊 Detailed Analysis Report")
        self.detailed_report_btn.setMinimumHeight(40)
        self.detailed_report_btn.setToolTip("Comprehensive analysis with plots and statistics")
        report_layout.addWidget(self.detailed_report_btn)

        self.comparison_report_btn = QPushButton("⚖️ Condition Comparison Report")
        self.comparison_report_btn.setMinimumHeight(40)
        self.comparison_report_btn.setToolTip("Statistical comparison between conditions")
        report_layout.addWidget(self.comparison_report_btn)

        self.quality_report_btn = QPushButton("✅ Quality Assessment Report")
        self.quality_report_btn.setMinimumHeight(40)
        self.quality_report_btn.setToolTip("Data quality and validation report")
        report_layout.addWidget(self.quality_report_btn)

        layout.addWidget(report_group)

        # Report customization
        custom_group = QGroupBox("📝 Report Customization")
        custom_layout = QFormLayout(custom_group)

        self.report_title_edit = QLineEdit()
        self.report_title_edit.setPlaceholderText("Enter report title...")
        custom_layout.addRow("Title:", self.report_title_edit)

        self.report_author_edit = QLineEdit()
        self.report_author_edit.setPlaceholderText("Enter author name...")
        custom_layout.addRow("Author:", self.report_author_edit)

        self.include_plots_cb = QCheckBox("Include plots and visualizations")
        self.include_plots_cb.setChecked(True)
        custom_layout.addRow("", self.include_plots_cb)

        self.include_stats_cb = QCheckBox("Include statistical analysis")
        self.include_stats_cb.setChecked(True)
        custom_layout.addRow("", self.include_stats_cb)

        self.include_methods_cb = QCheckBox("Include methods and parameters")
        custom_layout.addRow("", self.include_methods_cb)

        layout.addWidget(custom_group)

        # Recent exports
        recent_group = QGroupBox("🕒 Recent Exports")
        recent_layout = QVBoxLayout(recent_group)

        self.recent_exports_list = QListWidget()
        self.recent_exports_list.setMaximumHeight(120)
        recent_layout.addWidget(self.recent_exports_list)

        recent_buttons = QHBoxLayout()

        self.open_recent_btn = QPushButton("📂 Open Location")
        recent_buttons.addWidget(self.open_recent_btn)

        self.clear_recent_btn = QPushButton("🧹 Clear List")
        recent_buttons.addWidget(self.clear_recent_btn)

        recent_layout.addLayout(recent_buttons)
        layout.addWidget(recent_group)

        layout.addStretch()
        return panel


class RedesignedEnhancedMainWindow(QMainWindow):
    """Completely redesigned main window with better organization and accessibility."""

    def __init__(self, data_manager, analysis_engine, project_manager, config):
        super().__init__()

        self.logger = logging.getLogger(__name__)

        # Store core components
        self.data_manager = data_manager
        self.analysis_engine = analysis_engine
        self.project_manager = project_manager
        self.config = config

        # Settings
        self.settings = QSettings()

        # Initialize redesigned UI
        self._setup_redesigned_ui()
        self._connect_signals()
        self._restore_settings()

        self.logger.info("Redesigned enhanced main window initialized")

    def _setup_redesigned_ui(self):
        """Setup the completely redesigned user interface."""
        self.setWindowTitle("Enhanced Particle Tracking Analyzer")
        self.setMinimumSize(1600, 1000)  # Larger minimum size

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Create main tab widget - this is the key change
        self.main_tabs = QTabWidget()
        self.main_tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.main_tabs.setDocumentMode(True)
        self.main_tabs.setMovable(True)

        # Create dedicated tabs for major functionality areas

        try:
            # 1. Data Management Tab
            self.data_tab = DataManagementTab(self.data_manager, self.project_manager)
            self.main_tabs.addTab(self.data_tab, "📁 Data Management")
        except Exception as e:
            self.logger.error(f"Failed to create Data Management tab: {e}")
            # Add a placeholder tab
            placeholder = QLabel("Data Management tab failed to load")
            self.main_tabs.addTab(placeholder, "📁 Data Management")



        try:
            # 2: Analysis Presets (NEW - large preset buttons only)
            self.presets_tab = AnalysisPresetsTab(self.analysis_engine, self.data_manager, self)
            self.main_tabs.addTab(self.presets_tab, "🚀 Analysis Presets")
        except Exception as e:
            self.logger.error(f"Failed to create Analysis Presets tab: {e}")
            # Add a placeholder tab
            placeholder = QLabel("Analysis Presets tab failed to load")
            self.main_tabs.addTab(placeholder, "🚀 Analysis Presets")


        try:
            # 3. Analysis Setup Tab
            self.analysis_tab = AnalysisSetupTab(self.analysis_engine, self.data_manager)
            self.main_tabs.addTab(self.analysis_tab, "⚙️ Analysis Setup")
        except Exception as e:
            self.logger.error(f"Failed to create Analysis Setup tab: {e}")
            placeholder = QLabel("Analysis Setup tab failed to load")
            self.main_tabs.addTab(placeholder, "⚙️ Analysis Setup")

        try:
            # 4. Visualization & Results Tab
            self.viz_tab = VisualizationResultsTab(self.data_manager)
            self.main_tabs.addTab(self.viz_tab, "📊 Visualization & Results")
        except Exception as e:
            self.logger.error(f"Failed to create Visualization tab: {e}")
            placeholder = QLabel("Visualization tab failed to load")
            self.main_tabs.addTab(placeholder, "📊 Visualization & Results")

        try:
            # 5. Batch Processing Tab
            self.batch_tab = BatchProcessingTab(self.analysis_engine, self.data_manager)
            self.main_tabs.addTab(self.batch_tab, "📦 Batch Processing")
        except Exception as e:
            self.logger.error(f"Failed to create Batch Processing tab: {e}")
            placeholder = QLabel("Batch Processing tab failed to load")
            self.main_tabs.addTab(placeholder, "📦 Batch Processing")

        try:
            # 6. Export & Reports Tab
            self.export_tab = ExportReportsTab(self.data_manager)
            self.main_tabs.addTab(self.export_tab, "📁 Export & Reports")
        except Exception as e:
            self.logger.error(f"Failed to create Export tab: {e}")
            placeholder = QLabel("Export & Reports tab failed to load")
            self.main_tabs.addTab(placeholder, "📁 Export & Reports")

        layout.addWidget(self.main_tabs)

        # Create streamlined menu bar and status bar
        self._create_streamlined_menu_bar()
        self._create_streamlined_status_bar()

    def _create_streamlined_menu_bar(self):
        """Create a streamlined menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        file_menu.addAction("Open Data...", self._open_data_file)
        file_menu.addAction("Open Project...", self._open_project)
        file_menu.addSeparator()
        file_menu.addAction("Save Project", self._save_project)
        file_menu.addAction("Save Project As...", self._save_project_as)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        # Tools menu
        tools_menu = menubar.addMenu("Tools")

        tools_menu.addAction("Preferences...", self._show_preferences)
        tools_menu.addAction("Reset Layout", self._reset_layout)

        # Help menu
        help_menu = menubar.addMenu("Help")

        help_menu.addAction("User Guide", self._show_user_guide)
        help_menu.addAction("About", self._show_about)

    def _create_streamlined_status_bar(self):
        """Create a streamlined status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Memory usage
        self.memory_label = QLabel("Memory: 0 MB")
        self.memory_label.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
        self.status_bar.addPermanentWidget(self.memory_label)

        # Status message
        self.status_label = QLabel("Ready for enhanced analysis")
        self.status_bar.addWidget(self.status_label)

    def _connect_signals(self):
        """Connect signals between components."""
        # Connect tab change to update window title
        self.main_tabs.currentChanged.connect(self._on_tab_changed)

        # Update tab index references in connect_signals
        try:
            if hasattr(self.data_manager, 'dataLoaded'):
                self.data_manager.dataLoaded.connect(self._on_data_loaded)
                # Forward to visualization tab (now index 3 instead of 2)
                if hasattr(self, 'viz_tab'):
                    self.data_manager.dataLoaded.connect(self.viz_tab._on_data_loaded)
        except Exception as e:
            self.logger.warning(f"Could not connect data manager signals: {e}")

        # Connect analysis engine signals with error handling
        try:
            if hasattr(self.analysis_engine, 'progressUpdate'):
                self.analysis_engine.progressUpdate.connect(self._update_progress)
                # Forward to both analysis tabs
                if hasattr(self, 'presets_tab'):
                    self.analysis_engine.progressUpdate.connect(self.presets_tab._update_progress)
                if hasattr(self, 'analysis_tab'):
                    self.analysis_engine.progressUpdate.connect(self.analysis_tab._update_progress)
        except Exception as e:
            self.logger.warning(f"Could not connect analysis engine signals: {e}")

    def _on_tab_changed(self, index):
        """Handle main tab change."""
        tab_names = [
            "Data Management", "Analysis Setup", "Visualization & Results",
            "Batch Processing", "Export & Reports"
        ]

        if 0 <= index < len(tab_names):
            self.setWindowTitle(f"Enhanced Particle Tracking Analyzer - {tab_names[index]}")

    def _on_data_loaded(self, data_name: str, data: Any):
        """Handle data loading."""
        self.status_label.setText(f"Loaded: {data_name}")

        # Switch to visualization tab if data is loaded
        if isinstance(data, (np.ndarray, pd.DataFrame)):
            self.main_tabs.setCurrentIndex(2)  # Visualization tab

    def _update_progress(self, message: str, percentage: int):
        """Update progress display."""
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)
        self.progress_bar.setVisible(percentage < 100)

    def _restore_settings(self):
        """Restore window settings."""
        self.restoreGeometry(self.settings.value("geometry", b""))
        self.restoreState(self.settings.value("windowState", b""))

        # Restore last active tab
        last_tab = self.settings.value("lastActiveTab", 0, type=int)
        if 0 <= last_tab < self.main_tabs.count():
            self.main_tabs.setCurrentIndex(last_tab)

    def _save_settings(self):
        """Save window settings."""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.settings.setValue("lastActiveTab", self.main_tabs.currentIndex())

    def closeEvent(self, event):
        """Handle window close event."""
        # Stop any running analysis
        if hasattr(self.analysis_engine, 'stop_analysis'):
            self.analysis_engine.stop_analysis()

        self._save_settings()
        event.accept()

    # Implementation methods for menu actions
    def _open_data_file(self):
        """Open data file."""
        file_path, _ = QFileDialog.getOpenFileNames(
            self, "Open Data Files", "",
            "All Supported (*.tif *.tiff *.nd2 *.csv *.xlsx);;Image Files (*.tif *.tiff *.nd2);;Data Files (*.csv *.xlsx);;All Files (*)"
        )
        if file_path and hasattr(self.data_manager, 'load_file'):
            for path in file_path:
                try:
                    success = self.data_manager.load_file(path)
                    if success:
                        self.logger.info(f"Successfully loaded: {path}")
                        # Switch to visualization tab if loading images (now index 3)
                        if path.lower().endswith(('.tif', '.tiff', '.nd2')):
                            self.main_tabs.setCurrentIndex(3)  # Visualization tab
                    else:
                        self.logger.error(f"Failed to load: {path}")
                except Exception as e:
                    self.logger.error(f"Error loading {path}: {e}")

    def _open_project(self):
        """Open a project file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "Project Files (*.json *.xml);;All Files (*)"
        )
        if file_path:
            try:
                if hasattr(self.project_manager, 'load_project'):
                    self.project_manager.load_project(file_path)
                    self.logger.info(f"Project opened: {file_path}")
                else:
                    QMessageBox.information(self, "Open Project", "Project functionality will be implemented soon.")
            except Exception as e:
                self.logger.error(f"Error opening project: {e}")
                QMessageBox.critical(self, "Error", f"Failed to open project: {e}")

    def _save_project(self):
        """Save current project."""
        try:
            if hasattr(self.project_manager, 'save_project'):
                self.project_manager.save_project()
                self.logger.info("Project saved")
            else:
                QMessageBox.information(self, "Save Project", "Project functionality will be implemented soon.")
        except Exception as e:
            self.logger.error(f"Error saving project: {e}")

    def _save_project_as(self):
        """Save project with new name."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", "", "Project Files (*.json);;All Files (*)"
        )
        if file_path:
            try:
                if hasattr(self.project_manager, 'save_project_as'):
                    self.project_manager.save_project_as(file_path)
                    self.logger.info(f"Project saved as: {file_path}")
                else:
                    QMessageBox.information(self, "Save Project As", "Project functionality will be implemented soon.")
            except Exception as e:
                self.logger.error(f"Error saving project as: {e}")

    def _show_preferences(self):
        """Show preferences dialog."""
        QMessageBox.information(self, "Preferences", "Preferences dialog will be implemented soon.")

    def _reset_layout(self):
        """Reset window layout to defaults."""
        self.main_tabs.setCurrentIndex(0)  # Go to first tab
        QMessageBox.information(self, "Layout Reset", "Layout has been reset to defaults.")

    def _show_user_guide(self):
        """Show user guide."""
        QMessageBox.information(
            self, "User Guide",
            "Enhanced Particle Tracking Analyzer\n\n"
            "Quick Start:\n"
            "1. Go to 'Data Management' tab to load image stacks or trajectory data\n"
            "2. Use 'Analysis Setup' tab to configure parameters and run analysis\n"
            "3. View results in 'Visualization & Results' tab\n"
            "4. Use 'Batch Processing' for multiple files\n"
            "5. Export data and reports in 'Export & Reports' tab\n\n"
            "For detailed documentation, visit the project repository."
        )

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About Enhanced Particle Tracking Analyzer",
            "Enhanced Particle Tracking Analyzer\n\n"
            "Advanced analysis software for single particle tracking\n"
            "with enhanced features including:\n"
            "• Multi-radius density analysis\n"
            "• Advanced shape metrics\n"
            "• Scaled radius of gyration\n"
            "• Comprehensive classification\n"
            "• Batch processing capabilities\n\n"
            "Built with Python and PyQt6"
        )


# Compatibility alias for easier migration
EnhancedMainWindow = RedesignedEnhancedMainWindow
