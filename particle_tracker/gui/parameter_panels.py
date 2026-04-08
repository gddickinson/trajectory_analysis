#!/usr/bin/env python3
"""
Enhanced Parameter Panels Module
=================================

Updated GUI panels for configuring all enhanced analysis parameters including
multi-radius density analysis, advanced shape metrics, scaled Rg, and more.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import asdict

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QTabWidget, QGroupBox, QLabel, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QLineEdit, QPushButton, QFileDialog,
    QSlider, QFrame, QScrollArea, QTextEdit, QListWidget,
    QListWidgetItem, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

# Import AnalysisParameters with error handling
try:
    from particle_tracker.core.analysis_engine import AnalysisParameters
except ImportError:
    # Create a fallback AnalysisParameters class
    from dataclasses import dataclass
    from typing import List, Optional
    import numpy as np

    @dataclass
    class AnalysisParameters:
        """Enhanced AnalysisParameters class."""
        # Detection parameters
        detection_method: str = "threshold"
        detection_sigma: float = 1.6
        detection_threshold: float = 3.0

        # Linking parameters
        linking_method: str = "nearest_neighbor"
        max_distance: float = 5.0
        max_gap_frames: int = 2
        min_track_length: int = 3

        # Basic feature calculation parameters
        pixel_size: float = 108.0
        frame_rate: float = 10.0

        # Enhanced feature parameters
        calculate_density: bool = True
        density_radii: List[int] = None
        calculate_advanced_shape: bool = True
        calculate_scaled_rg: bool = True
        calculate_diffusion: bool = True
        calculate_precision: bool = True
        interpolate_trajectories: bool = False

        # Advanced shape analysis parameters
        linear_eigenvalue_threshold: float = 20.0
        linear_alignment_threshold: float = 0.7

        # Classification parameters
        mobility_threshold: float = 2.11

        # SVM parameters
        svm_training_data: Optional[str] = None
        svm_features: List[str] = None

        # Background subtraction parameters
        roi_background_data: Optional[np.ndarray] = None
        camera_black_data: Optional[np.ndarray] = None

        def __post_init__(self):
            if self.svm_features is None:
                self.svm_features = [
                    'radius_gyration', 'asymmetry', 'fracDimension',
                    'netDispl', 'Straight', 'kurtosis'
                ]
            if self.density_radii is None:
                self.density_radii = [3, 5, 10, 20, 30]


class ParameterWidget(QWidget):
    """Base class for parameter input widgets."""

    valueChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)

    def get_value(self) -> Any:
        """Get the current parameter value."""
        raise NotImplementedError

    def set_value(self, value: Any):
        """Set the parameter value."""
        raise NotImplementedError

    def reset_to_default(self):
        """Reset to default value."""
        raise NotImplementedError


class DetectionParametersWidget(ParameterWidget):
    """Widget for detection parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QFormLayout(self)

        # Detection method
        self.method_combo = QComboBox()
        self.method_combo.addItems(["threshold", "log", "trackpy"])
        self.method_combo.currentTextChanged.connect(self.valueChanged)
        layout.addRow("Detection Method:", self.method_combo)

        # Sigma parameter
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.1, 10.0)
        self.sigma_spin.setValue(1.6)
        self.sigma_spin.setDecimals(2)
        self.sigma_spin.setSuffix(" pixels")
        self.sigma_spin.valueChanged.connect(self.valueChanged)
        layout.addRow("Sigma:", self.sigma_spin)

        # Threshold parameter
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 20.0)
        self.threshold_spin.setValue(3.0)
        self.threshold_spin.setDecimals(1)
        self.threshold_spin.valueChanged.connect(self.valueChanged)
        layout.addRow("Threshold:", self.threshold_spin)

        # Min intensity
        self.min_intensity_spin = QSpinBox()
        self.min_intensity_spin.setRange(0, 100000)
        self.min_intensity_spin.setValue(100)
        self.min_intensity_spin.valueChanged.connect(self.valueChanged)
        layout.addRow("Min Intensity:", self.min_intensity_spin)

        # Max intensity
        self.max_intensity_spin = QSpinBox()
        self.max_intensity_spin.setRange(100, 1000000)
        self.max_intensity_spin.setValue(10000)
        self.max_intensity_spin.valueChanged.connect(self.valueChanged)
        layout.addRow("Max Intensity:", self.max_intensity_spin)

    def get_value(self) -> Dict[str, Any]:
        """Get detection parameters."""
        return {
            'detection_method': self.method_combo.currentText(),
            'detection_sigma': self.sigma_spin.value(),
            'detection_threshold': self.threshold_spin.value(),
            'min_intensity': self.min_intensity_spin.value(),
            'max_intensity': self.max_intensity_spin.value()
        }

    def set_value(self, params: Dict[str, Any]):
        """Set detection parameters."""
        if 'detection_method' in params:
            self.method_combo.setCurrentText(params['detection_method'])
        if 'detection_sigma' in params:
            self.sigma_spin.setValue(params['detection_sigma'])
        if 'detection_threshold' in params:
            self.threshold_spin.setValue(params['detection_threshold'])
        if 'min_intensity' in params:
            self.min_intensity_spin.setValue(params['min_intensity'])
        if 'max_intensity' in params:
            self.max_intensity_spin.setValue(params['max_intensity'])

    def reset_to_default(self):
        """Reset to default values."""
        self.method_combo.setCurrentText("threshold")
        self.sigma_spin.setValue(1.6)
        self.threshold_spin.setValue(3.0)
        self.min_intensity_spin.setValue(100)
        self.max_intensity_spin.setValue(10000)


class LinkingParametersWidget(ParameterWidget):
    """Widget for linking parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QFormLayout(self)

        # Linking method
        self.method_combo = QComboBox()
        self.method_combo.addItems(["nearest_neighbor", "trackpy"])
        self.method_combo.currentTextChanged.connect(self.valueChanged)
        layout.addRow("Linking Method:", self.method_combo)

        # Max distance
        self.max_distance_spin = QDoubleSpinBox()
        self.max_distance_spin.setRange(0.1, 50.0)
        self.max_distance_spin.setValue(5.0)
        self.max_distance_spin.setDecimals(1)
        self.max_distance_spin.setSuffix(" pixels")
        self.max_distance_spin.valueChanged.connect(self.valueChanged)
        layout.addRow("Max Distance:", self.max_distance_spin)

        # Max gap frames
        self.max_gap_spin = QSpinBox()
        self.max_gap_spin.setRange(0, 20)
        self.max_gap_spin.setValue(2)
        self.max_gap_spin.valueChanged.connect(self.valueChanged)
        layout.addRow("Max Gap Frames:", self.max_gap_spin)

        # Min track length
        self.min_length_spin = QSpinBox()
        self.min_length_spin.setRange(1, 1000)
        self.min_length_spin.setValue(3)
        self.min_length_spin.valueChanged.connect(self.valueChanged)
        layout.addRow("Min Track Length:", self.min_length_spin)

    def get_value(self) -> Dict[str, Any]:
        """Get linking parameters."""
        return {
            'linking_method': self.method_combo.currentText(),
            'max_distance': self.max_distance_spin.value(),
            'max_gap_frames': self.max_gap_spin.value(),
            'min_track_length': self.min_length_spin.value()
        }

    def set_value(self, params: Dict[str, Any]):
        """Set linking parameters."""
        if 'linking_method' in params:
            self.method_combo.setCurrentText(params['linking_method'])
        if 'max_distance' in params:
            self.max_distance_spin.setValue(params['max_distance'])
        if 'max_gap_frames' in params:
            self.max_gap_spin.setValue(params['max_gap_frames'])
        if 'min_track_length' in params:
            self.min_length_spin.setValue(params['min_track_length'])

    def reset_to_default(self):
        """Reset to default values."""
        self.method_combo.setCurrentText("nearest_neighbor")
        self.max_distance_spin.setValue(5.0)
        self.max_gap_spin.setValue(2)
        self.min_length_spin.setValue(3)


class EnhancedFeatureParametersWidget(ParameterWidget):
    """Widget for enhanced feature calculation parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)

        # Create main tabs for different feature categories
        self.feature_tabs = QTabWidget()

        # Basic Features Tab
        self.basic_tab = self._create_basic_features_tab()
        self.feature_tabs.addTab(self.basic_tab, "Basic Features")

        # Density Analysis Tab
        self.density_tab = self._create_density_analysis_tab()
        self.feature_tabs.addTab(self.density_tab, "Density Analysis")

        # Advanced Shape Tab
        self.shape_tab = self._create_advanced_shape_tab()
        self.feature_tabs.addTab(self.shape_tab, "Advanced Shape")

        # Mobility & Diffusion Tab
        self.mobility_tab = self._create_mobility_diffusion_tab()
        self.feature_tabs.addTab(self.mobility_tab, "Mobility & Diffusion")

        # Quality & Precision Tab
        self.quality_tab = self._create_quality_precision_tab()
        self.feature_tabs.addTab(self.quality_tab, "Quality & Precision")

        layout.addWidget(self.feature_tabs)

        # Global controls
        controls_layout = QHBoxLayout()

        self.enable_all_btn = QPushButton("Enable All")
        self.enable_all_btn.clicked.connect(self._enable_all_features)
        controls_layout.addWidget(self.enable_all_btn)

        self.disable_all_btn = QPushButton("Disable All")
        self.disable_all_btn.clicked.connect(self._disable_all_features)
        controls_layout.addWidget(self.disable_all_btn)

        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self._reset_to_defaults)
        controls_layout.addWidget(self.reset_btn)

        layout.addLayout(controls_layout)

    def _create_basic_features_tab(self) -> QWidget:
        """Create basic features configuration tab."""
        tab = QWidget()
        layout = QFormLayout(tab)

        # Pixel size
        self.pixel_size_spin = QDoubleSpinBox()
        self.pixel_size_spin.setRange(1.0, 1000.0)
        self.pixel_size_spin.setValue(108.0)
        self.pixel_size_spin.setDecimals(2)
        self.pixel_size_spin.setSuffix(" nm")
        self.pixel_size_spin.valueChanged.connect(self.valueChanged)
        layout.addRow("Pixel Size:", self.pixel_size_spin)

        # Frame rate
        self.frame_rate_spin = QDoubleSpinBox()
        self.frame_rate_spin.setRange(0.1, 1000.0)
        self.frame_rate_spin.setValue(10.0)
        self.frame_rate_spin.setDecimals(2)
        self.frame_rate_spin.setSuffix(" Hz")
        self.frame_rate_spin.valueChanged.connect(self.valueChanged)
        layout.addRow("Frame Rate:", self.frame_rate_spin)

        # Basic feature toggles
        features_group = QGroupBox("Basic Features")
        features_layout = QVBoxLayout(features_group)

        self.calc_rg_cb = QCheckBox("Radius of Gyration")
        self.calc_rg_cb.setChecked(True)
        self.calc_rg_cb.toggled.connect(self.valueChanged)
        features_layout.addWidget(self.calc_rg_cb)

        self.calc_velocity_cb = QCheckBox("Velocity Metrics")
        self.calc_velocity_cb.setChecked(True)
        self.calc_velocity_cb.toggled.connect(self.valueChanged)
        features_layout.addWidget(self.calc_velocity_cb)

        self.calc_nn_cb = QCheckBox("Nearest Neighbors")
        self.calc_nn_cb.setChecked(True)
        self.calc_nn_cb.toggled.connect(self.valueChanged)
        features_layout.addWidget(self.calc_nn_cb)

        layout.addRow(features_group)

        return tab

    def _create_density_analysis_tab(self) -> QWidget:
        """Create density analysis configuration tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Enable density analysis
        self.density_enabled_cb = QCheckBox("Enable Multi-Radius Density Analysis")
        self.density_enabled_cb.setChecked(True)
        self.density_enabled_cb.toggled.connect(self._on_density_enabled_changed)
        self.density_enabled_cb.toggled.connect(self.valueChanged)
        layout.addWidget(self.density_enabled_cb)

        # Density radii configuration
        radii_group = QGroupBox("Analysis Radii (pixels)")
        radii_layout = QVBoxLayout(radii_group)

        # Instructions
        info_label = QLabel("Select radii for neighbor counting. Default: 3, 5, 10, 20, 30 pixels")
        info_label.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
        radii_layout.addWidget(info_label)

        # Radii list widget
        self.radii_list = QListWidget()
        self.radii_list.setMaximumHeight(120)
        self.radii_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)

        # Add default radii
        default_radii = [3, 5, 10, 20, 30]
        for radius in default_radii:
            item = QListWidgetItem(f"{radius} pixels")
            item.setData(Qt.ItemDataRole.UserRole, radius)
            item.setSelected(True)
            self.radii_list.addItem(item)

        self.radii_list.itemSelectionChanged.connect(self.valueChanged)
        radii_layout.addWidget(self.radii_list)

        # Add/remove radii controls
        radii_controls = QHBoxLayout()

        self.add_radius_spin = QSpinBox()
        self.add_radius_spin.setRange(1, 100)
        self.add_radius_spin.setValue(15)
        self.add_radius_spin.setSuffix(" px")
        radii_controls.addWidget(QLabel("Add radius:"))
        radii_controls.addWidget(self.add_radius_spin)

        self.add_radius_btn = QPushButton("Add")
        self.add_radius_btn.clicked.connect(self._add_radius)
        radii_controls.addWidget(self.add_radius_btn)

        self.remove_radius_btn = QPushButton("Remove Selected")
        self.remove_radius_btn.clicked.connect(self._remove_radius)
        radii_controls.addWidget(self.remove_radius_btn)

        radii_layout.addLayout(radii_controls)

        # Preset buttons
        preset_layout = QHBoxLayout()

        self.dense_preset_btn = QPushButton("Dense (3,5,10)")
        self.dense_preset_btn.clicked.connect(lambda: self._set_radius_preset([3, 5, 10]))
        preset_layout.addWidget(self.dense_preset_btn)

        self.standard_preset_btn = QPushButton("Standard (3,5,10,20,30)")
        self.standard_preset_btn.clicked.connect(lambda: self._set_radius_preset([3, 5, 10, 20, 30]))
        preset_layout.addWidget(self.standard_preset_btn)

        self.extended_preset_btn = QPushButton("Extended (3,5,10,20,30,50)")
        self.extended_preset_btn.clicked.connect(lambda: self._set_radius_preset([3, 5, 10, 20, 30, 50]))
        preset_layout.addWidget(self.extended_preset_btn)

        radii_layout.addLayout(preset_layout)

        layout.addWidget(radii_group)
        layout.addStretch()

        return tab

    def _create_advanced_shape_tab(self) -> QWidget:
        """Create advanced shape analysis configuration tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Enable advanced shape analysis
        self.advanced_shape_cb = QCheckBox("Enable Advanced Shape Analysis")
        self.advanced_shape_cb.setChecked(True)
        self.advanced_shape_cb.toggled.connect(self._on_advanced_shape_changed)
        self.advanced_shape_cb.toggled.connect(self.valueChanged)
        layout.addWidget(self.advanced_shape_cb)

        # Shape metrics group
        metrics_group = QGroupBox("Shape Metrics")
        metrics_layout = QVBoxLayout(metrics_group)

        metrics_info = QLabel("Includes: eigenvalue ratios, linearity classification, directionality ratios, step alignment")
        metrics_info.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
        metrics_layout.addWidget(metrics_info)

        self.eigenvalue_ratio_cb = QCheckBox("Eigenvalue Ratio Analysis")
        self.eigenvalue_ratio_cb.setChecked(True)
        self.eigenvalue_ratio_cb.toggled.connect(self.valueChanged)
        metrics_layout.addWidget(self.eigenvalue_ratio_cb)

        self.step_alignment_cb = QCheckBox("Step Alignment Analysis")
        self.step_alignment_cb.setChecked(True)
        self.step_alignment_cb.toggled.connect(self.valueChanged)
        metrics_layout.addWidget(self.step_alignment_cb)

        self.directionality_cb = QCheckBox("Directionality Ratio")
        self.directionality_cb.setChecked(True)
        self.directionality_cb.toggled.connect(self.valueChanged)
        metrics_layout.addWidget(self.directionality_cb)

        layout.addWidget(metrics_group)

        # Linearity classification parameters
        linearity_group = QGroupBox("Linearity Classification Thresholds")
        linearity_layout = QFormLayout(linearity_group)

        self.eigenvalue_threshold_spin = QDoubleSpinBox()
        self.eigenvalue_threshold_spin.setRange(1.0, 100.0)
        self.eigenvalue_threshold_spin.setValue(20.0)
        self.eigenvalue_threshold_spin.setDecimals(1)
        self.eigenvalue_threshold_spin.valueChanged.connect(self.valueChanged)
        linearity_layout.addRow("Eigenvalue Ratio Threshold:", self.eigenvalue_threshold_spin)

        self.alignment_threshold_spin = QDoubleSpinBox()
        self.alignment_threshold_spin.setRange(0.1, 1.0)
        self.alignment_threshold_spin.setValue(0.7)
        self.alignment_threshold_spin.setDecimals(2)
        self.alignment_threshold_spin.valueChanged.connect(self.valueChanged)
        linearity_layout.addRow("Step Alignment Threshold:", self.alignment_threshold_spin)

        # Help text
        help_text = QLabel("Higher thresholds = more stringent linear classification")
        help_text.setStyleSheet("QLabel { color: gray; font-size: 9px; }")
        linearity_layout.addRow("", help_text)

        layout.addWidget(linearity_group)
        layout.addStretch()

        return tab

    def _create_mobility_diffusion_tab(self) -> QWidget:
        """Create mobility and diffusion analysis tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Scaled Rg group
        scaled_rg_group = QGroupBox("Scaled Radius of Gyration (sRg)")
        scaled_rg_layout = QFormLayout(scaled_rg_group)

        self.scaled_rg_cb = QCheckBox("Calculate Scaled Rg")
        self.scaled_rg_cb.setChecked(True)
        self.scaled_rg_cb.toggled.connect(self.valueChanged)
        scaled_rg_layout.addRow("", self.scaled_rg_cb)

        # Mobility threshold
        self.mobility_threshold_spin = QDoubleSpinBox()
        self.mobility_threshold_spin.setRange(0.1, 10.0)
        self.mobility_threshold_spin.setValue(2.11)
        self.mobility_threshold_spin.setDecimals(3)
        self.mobility_threshold_spin.valueChanged.connect(self.valueChanged)
        scaled_rg_layout.addRow("Mobility Threshold:", self.mobility_threshold_spin)

        mobility_help = QLabel("Golan & Sherman threshold: sRg > 2.11 = mobile")
        mobility_help.setStyleSheet("QLabel { color: gray; font-size: 9px; }")
        scaled_rg_layout.addRow("", mobility_help)

        layout.addWidget(scaled_rg_group)

        # Diffusion analysis group
        diffusion_group = QGroupBox("Diffusion Analysis")
        diffusion_layout = QVBoxLayout(diffusion_group)

        self.diffusion_cb = QCheckBox("Enable Comprehensive Diffusion Analysis")
        self.diffusion_cb.setChecked(True)
        self.diffusion_cb.toggled.connect(self.valueChanged)
        diffusion_layout.addWidget(self.diffusion_cb)

        diffusion_info = QLabel("Includes: MSD analysis, diffusion coefficients, trajectory origin analysis")
        diffusion_info.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
        diffusion_layout.addWidget(diffusion_info)

        layout.addWidget(diffusion_group)
        layout.addStretch()

        return tab

    def _create_quality_precision_tab(self) -> QWidget:
        """Create quality and precision analysis tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Localization precision group
        precision_group = QGroupBox("Localization Precision Analysis")
        precision_layout = QVBoxLayout(precision_group)

        self.precision_cb = QCheckBox("Enable Localization Precision Analysis")
        self.precision_cb.setChecked(True)
        self.precision_cb.toggled.connect(self.valueChanged)
        precision_layout.addWidget(self.precision_cb)

        precision_info = QLabel("Analyzes position variability and measurement precision")
        precision_info.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
        precision_layout.addWidget(precision_info)

        layout.addWidget(precision_group)

        # Trajectory interpolation group
        interp_group = QGroupBox("Trajectory Interpolation")
        interp_layout = QVBoxLayout(interp_group)

        self.interpolate_cb = QCheckBox("Interpolate Missing Timepoints")
        self.interpolate_cb.setChecked(False)
        self.interpolate_cb.toggled.connect(self.valueChanged)
        interp_layout.addWidget(self.interpolate_cb)

        interp_info = QLabel("Fill gaps in trajectories using linear interpolation")
        interp_info.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
        interp_layout.addWidget(interp_info)

        interp_warning = QLabel("⚠️ May affect statistical analysis - use carefully")
        interp_warning.setStyleSheet("QLabel { color: orange; font-size: 9px; }")
        interp_layout.addWidget(interp_warning)

        layout.addWidget(interp_group)
        layout.addStretch()

        return tab

    # Event handlers
    def _on_density_enabled_changed(self, enabled: bool):
        """Handle density analysis enable/disable."""
        self.radii_list.setEnabled(enabled)
        self.add_radius_spin.setEnabled(enabled)
        self.add_radius_btn.setEnabled(enabled)
        self.remove_radius_btn.setEnabled(enabled)

    def _on_advanced_shape_changed(self, enabled: bool):
        """Handle advanced shape analysis enable/disable."""
        self.eigenvalue_threshold_spin.setEnabled(enabled)
        self.alignment_threshold_spin.setEnabled(enabled)

    def _add_radius(self):
        """Add a new radius to the list."""
        radius = self.add_radius_spin.value()
        
        # Check if radius already exists
        for i in range(self.radii_list.count()):
            item = self.radii_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == radius:
                return  # Already exists

        # Add new radius
        item = QListWidgetItem(f"{radius} pixels")
        item.setData(Qt.ItemDataRole.UserRole, radius)
        item.setSelected(True)
        self.radii_list.addItem(item)
        self.valueChanged.emit()

    def _remove_radius(self):
        """Remove selected radii from the list."""
        for item in self.radii_list.selectedItems():
            row = self.radii_list.row(item)
            self.radii_list.takeItem(row)
        self.valueChanged.emit()

    def _set_radius_preset(self, radii: List[int]):
        """Set radii to a preset configuration."""
        self.radii_list.clear()
        for radius in sorted(radii):
            item = QListWidgetItem(f"{radius} pixels")
            item.setData(Qt.ItemDataRole.UserRole, radius)
            item.setSelected(True)
            self.radii_list.addItem(item)
        self.valueChanged.emit()

    def _enable_all_features(self):
        """Enable all feature calculations."""
        self.density_enabled_cb.setChecked(True)
        self.advanced_shape_cb.setChecked(True)
        self.scaled_rg_cb.setChecked(True)
        self.diffusion_cb.setChecked(True)
        self.precision_cb.setChecked(True)
        self.valueChanged.emit()

    def _disable_all_features(self):
        """Disable all feature calculations."""
        self.density_enabled_cb.setChecked(False)
        self.advanced_shape_cb.setChecked(False)
        self.scaled_rg_cb.setChecked(False)
        self.diffusion_cb.setChecked(False)
        self.precision_cb.setChecked(False)
        self.valueChanged.emit()

    def _reset_to_defaults(self):
        """Reset all parameters to default values."""
        # Reset basic parameters
        self.pixel_size_spin.setValue(108.0)
        self.frame_rate_spin.setValue(10.0)

        # Reset feature toggles
        self.density_enabled_cb.setChecked(True)
        self.advanced_shape_cb.setChecked(True)
        self.scaled_rg_cb.setChecked(True)
        self.diffusion_cb.setChecked(True)
        self.precision_cb.setChecked(True)
        self.interpolate_cb.setChecked(False)

        # Reset thresholds
        self.mobility_threshold_spin.setValue(2.11)
        self.eigenvalue_threshold_spin.setValue(20.0)
        self.alignment_threshold_spin.setValue(0.7)

        # Reset radii to defaults
        self._set_radius_preset([3, 5, 10, 20, 30])

        self.valueChanged.emit()

    def get_value(self) -> Dict[str, Any]:
        """Get all enhanced feature parameters."""
        # Get selected radii
        selected_radii = []
        for i in range(self.radii_list.count()):
            item = self.radii_list.item(i)
            if item.isSelected():
                selected_radii.append(item.data(Qt.ItemDataRole.UserRole))

        return {
            # Basic parameters
            'pixel_size': self.pixel_size_spin.value(),
            'frame_rate': self.frame_rate_spin.value(),

            # Enhanced feature toggles
            'calculate_density': self.density_enabled_cb.isChecked(),
            'density_radii': sorted(selected_radii),
            'calculate_advanced_shape': self.advanced_shape_cb.isChecked(),
            'calculate_scaled_rg': self.scaled_rg_cb.isChecked(),
            'calculate_diffusion': self.diffusion_cb.isChecked(),
            'calculate_precision': self.precision_cb.isChecked(),
            'interpolate_trajectories': self.interpolate_cb.isChecked(),

            # Advanced shape parameters
            'linear_eigenvalue_threshold': self.eigenvalue_threshold_spin.value(),
            'linear_alignment_threshold': self.alignment_threshold_spin.value(),

            # Mobility parameters
            'mobility_threshold': self.mobility_threshold_spin.value(),

            # Individual feature flags
            'calculate_rg': self.calc_rg_cb.isChecked(),
            'calculate_velocity': self.calc_velocity_cb.isChecked(),
            'calculate_nn': self.calc_nn_cb.isChecked(),
            'calculate_eigenvalue_ratio': self.eigenvalue_ratio_cb.isChecked(),
            'calculate_step_alignment': self.step_alignment_cb.isChecked(),
            'calculate_directionality': self.directionality_cb.isChecked(),
        }

    def set_value(self, params: Dict[str, Any]):
        """Set enhanced feature parameters."""
        # Basic parameters
        if 'pixel_size' in params:
            self.pixel_size_spin.setValue(params['pixel_size'])
        if 'frame_rate' in params:
            self.frame_rate_spin.setValue(params['frame_rate'])

        # Feature toggles
        if 'calculate_density' in params:
            self.density_enabled_cb.setChecked(params['calculate_density'])
        if 'calculate_advanced_shape' in params:
            self.advanced_shape_cb.setChecked(params['calculate_advanced_shape'])
        if 'calculate_scaled_rg' in params:
            self.scaled_rg_cb.setChecked(params['calculate_scaled_rg'])
        if 'calculate_diffusion' in params:
            self.diffusion_cb.setChecked(params['calculate_diffusion'])
        if 'calculate_precision' in params:
            self.precision_cb.setChecked(params['calculate_precision'])
        if 'interpolate_trajectories' in params:
            self.interpolate_cb.setChecked(params['interpolate_trajectories'])

        # Thresholds
        if 'mobility_threshold' in params:
            self.mobility_threshold_spin.setValue(params['mobility_threshold'])
        if 'linear_eigenvalue_threshold' in params:
            self.eigenvalue_threshold_spin.setValue(params['linear_eigenvalue_threshold'])
        if 'linear_alignment_threshold' in params:
            self.alignment_threshold_spin.setValue(params['linear_alignment_threshold'])

        # Density radii
        if 'density_radii' in params:
            self._set_radius_preset(params['density_radii'])

    def reset_to_default(self):
        """Reset to default values."""
        self._reset_to_defaults()


class ClassificationParametersWidget(ParameterWidget):
    """Widget for classification parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QFormLayout(self)

        # Classification method
        self.method_combo = QComboBox()
        self.method_combo.addItems(["threshold", "svm"])
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        self.method_combo.currentTextChanged.connect(self.valueChanged)
        layout.addRow("Classification Method:", self.method_combo)

        # SVM training data
        svm_layout = QHBoxLayout()
        self.svm_file_edit = QLineEdit()
        self.svm_file_edit.setPlaceholderText("Select SVM training data file...")
        self.svm_file_edit.textChanged.connect(self.valueChanged)

        self.svm_browse_btn = QPushButton("Browse...")
        self.svm_browse_btn.clicked.connect(self._browse_svm_file)

        svm_layout.addWidget(self.svm_file_edit)
        svm_layout.addWidget(self.svm_browse_btn)

        layout.addRow("SVM Training Data:", svm_layout)

        # Mobility threshold
        self.mobility_threshold_spin = QDoubleSpinBox()
        self.mobility_threshold_spin.setRange(0.1, 10.0)
        self.mobility_threshold_spin.setValue(2.11)
        self.mobility_threshold_spin.setDecimals(3)
        self.mobility_threshold_spin.valueChanged.connect(self.valueChanged)
        layout.addRow("Mobility Threshold:", self.mobility_threshold_spin)

        # Update initial state
        self._on_method_changed("threshold")

    def _on_method_changed(self, method: str):
        """Handle classification method change."""
        is_svm = method == "svm"
        self.svm_file_edit.setEnabled(is_svm)
        self.svm_browse_btn.setEnabled(is_svm)

    def _browse_svm_file(self):
        """Browse for SVM training data file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select SVM Training Data", "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.svm_file_edit.setText(file_path)

    def get_value(self) -> Dict[str, Any]:
        """Get classification parameters."""
        return {
            'classification_method': self.method_combo.currentText(),
            'svm_training_data': self.svm_file_edit.text(),
            'mobility_threshold': self.mobility_threshold_spin.value()
        }

    def set_value(self, params: Dict[str, Any]):
        """Set classification parameters."""
        if 'classification_method' in params:
            self.method_combo.setCurrentText(params['classification_method'])
        if 'svm_training_data' in params:
            self.svm_file_edit.setText(params['svm_training_data'])
        if 'mobility_threshold' in params:
            self.mobility_threshold_spin.setValue(params['mobility_threshold'])

    def reset_to_default(self):
        """Reset to default values."""
        self.method_combo.setCurrentText("threshold")
        self.svm_file_edit.clear()
        self.mobility_threshold_spin.setValue(2.11)


class ParameterPanelManager(QWidget):
    """Enhanced manager widget for all parameter panels."""

    parametersChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)

        # Create tab widget for different parameter categories
        self.tab_widget = QTabWidget()

        # Detection parameters tab
        self.detection_widget = DetectionParametersWidget()
        scroll_detection = QScrollArea()
        scroll_detection.setWidget(self.detection_widget)
        scroll_detection.setWidgetResizable(True)
        self.tab_widget.addTab(scroll_detection, "Detection")

        # Linking parameters tab
        self.linking_widget = LinkingParametersWidget()
        scroll_linking = QScrollArea()
        scroll_linking.setWidget(self.linking_widget)
        scroll_linking.setWidgetResizable(True)
        self.tab_widget.addTab(scroll_linking, "Linking")

        # Enhanced feature parameters tab
        self.enhanced_feature_widget = EnhancedFeatureParametersWidget()
        scroll_enhanced = QScrollArea()
        scroll_enhanced.setWidget(self.enhanced_feature_widget)
        scroll_enhanced.setWidgetResizable(True)
        self.tab_widget.addTab(scroll_enhanced, "Enhanced Features")

        # Classification parameters tab
        self.classification_widget = ClassificationParametersWidget()
        scroll_classification = QScrollArea()
        scroll_classification.setWidget(self.classification_widget)
        scroll_classification.setWidgetResizable(True)
        self.tab_widget.addTab(scroll_classification, "Classification")

        layout.addWidget(self.tab_widget)

        # Control buttons
        button_layout = QHBoxLayout()

        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self._reset_all_parameters)
        button_layout.addWidget(self.reset_button)

        self.save_button = QPushButton("Save Parameters")
        self.save_button.clicked.connect(self._save_parameters)
        button_layout.addWidget(self.save_button)

        self.load_button = QPushButton("Load Parameters")
        self.load_button.clicked.connect(self._load_parameters)
        button_layout.addWidget(self.load_button)

        layout.addLayout(button_layout)

    def _connect_signals(self):
        """Connect parameter change signals."""
        self.detection_widget.valueChanged.connect(self.parametersChanged)
        self.linking_widget.valueChanged.connect(self.parametersChanged)
        self.enhanced_feature_widget.valueChanged.connect(self.parametersChanged)
        self.classification_widget.valueChanged.connect(self.parametersChanged)

    def get_all_parameters(self) -> Dict[str, Any]:
        """Get all parameters including enhanced features."""
        detection_params = self.detection_widget.get_value()
        linking_params = self.linking_widget.get_value()
        enhanced_feature_params = self.enhanced_feature_widget.get_value()
        classification_params = self.classification_widget.get_value()

        # Merge all parameters
        all_params = {}
        all_params.update(detection_params)
        all_params.update(linking_params)
        all_params.update(enhanced_feature_params)
        all_params.update(classification_params)

        return all_params

    def set_all_parameters(self, params):
        """Set all parameters including enhanced features."""
        if hasattr(params, '__dict__'):
            params_dict = asdict(params) if hasattr(params, '__dataclass_fields__') else params.__dict__
        else:
            params_dict = params

        # Set parameters for each widget
        self.detection_widget.set_value(params_dict)
        self.linking_widget.set_value(params_dict)
        self.enhanced_feature_widget.set_value(params_dict)
        self.classification_widget.set_value(params_dict)

    def _reset_all_parameters(self):
        """Reset all parameters to defaults."""
        default_params = AnalysisParameters()
        self.set_all_parameters(default_params)
        self.logger.info("Parameters reset to enhanced defaults")

    def _save_parameters(self):
        """Save current parameters to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Parameters", "",
            "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            try:
                import json
                params = self.get_all_parameters()

                with open(file_path, 'w') as f:
                    json.dump(params, f, indent=2)

                self.logger.info(f"Enhanced parameters saved to {file_path}")

            except Exception as e:
                self.logger.error(f"Error saving enhanced parameters: {e}")

    def _load_parameters(self):
        """Load parameters from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Parameters", "",
            "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            try:
                import json

                with open(file_path, 'r') as f:
                    params_dict = json.load(f)

                self.set_all_parameters(params_dict)
                self.logger.info(f"Enhanced parameters loaded from {file_path}")

            except Exception as e:
                self.logger.error(f"Error loading enhanced parameters: {e}")