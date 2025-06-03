#!/usr/bin/env python3
"""
Parameter Panels Module
=======================

Provides GUI panels for configuring analysis parameters.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import asdict

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QTabWidget, QGroupBox, QLabel, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QLineEdit, QPushButton, QFileDialog,
    QSlider, QFrame, QScrollArea, QTextEdit, QListWidget
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

    @dataclass
    class AnalysisParameters:
        """Fallback AnalysisParameters class."""
        # Detection parameters
        detection_method: str = "threshold"
        detection_sigma: float = 1.6
        detection_threshold: float = 3.0

        # Linking parameters
        linking_method: str = "nearest_neighbor"
        max_distance: float = 5.0
        max_gap_frames: int = 2
        min_track_length: int = 3

        # Feature calculation parameters
        pixel_size: float = 108.0  # nm per pixel
        frame_rate: float = 10.0   # Hz

        # Classification parameters
        mobility_threshold: float = 2.11

        # SVM parameters
        svm_training_data: Optional[str] = None
        svm_features: List[str] = None

        def __post_init__(self):
            if self.svm_features is None:
                self.svm_features = [
                    'radius_gyration', 'asymmetry', 'fracDimension',
                    'netDispl', 'Straight', 'kurtosis'
                ]


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
    """Widget for particle detection parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QFormLayout(self)

        # Detection method
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "threshold", "log", "trackpy"
        ])
        self.method_combo.setCurrentText("threshold")
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        layout.addRow("Detection Method:", self.method_combo)

        # Sigma (spot size)
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.1, 10.0)
        self.sigma_spin.setSingleStep(0.1)
        self.sigma_spin.setValue(1.6)
        self.sigma_spin.setDecimals(2)
        self.sigma_spin.setSuffix(" pixels")
        self.sigma_spin.valueChanged.connect(self.valueChanged)
        layout.addRow("Sigma (spot size):", self.sigma_spin)

        # Threshold
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 100.0)
        self.threshold_spin.setSingleStep(0.1)
        self.threshold_spin.setValue(3.0)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setSuffix(" σ")
        self.threshold_spin.valueChanged.connect(self.valueChanged)
        layout.addRow("Threshold:", self.threshold_spin)

        # Minimum intensity
        self.min_intensity_spin = QSpinBox()
        self.min_intensity_spin.setRange(0, 65535)
        self.min_intensity_spin.setValue(100)
        self.min_intensity_spin.valueChanged.connect(self.valueChanged)
        layout.addRow("Min Intensity:", self.min_intensity_spin)

        # Maximum intensity
        self.max_intensity_spin = QSpinBox()
        self.max_intensity_spin.setRange(0, 65535)
        self.max_intensity_spin.setValue(10000)
        self.max_intensity_spin.valueChanged.connect(self.valueChanged)
        layout.addRow("Max Intensity:", self.max_intensity_spin)

        # Spot diameter (for trackpy)
        self.diameter_spin = QSpinBox()
        self.diameter_spin.setRange(3, 51)
        self.diameter_spin.setValue(7)
        self.diameter_spin.setSingleStep(2)
        self.diameter_spin.setSuffix(" pixels")
        self.diameter_spin.valueChanged.connect(self.valueChanged)
        layout.addRow("Spot Diameter:", self.diameter_spin)

        # Background subtraction
        self.background_cb = QCheckBox("Background Subtraction")
        self.background_cb.setChecked(True)
        self.background_cb.toggled.connect(self.valueChanged)
        layout.addRow("", self.background_cb)

        # Connect signals
        self.method_combo.currentTextChanged.connect(self.valueChanged)

    def _on_method_changed(self, method: str):
        """Handle detection method change."""
        # Enable/disable relevant controls based on method
        trackpy_controls = [self.diameter_spin]
        threshold_controls = [self.sigma_spin, self.threshold_spin]

        if method == "trackpy":
            for control in trackpy_controls:
                control.setEnabled(True)
        else:
            for control in trackpy_controls:
                control.setEnabled(False)

        if method in ["threshold", "log"]:
            for control in threshold_controls:
                control.setEnabled(True)
        else:
            for control in threshold_controls:
                control.setEnabled(False)

    def get_value(self) -> Dict[str, Any]:
        """Get detection parameters."""
        return {
            'detection_method': self.method_combo.currentText(),
            'detection_sigma': self.sigma_spin.value(),
            'detection_threshold': self.threshold_spin.value(),
            'min_intensity': self.min_intensity_spin.value(),
            'max_intensity': self.max_intensity_spin.value(),
            'spot_diameter': self.diameter_spin.value(),
            'background_subtraction': self.background_cb.isChecked()
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
        if 'spot_diameter' in params:
            self.diameter_spin.setValue(params['spot_diameter'])
        if 'background_subtraction' in params:
            self.background_cb.setChecked(params['background_subtraction'])


class LinkingParametersWidget(ParameterWidget):
    """Widget for particle linking parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QFormLayout(self)

        # Linking method
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "nearest_neighbor", "trackpy"
        ])
        self.method_combo.setCurrentText("nearest_neighbor")
        self.method_combo.currentTextChanged.connect(self.valueChanged)
        layout.addRow("Linking Method:", self.method_combo)

        # Maximum distance
        self.max_distance_spin = QDoubleSpinBox()
        self.max_distance_spin.setRange(0.1, 100.0)
        self.max_distance_spin.setSingleStep(0.1)
        self.max_distance_spin.setValue(5.0)
        self.max_distance_spin.setDecimals(2)
        self.max_distance_spin.setSuffix(" pixels")
        self.max_distance_spin.valueChanged.connect(self.valueChanged)
        layout.addRow("Max Distance:", self.max_distance_spin)

        # Maximum gap frames
        self.max_gap_spin = QSpinBox()
        self.max_gap_spin.setRange(0, 50)
        self.max_gap_spin.setValue(2)
        self.max_gap_spin.setSuffix(" frames")
        self.max_gap_spin.valueChanged.connect(self.valueChanged)
        layout.addRow("Max Gap Frames:", self.max_gap_spin)

        # Minimum track length
        self.min_length_spin = QSpinBox()
        self.min_length_spin.setRange(2, 1000)
        self.min_length_spin.setValue(3)
        self.min_length_spin.setSuffix(" points")
        self.min_length_spin.valueChanged.connect(self.valueChanged)
        layout.addRow("Min Track Length:", self.min_length_spin)

        # Maximum track length
        self.max_length_spin = QSpinBox()
        self.max_length_spin.setRange(10, 10000)
        self.max_length_spin.setValue(1000)
        self.max_length_spin.setSuffix(" points")
        self.max_length_spin.valueChanged.connect(self.valueChanged)
        layout.addRow("Max Track Length:", self.max_length_spin)

    def get_value(self) -> Dict[str, Any]:
        """Get linking parameters."""
        return {
            'linking_method': self.method_combo.currentText(),
            'max_distance': self.max_distance_spin.value(),
            'max_gap_frames': self.max_gap_spin.value(),
            'min_track_length': self.min_length_spin.value(),
            'max_track_length': self.max_length_spin.value()
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
        if 'max_track_length' in params:
            self.max_length_spin.setValue(params['max_track_length'])


class FeatureParametersWidget(ParameterWidget):
    """Widget for feature calculation parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QFormLayout(self)

        # Pixel size
        self.pixel_size_spin = QDoubleSpinBox()
        self.pixel_size_spin.setRange(1.0, 1000.0)
        self.pixel_size_spin.setSingleStep(1.0)
        self.pixel_size_spin.setValue(108.0)
        self.pixel_size_spin.setDecimals(2)
        self.pixel_size_spin.setSuffix(" nm")
        self.pixel_size_spin.valueChanged.connect(self.valueChanged)
        layout.addRow("Pixel Size:", self.pixel_size_spin)

        # Frame rate
        self.frame_rate_spin = QDoubleSpinBox()
        self.frame_rate_spin.setRange(0.1, 1000.0)
        self.frame_rate_spin.setSingleStep(0.1)
        self.frame_rate_spin.setValue(10.0)
        self.frame_rate_spin.setDecimals(2)
        self.frame_rate_spin.setSuffix(" Hz")
        self.frame_rate_spin.valueChanged.connect(self.valueChanged)
        layout.addRow("Frame Rate:", self.frame_rate_spin)

        # Feature selection
        features_group = QGroupBox("Features to Calculate")
        features_layout = QVBoxLayout(features_group)

        self.rg_cb = QCheckBox("Radius of Gyration")
        self.rg_cb.setChecked(True)
        self.rg_cb.toggled.connect(self.valueChanged)
        features_layout.addWidget(self.rg_cb)

        self.asymmetry_cb = QCheckBox("Asymmetry")
        self.asymmetry_cb.setChecked(True)
        self.asymmetry_cb.toggled.connect(self.valueChanged)
        features_layout.addWidget(self.asymmetry_cb)

        self.fractal_cb = QCheckBox("Fractal Dimension")
        self.fractal_cb.setChecked(True)
        self.fractal_cb.toggled.connect(self.valueChanged)
        features_layout.addWidget(self.fractal_cb)

        self.msd_cb = QCheckBox("Mean Square Displacement")
        self.msd_cb.setChecked(True)
        self.msd_cb.toggled.connect(self.valueChanged)
        features_layout.addWidget(self.msd_cb)

        self.velocity_cb = QCheckBox("Velocity")
        self.velocity_cb.setChecked(True)
        self.velocity_cb.toggled.connect(self.valueChanged)
        features_layout.addWidget(self.velocity_cb)

        self.nn_cb = QCheckBox("Nearest Neighbors")
        self.nn_cb.setChecked(True)
        self.nn_cb.toggled.connect(self.valueChanged)
        features_layout.addWidget(self.nn_cb)

        layout.addRow(features_group)

        # Mobility threshold
        self.mobility_threshold_spin = QDoubleSpinBox()
        self.mobility_threshold_spin.setRange(0.1, 10.0)
        self.mobility_threshold_spin.setSingleStep(0.01)
        self.mobility_threshold_spin.setValue(2.11)
        self.mobility_threshold_spin.setDecimals(3)
        self.mobility_threshold_spin.valueChanged.connect(self.valueChanged)
        layout.addRow("Mobility Threshold:", self.mobility_threshold_spin)

    def get_value(self) -> Dict[str, Any]:
        """Get feature parameters."""
        return {
            'pixel_size': self.pixel_size_spin.value(),
            'frame_rate': self.frame_rate_spin.value(),
            'calculate_rg': self.rg_cb.isChecked(),
            'calculate_asymmetry': self.asymmetry_cb.isChecked(),
            'calculate_fractal': self.fractal_cb.isChecked(),
            'calculate_msd': self.msd_cb.isChecked(),
            'calculate_velocity': self.velocity_cb.isChecked(),
            'calculate_nn': self.nn_cb.isChecked(),
            'mobility_threshold': self.mobility_threshold_spin.value()
        }

    def set_value(self, params: Dict[str, Any]):
        """Set feature parameters."""
        if 'pixel_size' in params:
            self.pixel_size_spin.setValue(params['pixel_size'])
        if 'frame_rate' in params:
            self.frame_rate_spin.setValue(params['frame_rate'])
        if 'calculate_rg' in params:
            self.rg_cb.setChecked(params['calculate_rg'])
        if 'calculate_asymmetry' in params:
            self.asymmetry_cb.setChecked(params['calculate_asymmetry'])
        if 'calculate_fractal' in params:
            self.fractal_cb.setChecked(params['calculate_fractal'])
        if 'calculate_msd' in params:
            self.msd_cb.setChecked(params['calculate_msd'])
        if 'calculate_velocity' in params:
            self.velocity_cb.setChecked(params['calculate_velocity'])
        if 'calculate_nn' in params:
            self.nn_cb.setChecked(params['calculate_nn'])
        if 'mobility_threshold' in params:
            self.mobility_threshold_spin.setValue(params['mobility_threshold'])


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
        self.method_combo.addItems([
            "svm", "threshold"
        ])
        self.method_combo.setCurrentText("svm")
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        layout.addRow("Classification Method:", self.method_combo)

        # SVM training data file
        svm_layout = QHBoxLayout()
        self.training_data_edit = QLineEdit()
        self.training_data_edit.setPlaceholderText("Path to training data...")
        self.training_data_edit.textChanged.connect(self.valueChanged)

        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._browse_training_data)

        svm_layout.addWidget(self.training_data_edit)
        svm_layout.addWidget(self.browse_button)
        layout.addRow("Training Data:", svm_layout)

        # SVM features
        features_group = QGroupBox("SVM Features")
        features_layout = QVBoxLayout(features_group)

        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)

        # Add common features
        features = [
            'radius_gyration', 'asymmetry', 'fracDimension',
            'netDispl', 'Straight', 'kurtosis', 'skewness',
            'velocity', 'diffusion_coefficient'
        ]

        for feature in features:
            self.feature_list.addItem(feature)

        # Select default features
        default_features = [
            'radius_gyration', 'asymmetry', 'fracDimension',
            'netDispl', 'Straight', 'kurtosis'
        ]

        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            if item.text() in default_features:
                item.setSelected(True)

        self.feature_list.itemSelectionChanged.connect(self.valueChanged)
        features_layout.addWidget(self.feature_list)

        layout.addRow(features_group)

        # Threshold parameters
        self.threshold_params_group = QGroupBox("Threshold Parameters")
        threshold_layout = QFormLayout(self.threshold_params_group)

        self.mobility_threshold_spin = QDoubleSpinBox()
        self.mobility_threshold_spin.setRange(0.1, 10.0)
        self.mobility_threshold_spin.setValue(2.11)
        self.mobility_threshold_spin.setDecimals(3)
        self.mobility_threshold_spin.valueChanged.connect(self.valueChanged)
        threshold_layout.addRow("Mobility Threshold:", self.mobility_threshold_spin)

        layout.addRow(self.threshold_params_group)

        # Initialize visibility
        self._on_method_changed("svm")

    def _on_method_changed(self, method: str):
        """Handle classification method change."""
        # Show/hide relevant parameter groups
        if method == "svm":
            self.training_data_edit.setEnabled(True)
            self.browse_button.setEnabled(True)
            self.feature_list.setEnabled(True)
            self.threshold_params_group.setVisible(False)
        elif method == "threshold":
            self.training_data_edit.setEnabled(False)
            self.browse_button.setEnabled(False)
            self.feature_list.setEnabled(False)
            self.threshold_params_group.setVisible(True)

    def _browse_training_data(self):
        """Browse for training data file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Training Data", "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.training_data_edit.setText(file_path)

    def get_value(self) -> Dict[str, Any]:
        """Get classification parameters."""
        selected_features = [
            item.text() for item in self.feature_list.selectedItems()
        ]

        return {
            'classification_method': self.method_combo.currentText(),
            'svm_training_data': self.training_data_edit.text(),
            'svm_features': selected_features,
            'mobility_threshold': self.mobility_threshold_spin.value()
        }

    def set_value(self, params: Dict[str, Any]):
        """Set classification parameters."""
        if 'classification_method' in params:
            self.method_combo.setCurrentText(params['classification_method'])
        if 'svm_training_data' in params:
            self.training_data_edit.setText(params['svm_training_data'])
        if 'svm_features' in params:
            # Clear selection and select specified features
            self.feature_list.clearSelection()
            for i in range(self.feature_list.count()):
                item = self.feature_list.item(i)
                if item.text() in params['svm_features']:
                    item.setSelected(True)
        if 'mobility_threshold' in params:
            self.mobility_threshold_spin.setValue(params['mobility_threshold'])


class ParameterPanelManager(QWidget):
    """Manager widget for all parameter panels."""

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

        # Feature parameters tab
        self.feature_widget = FeatureParametersWidget()
        scroll_feature = QScrollArea()
        scroll_feature.setWidget(self.feature_widget)
        scroll_feature.setWidgetResizable(True)
        self.tab_widget.addTab(scroll_feature, "Features")

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
        self.feature_widget.valueChanged.connect(self.parametersChanged)
        self.classification_widget.valueChanged.connect(self.parametersChanged)

    def get_detection_parameters(self) -> Dict[str, Any]:
        """Get detection parameters."""
        return self.detection_widget.get_value()

    def get_linking_parameters(self) -> Dict[str, Any]:
        """Get linking parameters."""
        return self.linking_widget.get_value()

    def get_feature_parameters(self) -> Dict[str, Any]:
        """Get feature parameters."""
        return self.feature_widget.get_value()

    def get_classification_parameters(self) -> Dict[str, Any]:
        """Get classification parameters."""
        return self.classification_widget.get_value()

    def get_all_parameters(self) -> AnalysisParameters:
        """Get all parameters as AnalysisParameters object."""
        detection_params = self.get_detection_parameters()
        linking_params = self.get_linking_parameters()
        feature_params = self.get_feature_parameters()
        classification_params = self.get_classification_parameters()

        # Create AnalysisParameters object
        params = AnalysisParameters()

        # Update with current values from detection parameters
        params.detection_method = detection_params.get('detection_method', params.detection_method)
        params.detection_sigma = detection_params.get('detection_sigma', params.detection_sigma)
        params.detection_threshold = detection_params.get('detection_threshold', params.detection_threshold)

        # Update with current values from linking parameters
        params.linking_method = linking_params.get('linking_method', params.linking_method)
        params.max_distance = linking_params.get('max_distance', params.max_distance)
        params.max_gap_frames = linking_params.get('max_gap_frames', params.max_gap_frames)
        params.min_track_length = linking_params.get('min_track_length', params.min_track_length)

        # Update with current values from feature parameters
        params.pixel_size = feature_params.get('pixel_size', params.pixel_size)
        params.frame_rate = feature_params.get('frame_rate', params.frame_rate)
        params.mobility_threshold = feature_params.get('mobility_threshold', params.mobility_threshold)

        # Update with current values from classification parameters
        params.svm_training_data = classification_params.get('svm_training_data')
        params.svm_features = classification_params.get('svm_features', params.svm_features)

        # Create a comprehensive parameter dictionary that includes ALL parameters
        # This ensures that parameters not in the AnalysisParameters dataclass are still available
        all_params_dict = {}
        all_params_dict.update(detection_params)
        all_params_dict.update(linking_params)
        all_params_dict.update(feature_params)
        all_params_dict.update(classification_params)

        # Convert AnalysisParameters to dict and merge
        params_dict = asdict(params)
        params_dict.update(all_params_dict)

        # Return the merged dictionary instead of just the AnalysisParameters object
        # This ensures all GUI parameters are available to the analysis engine
        return params_dict

    def set_all_parameters(self, params):
        """Set all parameters from AnalysisParameters object or dict."""
        if hasattr(params, '__dict__'):
            params_dict = asdict(params) if hasattr(params, '__dataclass_fields__') else params.__dict__
        else:
            params_dict = params

        # Split parameters by category and set them
        self.detection_widget.set_value(params_dict)
        self.linking_widget.set_value(params_dict)
        self.feature_widget.set_value(params_dict)
        self.classification_widget.set_value(params_dict)

    def _reset_all_parameters(self):
        """Reset all parameters to defaults."""
        default_params = AnalysisParameters()
        self.set_all_parameters(default_params)
        self.logger.info("Parameters reset to defaults")

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

                self.logger.info(f"Parameters saved to {file_path}")

            except Exception as e:
                self.logger.error(f"Error saving parameters: {e}")

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
                self.logger.info(f"Parameters loaded from {file_path}")

            except Exception as e:
                self.logger.error(f"Error loading parameters: {e}")
