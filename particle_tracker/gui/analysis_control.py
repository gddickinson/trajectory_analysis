#!/usr/bin/env python3
"""
Enhanced Analysis Control Widget
================================

Updated analysis control widget supporting all enhanced analysis capabilities
including multi-radius density analysis, advanced shape metrics, scaled Rg, and more.
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
    QHeaderView, QMenu, QMessageBox, QFileDialog, QFrame,
    QTabWidget, QScrollArea, QGridLayout, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import (
    Qt, QAbstractTableModel, QModelIndex, QVariant, pyqtSignal,
    QSortFilterProxyModel, QTimer
)
from PyQt6.QtGui import QFont, QAction, QIcon, QStandardItemModel, QStandardItem

from particle_tracker.core.data_manager import EnhancedDataManager as DataManager, DataType

# Import AnalysisEngine and AnalysisStep with error handling
try:
    from particle_tracker.core.analysis_engine import AnalysisEngine, AnalysisStep
except ImportError:
    # Create fallback classes
    from enum import Enum

    class AnalysisStep(Enum):
        """Enhanced enumeration of analysis steps."""
        DETECTION = "detection"
        LINKING = "linking"
        FEATURES = "features"
        ENHANCED_FEATURES = "enhanced_features"
        DENSITY_ANALYSIS = "density_analysis"
        ADVANCED_SHAPE = "advanced_shape"
        SCALED_RG = "scaled_rg"
        DIFFUSION_ANALYSIS = "diffusion_analysis"
        PRECISION_ANALYSIS = "precision_analysis"
        CLASSIFICATION = "classification"
        NEAREST_NEIGHBORS = "nearest_neighbors"
        VELOCITY = "velocity"


class AnalysisControlWidget(QWidget):
    """Enhanced widget for controlling analysis operations."""

    def __init__(self, analysis_engine, data_manager: DataManager, parameter_manager=None, parent=None):
        super().__init__(parent)

        self.logger = logging.getLogger(__name__)
        self.analysis_engine = analysis_engine
        self.data_manager = data_manager
        self.parameter_manager = parameter_manager

        self._setup_ui()
        self._connect_signals()

    def set_parameter_manager(self, parameter_manager):
        """Set the parameter manager reference."""
        self.parameter_manager = parameter_manager

    def _setup_ui(self):
        """Setup the enhanced user interface."""
        layout = QVBoxLayout(self)

        # Data selection
        data_group = QGroupBox("Input Data")
        data_layout = QVBoxLayout(data_group)

        self.data_combo = QComboBox()
        self.data_combo.setPlaceholderText("Select data...")
        data_layout.addWidget(QLabel("Data:"))
        data_layout.addWidget(self.data_combo)

        layout.addWidget(data_group)

        # Create analysis tabs for better organization
        self.analysis_tabs = QTabWidget()

        # Basic Analysis Tab
        basic_tab = self._create_basic_analysis_tab()
        self.analysis_tabs.addTab(basic_tab, "Basic Analysis")

        # Enhanced Features Tab
        enhanced_tab = self._create_enhanced_features_tab()
        self.analysis_tabs.addTab(enhanced_tab, "Enhanced Features")

        # Custom Pipeline Tab
        custom_tab = self._create_custom_pipeline_tab()
        self.analysis_tabs.addTab(custom_tab, "Custom Pipeline")

        layout.addWidget(self.analysis_tabs)

        # Quick analysis presets
        presets_group = QGroupBox("Analysis Presets")
        presets_layout = QGridLayout(presets_group)

        self.quick_analysis_btn = QPushButton("🚀 Quick Analysis")
        self.quick_analysis_btn.setToolTip("Detection → Linking → Basic Features → Classification")
        self.quick_analysis_btn.clicked.connect(self._run_quick_analysis)
        presets_layout.addWidget(self.quick_analysis_btn, 0, 0)

        self.comprehensive_btn = QPushButton("🔬 Comprehensive Analysis")
        self.comprehensive_btn.setToolTip("All enhanced features with optimal settings")
        self.comprehensive_btn.clicked.connect(self._run_comprehensive_analysis)
        presets_layout.addWidget(self.comprehensive_btn, 0, 1)

        self.mobility_focus_btn = QPushButton("🏃 Mobility-Focused")
        self.mobility_focus_btn.setToolTip("Scaled Rg, density analysis, and mobility classification")
        self.mobility_focus_btn.clicked.connect(self._run_mobility_focused_analysis)
        presets_layout.addWidget(self.mobility_focus_btn, 1, 0)

        self.shape_focus_btn = QPushButton("📐 Shape-Focused")
        self.shape_focus_btn.setToolTip("Advanced shape metrics and linearity analysis")
        self.shape_focus_btn.clicked.connect(self._run_shape_focused_analysis)
        presets_layout.addWidget(self.shape_focus_btn, 1, 1)

        layout.addWidget(presets_group)

        # Control buttons
        button_layout = QVBoxLayout()

        # Main control buttons
        main_buttons = QHBoxLayout()

        self.run_button = QPushButton("▶️ Run Selected Analysis")
        self.run_button.clicked.connect(self._run_selected_analysis)
        main_buttons.addWidget(self.run_button)

        self.stop_button = QPushButton("⏹️ Stop")
        self.stop_button.clicked.connect(self._stop_analysis)
        self.stop_button.setEnabled(False)
        main_buttons.addWidget(self.stop_button)

        button_layout.addLayout(main_buttons)

        # Utility buttons
        utility_buttons = QHBoxLayout()

        self.suggest_button = QPushButton("💡 Suggest Parameters")
        self.suggest_button.clicked.connect(self.suggest_parameter_improvements)
        utility_buttons.addWidget(self.suggest_button)

        self.validate_button = QPushButton("✅ Validate Setup")
        self.validate_button.clicked.connect(self._validate_analysis_setup)
        utility_buttons.addWidget(self.validate_button)

        button_layout.addLayout(utility_buttons)

        layout.addLayout(button_layout)

        # Progress and status
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready for enhanced analysis")
        self.status_label.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
        layout.addWidget(self.status_label)

        # Analysis summary
        summary_group = QGroupBox("Analysis Summary")
        summary_layout = QVBoxLayout(summary_group)

        self.summary_text = QTextEdit()
        self.summary_text.setMaximumHeight(100)
        self.summary_text.setReadOnly(True)
        self.summary_text.setFont(QFont("Courier", 9))
        summary_layout.addWidget(self.summary_text)

        layout.addWidget(summary_group)

        layout.addStretch()

    def _create_basic_analysis_tab(self) -> QWidget:
        """Create basic analysis steps tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        steps_group = QGroupBox("Basic Analysis Steps")
        steps_layout = QVBoxLayout(steps_group)

        self.detection_cb = QCheckBox("Particle Detection")
        self.detection_cb.setChecked(True)
        steps_layout.addWidget(self.detection_cb)

        self.linking_cb = QCheckBox("Trajectory Linking")
        self.linking_cb.setChecked(True)
        steps_layout.addWidget(self.linking_cb)

        self.basic_features_cb = QCheckBox("Basic Features")
        self.basic_features_cb.setChecked(True)
        steps_layout.addWidget(self.basic_features_cb)

        self.classification_cb = QCheckBox("Classification")
        self.classification_cb.setChecked(True)
        steps_layout.addWidget(self.classification_cb)

        layout.addWidget(steps_group)
        layout.addStretch()

        return tab

    def _create_enhanced_features_tab(self) -> QWidget:
        """Create enhanced features analysis tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Enhanced features group
        enhanced_group = QGroupBox("Enhanced Feature Analysis")
        enhanced_layout = QVBoxLayout(enhanced_group)

        self.enhanced_features_cb = QCheckBox("🔬 Comprehensive Enhanced Features")
        self.enhanced_features_cb.setChecked(True)
        self.enhanced_features_cb.setToolTip("All enhanced features in one step")
        enhanced_layout.addWidget(self.enhanced_features_cb)

        # Individual enhanced features
        individual_group = QGroupBox("Individual Enhanced Features")
        individual_layout = QVBoxLayout(individual_group)

        self.density_analysis_cb = QCheckBox("📊 Multi-Radius Density Analysis")
        self.density_analysis_cb.setChecked(False)
        self.density_analysis_cb.setToolTip("Neighbor counting at 3,5,10,20,30 pixel radii")
        individual_layout.addWidget(self.density_analysis_cb)

        self.advanced_shape_cb = QCheckBox("📐 Advanced Shape Metrics")
        self.advanced_shape_cb.setChecked(False)
        self.advanced_shape_cb.setToolTip("Eigenvalue ratios, linearity, directionality")
        individual_layout.addWidget(self.advanced_shape_cb)

        self.scaled_rg_cb = QCheckBox("🎯 Scaled Radius of Gyration")
        self.scaled_rg_cb.setChecked(False)
        self.scaled_rg_cb.setToolTip("Golan & Sherman mobility classification")
        individual_layout.addWidget(self.scaled_rg_cb)

        self.diffusion_analysis_cb = QCheckBox("🌊 Diffusion Analysis")
        self.diffusion_analysis_cb.setChecked(False)
        self.diffusion_analysis_cb.setToolTip("MSD, diffusion coefficients, origin analysis")
        individual_layout.addWidget(self.diffusion_analysis_cb)

        self.precision_analysis_cb = QCheckBox("🎱 Localization Precision")
        self.precision_analysis_cb.setChecked(False)
        self.precision_analysis_cb.setToolTip("Position variability and measurement precision")
        individual_layout.addWidget(self.precision_analysis_cb)

        self.nn_cb = QCheckBox("Nearest Neighbors")
        self.nn_cb.setChecked(False)
        individual_layout.addWidget(self.nn_cb)

        self.velocity_cb = QCheckBox("Velocity Analysis")
        self.velocity_cb.setChecked(False)
        individual_layout.addWidget(self.velocity_cb)

        enhanced_layout.addWidget(individual_group)

        # Note about comprehensive vs individual
        note_label = QLabel("💡 Tip: Use 'Comprehensive Enhanced Features' for best performance, "
                           "or select individual features for specific analysis needs.")
        note_label.setStyleSheet("QLabel { color: gray; font-size: 9px; }")
        note_label.setWordWrap(True)
        enhanced_layout.addWidget(note_label)

        layout.addWidget(enhanced_group)
        layout.addStretch()

        return tab

    def _create_custom_pipeline_tab(self) -> QWidget:
        """Create custom analysis pipeline tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Pipeline builder
        pipeline_group = QGroupBox("Custom Analysis Pipeline")
        pipeline_layout = QVBoxLayout(pipeline_group)

        instructions = QLabel("Build a custom analysis pipeline by selecting steps in order:")
        instructions.setStyleSheet("QLabel { font-weight: bold; }")
        pipeline_layout.addWidget(instructions)

        # Available steps
        self.available_steps = QListWidget()
        self.available_steps.setMaximumHeight(150)

        # Add all available steps
        steps_info = [
            ("detection", "Particle Detection", "Detect particles in images"),
            ("linking", "Trajectory Linking", "Link particles into trajectories"),
            ("features", "Basic Features", "Calculate basic trajectory features"),
            ("enhanced_features", "Enhanced Features", "Comprehensive enhanced features"),
            ("density_analysis", "Density Analysis", "Multi-radius neighbor counting"),
            ("advanced_shape", "Advanced Shape", "Shape and linearity metrics"),
            ("scaled_rg", "Scaled Rg", "Scaled radius of gyration"),
            ("diffusion_analysis", "Diffusion Analysis", "MSD and diffusion metrics"),
            ("precision_analysis", "Precision Analysis", "Localization precision"),
            ("classification", "Classification", "Trajectory classification"),
            ("nearest_neighbors", "Nearest Neighbors", "Traditional NN distance"),
            ("velocity", "Velocity Analysis", "Velocity and motion metrics")
        ]

        for step_id, name, description in steps_info:
            item = QListWidgetItem(f"{name} - {description}")
            item.setData(Qt.ItemDataRole.UserRole, step_id)
            self.available_steps.addItem(item)

        pipeline_layout.addWidget(QLabel("Available Steps:"))
        pipeline_layout.addWidget(self.available_steps)

        # Selected pipeline
        self.selected_pipeline = QListWidget()
        self.selected_pipeline.setMaximumHeight(150)
        pipeline_layout.addWidget(QLabel("Selected Pipeline:"))
        pipeline_layout.addWidget(self.selected_pipeline)

        # Pipeline controls
        pipeline_controls = QHBoxLayout()

        self.add_step_btn = QPushButton("Add Step →")
        self.add_step_btn.clicked.connect(self._add_pipeline_step)
        pipeline_controls.addWidget(self.add_step_btn)

        self.remove_step_btn = QPushButton("← Remove Step")
        self.remove_step_btn.clicked.connect(self._remove_pipeline_step)
        pipeline_controls.addWidget(self.remove_step_btn)

        self.clear_pipeline_btn = QPushButton("Clear All")
        self.clear_pipeline_btn.clicked.connect(self._clear_pipeline)
        pipeline_controls.addWidget(self.clear_pipeline_btn)

        pipeline_layout.addLayout(pipeline_controls)

        # Pipeline presets
        preset_controls = QHBoxLayout()

        self.load_basic_preset_btn = QPushButton("Load Basic Preset")
        self.load_basic_preset_btn.clicked.connect(self._load_basic_preset)
        preset_controls.addWidget(self.load_basic_preset_btn)

        self.load_full_preset_btn = QPushButton("Load Full Preset")
        self.load_full_preset_btn.clicked.connect(self._load_full_preset)
        preset_controls.addWidget(self.load_full_preset_btn)

        pipeline_layout.addLayout(preset_controls)

        layout.addWidget(pipeline_group)
        layout.addStretch()

        return tab

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

        # Connect step checkboxes to update summary
        checkboxes = [
            self.detection_cb, self.linking_cb, self.basic_features_cb, self.classification_cb,
            self.enhanced_features_cb, self.density_analysis_cb, self.advanced_shape_cb,
            self.scaled_rg_cb, self.diffusion_analysis_cb, self.precision_analysis_cb,
            self.nn_cb, self.velocity_cb
        ]

        for cb in checkboxes:
            cb.toggled.connect(self._update_analysis_summary)

    def _update_data_list(self):
        """Update the data selection combo box."""
        current_text = self.data_combo.currentText()

        self.data_combo.clear()
        data_names = self.data_manager.get_data_names()
        self.data_combo.addItems(data_names)

        # Restore selection if possible
        if current_text in data_names:
            self.data_combo.setCurrentText(current_text)

    def _get_selected_steps(self) -> List[AnalysisStep]:
        """Get currently selected analysis steps."""
        steps = []

        # Basic analysis steps
        if self.detection_cb.isChecked():
            steps.append(AnalysisStep.DETECTION)
        if self.linking_cb.isChecked():
            steps.append(AnalysisStep.LINKING)
        if self.basic_features_cb.isChecked():
            steps.append(AnalysisStep.FEATURES)

        # Enhanced features - either comprehensive or individual
        if self.enhanced_features_cb.isChecked():
            steps.append(AnalysisStep.ENHANCED_FEATURES)
        else:
            # Individual enhanced features
            if self.density_analysis_cb.isChecked():
                steps.append(AnalysisStep.DENSITY_ANALYSIS)
            if self.advanced_shape_cb.isChecked():
                steps.append(AnalysisStep.ADVANCED_SHAPE)
            if self.scaled_rg_cb.isChecked():
                steps.append(AnalysisStep.SCALED_RG)
            if self.diffusion_analysis_cb.isChecked():
                steps.append(AnalysisStep.DIFFUSION_ANALYSIS)
            if self.precision_analysis_cb.isChecked():
                steps.append(AnalysisStep.PRECISION_ANALYSIS)

        # Additional steps
        if self.classification_cb.isChecked():
            steps.append(AnalysisStep.CLASSIFICATION)
        if self.nn_cb.isChecked():
            steps.append(AnalysisStep.NEAREST_NEIGHBORS)
        if self.velocity_cb.isChecked():
            steps.append(AnalysisStep.VELOCITY)

        return steps

    def _get_custom_pipeline_steps(self) -> List[AnalysisStep]:
        """Get steps from custom pipeline."""
        steps = []
        for i in range(self.selected_pipeline.count()):
            item = self.selected_pipeline.item(i)
            step_id = item.data(Qt.ItemDataRole.UserRole)
            
            # Map step_id to AnalysisStep enum
            step_mapping = {
                'detection': AnalysisStep.DETECTION,
                'linking': AnalysisStep.LINKING,
                'features': AnalysisStep.FEATURES,
                'enhanced_features': AnalysisStep.ENHANCED_FEATURES,
                'density_analysis': AnalysisStep.DENSITY_ANALYSIS,
                'advanced_shape': AnalysisStep.ADVANCED_SHAPE,
                'scaled_rg': AnalysisStep.SCALED_RG,
                'diffusion_analysis': AnalysisStep.DIFFUSION_ANALYSIS,
                'precision_analysis': AnalysisStep.PRECISION_ANALYSIS,
                'classification': AnalysisStep.CLASSIFICATION,
                'nearest_neighbors': AnalysisStep.NEAREST_NEIGHBORS,
                'velocity': AnalysisStep.VELOCITY
            }
            
            if step_id in step_mapping:
                steps.append(step_mapping[step_id])

        return steps

    def _run_selected_analysis(self):
        """Run the selected analysis steps."""
        # Get selected data
        data_name = self.data_combo.currentText()
        if not data_name:
            QMessageBox.warning(self, "Warning", "Please select input data")
            return

        data = self.data_manager.get_data(data_name)
        if data is None:
            QMessageBox.warning(self, "Warning", "Selected data not found")
            return

        # Determine which steps to run
        current_tab = self.analysis_tabs.currentIndex()
        
        if current_tab == 2:  # Custom pipeline
            steps = self._get_custom_pipeline_steps()
        else:  # Basic or Enhanced tabs
            steps = self._get_selected_steps()

        if not steps:
            QMessageBox.warning(self, "Warning", "Please select at least one analysis step")
            return

        # Get parameters from parameter manager
        if self.parameter_manager is not None:
            try:
                parameters = self.parameter_manager.get_all_parameters()
                parameters = self._optimize_parameters(parameters, data, steps)
                self.logger.info("Using enhanced parameters from parameter manager")
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
                "Enhanced analysis engine not fully loaded. Please check the console for any import errors."
            )

    # Preset analysis methods
    def _run_quick_analysis(self):
        """Run quick analysis preset."""
        self._reset_all_checkboxes()
        self.detection_cb.setChecked(True)
        self.linking_cb.setChecked(True)
        self.basic_features_cb.setChecked(True)
        self.classification_cb.setChecked(True)
        self.analysis_tabs.setCurrentIndex(0)  # Switch to basic tab
        self._run_selected_analysis()

    def _run_comprehensive_analysis(self):
        """Run comprehensive analysis preset."""
        self._reset_all_checkboxes()
        self.detection_cb.setChecked(True)
        self.linking_cb.setChecked(True)
        self.enhanced_features_cb.setChecked(True)
        self.classification_cb.setChecked(True)
        self.analysis_tabs.setCurrentIndex(1)  # Switch to enhanced tab
        self._run_selected_analysis()

    def _run_mobility_focused_analysis(self):
        """Run mobility-focused analysis preset."""
        self._reset_all_checkboxes()
        self.detection_cb.setChecked(True)
        self.linking_cb.setChecked(True)
        self.scaled_rg_cb.setChecked(True)
        self.density_analysis_cb.setChecked(True)
        self.diffusion_analysis_cb.setChecked(True)
        self.classification_cb.setChecked(True)
        self.analysis_tabs.setCurrentIndex(1)  # Switch to enhanced tab
        self._run_selected_analysis()

    def _run_shape_focused_analysis(self):
        """Run shape-focused analysis preset."""
        self._reset_all_checkboxes()
        self.detection_cb.setChecked(True)
        self.linking_cb.setChecked(True)
        self.advanced_shape_cb.setChecked(True)
        self.precision_analysis_cb.setChecked(True)
        self.classification_cb.setChecked(True)
        self.analysis_tabs.setCurrentIndex(1)  # Switch to enhanced tab
        self._run_selected_analysis()

    def _reset_all_checkboxes(self):
        """Reset all analysis step checkboxes."""
        checkboxes = [
            self.detection_cb, self.linking_cb, self.basic_features_cb, self.classification_cb,
            self.enhanced_features_cb, self.density_analysis_cb, self.advanced_shape_cb,
            self.scaled_rg_cb, self.diffusion_analysis_cb, self.precision_analysis_cb,
            self.nn_cb, self.velocity_cb
        ]
        for cb in checkboxes:
            cb.setChecked(False)

    # Custom pipeline methods
    def _add_pipeline_step(self):
        """Add selected step to pipeline."""
        current_item = self.available_steps.currentItem()
        if current_item:
            step_id = current_item.data(Qt.ItemDataRole.UserRole)
            step_text = current_item.text()
            
            # Check if step already in pipeline
            for i in range(self.selected_pipeline.count()):
                item = self.selected_pipeline.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == step_id:
                    return  # Already added
            
            # Add to pipeline
            new_item = QListWidgetItem(step_text)
            new_item.setData(Qt.ItemDataRole.UserRole, step_id)
            self.selected_pipeline.addItem(new_item)

    def _remove_pipeline_step(self):
        """Remove selected step from pipeline."""
        current_item = self.selected_pipeline.currentItem()
        if current_item:
            row = self.selected_pipeline.row(current_item)
            self.selected_pipeline.takeItem(row)

    def _clear_pipeline(self):
        """Clear all steps from pipeline."""
        self.selected_pipeline.clear()

    def _load_basic_preset(self):
        """Load basic analysis preset into custom pipeline."""
        self._clear_pipeline()
        basic_steps = ['detection', 'linking', 'features', 'classification']
        self._add_steps_to_pipeline(basic_steps)

    def _load_full_preset(self):
        """Load full analysis preset into custom pipeline."""
        self._clear_pipeline()
        full_steps = ['detection', 'linking', 'enhanced_features', 'classification']
        self._add_steps_to_pipeline(full_steps)

    def _add_steps_to_pipeline(self, step_ids: List[str]):
        """Add list of step IDs to pipeline."""
        for step_id in step_ids:
            # Find corresponding item in available steps
            for i in range(self.available_steps.count()):
                item = self.available_steps.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == step_id:
                    new_item = QListWidgetItem(item.text())
                    new_item.setData(Qt.ItemDataRole.UserRole, step_id)
                    self.selected_pipeline.addItem(new_item)
                    break

    def _validate_analysis_setup(self):
        """Validate current analysis setup."""
        issues = []
        suggestions = []

        # Check data selection
        if not self.data_combo.currentText():
            issues.append("❌ No input data selected")

        # Check analysis steps
        steps = self._get_selected_steps() if self.analysis_tabs.currentIndex() != 2 else self._get_custom_pipeline_steps()
        
        if not steps:
            issues.append("❌ No analysis steps selected")
        else:
            # Check step dependencies
            if AnalysisStep.LINKING in steps and AnalysisStep.DETECTION not in steps:
                issues.append("❌ Linking requires Detection to be selected")
            
            if any(step in steps for step in [AnalysisStep.ENHANCED_FEATURES, AnalysisStep.DENSITY_ANALYSIS, 
                                            AnalysisStep.ADVANCED_SHAPE, AnalysisStep.SCALED_RG]) and AnalysisStep.LINKING not in steps:
                issues.append("❌ Enhanced features require trajectory data (Detection + Linking)")
            
            if AnalysisStep.CLASSIFICATION in steps and not any(step in steps for step in [AnalysisStep.FEATURES, AnalysisStep.ENHANCED_FEATURES]):
                issues.append("❌ Classification requires feature calculation")

        # Check parameters
        if self.parameter_manager:
            try:
                params = self.parameter_manager.get_all_parameters()
                
                # Check SVM training data if SVM classification selected
                if params.get('classification_method') == 'svm':
                    training_data = params.get('svm_training_data')
                    if not training_data or not Path(training_data).exists():
                        issues.append("❌ SVM classification requires valid training data file")

                # Check detection method compatibility
                detection_method = params.get('detection_method')
                linking_method = params.get('linking_method')
                
                if detection_method == 'trackpy' and linking_method != 'trackpy':
                    suggestions.append("💡 Consider using 'trackpy' linking with 'trackpy' detection")
                
                # Check distance parameters
                max_distance = params.get('max_distance', 0)
                if max_distance > 10:
                    suggestions.append("⚠️ Large max_distance may cause false linkages")

            except Exception as e:
                issues.append(f"❌ Error checking parameters: {e}")

        # Show results
        if issues or suggestions:
            result_text = "Analysis Setup Validation Results:\n\n"
            
            if issues:
                result_text += "🚨 Issues to Fix:\n"
                result_text += "\n".join(issues) + "\n\n"
            
            if suggestions:
                result_text += "💡 Suggestions:\n"
                result_text += "\n".join(suggestions)
            
            QMessageBox.warning(self, "Validation Results", result_text)
        else:
            QMessageBox.information(self, "Validation Results", "✅ Analysis setup looks good!")

    def _update_analysis_summary(self):
        """Update the analysis summary text."""
        steps = self._get_selected_steps() if self.analysis_tabs.currentIndex() != 2 else self._get_custom_pipeline_steps()
        
        if not steps:
            self.summary_text.setText("No analysis steps selected.")
            return

        summary_lines = [
            f"Selected Analysis Pipeline ({len(steps)} steps):",
            "=" * 40
        ]

        step_descriptions = {
            AnalysisStep.DETECTION: "🔍 Detect particles in images",
            AnalysisStep.LINKING: "🔗 Link particles into trajectories",
            AnalysisStep.FEATURES: "📊 Calculate basic trajectory features",
            AnalysisStep.ENHANCED_FEATURES: "🔬 Comprehensive enhanced analysis",
            AnalysisStep.DENSITY_ANALYSIS: "📈 Multi-radius density analysis",
            AnalysisStep.ADVANCED_SHAPE: "📐 Advanced shape & linearity metrics",
            AnalysisStep.SCALED_RG: "🎯 Scaled radius of gyration (mobility)",
            AnalysisStep.DIFFUSION_ANALYSIS: "🌊 Diffusion & MSD analysis",
            AnalysisStep.PRECISION_ANALYSIS: "🎱 Localization precision metrics",
            AnalysisStep.CLASSIFICATION: "🏷️ Trajectory classification",
            AnalysisStep.NEAREST_NEIGHBORS: "📏 Nearest neighbor distances",
            AnalysisStep.VELOCITY: "🏃 Velocity & motion analysis"
        }

        for i, step in enumerate(steps, 1):
            description = step_descriptions.get(step, step.value)
            summary_lines.append(f"{i}. {description}")

        self.summary_text.setText("\n".join(summary_lines))

    def _optimize_parameters(self, parameters: Dict[str, Any], data, steps) -> Dict[str, Any]:
        """Optimize parameters based on data characteristics and selected steps."""
        optimized = parameters.copy()

        # Enhanced parameter optimization
        if AnalysisStep.ENHANCED_FEATURES in steps or AnalysisStep.DENSITY_ANALYSIS in steps:
            # Ensure density analysis is enabled for enhanced features
            optimized['calculate_density'] = True
            
            # Optimize density radii based on expected particle density
            if isinstance(data, np.ndarray) and len(data.shape) >= 2:
                # Estimate optimal radii based on image size
                img_size = data.shape[-1] * data.shape[-2]
                if img_size < 100000:  # Small images
                    optimized['density_radii'] = [3, 5, 10]
                elif img_size > 500000:  # Large images
                    optimized['density_radii'] = [3, 5, 10, 20, 30, 50]

        # Auto-match detection and linking methods
        if AnalysisStep.DETECTION in steps and AnalysisStep.LINKING in steps:
            detection_method = parameters.get('detection_method', 'threshold')
            linking_method = parameters.get('linking_method', 'nearest_neighbor')

            if detection_method == 'trackpy' and linking_method != 'trackpy':
                optimized['linking_method'] = 'trackpy'
                self.logger.info("Auto-matched trackpy linking with trackpy detection")

        # Optimize for shape analysis
        if AnalysisStep.ADVANCED_SHAPE in steps:
            # Ensure advanced shape analysis is enabled
            optimized['calculate_advanced_shape'] = True
            
            # Optimize thresholds for more sensitive detection
            if optimized.get('linear_eigenvalue_threshold', 20.0) < 10.0:
                optimized['linear_eigenvalue_threshold'] = 15.0
                self.logger.info("Adjusted eigenvalue threshold for better linearity detection")

        return optimized

    def suggest_parameter_improvements(self):
        """Show suggestions for parameter improvements."""
        if self.parameter_manager is None:
            QMessageBox.information(self, "Parameter Suggestions", 
                                  "No parameter manager available for suggestions.")
            return

        try:
            current_params = self.parameter_manager.get_all_parameters()
            suggestions = []

            # Enhanced suggestions based on current analysis setup
            steps = self._get_selected_steps()

            # Detection-specific suggestions
            if AnalysisStep.DETECTION in steps:
                detection_method = current_params.get('detection_method', 'threshold')
                if detection_method == 'threshold':
                    suggestions.append("💡 Consider 'trackpy' detection for better performance with dense data")

            # Linking suggestions
            if AnalysisStep.LINKING in steps:
                max_distance = current_params.get('max_distance', 5.0)
                if max_distance > 5.0:
                    suggestions.append("⚠️ Large max_distance may cause false linkages. Start with 2-3 pixels.")

            # Enhanced features suggestions
            if AnalysisStep.ENHANCED_FEATURES in steps or any(step in steps for step in [
                AnalysisStep.DENSITY_ANALYSIS, AnalysisStep.ADVANCED_SHAPE, AnalysisStep.SCALED_RG
            ]):
                if not current_params.get('calculate_density', True):
                    suggestions.append("🔬 Enable density analysis for comprehensive enhanced features")
                
                density_radii = current_params.get('density_radii', [])
                if len(density_radii) < 3:
                    suggestions.append("📊 Use at least 3 density radii (e.g., 3,5,10) for better analysis")

            # Mobility analysis suggestions
            if AnalysisStep.SCALED_RG in steps:
                mobility_threshold = current_params.get('mobility_threshold', 2.11)
                if abs(mobility_threshold - 2.11) > 0.1:
                    suggestions.append("🎯 Standard mobility threshold is 2.11 (Golan & Sherman)")

            # Classification suggestions
            if AnalysisStep.CLASSIFICATION in steps:
                classification_method = current_params.get('classification_method', 'threshold')
                if classification_method == 'svm':
                    training_data = current_params.get('svm_training_data')
                    if not training_data:
                        suggestions.append("🤖 SVM classification requires training data file")

            # General suggestions
            if len(steps) > 5:
                suggestions.append("⚡ Consider using 'Comprehensive Enhanced Features' for better performance")

            if not suggestions:
                suggestions.append("✅ Current parameters look good for selected analysis!")

            suggestion_text = "Enhanced Parameter Suggestions:\n\n" + "\n\n".join(suggestions)
            QMessageBox.information(self, "Parameter Suggestions", suggestion_text)

        except Exception as e:
            self.logger.error(f"Error generating parameter suggestions: {e}")
            QMessageBox.warning(self, "Error", f"Error generating suggestions: {e}")

    def _get_default_parameters(self):
        """Get default parameters as fallback."""
        try:
            from particle_tracker.core.analysis_engine import AnalysisParameters
            return AnalysisParameters()
        except ImportError:
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
                'mobility_threshold': 2.11,
                'calculate_density': True,
                'density_radii': [3, 5, 10, 20, 30],
                'calculate_advanced_shape': True,
                'calculate_scaled_rg': True,
                'calculate_diffusion': True,
                'calculate_precision': True
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
        self.status_label.setText(f"Running enhanced analysis: {', '.join(steps)}")

    def _on_analysis_completed(self, result: Any):
        """Handle analysis completed."""
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Enhanced analysis completed successfully")

        # Update summary with results
        if isinstance(result, pd.DataFrame):
            n_tracks = result['track_number'].nunique() if 'track_number' in result.columns else 0
            n_points = len(result)
            self.summary_text.append(f"\n\nResults: {n_tracks} tracks, {n_points} localizations")

    def _update_progress(self, message: str, percentage: int):
        """Update progress display."""
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)

    def _on_analysis_error(self, error_message: str):
        """Handle analysis error."""
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Enhanced analysis failed")

        QMessageBox.critical(self, "Enhanced Analysis Error", error_message)


# For backward compatibility, alias the enhanced control widget
EnhancedAnalysisControlWidget = AnalysisControlWidget