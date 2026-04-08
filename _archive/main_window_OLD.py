#!/usr/bin/env python3
"""
Enhanced Main Window GUI Module
===============================

Fully integrated main application window with all advanced analysis capabilities
including enhanced features, batch processing, autocorrelation analysis, and
comprehensive visualization options.
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
    QHeaderView, QDialog, QDialogButtonBox, QInputDialog
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
    from .parameter_panels import ParameterPanel as ParameterPanelManager

try:
    from .data_browser import DataBrowserWidget
except ImportError:
    # Create a simple fallback
    class DataBrowserWidget(QWidget):
        def __init__(self, data_manager):
            super().__init__()
            self.data_manager = data_manager
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("Data Browser"))

        def get_selected_data_name(self):
            names = self.data_manager.get_data_names() if hasattr(self.data_manager, 'get_data_names') else []
            return names[0] if names else None

try:
    from .analysis_control import EnhancedAnalysisControlWidget
except ImportError:
    from .analysis_control import AnalysisControlWidget as EnhancedAnalysisControlWidget

try:
    from .logging_widget import LoggingWidget
except ImportError:
    # Create a simple fallback
    class LoggingWidget(QTextEdit):
        def __init__(self):
            super().__init__()
            self.setReadOnly(True)

# Core imports
from ..core.analysis_engine import AnalysisParameters, AnalysisStep
from ..core.batch_analysis import BatchAnalysisManager, BatchExperiment
from ..analysis.autocorrelation_analysis import DirectionAutocorrelationAnalyzer


class BatchExperimentDialog(QDialog):
    """Dialog for creating and configuring batch experiments."""

    def __init__(self, batch_manager: BatchAnalysisManager, parent=None):
        super().__init__(parent)
        self.batch_manager = batch_manager
        self.setWindowTitle("Batch Experiment Configuration")
        self.setMinimumSize(800, 600)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)

        # Experiment info
        info_group = QGroupBox("Experiment Information")
        info_layout = QFormLayout(info_group)

        self.name_edit = QLineEdit()
        info_layout.addRow("Experiment Name:", self.name_edit)

        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(80)
        info_layout.addRow("Description:", self.description_edit)

        self.output_dir_edit = QLineEdit()
        self.output_browse_btn = QPushButton("Browse...")
        self.output_browse_btn.clicked.connect(self._browse_output_dir)

        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(self.output_browse_btn)
        info_layout.addRow("Output Directory:", output_layout)

        layout.addWidget(info_group)

        # File selection
        files_group = QGroupBox("Files and Conditions")
        files_layout = QVBoxLayout(files_group)

        # File list
        self.files_table = QTableView()
        files_layout.addWidget(self.files_table)

        # File controls
        file_controls = QHBoxLayout()

        self.add_files_btn = QPushButton("Add Files...")
        self.add_files_btn.clicked.connect(self._add_files)
        file_controls.addWidget(self.add_files_btn)

        self.remove_files_btn = QPushButton("Remove Selected")
        self.remove_files_btn.clicked.connect(self._remove_files)
        file_controls.addWidget(self.remove_files_btn)

        self.set_condition_btn = QPushButton("Set Condition...")
        self.set_condition_btn.clicked.connect(self._set_condition)
        file_controls.addWidget(self.set_condition_btn)

        file_controls.addStretch()
        files_layout.addLayout(file_controls)

        layout.addWidget(files_group)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # Initialize file table model
        self._setup_file_table()

    def _setup_file_table(self):
        """Setup the file table model."""
        self.file_model = QStandardItemModel(0, 3)
        self.file_model.setHorizontalHeaderLabels(["File Path", "Condition", "Status"])
        self.files_table.setModel(self.file_model)
        self.files_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

    def _browse_output_dir(self):
        """Browse for output directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def _add_files(self):
        """Add files to the experiment."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Files for Batch Analysis", "",
            "Data Files (*.tif *.tiff *.csv *.txt *.json);;All Files (*)"
        )

        for file_path in file_paths:
            # Add row to table
            row = self.file_model.rowCount()
            self.file_model.insertRow(row)

            self.file_model.setItem(row, 0, QStandardItem(file_path))
            self.file_model.setItem(row, 1, QStandardItem("Condition1"))  # Default condition
            self.file_model.setItem(row, 2, QStandardItem("Pending"))

    def _remove_files(self):
        """Remove selected files."""
        selection = self.files_table.selectionModel().selectedRows()
        for index in sorted(selection, reverse=True):
            self.file_model.removeRow(index.row())

    def _set_condition(self):
        """Set condition for selected files."""
        selection = self.files_table.selectionModel().selectedRows()
        if not selection:
            QMessageBox.warning(self, "Warning", "Please select files to modify")
            return

        condition, ok = QInputDialog.getText(self, "Set Condition", "Condition name:")
        if ok and condition:
            for index in selection:
                self.file_model.setItem(index.row(), 1, QStandardItem(condition))

    def get_experiment_config(self) -> Dict[str, Any]:
        """Get the configured experiment details."""
        files = []
        for row in range(self.file_model.rowCount()):
            file_path = self.file_model.item(row, 0).text()
            condition = self.file_model.item(row, 1).text()
            files.append((file_path, condition))

        return {
            'name': self.name_edit.text(),
            'description': self.description_edit.toPlainText(),
            'output_directory': self.output_dir_edit.text(),
            'files': files
        }


class AutocorrelationAnalysisDialog(QDialog):
    """Dialog for configuring and running autocorrelation analysis."""

    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.setWindowTitle("Direction Autocorrelation Analysis")
        self.setMinimumSize(600, 400)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)

        # Data selection
        data_group = QGroupBox("Data Selection")
        data_layout = QFormLayout(data_group)

        self.data_combo = QComboBox()
        self._update_data_list()
        data_layout.addRow("Trajectory Data:", self.data_combo)

        layout.addWidget(data_group)

        # Analysis parameters
        params_group = QGroupBox("Analysis Parameters")
        params_layout = QFormLayout(params_group)

        self.max_lag_spin = QSpinBox()
        self.max_lag_spin.setRange(1, 100)
        self.max_lag_spin.setValue(20)
        self.max_lag_spin.setToolTip("Maximum lag time for autocorrelation (0 = auto)")
        params_layout.addRow("Max Lag:", self.max_lag_spin)

        self.min_track_length_spin = QSpinBox()
        self.min_track_length_spin.setRange(3, 1000)
        self.min_track_length_spin.setValue(5)
        params_layout.addRow("Min Track Length:", self.min_track_length_spin)

        layout.addWidget(params_group)

        # Output options
        output_group = QGroupBox("Output Options")
        output_layout = QVBoxLayout(output_group)

        self.save_plots_cb = QCheckBox("Generate Plots")
        self.save_plots_cb.setChecked(True)
        output_layout.addWidget(self.save_plots_cb)

        self.export_results_cb = QCheckBox("Export Results CSV")
        self.export_results_cb.setChecked(True)
        output_layout.addWidget(self.export_results_cb)

        # Output directory
        output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_browse_btn = QPushButton("Browse...")
        self.output_browse_btn.clicked.connect(self._browse_output_dir)
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(self.output_browse_btn)

        output_layout.addLayout(output_dir_layout)

        layout.addWidget(output_group)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _update_data_list(self):
        """Update the data selection combo."""
        self.data_combo.clear()
        if hasattr(self.data_manager, 'get_data_names'):
            trajectory_data = [name for name in self.data_manager.get_data_names()
                              if self._is_trajectory_data(name)]
            self.data_combo.addItems(trajectory_data)

    def _is_trajectory_data(self, data_name: str) -> bool:
        """Check if data contains trajectory information."""
        if hasattr(self.data_manager, 'get_data'):
            data = self.data_manager.get_data(data_name)
            return (isinstance(data, pd.DataFrame) and
                    'track_number' in data.columns and
                    'x' in data.columns and 'y' in data.columns)
        return False

    def _browse_output_dir(self):
        """Browse for output directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration."""
        return {
            'data_name': self.data_combo.currentText(),
            'max_lag': self.max_lag_spin.value() if self.max_lag_spin.value() > 0 else None,
            'min_track_length': self.min_track_length_spin.value(),
            'save_plots': self.save_plots_cb.isChecked(),
            'export_results': self.export_results_cb.isChecked(),
            'output_directory': self.output_dir_edit.text()
        }


class EnhancedMainWindow(QMainWindow):
    """Enhanced main application window with all advanced features."""

    def __init__(self, data_manager, analysis_engine, project_manager, config):
        super().__init__()

        self.logger = logging.getLogger(__name__)

        # Store core components
        self.data_manager = data_manager
        self.analysis_engine = analysis_engine
        self.project_manager = project_manager
        self.config = config

        # Enhanced components
        self.batch_manager = BatchAnalysisManager(analysis_engine, data_manager)
        self.autocorr_analyzer = DirectionAutocorrelationAnalyzer()

        # Settings
        self.settings = QSettings()

        # Initialize UI
        self._setup_ui()
        self._connect_signals()
        self._setup_default_parameters()
        self._restore_settings()

        self.logger.info("Enhanced main window initialized")

    def _setup_ui(self):
        """Setup the enhanced user interface."""
        self.setWindowTitle("Enhanced Particle Tracking Analyzer")
        self.setMinimumSize(1400, 900)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout with enhanced organization
        main_layout = QHBoxLayout(central_widget)

        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)

        # Create left panel (enhanced controls and data browser)
        left_panel = self._create_enhanced_left_panel()
        main_splitter.addWidget(left_panel)

        # Create center panel (enhanced visualization)
        center_panel = self._create_enhanced_center_panel()
        main_splitter.addWidget(center_panel)

        # Create right panel (enhanced analysis results)
        right_panel = self._create_enhanced_right_panel()
        main_splitter.addWidget(right_panel)

        # Set splitter proportions
        main_splitter.setSizes([350, 700, 350])

        # Create enhanced menu bar
        self._create_enhanced_menu_bar()

        # Create enhanced tool bar
        self._create_enhanced_tool_bar()

        # Create enhanced status bar
        self._create_enhanced_status_bar()

        # Create enhanced dock widgets
        self._create_enhanced_dock_widgets()

    def _create_enhanced_left_panel(self) -> QWidget:
        """Create the enhanced left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Create tabbed interface for better organization
        left_tabs = QTabWidget()

        # Data & Analysis tab
        data_analysis_tab = QWidget()
        data_analysis_layout = QVBoxLayout(data_analysis_tab)

        # Data browser
        self.data_browser = DataBrowserWidget(self.data_manager)
        data_group = QGroupBox("Data Browser")
        data_layout = QVBoxLayout(data_group)
        data_layout.addWidget(self.data_browser)
        data_analysis_layout.addWidget(data_group)

        # Enhanced analysis control
        self.analysis_control = EnhancedAnalysisControlWidget(
            self.analysis_engine, self.data_manager
        )
        analysis_group = QGroupBox("Analysis Control")
        analysis_layout = QVBoxLayout(analysis_group)
        analysis_layout.addWidget(self.analysis_control)
        data_analysis_layout.addWidget(analysis_group)

        left_tabs.addTab(data_analysis_tab, "Data & Analysis")

        # Parameters tab
        self.parameter_manager = ParameterPanelManager()

        param_scroll = QScrollArea()
        param_scroll.setWidget(self.parameter_manager)
        param_scroll.setWidgetResizable(True)
        left_tabs.addTab(param_scroll, "Parameters")

        # Batch Processing tab
        batch_tab = self._create_batch_processing_tab()
        left_tabs.addTab(batch_tab, "Batch Processing")

        layout.addWidget(left_tabs)
        return panel

    def _create_batch_processing_tab(self) -> QWidget:
        """Create batch processing tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Experiment management
        exp_group = QGroupBox("Experiment Management")
        exp_layout = QVBoxLayout(exp_group)

        # Current experiments list
        self.experiments_list = QComboBox()
        exp_layout.addWidget(QLabel("Current Experiments:"))
        exp_layout.addWidget(self.experiments_list)

        # Experiment controls
        exp_controls = QHBoxLayout()

        self.new_experiment_btn = QPushButton("New Experiment...")
        self.new_experiment_btn.clicked.connect(self._create_new_experiment)
        exp_controls.addWidget(self.new_experiment_btn)

        self.load_experiment_btn = QPushButton("Load...")
        self.load_experiment_btn.clicked.connect(self._load_experiment)
        exp_controls.addWidget(self.load_experiment_btn)

        self.save_experiment_btn = QPushButton("Save...")
        self.save_experiment_btn.clicked.connect(self._save_experiment)
        exp_controls.addWidget(self.save_experiment_btn)

        exp_layout.addLayout(exp_controls)

        layout.addWidget(exp_group)

        # Batch execution
        exec_group = QGroupBox("Batch Execution")
        exec_layout = QVBoxLayout(exec_group)

        self.run_batch_btn = QPushButton("🚀 Run Batch Analysis")
        self.run_batch_btn.clicked.connect(self._run_batch_analysis)
        exec_layout.addWidget(self.run_batch_btn)

        self.stop_batch_btn = QPushButton("⏹️ Stop Batch")
        self.stop_batch_btn.clicked.connect(self._stop_batch_analysis)
        self.stop_batch_btn.setEnabled(False)
        exec_layout.addWidget(self.stop_batch_btn)

        # Batch progress
        self.batch_progress = QProgressBar()
        self.batch_progress.setVisible(False)
        exec_layout.addWidget(self.batch_progress)

        self.batch_status_label = QLabel("No batch analysis running")
        self.batch_status_label.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
        exec_layout.addWidget(self.batch_status_label)

        layout.addWidget(exec_group)

        # Quick batch tools
        tools_group = QGroupBox("Quick Tools")
        tools_layout = QVBoxLayout(tools_group)

        self.autocorr_analysis_btn = QPushButton("📈 Autocorrelation Analysis...")
        self.autocorr_analysis_btn.clicked.connect(self._run_autocorrelation_analysis)
        tools_layout.addWidget(self.autocorr_analysis_btn)

        self.export_all_btn = QPushButton("📁 Export All Results...")
        self.export_all_btn.clicked.connect(self._export_all_results)
        tools_layout.addWidget(self.export_all_btn)

        layout.addWidget(tools_group)
        layout.addStretch()

        return tab

    def _create_enhanced_center_panel(self) -> QWidget:
        """Create the enhanced center visualization panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Enhanced visualization widget
        self.visualization = EnhancedVisualizationWidget(self.data_manager)
        layout.addWidget(self.visualization)

        # Enhanced visualization controls
        controls_layout = QHBoxLayout()

        # Frame controls
        frame_group = QGroupBox("Frame Control")
        frame_layout = QHBoxLayout(frame_group)

        # Frame slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)
        self.frame_slider.valueChanged.connect(lambda val: self.visualization.set_frame(val) if hasattr(self.visualization, 'set_frame') else None)

        self.frame_label = QLabel("Frame: 0/0")

        frame_layout.addWidget(QLabel("Frame:"))
        frame_layout.addWidget(self.frame_slider)
        frame_layout.addWidget(self.frame_label)

        # Playback controls
        self.play_button = QPushButton("▶️ Play")
        self.play_button.clicked.connect(self._toggle_playback)
        frame_layout.addWidget(self.play_button)

        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 100)
        self.speed_spin.setValue(10)
        self.speed_spin.setSuffix(" FPS")
        frame_layout.addWidget(self.speed_spin)

        controls_layout.addWidget(frame_group)

        # View controls
        view_group = QGroupBox("View Control")
        view_layout = QHBoxLayout(view_group)

        self.zoom_fit_btn = QPushButton("🔍 Fit")
        self.zoom_fit_btn.clicked.connect(lambda: self.visualization.zoom_fit() if hasattr(self.visualization, 'zoom_fit') else None)
        view_layout.addWidget(self.zoom_fit_btn)

        self.reset_view_btn = QPushButton("🏠 Reset")
        self.reset_view_btn.clicked.connect(lambda: self.visualization.reset_view() if hasattr(self.visualization, 'reset_view') else None)
        view_layout.addWidget(self.reset_view_btn)

        self.export_view_btn = QPushButton("📷 Export")
        self.export_view_btn.clicked.connect(self._export_current_view)
        view_layout.addWidget(self.export_view_btn)

        controls_layout.addWidget(view_group)

        layout.addLayout(controls_layout)

        return panel

    def _create_enhanced_right_panel(self) -> QWidget:
        """Create the enhanced right results panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Enhanced results tabs
        self.results_tabs = QTabWidget()

        # Statistics tab (enhanced)
        stats_tab = self._create_enhanced_stats_tab()
        self.results_tabs.addTab(stats_tab, "📊 Statistics")

        # Data table tab (enhanced)
        table_tab = self._create_enhanced_table_tab()
        self.results_tabs.addTab(table_tab, "📋 Data Table")

        # Feature analysis tab (new)
        feature_tab = self._create_feature_analysis_tab()
        self.results_tabs.addTab(feature_tab, "🔬 Features")

        # Export tab (enhanced)
        export_tab = self._create_enhanced_export_tab()
        self.results_tabs.addTab(export_tab, "📁 Export")

        layout.addWidget(self.results_tabs)

        return panel

    def _create_enhanced_stats_tab(self) -> QWidget:
        """Create enhanced statistics tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Statistics display
        self.stats_widget = QTextEdit()
        self.stats_widget.setReadOnly(True)
        self.stats_widget.setFont(QFont("Courier", 10))
        layout.addWidget(self.stats_widget)

        # Quick stats controls
        stats_controls = QHBoxLayout()

        self.refresh_stats_btn = QPushButton("🔄 Refresh")
        self.refresh_stats_btn.clicked.connect(self._refresh_statistics)
        stats_controls.addWidget(self.refresh_stats_btn)

        self.detailed_stats_btn = QPushButton("📊 Detailed Analysis")
        self.detailed_stats_btn.clicked.connect(self._show_detailed_statistics)
        stats_controls.addWidget(self.detailed_stats_btn)

        stats_controls.addStretch()
        layout.addLayout(stats_controls)

        return tab

    def _create_enhanced_table_tab(self) -> QWidget:
        """Create enhanced data table tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Table controls
        table_controls = QHBoxLayout()

        self.table_data_combo = QComboBox()
        table_controls.addWidget(QLabel("Data:"))
        table_controls.addWidget(self.table_data_combo)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter data...")
        self.filter_edit.textChanged.connect(self._filter_table)
        table_controls.addWidget(self.filter_edit)

        table_controls.addStretch()
        layout.addLayout(table_controls)

        # Enhanced results table
        self.results_table = QTableView()
        self.results_table.setSortingEnabled(True)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

        # Set up table headers
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)

        layout.addWidget(self.results_table)

        return tab

    def _create_feature_analysis_tab(self) -> QWidget:
        """Create feature analysis tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Feature summary
        self.feature_summary = QTextEdit()
        self.feature_summary.setReadOnly(True)
        self.feature_summary.setMaximumHeight(150)
        layout.addWidget(self.feature_summary)

        # Feature controls
        feature_controls = QHBoxLayout()

        self.analyze_features_btn = QPushButton("🔬 Analyze Features")
        self.analyze_features_btn.clicked.connect(self._analyze_current_features)
        feature_controls.addWidget(self.analyze_features_btn)

        self.compare_conditions_btn = QPushButton("⚖️ Compare Conditions")
        self.compare_conditions_btn.clicked.connect(self._compare_conditions)
        feature_controls.addWidget(self.compare_conditions_btn)

        feature_controls.addStretch()
        layout.addLayout(feature_controls)

        layout.addStretch()
        return tab

    def _create_enhanced_export_tab(self) -> QWidget:
        """Create enhanced export tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Export options
        export_options = QGroupBox("Export Options")
        options_layout = QVBoxLayout(export_options)

        self.export_csv_cb = QCheckBox("CSV Data Files")
        self.export_csv_cb.setChecked(True)
        options_layout.addWidget(self.export_csv_cb)

        self.export_excel_cb = QCheckBox("Excel Workbook")
        options_layout.addWidget(self.export_excel_cb)

        self.export_plots_cb = QCheckBox("Analysis Plots")
        options_layout.addWidget(self.export_plots_cb)

        self.export_report_cb = QCheckBox("Comprehensive Report")
        self.export_report_cb.setChecked(True)
        options_layout.addWidget(self.export_report_cb)

        layout.addWidget(export_options)

        # Export buttons
        export_buttons = QVBoxLayout()

        self.export_current_btn = QPushButton("📁 Export Current Results")
        self.export_current_btn.clicked.connect(self._export_current_results)
        export_buttons.addWidget(self.export_current_btn)

        self.export_all_data_btn = QPushButton("📁 Export All Data")
        self.export_all_data_btn.clicked.connect(self._export_all_data)
        export_buttons.addWidget(self.export_all_data_btn)

        self.export_comparison_btn = QPushButton("📊 Export Comparison Report")
        self.export_comparison_btn.clicked.connect(self._export_comparison_report)
        export_buttons.addWidget(self.export_comparison_btn)

        layout.addLayout(export_buttons)
        layout.addStretch()

        return tab

    def _create_enhanced_menu_bar(self):
        """Create the enhanced menu bar."""
        menubar = self.menuBar()

        # File menu (enhanced)
        file_menu = menubar.addMenu("📁 File")

        # Open actions
        open_image_action = QAction("🖼️ Open Image...", self)
        open_image_action.triggered.connect(self._open_image_file)
        file_menu.addAction(open_image_action)

        open_data_action = QAction("📊 Open Data...", self)
        open_data_action.triggered.connect(self._open_data_file)
        file_menu.addAction(open_data_action)

        file_menu.addSeparator()

        # Project actions
        new_project_action = QAction("🆕 New Project", self)
        new_project_action.triggered.connect(self._new_project)
        file_menu.addAction(new_project_action)

        open_project_action = QAction("📂 Open Project...", self)
        open_project_action.triggered.connect(self._open_project)
        file_menu.addAction(open_project_action)

        save_project_action = QAction("💾 Save Project", self)
        save_project_action.triggered.connect(self._save_project)
        file_menu.addAction(save_project_action)

        file_menu.addSeparator()

        # Exit action
        exit_action = QAction("🚪 Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Analysis menu (enhanced)
        analysis_menu = menubar.addMenu("🔬 Analysis")

        # Enhanced analysis actions
        comprehensive_action = QAction("🚀 Comprehensive Analysis", self)
        comprehensive_action.triggered.connect(self._run_comprehensive_analysis)
        analysis_menu.addAction(comprehensive_action)

        analysis_menu.addSeparator()

        detect_action = QAction("🔍 Detect Particles", self)
        detect_action.triggered.connect(self._run_detection)
        analysis_menu.addAction(detect_action)

        link_action = QAction("🔗 Link Trajectories", self)
        link_action.triggered.connect(self._run_linking)
        analysis_menu.addAction(link_action)

        enhanced_features_action = QAction("🔬 Enhanced Features", self)
        enhanced_features_action.triggered.connect(self._run_enhanced_features)
        analysis_menu.addAction(enhanced_features_action)

        classify_action = QAction("🏷️ Classify Trajectories", self)
        classify_action.triggered.connect(self._run_classification)
        analysis_menu.addAction(classify_action)

        analysis_menu.addSeparator()

        autocorr_action = QAction("📈 Autocorrelation Analysis...", self)
        autocorr_action.triggered.connect(self._run_autocorrelation_analysis)
        analysis_menu.addAction(autocorr_action)

        # Batch menu (new)
        batch_menu = menubar.addMenu("📦 Batch")

        new_experiment_action = QAction("🆕 New Experiment...", self)
        new_experiment_action.triggered.connect(self._create_new_experiment)
        batch_menu.addAction(new_experiment_action)

        load_experiment_action = QAction("📂 Load Experiment...", self)
        load_experiment_action.triggered.connect(self._load_experiment)
        batch_menu.addAction(load_experiment_action)

        batch_menu.addSeparator()

        run_batch_action = QAction("🚀 Run Batch Analysis", self)
        run_batch_action.triggered.connect(self._run_batch_analysis)
        batch_menu.addAction(run_batch_action)

        # Tools menu (enhanced)
        tools_menu = menubar.addMenu("🛠️ Tools")

        refresh_training_data_action = QAction("🔄 Refresh Training Data", self)
        refresh_training_data_action.triggered.connect(self._refresh_training_data)
        tools_menu.addAction(refresh_training_data_action)

        tools_menu.addSeparator()

        preferences_action = QAction("⚙️ Preferences...", self)
        preferences_action.triggered.connect(self._show_preferences)
        tools_menu.addAction(preferences_action)

        # Help menu
        help_menu = menubar.addMenu("❓ Help")

        user_guide_action = QAction("📖 User Guide", self)
        user_guide_action.triggered.connect(self._show_user_guide)
        help_menu.addAction(user_guide_action)

        about_action = QAction("ℹ️ About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _create_enhanced_tool_bar(self):
        """Create the enhanced tool bar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setObjectName("MainToolBar")
        self.addToolBar(toolbar)

        # File operations
        toolbar.addAction("📂", self._open_data_file).setToolTip("Open Data File")
        toolbar.addAction("💾", self._save_project).setToolTip("Save Project")
        toolbar.addSeparator()

        # Quick analysis
        toolbar.addAction("🚀", self._run_comprehensive_analysis).setToolTip("Comprehensive Analysis")
        toolbar.addAction("🔍", self._run_detection).setToolTip("Detect Particles")
        toolbar.addAction("🔗", self._run_linking).setToolTip("Link Trajectories")
        toolbar.addAction("🔬", self._run_enhanced_features).setToolTip("Enhanced Features")
        toolbar.addAction("🏷️", self._run_classification).setToolTip("Classify Trajectories")
        toolbar.addSeparator()

        # View operations
        toolbar.addAction("🔍", lambda: self.visualization.zoom_fit() if hasattr(self.visualization, 'zoom_fit') else None).setToolTip("Zoom Fit")
        toolbar.addAction("🏠", lambda: self.visualization.reset_view() if hasattr(self.visualization, 'reset_view') else None).setToolTip("Reset View")
        toolbar.addAction("📷", self._export_current_view).setToolTip("Export View")

    def _create_enhanced_status_bar(self):
        """Create the enhanced status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Enhanced progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Memory usage indicator
        self.memory_label = QLabel("Memory: 0 MB")
        self.memory_label.setStyleSheet("QLabel { color: gray; font-size: 9px; }")
        self.status_bar.addPermanentWidget(self.memory_label)

        # Status label
        self.status_label = QLabel("Ready for enhanced analysis")
        self.status_bar.addWidget(self.status_label)

    def _create_enhanced_dock_widgets(self):
        """Create enhanced dockable widgets."""
        # Enhanced logging dock
        log_dock = QDockWidget("📝 Analysis Log", self)
        log_dock.setObjectName("LogDock")
        self.log_widget = LoggingWidget()
        log_dock.setWidget(self.log_widget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, log_dock)

    def _connect_signals(self):
        """Connect enhanced signals between components."""
        # Data manager signals
        if hasattr(self.data_manager, 'dataLoaded'):
            self.data_manager.dataLoaded.connect(self._on_data_loaded)
        if hasattr(self.data_manager, 'progressUpdate'):
            self.data_manager.progressUpdate.connect(self._update_progress)

        # Analysis engine signals
        if hasattr(self.analysis_engine, 'analysisStarted'):
            self.analysis_engine.analysisStarted.connect(self._on_analysis_started)
        if hasattr(self.analysis_engine, 'stepCompleted'):
            self.analysis_engine.stepCompleted.connect(self._on_analysis_step_completed)
        if hasattr(self.analysis_engine, 'analysisCompleted'):
            self.analysis_engine.analysisCompleted.connect(self._on_analysis_completed)
        if hasattr(self.analysis_engine, 'progressUpdate'):
            self.analysis_engine.progressUpdate.connect(self._update_progress)
        if hasattr(self.analysis_engine, 'errorOccurred'):
            self.analysis_engine.errorOccurred.connect(self._on_analysis_error)

    def _setup_default_parameters(self):
        """Setup enhanced default parameters."""
        try:
            # Setup any default parameters
            pass
        except Exception as e:
            self.logger.warning(f"Error setting up enhanced default parameters: {e}")

    def _restore_settings(self):
        """Restore enhanced window settings."""
        self.restoreGeometry(self.settings.value("geometry", b""))
        self.restoreState(self.settings.value("windowState", b""))

    def _save_settings(self):
        """Save enhanced window settings."""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())

    def closeEvent(self, event):
        """Handle enhanced window close event."""
        # Stop any running analysis
        if hasattr(self.analysis_engine, 'stop_analysis'):
            self.analysis_engine.stop_analysis()
        if hasattr(self.batch_manager, 'stop_current_analysis'):
            self.batch_manager.stop_current_analysis()

        self._save_settings()
        event.accept()

    # Enhanced file operations
    def _open_image_file(self):
        """Open an image file with enhanced support."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "",
            "Image Files (*.tif *.tiff *.png *.jpg *.jpeg *.nd2 *.lsm);;All Files (*)"
        )
        if file_path and hasattr(self.data_manager, 'load_file'):
            self.data_manager.load_file(file_path)

    def _open_data_file(self):
        """Open a data file with enhanced support."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "",
            "Data Files (*.csv *.txt *.json *.xlsx *.h5 *.hdf5);;All Files (*)"
        )
        if file_path and hasattr(self.data_manager, 'load_file'):
            self.data_manager.load_file(file_path)

    def _run_autocorrelation_analysis(self):
        """Run autocorrelation analysis with dialog."""
        dialog = AutocorrelationAnalysisDialog(self.data_manager, self)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            config = dialog.get_analysis_config()

            if not config['data_name']:
                QMessageBox.warning(self, "Warning", "No trajectory data selected")
                return

            try:
                # Get trajectory data
                if hasattr(self.data_manager, 'get_data'):
                    trajectory_data = self.data_manager.get_data(config['data_name'])

                    # Filter by minimum track length if needed
                    if config['min_track_length'] > 3:
                        track_lengths = trajectory_data.groupby('track_number').size()
                        valid_tracks = track_lengths[track_lengths >= config['min_track_length']].index
                        trajectory_data = trajectory_data[trajectory_data['track_number'].isin(valid_tracks)]

                    # Run autocorrelation analysis
                    analyzer = DirectionAutocorrelationAnalyzer({'max_lag': config['max_lag']})

                    output_dir = config.get('output_directory')
                    if output_dir and not Path(output_dir).exists():
                        Path(output_dir).mkdir(parents=True, exist_ok=True)

                    results = analyzer.analyze_trajectory_data(trajectory_data, output_dir)

                    self._update_status("Autocorrelation analysis completed")
                    QMessageBox.information(self, "Success",
                                          f"Autocorrelation analysis completed.\n"
                                          f"Results saved to: {output_dir}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Autocorrelation analysis failed: {e}")

    def _create_new_experiment(self):
        """Create a new batch experiment."""
        dialog = BatchExperimentDialog(self.batch_manager, self)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            config = dialog.get_experiment_config()

            if not config['name']:
                QMessageBox.warning(self, "Warning", "Please enter an experiment name")
                return

            try:
                # Create experiment
                experiment = self.batch_manager.create_experiment(
                    config['name'],
                    config['description'],
                    config['output_directory']
                )

                # Add files
                if config['files']:
                    self.batch_manager.add_files_to_experiment(config['name'], config['files'])

                # Update UI
                self._update_experiments_list()
                self.experiments_list.setCurrentText(config['name'])

                self._update_status(f"Created batch experiment: {config['name']}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to create experiment: {e}")

    def _run_batch_analysis(self):
        """Run batch analysis for selected experiment."""
        experiment_name = self.experiments_list.currentText()
        if not experiment_name:
            QMessageBox.warning(self, "Warning", "No experiment selected")
            return

        try:
            self.batch_manager.run_experiment(experiment_name)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start batch analysis: {e}")

    def _stop_batch_analysis(self):
        """Stop current batch analysis."""
        self.batch_manager.stop_current_analysis()

    def _update_experiments_list(self):
        """Update the experiments list."""
        current_text = self.experiments_list.currentText()
        self.experiments_list.clear()
        self.experiments_list.addItems(self.batch_manager.get_experiment_list())

        # Restore selection
        index = self.experiments_list.findText(current_text)
        if index >= 0:
            self.experiments_list.setCurrentIndex(index)

    def _on_data_loaded(self, data_name: str, data: Any):
        """Handle enhanced data loading."""
        self._update_status(f"Loaded: {data_name}")

        # Update table data combo
        if hasattr(self.data_manager, 'get_data_names'):
            self.table_data_combo.clear()
            self.table_data_combo.addItems(self.data_manager.get_data_names())
            self.table_data_combo.setCurrentText(data_name)

        # Update visualization
        if isinstance(data, np.ndarray) and len(data.shape) >= 2:
            if hasattr(self.visualization, 'set_image_data'):
                self.visualization.set_image_data(data)

            if len(data.shape) == 3:
                self.frame_slider.setMaximum(data.shape[0] - 1)
                self.frame_slider.setValue(0)
                self._update_frame_display(0)

        elif isinstance(data, pd.DataFrame):
            self._update_results_display(data)

            if 'x' in data.columns and 'y' in data.columns:
                if hasattr(self.visualization, 'set_tracking_data'):
                    self.visualization.set_tracking_data(data)

    def _update_results_display(self, data: pd.DataFrame):
        """Update enhanced results display."""
        try:
            # Update statistics
            stats_text = f"Data: {len(data)} rows, {len(data.columns)} columns\n\n"

            if 'track_number' in data.columns:
                n_tracks = data['track_number'].nunique()
                stats_text += f"Number of tracks: {n_tracks}\n"

                track_lengths = data.groupby('track_number').size()
                stats_text += f"Mean track length: {track_lengths.mean():.1f}\n"
                stats_text += f"Median track length: {track_lengths.median():.1f}\n"

            if 'frame' in data.columns:
                n_frames = data['frame'].nunique()
                stats_text += f"Number of frames: {n_frames}\n"

            self.stats_widget.setText(stats_text)

            # Update table - create a simple table model
            # For now, just show basic info
            self.feature_summary.setText(f"Dataset: {len(data)} rows, {len(data.columns)} columns")

        except Exception as e:
            self.logger.error(f"Error updating results display: {e}")

    def _update_status(self, message: str):
        """Update enhanced status bar message."""
        self.status_label.setText(message)
        self.logger.info(message)

    def _update_progress(self, message: str, percentage: int):
        """Update progress display."""
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)
        self.progress_bar.setVisible(percentage < 100)

    def _update_frame_display(self, frame: int):
        max_frame = self.frame_slider.maximum()
        self.frame_label.setText(f"Frame: {frame}/{max_frame}")

    # Placeholder methods for remaining functionality
    def _new_project(self): pass
    def _open_project(self): pass
    def _save_project(self): pass
    def _refresh_training_data(self): pass
    def _show_preferences(self): pass
    def _show_user_guide(self): pass
    def _show_about(self): pass
    def _toggle_playback(self): pass
    def _export_current_view(self): pass
    def _filter_table(self): pass
    def _analyze_current_features(self): pass
    def _compare_conditions(self): pass
    def _export_current_results(self): pass
    def _export_all_data(self): pass
    def _export_comparison_report(self): pass
    def _export_all_results(self): pass
    def _load_experiment(self): pass
    def _save_experiment(self): pass
    def _refresh_statistics(self): pass
    def _show_detailed_statistics(self): pass
    def _run_comprehensive_analysis(self): pass
    def _run_detection(self): pass
    def _run_linking(self): pass
    def _run_enhanced_features(self): pass
    def _run_classification(self): pass
    def _on_analysis_started(self, steps): pass
    def _on_analysis_step_completed(self, step, result): pass
    def _on_analysis_completed(self, result): pass
    def _on_analysis_error(self, error): pass
