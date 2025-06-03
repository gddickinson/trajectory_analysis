#!/usr/bin/env python3
"""
Main Window GUI Module
======================

Main application window providing the user interface for particle tracking analysis.
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
    QHeaderView
)
from PyQt6.QtCore import (
    Qt, QTimer, QSettings, pyqtSignal, QAbstractTableModel,
    QModelIndex, QVariant, QSortFilterProxyModel
)
from PyQt6.QtGui import QAction, QFont, QIcon, QStandardItemModel, QStandardItem

import pandas as pd
import numpy as np

from particle_tracker.gui.visualization_widget import VisualizationWidget
from particle_tracker.gui.parameter_panels import ParameterPanelManager
from particle_tracker.gui.data_browser import DataBrowserWidget
from particle_tracker.gui.analysis_control import AnalysisControlWidget
from particle_tracker.gui.logging_widget import LoggingWidget
from particle_tracker.core.analysis_engine import AnalysisParameters, AnalysisStep


class PandasTableModel(QAbstractTableModel):
    """Table model for displaying pandas DataFrames."""

    def __init__(self, data: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._data = data.copy()

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._data)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self._data.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return QVariant()

        if role == Qt.ItemDataRole.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            if pd.isna(value):
                return "NaN"
            elif isinstance(value, float):
                return f"{value:.4f}"
            else:
                return str(value)

        return QVariant()

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])
            else:
                return str(section)
        return QVariant()

    def sort(self, column: int, order: Qt.SortOrder = Qt.SortOrder.AscendingOrder):
        """Sort the data by the given column."""
        self.beginResetModel()
        column_name = self._data.columns[column]
        ascending = order == Qt.SortOrder.AscendingOrder
        self._data = self._data.sort_values(by=column_name, ascending=ascending)
        self.endResetModel()


class MainWindow(QMainWindow):
    """Main application window."""

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

        # Initialize UI
        self._setup_ui()
        self._connect_signals()
        self._setup_default_parameters()  # Add this line
        self._restore_settings()

        self.logger.info("Main window initialized")

    def _setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("Particle Tracking Analyzer")
        self.setMinimumSize(1200, 800)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QHBoxLayout(central_widget)

        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)

        # Create left panel (controls and data browser)
        left_panel = self._create_left_panel()
        main_splitter.addWidget(left_panel)

        # Create center panel (visualization)
        center_panel = self._create_center_panel()
        main_splitter.addWidget(center_panel)

        # Create right panel (analysis results)
        right_panel = self._create_right_panel()
        main_splitter.addWidget(right_panel)

        # Set splitter proportions
        main_splitter.setSizes([300, 600, 300])

        # Create menu bar
        self._create_menu_bar()

        # Create tool bar
        self._create_tool_bar()

        # Create status bar
        self._create_status_bar()

        # Create dock widgets
        self._create_dock_widgets()


    def _create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        # Open actions
        open_image_action = QAction("Open Image...", self)
        open_image_action.triggered.connect(self._open_image_file)
        file_menu.addAction(open_image_action)

        open_data_action = QAction("Open Data...", self)
        open_data_action.triggered.connect(self._open_data_file)
        file_menu.addAction(open_data_action)

        file_menu.addSeparator()

        # Project actions
        new_project_action = QAction("New Project", self)
        new_project_action.triggered.connect(self._new_project)
        file_menu.addAction(new_project_action)

        open_project_action = QAction("Open Project...", self)
        open_project_action.triggered.connect(self._open_project)
        file_menu.addAction(open_project_action)

        save_project_action = QAction("Save Project", self)
        save_project_action.triggered.connect(self._save_project)
        file_menu.addAction(save_project_action)

        file_menu.addSeparator()

        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Analysis menu
        analysis_menu = menubar.addMenu("Analysis")

        detect_action = QAction("Detect Particles", self)
        detect_action.triggered.connect(self._run_detection)
        analysis_menu.addAction(detect_action)

        link_action = QAction("Link Trajectories", self)
        link_action.triggered.connect(self._run_linking)
        analysis_menu.addAction(link_action)

        features_action = QAction("Calculate Features", self)
        features_action.triggered.connect(self._run_features)
        analysis_menu.addAction(features_action)

        classify_action = QAction("Classify Trajectories", self)
        classify_action.triggered.connect(self._run_classification)
        analysis_menu.addAction(classify_action)

        analysis_menu.addSeparator()

        full_pipeline_action = QAction("Run Full Pipeline", self)
        full_pipeline_action.triggered.connect(self._run_full_pipeline)
        analysis_menu.addAction(full_pipeline_action)

        # Tools menu (NEW)
        tools_menu = menubar.addMenu("Tools")

        refresh_training_data_action = QAction("Refresh Training Data Path", self)
        refresh_training_data_action.triggered.connect(self._refresh_training_data)
        tools_menu.addAction(refresh_training_data_action)

        # View menu
        view_menu = menubar.addMenu("View")

        # Dock widget toggles will be added here

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _refresh_training_data(self):
        """Refresh the training data path."""
        try:
            new_path = self.config.refresh_training_data_path()
            if new_path:
                # Update parameter manager
                current_params = self.parameter_manager.get_all_parameters()
                if isinstance(current_params, dict):
                    current_params['svm_training_data'] = new_path
                else:
                    from dataclasses import asdict
                    current_params = asdict(current_params)
                    current_params['svm_training_data'] = new_path

                self.parameter_manager.set_all_parameters(current_params)

                QMessageBox.information(
                    self, "Training Data Updated",
                    f"SVM training data path updated to:\n{new_path}"
                )
            else:
                QMessageBox.warning(
                    self, "Training Data Not Found",
                    "Could not find default SVM training data.\n"
                    "Please manually browse for the training data file."
                )
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Error refreshing training data path:\n{e}"
            )


    def _create_left_panel(self) -> QWidget:
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Data browser
        self.data_browser = DataBrowserWidget(self.data_manager)
        data_group = QGroupBox("Data Browser")
        data_layout = QVBoxLayout(data_group)
        data_layout.addWidget(self.data_browser)
        layout.addWidget(data_group)

        # Parameter panels - CREATE FIRST
        self.parameter_manager = ParameterPanelManager()
        param_group = QGroupBox("Parameters")
        param_layout = QVBoxLayout(param_group)
        param_layout.addWidget(self.parameter_manager)
        layout.addWidget(param_group)

        # Analysis control - CREATE AFTER parameter manager
        self.analysis_control = AnalysisControlWidget(
            self.analysis_engine, self.data_manager, self.parameter_manager  # Pass parameter manager
        )
        analysis_group = QGroupBox("Analysis Control")
        analysis_layout = QVBoxLayout(analysis_group)
        analysis_layout.addWidget(self.analysis_control)
        layout.addWidget(analysis_group)

        return panel

    def _create_center_panel(self) -> QWidget:
        """Create the center visualization panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Visualization widget
        self.visualization = VisualizationWidget(self.data_manager)
        layout.addWidget(self.visualization)

        # Visualization controls
        controls_layout = QHBoxLayout()

        # Frame slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)
        self.frame_slider.valueChanged.connect(self.visualization.set_frame)

        self.frame_label = QLabel("Frame: 0/0")

        controls_layout.addWidget(QLabel("Frame:"))
        controls_layout.addWidget(self.frame_slider)
        controls_layout.addWidget(self.frame_label)

        # Playback controls
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._toggle_playback)
        controls_layout.addWidget(self.play_button)

        layout.addLayout(controls_layout)

        return panel

    def _create_right_panel(self) -> QWidget:
        """Create the right results panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Results tabs
        self.results_tabs = QTabWidget()

        # Statistics tab
        self.stats_widget = QTextEdit()
        self.stats_widget.setReadOnly(True)
        self.stats_widget.setFont(QFont("Courier", 10))
        self.results_tabs.addTab(self.stats_widget, "Statistics")

        # Data table tab
        self.results_table = QTableView()
        self.results_table.setSortingEnabled(True)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

        # Set up table headers
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)

        self.results_tabs.addTab(self.results_table, "Data Table")

        # Export tab
        export_widget = QWidget()
        export_layout = QVBoxLayout(export_widget)

        self.export_csv_btn = QPushButton("Export CSV")
        self.export_csv_btn.clicked.connect(self._export_csv)

        self.export_report_btn = QPushButton("Export Report")
        self.export_report_btn.clicked.connect(self._export_report)

        export_layout.addWidget(self.export_csv_btn)
        export_layout.addWidget(self.export_report_btn)
        export_layout.addStretch()

        self.results_tabs.addTab(export_widget, "Export")

        layout.addWidget(self.results_tabs)

        return panel

    def _create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        # Open actions
        open_image_action = QAction("Open Image...", self)
        open_image_action.triggered.connect(self._open_image_file)
        file_menu.addAction(open_image_action)

        open_data_action = QAction("Open Data...", self)
        open_data_action.triggered.connect(self._open_data_file)
        file_menu.addAction(open_data_action)

        file_menu.addSeparator()

        # Project actions
        new_project_action = QAction("New Project", self)
        new_project_action.triggered.connect(self._new_project)
        file_menu.addAction(new_project_action)

        open_project_action = QAction("Open Project...", self)
        open_project_action.triggered.connect(self._open_project)
        file_menu.addAction(open_project_action)

        save_project_action = QAction("Save Project", self)
        save_project_action.triggered.connect(self._save_project)
        file_menu.addAction(save_project_action)

        file_menu.addSeparator()

        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Analysis menu
        analysis_menu = menubar.addMenu("Analysis")

        detect_action = QAction("Detect Particles", self)
        detect_action.triggered.connect(self._run_detection)
        analysis_menu.addAction(detect_action)

        link_action = QAction("Link Trajectories", self)
        link_action.triggered.connect(self._run_linking)
        analysis_menu.addAction(link_action)

        features_action = QAction("Calculate Features", self)
        features_action.triggered.connect(self._run_features)
        analysis_menu.addAction(features_action)

        classify_action = QAction("Classify Trajectories", self)
        classify_action.triggered.connect(self._run_classification)
        analysis_menu.addAction(classify_action)

        analysis_menu.addSeparator()

        full_pipeline_action = QAction("Run Full Pipeline", self)
        full_pipeline_action.triggered.connect(self._run_full_pipeline)
        analysis_menu.addAction(full_pipeline_action)

        # View menu
        view_menu = menubar.addMenu("View")

        # Dock widget toggles will be added here

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _create_tool_bar(self):
        """Create the tool bar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setObjectName("MainToolBar")  # Set object name for saveState
        self.addToolBar(toolbar)

        # File operations
        toolbar.addAction("Open", self._open_data_file)
        toolbar.addAction("Save", self._save_project)
        toolbar.addSeparator()

        # Analysis operations
        toolbar.addAction("Detect", self._run_detection)
        toolbar.addAction("Link", self._run_linking)
        toolbar.addAction("Features", self._run_features)
        toolbar.addAction("Classify", self._run_classification)
        toolbar.addSeparator()

        # View operations
        toolbar.addAction("Zoom Fit", self.visualization.zoom_fit)
        toolbar.addAction("Reset View", self.visualization.reset_view)

    def _create_status_bar(self):
        """Create the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)

    def _create_dock_widgets(self):
        """Create dockable widgets."""
        # Logging dock
        log_dock = QDockWidget("Log", self)
        log_dock.setObjectName("LogDock")  # Set object name for saveState
        self.log_widget = LoggingWidget()
        log_dock.setWidget(self.log_widget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, log_dock)

        # Parameter dock (initially hidden)
        param_dock = QDockWidget("Advanced Parameters", self)
        param_dock.setObjectName("ParameterDock")  # Set object name for saveState
        advanced_params = QWidget()  # Create advanced parameter widget
        param_dock.setWidget(advanced_params)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, param_dock)
        param_dock.setVisible(False)

    def _connect_signals(self):
        """Connect signals between components."""
        # Data manager signals
        self.data_manager.dataLoaded.connect(self._on_data_loaded)
        self.data_manager.progressUpdate.connect(self._update_progress)

        # Analysis engine signals
        self.analysis_engine.analysisStarted.connect(self._on_analysis_started)
        self.analysis_engine.stepCompleted.connect(self._on_analysis_step_completed)
        self.analysis_engine.analysisCompleted.connect(self._on_analysis_completed)
        self.analysis_engine.progressUpdate.connect(self._update_progress)
        self.analysis_engine.errorOccurred.connect(self._on_analysis_error)

        # Visualization signals
        self.visualization.frameChanged.connect(self._update_frame_display)

    def _restore_settings(self):
        """Restore window settings."""
        self.restoreGeometry(self.settings.value("geometry", b""))
        self.restoreState(self.settings.value("windowState", b""))

    def _save_settings(self):
        """Save window settings."""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())

    def closeEvent(self, event):
        """Handle window close event."""
        self._save_settings()

        # Stop any running analysis
        self.analysis_engine.stop_analysis()

        event.accept()

    def _setup_default_parameters(self):
        """Setup default parameters including training data path."""
        try:
            # Get default training data path from config
            default_training_data = self.config.get_default_svm_training_data()

            if not default_training_data:
                # Try to refresh/auto-detect
                default_training_data = self.config.refresh_training_data_path()

            if default_training_data:
                # Update parameter manager with default training data
                current_params = self.parameter_manager.get_all_parameters()
                if isinstance(current_params, dict):
                    current_params['svm_training_data'] = default_training_data
                else:
                    # If it's an AnalysisParameters object, convert to dict and update
                    from dataclasses import asdict
                    current_params = asdict(current_params)
                    current_params['svm_training_data'] = default_training_data

                self.parameter_manager.set_all_parameters(current_params)
                self.logger.info(f"Set default SVM training data: {default_training_data}")

        except Exception as e:
            self.logger.warning(f"Error setting up default training data: {e}")

    # File operations
    def _open_image_file(self):
        """Open an image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "",
            "Image Files (*.tif *.tiff *.png *.jpg *.jpeg);;All Files (*)"
        )
        if file_path:
            self.data_manager.load_file(file_path)

    def _open_data_file(self):
        """Open a data file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "",
            "Data Files (*.csv *.txt *.json *.xlsx);;All Files (*)"
        )
        if file_path:
            self.data_manager.load_file(file_path)

    def _new_project(self):
        """Create a new project."""
        # Clear current data
        self.data_manager.clear_all_data()
        self.project_manager.new_project()
        self._update_status("New project created")

    def _open_project(self):
        """Open an existing project."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "",
            "Project Files (*.ptproj);;All Files (*)"
        )
        if file_path:
            if self.project_manager.load_project(file_path):
                self._update_status(f"Opened project: {Path(file_path).name}")
            else:
                QMessageBox.warning(self, "Error", "Failed to open project")

    def _save_project(self):
        """Save the current project."""
        if self.project_manager.current_project_path:
            self.project_manager.save_project()
            self._update_status("Project saved")
        else:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Project", "",
                "Project Files (*.ptproj);;All Files (*)"
            )
            if file_path:
                self.project_manager.save_project(file_path)
                self._update_status(f"Project saved: {Path(file_path).name}")

    # Analysis operations
    def _run_detection(self):
        """Run particle detection."""
        # Get current image data
        image_names = self.data_manager.get_data_names()
        if not image_names:
            QMessageBox.warning(self, "Warning", "No image data loaded")
            return

        # Get parameters
        params = self.parameter_manager.get_all_parameters()

        # Run detection
        image_data = self.data_manager.get_data(image_names[0])
        self.analysis_engine.run_analysis_pipeline(
            image_data, params, [AnalysisStep.DETECTION]
        )

    def _run_linking(self):
        """Run particle linking."""
        # Get current localization data
        localization_names = [name for name in self.data_manager.get_data_names()
                             if 'detection' in name or 'localization' in name]
        if not localization_names:
            QMessageBox.warning(self, "Warning", "No localization data found. Run detection first.")
            return

        # Get parameters
        params = self.parameter_manager.get_all_parameters()

        # Run linking
        localization_data = self.data_manager.get_data(localization_names[0])
        self.analysis_engine.run_analysis_pipeline(
            localization_data, params, [AnalysisStep.LINKING]
        )

    def _run_features(self):
        """Run feature calculation."""
        # Get current trajectory data
        trajectory_names = [name for name in self.data_manager.get_data_names()
                           if 'linking' in name or 'trajectory' in name or 'track' in name]
        if not trajectory_names:
            QMessageBox.warning(self, "Warning", "No trajectory data found. Run linking first.")
            return

        # Get parameters
        params = self.parameter_manager.get_all_parameters()

        # Run features
        trajectory_data = self.data_manager.get_data(trajectory_names[0])
        self.analysis_engine.run_analysis_pipeline(
            trajectory_data, params, [AnalysisStep.FEATURES]
        )

    def _run_classification(self):
        """Run trajectory classification."""
        # Get current feature data
        feature_names = [name for name in self.data_manager.get_data_names()
                        if 'feature' in name]
        if not feature_names:
            QMessageBox.warning(self, "Warning", "No feature data found. Run feature calculation first.")
            return

        # Get parameters
        params = self.parameter_manager.get_all_parameters()

        # Run classification
        feature_data = self.data_manager.get_data(feature_names[0])
        self.analysis_engine.run_analysis_pipeline(
            feature_data, params, [AnalysisStep.CLASSIFICATION]
        )

    def _run_full_pipeline(self):
        """Run the complete analysis pipeline."""
        image_names = self.data_manager.get_data_names()
        if not image_names:
            QMessageBox.warning(self, "Warning", "No data loaded")
            return

        # Get parameters
        params = self.parameter_manager.get_all_parameters()

        # Define pipeline steps
        steps = [
            AnalysisStep.DETECTION,
            AnalysisStep.LINKING,
            AnalysisStep.FEATURES,
            AnalysisStep.CLASSIFICATION
        ]

        # Run pipeline
        data = self.data_manager.get_data(image_names[0])
        self.analysis_engine.run_analysis_pipeline(data, params, steps)

    # Export operations
    def _export_csv(self):
        """Export current results to CSV."""
        # Get current results data
        current_data_name = self.data_browser.get_selected_data_name()
        if current_data_name is None:
            QMessageBox.warning(self, "Warning", "No data selected")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            if self.data_manager.save_data(current_data_name, file_path):
                self._update_status(f"Exported to {Path(file_path).name}")
            else:
                QMessageBox.warning(self, "Error", "Export failed")

    def _export_report(self):
        """Export analysis report."""
        current_data_name = self.data_browser.get_selected_data_name()
        if current_data_name is None:
            QMessageBox.warning(self, "Warning", "No data selected")
            return

        current_data = self.data_manager.get_data(current_data_name)
        if current_data is None or not isinstance(current_data, pd.DataFrame):
            QMessageBox.warning(self, "Warning", "No trajectory data selected")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Report", "",
            "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            if self.analysis_engine.export_analysis_report(current_data, file_path):
                self._update_status(f"Report exported to {Path(file_path).name}")
            else:
                QMessageBox.warning(self, "Error", "Report export failed")

    # Playback control
    def _toggle_playback(self):
        """Toggle playback of time series data."""
        if hasattr(self, 'playback_timer') and self.playback_timer.isActive():
            self.playback_timer.stop()
            self.play_button.setText("Play")
        else:
            if not hasattr(self, 'playback_timer'):
                self.playback_timer = QTimer()
                self.playback_timer.timeout.connect(self._advance_frame)

            self.playback_timer.start(100)  # 10 FPS
            self.play_button.setText("Pause")

    def _advance_frame(self):
        """Advance to next frame in playback."""
        current_value = self.frame_slider.value()
        max_value = self.frame_slider.maximum()

        if current_value < max_value:
            self.frame_slider.setValue(current_value + 1)
        else:
            self.frame_slider.setValue(0)  # Loop back to start

    # Signal handlers
    def _on_data_loaded(self, data_name: str, data: Any):
        """Handle data loaded signal."""
        self._update_status(f"Loaded: {data_name}")

        # Update visualization if it's image data
        if isinstance(data, np.ndarray) and len(data.shape) >= 2:
            self.visualization.set_image_data(data)

            # Update frame slider
            if len(data.shape) == 3:  # Time series
                self.frame_slider.setMaximum(data.shape[0] - 1)
                self.frame_slider.setValue(0)
                self._update_frame_display(0)

        # Update results if it's tabular data
        elif isinstance(data, pd.DataFrame):
            self._update_results_display(data)

    def _on_analysis_started(self, steps: List[str]):
        """Handle analysis started signal."""
        self._update_status(f"Running analysis: {', '.join(steps)}")
        self.progress_bar.setVisible(True)

    def _on_analysis_step_completed(self, step_name: str, result: Any):
        """Handle analysis step completed signal."""
        self._update_status(f"Completed: {step_name}")

        # Store result
        self.data_manager._data[f"result_{step_name}"] = result
        self.data_manager.dataLoaded.emit(f"result_{step_name}", result)

    def _on_analysis_completed(self, final_result: Any):
        """Handle analysis completed signal."""
        self._update_status("Analysis completed")
        self.progress_bar.setVisible(False)

        # Update displays
        if isinstance(final_result, pd.DataFrame):
            self._update_results_display(final_result)

    def _on_analysis_error(self, error_message: str):
        """Handle analysis error signal."""
        self._update_status("Analysis failed")
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Analysis Error", error_message)

    def _update_progress(self, message: str, percentage: int):
        """Update progress display."""
        self.status_label.setText(message)
        self.progress_bar.setValue(percentage)

    def _update_frame_display(self, frame: int):
        """Update frame display."""
        max_frame = self.frame_slider.maximum()
        self.frame_label.setText(f"Frame: {frame}/{max_frame}")

    def _update_results_display(self, data: pd.DataFrame):
        """Update results display with new data."""
        try:
            # Update statistics
            stats = self.analysis_engine.get_analysis_summary(data)
            stats_text = self._format_statistics(stats)
            self.stats_widget.setText(stats_text)

            # Update table (show first 1000 rows to avoid performance issues)
            display_data = data.head(1000)
            model = PandasTableModel(display_data)
            self.results_table.setModel(model)

            # Auto-resize columns
            self.results_table.resizeColumnsToContents()

        except Exception as e:
            self.logger.error(f"Error updating results display: {e}")

    def _format_statistics(self, stats: Dict[str, Any]) -> str:
        """Format statistics for display."""
        lines = ["Analysis Statistics", "=" * 30, ""]

        for key, value in stats.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.4f}")
            else:
                lines.append(f"{key}: {value}")

        return "\n".join(lines)

    def _update_status(self, message: str):
        """Update status bar message."""
        self.status_label.setText(message)
        self.logger.info(message)

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About Particle Tracking Analyzer",
            "Particle Tracking Analyzer v1.0\n\n"
            "A comprehensive application for analyzing particle trajectories "
            "from microscopy data.\n\n"
            "Features:\n"
            "• Particle detection and tracking\n"
            "• Trajectory feature calculation\n"
            "• Classification and analysis\n"
            "• Visualization and export tools"
        )
