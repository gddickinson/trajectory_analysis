#!/usr/bin/env python3
"""
Batch Control Widget Module
===========================

Provides comprehensive batch processing capabilities for hierarchical experiment analysis.
Supports experiment → conditions → files workflow with cross-condition statistical comparisons.
"""

import os
import logging
import json
import glob
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QGroupBox, QLabel, QPushButton, QLineEdit, QTextEdit, QComboBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QProgressBar, QTreeWidget,
    QTreeWidgetItem, QTabWidget, QSplitter, QFileDialog, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame, QScrollArea,
    QListWidget, QListWidgetItem, QSlider, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSettings, QAbstractTableModel,
    QModelIndex, QVariant
)
from PyQt6.QtGui import QFont, QIcon, QStandardItemModel, QStandardItem

from particle_tracker.core.analysis_engine import AnalysisEngine, AnalysisStep, AnalysisParameters
from particle_tracker.core.data_manager import EnhancedDataManager as DataManager, DataType


@dataclass
class BatchJobConfig:
    """Configuration for a batch processing job."""
    
    # Experiment structure
    experiment_dir: str = ""
    condition_folders: List[str] = None
    file_patterns: List[str] = None
    
    # Processing options
    process_hierarchy: bool = True  # True for experiment->conditions, False for single folder
    recursive_search: bool = True
    skip_existing: bool = True
    
    # Analysis parameters
    analysis_steps: List[str] = None
    parameters: Dict[str, Any] = None
    
    # Output options
    output_base_dir: str = ""
    create_summaries: bool = True
    create_cross_condition_plots: bool = True
    export_individual_results: bool = True
    export_formats: List[str] = None  # ['csv', 'excel', 'json']
    
    # Advanced options
    parallel_processing: bool = False
    max_workers: int = 4
    memory_limit_mb: int = 2048
    timeout_minutes: int = 60
    
    def __post_init__(self):
        if self.condition_folders is None:
            self.condition_folders = []
        if self.file_patterns is None:
            self.file_patterns = ["*.csv", "*.xlsx", "*.tif", "*.tiff"]
        if self.analysis_steps is None:
            self.analysis_steps = ["detection", "linking", "features", "classification"]
        if self.parameters is None:
            self.parameters = {}
        if self.export_formats is None:
            self.export_formats = ["csv", "excel"]


class BatchProgressModel(QAbstractTableModel):
    """Table model for displaying batch processing progress."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.jobs = []  # List of job dictionaries
        self.headers = ["File", "Condition", "Status", "Progress", "Time", "Results"]
    
    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self.jobs)
    
    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self.headers)
    
    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self.jobs):
            return QVariant()
        
        job = self.jobs[index.row()]
        col = index.column()
        
        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0:  # File
                return Path(job.get('file_path', '')).name
            elif col == 1:  # Condition
                return job.get('condition', '')
            elif col == 2:  # Status
                return job.get('status', 'Pending')
            elif col == 3:  # Progress
                progress = job.get('progress', 0)
                return f"{progress}%"
            elif col == 4:  # Time
                elapsed = job.get('elapsed_time', 0)
                if elapsed > 0:
                    return f"{elapsed:.1f}s"
                return ""
            elif col == 5:  # Results
                results = job.get('results', {})
                if results:
                    return f"{results.get('n_tracks', 0)} tracks"
                return ""
        
        return QVariant()
    
    def headerData(self, section: int, orientation: Qt.Orientation, 
                   role: int = Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.headers[section]
        return QVariant()
    
    def add_job(self, file_path: str, condition: str):
        """Add a new job to the model."""
        self.beginInsertRows(QModelIndex(), len(self.jobs), len(self.jobs))
        self.jobs.append({
            'file_path': file_path,
            'condition': condition,
            'status': 'Pending',
            'progress': 0,
            'elapsed_time': 0,
            'results': {}
        })
        self.endInsertRows()
    
    def update_job(self, index: int, **kwargs):
        """Update job information."""
        if 0 <= index < len(self.jobs):
            self.jobs[index].update(kwargs)
            self.dataChanged.emit(
                self.createIndex(index, 0),
                self.createIndex(index, len(self.headers) - 1)
            )


class BatchWorker(QThread):
    """Worker thread for batch processing."""
    
    # Signals
    jobStarted = pyqtSignal(int, str)  # job_index, file_path
    jobProgress = pyqtSignal(int, int)  # job_index, progress_percent
    jobCompleted = pyqtSignal(int, dict)  # job_index, results
    jobFailed = pyqtSignal(int, str)  # job_index, error_message
    batchCompleted = pyqtSignal(dict)  # summary_results
    
    def __init__(self, config: BatchJobConfig, analysis_engine: AnalysisEngine, 
                 data_manager: DataManager, parent=None):
        super().__init__(parent)
        self.config = config
        self.analysis_engine = analysis_engine
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        self.should_stop = False
        
        # Job queue
        self.jobs = []
        self.results_by_condition = {}
    
    def stop(self):
        """Stop the batch processing."""
        self.should_stop = True
        if self.analysis_engine:
            self.analysis_engine.stop_analysis()
    
    def add_jobs_from_config(self):
        """Build job queue from configuration."""
        self.jobs.clear()
        
        if self.config.process_hierarchy:
            # Process experiment directory with condition folders
            self._add_experiment_jobs()
        else:
            # Process single folder
            self._add_single_folder_jobs()
    
    def _add_experiment_jobs(self):
        """Add jobs for hierarchical experiment processing."""
        experiment_path = Path(self.config.experiment_dir)
        
        if not experiment_path.exists():
            self.logger.error(f"Experiment directory not found: {experiment_path}")
            return
        
        # Auto-detect condition folders if not specified
        if not self.config.condition_folders:
            condition_folders = [d.name for d in experiment_path.iterdir() 
                               if d.is_dir() and not d.name.startswith('.')]
        else:
            condition_folders = self.config.condition_folders
        
        # Add jobs for each condition
        for condition in condition_folders:
            condition_path = experiment_path / condition
            if condition_path.exists():
                files = self._find_files_in_directory(condition_path)
                for file_path in files:
                    self.jobs.append({
                        'file_path': str(file_path),
                        'condition': condition,
                        'output_dir': str(condition_path / 'analysis_results')
                    })
    
    def _add_single_folder_jobs(self):
        """Add jobs for single folder processing."""
        folder_path = Path(self.config.experiment_dir)
        files = self._find_files_in_directory(folder_path)
        
        for file_path in files:
            self.jobs.append({
                'file_path': str(file_path),
                'condition': 'default',
                'output_dir': str(folder_path / 'analysis_results')
            })
    
    def _find_files_in_directory(self, directory: Path) -> List[Path]:
        """Find files matching patterns in directory."""
        files = []
        
        for pattern in self.config.file_patterns:
            if self.config.recursive_search:
                pattern_path = directory / "**" / pattern
                files.extend(directory.glob(f"**/{pattern}"))
            else:
                files.extend(directory.glob(pattern))
        
        # Remove duplicates and sort
        files = sorted(list(set(files)))
        
        # Filter out already processed files if requested
        if self.config.skip_existing:
            filtered_files = []
            for file_path in files:
                output_file = self._get_output_file_path(file_path)
                if not output_file.exists():
                    filtered_files.append(file_path)
            files = filtered_files
        
        return files
    
    def _get_output_file_path(self, input_file: Path) -> Path:
        """Get the expected output file path for an input file."""
        # This would depend on your naming convention
        output_dir = input_file.parent / 'analysis_results'
        output_name = f"{input_file.stem}_analysis_results.csv"
        return output_dir / output_name
    
    def run(self):
        """Main processing loop."""
        try:
            self.add_jobs_from_config()
            
            if not self.jobs:
                self.logger.warning("No jobs to process")
                self.batchCompleted.emit({})
                return
            
            self.logger.info(f"Starting batch processing of {len(self.jobs)} files")
            
            # Initialize results storage
            self.results_by_condition = {}
            
            # Process each job
            for i, job in enumerate(self.jobs):
                if self.should_stop:
                    break
                
                self.jobStarted.emit(i, job['file_path'])
                
                try:
                    result = self._process_single_job(job, i)
                    self.jobCompleted.emit(i, result)
                    
                    # Store results by condition
                    condition = job['condition']
                    if condition not in self.results_by_condition:
                        self.results_by_condition[condition] = []
                    self.results_by_condition[condition].append(result)
                    
                except Exception as e:
                    self.logger.error(f"Job {i} failed: {e}")
                    self.jobFailed.emit(i, str(e))
            
            # Generate cross-condition summaries
            if self.config.create_summaries and not self.should_stop:
                summary_results = self._generate_summaries()
                self.batchCompleted.emit(summary_results)
            else:
                self.batchCompleted.emit({})
                
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            self.batchCompleted.emit({})
    
    def _process_single_job(self, job: Dict[str, Any], job_index: int) -> Dict[str, Any]:
        """Process a single file job."""
        file_path = job['file_path']
        output_dir = job['output_dir']
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.jobProgress.emit(job_index, 10)
        data_name = Path(file_path).stem
        success = self.data_manager.load_file(file_path, data_name)
        
        if not success:
            raise ValueError(f"Failed to load file: {file_path}")
        
        data = self.data_manager.get_data(data_name)
        
        # Create analysis parameters
        parameters = AnalysisParameters(**self.config.parameters)
        
        # Convert step names to AnalysisStep enums
        steps = []
        step_map = {
            'detection': AnalysisStep.DETECTION,
            'linking': AnalysisStep.LINKING,
            'features': AnalysisStep.FEATURES,
            'classification': AnalysisStep.CLASSIFICATION,
            'nearest_neighbors': AnalysisStep.NEAREST_NEIGHBORS,
            'diffusion': AnalysisStep.DIFFUSION,
            'velocity': AnalysisStep.VELOCITY
        }
        
        for step_name in self.config.analysis_steps:
            if step_name in step_map:
                steps.append(step_map[step_name])
        
        self.jobProgress.emit(job_index, 20)
        
        # Run analysis (this would need to be adapted to work synchronously)
        # For now, let's assume we have a synchronous analysis method
        result_data = self._run_analysis_sync(data, parameters, steps, job_index)
        
        self.jobProgress.emit(job_index, 90)
        
        # Save results
        if result_data is not None:
            output_file = Path(output_dir) / f"{data_name}_analysis_results.csv"
            if isinstance(result_data, pd.DataFrame):
                result_data.to_csv(output_file, index=False)
            
            # Generate results summary
            results = self._summarize_results(result_data)
            results['output_file'] = str(output_file)
            results['file_path'] = file_path
            
        else:
            results = {'error': 'Analysis failed'}
        
        self.jobProgress.emit(job_index, 100)
        return results
    
    def _run_analysis_sync(self, data, parameters, steps, job_index):
        """Run analysis synchronously (simplified version)."""
        # This is a simplified synchronous version
        # In practice, you'd need to adapt the analysis engine for batch processing
        
        current_data = data
        total_steps = len(steps)
        
        for i, step in enumerate(steps):
            if self.should_stop:
                break
            
            # Update progress
            step_progress = 20 + (60 * (i + 1) / total_steps)
            self.jobProgress.emit(job_index, int(step_progress))
            
            # Process step (this would need actual implementation)
            # For now, just return the input data
            pass
        
        return current_data
    
    def _summarize_results(self, data) -> Dict[str, Any]:
        """Generate summary statistics for analysis results."""
        summary = {}
        
        if isinstance(data, pd.DataFrame):
            summary['n_total_points'] = len(data)
            
            if 'track_number' in data.columns:
                summary['n_tracks'] = data['track_number'].nunique()
                track_lengths = data.groupby('track_number').size()
                summary['mean_track_length'] = track_lengths.mean()
                summary['median_track_length'] = track_lengths.median()
                summary['min_track_length'] = track_lengths.min()
                summary['max_track_length'] = track_lengths.max()
            
            if 'frame' in data.columns:
                summary['n_frames'] = data['frame'].nunique()
                summary['frame_range'] = (data['frame'].min(), data['frame'].max())
            
            # Add feature-specific summaries
            feature_columns = [
                'radius_gyration', 'asymmetry', 'fracDimension', 'velocity',
                'diffusion_coefficient', 'nn_distance', 'sRg'
            ]
            
            for col in feature_columns:
                if col in data.columns:
                    values = data[col].dropna()
                    if len(values) > 0:
                        summary[f'{col}_mean'] = values.mean()
                        summary[f'{col}_std'] = values.std()
                        summary[f'{col}_median'] = values.median()
            
            # Classification summaries
            if 'mobility_classification' in data.columns:
                mobility_counts = data.groupby('track_number')['mobility_classification'].first().value_counts()
                summary['mobility_counts'] = mobility_counts.to_dict()
            
            if 'SVM_label' in data.columns:
                svm_counts = data.groupby('track_number')['SVM_label'].first().value_counts()
                summary['svm_counts'] = svm_counts.to_dict()
        
        return summary
    
    def _generate_summaries(self) -> Dict[str, Any]:
        """Generate cross-condition summary statistics."""
        summary = {
            'total_conditions': len(self.results_by_condition),
            'total_files': sum(len(results) for results in self.results_by_condition.values()),
            'condition_summaries': {},
            'cross_condition_comparison': {}
        }
        
        # Generate per-condition summaries
        for condition, results in self.results_by_condition.items():
            if not results:
                continue
            
            condition_summary = {
                'n_files': len(results),
                'total_tracks': sum(r.get('n_tracks', 0) for r in results),
                'total_points': sum(r.get('n_total_points', 0) for r in results),
                'mean_tracks_per_file': np.mean([r.get('n_tracks', 0) for r in results]),
                'mean_track_length': np.mean([r.get('mean_track_length', 0) for r in results if r.get('mean_track_length')]),
            }
            
            # Aggregate feature statistics
            feature_means = {}
            for feature in ['radius_gyration_mean', 'velocity_mean', 'diffusion_coefficient_mean', 'sRg_mean']:
                values = [r.get(feature) for r in results if r.get(feature) is not None]
                if values:
                    feature_means[feature] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'sem': np.std(values) / np.sqrt(len(values))
                    }
            
            condition_summary['feature_statistics'] = feature_means
            summary['condition_summaries'][condition] = condition_summary
        
        # Generate cross-condition comparisons
        if len(self.results_by_condition) > 1:
            summary['cross_condition_comparison'] = self._generate_cross_condition_stats()
        
        return summary
    
    def _generate_cross_condition_stats(self) -> Dict[str, Any]:
        """Generate statistical comparisons across conditions."""
        comparison = {}
        
        # Compare key metrics across conditions
        metrics = ['n_tracks', 'mean_track_length', 'radius_gyration_mean', 'velocity_mean']
        
        for metric in metrics:
            condition_values = {}
            for condition, results in self.results_by_condition.items():
                values = [r.get(metric) for r in results if r.get(metric) is not None]
                if values:
                    condition_values[condition] = {
                        'values': values,
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'n': len(values)
                    }
            
            if len(condition_values) >= 2:
                comparison[metric] = condition_values
        
        return comparison


class BatchControlWidget(QWidget):
    """Main widget for batch processing control."""
    
    # Signals
    batchStarted = pyqtSignal()
    batchCompleted = pyqtSignal(dict)
    batchProgress = pyqtSignal(int, int)  # current_job, total_jobs
    
    def __init__(self, analysis_engine: AnalysisEngine, data_manager: DataManager, 
                 parameter_manager=None, parent=None):
        super().__init__(parent)
        
        self.logger = logging.getLogger(__name__)
        self.analysis_engine = analysis_engine
        self.data_manager = data_manager
        self.parameter_manager = parameter_manager
        
        # Batch processing
        self.current_worker = None
        self.config = BatchJobConfig()
        
        # UI components
        self.progress_model = BatchProgressModel()
        
        self._setup_ui()
        self._connect_signals()
        self._load_settings()
    
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        
        # Create tab widget for different sections
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Configuration tab
        config_tab = self._create_config_tab()
        self.tab_widget.addTab(config_tab, "Configuration")
        
        # Progress tab
        progress_tab = self._create_progress_tab()
        self.tab_widget.addTab(progress_tab, "Progress")
        
        # Results tab
        results_tab = self._create_results_tab()
        self.tab_widget.addTab(results_tab, "Results")
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Batch Processing")
        self.start_button.clicked.connect(self._start_batch)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop_batch)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        self.save_config_button = QPushButton("Save Configuration")
        self.save_config_button.clicked.connect(self._save_config)
        button_layout.addWidget(self.save_config_button)
        
        self.load_config_button = QPushButton("Load Configuration")
        self.load_config_button.clicked.connect(self._load_config)
        button_layout.addWidget(self.load_config_button)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
    
    def _create_config_tab(self) -> QWidget:
        """Create the configuration tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Experiment structure section
        experiment_group = QGroupBox("Experiment Structure")
        experiment_layout = QFormLayout(experiment_group)
        
        # Directory selection
        dir_layout = QHBoxLayout()
        self.experiment_dir_edit = QLineEdit()
        self.experiment_dir_button = QPushButton("Browse...")
        self.experiment_dir_button.clicked.connect(self._browse_experiment_dir)
        dir_layout.addWidget(self.experiment_dir_edit)
        dir_layout.addWidget(self.experiment_dir_button)
        experiment_layout.addRow("Experiment Directory:", dir_layout)
        
        # Processing mode
        self.hierarchy_radio = QRadioButton("Hierarchical (Experiment → Conditions)")
        self.single_folder_radio = QRadioButton("Single Folder")
        self.hierarchy_radio.setChecked(True)
        
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.hierarchy_radio)
        mode_layout.addWidget(self.single_folder_radio)
        experiment_layout.addRow("Processing Mode:", mode_layout)
        
        # Condition folders
        self.condition_list = QListWidget()
        self.condition_list.setMaximumHeight(100)
        condition_buttons = QHBoxLayout()
        self.add_condition_button = QPushButton("Add")
        self.add_condition_button.clicked.connect(self._add_condition)
        self.remove_condition_button = QPushButton("Remove")
        self.remove_condition_button.clicked.connect(self._remove_condition)
        self.auto_detect_button = QPushButton("Auto-Detect")
        self.auto_detect_button.clicked.connect(self._auto_detect_conditions)
        condition_buttons.addWidget(self.add_condition_button)
        condition_buttons.addWidget(self.remove_condition_button)
        condition_buttons.addWidget(self.auto_detect_button)
        condition_buttons.addStretch()
        
        condition_widget = QWidget()
        condition_layout = QVBoxLayout(condition_widget)
        condition_layout.addWidget(self.condition_list)
        condition_layout.addLayout(condition_buttons)
        experiment_layout.addRow("Condition Folders:", condition_widget)
        
        # File patterns
        self.file_patterns_edit = QLineEdit("*.csv,*.xlsx,*.tif,*.tiff")
        experiment_layout.addRow("File Patterns:", self.file_patterns_edit)
        
        layout.addWidget(experiment_group)
        
        # Processing options section
        processing_group = QGroupBox("Processing Options")
        processing_layout = QFormLayout(processing_group)
        
        self.recursive_cb = QCheckBox("Recursive search")
        self.recursive_cb.setChecked(True)
        processing_layout.addRow("", self.recursive_cb)
        
        self.skip_existing_cb = QCheckBox("Skip existing results")
        self.skip_existing_cb.setChecked(True)
        processing_layout.addRow("", self.skip_existing_cb)
        
        self.parallel_cb = QCheckBox("Parallel processing")
        processing_layout.addRow("", self.parallel_cb)
        
        self.max_workers_spin = QSpinBox()
        self.max_workers_spin.setRange(1, 16)
        self.max_workers_spin.setValue(4)
        processing_layout.addRow("Max Workers:", self.max_workers_spin)
        
        layout.addWidget(processing_group)
        
        # Analysis steps section
        steps_group = QGroupBox("Analysis Steps")
        steps_layout = QGridLayout(steps_group)
        
        self.step_checkboxes = {}
        steps = [
            ("detection", "Particle Detection"),
            ("linking", "Trajectory Linking"),
            ("features", "Feature Calculation"),
            ("classification", "Classification"),
            ("nearest_neighbors", "Nearest Neighbors"),
            ("diffusion", "Diffusion Analysis"),
            ("velocity", "Velocity Analysis")
        ]
        
        for i, (step_key, step_label) in enumerate(steps):
            cb = QCheckBox(step_label)
            cb.setChecked(step_key in ["detection", "linking", "features", "classification"])
            self.step_checkboxes[step_key] = cb
            steps_layout.addWidget(cb, i // 2, i % 2)
        
        layout.addWidget(steps_group)
        
        # Output options section
        output_group = QGroupBox("Output Options")
        output_layout = QFormLayout(output_group)
        
        self.create_summaries_cb = QCheckBox("Create condition summaries")
        self.create_summaries_cb.setChecked(True)
        output_layout.addRow("", self.create_summaries_cb)
        
        self.cross_condition_plots_cb = QCheckBox("Create cross-condition plots")
        self.cross_condition_plots_cb.setChecked(True)
        output_layout.addRow("", self.cross_condition_plots_cb)
        
        self.export_individual_cb = QCheckBox("Export individual results")
        self.export_individual_cb.setChecked(True)
        output_layout.addRow("", self.export_individual_cb)
        
        # Export formats
        export_layout = QHBoxLayout()
        self.export_csv_cb = QCheckBox("CSV")
        self.export_csv_cb.setChecked(True)
        self.export_excel_cb = QCheckBox("Excel")
        self.export_excel_cb.setChecked(True)
        self.export_json_cb = QCheckBox("JSON")
        export_layout.addWidget(self.export_csv_cb)
        export_layout.addWidget(self.export_excel_cb)
        export_layout.addWidget(self.export_json_cb)
        export_layout.addStretch()
        output_layout.addRow("Export Formats:", export_layout)
        
        layout.addWidget(output_group)
        
        layout.addStretch()
        return tab
    
    def _create_progress_tab(self) -> QWidget:
        """Create the progress monitoring tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Overall progress
        progress_group = QGroupBox("Overall Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.overall_progress = QProgressBar()
        progress_layout.addWidget(self.overall_progress)
        
        self.progress_label = QLabel("Ready")
        progress_layout.addWidget(self.progress_label)
        
        layout.addWidget(progress_group)
        
        # Job progress table
        table_group = QGroupBox("Job Progress")
        table_layout = QVBoxLayout(table_group)
        
        from PyQt6.QtWidgets import QTableView
        self.progress_table = QTableView()
        self.progress_table.setModel(self.progress_model)
        self.progress_table.horizontalHeader().setStretchLastSection(True)
        table_layout.addWidget(self.progress_table)
        
        layout.addWidget(table_group)
        
        return tab
    
    def _create_results_tab(self) -> QWidget:
        """Create the results summary tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Results summary
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Courier", 10))
        layout.addWidget(self.results_text)
        
        # Export buttons
        export_layout = QHBoxLayout()
        
        self.export_summary_button = QPushButton("Export Summary")
        self.export_summary_button.clicked.connect(self._export_summary)
        export_layout.addWidget(self.export_summary_button)
        
        self.export_plots_button = QPushButton("Export Plots")
        self.export_plots_button.clicked.connect(self._export_plots)
        export_layout.addWidget(self.export_plots_button)
        
        export_layout.addStretch()
        layout.addLayout(export_layout)
        
        return tab
    
    def _connect_signals(self):
        """Connect internal signals."""
        # Enable/disable condition controls based on processing mode
        self.hierarchy_radio.toggled.connect(self._on_mode_changed)
        self.single_folder_radio.toggled.connect(self._on_mode_changed)
    
    def _on_mode_changed(self):
        """Handle processing mode change."""
        hierarchy_mode = self.hierarchy_radio.isChecked()
        
        # Enable/disable condition-specific controls
        self.condition_list.setEnabled(hierarchy_mode)
        self.add_condition_button.setEnabled(hierarchy_mode)
        self.remove_condition_button.setEnabled(hierarchy_mode)
        self.auto_detect_button.setEnabled(hierarchy_mode)
        self.cross_condition_plots_cb.setEnabled(hierarchy_mode)
    
    def _browse_experiment_dir(self):
        """Browse for experiment directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Experiment Directory", self.experiment_dir_edit.text()
        )
        if dir_path:
            self.experiment_dir_edit.setText(dir_path)
            # Auto-detect conditions if in hierarchy mode
            if self.hierarchy_radio.isChecked():
                self._auto_detect_conditions()
    
    def _add_condition(self):
        """Add a condition folder."""
        from PyQt6.QtWidgets import QInputDialog
        condition, ok = QInputDialog.getText(
            self, "Add Condition", "Condition folder name:"
        )
        if ok and condition:
            self.condition_list.addItem(condition)
    
    def _remove_condition(self):
        """Remove selected condition."""
        current_row = self.condition_list.currentRow()
        if current_row >= 0:
            self.condition_list.takeItem(current_row)
    
    def _auto_detect_conditions(self):
        """Auto-detect condition folders in experiment directory."""
        experiment_dir = self.experiment_dir_edit.text()
        if not experiment_dir or not Path(experiment_dir).exists():
            return
        
        self.condition_list.clear()
        
        # Find subdirectories
        try:
            experiment_path = Path(experiment_dir)
            subdirs = [d.name for d in experiment_path.iterdir() 
                      if d.is_dir() and not d.name.startswith('.')]
            
            for subdir in sorted(subdirs):
                self.condition_list.addItem(subdir)
                
            self.logger.info(f"Auto-detected {len(subdirs)} condition folders")
            
        except Exception as e:
            self.logger.error(f"Error auto-detecting conditions: {e}")
    
    def _update_config_from_ui(self):
        """Update configuration from UI values."""
        # Basic settings
        self.config.experiment_dir = self.experiment_dir_edit.text()
        self.config.process_hierarchy = self.hierarchy_radio.isChecked()
        self.config.recursive_search = self.recursive_cb.isChecked()
        self.config.skip_existing = self.skip_existing_cb.isChecked()
        self.config.parallel_processing = self.parallel_cb.isChecked()
        self.config.max_workers = self.max_workers_spin.value()
        
        # Condition folders
        self.config.condition_folders = []
        for i in range(self.condition_list.count()):
            self.config.condition_folders.append(self.condition_list.item(i).text())
        
        # File patterns
        patterns_text = self.file_patterns_edit.text()
        self.config.file_patterns = [p.strip() for p in patterns_text.split(',') if p.strip()]
        
        # Analysis steps
        self.config.analysis_steps = []
        for step_key, checkbox in self.step_checkboxes.items():
            if checkbox.isChecked():
                self.config.analysis_steps.append(step_key)
        
        # Output options
        self.config.create_summaries = self.create_summaries_cb.isChecked()
        self.config.create_cross_condition_plots = self.cross_condition_plots_cb.isChecked()
        self.config.export_individual_results = self.export_individual_cb.isChecked()
        
        # Export formats
        self.config.export_formats = []
        if self.export_csv_cb.isChecked():
            self.config.export_formats.append('csv')
        if self.export_excel_cb.isChecked():
            self.config.export_formats.append('excel')
        if self.export_json_cb.isChecked():
            self.config.export_formats.append('json')
        
        # Get parameters from parameter manager if available
        if self.parameter_manager:
            try:
                params = self.parameter_manager.get_all_parameters()
                if hasattr(params, 'to_dict'):
                    self.config.parameters = params.to_dict()
                elif isinstance(params, dict):
                    self.config.parameters = params
                else:
                    from dataclasses import asdict
                    self.config.parameters = asdict(params)
            except Exception as e:
                self.logger.warning(f"Could not get parameters: {e}")
                self.config.parameters = {}
    
    def _start_batch(self):
        """Start batch processing."""
        try:
            # Update configuration from UI
            self._update_config_from_ui()
            
            # Validate configuration
            if not self.config.experiment_dir:
                QMessageBox.warning(self, "Configuration Error", 
                                  "Please select an experiment directory")
                return
            
            if not Path(self.config.experiment_dir).exists():
                QMessageBox.warning(self, "Configuration Error", 
                                  "Experiment directory does not exist")
                return
            
            if not self.config.analysis_steps:
                QMessageBox.warning(self, "Configuration Error", 
                                  "Please select at least one analysis step")
                return
            
            # Create and start worker
            self.current_worker = BatchWorker(
                self.config, self.analysis_engine, self.data_manager
            )
            
            # Connect worker signals
            self.current_worker.jobStarted.connect(self._on_job_started)
            self.current_worker.jobProgress.connect(self._on_job_progress)
            self.current_worker.jobCompleted.connect(self._on_job_completed)
            self.current_worker.jobFailed.connect(self._on_job_failed)
            self.current_worker.batchCompleted.connect(self._on_batch_completed)
            
            # Reset progress tracking
            self.progress_model.jobs.clear()
            self.progress_model.beginResetModel()
            self.progress_model.endResetModel()
            
            # Add jobs to progress model
            self.current_worker.add_jobs_from_config()
            for job in self.current_worker.jobs:
                self.progress_model.add_job(job['file_path'], job['condition'])
            
            # Update UI state
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.overall_progress.setValue(0)
            self.progress_label.setText("Starting batch processing...")
            
            # Switch to progress tab
            self.tab_widget.setCurrentIndex(1)
            
            # Start processing
            self.current_worker.start()
            self.batchStarted.emit()
            
            self.logger.info(f"Started batch processing with {len(self.current_worker.jobs)} jobs")
            
        except Exception as e:
            self.logger.error(f"Error starting batch processing: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start batch processing:\n{e}")
    
    def _stop_batch(self):
        """Stop batch processing."""
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.stop()
            self.current_worker.wait()
            
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.progress_label.setText("Batch processing stopped")
            
            self.logger.info("Batch processing stopped by user")
    
    def _on_job_started(self, job_index: int, file_path: str):
        """Handle job started signal."""
        self.progress_model.update_job(job_index, status="Processing")
        self.progress_label.setText(f"Processing {Path(file_path).name}...")
    
    def _on_job_progress(self, job_index: int, progress: int):
        """Handle job progress signal."""
        self.progress_model.update_job(job_index, progress=progress)
        
        # Update overall progress
        total_jobs = len(self.progress_model.jobs)
        completed_jobs = sum(1 for job in self.progress_model.jobs 
                           if job.get('status') == 'Completed')
        current_progress = (completed_jobs * 100 + progress) / total_jobs
        self.overall_progress.setValue(int(current_progress))
    
    def _on_job_completed(self, job_index: int, results: Dict[str, Any]):
        """Handle job completed signal."""
        self.progress_model.update_job(
            job_index, 
            status="Completed", 
            progress=100,
            results=results
        )
        
        # Update overall progress
        total_jobs = len(self.progress_model.jobs)
        completed_jobs = sum(1 for job in self.progress_model.jobs 
                           if job.get('status') in ['Completed', 'Failed'])
        overall_progress = (completed_jobs / total_jobs) * 100
        self.overall_progress.setValue(int(overall_progress))
        
        self.progress_label.setText(f"Completed {completed_jobs}/{total_jobs} jobs")
    
    def _on_job_failed(self, job_index: int, error_message: str):
        """Handle job failed signal."""
        self.progress_model.update_job(
            job_index, 
            status="Failed", 
            progress=0,
            results={'error': error_message}
        )
        
        self.logger.error(f"Job {job_index} failed: {error_message}")
    
    def _on_batch_completed(self, summary_results: Dict[str, Any]):
        """Handle batch completed signal."""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.overall_progress.setValue(100)
        self.progress_label.setText("Batch processing completed")
        
        # Display results summary
        self._display_results_summary(summary_results)
        
        # Switch to results tab
        self.tab_widget.setCurrentIndex(2)
        
        self.batchCompleted.emit(summary_results)
        self.logger.info("Batch processing completed")
    
    def _display_results_summary(self, summary_results: Dict[str, Any]):
        """Display results summary in the results tab."""
        if not summary_results:
            self.results_text.setText("No results available.")
            return
        
        lines = []
        lines.append("Batch Processing Results Summary")
        lines.append("=" * 50)
        lines.append("")
        
        # Overall statistics
        lines.append(f"Total Conditions: {summary_results.get('total_conditions', 0)}")
        lines.append(f"Total Files Processed: {summary_results.get('total_files', 0)}")
        lines.append("")
        
        # Per-condition summaries
        condition_summaries = summary_results.get('condition_summaries', {})
        if condition_summaries:
            lines.append("Per-Condition Summary:")
            lines.append("-" * 30)
            
            for condition, summary in condition_summaries.items():
                lines.append(f"\n{condition}:")
                lines.append(f"  Files: {summary.get('n_files', 0)}")
                lines.append(f"  Total Tracks: {summary.get('total_tracks', 0)}")
                lines.append(f"  Total Points: {summary.get('total_points', 0)}")
                lines.append(f"  Mean Tracks/File: {summary.get('mean_tracks_per_file', 0):.1f}")
                
                if 'mean_track_length' in summary and summary['mean_track_length']:
                    lines.append(f"  Mean Track Length: {summary['mean_track_length']:.1f}")
                
                # Feature statistics
                feature_stats = summary.get('feature_statistics', {})
                if feature_stats:
                    lines.append("  Feature Statistics:")
                    for feature, stats in feature_stats.items():
                        if stats and 'mean' in stats:
                            lines.append(f"    {feature}: {stats['mean']:.3f} ± {stats.get('sem', 0):.3f}")
        
        # Cross-condition comparisons
        cross_comparison = summary_results.get('cross_condition_comparison', {})
        if cross_comparison and len(condition_summaries) > 1:
            lines.append("\n\nCross-Condition Comparison:")
            lines.append("-" * 30)
            
            for metric, condition_data in cross_comparison.items():
                lines.append(f"\n{metric}:")
                for condition, stats in condition_data.items():
                    mean = stats.get('mean', 0)
                    std = stats.get('std', 0)
                    n = stats.get('n', 0)
                    lines.append(f"  {condition}: {mean:.3f} ± {std:.3f} (n={n})")
        
        # Job-level summary
        failed_jobs = [job for job in self.progress_model.jobs if job.get('status') == 'Failed']
        if failed_jobs:
            lines.append(f"\n\nFailed Jobs ({len(failed_jobs)}):")
            lines.append("-" * 20)
            for job in failed_jobs:
                file_name = Path(job.get('file_path', '')).name
                error = job.get('results', {}).get('error', 'Unknown error')
                lines.append(f"  {file_name}: {error}")
        
        self.results_text.setText("\n".join(lines))
    
    def _save_config(self):
        """Save current configuration to file."""
        self._update_config_from_ui()
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "",
            "JSON files (*.json);;All files (*.*)"
        )
        
        if file_path:
            try:
                config_dict = asdict(self.config)
                with open(file_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
                
                QMessageBox.information(self, "Success", 
                                      f"Configuration saved to {file_path}")
                self.logger.info(f"Configuration saved to {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", 
                                   f"Failed to save configuration:\n{e}")
                self.logger.error(f"Error saving configuration: {e}")
    
    def _load_config(self):
        """Load configuration from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "",
            "JSON files (*.json);;All files (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config_dict = json.load(f)
                
                self.config = BatchJobConfig(**config_dict)
                self._update_ui_from_config()
                
                QMessageBox.information(self, "Success", 
                                      f"Configuration loaded from {file_path}")
                self.logger.info(f"Configuration loaded from {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", 
                                   f"Failed to load configuration:\n{e}")
                self.logger.error(f"Error loading configuration: {e}")
    
    def _update_ui_from_config(self):
        """Update UI controls from configuration."""
        # Basic settings
        self.experiment_dir_edit.setText(self.config.experiment_dir)
        self.hierarchy_radio.setChecked(self.config.process_hierarchy)
        self.single_folder_radio.setChecked(not self.config.process_hierarchy)
        self.recursive_cb.setChecked(self.config.recursive_search)
        self.skip_existing_cb.setChecked(self.config.skip_existing)
        self.parallel_cb.setChecked(self.config.parallel_processing)
        self.max_workers_spin.setValue(self.config.max_workers)
        
        # Condition folders
        self.condition_list.clear()
        for condition in self.config.condition_folders:
            self.condition_list.addItem(condition)
        
        # File patterns
        self.file_patterns_edit.setText(','.join(self.config.file_patterns))
        
        # Analysis steps
        for step_key, checkbox in self.step_checkboxes.items():
            checkbox.setChecked(step_key in self.config.analysis_steps)
        
        # Output options
        self.create_summaries_cb.setChecked(self.config.create_summaries)
        self.cross_condition_plots_cb.setChecked(self.config.create_cross_condition_plots)
        self.export_individual_cb.setChecked(self.config.export_individual_results)
        
        # Export formats
        self.export_csv_cb.setChecked('csv' in self.config.export_formats)
        self.export_excel_cb.setChecked('excel' in self.config.export_formats)
        self.export_json_cb.setChecked('json' in self.config.export_formats)
        
        # Update mode-dependent controls
        self._on_mode_changed()
    
    def _export_summary(self):
        """Export results summary to file."""
        if not self.results_text.toPlainText():
            QMessageBox.warning(self, "No Results", "No results to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Summary", "batch_results_summary.txt",
            "Text files (*.txt);;All files (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.results_text.toPlainText())
                
                QMessageBox.information(self, "Success", 
                                      f"Summary exported to {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", 
                                   f"Failed to export summary:\n{e}")
    
    def _export_plots(self):
        """Export cross-condition comparison plots."""
        # This would integrate with your plotting functionality
        QMessageBox.information(self, "Plots", 
                               "Plot export functionality would be implemented here")
    
    def _load_settings(self):
        """Load widget settings."""
        settings = QSettings()
        settings.beginGroup("BatchControl")
        
        # Restore UI state
        self.experiment_dir_edit.setText(
            settings.value("experiment_dir", "")
        )
        self.hierarchy_radio.setChecked(
            settings.value("hierarchy_mode", True, type=bool)
        )
        
        settings.endGroup()
    
    def _save_settings(self):
        """Save widget settings."""
        settings = QSettings()
        settings.beginGroup("BatchControl")
        
        settings.setValue("experiment_dir", self.experiment_dir_edit.text())
        settings.setValue("hierarchy_mode", self.hierarchy_radio.isChecked())
        
        settings.endGroup()
    
    def closeEvent(self, event):
        """Handle widget close event."""
        # Stop any running batch processing
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.stop()
            self.current_worker.wait()
        
        self._save_settings()
        super().closeEvent(event)