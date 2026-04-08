#!/usr/bin/env python3
"""
Enhanced Project Manager
========================

Supports hierarchical experiment structure with sophisticated batch processing,
cross-condition analysis, and comprehensive metadata management.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
import zipfile
import shutil
from enum import Enum

from PyQt6.QtCore import QObject, pyqtSignal, QThread, pyqtSlot
import pandas as pd
import numpy as np


class ProjectType(Enum):
    """Types of projects supported."""
    SINGLE_FILE = "single_file"
    CONDITION = "condition"  # Multiple files, single condition
    EXPERIMENT = "experiment"  # Multiple conditions


class AnalysisStatus(Enum):
    """Status of analysis for files/conditions."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class FileInfo:
    """Information about a single data file."""
    file_path: str
    file_type: str = ""  # 'raw_image', 'localizations', 'trajectories', etc.
    file_size: int = 0
    creation_date: str = ""
    analysis_status: AnalysisStatus = AnalysisStatus.NOT_STARTED
    analysis_results: List[str] = field(default_factory=list)
    roi_files: List[str] = field(default_factory=list)
    background_files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_messages: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.creation_date:
            try:
                self.creation_date = datetime.fromtimestamp(
                    Path(self.file_path).stat().st_mtime
                ).isoformat()
                self.file_size = Path(self.file_path).stat().st_size
            except Exception:
                self.creation_date = datetime.now().isoformat()


@dataclass
class ConditionInfo:
    """Information about an experimental condition."""
    name: str
    description: str = ""
    files: List[FileInfo] = field(default_factory=list)
    condition_parameters: Dict[str, Any] = field(default_factory=dict)
    analysis_status: AnalysisStatus = AnalysisStatus.NOT_STARTED
    summary_statistics: Dict[str, Any] = field(default_factory=dict)
    output_directory: str = ""
    notes: str = ""

    def add_file(self, file_path: str, file_type: str = ""):
        """Add a file to this condition."""
        file_info = FileInfo(file_path=file_path, file_type=file_type)
        self.files.append(file_info)

    def get_files_by_type(self, file_type: str) -> List[FileInfo]:
        """Get all files of a specific type."""
        return [f for f in self.files if f.file_type == file_type]

    def get_completed_files(self) -> List[FileInfo]:
        """Get all files with completed analysis."""
        return [f for f in self.files if f.analysis_status == AnalysisStatus.COMPLETED]


@dataclass
class ExperimentInfo:
    """Information about a complete experiment with multiple conditions."""
    name: str
    description: str = ""
    experiment_type: str = ""
    created_date: str = ""
    modified_date: str = ""
    version: str = "2.0"
    
    # Experiment structure
    conditions: List[ConditionInfo] = field(default_factory=list)
    
    # Global parameters
    global_parameters: Dict[str, Any] = field(default_factory=dict)
    pixel_size: float = 108.0
    frame_rate: float = 10.0
    
    # Analysis configuration
    analysis_pipeline: List[str] = field(default_factory=list)
    comparison_groups: List[List[str]] = field(default_factory=list)  # Groups of conditions to compare
    
    # Results and statistics
    cross_condition_results: Dict[str, Any] = field(default_factory=dict)
    experiment_statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Output management
    base_output_directory: str = ""
    export_formats: List[str] = field(default_factory=lambda: ["csv", "excel"])
    
    # Metadata
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.created_date:
            self.created_date = datetime.now().isoformat()
        if not self.analysis_pipeline:
            self.analysis_pipeline = [
                "detection", "linking", "features", "classification", 
                "density_analysis", "autocorrelation"
            ]

    def add_condition(self, condition: ConditionInfo):
        """Add a condition to the experiment."""
        self.conditions.append(condition)

    def get_condition(self, name: str) -> Optional[ConditionInfo]:
        """Get a condition by name."""
        for condition in self.conditions:
            if condition.name == name:
                return condition
        return None

    def get_all_files(self) -> List[FileInfo]:
        """Get all files across all conditions."""
        all_files = []
        for condition in self.conditions:
            all_files.extend(condition.files)
        return all_files


# Legacy ProjectInfo for backward compatibility
@dataclass
class ProjectInfo:
    """Legacy project info for single-file projects."""
    name: str
    description: str = ""
    created_date: str = ""
    modified_date: str = ""
    version: str = "1.0"
    data_files: List[str] = field(default_factory=list)
    analysis_results: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    pixel_size: float = 108.0
    frame_rate: float = 10.0
    experiment_type: str = ""
    notes: str = ""

    def __post_init__(self):
        if not self.created_date:
            self.created_date = datetime.now().isoformat()


class BatchAnalysisWorker(QThread):
    """Worker thread for batch analysis operations."""
    
    progressUpdate = pyqtSignal(str, int)  # message, percentage
    conditionCompleted = pyqtSignal(str, dict)  # condition_name, results
    analysisCompleted = pyqtSignal(dict)  # final_results
    errorOccurred = pyqtSignal(str, str)  # condition_name, error_message
    
    def __init__(self, experiment: ExperimentInfo, analysis_engine, parent=None):
        super().__init__(parent)
        self.experiment = experiment
        self.analysis_engine = analysis_engine
        self.should_stop = False
        
    def run(self):
        """Run batch analysis for the entire experiment."""
        try:
            total_conditions = len(self.experiment.conditions)
            completed_conditions = 0
            
            for condition in self.experiment.conditions:
                if self.should_stop:
                    break
                    
                self.progressUpdate.emit(
                    f"Processing condition: {condition.name}", 
                    int(100 * completed_conditions / total_conditions)
                )
                
                try:
                    # Process all files in this condition
                    condition_results = self._process_condition(condition)
                    condition.analysis_status = AnalysisStatus.COMPLETED
                    condition.summary_statistics = condition_results
                    
                    self.conditionCompleted.emit(condition.name, condition_results)
                    
                except Exception as e:
                    condition.analysis_status = AnalysisStatus.FAILED
                    self.errorOccurred.emit(condition.name, str(e))
                
                completed_conditions += 1
            
            # Generate cross-condition analysis
            if not self.should_stop:
                self.progressUpdate.emit("Generating cross-condition analysis...", 95)
                cross_results = self._generate_cross_condition_analysis()
                self.experiment.cross_condition_results = cross_results
                
            self.progressUpdate.emit("Analysis complete", 100)
            self.analysisCompleted.emit(self.experiment.cross_condition_results)
            
        except Exception as e:
            self.errorOccurred.emit("General", str(e))
    
    def stop(self):
        """Stop the analysis."""
        self.should_stop = True
    
    def _process_condition(self, condition: ConditionInfo) -> Dict[str, Any]:
        """Process all files in a condition."""
        # Implementation would depend on your analysis engine
        # This is a placeholder for the actual analysis logic
        results = {
            'n_files': len(condition.files),
            'n_tracks': 0,
            'mean_track_length': 0,
            'mobile_fraction': 0,
            'linear_fraction': 0
        }
        
        for file_info in condition.files:
            # Process each file using the analysis engine
            # file_results = self.analysis_engine.process_file(file_info.file_path)
            # Accumulate results
            pass
            
        return results
    
    def _generate_cross_condition_analysis(self) -> Dict[str, Any]:
        """Generate cross-condition comparison analysis."""
        # Implementation for cross-condition statistics
        return {
            'condition_comparison': {},
            'statistical_tests': {},
            'summary_plots': []
        }


class EnhancedProjectManager(QObject):
    """Enhanced project manager supporting hierarchical experiments."""

    # Signals
    projectLoaded = pyqtSignal(str)  # project_path
    projectSaved = pyqtSignal(str)   # project_path
    projectClosed = pyqtSignal()
    batchAnalysisStarted = pyqtSignal(str)  # experiment_name
    batchAnalysisProgress = pyqtSignal(str, int)  # message, percentage
    batchAnalysisCompleted = pyqtSignal(dict)  # results
    conditionCompleted = pyqtSignal(str, dict)  # condition_name, results

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Current project (can be legacy or enhanced)
        self.current_project: Optional[Union[ProjectInfo, ExperimentInfo]] = None
        self.current_project_path: Optional[str] = None
        self.project_modified: bool = False
        self.project_type: ProjectType = ProjectType.SINGLE_FILE

        # Batch analysis
        self.batch_worker: Optional[BatchAnalysisWorker] = None
        self.analysis_engine = None  # Set from main application

        self.logger.info("Enhanced project manager initialized")

    def set_analysis_engine(self, analysis_engine):
        """Set the analysis engine for batch processing."""
        self.analysis_engine = analysis_engine

    # =================================================================
    # Project Creation Methods
    # =================================================================

    def new_single_file_project(self, name: str = "Untitled Project", 
                               description: str = "") -> ProjectInfo:
        """Create a new single-file project (legacy)."""
        self.current_project = ProjectInfo(name=name, description=description)
        self.current_project_path = None
        self.project_modified = True
        self.project_type = ProjectType.SINGLE_FILE
        
        self.logger.info(f"Created new single-file project: {name}")
        return self.current_project

    def new_condition_project(self, name: str, description: str = "") -> ExperimentInfo:
        """Create a new condition project (multiple files, single condition)."""
        experiment = ExperimentInfo(name=name, description=description)
        condition = ConditionInfo(name="Default", description="Default condition")
        experiment.add_condition(condition)
        
        self.current_project = experiment
        self.current_project_path = None
        self.project_modified = True
        self.project_type = ProjectType.CONDITION
        
        self.logger.info(f"Created new condition project: {name}")
        return self.current_project

    def new_experiment_project(self, name: str, description: str = "") -> ExperimentInfo:
        """Create a new experiment project (multiple conditions)."""
        self.current_project = ExperimentInfo(name=name, description=description)
        self.current_project_path = None
        self.project_modified = True
        self.project_type = ProjectType.EXPERIMENT
        
        self.logger.info(f"Created new experiment project: {name}")
        return self.current_project

    def create_from_directory_structure(self, base_path: str, 
                                       name: Optional[str] = None) -> ExperimentInfo:
        """Create an experiment project from directory structure.
        
        Expected structure:
        base_path/
        ├── condition1/
        │   ├── file1.csv
        │   ├── file2.csv
        │   └── ROI_files/ (optional)
        ├── condition2/
        │   ├── file1.csv
        │   └── file2.csv
        └── ...
        """
        base_path = Path(base_path)
        if name is None:
            name = base_path.name
            
        experiment = ExperimentInfo(
            name=name,
            description=f"Auto-created from {base_path}",
            base_output_directory=str(base_path / "results")
        )

        # Scan for condition directories
        for item in base_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                condition = ConditionInfo(
                    name=item.name,
                    description=f"Condition from {item.name}",
                    output_directory=str(item / "autocorrelation_output")
                )
                
                # Scan for data files in condition directory
                data_extensions = ['.csv', '.tif', '.tiff', '.xlsx']
                for file_path in item.iterdir():
                    if file_path.suffix.lower() in data_extensions:
                        file_type = self._determine_file_type(file_path)
                        condition.add_file(str(file_path), file_type)
                
                # Look for ROI files
                roi_dir = item / "ROI_files"
                if roi_dir.exists():
                    for roi_file in roi_dir.glob("*.txt"):
                        # Associate ROI files with data files
                        # Implementation depends on naming convention
                        pass
                
                if condition.files:  # Only add if it has data files
                    experiment.add_condition(condition)

        self.current_project = experiment
        self.project_type = ProjectType.EXPERIMENT
        self.project_modified = True
        
        self.logger.info(f"Created experiment from directory: {len(experiment.conditions)} conditions")
        return experiment

    def _determine_file_type(self, file_path: Path) -> str:
        """Determine file type based on name patterns and content."""
        name = file_path.name.lower()
        
        if 'tracks' in name or 'trajectory' in name:
            return 'trajectories'
        elif 'locs' in name or 'localization' in name:
            return 'localizations'
        elif file_path.suffix.lower() in ['.tif', '.tiff']:
            return 'raw_image'
        elif 'results' in name or 'analysis' in name:
            return 'analysis_results'
        else:
            return 'unknown'

    # =================================================================
    # Project Loading/Saving
    # =================================================================

    def load_project(self, project_path: str) -> bool:
        """Load a project from file (supports both legacy and enhanced formats)."""
        project_path = Path(project_path)

        if not project_path.exists():
            self.logger.error(f"Project file not found: {project_path}")
            return False

        try:
            project_dict = None
            
            if project_path.suffix.lower() == '.ptproj':
                # JSON format
                with open(project_path, 'r') as f:
                    project_dict = json.load(f)
            elif project_path.suffix.lower() == '.ptp':
                # Compressed format
                with zipfile.ZipFile(project_path, 'r') as zf:
                    with zf.open('project.json') as f:
                        project_dict = json.load(f)
            else:
                self.logger.error(f"Unsupported project format: {project_path.suffix}")
                return False

            # Determine project type and load accordingly
            version = project_dict.get('version', '1.0')
            
            if version.startswith('2.') or 'conditions' in project_dict:
                # Enhanced project format
                self.current_project = ExperimentInfo(**project_dict)
                self.project_type = (ProjectType.CONDITION if len(self.current_project.conditions) == 1 
                                   else ProjectType.EXPERIMENT)
            else:
                # Legacy project format
                self.current_project = ProjectInfo(**project_dict)
                self.project_type = ProjectType.SINGLE_FILE

            self.current_project_path = str(project_path)
            self.project_modified = False

            # Update modification date
            self.current_project.modified_date = datetime.now().isoformat()

            self.projectLoaded.emit(self.current_project_path)
            self.logger.info(f"Loaded project: {project_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading project: {e}")
            return False

    def save_project(self, project_path: Optional[str] = None) -> bool:
        """Save the current project."""
        if self.current_project is None:
            self.logger.warning("No project to save")
            return False

        # Use current path if none specified
        if project_path is None:
            project_path = self.current_project_path

        if project_path is None:
            self.logger.error("No project path specified")
            return False

        project_path = Path(project_path)

        try:
            # Update modification date
            self.current_project.modified_date = datetime.now().isoformat()

            # Convert to dictionary
            project_dict = asdict(self.current_project)

            if project_path.suffix.lower() == '.ptproj':
                # JSON format
                project_path.parent.mkdir(parents=True, exist_ok=True)
                with open(project_path, 'w') as f:
                    json.dump(project_dict, f, indent=2)

            elif project_path.suffix.lower() == '.ptp':
                # Compressed format with optional data archiving
                project_path.parent.mkdir(parents=True, exist_ok=True)
                
                with zipfile.ZipFile(project_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    # Save project info
                    zf.writestr('project.json', json.dumps(project_dict, indent=2))
                    
                    # Optionally archive data files
                    if isinstance(self.current_project, ExperimentInfo):
                        self._archive_experiment_data(zf, self.current_project)

            else:
                # Default to JSON format
                project_path = project_path.with_suffix('.ptproj')
                with open(project_path, 'w') as f:
                    json.dump(project_dict, f, indent=2)

            self.current_project_path = str(project_path)
            self.project_modified = False

            self.projectSaved.emit(self.current_project_path)
            self.logger.info(f"Saved project: {project_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving project: {e}")
            return False

    def _archive_experiment_data(self, zip_file: zipfile.ZipFile, 
                                experiment: ExperimentInfo):
        """Archive experiment data files in the project zip."""
        for condition in experiment.conditions:
            condition_dir = f"data/{condition.name}/"
            
            for file_info in condition.files:
                if Path(file_info.file_path).exists():
                    # Add file to archive with organized structure
                    archive_name = condition_dir + Path(file_info.file_path).name
                    zip_file.write(file_info.file_path, archive_name)

    # =================================================================
    # Experiment Management
    # =================================================================

    def add_condition(self, name: str, description: str = "") -> bool:
        """Add a new condition to the current experiment."""
        if not isinstance(self.current_project, ExperimentInfo):
            self.logger.error("Can only add conditions to experiment projects")
            return False

        condition = ConditionInfo(name=name, description=description)
        self.current_project.add_condition(condition)
        self.project_modified = True
        return True

    def add_files_to_condition(self, condition_name: str, 
                              file_paths: List[str]) -> bool:
        """Add files to a specific condition."""
        if not isinstance(self.current_project, ExperimentInfo):
            return False

        condition = self.current_project.get_condition(condition_name)
        if condition is None:
            return False

        for file_path in file_paths:
            file_type = self._determine_file_type(Path(file_path))
            condition.add_file(file_path, file_type)

        self.project_modified = True
        return True

    def set_comparison_groups(self, groups: List[List[str]]):
        """Set groups of conditions for statistical comparison."""
        if isinstance(self.current_project, ExperimentInfo):
            self.current_project.comparison_groups = groups
            self.project_modified = True

    # =================================================================
    # Batch Analysis
    # =================================================================

    def run_batch_analysis(self) -> bool:
        """Start batch analysis for the current experiment."""
        if not isinstance(self.current_project, ExperimentInfo):
            self.logger.error("Batch analysis only available for experiment projects")
            return False

        if self.analysis_engine is None:
            self.logger.error("Analysis engine not set")
            return False

        if self.batch_worker and self.batch_worker.isRunning():
            self.logger.warning("Batch analysis already running")
            return False

        # Create and start batch worker
        self.batch_worker = BatchAnalysisWorker(
            self.current_project, self.analysis_engine
        )
        
        # Connect signals
        self.batch_worker.progressUpdate.connect(self.batchAnalysisProgress)
        self.batch_worker.conditionCompleted.connect(self.conditionCompleted)
        self.batch_worker.analysisCompleted.connect(self.batchAnalysisCompleted)
        self.batch_worker.errorOccurred.connect(self._on_batch_error)

        self.batch_worker.start()
        self.batchAnalysisStarted.emit(self.current_project.name)
        return True

    def stop_batch_analysis(self):
        """Stop the current batch analysis."""
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.stop()
            self.batch_worker.wait()

    @pyqtSlot(str, str)
    def _on_batch_error(self, condition_name: str, error_message: str):
        """Handle batch analysis errors."""
        self.logger.error(f"Batch analysis error in {condition_name}: {error_message}")

    # =================================================================
    # Export and Reporting
    # =================================================================

    def export_experiment_report(self, output_path: str) -> bool:
        """Export a comprehensive experiment report."""
        if not isinstance(self.current_project, ExperimentInfo):
            return False

        try:
            report_lines = [
                f"Experiment Report: {self.current_project.name}",
                "=" * 60,
                "",
                f"Description: {self.current_project.description}",
                f"Experiment Type: {self.current_project.experiment_type}",
                f"Created: {self.current_project.created_date}",
                f"Modified: {self.current_project.modified_date}",
                "",
                f"Pixel Size: {self.current_project.pixel_size} nm",
                f"Frame Rate: {self.current_project.frame_rate} Hz",
                "",
                f"Analysis Pipeline: {', '.join(self.current_project.analysis_pipeline)}",
                "",
                "Conditions Overview:",
                "-" * 40,
            ]

            for condition in self.current_project.conditions:
                report_lines.extend([
                    f"\nCondition: {condition.name}",
                    f"  Description: {condition.description}",
                    f"  Files: {len(condition.files)}",
                    f"  Status: {condition.analysis_status.value}",
                ])

                if condition.summary_statistics:
                    report_lines.append("  Summary Statistics:")
                    for key, value in condition.summary_statistics.items():
                        report_lines.append(f"    {key}: {value}")

            # Add cross-condition results if available
            if self.current_project.cross_condition_results:
                report_lines.extend([
                    "",
                    "Cross-Condition Analysis:",
                    "-" * 40,
                ])
                # Add cross-condition summary

            if self.current_project.notes:
                report_lines.extend([
                    "",
                    "Notes:",
                    "-" * 20,
                    self.current_project.notes
                ])

            # Write report
            with open(output_path, 'w') as f:
                f.write('\n'.join(report_lines))

            self.logger.info(f"Experiment report exported to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting experiment report: {e}")
            return False

    def export_condition_data(self, condition_name: str, output_dir: str, 
                             formats: List[str] = None) -> bool:
        """Export all data for a specific condition."""
        if not isinstance(self.current_project, ExperimentInfo):
            return False

        condition = self.current_project.get_condition(condition_name)
        if condition is None:
            return False

        if formats is None:
            formats = ['csv']

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Export file list
            file_data = []
            for file_info in condition.files:
                file_data.append({
                    'file_path': file_info.file_path,
                    'file_type': file_info.file_type,
                    'analysis_status': file_info.analysis_status.value,
                    'file_size': file_info.file_size
                })

            df = pd.DataFrame(file_data)
            
            for fmt in formats:
                if fmt == 'csv':
                    df.to_csv(output_path / f"{condition_name}_files.csv", index=False)
                elif fmt == 'excel':
                    df.to_excel(output_path / f"{condition_name}_files.xlsx", index=False)

            return True

        except Exception as e:
            self.logger.error(f"Error exporting condition data: {e}")
            return False

    # =================================================================
    # Utility Methods
    # =================================================================

    def get_project_summary(self) -> Dict[str, Any]:
        """Get a summary of the current project."""
        if self.current_project is None:
            return {}

        if isinstance(self.current_project, ExperimentInfo):
            return {
                'type': 'experiment',
                'name': self.current_project.name,
                'n_conditions': len(self.current_project.conditions),
                'n_total_files': len(self.current_project.get_all_files()),
                'analysis_pipeline': self.current_project.analysis_pipeline,
                'base_output_directory': self.current_project.base_output_directory
            }
        else:
            return {
                'type': 'single_file',
                'name': self.current_project.name,
                'n_files': len(self.current_project.data_files),
                'experiment_type': self.current_project.experiment_type
            }

    def close_project(self):
        """Close the current project."""
        # Stop any running batch analysis
        self.stop_batch_analysis()

        self.current_project = None
        self.current_project_path = None
        self.project_modified = False
        self.project_type = ProjectType.SINGLE_FILE

        self.projectClosed.emit()
        self.logger.info("Project closed")

    # Legacy compatibility methods
    def new_project(self, name: str = "Untitled Project", 
                   description: str = "") -> ProjectInfo:
        """Legacy method for creating single-file projects."""
        return self.new_single_file_project(name, description)

    def add_data_file(self, file_path: str):
        """Legacy method for adding data files."""
        if isinstance(self.current_project, ProjectInfo):
            file_path = str(Path(file_path).absolute())
            if file_path not in self.current_project.data_files:
                self.current_project.data_files.append(file_path)
                self.project_modified = True

    def get_parameters(self) -> Dict[str, Any]:
        """Get analysis parameters from the project."""
        if isinstance(self.current_project, ProjectInfo):
            return self.current_project.parameters.copy()
        elif isinstance(self.current_project, ExperimentInfo):
            return self.current_project.global_parameters.copy()
        return {}

    def set_parameters(self, parameters: Dict[str, Any]):
        """Set analysis parameters for the project."""
        if isinstance(self.current_project, ProjectInfo):
            self.current_project.parameters = parameters.copy()
        elif isinstance(self.current_project, ExperimentInfo):
            self.current_project.global_parameters = parameters.copy()
        self.project_modified = True