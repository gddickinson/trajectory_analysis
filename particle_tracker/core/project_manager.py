#!/usr/bin/env python3
"""
Utility Modules
===============

Project management.
"""

import os
import json
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import pickle
import zipfile

from PyQt6.QtCore import QObject, pyqtSignal, QSettings



# ============================================================================
# PROJECT MANAGER
# ============================================================================

@dataclass
class ProjectInfo:
    """Information about a project."""

    name: str
    description: str = ""
    created_date: str = ""
    modified_date: str = ""
    version: str = "1.0"

    # Data references
    data_files: List[str] = None
    analysis_results: List[str] = None

    # Analysis parameters
    parameters: Dict[str, Any] = None

    # Metadata
    pixel_size: float = 108.0
    frame_rate: float = 10.0
    experiment_type: str = ""
    notes: str = ""

    def __post_init__(self):
        if self.data_files is None:
            self.data_files = []
        if self.analysis_results is None:
            self.analysis_results = []
        if self.parameters is None:
            self.parameters = {}
        if not self.created_date:
            self.created_date = datetime.now().isoformat()


class ProjectManager(QObject):
    """Manages project files and settings."""

    projectLoaded = pyqtSignal(str)  # project_path
    projectSaved = pyqtSignal(str)   # project_path
    projectClosed = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(__name__)

        # Current project
        self.current_project: Optional[ProjectInfo] = None
        self.current_project_path: Optional[str] = None
        self.project_modified: bool = False

        self.logger.info("Project manager initialized")

    def new_project(self, name: str = "Untitled Project",
                   description: str = "") -> ProjectInfo:
        """Create a new project."""

        self.current_project = ProjectInfo(
            name=name,
            description=description
        )
        self.current_project_path = None
        self.project_modified = True

        self.logger.info(f"Created new project: {name}")
        return self.current_project

    def load_project(self, project_path: str) -> bool:
        """Load a project from file."""

        project_path = Path(project_path)

        if not project_path.exists():
            self.logger.error(f"Project file not found: {project_path}")
            return False

        try:
            if project_path.suffix.lower() == '.ptproj':
                # JSON format
                with open(project_path, 'r') as f:
                    project_dict = json.load(f)

                self.current_project = ProjectInfo(**project_dict)

            elif project_path.suffix.lower() == '.ptp':
                # Compressed format
                with zipfile.ZipFile(project_path, 'r') as zf:
                    with zf.open('project.json') as f:
                        project_dict = json.load(f)

                    self.current_project = ProjectInfo(**project_dict)

            else:
                self.logger.error(f"Unsupported project format: {project_path.suffix}")
                return False

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
                # Compressed format
                project_path.parent.mkdir(parents=True, exist_ok=True)

                with zipfile.ZipFile(project_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    # Save project info
                    zf.writestr('project.json', json.dumps(project_dict, indent=2))

                    # TODO: Add data files to archive if requested

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

    def close_project(self):
        """Close the current project."""

        self.current_project = None
        self.current_project_path = None
        self.project_modified = False

        self.projectClosed.emit()
        self.logger.info("Project closed")

    def add_data_file(self, file_path: str):
        """Add a data file to the current project."""

        if self.current_project is None:
            return

        file_path = str(Path(file_path).absolute())

        if file_path not in self.current_project.data_files:
            self.current_project.data_files.append(file_path)
            self.project_modified = True

    def add_analysis_result(self, result_path: str):
        """Add an analysis result to the current project."""

        if self.current_project is None:
            return

        result_path = str(Path(result_path).absolute())

        if result_path not in self.current_project.analysis_results:
            self.current_project.analysis_results.append(result_path)
            self.project_modified = True

    def set_parameters(self, parameters: Dict[str, Any]):
        """Set analysis parameters for the project."""

        if self.current_project is None:
            return

        self.current_project.parameters = parameters.copy()
        self.project_modified = True

    def get_parameters(self) -> Dict[str, Any]:
        """Get analysis parameters from the project."""

        if self.current_project is None:
            return {}

        return self.current_project.parameters.copy()

    def is_modified(self) -> bool:
        """Check if the project has been modified."""
        return self.project_modified

    def get_project_info(self) -> Optional[ProjectInfo]:
        """Get current project information."""
        return self.current_project

    def export_project_report(self, output_path: str) -> bool:
        """Export a project report."""

        if self.current_project is None:
            return False

        try:
            report_lines = [
                f"Project Report: {self.current_project.name}",
                "=" * 50,
                "",
                f"Description: {self.current_project.description}",
                f"Created: {self.current_project.created_date}",
                f"Modified: {self.current_project.modified_date}",
                f"Experiment Type: {self.current_project.experiment_type}",
                "",
                f"Pixel Size: {self.current_project.pixel_size} nm",
                f"Frame Rate: {self.current_project.frame_rate} Hz",
                "",
                "Data Files:",
                "-" * 20,
            ]

            for file_path in self.current_project.data_files:
                report_lines.append(f"  - {Path(file_path).name}")

            report_lines.extend([
                "",
                "Analysis Results:",
                "-" * 20,
            ])

            for result_path in self.current_project.analysis_results:
                report_lines.append(f"  - {Path(result_path).name}")

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

            self.logger.info(f"Project report exported to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting project report: {e}")
            return False

