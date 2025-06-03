"""
Core Components Package
======================

Contains the core functionality for the particle tracking application:
- DataManager: Handles data loading, saving, and management
- AnalysisEngine: Coordinates analysis workflows
- ProjectManager: Manages projects and settings
"""

from .data_manager import DataManager, DataType, DataInfo
from .analysis_engine import AnalysisEngine, AnalysisParameters, AnalysisStep, AnalysisWorker
from .project_manager import ProjectManager, ProjectInfo

__all__ = [
    "DataManager",
    "DataType",
    "DataInfo",
    "AnalysisEngine",
    "AnalysisParameters",
    "AnalysisStep",
    "AnalysisWorker",
    "ProjectManager",
    "ProjectInfo",
]

# Core package version (can be different from main package)
__core_version__ = "1.0.0"

def get_core_info():
    """Get information about core components."""
    return {
        "version": __core_version__,
        "components": {
            "DataManager": "Handles data operations",
            "AnalysisEngine": "Coordinates analysis workflows",
            "ProjectManager": "Manages projects and settings"
        }
    }