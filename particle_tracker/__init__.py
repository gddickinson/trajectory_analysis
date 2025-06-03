"""
Particle Tracking Application
============================

A comprehensive application for analyzing particle trajectories from microscopy data.

Features:
- Particle detection using multiple algorithms
- Trajectory linking and filtering
- Feature calculation (radius of gyration, asymmetry, etc.)
- Classification using SVM or threshold methods
- Interactive visualization
- Project management
- Batch processing capabilities

Example usage:
    from particle_tracker import ParticleTrackingApp

    app = ParticleTrackingApp([])
    app.exec()
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main application class
try:
    from .app import ParticleTrackingApp
except ImportError:
    # Handle case where GUI dependencies are not available
    ParticleTrackingApp = None
    import warnings
    warnings.warn("GUI components not available. Install PyQt6 for full functionality.")

# Import core components
from .core.data_manager import DataManager, DataType, DataInfo
from .core.analysis_engine import AnalysisEngine, AnalysisParameters, AnalysisStep
from .core.project_manager import ProjectManager, ProjectInfo

# Import analysis components
from .analysis.detection import ParticleDetector
from .analysis.linking import ParticleLinker
from .analysis.features import FeatureCalculator
from .analysis.classification import TrajectoryClassifier

# Import utilities
from .utils.config_manager import ConfigManager, ApplicationConfig
from .utils.logging_config import setup_logging

# Define what gets imported with "from particle_tracker import *"
__all__ = [
    # Main application
    "ParticleTrackingApp",

    # Core components
    "DataManager",
    "DataType",
    "DataInfo",
    "AnalysisEngine",
    "AnalysisParameters",
    "AnalysisStep",
    "ProjectManager",
    "ProjectInfo",

    # Analysis components
    "ParticleDetector",
    "ParticleLinker",
    "FeatureCalculator",
    "TrajectoryClassifier",

    # Utilities
    "ConfigManager",
    "ApplicationConfig",
    "setup_logging",

    # Version info
    "__version__",
    "__author__",
    "__email__",
]

# Package metadata
__title__ = "particle-tracker"
__description__ = "A comprehensive application for analyzing particle trajectories from microscopy data"
__url__ = "https://github.com/yourusername/particle-tracker"
__license__ = "MIT"
__copyright__ = "Copyright 2024 Your Name"

# Minimum required versions for key dependencies
__requires__ = [
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "scipy>=1.7.0",
    "scikit-learn>=1.0.0",
    "scikit-image>=0.18.0",
]

# Optional dependencies
__extras__ = {
    "gui": ["PyQt6>=6.0.0", "pyqtgraph>=0.12.0"],
    "tracking": ["trackpy>=0.5.0"],
    "visualization": ["matplotlib>=3.4.0", "seaborn>=0.11.0"],
}

def get_version():
    """Get the current version of the package."""
    return __version__

def check_dependencies():
    """Check if required dependencies are available."""
    missing = []

    try:
        import numpy
        if numpy.__version__ < "1.21.0":
            missing.append(f"numpy>={numpy.__version__} (found {numpy.__version__})")
    except ImportError:
        missing.append("numpy>=1.21.0")

    try:
        import pandas
        if pandas.__version__ < "1.3.0":
            missing.append(f"pandas>={pandas.__version__} (found {pandas.__version__})")
    except ImportError:
        missing.append("pandas>=1.3.0")

    try:
        import scipy
    except ImportError:
        missing.append("scipy>=1.7.0")

    try:
        import sklearn
    except ImportError:
        missing.append("scikit-learn>=1.0.0")

    try:
        import skimage
    except ImportError:
        missing.append("scikit-image>=0.18.0")

    if missing:
        import warnings
        warnings.warn(f"Missing required dependencies: {', '.join(missing)}")
        return False

    return True

# Check dependencies on import
check_dependencies()