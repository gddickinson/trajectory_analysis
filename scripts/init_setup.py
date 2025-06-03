#!/usr/bin/env python3
"""
All __init__.py Files for Particle Tracking Application
======================================================

This file contains all the __init__.py files needed for the complete package structure.
"""

# ============================================================================
# 1. ROOT PACKAGE: particle_tracker/__init__.py
# ============================================================================

MAIN_PACKAGE_INIT = '''"""
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
'''

# ============================================================================
# 2. CORE PACKAGE: particle_tracker/core/__init__.py
# ============================================================================

CORE_PACKAGE_INIT = '''"""
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
'''

# ============================================================================
# 3. ANALYSIS PACKAGE: particle_tracker/analysis/__init__.py
# ============================================================================

ANALYSIS_PACKAGE_INIT = '''"""
Analysis Components Package
==========================

Contains all analysis algorithms and methods:
- Detection: Particle detection algorithms
- Linking: Trajectory linking methods
- Features: Feature calculation functions
- Classification: Trajectory classification methods
"""

from .detection import (
    ParticleDetector,
    DetectionMethod,
    ThresholdDetection,
    LoGDetection,
    TrackpyDetection
)

from .linking import (
    ParticleLinker,
    LinkingMethod,
    NearestNeighborLinking,
    TrackpyLinking
)

from .features import FeatureCalculator

from .classification import TrajectoryClassifier

__all__ = [
    # Detection
    "ParticleDetector",
    "DetectionMethod",
    "ThresholdDetection",
    "LoGDetection",
    "TrackpyDetection",

    # Linking
    "ParticleLinker",
    "LinkingMethod",
    "NearestNeighborLinking",
    "TrackpyLinking",

    # Features
    "FeatureCalculator",

    # Classification
    "TrajectoryClassifier",
]

# Analysis package metadata
__analysis_version__ = "1.0.0"

def get_available_methods():
    """Get information about available analysis methods."""
    return {
        "detection": {
            "threshold": "Simple threshold-based detection",
            "log": "Laplacian of Gaussian blob detection",
            "trackpy": "Trackpy feature detection (requires trackpy)",
        },
        "linking": {
            "nearest_neighbor": "Simple nearest neighbor linking",
            "trackpy": "Trackpy linking with gap filling (requires trackpy)",
        },
        "features": [
            "radius_gyration", "asymmetry", "skewness", "kurtosis",
            "fractal_dimension", "net_displacement", "straightness",
            "diffusion_coefficient", "velocity", "nearest_neighbors"
        ],
        "classification": {
            "svm": "Support Vector Machine classification",
            "threshold": "Simple threshold-based classification",
        }
    }

def check_optional_dependencies():
    """Check for optional analysis dependencies."""
    available = {}

    try:
        import trackpy
        available["trackpy"] = trackpy.__version__
    except ImportError:
        available["trackpy"] = None

    try:
        import cv2
        available["opencv"] = cv2.__version__
    except ImportError:
        available["opencv"] = None

    return available
'''

# ============================================================================
# 4. GUI PACKAGE: particle_tracker/gui/__init__.py
# ============================================================================

GUI_PACKAGE_INIT = '''"""
GUI Components Package
=====================

Contains all graphical user interface components:
- MainWindow: Main application window
- VisualizationWidget: Image and trajectory visualization
- ParameterPanels: Analysis parameter input controls
- DataBrowser: Data management interface
- AnalysisControl: Analysis workflow controls
- LoggingWidget: Log message display
"""

# Import GUI components with error handling
GUI_AVAILABLE = True
gui_import_error = None

try:
    from .main_window import MainWindow
    from .visualization_widget import VisualizationWidget
    from .parameter_panels import (
        ParameterPanelManager,
        DetectionParametersWidget,
        LinkingParametersWidget,
        FeatureParametersWidget,
        ClassificationParametersWidget
    )
    from .data_browser import DataBrowserWidget
    from .analysis_control import AnalysisControlWidget
    from .logging_widget import LoggingWidget, TextEditLogHandler

except ImportError as e:
    GUI_AVAILABLE = False
    gui_import_error = str(e)

    # Create dummy classes for when GUI is not available
    class MainWindow:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"GUI not available: {gui_import_error}")

    class VisualizationWidget:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"GUI not available: {gui_import_error}")

    class ParameterPanelManager:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"GUI not available: {gui_import_error}")

    # Set other classes to None
    DetectionParametersWidget = None
    LinkingParametersWidget = None
    FeatureParametersWidget = None
    ClassificationParametersWidget = None
    DataBrowserWidget = None
    AnalysisControlWidget = None
    LoggingWidget = None
    TextEditLogHandler = None

__all__ = [
    "MainWindow",
    "VisualizationWidget",
    "ParameterPanelManager",
    "DetectionParametersWidget",
    "LinkingParametersWidget",
    "FeatureParametersWidget",
    "ClassificationParametersWidget",
    "DataBrowserWidget",
    "AnalysisControlWidget",
    "LoggingWidget",
    "TextEditLogHandler",
    "GUI_AVAILABLE",
]

# GUI package metadata
__gui_version__ = "1.0.0"

def check_gui_dependencies():
    """Check if GUI dependencies are available."""
    missing = []

    try:
        import PyQt6
    except ImportError:
        missing.append("PyQt6")

    try:
        import pyqtgraph
    except ImportError:
        missing.append("pyqtgraph")

    return {
        "available": len(missing) == 0,
        "missing": missing,
        "error": gui_import_error
    }

def get_gui_info():
    """Get information about GUI components."""
    deps = check_gui_dependencies()

    return {
        "version": __gui_version__,
        "available": GUI_AVAILABLE,
        "dependencies": deps,
        "components": {
            "MainWindow": "Main application window",
            "VisualizationWidget": "Image and trajectory display",
            "ParameterPanels": "Analysis parameter controls",
            "DataBrowser": "Data management interface",
            "AnalysisControl": "Analysis workflow controls",
            "LoggingWidget": "Log message display"
        } if GUI_AVAILABLE else {}
    }
'''

# ============================================================================
# 5. UTILS PACKAGE: particle_tracker/utils/__init__.py
# ============================================================================

UTILS_PACKAGE_INIT = '''"""
Utilities Package
================

Contains utility functions and helper classes:
- ConfigManager: Application configuration management
- LoggingConfig: Logging setup and configuration
- FileUtils: File handling utilities
- PerformanceMonitor: Performance monitoring tools
"""

from .config_manager import ConfigManager, ApplicationConfig
from .logging_config import setup_logging
from .file_utils import (
    ensure_directory,
    get_app_data_directory,
    get_temp_directory,
    format_file_size,
    format_duration,
    backup_file,
    PerformanceMonitor,
    performance_monitor
)

# Import custom exceptions
from .file_utils import (
    ParticleTrackerError,
    DataLoadError,
    AnalysisError,
    ProjectError
)

__all__ = [
    # Configuration
    "ConfigManager",
    "ApplicationConfig",

    # Logging
    "setup_logging",

    # File utilities
    "ensure_directory",
    "get_app_data_directory",
    "get_temp_directory",
    "format_file_size",
    "format_duration",
    "backup_file",
    "PerformanceMonitor",
    "performance_monitor",

    # Exceptions
    "ParticleTrackerError",
    "DataLoadError",
    "AnalysisError",
    "ProjectError",
]

# Utils package metadata
__utils_version__ = "1.0.0"

def get_utils_info():
    """Get information about utility components."""
    return {
        "version": __utils_version__,
        "components": {
            "ConfigManager": "Application configuration management",
            "LoggingConfig": "Logging setup and configuration",
            "FileUtils": "File handling utilities",
            "PerformanceMonitor": "Performance monitoring tools"
        }
    }

def get_system_info():
    """Get system information for debugging."""
    import sys
    import platform

    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture(),
        "processor": platform.processor(),
    }

    # Add memory info if available
    try:
        import psutil
        memory = psutil.virtual_memory()
        info["memory_total_gb"] = memory.total / (1024**3)
        info["memory_available_gb"] = memory.available / (1024**3)
    except ImportError:
        info["memory_info"] = "psutil not available"

    return info
'''

# ============================================================================
# 6. RESOURCES PACKAGE: particle_tracker/resources/__init__.py
# ============================================================================

RESOURCES_PACKAGE_INIT = '''"""
Resources Package
================

Contains application resources:
- Icons: Application and UI icons
- Example Data: Sample datasets for testing
- Themes: UI themes and stylesheets
- Documentation: Embedded help files
"""

import os
from pathlib import Path

# Get the resources directory path
RESOURCES_DIR = Path(__file__).parent

# Define resource subdirectories
ICONS_DIR = RESOURCES_DIR / "icons"
EXAMPLE_DATA_DIR = RESOURCES_DIR / "example_data"
THEMES_DIR = RESOURCES_DIR / "themes"
DOCS_DIR = RESOURCES_DIR / "docs"

def get_resource_path(resource_type: str, filename: str) -> Path:
    """Get the full path to a resource file.

    Args:
        resource_type: Type of resource ("icons", "example_data", "themes", "docs")
        filename: Name of the resource file

    Returns:
        Path to the resource file
    """
    resource_dirs = {
        "icons": ICONS_DIR,
        "example_data": EXAMPLE_DATA_DIR,
        "themes": THEMES_DIR,
        "docs": DOCS_DIR
    }

    if resource_type not in resource_dirs:
        raise ValueError(f"Unknown resource type: {resource_type}")

    resource_dir = resource_dirs[resource_type]
    resource_path = resource_dir / filename

    return resource_path

def get_icon_path(icon_name: str) -> Path:
    """Get path to an icon file."""
    return get_resource_path("icons", icon_name)

def get_example_data_path(data_name: str) -> Path:
    """Get path to an example data file."""
    return get_resource_path("example_data", data_name)

def get_theme_path(theme_name: str) -> Path:
    """Get path to a theme file."""
    return get_resource_path("themes", theme_name)

def list_resources(resource_type: str) -> list:
    """List all available resources of a given type.

    Args:
        resource_type: Type of resource to list

    Returns:
        List of available resource files
    """
    resource_dirs = {
        "icons": ICONS_DIR,
        "example_data": EXAMPLE_DATA_DIR,
        "themes": THEMES_DIR,
        "docs": DOCS_DIR
    }

    if resource_type not in resource_dirs:
        return []

    resource_dir = resource_dirs[resource_type]

    if not resource_dir.exists():
        return []

    return [f.name for f in resource_dir.iterdir() if f.is_file()]

def ensure_resource_dirs():
    """Ensure all resource directories exist."""
    for resource_dir in [ICONS_DIR, EXAMPLE_DATA_DIR, THEMES_DIR, DOCS_DIR]:
        resource_dir.mkdir(parents=True, exist_ok=True)

# Create resource directories on import
ensure_resource_dirs()

__all__ = [
    "RESOURCES_DIR",
    "ICONS_DIR",
    "EXAMPLE_DATA_DIR",
    "THEMES_DIR",
    "DOCS_DIR",
    "get_resource_path",
    "get_icon_path",
    "get_example_data_path",
    "get_theme_path",
    "list_resources",
    "ensure_resource_dirs",
]

# Resources package metadata
__resources_version__ = "1.0.0"

def get_resources_info():
    """Get information about available resources."""
    return {
        "version": __resources_version__,
        "directories": {
            "icons": str(ICONS_DIR),
            "example_data": str(EXAMPLE_DATA_DIR),
            "themes": str(THEMES_DIR),
            "docs": str(DOCS_DIR)
        },
        "available_resources": {
            "icons": list_resources("icons"),
            "example_data": list_resources("example_data"),
            "themes": list_resources("themes"),
            "docs": list_resources("docs")
        }
    }
'''

# ============================================================================
# 7. TESTS PACKAGE: tests/__init__.py
# ============================================================================

TESTS_PACKAGE_INIT = '''"""
Tests Package
============

Contains unit tests and integration tests for the particle tracking application.

Test structure:
- test_data_manager.py: Tests for data management functionality
- test_analysis_engine.py: Tests for analysis coordination
- test_detection.py: Tests for particle detection algorithms
- test_linking.py: Tests for trajectory linking methods
- test_features.py: Tests for feature calculation
- test_classification.py: Tests for trajectory classification
- test_gui.py: Tests for GUI components (requires pytest-qt)
- test_integration.py: Full workflow integration tests
"""

import sys
import os
from pathlib import Path

# Add the parent directory to Python path for testing
TEST_DIR = Path(__file__).parent
PROJECT_DIR = TEST_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

# Test configuration
TEST_DATA_DIR = TEST_DIR / "test_data"
TEST_OUTPUT_DIR = TEST_DIR / "test_output"

def setup_test_environment():
    """Set up the test environment."""
    # Create test directories
    TEST_DATA_DIR.mkdir(exist_ok=True)
    TEST_OUTPUT_DIR.mkdir(exist_ok=True)

    # Generate test data if needed
    generate_test_data()

def generate_test_data():
    """Generate test data for unit tests."""
    import numpy as np
    import pandas as pd

    # Generate synthetic TIRF data
    if not (TEST_DATA_DIR / "test_movie.npy").exists():
        # Create a small test movie
        test_movie = np.random.randint(50, 200, (10, 64, 64), dtype=np.uint16)

        # Add some synthetic particles
        for frame in range(10):
            for i in range(5):  # 5 particles per frame
                x = np.random.randint(10, 54)
                y = np.random.randint(10, 54)
                intensity = np.random.randint(500, 1000)

                # Add Gaussian spot
                sigma = 1.5
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        if 0 <= y+dy < 64 and 0 <= x+dx < 64:
                            dist_sq = dx*dx + dy*dy
                            value = intensity * np.exp(-dist_sq / (2 * sigma*sigma))
                            test_movie[frame, y+dy, x+dx] += int(value)

        np.save(TEST_DATA_DIR / "test_movie.npy", test_movie)

    # Generate synthetic trajectory data
    if not (TEST_DATA_DIR / "test_trajectories.csv").exists():
        trajectories = []

        for track_id in range(10):
            n_points = np.random.randint(5, 20)
            start_x = np.random.uniform(10, 54)
            start_y = np.random.uniform(10, 54)

            for i in range(n_points):
                trajectories.append({
                    'track_number': track_id,
                    'frame': i,
                    'x': start_x + np.random.normal(0, 0.5),
                    'y': start_y + np.random.normal(0, 0.5),
                    'intensity': np.random.uniform(500, 1000)
                })

        df = pd.DataFrame(trajectories)
        df.to_csv(TEST_DATA_DIR / "test_trajectories.csv", index=False)

def cleanup_test_environment():
    """Clean up test environment."""
    import shutil

    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)

# Test utilities
def get_test_data_path(filename: str) -> Path:
    """Get path to test data file."""
    return TEST_DATA_DIR / filename

def get_test_output_path(filename: str) -> Path:
    """Get path for test output file."""
    TEST_OUTPUT_DIR.mkdir(exist_ok=True)
    return TEST_OUTPUT_DIR / filename

# Test fixtures and helpers
class TestDataGenerator:
    """Helper class for generating test data."""

    @staticmethod
    def create_synthetic_movie(n_frames=10, height=64, width=64, n_particles=5):
        """Create a synthetic microscopy movie."""
        movie = np.random.randint(50, 150, (n_frames, height, width), dtype=np.uint16)

        # Add particles with realistic trajectories
        for particle_id in range(n_particles):
            start_x = np.random.uniform(10, width-10)
            start_y = np.random.uniform(10, height-10)

            # Random walk with drift
            positions = [(start_x, start_y)]

            for frame in range(1, n_frames):
                prev_x, prev_y = positions[-1]

                # Diffusion step
                dx = np.random.normal(0, 1.0)
                dy = np.random.normal(0, 1.0)

                new_x = np.clip(prev_x + dx, 5, width-5)
                new_y = np.clip(prev_y + dy, 5, height-5)

                positions.append((new_x, new_y))

            # Add particles to movie
            intensity = np.random.uniform(300, 800)
            sigma = 1.5

            for frame, (x, y) in enumerate(positions):
                # Skip some frames (blinking)
                if np.random.random() < 0.1:
                    continue

                # Add Gaussian spot
                x_int, y_int = int(round(x)), int(round(y))

                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        px = x_int + dx
                        py = y_int + dy

                        if 0 <= px < width and 0 <= py < height:
                            dist_sq = dx*dx + dy*dy
                            value = intensity * np.exp(-dist_sq / (2 * sigma*sigma))
                            movie[frame, py, px] += int(value)

        return movie

    @staticmethod
    def create_test_trajectories(n_tracks=10, min_length=5, max_length=20):
        """Create test trajectory data."""
        trajectories = []

        for track_id in range(n_tracks):
            n_points = np.random.randint(min_length, max_length+1)

            # Random starting position
            start_x = np.random.uniform(10, 90)
            start_y = np.random.uniform(10, 90)

            # Random trajectory parameters
            drift_x = np.random.normal(0, 0.1)
            drift_y = np.random.normal(0, 0.1)
            diffusion = np.random.uniform(0.5, 2.0)

            x, y = start_x, start_y

            for i in range(n_points):
                trajectories.append({
                    'track_number': track_id,
                    'frame': i,
                    'x': x,
                    'y': y,
                    'intensity': np.random.uniform(400, 900)
                })

                # Update position
                x += drift_x + np.random.normal(0, diffusion)
                y += drift_y + np.random.normal(0, diffusion)

        return pd.DataFrame(trajectories)

# Setup test environment on import
setup_test_environment()

__all__ = [
    "TEST_DIR",
    "PROJECT_DIR",
    "TEST_DATA_DIR",
    "TEST_OUTPUT_DIR",
    "setup_test_environment",
    "cleanup_test_environment",
    "generate_test_data",
    "get_test_data_path",
    "get_test_output_path",
    "TestDataGenerator",
]
'''

# ============================================================================
# SUMMARY OF ALL FILES
# ============================================================================

def create_all_init_files():
    """Create all __init__.py files in the correct directories."""

    init_files = {
        "particle_tracker/__init__.py": MAIN_PACKAGE_INIT,
        "particle_tracker/core/__init__.py": CORE_PACKAGE_INIT,
        "particle_tracker/analysis/__init__.py": ANALYSIS_PACKAGE_INIT,
        "particle_tracker/gui/__init__.py": GUI_PACKAGE_INIT,
        "particle_tracker/utils/__init__.py": UTILS_PACKAGE_INIT,
        "particle_tracker/resources/__init__.py": RESOURCES_PACKAGE_INIT,
        "tests/__init__.py": TESTS_PACKAGE_INIT,
    }

    print("Creating all __init__.py files...")

    for file_path, content in init_files.items():
        # Create directory if needed
        from pathlib import Path
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # Write file
        with open(file_path, 'w') as f:
            f.write(content.strip())

        print(f"Created: {file_path}")

    print("All __init__.py files created successfully!")

# Print summary
print("""
Summary of __init__.py files:

1. particle_tracker/__init__.py - Main package initialization
   - Imports all core classes and functions
   - Handles missing GUI dependencies gracefully
   - Provides version info and dependency checking

2. particle_tracker/core/__init__.py - Core components
   - DataManager, AnalysisEngine, ProjectManager
   - Core functionality imports

3. particle_tracker/analysis/__init__.py - Analysis methods
   - All detection, linking, feature, and classification classes
   - Method availability checking
   - Optional dependency handling

4. particle_tracker/gui/__init__.py - GUI components
   - All GUI widgets and windows
   - Graceful handling of missing PyQt6
   - GUI dependency checking

5. particle_tracker/utils/__init__.py - Utilities
   - Configuration, logging, file utilities
   - Performance monitoring
   - Custom exceptions

6. particle_tracker/resources/__init__.py - Resources
   - Path management for icons, themes, example data
   - Resource directory creation and management

7. tests/__init__.py - Test framework
   - Test environment setup
   - Test data generation
   - Test utilities and fixtures
""")

if __name__ == "__main__":
    create_all_init_files()
