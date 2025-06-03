"""
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