"""
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