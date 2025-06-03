#!/usr/bin/env python3
"""
# ============================================================================
# PARTICLE DETECTION MODULE
# ============================================================================
"""

import logging
import math
from typing import Optional, Dict, List, Any, Tuple, Union
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

# Import scipy with error handling
try:
    from scipy import stats, spatial, ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available. Some detection methods may not work.")

from sklearn.neighbors import KDTree
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics
import skimage.io as skio

# Import scikit-image with error handling
try:
    from skimage import filters, feature, segmentation, measure
    from skimage.morphology import disk, white_tophat
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. Some detection methods may not work.")

from pathlib import Path
from tqdm import tqdm


def safe_disk(radius):
    """Safe disk function that works without skimage."""
    if SKIMAGE_AVAILABLE:
        try:
            return disk(radius)
        except:
            pass

    # Fallback: return a simple circular structuring element
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    return x*x + y*y <= radius*radius


class DetectionMethod(ABC):
    """Abstract base class for particle detection methods."""

    @abstractmethod
    def detect(self, image: np.ndarray, **kwargs) -> pd.DataFrame:
        """Detect particles in an image.

        Args:
            image: Input image (2D or 3D for time series)
            **kwargs: Method-specific parameters

        Returns:
            DataFrame with columns: frame, x, y, intensity
        """
        pass


class ThresholdDetection(DetectionMethod):
    """Simple threshold-based particle detection."""

    def detect(self, image: np.ndarray, threshold: float = 3.0,
               sigma: float = 1.6, min_intensity: int = 100,
               max_intensity: int = 10000, **kwargs) -> pd.DataFrame:
        """Detect particles using threshold method."""

        if len(image.shape) == 2:
            # Single frame
            return self._detect_frame(image, 0, threshold, sigma, min_intensity, max_intensity)
        elif len(image.shape) == 3:
            # Time series
            results = []
            for frame_idx in range(image.shape[0]):
                frame_result = self._detect_frame(
                    image[frame_idx], frame_idx, threshold, sigma,
                    min_intensity, max_intensity
                )
                if len(frame_result) > 0:
                    results.append(frame_result)

            return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

    def _detect_frame(self, frame: np.ndarray, frame_idx: int, threshold: float,
                     sigma: float, min_intensity: int, max_intensity: int) -> pd.DataFrame:
        """Detect particles in a single frame."""

        # Apply Gaussian filter
        if SCIPY_AVAILABLE:
            filtered = ndimage.gaussian_filter(frame, sigma=sigma)
        elif SKIMAGE_AVAILABLE:
            filtered = filters.gaussian(frame, sigma=sigma)
        else:
            # Simple fallback - no filtering
            filtered = frame.astype(float)

        # Calculate threshold
        noise_std = np.std(filtered)
        thresh_value = np.mean(filtered) + threshold * noise_std

        # Apply threshold
        binary = filtered > thresh_value

        # Remove small objects using safe_disk
        if SCIPY_AVAILABLE:
            binary = ndimage.binary_opening(binary, structure=safe_disk(1))

        # Label connected components
        if SKIMAGE_AVAILABLE:
            labeled = measure.label(binary)
            regions = measure.regionprops(labeled, intensity_image=frame)
        elif SCIPY_AVAILABLE:
            labeled, num_features = ndimage.label(binary)
            # Create a simpler version without full regionprops
            regions = []
            for i in range(1, num_features + 1):
                mask = labeled == i
                if np.sum(mask) > 0:  # Only process non-empty regions
                    y_coords, x_coords = np.where(mask)
                    centroid_y = np.mean(y_coords)
                    centroid_x = np.mean(x_coords)
                    mean_intensity = np.mean(frame[mask])
                    area = np.sum(mask)

                    # Create a simple region object
                    class SimpleRegion:
                        def __init__(self, centroid, mean_intensity, area):
                            self.centroid = centroid
                            self.mean_intensity = mean_intensity
                            self.area = area

                    regions.append(SimpleRegion((centroid_y, centroid_x), mean_intensity, area))
        else:
            # No region analysis available
            return pd.DataFrame()

        # Convert to DataFrame
        detections = []
        for region in regions:
            if hasattr(region, 'mean_intensity'):
                intensity = region.mean_intensity
            else:
                intensity = 1000  # default value

            # Filter by intensity
            if min_intensity <= intensity <= max_intensity:
                if hasattr(region, 'centroid'):
                    centroid = region.centroid
                    y, x = centroid if len(centroid) == 2 else (centroid[0], centroid[1])
                else:
                    # Fallback if no centroid available
                    continue

                detections.append({
                    'frame': frame_idx,
                    'x': float(x),
                    'y': float(y),
                    'intensity': float(intensity),
                    'area': getattr(region, 'area', 1)
                })

        return pd.DataFrame(detections)


class LoGDetection(DetectionMethod):
    """Laplacian of Gaussian blob detection."""

    def detect(self, image: np.ndarray, sigma: float = 1.6,
               threshold: float = 0.1, **kwargs) -> pd.DataFrame:
        """Detect particles using LoG method."""

        if len(image.shape) == 2:
            return self._detect_frame_log(image, 0, sigma, threshold)
        elif len(image.shape) == 3:
            results = []
            for frame_idx in range(image.shape[0]):
                frame_result = self._detect_frame_log(image[frame_idx], frame_idx, sigma, threshold)
                if len(frame_result) > 0:
                    results.append(frame_result)

            return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def _detect_frame_log(self, frame: np.ndarray, frame_idx: int,
                         sigma: float, threshold: float) -> pd.DataFrame:
        """Detect particles using LoG in a single frame."""

        if not SKIMAGE_AVAILABLE:
            print("Warning: scikit-image required for LoG detection")
            return pd.DataFrame()

        # Apply LoG filter
        try:
            log_filtered = -filters.laplace(filters.gaussian(frame, sigma=sigma))
        except:
            # Fallback to scipy if available
            if SCIPY_AVAILABLE:
                gaussian = ndimage.gaussian_filter(frame, sigma=sigma)
                log_filtered = -ndimage.laplace(gaussian)
            else:
                return pd.DataFrame()

        # Find local maxima
        try:
            from skimage.feature import peak_local_maxima
            peaks = peak_local_maxima(log_filtered, min_distance=int(2*sigma),
                                    threshold_abs=threshold)
        except ImportError:
            # Fallback: simple peak detection
            peaks = self._simple_peak_detection(log_filtered, int(2*sigma), threshold)

        # Extract intensities
        detections = []
        if len(peaks) > 0 and len(peaks[0]) > 0:
            for i in range(len(peaks[0])):
                y = peaks[0][i]
                x = peaks[1][i] if len(peaks) > 1 else 0

                if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
                    intensity = float(frame[y, x])

                    detections.append({
                        'frame': frame_idx,
                        'x': float(x),
                        'y': float(y),
                        'intensity': intensity
                    })

        return pd.DataFrame(detections)

    def _simple_peak_detection(self, image, min_distance, threshold):
        """Simple peak detection fallback."""
        # Find points above threshold
        candidates = np.where(image > threshold)

        if len(candidates[0]) == 0:
            return [[], []]

        # Simple non-maximum suppression
        peaks_y = []
        peaks_x = []

        for i in range(len(candidates[0])):
            y, x = candidates[0][i], candidates[1][i]

            # Check if this point is a local maximum
            y_min = max(0, y - min_distance)
            y_max = min(image.shape[0], y + min_distance + 1)
            x_min = max(0, x - min_distance)
            x_max = min(image.shape[1], x + min_distance + 1)

            local_region = image[y_min:y_max, x_min:x_max]

            if image[y, x] == np.max(local_region):
                peaks_y.append(y)
                peaks_x.append(x)

        return [np.array(peaks_y), np.array(peaks_x)]


class TrackpyDetection(DetectionMethod):
    """Trackpy-based particle detection."""

    def detect(self, image: np.ndarray, diameter: int = 7,
               min_intensity: int = 100, **kwargs) -> pd.DataFrame:
        """Detect particles using trackpy."""

        try:
            import trackpy as tp
        except ImportError:
            raise ImportError("Trackpy is required for trackpy detection method")

        # Suppress trackpy warnings
        tp.quiet()

        if len(image.shape) == 2:
            # Single frame - add frame dimension
            image = image[np.newaxis, ...]

        # Detect particles
        features = tp.batch(image, diameter=diameter, minmass=min_intensity)

        # Convert to our format
        if len(features) > 0:
            detections = pd.DataFrame({
                'frame': features['frame'].astype(int),
                'x': features['x'],
                'y': features['y'],
                'intensity': features['mass']
            })
        else:
            detections = pd.DataFrame(columns=['frame', 'x', 'y', 'intensity'])

        return detections


class ParticleDetector:
    """Main particle detector class."""

    def __init__(self, parameters=None):
        self.logger = logging.getLogger(__name__)
        self.parameters = parameters or {}

        # Initialize detection methods
        self.methods = {
            'threshold': ThresholdDetection(),
            'log': LoGDetection(),
            'trackpy': TrackpyDetection()
        }

    def detect_particles(self, image: np.ndarray, method: str = 'threshold',
                        **kwargs) -> pd.DataFrame:
        """Detect particles using specified method."""

        if method not in self.methods:
            raise ValueError(f"Unknown detection method: {method}")

        detector = self.methods[method]

        # Merge parameters from constructor and method call
        params = {**self.parameters, **kwargs}

        # Log parameters for debugging
        self.logger.info(f"Detecting particles using {method} method")
        self.logger.info(f"Detection parameters: {params}")

        result = detector.detect(image, **params)

        # Add unique IDs
        if len(result) > 0:
            result['id'] = range(1, len(result) + 1)

        self.logger.info(f"Detected {len(result)} particles")
        return result

    def update_parameters(self, parameters):
        """Update detection parameters."""
        self.parameters = parameters
