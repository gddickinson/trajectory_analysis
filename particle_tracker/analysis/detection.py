#!/usr/bin/env python3
"""
Enhanced Particle Detection Module
==================================

Provides sophisticated particle detection capabilities for TIRF microscopy data
including background subtraction, ROI-based detection, and advanced filtering.
"""

import logging
import math
from typing import Optional, Dict, List, Any, Tuple, Union
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

# Import scipy with error handling
try:
    from scipy import stats, spatial, ndimage
    from scipy.signal import convolve2d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available. Some detection methods may not work.")

from sklearn.neighbors import KDTree
import skimage.io as skio

# Import scikit-image with error handling
try:
    from skimage import filters, feature, segmentation, measure, morphology, restoration
    from skimage.morphology import disk, white_tophat, opening, closing
    from skimage.filters import gaussian, median, rank
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


class ROIManager:
    """Manage regions of interest for detection and background subtraction."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rois = {}
        
    def load_roi_file(self, roi_path: str) -> Dict[str, np.ndarray]:
        """Load ROI data from file (supports .txt and .csv formats)."""
        try:
            roi_path = Path(roi_path)
            
            if roi_path.suffix.lower() == '.txt':
                # Load ImageJ-style ROI text file
                return self._load_imagej_roi(roi_path)
            elif roi_path.suffix.lower() == '.csv':
                # Load CSV ROI file
                return self._load_csv_roi(roi_path)
            else:
                self.logger.error(f"Unsupported ROI file format: {roi_path.suffix}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error loading ROI file {roi_path}: {e}")
            return {}
    
    def _load_imagej_roi(self, roi_path: Path) -> Dict[str, np.ndarray]:
        """Load ImageJ-style ROI file."""
        rois = {}
        try:
            with open(roi_path, 'r') as f:
                lines = f.readlines()
                
            # Parse ROI coordinates (simplified parser)
            current_roi = None
            coordinates = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('ROI'):
                    if current_roi and coordinates:
                        rois[current_roi] = np.array(coordinates)
                    current_roi = line
                    coordinates = []
                elif line and not line.startswith('#'):
                    # Try to parse coordinate
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            x, y = float(parts[0]), float(parts[1])
                            coordinates.append([x, y])
                    except:
                        continue
                        
            # Add last ROI
            if current_roi and coordinates:
                rois[current_roi] = np.array(coordinates)
                
        except Exception as e:
            self.logger.error(f"Error parsing ImageJ ROI file: {e}")
            
        return rois
    
    def _load_csv_roi(self, roi_path: Path) -> Dict[str, np.ndarray]:
        """Load CSV ROI file."""
        try:
            df = pd.read_csv(roi_path)
            rois = {}
            
            # Assume CSV has columns: roi_name, x, y
            if 'roi_name' in df.columns:
                for roi_name in df['roi_name'].unique():
                    roi_data = df[df['roi_name'] == roi_name]
                    coordinates = roi_data[['x', 'y']].values
                    rois[roi_name] = coordinates
            else:
                # Single ROI case
                rois['ROI_1'] = df[['x', 'y']].values
                
            return rois
            
        except Exception as e:
            self.logger.error(f"Error parsing CSV ROI file: {e}")
            return {}
    
    def get_roi_mask(self, roi_coords: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """Create a binary mask from ROI coordinates."""
        mask = np.zeros(image_shape, dtype=bool)
        
        try:
            # Simple rectangular ROI for now
            if len(roi_coords) == 2:  # Two points define rectangle
                x1, y1 = roi_coords[0]
                x2, y2 = roi_coords[1]
                
                x1, x2 = int(min(x1, x2)), int(max(x1, x2))
                y1, y2 = int(min(y1, y2)), int(max(y1, y2))
                
                mask[y1:y2, x1:x2] = True
            elif len(roi_coords) > 2:
                # Polygon ROI - use simple point-in-polygon
                from matplotlib.path import Path as MplPath
                path = MplPath(roi_coords)
                
                y_coords, x_coords = np.mgrid[:image_shape[0], :image_shape[1]]
                points = np.vstack((x_coords.ravel(), y_coords.ravel())).T
                mask = path.contains_points(points).reshape(image_shape)
                
        except Exception as e:
            self.logger.error(f"Error creating ROI mask: {e}")
            
        return mask


class BackgroundSubtractor:
    """Advanced background subtraction methods."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def subtract_background(self, image: np.ndarray, method: str = 'rolling_ball',
                          **kwargs) -> np.ndarray:
        """Apply background subtraction to image."""
        
        if method == 'rolling_ball':
            return self._rolling_ball_background(image, **kwargs)
        elif method == 'median_filter':
            return self._median_filter_background(image, **kwargs)
        elif method == 'morphological':
            return self._morphological_background(image, **kwargs)
        elif method == 'roi_based':
            return self._roi_based_background(image, **kwargs)
        elif method == 'temporal_median':
            return self._temporal_median_background(image, **kwargs)
        else:
            self.logger.warning(f"Unknown background subtraction method: {method}")
            return image
    
    def _rolling_ball_background(self, image: np.ndarray, radius: int = 50) -> np.ndarray:
        """Rolling ball background subtraction."""
        try:
            if SKIMAGE_AVAILABLE:
                # Use morphological opening as approximation
                selem = disk(radius)
                background = morphology.opening(image, selem)
                return np.maximum(image - background, 0)
            else:
                # Fallback to simple background estimation
                background = ndimage.uniform_filter(image.astype(float), size=radius*2)
                return np.maximum(image - background, 0)
        except Exception as e:
            self.logger.error(f"Error in rolling ball background subtraction: {e}")
            return image
    
    def _median_filter_background(self, image: np.ndarray, kernel_size: int = 25) -> np.ndarray:
        """Median filter background subtraction."""
        try:
            if SKIMAGE_AVAILABLE:
                background = median(image, disk(kernel_size))
            else:
                background = ndimage.median_filter(image, size=kernel_size)
            return np.maximum(image - background, 0)
        except Exception as e:
            self.logger.error(f"Error in median filter background subtraction: {e}")
            return image
    
    def _morphological_background(self, image: np.ndarray, 
                                 opening_size: int = 5, closing_size: int = 15) -> np.ndarray:
        """Morphological background subtraction."""
        try:
            if SKIMAGE_AVAILABLE:
                # Apply morphological opening followed by closing
                opened = opening(image, disk(opening_size))
                background = closing(opened, disk(closing_size))
            else:
                # Fallback using scipy
                opened = ndimage.grey_opening(image, size=opening_size)
                background = ndimage.grey_closing(opened, size=closing_size)
            
            return np.maximum(image - background, 0)
        except Exception as e:
            self.logger.error(f"Error in morphological background subtraction: {e}")
            return image
    
    def _roi_based_background(self, image: np.ndarray, roi_mask: np.ndarray = None) -> np.ndarray:
        """ROI-based background subtraction."""
        try:
            if roi_mask is not None:
                # Calculate background from ROI
                background_value = np.median(image[roi_mask])
                return np.maximum(image - background_value, 0)
            else:
                # Use image edges as background estimate
                edge_pixels = np.concatenate([
                    image[0, :].ravel(),   # top edge
                    image[-1, :].ravel(),  # bottom edge
                    image[:, 0].ravel(),   # left edge
                    image[:, -1].ravel()   # right edge
                ])
                background_value = np.median(edge_pixels)
                return np.maximum(image - background_value, 0)
        except Exception as e:
            self.logger.error(f"Error in ROI-based background subtraction: {e}")
            return image
    
    def _temporal_median_background(self, image_stack: np.ndarray) -> np.ndarray:
        """Temporal median background subtraction for image stacks."""
        try:
            if len(image_stack.shape) != 3:
                raise ValueError("Temporal background subtraction requires 3D image stack")
            
            # Calculate temporal median
            background = np.median(image_stack, axis=0)
            
            # Subtract from each frame
            result = np.zeros_like(image_stack)
            for i in range(image_stack.shape[0]):
                result[i] = np.maximum(image_stack[i] - background, 0)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in temporal median background subtraction: {e}")
            return image_stack


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


class EnhancedThresholdDetection(DetectionMethod):
    """Enhanced threshold-based particle detection with background subtraction."""

    def __init__(self, background_subtractor: BackgroundSubtractor = None, 
                 roi_manager: ROIManager = None):
        self.background_subtractor = background_subtractor or BackgroundSubtractor()
        self.roi_manager = roi_manager or ROIManager()
        self.logger = logging.getLogger(__name__)

    def detect(self, image: np.ndarray, threshold: float = 3.0,
               sigma: float = 1.6, min_intensity: int = 100,
               max_intensity: int = 10000, background_method: str = 'rolling_ball',
               background_params: Dict = None, roi_mask: np.ndarray = None,
               **kwargs) -> pd.DataFrame:
        """Enhanced threshold detection with background subtraction."""
        
        if background_params is None:
            background_params = {}
        
        self.logger.info(f"Starting enhanced threshold detection with background method: {background_method}")
        
        if len(image.shape) == 2:
            # Single frame
            return self._detect_frame_enhanced(
                image, 0, threshold, sigma, min_intensity, max_intensity,
                background_method, background_params, roi_mask
            )
        elif len(image.shape) == 3:
            # Time series
            results = []
            for frame_idx in tqdm(range(image.shape[0]), desc="Detecting particles"):
                frame_result = self._detect_frame_enhanced(
                    image[frame_idx], frame_idx, threshold, sigma,
                    min_intensity, max_intensity, background_method,
                    background_params, roi_mask
                )
                if len(frame_result) > 0:
                    results.append(frame_result)

            return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

    def _detect_frame_enhanced(self, frame: np.ndarray, frame_idx: int, threshold: float,
                              sigma: float, min_intensity: int, max_intensity: int,
                              background_method: str, background_params: Dict,
                              roi_mask: np.ndarray = None) -> pd.DataFrame:
        """Detect particles in a single frame with enhancement."""

        # Apply background subtraction
        if background_method != 'none':
            if roi_mask is not None:
                background_params['roi_mask'] = roi_mask
            
            frame_bg_subtracted = self.background_subtractor.subtract_background(
                frame, background_method, **background_params
            )
        else:
            frame_bg_subtracted = frame.copy()

        # Apply Gaussian filter
        if SCIPY_AVAILABLE:
            filtered = ndimage.gaussian_filter(frame_bg_subtracted, sigma=sigma)
        elif SKIMAGE_AVAILABLE:
            filtered = filters.gaussian(frame_bg_subtracted, sigma=sigma)
        else:
            # Simple fallback - no filtering
            filtered = frame_bg_subtracted.astype(float)

        # Calculate adaptive threshold
        if roi_mask is not None:
            # Use ROI for noise estimation
            noise_region = filtered[roi_mask]
            if len(noise_region) > 0:
                noise_std = np.std(noise_region)
                mean_bg = np.mean(noise_region)
            else:
                noise_std = np.std(filtered)
                mean_bg = np.mean(filtered)
        else:
            noise_std = np.std(filtered)
            mean_bg = np.mean(filtered)

        thresh_value = mean_bg + threshold * noise_std

        # Apply threshold
        binary = filtered > thresh_value

        # Morphological filtering
        if SCIPY_AVAILABLE:
            # Remove small objects and smooth boundaries
            binary = ndimage.binary_opening(binary, structure=safe_disk(1))
            binary = ndimage.binary_closing(binary, structure=safe_disk(1))

        # Label connected components
        if SKIMAGE_AVAILABLE:
            labeled = measure.label(binary)
            regions = measure.regionprops(labeled, intensity_image=frame)
        elif SCIPY_AVAILABLE:
            labeled, num_features = ndimage.label(binary)
            # Create simplified region properties
            regions = []
            for i in range(1, num_features + 1):
                mask = labeled == i
                if np.sum(mask) > 0:
                    y_coords, x_coords = np.where(mask)
                    centroid_y = np.mean(y_coords)
                    centroid_x = np.mean(x_coords)
                    mean_intensity = np.mean(frame[mask])
                    area = np.sum(mask)
                    
                    class SimpleRegion:
                        def __init__(self, centroid, mean_intensity, area):
                            self.centroid = centroid
                            self.mean_intensity = mean_intensity
                            self.area = area

                    regions.append(SimpleRegion((centroid_y, centroid_x), mean_intensity, area))
        else:
            return pd.DataFrame()

        # Convert to DataFrame with enhanced filtering
        detections = []
        for region in regions:
            if hasattr(region, 'mean_intensity'):
                intensity = region.mean_intensity
            else:
                intensity = 1000

            # Enhanced filtering criteria
            if min_intensity <= intensity <= max_intensity:
                if hasattr(region, 'centroid'):
                    centroid = region.centroid
                    y, x = centroid if len(centroid) == 2 else (centroid[0], centroid[1])
                    
                    # Additional quality filters
                    area = getattr(region, 'area', 1)
                    
                    # Filter by area (remove very large or very small objects)
                    if 1 <= area <= 100:  # Adjust based on expected particle size
                        # Calculate signal-to-noise ratio
                        local_bg = self._estimate_local_background(frame, int(x), int(y), radius=10)
                        snr = intensity / max(local_bg, 1)
                        
                        # Only keep high SNR detections
                        if snr > 2.0:  # Adjustable SNR threshold
                            detections.append({
                                'frame': frame_idx,
                                'x': float(x),
                                'y': float(y),
                                'intensity': float(intensity),
                                'area': float(area),
                                'snr': float(snr),
                                'background': float(local_bg)
                            })

        return pd.DataFrame(detections)
    
    def _estimate_local_background(self, image: np.ndarray, x: int, y: int, radius: int = 10) -> float:
        """Estimate local background around a point."""
        try:
            # Create annulus around the point
            h, w = image.shape
            y_min = max(0, y - radius)
            y_max = min(h, y + radius + 1)
            x_min = max(0, x - radius)
            x_max = min(w, x + radius + 1)
            
            # Get local region
            local_region = image[y_min:y_max, x_min:x_max]
            
            # Use percentile to estimate background (robust to outliers)
            return np.percentile(local_region, 25)  # 25th percentile
            
        except Exception:
            return np.median(image)


class EnhancedLoGDetection(DetectionMethod):
    """Enhanced Laplacian of Gaussian blob detection with background subtraction."""

    def __init__(self, background_subtractor: BackgroundSubtractor = None):
        self.background_subtractor = background_subtractor or BackgroundSubtractor()
        self.logger = logging.getLogger(__name__)

    def detect(self, image: np.ndarray, sigma: float = 1.6,
               threshold: float = 0.1, background_method: str = 'rolling_ball',
               background_params: Dict = None, **kwargs) -> pd.DataFrame:
        """Enhanced LoG detection with background subtraction."""
        
        if background_params is None:
            background_params = {}
        
        self.logger.info(f"Starting enhanced LoG detection with background method: {background_method}")

        if len(image.shape) == 2:
            return self._detect_frame_log_enhanced(
                image, 0, sigma, threshold, background_method, background_params
            )
        elif len(image.shape) == 3:
            results = []
            for frame_idx in tqdm(range(image.shape[0]), desc="LoG detection"):
                frame_result = self._detect_frame_log_enhanced(
                    image[frame_idx], frame_idx, sigma, threshold,
                    background_method, background_params
                )
                if len(frame_result) > 0:
                    results.append(frame_result)

            return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def _detect_frame_log_enhanced(self, frame: np.ndarray, frame_idx: int,
                                  sigma: float, threshold: float,
                                  background_method: str, background_params: Dict) -> pd.DataFrame:
        """Enhanced LoG detection in a single frame."""

        if not SKIMAGE_AVAILABLE:
            self.logger.warning("scikit-image required for LoG detection")
            return pd.DataFrame()

        # Apply background subtraction
        if background_method != 'none':
            frame_processed = self.background_subtractor.subtract_background(
                frame, background_method, **background_params
            )
        else:
            frame_processed = frame.copy()

        try:
            # Apply LoG filter with multiple scales for better detection
            scales = [sigma * 0.8, sigma, sigma * 1.2]  # Multi-scale detection
            responses = []
            
            for scale in scales:
                log_filtered = -filters.laplacian(filters.gaussian(frame_processed, sigma=scale))
                responses.append(log_filtered)
            
            # Combine responses (take maximum)
            combined_response = np.maximum.reduce(responses)
            
            # Find local maxima with non-maximum suppression
            from skimage.feature import peak_local_maxima
            peaks = peak_local_maxima(
                combined_response, 
                min_distance=int(2*sigma),
                threshold_abs=threshold,
                exclude_border=int(sigma)
            )

        except ImportError:
            # Fallback for older scikit-image versions
            log_filtered = -filters.laplacian(filters.gaussian(frame_processed, sigma=sigma))
            peaks = self._simple_peak_detection(log_filtered, int(2*sigma), threshold)

        # Extract detections with quality metrics
        detections = []
        if len(peaks) > 0 and len(peaks[0]) > 0:
            for i in range(len(peaks[0])):
                y = peaks[0][i]
                x = peaks[1][i] if len(peaks) > 1 else 0

                if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
                    intensity = float(frame[y, x])
                    response = float(combined_response[y, x])
                    
                    # Quality assessment
                    local_bg = self._estimate_local_background(frame, x, y)
                    snr = intensity / max(local_bg, 1)

                    detections.append({
                        'frame': frame_idx,
                        'x': float(x),
                        'y': float(y),
                        'intensity': intensity,
                        'log_response': response,
                        'snr': snr,
                        'background': local_bg
                    })

        return pd.DataFrame(detections)

    def _simple_peak_detection(self, image, min_distance, threshold):
        """Simple peak detection fallback."""
        candidates = np.where(image > threshold)

        if len(candidates[0]) == 0:
            return [[], []]

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
    
    def _estimate_local_background(self, image: np.ndarray, x: int, y: int, radius: int = 10) -> float:
        """Estimate local background around a point."""
        try:
            h, w = image.shape
            y_min = max(0, y - radius)
            y_max = min(h, y + radius + 1)
            x_min = max(0, x - radius)
            x_max = min(w, x + radius + 1)
            
            local_region = image[y_min:y_max, x_min:x_max]
            return np.percentile(local_region, 25)
            
        except Exception:
            return np.median(image)


class EnhancedTrackpyDetection(DetectionMethod):
    """Enhanced trackpy-based particle detection with preprocessing."""

    def __init__(self, background_subtractor: BackgroundSubtractor = None):
        self.background_subtractor = background_subtractor or BackgroundSubtractor()
        self.logger = logging.getLogger(__name__)

    def detect(self, image: np.ndarray, diameter: int = 7,
               min_intensity: int = 100, preprocess: bool = True,
               background_method: str = 'rolling_ball',
               background_params: Dict = None, **kwargs) -> pd.DataFrame:
        """Enhanced trackpy detection with preprocessing."""

        try:
            import trackpy as tp
        except ImportError:
            raise ImportError("Trackpy is required for trackpy detection method")

        if background_params is None:
            background_params = {}

        self.logger.info("Starting enhanced trackpy detection")
        tp.quiet()

        # Preprocess image stack
        if preprocess:
            processed_image = self._preprocess_image(image, background_method, background_params)
        else:
            processed_image = image

        if len(processed_image.shape) == 2:
            # Single frame - add frame dimension
            processed_image = processed_image[np.newaxis, ...]

        # Enhanced trackpy detection with optimized parameters
        try:
            # Use adaptive parameters based on image characteristics
            adaptive_diameter = self._estimate_optimal_diameter(processed_image[0], diameter)
            adaptive_minmass = self._estimate_optimal_minmass(processed_image[0], min_intensity)
            
            self.logger.info(f"Using adaptive diameter: {adaptive_diameter}, minmass: {adaptive_minmass}")
            
            features = tp.batch(
                processed_image, 
                diameter=adaptive_diameter, 
                minmass=adaptive_minmass,
                # Enhanced trackpy parameters
                noise_size=1,           # Background noise correlation length
                smoothing_size=None,    # Let trackpy decide
                threshold=None,         # Let trackpy decide adaptively
                invert=False,          # Assuming bright particles on dark background
                percentile=64,         # Background percentile
                topn=None,             # Keep all detections above threshold
                preprocess=True,       # Additional preprocessing
                max_iterations=10,     # Convergence iterations
                filter_before=True,    # Apply bandpass before detection
                filter_after=True      # Apply filters after detection
            )
            
        except Exception as e:
            self.logger.warning(f"Enhanced trackpy detection failed: {e}, falling back to basic detection")
            # Fallback to basic detection
            features = tp.batch(processed_image, diameter=diameter, minmass=min_intensity)

        # Convert to our format with enhanced metadata
        if len(features) > 0:
            detections = pd.DataFrame({
                'frame': features['frame'].astype(int),
                'x': features['x'],
                'y': features['y'],
                'intensity': features['mass'],
                'size': features.get('size', diameter),
                'ecc': features.get('ecc', np.nan),  # Eccentricity
                'signal': features.get('signal', np.nan),  # Signal strength
                'raw_mass': features.get('raw_mass', features['mass'])  # Before background subtraction
            })
        else:
            detections = pd.DataFrame(columns=['frame', 'x', 'y', 'intensity', 'size', 'ecc', 'signal', 'raw_mass'])

        self.logger.info(f"Detected {len(detections)} particles using enhanced trackpy")
        return detections

    def _preprocess_image(self, image: np.ndarray, background_method: str, 
                         background_params: Dict) -> np.ndarray:
        """Preprocess image before detection."""
        
        if background_method != 'none':
            if len(image.shape) == 3:
                # Process each frame
                processed = np.zeros_like(image)
                for i in range(image.shape[0]):
                    processed[i] = self.background_subtractor.subtract_background(
                        image[i], background_method, **background_params
                    )
                return processed
            else:
                return self.background_subtractor.subtract_background(
                    image, background_method, **background_params
                )
        else:
            return image
    
    def _estimate_optimal_diameter(self, image: np.ndarray, initial_diameter: int) -> int:
        """Estimate optimal diameter based on image characteristics."""
        try:
            # Simple heuristic: use autocorrelation to estimate feature size
            if SCIPY_AVAILABLE:
                # Calculate autocorrelation in small region
                h, w = image.shape
                center_region = image[h//4:3*h//4, w//4:3*w//4]
                
                # Find typical feature size using Fourier analysis
                fft = np.fft.fft2(center_region)
                power_spectrum = np.abs(fft)**2
                
                # Estimate dominant frequency
                freqs = np.fft.fftfreq(min(center_region.shape))
                dominant_freq_idx = np.argmax(np.mean(power_spectrum, axis=1))
                
                if freqs[dominant_freq_idx] != 0:
                    estimated_size = int(1.0 / abs(freqs[dominant_freq_idx]))
                    # Clamp to reasonable range
                    estimated_size = max(3, min(21, estimated_size))
                    
                    # Make odd
                    if estimated_size % 2 == 0:
                        estimated_size += 1
                        
                    return estimated_size
                    
        except Exception as e:
            self.logger.debug(f"Could not estimate optimal diameter: {e}")
            
        return initial_diameter
    
    def _estimate_optimal_minmass(self, image: np.ndarray, initial_minmass: int) -> int:
        """Estimate optimal minimum mass based on image statistics."""
        try:
            # Use image statistics to estimate appropriate threshold
            image_std = np.std(image)
            image_mean = np.mean(image)
            
            # Estimate noise level
            noise_level = image_std
            
            # Set minmass to be significantly above noise
            estimated_minmass = int(image_mean + 3 * noise_level)
            
            # Use the higher of the estimated or initial value
            return max(initial_minmass, estimated_minmass)
            
        except Exception as e:
            self.logger.debug(f"Could not estimate optimal minmass: {e}")
            
        return initial_minmass


class ParticleDetector:
    """Enhanced main particle detector class with advanced capabilities."""

    def __init__(self, parameters=None):
        self.logger = logging.getLogger(__name__)
        self.parameters = parameters or {}
        
        # Initialize components
        self.background_subtractor = BackgroundSubtractor()
        self.roi_manager = ROIManager()

        # Initialize detection methods with enhancement capabilities
        self.methods = {
            'threshold': EnhancedThresholdDetection(self.background_subtractor, self.roi_manager),
            'log': EnhancedLoGDetection(self.background_subtractor),
            'trackpy': EnhancedTrackpyDetection(self.background_subtractor)
        }

    def detect_particles(self, image: np.ndarray, method: str = 'threshold',
                        roi_file: str = None, **kwargs) -> pd.DataFrame:
        """Enhanced particle detection with ROI support."""

        if method not in self.methods:
            raise ValueError(f"Unknown detection method: {method}")

        detector = self.methods[method]

        # Merge parameters from constructor and method call
        params = {**self.parameters, **kwargs}
        
        # Load ROI if specified
        roi_mask = None
        if roi_file and Path(roi_file).exists():
            self.logger.info(f"Loading ROI file: {roi_file}")
            rois = self.roi_manager.load_roi_file(roi_file)
            
            if rois:
                # Use first ROI for background estimation
                roi_name = list(rois.keys())[0]
                roi_coords = rois[roi_name]
                
                # Create mask for background region
                if len(image.shape) == 3:
                    roi_mask = self.roi_manager.get_roi_mask(roi_coords, image.shape[1:])
                else:
                    roi_mask = self.roi_manager.get_roi_mask(roi_coords, image.shape)
                
                params['roi_mask'] = roi_mask
                self.logger.info(f"Using ROI '{roi_name}' with {np.sum(roi_mask)} pixels")

        # Log parameters for debugging
        self.logger.info(f"Detecting particles using enhanced {method} method")
        self.logger.info(f"Detection parameters: {params}")

        result = detector.detect(image, **params)

        # Add unique IDs and additional metadata
        if len(result) > 0:
            result['id'] = range(1, len(result) + 1)
            
            # Add detection metadata
            result['detection_method'] = method
            result['background_subtracted'] = params.get('background_method', 'none') != 'none'
            
            # Calculate additional quality metrics
            if 'snr' not in result.columns and 'intensity' in result.columns:
                # Estimate SNR if not already calculated
                result['snr'] = self._estimate_snr(image, result)

        self.logger.info(f"Detected {len(result)} particles")
        
        # Log detection statistics
        if len(result) > 0:
            intensity_stats = result['intensity'].describe()
            self.logger.info(f"Intensity statistics: mean={intensity_stats['mean']:.1f}, "
                           f"std={intensity_stats['std']:.1f}, "
                           f"range=({intensity_stats['min']:.1f}, {intensity_stats['max']:.1f})")
            
            if 'snr' in result.columns:
                snr_mean = result['snr'].mean()
                self.logger.info(f"Average SNR: {snr_mean:.2f}")

        return result
    
    def _estimate_snr(self, image: np.ndarray, detections: pd.DataFrame) -> List[float]:
        """Estimate signal-to-noise ratio for detections."""
        snr_values = []
        
        for _, detection in detections.iterrows():
            frame = int(detection['frame']) if 'frame' in detection else 0
            x = int(detection['x'])
            y = int(detection['y'])
            intensity = detection['intensity']
            
            try:
                if len(image.shape) == 3:
                    frame_image = image[frame]
                else:
                    frame_image = image
                
                # Estimate local background
                h, w = frame_image.shape
                radius = 10
                y_min = max(0, y - radius)
                y_max = min(h, y + radius + 1)
                x_min = max(0, x - radius)
                x_max = min(w, x + radius + 1)
                
                local_region = frame_image[y_min:y_max, x_min:x_max]
                background = np.percentile(local_region, 25)
                noise = np.std(local_region)
                
                snr = (intensity - background) / max(noise, 1)
                snr_values.append(max(0, snr))  # Ensure non-negative
                
            except Exception:
                snr_values.append(1.0)  # Default SNR
        
        return snr_values

    def batch_detect(self, image_paths: List[str], method: str = 'threshold',
                    output_dir: str = None, **kwargs) -> Dict[str, pd.DataFrame]:
        """Batch detection across multiple image files."""
        
        results = {}
        
        for image_path in tqdm(image_paths, desc="Batch detection"):
            try:
                self.logger.info(f"Processing {image_path}")
                
                # Load image
                image = skio.imread(image_path)
                
                # Detect particles
                detections = self.detect_particles(image, method, **kwargs)
                
                # Store results
                results[image_path] = detections
                
                # Save individual results if output directory specified
                if output_dir:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    filename = Path(image_path).stem + '_detections.csv'
                    output_file = output_path / filename
                    detections.to_csv(output_file, index=False)
                    
                    self.logger.info(f"Saved detections to {output_file}")
                
            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {e}")
                results[image_path] = pd.DataFrame()
        
        return results

    def update_parameters(self, parameters):
        """Update detection parameters."""
        self.parameters.update(parameters)
        
        # Update component parameters
        for method in self.methods.values():
            if hasattr(method, 'parameters'):
                method.parameters = self.parameters

    def get_supported_background_methods(self) -> List[str]:
        """Get list of supported background subtraction methods."""
        return ['none', 'rolling_ball', 'median_filter', 'morphological', 'roi_based', 'temporal_median']
    
    def validate_parameters(self, method: str, params: Dict) -> Dict[str, Any]:
        """Validate and suggest optimal parameters for detection method."""
        
        validated_params = params.copy()
        warnings = []
        
        if method == 'threshold':
            # Validate threshold parameters
            if params.get('threshold', 3.0) < 1.0:
                warnings.append("Threshold < 1.0 may result in many false positives")
            if params.get('sigma', 1.6) > 5.0:
                warnings.append("Large sigma values may reduce detection sensitivity")
                
        elif method == 'log':
            # Validate LoG parameters
            if params.get('sigma', 1.6) < 0.5:
                warnings.append("Very small sigma may not capture particle features well")
                
        elif method == 'trackpy':
            # Validate trackpy parameters
            diameter = params.get('diameter', 7)
            if diameter % 2 == 0:
                validated_params['diameter'] = diameter + 1
                warnings.append(f"Trackpy diameter must be odd, adjusted to {diameter + 1}")
        
        if warnings:
            for warning in warnings:
                self.logger.warning(warning)
        
        return validated_params