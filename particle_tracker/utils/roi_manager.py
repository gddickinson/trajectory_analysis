#!/usr/bin/env python3
"""
ROI Manager Module
==================

Handles Region of Interest (ROI) operations for background subtraction and intensity analysis.
Supports multiple ROI formats and provides sophisticated background correction methods.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
import csv

try:
    import flika
    from flika.roi import open_rois
    from flika.process.file_ import open_file
    FLIKA_AVAILABLE = True
except ImportError:
    FLIKA_AVAILABLE = False


class ROIType(Enum):
    """Types of ROI supported."""
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    POLYGON = "polygon"
    FREEHAND = "freehand"


@dataclass
class ROI:
    """Region of Interest data structure."""
    name: str
    roi_type: ROIType
    coordinates: np.ndarray  # Shape depends on ROI type
    frame_range: Optional[Tuple[int, int]] = None  # (start_frame, end_frame)
    properties: Dict = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class ROIManager:
    """Manages ROI operations for background subtraction and analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rois: Dict[str, ROI] = {}
        self.background_traces: Dict[str, np.ndarray] = {}
        self.camera_estimates: Optional[np.ndarray] = None
        
    def load_rois_from_flika(self, roi_file_path: str, image_data: np.ndarray) -> bool:
        """
        Load ROIs from FLIKA ROI file format.
        
        Args:
            roi_file_path: Path to the ROI file
            image_data: Image stack for extracting traces
            
        Returns:
            True if successful, False otherwise
        """
        if not FLIKA_AVAILABLE:
            self.logger.error("FLIKA not available for ROI loading")
            return False
            
        try:
            from flika import start_flika
            
            # Start FLIKA session
            fa = start_flika()
            
            # Create temporary window for ROI operations
            from flika.window import Window
            temp_window = Window(image_data)
            
            # Load ROIs
            flika_rois = open_rois(roi_file_path)
            
            self.logger.info(f"Loaded {len(flika_rois)} ROIs from {roi_file_path}")
            
            # Convert FLIKA ROIs to our format and extract traces
            for i, flika_roi in enumerate(flika_rois):
                roi_name = f"roi_{i+1}"
                
                # Extract trace from ROI
                trace = flika_roi.getTrace()
                self.background_traces[roi_name] = trace
                
                # Try to get ROI coordinates (this depends on FLIKA ROI type)
                try:
                    # Get ROI coordinates - this is approximate since FLIKA ROI structure varies
                    coords = self._extract_flika_roi_coordinates(flika_roi)
                    roi_type = self._detect_roi_type(coords)
                    
                    roi = ROI(
                        name=roi_name,
                        roi_type=roi_type,
                        coordinates=coords
                    )
                    
                    self.rois[roi_name] = roi
                    
                except Exception as e:
                    self.logger.warning(f"Could not extract coordinates for ROI {i+1}: {e}")
                    # Still keep the trace even if coordinates extraction fails
                    roi = ROI(
                        name=roi_name,
                        roi_type=ROIType.RECTANGLE,
                        coordinates=np.array([[0, 0], [10, 10]])  # Dummy coordinates
                    )
                    self.rois[roi_name] = roi
            
            # Calculate camera black estimates
            self.camera_estimates = np.min(image_data, axis=(1, 2))
            
            # Clean up FLIKA session
            fa.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading FLIKA ROIs: {e}")
            return False
    
    def _extract_flika_roi_coordinates(self, flika_roi) -> np.ndarray:
        """
        Extract coordinates from FLIKA ROI object.
        
        This is a best-effort extraction since FLIKA ROI internal structure can vary.
        """
        try:
            # Try to get coordinates from different possible attributes
            if hasattr(flika_roi, 'roi'):
                # Some FLIKA ROIs store coordinates in roi attribute
                roi_data = flika_roi.roi
                if hasattr(roi_data, 'coords'):
                    return np.array(roi_data.coords)
            
            # Try other common attributes
            for attr in ['coordinates', 'points', 'coords', 'pos']:
                if hasattr(flika_roi, attr):
                    coords = getattr(flika_roi, attr)
                    if isinstance(coords, (list, tuple, np.ndarray)):
                        return np.array(coords)
            
            # If we can't extract coordinates, create a default rectangle
            return np.array([[0, 0], [50, 50]])  # Default 50x50 rectangle
            
        except Exception as e:
            self.logger.warning(f"Error extracting ROI coordinates: {e}")
            return np.array([[0, 0], [50, 50]])
    
    def _detect_roi_type(self, coordinates: np.ndarray) -> ROIType:
        """Detect ROI type based on coordinates."""
        if len(coordinates) == 2:
            return ROIType.RECTANGLE
        elif len(coordinates) == 4:
            return ROIType.RECTANGLE
        elif len(coordinates) > 4:
            return ROIType.POLYGON
        else:
            return ROIType.RECTANGLE
    
    def load_rois_from_csv(self, csv_file_path: str) -> bool:
        """
        Load ROIs from CSV file.
        
        Expected CSV format:
        name, type, x1, y1, x2, y2, [additional coordinates for polygons]
        """
        try:
            with open(csv_file_path, 'r') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    roi_name = row['name']
                    roi_type = ROIType(row['type'])
                    
                    # Extract coordinates based on type
                    if roi_type == ROIType.RECTANGLE:
                        coords = np.array([
                            [float(row['x1']), float(row['y1'])],
                            [float(row['x2']), float(row['y2'])]
                        ])
                    elif roi_type == ROIType.CIRCLE:
                        # For circles: x1,y1 = center, x2 = radius
                        center_x, center_y = float(row['x1']), float(row['y1'])
                        radius = float(row['x2'])
                        coords = np.array([center_x, center_y, radius])
                    else:
                        # For polygons, pack all coordinate pairs
                        coords = []
                        i = 1
                        while f'x{i}' in row and f'y{i}' in row:
                            if row[f'x{i}'] and row[f'y{i}']:
                                coords.append([float(row[f'x{i}']), float(row[f'y{i}'])])
                            i += 1
                        coords = np.array(coords)
                    
                    roi = ROI(
                        name=roi_name,
                        roi_type=roi_type,
                        coordinates=coords
                    )
                    
                    self.rois[roi_name] = roi
            
            self.logger.info(f"Loaded {len(self.rois)} ROIs from {csv_file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading ROIs from CSV: {e}")
            return False
    
    def load_rois_from_json(self, json_file_path: str) -> bool:
        """Load ROIs from JSON file."""
        try:
            with open(json_file_path, 'r') as f:
                roi_data = json.load(f)
            
            for roi_dict in roi_data['rois']:
                roi = ROI(
                    name=roi_dict['name'],
                    roi_type=ROIType(roi_dict['type']),
                    coordinates=np.array(roi_dict['coordinates']),
                    frame_range=tuple(roi_dict['frame_range']) if roi_dict.get('frame_range') else None,
                    properties=roi_dict.get('properties', {})
                )
                
                self.rois[roi.name] = roi
            
            self.logger.info(f"Loaded {len(self.rois)} ROIs from {json_file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading ROIs from JSON: {e}")
            return False
    
    def create_manual_roi(self, name: str, roi_type: ROIType, coordinates: np.ndarray,
                         frame_range: Optional[Tuple[int, int]] = None) -> bool:
        """
        Create a manual ROI.
        
        Args:
            name: ROI name
            roi_type: Type of ROI
            coordinates: ROI coordinates
            frame_range: Optional frame range (start, end)
            
        Returns:
            True if successful
        """
        try:
            roi = ROI(
                name=name,
                roi_type=roi_type,
                coordinates=coordinates,
                frame_range=frame_range
            )
            
            self.rois[name] = roi
            self.logger.info(f"Created manual ROI: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating manual ROI: {e}")
            return False
    
    def extract_roi_traces(self, image_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract intensity traces from all ROIs.
        
        Args:
            image_data: Image stack (frames, height, width)
            
        Returns:
            Dictionary mapping ROI names to intensity traces
        """
        traces = {}
        
        for roi_name, roi in self.rois.items():
            try:
                trace = self._extract_single_roi_trace(image_data, roi)
                traces[roi_name] = trace
                self.background_traces[roi_name] = trace
                
            except Exception as e:
                self.logger.error(f"Error extracting trace for ROI {roi_name}: {e}")
                continue
        
        # Also calculate camera estimates if not already done
        if self.camera_estimates is None:
            self.camera_estimates = np.min(image_data, axis=(1, 2))
        
        return traces
    
    def _extract_single_roi_trace(self, image_data: np.ndarray, roi: ROI) -> np.ndarray:
        """Extract intensity trace from a single ROI."""
        n_frames = image_data.shape[0]
        trace = np.zeros(n_frames)
        
        # Create mask for the ROI
        mask = self._create_roi_mask(roi, image_data.shape[1:])
        
        # Extract mean intensity for each frame
        for frame_idx in range(n_frames):
            if roi.frame_range:
                start_frame, end_frame = roi.frame_range
                if not (start_frame <= frame_idx <= end_frame):
                    trace[frame_idx] = np.nan
                    continue
            
            frame_data = image_data[frame_idx]
            roi_pixels = frame_data[mask]
            trace[frame_idx] = np.mean(roi_pixels) if len(roi_pixels) > 0 else np.nan
        
        return trace
    
    def _create_roi_mask(self, roi: ROI, image_shape: Tuple[int, int]) -> np.ndarray:
        """Create a boolean mask for the ROI."""
        height, width = image_shape
        mask = np.zeros((height, width), dtype=bool)
        
        if roi.roi_type == ROIType.RECTANGLE:
            if len(roi.coordinates) == 2:
                # Two corners format
                x1, y1 = roi.coordinates[0]
                x2, y2 = roi.coordinates[1]
            else:
                # x, y, width, height format
                x1, y1, w, h = roi.coordinates
                x2, y2 = x1 + w, y1 + h
            
            x1, x2 = int(min(x1, x2)), int(max(x1, x2))
            y1, y2 = int(min(y1, y2)), int(max(y1, y2))
            
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, width-1))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height-1))
            y2 = max(0, min(y2, height))
            
            mask[y1:y2, x1:x2] = True
            
        elif roi.roi_type == ROIType.CIRCLE:
            center_x, center_y, radius = roi.coordinates
            y_coords, x_coords = np.ogrid[:height, :width]
            mask = ((x_coords - center_x)**2 + (y_coords - center_y)**2) <= radius**2
            
        elif roi.roi_type == ROIType.POLYGON:
            # Use matplotlib's path for polygon masking
            try:
                from matplotlib.path import Path as MPLPath
                
                # Create path from polygon coordinates
                path = MPLPath(roi.coordinates)
                
                # Create coordinate grids
                x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
                points = np.column_stack((x_coords.ravel(), y_coords.ravel()))
                
                # Test which points are inside the polygon
                mask = path.contains_points(points).reshape(height, width)
                
            except ImportError:
                self.logger.warning("matplotlib not available for polygon ROI, using bounding box")
                # Fallback to bounding box
                min_x = int(np.min(roi.coordinates[:, 0]))
                max_x = int(np.max(roi.coordinates[:, 0]))
                min_y = int(np.min(roi.coordinates[:, 1]))
                max_y = int(np.max(roi.coordinates[:, 1]))
                
                min_x = max(0, min(min_x, width-1))
                max_x = max(0, min(max_x, width))
                min_y = max(0, min(min_y, height-1))
                max_y = max(0, min(max_y, height))
                
                mask[min_y:max_y, min_x:max_x] = True
        
        return mask
    
    def apply_background_subtraction(self, df: pd.DataFrame, 
                                   primary_roi: str = "roi_1",
                                   methods: List[str] = None) -> pd.DataFrame:
        """
        Apply background subtraction to trajectory data.
        
        Args:
            df: DataFrame with trajectory data including 'frame' and 'intensity' columns
            primary_roi: Name of primary background ROI
            methods: List of background subtraction methods to apply
            
        Returns:
            DataFrame with background-subtracted intensity columns
        """
        if methods is None:
            methods = ["mean_roi", "mean_roi_and_black", "smoothed_roi"]
        
        result_df = df.copy()
        
        # Get background traces
        if primary_roi not in self.background_traces:
            self.logger.error(f"Background ROI {primary_roi} not found")
            return result_df
        
        roi_trace = self.background_traces[primary_roi]
        
        # Apply each background subtraction method
        for method in methods:
            if method == "mean_roi":
                self._apply_mean_roi_subtraction(result_df, roi_trace, primary_roi)
            elif method == "mean_roi_and_black":
                self._apply_mean_roi_and_black_subtraction(result_df, roi_trace, primary_roi)
            elif method == "smoothed_roi":
                self._apply_smoothed_roi_subtraction(result_df, roi_trace, primary_roi)
            elif method == "frame_by_frame":
                self._apply_frame_by_frame_subtraction(result_df, roi_trace, primary_roi)
        
        return result_df
    
    def _apply_mean_roi_subtraction(self, df: pd.DataFrame, roi_trace: np.ndarray, roi_name: str):
        """Apply mean ROI background subtraction."""
        # Add ROI values for each frame
        for frame, value in enumerate(roi_trace):
            df.loc[df['frame'] == frame, roi_name] = value
        
        # Calculate mean background and subtract
        mean_background = np.nanmean(roi_trace)
        df[f'intensity_minus_mean_{roi_name}'] = df['intensity'] - mean_background
        
        self.logger.info(f"Applied mean ROI subtraction using {roi_name}")
    
    def _apply_mean_roi_and_black_subtraction(self, df: pd.DataFrame, roi_trace: np.ndarray, roi_name: str):
        """Apply mean ROI + camera black subtraction."""
        if self.camera_estimates is None:
            self.logger.warning("No camera estimates available for black subtraction")
            return
        
        # Add camera estimates for each frame
        for frame, value in enumerate(self.camera_estimates):
            df.loc[df['frame'] == frame, 'camera_black_estimate'] = value
        
        # Calculate combined background
        mean_roi_background = np.nanmean(roi_trace)
        mean_black = np.nanmean(self.camera_estimates)
        
        df[f'intensity_minus_mean_{roi_name}_and_black'] = (
            df['intensity'] - mean_roi_background - mean_black
        )
        
        self.logger.info(f"Applied mean ROI + black subtraction using {roi_name}")
    
    def _apply_smoothed_roi_subtraction(self, df: pd.DataFrame, roi_trace: np.ndarray, roi_name: str):
        """Apply smoothed ROI background subtraction."""
        # Smooth the ROI signal
        smoothing_window = max(1, len(roi_trace) // 10)  # 10% of total frames
        smoothed_roi = self._moving_average(roi_trace, smoothing_window)
        
        # Add smoothed ROI values for each frame
        for frame, value in enumerate(smoothed_roi):
            df.loc[df['frame'] == frame, f'{roi_name}_smoothed'] = value
        
        # Apply subtraction
        df[f'intensity_minus_smoothed_{roi_name}'] = (
            df['intensity'] - df[f'{roi_name}_smoothed']
        )
        
        self.logger.info(f"Applied smoothed ROI subtraction using {roi_name}")
    
    def _apply_frame_by_frame_subtraction(self, df: pd.DataFrame, roi_trace: np.ndarray, roi_name: str):
        """Apply frame-by-frame ROI background subtraction."""
        # Add ROI values for each frame
        for frame, value in enumerate(roi_trace):
            df.loc[df['frame'] == frame, f'{roi_name}_frame'] = value
        
        # Apply frame-by-frame subtraction
        df[f'intensity_minus_{roi_name}_frame'] = (
            df['intensity'] - df[f'{roi_name}_frame']
        )
        
        self.logger.info(f"Applied frame-by-frame ROI subtraction using {roi_name}")
    
    def _moving_average(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate moving average with edge padding."""
        if window_size <= 1:
            return data
        
        # Use convolution for moving average
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(data, kernel, mode='valid')
        
        # Pad edges to maintain original length
        start_pad = (len(data) - len(smoothed)) // 2
        end_pad = len(data) - len(smoothed) - start_pad
        
        smoothed = np.pad(smoothed, (start_pad, end_pad), mode='edge')
        
        return smoothed
    
    def get_roi_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all ROI traces."""
        stats = {}
        
        for roi_name, trace in self.background_traces.items():
            valid_trace = trace[~np.isnan(trace)]
            
            if len(valid_trace) > 0:
                stats[roi_name] = {
                    'mean': np.mean(valid_trace),
                    'std': np.std(valid_trace),
                    'min': np.min(valid_trace),
                    'max': np.max(valid_trace),
                    'median': np.median(valid_trace),
                    'n_frames': len(valid_trace)
                }
            else:
                stats[roi_name] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'min': np.nan,
                    'max': np.nan,
                    'median': np.nan,
                    'n_frames': 0
                }
        
        return stats
    
    def save_rois_to_json(self, file_path: str) -> bool:
        """Save ROIs to JSON file."""
        try:
            roi_data = {
                'rois': []
            }
            
            for roi in self.rois.values():
                roi_dict = {
                    'name': roi.name,
                    'type': roi.roi_type.value,
                    'coordinates': roi.coordinates.tolist(),
                    'frame_range': roi.frame_range,
                    'properties': roi.properties
                }
                roi_data['rois'].append(roi_dict)
            
            with open(file_path, 'w') as f:
                json.dump(roi_data, f, indent=2)
            
            self.logger.info(f"Saved {len(self.rois)} ROIs to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving ROIs to JSON: {e}")
            return False
    
    def save_roi_traces_to_csv(self, file_path: str) -> bool:
        """Save ROI traces to CSV file."""
        try:
            if not self.background_traces:
                self.logger.warning("No ROI traces to save")
                return False
            
            # Create DataFrame with all traces
            max_frames = max(len(trace) for trace in self.background_traces.values())
            
            data = {'frame': range(max_frames)}
            for roi_name, trace in self.background_traces.items():
                # Pad trace if necessary
                padded_trace = np.full(max_frames, np.nan)
                padded_trace[:len(trace)] = trace
                data[roi_name] = padded_trace
            
            # Add camera estimates if available
            if self.camera_estimates is not None:
                padded_camera = np.full(max_frames, np.nan)
                padded_camera[:len(self.camera_estimates)] = self.camera_estimates
                data['camera_black_estimate'] = padded_camera
            
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            
            self.logger.info(f"Saved ROI traces to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving ROI traces: {e}")
            return False
    
    def auto_detect_roi_file(self, data_file_path: str) -> Optional[str]:
        """
        Auto-detect ROI file based on data file path.
        
        Follows the naming convention from the original scripts:
        - For binned data: <basename>_ROI.txt
        - For regular data: ROI_<basename>.txt
        """
        data_path = Path(data_file_path)
        data_dir = data_path.parent
        base_name = data_path.stem
        
        # Try different ROI file naming patterns
        patterns = [
            f"{base_name}_ROI.txt",
            f"ROI_{base_name}.txt",
            f"{base_name.split('_bin')[0]}_ROI.txt",  # For binned data
            f"{base_name.split('_locs')[0]}_ROI.txt",  # For localization data
            "roi.txt",
            "ROI.txt"
        ]
        
        for pattern in patterns:
            roi_file = data_dir / pattern
            if roi_file.exists():
                self.logger.info(f"Auto-detected ROI file: {roi_file}")
                return str(roi_file)
        
        self.logger.warning(f"No ROI file found for {data_file_path}")
        return None
    
    def clear_rois(self):
        """Clear all loaded ROIs and traces."""
        self.rois.clear()
        self.background_traces.clear()
        self.camera_estimates = None
        self.logger.info("Cleared all ROIs and traces")
    
    def get_roi_names(self) -> List[str]:
        """Get list of ROI names."""
        return list(self.rois.keys())
    
    def get_background_trace_names(self) -> List[str]:
        """Get list of background trace names."""
        return list(self.background_traces.keys())
    
    def has_roi(self, roi_name: str) -> bool:
        """Check if ROI exists."""
        return roi_name in self.rois
    
    def has_background_trace(self, roi_name: str) -> bool:
        """Check if background trace exists."""
        return roi_name in self.background_traces