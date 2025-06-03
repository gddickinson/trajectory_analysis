#!/usr/bin/env python3
"""
Data Manager Module
==================

Handles all data operations including loading, saving, and managing different
data types (raw images, localizations, trajectories, analysis results).
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import skimage.io as skio
from PyQt6.QtCore import QObject, pyqtSignal

@dataclass
class DataInfo:
    """Metadata about loaded data."""
    name: str
    data_type: str
    shape: Tuple[int, ...]
    dtype: str
    file_path: Optional[str] = None
    creation_time: Optional[str] = None
    n_tracks: Optional[int] = None
    n_frames: Optional[int] = None
    pixel_size: Optional[float] = None
    frame_rate: Optional[float] = None


class DataType(Enum):
    """Enumeration of supported data types."""
    RAW_IMAGE = "raw_image"
    LOCALIZATIONS = "localizations"
    TRAJECTORIES = "trajectories"
    ANALYSIS_RESULTS = "analysis_results"
    BINARY_MASK = "binary_mask"
    ROI_DATA = "roi_data"


class DataManager(QObject):
    """Manages all data operations for the particle tracking application."""
    
    # Signals
    dataLoaded = pyqtSignal(str, object)  # data_name, data
    dataChanged = pyqtSignal(str)  # data_name
    dataRemoved = pyqtSignal(str)  # data_name
    progressUpdate = pyqtSignal(str, int)  # message, percentage
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self._data: Dict[str, Any] = {}
        self._data_info: Dict[str, DataInfo] = {}
        
        # Supported file formats
        self.supported_formats = {
            'image': ['.tif', '.tiff', '.png', '.jpg', '.jpeg'],
            'localization': ['.csv', '.txt', '.json'],
            'trajectory': ['.csv', '.txt', '.json'],
            'analysis': ['.csv', '.txt', '.json', '.xlsx']
        }
        
        self.logger.info("Data Manager initialized")
        
    def load_file(self, file_path: Union[str, Path], data_name: Optional[str] = None,
                  data_type: Optional[DataType] = None, **kwargs) -> bool:
        """Load data from file.
        
        Args:
            file_path: Path to the file
            data_name: Name for the data (defaults to filename)
            data_type: Type of data being loaded
            **kwargs: Additional parameters for loading
            
        Returns:
            True if successful, False otherwise
        """
        file_path = Path(file_path)
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return False
            
        if data_name is None:
            data_name = file_path.stem
            
        self.logger.info(f"Loading file: {file_path}")
        self.progressUpdate.emit(f"Loading {file_path.name}...", 0)
        
        try:
            # Auto-detect data type if not specified
            if data_type is None:
                data_type = self._detect_data_type(file_path)
                
            # Load based on data type
            if data_type == DataType.RAW_IMAGE:
                data = self._load_image(file_path, **kwargs)
            elif data_type in [DataType.LOCALIZATIONS, DataType.TRAJECTORIES, DataType.ANALYSIS_RESULTS]:
                data = self._load_tabular_data(file_path, **kwargs)
            elif data_type == DataType.BINARY_MASK:
                data = self._load_binary_mask(file_path, **kwargs)
            else:
                self.logger.error(f"Unsupported data type: {data_type}")
                return False
                
            # Store data and metadata
            self._data[data_name] = data
            self._data_info[data_name] = self._create_data_info(
                data_name, data, data_type, file_path
            )
            
            self.progressUpdate.emit("Loading complete", 100)
            self.dataLoaded.emit(data_name, data)
            self.logger.info(f"Successfully loaded: {data_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            self.progressUpdate.emit("Loading failed", 0)
            return False
            
    def _detect_data_type(self, file_path: Path) -> DataType:
        """Auto-detect data type based on file extension and content."""
        ext = file_path.suffix.lower()
        
        if ext in self.supported_formats['image']:
            return DataType.RAW_IMAGE
        elif ext in ['.csv', '.txt']:
            # Try to determine if it's localizations or trajectories
            try:
                df = pd.read_csv(file_path, nrows=5)
                if 'track_number' in df.columns or 'particle' in df.columns:
                    return DataType.TRAJECTORIES
                elif any(col in df.columns for col in ['x [nm]', 'y [nm]', 'frame']):
                    return DataType.LOCALIZATIONS
                else:
                    return DataType.ANALYSIS_RESULTS
            except:
                return DataType.ANALYSIS_RESULTS
        elif ext == '.json':
            return DataType.TRAJECTORIES
        else:
            return DataType.ANALYSIS_RESULTS
            
    def _load_image(self, file_path: Path, **kwargs) -> np.ndarray:
        """Load image data."""
        try:
            # Use skimage for TIFF files (better support for scientific formats)
            if file_path.suffix.lower() in ['.tif', '.tiff']:
                image = skio.imread(str(file_path), plugin='tifffile')
            else:
                image = skio.imread(str(file_path))
                
            self.logger.info(f"Loaded image with shape: {image.shape}, dtype: {image.dtype}")
            return image
            
        except Exception as e:
            self.logger.error(f"Error loading image {file_path}: {e}")
            raise
            
    def _load_tabular_data(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load tabular data (CSV, Excel, etc.)."""
        try:
            if file_path.suffix.lower() == '.csv':
                # Try different separators and encodings
                for sep in [',', '\t', ';']:
                    try:
                        df = pd.read_csv(file_path, sep=sep, **kwargs)
                        if len(df.columns) > 1:  # Found correct separator
                            break
                    except:
                        continue
                else:
                    df = pd.read_csv(file_path, **kwargs)  # Default
                    
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, **kwargs)
            elif file_path.suffix.lower() == '.json':
                df = self._load_json_trajectories(file_path)
            else:
                df = pd.read_csv(file_path, **kwargs)  # Try as CSV
                
            self.logger.info(f"Loaded tabular data: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading tabular data {file_path}: {e}")
            raise
            
    def _load_json_trajectories(self, file_path: Path) -> pd.DataFrame:
        """Load trajectory data from JSON format (e.g., from tracking software)."""
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Convert to DataFrame format
        if 'txy_pts' in data and 'tracks' in data:
            # Format from your existing scripts
            frames, track_numbers, x_coords, y_coords = [], [], [], []
            txy_pts = np.array(data['txy_pts'])
            
            for track_idx, track in enumerate(data['tracks']):
                for point_id in track:
                    point_data = txy_pts[point_id]
                    frames.append(point_data[0])
                    x_coords.append(point_data[1])
                    y_coords.append(point_data[2])
                    track_numbers.append(track_idx)
                    
            df = pd.DataFrame({
                'frame': frames,
                'track_number': track_numbers,
                'x': x_coords,
                'y': y_coords
            })
        else:
            # Try to load as regular JSON
            df = pd.read_json(file_path)
            
        return df
        
    def _load_binary_mask(self, file_path: Path, **kwargs) -> np.ndarray:
        """Load binary mask data."""
        mask = skio.imread(str(file_path))
        if mask.dtype != bool:
            mask = mask > 0  # Convert to boolean
        return mask
        
    def _create_data_info(self, name: str, data: Any, data_type: DataType, 
                          file_path: Path) -> DataInfo:
        """Create metadata for loaded data."""
        if isinstance(data, np.ndarray):
            shape = data.shape
            dtype = str(data.dtype)
            n_frames = shape[0] if len(shape) >= 3 else None
        elif isinstance(data, pd.DataFrame):
            shape = data.shape
            dtype = "DataFrame"
            n_frames = data['frame'].nunique() if 'frame' in data.columns else None
        else:
            shape = ()
            dtype = type(data).__name__
            n_frames = None
            
        # Calculate number of tracks for trajectory data
        n_tracks = None
        if isinstance(data, pd.DataFrame) and 'track_number' in data.columns:
            n_tracks = data['track_number'].nunique()
            
        return DataInfo(
            name=name,
            data_type=data_type.value,
            shape=shape,
            dtype=dtype,
            file_path=str(file_path),
            n_tracks=n_tracks,
            n_frames=n_frames
        )
        
    def save_data(self, data_name: str, file_path: Union[str, Path], 
                  format: Optional[str] = None) -> bool:
        """Save data to file.
        
        Args:
            data_name: Name of data to save
            file_path: Output file path
            format: Output format (auto-detected if None)
            
        Returns:
            True if successful, False otherwise
        """
        if data_name not in self._data:
            self.logger.error(f"Data not found: {data_name}")
            return False
            
        data = self._data[data_name]
        file_path = Path(file_path)
        
        try:
            if isinstance(data, pd.DataFrame):
                if file_path.suffix.lower() == '.csv':
                    data.to_csv(file_path, index=False)
                elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                    data.to_excel(file_path, index=False)
                else:
                    data.to_csv(file_path, index=False)  # Default to CSV
                    
            elif isinstance(data, np.ndarray):
                if file_path.suffix.lower() in ['.tif', '.tiff']:
                    skio.imsave(str(file_path), data, plugin='tifffile')
                else:
                    skio.imsave(str(file_path), data)
                    
            self.logger.info(f"Saved data: {data_name} -> {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving {data_name}: {e}")
            return False
            
    def get_data(self, data_name: str) -> Optional[Any]:
        """Get data by name."""
        return self._data.get(data_name)
        
    def get_data_info(self, data_name: str) -> Optional[DataInfo]:
        """Get data metadata by name."""
        return self._data_info.get(data_name)
        
    def get_data_names(self, data_type: Optional[DataType] = None) -> List[str]:
        """Get list of data names, optionally filtered by type."""
        if data_type is None:
            return list(self._data.keys())
        else:
            return [name for name, info in self._data_info.items() 
                   if info.data_type == data_type.value]
                   
    def remove_data(self, data_name: str) -> bool:
        """Remove data from memory."""
        if data_name in self._data:
            del self._data[data_name]
            del self._data_info[data_name]
            self.dataRemoved.emit(data_name)
            self.logger.info(f"Removed data: {data_name}")
            return True
        return False
        
    def clear_all_data(self):
        """Clear all loaded data."""
        data_names = list(self._data.keys())
        for name in data_names:
            self.remove_data(name)
        self.logger.info("Cleared all data")
        
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage for each dataset."""
        usage = {}
        for name, data in self._data.items():
            if isinstance(data, np.ndarray):
                usage[name] = data.nbytes
            elif isinstance(data, pd.DataFrame):
                usage[name] = data.memory_usage(deep=True).sum()
            else:
                usage[name] = 0
        return usage
        
    def duplicate_data(self, source_name: str, new_name: str) -> bool:
        """Create a copy of existing data."""
        if source_name not in self._data:
            return False
            
        try:
            source_data = self._data[source_name]
            if isinstance(source_data, pd.DataFrame):
                new_data = source_data.copy()
            elif isinstance(source_data, np.ndarray):
                new_data = source_data.copy()
            else:
                new_data = source_data  # For immutable types
                
            self._data[new_name] = new_data
            
            # Copy and update metadata
            source_info = self._data_info[source_name]
            new_info = DataInfo(
                name=new_name,
                data_type=source_info.data_type,
                shape=source_info.shape,
                dtype=source_info.dtype,
                file_path=None,  # Not from file
                n_tracks=source_info.n_tracks,
                n_frames=source_info.n_frames,
                pixel_size=source_info.pixel_size,
                frame_rate=source_info.frame_rate
            )
            self._data_info[new_name] = new_info
            
            self.dataLoaded.emit(new_name, new_data)
            return True
            
        except Exception as e:
            self.logger.error(f"Error duplicating data: {e}")
            return False
