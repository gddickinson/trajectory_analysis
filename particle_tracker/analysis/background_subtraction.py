#!/usr/bin/env python3
"""
Background Subtraction Module
============================

Provides sophisticated background subtraction capabilities for particle tracking analysis,
including ROI-based background estimation and camera black level correction.

This module implements the functionality from Step_9_addBackgroundSubtractedIntensity.py
but in a more modular, app-integrated way without FLIKA dependency.
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple, Union
import numpy as np
import pandas as pd
import skimage.io as skio
from scipy import ndimage


class ROIManager:
    """Manages Region of Interest (ROI) files and operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supported_formats = ['.txt', '.csv', '.roi']
    
    def load_roi_file(self, roi_path: str) -> Dict[str, np.ndarray]:
        """
        Load ROI data from various file formats.
        
        Args:
            roi_path: Path to ROI file
            
        Returns:
            Dictionary mapping ROI names to intensity traces
        """
        roi_path = Path(roi_path)
        
        if not roi_path.exists():
            raise FileNotFoundError(f"ROI file not found: {roi_path}")
        
        self.logger.info(f"Loading ROI file: {roi_path}")
        
        if roi_path.suffix.lower() == '.txt':
            return self._load_roi_txt(roi_path)
        elif roi_path.suffix.lower() == '.csv':
            return self._load_roi_csv(roi_path)
        else:
            raise ValueError(f"Unsupported ROI file format: {roi_path.suffix}")
    
    def _load_roi_txt(self, roi_path: Path) -> Dict[str, np.ndarray]:
        """Load ROI data from text file (FLIKA format)."""
        try:
            # Try to load as simple numeric data first
            data = np.loadtxt(roi_path)
            
            # If it's a 1D array, treat it as a single ROI
            if data.ndim == 1:
                return {'roi_1': data}
            elif data.ndim == 2:
                # Multiple ROIs as columns
                roi_dict = {}
                for i in range(data.shape[1]):
                    roi_dict[f'roi_{i+1}'] = data[:, i]
                return roi_dict
            else:
                raise ValueError("Invalid ROI file format")
                
        except Exception as e:
            self.logger.error(f"Error loading ROI text file: {e}")
            # Try to parse as more complex format
            return self._parse_complex_roi_txt(roi_path)
    
    def _parse_complex_roi_txt(self, roi_path: Path) -> Dict[str, np.ndarray]:
        """Parse more complex ROI text file formats."""
        roi_dict = {}
        
        with open(roi_path, 'r') as f:
            lines = f.readlines()
        
        current_roi = None
        current_data = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Check if this is a ROI header
            if 'roi' in line.lower() or 'region' in line.lower():
                # Save previous ROI if exists
                if current_roi is not None and current_data:
                    roi_dict[current_roi] = np.array(current_data)
                
                # Start new ROI
                current_roi = line.replace(':', '').strip()
                current_data = []
            else:
                # Try to parse as numeric data
                try:
                    values = [float(x) for x in line.split()]
                    current_data.extend(values)
                except ValueError:
                    continue
        
        # Save last ROI
        if current_roi is not None and current_data:
            roi_dict[current_roi] = np.array(current_data)
        
        # If no ROI structure found, treat as single ROI
        if not roi_dict and current_data:
            roi_dict['roi_1'] = np.array(current_data)
        
        return roi_dict
    
    def _load_roi_csv(self, roi_path: Path) -> Dict[str, np.ndarray]:
        """Load ROI data from CSV file."""
        try:
            df = pd.read_csv(roi_path)
            roi_dict = {}
            
            for col in df.columns:
                if 'roi' in col.lower() or 'intensity' in col.lower():
                    roi_dict[col] = df[col].values
                else:
                    # Assume all numeric columns are ROIs
                    try:
                        roi_dict[col] = pd.to_numeric(df[col], errors='coerce').values
                    except:
                        continue
            
            return roi_dict
            
        except Exception as e:
            self.logger.error(f"Error loading ROI CSV file: {e}")
            raise
    
    def find_roi_file(self, data_file_path: str, roi_patterns: List[str] = None) -> Optional[str]:
        """
        Automatically find ROI file associated with a data file.
        
        Args:
            data_file_path: Path to the data file
            roi_patterns: List of patterns to search for ROI files
            
        Returns:
            Path to ROI file if found, None otherwise
        """
        data_path = Path(data_file_path)
        data_dir = data_path.parent
        base_name = data_path.stem
        
        # Default patterns based on your original scripts
        if roi_patterns is None:
            roi_patterns = [
                f"ROI_{base_name}.txt",
                f"{base_name}_ROI.txt", 
                f"{base_name.split('_bin')[0]}_ROI.txt",  # Handle binned files
                f"ROI_{base_name.split('_locs')[0]}.txt",  # Handle locs files
                "roi.txt",
                "ROI.txt"
            ]
        
        # Search for ROI files
        for pattern in roi_patterns:
            roi_path = data_dir / pattern
            if roi_path.exists():
                self.logger.info(f"Found ROI file: {roi_path}")
                return str(roi_path)
        
        # Also search in parent directory
        parent_dir = data_dir.parent
        for pattern in roi_patterns:
            roi_path = parent_dir / pattern
            if roi_path.exists():
                self.logger.info(f"Found ROI file in parent directory: {roi_path}")
                return str(roi_path)
        
        self.logger.warning(f"No ROI file found for {data_file_path}")
        return None


class BackgroundSubtractor:
    """Handles background subtraction operations for particle tracking data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.roi_manager = ROIManager()
    
    def calculate_camera_black_estimates(self, image_data: np.ndarray) -> np.ndarray:
        """
        Calculate camera black level estimates for each frame.
        
        Args:
            image_data: 3D array (frames, height, width) or 2D array (height, width)
            
        Returns:
            Array of minimum intensity values per frame
        """
        if len(image_data.shape) == 2:
            # Single frame
            return np.array([np.min(image_data)])
        elif len(image_data.shape) == 3:
            # Time series - calculate minimum for each frame
            return np.min(image_data, axis=(1, 2))
        else:
            raise ValueError(f"Unsupported image shape: {image_data.shape}")
    
    def load_image_data(self, image_path: str) -> np.ndarray:
        """
        Load image data from file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image array
        """
        try:
            image_data = skio.imread(image_path, plugin='tifffile')
            self.logger.info(f"Loaded image data with shape: {image_data.shape}")
            return image_data
        except Exception as e:
            self.logger.error(f"Error loading image data: {e}")
            raise
    
    def add_background_subtracted_intensity(self, 
                                          df: pd.DataFrame, 
                                          image_path: str = None,
                                          image_data: np.ndarray = None,
                                          roi_data: Dict[str, np.ndarray] = None,
                                          roi_path: str = None,
                                          method: str = 'roi_and_black') -> pd.DataFrame:
        """
        Add background-subtracted intensity columns to trajectory DataFrame.
        
        Args:
            df: DataFrame with trajectory data
            image_path: Path to image file (if image_data not provided)
            image_data: Pre-loaded image data (if image_path not provided)
            roi_data: Pre-loaded ROI data (if roi_path not provided)
            roi_path: Path to ROI file (if roi_data not provided)
            method: Background subtraction method ('roi', 'black', 'roi_and_black')
            
        Returns:
            DataFrame with added background subtraction columns
        """
        df = df.copy()
        
        # Load image data if not provided
        if image_data is None:
            if image_path is None:
                raise ValueError("Either image_path or image_data must be provided")
            image_data = self.load_image_data(image_path)
        
        # Calculate camera black estimates
        camera_estimates = self.calculate_camera_black_estimates(image_data)
        self.logger.info(f"Calculated camera black estimates for {len(camera_estimates)} frames")
        
        # Load ROI data if needed and not provided
        if method in ['roi', 'roi_and_black'] and roi_data is None:
            if roi_path is None:
                # Try to auto-find ROI file
                if image_path is not None:
                    roi_path = self.roi_manager.find_roi_file(image_path)
                    if roi_path is None:
                        self.logger.warning("No ROI file found, using camera black only")
                        method = 'black'
                        roi_data = {}
                    else:
                        roi_data = self.roi_manager.load_roi_file(roi_path)
                else:
                    self.logger.warning("No ROI path provided, using camera black only")
                    method = 'black'
                    roi_data = {}
            else:
                roi_data = self.roi_manager.load_roi_file(roi_path)
        
        # Add frame-based background data to DataFrame
        if 'frame' in df.columns:
            # Add camera black estimates
            frame_to_black = {}
            for frame_idx, black_value in enumerate(camera_estimates):
                frame_to_black[frame_idx] = black_value
            
            df['camera_black_estimate'] = df['frame'].map(frame_to_black)
            
            # Add ROI data if available
            if roi_data and method in ['roi', 'roi_and_black']:
                # Use the first ROI by default (can be extended for multiple ROIs)
                roi_name = list(roi_data.keys())[0]
                roi_trace = roi_data[roi_name]
                
                self.logger.info(f"Using ROI '{roi_name}' with {len(roi_trace)} values")
                
                # Map ROI values to frames
                frame_to_roi = {}
                for frame_idx, roi_value in enumerate(roi_trace):
                    if frame_idx < len(roi_trace):
                        frame_to_roi[frame_idx] = roi_value
                
                df['roi_1'] = df['frame'].map(frame_to_roi)
                
                # Calculate background-subtracted intensities
                if 'intensity' in df.columns:
                    # Method 1: Intensity - mean ROI
                    mean_roi = np.nanmean(roi_trace)
                    df['intensity_minus_mean_roi1'] = df['intensity'] - mean_roi
                    
                    # Method 2: Intensity - mean ROI - mean camera black
                    mean_black = np.nanmean(camera_estimates)
                    df['intensity_minus_mean_roi1_and_black'] = (df['intensity'] - 
                                                                mean_roi - 
                                                                mean_black)
                    
                    self.logger.info(f"Added background subtraction: mean ROI = {mean_roi:.2f}, "
                                   f"mean black = {mean_black:.2f}")
                else:
                    self.logger.warning("No 'intensity' column found in DataFrame")
            
            # Camera black only method
            elif method == 'black':
                if 'intensity' in df.columns:
                    mean_black = np.nanmean(camera_estimates)
                    df['intensity_minus_black'] = df['intensity'] - mean_black
                    self.logger.info(f"Added camera black subtraction: mean black = {mean_black:.2f}")
        
        else:
            self.logger.warning("No 'frame' column found in DataFrame")
        
        return df
    
    def subtract_background_batch(self, 
                                data_files: List[str], 
                                output_suffix: str = '_BGsubtract',
                                method: str = 'roi_and_black',
                                roi_patterns: List[str] = None) -> List[str]:
        """
        Process multiple files for background subtraction.
        
        Args:
            data_files: List of paths to data files
            output_suffix: Suffix for output files
            method: Background subtraction method
            roi_patterns: Patterns to search for ROI files
            
        Returns:
            List of output file paths
        """
        output_files = []
        
        for data_file in data_files:
            try:
                self.logger.info(f"Processing {data_file}")
                
                # Load trajectory data
                df = pd.read_csv(data_file)
                
                # Find corresponding image file
                image_path = self._find_image_file(data_file)
                if image_path is None:
                    self.logger.warning(f"No image file found for {data_file}, skipping")
                    continue
                
                # Find ROI file
                roi_path = self.roi_manager.find_roi_file(data_file, roi_patterns)
                
                # Process background subtraction
                result_df = self.add_background_subtracted_intensity(
                    df, image_path=image_path, roi_path=roi_path, method=method
                )
                
                # Save result
                output_path = self._generate_output_path(data_file, output_suffix)
                result_df.to_csv(output_path, index=False)
                output_files.append(output_path)
                
                self.logger.info(f"Saved background-subtracted data to {output_path}")
                
            except Exception as e:
                self.logger.error(f"Error processing {data_file}: {e}")
                continue
        
        return output_files
    
    def _find_image_file(self, data_file: str) -> Optional[str]:
        """Find the corresponding image file for a data file."""
        data_path = Path(data_file)
        data_dir = data_path.parent
        base_name = data_path.stem
        
        # Common patterns for image files
        patterns = [
            base_name.split('_locs')[0] + '.tif',
            base_name.split('_tracks')[0] + '.tif',
            base_name.split('_NNcount')[0].split('_locs')[0] + '.tif',
            base_name + '.tif',
            base_name.replace('_BGsubtract', '') + '.tif'
        ]
        
        for pattern in patterns:
            image_path = data_dir / pattern
            if image_path.exists():
                return str(image_path)
        
        return None
    
    def _generate_output_path(self, input_path: str, suffix: str) -> str:
        """Generate output path with suffix."""
        input_path = Path(input_path)
        return str(input_path.parent / f"{input_path.stem}{suffix}{input_path.suffix}")


class BackgroundSubtractionParameters:
    """Parameter container for background subtraction operations."""
    
    def __init__(self):
        self.method = 'roi_and_black'  # 'roi', 'black', 'roi_and_black'
        self.roi_patterns = None  # Auto-detect if None
        self.output_suffix = '_BGsubtract'
        self.auto_find_files = True
        self.save_intermediate = True
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'method': self.method,
            'roi_patterns': self.roi_patterns,
            'output_suffix': self.output_suffix,
            'auto_find_files': self.auto_find_files,
            'save_intermediate': self.save_intermediate
        }
    
    def from_dict(self, params: Dict[str, Any]):
        """Load from dictionary."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)


# Integration functions for the main particle tracking app

def add_background_subtraction_to_pipeline(analysis_engine, data_manager):
    """
    Add background subtraction capability to the existing analysis pipeline.
    
    Args:
        analysis_engine: The main analysis engine
        data_manager: The data manager instance
    """
    # Add background subtraction as a new analysis step
    from particle_tracker.core.analysis_engine import AnalysisStep
    
    # Extend AnalysisStep enum if needed
    if not hasattr(AnalysisStep, 'BACKGROUND_SUBTRACTION'):
        AnalysisStep.BACKGROUND_SUBTRACTION = "background_subtraction"
    
    # Add method to analysis engine
    def run_background_subtraction(self, data, parameters):
        """Run background subtraction analysis."""
        subtractor = BackgroundSubtractor()
        
        # Extract parameters
        bg_params = BackgroundSubtractionParameters()
        if 'background_subtraction' in parameters:
            bg_params.from_dict(parameters['background_subtraction'])
        
        # Determine data type and process accordingly
        if isinstance(data, pd.DataFrame):
            # Find associated image file
            if 'image_path' in parameters:
                image_path = parameters['image_path']
            else:
                # Try to infer from data or use data manager
                image_path = None
            
            # Process background subtraction
            result = subtractor.add_background_subtracted_intensity(
                data, 
                image_path=image_path,
                method=bg_params.method
            )
            
            return result
        else:
            raise ValueError("Background subtraction requires DataFrame input")
    
    # Bind method to analysis engine
    analysis_engine.run_background_subtraction = run_background_subtraction.__get__(analysis_engine)


def create_background_subtraction_widget():
    """Create a GUI widget for background subtraction parameters."""
    from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QComboBox, 
                                QLineEdit, QPushButton, QCheckBox, QFileDialog)
    from PyQt6.QtCore import pyqtSignal
    
    class BackgroundSubtractionWidget(QWidget):
        parametersChanged = pyqtSignal()
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self._setup_ui()
        
        def _setup_ui(self):
            layout = QVBoxLayout(self)
            form_layout = QFormLayout()
            
            # Method selection
            self.method_combo = QComboBox()
            self.method_combo.addItems(['roi_and_black', 'roi', 'black'])
            self.method_combo.currentTextChanged.connect(self.parametersChanged)
            form_layout.addRow("Method:", self.method_combo)
            
            # ROI file selection
            roi_layout = QVBoxLayout()
            self.roi_path_edit = QLineEdit()
            self.roi_browse_btn = QPushButton("Browse...")
            self.roi_browse_btn.clicked.connect(self._browse_roi_file)
            
            roi_layout.addWidget(self.roi_path_edit)
            roi_layout.addWidget(self.roi_browse_btn)
            form_layout.addRow("ROI File:", roi_layout)
            
            # Auto-find files option
            self.auto_find_cb = QCheckBox("Auto-find associated files")
            self.auto_find_cb.setChecked(True)
            self.auto_find_cb.toggled.connect(self.parametersChanged)
            form_layout.addRow("", self.auto_find_cb)
            
            layout.addLayout(form_layout)
        
        def _browse_roi_file(self):
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select ROI File", "",
                "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)"
            )
            if file_path:
                self.roi_path_edit.setText(file_path)
                self.parametersChanged.emit()
        
        def get_parameters(self) -> Dict[str, Any]:
            return {
                'method': self.method_combo.currentText(),
                'roi_path': self.roi_path_edit.text() if self.roi_path_edit.text() else None,
                'auto_find_files': self.auto_find_cb.isChecked()
            }
        
        def set_parameters(self, params: Dict[str, Any]):
            if 'method' in params:
                self.method_combo.setCurrentText(params['method'])
            if 'roi_path' in params and params['roi_path']:
                self.roi_path_edit.setText(params['roi_path'])
            if 'auto_find_files' in params:
                self.auto_find_cb.setChecked(params['auto_find_files'])
    
    return BackgroundSubtractionWidget


# Example usage and testing
def main():
    """Example usage of the background subtraction module."""
    import glob
    
    # Example: Process files similar to your original Step_9 script
    path = '/path/to/your/data'  # Update this path
    
    # Find files to process
    file_list = glob.glob(os.path.join(path, '**/*_NNcount.csv'), recursive=True)
    
    # Create background subtractor
    subtractor = BackgroundSubtractor()
    
    # Process files
    output_files = subtractor.subtract_background_batch(
        file_list, 
        output_suffix='_BGsubtract',
        method='roi_and_black'
    )
    
    print(f"Processed {len(output_files)} files")


if __name__ == "__main__":
    main()
