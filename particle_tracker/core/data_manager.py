#!/usr/bin/env python3
"""
Enhanced Data Manager Module
============================

Handles all data operations including loading, saving, and managing different
data types with support for hierarchical experiments, ROI data, batch processing,
and sophisticated analysis workflows.

Key Enhancements:
- Hierarchical experiment/condition/file structure
- ROI data management for background subtraction
- Batch processing support
- Enhanced metadata tracking
- Multiple export formats
- Data validation and quality checks
- Memory management for large datasets
- Background subtraction support
"""

import os
import json
import logging
import pickle
import zipfile
from pathlib import Path
from typing import Optional, Dict, List, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import shutil
import tempfile

import numpy as np
import pandas as pd
import skimage.io as skio
from PyQt6.QtCore import QObject, pyqtSignal, QThread, QTimer

# Try to import flika for ROI support
try:
    from flika.roi import open_rois
    FLIKA_AVAILABLE = True
except ImportError:
    FLIKA_AVAILABLE = False
    print("Warning: Flika not available. ROI functionality will be limited.")


@dataclass
class DataInfo:
    """Enhanced metadata about loaded data."""
    name: str
    data_type: str
    shape: Tuple[int, ...]
    dtype: str
    file_path: Optional[str] = None
    creation_time: Optional[str] = None
    modification_time: Optional[str] = None

    # Analysis-specific metadata
    n_tracks: Optional[int] = None
    n_frames: Optional[int] = None
    n_localizations: Optional[int] = None
    pixel_size: Optional[float] = None
    frame_rate: Optional[float] = None

    # Processing history
    processing_steps: List[str] = field(default_factory=list)
    parameters_used: Dict[str, Any] = field(default_factory=dict)

    # Quality metrics
    memory_usage_mb: Optional[float] = None
    has_missing_values: bool = False
    data_quality_score: Optional[float] = None

    # Hierarchical structure info
    experiment_name: Optional[str] = None
    condition_name: Optional[str] = None
    replicate_number: Optional[int] = None

    # Background subtraction info
    has_roi_data: bool = False
    roi_file_path: Optional[str] = None
    background_method: Optional[str] = None


@dataclass
class ROIData:
    """Container for ROI data used in background subtraction."""
    roi_traces: Dict[str, np.ndarray] = field(default_factory=dict)
    camera_estimates: Optional[np.ndarray] = None
    roi_file_path: Optional[str] = None
    roi_names: List[str] = field(default_factory=list)

    def get_primary_roi(self) -> Optional[np.ndarray]:
        """Get the primary ROI trace (usually the first one)."""
        if self.roi_traces and self.roi_names:
            return self.roi_traces.get(self.roi_names[0])
        return None


@dataclass
class ExperimentStructure:
    """Represents a hierarchical experiment structure."""
    experiment_name: str
    experiment_path: str
    conditions: Dict[str, List[str]] = field(default_factory=dict)  # condition -> list of files
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_total_files(self) -> int:
        """Get total number of files across all conditions."""
        return sum(len(files) for files in self.conditions.values())

    def get_condition_names(self) -> List[str]:
        """Get list of condition names."""
        return list(self.conditions.keys())


class DataType(Enum):
    """Enhanced enumeration of supported data types."""
    RAW_IMAGE = "raw_image"
    LOCALIZATIONS = "localizations"
    TRAJECTORIES = "trajectories"
    ANALYSIS_RESULTS = "analysis_results"
    BINARY_MASK = "binary_mask"
    ROI_DATA = "roi_data"
    BACKGROUND_DATA = "background_data"
    INTENSITY_TRACES = "intensity_traces"
    STATISTICS = "statistics"
    AUTOCORRELATION = "autocorrelation"
    DENSITY_ANALYSIS = "density_analysis"


class DataValidationError(Exception):
    """Exception raised when data validation fails."""
    pass


class BatchProcessor(QThread):
    """Thread for batch processing operations."""

    progressUpdate = pyqtSignal(str, int)  # message, percentage
    fileProcessed = pyqtSignal(str, bool)  # file_path, success
    batchCompleted = pyqtSignal(dict)  # results summary
    errorOccurred = pyqtSignal(str)

    def __init__(self, file_list: List[str], processing_function, **kwargs):
        super().__init__()
        self.file_list = file_list
        self.processing_function = processing_function
        self.kwargs = kwargs
        self.should_stop = False

    def run(self):
        """Run batch processing."""
        results = {"success": 0, "failed": 0, "errors": []}

        for i, file_path in enumerate(self.file_list):
            if self.should_stop:
                break

            try:
                self.progressUpdate.emit(f"Processing {Path(file_path).name}...",
                                       int(100 * i / len(self.file_list)))

                success = self.processing_function(file_path, **self.kwargs)

                if success:
                    results["success"] += 1
                    self.fileProcessed.emit(file_path, True)
                else:
                    results["failed"] += 1
                    self.fileProcessed.emit(file_path, False)

            except Exception as e:
                results["failed"] += 1
                error_msg = f"Error processing {file_path}: {str(e)}"
                results["errors"].append(error_msg)
                self.errorOccurred.emit(error_msg)
                self.fileProcessed.emit(file_path, False)

        self.progressUpdate.emit("Batch processing completed", 100)
        self.batchCompleted.emit(results)

    def stop(self):
        """Stop batch processing."""
        self.should_stop = True


class EnhancedDataManager(QObject):
    """Enhanced data manager with hierarchical structure support and advanced features."""

    # Signals
    dataLoaded = pyqtSignal(str, object)  # data_name, data
    dataChanged = pyqtSignal(str)  # data_name
    dataRemoved = pyqtSignal(str)  # data_name
    progressUpdate = pyqtSignal(str, int)  # message, percentage
    experimentLoaded = pyqtSignal(str)  # experiment_name
    batchProcessStarted = pyqtSignal(int)  # total_files
    batchProcessCompleted = pyqtSignal(dict)  # results
    memoryWarning = pyqtSignal(str)  # warning_message

    def __init__(self, max_memory_mb: int = 2048):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Data storage
        self._data: Dict[str, Any] = {}
        self._data_info: Dict[str, DataInfo] = {}
        self._roi_data: Dict[str, ROIData] = {}  # keyed by data name

        # Experiment structure
        self.current_experiment: Optional[ExperimentStructure] = None
        self.experiment_results: Dict[str, Dict[str, Any]] = {}  # condition -> results

        # Memory management
        self.max_memory_mb = max_memory_mb
        self.memory_check_timer = QTimer()
        self.memory_check_timer.timeout.connect(self._check_memory_usage)
        self.memory_check_timer.start(10000)  # Check every 10 seconds

        # Batch processing
        self.batch_processor: Optional[BatchProcessor] = None

        # Supported file formats (expanded)
        self.supported_formats = {
            'image': ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.nd2', '.lsm', '.czi'],
            'localization': ['.csv', '.txt', '.json', '.xlsx', '.xls'],
            'trajectory': ['.csv', '.txt', '.json', '.xlsx', '.xls'],
            'analysis': ['.csv', '.txt', '.json', '.xlsx', '.xls', '.pkl'],
            'roi': ['.txt', '.csv', '.json', '.roi', '.zip'],
            'project': ['.ptproj', '.ptp']
        }

        # Data validation rules
        self.validation_rules = {
            'localizations': {
                'required_columns': ['frame', 'x', 'y'],
                'optional_columns': ['intensity', 'id', 'sigma', 'chi2'],
                'min_rows': 1
            },
            'trajectories': {
                'required_columns': ['track_number', 'frame', 'x', 'y'],
                'optional_columns': ['intensity', 'id'],
                'min_rows': 1,
                'min_track_length': 2
            }
        }

        self.logger.info("Enhanced Data Manager initialized")

    def load_experiment_structure(self, experiment_path: str,
                                file_pattern: str = "*.csv") -> bool:
        """
        Load a hierarchical experiment structure.

        Args:
            experiment_path: Path to experiment directory containing condition folders
            file_pattern: Glob pattern for matching files

        Returns:
            True if successful, False otherwise
        """
        experiment_path = Path(experiment_path)

        if not experiment_path.exists() or not experiment_path.is_dir():
            self.logger.error(f"Experiment path not found: {experiment_path}")
            return False

        try:
            # Find condition folders
            condition_folders = [d for d in experiment_path.iterdir()
                               if d.is_dir() and not d.name.startswith('.')]

            if not condition_folders:
                self.logger.warning(f"No condition folders found in {experiment_path}")
                return False

            # Create experiment structure
            experiment = ExperimentStructure(
                experiment_name=experiment_path.name,
                experiment_path=str(experiment_path)
            )

            # Scan each condition folder for files
            for condition_folder in condition_folders:
                condition_name = condition_folder.name

                # Find matching files
                files = list(condition_folder.glob(file_pattern))
                file_paths = [str(f) for f in files]

                if file_paths:
                    experiment.conditions[condition_name] = file_paths
                    self.logger.info(f"Found {len(file_paths)} files in condition '{condition_name}'")

            if not experiment.conditions:
                self.logger.warning("No files found in any condition folders")
                return False

            self.current_experiment = experiment
            self.experimentLoaded.emit(experiment.experiment_name)

            self.logger.info(f"Loaded experiment '{experiment.experiment_name}' with "
                           f"{len(experiment.conditions)} conditions and "
                           f"{experiment.get_total_files()} total files")

            return True

        except Exception as e:
            self.logger.error(f"Error loading experiment structure: {e}")
            return False

    def load_roi_data(self, file_path: str, data_name: Optional[str] = None) -> bool:
        """
        Load ROI data for background subtraction.

        Args:
            file_path: Path to ROI file
            data_name: Associated data name (if None, use filename)

        Returns:
            True if successful, False otherwise
        """
        file_path = Path(file_path)

        if not file_path.exists():
            self.logger.error(f"ROI file not found: {file_path}")
            return False

        if data_name is None:
            data_name = file_path.stem

        try:
            roi_data = ROIData(roi_file_path=str(file_path))

            if file_path.suffix.lower() == '.txt':
                # Load Flika-style ROI text file
                if FLIKA_AVAILABLE:
                    try:
                        rois = open_rois(str(file_path))
                        for i, roi in enumerate(rois):
                            roi_name = f"roi_{i+1}"
                            trace = roi.getTrace()
                            roi_data.roi_traces[roi_name] = trace
                            roi_data.roi_names.append(roi_name)
                    except Exception as e:
                        self.logger.warning(f"Could not load ROI with Flika: {e}")
                        # Fallback to simple text loading
                        data = np.loadtxt(file_path)
                        roi_data.roi_traces['roi_1'] = data
                        roi_data.roi_names = ['roi_1']
                else:
                    # Simple text file loading
                    data = np.loadtxt(file_path)
                    roi_data.roi_traces['roi_1'] = data
                    roi_data.roi_names = ['roi_1']

            elif file_path.suffix.lower() == '.csv':
                # Load CSV ROI file
                df = pd.read_csv(file_path)
                for i, col in enumerate(df.columns):
                    roi_name = f"roi_{i+1}" if col.startswith('roi') else col
                    roi_data.roi_traces[roi_name] = df[col].values
                    roi_data.roi_names.append(roi_name)

            else:
                self.logger.error(f"Unsupported ROI file format: {file_path.suffix}")
                return False

            self._roi_data[data_name] = roi_data
            self.logger.info(f"Loaded ROI data for '{data_name}' with {len(roi_data.roi_names)} ROIs")

            return True

        except Exception as e:
            self.logger.error(f"Error loading ROI data: {e}")
            return False

    def load_file(self, file_path: Union[str, Path], data_name: Optional[str] = None,
                  data_type: Optional[DataType] = None,
                  validate_data: bool = True, **kwargs) -> bool:
        """
        Enhanced file loading with validation and metadata tracking.

        Args:
            file_path: Path to the file
            data_name: Name for the data (defaults to filename)
            data_type: Type of data being loaded
            validate_data: Whether to validate the loaded data
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
            elif data_type == DataType.ROI_DATA:
                return self.load_roi_data(str(file_path), data_name)
            else:
                self.logger.error(f"Unsupported data type: {data_type}")
                return False

            # Validate data if requested
            if validate_data and data_type in [DataType.LOCALIZATIONS, DataType.TRAJECTORIES]:
                try:
                    self._validate_data(data, data_type)
                except DataValidationError as e:
                    self.logger.warning(f"Data validation warning for {file_path}: {e}")
                    # Continue loading despite validation warnings

            # Create enhanced metadata
            data_info = self._create_enhanced_data_info(
                data_name, data, data_type, file_path, **kwargs
            )

            # Store data and metadata
            self._data[data_name] = data
            self._data_info[data_name] = data_info

            # Check for associated ROI file
            self._check_for_roi_file(file_path, data_name)

            # Update memory usage
            self._update_memory_usage()

            self.progressUpdate.emit("Loading complete", 100)
            self.dataLoaded.emit(data_name, data)
            self.logger.info(f"Successfully loaded: {data_name}")

            return True

        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            self.progressUpdate.emit("Loading failed", 0)
            return False

    def _validate_data(self, data: Any, data_type: DataType):
        """
        Validate loaded data according to predefined rules.

        Args:
            data: Loaded data
            data_type: Type of the data

        Raises:
            DataValidationError: If validation fails
        """
        if not isinstance(data, pd.DataFrame):
            return  # Only validate DataFrame data

        validation_key = data_type.value
        if validation_key not in self.validation_rules:
            return

        rules = self.validation_rules[validation_key]

        # Check minimum rows
        if len(data) < rules.get('min_rows', 1):
            raise DataValidationError(f"Data has {len(data)} rows, minimum required: {rules['min_rows']}")

        # Check required columns
        required_cols = rules.get('required_columns', [])
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise DataValidationError(f"Missing required columns: {missing_cols}")

        # Type-specific validation
        if data_type == DataType.TRAJECTORIES:
            min_track_length = rules.get('min_track_length', 2)
            if 'track_number' in data.columns:
                track_lengths = data.groupby('track_number').size()
                short_tracks = track_lengths[track_lengths < min_track_length]
                if len(short_tracks) > 0:
                    self.logger.warning(f"Found {len(short_tracks)} tracks shorter than {min_track_length} points")

    def _check_for_roi_file(self, data_file_path: Path, data_name: str):
        """
        Check for associated ROI file and load it automatically.

        Args:
            data_file_path: Path to the data file
            data_name: Name of the loaded data
        """
        # Common ROI file naming patterns
        patterns = [
            f"ROI_{data_file_path.stem}.txt",
            f"{data_file_path.stem}_ROI.txt",
            f"roi_{data_file_path.stem}.txt",
            f"{data_file_path.stem}.roi"
        ]

        roi_dir = data_file_path.parent

        for pattern in patterns:
            roi_path = roi_dir / pattern
            if roi_path.exists():
                self.logger.info(f"Found associated ROI file: {roi_path}")
                if self.load_roi_data(str(roi_path), data_name):
                    # Update data info to indicate ROI data is available
                    if data_name in self._data_info:
                        self._data_info[data_name].has_roi_data = True
                        self._data_info[data_name].roi_file_path = str(roi_path)
                break

    def _create_enhanced_data_info(self, name: str, data: Any, data_type: DataType,
                                 file_path: Path, **kwargs) -> DataInfo:
        """Create enhanced metadata for loaded data."""
        # Basic info
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

        # Calculate statistics
        n_tracks = None
        n_localizations = None
        has_missing_values = False

        if isinstance(data, pd.DataFrame):
            if 'track_number' in data.columns:
                n_tracks = data['track_number'].nunique()
            n_localizations = len(data)
            has_missing_values = data.isnull().any().any()

        # Memory usage
        memory_usage_mb = self._calculate_memory_usage(data)

        # File timestamps
        stat = file_path.stat()
        creation_time = datetime.fromtimestamp(stat.st_ctime).isoformat()
        modification_time = datetime.fromtimestamp(stat.st_mtime).isoformat()

        # Extract hierarchical info from path if in experiment structure
        experiment_name = None
        condition_name = None
        if self.current_experiment:
            if str(file_path).startswith(self.current_experiment.experiment_path):
                experiment_name = self.current_experiment.experiment_name
                # Try to determine condition from path
                for condition, files in self.current_experiment.conditions.items():
                    if str(file_path) in files:
                        condition_name = condition
                        break

        return DataInfo(
            name=name,
            data_type=data_type.value,
            shape=shape,
            dtype=dtype,
            file_path=str(file_path),
            creation_time=creation_time,
            modification_time=modification_time,
            n_tracks=n_tracks,
            n_frames=n_frames,
            n_localizations=n_localizations,
            pixel_size=kwargs.get('pixel_size'),
            frame_rate=kwargs.get('frame_rate'),
            memory_usage_mb=memory_usage_mb,
            has_missing_values=has_missing_values,
            experiment_name=experiment_name,
            condition_name=condition_name
        )

    def _calculate_memory_usage(self, data: Any) -> float:
        """Calculate memory usage of data in MB."""
        if isinstance(data, np.ndarray):
            return data.nbytes / (1024 * 1024)
        elif isinstance(data, pd.DataFrame):
            return data.memory_usage(deep=True).sum() / (1024 * 1024)
        else:
            return 0.0

    def _update_memory_usage(self):
        """Update total memory usage and check limits."""
        total_memory = sum(info.memory_usage_mb or 0 for info in self._data_info.values())

        if total_memory > self.max_memory_mb:
            warning_msg = f"Memory usage ({total_memory:.1f} MB) exceeds limit ({self.max_memory_mb} MB)"
            self.logger.warning(warning_msg)
            self.memoryWarning.emit(warning_msg)

    def _check_memory_usage(self):
        """Periodic memory usage check."""
        self._update_memory_usage()

    def get_data_names(self) -> List[str]:
        """Get list of all loaded data names."""
        return list(self._data.keys())

    def get_data(self, data_name: str) -> Any:
        """Get data by name."""
        return self._data.get(data_name)

    def remove_data(self, data_name: str) -> bool:
        """Remove data by name."""
        if data_name in self._data:
            del self._data[data_name]
            if data_name in self._data_info:
                del self._data_info[data_name]
            if data_name in self._roi_data:
                del self._roi_data[data_name]
            self.dataRemoved.emit(data_name)
            return True
        return False


    def start_batch_processing(self, condition_name: str = None,
                             processing_function: callable = None,
                             **processing_kwargs) -> bool:
        """
        Start batch processing of files.

        Args:
            condition_name: Specific condition to process (None for all)
            processing_function: Function to process each file
            **processing_kwargs: Arguments for processing function

        Returns:
            True if batch processing started successfully
        """
        if not self.current_experiment:
            self.logger.error("No experiment loaded for batch processing")
            return False

        if processing_function is None:
            self.logger.error("No processing function provided")
            return False

        # Get files to process
        files_to_process = []
        if condition_name:
            if condition_name in self.current_experiment.conditions:
                files_to_process = self.current_experiment.conditions[condition_name]
            else:
                self.logger.error(f"Condition '{condition_name}' not found")
                return False
        else:
            # Process all files
            for condition_files in self.current_experiment.conditions.values():
                files_to_process.extend(condition_files)

        if not files_to_process:
            self.logger.warning("No files to process")
            return False

        # Start batch processor
        self.batch_processor = BatchProcessor(
            files_to_process, processing_function, **processing_kwargs
        )

        # Connect signals
        self.batch_processor.progressUpdate.connect(self.progressUpdate)
        self.batch_processor.batchCompleted.connect(self.batchProcessCompleted)
        self.batch_processor.fileProcessed.connect(
            lambda path, success: self.logger.info(f"Processed {Path(path).name}: {'Success' if success else 'Failed'}")
        )

        self.batchProcessStarted.emit(len(files_to_process))
        self.batch_processor.start()

        return True

    def stop_batch_processing(self):
        """Stop current batch processing."""
        if self.batch_processor and self.batch_processor.isRunning():
            self.batch_processor.stop()
            self.batch_processor.wait()

    def get_roi_data(self, data_name: str) -> Optional[ROIData]:
        """Get ROI data for a dataset."""
        return self._roi_data.get(data_name)

    def apply_background_subtraction(self, data_name: str,
                                   method: str = "roi_mean") -> bool:
        """
        Apply background subtraction to intensity data.

        Args:
            data_name: Name of the dataset
            method: Background subtraction method

        Returns:
            True if successful
        """
        if data_name not in self._data:
            self.logger.error(f"Data '{data_name}' not found")
            return False

        data = self._data[data_name]
        roi_data = self._roi_data.get(data_name)

        if not isinstance(data, pd.DataFrame):
            self.logger.error("Background subtraction only supported for DataFrame data")
            return False

        if 'intensity' not in data.columns:
            self.logger.error("No intensity column found for background subtraction")
            return False

        try:
            if method == "roi_mean" and roi_data:
                # Use ROI mean for background subtraction
                primary_roi = roi_data.get_primary_roi()
                if primary_roi is not None:
                    roi_mean = np.mean(primary_roi)
                    data['intensity_bg_subtracted'] = data['intensity'] - roi_mean
                    self.logger.info(f"Applied ROI mean background subtraction (mean={roi_mean:.2f})")
                else:
                    self.logger.error("No ROI data available for background subtraction")
                    return False

            elif method == "frame_min":
                # Use per-frame minimum
                frame_mins = data.groupby('frame')['intensity'].min()
                data['intensity_bg_subtracted'] = data.apply(
                    lambda row: row['intensity'] - frame_mins[row['frame']], axis=1
                )
                self.logger.info("Applied per-frame minimum background subtraction")

            else:
                self.logger.error(f"Unknown background subtraction method: {method}")
                return False

            # Update data info
            if data_name in self._data_info:
                self._data_info[data_name].processing_steps.append(f"background_subtraction_{method}")
                self._data_info[data_name].background_method = method

            self.dataChanged.emit(data_name)
            return True

        except Exception as e:
            self.logger.error(f"Error in background subtraction: {e}")
            return False

    def export_experiment_summary(self, output_path: str) -> bool:
        """
        Export a summary of the current experiment.

        Args:
            output_path: Path for the summary file

        Returns:
            True if successful
        """
        if not self.current_experiment:
            self.logger.error("No experiment loaded")
            return False

        try:
            summary = {
                "experiment_name": self.current_experiment.experiment_name,
                "experiment_path": self.current_experiment.experiment_path,
                "created": datetime.now().isoformat(),
                "conditions": {},
                "total_files": self.current_experiment.get_total_files(),
                "total_memory_mb": sum(info.memory_usage_mb or 0 for info in self._data_info.values())
            }

            # Add condition summaries
            for condition, files in self.current_experiment.conditions.items():
                condition_summary = {
                    "file_count": len(files),
                    "files": [str(Path(f).name) for f in files],
                    "loaded_datasets": []
                }

                # Find loaded datasets for this condition
                for data_name, info in self._data_info.items():
                    if info.condition_name == condition:
                        condition_summary["loaded_datasets"].append({
                            "name": data_name,
                            "type": info.data_type,
                            "n_tracks": info.n_tracks,
                            "n_frames": info.n_frames,
                            "memory_mb": info.memory_usage_mb
                        })

                summary["conditions"][condition] = condition_summary

            # Save summary
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)

            self.logger.info(f"Exported experiment summary to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting experiment summary: {e}")
            return False

    def get_experiment_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the current experiment."""
        if not self.current_experiment:
            return {}

        stats = {
            "experiment_name": self.current_experiment.experiment_name,
            "n_conditions": len(self.current_experiment.conditions),
            "total_files": self.current_experiment.get_total_files(),
            "loaded_datasets": len(self._data),
            "total_memory_mb": sum(info.memory_usage_mb or 0 for info in self._data_info.values()),
            "conditions": {}
        }

        # Per-condition statistics
        for condition in self.current_experiment.conditions:
            condition_datasets = [
                info for info in self._data_info.values()
                if info.condition_name == condition
            ]

            condition_stats = {
                "n_files": len(self.current_experiment.conditions[condition]),
                "n_loaded": len(condition_datasets),
                "total_tracks": sum(info.n_tracks or 0 for info in condition_datasets),
                "total_localizations": sum(info.n_localizations or 0 for info in condition_datasets),
                "memory_mb": sum(info.memory_usage_mb or 0 for info in condition_datasets)
            }

            stats["conditions"][condition] = condition_stats

        return stats

    # Override parent methods with enhancements
    def _detect_data_type(self, file_path: Path) -> DataType:
        """Enhanced auto-detection of data type."""
        ext = file_path.suffix.lower()

        if ext in self.supported_formats['image']:
            return DataType.RAW_IMAGE
        elif ext in self.supported_formats['roi']:
            return DataType.ROI_DATA
        elif ext in ['.csv', '.txt', '.xlsx', '.xls']:
            # Try to determine from content or filename
            filename_lower = file_path.name.lower()

            if any(keyword in filename_lower for keyword in ['roi', 'background']):
                return DataType.ROI_DATA
            elif any(keyword in filename_lower for keyword in ['track', 'trajectory']):
                return DataType.TRAJECTORIES
            elif any(keyword in filename_lower for keyword in ['loc', 'detection']):
                return DataType.LOCALIZATIONS
            elif any(keyword in filename_lower for keyword in ['result', 'analysis', 'svm', 'classification']):
                return DataType.ANALYSIS_RESULTS
            else:
                # Try to read a few lines and detect based on columns
                try:
                    df = pd.read_csv(file_path, nrows=5)
                    if 'track_number' in df.columns:
                        return DataType.TRAJECTORIES
                    elif any(col in df.columns for col in ['x', 'y', 'frame']):
                        return DataType.LOCALIZATIONS
                    else:
                        return DataType.ANALYSIS_RESULTS
                except:
                    return DataType.ANALYSIS_RESULTS
        else:
            return DataType.ANALYSIS_RESULTS

    def _load_image(self, file_path: Path, **kwargs) -> np.ndarray:
        """Enhanced image loading with better format support."""
        try:
            # Use skimage for most formats
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
        """Enhanced tabular data loading with better error handling."""
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
            elif file_path.suffix.lower() == '.txt':
                # Try as tab-separated first, then comma-separated
                try:
                    df = pd.read_csv(file_path, sep='\t', **kwargs)
                except:
                    df = pd.read_csv(file_path, sep=',', **kwargs)
            else:
                df = pd.read_csv(file_path, **kwargs)  # Try as CSV

            self.logger.info(f"Loaded tabular data: {len(df)} rows, {len(df.columns)} columns")
            return df

        except Exception as e:
            self.logger.error(f"Error loading tabular data {file_path}: {e}")
            raise

    def save_data(self, data_name: str, file_path: Union[str, Path],
                  format: Optional[str] = None, include_metadata: bool = True) -> bool:
        """Enhanced data saving with metadata preservation."""
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
                    # Save with metadata if requested
                    if include_metadata and data_name in self._data_info:
                        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                            data.to_excel(writer, sheet_name='Data', index=False)

                            # Add metadata sheet
                            info = self._data_info[data_name]
                            metadata_df = pd.DataFrame([{
                                'Property': k,
                                'Value': str(v)
                            } for k, v in info.__dict__.items() if v is not None])
                            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
                    else:
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

    # Memory management methods
    def cleanup_unused_data(self, keep_recent: int = 5):
        """Remove least recently used data to free memory."""
        if len(self._data) <= keep_recent:
            return

        # Sort by modification time (or creation time if no modification)
        data_items = [(name, info) for name, info in self._data_info.items()]
        data_items.sort(key=lambda x: x[1].modification_time or x[1].creation_time)

        # Remove oldest items
        to_remove = data_items[:-keep_recent]
        for name, _ in to_remove:
            self.remove_data(name)
            self.logger.info(f"Removed old data '{name}' to free memory")

    def get_memory_report(self) -> Dict[str, Any]:
        """Get detailed memory usage report."""
        total_memory = 0
        data_memory = {}

        for name, info in self._data_info.items():
            memory_mb = info.memory_usage_mb or 0
            total_memory += memory_mb
            data_memory[name] = {
                'memory_mb': memory_mb,
                'data_type': info.data_type,
                'shape': info.shape
            }

        return {
            'total_memory_mb': total_memory,
            'max_memory_mb': self.max_memory_mb,
            'memory_usage_pct': (total_memory / self.max_memory_mb) * 100,
            'data_memory': data_memory
        }
