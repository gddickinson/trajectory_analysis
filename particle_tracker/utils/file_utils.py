#!/usr/bin/env python3
"""
Enhanced File Utilities Module
==============================

Comprehensive file handling utilities for particle tracking analysis,
supporting batch processing, multiple file formats, ROI management,
and hierarchical experiment structures.
"""

import os
import json
import logging
import logging.handlers
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Generator
from dataclasses import dataclass, asdict
from datetime import datetime
import pickle
import glob
from collections import defaultdict

import pandas as pd
import numpy as np


# ============================================================================
# ENHANCED FILE TYPE DETECTION AND HANDLING
# ============================================================================

@dataclass
class FileInfo:
    """Enhanced file information container."""
    path: Path
    size_bytes: int
    created: datetime
    modified: datetime
    file_type: str
    format_type: str  # 'image', 'trajectory', 'analysis', 'roi', 'project'
    estimated_tracks: Optional[int] = None
    estimated_frames: Optional[int] = None
    estimated_localizations: Optional[int] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EnhancedFileManager:
    """Enhanced file manager with support for multiple formats and batch operations."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Supported file formats organized by category
        self.supported_formats = {
            'image': {
                'extensions': ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'],
                'patterns': ['*_bin*.tif', '*_crop*.tif', '*_piezo*.tif', '*MMStack*.tif']
            },
            'trajectory': {
                'extensions': ['.csv', '.txt', '.json', '.xlsx', '.xls'],
                'patterns': ['*_tracks*.csv', '*_locsID*.csv', '*_locs.csv'],
                'required_columns': [['track_number', 'frame', 'x', 'y'],
                                   ['particle', 'frame', 'x', 'y'],
                                   ['id', 'frame', 'x [nm]', 'y [nm]']]
            },
            'analysis': {
                'extensions': ['.csv', '.txt', '.xlsx', '.xls'],
                'patterns': ['*_SVMPredicted*.csv', '*_features*.csv', '*_metrics*.csv',
                           '*_RG*.csv', '*_diffusion*.csv', '*_velocity*.csv']
            },
            'roi': {
                'extensions': ['.txt', '.csv', '.roi', '.zip'],
                'patterns': ['ROI_*.txt', '*_ROI.txt', '*_roi*.csv']
            },
            'project': {
                'extensions': ['.ptproj', '.ptp', '.json']
            }
        }

    def get_file_info(self, file_path: Union[str, Path]) -> FileInfo:
        """Get comprehensive file information."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        stat = file_path.stat()

        # Basic file info
        file_info = FileInfo(
            path=file_path,
            size_bytes=stat.st_size,
            created=datetime.fromtimestamp(stat.st_ctime),
            modified=datetime.fromtimestamp(stat.st_mtime),
            file_type=file_path.suffix.lower(),
            format_type=self._detect_format_type(file_path)
        )

        # Try to get additional metadata
        try:
            if file_info.format_type in ['trajectory', 'analysis']:
                self._analyze_tabular_file(file_info)
            elif file_info.format_type == 'roi':
                self._analyze_roi_file(file_info)
        except Exception as e:
            self.logger.debug(f"Could not analyze file metadata for {file_path}: {e}")

        return file_info

    def _detect_format_type(self, file_path: Path) -> str:
        """Detect the format type of a file."""
        file_name = file_path.name.lower()
        extension = file_path.suffix.lower()

        # Check each format category
        for format_type, config in self.supported_formats.items():
            # Check extension
            if extension in config['extensions']:
                # Check patterns if available
                if 'patterns' in config:
                    for pattern in config['patterns']:
                        if file_path.match(pattern.lower()):
                            return format_type
                else:
                    return format_type

        return 'unknown'

    def _analyze_tabular_file(self, file_info: FileInfo):
        """Analyze tabular files to extract metadata."""
        try:
            # Quick sample read to get basic info
            if file_info.file_type == '.csv':
                df = pd.read_csv(file_info.path, nrows=10)
            elif file_info.file_type in ['.xlsx', '.xls']:
                df = pd.read_excel(file_info.path, nrows=10)
            else:
                return

            file_info.metadata['columns'] = list(df.columns)
            file_info.metadata['sample_rows'] = len(df)

            # Estimate full file size
            if 'track_number' in df.columns:
                file_info.estimated_tracks = df['track_number'].nunique()
            elif 'particle' in df.columns:
                file_info.estimated_tracks = df['particle'].nunique()

            if 'frame' in df.columns:
                file_info.estimated_frames = df['frame'].nunique()

            # Rough estimation of total rows
            file_info.estimated_localizations = int(file_info.size_bytes / 100)  # Rough estimate

        except Exception as e:
            self.logger.debug(f"Error analyzing tabular file: {e}")

    def _analyze_roi_file(self, file_info: FileInfo):
        """Analyze ROI files to extract metadata."""
        try:
            if file_info.file_type == '.txt':
                with open(file_info.path, 'r') as f:
                    lines = f.readlines()
                    file_info.metadata['roi_count'] = len([l for l in lines if l.strip()])
            elif file_info.file_type == '.csv':
                df = pd.read_csv(file_info.path, nrows=5)
                file_info.metadata['columns'] = list(df.columns)
        except Exception as e:
            self.logger.debug(f"Error analyzing ROI file: {e}")


# ============================================================================
# BATCH PROCESSING UTILITIES
# ============================================================================

class BatchFileProcessor:
    """Enhanced batch file processor with hierarchical experiment support."""

    def __init__(self, file_manager: Optional[EnhancedFileManager] = None):
        self.logger = logging.getLogger(__name__)
        self.file_manager = file_manager or EnhancedFileManager()

    def discover_experiment_structure(self, root_path: Union[str, Path]) -> Dict[str, Any]:
        """Discover hierarchical experiment structure (experiment → conditions → files)."""
        root_path = Path(root_path)

        if not root_path.exists():
            raise FileNotFoundError(f"Root path not found: {root_path}")

        structure = {
            'root_path': root_path,
            'experiment_name': root_path.name,
            'conditions': {},
            'total_files': 0,
            'total_size_bytes': 0
        }

        # Look for condition folders
        for item in root_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                condition_info = self._analyze_condition_folder(item)
                if condition_info['files']:  # Only include folders with relevant files
                    structure['conditions'][item.name] = condition_info
                    structure['total_files'] += condition_info['file_count']
                    structure['total_size_bytes'] += condition_info['total_size_bytes']

        # If no condition folders found, treat root as single condition
        if not structure['conditions']:
            root_condition = self._analyze_condition_folder(root_path)
            if root_condition['files']:
                structure['conditions']['root'] = root_condition
                structure['total_files'] = root_condition['file_count']
                structure['total_size_bytes'] = root_condition['total_size_bytes']

        return structure

    def _analyze_condition_folder(self, folder_path: Path) -> Dict[str, Any]:
        """Analyze a single condition folder."""
        condition_info = {
            'path': folder_path,
            'files': defaultdict(list),
            'file_count': 0,
            'total_size_bytes': 0,
            'roi_files': [],
            'output_exists': False
        }

        # Scan for relevant files
        for file_path in folder_path.rglob('*'):
            if file_path.is_file() and not file_path.name.startswith('.'):
                try:
                    file_info = self.file_manager.get_file_info(file_path)

                    if file_info.format_type != 'unknown':
                        condition_info['files'][file_info.format_type].append(file_info)
                        condition_info['file_count'] += 1
                        condition_info['total_size_bytes'] += file_info.size_bytes

                        if file_info.format_type == 'roi':
                            condition_info['roi_files'].append(file_info)

                except Exception as e:
                    self.logger.debug(f"Error analyzing file {file_path}: {e}")

        # Check for existing output folders
        output_folders = ['autocorrelation_output', 'results', 'analysis_output']
        condition_info['output_exists'] = any(
            (folder_path / folder).exists() for folder in output_folders
        )

        return condition_info

    def get_file_pairs(self, condition_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get matched file pairs (data + ROI) for analysis."""
        pairs = []

        # Get trajectory/analysis files
        data_files = []
        data_files.extend(condition_info['files'].get('trajectory', []))
        data_files.extend(condition_info['files'].get('analysis', []))

        roi_files = condition_info['files'].get('roi', [])

        for data_file in data_files:
            pair = {
                'data_file': data_file,
                'roi_file': None,
                'image_file': None,
                'base_name': self._get_base_name(data_file.path)
            }

            # Try to find matching ROI file
            for roi_file in roi_files:
                if self._files_match(data_file.path, roi_file.path):
                    pair['roi_file'] = roi_file
                    break

            # Try to find matching image file
            image_files = condition_info['files'].get('image', [])
            for image_file in image_files:
                if self._files_match(data_file.path, image_file.path):
                    pair['image_file'] = image_file
                    break

            pairs.append(pair)

        return pairs

    def _get_base_name(self, file_path: Path) -> str:
        """Extract base name from file path, removing common suffixes."""
        name = file_path.stem

        # Remove common suffixes
        suffixes_to_remove = [
            '_locsID', '_tracks', '_locs', '_SVMPredicted', '_features',
            '_metrics', '_RG', '_diffusion', '_velocity', '_NN', '_AllLocs',
            '_BGsubtract', '_trapped-AllFrames', '_bin10', '_bin20', '_crop100'
        ]

        for suffix in suffixes_to_remove:
            if suffix in name:
                name = name.split(suffix)[0]
                break

        return name

    def _files_match(self, file1: Path, file2: Path) -> bool:
        """Check if two files are related based on naming patterns."""
        base1 = self._get_base_name(file1)
        base2 = self._get_base_name(file2)

        # Direct match
        if base1 == base2:
            return True

        # Check if one is contained in the other
        if base1 in base2 or base2 in base1:
            return True

        # Check for common patterns like ROI_filename.txt
        if file2.name.startswith('ROI_') and base1 in file2.name:
            return True

        return False


# ============================================================================
# ROI FILE HANDLING
# ============================================================================

class ROIManager:
    """Manager for ROI (Region of Interest) files."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def read_roi_file(self, roi_path: Union[str, Path]) -> Dict[str, Any]:
        """Read ROI file and return ROI data."""
        roi_path = Path(roi_path)

        if not roi_path.exists():
            raise FileNotFoundError(f"ROI file not found: {roi_path}")

        if roi_path.suffix.lower() == '.txt':
            return self._read_roi_txt(roi_path)
        elif roi_path.suffix.lower() == '.csv':
            return self._read_roi_csv(roi_path)
        else:
            raise ValueError(f"Unsupported ROI file format: {roi_path.suffix}")

    def _read_roi_txt(self, roi_path: Path) -> Dict[str, Any]:
        """Read ROI data from text file."""
        try:
            with open(roi_path, 'r') as f:
                lines = f.readlines()

            # Parse the ROI data (assuming it's trace data)
            data = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        data.append(float(line))
                    except ValueError:
                        continue

            return {
                'type': 'trace',
                'data': np.array(data),
                'length': len(data),
                'source_file': str(roi_path)
            }

        except Exception as e:
            self.logger.error(f"Error reading ROI txt file {roi_path}: {e}")
            raise

    def _read_roi_csv(self, roi_path: Path) -> Dict[str, Any]:
        """Read ROI data from CSV file."""
        try:
            df = pd.read_csv(roi_path)

            return {
                'type': 'table',
                'data': df,
                'columns': list(df.columns),
                'length': len(df),
                'source_file': str(roi_path)
            }

        except Exception as e:
            self.logger.error(f"Error reading ROI csv file {roi_path}: {e}")
            raise

    def find_roi_for_file(self, data_file: Union[str, Path],
                         search_dir: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """Find corresponding ROI file for a data file."""
        data_file = Path(data_file)

        if search_dir is None:
            search_dir = data_file.parent
        else:
            search_dir = Path(search_dir)

        # Extract base name
        base_name = data_file.stem
        for suffix in ['_locsID', '_tracks', '_locs', '_bin10', '_bin20']:
            if suffix in base_name:
                base_name = base_name.split(suffix)[0]
                break

        # Try different ROI naming patterns
        roi_patterns = [
            f"ROI_{base_name}.txt",
            f"{base_name}_ROI.txt",
            f"{base_name}_roi.txt",
            f"roi_{base_name}.txt"
        ]

        for pattern in roi_patterns:
            roi_path = search_dir / pattern
            if roi_path.exists():
                return roi_path

        # Try recursive search in subdirectories
        for roi_file in search_dir.rglob("ROI_*.txt"):
            if base_name in roi_file.name:
                return roi_file

        return None


# ============================================================================
# EXPORT MANAGEMENT
# ============================================================================

class ExportManager:
    """Enhanced export manager with multiple format support."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def export_dataframe(self, df: pd.DataFrame, output_path: Union[str, Path],
                        format: str = 'csv', **kwargs) -> bool:
        """Export DataFrame to various formats."""
        output_path = Path(output_path)

        try:
            if format.lower() == 'csv':
                df.to_csv(output_path, index=False, **kwargs)
            elif format.lower() in ['xlsx', 'excel']:
                df.to_excel(output_path, index=False, **kwargs)
            elif format.lower() == 'json':
                df.to_json(output_path, orient='records', **kwargs)
            elif format.lower() == 'parquet':
                df.to_parquet(output_path, **kwargs)
            else:
                raise ValueError(f"Unsupported export format: {format}")

            self.logger.info(f"Exported DataFrame to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting DataFrame: {e}")
            return False

    def create_analysis_package(self, results: Dict[str, Any],
                               output_dir: Union[str, Path]) -> bool:
        """Create a comprehensive analysis package with all results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Create subdirectories
            (output_dir / 'data').mkdir(exist_ok=True)
            (output_dir / 'plots').mkdir(exist_ok=True)
            (output_dir / 'statistics').mkdir(exist_ok=True)
            (output_dir / 'metadata').mkdir(exist_ok=True)

            # Export data files
            if 'dataframes' in results:
                for name, df in results['dataframes'].items():
                    self.export_dataframe(df, output_dir / 'data' / f"{name}.csv")

            # Export plots
            if 'plots' in results:
                for name, plot_data in results['plots'].items():
                    if hasattr(plot_data, 'savefig'):  # matplotlib figure
                        plot_data.savefig(output_dir / 'plots' / f"{name}.png", dpi=300)
                        plot_data.savefig(output_dir / 'plots' / f"{name}.pdf")

            # Export statistics
            if 'statistics' in results:
                with open(output_dir / 'statistics' / 'summary.json', 'w') as f:
                    json.dump(results['statistics'], f, indent=2, default=str)

            # Export metadata
            metadata = {
                'export_date': datetime.now().isoformat(),
                'analysis_parameters': results.get('parameters', {}),
                'file_info': results.get('file_info', {}),
                'software_version': results.get('version', 'unknown')
            }

            with open(output_dir / 'metadata' / 'analysis_info.json', 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            # Create README
            self._create_analysis_readme(output_dir, results)

            self.logger.info(f"Created analysis package in {output_dir}")
            return True

        except Exception as e:
            self.logger.error(f"Error creating analysis package: {e}")
            return False

    def _create_analysis_readme(self, output_dir: Path, results: Dict[str, Any]):
        """Create a README file for the analysis package."""
        readme_content = f"""
# Particle Tracking Analysis Results

## Analysis Information
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Software**: Particle Tracking Analyzer
- **Version**: {results.get('version', 'unknown')}

## Directory Structure
- `data/`: Exported data files (CSV format)
- `plots/`: Generated plots (PNG and PDF formats)
- `statistics/`: Statistical summaries (JSON format)
- `metadata/`: Analysis metadata and parameters

## Data Files
"""

        if 'dataframes' in results:
            for name, df in results['dataframes'].items():
                readme_content += f"- `{name}.csv`: {len(df)} rows, {len(df.columns)} columns\n"

        readme_content += f"""
## Analysis Parameters
"""

        if 'parameters' in results:
            for key, value in results['parameters'].items():
                readme_content += f"- **{key}**: {value}\n"

        with open(output_dir / 'README.md', 'w') as f:
            f.write(readme_content)


# ============================================================================
# ENHANCED UTILITY FUNCTIONS
# ============================================================================

def ensure_directory(path: Union[str, Path], create_parents: bool = True) -> Path:
    """Ensure directory exists, create if necessary with enhanced options."""
    path = Path(path)

    if create_parents:
        path.mkdir(parents=True, exist_ok=True)
    else:
        if not path.parent.exists():
            raise FileNotFoundError(f"Parent directory does not exist: {path.parent}")
        path.mkdir(exist_ok=True)

    return path


def get_app_data_directory() -> Path:
    """Get application data directory with enhanced structure."""
    base_dir = Path.home() / ".particle_tracker"

    # Create subdirectories
    subdirs = ['logs', 'config', 'temp', 'cache', 'exports', 'backups']
    for subdir in subdirs:
        ensure_directory(base_dir / subdir)

    return base_dir


def get_temp_directory() -> Path:
    """Get temporary directory for the application."""
    return ensure_directory(get_app_data_directory() / "temp")


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format with enhanced precision."""
    if size_bytes == 0:
        return "0 B"

    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if size_bytes < 1024.0:
            if unit == 'B':
                return f"{size_bytes} {unit}"
            else:
                return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0

    return f"{size_bytes:.1f} EB"


def format_duration(seconds: float) -> str:
    """Format duration in human readable format with enhanced precision."""
    if seconds < 0:
        return "Invalid duration"

    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"


def backup_file(file_path: Union[str, Path],
               backup_dir: Optional[Union[str, Path]] = None,
               max_backups: int = 5) -> Path:
    """Create a backup of a file with rotation support."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File to backup not found: {file_path}")

    if backup_dir is None:
        backup_dir = get_app_data_directory() / "backups"
    else:
        backup_dir = Path(backup_dir)

    ensure_directory(backup_dir)

    # Create backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
    backup_path = backup_dir / backup_name

    # Copy file
    shutil.copy2(file_path, backup_path)

    # Rotate old backups
    _rotate_backups(backup_dir, file_path.stem, max_backups)

    return backup_path


def _rotate_backups(backup_dir: Path, file_stem: str, max_backups: int):
    """Rotate old backup files, keeping only the most recent ones."""
    pattern = f"{file_stem}_*.csv"  # Adjust pattern as needed
    backup_files = sorted(backup_dir.glob(pattern), key=lambda x: x.stat().st_mtime)

    # Remove excess backups
    while len(backup_files) > max_backups:
        oldest_backup = backup_files.pop(0)
        oldest_backup.unlink()


def find_files_with_pattern(directory: Union[str, Path],
                           patterns: Union[str, List[str]],
                           recursive: bool = True) -> List[Path]:
    """Find files matching patterns with enhanced search options."""
    directory = Path(directory)

    if isinstance(patterns, str):
        patterns = [patterns]

    found_files = []

    for pattern in patterns:
        if recursive:
            found_files.extend(directory.rglob(pattern))
        else:
            found_files.extend(directory.glob(pattern))

    # Remove duplicates and sort
    return sorted(list(set(found_files)))


def validate_csv_structure(file_path: Union[str, Path],
                          required_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """Validate CSV file structure and return detailed information."""
    file_path = Path(file_path)

    validation_result = {
        'valid': False,
        'error': None,
        'columns': [],
        'row_count': 0,
        'missing_columns': [],
        'extra_info': {}
    }

    try:
        # Read first few rows to check structure
        df = pd.read_csv(file_path, nrows=10)

        validation_result['columns'] = list(df.columns)
        validation_result['row_count'] = len(df)

        # Check required columns if specified
        if required_columns:
            missing = [col for col in required_columns if col not in df.columns]
            validation_result['missing_columns'] = missing
            validation_result['valid'] = len(missing) == 0
        else:
            validation_result['valid'] = True

        # Get additional info
        validation_result['extra_info'] = {
            'has_track_column': any('track' in col.lower() for col in df.columns),
            'has_coordinates': 'x' in df.columns and 'y' in df.columns,
            'has_frame_column': 'frame' in df.columns,
            'has_intensity': any('intensity' in col.lower() for col in df.columns)
        }

    except Exception as e:
        validation_result['error'] = str(e)

    return validation_result


# ============================================================================
# ENHANCED PERFORMANCE MONITORING
# ============================================================================

class EnhancedPerformanceMonitor:
    """Enhanced performance monitor with batch processing support."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.start_times = {}
        self.memory_usage = {}
        self.operation_stats = defaultdict(list)

    def start_timing(self, operation: str, details: Optional[str] = None):
        """Start timing an operation with optional details."""
        key = f"{operation}_{details}" if details else operation
        self.start_times[key] = time.time()
        self.logger.debug(f"Started timing: {key}")

    def end_timing(self, operation: str, details: Optional[str] = None) -> float:
        """End timing an operation and return duration."""
        key = f"{operation}_{details}" if details else operation

        if key not in self.start_times:
            self.logger.warning(f"No start time found for operation: {key}")
            return 0

        duration = time.time() - self.start_times[key]
        self.operation_stats[operation].append({
            'duration': duration,
            'details': details,
            'timestamp': datetime.now()
        })

        self.logger.info(f"{key} completed in {format_duration(duration)}")
        del self.start_times[key]

        return duration

    def get_operation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive operation statistics."""
        stats = {}

        for operation, measurements in self.operation_stats.items():
            durations = [m['duration'] for m in measurements]

            stats[operation] = {
                'count': len(durations),
                'total_time': sum(durations),
                'average_time': np.mean(durations),
                'min_time': min(durations),
                'max_time': max(durations),
                'std_time': np.std(durations) if len(durations) > 1 else 0
            }

        return stats

    def log_memory_usage(self, label: str):
        """Log current memory usage with enhanced tracking."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()

            self.memory_usage[label] = {
                'rss': memory_info.rss / 1024 / 1024,  # MB
                'vms': memory_info.vms / 1024 / 1024,  # MB
                'timestamp': datetime.now()
            }

            self.logger.debug(f"Memory usage ({label}): RSS={self.memory_usage[label]['rss']:.1f} MB")

        except ImportError:
            self.logger.debug("psutil not available for memory monitoring")
        except Exception as e:
            self.logger.debug(f"Error monitoring memory: {e}")

    def get_memory_report(self) -> str:
        """Get detailed memory usage report."""
        if not self.memory_usage:
            return "No memory data available"

        lines = ["Memory Usage Report:", "-" * 30]

        for label, info in self.memory_usage.items():
            timestamp = info['timestamp'].strftime('%H:%M:%S')
            lines.append(f"{timestamp} - {label}: RSS={info['rss']:.1f} MB, VMS={info['vms']:.1f} MB")

        return "\n".join(lines)


# Initialize global performance monitor
performance_monitor = EnhancedPerformanceMonitor()


# ============================================================================
# BATCH OPERATION UTILITIES
# ============================================================================

def process_files_in_batches(file_list: List[Path],
                            batch_size: int = 10,
                            callback: callable = None) -> Generator[List[Path], None, None]:
    """Process files in batches with progress tracking."""
    total_files = len(file_list)

    for i in range(0, total_files, batch_size):
        batch = file_list[i:i + batch_size]

        if callback:
            progress = (i + len(batch)) / total_files * 100
            callback(f"Processing batch {i//batch_size + 1}/{(total_files-1)//batch_size + 1}", progress)

        yield batch


def create_directory_structure(base_dir: Union[str, Path],
                              structure: Dict[str, Any]) -> Path:
    """Create a directory structure from a nested dictionary."""
    base_dir = Path(base_dir)
    ensure_directory(base_dir)

    def _create_recursive(current_dir: Path, struct: Dict[str, Any]):
        for name, content in struct.items():
            new_dir = current_dir / name
            ensure_directory(new_dir)

            if isinstance(content, dict):
                _create_recursive(new_dir, content)

    _create_recursive(base_dir, structure)
    return base_dir


# ============================================================================
# ERROR HANDLING
# ============================================================================

class ParticleTrackerError(Exception):
    """Base exception for particle tracker application."""
    pass


class FileProcessingError(ParticleTrackerError):
    """Error during file processing."""
    pass


class BatchProcessingError(ParticleTrackerError):
    """Error during batch processing."""
    pass


class ROIError(ParticleTrackerError):
    """Error with ROI operations."""
    pass


class ExportError(ParticleTrackerError):
    """Error during export operations."""
    pass

class DataLoadError(ParticleTrackerError):
    """Error during data loading operations."""
    pass


class AnalysisError(ParticleTrackerError):
    """Error during analysis operations."""
    pass


class ProjectError(ParticleTrackerError):
    """Error during project operations."""
    pass

def handle_exception(exc_type, exc_value, exc_traceback):
    """Enhanced global exception handler."""
    logger = logging.getLogger(__name__)

    if issubclass(exc_type, KeyboardInterrupt):
        # Allow keyboard interrupt to pass through
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # Log the exception with enhanced context
    logger.error(
        f"Uncaught {exc_type.__name__}: {exc_value}",
        exc_info=(exc_type, exc_value, exc_traceback),
        extra={
            'exception_type': exc_type.__name__,
            'exception_message': str(exc_value),
            'timestamp': datetime.now().isoformat()
        }
    )


# Set enhanced global exception handler
import sys
sys.excepthook = handle_exception
