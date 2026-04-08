#!/usr/bin/env python3
"""
Enhanced Path Utilities
=======================

Comprehensive path utilities for hierarchical particle tracking analysis.
Supports experiment → condition → file workflows with sophisticated file discovery,
ROI management, and organized output structures.
"""

import os
import re
import glob
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union, Set
import logging
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class FileInfo:
    """Information about a discovered file."""
    path: Path
    filename: str
    base_name: str
    extension: str
    file_type: str  # 'image', 'localization', 'trajectory', 'roi', 'training'
    experiment: Optional[str] = None
    condition: Optional[str] = None
    processing_stage: Optional[str] = None  # 'raw', 'detected', 'linked', 'features', etc.


@dataclass
class ExperimentStructure:
    """Structure representing a hierarchical experiment."""
    experiment_path: Path
    experiment_name: str
    conditions: Dict[str, Path]  # condition_name -> condition_path
    files_by_condition: Dict[str, List[FileInfo]]  # condition -> files
    total_files: int


class PathUtilities:
    """Enhanced path utilities for particle tracking analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # File type patterns
        self.file_patterns = {
            'image': [
                r'.*\.tif{1,2}$',
                r'.*\.png$',
                r'.*\.jpg$',
                r'.*\.jpeg$',
                r'.*_bin\d+\.tif{1,2}$',  # Binned images
                r'.*_crop\d+\.tif{1,2}$',  # Cropped images
                r'.*_piezo\d+\.tif{1,2}$',  # Piezo images
            ],
            'localization': [
                r'.*_locs\.csv$',
                r'.*_locsID\.csv$',
                r'.*_localization\.csv$',
                r'.*_detections\.csv$',
                r'.*_thunderstorm\.csv$',
            ],
            'trajectory': [
                r'.*_tracks\.csv$',
                r'.*_tracksRG\.csv$',
                r'.*_trajectories\.csv$',
                r'.*_linked\.csv$',
                r'.*_features\.csv$',
            ],
            'analysis_results': [
                r'.*_SVMPredicted.*\.csv$',
                r'.*_classified\.csv$',
                r'.*_NN\.csv$',
                r'.*_diffusion\.csv$',
                r'.*_velocity\.csv$',
                r'.*_BGsubtract\.csv$',
                r'.*_metrics\.csv$',
                r'.*_autocorrelation.*\.csv$',
            ],
            'roi': [
                r'.*_ROI\.txt$',
                r'.*ROI.*\.txt$',
                r'.*\.roi$',
                r'.*regions\.csv$',
            ],
            'training': [
                r'.*training.*\.csv$',
                r'.*_training_feats\.csv$',
            ]
        }
        
        # Processing stage patterns
        self.stage_patterns = {
            'raw': [r'(?<!_\w+)\.tif{1,2}$', r'(?<!_\w+)\.csv$'],
            'detected': [r'_locs\.csv$', r'_locsID\.csv$', r'_detections\.csv$'],
            'linked': [r'_tracks\.csv$', r'_linked\.csv$', r'_trajectories\.csv$'],
            'features': [r'_tracksRG\.csv$', r'_features\.csv$', r'_metrics\.csv$'],
            'classified': [r'_SVMPredicted.*\.csv$', r'_classified\.csv$'],
            'analyzed': [r'_NN\.csv$', r'_diffusion\.csv$', r'_velocity\.csv$', r'_BGsubtract\.csv$'],
            'final': [r'_AllLocs\.csv$', r'_trapped-AllFrames\.csv$']
        }

    def find_project_root(self, start_path: Optional[Path] = None) -> Path:
        """Find the project root directory containing particle_tracker module."""
        if start_path is None:
            start_path = Path(__file__).resolve()
        
        current_path = start_path
        
        # Look for particle_tracker directory going up the tree
        for parent in current_path.parents:
            particle_tracker_dir = parent / "particle_tracker"
            if particle_tracker_dir.exists() and particle_tracker_dir.is_dir():
                # Check if it's really the particle_tracker module
                if (particle_tracker_dir / "__init__.py").exists():
                    return parent
        
        # Fallback: assume we're already in the project
        return Path.cwd()

    def get_resources_directory(self) -> Path:
        """Get the resources directory path."""
        project_root = self.find_project_root()
        return project_root / "particle_tracker" / "resources"

    def get_training_data_directory(self) -> Path:
        """Get the training data directory path."""
        return self.get_resources_directory() / "training_data"

    def get_example_data_directory(self) -> Path:
        """Get the example data directory path."""
        return self.get_resources_directory() / "example_data"

    def get_default_training_data_path(self) -> Optional[str]:
        """Get the default path to SVM training data."""
        try:
            training_data_path = (self.get_training_data_directory() / 
                                "tdTomato_37Degree_CytoD_training_feats.csv")
            
            if training_data_path.exists():
                return str(training_data_path)
            else:
                self.logger.info(f"Default training data not found at: {training_data_path}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Error finding default training data path: {e}")
            return None

    def classify_file_type(self, file_path: Path) -> str:
        """Classify a file based on its name and extension."""
        filename = file_path.name.lower()
        
        for file_type, patterns in self.file_patterns.items():
            for pattern in patterns:
                if re.match(pattern, filename, re.IGNORECASE):
                    return file_type
        
        # Default classification based on extension
        ext = file_path.suffix.lower()
        if ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
            return 'image'
        elif ext in ['.csv', '.txt']:
            return 'data'
        else:
            return 'unknown'

    def determine_processing_stage(self, file_path: Path) -> str:
        """Determine the processing stage of a file."""
        filename = file_path.name.lower()
        
        for stage, patterns in self.stage_patterns.items():
            for pattern in patterns:
                if re.search(pattern, filename, re.IGNORECASE):
                    return stage
        
        return 'unknown'

    def create_file_info(self, file_path: Path, experiment: Optional[str] = None, 
                        condition: Optional[str] = None) -> FileInfo:
        """Create a FileInfo object from a file path."""
        file_path = Path(file_path)
        
        return FileInfo(
            path=file_path,
            filename=file_path.name,
            base_name=file_path.stem,
            extension=file_path.suffix,
            file_type=self.classify_file_type(file_path),
            experiment=experiment,
            condition=condition,
            processing_stage=self.determine_processing_stage(file_path)
        )

    def discover_experiment_structure(self, experiment_path: Union[str, Path]) -> ExperimentStructure:
        """Discover the structure of an experiment directory."""
        experiment_path = Path(experiment_path)
        
        if not experiment_path.exists():
            raise ValueError(f"Experiment path does not exist: {experiment_path}")
        
        experiment_name = experiment_path.name
        
        # Find condition directories (subdirectories that contain data files)
        conditions = {}
        files_by_condition = defaultdict(list)
        total_files = 0
        
        # Check if this is a flat structure (files directly in experiment directory)
        direct_files = self._find_data_files(experiment_path, max_depth=1)
        
        if direct_files:
            # Flat structure - treat experiment directory as single condition
            conditions['default'] = experiment_path
            for file_path in direct_files:
                file_info = self.create_file_info(file_path, experiment_name, 'default')
                files_by_condition['default'].append(file_info)
                total_files += 1
        else:
            # Hierarchical structure - look for condition subdirectories
            for item in experiment_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    condition_files = self._find_data_files(item)
                    if condition_files:
                        condition_name = item.name
                        conditions[condition_name] = item
                        
                        for file_path in condition_files:
                            file_info = self.create_file_info(file_path, experiment_name, condition_name)
                            files_by_condition[condition_name].append(file_info)
                            total_files += 1
        
        return ExperimentStructure(
            experiment_path=experiment_path,
            experiment_name=experiment_name,
            conditions=conditions,
            files_by_condition=dict(files_by_condition),
            total_files=total_files
        )

    def _find_data_files(self, directory: Path, max_depth: Optional[int] = None) -> List[Path]:
        """Find data files in a directory."""
        files = []
        
        if max_depth is not None and max_depth <= 0:
            return files
        
        try:
            for item in directory.iterdir():
                if item.is_file():
                    file_type = self.classify_file_type(item)
                    if file_type in ['image', 'localization', 'trajectory', 'analysis_results']:
                        files.append(item)
                elif item.is_dir() and not item.name.startswith('.'):
                    if max_depth is None or max_depth > 1:
                        next_depth = None if max_depth is None else max_depth - 1
                        files.extend(self._find_data_files(item, next_depth))
        except PermissionError:
            self.logger.warning(f"Permission denied accessing {directory}")
        
        return files

    def find_files_by_pattern(self, directory: Path, patterns: List[str], 
                             recursive: bool = True) -> List[Path]:
        """Find files matching specific patterns."""
        files = []
        
        for pattern in patterns:
            if recursive:
                glob_pattern = f"**/{pattern}"
                matches = directory.glob(glob_pattern)
            else:
                matches = directory.glob(pattern)
            
            files.extend(matches)
        
        return sorted(list(set(files)))  # Remove duplicates and sort

    def find_roi_files(self, data_file: Path) -> List[Path]:
        """Find ROI files associated with a data file."""
        roi_files = []
        
        # Get the base name without processing suffixes
        base_name = self._get_clean_base_name(data_file)
        directory = data_file.parent
        
        # Common ROI naming patterns
        roi_patterns = [
            f"ROI_{base_name}.txt",
            f"{base_name}_ROI.txt",
            f"ROI_{base_name}_*.txt",
            f"{base_name}_regions.csv",
            f"{base_name}.roi"
        ]
        
        for pattern in roi_patterns:
            matches = directory.glob(pattern)
            roi_files.extend(matches)
        
        return roi_files

    def _get_clean_base_name(self, file_path: Path) -> str:
        """Get clean base name removing common processing suffixes."""
        name = file_path.stem
        
        # Remove common suffixes
        suffixes_to_remove = [
            '_locs', '_locsID', '_tracks', '_tracksRG', '_features',
            '_SVMPredicted', '_NN', '_diffusion', '_velocity', '_BGsubtract',
            '_AllLocs', '_trapped-AllFrames', '_bin10', '_bin20', '_crop100'
        ]
        
        for suffix in suffixes_to_remove:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break
        
        return name

    def create_output_structure(self, base_path: Path, 
                               experiment_name: Optional[str] = None) -> Dict[str, Path]:
        """Create organized output directory structure."""
        if experiment_name:
            output_root = base_path / f"{experiment_name}_analysis_output"
        else:
            output_root = base_path / "analysis_output"
        
        # Create directory structure
        directories = {
            'root': output_root,
            'raw_data': output_root / "01_raw_data",
            'detections': output_root / "02_detections", 
            'trajectories': output_root / "03_trajectories",
            'features': output_root / "04_features",
            'classifications': output_root / "05_classifications",
            'advanced_analysis': output_root / "06_advanced_analysis",
            'statistics': output_root / "07_statistics",
            'visualizations': output_root / "08_visualizations",
            'exports': output_root / "09_exports",
            'logs': output_root / "logs"
        }
        
        # Create subdirectories for advanced analysis
        advanced_subdirs = [
            'density_analysis', 'autocorrelation', 'background_subtraction',
            'trajectory_interpolation', 'localization_precision'
        ]
        
        for subdir in advanced_subdirs:
            directories[f'advanced_{subdir}'] = directories['advanced_analysis'] / subdir
        
        # Create all directories
        for dir_path in directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        return directories

    def generate_output_filename(self, input_file: Path, stage: str, 
                                suffix: str = "", extension: str = ".csv") -> str:
        """Generate standardized output filename."""
        base_name = self._get_clean_base_name(input_file)
        
        # Stage-specific suffixes
        stage_suffixes = {
            'detection': '_locs',
            'linking': '_tracks', 
            'features': '_tracksRG',
            'classification': '_SVMPredicted',
            'density': '_NN',
            'diffusion': '_diffusion',
            'velocity': '_velocity',
            'background': '_BGsubtract',
            'interpolation': '_trapped-AllFrames',
            'precision': '_locErr',
            'autocorrelation': '_autocorr',
            'final': '_AllLocs'
        }
        
        stage_suffix = stage_suffixes.get(stage, f"_{stage}")
        
        if suffix:
            filename = f"{base_name}{stage_suffix}_{suffix}{extension}"
        else:
            filename = f"{base_name}{stage_suffix}{extension}"
        
        return filename

    def find_input_files_for_stage(self, directory: Path, stage: str) -> List[FileInfo]:
        """Find input files appropriate for a specific analysis stage."""
        files = []
        
        # Define what file types each stage expects
        stage_requirements = {
            'detection': ['image'],
            'linking': ['localization'],
            'features': ['trajectory'],
            'classification': ['trajectory', 'analysis_results'],
            'density': ['trajectory', 'analysis_results'],
            'background': ['trajectory', 'analysis_results'],
            'interpolation': ['analysis_results'],
            'autocorrelation': ['trajectory', 'analysis_results']
        }
        
        required_types = stage_requirements.get(stage, ['data'])
        
        # Find files in directory
        for file_path in self._find_data_files(directory):
            file_info = self.create_file_info(file_path)
            if file_info.file_type in required_types:
                files.append(file_info)
        
        return files

    def find_matching_files(self, reference_file: Path, file_types: List[str]) -> Dict[str, Path]:
        """Find files that match a reference file (same base name, different types)."""
        base_name = self._get_clean_base_name(reference_file)
        directory = reference_file.parent
        matching_files = {}
        
        for file_type in file_types:
            patterns = self.file_patterns.get(file_type, [])
            
            for pattern in patterns:
                # Adapt pattern to use the base name
                adapted_pattern = pattern.replace('.*', base_name + '*')
                matches = directory.glob(adapted_pattern)
                
                for match in matches:
                    if self.classify_file_type(match) == file_type:
                        matching_files[file_type] = match
                        break
        
        return matching_files

    def organize_files_by_experiment(self, file_paths: List[Path]) -> Dict[str, Dict[str, List[Path]]]:
        """Organize files by experiment and condition."""
        organization = defaultdict(lambda: defaultdict(list))
        
        for file_path in file_paths:
            # Try to extract experiment and condition from path
            parts = file_path.parts
            
            # Look for common experiment/condition patterns
            experiment = None
            condition = None
            
            # If file is in nested structure, use parent directories
            if len(parts) >= 3:
                # Assume structure like: .../experiment/condition/file
                experiment = parts[-3]
                condition = parts[-2]
            elif len(parts) >= 2:
                # Assume structure like: .../experiment/file
                experiment = parts[-2]
                condition = 'default'
            else:
                experiment = 'unknown'
                condition = 'default'
            
            organization[experiment][condition].append(file_path)
        
        return dict(organization)

    def validate_file_chain(self, files: List[Path]) -> Dict[str, bool]:
        """Validate that files form a complete processing chain."""
        validation = {
            'has_raw_data': False,
            'has_detections': False,
            'has_trajectories': False,
            'has_features': False,
            'has_classification': False,
            'chain_complete': False
        }
        
        file_types = set()
        processing_stages = set()
        
        for file_path in files:
            file_info = self.create_file_info(file_path)
            file_types.add(file_info.file_type)
            processing_stages.add(file_info.processing_stage)
        
        # Check for each stage
        validation['has_raw_data'] = 'image' in file_types or 'raw' in processing_stages
        validation['has_detections'] = 'localization' in file_types or 'detected' in processing_stages
        validation['has_trajectories'] = 'trajectory' in file_types or 'linked' in processing_stages
        validation['has_features'] = 'features' in processing_stages
        validation['has_classification'] = 'classified' in processing_stages
        
        # Chain is complete if all major stages are present
        validation['chain_complete'] = all([
            validation['has_raw_data'],
            validation['has_detections'],
            validation['has_trajectories']
        ])
        
        return validation

    def get_file_dependencies(self, file_path: Path) -> List[Path]:
        """Get the files that this file depends on (processing chain)."""
        dependencies = []
        file_info = self.create_file_info(file_path)
        base_name = self._get_clean_base_name(file_path)
        directory = file_path.parent
        
        # Define dependency chains
        if file_info.processing_stage == 'linked':
            # Trajectories depend on detections
            detection_patterns = [f"{base_name}_locs.csv", f"{base_name}_locsID.csv"]
            for pattern in detection_patterns:
                dep_file = directory / pattern
                if dep_file.exists():
                    dependencies.append(dep_file)
        
        elif file_info.processing_stage == 'features':
            # Features depend on trajectories
            trajectory_patterns = [f"{base_name}_tracks.csv", f"{base_name}_linked.csv"]
            for pattern in trajectory_patterns:
                dep_file = directory / pattern
                if dep_file.exists():
                    dependencies.append(dep_file)
        
        elif file_info.processing_stage == 'classified':
            # Classification depends on features
            feature_patterns = [f"{base_name}_tracksRG.csv", f"{base_name}_features.csv"]
            for pattern in feature_patterns:
                dep_file = directory / pattern
                if dep_file.exists():
                    dependencies.append(dep_file)
        
        return dependencies

    def suggest_next_processing_steps(self, files: List[Path]) -> List[str]:
        """Suggest next processing steps based on available files."""
        suggestions = []
        
        file_stages = set()
        for file_path in files:
            file_info = self.create_file_info(file_path)
            file_stages.add(file_info.processing_stage)
        
        # Suggest based on what's available
        if 'raw' in file_stages and 'detected' not in file_stages:
            suggestions.append('detection')
        if 'detected' in file_stages and 'linked' not in file_stages:
            suggestions.append('linking')
        if 'linked' in file_stages and 'features' not in file_stages:
            suggestions.append('feature_calculation')
        if 'features' in file_stages and 'classified' not in file_stages:
            suggestions.append('classification')
        if 'classified' in file_stages:
            suggestions.extend(['density_analysis', 'autocorrelation', 'background_subtraction'])
        
        return suggestions


# Create a global instance for convenience
path_utils = PathUtilities()

# Convenience functions
def find_project_root() -> Path:
    """Find the project root directory."""
    return path_utils.find_project_root()

def get_default_training_data_path() -> Optional[str]:
    """Get the default path to SVM training data."""
    return path_utils.get_default_training_data_path()

def get_resources_directory() -> Path:
    """Get the resources directory path."""
    return path_utils.get_resources_directory()

def get_training_data_directory() -> Path:
    """Get the training data directory path."""
    return path_utils.get_training_data_directory()

def get_example_data_directory() -> Path:
    """Get the example data directory path."""
    return path_utils.get_example_data_directory()

def discover_experiment_structure(experiment_path: Union[str, Path]) -> ExperimentStructure:
    """Discover the structure of an experiment directory."""
    return path_utils.discover_experiment_structure(experiment_path)

def create_output_structure(base_path: Path, experiment_name: Optional[str] = None) -> Dict[str, Path]:
    """Create organized output directory structure."""
    return path_utils.create_output_structure(base_path, experiment_name)
