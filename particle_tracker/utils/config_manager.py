#!/usr/bin/env python3
"""
Enhanced Configuration Manager Module
====================================

Comprehensive configuration management for particle tracking analysis
supporting hierarchical experiments, batch processing, and advanced analysis workflows.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
import copy

from PyQt6.QtCore import QObject, pyqtSignal, QSettings


@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters."""
    
    # Detection parameters
    detection_method: str = "threshold"
    detection_sigma: float = 1.6
    detection_threshold: float = 3.0
    min_intensity: int = 100
    max_intensity: int = 10000
    spot_diameter: int = 7
    background_subtraction: bool = True
    
    # Linking parameters
    linking_method: str = "trackpy"
    max_distance: float = 2.0
    max_gap_frames: int = 1
    min_track_length: int = 5
    adaptive_search: bool = True
    link_strategy: str = "auto"
    
    # Feature calculation parameters
    pixel_size: float = 108.0  # nm per pixel
    frame_rate: float = 10.0   # Hz
    calculate_rg: bool = True
    calculate_srg: bool = True  # NEW: Scaled radius of gyration
    calculate_asymmetry: bool = True
    calculate_fractal: bool = True
    calculate_msd: bool = True
    calculate_velocity: bool = True
    calculate_nn: bool = True
    
    # NEW: Advanced shape metrics (from trajectory_analyzer.py)
    rg_method: str = "simple"  # "simple" or "tensor"
    include_shape_metrics: bool = True
    include_linear_metrics: bool = True
    linear_eigenvalue_ratio_cutoff: float = 20.0
    linear_step_alignment_cutoff: float = 0.7
    linear_directionality_cutoff: float = 0.8
    linear_perpendicular_cutoff: float = 0.15
    
    # Classification parameters
    classification_method: str = "svm"
    mobility_threshold: float = 2.11  # Standard threshold
    srg_cutoff: float = 2.22236433588659  # Golan & Sherman threshold
    
    # SVM parameters
    svm_training_data: Optional[str] = None
    svm_features: List[str] = field(default_factory=lambda: [
        'radius_gyration', 'asymmetry', 'fracDimension',
        'netDispl', 'Straight', 'kurtosis'
    ])
    svm_multi_round: bool = True  # NEW: Multi-round SVM classification
    
    # NEW: Multi-radius density analysis (Step_8)
    density_analysis_enabled: bool = True
    density_radii: List[int] = field(default_factory=lambda: [3, 5, 10, 20, 30])
    
    # NEW: Background subtraction parameters (Step_9)
    background_subtraction_enabled: bool = True
    roi_background_method: str = "mean"  # "mean", "median", "percentile"
    camera_noise_subtraction: bool = True
    
    # NEW: Trajectory interpolation parameters (Step_10)
    interpolation_enabled: bool = False
    interpolation_target_class: int = 3  # SVM class to interpolate (trapped)
    interpolation_grouping: str = "hcluster"  # "none", "in pixel", "hcluster"
    interpolation_smoothing_window: int = 10
    
    # NEW: Autocorrelation analysis parameters
    autocorr_enabled: bool = False
    autocorr_time_interval: float = 200.0  # Time between frames (ms)
    autocorr_num_intervals: int = 25
    autocorr_max_tracks_plot: int = 100
    autocorr_individual_tracks: bool = True
    
    # NEW: Localization precision parameters (Step_11)
    localization_precision_enabled: bool = False
    
    # NEW: Velocity and diffusion analysis parameters
    diffusion_analysis_enabled: bool = True
    velocity_analysis_enabled: bool = True
    origin_analysis_enabled: bool = True


@dataclass 
class ExperimentConfig:
    """Configuration for experiment-level settings."""
    
    # Experiment metadata
    experiment_name: str = ""
    experiment_description: str = ""
    experiment_type: str = ""  # e.g., "dose_response", "time_course", "comparison"
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Hierarchical structure
    condition_folders: List[str] = field(default_factory=list)
    file_patterns: List[str] = field(default_factory=lambda: ["*.csv", "*.xlsx", "*.tif"])
    
    # Batch processing
    parallel_processing: bool = True
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    
    # Cross-condition analysis
    cross_condition_comparison: bool = True
    statistical_tests: List[str] = field(default_factory=lambda: ["t_test", "anova"])
    
    # Output settings
    generate_summary_stats: bool = True
    generate_plots: bool = True
    export_individual_tracks: bool = True
    output_formats: List[str] = field(default_factory=lambda: ["csv", "xlsx"])


@dataclass
class ROIConfig:
    """Configuration for ROI management."""
    
    # ROI file settings
    roi_file_pattern: str = "*_ROI.txt"
    roi_background_index: int = 0  # Which ROI to use for background
    
    # ROI processing
    roi_smoothing: bool = True
    roi_smoothing_window: int = 10
    roi_outlier_removal: bool = True
    roi_outlier_threshold: float = 3.0  # Standard deviations


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    
    # Plot appearance
    plot_style: str = "default"
    color_scheme: str = "viridis"
    figure_dpi: int = 300
    
    # Autocorrelation plots
    autocorr_plot_y_min: float = -0.2
    autocorr_plot_y_max: float = 1.0
    autocorr_plot_x_min: float = 0
    autocorr_plot_x_max: Optional[float] = 2000
    
    # Trajectory visualization
    show_individual_tracks: bool = True
    track_color_by: str = "classification"  # "classification", "velocity", "frame"
    max_tracks_display: int = 1000
    
    # Cross-condition plots
    cross_condition_title: Optional[str] = None


@dataclass
class ExportConfig:
    """Configuration for data export."""
    
    # File formats
    default_format: str = "csv"
    include_metadata: bool = True
    
    # Split outputs
    split_by_classification: bool = True
    split_by_mobility: bool = True
    mobile_only_splits: bool = True
    
    # Compression
    compress_large_files: bool = True
    compression_threshold_mb: float = 50.0
    
    # Archive settings
    create_archive: bool = False
    archive_format: str = "zip"  # "zip", "tar", "tar.gz"


@dataclass
class ApplicationConfig:
    """Main application configuration containing all sub-configurations."""
    
    # File paths
    last_data_directory: str = ""
    last_project_directory: str = ""
    last_experiment_directory: str = ""  # NEW
    recent_files: List[str] = field(default_factory=list)
    recent_projects: List[str] = field(default_factory=list)
    recent_experiments: List[str] = field(default_factory=list)  # NEW
    
    # Training data
    default_svm_training_data: str = ""
    svm_auto_detect: bool = True
    
    # UI settings
    window_geometry: str = ""
    window_state: str = ""
    theme: str = "default"
    show_advanced_options: bool = False
    
    # Performance settings
    max_memory_usage_mb: int = 4096  # Increased default
    num_threads: int = 4
    enable_gpu: bool = False
    
    # Analysis configurations
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    
    # NEW: Analysis presets for different experiment types
    analysis_presets: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class ConfigManager(QObject):
    """Enhanced configuration manager supporting hierarchical experiments and advanced analysis."""

    configChanged = pyqtSignal(str, object)  # key, value
    presetLoaded = pyqtSignal(str)  # preset_name
    experimentConfigUpdated = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger(__name__)

        # Configuration file paths
        self.config_dir = Path.home() / ".particle_tracker"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / "config.json"
        self.presets_file = self.config_dir / "analysis_presets.json"
        
        # Load configuration
        self.config = self._load_config()
        self._load_presets()
        
        # Create default presets if none exist
        if not self.config.analysis_presets:
            self._create_default_presets()

        self.logger.info("Enhanced configuration manager initialized")

    def _load_config(self) -> ApplicationConfig:
        """Load configuration from file with enhanced settings."""
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_dict = json.load(f)

                # Handle nested dataclass loading
                config = self._dict_to_config(config_dict)
                self.logger.info(f"Configuration loaded from {self.config_file}")

            except Exception as e:
                self.logger.warning(f"Error loading config: {e}, using defaults")
                config = ApplicationConfig()
        else:
            config = ApplicationConfig()

        # Auto-detect training data if enabled
        if config.analysis.svm_training_data is None and config.svm_auto_detect:
            self._auto_detect_training_data(config)

        return config

    def _dict_to_config(self, config_dict: dict) -> ApplicationConfig:
        """Convert nested dictionary to ApplicationConfig with proper dataclass handling."""
        
        # Extract nested configurations
        analysis_dict = config_dict.pop('analysis', {})
        experiment_dict = config_dict.pop('experiment', {})
        roi_dict = config_dict.pop('roi', {})
        visualization_dict = config_dict.pop('visualization', {})
        export_dict = config_dict.pop('export', {})
        
        # Create main config
        config = ApplicationConfig(**config_dict)
        
        # Update nested configurations
        if analysis_dict:
            config.analysis = AnalysisConfig(**analysis_dict)
        if experiment_dict:
            config.experiment = ExperimentConfig(**experiment_dict)
        if roi_dict:
            config.roi = ROIConfig(**roi_dict)
        if visualization_dict:
            config.visualization = VisualizationConfig(**visualization_dict)
        if export_dict:
            config.export = ExportConfig(**export_dict)
            
        return config

    def _auto_detect_training_data(self, config: ApplicationConfig):
        """Auto-detect SVM training data with enhanced path resolution."""
        try:
            from particle_tracker.utils.path_utils import get_default_training_data_path
            default_path = get_default_training_data_path()
            if default_path:
                config.analysis.svm_training_data = default_path
                self.logger.info(f"Auto-detected SVM training data: {default_path}")
                self.save_config()
        except Exception as e:
            self.logger.debug(f"Could not auto-detect training data: {e}")

    def _load_presets(self):
        """Load analysis presets from file."""
        if self.presets_file.exists():
            try:
                with open(self.presets_file, 'r') as f:
                    presets = json.load(f)
                self.config.analysis_presets = presets
                self.logger.info(f"Loaded {len(presets)} analysis presets")
            except Exception as e:
                self.logger.warning(f"Error loading presets: {e}")

    def _create_default_presets(self):
        """Create default analysis presets for common experiment types."""
        
        # High-density tracking preset
        high_density = {
            "description": "Settings optimized for high-density particle tracking",
            "detection_method": "trackpy",
            "linking_method": "trackpy", 
            "max_distance": 1.5,
            "max_gap_frames": 0,
            "min_track_length": 10,
            "density_analysis_enabled": True,
            "density_radii": [1, 2, 3, 5, 10],
            "background_subtraction_enabled": True,
            "interpolation_enabled": False
        }
        
        # Single-molecule tracking preset
        single_molecule = {
            "description": "Settings for single-molecule tracking experiments",
            "detection_method": "log",
            "linking_method": "trackpy",
            "max_distance": 3.0,
            "max_gap_frames": 2,
            "min_track_length": 5,
            "rg_method": "tensor",
            "include_linear_metrics": True,
            "autocorr_enabled": True,
            "localization_precision_enabled": True
        }
        
        # Live-cell imaging preset
        live_cell = {
            "description": "Settings for live-cell membrane protein tracking",
            "detection_method": "threshold",
            "linking_method": "trackpy",
            "max_distance": 2.0,
            "max_gap_frames": 1,
            "min_track_length": 5,
            "background_subtraction_enabled": True,
            "interpolation_enabled": True,
            "interpolation_target_class": 3,
            "velocity_analysis_enabled": True,
            "diffusion_analysis_enabled": True
        }
        
        # Fast dynamics preset
        fast_dynamics = {
            "description": "Settings for fast dynamic processes",
            "detection_method": "trackpy",
            "linking_method": "trackpy",
            "max_distance": 4.0,
            "max_gap_frames": 2,
            "min_track_length": 3,
            "frame_rate": 100.0,  # High frame rate
            "autocorr_enabled": True,
            "autocorr_time_interval": 10.0,  # Short intervals
            "autocorr_num_intervals": 50
        }
        
        self.config.analysis_presets = {
            "high_density": high_density,
            "single_molecule": single_molecule,
            "live_cell": live_cell,
            "fast_dynamics": fast_dynamics
        }
        
        self._save_presets()
        self.logger.info("Created default analysis presets")

    def save_config(self):
        """Save current configuration to file."""
        try:
            config_dict = asdict(self.config)
            
            with open(self.config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)

            self.logger.info(f"Configuration saved to {self.config_file}")

        except Exception as e:
            self.logger.error(f"Error saving config: {e}")

    def _save_presets(self):
        """Save analysis presets to file."""
        try:
            with open(self.presets_file, 'w') as f:
                json.dump(self.config.analysis_presets, f, indent=2)
            self.logger.info("Analysis presets saved")
        except Exception as e:
            self.logger.error(f"Error saving presets: {e}")

    # Enhanced getters and setters
    def get_analysis_config(self) -> AnalysisConfig:
        """Get analysis configuration."""
        return self.config.analysis

    def get_experiment_config(self) -> ExperimentConfig:
        """Get experiment configuration."""
        return self.config.experiment

    def get_roi_config(self) -> ROIConfig:
        """Get ROI configuration."""
        return self.config.roi

    def get_visualization_config(self) -> VisualizationConfig:
        """Get visualization configuration."""
        return self.config.visualization

    def get_export_config(self) -> ExportConfig:
        """Get export configuration."""
        return self.config.export

    def update_analysis_config(self, **kwargs):
        """Update analysis configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config.analysis, key):
                setattr(self.config.analysis, key, value)
                self.configChanged.emit(f"analysis.{key}", value)
        self.save_config()

    def update_experiment_config(self, **kwargs):
        """Update experiment configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config.experiment, key):
                setattr(self.config.experiment, key, value)
                self.configChanged.emit(f"experiment.{key}", value)
        self.experimentConfigUpdated.emit()
        self.save_config()

    # Preset management
    def get_preset_names(self) -> List[str]:
        """Get list of available preset names."""
        return list(self.config.analysis_presets.keys())

    def load_preset(self, preset_name: str) -> bool:
        """Load an analysis preset."""
        if preset_name not in self.config.analysis_presets:
            self.logger.warning(f"Preset '{preset_name}' not found")
            return False
        
        preset_data = self.config.analysis_presets[preset_name]
        
        # Update analysis config with preset values
        for key, value in preset_data.items():
            if hasattr(self.config.analysis, key):
                setattr(self.config.analysis, key, value)
        
        self.presetLoaded.emit(preset_name)
        self.save_config()
        self.logger.info(f"Loaded preset: {preset_name}")
        return True

    def save_preset(self, preset_name: str, description: str = "", 
                   config_subset: Optional[Dict[str, Any]] = None) -> bool:
        """Save current analysis settings as a preset."""
        try:
            if config_subset is None:
                # Save current analysis config
                preset_data = asdict(self.config.analysis)
            else:
                preset_data = config_subset.copy()
            
            preset_data["description"] = description
            
            self.config.analysis_presets[preset_name] = preset_data
            self._save_presets()
            
            self.logger.info(f"Saved preset: {preset_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving preset: {e}")
            return False

    def delete_preset(self, preset_name: str) -> bool:
        """Delete an analysis preset."""
        if preset_name in self.config.analysis_presets:
            del self.config.analysis_presets[preset_name]
            self._save_presets()
            self.logger.info(f"Deleted preset: {preset_name}")
            return True
        return False

    # Enhanced recent items management
    def add_recent_experiment(self, experiment_path: str):
        """Add experiment to recent experiments list."""
        experiment_path = str(Path(experiment_path).absolute())
        
        if experiment_path in self.config.recent_experiments:
            self.config.recent_experiments.remove(experiment_path)
        
        self.config.recent_experiments.insert(0, experiment_path)
        self.config.recent_experiments = self.config.recent_experiments[:10]
        self.save_config()

    def get_recent_experiments(self) -> List[str]:
        """Get list of recent experiments (existing only)."""
        return [e for e in self.config.recent_experiments if Path(e).exists()]

    # Analysis parameter optimization
    def suggest_parameters_for_data(self, data_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest optimal parameters based on data characteristics."""
        suggestions = {}
        
        # Get data properties
        particle_density = data_characteristics.get('particle_density', 'medium')
        motion_type = data_characteristics.get('motion_type', 'mixed')
        frame_rate = data_characteristics.get('frame_rate', 10.0)
        
        # Suggest based on density
        if particle_density == 'high':
            suggestions.update({
                'max_distance': 1.5,
                'max_gap_frames': 0,
                'min_track_length': 10,
                'density_radii': [1, 2, 3, 5, 10]
            })
        elif particle_density == 'low':
            suggestions.update({
                'max_distance': 5.0,
                'max_gap_frames': 3,
                'min_track_length': 3,
                'density_radii': [5, 10, 20, 30, 50]
            })
        
        # Suggest based on motion type
        if motion_type == 'fast':
            suggestions.update({
                'max_distance': min(4.0, suggestions.get('max_distance', 3.0) * 1.5),
                'autocorr_time_interval': max(10.0, 1000.0 / frame_rate),
                'autocorr_num_intervals': 50
            })
        elif motion_type == 'confined':
            suggestions.update({
                'interpolation_enabled': True,
                'interpolation_target_class': 3,
                'linear_eigenvalue_ratio_cutoff': 50.0  # More stringent for confined
            })
        
        return suggestions

    # Validation methods
    def validate_config(self) -> List[str]:
        """Validate current configuration and return list of issues."""
        issues = []
        
        # Check file paths
        if self.config.analysis.svm_training_data:
            if not Path(self.config.analysis.svm_training_data).exists():
                issues.append("SVM training data file not found")
        
        # Check parameter ranges
        analysis = self.config.analysis
        if analysis.max_distance <= 0:
            issues.append("Max distance must be positive")
        if analysis.min_track_length < 2:
            issues.append("Min track length must be at least 2")
        if analysis.pixel_size <= 0:
            issues.append("Pixel size must be positive")
        if analysis.frame_rate <= 0:
            issues.append("Frame rate must be positive")
        
        # Check density radii
        if analysis.density_analysis_enabled:
            if not analysis.density_radii or min(analysis.density_radii) <= 0:
                issues.append("Density radii must be positive")
        
        # Check autocorrelation parameters
        if analysis.autocorr_enabled:
            if analysis.autocorr_time_interval <= 0:
                issues.append("Autocorrelation time interval must be positive")
            if analysis.autocorr_num_intervals <= 0:
                issues.append("Number of autocorrelation intervals must be positive")
        
        return issues

    # Configuration export/import
    def export_config(self, file_path: str) -> bool:
        """Export configuration to a file."""
        try:
            config_dict = asdict(self.config)
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            self.logger.info(f"Configuration exported to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting config: {e}")
            return False

    def import_config(self, file_path: str) -> bool:
        """Import configuration from a file."""
        try:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
            
            self.config = self._dict_to_config(config_dict)
            self.save_config()
            self.logger.info(f"Configuration imported from {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error importing config: {e}")
            return False

    # Legacy compatibility methods
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        try:
            obj = self.config
            for part in key.split('.'):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            return default

    def set(self, key: str, value: Any):
        """Set configuration value using dot notation."""
        try:
            parts = key.split('.')
            obj = self.config
            
            # Navigate to parent object
            for part in parts[:-1]:
                obj = getattr(obj, part)
            
            # Set the final attribute
            setattr(obj, parts[-1], value)
            self.configChanged.emit(key, value)
            self.save_config()
            
        except AttributeError as e:
            self.logger.warning(f"Could not set config key '{key}': {e}")

    # Convenience methods for common operations
    def enable_advanced_analysis(self):
        """Enable all advanced analysis features."""
        self.update_analysis_config(
            density_analysis_enabled=True,
            background_subtraction_enabled=True,
            interpolation_enabled=True,
            autocorr_enabled=True,
            localization_precision_enabled=True,
            include_linear_metrics=True,
            svm_multi_round=True
        )

    def set_high_performance_mode(self):
        """Configure for high-performance analysis."""
        self.config.num_threads = min(os.cpu_count() or 4, 8)
        self.config.max_memory_usage_mb = 8192
        self.config.experiment.parallel_processing = True
        self.config.experiment.max_workers = self.config.num_threads
        self.save_config()

    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self.config = ApplicationConfig()
        self._create_default_presets()
        self.save_config()
        self.logger.info("Configuration reset to defaults")
