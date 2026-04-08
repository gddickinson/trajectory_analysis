#!/usr/bin/env python3
"""
Enhanced Analysis Engine Module
===============================

Updated to integrate the comprehensive feature calculation capabilities
from the enhanced features module.
"""

import logging
import math
from typing import Optional, Dict, List, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats, spatial
from sklearn.neighbors import KDTree
from PyQt6.QtCore import QObject, pyqtSignal, QThread
from tqdm import tqdm

# Import detection and tracking methods
from particle_tracker.analysis.detection import ParticleDetector
from particle_tracker.analysis.linking import ParticleLinker

# Import the enhanced features module
from particle_tracker.analysis.features import FeatureCalculator
from particle_tracker.analysis.classification import TrajectoryClassifier


@dataclass
class AnalysisParameters:
    """Enhanced container for analysis parameters."""
    # Detection parameters
    detection_method: str = "threshold"
    detection_sigma: float = 1.6
    detection_threshold: float = 3.0

    # Linking parameters
    linking_method: str = "nearest_neighbor"
    max_distance: float = 5.0
    max_gap_frames: int = 2
    min_track_length: int = 3

    # Basic feature calculation parameters
    pixel_size: float = 108.0  # nm per pixel
    frame_rate: float = 10.0   # Hz

    # Enhanced feature parameters
    calculate_density: bool = True
    density_radii: List[int] = None
    calculate_advanced_shape: bool = True
    calculate_scaled_rg: bool = True
    calculate_diffusion: bool = True
    calculate_precision: bool = True
    interpolate_trajectories: bool = False

    # Advanced shape analysis parameters
    linear_eigenvalue_threshold: float = 20.0
    linear_alignment_threshold: float = 0.7

    # Classification parameters
    mobility_threshold: float = 2.11

    # SVM parameters
    svm_training_data: Optional[str] = None
    svm_features: List[str] = None

    # Background subtraction parameters
    roi_background_data: Optional[np.ndarray] = None
    camera_black_data: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.svm_features is None:
            self.svm_features = [
                'radius_gyration', 'asymmetry', 'fracDimension',
                'netDispl', 'Straight', 'kurtosis'
            ]
        if self.density_radii is None:
            self.density_radii = [3, 5, 10, 20, 30]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class AnalysisStep(Enum):
    """Enhanced enumeration of analysis steps."""
    DETECTION = "detection"
    LINKING = "linking"
    FEATURES = "features"
    ENHANCED_FEATURES = "enhanced_features"
    DENSITY_ANALYSIS = "density_analysis"
    ADVANCED_SHAPE = "advanced_shape"
    SCALED_RG = "scaled_rg"
    DIFFUSION_ANALYSIS = "diffusion_analysis"
    PRECISION_ANALYSIS = "precision_analysis"
    CLASSIFICATION = "classification"
    NEAREST_NEIGHBORS = "nearest_neighbors"
    VELOCITY = "velocity"


class AnalysisWorker(QThread):
    """Enhanced worker thread for running analysis steps."""

    progressUpdate = pyqtSignal(str, int)
    stepCompleted = pyqtSignal(str, object)
    analysisCompleted = pyqtSignal(object)
    errorOccurred = pyqtSignal(str)

    def __init__(self, data, parameters, steps, parent=None):
        super().__init__(parent)
        self.data = data
        self.parameters = self._prepare_parameters(parameters)
        self.steps = steps
        self.logger = logging.getLogger(__name__)

    def _prepare_parameters(self, parameters):
        """Prepare parameters for analysis."""
        if hasattr(parameters, 'to_dict'):
            return parameters.to_dict()
        elif isinstance(parameters, dict):
            return parameters
        else:
            try:
                return asdict(parameters)
            except:
                return {}

    def run(self):
        """Run the enhanced analysis pipeline."""
        try:
            current_data = self.data.copy() if isinstance(self.data, pd.DataFrame) else self.data

            for i, step in enumerate(self.steps):
                self.progressUpdate.emit(f"Running {step.value}...",
                                       int(100 * i / len(self.steps)))

                if step == AnalysisStep.DETECTION:
                    result = self._run_detection(current_data)
                elif step == AnalysisStep.LINKING:
                    result = self._run_linking(current_data)
                elif step == AnalysisStep.FEATURES:
                    result = self._run_basic_features(current_data)
                elif step == AnalysisStep.ENHANCED_FEATURES:
                    result = self._run_enhanced_features(current_data)
                elif step == AnalysisStep.DENSITY_ANALYSIS:
                    result = self._run_density_analysis(current_data)
                elif step == AnalysisStep.ADVANCED_SHAPE:
                    result = self._run_advanced_shape(current_data)
                elif step == AnalysisStep.SCALED_RG:
                    result = self._run_scaled_rg(current_data)
                elif step == AnalysisStep.DIFFUSION_ANALYSIS:
                    result = self._run_diffusion_analysis(current_data)
                elif step == AnalysisStep.PRECISION_ANALYSIS:
                    result = self._run_precision_analysis(current_data)
                elif step == AnalysisStep.CLASSIFICATION:
                    result = self._run_classification(current_data)
                elif step == AnalysisStep.NEAREST_NEIGHBORS:
                    result = self._run_nearest_neighbors(current_data)
                elif step == AnalysisStep.VELOCITY:
                    result = self._run_velocity_analysis(current_data)
                else:
                    continue

                current_data = result
                self.stepCompleted.emit(step.value, result)

            self.progressUpdate.emit("Analysis complete", 100)
            self.analysisCompleted.emit(current_data)

        except Exception as e:
            self.logger.error(f"Analysis error: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.errorOccurred.emit(str(e))

    def _run_detection(self, image_data):
        """Run particle detection."""
        detector = ParticleDetector(self.parameters)
        detection_params = {
            'method': self.parameters.get('detection_method', 'threshold'),
            'sigma': self.parameters.get('detection_sigma', 1.6),
            'threshold': self.parameters.get('detection_threshold', 3.0),
            'min_intensity': self.parameters.get('min_intensity', 100),
            'max_intensity': self.parameters.get('max_intensity', 10000),
            'diameter': self.parameters.get('spot_diameter', 7),
            'background_subtraction': self.parameters.get('background_subtraction', True)
        }
        return detector.detect_particles(image_data, **detection_params)

    def _run_linking(self, localization_data):
        """Run particle linking step with error handling."""
        try:
            # Check if we have valid data
            if localization_data is None or len(localization_data) == 0:
                self.logger.warning("No particles detected - skipping linking step")
                # Return empty DataFrame with correct structure
                import pandas as pd
                empty_result = pd.DataFrame(columns=['frame', 'x', 'y', 'particle'])
                return empty_result

            # Check for required columns
            required_columns = ['frame', 'x', 'y']
            missing_columns = [col for col in required_columns if col not in localization_data.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns for linking: {missing_columns}")
                self.logger.error(f"Available columns: {list(localization_data.columns)}")
                # Try to create missing columns if possible
                if 'frame' not in localization_data.columns and 't' in localization_data.columns:
                    localization_data['frame'] = localization_data['t']
                    self.logger.info("Mapped 't' column to 'frame'")
                else:
                    raise ValueError("Cannot proceed with linking - missing required columns")

            self.logger.info(f"Linking {len(localization_data)} detections across {localization_data['frame'].nunique()} frames")

            # Get linking parameters
            linking_params = self.parameters.get_linking_parameters()

            # Initialize linker
            from ..analysis.linking import ParticleLinker
            linker = ParticleLinker(logger=self.logger)

            # Run linking
            return linker.link_particles(localization_data, **linking_params)

        except Exception as e:
            self.logger.error(f"Linking failed: {e}")
            # Return empty result rather than crashing
            import pandas as pd
            empty_result = pd.DataFrame(columns=['frame', 'x', 'y', 'particle'])
            return empty_result

    def _run_basic_features(self, trajectory_data):
        """Run basic feature calculation for backward compatibility."""
        # Use the enhanced calculator but only basic features
        calculator = FeatureCalculator(self.parameters)
        return calculator._calculate_basic_features(trajectory_data)

    def _run_enhanced_features(self, trajectory_data):
        """Run comprehensive enhanced feature calculation."""
        calculator = FeatureCalculator(self.parameters)
        return calculator.calculate_features(trajectory_data)

    def _run_density_analysis(self, trajectory_data):
        """Run multi-radius density analysis."""
        calculator = FeatureCalculator(self.parameters)
        return calculator.density_analyzer.analyze_frame_density(trajectory_data)

    def _run_advanced_shape(self, trajectory_data):
        """Run advanced shape analysis."""
        calculator = FeatureCalculator(self.parameters)
        return calculator._calculate_advanced_shape_features(trajectory_data)

    def _run_scaled_rg(self, trajectory_data):
        """Run scaled radius of gyration calculation."""
        calculator = FeatureCalculator(self.parameters)
        return calculator._calculate_scaled_rg_features(trajectory_data)

    def _run_diffusion_analysis(self, trajectory_data):
        """Run enhanced diffusion analysis."""
        calculator = FeatureCalculator(self.parameters)
        return calculator._calculate_diffusion_metrics(trajectory_data)

    def _run_precision_analysis(self, trajectory_data):
        """Run localization precision analysis."""
        # For now, we'll include this in the enhanced features
        # In the future, this could be a separate analysis step
        return trajectory_data

    def _run_classification(self, feature_data):
        """Run trajectory classification."""
        classifier = TrajectoryClassifier(self.parameters)
        method = self.parameters.get('classification_method', 'threshold')
        return classifier.classify_trajectories(feature_data, method=method)

    def _run_nearest_neighbors(self, data):
        """Calculate traditional nearest neighbor distances."""
        return self._calculate_nearest_neighbors(data)

    def _run_velocity_analysis(self, data):
        """Add velocity analysis."""
        return self._add_velocity_metrics(data)

    def _calculate_nearest_neighbors(self, df):
        """Calculate nearest neighbor distances for each frame."""
        df = df.sort_values(by=['frame'])
        nn_dist_list = []
        frames = df['frame'].unique()

        for frame in frames:
            frame_data = df[df['frame'] == frame]
            if len(frame_data) < 2:
                nn_dist_list.extend([np.nan] * len(frame_data))
                continue

            coords = frame_data[['x', 'y']].values
            tree = KDTree(coords)
            distances, _ = tree.query(coords, k=2)
            nn_dist_list.extend(distances[:, 1])

        df['nn_distance'] = nn_dist_list
        return df

    def _add_velocity_metrics(self, df):
        """Add velocity-related metrics."""
        tracks = df.groupby('track_number')
        velocity_data = []

        for track_id, track_data in tracks:
            track_data = track_data.sort_values('frame').copy()

            # Calculate step displacements
            x_diff = np.diff(track_data['x'].values)
            y_diff = np.diff(track_data['y'].values)
            step_distances = np.sqrt(x_diff**2 + y_diff**2)

            # Calculate time differences
            time_diff = np.diff(track_data['frame'].values)

            # Calculate velocities
            velocities = np.concatenate([[0], step_distances / np.maximum(time_diff, 1)])
            track_data['velocity'] = velocities
            track_data['mean_velocity'] = np.mean(velocities[1:])

            velocity_data.append(track_data)

        return pd.concat(velocity_data, ignore_index=True)


class AnalysisEngine(QObject):
    """Enhanced main analysis engine."""

    # Signals
    analysisStarted = pyqtSignal(list)
    stepCompleted = pyqtSignal(str, object)
    analysisCompleted = pyqtSignal(object)
    progressUpdate = pyqtSignal(str, int)
    errorOccurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Initialize analysis components
        self.detector = ParticleDetector()
        self.linker = ParticleLinker()
        self.feature_calculator = FeatureCalculator()  # Enhanced calculator
        self.classifier = TrajectoryClassifier()

        # Current analysis worker
        self.analysis_worker = None

        self.logger.info("Enhanced Analysis Engine initialized")

    def run_analysis_pipeline(self, data: Any, parameters: Union[AnalysisParameters, Dict[str, Any]],
                            steps: List[AnalysisStep]):
        """Run enhanced analysis pipeline."""
        if self.analysis_worker and self.analysis_worker.isRunning():
            self.logger.warning("Analysis already running")
            return

        self.logger.info(f"Starting enhanced analysis pipeline with {len(steps)} steps")

        # Create and configure worker
        self.analysis_worker = AnalysisWorker(data, parameters, steps)
        self.analysis_worker.progressUpdate.connect(self.progressUpdate)
        self.analysis_worker.stepCompleted.connect(self.stepCompleted)
        self.analysis_worker.analysisCompleted.connect(self.analysisCompleted)
        self.analysis_worker.errorOccurred.connect(self.errorOccurred)

        # Start analysis
        self.analysisStarted.emit([step.value for step in steps])
        self.analysis_worker.start()

    def run_comprehensive_analysis(self, data: Any, parameters: Union[AnalysisParameters, Dict[str, Any]]):
        """Run the complete enhanced analysis pipeline."""
        comprehensive_steps = [
            AnalysisStep.DETECTION,
            AnalysisStep.LINKING,
            AnalysisStep.ENHANCED_FEATURES,  # Use enhanced features instead of basic
            AnalysisStep.CLASSIFICATION
        ]

        self.run_analysis_pipeline(data, parameters, comprehensive_steps)

    def calculate_enhanced_features_only(self, trajectory_data: pd.DataFrame,
                                       parameters: Union[AnalysisParameters, Dict[str, Any]]) -> pd.DataFrame:
        """Calculate only enhanced features for existing trajectory data."""
        if hasattr(parameters, 'to_dict'):
            params_dict = parameters.to_dict()
        elif isinstance(parameters, dict):
            params_dict = parameters
        else:
            try:
                params_dict = asdict(parameters)
            except:
                params_dict = {}

        self.feature_calculator = FeatureCalculator(params_dict)
        return self.feature_calculator.calculate_features(trajectory_data)

    def get_enhanced_analysis_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get enhanced analysis summary with new metrics."""
        return self.feature_calculator.get_analysis_summary(data)

    def stop_analysis(self):
        """Stop the current analysis."""
        if self.analysis_worker and self.analysis_worker.isRunning():
            self.analysis_worker.terminate()
            self.analysis_worker.wait()
            self.logger.info("Analysis stopped")

    # Convenience methods for specific analyses
    def run_density_analysis_only(self, trajectory_data: pd.DataFrame,
                                 radii: List[int] = None) -> pd.DataFrame:
        """Run only multi-radius density analysis."""
        params = {'density_radii': radii or [3, 5, 10, 20, 30]}
        calculator = FeatureCalculator(params)
        return calculator.density_analyzer.analyze_frame_density(trajectory_data)

    def run_advanced_shape_analysis_only(self, trajectory_data: pd.DataFrame) -> pd.DataFrame:
        """Run only advanced shape analysis."""
        calculator = FeatureCalculator()
        return calculator._calculate_advanced_shape_features(trajectory_data)

    def run_scaled_rg_analysis_only(self, trajectory_data: pd.DataFrame,
                                   mobility_threshold: float = 2.11) -> pd.DataFrame:
        """Run only scaled radius of gyration analysis."""
        params = {'mobility_threshold': mobility_threshold}
        calculator = FeatureCalculator(params)
        return calculator._calculate_scaled_rg_features(trajectory_data)

    # Legacy compatibility methods
    def calculate_trajectory_features(self, trajectory_data: pd.DataFrame,
                                    parameters: Union[AnalysisParameters, Dict[str, Any]]) -> pd.DataFrame:
        """Legacy method - now uses enhanced features."""
        return self.calculate_enhanced_features_only(trajectory_data, parameters)

    def get_analysis_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Legacy method - now returns enhanced summary."""
        return self.get_enhanced_analysis_summary(data)

    # Additional convenience methods
    def export_enhanced_analysis_report(self, data: pd.DataFrame, output_path: str) -> bool:
        """Export comprehensive enhanced analysis report."""
        try:
            summary = self.get_enhanced_analysis_summary(data)

            report_lines = [
                "Enhanced Particle Tracking Analysis Report",
                "=" * 50,
                "",
                f"Number of tracks: {summary.get('n_tracks', 'N/A')}",
                f"Number of localizations: {summary.get('n_localizations', 'N/A')}",
                f"Mean track length: {summary.get('mean_track_length', 'N/A'):.2f}",
                "",
                "Mobility Classification:",
                "-" * 25,
            ]

            # Add mobility distribution
            if 'mobility_distribution' in summary:
                for category, count in summary['mobility_distribution'].items():
                    report_lines.append(f"{category}: {count}")

                if 'percent_mobile' in summary:
                    report_lines.append(f"Percent mobile: {summary['percent_mobile']:.1f}%")

            # Add linearity distribution
            if 'linearity_distribution' in summary:
                report_lines.extend([
                    "",
                    "Linearity Classification:",
                    "-" * 25,
                ])
                for category, count in summary['linearity_distribution'].items():
                    report_lines.append(f"{category}: {count}")

            # Add feature statistics
            report_lines.extend([
                "",
                "Enhanced Feature Statistics:",
                "-" * 30,
            ])

            feature_stats = {k: v for k, v in summary.items() if '_mean' in k}
            for feature_name, mean_val in feature_stats.items():
                base_name = feature_name.replace('_mean', '')
                std_val = summary.get(f'{base_name}_std', 0)
                report_lines.append(f"{base_name}: {mean_val:.4f} ± {std_val:.4f}")

            # Add density analysis results
            if 'density_metrics' in summary:
                report_lines.extend([
                    "",
                    "Density Analysis:",
                    "-" * 18,
                ])
                for radius, stats in summary['density_metrics'].items():
                    report_lines.append(f"{radius}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")

            # Write report
            with open(output_path, 'w') as f:
                f.write('\n'.join(report_lines))

            self.logger.info(f"Enhanced analysis report exported to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting enhanced report: {e}")
            return False
