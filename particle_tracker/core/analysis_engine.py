#!/usr/bin/env python3
"""
Analysis Engine Module
======================

Consolidates all analysis functionality including particle detection, linking,
feature calculation, and classification.
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
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics
import skimage.io as skio
from PyQt6.QtCore import QObject, pyqtSignal, QThread
from tqdm import tqdm

# Import detection and tracking methods
from particle_tracker.analysis.detection import ParticleDetector
from particle_tracker.analysis.linking import ParticleLinker
from particle_tracker.analysis.features import FeatureCalculator
from particle_tracker.analysis.classification import TrajectoryClassifier


@dataclass
class AnalysisParameters:
    """Container for analysis parameters."""
    # Detection parameters
    detection_method: str = "threshold"
    detection_sigma: float = 1.6
    detection_threshold: float = 3.0

    # Linking parameters
    linking_method: str = "nearest_neighbor"
    max_distance: float = 5.0
    max_gap_frames: int = 2
    min_track_length: int = 3

    # Feature calculation parameters
    pixel_size: float = 108.0  # nm per pixel
    frame_rate: float = 10.0   # Hz

    # Classification parameters
    mobility_threshold: float = 2.11

    # SVM parameters
    svm_training_data: Optional[str] = None
    svm_features: List[str] = None

    def __post_init__(self):
        if self.svm_features is None:
            self.svm_features = [
                'radius_gyration', 'asymmetry', 'fracDimension',
                'netDispl', 'Straight', 'kurtosis'
            ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class AnalysisStep(Enum):
    """Enumeration of analysis steps."""
    DETECTION = "detection"
    LINKING = "linking"
    FEATURES = "features"
    CLASSIFICATION = "classification"
    NEAREST_NEIGHBORS = "nearest_neighbors"
    DIFFUSION = "diffusion"
    VELOCITY = "velocity"


class AnalysisWorker(QThread):
    """Worker thread for running analysis steps."""

    progressUpdate = pyqtSignal(str, int)
    stepCompleted = pyqtSignal(str, object)  # step_name, result_data
    analysisCompleted = pyqtSignal(object)   # final_results
    errorOccurred = pyqtSignal(str)

    def __init__(self, data, parameters, steps, parent=None):
        super().__init__(parent)
        self.data = data

        # Convert AnalysisParameters to dict if needed
        if hasattr(parameters, 'to_dict'):
            self.parameters = parameters.to_dict()
        elif isinstance(parameters, dict):
            self.parameters = parameters
        else:
            # Try to convert dataclass to dict
            try:
                self.parameters = asdict(parameters)
            except:
                # Fallback to empty dict
                self.parameters = {}

        self.steps = steps
        self.logger = logging.getLogger(__name__)

    def run(self):
        """Run the analysis pipeline."""
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
                    result = self._run_feature_calculation(current_data)
                elif step == AnalysisStep.CLASSIFICATION:
                    result = self._run_classification(current_data)
                elif step == AnalysisStep.NEAREST_NEIGHBORS:
                    result = self._run_nearest_neighbors(current_data)
                elif step == AnalysisStep.DIFFUSION:
                    result = self._run_diffusion_analysis(current_data)
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

        # Extract detection-specific parameters with more comprehensive mapping
        detection_params = {
            'method': self.parameters.get('detection_method', 'threshold'),
            'sigma': self.parameters.get('detection_sigma', 1.6),
            'threshold': self.parameters.get('detection_threshold', 3.0),
            'min_intensity': self.parameters.get('min_intensity', 100),
            'max_intensity': self.parameters.get('max_intensity', 10000),
            'diameter': self.parameters.get('spot_diameter', 7),  # For trackpy
            'background_subtraction': self.parameters.get('background_subtraction', True)
        }

        self.logger.info(f"Detection parameters: {detection_params}")
        return detector.detect_particles(image_data, **detection_params)

    def _run_linking(self, localization_data):
        """Run particle linking."""
        linker = ParticleLinker(self.parameters)

        # Extract linking-specific parameters
        linking_params = {
            'method': self.parameters.get('linking_method', 'nearest_neighbor'),
            'max_distance': self.parameters.get('max_distance', 5.0),
            'max_gap_frames': self.parameters.get('max_gap_frames', 2),
            'min_track_length': self.parameters.get('min_track_length', 3)
        }

        self.logger.info(f"Linking parameters: {linking_params}")
        return linker.link_particles(localization_data, **linking_params)

    def _run_feature_calculation(self, trajectory_data):
        """Calculate trajectory features."""
        calculator = FeatureCalculator(self.parameters)

        # Log the parameters being used
        feature_params = {
            'pixel_size': self.parameters.get('pixel_size', 108.0),
            'frame_rate': self.parameters.get('frame_rate', 10.0),
            'mobility_threshold': self.parameters.get('mobility_threshold', 2.11)
        }

        self.logger.info(f"Feature calculation parameters: {feature_params}")
        return calculator.calculate_features(trajectory_data)

    def _run_classification(self, feature_data):
        """Run trajectory classification."""
        classifier = TrajectoryClassifier(self.parameters)

        # Extract classification-specific parameters
        method = self.parameters.get('classification_method', 'threshold')

        # Log the method being used
        self.logger.info(f"Running classification using method: {method}")

        # Call classify_trajectories with the method and let it handle all parameters
        return classifier.classify_trajectories(feature_data, method=method)

    def _run_nearest_neighbors(self, data):
        """Calculate nearest neighbor distances."""
        return self._calculate_nearest_neighbors(data)

    def _run_diffusion_analysis(self, data):
        """Add diffusion analysis."""
        return self._add_diffusion_metrics(data)

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
            nn_dist_list.extend(distances[:, 1])  # Second closest (first is self)

        df['nn_distance'] = nn_dist_list
        return df

    def _add_diffusion_metrics(self, df):
        """Add diffusion-related metrics."""
        tracks = df.groupby('track_number')

        diffusion_data = []
        for track_id, track_data in tracks:
            track_data = track_data.sort_values('frame')

            # Set origin
            min_frame = track_data['frame'].min()
            origin_x = track_data[track_data['frame'] == min_frame]['x'].iloc[0]
            origin_y = track_data[track_data['frame'] == min_frame]['y'].iloc[0]

            track_data = track_data.copy()
            track_data['zeroed_x'] = track_data['x'] - origin_x
            track_data['zeroed_y'] = track_data['y'] - origin_y
            track_data['lag_number'] = track_data['frame'] - min_frame
            track_data['distance_from_origin'] = np.sqrt(
                track_data['zeroed_x']**2 + track_data['zeroed_y']**2
            )

            diffusion_data.append(track_data)

        return pd.concat(diffusion_data, ignore_index=True)

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
            track_data['mean_velocity'] = np.mean(velocities[1:])  # Exclude first zero

            velocity_data.append(track_data)

        return pd.concat(velocity_data, ignore_index=True)


class AnalysisEngine(QObject):
    """Main analysis engine coordinating all analysis operations."""

    # Signals
    analysisStarted = pyqtSignal(list)  # steps
    stepCompleted = pyqtSignal(str, object)  # step_name, result
    analysisCompleted = pyqtSignal(object)  # final_result
    progressUpdate = pyqtSignal(str, int)  # message, percentage
    errorOccurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Initialize analysis components
        self.detector = ParticleDetector()
        self.linker = ParticleLinker()
        self.feature_calculator = FeatureCalculator()
        self.classifier = TrajectoryClassifier()

        # Current analysis worker
        self.analysis_worker = None

        self.logger.info("Analysis Engine initialized")

    def run_analysis_pipeline(self, data: Any, parameters: Union[AnalysisParameters, Dict[str, Any]],
                            steps: List[AnalysisStep]):
        """Run a complete analysis pipeline.

        Args:
            data: Input data (image array or DataFrame)
            parameters: Analysis parameters (AnalysisParameters object or dict)
            steps: List of analysis steps to perform
        """
        if self.analysis_worker and self.analysis_worker.isRunning():
            self.logger.warning("Analysis already running")
            return

        self.logger.info(f"Starting analysis pipeline with {len(steps)} steps")

        # Create and configure worker
        self.analysis_worker = AnalysisWorker(data, parameters, steps)
        self.analysis_worker.progressUpdate.connect(self.progressUpdate)
        self.analysis_worker.stepCompleted.connect(self.stepCompleted)
        self.analysis_worker.analysisCompleted.connect(self.analysisCompleted)
        self.analysis_worker.errorOccurred.connect(self.errorOccurred)

        # Start analysis
        self.analysisStarted.emit([step.value for step in steps])
        self.analysis_worker.start()

    def stop_analysis(self):
        """Stop the current analysis."""
        if self.analysis_worker and self.analysis_worker.isRunning():
            self.analysis_worker.terminate()
            self.analysis_worker.wait()
            self.logger.info("Analysis stopped")

    def calculate_radius_of_gyration(self, trajectory_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate radius of gyration for trajectories."""
        return self.feature_calculator._calculate_radius_of_gyration(trajectory_data)

    def calculate_trajectory_features(self, trajectory_data: pd.DataFrame,
                                    parameters: Union[AnalysisParameters, Dict[str, Any]]) -> pd.DataFrame:
        """Calculate comprehensive trajectory features."""
        # Convert parameters if needed
        if hasattr(parameters, 'to_dict'):
            params_dict = parameters.to_dict()
        elif isinstance(parameters, dict):
            params_dict = parameters
        else:
            try:
                params_dict = asdict(parameters)
            except:
                params_dict = {}

        self.feature_calculator.update_parameters(params_dict)
        return self.feature_calculator.calculate_features(trajectory_data)

    def classify_trajectories_svm(self, feature_data: pd.DataFrame,
                                training_data_path: str) -> pd.DataFrame:
        """Classify trajectories using SVM."""
        return self.classifier._classify_svm(feature_data)

    def detect_particles_threshold(self, image_data: np.ndarray,
                                 parameters: Union[AnalysisParameters, Dict[str, Any]]) -> pd.DataFrame:
        """Detect particles using threshold method."""
        # Convert parameters if needed
        if hasattr(parameters, 'to_dict'):
            params_dict = parameters.to_dict()
        elif isinstance(parameters, dict):
            params_dict = parameters
        else:
            try:
                params_dict = asdict(parameters)
            except:
                params_dict = {}

        self.detector.update_parameters(params_dict)
        return self.detector.detect_particles(image_data, method='threshold')

    def link_particles_nearest_neighbor(self, localizations: pd.DataFrame,
                                      parameters: Union[AnalysisParameters, Dict[str, Any]]) -> pd.DataFrame:
        """Link particles using nearest neighbor method."""
        # Convert parameters if needed
        if hasattr(parameters, 'to_dict'):
            params_dict = parameters.to_dict()
        elif isinstance(parameters, dict):
            params_dict = parameters
        else:
            try:
                params_dict = asdict(parameters)
            except:
                params_dict = {}

        self.linker.update_parameters(params_dict)
        return self.linker.link_particles(localizations, method='nearest_neighbor')

    def link_particles_trackpy(self, localizations: pd.DataFrame,
                             parameters: Union[AnalysisParameters, Dict[str, Any]]) -> pd.DataFrame:
        """Link particles using trackpy library."""
        # Convert parameters if needed
        if hasattr(parameters, 'to_dict'):
            params_dict = parameters.to_dict()
        elif isinstance(parameters, dict):
            params_dict = parameters
        else:
            try:
                params_dict = asdict(parameters)
            except:
                params_dict = {}

        self.linker.update_parameters(params_dict)
        return self.linker.link_particles(localizations, method='trackpy')

    def calculate_msd(self, trajectory_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean squared displacement."""
        tracks = trajectory_data.groupby('track_number')
        msd_data = []

        for track_id, track in tracks:
            track = track.sort_values('frame')
            positions = track[['x', 'y']].values

            if len(positions) < 3:
                continue

            # Calculate MSD for different lag times
            max_lag = min(len(positions) - 1, 20)  # Limit to reasonable lag times
            msd_values = []
            lag_times = []

            for lag in range(1, max_lag + 1):
                displacements = positions[lag:] - positions[:-lag]
                squared_displacements = np.sum(displacements**2, axis=1)
                msd = np.mean(squared_displacements)

                msd_values.append(msd)
                lag_times.append(lag)

            # Add MSD data to each point in the track
            track_msd_data = track.copy()
            track_msd_data['msd_slope'] = np.nan
            track_msd_data['msd_intercept'] = np.nan
            track_msd_data['diffusion_coefficient'] = np.nan

            if len(msd_values) >= 3:
                # Fit linear regression to get diffusion coefficient
                log_lag = np.log(lag_times)
                log_msd = np.log(msd_values)

                # Fit to first few points for better linear approximation
                fit_points = min(5, len(log_lag))
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    log_lag[:fit_points], log_msd[:fit_points]
                )

                # Diffusion coefficient D = MSD / (4 * dt) for 2D
                # Here we use the slope of log-log plot
                D = np.exp(intercept) / 4.0  # Simplified calculation

                track_msd_data['msd_slope'] = slope
                track_msd_data['msd_intercept'] = intercept
                track_msd_data['diffusion_coefficient'] = D

            msd_data.append(track_msd_data)

        return pd.concat(msd_data, ignore_index=True) if msd_data else pd.DataFrame()

    def filter_tracks_by_length(self, trajectory_data: pd.DataFrame,
                               min_length: int) -> pd.DataFrame:
        """Filter tracks by minimum length."""
        track_lengths = trajectory_data.groupby('track_number').size()
        valid_tracks = track_lengths[track_lengths >= min_length].index
        return trajectory_data[trajectory_data['track_number'].isin(valid_tracks)]

    def get_analysis_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for analysis results."""
        summary = {}

        if 'track_number' in data.columns:
            summary['n_tracks'] = data['track_number'].nunique()
            summary['n_localizations'] = len(data)

            track_lengths = data.groupby('track_number').size()
            summary['mean_track_length'] = track_lengths.mean()
            summary['median_track_length'] = track_lengths.median()
            summary['min_track_length'] = track_lengths.min()
            summary['max_track_length'] = track_lengths.max()

        if 'frame' in data.columns:
            summary['n_frames'] = data['frame'].nunique()
            summary['frame_range'] = (data['frame'].min(), data['frame'].max())

        if 'x' in data.columns and 'y' in data.columns:
            summary['x_range'] = (data['x'].min(), data['x'].max())
            summary['y_range'] = (data['y'].min(), data['y'].max())

        # Add feature-specific summaries
        feature_columns = [
            'radius_gyration', 'asymmetry', 'fracDimension', 'velocity',
            'diffusion_coefficient', 'nn_distance'
        ]

        for col in feature_columns:
            if col in data.columns:
                values = data[col].dropna()
                if len(values) > 0:
                    summary[f'{col}_mean'] = values.mean()
                    summary[f'{col}_std'] = values.std()
                    summary[f'{col}_median'] = values.median()

        return summary

    def export_analysis_report(self, data: pd.DataFrame, output_path: str) -> bool:
        """Export a comprehensive analysis report."""
        try:
            summary = self.get_analysis_summary(data)

            # Create report
            report_lines = [
                "Particle Tracking Analysis Report",
                "=" * 40,
                "",
                f"Number of tracks: {summary.get('n_tracks', 'N/A')}",
                f"Number of localizations: {summary.get('n_localizations', 'N/A')}",
                f"Number of frames: {summary.get('n_frames', 'N/A')}",
                f"Mean track length: {summary.get('mean_track_length', 'N/A'):.2f}",
                f"Median track length: {summary.get('median_track_length', 'N/A'):.2f}",
                "",
                "Feature Statistics:",
                "-" * 20,
            ]

            # Add feature statistics
            feature_stats = {k: v for k, v in summary.items() if '_mean' in k}
            for feature_name, mean_val in feature_stats.items():
                base_name = feature_name.replace('_mean', '')
                std_val = summary.get(f'{base_name}_std', 0)
                report_lines.append(f"{base_name}: {mean_val:.4f} ± {std_val:.4f}")

            # Write report
            with open(output_path, 'w') as f:
                f.write('\n'.join(report_lines))

            self.logger.info(f"Analysis report exported to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting report: {e}")
            return False
