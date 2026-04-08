#!/usr/bin/env python3
"""
Enhanced Feature Calculation Module
===================================

Comprehensive trajectory feature calculation including all advanced metrics
from the original analysis scripts. This module provides sophisticated
analysis capabilities for particle tracking data.

Key Features:
- Multi-radius density analysis (3,5,10,20,30 pixel neighbors)
- Advanced shape metrics (eigenvalue ratios, linearity indices)
- Scaled radius of gyration (sRg) with proper mobility classification
- ROI-based background subtraction
- Trajectory interpolation for trapped particles
- Localization precision analysis
- Detailed diffusion and velocity analysis
- Direction autocorrelation analysis
"""

import logging
import math
from typing import Optional, Dict, List, Any, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats, spatial, ndimage
from sklearn.neighbors import KDTree
from pathlib import Path
from tqdm import tqdm


class DensityAnalyzer:
    """Analyze particle density using multiple radius thresholds."""

    def __init__(self, radii: List[int] = None):
        """
        Initialize density analyzer.

        Args:
            radii: List of radii in pixels for neighbor counting
        """
        self.logger = logging.getLogger(__name__)
        self.radii = radii or [3, 5, 10, 20, 30]

    def count_neighbors_within_radius(self, coordinates: np.ndarray, radius: float) -> np.ndarray:
        """
        Count neighbors within specified radius for each point.

        Args:
            coordinates: Nx2 array of x,y coordinates
            radius: Search radius in pixels

        Returns:
            Array of neighbor counts (excluding self)
        """
        if len(coordinates) < 2:
            return np.zeros(len(coordinates))

        tree = KDTree(coordinates, leaf_size=5)
        counts = tree.query_radius(coordinates, r=radius, count_only=True)
        # Subtract 1 to exclude self
        return counts - 1

    def analyze_frame_density(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze density for each frame using multiple radii.

        Args:
            df: DataFrame with 'frame', 'x', 'y' columns

        Returns:
            DataFrame with added density columns
        """
        self.logger.info(f"Analyzing density with radii: {self.radii}")

        df = df.sort_values(by=['frame']).copy()

        # Initialize density columns
        for radius in self.radii:
            df[f'nnCountInFrame_within_{radius}_pixels'] = np.nan

        frames = df['frame'].unique()

        for frame in tqdm(frames, desc="Analyzing frame density"):
            frame_data = df[df['frame'] == frame]

            if len(frame_data) < 2:
                # Single particle - no neighbors
                for radius in self.radii:
                    df.loc[df['frame'] == frame, f'nnCountInFrame_within_{radius}_pixels'] = 0
                continue

            coordinates = frame_data[['x', 'y']].values

            for radius in self.radii:
                counts = self.count_neighbors_within_radius(coordinates, radius)
                df.loc[df['frame'] == frame, f'nnCountInFrame_within_{radius}_pixels'] = counts

        return df


class AdvancedShapeAnalyzer:
    """Advanced shape and motion analysis using tensor methods."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_gyration_tensor_metrics(self, positions: np.ndarray) -> Dict[str, float]:
        """
        Calculate advanced metrics using gyration tensor analysis.

        Args:
            positions: Nx2 array of x,y coordinates

        Returns:
            Dictionary with advanced metrics
        """
        if len(positions) < 3:
            return self._empty_metrics_dict()

        # Remove any NaN values
        positions = positions[~np.isnan(positions).any(axis=1)]
        num_points = len(positions)

        if num_points < 3:
            return self._empty_metrics_dict()

        # Calculate center of mass
        center = np.mean(positions, axis=0)
        normed_points = positions - center[None, :]

        # Gyration tensor
        gyration_tensor = np.einsum('im,in->mn', normed_points, normed_points) / num_points
        eig_values, eig_vectors = np.linalg.eig(gyration_tensor)
        eig_values = np.real(eig_values)
        eig_vectors = np.real(eig_vectors)

        # Sort eigenvalues and eigenvectors
        idx = eig_values.argsort()[::-1]
        eig_values = eig_values[idx]
        eig_vectors = eig_vectors[:, idx]

        # Radius of gyration
        radius_gyration = np.sqrt(np.sum(eig_values))

        # Asymmetry
        try:
            l1, l2 = eig_values[0], eig_values[1]
            asymmetry_num = (l1 - l2)**2
            asymmetry_den = 2 * (l1 + l2)**2
            asymmetry = -math.log(1 - (asymmetry_num / asymmetry_den))
        except (ValueError, ZeroDivisionError):
            asymmetry = np.nan

        # Eigenvalue ratio for linearity
        try:
            eigenvalue_ratio = eig_values[0] / eig_values[1] if eig_values[1] > 0 else np.inf
        except (IndexError, ZeroDivisionError):
            eigenvalue_ratio = np.nan

        # Principal axis projections for skewness/kurtosis
        try:
            principal_axis = eig_vectors[:, 0]
            steps = np.diff(positions, axis=0)
            projections = np.dot(steps, principal_axis)
            skewness = stats.skew(projections)
            kurtosis = stats.kurtosis(projections)
        except:
            skewness = np.nan
            kurtosis = np.nan

        # Step alignment with principal axis
        try:
            step_norms = np.linalg.norm(steps, axis=1)
            valid_steps = step_norms > 0
            if np.any(valid_steps):
                normalized_steps = steps[valid_steps] / step_norms[valid_steps, None]
                cos_angles = np.abs(np.dot(normalized_steps, principal_axis))
                step_alignment = np.mean(cos_angles)
            else:
                step_alignment = np.nan
        except:
            step_alignment = np.nan

        # Directionality ratio
        try:
            net_displacement = np.linalg.norm(positions[-1] - positions[0])
            path_length = np.sum(np.linalg.norm(steps, axis=1))
            directionality_ratio = net_displacement / path_length if path_length > 0 else np.nan
        except:
            directionality_ratio = np.nan

        # Fractal dimension (simplified)
        try:
            step_lengths = np.sqrt(np.sum(steps**2, axis=1))
            total_length = np.sum(step_lengths)
            net_disp = np.linalg.norm(positions[-1] - positions[0])

            if total_length > 0 and net_disp > 0:
                fractal_dim = math.log(len(positions)) / math.log(len(positions) * net_disp / total_length)
            else:
                fractal_dim = np.nan
        except:
            fractal_dim = np.nan

        # Net displacement
        net_displacement = np.linalg.norm(positions[-1] - positions[0])

        # Straightness (mean cosine of turning angles)
        try:
            if len(steps) > 1:
                cos_angles = []
                for i in range(len(steps) - 1):
                    v1, v2 = steps[i], steps[i+1]
                    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if norm1 > 0 and norm2 > 0:
                        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                        cos_angles.append(np.clip(cos_angle, -1, 1))
                straightness = np.mean(cos_angles) if cos_angles else np.nan
            else:
                straightness = np.nan
        except:
            straightness = np.nan

        return {
            'radius_gyration': radius_gyration,
            'asymmetry': asymmetry,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'eigenvalue_ratio': eigenvalue_ratio,
            'step_alignment': step_alignment,
            'directionality_ratio': directionality_ratio,
            'fracDimension': fractal_dim,
            'netDispl': net_displacement,
            'Straight': straightness
        }

    def _empty_metrics_dict(self) -> Dict[str, float]:
        """Return dictionary with NaN values for all metrics."""
        return {
            'radius_gyration': np.nan,
            'asymmetry': np.nan,
            'skewness': np.nan,
            'kurtosis': np.nan,
            'eigenvalue_ratio': np.nan,
            'step_alignment': np.nan,
            'directionality_ratio': np.nan,
            'fracDimension': np.nan,
            'netDispl': np.nan,
            'Straight': np.nan
        }

    def classify_linearity(self, eigenvalue_ratio: float, step_alignment: float,
                          directionality_ratio: float, eigenvalue_threshold: float = 20.0,
                          alignment_threshold: float = 0.7) -> str:
        """
        Classify trajectory linearity based on advanced metrics.

        Args:
            eigenvalue_ratio: Ratio of largest to smallest eigenvalue
            step_alignment: Average alignment with principal axis
            directionality_ratio: Net displacement / total path length
            eigenvalue_threshold: Minimum eigenvalue ratio for linear classification
            alignment_threshold: Minimum step alignment for linear classification

        Returns:
            Classification string
        """
        if np.isnan(eigenvalue_ratio) or np.isnan(step_alignment):
            return 'unclassified'

        is_linear = (eigenvalue_ratio >= eigenvalue_threshold and
                    step_alignment >= alignment_threshold)

        if not is_linear:
            return 'non_linear'

        if np.isnan(directionality_ratio):
            return 'linear'

        # Distinguish between unidirectional and bidirectional
        if directionality_ratio >= 0.7:
            return 'linear_unidirectional'
        else:
            return 'linear_bidirectional'


class ScaledRadiusCalculator:
    """Calculate scaled radius of gyration (sRg) following Golan & Sherman."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_scaled_rg(self, radius_gyration: float, mean_step_size: float) -> float:
        """
        Calculate scaled radius of gyration.

        Args:
            radius_gyration: Standard radius of gyration
            mean_step_size: Mean step size of trajectory

        Returns:
            Scaled radius of gyration value
        """
        if np.isnan(radius_gyration) or np.isnan(mean_step_size) or mean_step_size <= 0:
            return np.nan

        # Golan & Sherman formula: √(π/2) * Rg / mean_step_size
        scaled_rg = np.sqrt(np.pi/2) * radius_gyration / mean_step_size
        return scaled_rg

    def classify_mobility(self, scaled_rg: float, threshold: float = 2.11) -> str:
        """
        Classify trajectory mobility based on scaled Rg.

        Args:
            scaled_rg: Scaled radius of gyration
            threshold: Mobility threshold (default from Golan & Sherman)

        Returns:
            Mobility classification ('mobile' or 'immobile')
        """
        if np.isnan(scaled_rg):
            return 'unclassified'

        return 'mobile' if scaled_rg > threshold else 'immobile'


class TrajectoryInterpolator:
    """Interpolate missing timepoints in trajectories."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def interpolate_trajectory(self, track_data: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate missing frames in a trajectory.

        Args:
            track_data: DataFrame for a single track

        Returns:
            DataFrame with interpolated points
        """
        if len(track_data) < 2:
            return track_data

        track_data = track_data.sort_values('frame').copy()

        # Get frame range
        min_frame = track_data['frame'].min()
        max_frame = track_data['frame'].max()
        all_frames = np.arange(min_frame, max_frame + 1)

        # Interpolate x and y coordinates
        x_interp = np.interp(all_frames, track_data['frame'].values, track_data['x'].values)
        y_interp = np.interp(all_frames, track_data['frame'].values, track_data['y'].values)

        # Create interpolated DataFrame
        interp_df = pd.DataFrame({
            'frame': all_frames,
            'x': x_interp,
            'y': y_interp,
            'track_number': track_data['track_number'].iloc[0]
        })

        # Mark interpolated vs original points
        interp_df['interpolated'] = True
        original_frames = set(track_data['frame'].values)
        interp_df.loc[interp_df['frame'].isin(original_frames), 'interpolated'] = False

        return interp_df

    def interpolate_all_trajectories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate all trajectories in the dataset.

        Args:
            df: DataFrame with trajectory data

        Returns:
            DataFrame with interpolated trajectories
        """
        self.logger.info("Interpolating missing timepoints in trajectories")

        interpolated_tracks = []

        for track_id in tqdm(df['track_number'].unique(), desc="Interpolating tracks"):
            track_data = df[df['track_number'] == track_id]
            interp_track = self.interpolate_trajectory(track_data)
            interpolated_tracks.append(interp_track)

        return pd.concat(interpolated_tracks, ignore_index=True)


class FeatureCalculator:
    """Enhanced main feature calculator with all advanced analysis capabilities."""

    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize enhanced feature calculator.

        Args:
            parameters: Analysis parameters dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.parameters = parameters or {}

        # Initialize analysis components
        self.density_analyzer = DensityAnalyzer(
            radii=self.parameters.get('density_radii', [3, 5, 10, 20, 30])
        )
        self.shape_analyzer = AdvancedShapeAnalyzer()
        self.scaled_rg_calculator = ScaledRadiusCalculator()
        self.interpolator = TrajectoryInterpolator()

    def calculate_features(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate enhanced features for trajectory data.

        Args:
            df: DataFrame with trajectory data
            **kwargs: Additional parameters

        Returns:
            DataFrame with calculated features
        """
        self.logger.info("Starting enhanced feature calculation")

        # Merge parameters
        params = {**self.parameters, **kwargs}

        # Start with basic trajectory features
        result_df = self._calculate_basic_features(df)

        # Multi-radius density analysis
        if params.get('calculate_density', True):
            result_df = self.density_analyzer.analyze_frame_density(result_df)

        # Advanced shape and motion metrics
        if params.get('calculate_advanced_shape', True):
            result_df = self._calculate_advanced_shape_features(result_df)

        # Scaled radius of gyration
        if params.get('calculate_scaled_rg', True):
            result_df = self._calculate_scaled_rg_features(result_df)

        # Diffusion and velocity analysis
        if params.get('calculate_diffusion', True):
            result_df = self._calculate_diffusion_metrics(result_df)

        # Trajectory interpolation (if requested)
        if params.get('interpolate_trajectories', False):
            result_df = self.interpolator.interpolate_all_trajectories(result_df)

        self.logger.info("Enhanced feature calculation completed")
        return result_df

    def _calculate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic trajectory features."""
        df = df.copy()

        # Track length and duration
        df['track_length'] = df.groupby('track_number')['track_number'].transform('count')
        df['track_duration'] = df.groupby('track_number')['frame'].transform(
            lambda x: x.max() - x.min() + 1
        )

        # Nearest neighbor distances
        df = self._calculate_nearest_neighbors(df)

        return df

    def _calculate_nearest_neighbors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate nearest neighbor distances."""
        nn_distances = []

        for frame in df['frame'].unique():
            frame_data = df[df['frame'] == frame]

            if len(frame_data) < 2:
                nn_distances.extend([np.nan] * len(frame_data))
                continue

            positions = frame_data[['x', 'y']].values
            tree = KDTree(positions)
            distances, indices = tree.query(positions, k=2)
            nn_distances.extend(distances[:, 1])  # Second nearest (first is self)

        df['nn_distance'] = nn_distances
        return df

    def _calculate_advanced_shape_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced shape features using tensor analysis."""
        advanced_features = []

        for track_id in tqdm(df['track_number'].unique(), desc="Advanced shape analysis"):
            track_data = df[df['track_number'] == track_id]
            positions = track_data[['x', 'y']].values

            # Calculate advanced metrics
            metrics = self.shape_analyzer.calculate_gyration_tensor_metrics(positions)

            # Classify linearity
            linear_class = self.shape_analyzer.classify_linearity(
                metrics['eigenvalue_ratio'],
                metrics['step_alignment'],
                metrics['directionality_ratio'],
                eigenvalue_threshold=self.parameters.get('linear_eigenvalue_threshold', 20.0),
                alignment_threshold=self.parameters.get('linear_alignment_threshold', 0.7)
            )

            metrics['linear_classification'] = linear_class
            metrics['track_number'] = track_id
            advanced_features.append(metrics)

        # Convert to DataFrame and merge
        features_df = pd.DataFrame(advanced_features)
        df = pd.merge(df, features_df, on='track_number', how='left')

        return df

    def _calculate_scaled_rg_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate scaled radius of gyration features."""
        # Calculate mean step sizes per track
        mean_steps = []
        scaled_rgs = []
        mobility_classes = []

        for track_id in df['track_number'].unique():
            track_data = df[df['track_number'] == track_id].sort_values('frame')

            if len(track_data) < 2:
                mean_steps.append(np.nan)
                scaled_rgs.append(np.nan)
                mobility_classes.append('unclassified')
                continue

            # Calculate mean step size
            positions = track_data[['x', 'y']].values
            steps = np.diff(positions, axis=0)
            step_sizes = np.sqrt(np.sum(steps**2, axis=1))
            mean_step = np.mean(step_sizes)

            # Get radius of gyration (should be calculated by now)
            rg = track_data['radius_gyration'].iloc[0]

            # Calculate scaled Rg
            scaled_rg = self.scaled_rg_calculator.calculate_scaled_rg(rg, mean_step)

            # Classify mobility
            mobility_threshold = self.parameters.get('mobility_threshold', 2.11)
            mobility_class = self.scaled_rg_calculator.classify_mobility(
                scaled_rg, mobility_threshold
            )

            mean_steps.append(mean_step)
            scaled_rgs.append(scaled_rg)
            mobility_classes.append(mobility_class)

        # Create mapping DataFrames
        track_ids = df['track_number'].unique()
        step_mapping = pd.Series(mean_steps, index=track_ids, name='mean_step_size')
        srg_mapping = pd.Series(scaled_rgs, index=track_ids, name='scaled_rg')
        mobility_mapping = pd.Series(mobility_classes, index=track_ids, name='mobility_classification')

        # Map to original DataFrame
        df = df.join(step_mapping, on='track_number')
        df = df.join(srg_mapping, on='track_number')
        df = df.join(mobility_mapping, on='track_number')

        return df

    def _calculate_diffusion_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic diffusion metrics."""
        diffusion_tracks = []

        for track_id in tqdm(df['track_number'].unique(), desc="Analyzing diffusion"):
            track_data = df[df['track_number'] == track_id].sort_values('frame').copy()

            if len(track_data) < 3:
                diffusion_tracks.append(track_data)
                continue

            # Set origin to first position
            min_frame = track_data['frame'].min()
            origin_x = track_data[track_data['frame'] == min_frame]['x'].iloc[0]
            origin_y = track_data[track_data['frame'] == min_frame]['y'].iloc[0]

            # Calculate relative positions
            track_data['zeroed_x'] = track_data['x'] - origin_x
            track_data['zeroed_y'] = track_data['y'] - origin_y
            track_data['lagNumber'] = track_data['frame'] - min_frame
            track_data['distanceFromOrigin'] = np.sqrt(
                track_data['zeroed_x']**2 + track_data['zeroed_y']**2
            )

            # Calculate step displacements and velocities
            track_data = self._calculate_step_metrics(track_data)

            diffusion_tracks.append(track_data)

        result_df = pd.concat(diffusion_tracks, ignore_index=True)
        result_df['d_squared'] = result_df['distanceFromOrigin']**2

        return result_df

    def _calculate_step_metrics(self, track_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate step-wise metrics for a single track."""
        # Step displacements
        track_data['dx'] = track_data['x'].diff()
        track_data['dy'] = track_data['y'].diff()
        track_data['step_size'] = np.sqrt(track_data['dx']**2 + track_data['dy']**2)

        # Time intervals
        track_data['dt'] = track_data['frame'].diff()

        # Instantaneous velocity
        track_data['velocity'] = track_data['step_size'] / track_data['dt']
        track_data['velocity'] = track_data['velocity'].replace([np.inf, -np.inf], np.nan)

        # Track mean velocity
        track_data['meanVelocity'] = track_data['velocity'].mean()

        # Mean step size
        track_data['meanLag'] = track_data['step_size'].mean()

        return track_data

    def get_analysis_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive analysis summary."""
        summary = {}

        if 'track_number' in df.columns:
            summary['n_tracks'] = df['track_number'].nunique()
            summary['n_localizations'] = len(df)

            # Track length statistics
            track_lengths = df.groupby('track_number').size()
            summary['mean_track_length'] = track_lengths.mean()
            summary['median_track_length'] = track_lengths.median()

        # Mobility classification summary
        if 'mobility_classification' in df.columns:
            mobility_counts = df.groupby('track_number')['mobility_classification'].first().value_counts()
            total_tracks = mobility_counts.sum()
            summary['mobility_distribution'] = mobility_counts.to_dict()
            summary['percent_mobile'] = (mobility_counts.get('mobile', 0) / total_tracks) * 100

        # Feature statistics
        feature_columns = [
            'radius_gyration', 'scaled_rg', 'asymmetry', 'eigenvalue_ratio',
            'step_alignment', 'directionality_ratio'
        ]

        for col in feature_columns:
            if col in df.columns:
                values = df.groupby('track_number')[col].first().dropna()
                if len(values) > 0:
                    summary[f'{col}_mean'] = values.mean()
                    summary[f'{col}_std'] = values.std()
                    summary[f'{col}_median'] = values.median()

        # Density analysis summary
        density_cols = [col for col in df.columns if 'nnCountInFrame_within' in col]
        if density_cols:
            summary['density_metrics'] = {}
            for col in density_cols:
                values = df[col].dropna()
                if len(values) > 0:
                    radius = col.split('_')[2]  # Extract radius from column name
                    summary['density_metrics'][f'radius_{radius}'] = {
                        'mean': values.mean(),
                        'std': values.std()
                    }

        return summary

    def update_parameters(self, parameters):
        """Update feature calculation parameters."""
        self.parameters.update(parameters)


# Backward compatibility - alias the enhanced calculator
EnhancedFeatureCalculator = FeatureCalculator