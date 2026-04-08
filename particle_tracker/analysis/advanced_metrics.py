#!/usr/bin/env python3
"""
Advanced Metrics Module
========================

Comprehensive trajectory analysis including all sophisticated metrics from the original scripts:
- Scaled radius of gyration (sRg) using Golan & Sherman method
- Advanced shape metrics (eigenvalue ratios, linearity classification)
- Multi-radius density analysis (3,5,10,20,30 pixel neighbors) 
- Direction autocorrelation analysis
- Localization precision metrics
- Advanced diffusion analysis
- Trajectory interpolation for trapped particles

This module consolidates the most sophisticated analysis methods from the original scripts
into a unified, well-structured module for the particle tracking application.
"""

import logging
import math
from typing import Optional, Dict, List, Any, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from scipy import stats, spatial
from sklearn.neighbors import KDTree
from tqdm import tqdm


class AdvancedMetricsCalculator:
    """Calculate advanced trajectory metrics using methods from original analysis scripts."""
    
    def __init__(self, parameters=None):
        self.logger = logging.getLogger(__name__)
        self.parameters = parameters or {}
        
    def calculate_all_advanced_metrics(self, trajectory_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all advanced metrics for trajectory data.
        
        Args:
            trajectory_data: DataFrame with trajectory data
            
        Returns:
            DataFrame with advanced metrics added
        """
        df = trajectory_data.copy()
        
        # Calculate scaled radius of gyration (sRg) - Gold standard for mobility
        df = self._calculate_scaled_radius_of_gyration(df)
        
        # Calculate advanced shape metrics with eigenvalue analysis
        df = self._calculate_advanced_shape_metrics(df)
        
        # Calculate multi-radius density analysis
        df = self._calculate_multi_radius_density(df)
        
        # Calculate localization precision metrics
        df = self._calculate_localization_precision(df)
        
        # Calculate advanced diffusion metrics
        df = self._calculate_advanced_diffusion_metrics(df)
        
        # Calculate trajectory quality metrics
        df = self._calculate_trajectory_quality_metrics(df)
        
        return df
    
    def _calculate_scaled_radius_of_gyration(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate scaled radius of gyration using Golan & Sherman method.
        
        This is the gold standard for mobility classification:
        sRg = √(π/2) * Rg / mean_step_size
        """
        def calculate_srg_for_track(track_data):
            track_data = track_data.sort_values('frame')
            positions = track_data[['x', 'y']].values
            
            if len(positions) < 2:
                return pd.Series({
                    'sRg': np.nan,
                    'mean_step_size': np.nan,
                    'mobility_sRg': 'immobile'
                })
            
            # Calculate basic radius of gyration
            center = np.mean(positions, axis=0)
            normed_points = positions - center[None, :]
            rg = np.sqrt(np.mean(np.sum(normed_points**2, axis=1)))
            
            # Calculate mean step size
            steps = np.diff(positions, axis=0)
            step_sizes = np.sqrt(np.sum(steps**2, axis=1))
            mean_step_size = np.mean(step_sizes)
            
            # Calculate scaled Rg
            if mean_step_size > 0:
                sRg = np.sqrt(np.pi/2) * rg / mean_step_size
            else:
                sRg = np.nan
            
            # Classify mobility using Golan & Sherman threshold
            mobility_threshold = self.parameters.get('sRg_threshold', 2.11)
            mobility_sRg = 'mobile' if sRg > mobility_threshold else 'immobile'
            
            return pd.Series({
                'sRg': sRg,
                'mean_step_size': mean_step_size,
                'mobility_sRg': mobility_sRg
            })
        
        # Calculate for each track
        srg_results = df.groupby('track_number').apply(calculate_srg_for_track)
        
        # Join back to original dataframe
        for col in srg_results.columns:
            df = df.join(srg_results[col], on='track_number', rsuffix='_temp')
        
        return df
    
    def _calculate_advanced_shape_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced shape metrics using tensor-based methods."""
        
        def calculate_advanced_shape_for_track(track_data):
            track_data = track_data.sort_values('frame')
            positions = track_data[['x', 'y']].values
            
            if len(positions) < 3:
                return pd.Series({
                    'eigenvalue_ratio': np.nan,
                    'step_alignment': np.nan,
                    'directionality_ratio': np.nan,
                    'linearity_classification': 'unclassified',
                    'fractal_dimension': np.nan,
                    'net_displacement': np.nan,
                    'efficiency': np.nan,
                    'straightness': np.nan
                })
            
            # Calculate center and normalize
            center = np.mean(positions, axis=0)
            normed_points = positions - center[None, :]
            
            # Calculate gyration tensor
            gyration_tensor = np.einsum('im,in->mn', normed_points, normed_points) / len(positions)
            eig_values, eig_vectors = np.linalg.eig(gyration_tensor)
            eig_values = np.real(eig_values)
            
            # Sort eigenvalues
            idx = eig_values.argsort()[::-1]
            eig_values = eig_values[idx]
            eig_vectors = np.real(eig_vectors[:, idx])
            
            # Calculate eigenvalue ratio (key indicator for linear motion)
            eigenvalue_ratio = eig_values[0] / eig_values[1] if eig_values[1] > 0 else np.inf
            
            # Calculate step alignment with principal axis
            steps = np.diff(positions, axis=0)
            step_norms = np.linalg.norm(steps, axis=1)
            valid_steps = step_norms > 0
            
            if np.any(valid_steps):
                normalized_steps = steps[valid_steps] / step_norms[valid_steps, None]
                principal_eigenvector = eig_vectors[:, 0]
                cos_angles = np.abs(np.dot(normalized_steps, principal_eigenvector))
                step_alignment = np.mean(cos_angles)
            else:
                step_alignment = np.nan
            
            # Calculate directionality ratio
            net_displacement = np.linalg.norm(positions[-1] - positions[0])
            path_length = np.sum(step_norms)
            directionality_ratio = net_displacement / path_length if path_length > 0 else np.nan
            
            # Classify linearity
            linearity_classification = self._classify_linearity(
                eigenvalue_ratio, step_alignment, directionality_ratio
            )
            
            # Calculate fractal dimension (from original scripts)
            fractal_dimension = self._calculate_fractal_dimension(positions)
            
            # Calculate net displacement and efficiency
            efficiency = net_displacement**2 / ((len(positions)-1) * np.sum(step_norms**2)) if len(positions) > 1 else np.nan
            
            # Calculate straightness (mean cosine of turning angles)
            straightness = self._calculate_straightness(positions)
            
            return pd.Series({
                'eigenvalue_ratio': eigenvalue_ratio,
                'step_alignment': step_alignment,
                'directionality_ratio': directionality_ratio,
                'linearity_classification': linearity_classification,
                'fractal_dimension': fractal_dimension,
                'net_displacement': net_displacement,
                'efficiency': efficiency,
                'straightness': straightness
            })
        
        # Calculate for each track
        shape_results = df.groupby('track_number').apply(calculate_advanced_shape_for_track)
        
        # Join back to original dataframe
        for col in shape_results.columns:
            df = df.join(shape_results[col], on='track_number', rsuffix='_temp')
        
        return df
    
    def _calculate_multi_radius_density(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate nearest neighbor counts within multiple radii (3,5,10,20,30 pixels)."""
        
        # Default radii from original scripts
        radii = self.parameters.get('density_radii', [3, 5, 10, 20, 30])
        
        # Sort by frame for efficient processing
        df = df.sort_values(['frame'])
        
        # Initialize columns for each radius
        for radius in radii:
            df[f'nn_count_r{radius}'] = np.nan
        
        # Process each frame
        frames = df['frame'].unique()
        
        for frame in frames:
            frame_data = df[df['frame'] == frame]
            
            if len(frame_data) < 2:
                continue
                
            positions = frame_data[['x', 'y']].values
            tree = KDTree(positions)
            
            # For each radius, count neighbors
            for radius in radii:
                counts = tree.query_radius(positions, r=radius, count_only=True)
                # Subtract 1 to exclude self
                counts = counts - 1
                
                # Update dataframe
                df.loc[df['frame'] == frame, f'nn_count_r{radius}'] = counts
        
        return df
    
    def _calculate_localization_precision(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate localization precision metrics for trapped particles."""
        
        def calculate_precision_for_track(track_data):
            positions = track_data[['x', 'y']].values
            
            if len(positions) < 3:
                return pd.Series({
                    'localization_precision_x': np.nan,
                    'localization_precision_y': np.nan,
                    'localization_precision_total': np.nan,
                    'mean_distance_from_center': np.nan
                })
            
            # Calculate mean position
            mean_x = np.mean(positions[:, 0])
            mean_y = np.mean(positions[:, 1])
            
            # Calculate standard deviations (localization precision)
            precision_x = np.std(positions[:, 0])
            precision_y = np.std(positions[:, 1])
            precision_total = np.sqrt(precision_x**2 + precision_y**2)
            
            # Calculate mean distance from center
            distances = np.sqrt((positions[:, 0] - mean_x)**2 + (positions[:, 1] - mean_y)**2)
            mean_distance = np.mean(distances)
            
            return pd.Series({
                'localization_precision_x': precision_x,
                'localization_precision_y': precision_y,
                'localization_precision_total': precision_total,
                'mean_distance_from_center': mean_distance
            })
        
        # Calculate for each track
        precision_results = df.groupby('track_number').apply(calculate_precision_for_track)
        
        # Join back to original dataframe
        for col in precision_results.columns:
            df = df.join(precision_results[col], on='track_number', rsuffix='_temp')
        
        return df
    
    def _calculate_advanced_diffusion_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced diffusion metrics including MSD analysis."""
        
        def calculate_diffusion_for_track(track_data):
            track_data = track_data.sort_values('frame')
            positions = track_data[['x', 'y']].values
            frames = track_data['frame'].values
            
            if len(positions) < 3:
                return pd.Series({
                    'msd_slope': np.nan,
                    'msd_intercept': np.nan,
                    'diffusion_coefficient_advanced': np.nan,
                    'alpha_exponent': np.nan,
                    'confinement_strength': np.nan
                })
            
            # Calculate MSD for different lag times
            max_lag = min(len(positions) - 1, 10)
            msd_values = []
            lag_times = []
            
            for lag in range(1, max_lag + 1):
                if lag >= len(positions):
                    break
                    
                displacements = positions[lag:] - positions[:-lag]
                squared_displacements = np.sum(displacements**2, axis=1)
                msd = np.mean(squared_displacements)
                
                msd_values.append(msd)
                lag_times.append(lag)
            
            if len(msd_values) < 3:
                return pd.Series({
                    'msd_slope': np.nan,
                    'msd_intercept': np.nan,
                    'diffusion_coefficient_advanced': np.nan,
                    'alpha_exponent': np.nan,
                    'confinement_strength': np.nan
                })
            
            # Fit linear regression to get diffusion coefficient
            log_lag = np.log(lag_times)
            log_msd = np.log(msd_values)
            
            # Fit to first few points for better linear approximation
            fit_points = min(5, len(log_lag))
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                log_lag[:fit_points], log_msd[:fit_points]
            )
            
            # Calculate diffusion coefficient (D = MSD / (4 * dt) for 2D)
            D = np.exp(intercept) / 4.0
            
            # Alpha exponent indicates type of diffusion
            alpha_exponent = slope
            
            # Calculate confinement strength for subdiffusive motion
            if alpha_exponent < 1.0:
                # Simple measure: ratio of actual MSD to expected free diffusion
                expected_msd = 4 * D * lag_times[0]  # For lag=1
                actual_msd = msd_values[0]
                confinement_strength = 1 - (actual_msd / expected_msd) if expected_msd > 0 else np.nan
            else:
                confinement_strength = 0.0
            
            return pd.Series({
                'msd_slope': slope,
                'msd_intercept': intercept,
                'diffusion_coefficient_advanced': D,
                'alpha_exponent': alpha_exponent,
                'confinement_strength': confinement_strength
            })
        
        # Calculate for each track
        diffusion_results = df.groupby('track_number').apply(calculate_diffusion_for_track)
        
        # Join back to original dataframe
        for col in diffusion_results.columns:
            df = df.join(diffusion_results[col], on='track_number', rsuffix='_temp')
        
        return df
    
    def _calculate_trajectory_quality_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trajectory quality and completeness metrics."""
        
        def calculate_quality_for_track(track_data):
            track_data = track_data.sort_values('frame')
            frames = track_data['frame'].values
            
            # Calculate frame completeness
            min_frame = frames.min()
            max_frame = frames.max()
            expected_frames = max_frame - min_frame + 1
            actual_frames = len(frames)
            frame_completeness = actual_frames / expected_frames
            
            # Calculate frame gaps
            frame_gaps = np.diff(frames) - 1  # Gaps (missing frames)
            max_gap = np.max(frame_gaps) if len(frame_gaps) > 0 else 0
            total_gaps = np.sum(frame_gaps)
            
            # Calculate trajectory smoothness (variance in step sizes)
            if len(track_data) >= 3:
                positions = track_data[['x', 'y']].values
                steps = np.diff(positions, axis=0)
                step_sizes = np.sqrt(np.sum(steps**2, axis=1))
                step_size_cv = np.std(step_sizes) / np.mean(step_sizes) if np.mean(step_sizes) > 0 else np.nan
            else:
                step_size_cv = np.nan
            
            return pd.Series({
                'frame_completeness': frame_completeness,
                'max_frame_gap': max_gap,
                'total_frame_gaps': total_gaps,
                'step_size_cv': step_size_cv
            })
        
        # Calculate for each track
        quality_results = df.groupby('track_number').apply(calculate_quality_for_track)
        
        # Join back to original dataframe
        for col in quality_results.columns:
            df = df.join(quality_results[col], on='track_number', rsuffix='_temp')
        
        return df
    
    def calculate_direction_autocorrelation(self, trajectory_data: pd.DataFrame, 
                                         time_interval: float = 1.0, 
                                         num_intervals: int = 25) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate direction autocorrelation for trajectories.
        
        Args:
            trajectory_data: DataFrame with trajectory data
            time_interval: Time interval between frames
            num_intervals: Number of time intervals to analyze
            
        Returns:
            Tuple of (combined_results, individual_track_results)
        """
        # Calculate normalized vectors for each step
        df_with_vectors = self._calculate_normalized_vectors(trajectory_data)
        
        # Find trajectory boundaries
        traj_starts = self._find_trajectory_starts(df_with_vectors)
        
        # Calculate scalar products for different time intervals
        combined_df, tracks_df = self._calculate_scalar_products(
            df_with_vectors, traj_starts, time_interval, num_intervals
        )
        
        return combined_df, tracks_df
    
    def interpolate_trapped_trajectories(self, trajectory_data: pd.DataFrame, 
                                       classification_column: str = 'mobility_sRg') -> pd.DataFrame:
        """Interpolate missing points for trapped/immobile trajectories.
        
        This fills in missing timepoints for trapped particles to enable
        complete temporal analysis as in Step_10 from original scripts.
        """
        df = trajectory_data.copy()
        
        # Filter for trapped/immobile tracks
        if classification_column in df.columns:
            trapped_tracks = df[df[classification_column] == 'immobile']['track_number'].unique()
        else:
            # Fallback: use sRg threshold
            srg_threshold = self.parameters.get('sRg_threshold', 2.11)
            if 'sRg' in df.columns:
                trapped_tracks = df[df['sRg'] < srg_threshold]['track_number'].unique()
            else:
                self.logger.warning("No mobility classification available for interpolation")
                return df
        
        interpolated_data = []
        
        for track_id in trapped_tracks:
            track_data = df[df['track_number'] == track_id].sort_values('frame')
            
            if len(track_data) < 2:
                continue
            
            # Get frame range
            min_frame = track_data['frame'].min()
            max_frame = track_data['frame'].max()
            all_frames = np.arange(min_frame, max_frame + 1)
            
            # Interpolate X and Y coordinates
            x_interp = np.interp(all_frames, track_data['frame'], track_data['x'])
            y_interp = np.interp(all_frames, track_data['frame'], track_data['y'])
            
            # Create interpolated dataframe
            interp_df = pd.DataFrame({
                'frame': all_frames,
                'track_number': track_id,
                'x': x_interp,
                'y': y_interp,
                'interpolated': True
            })
            
            # Mark original points
            interp_df.loc[interp_df['frame'].isin(track_data['frame']), 'interpolated'] = False
            
            # Copy other columns from original data
            for col in track_data.columns:
                if col not in ['frame', 'track_number', 'x', 'y']:
                    # Use forward fill for categorical data, interpolation for numerical
                    if track_data[col].dtype in ['object', 'category']:
                        interp_df[col] = track_data[col].iloc[0]  # Use first value
                    else:
                        # Interpolate numerical columns
                        try:
                            interp_df[col] = np.interp(all_frames, track_data['frame'], track_data[col])
                        except Exception:
                            interp_df[col] = track_data[col].iloc[0]  # Fallback
            
            interpolated_data.append(interp_df)
        
        # Combine with non-trapped tracks
        if interpolated_data:
            non_trapped_df = df[~df['track_number'].isin(trapped_tracks)].copy()
            non_trapped_df['interpolated'] = False
            
            interpolated_df = pd.concat(interpolated_data + [non_trapped_df], ignore_index=True)
            return interpolated_df.sort_values(['track_number', 'frame'])
        else:
            df['interpolated'] = False
            return df
    
    # Helper methods
    def _classify_linearity(self, eigenvalue_ratio: float, step_alignment: float, 
                          directionality_ratio: float) -> str:
        """Classify trajectory linearity based on multiple metrics."""
        
        # Thresholds from parameters or defaults
        eig_ratio_threshold = self.parameters.get('linear_eigenvalue_ratio_cutoff', 20.0)
        step_align_threshold = self.parameters.get('linear_step_alignment_cutoff', 0.7)
        directionality_threshold = self.parameters.get('linear_directionality_cutoff', 0.7)
        
        if np.isnan(eigenvalue_ratio) or np.isnan(step_alignment):
            return 'unclassified'
        
        # Check for linear motion
        is_linear = (eigenvalue_ratio >= eig_ratio_threshold and 
                    step_alignment >= step_align_threshold)
        
        if not is_linear:
            return 'non_linear'
        
        # Distinguish between unidirectional and bidirectional
        if np.isnan(directionality_ratio):
            return 'linear'
        
        if directionality_ratio >= directionality_threshold:
            return 'linear_unidirectional'
        else:
            return 'linear_bidirectional'
    
    def _calculate_fractal_dimension(self, positions: np.ndarray) -> float:
        """Calculate fractal dimension as in original scripts."""
        try:
            # Check if points are collinear
            if len(positions) < 3:
                return np.nan
            
            x0, y0 = positions[0]
            points = [(x, y) for x, y in positions if (x != x0) or (y != y0)]
            
            if len(points) < 2:
                return np.nan
            
            slopes = [((y - y0) / (x - x0)) if (x != x0) else None for x, y in points]
            if all(s == slopes[0] for s in slopes):
                return np.nan  # Collinear points
            
            # Calculate total path length
            total_path_length = np.sum(np.sqrt(np.sum((positions[1:] - positions[:-1])**2, axis=1)))
            
            # Find convex hull and largest distance
            if len(positions) >= 3:
                hull = spatial.ConvexHull(positions)
                candidates = positions[hull.vertices]
                dist_mat = spatial.distance_matrix(candidates, candidates)
                largest_distance = np.max(dist_mat)
            else:
                largest_distance = np.linalg.norm(positions[-1] - positions[0])
            
            # Calculate fractal dimension
            step_count = len(positions)
            if largest_distance > 0 and total_path_length > 0:
                fractal_dim = math.log(step_count) / math.log(step_count * largest_distance / total_path_length)
                return fractal_dim
            else:
                return np.nan
                
        except Exception:
            return np.nan
    
    def _calculate_straightness(self, positions: np.ndarray) -> float:
        """Calculate straightness (mean cosine of turning angles)."""
        try:
            if len(positions) < 3:
                return np.nan
            
            steps = np.diff(positions, axis=0)
            
            # Calculate cosine of angles between consecutive steps
            cos_angles = []
            for i in range(len(steps) - 1):
                v1, v2 = steps[i], steps[i+1]
                norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
                
                if norm1 > 0 and norm2 > 0:
                    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                    cos_angles.append(np.clip(cos_angle, -1, 1))
            
            return np.mean(cos_angles) if cos_angles else np.nan
            
        except Exception:
            return np.nan
    
    def _calculate_normalized_vectors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate normalized direction vectors for autocorrelation analysis."""
        result_df = df.copy()
        result_df['x_vector'] = np.nan
        result_df['y_vector'] = np.nan
        
        # Group by track and calculate vectors
        for track_id in df['track_number'].unique():
            track_mask = df['track_number'] == track_id
            track_data = df[track_mask].sort_values('frame')
            
            if len(track_data) < 2:
                continue
            
            # Calculate step vectors
            x_diff = -track_data['x'].diff(-1).iloc[:-1]  # Negative for original script compatibility
            y_diff = -track_data['y'].diff(-1).iloc[:-1]
            
            # Calculate magnitudes
            magnitudes = np.sqrt(x_diff**2 + y_diff**2)
            
            # Normalize where magnitude > 0
            valid_moves = magnitudes > 0
            normalized_x = np.where(valid_moves, x_diff / magnitudes, np.nan)
            normalized_y = np.where(valid_moves, y_diff / magnitudes, np.nan)
            
            # Update result dataframe
            track_indices = track_data.index[1:]  # Skip first point
            result_df.loc[track_indices, 'x_vector'] = normalized_x.values
            result_df.loc[track_indices, 'y_vector'] = normalized_y.values
        
        return result_df
    
    def _find_trajectory_starts(self, df: pd.DataFrame) -> List[int]:
        """Find the start indices of each trajectory."""
        # Sort by track number and frame
        df_sorted = df.sort_values(['track_number', 'frame'])
        
        # Find where track number changes
        track_changes = df_sorted['track_number'].diff() != 0
        track_changes.iloc[0] = True  # First row is always a start
        
        # Get indices of trajectory starts
        start_indices = df_sorted[track_changes].index.tolist()
        start_indices.append(len(df_sorted))  # Add end boundary
        
        return start_indices
    
    def _calculate_scalar_products(self, df: pd.DataFrame, traj_starts: List[int],
                                 time_interval: float, num_intervals: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate scalar products for autocorrelation analysis."""
        
        # Initialize results
        combined_results = {time_interval * step: [] for step in range(1, num_intervals + 1)}
        individual_tracks = []
        
        # Process each trajectory
        for i in range(len(traj_starts) - 1):
            start_idx = traj_starts[i]
            end_idx = traj_starts[i + 1]
            
            track_data = df.iloc[start_idx:end_idx]
            track_id = f"track_{i+1}"
            
            # Skip short trajectories
            if len(track_data) < 2:
                continue
            
            # Get vectors
            x_vectors = track_data['x_vector'].values
            y_vectors = track_data['y_vector'].values
            
            # Calculate scalar products for different time intervals
            for step in range(1, min(num_intervals, len(track_data))):
                time_point = time_interval * step
                
                # Get vector pairs
                x_vecs1 = x_vectors[:-step]
                y_vecs1 = y_vectors[:-step]
                x_vecs2 = x_vectors[step:]
                y_vecs2 = y_vectors[step:]
                
                # Calculate dot products
                dot_products = x_vecs1 * x_vecs2 + y_vecs1 * y_vecs2
                
                # Filter valid values
                valid_mask = ~np.isnan(dot_products)
                valid_dots = dot_products[valid_mask]
                
                if len(valid_dots) > 0:
                    # Add to combined results
                    combined_results[time_point].extend(valid_dots.tolist())
                    
                    # Store individual track result
                    individual_tracks.append({
                        'track_id': track_id,
                        'time_interval': time_point,
                        'correlation': np.mean(valid_dots)
                    })
        
        # Convert to DataFrames
        combined_df = pd.DataFrame({k: pd.Series(v) for k, v in combined_results.items()})
        tracks_df = pd.DataFrame(individual_tracks)
        
        return combined_df, tracks_df
    
    def update_parameters(self, parameters: Dict[str, Any]):
        """Update calculation parameters."""
        self.parameters.update(parameters)


# Convenience functions for specific metric calculations
def calculate_scaled_rg(rg: float, mean_step_size: float) -> float:
    """Calculate scaled radius of gyration using Golan & Sherman method."""
    if np.isnan(rg) or np.isnan(mean_step_size) or mean_step_size == 0:
        return np.nan
    return np.sqrt(np.pi/2) * rg / mean_step_size


def classify_mobility_by_srg(srg: float, threshold: float = 2.11) -> str:
    """Classify mobility using scaled radius of gyration threshold."""
    if np.isnan(srg):
        return 'unclassified'
    return 'mobile' if srg > threshold else 'immobile'


def calculate_multi_radius_neighbors(positions: np.ndarray, radii: List[int] = [3, 5, 10, 20, 30]) -> Dict[int, int]:
    """Calculate neighbor counts within multiple radii for a single frame."""
    if len(positions) < 2:
        return {r: 0 for r in radii}
    
    tree = KDTree(positions)
    neighbor_counts = {}
    
    for radius in radii:
        counts = tree.query_radius(positions, r=radius, count_only=True)
        # Subtract 1 to exclude self, take mean
        neighbor_counts[radius] = np.mean(counts - 1)
    
    return neighbor_counts
