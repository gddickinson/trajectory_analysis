#!/usr/bin/env python3
"""
Trajectory Interpolation Module
===============================

Provides interpolation capabilities for particle trajectories, particularly useful
for analyzing trapped particles (SVM=3) by filling in missing timepoints and 
calculating comprehensive intensity metrics.

This module is based on the functionality from Step_10_addInterpolatedPointstoBinnedRecording.py
but adapted for the particle tracking application architecture.
"""

import logging
from typing import Optional, Dict, List, Any, Tuple, Union
import numpy as np
import pandas as pd
import skimage.io as skio
from sklearn.neighbors import KDTree
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod


class GroupingMethod(Enum):
    """Methods for grouping tracks during interpolation."""
    NONE = "none"
    IN_PIXEL = "in_pixel"
    HCLUSTER = "hcluster"


class InterpolationMode(Enum):
    """Different modes of interpolation."""
    BINNED_TO_UNBINNED = "binned_to_unbinned"
    TRAPPED_ALL_FRAMES = "trapped_all_frames"
    MISSING_POINTS_ONLY = "missing_points_only"


@dataclass
class InterpolationParameters:
    """Parameters for trajectory interpolation."""
    mode: InterpolationMode = InterpolationMode.TRAPPED_ALL_FRAMES
    grouping_method: GroupingMethod = GroupingMethod.HCLUSTER
    tracks_to_keep: str = "all"  # "all" or "svm3_only"
    bin_size: int = 10
    smoothing_window_fraction: float = 0.1  # Fraction of total frames for smoothing
    clustering_threshold: float = 3.0  # Distance threshold for hierarchical clustering
    nn_radii: List[int] = None  # Radii for nearest neighbor counting

    def __post_init__(self):
        if self.nn_radii is None:
            self.nn_radii = [3, 5, 10, 20, 30]


class TrajectoryInterpolator:
    """Main class for trajectory interpolation analysis."""

    def __init__(self, parameters: Optional[InterpolationParameters] = None):
        self.logger = logging.getLogger(__name__)
        self.parameters = parameters or InterpolationParameters()

    def interpolate_trajectories(self, df: pd.DataFrame, 
                               image_data: Optional[np.ndarray] = None,
                               roi_data: Optional[np.ndarray] = None,
                               camera_background: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Main method to interpolate trajectories based on the specified mode.
        
        Args:
            df: DataFrame with trajectory data
            image_data: Optional image stack for intensity extraction
            roi_data: Optional ROI background data
            camera_background: Optional camera background estimates
            
        Returns:
            DataFrame with interpolated trajectories
        """
        self.logger.info(f"Starting trajectory interpolation in {self.parameters.mode.value} mode")
        
        if self.parameters.mode == InterpolationMode.BINNED_TO_UNBINNED:
            return self._interpolate_binned_to_unbinned(df, image_data, roi_data, camera_background)
        elif self.parameters.mode == InterpolationMode.TRAPPED_ALL_FRAMES:
            return self._interpolate_trapped_all_frames(df, image_data, roi_data, camera_background)
        elif self.parameters.mode == InterpolationMode.MISSING_POINTS_ONLY:
            return self._interpolate_missing_points(df, image_data, roi_data, camera_background)
        else:
            raise ValueError(f"Unknown interpolation mode: {self.parameters.mode}")

    def _interpolate_binned_to_unbinned(self, df: pd.DataFrame,
                                      image_data: np.ndarray,
                                      roi_data: Optional[np.ndarray] = None,
                                      camera_background: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Interpolate binned trajectory data to unbinned time series.
        
        This method takes trajectories from binned data and interpolates them
        to the original unbinned timeframe.
        """
        self.logger.info("Interpolating binned tracks to unbinned recording")
        
        # Remove unlinked points
        df = df[~df['track_number'].isna()].copy()
        unlinked = df[df['track_number'].isna()].copy()
        
        # Get unique track numbers
        track_list = df['track_number'].unique()
        
        # Scale frame numbers to unbinned recording
        df['frame_unbinned'] = df['frame'] * self.parameters.bin_size
        
        new_df = pd.DataFrame()
        
        for track_number in track_list:
            self.logger.debug(f"Processing track {track_number}")
            track_df = df[df['track_number'] == track_number].copy()
            
            # Extract trajectory points
            points = np.column_stack((
                track_df['frame_unbinned'].values,
                track_df['x'].values,
                track_df['y'].values
            ))
            
            # Interpolate missing frames
            all_frames = range(int(points[:, 0].min()), int(points[:, 0].max()) + 1)
            x_interp = np.interp(all_frames, points[:, 0], points[:, 1])
            y_interp = np.interp(all_frames, points[:, 0], points[:, 2])
            
            # Create interpolated trajectory
            interp_points = np.column_stack((all_frames, x_interp, y_interp))
            
            # Extract intensities if image data provided
            if image_data is not None:
                intensities = self._get_intensities_from_image(image_data, interp_points)
            else:
                intensities = np.full(len(all_frames), np.nan)
            
            # Create temporary DataFrame for this track
            temp_df = pd.DataFrame({
                'frame': all_frames,
                'track_number': track_number,
                'x': x_interp,
                'y': y_interp,
                'intensity': intensities
            })
            
            # Add diffusion metrics
            temp_df = self._add_diffusion_metrics(temp_df)
            
            # Copy over properties from original data
            for prop in ['radius_gyration', 'asymmetry', 'skewness', 'kurtosis',
                        'fracDimension', 'netDispl', 'Straight', 'SVM']:
                if prop in track_df.columns:
                    temp_df[prop] = track_df[prop].iloc[0]
            
            temp_df['n_segments'] = len(all_frames)
            
            # Add lag displacement analysis
            temp_df = self._add_lag_displacement(temp_df)
            
            # Add background analysis if data provided
            if roi_data is not None:
                temp_df = self._add_background_analysis(temp_df, roi_data, camera_background)
            
            new_df = pd.concat([new_df, temp_df], ignore_index=True)
        
        # Add final calculations
        new_df = self._add_final_calculations(new_df)
        
        # Add unlinked points back if they exist
        if len(unlinked) > 0:
            # Ensure unlinked has same columns
            for col in new_df.columns:
                if col not in unlinked.columns:
                    unlinked[col] = np.nan
            new_df = pd.concat([new_df, unlinked], ignore_index=True)
        
        return new_df

    def _interpolate_trapped_all_frames(self, df: pd.DataFrame,
                                      image_data: np.ndarray,
                                      roi_data: Optional[np.ndarray] = None,
                                      camera_background: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Interpolate trapped particles (SVM=3) to fill all frames in the recording.
        
        This is the main method for analyzing trapped particles by extending
        their trajectories to cover the entire recording duration.
        """
        self.logger.info("Interpolating trapped particles to all frames")
        
        # Prepare output columns
        df = self._prepare_output_columns(df)
        
        # Remove unlinked points
        df = df[~df['track_number'].isna()].copy()
        unlinked = df[df['track_number'].isna()].copy()
        
        # Separate mobile and trapped particles
        df_mobile = df[df['SVM'] != 3].copy()
        df_trapped = df[df['SVM'] == 3].copy()
        
        if len(df_trapped) == 0:
            self.logger.warning("No trapped particles (SVM=3) found")
            return df
        
        # Get track IDs based on grouping method
        track_list = self._get_track_ids(df_trapped)
        
        if image_data is None:
            raise ValueError("Image data is required for trapped particle interpolation")
        
        # Get total recording length
        n_frames, height, width = image_data.shape
        
        new_df = pd.DataFrame()
        
        for track_number in track_list:
            self.logger.debug(f"Processing trapped track {track_number}")
            track_df = df_trapped[df_trapped['track_number'] == track_number].copy()
            
            # Extract trajectory points
            points = np.column_stack((
                track_df['frame'].values,
                track_df['x'].values,
                track_df['y'].values
            ))
            
            # Interpolate missing frames within track duration
            interp_frames = range(int(points[:, 0].min()), int(points[:, 0].max()) + 1)
            x_interp = np.interp(interp_frames, points[:, 0], points[:, 1])
            y_interp = np.interp(interp_frames, points[:, 0], points[:, 2])
            
            # Extend to full recording using mean position
            all_frames = np.arange(n_frames)
            all_x = np.full(n_frames, np.mean(x_interp))
            all_y = np.full(n_frames, np.mean(y_interp))
            
            # Fill in interpolated values where available
            frame_start = int(points[:, 0].min())
            frame_end = int(points[:, 0].max())
            all_x[frame_start:frame_end+1] = x_interp
            all_y[frame_start:frame_end+1] = y_interp
            
            # Extract intensities at trajectory positions
            traj_points = np.column_stack((all_frames, all_x, all_y))
            intensities = self._get_intensities_from_image(image_data, traj_points)
            
            # Extract intensities at fixed mean position
            mean_x, mean_y = np.mean(all_x), np.mean(all_y)
            mean_points = np.column_stack((all_frames, 
                                         np.full(n_frames, mean_x),
                                         np.full(n_frames, mean_y)))
            intensities_mean_xy = self._get_intensities_from_image(image_data, mean_points)
            
            # Create DataFrame for this track
            temp_df = pd.DataFrame({
                'frame': all_frames,
                'track_number': track_number,
                'x': all_x,
                'y': all_y,
                'intensity': intensities,
                'intensity_roiOnMeanXY': intensities_mean_xy
            })
            
            # Add diffusion metrics
            temp_df = self._add_diffusion_metrics(temp_df)
            
            # Copy over properties from original data
            props_list = ['radius_gyration', 'asymmetry', 'skewness', 'kurtosis',
                         'fracDimension', 'netDispl', 'Straight', 'SVM', 'nnDist_inFrame']
            
            for prop in props_list:
                if prop in track_df.columns:
                    temp_df[prop] = track_df[prop].iloc[0]
            
            temp_df['n_segments'] = n_frames
            
            # Add lag displacement analysis
            temp_df = self._add_lag_displacement(temp_df)
            
            # Add background analysis
            if roi_data is not None:
                temp_df = self._add_background_analysis(temp_df, roi_data, camera_background)
            
            new_df = pd.concat([new_df, temp_df], ignore_index=True)
        
        # Add final calculations
        new_df = self._add_final_calculations(new_df)
        
        # Add mobile and unlinked particles back if keeping all tracks
        if self.parameters.tracks_to_keep == "all":
            # Ensure mobile tracks have same columns
            for col in new_df.columns:
                if col not in df_mobile.columns:
                    df_mobile[col] = np.nan
            new_df = pd.concat([new_df, df_mobile], ignore_index=True)
        
        # Add unlinked points
        if len(unlinked) > 0:
            for col in new_df.columns:
                if col not in unlinked.columns:
                    unlinked[col] = np.nan
            new_df = pd.concat([new_df, unlinked], ignore_index=True)
        
        new_df = new_df.sort_values(by='track_number', ignore_index=True)
        return new_df

    def _interpolate_missing_points(self, df: pd.DataFrame,
                                  image_data: Optional[np.ndarray] = None,
                                  roi_data: Optional[np.ndarray] = None,
                                  camera_background: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Simple interpolation to fill missing points within existing trajectories.
        """
        self.logger.info("Interpolating missing points within trajectories")
        
        new_df = pd.DataFrame()
        
        for track_number in df['track_number'].unique():
            if pd.isna(track_number):
                continue
                
            track_df = df[df['track_number'] == track_number].copy()
            track_df = track_df.sort_values('frame')
            
            # Get frame range
            min_frame = track_df['frame'].min()
            max_frame = track_df['frame'].max()
            all_frames = range(min_frame, max_frame + 1)
            
            # Interpolate x and y coordinates
            x_interp = np.interp(all_frames, track_df['frame'], track_df['x'])
            y_interp = np.interp(all_frames, track_df['frame'], track_df['y'])
            
            # Create interpolated DataFrame
            interp_df = pd.DataFrame({
                'frame': all_frames,
                'track_number': track_number,
                'x': x_interp,
                'y': y_interp
            })
            
            # Copy other columns from original data
            for col in track_df.columns:
                if col not in ['frame', 'track_number', 'x', 'y']:
                    # Use forward fill for discrete values
                    interp_df[col] = np.interp(all_frames, track_df['frame'], track_df[col])
            
            # Extract intensities if image data provided
            if image_data is not None:
                points = np.column_stack((all_frames, x_interp, y_interp))
                intensities = self._get_intensities_from_image(image_data, points)
                interp_df['intensity_interpolated'] = intensities
            
            new_df = pd.concat([new_df, interp_df], ignore_index=True)
        
        return new_df

    def _get_track_ids(self, df: pd.DataFrame) -> List[int]:
        """
        Get track IDs based on the specified grouping method.
        """
        if self.parameters.grouping_method == GroupingMethod.NONE:
            return df['track_number'].unique().tolist()
        
        elif self.parameters.grouping_method == GroupingMethod.IN_PIXEL:
            # Group tracks at same pixel location, keeping longest track
            df['meanXloc'] = df.groupby('track_number')['x'].transform('mean').round(0)
            df['meanYloc'] = df.groupby('track_number')['y'].transform('mean').round(0)
            
            temp_df = df[['track_number', 'n_segments', 'meanXloc', 'meanYloc']].copy()
            temp_df = temp_df.sort_values('n_segments')
            temp_group = temp_df.drop_duplicates(subset=['meanXloc', 'meanYloc'], keep='last')
            
            return temp_group['track_number'].unique().tolist()
        
        elif self.parameters.grouping_method == GroupingMethod.HCLUSTER:
            # Use hierarchical clustering to group nearby tracks
            try:
                import scipy.cluster.hierarchy as hcluster
                
                # Get mean positions for each track
                track_positions = df.groupby('track_number')[['x', 'y']].mean()
                data = track_positions.values
                
                # Perform clustering
                clusters = hcluster.fclusterdata(
                    data, self.parameters.clustering_threshold, criterion="distance"
                )
                
                # Create DataFrame with cluster assignments
                track_positions['cluster'] = clusters
                track_positions['n_segments'] = df.groupby('track_number').size()
                
                # Keep longest track in each cluster
                track_positions = track_positions.sort_values('n_segments')
                selected_tracks = track_positions.drop_duplicates(subset=['cluster'], keep='last')
                
                return selected_tracks.index.tolist()
                
            except ImportError:
                self.logger.warning("scipy not available, falling back to pixel grouping")
                return self._get_track_ids_pixel_grouping(df)
        
        else:
            raise ValueError(f"Unknown grouping method: {self.parameters.grouping_method}")

    def _get_intensities_from_image(self, image_data: np.ndarray, 
                                   points: np.ndarray) -> np.ndarray:
        """
        Extract intensity values from image data at specified points.
        
        Args:
            image_data: 3D array (frames, height, width)
            points: Nx3 array of (frame, x, y) coordinates
            
        Returns:
            Array of intensity values
        """
        n_frames, height, width = image_data.shape
        intensities = []
        
        for point in points:
            frame = int(round(point[0]))
            x = int(round(point[1]))
            y = int(round(point[2]))
            
            # Bounds for 3x3 pixel region
            x_min = max(0, x - 1)
            x_max = min(width, x + 2)
            y_min = max(0, y - 1)
            y_max = min(height, y + 2)
            
            # Ensure frame is within bounds
            if 0 <= frame < n_frames:
                # Extract mean intensity from 3x3 region
                region = image_data[frame, y_min:y_max, x_min:x_max]
                intensity = np.mean(region)
            else:
                intensity = np.nan
            
            intensities.append(intensity)
        
        return np.array(intensities)

    def _add_diffusion_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add diffusion-related metrics to trajectory data."""
        # Set positions relative to origin
        min_frame = df['frame'].min()
        origin_x = df[df['frame'] == min_frame]['x'].iloc[0]
        origin_y = df[df['frame'] == min_frame]['y'].iloc[0]
        
        df['zeroed_X'] = df['x'] - origin_x
        df['zeroed_Y'] = df['y'] - origin_y
        df['lagNumber'] = df['frame'] - min_frame
        df['distanceFromOrigin'] = np.sqrt(df['zeroed_X']**2 + df['zeroed_Y']**2)
        
        # Add differential for distance
        diff = np.diff(df['distanceFromOrigin'].values) / np.diff(df['lagNumber'].values)
        diff = np.insert(diff, 0, 0)
        df['dy-dt: distance'] = diff
        
        return df

    def _add_lag_displacement(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag displacement analysis."""
        # Calculate step displacements
        df['x2'] = df['x'].shift(-1)
        df['y2'] = df['y'].shift(-1)
        df['x2-x1_sqr'] = (df['x2'] - df['x'])**2
        df['y2-y1_sqr'] = (df['y2'] - df['y'])**2
        df['distance'] = np.sqrt(df['x2-x1_sqr'] + df['y2-y1_sqr'])
        
        # Mask final track position
        df['mask'] = True
        df.loc[df.index[-1], 'mask'] = False
        
        # Calculate lag
        df['lag'] = df['distance'].where(df['mask'])
        
        # Add summary statistics
        df['meanLag'] = df['lag'].mean()
        df['track_length'] = df['lag'].sum()
        
        return df

    def _add_background_analysis(self, df: pd.DataFrame,
                               roi_data: np.ndarray,
                               camera_background: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Add background subtraction analysis."""
        # Add ROI background values
        for frame, value in enumerate(roi_data):
            if frame < len(df):
                df.loc[df['frame'] == frame, 'roi_1'] = value
        
        # Add camera background if provided
        if camera_background is not None:
            for frame, value in enumerate(camera_background):
                if frame < len(df):
                    df.loc[df['frame'] == frame, 'camera_black_estimate'] = value
        
        # Smooth the ROI signal
        smoothing_window = max(1, int(len(roi_data) * self.parameters.smoothing_window_fraction))
        roi_smoothed = self._moving_average(roi_data, smoothing_window)
        
        for frame, value in enumerate(roi_smoothed):
            if frame < len(df):
                df.loc[df['frame'] == frame, 'roi_1_smoothed'] = value
        
        return df

    def _add_final_calculations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add final calculations and cleanup."""
        # Squared values for diffusion analysis
        df['d_squared'] = df['distanceFromOrigin']**2
        if 'lag' in df.columns:
            df['lag_squared'] = df['lag']**2
        
        # Delta-t and velocity calculations
        df['dt'] = df['frame'].diff()
        df['dt'] = df['dt'].where(df['dt'] > 0)
        
        if 'lag' in df.columns:
            df['velocity'] = df['lag'] / df['dt']
        
        # Direction relative to origin
        degrees = np.arctan2(df['zeroed_Y'], df['zeroed_X']) / np.pi * 180
        degrees[degrees < 0] = 360 + degrees[degrees < 0]
        df['direction_Relative_To_Origin'] = degrees
        
        # Mean velocity per track
        if 'velocity' in df.columns:
            df['meanVelocity'] = df.groupby('track_number')['velocity'].transform('mean')
        
        # Background subtracted intensities
        if 'roi_1' in df.columns:
            df['intensity - mean roi1'] = df['intensity'] - df['roi_1'].mean()
            
            if 'camera_black_estimate' in df.columns:
                df['intensity - mean roi1 and black'] = (
                    df['intensity'] - df['roi_1'].mean() - df['camera_black_estimate'].mean()
                )
            
            if 'intensity_roiOnMeanXY' in df.columns:
                df['intensity_roiOnMeanXY - mean roi1'] = (
                    df['intensity_roiOnMeanXY'] - df['roi_1'].mean()
                )
                
                if 'camera_black_estimate' in df.columns:
                    df['intensity_roiOnMeanXY - mean roi1 and black'] = (
                        df['intensity_roiOnMeanXY'] - df['roi_1'].mean() - 
                        df['camera_black_estimate'].mean()
                    )
            
            if 'roi_1_smoothed' in df.columns:
                df['intensity - smoothed roi_1'] = df['intensity'] - df['roi_1_smoothed']
                if 'intensity_roiOnMeanXY' in df.columns:
                    df['intensity_roiOnMeanXY - smoothed roi_1'] = (
                        df['intensity_roiOnMeanXY'] - df['roi_1_smoothed']
                    )
        
        # Cleanup intermediate columns
        columns_to_drop = ['x2', 'y2', 'x2-x1_sqr', 'y2-y1_sqr', 'distance', 'mask']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        return df

    def _prepare_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare output columns for interpolation."""
        # Add columns that will be populated during interpolation
        new_columns = {
            'intensity_roiOnMeanXY': np.nan,
            'intensity_roiOnMeanXY - mean roi1': np.nan,
            'intensity_roiOnMeanXY - mean roi1 and black': np.nan,
            'roi_1_smoothed': np.nan,
            'intensity_roiOnMeanXY - smoothed roi_1': np.nan,
            'intensity - smoothed roi_1': np.nan
        }
        
        for col, default_val in new_columns.items():
            if col not in df.columns:
                df[col] = default_val
        
        return df

    def _moving_average(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate moving average with edge padding."""
        if window_size <= 1:
            return data
        
        window = np.ones(window_size) / window_size
        smoothed = np.convolve(data, window, 'valid')
        
        # Pad edges
        start_pad = (len(data) - len(smoothed)) // 2
        end_pad = len(data) - len(smoothed) - start_pad
        
        return np.pad(smoothed, (start_pad, end_pad), 'edge')

    def add_nearest_neighbor_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add nearest neighbor counts within specified radii for each frame.
        
        This implements the multi-radius density analysis from Step_8.
        """
        self.logger.info("Adding nearest neighbor counts")
        
        # Sort by frame
        df = df.sort_values(by=['frame'])
        
        for radius in self.parameters.nn_radii:
            self.logger.debug(f"Calculating NN counts for radius {radius}")
            
            count_list = []
            frames = df['frame'].unique()
            
            for frame in frames:
                frame_data = df[df['frame'] == frame]
                
                if len(frame_data) < 2:
                    count_list.extend([0] * len(frame_data))
                    continue
                
                # Extract coordinates
                coords = frame_data[['x', 'y']].values
                
                # Build KDTree and count neighbors within radius
                tree = KDTree(coords)
                counts = tree.query_radius(coords, r=radius, count_only=True)
                
                # Subtract 1 to exclude self
                counts = counts - 1
                count_list.extend(counts)
            
            # Add to DataFrame
            df[f'nnCountInFrame_within_{radius}_pixels'] = count_list
        
        return df.sort_values(['track_number', 'frame'], ignore_index=True)

    def update_parameters(self, parameters: InterpolationParameters):
        """Update interpolation parameters."""
        self.parameters = parameters
