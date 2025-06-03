#!/usr/bin/env python3
"""
Core Analysis Modules
====================

Detection, linking, feature calculation, and classification modules.
"""

import logging
import math
from typing import Optional, Dict, List, Any, Tuple, Union
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy import stats, spatial, ndimage
from sklearn.neighbors import KDTree
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics
import skimage.io as skio
from skimage import filters, feature, segmentation, measure
from skimage.morphology import disk, white_tophat
from pathlib import Path
from tqdm import tqdm



# ============================================================================
# FEATURE CALCULATION MODULE
# ============================================================================

class FeatureCalculator:
    """Calculate trajectory features."""

    def __init__(self, parameters=None):
        self.logger = logging.getLogger(__name__)
        self.parameters = parameters or {}

    def calculate_features(self, trajectory_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all trajectory features."""

        df = trajectory_data.copy()

        # Calculate basic features
        df = self._add_basic_features(df)

        # Calculate radius of gyration
        df = self._calculate_radius_of_gyration(df)

        # Calculate asymmetry and shape features
        df = self._calculate_shape_features(df)

        # Calculate diffusion features
        df = self._calculate_diffusion_features(df)

        # Calculate velocity features
        df = self._calculate_velocity_features(df)

        # Calculate nearest neighbors
        df = self._calculate_nearest_neighbors(df)

        return df

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic trajectory features."""

        # Track length
        df['track_length'] = df.groupby('track_number')['track_number'].transform('count')

        # Track duration (in frames)
        df['track_duration'] = df.groupby('track_number')['frame'].transform(
            lambda x: x.max() - x.min() + 1
        )

        return df

    def _calculate_radius_of_gyration(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate radius of gyration for each track."""

        def calculate_rg(track_data):
            positions = track_data[['x', 'y']].values
            if len(positions) < 2:
                return np.nan

            # Calculate center of mass
            center = np.mean(positions, axis=0)

            # Calculate squared distances from center
            squared_distances = np.sum((positions - center)**2, axis=1)

            # Radius of gyration
            rg = np.sqrt(np.mean(squared_distances))

            return rg

        # Calculate for each track
        track_rg = df.groupby('track_number').apply(calculate_rg)
        track_rg.name = 'radius_gyration'

        # Map back to original dataframe
        df = df.join(track_rg, on='track_number')

        return df

    def _calculate_shape_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate asymmetry, skewness, and kurtosis."""

        def calculate_shape_features(track_data):
            positions = track_data[['x', 'y']].values
            if len(positions) < 3:
                return pd.Series({
                    'asymmetry': np.nan,
                    'skewness': np.nan,
                    'kurtosis': np.nan,
                    'fracDimension': np.nan,
                    'netDispl': np.nan,
                    'Straight': np.nan
                })

            center = np.mean(positions, axis=0)
            normed_points = positions - center

            # Gyration tensor
            tensor = np.einsum('im,in->mn', normed_points, normed_points) / len(positions)
            eigenvalues, eigenvectors = np.linalg.eig(tensor)
            eigenvalues = np.real(eigenvalues)

            # Asymmetry
            try:
                l1, l2 = sorted(eigenvalues, reverse=True)
                asymmetry_num = (l1 - l2)**2
                asymmetry_den = 2 * (l1 + l2)**2
                asymmetry = -math.log(1 - (asymmetry_num / asymmetry_den))
            except (ValueError, ZeroDivisionError):
                asymmetry = np.nan

            # Principal axis projections for skewness/kurtosis
            try:
                principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
                steps = np.diff(positions, axis=0)
                projections = np.dot(steps, principal_axis)
                skewness = stats.skew(projections)
                kurtosis = stats.kurtosis(projections)
            except:
                skewness = np.nan
                kurtosis = np.nan

            # Fractal dimension (simplified)
            try:
                step_lengths = np.sqrt(np.sum(steps**2, axis=1))
                total_length = np.sum(step_lengths)
                net_displacement = np.linalg.norm(positions[-1] - positions[0])

                if total_length > 0 and net_displacement > 0:
                    fractal_dim = math.log(len(positions)) / math.log(len(positions) * net_displacement / total_length)
                else:
                    fractal_dim = np.nan
            except:
                fractal_dim = np.nan

            # Net displacement
            net_disp = np.linalg.norm(positions[-1] - positions[0])

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

            return pd.Series({
                'asymmetry': asymmetry,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'fracDimension': fractal_dim,
                'netDispl': net_disp,
                'Straight': straightness
            })

        # Calculate for each track
        shape_features = df.groupby('track_number').apply(calculate_shape_features)

        # Unstack and join to original dataframe
        for feature in shape_features.columns:
            df = df.join(shape_features[feature], on='track_number', rsuffix='_temp')

        return df

    def _calculate_diffusion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate diffusion-related features."""

        def calculate_diffusion(track_data):
            track_data = track_data.sort_values('frame')
            positions = track_data[['x', 'y']].values

            if len(positions) < 3:
                return pd.Series({'diffusion_coefficient': np.nan})

            # Calculate MSD for first few lag times
            max_lag = min(len(positions) - 1, 10)
            msd_values = []

            for lag in range(1, max_lag + 1):
                displacements = positions[lag:] - positions[:-lag]
                squared_displacements = np.sum(displacements**2, axis=1)
                msd = np.mean(squared_displacements)
                msd_values.append(msd)

            # Fit linear regression to get diffusion coefficient
            if len(msd_values) >= 3:
                lag_times = np.arange(1, len(msd_values) + 1)
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    lag_times[:5], msd_values[:5]  # Use first 5 points
                )
                # D = slope / 4 for 2D diffusion
                diffusion_coeff = slope / 4.0
            else:
                diffusion_coeff = np.nan

            return pd.Series({'diffusion_coefficient': diffusion_coeff})

        # Calculate for each track
        diffusion_features = df.groupby('track_number').apply(calculate_diffusion)
        df = df.join(diffusion_features, on='track_number')

        return df

    def _calculate_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate velocity features."""

        # Sort by track and frame
        df = df.sort_values(['track_number', 'frame'])

        # Calculate step displacements
        df['dx'] = df.groupby('track_number')['x'].diff()
        df['dy'] = df.groupby('track_number')['y'].diff()
        df['step_size'] = np.sqrt(df['dx']**2 + df['dy']**2)

        # Calculate instantaneous velocity (assuming frame rate from parameters)
        frame_rate = self.parameters.get('frame_rate', 1.0)
        df['dt'] = df.groupby('track_number')['frame'].diff()
        df['velocity'] = df['step_size'] / (df['dt'] / frame_rate)

        # Mean velocity per track
        df['mean_velocity'] = df.groupby('track_number')['velocity'].transform('mean')

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

            # Query for 2 nearest neighbors (including self)
            distances, indices = tree.query(positions, k=2)

            # Take the second nearest (first is self)
            nn_distances.extend(distances[:, 1])

        df['nn_distance'] = nn_distances

        return df

    def update_parameters(self, parameters):
        """Update calculation parameters."""
        self.parameters = parameters


