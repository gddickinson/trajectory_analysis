#!/usr/bin/env python3
"""
# ============================================================================
# PARTICLE LINKING MODULE
# ============================================================================
"""

import logging
import math
from typing import Optional, Dict, List, Any, Tuple, Union
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

# Import scipy with error handling
try:
    from scipy import stats, spatial, ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from sklearn.neighbors import KDTree
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics
import skimage.io as skio
# Import scikit-image with error handling
try:
    from skimage import filters, feature, segmentation, measure
    from skimage.morphology import disk, white_tophat
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. Some detection methods may not work.")
from pathlib import Path
from tqdm import tqdm



class LinkingMethod(ABC):
    """Abstract base class for particle linking methods."""

    @abstractmethod
    def link(self, localizations: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Link particles into trajectories.

        Args:
            localizations: DataFrame with particle detections
            **kwargs: Method-specific parameters

        Returns:
            DataFrame with added track_number column
        """
        pass


class NearestNeighborLinking(LinkingMethod):
    """Simple nearest neighbor linking."""

    def link(self, localizations: pd.DataFrame, max_distance: float = 5.0,
             max_gap_frames: int = 2, min_track_length: int = 3,
             **kwargs) -> pd.DataFrame:
        """Link particles using nearest neighbor method."""

        # Sort by frame
        df = localizations.sort_values('frame').copy()

        # Initialize track numbers
        df['track_number'] = -1
        current_track_id = 0

        # Process frame by frame
        frames = df['frame'].unique()

        for i, frame in enumerate(frames):
            frame_data = df[df['frame'] == frame]

            if i == 0:
                # First frame - start new tracks
                track_ids = list(range(current_track_id, current_track_id + len(frame_data)))
                df.loc[df['frame'] == frame, 'track_number'] = track_ids
                current_track_id += len(frame_data)
            else:
                # Link to previous frames
                self._link_frame(df, frame, frames[:i], max_distance,
                               max_gap_frames, current_track_id)
                current_track_id = df['track_number'].max() + 1

        # Filter by minimum track length
        if min_track_length > 1:
            track_lengths = df.groupby('track_number').size()
            valid_tracks = track_lengths[track_lengths >= min_track_length].index
            df = df[df['track_number'].isin(valid_tracks)]

        return df

    def _link_frame(self, df: pd.DataFrame, current_frame: int,
                   previous_frames: List[int], max_distance: float,
                   max_gap_frames: int, current_track_id: int):
        """Link particles in current frame to previous tracks."""

        current_data = df[df['frame'] == current_frame].copy()

        # Get recent tracks (within gap tolerance)
        recent_frames = [f for f in previous_frames
                        if current_frame - f <= max_gap_frames + 1]

        if not recent_frames:
            # No recent tracks - start new ones
            track_ids = list(range(current_track_id,
                                 current_track_id + len(current_data)))
            df.loc[df['frame'] == current_frame, 'track_number'] = track_ids
            return

        # Get last known positions of active tracks
        recent_data = df[df['frame'].isin(recent_frames)]
        active_tracks = recent_data.groupby('track_number').last()

        # Calculate distances
        current_positions = current_data[['x', 'y']].values
        track_positions = active_tracks[['x', 'y']].values

        if len(track_positions) == 0:
            # No active tracks
            track_ids = list(range(current_track_id,
                                 current_track_id + len(current_data)))
            df.loc[df['frame'] == current_frame, 'track_number'] = track_ids
            return

        # Calculate distance matrix
        distances = spatial.distance.cdist(current_positions, track_positions)

        # Assign particles to tracks (greedy assignment)
        assignments = {}
        used_tracks = set()

        # Sort by distance and assign - Fixed the function name
        indices = np.unravel_index(np.argsort(distances, axis=None), distances.shape)

        for particle_idx, track_idx in zip(indices[0], indices[1]):
            if particle_idx in assignments or track_idx in used_tracks:
                continue

            if distances[particle_idx, track_idx] <= max_distance:
                track_number = active_tracks.index[track_idx]
                assignments[particle_idx] = track_number
                used_tracks.add(track_idx)

        # Apply assignments
        for particle_idx, track_number in assignments.items():
            particle_id = current_data.index[particle_idx]
            df.loc[particle_id, 'track_number'] = track_number

        # Start new tracks for unassigned particles
        unassigned = [i for i in range(len(current_data)) if i not in assignments]
        if unassigned:
            new_track_ids = list(range(current_track_id,
                                     current_track_id + len(unassigned)))
            unassigned_indices = [current_data.index[i] for i in unassigned]
            df.loc[unassigned_indices, 'track_number'] = new_track_ids


class TrackpyLinking(LinkingMethod):
    """Trackpy-based particle linking."""

    def link(self, localizations: pd.DataFrame, max_distance: float = 5.0,
             max_gap_frames: int = 2, min_track_length: int = 3,
             **kwargs) -> pd.DataFrame:
        """Link particles using trackpy."""

        try:
            import trackpy as tp
        except ImportError:
            raise ImportError("Trackpy is required for trackpy linking method")

        tp.quiet()

        # Prepare data for trackpy
        trackpy_data = localizations[['frame', 'x', 'y']].copy()

        # Link trajectories
        trajectories = tp.link(trackpy_data, search_range=max_distance,
                             memory=max_gap_frames)

        # Filter short trajectories
        if min_track_length > 1:
            trajectories = tp.filter_stubs(trajectories, threshold=min_track_length)

        # Merge back with original data
        result = localizations.copy()

        # Map particle indices to track numbers
        if len(trajectories) > 0:
            track_mapping = dict(zip(trajectories.index, trajectories['particle']))
            result['track_number'] = result.index.map(track_mapping)
            result['track_number'] = result['track_number'].fillna(-1).astype(int)
        else:
            result['track_number'] = -1

        return result


class ParticleLinker:
    """Main particle linker class."""

    def __init__(self, parameters=None):
        self.logger = logging.getLogger(__name__)
        self.parameters = parameters or {}

        # Initialize linking methods
        self.methods = {
            'nearest_neighbor': NearestNeighborLinking(),
            'trackpy': TrackpyLinking()
        }

    def link_particles(self, localizations: pd.DataFrame, method: str = 'nearest_neighbor',
                      **kwargs) -> pd.DataFrame:
        """Link particles into trajectories."""

        if method not in self.methods:
            raise ValueError(f"Unknown linking method: {method}")

        linker = self.methods[method]

        # Merge parameters
        params = {**self.parameters, **kwargs}

        self.logger.info(f"Linking particles using {method} method")
        result = linker.link(localizations, **params)

        # Count tracks
        n_tracks = result['track_number'].nunique() if 'track_number' in result.columns else 0
        self.logger.info(f"Created {n_tracks} trajectories")

        return result

    def update_parameters(self, parameters):
        """Update linking parameters."""
        self.parameters = parameters
