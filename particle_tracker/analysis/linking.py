#!/usr/bin/env python3
"""
# ============================================================================
# IMPROVED PARTICLE LINKING MODULE
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

        # Sort by distance and assign
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
    """Trackpy-based particle linking with improved parameters."""

    def link(self, localizations: pd.DataFrame, max_distance: float = 5.0,
             max_gap_frames: int = 2, min_track_length: int = 3,
             **kwargs) -> pd.DataFrame:
        """Link particles using trackpy with optimized parameters."""

        try:
            import trackpy as tp
        except ImportError:
            raise ImportError("Trackpy is required for trackpy linking method")

        tp.quiet()

        # Prepare data for trackpy
        trackpy_data = localizations[['frame', 'x', 'y']].copy()

        # Estimate better search_range based on data
        search_range = self._estimate_search_range(trackpy_data, max_distance)

        # Adjust memory parameter based on data density
        memory = min(max_gap_frames, self._estimate_memory(trackpy_data))

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Trackpy linking with search_range={search_range:.2f}, memory={memory}")

        # Link trajectories with improved parameters
        try:
            trajectories = tp.link(
                trackpy_data,
                search_range=search_range,
                memory=memory,
                adaptive_stop=search_range * 0.1,  # Stop adaptive search at 10% of original
                adaptive_step=0.95,  # Reduce search range by 5% each step
                link_strategy='auto'  # Use best available algorithm
            )
        except Exception as e:
            self.logger.warning(f"Trackpy linking failed with optimized parameters: {e}")
            self.logger.info("Falling back to basic trackpy parameters")

            # Fallback to simpler parameters
            trajectories = tp.link(
                trackpy_data,
                search_range=min(search_range, 3.0),  # Smaller, safer range
                memory=min(memory, 1)  # Smaller memory
            )

        # Filter short trajectories
        if min_track_length > 1:
            trajectories = tp.filter_stubs(trajectories, threshold=min_track_length)

        # Merge back with original data
        result = localizations.copy()

        # Map particle indices to track numbers
        if len(trajectories) > 0:
            # Create mapping from (frame, x, y) to track number
            track_mapping = {}
            for _, row in trajectories.iterrows():
                key = (row['frame'], row['x'], row['y'])
                track_mapping[key] = row['particle']

            # Apply mapping to original data
            def get_track_number(row):
                key = (row['frame'], row['x'], row['y'])
                return track_mapping.get(key, -1)

            result['track_number'] = result.apply(get_track_number, axis=1)

            # Remove unlinked particles
            result = result[result['track_number'] != -1]
        else:
            result['track_number'] = -1

        return result

    def _estimate_search_range(self, data: pd.DataFrame, max_distance: float) -> float:
        """Estimate appropriate search range based on data characteristics."""

        if len(data) < 10:
            return min(max_distance, 2.0)

        # Calculate typical inter-particle distances
        try:
            # Sample a few frames to estimate particle density
            sample_frames = data['frame'].unique()[:5]
            sample_data = data[data['frame'].isin(sample_frames)]

            if len(sample_data) < 5:
                return min(max_distance, 2.0)

            # Calculate nearest neighbor distances
            from sklearn.neighbors import NearestNeighbors
            positions = sample_data[['x', 'y']].values
            nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(positions)
            distances, indices = nbrs.kneighbors(positions)

            # Use 90th percentile of nearest neighbor distances as base
            nn_distances = distances[:, 1]  # Second nearest (first is self)
            typical_distance = np.percentile(nn_distances, 90)

            # Search range should be 2-3x typical movement, but not too large
            estimated_range = min(typical_distance * 2.5, max_distance, 3.0)

            return max(estimated_range, 0.5)  # At least 0.5 pixels

        except Exception:
            # Fallback to conservative estimate
            return min(max_distance * 0.5, 2.0)

    def _estimate_memory(self, data: pd.DataFrame) -> int:
        """Estimate appropriate memory parameter based on data."""

        # Check frame density - if frames are consecutive, lower memory is fine
        frames = data['frame'].unique()
        if len(frames) < 3:
            return 0

        frame_gaps = np.diff(sorted(frames))
        avg_gap = np.mean(frame_gaps)

        # If frames are mostly consecutive, use lower memory
        if avg_gap <= 1.1:
            return min(2, max(0, int(np.max(frame_gaps))))
        else:
            # Larger gaps between frames, might need more memory
            return min(3, int(avg_gap))


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

        # Log linking info
        self.logger.info(f"Linking particles using {method} method")
        self.logger.info(f"Input: {len(localizations)} localizations across {localizations['frame'].nunique()} frames")

        # Auto-select method based on detection method if not explicitly chosen
        if method == 'nearest_neighbor' and 'detection_method' in params:
            if params['detection_method'] == 'trackpy':
                self.logger.info("Auto-switching to trackpy linking for trackpy detections")
                method = 'trackpy'
                linker = self.methods[method]

        result = linker.link(localizations, **params)

        # Count tracks
        if 'track_number' in result.columns:
            n_tracks = result['track_number'].nunique()
            track_lengths = result.groupby('track_number').size()
            avg_length = track_lengths.mean()

            self.logger.info(f"Created {n_tracks} trajectories")
            self.logger.info(f"Average track length: {avg_length:.1f} points")
            self.logger.info(f"Track length range: {track_lengths.min()}-{track_lengths.max()}")
        else:
            self.logger.warning("No track_number column in result")

        return result

    def update_parameters(self, parameters):
        """Update linking parameters."""
        self.parameters = parameters

    def suggest_parameters(self, localizations: pd.DataFrame) -> Dict[str, Any]:
        """Suggest linking parameters based on data characteristics."""

        suggestions = {}

        if len(localizations) == 0:
            return suggestions

        # Analyze data characteristics
        n_frames = localizations['frame'].nunique()
        particles_per_frame = len(localizations) / n_frames

        # Estimate particle density
        if 'x' in localizations.columns and 'y' in localizations.columns:
            x_range = localizations['x'].max() - localizations['x'].min()
            y_range = localizations['y'].max() - localizations['y'].min()
            area = x_range * y_range

            if area > 0:
                density = particles_per_frame / area

                # Suggest parameters based on density
                if density > 0.01:  # High density
                    suggestions['max_distance'] = 2.0
                    suggestions['max_gap_frames'] = 1
                    suggestions['method'] = 'trackpy'
                elif density > 0.001:  # Medium density
                    suggestions['max_distance'] = 3.0
                    suggestions['max_gap_frames'] = 2
                    suggestions['method'] = 'trackpy'
                else:  # Low density
                    suggestions['max_distance'] = 5.0
                    suggestions['max_gap_frames'] = 3
                    suggestions['method'] = 'nearest_neighbor'

        return suggestions
