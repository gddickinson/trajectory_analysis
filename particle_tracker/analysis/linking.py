#!/usr/bin/env python3
"""
Enhanced Particle Linking Module
================================

Advanced particle linking with optimized parameters, quality metrics,
and support for batch processing workflows.
"""

import logging
import math
import time
from typing import Optional, Dict, List, Any, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

# Import scipy with error handling
try:
    from scipy import stats, spatial, ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from sklearn.neighbors import KDTree
from pathlib import Path
from tqdm import tqdm


class LinkingStrategy(Enum):
    """Enumeration of linking strategies."""
    NEAREST_NEIGHBOR = "nearest_neighbor"
    TRACKPY_STANDARD = "trackpy_standard"
    TRACKPY_ADAPTIVE = "trackpy_adaptive"
    TRACKPY_VELOCITY = "trackpy_velocity"
    TRACKPY_HYBRID = "trackpy_hybrid"


@dataclass
class LinkingQualityMetrics:
    """Container for linking quality assessment metrics."""
    total_detections: int = 0
    linked_detections: int = 0
    unlinked_detections: int = 0
    total_tracks: int = 0
    avg_track_length: float = 0.0
    median_track_length: float = 0.0
    min_track_length: int = 0
    max_track_length: int = 0
    linking_efficiency: float = 0.0  # Percentage of detections linked
    track_length_distribution: Dict[int, int] = None

    def __post_init__(self):
        if self.track_length_distribution is None:
            self.track_length_distribution = {}


@dataclass
class LinkingParameters:
    """Container for linking parameters with smart defaults."""
    strategy: LinkingStrategy = LinkingStrategy.TRACKPY_STANDARD
    max_distance: float = 2.0
    max_gap_frames: int = 1
    min_track_length: int = 5
    adaptive_stop: float = 0.1
    adaptive_step: float = 0.95
    memory: int = 1
    link_strategy: str = "auto"
    predict_velocity: bool = False
    quality_threshold: float = 0.8
    density_threshold: float = 0.01
    frame_rate: float = 10.0
    pixel_size: float = 108.0


class LinkingMethod(ABC):
    """Abstract base class for particle linking methods."""

    @abstractmethod
    def link(self, localizations: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, LinkingQualityMetrics]:
        """Link particles into trajectories.

        Args:
            localizations: DataFrame with particle detections
            **kwargs: Method-specific parameters

        Returns:
            Tuple of (DataFrame with added track_number column, quality metrics)
        """
        pass


class NearestNeighborLinking(LinkingMethod):
    """Enhanced nearest neighbor linking with better performance and metrics."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def link(self, localizations: pd.DataFrame, max_distance: float = 5.0,
             max_gap_frames: int = 2, min_track_length: int = 3,
             **kwargs) -> Tuple[pd.DataFrame, LinkingQualityMetrics]:
        """Link particles using enhanced nearest neighbor method."""

        start_time = time.time()
        self.logger.info(f"Starting nearest neighbor linking with {len(localizations)} detections")

        # Sort by frame for efficient processing
        df = localizations.sort_values('frame').copy()

        # Initialize track numbers
        df['track_number'] = -1
        current_track_id = 0

        # Track active trajectories
        active_tracks = {}  # {track_id: {frame: index, x: float, y: float}}

        # Process frame by frame
        frames = df['frame'].unique()
        total_frames = len(frames)

        for frame_idx, frame in enumerate(frames):
            if frame_idx % 100 == 0:
                self.logger.debug(f"Processing frame {frame_idx}/{total_frames}")

            frame_data = df[df['frame'] == frame].copy()

            if frame_idx == 0:
                # First frame - start new tracks
                track_ids = list(range(current_track_id, current_track_id + len(frame_data)))
                df.loc[df['frame'] == frame, 'track_number'] = track_ids

                # Initialize active tracks
                for i, (idx, row) in enumerate(frame_data.iterrows()):
                    track_id = track_ids[i]
                    active_tracks[track_id] = {
                        'frame': frame,
                        'x': row['x'],
                        'y': row['y'],
                        'last_seen': frame
                    }

                current_track_id += len(frame_data)
            else:
                # Link to previous tracks
                assignments, new_tracks = self._link_frame_optimized(
                    frame_data, active_tracks, frame, max_distance, max_gap_frames
                )

                # Apply assignments
                for particle_idx, track_id in assignments.items():
                    df_idx = frame_data.index[particle_idx]
                    df.loc[df_idx, 'track_number'] = track_id

                    # Update active track
                    row = frame_data.iloc[particle_idx]
                    active_tracks[track_id].update({
                        'frame': frame,
                        'x': row['x'],
                        'y': row['y'],
                        'last_seen': frame
                    })

                # Start new tracks for unassigned particles
                for particle_idx in new_tracks:
                    df_idx = frame_data.index[particle_idx]
                    df.loc[df_idx, 'track_number'] = current_track_id

                    row = frame_data.iloc[particle_idx]
                    active_tracks[current_track_id] = {
                        'frame': frame,
                        'x': row['x'],
                        'y': row['y'],
                        'last_seen': frame
                    }
                    current_track_id += 1

                # Remove stale tracks
                stale_tracks = [
                    track_id for track_id, track_info in active_tracks.items()
                    if frame - track_info['last_seen'] > max_gap_frames
                ]
                for track_id in stale_tracks:
                    del active_tracks[track_id]

        # Filter by minimum track length
        if min_track_length > 1:
            track_lengths = df.groupby('track_number').size()
            valid_tracks = track_lengths[track_lengths >= min_track_length].index
            df = df[df['track_number'].isin(valid_tracks)]

        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(df, localizations)

        elapsed_time = time.time() - start_time
        self.logger.info(f"Nearest neighbor linking completed in {elapsed_time:.2f}s")

        return df, quality_metrics

    def _link_frame_optimized(self, frame_data: pd.DataFrame, active_tracks: Dict,
                             current_frame: int, max_distance: float,
                             max_gap_frames: int) -> Tuple[Dict[int, int], List[int]]:
        """Optimized frame linking using vectorized operations."""

        if not active_tracks:
            return {}, list(range(len(frame_data)))

        # Get current particle positions
        current_positions = frame_data[['x', 'y']].values

        # Get active track positions and IDs
        track_positions = []
        track_ids = []

        for track_id, track_info in active_tracks.items():
            if current_frame - track_info['last_seen'] <= max_gap_frames:
                track_positions.append([track_info['x'], track_info['y']])
                track_ids.append(track_id)

        if not track_positions:
            return {}, list(range(len(frame_data)))

        track_positions = np.array(track_positions)

        # Calculate distance matrix efficiently
        distances = spatial.distance.cdist(current_positions, track_positions)

        # Hungarian assignment (greedy approximation for speed)
        assignments = {}
        used_tracks = set()

        # Sort by distance and assign greedily
        flat_indices = np.unravel_index(np.argsort(distances.ravel()), distances.shape)

        for particle_idx, track_idx in zip(flat_indices[0], flat_indices[1]):
            if particle_idx in assignments or track_idx in used_tracks:
                continue

            if distances[particle_idx, track_idx] <= max_distance:
                assignments[particle_idx] = track_ids[track_idx]
                used_tracks.add(track_idx)

        # Identify unassigned particles
        unassigned = [i for i in range(len(frame_data)) if i not in assignments]

        return assignments, unassigned

    def _calculate_quality_metrics(self, linked_df: pd.DataFrame,
                                 original_df: pd.DataFrame) -> LinkingQualityMetrics:
        """Calculate linking quality metrics."""

        total_detections = len(original_df)
        linked_detections = len(linked_df)
        unlinked_detections = total_detections - linked_detections

        if linked_detections == 0:
            return LinkingQualityMetrics(
                total_detections=total_detections,
                unlinked_detections=unlinked_detections
            )

        # Track statistics
        track_lengths = linked_df.groupby('track_number').size()
        total_tracks = len(track_lengths)

        track_length_dist = track_lengths.value_counts().to_dict()

        return LinkingQualityMetrics(
            total_detections=total_detections,
            linked_detections=linked_detections,
            unlinked_detections=unlinked_detections,
            total_tracks=total_tracks,
            avg_track_length=track_lengths.mean(),
            median_track_length=track_lengths.median(),
            min_track_length=track_lengths.min(),
            max_track_length=track_lengths.max(),
            linking_efficiency=(linked_detections / total_detections) * 100,
            track_length_distribution=track_length_dist
        )


class TrackpyLinking(LinkingMethod):
    """Enhanced trackpy-based linking with adaptive parameters and quality assessment."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def link(self, localizations: pd.DataFrame, strategy: LinkingStrategy = LinkingStrategy.TRACKPY_STANDARD,
             max_distance: float = 2.0, max_gap_frames: int = 1, min_track_length: int = 5,
             adaptive_stop: float = 0.1, adaptive_step: float = 0.95,
             memory: int = 1, link_strategy: str = "auto", **kwargs) -> Tuple[pd.DataFrame, LinkingQualityMetrics]:
        """Link particles using enhanced trackpy with multiple strategies."""

        try:
            import trackpy as tp
            tp.quiet()
        except ImportError:
            raise ImportError("Trackpy is required for trackpy linking methods")

        start_time = time.time()
        self.logger.info(f"Starting trackpy linking ({strategy.value}) with {len(localizations)} detections")

        # Prepare data for trackpy
        trackpy_data = localizations[['frame', 'x', 'y']].copy()

        # Add ID column if not present
        if 'id' not in trackpy_data.columns:
            trackpy_data = trackpy_data.reset_index()
            trackpy_data['id'] = trackpy_data.index

        # Estimate optimal search range if not provided
        if max_distance == 'auto':
            max_distance = self._estimate_search_range(trackpy_data)
            self.logger.info(f"Auto-estimated search range: {max_distance:.2f}")

        # Estimate optimal memory if not provided
        if memory == 'auto':
            memory = self._estimate_memory(trackpy_data, max_gap_frames)
            self.logger.info(f"Auto-estimated memory: {memory}")

        # Link trajectories based on strategy
        try:
            if strategy == LinkingStrategy.TRACKPY_STANDARD:
                trajectories = tp.link(
                    trackpy_data,
                    search_range=max_distance,
                    memory=memory,
                    link_strategy=link_strategy
                )

            elif strategy == LinkingStrategy.TRACKPY_ADAPTIVE:
                trajectories = tp.link(
                    trackpy_data,
                    search_range=max_distance,
                    memory=memory,
                    adaptive_stop=adaptive_stop,
                    adaptive_step=adaptive_step,
                    link_strategy=link_strategy
                )

            elif strategy == LinkingStrategy.TRACKPY_VELOCITY:
                pred = tp.predict.NearestVelocityPredict()
                trajectories = pred.link_df(
                    trackpy_data,
                    search_range=max_distance,
                    memory=memory
                )

            elif strategy == LinkingStrategy.TRACKPY_HYBRID:
                # Try adaptive first, fall back to standard if it fails
                try:
                    trajectories = tp.link(
                        trackpy_data,
                        search_range=max_distance,
                        memory=memory,
                        adaptive_stop=adaptive_stop,
                        adaptive_step=adaptive_step,
                        link_strategy=link_strategy
                    )
                except Exception as e:
                    self.logger.warning(f"Adaptive linking failed: {e}, falling back to standard")
                    trajectories = tp.link(
                        trackpy_data,
                        search_range=max_distance,
                        memory=memory,
                        link_strategy=link_strategy
                    )

            else:
                raise ValueError(f"Unknown trackpy strategy: {strategy}")

        except Exception as e:
            self.logger.error(f"Trackpy linking failed: {e}")
            # Fall back to conservative parameters
            self.logger.info("Falling back to conservative trackpy parameters")
            trajectories = tp.link(
                trackpy_data,
                search_range=min(max_distance, 2.0),
                memory=min(memory, 1),
                link_strategy='auto'
            )

        # Filter short trajectories
        if min_track_length > 1:
            trajectories = tp.filter_stubs(trajectories, threshold=min_track_length)

        # Merge back with original data
        result = localizations.copy()

        if len(trajectories) > 0:
            # Create mapping from original indices to track numbers
            track_mapping = {}

            # Map using the index if available, otherwise use position matching
            if 'id' in trajectories.columns and 'id' in localizations.columns:
                for _, row in trajectories.iterrows():
                    track_mapping[row['id']] = row['particle']

                result['track_number'] = result['id'].map(track_mapping).fillna(-1)
            else:
                # Position-based matching as fallback
                for _, row in trajectories.iterrows():
                    # Find matching row in original data
                    matches = localizations[
                        (localizations['frame'] == row['frame']) &
                        (np.abs(localizations['x'] - row['x']) < 0.001) &
                        (np.abs(localizations['y'] - row['y']) < 0.001)
                    ]

                    if len(matches) > 0:
                        result.loc[matches.index[0], 'track_number'] = row['particle']

            # Remove unlinked particles
            result = result[result['track_number'] != -1]
        else:
            result['track_number'] = -1

        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(result, localizations)

        elapsed_time = time.time() - start_time
        self.logger.info(f"Trackpy linking completed in {elapsed_time:.2f}s")

        return result, quality_metrics

    def _estimate_search_range(self, data: pd.DataFrame) -> float:
        """Estimate appropriate search range based on data characteristics."""

        if len(data) < 10:
            return 2.0

        try:
            # Sample frames to estimate particle density and typical movements
            sample_frames = data['frame'].unique()[:min(10, len(data['frame'].unique()))]
            sample_data = data[data['frame'].isin(sample_frames)]

            if len(sample_data) < 5:
                return 2.0

            # Estimate typical inter-particle distances
            from sklearn.neighbors import NearestNeighbors
            positions = sample_data[['x', 'y']].values

            if len(positions) < 2:
                return 2.0

            nbrs = NearestNeighbors(n_neighbors=min(3, len(positions)), algorithm='ball_tree').fit(positions)
            distances, _ = nbrs.kneighbors(positions)

            # Use 75th percentile of second nearest neighbor distances
            if distances.shape[1] >= 2:
                nn_distances = distances[:, 1]  # Second nearest (first is self)
                typical_distance = np.percentile(nn_distances, 75)
            else:
                typical_distance = np.mean(distances[:, 0])

            # Search range should be 1.5-2x typical movement
            estimated_range = typical_distance * 1.8

            # Clamp to reasonable bounds
            return max(0.5, min(estimated_range, 5.0))

        except Exception as e:
            self.logger.warning(f"Error estimating search range: {e}")
            return 2.0

    def _estimate_memory(self, data: pd.DataFrame, max_gap_frames: int) -> int:
        """Estimate appropriate memory parameter based on data."""

        frames = data['frame'].unique()
        if len(frames) < 3:
            return 0

        # Check frame gaps
        frame_gaps = np.diff(sorted(frames))
        max_gap = np.max(frame_gaps)
        typical_gap = np.median(frame_gaps)

        # If frames are mostly consecutive, use lower memory
        if typical_gap <= 1.2:
            return min(max_gap_frames, max(0, int(max_gap - 1)))
        else:
            # Larger gaps, might need more memory
            return min(max_gap_frames, int(typical_gap))

    def _calculate_quality_metrics(self, linked_df: pd.DataFrame,
                                 original_df: pd.DataFrame) -> LinkingQualityMetrics:
        """Calculate linking quality metrics."""

        total_detections = len(original_df)
        linked_detections = len(linked_df)
        unlinked_detections = total_detections - linked_detections

        if linked_detections == 0:
            return LinkingQualityMetrics(
                total_detections=total_detections,
                unlinked_detections=unlinked_detections
            )

        # Track statistics
        track_lengths = linked_df.groupby('track_number').size()
        total_tracks = len(track_lengths)

        track_length_dist = track_lengths.value_counts().to_dict()

        return LinkingQualityMetrics(
            total_detections=total_detections,
            linked_detections=linked_detections,
            unlinked_detections=unlinked_detections,
            total_tracks=total_tracks,
            avg_track_length=track_lengths.mean(),
            median_track_length=track_lengths.median(),
            min_track_length=track_lengths.min(),
            max_track_length=track_lengths.max(),
            linking_efficiency=(linked_detections / total_detections) * 100,
            track_length_distribution=track_length_dist
        )


class ParticleLinker:
    """Enhanced particle linker with smart parameter optimization and quality assessment."""

    def __init__(self, parameters: Optional[LinkingParameters] = None):
        self.logger = logging.getLogger(__name__)
        self.parameters = parameters or LinkingParameters()

        # Initialize linking methods
        self.methods = {
            LinkingStrategy.NEAREST_NEIGHBOR: NearestNeighborLinking(),
            LinkingStrategy.TRACKPY_STANDARD: TrackpyLinking(),
            LinkingStrategy.TRACKPY_ADAPTIVE: TrackpyLinking(),
            LinkingStrategy.TRACKPY_VELOCITY: TrackpyLinking(),
            LinkingStrategy.TRACKPY_HYBRID: TrackpyLinking()
        }

    def link_particles(self, localizations: pd.DataFrame,
                      strategy: Optional[LinkingStrategy] = None,
                      **kwargs) -> Tuple[pd.DataFrame, LinkingQualityMetrics]:
        """Link particles into trajectories with enhanced capabilities."""

        try:
            # Validate input data
            if localizations is None or len(localizations) == 0:
                self.logger.warning("No particles to link - returning empty DataFrame")
                import pandas as pd
                return pd.DataFrame(columns=['frame', 'x', 'y', 'particle'])

            # Check required columns
            required_columns = ['frame', 'x', 'y']
            missing_columns = [col for col in required_columns if col not in localizations.columns]

            if missing_columns:
                self.logger.error(f"Missing required columns. Need: {required_columns}")
                self.logger.error(f"Available columns: {list(localizations.columns)}")

                # Try to fix common column name issues
                if 'frame' not in localizations.columns:
                    if 't' in localizations.columns:
                        localizations['frame'] = localizations['t']
                        self.logger.info("Mapped 't' column to 'frame'")
                    elif 'time' in localizations.columns:
                        localizations['frame'] = localizations['time']
                        self.logger.info("Mapped 'time' column to 'frame'")
                    else:
                        raise ValueError("Invalid input data for particle linking")

                # Recheck after potential fixes
                missing_columns = [col for col in required_columns if col not in localizations.columns]
                if missing_columns:
                    raise ValueError("Invalid input data for particle linking")

            # Handle both string and enum strategy types
            if hasattr(strategy, 'value'):
                strategy_name = strategy.value
            else:
                strategy_name = str(strategy)
            self.logger.info(f"Using strategy: {strategy_name}")

            # Continue with original linking logic...
            # (rest of your existing link_particles method)

        except Exception as e:
            self.logger.error(f"Particle linking failed: {e}")
            import pandas as pd
            return pd.DataFrame(columns=['frame', 'x', 'y', 'particle'])


        if strategy is None:
            strategy = self.parameters.get('strategy', 'nearest_neighbor')  # Use default if not specified

        # Validate input data
        if not self._validate_input_data(localizations):
            raise ValueError("Invalid input data for particle linking")

        # Auto-optimize parameters based on data characteristics
        optimized_params = self.optimize_parameters(localizations, strategy, **kwargs)

        # Get the appropriate linker
        if strategy == LinkingStrategy.NEAREST_NEIGHBOR:
            linker = self.methods[LinkingStrategy.NEAREST_NEIGHBOR]
        else:
            linker = self.methods[LinkingStrategy.TRACKPY_STANDARD]  # All trackpy strategies use same class

        # Log linking info
        self.logger.info(f"Linking {len(localizations)} detections across {localizations['frame'].nunique()} frames")
        # Handle both string and enum strategy types
        if hasattr(strategy, 'value'):
            strategy_name = strategy.value
        else:
            strategy_name = str(strategy)
        self.logger.info(f"Using strategy: {strategy_name}")
        self.logger.info(f"Optimized parameters: {optimized_params}")

        # Perform linking
        if strategy == LinkingStrategy.NEAREST_NEIGHBOR:
            result, metrics = linker.link(localizations, **optimized_params)
        else:
            result, metrics = linker.link(localizations, strategy=strategy, **optimized_params)

        # Log results
        self.logger.info(f"Linking completed: {metrics.total_tracks} tracks, "
                        f"{metrics.linking_efficiency:.1f}% efficiency")

        return result, metrics

    def _validate_input_data(self, localizations: pd.DataFrame) -> bool:
        """Validate input localization data."""

        required_columns = ['frame', 'x', 'y']
        if not all(col in localizations.columns for col in required_columns):
            self.logger.error(f"Missing required columns. Need: {required_columns}")
            return False

        if len(localizations) == 0:
            self.logger.error("Empty localization data")
            return False

        # Check for reasonable coordinate values
        if localizations['x'].isna().all() or localizations['y'].isna().all():
            self.logger.error("All coordinate values are NaN")
            return False

        return True

    def optimize_parameters(self, localizations: pd.DataFrame,
                           strategy: LinkingStrategy, **kwargs) -> Dict[str, Any]:
        """Optimize linking parameters based on data characteristics."""
        # Start with default parameters
        params = {
            'max_distance': self.parameters.get('max_distance', 5.0),
            'max_gap_frames': self.parameters.get('max_gap_frames', 2),
            'min_track_length': self.parameters.get('min_track_length', 3),
            'memory': self.parameters.get('memory', 3),
            'adaptive_stop': self.parameters.get('adaptive_stop', True),
            'adaptive_step': self.parameters.get('adaptive_step', 0.95),
            'link_strategy': self.parameters.get('link_strategy', 'nearest_neighbor')
        }

        # Update with any provided kwargs
        params.update(kwargs)

        # Analyze data characteristics
        data_analysis = self._analyze_data_characteristics(localizations)

        # Optimize based on data characteristics
        if data_analysis['particle_density'] > self.parameters.get('density_threshold', 10.0):
            # High density data - use more conservative parameters
            params['max_distance'] = min(params['max_distance'], 1.5)
            params['memory'] = min(params['memory'], 1)
            params['min_track_length'] = max(params['min_track_length'], 8)
            self.logger.info("High density detected - using conservative parameters")

        if data_analysis['temporal_sparsity'] > 0.3:
            # Sparse temporal sampling - might need more memory
            params['memory'] = min(params['memory'] + 1, 3)
            params['max_gap_frames'] = min(params['max_gap_frames'] + 1, 5)
            self.logger.info("Sparse temporal sampling detected - increasing memory")

        if data_analysis['n_frames'] < 10:
            # Short sequences - reduce minimum track length
            params['min_track_length'] = max(2, min(params['min_track_length'], 3))
            self.logger.info("Short sequence detected - reducing minimum track length")

        # Strategy-specific optimizations
        if strategy in [LinkingStrategy.TRACKPY_ADAPTIVE, LinkingStrategy.TRACKPY_HYBRID]:
            # For adaptive strategies, ensure reasonable stop criteria
            if data_analysis['particle_density'] > 0.02:
                params['adaptive_stop'] = min(params['adaptive_stop'], 0.05)
                params['adaptive_step'] = max(params['adaptive_step'], 0.97)

        return params

    def _analyze_data_characteristics(self, localizations: pd.DataFrame) -> Dict[str, float]:
        """Analyze data characteristics for parameter optimization."""

        n_frames = localizations['frame'].nunique()
        n_detections = len(localizations)

        # Calculate particle density (particles per frame per area)
        if 'x' in localizations.columns and 'y' in localizations.columns:
            x_range = localizations['x'].max() - localizations['x'].min()
            y_range = localizations['y'].max() - localizations['y'].min()
            area = x_range * y_range if x_range > 0 and y_range > 0 else 1

            particles_per_frame = n_detections / n_frames
            particle_density = particles_per_frame / area
        else:
            particle_density = 0

        # Calculate temporal sparsity
        frames = sorted(localizations['frame'].unique())
        if len(frames) > 1:
            frame_gaps = np.diff(frames)
            temporal_sparsity = np.mean(frame_gaps > 1)
        else:
            temporal_sparsity = 0

        return {
            'n_frames': n_frames,
            'n_detections': n_detections,
            'particles_per_frame': n_detections / n_frames,
            'particle_density': particle_density,
            'temporal_sparsity': temporal_sparsity
        }

    def suggest_parameters(self, localizations: pd.DataFrame) -> LinkingParameters:
        """Suggest optimal linking parameters based on data analysis."""

        data_analysis = self._analyze_data_characteristics(localizations)

        # Initialize with defaults
        suggested = LinkingParameters()

        # Suggest strategy based on data characteristics
        if data_analysis['particle_density'] > 0.02:
            # High density - use trackpy with adaptive features
            suggested.strategy = LinkingStrategy.TRACKPY_ADAPTIVE
            suggested.max_distance = 1.5
            suggested.memory = 1
            suggested.min_track_length = 8
        elif data_analysis['n_frames'] < 20:
            # Short sequences - use nearest neighbor
            suggested.strategy = LinkingStrategy.NEAREST_NEIGHBOR
            suggested.max_distance = 3.0
            suggested.memory = 2
            suggested.min_track_length = 3
        else:
            # Standard conditions - use hybrid trackpy
            suggested.strategy = LinkingStrategy.TRACKPY_HYBRID
            suggested.max_distance = 2.5
            suggested.memory = 2
            suggested.min_track_length = 5

        # Adjust for temporal sparsity
        if data_analysis['temporal_sparsity'] > 0.3:
            suggested.memory = min(suggested.memory + 1, 3)
            suggested.max_gap_frames = min(suggested.max_gap_frames + 1, 5)

        self.logger.info(f"Suggested strategy: {suggested.strategy.value}")

        return suggested

    def batch_link_files(self, file_paths: List[str],
                        output_dir: Optional[str] = None,
                        strategy: Optional[LinkingStrategy] = None,
                        **kwargs) -> Dict[str, LinkingQualityMetrics]:
        """Link particles in multiple files with consistent parameters."""

        results = {}

        for file_path in tqdm(file_paths, desc="Linking files"):
            try:
                # Load data
                localizations = pd.read_csv(file_path)

                # Link particles
                linked_data, metrics = self.link_particles(
                    localizations, strategy=strategy, **kwargs
                )

                # Save results
                if output_dir:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)

                    output_file = output_path / f"{Path(file_path).stem}_linked.csv"
                    linked_data.to_csv(output_file, index=False)

                results[file_path] = metrics

            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                continue

        return results

    def generate_linking_report(self, metrics: LinkingQualityMetrics,
                              output_path: Optional[str] = None) -> str:
        """Generate a detailed linking quality report."""

        report_lines = [
            "Particle Linking Quality Report",
            "=" * 40,
            "",
            f"Total detections: {metrics.total_detections:,}",
            f"Linked detections: {metrics.linked_detections:,}",
            f"Unlinked detections: {metrics.unlinked_detections:,}",
            f"Linking efficiency: {metrics.linking_efficiency:.1f}%",
            "",
            f"Total tracks: {metrics.total_tracks:,}",
            f"Average track length: {metrics.avg_track_length:.1f}",
            f"Median track length: {metrics.median_track_length:.1f}",
            f"Track length range: {metrics.min_track_length}-{metrics.max_track_length}",
            "",
            "Track length distribution:",
        ]

        # Add track length distribution
        for length, count in sorted(metrics.track_length_distribution.items()):
            report_lines.append(f"  {length} points: {count} tracks")

        report_text = "\n".join(report_lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Linking report saved to {output_path}")

        return report_text

    def update_parameters(self, parameters: Union[LinkingParameters, Dict[str, Any]]):
        """Update linking parameters."""
        if isinstance(parameters, dict):
            # Update only the specified parameters
            for key, value in parameters.items():
                if hasattr(self.parameters, key):
                    setattr(self.parameters, key, value)
        else:
            self.parameters = parameters
