#!/usr/bin/env python3
"""
Localization Precision Analysis Module
=====================================

Provides comprehensive analysis of localization precision and data quality metrics
for single-molecule tracking data. This module extends the basic localization error
analysis from the original scripts with advanced precision metrics.

Key Features:
- Track compactness and localization scatter analysis
- Precision-based track filtering and quality assessment
- Coordinate precision and uncertainty analysis
- Drift detection and correction assessment
- Sub-pixel precision analysis
- Localization clustering analysis
- Data quality metrics and recommendations

Based on the original Step_11_addLocalizationError.py but significantly enhanced.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any, Tuple
from scipy import stats
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class LocalizationPrecisionAnalyzer:
    """Analyzer for localization precision and data quality metrics."""
    
    def __init__(self, parameters=None):
        self.logger = logging.getLogger(__name__)
        self.parameters = parameters or {}
        
        # Default thresholds for quality assessment
        self.default_thresholds = {
            'max_localization_error': 50.0,  # nm
            'max_track_spread': 200.0,       # nm  
            'min_track_density': 0.5,        # localizations per frame
            'max_drift_rate': 10.0,          # nm per frame
            'min_precision_percentile': 90,   # % of tracks to consider high precision
            'clustering_eps': 20.0,          # nm for DBSCAN clustering
            'clustering_min_samples': 3      # minimum localizations per cluster
        }
        
    def analyze_localization_precision(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive localization precision analysis.
        
        Args:
            df: DataFrame with trajectory data containing at least:
                'track_number', 'frame', 'x', 'y' columns
                
        Returns:
            DataFrame with original data plus precision metrics
        """
        self.logger.info("Starting comprehensive localization precision analysis")
        
        # Validate input data
        required_cols = ['track_number', 'frame', 'x', 'y']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Need: {required_cols}")
            
        df = df.copy()
        
        # Basic localization error analysis (from original Step_11)
        df = self._add_basic_localization_error(df)
        
        # Advanced precision metrics
        df = self._add_track_compactness_metrics(df)
        df = self._add_coordinate_precision_metrics(df)
        df = self._add_temporal_precision_metrics(df)
        df = self._add_spatial_clustering_metrics(df)
        
        # Quality assessment and filtering recommendations
        df = self._add_quality_metrics(df)
        df = self._add_precision_classifications(df)
        
        # Drift analysis
        df = self._add_drift_metrics(df)
        
        self.logger.info("Localization precision analysis completed")
        return df
        
    def _add_basic_localization_error(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic localization error metrics (from original Step_11)."""
        
        # Track mean X,Y positions
        df['mean_X'] = df.groupby('track_number')['x'].transform('mean')
        df['mean_Y'] = df.groupby('track_number')['y'].transform('mean')
        
        # Euclidean distance from track centroid
        df['distance_from_centroid'] = np.sqrt(
            (df['mean_X'] - df['x'])**2 + (df['mean_Y'] - df['y'])**2
        )
        
        # Mean localization distance for each track
        df['mean_localization_error'] = df.groupby('track_number')['distance_from_centroid'].transform('mean')
        
        return df
        
    def _add_track_compactness_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add track compactness and spread metrics."""
        
        def calculate_track_compactness(track_data):
            """Calculate various compactness metrics for a track."""
            positions = track_data[['x', 'y']].values
            
            if len(positions) < 2:
                return pd.Series({
                    'track_spread_x': np.nan,
                    'track_spread_y': np.nan,
                    'track_spread_total': np.nan,
                    'track_area_convex_hull': np.nan,
                    'track_area_bounding_box': np.nan,
                    'track_circularity': np.nan,
                    'track_eccentricity': np.nan,
                    'localization_density': np.nan
                })
                
            # Spread in X and Y directions
            spread_x = np.max(positions[:, 0]) - np.min(positions[:, 0])
            spread_y = np.max(positions[:, 1]) - np.min(positions[:, 1])
            spread_total = np.sqrt(spread_x**2 + spread_y**2)
            
            # Convex hull area (requires at least 3 points)
            if len(positions) >= 3:
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(positions)
                    convex_hull_area = hull.volume  # In 2D, volume = area
                except Exception:
                    convex_hull_area = np.nan
            else:
                convex_hull_area = 0.0
                
            # Bounding box area
            bounding_box_area = spread_x * spread_y
            
            # Circularity (how circular is the track)
            if convex_hull_area > 0:
                # Perimeter of convex hull
                try:
                    hull_perimeter = np.sum([
                        np.linalg.norm(positions[hull.vertices[i]] - positions[hull.vertices[i-1]])
                        for i in range(len(hull.vertices))
                    ])
                    circularity = 4 * np.pi * convex_hull_area / (hull_perimeter**2)
                except Exception:
                    circularity = np.nan
            else:
                circularity = np.nan
                
            # Eccentricity (elongation measure)
            try:
                # Calculate covariance matrix
                cov_matrix = np.cov(positions.T)
                eigenvals = np.linalg.eigvals(cov_matrix)
                eigenvals = np.sort(eigenvals)[::-1]  # Sort descending
                
                if eigenvals[1] > 0:
                    eccentricity = np.sqrt(1 - eigenvals[1]/eigenvals[0])
                else:
                    eccentricity = 1.0  # Maximum eccentricity
            except Exception:
                eccentricity = np.nan
                
            # Localization density (localizations per unit area)
            if convex_hull_area > 0:
                localization_density = len(positions) / convex_hull_area
            else:
                localization_density = np.inf if len(positions) > 0 else 0
                
            return pd.Series({
                'track_spread_x': spread_x,
                'track_spread_y': spread_y, 
                'track_spread_total': spread_total,
                'track_area_convex_hull': convex_hull_area,
                'track_area_bounding_box': bounding_box_area,
                'track_circularity': circularity,
                'track_eccentricity': eccentricity,
                'localization_density': localization_density
            })
            
        # Calculate compactness metrics for each track
        compactness_metrics = df.groupby('track_number').apply(calculate_track_compactness)
        
        # Join back to original dataframe
        for col in compactness_metrics.columns:
            df = df.join(compactness_metrics[col], on='track_number', rsuffix='_temp')
            
        return df
        
    def _add_coordinate_precision_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add coordinate-specific precision metrics."""
        
        def calculate_coordinate_precision(track_data):
            """Calculate precision metrics for X and Y coordinates separately."""
            if len(track_data) < 3:
                return pd.Series({
                    'x_precision_std': np.nan,
                    'y_precision_std': np.nan,
                    'x_precision_mad': np.nan,
                    'y_precision_mad': np.nan,
                    'xy_correlation': np.nan,
                    'coordinate_drift_x': np.nan,
                    'coordinate_drift_y': np.nan
                })
                
            x_vals = track_data['x'].values
            y_vals = track_data['y'].values
            
            # Standard deviation of coordinates
            x_std = np.std(x_vals)
            y_std = np.std(y_vals)
            
            # Median absolute deviation (more robust to outliers)
            x_mad = stats.median_abs_deviation(x_vals)
            y_mad = stats.median_abs_deviation(y_vals)
            
            # Correlation between X and Y coordinates
            try:
                xy_corr = np.corrcoef(x_vals, y_vals)[0, 1]
            except Exception:
                xy_corr = np.nan
                
            # Linear drift in coordinates over time
            frames = track_data['frame'].values
            if len(np.unique(frames)) > 2:
                try:
                    # Linear regression slope indicates drift rate
                    x_drift, _, _, _, _ = stats.linregress(frames, x_vals)
                    y_drift, _, _, _, _ = stats.linregress(frames, y_vals)
                except Exception:
                    x_drift = np.nan
                    y_drift = np.nan
            else:
                x_drift = np.nan
                y_drift = np.nan
                
            return pd.Series({
                'x_precision_std': x_std,
                'y_precision_std': y_std,
                'x_precision_mad': x_mad,
                'y_precision_mad': y_mad,
                'xy_correlation': xy_corr,
                'coordinate_drift_x': x_drift,
                'coordinate_drift_y': y_drift
            })
            
        # Calculate coordinate precision for each track
        coord_metrics = df.groupby('track_number').apply(calculate_coordinate_precision)
        
        # Join back to original dataframe
        for col in coord_metrics.columns:
            df = df.join(coord_metrics[col], on='track_number', rsuffix='_temp')
            
        return df
        
    def _add_temporal_precision_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal precision and consistency metrics."""
        
        def calculate_temporal_metrics(track_data):
            """Calculate temporal precision metrics."""
            track_data = track_data.sort_values('frame')
            
            if len(track_data) < 3:
                return pd.Series({
                    'temporal_gaps': 0,
                    'gap_fraction': 0.0,
                    'temporal_consistency': np.nan,
                    'frame_density': np.nan
                })
                
            frames = track_data['frame'].values
            
            # Count gaps in temporal sequence
            expected_frames = np.arange(frames.min(), frames.max() + 1)
            missing_frames = set(expected_frames) - set(frames)
            temporal_gaps = len(missing_frames)
            
            # Fraction of frames with gaps
            total_expected = len(expected_frames)
            gap_fraction = temporal_gaps / total_expected if total_expected > 0 else 0.0
            
            # Temporal consistency (regularity of frame intervals)
            frame_intervals = np.diff(frames)
            if len(frame_intervals) > 1:
                temporal_consistency = 1.0 / (1.0 + np.std(frame_intervals))
            else:
                temporal_consistency = 1.0
                
            # Frame density (actual frames / expected frames)
            frame_density = len(frames) / total_expected if total_expected > 0 else 0.0
            
            return pd.Series({
                'temporal_gaps': temporal_gaps,
                'gap_fraction': gap_fraction,
                'temporal_consistency': temporal_consistency,
                'frame_density': frame_density
            })
            
        # Calculate temporal metrics for each track
        temporal_metrics = df.groupby('track_number').apply(calculate_temporal_metrics)
        
        # Join back to original dataframe
        for col in temporal_metrics.columns:
            df = df.join(temporal_metrics[col], on='track_number', rsuffix='_temp')
            
        return df
        
    def _add_spatial_clustering_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add spatial clustering analysis for localizations within tracks."""
        
        def analyze_spatial_clustering(track_data):
            """Analyze spatial clustering within a track using DBSCAN."""
            positions = track_data[['x', 'y']].values
            
            if len(positions) < 3:
                return pd.Series({
                    'n_spatial_clusters': 0,
                    'cluster_sizes_mean': np.nan,
                    'cluster_sizes_std': np.nan,
                    'clustered_fraction': 0.0,
                    'largest_cluster_size': 0
                })
                
            # DBSCAN clustering
            eps = self.default_thresholds['clustering_eps']
            min_samples = min(self.default_thresholds['clustering_min_samples'], len(positions) // 3)
            
            try:
                clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
                labels = clustering.labels_
                
                # Number of clusters (excluding noise points labeled as -1)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                # Cluster sizes (excluding noise)
                cluster_sizes = []
                for cluster_id in set(labels):
                    if cluster_id != -1:  # Exclude noise
                        cluster_sizes.append(np.sum(labels == cluster_id))
                        
                if cluster_sizes:
                    cluster_sizes_mean = np.mean(cluster_sizes)
                    cluster_sizes_std = np.std(cluster_sizes)
                    largest_cluster_size = max(cluster_sizes)
                    clustered_fraction = sum(cluster_sizes) / len(positions)
                else:
                    cluster_sizes_mean = np.nan
                    cluster_sizes_std = np.nan
                    largest_cluster_size = 0
                    clustered_fraction = 0.0
                    
            except Exception:
                n_clusters = 0
                cluster_sizes_mean = np.nan
                cluster_sizes_std = np.nan
                clustered_fraction = 0.0
                largest_cluster_size = 0
                
            return pd.Series({
                'n_spatial_clusters': n_clusters,
                'cluster_sizes_mean': cluster_sizes_mean,
                'cluster_sizes_std': cluster_sizes_std,
                'clustered_fraction': clustered_fraction,
                'largest_cluster_size': largest_cluster_size
            })
            
        # Calculate clustering metrics for each track
        clustering_metrics = df.groupby('track_number').apply(analyze_spatial_clustering)
        
        # Join back to original dataframe
        for col in clustering_metrics.columns:
            df = df.join(clustering_metrics[col], on='track_number', rsuffix='_temp')
            
        return df
        
    def _add_drift_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add global drift detection metrics."""
        
        # Calculate frame-by-frame centroid positions
        frame_centroids = df.groupby('frame')[['x', 'y']].mean().reset_index()
        frame_centroids = frame_centroids.sort_values('frame')
        
        if len(frame_centroids) < 3:
            df['global_drift_x'] = 0.0
            df['global_drift_y'] = 0.0
            df['global_drift_magnitude'] = 0.0
            return df
            
        # Calculate drift as linear trend in centroid positions
        frames = frame_centroids['frame'].values
        x_centroids = frame_centroids['x'].values
        y_centroids = frame_centroids['y'].values
        
        try:
            # Linear regression to detect drift
            x_drift_rate, _, _, _, _ = stats.linregress(frames, x_centroids)
            y_drift_rate, _, _, _, _ = stats.linregress(frames, y_centroids)
            drift_magnitude = np.sqrt(x_drift_rate**2 + y_drift_rate**2)
        except:
            x_drift_rate = 0.0
            y_drift_rate = 0.0
            drift_magnitude = 0.0
            
        # Add drift metrics to all rows
        df['global_drift_x'] = x_drift_rate
        df['global_drift_y'] = y_drift_rate
        df['global_drift_magnitude'] = drift_magnitude
        
        return df
        
    def _add_quality_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add overall quality metrics and scores."""
        
        def calculate_quality_score(track_data):
            """Calculate composite quality score for a track."""
            
            # Get first row values (same for all rows in track)
            first_row = track_data.iloc[0]
            
            # Quality components (normalized to 0-1 scale, higher = better quality)
            components = {}
            
            # 1. Localization precision (lower error = higher quality)
            max_error = self.default_thresholds['max_localization_error']
            error = first_row.get('mean_localization_error', max_error)
            components['precision'] = max(0, 1 - error / max_error)
            
            # 2. Temporal completeness (higher frame density = higher quality)
            frame_density = first_row.get('frame_density', 0.0)
            components['completeness'] = min(1.0, frame_density)
            
            # 3. Spatial compactness (lower spread = higher quality for stationary particles)
            max_spread = self.default_thresholds['max_track_spread']
            spread = first_row.get('track_spread_total', max_spread)
            components['compactness'] = max(0, 1 - spread / max_spread)
            
            # 4. Temporal consistency (more regular = higher quality)
            consistency = first_row.get('temporal_consistency', 0.0)
            components['consistency'] = consistency
            
            # 5. Low drift (less coordinate drift = higher quality)
            max_drift = self.default_thresholds['max_drift_rate']
            x_drift = abs(first_row.get('coordinate_drift_x', 0.0))
            y_drift = abs(first_row.get('coordinate_drift_y', 0.0))
            drift_magnitude = np.sqrt(x_drift**2 + y_drift**2)
            components['stability'] = max(0, 1 - drift_magnitude / max_drift)
            
            # Weighted composite score
            weights = {
                'precision': 0.3,
                'completeness': 0.2,
                'compactness': 0.2,
                'consistency': 0.15,
                'stability': 0.15
            }
            
            quality_score = sum(components[key] * weights[key] for key in components)
            
            return pd.Series({
                'quality_score': quality_score,
                'quality_precision': components['precision'],
                'quality_completeness': components['completeness'],
                'quality_compactness': components['compactness'],
                'quality_consistency': components['consistency'],
                'quality_stability': components['stability']
            })
            
        # Calculate quality metrics for each track
        quality_metrics = df.groupby('track_number').apply(calculate_quality_score)
        
        # Join back to original dataframe
        for col in quality_metrics.columns:
            df = df.join(quality_metrics[col], on='track_number', rsuffix='_temp')
            
        return df
        
    def _add_precision_classifications(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add precision-based classifications and filtering recommendations."""
        
        # Calculate percentile thresholds for classification
        if 'quality_score' in df.columns:
            unique_tracks = df.groupby('track_number')['quality_score'].first()
            
            # Define quality categories based on percentiles
            high_quality_threshold = np.percentile(unique_tracks.dropna(), 80)
            medium_quality_threshold = np.percentile(unique_tracks.dropna(), 50)
            
            def classify_precision(quality_score):
                if pd.isna(quality_score):
                    return 'unknown'
                elif quality_score >= high_quality_threshold:
                    return 'high_precision'
                elif quality_score >= medium_quality_threshold:
                    return 'medium_precision'
                else:
                    return 'low_precision'
                    
            df['precision_classification'] = df['quality_score'].apply(classify_precision)
            
        # Filtering recommendations
        def get_filter_recommendation(row):
            """Recommend whether to include track in analysis."""
            reasons = []
            
            # Check localization error
            if row.get('mean_localization_error', 0) > self.default_thresholds['max_localization_error']:
                reasons.append('high_localization_error')
                
            # Check temporal completeness
            if row.get('frame_density', 1.0) < 0.5:
                reasons.append('low_temporal_density')
                
            # Check drift
            if row.get('global_drift_magnitude', 0) > self.default_thresholds['max_drift_rate']:
                reasons.append('high_drift')
                
            # Check quality score
            if row.get('quality_score', 1.0) < 0.3:
                reasons.append('low_quality_score')
                
            if reasons:
                return f"exclude_{'_'.join(reasons)}"
            else:
                return 'include'
                
        df['filter_recommendation'] = df.apply(get_filter_recommendation, axis=1)
        
        return df
        
    def get_precision_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for localization precision analysis."""
        
        # Get unique tracks data
        track_data = df.groupby('track_number').first()
        
        summary = {
            'total_tracks': len(track_data),
            'total_localizations': len(df)
        }
        
        # Quality distribution
        if 'precision_classification' in track_data.columns:
            quality_dist = track_data['precision_classification'].value_counts()
            summary['quality_distribution'] = quality_dist.to_dict()
            
        # Filter recommendations
        if 'filter_recommendation' in track_data.columns:
            filter_dist = track_data['filter_recommendation'].value_counts()
            summary['filter_recommendations'] = filter_dist.to_dict()
            
        # Precision metrics statistics
        precision_cols = [
            'mean_localization_error', 'quality_score', 'track_spread_total',
            'temporal_consistency', 'frame_density'
        ]
        
        for col in precision_cols:
            if col in track_data.columns:
                values = track_data[col].dropna()
                if len(values) > 0:
                    summary[f'{col}_mean'] = float(values.mean())
                    summary[f'{col}_std'] = float(values.std())
                    summary[f'{col}_median'] = float(values.median())
                    
        # Drift analysis
        if 'global_drift_magnitude' in df.columns:
            drift_mag = df['global_drift_magnitude'].iloc[0]  # Same for all rows
            summary['global_drift_magnitude'] = float(drift_mag)
            
        return summary
        
    def export_precision_report(self, df: pd.DataFrame, output_path: str) -> bool:
        """Export comprehensive precision analysis report."""
        
        try:
            summary = self.get_precision_summary(df)
            
            # Create report
            report_lines = [
                "Localization Precision Analysis Report",
                "=" * 50,
                "",
                f"Total tracks analyzed: {summary['total_tracks']}",
                f"Total localizations: {summary['total_localizations']}",
                "",
                "Quality Distribution:",
                "-" * 25
            ]
            
            if 'quality_distribution' in summary:
                for quality, count in summary['quality_distribution'].items():
                    pct = (count / summary['total_tracks']) * 100
                    report_lines.append(f"{quality}: {count} tracks ({pct:.1f}%)")
                    
            report_lines.extend([
                "",
                "Filter Recommendations:",
                "-" * 25
            ])
            
            if 'filter_recommendations' in summary:
                for recommendation, count in summary['filter_recommendations'].items():
                    pct = (count / summary['total_tracks']) * 100
                    report_lines.append(f"{recommendation}: {count} tracks ({pct:.1f}%)")
                    
            report_lines.extend([
                "",
                "Precision Statistics:",
                "-" * 20
            ])
            
            # Add precision statistics
            precision_stats = [
                ('Mean Localization Error', 'mean_localization_error_mean', 'nm'),
                ('Quality Score', 'quality_score_mean', ''),
                ('Track Spread', 'track_spread_total_mean', 'nm'),
                ('Temporal Consistency', 'temporal_consistency_mean', ''),
                ('Frame Density', 'frame_density_mean', '')
            ]
            
            for name, key, unit in precision_stats:
                if key in summary:
                    value = summary[key]
                    report_lines.append(f"{name}: {value:.3f} {unit}")
                    
            if 'global_drift_magnitude' in summary:
                report_lines.extend([
                    "",
                    f"Global drift magnitude: {summary['global_drift_magnitude']:.3f} nm/frame"
                ])
                
            # Write report
            with open(output_path, 'w') as f:
                f.write('\n'.join(report_lines))
                
            self.logger.info(f"Precision analysis report exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting precision report: {e}")
            return False


def analyze_localization_precision(df: pd.DataFrame, parameters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Convenience function for localization precision analysis.
    
    Args:
        df: DataFrame with trajectory data
        parameters: Optional analysis parameters
        
    Returns:
        DataFrame with precision metrics added
    """
    analyzer = LocalizationPrecisionAnalyzer(parameters)
    return analyzer.analyze_localization_precision(df)


def filter_tracks_by_precision(df: pd.DataFrame, quality_threshold: float = 0.5,
                              precision_class: str = 'medium_precision') -> pd.DataFrame:
    """
    Filter tracks based on precision criteria.
    
    Args:
        df: DataFrame with precision metrics
        quality_threshold: Minimum quality score (0-1)
        precision_class: Minimum precision class to include
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    # Filter by quality score
    if 'quality_score' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['quality_score'] >= quality_threshold]
        
    # Filter by precision classification
    if 'precision_classification' in filtered_df.columns:
        if precision_class == 'high_precision':
            filtered_df = filtered_df[filtered_df['precision_classification'] == 'high_precision']
        elif precision_class == 'medium_precision':
            filtered_df = filtered_df[filtered_df['precision_classification'].isin(['high_precision', 'medium_precision'])]
        # 'low_precision' includes all tracks
        
    return filtered_df


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    
    # Generate sample trajectory data
    n_tracks = 10
    n_frames = 50
    
    sample_data = []
    for track_id in range(n_tracks):
        # Random walk with some noise
        x_center = np.random.uniform(100, 500)
        y_center = np.random.uniform(100, 500)
        
        for frame in range(n_frames):
            # Add some missing frames randomly
            if np.random.random() > 0.1:  # 90% frame presence
                x = x_center + np.random.normal(0, 20)  # 20nm precision
                y = y_center + np.random.normal(0, 20)
                
                sample_data.append({
                    'track_number': track_id,
                    'frame': frame,
                    'x': x,
                    'y': y
                })
                
    df = pd.DataFrame(sample_data)
    
    # Run analysis
    analyzer = LocalizationPrecisionAnalyzer()
    result_df = analyzer.analyze_localization_precision(df)
    
    # Print summary
    summary = analyzer.get_precision_summary(result_df)
    print("Localization Precision Analysis Summary:")
    print(f"Total tracks: {summary['total_tracks']}")
    print(f"Quality score mean: {summary.get('quality_score_mean', 'N/A'):.3f}")
    print(f"Mean localization error: {summary.get('mean_localization_error_mean', 'N/A'):.1f} nm")
