#!/usr/bin/env python3
"""
Multi-Radius Density Analysis Module
====================================

Comprehensive density analysis for particle tracking data including:
- Multi-radius neighbor counting (3, 5, 10, 20, 30 pixel radii)
- Nearest neighbor distance calculations
- Density visualization and statistics
- Support for different coordinate systems and units

Based on Step_8_addNNcounts.py and Step_3_nearestNeighbour.py from the
original analysis scripts, enhanced for integration with the particle
tracking application.
"""

import logging
import warnings
from typing import Optional, Dict, List, Any, Tuple, Union
import time

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree, NearestNeighbors
from scipy import stats, spatial
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.simplefilter(action='ignore', category=Warning)


class DensityAnalyzer:
    """Comprehensive density analysis for particle tracking data."""

    def __init__(self, parameters=None):
        """Initialize the density analyzer.
        
        Args:
            parameters: Optional dictionary of analysis parameters
        """
        self.logger = logging.getLogger(__name__)
        self.parameters = parameters or {}
        
        # Default analysis parameters
        self.default_radii = [3, 5, 10, 20, 30]  # pixels
        self.default_unit = "pixels"
        self.default_pixel_size = 108.0  # nm per pixel

    def calculate_nearest_neighbors(self, df: pd.DataFrame, 
                                  coordinate_columns: Tuple[str, str] = ('x', 'y'),
                                  frame_column: str = 'frame',
                                  unit: str = 'pixels') -> pd.DataFrame:
        """Calculate nearest neighbor distances for each particle in each frame.
        
        Based on Step_3_nearestNeighbour.py functionality.
        
        Args:
            df: DataFrame with particle coordinates
            coordinate_columns: Tuple of (x_column, y_column) names
            frame_column: Name of frame column
            unit: Unit for distances ('pixels' or 'nm')
            
        Returns:
            DataFrame with added nearest neighbor distance column
        """
        self.logger.info("Calculating nearest neighbor distances...")
        start_time = time.time()
        
        df = df.copy()
        x_col, y_col = coordinate_columns
        
        # Validate required columns
        required_cols = [x_col, y_col, frame_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Sort by frame for efficient processing
        df = df.sort_values(by=[frame_column])
        
        # Initialize results list
        nn_distances = []
        nn_indices = []
        
        # Get unique frames
        frames = df[frame_column].unique()
        total_frames = len(frames)
        
        self.logger.info(f"Processing {total_frames} frames for nearest neighbor analysis")
        
        # Process each frame
        for i, frame in enumerate(frames):
            if i % 50 == 0 or i == total_frames - 1:
                self.logger.debug(f"Processing frame {i+1}/{total_frames}")
            
            # Filter data for current frame
            frame_data = df[df[frame_column] == frame]
            
            if len(frame_data) < 2:
                # Not enough points for nearest neighbor calculation
                nn_distances.extend([np.nan] * len(frame_data))
                nn_indices.extend([np.nan] * len(frame_data))
                continue
            
            # Extract coordinates
            coordinates = frame_data[[x_col, y_col]].values
            
            # Calculate nearest neighbors
            distances, indices = self._get_nearest_neighbors(coordinates, k=2)
            
            if np.isnan(distances).any():
                nn_distances.extend([np.nan] * len(frame_data))
                nn_indices.extend([np.nan] * len(frame_data))
            else:
                # Take second closest (first is self)
                nn_distances.extend(distances[:, 1])
                nn_indices.extend(indices[:, 1])
        
        # Add results to dataframe
        df['nn_distance'] = nn_distances
        df['nn_index_in_frame'] = nn_indices
        
        # Convert units if necessary
        if unit == 'nm' and 'nn_distance' in df.columns:
            pixel_size = self.parameters.get('pixel_size', self.default_pixel_size)
            df['nn_distance_nm'] = df['nn_distance'] * pixel_size
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Nearest neighbor analysis completed in {elapsed_time:.2f} seconds")
        
        return df

    def calculate_multi_radius_density(self, df: pd.DataFrame,
                                     radii: List[float] = None,
                                     coordinate_columns: Tuple[str, str] = ('x', 'y'),
                                     frame_column: str = 'frame',
                                     unit: str = 'pixels',
                                     pixel_size: float = None) -> pd.DataFrame:
        """Calculate neighbor counts within multiple radii for each particle.
        
        Based on Step_8_addNNcounts.py functionality.
        
        Args:
            df: DataFrame with particle coordinates
            radii: List of radii to analyze (default: [3, 5, 10, 20, 30])
            coordinate_columns: Tuple of (x_column, y_column) names
            frame_column: Name of frame column
            unit: Unit for radii ('pixels' or 'nm')
            pixel_size: Pixel size in nm (for unit conversion)
            
        Returns:
            DataFrame with added neighbor count columns
        """
        if radii is None:
            radii = self.default_radii.copy()
        
        if pixel_size is None:
            pixel_size = self.parameters.get('pixel_size', self.default_pixel_size)
        
        self.logger.info(f"Calculating multi-radius density analysis for radii: {radii} {unit}")
        start_time = time.time()
        
        df = df.copy()
        x_col, y_col = coordinate_columns
        
        # Validate required columns
        required_cols = [x_col, y_col, frame_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert radii to pixels if necessary
        if unit == 'nm':
            radii_pixels = [r / pixel_size for r in radii]
            self.logger.info(f"Converted radii from nm to pixels: {radii} nm -> {radii_pixels} pixels")
        else:
            radii_pixels = radii
        
        # Sort by frame for efficient processing
        df = df.sort_values(by=[frame_column])
        
        # Initialize result columns
        for r in radii:
            col_name = f'nn_count_within_{r}_{unit}'
            df[col_name] = np.nan
        
        # Get unique frames
        frames = df[frame_column].unique()
        total_frames = len(frames)
        
        self.logger.info(f"Processing {total_frames} frames for multi-radius density analysis")
        
        # Process each radius
        for radius_idx, radius_pixels in enumerate(radii_pixels):
            radius_original = radii[radius_idx]
            col_name = f'nn_count_within_{radius_original}_{unit}'
            
            self.logger.debug(f"Processing radius {radius_original} {unit} ({radius_pixels} pixels)")
            
            # Process each frame for this radius
            for i, frame in enumerate(frames):
                if i % 100 == 0 or i == total_frames - 1:
                    progress = f"Radius {radius_idx+1}/{len(radii)}, Frame {i+1}/{total_frames}"
                    self.logger.debug(f"Progress: {progress}")
                
                # Filter data for current frame
                frame_data = df[df[frame_column] == frame]
                frame_indices = frame_data.index
                
                if len(frame_data) < 2:
                    # Not enough points for neighbor counting
                    df.loc[frame_indices, col_name] = 0
                    continue
                
                # Extract coordinates
                coordinates = frame_data[[x_col, y_col]].values
                
                # Count neighbors within radius
                neighbor_counts = self._count_neighbors_within_radius(
                    coordinates, radius_pixels
                )
                
                # Subtract 1 to exclude self
                neighbor_counts = neighbor_counts - 1
                
                # Store results
                df.loc[frame_indices, col_name] = neighbor_counts
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Multi-radius density analysis completed in {elapsed_time:.2f} seconds")
        
        return df

    def _get_nearest_neighbors(self, coordinates: np.ndarray, k: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate nearest neighbors for a set of coordinates.
        
        Args:
            coordinates: Nx2 array of (x, y) coordinates
            k: Number of nearest neighbors to find
            
        Returns:
            Tuple of (distances, indices) arrays
        """
        try:
            tree = KDTree(coordinates, leaf_size=5)
            
            if k > len(coordinates):
                # Not enough neighbors available
                distances = np.full((len(coordinates), k), np.nan)
                indices = np.full((len(coordinates), k), np.nan)
                return distances, indices
            
            distances, indices = tree.query(coordinates, k=k)
            return distances, indices
            
        except Exception as e:
            self.logger.error(f"Error in nearest neighbor calculation: {e}")
            distances = np.full((len(coordinates), k), np.nan)
            indices = np.full((len(coordinates), k), np.nan)
            return distances, indices

    def _count_neighbors_within_radius(self, coordinates: np.ndarray, radius: float) -> np.ndarray:
        """Count neighbors within a specified radius for each point.
        
        Args:
            coordinates: Nx2 array of (x, y) coordinates
            radius: Search radius
            
        Returns:
            Array of neighbor counts
        """
        try:
            tree = KDTree(coordinates, leaf_size=5)
            neighbor_counts = tree.query_radius(coordinates, r=radius, count_only=True)
            return neighbor_counts
            
        except Exception as e:
            self.logger.error(f"Error in radius neighbor counting: {e}")
            return np.zeros(len(coordinates))

    def calculate_density_statistics(self, df: pd.DataFrame, 
                                   radii: List[float] = None,
                                   unit: str = 'pixels',
                                   group_by: str = None) -> Dict[str, Any]:
        """Calculate density statistics across different radii.
        
        Args:
            df: DataFrame with density analysis results
            radii: List of radii that were analyzed
            unit: Unit of radii
            group_by: Optional column to group statistics by (e.g., 'condition', 'track_number')
            
        Returns:
            Dictionary with density statistics
        """
        if radii is None:
            radii = self.default_radii
        
        self.logger.info("Calculating density statistics")
        
        stats_dict = {}
        
        # Find density columns
        density_columns = []
        for r in radii:
            col_name = f'nn_count_within_{r}_{unit}'
            if col_name in df.columns:
                density_columns.append(col_name)
        
        if not density_columns:
            self.logger.warning("No density columns found for statistics calculation")
            return stats_dict
        
        if group_by and group_by in df.columns:
            # Group statistics
            grouped = df.groupby(group_by)
            
            for group_name, group_data in grouped:
                group_stats = {}
                
                for col in density_columns:
                    values = group_data[col].dropna()
                    if len(values) > 0:
                        group_stats[col] = {
                            'mean': values.mean(),
                            'std': values.std(),
                            'median': values.median(),
                            'min': values.min(),
                            'max': values.max(),
                            'count': len(values)
                        }
                
                stats_dict[f'group_{group_name}'] = group_stats
        
        else:
            # Overall statistics
            overall_stats = {}
            
            for col in density_columns:
                values = df[col].dropna()
                if len(values) > 0:
                    overall_stats[col] = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'median': values.median(),
                        'min': values.min(),
                        'max': values.max(),
                        'count': len(values)
                    }
            
            stats_dict['overall'] = overall_stats
        
        # Add nearest neighbor statistics if available
        if 'nn_distance' in df.columns:
            nn_values = df['nn_distance'].dropna()
            if len(nn_values) > 0:
                nn_stats = {
                    'mean': nn_values.mean(),
                    'std': nn_values.std(),
                    'median': nn_values.median(),
                    'min': nn_values.min(),
                    'max': nn_values.max(),
                    'count': len(nn_values)
                }
                
                if group_by and group_by in df.columns:
                    # Add NN stats for each group
                    grouped = df.groupby(group_by)
                    for group_name, group_data in grouped:
                        group_nn = group_data['nn_distance'].dropna()
                        if len(group_nn) > 0:
                            if f'group_{group_name}' not in stats_dict:
                                stats_dict[f'group_{group_name}'] = {}
                            stats_dict[f'group_{group_name}']['nn_distance'] = {
                                'mean': group_nn.mean(),
                                'std': group_nn.std(),
                                'median': group_nn.median(),
                                'min': group_nn.min(),
                                'max': group_nn.max(),
                                'count': len(group_nn)
                            }
                else:
                    stats_dict['overall']['nn_distance'] = nn_stats
        
        return stats_dict

    def create_density_plots(self, df: pd.DataFrame, 
                           radii: List[float] = None,
                           unit: str = 'pixels',
                           output_path: str = None,
                           group_by: str = None) -> Dict[str, str]:
        """Create density visualization plots.
        
        Args:
            df: DataFrame with density analysis results
            radii: List of radii that were analyzed
            unit: Unit of radii
            output_path: Optional path to save plots
            group_by: Optional column to group plots by
            
        Returns:
            Dictionary mapping plot types to file paths (if saved)
        """
        if radii is None:
            radii = self.default_radii
        
        self.logger.info("Creating density visualization plots")
        
        plot_paths = {}
        
        # Find density columns
        density_columns = []
        for r in radii:
            col_name = f'nn_count_within_{r}_{unit}'
            if col_name in df.columns:
                density_columns.append(col_name)
        
        if not density_columns:
            self.logger.warning("No density columns found for plotting")
            return plot_paths
        
        # 1. Histogram of neighbor counts for each radius
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(density_columns[:6]):  # Limit to 6 plots
            if i < len(axes):
                ax = axes[i]
                values = df[col].dropna()
                
                if len(values) > 0:
                    # Extract radius from column name
                    radius_str = col.split('_')[3]
                    
                    if group_by and group_by in df.columns:
                        # Grouped histogram
                        for group_name in df[group_by].unique():
                            group_values = df[df[group_by] == group_name][col].dropna()
                            if len(group_values) > 0:
                                ax.hist(group_values, alpha=0.7, label=str(group_name), bins=20)
                        ax.legend()
                    else:
                        ax.hist(values, bins=20, alpha=0.7, color='skyblue')
                    
                    ax.set_xlabel(f'Neighbor Count (r={radius_str} {unit})')
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'Distribution of Neighbor Counts\n(Radius = {radius_str} {unit})')
                    ax.grid(True, alpha=0.3)
        
        # Remove unused subplots
        for i in range(len(density_columns), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if output_path:
            hist_path = f"{output_path}_density_histograms.png"
            plt.savefig(hist_path, dpi=300, bbox_inches='tight')
            plot_paths['histograms'] = hist_path
            self.logger.info(f"Saved density histograms to {hist_path}")
        
        plt.close()
        
        # 2. Mean neighbor count vs radius
        plt.figure(figsize=(10, 6))
        
        mean_counts = []
        std_counts = []
        radius_values = []
        
        for col in density_columns:
            values = df[col].dropna()
            if len(values) > 0:
                mean_counts.append(values.mean())
                std_counts.append(values.std())
                # Extract radius from column name
                radius_str = col.split('_')[3]
                radius_values.append(float(radius_str))
        
        if mean_counts:
            if group_by and group_by in df.columns:
                # Plot for each group
                for group_name in df[group_by].unique():
                    group_means = []
                    group_stds = []
                    
                    for col in density_columns:
                        group_values = df[df[group_by] == group_name][col].dropna()
                        if len(group_values) > 0:
                            group_means.append(group_values.mean())
                            group_stds.append(group_values.std())
                        else:
                            group_means.append(0)
                            group_stds.append(0)
                    
                    plt.errorbar(radius_values, group_means, yerr=group_stds, 
                               marker='o', label=str(group_name), capsize=5)
                
                plt.legend()
            else:
                plt.errorbar(radius_values, mean_counts, yerr=std_counts, 
                           marker='o', color='blue', capsize=5)
            
            plt.xlabel(f'Radius ({unit})')
            plt.ylabel('Mean Neighbor Count')
            plt.title('Mean Neighbor Count vs Search Radius')
            plt.grid(True, alpha=0.3)
            plt.xlim(0, max(radius_values) * 1.1)
            plt.ylim(0, None)
        
        if output_path:
            radius_path = f"{output_path}_neighbor_vs_radius.png"
            plt.savefig(radius_path, dpi=300, bbox_inches='tight')
            plot_paths['radius_plot'] = radius_path
            self.logger.info(f"Saved radius plot to {radius_path}")
        
        plt.close()
        
        # 3. Nearest neighbor distance distribution
        if 'nn_distance' in df.columns:
            plt.figure(figsize=(10, 6))
            
            nn_values = df['nn_distance'].dropna()
            if len(nn_values) > 0:
                if group_by and group_by in df.columns:
                    for group_name in df[group_by].unique():
                        group_nn = df[df[group_by] == group_name]['nn_distance'].dropna()
                        if len(group_nn) > 0:
                            plt.hist(group_nn, alpha=0.7, label=str(group_name), 
                                   bins=30, density=True)
                    plt.legend()
                else:
                    plt.hist(nn_values, bins=30, alpha=0.7, color='lightcoral', density=True)
                
                plt.xlabel(f'Nearest Neighbor Distance ({unit})')
                plt.ylabel('Density')
                plt.title('Distribution of Nearest Neighbor Distances')
                plt.grid(True, alpha=0.3)
            
            if output_path:
                nn_path = f"{output_path}_nn_distance_distribution.png"
                plt.savefig(nn_path, dpi=300, bbox_inches='tight')
                plot_paths['nn_distribution'] = nn_path
                self.logger.info(f"Saved NN distribution plot to {nn_path}")
            
            plt.close()
        
        # 4. Correlation heatmap between different radii
        if len(density_columns) > 1:
            plt.figure(figsize=(8, 6))
            
            # Create correlation matrix
            corr_data = df[density_columns].corr()
            
            # Create labels with just radius values
            labels = [col.split('_')[3] for col in density_columns]
            
            # Plot heatmap
            sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
                       xticklabels=labels, yticklabels=labels,
                       square=True, linewidths=0.5)
            
            plt.title(f'Correlation Between Neighbor Counts\n(Different Radii in {unit})')
            plt.tight_layout()
            
            if output_path:
                corr_path = f"{output_path}_density_correlation.png"
                plt.savefig(corr_path, dpi=300, bbox_inches='tight')
                plot_paths['correlation'] = corr_path
                self.logger.info(f"Saved correlation heatmap to {corr_path}")
            
            plt.close()
        
        return plot_paths

    def create_density_heatmap(self, df: pd.DataFrame,
                             coordinate_columns: Tuple[str, str] = ('x', 'y'),
                             frame_column: str = 'frame',
                             radius: float = 10,
                             unit: str = 'pixels',
                             grid_size: int = 50,
                             output_path: str = None) -> Optional[str]:
        """Create a spatial density heatmap for a specific radius.
        
        Args:
            df: DataFrame with density analysis results
            coordinate_columns: Tuple of (x_column, y_column) names
            frame_column: Name of frame column
            radius: Radius for density calculation
            unit: Unit of radius
            grid_size: Size of the grid for heatmap
            output_path: Optional path to save plot
            
        Returns:
            Path to saved plot (if output_path provided)
        """
        self.logger.info(f"Creating spatial density heatmap for radius {radius} {unit}")
        
        x_col, y_col = coordinate_columns
        density_col = f'nn_count_within_{radius}_{unit}'
        
        if density_col not in df.columns:
            self.logger.error(f"Density column {density_col} not found")
            return None
        
        # Get coordinate ranges
        x_min, x_max = df[x_col].min(), df[x_col].max()
        y_min, y_max = df[y_col].min(), df[y_col].max()
        
        # Create grid
        x_edges = np.linspace(x_min, x_max, grid_size + 1)
        y_edges = np.linspace(y_min, y_max, grid_size + 1)
        
        # Calculate grid centers
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        
        # Create 2D histogram weighted by density values
        density_grid = np.zeros((grid_size, grid_size))
        count_grid = np.zeros((grid_size, grid_size))
        
        for _, row in df.iterrows():
            if pd.notna(row[density_col]):
                # Find grid indices
                x_idx = np.digitize(row[x_col], x_edges) - 1
                y_idx = np.digitize(row[y_col], y_edges) - 1
                
                # Ensure indices are within bounds
                x_idx = np.clip(x_idx, 0, grid_size - 1)
                y_idx = np.clip(y_idx, 0, grid_size - 1)
                
                # Accumulate density values
                density_grid[y_idx, x_idx] += row[density_col]
                count_grid[y_idx, x_idx] += 1
        
        # Calculate average density in each grid cell
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_density_grid = density_grid / count_grid
            avg_density_grid[count_grid == 0] = np.nan
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        
        im = plt.imshow(avg_density_grid, extent=[x_min, x_max, y_min, y_max],
                       origin='lower', cmap='viridis', aspect='equal')
        
        plt.colorbar(im, label=f'Average Neighbor Count (r={radius} {unit})')
        plt.xlabel(f'X Position ({unit})')
        plt.ylabel(f'Y Position ({unit})')
        plt.title(f'Spatial Density Heatmap\n(Radius = {radius} {unit})')
        
        # Add scatter plot of particle positions with low alpha
        plt.scatter(df[x_col], df[y_col], c='white', s=1, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            heatmap_path = f"{output_path}_density_heatmap_r{radius}.png"
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved density heatmap to {heatmap_path}")
            plt.close()
            return heatmap_path
        else:
            plt.show()
            return None

    def export_density_results(self, df: pd.DataFrame, 
                             output_path: str,
                             include_statistics: bool = True,
                             include_plots: bool = True) -> Dict[str, str]:
        """Export density analysis results to files.
        
        Args:
            df: DataFrame with density analysis results
            output_path: Base path for output files
            include_statistics: Whether to export statistics
            include_plots: Whether to create and save plots
            
        Returns:
            Dictionary mapping export types to file paths
        """
        self.logger.info(f"Exporting density analysis results to {output_path}")
        
        export_paths = {}
        
        # Export main data
        data_path = f"{output_path}_density_analysis.csv"
        df.to_csv(data_path, index=False)
        export_paths['data'] = data_path
        self.logger.info(f"Exported density data to {data_path}")
        
        # Export statistics
        if include_statistics:
            stats = self.calculate_density_statistics(df)
            if stats:
                stats_path = f"{output_path}_density_statistics.json"
                import json
                with open(stats_path, 'w') as f:
                    # Convert numpy types to Python types for JSON serialization
                    def convert_numpy(obj):
                        if isinstance(obj, np.integer):
                            return int(obj)
                        elif isinstance(obj, np.floating):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        return obj
                    
                    def deep_convert(obj):
                        if isinstance(obj, dict):
                            return {k: deep_convert(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [deep_convert(v) for v in obj]
                        else:
                            return convert_numpy(obj)
                    
                    json.dump(deep_convert(stats), f, indent=2)
                
                export_paths['statistics'] = stats_path
                self.logger.info(f"Exported density statistics to {stats_path}")
        
        # Create plots
        if include_plots:
            plot_paths = self.create_density_plots(df, output_path=output_path)
            export_paths.update(plot_paths)
        
        return export_paths

    def update_parameters(self, parameters: Dict[str, Any]):
        """Update analysis parameters.
        
        Args:
            parameters: Dictionary of new parameters
        """
        self.parameters.update(parameters)
        self.logger.debug(f"Updated density analyzer parameters: {parameters}")

    def get_density_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get a summary of density analysis results.
        
        Args:
            df: DataFrame with density analysis results
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_particles': len(df),
            'total_frames': df['frame'].nunique() if 'frame' in df.columns else 1,
            'coordinate_range': {}
        }
        
        # Add coordinate ranges
        if 'x' in df.columns and 'y' in df.columns:
            summary['coordinate_range'] = {
                'x_min': float(df['x'].min()),
                'x_max': float(df['x'].max()),
                'y_min': float(df['y'].min()),
                'y_max': float(df['y'].max())
            }
        
        # Add nearest neighbor summary
        if 'nn_distance' in df.columns:
            nn_values = df['nn_distance'].dropna()
            if len(nn_values) > 0:
                summary['nearest_neighbor'] = {
                    'mean_distance': float(nn_values.mean()),
                    'median_distance': float(nn_values.median()),
                    'std_distance': float(nn_values.std()),
                    'min_distance': float(nn_values.min()),
                    'max_distance': float(nn_values.max())
                }
        
        # Add density column summaries
        density_cols = [col for col in df.columns if 'nn_count_within_' in col]
        if density_cols:
            summary['density_radii'] = {}
            for col in density_cols:
                values = df[col].dropna()
                if len(values) > 0:
                    # Extract radius and unit from column name
                    parts = col.split('_')
                    radius = parts[3]
                    unit = parts[4] if len(parts) > 4 else 'pixels'
                    
                    summary['density_radii'][f'{radius}_{unit}'] = {
                        'mean_count': float(values.mean()),
                        'median_count': float(values.median()),
                        'std_count': float(values.std()),
                        'max_count': float(values.max())
                    }
        
        return summary