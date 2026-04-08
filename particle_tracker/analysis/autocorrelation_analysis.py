#!/usr/bin/env python3
"""
Direction Autocorrelation Analysis Module
=========================================

Provides direction autocorrelation analysis for characterizing motion persistence
and directionality in particle trajectories. This is one of the critical missing
components from the original analysis scripts.

Key Features:
- Individual track autocorrelation calculation
- Ensemble autocorrelation analysis
- Persistence length estimation
- Directional bias detection
- Autocorrelation plotting and visualization
"""

import logging
import math
from typing import Optional, Dict, List, Any, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server compatibility


class DirectionAutocorrelationAnalyzer:
    """Analyze direction autocorrelation in particle trajectories."""

    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize autocorrelation analyzer.

        Args:
            parameters: Analysis parameters dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.parameters = parameters or {}

    def calculate_track_autocorrelation(self, track_data: pd.DataFrame,
                                      max_lag: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate direction autocorrelation for a single track.

        Args:
            track_data: DataFrame for a single track with x, y, frame columns
            max_lag: Maximum lag time for autocorrelation (defaults to track_length/2)

        Returns:
            Dictionary with autocorrelation results
        """
        if len(track_data) < 3:
            return self._empty_autocorr_result()

        # Sort by frame and calculate step vectors
        track_data = track_data.sort_values('frame').copy()
        positions = track_data[['x', 'y']].values

        # Calculate step vectors
        steps = np.diff(positions, axis=0)
        step_magnitudes = np.linalg.norm(steps, axis=1)

        # Remove zero-length steps
        valid_steps = step_magnitudes > 0
        if np.sum(valid_steps) < 2:
            return self._empty_autocorr_result()

        steps = steps[valid_steps]
        step_magnitudes = step_magnitudes[valid_steps]

        # Calculate unit vectors (directions)
        unit_vectors = steps / step_magnitudes[:, np.newaxis]

        # Determine maximum lag
        if max_lag is None:
            max_lag = min(len(unit_vectors) - 1, 20)  # Reasonable default
        else:
            max_lag = min(max_lag, len(unit_vectors) - 1)

        if max_lag < 1:
            return self._empty_autocorr_result()

        # Calculate autocorrelation for each lag
        lags = np.arange(1, max_lag + 1)
        autocorr_values = []

        for lag in lags:
            if lag >= len(unit_vectors):
                autocorr_values.append(np.nan)
                continue

            # Calculate dot products between vectors separated by lag
            dot_products = []
            for i in range(len(unit_vectors) - lag):
                dot_product = np.dot(unit_vectors[i], unit_vectors[i + lag])
                # Ensure dot product is in valid range [-1, 1]
                dot_product = np.clip(dot_product, -1.0, 1.0)
                dot_products.append(dot_product)

            if len(dot_products) > 0:
                autocorr_values.append(np.mean(dot_products))
            else:
                autocorr_values.append(np.nan)

        autocorr_values = np.array(autocorr_values)

        # Calculate persistence length (characteristic decay length)
        persistence_length = self._calculate_persistence_length(lags, autocorr_values)

        # Calculate directional bias (mean autocorrelation)
        valid_autocorr = autocorr_values[~np.isnan(autocorr_values)]
        directional_bias = np.mean(valid_autocorr) if len(valid_autocorr) > 0 else np.nan

        # Fit exponential decay if possible
        decay_params = self._fit_exponential_decay(lags, autocorr_values)

        return {
            'track_id': track_data['track_number'].iloc[0] if 'track_number' in track_data.columns else None,
            'n_steps': len(unit_vectors),
            'max_lag': max_lag,
            'lags': lags,
            'autocorr_values': autocorr_values,
            'persistence_length': persistence_length,
            'directional_bias': directional_bias,
            'decay_constant': decay_params['decay_constant'],
            'decay_amplitude': decay_params['amplitude'],
            'decay_offset': decay_params['offset'],
            'decay_r_squared': decay_params['r_squared']
        }

    def _empty_autocorr_result(self) -> Dict[str, Any]:
        """Return empty autocorrelation result."""
        return {
            'track_id': None,
            'n_steps': 0,
            'max_lag': 0,
            'lags': np.array([]),
            'autocorr_values': np.array([]),
            'persistence_length': np.nan,
            'directional_bias': np.nan,
            'decay_constant': np.nan,
            'decay_amplitude': np.nan,
            'decay_offset': np.nan,
            'decay_r_squared': np.nan
        }

    def _calculate_persistence_length(self, lags: np.ndarray, autocorr: np.ndarray) -> float:
        """
        Calculate persistence length from autocorrelation decay.

        Args:
            lags: Lag times
            autocorr: Autocorrelation values

        Returns:
            Persistence length (lag where autocorr drops to 1/e of initial value)
        """
        valid_mask = ~np.isnan(autocorr)
        if np.sum(valid_mask) < 2:
            return np.nan

        valid_lags = lags[valid_mask]
        valid_autocorr = autocorr[valid_mask]

        if len(valid_autocorr) == 0 or valid_autocorr[0] <= 0:
            return np.nan

        # Find where autocorrelation drops to 1/e of initial value
        target_value = valid_autocorr[0] / np.e

        # Find the first crossing
        crossing_indices = np.where(valid_autocorr <= target_value)[0]

        if len(crossing_indices) > 0:
            crossing_idx = crossing_indices[0]
            if crossing_idx == 0:
                return valid_lags[0]
            
            # Interpolate for more precise estimate
            if crossing_idx < len(valid_autocorr):
                x1, x2 = valid_lags[crossing_idx - 1], valid_lags[crossing_idx]
                y1, y2 = valid_autocorr[crossing_idx - 1], valid_autocorr[crossing_idx]
                
                # Linear interpolation
                if y2 != y1:
                    persistence_length = x1 + (target_value - y1) * (x2 - x1) / (y2 - y1)
                else:
                    persistence_length = x1
                
                return persistence_length

        # If no crossing found, extrapolate
        return valid_lags[-1] * 2  # Conservative estimate

    def _fit_exponential_decay(self, lags: np.ndarray, autocorr: np.ndarray) -> Dict[str, float]:
        """
        Fit exponential decay model to autocorrelation data.

        Model: f(x) = A * exp(-x/tau) + C

        Args:
            lags: Lag times
            autocorr: Autocorrelation values

        Returns:
            Dictionary with fitted parameters
        """
        try:
            valid_mask = ~np.isnan(autocorr)
            if np.sum(valid_mask) < 3:
                return {'decay_constant': np.nan, 'amplitude': np.nan, 
                       'offset': np.nan, 'r_squared': np.nan}

            valid_lags = lags[valid_mask]
            valid_autocorr = autocorr[valid_mask]

            # Initial parameter estimates
            A_init = valid_autocorr[0] if len(valid_autocorr) > 0 else 1.0
            C_init = valid_autocorr[-1] if len(valid_autocorr) > 0 else 0.0
            tau_init = len(valid_lags) / 3.0  # Initial guess for decay constant

            # Try to fit exponential decay
            from scipy.optimize import curve_fit

            def exp_decay(x, A, tau, C):
                return A * np.exp(-x / tau) + C

            # Fit the model
            try:
                popt, pcov = curve_fit(
                    exp_decay, valid_lags, valid_autocorr,
                    p0=[A_init, tau_init, C_init],
                    bounds=([0, 0.1, -1], [10, 100, 1]),
                    maxfev=1000
                )

                A_fit, tau_fit, C_fit = popt

                # Calculate R-squared
                y_pred = exp_decay(valid_lags, A_fit, tau_fit, C_fit)
                ss_res = np.sum((valid_autocorr - y_pred) ** 2)
                ss_tot = np.sum((valid_autocorr - np.mean(valid_autocorr)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                return {
                    'decay_constant': tau_fit,
                    'amplitude': A_fit,
                    'offset': C_fit,
                    'r_squared': r_squared
                }

            except Exception as fit_error:
                self.logger.debug(f"Exponential fit failed: {fit_error}")
                return {'decay_constant': np.nan, 'amplitude': np.nan,
                       'offset': np.nan, 'r_squared': np.nan}

        except Exception as e:
            self.logger.debug(f"Error in exponential decay fitting: {e}")
            return {'decay_constant': np.nan, 'amplitude': np.nan,
                   'offset': np.nan, 'r_squared': np.nan}

    def analyze_all_tracks(self, trajectory_data: pd.DataFrame,
                          max_lag: Optional[int] = None) -> pd.DataFrame:
        """
        Analyze autocorrelation for all tracks in the dataset.

        Args:
            trajectory_data: DataFrame with trajectory data
            max_lag: Maximum lag time for autocorrelation

        Returns:
            DataFrame with autocorrelation results for each track
        """
        self.logger.info("Calculating direction autocorrelation for all tracks")

        if 'track_number' not in trajectory_data.columns:
            self.logger.error("No track_number column found in trajectory data")
            return pd.DataFrame()

        autocorr_results = []
        track_ids = trajectory_data['track_number'].unique()

        for track_id in tqdm(track_ids, desc="Analyzing track autocorrelations"):
            track_data = trajectory_data[trajectory_data['track_number'] == track_id]
            
            if len(track_data) < 3:
                continue

            result = self.calculate_track_autocorrelation(track_data, max_lag)
            
            # Add track-specific metadata
            result['track_length'] = len(track_data)
            result['frame_span'] = track_data['frame'].max() - track_data['frame'].min() + 1 if 'frame' in track_data.columns else len(track_data)
            
            autocorr_results.append(result)

        if not autocorr_results:
            self.logger.warning("No valid tracks found for autocorrelation analysis")
            return pd.DataFrame()

        # Convert to DataFrame
        results_df = pd.DataFrame(autocorr_results)

        # Calculate ensemble statistics
        ensemble_stats = self._calculate_ensemble_statistics(autocorr_results)
        self.logger.info(f"Autocorrelation analysis completed for {len(results_df)} tracks")
        self.logger.info(f"Mean persistence length: {ensemble_stats.get('mean_persistence_length', 0):.2f}")
        self.logger.info(f"Mean directional bias: {ensemble_stats.get('mean_directional_bias', 0):.3f}")

        return results_df

    def _calculate_ensemble_statistics(self, autocorr_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate ensemble statistics from individual track results."""
        if not autocorr_results:
            return {}

        # Extract metrics
        persistence_lengths = [r['persistence_length'] for r in autocorr_results 
                             if not np.isnan(r['persistence_length'])]
        directional_biases = [r['directional_bias'] for r in autocorr_results 
                            if not np.isnan(r['directional_bias'])]
        decay_constants = [r['decay_constant'] for r in autocorr_results 
                         if not np.isnan(r['decay_constant'])]

        stats = {}
        
        if persistence_lengths:
            stats['mean_persistence_length'] = np.mean(persistence_lengths)
            stats['std_persistence_length'] = np.std(persistence_lengths)
            stats['median_persistence_length'] = np.median(persistence_lengths)

        if directional_biases:
            stats['mean_directional_bias'] = np.mean(directional_biases)
            stats['std_directional_bias'] = np.std(directional_biases)

        if decay_constants:
            stats['mean_decay_constant'] = np.mean(decay_constants)
            stats['std_decay_constant'] = np.std(decay_constants)

        return stats

    def calculate_ensemble_autocorrelation(self, trajectory_data: pd.DataFrame,
                                         max_lag: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate ensemble-averaged autocorrelation across all tracks.

        Args:
            trajectory_data: DataFrame with trajectory data
            max_lag: Maximum lag time for autocorrelation

        Returns:
            Dictionary with ensemble autocorrelation results
        """
        self.logger.info("Calculating ensemble autocorrelation")

        if 'track_number' not in trajectory_data.columns:
            return {}

        # Collect autocorrelation curves from all tracks
        all_autocorr_curves = []
        track_ids = trajectory_data['track_number'].unique()

        for track_id in track_ids:
            track_data = trajectory_data[trajectory_data['track_number'] == track_id]
            result = self.calculate_track_autocorrelation(track_data, max_lag)
            
            if len(result['autocorr_values']) > 0:
                all_autocorr_curves.append({
                    'lags': result['lags'],
                    'autocorr': result['autocorr_values']
                })

        if not all_autocorr_curves:
            return {}

        # Find common lag range
        max_common_lag = min([len(curve['lags']) for curve in all_autocorr_curves])
        if max_common_lag == 0:
            return {}

        # Calculate ensemble average
        ensemble_lags = np.arange(1, max_common_lag + 1)
        ensemble_autocorr = np.zeros(max_common_lag)
        ensemble_std = np.zeros(max_common_lag)

        for lag_idx in range(max_common_lag):
            lag_values = []
            for curve in all_autocorr_curves:
                if lag_idx < len(curve['autocorr']) and not np.isnan(curve['autocorr'][lag_idx]):
                    lag_values.append(curve['autocorr'][lag_idx])
            
            if lag_values:
                ensemble_autocorr[lag_idx] = np.mean(lag_values)
                ensemble_std[lag_idx] = np.std(lag_values)
            else:
                ensemble_autocorr[lag_idx] = np.nan
                ensemble_std[lag_idx] = np.nan

        # Calculate ensemble persistence length
        ensemble_persistence = self._calculate_persistence_length(ensemble_lags, ensemble_autocorr)

        # Fit ensemble decay
        ensemble_decay = self._fit_exponential_decay(ensemble_lags, ensemble_autocorr)

        return {
            'n_tracks': len(all_autocorr_curves),
            'max_lag': max_common_lag,
            'lags': ensemble_lags,
            'ensemble_autocorr': ensemble_autocorr,
            'ensemble_std': ensemble_std,
            'ensemble_persistence_length': ensemble_persistence,
            'ensemble_decay_constant': ensemble_decay['decay_constant'],
            'ensemble_r_squared': ensemble_decay['r_squared']
        }

    def plot_individual_autocorrelations(self, autocorr_results: pd.DataFrame,
                                       output_path: str, max_tracks: int = 20) -> bool:
        """
        Plot individual track autocorrelations.

        Args:
            autocorr_results: DataFrame with autocorrelation results
            output_path: Path for output plot
            max_tracks: Maximum number of tracks to plot

        Returns:
            True if successful
        """
        try:
            # Select subset of tracks if too many
            if len(autocorr_results) > max_tracks:
                # Select tracks with longest persistence lengths for better visualization
                autocorr_results = autocorr_results.nlargest(max_tracks, 'persistence_length')

            plt.figure(figsize=(10, 8))

            # Plot individual tracks
            for idx, row in autocorr_results.iterrows():
                if len(row['lags']) > 0 and len(row['autocorr_values']) > 0:
                    plt.plot(row['lags'], row['autocorr_values'], 
                           alpha=0.3, color='gray', linewidth=0.5)

            plt.xlabel('Lag (steps)')
            plt.ylabel('Direction Autocorrelation')
            plt.title(f'Individual Track Autocorrelations (n={len(autocorr_results)})')
            plt.grid(True, alpha=0.3)
            plt.ylim(-1, 1)

            # Add horizontal line at 1/e
            plt.axhline(y=1/np.e, color='red', linestyle='--', alpha=0.7, 
                       label=f'1/e ≈ {1/np.e:.3f}')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)

            plt.legend()
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Individual autocorrelation plot saved to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error plotting individual autocorrelations: {e}")
            return False

    def plot_ensemble_autocorrelation(self, ensemble_result: Dict[str, Any],
                                    output_path: str) -> bool:
        """
        Plot ensemble autocorrelation with error bars.

        Args:
            ensemble_result: Result from calculate_ensemble_autocorrelation
            output_path: Path for output plot

        Returns:
            True if successful
        """
        try:
            if not ensemble_result or len(ensemble_result.get('lags', [])) == 0:
                self.logger.warning("No ensemble autocorrelation data to plot")
                return False

            plt.figure(figsize=(10, 6))

            lags = ensemble_result['lags']
            autocorr = ensemble_result['ensemble_autocorr']
            std = ensemble_result['ensemble_std']

            # Plot ensemble average with error bars
            plt.errorbar(lags, autocorr, yerr=std, 
                        marker='o', markersize=4, capsize=3,
                        color='blue', label=f"Ensemble (n={ensemble_result['n_tracks']})")

            # Plot fitted exponential decay if available
            if not np.isnan(ensemble_result.get('ensemble_decay_constant', np.nan)):
                decay_const = ensemble_result['ensemble_decay_constant']
                amplitude = ensemble_result.get('ensemble_amplitude', autocorr[0])
                offset = ensemble_result.get('ensemble_offset', 0)
                
                # Generate smooth curve
                x_smooth = np.linspace(lags[0], lags[-1], 100)
                y_smooth = amplitude * np.exp(-x_smooth / decay_const) + offset
                
                plt.plot(x_smooth, y_smooth, 'r--', 
                        label=f'Exp. fit (τ={decay_const:.1f})')

            plt.xlabel('Lag (steps)')
            plt.ylabel('Direction Autocorrelation')
            plt.title('Ensemble Direction Autocorrelation')
            plt.grid(True, alpha=0.3)
            plt.ylim(-1, 1)

            # Add reference lines
            plt.axhline(y=1/np.e, color='red', linestyle=':', alpha=0.7, 
                       label=f'1/e ≈ {1/np.e:.3f}')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)

            # Add persistence length annotation
            if not np.isnan(ensemble_result.get('ensemble_persistence_length', np.nan)):
                pers_length = ensemble_result['ensemble_persistence_length']
                plt.axvline(x=pers_length, color='green', linestyle='--', alpha=0.7,
                           label=f'Persistence length: {pers_length:.1f}')

            plt.legend()
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Ensemble autocorrelation plot saved to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error plotting ensemble autocorrelation: {e}")
            return False

    def export_autocorrelation_results(self, autocorr_results: pd.DataFrame,
                                     ensemble_result: Dict[str, Any],
                                     output_path: str) -> bool:
        """
        Export autocorrelation results to CSV.

        Args:
            autocorr_results: Individual track results
            ensemble_result: Ensemble results
            output_path: Path for output CSV

        Returns:
            True if successful
        """
        try:
            # Prepare export data
            export_data = autocorr_results.copy()
            
            # Remove complex columns that can't be easily exported
            columns_to_remove = ['lags', 'autocorr_values']
            for col in columns_to_remove:
                if col in export_data.columns:
                    export_data = export_data.drop(columns=[col])

            # Add ensemble statistics as additional rows
            if ensemble_result:
                ensemble_row = {
                    'track_id': 'ENSEMBLE',
                    'n_steps': np.nan,
                    'max_lag': ensemble_result.get('max_lag', np.nan),
                    'persistence_length': ensemble_result.get('ensemble_persistence_length', np.nan),
                    'directional_bias': np.nan,
                    'decay_constant': ensemble_result.get('ensemble_decay_constant', np.nan),
                    'decay_amplitude': np.nan,
                    'decay_offset': np.nan,
                    'decay_r_squared': ensemble_result.get('ensemble_r_squared', np.nan),
                    'track_length': ensemble_result.get('n_tracks', np.nan),
                    'frame_span': np.nan
                }
                
                # Ensure all columns exist
                for col in export_data.columns:
                    if col not in ensemble_row:
                        ensemble_row[col] = np.nan
                
                ensemble_df = pd.DataFrame([ensemble_row])
                export_data = pd.concat([export_data, ensemble_df], ignore_index=True)

            # Export to CSV
            export_data.to_csv(output_path, index=False)

            self.logger.info(f"Autocorrelation results exported to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting autocorrelation results: {e}")
            return False

    def analyze_trajectory_data(self, trajectory_data: pd.DataFrame,
                              output_dir: str = None) -> Dict[str, Any]:
        """
        Complete autocorrelation analysis pipeline.

        Args:
            trajectory_data: DataFrame with trajectory data
            output_dir: Directory for output files (optional)

        Returns:
            Dictionary with all analysis results
        """
        self.logger.info("Starting complete autocorrelation analysis")

        # Individual track analysis
        autocorr_results = self.analyze_all_tracks(trajectory_data)
        
        if len(autocorr_results) == 0:
            self.logger.warning("No tracks analyzed successfully")
            return {}

        # Ensemble analysis
        ensemble_result = self.calculate_ensemble_autocorrelation(trajectory_data)

        # Create output directory if specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate plots
            self.plot_individual_autocorrelations(
                autocorr_results, output_dir / "individual_autocorrelations.png"
            )
            
            if ensemble_result:
                self.plot_ensemble_autocorrelation(
                    ensemble_result, output_dir / "ensemble_autocorrelation.png"
                )

            # Export results
            self.export_autocorrelation_results(
                autocorr_results, ensemble_result, output_dir / "autocorrelation_results.csv"
            )

        return {
            'individual_results': autocorr_results,
            'ensemble_result': ensemble_result,
            'summary_statistics': self._calculate_ensemble_statistics(
                autocorr_results.to_dict('records')
            )
        }


# Convenience functions
def analyze_direction_autocorrelation(trajectory_data: pd.DataFrame,
                                    output_dir: str = None,
                                    max_lag: int = None) -> Dict[str, Any]:
    """
    Convenience function for direction autocorrelation analysis.

    Args:
        trajectory_data: DataFrame with trajectory data
        output_dir: Directory for output files
        max_lag: Maximum lag for autocorrelation analysis

    Returns:
        Analysis results dictionary
    """
    analyzer = DirectionAutocorrelationAnalyzer({'max_lag': max_lag})
    return analyzer.analyze_trajectory_data(trajectory_data, output_dir)


def calculate_track_autocorrelation(track_data: pd.DataFrame, max_lag: int = None) -> Dict[str, Any]:
    """
    Convenience function for single track autocorrelation.

    Args:
        track_data: DataFrame for a single track
        max_lag: Maximum lag for analysis

    Returns:
        Autocorrelation results for the track
    """
    analyzer = DirectionAutocorrelationAnalyzer()
    return analyzer.calculate_track_autocorrelation(track_data, max_lag)