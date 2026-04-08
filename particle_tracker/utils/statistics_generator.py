#!/usr/bin/env python3
"""
Enhanced Statistics Generator Module
====================================

Comprehensive statistics generation for particle tracking analysis with hierarchical 
experiment processing, cross-condition comparisons, and advanced metrics calculation.

This module provides:
- Single file, condition, and experiment-level statistics
- Cross-condition statistical comparisons
- Advanced trajectory metrics (sRg, linearity, density analysis)
- Multiple export formats (CSV, Excel, JSON)
- Individual track analysis and aggregation
- Automated statistical reporting

Based on patterns from autocorrelation.py and trajectory_analyzer.py scripts.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import json

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from PyQt6.QtCore import QObject, pyqtSignal, QThread
from PyQt6.QtWidgets import QMessageBox


@dataclass
class StatisticsConfig:
    """Configuration for statistics generation."""
    
    # Analysis parameters
    mobility_threshold: float = 2.11  # sRg threshold for mobility classification
    linearity_eigenvalue_threshold: float = 20.0  # Eigenvalue ratio for linearity
    linearity_step_alignment_threshold: float = 0.7  # Step alignment threshold
    density_radii: List[int] = None  # Radii for density analysis
    
    # Output options
    generate_plots: bool = True
    plot_format: str = 'both'  # 'png', 'pdf', 'both'
    export_individual_tracks: bool = True
    include_advanced_metrics: bool = True
    
    # Statistical options
    confidence_level: float = 0.95
    use_robust_statistics: bool = True  # Use median/IQR instead of mean/std when appropriate
    
    def __post_init__(self):
        if self.density_radii is None:
            self.density_radii = [3, 5, 10, 20, 30]


@dataclass 
class TrackStatistics:
    """Statistics for an individual track."""
    
    track_id: int
    track_length: int
    
    # Basic metrics
    radius_gyration: float
    scaled_radius_gyration: float
    mean_step_size: float
    
    # Mobility classification
    mobility_classification: str
    mobility_confidence: float = np.nan
    
    # Linearity metrics
    eigenvalue_ratio: float = np.nan
    step_alignment: float = np.nan
    directionality_ratio: float = np.nan
    linearity_classification: str = "unclassified"
    
    # Shape metrics
    asymmetry: float = np.nan
    skewness: float = np.nan
    kurtosis: float = np.nan
    
    # Density metrics
    mean_nn_distance: float = np.nan
    density_counts: Dict[int, float] = None  # {radius: mean_count}
    
    # Intensity metrics
    mean_intensity: float = np.nan
    intensity_std: float = np.nan
    background_subtracted_intensity: float = np.nan
    
    # Velocity and diffusion
    mean_velocity: float = np.nan
    diffusion_coefficient: float = np.nan
    
    def __post_init__(self):
        if self.density_counts is None:
            self.density_counts = {}


@dataclass
class FileStatistics:
    """Statistics for a single file."""
    
    filename: str
    total_tracks: int
    total_localizations: int
    total_frames: int
    
    # Track length statistics
    mean_track_length: float
    median_track_length: float
    std_track_length: float
    
    # Mobility statistics
    mobile_tracks: int
    immobile_tracks: int
    mobile_percentage: float
    
    # Linearity statistics (for mobile tracks)
    linear_tracks: int
    nonlinear_tracks: int
    linear_percentage: float
    unidirectional_tracks: int
    bidirectional_tracks: int
    
    # Metric means and SEMs
    mean_radius_gyration: float = np.nan
    sem_radius_gyration: float = np.nan
    mean_scaled_rg: float = np.nan
    sem_scaled_rg: float = np.nan
    mean_step_size: float = np.nan
    sem_step_size: float = np.nan
    
    # Advanced metrics
    mean_eigenvalue_ratio: float = np.nan
    sem_eigenvalue_ratio: float = np.nan
    mean_directionality_ratio: float = np.nan
    sem_directionality_ratio: float = np.nan
    
    # Density metrics
    mean_nn_distance: float = np.nan
    sem_nn_distance: float = np.nan
    density_metrics: Dict[int, Tuple[float, float]] = None  # {radius: (mean, sem)}
    
    # Quality metrics
    localization_precision: float = np.nan
    tracking_efficiency: float = np.nan  # fraction of localizations that were linked
    
    def __post_init__(self):
        if self.density_metrics is None:
            self.density_metrics = {}


@dataclass
class ConditionStatistics:
    """Statistics aggregated across files in a condition."""
    
    condition_name: str
    file_count: int
    total_tracks: int
    total_localizations: int
    
    # Aggregated metrics (mean across files)
    mean_mobile_percentage: float
    sem_mobile_percentage: float
    mean_linear_percentage: float
    sem_linear_percentage: float
    
    # Trajectory metrics
    mean_radius_gyration: float
    sem_radius_gyration: float
    mean_scaled_rg: float
    sem_scaled_rg: float
    
    # File-level statistics
    file_statistics: List[FileStatistics] = None
    
    def __post_init__(self):
        if self.file_statistics is None:
            self.file_statistics = []


class StatisticsWorker(QThread):
    """Worker thread for generating statistics."""
    
    progressUpdate = pyqtSignal(str, int)  # message, percentage
    statisticsCompleted = pyqtSignal(object)  # statistics_result
    errorOccurred = pyqtSignal(str)
    
    def __init__(self, data, config: StatisticsConfig, analysis_type: str = "file"):
        super().__init__()
        self.data = data
        self.config = config
        self.analysis_type = analysis_type  # "file", "condition", "experiment"
        self.logger = logging.getLogger(__name__)
    
    def run(self):
        """Run statistics generation."""
        try:
            if self.analysis_type == "file":
                result = self._generate_file_statistics(self.data)
            elif self.analysis_type == "condition":
                result = self._generate_condition_statistics(self.data)
            elif self.analysis_type == "experiment":
                result = self._generate_experiment_statistics(self.data)
            else:
                raise ValueError(f"Unknown analysis type: {self.analysis_type}")
            
            self.statisticsCompleted.emit(result)
            
        except Exception as e:
            self.logger.error(f"Statistics generation error: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.errorOccurred.emit(str(e))
    
    def _generate_file_statistics(self, df: pd.DataFrame) -> FileStatistics:
        """Generate statistics for a single file."""
        self.progressUpdate.emit("Analyzing file statistics...", 10)
        
        # Basic counts
        total_tracks = df['track_number'].nunique() if 'track_number' in df.columns else 0
        total_localizations = len(df)
        total_frames = df['frame'].nunique() if 'frame' in df.columns else 0
        
        self.progressUpdate.emit("Calculating track metrics...", 30)
        
        if total_tracks == 0:
            # Return empty statistics
            return FileStatistics(
                filename="unknown",
                total_tracks=0,
                total_localizations=total_localizations,
                total_frames=total_frames,
                mean_track_length=0,
                median_track_length=0,
                std_track_length=0,
                mobile_tracks=0,
                immobile_tracks=0,
                mobile_percentage=0,
                linear_tracks=0,
                nonlinear_tracks=0,
                linear_percentage=0,
                unidirectional_tracks=0,
                bidirectional_tracks=0
            )
        
        # Track length statistics
        track_lengths = df.groupby('track_number').size()
        mean_track_length = track_lengths.mean()
        median_track_length = track_lengths.median()
        std_track_length = track_lengths.std()
        
        self.progressUpdate.emit("Analyzing mobility classification...", 50)
        
        # Mobility classification
        mobile_tracks = immobile_tracks = 0
        if 'mobility_classification' in df.columns:
            mobility_counts = df.groupby('track_number')['mobility_classification'].first().value_counts()
            mobile_tracks = mobility_counts.get('mobile', 0)
            immobile_tracks = mobility_counts.get('immobile', 0)
        elif 'scaled_radius_gyration' in df.columns or 'sRg' in df.columns:
            # Calculate mobility from sRg
            srg_col = 'scaled_radius_gyration' if 'scaled_radius_gyration' in df.columns else 'sRg'
            track_srg = df.groupby('track_number')[srg_col].first()
            mobile_tracks = (track_srg >= self.config.mobility_threshold).sum()
            immobile_tracks = (track_srg < self.config.mobility_threshold).sum()
        
        mobile_percentage = (mobile_tracks / total_tracks * 100) if total_tracks > 0 else 0
        
        self.progressUpdate.emit("Analyzing linearity classification...", 70)
        
        # Linearity classification (for mobile tracks)
        linear_tracks = nonlinear_tracks = 0
        unidirectional_tracks = bidirectional_tracks = 0
        
        if 'linearity_classification' in df.columns:
            # Get mobile track numbers
            if 'mobility_classification' in df.columns:
                mobile_track_nums = df[df['mobility_classification'] == 'mobile']['track_number'].unique()
                mobile_df = df[df['track_number'].isin(mobile_track_nums)]
            else:
                mobile_df = df  # Use all tracks if no mobility classification
            
            linearity_counts = mobile_df.groupby('track_number')['linearity_classification'].first().value_counts()
            linear_tracks = linearity_counts.get('linear_unidirectional', 0) + linearity_counts.get('linear_bidirectional', 0)
            nonlinear_tracks = linearity_counts.get('non_linear', 0)
            unidirectional_tracks = linearity_counts.get('linear_unidirectional', 0)
            bidirectional_tracks = linearity_counts.get('linear_bidirectional', 0)
        
        mobile_count = mobile_tracks if mobile_tracks > 0 else total_tracks
        linear_percentage = (linear_tracks / mobile_count * 100) if mobile_count > 0 else 0
        
        self.progressUpdate.emit("Calculating aggregate metrics...", 90)
        
        # Calculate means and SEMs for various metrics
        track_metrics = df.groupby('track_number').first()  # Get one row per track
        
        # Radius of gyration
        mean_rg = sem_rg = np.nan
        if 'radius_gyration' in track_metrics.columns:
            rg_values = track_metrics['radius_gyration'].dropna()
            if len(rg_values) > 0:
                mean_rg = rg_values.mean()
                sem_rg = rg_values.sem()
        
        # Scaled radius of gyration
        mean_srg = sem_srg = np.nan
        srg_col = None
        for col in ['scaled_radius_gyration', 'sRg', 'radius_gyration_ratio_to_mean_step_size']:
            if col in track_metrics.columns:
                srg_col = col
                break
        
        if srg_col:
            srg_values = track_metrics[srg_col].dropna()
            if len(srg_values) > 0:
                mean_srg = srg_values.mean()
                sem_srg = srg_values.sem()
        
        # Step size
        mean_step = sem_step = np.nan
        step_col = None
        for col in ['mean_step_length', 'meanLag']:
            if col in track_metrics.columns:
                step_col = col
                break
        
        if step_col:
            step_values = track_metrics[step_col].dropna()
            if len(step_values) > 0:
                mean_step = step_values.mean()
                sem_step = step_values.sem()
        
        # Advanced metrics
        mean_eig_ratio = sem_eig_ratio = np.nan
        if 'eigenvalue_ratio' in track_metrics.columns:
            eig_values = track_metrics['eigenvalue_ratio'].dropna()
            if len(eig_values) > 0:
                mean_eig_ratio = eig_values.mean()
                sem_eig_ratio = eig_values.sem()
        
        mean_dir_ratio = sem_dir_ratio = np.nan
        if 'directionality_ratio' in track_metrics.columns:
            dir_values = track_metrics['directionality_ratio'].dropna()
            if len(dir_values) > 0:
                mean_dir_ratio = dir_values.mean()
                sem_dir_ratio = dir_values.sem()
        
        # Nearest neighbor distance
        mean_nn = sem_nn = np.nan
        nn_col = None
        for col in ['nn_distance', 'nnDist', 'nnDist_inFrame']:
            if col in df.columns:
                nn_col = col
                break
        
        if nn_col:
            nn_values = df[nn_col].dropna()
            if len(nn_values) > 0:
                mean_nn = nn_values.mean()
                sem_nn = nn_values.sem()
        
        # Density metrics
        density_metrics = {}
        for radius in self.config.density_radii:
            col_name = f'nnCountInFrame_within_{radius}_pixels'
            if col_name in df.columns:
                density_values = df[col_name].dropna()
                if len(density_values) > 0:
                    density_metrics[radius] = (density_values.mean(), density_values.sem())
        
        # Quality metrics
        tracking_efficiency = np.nan
        if 'id' in df.columns and total_tracks > 0:
            # Estimate tracking efficiency as fraction of localizations that were linked
            linked_localizations = df[df['track_number'].notna()]['id'].nunique()
            total_unique_localizations = df['id'].nunique()
            tracking_efficiency = linked_localizations / total_unique_localizations if total_unique_localizations > 0 else 0
        
        self.progressUpdate.emit("Statistics generation complete", 100)
        
        return FileStatistics(
            filename="current_file",
            total_tracks=total_tracks,
            total_localizations=total_localizations,
            total_frames=total_frames,
            mean_track_length=mean_track_length,
            median_track_length=median_track_length,
            std_track_length=std_track_length,
            mobile_tracks=mobile_tracks,
            immobile_tracks=immobile_tracks,
            mobile_percentage=mobile_percentage,
            linear_tracks=linear_tracks,
            nonlinear_tracks=nonlinear_tracks,
            linear_percentage=linear_percentage,
            unidirectional_tracks=unidirectional_tracks,
            bidirectional_tracks=bidirectional_tracks,
            mean_radius_gyration=mean_rg,
            sem_radius_gyration=sem_rg,
            mean_scaled_rg=mean_srg,
            sem_scaled_rg=sem_srg,
            mean_step_size=mean_step,
            sem_step_size=sem_step,
            mean_eigenvalue_ratio=mean_eig_ratio,
            sem_eigenvalue_ratio=sem_eig_ratio,
            mean_directionality_ratio=mean_dir_ratio,
            sem_directionality_ratio=sem_dir_ratio,
            mean_nn_distance=mean_nn,
            sem_nn_distance=sem_nn,
            density_metrics=density_metrics,
            tracking_efficiency=tracking_efficiency
        )
    
    def _generate_condition_statistics(self, file_stats_list: List[FileStatistics]) -> ConditionStatistics:
        """Generate statistics for a condition (multiple files)."""
        self.progressUpdate.emit("Aggregating condition statistics...", 50)
        
        if not file_stats_list:
            return ConditionStatistics(
                condition_name="unknown",
                file_count=0,
                total_tracks=0,
                total_localizations=0,
                mean_mobile_percentage=0,
                sem_mobile_percentage=0,
                mean_linear_percentage=0,
                sem_linear_percentage=0,
                mean_radius_gyration=np.nan,
                sem_radius_gyration=np.nan,
                mean_scaled_rg=np.nan,
                sem_scaled_rg=np.nan
            )
        
        # Aggregate basic counts
        total_tracks = sum(fs.total_tracks for fs in file_stats_list)
        total_localizations = sum(fs.total_localizations for fs in file_stats_list)
        
        # Calculate means and SEMs across files
        mobile_percentages = [fs.mobile_percentage for fs in file_stats_list]
        linear_percentages = [fs.linear_percentage for fs in file_stats_list]
        
        mean_mobile_pct = np.mean(mobile_percentages)
        sem_mobile_pct = np.std(mobile_percentages, ddof=1) / np.sqrt(len(mobile_percentages)) if len(mobile_percentages) > 1 else 0
        
        mean_linear_pct = np.mean(linear_percentages)
        sem_linear_pct = np.std(linear_percentages, ddof=1) / np.sqrt(len(linear_percentages)) if len(linear_percentages) > 1 else 0
        
        # Aggregate trajectory metrics
        rg_values = [fs.mean_radius_gyration for fs in file_stats_list if not np.isnan(fs.mean_radius_gyration)]
        mean_rg = np.mean(rg_values) if rg_values else np.nan
        sem_rg = np.std(rg_values, ddof=1) / np.sqrt(len(rg_values)) if len(rg_values) > 1 else 0
        
        srg_values = [fs.mean_scaled_rg for fs in file_stats_list if not np.isnan(fs.mean_scaled_rg)]
        mean_srg = np.mean(srg_values) if srg_values else np.nan
        sem_srg = np.std(srg_values, ddof=1) / np.sqrt(len(srg_values)) if len(srg_values) > 1 else 0
        
        self.progressUpdate.emit("Condition statistics complete", 100)
        
        return ConditionStatistics(
            condition_name="current_condition",
            file_count=len(file_stats_list),
            total_tracks=total_tracks,
            total_localizations=total_localizations,
            mean_mobile_percentage=mean_mobile_pct,
            sem_mobile_percentage=sem_mobile_pct,
            mean_linear_percentage=mean_linear_pct,
            sem_linear_percentage=sem_linear_pct,
            mean_radius_gyration=mean_rg,
            sem_radius_gyration=sem_rg,
            mean_scaled_rg=mean_srg,
            sem_scaled_rg=sem_srg,
            file_statistics=file_stats_list
        )
    
    def _generate_experiment_statistics(self, condition_stats_list: List[ConditionStatistics]) -> Dict[str, Any]:
        """Generate statistics for an experiment (multiple conditions)."""
        self.progressUpdate.emit("Generating experiment-wide statistics...", 75)
        
        experiment_stats = {
            'condition_count': len(condition_stats_list),
            'total_files': sum(cs.file_count for cs in condition_stats_list),
            'total_tracks': sum(cs.total_tracks for cs in condition_stats_list),
            'total_localizations': sum(cs.total_localizations for cs in condition_stats_list),
            'conditions': {cs.condition_name: cs for cs in condition_stats_list}
        }
        
        self.progressUpdate.emit("Experiment statistics complete", 100)
        
        return experiment_stats


class StatisticsGenerator(QObject):
    """Main statistics generator class."""
    
    statisticsGenerated = pyqtSignal(object)  # statistics_result
    progressUpdate = pyqtSignal(str, int)  # message, percentage
    errorOccurred = pyqtSignal(str)
    
    def __init__(self, config: Optional[StatisticsConfig] = None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config or StatisticsConfig()
        self.stats_worker = None
    
    def generate_file_statistics(self, df: pd.DataFrame, filename: str = None) -> FileStatistics:
        """Generate statistics for a single file synchronously."""
        self.logger.info("Generating file statistics")
        
        try:
            worker = StatisticsWorker(df, self.config, "file")
            worker.run()  # Run synchronously
            
            # Get the result (this is a bit hacky for sync operation)
            # In real implementation, you'd want to properly handle this
            stats = worker._generate_file_statistics(df)
            
            if filename:
                stats.filename = filename
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error generating file statistics: {e}")
            raise
    
    def generate_condition_statistics(self, file_data_list: List[Tuple[pd.DataFrame, str]], 
                                    condition_name: str) -> ConditionStatistics:
        """Generate statistics for a condition (multiple files)."""
        self.logger.info(f"Generating condition statistics for: {condition_name}")
        
        try:
            # Generate file statistics for each file
            file_stats_list = []
            for df, filename in file_data_list:
                file_stats = self.generate_file_statistics(df, filename)
                file_stats_list.append(file_stats)
            
            # Aggregate condition statistics
            worker = StatisticsWorker(file_stats_list, self.config, "condition")
            condition_stats = worker._generate_condition_statistics(file_stats_list)
            condition_stats.condition_name = condition_name
            
            return condition_stats
            
        except Exception as e:
            self.logger.error(f"Error generating condition statistics: {e}")
            raise
    
    def generate_experiment_statistics(self, condition_data: Dict[str, List[Tuple[pd.DataFrame, str]]]) -> Dict[str, Any]:
        """Generate statistics for an experiment (multiple conditions)."""
        self.logger.info("Generating experiment statistics")
        
        try:
            condition_stats_list = []
            
            for condition_name, file_data_list in condition_data.items():
                condition_stats = self.generate_condition_statistics(file_data_list, condition_name)
                condition_stats_list.append(condition_stats)
            
            # Generate experiment-wide statistics
            worker = StatisticsWorker(condition_stats_list, self.config, "experiment")
            experiment_stats = worker._generate_experiment_statistics(condition_stats_list)
            
            return experiment_stats
            
        except Exception as e:
            self.logger.error(f"Error generating experiment statistics: {e}")
            raise
    
    def generate_statistics_async(self, data: Any, analysis_type: str = "file"):
        """Generate statistics asynchronously using worker thread."""
        if self.stats_worker and self.stats_worker.isRunning():
            self.logger.warning("Statistics generation already running")
            return
        
        self.stats_worker = StatisticsWorker(data, self.config, analysis_type)
        self.stats_worker.progressUpdate.connect(self.progressUpdate)
        self.stats_worker.statisticsCompleted.connect(self.statisticsGenerated)
        self.stats_worker.errorOccurred.connect(self.errorOccurred)
        
        self.stats_worker.start()
    
    def export_statistics(self, statistics: Any, output_path: str, format_type: str = "excel"):
        """Export statistics to file."""
        self.logger.info(f"Exporting statistics to {output_path}")
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if isinstance(statistics, FileStatistics):
                self._export_file_statistics(statistics, output_path, format_type)
            elif isinstance(statistics, ConditionStatistics):
                self._export_condition_statistics(statistics, output_path, format_type)
            elif isinstance(statistics, dict) and 'conditions' in statistics:
                self._export_experiment_statistics(statistics, output_path, format_type)
            else:
                raise ValueError(f"Unknown statistics type: {type(statistics)}")
            
            self.logger.info(f"Statistics exported successfully to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting statistics: {e}")
            raise
    
    def _export_file_statistics(self, stats: FileStatistics, output_path: Path, format_type: str):
        """Export file statistics."""
        # Convert to dictionary for export
        stats_dict = asdict(stats)
        
        # Flatten nested dictionaries
        if stats_dict['density_metrics']:
            for radius, (mean_val, sem_val) in stats_dict['density_metrics'].items():
                stats_dict[f'density_mean_r{radius}'] = mean_val
                stats_dict[f'density_sem_r{radius}'] = sem_val
            del stats_dict['density_metrics']
        
        # Create DataFrame
        df = pd.DataFrame([stats_dict])
        
        if format_type.lower() == "excel":
            df.to_excel(output_path.with_suffix('.xlsx'), index=False)
        elif format_type.lower() == "csv":
            df.to_csv(output_path.with_suffix('.csv'), index=False)
        elif format_type.lower() == "json":
            with open(output_path.with_suffix('.json'), 'w') as f:
                json.dump(stats_dict, f, indent=2, default=str)
    
    def _export_condition_statistics(self, stats: ConditionStatistics, output_path: Path, format_type: str):
        """Export condition statistics."""
        if format_type.lower() == "excel":
            with pd.ExcelWriter(output_path.with_suffix('.xlsx')) as writer:
                # Summary sheet
                summary_dict = asdict(stats)
                if 'file_statistics' in summary_dict:
                    del summary_dict['file_statistics']  # Remove nested data
                summary_df = pd.DataFrame([summary_dict])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Individual file statistics
                if stats.file_statistics:
                    files_data = []
                    for file_stat in stats.file_statistics:
                        file_dict = asdict(file_stat)
                        # Flatten density metrics
                        if file_dict['density_metrics']:
                            for radius, (mean_val, sem_val) in file_dict['density_metrics'].items():
                                file_dict[f'density_mean_r{radius}'] = mean_val
                                file_dict[f'density_sem_r{radius}'] = sem_val
                            del file_dict['density_metrics']
                        files_data.append(file_dict)
                    
                    files_df = pd.DataFrame(files_data)
                    files_df.to_excel(writer, sheet_name='File_Statistics', index=False)
        
        elif format_type.lower() == "csv":
            # Export summary
            summary_dict = asdict(stats)
            if 'file_statistics' in summary_dict:
                del summary_dict['file_statistics']
            summary_df = pd.DataFrame([summary_dict])
            summary_df.to_csv(output_path.with_suffix('_summary.csv'), index=False)
            
            # Export file statistics
            if stats.file_statistics:
                files_data = []
                for file_stat in stats.file_statistics:
                    file_dict = asdict(file_stat)
                    if file_dict['density_metrics']:
                        for radius, (mean_val, sem_val) in file_dict['density_metrics'].items():
                            file_dict[f'density_mean_r{radius}'] = mean_val
                            file_dict[f'density_sem_r{radius}'] = sem_val
                        del file_dict['density_metrics']
                    files_data.append(file_dict)
                
                files_df = pd.DataFrame(files_data)
                files_df.to_csv(output_path.with_suffix('_files.csv'), index=False)
    
    def _export_experiment_statistics(self, stats: Dict[str, Any], output_path: Path, format_type: str):
        """Export experiment statistics."""
        if format_type.lower() == "excel":
            with pd.ExcelWriter(output_path.with_suffix('.xlsx')) as writer:
                # Experiment summary
                experiment_summary = {
                    'condition_count': stats['condition_count'],
                    'total_files': stats['total_files'],
                    'total_tracks': stats['total_tracks'],
                    'total_localizations': stats['total_localizations']
                }
                exp_df = pd.DataFrame([experiment_summary])
                exp_df.to_excel(writer, sheet_name='Experiment_Summary', index=False)
                
                # Condition comparison
                condition_data = []
                for condition_name, condition_stats in stats['conditions'].items():
                    cond_dict = asdict(condition_stats)
                    if 'file_statistics' in cond_dict:
                        del cond_dict['file_statistics']
                    condition_data.append(cond_dict)
                
                conditions_df = pd.DataFrame(condition_data)
                conditions_df.to_excel(writer, sheet_name='Condition_Comparison', index=False)
                
                # Individual condition details
                for condition_name, condition_stats in stats['conditions'].items():
                    if condition_stats.file_statistics:
                        files_data = []
                        for file_stat in condition_stats.file_statistics:
                            file_dict = asdict(file_stat)
                            if file_dict['density_metrics']:
                                for radius, (mean_val, sem_val) in file_dict['density_metrics'].items():
                                    file_dict[f'density_mean_r{radius}'] = mean_val
                                    file_dict[f'density_sem_r{radius}'] = sem_val
                                del file_dict['density_metrics']
                            files_data.append(file_dict)
                        
                        files_df = pd.DataFrame(files_data)
                        sheet_name = f"{condition_name}_Files"[:31]  # Excel sheet name limit
                        files_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    def generate_comparison_plots(self, experiment_stats: Dict[str, Any], output_dir: str):
        """Generate comparison plots for experiment statistics."""
        self.logger.info("Generating comparison plots")
        
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if not experiment_stats.get('conditions'):
                self.logger.warning("No conditions found for plotting")
                return
            
            # Set up plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Extract data for plotting
            conditions = list(experiment_stats['conditions'].keys())
            mobile_percentages = [stats['conditions'][cond].mean_mobile_percentage 
                                for cond in conditions]
            mobile_sems = [stats['conditions'][cond].sem_mobile_percentage 
                          for cond in conditions]
            linear_percentages = [stats['conditions'][cond].mean_linear_percentage 
                                for cond in conditions]
            linear_sems = [stats['conditions'][cond].sem_linear_percentage 
                          for cond in conditions]
            
            # Create comparison plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Mobile percentage comparison
            ax1.bar(conditions, mobile_percentages, yerr=mobile_sems, capsize=5)
            ax1.set_ylabel('Mobile Percentage (%)')
            ax1.set_title('Mobility Across Conditions')
            ax1.tick_params(axis='x', rotation=45)
            
            # Linear percentage comparison
            ax2.bar(conditions, linear_percentages, yerr=linear_sems, capsize=5, color='orange')
            ax2.set_ylabel('Linear Percentage (%)')
            ax2.set_title('Linearity Across Conditions (Mobile Tracks)')
            ax2.tick_params(axis='x', rotation=45)
            
            # Radius of gyration comparison
            rg_means = [stats['conditions'][cond].mean_radius_gyration for cond in conditions]
            rg_sems = [stats['conditions'][cond].sem_radius_gyration for cond in conditions]
            ax3.bar(conditions, rg_means, yerr=rg_sems, capsize=5, color='green')
            ax3.set_ylabel('Radius of Gyration')
            ax3.set_title('Radius of Gyration Across Conditions')
            ax3.tick_params(axis='x', rotation=45)
            
            # Scaled radius of gyration comparison
            srg_means = [stats['conditions'][cond].mean_scaled_rg for cond in conditions]
            srg_sems = [stats['conditions'][cond].sem_scaled_rg for cond in conditions]
            ax4.bar(conditions, srg_means, yerr=srg_sems, capsize=5, color='red')
            ax4.set_ylabel('Scaled Radius of Gyration')
            ax4.set_title('Scaled Rg Across Conditions')
            ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save plots
            if self.config.plot_format in ['png', 'both']:
                plt.savefig(output_dir / 'condition_comparison.png', dpi=300, bbox_inches='tight')
            if self.config.plot_format in ['pdf', 'both']:
                plt.savefig(output_dir / 'condition_comparison.pdf', bbox_inches='tight')
            
            plt.close()
            
            self.logger.info(f"Comparison plots saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error generating comparison plots: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def generate_summary_report(self, statistics: Any, output_path: str) -> str:
        """Generate a text summary report."""
        self.logger.info("Generating summary report")
        
        try:
            output_path = Path(output_path)
            
            report_lines = [
                "Particle Tracking Analysis Report",
                "=" * 50,
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ""
            ]
            
            if isinstance(statistics, FileStatistics):
                report_lines.extend(self._format_file_report(statistics))
            elif isinstance(statistics, ConditionStatistics):
                report_lines.extend(self._format_condition_report(statistics))
            elif isinstance(statistics, dict) and 'conditions' in statistics:
                report_lines.extend(self._format_experiment_report(statistics))
            
            # Write report
            with open(output_path.with_suffix('.txt'), 'w') as f:
                f.write('\n'.join(report_lines))
            
            self.logger.info(f"Summary report saved to {output_path}")
            
            return '\n'.join(report_lines)
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {e}")
            raise
    
    def _format_file_report(self, stats: FileStatistics) -> List[str]:
        """Format file statistics for text report."""
        lines = [
            f"File: {stats.filename}",
            "-" * 30,
            f"Total Tracks: {stats.total_tracks}",
            f"Total Localizations: {stats.total_localizations}",
            f"Total Frames: {stats.total_frames}",
            "",
            "Track Length Statistics:",
            f"  Mean: {stats.mean_track_length:.2f}",
            f"  Median: {stats.median_track_length:.2f}",
            f"  Std: {stats.std_track_length:.2f}",
            "",
            "Mobility Classification:",
            f"  Mobile Tracks: {stats.mobile_tracks} ({stats.mobile_percentage:.1f}%)",
            f"  Immobile Tracks: {stats.immobile_tracks} ({100-stats.mobile_percentage:.1f}%)",
            "",
            "Linearity Classification (Mobile Tracks):",
            f"  Linear Tracks: {stats.linear_tracks} ({stats.linear_percentage:.1f}%)",
            f"  Non-linear Tracks: {stats.nonlinear_tracks} ({100-stats.linear_percentage:.1f}%)",
            f"    Unidirectional: {stats.unidirectional_tracks}",
            f"    Bidirectional: {stats.bidirectional_tracks}",
            ""
        ]
        
        # Add trajectory metrics if available
        if not np.isnan(stats.mean_radius_gyration):
            lines.extend([
                "Trajectory Metrics:",
                f"  Radius of Gyration: {stats.mean_radius_gyration:.3f} ± {stats.sem_radius_gyration:.3f}",
                f"  Scaled Rg: {stats.mean_scaled_rg:.3f} ± {stats.sem_scaled_rg:.3f}",
                f"  Mean Step Size: {stats.mean_step_size:.3f} ± {stats.sem_step_size:.3f}",
                ""
            ])
        
        # Add quality metrics
        if not np.isnan(stats.tracking_efficiency):
            lines.extend([
                "Quality Metrics:",
                f"  Tracking Efficiency: {stats.tracking_efficiency:.3f}",
                ""
            ])
        
        return lines
    
    def _format_condition_report(self, stats: ConditionStatistics) -> List[str]:
        """Format condition statistics for text report."""
        lines = [
            f"Condition: {stats.condition_name}",
            "-" * 30,
            f"Number of Files: {stats.file_count}",
            f"Total Tracks: {stats.total_tracks}",
            f"Total Localizations: {stats.total_localizations}",
            "",
            "Aggregated Metrics (Mean ± SEM across files):",
            f"  Mobile Percentage: {stats.mean_mobile_percentage:.1f} ± {stats.sem_mobile_percentage:.1f}%",
            f"  Linear Percentage: {stats.mean_linear_percentage:.1f} ± {stats.sem_linear_percentage:.1f}%",
            f"  Radius of Gyration: {stats.mean_radius_gyration:.3f} ± {stats.sem_radius_gyration:.3f}",
            f"  Scaled Rg: {stats.mean_scaled_rg:.3f} ± {stats.sem_scaled_rg:.3f}",
            ""
        ]
        
        # Add individual file summary
        if stats.file_statistics:
            lines.extend([
                "Individual Files:",
                "-" * 20
            ])
            for file_stat in stats.file_statistics:
                lines.append(f"  {file_stat.filename}: {file_stat.total_tracks} tracks, "
                           f"{file_stat.mobile_percentage:.1f}% mobile")
        
        return lines
    
    def _format_experiment_report(self, stats: Dict[str, Any]) -> List[str]:
        """Format experiment statistics for text report."""
        lines = [
            "Experiment Summary",
            "-" * 30,
            f"Number of Conditions: {stats['condition_count']}",
            f"Total Files: {stats['total_files']}",
            f"Total Tracks: {stats['total_tracks']}",
            f"Total Localizations: {stats['total_localizations']}",
            "",
            "Condition Comparison:",
            "-" * 20
        ]
        
        for condition_name, condition_stats in stats['conditions'].items():
            lines.extend([
                f"{condition_name}:",
                f"  Files: {condition_stats.file_count}",
                f"  Tracks: {condition_stats.total_tracks}",
                f"  Mobile: {condition_stats.mean_mobile_percentage:.1f} ± {condition_stats.sem_mobile_percentage:.1f}%",
                f"  Linear: {condition_stats.mean_linear_percentage:.1f} ± {condition_stats.sem_linear_percentage:.1f}%",
                ""
            ])
        
        return lines