#!/usr/bin/env python3
"""
Enhanced Export Manager Module
=============================

Comprehensive export system supporting multiple formats, hierarchical experiment 
processing, cross-condition comparisons, and batch operations.

Features:
- Multiple export formats (CSV, Excel, JSON, TXT)
- Hierarchical experiment organization (experiment → conditions → files)
- Cross-condition statistical comparisons
- Batch export with automated organization
- Split outputs by classification (mobile/immobile, linear/non-linear)
- ROI and background subtraction data integration
- Autocorrelation analysis export
- Comprehensive statistics generation
- Result archiving and comparison
- Multi-radius density analysis export
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import shutil
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd
from PyQt6.QtCore import QObject, pyqtSignal, QThread

# For Excel export with multiple sheets
try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False


@dataclass
class ExportConfiguration:
    """Configuration for export operations."""
    
    # Output formats
    export_csv: bool = True
    export_excel: bool = True
    export_json: bool = False
    export_statistics: bool = True
    
    # Content options
    include_raw_data: bool = True
    include_features: bool = True
    include_classifications: bool = True
    include_individual_tracks: bool = True
    include_roi_data: bool = True
    include_background_data: bool = True
    
    # Organization options
    split_by_classification: bool = True
    split_by_mobility: bool = True
    split_by_linearity: bool = True
    create_condition_summaries: bool = True
    create_experiment_summaries: bool = True
    
    # Analysis-specific options
    export_autocorrelation: bool = True
    export_density_analysis: bool = True
    export_interpolated_tracks: bool = True
    export_localization_precision: bool = True
    
    # File organization
    use_hierarchical_folders: bool = True
    archive_results: bool = False
    timestamp_folders: bool = True
    
    # Quality control
    min_track_length: int = 3
    mobile_only_for_splits: bool = True


class ExportWorker(QThread):
    """Worker thread for export operations."""
    
    progressUpdate = pyqtSignal(str, int)  # message, percentage
    exportCompleted = pyqtSignal(str)      # output_path
    errorOccurred = pyqtSignal(str)        # error_message
    
    def __init__(self, export_manager, export_type, data, config, output_path):
        super().__init__()
        self.export_manager = export_manager
        self.export_type = export_type
        self.data = data
        self.config = config
        self.output_path = output_path
        self.logger = logging.getLogger(__name__)
    
    def run(self):
        """Run the export operation."""
        try:
            if self.export_type == 'single_file':
                self.export_manager._export_single_file_worker(
                    self.data, self.config, self.output_path, self.progressUpdate
                )
            elif self.export_type == 'condition':
                self.export_manager._export_condition_worker(
                    self.data, self.config, self.output_path, self.progressUpdate
                )
            elif self.export_type == 'experiment':
                self.export_manager._export_experiment_worker(
                    self.data, self.config, self.output_path, self.progressUpdate
                )
            elif self.export_type == 'cross_condition':
                self.export_manager._export_cross_condition_worker(
                    self.data, self.config, self.output_path, self.progressUpdate
                )
            
            self.exportCompleted.emit(self.output_path)
            
        except Exception as e:
            self.logger.error(f"Export error: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.errorOccurred.emit(str(e))


class ExportManager(QObject):
    """Main export manager for comprehensive data export."""
    
    exportStarted = pyqtSignal(str)     # export_type
    exportCompleted = pyqtSignal(str)   # output_path
    progressUpdate = pyqtSignal(str, int)  # message, percentage
    errorOccurred = pyqtSignal(str)     # error_message
    
    def __init__(self, data_manager=None, analysis_engine=None, project_manager=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.data_manager = data_manager
        self.analysis_engine = analysis_engine
        self.project_manager = project_manager
        
        # Export worker
        self.export_worker = None
        
        self.logger.info("Export Manager initialized")
    
    def export_single_file(self, data_name: str, config: ExportConfiguration, 
                          output_path: str) -> bool:
        """Export a single file's analysis results."""
        if self.export_worker and self.export_worker.isRunning():
            self.logger.warning("Export already running")
            return False
        
        try:
            data = self.data_manager.get_data(data_name)
            if data is None:
                raise ValueError(f"Data '{data_name}' not found")
            
            self.exportStarted.emit('single_file')
            
            # Create worker
            self.export_worker = ExportWorker(
                self, 'single_file', {data_name: data}, config, output_path
            )
            self.export_worker.progressUpdate.connect(self.progressUpdate)
            self.export_worker.exportCompleted.connect(self.exportCompleted)
            self.export_worker.errorOccurred.connect(self.errorOccurred)
            
            self.export_worker.start()
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting single file export: {e}")
            self.errorOccurred.emit(str(e))
            return False
    
    def export_condition(self, condition_data: Dict[str, Any], config: ExportConfiguration,
                        output_path: str) -> bool:
        """Export results for a complete condition (multiple files)."""
        if self.export_worker and self.export_worker.isRunning():
            self.logger.warning("Export already running")
            return False
        
        try:
            self.exportStarted.emit('condition')
            
            # Create worker
            self.export_worker = ExportWorker(
                self, 'condition', condition_data, config, output_path
            )
            self.export_worker.progressUpdate.connect(self.progressUpdate)
            self.export_worker.exportCompleted.connect(self.exportCompleted)
            self.export_worker.errorOccurred.connect(self.errorOccurred)
            
            self.export_worker.start()
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting condition export: {e}")
            self.errorOccurred.emit(str(e))
            return False
    
    def export_experiment(self, experiment_data: Dict[str, Dict[str, Any]], 
                         config: ExportConfiguration, output_path: str) -> bool:
        """Export results for a complete experiment (multiple conditions)."""
        if self.export_worker and self.export_worker.isRunning():
            self.logger.warning("Export already running")
            return False
        
        try:
            self.exportStarted.emit('experiment')
            
            # Create worker
            self.export_worker = ExportWorker(
                self, 'experiment', experiment_data, config, output_path
            )
            self.export_worker.progressUpdate.connect(self.progressUpdate)
            self.export_worker.exportCompleted.connect(self.exportCompleted)
            self.export_worker.errorOccurred.connect(self.errorOccurred)
            
            self.export_worker.start()
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting experiment export: {e}")
            self.errorOccurred.emit(str(e))
            return False
    
    def export_cross_condition_comparison(self, comparison_data: Dict[str, Any],
                                        config: ExportConfiguration, output_path: str) -> bool:
        """Export cross-condition comparison results."""
        if self.export_worker and self.export_worker.isRunning():
            self.logger.warning("Export already running")
            return False
        
        try:
            self.exportStarted.emit('cross_condition')
            
            # Create worker
            self.export_worker = ExportWorker(
                self, 'cross_condition', comparison_data, config, output_path
            )
            self.export_worker.progressUpdate.connect(self.progressUpdate)
            self.export_worker.exportCompleted.connect(self.exportCompleted)
            self.export_worker.errorOccurred.connect(self.errorOccurred)
            
            self.export_worker.start()
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting cross-condition export: {e}")
            self.errorOccurred.emit(str(e))
            return False
    
    def _export_single_file_worker(self, data_dict: Dict[str, Any], 
                                  config: ExportConfiguration, output_path: str,
                                  progress_callback=None):
        """Worker method for single file export."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if progress_callback:
            progress_callback.emit("Starting single file export...", 0)
        
        data_name, data = next(iter(data_dict.items()))
        
        # Main data export
        if config.export_csv:
            csv_path = output_dir / f"{data_name}.csv"
            data.to_csv(csv_path, index=False)
            self.logger.info(f"Exported CSV: {csv_path}")
        
        if config.export_excel and OPENPYXL_AVAILABLE:
            excel_path = output_dir / f"{data_name}.xlsx"
            self._export_to_excel_single(data, excel_path, data_name, config)
            self.logger.info(f"Exported Excel: {excel_path}")
        
        if progress_callback:
            progress_callback.emit("Exporting classifications...", 30)
        
        # Split by classification if requested
        if config.split_by_classification:
            self._export_classification_splits(data, output_dir, data_name, config)
        
        if progress_callback:
            progress_callback.emit("Generating statistics...", 60)
        
        # Statistics
        if config.export_statistics:
            stats_path = output_dir / f"{data_name}_statistics.csv"
            stats = self._generate_file_statistics(data)
            stats.to_csv(stats_path)
            self.logger.info(f"Exported statistics: {stats_path}")
        
        if progress_callback:
            progress_callback.emit("Exporting analysis-specific data...", 80)
        
        # Analysis-specific exports
        self._export_analysis_specific_data(data, output_dir, data_name, config)
        
        if progress_callback:
            progress_callback.emit("Export complete", 100)
    
    def _export_condition_worker(self, condition_data: Dict[str, Any],
                                config: ExportConfiguration, output_path: str,
                                progress_callback=None):
        """Worker method for condition export."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if progress_callback:
            progress_callback.emit("Starting condition export...", 0)
        
        condition_name = condition_data.get('name', 'condition')
        files_data = condition_data.get('files', {})
        
        # Create condition directory
        condition_dir = output_dir / condition_name
        condition_dir.mkdir(parents=True, exist_ok=True)
        
        # Export individual files
        total_files = len(files_data)
        for i, (file_name, file_data) in enumerate(files_data.items()):
            if progress_callback:
                progress = int(20 + (i / total_files) * 50)
                progress_callback.emit(f"Exporting file {i+1}/{total_files}: {file_name}", progress)
            
            file_dir = condition_dir / 'individual_files' / file_name
            file_dir.mkdir(parents=True, exist_ok=True)
            
            self._export_single_file_worker({file_name: file_data}, config, str(file_dir))
        
        if progress_callback:
            progress_callback.emit("Creating condition summary...", 75)
        
        # Create condition summary
        if config.create_condition_summaries:
            self._create_condition_summary(files_data, condition_dir, condition_name, config)
        
        if progress_callback:
            progress_callback.emit("Condition export complete", 100)
    
    def _export_experiment_worker(self, experiment_data: Dict[str, Dict[str, Any]],
                                 config: ExportConfiguration, output_path: str,
                                 progress_callback=None):
        """Worker method for experiment export."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if progress_callback:
            progress_callback.emit("Starting experiment export...", 0)
        
        # Export each condition
        total_conditions = len(experiment_data)
        for i, (condition_name, condition_data) in enumerate(experiment_data.items()):
            if progress_callback:
                progress = int((i / total_conditions) * 80)
                progress_callback.emit(f"Exporting condition {i+1}/{total_conditions}: {condition_name}", progress)
            
            condition_dir = output_dir / condition_name
            self._export_condition_worker(condition_data, config, str(condition_dir))
        
        if progress_callback:
            progress_callback.emit("Creating experiment summary...", 85)
        
        # Create experiment-level summary
        if config.create_experiment_summaries:
            self._create_experiment_summary(experiment_data, output_dir, config)
        
        if progress_callback:
            progress_callback.emit("Creating cross-condition comparisons...", 95)
        
        # Cross-condition analysis
        self._create_cross_condition_analysis(experiment_data, output_dir, config)
        
        if progress_callback:
            progress_callback.emit("Experiment export complete", 100)
    
    def _export_cross_condition_worker(self, comparison_data: Dict[str, Any],
                                      config: ExportConfiguration, output_path: str,
                                      progress_callback=None):
        """Worker method for cross-condition comparison export."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if progress_callback:
            progress_callback.emit("Starting cross-condition comparison...", 0)
        
        # Export comparison statistics
        if 'statistics' in comparison_data:
            stats_path = output_dir / "cross_condition_statistics.csv"
            comparison_data['statistics'].to_csv(stats_path, index=False)
        
        if progress_callback:
            progress_callback.emit("Exporting comparison plots...", 50)
        
        # Export plots data for visualization
        if 'plots_data' in comparison_data:
            plots_dir = output_dir / "plots_data"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            for plot_name, plot_data in comparison_data['plots_data'].items():
                plot_path = plots_dir / f"{plot_name}.csv"
                plot_data.to_csv(plot_path, index=False)
        
        if progress_callback:
            progress_callback.emit("Cross-condition comparison complete", 100)
    
    def _export_to_excel_single(self, data: pd.DataFrame, excel_path: Path, 
                               data_name: str, config: ExportConfiguration):
        """Export single dataset to Excel with multiple sheets."""
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Main data sheet
            data.to_excel(writer, sheet_name='Data', index=False)
            
            # Summary statistics sheet
            if config.export_statistics:
                stats = self._generate_file_statistics(data)
                stats.to_excel(writer, sheet_name='Statistics')
            
            # Classification splits
            if config.split_by_classification and 'mobility_classification' in data.columns:
                mobile_data = data[data['mobility_classification'] == 'mobile']
                immobile_data = data[data['mobility_classification'] == 'immobile']
                
                if len(mobile_data) > 0:
                    mobile_data.to_excel(writer, sheet_name='Mobile', index=False)
                if len(immobile_data) > 0:
                    immobile_data.to_excel(writer, sheet_name='Immobile', index=False)
                
                # Linear classification for mobile tracks
                if config.split_by_linearity and 'linear_classification' in mobile_data.columns:
                    linear_data = mobile_data[mobile_data['linear_classification'].str.contains('linear', na=False)]
                    nonlinear_data = mobile_data[mobile_data['linear_classification'] == 'non_linear']
                    
                    if len(linear_data) > 0:
                        linear_data.to_excel(writer, sheet_name='Mobile_Linear', index=False)
                    if len(nonlinear_data) > 0:
                        nonlinear_data.to_excel(writer, sheet_name='Mobile_Nonlinear', index=False)
    
    def _export_classification_splits(self, data: pd.DataFrame, output_dir: Path,
                                     data_name: str, config: ExportConfiguration):
        """Export data split by classification."""
        # Mobility classification
        if 'mobility_classification' in data.columns:
            mobile_data = data[data['mobility_classification'] == 'mobile']
            immobile_data = data[data['mobility_classification'] == 'immobile']
            
            if len(mobile_data) > 0:
                mobile_dir = output_dir / 'mobile'
                mobile_dir.mkdir(parents=True, exist_ok=True)
                mobile_data.to_csv(mobile_dir / f"{data_name}_mobile.csv", index=False)
            
            if len(immobile_data) > 0:
                immobile_dir = output_dir / 'immobile'
                immobile_dir.mkdir(parents=True, exist_ok=True)
                immobile_data.to_csv(immobile_dir / f"{data_name}_immobile.csv", index=False)
            
            # Linear classification for mobile tracks
            if config.split_by_linearity and 'linear_classification' in mobile_data.columns:
                self._export_linearity_splits(mobile_data, output_dir, data_name, 'mobile')
    
    def _export_linearity_splits(self, data: pd.DataFrame, output_dir: Path,
                                data_name: str, prefix: str = ''):
        """Export data split by linearity classification."""
        if 'linear_classification' not in data.columns:
            return
        
        # Linear tracks (includes unidirectional, bidirectional, and general linear)
        linear_data = data[data['linear_classification'].str.contains('linear', na=False)]
        nonlinear_data = data[data['linear_classification'] == 'non_linear']
        
        if len(linear_data) > 0:
            linear_dir = output_dir / f'{prefix}_linear' if prefix else output_dir / 'linear'
            linear_dir.mkdir(parents=True, exist_ok=True)
            linear_data.to_csv(linear_dir / f"{data_name}_{prefix}_linear.csv", index=False)
        
        if len(nonlinear_data) > 0:
            nonlinear_dir = output_dir / f'{prefix}_nonlinear' if prefix else output_dir / 'nonlinear'
            nonlinear_dir.mkdir(parents=True, exist_ok=True)
            nonlinear_data.to_csv(nonlinear_dir / f"{data_name}_{prefix}_nonlinear.csv", index=False)
        
        # Further split linear tracks
        unidirectional_data = data[data['linear_classification'] == 'linear_unidirectional']
        bidirectional_data = data[data['linear_classification'] == 'linear_bidirectional']
        
        if len(unidirectional_data) > 0:
            uni_dir = output_dir / f'{prefix}_linear_unidirectional' if prefix else output_dir / 'linear_unidirectional'
            uni_dir.mkdir(parents=True, exist_ok=True)
            unidirectional_data.to_csv(uni_dir / f"{data_name}_{prefix}_linear_unidirectional.csv", index=False)
        
        if len(bidirectional_data) > 0:
            bidir_dir = output_dir / f'{prefix}_linear_bidirectional' if prefix else output_dir / 'linear_bidirectional'
            bidir_dir.mkdir(parents=True, exist_ok=True)
            bidirectional_data.to_csv(bidir_dir / f"{data_name}_{prefix}_linear_bidirectional.csv", index=False)
    
    def _export_analysis_specific_data(self, data: pd.DataFrame, output_dir: Path,
                                      data_name: str, config: ExportConfiguration):
        """Export analysis-specific data (autocorrelation, density, etc.)."""
        analysis_dir = output_dir / 'analysis_specific'
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Multi-radius density analysis
        if config.export_density_analysis:
            density_cols = [col for col in data.columns if 'nnCountInFrame_within_' in col]
            if density_cols:
                density_data = data[['track_number', 'frame', 'x', 'y'] + density_cols]
                density_data.to_csv(analysis_dir / f"{data_name}_density_analysis.csv", index=False)
        
        # Background subtraction data
        if config.export_background_data:
            bg_cols = [col for col in data.columns if 'roi' in col.lower() or 'background' in col.lower()]
            if bg_cols:
                bg_data = data[['track_number', 'frame', 'x', 'y', 'intensity'] + bg_cols]
                bg_data.to_csv(analysis_dir / f"{data_name}_background_data.csv", index=False)
        
        # Interpolated tracks (trapped particles)
        if config.export_interpolated_tracks:
            if 'SVM' in data.columns:
                trapped_data = data[data['SVM'] == 3]  # SVM class 3 = trapped
                if len(trapped_data) > 0:
                    trapped_data.to_csv(analysis_dir / f"{data_name}_trapped_interpolated.csv", index=False)
        
        # Localization precision data
        if config.export_localization_precision:
            precision_cols = [col for col in data.columns if 'localization' in col.lower() or 'precision' in col.lower()]
            if precision_cols:
                precision_data = data[['track_number', 'frame', 'x', 'y'] + precision_cols]
                precision_data.to_csv(analysis_dir / f"{data_name}_localization_precision.csv", index=False)
    
    def _generate_file_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive statistics for a single file."""
        stats = {}
        
        # Basic track statistics
        if 'track_number' in data.columns:
            unique_tracks = data['track_number'].nunique()
            stats['total_tracks'] = unique_tracks
            
            # Track length statistics
            track_lengths = data.groupby('track_number').size()
            stats['mean_track_length'] = track_lengths.mean()
            stats['median_track_length'] = track_lengths.median()
            stats['min_track_length'] = track_lengths.min()
            stats['max_track_length'] = track_lengths.max()
        
        # Frame statistics
        if 'frame' in data.columns:
            stats['total_frames'] = data['frame'].nunique()
            stats['frame_range'] = f"{data['frame'].min()}-{data['frame'].max()}"
        
        # Mobility classification statistics
        if 'mobility_classification' in data.columns:
            mobility_counts = data.groupby('track_number')['mobility_classification'].first().value_counts()
            total_tracks = data['track_number'].nunique()
            
            for classification, count in mobility_counts.items():
                stats[f'{classification}_tracks'] = count
                stats[f'{classification}_percentage'] = (count / total_tracks) * 100
        
        # Linearity classification statistics
        if 'linear_classification' in data.columns:
            linearity_counts = data.groupby('track_number')['linear_classification'].first().value_counts()
            total_tracks = data['track_number'].nunique()
            
            for classification, count in linearity_counts.items():
                stats[f'{classification}_tracks'] = count
                stats[f'{classification}_percentage'] = (count / total_tracks) * 100
        
        # Feature statistics
        feature_columns = ['radius_gyration', 'sRg', 'asymmetry', 'velocity', 'diffusion_coefficient']
        for col in feature_columns:
            if col in data.columns:
                track_values = data.groupby('track_number')[col].first().dropna()
                if len(track_values) > 0:
                    stats[f'{col}_mean'] = track_values.mean()
                    stats[f'{col}_std'] = track_values.std()
                    stats[f'{col}_median'] = track_values.median()
        
        return pd.DataFrame([stats])
    
    def _create_condition_summary(self, files_data: Dict[str, pd.DataFrame],
                                 output_dir: Path, condition_name: str,
                                 config: ExportConfiguration):
        """Create summary statistics for a condition."""
        summary_stats = []
        
        for file_name, file_data in files_data.items():
            file_stats = self._generate_file_statistics(file_data)
            file_stats['file_name'] = file_name
            summary_stats.append(file_stats)
        
        # Combine all file statistics
        condition_summary = pd.concat(summary_stats, ignore_index=True)
        
        # Calculate condition-level aggregates
        numeric_columns = condition_summary.select_dtypes(include=[np.number]).columns
        aggregates = {}
        
        for col in numeric_columns:
            aggregates[f'{col}_mean'] = condition_summary[col].mean()
            aggregates[f'{col}_std'] = condition_summary[col].std()
            aggregates[f'{col}_sem'] = condition_summary[col].sem()
        
        aggregate_row = pd.DataFrame([aggregates])
        aggregate_row['file_name'] = 'CONDITION_AGGREGATE'
        
        # Combine summary with aggregates
        final_summary = pd.concat([condition_summary, aggregate_row], ignore_index=True)
        
        # Export summary
        summary_path = output_dir / f"{condition_name}_condition_summary.csv"
        final_summary.to_csv(summary_path, index=False)
        
        if config.export_excel and OPENPYXL_AVAILABLE:
            excel_path = output_dir / f"{condition_name}_condition_summary.xlsx"
            final_summary.to_excel(excel_path, index=False)
    
    def _create_experiment_summary(self, experiment_data: Dict[str, Dict[str, Any]],
                                  output_dir: Path, config: ExportConfiguration):
        """Create experiment-level summary."""
        experiment_stats = []
        
        for condition_name, condition_data in experiment_data.items():
            # Calculate condition-level statistics
            files_data = condition_data.get('files', {})
            condition_aggregates = self._calculate_condition_aggregates(files_data)
            condition_aggregates['condition_name'] = condition_name
            experiment_stats.append(condition_aggregates)
        
        # Create experiment summary DataFrame
        experiment_summary = pd.DataFrame(experiment_stats)
        
        # Export experiment summary
        summary_path = output_dir / "experiment_summary.csv"
        experiment_summary.to_csv(summary_path, index=False)
        
        if config.export_excel and OPENPYXL_AVAILABLE:
            excel_path = output_dir / "experiment_summary.xlsx"
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                experiment_summary.to_excel(writer, sheet_name='Experiment_Summary', index=False)
                
                # Add individual condition details
                for condition_name, condition_data in experiment_data.items():
                    files_data = condition_data.get('files', {})
                    condition_details = self._create_condition_details(files_data)
                    sheet_name = condition_name[:31]  # Excel sheet name limit
                    condition_details.to_excel(writer, sheet_name=sheet_name, index=False)
    
    def _calculate_condition_aggregates(self, files_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate aggregate statistics for a condition."""
        all_file_stats = []
        
        for file_name, file_data in files_data.items():
            file_stats = self._generate_file_statistics(file_data)
            all_file_stats.append(file_stats)
        
        if not all_file_stats:
            return {}
        
        # Combine all file statistics
        combined_stats = pd.concat(all_file_stats, ignore_index=True)
        
        # Calculate aggregates
        aggregates = {}
        numeric_columns = combined_stats.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            aggregates[f'{col}_mean'] = combined_stats[col].mean()
            aggregates[f'{col}_std'] = combined_stats[col].std()
            aggregates[f'{col}_sem'] = combined_stats[col].sem()
        
        aggregates['total_files'] = len(files_data)
        
        return aggregates
    
    def _create_condition_details(self, files_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create detailed statistics for each file in a condition."""
        file_details = []
        
        for file_name, file_data in files_data.items():
            file_stats = self._generate_file_statistics(file_data)
            file_stats['file_name'] = file_name
            file_details.append(file_stats)
        
        return pd.concat(file_details, ignore_index=True) if file_details else pd.DataFrame()
    
    def _create_cross_condition_analysis(self, experiment_data: Dict[str, Dict[str, Any]],
                                        output_dir: Path, config: ExportConfiguration):
        """Create cross-condition comparison analysis."""
        comparison_dir = output_dir / 'cross_condition_analysis'
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect data for cross-condition comparison
        condition_summaries = {}
        
        for condition_name, condition_data in experiment_data.items():
            files_data = condition_data.get('files', {})
            condition_summary = self._calculate_condition_aggregates(files_data)
            condition_summaries[condition_name] = condition_summary
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame.from_dict(condition_summaries, orient='index')
        comparison_df.index.name = 'condition'
        comparison_df.reset_index(inplace=True)
        
        # Export cross-condition comparison
        comparison_path = comparison_dir / "cross_condition_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        # Create statistical comparison (ANOVA-ready format)
        self._create_statistical_comparison(experiment_data, comparison_dir)
    
    def _create_statistical_comparison(self, experiment_data: Dict[str, Dict[str, Any]],
                                      output_dir: Path):
        """Create statistical comparison data for ANOVA/t-tests."""
        # Prepare data for statistical analysis
        statistical_data = []
        
        for condition_name, condition_data in experiment_data.items():
            files_data = condition_data.get('files', {})
            
            for file_name, file_data in files_data.items():
                if 'track_number' in file_data.columns:
                    # Get track-level data for each metric
                    track_data = file_data.groupby('track_number').first()
                    
                    for _, track in track_data.iterrows():
                        row = {
                            'condition': condition_name,
                            'file': file_name,
                            'track_number': track.name
                        }
                        
                        # Add relevant metrics
                        metrics = ['radius_gyration', 'sRg', 'asymmetry', 'velocity', 
                                 'diffusion_coefficient', 'mobility_classification', 
                                 'linear_classification']
                        
                        for metric in metrics:
                            if metric in track.index:
                                row[metric] = track[metric]
                        
                        statistical_data.append(row)
        
        # Create statistical DataFrame
        stats_df = pd.DataFrame(statistical_data)
        
        # Export for statistical analysis
        stats_path = output_dir / "statistical_analysis_data.csv"
        stats_df.to_csv(stats_path, index=False)
        
        # Create summary statistics by condition
        summary_stats = []
        for condition in stats_df['condition'].unique():
            condition_data = stats_df[stats_df['condition'] == condition]
            
            summary = {'condition': condition, 'n_tracks': len(condition_data)}
            
            # Calculate means and SEMs for numeric columns
            numeric_cols = condition_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'track_number':
                    values = condition_data[col].dropna()
                    if len(values) > 0:
                        summary[f'{col}_mean'] = values.mean()
                        summary[f'{col}_sem'] = values.sem()
            
            summary_stats.append(summary)
        
        summary_df = pd.DataFrame(summary_stats)
        summary_path = output_dir / "condition_summary_statistics.csv"
        summary_df.to_csv(summary_path, index=False)
    
    def create_archive(self, source_dir: str, archive_path: str, 
                      include_raw_data: bool = True) -> bool:
        """Create a compressed archive of export results."""
        try:
            source_path = Path(source_dir)
            archive_path = Path(archive_path)
            
            # Ensure archive has .zip extension
            if archive_path.suffix != '.zip':
                archive_path = archive_path.with_suffix('.zip')
            
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add export metadata
                metadata = {
                    'export_date': datetime.now().isoformat(),
                    'source_directory': str(source_path),
                    'include_raw_data': include_raw_data,
                    'created_by': 'Particle Tracking Analyzer'
                }
                
                zipf.writestr('export_metadata.json', json.dumps(metadata, indent=2))
                
                # Add all files
                for file_path in source_path.rglob('*'):
                    if file_path.is_file():
                        # Skip raw data files if not requested
                        if not include_raw_data and self._is_raw_data_file(file_path):
                            continue
                        
                        arcname = file_path.relative_to(source_path)
                        zipf.write(file_path, arcname)
            
            self.logger.info(f"Created archive: {archive_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating archive: {e}")
            return False
    
    def _is_raw_data_file(self, file_path: Path) -> bool:
        """Determine if a file is raw data (to exclude from archives if requested)."""
        raw_data_patterns = [
            '*_locsID.csv',
            '*_locs.csv',
            '*.tif',
            '*.tiff',
            '*_bin*.tif'
        ]
        
        for pattern in raw_data_patterns:
            if file_path.match(pattern):
                return True
        
        return False
    
    def stop_export(self):
        """Stop the current export operation."""
        if self.export_worker and self.export_worker.isRunning():
            self.export_worker.terminate()
            self.export_worker.wait()
            self.logger.info("Export stopped")
    
    def get_export_templates(self) -> Dict[str, ExportConfiguration]:
        """Get predefined export configuration templates."""
        templates = {
            'basic': ExportConfiguration(
                export_excel=False,
                split_by_classification=False,
                export_autocorrelation=False,
                export_density_analysis=False
            ),
            'comprehensive': ExportConfiguration(
                export_csv=True,
                export_excel=True,
                export_statistics=True,
                split_by_classification=True,
                export_autocorrelation=True,
                export_density_analysis=True,
                archive_results=True
            ),
            'publication': ExportConfiguration(
                export_csv=True,
                export_excel=True,
                export_statistics=True,
                split_by_classification=True,
                create_condition_summaries=True,
                create_experiment_summaries=True,
                export_autocorrelation=True,
                use_hierarchical_folders=True,
                archive_results=True
            ),
            'mobile_only': ExportConfiguration(
                split_by_classification=True,
                split_by_mobility=True,
                split_by_linearity=True,
                mobile_only_for_splits=True,
                export_autocorrelation=True
            )
        }
        
        return templates
