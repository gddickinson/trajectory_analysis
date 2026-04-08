#!/usr/bin/env python3
"""
Batch Analysis Module
=====================

Provides hierarchical batch processing capabilities for analyzing multiple files,
conditions, and experiments. This addresses the gap between the original scripts'
comprehensive workflow and the current app's single-file focus.

Key Features:
- Experiment → Conditions → Files hierarchy
- Cross-condition statistical comparisons
- Automated parameter optimization
- Batch export with multiple output formats
- Progress tracking and error handling
- Result archiving and comparison
"""

import logging
import json
import time
from typing import Optional, Dict, List, Any, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

import numpy as np
import pandas as pd
from PyQt6.QtCore import QObject, pyqtSignal, QThread
from tqdm import tqdm

# Import analysis components
from particle_tracker.core.data_manager import EnhancedDataManager as DataManager, DataType
from particle_tracker.core.analysis_engine import AnalysisEngine, AnalysisStep, AnalysisParameters
from particle_tracker.analysis.features import FeatureCalculator
from particle_tracker.analysis.autocorrelation_analysis import DirectionAutocorrelationAnalyzer


@dataclass
class BatchFile:
    """Information about a file in the batch."""
    file_path: str
    condition: str
    experiment: str
    data_type: str = "auto"  # "image", "trajectories", "localizations", "auto"
    parameters: Dict[str, Any] = None
    status: str = "pending"  # "pending", "processing", "completed", "failed"
    results_path: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class BatchExperiment:
    """Information about an experiment containing multiple conditions."""
    experiment_name: str
    description: str = ""
    conditions: List[str] = None
    files: List[BatchFile] = None
    output_directory: str = ""
    parameters: Dict[str, Any] = None
    created_date: str = ""
    completed_date: Optional[str] = None

    def __post_init__(self):
        if self.conditions is None:
            self.conditions = []
        if self.files is None:
            self.files = []
        if self.parameters is None:
            self.parameters = {}
        if not self.created_date:
            from datetime import datetime
            self.created_date = datetime.now().isoformat()


class BatchAnalysisWorker(QThread):
    """Worker thread for batch analysis processing."""

    progressUpdate = pyqtSignal(str, int)  # message, percentage
    fileCompleted = pyqtSignal(str, str, str)  # file_path, status, results_path
    experimentCompleted = pyqtSignal(str)  # experiment_name
    errorOccurred = pyqtSignal(str, str)  # file_path, error_message

    def __init__(self, experiment: BatchExperiment, analysis_engine: AnalysisEngine,
                 data_manager: DataManager, parent=None):
        super().__init__(parent)
        self.experiment = experiment
        self.analysis_engine = analysis_engine
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        self.should_stop = False

    def run(self):
        """Run batch analysis for the experiment."""
        try:
            self.logger.info(f"Starting batch analysis for experiment: {self.experiment.experiment_name}")

            total_files = len(self.experiment.files)
            completed_files = 0

            # Create output directory structure
            self._create_output_structure()

            # Process files
            for file_info in self.experiment.files:
                if self.should_stop:
                    break

                self.progressUpdate.emit(
                    f"Processing {Path(file_info.file_path).name}...",
                    int(100 * completed_files / total_files)
                )

                success, results_path, error_msg = self._process_file(file_info)

                if success:
                    file_info.status = "completed"
                    file_info.results_path = results_path
                    self.fileCompleted.emit(file_info.file_path, "completed", results_path)
                else:
                    file_info.status = "failed"
                    file_info.error_message = error_msg
                    self.errorOccurred.emit(file_info.file_path, error_msg)

                completed_files += 1

            if not self.should_stop:
                # Generate experiment summary
                self._generate_experiment_summary()

                # Update completion date
                from datetime import datetime
                self.experiment.completed_date = datetime.now().isoformat()

                self.progressUpdate.emit("Experiment completed", 100)
                self.experimentCompleted.emit(self.experiment.experiment_name)

        except Exception as e:
            self.logger.error(f"Error in batch analysis: {e}")
            self.errorOccurred.emit("", str(e))

    def stop(self):
        """Stop the batch analysis."""
        self.should_stop = True

    def _create_output_structure(self):
        """Create organized output directory structure."""
        base_path = Path(self.experiment.output_directory)
        base_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for each condition
        for condition in self.experiment.conditions:
            condition_path = base_path / condition
            condition_path.mkdir(exist_ok=True)

            # Create analysis subdirectories
            (condition_path / "individual_results").mkdir(exist_ok=True)
            (condition_path / "plots").mkdir(exist_ok=True)
            (condition_path / "exports").mkdir(exist_ok=True)

        # Create experiment-level directories
        (base_path / "comparisons").mkdir(exist_ok=True)
        (base_path / "summary").mkdir(exist_ok=True)

    def _process_file(self, file_info: BatchFile) -> Tuple[bool, Optional[str], Optional[str]]:
        """Process a single file."""
        try:
            start_time = time.time()

            # Load data
            file_path = Path(file_info.file_path)

            if not file_path.exists():
                return False, None, f"File not found: {file_path}"

            # Determine data type and load
            success = self.data_manager.load_file(str(file_path))
            if not success:
                return False, None, f"Failed to load file: {file_path}"

            data_name = file_path.stem
            data = self.data_manager.get_data(data_name)

            if data is None:
                return False, None, f"No data loaded from: {file_path}"

            # Merge file-specific parameters with experiment parameters
            analysis_params = {**self.experiment.parameters, **file_info.parameters}

            # Determine analysis steps based on data type
            steps = self._determine_analysis_steps(data, analysis_params)

            # Run analysis
            results = self._run_file_analysis(data, analysis_params, steps)

            # Save results
            results_path = self._save_file_results(file_info, results, data_name)

            # Update processing time
            file_info.processing_time = time.time() - start_time

            return True, results_path, None

        except Exception as e:
            error_msg = f"Error processing {file_info.file_path}: {str(e)}"
            self.logger.error(error_msg)
            return False, None, error_msg

    def _determine_analysis_steps(self, data: Any, parameters: Dict[str, Any]) -> List[AnalysisStep]:
        """Determine appropriate analysis steps based on data type."""
        steps = []

        # Check data type and determine pipeline
        if isinstance(data, np.ndarray):
            # Image data - full pipeline
            steps = [
                AnalysisStep.DETECTION,
                AnalysisStep.LINKING,
                AnalysisStep.ENHANCED_FEATURES,
                AnalysisStep.CLASSIFICATION
            ]
        elif isinstance(data, pd.DataFrame):
            if 'track_number' in data.columns:
                # Already has trajectories
                steps = [AnalysisStep.ENHANCED_FEATURES, AnalysisStep.CLASSIFICATION]
            elif 'x' in data.columns and 'y' in data.columns:
                # Localizations data
                steps = [AnalysisStep.LINKING, AnalysisStep.ENHANCED_FEATURES, AnalysisStep.CLASSIFICATION]

        return steps

    def _run_file_analysis(self, data: Any, parameters: Dict[str, Any],
                          steps: List[AnalysisStep]) -> Dict[str, Any]:
        """Run analysis pipeline for a single file."""
        # Convert parameters to AnalysisParameters if needed
        if not isinstance(parameters, AnalysisParameters):
            analysis_params = AnalysisParameters(**{k: v for k, v in parameters.items()
                                                   if k in AnalysisParameters.__dataclass_fields__})
        else:
            analysis_params = parameters

        # Run main analysis pipeline synchronously
        results = {}
        current_data = data

        try:
            # Create analysis engine components
            from particle_tracker.analysis.detection import ParticleDetector
            from particle_tracker.analysis.linking import ParticleLinker
            from particle_tracker.analysis.classification import TrajectoryClassifier

            detector = ParticleDetector(analysis_params.to_dict())
            linker = ParticleLinker(analysis_params.to_dict())
            feature_calculator = FeatureCalculator(analysis_params.to_dict())
            classifier = TrajectoryClassifier(analysis_params.to_dict())

            # Run each step
            for step in steps:
                if step == AnalysisStep.DETECTION:
                    current_data = detector.detect_particles(current_data, method=analysis_params.detection_method)
                    results['detection'] = current_data

                elif step == AnalysisStep.LINKING:
                    current_data = linker.link_particles(current_data, method=analysis_params.linking_method)
                    results['linking'] = current_data

                elif step == AnalysisStep.ENHANCED_FEATURES:
                    current_data = feature_calculator.calculate_features(current_data)
                    results['features'] = current_data

                elif step == AnalysisStep.CLASSIFICATION:
                    method = analysis_params.to_dict().get('classification_method', 'threshold')
                    current_data = classifier.classify_trajectories(current_data, method=method)
                    results['classification'] = current_data

            # Run autocorrelation analysis if trajectory data is available
            if isinstance(current_data, pd.DataFrame) and 'track_number' in current_data.columns:
                if parameters.get('include_autocorrelation', True):
                    autocorr_analyzer = DirectionAutocorrelationAnalyzer(analysis_params.to_dict())
                    autocorr_results = autocorr_analyzer.analyze_all_tracks(current_data)
                    ensemble_autocorr = autocorr_analyzer.calculate_ensemble_autocorrelation(current_data)

                    results['autocorrelation_individual'] = autocorr_results
                    results['autocorrelation_ensemble'] = ensemble_autocorr

            results['final_data'] = current_data

        except Exception as e:
            self.logger.error(f"Error in analysis pipeline: {e}")
            results['error'] = str(e)

        return results

    def _save_file_results(self, file_info: BatchFile, results: Dict[str, Any],
                          data_name: str) -> str:
        """Save analysis results for a single file."""
        # Create file-specific output directory
        condition_path = Path(self.experiment.output_directory) / file_info.condition
        file_output_path = condition_path / "individual_results" / data_name
        file_output_path.mkdir(parents=True, exist_ok=True)

        # Save main results CSV
        if 'final_data' in results and isinstance(results['final_data'], pd.DataFrame):
            csv_path = file_output_path / f"{data_name}_results.csv"
            results['final_data'].to_csv(csv_path, index=False)

        # Save intermediate results
        for step_name, step_data in results.items():
            if step_name in ['detection', 'linking', 'features', 'classification']:
                if isinstance(step_data, pd.DataFrame):
                    step_path = file_output_path / f"{data_name}_{step_name}.csv"
                    step_data.to_csv(step_path, index=False)

        # Save autocorrelation results
        if 'autocorrelation_individual' in results:
            autocorr_path = file_output_path / f"{data_name}_autocorrelation.csv"
            autocorr_analyzer = DirectionAutocorrelationAnalyzer()
            autocorr_analyzer.export_autocorrelation_results(
                results['autocorrelation_individual'],
                results.get('autocorrelation_ensemble', {}),
                str(autocorr_path)
            )

        # Generate analysis summary for this file
        summary = self._generate_file_summary(results, data_name)
        summary_path = file_output_path / f"{data_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        return str(file_output_path)

    def _generate_file_summary(self, results: Dict[str, Any], data_name: str) -> Dict[str, Any]:
        """Generate summary statistics for a single file."""
        summary = {
            'file_name': data_name,
            'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'steps_completed': list(results.keys())
        }

        # Extract key metrics from final data
        if 'final_data' in results and isinstance(results['final_data'], pd.DataFrame):
            data = results['final_data']

            if 'track_number' in data.columns:
                summary['n_tracks'] = data['track_number'].nunique()
                summary['n_localizations'] = len(data)

                # Track length statistics
                track_lengths = data.groupby('track_number').size()
                summary['mean_track_length'] = float(track_lengths.mean())
                summary['median_track_length'] = float(track_lengths.median())

                # Mobility classification
                if 'mobility_classification' in data.columns:
                    mobility_counts = data.groupby('track_number')['mobility_classification'].first().value_counts()
                    total = mobility_counts.sum()
                    summary['mobility_distribution'] = {k: int(v) for k, v in mobility_counts.items()}
                    if 'mobile' in mobility_counts:
                        summary['percent_mobile'] = float(mobility_counts['mobile'] / total * 100)

                # Feature statistics
                feature_cols = ['scaled_rg', 'velocity', 'eigenvalue_ratio', 'radius_gyration']
                for col in feature_cols:
                    if col in data.columns:
                        values = data.groupby('track_number')[col].first().dropna()
                        if len(values) > 0:
                            summary[f'{col}_mean'] = float(values.mean())
                            summary[f'{col}_std'] = float(values.std())

        # Autocorrelation summary
        if 'autocorrelation_individual' in results:
            autocorr_data = results['autocorrelation_individual']
            if len(autocorr_data) > 0:
                pers_lengths = autocorr_data['persistence_length'].dropna()
                if len(pers_lengths) > 0:
                    summary['mean_persistence_length'] = float(pers_lengths.mean())
                    summary['std_persistence_length'] = float(pers_lengths.std())

        return summary

    def _generate_experiment_summary(self):
        """Generate comprehensive experiment summary and comparisons."""
        try:
            base_path = Path(self.experiment.output_directory)
            summary_path = base_path / "summary"

            # Collect all file summaries by condition
            condition_summaries = {}

            for condition in self.experiment.conditions:
                condition_path = base_path / condition / "individual_results"
                condition_files = []

                if condition_path.exists():
                    for file_dir in condition_path.iterdir():
                        if file_dir.is_dir():
                            summary_file = file_dir / f"{file_dir.name}_summary.json"
                            if summary_file.exists():
                                with open(summary_file, 'r') as f:
                                    file_summary = json.load(f)
                                    condition_files.append(file_summary)

                condition_summaries[condition] = condition_files

            # Generate cross-condition comparisons
            comparison_results = self._generate_cross_condition_comparisons(condition_summaries)

            # Save experiment summary
            experiment_summary = {
                'experiment_name': self.experiment.experiment_name,
                'description': self.experiment.description,
                'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'conditions': list(condition_summaries.keys()),
                'condition_summaries': condition_summaries,
                'cross_condition_comparisons': comparison_results,
                'total_files_processed': sum(len(files) for files in condition_summaries.values())
            }

            summary_file = summary_path / "experiment_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(experiment_summary, f, indent=2, default=str)

            # Generate summary plots
            self._generate_summary_plots(condition_summaries, summary_path)

        except Exception as e:
            self.logger.error(f"Error generating experiment summary: {e}")

    def _generate_cross_condition_comparisons(self, condition_summaries: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Generate statistical comparisons between conditions."""
        comparisons = {}

        # Metrics to compare
        metrics = [
            'n_tracks', 'mean_track_length', 'percent_mobile',
            'scaled_rg_mean', 'velocity_mean', 'eigenvalue_ratio_mean',
            'mean_persistence_length'
        ]

        for metric in metrics:
            metric_data = {}

            # Collect data for each condition
            for condition, files in condition_summaries.items():
                values = []
                for file_summary in files:
                    if metric in file_summary and file_summary[metric] is not None:
                        values.append(file_summary[metric])

                if values:
                    metric_data[condition] = {
                        'values': values,
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'n': len(values)
                    }

            # Perform statistical comparisons if we have at least 2 conditions
            if len(metric_data) >= 2:
                comparisons[metric] = self._compare_conditions_statistically(metric_data)

        return comparisons

    def _compare_conditions_statistically(self, metric_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Perform statistical comparison between conditions."""
        try:
            from scipy import stats as scipy_stats

            conditions = list(metric_data.keys())
            comparison_result = {
                'conditions': conditions,
                'condition_stats': metric_data
            }

            # Perform pairwise t-tests if we have exactly 2 conditions
            if len(conditions) == 2:
                cond1, cond2 = conditions
                values1 = metric_data[cond1]['values']
                values2 = metric_data[cond2]['values']

                if len(values1) > 1 and len(values2) > 1:
                    # Welch's t-test (unequal variances)
                    t_stat, p_value = scipy_stats.ttest_ind(values1, values2, equal_var=False)

                    comparison_result['t_test'] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    }

            # Perform ANOVA if we have more than 2 conditions
            elif len(conditions) > 2:
                all_values = [metric_data[cond]['values'] for cond in conditions
                            if len(metric_data[cond]['values']) > 0]

                if len(all_values) > 2 and all(len(vals) > 0 for vals in all_values):
                    f_stat, p_value = scipy_stats.f_oneway(*all_values)

                    comparison_result['anova'] = {
                        'f_statistic': float(f_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    }

            return comparison_result

        except Exception as e:
            self.logger.error(f"Error in statistical comparison: {e}")
            return {'error': str(e)}

    def _generate_summary_plots(self, condition_summaries: Dict[str, List[Dict]], output_path: Path):
        """Generate summary plots for the experiment."""
        try:
            import matplotlib.pyplot as plt
            plt.style.use('default')

            # Plot 1: Track count comparison
            self._plot_condition_comparison(
                condition_summaries, 'n_tracks', 'Number of Tracks',
                output_path / "track_count_comparison.png"
            )

            # Plot 2: Mobility percentage comparison
            self._plot_condition_comparison(
                condition_summaries, 'percent_mobile', 'Percent Mobile (%)',
                output_path / "mobility_comparison.png"
            )

            # Plot 3: Scaled Rg comparison
            self._plot_condition_comparison(
                condition_summaries, 'scaled_rg_mean', 'Scaled Radius of Gyration',
                output_path / "scaled_rg_comparison.png"
            )

            # Plot 4: Persistence length comparison
            self._plot_condition_comparison(
                condition_summaries, 'mean_persistence_length', 'Persistence Length',
                output_path / "persistence_length_comparison.png"
            )

        except Exception as e:
            self.logger.error(f"Error generating summary plots: {e}")

    def _plot_condition_comparison(self, condition_summaries: Dict[str, List[Dict]],
                                 metric: str, ylabel: str, output_path: Path):
        """Plot comparison of a specific metric across conditions."""
        try:
            import matplotlib.pyplot as plt

            data_for_plot = []
            labels = []

            for condition, files in condition_summaries.items():
                values = []
                for file_summary in files:
                    if metric in file_summary and file_summary[metric] is not None:
                        values.append(file_summary[metric])

                if values:
                    data_for_plot.append(values)
                    labels.append(condition)

            if len(data_for_plot) < 2:
                return  # Need at least 2 conditions to compare

            plt.figure(figsize=(10, 6))

            # Box plot
            plt.boxplot(data_for_plot, labels=labels)
            plt.ylabel(ylabel)
            plt.xlabel('Condition')
            plt.title(f'{ylabel} Comparison Across Conditions')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.error(f"Error plotting {metric}: {e}")


class BatchAnalysisManager(QObject):
    """Manager for batch analysis operations."""

    # Signals
    experimentAdded = pyqtSignal(str)  # experiment_name
    batchStarted = pyqtSignal(str)  # experiment_name
    batchCompleted = pyqtSignal(str)  # experiment_name
    progressUpdate = pyqtSignal(str, int)  # message, percentage
    fileCompleted = pyqtSignal(str, str, str)  # file_path, status, results_path
    errorOccurred = pyqtSignal(str, str)  # context, error_message

    def __init__(self, analysis_engine: AnalysisEngine, data_manager: DataManager):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.analysis_engine = analysis_engine
        self.data_manager = data_manager

        # Storage for experiments
        self.experiments: Dict[str, BatchExperiment] = {}
        self.current_worker: Optional[BatchAnalysisWorker] = None

    def create_experiment(self, experiment_name: str, description: str = "",
                         output_directory: str = "") -> BatchExperiment:
        """Create a new batch experiment."""
        if experiment_name in self.experiments:
            raise ValueError(f"Experiment '{experiment_name}' already exists")

        experiment = BatchExperiment(
            experiment_name=experiment_name,
            description=description,
            output_directory=output_directory or str(Path.cwd() / "batch_results" / experiment_name)
        )

        self.experiments[experiment_name] = experiment
        self.experimentAdded.emit(experiment_name)

        self.logger.info(f"Created experiment: {experiment_name}")
        return experiment

    def add_files_to_experiment(self, experiment_name: str, files: List[Tuple[str, str]],
                               condition_parameters: Dict[str, Dict[str, Any]] = None):
        """
        Add files to an experiment.

        Args:
            experiment_name: Name of the experiment
            files: List of (file_path, condition) tuples
            condition_parameters: Optional condition-specific parameters
        """
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        experiment = self.experiments[experiment_name]
        condition_parameters = condition_parameters or {}

        for file_path, condition in files:
            # Add condition to experiment if new
            if condition not in experiment.conditions:
                experiment.conditions.append(condition)

            # Create batch file
            batch_file = BatchFile(
                file_path=file_path,
                condition=condition,
                experiment=experiment_name,
                parameters=condition_parameters.get(condition, {})
            )

            experiment.files.append(batch_file)

        self.logger.info(f"Added {len(files)} files to experiment '{experiment_name}'")

    def set_experiment_parameters(self, experiment_name: str, parameters: Dict[str, Any]):
        """Set global parameters for an experiment."""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        self.experiments[experiment_name].parameters = parameters
        self.logger.info(f"Updated parameters for experiment '{experiment_name}'")

    def run_experiment(self, experiment_name: str):
        """Run batch analysis for an experiment."""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        if self.current_worker and self.current_worker.isRunning():
            raise RuntimeError("Another batch analysis is already running")

        experiment = self.experiments[experiment_name]

        if not experiment.files:
            raise ValueError(f"No files added to experiment '{experiment_name}'")

        # Create and start worker
        self.current_worker = BatchAnalysisWorker(experiment, self.analysis_engine, self.data_manager)

        # Connect signals
        self.current_worker.progressUpdate.connect(self.progressUpdate)
        self.current_worker.fileCompleted.connect(self.fileCompleted)
        self.current_worker.experimentCompleted.connect(self.batchCompleted)
        self.current_worker.errorOccurred.connect(self.errorOccurred)

        self.current_worker.start()
        self.batchStarted.emit(experiment_name)

        self.logger.info(f"Started batch analysis for experiment '{experiment_name}'")

    def stop_current_analysis(self):
        """Stop the currently running analysis."""
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.stop()
            self.current_worker.wait()
            self.logger.info("Stopped batch analysis")

    def get_experiment_status(self, experiment_name: str) -> Dict[str, Any]:
        """Get status information for an experiment."""
        if experiment_name not in self.experiments:
            return {}

        experiment = self.experiments[experiment_name]

        total_files = len(experiment.files)
        completed_files = sum(1 for f in experiment.files if f.status == "completed")
        failed_files = sum(1 for f in experiment.files if f.status == "failed")

        return {
            'experiment_name': experiment_name,
            'total_files': total_files,
            'completed_files': completed_files,
            'failed_files': failed_files,
            'progress_percentage': int(100 * completed_files / total_files) if total_files > 0 else 0,
            'is_completed': experiment.completed_date is not None,
            'conditions': experiment.conditions,
            'output_directory': experiment.output_directory
        }

    def export_experiment_config(self, experiment_name: str, output_path: str) -> bool:
        """Export experiment configuration to JSON."""
        if experiment_name not in self.experiments:
            return False

        try:
            experiment = self.experiments[experiment_name]
            config = asdict(experiment)

            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)

            self.logger.info(f"Exported experiment config to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting experiment config: {e}")
            return False

    def import_experiment_config(self, config_path: str) -> str:
        """Import experiment configuration from JSON."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            experiment = BatchExperiment(**config)
            self.experiments[experiment.experiment_name] = experiment
            self.experimentAdded.emit(experiment.experiment_name)

            self.logger.info(f"Imported experiment: {experiment.experiment_name}")
            return experiment.experiment_name

        except Exception as e:
            self.logger.error(f"Error importing experiment config: {e}")
            raise

    def get_experiment_list(self) -> List[str]:
        """Get list of experiment names."""
        return list(self.experiments.keys())

    def remove_experiment(self, experiment_name: str) -> bool:
        """Remove an experiment."""
        if experiment_name in self.experiments:
            del self.experiments[experiment_name]
            self.logger.info(f"Removed experiment: {experiment_name}")
            return True
        return False

    def duplicate_experiment(self, source_name: str, new_name: str) -> BatchExperiment:
        """Duplicate an existing experiment."""
        if source_name not in self.experiments:
            raise ValueError(f"Source experiment '{source_name}' not found")

        if new_name in self.experiments:
            raise ValueError(f"Experiment '{new_name}' already exists")

        # Deep copy the experiment
        source_exp = self.experiments[source_name]
        new_exp = BatchExperiment(
            experiment_name=new_name,
            description=f"Copy of {source_exp.description}",
            conditions=source_exp.conditions.copy(),
            files=[BatchFile(**asdict(f)) for f in source_exp.files],
            output_directory=str(Path(source_exp.output_directory).parent / new_name),
            parameters=source_exp.parameters.copy()
        )

        # Update experiment name in files
        for file_info in new_exp.files:
            file_info.experiment = new_name
            file_info.status = "pending"  # Reset status
            file_info.results_path = None
            file_info.error_message = None

        self.experiments[new_name] = new_exp
        self.experimentAdded.emit(new_name)

        self.logger.info(f"Duplicated experiment '{source_name}' as '{new_name}'")
        return new_exp


# Convenience functions
def create_batch_experiment(experiment_name: str, files: List[Tuple[str, str]],
                          parameters: Dict[str, Any], output_dir: str,
                          analysis_engine: AnalysisEngine, data_manager: DataManager) -> BatchAnalysisManager:
    """
    Convenience function to create and configure a batch experiment.

    Args:
        experiment_name: Name for the experiment
        files: List of (file_path, condition) tuples
        parameters: Analysis parameters
        output_dir: Output directory
        analysis_engine: Analysis engine instance
        data_manager: Data manager instance

    Returns:
        Configured BatchAnalysisManager
    """
    manager = BatchAnalysisManager(analysis_engine, data_manager)

    experiment = manager.create_experiment(experiment_name, output_directory=output_dir)
    manager.add_files_to_experiment(experiment_name, files)
    manager.set_experiment_parameters(experiment_name, parameters)

    return manager
