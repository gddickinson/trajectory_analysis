#!/usr/bin/env python3
"""
Enhanced SVM Classification with Advanced Metrics
=================================================

This enhanced version includes:
- Scaled radius of gyration (sRg) calculation
- Advanced linearity metrics (eigenvalue ratios, step alignment)
- Multi-round SVM classification workflows
- Better column name mapping and feature handling
- Integration with trajectory analyzer metrics
"""

import logging
import math
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any, Tuple, Union
from abc import ABC, abstractmethod
from pathlib import Path
from scipy import stats, spatial
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics


class TrajectoryClassifier:
    """Enhanced trajectory classifier with advanced metrics and multi-round classification."""

    def __init__(self, parameters=None):
        self.logger = logging.getLogger(__name__)
        self.parameters = parameters or {}

        # Enhanced column name mapping for different naming conventions
        self.column_mappings = {
            # Basic coordinate columns
            'x': ['x', 'x_coord', 'x_coordinate', 'xpos', 'x_position', 'x-position', 'x position'],
            'y': ['y', 'y_coord', 'y_coordinate', 'ypos', 'y_position', 'y-position', 'y position'],
            'frame': ['frame', 'frames', 'frame_number', 'frameno', 'time', 'f', '#frame'],
            'track_number': ['track_number', 'track_id', 'particle', 'trajectory_id', 'track'],
            
            # Feature columns
            'radius_gyration': ['radius_gyration', 'radiusGyration', 'radius_of_gyration', 'rg'],
            'asymmetry': ['asymmetry', 'Asymmetry', 'asym'],
            'fracDimension': ['fracDimension', 'frac_dimension', 'fractal_dimension', 'fd'],
            'netDispl': ['netDispl', 'NetDispl', 'net_displacement', 'net_displ'],
            'Straight': ['Straight', 'straight', 'straightness'],
            'kurtosis': ['kurtosis', 'Kurtosis', 'kurt'],
            'skewness': ['skewness', 'Skewness', 'skew'],
            'velocity': ['velocity', 'Velocity', 'vel', 'speed'],
            'diffusion_coefficient': ['diffusion_coefficient', 'diffusion_coeff', 'D', 'diff_coeff'],
            
            # Advanced metrics
            'mean_step_length': ['mean_step_length', 'meanLag', 'mean_lag', 'step_size'],
            'eigenvalue_ratio': ['eigenvalue_ratio', 'eig_ratio', 'linearity_ratio'],
            'step_alignment': ['step_alignment', 'alignment', 'directional_alignment'],
            'directionality_ratio': ['directionality_ratio', 'dir_ratio', 'net_to_gross_ratio'],
        }

    def classify_trajectories(self, feature_data: pd.DataFrame, 
                            method: str = 'svm', **kwargs) -> pd.DataFrame:
        """Enhanced trajectory classification with multiple methods and workflows.

        Args:
            feature_data: DataFrame containing trajectory features
            method: Classification method ('svm', 'threshold', 'multi_round', 'advanced')
            **kwargs: Additional parameters
        """
        # Merge parameters from initialization and method call
        merged_params = {**self.parameters, **kwargs}
        original_params = self.parameters.copy()
        self.parameters.update(merged_params)

        try:
            if method == 'svm':
                result = self._classify_svm(feature_data)
            elif method == 'threshold':
                result = self._classify_threshold(feature_data)
            elif method == 'multi_round':
                result = self._classify_multi_round(feature_data)
            elif method == 'advanced':
                result = self._classify_advanced(feature_data)
            else:
                raise ValueError(f"Unknown classification method: {method}")

            return result

        finally:
            # Restore original parameters
            self.parameters = original_params

    def _calculate_scaled_rg(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate scaled radius of gyration (sRg) as in Golan & Sherman Nat Comm 2017.
        
        sRg = sqrt(π/2) * Rg / mean_step_size
        """
        self.logger.info("Calculating scaled radius of gyration (sRg)")
        
        # Find required columns
        rg_col = self._find_column(df, 'radius_gyration')
        step_col = self._find_column(df, 'mean_step_length')
        track_col = self._find_column(df, 'track_number')
        
        if not all([rg_col, step_col, track_col]):
            self.logger.warning("Missing required columns for sRg calculation")
            return df
        
        # Calculate sRg for each track
        def calculate_srg_for_track(track_data):
            rg = track_data[rg_col].iloc[0]
            mean_step = track_data[step_col].iloc[0]
            
            if pd.isna(rg) or pd.isna(mean_step) or mean_step == 0:
                return np.nan
            
            srg = np.sqrt(np.pi/2) * rg / mean_step
            return srg
        
        # Apply to each track
        track_srg = df.groupby(track_col).apply(calculate_srg_for_track)
        track_srg.name = 'sRg'
        
        # Map back to original dataframe
        df = df.join(track_srg, on=track_col)
        
        # Classify mobility based on sRg threshold
        mobility_threshold = self.parameters.get('srg_mobility_threshold', 2.22236433588659)
        df['sRg_mobility_classification'] = df['sRg'].apply(
            lambda x: 'mobile' if x > mobility_threshold else 'immobile' if not pd.isna(x) else 'unclassified'
        )
        
        self.logger.info(f"sRg calculated with threshold {mobility_threshold}")
        return df

    def _calculate_advanced_linearity_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced linearity metrics including eigenvalue ratios and step alignment."""
        self.logger.info("Calculating advanced linearity metrics")
        
        track_col = self._find_column(df, 'track_number')
        x_col = self._find_column(df, 'x')
        y_col = self._find_column(df, 'y')
        frame_col = self._find_column(df, 'frame')
        
        if not all([track_col, x_col, y_col, frame_col]):
            self.logger.warning("Missing required columns for linearity analysis")
            return df
        
        def calculate_linearity_for_track(track_data):
            """Calculate linearity metrics for a single track."""
            track_data = track_data.sort_values(frame_col)
            
            # Get coordinates
            xy = track_data[[x_col, y_col]].values
            xy = xy[~np.isnan(xy).any(axis=1)]  # Remove NaN values
            
            if len(xy) < 3:
                return pd.Series({
                    'eigenvalue_ratio': np.nan,
                    'step_alignment': np.nan,
                    'directionality_ratio': np.nan,
                    'linearity_classification': 'unclassified'
                })
            
            # Calculate center and normalized points
            center = np.mean(xy, axis=0)
            normed_points = xy - center[None, :]
            
            # Calculate gyration tensor
            gyration_tensor = np.einsum('im,in->mn', normed_points, normed_points) / len(xy)
            
            # Get eigenvalues and eigenvectors
            eig_values, eig_vectors = np.linalg.eig(gyration_tensor)
            eig_values = np.real(eig_values)
            
            # Sort eigenvalues in descending order
            idx = eig_values.argsort()[::-1]
            eig_values = eig_values[idx]
            eig_vectors = np.real(eig_vectors[:, idx])
            
            # Calculate eigenvalue ratio
            eigenvalue_ratio = eig_values[0] / eig_values[1] if eig_values[1] > 0 else np.inf
            
            # Calculate step alignment with principal axis
            try:
                principal_eigenvector = eig_vectors[:, 0]
                steps = np.diff(xy, axis=0)
                step_norms = np.linalg.norm(steps, axis=1)
                valid_steps = step_norms > 0
                normalized_steps = steps[valid_steps] / step_norms[valid_steps, None]
                
                # Calculate absolute cosine similarity
                cos_angles = np.abs(np.dot(normalized_steps, principal_eigenvector))
                step_alignment = np.mean(cos_angles)
            except (ValueError, ZeroDivisionError, IndexError):
                step_alignment = np.nan
            
            # Calculate directionality ratio
            try:
                net_displacement = np.linalg.norm(xy[-1] - xy[0])
                path_length = np.sum(np.linalg.norm(steps, axis=1))
                directionality_ratio = net_displacement / path_length if path_length > 0 else np.nan
            except:
                directionality_ratio = np.nan
            
            # Classify linearity
            linearity_classification = self._classify_linearity(
                eigenvalue_ratio, step_alignment, directionality_ratio
            )
            
            return pd.Series({
                'eigenvalue_ratio': eigenvalue_ratio,
                'step_alignment': step_alignment,
                'directionality_ratio': directionality_ratio,
                'linearity_classification': linearity_classification
            })
        
        # Calculate for each track
        linearity_metrics = df.groupby(track_col).apply(calculate_linearity_for_track)
        
        # Join back to original dataframe
        for col in linearity_metrics.columns:
            df = df.join(linearity_metrics[col], on=track_col, rsuffix='_temp')
        
        return df

    def _classify_linearity(self, eigenvalue_ratio: float, step_alignment: float, 
                          directionality_ratio: float) -> str:
        """Classify trajectory linearity based on metrics."""
        eig_threshold = self.parameters.get('linear_eigenvalue_ratio_cutoff', 20.0)
        align_threshold = self.parameters.get('linear_step_alignment_cutoff', 0.7)
        dir_threshold = self.parameters.get('linear_directionality_cutoff', 0.7)
        
        if np.isnan(eigenvalue_ratio) or np.isnan(step_alignment):
            return 'unclassified'
        
        # Check for linear motion
        is_linear = (eigenvalue_ratio >= eig_threshold and step_alignment >= align_threshold)
        
        if not is_linear:
            return 'non_linear'
        
        # Distinguish between unidirectional and bidirectional
        if np.isnan(directionality_ratio):
            return 'linear'
        
        if directionality_ratio >= dir_threshold:
            return 'linear_unidirectional'
        else:
            return 'linear_bidirectional'

    def _classify_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced classification combining sRg, SVM, and linearity metrics."""
        self.logger.info("Running advanced classification workflow")
        
        # Step 1: Calculate scaled Rg
        df = self._calculate_scaled_rg(df)
        
        # Step 2: Calculate advanced linearity metrics
        df = self._calculate_advanced_linearity_metrics(df)
        
        # Step 3: Run SVM classification
        df = self._classify_svm(df)
        
        # Step 4: Create combined classification
        df = self._create_combined_classification(df)
        
        return df

    def _create_combined_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a combined classification using multiple metrics."""
        def combine_classifications(row):
            """Combine different classification results."""
            svm_class = row.get('SVM_label', 'unknown')
            srg_class = row.get('sRg_mobility_classification', 'unknown')
            linearity_class = row.get('linearity_classification', 'unknown')
            
            # Priority: SVM > sRg > linearity
            if svm_class in ['mobile', 'confined', 'trapped']:
                base_class = svm_class
            elif srg_class in ['mobile', 'immobile']:
                base_class = srg_class
            else:
                base_class = 'unknown'
            
            # Add linearity modifier for mobile tracks
            if base_class == 'mobile' and linearity_class != 'unknown':
                if linearity_class == 'linear_unidirectional':
                    return 'mobile_linear_uni'
                elif linearity_class == 'linear_bidirectional':
                    return 'mobile_linear_bi'
                elif linearity_class == 'non_linear':
                    return 'mobile_nonlinear'
                else:
                    return 'mobile_linear'
            
            return base_class
        
        df['combined_classification'] = df.apply(combine_classifications, axis=1)
        return df

    def _classify_multi_round(self, df: pd.DataFrame) -> pd.DataFrame:
        """Multi-round SVM classification (SVM-3, then SVM-2, etc.)."""
        self.logger.info("Running multi-round SVM classification")
        
        # Start with full SVM classification
        df = self._classify_svm(df)
        
        if 'SVM_prediction' not in df.columns:
            self.logger.warning("SVM classification failed, cannot proceed with multi-round")
            return df
        
        # Round 1: Extract SVM class 3 (trapped)
        svm3_tracks = df[df['SVM_prediction'] == 3]['track_number'].unique()
        df['classification_round_1'] = df['SVM_prediction'].map({
            1: 'mobile', 2: 'confined', 3: 'trapped'
        })
        
        # Round 2: Re-classify remaining particles (mobile + confined)
        remaining_df = df[~df['track_number'].isin(svm3_tracks)]
        
        if len(remaining_df) > 0:
            # Re-run SVM on remaining data
            remaining_classified = self._classify_svm(remaining_df.copy())
            
            if 'SVM_prediction' in remaining_classified.columns:
                # Extract new SVM class 2 (confined)
                svm2_tracks = remaining_classified[remaining_classified['SVM_prediction'] == 2]['track_number'].unique()
                
                # Update round 2 classification
                df.loc[df['track_number'].isin(svm2_tracks), 'classification_round_2'] = 'confined'
                df.loc[df['track_number'].isin(svm3_tracks), 'classification_round_2'] = 'trapped'
                
                # Remaining are mobile
                remaining_tracks = df[~df['track_number'].isin(list(svm2_tracks) + list(svm3_tracks))]['track_number'].unique()
                df.loc[df['track_number'].isin(remaining_tracks), 'classification_round_2'] = 'mobile'
        
        # Create final multi-round classification
        df['multi_round_classification'] = df.get('classification_round_2', df.get('classification_round_1', 'unknown'))
        
        return df

    def _classify_svm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced SVM classification with better feature mapping."""
        training_data_path = self.parameters.get('svm_training_data')
        if not training_data_path or not Path(training_data_path).exists():
            self.logger.warning("No training data specified for SVM classification")
            return self._classify_threshold(df)

        try:
            # Load training data
            self.logger.info(f"Loading training data from: {training_data_path}")
            training_df = pd.read_csv(training_data_path)
            
            # Enhanced label column detection
            label_columns = [
                'Elected_Label', 'elected_label', 'label', 'Label', 'class', 'Class',
                'classification', 'Classification', 'target', 'Target'
            ]
            label_column = self._find_column(training_df, 'label', label_columns)
            
            if label_column is None:
                self.logger.error("No label column found in training data")
                return self._classify_threshold(df)

            # Enhanced feature selection
            requested_features = self.parameters.get('svm_features', [
                'radius_gyration', 'asymmetry', 'fracDimension',
                'netDispl', 'Straight', 'kurtosis'
            ])
            
            # Map features between datasets
            feature_mapping = self._create_feature_mapping(df, training_df, requested_features)
            
            if not feature_mapping:
                self.logger.warning("No features could be mapped between datasets")
                return self._classify_threshold(df)

            # Prepare training data
            training_features = [pair[1] for pair in feature_mapping]
            X_train = training_df[training_features].fillna(0)

            # Enhanced label mapping
            y_train = self._map_labels(training_df[label_column])
            
            # Remove invalid labels
            valid_labels = ~y_train.isna()
            X_train = X_train[valid_labels]
            y_train = y_train[valid_labels]

            if len(X_train) == 0:
                self.logger.error("No valid training samples after label mapping")
                return self._classify_threshold(df)

            # Prepare test data
            test_features = [pair[0] for pair in feature_mapping]
            
            # Group by track for prediction
            track_col = self._find_column(df, 'track_number')
            if track_col:
                X_test = df.groupby(track_col)[test_features].first().fillna(0)
            else:
                X_test = df[test_features].fillna(0)

            # Enhanced SVM pipeline
            n_components = min(len(feature_mapping), len(X_train), 6)
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=n_components)),
                ('svm', SVC(
                    kernel='rbf', 
                    C=self.parameters.get('svm_c', 10.0),
                    gamma=self.parameters.get('svm_gamma', 'scale'),
                    probability=True
                ))
            ])

            # Train and predict
            self.logger.info("Training enhanced SVM model...")
            pipeline.fit(X_train, y_train)
            
            predictions = pipeline.predict(X_test)
            probabilities = pipeline.predict_proba(X_test)

            # Map predictions back to trajectory data
            if track_col:
                track_predictions = pd.Series(predictions, index=X_test.index, name='SVM_prediction')
                df = df.join(track_predictions, on=track_col)

                max_probs = np.max(probabilities, axis=1)
                track_probs = pd.Series(max_probs, index=X_test.index, name='SVM_confidence')
                df = df.join(track_probs, on=track_col)
            else:
                df['SVM_prediction'] = predictions
                df['SVM_confidence'] = np.max(probabilities, axis=1)

            # Enhanced label mapping
            label_map = {1: 'mobile', 2: 'confined', 3: 'trapped'}
            if track_col:
                track_labels = df.groupby(track_col)['SVM_prediction'].first().map(label_map)
                track_labels.name = 'SVM_label'
                df = df.join(track_labels, on=track_col)
            else:
                df['SVM_label'] = df['SVM_prediction'].map(label_map)

            self.logger.info(f"SVM classification completed for {len(X_test)} tracks")
            
            # Log detailed results
            pred_counts = pd.Series(predictions).value_counts().sort_index()
            for pred, count in pred_counts.items():
                label = label_map.get(pred, pred)
                self.logger.info(f"  {label}: {count} tracks")

        except Exception as e:
            self.logger.error(f"Error in enhanced SVM classification: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return self._classify_threshold(df)

        return df

    def _map_labels(self, labels: pd.Series) -> pd.Series:
        """Enhanced label mapping for various formats."""
        label_mapping = {
            # String labels
            'mobile': 1, 'Mobile': 1, 'MOBILE': 1,
            'confined': 2, 'Confined': 2, 'CONFINED': 2,
            'trapped': 3, 'Trapped': 3, 'TRAPPED': 3,
            'immobile': 2, 'Immobile': 2, 'IMMOBILE': 2,
            
            # Numeric labels
            1: 1, 2: 2, 3: 3,
            '1': 1, '2': 2, '3': 3,
            
            # Alternative naming
            'free': 1, 'Free': 1,
            'slow': 2, 'Slow': 2,
            'static': 3, 'Static': 3,
        }
        
        return labels.map(label_mapping)

    def _create_feature_mapping(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                               requested_features: List[str]) -> List[Tuple[str, str]]:
        """Enhanced feature mapping between datasets."""
        mapping = []
        
        for feature in requested_features:
            df1_col = self._find_column(df1, feature)
            df2_col = self._find_column(df2, feature)
            
            if df1_col and df2_col:
                mapping.append((df1_col, df2_col))
                self.logger.info(f"Mapped feature: {df1_col} <-> {df2_col}")
        
        return mapping

    def _find_column(self, df: pd.DataFrame, standard_name: str, 
                    variants: Optional[List[str]] = None) -> Optional[str]:
        """Find a column in DataFrame using various naming conventions."""
        if variants is None:
            variants = self.column_mappings.get(standard_name, [standard_name])
        
        # Create lowercase mapping
        cols_lower = {col.lower(): col for col in df.columns}
        
        # Look for variations
        for variant in variants:
            if variant.lower() in cols_lower:
                return cols_lower[variant.lower()]
        
        return None

    def _classify_threshold(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced threshold classification with sRg."""
        self.logger.info("Running enhanced threshold classification")
        
        # Calculate sRg if not already present
        if 'sRg' not in df.columns:
            df = self._calculate_scaled_rg(df)
        
        # Use sRg threshold if available
        if 'sRg' in df.columns:
            srg_threshold = self.parameters.get('srg_mobility_threshold', 2.22236433588659)
            
            track_col = self._find_column(df, 'track_number')
            if track_col:
                track_srg = df.groupby(track_col)['sRg'].first()
                mobility = (track_srg > srg_threshold).astype(int) + 1  # 1=immobile, 2=mobile
                
                mobility_series = pd.Series(mobility, name='threshold_classification')
                df = df.join(mobility_series, on=track_col)
                
                mobility_labels = mobility.map({1: 'immobile', 2: 'mobile'})
                mobility_labels.name = 'threshold_label'
                df = df.join(mobility_labels, on=track_col)
            else:
                mobility = (df['sRg'] > srg_threshold).astype(int) + 1
                df['threshold_classification'] = mobility
                df['threshold_label'] = mobility.map({1: 'immobile', 2: 'mobile'})
        
        else:
            # Fallback to basic Rg threshold
            rg_threshold = self.parameters.get('mobility_threshold', 2.11)
            rg_col = self._find_column(df, 'radius_gyration')
            
            if rg_col:
                track_col = self._find_column(df, 'track_number')
                if track_col:
                    rg = df.groupby(track_col)[rg_col].first()
                    mobility = (rg > rg_threshold).astype(int) + 1
                    
                    mobility_series = pd.Series(mobility, name='threshold_classification')
                    df = df.join(mobility_series, on=track_col)
                    
                    mobility_labels = mobility.map({1: 'immobile', 2: 'mobile'})
                    mobility_labels.name = 'threshold_label'
                    df = df.join(mobility_labels, on=track_col)
        
        return df

    def export_classification_summary(self, df: pd.DataFrame, output_path: str) -> bool:
        """Export detailed classification summary."""
        try:
            summary_lines = [
                "Classification Summary Report",
                "=" * 40,
                "",
                f"Total tracks analyzed: {df['track_number'].nunique()}",
                ""
            ]
            
            # SVM classification summary
            if 'SVM_label' in df.columns:
                svm_counts = df.groupby('track_number')['SVM_label'].first().value_counts()
                summary_lines.extend([
                    "SVM Classification:",
                    "-" * 20
                ])
                for label, count in svm_counts.items():
                    pct = (count / len(svm_counts)) * 100
                    summary_lines.append(f"  {label}: {count} tracks ({pct:.1f}%)")
                summary_lines.append("")
            
            # sRg classification summary
            if 'sRg_mobility_classification' in df.columns:
                srg_counts = df.groupby('track_number')['sRg_mobility_classification'].first().value_counts()
                summary_lines.extend([
                    "sRg Mobility Classification:",
                    "-" * 30
                ])
                for label, count in srg_counts.items():
                    pct = (count / len(srg_counts)) * 100
                    summary_lines.append(f"  {label}: {count} tracks ({pct:.1f}%)")
                summary_lines.append("")
            
            # Linearity classification summary
            if 'linearity_classification' in df.columns:
                linearity_counts = df.groupby('track_number')['linearity_classification'].first().value_counts()
                summary_lines.extend([
                    "Linearity Classification:",
                    "-" * 25
                ])
                for label, count in linearity_counts.items():
                    pct = (count / len(linearity_counts)) * 100
                    summary_lines.append(f"  {label}: {count} tracks ({pct:.1f}%)")
                summary_lines.append("")
            
            # Feature statistics
            feature_cols = ['sRg', 'eigenvalue_ratio', 'step_alignment', 'directionality_ratio']
            available_features = [col for col in feature_cols if col in df.columns]
            
            if available_features:
                summary_lines.extend([
                    "Feature Statistics (per track):",
                    "-" * 30
                ])
                
                for feature in available_features:
                    track_features = df.groupby('track_number')[feature].first().dropna()
                    if len(track_features) > 0:
                        mean_val = track_features.mean()
                        std_val = track_features.std()
                        summary_lines.append(f"  {feature}: {mean_val:.3f} ± {std_val:.3f}")
                summary_lines.append("")
            
            # Write summary
            with open(output_path, 'w') as f:
                f.write('\n'.join(summary_lines))
            
            self.logger.info(f"Classification summary exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting classification summary: {e}")
            return False

    def update_parameters(self, parameters):
        """Update classification parameters."""
        self.parameters.update(parameters)
        self.logger.info("Classification parameters updated")
