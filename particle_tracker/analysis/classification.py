#!/usr/bin/env python3
"""
SVM Classification with Column Name Mapping
================================================

This version handles different column naming conventions between
training data and analysis results.
"""

import logging
import math
from typing import Optional, Dict, List, Any, Tuple, Union
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy import stats, spatial, ndimage
from sklearn.neighbors import KDTree
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics
import skimage.io as skio
from skimage import filters, feature, segmentation, measure
from skimage.morphology import disk, white_tophat
from pathlib import Path
from tqdm import tqdm


class TrajectoryClassifier:
    """Classify trajectories using various methods with flexible column mapping."""

    def __init__(self, parameters=None):
        self.logger = logging.getLogger(__name__)
        self.parameters = parameters or {}

        # Column name mapping for different naming conventions
        self.column_mappings = {
            # Format: 'standard_name': ['possible_variant1', 'possible_variant2', ...]
            'radius_gyration': ['radius_gyration', 'radiusGyration', 'radius_of_gyration', 'rg'],
            'asymmetry': ['asymmetry', 'Asymmetry', 'asym'],
            'fracDimension': ['fracDimension', 'frac_dimension', 'fractal_dimension', 'fd'],
            'netDispl': ['netDispl', 'NetDispl', 'net_displacement', 'net_displ'],
            'Straight': ['Straight', 'straight', 'straightness'],
            'kurtosis': ['kurtosis', 'Kurtosis', 'kurt'],
            'skewness': ['skewness', 'Skewness', 'skew'],
            'velocity': ['velocity', 'Velocity', 'vel', 'speed'],
            'diffusion_coefficient': ['diffusion_coefficient', 'diffusion_coeff', 'D', 'diff_coeff']
        }

    def classify_trajectories(self, feature_data: pd.DataFrame,
                            method: str = 'svm', **kwargs) -> pd.DataFrame:
        """Classify trajectories.

        Args:
            feature_data: DataFrame containing trajectory features
            method: Classification method ('svm' or 'threshold')
            **kwargs: Additional parameters (will be merged with self.parameters)
        """

        # Merge parameters from initialization and method call
        merged_params = {**self.parameters, **kwargs}

        # Temporarily update parameters for this classification
        original_params = self.parameters.copy()
        self.parameters.update(merged_params)

        try:
            if method == 'svm':
                result = self._classify_svm(feature_data)
            elif method == 'threshold':
                result = self._classify_threshold(feature_data)
            else:
                raise ValueError(f"Unknown classification method: {method}")

            return result

        finally:
            # Restore original parameters
            self.parameters = original_params

    def _map_column_names(self, df_columns: List[str], training_columns: List[str]) -> Dict[str, str]:
        """Create mapping between DataFrame columns and training data columns."""

        mapping = {}

        self.logger.info(f"Available feature columns: {df_columns}")
        self.logger.info(f"Training data columns: {training_columns}")

        # For each standard feature name, find matching columns in both datasets
        for standard_name, variants in self.column_mappings.items():
            df_match = None
            training_match = None

            # Find matching column in DataFrame
            for variant in variants:
                if variant in df_columns:
                    df_match = variant
                    break

            # Find matching column in training data
            for variant in variants:
                if variant in training_columns:
                    training_match = variant
                    break

            if df_match and training_match:
                mapping[df_match] = training_match
                self.logger.info(f"Mapped: {df_match} -> {training_match}")

        return mapping

    def _classify_svm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify using SVM with flexible column mapping."""

        training_data_path = self.parameters.get('svm_training_data')
        if not training_data_path or not Path(training_data_path).exists():
            self.logger.warning("No training data specified for SVM classification")
            # Fall back to threshold classification
            return self._classify_threshold(df)

        try:
            # Load training data
            self.logger.info(f"Loading training data from: {training_data_path}")
            training_df = pd.read_csv(training_data_path)
            self.logger.info(f"Training data shape: {training_df.shape}")
            self.logger.info(f"Training data columns: {list(training_df.columns)}")

            # Check for required label column
            label_columns = ['Elected_Label', 'elected_label', 'label', 'Label', 'class', 'Class']
            label_column = None
            for col in label_columns:
                if col in training_df.columns:
                    label_column = col
                    break

            if label_column is None:
                self.logger.error("No label column found in training data")
                return self._classify_threshold(df)

            # Get feature columns from parameters
            requested_features = self.parameters.get('svm_features', [
                'radius_gyration', 'asymmetry', 'fracDimension',
                'netDispl', 'Straight', 'kurtosis'
            ])

            # Create column mapping
            column_mapping = self._map_column_names(df.columns.tolist(), training_df.columns.tolist())

            if not column_mapping:
                self.logger.warning("No feature columns could be mapped between datasets")
                return self._classify_threshold(df)

            # Filter requested features to only those we can map
            available_feature_pairs = []
            for requested_feature in requested_features:
                # Look for this feature in our DataFrame
                df_column = None
                for df_col, training_col in column_mapping.items():
                    if any(variant == requested_feature for variant in self.column_mappings.get(requested_feature, [requested_feature])):
                        if any(variant == df_col for variant in self.column_mappings.get(requested_feature, [requested_feature])):
                            df_column = df_col
                            training_column = training_col
                            available_feature_pairs.append((df_column, training_column))
                            break

            if not available_feature_pairs:
                self.logger.warning("No requested features available in both datasets")
                # Try using any available mapped features
                available_feature_pairs = list(column_mapping.items())[:6]  # Use first 6 available

            if not available_feature_pairs:
                self.logger.warning("No features available for SVM classification")
                return self._classify_threshold(df)

            self.logger.info(f"Using {len(available_feature_pairs)} features for SVM classification:")
            for df_col, training_col in available_feature_pairs:
                self.logger.info(f"  {df_col} <-> {training_col}")

            # Prepare training data
            training_feature_columns = [pair[1] for pair in available_feature_pairs]
            X_train = training_df[training_feature_columns].fillna(0)

            # Map labels to numbers
            y_train = training_df[label_column].map({
                'mobile': 1, 'confined': 2, 'trapped': 3,
                'Mobile': 1, 'Confined': 2, 'Trapped': 3,
                1: 1, 2: 2, 3: 3  # In case already numeric
            })

            # Remove any unmapped labels
            valid_labels = ~y_train.isna()
            X_train = X_train[valid_labels]
            y_train = y_train[valid_labels]

            if len(X_train) == 0:
                self.logger.error("No valid training samples after label mapping")
                return self._classify_threshold(df)

            self.logger.info(f"Training with {len(X_train)} samples")
            self.logger.info(f"Label distribution: {y_train.value_counts().to_dict()}")

            # Prepare test data - group by track first
            df_feature_columns = [pair[0] for pair in available_feature_pairs]

            if 'track_number' in df.columns:
                X_test = df.groupby('track_number')[df_feature_columns].first().fillna(0)
            else:
                self.logger.warning("No track_number column found, using all data points")
                X_test = df[df_feature_columns].fillna(0)

            self.logger.info(f"Test data shape: {X_test.shape}")

            # Create and train SVM pipeline
            n_components = min(len(available_feature_pairs), len(X_train), 3)
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=n_components)),
                ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True))
            ])

            self.logger.info("Training SVM model...")
            pipeline.fit(X_train, y_train)

            # Predict
            self.logger.info("Making predictions...")
            predictions = pipeline.predict(X_test)
            probabilities = pipeline.predict_proba(X_test)

            # Map predictions back to trajectory data
            if 'track_number' in df.columns:
                track_predictions = pd.Series(predictions, index=X_test.index, name='SVM_prediction')
                df = df.join(track_predictions, on='track_number')

                # Also add probabilities for the predicted class
                max_probs = np.max(probabilities, axis=1)
                track_probs = pd.Series(max_probs, index=X_test.index, name='SVM_confidence')
                df = df.join(track_probs, on='track_number')
            else:
                df['SVM_prediction'] = predictions
                df['SVM_confidence'] = np.max(probabilities, axis=1)

            # Add human-readable labels
            label_map = {1: 'mobile', 2: 'confined', 3: 'trapped'}
            if 'track_number' in df.columns:
                track_labels = df.groupby('track_number')['SVM_prediction'].first().map(label_map)
                track_labels.name = 'SVM_label'
                df = df.join(track_labels, on='track_number')
            else:
                df['SVM_label'] = df['SVM_prediction'].map(label_map)

            self.logger.info(f"SVM classification completed for {len(X_test)} tracks")

            # Log prediction distribution
            pred_counts = pd.Series(predictions).value_counts().sort_index()
            self.logger.info(f"Prediction distribution: {pred_counts.to_dict()}")

        except Exception as e:
            self.logger.error(f"Error in SVM classification: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.logger.info("Falling back to threshold classification")
            return self._classify_threshold(df)

        return df

    def _classify_threshold(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify using simple thresholds with flexible column names."""

        # Mobility classification based on radius of gyration
        mobility_threshold = self.parameters.get('mobility_threshold', 2.11)

        self.logger.info(f"Using mobility threshold: {mobility_threshold}")

        # Find radius of gyration column
        rg_column = None
        for variant in self.column_mappings['radius_gyration']:
            if variant in df.columns:
                rg_column = variant
                break

        if rg_column:
            self.logger.info(f"Using radius of gyration column: {rg_column}")

            # Calculate scaled radius of gyration
            if 'step_size' in df.columns:
                mean_step = df.groupby('track_number')['step_size'].mean()
                rg = df.groupby('track_number')[rg_column].first()

                scaled_rg = np.sqrt(np.pi/2) * rg / mean_step
                mobility = (scaled_rg > mobility_threshold).astype(int) + 1  # 1=immobile, 2=mobile

                mobility_series = pd.Series(mobility, name='mobility_classification')
                df = df.join(mobility_series, on='track_number')

                # Also add human-readable labels
                mobility_labels = mobility.map({1: 'immobile', 2: 'mobile'})
                mobility_labels.name = 'mobility_label'
                df = df.join(mobility_labels, on='track_number')

            else:
                # Simpler classification based on radius of gyration directly
                if 'track_number' in df.columns:
                    rg = df.groupby('track_number')[rg_column].first()
                    mobility = (rg > mobility_threshold).astype(int) + 1  # 1=immobile, 2=mobile

                    mobility_series = pd.Series(mobility, name='mobility_classification')
                    df = df.join(mobility_series, on='track_number')

                    # Also add human-readable labels
                    mobility_labels = mobility.map({1: 'immobile', 2: 'mobile'})
                    mobility_labels.name = 'mobility_label'
                    df = df.join(mobility_labels, on='track_number')
                else:
                    # No track grouping available
                    mobility = (df[rg_column] > mobility_threshold).astype(int) + 1
                    df['mobility_classification'] = mobility
                    df['mobility_label'] = mobility.map({1: 'immobile', 2: 'mobile'})

            mobile_count = (mobility == 2).sum() if 'mobility' in locals() else 0
            immobile_count = (mobility == 1).sum() if 'mobility' in locals() else 0
            self.logger.info(f"Threshold classification completed. Mobile tracks: {mobile_count}, Immobile tracks: {immobile_count}")
        else:
            self.logger.warning("No radius of gyration column found for threshold classification")
            df['mobility_classification'] = 1  # Default to immobile
            df['mobility_label'] = 'immobile'

        return df

    def update_parameters(self, parameters):
        """Update classification parameters."""
        self.parameters.update(parameters)
