#!/usr/bin/env python3
"""
# ============================================================================
# TRAJECTORY CLASSIFICATION MODULE
# ============================================================================
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
    """Classify trajectories using various methods."""

    def __init__(self, parameters=None):
        self.logger = logging.getLogger(__name__)
        self.parameters = parameters or {}

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

    def _classify_svm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify using SVM."""

        training_data_path = self.parameters.get('svm_training_data')
        if not training_data_path or not Path(training_data_path).exists():
            self.logger.warning("No training data specified for SVM classification")
            # Fall back to threshold classification
            return self._classify_threshold(df)

        try:
            # Load training data
            training_df = pd.read_csv(training_data_path)

            # Prepare features
            feature_columns = self.parameters.get('svm_features', [
                'radius_gyration', 'asymmetry', 'fracDimension',
                'netDispl', 'Straight', 'kurtosis'
            ])

            # Filter to available features
            available_features = [col for col in feature_columns if col in df.columns]

            if not available_features:
                self.logger.warning("No suitable features found for SVM classification")
                return self._classify_threshold(df)

            # Prepare training data
            X_train = training_df[available_features].fillna(0)
            y_train = training_df['Elected_Label'].map({'mobile': 1, 'confined': 2, 'trapped': 3})

            # Prepare test data
            X_test = df.groupby('track_number')[available_features].first().fillna(0)

            # Create and train SVM pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=min(3, len(available_features)))),
                ('svm', SVC(kernel='rbf', C=1.0, gamma='scale'))
            ])

            pipeline.fit(X_train, y_train)

            # Predict
            predictions = pipeline.predict(X_test)

            # Map predictions back to trajectory data
            track_predictions = pd.Series(predictions, index=X_test.index, name='SVM')
            df = df.join(track_predictions, on='track_number')

            self.logger.info(f"SVM classification completed for {len(X_test)} tracks")

        except Exception as e:
            self.logger.error(f"Error in SVM classification: {e}")
            self.logger.info("Falling back to threshold classification")
            return self._classify_threshold(df)

        return df

    def _classify_threshold(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify using simple thresholds."""

        # Mobility classification based on radius of gyration
        mobility_threshold = self.parameters.get('mobility_threshold', 2.11)

        self.logger.info(f"Using mobility threshold: {mobility_threshold}")

        if 'radius_gyration' in df.columns:
            # Calculate scaled radius of gyration
            if 'step_size' in df.columns:
                mean_step = df.groupby('track_number')['step_size'].mean()
                rg = df.groupby('track_number')['radius_gyration'].first()

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
                rg = df.groupby('track_number')['radius_gyration'].first()
                mobility = (rg > mobility_threshold).astype(int) + 1  # 1=immobile, 2=mobile

                mobility_series = pd.Series(mobility, name='mobility_classification')
                df = df.join(mobility_series, on='track_number')

                # Also add human-readable labels
                mobility_labels = mobility.map({1: 'immobile', 2: 'mobile'})
                mobility_labels.name = 'mobility_label'
                df = df.join(mobility_labels, on='track_number')

            self.logger.info(f"Threshold classification completed. Mobile tracks: {(mobility == 2).sum()}, Immobile tracks: {(mobility == 1).sum()}")
        else:
            self.logger.warning("No radius_gyration column found for threshold classification")
            df['mobility_classification'] = 1  # Default to immobile

        return df

    def update_parameters(self, parameters):
        """Update classification parameters."""
        self.parameters.update(parameters)
