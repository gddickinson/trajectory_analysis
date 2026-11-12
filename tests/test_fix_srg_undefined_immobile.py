"""Regression tests (fixed 2026-11): tracks whose sRg cannot be computed were called 'immobile', and the threshold
classifier compared the raw radius of gyration (pixels) with the dimensionless sRg threshold.

- AdvancedMetricsCalculator: a one-point track got mobility_sRg = 'immobile' (now 'unclassified').
- TrajectoryClassifier threshold method: a NaN sRg became 'immobile'; and without a mean-step column it compared
  the raw radius of gyration (pixels) with the sRg threshold 2.11 instead of computing sRg from x / y.
See SESSION_LOG.md, "Bugs fixed".
"""
import unittest

import numpy as np
import pandas as pd

from particle_tracker.analysis.advanced_metrics import AdvancedMetricsCalculator
from particle_tracker.analysis.classification import TrajectoryClassifier
from particle_tracker.analysis.features import FeatureCalculator


class TestUndefinedSrgNotImmobile(unittest.TestCase):
    def test_advanced_metrics_single_point(self):
        df = pd.DataFrame({'track_number': [1, 2, 2, 2], 'frame': [0, 0, 1, 2],
                           'x': [5.0, 1.0, 2.0, 1.5], 'y': [5.0, 1.0, 1.0, 2.0]})
        out = AdvancedMetricsCalculator()._calculate_scaled_radius_of_gyration(df)
        self.assertNotEqual(out.loc[out.track_number == 1, 'mobility_sRg'].iloc[0], 'immobile')

    def test_threshold_classifier_nan_srg(self):
        df = pd.DataFrame({'track_number': [1, 1, 2, 2], 'radius_gyration': [0.0, 0.0, 3.0, 3.0],
                           'meanLag': [0.0, 0.0, 1.0, 1.0]})
        out = TrajectoryClassifier().classify_trajectories(df, method='threshold')
        self.assertNotEqual(out.loc[out.track_number == 1, 'threshold_label'].iloc[0], 'immobile')

    def test_threshold_classifier_feature_output(self):
        # A stationary particle with big localisation noise: Rg ~ 3 px (> 2.11) but sRg ~ 1 (immobile).
        rng = np.random.default_rng(0)
        xy = 20 + 2.2 * rng.normal(size=(200, 2))
        df = pd.DataFrame({'track_number': 1, 'frame': np.arange(200), 'x': xy[:, 0], 'y': xy[:, 1]})
        feats = FeatureCalculator().calculate_features(df)
        self.assertGreater(feats['radius_gyration'].iloc[0], 2.11)
        out = TrajectoryClassifier().classify_trajectories(feats, method='threshold')
        self.assertEqual(out['threshold_label'].iloc[0], 'immobile')
        # Same track with only Rg (no mean-step column): sRg must come from x / y, not Rg in pixels.
        only_rg = feats[['track_number', 'frame', 'x', 'y', 'radius_gyration']]
        out = TrajectoryClassifier().classify_trajectories(only_rg, method='threshold')
        self.assertEqual(out['threshold_label'].iloc[0], 'immobile')


if __name__ == '__main__':
    unittest.main()
