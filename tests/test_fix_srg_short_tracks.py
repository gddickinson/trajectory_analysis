"""Regression tests (fixed 2026-11): short tracks were classified by sRg (mostly 'immobile' when diffusing), and the
sRg definition / threshold / defaults differed between modules (2.11 vs 2.2224).

sRg of a freely diffusing track grows ~ sqrt(N / 3), so with the 2.11 threshold 94 % of random walks shorter than 15
points were called immobile. Tracks below `min_classify_length` (default 50, see docs/srg_calibration.md) are now
'too_short'. The options live in particle_tracker/analysis/mobility.py and are shared by FeatureCalculator,
AdvancedMetricsCalculator and TrajectoryClassifier; the 'legacy' preset reproduces the old FeatureCalculator results.
See SESSION_LOG.md, "Bugs fixed".
"""
import math
import unittest

import numpy as np
import pandas as pd

from particle_tracker.analysis import mobility
from particle_tracker.analysis.advanced_metrics import AdvancedMetricsCalculator
from particle_tracker.analysis.classification import TrajectoryClassifier
from particle_tracker.analysis.features import FeatureCalculator


def walks(n, n_tracks=6, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tracks):
        xy = np.cumsum(rng.normal(size=(n, 2)), axis=0) + 50
        rows += [{'track_number': t, 'frame': f, 'x': x, 'y': y} for f, (x, y) in enumerate(xy)]
    return pd.DataFrame(rows)


def per_track(df, col):
    return df.groupby('track_number')[col].first()


class TestShortTracks(unittest.TestCase):
    def test_short_walks_are_too_short_not_immobile(self):
        out = per_track(FeatureCalculator().calculate_features(walks(10)), 'mobility_classification')
        self.assertTrue((out == 'too_short').all(), out.tolist())

    def test_min_length_is_settable(self):
        out = per_track(FeatureCalculator({'min_classify_length': 5}).calculate_features(walks(10)),
                        'mobility_classification')
        self.assertTrue(out.isin(['mobile', 'immobile']).all())

    def test_legacy_preset_reproduces_old_classification(self):
        df = walks(10)
        legacy = FeatureCalculator({'srg_preset': 'legacy'}).calculate_features(df)
        srg = per_track(legacy, 'scaled_rg')
        expected = np.where(srg > 2.11, 'mobile', 'immobile')
        self.assertEqual(per_track(legacy, 'mobility_classification').tolist(), expected.tolist())

    def test_long_walks_still_classified(self):
        out = per_track(FeatureCalculator().calculate_features(walks(80)), 'mobility_classification')
        self.assertTrue((out == 'mobile').all(), out.tolist())


class TestDefinitions(unittest.TestCase):
    def test_rg_over_step_definition_and_threshold_conversion(self):
        df = walks(60)
        gs = FeatureCalculator().calculate_features(df)
        raw = FeatureCalculator({'srg_definition': 'rg_over_step'}).calculate_features(df)
        ratio = per_track(gs, 'scaled_rg') / per_track(raw, 'scaled_rg')
        np.testing.assert_allclose(ratio, math.sqrt(math.pi / 2), rtol=1e-12)
        # the default threshold is converted, so the classes agree
        self.assertEqual(per_track(gs, 'mobility_classification').tolist(),
                         per_track(raw, 'mobility_classification').tolist())

    def test_resolve(self):
        r = mobility.resolve({})
        self.assertEqual((r['srg_definition'], r['mobility_threshold'], r['min_classify_length']),
                         ('golan_sherman', 2.11, 50))
        r = mobility.resolve({'srg_preset': 'piezo1_tirf_10fps', 'min_classify_length': 30})
        self.assertAlmostEqual(r['mobility_threshold'], 2.22236433588659)
        self.assertEqual(r['min_classify_length'], 30)
        r = mobility.resolve({'srg_definition': 'rg_over_step'})
        self.assertAlmostEqual(r['mobility_threshold'], 2.11 / math.sqrt(math.pi / 2))
        r = mobility.resolve({'srg_definition': 'rg_over_step', 'mobility_threshold': 2.11})
        self.assertEqual(r['mobility_threshold'], 2.11)
        with self.assertRaises(ValueError):
            mobility.resolve({'srg_preset': 'nope'})
        with self.assertRaises(ValueError):
            mobility.resolve({'srg_definition': 'nope'})

    def test_immobile_noise_is_about_one(self):
        rng = np.random.default_rng(1)
        s = [mobility.track_srg(rng.normal(size=(300, 2)))[2] for _ in range(200)]
        self.assertAlmostEqual(float(np.mean(s)), 1.0, delta=0.01)

    def test_threshold_from_reference(self):
        self.assertAlmostEqual(mobility.threshold_from_reference(np.arange(101.0)), 95.0)


class TestModulesAgree(unittest.TestCase):
    def test_same_defaults_everywhere(self):
        df = walks(30, n_tracks=4).groupby('track_number').head(30)
        df = pd.concat([df, walks(60, n_tracks=4, seed=3).assign(track_number=lambda d: d.track_number + 10)])
        fc = per_track(FeatureCalculator().calculate_features(df), 'mobility_classification')
        am = per_track(AdvancedMetricsCalculator()._calculate_scaled_radius_of_gyration(df.copy()), 'mobility_sRg')
        tc = per_track(TrajectoryClassifier().classify_trajectories(df.copy(), method='threshold'), 'threshold_label')
        self.assertEqual(fc.tolist(), am.tolist())
        self.assertEqual(fc.tolist(), tc.tolist())
        self.assertTrue((fc.loc[:9] == 'too_short').all())

    def test_old_parameter_names_still_work(self):
        df = walks(60)
        am = AdvancedMetricsCalculator({'sRg_threshold': 1000.0})._calculate_scaled_radius_of_gyration(df.copy())
        self.assertTrue((am['mobility_sRg'] == 'immobile').all())
        tc = TrajectoryClassifier({'srg_mobility_threshold': 1000.0}).classify_trajectories(df.copy(),
                                                                                         method='threshold')
        self.assertTrue((tc['threshold_label'] == 'immobile').all())


if __name__ == '__main__':
    unittest.main()
