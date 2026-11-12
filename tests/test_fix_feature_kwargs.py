"""Regression test (fixed 2026-11): parameters passed to FeatureCalculator.calculate_features(**kwargs) were ignored
by the sRg / mobility and linearity steps, which read only the constructor's parameters.

See SESSION_LOG.md, "Bugs fixed".
"""
import unittest

import numpy as np
import pandas as pd

from particle_tracker.analysis.features import FeatureCalculator


def walks(n_tracks=4, n=60, seed=5):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tracks):
        xy = np.cumsum(rng.normal(size=(n, 2)), axis=0) + 50
        rows += [{'track_number': t, 'frame': f, 'x': x, 'y': y} for f, (x, y) in enumerate(xy)]
    return pd.DataFrame(rows)


class TestCalculateFeaturesKwargs(unittest.TestCase):
    def test_mobility_threshold_kwarg_is_used(self):
        df = walks()
        by_ctor = FeatureCalculator({'mobility_threshold': 1000.0}).calculate_features(df)
        by_kwarg = FeatureCalculator().calculate_features(df, mobility_threshold=1000.0)
        self.assertTrue((by_ctor['mobility_classification'] == 'immobile').all())
        self.assertTrue((by_kwarg['mobility_classification'] == 'immobile').all())

    def test_linear_threshold_kwargs_are_used(self):
        df = walks()
        out = FeatureCalculator().calculate_features(df, linear_eigenvalue_threshold=0.0,
                                                     linear_alignment_threshold=-1.0)
        self.assertFalse((out['linear_classification'] == 'non_linear').any())


if __name__ == '__main__':
    unittest.main()
