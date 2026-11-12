"""Regression test (fixed 2026-11): FeatureCalculator nearest-neighbour distances and per-track shape metrics depend on frame order, not row order.

See SESSION_LOG.md, "Bugs fixed".
"""
import unittest

import numpy as np
import pandas as pd

from particle_tracker.analysis.features import FeatureCalculator



def random_tracks(seed=3, n_tracks=8, n_frames=15):
    rng = np.random.RandomState(seed)
    rows = []
    for t in range(n_tracks):
        x, y = rng.uniform(0, 40, 2)
        for f in range(n_frames):
            rows.append({'track_number': t, 'frame': f, 'x': x, 'y': y})
            x += rng.normal(0, 1.0)
            y += rng.normal(0, 1.0)
    return pd.DataFrame(rows)


class TestNearestNeighbourRowOrder(unittest.TestCase):
    def test_nn_distance_belongs_to_its_row(self):
        df = random_tracks()  # sorted by track, then frame
        out = FeatureCalculator()._calculate_basic_features(df)
        for _, r in out.iterrows():
            same = out[(out['frame'] == r['frame'])]
            d = np.hypot(same['x'] - r['x'], same['y'] - r['y'])
            expected = np.sort(d.values)[1]
            self.assertAlmostEqual(r['nn_distance'], expected, places=12)


class TestShapeMetricsRowOrder(unittest.TestCase):
    def test_shuffled_rows_give_same_track_features(self):
        df = random_tracks()
        calc = FeatureCalculator({'calculate_density': False})
        a = calc.calculate_features(df).groupby('track_number')[['Straight', 'skewness', 'netDispl']].first()
        shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
        b = calc.calculate_features(shuffled).groupby('track_number')[['Straight', 'skewness', 'netDispl']].first()
        pd.testing.assert_frame_equal(a, b, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
