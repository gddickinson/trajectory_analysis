"""Regression test (fixed 2026-11): MSD lags in AdvancedMetricsCalculator are frame differences, not row counts.

See SESSION_LOG.md, "Bugs fixed".
"""
import unittest

import numpy as np
import pandas as pd

from particle_tracker.analysis.advanced_metrics import AdvancedMetricsCalculator



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


class TestMsdFrameLags(unittest.TestCase):
    def test_gapped_straight_line(self):
        frames = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]
        df = pd.DataFrame({'track_number': 0, 'frame': frames, 'x': [0.5 * f for f in frames], 'y': 0.0})
        out = AdvancedMetricsCalculator()._calculate_advanced_diffusion_metrics(df)
        # MSD(lag) = (0.5 lag)^2 exactly: slope 2 on log-log, and D = MSD(1 frame) / 4
        self.assertAlmostEqual(out['alpha_exponent'].iloc[0], 2.0, places=12)
        self.assertAlmostEqual(out['diffusion_coefficient_advanced'].iloc[0], 0.25 / 4, places=12)

    def test_gap_free_unchanged(self):
        df = random_tracks(n_tracks=1, n_frames=30)
        out = AdvancedMetricsCalculator()._calculate_advanced_diffusion_metrics(df)
        p = df[['x', 'y']].values
        msd = [np.mean(np.sum((p[k:] - p[:-k]) ** 2, axis=1)) for k in range(1, 6)]
        slope = np.polyfit(np.log(np.arange(1, 6)), np.log(msd), 1)[0]
        self.assertAlmostEqual(out['alpha_exponent'].iloc[0], slope, places=10)


if __name__ == '__main__':
    unittest.main()
