"""Regression test (fixed 2026-11): Direction autocorrelation lags are frame differences between step start frames.

See SESSION_LOG.md, "Bugs fixed".
"""
import unittest

import numpy as np
import pandas as pd

from particle_tracker.analysis.autocorrelation_analysis import DirectionAutocorrelationAnalyzer



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


class TestAutocorrelationFrameLags(unittest.TestCase):
    def test_zero_length_step_keeps_its_place(self):
        # steps: +x (f0-1), none (f1-2), +x (f2-3), +y (f3-4)
        df = pd.DataFrame({'track_number': 0, 'frame': [0, 1, 2, 3, 4],
                           'x': [0.0, 1.0, 1.0, 2.0, 2.0], 'y': [0.0, 0.0, 0.0, 0.0, 1.0]})
        r = DirectionAutocorrelationAnalyzer().calculate_track_autocorrelation(df)
        ac = dict(zip(r['lags'], r['autocorr_values']))
        self.assertAlmostEqual(ac[1], 0.0)  # only (f2-3, f3-4): +x . +y
        self.assertAlmostEqual(ac[2], 1.0)  # only (f0-1, f2-3): +x . +x

    def test_frame_gap(self):
        # frames 0,1,3,4: steps +x (start 0), +x (start 1, spans the gap), -x (start 3)
        df = pd.DataFrame({'track_number': 0, 'frame': [0, 1, 3, 4], 'x': [0.0, 1.0, 2.0, 1.0], 'y': 0.0})
        r = DirectionAutocorrelationAnalyzer().calculate_track_autocorrelation(df)
        ac = dict(zip(r['lags'], r['autocorr_values']))
        self.assertAlmostEqual(ac[1], 1.0)   # (start 0, start 1)
        # starts 1 and 3 are two frames apart: +x . -x = -1
        self.assertAlmostEqual(ac[2], -1.0)

    def test_gap_free_unchanged(self):
        df = random_tracks(n_tracks=1, n_frames=25)
        r = DirectionAutocorrelationAnalyzer().calculate_track_autocorrelation(df)
        p = df[['x', 'y']].values
        s = np.diff(p, axis=0)
        u = s / np.linalg.norm(s, axis=1)[:, None]
        for lag, v in zip(r['lags'], r['autocorr_values']):
            self.assertAlmostEqual(v, np.mean(np.sum(u[:-lag] * u[lag:], axis=1)), places=12)


if __name__ == '__main__':
    unittest.main()
