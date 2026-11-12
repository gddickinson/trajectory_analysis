"""Regression test (fixed 2026-11): when a direction-autocorrelation curve never falls to 1/e of its lag-1 value, the
persistence length is reported as censored at the last lag ("> last lag"), not as an invented 2 x last lag."""
import unittest

import numpy as np
import pandas as pd

from particle_tracker.analysis.autocorrelation_analysis import DirectionAutocorrelationAnalyzer


class TestPersistenceCensoring(unittest.TestCase):
    def test_straight_line_is_censored_at_last_lag(self):
        df = pd.DataFrame({'track_number': 0, 'frame': range(12), 'x': np.arange(12.0), 'y': 0.0})
        r = DirectionAutocorrelationAnalyzer().calculate_track_autocorrelation(df)
        self.assertTrue(r['persistence_censored'])
        self.assertEqual(r['persistence_length'], r['lags'][-1])   # lower bound: "> 10 frames"

    def test_crossing_is_not_censored(self):
        df = pd.DataFrame({'track_number': 0, 'frame': range(5), 'x': [0, 1, 2, 2, 1.0], 'y': [0, 0, 0, 1, 1.0]})
        r = DirectionAutocorrelationAnalyzer().calculate_track_autocorrelation(df)
        self.assertFalse(r['persistence_censored'])

    def test_censored_tracks_left_out_of_mean_persistence(self):
        a = DirectionAutocorrelationAnalyzer()
        stats = a._calculate_ensemble_statistics([
            {'persistence_length': 3.0, 'persistence_censored': True, 'directional_bias': 1.0, 'decay_constant': np.nan},
            {'persistence_length': 1.5, 'persistence_censored': False, 'directional_bias': 0.0, 'decay_constant': np.nan}])
        self.assertEqual(stats['mean_persistence_length'], 1.5)
        self.assertEqual(stats['n_persistence_censored'], 1)


if __name__ == '__main__':
    unittest.main()
