"""Regression test (fixed 2026-11): the ensemble direction autocorrelation used to stop at the lag range of the
shortest track, so one short track cut the curve to a few lags. It now covers every lag reached by at least
min_tracks tracks, pooling the step pairs of all tracks (a mean weighted by pair counts), and reports the number of
tracks and pairs at each lag."""
import unittest

import numpy as np
import pandas as pd

from particle_tracker.analysis.autocorrelation_analysis import DirectionAutocorrelationAnalyzer


def tracks():
    rng = np.random.RandomState(5)
    rows = []
    for t, n in enumerate([5, 30, 30, 30, 18, 25]):
        x = np.cumsum(rng.normal(0, 1, n))
        y = np.cumsum(rng.normal(0, 1, n))
        rows += [{'track_number': t, 'frame': f, 'x': x[f], 'y': y[f]} for f in range(n)]
    return pd.DataFrame(rows)


class TestEnsembleLags(unittest.TestCase):
    def test_short_track_does_not_truncate(self):
        a = DirectionAutocorrelationAnalyzer()
        e = a.calculate_ensemble_autocorrelation(tracks(), max_lag=10, min_tracks=3)
        self.assertEqual(list(e['lags']), list(range(1, 11)))   # shortest track (5 points) reaches lag 3 only
        self.assertEqual(e['n_tracks_per_lag'][0], 6)
        self.assertEqual(e['n_tracks_per_lag'][-1], 5)

    def test_pair_weighted_mean(self):
        df = tracks()
        a = DirectionAutocorrelationAnalyzer()
        e = a.calculate_ensemble_autocorrelation(df, max_lag=10, min_tracks=3)
        for k, lag in enumerate(e['lags']):
            num = den = 0.0
            for _, g in df.groupby('track_number'):
                r = a.calculate_track_autocorrelation(g, 10)
                if lag <= len(r['lags']) and r['pair_counts'][lag - 1]:
                    num += r['autocorr_values'][lag - 1] * r['pair_counts'][lag - 1]
                    den += r['pair_counts'][lag - 1]
            self.assertAlmostEqual(e['ensemble_autocorr'][k], num / den, places=12)
            self.assertEqual(e['n_pairs_per_lag'][k], den)

    def test_lags_with_too_few_tracks_dropped(self):
        a = DirectionAutocorrelationAnalyzer()
        e = a.calculate_ensemble_autocorrelation(tracks(), max_lag=20, min_tracks=4)
        self.assertTrue(all(n >= 4 for n in e['n_tracks_per_lag']))
        self.assertEqual(max(e['lags']), 20)   # four 25+ point tracks reach lag 20 (18-point track stops at 16)


if __name__ == '__main__':
    unittest.main()
