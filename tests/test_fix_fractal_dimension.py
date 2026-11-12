"""Regression test (fixed 2026-11): features.py fractal dimension uses the largest distance between points, as the original scripts did.

See SESSION_LOG.md, "Bugs fixed".
"""
import unittest
from pathlib import Path

import pandas as pd

from particle_tracker.analysis.features import AdvancedShapeAnalyzer

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / 'particle_tracker' / 'resources' / 'example_data' / 'Endothelial_NonBapta_bin10_crop_RESULTS-FILE.csv'



class TestFractalDimensionMatchesOriginalScripts(unittest.TestCase):
    def test_example_data_fracdimension(self):
        """The example results file was written by the original scripts; tracks whose positions were not
        interpolated afterwards (their radius of gyration still matches) must reproduce its fracDimension."""
        d = pd.read_csv(EXAMPLE)
        shape = AdvancedShapeAnalyzer()
        checked = 0
        for _, g in d.groupby('track_number'):
            g = g.sort_values('frame')
            m = shape.calculate_gyration_tensor_metrics(g[['x', 'y']].values)
            if abs(m['radius_gyration'] - g['radius_gyration'].iloc[0]) > 6e-4:
                continue  # trapped track interpolated after the features were written
            checked += 1
            self.assertAlmostEqual(m['fracDimension'], g['fracDimension'].iloc[0], delta=6e-4)
        self.assertGreater(checked, 100)


if __name__ == '__main__':
    unittest.main()
