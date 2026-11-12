"""Regression test (fixed 2026-11): TestDataGenerator in tests/__init__.py used numpy and pandas without importing
them at module level, so every call raised NameError."""
import unittest

from tests import TestDataGenerator


class TestDataGeneratorWorks(unittest.TestCase):
    def test_create_test_trajectories(self):
        df = TestDataGenerator.create_test_trajectories(n_tracks=3, min_length=5, max_length=6)
        self.assertEqual(df['track_number'].nunique(), 3)
        self.assertTrue(set(['track_number', 'frame', 'x', 'y']).issubset(df.columns))

    def test_create_synthetic_movie(self):
        movie = TestDataGenerator.create_synthetic_movie(n_frames=3, height=32, width=32, n_particles=2)
        self.assertEqual(movie.shape, (3, 32, 32))


if __name__ == '__main__':
    unittest.main()
