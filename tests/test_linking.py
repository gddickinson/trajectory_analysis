#!/usr/bin/env python3
"""
Unit tests for the enhanced particle linking module.

Tests linking methods with synthetic trajectory data.
"""

import unittest
import numpy as np
import pandas as pd


def make_synthetic_localizations(n_tracks=3, track_length=10, noise=0.5, seed=42):
    """Create synthetic particle localizations for known trajectories."""
    rng = np.random.RandomState(seed)
    rows = []
    for track_id in range(n_tracks):
        x0 = rng.uniform(10, 90)
        y0 = rng.uniform(10, 90)
        vx = rng.uniform(-0.5, 0.5)
        vy = rng.uniform(-0.5, 0.5)
        for frame in range(track_length):
            x = x0 + vx * frame + rng.normal(0, noise)
            y = y0 + vy * frame + rng.normal(0, noise)
            rows.append({
                'frame': frame,
                'x': x,
                'y': y,
                'intensity': 1000.0,
                'true_track': track_id,
            })
    return pd.DataFrame(rows)


class TestNearestNeighborLinking(unittest.TestCase):
    """Tests for enhanced nearest neighbor linking."""

    def setUp(self):
        from particle_tracker.analysis.linking import NearestNeighborLinking
        self.linker = NearestNeighborLinking()

    def test_basic_linking(self):
        """Well-separated tracks should be linked correctly."""
        df = make_synthetic_localizations(n_tracks=3, track_length=10, noise=0.1)
        result, metrics = self.linker.link(
            df[['frame', 'x', 'y', 'intensity']].copy(),
            max_distance=5.0, max_gap_frames=2, min_track_length=3
        )
        self.assertIn('track_number', result.columns)
        self.assertGreater(result['track_number'].nunique(), 0)
        self.assertGreater(metrics.linking_efficiency, 0)

    def test_single_particle(self):
        """A single particle should form one trajectory."""
        df = pd.DataFrame({
            'frame': list(range(5)),
            'x': [10.0 + i * 0.1 for i in range(5)],
            'y': [20.0 + i * 0.1 for i in range(5)],
            'intensity': [1000.0] * 5,
        })
        result, metrics = self.linker.link(df, max_distance=5.0,
                                            max_gap_frames=2, min_track_length=3)
        self.assertEqual(result['track_number'].nunique(), 1)


class TestLinkingQualityMetrics(unittest.TestCase):
    """Tests for LinkingQualityMetrics dataclass."""

    def test_default_values(self):
        from particle_tracker.analysis.linking import LinkingQualityMetrics
        m = LinkingQualityMetrics()
        self.assertEqual(m.total_detections, 0)
        self.assertEqual(m.linking_efficiency, 0.0)
        self.assertIsInstance(m.track_length_distribution, dict)


class TestLinkingParameters(unittest.TestCase):
    """Tests for LinkingParameters dataclass."""

    def test_defaults(self):
        from particle_tracker.analysis.linking import LinkingParameters, LinkingStrategy
        p = LinkingParameters()
        self.assertEqual(p.strategy, LinkingStrategy.TRACKPY_STANDARD)
        self.assertEqual(p.max_distance, 2.0)


class TestParticleLinker(unittest.TestCase):
    """Tests for the enhanced ParticleLinker class."""

    def test_instantiation(self):
        from particle_tracker.analysis.linking import ParticleLinker
        linker = ParticleLinker()
        self.assertIsNotNone(linker.methods)

    def test_suggest_parameters(self):
        from particle_tracker.analysis.linking import ParticleLinker
        linker = ParticleLinker()
        df = make_synthetic_localizations(n_tracks=5, track_length=20)
        suggestions = linker.suggest_parameters(df)
        self.assertIsNotNone(suggestions)


if __name__ == '__main__':
    unittest.main()
