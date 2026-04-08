#!/usr/bin/env python3
"""
Unit tests for the enhanced particle detection module.

Tests detection methods with synthetic particle data.
"""

import unittest
import numpy as np
import pandas as pd


class TestSyntheticData(unittest.TestCase):
    """Shared synthetic data generation utilities."""

    @staticmethod
    def make_gaussian_spot(size=64, center=(32, 32), sigma=2.0, intensity=1000):
        """Create a 2D Gaussian spot."""
        y, x = np.ogrid[:size, :size]
        cy, cx = center
        img = intensity * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
        return img.astype(np.float64)

    @staticmethod
    def make_multi_particle_image(size=128, n_particles=5, sigma=2.0,
                                   intensity=1000, seed=42):
        """Create an image with multiple Gaussian particles."""
        rng = np.random.RandomState(seed)
        img = rng.normal(100, 10, (size, size)).astype(np.float64)
        positions = []
        margin = 10
        for _ in range(n_particles):
            cx = rng.randint(margin, size - margin)
            cy = rng.randint(margin, size - margin)
            y, x = np.ogrid[:size, :size]
            spot = intensity * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
            img += spot
            positions.append((cy, cx))
        return img, positions


class TestThresholdDetection(unittest.TestCase):
    """Tests for threshold-based particle detection."""

    def setUp(self):
        from particle_tracker.analysis.detection import EnhancedThresholdDetection
        self.detector = EnhancedThresholdDetection()

    def test_single_bright_particle(self):
        """A single bright Gaussian should be detected."""
        img = TestSyntheticData.make_gaussian_spot(
            size=64, center=(32, 32), sigma=2.0, intensity=5000
        )
        img += np.random.normal(100, 10, img.shape)
        result = self.detector.detect(img, threshold=3.0, sigma=1.6,
                                       min_intensity=0, max_intensity=100000)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

    def test_empty_image(self):
        """A uniform image should yield zero or very few detections."""
        img = np.ones((64, 64), dtype=np.float64) * 100
        result = self.detector.detect(img, threshold=3.0, sigma=1.6,
                                       min_intensity=0, max_intensity=100000)
        self.assertIsInstance(result, pd.DataFrame)

    def test_3d_stack(self):
        """Detection should work on 3D stacks."""
        frames = []
        for _ in range(3):
            img, _ = TestSyntheticData.make_multi_particle_image(size=64, n_particles=3)
            frames.append(img)
        stack = np.array(frames)
        result = self.detector.detect(stack, threshold=2.0, sigma=1.6,
                                       min_intensity=0, max_intensity=100000)
        self.assertIsInstance(result, pd.DataFrame)


class TestParticleDetector(unittest.TestCase):
    """Tests for the main ParticleDetector class."""

    def test_instantiation(self):
        from particle_tracker.analysis.detection import ParticleDetector
        detector = ParticleDetector()
        self.assertIn('threshold', detector.methods)

    def test_invalid_method_raises(self):
        from particle_tracker.analysis.detection import ParticleDetector
        detector = ParticleDetector()
        img = np.random.rand(64, 64) * 100
        with self.assertRaises(ValueError):
            detector.detect_particles(img, method='nonexistent')

    def test_detect_returns_dataframe(self):
        from particle_tracker.analysis.detection import ParticleDetector
        detector = ParticleDetector()
        img, _ = TestSyntheticData.make_multi_particle_image(size=64, n_particles=3)
        result = detector.detect_particles(img, method='threshold',
                                            min_intensity=0, max_intensity=100000)
        self.assertIsInstance(result, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
