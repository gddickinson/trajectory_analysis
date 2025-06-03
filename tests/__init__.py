"""
Tests Package
============

Contains unit tests and integration tests for the particle tracking application.

Test structure:
- test_data_manager.py: Tests for data management functionality
- test_analysis_engine.py: Tests for analysis coordination
- test_detection.py: Tests for particle detection algorithms
- test_linking.py: Tests for trajectory linking methods
- test_features.py: Tests for feature calculation
- test_classification.py: Tests for trajectory classification
- test_gui.py: Tests for GUI components (requires pytest-qt)
- test_integration.py: Full workflow integration tests
"""

import sys
import os
from pathlib import Path

# Add the parent directory to Python path for testing
TEST_DIR = Path(__file__).parent
PROJECT_DIR = TEST_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

# Test configuration
TEST_DATA_DIR = TEST_DIR / "test_data"
TEST_OUTPUT_DIR = TEST_DIR / "test_output"

def setup_test_environment():
    """Set up the test environment."""
    # Create test directories
    TEST_DATA_DIR.mkdir(exist_ok=True)
    TEST_OUTPUT_DIR.mkdir(exist_ok=True)

    # Generate test data if needed
    generate_test_data()

def generate_test_data():
    """Generate test data for unit tests."""
    import numpy as np
    import pandas as pd

    # Generate synthetic TIRF data
    if not (TEST_DATA_DIR / "test_movie.npy").exists():
        # Create a small test movie
        test_movie = np.random.randint(50, 200, (10, 64, 64), dtype=np.uint16)

        # Add some synthetic particles
        for frame in range(10):
            for i in range(5):  # 5 particles per frame
                x = np.random.randint(10, 54)
                y = np.random.randint(10, 54)
                intensity = np.random.randint(500, 1000)

                # Add Gaussian spot
                sigma = 1.5
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        if 0 <= y+dy < 64 and 0 <= x+dx < 64:
                            dist_sq = dx*dx + dy*dy
                            value = intensity * np.exp(-dist_sq / (2 * sigma*sigma))
                            test_movie[frame, y+dy, x+dx] += int(value)

        np.save(TEST_DATA_DIR / "test_movie.npy", test_movie)

    # Generate synthetic trajectory data
    if not (TEST_DATA_DIR / "test_trajectories.csv").exists():
        trajectories = []

        for track_id in range(10):
            n_points = np.random.randint(5, 20)
            start_x = np.random.uniform(10, 54)
            start_y = np.random.uniform(10, 54)

            for i in range(n_points):
                trajectories.append({
                    'track_number': track_id,
                    'frame': i,
                    'x': start_x + np.random.normal(0, 0.5),
                    'y': start_y + np.random.normal(0, 0.5),
                    'intensity': np.random.uniform(500, 1000)
                })

        df = pd.DataFrame(trajectories)
        df.to_csv(TEST_DATA_DIR / "test_trajectories.csv", index=False)

def cleanup_test_environment():
    """Clean up test environment."""
    import shutil

    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)

# Test utilities
def get_test_data_path(filename: str) -> Path:
    """Get path to test data file."""
    return TEST_DATA_DIR / filename

def get_test_output_path(filename: str) -> Path:
    """Get path for test output file."""
    TEST_OUTPUT_DIR.mkdir(exist_ok=True)
    return TEST_OUTPUT_DIR / filename

# Test fixtures and helpers
class TestDataGenerator:
    """Helper class for generating test data."""

    @staticmethod
    def create_synthetic_movie(n_frames=10, height=64, width=64, n_particles=5):
        """Create a synthetic microscopy movie."""
        movie = np.random.randint(50, 150, (n_frames, height, width), dtype=np.uint16)

        # Add particles with realistic trajectories
        for particle_id in range(n_particles):
            start_x = np.random.uniform(10, width-10)
            start_y = np.random.uniform(10, height-10)

            # Random walk with drift
            positions = [(start_x, start_y)]

            for frame in range(1, n_frames):
                prev_x, prev_y = positions[-1]

                # Diffusion step
                dx = np.random.normal(0, 1.0)
                dy = np.random.normal(0, 1.0)

                new_x = np.clip(prev_x + dx, 5, width-5)
                new_y = np.clip(prev_y + dy, 5, height-5)

                positions.append((new_x, new_y))

            # Add particles to movie
            intensity = np.random.uniform(300, 800)
            sigma = 1.5

            for frame, (x, y) in enumerate(positions):
                # Skip some frames (blinking)
                if np.random.random() < 0.1:
                    continue

                # Add Gaussian spot
                x_int, y_int = int(round(x)), int(round(y))

                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        px = x_int + dx
                        py = y_int + dy

                        if 0 <= px < width and 0 <= py < height:
                            dist_sq = dx*dx + dy*dy
                            value = intensity * np.exp(-dist_sq / (2 * sigma*sigma))
                            movie[frame, py, px] += int(value)

        return movie

    @staticmethod
    def create_test_trajectories(n_tracks=10, min_length=5, max_length=20):
        """Create test trajectory data."""
        trajectories = []

        for track_id in range(n_tracks):
            n_points = np.random.randint(min_length, max_length+1)

            # Random starting position
            start_x = np.random.uniform(10, 90)
            start_y = np.random.uniform(10, 90)

            # Random trajectory parameters
            drift_x = np.random.normal(0, 0.1)
            drift_y = np.random.normal(0, 0.1)
            diffusion = np.random.uniform(0.5, 2.0)

            x, y = start_x, start_y

            for i in range(n_points):
                trajectories.append({
                    'track_number': track_id,
                    'frame': i,
                    'x': x,
                    'y': y,
                    'intensity': np.random.uniform(400, 900)
                })

                # Update position
                x += drift_x + np.random.normal(0, diffusion)
                y += drift_y + np.random.normal(0, diffusion)

        return pd.DataFrame(trajectories)

# Setup test environment on import
setup_test_environment()

__all__ = [
    "TEST_DIR",
    "PROJECT_DIR",
    "TEST_DATA_DIR",
    "TEST_OUTPUT_DIR",
    "setup_test_environment",
    "cleanup_test_environment",
    "generate_test_data",
    "get_test_data_path",
    "get_test_output_path",
    "TestDataGenerator",
]