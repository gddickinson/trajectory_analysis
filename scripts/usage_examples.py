# ============================================================================
# EXAMPLE USAGE SCRIPT
# ============================================================================

#!/usr/bin/env python3
"""
Example Usage of Particle Tracking Application
==============================================

This script demonstrates various ways to use the particle tracking application.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Import the particle tracking modules
from particle_tracker import (
    ParticleTrackingApp, DataManager, AnalysisEngine,
    AnalysisParameters, ParticleDetector, ParticleLinker
)
from particle_tracker.analysis.detection import ThresholdDetection
from particle_tracker.analysis.linking import NearestNeighborLinking


def example_gui_application():
    """Example 1: Launch the GUI application."""
    import sys

    print("Example 1: Launching GUI application...")

    # Create and run the application
    app = ParticleTrackingApp(sys.argv)

    # The application will start with the main window
    # Users can load data, set parameters, and run analysis interactively

    return app.exec()


def example_programmatic_analysis():
    """Example 2: Programmatic analysis without GUI."""

    print("Example 2: Programmatic analysis...")

    # Generate synthetic data for demonstration
    image_data = generate_synthetic_tirf_data()

    # Create analysis components
    data_manager = DataManager()
    analysis_engine = AnalysisEngine()

    # Load synthetic data
    data_manager._data['synthetic'] = image_data

    # Set up analysis parameters
    params = AnalysisParameters(
        detection_method="threshold",
        detection_sigma=1.6,
        detection_threshold=3.0,
        linking_method="nearest_neighbor",
        max_distance=5.0,
        max_gap_frames=2,
        min_track_length=5,
        pixel_size=108.0,  # nm per pixel
        frame_rate=10.0    # Hz
    )

    # Run detection
    print("Running particle detection...")
    detector = ParticleDetector()
    detections = detector.detect_particles(
        image_data,
        method=params.detection_method,
        sigma=params.detection_sigma,
        threshold=params.detection_threshold
    )
    print(f"Detected {len(detections)} particles")

    # Run linking
    print("Running trajectory linking...")
    linker = ParticleLinker()
    trajectories = linker.link_particles(
        detections,
        method=params.linking_method,
        max_distance=params.max_distance,
        max_gap_frames=params.max_gap_frames,
        min_track_length=params.min_track_length
    )

    n_tracks = trajectories['track_number'].nunique()
    print(f"Created {n_tracks} trajectories")

    # Calculate features
    print("Calculating features...")
    from particle_tracker.analysis.features import FeatureCalculator

    calculator = FeatureCalculator({'pixel_size': params.pixel_size})
    featured_trajectories = calculator.calculate_features(trajectories)

    print("Feature calculation complete!")
    print(f"Available features: {list(featured_trajectories.columns)}")

    # Save results
    output_path = "example_results.csv"
    featured_trajectories.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    return featured_trajectories


def example_custom_detection():
    """Example 3: Using custom detection method."""

    print("Example 3: Custom detection method...")

    # Create custom detection method
    class MyCustomDetection:
        def detect(self, image, **kwargs):
            """Simple peak finding detection."""
            from scipy import ndimage
            from skimage.feature import peak_local_maxima

            detections = []

            if len(image.shape) == 3:
                # Time series
                for frame_idx in range(image.shape[0]):
                    frame = image[frame_idx]

                    # Find local maxima
                    peaks = peak_local_maxima(
                        frame,
                        min_distance=3,
                        threshold_abs=kwargs.get('threshold', 100)
                    )

                    for y, x in zip(peaks[0], peaks[1]):
                        detections.append({
                            'frame': frame_idx,
                            'x': float(x),
                            'y': float(y),
                            'intensity': float(frame[y, x])
                        })

            return pd.DataFrame(detections)

    # Use custom detection
    image_data = generate_synthetic_tirf_data()

    custom_detector = MyCustomDetection()
    detections = custom_detector.detect(image_data, threshold=50)

    print(f"Custom detection found {len(detections)} particles")
    return detections


def example_batch_processing():
    """Example 4: Batch processing multiple files."""

    print("Example 4: Batch processing...")

    # This would process multiple files in a directory
    data_directory = Path("example_data")
    output_directory = Path("batch_results")

    # Create directories for demo
    data_directory.mkdir(exist_ok=True)
    output_directory.mkdir(exist_ok=True)

    # Generate some example files
    for i in range(3):
        synthetic_data = generate_synthetic_tirf_data(n_frames=50)
        np.save(data_directory / f"movie_{i}.npy", synthetic_data)

    # Set up analysis parameters
    params = AnalysisParameters(
        detection_method="threshold",
        detection_threshold=2.5,
        max_distance=4.0,
        min_track_length=10
    )

    # Process each file
    results_summary = []

    for data_file in data_directory.glob("*.npy"):
        print(f"Processing {data_file.name}...")

        # Load data
        image_data = np.load(data_file)

        # Run analysis
        detector = ParticleDetector()
        linker = ParticleLinker()

        detections = detector.detect_particles(image_data, **params.__dict__)
        trajectories = linker.link_particles(detections, **params.__dict__)

        # Save results
        output_file = output_directory / f"{data_file.stem}_trajectories.csv"
        trajectories.to_csv(output_file, index=False)

        # Collect summary statistics
        n_particles = len(detections)
        n_tracks = trajectories['track_number'].nunique()
        mean_track_length = trajectories.groupby('track_number').size().mean()

        results_summary.append({
            'file': data_file.name,
            'n_particles': n_particles,
            'n_tracks': n_tracks,
            'mean_track_length': mean_track_length
        })

    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(output_directory / "batch_summary.csv", index=False)

    print("Batch processing complete!")
    print(summary_df)

    return summary_df


def generate_synthetic_tirf_data(n_frames=100, image_size=(256, 256), n_particles=50):
    """Generate synthetic TIRF microscopy data for testing."""

    print(f"Generating synthetic data: {n_frames} frames, {n_particles} particles...")

    # Create empty image stack
    images = np.zeros((n_frames,) + image_size, dtype=np.uint16)

    # Add background noise
    background_level = 100
    noise_level = 20

    for t in range(n_frames):
        images[t] = np.random.normal(background_level, noise_level, image_size).astype(np.uint16)

    # Generate particle trajectories
    np.random.seed(42)  # For reproducible results

    for particle_id in range(n_particles):
        # Random starting position
        start_x = np.random.uniform(20, image_size[1] - 20)
        start_y = np.random.uniform(20, image_size[0] - 20)

        # Random trajectory parameters
        diffusion_coeff = np.random.uniform(0.1, 2.0)  # pixels^2 per frame
        directed_velocity = np.random.uniform(-0.5, 0.5)  # pixels per frame

        # Generate trajectory
        positions = [(start_x, start_y)]

        for t in range(1, n_frames):
            # Previous position
            prev_x, prev_y = positions[-1]

            # Random diffusion step
            dx = np.random.normal(0, np.sqrt(2 * diffusion_coeff))
            dy = np.random.normal(0, np.sqrt(2 * diffusion_coeff))

            # Directed motion
            dx += directed_velocity * np.random.normal(0, 0.1)
            dy += directed_velocity * np.random.normal(0, 0.1)

            # New position
            new_x = prev_x + dx
            new_y = prev_y + dy

            # Keep in bounds
            new_x = np.clip(new_x, 5, image_size[1] - 5)
            new_y = np.clip(new_y, 5, image_size[0] - 5)

            positions.append((new_x, new_y))

        # Add particle intensity to images
        intensity = np.random.uniform(200, 800)
        sigma = np.random.uniform(1.0, 2.0)  # PSF width

        for t, (x, y) in enumerate(positions):
            # Skip some frames randomly (blinking)
            if np.random.random() < 0.1:  # 10% blinking probability
                continue

            # Add Gaussian spot
            y_int, x_int = int(round(y)), int(round(x))

            # Create small Gaussian kernel
            kernel_size = int(4 * sigma)
            y_kernel, x_kernel = np.ogrid[-kernel_size:kernel_size+1, -kernel_size:kernel_size+1]

            gaussian = intensity * np.exp(-(x_kernel**2 + y_kernel**2) / (2 * sigma**2))

            # Add to image with bounds checking
            y_min = max(0, y_int - kernel_size)
            y_max = min(image_size[0], y_int + kernel_size + 1)
            x_min = max(0, x_int - kernel_size)
            x_max = min(image_size[1], x_int + kernel_size + 1)

            ky_min = max(0, kernel_size - y_int)
            ky_max = kernel_size + 1 + min(0, image_size[0] - y_int - kernel_size - 1)
            kx_min = max(0, kernel_size - x_int)
            kx_max = kernel_size + 1 + min(0, image_size[1] - x_int - kernel_size - 1)

            images[t, y_min:y_max, x_min:x_max] += gaussian[ky_min:ky_max, kx_min:kx_max].astype(np.uint16)

    print("Synthetic data generation complete!")
    return images


def main():
    """Run all examples."""

    print("Particle Tracking Application Examples")
    print("=" * 50)

    # Run examples (comment out GUI example for automated testing)
    # example_gui_application()  # Uncomment to run GUI

    trajectories = example_programmatic_analysis()
    detections = example_custom_detection()
    summary = example_batch_processing()

    print("\nAll examples completed successfully!")
    print(f"Final trajectory dataset shape: {trajectories.shape}")

    return trajectories, detections, summary


if __name__ == "__main__":
    main()

