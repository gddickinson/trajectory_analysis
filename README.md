# ============================================================================
# README.MD
# ============================================================================

# Particle Tracking Application

A comprehensive, modular application for analyzing particle trajectories from TIRF microscopy recordings and other single-molecule imaging data.

## Features

### 🔬 **Multi-Method Particle Detection**
- Threshold-based detection
- Laplacian of Gaussian (LoG) blob detection
- Trackpy integration
- Background subtraction and filtering

### 🔗 **Advanced Trajectory Linking**
- Nearest neighbor linking
- Trackpy-based linking with gap filling
- Adaptive search parameters
- Minimum track length filtering

### 📊 **Comprehensive Feature Analysis**
- Radius of gyration (simple and tensor methods)
- Asymmetry, skewness, kurtosis
- Fractal dimension
- Mean squared displacement (MSD)
- Velocity and diffusion analysis
- Nearest neighbor distances

### 🤖 **Machine Learning Classification**
- SVM-based trajectory classification
- Threshold-based mobility classification
- Custom feature selection
- Training data import

### 📈 **Interactive Visualization**
- Real-time image display with overlays
- Track visualization with customizable colors
- Feature-based coloring schemes
- Playback controls for time series
- Export capabilities

### 💼 **Project Management**
- Save and load analysis projects
- Parameter persistence
- Data file management
- Analysis result tracking

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install from PyPI (when available)
```bash
pip install particle-tracker
```

### Install from Source
```bash
git clone https://github.com/yourusername/particle-tracker.git
cd particle-tracker
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/yourusername/particle-tracker.git
cd particle-tracker
pip install -e ".[dev,docs,optional]"
```

## Quick Start

### GUI Application
```bash
# Start the application
particle-tracker

# Or run directly
python main.py

# With debug logging
python main.py --debug

# Load a project on startup
python main.py --project my_analysis.ptproj

# Load data on startup
python main.py --data my_tirf_movie.tif
```

### Python API
```python
import numpy as np
from particle_tracker import ParticleTrackingApp, AnalysisEngine, AnalysisParameters

# Load your data
image_data = np.load("my_tirf_data.npy")

# Set up analysis parameters
params = AnalysisParameters(
    detection_method="threshold",
    detection_threshold=3.0,
    linking_method="nearest_neighbor",
    max_distance=5.0,
    pixel_size=108.0,  # nm per pixel
    frame_rate=10.0    # Hz
)

# Create analysis engine
engine = AnalysisEngine()

# Run full analysis pipeline
result = engine.run_analysis_pipeline(
    image_data,
    params,
    steps=['detection', 'linking', 'features', 'classification']
)
```

## Usage Guide

### 1. Loading Data
The application supports various data formats:
- **Images**: TIFF, PNG, JPEG (2D or 3D time series)
- **Localizations**: CSV files with x, y, frame columns
- **Trajectories**: CSV files with track_number column
- **Analysis Results**: Excel, JSON formats

### 2. Analysis Workflow

#### Step 1: Particle Detection
Configure detection parameters:
- **Method**: Choose from threshold, LoG, or trackpy
- **Sigma**: Expected particle size (1-3 pixels typical)
- **Threshold**: Detection sensitivity (2-5σ recommended)

#### Step 2: Trajectory Linking
Set linking parameters:
- **Max Distance**: Maximum particle movement between frames
- **Max Gap**: Number of frames a particle can disappear
- **Min Track Length**: Filter short, noisy trajectories

#### Step 3: Feature Calculation
Calculate trajectory features:
- **Radius of Gyration**: Spatial extent of trajectories
- **Asymmetry**: Shape anisotropy
- **Diffusion Coefficient**: From MSD analysis
- **Velocity**: Instantaneous and mean velocities

#### Step 4: Classification
Classify trajectory behavior:
- **SVM Method**: Train on labeled data
- **Threshold Method**: Simple mobility criteria

### 3. Visualization and Analysis
- Use the visualization panel to inspect results
- Color trajectories by features or classification
- Export results to CSV or generate reports

## File Formats

### Project Files (.ptproj)
JSON format containing:
- Analysis parameters
- Data file references
- Results metadata
- Project notes

### Data Export Formats
- **CSV**: Comma-separated values
- **Excel**: Multi-sheet workbooks
- **JSON**: Structured data format

## Advanced Usage

### Batch Processing
```python
from particle_tracker.batch import BatchProcessor

processor = BatchProcessor()
processor.process_directory(
    input_dir="data/",
    output_dir="results/",
    parameters=params
)
```

### Custom Analysis Methods
```python
from particle_tracker.analysis.detection import DetectionMethod

class MyCustomDetection(DetectionMethod):
    def detect(self, image, **kwargs):
        # Your custom detection algorithm
        return detections_dataframe

# Register your method
detector = ParticleDetector()
detector.methods['my_method'] = MyCustomDetection()
```

## API Reference

### Core Classes
- `ParticleTrackingApp`: Main application class
- `DataManager`: Handles data loading and management
- `AnalysisEngine`: Coordinates analysis workflows
- `ProjectManager`: Manages projects and settings

### Analysis Components
- `ParticleDetector`: Particle detection methods
- `ParticleLinker`: Trajectory linking algorithms
- `FeatureCalculator`: Trajectory feature computation
- `TrajectoryClassifier`: Classification methods

### GUI Components
- `MainWindow`: Main application interface
- `VisualizationWidget`: Image and trajectory display
- `ParameterPanels`: Analysis parameter controls

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
git clone https://github.com/yourusername/particle-tracker.git
cd particle-tracker
pip install -e ".[dev]"
pre-commit install
```

### Running Tests
```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{particle_tracker,
  title={Particle Tracking Application},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/particle-tracker}
}
```

## Acknowledgments

- Built on scientific Python ecosystem (NumPy, SciPy, scikit-learn)
- GUI powered by PyQt6 and pyqtgraph
- Inspired by particle tracking workflows in biophysics research
- Consolidates and extends functionality from multiple analysis scripts

## Support

- 📖 [Documentation](https://particle-tracker.readthedocs.io/)
- 🐛 [Issue Tracker](https://github.com/yourusername/particle-tracker/issues)
- 💬 [Discussions](https://github.com/yourusername/particle-tracker/discussions)
- 📧 [Email Support](mailto:your.email@example.com)

