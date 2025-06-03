# Particle Tracking Application - Installation & Quick Start

## 🚀 Installation Guide

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for development installation)

### Option 1: Quick Installation (Recommended)

```bash
# Create a new conda environment (optional but recommended)
conda create -n particle_tracker python=3.10
conda activate particle_tracker

# Install the package
pip install -e .

# Or install with all optional dependencies
pip install -e ".[dev,docs,optional]"
```

### Option 2: Development Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/particle-tracker.git
cd particle-tracker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pre-commit install
```

### Option 3: From Requirements File

```bash
# Install dependencies
pip install -r requirements.txt

# Run directly from source
python main.py
```

## 🎯 Quick Start

### 1. Launch the Application

```bash
# Start the GUI application
particle-tracker

# Or run directly
python main.py

# With debug logging
python main.py --debug
```

### 2. Load Your Data

The application supports multiple data formats:

- **TIFF/TIF files**: Time-lapse microscopy movies
- **CSV files**: Pre-computed localizations or trajectories  
- **NPY files**: NumPy arrays with image data
- **JSON files**: Trackpy-format trajectory data

**From GUI:**
1. Use `File → Open Image...` for microscopy movies
2. Use `File → Open Data...` for localization/trajectory files
3. Drag and drop files onto the application window

**From Command Line:**
```bash
python main.py --data your_tirf_movie.tif
```

### 3. Configure Analysis Parameters

**Detection Parameters:**
- **Method**: Choose threshold, LoG, or trackpy
- **Sigma**: Expected particle size (1.0-3.0 pixels typical)
- **Threshold**: Detection sensitivity (2-5σ recommended)

**Linking Parameters:**
- **Max Distance**: Maximum movement between frames (3-10 pixels)
- **Max Gap**: Frames a particle can disappear (0-5 frames)
- **Min Track Length**: Filter short tracks (≥3 points recommended)

**Feature Parameters:**
- **Pixel Size**: nm per pixel (e.g., 108 nm for 60x objective)
- **Frame Rate**: Acquisition rate in Hz

### 4. Run Analysis

**Interactive Mode:**
1. Select analysis steps in the Analysis Control panel
2. Click "Run Analysis"
3. Monitor progress in the status bar
4. View results in the visualization panel

**Programmatic Mode:**
```python
from particle_tracker import AnalysisEngine, AnalysisParameters
import numpy as np

# Load your data
image_data = np.load("your_data.npy")

# Configure parameters
params = AnalysisParameters(
    detection_method="threshold",
    detection_threshold=3.0,
    max_distance=5.0,
    pixel_size=108.0
)

# Run analysis
engine = AnalysisEngine()
result = engine.run_analysis_pipeline(
    image_data, 
    params, 
    steps=['detection', 'linking', 'features']
)
```

### 5. Visualize and Export Results

**Visualization:**
- Use frame slider to navigate time-lapse data
- Toggle track/localization display
- Color-code by features or classification
- Zoom and pan with mouse controls

**Export Options:**
- **CSV**: Full trajectory data with features
- **Excel**: Multi-sheet workbooks with summaries
- **Reports**: Text-based analysis summaries
- **Images**: Export current view as PNG/PDF

## 📊 Example Analysis Workflow

### Single-Molecule TIRF Data

```python
import numpy as np
from particle_tracker import ParticleTrackingApp

# 1. Load TIRF movie (shape: frames × height × width)
tirf_movie = np.load("tirf_data.npy")

# 2. Set analysis parameters
params = {
    'detection_method': 'threshold',
    'detection_sigma': 1.6,        # PSF width
    'detection_threshold': 3.0,    # 3σ above background
    'max_distance': 5.0,           # pixels
    'max_gap_frames': 2,           # allow 2-frame gaps
    'min_track_length': 5,         # minimum 5 localizations
    'pixel_size': 108.0,           # nm per pixel
    'frame_rate': 10.0             # Hz
}

# 3. Run analysis (programmatically)
from particle_tracker import AnalysisEngine, AnalysisParameters

engine = AnalysisEngine()
parameters = AnalysisParameters(**params)

trajectories = engine.run_analysis_pipeline(
    tirf_movie, 
    parameters,
    steps=['detection', 'linking', 'features', 'classification']
)

# 4. Analyze results
print(f"Detected {len(trajectories)} localizations")
print(f"Created {trajectories['track_number'].nunique()} trajectories")

# Get mobile trajectories
mobile_tracks = trajectories[trajectories['mobility_classification'] == 'mobile']
print(f"Mobile fraction: {len(mobile_tracks) / len(trajectories):.2%}")
```

### Batch Processing Multiple Files

```python
from pathlib import Path
from particle_tracker.batch import BatchProcessor

# Process all TIFF files in a directory
input_dir = Path("raw_data/")
output_dir = Path("analysis_results/")

# Configure batch processor
processor = BatchProcessor()
results = processor.process_directory(
    input_dir=input_dir,
    output_dir=output_dir,
    file_pattern="*.tif",
    parameters=parameters
)

# Generate summary report
processor.generate_summary_report(output_dir / "batch_summary.csv")
```

## 🔧 Troubleshooting

### Common Issues

**"No module named 'particle_tracker'"**
- Make sure you've installed the package: `pip install -e .`
- Check your Python environment is activated

**"trackpy not found" warning**
- Install trackpy: `pip install trackpy`
- Or use built-in detection methods

**GUI doesn't start**
- Install PyQt6: `pip install PyQt6`
- On Linux, may need: `sudo apt-get install python3-pyqt6`

**Memory errors with large datasets**
- Reduce image size or number of frames
- Use batch processing for multiple files
- Increase virtual memory/swap space

**Poor detection results**
- Adjust sigma parameter to match particle size
- Try different detection methods
- Check image contrast and background subtraction

### Performance Tips

1. **Detection Optimization:**
   - Use appropriate sigma (1-2x particle radius)
   - Set reasonable intensity thresholds
   - Consider background subtraction

2. **Linking Optimization:**
   - Set max_distance based on particle speed
   - Use gap filling for blinking particles
   - Filter short trajectories post-linking

3. **Memory Management:**
   - Process large datasets in chunks
   - Use smaller regions of interest
   - Close unused data in Data Browser

4. **Visualization Performance:**
   - Limit track length display for dense data
   - Use feature-based coloring sparingly
   - Reduce point size for large datasets

## 📚 Next Steps

1. **Read the Documentation:**
   - User Guide: `docs/user_guide.md`
   - API Reference: `docs/api_reference.md`
   - Tutorials: `docs/tutorials/`

2. **Try the Examples:**
   - `examples/usage_examples.py`
   - `examples/synthetic_data.py`
   - `examples/batch_processing.py`

3. **Join the Community:**
   - GitHub Issues for bug reports
   - Discussions for questions
   - Contribute new features

4. **Extend the Application:**
   - Add custom detection methods
   - Implement new linking algorithms
   - Create specialized analysis workflows

## 🆘 Getting Help

- **Documentation**: Check the `docs/` directory
- **Examples**: See `examples/` for common use cases
- **Issues**: Report bugs on GitHub Issues
- **Support**: Email your.email@example.com
- **Community**: Join GitHub Discussions

Happy tracking! 🔬✨