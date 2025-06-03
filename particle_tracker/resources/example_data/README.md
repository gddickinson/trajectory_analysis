# Minimal Test Model

A minimal example dataset and analysis workflow for validating the PIEZO1-HaloTag tracking and analysis pipeline. This dataset demonstrates the complete workflow from particle localization through analysis and visualization, using endothelial cell recordings that show high expression of PIEZO1.

## Dataset Contents

### Microscopy Data
- `Endothelial_NonBapta_bin10_crop.tif`
  - TIRF microscopy recording of PIEZO1-HaloTag labeled endothelial cells
  - 10ms per frame acquisition, binned by 10 frames
  - Labeled with JF646 HaloTag ligand
  - Shows punctate PIEZO1 signal as previously observed for endogenous tagged channels

### Analysis Files
- `Endothelial_NonBapta_bin10_crop_RESULTS-FILE.csv`
  - ThunderSTORM particle localization output
  - Contains frame-by-frame puncta locations
  - Precision ~20nm using multi-emitter fitting

- `Endothelial_NonBapta_bin10_crop_locs-protocol.txt`
  - ThunderSTORM parameters and settings
  - Documents localization protocol

- `ROI_Endothelial_NonBapta_bin10_crop.txt`
  - Region definitions for analysis
  - Includes background measurement regions
  - FLIKA-compatible format

- `thunderStorm_macro_auto.ijm`
  - ImageJ/FIJI macro for automated processing
  - Configured for batch analysis
  - Standard ThunderSTORM settings

## Analysis Pipeline

### 1. Particle Localization (Pre-processed)
- Uses ThunderSTORM for super-resolution localization
- Multi-emitter fitting enabled
- Results file provided with centroids

### 2. Trajectory Analysis
Run steps 2-10 sequentially:
```python
from flika_scripts.piezo1_analysis.2_analysis.Step_2_linkingAndClassification import linkFiles

# Example usage
linkFiles(tiffList='Endothelial_NonBapta_bin10_crop.tif')
```

Key analysis steps include:
- Particle linking across frames
- Diffusion coefficient calculation
- Nearest neighbor analysis 
- Velocity measurements
- Background subtraction
- Point interpolation

### 3. Visualization
Using the locsAndTracks FLIKA plugin:
```python
from flika_plugins.locsAndTracksPlotter import LocsAndTracksPlotter
plotter = LocsAndTracksPlotter()
plotter.loadData('*_RESULTS-FILE.csv')
```

## Expected Results

### Particle Detection
- Punctate PIEZO1 signals visible in TIRF recordings
- Two distinct mobility populations:
  - Immobile puncta (diffusion coefficient ~0.003 μm²/s)
  - Mobile puncta (diffusion coefficient ~0.029 μm²/s)
- Signal-to-background ratio ~5.6 for labeled puncta

### Track Analysis
The analysis should reveal:
- Track durations
- Movement patterns
- Diffusion characteristics
- Density distributions
- Velocity profiles

### Data Specifications

#### Raw Data
- Format: TIFF
- Dimensions: [x, y, t]
- Binning: 10 frames
- Cell type: Endothelial
- Label: JF646 HaloTag ligand

#### Analysis Results
Columns in output CSVs include:
- frame
- track_number
- x, y coordinates
- intensity
- diffusion metrics
- velocity measurements
- nearest neighbor counts
- background measurements

## Validation Steps

1. **Check Localization**
   - Verify puncta detection
   - Confirm precision (~20nm)
   - Check background levels

2. **Examine Tracks**
   - Validate linking parameters
   - Review track lengths
   - Assess mobility patterns

3. **Verify Measurements**
   - Check diffusion coefficients
   - Validate velocity calculations
   - Confirm neighbor counts

## Troubleshooting

Common issues and solutions:

1. **Detection Problems**
   - Verify ThunderSTORM settings
   - Check signal-to-noise ratio
   - Adjust detection threshold

2. **Linking Issues**
   - Review gap frame settings
   - Adjust linking distance
   - Check for track fragmentation

3. **Analysis Errors**
   - Verify input file formats
   - Check parameter settings
   - Validate ROI definitions

## References

This dataset and analysis pipeline are described in:
```
Bertaccini et al. (2023). PIEZO1-HaloTag hiPSCs: Bridging Molecular, 
Cellular and Tissue Imaging. bioRxiv 2023.12.22.573117
```

## Support

For assistance:
- Check main documentation
- Review plugin guides
- Contact repository maintainers