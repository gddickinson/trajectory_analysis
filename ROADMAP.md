# Particle Tracking (Enhanced) -- Roadmap

## Current State
Large, well-organized PyQt6 application with deep module tree under `particle_tracker/`: analysis (detection, linking, classification, features, density, autocorrelation, background subtraction, localization precision, trajectory interpolation, advanced metrics), core (analysis engine, data manager, project manager, batch analysis), GUI (main window, parameter panels, visualization widget, analysis control, batch control, data browser, logging widget), and utilities (config manager, export manager, ROI manager, statistics generator, file utils, path utils, logging config). Has `setup.py`, tests (`test_installation.py`, debug scripts), and a stale `main_window_OLD.py`. README is polished but references features not yet verified (batch processor class, custom detection method registration).

## Short-term Improvements
- [ ] Delete `particle_tracker/gui/main_window_OLD.py` -- dead code
- [ ] Add real unit tests beyond `test_installation.py` -- test detection, linking, feature calculation with synthetic data
- [ ] Add `pyproject.toml` to replace or supplement `setup.py` (modern packaging)
- [ ] Validate that the `BatchProcessor` API described in README actually exists in `core/batch_analysis.py`
- [ ] Add input validation in `detection.py` and `linking.py` for edge cases (empty frames, single-pixel images)
- [ ] Fix `debug_imports.py` and `debug_test.py` -- either promote to proper tests or remove

## Feature Enhancements
- [ ] Implement the custom detection method registration pattern described in README (`detector.methods['my_method']`)
- [ ] Add progress reporting from `analysis_engine.py` back to GUI during long batch runs
- [ ] Add trajectory animation/playback in `visualization_widget.py`
- [ ] Implement MSD ensemble averaging and anomalous diffusion fitting in `advanced_metrics.py`
- [ ] Add support for multi-channel TIFF stacks (co-localization analysis)
- [ ] Add histogram and scatter plot views for feature distributions in the GUI

## Long-term Vision
- [ ] GPU-accelerated detection using cupy or torch for large datasets
- [ ] Plugin system for custom analysis modules (similar to unified_spt architecture)
- [ ] Web-based results viewer for sharing analysis with collaborators
- [ ] Integration with napari for 3D trajectory visualization
- [ ] Publish to PyPI with proper entry points (`particle-tracker` CLI command)

## Technical Debt
- [ ] `package_structure.py` appears to be a planning artifact -- remove or convert to documentation
- [ ] `scripts/init_setup.py` and `scripts/usage_examples.py` may be out of date with current API
- [ ] Ensure all `__init__.py` files have proper `__all__` exports
- [ ] Standardize logging: some modules use `logging_config.py`, verify consistency
- [ ] Review `config_manager.py` vs `project_manager.py` for overlapping responsibilities
