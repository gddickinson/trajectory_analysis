# INTERFACE.md: navigation map

Particle Tracking Application (repo `trajectory_analysis`): PyQt6 desktop app for detecting, linking and analysing
particle trajectories, plus a FastAPI web interface / static GitHub Pages site for analysing finished track tables.
Many pre-existing modules are over the 500-line budget (see "Files over budget"); new code is kept under it.

## Entry points
| File | Purpose |
|---|---|
| `main.py` | Starts the desktop app (`particle_tracker.app.main`). |
| `run_web.py` | Starts the web app on http://127.0.0.1:8212 (uvicorn, `particle_tracker.web.app:app`). |
| `scripts/build_static.py` | Exports the static site (front end + every GET in `STATIC_ENDPOINTS` as JSON) to `site/`. |
| `scripts/screenshots.py` | Playwright smoke test of routes (`--static` / `--base URL`), phone-width check, README shots (`--shots`). |
| `scripts/make_engine_fixture.py` | Writes `tests/fixtures/engine_fixture.json` (Python results the JS engine must reproduce). |
| `scripts/init_setup.py`, `scripts/usage_examples.py` | Older setup / example scripts (may be out of date, see ROADMAP). |

## Package `particle_tracker/`
- `__init__.py`: version; public classes loaded lazily (PEP 562 `__getattr__`) so analysis modules import without PyQt6.
- `app.py`: `EnhancedParticleTrackingApp` / `ParticleTrackingApp` (Qt application, main window).

### `analysis/` (pure numpy/pandas/scipy/sklearn; no Qt)
| Module | Key contents |
|---|---|
| `detection.py` | `ParticleDetector` (+ threshold, LoG, trackpy `DetectionMethod`s; `detector.methods` dict). |
| `linking.py` | `ParticleLinker`, nearest-neighbour and trackpy linking. |
| `features.py` | `FeatureCalculator.calculate_features` (main feature step): `DensityAnalyzer` (neighbour counts 3–30 px), nearest-neighbour distance, `AdvancedShapeAnalyzer` (gyration tensor: Rg, asymmetry, skewness, kurtosis, eigenvalue ratio, step alignment, directionality, fractal dimension, straightness, `classify_linearity`), `ScaledRadiusCalculator` (sRg, mobile/immobile), step sizes / velocities, `TrajectoryInterpolator`. |
| `track_geometry.py` | `fractal_dimension` (original scripts' formula), `frame_lag_msd` (MSD from pairs exactly k frames apart). Shared by features / advanced_metrics. |
| `advanced_metrics.py` | `AdvancedMetricsCalculator` (sRg, shape, density, localisation precision, MSD α/D via `frame_lag_msd`, quality metrics, row-based direction autocorrelation for scripts). Not used by the GUI pipeline; the web app uses its MSD step. |
| `autocorrelation_analysis.py` | `DirectionAutocorrelationAnalyzer`: per-track curve (lags = step start-frame differences), persistence (1/e), directional bias, exponential fit, ensemble, plots, export. |
| `classification.py` | `TrajectoryClassifier`: SVM (training data in `resources/training_data/`), threshold (sRg), linearity, multi-round; `column_mappings` (accepted column names). |
| `density_analysis.py`, `localization_precision.py`, `trajectory_interpolation.py`, `background_subtraction.py` | Further analyses used by the GUI. |

### `core/` (Qt workers and managers)
`analysis_engine.py` (`AnalysisParameters`, `AnalysisStep`, `AnalysisWorker`, `AnalysisEngine`), `data_manager.py`,
`project_manager.py`, `batch_analysis.py` (`BatchAnalysisManager`).

### `gui/`
`main_window.py` (3233 lines), `visualization_widget.py`, `parameter_panels.py`, `analysis_control.py`, `batch_control.py`,
`data_browser.py`, `logging_widget.py`.

### `utils/`
`config_manager.py`, `export_manager.py`, `file_utils.py`, `path_utils.py`, `logging_config.py`, `roi_manager.py`, `statistics_generator.py`.

### `resources/`
`example_data/` (real PIEZO1 example: `Endothelial_NonBapta_bin10_crop_RESULTS-FILE.csv` + README), `training_data/` (SVM features).

### `web/` (web interface; see its `__init__.py`)
| File | Purpose |
|---|---|
| `app.py` | FastAPI routes: `/api/info`, `/api/demo/{id}/result`, `/api/demo/{id}/tracks`, `POST /api/analyze`; `STATIC_ENDPOINTS`. |
| `analysis.py` | `analyse(df, params, name)`: runs FeatureCalculator, the MSD step of AdvancedMetricsCalculator and DirectionAutocorrelationAnalyzer; JSON in pixels/frames. |
| `tracks_io.py` | Column names, `DEFAULTS` (units, thresholds), `parse_params`, `standardise`, `read_tracks_csv`, `to_pixels`, `coords`. |
| `demos.py` | Demo datasets: `endothelial` (real example file, SVM label) and `synthetic` (repo test generators, seed 2026). |
| `static/index.html`, `style.css` (kit theme), `style_app.css`, `core.js` (kit router/api), `main.js`, `icon.svg` | Shell. |
| `static/engine.js` | Browser port of `analysis.analyse` (same JSON); re-exports `engine_io.js`. |
| `static/engine_io.js` | CSV parsing, column detection, standardise, params, coords (mirrors `tracks_io.py`). |
| `static/engine_stats.js` | `eigSym2` (LAPACK dlanv2, numpy's eigenvector signs), skew/kurtosis, linregress, per-frame neighbours. |
| `static/engine_track.js` | Per-track ports: shape metrics, fractal dimension, sRg, linearity, velocity, MSD fit, autocorrelation, persistence, ensemble. |
| `static/engine.test.mjs` | Node test: JS vs Python on `tests/fixtures/engine_fixture.json` (rtol 1e-9). |
| `static/static_api.js` | Static-site `api()`: exported JSON + in-browser analysis for uploads/re-runs. |
| `static/state.js`, `fmt.js`, `charts.js` | Current result, unit conversion/formatting/CSV export, SVG/canvas charts. |
| `static/home.js`, `analyze.js`, `tracks.js`, `features.js`, `motion.js`, `density.js`, `lengthcheck.js`, `about.js` | Pages (about.js has Method and About). |

## Tests (`tests/`)
`test_detection.py`, `test_linking.py`, `test_installation.py`; regression tests `test_fix_*.py` (feature row order,
fractal dimension, MSD frame lags, autocorrelation frame lags, TestDataGenerator); `test_web.py` (API, static build,
fixture current, node engine test). `tests/__init__.py` holds `TestDataGenerator` and generates `tests/test_data/`.

## Files over budget (pre-existing, not split)
Most of `analysis/`, `core/`, `gui/`, `utils/` (500–3233 lines), `scripts/init_setup.py`.
