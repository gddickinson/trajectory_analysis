# SESSION_LOG

## 2026-11-12: web interface, static site, bug fixes (branch `web-interface`)

### Added
- Web interface `particle_tracker/web/` (FastAPI, port 8212, `run_web.py`) and static GitHub Pages site
  (`scripts/build_static.py`, `.github/workflows/pages.yml`). Pages: Home, Analyze, Tracks, Features, Motion, Density,
  Length check, Method, About.
- Browser engine (`static/engine*.js`): line-by-line port of FeatureCalculator, the MSD step of AdvancedMetricsCalculator
  and DirectionAutocorrelationAnalyzer; `engine.test.mjs` checks it against Python on
  `tests/fixtures/engine_fixture.json` (31,536 checks, rtol 1e-9). Includes a port of LAPACK's 2x2 Schur step so the
  principal-axis sign (and so the skewness sign) matches numpy.
- Demo data: the real example results file (119 tracks) and tracks from `tests/__init__.py` generators (labelled simulated).
- `particle_tracker/__init__.py` imports its public classes lazily so the analysis modules (and the web app) import
  without PyQt6; `from particle_tracker import X` is unchanged and the desktop app still starts.
- README: web section; corrected claims about a `particle_tracker.batch.BatchProcessor` (does not exist; batch runs
  are in the GUI) and `run_analysis_pipeline` (asynchronous, takes `AnalysisStep` values); placeholder URLs replaced.
- INTERFACE.md, CLAUDE.md, this log.

### Bugs fixed (each with a regression test that failed before; separate commits)
1. `FeatureCalculator._calculate_nearest_neighbors` assigned distances in frame order to rows in table order: wrong
   rows whenever the table was not sorted by frame (e.g. sorted by track). `_calculate_advanced_shape_features` used
   rows in table order instead of frame order. (`tests/test_fix_feature_row_order.py`)
2. `features.py` fractal dimension used the end-to-end distance; the original scripts (and advanced_metrics) use the
   largest point-to-point distance. Now shared in `analysis/track_geometry.py`; reproduces the example file's
   fracDimension on 111/119 tracks (was 12/119; the other 8 are trapped tracks whose stored x/y no longer reproduce the
   file's own Rg). (`tests/test_fix_fractal_dimension.py`)
3. MSD lags in `AdvancedMetricsCalculator` counted rows: wrong lag times for tracks with missing frames. Now frame
   differences (`track_geometry.frame_lag_msd`). (`tests/test_fix_msd_frame_lags.py`)
4. Direction autocorrelation dropped zero-length steps and ignored gaps before pairing steps k apart, shifting lags.
   Steps now keep their start frame; lag k pairs steps starting k frames apart. (`tests/test_fix_autocorr_frame_lags.py`)
5. `TestDataGenerator` (tests/__init__.py) used np/pd without importing them (NameError). (`tests/test_fix_test_data_generator.py`)
Gap-free, frame-sorted data give the same results as before for 1, 3 and 4.

### Checked, not a bug of the kind found elsewhere
- No length-dependent "confinement ratio" in the live code. `AdvancedMetricsCalculator.confinement_strength` is
  1 − MSD(1)/fitted MSD(1) (a fit residual, not a confinement measure); not shown on the site.

### Open questions for George (not changed)
- **sRg mobility depends on track length.** sRg grows roughly as √n for diffusing tracks and stays ≈ 1 for stationary
  ones. On the synthetic demo, random walks shorter than ~15 points are mostly "immobile" at the 2.11 threshold. Shown
  on the Length check page.
- **Two sRg definitions / thresholds.** features.py: sRg = √(π/2)·Rg/mean step, mobile above 2.11. The example file's
  `radius_gyration_scaled` (original scripts) equals Rg/mean step exactly (no √(π/2)); classification.py uses 2.2224.
- **Ensemble autocorrelation is truncated to the shortest track's lag range** (one 5-point track limits it to 3 lags).
  Kept; the web app lets the user raise the minimum track length.
- **Persistence when the curve never falls to 1/e** is reported as 2 × last lag (an invented value). Kept in Python;
  the site shows "no 1/e drop" instead.
- The example file's `nnDist_inFrame` column disagrees with distances recomputed from its own x/y in 42 % of rows and
  contradicts its own neighbour counts in 207 rows; how it was computed is not in the repo.

### Tests (2026-11-12)
- `pytest -q tests`: all pass (see report); `pytest -q tests/test_web.py`: all pass; node engine test ok;
  `screenshots.py --static` and `--base http://127.0.0.1:8212` all routes ok, phone overflow 0.

### Mac mini
Not used.
