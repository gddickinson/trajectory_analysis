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
- The example file's `nnDist_inFrame` column disagrees with distances recomputed from its own x/y in 42 % of rows and
  contradicts its own neighbour counts in 207 rows; how it was computed is not in the repo.

### Follow-up 2026-11-12 (coordinator: fix the two autocorrelation issues)
6. Persistence with no 1/e crossing was reported as 2 × last lag (invented). Now the last lag as a lower bound with
   `persistence_censored` ("> last lag"); censored tracks left out of persistence statistics. (`tests/test_fix_autocorr_censoring.py`)
7. The ensemble autocorrelation stopped at the shortest track's lag range (example data: 3 lags). Now every lag reached
   by at least `min_tracks` tracks (default 5, web parameter `ensemble_min_tracks`), pair-weighted mean (= mean over
   all pairs), pair-weighted SD, tracks and pairs per lag. Example data: 20 lags. (`tests/test_fix_autocorr_ensemble_lags.py`)
JS port, fixture (31,665 checks) and pages updated.

### Tests (2026-11-12)
- `pytest -q tests`: all pass (see report); `pytest -q tests/test_web.py`: all pass; node engine test ok;
  `screenshots.py --static` and `--base http://127.0.0.1:8212` all routes ok, phone overflow 0.

### Mac mini
Not used.

## 2026-11-12: sRg definition, presets and minimum track length (George's answer to the open sRg questions)

### Research (DOIs checked on Crossref)
- Golan & Sherman 2017, Nat Commun 8:15851, doi:10.1038/ncomms15851: Rg² = (1/N)Σ(rᵢ−⟨r⟩)² (Eq. 7); for an immobile
  particle Rg/⟨|Δr|⟩ = √(2/π), so "the normalized ratio √(π/2)Rg/⟨|Δr|⟩" ≈ 1 independent of the localisation error;
  "95% of the normalized ratio values fall below the value 2.11 and 99% below 3.89" (fixed cells, gold beads); tracks
  ≥ 50 frames. So 2.11 belongs to the √(π/2) definition.
- Ly, Freites, …, Dickinson, Tobias, Pathak 2025, Biophys J 124:4457, doi:10.1016/j.bpj.2025.01.010 (PIEZO1-tdTomato):
  Eq. 1 sRg = √(π/2)Rg/⟨r⟩; threshold = 95th percentile of fixed-MEF sRg (not printed; Fig. 2A dashed line ≈ 2.2);
  tracks ≥ 200 positions. 2.22236433588659 is the value in the lab code (particle_trajectory_analyzer config, from
  J. A. Freites' R code); its derivation is not printed anywhere I found.
- Bertaccini et al. 2025 Nat Commun 16:5556 (doi:10.1038/s41467-025-59150-1) does not use sRg.
- The pathak_lab scripts' `radius_gyration_scaled` = Rg/meanLag (no √(π/2)), with gap-spanning steps counted as one step
  (as here); no threshold is applied to it in those scripts.

### Calibration
`scripts/srg_calibration.py` -> `docs/srg_calibration.md` (seed 20261112, 4000 tracks per cell): ideal immobile tracks
have sRg ≈ 1 at every length (95th pct 1.04-1.2, far below 2.11: real fixed controls carry extra apparent motion);
free diffusion called immobile at 2.11: 54 % (15 points), 12 % (30), 1.4 % (50), 0 % (100) without localisation noise,
7.8 % at 50 points with noise SD = half the step SD, 38 % with noise = step (≤ 5 % needs 150 points). Rg/⟨step⟩ against
2.11 (= 2.64 on the √(π/2) scale) makes the false-immobile rate 3-8 times higher at 30-50 points. Confined diffusion reads as
immobile only when the confinement radius is ≲ 2 steps.

### Changes (separate commits)
- Fix: `calculate_features(**kwargs)` ignored per-call parameters in the sRg and linearity steps.
- Fix: tracks without an sRg (1 point, zero step) were 'immobile' (AdvancedMetricsCalculator, TrajectoryClassifier);
  the threshold classifier compared Rg in pixels with 2.11 when no mean-step column was present.
- Fix: short tracks are 'too_short' below `min_classify_length` (default 50) instead of being classified. New
  `analysis/mobility.py` shared by all modules (classification.py's default 2.2224 -> the shared 2.11; 2.2224 is the
  `piezo1_tirf_10fps` preset). Options: `srg_preset` (membrane_tirf, piezo1_tirf_10fps, fixed_reference_strict,
  fast_cytosolic, legacy, custom), `srg_definition` (golan_sherman default, rg_over_step), `mobility_threshold`,
  `min_classify_length`. `legacy` reproduces the old FeatureCalculator results.
- GUI (`gui/srg_options.py`), config/AnalysisParameters/StatisticsConfig (StatisticsGenerator also used >= and ignored
  `scaled_rg`), command line `scripts/classify_mobility.py` (incl. `--reference` fixed sample -> 95th percentile).
- Web: Analyze form (preset, definition, threshold, minimum), 'too short' class on every page, Method page with sources
  and the presets table, Length check text from the calibration; `static/mobility.js` checked against Python by the
  node test. Demo results change: example data 31 mobile / 0 immobile / 88 too short (legacy: 56 / 63 / 0).

### Tests
- `pytest -q tests`: all pass; node engine test ok (78,297 checks; mutations of the minimum-length comparison or the
  threshold conversion fail it); `screenshots.py --static` and `--base http://127.0.0.1:8212` all routes ok; phone
  width 390 px overflow 0 on every route; desktop app starts offscreen and passes the new options.

## 2026-11-13: PIEZO1 sRg threshold sourced from Ly et al. 2025 (branch `web-interface`, not pushed)
- Read the published article (Biophys J 124:4457-4470, doi:10.1016/j.bpj.2025.01.010; supporting material not in the PDF).
  Eq. 1 (p. 4461) = the preset's `golan_sherman` definition. Threshold = "the 95th percentile of a selected set of
  fixed-cell trajectories sRg distribution" (p. 4461; Fig. 2A legend), the set selected by TAMSD mixture models
  (Methods, "Fixed-cell trajectory analysis", p. 4459). No number is printed; Fig. 2A's dashed line measures
  2.23 +/- 0.03, consistent with 2.2224, so the preset value is unchanged. Not reproduced: NaN placeholders for skipped
  frames, removal of live tracks with TAMSD below the localisation error, the fixed-set selection.
- Updated: mobility.py docstring and preset basis (+ mobility.js, fixture), docs/srg_calibration.md section 5 (from
  scripts/srg_calibration.py), web Method page, config_manager comment, README pointer.
