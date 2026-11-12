// Method (what is computed, from which Python class, with units) and About (desktop app, web app, data, credits).
import { app, isStatic, route } from './core.js';
import { getInfo } from './state.js';

route('method', async () => {
  const info = await getInfo(), d = info.defaults;
  app().innerHTML = `<p class="kicker">Docs</p><h1>Method</h1>
  <p class="lead">What is computed, as in the repo's <code>particle_tracker/analysis</code> modules. All calculations run in
    pixels and frames, as the Python code does; results are converted to µm and seconds for display and export.</p>
  <h2>Units and input</h2>
  <p>One row per localisation with a track id, a frame number and x, y. Give the pixel size in µm/pixel (default ${d.pixel_size_um},
    the README's example 108 nm) and the frame interval in s (default ${d.frame_interval_s} s, the README's 10 Hz). Coordinates
    already in µm are divided by the pixel size first, because the neighbour radii are defined in pixels. Only distances are used,
    so the coordinate origin (pixel centre or corner) does not matter. Each track is sorted by frame; a track may skip frames,
    but may not have two localisations in one frame. Tracks shorter than the minimum length (default ${d.min_track_length}) are left out.</p>
  <h2>Shape (<code>features.AdvancedShapeAnalyzer</code>, tracks of ≥ 3 points)</h2>
  <ul>
    <li><b>Gyration tensor</b> T = mean of (r − r̄)(r − r̄)ᵀ, eigenvalues λ₁ ≥ λ₂, main axis e₁. <b>Rg</b> = √(λ₁ + λ₂).</li>
    <li><b>Asymmetry</b> = −ln(1 − (λ₁ − λ₂)² / (2(λ₁ + λ₂)²)); <b>eigenvalue ratio</b> λ₁/λ₂ (∞ when λ₂ = 0).</li>
    <li><b>Skewness, kurtosis</b> (biased; Fisher kurtosis) of the steps projected on e₁. The sign of the skewness follows the
      sign LAPACK gives e₁; the browser port reproduces it.</li>
    <li><b>Step alignment</b>: mean |cos| between each non-zero step and e₁. <b>Directionality</b>: net displacement / path length.
      <b>Straightness</b>: mean cosine of the turning angle between consecutive non-zero steps.</li>
    <li><b>Fractal dimension</b> (<code>track_geometry.fractal_dimension</code>): log N / log(N·d<sub>max</sub>/L), N points,
      L path length, d<sub>max</sub> the largest distance between two points; undefined when all points lie on one line through the first.</li>
    <li><b>Linearity</b> (<code>classify_linearity</code>): linear when λ₁/λ₂ ≥ ${d.linear_eigenvalue_threshold} and alignment ≥
      ${d.linear_alignment_threshold}; then unidirectional when directionality ≥ 0.7, else bidirectional.</li>
  </ul>
  <h2>Mobility (<code>features.ScaledRadiusCalculator</code>)</h2>
  <p>sRg = √(π/2)·Rg / mean step length; <b>mobile</b> when sRg &gt; ${d.mobility_threshold}, else immobile (code comments cite Golan &amp; Sherman).
    For diffusing particles sRg grows with the number of points, so the class depends on track length: see <a href="#/lengthcheck">Length check</a>.</p>
  <h2>Steps and speed (<code>FeatureCalculator._calculate_step_metrics</code>)</h2>
  <p>Step length between consecutive localisations; speed = step / frames spanned (so a step across a gap is divided by the gap);
    mean speed per track in µm/s.</p>
  <h2>MSD (<code>AdvancedMetricsCalculator._calculate_advanced_diffusion_metrics</code>)</h2>
  <p>MSD at lags of 1 to min(n − 1, ${info.msd_max_lag}) frames, averaging squared displacements over pairs exactly that many frames apart.
    A straight-line fit of log MSD against log lag over the first 5 lags gives <b>α</b> (slope) and <b>D</b> = exp(intercept) / 4,
    the fitted MSD at one frame divided by 4τ (µm²/s). Needs at least 3 lags.</p>
  <h2>Direction autocorrelation (<code>autocorrelation_analysis.DirectionAutocorrelationAnalyzer</code>)</h2>
  <p>Unit vectors of the non-zero steps; the value at a lag of k frames is the mean dot product of steps that start k frames apart,
    for k = 1 to min(${d.autocorr_max_lag}, number of non-zero steps − 1). <b>Persistence</b>: the lag where the curve first drops to
    1/e of its lag-1 value (linear interpolation); when it never does, the analyser reports 2 × the last lag, which this site
    marks as “no 1/e drop” instead of showing it as a measurement. <b>Directional bias</b>: mean of the curve. The <b>ensemble</b>
    is the mean ± SD over tracks at each lag, over the lags every track reaches. The analyser's exponential-decay fit is not shown.</p>
  <h2>Density (<code>features.DensityAnalyzer</code>, <code>FeatureCalculator._calculate_nearest_neighbors</code>)</h2>
  <p>Per frame: the distance from each localisation to its nearest neighbour (undefined when alone) and the number of others within
    ${info.density_radii.join(', ')} pixels (distance ≤ r).</p>
  <h2>Corrections made in November 2026</h2>
  <ul class="small">
    <li>Nearest-neighbour distances were attached to the wrong rows whenever the table was not sorted by frame.</li>
    <li>Shape metrics used rows in file order instead of frame order.</li>
    <li>The fractal dimension in <code>features.py</code> used the end-to-end distance instead of the largest distance used by the
      original scripts; it now reproduces the example file's <code>fracDimension</code> on all 111 tracks whose positions still reproduce the file's Rg.</li>
    <li>MSD and autocorrelation lags counted rows (and, for the autocorrelation, skipped zero-length steps), so gaps shifted lag times;
      lags are now frame differences. Gap-free tracks give the same values as before.</li>
  </ul>
  <h2>Browser and Python</h2>
  <p>The local web app runs the Python classes. The static site runs <code>engine.js</code>, a line-by-line port; a node test
    (<code>web/static/engine.test.mjs</code>) checks every per-track value and class, the ensemble curve and every density value
    against the Python output (relative tolerance 1e-9) on the example data, the synthetic tracks in µm with other parameters, and edge cases
    (gaps, zero-length steps, collinear and stationary tracks, shuffled rows).</p>`;
});

route('about', async () => {
  const info = await getInfo();
  app().innerHTML = `<p class="kicker">Docs</p><h1>About</h1>
  <p class="lead">Particle Tracking Application: a comprehensive, modular application for analysing particle trajectories from TIRF
    microscopy recordings and other single-molecule imaging data. This web interface covers the analysis of finished tracks.</p>
  <h2>Desktop app (full pipeline)</h2>
  <p>The PyQt6 app loads image stacks or localisation / track files, detects particles (threshold, LoG, trackpy), links them into
    trajectories, computes the features shown here, classifies tracks (SVM trained on labelled features, or thresholds) and exports
    results. It also does background subtraction, localisation precision and trajectory interpolation, which need images or
    are not part of this site.</p>
  <pre>pip install -r requirements.txt
python main.py</pre>
  <h2>Web app</h2>
  <pre>pip install -r requirements-web.txt
python run_web.py          # http://127.0.0.1:8212</pre>
  <p>${isStatic() ? 'This static site runs the analysis in your browser: uploaded files are read locally and never sent anywhere.' : 'You are running the local app: uploads are analysed by the Python code on this computer.'}
    Accepted columns (case-insensitive): ${Object.entries(info.columns).map(([k, v]) => `<b>${k}</b> (${v.join(', ')})`).join('; ')}.
    Rows without numeric track, frame, x and y are skipped.</p>
  <h2>Demo data</h2>
  <ul>${info.demos.map(d => `<li><b>${d.title}</b> (${d.kind}): ${d.description}<br><code class="small">${d.source}</code><br><span class="small mute">Units: ${d.pixel_note}.</span></li>`).join('')}</ul>
  <p class="small mute">The example recording (PIEZO1-HaloTag in endothelial cells, JF646, TIRF, 10 ms frames binned by 10) is described in
    <code>particle_tracker/resources/example_data/README.md</code>; its tracks come from ThunderSTORM localisations linked by the original analysis
    scripts. For 8 trapped tracks the file's x / y no longer reproduce the file's own shape features (Rg etc.), consistent with the
    interpolation of trapped tracks that the original scripts did after computing features.</p>
  <h2>Credits</h2>
  <p>Built on the scientific Python ecosystem (NumPy, SciPy, scikit-learn) with a PyQt6 / pyqtgraph GUI; consolidates and extends functionality
    from earlier analysis scripts (README). Code: George Dickinson, MIT licence.
    Source: <a href="https://github.com/gddickinson/trajectory_analysis">github.com/gddickinson/trajectory_analysis</a>.</p>`;
});
