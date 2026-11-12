// Per-track analyses for the browser engine, ported line by line from the repo's Python (all in pixels and frames):
//  shapeMetrics      features.AdvancedShapeAnalyzer.calculate_gyration_tensor_metrics (+ track_geometry.fractal_dimension)
//  linearity         features.AdvancedShapeAnalyzer.classify_linearity (directionality cut 0.7)
//  scaledRg          features.ScaledRadiusCalculator via mobility.js: sRg (definition), mobile above the threshold,
//                    'too_short' below min_classify_length points
//  meanVelocity      features.FeatureCalculator._calculate_step_metrics (tracks of >= 3 points)
//  msdFit            advanced_metrics._calculate_advanced_diffusion_metrics: MSD at 1..min(n-1, 10) frame lags
//                    (pairs exactly that many frames apart), log-log fit of the first 5 -> alpha, D = exp(b) / 4
//  autocorrelation   autocorrelation_analysis.calculate_track_autocorrelation (lags = start-frame differences)
//  persistence       _calculate_persistence_length (1/e crossing; censored at the last lag when there is none)
//  ensembleAutocorr  calculate_ensemble_autocorrelation (pair-weighted mean and SD; lags with >= N tracks)
import { eigSym2, linregress, mean, nanMean, skewKurt } from './engine_stats.js';
import { classify, scaledRgOf } from './mobility.js';

const EMPTY_SHAPE = {
  radius_gyration: NaN, asymmetry: NaN, skewness: NaN, kurtosis: NaN, eigenvalue_ratio: NaN, step_alignment: NaN,
  directionality_ratio: NaN, fracDimension: NaN, netDispl: NaN, Straight: NaN,
};

function steps(xs, ys) {
  const dx = [], dy = [], len = [];
  for (let i = 1; i < xs.length; i++) {
    dx.push(xs[i] - xs[i - 1]); dy.push(ys[i] - ys[i - 1]); len.push(Math.hypot(dx[i - 1], dy[i - 1]));
  }
  return { dx, dy, len };
}

// track_geometry.fractal_dimension: log(N) / log(N d_max / L); NaN when collinear through the first point.
export function fractalDimension(xs, ys) {
  const n = xs.length;
  if (n < 3) return NaN;
  const x0 = xs[0], y0 = ys[0], slopes = [];
  for (let i = 0; i < n; i++) {
    if (xs[i] !== x0 || ys[i] !== y0) slopes.push(xs[i] !== x0 ? (ys[i] - y0) / (xs[i] - x0) : null);
  }
  if (slopes.length < 2 || slopes.every(s => s === slopes[0])) return NaN;
  let path = 0, dmax = 0;
  for (let i = 1; i < n; i++) path += Math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]);
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) dmax = Math.max(dmax, Math.hypot(xs[j] - xs[i], ys[j] - ys[i]));
  if (!(dmax > 0 && path > 0)) return NaN;
  const den = Math.log(n * dmax / path);
  return den === 0 ? NaN : Math.log(n) / den;
}

export function shapeMetrics(xs, ys) {
  const n = xs.length;
  if (n < 3) return { ...EMPTY_SHAPE };
  const cx = mean(xs), cy = mean(ys);
  let gxx = 0, gxy = 0, gyy = 0;
  for (let i = 0; i < n; i++) { const a = xs[i] - cx, b = ys[i] - cy; gxx += a * a; gxy += a * b; gyy += b * b; }
  const { w, v } = eigSym2(gxx / n, gxy / n, gyy / n);
  const order = w[0] > w[1] ? [0, 1] : [1, 0];
  const l1 = w[order[0]], l2 = w[order[1]], axis = v[order[0]];
  const s = steps(xs, ys);
  const radius_gyration = Math.sqrt(l1 + l2);
  const q = 1 - (l1 - l2) ** 2 / (2 * (l1 + l2) ** 2);
  const asymmetry = q > 0 ? -Math.log(q) : NaN;
  const eigenvalue_ratio = l2 > 0 ? l1 / l2 : Infinity;
  const { skew, kurt } = skewKurt(s.dx.map((d, i) => d * axis[0] + s.dy[i] * axis[1]));
  const cosAxis = [];
  s.len.forEach((L, i) => { if (L > 0) cosAxis.push(Math.abs((s.dx[i] / L) * axis[0] + (s.dy[i] / L) * axis[1])); });
  const net = Math.hypot(xs[n - 1] - xs[0], ys[n - 1] - ys[0]);
  const path = s.len.reduce((a, b) => a + b, 0);
  const turn = [];
  for (let i = 0; i < s.dx.length - 1; i++) {
    const n1 = s.len[i], n2 = s.len[i + 1];
    if (n1 > 0 && n2 > 0) turn.push(Math.min(1, Math.max(-1, (s.dx[i] * s.dx[i + 1] + s.dy[i] * s.dy[i + 1]) / (n1 * n2))));
  }
  return {
    radius_gyration, asymmetry, skewness: skew, kurtosis: kurt, eigenvalue_ratio,
    step_alignment: cosAxis.length ? mean(cosAxis) : NaN,
    directionality_ratio: path > 0 ? net / path : NaN,
    fracDimension: fractalDimension(xs, ys), netDispl: net, Straight: turn.length ? mean(turn) : NaN,
  };
}

export function linearity(m, eigThr, alignThr) {
  if (Number.isNaN(m.eigenvalue_ratio) || Number.isNaN(m.step_alignment)) return 'unclassified';
  if (!(m.eigenvalue_ratio >= eigThr && m.step_alignment >= alignThr)) return 'non_linear';
  if (Number.isNaN(m.directionality_ratio)) return 'linear';
  return m.directionality_ratio >= 0.7 ? 'linear_unidirectional' : 'linear_bidirectional';
}

// p: resolved params (srg_definition, mobility_threshold, min_classify_length; mobility.js resolve).
export function scaledRg(xs, ys, rg, p) {
  const n = xs.length, cls = srg => classify(srg, n, p.mobility_threshold, p.min_classify_length);
  if (n < 2) return { mean_step_size: NaN, scaled_rg: NaN, mobility_classification: cls(NaN) };
  const meanStep = mean(steps(xs, ys).len);
  const srg = scaledRgOf(rg, meanStep, p.srg_definition);
  return { mean_step_size: meanStep, scaled_rg: srg, mobility_classification: cls(srg) };
}

export function meanVelocity(frames, xs, ys) {
  if (xs.length < 3) return NaN;
  const s = steps(xs, ys);
  return nanMean(s.len.map((L, i) => { const v = L / (frames[i + 1] - frames[i]); return Number.isFinite(v) ? v : NaN; }));
}

// MSD at lags 1..maxLag frames (track_geometry.frame_lag_msd): only lags with at least one pair.
export function frameLagMsd(frames, xs, ys, maxLag) {
  const index = new Map();
  frames.forEach((f, i) => { if (!index.has(f)) index.set(f, i); });
  const lags = [], msd = [], pairs = [];
  for (let lag = 1; lag <= maxLag; lag++) {
    let s = 0, k = 0;
    for (const [f, i] of index) {
      const j = index.get(f + lag);
      if (j === undefined) continue;
      s += (xs[j] - xs[i]) ** 2 + (ys[j] - ys[i]) ** 2; k++;
    }
    if (k) { lags.push(lag); msd.push(s / k); pairs.push(k); }
  }
  return { lags, msd, pairs };
}

export function msdFit(frames, xs, ys) {
  const none = { alpha: NaN, D: NaN };
  if (xs.length < 3) return none;
  const { lags, msd } = frameLagMsd(frames, xs, ys, Math.min(xs.length - 1, 10));
  if (msd.length < 3) return none;
  const k = Math.min(5, lags.length);
  const { slope, intercept } = linregress(lags.slice(0, k).map(Math.log), msd.slice(0, k).map(Math.log));
  return { alpha: slope, D: Math.exp(intercept) / 4 };
}

export function persistence(lags, ac) {
  const vl = [], va = [];
  ac.forEach((v, i) => { if (!Number.isNaN(v)) { vl.push(lags[i]); va.push(v); } });
  if (va.length < 2 || va[0] <= 0) return { value: NaN, censored: false };
  const target = va[0] / Math.E, c = va.findIndex(v => v <= target);
  if (c < 0) return { value: vl[vl.length - 1], censored: true };   // censored: only known to exceed the last lag
  if (c === 0) return { value: vl[0], censored: false };
  const [x1, x2, y1, y2] = [vl[c - 1], vl[c], va[c - 1], va[c]];
  return { value: y2 !== y1 ? x1 + (target - y1) * (x2 - x1) / (y2 - y1) : x1, censored: false };
}

export function autocorrelation(frames, xs, ys, maxLagParam) {
  const empty = { n_steps: 0, lags: [], values: [], pairs: [], persistence: NaN, censored: false, bias: NaN };
  if (xs.length < 3) return empty;
  const s = steps(xs, ys), ux = [], uy = [], sf = [];
  s.len.forEach((L, i) => { if (L > 0) { ux.push(s.dx[i] / L); uy.push(s.dy[i] / L); sf.push(frames[i]); } });
  if (ux.length < 2) return empty;
  const maxLag = Math.min(maxLagParam ?? 20, ux.length - 1);
  if (maxLag < 1) return empty;
  const index = new Map();
  sf.forEach((f, i) => { if (!index.has(f)) index.set(f, i); });
  const lags = [], values = [], pairs = [];
  for (let lag = 1; lag <= maxLag; lag++) {
    const dots = [];
    for (const [f, i] of index) {
      const j = index.get(f + lag);
      if (j !== undefined) dots.push(Math.min(1, Math.max(-1, ux[i] * ux[j] + uy[i] * uy[j])));
    }
    lags.push(lag); values.push(dots.length ? mean(dots) : NaN); pairs.push(dots.length);
  }
  const per = persistence(lags, values), ok = values.filter(v => !Number.isNaN(v));
  return { n_steps: ux.length, lags, values, pairs, persistence: per.value, censored: per.censored, bias: ok.length ? mean(ok) : NaN };
}

// curves: per-track {values, pairs} (tracks with at least one lag), in track order. Pair-weighted mean and SD at every
// lag reached by at least minTracks tracks (calculate_ensemble_autocorrelation).
export function ensembleAutocorr(curves, minTracks = 5) {
  if (!curves.length) return null;
  const lags = [], meanv = [], sd = [], nT = [], nP = [];
  const longest = Math.max(...curves.map(c => c.values.length));
  for (let k = 0; k < longest; k++) {
    const v = [], w = [];
    for (const c of curves) if (k < c.values.length && c.pairs[k] > 0) { v.push(c.values[k]); w.push(c.pairs[k]); }
    if (v.length < Math.max(minTracks, 1)) continue;
    let sw = 0, swv = 0;
    v.forEach((x, i) => { sw += w[i]; swv += w[i] * x; });
    const m = swv / sw;
    let ss = 0;
    v.forEach((x, i) => { ss += w[i] * (x - m) ** 2; });
    lags.push(k + 1); meanv.push(m); sd.push(Math.sqrt(ss / sw)); nT.push(v.length); nP.push(sw);
  }
  if (!lags.length) return null;
  const per = persistence(lags, meanv);
  return {
    n_tracks: curves.length, lags, mean: meanv, std: sd, n_tracks_per_lag: nT, n_pairs_per_lag: nP, min_tracks: minTracks,
    persistence_length: per.value, persistence_censored: per.censored,
  };
}
