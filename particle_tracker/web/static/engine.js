// Browser engine: the same analysis as web/analysis.py (FeatureCalculator, the MSD fit of AdvancedMetricsCalculator
// and DirectionAutocorrelationAnalyzer), returning the same JSON shape. Used on the static site for uploads and
// re-runs; checked against the Python code by engine.test.mjs. Everything is in pixels and frames; the pages convert
// to µm and s with result.params.
import { DENSITY_RADII, AnalysisError, toTracks } from './engine_io.js';
import { frameNeighbours, nanMean } from './engine_stats.js';
import { autocorrelation, ensembleAutocorr, linearity, meanVelocity, msdFit, scaledRg, shapeMetrics } from './engine_track.js';

export * from './engine_io.js';

const out = v => (Number.isNaN(v) ? null : v === Infinity ? 'inf' : v === -Infinity ? '-inf' : v);

// One track (pixels) -> the per-track record of the result.
export function trackRecord(t, p) {
  const { frames, xs, ys } = t, n = xs.length;
  const shape = shapeMetrics(xs, ys), srg = scaledRg(xs, ys, shape.radius_gyration, p.mobility_threshold);
  const msd = msdFit(frames, xs, ys), ac = autocorrelation(frames, xs, ys, p.autocorr_max_lag);
  const span = frames[n - 1] - frames[0] + 1;
  const r = {
    track: t.key, n_points: n, duration_frames: span, missing_frames: span - n,
    radius_gyration: out(shape.radius_gyration), asymmetry: out(shape.asymmetry), skewness: out(shape.skewness),
    kurtosis: out(shape.kurtosis), eigenvalue_ratio: out(shape.eigenvalue_ratio), step_alignment: out(shape.step_alignment),
    directionality_ratio: out(shape.directionality_ratio), fracDimension: out(shape.fracDimension),
    netDispl: out(shape.netDispl), Straight: out(shape.Straight), mean_step_size: out(srg.mean_step_size),
    scaled_rg: out(srg.scaled_rg), meanVelocity: out(meanVelocity(frames, xs, ys)),
    linear_classification: linearity(shape, p.linear_eigenvalue_threshold, p.linear_alignment_threshold),
    mobility_classification: srg.mobility_classification,
    msd_alpha: out(msd.alpha), msd_D: out(msd.D),
    autocorr_steps: ac.n_steps, persistence_length: out(ac.persistence), persistence_censored: ac.censored,
    directional_bias: out(ac.bias),
  };
  if ('label' in t) r.label = t.label;
  return { record: r, curve: ac.values };
}

// Per-localisation nearest-neighbour distance and neighbour counts (track / frame order), per-frame summary.
function density(tracks) {
  const byFrame = new Map();
  let k = 0;
  const rows = [];
  for (const t of tracks) t.frames.forEach((f, i) => {
    rows.push({ f, x: t.xs[i], y: t.ys[i] });
    if (!byFrame.has(f)) byFrame.set(f, []);
    byFrame.get(f).push(k++);
  });
  const nn = new Array(rows.length).fill(NaN), counts = DENSITY_RADII.map(() => new Array(rows.length).fill(0));
  const frames = [...byFrame.keys()].sort((a, b) => a - b), perN = [], perNn = [];
  for (const f of frames) {
    const idx = byFrame.get(f), r = frameNeighbours(idx.map(i => rows[i].x), idx.map(i => rows[i].y), DENSITY_RADII);
    idx.forEach((i, j) => { nn[i] = r.nn[j]; DENSITY_RADII.forEach((_, m) => { counts[m][i] = r.counts[m][j]; }); });
    perN.push(idx.length); perNn.push(nanMean(idx.map(i => nn[i])));
  }
  return {
    radii: DENSITY_RADII, nn_distance: nn.map(out),
    counts: Object.fromEntries(DENSITY_RADII.map((r, m) => [String(r), counts[m]])),
    frames, per_frame_n: perN, per_frame_mean_nn: perNn.map(out),
  };
}

// std = {rows, hasLabels} from standardise(); p = checkParams(...). Returns the analysis result.
export function analyse(std, p, name) {
  const all = toTracks(std);
  const kept = all.filter(t => t.xs.length >= p.min_track_length);
  if (!kept.length) throw new AnalysisError(`No track has at least ${p.min_track_length} points.`);
  const k = p.coord_unit === 'um' ? 1 / p.pixel_size_um : 1;
  const px = kept.map(t => ({ ...t, xs: k === 1 ? t.xs : t.xs.map(v => v / p.pixel_size_um), ys: k === 1 ? t.ys : t.ys.map(v => v / p.pixel_size_um) }));
  const tracks = [], curves = [];
  for (const t of px) {
    const { record, curve } = trackRecord(t, p);
    tracks.push(record);
    if (curve.length) curves.push(curve);
  }
  const nRows = kept.reduce((s, t) => s + t.xs.length, 0);
  return {
    name, computed_by: 'browser', params: p, n_tracks: tracks.length, n_rows: nRows,
    n_tracks_dropped: all.length - kept.length, n_rows_dropped: std.rows.length - nRows, has_labels: !!std.hasLabels,
    tracks, autocorr_ensemble: ensembleAutocorr(curves), density: density(px),
  };
}
