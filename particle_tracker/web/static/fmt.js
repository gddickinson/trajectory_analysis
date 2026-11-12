// Formatting, category colours, unit conversion (the analysis works in pixels and frames) and the per-track CSV.
import { esc } from './core.js';
import { num } from './engine_io.js';

export const COLORS = {
  mobile: '#4fd1ff', immobile: '#ff7a59', unclassified: '#7d89a8',
  non_linear: '#8fa3c8', linear_unidirectional: '#ffd23f', linear_bidirectional: '#45f0a0', linear: '#ffd23f',
};
const LABEL_PALETTE = ['#45f0a0', '#ff5c7a', '#ffb02e', '#c58cff', '#59b8ff', '#ffd23f', '#ff4fd8', '#9be564'];
const labelColors = new Map();
export function colorOf(c) {
  if (COLORS[c]) return COLORS[c];
  if (!labelColors.has(c)) labelColors.set(c, LABEL_PALETTE[labelColors.size % LABEL_PALETTE.length]);
  return labelColors.get(c);
}
export const NAMES = {
  mobile: 'mobile', immobile: 'immobile', unclassified: 'unclassified (too short)', non_linear: 'non-linear',
  linear_unidirectional: 'linear, unidirectional', linear_bidirectional: 'linear, bidirectional', linear: 'linear',
};
export const nice = s => NAMES[s] || String(s ?? '');
export const cls = s => `<span class="cls" style="--c:${colorOf(s)}">${esc(nice(s))}</span>`;
export const legend = keys => `<div class="legend">${keys.map(cls).join('')}</div>`;

// NaN/null -> '–', ±inf -> ∞, otherwise 4 significant digits (integers as is).
export function f(v, digits = 4) {
  const x = num(v);
  if (Number.isNaN(x)) return '–';
  if (!Number.isFinite(x)) return x > 0 ? '∞' : '−∞';
  if (Number.isInteger(x) && Math.abs(x) < 1e6) return String(x);
  const a = Math.abs(x);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return x.toExponential(digits - 1);
  return String(Number(x.toPrecision(digits)));
}
// Value with a unit, or '–' alone when undefined.
export const fu = (v, unit, digits = 4) => (Number.isFinite(num(v)) ? `${f(v, digits)} ${unit}` : f(v, digits));
export const pct = (a, b) => (b ? `${(100 * a / b).toFixed(1)}%` : '–');
export function median(values) {
  const v = values.map(num).filter(Number.isFinite).sort((a, b) => a - b);
  if (!v.length) return NaN;
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}
export const count = (rows, key) => rows.reduce((m, t) => m.set(t[key], (m.get(t[key]) || 0) + 1), new Map());

// Unit conversion of result values (pixels, frames) with the result's own parameters.
export function units(p) {
  const ps = p.pixel_size_um, dt = p.frame_interval_s;
  return {
    ps, dt,
    um: v => num(v) * ps,                    // length: px -> µm
    D: v => num(v) * ps * ps / dt,           // px² / frame -> µm² / s
    speed: v => num(v) * ps / dt,            // px / frame -> µm / s
    s: v => num(v) * dt,                     // frames -> s
  };
}

export function unitsText(p) {
  const xy = p.coord_unit === 'um' ? `x / y given in µm (pixel size ${f(p.pixel_size_um)} µm/pixel for the pixel radii)`
    : `x / y in pixels × ${f(p.pixel_size_um)} µm/pixel`;
  return `${xy}; frame interval ${f(p.frame_interval_s)} s (${f(1 / p.frame_interval_s)} Hz)`;
}

export function provenance(r) {
  return r.computed_by === 'python'
    ? '<span class="badge py">computed by the repo\'s Python code</span>'
    : '<span class="badge js">computed in your browser (JS port tested against the Python code)</span>';
}

// Per-track export: one row per track, units in the column names and a header comment with the parameters.
export const TRACK_EXPORT = [
  ['track', 'track_number', v => v], ['n_points', 'n_points', v => v], ['duration_frames', 'duration_frames', v => v],
  ['missing_frames', 'missing_frames', v => v], ['radius_gyration', 'radius_gyration_um', 'um'],
  ['scaled_rg', 'scaled_rg', v => v], ['mobility_classification', 'mobility_class', v => v],
  ['asymmetry', 'asymmetry', v => v], ['skewness', 'skewness', v => v], ['kurtosis', 'kurtosis', v => v],
  ['eigenvalue_ratio', 'eigenvalue_ratio', v => v], ['step_alignment', 'step_alignment', v => v],
  ['directionality_ratio', 'directionality_ratio', v => v], ['linear_classification', 'linear_class', v => v],
  ['fracDimension', 'fractal_dimension', v => v], ['netDispl', 'net_displacement_um', 'um'], ['Straight', 'straightness', v => v],
  ['mean_step_size', 'mean_step_um', 'um'], ['meanVelocity', 'mean_velocity_um_per_s', 'speed'],
  ['msd_alpha', 'msd_alpha', v => v], ['msd_D', 'msd_D_um2_per_s', 'D'],
  ['persistence_length', 'persistence_s', 's'], ['persistence_censored', 'persistence_censored_lower_bound', v => v],
  ['directional_bias', 'directional_bias', v => v],
];

export function trackCsv(r) {
  const u = units(r.params), q = s => (/[",\n]/.test(String(s)) ? `"${String(s).replace(/"/g, '""')}"` : String(s));
  const cols = TRACK_EXPORT.concat(r.has_labels ? [['label', 'reference_label', v => v]] : []);
  const cell = (t, [k, , conv]) => {
    const v = t[k];
    if (v === null || v === undefined) return '';
    if (typeof conv === 'string') { const x = u[conv](v); return Number.isFinite(x) ? x : (Number.isNaN(x) ? '' : String(x)); }
    return q(v === 'inf' ? 'inf' : v);
  };
  const p = r.params;
  const meta = `# ${r.name}; ${unitsText(p)}; min track length ${p.min_track_length}; sRg mobile above ${p.mobility_threshold}; ` +
    `linear: eigenvalue ratio >= ${p.linear_eigenvalue_threshold} and step alignment >= ${p.linear_alignment_threshold}; ` +
    `autocorrelation max lag ${p.autocorr_max_lag} frames; ensemble lags need ${p.ensemble_min_tracks} tracks; computed by ${r.computed_by}`;
  return [meta, cols.map(c => c[1]).join(','), ...r.tracks.map(t => cols.map(c => cell(t, c)).join(','))].join('\n') + '\n';
}
