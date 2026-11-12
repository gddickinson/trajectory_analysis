// Track table I/O for the browser engine: CSV parsing, column names, parameters, units. Mirrors web/tracks_io.py.
// Coordinates are used as given (an origin offset cancels in every distance computed here); µm input is divided by
// the pixel size because the analysis works in pixels (its density radii are pixel radii).

export const COLUMNS = {
  track_number: ['track_number', 'track_id', 'particle', 'trajectory_id', 'track'],
  frame: ['frame', 'frames', 'frame_number', 'frameno', 'time', 'f', '#frame'],
  x: ['x', 'x_coord', 'x_coordinate', 'xpos', 'x_position', 'x-position', 'x position'],
  y: ['y', 'y_coord', 'y_coordinate', 'ypos', 'y_position', 'y-position', 'y position'],
};
export const LABEL_COLUMNS = ['label', 'reference_label', 'class'];
export const DEFAULTS = {
  coord_unit: 'px', pixel_size_um: 0.108, frame_interval_s: 0.1, min_track_length: 3, mobility_threshold: 2.11,
  linear_eigenvalue_threshold: 20.0, linear_alignment_threshold: 0.7, autocorr_max_lag: 20,
};
export const DENSITY_RADII = [3, 5, 10, 20, 30];

export class AnalysisError extends Error {}

// Minimal RFC-4180 CSV reader: quoted fields, "" escapes, CRLF; blank lines skipped (as pandas does).
export function parseCsv(text) {
  const rows = [];
  let row = [], field = '', q = false;
  const s = String(text).replace(/^﻿/, '');
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (q) {
      if (c === '"') { if (s[i + 1] === '"') { field += '"'; i++; } else q = false; } else field += c;
    } else if (c === '"') q = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && s[i + 1] === '\n') i++;
      row.push(field); field = '';
      if (row.length > 1 || row[0] !== '') rows.push(row);
      row = [];
    } else field += c;
  }
  row.push(field);
  if (row.length > 1 || row[0] !== '') rows.push(row);
  if (!rows.length) throw new AnalysisError('The file is empty.');
  return { header: rows[0], rows: rows.slice(1) };
}

// Strings pandas.read_csv reads as missing (its default na_values).
const NA = new Set(['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A',
  'NA', 'NULL', 'NaN', 'None', 'n/a', 'nan', 'null']);
export const isNA = v => NA.has(String(v ?? '').trim());
export const toNum = v => (isNA(v) ? NaN : Number(String(v).trim()));
// JSON numbers from the Python side: null = NaN, 'inf' / '-inf' strings.
export const num = v => (v === null || v === undefined ? NaN : v === 'inf' ? Infinity : v === '-inf' ? -Infinity : Number(v));

export function findColumn(header, options) {
  const lower = new Map();
  header.forEach((h, i) => { const k = String(h).trim().toLowerCase(); if (!lower.has(k)) lower.set(k, i); });
  for (const o of options) if (lower.has(o.toLowerCase())) return lower.get(o.toLowerCase());
  return null;
}

// Track id as text, as tracks_io.key: integers without '.0'.
export const trackKey = t => (Number.isInteger(t) ? String(t) : String(t));

// Parsed CSV -> rows [{track, frame, x, y, label}] sorted by track then frame (tracks_io.standardise).
export function standardise({ header, rows }) {
  const idx = {}, missing = [];
  for (const [k, opts] of Object.entries(COLUMNS)) {
    idx[k] = findColumn(header, opts);
    if (idx[k] === null) missing.push(`${k} (one of ${opts.join(', ')})`);
  }
  if (missing.length) throw new AnalysisError('Missing column(s): ' + missing.join(', '));
  const li = findColumn(header, LABEL_COLUMNS);
  const out = [];
  for (const r of rows) {
    const t = toNum(r[idx.track_number]), f = toNum(r[idx.frame]), x = toNum(r[idx.x]), y = toNum(r[idx.y]);
    if ([t, f, x, y].some(Number.isNaN)) continue;
    const row = { track: t, frame: f, x, y };
    if (li !== null) row.label = isNA(r[li]) ? '' : String(r[li]);
    out.push(row);
  }
  if (!out.length) throw new AnalysisError('No rows with numeric track, frame, x and y values.');
  if (out.some(r => r.frame !== Math.round(r.frame))) throw new AnalysisError('Frame numbers must be whole numbers.');
  out.forEach((r, i) => { r._i = i; });
  out.sort((a, b) => a.track - b.track || a.frame - b.frame || a._i - b._i);
  for (let i = 1; i < out.length; i++) {
    if (out[i].track === out[i - 1].track && out[i].frame === out[i - 1].frame) {
      throw new AnalysisError('A track has two localisations in the same frame; one per frame is expected.');
    }
  }
  for (const r of out) delete r._i;
  return { rows: out, hasLabels: li !== null };
}

// Sorted rows -> tracks [{id, key, frames, xs, ys, label}] in track order.
export function toTracks({ rows, hasLabels }) {
  const tracks = [];
  let cur = null;
  for (const r of rows) {
    if (!cur || cur.id !== r.track) {
      cur = { id: r.track, key: trackKey(r.track), frames: [], xs: [], ys: [] };
      if (hasLabels) cur.label = r.label;
      tracks.push(cur);
    }
    cur.frames.push(r.frame); cur.xs.push(r.x); cur.ys.push(r.y);
  }
  return tracks;
}

// Compact coordinates {key: [frames, xs, ys]} (tracks_io.coords) and back.
export function toCoords(tracks) {
  return Object.fromEntries(tracks.map(t => [t.key, [t.frames, t.xs, t.ys]]));
}
export function fromCoords(coords, labels = null) {
  const list = Object.entries(coords).map(([k, [frames, xs, ys]]) => {
    const t = { id: Number(k), key: k, frames, xs, ys };
    if (labels) t.label = labels[k] ?? '';
    return t;
  });
  list.sort((a, b) => a.id - b.id);
  return list;
}

// Typed, checked parameters (tracks_io.parse_params); missing values take the defaults.
export function checkParams(raw) {
  const p = {};
  for (const [k, d] of Object.entries(DEFAULTS)) {
    const v = raw[k];
    if (v === undefined || v === null || v === '') { p[k] = d; continue; }
    if (k === 'coord_unit') {
      if (v !== 'px' && v !== 'um') throw new AnalysisError("coord_unit must be 'px' or 'um'");
      p[k] = v; continue;
    }
    const x = Number(v);
    if (!Number.isFinite(x)) throw new AnalysisError(`${k} must be a finite number`);
    if ((k === 'min_track_length' || k === 'autocorr_max_lag') && !Number.isInteger(x)) {
      throw new AnalysisError(`${k} must be a whole number`);
    }
    p[k] = x;
  }
  if (p.pixel_size_um <= 0) throw new AnalysisError('pixel_size_um must be > 0');
  if (p.frame_interval_s <= 0) throw new AnalysisError('frame_interval_s must be > 0');
  if (p.min_track_length < 2) throw new AnalysisError('min_track_length must be at least 2');
  if (p.autocorr_max_lag < 1 || p.autocorr_max_lag > 1000) throw new AnalysisError('autocorr_max_lag must be between 1 and 1000');
  return p;
}

// CSV of the track table (pixels or the input units), for export.
export function tracksCsv(tracks) {
  const lines = ['track_number,frame,x,y' + (tracks.some(t => 'label' in t) ? ',label' : '')];
  const q = s => (/[",\n]/.test(s) ? `"${String(s).replace(/"/g, '""')}"` : s);
  for (const t of tracks) {
    t.frames.forEach((f, i) => lines.push(`${t.key},${f},${t.xs[i]},${t.ys[i]}` + ('label' in t ? ',' + q(t.label ?? '') : '')));
  }
  return lines.join('\n') + '\n';
}
