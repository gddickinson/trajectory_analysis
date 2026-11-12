// The current analysis result, shared by all result pages. Until the user runs an analysis, the first demo dataset
// analysed by the Python code (default parameters) is shown.
import { api } from './core.js';
import { fromCoords } from './engine_io.js';

let info = null, current = null;
export async function getInfo() { return info || (info = await api('/info')); }

export async function demoResult(id) {
  return { ...(await api(`/demo/${id}/result`)), coords_ref: `/demo/${id}/tracks` };
}

export function setResult(r) { current = r; return r; }

export async function getResult() {
  if (current) return current;
  const i = await getInfo();
  return setResult(await demoResult(i.demos[0].id));
}

// Track list [{id, key, frames, xs, ys, label}] of a result, in the result's track order (coordinates as in the file).
export async function tracksOf(r) {
  if (r._tracks) return r._tracks;
  let list;
  if (r.coords) list = fromCoords(r.coords);
  else { const d = await api(r.coords_ref); list = fromCoords(d.coords, d.labels); }
  const byKey = new Map(list.map(t => [t.key, t]));
  r._tracks = r.tracks.map(t => byKey.get(t.track)).filter(Boolean);
  return r._tracks;
}

// Coordinates of a track in µm.
export function trackUm(t, p) {
  const k = p.coord_unit === 'um' ? 1 : p.pixel_size_um;
  return { xs: t.xs.map(v => v * k), ys: t.ys.map(v => v * k) };
}

export function sourceLine(r) {
  const s = r.source || {};
  const dropped = r.n_tracks_dropped ? ` (${r.n_tracks_dropped} shorter tracks, ${r.n_rows_dropped} rows, left out)` : '';
  return `${s.title || r.name} · ${r.n_tracks} tracks · ${r.n_rows.toLocaleString()} localisations${dropped}`;
}
