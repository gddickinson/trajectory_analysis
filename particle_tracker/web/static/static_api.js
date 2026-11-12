// Static-site data layer: answers api() calls from the JSON exported by scripts/build_static.py, and runs the
// analysis itself in the browser (engine.js, tested against the Python code) for uploads and for demos analysed
// with other units or parameters. Nothing is sent anywhere: an uploaded file stays on your computer.
import { analyse, checkParams, fromCoords, parseCsv, standardise, toCoords, toTracks } from './engine.js';

const files = new Map();
async function file(rel) {
  if (!files.has(rel)) files.set(rel, fetch('api/' + rel + '.json').then(r => { if (!r.ok) throw new Error(`${r.status} ${rel}`); return r.json(); }));
  return files.get(rel);
}

export async function staticApi(path, body) {
  const clean = path.split('?')[0].replace(/^\/+/, '');
  if (clean === 'analyze') return analyze(body);
  return file(clean);
}

const KEYS = ['coord_unit', 'pixel_size_um', 'frame_interval_s', 'min_track_length', 'mobility_threshold',
  'linear_eigenvalue_threshold', 'linear_alignment_threshold', 'autocorr_max_lag'];

async function analyze(fd) {
  const p = checkParams(Object.fromEntries(KEYS.map(k => [k, fd.get(k)])));
  const upload = fd.get('file');
  if (upload && upload.name) {
    const std = standardise(parseCsv(await upload.text()));
    const r = analyse(std, p, upload.name);
    r.source = { id: 'upload', title: upload.name, description: 'Uploaded file (analysed in your browser)' };
    r.coords = toCoords(toTracks(std));
    return r;
  }
  const demo = fd.get('demo'), info = await file('info'), meta = info.demos.find(d => d.id === demo);
  if (!meta) throw new Error('Choose a demo dataset or upload a CSV file.');
  const d = await file(`demo/${demo}/tracks`), tracks = fromCoords(d.coords, d.labels);
  const rows = tracks.flatMap(t => t.frames.map((f, i) => ({ track: t.id, frame: f, x: t.xs[i], y: t.ys[i], label: t.label })));
  const r = analyse({ rows, hasLabels: true }, p, `${demo}.csv`);
  r.source = meta;
  r.coords_ref = `/demo/${demo}/tracks`;
  return r;
}
