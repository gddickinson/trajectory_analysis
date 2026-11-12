// Analyze: data source, units and parameters -> run the analyses (Python server, or the JS engine on the static site)
// -> per-class summary, warnings and exports.
import { $, api, app, download, esc, isStatic, list, route } from './core.js';
import { tracksCsv } from './engine_io.js';
import { cls, count, f, median, pct, provenance, trackCsv, units, unitsText } from './fmt.js';
import { getInfo, getResult, setResult, sourceLine, tracksOf } from './state.js';

const num = (id, label, unit, attrs, hint = '') => `<label for="${id}">${label} <span class="unit">${unit}</span></label>
  <input id="${id}" name="${id}" type="number" ${attrs}>${hint ? `<p class="hint">${hint}</p>` : ''}`;
const KEYS = ['coord_unit', 'pixel_size_um', 'frame_interval_s', 'min_track_length', 'mobility_threshold',
  'linear_eigenvalue_threshold', 'linear_alignment_threshold', 'autocorr_max_lag'];
const ANALYSIS_KEYS = KEYS.slice(3);

route('analyze', async () => {
  const [info, result] = await Promise.all([getInfo(), getResult()]);
  const d = info.defaults, cols = info.columns;
  app().innerHTML = `<p class="kicker">Run the analysis</p><h1>Analyze tracks</h1>
  <p class="lead">Per track: shape from the gyration tensor (Rg, asymmetry, skewness, kurtosis, linearity), scaled Rg and a
    mobile / immobile class, fractal dimension, straightness, step sizes and speed, MSD exponent α and D, and the direction
    autocorrelation; per localisation: nearest-neighbour distance and neighbour counts.
    ${isStatic() ? 'Runs in your browser; the file never leaves your computer.' : 'Runs the repo\'s Python classes on this computer.'}</p>
  <div class="side">
  <form class="params panel" id="pf" novalidate>
    <fieldset class="src"><legend>Data</legend>
      ${list(info.demos, x => `<label><input type="radio" name="source" value="${x.id}"> <span>${esc(x.title)}<br><small class="mute">${x.n_tracks} tracks · ${x.kind} · labelled</small></span></label>`)}
      <label><input type="radio" name="source" value="upload"> <span>Upload a CSV file</span></label>
      <div id="upwrap" class="hide"><label for="file">Track CSV</label>
        <input id="file" name="file" type="file" accept=".csv,text/csv">
        <p class="hint">One row per localisation. Columns (case-insensitive, first match): track = ${cols.track_number.map(esc).join(' / ')};
          frame = ${cols.frame.map(esc).join(' / ')}; x = ${cols.x.map(esc).join(' / ')}; y = ${cols.y.map(esc).join(' / ')}.
          Optional per-track text label: ${info.label_columns.map(esc).join(' / ')}.</p></div>
    </fieldset>
    <fieldset><legend>Units</legend>
      <label for="coord_unit">x / y columns are in</label>
      <select id="coord_unit" name="coord_unit"><option value="px">pixels</option><option value="um">µm</option></select>
      ${num('pixel_size_um', 'Pixel size', 'µm / pixel', 'min="0" step="any"', 'Always needed: lengths are reported in µm, and the neighbour radii (3–30) are in pixels.')}
      ${num('frame_interval_s', 'Frame interval', 's / frame', 'min="0" step="any"', 'Speeds come out in µm/s, D in µm²/s, persistence in s.')}
    </fieldset>
    <fieldset><legend>Parameters</legend>
      ${num('min_track_length', 'Minimum track length', 'points', 'min="2" step="1"', 'Shorter tracks are left out of every analysis (shape metrics need 3).')}
      ${num('mobility_threshold', 'Mobile above', 'scaled Rg', 'step="any"', `FeatureCalculator default ${d.mobility_threshold}. See Length check.`)}
      ${num('linear_eigenvalue_threshold', 'Linear if eigenvalue ratio ≥', 'λ₁/λ₂', 'step="any"')}
      ${num('linear_alignment_threshold', 'and step alignment ≥', '0–1', 'min="0" max="1" step="any"')}
      ${num('autocorr_max_lag', 'Autocorrelation lags up to', 'frames', 'min="1" step="1"')}
    </fieldset>
    <button type="submit" id="run">Run analysis</button><button type="button" class="alt" id="reset">Default parameters</button>
    <p id="msg" class="small" role="status"></p>
  </form>
  <div id="out"></div></div>`;

  const form = $('#pf');
  const set = (id, v) => { $('#' + id).value = v ?? ''; };
  const toggle = () => { $('#upwrap').classList.toggle('hide', form.source.value !== 'upload'); };
  for (const k of KEYS) set(k, result.params[k]);
  (form.querySelector(`input[name=source][value="${result.source?.id}"]`) || form.querySelector('input[name=source][value=upload]')).checked = true;
  toggle();
  form.addEventListener('change', e => {
    if (e.target.name === 'source' && e.target.value !== 'upload') {
      const m = info.demos.find(x => x.id === e.target.value);
      set('coord_unit', m.coord_unit); set('pixel_size_um', m.pixel_size_um); set('frame_interval_s', m.frame_interval_s);
    }
    toggle();
  });
  $('#reset').onclick = () => { for (const k of ANALYSIS_KEYS) set(k, d[k]); };

  form.addEventListener('submit', async e => {
    e.preventDefault();
    const fd = new FormData(), msg = $('#msg');
    if (form.source.value === 'upload') {
      const file = $('#file').files[0];
      if (!file) { msg.innerHTML = '<span class="err">Choose a CSV file first.</span>'; return; }
      fd.append('file', file);
    } else fd.append('demo', form.source.value);
    for (const k of KEYS) {
      const v = $('#' + k).value;
      if (v === '') { msg.innerHTML = `<span class="err">${esc(k)} is empty.</span>`; return; }
      fd.append(k, v);
    }
    $('#run').disabled = true; msg.textContent = 'Running...';
    try {
      const r = setResult(await api('/analyze', { body: fd }));
      msg.innerHTML = `<span class="ok">Done: ${r.n_tracks} tracks analysed.</span>`;
      show(r);
    } catch (err) {
      msg.innerHTML = `<span class="err">${esc(err.message)}</span>`;
    } finally { $('#run').disabled = false; }
  });
  show(result);
});

function show(r) {
  const p = r.params, t = r.tracks, n = t.length, u = units(p);
  const mob = count(t, 'mobility_classification'), lin = count(t, 'linear_classification');
  const classes = ['mobile', 'immobile', 'unclassified'].filter(c => mob.get(c));
  const gaps = t.filter(x => x.missing_frames > 0).length;
  const row = c => {
    const s = t.filter(x => x.mobility_classification === c);
    return `<tr><td>${cls(c)}</td><td>${s.length}</td><td>${f(median(s.map(x => u.um(x.radius_gyration))))}</td>
      <td>${f(median(s.map(x => x.scaled_rg)))}</td><td>${f(median(s.map(x => u.D(x.msd_D))))}</td><td>${f(median(s.map(x => x.msd_alpha)), 3)}</td>
      <td>${f(median(s.map(x => u.speed(x.meanVelocity))))}</td><td>${f(median(s.map(x => x.n_points)))}</td></tr>`;
  };
  $('#out').innerHTML = `${provenance(r)}
  <p class="small mute">${esc(sourceLine(r))}<br>${esc(unitsText(p))}</p>
  ${r.source?.pixel_note ? `<p class="small mute">Units: ${esc(r.source.pixel_note)}.</p>` : ''}
  <div class="tiles">
    <div class="tile"><b>${n}</b><span>tracks (≥ ${p.min_track_length} points)</span></div>
    <div class="tile"><b>${pct(mob.get('mobile') || 0, n)}</b><span>mobile (scaled Rg &gt; ${f(p.mobility_threshold)})</span></div>
    <div class="tile"><b>${(lin.get('linear_unidirectional') || 0) + (lin.get('linear_bidirectional') || 0)}</b><span>linear tracks</span></div>
    <div class="tile"><b>${f(median(r.density.nn_distance.map(v => u.um(v))), 3)} µm</b><span>median nearest-neighbour distance</span></div>
  </div>
  ${gaps ? `<p class="small mute">${gaps} tracks have missing frames. MSD and autocorrelation lags are frame differences, so their lag times are right; step speeds divide by the frames each step spans.</p>` : ''}
  <div class="scroll"><table class="tbl"><thead><tr><th>class</th><th>tracks</th><th>median Rg <span class="unit">µm</span></th><th>median sRg</th>
    <th>median D <span class="unit">µm²/s</span></th><th>median α</th><th>median speed <span class="unit">µm/s</span></th><th>median points</th></tr></thead>
    <tbody>${list(classes, row)}</tbody></table></div>
  <p class="small mute">Linearity: ${list([...lin.entries()], ([k, v]) => `${cls(k)} ${v}`, ' · ')}</p>
  <h3>Export</h3>
  <button type="button" id="x-tracks">Per-track CSV</button>
  <button type="button" class="alt" id="x-json">Result JSON</button>
  <button type="button" class="alt" id="x-input">Track table CSV</button>
  <p class="small mute">The per-track CSV has the units in its column names and the parameters in its first line.</p>
  <p><a href="#/tracks">Tracks →</a> · <a href="#/features">Features →</a> · <a href="#/motion">Motion →</a> · <a href="#/density">Density →</a> · <a href="#/lengthcheck">Length check →</a></p>`;
  const stem = (r.name || 'tracks').replace(/\.csv$/i, '');
  $('#x-tracks').onclick = () => download(`${stem}_track_features.csv`, trackCsv(r), 'text/csv');
  $('#x-json').onclick = () => download(`${stem}_analysis.json`, JSON.stringify({ ...r, coords: undefined, _tracks: undefined }), 'application/json');
  $('#x-input').onclick = async () => download(`${stem}_tracks.csv`, tracksCsv(await tracksOf(r)), 'text/csv');
}
