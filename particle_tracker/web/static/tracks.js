// Tracks: all tracks on a map (coloured by mobility, linearity or reference label), a sortable per-track table, and one
// track in detail: trajectory in µm, every feature, its MSD with the fit behind α and D, its direction autocorrelation.
import { $, $$, app, esc, list, route } from './core.js';
import { drawMap, lineChart, msdPlot, trackSvg } from './charts.js';
import { num } from './engine_io.js';
import { autocorrelation, frameLagMsd } from './engine_track.js';
import { linregress } from './engine_stats.js';
import { cls, colorOf, f, fu, legend, provenance, units } from './fmt.js';
import { getResult, sourceLine, trackUm, tracksOf } from './state.js';

const PAGE = 25;
const COLS = [['track', 'track'], ['n_points', 'points'], ['radius_gyration', 'Rg (µm)'], ['scaled_rg', 'sRg'],
  ['mobility_classification', 'mobility'], ['linear_classification', 'linearity'], ['msd_alpha', 'α'],
  ['msd_D', 'D (µm²/s)'], ['persistence_length', 'persistence (s)']];
let sortKey = 'track', sortDir = 1, page = 0, selected = null, colorBy = 'mobility_classification';

route('tracks', async (args, query) => {
  const r = await getResult(), tracks = await tracksOf(r), u = units(r.params);
  const rows = r.tracks;
  if (!r.has_labels && colorBy === 'label') colorBy = 'mobility_classification';
  if (query.id && rows.some(t => t.track === query.id)) selected = query.id;
  if (!rows.some(t => t.track === selected)) selected = rows[0]?.track ?? null;
  const byKey = new Map(rows.map(t => [t.track, t]));
  app().innerHTML = `<p class="kicker">Per track</p><h1>Tracks</h1>
  <p class="small mute">${provenance(r)} ${esc(sourceLine(r))}</p>
  <label for="colorby">Colour tracks by</label>
  <select id="colorby"><option value="mobility_classification">mobility (scaled Rg)</option><option value="linear_classification">linearity</option>
    ${r.has_labels ? '<option value="label">reference label</option>' : ''}</select>
  <div id="lg"></div>
  <div class="two">
    <div><div class="mapwrap"><canvas id="map" width="900" height="500" aria-label="Map of all tracks; click a track to select it"></canvas></div>
      <p class="small mute">Image orientation (y down), coordinates as in the file. Click near a track to select it.</p></div>
    <div id="detail" class="panel" aria-live="polite"></div>
  </div>
  <h2>All tracks</h2>
  <div class="scroll"><table class="tbl"><thead><tr>${list(COLS, ([k, l]) => `<th><button type="button" data-k="${k}">${esc(l)}</button></th>`)}${r.has_labels ? '<th><button type="button" data-k="label">reference label</button></th>' : ''}</tr></thead><tbody id="tb"></tbody></table></div>
  <div class="pager"><button type="button" class="alt" id="prev">Previous</button><span id="pg" class="small mute"></span><button type="button" class="alt" id="next">Next</button></div>`;

  const canvas = $('#map');
  let tf = null;
  const colour = key => colorOf(byKey.get(key)?.[colorBy]);
  const redraw = () => {
    $('#lg').innerHTML = legend([...new Set(rows.map(t => t[colorBy]))]);
    tf = drawMap(canvas, tracks.map(t => ({ ...t, id: t.key })), colour, selected);
  };
  const pick = key => { selected = key; redraw(); detail(r, tracks, key, u); table(); };
  $('#colorby').value = colorBy;
  $('#colorby').onchange = e => { colorBy = e.target.value; redraw(); };
  canvas.addEventListener('click', e => {
    const b = canvas.getBoundingClientRect(), s = canvas.width / b.width;
    const X = (e.clientX - b.left) * s, Y = (e.clientY - b.top) * s;
    let best = null, bd = 12 * s;
    for (const t of tracks) for (let i = 0; i < t.xs.length; i++) {
      const d = Math.hypot(tf.pad + (t.xs[i] - tf.x0) * tf.k - X, tf.pad + (t.ys[i] - tf.y0) * tf.k - Y);
      if (d < bd) { bd = d; best = t.key; }
    }
    if (best !== null) pick(best);
  });

  const sorted = () => {
    if (sortKey === 'track') return sortDir === 1 ? rows : [...rows].reverse();
    const v = t => (typeof t[sortKey] === 'string' && !['inf', '-inf'].includes(t[sortKey]) ? t[sortKey] : num(t[sortKey]));
    return [...rows].sort((a, b) => {
      const x = v(a), y = v(b);
      if (typeof x === 'string') return sortDir * String(x).localeCompare(String(y));
      if (Number.isNaN(x)) return 1; if (Number.isNaN(y)) return -1;
      return sortDir * (x - y);
    });
  };
  const table = () => {
    const all = sorted(), pages = Math.max(1, Math.ceil(all.length / PAGE));
    page = Math.min(page, pages - 1);
    $('#tb').innerHTML = list(all.slice(page * PAGE, page * PAGE + PAGE), t => `<tr class="click${t.track === selected ? ' sel' : ''}" tabindex="0" data-id="${esc(t.track)}">
      <td>${esc(t.track)}</td><td>${t.n_points}</td><td>${f(u.um(t.radius_gyration))}</td><td>${f(t.scaled_rg, 3)}</td>
      <td>${cls(t.mobility_classification)}</td><td>${cls(t.linear_classification)}</td><td>${f(t.msd_alpha, 3)}</td>
      <td>${f(u.D(t.msd_D))}</td><td>${t.persistence_censored ? '&gt; ' : ''}${f(u.s(t.persistence_length))}</td>
      ${r.has_labels ? `<td>${esc(t.label)}</td>` : ''}</tr>`);
    $('#pg').textContent = `page ${page + 1} of ${pages} · ${all.length} tracks`;
    $$('#tb tr').forEach(tr => {
      tr.onclick = () => pick(tr.dataset.id);
      tr.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(tr.dataset.id); } };
    });
  };
  $$('th button').forEach(b => { b.onclick = () => { const k = b.dataset.k; sortDir = sortKey === k ? -sortDir : 1; sortKey = k; table(); }; });
  $('#prev').onclick = () => { page = Math.max(0, page - 1); table(); };
  $('#next').onclick = () => { page++; table(); };
  redraw(); table(); if (selected !== null) detail(r, tracks, selected, u);
});

function detail(r, tracks, key, u) {
  const t = tracks.find(q => q.key === key), m = r.tracks.find(q => q.track === key);
  if (!t || !m) { $('#detail').innerHTML = '<p class="mute">No track selected.</p>'; return; }
  const p = r.params, { xs, ys } = trackUm(t, p), n = xs.length;
  // MSD as advanced_metrics computes it (pixels -> µm², frames -> s), with the log-log fit over its first 5 lags.
  const ms = frameLagMsd(t.frames, xs, ys, Math.min(n - 1, 10)), lag = ms.lags.map(l => l * p.frame_interval_s);
  const fits = [];
  if (ms.msd.length >= 3) {
    const k = Math.min(5, ms.lags.length), L = linregress(ms.lags.slice(0, k).map(Math.log), ms.msd.slice(0, k).map(Math.log));
    if (Number.isFinite(L.slope)) fits.push({ name: `log-log fit, lags 1–${ms.lags[k - 1]} frames: α = ${f(L.slope, 3)}`, color: '#ff4fd8', range: [lag[0], lag[k - 1]], fn: s => Math.exp(L.intercept) * (s / p.frame_interval_s) ** L.slope });
  }
  const ac = autocorrelation(t.frames, t.xs, t.ys, p.autocorr_max_lag);
  const kv = [['points', `${n} over ${m.duration_frames} frames (${f(u.s(m.duration_frames - 1))} s), ${m.missing_frames} missing`],
    ['Rg', `${f(u.um(m.radius_gyration))} µm`], ['scaled Rg', `${f(m.scaled_rg, 3)} (mobile above ${p.mobility_threshold})`],
    ['asymmetry', f(m.asymmetry, 3)], ['skewness / kurtosis', `${f(m.skewness, 3)} / ${f(m.kurtosis, 3)}`],
    ['eigenvalue ratio', f(m.eigenvalue_ratio, 3)], ['step alignment', f(m.step_alignment, 3)],
    ['directionality', f(m.directionality_ratio, 3)], ['fractal dimension', f(m.fracDimension, 3)],
    ['net displacement', `${f(u.um(m.netDispl))} µm`], ['straightness', f(m.Straight, 3)],
    ['mean step / speed', `${f(u.um(m.mean_step_size))} µm / ${f(u.speed(m.meanVelocity))} µm/s`],
    ['MSD α / D', `${f(m.msd_alpha, 3)} / ${f(u.D(m.msd_D))} µm²/s`],
    ['persistence', m.persistence_censored ? `&gt; ${f(u.s(m.persistence_length))} s (no 1/e drop within the measured lags)` : fu(u.s(m.persistence_length), 's')],
    ['directional bias', f(m.directional_bias, 3)]];
  $('#detail').innerHTML = `<h3>Track ${esc(key)} · ${cls(m.mobility_classification)} ${cls(m.linear_classification)}${m.label ? ` <span class="small mute">reference: ${esc(m.label)}</span>` : ''}</h3>
    <dl class="kv">${list(kv, ([a, b]) => `<dt>${a}</dt><dd>${b}</dd>`)}</dl>
    <div class="charts mini">
      <div>${trackSvg(xs, ys, { color: colorOf(m.mobility_classification) })}<p class="small mute">Trajectory in µm: start amber, end magenta.</p></div>
      <div>${msdPlot([{ name: 'MSD', color: colorOf(m.mobility_classification), x: lag, y: ms.msd, dots: true, fits }])}
        <p class="small mute">${fits.length ? `<span style="color:${fits[0].color}">- -</span> ${esc(fits[0].name)}; D = MSD(1 frame)/4 from the fit.` : 'Too few lags for the fit (needs 3).'}</p></div>
      <div>${lineChart([{ name: 'direction autocorrelation', color: '#4fd1ff', x: ac.lags.map(l => l * p.frame_interval_s), y: ac.values, dots: true }], { xlabel: 'lag (s)', ylabel: '⟨cos θ⟩', yrange: [-1, 1], hlines: [{ y: 0 }] })}
        <p class="small mute">Mean cosine between step directions that start k frames apart.</p></div>
    </div>`;
}
