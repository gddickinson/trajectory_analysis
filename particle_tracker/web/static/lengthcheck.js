// Length check: does the scaled-Rg mobility class depend on track length? Fraction "mobile" per length bin and
// reference label (e.g. the synthetic demo's stationary vs random-walk tracks), with a live threshold slider.
import { $, app, esc, list, route, slider } from './core.js';
import { scatter } from './charts.js';
import { num } from './engine_io.js';
import { cls, legend, pct, provenance } from './fmt.js';
import { getResult, sourceLine } from './state.js';
import { DEFINITIONS } from './mobility.js';

const BINS = [[2, 9], [10, 14], [15, 19], [20, 29], [30, 49], [50, 99], [100, Infinity]];

route('lengthcheck', async () => {
  const r = await getResult(), t = r.tracks.filter(q => Number.isFinite(num(q.scaled_rg)));
  const key = r.has_labels ? 'label' : 'mobility_classification';
  const groups = [...new Set(t.map(q => q[key]))];
  let thr = r.params.mobility_threshold;
  app().innerHTML = `<p class="kicker">Validation</p><h1>Length check</h1>
  <p class="small mute">${provenance(r)} ${esc(sourceLine(r))}</p>
  <p class="lead">The mobility class compares the scaled radius of gyration, sRg = ${esc(DEFINITIONS[r.params.srg_definition]?.label || '')},
    with a fixed threshold. For a particle that is not moving (only localisation noise) sRg stays near 1 whatever the track length; for a
    diffusing particle it keeps growing with the number of points (about √(N/3)), so a short diffusing track falls below the threshold.
    That is why tracks with fewer than ${r.params.min_classify_length} points are reported as <b>too short</b> (vertical line) rather than immobile.
    The table uses every track with an sRg, including the too-short ones, to show the effect.</p>
  ${r.has_labels ? '' : '<p class="callout small">The current data has no reference labels, so the rows below are the assigned classes. The synthetic demo (stationary and random-walk tracks from the repo\'s test generators) is the clearest test: pick it on <a href="#/analyze">Analyze</a>.</p>'}
  <div class="panel"><h3>Threshold (live)</h3><div id="sl"></div></div>
  <div id="tab"></div>
  <div class="charts"><div class="panel"><h3>sRg against track length</h3><div id="sc"></div>${legend(groups)}</div>
    <div class="panel"><h3>What this means</h3><p class="small">Simulated free diffusion (repo script <code>scripts/srg_calibration.py</code>,
      table in <code>docs/srg_calibration.md</code>) is called immobile at threshold 2.11 in 54 % of 15-point tracks, 12 % at 30 points,
      1.4 % at 50 and 0 % from 100 points when the localisation error is small; with an error as large as the step it is still 38 % at 50
      points and needs about 150. Immobile tracks (noise only) stay near sRg 1 at every length. Choose a minimum length that fits your
      particles (presets on <a href="#/analyze">Analyze</a>), and compare conditions with matched track lengths.</p>
      <p class="small mute">The lab's original scripts stored <code>radius_gyration_scaled</code> = Rg / mean step (without √(π/2));
      it is available as the second sRg definition, with the threshold converted.</p></div></div>`;
  const draw = () => {
    $('#tab').innerHTML = `<div class="scroll"><table class="tbl"><thead><tr><th>${r.has_labels ? 'reference label' : 'class'}</th>
      ${list(BINS, ([a, b]) => `<th>${b === Infinity ? `≥ ${a}` : `${a}–${b}`} points</th>`)}</tr></thead><tbody>
      ${list(groups, g => `<tr><td>${cls(g)}</td>${list(BINS, ([a, b]) => {
        const s = t.filter(q => q[key] === g && q.n_points >= a && q.n_points <= b), m = s.filter(q => num(q.scaled_rg) > thr).length;
        return `<td>${s.length ? `${pct(m, s.length)} <span class="mute small">(${m}/${s.length})</span>` : '<span class="mute">–</span>'}</td>`;
      })}</tr>`)}</tbody></table></div>
      <p class="small mute">Share of tracks called <b>mobile</b> (sRg &gt; ${thr.toFixed(2)}) in each track-length bin.</p>`;
    $('#sc').innerHTML = scatter(t.map(q => ({ x: q.n_points, y: num(q.scaled_rg), c: q[key] })), { xlabel: 'points in track', ylabel: 'scaled Rg', lines: { y: [thr], x: r.params.min_classify_length > 0 ? [r.params.min_classify_length] : [] } });
  };
  slider($('#sl'), { id: 'thr', label: 'Mobile above (sRg)', min: 1, max: 5, step: 0.01, value: thr, fmt: v => (+v).toFixed(2) }, v => { thr = v; draw(); });
  draw();
});
