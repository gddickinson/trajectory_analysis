// Density: per-localisation nearest-neighbour distance and neighbour counts within 3-30 pixels (FeatureCalculator /
// DensityAnalyzer), and how they change over the recording.
import { app, esc, route } from './core.js';
import { bars, histogram, lineChart } from './charts.js';
import { num } from './engine_io.js';
import { f, median, provenance, units } from './fmt.js';
import { getResult, sourceLine } from './state.js';
import { mean, std } from './engine_stats.js';

route('density', async () => {
  const r = await getResult(), p = r.params, u = units(p), d = r.density;
  const nn = d.nn_distance.map(v => u.um(v)), alone = nn.filter(Number.isNaN).length;
  const items = d.radii.map(rad => {
    const c = d.counts[String(rad)];
    return { label: `${rad} px = ${f(rad * u.ps, 3)} µm`, value: mean(c), note: `± ${f(std(c), 3)} SD` };
  });
  const t = d.frames.map(fr => u.s(fr));
  app().innerHTML = `<p class="kicker">Crowding</p><h1>Density</h1>
  <p class="small mute">${provenance(r)} ${esc(sourceLine(r))}</p>
  <p class="lead">For every localisation: the distance to the nearest other localisation in the same frame, and how many
    others lie within 3, 5, 10, 20 and 30 pixels (the analysis code's radii are in pixels; µm with ${f(u.ps)} µm/pixel).
    Only localisations in the analysed tracks are counted.</p>
  <div class="tiles">
    <div class="tile"><b>${f(median(nn), 3)} µm</b><span>median nearest-neighbour distance</span></div>
    <div class="tile"><b>${d.frames.length}</b><span>frames with localisations</span></div>
    <div class="tile"><b>${f(mean(d.per_frame_n), 3)}</b><span>localisations per frame (mean)</span></div>
    <div class="tile"><b>${alone}</b><span>localisations alone in their frame</span></div>
  </div>
  <div class="charts">
    <div class="panel"><h3>Nearest-neighbour distance</h3>${histogram(nn.filter(Number.isFinite).map(v => ({ v, c: 'all' })), { xlabel: 'distance to nearest localisation in the frame (µm)' })}</div>
    <div class="panel"><h3>Neighbours within a radius</h3>${bars(items)}
      <p class="small mute">Mean (± SD) number of other localisations within each radius, over all localisations.</p></div>
    <div class="panel"><h3>Over time</h3>${lineChart([{ name: 'localisations per frame', color: '#ffb02e', x: t, y: d.per_frame_n }], { xlabel: 'time (s)', ylabel: 'localisations per frame' })}
      ${lineChart([{ name: 'mean nearest-neighbour distance', color: '#4fd1ff', x: t, y: d.per_frame_mean_nn.map(v => u.um(num(v))) }], { xlabel: 'time (s)', ylabel: 'mean NN distance (µm)' })}</div>
  </div>
  ${r.source?.id === 'endothelial' ? `<p class="callout small">The example results file also has an <code>nnDist_inFrame</code> column written by the
    original scripts. In 42 % of its rows it does not match the distance recomputed from the file's own x / y, and in 207 rows it
    contradicts the file's own neighbour counts (e.g. row 1: 24 px to the nearest neighbour, yet one neighbour within 5 px); how it was
    computed is not recorded in the repo. The distances shown here are recomputed; the neighbour counts agree with the file's <code>nnCountInFrame_within_*</code> columns exactly.</p>` : ''}`;
});
