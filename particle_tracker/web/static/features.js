// Features: distributions of the per-track features (FeatureCalculator), stacked by mobility class, in µm where they
// have a length.
import { app, esc, route } from './core.js';
import { histogram, scatter } from './charts.js';
import { num } from './engine_io.js';
import { MOBILITY_CLASSES, f, legend, median, provenance, units } from './fmt.js';
import { DEFINITIONS } from './mobility.js';
import { getResult, sourceLine } from './state.js';

route('features', async () => {
  const r = await getResult(), p = r.params, t = r.tracks, u = units(p);
  const by = (fn, c = 'mobility_classification') => t.map(q => ({ v: fn(q), c: q[c] }));
  const lin = t.filter(q => q.linear_classification !== 'unclassified');
  const ratioInf = t.filter(q => num(q.eigenvalue_ratio) === Infinity).length;
  const panel = (title, body, note = '') => `<div class="panel"><h3>${title}</h3>${body}${note ? `<p class="small mute">${note}</p>` : ''}</div>`;
  app().innerHTML = `<p class="kicker">Per-track features</p><h1>Features</h1>
  <p class="small mute">${provenance(r)} ${esc(sourceLine(r))}</p>
  ${legend(MOBILITY_CLASSES.filter(c => t.some(q => q.mobility_classification === c)))}
  <div class="charts">
    ${panel('Radius of gyration', histogram(by(q => u.um(q.radius_gyration)), { xlabel: 'Rg (µm)' }),
      'Square root of the trace of the gyration tensor (positions about their mean).')}
    ${panel('Scaled radius of gyration', histogram(by(q => num(q.scaled_rg)), { xlabel: 'sRg = ' + (DEFINITIONS[p.srg_definition]?.label || p.srg_definition), marks: [{ x: p.mobility_threshold, label: 'mobile above' }] }),
      `Mobile above ${f(p.mobility_threshold)}; tracks with fewer than ${p.min_classify_length} points are not classified (too short). sRg grows with track length for diffusing particles: see <a href="#/lengthcheck">Length check</a>.`)}
    ${panel('sRg against track length', scatter(t.map(q => ({ x: q.n_points, y: num(q.scaled_rg), c: q.mobility_classification })), { xlabel: 'points in track', ylabel: 'scaled Rg', lines: { y: [p.mobility_threshold], x: p.min_classify_length > 0 ? [p.min_classify_length] : [] } }))}
    ${panel('Asymmetry', histogram(by(q => num(q.asymmetry)), { xlabel: 'asymmetry −ln(1 − (λ₁−λ₂)²/2(λ₁+λ₂)²)' }))}
    ${panel('Skewness and kurtosis', histogram(by(q => num(q.skewness)), { xlabel: 'skewness of steps along the main axis' }) + histogram(by(q => num(q.kurtosis)), { xlabel: 'kurtosis (Fisher) of steps along the main axis' }))}
    ${panel('Fractal dimension', histogram(by(q => num(q.fracDimension)), { xlabel: 'fractal dimension' }),
      'log N / log(N d<sub>max</sub> / L): 1 for a straight line, about 2 for a random walk; undefined for collinear tracks.')}
    ${panel('Straightness', histogram(by(q => num(q.Straight)), { xlabel: 'mean cosine of turning angles', range: [-1, 1], bins: 20 }))}
    ${panel('Net displacement', histogram(by(q => u.um(q.netDispl)), { xlabel: 'first to last point (µm)' }))}
    ${panel('Mean speed', histogram(by(q => u.speed(q.meanVelocity)), { xlabel: 'mean step / frames spanned (µm/s)' }),
      `Mean step length: median ${f(median(t.map(q => u.um(q.mean_step_size))))} µm.`)}
    ${panel('Linearity', scatter(lin.map(q => ({ x: Math.min(num(q.eigenvalue_ratio), 1e6), y: num(q.step_alignment), c: q.linear_classification })), { xlabel: 'eigenvalue ratio λ₁/λ₂', ylabel: 'step alignment', xlog: true, lines: { x: [p.linear_eigenvalue_threshold], y: [p.linear_alignment_threshold] } }) + legend([...new Set(lin.map(q => q.linear_classification))]),
      `Linear when λ₁/λ₂ ≥ ${p.linear_eigenvalue_threshold} and alignment ≥ ${p.linear_alignment_threshold}; unidirectional when net / path length ≥ 0.7.${ratioInf ? ` ${ratioInf} perfectly collinear tracks (ratio ∞) are drawn at 10⁶.` : ''}`)}
    ${panel('Directionality', histogram(by(q => num(q.directionality_ratio)), { xlabel: 'net displacement / path length', range: [0, 1], bins: 20 }))}
  </div>`;
});

