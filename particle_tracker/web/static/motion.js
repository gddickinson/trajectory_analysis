// Motion: MSD exponent and diffusion coefficient per track (AdvancedMetricsCalculator) and the direction
// autocorrelation (DirectionAutocorrelationAnalyzer): ensemble curve, persistence and directional bias.
import { app, esc, route } from './core.js';
import { histogram, lineChart, scatter } from './charts.js';
import { num } from './engine_io.js';
import { fu, legend, provenance, units } from './fmt.js';
import { getResult, sourceLine } from './state.js';

route('motion', async () => {
  const r = await getResult(), p = r.params, t = r.tracks, u = units(p), e = r.autocorr_ensemble;
  const by = fn => t.map(q => ({ v: fn(q), c: q.mobility_classification }));
  const posD = t.filter(q => u.D(q.msd_D) > 0);
  const persist = t.filter(q => Number.isFinite(num(q.persistence_length)) && !q.persistence_censored);
  const cens = t.filter(q => q.persistence_censored).length;
  const ens = e ? lineChart([{
    name: `mean over ${e.n_tracks} tracks`, color: '#4fd1ff', dots: true,
    x: e.lags.map(l => l * p.frame_interval_s), y: e.mean.map(num),
    band: [e.mean.map((m, i) => num(m) - num(e.std[i])), e.mean.map((m, i) => num(m) + num(e.std[i]))],
  }], { xlabel: 'lag (s)', ylabel: '⟨cos θ⟩', yrange: [-1, 1], hlines: [{ y: 0 }] }) : '<p class="mute">No track has enough steps for an autocorrelation.</p>';
  app().innerHTML = `<p class="kicker">Diffusion and persistence</p><h1>Motion</h1>
  <p class="small mute">${provenance(r)} ${esc(sourceLine(r))}</p>
  ${legend(['mobile', 'immobile', 'unclassified'].filter(c => t.some(q => q.mobility_classification === c)))}
  <div class="charts">
    <div class="panel"><h3>MSD exponent <span class="gk">α</span></h3>${histogram(by(q => num(q.msd_alpha)), { xlabel: 'α (log-log slope, lags 1–5 frames)', marks: [{ x: 1, label: 'free' }] })}
      <p class="small mute">α &lt; 1 sub-, α &gt; 1 superdiffusive. From the MSD at lags of 1 to min(n − 1, 10) frames (pairs exactly that many frames
        apart), fitted on log-log axes over the first 5 lags. Needs at least 3 lags.</p></div>
    <div class="panel"><h3>Diffusion coefficient</h3>${histogram(posD.map(q => ({ v: Math.log10(u.D(q.msd_D)), c: q.mobility_classification })), { xlabel: 'log10 D (D in µm²/s)' })}
      <p class="small mute">D = exp(intercept) / 4 of the same fit, i.e. the fitted MSD at one frame over 4τ. ${t.length - posD.length} tracks without a fit are not shown.</p></div>
    <div class="panel"><h3><span class="gk">α</span> against D</h3>${scatter(posD.map(q => ({ x: u.D(q.msd_D), y: num(q.msd_alpha), c: q.mobility_classification })), { xlabel: 'D (µm²/s)', ylabel: 'α', xlog: true, lines: { y: [1] } })}</div>
    <div class="panel"><h3>Ensemble direction autocorrelation</h3>${ens}
      ${e ? `<p class="small mute">Mean over all step pairs at each lag (the per-track values weighted by their pair counts) ± the
        pair-weighted SD across tracks, for every lag reached by at least ${e.min_tracks} tracks (${e.n_tracks} tracks in all; at the
        longest lag shown, ${e.n_tracks_per_lag[e.n_tracks_per_lag.length - 1]} tracks and ${e.n_pairs_per_lag[e.n_pairs_per_lag.length - 1]} pairs).
        Ensemble persistence: ${e.persistence_censored ? `&gt; ${fu(u.s(e.persistence_length), 's')} (no 1/e drop)` : fu(u.s(e.persistence_length), 's')}.</p>
      ${lineChart([{ name: 'tracks contributing', color: '#ffb02e', dots: true, x: e.lags.map(l => l * p.frame_interval_s), y: e.n_tracks_per_lag }], { xlabel: 'lag (s)', ylabel: 'tracks at this lag' })}</p>` : ''}</div>
    <div class="panel"><h3>Persistence time</h3>${histogram(persist.map(q => ({ v: u.s(q.persistence_length), c: q.mobility_classification })), { xlabel: 'lag where the autocorrelation falls to 1/e of lag 1 (s)' })}
      <p class="small mute">${cens} tracks whose autocorrelation never falls to 1/e within their lags are left out: their persistence is
        censored (only known to exceed the last lag, reported as “&gt; last lag”). Undefined when the lag-1 value is ≤ 0.</p></div>
    <div class="panel"><h3>Directional bias</h3>${histogram(by(q => num(q.directional_bias)), { xlabel: 'mean autocorrelation over lags', range: [-1, 1], bins: 20 })}
      <p class="small mute">Positive: steps keep their direction; negative: they reverse (typical of confinement or localisation noise).</p></div>
  </div>`;
});
