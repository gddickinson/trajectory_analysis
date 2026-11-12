// Home: what the tool does, live counts from the demo analysis, links to the pages.
import { app, isStatic, route } from './core.js';
import { count, pct } from './fmt.js';
import { demoResult, getInfo } from './state.js';

route('home', async () => {
  const info = await getInfo();
  const first = await demoResult(info.demos[0].id), mob = count(first.tracks, 'mobility_classification');
  const nTracks = info.demos.reduce((s, d) => s + d.n_tracks, 0);
  app().innerHTML = `<section class="hero">
    <p class="kicker">Single-particle tracking · trajectory features</p>
    <h1 class="glow">Trajectory Analysis</h1>
    <p class="lead">A comprehensive, modular application for analysing particle trajectories from TIRF and other single-molecule
      recordings. Here: the repo's own feature, diffusion, autocorrelation and density analyses, run on your track CSV
      with explicit units (µm, s).</p>
    <a class="btn" href="#/analyze">Analyze tracks</a><a class="btn alt" href="#/tracks">Explore the example data</a>
  </section>
  <div class="tiles">
    <div class="tile"><b>${first.n_tracks}</b><span>PIEZO1 tracks in the repo's real example recording (${first.n_rows.toLocaleString()} localisations)</span></div>
    <div class="tile"><b>${pct(mob.get('mobile') || 0, first.n_tracks)}</b><span>of them mobile by scaled Rg (default threshold)</span></div>
    <div class="tile"><b>${info.demos.length}</b><span>demo datasets, ${nTracks} tracks, all from the repo</span></div>
    <div class="tile"><b>µm · s</b><span>explicit units: pixel size in µm/pixel, frame interval in s</span></div>
  </div>
  <div class="grid">
    <a class="card" href="#/analyze"><b>Analyze</b><small>upload or demo</small><span>Load a track table (track, frame, x, y), set units and parameters, get per-class summaries and CSV exports.</span></a>
    <a class="card" href="#/tracks"><b>Tracks</b><small>map and per-track detail</small><span>Every track coloured by mobility, linearity or reference label; one track's features, MSD and autocorrelation.</span></a>
    <a class="card" href="#/features"><b>Features</b><small>shape metrics</small><span>Radius of gyration, scaled Rg, asymmetry, skewness, kurtosis, fractal dimension, straightness, linearity.</span></a>
    <a class="card" href="#/motion"><b>Motion</b><small>MSD and persistence</small><span>MSD exponent and diffusion coefficient per track; direction autocorrelation and persistence time.</span></a>
    <a class="card" href="#/density"><b>Density</b><small>crowding</small><span>Nearest-neighbour distances and neighbour counts within 3–30 pixels, over the recording.</span></a>
    <a class="card" href="#/lengthcheck"><b>Length check</b><small>validation</small><span>How the scaled-Rg mobility class depends on track length, on labelled tracks from the repo's generators.</span></a>
    <a class="card" href="#/method"><b>Method</b><small>formulas</small><span>Exactly what is computed, and from which Python class.</span></a>
    <a class="card" href="#/about"><b>About</b><small>desktop app, credits</small><span>The PyQt6 desktop app (detection, linking, SVM classification) and how to run everything.</span></a>
  </div>
  <p class="small mute">${isStatic() ? 'On this static site the analysis runs in your browser; uploaded files never leave your computer.' : 'Running locally: the analysis runs in Python.'}
    Detection, linking, background subtraction and SVM classification need images or training data and are in the desktop app only.</p>`;
});
