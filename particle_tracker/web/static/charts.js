// Client-side SVG charts (stacked histogram, scatter, log-log MSD curves, line chart with a band, one trajectory,
// bars) and a canvas track map. Adapted from the particle_tracking web interface.
import { esc } from './core.js';
import { colorOf, f } from './fmt.js';

const W = 380, H = 250, M = { l: 50, r: 12, t: 12, b: 40 };
const fin = Number.isFinite;

function quantile(sorted, q) {
  if (!sorted.length) return NaN;
  const i = (sorted.length - 1) * q, lo = Math.floor(i);
  return sorted[lo] + (sorted[Math.min(lo + 1, sorted.length - 1)] - sorted[lo]) * (i - lo);
}

function ticks(lo, hi, n = 5) {
  const span = hi - lo || 1, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * span; v += step) out.push(+v.toPrecision(12));
  return out;
}

function scale(d0, d1, r0, r1, log = false) {
  const t = log ? Math.log10 : v => v, a = t(d0), b = t(d1);
  const s = v => r0 + (t(v) - a) / ((b - a) || 1) * (r1 - r0);
  s.domain = [d0, d1];
  if (log) {
    const e = ticks(Math.floor(a), Math.ceil(b), 6).filter(Number.isInteger).map(k => 10 ** k).filter(v => v >= d0 && v <= d1);
    s.ticks = e.length >= 2 ? e : [d0, d1];
  }
  return s;
}

function axes(xs, ys, xl, yl) {
  let s = `<line class="axis" x1="${M.l}" y1="${H - M.b}" x2="${W - M.r}" y2="${H - M.b}"/><line class="axis" x1="${M.l}" y1="${M.t}" x2="${M.l}" y2="${H - M.b}"/>`;
  for (const v of xs.ticks || ticks(...xs.domain)) s += `<text x="${xs(v)}" y="${H - M.b + 14}" text-anchor="middle">${f(v, 3)}</text>`;
  for (const v of ys.ticks || ticks(...ys.domain, 4)) s += `<text x="${M.l - 5}" y="${ys(v) + 4}" text-anchor="end">${f(v, 3)}</text>`;
  s += `<text x="${(M.l + W - M.r) / 2}" y="${H - 5}" text-anchor="middle">${esc(xl)}</text>`;
  s += `<text transform="translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">${esc(yl)}</text>`;
  return s;
}
const svg = (body, label) => `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label)}">${body}</svg>`;
const thr = (xs, x, label) => `<line class="thr" x1="${xs(x)}" x2="${xs(x)}" y1="${M.t}" y2="${H - M.b}"/><text class="thrl" x="${xs(x) + 4}" y="${M.t + 10}">${esc(label)}</text>`;

// Stacked histogram: items [{v, c}] (c = class), bins over [lo, hi] (default: 1-99 % quantiles); marks [{x, label}].
export function histogram(items, { xlabel, bins = 24, range, marks = [] } = {}) {
  const ok = items.filter(d => fin(d.v));
  if (!ok.length) return '<p class="mute small">No finite values to plot.</p>';
  const sorted = ok.map(d => d.v).sort((a, b) => a - b);
  let [lo, hi] = range || [quantile(sorted, 0.01), quantile(sorted, 0.99)];
  if (hi <= lo) { lo -= 0.5; hi += 0.5; }
  const w = (hi - lo) / bins, stack = Array.from({ length: bins }, () => ({}));
  let out = 0;
  for (const d of ok) {
    let i = Math.floor((d.v - lo) / w);
    if (d.v === hi) i = bins - 1;
    if (i < 0 || i >= bins) { out++; continue; }
    stack[i][d.c] = (stack[i][d.c] || 0) + 1;
  }
  const tot = stack.map(o => Object.values(o).reduce((a, b) => a + b, 0));
  const xs = scale(lo, hi, M.l, W - M.r), ys = scale(0, Math.max(1, ...tot), H - M.b, M.t);
  let s = axes(xs, ys, xlabel, 'tracks');
  stack.forEach((o, i) => {
    let base = 0;
    for (const [c, n] of Object.entries(o)) {
      s += `<rect x="${xs(lo + i * w) + 0.5}" y="${ys(base + n)}" width="${Math.max(0.5, xs(lo + w) - xs(lo) - 1)}" height="${ys(base) - ys(base + n)}" fill="${colorOf(c)}"><title>${esc(c)}: ${n} (${f(lo + i * w)} to ${f(lo + (i + 1) * w)})</title></rect>`;
      base += n;
    }
  });
  for (const m of marks) if (m.x >= lo && m.x <= hi) s += thr(xs, m.x, m.label);
  return svg(s, `Histogram of ${xlabel}`) + (out ? `<p class="small mute">${out} of ${ok.length} values outside the plotted range.</p>` : '');
}

// Scatter of [{x, y, c}] with optional threshold lines {x: [], y: []} and an optional curve [[x, y], ...].
export function scatter(points, { xlabel, ylabel, xlog = false, ylog = false, lines = {}, curve = null, curveLabel = '' } = {}) {
  const pts = points.filter(p => fin(p.x) && fin(p.y) && (!xlog || p.x > 0) && (!ylog || p.y > 0));
  if (!pts.length) return '<p class="mute small">No points to plot.</p>';
  const sx = pts.map(p => p.x).sort((a, b) => a - b), sy = pts.map(p => p.y).sort((a, b) => a - b);
  const pad = (a, b, log) => (log ? [a / 1.2, b * 1.2] : [Math.min(0, a), b + 0.05 * (b - a || 1)]);
  const xr = pad(sx[0], quantile(sx, 0.99), xlog), yr = pad(sy[0], quantile(sy, 0.99), ylog);
  for (const v of lines.y || []) yr[1] = Math.max(yr[1], v * 1.1);
  const xs = scale(xr[0], xr[1], M.l, W - M.r, xlog), ys = scale(yr[0], yr[1], H - M.b, M.t, ylog);
  let s = axes(xs, ys, xlabel, ylabel), hidden = 0;
  for (const v of lines.x || []) if (v >= xr[0] && v <= xr[1]) s += `<line class="thr" x1="${xs(v)}" x2="${xs(v)}" y1="${M.t}" y2="${H - M.b}"/>`;
  for (const v of lines.y || []) if (v >= yr[0] && v <= yr[1]) s += `<line class="thr" y1="${ys(v)}" y2="${ys(v)}" x1="${M.l}" x2="${W - M.r}"/>`;
  if (curve) {
    const c = curve.filter(([x, y]) => x >= xr[0] && x <= xr[1] && y >= yr[0] && y <= yr[1]);
    if (c.length > 1) s += `<path class="curve" d="${c.map(([x, y], i) => `${i ? 'L' : 'M'}${xs(x).toFixed(1)} ${ys(y).toFixed(1)}`).join('')}"><title>${esc(curveLabel)}</title></path>`;
  }
  for (const p of pts) {
    if (p.x < xr[0] || p.x > xr[1] || p.y < yr[0] || p.y > yr[1]) { hidden++; continue; }
    s += `<circle cx="${xs(p.x).toFixed(1)}" cy="${ys(p.y).toFixed(1)}" r="2.4" fill="${colorOf(p.c)}" fill-opacity=".8"/>`;
  }
  return svg(s, `${ylabel} against ${xlabel}`) + (hidden ? `<p class="small mute">${hidden} of ${pts.length} tracks outside the plotted range.</p>` : '');
}

// Log-log MSD curves: series [{name, color, x: lags (s), y: MSD (µm²), fit: t => msd | null, dots}].
export function msdPlot(series, { xlabel = 'lag time (s)', ylabel = 'MSD (µm²)' } = {}) {
  const all = series.flatMap(s => s.x.map((x, i) => [x, s.y[i]])).filter(([x, y]) => x > 0 && y > 0);
  if (!all.length) return '<p class="mute small">No positive MSD values to plot.</p>';
  const xr = [Math.min(...all.map(a => a[0])), Math.max(...all.map(a => a[0]))];
  const yr = [Math.min(...all.map(a => a[1])) / 1.3, Math.max(...all.map(a => a[1])) * 1.3];
  const xs = scale(xr[0], xr[1], M.l, W - M.r, true), ys = scale(yr[0], yr[1], H - M.b, M.t, true);
  let s = axes(xs, ys, xlabel, ylabel);
  for (const se of series) {
    const p = se.x.map((x, i) => [x, se.y[i]]).filter(([x, y]) => x > 0 && y > 0);
    s += `<path d="${p.map(([x, y], i) => `${i ? 'L' : 'M'}${xs(x).toFixed(1)} ${ys(y).toFixed(1)}`).join('')}" fill="none" stroke="${se.color}" stroke-width="1.6"><title>${esc(se.name)}</title></path>`;
    if (se.dots) for (const [x, y] of p) s += `<circle cx="${xs(x).toFixed(1)}" cy="${ys(y).toFixed(1)}" r="2" fill="${se.color}"/>`;
    for (const ft of se.fits || []) {
      const [a, b] = ft.range || xr, q = [];
      for (let i = 0; i <= 24; i++) { const t = a * (b / a) ** (i / 24), y = ft.fn(t); if (y >= yr[0] && y <= yr[1]) q.push([t, y]); }
      if (q.length > 1) s += `<path d="${q.map(([x, y], i) => `${i ? 'L' : 'M'}${xs(x).toFixed(1)} ${ys(y).toFixed(1)}`).join('')}" fill="none" stroke="${ft.color || se.color}" stroke-dasharray="5 4" stroke-width="1.4"><title>${esc(ft.name)}</title></path>`;
    }
  }
  return svg(s, 'Mean square displacement against lag time, log-log');
}

// One trajectory in µm: path, start (amber) and end (magenta), scale bar.
export function trackSvg(xs, ys, { color = '#59b8ff', size = 380 } = {}) {
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const span = Math.max(x1 - x0, y1 - y0) || 1, pad = 26, k = (size - 2 * pad) / span;
  const X = x => pad + (x - x0) * k + ((span - (x1 - x0)) * k) / 2, Y = y => pad + (y - y0) * k + ((span - (y1 - y0)) * k) / 2;
  const d = xs.map((x, i) => `${i ? 'L' : 'M'}${X(x).toFixed(1)} ${Y(ys[i]).toFixed(1)}`).join('');
  const n = xs.length - 1, bar = 10 ** Math.floor(Math.log10(span / 2.5)), bl = bar * k;
  return `<svg class="chart" viewBox="0 0 ${size} ${size}" role="img" aria-label="Trajectory in µm (image orientation, y down)">
    <path d="${d}" fill="none" stroke="${color}" stroke-width="1.4" stroke-linejoin="round"/>
    <circle cx="${X(xs[0])}" cy="${Y(ys[0])}" r="5" fill="#ffb02e"><title>start</title></circle>
    <circle cx="${X(xs[n])}" cy="${Y(ys[n])}" r="5" fill="#ff4fd8"><title>end</title></circle>
    <line x1="${pad}" y1="${size - 10}" x2="${pad + bl}" y2="${size - 10}" stroke="#e8f1ff" stroke-width="2"/>
    <text x="${pad + bl + 6}" y="${size - 6}">${f(bar)} µm</text></svg>`;
}

// All tracks on a canvas (image orientation), coloured by colorOf(id); returns the transform for hit testing.
export function drawMap(canvas, tracks, colorOf, highlight = null) {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const t of tracks) for (let i = 0; i < t.xs.length; i++) {
    x0 = Math.min(x0, t.xs[i]); x1 = Math.max(x1, t.xs[i]); y0 = Math.min(y0, t.ys[i]); y1 = Math.max(y1, t.ys[i]);
  }
  const w = canvas.width, pad = 10, k = (w - 2 * pad) / ((x1 - x0) || 1);
  canvas.height = Math.min(1400, Math.max(160, Math.round((y1 - y0) * k + 2 * pad)));
  const kk = Math.min(k, (canvas.height - 2 * pad) / ((y1 - y0) || 1));
  const g = canvas.getContext('2d');
  g.fillStyle = '#000'; g.fillRect(0, 0, w, canvas.height); g.lineJoin = 'round';
  const path = t => { g.beginPath(); t.xs.forEach((x, i) => { const X = pad + (x - x0) * kk, Y = pad + (t.ys[i] - y0) * kk; i ? g.lineTo(X, Y) : g.moveTo(X, Y); }); g.stroke(); };
  for (const t of tracks) { g.strokeStyle = colorOf(t.id); g.lineWidth = 1; g.globalAlpha = highlight ? 0.45 : 0.9; path(t); }
  g.globalAlpha = 1;
  if (highlight) { const t = tracks.find(q => q.id === highlight); if (t) { g.strokeStyle = '#fff'; g.lineWidth = 2.5; path(t); } }
  return { x0, y0, k: kk, pad };
}

// Line chart: series [{name, color, x, y, band: [lo, hi] arrays (optional), dots}], optional horizontal lines.
export function lineChart(series, { xlabel, ylabel, yrange = null, hlines = [] } = {}) {
  const pts = series.flatMap(s => s.x.map((x, i) => [x, s.y[i]])).filter(([x, y]) => fin(x) && fin(y));
  if (!pts.length) return '<p class="mute small">Nothing to plot.</p>';
  const xs0 = pts.map(p => p[0]), ysAll = pts.map(p => p[1]);
  for (const s of series) if (s.band) for (const b of s.band) for (const v of b) if (fin(v)) ysAll.push(v);
  const [ylo, yhi] = yrange || [Math.min(...ysAll), Math.max(...ysAll)];
  const xs = scale(Math.min(...xs0), Math.max(...xs0), M.l, W - M.r), ys = scale(ylo, yhi === ylo ? ylo + 1 : yhi, H - M.b, M.t);
  const cy = v => ys(Math.min(yhi, Math.max(ylo, v)));
  let s = axes(xs, ys, xlabel, ylabel);
  for (const h of hlines) if (h.y >= ylo && h.y <= yhi) s += `<line class="thr" x1="${M.l}" x2="${W - M.r}" y1="${ys(h.y)}" y2="${ys(h.y)}"/><text class="thrl" x="${W - M.r - 4}" y="${ys(h.y) - 4}" text-anchor="end">${esc(h.label || '')}</text>`;
  for (const se of series) {
    const p = se.x.map((x, i) => [x, se.y[i], i]).filter(([x, y]) => fin(x) && fin(y));
    if (se.band) {
      const up = p.map(([x, , i]) => [x, se.band[1][i]]).filter(([, v]) => fin(v)), lo = p.map(([x, , i]) => [x, se.band[0][i]]).filter(([, v]) => fin(v)).reverse();
      if (up.length > 1) s += `<path d="${up.concat(lo).map(([x, y], i) => `${i ? 'L' : 'M'}${xs(x).toFixed(1)} ${cy(y).toFixed(1)}`).join('')}Z" fill="${se.color}" fill-opacity=".14" stroke="none"/>`;
    }
    s += `<path d="${p.map(([x, y], i) => `${i ? 'L' : 'M'}${xs(x).toFixed(1)} ${cy(y).toFixed(1)}`).join('')}" fill="none" stroke="${se.color}" stroke-width="1.6"><title>${esc(se.name)}</title></path>`;
    if (se.dots) for (const [x, y] of p) s += `<circle cx="${xs(x).toFixed(1)}" cy="${cy(y).toFixed(1)}" r="2.2" fill="${se.color}"/>`;
  }
  return svg(s, `${ylabel} against ${xlabel}`);
}

// Horizontal bar chart of [{label, value, color, note}] (values >= 0).
export function bars(items, { unit = '' } = {}) {
  const max = Math.max(1e-12, ...items.map(d => d.value).filter(fin));
  return `<div class="bars">${items.map(d => `<div class="bar"><span class="bl">${esc(d.label)}</span>
    <span class="bt"><span style="width:${fin(d.value) ? (100 * d.value / max).toFixed(1) : 0}%;background:${d.color || 'var(--accent)'}"></span></span>
    <span class="bv">${f(d.value)}${unit}${d.note ? ` <span class="mute">${esc(d.note)}</span>` : ''}</span></div>`).join('')}</div>`;
}
