// Numerical helpers for the browser engine, written to reproduce numpy / scipy / scikit-learn results:
// eigSym2 (numpy.linalg.eig on a symmetric 2x2: LAPACK dgeev's dlanv2 Schur step, so the eigenvector signs match,
// which fixes the sign of the skewness), scipy.stats.skew / kurtosis (biased, Fisher), scipy.stats.linregress
// (slope, intercept), and per-frame nearest-neighbour distances / neighbour counts (sklearn KDTree semantics:
// counts include distances <= r, the point itself excluded).

const EPS = 2 ** -52, SAFMIN = 2 ** -1022, SMLNUM = SAFMIN * (2 / EPS);
const sign = (a, b) => (b >= 0 ? Math.abs(a) : -Math.abs(a));

// Eigen-decomposition of [[a, b], [b, c]] as numpy.linalg.eig returns it: {w: [w0, w1], v: [col0, col1]}.
export function eigSym2(a, b, c) {
  if (b === 0) return { w: [a, c], v: [[1, 0], [0, 1]] };
  let negligible = Math.abs(b) <= SMLNUM;
  if (!negligible && Math.abs(b) <= EPS * (Math.abs(a) + Math.abs(c))) {   // dlahqr (Ahues & Tisseur) test
    const ab = Math.abs(b), ba = ab, aa = Math.max(Math.abs(c), Math.abs(a - c)), bb = Math.min(Math.abs(c), Math.abs(a - c));
    const s = aa + ab;
    negligible = ba * (ab / s) <= Math.max(SMLNUM, EPS * (bb * (aa / s)));
  }
  if (negligible) {
    const x1 = a !== c ? -b / (a - c) : 0, n = Math.hypot(x1, 1);
    return { w: [a, c], v: [[1, 0], [x1 / n, 1 / n]] };
  }
  // dlanv2
  let A = a, B = b, C = b, D = c, cs, sn;
  const temp = A - D, bcmax = Math.max(Math.abs(B), Math.abs(C));
  const bcmis = Math.min(Math.abs(B), Math.abs(C)) * sign(1, B) * sign(1, C);
  let p = 0.5 * temp;
  const scale = Math.max(Math.abs(p), bcmax);
  let z = p / scale * p + bcmax / scale * bcmis;
  if (z >= 4 * EPS) {
    z = p + sign(Math.sqrt(scale) * Math.sqrt(z), p);
    A = D + z; D = D - bcmax / z * bcmis;
    const tau = Math.hypot(C, z);
    cs = z / tau; sn = C / tau;
    return { w: [A, D], v: [[cs, sn], [-sn, cs]] };
  }
  const sigma = B + C, tau0 = Math.hypot(sigma, temp);
  cs = Math.sqrt(0.5 * (1 + Math.abs(sigma) / tau0));
  sn = -(p / (tau0 * cs)) * sign(1, sigma);
  const AA = A * cs + B * sn, BB = -A * sn + B * cs, CC = C * cs + D * sn, DD = -C * sn + D * cs;
  A = AA * cs + CC * sn; B = BB * cs + DD * sn; C = -AA * sn + CC * cs; D = -BB * sn + DD * cs;
  const t = 0.5 * (A + D);
  A = t; D = t;
  if (C !== 0) {
    if (B !== 0) {
      if (sign(1, B) === sign(1, C)) {
        const sab = Math.sqrt(Math.abs(B)), sac = Math.sqrt(Math.abs(C));
        p = sign(sab * sac, C);
        const tau = 1 / Math.sqrt(Math.abs(B + C));
        A = t + p; D = t - p; B = B - C; C = 0;
        const cs1 = sab * tau, sn1 = sac * tau, tt = cs * cs1 - sn * sn1;
        sn = cs * sn1 + sn * cs1; cs = tt;
      }
    } else { B = -C; C = 0; const tt = cs; cs = -sn; sn = tt; }
  }
  const x1 = A !== D ? -B / (A - D) : 0, vx = x1 * cs - sn, vy = x1 * sn + cs, n = Math.hypot(vx, vy);
  return { w: [A, D], v: [[cs, sn], [vx / n, vy / n]] };
}

export function mean(a) {
  let s = 0;
  for (const v of a) s += v;
  return s / a.length;
}

// Mean of the non-NaN values (pandas .mean()), NaN when there are none.
export function nanMean(a) {
  let s = 0, n = 0;
  for (const v of a) if (!Number.isNaN(v)) { s += v; n++; }
  return n ? s / n : NaN;
}

// Population standard deviation (numpy.std, ddof 0).
export function std(a) {
  const m = mean(a);
  let s = 0;
  for (const v of a) s += (v - m) * (v - m);
  return Math.sqrt(s / a.length);
}

// scipy.stats.skew and kurtosis (bias=True, Fisher): NaN when the variance is zero to machine resolution.
export function skewKurt(a) {
  if (!a.length) return { skew: NaN, kurt: NaN };
  const m = mean(a);
  let m2 = 0, m3 = 0, m4 = 0;
  for (const v of a) { const d = v - m, d2 = d * d; m2 += d2; m3 += d2 * d; m4 += d2 * d2; }
  m2 /= a.length; m3 /= a.length; m4 /= a.length;
  if (m2 <= (1e-15 * m) ** 2) return { skew: NaN, kurt: NaN };
  return { skew: m3 / m2 ** 1.5, kurt: m4 / (m2 * m2) - 3 };
}

// scipy.stats.linregress slope and intercept (NaN if any value is not finite).
export function linregress(x, y) {
  if (x.some(v => !Number.isFinite(v)) || y.some(v => !Number.isFinite(v))) return { slope: NaN, intercept: NaN };
  const mx = mean(x), my = mean(y);
  let sxx = 0, sxy = 0;
  for (let i = 0; i < x.length; i++) { sxx += (x[i] - mx) * (x[i] - mx); sxy += (x[i] - mx) * (y[i] - my); }
  const slope = sxy / sxx;
  return { slope, intercept: my - slope * mx };
}

// Nearest-neighbour distance (NaN when alone) and counts within each radius, for points of one frame.
// Points are swept in x order, so frames with many localisations stay fast.
export function frameNeighbours(xs, ys, radii) {
  const n = xs.length, nn = new Float64Array(n).fill(NaN), counts = radii.map(() => new Int32Array(n));
  if (n < 2) return { nn, counts };
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => xs[a] - xs[b]);
  const rmax = Math.max(...radii), r2 = radii.map(r => r * r);
  for (let oi = 0; oi < n; oi++) {
    const i = order[oi];
    let best = Infinity;
    for (const dir of [-1, 1]) {
      for (let oj = oi + dir; oj >= 0 && oj < n; oj += dir) {
        const j = order[oj], dx = xs[j] - xs[i];
        if (dx * dx > Math.max(best, rmax * rmax)) break;
        const dy = ys[j] - ys[i], d2 = dx * dx + dy * dy;
        if (d2 < best) best = d2;
        for (let k = 0; k < radii.length; k++) if (d2 <= r2[k]) counts[k][i]++;
      }
    }
    nn[i] = Math.sqrt(best);
  }
  return { nn, counts };
}
