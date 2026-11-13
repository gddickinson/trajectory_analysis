"""Calibrate the sRg mobile / immobile classification by simulation; writes docs/srg_calibration.md.

Seeded 2D simulations (numpy default_rng, seed below), lengths N = points per track:
- immobile: a fixed position plus Gaussian localisation noise (SD sigma per axis), optionally with a slow linear
  drift (stage drift that a fixed sample might show);
- free diffusion: Gaussian steps (SD s per axis) plus localisation noise of SD sigma = ratio * s;
- confined diffusion: the same steps reflected inside a disc of radius R (start uniform in the disc), noise 0.25 s.
sRg is computed exactly as mobility.track_srg / FeatureCalculator (Rg with 1/N, mean of consecutive steps), for both
definitions: golan_sherman = sqrt(pi/2) Rg / mean step, rg_over_step = Rg / mean step.

Usage: python scripts/srg_calibration.py [--trials 4000] [--out docs/srg_calibration.md]
"""
import argparse
import datetime
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from particle_tracker.analysis.mobility import PRESETS, SQRT_PI_2, track_srg  # noqa: E402

SEED = 20261112
LENGTHS = [5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 300]
NOISE_RATIOS = [0.0, 0.25, 0.5, 1.0, 2.0]
T_GS, T_PIEZO, T_STRICT = 2.11, 2.22236433588659, 3.89

PIEZO1_SOURCE = """
## 5. Source of the PIEZO1 threshold (`piezo1_tirf_10fps`, 2.2224)

Ly AT, Freites JA, Bertaccini GA, Evans EL, Dickinson GD, Tobias DJ, Pathak MM (2025). Single-particle tracking reveals
heterogeneous PIEZO1 diffusion. *Biophys J* 124:4457-4470. [doi:10.1016/j.bpj.2025.01.010](https://doi.org/10.1016/j.bpj.2025.01.010).
Page numbers are the journal's; the supporting material is not part of the article PDF and was not checked.

- **Definition** (Results, "PIEZO1-tdTomato puncta exhibit heterogeneous mobility", p. 4461, Eq. 1):
  sRg = √(π/2)·Rg/⟨r⟩ with Rg = √((1/N)Σ|x⃗ᵢ − ⟨x⃗⟩|²) and ⟨r⟩ = (1/(N−1))Σ|x⃗ᵢ₊₁ − x⃗ᵢ|, "as proposed by Golan and
  Sherman". (As printed, the centroid ⟨x⃗⟩ = Σx⃗ᵢ lacks its 1/N.) This is `golan_sherman` / `mobility.track_srg` for a
  track without gaps. "For immobile trajectories with apparent motion resulting exclusively from localization error, the
  sRg distribution should approximate a Gaussian distribution with mean of 1 independent of the magnitude of the
  localization error."
- **Threshold** (Results, p. 4461): "Similar to Golan and Sherman (46), we used the 95th percentile of a selected set of
  fixed-cell trajectories sRg distribution as a threshold to categorize trajectories as immobile or mobile in live
  cells." Fig. 2A legend: "we take the 95th percentile of the MEFs fixed cells' sRg distribution (vertical dashed
  line) as cutoff"; with it "~40% of the observed trajectories are identified as immobile in both cell types"
  (MEFs and mLSECs; the text says 40 %).
- **How the fixed-cell set was selected** (Materials and methods, "Fixed-cell trajectory analysis", p. 4459): PIEZO1-tdTomato
  MEFs fixed with 4 % paraformaldehyde. Each trajectory's TAMSD was fitted as a power law K_α(nΔt)^α over lags 2-4
  (expected α ≈ 0). A two-component Gaussian mixture on α kept the component with the lower mean α, then a
  two-component kernel mixture on K_α kept the component with the lower mean K_α (R package mixtools). "We also
  computed the corresponding scaled radius of gyration, sRg (see Eq. 1), for this selected data set of fixed-cell
  trajectories and used the 95th percentile of the corresponding distribution as a threshold to identify trajectories
  as immobile or mobile in live cells."
- **The number**: not printed in the text, figure legends or methods. Measured against the axis ticks of Fig. 2A
  (rendered at 600 dpi), the dashed line sits at sRg = 2.23 ± 0.03, consistent with 2.22236433588659, the value in the
  Pathak-lab analysis code (particle_trajectory_analyzer config, adapted from J. A. Freites' R code). The paper gives no
  other derivation of the value, so 2.2224 is kept; its definition matches the preset's.
- **Track length** ("PIEZO1-tdTomato trajectory generation", p. 4459): "We limited our analysis to trajectories that were
  at least 20 s in length, which at a frame rate of 100 ms resulted in a minimum of 200 positions per trajectory" (the
  preset's minimum of 200 points).
- **Not reproduced by the preset** (differences, all from the same pages):
  - Skipped frames "were handled by inserting a placeholder value (numpy.nan) for missing coordinates" (p. 4459). The
    paper does not say how steps next to a placeholder enter ⟨r⟩. Here, a step across missing frames counts as one
    step, so ⟨r⟩ (and so sRg) can differ for tracks with gaps. Whether the 200-position minimum counts placeholders is
    also not stated; here it counts localisations.
  - Live-cell trajectories with TAMSD below the fixed-cell localisation-error estimate were removed before the analyses
    ("Live-cell trajectory analysis", pp. 4459-4460).
  - The threshold came from a TAMSD-selected fixed-cell set; `threshold_from_reference` / `--reference` take the 95th
    percentile of whatever reference sRg values they are given, without that selection.
"""


def srg_batch(xy):
    """golan_sherman sRg of every track in an (M, N, 2) array (same maths as mobility.track_srg)."""
    c = xy - xy.mean(axis=1, keepdims=True)
    rg = np.sqrt(np.mean(np.sum(c ** 2, axis=2), axis=1))
    step = np.mean(np.sqrt(np.sum(np.diff(xy, axis=1) ** 2, axis=2)), axis=1)
    return SQRT_PI_2 * rg / step


def immobile(rng, m, n, drift=0.0):
    xy = rng.normal(size=(m, n, 2))
    if drift:
        ang = rng.uniform(0, 2 * math.pi, size=m)
        d = np.stack([np.cos(ang), np.sin(ang)], axis=1)[:, None, :] * drift
        xy = xy + d * np.arange(n)[None, :, None]
    return xy


def free(rng, m, n, ratio):
    return np.cumsum(rng.normal(size=(m, n, 2)), axis=1) + ratio * rng.normal(size=(m, n, 2))


def confined(rng, m, n, radius, ratio=0.25):
    r = radius * np.sqrt(rng.uniform(size=m))
    a = rng.uniform(0, 2 * math.pi, size=m)
    p = np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
    out = np.empty((m, n, 2))
    out[:, 0] = p
    for i in range(1, n):
        p = p + rng.normal(size=(m, 2))
        d = np.hypot(p[:, 0], p[:, 1])
        out_of = d > radius
        new = np.minimum(np.abs(2 * radius - d[out_of]), radius)
        p[out_of] *= (new / d[out_of])[:, None]
        out[:, i] = p
    return out + ratio * rng.normal(size=(m, n, 2))


def pct(x):
    return f"{100 * x:.1f}"


def run(trials):
    rng = np.random.default_rng(SEED)
    # self-check: the vectorised sRg equals mobility.track_srg
    probe = free(rng, 3, 12, 0.5)
    assert np.allclose(srg_batch(probe), [track_srg(t)[2] for t in probe], rtol=1e-12)

    L = []
    L.append("# sRg calibration (simulation)\n")
    L.append(f"Generated by `scripts/srg_calibration.py` (seed {SEED}, {trials} tracks per cell, "
             f"{datetime.date.today().isoformat()}). sRg = √(π/2)·Rg/⟨step⟩ (`golan_sherman`) unless stated; "
             "`rg_over_step` = Rg/⟨step⟩ = sRg/1.2533. Steps have SD s per axis; localisation noise SD σ per axis. "
             "Percentages are of simulated tracks.\n")

    L.append("## 1. Immobile particles (localisation noise only)\n")
    L.append("sRg of pure noise does not depend on σ. Golan & Sherman's 2.11 and 3.89 are the 95th / 99th percentiles "
             "of *real* fixed controls; ideal noise is much tighter, so real fixed samples contain extra apparent "
             "motion (drift, mislinking, blinking). Last columns: immobile tracks called mobile at 2.11 with a linear "
             "drift of 0.05σ or 0.1σ per frame.\n")
    L.append("| N | median | 95th pct | 99th pct | mobile at 2.11 | mobile at 2.2224 | mobile at 3.89 | "
             "mobile at 2.11, drift 0.05σ | drift 0.1σ |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for n in LENGTHS:
        s = srg_batch(immobile(rng, trials, n))
        d1 = srg_batch(immobile(rng, trials, n, 0.05))
        d2 = srg_batch(immobile(rng, trials, n, 0.1))
        q = np.percentile(s, [50, 95, 99])
        L.append(f"| {n} | {q[0]:.3f} | {q[1]:.3f} | {q[2]:.3f} | {pct((s > T_GS).mean())} | "
                 f"{pct((s > T_PIEZO).mean())} | {pct((s > T_STRICT).mean())} | {pct((d1 > T_GS).mean())} | "
                 f"{pct((d2 > T_GS).mean())} |")

    L.append("\n## 2. Free diffusion called immobile (false-immobile %)\n")
    L.append("Columns: noise ratio σ/s. `GS 2.11` = golan_sherman sRg with threshold 2.11 (consistent pairing); "
             "`raw 2.11` = Rg/⟨step⟩ compared with 2.11 (the mixed pairing, equivalent to a golan_sherman threshold "
             "of 2.64); `GS 2.2224` = PIEZO1 lab value.\n")
    head = " | ".join(f"σ/s={r} GS 2.11 | raw 2.11 | GS 2.2224" for r in NOISE_RATIOS)
    L.append(f"| N | {head} |")
    L.append("|---|" + "---|" * (3 * len(NOISE_RATIOS)))
    min_len = {}
    for n in LENGTHS:
        cells = []
        for r in NOISE_RATIOS:
            s = srg_batch(free(rng, trials, n, r))
            fi = (s <= T_GS).mean()
            cells += [pct(fi), pct((s / SQRT_PI_2 <= T_GS).mean()), pct((s <= T_PIEZO).mean())]
            if fi <= 0.05 and r not in min_len:
                min_len[r] = n
        L.append(f"| {n} | " + " | ".join(cells) + " |")

    L.append("\n### Shortest tested length with ≤ 5 % false-immobile (golan_sherman, 2.11)\n")
    L.append("| σ/s | shortest N |")
    L.append("|---|---|")
    for r in NOISE_RATIOS:
        L.append(f"| {r} | {min_len.get(r, f'> {LENGTHS[-1]}')} |")

    L.append("\n## 3. Confined diffusion (reflecting disc of radius R, σ = 0.25 s): % called mobile at 2.11\n")
    L.append("sRg measures spread relative to the step, so a particle confined to a region much larger than one step "
             "is 'mobile'; only confinement on the scale of a step (or the noise) reads as immobile.\n")
    Ns = [20, 50, 200]
    L.append("| R / s | " + " | ".join(f"N={n}" for n in Ns) + " |")
    L.append("|---|" + "---|" * len(Ns))
    for radius in [0.5, 1, 2, 5, 10]:
        cells = [pct((srg_batch(confined(rng, trials, n, radius)) > T_GS).mean()) for n in Ns]
        L.append(f"| {radius} | " + " | ".join(cells) + " |")

    L.append("\n## 4. Presets (particle_tracker/analysis/mobility.py)\n")
    L.append("| preset | threshold | min points | basis |")
    L.append("|---|---|---|---|")
    for k, v in PRESETS.items():
        L.append(f"| `{k}` ({v['label']}) | {v['mobility_threshold']:.4g} | {v['min_classify_length']} | "
                 f"{v['basis']} |")
    L.append(PIEZO1_SOURCE)
    return "\n".join(L), min_len


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=4000)
    ap.add_argument("--out", default=str(ROOT / "docs" / "srg_calibration.md"))
    a = ap.parse_args()
    text, min_len = run(a.trials)
    pathlib.Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
