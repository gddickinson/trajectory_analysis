"""Mobile / immobile classification by the scaled radius of gyration (sRg): definitions, presets, minimum length.

Sources (DOIs checked on Crossref, 2026-11-12):
- Golan Y, Sherman E (2017) Resolving mixed mechanisms of protein subdiffusion at the T cell plasma membrane.
  Nat Commun 8, 15851. doi:10.1038/ncomms15851. Rg^2 = (1/N) sum (r_i - <r>)^2 (their Eq. 7); for localisations
  drawn from a 2D Gaussian, Rg / <|dr|> = sqrt(2/pi), so "the normalized ratio sqrt(pi/2) Rg / <|dr|>" is ~1 for an
  immobile particle whatever the localisation error. "For fixed cells and for immobile 40 nm gold beads fixed to
  the coverslip, we find that 95% of the normalized ratio values fall below the value 2.11 and 99% below 3.89."
  Trajectories shorter than 50 frames were discarded (85 fps).
- Ly AT, Freites JA, Bertaccini GA, Evans EL, Dickinson GD, Tobias DJ, Pathak MM (2025) Single-particle tracking
  reveals heterogeneous PIEZO1 diffusion. Biophys J 124, 4457-4470. doi:10.1016/j.bpj.2025.01.010.
  Definition (Results, p. 4461, Eq. 1): sRg = sqrt(pi/2) Rg / <r>, Rg = sqrt((1/N) sum |x_i - <x>|^2),
  <r> = (1/(N-1)) sum_{i<N} |x_{i+1} - x_i| -- the same as track_srg below for a track without gaps.
  Threshold (Materials and methods, "Fixed-cell trajectory analysis", p. 4459; Results p. 4461; Fig. 2A legend):
  "the 95th percentile of a selected set of fixed-cell trajectories sRg distribution", the fixed-cell set being
  PFA-fixed MEF trajectories kept by two mixture-model selections on a power-law fit to their TAMSD (lags 2-4):
  the lower-alpha Gaussian component, then the lower-K_alpha kernel component. The number is not printed (text,
  legends, methods); the dashed line in Fig. 2A reads 2.23 +/- 0.03 when measured against the axis ticks, which
  matches the Pathak-lab code value 2.22236433588659 (particle_trajectory_analyzer config, adapted from
  J. A. Freites' R code). The paper does not give the derivation beyond the percentile rule above.
  Length: trajectories "at least 20 s in length, which at a frame rate of 100 ms resulted in a minimum of 200
  positions per trajectory" ("PIEZO1-tdTomato trajectory generation", p. 4459). Not reproduced here: skipped frames
  were filled with NaN placeholders (same section; the paper does not say how such steps enter <r>), and live-cell
  trajectories with TAMSD below the localisation-error estimate were removed first ("Live-cell trajectory
  analysis", pp. 4459-4460). Full notes: docs/srg_calibration.md, section 5.

Calibration of every default here: scripts/srg_calibration.py -> docs/srg_calibration.md (seeded simulations).

Classes returned by classify_mobility: 'mobile', 'immobile', 'too_short' (fewer points than min_classify_length;
not classified, because sRg of a diffusing track grows ~ sqrt(N/3) and short free tracks fall below any fixed
threshold), 'unclassified' (sRg undefined: < 2 points or zero mean step).
"""
import math

import numpy as np

SQRT_PI_2 = math.sqrt(math.pi / 2)

# name -> (factor applied to Rg / mean step, description)
SRG_DEFINITIONS = {
    "golan_sherman": (SQRT_PI_2, "sqrt(pi/2) * Rg / mean step (Golan & Sherman 2017; Ly et al. 2025 Eq. 1): "
                                 "~1 for an immobile particle; the published thresholds 2.11 / 3.89 / 2.2224 use it"),
    "rg_over_step": (1.0, "Rg / mean step (the pathak_lab scripts' radius_gyration_scaled column): "
                          "~0.80 for an immobile particle; thresholds must be divided by sqrt(pi/2) = 1.2533"),
}
DEFAULT_DEFINITION = "golan_sherman"

# Presets. Thresholds are on the golan_sherman scale; resolve() converts them for the other definition.
# "basis" says where each number comes from: [source] = a cited paper, [calibration] = docs/srg_calibration.md.
PRESETS = {
    "membrane_tirf": {
        "label": "Membrane protein, TIRF (Golan & Sherman 2017)",
        "mobility_threshold": 2.11, "min_classify_length": 50,
        "basis": "threshold 2.11 [source: 95th percentile of fixed cells / immobile gold beads, Golan & Sherman 2017]; "
                 "minimum 50 points [source: same paper keeps trajectories of >= 50 frames]; [calibration] at 50 "
                 "points free diffusion is called immobile in 1.4 % of tracks without localisation noise, 2.5 % "
                 "with noise SD = 0.25 x step SD and 7.8 % at 0.5 x; slow particles (noise ~ step) need >= 150.",
    },
    "piezo1_tirf_10fps": {
        "label": "PIEZO1-tdTomato, TIRF 10 fps (Pathak lab)",
        "mobility_threshold": 2.22236433588659, "min_classify_length": 200,
        "basis": "threshold 2.2224 [Pathak-lab analysis code value. Ly et al. 2025 (Biophys J 124:4457; Methods "
                 "'Fixed-cell trajectory analysis', p. 4459; Eq. 1 and Fig. 2A, p. 4461) define the cutoff as the "
                 "95th percentile of sRg for a TAMSD-selected set of fixed-MEF trajectories; the number is not "
                 "printed; the Fig. 2A dashed line measures 2.23 +/- 0.03 against the axis ticks]; minimum 200 "
                 "points [source: Ly et al. 2025, p. 4459: trajectories of at least 20 s at 10 fps, 'a minimum of "
                 "200 positions per trajectory'].",
    },
    "fixed_reference_strict": {
        "label": "Immobile / fixed reference, strict (1 % false mobile)",
        "mobility_threshold": 3.89, "min_classify_length": 50,
        "basis": "threshold 3.89 [source: 99 % of fixed-control values fall below it, Golan & Sherman 2017], for "
                 "checking a fixed or immobile reference sample or when false 'mobile' calls are costly; minimum 50 "
                 "[source, as above]. Best practice (both papers): set the threshold from your own fixed sample "
                 "(threshold_from_reference).",
    },
    "fast_cytosolic": {
        "label": "Fast diffuser, short tracks (step >> localisation error)",
        "mobility_threshold": 2.11, "min_classify_length": 40,
        "basis": "threshold 2.11 [source: Golan & Sherman 2017]; minimum 40 points [calibration: shortest tested "
                 "length at which noise-free free diffusion is called immobile in <= 5 % of tracks]; only valid "
                 "when the localisation error is small compared with the step.",
    },
    "legacy": {
        "label": "Legacy (before 2026-11: every track classified)",
        "mobility_threshold": 2.11, "min_classify_length": 0,
        "basis": "reproduces FeatureCalculator results from before the minimum length was added (no minimum, "
                 "so short diffusing tracks are mostly called immobile).",
    },
}
CUSTOM = "custom"
DEFAULT_PRESET = "membrane_tirf"

MOBILITY_DEFAULTS = {
    "srg_preset": DEFAULT_PRESET,
    "srg_definition": DEFAULT_DEFINITION,
    "mobility_threshold": PRESETS[DEFAULT_PRESET]["mobility_threshold"],
    "min_classify_length": PRESETS[DEFAULT_PRESET]["min_classify_length"],
}
CLASSES = ("mobile", "immobile", "too_short", "unclassified")


def convert_threshold(value: float, from_definition: str, to_definition: str) -> float:
    """The same cut expressed on another definition's scale."""
    return value * SRG_DEFINITIONS[to_definition][0] / SRG_DEFINITIONS[from_definition][0]


def resolve(params: dict, threshold_aliases=()) -> dict:
    """srg_preset / srg_definition / mobility_threshold / min_classify_length from a parameter dict.

    A named preset supplies values for keys not given explicitly; explicit keys always win. A preset threshold (or
    the default) is converted to the chosen definition's scale; an explicit threshold is used as given.
    threshold_aliases: older parameter names for the threshold (e.g. 'sRg_threshold'), used when
    'mobility_threshold' is absent. Raises ValueError on an unknown preset or definition or a negative minimum.
    """
    preset = params.get("srg_preset") or DEFAULT_PRESET
    if preset != CUSTOM and preset not in PRESETS:
        raise ValueError(f"srg_preset must be one of {', '.join(list(PRESETS) + [CUSTOM])}")
    base = PRESETS.get(preset, PRESETS[DEFAULT_PRESET])
    definition = params.get("srg_definition") or DEFAULT_DEFINITION
    if definition not in SRG_DEFINITIONS:
        raise ValueError(f"srg_definition must be one of {', '.join(SRG_DEFINITIONS)}")
    threshold = params.get("mobility_threshold")
    for alias in threshold_aliases:
        if threshold is None:
            threshold = params.get(alias)
    if threshold is None:
        threshold = convert_threshold(base["mobility_threshold"], DEFAULT_DEFINITION, definition)
    min_len = params.get("min_classify_length")
    if min_len is None:
        min_len = base["min_classify_length"]
    if min_len < 0 or int(min_len) != min_len:
        raise ValueError("min_classify_length must be a whole number >= 0")
    return {"srg_preset": preset, "srg_definition": definition,
            "mobility_threshold": float(threshold), "min_classify_length": int(min_len)}


def scaled_rg(radius_gyration: float, mean_step_size: float, definition: str = DEFAULT_DEFINITION) -> float:
    """sRg under the chosen definition; NaN if either input is NaN or the mean step is not positive."""
    if np.isnan(radius_gyration) or np.isnan(mean_step_size) or mean_step_size <= 0:
        return np.nan
    return SRG_DEFINITIONS[definition][0] * radius_gyration / mean_step_size


def classify_mobility(srg: float, n_points: int, threshold: float, min_classify_length: int = 0) -> str:
    """'too_short' below the minimum length, 'unclassified' if sRg is NaN, else mobile when sRg > threshold."""
    if n_points < min_classify_length:
        return "too_short"
    if np.isnan(srg):
        return "unclassified"
    return "mobile" if srg > threshold else "immobile"


def track_srg(xy: np.ndarray, definition: str = DEFAULT_DEFINITION):
    """(Rg, mean step, sRg) of an (N, 2) track in frame order; Rg with 1/N as in Golan & Sherman Eq. 7.

    Rows are consecutive localisations: a step across missing frames counts as one step, as in FeatureCalculator
    and the example results file's meanLag.
    """
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 2:
        return np.nan, np.nan, np.nan
    rg = math.sqrt(float(np.mean(np.sum((xy - xy.mean(axis=0)) ** 2, axis=1))))
    step = float(np.mean(np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))))
    return rg, step, scaled_rg(rg, step, definition)


def threshold_from_reference(srg_values, percentile: float = 95.0) -> float:
    """Threshold from a fixed / immobile reference sample, as both papers do (95th percentile = 5 % false mobile)."""
    v = np.asarray(srg_values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("no finite sRg values in the reference sample")
    return float(np.percentile(v, percentile))


def summarise(classes) -> dict:
    """Counts per class, in CLASSES order (zeros included)."""
    classes = list(classes)
    return {c: classes.count(c) for c in CLASSES}
