"""Run the repo's own trajectory analyses on a track table and reshape the results to JSON for the web front end.

Calls, unchanged:
- FeatureCalculator.calculate_features (features.py): nearest-neighbour distance and neighbour counts per frame,
  gyration-tensor shape metrics, linearity class, scaled radius of gyration (sRg) and mobility class, step sizes
  and velocities;
- AdvancedMetricsCalculator._calculate_advanced_diffusion_metrics (advanced_metrics.py): MSD over lags of
  1..min(n - 1, 10) frames, log-log fit of the first 5 lags -> alpha and D = exp(intercept) / 4;
- DirectionAutocorrelationAnalyzer (autocorrelation_analysis.py): per-track direction autocorrelation, persistence
  (1/e crossing) and directional bias, and the ensemble curve (pair-weighted mean over every lag reached by at least
  ensemble_min_tracks tracks).
The exponential-decay fit of the autocorrelation (scipy curve_fit) is not part of the web output.

Everything is computed in pixels and frames, as the analysis code does; the front end converts to µm and s with the
pixel size and frame interval stored in `params`. The same calculation runs in the browser (static/engine*.js),
checked against this module by static/engine.test.mjs on tests/fixtures/engine_fixture.json.
"""
import math
import os

os.environ.setdefault("TQDM_DISABLE", "1")   # the analysis classes draw tqdm progress bars

import pandas as pd  # noqa: E402

from ..analysis.advanced_metrics import AdvancedMetricsCalculator  # noqa: E402
from ..analysis.autocorrelation_analysis import DirectionAutocorrelationAnalyzer  # noqa: E402
from ..analysis.features import FeatureCalculator  # noqa: E402
from .tracks_io import DENSITY_RADII, AnalysisError, key, to_pixels  # noqa: E402

# Per-track columns taken from FeatureCalculator's output (first row of each track), renamed for the JSON.
FEATURES = {
    "radius_gyration": "radius_gyration", "asymmetry": "asymmetry", "skewness": "skewness", "kurtosis": "kurtosis",
    "eigenvalue_ratio": "eigenvalue_ratio", "step_alignment": "step_alignment",
    "directionality_ratio": "directionality_ratio", "fracDimension": "fracDimension", "netDispl": "netDispl",
    "Straight": "Straight", "mean_step_size": "mean_step_size", "scaled_rg": "scaled_rg",
    "meanVelocity": "meanVelocity",
}


def clean(v):
    """JSON-safe number: NaN -> None, ±inf -> '±inf' (the front end reads them back)."""
    if v is None:
        return None
    if isinstance(v, (str, bool)):
        return v
    v = float(v)
    if math.isnan(v):
        return None
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return v


def analyse(df: pd.DataFrame, p: dict, name: str) -> dict:
    """Analyse a standardised track table (tracks_io.standardise) with parameters p (tracks_io.parse_params)."""
    lengths = df.groupby("track_number")["frame"].transform("size")
    kept = df[lengths >= p["min_track_length"]].reset_index(drop=True)
    if kept.empty:
        raise AnalysisError(f"No track has at least {p['min_track_length']} points.")
    px = to_pixels(kept, p)[["track_number", "frame", "x", "y"]]

    feats = FeatureCalculator({
        "mobility_threshold": p["mobility_threshold"],
        "linear_eigenvalue_threshold": p["linear_eigenvalue_threshold"],
        "linear_alignment_threshold": p["linear_alignment_threshold"],
        "density_radii": DENSITY_RADII,
    }).calculate_features(px)
    per_track = feats.groupby("track_number", sort=True).first()

    msd = AdvancedMetricsCalculator()._calculate_advanced_diffusion_metrics(px)
    msd = msd.groupby("track_number", sort=True)[["alpha_exponent", "diffusion_coefficient_advanced"]].first()

    ac = DirectionAutocorrelationAnalyzer()
    labels = kept.groupby("track_number")["label"].first() if "label" in kept.columns else None
    tracks = []
    for t, g in px.groupby("track_number", sort=True):
        f = per_track.loc[t]
        frames = g["frame"].values
        r = ac.calculate_track_autocorrelation(g, p["autocorr_max_lag"])
        row = {"track": key(t), "n_points": int(len(g)), "duration_frames": int(frames.max() - frames.min() + 1),
               "missing_frames": int(frames.max() - frames.min() + 1 - len(g))}
        row.update({k: clean(f[c]) for k, c in FEATURES.items()})
        row["linear_classification"] = f["linear_classification"]
        row["mobility_classification"] = f["mobility_classification"]
        row["msd_alpha"] = clean(msd.loc[t, "alpha_exponent"])
        row["msd_D"] = clean(msd.loc[t, "diffusion_coefficient_advanced"])
        row["autocorr_steps"] = int(r["n_steps"])
        row["persistence_length"] = clean(r["persistence_length"])
        row["persistence_censored"] = bool(r["persistence_censored"])
        row["directional_bias"] = clean(r["directional_bias"])
        if labels is not None:
            row["label"] = labels.loc[t]
        tracks.append(row)

    ens = ac.calculate_ensemble_autocorrelation(px, p["autocorr_max_lag"], p["ensemble_min_tracks"])
    ensemble = None
    if ens:
        ensemble = {"n_tracks": int(ens["n_tracks"]), "lags": [int(v) for v in ens["lags"]],
                    "mean": [clean(v) for v in ens["ensemble_autocorr"]],
                    "std": [clean(v) for v in ens["ensemble_std"]],
                    "n_tracks_per_lag": [int(v) for v in ens["n_tracks_per_lag"]],
                    "n_pairs_per_lag": [int(v) for v in ens["n_pairs_per_lag"]],
                    "min_tracks": int(ens["min_tracks"]),
                    "persistence_length": clean(ens["ensemble_persistence_length"]),
                    "persistence_censored": bool(ens["ensemble_persistence_censored"])}

    return {
        "name": name, "computed_by": "python", "params": p,
        "n_tracks": len(tracks), "n_rows": int(len(kept)),
        "n_tracks_dropped": int(df["track_number"].nunique() - kept["track_number"].nunique()),
        "n_rows_dropped": int(len(df) - len(kept)),
        "has_labels": labels is not None,
        "tracks": tracks, "autocorr_ensemble": ensemble, "density": density(feats),
    }


def density(feats: pd.DataFrame) -> dict:
    """Per-localisation nearest-neighbour distance and neighbour counts (pixels), in track / frame order."""
    d = feats.sort_values(["track_number", "frame"], kind="stable")
    frames = d.groupby("frame", sort=True)
    return {
        "radii": DENSITY_RADII,
        "nn_distance": [clean(v) for v in d["nn_distance"]],
        "counts": {str(r): [int(v) for v in d[f"nnCountInFrame_within_{r}_pixels"]] for r in DENSITY_RADII},
        "frames": [int(f) for f in frames.groups.keys()],
        "per_frame_n": [int(v) for v in frames.size()],
        "per_frame_mean_nn": [clean(v) for v in frames["nn_distance"].mean()],
    }
