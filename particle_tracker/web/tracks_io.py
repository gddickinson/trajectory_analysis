"""Reading a user's track table for the web interface: column names, units and analysis parameters.

Units are explicit. x / y are in pixels (with the pixel size in µm / pixel) or in µm; the analysis code works in
pixels (its density radii are pixel radii), so µm coordinates are divided by the pixel size first. Frames are frame
numbers; the frame interval (s / frame) converts per-frame results to seconds. Coordinates are used as given (any
origin convention cancels in every distance-based quantity computed here).

The browser port (static/engine_io.js) mirrors this module.
"""
import io
import math

import numpy as np
import pandas as pd

from ..analysis.classification import TrajectoryClassifier

# Accepted column names, from TrajectoryClassifier.column_mappings (first match wins, case-insensitive).
_MAP = TrajectoryClassifier().column_mappings
COLUMNS = {k: list(_MAP[k]) for k in ("track_number", "frame", "x", "y")}
LABEL_COLUMNS = ["label", "reference_label", "class"]   # optional per-track label shown next to the results

DEFAULTS = {
    "coord_unit": "px",
    "pixel_size_um": 0.108,        # README example (AnalysisParameters pixel_size=108.0 nm per pixel)
    "frame_interval_s": 0.1,       # README example (frame_rate=10.0 Hz)
    "min_track_length": 3,         # shape metrics need >= 3 points; shorter tracks are dropped first
    "mobility_threshold": 2.11,    # FeatureCalculator default (ScaledRadiusCalculator.classify_mobility)
    "linear_eigenvalue_threshold": 20.0,   # FeatureCalculator default
    "linear_alignment_threshold": 0.7,     # FeatureCalculator default
    "autocorr_max_lag": 20,        # DirectionAutocorrelationAnalyzer default cap (steps)
    "ensemble_min_tracks": 5,      # calculate_ensemble_autocorrelation default: lags reached by >= 5 tracks
}
INTEGER_PARAMS = ("min_track_length", "autocorr_max_lag", "ensemble_min_tracks")
DENSITY_RADII = [3, 5, 10, 20, 30]   # DensityAnalyzer default radii (pixels)
MSD_MAX_LAG = 10                      # AdvancedMetricsCalculator: lags 1..min(n - 1, 10) frames, fit first 5


class AnalysisError(ValueError):
    """A problem with the user's file or parameters (shown to the user, HTTP 400)."""


def parse_params(form: dict) -> dict:
    """Form strings -> typed parameters (empty = default). Raises AnalysisError on bad values."""
    p = {}
    for k, d in DEFAULTS.items():
        v = form.get(k, "")
        if v in ("", None):
            p[k] = d
            continue
        if k == "coord_unit":
            if v not in ("px", "um"):
                raise AnalysisError("coord_unit must be 'px' or 'um'")
            p[k] = v
            continue
        try:
            p[k] = float(v)
        except (TypeError, ValueError):
            raise AnalysisError(f"{k} must be a number")
        if not math.isfinite(p[k]):
            raise AnalysisError(f"{k} must be a finite number")
        if k in INTEGER_PARAMS:
            if p[k] != int(p[k]):
                raise AnalysisError(f"{k} must be a whole number")
            p[k] = int(p[k])
    if p["pixel_size_um"] <= 0:
        raise AnalysisError("pixel_size_um must be > 0")
    if p["frame_interval_s"] <= 0:
        raise AnalysisError("frame_interval_s must be > 0")
    if p["min_track_length"] < 2:
        raise AnalysisError("min_track_length must be at least 2")
    if p["ensemble_min_tracks"] < 1:
        raise AnalysisError("ensemble_min_tracks must be at least 1")
    if not 1 <= p["autocorr_max_lag"] <= 1000:
        raise AnalysisError("autocorr_max_lag must be between 1 and 1000")
    return p


def find_column(columns, options):
    lower = {}
    for c in columns:
        lower.setdefault(str(c).strip().lower(), c)
    for o in options:
        if o.lower() in lower:
            return lower[o.lower()]
    return None


def standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Rename to track_number / frame / x / y (+ label), keep complete numeric rows, sort by track then frame."""
    found = {k: find_column(df.columns, opts) for k, opts in COLUMNS.items()}
    missing = [k for k, v in found.items() if v is None]
    if missing:
        raise AnalysisError("Missing column(s): " + ", ".join(
            f"{k} (one of {', '.join(COLUMNS[k])})" for k in missing))
    out = pd.DataFrame({k: df[v] for k, v in found.items()})
    label = find_column(df.columns, LABEL_COLUMNS)
    for k in ("track_number", "frame", "x", "y"):
        col = out[k].str.strip() if out[k].dtype == object else out[k]
        out[k] = pd.to_numeric(col, errors="coerce")
    if label is not None:
        out["label"] = df[label].where(df[label].notna(), "").astype(str)
    out = out.dropna(subset=["track_number", "frame", "x", "y"])
    if out.empty:
        raise AnalysisError("No rows with numeric track, frame, x and y values.")
    if (out["frame"] != np.round(out["frame"])).any():
        raise AnalysisError("Frame numbers must be whole numbers.")
    if out.duplicated(["track_number", "frame"]).any():
        raise AnalysisError("A track has two localisations in the same frame; one per frame is expected.")
    return out.sort_values(["track_number", "frame"], kind="stable").reset_index(drop=True)


def read_tracks_csv(data: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(data), dtype=str)   # labels stay text; numbers parsed in standardise
    except Exception as e:  # pandas raises many kinds
        raise AnalysisError(f"Could not read the CSV file: {e}")
    return standardise(df)


def to_pixels(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Coordinates in pixels (µm input divided by the pixel size)."""
    if p["coord_unit"] == "um":
        df = df.copy()
        df["x"] = df["x"] / p["pixel_size_um"]
        df["y"] = df["y"] / p["pixel_size_um"]
    return df


def coords(df: pd.DataFrame) -> dict:
    """Compact per-track coordinates in the table's own units: {track: [frames, xs, ys]}."""
    out = {}
    for t, g in df.groupby("track_number", sort=True):
        out[key(t)] = [g["frame"].astype(int).tolist(), g["x"].tolist(), g["y"].tolist()]
    return out


def key(t) -> str:
    """Track id as text: integers without '.0'."""
    f = float(t)
    return str(int(f)) if f.is_integer() else repr(f)
