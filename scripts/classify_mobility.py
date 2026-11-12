"""Command line: mobile / immobile / too-short classification of a track table by the scaled radius of gyration.

Reads a CSV with track / frame / x / y columns (names as accepted by the web app, e.g. TrackMate TRACK_ID), computes
Rg, mean step and sRg per track (analysis/mobility.py: same maths as FeatureCalculator) and writes one row per track.

Examples:
  python scripts/classify_mobility.py tracks.csv                          # default preset (membrane_tirf)
  python scripts/classify_mobility.py tracks.csv --preset piezo1_tirf_10fps -o out.csv
  python scripts/classify_mobility.py tracks.csv --min-length 30 --threshold 2.5
  python scripts/classify_mobility.py live.csv --reference fixed.csv      # threshold = 95th pct of the fixed sample
  python scripts/classify_mobility.py --list-presets
"""
import argparse
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from particle_tracker.analysis import mobility  # noqa: E402
from particle_tracker.web.tracks_io import read_tracks_csv  # noqa: E402


def per_track(df: pd.DataFrame, definition: str) -> pd.DataFrame:
    rows = []
    for t, g in df.groupby("track_number", sort=True):
        g = g.sort_values("frame")
        rg, step, srg = mobility.track_srg(g[["x", "y"]].to_numpy(), definition)
        rows.append({"track_number": t, "n_points": len(g), "radius_gyration": rg, "mean_step_size": step,
                     "scaled_rg": srg})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="sRg mobility classification of a track CSV "
                                             "(see docs/srg_calibration.md for the presets' basis).")
    ap.add_argument("csv", nargs="?", help="track table (track, frame, x, y)")
    ap.add_argument("-o", "--out", help="output CSV (default: <csv>_mobility.csv)")
    ap.add_argument("--preset", default=mobility.DEFAULT_PRESET, choices=list(mobility.PRESETS) + [mobility.CUSTOM])
    ap.add_argument("--definition", choices=list(mobility.SRG_DEFINITIONS), help="sRg definition")
    ap.add_argument("--threshold", type=float, help="mobile above this sRg (on the chosen definition's scale)")
    ap.add_argument("--min-length", type=int, help="tracks with fewer points are 'too_short'")
    ap.add_argument("--reference", help="CSV of a fixed / immobile sample: threshold = its sRg percentile")
    ap.add_argument("--percentile", type=float, default=95.0, help="percentile for --reference (default 95)")
    ap.add_argument("--list-presets", action="store_true")
    a = ap.parse_args(argv)

    if a.list_presets:
        for k, p in mobility.PRESETS.items():
            print(f"{k}: {p['label']}\n  threshold {p['mobility_threshold']}, min points {p['min_classify_length']}"
                  f"\n  {p['basis']}\n")
        return 0
    if not a.csv:
        ap.error("a track CSV is required")

    params = {"srg_preset": a.preset, "srg_definition": a.definition, "mobility_threshold": a.threshold,
              "min_classify_length": a.min_length}
    if a.reference:
        ref = per_track(read_tracks_csv(pathlib.Path(a.reference).read_bytes()), a.definition or
                        mobility.DEFAULT_DEFINITION)
        opts0 = mobility.resolve({k: v for k, v in params.items() if v is not None})
        ref = ref[ref.n_points >= opts0["min_classify_length"]]
        params["mobility_threshold"] = mobility.threshold_from_reference(ref.scaled_rg, a.percentile)
        print(f"threshold from {len(ref)} reference tracks ({a.percentile:g}th percentile): "
              f"{params['mobility_threshold']:.4f}")
    opts = mobility.resolve({k: v for k, v in params.items() if v is not None})

    tracks = per_track(read_tracks_csv(pathlib.Path(a.csv).read_bytes()), opts["srg_definition"])
    tracks["mobility_classification"] = [
        mobility.classify_mobility(s, n, opts["mobility_threshold"], opts["min_classify_length"])
        for s, n in zip(tracks.scaled_rg, tracks.n_points)]
    out = pathlib.Path(a.out or pathlib.Path(a.csv).with_name(pathlib.Path(a.csv).stem + "_mobility.csv"))
    with open(out, "w") as fh:
        fh.write(f"# srg_preset={opts['srg_preset']}; srg_definition={opts['srg_definition']}; "
                 f"mobility_threshold={opts['mobility_threshold']:.6g}; "
                 f"min_classify_length={opts['min_classify_length']}\n")
        tracks.to_csv(fh, index=False)
    counts = mobility.summarise(tracks.mobility_classification)
    print(f"{len(tracks)} tracks -> {out}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
