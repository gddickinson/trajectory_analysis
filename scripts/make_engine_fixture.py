"""Write tests/fixtures/engine_fixture.json: track tables with the Python analysis results (web/analysis.py), used by
particle_tracker/web/static/engine.test.mjs to check the browser engine, and by tests/test_web.py to check that the
fixture is still current.

Cases: the first 40 tracks of the example data (default parameters); 40 tracks of the synthetic demo in µm with
other parameters; hand-made edge cases (gaps, zero-length and collinear steps, a stationary point, 2-point tracks,
shuffled rows, a lone localisation in a frame, labels).

    python scripts/make_engine_fixture.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from particle_tracker.web.analysis import analyse  # noqa: E402
from particle_tracker.web.demos import demo_frame  # noqa: E402
from particle_tracker.web.tracks_io import DEFAULTS, standardise  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "engine_fixture.json"


def edge_tracks() -> pd.DataFrame:
    rows = []

    def add(t, frames, xs, ys, label):
        rows.extend({"track_number": t, "frame": f, "x": x, "y": y, "label": label} for f, x, y in zip(frames, xs, ys))

    f = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]
    add(1, f, [0.5 * v for v in f], [0.0] * len(f), "gapped line")             # collinear, gaps
    add(2, [0, 1, 2, 3, 4], [0, 1, 1, 2, 2], [0, 0, 0, 0, 1], "zero step")     # zero-length step
    add(3, range(6), [3.0] * 6, [4.0] * 6, "stationary")                        # every point identical
    add(4, [0, 1], [10.0, 11.0], [10.0, 10.5], "two points")
    rng = np.random.RandomState(11)
    xs = np.cumsum(rng.normal(0, 1.2, 40)) + 20
    ys = np.cumsum(rng.normal(0, 1.2, 40)) + 20
    add(5, range(40), xs, ys, "random walk")
    add(6, [50], [80.0], [80.0], "single")                                      # alone in its frame
    t = np.arange(30)
    add(7, t, 30 + 0.8 * t + rng.normal(0, 0.05, 30), 5 + 0.1 * t + rng.normal(0, 0.05, 30), "directed")
    add(8, [3, 5, 6, 9, 10, 12, 13, 14], rng.normal(25, 0.3, 8), rng.normal(22, 0.3, 8), "gapped jitter")
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=4).reset_index(drop=True)   # rows shuffled on purpose


def table(df: pd.DataFrame) -> dict:
    cols = ["track_number", "frame", "x", "y"] + (["label"] if "label" in df.columns else [])
    return {"header": cols, "rows": df[cols].astype(object).values.tolist()}


def cases():
    endo = demo_frame("endothelial")
    endo = endo[endo["track_number"].isin(sorted(endo["track_number"].unique())[:40])]
    synth = demo_frame("synthetic").copy()
    synth = synth[synth["track_number"].isin(sorted(synth["track_number"].unique())[::4])]
    synth[["x", "y"]] = synth[["x", "y"]] * 0.108
    other = {**DEFAULTS, "coord_unit": "um", "pixel_size_um": 0.108, "frame_interval_s": 0.05, "min_track_length": 5,
             "mobility_threshold": 2.0, "linear_eigenvalue_threshold": 10.0, "linear_alignment_threshold": 0.6,
             "autocorr_max_lag": 8}
    edges = edge_tracks()
    yield "endothelial40", endo, dict(DEFAULTS)
    yield "synthetic_um", synth, other
    yield "edges", edges, {**DEFAULTS, "min_track_length": 2, "autocorr_max_lag": 6}
    yield "edges_strict", edges, {**DEFAULTS, "min_track_length": 6, "linear_eigenvalue_threshold": 5.0}


def build() -> dict:
    out = []
    for name, df, p in cases():
        tab = table(df)
        res = analyse(standardise(pd.DataFrame(tab["rows"], columns=tab["header"])), p, name + ".csv")
        out.append({"name": name, "table": tab, "params": p, "result": res})
    return {"cases": out}


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
