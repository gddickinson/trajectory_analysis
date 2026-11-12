"""Demo track tables for the web interface; both come from the repo itself and are labelled as such.

- endothelial: real data shipped with the repo, particle_tracker/resources/example_data/
  Endothelial_NonBapta_bin10_crop_RESULTS-FILE.csv (PIEZO1-HaloTag in endothelial cells, TIRF; ThunderSTORM
  localisations linked by the original analysis scripts). Only track_number, frame, x, y are used (pixels). The
  file's SVM column (the original scripts' class: 1 mobile, 2 confined, 3 trapped, as in classification.py's
  label_map) is shown as a reference label. Frame interval 0.1 s from the example data README ("10ms per frame
  acquisition, binned by 10 frames"); the pixel size is not stated in the repo, so the README's example value of
  0.108 µm / pixel is used and can be changed.
- synthetic: the repo's test generators. The 10 tracks of tests/test_data/test_trajectories.csv
  (tests/__init__.py generate_test_data: every point is the start position plus N(0, 0.5 px) noise, i.e.
  stationary) and 150 tracks from TestDataGenerator.create_test_trajectories (random walk with drift; per-track
  step SD 0.5 to 2 px) with numpy seed 2026 and 5 to 100 points, renumbered from 100. Labels: "stationary" and
  "random walk". Pixel size and frame interval are arbitrary for simulated data (README example values).
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import analyse
from .tracks_io import DEFAULTS, coords, parse_params, standardise

ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLE = ROOT / "particle_tracker" / "resources" / "example_data" / "Endothelial_NonBapta_bin10_crop_RESULTS-FILE.csv"
TEST_TRACKS = ROOT / "tests" / "test_data" / "test_trajectories.csv"
SVM_NAMES = {1: "mobile (SVM 1)", 2: "confined (SVM 2)", 3: "trapped (SVM 3)"}
SYNTH = {"seed": 2026, "n_tracks": 150, "min_length": 5, "max_length": 100}

DEMOS = {
    "endothelial": {
        "title": "PIEZO1 in endothelial cells (real data)",
        "description": "TIRF recording of PIEZO1-HaloTag puncta; tracks from the repo's example results file. "
                       "Reference label: the original scripts' SVM class.",
        "source": "particle_tracker/resources/example_data/Endothelial_NonBapta_bin10_crop_RESULTS-FILE.csv",
        "kind": "real", "pixel_note": "pixel size not stated in the repo: README example value",
    },
    "synthetic": {
        "title": "Stationary and random-walk tracks (repo test generators)",
        "description": "10 stationary tracks (tests/test_data/test_trajectories.csv) and 150 random walks with "
                       "drift, 5 to 100 points (TestDataGenerator.create_test_trajectories, seed 2026).",
        "source": "tests/__init__.py (generate_test_data, TestDataGenerator.create_test_trajectories)",
        "kind": "simulated", "pixel_note": "simulated: units are arbitrary (README example values)",
    },
}


def _endothelial() -> pd.DataFrame:
    d = pd.read_csv(EXAMPLE)
    d["label"] = d["SVM"].map(SVM_NAMES).fillna("")
    return standardise(d[["track_number", "frame", "x", "y", "label"]])


def _synthetic() -> pd.DataFrame:
    from tests import TestDataGenerator   # repo test helpers (tests/__init__.py)
    still = pd.read_csv(TEST_TRACKS)[["track_number", "frame", "x", "y"]].assign(label="stationary")
    state = np.random.get_state()
    np.random.seed(SYNTH["seed"])
    try:
        walks = TestDataGenerator.create_test_trajectories(SYNTH["n_tracks"], SYNTH["min_length"], SYNTH["max_length"])
    finally:
        np.random.set_state(state)
    walks = walks[["track_number", "frame", "x", "y"]].assign(label="random walk")
    walks["track_number"] += 100
    both = pd.concat([still, walks], ignore_index=True)
    both[["x", "y"]] = both[["x", "y"]].round(6)
    return standardise(both)


@lru_cache(maxsize=None)
def demo_frame(demo_id: str) -> pd.DataFrame:
    return _endothelial() if demo_id == "endothelial" else _synthetic()


def demo_meta(demo_id: str) -> dict:
    df = demo_frame(demo_id)
    return {"id": demo_id, **DEMOS[demo_id], "n_tracks": int(df["track_number"].nunique()), "n_rows": int(len(df)),
            "coord_unit": "px", "pixel_size_um": DEFAULTS["pixel_size_um"],
            "frame_interval_s": DEFAULTS["frame_interval_s"]}


def demo_params(demo_id: str) -> dict:
    m = demo_meta(demo_id)
    return parse_params({"coord_unit": m["coord_unit"], "pixel_size_um": m["pixel_size_um"],
                         "frame_interval_s": m["frame_interval_s"]})


@lru_cache(maxsize=None)
def demo_result(demo_id: str) -> dict:
    r = analyse(demo_frame(demo_id), demo_params(demo_id), f"{demo_id}.csv")
    r["source"] = demo_meta(demo_id)
    return r


@lru_cache(maxsize=None)
def demo_coords(demo_id: str) -> dict:
    df = demo_frame(demo_id)
    labels = df.groupby("track_number")["label"].first()
    return {"coords": coords(df), "labels": {str(int(k)): v for k, v in labels.items()}}
