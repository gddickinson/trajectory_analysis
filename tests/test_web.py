"""Web interface tests: every exported endpoint answers, the static build is complete, and the JS engine tests pass.

Run: python -m pytest -q tests/test_web.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from particle_tracker.web.app import STATIC, STATIC_ENDPOINTS, app  # noqa: E402

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "static/main.js" in r.text


@pytest.mark.parametrize("path", STATIC_ENDPOINTS)
def test_endpoint(path):
    r = client.get("/api" + path)
    assert r.status_code == 200, r.text
    assert r.json() is not None


def test_static_build(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    assert (tmp_path / "site" / "index.html").read_text().count("APP_STATIC") == 1
    for path in STATIC_ENDPOINTS:
        assert (tmp_path / "site" / "api" / (path.strip("/") + ".json")).exists()


def test_static_assets_referenced_exist():
    html = (STATIC / "index.html").read_text()
    for name in ("style.css", "main.js"):
        assert f"static/{name}" in html and (STATIC / name).exists()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_engine():
    tests = sorted(STATIC.glob("*.test.mjs"))
    for t in tests:
        r = subprocess.run(["node", str(t)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


# --- project-specific API tests -------------------------------------------------------------------------------------
import json  # noqa: E402

import numpy as np  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "engine_fixture.json"


def _close(a, b, where):
    if isinstance(b, dict):
        for k in b:
            _close(a[k], b[k], f"{where}.{k}")
    elif isinstance(b, list):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{where}[{i}]")
    elif isinstance(b, float) and isinstance(a, (int, float)):
        assert np.isclose(a, b, rtol=1e-9, atol=1e-12), (where, a, b)
    else:
        assert a == b, (where, a, b)


def test_fixture_is_current():
    """The Python-vs-JS fixture still matches what the Python code computes (numbers with a tolerance: CI is Linux)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_engine_fixture
    fresh = json.loads(json.dumps(make_engine_fixture.build()))
    _close(fresh, json.loads(FIXTURE.read_text()), "fixture")


def test_info_lists_demos_and_units():
    i = client.get("/api/info").json()
    assert {d["id"] for d in i["demos"]} == {"endothelial", "synthetic"}
    assert i["defaults"]["pixel_size_um"] > 0 and i["defaults"]["frame_interval_s"] > 0
    assert i["density_radii"] == [3, 5, 10, 20, 30]


def test_demo_result_shape():
    r = client.get("/api/demo/endothelial/result").json()
    assert r["computed_by"] == "python" and r["n_tracks"] == 119 and r["n_rows"] == 4137
    assert r["has_labels"] and {t["label"] for t in r["tracks"]} == {"mobile (SVM 1)", "confined (SVM 2)", "trapped (SVM 3)"}
    assert len(r["density"]["nn_distance"]) == r["n_rows"]
    t = r["tracks"][0]
    for k in ("radius_gyration", "scaled_rg", "mobility_classification", "msd_alpha", "persistence_length"):
        assert k in t


def test_upload_um_equals_pixels():
    """The same tracks in µm (with the pixel size) give the same results as in pixels."""
    rows = ["track_number,frame,x,y"] + [f"{t},{f},{10 + t + 0.3 * f + 0.1 * (f % 3)},{5 + 0.2 * f * t}"
                                         for t in range(3) for f in range(12)]
    px = client.post("/api/analyze", files={"file": ("a.csv", "\n".join(rows))}, data={"pixel_size_um": "0.2"}).json()
    um_rows = [rows[0]] + [",".join([*r.split(",")[:2], str(float(r.split(",")[2]) * 0.2), str(float(r.split(",")[3]) * 0.2)])
                           for r in rows[1:]]
    um = client.post("/api/analyze", files={"file": ("a.csv", "\n".join(um_rows))},
                     data={"coord_unit": "um", "pixel_size_um": "0.2"}).json()
    _close(um["tracks"], px["tracks"], "tracks")


def test_upload_errors():
    r = client.post("/api/analyze", files={"file": ("a.csv", "a,b\n1,2\n")})
    assert r.status_code == 400 and "Missing column" in r.text
    r = client.post("/api/analyze", files={"file": ("a.csv", "track,frame,x,y\n1,1,0,0\n1,1,1,1\n")})
    assert r.status_code == 400 and "same frame" in r.text
    r = client.post("/api/analyze", data={"demo": "endothelial", "pixel_size_um": "-1"})
    assert r.status_code == 400
    r = client.post("/api/analyze", data={"demo": "endothelial", "min_track_length": "500"})
    assert r.status_code == 400 and "No track" in r.text


def test_min_track_length_drops_short_tracks():
    r = client.post("/api/analyze", data={"demo": "endothelial", "min_track_length": "20"}).json()
    assert r["n_tracks_dropped"] > 0 and all(t["n_points"] >= 20 for t in r["tracks"])
    assert r["n_tracks"] + r["n_tracks_dropped"] == 119


def _classes(r):
    return [t["mobility_classification"] for t in r["tracks"]]


def test_srg_options():
    """Presets, the second definition, explicit values and the minimum length reach the analysis."""
    info = client.get("/api/info").json()
    assert info["defaults"]["min_classify_length"] == 50 and info["defaults"]["mobility_threshold"] == 2.11
    assert "piezo1_tirf_10fps" in info["mobility"]["presets"]
    d = client.post("/api/analyze", data={"demo": "endothelial"}).json()
    assert all(c == "too_short" for t, c in zip(d["tracks"], _classes(d)) if t["n_points"] < 50)
    assert "too_short" not in _classes(client.post("/api/analyze", data={"demo": "endothelial",
                                                                         "srg_preset": "legacy"}).json())
    p = client.post("/api/analyze", data={"demo": "endothelial", "srg_preset": "piezo1_tirf_10fps"}).json()["params"]
    assert (p["mobility_threshold"], p["min_classify_length"]) == (2.22236433588659, 200)
    raw = client.post("/api/analyze", data={"demo": "endothelial", "srg_definition": "rg_over_step"}).json()
    assert _classes(raw) == _classes(d)   # threshold converted with the definition
    assert abs(raw["params"]["mobility_threshold"] - 2.11 / 1.2533141373155003) < 1e-12
    c = client.post("/api/analyze", data={"demo": "endothelial", "srg_preset": "custom", "mobility_threshold": "3",
                                           "min_classify_length": "10"}).json()
    assert (c["params"]["mobility_threshold"], c["params"]["min_classify_length"]) == (3.0, 10)
    for bad in ({"srg_preset": "nope"}, {"srg_definition": "nope"}, {"min_classify_length": "-1"},
                {"mobility_threshold": "0"}):
        assert client.post("/api/analyze", data={"demo": "endothelial", **bad}).status_code == 400
