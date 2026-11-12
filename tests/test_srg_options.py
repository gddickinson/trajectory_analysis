"""sRg options outside the analysis classes: command line (scripts/classify_mobility.py) and the Qt widget
(gui/srg_options.py, run offscreen in a subprocess)."""
import importlib.util
import os
import pathlib
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("classify_mobility", ROOT / "scripts" / "classify_mobility.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def write_tracks(path, n_short=3, n_long=3, seed=4):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_short + n_long):
        n = 10 if t < n_short else 120
        xy = np.cumsum(rng.normal(size=(n, 2)), axis=0)
        rows += [(t, f, x, y) for f, (x, y) in enumerate(xy)]
    pd.DataFrame(rows, columns=["TRACK_ID", "FRAME", "x", "y"]).to_csv(path, index=False)


def read_out(path):
    return pd.read_csv(path, comment="#")


def test_cli_default_and_legacy(tmp_path):
    src, out = tmp_path / "t.csv", tmp_path / "o.csv"
    write_tracks(src)
    assert cli.main([str(src), "-o", str(out)]) == 0
    d = read_out(out)
    assert d.mobility_classification.tolist() == ["too_short"] * 3 + ["mobile"] * 3
    assert "min_classify_length=50" in out.read_text().splitlines()[0]
    cli.main([str(src), "-o", str(out), "--preset", "legacy"])
    assert "too_short" not in read_out(out).mobility_classification.tolist()


def test_cli_reference_threshold(tmp_path):
    rng = np.random.default_rng(2)
    rows = [(t, f, *rng.normal(size=2)) for t in range(20) for f in range(60)]
    ref = tmp_path / "fixed.csv"
    pd.DataFrame(rows, columns=["track", "frame", "x", "y"]).to_csv(ref, index=False)
    src, out = tmp_path / "t.csv", tmp_path / "o.csv"
    write_tracks(src)
    cli.main([str(src), "-o", str(out), "--reference", str(ref)])
    thr = float(out.read_text().split("mobility_threshold=")[1].split(";")[0])
    assert 1.0 < thr < 1.3   # pure localisation noise: sRg ~ 1


QT_SCRIPT = textwrap.dedent("""
    from PyQt6.QtWidgets import QApplication
    app = QApplication([])
    from particle_tracker.gui.srg_options import SrgOptionsWidget
    from particle_tracker.analysis import mobility
    w = SrgOptionsWidget()
    assert w.values() == mobility.resolve({}), w.values()
    w.preset_combo.setCurrentIndex(w.preset_combo.findData('piezo1_tirf_10fps'))
    assert w.values()['mobility_threshold'] == 2.22236433588659 and w.values()['min_classify_length'] == 200
    w.definition_combo.setCurrentIndex(w.definition_combo.findData('rg_over_step'))
    assert abs(w.values()['mobility_threshold'] - 2.22236433588659 / 1.2533141373155003) < 1e-12
    w.min_length_spin.setValue(30)
    assert w.values()['srg_preset'] == 'custom' and w.values()['min_classify_length'] == 30
    w.set_values({'srg_preset': 'legacy'})
    assert w.values()['min_classify_length'] == 0
    print('ok')
""")


def test_qt_widget_offscreen():
    pytest.importorskip("PyQt6.QtWidgets")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    r = subprocess.run([sys.executable, "-c", QT_SCRIPT], cwd=ROOT, env=env, capture_output=True, text=True,
                       timeout=120)
    if r.returncode != 0 and "ImportError" in r.stderr and "PyQt6" in r.stderr:
        pytest.skip("PyQt6 not usable here")
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-2000:]
