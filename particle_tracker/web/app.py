"""FastAPI web app for Trajectory Analysis: serves the front end in static/ and the JSON API under /api.

Run with `python run_web.py` (http://127.0.0.1:8212, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site, so keep that list in step with the GET routes below.

The analyses are the repo's own (see web/analysis.py); demo tracks come from the repo's example data and test
generators (web/demos.py). On the static site the same analyses run in the browser (static/engine*.js).
"""
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .analysis import analyse
from .demos import DEMOS, demo_coords, demo_frame, demo_meta, demo_result
from .tracks_io import (COLUMNS, DEFAULTS, DENSITY_RADII, LABEL_COLUMNS, MSD_MAX_LAG, AnalysisError, coords,
                        parse_params, read_tracks_csv)

STATIC = Path(__file__).resolve().parent / "static"
TITLE = "Trajectory Analysis"
MAX_UPLOAD = 100 * 1024 * 1024

app = FastAPI(title=TITLE)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    """Project summary: analysis defaults, accepted columns, demo datasets."""
    return {"title": TITLE, "version": "1", "defaults": DEFAULTS, "columns": COLUMNS, "label_columns": LABEL_COLUMNS,
            "density_radii": DENSITY_RADII, "msd_max_lag": MSD_MAX_LAG, "demos": [demo_meta(d) for d in DEMOS]}


@app.get("/api/demo/{demo_id}/result")
def demo(demo_id: str):
    """A demo dataset analysed in Python with its own units and the default parameters."""
    if demo_id not in DEMOS:
        raise HTTPException(404, "unknown demo")
    return demo_result(demo_id)


@app.get("/api/demo/{demo_id}/tracks")
def demo_tracks(demo_id: str):
    """Track coordinates of a demo dataset ({track: [frames, xs, ys]}, pixels) and its reference labels."""
    if demo_id not in DEMOS:
        raise HTTPException(404, "unknown demo")
    return demo_coords(demo_id)


@app.post("/api/analyze")
async def analyze(file: UploadFile | None = File(None), demo: str = Form(""), coord_unit: str = Form(""),
                  pixel_size_um: str = Form(""), frame_interval_s: str = Form(""), min_track_length: str = Form(""),
                  mobility_threshold: str = Form(""), linear_eigenvalue_threshold: str = Form(""),
                  linear_alignment_threshold: str = Form(""), autocorr_max_lag: str = Form("")):
    """Run the analyses on an uploaded track CSV (or a demo dataset) with the given units and parameters."""
    form = dict(coord_unit=coord_unit, pixel_size_um=pixel_size_um, frame_interval_s=frame_interval_s,
                min_track_length=min_track_length, mobility_threshold=mobility_threshold,
                linear_eigenvalue_threshold=linear_eigenvalue_threshold,
                linear_alignment_threshold=linear_alignment_threshold, autocorr_max_lag=autocorr_max_lag)
    try:
        p = parse_params(form)
        if file is not None and file.filename:
            data = await file.read(MAX_UPLOAD + 1)
            if len(data) > MAX_UPLOAD:
                raise AnalysisError("File too large (limit 100 MB).")
            name = Path(file.filename).name
            df = read_tracks_csv(data)
            r = analyse(df, p, name)
            r["source"] = {"id": "upload", "title": name, "description": "Uploaded file"}
            r["coords"] = coords(df)
        elif demo in DEMOS:
            r = analyse(demo_frame(demo), p, f"{demo}.csv")
            r["source"] = demo_meta(demo)
            r["coords_ref"] = f"/demo/{demo}/tracks"
        else:
            raise AnalysisError("Choose a demo dataset or upload a CSV file.")
    except AnalysisError as e:
        raise HTTPException(400, str(e))
    return r


# GET paths (without the /api prefix) written to the static site as api/<path>.json.
STATIC_ENDPOINTS = ["/info"] + [f"/demo/{d}/{k}" for d in DEMOS for k in ("result", "tracks")]
