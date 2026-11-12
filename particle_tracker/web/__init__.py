"""Web interface for Trajectory Analysis (FastAPI + plain ES modules; port 8212).

app.py        routes under /api and STATIC_ENDPOINTS (exported by scripts/build_static.py)
analysis.py   runs FeatureCalculator, AdvancedMetricsCalculator (MSD) and DirectionAutocorrelationAnalyzer -> JSON
tracks_io.py  column names, units, parameters, CSV reading
demos.py      demo datasets from the repo (example results file, test generators)
static/       front end; engine*.js is the browser port of the analyses (engine.test.mjs checks it against Python)
"""
