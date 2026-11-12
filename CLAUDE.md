# trajectory_analysis

@INTERFACE.md

- Python: the analysis modules (`particle_tracker/analysis`) must stay Qt-free; the package `__init__` imports lazily.
- Web: `python run_web.py` (port 8212); `python -m pytest -q tests/test_web.py`; after changing any analysis used by the
  web app, run `python scripts/make_engine_fixture.py` and update `particle_tracker/web/static/engine*.js` until
  `node particle_tracker/web/static/engine.test.mjs` passes.
- Keep `SESSION_LOG.md` up to date.
