// sRg mobility options for the browser engine; mirrors particle_tracker/analysis/mobility.py (presets, definitions,
// resolve, classify). engine.test.mjs checks these tables against the Python ones (tests/fixtures/engine_fixture.json).
// Sources: Golan & Sherman 2017 Nat Commun 8:15851 (doi:10.1038/ncomms15851); Ly et al. 2025 Biophys J 124:4457
// (doi:10.1016/j.bpj.2025.01.010). Calibration: docs/srg_calibration.md.

export const SQRT_PI_2 = Math.sqrt(Math.PI / 2);
export const DEFINITIONS = {
  golan_sherman: { factor: SQRT_PI_2, label: '√(π/2)·Rg / mean step (Golan & Sherman; recommended)' },
  rg_over_step: { factor: 1, label: 'Rg / mean step (pathak_lab scripts\' radius_gyration_scaled)' },
};
export const DEFAULT_DEFINITION = 'golan_sherman';
export const CUSTOM = 'custom';
export const DEFAULT_PRESET = 'membrane_tirf';
export const PRESETS = {
  "membrane_tirf": {
    "label": "Membrane protein, TIRF (Golan & Sherman 2017)",
    "mobility_threshold": 2.11,
    "min_classify_length": 50,
    "basis": "threshold 2.11 [source: 95th percentile of fixed cells / immobile gold beads, Golan & Sherman 2017]; minimum 50 points [source: same paper keeps trajectories of >= 50 frames]; [calibration] at 50 points free diffusion is called immobile in 1.4 % of tracks without localisation noise, 2.5 % with noise SD = 0.25 x step SD and 7.8 % at 0.5 x; slow particles (noise ~ step) need >= 150."
  },
  "piezo1_tirf_10fps": {
    "label": "PIEZO1-tdTomato, TIRF 10 fps (Pathak lab)",
    "mobility_threshold": 2.22236433588659,
    "min_classify_length": 200,
    "basis": "threshold 2.2224 [Pathak-lab analysis code; Ly et al. 2025 define it as the 95th percentile of fixed-MEF sRg and plot it at ~2.2, but do not print the number]; minimum 200 points [source: Ly et al. 2025 analyse trajectories of >= 20 s at 10 fps]."
  },
  "fixed_reference_strict": {
    "label": "Immobile / fixed reference, strict (1 % false mobile)",
    "mobility_threshold": 3.89,
    "min_classify_length": 50,
    "basis": "threshold 3.89 [source: 99 % of fixed-control values fall below it, Golan & Sherman 2017], for checking a fixed or immobile reference sample or when false 'mobile' calls are costly; minimum 50 [source, as above]. Best practice (both papers): set the threshold from your own fixed sample (threshold_from_reference)."
  },
  "fast_cytosolic": {
    "label": "Fast diffuser, short tracks (step >> localisation error)",
    "mobility_threshold": 2.11,
    "min_classify_length": 40,
    "basis": "threshold 2.11 [source: Golan & Sherman 2017]; minimum 40 points [calibration: shortest tested length at which noise-free free diffusion is called immobile in <= 5 % of tracks]; only valid when the localisation error is small compared with the step."
  },
  "legacy": {
    "label": "Legacy (before 2026-11: every track classified)",
    "mobility_threshold": 2.11,
    "min_classify_length": 0,
    "basis": "reproduces FeatureCalculator results from before the minimum length was added (no minimum, so short diffusing tracks are mostly called immobile)."
  }
};

export const convertThreshold = (v, from, to) => v * DEFINITIONS[to].factor / DEFINITIONS[from].factor;

const blank = v => v === undefined || v === null || v === '';

// mobility.resolve: a named preset fills keys not given; explicit keys win; a preset (or default) threshold is
// converted to the chosen definition's scale. Throws Error on an unknown preset / definition or a bad minimum.
export function resolve(raw) {
  const preset = blank(raw.srg_preset) ? DEFAULT_PRESET : raw.srg_preset;
  if (preset !== CUSTOM && !(preset in PRESETS)) throw new Error(`srg_preset must be one of ${[...Object.keys(PRESETS), CUSTOM].join(', ')}`);
  const base = PRESETS[preset] || PRESETS[DEFAULT_PRESET];
  const definition = blank(raw.srg_definition) ? DEFAULT_DEFINITION : raw.srg_definition;
  if (!(definition in DEFINITIONS)) throw new Error(`srg_definition must be one of ${Object.keys(DEFINITIONS).join(', ')}`);
  const threshold = blank(raw.mobility_threshold)
    ? convertThreshold(base.mobility_threshold, DEFAULT_DEFINITION, definition) : Number(raw.mobility_threshold);
  const minLen = blank(raw.min_classify_length) ? base.min_classify_length : Number(raw.min_classify_length);
  if (!(minLen >= 0) || !Number.isInteger(minLen)) throw new Error('min_classify_length must be a whole number >= 0');
  return { srg_preset: preset, srg_definition: definition, mobility_threshold: threshold, min_classify_length: minLen };
}

export function scaledRgOf(rg, meanStep, definition) {
  if (Number.isNaN(rg) || Number.isNaN(meanStep) || !(meanStep > 0)) return NaN;
  return DEFINITIONS[definition].factor * rg / meanStep;
}

export function classify(srg, n, threshold, minLen) {
  if (n < minLen) return 'too_short';
  if (Number.isNaN(srg)) return 'unclassified';
  return srg > threshold ? 'mobile' : 'immobile';
}
