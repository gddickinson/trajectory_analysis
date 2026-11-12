// Node test: the browser engine reproduces the Python analysis (web/analysis.py) on tests/fixtures/engine_fixture.json
// (regenerate with scripts/make_engine_fixture.py). Numbers within rtol 1e-9 (atol 1e-12); classes, labels and
// counts exactly.  Run: node particle_tracker/web/static/engine.test.mjs
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { analyse, checkParams, num, parseCsv, standardise } from './engine.js';
import { eigSym2 } from './engine_stats.js';
import { DEFAULT_DEFINITION, DEFAULT_PRESET, DEFINITIONS, PRESETS } from './mobility.js';

const here = dirname(fileURLToPath(import.meta.url));
const fixture = JSON.parse(readFileSync(join(here, '../../../tests/fixtures/engine_fixture.json'), 'utf8'));
let checks = 0, fails = 0;

function close(a, b, where) {
  checks++;
  const x = num(a), y = num(b);
  const ok = (Number.isNaN(x) && Number.isNaN(y)) || x === y || Math.abs(x - y) <= 1e-12 + 1e-9 * Math.abs(y);
  if (!ok) { fails++; if (fails < 25) console.log(`MISMATCH ${where}: js ${a} python ${b}`); }
}
function same(a, b, where) {
  checks++;
  if (a !== b) { fails++; if (fails < 25) console.log(`MISMATCH ${where}: js ${JSON.stringify(a)} python ${JSON.stringify(b)}`); }
}
function compare(js, py, where) {
  if (Array.isArray(py)) {
    same(js.length, py.length, where + '.length');
    py.forEach((v, i) => compare(js[i], v, `${where}[${i}]`));
  } else if (py && typeof py === 'object') {
    for (const k of Object.keys(py)) compare(js?.[k], py[k], `${where}.${k}`);
  } else if (typeof py === 'number' || py === null || py === 'inf' || py === '-inf') close(js, py, where);
  else same(js, py, where);
}

for (const c of fixture.cases) {
  const p = checkParams(c.params);
  const r = analyse(standardise({ header: c.table.header, rows: c.table.rows.map(row => row.map(String)) }), p, c.name + '.csv');
  const py = { ...c.result };
  delete py.computed_by;
  compare(r, py, c.name);   // includes the resolved params (preset, definition, threshold, minimum)
}

// sRg presets and definitions: the same tables as analysis/mobility.py.
{
  const m = fixture.mobility;
  compare(PRESETS, m.presets, 'presets');
  same(Object.keys(PRESETS).join(), Object.keys(m.presets).join(), 'preset keys');
  for (const [k, f] of Object.entries(m.factors)) close(DEFINITIONS[k]?.factor, f, `factor.${k}`);
  same(Object.keys(DEFINITIONS).join(), Object.keys(m.factors).join(), 'definition keys');
  same(DEFAULT_PRESET, m.default_preset, 'default preset'); same(DEFAULT_DEFINITION, m.default_definition, 'default definition');
}

// CSV path: the same table through parseCsv gives the same result.
{
  const c = fixture.cases[2];
  const csv = [c.table.header.join(',')].concat(c.table.rows.map(r => r.join(','))).join('\r\n');
  const r = analyse(standardise(parseCsv(csv)), checkParams(c.params), 'x.csv');
  compare(r.tracks, c.result.tracks, 'csv');
}

// eigSym2: eigenvalues of known matrices and unit eigenvectors.
{
  const { w, v } = eigSym2(2, 1, 2);
  close(Math.max(...w), 3, 'eig.max'); close(Math.min(...w), 1, 'eig.min');
  for (const e of v) close(Math.hypot(...e), 1, 'eig.norm');
}

console.log(fails ? `FAILED: ${fails} of ${checks} checks` : `ok (${checks} checks)`);
process.exit(fails ? 1 : 0);
