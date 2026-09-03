#!/usr/bin/env node
// Node-side reference runner for haloperidol (CASO II) canonical parity.
// Consumes the SAME pure-JS core (website/src/lib/pbpk28_core.mjs,
// runHaloperidolScenario) the dissertation 3D viewer uses, and emits prefixed
// records bit-compatible with
// tests/run-pass/dissertation_pbpk28_parity_ref_haloperidol.sio:
//
//   HALO|t / HALO|plasma / HALO|brain / HALO|isf_free / HALO|icf_free
//        / HALO|kpuu / HALO|d2occ
//
// Parity surface = plasma PK · BBB Kpuu (ISF/ICF) · D2 occupancy, fixed-step RK4
// (the same scheme the engine runs — see the core's haloperidol note). Gate
// cases 14-16 diff Sounio↔Node within 1.0% RMSE per series.

import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const CORE_PATH = resolve(__dirname, '../../website/src/lib/pbpk28_core.mjs');
const core = await import(CORE_PATH);
const { runHaloperidolScenario } = core;

const DT = 0.002;
const DOSE = 5.0;
const SAMPLES = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 18.0, 24.0, 36.0, 48.0, 72.0];

function fmt(x) {
  if (Math.abs(x) >= 1e-6 || x === 0) return Number(x).toFixed(6);
  return Number(x).toExponential(6);
}

const rows = runHaloperidolScenario(SAMPLES, { dt: DT, doseMg: DOSE });

const out = [];
out.push('DISSERTATION_PBPK28_HALOPERIDOL_PARITY_NODE v1');
out.push('model=pbpk14_wellstirred+bbb_isf_icf+d2_occupancy');
out.push('integrator=fixed_step_rk4');
out.push('dt=0.002');
out.push('dose_mg=5.0');
out.push('drug=haloperidol');
out.push(`samples=${SAMPLES.length}`);

for (const r of rows) {
  out.push(`HALO|t=${Number(r.t).toFixed(6)}`);
  out.push(`HALO|plasma=${fmt(r.plasma)}`);
  out.push(`HALO|brain=${fmt(r.brain)}`);
  out.push(`HALO|isf_free=${fmt(r.isfFree)}`);
  out.push(`HALO|icf_free=${fmt(r.icfFree)}`);
  out.push(`HALO|kpuu=${fmt(r.kpuu)}`);
  out.push(`HALO|d2occ=${fmt(r.d2occ)}`);
}

out.push('DISSERTATION_PBPK28_HALOPERIDOL_PARITY_DONE');
process.stdout.write(out.join('\n') + '\n');
