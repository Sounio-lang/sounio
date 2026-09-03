#!/usr/bin/env node
// Node-side reference runner for the dissertation 3D viewer's PBPK14 RK4.
//
// Consumes the SAME pure-JS core (website/src/lib/pbpk14_core.mjs) that the
// React hook uses. Emits PARITY|t / PARITY|i / PARITY|c records bit-compatible
// with tests/run-pass/dissertation_frontend_parity_ref.sio. The orchestrator
// gate scripts/ci/dissertation_frontend_parity_gate.sh diffs the two.
//
// Source of truth on the *math* is whichever side compiles to the canonical
// dissertation submission — but agreement to within 1% RMSE per compartment is
// non-negotiable. Without it the in-browser demo could silently drift from the
// Sounio reference and break the dissertation's epistemic claims.

import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Resolve the core relative to this script — repo-portable.
const CORE_PATH = resolve(__dirname, '../../website/src/lib/pbpk14_core.mjs');
const core = await import(CORE_PATH);
const { initialState, makeScratch, stepRK4, V_REF, N } = core;

// Identical knobs to tests/run-pass/dissertation_frontend_parity_ref.sio:
const DT = 0.01;
const BOLUS_MG = 0.05;
const CL_HEP = 12.4;
const SAMPLES = [0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 24.0, 30.0];

const params = {
  clHep: CL_HEP,
  higuchiScale: 1.0,
  vdScale: 1.0,
  bolusMg: BOLUS_MG,
  stentActive: false,
};

const C = initialState(params);
const scratch = makeScratch();
let t = 0;

const out = [];
out.push('DISSERTATION_PARITY_NODE v1');
out.push(`compartments=${N}`);
out.push(`dt=${DT}`);
out.push(`bolus_mg=${BOLUS_MG}`);
out.push(`cl_hep_L_per_h=${CL_HEP}`);
out.push(`samples=${SAMPLES.length}`);

// Same f64-with-6-decimal format Sounio emits, so the orchestrator can diff
// the two streams with awk and parse identically.
function f6(x) {
  return Number(x).toFixed(6);
}

for (const target of SAMPLES) {
  while (t + DT * 0.5 < target) {
    stepRK4(C, t, DT, params, scratch);
    t += DT;
  }
  for (let i = 0; i < N; i++) {
    out.push(`PARITY|t=${f6(target)}`);
    out.push(`PARITY|i=${i}`);
    out.push(`PARITY|c=${f6(C[i])}`);
  }
}
out.push('DISSERTATION_PARITY_DONE');
process.stdout.write(out.join('\n') + '\n');
