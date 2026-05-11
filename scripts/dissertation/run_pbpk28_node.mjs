#!/usr/bin/env node
// Node-side reference runner for the dissertation 3D viewer's PBPK28
// permeability-limited RK4 + Strang splitting on PS coupling.
//
// Consumes the SAME pure-JS core (website/src/lib/pbpk28_core.mjs) that the
// React hook will use. Emits PARITY|t / PARITY|i / PARITY|cv / PARITY|ct /
// PARITY|cavg records bit-compatible with
// tests/run-pass/dissertation_pbpk28_parity_ref_rapamycin.sio.
//
// Gate: scripts/ci/dissertation_pbpk28_parity_gate.sh diffs the two within
// 1.0% RMSE per organ on the organ-average (cavg) trajectory.

import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const CORE_PATH = resolve(__dirname, '../../website/src/lib/pbpk28_core.mjs');
const core = await import(CORE_PATH);
const {
  initialState, makeScratch, stepStrang, organAverage,
  DEFAULT_PARAMS_RAPAMYCIN, N,
} = core;

// Identical knobs to tests/run-pass/dissertation_pbpk28_parity_ref_rapamycin.sio:
const DT = 0.001;
const BOLUS_MG = 0.05;
const CL_HEP = 12.4;
const SAMPLES = [0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 24.0, 30.0];

const params = {
  ...DEFAULT_PARAMS_RAPAMYCIN,
  clHep: CL_HEP,
  higuchiScale: 1.0,
  vdScale: 1.0,
  bolusMg: BOLUS_MG,
  stentActive: false,
};

const { Cv, Ct } = initialState(params);
const scratch = makeScratch();
let t = 0;

const out = [];
out.push('DISSERTATION_PBPK28_PARITY_NODE v1');
out.push(`compartments=${N}`);
out.push(`states=${2 * N}`);
out.push(`dt=${DT}`);
out.push(`bolus_mg=${BOLUS_MG}`);
out.push(`cl_hep_L_per_h=${CL_HEP}`);
out.push('integrator=strang_split_PS_then_RK4_then_PS');
out.push('drug=rapamycin');
out.push(`samples=${SAMPLES.length}`);

// Match Sounio f64 println format ("%.6f" up to 6 decimals, then bare exponential
// when the value falls below ~1e-6 — Sounio's f64 printer switches to scientific
// when fixed-decimal would lose precision). For parity-gate parsing we emit the
// 6-decimal form when |x| ≥ 1e-6, scientific (e-form) when smaller.
function fmt(x) {
  if (Math.abs(x) >= 1e-6 || x === 0) return Number(x).toFixed(6);
  return Number(x).toExponential(6);
}

for (const target of SAMPLES) {
  while (t + DT * 0.5 < target) {
    stepStrang(Cv, Ct, t, DT, params, scratch);
    t += DT;
  }
  const avg = organAverage(Cv, Ct, params);
  for (let i = 0; i < N; i++) {
    out.push(`PARITY|t=${Number(target).toFixed(6)}`);
    out.push(`PARITY|i=${i}`);
    out.push(`PARITY|cv=${fmt(Cv[i])}`);
    out.push(`PARITY|ct=${fmt(Ct[i])}`);
    out.push(`PARITY|cavg=${fmt(avg[i])}`);
  }
}
out.push('DISSERTATION_PBPK28_PARITY_DONE');
process.stdout.write(out.join('\n') + '\n');
