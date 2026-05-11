#!/usr/bin/env node
// Node-side reference runner for the dissertation 3D viewer's PBPK28
// permeability-limited model with fully-coupled Crank-Nicolson integration.
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
  DEFAULT_PARAMS_RAPAMYCIN, degenerateParams, N,
} = core;

// CLI: --case=literature | --case=degenerate
//   literature (default): rapamycin at lit PS + vasc_frac (matches Sounio
//     dissertation_pbpk28_parity_ref_rapamycin.sio).
//   degenerate: vasc_frac=1e-3 ∀ organs, psScale=1e4 — exercises CN's A-stable
//     handling of the V_v→0 regime; matches Sounio
//     dissertation_pbpk28_degenerate_parity_ref.sio. Trajectory must converge to
//     PBPK14 well-stirred (Stage C ref) for the asymptotic reduction Case 2.
const args = Object.fromEntries(
  process.argv.slice(2)
    .filter(a => a.startsWith('--'))
    .map(a => a.slice(2).split('='))
    .map(([k, v]) => [k, v === undefined ? true : v])
);
const CASE = args.case || 'literature';

const DT = 0.001;
const BOLUS_MG = 0.05;
const CL_HEP = 12.4;
const SAMPLES = [0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 24.0, 30.0];

const baseParams = {
  ...DEFAULT_PARAMS_RAPAMYCIN,
  clHep: CL_HEP,
  higuchiScale: 1.0,
  vdScale: 1.0,
  bolusMg: BOLUS_MG,
  stentActive: false,
};
const params = (CASE === 'degenerate')
  ? degenerateParams(baseParams, { eps: 1e-3, psScale: 1e4 })
  : baseParams;

const { Cv, Ct, Rfree, DR } = initialState(params);
const scratch = makeScratch();
let t = 0;

const out = [];
out.push('DISSERTATION_PBPK28_PARITY_NODE v1');
out.push(`compartments=${N}`);
out.push(`states=${2 * N}`);
out.push(`dt=${DT}`);
out.push(`bolus_mg=${BOLUS_MG}`);
out.push(`cl_hep_L_per_h=${CL_HEP}`);
out.push('integrator=fully_coupled_CN_27state');
out.push('drug=rapamycin');
out.push(`case=${CASE}`);
out.push(`samples=${SAMPLES.length}`);

// Match Sounio f64 println format ("%.6f" up to 6 decimals, then bare exponential
// when the value falls below ~1e-6 — Sounio's f64 printer switches to scientific
// when fixed-decimal would lose precision). For parity-gate parsing we emit the
// 6-decimal form when |x| ≥ 1e-6, scientific (e-form) when smaller.
function fmt(x) {
  if (Math.abs(x) >= 1e-6 || x === 0) return Number(x).toFixed(6);
  return Number(x).toExponential(6);
}

const TMDD_SET = new Set(params.tmddOrgans || []);

for (const target of SAMPLES) {
  while (t + DT * 0.5 < target) {
    stepStrang(Cv, Ct, t, DT, params, scratch, Rfree, DR);
    t += DT;
  }
  const avg = organAverage(Cv, Ct, params);
  for (let i = 0; i < N; i++) {
    out.push(`PARITY|t=${Number(target).toFixed(6)}`);
    out.push(`PARITY|i=${i}`);
    out.push(`PARITY|cv=${fmt(Cv[i])}`);
    out.push(`PARITY|ct=${fmt(Ct[i])}`);
    out.push(`PARITY|cavg=${fmt(avg[i])}`);
    if (TMDD_SET.has(i)) {
      out.push(`PARITY|rfree=${fmt(Rfree[i])}`);
      out.push(`PARITY|dr=${fmt(DR[i])}`);
    }
  }
}
out.push('DISSERTATION_PBPK28_PARITY_DONE');
process.stdout.write(out.join('\n') + '\n');
