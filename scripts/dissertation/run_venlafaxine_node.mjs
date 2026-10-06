#!/usr/bin/env node
// Node-side reference runner for the venlafaxine XR canonical parity (SISTEMA 2
// controlled-release witness). Consumes the pure-JS core
// (website/src/lib/pbpk28_core.mjs: runVenlafaxineScenario, runVenlafaxineSteadyState)
// and emits prefixed PARITY records bit-compatible with
// tests/run-pass/dissertation_pbpk28_parity_ref_venlafaxine.sio:
//
//   VPARENT|t / VPARENT|i / VPARENT|cv / VPARENT|ct / VPARENT|cavg   (14 organs)
//   VODV|t    / VODV|i    / VODV|cv    / VODV|ct    / VODV|cavg      (14 organs)
//   VMATRIX|t / VMATRIX|rel                                          (cumulative mg)
//   VMASS|p   / VMASS|o                                              (body mass, mg)
//   VSS|dt / auc_parent / auc_odv / ratio / cf / rel_err_cf_e12 /
//   VSS|rel_halfwidth_e12 / certified                                    (steady-state
//                                                                     C_avg ODV/parent)
//
// Gate: scripts/ci/dissertation_pbpk28_parity_gate.sh cases 10-13 diff Sounio↔Node
// within 1.0% RMSE per organ on cavg (parent + ODV), plus matrix release and the
// steady-state ratio (case 13: engine parity at each dt, and the closed form
// inside each engine's certified interval).
// Parity runs at the NM phenotype (R7); PM/IM/UM are verified by the pgx smoke.

import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const CORE_PATH = resolve(__dirname, '../../website/src/lib/pbpk28_core.mjs');
const core = await import(CORE_PATH);
const { runVenlafaxineScenario, runVenlafaxineSteadyState, vfxSsRatioClosedForm, N } = core;

// dt and sample times MUST match the Sounio ref exactly.
const DT = 0.5;
const SAMPLES = [1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 18.0, 24.0, 36.0, 48.0, 72.0, 96.0];
// Steady-state readout (75 mg XR q24h, 10th interval) at two step sizes: the
// ratio is dt-independent up to the certified bound, so both must certify.
const SS_DTS = [0.5, 0.25];

// Match Sounio f64 println: 6-decimal fixed when |x| ≥ 1e-6 (or 0), else scientific.
function fmt(x) {
  if (Math.abs(x) >= 1e-6 || x === 0) return Number(x).toFixed(6);
  return Number(x).toExponential(6);
}

const rows = runVenlafaxineScenario(SAMPLES, { dt: DT, pheno: 2 });

const out = [];
out.push('DISSERTATION_PBPK28_VENLAFAXINE_PARITY_NODE v1');
out.push(`compartments=${N}`);
out.push('dt=0.5');
out.push('integrator=fully_coupled_CN_27state_x2');
out.push('drug=venlafaxine');
out.push('release=korsmeyer_peppas_matrix');
out.push(`samples=${SAMPLES.length}`);

for (const r of rows) {
  for (let i = 0; i < N; i++) {
    out.push(`VPARENT|t=${Number(r.t).toFixed(6)}`);
    out.push(`VPARENT|i=${i}`);
    out.push(`VPARENT|cv=${fmt(r.pCv[i])}`);
    out.push(`VPARENT|ct=${fmt(r.pCt[i])}`);
    out.push(`VPARENT|cavg=${fmt(r.pAvg[i])}`);
  }
  for (let i = 0; i < N; i++) {
    out.push(`VODV|t=${Number(r.t).toFixed(6)}`);
    out.push(`VODV|i=${i}`);
    out.push(`VODV|cv=${fmt(r.oCv[i])}`);
    out.push(`VODV|ct=${fmt(r.oCt[i])}`);
    out.push(`VODV|cavg=${fmt(r.oAvg[i])}`);
  }
  out.push(`VMATRIX|t=${Number(r.t).toFixed(6)}`);
  out.push(`VMATRIX|rel=${fmt(r.released)}`);
  out.push(`VMASS|p=${fmt(r.pMass)}`);
  out.push(`VMASS|o=${fmt(r.oMass)}`);
}

const cf = vfxSsRatioClosedForm(2);
for (const dt of SS_DTS) {
  const ss = runVenlafaxineSteadyState({ dt, pheno: 2, nDoses: 10, tau: 24.0 });
  out.push(`VSS|dt=${Number(dt).toFixed(6)}`);
  out.push(`VSS|auc_parent=${fmt(ss.aucParent)}`);
  out.push(`VSS|auc_odv=${fmt(ss.aucOdv)}`);
  out.push(`VSS|ratio=${fmt(ss.ratio)}`);
  out.push(`VSS|cf=${fmt(cf)}`);
  // ×1e12, as the Sounio ref (Madaros prints f64 as 6-decimal fixed).
  out.push(`VSS|rel_err_cf_e12=${fmt(1.0e12 * (ss.ratio - cf) / cf)}`);
  out.push(`VSS|rel_halfwidth_e12=${fmt(1.0e12 * 0.5 * (ss.ratioHi - ss.ratioLo) / ss.ratio)}`);
  out.push(`VSS|certified=${ss.bounded && ss.ratioLo <= cf && cf <= ss.ratioHi ? 1 : 0}`);
}

out.push('DISSERTATION_PBPK28_VENLAFAXINE_PARITY_DONE');
process.stdout.write(out.join('\n') + '\n');
