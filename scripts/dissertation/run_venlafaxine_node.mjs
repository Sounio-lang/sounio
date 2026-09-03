#!/usr/bin/env node
// Node-side reference runner for the venlafaxine XR canonical parity (SISTEMA 2
// controlled-release witness). Consumes the SAME pure-JS core
// (website/src/lib/pbpk28_core.mjs, runVenlafaxineScenario) that the dissertation
// 3D viewer uses, and emits prefixed PARITY records bit-compatible with
// tests/run-pass/dissertation_pbpk28_parity_ref_venlafaxine.sio:
//
//   VPARENT|t / VPARENT|i / VPARENT|cv / VPARENT|ct / VPARENT|cavg   (14 organs)
//   VODV|t    / VODV|i    / VODV|cv    / VODV|ct    / VODV|cavg      (14 organs)
//   VMATRIX|t / VMATRIX|rel                                          (cumulative mg)
//   VRATIO|t  / VRATIO|nm                                            (ODV/parent mass)
//
// Gate: scripts/ci/dissertation_pbpk28_parity_gate.sh cases 10-13 diff Sounio↔Node
// within 1.0% RMSE per organ on cavg (parent + ODV), plus matrix-release and ratio.
// Parity runs at the NM phenotype (R7); PM/IM/UM are verified by the pgx smoke.

import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const CORE_PATH = resolve(__dirname, '../../website/src/lib/pbpk28_core.mjs');
const core = await import(CORE_PATH);
const { runVenlafaxineScenario, N } = core;

// dt and sample times MUST match the Sounio ref exactly.
const DT = 0.5;
const SAMPLES = [1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 18.0, 24.0, 36.0, 48.0, 72.0, 96.0];

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
  out.push(`VRATIO|t=${Number(r.t).toFixed(6)}`);
  out.push(`VRATIO|nm=${fmt(r.ratio)}`);
  out.push(`VMASS|p=${fmt(r.pMass)}`);
  out.push(`VMASS|o=${fmt(r.oMass)}`);
}

out.push('DISSERTATION_PBPK28_VENLAFAXINE_PARITY_DONE');
process.stdout.write(out.join('\n') + '\n');
