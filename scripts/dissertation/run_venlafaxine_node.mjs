#!/usr/bin/env node
// Node-side reference runner for the venlafaxine XR canonical parity (SISTEMA 2
// controlled-release witness). Consumes the pure-JS core
// (website/src/lib/pbpk28_core.mjs, runVenlafaxineScenario) — an independent
// reimplementation of the portal first-pass scenario and its TR-BDF2 kernel —
// and emits prefixed PARITY records bit-compatible with
// tests/run-pass/dissertation_pbpk28_parity_ref_venlafaxine.sio, which drives
// the stdlib scenario's own step functions:
//
//   VPARENT|t / VPARENT|i / VPARENT|cv / VPARENT|ct / VPARENT|cavg   (14 organs)
//   VODV|t    / VODV|i    / VODV|cv    / VODV|ct    / VODV|cavg      (14 organs)
//   VMATRIX|t / VMATRIX|rel                                          (cumulative mg)
//   VRATIO|t  / VRATIO|nm / VRATIO|auc              (ODV/parent mass; blood AUC)
//   VMASS|p   / VMASS|o                                              (body mg)
//   VBAL|t / released / gut / portal / lost / fabs /
//   resid_{gut,split}_e12 / bound_slack_e12 / resid_{p,o}_e12 / neg_e12 / steps      (mass account; residuals ×1e12 mg)
//   VFINAL|t / aucr / aucr_pred / clf / clf_pred / foral_pred / closed_form (t = 120 h)
//
// Gate: scripts/ci/dissertation_pbpk28_parity_gate.sh cases 10-13 diff Sounio↔Node
// within 1.0% RMSE per organ on cavg (parent + ODV), plus matrix release, the
// ratios, the mass account and the closed-form check.
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
const T_FINAL = 120.0;

// Match Sounio f64 println: 6-decimal fixed when |x| ≥ 1e-6 (or 0), else scientific.
function fmt(x) {
  if (Math.abs(x) >= 1e-6 || x === 0) return Number(x).toFixed(6);
  return Number(x).toExponential(6);
}

const rows = runVenlafaxineScenario([...SAMPLES, T_FINAL], { dt: DT, pheno: 2 });

const out = [];
out.push('DISSERTATION_PBPK28_VENLAFAXINE_PARITY_NODE v2');
out.push(`compartments=${N}`);
out.push(`dt=${DT}`);
out.push('absorption=portal_first_pass');
out.push('integrator=trbdf2_routed_sink_27state_x2');
out.push('drug=venlafaxine');
out.push('release=korsmeyer_peppas_matrix');
out.push(`samples=${SAMPLES.length}`);

for (const r of rows.slice(0, SAMPLES.length)) {
  const t = fmt(r.t);
  for (let i = 0; i < N; i++) {
    out.push(`VPARENT|t=${t}`, `VPARENT|i=${i}`,
             `VPARENT|cv=${fmt(r.pCv[i])}`, `VPARENT|ct=${fmt(r.pCt[i])}`,
             `VPARENT|cavg=${fmt(r.pAvg[i])}`);
  }
  for (let i = 0; i < N; i++) {
    out.push(`VODV|t=${t}`, `VODV|i=${i}`,
             `VODV|cv=${fmt(r.oCv[i])}`, `VODV|ct=${fmt(r.oCt[i])}`,
             `VODV|cavg=${fmt(r.oAvg[i])}`);
  }
  out.push(`VMATRIX|t=${t}`, `VMATRIX|rel=${fmt(r.released)}`);
  out.push(`VRATIO|t=${t}`, `VRATIO|nm=${fmt(r.ratio)}`, `VRATIO|auc=${fmt(r.aucRatio)}`);
  out.push(`VMASS|p=${fmt(r.pMass)}`, `VMASS|o=${fmt(r.oMass)}`);
  out.push(`VBAL|t=${t}`, `VBAL|released=${fmt(r.released)}`, `VBAL|gut=${fmt(r.gut)}`,
           `VBAL|portal=${fmt(r.portal)}`, `VBAL|lost=${fmt(r.lost)}`, `VBAL|fabs=${fmt(r.fAbs)}`,
           `VBAL|resid_gut_e12=${fmt(1.0e12 * r.residGut)}`, `VBAL|resid_split_e12=${fmt(1.0e12 * r.residSplit)}`, `VBAL|bound_slack_e12=${fmt(1.0e12 * r.boundSlack)}`,
           `VBAL|resid_p_e12=${fmt(1.0e12 * r.residP)}`,
           `VBAL|resid_o_e12=${fmt(1.0e12 * r.residO)}`, `VBAL|neg_e12=${fmt(1.0e12 * r.negMass)}`, `VBAL|steps=${r.steps}`);
}

// Closed-form check at the scenario horizon (same criterion as the Sounio ref).
const f = rows[rows.length - 1];
const e1 = Math.abs(f.clf - f.clfPred) / f.clfPred;
const e2 = Math.abs(f.aucRatio - f.aucRatioPred) / f.aucRatioPred;
const ok = e1 <= f.tailP + 1.0e-12 && e2 <= f.tailP + f.tailO + f.tailP * f.tailO + 1.0e-12;
out.push(`VFINAL|t=${fmt(f.t)}`, `VFINAL|aucr=${fmt(f.aucRatio)}`, `VFINAL|aucr_pred=${fmt(f.aucRatioPred)}`,
         `VFINAL|clf=${fmt(f.clf)}`, `VFINAL|clf_pred=${fmt(f.clfPred)}`,
         `VFINAL|foral_pred=${fmt(f.fOralPred)}`, `VFINAL|closed_form=${ok ? 'PASS' : 'FAIL'}`);
out.push('DISSERTATION_PBPK28_VENLAFAXINE_PARITY_DONE');
process.stdout.write(out.join('\n') + '\n');
process.exit(ok ? 0 : 1);
