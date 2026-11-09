#!/usr/bin/env node
// Node-side reference runner for the risperidone + 9-OH canonical parity
// (SIO-522). Consumes the pure-JS core (website/src/lib/pbpk28_core.mjs:
// runRisperidoneScenario) and emits prefixed PARITY records bit-compatible
// with tests/run-pass/dissertation_pbpk28_parity_ref_risperidone.sio:
//
//   RISP|t / RISP|i / RISP|cv / RISP|ct / RISP|cavg     (parent, 14 organs)
//   R9OH|t / R9OH|i / R9OH|cv / R9OH|ct / R9OH|cavg     (9-OH, 14 organs)
//   RMASS|t / RMASS|p / RMASS|o                          (body masses, mg)
//   RREL|t / RREL|abs / RREL|form                        (cumulative mg)
//   RRES|p / RRES|o                                      (two-sided CN mass
//                                                          identity, mg)
//
// Gate: scripts/ci/dissertation_pbpk28_parity_gate.sh case 28 diffs Sounio↔Node
// within 1.0% RMSE per organ on cavg (parent + 9-OH), plus mass envelopes
// (parent within [0, absorbed], 9-OH within [0, formed x MW ratio]).
// Parity runs at the NM phenotype (2); PM/IM/UM scale by convention only.

import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const CORE_PATH = resolve(__dirname, '../../website/src/lib/pbpk28_core.mjs');
const core = await import(CORE_PATH);
const { runRisperidoneScenario, N } = core;

// dt and sample times MUST match the Sounio ref exactly.
const DT = 0.01;
const SAMPLES = [1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 18.0, 24.0, 36.0, 48.0, 72.0, 96.0];

// Match Sounio f64 println: 6-decimal fixed when |x| ≥ 1e-6 (or 0), else scientific.
function fmt(x) {
  if (Math.abs(x) >= 1e-6 || x === 0) return Number(x).toFixed(6);
  return Number(x).toExponential(6);
}

// Match Sounio println_f64_sci(_, 10): 10 significant digits, always scientific.
function fmtSci(x) {
  if (x === 0) return '0.0000000000e+00';
  return Number(x).toExponential(10);
}

const rows = runRisperidoneScenario(SAMPLES, { dt: DT, pheno: 2 });

const out = [];
out.push('DISSERTATION_PBPK28_RISPERIDONE_PARITY_NODE v1');
out.push(`compartments=${N}`);
out.push('systems=2');
out.push(`dt=${DT}`);
out.push('integrator=fully_coupled_CN_27state_x2');
out.push('drug=risperidone');
out.push('metabolite=9-hydroxyrisperidone');
out.push(`samples=${SAMPLES.length}`);

for (const r of rows) {
  for (let i = 0; i < N; i++) {
    out.push(`RISP|t=${Number(r.t).toFixed(6)}`);
    out.push(`RISP|i=${i}`);
    out.push(`RISP|cv=${fmtSci(r.pCv[i])}`);
    out.push(`RISP|ct=${fmtSci(r.pCt[i])}`);
    out.push(`RISP|cavg=${fmtSci(r.pAvg[i])}`);
  }
  for (let i = 0; i < N; i++) {
    out.push(`R9OH|t=${Number(r.t).toFixed(6)}`);
    out.push(`R9OH|i=${i}`);
    out.push(`R9OH|cv=${fmtSci(r.oCv[i])}`);
    out.push(`R9OH|ct=${fmtSci(r.oCt[i])}`);
    out.push(`R9OH|cavg=${fmtSci(r.oAvg[i])}`);
  }
  out.push(`RMASS|t=${Number(r.t).toFixed(6)}`);
  out.push(`RMASS|p=${fmt(r.pMass)}`);
  out.push(`RMASS|o=${fmt(r.oMass)}`);
  out.push(`RREL|t=${Number(r.t).toFixed(6)}`);
  out.push(`RREL|abs=${fmt(r.absorbed)}`);
  out.push(`RREL|form=${fmt(r.formed)}`);
  out.push(`RRES|p=${fmtSci(r.resP)}`);
  out.push(`RRES|o=${fmtSci(r.resO)}`);
}

out.push('DISSERTATION_PBPK28_RISPERIDONE_PARITY_DONE');
process.stdout.write(out.join('\n') + '\n');
