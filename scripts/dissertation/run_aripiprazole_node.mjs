#!/usr/bin/env node
// Node-side reference runner for the aripiprazole canonical parity
// (SIO-523). Consumes the pure-JS core (website/src/lib/pbpk28_core.mjs:
// runAripiprazoleScenario) and emits prefixed PARITY records bit-compatible
// with tests/run-pass/dissertation_pbpk28_parity_ref_aripiprazole.sio:
//
//   ARIP|t / ARIP|i / ARIP|cv / ARIP|ct / ARIP|cavg   (parent, 14 organs)
//   AMASS|t / AMASS|m                                 (body mass, mg)
//   AREL|t / AREL|abs / AREL|cyp                      (cumulative mg)
//   ARES|m                                           (exact CN mass
//                                                      identity, mg)
//
// Gate: scripts/ci/dissertation_pbpk28_parity_gate.sh case 29 diffs Sounio↔Node
// within 1.0% RMSE per organ on cavg, plus the mass envelope
// (0 <= m <= absorbed) with the exact residual printed as evidence.
// Parity runs at the NM phenotype (2); IM/PM/UM scale by convention only.

import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const CORE_PATH = resolve(__dirname, '../../website/src/lib/pbpk28_core.mjs');
const core = await import(CORE_PATH);
const { runAripiprazoleScenario, N } = core;

// dt and sample times MUST match the Sounio ref exactly.
const DT = 0.01;
const SAMPLES = [1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0, 72.0, 120.0, 168.0, 240.0, 336.0];

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

const rows = runAripiprazoleScenario(SAMPLES, { dt: DT, pheno: 2 });

const out = [];
out.push('DISSERTATION_PBPK28_ARIPIPRAZOLE_PARITY_NODE v1');
out.push(`compartments=${N}`);
out.push('systems=1');
out.push(`dt=${DT}`);
out.push('integrator=fully_coupled_CN_27state_sink');
out.push('drug=aripiprazole');
out.push('metabolite=dehydro-aripiprazole OUT OF SCOPE');
out.push(`samples=${SAMPLES.length}`);

for (const r of rows) {
  for (let i = 0; i < N; i++) {
    out.push(`ARIP|t=${Number(r.t).toFixed(6)}`);
    out.push(`ARIP|i=${i}`);
    out.push(`ARIP|cv=${fmtSci(r.pCv[i])}`);
    out.push(`ARIP|ct=${fmtSci(r.pCt[i])}`);
    out.push(`ARIP|cavg=${fmtSci(r.avg[i])}`);
  }
  out.push(`AMASS|t=${Number(r.t).toFixed(6)}`);
  out.push(`AMASS|m=${fmt(r.mass)}`);
  out.push(`AREL|t=${Number(r.t).toFixed(6)}`);
  out.push(`AREL|abs=${fmt(r.absorbed)}`);
  out.push(`AREL|cyp=${fmt(r.cyp)}`);
  out.push(`ARES|m=${fmtSci(r.res)}`);
}

out.push('DISSERTATION_PBPK28_ARIPIPRAZOLE_PARITY_DONE');
process.stdout.write(out.join('\n') + '\n');
