#!/usr/bin/env node
// Input and interval guards of the Node venlafaxine core
// (website/src/lib/pbpk28_core.mjs), run by
// scripts/ci/dissertation_pbpk28_parity_gate.sh after cases 10-13.
//
//   1. runVenlafaxineScenario rejects negative or non-increasing sample times
//      ([1, 0] used to emit the 1 h state labelled t: 0).
//   2. runVenlafaxineSteadyState's certified interval follows vfx_ss_ratio_lo/hi:
//      when err_parent > AUC_parent (dt = 6 h, NM) the upper end is the 1e300
//      sentinel, not a negative quotient (it was −4.23).
//   3. Large dt is accepted: merExpNeg range-reduces like mer_exp_neg (dt = 24 h
//      used to throw on ka·dt > 1).
// Prints VFX_NODE_GUARDS_PASS, or FAIL lines and exits 1.

import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const core = await import(resolve(__dirname, '../../website/src/lib/pbpk28_core.mjs'));
const { runVenlafaxineScenario, runVenlafaxineSteadyState, vfxSsRatioClosedForm } = core;

let fails = 0;
function fail(msg) { fails++; console.log(`FAIL: ${msg}`); }

for (const bad of [[1.0, 0.0], [-1.0], [2.0, 2.0], [1.0, 4.0, 2.0]]) {
  let threw = false;
  try { runVenlafaxineScenario(bad); } catch (e) { threw = e instanceof RangeError; }
  if (!threw) fail(`sampleTimes ${JSON.stringify(bad)} accepted`);
}
{
  const rows = runVenlafaxineScenario([0.0, 1.0, 2.0]);
  if (!(rows.length === 3 && rows[0].pMass === 0)) fail('sampleTimes [0, 1, 2] not accepted as expected');
}

{
  const ss = runVenlafaxineSteadyState({ dt: 6.0, pheno: 2 });
  console.log(`dt=6 NM: auc_parent=${ss.aucParent} err_parent=${ss.errAucParent} lo=${ss.ratioLo} hi=${ss.ratioHi}`);
  if (!(ss.errAucParent > ss.aucParent)) fail('dt = 6 h no longer exercises err_parent > AUC_parent');
  if (ss.bounded) fail('dt = 6 h reported bounded');
  if (ss.ratioHi !== 1.0e300) fail(`ratioHi = ${ss.ratioHi}, expected the 1e300 sentinel`);
  if (!(ss.ratioLo >= 0.0 && ss.ratioLo <= ss.ratio)) fail(`ratioLo = ${ss.ratioLo} not in [0, ratio]`);
  const cf = vfxSsRatioClosedForm(2);
  if (!(ss.ratioLo <= cf && cf <= ss.ratioHi)) fail('closed form outside the dt = 6 h interval');
}
{
  const ss = runVenlafaxineSteadyState({ dt: 0.5, pheno: 2 });
  if (!(ss.bounded && ss.ratioLo <= ss.ratio && ss.ratio <= ss.ratioHi && ss.ratioHi < 1.0e300)) {
    fail('dt = 0.5 h interval not finite and ordered');
  }
}
for (const dt of [1.0, 24.0]) {
  try {
    const ss = runVenlafaxineSteadyState({ dt, pheno: 2 });
    const cf = vfxSsRatioClosedForm(2);
    if (!(ss.ratioLo <= cf && cf <= ss.ratioHi)) fail(`closed form outside the dt = ${dt} h interval`);
  } catch (e) { fail(`dt = ${dt} h rejected: ${e.message}`); }
}

if (fails > 0) { console.log(`VFX_NODE_GUARDS_FAIL ${fails}`); process.exit(1); }
console.log('VFX_NODE_GUARDS_PASS');
