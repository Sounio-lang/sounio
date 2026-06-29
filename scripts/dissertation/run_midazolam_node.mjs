#!/usr/bin/env node
// Node-side reference runner for midazolam — fifth canonical drug, the suite's
// first CYP3A drug–drug interaction (DDI). Consumes the SAME pure-JS core
// (website/src/lib/pbpk28_core.mjs, runMidazolamScenario / runMidazolamDDIResponse)
// and emits prefixed records bit-compatible with
// tests/run-pass/dissertation_pbpk28_parity_ref_midazolam.sio:
//
//   MDZ|t / MDZ|plasma_solo / MDZ|plasma_inh           (time-series, 12 samples)
//   MDZ|igrid / MDZ|F / MDZ|clh / MDZ|aucr             (DDI dose-response, 8 points)
//
// Parity surface (gate cases 17-19): solo oral PK · ketoconazole-inhibited oral PK
// · DDI inhibition curve (oral F, systemic CL_h, AUC ratio across I/Ki). Both sides
// use a FIXED-STEP RK4 systemic integrator at the same dt (the production scenario
// uses adaptive Tsit5 — an intractable bit-for-bit parity target); the inhibitor is
// held STATIC at its steady-state unbound concentration.

import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const CORE_PATH = resolve(__dirname, '../../website/src/lib/pbpk28_core.mjs');
const core = await import(CORE_PATH);
const { runMidazolamScenario, runMidazolamDDIResponse, MDZ_KETO } = core;

const DT = 0.002;
const DOSE = 5.0;
const SAMPLES = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 18.0, 24.0, 36.0];
const IGRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0];

function fmt(x) {
  if (Math.abs(x) >= 1e-6 || x === 0) return Number(x).toFixed(6);
  return Number(x).toExponential(6);
}

const solo = runMidazolamScenario(SAMPLES, { dt: DT, doseMg: DOSE, I: 0.0 });
const inh  = runMidazolamScenario(SAMPLES, { dt: DT, doseMg: DOSE, I: MDZ_KETO.iSteadyState });
const dd   = runMidazolamDDIResponse(IGRID);

const out = [];
out.push('DISSERTATION_PBPK28_MIDAZOLAM_PARITY_NODE v1');
out.push('model=pbpk14_wellstirred+mechanistic_firstpass+cyp3a_competitive_ddi');
out.push('integrator=fixed_step_rk4');
out.push('dt=0.002');
out.push('dose_mg=5.0');
out.push('inhibitor=ketoconazole_static_steady_state');
out.push('drug=midazolam');
out.push(`samples=${SAMPLES.length}`);
out.push(`igrid=${IGRID.length}`);

for (let i = 0; i < solo.length; i++) {
  out.push(`MDZ|t=${Number(solo[i].t).toFixed(6)}`);
  out.push(`MDZ|plasma_solo=${fmt(solo[i].plasma)}`);
  out.push(`MDZ|plasma_inh=${fmt(inh[i].plasma)}`);
}

for (const d of dd) {
  out.push(`MDZ|igrid=${Number(d.iOverKi).toFixed(6)}`);
  out.push(`MDZ|F=${fmt(d.F)}`);
  out.push(`MDZ|clh=${fmt(d.clH)}`);
  out.push(`MDZ|aucr=${fmt(d.aucr)}`);
}

out.push('DISSERTATION_PBPK28_MIDAZOLAM_PARITY_DONE');
process.stdout.write(out.join('\n') + '\n');
