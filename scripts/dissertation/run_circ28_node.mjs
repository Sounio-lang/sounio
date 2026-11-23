#!/usr/bin/env node
// Node-side runner for the closed-circulation PBPK kernel mirror
// (website/src/lib/circ28_core.mjs). Emits the same `KEY value` lines as
// tests/run-pass/circ28_parity_emit.sio; scripts/ci/circ28_parity_gate.sh
// compares the two key by key.
//
// Number format: value = M p E means M * 2^E exactly (M a signed integer of at
// most 53 bits, normalised to 2^52 <= |M| < 2^53, or `0p0`). Sounio prints
// f64 with six decimals (println) or at most 15 significant digits
// (io::sci_print), neither of which round-trips a double; the binary form
// does, bit for bit.
//
// Flags (negative control only): --perturb-qha=<rel> scales the hepatic
// artery flow q[1] by (1 + rel).

import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const core = await import(resolve(here, '../../website/src/lib/circ28_core.mjs'));

const args = Object.fromEntries(
  process.argv.slice(2).filter((a) => a.startsWith('--')).map((a) => a.slice(2).split('='))
);
const perturbQha = args['perturb-qha'] !== undefined ? Number(args['perturb-qha']) : 0;

const buf = new DataView(new ArrayBuffer(8));
export function fmt(x) {
  if (x === 0) return '0p0';
  if (!Number.isFinite(x)) return String(x);       // refused by the gate
  buf.setFloat64(0, x);
  const hi = buf.getUint32(0), lo = buf.getUint32(4);
  const sign = hi >>> 31;
  const be = (hi >>> 20) & 0x7ff;
  if (be === 0) return String(x);                  // subnormal: not expected, refused
  const frac = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
  const m = (1n << 52n) | frac;
  return `${sign ? '-' : ''}${m}p${be - 1075}`;
}

const lines = [];
const emit = (tag, name, x) => lines.push(`${tag}_${name} ${fmt(x)}`);

const TIMES = [0.5, 2, 8, 24, 48];
const TIME_LABELS = ['0.5', '2', '8', '24', '48'];

function scenario(tag, p, spec) {
  if (perturbQha !== 0) p.q[1] = p.q[1] * (1 + perturbQha);
  const r = core.runScenario({ p, sampleTimes: TIMES, ...spec });
  TIME_LABELS.forEach((lab, j) => {
    emit(tag, `cven_t${lab}`, r.samples[j].ven);
    emit(tag, `cart_t${lab}`, r.samples[j].art);
  });
  emit(tag, 'liver_T_end', r.x[core.it(1)]);
  emit(tag, 'adipose_T_end', r.x[core.it(7)]);
  emit(tag, 'hepatic', r.el.hepatic);
  emit(tag, 'formed', r.el.formed);
  emit(tag, 'renal', r.el.renal);
  emit(tag, 'plasma', r.el.plasma);
  emit(tag, 'gut', r.el.gut);
  emit(tag, 'auc_ven', r.el.auc_ven);
  emit(tag, 'auc_art', r.el.auc_art);
  emit(tag, 'mass_end', r.massEnd);
  emit(tag, 'eh', core.hepaticExtraction(p));
  emit(tag, 'fg', core.gutAvailability(p));
  emit(tag, 'eg', core.gutExtraction(p));
  emit(tag, 'cl_splanchnic', core.splanchnicClearance(p));
  emit(tag, 'vss', core.vssPlasma(p));
  emit(tag, 'co', core.cardiacOutput(p));
  emit(tag, 'vb', core.bloodVolume(p));
}

// S1 IV infusion 10 mg over 0.5 h, plasma clearance 2 L/h
{ const p = core.ckSet(); p.cl_plasma = 2.0;
  scenario('S1', p, { dt: 0.05, horizon: 48, route: 'iv', dose: 10, dur: 0.5 }); }
// S2 IV, hepatic clint 500 (clform 100 added so the `formed` tally is exercised)
{ const p = core.ckSet(); p.clint = 500.0; p.clform = 100.0;
  scenario('S2', p, { dt: 0.05, horizon: 48, route: 'iv', dose: 10, dur: 0.5 }); }
// S3 oral first order ka 1/h, clint 500, clint_gut 300
{ const p = core.ckSet(); p.clint = 500.0; p.clint_gut = 300.0;
  scenario('S3', p, { dt: 0.05, horizon: 48, route: 'oral', dose: 10, ka: 1.0 }); }
// S4 renal clearance 5 L/h
{ const p = core.ckSet(); p.cl_renal = 5.0;
  scenario('S4', p, { dt: 0.05, horizon: 48, route: 'iv', dose: 10, dur: 0.5 }); }
// S5 stiff: 10 mg infused over one 0.25 h step, plasma clearance 2 L/h
{ const p = core.ckSet(); p.cl_plasma = 2.0;
  scenario('S5', p, { dt: 0.25, horizon: 48, route: 'iv', dose: 10, dur: 0.25 }); }

console.log(lines.join('\n'));
console.log('CIRC28_PARITY_EMIT_DONE');
