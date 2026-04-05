#!/usr/bin/env node
// wasm_validate_pbpk.mjs — Validate pbpk_rapa_demo output
//
// Usage:
//   target/debug/souc run self-hosted/wasm/pbpk_rapa_demo.sio | node scripts/wasm_validate_pbpk.mjs
//
// Exits 0 if all constraints pass, 1 otherwise.

import { createInterface } from 'readline';

const CONSTRAINTS = [
  { key: 'bpr_mean',    label: 'BPR < 0.15 (BBB low penetration)',   check: v => v < 0.15 },
  { key: 'cv_auc',      label: 'CV(AUC) > 0.30 (CYP3A4 variability)', check: v => v > 0.30 },
  { key: 's_cl',        label: 's_cl > 0.50 (CL dominates uncertainty)', check: v => v > 0.50 },
];

const metrics = {};
const rl = createInterface({ input: process.stdin });

rl.on('line', line => {
  try {
    const obj = JSON.parse(line);
    if (obj.metric && obj.value !== undefined) {
      metrics[obj.metric] = obj.value;
    }
  } catch { /* skip non-JSON lines */ }
});

rl.on('close', () => {
  let allPass = true;
  console.log('\n=== PBPK rapamycin CNS — epistemic validation ===\n');

  for (const { key, label, check } of CONSTRAINTS) {
    const val = metrics[key];
    if (val === undefined) {
      console.error(`  MISSING  ${key}: metric not found in output`);
      allPass = false;
      continue;
    }
    const pass = check(val);
    const tag = pass ? '  PASS  ' : '  FAIL  ';
    console.log(`${tag} ${label}`);
    console.log(`         ${key} = ${val.toFixed(4)}`);
    if (!pass) allPass = false;
  }

  console.log('\n' + (allPass ? '✓ all constraints satisfied' : '✗ one or more constraints failed'));
  process.exit(allPass ? 0 : 1);
});
