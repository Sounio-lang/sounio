#!/usr/bin/env node
// Node half of docs/audit/repro/matrix_er_kp_bits_probe.sio. Same grid, same arithmetic:
// cur* = verbatim website/src/lib/pbpk28_core.mjs merLnUnit/merExp/merPow (n != 0.5, 1);
// new* = verbatim port of stdlib/math/pure.sio ln()/exp() (`as i32` -> Math.trunc).
// Usage: node matrix_er_kp_bits_probe.mjs [sounio_probe_output.txt]
//   no arg  -> prints the Node KP| lines plus accuracy vs Math.pow
//   one arg -> additionally diffs bits against the Sounio run and prints PARITY lines;
//              exits 1 (PARITY_FAIL) on a row-count mismatch or any differing bit
import { readFileSync } from 'node:fs';

function curLnUnit(x) { if (x <= 0.0) return -1.0e6; const y = (x - 1.0) / (x + 1.0); const y2 = y * y;
  let term = y, sum = term; for (let k = 1; k < 20; k++) { term = term * y2; sum = sum + term / (2.0 * k + 1.0); } return 2.0 * sum; }
function curExp(x) { let r = 1.0 + x / 1024.0; for (let i = 0; i < 10; i++) r = r * r; return r; }
function curPow(t, n) { if (t <= 0.0) return 0.0; const lnT = (t > 2.0) ? (0.6931471805599453 + curLnUnit(t / 2.0)) : curLnUnit(t); return curExp(n * lnT); }
const LN2 = 0.6931471805599453;
function pureLn(x) { if (x <= 0.0) return -1.0e30; if (x === 1.0) return 0.0;
  let m = x, e = 0; while (m >= 2.0) { m = m / 2.0; e = e + 1; } while (m < 0.5) { m = m * 2.0; e = e - 1; }
  const t = (m - 1.0) / (m + 1.0), t2 = t * t; let sum = t, term = t;
  for (let k = 1; k < 30; k++) { term = term * t2; sum = sum + term / (2 * k + 1); } return 2.0 * sum + e * LN2; }
function pureExp(x) { if (x > 500.0) return 1.0e200; if (x < -500.0) return 0.0;
  const kf = x / LN2; const k = kf >= 0.0 ? Math.trunc(kf) : Math.trunc(kf) - 1; const r = x - k * LN2;
  let sum = 1.0, term = 1.0; for (let n = 1; n < 25; n++) { term = term * r / n; sum = sum + term; }
  let res = sum; if (k >= 0) { for (let i = 0; i < k; i++) res = res * 2.0; } else { for (let i = 0; i < -k; i++) res = res / 2.0; } return res; }
function newPow(t, n) { if (t <= 0.0) return 0.0; return pureExp(n * pureLn(t)); }
const cap = f => (f > 1.0 ? 1.0 : f);
const buf = new DataView(new ArrayBuffer(8));
const bits = x => { buf.setFloat64(0, x); return buf.getBigInt64(0).toString(); };

const rows = [];
const emit = (tag, t) => rows.push([tag, t, cap(0.199 * curPow(t, 0.65)), cap(0.199 * newPow(t, 0.65))]);
for (const t of [3.5e-15, 1.0e-12, 1.0e-9, 1.0e-6, 1.0e-4, 1.0e-3, 0.01, 0.02]) emit(1, t);
for (const [dt, spt] of [[0.4, 60], [0.2, 120], [0.1, 240], [0.08, 300], [0.05, 480]])
  for (let k = 1; k <= 5; k++) emit(2, (k * spt - 1) * dt - k * 24.0 + dt);
for (let i = 1; i <= 490; i++) emit(3, i * 0.05);

let maxCur = 0, maxNew = 0, maxCurMid = 0;
for (const [tag, t, fc, fn] of rows) {
  console.log(`KP|${tag}|${bits(t)}|${bits(fc)}|${bits(fn)}`);
  if (t <= 0) continue;
  const ex = cap(0.199 * Math.pow(t, 0.65));
  const ec = Math.abs(fc - ex) / ex, en = Math.abs(fn - ex) / ex;
  if (ec > maxCur) maxCur = ec; if (en > maxNew) maxNew = en;
  if (tag === 3 && ec > maxCurMid) maxCurMid = ec;
}
console.log(`ACCURACY cur_max_rel=${maxCur.toExponential(3)} cur_max_rel_t>=0.05=${maxCurMid.toExponential(3)} new_max_rel=${maxNew.toExponential(3)} (vs Math.pow)`);
const clocks = rows.filter(r => r[0] === 2).map(r => r[1]);
console.log(`CLOCKS (n*dt-k*24)+dt = ${clocks.map(c => c.toExponential(2)).join(' ')}`);

if (process.argv[2]) {
  const sio = readFileSync(process.argv[2], 'utf8').split('\n').filter(l => l.startsWith('KP|'));
  const node = rows.map(([tag, t, fc, fn]) => `KP|${tag}|${bits(t)}|${bits(fc)}|${bits(fn)}`);
  let tDiff = 0, cDiff = 0, nDiff = 0;
  for (let i = 0; i < node.length; i++) {
    const a = (sio[i] || '').split('|'), b = node[i].split('|');
    if (a[2] !== b[2]) tDiff++; if (a[3] !== b[3]) cDiff++; if (a[4] !== b[4]) nDiff++;
  }
  console.log(`PARITY rows sio=${sio.length} node=${node.length} t_bits_diff=${tDiff} cur_bits_diff=${cDiff} new_bits_diff=${nDiff}`);
  if (sio.length !== node.length || tDiff || cDiff || nDiff) { console.log('PARITY_FAIL'); process.exit(1); }
  console.log('PARITY_OK');
}
