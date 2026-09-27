// Evidence: docs/audit/MATH_PURE_LN_POS_INF_DISPATCH_2026-09-27.md. Run: node docs/audit/repro/math_pure_ln_grid.mjs
// pureLn (PR #2722 website/src/lib/pbpk28_core.mjs @0466ac7d) vs the +inf-guarded version.
const PURE_LN2 = 0.6931471805599453;
function body(x) {
  let m = x, e = 0;
  while (m >= 2.0) { m = m / 2.0; e = e + 1; }
  while (m < 0.5) { m = m * 2.0; e = e - 1; }
  const t = (m - 1.0) / (m + 1.0), t2 = t * t;
  let sum = t, term = t;
  for (let k = 1; k < 30; k++) { term = term * t2; sum = sum + term / (2 * k + 1); }
  return 2.0 * sum + e * PURE_LN2;
}
function oldLn(x) { if (x <= 0.0) return -1.0e30; if (x === 1.0) return 0.0; return body(x); }
function newLn(x) { if (x <= 0.0) return -1.0e30; if (x === 1.0) return 0.0; if (x / 2.0 === x) return x; return body(x); }
const f = new Float64Array(1), b = new BigInt64Array(f.buffer);
const fromBits = (v) => { b[0] = v; return f[0]; }, bits = (x) => { f[0] = x; return b[0]; };
let n = 0, bad = 0;
const cmp = (x) => { n++; if (bits(oldLn(x)) !== bits(newLn(x))) { bad++; if (bad < 5) console.log('MISMATCH', x); } };
const TOP = 0x7FEFFFFFFFFFFFFFn;
for (let v = 1n; v <= TOP; v += 2199023255531n) cmp(fromBits(v));            // strided, all binades
for (let k = 1n; k <= 65536n; k++) { cmp(fromBits(0x3FF0000000000000n + k)); cmp(fromBits(0x3FF0000000000000n - k)); }
for (let e = 1n; e <= 2046n; e++) for (let j = -8n; j <= 8n; j++) cmp(fromBits(e * 4503599627370496n + j));
for (let t = 0n; t < 65536n; t++) cmp(fromBits(TOP - t));
for (let i = 1; i <= 2000000; i++) cmp(i * 0.001);
for (const x of [0, -0, -1, -1e308, -Infinity, NaN]) cmp(x);
console.log(`JS_GRID n=${n} bad=${bad}`);
console.log(`newLn(Infinity)=${newLn(Infinity)} log10=${newLn(Infinity) / 2.302585092994046}`);
if (bad !== 0) process.exitCode = 1;       // a mismatch must fail a scripted reproduction
