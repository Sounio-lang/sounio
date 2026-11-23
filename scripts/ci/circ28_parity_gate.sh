#!/usr/bin/env bash
# scripts/ci/circ28_parity_gate.sh
#
# Parity gate for the closed-circulation PBPK kernel stdlib/darwin_pbpk/circ28.sio.
# Compares it, key by key, with an independent second implementation in Node
# (website/src/lib/circ28_core.mjs, run by scripts/dissertation/run_circ28_node.mjs):
# mass-form assembly, stage-value SDIRK2, LU with partial pivoting, against the
# Sounio kernel's slope-form SDIRK2 and unpivoted LU.
#
#   1. HARD GATE   Sounio emitter (tests/run-pass/circ28_parity_emit.sio) vs the
#                  Node runner, 5 scenarios x 27 quantities, relative tolerance
#                  CIRC28_PARITY_TOL (default 1e-11). Fails closed on a missing
#                  or extra key, a duplicate, a value that is not `MpE`, or a
#                  missing DONE marker.
#   2. NEGATIVE    the Node runner is rerun with the hepatic artery flow raised
#      CONTROL     by 1e-6 relative; the same comparison must FAIL. A gate that
#                  cannot fail proves nothing.
#
# Values are `M p E` = M * 2^E exactly (see the header of the emitter), so the
# comparison starts from the doubles themselves. Relative discrepancy of a
# pair is |a - b| / max(|a|, |b|), 0 when both are zero.
#
# Tolerance. Measured on 2026-11-15 (Madaros v0.80.0, Node 22): max relative
# discrepancy 9.3e-13 (scenario S5, the stiff one-step infusion, liver tissue
# at 48 h); 5.2e-13, 2.8e-13, 1.9e-13, 6.2e-13 for S1-S4. Default tolerance
# 1e-11, about 10x the measured maximum: two double-precision implementations
# with different operation order agree to roundoff amplified by the condition
# of (I - gamma dt A) (stiff, ps/V up to 1e4 /h) over 192-960 steps, and the
# oral input uses each platform's exp(). The control perturbs by 1e-6, five
# orders above the tolerance.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_CIRC28_PARITY_DIR:-$(mktemp -d /tmp/sounio-circ28-parity.XXXXXX)}"
TOL="${CIRC28_PARITY_TOL:-1e-11}"
EXPECTED_KEYS=135
EMIT_SRC="tests/run-pass/circ28_parity_emit.sio"
NODE_RUNNER="$ROOT_DIR/scripts/dissertation/run_circ28_node.mjs"

if ! command -v node >/dev/null 2>&1; then
  echo "CIRC28_PARITY_GATE_FAIL node not found (the Node mirror is the second implementation)" >&2
  exit 1
fi

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

echo "[circ28-parity] out=$OUT_DIR tol=$TOL"

# ---- comparator ------------------------------------------------------------
CMP="$OUT_DIR/compare.mjs"
cat > "$CMP" <<'JS'
import fs from 'node:fs';
const [fa, fb, tolArg, expectedArg, mode] = process.argv.slice(2);
const tol = Number(tolArg), expected = Number(expectedArg);
const bad = [];
const LINE = /^(S[0-9]+_[A-Za-z0-9_.]+) (\S+)$/;
const NUM = /^(0p0|-?[1-9][0-9]*p-?[0-9]+)$/;
const TWO52 = 1n << 52n, TWO53 = 1n << 53n;
function read(f) {
  const m = new Map();
  for (const line of fs.readFileSync(f, 'utf8').split('\n')) {
    if (!/^S[0-9]+_/.test(line)) continue;
    const g = LINE.exec(line.trimEnd());
    if (!g) { bad.push(`${f}: malformed line: ${line}`); continue; }
    if (m.has(g[1])) bad.push(`${f}: duplicate key ${g[1]}`);
    m.set(g[1], g[2]);
  }
  return m;
}
function val(key, s, f) {
  if (!NUM.test(s)) { bad.push(`${f}: ${key}: non-numeric value '${s}'`); return NaN; }
  const [ms, es] = s.split('p');
  const M = BigInt(ms);
  const a = M < 0n ? -M : M;
  if (a !== 0n && (a < TWO52 || a >= TWO53)) { bad.push(`${f}: ${key}: mantissa out of range '${s}'`); return NaN; }
  const e = Number(es);
  if (e < -1074 || e > 1023) { bad.push(`${f}: ${key}: exponent out of range '${s}'`); return NaN; }
  return Number(M) * 2 ** e;
}
const A = read(fa), B = read(fb);
for (const k of A.keys()) if (!B.has(k)) bad.push(`missing in ${fb}: ${k}`);
for (const k of B.keys()) if (!A.has(k)) bad.push(`missing in ${fa}: ${k}`);
if (A.size !== expected) bad.push(`${fa}: ${A.size} keys, expected ${expected}`);
if (B.size !== expected) bad.push(`${fb}: ${B.size} keys, expected ${expected}`);
let maxRel = 0, nChecked = 0;
const per = {}, over = [];
for (const [k, sa] of A) {
  if (!B.has(k)) continue;
  const x = val(k, sa, fa), y = val(k, B.get(k), fb);
  if (Number.isNaN(x) || Number.isNaN(y)) continue;
  const m = Math.max(Math.abs(x), Math.abs(y));
  const rel = m === 0 ? 0 : Math.abs(x - y) / m;
  nChecked++;
  if (rel > maxRel) maxRel = rel;
  const sc = k.split('_')[0];
  per[sc] = Math.max(per[sc] ?? 0, rel);
  if (!(rel <= tol)) over.push([k, rel]);
}
const fmt = (r) => r.toExponential(3);
console.log('[circ28-parity] max relative discrepancy per scenario: ' +
  Object.keys(per).sort().map((s) => `${s}=${fmt(per[s])}`).join(' '));
const fail = bad.length > 0 || over.length > 0;
if (fail) {
  for (const b of bad) console.log(`[circ28-parity] ${b}`);
  over.sort((p, q) => q[1] - p[1]);
  for (const [k, r] of over.slice(0, mode === 'control' ? 5 : 40)) console.log(`[circ28-parity] over tolerance: ${k} rel=${fmt(r)}`);
  console.log(`CIRC28_PARITY_GATE_FAIL n_keys=${nChecked} n_over=${over.length} n_bad=${bad.length} max_rel=${fmt(maxRel)}`);
  process.exit(1);
}
console.log(`CIRC28_PARITY_GATE_PASS n_keys=${nChecked} max_rel=${fmt(maxRel)}`);
JS

# ---- Sounio side -----------------------------------------------------------
SIO_LOG="$OUT_DIR/sounio.txt"
echo "[circ28-parity] Sounio: $SOUC_BIN run $EMIT_SRC"
"$SOUC_BIN" run "$EMIT_SRC" > "$SIO_LOG" 2>"$OUT_DIR/sounio.err" || {
  echo "CIRC28_PARITY_GATE_FAIL Sounio emitter returned non-zero" >&2
  tail -n 30 "$SIO_LOG" "$OUT_DIR/sounio.err" >&2
  exit 1
}
if ! grep -q '^CIRC28_PARITY_EMIT_DONE$' "$SIO_LOG"; then
  echo "CIRC28_PARITY_GATE_FAIL Sounio emitter did not print CIRC28_PARITY_EMIT_DONE" >&2
  tail -n 30 "$SIO_LOG" >&2
  exit 1
fi

# ---- Node side -------------------------------------------------------------
NODE_LOG="$OUT_DIR/node.txt"
node "$NODE_RUNNER" > "$NODE_LOG" 2>"$OUT_DIR/node.err" || {
  echo "CIRC28_PARITY_GATE_FAIL Node runner returned non-zero" >&2
  tail -n 30 "$OUT_DIR/node.err" >&2
  exit 1
}
if ! grep -q '^CIRC28_PARITY_EMIT_DONE$' "$NODE_LOG"; then
  echo "CIRC28_PARITY_GATE_FAIL Node runner did not print CIRC28_PARITY_EMIT_DONE" >&2
  exit 1
fi

# ---- 1. hard gate ----------------------------------------------------------
if ! node "$CMP" "$SIO_LOG" "$NODE_LOG" "$TOL" "$EXPECTED_KEYS" gate | tee "$OUT_DIR/gate.summary"; then
  exit 1
fi

# ---- 2. negative control ---------------------------------------------------
CTRL_LOG="$OUT_DIR/node_perturbed.txt"
node "$NODE_RUNNER" --perturb-qha=1e-6 > "$CTRL_LOG" 2>"$OUT_DIR/node_perturbed.err" || {
  echo "CIRC28_PARITY_GATE_FAIL negative-control Node run returned non-zero" >&2
  exit 1
}
if node "$CMP" "$SIO_LOG" "$CTRL_LOG" "$TOL" "$EXPECTED_KEYS" control > "$OUT_DIR/control.summary"; then
  cat "$OUT_DIR/control.summary"
  echo "CIRC28_PARITY_GATE_FAIL negative control (hepatic artery flow +1e-6) was NOT detected: the gate cannot fail" >&2
  exit 1
fi
echo "[circ28-parity] negative control (q_hepatic_artery x (1+1e-6)) detected:"
sed 's/^/[circ28-parity]   /' "$OUT_DIR/control.summary" | tail -n 4
echo "CIRC28_PARITY_NEGATIVE_CONTROL_DETECTED"
echo "[circ28-parity] done"
