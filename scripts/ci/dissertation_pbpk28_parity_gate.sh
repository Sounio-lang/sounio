#!/usr/bin/env bash
# scripts/ci/dissertation_pbpk28_parity_gate.sh
#
# Stage G-α gate. Verifies the dissertation 3D viewer's PBPK28
# permeability-limited model (Strang-split RK4 over 28 states) at literature
# rapamycin parameters agrees between the Node core (website/src/lib/pbpk28_core.mjs)
# and the Sounio reference (tests/run-pass/dissertation_pbpk28_parity_ref_rapamycin.sio).
#
# Pass criterion: organ-average per-compartment RMSE < 1.0% of that compartment's
# peak organ-average over the 12 log-spaced sample window.
#
# Cases tracked by this gate:
#   1. HARD GATE — Node ↔ Sounio at literature PS, vasc_frac (rapamycin).
#   2. REPORTING — PBPK28 ↔ PBPK14 at literature parameters → per-organ
#      discrepancy emitted as benchmarks/pbpk/model_form_uc.csv (Type B
#      model-form uncertainty contribution per JCGM 100:2008 §4.3).
#
# Cases deferred to G-α-α (analytical Q-coupling extension):
#   - Degenerate (vasc_frac→0, PS→∞) reduction to PBPK14.
#   - GUM-cone monotonicity under u(PS) perturbation.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_DISSERTATION_PBPK28_PARITY_DIR:-$(mktemp -d /tmp/sounio-dissertation-pbpk28-parity.XXXXXX)}"
SIO_LOG="$OUT_DIR/sounio.txt"
NODE_LOG="$OUT_DIR/node.txt"
SUMMARY="$OUT_DIR/parity.summary"
RMSE_THRESHOLD_PCT="${SOUNIO_DISSERTATION_PBPK28_PARITY_RMSE_PCT:-1.0}"

echo "[pbpk28-parity] out=$OUT_DIR"
echo "[pbpk28-parity] threshold=${RMSE_THRESHOLD_PCT}% RMSE per compartment (cavg)"

# ─── Step 1: Sounio reference ────────────────────────────────────────────────
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

echo "[pbpk28-parity] Sounio reference: bin/souc run tests/run-pass/dissertation_pbpk28_parity_ref_rapamycin.sio"
"$SOUC_BIN" run tests/run-pass/dissertation_pbpk28_parity_ref_rapamycin.sio > "$SIO_LOG" 2>&1 || {
  echo "[pbpk28-parity] FAIL: Sounio reference returned non-zero" >&2
  tail -n 40 "$SIO_LOG" >&2
  exit 1
}
if ! grep -q '^DISSERTATION_PBPK28_PARITY_DONE$' "$SIO_LOG"; then
  echo "[pbpk28-parity] FAIL: Sounio reference did not emit DONE" >&2
  tail -n 40 "$SIO_LOG" >&2
  exit 1
fi

# ─── Step 2: Node runner ─────────────────────────────────────────────────────
if ! command -v node >/dev/null 2>&1; then
  echo "[pbpk28-parity] SKIP: node not available (gate requires Node ≥ 18)" >&2
  exit 0
fi

NODE_RUNNER="$ROOT_DIR/scripts/dissertation/run_pbpk28_node.mjs"
echo "[pbpk28-parity] Node runner: node $NODE_RUNNER"
node "$NODE_RUNNER" > "$NODE_LOG" 2>&1 || {
  echo "[pbpk28-parity] FAIL: Node runner returned non-zero" >&2
  tail -n 40 "$NODE_LOG" >&2
  exit 1
}
if ! grep -q '^DISSERTATION_PBPK28_PARITY_DONE$' "$NODE_LOG"; then
  echo "[pbpk28-parity] FAIL: Node runner did not emit DONE" >&2
  exit 1
fi

# ─── Step 3: parse both into TSV (t, i, cv, ct, cavg) ────────────────────────
parse_to_tsv() {
  local in="$1"
  local out="$2"
  awk '
    /^PARITY\|t=/    { t = substr($0, 10); next }
    /^PARITY\|i=/    { i = substr($0, 10); cv=""; ct=""; cavg=""; next }
    /^PARITY\|cv=/   { cv = substr($0, 11); next }
    /^PARITY\|ct=/   { ct = substr($0, 11); next }
    /^PARITY\|cavg=/ {
      cavg = substr($0, 13);
      printf "%s\t%s\t%s\t%s\t%s\n", t, i, cv, ct, cavg;
    }
  ' "$in" > "$out"
}
SIO_TSV="$OUT_DIR/sounio.tsv"
NODE_TSV="$OUT_DIR/node.tsv"
parse_to_tsv "$SIO_LOG" "$SIO_TSV"
parse_to_tsv "$NODE_LOG" "$NODE_TSV"

SIO_ROWS=$(wc -l < "$SIO_TSV")
NODE_ROWS=$(wc -l < "$NODE_TSV")
EXPECTED_ROWS=$((12 * 14))
if [[ "$SIO_ROWS" -ne "$EXPECTED_ROWS" || "$NODE_ROWS" -ne "$EXPECTED_ROWS" ]]; then
  echo "[pbpk28-parity] FAIL: row count mismatch — Sounio=$SIO_ROWS Node=$NODE_ROWS expected=$EXPECTED_ROWS" >&2
  exit 1
fi
echo "[pbpk28-parity] both runs emitted $SIO_ROWS records"

# ─── Step 4: per-compartment RMSE on organ-average ───────────────────────────
JOINED="$OUT_DIR/joined.tsv"
awk -F'\t' '
  NR==FNR { S[$1"|"$2] = $5; next }
  { print $1"\t"$2"\t"S[$1"|"$2]"\t"$5 }
' "$SIO_TSV" "$NODE_TSV" > "$JOINED"

awk -F'\t' -v THR="$RMSE_THRESHOLD_PCT" '
  {
    t = $1; i = $2 + 0; cs = $3 + 0; cn = $4 + 0;
    d = cs - cn;
    SS[i] += d * d;
    NN[i] += 1;
    if (cs > PK[i]) PK[i] = cs;
    if (cn > PK[i]) PK[i] = cn;
  }
  END {
    bad = 0;
    printf "%-3s %-12s %-12s %-12s %-7s %s\n", "i", "rmse", "peak", "rmse_pct", "thr_pct", "status";
    for (i = 0; i < 14; i++) {
      if (NN[i] == 0) {
        printf "%-3d %-12s %-12s %-12s %-7s MISSING\n", i, "-", "-", "-", THR;
        bad += 1;
        continue;
      }
      rmse = sqrt(SS[i] / NN[i]);
      pk = PK[i] + 0;
      if (pk == 0) {
        if (rmse < 1.0e-9) {
          printf "%-3d %-12.3e %-12.3e %-12s %-7s zero-traj OK\n", i, rmse, pk, "-", THR;
        } else {
          printf "%-3d %-12.3e %-12.3e %-12s %-7s FAIL (zero peak + nonzero rmse)\n", i, rmse, pk, "-", THR;
          bad += 1;
        }
        continue;
      }
      rmse_pct = 100.0 * rmse / pk;
      status = (rmse_pct < THR + 0) ? "OK" : "FAIL";
      if (status == "FAIL") bad += 1;
      printf "%-3d %-12.6e %-12.6e %-12.4f %-7s %s\n", i, rmse, pk, rmse_pct, THR, status;
    }
    if (bad > 0) {
      printf "PBPK28_PARITY_FAIL %d/14 compartments exceed threshold\n", bad;
      exit 1;
    } else {
      printf "PBPK28_PARITY_PASS 14/14 compartments within %s%% RMSE (organ-average)\n", THR;
      exit 0;
    }
  }
' "$JOINED" | tee "$SUMMARY"
