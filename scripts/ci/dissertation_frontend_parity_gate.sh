#!/usr/bin/env bash
# scripts/ci/dissertation_frontend_parity_gate.sh
#
# Verifies the dissertation 3D viewer's in-browser PBPK14 RK4
# (website/src/lib/pbpk14_core.mjs) agrees with the Sounio reference
# (tests/run-pass/dissertation_frontend_parity_ref.sio) at 12 log-spaced
# sample times across 14 compartments. Pass criterion: per-compartment
# RMSE < 1% of that compartment's peak trajectory value.
#
# Without this gate, the in-browser demo could silently drift from the
# committed Sounio gates, breaking the dissertation's epistemic claims
# (contribution #2 — compile-time confidence gates).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_DISSERTATION_PARITY_DIR:-$(mktemp -d /tmp/sounio-dissertation-parity.XXXXXX)}"
SIO_LOG="$OUT_DIR/sounio.txt"
NODE_LOG="$OUT_DIR/node.txt"
SUMMARY="$OUT_DIR/parity.summary"
RMSE_THRESHOLD_PCT="${SOUNIO_DISSERTATION_PARITY_RMSE_PCT:-1.0}"

echo "[parity] out=$OUT_DIR"
echo "[parity] threshold=${RMSE_THRESHOLD_PCT}% RMSE per compartment"

# ─── Step 1: Sounio reference ─────────────────────────────────────────────────
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

echo "[parity] Sounio reference: bin/souc run tests/run-pass/dissertation_frontend_parity_ref.sio"
"$SOUC_BIN" run tests/run-pass/dissertation_frontend_parity_ref.sio > "$SIO_LOG" 2>&1 || {
  echo "[parity] FAIL: Sounio reference returned non-zero" >&2
  tail -n 40 "$SIO_LOG" >&2
  exit 1
}

if ! grep -q '^DISSERTATION_PARITY_DONE$' "$SIO_LOG"; then
  echo "[parity] FAIL: Sounio reference did not emit DISSERTATION_PARITY_DONE" >&2
  tail -n 40 "$SIO_LOG" >&2
  exit 1
fi

# ─── Step 2: Node runner ──────────────────────────────────────────────────────
if ! command -v node >/dev/null 2>&1; then
  echo "[parity] SKIP: node not available (gate requires Node ≥ 18 for ESM .mjs)" >&2
  exit 0
fi

NODE_RUNNER="$ROOT_DIR/scripts/dissertation/run_pbpk14_node.mjs"
echo "[parity] Node runner: node $NODE_RUNNER"
node "$NODE_RUNNER" > "$NODE_LOG" 2>&1 || {
  echo "[parity] FAIL: Node runner returned non-zero" >&2
  tail -n 40 "$NODE_LOG" >&2
  exit 1
}

if ! grep -q '^DISSERTATION_PARITY_DONE$' "$NODE_LOG"; then
  echo "[parity] FAIL: Node runner did not emit DISSERTATION_PARITY_DONE" >&2
  tail -n 40 "$NODE_LOG" >&2
  exit 1
fi

# ─── Step 3: parse both into TSV (sample_idx, organ_idx, t, C) ───────────────
parse_to_tsv() {
  local in="$1"
  local out="$2"
  awk '
    /^PARITY\|t=/ { t = substr($0, 10); next }
    /^PARITY\|i=/ { i = substr($0, 10); next }
    /^PARITY\|c=/ { c = substr($0, 10); printf "%s\t%s\t%s\n", t, i, c; next }
  ' "$in" > "$out"
}
SIO_TSV="$OUT_DIR/sounio.tsv"
NODE_TSV="$OUT_DIR/node.tsv"
parse_to_tsv "$SIO_LOG" "$SIO_TSV"
parse_to_tsv "$NODE_LOG" "$NODE_TSV"

SIO_ROWS=$(wc -l < "$SIO_TSV")
NODE_ROWS=$(wc -l < "$NODE_TSV")
if [[ "$SIO_ROWS" -ne "$NODE_ROWS" ]]; then
  echo "[parity] FAIL: row count mismatch — Sounio=$SIO_ROWS Node=$NODE_ROWS" >&2
  exit 1
fi
EXPECTED_ROWS=$((12 * 14))
if [[ "$SIO_ROWS" -ne "$EXPECTED_ROWS" ]]; then
  echo "[parity] FAIL: expected $EXPECTED_ROWS rows, got $SIO_ROWS" >&2
  exit 1
fi
echo "[parity] both runs emitted $SIO_ROWS records"

# ─── Step 4: compute per-compartment RMSE and check threshold ─────────────────
# Join on (t, i) and compute (sounio_c - node_c)^2 grouped by i.
JOINED="$OUT_DIR/joined.tsv"
awk -F'\t' '
  NR==FNR { S[$1"|"$2] = $3; next }
  { print $1"\t"$2"\t"S[$1"|"$2]"\t"$3 }
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
        # No measurable trajectory — accept iff RMSE is essentially zero.
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
      printf "PARITY_FAIL %d/14 compartments exceed threshold\n", bad;
      exit 1;
    } else {
      printf "PARITY_PASS 14/14 compartments within %s%% RMSE\n", THR;
      exit 0;
    }
  }
' "$JOINED" | tee "$SUMMARY"

# Exit code propagates from awk via pipefail.
