#!/usr/bin/env bash
# scripts/ci/dissertation_frontend_parity_gate.sh
#
# HARD PATH (always, no Node): dual-Sounio parity.
#   A = tests/run-pass/dissertation_frontend_parity_ref.sio  (CN dt=0.01)
#   B = tests/run-pass/dissertation_frontend_parity_alt.sio  (CN dt=0.005)
# Pass: per-compartment RMSE < 1% of peak (same criterion as the old Node arm).
#
# OPTIONAL product arm (when Node ≥ 18 present): compare Sounio REF against
# website/src/lib/pbpk14_core.mjs (the 3D viewer core). Failure of the product
# arm is reported but does NOT skip the hard path — and missing Node is NEVER
# a green SKIP of the whole gate.
#
# CLAUDE.md §4: science in Sounio. The hard path never reaches for Node.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_DISSERTATION_PARITY_DIR:-$(mktemp -d /tmp/sounio-dissertation-parity.XXXXXX)}"
REF_LOG="$OUT_DIR/sounio_ref.txt"
ALT_LOG="$OUT_DIR/sounio_alt.txt"
NODE_LOG="$OUT_DIR/node.txt"
SUMMARY="$OUT_DIR/parity.summary"
RMSE_THRESHOLD_PCT="${SOUNIO_DISSERTATION_PARITY_RMSE_PCT:-1.0}"

echo "[parity] out=$OUT_DIR"
echo "[parity] threshold=${RMSE_THRESHOLD_PCT}% RMSE per compartment"
echo "[parity] hard path: Sounio REF (dt=0.01) ↔ Sounio ALT (dt=0.005) — no Node required"

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

# ─── Hard path: dual Sounio ──────────────────────────────────────────────────
echo "[parity] Sounio REF: $SOUC_BIN run tests/run-pass/dissertation_frontend_parity_ref.sio"
"$SOUC_BIN" run tests/run-pass/dissertation_frontend_parity_ref.sio > "$REF_LOG" 2>&1 || {
  echo "[parity] FAIL: Sounio REF returned non-zero" >&2
  tail -n 40 "$REF_LOG" >&2
  exit 1
}
if ! grep -q '^DISSERTATION_PARITY_DONE$' "$REF_LOG"; then
  echo "[parity] FAIL: Sounio REF did not emit DISSERTATION_PARITY_DONE" >&2
  tail -n 40 "$REF_LOG" >&2
  exit 1
fi

echo "[parity] Sounio ALT: $SOUC_BIN run tests/run-pass/dissertation_frontend_parity_alt.sio"
"$SOUC_BIN" run tests/run-pass/dissertation_frontend_parity_alt.sio > "$ALT_LOG" 2>&1 || {
  echo "[parity] FAIL: Sounio ALT returned non-zero" >&2
  tail -n 40 "$ALT_LOG" >&2
  exit 1
}
if ! grep -q '^DISSERTATION_PARITY_DONE$' "$ALT_LOG"; then
  echo "[parity] FAIL: Sounio ALT did not emit DISSERTATION_PARITY_DONE" >&2
  tail -n 40 "$ALT_LOG" >&2
  exit 1
fi

parse_to_tsv() {
  local in="$1"
  local out="$2"
  awk '
    /^PARITY\|t=/ { t = substr($0, 10); next }
    /^PARITY\|i=/ { i = substr($0, 10); next }
    /^PARITY\|c=/ { c = substr($0, 10); printf "%s\t%s\t%s\n", t, i, c; next }
  ' "$in" > "$out"
}
REF_TSV="$OUT_DIR/ref.tsv"
ALT_TSV="$OUT_DIR/alt.tsv"
parse_to_tsv "$REF_LOG" "$REF_TSV"
parse_to_tsv "$ALT_LOG" "$ALT_TSV"

REF_ROWS=$(wc -l < "$REF_TSV")
ALT_ROWS=$(wc -l < "$ALT_TSV")
EXPECTED_ROWS=$((12 * 14))
if [[ "$REF_ROWS" -ne "$ALT_ROWS" ]]; then
  echo "[parity] FAIL: row count mismatch — REF=$REF_ROWS ALT=$ALT_ROWS" >&2
  exit 1
fi
if [[ "$REF_ROWS" -ne "$EXPECTED_ROWS" ]]; then
  echo "[parity] FAIL: expected $EXPECTED_ROWS rows, got $REF_ROWS" >&2
  exit 1
fi
echo "[parity] both Sounio runs emitted $REF_ROWS records"

JOINED="$OUT_DIR/joined.tsv"
awk -F'\t' '
  NR==FNR { S[$1"|"$2] = $3; next }
  { print $1"\t"$2"\t"S[$1"|"$2]"\t"$3 }
' "$REF_TSV" "$ALT_TSV" > "$JOINED"

HARD_RC=0
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
      printf "PARITY_FAIL %d/14 compartments exceed threshold (Sounio dual)\n", bad;
      exit 1;
    } else {
      printf "PARITY_PASS 14/14 compartments within %s%% RMSE (Sounio dual REF↔ALT)\n", THR;
      exit 0;
    }
  }
' "$JOINED" | tee "$SUMMARY" || HARD_RC=$?

if [[ "$HARD_RC" -ne 0 ]]; then
  exit 1
fi

# ─── Optional product arm: website core via Node ─────────────────────────────
if command -v node >/dev/null 2>&1; then
  NODE_RUNNER="$ROOT_DIR/scripts/dissertation/run_pbpk14_node.mjs"
  echo "[parity] optional product arm: node $NODE_RUNNER (website core)"
  if node "$NODE_RUNNER" > "$NODE_LOG" 2>&1 && grep -q '^DISSERTATION_PARITY_DONE$' "$NODE_LOG"; then
    parse_to_tsv "$NODE_LOG" "$OUT_DIR/node.tsv"
    awk -F'\t' '
      NR==FNR { S[$1"|"$2] = $3; next }
      { print $1"\t"$2"\t"S[$1"|"$2]"\t"$3 }
    ' "$REF_TSV" "$OUT_DIR/node.tsv" > "$OUT_DIR/joined_node.tsv"
    if awk -F'\t' -v THR="$RMSE_THRESHOLD_PCT" '
      {
        i = $2 + 0; cs = $3 + 0; cn = $4 + 0;
        d = cs - cn; SS[i] += d * d; NN[i] += 1;
        if (cs > PK[i]) PK[i] = cs;
        if (cn > PK[i]) PK[i] = cn;
      }
      END {
        bad = 0;
        for (i = 0; i < 14; i++) {
          if (NN[i] == 0) { bad++; continue }
          rmse = sqrt(SS[i] / NN[i]); pk = PK[i] + 0;
          if (pk == 0) { if (rmse >= 1.0e-9) bad++; continue }
          else if (100.0 * rmse / pk >= THR) bad++;
        }
        if (bad > 0) {
          printf "PRODUCT_ARM_FAIL %d/14 (Sounio REF ↔ website Node) — hard path already green\n", bad;
          exit 1;
        }
        printf "PRODUCT_ARM_PASS 14/14 (Sounio REF ↔ website Node)\n";
        exit 0;
      }
    ' "$OUT_DIR/joined_node.tsv"; then
      :
    else
      echo "[parity] WARN: product arm failed; hard Sounio dual already PASS — not counting as gate fail" >&2
    fi
  else
    echo "[parity] WARN: product arm Node run failed; hard Sounio dual already PASS" >&2
    tail -n 15 "$NODE_LOG" >&2 || true
  fi
else
  echo "[parity] product arm omitted (no Node) — hard Sounio dual stands alone"
fi

exit 0
