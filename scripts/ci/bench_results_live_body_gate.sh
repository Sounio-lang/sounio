#!/usr/bin/env bash
# scripts/ci/bench_results_live_body_gate.sh
#
# A benchmark row in scripts/bench/RESULTS.md that credits a grouped verb's
# speed to a "dense" (direct-index, no hashing) engine must describe the body
# that callers actually reach (#2375).
#
# Why: a top-level `fn` defined twice is resolved first-definition-wins, so a
# PR that adds a second, dense `bf_cummax_by` ships dead code, and a benchmark
# attached to it credits an engine no caller runs. #1301 did exactly that
# (cummax_by / cummin_by "0.59x, dense direct-index"), and so did #1327
# (cumprod_by "0.70x, dense running product"); the live bodies were the
# open-addressing hash implementations. #2372 deleted the dead bodies but the
# claims stayed.
#
# Check: for every table row whose notes cell says "dense", each `<verb>_by`
# named in its first cell is resolved to the FIRST top-level definition of
# `bf_<verb>_by` under stdlib/data/, and that body must not probe the
# open-addressing hash (`bf_ghash(`). A row naming a verb with no definition
# is reported, not silently skipped. A verb defined more than once fails too:
# the second body is dead and any number attached to it is unattributable.
#
# GATE_CONTRACT: v0
# GATE_ID: bench_results_live_body
# GATE_CLAIMS: every RESULTS.md row crediting a dense engine to a grouped
#              bigframe verb describes that verb's live (first) definition
# GATE_ENGINE: none (text over the tree)
# GATE_RESULT_ON_SKIP: fail
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS="${RESULTS:-$ROOT_DIR/scripts/bench/RESULTS.md}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/stdlib/data}"

[[ -f "$RESULTS" ]] || { echo "error: missing $RESULTS" >&2; exit 1; }
[[ -d "$DATA_DIR" ]] || { echo "error: missing $DATA_DIR" >&2; exit 1; }

echo "BENCH_RESULTS_LIVE_BODY_GATE_START"

# live_body VERB -> prints the first definition's body; returns 1 if none.
live_body() {
  local fn="$1" f
  for f in "$DATA_DIR"/*.sio; do
    if grep -qE "^(pub )?fn ${fn}\(" "$f"; then
      awk -v fn="$fn" '
        $0 ~ "^(pub )?fn " fn "\\(" { on = 1 }
        on { print }
        on && /^}/ { exit }
      ' "$f"
      return 0
    fi
  done
  return 1
}

n_defs() {
  local fn="$1"
  { grep -hE "^(pub )?fn ${fn}\(" "$DATA_DIR"/*.sio || true; } | wc -l | tr -d ' '
}

fails=0
checked=0
rows=0
while IFS= read -r line; do
  # Table rows only; the notes cell is the last non-empty cell.
  [[ "$line" == "|"* ]] || continue
  notes="$(printf '%s' "$line" | awk -F'|' '{ for (i = NF; i > 0; i--) if ($i ~ /[^ ]/) { print $i; exit } }')"
  printf '%s' "$notes" | grep -qi 'dense' || continue
  first="$(printf '%s' "$line" | awk -F'|' '{ print $2 }')"
  verbs="$(printf '%s' "$first" | grep -oE '\b[a-z][a-z0-9_]*_by\b' | sort -u || true)"
  [[ -n "$verbs" ]] || continue
  rows=$((rows + 1))
  for v in $verbs; do
    fn="bf_$v"
    nd="$(n_defs "$fn")"
    if [[ "$nd" -eq 0 ]]; then
      # Rows also name grouped verbs that are implemented under another name
      # (e.g. a shared engine); only verbs that exist as bf_<verb> are checked.
      continue
    fi
    checked=$((checked + 1))
    if [[ "$nd" -gt 1 ]]; then
      echo "FAIL  $fn is defined $nd times under stdlib/data; only the first runs"
      fails=$((fails + 1))
      continue
    fi
    if live_body "$fn" | grep -q 'bf_ghash('; then
      echo "FAIL  $fn: RESULTS.md credits a dense engine, but the live body hashes (bf_ghash)"
      echo "      row: $(printf '%s' "$line" | cut -c1-140)"
      fails=$((fails + 1))
    fi
  done
done < "$RESULTS"

echo "dense_rows=$rows verbs_checked=$checked failures=$fails"
if [[ "$checked" -eq 0 ]]; then
  echo "BENCH_RESULTS_LIVE_BODY_GATE=FAIL (checked nothing: the row or definition patterns no longer match)" >&2
  exit 1
fi
if [[ "$fails" -ne 0 ]]; then
  echo "BENCH_RESULTS_LIVE_BODY_GATE=FAIL" >&2
  exit 1
fi
echo "BENCH_RESULTS_LIVE_BODY_GATE_OK"
