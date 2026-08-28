#!/usr/bin/env bash
# examples/erdos/portfolio.sh — P1 fork-based diversified portfolio for souc-sat.
#
# Launches N copies of the souc-sat engine (examples/erdos/souc_sat.sio) with
# distinct seeds. Each seed perturbs ONLY the search order (initial phase,
# activity bias, restart cadence) — soundness is seed-independent and every
# worker's DRAT proof is checked by drat-trim. First worker to reach UNSAT
# wins; its certificate (souc_sat_worker.cnf/.drat, written in the worker's
# private directory) is the portfolio certificate.
#
# Usage:  examples/erdos/portfolio.sh [CLIQUE_N] [NUM_SEEDS] [DRAT_TRIM]
#   CLIQUE_N   clique size; solves K_N/(N-1) colouring (UNSAT). Default 7.
#              (N>=8 needs the streamed/heap proof arena — E0 follow-up.)
#   NUM_SEEDS  number of parallel workers. Default 8.
#   DRAT_TRIM  path to drat-trim for cert verification. Default: auto-detect.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SOUC="${SOUC:-$ROOT/bin/souc}"
SRC="$HERE/souc_sat.sio"

CLIQUE_N="${1:-7}"
NUM_SEEDS="${2:-8}"
DRAT_TRIM="${3:-$(command -v drat-trim || echo /tmp/drat-trim)}"

WORKDIR="$(mktemp -d /tmp/souc_portfolio.XXXXXX)"
ELF="$WORKDIR/souc_sat.elf"

echo "== souc-sat portfolio =="
echo "  source     : $SRC"
echo "  clique     : K_${CLIQUE_N}/$((CLIQUE_N-1))  (UNSAT)"
echo "  workers    : $NUM_SEEDS"
echo "  workdir    : $WORKDIR"
echo "  drat-trim  : $DRAT_TRIM"

echo "== compiling engine =="
"$SOUC" "$SRC" "$ELF" 2>&1 | tail -1
chmod +x "$ELF"

echo "== launching $NUM_SEEDS workers on K_${CLIQUE_N}/$((CLIQUE_N-1)) =="
declare -a PIDS
for ((s=1; s<=NUM_SEEDS; s++)); do
  d="$WORKDIR/seed_$s"
  mkdir -p "$d"
  ( cd "$d" && "$ELF" "$s" "$CLIQUE_N" > worker.log 2>&1 ) &
  PIDS+=($!)
done

# Poll for the first worker to report UNSAT.
WINNER=""
while :; do
  alive=0
  for ((s=1; s<=NUM_SEEDS; s++)); do
    d="$WORKDIR/seed_$s"
    if [[ -z "$WINNER" ]] && grep -q "UNSAT" "$d/worker.log" 2>/dev/null; then
      WINNER="$s"
    fi
    if kill -0 "${PIDS[$((s-1))]}" 2>/dev/null; then alive=1; fi
  done
  [[ -n "$WINNER" || "$alive" -eq 0 ]] && break
  sleep 0.05
done

# Stop stragglers once we have a winner.
for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done
wait 2>/dev/null

echo "== per-worker results =="
for ((s=1; s<=NUM_SEEDS; s++)); do
  line="$(grep -E 'UNSAT|OVERFLOW|NORESULT' "$WORKDIR/seed_$s/worker.log" 2>/dev/null | tail -1)"
  printf '  seed %-3s : %s\n' "$s" "${line:-<incomplete>}"
done

if [[ -z "$WINNER" ]]; then
  echo "== no worker closed UNSAT (try larger budget or smaller clique) =="
  exit 1
fi

echo "== winner: seed $WINNER =="
CERT_DIR="$WORKDIR/seed_$WINNER"
if [[ -x "$DRAT_TRIM" && -f "$CERT_DIR/souc_sat_worker.drat" ]]; then
  echo "== verifying winner certificate with drat-trim =="
  "$DRAT_TRIM" "$CERT_DIR/souc_sat_worker.cnf" "$CERT_DIR/souc_sat_worker.drat" 2>&1 | tail -4
else
  echo "  (drat-trim not found at $DRAT_TRIM — cert at $CERT_DIR/souc_sat_worker.drat)"
fi
echo "== certificate: $CERT_DIR/souc_sat_worker.{cnf,drat} =="
