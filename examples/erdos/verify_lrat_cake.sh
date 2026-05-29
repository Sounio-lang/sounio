#!/usr/bin/env bash
# examples/erdos/verify_lrat_cake.sh
#
# Reproduce the formally-verified LRAT check of Heule's 529-vertex de Grey graph
# G_529 4-colouring UNSAT certificate using cake_lpr (CakeML/HOL4 machine-code
# extraction — stronger than unverified drat-trim).
#
# Usage:
#   examples/erdos/verify_lrat_cake.sh [WORKDIR]
#
# WORKDIR defaults to /tmp/cake_g529_lrat (created if missing).
# Exit 0 only if drat-trim reports s VERIFIED and cake_lpr prints s VERIFIED UNSAT.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

WORKDIR="${1:-/tmp/cake_g529_lrat}"
GRAPH="$HERE/data/degrey_529.edge"
SOUC_SRC="$HERE/souc_sat.sio"

# CRITICAL: avoid bin/souc infinite recursion; compile with the native self-hosted binary.
export SOUNIO_SOUC_BIN="${SOUNIO_SOUC_BIN:-$ROOT/artifacts/self-hosted/souc-self-hosted-x86_64}"
if [[ ! -x "$SOUNIO_SOUC_BIN" ]]; then
  echo "error: missing self-hosted compiler at $SOUNIO_SOUC_BIN" >&2
  exit 1
fi

DRAT_TRIM_SRC="${DRAT_TRIM_SRC:-/tmp/drat-trim-src}"
DRAT_TRIM_BIN="${DRAT_TRIM_BIN:-/tmp/dtrim}"
CAKE_LPR_SRC="${CAKE_LPR_SRC:-/tmp/cake_lpr_repo}"
CAKE_LPR_BIN="${CAKE_LPR_BIN:-$CAKE_LPR_SRC/cake_lpr}"

ELF="$WORKDIR/souc_sat.elf"
CNF="$WORKDIR/souc_sat_worker.cnf"
DRAT="$WORKDIR/souc_sat_worker.drat"
LRAT="$WORKDIR/g529.lrat"

echo "== G_529 verified LRAT pipeline =="
echo "  graph      : $GRAPH"
echo "  workdir    : $WORKDIR"
echo "  souc bin   : $SOUNIO_SOUC_BIN"
echo "  drat-trim  : $DRAT_TRIM_BIN"
echo "  cake_lpr   : $CAKE_LPR_BIN"

mkdir -p "$WORKDIR"

echo "== 1. compile souc_sat.sio =="
"$SOUNIO_SOUC_BIN" "$SOUC_SRC" "$ELF" >/dev/null
chmod +x "$ELF"

echo "== 2. run CDCL on degrey_529.edge (4-colouring CNF) =="
SOLVER_LOG="$WORKDIR/solver.log"
(
  cd "$WORKDIR"
  "$ELF" 0 4 1 1 "$GRAPH"
) 2>&1 | tee "$SOLVER_LOG"
if ! grep -q 'UNSAT' "$SOLVER_LOG"; then
  echo "error: solver did not report UNSAT (see $SOLVER_LOG)" >&2
  exit 1
fi
if [[ ! -f "$CNF" || ! -f "$DRAT" ]]; then
  echo "error: solver did not write $CNF / $DRAT" >&2
  exit 1
fi
echo "  wrote $(stat -c '%s bytes' "$CNF") CNF, $(stat -c '%s bytes' "$DRAT") DRAT"

echo "== 3. build drat-trim (unverified C; used only to emit LRAT hints) =="
if [[ ! -x "$DRAT_TRIM_BIN" ]]; then
  rm -rf "$DRAT_TRIM_SRC"
  git clone --depth 1 https://github.com/marijnheule/drat-trim.git "$DRAT_TRIM_SRC"
  gcc -O2 "$DRAT_TRIM_SRC/drat-trim.c" -o "$DRAT_TRIM_BIN"
fi

echo "== 4. DRAT verify + LRAT generation (drat-trim -L) =="
(
  cd "$WORKDIR"
  "$DRAT_TRIM_BIN" souc_sat_worker.cnf souc_sat_worker.drat -L g529.lrat 2>&1 \
    | tr '\r' '\n' | tee "$WORKDIR/drat_trim.log"
)
if ! grep -q '^s VERIFIED' "$WORKDIR/drat_trim.log"; then
  echo "error: drat-trim did not report s VERIFIED" >&2
  exit 1
fi
if [[ ! -s "$LRAT" ]]; then
  echo "error: empty LRAT at $LRAT" >&2
  exit 1
fi
echo "  wrote $(stat -c '%s bytes' "$LRAT") LRAT"

echo "== 5. obtain cake_lpr (verified LRAT checker) =="
# Pre-extracted CakeML x86-64 assembly — no HOL4/PolyML bootstrap required.
# Source/proofs: https://github.com/CakeML/cakeml/tree/master/examples/lpr_checker
# Binary release repo: https://github.com/tanyongkiam/cake_lpr
if [[ ! -x "$CAKE_LPR_BIN" ]]; then
  rm -rf "$CAKE_LPR_SRC"
  git clone --depth 1 https://github.com/tanyongkiam/cake_lpr.git "$CAKE_LPR_SRC"
  make -C "$CAKE_LPR_SRC" cake_lpr
fi
echo "  cake_lpr repo: $(git -C "$CAKE_LPR_SRC" rev-parse HEAD 2>/dev/null || echo unknown)"

echo "== 6. cake_lpr verified LRAT check =="
# Large proofs need a bigger CakeML heap/stack (see cake_lpr README / basis_ffi.c).
export CML_HEAP_SIZE="${CML_HEAP_SIZE:-65536}"
export CML_STACK_SIZE="${CML_STACK_SIZE:-16384}"
(
  cd "$WORKDIR"
  "$CAKE_LPR_BIN" souc_sat_worker.cnf g529.lrat 2>&1 | tee "$WORKDIR/cake_lpr.log"
)
if ! grep -q '^s VERIFIED UNSAT' "$WORKDIR/cake_lpr.log"; then
  echo "error: cake_lpr did not report s VERIFIED UNSAT" >&2
  exit 1
fi

echo "== done: G_529 4-colouring UNSAT formally verified via cake_lpr =="
echo "  certificate: $CNF"
echo "  LRAT proof : $LRAT"
echo "  logs       : $WORKDIR/drat_trim.log $WORKDIR/cake_lpr.log"
