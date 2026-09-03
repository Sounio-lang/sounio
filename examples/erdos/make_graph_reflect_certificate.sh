#!/usr/bin/env bash
# Produce a Lean reflected-LRAT colouring certificate for a DIMACS edge graph.
#
# This is an untrusted certificate producer. It compiles/runs `souc_sat`,
# converts a RUP-addition proof to LRAT with repo-local tooling, then emits a
# Lean module whose `*_unsat` theorem must be checked by Lean's verified LRAT
# checker. A producer bug should make Lean reject; it must not be treated as
# mathematical evidence by itself.
#
# Usage:
#   make_graph_reflect_certificate.sh <edgefile> <k> <out.lean> <name> <Module> [workdir]
#
# Scope:
#   * default: plain `colourCNF n k edges`.
#   * opt-in `SB_MODE=1`: souc_sat triangle precolour, emitted as `colourCNFsb`
#     for k=4 or `colourCNFsb5` for k=5.
#   * intended first for small RUP-addition smoke certificates. Larger proofs
#     with deletions should go through drat-trim -L or a deletion-aware converter.
#   * souc_sat worker CLI here is:
#       ./souc_sat.elf <seed> <k> <use_lrb> <sb_mode> <edgefile>
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 <edgefile> <k> <out.lean> <name> <Module> [workdir]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EDGEFILE="$1"
K="$2"
OUT_LEAN="$3"
NAME="$4"
MODULE="$5"
WORK="${6:-${WORK:-$(mktemp -d)}}"
SOUC="${SOUC:-$ROOT/artifacts/self-hosted/souc-self-hosted-x86_64}"
SEED="${SEED:-0}"         # deterministic search perturbation seed
USE_LRB="${USE_LRB:-1}"   # 1 = LRB branching, 0 = VSIDS fallback
SB_MODE="${SB_MODE:-0}"   # 0 = plain colourCNF, 1 = triangle-precolour SB

[[ -s "$EDGEFILE" ]] || { echo "error: missing/empty edge file $EDGEFILE" >&2; exit 2; }
N="$(awk '$1 == "p" && $2 == "edge" {print $3; exit}' "$EDGEFILE")"
[[ -n "$N" ]] || { echo "error: edge file lacks 'p edge N M' header: $EDGEFILE" >&2; exit 2; }
[[ "$N" =~ ^[0-9]+$ ]] || { echo "error: non-numeric vertex count in $EDGEFILE: $N" >&2; exit 2; }
[[ "$K" =~ ^[0-9]+$ ]] || { echo "error: non-numeric colour count: $K" >&2; exit 2; }
[[ "$K" -gt 0 ]] || { echo "error: colour count must be positive" >&2; exit 2; }
if [[ "$SB_MODE" != "0" && "$SB_MODE" != "1" ]]; then
  echo "error: this reflected producer supports SB_MODE=0 or SB_MODE=1, got $SB_MODE" >&2
  exit 2
fi
if [[ "$SB_MODE" == "1" && "$K" != "4" && "$K" != "5" ]]; then
  echo "error: SB_MODE=1 reflection currently supports k=4 or k=5, got k=$K" >&2
  exit 2
fi

mkdir -p "$WORK"
SOUC_ABS="$(cd "$(dirname "$SOUC")" && pwd)/$(basename "$SOUC")"
EDGE_ABS="$(cd "$(dirname "$EDGEFILE")" && pwd)/$(basename "$EDGEFILE")"
OUT_ABS="$(cd "$(dirname "$OUT_LEAN")" && pwd)/$(basename "$OUT_LEAN")"
CONV="$ROOT/examples/erdos/drup_to_lrat_rup.py"

[[ -x "$SOUC_ABS" ]] || { echo "error: SOUC is not executable: $SOUC_ABS" >&2; exit 2; }
[[ -x "$CONV" ]] || { echo "error: converter is not executable: $CONV" >&2; exit 2; }

"$SOUC_ABS" "$ROOT/examples/erdos/souc_sat.sio" "$WORK/souc_sat.elf"
chmod +x "$WORK/souc_sat.elf"

(
  cd "$WORK"
  ./souc_sat.elf "$SEED" "$K" "$USE_LRB" "$SB_MODE" "$EDGE_ABS" 2>&1 | tee "$WORK/souc_sat.stdout"
)

GEN_ARGS=(
  "$WORK/souc_sat_worker.cnf"
  "$WORK/${NAME}.lrat"
  "$OUT_ABS"
  "$NAME" "$MODULE" "$N" "$K" "$EDGE_ABS"
)

if [[ "$SB_MODE" == "1" ]]; then
  mapfile -t TRIANGLES < <(
    sed -nE 's/^\[precolour triangle ([0-9]+),([0-9]+),([0-9]+)\]$/\1 \2 \3/p' \
      "$WORK/souc_sat.stdout"
  )
  if [[ "${#TRIANGLES[@]}" -ne 1 ]]; then
    echo "error: expected exactly one '[precolour triangle a,b,c]' line, found ${#TRIANGLES[@]}" >&2
    exit 2
  fi
  read -r TRI_A TRI_B TRI_C <<<"${TRIANGLES[0]}"
  GEN_ARGS+=("$TRI_A" "$TRI_B" "$TRI_C")
fi

CNF_VARS="$(awk '$1 == "p" && $2 == "cnf" {print $3; exit}' "$WORK/souc_sat_worker.cnf")"
EXPECTED_VARS=$((N * K))
if [[ "$CNF_VARS" != "$EXPECTED_VARS" ]]; then
  echo "error: CNF var count mismatch: got $CNF_VARS, expected N*K=$EXPECTED_VARS" >&2
  exit 2
fi

if grep -Eq '^[[:space:]]*d[[:space:]]' "$WORK/souc_sat_worker.drat"; then
  echo "error: streamed proof contains deletion records; use drat-trim -L or a deletion-aware converter" >&2
  exit 2
fi

"$CONV" \
  "$WORK/souc_sat_worker.cnf" \
  "$WORK/souc_sat_worker.drat" \
  "$WORK/${NAME}.lrat"

"$ROOT/examples/erdos/gen_lean_sat_reflect.sh" "${GEN_ARGS[@]}"

echo "workdir=$WORK"
echo "edgefile=$EDGE_ABS"
echo "generated=$OUT_ABS"
echo "sb_mode=$SB_MODE"
if [[ "$SB_MODE" == "1" ]]; then
  echo "sb_triangle=$TRI_A,$TRI_B,$TRI_C"
fi
echo "next: cd $ROOT/formal/lean4 && lake build $MODULE"
