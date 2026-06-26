#!/usr/bin/env bash
# S4 cube-and-conquer runner: dispatch 64 cubes across 64 cores in parallel.
#
# Each cube: souc_sat with SB=1 (triangle precolour) + cube file (extra unit clauses).
# All cubes UNSAT => formula UNSAT.
#
# Usage: run_cube_conquer.sh <souc_sat.elf> <edge_file> <cube_dir>
set -euo pipefail

SOLVER="${1:?missing solver ELF}"
EDGE="${2:?missing edge file}"
CUBEDIR="${3:?missing cube dir}"
K=4
SEED=42
WORK="${CUBEDIR}/run"
mkdir -p "$WORK"

CUBES=$(cat "${CUBEDIR}/manifest.txt")
NCUBES=$(echo "$CUBES" | wc -l)
echo "Dispatching $NCUBES cubes across $(nproc) cores..."

run_cube() {
    local cube="$1"
    local name=$(basename "$cube" .cube)
    local out="$WORK/${name}.out"
    local dratdir="$WORK/${name}"
    mkdir -p "$dratdir"
    # Each cube needs its own working dir for DRAT output
    (cd "$dratdir" && timeout 120 "$SOLVER" "$SEED" "$K" 1 1 "$EDGE" "${CUBEDIR}/${cube}" > "$out" 2>&1)
    # Extract result
    if grep -q "UNSAT" "$out"; then
        echo "  $name: UNSAT"
    elif grep -q "SAT" "$out"; then
        echo "  $name: *** SAT *** (unexpected!)"
    else
        echo "  $name: TIMEOUT/ERROR"
    fi
}

export -f run_cube
export SOLVER EDGE CUBEDIR K SEED WORK

START=$(date +%s.%N)

echo "$CUBES" | xargs -P "$(nproc)" -I{} bash -c 'run_cube "{}"'

END=$(date +%s.%N)
ELAPSED=$(echo "$END - $START" | bc)

# Collect results
N_UNSAT=$(grep -rl "UNSAT" "$WORK"/*.out 2>/dev/null | wc -l)
N_SAT=$(grep -rl "SAT colouring" "$WORK"/*.out 2>/dev/null | wc -l)
N_FAIL=$((NCUBES - N_UNSAT - N_SAT))

echo ""
echo "=== Cube-and-conquer result ==="
echo "  cubes: $NCUBES"
echo "  UNSAT: $N_UNSAT"
echo "  SAT:   $N_SAT"
echo "  FAIL:  $N_FAIL"
echo "  wall-clock: ${ELAPSED}s"

if [ "$N_SAT" -gt 0 ]; then
    echo "  RESULT: SAT (formula is colourable — check cube)"
elif [ "$N_UNSAT" -eq "$NCUBES" ]; then
    echo "  RESULT: UNSAT (all cubes refuted — formula is NOT 4-colourable)"
else
    echo "  RESULT: INCOMPLETE ($N_FAIL cubes timed out)"
fi

# Total conflicts across all cubes
TOTAL_CONF=$(grep -h "conflicts=" "$WORK"/*.out 2>/dev/null | sed 's/.*conflicts=\([0-9]*\).*/\1/' | awk '{s+=$1} END{print s}')
echo "  total conflicts across all cubes: $TOTAL_CONF"
