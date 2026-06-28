#!/usr/bin/env bash
# slurm-jobs/queen-sat-assault/submit.sh
#
# Cube-and-conquer + Thompson seed ensemble on queen graph colouring,
# routed through SLURM to avoid pod CPU saturation.
#
# Ships the solver ELF + all cube/edge files to the cluster node via
# base64 over srun stdin (nodes don't mount /workspace).
#
# Usage: bash submit.sh <queen_n> <k> <cube_dir> <seeds> <timeout>
# Example: bash submit.sh 13 11 /tmp/cubes_q13k11 32 60
set -euo pipefail

QUEEN_N="${1:?missing queen_n}"
K="${2:?missing k}"
CUBE_DIR="${3:?missing cube_dir}"
N_SEEDS="${4:-32}"
TIMEOUT="${5:-60}"
PART="${6:-cpu-ops}"
ELF="/tmp/souc_sat_ts.elf"
EDGE="/tmp/bench/queen_${QUEEN_N}.edge"

[[ -x "$ELF" ]] || { echo "ERROR: solver ELF not found at $ELF"; exit 1; }
[[ -f "$EDGE" ]] || { echo "ERROR: edge file not found at $EDGE"; exit 1; }
[[ -d "$CUBE_DIR" ]] || { echo "ERROR: cube dir not found at $CUBE_DIR"; exit 1; }

NCUBES=$(wc -l < "${CUBE_DIR}/manifest.txt")
echo "[assault] queen_${QUEEN_N} k=${K}: ${NCUBES} cubes × ${N_SEEDS} seeds × ${TIMEOUT}s on ${PART}"

# Build a self-contained payload: ELF + edge + all cubes as a tarball, base64-encoded
PAYLOAD=$(mktemp /tmp/payload_XXXX.tar.gz)
tar czf "$PAYLOAD" -C / "$(echo $ELF | sed 's|^/||')" 2>/dev/null || true
# Simpler: just tar the specific files
tar czf "$PAYLOAD" \
  -C "$(dirname $ELF)" "$(basename $ELF)" \
  -C "$(dirname $EDGE)" "$(basename $EDGE)" \
  -C "$CUBE_DIR" manifest.txt \
  $(cat "${CUBE_DIR}/manifest.txt" | sed "s|^|${CUBE_DIR}/|" | tr '\n' ' ')

PAYLOAD_SIZE=$(stat -c%s "$PAYLOAD")
echo "[assault] payload: ${PAYLOAD_SIZE} bytes ($(echo "$PAYLOAD_SIZE / 1048576" | bc)MB)"

# Ship to node and run
echo "[assault] submitting to ${PART}..."
base64 -w0 "$PAYLOAD" | srun --partition="$PART" --time=04:00:00 --job-name="queen${QUEEN_N}_k${K}" bash -c '
  set -e
  WORK=/orangefs/training/queen_assult_$$
  mkdir -p "$WORK"
  base64 -d > "$WORK/payload.tar.gz"
  tar xzf "$WORK/payload.tar.gz" -C "$WORK"
  ELF="$WORK/souc_sat_ts.elf"
  chmod +x "$ELF"
  EDGE_FILE=$(ls "$WORK"/queen_*.edge | head -1)
  CUBES="$WORK/manifest.txt"

  echo "=== node $(hostname) cores=$(nproc) ==="
  echo "=== queen assault: ${'"${NCUBES}"'} cubes × ${'"${N_SEEDS}"'} seeds ==="

  NCUBES=$(wc -l < "$CUBES")
  SOLVED_FILE="$WORK/solved.txt"
  > "$SOLVED_FILE"

  run_cube() {
    local cube="$1"
    local seed="$2"
    local name=$(basename "$cube" .cube)
    # Skip if already solved
    grep -q "^${cube}$" "$SOLVED_FILE" 2>/dev/null && return 0
    local wd="$WORK/run_${seed}_${name}"
    mkdir -p "$wd"
    (
      cd "$wd"
      result=$(timeout '"${TIMEOUT}"' "$ELF" "$seed" "'"$K"'" 3 1 "$EDGE_FILE" "$WORK/$cube" 2>&1)
      if echo "$result" | grep -q "UNSAT"; then
        echo "$cube" >> "$SOLVED_FILE"
        echo "CRACKED $cube seed=$seed"
      fi
    )
  }
  export -f run_cube
  export ELF EDGE_FILE SOLVED_FILE WORK K
  export -f run_cube

  # Generate (cube, seed) pairs and run in parallel
  cat "$CUBES" | while read cube; do
    for seed in $(seq 1 '"${N_SEEDS}"'); do
      echo "$cube $seed"
    done
  done | xargs -P '"$(nproc)"' -I{} bash -c "run_cube {}"

  SOLVED_COUNT=$(sort -u "$SOLVED_FILE" | wc -l)
  echo ""
  echo "=== RESULT: ${SOLVED_COUNT}/${NCUBES} cubes cracked ==="
  rm -rf "$WORK"
'

echo "[assault] done."
rm -f "$PAYLOAD"
