#!/usr/bin/env bash
# SLURM-routed cube-and-conquer + Thompson seed ensemble.
set -euo pipefail

QUEEN_N="${1:?queen_n}"
K="${2:?k}"
CUBE_DIR="${3:?cube_dir}"
N_SEEDS="${4:-32}"
TIMEOUT="${5:-60}"
PART="${6:-cpu-ops}"

ELF="/tmp/souc_sat_ts.elf"
EDGE="/tmp/bench/queen_${QUEEN_N}.edge"

NCUBES=$(wc -l < "${CUBE_DIR}/manifest.txt")
echo "[1/3] queen_${QUEEN_N} k=${K}: ${NCUBES} cubes × ${N_SEEDS} seeds × ${TIMEOUT}s → ${PART}"

# Build payload: ELF + edge + all cubes
PAYLOAD=$(mktemp /tmp/qpayload_XXXX.tar.gz)
tar czf "$PAYLOAD" \
  -C /tmp souc_sat_ts.elf \
  -C /tmp/bench "queen_${QUEEN_N}.edge" \
  -C "$CUBE_DIR" manifest.txt $(cat "${CUBE_DIR}/manifest.txt" | tr '\n' ' ')
echo "  payload: $(stat -c%s "$PAYLOAD") bytes"

# Ship payload via stdin, decode + run on node
echo "[2/3] submitting to ${PART}..."
base64 -w0 "$PAYLOAD" | srun --partition="$PART" --time=04:00:00 \
  --job-name="q${QUEEN_N}k${K}" bash -c '
    set +e
    WORK="/orangefs/training/assault_$$"
    mkdir -p "$WORK"
    cd "$WORK"
    base64 -d | tar xzf -
    chmod +x souc_sat_ts.elf
    EDGE=$(ls queen_*.edge | head -1)
    echo "=== $(hostname) cores=$(nproc) ==="
    echo "=== '"${N_SEEDS}"' seeds × '"${TIMEOUT}"'s per cube ==="
    
    K='"${K}"'
    TMO='"${TIMEOUT}"'
    NSEEDS='"${N_SEEDS}"'
    SOLVED=0
    TOTAL=0
    
    cat manifest.txt | while read cube; do
      for seed in $(seq 1 "$NSEEDS"); do
        echo "$cube $seed"
      done
    done | xargs -P "$(nproc)" -I{} bash -c "
      read cube seed <<< \"{}\"
      d=\"r_\${seed}_\${cube}\"
      mkdir -p \"\$d\" 2>/dev/null
      ( cd \"\$d\" && timeout \"\$1\" \"../souc_sat_ts.elf\" \"\$seed\" \"\$2\" 3 1 \"../\$3\" \"../\$cube\" 2>&1 | grep -q UNSAT && echo \"CRACKED \$cube seed=\$seed\" )
    " _ "$TMO" "$K" "$EDGE"
    
    echo "=== DONE ==="
    cd / && rm -rf "$WORK"
  '

echo "[3/3] done."
rm -f "$PAYLOAD"
