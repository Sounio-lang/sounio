#!/usr/bin/env bash
# Worker-side ptxas-acceptance runner for K-AXI→PTX emitter output.
# Assembles every staged PTX kernel with ptxas (no GPU execution needed) and
# records per-file acceptance. Run from the extracted payload root, where
# ./ptx/<mode>__<pattern>.ptx live.
set -uo pipefail

PTX_DIR="${1:-./ptx}"
RESULTS_DIR="${2:-./results}"
mkdir -p "$RESULTS_DIR"

# --- locate ptxas -----------------------------------------------------------
PTXAS=""
for c in ptxas /usr/local/cuda/bin/ptxas; do
  if command -v "$c" >/dev/null 2>&1; then PTXAS="$c"; break; fi
done
if [[ -z "$PTXAS" ]]; then
  for d in /usr/local/cuda-*/bin; do
    [[ -x "$d/ptxas" ]] && { PTXAS="$d/ptxas"; break; }
  done
fi
if [[ -z "$PTXAS" ]] && command -v module >/dev/null 2>&1; then
  module load cuda 2>/dev/null || true
  command -v ptxas >/dev/null 2>&1 && PTXAS="ptxas"
fi
if [[ -z "$PTXAS" ]]; then
  echo "PTXAS_NOT_FOUND" | tee "$RESULTS_DIR/summary.txt"
  exit 3
fi
echo "ptxas: $PTXAS"; "$PTXAS" --version 2>&1 | head -2 || true

# --- node compute capability (for a second, runtime-faithful pass) ----------
NODE_CC=""
if command -v nvidia-smi >/dev/null 2>&1; then
  NODE_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')"
fi
[[ -n "$NODE_CC" ]] && echo "node compute_cap -> sm_${NODE_CC}" || echo "node compute_cap: unknown"

# Architectures to assemble against: the emitter's declared floor (sm_50) and,
# if detectable, the node's real cc (what the driver JIT would target).
ARCHES=("sm_50")
[[ -n "$NODE_CC" && "$NODE_CC" != "50" ]] && ARCHES+=("sm_${NODE_CC}")

SUMMARY="$RESULTS_DIR/summary.txt"
DETAIL="$RESULTS_DIR/detail.tsv"
: > "$SUMMARY"; printf "arch\tfile\tresult\trc\n" > "$DETAIL"

overall_fail=0
for arch in "${ARCHES[@]}"; do
  pass=0; fail=0
  echo "===== ptxas -arch=$arch =====" | tee -a "$SUMMARY"
  for f in "$PTX_DIR"/*.ptx; do
    base="$(basename "$f")"
    errf="$RESULTS_DIR/${arch}__${base}.err"
    "$PTXAS" -arch="$arch" "$f" -o /dev/null 2>"$errf"
    rc=$?
    if [[ $rc -eq 0 ]]; then
      pass=$((pass+1)); printf "%s\t%s\tPASS\t0\n" "$arch" "$base" >> "$DETAIL"; rm -f "$errf"
    else
      fail=$((fail+1)); overall_fail=$((overall_fail+1))
      printf "%s\t%s\tFAIL\t%d\n" "$arch" "$base" "$rc" >> "$DETAIL"
      echo "  FAIL[$arch] $base (rc=$rc): $(head -1 "$errf")" | tee -a "$SUMMARY"
    fi
  done
  echo "arch=$arch PASS=$pass FAIL=$fail" | tee -a "$SUMMARY"
done

echo "OVERALL_FAIL=$overall_fail" | tee -a "$SUMMARY"
if [[ $overall_fail -eq 0 ]]; then
  echo "KAXI_PTXAS_ACCEPT_OK" | tee -a "$SUMMARY"
else
  echo "KAXI_PTXAS_ACCEPT_FAIL" | tee -a "$SUMMARY"
  exit 1
fi
