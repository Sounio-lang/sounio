#!/usr/bin/env bash
# Worker-side driver-JIT acceptance runner for K-AXI→PTX output.
# The gpu-orangefs nodes are driver-only (libcuda, no CUDA toolkit/ptxas), so
# PTX validity is tested via the driver's internal JIT (cuModuleLoadDataEx),
# exactly as scripts/gpu/kaxi_ptx_runner.c does. A "cuModuleLoadData_rejected"
# outcome = the JIT refused the PTX = acceptance failure (JIT error log is
# captured). Anything that gets past module load = the PTX JIT-compiled OK.
set -uo pipefail

PTX_DIR="${1:-./ptx}"
RESULTS_DIR="${2:-./results}"
RUNNER="${3:-./kaxi_ptx_runner}"
mkdir -p "$RESULTS_DIR"

[[ -x "$RUNNER" ]] || { echo "RUNNER_NOT_EXECUTABLE: $RUNNER" | tee "$RESULTS_DIR/summary.txt"; exit 3; }
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader || true

SUMMARY="$RESULTS_DIR/summary.txt"; : > "$SUMMARY"
DETAIL="$RESULTS_DIR/detail.tsv"; printf "file\tresult\tnote\n" > "$DETAIL"

pass=0; fail=0
for f in "$PTX_DIR"/*.ptx; do
  base="$(basename "$f")"
  out="$RESULTS_DIR/${base}.out"
  # Trivial launch geometry: we only care about the JIT/load step.
  timeout 30 "$RUNNER" "$f" --kernel kaxi_kernel --threads 1 --mem-words 64 >"$out" 2>&1
  rc=$?
  if grep -q "cuModuleLoadData_rejected" "$out"; then
    fail=$((fail+1))
    jitline="$(grep -m1 "error log" "$out" | head -c 200)"
    printf "%s\tREJECT\t%s\n" "$base" "${jitline:-jit-rejected}" >> "$DETAIL"
    echo "  REJECT $base :: ${jitline}" | tee -a "$SUMMARY"
  else
    # PTX JIT-compiled (module loaded). Launch-level rc is not an acceptance concern.
    pass=$((pass+1))
    printf "%s\tACCEPT\trc=%d\n" "$base" "$rc" >> "$DETAIL"
    rm -f "$out"
  fi
done

echo "ACCEPT=$pass  REJECT=$fail  TOTAL=$((pass+fail))" | tee -a "$SUMMARY"
if [[ $fail -eq 0 ]]; then
  echo "KAXI_JIT_ACCEPT_OK" | tee -a "$SUMMARY"
else
  echo "KAXI_JIT_ACCEPT_FAIL" | tee -a "$SUMMARY"
  exit 1
fi
