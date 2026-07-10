#!/usr/bin/env bash
# Worker-side NUMERIC runner for the od256 octuple kernels.
#
# Unlike run_jit.sh (which only checks that the PTX JIT-loads), this actually
# EXECUTES each kernel on real inputs and copies the output limbs back, so the
# host can compare them to the mpmath oracle (scripts/ci/od256_numeric_gate.py
# --check). od256 kernels are tid-parallel: thread t reads/writes at slot
# t*stride, so all N cases run in a single launch with --threads N.
#
# Payload layout (untarred into $PWD by the sbatch wrapper):
#   ptx/<kernel>.ptx
#   fixtures/<kernel>.in.f64        (raw little-endian doubles, packed input mem)
#   fixtures/manifest.tsv           (kernel \t ptx \t mem_words \t threads)
#   fixtures/<kernel>.truth.json    (carried through to results for the host)
#   kaxi_ptx_runner.c               (built here — dlopens libcuda, gcc-only)
#
# Results (fetched by the host): results/<kernel>.out.f64 + copied truth.json.
set -uo pipefail

PTX_DIR="${1:-./ptx}"
FIX_DIR="${2:-./fixtures}"
RESULTS_DIR="${3:-./results}"
RUNNER_SRC="${4:-./kaxi_ptx_runner.c}"
mkdir -p "$RESULTS_DIR"

SUMMARY="$RESULTS_DIR/summary.txt"; : > "$SUMMARY"
command -v nvidia-smi >/dev/null 2>&1 && \
  nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | tee -a "$SUMMARY" || true

# Runner: prefer a prebuilt binary shipped in the payload (gpu-orangefs nodes are
# driver-only — no gcc). Fall back to building from source if a compiler exists.
RUNNER=./kaxi_ptx_runner
if [[ -x "$RUNNER" ]]; then
  chmod +x "$RUNNER"; echo "  using prebuilt runner" | tee -a "$SUMMARY"
elif command -v gcc >/dev/null 2>&1 && gcc -O2 -o "$RUNNER" "$RUNNER_SRC" -ldl -lm 2>"$RESULTS_DIR/build.log"; then
  echo "  built runner from source" | tee -a "$SUMMARY"
else
  echo "RUNNER_UNAVAILABLE (no prebuilt binary, no gcc)" | tee -a "$SUMMARY"
  [[ -f "$RESULTS_DIR/build.log" ]] && sed 's/^/  /' "$RESULTS_DIR/build.log"; exit 3
fi

# The kernels index by %tid.x only (single-block, no blockIdx), so all cases must
# fit in ONE block. Register pressure caps that: od256_mul uses ~336 32-bit regs
# per thread (%fd<168>), and 512 threads busts the L4's 64K-reg file → launch
# OUT_OF_RESOURCES. So we chunk any kernel with threads > MAX_THREADS into
# sequential launches, slicing the packed input per chunk and concatenating the
# dumped outputs (case t always lands at byte t*stride*8, so cat preserves order).
MAX_THREADS="${MAX_THREADS:-128}"

ok=0; bad=0
while IFS=$'\t' read -r kernel ptx mem_words threads; do
  [[ -z "$kernel" ]] && continue
  pfile="$PTX_DIR/$ptx"
  infile="$FIX_DIR/${kernel}.in.f64"
  outfile="$RESULTS_DIR/${kernel}.out.f64"
  [[ -f "$pfile" && -f "$infile" ]] || { echo "MISSING $kernel (ptx/in)" | tee -a "$SUMMARY"; bad=$((bad+1)); continue; }
  cp -f "$FIX_DIR/${kernel}.truth.json" "$RESULTS_DIR/" 2>/dev/null || true
  stride=$(( mem_words / threads ))            # words per case
  rm -f "$outfile"; kok=1; nchunk=0
  off=0
  while [[ $off -lt $threads ]]; do
    chunk=$(( threads - off )); [[ $chunk -gt $MAX_THREADS ]] && chunk=$MAX_THREADS
    cwords=$(( chunk * stride ))
    cin="$RESULTS_DIR/.${kernel}.chunk.in"; cout="$RESULTS_DIR/.${kernel}.chunk.out"
    # slice cases [off, off+chunk): bs = one case (stride*8 bytes)
    dd if="$infile" of="$cin" bs=$(( stride * 8 )) skip=$off count=$chunk status=none
    timeout 120 "$RUNNER" "$pfile" --kernel kaxi_kernel --type f64 \
        --threads "$chunk" --mem-words "$cwords" \
        --init-file "$cin" --dump-file "$cout" >>"$RESULTS_DIR/${kernel}.log" 2>&1
    rc=$?
    if [[ $rc -ne 0 || ! -s "$cout" ]]; then kok=0; break; fi
    cat "$cout" >> "$outfile"
    off=$(( off + chunk )); nchunk=$(( nchunk + 1 ))
  done
  rm -f "$RESULTS_DIR/.${kernel}.chunk.in" "$RESULTS_DIR/.${kernel}.chunk.out"
  if [[ $kok -eq 1 && -s "$outfile" ]]; then
    ok=$((ok+1));  echo "  RAN    $kernel (threads=$threads words=$mem_words, ${nchunk} chunk(s) x<=$MAX_THREADS)" | tee -a "$SUMMARY"
  else
    bad=$((bad+1)); echo "  FAILED $kernel rc=$rc" | tee -a "$SUMMARY"; sed 's/^/    /' "$RESULTS_DIR/${kernel}.log" | tail -8
  fi
done < "$FIX_DIR/manifest.tsv"

echo "RAN=$ok FAILED=$bad" | tee -a "$SUMMARY"
if [[ $bad -eq 0 ]]; then echo "OD256_NUMERIC_RUN_OK" | tee -a "$SUMMARY"; else echo "OD256_NUMERIC_RUN_FAIL" | tee -a "$SUMMARY"; exit 1; fi
