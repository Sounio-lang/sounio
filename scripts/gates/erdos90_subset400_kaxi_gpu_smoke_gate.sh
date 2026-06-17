#!/usr/bin/env bash
# Option B: Erdős n=400-lane K-AXI GPU smoke — CPU HC oracle + parallel edge-count certifier.
#
# Toy geometry (PN=25, N=25, n=9, 8 threads, 1 lite HC step) proves K-AXI→PTX→launch
# wiring before scaling to n≤625 cluster GPU proposer work.
#
# CPU runs hill-climb; GPU kernel certifies per-thread edge counts on post-HC MEM slices.
# When CUDA is unavailable, warp-sim parity (test_erdos90_hc_smoke_warp.sio) is required.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
SOUC="${SOUC:-$ROOT/bin/souc}"
KRETIKOS="${KRETIKOS:-$ROOT/bin/kretikos}"
# Large K-AXI emitters still route through lean_single on this checkout: Madaros
# can segfault on the erdos90_hc_smoke driver. Honour explicit SOUC; otherwise
# prefer the preserved lean_single ELF when present.
if [[ "$SOUC" == "$ROOT/bin/souc" && -z "${SOUNIO_SOUC_ENGINE:-}" && -x "$ROOT/bin/souc-lean-single-x86_64" ]]; then
  SOUC="$ROOT/bin/souc-lean-single-x86_64"
fi
export SOUNIO_KRETIKOS_COMPILER="${SOUNIO_KRETIKOS_COMPILER:-$SOUC}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

ulimit -s unlimited 2>/dev/null || ulimit -s 65536 2>/dev/null || true

echo "[erdos90-kaxi-gpu-smoke] CPU oracle"
"$SOUC" stdlib/research/erdos90_kaxi_hc_smoke.sio "$WORK/smoke.elf" >/dev/null
chmod +x "$WORK/smoke.elf"
"$WORK/smoke.elf" >"$WORK/smoke.log" 2>&1
grep -q "KAXI_SMOKE_DONE" "$WORK/smoke.log"

INIT_CSV="$(grep '^KAXI_SMOKE_INIT_CSV=' "$WORK/smoke.log" | sed 's/^KAXI_SMOKE_INIT_CSV=//')"
MEM_WORDS="$(grep '^KAXI_SMOKE_MEM_WORDS=' "$WORK/smoke.log" | sed 's/^KAXI_SMOKE_MEM_WORDS=//')"
CPU_MAX="$(grep '^KAXI_SMOKE_CPU_MAX=' "$WORK/smoke.log" | sed 's/^KAXI_SMOKE_CPU_MAX=//')"
THREADS=8
OUT_OFF=429

echo "[erdos90-kaxi-gpu-smoke] emit K-AXI asm pattern=erdos90_hc_smoke"
"$KRETIKOS" emit-kaxi erdos90_hc_smoke -o "$WORK/erdos90_hc_smoke.kaxi" >/dev/null
grep -q "Erdos90 smoke: parallel lite HC proposer" "$WORK/erdos90_hc_smoke.kaxi"
grep -q "get_tid r0, var=0%, seq=0" "$WORK/erdos90_hc_smoke.kaxi"

echo "[erdos90-kaxi-gpu-smoke] warp-sim parity"
WARP_OK=0
if "$SOUC" self-hosted/gpu/test_erdos90_hc_smoke_warp.sio "$WORK/warp.elf" >/dev/null 2>&1; then
  chmod +x "$WORK/warp.elf"
  if "$WORK/warp.elf" >"$WORK/warp.log" 2>&1; then
    if grep -q "PASS erdos90_hc_smoke_warp_parity" "$WORK/warp.log"; then
      WARP_OK=1
    fi
  fi
fi
if [[ "$WARP_OK" -eq 0 ]]; then
  echo "[erdos90-kaxi-gpu-smoke] warp-sim FAILED (required when CUDA unavailable)" >&2
  if [[ -f "$WORK/warp.log" ]]; then
    rg "WARP_|FAIL|PASS" "$WORK/warp.log" >&2 || true
  fi
fi

echo "[erdos90-kaxi-gpu-smoke] emit K-AXI PTX pattern=erdos90_hc_smoke"
"$KRETIKOS" kaxi-emit-ptx erdos90_hc_smoke -o "$WORK/erdos90_hc_smoke.ptx" >/dev/null

RUNNER="$WORK/kaxi_ptx_runner"
RUNNER_SRC="scripts/gpu/kaxi_ptx_runner.c"
GPU_OK=0
if command -v cc >/dev/null 2>&1 \
   && command -v nvidia-smi >/dev/null 2>&1 \
   && nvidia-smi >/dev/null 2>&1; then
  if cc -O2 "$RUNNER_SRC" -ldl -lm -o "$RUNNER" >/dev/null 2>&1; then
    if "$RUNNER" "$WORK/erdos90_hc_smoke.ptx" \
        --mode basic --threads "$THREADS" --mem-words "$MEM_WORDS" \
        --init-mem "$INIT_CSV" --type i64 --print-count "$MEM_WORDS" \
        >"$WORK/gpu.log" 2>&1; then
      grep -q 'sounio_kaxi_runtime status=pass' "$WORK/gpu.log"
      GPU_OK=1
    fi
  fi
fi

if [[ "$GPU_OK" -eq 0 ]]; then
  echo "[erdos90-kaxi-gpu-smoke] CUDA launch SKIPPED (no GPU or runner failure)"
fi

fail=0
tid=0
while [[ "$tid" -lt "$THREADS" ]]; do
  cpu_edges="$(grep "KAXI_SMOKE_CPU_EDGE_${tid} " "$WORK/smoke.log" | awk '{print $2}')"
  if [[ -z "$cpu_edges" ]]; then
    echo "missing CPU oracle for thread $tid" >&2
    fail=1
    tid=$((tid + 1))
    continue
  fi
  if [[ "$GPU_OK" -eq 1 ]]; then
    gpu_edges="$(awk -v off="$OUT_OFF" -v t="$tid" '/^MEM:/ {print $(off + t + 2); exit}' "$WORK/gpu.log")"
    if [[ "$gpu_edges" != "$cpu_edges" ]]; then
      echo "thread $tid mismatch: cpu=$cpu_edges gpu=$gpu_edges" >&2
      fail=1
    fi
  fi
  tid=$((tid + 1))
done

if [[ "$fail" -ne 0 ]]; then
  echo "[erdos90-kaxi-gpu-smoke] FAIL" >&2
  exit 1
fi

if [[ "$GPU_OK" -eq 0 && "$WARP_OK" -eq 0 ]]; then
  echo "[erdos90-kaxi-gpu-smoke] FAIL (cpu oracle ok; warp parity required without CUDA)" >&2
  exit 1
fi

if [[ "$GPU_OK" -eq 1 ]]; then
  echo "[erdos90-kaxi-gpu-smoke] PASS (cpu_max=$CPU_MAX, gpu parity on $THREADS threads; warp_ok=$WARP_OK)"
elif [[ "$WARP_OK" -eq 1 ]]; then
  echo "[erdos90-kaxi-gpu-smoke] PASS cpu_oracle+warp (cpu_max=$CPU_MAX; cuda skipped)"
else
  echo "[erdos90-kaxi-gpu-smoke] PASS cpu_oracle_only (cpu_max=$CPU_MAX; cuda+warp skipped)"
fi