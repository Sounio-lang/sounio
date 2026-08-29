#!/usr/bin/env bash
# scripts/ci/kretikos_associator_emit_gate.sh
#
# Local emission gate for the octonion / sedenion associator K-AXI kernels.
# Validates that:
#   - the emitter compiles + runs and produces a non-trivial K-AXI assembly
#   - the K-AXI → PTX lowering succeeds in f32 mode (the mode used on L4)
#   - the PTX has the expected structural shape (entry, register banks, fp ops,
#     store_global, ret) so an L4 submission won't fail at parse time
#
# Real bit-equality validation against the CPU oracle happens via
# slurm-jobs/kretikos/submit-kaxi-ptx-matrix.sh (the case is wired there)
# and the umbrella `kretikos_kaxi_l4_launch_gate.sh`.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_KRETIKOS_ASSOCIATOR_GATE_DIR:-$(mktemp -d /tmp/kretikos-associator-emit.XXXXXX)}"
mkdir -p "$OUT_DIR"

echo "kretikos_associator_emit_gate: shell-syntax"
bash -n "$0"
bash -n bin/kretikos

emit_and_check() {
  local pattern="$1"
  local kaxi_out="$OUT_DIR/${pattern}.kaxi"
  local ptx_out="$OUT_DIR/${pattern}.f32.ptx"

  echo "kretikos_associator_emit_gate: emit-kaxi $pattern"
  ./bin/kretikos emit-kaxi "$pattern" -o "$kaxi_out" >/dev/null
  local kaxi_lines
  kaxi_lines=$(wc -l <"$kaxi_out")
  if [[ "$kaxi_lines" -lt 100 ]]; then
    echo "FAIL: $pattern K-AXI is suspiciously short ($kaxi_lines lines)" >&2
    exit 1
  fi
  echo "  K-AXI lines: $kaxi_lines"

  echo "kretikos_associator_emit_gate: kaxi-emit-ptx $pattern --f32"
  ./bin/kretikos kaxi-emit-ptx "$pattern" --f32 -o "$ptx_out" >/dev/null
  local ptx_bytes
  ptx_bytes=$(wc -c <"$ptx_out")
  if [[ "$ptx_bytes" -lt 1000 ]]; then
    echo "FAIL: $pattern PTX is suspiciously small ($ptx_bytes bytes)" >&2
    exit 1
  fi
  echo "  PTX bytes:   $ptx_bytes"

  for needed in '.visible .entry kaxi_kernel' '.reg .f32 %f<128>' '.reg .b64 %rd<128>' 'mul.rn.f32' 'sub.rn.f32' 'add.rn.f32' 'st.global' 'ret;'; do
    if ! grep -qF "$needed" "$ptx_out"; then
      echo "FAIL: $pattern PTX missing expected token: $needed" >&2
      exit 1
    fi
  done
  echo "  PTX shape:   ok (entry + reg banks + f32 add/sub/mul + st.global + ret)"
}

emit_and_check "octonion_associator"

# When sedenion_associator is wired (Phase B), append it here.
if ./bin/kretikos emit-kaxi sedenion_associator -o "$OUT_DIR/sedenion_probe.kaxi" 2>/dev/null; then
  emit_and_check "sedenion_associator"
fi

echo "kretikos_associator_emit_gate: PASS"
