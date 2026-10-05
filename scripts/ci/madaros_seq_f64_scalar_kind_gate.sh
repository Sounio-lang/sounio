#!/usr/bin/env bash
# Seq<f64> element reads classify as float in every spelling.
#
# Guards BLK-20260904-seq-f64-element-scalar-kind. Seq access is an intrinsic
# with no fn symbol, so both expression classifiers missed on the mangled-method
# lookup and defaulted to integer: `w.get(0) + w.get(1)` produced 2^63 and
# println refused with "unresolved scalar kind". Wrong code, not a rejection.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# Always pin this worktree's stdlib (never inherit a foreign SOUNIO_STDLIB_PATH).
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
unset SOUNIO_SOUC_ENGINE || true
SOUC="${SOUC:-$ROOT/bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT

echo "== madaros_seq_f64_scalar_kind_gate =="

run_fixture() {
  local src="$1" sentinel="$2"
  local elf="$OUT/$(basename "$src" .sio).elf"
  local log="$OUT/$(basename "$src" .sio).log"
  if ! "$SOUC" compile "$src" -o "$elf" >"$log" 2>&1; then
    echo "FAIL: compile $src"
    tail -40 "$log" || true
    exit 1
  fi
  chmod +x "$elf"
  if ! "$elf" >"$log" 2>&1; then
    echo "FAIL: run $src"
    cat "$log" || true
    exit 1
  fi
  grep -q "$sentinel" "$log" || {
    echo "FAIL: $src missing sentinel $sentinel"
    cat "$log" || true
    exit 1
  }
}

run_fixture tests/run-pass/seq_f64_element_scalar_kind.sio SEQ_F64_SCALAR_KIND_OK
# The stdlib consumer: graph::sinkhorn carries Seq<f64> measures and matrices,
# and was migrated onto them. It reads them through &ProbMeasure / &CostMatrix
# receivers, which is the shape this fix covers.
run_fixture tests/stdlib/graph/test_sinkhorn_e2e.sio GRAPH_SINKHORN_OK

echo "MADAROS_SEQ_F64_SCALAR_KIND_GATE_OK"
