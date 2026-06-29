#!/usr/bin/env bash
# Gate for public GPU backend CFG lowering through `souc build --backend gpu`.
#
# This intentionally checks structural PTX emission only. It does not claim
# CUDA/PTX assembler validation or full kernel semantic correctness.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC="${SOUC:-./bin/souc}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

PASS=0
FAIL=0

pass() {
    PASS=$((PASS + 1))
    echo "PASS  $1"
}

fail() {
    FAIL=$((FAIL + 1))
    echo "FAIL  $1: $2"
}

require_grep() {
    local label="$1"
    local pattern="$2"
    local file="$3"
    if grep -q "$pattern" "$file"; then
        pass "$label"
    else
        fail "$label" "missing pattern $pattern in $file"
    fi
}

build_gpu() {
    local label="$1"
    local src="$2"
    local out="$3"
    local log="$TMP_DIR/${label}.log"
    local rc=0

    timeout 30 "$SOUC" build "$src" --backend gpu -o "$out" >"$log" 2>&1 || rc=$?
    if [ "$rc" -eq 124 ]; then
        fail "$label" "timeout"
        return
    fi
    if [ "$rc" -ne 0 ]; then
        fail "$label" "exit $rc: $(tail -5 "$log" | tr '\n' ' ')"
        return
    fi
    if [ ! -s "$out" ]; then
        fail "$label" "empty PTX output"
        return
    fi
    pass "$label"
}

if [ ! -x "$SOUC" ]; then
    echo "FATAL: souc binary not executable: $SOUC"
    exit 1
fi

cat > "$TMP_DIR/cfg_if_kernel.sio" <<'EOF'
kernel fn k(n: i64) {
  let tid = gpu_thread_id_x()
  if tid < n {
    let next = tid + 1
  }
}
EOF

echo "=== Public GPU CFG Build Gate ==="
echo "SOUC:               $SOUC"
echo "SOUNIO_STDLIB_PATH: $SOUNIO_STDLIB_PATH"
echo ""

MIN_PTX="$TMP_DIR/cfg_if_kernel.ptx"
build_gpu "cfg_if_kernel_builds" "$TMP_DIR/cfg_if_kernel.sio" "$MIN_PTX"
require_grep "cfg_if_kernel_entry" "\\.visible \\.entry k" "$MIN_PTX"
require_grep "cfg_if_kernel_labels" "LBB0_" "$MIN_PTX"
require_grep "cfg_if_kernel_branches" "bra LBB0_" "$MIN_PTX"
require_grep "cfg_if_kernel_ret" "ret;" "$MIN_PTX"

E2E_PTX="$TMP_DIR/gpu_vec_add_e2e.ptx"
build_gpu "gpu_vec_add_e2e_builds" "tests/run-pass/gpu_vec_add_e2e.sio" "$E2E_PTX"
require_grep "gpu_vec_add_e2e_entry" "\\.visible \\.entry vec_add" "$E2E_PTX"
require_grep "gpu_vec_add_e2e_labels" "LBB0_" "$E2E_PTX"
require_grep "gpu_vec_add_e2e_branches" "bra LBB0_" "$E2E_PTX"

echo ""
echo "Summary: pass=$PASS fail=$FAIL"

if [ "$FAIL" -ne 0 ]; then
    exit 1
fi
