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

require_egrep() {
    local label="$1"
    local pattern="$2"
    local file="$3"
    if grep -Eq "$pattern" "$file"; then
        pass "$label"
    else
        fail "$label" "missing extended pattern $pattern in $file"
    fi
}

reject_egrep() {
    local label="$1"
    local pattern="$2"
    local file="$3"
    if grep -Eq "$pattern" "$file"; then
        fail "$label" "unexpected extended pattern $pattern in $file"
    else
        pass "$label"
    fi
}

require_egrep_count() {
    local label="$1"
    local pattern="$2"
    local expected="$3"
    local file="$4"
    local count
    count="$(grep -Ec "$pattern" "$file" || true)"
    if [ "$count" -eq "$expected" ]; then
        pass "$label"
    else
        fail "$label" "expected $expected matches for $pattern in $file, got $count"
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
require_egrep "gpu_vec_add_e2e_declares_pred_regs" "\\.reg \\.pred %p<[0-9]+>;" "$E2E_PTX"
require_egrep "gpu_vec_add_e2e_declares_u64_regs" "\\.reg \\.b64 %rd<[0-9]+>;" "$E2E_PTX"
require_egrep "gpu_vec_add_e2e_loads_n_param" "ld\\.param\\.u64 %rd0, \\[n\\];" "$E2E_PTX"
require_egrep "gpu_vec_add_e2e_loads_a_param" "ld\\.param\\.u64 %rd1, \\[a\\];" "$E2E_PTX"
require_egrep "gpu_vec_add_e2e_loads_b_param" "ld\\.param\\.u64 %rd2, \\[b\\];" "$E2E_PTX"
require_egrep "gpu_vec_add_e2e_loads_c_param" "ld\\.param\\.u64 %rd3, \\[c\\];" "$E2E_PTX"
require_egrep "gpu_vec_add_e2e_compares_tid_to_n" "setp\\.lt\\.s64 %p[0-9]+, %rd[0-9]+, %rd0;" "$E2E_PTX"
reject_egrep "gpu_vec_add_e2e_rejects_degenerate_compare" "setp\\.[a-z]+\\.[a-z0-9]+ %p[0-9]+, %(r|rd)0, %(r|rd)0;" "$E2E_PTX"
require_egrep "gpu_vec_add_e2e_scales_index" "mul\\.lo\\.u64 %rd[0-9]+, %rd[0-9]+, 8;" "$E2E_PTX"
require_egrep "gpu_vec_add_e2e_addresses_a" "add\\.u64 %rd[0-9]+, %rd1, %rd[0-9]+;" "$E2E_PTX"
require_egrep "gpu_vec_add_e2e_addresses_b" "add\\.u64 %rd[0-9]+, %rd2, %rd[0-9]+;" "$E2E_PTX"
require_egrep "gpu_vec_add_e2e_addresses_c" "add\\.u64 %rd[0-9]+, %rd3, %rd[0-9]+;" "$E2E_PTX"
require_egrep "gpu_vec_add_e2e_declares_f64_regs" "\\.reg \\.f64 %fd<[0-9]+>;" "$E2E_PTX"
require_egrep "gpu_vec_add_e2e_loads_f64_inputs" "ld\\.global\\.f64 %fd[0-9]+, \\[%rd[0-9]+\\];" "$E2E_PTX"
require_egrep "gpu_vec_add_e2e_adds_f64_inputs" "add\\.f64 %fd[0-9]+, %fd[0-9]+, %fd[0-9]+;" "$E2E_PTX"
require_egrep "gpu_vec_add_e2e_stores_f64_output" "st\\.global\\.f64 \\[%rd[0-9]+\\], %fd[0-9]+;" "$E2E_PTX"
reject_egrep "gpu_vec_add_e2e_rejects_s64_value_loads" "ld\\.global\\.s64 %rd[0-9]+, \\[%rd[0-9]+\\];" "$E2E_PTX"
reject_egrep "gpu_vec_add_e2e_rejects_s64_value_stores" "st\\.global\\.s64 \\[%rd[0-9]+\\], %rd[0-9]+;" "$E2E_PTX"

SLICES_PTX="$TMP_DIR/gpu_launch_vec_slices.ptx"
build_gpu "gpu_launch_vec_slices_builds" "tests/run-pass/gpu_launch_vec_slices.sio" "$SLICES_PTX"
require_egrep_count "gpu_launch_vec_slices_single_ptx_version" "^\\.version " 1 "$SLICES_PTX"
require_egrep_count "gpu_launch_vec_slices_single_ptx_target" "^\\.target " 1 "$SLICES_PTX"
require_egrep_count "gpu_launch_vec_slices_single_ptx_address_size" "^\\.address_size " 1 "$SLICES_PTX"
require_egrep "gpu_launch_vec_slices_loads_f64_scalar_params" "ld\\.param\\.f64 %fd[0-9]+, \\[(factor|bias)\\];" "$SLICES_PTX"
reject_egrep "gpu_launch_vec_slices_rejects_u64_load_to_f64_param" "ld\\.param\\.u64 %fd[0-9]+, \\[(factor|bias)\\];" "$SLICES_PTX"
require_egrep "gpu_launch_vec_slices_multiplies_f64" "mul\\.rn\\.f64 %fd[0-9]+, %fd[0-9]+, %fd[0-9]+;" "$SLICES_PTX"
require_egrep "gpu_launch_vec_slices_divides_f64" "div\\.rn\\.f64 %fd[0-9]+, %fd[0-9]+, %fd[0-9]+;" "$SLICES_PTX"

echo ""
echo "Summary: pass=$PASS fail=$FAIL"

if [ "$FAIL" -ne 0 ]; then
    exit 1
fi
