#!/usr/bin/env bash
# madaros_metal_gate.sh — Metal MSL codegen gate (60 unit tests)
#
# Builds a standalone Metal test binary from:
#   self-hosted/gpu/kernel_ir.sio  (GPU kernel IR + Name helpers)
#   self-hosted/gpu/ptx.sio        (ptx_str_contains + PtxBuf)
#   self-hosted/gpu/metal.sio      (MSL emitter)
#   self-hosted/test_gpu_metal.sio (60 unit tests)
#
# The bundle is compiled by the lean_single bootstrap ELF and run with a
# 256MB stack (GpuKernelIr is ~175KB by-value; deep call chains need it).
#
# Exit 0 = all 60 PASS; non-zero = failures detected.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SOUC="${SOUC_BIN:-$ROOT/bin/souc-lean-single-x86_64}"
if [[ ! -x "$SOUC" ]]; then
    echo "error: lean_single ELF not found: $SOUC" >&2
    exit 1
fi

HELPERS=$(mktemp /tmp/metal_gate_helpers.XXXXXX.sio)
BUNDLE=$(mktemp /tmp/metal_gate_bundle.XXXXXX.sio)
ELF=$(mktemp /tmp/metal_gate.XXXXXX.elf)
trap 'rm -f "$HELPERS" "$BUNDLE" "$ELF"' EXIT

# i64_to_string + name_to_string helpers (avoid lower_to_ptx.sio bulk)
cat > "$HELPERS" << 'HELPERS_EOF'
fn i64_to_string(n: i64) -> string with Mut, Panic, Div, Alloc {
    if n < 0 { return "0" }
    if n == 0 { return "0" }
    var digits: [i8; 32] = [0; 32]
    var rev_len: i64 = 0
    var value: i64 = n
    while value > 0 {
        digits[rev_len as usize] = (48 + value % 10) as i8
        rev_len = rev_len + 1
        value = value / 10
    }
    var bytes: [i8; 32] = [0; 32]
    var i: i64 = 0
    while i < rev_len {
        bytes[i as usize] = digits[(rev_len - i - 1) as usize]
        i = i + 1
    }
    str_from_bytes(bytes, rev_len)
}
fn name_to_string(name: Name) -> string with Mut, Panic, Div, Alloc {
    str_from_bytes(name.buf, name.len)
}
HELPERS_EOF

cat self-hosted/gpu/kernel_ir.sio \
    self-hosted/gpu/ptx.sio \
    "$HELPERS" \
    self-hosted/gpu/metal.sio \
    self-hosted/test_gpu_metal.sio > "$BUNDLE"

cat >> "$BUNDLE" << 'MAIN'

fn main() with IO, Mut, Panic, Div, Alloc {
    let rc = run_gpu_metal_tests()
    if rc != 0 {
        panic("GPU Metal gate: tests failed")
    }
}
MAIN

# Compile
if ! "$SOUC" "$BUNDLE" "$ELF" 2>&1 | grep -v "^warning:"; then
    :
fi

if [[ ! -s "$ELF" ]]; then
    echo "error: Metal gate ELF not built" >&2
    exit 1
fi

chmod +x "$ELF"

# Run with large stack (GpuKernelIr is ~175KB by-value)
(ulimit -s $((256 * 1024)) && "$ELF")
RC=$?

if [[ $RC -ne 0 ]]; then
    echo "Metal gate FAILED (exit $RC)" >&2
    exit 1
fi

echo "Metal gate PASSED: 60/60"
