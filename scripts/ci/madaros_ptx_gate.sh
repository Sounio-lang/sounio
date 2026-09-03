#!/usr/bin/env bash
# madaros_ptx_gate.sh — PTX codegen gate (100 unit tests)
#
# Builds a standalone PTX test binary from:
#   self-hosted/gpu/kernel_ir.sio    (GPU kernel IR + Name helpers)
#   self-hosted/gpu/ptx.sio          (PtxBuf, ptx_push_*, demo helpers)
#   self-hosted/gpu/lower_to_ptx.sio (gpu_lower_to_ptx, gap-C width fixes)
#   self-hosted/test_gpu_ptx.sio     (100 unit tests)
#
# Compiled by the lean_single bootstrap ELF, run with 256MB stack
# (GpuKernelIr ~200KB + PtxBuf 64KB by-value; deep call chains need it).
#
# Exit 0 = all 100 PASS; non-zero = failures detected.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SOUC="${SOUC_BIN:-$ROOT/bin/souc-lean-single-x86_64}"
if [[ ! -x "$SOUC" ]]; then
    echo "error: lean_single ELF not found: $SOUC" >&2
    exit 1
fi

BUNDLE=$(mktemp /tmp/ptx_gate_bundle.XXXXXX.sio)
ELF=$(mktemp /tmp/ptx_gate.XXXXXX.elf)
trap 'rm -f "$BUNDLE" "$ELF"' EXIT

# ends_with is not a lean_single builtin; inject helper before test file
cat > "$BUNDLE" << 'HELPERS_EOF'
fn ends_with(s: string, suffix: string) -> bool with Mut, Panic, Div {
    let slen = str_len(s)
    let sflen = str_len(suffix)
    if sflen > slen { return false }
    var i: i64 = 0
    while i < sflen {
        if str_char_at(s, slen - sflen + i) != str_char_at(suffix, i) { return false }
        i = i + 1
    }
    true
}
HELPERS_EOF

cat self-hosted/gpu/kernel_ir.sio \
    self-hosted/gpu/ptx.sio \
    self-hosted/gpu/lower_to_ptx.sio >> "$BUNDLE"

# PtxValueMap: extracted from ptx_advanced.sio — needed by T100
# Injected here (after GpuType is defined) to avoid pulling in the full ptx_advanced.sio
cat >> "$BUNDLE" << 'PTX_VALUE_MAP_EOF'
struct PtxValueMap {
    op_ids: [i64; 4096],
    reg_names: [i64; 4096],
    reg_lens: [i64; 4096],
    types: [i64; 4096],
    count: i64,
    name_buf: [i8; 32768],
    name_buf_len: i64,
}
fn ptx_value_map_new() -> PtxValueMap {
    PtxValueMap {
        op_ids: [0; 4096],
        reg_names: [0; 4096],
        reg_lens: [0; 4096],
        types: [0; 4096],
        count: 0,
        name_buf: [0; 32768],
        name_buf_len: 0,
    }
}
fn ptx_value_map_insert(vm: PtxValueMap, op_idx: i64, name: string, ty: GpuType) -> PtxValueMap with Mut, Panic, Div {
    var out = vm
    let c = out.count
    if c >= 4096 { return out }
    out.op_ids[c as usize] = op_idx
    let name_start = out.name_buf_len
    let nlen = str_len(name)
    var i: i64 = 0
    while i < nlen {
        if out.name_buf_len < 32768 {
            out.name_buf[out.name_buf_len as usize] = str_char_at(name, i) as i8
            out.name_buf_len = out.name_buf_len + 1
        }
        i = i + 1
    }
    out.reg_names[c as usize] = name_start
    out.reg_lens[c as usize] = nlen
    var ty_val: i64 = 0
    if ty == GpuType::GpuU32 { ty_val = 1 }
    if ty == GpuType::GpuU64 { ty_val = 2 }
    if ty == GpuType::GpuF32 { ty_val = 3 }
    if ty == GpuType::GpuF64 { ty_val = 4 }
    if ty == GpuType::GpuI32 { ty_val = 5 }
    if ty == GpuType::GpuI64 { ty_val = 6 }
    if ty == GpuType::GpuPtr { ty_val = 7 }
    out.types[c as usize] = ty_val
    out.count = c + 1
    out
}
fn ptx_value_map_get_name(vm: PtxValueMap, op_idx: i64) -> string with Mut, Panic, Div, Alloc {
    var i: i64 = 0
    while i < vm.count {
        if vm.op_ids[i as usize] == op_idx {
            let start = vm.reg_names[i as usize]
            let len = vm.reg_lens[i as usize]
            var end = start + len
            if end > vm.name_buf_len { end = vm.name_buf_len }
            if start < 0 || end <= start { return "UNKNOWN" }
            let prefix = str_from_bytes(vm.name_buf, end)
            return str_slice(prefix, start, end)
        }
        i = i + 1
    }
    "UNKNOWN"
}
PTX_VALUE_MAP_EOF

cat self-hosted/test_gpu_ptx.sio >> "$BUNDLE"

cat >> "$BUNDLE" << 'MAIN'

fn main() with IO, Mut, Panic, Div, Alloc {
    let rc = run_gpu_ptx_tests()
    if rc != 0 {
        panic("GPU PTX gate: tests failed")
    }
}
MAIN

# Compile (suppress warnings, keep errors)
if ! "$SOUC" "$BUNDLE" "$ELF" 2>&1 | grep -v "^warning:"; then
    :
fi

if [[ ! -s "$ELF" ]]; then
    echo "error: PTX gate ELF not built" >&2
    exit 1
fi

chmod +x "$ELF"

# Run with large stack (GpuKernelIr ~200KB + PtxBuf 64KB by-value chains)
(ulimit -s $((256 * 1024)) && "$ELF")
RC=$?

if [[ $RC -ne 0 ]]; then
    echo "PTX gate FAILED (exit $RC)" >&2
    exit 1
fi

echo "PTX gate PASSED: 100/100"
