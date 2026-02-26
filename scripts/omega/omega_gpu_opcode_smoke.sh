#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PTX_PATH="$ROOT_DIR/self-hosted/gpu/ptx.sio"
KERNEL_IR_PATH="$ROOT_DIR/self-hosted/gpu/kernel_ir.sio"
LOWER_TO_PTX_PATH="$ROOT_DIR/self-hosted/gpu/lower_to_ptx.sio"
PTX_ADVANCED_PATH="$ROOT_DIR/self-hosted/gpu/ptx_advanced.sio"
METAL_PATH="$ROOT_DIR/self-hosted/gpu/metal.sio"
SMOKE_LOG="${OMEGA_GPU_OPCODE_SMOKE_LOG:-$ROOT_DIR/artifacts/omega/gpu_opcode_smoke.log}"

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

mkdir -p "$(dirname "$SMOKE_LOG")"
: >"$SMOKE_LOG"

require_file() {
  if [ ! -f "$1" ]; then
    echo "error: missing required file: $1" >&2
    exit 2
  fi
}

trim_before_main() {
  awk '
    /^fn main\(/ { exit }
    { print }
  ' "$1"
}

emit_name_stub() {
  cat <<'SIO'
struct Name {
    buf: [i8; 128],
    len: i64,
}

fn ir_empty_name() -> Name {
    Name { buf: [0; 128], len: 0 }
}

fn ir_name_from_bytes(a: i8, b: i8, c: i8, d: i8, len: i64) -> Name with Mut, Panic, Div {
    var n = Name { buf: [0; 128], len: len }
    if len > 0 { n.buf[0] = a }
    if len > 1 { n.buf[1] = b }
    if len > 2 { n.buf[2] = c }
    if len > 3 { n.buf[3] = d }
    n
}
SIO
}

for required in \
  "$PTX_PATH" \
  "$KERNEL_IR_PATH" \
  "$LOWER_TO_PTX_PATH" \
  "$PTX_ADVANCED_PATH" \
  "$METAL_PATH"; do
  require_file "$required"
done

sounio_require_souc

tmp_ptx="$(mktemp "${TMPDIR:-/tmp}/omega_gpu_opcode_ptx.XXXXXX.sio")"
tmp_metal="$(mktemp "${TMPDIR:-/tmp}/omega_gpu_opcode_metal.XXXXXX.sio")"
cleanup() {
  rm -f "$tmp_ptx" "$tmp_metal"
}
trap cleanup EXIT

{
  emit_name_stub
  cat "$KERNEL_IR_PATH"
  cat "$PTX_PATH"
  cat "$LOWER_TO_PTX_PATH"
  trim_before_main "$PTX_ADVANCED_PATH"
  cat <<'SIO'

fn omega_ptx_opcode_pattern_smoke() -> bool with Mut, Panic, Div, Alloc {
    var b = ptx_buf_new()
    b = ptx_emit_shfl_up(b, "%r0", "%r1", "1", 31)
    b = ptx_emit_shfl_bfly(b, "%r2", "%r3", "2", 31)
    b = ptx_emit_vote_ballot(b, "%r4", "%p0")
    b = ptx_emit_sqrt(b, "%f0", "%f1", GpuType::GpuF32)
    b = ptx_emit_rsqrt(b, "%f2", "%f3", GpuType::GpuF32)
    b = ptx_emit_exp2(b, "%f4", "%f5", GpuType::GpuF32)
    b = ptx_emit_lg2(b, "%f6", "%f7", GpuType::GpuF32)
    b = ptx_emit_sin(b, "%f8", "%f9", GpuType::GpuF32)
    b = ptx_emit_cos(b, "%f10", "%f11", GpuType::GpuF32)
    b = ptx_emit_abs(b, "%f12", "%f13", GpuType::GpuF32)
    b = ptx_emit_rcp(b, "%f14", "%f15", GpuType::GpuF32)
    b = ptx_emit_setp_gt(b, "%p1", "%f0", "%f1", GpuType::GpuF32)
    b = ptx_emit_setp_ne(b, "%p2", "%f0", "%f1", GpuType::GpuF32)
    b = ptx_emit_bra(b, "L_exit")
    b = ptx_emit_exit(b)
    let s = ptx_to_string(b)
    ptx_str_contains(s, "shfl.sync.up.b32") &&
    ptx_str_contains(s, "shfl.sync.bfly.b32") &&
    ptx_str_contains(s, "vote.ballot.b32") &&
    ptx_str_contains(s, "sqrt.approx.f32") &&
    ptx_str_contains(s, "rsqrt.approx.f32") &&
    ptx_str_contains(s, "ex2.approx.f32") &&
    ptx_str_contains(s, "lg2.approx.f32") &&
    ptx_str_contains(s, "sin.approx.f32") &&
    ptx_str_contains(s, "cos.approx.f32") &&
    ptx_str_contains(s, "abs.f32") &&
    ptx_str_contains(s, "rcp.approx.f32") &&
    ptx_str_contains(s, "setp.gt.f32") &&
    ptx_str_contains(s, "setp.ne.f32") &&
    ptx_str_contains(s, "bra L_exit;") &&
    ptx_str_contains(s, "exit;")
}

fn main() -> i32 with IO, Mut, Panic, Div, Alloc {
    let ok = ptx_advanced_self_check() && omega_ptx_opcode_pattern_smoke()
    if ok {
        print("PTX_OPCODE_SMOKE_PASS\n")
        return 0 as i32
    } else {
        print("PTX_OPCODE_SMOKE_FAIL\n")
        return 1 as i32
    }
}
SIO
} >"$tmp_ptx"

{
  emit_name_stub
  cat "$KERNEL_IR_PATH"
  cat "$PTX_PATH"
  cat "$LOWER_TO_PTX_PATH"
  trim_before_main "$METAL_PATH"
  cat <<'SIO'

fn omega_metal_opcode_pattern_smoke() -> bool with Mut, Panic, Div, Alloc {
    var helper_buf = metal_buf_new()
    helper_buf = metal_emit_sqrt(helper_buf, "f0", "f1")
    helper_buf = metal_emit_rsqrt(helper_buf, "f2", "f3")
    helper_buf = metal_emit_exp2(helper_buf, "f4", "f5")
    helper_buf = metal_emit_log2(helper_buf, "f6", "f7")
    helper_buf = metal_emit_sin(helper_buf, "f8", "f9")
    helper_buf = metal_emit_cos(helper_buf, "f10", "f11")
    helper_buf = metal_emit_abs(helper_buf, "f12", "f13")
    helper_buf = metal_emit_rcp(helper_buf, "f14", "f15")
    helper_buf = metal_emit_setp_gt(helper_buf, "p0", "a", "b")
    helper_buf = metal_emit_setp_ne(helper_buf, "p1", "a", "b")
    helper_buf = metal_emit_bra(helper_buf, "L_exit")
    helper_buf = metal_emit_exit(helper_buf)
    let helper_text = metal_to_string(helper_buf)

    ptx_str_contains(helper_text, "sqrt(f1)") &&
    ptx_str_contains(helper_text, "rsqrt(f3)") &&
    ptx_str_contains(helper_text, "exp2(f5)") &&
    ptx_str_contains(helper_text, "log2(f7)") &&
    ptx_str_contains(helper_text, "sin(f9)") &&
    ptx_str_contains(helper_text, "cos(f11)") &&
    ptx_str_contains(helper_text, "abs(f13)") &&
    ptx_str_contains(helper_text, "1.0f / f15") &&
    ptx_str_contains(helper_text, "bool p0 = a > b;") &&
    ptx_str_contains(helper_text, "bool p1 = a != b;") &&
    ptx_str_contains(helper_text, "control: branch L_exit") &&
    ptx_str_contains(helper_text, "return;")
}

fn main() -> i32 with IO, Mut, Panic, Div, Alloc {
    let ok = metal_self_check() && omega_metal_opcode_pattern_smoke()
    if ok {
        print("METAL_OPCODE_SMOKE_PASS\n")
        return 0 as i32
    } else {
        print("METAL_OPCODE_SMOKE_FAIL\n")
        return 1 as i32
    }
}
SIO
} >"$tmp_metal"

set +e
ptx_output="$(PATH="$ROOT_DIR:$PATH" "$SOUC_BIN" run "$tmp_ptx" 2>&1)"
ptx_rc=$?
set -e
printf '%s\n' "$ptx_output" | tee -a "$SMOKE_LOG"
if [ "$ptx_rc" -ne 0 ]; then
  echo "error: PTX opcode smoke failed (exit=$ptx_rc)" >&2
  exit "$ptx_rc"
fi
if ! printf '%s\n' "$ptx_output" | rg -q "PTX_OPCODE_SMOKE_PASS"; then
  echo "error: missing PTX opcode smoke pass marker" >&2
  exit 2
fi

set +e
metal_output="$(PATH="$ROOT_DIR:$PATH" "$SOUC_BIN" run "$tmp_metal" 2>&1)"
metal_rc=$?
set -e
printf '%s\n' "$metal_output" | tee -a "$SMOKE_LOG"
if [ "$metal_rc" -ne 0 ]; then
  echo "error: Metal opcode smoke failed (exit=$metal_rc)" >&2
  exit "$metal_rc"
fi
if ! printf '%s\n' "$metal_output" | rg -q "METAL_OPCODE_SMOKE_PASS"; then
  echo "error: missing Metal opcode smoke pass marker" >&2
  exit 2
fi

echo "GPU_OPCODE_SMOKE_PASS"
