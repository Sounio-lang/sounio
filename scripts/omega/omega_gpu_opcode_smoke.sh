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

emit_sio_decl() {
  local file="$1"
  local kind="$2"
  local name="$3"
  awk -v kind="$kind" -v name="$name" '
    function is_target_decl(line) {
      if (kind == "fn") {
        return line ~ ("^(pub )?fn " name "\\(")
      }
      return line ~ ("^(pub )?" kind " " name "([ {]|$)")
    }

    function is_top_level_decl(line) {
      return line ~ /^(pub )?(fn|struct) /
    }

    is_target_decl($0) {
      printing = 1
    }

    printing && is_top_level_decl($0) && !is_target_decl($0) {
      exit
    }

    printing {
      print
    }
  ' "$file"
}

emit_i64_to_string_stub() {
  cat <<'SIO'
fn i64_to_string(n: i64) -> string with Mut, Panic, Div, Alloc {
    if n < 0 {
        return "0"
    }
    if n == 0 { return "0" }

    var digits: [i8; 32] = [0; 32]
    var rev_len: i64 = 0
    var value: i64 = n

    while value > 0 {
        let digit = value % 10
        digits[rev_len as usize] = 48 + digit as i8
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
SIO
}

for required in \
  "$PTX_PATH" \
  "$KERNEL_IR_PATH" \
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
  emit_i64_to_string_stub
  cat "$KERNEL_IR_PATH"
  cat "$PTX_PATH"
  sed -n '319,328p' "$PTX_ADVANCED_PATH"
  sed -n '636,662p' "$PTX_ADVANCED_PATH"
  sed -n '680,711p' "$PTX_ADVANCED_PATH"
  sed -n '731,781p' "$PTX_ADVANCED_PATH"
  sed -n '791,885p' "$PTX_ADVANCED_PATH"
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
    let ok = omega_ptx_opcode_pattern_smoke()
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
  emit_i64_to_string_stub
  emit_sio_decl "$METAL_PATH" struct MetalBuf
  emit_sio_decl "$METAL_PATH" fn metal_buf_new
  emit_sio_decl "$METAL_PATH" fn metal_push_byte
  emit_sio_decl "$METAL_PATH" fn metal_push_str
  emit_sio_decl "$METAL_PATH" fn metal_buf_contains
  emit_sio_decl "$METAL_PATH" fn metal_emit_sqrt
  emit_sio_decl "$METAL_PATH" fn metal_emit_rsqrt
  emit_sio_decl "$METAL_PATH" fn metal_emit_exp2
  emit_sio_decl "$METAL_PATH" fn metal_emit_log2
  emit_sio_decl "$METAL_PATH" fn metal_emit_sin
  emit_sio_decl "$METAL_PATH" fn metal_emit_cos
  emit_sio_decl "$METAL_PATH" fn metal_emit_abs
  emit_sio_decl "$METAL_PATH" fn metal_emit_rcp
  emit_sio_decl "$METAL_PATH" fn metal_emit_setp_gt
  emit_sio_decl "$METAL_PATH" fn metal_emit_setp_ne
  emit_sio_decl "$METAL_PATH" fn metal_emit_bra
  emit_sio_decl "$METAL_PATH" fn metal_emit_exit
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

    metal_buf_contains(helper_buf, "sqrt(f1)") &&
    metal_buf_contains(helper_buf, "rsqrt(f3)") &&
    metal_buf_contains(helper_buf, "exp2(f5)") &&
    metal_buf_contains(helper_buf, "log2(f7)") &&
    metal_buf_contains(helper_buf, "sin(f9)") &&
    metal_buf_contains(helper_buf, "cos(f11)") &&
    metal_buf_contains(helper_buf, "abs(f13)") &&
    metal_buf_contains(helper_buf, "1.0f / f15") &&
    metal_buf_contains(helper_buf, "bool p0 = a > b;") &&
    metal_buf_contains(helper_buf, "bool p1 = a != b;") &&
    metal_buf_contains(helper_buf, "control: branch L_exit") &&
    metal_buf_contains(helper_buf, "return;")
}

fn main() -> i32 with IO, Mut, Panic, Div, Alloc {
    let ok = omega_metal_opcode_pattern_smoke()
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
if command -v rg >/dev/null 2>&1; then
  if ! printf '%s\n' "$ptx_output" | rg -qF "PTX_OPCODE_SMOKE_PASS"; then
    echo "error: missing PTX opcode smoke pass marker" >&2
    exit 2
  fi
elif ! printf '%s\n' "$ptx_output" | grep -qF "PTX_OPCODE_SMOKE_PASS"; then
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
if command -v rg >/dev/null 2>&1; then
  if ! printf '%s\n' "$metal_output" | rg -qF "METAL_OPCODE_SMOKE_PASS"; then
    echo "error: missing Metal opcode smoke pass marker" >&2
    exit 2
  fi
elif ! printf '%s\n' "$metal_output" | grep -qF "METAL_OPCODE_SMOKE_PASS"; then
  echo "error: missing Metal opcode smoke pass marker" >&2
  exit 2
fi

echo "GPU_OPCODE_SMOKE_PASS"
