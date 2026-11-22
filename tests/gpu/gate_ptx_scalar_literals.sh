#!/usr/bin/env bash
# Gate for scalar literals, and for constructs the emitter used to drop, in
# `kernel fn` bodies through `souc build --backend gpu`
# (P0.8.1: a wrong number that prints is worse than a refusal).
#
# Before this gate, `let b = a * 2.0` emitted `mov.u64 %rd2, 0` and a multiply that
# read the never-written `%fd2`; `-x` emitted `sub dst, %rd0, src` (register 0 is the
# first kernel parameter); a bool literal branched on an unwritten predicate.
#
# Each LOWERED case must reproduce its golden PTX byte for byte
# (tests/golden/gpu_ptx_literals/<case>.ptx) AND carry the exact literal encoding
# checked below. Each REFUSED case must exit non-zero, write no PTX, and say why.
#
# Structural check only: no ptxas / driver run (see gate_public_gpu_cfg_build.sh).
# Regenerate goldens after an intended emitter change with:
#   SOUNIO_GPU_LITERAL_GOLDEN_UPDATE=1 bash tests/gpu/gate_ptx_scalar_literals.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC="${SOUC:-./bin/souc}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
CASES="tests/gpu/literals"
GOLDEN="tests/golden/gpu_ptx_literals"
UPDATE="${SOUNIO_GPU_LITERAL_GOLDEN_UPDATE:-0}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

PASS=0
FAIL=0
pass() { PASS=$((PASS + 1)); echo "PASS  $1"; }
fail() { FAIL=$((FAIL + 1)); echo "FAIL  $1: $2"; }

build() { # case -> sets RC, OUT, LOG
    OUT="$TMP_DIR/$1.ptx"
    LOG="$TMP_DIR/$1.log"
    RC=0
    timeout 60 "$SOUC" build "$CASES/$1.sio" --backend gpu -o "$OUT" >"$LOG" 2>&1 || RC=$?
}

# lowered <case> <egrep pattern that must match> [<egrep pattern that must not match>]
lowered() {
    local c="$1" must="$2" mustnot="${3:-}"
    build "$c"
    if [ "$RC" -ne 0 ] || [ ! -s "$OUT" ]; then
        fail "$c" "expected PTX, got rc=$RC: $(tail -3 "$LOG" | tr '\n' ' ')"
        return
    fi
    if [ "$UPDATE" = "1" ]; then
        cp "$OUT" "$GOLDEN/$c.ptx"
    fi
    if ! cmp -s "$OUT" "$GOLDEN/$c.ptx"; then
        fail "$c" "PTX differs from $GOLDEN/$c.ptx: $(diff "$GOLDEN/$c.ptx" "$OUT" | head -6 | tr '\n' ' ')"
        return
    fi
    if ! grep -Eq "$must" "$OUT"; then
        fail "$c" "missing literal encoding /$must/"
        return
    fi
    if [ -n "$mustnot" ] && grep -Eq "$mustnot" "$OUT"; then
        fail "$c" "forbidden pattern /$mustnot/ present"
        return
    fi
    pass "$c lowered"
}

# refused <case> <egrep pattern for the diagnostic>
refused() {
    local c="$1" why="$2"
    build "$c"
    if [ "$RC" -eq 0 ]; then
        fail "$c" "expected a refusal, compiler exited 0"
        return
    fi
    if [ -s "$OUT" ]; then
        fail "$c" "refused but still wrote PTX"
        return
    fi
    if ! grep -Eq "$why" "$LOG"; then
        fail "$c" "refused without the expected diagnostic /$why/: $(tail -3 "$LOG" | tr '\n' ' ')"
        return
    fi
    pass "$c refused"
}

if [ ! -x "$SOUC" ]; then
    echo "FATAL: souc not executable: $SOUC"
    exit 1
fi

echo "=== GPU scalar-literal PTX gate ==="
echo "SOUC: $SOUC"
echo ""

# No integer move may feed a float register, and nothing may read %rd0/%fd0 as a zero.
NOT_INT_INTO_FLOAT='mov\.u64 %rd[0-9]+, 0;'

# a * 2.0 : 2.0 = 0x4000000000000000
lowered scalar_param_lit 'mov\.f64 %fd2, 0d4000000000000000;' "$NOT_INT_INTO_FLOAT"
lowered f64_lit_rhs      'mov\.f64 %fd[0-9]+, 0d4000000000000000;' "$NOT_INT_INTO_FLOAT"
# 0.1 + a : 0.1 = 0x3FB999999999999A (literal on the left)
lowered f64_lit_lhs      'mov\.f64 %fd[0-9]+, 0d3FB999999999999A;' "$NOT_INT_INTO_FLOAT"
# a * -2.5 : 2.5 = 0x4004000000000000, negated as x * -1.0 (0xBFF0000000000000)
lowered f64_lit_neg      'mov\.f64 %fd[0-9]+, 0dBFF0000000000000;' "sub\\.[su]64 %rd[0-9]+, %rd0,"
# -x on an f64 value
lowered f64_neg_value    'mul\.rn\.f64 %fd[0-9]+, %fd[0-9]+, %fd[0-9]+;' "sub\\.[su]64 %rd[0-9]+, %rd0,|st\\.global\\.s64"
# a * 0.1 with a: f32 : 0.1f = 0x3DCCCCCD, multiply and store in f32
lowered f32_lit          'mov\.f32 %f[0-9]+, 0f3DCCCCCD;' "mul\\.rn\\.f64|st\\.global\\.f64"
# tid + -5 : integer negation from its own zero, never from %rd0
lowered int_lit_neg      'sub\.s64 %rd[0-9]+, %rd[0-9]+, %rd[0-9]+;' "sub\\.[su]64 %rd[0-9]+, %rd0,"

refused bool_lit          'refus(ed|ing to emit PTX)'
refused int_lit_float_ctx 'E004|cannot be combined'

# Constructs the emitter used to drop without a word (follow-up to P0.8.1).
# gpu_thread_id_y() was matched by prefix and read %tid.x.
lowered thread_id_y       'mov\.u32 %r[0-9]+, %tid\.y;' 'tid\.x'
# `var` locals: the stack slot was never allocated (st/ld through an unwritten %rd).
refused var_local          'GPU codegen refused: a `var` local'
# A call to a user fn vanished; the store wrote an unwritten register.
refused unknown_call       'GPU codegen refused: a call .* \(call `scale`\)'
# Not an intrinsic axis: refused as an unknown call, never read as %tid.x.
refused thread_id_bad_axis 'GPU codegen refused: a call .* \(call `gpu_thread_id_w`\)'
# An if-expression value (phi) is refused (HLIR already refuses this shape).
refused if_value           'HLIR_LOWERING_REFUSED|GPU codegen refused'
# A block expression's value is its tail; it used to be unit, so `y` stored 0.
lowered block_value        'st\.global\.f64 \[%rd[0-9]+\], %fd[0-9]+;' 'st\.global\.s64'
# `match` lowered only its first arm, unconditionally, for every thread.
refused match_stmt         'GPU codegen refused: `match` in a kernel body'

echo ""
echo "Summary: pass=$PASS fail=$FAIL"
[ "$FAIL" -eq 0 ]
