#!/usr/bin/env bash
# madaros_f128_f256_v0e57_language_arrays_gate.sh — V0-E.5.7 language f128 in
# fixed arrays.
#
# Spec: docs/architecture/F128_F256_LADDER.md §V0-E (V0-E.5.7 slice)
# Semantic-Lane-ID: WS-G-V0E-STDLIB-GUM-SURFACE
# Claim clock: ADR-008 / ADR-009 — oracle_class=sounio_native_expected
#
# V0-E.5.7 green (this gate):
#   - Madaros-run `[f128; N]` locals: every slot holds an F128Bits handle (the
#     ordinary boxed one-word array slot; no f64 payload anywhere)
#   - element identity lives in ONE place per binding (the local's
#     array_elem_wide_bits, stamped from the annotation and propagated by
#     `let zs = xs`; a struct field's layout element type for `arr: [f128; N]`),
#     and every `a[i]` consumer reads it — operand, `let` RHS (annotated or not),
#     scalar kind, store target
#   - array literal `[a, b, c]` and repeat literal `[v; N]` in f128 position
#     lower element-wise through the f128 value path (never the f64
#     array-literal classifier); repeat fills ALL N slots (null handle ≠ 0.0)
#   - stores: `a[i] = 2.0` is binary128, `a[i] = y` / `a[i] = b[j]` copy
#   - `&[f128; N]` params, by-value `[f128; N]` params (slot copy, no aliasing),
#     `let zs = xs` copies (no aliasing), comparisons over elements
#   - struct field `arr: [f128; N]`: literal init, read, store through `q.arr[i]`
#   - DCE: a `[f128; N]` type anywhere (let / param / field) marks the softfloat
#     desugar targets even when the program has no scalar f128 (array_only probe)
#   - Anti-f64 through an element: (1+~1e-20)^2 ≠ 1
#   - `[f256; N]`, an inexact element literal (0.1), `+=` on an element and a
#     `[f128; N]` let without initialiser still fail closed (no ELF)
#
# Explicitly NOT claimed:
#   - lean_single language f128 (still f64 greenwash)
#   - bare float literals as `[f128; N]` elements (checker infers [f64; N] — E001)
#   - fns returning `[f128; N]`, Seq<f128>, `for x in xs`, nested arrays
#   - f256, methods returning f128, `+=`, `%`, print_f128, GUM
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

unset SOUC_BIN SOUNIO_SOUC_BIN || true
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

SOUC="${MADAROS_RAW_BIN:-${SOUC:-$ROOT_DIR/bin/souc}}"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/f128-ladder-v0e57.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

PASS=0
FAIL=0
FAILURES=()
note_pass() { PASS=$((PASS+1)); echo "PASS $1"; }
note_fail() { FAIL=$((FAIL+1)); FAILURES+=("$1"); echo "FAIL $1" >&2; }

echo "=== madaros_f128_f256_ladder_gate stage=v0e57 ==="
echo "slice=v0e57_language_f128_fixed_arrays"
echo "claim_clock=sounio_native_expected"
echo "adr=ADR-008+ADR-009"

NATIVE_EXPECT=(
  "wire_sum3=0000000000000000:4001800000000000"
  "wire_loop_sum=0000000000000000:4001800000000000"
  "wire_elem_sq=00005e7284324908:3fff000000000000"
  "wire_field_arr_sum=0000000000000000:4001400000000000"
)

LOWER=self-hosted/ir/lower.sio
if grep -Fq 'array_elem_wide_bits: Box<LowerLocalI64Slots>' "$LOWER" \
  && grep -Fq 'fn lower_index_base_is_f128_array_ref' "$LOWER" \
  && grep -Fq 'fn lower_let_stmt_f128_array_ref' "$LOWER" \
  && grep -Fq 'fn lower_f128_array_literal_ref' "$LOWER" \
  && grep -Fq 'fn lower_type_expr_is_array_of_f128' "$LOWER" \
  && grep -Fq 'indexed element in f128 position is not an element of a [f128; N] array (V0-E.5.7)' "$LOWER"; then
  note_pass "lower_f128_array_markers"
else
  note_fail "lower_f128_array_markers_missing"
fi

if grep -Fq 'fn spec_dce_type_expr_mentions_f128' self-hosted/check/specializer.sio; then
  note_pass "dce_f128_array_type_marker"
else
  note_fail "dce_f128_array_type_marker_missing"
fi

SMOKE=tests/run-pass/f128_v0e57_language_arrays.sio
if grep -Fq 'use math::softfloat_f128::{' "$SMOKE" \
  && grep -Fq 'let xs: [f128; 3] = [one, two, three]' "$SMOKE" \
  && grep -Fq 'var ys: [f128; 4] = [one; 4]' "$SMOKE" \
  && grep -Fq 'fn sum3(a: &[f128; 3]) -> f128' "$SMOKE" \
  && grep -Fq 'fn sum3_byval(a: [f128; 3]) -> f128' "$SMOKE" \
  && grep -Fq 'struct Pair2 { arr: [f128; 2], tag: i64 }' "$SMOKE" \
  && grep -Fq 'acc = acc + xs[i]' "$SMOKE" \
  && grep -Fq 'ys[1] = 2.0' "$SMOKE" \
  && grep -Fq 'ys[3] = y' "$SMOKE" \
  && grep -Fq 'var zs = xs' "$SMOKE" \
  && grep -Fq 'q.arr[1] = three' "$SMOKE" \
  && grep -Fq 'v0e57_main_entered' "$SMOKE" \
  && ! grep -Fq 'F128Bits {' "$SMOKE" \
  && ! grep -Fq 'f128_bits_soft_' "$SMOKE"; then
  note_pass "smoke_is_language_f128_arrays_not_f128bits_copy"
else
  note_fail "smoke_must_be_language_f128_arrays"
fi

REFUSE_SENTINEL='f128/f256 Madaros-run softfloat lowering is not implemented (V0-E.4.1 fail-closed; no f64 greenwash)'

# f256 array: same-format ops typecheck (V0-E.2) but there is no f256 payload —
# must refuse with the V0-E.4.1 sentinel, never emit an ELF.
cat >"$TMP_DIR/lang_f256_array.sio" <<'EOF'
use math::softfloat_f128::{f128_from_limbs, f128_to_lo, f128_to_hi}

fn main() -> i32 with IO, Mut, Panic, Div {
    let x: f256 = 1.0
    let xs: [f256; 2] = [x, x]
    let y: f256 = xs[0] * xs[1]
    return 0
}
EOF

# Inexact literal stored into an f128 element: 0.1 must never be f64-widened.
cat >"$TMP_DIR/lang_inexact_elem_store.sio" <<'EOF'
use math::softfloat_f128::{f128_from_limbs, f128_to_lo, f128_to_hi}

fn main() -> i32 with IO, Mut, Panic, Div {
    let one: f128 = 1.0
    var xs: [f128; 2] = [one, one]
    xs[0] = 0.1
    if f128_to_hi(xs[0]) == 0 { return 1 }
    return 0
}
EOF

# Inexact repeat literal as [f128; N] initialiser. Today the checker refuses the
# literal shape (E001: found [f64; 2]); if that is ever relaxed the lowerer's
# "no f64 widen" refusal takes over. Either way: non-zero rc, no ELF.
cat >"$TMP_DIR/lang_inexact_elem_literal.sio" <<'EOF'
use math::softfloat_f128::{f128_from_limbs, f128_to_lo, f128_to_hi}

fn main() -> i32 with IO, Mut, Panic, Div {
    let xs: [f128; 2] = [0.1; 2]
    if f128_to_hi(xs[0]) == 0 { return 1 }
    return 0
}
EOF

# Compound assignment on an f128 element stays fail-closed.
cat >"$TMP_DIR/lang_compound_elem.sio" <<'EOF'
use math::softfloat_f128::{f128_from_limbs, f128_to_lo, f128_to_hi}

fn main() -> i32 with IO, Mut, Panic, Div {
    let one: f128 = 1.0
    var xs: [f128; 2] = [one, one]
    xs[0] += one
    if f128_to_hi(xs[0]) == 0 { return 1 }
    return 0
}
EOF

# DCE trigger: no scalar f128 let / param / return / field anywhere — the ONLY
# f128 in the program is a `[f128; N]` type (param) plus an element compare.
# Without the array-aware mark the desugar targets are dropped and the binary
# hits the body-less stub (SIGILL, exit 132).
cat >"$TMP_DIR/array_only.sio" <<'EOF'
use math::softfloat_f128::{f128_from_limbs, f128_to_lo, f128_to_hi}

fn first_lt_second(a: &[f128; 2]) -> bool { a[0] < a[1] }

fn main() -> i32 with IO, Mut, Panic, Div {
    let xs: [f128; 2] = [f128_from_limbs(0, 4611404543450677248), f128_from_limbs(0, 4611686018427387904)]
    if !first_lt_second(&xs) { return 1 }
    if f128_to_hi(xs[1]) != 4611686018427387904 { return 2 }
    return 0
}
EOF

# True when the log contains ANY of the `@@`-separated fixed strings.
log_has_any() {
  local log="$1" wants="$2" w
  while [[ -n "$wants" ]]; do
    w="${wants%%@@*}"
    if grep -Fq "$w" "$log"; then return 0; fi
    if [[ "$wants" == *@@* ]]; then wants="${wants#*@@}"; else wants=""; fi
  done
  return 1
}

if [[ -x "$SOUC" ]]; then
  for neg in lang_f256_array:"$REFUSE_SENTINEL":language_f256_array_still_fail_closed \
             lang_inexact_elem_store:"no f64 widen":language_f128_inexact_elem_store_still_fail_closed \
             lang_inexact_elem_literal:"no f64 widen@@expected [f128; 2]":language_f128_inexact_elem_literal_still_refused \
             lang_compound_elem:"$REFUSE_SENTINEL":language_f128_compound_elem_assign_still_fail_closed; do
    name="${neg%%:*}"; rest="${neg#*:}"; want="${rest%:*}"; label="${rest##*:}"
    set +e
    "$SOUC" compile "$TMP_DIR/$name.sio" -o "$TMP_DIR/$name.elf" >"$TMP_DIR/$name.compile.log" 2>&1
    n_rc=$?
    set -e
    if [[ "$n_rc" -ne 0 ]] && log_has_any "$TMP_DIR/$name.compile.log" "$want" && [[ ! -s "$TMP_DIR/$name.elf" ]]; then
      note_pass "$label"
    else
      note_fail "${label%_still_*}_fail_closed_regression rc=$n_rc"
      tail -30 "$TMP_DIR/$name.compile.log" >&2 || true
    fi
  done

  set +e
  "$SOUC" run "$TMP_DIR/array_only.sio" >"$TMP_DIR/array_only.run.log" 2>&1
  ao_rc=$?
  set -e
  if [[ "$ao_rc" -eq 0 ]]; then
    note_pass "madaros_run_array_only_dce_trigger"
  else
    note_fail "madaros_run_array_only_dce_trigger rc=$ao_rc"
    tail -30 "$TMP_DIR/array_only.run.log" >&2 || true
  fi

  set +e
  "$SOUC" run "$SMOKE" >"$TMP_DIR/madaros.run.log" 2>&1
  m_rc=$?
  set -e
  if [[ "$m_rc" -eq 0 ]] && grep -Fq 'PASS f128_v0e57_language_arrays' "$TMP_DIR/madaros.run.log"; then
    note_pass "madaros_run_language_f128_arrays"
    for want in "${NATIVE_EXPECT[@]}"; do
      if grep -Fq "$want" "$TMP_DIR/madaros.run.log"; then
        note_pass "madaros_native_hex:${want%%=*}"
      else
        note_fail "madaros_native_hex_mismatch:${want%%=*}"
        echo "want $want" >&2
        cat "$TMP_DIR/madaros.run.log" >&2 || true
      fi
    done
  else
    note_fail "madaros_run_language_f128_arrays rc=$m_rc"
    tail -40 "$TMP_DIR/madaros.run.log" >&2 || true
  fi
else
  note_fail "souc_missing"
fi

echo "NOTE v0e57_deferred f256=pending f128_methods=pending array_return=pending seq_f128=pending for_in_f128_array=pending elem_float_literals=checker_E001 compound_assign=pending print_builtin=pending gum=pending"
echo "NOTE adr009 python_softfloat=not_claim_clock rust=not_claim_clock lean_single_language=greenwash"

echo "---"
echo "PASS_COUNT=$PASS"
echo "FAIL_COUNT=$FAIL"
if [[ "$FAIL" -eq 0 ]]; then
  echo "PASS f128_f256_v0e57_language_arrays general=ieee754 anti_f64=green language_arrays=madaros claim_clock=sounio_native_expected"
  echo "PASS madaros_f128_f256_ladder_gate stage=v0e57"
  exit 0
fi
echo "FAIL madaros_f128_f256_ladder_gate stage=v0e57" >&2
for f in "${FAILURES[@]}"; do echo "  - $f" >&2; done
exit 1
