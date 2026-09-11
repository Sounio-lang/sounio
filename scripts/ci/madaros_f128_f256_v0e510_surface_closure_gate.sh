#!/usr/bin/env bash
# madaros_f128_f256_v0e510_surface_closure_gate.sh — V0-E.5.10 surface
# closure: the four gaps V0-E.5.9 left documented in KNOWN_LIMITATIONS.
#
# Spec: docs/architecture/F128_F256_LADDER.md §V0-E (V0-E.5.10 slice)
# Semantic-Lane-ID: WS-G-V0E-STDLIB-GUM-SURFACE
# Claim clock: ADR-008 / ADR-009 — oracle_class=sounio_native_expected
#
# V0-E.5.10 green (this gate):
#   1. hex-float on the f64 path: `let h: f64 = 0x1.8p+0` is 1.5 (bits
#      0x3ff8000000000000), not the parser placeholder; single rounding
#      (nearest-even with a sticky bit: 0x1.00000000000008p+0 ties to 1.0,
#      0x1.00000000000008000000000001p+0 rounds up to 1+2^-52); subnormals
#      and the binary64 max are exact. Oracle: Python float.fromhex bits.
#      Also: the optional `f128`/`f256` suffix after a hex-float is taken
#      only when glued to the exponent digits — an identifier on the next
#      line used to be swallowed as the suffix (E017 on the following `(`).
#   2. `fn q() -> f128 { 0.25 }`, `return 1.5`, `-0.5` tail: check accepts
#      the float-literal tail against a wide return (last_literal_kind == 2),
#      and the lowerer routes the body-block tail / `return` literal through
#      lower_f128_value_ref (exact limbs or fail-closed) — `return 1.5` used
#      to emit binary64 bits into the F128Bits slot.
#   3. `[f128; N] = [1.0, 2.0]` / `[0.5; 3]` / struct field: check accepts a
#      literal array of float literals against a wide-float array.
#   4. `1_024.0`, `1_000`, `0x1_0`, `1_0.5e0_1`: the live flat lexer accepts
#      `_` before a digit (it split at `_` before, E137).
#   5. language f128 without `use math::softfloat_f128`: the driver adds the
#      stdlib module to the graph itself (implicit_import trace line) and the
#      program runs; when the stdlib cannot be resolved the compile refuses
#      (no ELF, sentinel) instead of emitting an ELF that traps with SIGILL.
#
# Explicitly NOT claimed:
#   - lean_single language f128 (still f64 greenwash)
#   - a float literal as the tail of a NESTED block (`if c { 1.0 } else { 2.0 }`
#     as the fn tail) in an -> f128 fn — still E008 / f64; only the body block
#     and `return` are routed
#   - the legacy driver lexer (native_compile_driver.sio) and `_` separators
#   - f256 literals (V0-E.4.1 sentinel unchanged), `%`, `+=`, GUM
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

unset SOUC_BIN SOUNIO_SOUC_BIN || true
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

SOUC="${MADAROS_RAW_BIN:-${SOUC:-$ROOT_DIR/bin/souc}}"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/f128-ladder-v0e510.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

PASS=0
FAIL=0
FAILURES=()
note_pass() { PASS=$((PASS+1)); echo "PASS $1"; }
note_fail() { FAIL=$((FAIL+1)); FAILURES+=("$1"); echo "FAIL $1" >&2; }

echo "=== madaros_f128_f256_ladder_gate stage=v0e510 ==="
echo "slice=v0e510_surface_closure_hexf64_tail_array_separators_implicit_import"
echo "claim_clock=sounio_native_expected"
echo "adr=ADR-008+ADR-009"

NATIVE_EXPECT=(
  "hex_f64_1_5_bits=4609434218613702656"
  "hex_f64_tie_even_bits=4607182418800017408"
  "hex_f64_sticky_up_bits=4607182418800017409"
  "tail_quarter=2.50000000000000000000000000000000000e-0001"
)

LITMOD=self-hosted/parser/f128_literal.sio
LOWER=self-hosted/ir/lower.sio
CHECK=self-hosted/check/check.sio
LEXER=self-hosted/lexer/mod.sio
DRIVER=self-hosted/compiler/module_frontend.sio
SPEC=self-hosted/check/specializer.sio

# 1. hex-float f64 reader present and wired at both parser entry points.
PARSER=self-hosted/parser/parser.sio
if grep -Fq 'pub fn f64_hex_literal_from_source(start: i64, end: i64) -> f64' "$PARSER" \
  && grep -Fq 'fn f64lit_scale_pow2(' "$PARSER" \
  && ! grep -v '^[[:space:]]*//' "$LITMOD" | grep -Eq '(^|[^a-z_])f64([^a-z_0-9]|$)' \
  && grep -Fq 'e.float_val = f64_hex_literal_from_source(start.start, parser_prev_token_end(p))' self-hosted/parser/exprs.sio \
  && grep -Fq 'return f64_hex_literal_from_source(start, end)' self-hosted/parser/parser.sio \
  && grep -Fq 'if sfx_span.start == parser_prev_token_end(p)' self-hosted/parser/exprs.sio \
  && ! grep -Fq 'e.float_val = 1.0' self-hosted/parser/exprs.sio; then
  note_pass "parser_hex_float_f64_exact_reader_wired"
else
  note_fail "parser_hex_float_f64_reader_missing_or_placeholder_still_set"
fi

# 2. checker: float-literal tail against wide return; literal array vs wide array.
if grep -Fq 'let float_lit_wide = (*c).last_literal_kind == 2' "$CHECK" \
  && grep -Fq 'int_lit_narrow || float_lit_wide || enum_int_ok' "$CHECK" \
  && grep -Fq 'fn checker_array_float_literal_compatible(' "$CHECK" \
  && grep -Fq 'checker_array_float_literal_compatible(expr, actual_ty, expected_ty)' "$CHECK"; then
  note_pass "checker_accepts_float_literal_tail_and_array"
else
  note_fail "checker_float_literal_tail_or_array_path_missing"
fi

# 2. lowerer: body-block tail and `return` literal routed to f128 limbs.
if grep -Fq 'var LOWER_F128_FN_BODY_PENDING: bool = false' "$LOWER" \
  && grep -Fq 'LOWER_F128_FN_BODY_PENDING = (*lo).current_func_loaded && lower_fn_returns_f128(&(*(*lo).current_func))' "$LOWER" \
  && grep -Fq 'if f128_body_tail && is_tail_stmt && lower_expr_is_float_literal_like_ref(&(*expr_box))' "$LOWER" \
  && grep -Fq 'Some(ret_box2) => self.lower_f128_value_ref(&(*ret_box2))' "$LOWER"; then
  note_pass "lower_routes_f128_fn_literal_tail_and_return"
else
  note_fail "lower_f128_fn_literal_tail_or_return_not_routed"
fi

# 4. lexer: separators in the live flat path.
if grep -Fq 'fn lex_flat_sep_before_digit(' "$LEXER" \
  && grep -Fq 'fn lex_flat_sep_before_hex_digit(' "$LEXER" \
  && [[ "$(grep -c 'lex_flat_sep_before_digit(pos, n)' "$LEXER")" -ge 3 ]] \
  && grep -Fq 'lex_flat_sep_before_hex_digit(pos, n)' "$LEXER"; then
  note_pass "lexer_flat_accepts_digit_separators"
else
  note_fail "lexer_flat_digit_separators_missing"
fi

# 5. driver: implicit import + fail-closed net.
if grep -Fq 'pub fn spec_program_mentions_f128(prog: &Program) -> bool' "$SPEC" \
  && grep -Fq 'implicit_import math/softfloat_f128.sio reason=f128_in_ast' "$DRIVER" \
  && grep -Fq 'fn module_frontend_bodyless_f128_targets(' "$DRIVER" \
  && [[ "$(grep -c 'if module_frontend_bodyless_f128_targets(&(\*module_box))' "$DRIVER")" -ge 2 ]]; then
  note_pass "driver_implicit_softfloat_import_and_bodyless_net"
else
  note_fail "driver_implicit_import_or_bodyless_net_missing"
fi

SMOKE=tests/run-pass/f128_v0e510_language_surface_closure.sio
if grep -Fq 'fn quarter() -> f128 { 0.25 }' "$SMOKE" \
  && grep -Fq 'fn one_and_half() -> f128 { return 1.5 }' "$SMOKE" \
  && grep -Fq 'let arr: [f128; 2] = [1.0, 2.0]' "$SMOKE" \
  && grep -Fq 'let rep: [f128; 3] = [0.5; 3]' "$SMOKE" \
  && grep -Fq 'let h15: f64 = 0x1.8p+0' "$SMOKE" \
  && grep -Fq 'let up: f64 = 0x1.00000000000008000000000001p+0' "$SMOKE" \
  && grep -Fq 'let sf: f64 = 1_024.0' "$SMOKE" \
  && grep -Fq 'let sw: f128 = 1_024.0' "$SMOKE" \
  && grep -Fq 'v0e510_main_entered' "$SMOKE" \
  && ! grep -Fq 'F128Bits {' "$SMOKE" \
  && ! grep -Fq 'f128_bits_soft_' "$SMOKE"; then
  note_pass "smoke_covers_the_four_surfaces"
else
  note_fail "smoke_must_cover_hexf64_tail_return_array_separators"
fi

INEXACT_SENTINEL='V0-E.5.9; no f64 widen'
BODYLESS_SENTINEL='an f128 desugar target has no body (V0-E.5.10'

# Negatives: an inexact literal in the NEW positions must still fail closed.
cat >"$TMP_DIR/neg_tail_tenth.sio" <<'EOF'
fn tenth() -> f128 { 0.1 }

fn main() -> i32 with IO, Mut, Panic, Div {
    let x = tenth()
    let y: f128 = 1.0
    if x == y { return 1 }
    return 0
}
EOF
cat >"$TMP_DIR/neg_return_tenth.sio" <<'EOF'
fn tenth() -> f128 { return 0.1 }

fn main() -> i32 with IO, Mut, Panic, Div {
    let x = tenth()
    let y: f128 = 1.0
    if x == y { return 1 }
    return 0
}
EOF
cat >"$TMP_DIR/neg_array_tenth.sio" <<'EOF'
fn main() -> i32 with IO, Mut, Panic, Div {
    let a: [f128; 1] = [0.1]
    let y: f128 = 1.0
    if a[0] == y { return 1 }
    return 0
}
EOF

# Implicit import: NO `use` at all, language f128 in let / fn tail / array /
# arithmetic. Exit 0 and a PASS line means the desugar targets were bound to
# real bodies through the implicit import, and the trace line must be there.
cat >"$TMP_DIR/no_import.sio" <<'EOF'
fn half() -> f128 { 0.5 }

fn main() -> i32 with IO, Mut, Panic, Div, Alloc {
    let a: f128 = 1.5
    let b: f128 = 0.25
    let s = a + b
    let want: f128 = 1.75
    if s != want { return 11 }
    let h = half()
    let two: f128 = 2.0
    let one: f128 = 1.0
    if h * two != one { return 12 }
    let arr: [f128; 2] = [1.0, 2.0]
    if arr[1] - arr[0] != one { return 13 }
    println("PASS no_import_language_f128")
    return 0
}
EOF

if [[ -x "$SOUC" ]]; then
  for neg in neg_tail_tenth:"$INEXACT_SENTINEL":inexact_literal_fn_tail_fail_closed \
             neg_return_tenth:"$INEXACT_SENTINEL":inexact_literal_return_fail_closed \
             neg_array_tenth:"$INEXACT_SENTINEL":inexact_literal_array_element_fail_closed; do
    name="${neg%%:*}"; rest="${neg#*:}"; want="${rest%:*}"; label="${rest##*:}"
    set +e
    "$SOUC" compile "$TMP_DIR/$name.sio" -o "$TMP_DIR/$name.elf" >"$TMP_DIR/$name.compile.log" 2>&1
    n_rc=$?
    set -e
    if [[ "$n_rc" -ne 0 ]] && grep -Fq "$want" "$TMP_DIR/$name.compile.log" && [[ ! -s "$TMP_DIR/$name.elf" ]]; then
      note_pass "$label"
    else
      note_fail "${label%_fail_closed}_fail_closed_regression rc=$n_rc"
      tail -30 "$TMP_DIR/$name.compile.log" >&2 || true
    fi
  done

  # 5a. implicit import resolves and the program runs.
  set +e
  "$SOUC" run "$TMP_DIR/no_import.sio" >"$TMP_DIR/no_import.run.log" 2>&1
  ni_rc=$?
  set -e
  if [[ "$ni_rc" -eq 0 ]] && grep -Fq 'PASS no_import_language_f128' "$TMP_DIR/no_import.run.log" \
     && grep -Fq 'implicit_import math/softfloat_f128.sio reason=f128_in_ast' "$TMP_DIR/no_import.run.log"; then
    note_pass "madaros_run_language_f128_without_softfloat_import"
  else
    note_fail "madaros_run_language_f128_without_softfloat_import rc=$ni_rc"
    tail -40 "$TMP_DIR/no_import.run.log" >&2 || true
  fi

  # 5b. stdlib unreachable: the implicit import cannot resolve -> refuse, no ELF.
  #     Run from a directory with no ./stdlib so the repo-relative fallback is off.
  set +e
  ( cd "$TMP_DIR" && SOUNIO_STDLIB_PATH="$TMP_DIR/nonexistent-stdlib" "$SOUC" compile "$TMP_DIR/no_import.sio" -o "$TMP_DIR/no_import_nostd.elf" ) >"$TMP_DIR/no_import_nostd.log" 2>&1
  ns_rc=$?
  set -e
  if [[ "$ns_rc" -ne 0 ]] && grep -Fq "$BODYLESS_SENTINEL" "$TMP_DIR/no_import_nostd.log" && [[ ! -s "$TMP_DIR/no_import_nostd.elf" ]]; then
    note_pass "no_stdlib_f128_program_refused_not_sigill"
  else
    note_fail "no_stdlib_f128_program_refused_not_sigill rc=$ns_rc"
    tail -30 "$TMP_DIR/no_import_nostd.log" >&2 || true
  fi

  set +e
  "$SOUC" run "$SMOKE" >"$TMP_DIR/madaros.run.log" 2>&1
  m_rc=$?
  set -e
  if [[ "$m_rc" -eq 0 ]] && grep -Fq 'PASS f128_v0e510_language_surface_closure' "$TMP_DIR/madaros.run.log"; then
    note_pass "madaros_run_language_f128_surface_closure"
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
    note_fail "madaros_run_language_f128_surface_closure rc=$m_rc"
    tail -40 "$TMP_DIR/madaros.run.log" >&2 || true
  fi
else
  note_fail "souc_missing"
fi

echo "NOTE v0e510_deferred driver_lexer_separators=pending f256=pending gum=pending kl8_residuals=closed_elsewhere"
echo "NOTE adr009 python_softfloat=not_claim_clock rust=not_claim_clock lean_single_language=greenwash"

echo "---"
echo "PASS_COUNT=$PASS"
echo "FAIL_COUNT=$FAIL"
if [[ "$FAIL" -eq 0 ]]; then
  echo "PASS f128_f256_v0e510_surface_closure hex_f64=single_rounding tail_return=f128_limbs array_literal=accepted separators=lexed implicit_import=madaros claim_clock=sounio_native_expected"
  echo "PASS madaros_f128_f256_ladder_gate stage=v0e510"
  exit 0
fi
echo "FAIL madaros_f128_f256_ladder_gate stage=v0e510" >&2
for f in "${FAILURES[@]}"; do echo "  - $f" >&2; done
exit 1
