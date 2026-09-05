#!/usr/bin/env bash
# ADR-009 verified_foreign_reference gate: Koka for effect-row semantics.
#
# Sounio's self-hosted/check/effects_row.sio implements row-polymorphic
# effect subtyping ({IO | r} <= {IO, Mut | r'} if r <= r'). Koka is the
# reference implementation of algebraic effect handlers and row-typed
# effects, so it is an independent authority for the *same abstract
# rule*, not a reimplementation derived from Sounio's checker.
#
# This gate compiles 6 paired test cases (tests/frontend/effect_row_koka_parity/*.sio
# and tools/koka/effect_row_parity/*.kk) with both compilers and asserts
# their accept/reject verdicts match, case by case. Cases:
#   a: reflexive            {IO} <= {IO}                -> ACCEPT (both)
#   b: missing effect       {IO,Mut} <= {IO}             -> REJECT (both)
#   c: pure in effectful    {}  <= {IO}                  -> ACCEPT (both)
#   d: superset context     {IO} <= {IO,Mut,Div}         -> ACCEPT (both)
#   e: disjoint             {Panic} <= {IO,Mut,Div}      -> REJECT (both)
#   f: exact match          {IO,Mut} <= {IO,Mut}         -> ACCEPT (both)
#
# Effect names in the Koka cases are user-declared custom effects
# (io/mut/divv/pan), not Koka's builtin io/st/exn/div -- this keeps the
# comparison about row-subtyping structure only, not about matching
# Koka's built-in effect semantics.

set -euo pipefail
umask 077

fail() {
  printf 'effect-row-koka-parity: FAIL: %s\n' "$*" >&2
  exit 1
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
SOUC="${SOUNIO_TEST_SOUC_BIN:-$ROOT_DIR/bin/souc}"
KOKA_CASES_DIR="$ROOT_DIR/tools/koka/effect_row_parity"
SIO_CASES_DIR="$ROOT_DIR/tests/frontend/effect_row_koka_parity"
EXPECTED_KOKA_VERSION="3.2.3"

[[ -x "$SOUC" ]] || fail "souc not executable: $SOUC"
command -v koka >/dev/null 2>&1 || \
  fail "koka not on PATH; install ${EXPECTED_KOKA_VERSION} from https://github.com/koka-lang/koka/releases"

observed_version="$(koka --version 2>&1 | sed -n '1s/^Koka \([0-9.]*\),.*/\1/p')"
[[ "$observed_version" == "$EXPECTED_KOKA_VERSION" ]] || \
  fail "koka version drift: expected $EXPECTED_KOKA_VERSION, observed ${observed_version:-unknown}"

work="$(mktemp -d "${TMPDIR:-/tmp}/effect-row-koka-parity.XXXXXX")"
trap 'rm -rf "$work"' EXIT

# Cases b and e must be rejected; a, c, d, f must be accepted.
declare -A EXPECT=( [a]=ACCEPT [b]=REJECT [c]=ACCEPT [d]=ACCEPT [e]=REJECT [f]=ACCEPT )

mismatches=0
for case in a b c d e f; do
  sio="$SIO_CASES_DIR/case_$case.sio"
  kk="$KOKA_CASES_DIR/case_$case.kk"
  [[ -r "$sio" ]] || fail "missing case source: $sio"
  [[ -r "$kk" ]]  || fail "missing case source: $kk"

  sio_verdict=ACCEPT
  "$SOUC" compile "$sio" -o "$work/case_$case.elf" >"$work/sio_$case.log" 2>&1 || sio_verdict=REJECT
  grep -q "Compilation failed" "$work/sio_$case.log" && sio_verdict=REJECT

  kk_dir="$work/kk_$case"
  mkdir -p "$kk_dir"
  cp "$kk" "$kk_dir/prog.kk"
  koka_verdict=ACCEPT
  ( cd "$kk_dir" && koka -l prog.kk ) >"$work/koka_$case.log" 2>&1 || koka_verdict=REJECT

  expected="${EXPECT[$case]}"
  status="ok"
  if [[ "$sio_verdict" != "$expected" ]]; then
    status="MISMATCH (sounio=$sio_verdict expected=$expected)"
    mismatches=$((mismatches + 1))
  elif [[ "$koka_verdict" != "$expected" ]]; then
    status="MISMATCH (koka=$koka_verdict expected=$expected)"
    mismatches=$((mismatches + 1))
  fi
  printf 'case_%s: sounio=%s koka=%s expected=%s %s\n' \
    "$case" "$sio_verdict" "$koka_verdict" "$expected" "$status"
done

[[ "$mismatches" -eq 0 ]] || fail "$mismatches/6 cases diverged between Sounio and Koka"

printf 'EFFECT_ROW_KOKA_PARITY_V1\n'
printf 'oracle_class=verified_foreign_reference\n'
printf 'producer_language=Koka\n'
printf 'producer_role=EFFECT_ROW_SUBTYPING_PARITY\n'
printf 'semantic_authority_language=Sounio\n'
printf 'koka_version=%s\n' "$observed_version"
printf 'cases_checked=6\n'
printf 'mismatches=0\n'
printf 'result=PASS\n'
