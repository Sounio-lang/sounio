#!/usr/bin/env bash
# Tier 0 witness matrix.
#
# Measures the canonical witnesses from .claude/workflows/*-round.js against
# WHATEVER compiler is handed to it. Every defect below is CLOSED on branch
# integration/native-v2-honest (tip 60686c617b) and was NEVER MERGED to main.
# This answers the only question that matters for launch: does the shipped
# tree exhibit them? Measured 2026-08-14: 8/10 closed, w4 and w7 OPEN --
# both later fixed and merged to main (PR #1737).
#
# 2026-08-15: pointing this script at "souc compile" (arg 1, default
# ./bin/souc) measures the CHECKED-IN PREBUILT bin/madaros-linux-x86_64, not
# a build of the PR's own self-hosted/ source. That prebuilt only updates via
# the scheduled madaros-prebuilt-refresh.yml workflow, so it can be (and, on
# 2026-08-15, was) stale relative to main by exactly the fixes this script
# exists to guard. To measure the PR's own source, build Madaros fresh
# (scripts/ci/build_modular_madaros.sh) and pass BOTH the souc wrapper (arg 1,
# with MADAROS_RAW_BIN set to the fresh ELF so "souc compile" uses it) and the
# raw ELF path again as arg 2 (for w14's direct --native-v2-emit-scalar
# probe, which bypasses the wrapper entirely). See DECLARED_OPEN below for
# the residuals that are real on a fresh build and not on the prebuilt.
#
# Usage: bash tests/witness_matrix/run.sh [path-to-souc] [path-to-raw-elf]
set -u
SOUC="${1:-./bin/souc}"
SOUC="$(cd "$(dirname "$SOUC")" && pwd)/$(basename "$SOUC")"  # absolutise: run_mm/run_reject cd into the case dir
RAW="${2:-}"  # optional: raw --native-v2-emit-* ELF (bypasses the souc/madaros wrapper) for w14
ROOT="$(cd "$(dirname "$0")" && pwd)"
CASES="$ROOT/cases"
TMP="$(mktemp -d)"
PASS=0
TOTAL=0
OPEN=0
OPEN_IDS=""
# Residual-by-identity allowlist (release_gate.sh pattern): a witness may be
# declared open ONLY by exact id + reason below. The gate fails if the actual
# open set differs from this set in ANY direction -- new opens, or a declared
# residual silently starting to pass (promotion must be witnessed, not assumed).
DECLARED_OPEN="w16 w17"
# w5 and w14 were open 2026-08-14/15 on a Madaros built fresh from current
# main.sio source: ir_empty_function() leaves its region unallocated by
# design (see ir_function_alloc_region's comment in ir.sio), and both the
# --native-v2-emit-{scalar,call,call5,call6} witness-harness builders in
# main.sio (38 call sites across the file) AND lower_closure_expr_ref in
# lower.sio (the real closure-lowering path) were never given the
# allocation call when the arena landed -- every one of them hit "IR
# instruction arena contract violated ... region slot -1 generation -1".
# FIXED in the same commit that removed them from this list: main.sio gained
# the missing ir_function_alloc_region call at all 38 sites; lower.sio's
# closure path gained the same guard used by every other real-function
# lowering entry point. Verified by running: all four --native-v2-emit-*
# probes emit correct-value ELFs, and w5's no-capture closure compiles and
# runs to 42.
#
# w16 is a DIFFERENT, newly-discovered defect the arena bug had been masking
# (fail-closed hid it): a capturing closure called DIRECTLY computes
# correctly, but the SAME capture passed as an argument to a higher-order
# function and called through the fn-pointer parameter silently drops the
# captured value -- a real silent miscompile, not a crash or reject. Left
# open and undisguised rather than folded into "closures are fixed."
#
# w17 is a SECOND thing the arena bug's fail-closed had been masking: the
# effect checker does not catch an IO-effect closure passed to an
# unannotated HOF and called from a pure function (checked:
# tests/compile-fail/closure_effect_escape.sio itself, which had been
# "passing" -- correctly rejected -- for the wrong reason: lowering crashed
# on the arena bug before it ever reached this file's actual effect leak).
# `souc check` on that file returns "check: OK" post-fix. Pre-existing gap
# in self-hosted/check/, unrelated to the IR-lowering fix, not attempted.

run_value() {
  id="$1"; file="$2"; want="$3"; wrong="$4"; origin="$5"
  TOTAL=$((TOTAL+1))
  elf="$TMP/$id.elf"
  rm -f "$elf"
  "$SOUC" compile "$file" -o "$elf" >"$TMP/$id.log" 2>&1
  if [ ! -f "$elf" ]; then
    printf "%-5s %-20s %-9s %-7s %s\\n" "$id" "COMPILE-FAIL" "-" "$want" "$origin"
    OPEN=$((OPEN+1)); OPEN_IDS="$OPEN_IDS $id"
    return
  fi
  chmod +x "$elf" 2>/dev/null
  "$elf" >/dev/null 2>&1
  got=$?
  if [ "$got" = "$want" ]; then
    status="CLOSED"; PASS=$((PASS+1))
  elif [ "$got" = "$wrong" ]; then
    status="OPEN-MISCOMPILE"; OPEN=$((OPEN+1)); OPEN_IDS="$OPEN_IDS $id"
  else
    status="OPEN-OTHER"; OPEN=$((OPEN+1)); OPEN_IDS="$OPEN_IDS $id"
  fi
  printf "%-5s %-20s %-9s %-7s %s\\n" "$id" "$status" "$got" "$want" "$origin"
}

run_reject() {
  id="$1"; file="$2"; origin="$3"
  TOTAL=$((TOTAL+1))
  if (cd "$(dirname "$file")" && "$SOUC" check "$(basename "$file")" >/dev/null 2>&1); then
    printf "%-5s %-20s %-9s %-7s %s\\n" "$id" "OPEN-ACCEPTS-ILLTYPED" "accepted" "REJECT" "$origin"
    OPEN=$((OPEN+1)); OPEN_IDS="$OPEN_IDS $id"
  else
    printf "%-5s %-20s %-9s %-7s %s\\n" "$id" "CLOSED" "rejected" "REJECT" "$origin"
    PASS=$((PASS+1))
  fi
}

run_mm() {
  id="$1"; want="$2"; origin="$3"
  TOTAL=$((TOTAL+1))
  rm -f "$TMP/$id.elf"
  (cd "$CASES/mm" && "$SOUC" compile prog.sio -o "$TMP/$id.elf") >"$TMP/$id.log" 2>&1
  if [ ! -f "$TMP/$id.elf" ]; then
    printf "%-5s %-20s %-9s %-7s %s\\n" "$id" "COMPILE-FAIL" "-" "$want" "$origin"
    OPEN=$((OPEN+1)); OPEN_IDS="$OPEN_IDS $id"
    return
  fi
  chmod +x "$TMP/$id.elf" 2>/dev/null
  "$TMP/$id.elf" >/dev/null 2>&1
  got=$?
  if [ "$got" = "$want" ]; then
    printf "%-5s %-20s %-9s %-7s %s\\n" "$id" "CLOSED" "$got" "$want" "$origin"
    PASS=$((PASS+1))
  else
    printf "%-5s %-20s %-9s %-7s %s\\n" "$id" "OPEN" "$got" "$want" "$origin"
    OPEN=$((OPEN+1)); OPEN_IDS="$OPEN_IDS $id"
  fi
}

run_scalar_emit() {
  id="$1"; want="$2"; origin="$3"
  TOTAL=$((TOTAL+1))
  if [ -z "$RAW" ]; then
    printf "%-5s %-20s %-9s %-7s %s\n" "$id" "SKIPPED-NO-RAW" "-" "$want" "$origin"
    OPEN=$((OPEN+1)); OPEN_IDS="$OPEN_IDS $id"
    return
  fi
  elf="$TMP/$id.elf"
  rm -f "$elf"
  "$RAW" --native-v2-emit-scalar "$want" "$elf" >"$TMP/$id.log" 2>&1
  if [ ! -f "$elf" ]; then
    printf "%-5s %-20s %-9s %-7s %s\n" "$id" "OPEN-EMIT-FAIL" "-" "$want" "$origin"
    OPEN=$((OPEN+1)); OPEN_IDS="$OPEN_IDS $id"
    return
  fi
  chmod +x "$elf" 2>/dev/null
  "$elf" >/dev/null 2>&1
  got=$?
  if [ "$got" = "$want" ]; then
    printf "%-5s %-20s %-9s %-7s %s\n" "$id" "CLOSED" "$got" "$want" "$origin"
    PASS=$((PASS+1))
  else
    printf "%-5s %-20s %-9s %-7s %s\n" "$id" "OPEN" "$got" "$want" "$origin"
    OPEN=$((OPEN+1)); OPEN_IDS="$OPEN_IDS $id"
  fi
}

echo "witness matrix -- compiler: $SOUC"
"$SOUC" --version 2>&1 | head -1
echo
printf "%-5s %-20s %-9s %-7s %s\\n" ID STATUS GOT WANT ORIGIN

run_value w1  "$CASES/w1_literal_coercion.sio"            0  99 "literal-coercion 4acea3a59e"
run_value w2  "$CASES/w2_float_arith_params.sio"          7   0 "float-arith params 2286fb6d5d"
run_value w3  "$CASES/w3_float_arith_fields.sio"          7   0 "float-arith fields c2a783f270"
run_value w3c "$CASES/w3c_float_memlit_control.sio"       4  99 "CONTROL mem+literal"
run_value w4  "$CASES/w4_f32_compare.sio"                12   0 "M1 a436a68712 -- see RESULTS.md: real cause is as-f32 truncation"
run_value w4c "$CASES/w4c_f64_compare_control.sio"       12   0 "CONTROL f64 compare"
run_value w5  "$CASES/w5_closure_nocapture.sio"          42   1 "M2 320f4d2352"
run_value w6  "$CASES/w6_match_guard.sio"                 7 100 "soundness-3 BUG A 5ca40eee31"
run_value w7  "$CASES/w7_enum_discriminant.sio"          30  20 "soundness-3 BUG B e71eac8d99"
run_value w8  "$CASES/w8_field_index_collision.sio"      42  74 "soundness-3 BUG C 6534d904e5"
run_value w8c "$CASES/w8c_field_nocollision_control.sio" 42  99 "CONTROL no collision"
run_value w4b "$CASES/w4b_f32_roundtrip.sio"             150 100 "ROOT CAUSE of w4: as-f32 fell through to float_to_int"
run_value w7b "$CASES/w7b_enum_reverse.sio"                1   2 "CONTROL (passes at baseline): a table reorder must not break Shape"
run_value w11 "$CASES/w11_float_literal_f32.sio"          12  99 "float literal coercion to f32 (contextual + operator)"
run_value w15 "$CASES/w15_field_triple_collision.sio"     42  99 "soundness-3 BUG C residue: 3-field same-first-letter collision"
run_value w16 "$CASES/w16_hof_capturing_closure.sio"     130  30 "NEW 2026-08-15: capturing closure via HOF fn-pointer drops the capture"
run_reject w12 "$CASES/w12_nonliteral_must_reject.sio" "CARDINAL: only literals coerce"
run_reject w13 "$CASES/w13_magnitude_must_reject.sio" "CARDINAL: magnitude guard on f32 narrowing"
run_reject w17 "$CASES/w17_effect_leak_via_hof.sio" "NEW 2026-08-15: IO effect leaks through an unannotated HOF, uncaught"
run_mm    w9  42 "release wall 8765ca1dc4"
run_reject w10 "$CASES/mm/prog2.sio" "import typecheck bypass e1ac6f7c87"
run_scalar_emit w14 42 "IR arena contract violated on native-v2-emit-scalar (fresh source build only) 2026-08-15"

echo
echo "correct: $PASS/$TOTAL   open: $OPEN"

# Residual-by-identity gate: pass only if the actual open set equals DECLARED_OPEN exactly.
actual_sorted="$(printf '%s\n' $OPEN_IDS | sort -u | tr '\n' ' ' | sed 's/ $//')"
declared_sorted="$(printf '%s\n' $DECLARED_OPEN | sort -u | tr '\n' ' ' | sed 's/ $//')"
rm -rf "$TMP"
if [ "$actual_sorted" = "$declared_sorted" ]; then
  echo "residuals match declared set: [$declared_sorted]"
  exit 0
else
  echo "RESIDUAL MISMATCH: actual=[$actual_sorted] declared=[$declared_sorted]"
  echo "A new witness opened, or a declared residual changed status -- both require a human look, not a silent gate change."
  exit 1
fi
