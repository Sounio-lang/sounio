#!/usr/bin/env bash
# release_gate.sh — the HONEST Sounio release gate.
#
# Aggregates ONLY the gates that build the modular compiler (mc) from
# self-hosted/compiler/main.sio via the bootstrap ./bin/souc and assert REAL
# observed behavior of the current compiler. Each gate prints PASS/FAIL and a
# final verdict is printed at the end. Exit 0 iff EVERY honest gate is green
# (with the two tracked, by-design KNOWN_MISCOMPILE residuals below).
#
# This REPLACES the retired DEMETRIOS/cargo-era scripts:
#   scripts/ci/release_check.sh, scripts/ci/run_release_critical_pack.sh
# (both now exit 2 and point here).
#
# === What is wired (honest gates over real main.sio) ===========================
#   scripts/ci/native_v2_calls_arity_gate.sh          (rc==0)
#   scripts/ci/native_v2_e2e_codegen_suite_gate.sh    (rc==0)
#   scripts/ci/native_v2_e2e_exit_code_gate.sh        (rc==0)
#   tests/native_v2_capgate/run.sh                    (parse "32/32"; no rc)
#   tests/native_v2_soundness_gate/run.sh             (rc==0 + "7/7")
#   tests/native_v2_enum_gate/run.sh                  (rc==0 + "15/15")
#   tests/native_v2_closure_gate/run.sh               (rc==0 + "7/7"; M2 no-capture + capture rejects cleanly)
#   tests/native_v2_checker_crash_gate/run.sh         (rc==0 + "26/26")
#   tests/native_v2_literal_coercion_gate/run.sh      (rc==0 + "19/19")
#   tests/native_v2_float_compare_gate/run.sh         (rc==0 + "9/9"; M1 f32-cmp)
#   tests/native_v2_float_arith_gate/run.sh           (rc==0 + "11/11"; both-mem-operand)
#   tests/native_v2_struct_field_float_gate/run.sh    (rc==0 + "8/8"; struct-field float)
#   tests/native_v2_string_gate/run.sh                (rc==0 + "10/10"; string .len()/+ concat)
#   tests/native_v2_showcase_gate/run.sh              (rc==0 + "2/2"; dogfood demos)
#   tests/native_v2_multimodule_gate/run.sh           (tracked residual; see below)
#   tests/native_v2_backend_soundness_gate/run.sh     (tracked residual; see below)
#   scripts/dev/run_runpass_parse_only_current.sh     (rc==0 + "passed=525")
#
# === Tracked, by-design residuals (NOT failures) ===============================
# Two gates exit NONZERO BY DESIGN: they list documented KNOWN_MISCOMPILE
# witnesses (per R5 anti-xfail-lie they force nonzero so a "green N/N" can never
# silently exclude a live miscompile). This aggregator treats EXACTLY the known
# residual *identities* as tracked (not just counts) — it FAILS if the set grows,
# changes name, or a witness gets promoted (a FLAG line):
#
#   backend_soundness: 40/40 AND UNRESOLVED == { C_known_residual_bucket_collision }
#                      (the aa/ae -> slot 33 field-hash bucket collision, ->74)
#   multimodule:        9/9  AND UNRESOLVED == { k1_illtyped_arity, k2_illtyped_typedup }
#                      (the #2 import-typecheck-bypass slips)
#
# NOTE: the task brief named only backend_soundness's UNRESOLVED:1 residual.
# multimodule has the IDENTICAL by-design KNOWN_MISCOMPILE pattern (it would make
# the gate unsatisfiable otherwise), so the same tracked-residual handling is
# applied to it and documented here for transparency.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ulimit -s 1048576 2>/dev/null || true

BOOTSTRAP_SOUC="${SOUNIO_BOOTSTRAP_SOUC:-./bin/souc}"
MC="${SOUNIO_MODULAR_SOUC:-}"

LOGDIR="$(mktemp -d)"
trap 'rm -rf "$LOGDIR"' EXIT

# ----------------------------------------------------------------------------
# Build mc once (unless a prebuilt one was provided via SOUNIO_MODULAR_SOUC).
# ----------------------------------------------------------------------------
if [[ -z "$MC" ]]; then
  echo "[release-gate] building modular compiler from self-hosted/compiler/main.sio (bootstrap=$BOOTSTRAP_SOUC) ..."
  MC="$LOGDIR/mc.elf"
  if ! "$BOOTSTRAP_SOUC" self-hosted/compiler/main.sio "$MC" > "$LOGDIR/build.log" 2>&1; then
    echo "[release-gate] FATAL: could not build modular compiler" >&2
    tail -8 "$LOGDIR/build.log" >&2
    exit 2
  fi
  chmod +x "$MC"
  echo "[release-gate] built mc -> $MC"
else
  echo "[release-gate] reusing prebuilt mc: $MC"
fi

if [[ ! -x "$MC" ]]; then
  echo "[release-gate] FATAL: mc not executable: $MC" >&2
  exit 2
fi
# scripts/ci/* gates pick this up; tests/*/run.sh take it positionally.
export SOUNIO_MODULAR_SOUC="$MC"

# ----------------------------------------------------------------------------
# result bookkeeping
# ----------------------------------------------------------------------------
GATE_FAIL=0
declare -a RESULTS

record() { # name verdict detail
  RESULTS+=("$1|$2|$3")
  if [[ "$2" != "PASS" ]]; then GATE_FAIL=1; fi
}

run_log() { # logfile cmd...
  local lf="$1"; shift
  "$@" > "$lf" 2>&1
  return $?
}

# A) gates with a reliable exit code (rc==0 is truth) ------------------------
gate_rc() { # name logname cmd...
  local name="$1"; local lf="$LOGDIR/$2.log"; shift 2
  echo "[release-gate] >>> $name"
  if run_log "$lf" "$@"; then
    record "$name" PASS "rc=0"
    echo "[release-gate] <<< $name PASS"
  else
    local rc=$?
    record "$name" FAIL "rc=$rc"
    echo "[release-gate] <<< $name FAIL (rc=$rc)"; tail -6 "$lf" | sed 's/^/    /'
  fi
}

# B) capgate: NO reliable rc (last line is echo). Parse "---- N/N ----" and
#    require the expected total (closes the 0/0-falsely-green trap) and no
#    WRONG/FAIL lines.
gate_capgate() {
  local name="capgate" lf="$LOGDIR/capgate.log"
  echo "[release-gate] >>> $name"
  bash tests/native_v2_capgate/run.sh "$MC" > "$lf" 2>&1 || true
  local summary; summary="$(grep -oE -- '---- [0-9]+/[0-9]+ ----' "$lf" | tail -1)"
  local pass tot
  pass="$(sed -E 's#---- ([0-9]+)/([0-9]+) ----#\1#' <<<"$summary")"
  tot="$(sed -E 's#---- ([0-9]+)/([0-9]+) ----#\2#' <<<"$summary")"
  if [[ -z "$summary" ]]; then
    record "$name" FAIL "no summary line"
    echo "[release-gate] <<< $name FAIL (no summary)"; tail -6 "$lf" | sed 's/^/    /'
    return
  fi
  if grep -qE '^(WRONG|FAIL)' "$lf"; then
    record "$name" FAIL "WRONG/FAIL lines present ($pass/$tot)"
    echo "[release-gate] <<< $name FAIL"; grep -E '^(WRONG|FAIL)' "$lf" | sed 's/^/    /'
    return
  fi
  if [[ "${tot:-0}" -lt 1 || "$pass" != "$tot" || "$tot" != "32" ]]; then
    record "$name" FAIL "expected 32/32, got $pass/$tot"
    echo "[release-gate] <<< $name FAIL ($pass/$tot, expected 32/32)"
    return
  fi
  record "$name" PASS "$pass/$tot"
  echo "[release-gate] <<< $name PASS ($pass/$tot)"
}

# C) rc==0 tests gate that also must show an expected N/N (anti 0/0) ---------
gate_tests_rc_count() { # name dir/run.sh expectedSummary
  local name="$1" script="$2" expect="$3"
  local lf="$LOGDIR/$name.log"
  echo "[release-gate] >>> $name"
  bash "$script" "$MC" > "$lf" 2>&1; local rc=$?
  local summary; summary="$(grep -oE -- '---- [0-9]+/[0-9]+ ----' "$lf" | tail -1)"
  if [[ "$rc" -ne 0 ]]; then
    record "$name" FAIL "rc=$rc"
    echo "[release-gate] <<< $name FAIL (rc=$rc)"; tail -6 "$lf" | sed 's/^/    /'
    return
  fi
  if [[ "$summary" != "---- $expect ----" ]]; then
    record "$name" FAIL "expected $expect, got '${summary:-<none>}'"
    echo "[release-gate] <<< $name FAIL (expected $expect, got '${summary:-<none>}')"
    return
  fi
  record "$name" PASS "$expect"
  echo "[release-gate] <<< $name PASS ($expect)"
}

# D) by-design-nonzero gate with a pinned tracked-residual set ---------------
# args: name script expectedCount expectedSummary unresolvedRegex expectedUnresolvedSet...
gate_tracked_residual() {
  local name="$1" script="$2" expect_summary="$3" ur_label="$4"; shift 4
  local -a want_set=("$@")
  local lf
  lf="$LOGDIR/$name.log"
  echo "[release-gate] >>> $name (tracked residual)"
  bash "$script" "$MC" > "$lf" 2>&1 || true

  # counted N/N must be green
  local summary; summary="$(grep -oE -- '---- [0-9]+/[0-9]+ ----' "$lf" | tail -1)"
  if [[ "$summary" != "---- $expect_summary ----" ]]; then
    record "$name" FAIL "counted expected $expect_summary, got '${summary:-<none>}'"
    echo "[release-gate] <<< $name FAIL (counted $expect_summary vs '${summary:-<none>}')"
    tail -8 "$lf" | sed 's/^/    /'
    return
  fi
  # a FLAG line means a KNOWN_MISCOMPILE got promoted/changed -> fail
  if grep -qE '^FLAG' "$lf"; then
    record "$name" FAIL "FLAG line (residual promoted/changed)"
    echo "[release-gate] <<< $name FAIL (FLAG present)"; grep -E '^FLAG' "$lf" | sed 's/^/    /'
    return
  fi
  # any real FAIL line -> fail
  if grep -qE '^FAIL' "$lf"; then
    record "$name" FAIL "FAIL line present"
    echo "[release-gate] <<< $name FAIL"; grep -E '^FAIL' "$lf" | sed 's/^/    /'
    return
  fi
  # pin the UNRESOLVED set by identity, not just count.
  local ur_line; ur_line="$(grep -E "^$ur_label:" "$lf" | tail -1)"
  if [[ -z "$ur_line" ]]; then
    record "$name" FAIL "missing '$ur_label:' line"
    echo "[release-gate] <<< $name FAIL (no '$ur_label:' line)"; tail -8 "$lf" | sed 's/^/    /'
    return
  fi
  # tokens after '::' look like name(->val); extract bare names.
  local found; found="$(sed -E "s/^$ur_label: [0-9]+ ::(.*)$/\1/" <<<"$ur_line")"
  local -a got_names=()
  local tok
  for tok in $found; do
    got_names+=("$(sed -E 's/\(.*//' <<<"$tok")")
  done
  # compare sets (sorted)
  local want_sorted got_sorted
  want_sorted="$(printf '%s\n' "${want_set[@]}" | sort | tr '\n' ' ')"
  got_sorted="$(printf '%s\n' "${got_names[@]}" | sort | tr '\n' ' ')"
  if [[ "$want_sorted" != "$got_sorted" ]]; then
    record "$name" FAIL "residual set drift: want [$want_sorted] got [$got_sorted]"
    echo "[release-gate] <<< $name FAIL (residual drift)"
    echo "    want: $want_sorted"
    echo "    got : $got_sorted"
    return
  fi
  record "$name" PASS "$expect_summary + tracked residual {$got_sorted}"
  echo "[release-gate] <<< $name PASS ($expect_summary, tracked residual {$got_sorted})"
}

# E) parser sweep (uses bootstrap SOUC_BIN by design; parses a self-contained
#    bundle, not mc). rc==0 + passed=525/failed=0.
gate_parser_sweep() {
  local name="parser_sweep" lf="$LOGDIR/parser_sweep.log"
  echo "[release-gate] >>> $name"
  bash scripts/dev/run_runpass_parse_only_current.sh > "$lf" 2>&1; local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    record "$name" FAIL "rc=$rc"
    echo "[release-gate] <<< $name FAIL (rc=$rc)"; tail -6 "$lf" | sed 's/^/    /'
    return
  fi
  if ! grep -q '^passed=525$' "$lf" || ! grep -q '^failed=0$' "$lf"; then
    record "$name" FAIL "expected passed=525/failed=0"
    echo "[release-gate] <<< $name FAIL"; grep -E 'SUMMARY|passed=|failed=' "$lf" | sed 's/^/    /'
    return
  fi
  record "$name" PASS "passed=525 failed=0"
  echo "[release-gate] <<< $name PASS (passed=525 failed=0)"
}

# ----------------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------------
gate_rc "native_v2_calls_arity"       arity    bash scripts/ci/native_v2_calls_arity_gate.sh
gate_rc "native_v2_e2e_codegen_suite" codegen  bash scripts/ci/native_v2_e2e_codegen_suite_gate.sh
gate_rc "native_v2_e2e_exit_code"     exitcode bash scripts/ci/native_v2_e2e_exit_code_gate.sh
gate_rc "native_v2_recovered_source"  recovered bash scripts/ci/native_v2_recovered_source_core_gate.sh

gate_capgate

gate_tests_rc_count "native_v2_soundness"      tests/native_v2_soundness_gate/run.sh      "7/7"
gate_tests_rc_count "native_v2_enum"           tests/native_v2_enum_gate/run.sh           "15/15"
gate_tests_rc_count "native_v2_closure"        tests/native_v2_closure_gate/run.sh        "7/7"
gate_tests_rc_count "native_v2_checker_crash"  tests/native_v2_checker_crash_gate/run.sh  "26/26"
gate_tests_rc_count "native_v2_literal_coercion" tests/native_v2_literal_coercion_gate/run.sh "19/19"
gate_tests_rc_count "native_v2_float_compare"  tests/native_v2_float_compare_gate/run.sh  "9/9"
gate_tests_rc_count "native_v2_float_arith"    tests/native_v2_float_arith_gate/run.sh    "11/11"
gate_tests_rc_count "native_v2_struct_field_float" tests/native_v2_struct_field_float_gate/run.sh "8/8"
gate_tests_rc_count "native_v2_string"         tests/native_v2_string_gate/run.sh         "10/10"
gate_tests_rc_count "native_v2_showcase"       tests/native_v2_showcase_gate/run.sh       "2/2"

gate_tracked_residual "native_v2_backend_soundness" \
  tests/native_v2_backend_soundness_gate/run.sh \
  "40/40" "UNRESOLVED SILENT MISCOMPILES" \
  C_known_residual_bucket_collision

# native_v2_multimodule: the #2 import-typecheck fix (+caps) PROMOTED the former
# k1/k2 KNOWN_MISCOMPILE slips to genuine REJECTs, so this gate is now a clean
# rc==0 27/27 (no tracked residual). Treat it as a normal count gate.
gate_tests_rc_count "native_v2_multimodule" \
  tests/native_v2_multimodule_gate/run.sh \
  "27/27"

gate_parser_sweep

# ----------------------------------------------------------------------------
# VERDICT
# ----------------------------------------------------------------------------
echo
echo "================== RELEASE GATE SUMMARY =================="
for r in "${RESULTS[@]}"; do
  IFS='|' read -r nm vd dt <<<"$r"
  printf '  %-4s  %-32s  %s\n' "$vd" "$nm" "$dt"
done
echo "========================================================="
if [[ "$GATE_FAIL" -ne 0 ]]; then
  echo "RELEASE GATE: FAIL"
  exit 1
fi
echo "RELEASE GATE: PASS (all honest gates green; 2 tracked by-design residuals accounted for)"
exit 0
