#!/usr/bin/env bash
# classify_compile_selftest.sh — the classifier must fail closed under sabotage.
#
# Why this exists
# ---------------
# A gate that decides "the compiler refused this program" from `rc != 0` cannot
# tell a refusal from a segfault, a timeout, an OOM kill, or a missing binary.
# scripts/lib/classify_compile.sh exists to make that distinction; this gate
# exists to prove the distinction actually discriminates, rather than being a
# rename of the same permissiveness.
#
# The method is the one lean_single_pub_use_reexport_gate.sh already uses for
# its own predicate: synthesise the adversarial outcomes directly, rather than
# hoping the real corpus happens to contain them. No compiler runs here, so the
# gate costs ~1 s and cannot itself be confounded by a compiler defect.
#
# GATE_CONTRACT: v0
# GATE_ID: classify_compile_selftest
# GATE_CLAIMS: sounio_classify_compile assigns exactly one named class to every
#              outcome, and never classifies a crash, timeout, silent failure or
#              infrastructure fault as REFUSED
# GATE_ENGINE: none (pure shell, synthetic fixtures)
# GATE_RESULT_ON_SKIP: fail
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
# shellcheck source=../lib/classify_compile.sh
source "$ROOT_DIR/scripts/lib/classify_compile.sh"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/classify-selftest.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

fails=0
checks=0

# check LABEL EXPECTED_CLASS RC LOGTEXT [ARTIFACT_BYTES]
check() {
  local label="$1" want="$2" rc="$3" logtext="$4" bytes="${5:-0}"
  local log="$TMP_DIR/$label.log" out="$TMP_DIR/$label.out"
  checks=$((checks + 1))
  printf '%s\n' "$logtext" > "$log"
  rm -f "$out"
  if [[ "$bytes" -gt 0 ]]; then
    head -c "$bytes" /dev/zero > "$out"
  fi
  sounio_classify_compile "$rc" "$log" "$out"
  if [[ "$SOUNIO_CC_CLASS" != "$want" ]]; then
    printf 'FAIL  %-24s want=%-22s got=%-22s (%s)\n' \
      "$label" "$want" "$SOUNIO_CC_CLASS" "$SOUNIO_CC_DETAIL" >&2
    fails=$((fails + 1))
  else
    printf 'ok    %-24s %s\n' "$label" "$SOUNIO_CC_CLASS"
  fi
}

# expect_refusal_is LABEL EXPECT(yes|no) RC LOGTEXT RULE [ARTIFACT_BYTES]
expect_refusal_is() {
  local label="$1" expect="$2" rc="$3" logtext="$4" rule="${5:-}" bytes="${6:-0}"
  local log="$TMP_DIR/$label.log" out="$TMP_DIR/$label.out"
  checks=$((checks + 1))
  printf '%s\n' "$logtext" > "$log"
  rm -f "$out"
  if [[ "$bytes" -gt 0 ]]; then
    head -c "$bytes" /dev/zero > "$out"
  fi
  local got="no"
  if sounio_expect_refusal "$rc" "$log" "$out" "$rule"; then
    got="yes"
  fi
  if [[ "$got" != "$expect" ]]; then
    printf 'FAIL  %-24s expect_refusal want=%s got=%s (rule=%s class=%s)\n' \
      "$label" "$expect" "$got" "${rule:-none}" "$SOUNIO_CC_CLASS" >&2
    fails=$((fails + 1))
  else
    printf 'ok    %-24s expect_refusal=%s\n' "$label" "$got"
  fi
}

echo "CLASSIFY_COMPILE_SELFTEST_START"

# --- the four outcomes a64_compile_fail_parity_gate.sh cannot tell apart -----
# Under `rc != 0`, the first three are indistinguishable from each other.
check clean_refusal        REFUSED               1   'error[E218] in m::f at 3..9: reserved'
check both_segv            CRASHED             139   'Segmentation fault (core dumped)'
check timeout_124          TIMEOUT             124   ''
check sigkill_137          TIMEOUT             137   ''

# --- the raw-ELF pathology documented in scripts/ci/souc-native-wrapper.sh ---
# rc=0 with no success marker is a silent failure, not an acceptance. The 36447
# byte size is the actual size of the stub leaked into the repo root as `-o`.
check rc0_stub_artifact    SILENT_FAILURE        0   'parsed 3 items'          36447
check rc0_no_marker        SILENT_FAILURE        0   ''
check rc0_real_success     ACCEPTED              0   'compile: fns=42'          4096
check rc0_marker_no_elf    SILENT_FAILURE        0   'compile: fns=42'

# --- a crash masked behind a tidy exit code ---------------------------------
# The diagnostic is real AND the process died. It must not count as a refusal:
# nothing proves the check fired rather than the crash preempting it.
check masked_segv          CRASHED               1   'error[E001]: x
Segmentation fault (core dumped)'

# --- a refusal that leaked an artifact: contract violation, not a refusal ----
check refusal_with_elf     REFUSED_WITH_ARTIFACT 1   'error[E019]: nope'       36447

# --- rejected but unattributable ---------------------------------------------
check silent_reject        REFUSED_UNDIAGNOSED   1   ''

# --- infrastructure ----------------------------------------------------------
check not_found            INFRA               127   ''
check not_executable       INFRA               126   ''

# --- uncoded diagnostics still count -----------------------------------------
# A large fraction of lean_single.sio diagnostics carry no E-code.
check uncoded_diag         REFUSED               1   'error: unresolved function body for call target fn#7 g at line 12'
check verdict_marker_only  REFUSED               1   'typecheck: failed'

# --- rule matching must discriminate -----------------------------------------
expect_refusal_is rule_right_code    yes   1   'error[E218] in m::f at 1..2: reserved'  E218
expect_refusal_is rule_wrong_code    no    1   'error[E218] in m::f at 1..2: reserved'  E019
expect_refusal_is rule_message_frag  yes   1   'error[E218]: f128 is reserved'          'is reserved'
expect_refusal_is rule_no_constraint yes   1   'error[E218]: whatever'                  ''

# --- NEGATIVE CONTROLS -------------------------------------------------------
# Each of these carries the *correct* diagnostic and must still be refused as
# evidence, because the process did not survive to make the decision. Without
# these, the rule check above would pass vacuously.
expect_refusal_is nc_crash_right_code  no  139 'error[E218] reserved
Segmentation fault (core dumped)'                                                E218
expect_refusal_is nc_timeout_right_code no 124 'error[E218] reserved'            E218
expect_refusal_is nc_rc0_right_code     no   0 'error[E218] reserved'            E218
expect_refusal_is nc_artifact_right_code no  1 'error[E218] reserved'            E218  36447

# --- E-code extraction across the three emitter shapes -----------------------
printf 'error[E218] in m::f at 3..9: a\nerror[E170]: b at line 4\nE200 `x` at line 9\n' \
  > "$TMP_DIR/codes.log"
got_codes="$(sounio_diag_codes "$TMP_DIR/codes.log" | tr '\n' ',' | sed 's/,$//')"
checks=$((checks + 1))
if [[ "$got_codes" != "E170,E200,E218" ]]; then
  printf 'FAIL  %-24s want=E170,E200,E218 got=%s\n' "diag_codes_all_shapes" "$got_codes" >&2
  fails=$((fails + 1))
else
  printf 'ok    %-24s %s\n' "diag_codes_all_shapes" "$got_codes"
fi

checks=$((checks + 1))
if [[ "$(sounio_primary_diag "$TMP_DIR/codes.log")" != "E218" ]]; then
  printf 'FAIL  %-24s primary diag is emission order, not sort order\n' "primary_diag" >&2
  fails=$((fails + 1))
else
  printf 'ok    %-24s E218\n' "primary_diag"
fi

echo "checks=$checks failures=$fails"
if [[ "$fails" -ne 0 ]]; then
  echo "CLASSIFY_COMPILE_SELFTEST=FAIL ($fails of $checks)" >&2
  exit 1
fi
echo "CLASSIFY_COMPILE_SELFTEST_GATE_OK"
