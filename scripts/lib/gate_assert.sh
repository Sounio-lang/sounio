#!/usr/bin/env bash
# Shared assertions for CI gates. Source it:
#
#     . "$(dirname "${BASH_SOURCE[0]}")/../lib/gate_assert.sh"
#     gate_name "my_gate"
#
# BEFORE YOU ASSERT ANYTHING: the file you compiled exists and starts with
# \x7fELF; the compile log names the engine that actually ran; the rc you
# hold was written by that process to a file; a missing tool is SKIPPED
# (rc!=0, measured=no), never a green 0. An empty or wrong artefact
# read as success is not a measurement.
#
# WHY THIS EXISTS. A census on 2026-08-04 over the 417 *_gate.{sh,py} in
# scripts/ci and scripts/dev found:
#
#     258 (67%)  no emptiness guard of any kind
#      46 (12%)  an explicit check that the gate is still reading real content
#     125        define their own private fail()
#       0        shared assertion library
#
# The 12% is concentrated in two families. Outside them, a gate that stops
# finding its subject usually reports success. Measured examples on that date:
# run_pass_output_gate.sh printed "PASS (strict): all 0 ... tests" if the
# //@ expect-stdout: marker were renamed; reproduce_artifact.sh printed
# "PASS: test suite" when the suite crashed on startup; the *anti-vacuity lane*
# of eisa_bridge_conformance_gate.sh printed "PASS anti-vacuity" when its own
# extraction came back empty.
#
# The failure is always the same shape: AN EMPTY ANSWER READ AS A MEANINGFUL
# ONE. These helpers make the empty case loud.

if [[ -n "${_SOUNIO_GATE_ASSERT_SOURCED:-}" ]]; then return 0; fi
_SOUNIO_GATE_ASSERT_SOURCED=1

_GATE_NAME="${_GATE_NAME:-gate}"

gate_name() { _GATE_NAME="$1"; }

gate_fail() {
  echo "$(printf '%s' "$_GATE_NAME" | tr '[:lower:]' '[:upper:]')_FAIL: $*" >&2
  exit 1
}

gate_pass() {
  echo "$(printf '%s' "$_GATE_NAME" | tr '[:lower:]' '[:upper:]')_OK${1:+: $1}"
}

# Cursor-2's measured=no column: a gate that lacks ptxas/libcuda and
# `exit 0` is skip-vacuous — published numbers, nothing measured.
# SKIPPED is not green. 77 is GNU automake's skip; workflows that treat
# 0 as pass will see this as a fail unless they handle 77 explicitly.
GATE_SKIPPED_RC=77

gate_skip() {
  echo "$(printf '%s' "$_GATE_NAME" | tr '[:lower:]' '[:upper:]')_SKIPPED: $*"
  echo "measured=no"
  exit "$GATE_SKIPPED_RC"
}

# require_tool <cmd> [why]
# Missing hardware/toolchain is a skip, not a pass.
require_tool() {
  local cmd="$1" why="${2:-missing tool: $1}"
  command -v "$cmd" >/dev/null 2>&1 || gate_skip "$why"
}

require_file() {
  [[ -f "$1" ]] || gate_fail "${2:-missing file: $1}"
}

require_executable() {
  [[ -x "$1" ]] || gate_fail "${2:-not executable: $1}"
}

# The one that would have caught most of the census. Use it on every value
# pulled out of a tool's output before comparing it to anything.
require_nonempty() {
  [[ -n "${1:-}" ]] || gate_fail "${2:-a value the gate depends on came back empty — it is not measuring what it claims}"
}

require_nonempty_file() {
  require_file "$1" "${2:-}"
  [[ -s "$1" ]] || gate_fail "${2:-$1 is empty — whatever produced it produced nothing}"
}

# Counting that distinguishes "no match" from "the tool broke".
#
# `grep -c ... || true` — the dominant idiom in this repo — collapses rc 1
# (no match, count really is 0) and rc 2 (bad regex, unreadable file, tool
# missing) into the same 0. Modelled on scripts/ci/assert_no_rust_markers.sh:38-45,
# the only place in the tree that already gets this right.
#
#   count_matches <pattern> <file> [--fixed]
count_matches() {
  local pattern="$1" file="$2" mode="${3:-}" count rc
  require_file "$file" "count_matches: no such file: $file"
  set +e
  if [[ "$mode" == "--fixed" ]]; then
    count="$(grep -F -c -- "$pattern" "$file")"
  else
    count="$(grep -E -c -- "$pattern" "$file")"
  fi
  rc=$?
  set -e
  case "$rc" in
    0) ;;
    1) count=0 ;;
    *) gate_fail "failed to scan $file for '$pattern' (grep rc=$rc) — this is a tool error, not a zero count" ;;
  esac
  printf '%s' "$count"
}

# A floor on how much the gate saw. This is the anti-vacuity primitive: a sweep
# that finds nothing to check must be red, not green.
require_min_count() {
  local actual="$1" minimum="$2" subject="${3:-items}"
  require_nonempty "$actual" "count of $subject came back empty"
  [[ "$actual" -ge "$minimum" ]] \
    || gate_fail "only $actual $subject (expected at least $minimum) — the gate found nothing to measure, which is not the same as everything passing"
}

# Text anchors. Folds three near-identical private copies that were living in
# madaros_operational_contract_gate.sh:18-31,
# exact_bitwise_rebracket_source_ir_gate.sh:29-41 and
# ordered_path_provenance_source_ir_gate.sh:33-45.
require_text() {
  local pattern="$1" file="$2"
  require_file "$file"
  grep -Fq -- "$pattern" "$file" \
    || gate_fail "missing required anchor in $file: $pattern"
}

require_text_regex() {
  local pattern="$1" file="$2"
  require_file "$file"
  grep -Eq -- "$pattern" "$file" \
    || gate_fail "missing required anchor in $file: /$pattern/"
}

# Run a command and require it to FAIL. The other half of an instrument: a check
# that only ever confirms the good case has not been shown to discriminate.
require_rejects() {
  local why="$1"; shift
  if "$@" >/dev/null 2>&1; then
    gate_fail "the check accepted $why — it is not measuring that"
  fi
}

# ---------------------------------------------------------------------------
# 2026-08-18 — ELF / engine / rc. Census in
# docs/audit/GATE_INSTRUMENT_CENSUS_2026-08-18.md
#
# Pattern 2 (rc through a pipe) is empty in scripts/ci. Patterns 1 and 3 are
# not: 98 compile-invoking gates never check \x7fELF; 72 never record which
# engine compiled. That is a helper, not 98 patches. New gates source this
# file and call require_elf / classify_compile_log / gate_capture_rc.
# ---------------------------------------------------------------------------

# require_elf <path> [why]
# Exists, non-empty, first four bytes are \x7fELF. A compile that exits 0
# and writes a file named `-o` (the wrapper swallow, #1885) is the case
# this refuses. `[[ -s ]]` alone is not enough: a non-empty `-o` is still
# not the named ELF.
require_elf() {
  local path="$1" why="${2:-$1 is not a native ELF}"
  [[ -e "$path" ]] || gate_fail "$why (missing: $path)"
  [[ -s "$path" ]] || gate_fail "$why (empty: $path)"
  local magic
  magic="$(od -An -tx1 -N4 "$path" | tr -d ' \n')"
  [[ "$magic" == "7f454c46" ]] \
    || gate_fail "$why (magic=$magic, want 7f454c46; $path)"
}

# classify_compile_log <compile-log>
# Prints madaros | lean_single | unknown.
#
# MUST read the compile log, not `souc --version`. The wrapper prints
# Madaros on --version even when `souc src.sio -o dest` routed to
# lean_single and wrote a file named `-o`.
classify_compile_log() {
  local log="$1"
  require_file "$log" "classify_compile_log: no compile log: $log"
  if grep -qE '(^Madaros |Compilation successful)' "$log"; then
    printf 'madaros'
  elif grep -qE '(^source: |^tokens:|^elf: )' "$log"; then
    printf 'lean_single'
  else
    printf 'unknown'
  fi
}

# require_compile_engine <compile-log> <expected>
require_compile_engine() {
  local log="$1" expected="$2" got
  got="$(classify_compile_log "$log")"
  require_nonempty "$got" "classify_compile_log returned empty"
  [[ "$got" == "$expected" ]] \
    || gate_fail "compile log named engine=$got (want $expected) — $log"
}

# gate_capture_rc <dest-file> -- <cmd...>
# Writes the child's rc to dest-file. Never read rc through a pipe: the
# last command in the pipe is grep/tee/awk, and that gate is always green.
gate_capture_rc() {
  local dest="$1"
  shift
  [[ "${1:-}" == "--" ]] && shift
  [[ $# -ge 1 ]] || gate_fail "gate_capture_rc: no command"
  local rc=0
  set +e
  "$@"
  rc=$?
  set -e
  printf '%s\n' "$rc" > "$dest"
}

# require_rc_file <dest-file> [expected]
# The file must exist and hold a decimal integer. If expected is set, it
# must match. An empty file is the pipe-grep miss from E230 round 3.
require_rc_file() {
  local dest="$1" expected="${2:-}"
  require_nonempty_file "$dest" "run_rc file missing/empty: $dest (rc was not captured — do not grep it out of a pipe)"
  local got
  got="$(cat "$dest")"
  [[ "$got" =~ ^[0-9]+$ ]] || gate_fail "run_rc file is not a number: $dest ($got)"
  if [[ -n "$expected" && "$got" != "$expected" ]]; then
    gate_fail "run_rc=$got (want $expected) from $dest"
  fi
}

# gate_measured_yes
# Cursor-2's other half: only print measured=yes after the artefact,
# engine, and rc checks have passed.
gate_measured_yes() {
  echo "measured=yes"
}
