#!/usr/bin/env bash
# scripts/ci/run_pass_output_gate.sh
#
# Output-checking gate for the run-pass corpus. For every tests/run-pass/*.sio
# that declares `//@ expect-stdout: <text>`, this:
#   1. compiles it with souc  (`$SOUC_BIN compile <src> -o <elf>`),
#   2. runs the ELF and captures stdout,
#   3. compares stdout (trailing-newline-trimmed) against the declared text.
#
# WHY THIS EXISTS — it is the ONE check that catches a move-codegen / in-place-SRET
# *aliasing miscompile*. Such a bug self-reproduces clean (fixed point converges),
# compiles main.sio with 0 errors, yet emits WRONG RUNTIME OUTPUT. Byte-identical
# fixed-point gates do NOT see it; only running the corpus and diffing stdout does.
# See .claude/plans/glowing-jingling-boole.md (Step 0) and memory
# project-sota-linearity-move-arch.
#
# MODES
#   (default)      compare current run against the recorded baseline; exit 1 on any
#                  REGRESSION (a test that PASSed in the baseline now MISMATCHes,
#                  fails to compile, or fails to run). Pre-existing failures do not
#                  fail the gate. With no baseline present, falls back to STRICT.
#   --baseline     record the current per-test status to the baseline file and
#                  exit 0 (run this with the current canonical souc BEFORE editing
#                  codegen, to freeze the reference).
#   --strict       ignore the baseline; exit 1 if ANY expect-stdout test does not
#                  produce its expected output.
#
# Statuses per test: PASS | MISMATCH | COMPILE-ERR | RUN-ERR
# The move-codegen danger transition is PASS -> {MISMATCH,RUN-ERR}; that is a
# REGRESSION and fails the gate.
#
# Exit 0 = PASS (no regression / baseline recorded). Exit 1 = regression / strict failure.
# Exit 0 with SKIP message on non-Linux/non-x86-64 hosts.

set -uo pipefail   # NOTE: not -e; we drive control flow on per-test exit codes.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

case "$(uname -s 2>/dev/null || echo unknown)/$(uname -m 2>/dev/null || echo unknown)" in
  Linux/x86_64|Linux/amd64) ;;
  *) echo "[run-pass-output] SKIP: x86-64 Linux only"; exit 0 ;;
esac

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

MODE="compare"
case "${1:-}" in
  --baseline) MODE="baseline" ;;
  --strict)   MODE="strict" ;;
  "")         MODE="compare" ;;
  *) echo "[run-pass-output] unknown arg: $1 (expected --baseline|--strict|<none>)"; exit 2 ;;
esac

BASELINE_FILE="${RUN_PASS_OUTPUT_BASELINE:-$ROOT_DIR/artifacts/omega/run_pass_output_baseline.txt}"
TEST_TIMEOUT="${RUN_PASS_TEST_TIMEOUT:-20}"   # seconds per compiled test, guards hangs
TMP="$(mktemp -d /tmp/run-pass-output.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

CUR="$TMP/current_status.txt"   # "name<TAB>STATUS" lines, sorted by name
: > "$CUR"

n_pass=0 n_mismatch=0 n_compile_err=0 n_run_err=0 n_total=0

# Iterate deterministically. Only files that declare an expected stdout are checkable.
for src in $(ls "$ROOT_DIR"/tests/run-pass/*.sio 2>/dev/null | sort); do
  exp_line="$(grep -m1 '//@ expect-stdout:' "$src" 2>/dev/null || true)"
  [ -z "$exp_line" ] && continue
  name="$(basename "$src" .sio)"
  # Everything after the marker, with one optional leading space trimmed.
  expect="${exp_line#*//@ expect-stdout:}"
  expect="${expect# }"
  n_total=$((n_total + 1))

  elf="$TMP/$name.elf"
  # The self-hosted souc takes positional `<src> <outbin>` (same form the
  # load-bearing lean_single_fixed_point_gate.sh uses); it has no `compile -o`
  # subcommand. The emitted ELF is not +x — chmod it before running.
  if ! "$SOUC_BIN" "$src" "$elf" >"$TMP/$name.build.log" 2>&1; then
    printf '%s\t%s\n' "$name" "COMPILE-ERR" >> "$CUR"
    n_compile_err=$((n_compile_err + 1))
    continue
  fi
  chmod +x "$elf" 2>/dev/null || true

  # Capture stdout; command substitution trims trailing newlines, matching how the
  # reference gate (ontology_typed_bridge_gate.sh) compares.
  if out="$(timeout "$TEST_TIMEOUT" "$elf" 2>"$TMP/$name.run.err")"; then
    # Convention in this corpus: `expect-stdout` declares that the expected string
    # appears as a complete output LINE — it may be the whole output, or one line
    # among several (the contract line is not always last, e.g. closure_*_hof).
    # souc's print() emits NO trailing newline, so some tests concatenate their
    # whole report onto one line ending in the verdict (associator_field_pentagon,
    # tac_compile_gate_pass) — for those (single-line only) accept a trailing
    # suffix. Multi-line outputs must contain expect as an exact line, so a genuine
    # FAIL (ptx_maxntid: "FAIL" vs want "PASS") is NOT masked. (out already has
    # trailing newlines stripped by command substitution.)
    matched=0
    if [ "$out" = "$expect" ]; then
      matched=1
    elif printf '%s\n' "$out" | grep -Fxq -- "$expect"; then
      matched=1                                   # expect is an exact \n-delimited line
    elif [ "${out#*$'\n'}" = "$out" ]; then       # single-line output (no newline)
      case "$out" in *"$expect") matched=1 ;; esac # ...ends with expect (no-newline quirk)
    fi
    if [ "$matched" -eq 1 ]; then
      printf '%s\t%s\n' "$name" "PASS" >> "$CUR"
      n_pass=$((n_pass + 1))
    else
      printf '%s\t%s\n' "$name" "MISMATCH" >> "$CUR"
      n_mismatch=$((n_mismatch + 1))
      printf '  MISMATCH %-44s got=[%s] want=[%s]\n' "$name" "$out" "$expect"
    fi
  else
    rc=$?
    printf '%s\t%s\n' "$name" "RUN-ERR" >> "$CUR"
    n_run_err=$((n_run_err + 1))
    printf '  RUN-ERR  %-44s exit=%s want=[%s]\n' "$name" "$rc" "$expect"
  fi
done

sort -o "$CUR" "$CUR"

echo "[run-pass-output] souc=$SOUC_BIN"
printf '[run-pass-output] checked=%d  PASS=%d  MISMATCH=%d  COMPILE-ERR=%d  RUN-ERR=%d\n' \
  "$n_total" "$n_pass" "$n_mismatch" "$n_compile_err" "$n_run_err"

if [ "$MODE" = "baseline" ]; then
  # A baseline of zero tests blesses every future empty comparison; the scan
  # must have selected something before its verdict is worth recording.
  if [ "$n_total" -lt 1 ]; then
    echo "[run-pass-output] FAIL: no expect-stdout tests found -- the corpus scan answered nothing (marker drift or missing corpus?)" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$BASELINE_FILE")"
  cp "$CUR" "$BASELINE_FILE"
  echo "[run-pass-output] BASELINE recorded -> $BASELINE_FILE ($n_total tests)"
  exit 0
fi

if [ "$MODE" = "strict" ] || [ ! -f "$BASELINE_FILE" ]; then
  [ "$MODE" != "strict" ] && echo "[run-pass-output] no baseline at $BASELINE_FILE -> STRICT mode"
  if [ "$n_total" -lt 1 ]; then
    echo "[run-pass-output] FAIL (strict): no expect-stdout tests found -- the corpus scan answered nothing (marker drift or missing corpus?)" >&2
    exit 1
  fi
  if [ "$n_mismatch" -eq 0 ] && [ "$n_run_err" -eq 0 ] && [ "$n_compile_err" -eq 0 ]; then
    echo "[run-pass-output] PASS (strict): all $n_total expect-stdout tests produce expected output"
    exit 0
  fi
  echo "[run-pass-output] FAIL (strict): $n_mismatch mismatch, $n_run_err run-err, $n_compile_err compile-err"
  exit 1
fi

# compare mode: a REGRESSION is a test that was PASS in the baseline but is now not PASS.
regressions=0
while IFS=$'\t' read -r name base_status; do
  [ "$base_status" = "PASS" ] || continue
  cur_status="$(grep -P "^${name}\t" "$CUR" | head -1 | cut -f2)"
  if [ "$cur_status" != "PASS" ]; then
    printf '  REGRESSION %-42s baseline=PASS now=%s\n' "$name" "${cur_status:-MISSING}"
    regressions=$((regressions + 1))
  fi
done < "$BASELINE_FILE"

if [ "$regressions" -eq 0 ]; then
  echo "[run-pass-output] PASS: no regression vs baseline ($BASELINE_FILE)"
  exit 0
fi
echo "[run-pass-output] FAIL: $regressions regression(s) vs baseline — a PASS test now miscompiles"
exit 1
