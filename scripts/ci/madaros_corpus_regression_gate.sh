#!/usr/bin/env bash
# Run the run-pass corpus under the MODULAR MADAROS compiler and fail only on
# regressions against a checked-in baseline.
#
# Why this exists
# ---------------
# CI's `full-test-suite` runs souc-stage2 (lean_single), the frozen bootstrap
# seed. Most language guarantees live in the modular Madaros compiler, which
# lean_single does not implement. Measured on 2026-07-26: three silent
# miscompiles (#1454/#1194, #1474, #1475) were fully green in CI while Madaros
# computed wrong answers -- and one was corrupting a dissertation-path PBPK
# variance decomposition into the wrong pharmacological conclusion.
#
# The pre-existing Madaros lane (`madaros_changed_tests_gate.sh`) runs only
# tests that BOTH changed in the PR and carry `//@ requires: madaros`, and skips
# entirely when there are none. It could not have caught any of those defects,
# because they broke existing, unchanged tests.
#
# Why a baseline instead of "must be all green"
# ---------------------------------------------
# Madaros is less mature than lean_single on several paths, so the corpus has
# genuine pre-existing failures under it. A gate that is red on arrival is a
# gate everyone learns to ignore. This compares the failure LIST against a
# checked-in baseline and blocks only on entries that are NEW -- the same
# discipline this repository already applies by hand ("compare the list, not the
# totals", because pass/known-failure totals wobble run to run on an unchanged
# tree).
#
# Entries that disappear are reported but do not block: fixing a test must never
# be what turns CI red.
#
# Usage
#   SOUNIO_MADAROS_CORPUS_BIN=/path/to/madaros bash scripts/ci/madaros_corpus_regression_gate.sh
#   SOUNIO_MADAROS_CORPUS_REFRESH=1 ...   # rewrite the baseline instead of comparing

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASELINE="tests/madaros_corpus_baseline.txt"
MADAROS="${SOUNIO_MADAROS_CORPUS_BIN:-}"
REFRESH="${SOUNIO_MADAROS_CORPUS_REFRESH:-0}"
JOBS="${SOUNIO_TEST_JOBS:-4}"

fail() {
  echo "[madaros-corpus] FAIL: $*" >&2
  exit 1
}

[[ -n "$MADAROS" ]] || fail "SOUNIO_MADAROS_CORPUS_BIN must name a current-source Madaros ELF"
[[ -x "$MADAROS" ]] || fail "not executable: $MADAROS"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-madaros-corpus.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

echo "[madaros-corpus] compiler: $MADAROS"
"$MADAROS" --version 2>&1 | head -1 | sed 's/^/[madaros-corpus] /'

# Only run-pass programs. compile-fail and typecheck-fail tests have their own
# gates and their verdicts are engine-specific by design.
# Exclude files that are not programs. Library leaves and `//@ ignore` files
# compile fine and then die with SIGSEGV because the ELF has no entry point --
# 11 of them sat in the baseline as `run` failures, saying nothing about the
# compiler. Same blind spot the parity gate had (#1601, #1593).
#
# Only these two exclusions. `//@ check-only` is deliberately NOT one: measured
# on the parity side, 15 check-only files also declare //@ run-pass and have a
# main, and one check-only file with no run-pass executes and agrees across both
# engines. The marker says which harness checks the file, not whether it runs.
#
# Filtered HERE rather than inside run_one.sh so the count below is the number
# actually exercised. Reporting 1699 while skipping 11 of them is the kind of
# misleading instrument this gate exists to catch.
corpus_is_program() {
    local f="$1"
    head -n 8 "$f" | grep -qE '^//@[[:space:]]*ignore\b' && return 1
    grep -qE '^[[:space:]]*(pub[[:space:]]+)?fn[[:space:]]+main[[:space:]]*\(' "$f"
}
mapfile -t ALL_SIO < <(ls tests/run-pass/*.sio 2>/dev/null | sort)
PROGRAMS=()
skipped=0
for f in "${ALL_SIO[@]}"; do
    if corpus_is_program "$f"; then PROGRAMS+=("$f"); else skipped=$((skipped + 1)); fi
done
[[ ${#PROGRAMS[@]} -gt 0 ]] || fail "no programs found under tests/run-pass"
echo "[madaros-corpus] programs: ${#PROGRAMS[@]} (skipped $skipped: //@ ignore or no fn main)"

cat > "$WORK/run_one.sh" <<'INNER'
#!/usr/bin/env bash
src="$1"
name="$(basename "$src")"
# Heartbeat: written before the early exits below, so every selected program
# answers exactly once regardless of verdict. The gate asserts on this file
# because actual.txt only records failures -- an empty actual.txt must mean
# "all passed", never "the instrument answered nothing".
printf '%s\n' "$name" >> "$WORK/ran.txt"
elf="$WORK/${name%.sio}.elf"

# A test declaring an environment feature we are not providing is not a failure.
if grep -qE '^//@ requires: (gpu|llvm)' "$src" 2>/dev/null; then exit 0; fi
# Tests the repo already declares as known failures are not regressions.
if grep -qE '^//@ known-failure' "$src" 2>/dev/null; then exit 0; fi

if ! "$MADAROS" compile "$src" -o "$elf" >/dev/null 2>&1; then
  echo "$name compile"
  exit 0
fi
chmod +x "$elf" 2>/dev/null
out="$("$WORK/timeout_run.sh" "$elf")" || { echo "$name run"; exit 0; }

# If the test declares an expected marker, assert it. Exit status alone is not
# evidence: a CPC 2026 receipt was found compiling, exiting 0 and printing ZERO
# BYTES under Madaros while lean_single printed the full receipt, and this gate
# reported it as newly fixed (#1498). A program that produces nothing scored the
# same as one that produced the right answer.
marker="$(sed -n 's|^//@ expect-stdout:[[:space:]]*||p' "$src" | head -1)"
if [ -n "$marker" ]; then
  case "$out" in
    *"$marker"*) ;;
    *) echo "$name stdout" ;;
  esac
fi
exit 0
INNER
chmod +x "$WORK/run_one.sh"

cat > "$WORK/timeout_run.sh" <<'TR'
#!/usr/bin/env bash
timeout 30 "$1" 2>/dev/null
TR
chmod +x "$WORK/timeout_run.sh"
export MADAROS WORK

: > "$WORK/ran.txt"
printf '%s\n' "${PROGRAMS[@]}" \
  | xargs -P "$JOBS" -I{} "$WORK/run_one.sh" {} \
  | sort > "$WORK/actual.txt"

# Completeness floor: every selected program must have heartbeated. A dead
# xargs or a broken run_one leaves actual.txt empty, which the comparison
# below would read as "no new failures" -- zero evidence reading green.
sort -u -o "$WORK/ran.txt" "$WORK/ran.txt"
RAN_COUNT="$(grep -c . "$WORK/ran.txt" || true)"
if [[ "$RAN_COUNT" -ne "${#PROGRAMS[@]}" ]]; then
  printf '%s\n' "${PROGRAMS[@]##*/}" | sort > "$WORK/expected.txt"
  comm -13 "$WORK/ran.txt" "$WORK/expected.txt" > "$WORK/unanswered.txt"
  fail "incomplete run: $RAN_COUNT of ${#PROGRAMS[@]} programs answered; $(wc -l < "$WORK/unanswered.txt" | tr -d ' ') silent (first 10): $(head -10 "$WORK/unanswered.txt" | tr '\n' ' ')"
fi

ACTUAL_COUNT="$(wc -l < "$WORK/actual.txt" | tr -d ' ')"
echo "[madaros-corpus] failures observed: $ACTUAL_COUNT / ${#PROGRAMS[@]}"

if [[ "$REFRESH" == "1" ]]; then
  {
    echo "# Failure baseline for tests/run-pass under the modular Madaros compiler."
    echo "#"
    echo "# Regenerate:"
    echo "#   SOUNIO_MADAROS_CORPUS_BIN=<madaros> SOUNIO_MADAROS_CORPUS_REFRESH=1 \\"
    echo "#     bash scripts/ci/madaros_corpus_regression_gate.sh"
    echo "#"
    echo "# One entry per failing program:
#   '<name>.sio compile'  -- did not compile
#   '<name>.sio run'      -- compiled, non-zero exit or timeout
#   '<name>.sio stdout'   -- ran fine but did not print its //@ expect-stdout
#                            marker (165 of 1688 tests declare one)"
    echo "# These are PRE-EXISTING Madaros failures, NOT approvals. Shrinking this"
    echo "# file is the point; the gate blocks only on entries that are NEW."
    cat "$WORK/actual.txt"
  } > "$BASELINE"
  echo "[madaros-corpus] baseline refreshed: $BASELINE ($ACTUAL_COUNT entries)"
  exit 0
fi

[[ -f "$BASELINE" ]] || fail "missing baseline $BASELINE -- generate it with SOUNIO_MADAROS_CORPUS_REFRESH=1"

grep -vE '^[[:space:]]*#' "$BASELINE" | grep -vE '^[[:space:]]*$' | sort > "$WORK/baseline.txt"

comm -13 "$WORK/baseline.txt" "$WORK/actual.txt" > "$WORK/new.txt"
comm -23 "$WORK/baseline.txt" "$WORK/actual.txt" > "$WORK/fixed.txt"

FIXED_COUNT="$(wc -l < "$WORK/fixed.txt" | tr -d ' ')"
if [[ "$FIXED_COUNT" -gt 0 ]]; then
  echo "[madaros-corpus] $FIXED_COUNT baseline entries now PASS -- refresh the baseline to lock them in:"
  sed 's/^/    + /' "$WORK/fixed.txt"
fi

NEW_COUNT="$(wc -l < "$WORK/new.txt" | tr -d ' ')"
if [[ "$NEW_COUNT" -gt 0 ]]; then
  echo "[madaros-corpus] $NEW_COUNT NEW failure(s) under Madaros:" >&2
  sed 's/^/    - /' "$WORK/new.txt" >&2
  echo "" >&2
  echo "These pass on the baseline and fail on this change. CI's full-test-suite" >&2
  echo "runs lean_single and will not show this." >&2
  fail "regression under the modular Madaros compiler"
fi

echo "[madaros-corpus] PASS: no new failures under Madaros ($ACTUAL_COUNT known, $FIXED_COUNT newly fixed)"
