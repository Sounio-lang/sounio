#!/usr/bin/env bash
# a64_compile_fail_parity_gate.sh — the arm64 backend must refuse what the
# x86 backend refuses.
#
# Why this exists
# ---------------
# `compile_all_arm64` re-emits pass 2 with its own mirror of the diagnostics,
# and until this gate's companion fix it never read TYPECHECK_FAILED: every
# error the arm64 pass raised was printed and then discarded, the compiler
# exited 0, and an ELF was written. `compile_all` cannot cover it either — it
# returns early at `if TARGET_ARCH == 1`, before its own gate.
#
# The divergence grew unseen because nothing ever ran the suite against
# --target aarch64-linux. scripts/dev/run_sio_test_suite_v2.sh has no concept
# of a target, and tests/selfhost/aarch64_compile/ carries 4 cases whose only
# assertion is that `file` reports an aarch64 ELF — an assertion that cannot,
# by construction, notice a wrongly accepted program.
#
# GATE_CONTRACT: v0
# GATE_ID: a64_compile_fail_parity
# GATE_CLAIMS: every tests/compile-fail case the x86 target refuses is also refused by aarch64-linux
# GATE_ENGINE: lean_single (bin/souc-lean-single-x86_64)
# GATE_RESULT_ON_SKIP: fail
#
# This is an ACCUSATION gate with a pinned allowance. The arm64 mirror still
# lacks checks the x86 path has — measured at 55 cases on 2026-09-04, listed in
# the baseline file. The gate FAILS if that number grows, and fails if any file
# NOT in the baseline diverges. Shrinking the baseline is the point; each
# removal belongs to a commit that adds the missing arm64 check.
#
# Do not add a case to the baseline to make the gate pass. A new divergence is
# a regression in the arm64 mirror, not a fact about the corpus.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC="${SOUNIO_LEAN_SINGLE_BIN:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
BASELINE="${BASELINE:-$ROOT_DIR/tests/selfhost/aarch64_compile/compile_fail_parity_baseline.txt}"
JOBS="${JOBS:-8}"
TIMEOUT_SECS="${TIMEOUT_SECS:-60}"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sounio-a64-parity.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

echo "A64_COMPILE_FAIL_PARITY_GATE_START"
echo "souc=$SOUC"
echo "baseline=$BASELINE"

[[ -x "$SOUC" ]] || { echo "error: missing lean_single compiler at $SOUC" >&2; exit 1; }
[[ -f "$BASELINE" ]] || { echo "error: missing baseline at $BASELINE" >&2; exit 1; }

# Compile one case for both targets and report the pair of exit codes.
# The compiler resolves `import` relative to the CURRENT DIRECTORY, so this
# must run from the repository root — running it elsewhere makes roughly 40 %
# of the corpus fail with E224 on both targets and silently reports parity.
probe() {
    local file="$1"
    local base rc_x86 rc_a64
    base="$(basename "$file" .sio)"
    rc_x86=0
    timeout "$TIMEOUT_SECS" "$SOUC" "$file" "$WORK_DIR/$base.x86" >/dev/null 2>&1 || rc_x86=$?
    rc_a64=0
    timeout "$TIMEOUT_SECS" "$SOUC" "$file" "$WORK_DIR/$base.a64" --target aarch64-linux >/dev/null 2>&1 || rc_a64=$?
    rm -f "$WORK_DIR/$base.x86" "$WORK_DIR/$base.a64"
    # Only x86-refused cases are in scope. A case x86 also accepts says nothing
    # about arm64 parity — it is a fact about the corpus, handled elsewhere.
    if [[ "$rc_x86" -ne 0 && "$rc_a64" -eq 0 ]]; then
        printf '%s\n' "$base"
    fi
}
export -f probe
export SOUC WORK_DIR TIMEOUT_SECS

find tests/compile-fail -name '*.sio' -print0 \
  | xargs -0 -P "$JOBS" -I{} bash -c 'probe "$@"' _ {} \
  | sort > "$WORK_DIR/diverged.txt"

# `|| true`: an EMPTY baseline is the goal state, and grep exits 1 when it
# matches nothing. Under `set -e` that killed the gate before it could
# report -- an all-comments baseline made the gate exit 1 with no verdict
# line, which reads as a failure and is in fact total success.
grep -vE '^\s*(#|$)' "$BASELINE" | sort > "$WORK_DIR/baseline.txt" || true

n_div="$(wc -l < "$WORK_DIR/diverged.txt" | tr -d ' ')"
n_base="$(wc -l < "$WORK_DIR/baseline.txt" | tr -d ' ')"

echo "diverged=$n_div"
echo "baseline=$n_base"

new_only="$(comm -23 "$WORK_DIR/diverged.txt" "$WORK_DIR/baseline.txt")"
fixed="$(comm -13 "$WORK_DIR/diverged.txt" "$WORK_DIR/baseline.txt")"

if [[ -n "$fixed" ]]; then
    echo "--- newly fixed (remove these from the baseline, in the commit that fixed them) ---"
    printf '%s\n' "$fixed"
fi

if [[ -n "$new_only" ]]; then
    echo "--- REGRESSION: refused by x86, accepted by aarch64-linux, not in the baseline ---"
    printf '%s\n' "$new_only"
    echo "A64_COMPILE_FAIL_PARITY_GATE=FAIL"
    exit 1
fi

if [[ "$n_div" -gt "$n_base" ]]; then
    echo "A64_COMPILE_FAIL_PARITY_GATE=FAIL (count grew: $n_div > $n_base)"
    exit 1
fi

echo "A64_COMPILE_FAIL_PARITY_GATE=PASS"
