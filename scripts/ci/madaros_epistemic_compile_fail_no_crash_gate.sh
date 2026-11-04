#!/usr/bin/env bash
# A compile-fail fixture for an epistemic builtin must be REFUSED, not CRASHED.
#
# Why this gate exists
# --------------------
# scripts/lib/classify_compile.sh already states the systemic problem and
# measures it: "over the 18 gates that assert a compile must fail: 2 check for
# a signal, 0 check for a timeout". The epistemic builtins sat in that blind
# spot. Measured 2026-11-04 on a Madaros built from self-hosted/: 39 of the 42
# compile-fail fixtures that call one of these builtins exit 139 -- they print
# the correct diagnostic and THEN segfault, so every gate that reads `rc != 0`
# as "refused", or that greps the log for the expected text, scored them as
# passing refusals.
#
# a64_compile_fail_parity_gate.sh does use the classifier, but it runs
# lean_single (PARITY_ENGINE), which refuses these fixtures with E200 because
# it does not implement the builtins at all. That is coverage on paper for this
# class: the engine under test never ran the check the fixture asks about.
#
# So this gate runs the same corpus through Madaros and classifies the outcome.
#
# The ratchet
# -----------
# The crashes are a known, documented defect -- the layout-sensitive
# stale-stack / SRET smash in docs/audit/MODULAR_COMPILER_STACK_CLASH_2026-05-29.md,
# whose sanctioned remedy is porting each handler to *mut Checker (see
# checker_check_measure_expr_inplace and checker_check_defer_action_expr_inplace).
# Repairing all of them is not one change, so the current crash set is pinned
# in a baseline and this gate fails on movement in EITHER direction:
#
#   not pinned and not REFUSED -> the fixture no longer earns a clean refusal.
#                              CRASHED is the regression this gate was built
#                              for, but the classifier distinguishes eight
#                              outcomes and only one of them is a refusal, so
#                              the predicate is closed over the whole set:
#                              TIMEOUT, INFRA, ACCEPTED, SILENT_FAILURE,
#                              REFUSED_UNDIAGNOSED and REFUSED_WITH_ARTIFACT
#                              all fail too. Checking CRASHED membership alone
#                              was fail-open -- a fixture that started timing
#                              out or exiting 0 left the crashed set unchanged
#                              and the gate passed. Found in review of the
#                              first version of this gate.
#   pinned and not CRASHED  -> repaired; drop it from the baseline. The pinned
#                              set may only shrink, matching the rule in
#                              scripts/ci/known_failure_pin_gate.sh.
#
# Usage:
#   bash scripts/ci/madaros_epistemic_compile_fail_no_crash_gate.sh
#   MADAROS_RAW_BIN=/path/to/madaros bash scripts/ci/...   # pin a binary
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
# shellcheck source=../lib/classify_compile.sh
source "$ROOT_DIR/scripts/lib/classify_compile.sh"

SOUC="${SOUC_BIN:-$ROOT_DIR/bin/souc}"
BASELINE="${BASELINE:-$ROOT_DIR/tests/selfhost/madaros_epistemic_compile_fail_crashers.txt}"
TIMEOUT_SECS="${TIMEOUT_SECS:-120}"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sounio-epi-nocrash.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

# The builtins whose handlers thread a by-value Checker. Printed, never silent:
# a gate that quietly narrows its own corpus is the failure mode it exists to
# detect.
BUILTINS='defer_action|admit_action|deferral_reason|plan_acquisition|acquisition_reason|plan_recourse|recourse_reason|propose_alternatives|alternative_option|alternative_count|commit_alternative|transition_reason|observe_transition|rollback_transition|prove_robust'

echo "MADAROS_EPI_COMPILE_FAIL_GATE_START"
echo "souc=$SOUC"
echo "baseline=$BASELINE"
echo "builtins=$BUILTINS"

[[ -x "$SOUC" ]] || { echo "error: missing souc at $SOUC" >&2; exit 1; }
[[ -f "$BASELINE" ]] || { echo "error: missing baseline at $BASELINE" >&2; exit 1; }

grep -lE "\b($BUILTINS)\(" tests/compile-fail/*.sio 2>/dev/null \
  | LC_ALL=C sort > "$WORK_DIR/corpus.txt" || true
n_corpus="$(wc -l < "$WORK_DIR/corpus.txt" | tr -d ' ')"
[[ "$n_corpus" -gt 0 ]] || { echo "error: corpus selector matched no fixture" >&2; exit 1; }
echo "corpus_files=$n_corpus"

grep -vE '^[[:space:]]*(#|$)' "$BASELINE" | LC_ALL=C sort -u > "$WORK_DIR/pinned.txt"
echo "pinned=$(wc -l < "$WORK_DIR/pinned.txt" | tr -d ' ')"

: > "$WORK_DIR/census.tsv"
while IFS= read -r f; do
    base="$(basename "$f" .sio)"
    log="$WORK_DIR/$base.log"
    set +e
    timeout "$TIMEOUT_SECS" "$SOUC" check "$f" >"$log" 2>&1
    rc=$?
    set -e
    sounio_classify_compile "$rc" "$log" ""
    # A fixture that declares its expected diagnostic must still show it: a
    # clean refusal for an unrelated reason is not a witness for the check the
    # fixture is about.
    want="$(sed -n 's|^//@[[:space:]]*error-pattern:[[:space:]]*\(.*\)|\1|p' "$f" | head -1)"
    pat_ok="n/a"
    if [[ -n "$want" ]]; then
        if grep -Fq -- "$want" "$log"; then pat_ok="ok"; else pat_ok="MISSING:$want"; fi
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$base" "$rc" "$SOUNIO_CC_CLASS" "${SOUNIO_CC_DETAIL:-none}" "$pat_ok" \
      >> "$WORK_DIR/census.tsv"
done < "$WORK_DIR/corpus.txt"

echo "--- census (fixture / rc / class / detail / error-pattern) ---"
cat "$WORK_DIR/census.tsv"

awk -F'\t' '$3=="CRASHED"{print $1}' "$WORK_DIR/census.tsv" | LC_ALL=C sort -u > "$WORK_DIR/crashed.txt"
# Closed over the classifier's whole result set: exactly REFUSED is acceptable
# for an unpinned fixture. Everything else -- CRASHED, TIMEOUT, INFRA,
# ACCEPTED, SILENT_FAILURE, REFUSED_UNDIAGNOSED, REFUSED_WITH_ARTIFACT -- is a
# failure, so no new outcome can slip through as a pass.
awk -F'\t' '$3!="REFUSED"{print $1}' "$WORK_DIR/census.tsv" | LC_ALL=C sort -u > "$WORK_DIR/not_refused.txt"
bad_unpinned="$(comm -23 "$WORK_DIR/not_refused.txt" "$WORK_DIR/pinned.txt")"
repaired="$(comm -13 "$WORK_DIR/crashed.txt" "$WORK_DIR/pinned.txt")"
# Scoped to unpinned rows: a pinned crasher cannot be held to its declared
# diagnostic, and some declare a lean_single code (observe_transition_requires_plan
# declares E200) that Madaros has no reason to emit. The pin is the statement
# about those; the pattern is a statement about fixtures that do refuse.
awk -F'\t' '$5 ~ /^MISSING:/{print $1}' "$WORK_DIR/census.tsv" | LC_ALL=C sort -u > "$WORK_DIR/pat_missing.txt"
bad_pattern="$(comm -23 "$WORK_DIR/pat_missing.txt" "$WORK_DIR/pinned.txt")"

n_crashed="$(wc -l < "$WORK_DIR/crashed.txt" | tr -d ' ')"
n_new="$(printf '%s' "$bad_unpinned" | grep -c . || true)"
n_rep="$(printf '%s' "$repaired" | grep -c . || true)"
n_pat="$(printf '%s' "$bad_pattern" | grep -c . || true)"
echo "MADAROS_EPI_COMPILE_FAIL_DERIVED corpus=$n_corpus crashed=$n_crashed unpinned_not_refused=$n_new repaired_still_pinned=$n_rep missing_pattern=$n_pat"

fail=0
if [[ "$n_new" -gt 0 ]]; then
    echo "--- unpinned and not REFUSED: the fixture lost its clean refusal ---" >&2
    while IFS= read -r b; do
        [[ -z "$b" ]] && continue
        echo "$b -> $(awk -F'\t' -v k="$b" '$1==k{print $3" ("$4")"}' "$WORK_DIR/census.tsv")" >&2
        [[ -f "$WORK_DIR/$b.log" ]] && { echo "=== $b (tail) ===" >&2; tail -n 12 "$WORK_DIR/$b.log" >&2; }
    done <<< "$bad_unpinned"
    fail=1
fi
if [[ "$n_pat" -gt 0 ]]; then
    echo "--- refused, but not with the diagnostic the fixture declares ---" >&2
    while IFS= read -r b; do
        [[ -n "$b" ]] && echo "$b -> $(awk -F'\t' -v k="$b" '$1==k{print $5}' "$WORK_DIR/census.tsv")" >&2
    done <<< "$bad_pattern"
    fail=1
fi
if [[ "$n_rep" -gt 0 ]]; then
    echo "--- pinned but no longer crashing: drop these from the baseline ---" >&2
    printf '%s\n' "$repaired" >&2
    fail=1
fi

if [[ "$fail" -ne 0 ]]; then
    echo "MADAROS_EPI_COMPILE_FAIL_GATE=FAIL unpinned_not_refused=$n_new repaired_still_pinned=$n_rep missing_pattern=$n_pat" >&2
    exit 1
fi
echo "MADAROS_EPI_COMPILE_FAIL_GATE=PASS corpus=$n_corpus crashed=$n_crashed (all pinned)"
