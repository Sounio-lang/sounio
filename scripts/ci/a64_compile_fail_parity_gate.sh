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
# GATE_CLAIMS: every tests/compile-fail case the x86 target refuses with a given
#              diagnostic is also refused by aarch64-linux with that same
#              diagnostic, and every case whose outcome cannot be classified as
#              a refusal is reported rather than counted as parity
# GATE_ENGINE: lean_single (bin/souc-lean-single-x86_64)
# GATE_RESULT_ON_SKIP: fail
#
# SCOPE, stated precisely: this gate is UNIDIRECTIONAL. It asks whether arm64
# refuses what x86 refuses. The converse — arm64 refusing what x86 accepts —
# is a real divergence and is NOT watched here. "parity" in the gate id is
# therefore narrower than the word suggests.
#
# This is an ACCUSATION gate with a pinned allowance. The divergence baseline
# went 265 -> 55 -> 8 -> 0 over 2026-09-04..05 and is now empty: the gate
# asserts complete refusal parity over the whole corpus. That is a strong claim
# resting entirely on how an outcome is classified, which is why the exit-code
# predicate it used to rely on was replaced (see probe() below). A second
# baseline, compile_fail_rule_drift_baseline.txt, pins cases where both targets
# refuse for DIFFERENT reasons — an observation the old probe could not make.
#
# Do not add a case to either baseline to make the gate pass. A new divergence
# is a regression in the arm64 mirror, not a fact about the corpus.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
# shellcheck source=../lib/classify_compile.sh
source "$ROOT_DIR/scripts/lib/classify_compile.sh"

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

# Compile one case for both targets and CLASSIFY each outcome.
#
# The previous version of this probe decided divergence from the pair of exit
# codes alone: `rc_x86 != 0 && rc_a64 == 0`. That predicate cannot tell a
# refusal from a segfault, a timeout or an OOM kill, so crash/crash and
# refusal/timeout both read as parity. Worse, it read those exit codes from the
# RAW lean_single ELF, whose own wrapper (scripts/ci/souc-native-wrapper.sh)
# exists because "the raw ELF exits 0 even when it fails to produce an output
# ELF" and emits a ~35 kB stub instead -- so `rc_a64 == 0` did not mean
# "accepted" either. It then sent both compilers' output to /dev/null and
# rm -f'd the output paths without ever testing whether they were written,
# discarding the two signals that would have settled the question.
#
# Classification happens BEFORE the artifacts are deleted: artifact presence is
# evidence, not garbage. See scripts/lib/classify_compile.sh.
#
# The compiler resolves `import` relative to the CURRENT DIRECTORY, so this
# must run from the repository root — running it elsewhere makes roughly 40 %
# of the corpus fail with E224 on both targets and silently reports parity.
# That was a known hazard with no detector; the census below is the detector.
probe() {
    local file="$1"
    local base rc_x86 rc_a64 cls_x86 cls_a64 d_x86 d_a64 verdict
    base="$(basename "$file" .sio)"
    local out_x="$WORK_DIR/$base.x86" out_a="$WORK_DIR/$base.a64"
    local log_x="$WORK_DIR/$base.x86.log" log_a="$WORK_DIR/$base.a64.log"

    rc_x86=0
    timeout "$TIMEOUT_SECS" "$SOUC" "$file" "$out_x" >"$log_x" 2>&1 || rc_x86=$?
    rc_a64=0
    timeout "$TIMEOUT_SECS" "$SOUC" "$file" "$out_a" --target aarch64-linux >"$log_a" 2>&1 || rc_a64=$?

    sounio_classify_compile "$rc_x86" "$log_x" "$out_x"; cls_x86="$SOUNIO_CC_CLASS"
    sounio_classify_compile "$rc_a64" "$log_a" "$out_a"; cls_a64="$SOUNIO_CC_CLASS"
    d_x86="$(sounio_primary_diag "$log_x" | tr -d '\n')"
    d_a64="$(sounio_primary_diag "$log_a" | tr -d '\n')"
    rm -f "$out_x" "$out_a" "$log_x" "$log_a"

    # PARITY        both refused, same primary diagnostic
    # DIVERGENCE    x86 refused, a64 accepted — the claim this gate defends
    # RULE_DRIFT    both refused, but for different reasons. Not parity: it is
    #               how the wrong-cwd mode looks (everything refuses with E224)
    #               and how a mirror that refuses by accident looks.
    # OUT_OF_SCOPE  x86 did not refuse. Declared out of scope by this gate.
    # INSTRUMENT    crash / timeout / silent failure / infra on either side.
    #               Not evidence of anything about parity.
    case "$cls_x86:$cls_a64" in
      REFUSED:ACCEPTED) verdict="DIVERGENCE" ;;
      REFUSED:REFUSED)
          if [[ "$d_x86" == "$d_a64" ]]; then verdict="PARITY"; else verdict="RULE_DRIFT"; fi ;;
      ACCEPTED:*)       verdict="OUT_OF_SCOPE" ;;
      *)                verdict="INSTRUMENT" ;;
    esac
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$base" "$cls_x86" "$cls_a64" "${d_x86:-none}" "${d_a64:-none}" "$verdict"
}
export -f probe sounio_classify_compile sounio_primary_diag sounio_diag_codes \
          sounio_is_fatal_log _sounio_has_diagnostic
export SOUC WORK_DIR TIMEOUT_SECS
export _SOUNIO_CC_FATAL_RE _SOUNIO_CC_SUCCESS_RE _SOUNIO_CC_VERDICT_FAIL_RE

# Corpus scope: this gate runs lean_single, so a fixture whose `//@ requires:`
# names a different engine is not evidence about it. 288 of the 680 files in
# tests/compile-fail carry `//@ requires: madaros`; running them here produced
# 257 PARITY verdicts in which two targets of the WRONG engine agreed about a
# check the fixture never asked lean_single to have. That is coverage on paper.
#
# The semantics mirror scripts/dev/run_sio_test_suite_v2.sh:417-459, including
# its rule that an unrecognised value must not fall through silently -- a typo
# like `requires: madros` would otherwise re-admit the whole class.
#
# Exclusions are printed, never silent: a gate that quietly narrows its own
# corpus is the failure mode this gate exists to detect.
ENGINE="${PARITY_ENGINE:-lean_single}"
: > "$WORK_DIR/in_scope.txt"
: > "$WORK_DIR/excluded.txt"
unknown_requires=""
while IFS= read -r f; do
    req="$(sed -n 's|^//@[[:space:]]*requires:[[:space:]]*\([A-Za-z_][A-Za-z_0-9]*\).*|\1|p' "$f" | head -1)"
    case "$req" in
        "")               printf '%s\n' "$f" >> "$WORK_DIR/in_scope.txt" ;;
        gpu|llvm)         printf '%s\n' "$f" >> "$WORK_DIR/in_scope.txt" ;;
        "$ENGINE")        printf '%s\n' "$f" >> "$WORK_DIR/in_scope.txt" ;;
        madaros|lean_single)
                          printf '%s\t%s\n' "$(basename "$f" .sio)" "$req" >> "$WORK_DIR/excluded.txt" ;;
        *)                unknown_requires="$unknown_requires $f:$req" ;;
    esac
done < <(find tests/compile-fail -name '*.sio' | LC_ALL=C sort)

if [[ -n "$unknown_requires" ]]; then
    echo "--- unrecognised '//@ requires:' values (expected: gpu|llvm|madaros|lean_single) ---"
    printf '%s\n' $unknown_requires
    echo "A64_COMPILE_FAIL_PARITY_GATE=FAIL (unknown requires would silently re-admit the wrong engine)"
    exit 1
fi

n_corpus="$(wc -l < "$WORK_DIR/in_scope.txt" | tr -d ' ')"
n_excluded="$(wc -l < "$WORK_DIR/excluded.txt" | tr -d ' ')"
echo "engine=$ENGINE corpus_files=$n_corpus excluded_other_engine=$n_excluded"

tr '\n' '\0' < "$WORK_DIR/in_scope.txt" \
  | xargs -0 -P "$JOBS" -I{} bash -c 'probe "$@"' _ {} \
  | LC_ALL=C sort > "$WORK_DIR/census.tsv"

# --- census -----------------------------------------------------------------
# The census is the detector the exit-code probe lacked. A gate that reports
# only "0 divergences" cannot distinguish a clean corpus from a corpus that
# never really ran.
# The full per-case census is the gate's primary evidence; keep it on request
# so a failure can be triaged without re-running the corpus.
if [[ -n "${CENSUS_OUT:-}" ]]; then
    cp "$WORK_DIR/census.tsv" "$CENSUS_OUT"
    echo "census=$CENSUS_OUT"
fi

count_of() { awk -F'\t' -v v="$1" '$6==v' "$WORK_DIR/census.tsv" | wc -l | tr -d ' '; }
n_total="$(wc -l < "$WORK_DIR/census.tsv" | tr -d ' ')"
n_parity="$(count_of PARITY)"
n_drift="$(count_of RULE_DRIFT)"
n_scope="$(count_of OUT_OF_SCOPE)"
n_instr="$(count_of INSTRUMENT)"
n_in_scope=$(( n_parity + $(count_of DIVERGENCE) + n_drift ))

echo "corpus_total=$n_total"
echo "parity=$n_parity divergence=$(count_of DIVERGENCE) rule_drift=$n_drift out_of_scope=$n_scope instrument=$n_instr"

# Corpus floor, over the engine-scoped population above (392 of 680 files on
# 2026-09-06). The corpus grows, so these are floors rather than equalities and
# are overridable for a filtered run. A collapse in the in-scope population is
# the signature of the wrong-cwd or wrong-stdlib false green described above.
CORPUS_MIN="${CORPUS_MIN:-392}"
IN_SCOPE_MIN="${IN_SCOPE_MIN:-300}"
if [[ "$n_total" -lt "$CORPUS_MIN" ]]; then
    echo "A64_COMPILE_FAIL_PARITY_GATE=FAIL (corpus shrank: $n_total < $CORPUS_MIN)"
    exit 1
fi
if [[ "$n_in_scope" -lt "$IN_SCOPE_MIN" ]]; then
    echo "--- corpus health: too few cases actually reached a refusal decision ---"
    awk -F'\t' '$6!="PARITY"' "$WORK_DIR/census.tsv" | head -20
    echo "A64_COMPILE_FAIL_PARITY_GATE=FAIL (in-scope collapsed: $n_in_scope < $IN_SCOPE_MIN — check cwd and stdlib path)"
    exit 1
fi

# Import-resolution ceiling. This is the detector for the failure mode named in
# probe()'s comment: run from the wrong directory, ~40 % of the corpus fails
# with E224 on BOTH targets, and the gate reports parity. Per-case
# classification does not catch it — two identical E224 refusals genuinely are
# "the same diagnostic on both targets". What gives it away is the
# DISTRIBUTION: a corpus of deliberately varied compile-fail cases should not
# collapse onto one import error. Verified against a synthetic compiler that
# emits E224 unconditionally: without this check the run reports 100 % parity.
IMPORT_ERR_CODE="${IMPORT_ERR_CODE:-E224}"
n_import="$(awk -F'\t' -v c="$IMPORT_ERR_CODE" '$4==c && $5==c' "$WORK_DIR/census.tsv" | wc -l | tr -d ' ')"
echo "import_errors_both_targets=$n_import ($IMPORT_ERR_CODE)"
IMPORT_CEILING="${IMPORT_CEILING:-$(( n_total / 10 ))}"
if [[ "$n_import" -gt "$IMPORT_CEILING" ]]; then
    echo "--- $IMPORT_ERR_CODE on both targets in $n_import of $n_total cases ---"
    echo "This is what a wrong working directory or stdlib path looks like: the"
    echo "corpus never reaches the checks it exists to exercise, and every case"
    echo "agrees, so the run reports parity. Confirm cwd is the repository root."
    awk -F'\t' -v c="$IMPORT_ERR_CODE" '$4==c && $5==c {print $1}' "$WORK_DIR/census.tsv" | head -10
    echo "A64_COMPILE_FAIL_PARITY_GATE=FAIL (import-resolution ceiling: $n_import > $IMPORT_CEILING)"
    exit 1
fi

# An outcome that cannot be classified is never averaged into a verdict.
# Mirrors SOUNIO_WITNESS_UNCLEAN_CEILING in witness_declares_its_sabotage_gate.sh.
# Pinned like the divergence list, for the same reason: these are pre-existing
# conditions the old exit-code probe could not see, so failing the whole gate on
# day one would only delete the measurement. Each pinned case is a defect --
# a compiler that segfaults, hangs, or rejects without saying why -- and belongs
# to an issue, not to this file forever.
INSTRUMENT_BASELINE="${INSTRUMENT_BASELINE:-$ROOT_DIR/tests/selfhost/aarch64_compile/compile_fail_instrument_baseline.txt}"
awk -F'\t' '$6=="INSTRUMENT" {printf "%s\t%s\t%s\n", $1, $2, $3}' "$WORK_DIR/census.tsv" \
  | LC_ALL=C sort > "$WORK_DIR/instrument.txt"
if [[ -f "$INSTRUMENT_BASELINE" ]]; then
    grep -vE '^[[:space:]]*(#|$)' "$INSTRUMENT_BASELINE" | LC_ALL=C sort > "$WORK_DIR/instrument_base.txt" || true
    instr_new="$(comm -23 "$WORK_DIR/instrument.txt" "$WORK_DIR/instrument_base.txt")"
    instr_fixed="$(comm -13 "$WORK_DIR/instrument.txt" "$WORK_DIR/instrument_base.txt")"
    if [[ -n "$instr_fixed" ]]; then
        echo "--- newly clean (remove from the instrument baseline, in the commit that fixed them) ---"
        printf '%s\n' "$instr_fixed"
    fi
    if [[ -n "$instr_new" ]]; then
        echo "--- REGRESSION: outcome not classifiable as a refusal, not in the baseline ---"
        printf '%s\n' "$instr_new"
        echo "A64_COMPILE_FAIL_PARITY_GATE=FAIL (new instrument faults)"
        exit 1
    fi
elif [[ "$n_instr" -gt "${INSTRUMENT_BUDGET:-0}" ]]; then
    echo "--- INSTRUMENT FAULTS (crash / timeout / silent failure / infra — not parity evidence) ---"
    cat "$WORK_DIR/instrument.txt"
    echo "A64_COMPILE_FAIL_PARITY_GATE=FAIL (instrument faults: $n_instr, budget ${INSTRUMENT_BUDGET:-0})"
    exit 1
fi

# RULE_DRIFT is a new observation this gate could not previously make, so it
# carries its own pinned baseline rather than failing the gate on day one.
DRIFT_BASELINE="${DRIFT_BASELINE:-$ROOT_DIR/tests/selfhost/aarch64_compile/compile_fail_rule_drift_baseline.txt}"
drift_now="$(awk -F'\t' '$6=="RULE_DRIFT" {printf "%s\t%s\t%s\n", $1, $4, $5}' "$WORK_DIR/census.tsv")"
printf '%s' "$drift_now" > "$WORK_DIR/drift.txt"
if [[ -f "$DRIFT_BASELINE" ]]; then
    grep -vE '^\s*(#|$)' "$DRIFT_BASELINE" | LC_ALL=C sort > "$WORK_DIR/drift_base.txt" || true
    LC_ALL=C sort "$WORK_DIR/drift.txt" > "$WORK_DIR/drift_sorted.txt"
    drift_new="$(comm -23 "$WORK_DIR/drift_sorted.txt" "$WORK_DIR/drift_base.txt")"
    if [[ -n "$drift_new" ]]; then
        echo "--- REGRESSION: refused by both targets for DIFFERENT reasons, not in the baseline ---"
        printf '%s\n' "$drift_new"
        echo "A64_COMPILE_FAIL_PARITY_GATE=FAIL (new rule drift)"
        exit 1
    fi
elif [[ "$n_drift" -gt 0 ]]; then
    echo "--- rule drift observed, no baseline pinned yet (both refuse, different diagnostic) ---"
    printf '%s\n' "$drift_now" | head -20
fi

# The historical divergence list keeps its original file format: one basename
# per line, so the existing baseline and its history stay valid.
awk -F'\t' '$6=="DIVERGENCE" {print $1}' "$WORK_DIR/census.tsv" \
  | LC_ALL=C sort > "$WORK_DIR/diverged.txt"

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
