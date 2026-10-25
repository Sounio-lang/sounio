#!/usr/bin/env bash
# Self-hosted Sounio test suite runner - Version 2 with parallel execution and JUnit output
#
# New features:
#   - Parallel test execution (background jobs with job limit)
#   - JUnit/XML output format for CI integration
#   - New annotations: @known-failure, @skip-if, @requires, @flaky
#
# Annotations:
#   //@ run-pass              — expect exit 0
#   //@ compile-fail          — expect a compiler diagnostic, not a timeout or signal
#   //@ ignore                — skip this test
#   //@ check-only            — compile only, do not execute
#   //@ expect-stdout: X      — stdout must contain X (run-pass only)
#   //@ expect-stdout-contains: X — stdout must contain X (run-pass only)
#   //@ error-pattern: X      — stderr/stdout must contain X (compile-fail only)
#   //@ known-failure: REASON — documented accepted failure
#   //@ xfail-expect: TEXT    — the known failure's output must contain TEXT
#                               (repeatable; enforced in every mode)
#   //@ xfail-exit: N         — the known failure's souc exit code must be N
#                               (124 = timeout); mismatch is a fresh FAIL
#   //@ skip-if: CONDITION    — conditional skip (e.g., skip-if: no-gpu)
#   //@ requires: FEATURE     — feature dependency (gpu|llvm|madaros|lean_single|slow|vendor-material)
#   //@ flaky                 — known flaky test
#   //@ timeout: SECONDS      — override default timeout
#
# Unknown `expect-*` / `expected-*` header keys fail the test. They used to
# be skipped silently, so `expect-stdout-contains` asserted nothing.
#
# The header is the leading run of `//@ ` lines, `//` comment lines and blank
# lines; the first other line ends it and nothing after it is read. A bare `//`
# (no trailing space) is a comment line too. It used to end the header, so a
# `//@ requires:` or `//@ expect-stdout:` placed after a `//` paragraph break
# was silently ignored (madaros_uninit_struct_local.sio, global_array_len1_index.sio,
# madaros_wide_struct_variance_field.sio, measured 2026-10-06).
# scripts/ci/test_annotation_coverage_gate.sh refuses a `//@ ` line past the
# header, so an annotation cannot go unread again.
#
# Usage:
#   bash scripts/dev/run_sio_test_suite_v2.sh [--filter PATTERN] [--verbose] [--format junit] [--jobs N]
#   bash scripts/dev/run_sio_test_suite_v2.sh --filter-prefix PREFIX [--verbose] [--format junit] [--jobs N]
#   bash scripts/dev/run_sio_test_suite_v2.sh --filter-exact BASENAME [--verbose] [--format junit] [--jobs N]
#   bash scripts/dev/run_sio_test_suite_v2.sh [--filter PATTERN] --list-tests
#   bash scripts/dev/run_sio_test_suite_v2.sh --test-list FILE [--verbose] [--format junit] [--jobs N]

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/../lib/resolve_souc.sh" && -d "$SCRIPT_DIR/../../tests" ]]; then
    ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$ROOT_DIR"

# Harness-local override. SOUNIO_TEST_SOUC_BIN may point at one of two
# distinct interfaces:
#
#   1. A wrapper-style executable that accepts the harness subcommand
#      interface (check/run/compile/info), e.g. the rebuilt ontology
#      validation wrapper produced by scripts/ci/build_ontology_validation_souc.sh.
#      These are bash scripts (#!/usr/bin/env bash). Use them directly as
#      $SOUC_BIN so the harness calls them as `$SOUC_BIN check file.sio`.
#
#   2. A raw self-hosted ELF binary, e.g. the CI `souc-stage2` artifact,
#      which only accepts the raw `<src> <out>` interface and would treat
#      `check` as a source path. For these, route through the repo wrapper
#      at scripts/ci/souc-native-wrapper.sh, which translates subcommands to
#      the raw interface
#      via SOUNIO_SOUC_BIN.
#
# We discriminate by sniffing the first two bytes for a shebang.
if [[ -n "${SOUNIO_TEST_SOUC_BIN:-}" ]]; then
    if [[ -r "$SOUNIO_TEST_SOUC_BIN" ]] \
       && [[ "$(head -c 2 "$SOUNIO_TEST_SOUC_BIN" 2>/dev/null)" == "#!" ]]; then
        export SOUNIO_SOUC_BIN="$SOUNIO_TEST_SOUC_BIN"
        SOUC_BIN="$SOUNIO_TEST_SOUC_BIN"
    else
        export SOUNIO_SOUC_BIN="$SOUNIO_TEST_SOUC_BIN"
        SOUC_BIN="$ROOT_DIR/scripts/ci/souc-native-wrapper.sh"
    fi
fi
if [[ -n "${SOUNIO_TEST_NATIVE_BIN:-}" ]]; then
    export SOUC_NATIVE_BIN="$SOUNIO_TEST_NATIVE_BIN"
fi

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

# ── Which compiler is under test (G8) ───────────────────────────────────────
# A local `make build` leaves gen3.elf in the tree. It does NOT replace
# bin/souc-lean-single-x86_64, which is what bin/souc execs for the lean_single
# engine, so a lean_single run measures the compiler as it was BEFORE the change
# and reports nothing about it -- silently, and with a green suite.
#
# This is not hypothetical. A reaction-literal feature was measured this way,
# pronounced inert because three compile-fail fixtures compiled cleanly, and was
# correct the whole time; run against gen3.elf directly the same fixtures gave
# E188/E189/E190.
#
# Madaros does not have this hole: bin/souc already resolves a local
# artifacts/self-hosted/madaros ahead of the committed ELF and says so. The
# lean_single path prints no provenance at all, which is why this lives here.
_sounio_engine_md5() {
    if [[ -r "$1" ]]; then md5sum "$1" 2>/dev/null | cut -c1-8; else echo "absent"; fi
}
SOUNIO_ENGINE_KIND="madaros"
SOUNIO_ENGINE_ELF="$ROOT_DIR/artifacts/self-hosted/madaros"
if [[ "${SOUNIO_SOUC_ENGINE:-}" == "lean_single" ]] || [[ ! -x "$ROOT_DIR/artifacts/self-hosted/madaros" ]]; then
    SOUNIO_ENGINE_KIND="lean_single"
    SOUNIO_ENGINE_ELF="$ROOT_DIR/bin/souc-lean-single-x86_64"
    [[ -x "$SOUNIO_ENGINE_ELF" ]] || SOUNIO_ENGINE_ELF="$ROOT_DIR/bin/souc-linux-x86_64"
fi
if [[ -n "${SOUNIO_TEST_SOUC_BIN:-}" ]]; then
    SOUNIO_ENGINE_KIND="explicit"
    SOUNIO_ENGINE_ELF="$SOUNIO_TEST_SOUC_BIN"
fi

# Refuse to guess: a newer local build of the engine that is NOT the one about
# to run means the result would describe the wrong compiler.
if [[ "$SOUNIO_ENGINE_KIND" == "lean_single" \
      && -f "$ROOT_DIR/gen3.elf" \
      && "$ROOT_DIR/gen3.elf" -nt "$SOUNIO_ENGINE_ELF" \
      && "${SOUNIO_TEST_ALLOW_STALE_ENGINE:-0}" != "1" ]]; then
    echo "harness: refusing to run -- a locally built gen3.elf is newer than the" >&2
    echo "         lean_single binary this run would use, so the result would" >&2
    echo "         describe the committed seed, not the tree." >&2
    echo "           would run:  $SOUNIO_ENGINE_ELF  (md5 $(_sounio_engine_md5 "$SOUNIO_ENGINE_ELF"))" >&2
    echo "           local build: $ROOT_DIR/gen3.elf  (md5 $(_sounio_engine_md5 "$ROOT_DIR/gen3.elf"))" >&2
    echo "         pick one:" >&2
    echo "           SOUNIO_TEST_SOUC_BIN=$ROOT_DIR/gen3.elf  $0 ...   # test the build" >&2
    echo "           SOUNIO_TEST_ALLOW_STALE_ENGINE=1         $0 ...   # test the seed, on purpose" >&2
    exit 2
fi

# Every run says what produced it, so no result can be quoted without it.
# stderr, not stdout: run_ontology_validation.sh captures this script's
# stdout under --list-tests as a literal test-path list (build_validated_harness_list
# reads every line as a fixture). A confirmation line on stdout was treated as
# a selected test and rejected as "non-kernel fixture" -- measured on CI,
# 2026-09-03, PR #2394. stdout here must be ONLY the test list or the JSON a
# caller asked for; every human-facing status line belongs on stderr, as the
# refusal branch above it already does.
echo "harness: engine=$SOUNIO_ENGINE_KIND elf=$SOUNIO_ENGINE_ELF md5=$(_sounio_engine_md5 "$SOUNIO_ENGINE_ELF")" >&2

# If SOUC_BIN resolved to a raw ELF (not a shell script), route through the
# subcommand wrapper so the harness can call `check`/`run`/`compile` correctly.
if [[ "$(head -c 2 "$SOUC_BIN" 2>/dev/null)" != "#!" ]]; then
    _NATIVE_WRAPPER="$ROOT_DIR/scripts/ci/souc-native-wrapper.sh"
    if [[ -f "$_NATIVE_WRAPPER" ]]; then
        export SOUNIO_SOUC_BIN="$SOUC_BIN"
        SOUC_BIN="$_NATIVE_WRAPPER"
        chmod +x "$_NATIVE_WRAPPER"
    fi
    unset _NATIVE_WRAPPER
fi

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

# Parse arguments
FILTER=""
FILTER_MODE="contains"
VERBOSE=""
FORMAT="text"
JOBS="${SOUNIO_TEST_JOBS:-$(nproc 2>/dev/null || echo 4)}"
JOBS_EXPLICIT=0
if [[ -n "${SOUNIO_TEST_JOBS:-}" ]]; then JOBS_EXPLICIT=1; fi
LIST_TESTS=0
TEST_LIST_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --filter)
            FILTER="$2"
            FILTER_MODE="contains"
            shift 2
            ;;
        --filter-prefix)
            FILTER="$2"
            FILTER_MODE="prefix"
            shift 2
            ;;
        --filter-exact)
            FILTER="$2"
            FILTER_MODE="exact"
            shift 2
            ;;
        --verbose)
            VERBOSE="1"
            shift
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --jobs)
            JOBS="$2"
            JOBS_EXPLICIT=1
            shift 2
            ;;
        --list-tests)
            LIST_TESTS=1
            shift
            ;;
        --test-list)
            TEST_LIST_FILE="$2"
            shift 2
            ;;
        *)
            FILTER="$1"
            FILTER_MODE="contains"
            shift
            ;;
    esac
done

# CI parallelism: scale with available CPU and memory instead of a fixed four.
# The cap prevents oversubscription, while SOUNIO_TEST_JOBS_DYNAMIC=0 restores
# the exact --jobs value for debugging or constrained runners.
if [[ "${CI:-}" == "true" && "${SOUNIO_TEST_JOBS_DYNAMIC:-1}" == "1" ]]; then
    ci_cpus="${SOUNIO_TEST_CI_CPUS:-$(nproc 2>/dev/null || echo 1)}"
    ci_mem_kb="${SOUNIO_TEST_CI_MEM_KB:-$(awk '/MemAvailable/ {print $2}' /proc/meminfo 2>/dev/null || echo 1048576)}"
    ci_cpu_jobs=$((ci_cpus * 2))
    ci_mem_jobs=$((ci_mem_kb / 524288))
    ci_cap="${SOUNIO_TEST_JOBS_CAP:-8}"
    [[ "$ci_cpu_jobs" =~ ^[0-9]+$ ]] || ci_cpu_jobs=1
    [[ "$ci_mem_jobs" =~ ^[0-9]+$ ]] || ci_mem_jobs=1
    [[ "$ci_cap" =~ ^[0-9]+$ ]] || ci_cap=8
    if [[ "$ci_cpu_jobs" -lt 1 ]]; then ci_cpu_jobs=1; fi
    if [[ "$ci_mem_jobs" -lt 1 ]]; then ci_mem_jobs=1; fi
    if [[ "$ci_cap" -lt 1 ]]; then ci_cap=1; fi
    if [[ "$ci_cpu_jobs" -lt "$ci_mem_jobs" ]]; then
        ci_dynamic_jobs="$ci_cpu_jobs"
    else
        ci_dynamic_jobs="$ci_mem_jobs"
    fi
    if [[ "$ci_dynamic_jobs" -gt "$ci_cap" ]]; then
        ci_dynamic_jobs="$ci_cap"
    fi
    # Implicit nproc defaults must not bypass the memory guard. Preserve a
    # higher count only when the caller explicitly requested it. CI's current
    # --jobs 4 becomes --jobs 8 on a four-vCPU runner when memory permits.
    if [[ "$JOBS_EXPLICIT" -eq 0 ]] || [[ "$ci_dynamic_jobs" -gt "$JOBS" ]]; then
        JOBS="$ci_dynamic_jobs"
    fi
fi

# Test counters
PASS=0
FAIL=0
SKIP=0
KNOWN_FAILURE=0
XPAS=0
XPAS_LIST=""
FLAKY=0
VACUOUS_KNOWN=0
VACUOUS_STALE=""
ERRORS=""

# Repo-level blocker manifest. This lets CI stay strict about new failures while
# keeping old, audited hardening backlog items visible as xfails instead of noise.
# An entry may pin the failure mode it was audited for with `path|substring`: the
# manifest only launders a failure whose test_output contains that substring, so a
# regression that fails a DIFFERENT way (a new SIGSEGV, a wrong answer) still reports
# as a fresh fail instead of being silently absorbed by an old entry that was about
# something else. Plain `path` entries (most of the file) are unchecked, as before.
KNOWN_FAILURES_FILE="${SOUNIO_TEST_KNOWN_FAILURES_FILE:-}"
declare -A KNOWN_FAILURE_MAP=()
declare -A KNOWN_FAILURE_REASON_MAP=()
if [[ -z "$KNOWN_FAILURES_FILE" && -z "$FILTER" && "$FORMAT" == "junit" ]]; then
    KNOWN_FAILURES_FILE="$ROOT_DIR/tests/known_failures/hardened_diagnostics_full_suite.txt"
fi
if [[ -n "$KNOWN_FAILURES_FILE" && -f "$KNOWN_FAILURES_FILE" ]]; then
    while IFS= read -r line; do
        line="${line%%#*}"
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [[ -z "$line" ]] && continue
        kf_path="$line"; kf_reason=""
        if [[ "$line" == *"|"* ]]; then
            kf_path="${line%%|*}"
            kf_reason="${line#*|}"
            kf_path="${kf_path%"${kf_path##*[![:space:]]}"}"
            kf_reason="${kf_reason#"${kf_reason%%[![:space:]]*}"}"
        fi
        KNOWN_FAILURE_MAP["$kf_path"]=1
        [[ -n "$kf_reason" ]] && KNOWN_FAILURE_REASON_MAP["$kf_path"]="$kf_reason"
    done < "$KNOWN_FAILURES_FILE"
fi

# Vacuous-expect-stdout/error-pattern baseline. The //@ expect-stdout: / //@
# error-pattern: extraction below used to quote the whole `=~` pattern, which
# makes bash treat it as a literal string instead of a regex, so the capture
# group never captured and every such assertion matched vacuously (an empty
# expected string always matches). Fixing the extraction (see the comment at
# the expect_stdout/error_patterns parse loop) makes every test whose
# annotation was actually wrong fail for real, for the first time -- these are
# PRE-EXISTING wrong annotations, not regressions caused by the fix. Mirrors
# scripts/ci/madaros_corpus_regression_gate.sh: compare the failure LIST
# against a checked-in baseline (tests/vacuous_expect_baseline.txt) and
# tolerate only entries already listed there, so the fix can land without
# turning the required `full-test-suite` CI job red. Unlike
# KNOWN_FAILURES_FILE above, this baseline is always active (not gated to
# --format junit with no filter) so it also covers local/manual runs.
#
# Regenerate: SOUNIO_VACUOUS_BASELINE_REFRESH=1 bash scripts/run_sio_test_suite.sh
VACUOUS_BASELINE_FILE="$ROOT_DIR/tests/vacuous_expect_baseline.txt"
VACUOUS_REFRESH="${SOUNIO_VACUOUS_BASELINE_REFRESH:-0}"
declare -A VACUOUS_BASELINE_MAP=()
if [[ "$VACUOUS_REFRESH" != "1" && -f "$VACUOUS_BASELINE_FILE" ]]; then
    while IFS= read -r line; do
        line="${line%%#*}"
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [[ -z "$line" ]] && continue
        VACUOUS_BASELINE_MAP["$line"]=1
    done < "$VACUOUS_BASELINE_FILE"
fi

# JUnit XML output file
JUNIT_FILE="${SOUNIO_TEST_JUNIT_FILE:-$ROOT_DIR/test-results.xml}"

# Temporary directory for parallel execution results
TMPDIR="${TMPDIR:-/tmp}"
TEST_TMP=$(mktemp -d "$TMPDIR/sounio-test-XXXXXX")
if [[ -z "${SOUNIO_TEST_PRESERVE_JSON:-}" ]]; then
    trap "rm -rf $TEST_TMP" EXIT
fi
if [[ -n "${SOUNIO_TEST_RESULTS_DIR:-}" ]]; then
    mkdir -p "$SOUNIO_TEST_RESULTS_DIR"
fi

test_matches_filter() {
    local basename="$1"
    case "$FILTER_MODE" in
        contains)
            [[ -z "$FILTER" || "$basename" == *"$FILTER"* ]]
            ;;
        prefix)
            [[ -z "$FILTER" || "$basename" == "$FILTER"* ]]
            ;;
        exact)
            [[ -z "$FILTER" || "$basename" == "$FILTER" ]]
            ;;
        *)
            echo "error: unknown filter mode: $FILTER_MODE" >&2
            return 2
            ;;
    esac
}

# Function to run a single test
source "$ROOT_DIR/scripts/dev/sio_test_case.sh"

# Collect all test files
TEST_FILES=()
TEST_LIST_HEADER_MODE=""
TEST_LIST_HEADER_FILTER=""
if [[ -n "$TEST_LIST_FILE" ]]; then
    if [[ ! -f "$TEST_LIST_FILE" ]]; then
        echo "error: --test-list file not found: $TEST_LIST_FILE" >&2
        exit 2
    fi
    while IFS= read -r f; do
        if [[ "$f" == "# sounio-test-list"$'\t'* ]]; then
            IFS=$'\t' read -r _ TEST_LIST_HEADER_MODE TEST_LIST_HEADER_FILTER <<< "$f"
            continue
        fi
        f="${f%%#*}"
        f="${f#"${f%%[![:space:]]*}"}"
        f="${f%"${f##*[![:space:]]}"}"
        [[ -z "$f" ]] && continue
        if [[ "$f" != /* ]]; then
            f="$ROOT_DIR/$f"
        fi
        if [[ ! -f "$f" ]]; then
            echo "error: --test-list entry not found: $f" >&2
            exit 2
        fi
        TEST_FILES+=("$f")
    done <"$TEST_LIST_FILE"
else
    for f in "$ROOT_DIR"/tests/run-pass/*.sio; do
        [[ -f "$f" ]] && TEST_FILES+=("$f")
    done
    for f in "$ROOT_DIR"/tests/compile-fail/*.sio; do
        [[ -f "$f" ]] && TEST_FILES+=("$f")
    done
    for f in "$ROOT_DIR"/tests/ui/type/*.sio "$ROOT_DIR"/tests/ui/effect/*.sio "$ROOT_DIR"/tests/ui/ownership/*.sio "$ROOT_DIR"/tests/ui/resolve/*.sio "$ROOT_DIR"/tests/ui/pattern/*.sio; do
        [[ -f "$f" ]] && TEST_FILES+=("$f")
    done
    for f in "$ROOT_DIR"/tests/stdlib/*/test_*.sio; do
        [[ -f "$f" ]] && TEST_FILES+=("$f")
    done
    for f in "$ROOT_DIR"/tests/gpu/*.sio; do
        [[ -f "$f" ]] && TEST_FILES+=("$f")
    done
    # Self-hosted compiler tests (e.g. tests/self-hosted/gen_fano_raw_kernel.sio).
    # Single level, same as every glob above. These were invisible to the suite
    # because this list never scanned the directory, so a CI run measured
    # nothing about them while reporting green. Additive: the globs above are
    # untouched, and the [[ -f ]] guard makes an absent directory select
    # nothing rather than fail (same contract as the globs above). Tests here
    # are gated by their own annotations like every other fixture -- a
    # `//@ requires: madaros` header is enforced by run_test, so these run
    # only on the Madaros job (SOUNIO_MADAROS_AVAILABLE=1) and skip elsewhere.
    for f in "$ROOT_DIR"/tests/self-hosted/*.sio; do
        [[ -f "$f" ]] && TEST_FILES+=("$f")
    done
fi

if [[ -n "$TEST_LIST_HEADER_MODE" ]]; then
    if [[ "$TEST_LIST_HEADER_MODE" != "$FILTER_MODE" || "$TEST_LIST_HEADER_FILTER" != "$FILTER" ]]; then
        echo "error: --test-list filter header does not match active filter" >&2
        echo "  header: mode=$TEST_LIST_HEADER_MODE filter=$TEST_LIST_HEADER_FILTER" >&2
        echo "  active: mode=$FILTER_MODE filter=$FILTER" >&2
        exit 2
    fi
fi

if [[ -n "$TEST_LIST_FILE" && -n "$FILTER" ]]; then
    for f in "${TEST_FILES[@]}"; do
        basename="$(basename "$f")"
        if ! test_matches_filter "$basename"; then
            echo "error: --test-list entry does not match active filter: ${f#$ROOT_DIR/}" >&2
            exit 2
        fi
    done
fi

if [[ "$LIST_TESTS" == "1" ]]; then
    for f in "${TEST_FILES[@]}"; do
        basename="$(basename "$f")"
        test_matches_filter "$basename" || continue
        printf '%s\n' "${f#$ROOT_DIR/}"
    done
    exit 0
fi

export SOUC_BIN ROOT_DIR FILTER TEST_TMP SOUNIO_STDLIB_PATH CI SOUNIO_GPU_AVAILABLE SOUNIO_LLVM_AVAILABLE

# Header
echo "=== Sounio Test Suite ==="
echo "Using souc: $SOUC_BIN"
echo "Parallel jobs: $JOBS"
if [[ -n "${SOUC_NATIVE_BIN:-}" ]]; then
    echo "Using native backend: $SOUC_NATIVE_BIN"
fi
echo ""

echo "Found ${#TEST_FILES[@]} test files"
echo ""

# Every test that passes the filter must produce exactly one result_*.json:
# each path through run_test after the filter check writes one, and nothing
# else in TEST_TMP does. Counting them here lets the collector prove below
# that the summary is a measurement of this run -- a worker killed before its
# write (OOM under the job cap is the realistic producer) otherwise vanishes
# from the totals, and a dropped failing test is a silent green.
EXPECTED_RESULTS=0
for f in "${TEST_FILES[@]}"; do
    basename="$(basename "$f")"
    if test_matches_filter "$basename"; then
        ((EXPECTED_RESULTS++))
    fi
done

# Run tests in parallel with job limit
job_count=0
idx=0
for f in "${TEST_FILES[@]}"; do
    ((idx++))
    run_test "$f" "$idx" &
    ((job_count++))
    if [[ $job_count -ge $JOBS ]]; then
        wait -n 2>/dev/null || wait
        ((job_count--))
    fi
done
wait

# Collect results
RESULT_FILES=0
UNPARSED=0
for f in "$TEST_TMP"/result_*.json; do
    [[ -f "$f" ]] || continue
    ((RESULT_FILES++))
    result=$(cat "$f")
    status=$(echo "$result" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
    category=$(echo "$result" | grep -o '"category":"[^"]*"' | cut -d'"' -f4)
    name=$(echo "$result" | grep -o '"name":"[^"]*"' | cut -d'"' -f4)
    
    case "$status" in
        pass)
            ((PASS++))
            if [[ "$VERBOSE" == "1" ]]; then
                echo "  PASS  $name"
            fi
            ;;
        fail)
            ((FAIL++))
            output=$(echo "$result" | grep -o '"output":"[^"]*"' | cut -d'"' -f4)
            ERRORS="${ERRORS}  FAIL  $name"
            [[ -n "$output" ]] && ERRORS="$ERRORS ($output)"
            ERRORS="$ERRORS
"
            ;;
        xfail)
            ((KNOWN_FAILURE++))
            ;;
        xpas)
            # A known-failure that passes is a stale claim, not a green test.
            # Counted separately and always announced: swallowing it as PASS
            # is how 240 imported/native 139 tags sat green until a census
            # (docs/audit/KNOWN_FAILURE_XPAS_SIGNAL_2026-08-18.md). Same
            # lesson as vxpas below.
            ((XPAS++))
            XPAS_LIST="${XPAS_LIST}    $name
"
            echo "  XPAS  $name (known failure now passes)"
            ;;
        vxfail)
            # Tolerated because it is listed in tests/vacuous_expect_baseline.txt.
            # Counted and reported, never silently dropped: a tolerated failure
            # that does not appear in the summary is indistinguishable from a
            # test that never ran.
            ((VACUOUS_KNOWN++))
            if [[ "$VERBOSE" == "1" ]]; then
                echo "  VXFAIL  $name (vacuous-annotation baseline)"
            fi
            ;;
        vxpas)
            # Listed in the baseline but now passing. Always announced, not only
            # under --verbose: the baseline must shrink as annotations are fixed,
            # and a stale entry silently absorbing a pass is how a baseline rots
            # into a permanent mute.
            ((PASS++))
            VACUOUS_STALE="${VACUOUS_STALE}    $name
"
            ;;
        skip)
            ((SKIP++))
            reason=$(echo "$result" | grep -o '"reason":"[^"]*"' | cut -d'"' -f4)
            if [[ "$VERBOSE" == "1" ]]; then
                echo "  SKIP  $name ($reason)"
            fi
            ;;
        *)
            # A result file whose status cannot be parsed -- truncated or
            # malformed JSON, the realistic producer being a worker killed
            # mid-write. Dropping it made the totals undercount: a failing
            # test that vanishes is a silent green (the same reasoning as the
            # vxfail comment above, applied to the parse-failure path).
            # Count it and fail below instead of silently omitting it.
            ((UNPARSED++))
            echo "  UNPARSED  $f (status=[$status])" >&2
            ;;
    esac
    if [[ "$status" == "skip" ]]; then
        reason=$(echo "$result" | grep -o '"reason":"[^"]*"' | cut -d'"' -f4)
        if [[ -n "$reason" && "$VERBOSE" == "1" ]]; then
            echo "    reason: $reason"
        fi
    fi
done

# Generate output
echo ""
echo "=== Results ==="
echo "  Pass: $PASS"
echo "  Fail: $FAIL"
[[ $KNOWN_FAILURE -gt 0 ]] && echo "  Known failures: $KNOWN_FAILURE"
[[ $XPAS -gt 0 ]] && echo "  Unexpected passes (stale known-failure): $XPAS"
[[ $VACUOUS_KNOWN -gt 0 ]] && echo "  Vacuous-annotation baseline (tolerated): $VACUOUS_KNOWN"
[[ $FLAKY -gt 0 ]] && echo "  Flaky: $FLAKY"
echo "  Skip: $SKIP"
echo "  Total: $((PASS + FAIL + SKIP + KNOWN_FAILURE + VACUOUS_KNOWN + XPAS))"
[[ $UNPARSED -gt 0 ]] && echo "  Unparsed: $UNPARSED"

# Completeness: the counts above must describe every filtered test exactly
# once. Three ways to lose that, each previously silent:
#   - a result file no parser branch recognizes (UNPARSED above);
#   - a filtered test whose result file never appeared (worker died first);
#   - a run that selected nothing at all (0 == 0 reading as "all tests
#     passed" -- the vacuity failure mode; in CI an empty selection means the
#     corpus or the glob broke, so it must fail, not report green).
# CI-state readers must distinguish "the instrument did not answer" from "the
# answer was an empty selected set" (docs/governance/CI_TRUST_CONTRACT.md).
if [[ $UNPARSED -gt 0 ]]; then
    echo "INCOMPLETE: $UNPARSED result file(s) had no readable status -- verdicts unknown, failing instead of dropping them" >&2
    exit 1
fi
if [[ $RESULT_FILES -ne $EXPECTED_RESULTS ]]; then
    echo "INCOMPLETE: $EXPECTED_RESULTS filtered test(s) ran but $RESULT_FILES result file(s) were found -- a worker exited before writing its result, so the totals above undercount" >&2
    exit 1
fi
if [[ $EXPECTED_RESULTS -eq 0 ]]; then
    if [[ -n "${CI:-}" ]]; then
        echo "INCOMPLETE: zero test files matched (filter=${FILTER:-none}) -- this run measured nothing; refusing to report green on an empty suite" >&2
        exit 1
    fi
    echo "WARNING: no test files matched the active filter -- this run measured nothing" >&2
fi

if [[ -n "$XPAS_LIST" ]]; then
    echo ""
    echo "=== Known-failure tags that passed in THIS run ==="
    printf '%s' "$XPAS_LIST"
    echo "  A known-failure that passes is a stale claim about this engine,"
    echo "  not a green test. Drop the tag, or add //@ requires: <engine> if"
    echo "  the claim is about a different engine than the one that just ran."
    echo "  Madaros decides a Madaros-named tag; lean_single decides whether"
    echo "  the file needs requires: madaros. Zeros on one engine do not"
    echo "  license dropping a tag about the other."
fi

if [[ -n "$VACUOUS_STALE" ]]; then
    echo ""
    echo "=== Vacuous-annotation baseline entries that passed in THIS run ==="
    printf '%s' "$VACUOUS_STALE"
    echo "  The baseline is calibrated against the CI engine (souc-stage2 /"
    echo "  lean_single, via SOUNIO_TEST_SOUC_BIN). Under a different engine most"
    echo "  entries pass, so this list is only a removal instruction when the run"
    echo "  used the CI engine. Confirm there before deleting an entry."
fi

# Generate JUnit XML if requested
if [[ "$FORMAT" == "junit" ]]; then
    cat > "$JUNIT_FILE" << 'XMLEOF'
<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
XMLEOF
    
    echo "  <testsuite name=\"sounio-test-suite\" tests=\"$((PASS + FAIL + KNOWN_FAILURE + XPAS))\" failures=\"$((FAIL + XPAS))\" skipped=\"$SKIP\" errors=\"0\">" >> "$JUNIT_FILE"
    
    for f in "$TEST_TMP"/result_*.json; do
        [[ -f "$f" ]] || continue
        result=$(cat "$f")
        name=$(echo "$result" | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | sed 's/\.sio$//')
        [[ -n "$name" ]] || continue
        status=$(echo "$result" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
        time=$(echo "$result" | grep -o '"time":[^,}]*' | cut -d':' -f2)
        output=$(echo "$result" | grep -o '"output":"[^"]*"' | cut -d'"' -f4)
        
        case "$status" in
            pass)
                echo "    <testcase name=\"$name\" time=\"$time\"/>" >> "$JUNIT_FILE"
                ;;
            xpas)
                echo "    <testcase name=\"$name\" time=\"$time\">" >> "$JUNIT_FILE"
                echo "      <failure message=\"stale known-failure: test now passes on this engine\"/>" >> "$JUNIT_FILE"
                echo "    </testcase>" >> "$JUNIT_FILE"
                ;;
            fail)
                echo "    <testcase name=\"$name\" time=\"$time\">" >> "$JUNIT_FILE"
                echo "      <failure message=\"$output\"/>" >> "$JUNIT_FILE"
                echo "    </testcase>" >> "$JUNIT_FILE"
                ;;
            xfail)
                echo "    <testcase name=\"$name\" time=\"$time\">" >> "$JUNIT_FILE"
                echo "      <skipped message=\"Known failure\"/>" >> "$JUNIT_FILE"
                echo "    </testcase>" >> "$JUNIT_FILE"
                ;;
            vxfail)
                echo "    <testcase name=\"$name\" time=\"$time\">" >> "$JUNIT_FILE"
                echo "      <skipped message=\"Vacuous-annotation baseline: $output\"/>" >> "$JUNIT_FILE"
                echo "    </testcase>" >> "$JUNIT_FILE"
                ;;
            vxpas)
                echo "    <testcase name=\"$name\" time=\"$time\">" >> "$JUNIT_FILE"
                echo "      <system-out>Vacuous-annotation baseline entry now passes; remove it</system-out>" >> "$JUNIT_FILE"
                echo "    </testcase>" >> "$JUNIT_FILE"
                ;;
            skip)
                echo "    <testcase name=\"$name\" time=\"$time\">" >> "$JUNIT_FILE"
                echo "      <skipped/>" >> "$JUNIT_FILE"
                echo "    </testcase>" >> "$JUNIT_FILE"
                ;;
        esac
    done
    
    echo "  </testsuite>" >> "$JUNIT_FILE"
    echo "</testsuites>" >> "$JUNIT_FILE"
    
    echo ""
    echo "JUnit XML written to: $JUNIT_FILE"
fi

if [[ $FAIL -gt 0 ]]; then
    echo ""
    echo "=== Failures ==="
    echo -n "$ERRORS"
    exit 1
fi

# SOUNIO_XPAS_FATAL=1 makes a stale known-failure tag fail the job.
# Default off until the remaining seed XPASses owned by other lanes
# (gum_fo_across_call, turbofish_concrete_type_mismatch) are classified.
# The Madaros known-failure recheck sets this so compiler-only PRs cannot
# rot requires:madaros tags in silence.
if [[ $XPAS -gt 0 && "${SOUNIO_XPAS_FATAL:-}" == "1" ]]; then
    echo ""
    echo "XPAS_FATAL: $XPAS known-failure tag(s) passed on this engine"
    exit 1
fi

echo ""
if [[ $XPAS -gt 0 ]]; then
    echo "Suite finished with $XPAS stale known-failure tag(s) (not a silent pass)."
else
    echo "All tests passed!"
fi
