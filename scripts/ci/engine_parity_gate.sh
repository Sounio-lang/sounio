#!/usr/bin/env bash
# Differential engine-parity gate: run the corpus under BOTH engines and compare
# what the programs actually print.
#
# Why this exists
# ---------------
# Every instrument in this repository so far asks one engine one question:
#
#   - full-test-suite            runs lean_single, and its `expect-stdout`
#                                extraction is quoted so the capture comes back
#                                empty and the assertion matches vacuously
#                                (scripts/dev/run_sio_test_suite.sh:306).
#   - madaros_corpus_regression  runs Madaros, and compares exit status only.
#
# Neither can see a program that compiles, exits 0, and computes the wrong
# answer. Measured on 2026-07-26, that blind spot was hiding at least two
# defects of one shape:
#
#   #1504  gpu_thread_id_x() named a function nothing bound, so its destination
#          register was never written and read as 0 — every kernel ran as lane 0.
#   #1502  hessian_of() is bound by check.sio:3342 as an unknown import and
#          implemented nowhere on the Madaros path, so it returns 0.0 for every
#          entry. Its test asserted three properties, two of which expected zero
#          and therefore could not fail.
#
# Both were invisible to a single-engine instrument and obvious the moment the
# two engines were asked the same question. This gate asks it for every program.
#
# What it does NOT do
# -------------------
# It does not decide which engine is right. Divergence is the signal; which side
# is correct is a judgement call for the reviewer. A program that fails to
# compile, times out, or crashes under an engine is recorded as such rather than
# silently dropped, because "did not run" and "agreed" must never look alike —
# that conflation is the same bug this gate exists to catch.
#
# Usage
#   scripts/ci/engine_parity_gate.sh                    # gate against baseline
#   scripts/ci/engine_parity_gate.sh --update-baseline  # regenerate baseline
#   scripts/ci/engine_parity_gate.sh --only PATTERN     # triage a subset
#
# `--only` is for triage and for regenerating a scoped baseline. Do NOT gate
# with it: the observed set is compared against the WHOLE baseline, so every
# entry outside the filter reads as "improved". Gate on the full corpus.
#   SOUNIO_PARITY_JOBS=8 scripts/ci/engine_parity_gate.sh
#
# Environment
#   SOUNIO_PARITY_MADAROS  Madaros ELF        (default artifacts/self-hosted/madaros)
#   SOUNIO_PARITY_LEAN     lean_single ELF    (default bin/souc-lean-single-x86_64)
#   SOUNIO_PARITY_JOBS     parallelism        (default: cores-2, capped at 8)
#   SOUNIO_PARITY_TIMEOUT  per-run seconds    (default 120; see #1591)

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 1

MADAROS="${SOUNIO_PARITY_MADAROS:-$ROOT_DIR/artifacts/self-hosted/madaros}"
LEAN="${SOUNIO_PARITY_LEAN:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
BASELINE="$ROOT_DIR/tests/engine_parity_baseline.txt"
# 30s put the imported lorenz/solver_portfolio family right on the boundary, so
# their verdict flipped between LEAN-ONLY and NEITHER run to run with nothing in
# either compiler changing. Measured on a quiet machine with the old limit:
#   lorenz_i256_ball_fixed_bridge_imported   rc=124 at 30189ms
#   solver_portfolio_v16_coverage_imported   rc=0   at 29198ms  <- 800ms of margin
#   lorenz_i256_step_certificate_imported    rc=124 at 30192ms
# The slowest of them finishes in 68499ms when allowed to. They are slow, not
# hung, so the bound now clears the measured worst case with headroom (#1591).
TIMEOUT="${SOUNIO_PARITY_TIMEOUT:-120}"

cores=$(nproc 2>/dev/null || echo 4)
default_jobs=$(( cores - 2 ))
[ "$default_jobs" -lt 1 ] && default_jobs=1
[ "$default_jobs" -gt 8 ] && default_jobs=8
JOBS="${SOUNIO_PARITY_JOBS:-$default_jobs}"

UPDATE_BASELINE=0
ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --update-baseline) UPDATE_BASELINE=1; shift ;;
        --only) ONLY="${2:-}"; shift 2 ;;
        *) echo "engine-parity: unknown argument: $1" >&2; exit 2 ;;
    esac
done

for bin in "$MADAROS" "$LEAN"; do
    if [ ! -x "$bin" ]; then
        echo "[engine-parity] FAIL: compiler not executable: $bin" >&2
        echo "[engine-parity] build Madaros with scripts/ci/build_modular_madaros.sh" >&2
        exit 1
    fi
done

# A prebuilt binary that lags the working tree makes every verdict meaningless.
# Refuse rather than produce a confident wrong answer.
if [ -n "$(find self-hosted -name '*.sio' -newer "$MADAROS" -print -quit 2>/dev/null)" ]; then
    echo "[engine-parity] FAIL: $MADAROS is older than a self-hosted source file." >&2
    echo "[engine-parity] Rebuild first — a stale compiler cannot certify parity." >&2
    exit 1
fi

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-parity.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# --- per-file worker -------------------------------------------------------
# Emits one line: "<status>\t<relpath>" where status is one of
#   AGREE          both engines produced identical stdout
#   DIVERGE        both ran, stdout differs
#   MADAROS-ONLY   only Madaros produced a running binary
#   LEAN-ONLY      only lean_single did
#   NEITHER        neither did
#   NONDETERMINISTIC  the program does not agree with ITSELF across two runs of
#                  the same engine, so no byte comparison between engines can
#                  mean anything (typically `println` of a struct, which prints
#                  an address)
#   TIMEOUT        at least one engine was killed by the clock -- "too slow to
#                  measure" is NOT "neither engine builds this", and conflating
#                  them made a slow program read as a rejected one (#1591)
cat > "$WORK/parity_one.sh" <<'WORKER'
#!/usr/bin/env bash
set -uo pipefail
src="$1"; madaros="$2"; lean="$3"; work="$4"; to="$5"
tag=$(printf '%s' "$src" | tr '/.' '__')

run_engine() {
    # Separate declarations: under `set -u`, referencing ${kind} in the same
    # `local` statement that assigns it expands before the assignment lands.
    local kind="$1"
    local elf="$work/${tag}_${kind}.elf"
    local out="$work/${tag}_${kind}.out"
    rm -f "$elf"
    local crc
    if [ "$kind" = "mad" ]; then
        ( ulimit -v 8000000 2>/dev/null || true; timeout "$to" "$madaros" "$src" -o "$elf" ) >/dev/null 2>&1
        crc=$?
    else
        ( ulimit -v 8000000 2>/dev/null || true; timeout "$to" "$lean" "$src" "$elf" ) >/dev/null 2>&1
        crc=$?
    fi
    if [ ! -s "$elf" ]; then
        # EXACTLY 124 is timeout(1) reporting that IT killed the compiler.
        # Anything above that is 128+signal -- a crash, not a clock. Using
        # -ge here misfiled every SIGSEGV (139) as a timeout, which is the
        # same conflation this change exists to remove.
        [ "$crc" -eq 124 ] && return 2
        return 1
    fi
    chmod +x "$elf" 2>/dev/null
    timeout "$to" "$elf" >"$out" 2>/dev/null
    local rc=$?
    rm -f "$elf"
    # A crash or timeout is not an output to compare, so neither can be mistaken
    # for agreement -- but they are reported differently. Exactly 124 is the
    # clock; 128+signal (139 = SIGSEGV, 137 = SIGKILL) is the program dying,
    # and those stay "did not run" exactly as before. A nonzero exit below 124
    # still counts as "ran": programs that print FAIL and exit 1 are real
    # observations and were always compared.
    [ "$rc" -eq 124 ] && return 2
    [ "$rc" -gt 124 ] && return 1
    return 0
}

ok_m=0; ok_l=0; slow_m=0; slow_l=0
run_engine mad;  m_rc=$?
[ "$m_rc" -eq 0 ] && ok_m=1
[ "$m_rc" -eq 2 ] && slow_m=1
run_engine lean; l_rc=$?
[ "$l_rc" -eq 0 ] && ok_l=1
[ "$l_rc" -eq 2 ] && slow_l=1

if [ "$ok_m" = 1 ] && [ "$ok_l" = 1 ]; then
    if cmp -s "$work/${tag}_mad.out" "$work/${tag}_lean.out"; then
        printf 'AGREE\t%s\n' "$src"
    else
        # Before calling it a divergence, check the program is comparable at all.
        # `println` of a struct prints its ADDRESS, so such a program prints
        # something different on every run under EITHER engine -- three in the
        # corpus do (covid_2020_kernel, epsilon_comparison_valid,
        # knightian_syntax, all `println(<Knowledge struct>)`). Byte-comparing
        # those says nothing about the engines, and pinning three filenames would
        # go stale, so the property is MEASURED: run each engine a second time and
        # see whether it still agrees with itself.
        cp "$work/${tag}_mad.out" "$work/${tag}_mad.out1" 2>/dev/null
        cp "$work/${tag}_lean.out" "$work/${tag}_lean.out1" 2>/dev/null
        nondet=0
        run_engine mad  && { cmp -s "$work/${tag}_mad.out"  "$work/${tag}_mad.out1"  || nondet=1; }
        run_engine lean && { cmp -s "$work/${tag}_lean.out" "$work/${tag}_lean.out1" || nondet=1; }
        rm -f "$work/${tag}_mad.out1" "$work/${tag}_lean.out1"
        if [ "$nondet" = 1 ]; then
            printf 'NONDETERMINISTIC\t%s\n' "$src"
        else
            printf 'DIVERGE\t%s\n' "$src"
        fi
    fi
elif [ "$ok_m" = 1 ]; then
    printf 'MADAROS-ONLY\t%s\n' "$src"
elif [ "$ok_l" = 1 ]; then
    printf 'LEAN-ONLY\t%s\n' "$src"
elif [ "$slow_m" = 1 ] || [ "$slow_l" = 1 ]; then
    printf 'TIMEOUT\t%s\n' "$src"
else
    printf 'NEITHER\t%s\n' "$src"
fi
rm -f "$work/${tag}_mad.out" "$work/${tag}_lean.out"
WORKER
chmod +x "$WORK/parity_one.sh"

# --- corpus ----------------------------------------------------------------
# Skip sources that cannot be programs. The glob used to take every .sio under
# tests/run-pass and compile+RUN it, including library leaves and fixtures with
# no `fn main`. Their ELF has no entry point, so it dies with SIGSEGV and the
# gate recorded that as a parity observation -- "the ELF of a main-less library
# segfaulted" says nothing about whether two engines agree, and it inflated
# NEITHER/MADAROS-ONLY counts that get cited as capability measurements (#1593).
#
# TWO exclusions only:
#   //@ ignore    the file says skip me
#   no `fn main`  there is no program to run
#
# `//@ check-only` is deliberately NOT an exclusion, though the issue proposed
# it. Measured: of the 32 check-only files, 15 also declare //@ run-pass, and
# clinical_dyadic_non_reduction_witness.sio -- check-only with no run-pass --
# has a main, executes, and AGREES byte-for-byte across both engines. Its own
# header explains why: "Madaros typechecks this two-module program and can
# execute it through the full-IR route after the compact modular emitter
# declines it." The marker says which harness checks the file, not whether it
# runs, so excluding on it would have deleted 16 real observations while looking
# like housekeeping.
#
# Header-only match (first 8 lines): `//@` is a file directive, and matching
# further in would catch prose in comments.
parity_is_runnable_program() {
    local f="$1"
    if head -n 8 "$f" | grep -qE '^//@[[:space:]]*ignore\b'; then
        return 1
    fi
    grep -qE '^[[:space:]]*(pub[[:space:]]+)?fn[[:space:]]+main[[:space:]]*\(' "$f"
}

collect_sources() {
    local f
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        if parity_is_runnable_program "$f"; then
            printf '%s\n' "$f"
        fi
    done
}

if [ -n "$ONLY" ]; then
    sources=$(find tests/run-pass -name '*.sio' | grep -F "$ONLY" | sort | collect_sources)
else
    sources=$(find tests/run-pass -name '*.sio' | sort | collect_sources)
fi

total=$(printf '%s\n' "$sources" | grep -c . || true)
if [ "$total" -eq 0 ]; then
    echo "[engine-parity] FAIL: no sources selected" >&2
    exit 1
fi
echo "[engine-parity] corpus=$total jobs=$JOBS timeout=${TIMEOUT}s"
# Progress is checkable without polling the process table. A full run compiles and
# executes every program under BOTH engines and takes ~13 min on a 6-job pod, which
# is long enough that callers reach for a wait loop.
#
# Do NOT wait with `until ! pgrep -f engine_parity_gate`: that pattern matches the
# waiting shell's own command line, so two such loops see each other and block
# forever. Measured 2026-07-28 — three of them deadlocked for 5h13 on a gate that
# had already exited. Poll this file instead, or run the gate in the foreground
# and read its exit status.
echo "[engine-parity] progress: wc -l $WORK/results.tsv   (of $total)"
echo "[engine-parity] madaros=$MADAROS"
echo "[engine-parity] lean   =$LEAN"

printf '%s\n' "$sources" \
    | xargs -P "$JOBS" -I{} "$WORK/parity_one.sh" {} "$MADAROS" "$LEAN" "$WORK" "$TIMEOUT" \
    > "$WORK/results.tsv" 2>/dev/null

sort -o "$WORK/results.tsv" "$WORK/results.tsv"

# Completeness floor: parity_one prints exactly one verdict row per program,
# so any shortfall is a worker that died before answering (OOM kill, broken
# worker script, dead xargs) -- not a cleaner corpus. Under a wholesale
# failure every count below reads 0, observed.txt reads empty, and comm
# reports "no new divergences"; zero evidence must not read as a clean run
# (the same contract the v2 suite's result-completeness asserts enforce).
n_rows=$(grep -c . "$WORK/results.tsv" || true)
if [ "$n_rows" -ne "$total" ]; then
    echo "[engine-parity] FAIL: incomplete results -- $n_rows of $total programs answered" >&2
    cut -f2 "$WORK/results.tsv" | sort > "$WORK/answered.txt"
    printf '%s\n' "$sources" | sort > "$WORK/expected.txt"
    comm -13 "$WORK/answered.txt" "$WORK/expected.txt" | sed 's/^/  no-verdict: /' >&2
    exit 1
fi

agree=$(grep -c '^AGREE'           "$WORK/results.tsv" || true)
diverge=$(grep -c '^DIVERGE'       "$WORK/results.tsv" || true)
mad_only=$(grep -c '^MADAROS-ONLY' "$WORK/results.tsv" || true)
lean_only=$(grep -c '^LEAN-ONLY'   "$WORK/results.tsv" || true)
neither=$(grep -c '^NEITHER'       "$WORK/results.tsv" || true)
timedout=$(grep -c '^TIMEOUT'      "$WORK/results.tsv" || true)
nondet_n=$(grep -c '^NONDETERMINISTIC' "$WORK/results.tsv" || true)

echo "[engine-parity] agree=$agree diverge=$diverge madaros-only=$mad_only lean-only=$lean_only neither=$neither timeout=$timedout nondet=$nondet_n"

# The baseline records every non-AGREE verdict. AGREE is the goal state and is
# deliberately not recorded, so the file shrinks as the engines converge.
grep -v '^AGREE' "$WORK/results.tsv" > "$WORK/observed.txt" || true

if [ "$UPDATE_BASELINE" = 1 ]; then
    cp "$WORK/observed.txt" "$BASELINE"
    echo "[engine-parity] baseline updated: $(wc -l < "$BASELINE") entries"
    exit 0
fi

if [ ! -f "$BASELINE" ]; then
    echo "[engine-parity] FAIL: no baseline at $BASELINE — run with --update-baseline" >&2
    exit 1
fi

new=$(comm -13 "$BASELINE" "$WORK/observed.txt" || true)
fixed=$(comm -23 "$BASELINE" "$WORK/observed.txt" || true)

new_count=$(printf '%s\n' "$new" | grep -c . || true)
fixed_count=$(printf '%s\n' "$fixed" | grep -c . || true)

if [ "$fixed_count" -gt 0 ]; then
    echo "[engine-parity] $fixed_count entries improved (now agreeing or newly running):"
    printf '%s\n' "$fixed" | sed 's/^/  + /'
fi

if [ "$new_count" -gt 0 ]; then
    echo "[engine-parity] FAIL: $new_count new engine divergences" >&2
    printf '%s\n' "$new" | sed 's/^/  - /' >&2
    exit 1
fi

echo "[engine-parity] PASS: no new engine divergences ($(wc -l < "$BASELINE") known)"
exit 0
