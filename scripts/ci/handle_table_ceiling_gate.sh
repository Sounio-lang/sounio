#!/usr/bin/env bash
# handle_table_ceiling_gate.sh — refuse silent handle-table exhaustion.
#
# ENGINE: Madaros (default bin/souc after source build). lean_single is not
# the contract surface for this bug (measured on Madaros).
#
# Measured boundary on prebuilt Madaros (2026-08-17):
#   - Capacity is 4194304 (2^22), set in self-hosted/native/gc.sio
#   - Exhaustion is fail-closed emit_exit(c.code, 182) at codegen_x86_linux.sio:6379
#   - Pre-patch: the exit printed nothing; the user had to re-derive what
#     happened from rc=182 and partial output (silent-defect class).
#
# Patch (.scratch/e230_diagnostic.patch) v3 adds:
#   - 90% drift warning fired once per process at handle_count = 3774873
#     (floor(capacity * 9 / 10)). Prints
#         "madaros: warning[E230] drift 90% of capacity: count=3774873 of 4194304"
#   - Failure diagnostic for rc=181/182 now prints
#         "madaros: handles full: count=N of M (2^22)"
#     instead of returning silently.
#
# Witnesses below exercise the runtime diagnostic end-to-end. Each witness
# uses a 3-i64-field struct (W2/W3/W4: 24 bytes, > 16-byte unbox threshold)
# so each alloc consumes a handle — a 1-i64-field struct (8 bytes) would
# be returned in registers by SysV without ever touching the handle table.
#
# W1 (compile-time refusal of a > capacity MIR_OP_ALLOC program) was
# removed from this gate for two reasons:
#   (a) the source it generated exceeds the 16 MB / 2048 locals IR ceiling.
#   (b) the E230 arm in the checker is not yet present, so the
#       error[E230] grep would not match in any case.
# The Layer-1 path is documented as the design-layer path in
# docs/audit/HANDLE_TABLE_CEILING_REFUSAL_REFINEMENT_2026-08-17.md §3
# and will return when the checker grows the arm. For now W2 (warns at
# 90%) and W3 (refuses at 100%) cover the d2_gum-class loop-driven
# failure family.
#
# Witness contract:
#   W2 PASS: 3774973 allocs -> warning[E230] fires ONCE, rc=0
#   W3 PASS: 4194320 allocs -> warning at 90%, refusal (rc != 0)
#            with stderr naming the 100% crossing.
#   W4 PASS: 1 alloc -> rc=0, NO E230 anywhere.
#
# Honesty: a program that dies mid-study after printing partial results is
# WORSE than one that refuses to start. The E230 refusal names the
# ceiling and the program's demand so the user can fix the budget
# before re-running.
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

unset SOUC_BIN SOUNIO_SOUC_BIN SOUNIO_SOUC_ENGINE || true
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
SOUC="${SOUC:-$ROOT/bin/souc}"
if [[ ! -x "$SOUC" ]]; then
    echo "FAIL souc not executable: $SOUC" >&2
    exit 2
fi
if [[ "${SOUNIO_SOUC_ENGINE:-}" == "lean_single" ]]; then
    echo "FAIL this gate asserts Madaros handle-table emission; refuse lean_single" >&2
    exit 2
fi

echo "=== handle_table_ceiling_gate ==="
echo "engine=Madaros"
echo "souc=$SOUC"
echo "souc_version=$("$SOUC" --version 2>/dev/null | head -1 || echo unknown)"
if [[ -n "${MADAROS_RAW_BIN:-}" ]]; then
    echo "MADAROS_RAW_BIN=$MADAROS_RAW_BIN"
fi

# Capacity constant — must match self-hosted/native/gc.sio::native_v2_handle_table_capacity_default()
CAPACITY=4194304
# 90% drift threshold — floor(capacity * 9 / 10).
DRIFT_WARN_90=$((CAPACITY * 9 / 10))
# Witness budgets.
W2_ITERS=$((DRIFT_WARN_90 + 100))      # 3774973 — fires once at iter 3774873 (i.e. handle_count = 3774873)
W3_ITERS=$((CAPACITY + 16))            # 4194320 — crosses 100% at iter 4194303

TMP=$(mktemp -d "${TMPDIR:-/tmp}/handle-ceiling-gate.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

PASS=0
FAIL=0
pass() { PASS=$((PASS+1)); echo "PASS $1"; }
fail() { FAIL=$((FAIL+1)); echo "FAIL $1" >&2; }

###############################################################################
# Gate self-control: assert a generated witness source has the expected shape.
#
# Args:
#   $1     — label (W2..W4) for diagnostics
#   $2     — path to the generated .sio file
#   $3..N  — required string fragments (each must appear as a substring)
#
# Behaviour:
#   - If file is missing or empty: FAIL with "source file missing or empty",
#     print actual file path, exit 2 (distinct from witness pass/fail).
#   - If file is missing any required fragment: FAIL with the missing fragment,
#     dump the first 30 lines of the actual file so the caller can see why the
#     generator aborted, exit 2.
#   - Otherwise: return 0 and continue to the witness compile/run.
#
# Why this is the right control (not just [[ -s ]]):
#   - The first failure of this gate (var i=0 leaking Sounio syntax into the
#     W1 generator) aborted the heredoc BEFORE writing the expected content,
#     but the file *was* created (bash opens the redirect before running the
#     generator body). A size check alone would have missed it. The content
#     shape check is what surfaces the abort.
#   - Self-test failures exit 2 and tag the run as SELFTEST_FAIL, so a CI
#     consumer can distinguish gate-broke-itself from witness-really-failed.
###############################################################################
gate_assert_witness_source() {
    local label="$1"
    local path="$2"
    shift 2
    if [[ ! -s "$path" ]]; then
        echo "FAIL gate self-test [$label]: source file missing or empty: $path" >&2
        echo "HANDLE_TABLE_CEILING_GATE_SELFTEST_FAIL [$label]" >&2
        exit 2
    fi
    local frag
    for frag in "$@"; do
        if ! grep -qF -- "$frag" "$path"; then
            echo "FAIL gate self-test [$label]: source missing expected fragment: $frag" >&2
            echo "  actual file content (first 30 lines):" >&2
            sed -n '1,30p' "$path" | sed 's/^/    /' >&2
            echo "HANDLE_TABLE_CEILING_GATE_SELFTEST_FAIL [$label]" >&2
            exit 2
        fi
    done
}

###############################################################################
# run_witness <label> <sio> <elf>
#
# Compile <sio> with the configured Madaros compiler, then execute the
# resulting ELF under timeout. Echos a one-line "compile_rc=X run_rc=Y"
# summary; copies the combined stdout+stderr to "$TMP/<label>.combined.out"
# for grep matching by the caller.
#
# Returns: sets $COMPILE_RC and $RUN_RC in the caller (no shell export —
# the caller reads from $TMP/<label>.combined.out instead).
#
# Failure modes:
#   - compile non-zero            → echo error, run_rc unset.
#   - compile produces no ELF     → exit 2 (gate-broken, not witness-failed).
###############################################################################
run_witness() {
    local label="$1"
    local sio="$2"
    local elf="$3"
    local compile_out="$TMP/${label}.compile.out"
    local run_out="$TMP/${label}.run.out"
    local run_err="$TMP/${label}.run.err"
    local combined="$TMP/${label}.combined.out"

    local compile_rc=0 run_rc=0
    set +e
    timeout 300 "$SOUC" "$sio" -o "$elf" > "$compile_out" 2>&1
    compile_rc=$?
    set -e

    echo
    echo "compile_rc=$compile_rc run_rc=pending — $label"
    if [[ $compile_rc -ne 0 ]]; then
        echo "compile output (first 60 lines):" >&2
        sed -n '1,60p' "$compile_out" | sed 's/^/    /' >&2
        echo "FAIL gate self-test [$label]: compiler refused to build the witness — gate-broken" >&2
        echo "HANDLE_TABLE_CEILING_GATE_SELFTEST_FAIL [$label]" >&2
        exit 2
    fi
    if [[ ! -x "$elf" ]]; then
        echo "FAIL gate self-test [$label]: compile produced no ELF at $elf" >&2
        echo "HANDLE_TABLE_CEILING_GATE_SELFTEST_FAIL [$label]" >&2
        exit 2
    fi

    set +e
    timeout 600 "$elf" > "$run_out" 2> "$run_err"
    run_rc=$?
    set -e

    cat "$run_out" "$run_err" > "$combined" 2>/dev/null

    echo "run_rc=$run_rc"
    echo "combined stdout+stderr (last 20 lines):"
    tail -20 "$combined" | sed 's/^/    /'
    echo "COMPILE_RC=$compile_rc RUN_RC=$run_rc"
}

###############################################################################
# W2: 90% drift warning positive control — dynamic count crosses 90% but
# stays below 100%. Isolated from W3 (which crosses all bands and exits
# non-zero). W2 must fire `warning[E230] drift 90% of capacity` exactly
# once and exit 0.
#
# Iteration budget = 3774973 (DRIFT_WARN_90 + 100).
# - Iteration 3774873 (= DRIFT_WARN_90): before alloc, h = 3774873. Patch
#   compares h (NOT h+1) against 3774873; cmp fails JL → fire body runs.
#   The fire body sets the fired flag and prints one warning line.
# - Iterations 3774874..3774972: flag is set, skip.
# - Loop ends with h = 3774972, still < capacity → no refusal.
# - Program exits 0 (success).
#
# This is the CLEAN positive control: a fix that breaks the 90% band but
# keeps the 100% refusal would not be caught by W3 alone. W2 isolates the
# band.
###############################################################################
W2_TMP="$TMP/w2.sio"
cat > "$W2_TMP" <<EOF
// W2: 90% drift warning positive control — dynamic crosses 90%, not 100%.
// 3 i64 fields (tag = 24 > unbox threshold) so each alloc consumes a handle.
struct W2 { x: i64, y: i64, z: i64 }
fn alloc_one() -> W2 with Alloc { W2 { x: 1, y: 1, z: 1 } }
fn main() -> i64 with IO, Mut, Panic, Div, Alloc {
    var i: i64 = 0
    while i < $W2_ITERS {
        let _x = alloc_one()
        i = i + 1
    }
    print("done\n")
    0
}
EOF

gate_assert_witness_source W2 "$W2_TMP" \
    '// W2:' \
    'struct W2 { x: i64, y: i64, z: i64 }' \
    'fn alloc_one()' \
    'while i < '"$W2_ITERS" \
    'print("done\n")'

echo
echo "--- W2: loop with $W2_ITERS allocs (90% boundary at iteration $DRIFT_WARN_90) ---"
run_witness W2 "$W2_TMP" "$TMP/W2.elf"

# run_witness writes to "$TMP/${label}.*.out" — re-discover the exact paths
# from the label so that path naming stays consistent across the gate.
W2_RUN_OUT="$TMP/W2.run.out"
W2_RUN_ERR="$TMP/W2.run.err"
W2_RUN_RC=$(tail -25 "$TMP/W2.combined.out" | grep -E '^run_rc=' | tail -1 | awk -F= '{print $2}')
if [[ -z "$W2_RUN_RC" ]]; then W2_RUN_RC="missing"; fi

if [[ "$W2_RUN_RC" == "0" ]] && grep -qE "warning\[E230\].*drift.*90% of capacity|count=3774873.*of.*4194304" "$W2_RUN_ERR" "$W2_RUN_OUT"; then
    W2_FIRE_COUNT=$(cat "$W2_RUN_ERR" "$W2_RUN_OUT" | grep -cE "warning\[E230\]" || true)
    if [[ $W2_FIRE_COUNT -eq 1 ]]; then
        pass "W2 90% drift warning fires once with count=3774873 of 4194304, rc=0"
    else
        fail "W2 warning fired $W2_FIRE_COUNT times (expected exactly 1; runtime flag not gated)"
    fi
elif [[ "$W2_RUN_RC" == "0" ]] && cat "$W2_RUN_ERR" "$W2_RUN_OUT" | grep -q "warning\[E230\]"; then
    fail "W2 warning present but message format unexpected (expected 'drift 90% of capacity: count=3774873 of 4194304')"
elif [[ "$W2_RUN_RC" != "0" ]]; then
    fail "W2 expected rc=0; got rc=$W2_RUN_RC — patch may have broken the warning gate"
else
    fail "W2 expected warning[E230] in output; rc=0 but no warning printed — 90% drift detector missing"
fi

###############################################################################
# W3: hot-loop drift detector — small static count, large dynamic count.
#
# A program that allocates one struct inside a tight loop that iterates more
# than capacity times will eventually exhaust the table at runtime. The fix
# must produce a warning at 90% AND a refusal at 100%, each naming the count
# and capacity.
#
# Iteration budget = CAPACITY + 16 = 4194320.
# - iter 3774873 (= DRIFT_WARN_90): warning fires once (h before alloc = 3774873).
# - iter 4194303: rbx = h+1 = 4194304, cmp 4194304, capacity → SETAE al →
#   nc_core_emit_alloc_fail_into → nc_core_emit_alloc_failure_diagnostic_into(..., 182)
#   prints "madaros: handles full: count=4194304 of 4194304 (2^22)" then exit 182.
###############################################################################
W3_TMP="$TMP/w3.sio"
cat > "$W3_TMP" <<EOF
// W3: hot-loop drift detector witness — small static, large dynamic.
// 3 i64 fields (tag = 24 > unbox threshold) so each alloc consumes a handle.
struct W3 { x: i64, y: i64, z: i64 }
fn alloc_one() -> W3 with Alloc { W3 { x: 1, y: 1, z: 1 } }
fn main() -> i64 with IO, Mut, Panic, Div, Alloc {
    var i: i64 = 0
    while i < $W3_ITERS {
        let _x = alloc_one()
        i = i + 1
    }
    print("done\n")
    0
}
EOF

gate_assert_witness_source W3 "$W3_TMP" \
    '// W3:' \
    'struct W3 { x: i64, y: i64, z: i64 }' \
    'fn alloc_one()' \
    'while i < '"$W3_ITERS" \
    'print("done\n")'

echo
echo "--- W3: loop with $W3_ITERS allocs (capacity is $CAPACITY) ---"
run_witness W3 "$W3_TMP" "$TMP/W3.elf"

W3_RUN_OUT="$TMP/W3.run.out"
W3_RUN_ERR="$TMP/W3.run.err"
W3_RUN_RC=$(tail -25 "$TMP/W3.combined.out" | grep -E '^run_rc=' | tail -1 | awk -F= '{print $2}')
if [[ -z "$W3_RUN_RC" ]]; then W3_RUN_RC="missing"; fi

if [[ "$W3_RUN_RC" != "0" ]] && grep -qE "madaros: handles full: count=4194304 of 4194304" "$W3_RUN_ERR" "$W3_RUN_OUT"; then
    if cat "$W3_RUN_ERR" "$W3_RUN_OUT" | grep -qE "warning\[E230\].*drift.*90% of capacity"; then
        pass "W3 hot-loop drift: warning at 90% fires once, refusal at 100% with handles-full + (2^22) marker, rc=$W3_RUN_RC"
    else
        fail "W3 refused with handles-full marker but no 90% warning emitted"
    fi
else
    fail "W3 expected nonzero rc with 'handles full: count=4194304 of 4194304' marker; got rc=$W3_RUN_RC"
fi

###############################################################################
# W4: negative control — a tiny program must NOT print E230 / madaros: handles full.
###############################################################################
W4_TMP="$TMP/w4.sio"
cat > "$W4_TMP" <<'EOF'
// W4: negative control — one allocation, must not trigger E230.
// 3 i64 fields (tag = 24 > unbox threshold) so the alloc DOES consume a handle,
// but it stays well below the 90% drift threshold so no warning should fire.
struct W4 { x: i64, y: i64, z: i64 }
fn main() -> i64 with IO, Mut, Panic, Div, Alloc {
    let _x = W4 { x: 1, y: 1, z: 1 }
    print("hi\n")
    0
}
EOF

gate_assert_witness_source W4 "$W4_TMP" \
    '// W4:' \
    'struct W4 { x: i64, y: i64, z: i64 }' \
    'fn main() -> i64 with IO, Mut, Panic, Div, Alloc {' \
    'let _x = W4 { x: 1, y: 1, z: 1 }' \
    'print("hi\n")'

echo
echo "--- W4: negative control (1 alloc) ---"
run_witness W4 "$W4_TMP" "$TMP/W4.elf"

W4_RUN_OUT="$TMP/W4.run.out"
W4_RUN_ERR="$TMP/W4.run.err"
W4_RUN_RC=$(tail -25 "$TMP/W4.combined.out" | grep -E '^run_rc=' | tail -1 | awk -F= '{print $2}')
if [[ -z "$W4_RUN_RC" ]]; then W4_RUN_RC="missing"; fi

if [[ "$W4_RUN_RC" == "0" ]] \
        && ! cat "$W4_RUN_ERR" "$W4_RUN_OUT" | grep -qE "warning\[E230\]|error\[E230\]|handles full"; then
    pass "W4 negative control: small program runs to rc=0 with no E230 / handles-full"
else
    fail "W4 negative control: rc=$W4_RUN_RC, expected rc=0 with no E230 / handles-full"
fi

###############################################################################
# Final summary
###############################################################################
echo
echo "=== handle_table_ceiling_gate summary ==="
echo "PASS=$PASS  FAIL=$FAIL"
if [[ $FAIL -eq 0 ]]; then
    echo "HANDLE_TABLE_CEILING_GATE_OK"
    exit 0
else
    echo "HANDLE_TABLE_CEILING_GATE_FAIL"
    exit 1
fi
