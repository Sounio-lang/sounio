#!/usr/bin/env bash
# Prove the native-v2 handle-table lifetime contract around exit 182.
#
# Default mode is intentionally a post-fix gate: it runs a managed (>16 B)
# aggregate through more than the source-declared handle capacity and requires
# the exact result. The separate escape positive control proves that reclamation
# does not invalidate a returned/live aggregate. The baseline mode is for
# preserving the pre-fix receipt; it expects the old fail-closed exit 182 and is
# never a success path for the reclamation claim.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$BASH_SOURCE")/../.." && pwd)"
cd "$ROOT_DIR"

fail() {
    printf 'MADAROS_HANDLE_182_GATE_FAIL reason=%s\n' "$*"
    exit 1
}

WORK="$(mktemp -d /tmp/madaros-handle-182.XXXXXX)"
KEEP_WORK="$(printenv SOUNIO_HANDLE_182_GATE_KEEP 2>/dev/null || true)"
if [[ "$KEEP_WORK" != "1" ]]; then
    trap 'rm -rf "$WORK"' EXIT
fi

CAPACITY_SOURCE="$ROOT_DIR/self-hosted/native/gc.sio"
RECLAIM_TEMPLATE="$ROOT_DIR/scripts/ci/fixtures/madaros_handle_table_182/reclaim.sio"
ESCAPE_SOURCE="$ROOT_DIR/scripts/ci/fixtures/madaros_handle_table_182/escape.sio"

[[ -f "$CAPACITY_SOURCE" ]] || fail "missing_capacity_source"
[[ -f "$RECLAIM_TEMPLATE" ]] || fail "missing_reclaim_fixture"
[[ -f "$ESCAPE_SOURCE" ]] || fail "missing_escape_fixture"

# Derive the boundary from the source under test. A literal here would test a
# historical number instead of the actual allocator contract.
HANDLE_CAPACITY="$(sed -n \
    's/^pub fn native_v2_handle_table_capacity_default() -> i64 { \([0-9][0-9]*\) }$/\1/p' \
    "$CAPACITY_SOURCE" | sed -n '1p')"
[[ "$HANDLE_CAPACITY" =~ ^[0-9]+$ ]] || fail "handle_capacity_not_found"
(( HANDLE_CAPACITY > 0 )) || fail "handle_capacity_not_positive"

MADAROS_ELF="$(printenv SOUNIO_HANDLE_182_GATE_BIN 2>/dev/null || true)"
if [[ -z "$MADAROS_ELF" ]]; then
    MADAROS_ELF="$WORK/madaros"
fi
RUN_TIMEOUT="$(printenv SOUNIO_HANDLE_182_GATE_TIMEOUT 2>/dev/null || true)"
if [[ -z "$RUN_TIMEOUT" ]]; then
    RUN_TIMEOUT=300
fi
BASELINE="$(printenv SOUNIO_HANDLE_182_GATE_BASELINE 2>/dev/null || true)"
if [[ -z "$BASELINE" ]]; then
    BASELINE=0
fi
RUN_N="$(printenv SOUNIO_HANDLE_182_GATE_N 2>/dev/null || true)"
if [[ -z "$RUN_N" ]]; then
    RUN_N=$((HANDLE_CAPACITY + 4096))
fi

if [[ "$BASELINE" == "1" ]]; then
    RUN_N="$HANDLE_CAPACITY"
else
    [[ "$RUN_N" =~ ^[0-9]+$ ]] || fail "run_n_not_numeric"
    (( RUN_N > HANDLE_CAPACITY )) || fail "post_fix_run_must_exceed_capacity"
fi

if [[ -z "$(printenv SOUNIO_HANDLE_182_GATE_BIN 2>/dev/null || true)" ]]; then
    if ! bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$MADAROS_ELF" \
        >"$WORK/build.log" 2>&1; then
        tail -n 100 "$WORK/build.log" >&2 || true
        fail "current_source_madaros_build_failed"
    fi
fi
[[ -x "$MADAROS_ELF" ]] || fail "madaros_binary_missing"
if head -c 2 "$MADAROS_ELF" 2>/dev/null | grep -q '^#!'; then
    fail "madaros_binary_is_wrapper"
fi
if ! "$MADAROS_ELF" --version >"$WORK/version.log" 2>&1; then
    cat "$WORK/version.log" >&2
    fail "madaros_version_failed"
fi
grep -qi 'Madaros' "$WORK/version.log" || {
    cat "$WORK/version.log" >&2
    fail "compiler_is_not_madaros"
}

export SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib"

render_reclaim() {
    local out="$1" n="$2"
    sed "s/let n: i64 = 1/let n: i64 = $n/" "$RECLAIM_TEMPLATE" >"$out"
    grep -Fq "let n: i64 = $n" "$out" || fail "reclaim_fixture_render_failed"
}

compile_fixture() {
    local name="$1" source="$2" elf="$3" log="$4"
    if ! MADAROS_RAW_BIN="$MADAROS_ELF" "$ROOT_DIR/bin/madaros" compile \
        "$source" -o "$elf" >"$log" 2>&1; then
        cat "$log" >&2
        fail "compile_$name"
    fi
    [[ -x "$elf" ]] || fail "compile_$name""_did_not_produce_executable"
}

run_fixture() {
    local name="$1" elf="$2" log="$3"
    local rc
    if timeout "$RUN_TIMEOUT" "$elf" >"$log" 2>&1; then
        rc=0
    else
        rc=$?
    fi
    printf '%s\n' "$rc"
}

# This control is cheap and must pass before the high-count arm. It is also a
# guard against a reclamation implementation that resets a caller's watermark
# and then happens to make the high-count loop terminate.
ESCAPE_ELF="$WORK/escape.elf"
compile_fixture escape "$ESCAPE_SOURCE" "$ESCAPE_ELF" "$WORK/escape.compile.log"
escape_rc="$(run_fixture escape "$ESCAPE_ELF" "$WORK/escape.run.log")"
if [[ "$escape_rc" != "0" ]] || ! grep -Fq 'MADAROS_HANDLE_182_ESCAPE_OK value=36' "$WORK/escape.run.log"; then
    cat "$WORK/escape.run.log" >&2
    fail "escape_positive_control_rc_$escape_rc"
fi
printf 'MADAROS_HANDLE_182_ESCAPE_PASS value=36\n'

RECLAIM_SOURCE="$WORK/reclaim.sio"
RECLAIM_ELF="$WORK/reclaim.elf"
render_reclaim "$RECLAIM_SOURCE" "$RUN_N"

if [[ "$BASELINE" == "1" ]]; then
    # This branch records the old, honest failure. A zero exit here is a
    # regression in the witness: the supposedly exact wall did not exercise
    # the managed-handle allocator.
    compile_fixture reclaim "$RECLAIM_SOURCE" "$RECLAIM_ELF" "$WORK/reclaim.compile.log"
    reclaim_rc="$(run_fixture reclaim "$RECLAIM_ELF" "$WORK/reclaim.run.log")"
    if [[ "$reclaim_rc" != "182" ]] || ! grep -Fq 'handles full' "$WORK/reclaim.run.log"; then
        cat "$WORK/reclaim.run.log" >&2
        fail "baseline_wall_not_reproduced_rc_$reclaim_rc"
    fi
    printf 'MADAROS_HANDLE_182_BASELINE_OK capacity=%s rc=182\n' "$HANDLE_CAPACITY"
    exit 0
fi

compile_fixture reclaim "$RECLAIM_SOURCE" "$RECLAIM_ELF" "$WORK/reclaim.compile.log"
reclaim_rc="$(run_fixture reclaim "$RECLAIM_ELF" "$WORK/reclaim.run.log")"
expected_sum=$((RUN_N * (RUN_N - 1) / 2))
expected_marker="MADAROS_HANDLE_182_RECLAIM_OK n=$RUN_N sum=$expected_sum"
if [[ "$reclaim_rc" != "0" ]] || ! grep -Fq "$expected_marker" "$WORK/reclaim.run.log"; then
    cat "$WORK/reclaim.run.log" >&2
    fail "reclaim_run_rc_$reclaim_rc expected=$expected_marker"
fi

printf 'MADAROS_HANDLE_182_RECLAMATION_GATE_OK capacity=%s n=%s escape=36 sum=%s\n' \
    "$HANDLE_CAPACITY" "$RUN_N" "$expected_sum"
