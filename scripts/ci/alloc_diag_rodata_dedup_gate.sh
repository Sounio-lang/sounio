#!/usr/bin/env bash
# alloc_diag_rodata_dedup_gate.sh — one copy of each allocation-failure message per ELF.
#
# #2836: PR #2793 made the runtime "arena full" / "handles full" message name
# the allocation size, the limit and the fix (~700 bytes), and appended that
# text to .rodata at EVERY allocation site. self-hosted/compiler/main.sio then
# needed 23,676,227 bytes of .rodata against the 2 MiB NC_BIG_RODATA, and the
# Madaros self-compile rung (gen1 -> gen2) refused with rc=23 on main.
#
# The message depends only on (fail_code, request_bytes), so the emitter now
# appends each distinct message once per program and points every site at it
# (NC_ALLOC_DIAG_* in self-hosted/native/frame.sio).
#
# Witness: SITES functions that each build the same 24-byte struct by value.
# Every one is an allocation site with the same request size. The gate asserts
#   - the program compiles and runs (rc=0, right answer),
#   - the explanatory tail of the arena message occurs exactly once in the ELF,
#     where before the fix it occurred once per site.
# The text of the message itself stays pinned by arena_full_diagnostic_gate.sh.
#
# Compile is `souc compile <src> -o <out>` (never the bare form).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
. "$ROOT/scripts/lib/gate_assert.sh"
gate_name "alloc_diag_rodata_dedup_gate"

unset SOUC_BIN SOUNIO_SOUC_BIN SOUNIO_SOUC_ENGINE || true
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
SOUC="${SOUC:-$ROOT/bin/souc}"
[[ -x "$SOUC" ]] || { echo "FAIL souc not executable: $SOUC" >&2; exit 2; }
SITES="${SITES:-200}"

echo "=== alloc_diag_rodata_dedup_gate ==="
echo "souc=$SOUC sites=$SITES"
[[ -n "${MADAROS_RAW_BIN:-}" ]] && echo "MADAROS_RAW_BIN=$MADAROS_RAW_BIN"

TMP=$(mktemp -d "${TMPDIR:-/tmp}/alloc-diag-dedup.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

{
    echo 'struct Tri { a: f64, b: f64, c: f64 }'
    echo
    for ((k = 0; k < SITES; k++)); do
        echo "fn mk$k(x: f64) -> Tri {"
        echo "    Tri { a: x, b: x + 1.0, c: x + $k.0 }"
        echo "}"
        echo
    done
    echo 'fn main() -> i64 with IO, Mut, Panic, Div {'
    echo '    var acc: f64 = 0.0'
    for ((k = 0; k < SITES; k++)); do
        echo "    acc = acc + mk$k(1.0).c"
    done
    echo '    println(acc)'
    echo '    0'
    echo '}'
} > "$TMP/witness.sio"

# sum over k of (1 + k) = SITES + SITES*(SITES-1)/2
EXPECT="$(awk -v n="$SITES" 'BEGIN { printf "%.6f", n + n * (n - 1) / 2 }')"

set +e
timeout 600 "$SOUC" compile "$TMP/witness.sio" -o "$TMP/witness.elf" > "$TMP/compile.log" 2>&1
CRC=$?
set -e
if [[ "$CRC" != "0" || ! -x "$TMP/witness.elf" ]]; then
    sed -n '1,40p' "$TMP/compile.log" | sed 's/^/    /' >&2
    echo "FAIL compile rc=$CRC" >&2
    exit 2
fi
ENGINE="$(classify_compile_log "$TMP/compile.log")"
echo "compile_engine=$ENGINE"
[[ "$ENGINE" == "madaros" ]] || { echo "FAIL compile log named engine=$ENGINE; this gate measures Madaros codegen" >&2; exit 2; }

set +e
timeout 120 "$TMP/witness.elf" > "$TMP/run.out" 2> "$TMP/run.err"
RRC=$?
set -e
echo "run_rc=$RRC stdout=$(tr -d '\n' < "$TMP/run.out")"

fail=0
[[ "$RRC" == "0" ]] || { echo "FAIL want rc=0, got $RRC" >&2; sed 's/^/    /' "$TMP/run.err" | head -8 >&2; fail=1; }
grep -qF -- "$EXPECT" "$TMP/run.out" || { echo "FAIL want $EXPECT on stdout" >&2; fail=1; }

# A line of the arena-full explanation that no other text contains.
NEEDLE='takes fresh arena space for the life of the process'
COPIES="$(grep -aoF -- "$NEEDLE" "$TMP/witness.elf" | wc -l | tr -d ' ')"
echo "arena_message_copies=$COPIES"
if [[ "$COPIES" != "1" ]]; then
    echo "FAIL the arena-full explanation occurs $COPIES times in the ELF; want exactly 1 (one copy per distinct message, #2836)" >&2
    fail=1
fi

if [[ "$fail" == "0" ]]; then
    echo "PASS $SITES same-size allocation sites share one copy of the diagnostic"
    echo "ALLOC_DIAG_RODATA_DEDUP_GATE_OK"
    exit 0
fi
echo "ALLOC_DIAG_RODATA_DEDUP_GATE_FAIL" >&2
exit 1
