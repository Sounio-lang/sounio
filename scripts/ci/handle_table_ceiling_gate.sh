#!/usr/bin/env bash
# handle_table_ceiling_gate.sh — the handle wall is fail-closed and named.
#
# Closed by refutation 2026-08-18 (docs/audit/HANDLE_TABLE_E230_REFUTATION_2026-08-18.md).
# Current Madaros prints "madaros: handles full" and exits 182. It does not
# print E230, count=, or 4194304. This gate asserts only the string and rc
# that exist. Do not add greps for a diagnostic that is not on the path.
#
# Compile is `souc compile <src> -o <out>` (never the bare form).
# A 3-slot aggregate must RUN before any ceiling number is emitted.
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
. "$ROOT/scripts/lib/gate_assert.sh"
gate_name "handle_table_ceiling_gate"

unset SOUC_BIN SOUNIO_SOUC_BIN SOUNIO_SOUC_ENGINE || true
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
SOUC="${SOUC:-$ROOT/bin/souc}"
require_tool timeout "timeout(1) missing — cannot bound compile/run"
require_tool od "od(1) missing — cannot check ELF magic"
[[ -x "$SOUC" ]] || { echo "FAIL souc not executable: $SOUC" >&2; exit 2; }
if [[ "${SOUNIO_SOUC_ENGINE:-}" == "lean_single" ]]; then
    echo "FAIL this gate asserts the Madaros handle wall; refuse lean_single" >&2
    exit 2
fi

CAPACITY=4194304
CEILING_ITERS=$((CAPACITY + 16))

echo "=== handle_table_ceiling_gate ==="
echo "souc=$SOUC"
echo "souc_version=$("$SOUC" --version 2>/dev/null | head -1 || echo unknown)"
[[ -n "${MADAROS_RAW_BIN:-}" ]] && echo "MADAROS_RAW_BIN=$MADAROS_RAW_BIN"
echo "capacity=$CAPACITY ceiling_iters=$CEILING_ITERS"

TMP=$(mktemp -d "${TMPDIR:-/tmp}/handle-ceiling-gate.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

selftest_fail() {
    echo "FAIL gate self-test [$1]: $2" >&2
    echo "HANDLE_TABLE_CEILING_GATE_SELFTEST_FAIL [$1]" >&2
    exit 2
}

compile_and_run() {
    local label="$1" sio="$2" elf="$3"
    local clog="$TMP/${label}.compile.out"
    rm -f -- "$PWD/-o" "$elf"
    gate_capture_rc "$TMP/${label}.compile.rc" -- \
        timeout 300 "$SOUC" compile "$sio" -o "$elf" > "$clog" 2>&1
    local crc engine
    crc="$(cat "$TMP/${label}.compile.rc")"
    engine="$(classify_compile_log "$clog")"
    echo "compile_rc=$crc compile_engine=$engine — $label"
    [[ "$crc" == "0" ]] || {
        sed -n '1,40p' "$clog" | sed 's/^/    /' >&2
        selftest_fail "$label" "compiler refused the witness"
    }
    [[ -e "$PWD/-o" ]] && selftest_fail "$label" "compile wrote a literal -o file"
    require_elf "$elf" "$label compile artefact"
    echo "elf_ok=$elf bytes=$(stat -c%s "$elf") magic=7f454c46"
    [[ "$engine" == "madaros" ]] || selftest_fail "$label" "compile log named engine=$engine; need Madaros"
    gate_capture_rc "$TMP/${label}.run.rc" -- \
        timeout 600 "$elf" > "$TMP/${label}.run.out" 2> "$TMP/${label}.run.err"
    require_rc_file "$TMP/${label}.run.rc"
    echo "run_rc=$(cat "$TMP/${label}.run.rc")"
    echo "stderr (last 10):"
    tail -10 "$TMP/${label}.run.err" | sed 's/^/    /'
}

###############################################################################
# S1 — 3-slot aggregate must RUN. If this compiler cannot construct a
# 3-i64 struct, every later number would be a crash report. Refuse.
###############################################################################
cat > "$TMP/s1.sio" <<'EOF'
struct S1 { x: i64, y: i64, z: i64 }
fn main() -> i64 {
    let _s = S1 { x: 1, y: 2, z: 3 }
    0
}
EOF
echo
echo "--- S1: 3-slot aggregate must run ---"
compile_and_run S1 "$TMP/s1.sio" "$TMP/S1.elf"
S1_RC="$(cat "$TMP/S1.run.rc")"
if [[ "$S1_RC" != "0" ]]; then
    echo "HANDLE_TABLE_CEILING_GATE_REFUSE_MEASURE [S1] rc=$S1_RC" >&2
    selftest_fail S1 "3-slot aggregate did not run (rc=$S1_RC); refuse to measure the ceiling"
fi
echo "S1_OK rc=0"

###############################################################################
# Ceiling witness — capacity+16 handle-consuming allocs.
# PASS: rc=182 and stderr contains "madaros: handles full".
###############################################################################
cat > "$TMP/ceiling.sio" <<EOF
struct H { x: i64, y: i64, z: i64 }
fn alloc_one() -> H with Alloc { H { x: 1, y: 1, z: 1 } }
fn main() -> i64 with IO, Mut, Panic, Div, Alloc {
    var i: i64 = 0
    while i < $CEILING_ITERS {
        let _x = alloc_one()
        i = i + 1
    }
    print("done\\n")
    0
}
EOF
echo
echo "--- ceiling: $CEILING_ITERS allocs (capacity $CAPACITY) ---"
compile_and_run ceiling "$TMP/ceiling.sio" "$TMP/ceiling.elf"
C_RC="$(cat "$TMP/ceiling.run.rc")"

echo
echo "=== handle_table_ceiling_gate verdict ==="
if [[ "$C_RC" == "182" ]] && grep -qF 'madaros: handles full' "$TMP/ceiling.run.err"; then
    echo "PASS ceiling rc=182 named 'madaros: handles full'"
    gate_measured_yes
    echo "HANDLE_TABLE_CEILING_GATE_OK"
    exit 0
fi
echo "FAIL ceiling: want rc=182 and stderr 'madaros: handles full'; got rc=$C_RC" >&2
echo "measured=no"
echo "HANDLE_TABLE_CEILING_GATE_FAIL" >&2
exit 1
