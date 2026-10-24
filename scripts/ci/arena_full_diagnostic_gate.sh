#!/usr/bin/env bash
# arena_full_diagnostic_gate.sh — the runtime-arena wall is fail-closed AND explained.
#
# Added 2026-10-06 (P0.4, hydrogen flagship). A Madaros-compiled program
# bump-allocates every aggregate value from one fixed ~2 GiB arena that is
# never reclaimed. When it runs out the program exits 181. Before P0.4 the
# only output was the bare line "madaros: arena full", which named neither the
# limit nor the construct; demos/hydrogen/uhs_brine_calcite.sio and
# site_screening.sio stopped there with no hint that a by-value 32 KiB MatNM
# per RK4 step was the cause.
#
# This gate builds the minimal form of that construct (a 32 KiB struct rebuilt
# by value 100000 times, ~3.3 GB of arena) and asserts:
#   rc == 181,
#   stderr keeps the historical prefix "madaros: arena full",
#   stderr names the allocation size ("(allocation of 32800 bytes)"),
#   stderr names the limit ("2 GiB") and the fix ("&! references").
# It does not assert that the wall exists forever: if a collector lands and the
# witness runs to completion with the right answer, that is reported as PASS
# with a note, because then there is no wall left to explain.
#
# Compile is `souc compile <src> -o <out>` (never the bare form).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
. "$ROOT/scripts/lib/gate_assert.sh"
gate_name "arena_full_diagnostic_gate"

unset SOUC_BIN SOUNIO_SOUC_BIN SOUNIO_SOUC_ENGINE || true
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
SOUC="${SOUC:-$ROOT/bin/souc}"
[[ -x "$SOUC" ]] || { echo "FAIL souc not executable: $SOUC" >&2; exit 2; }

echo "=== arena_full_diagnostic_gate ==="
echo "souc=$SOUC"
[[ -n "${MADAROS_RAW_BIN:-}" ]] && echo "MADAROS_RAW_BIN=$MADAROS_RAW_BIN"

TMP=$(mktemp -d "${TMPDIR:-/tmp}/arena-full-gate.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/witness.sio" <<'EOF'
struct Big { data: [f64; 4096], n: i64 }

fn set0(b: Big, v: f64) -> Big with Mut, Panic {
    var r = b
    r.data[0] = v
    return r
}

fn main() -> i64 with IO, Mut, Panic, Div {
    var b = Big { data: [0.0; 4096], n: 0 }
    var i: i64 = 0
    while i < 100000 {
        b = set0(b, i as f64)
        i = i + 1
    }
    println(b.data[0])
    0
}
EOF

set +e
timeout 300 "$SOUC" compile "$TMP/witness.sio" -o "$TMP/witness.elf" > "$TMP/compile.log" 2>&1
CRC=$?
set -e
if [[ "$CRC" != "0" || ! -x "$TMP/witness.elf" ]]; then
    sed -n '1,40p' "$TMP/compile.log" | sed 's/^/    /' >&2
    echo "FAIL compile rc=$CRC" >&2
    exit 2
fi
ENGINE="$(classify_compile_log "$TMP/compile.log")"
echo "compile_engine=$ENGINE"
[[ "$ENGINE" == "madaros" ]] || { echo "FAIL compile log named engine=$ENGINE; this gate measures the Madaros runtime" >&2; exit 2; }

set +e
timeout 600 "$TMP/witness.elf" > "$TMP/run.out" 2> "$TMP/run.err"
RRC=$?
set -e
echo "run_rc=$RRC"
sed 's/^/    /' "$TMP/run.err" | head -12

if [[ "$RRC" == "0" ]] && grep -qF '99999.000000' "$TMP/run.out"; then
    echo "PASS witness ran to completion (no arena wall on this build)"
    echo "ARENA_FULL_DIAGNOSTIC_GATE_OK"
    exit 0
fi
fail=0
[[ "$RRC" == "181" ]] || { echo "FAIL want rc=181, got $RRC" >&2; fail=1; }
for needle in 'madaros: arena full' '(allocation of 32800 bytes)' '2 GiB' '&! references'; do
    grep -qF -- "$needle" "$TMP/run.err" || { echo "FAIL stderr lacks: $needle" >&2; fail=1; }
done
if [[ "$fail" == "0" ]]; then
    echo "PASS rc=181 with a diagnostic that names the size, the limit and the fix"
    echo "ARENA_FULL_DIAGNOSTIC_GATE_OK"
    exit 0
fi
echo "ARENA_FULL_DIAGNOSTIC_GATE_FAIL" >&2
exit 1
