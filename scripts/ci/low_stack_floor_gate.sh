#!/usr/bin/env bash
# low_stack_floor_gate.sh — pin the raw-ELF stack floor (issue #3005).
#
# WHAT THIS ASSERTS
#
# A raw Madaros ELF built from this tree compiles a nested-for witness with the
# process stack limited to the floor the launcher documents
# (bin/madaros: "Measured floor 2026-08-13: minimal ok at 16 MiB+"). 16 MiB is
# the contract; the launcher's 512 MiB default is headroom on top of it.
#
# WHY IT EXISTS
#
# #2981 (7f13ba664a) grew TypeEnv.bindings from 8192 to 12288 entries. Walked
# 8 MiB -> 9216 -> 9472 by controlled ablation (2026-10-23): the tree that
# carries 12288 needs ~9.3 MiB, the one that carries 8192 needs <8 MiB, and the
# exact growth lands in a 4.64 MiB by-value frame inside the checker path
# (`sub rsp,0x4a31e0` exists in the 12288 artifact and not in the 8192 one).
# Nothing in that walk left the documented floor: 8 MiB never was the floor,
# it was headroom. This gate makes the floor a test instead of a comment, so a
# future capacity change cannot quietly spend it.
#
# It deliberately does NOT run at 8 MiB: asserting an unsupported limit would
# turn a ratchet into a promise the engine never made.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

[[ "$(uname -s)/$(uname -m)" = Linux/x86_64 ]] || { echo 'Linux x86-64 required (raw ELF + RLIMIT_STACK)' >&2; exit 2; }

MADAROS="${MADAROS_RAW_BIN:-$ROOT/artifacts/self-hosted/madaros}"
if [[ ! -x "$MADAROS" ]]; then
    echo "FAIL: no raw Madaros at $MADAROS — run make build-madaros first" >&2
    exit 2
fi
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"

FLOOR_KB=16384   # 16 MiB — the documented minimal floor, not the 512 MiB default.

TMP=$(mktemp -d "${TMPDIR:-/tmp}/low-stack-floor-gate.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

# Same witness as scripts/ci/madaros_loop_gate.sh nested_for (exit 9): nested
# loops exercise the range/binary span paths that meet the big checker frames.
printf '%s\n' 'fn main() -> i64 { var s = 0
    for i in 0..3 { for j in 0..3 { s = s + 1 } }
    s }' > "$TMP/nested_for.sio"

# A helper-call witness for the "helper at 32 MiB+" half of the floor note.
printf '%s\n' 'fn helper(x: i64) -> i64 { x + 1 }
fn main() -> i64 { helper(41) }' > "$TMP/helper.sio"

run_at_floor() {
    local label="$1" src="$2" want="$3"
    local rc=0
    ( ulimit -s "$FLOOR_KB"; "$MADAROS" build "$src" -o "$TMP/$label.elf" ) \
        > "$TMP/$label.log" 2>&1 || rc=$?
    if [[ "$rc" != "0" ]]; then
        echo "FAIL: $label did not compile at the ${FLOOR_KB}K floor (rc=$rc)" >&2
        tail -5 "$TMP/$label.log" | sed 's/^/    /' >&2
        return 1
    fi
    chmod +x "$TMP/$label.elf"
    local got=0
    "$TMP/$label.elf" || got=$?
    if [[ "$got" != "$want" ]]; then
        echo "FAIL: $label ran with exit $got, want $want (compiled, wrong result)" >&2
        return 1
    fi
    echo "PASS: $label compiled at ${FLOOR_KB}K and exited $want"
}

echo "=== low_stack_floor_gate (floor=${FLOOR_KB}K) ==="
run_at_floor nested_for "$TMP/nested_for.sio" 9
run_at_floor helper     "$TMP/helper.sio"     42
echo "PASS: raw-ELF stack floor held (issue #3005)"
