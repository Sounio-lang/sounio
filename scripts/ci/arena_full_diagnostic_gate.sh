#!/usr/bin/env bash
# arena_full_diagnostic_gate.sh — the runtime-arena wall is fail-closed AND
# explained, and it is reached by live data only.
#
# Added 2026-10-06 (P0.4, hydrogen flagship). A Madaros-compiled program
# bump-allocates its aggregates from one arena; when it runs out the program
# exits 181. Before P0.4 the only output was the bare line "madaros: arena
# full", which named neither the limit nor the construct (demos/hydrogen/
# uhs_brine_calcite.sio and site_screening.sio stopped there: a by-value
# 32 KiB MatNM per RK4 step).
#
# Since then the arena is reserved at start-up to RAM + swap (never below
# 2 GiB), and region reclamation (ADR-010, with loop regions) returns what a
# function or a loop iteration made. The 2026-10 witness -- a 32 KiB struct
# rebuilt by value 100000 times -- is garbage, and no longer reaches the wall.
# A wall witness built on garbage would measure a missing reclamation, so:
#
#   W  wall: a linked chain of 32 KiB nodes, all reachable, under an
#      address-space limit (ulimit -v; the runtime sizes its reservation to
#      fit it). PASS: rc 181 and the diagnostic names the prefix, the
#      allocation size, the reservation rule and both causes.
#   R  reclaim: 200000 iterations of a carried 32 KiB struct (6.4 GB if
#      nothing were returned) under the same limit. PASS: rc 0 and the exact
#      result. Without loop regions this exits 181 within a fraction of a
#      second (measured 2026-11-14).
#
# The limit is what keeps both runs far from the OOM killer; if it cannot be
# set the gate gives no verdict.
#
# Compile is `souc compile <src> -o <out>` (never the bare form).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
. "$ROOT/scripts/lib/gate_assert.sh"
gate_name "arena_full_diagnostic_gate"

unset SOUC_BIN SOUNIO_SOUC_BIN SOUNIO_SOUC_ENGINE SOUNIO_NO_REGION_RECLAIM || true
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
SOUC="${SOUC:-$ROOT/bin/souc}"
[[ -x "$SOUC" ]] || { echo "FAIL souc not executable: $SOUC" >&2; exit 2; }
AS_LIMIT_KB=4194304

echo "=== arena_full_diagnostic_gate ==="
echo "souc=$SOUC"
[[ -n "${MADAROS_RAW_BIN:-}" ]] && echo "MADAROS_RAW_BIN=$MADAROS_RAW_BIN"
echo "address_space_limit_kb=$AS_LIMIT_KB"

TMP=$(mktemp -d "${TMPDIR:-/tmp}/arena-full-gate.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

( ulimit -v "$AS_LIMIT_KB" ) 2>/dev/null || { echo "FAIL cannot set ulimit -v $AS_LIMIT_KB; no verdict" >&2; exit 2; }

compile_witness() {
    local label="$1"
    set +e
    timeout 300 "$SOUC" compile "$TMP/$label.sio" -o "$TMP/$label.elf" > "$TMP/$label.compile.log" 2>&1
    local crc=$?
    set -e
    if [[ "$crc" != "0" || ! -x "$TMP/$label.elf" ]]; then
        sed -n '1,40p' "$TMP/$label.compile.log" | sed 's/^/    /' >&2
        echo "FAIL $label: compile rc=$crc" >&2
        exit 2
    fi
    local engine
    engine="$(classify_compile_log "$TMP/$label.compile.log")"
    echo "$label compile_engine=$engine"
    [[ "$engine" == "madaros" ]] || { echo "FAIL $label: compile log named engine=$engine; this gate measures the Madaros runtime" >&2; exit 2; }
}

run_limited() {
    local label="$1"
    set +e
    ( ulimit -v "$AS_LIMIT_KB"; exec timeout 600 "$TMP/$label.elf" ) > "$TMP/$label.out" 2> "$TMP/$label.err"
    local rc=$?
    set -e
    echo "$rc" > "$TMP/$label.rc"
    echo "$label run_rc=$rc"
    sed 's/^/    /' "$TMP/$label.err" | head -12
}

cat > "$TMP/wall.sio" <<'EOF'
struct Big { data: [f64; 4096], n: i64 }
struct Node { b: Big, next: Option<Box<Node>> }

fn main() -> i64 with IO, Mut, Panic, Div, Alloc {
    var head: Option<Box<Node>> = None
    var i: i64 = 0
    while i < 400000 {
        head = Some(Box::new(Node { b: Big { data: [0.0; 4096], n: i }, next: head }))
        i = i + 1
    }
    println("done")
    0
}
EOF

cat > "$TMP/reclaim.sio" <<'EOF'
struct Big { data: [f64; 4096], n: i64 }

fn bump(b: &Big, v: f64) -> Big with Mut, Panic {
    var r = Big { data: [0.0; 4096], n: (*b).n + 1 }
    r.data[0] = (*b).data[0] + v
    r
}

fn main() -> i64 with IO, Mut, Panic, Div {
    var b = Big { data: [0.0; 4096], n: 0 }
    var i: i64 = 0
    while i < 200000 {
        b = bump(&b, 1.0)
        i = i + 1
    }
    println(b.data[0])
    println(b.n)
    0
}
EOF

compile_witness wall
compile_witness reclaim
run_limited wall
run_limited reclaim

fail=0
W_RC="$(cat "$TMP/wall.rc")"
[[ "$W_RC" == "181" ]] || { echo "FAIL wall: want rc=181, got $W_RC" >&2; fail=1; }
for needle in 'madaros: arena full' '(allocation of ' 'RAM + swap' 'reachable at once' 'a value made in it escaped'; do
    grep -qF -- "$needle" "$TMP/wall.err" || { echo "FAIL wall: stderr lacks: $needle" >&2; fail=1; }
done
[[ "$fail" == "0" ]] && echo "PASS wall: rc=181 with a diagnostic that names the size, the reservation and both causes"

R_RC="$(cat "$TMP/reclaim.rc")"
if [[ "$R_RC" == "0" ]] && grep -qxF '200000.000000' "$TMP/reclaim.out" && grep -qxF '200000' "$TMP/reclaim.out"; then
    echo "PASS reclaim: 200000 carried 32 KiB values ran to completion under the limit"
else
    echo "FAIL reclaim: want rc=0 and '200000.000000' / '200000'; got rc=$R_RC" >&2
    fail=1
fi

if [[ "$fail" == "0" ]]; then
    gate_measured_yes
    echo "ARENA_FULL_DIAGNOSTIC_GATE_OK"
    exit 0
fi
echo "ARENA_FULL_DIAGNOSTIC_GATE_FAIL" >&2
exit 1
