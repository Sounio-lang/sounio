#!/usr/bin/env bash
# Copilot review (PR #2516), comment 4115267768: lower_program_to_ir_probe_
# callsite (self-hosted/ir/lower.sio) is a SEPARATE public entry point from
# lower_program_to_ir -- it reaches the same process-global tuple-array-mask
# collector (LOWER_FN_TUPLE_ARR_*) through lower_program_bodies_filtered_ref,
# but did not reset the table itself before this round's fix. Collection
# dedupes by name, so two back-to-back calls to THIS API in one process for
# two DIFFERENT inputs that both define a function of the same name could let
# the second call's lowering silently reuse the first call's mask -- the
# identical shape madaros_tuple_arr_reset_boundary_gate.sh already pins for
# lower_program_to_ir.
#
# This gate drives that exact shape through the --probe-tuple-arr-callsite-
# reset-boundary CLI flag added in main.sio for this purpose: file A defines
# `pair` returning an all-f64-array tuple (mask=3), file B defines `pair`
# (same name) returning an all-integer-array tuple (mask=0). Lowering A then
# B in one process through lower_program_to_ir_probe_callsite must report
# mask=3 after A and mask=0 after B.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KEEP_WORK="${SOUNIO_MADAROS_CALLSITE_RESET_BOUNDARY_GATE_KEEP:-0}"

fail() {
  echo "[madaros-tuple-arr-callsite-reset-boundary] FAIL: $*" >&2
  exit 1
}

if [[ -n "${SOUNIO_MADAROS_CALLSITE_RESET_BOUNDARY_GATE_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_CALLSITE_RESET_BOUNDARY_GATE_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir "$WORK" || fail "could not create gate directory: $WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-callsite-reset-boundary.XXXXXX)"
fi

MADAROS_ELF="${SOUNIO_MADAROS_CALLSITE_RESET_BOUNDARY_GATE_BIN:-$WORK/madaros}"

if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

if [[ -z "${SOUNIO_MADAROS_CALLSITE_RESET_BOUNDARY_GATE_BIN:-}" ]]; then
  if ! bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$MADAROS_ELF" >"$WORK/build.log" 2>&1; then
    tail -n 80 "$WORK/build.log" >&2 || true
    fail "current-source Madaros build failed"
  fi
fi
[[ -x "$MADAROS_ELF" ]] || fail "Madaros is missing or not executable: $MADAROS_ELF"

FILE_A="$WORK/callsite_reset_boundary_a.sio"
FILE_B="$WORK/callsite_reset_boundary_b.sio"

cat > "$FILE_A" <<'EOF'
fn pair() -> ([f64; 2], [f64; 2]) with Mut, Panic {
    var a: [f64; 2] = [0.0; 2]
    var b: [f64; 2] = [0.0; 2]
    a[0] = 1.5
    b[0] = 0.25
    (a, b)
}

fn main() -> i32 with IO, Mut, Panic {
    let (oa, ob) = pair()
    let s0: f64 = oa[0] * 2.0
    let s1: f64 = 2.0 * ob[0]
    if s0 == 3.0 && s1 == 0.5 {
        println("CALLSITE_RESET_BOUNDARY_A PASS")
        return 0
    }
    1
}
EOF

cat > "$FILE_B" <<'EOF'
fn pair() -> ([i64; 2], [i64; 2]) with Mut, Panic {
    var a: [i64; 2] = [0; 2]
    var b: [i64; 2] = [0; 2]
    a[0] = 3
    b[0] = 9
    (a, b)
}

fn main() -> i32 with IO, Mut, Panic {
    let (oa, ob) = pair()
    if oa[0] == 3 && ob[0] == 9 {
        println("CALLSITE_RESET_BOUNDARY_B PASS")
        return 0
    }
    1
}
EOF

set +e
"$MADAROS_ELF" --probe-tuple-arr-callsite-reset-boundary "$FILE_A" "$FILE_B" pair >"$WORK/probe.log" 2>&1
probe_rc=$?
set -e

if [[ "$probe_rc" != "0" ]]; then
  cat "$WORK/probe.log" >&2
  fail "--probe-tuple-arr-callsite-reset-boundary exited rc=$probe_rc"
fi

A_LINE="$(grep -E '^probe_tuple_arr_callsite_reset_boundary: after_a fn=pair mask=[0-9]+$' "$WORK/probe.log" || true)"
B_LINE="$(grep -E '^probe_tuple_arr_callsite_reset_boundary: after_b fn=pair mask=[0-9]+$' "$WORK/probe.log" || true)"
if [[ -z "$A_LINE" || -z "$B_LINE" ]]; then
  cat "$WORK/probe.log" >&2
  fail "missing after_a/after_b mask lines for fn=pair"
fi
A_MASK="${A_LINE##*mask=}"
B_MASK="${B_LINE##*mask=}"

if [[ "$A_MASK" != "3" ]]; then
  cat "$WORK/probe.log" >&2
  fail "file A's pair has mask=$A_MASK, expected 3 (both slots are [f64;2])"
fi
if [[ "$B_MASK" != "0" ]]; then
  cat "$WORK/probe.log" >&2
  fail "file B's pair has mask=$B_MASK, expected 0 (both slots are [i64;2]) -- lower_program_to_ir_probe_callsite did not reset the tuple-array table before lowering file B, so file A's mask leaked forward"
fi

echo "[madaros-tuple-arr-callsite-reset-boundary] PASS: lower_program_to_ir_probe_callsite correctly reset the tuple-array table between file A (mask=$A_MASK) and file B (mask=$B_MASK) for the same function name"
