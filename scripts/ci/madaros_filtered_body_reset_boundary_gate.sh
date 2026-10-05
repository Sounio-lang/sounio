#!/usr/bin/env bash
# Copilot review (PR #2516): lower_program_bodies_filtered_ref (reached via
# the public lower_program_function_body_from_summary_with_epistemic_ref)
# collects into the same process-global tuple-array-mask table as
# lower_program_to_ir, whose own reset-boundary leak
# madaros_tuple_arr_reset_boundary_gate.sh already covers -- but that gate
# only drives lower_program_to_ir. This streaming single-function lane is a
# DIFFERENT entry point (main.sio's run_probe_filtered_body_lower and
# streaming_body_lower_smoke.sio both reach it directly), and nothing reset
# the table before it collected, so two back-to-back lowerings through THIS
# lane in one process for two different inputs that both define a
# same-named function could leak the first input's mask into the second's.
#
# This gate drives that exact shape through the
# --probe-filtered-body-reset-boundary CLI flag added alongside the fix:
# file A defines `pair` returning an all-f64-array tuple (mask=3), file B
# defines `pair` (same name) returning an all-integer-array tuple (mask=0).
# Lowering A then B through lower_program_function_body_from_summary_with_
# epistemic_ref in one process must report mask=3 after A and mask=0 after
# B -- if the table isn't reset between calls, the collector's dedup guard
# (only inserts when the existing mask is 0) leaves A's mask=3 in place and
# B is misreported as mask=3 too.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KEEP_WORK="${SOUNIO_MADAROS_FILTERED_BODY_RESET_BOUNDARY_GATE_KEEP:-0}"

fail() {
  echo "[madaros-filtered-body-reset-boundary] FAIL: $*" >&2
  exit 1
}

if [[ -n "${SOUNIO_MADAROS_FILTERED_BODY_RESET_BOUNDARY_GATE_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_FILTERED_BODY_RESET_BOUNDARY_GATE_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir "$WORK" || fail "could not create gate directory: $WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-filtered-body-reset-boundary.XXXXXX)"
fi

MADAROS_ELF="${SOUNIO_MADAROS_FILTERED_BODY_RESET_BOUNDARY_GATE_BIN:-$WORK/madaros}"

if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

if [[ -z "${SOUNIO_MADAROS_FILTERED_BODY_RESET_BOUNDARY_GATE_BIN:-}" ]]; then
  if ! bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$MADAROS_ELF" >"$WORK/build.log" 2>&1; then
    tail -n 80 "$WORK/build.log" >&2 || true
    fail "current-source Madaros build failed"
  fi
fi
[[ -x "$MADAROS_ELF" ]] || fail "Madaros is missing or not executable: $MADAROS_ELF"

FILE_A="$WORK/filtered_body_reset_boundary_a.sio"
FILE_B="$WORK/filtered_body_reset_boundary_b.sio"

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
        println("FILTERED_BODY_RESET_BOUNDARY_A PASS")
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
        println("FILTERED_BODY_RESET_BOUNDARY_B PASS")
        return 0
    }
    1
}
EOF

set +e
"$MADAROS_ELF" --probe-filtered-body-reset-boundary "$FILE_A" "$FILE_B" pair >"$WORK/probe.log" 2>&1
probe_rc=$?
set -e

if [[ "$probe_rc" != "0" ]]; then
  cat "$WORK/probe.log" >&2
  fail "--probe-filtered-body-reset-boundary exited rc=$probe_rc"
fi

A_LINE="$(grep -E '^probe_filtered_body_reset_boundary: after_a fn=pair mask=[0-9]+$' "$WORK/probe.log" || true)"
B_LINE="$(grep -E '^probe_filtered_body_reset_boundary: after_b fn=pair mask=[0-9]+$' "$WORK/probe.log" || true)"
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
  fail "file B's pair has mask=$B_MASK, expected 0 (both slots are [i64;2]) -- lower_program_function_body_from_summary_with_epistemic_ref's filtered-body lane did not reset the tuple-array table between file A and file B, so file A's mask leaked forward"
fi

echo "[madaros-filtered-body-reset-boundary] PASS: the filtered-body streaming lane correctly reset the tuple-array table between file A (mask=$A_MASK) and file B (mask=$B_MASK) for the same function name"
