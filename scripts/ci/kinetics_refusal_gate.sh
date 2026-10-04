#!/usr/bin/env bash
# Runtime contract guards in stdlib/chemistry/kinetics.sio that a run-pass
# fixture cannot observe.
#
# The v2 harness has no run-fail mode: a panic is an exit 1 with no message,
# so "this call is refused" cannot be written as a run-pass assertion. Each
# probe below is compiled and run here and must exit with EXACTLY the panic
# status (1) WITHOUT printing its escape sentinel; the control must exit 0
# AND print the sentinel. Any other status (a segfault 139, a trap 136, a
# stray exit code) is a failure, not a refusal. The gate checks its own
# classifier first: a probe returning 7 and a stub dying of SIGSEGV must both
# be classified as NOT refused. Same structure as ep_gum_covariance_gate.sh
# and pbpk28_refusal_gate.sh.
#
# Guards (PR #2694 review, "add automated negative tests"):
#   cp_zero        checkpoint_steps[0] = 0 can never be reached by the
#                  stepper; the slot used to stay at its zero default.
#   cp_decreasing  checkpoint_steps must be non-decreasing; an out-of-order
#                  array used to stop early and leave zeros.
#   species_neg    species_idx < 0.
#   species_oob    species_idx >= nsp reads never-updated zero padding.
#   nsp_nine       nsp = 9 with a matching 9x2 matrix: nsp must be in [1, 8]
#                  because the state lives in [f64; 8] buffers.
#   nrxn_nine      nrxn = 9 with a matching 3x9 matrix: nrxn must be in [0, 8].
#                  Without either range guard the probe dies of SIGSEGV (139),
#                  which the exact-status rule rejects. The lower bound nsp >= 1
#                  is not probed separately: a 0-row matrix cannot carry the
#                  entries the probe sets, and with a 3x2 matrix nsp = 0 is
#                  refused by the shape guard instead.
#   dims_mismatch  nu is 3x2 but nrxn = 1: matnm_mul and the rate loop would
#                  disagree about the reaction count.
#   control        the same valid call with every guard satisfied (3x2 A->B->C,
#                  checkpoints 5,10,10,20, species 0) runs to completion.
#   direct_*       every probe above calls simulate_general_crn_checkpoints,
#                  never simulate_general_crn itself (Copilot review,
#                  sounio-lang/sounio#2694, "add coverage for the direct
#                  entry point"): both functions call check_general_crn_dims
#                  independently, so deleting that call from
#                  simulate_general_crn's own body left this gate green
#                  while the public, species_idx-free entry point silently
#                  accepted out-of-range dims. direct_control and
#                  direct_nsp_zero/direct_dims_mismatch below call
#                  simulate_general_crn directly to close that gap.
#
# Engine: lean_single only. kinetics.sio checks clean under Madaros but hits
# the documented multimodule native-link limitation at `run` (see
# tests/stdlib/chemistry/test_kinetics_fixed_regressions.sio), so a probe
# would fail at link time there and prove nothing about the guard.
# lean_single resolves stdlib/ relative to the working directory, so the
# gate runs from the repository root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
export SOUNIO_SOUC_ENGINE=lean_single
SOUC="${SOUC:-$ROOT/bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT

echo "== kinetics_refusal_gate (engine: lean_single) =="

# write_probe NAME NSP NRXN CHECKPOINTS SPECIES [ROWS COLS]: a ROWS x COLS
# stoichiometry matrix (default 3x2, A->B->C), then one
# simulate_general_crn_checkpoints call with the given nsp/nrxn/checkpoints/
# species, then the sentinel. The range probes (nsp_nine, nrxn_nine) pass a
# matrix whose shape MATCHES their nsp/nrxn, so that the shape-match guard
# cannot fire in their place: with a 3x2 matrix both escaped the revert
# control of their own range guard (the shape guard refused them instead).
write_probe() {
  local name="$1" nsp="$2" nrxn="$3" cps="$4" sp="$5" rows="${6:-3}" cols="${7:-2}"
  cat > "$OUT/kin_refusal_${name}.sio" <<PROBE
//@ run-pass
use linalg::matnm::{matnm_new, matnm_set, MatNM}
use chemistry::kinetics::*
fn main() -> i32 with Mut, Div, Panic, IO {
    var nu = matnm_new(${rows} as i64, ${cols} as i64)
    nu = matnm_set(nu, 0 as i64, 0 as i64, 0.0 - 1.0)
    nu = matnm_set(nu, 1 as i64, 0 as i64, 1.0)
    let initv: [f64; 8] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    let ks: [f64; 8] = [0.12, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    let cps: [i64; 4] = ${cps}
    let sim = simulate_general_crn_checkpoints(&initv, &ks, nu, ${nsp}, ${nrxn}, 0.2, &cps, ${sp})
    print("sim0 ") println(sim[0])
    println("KIN_REFUSAL_ESCAPED")
    return 0
}
PROBE
}

# write_direct_probe NAME NSP NRXN STEPS [ROWS COLS]: like write_probe, but
# calls simulate_general_crn directly instead of simulate_general_crn_checkpoints
# -- no checkpoints array, no species_idx -- so these probes isolate
# check_general_crn_dims at simulate_general_crn's own call site.
write_direct_probe() {
  local name="$1" nsp="$2" nrxn="$3" steps="$4" rows="${5:-3}" cols="${6:-2}"
  cat > "$OUT/kin_refusal_${name}.sio" <<PROBE
//@ run-pass
use linalg::matnm::{matnm_new, matnm_set, MatNM}
use chemistry::kinetics::*
fn main() -> i32 with Mut, Div, Panic, IO {
    var nu = matnm_new(${rows} as i64, ${cols} as i64)
    nu = matnm_set(nu, 0 as i64, 0 as i64, 0.0 - 1.0)
    nu = matnm_set(nu, 1 as i64, 0 as i64, 1.0)
    let initv: [f64; 8] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    let ks: [f64; 8] = [0.12, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    let sim = simulate_general_crn(&initv, &ks, nu, ${nsp}, ${nrxn}, 0.2, ${steps})
    print("sim0 ") println(sim[0])
    println("KIN_REFUSAL_ESCAPED")
    return 0
}
PROBE
}

run_probe() {
  local name="$1" src="$OUT/kin_refusal_$1.sio" elf="$OUT/kin_refusal_$1.elf"
  if ! "$SOUC" compile "$src" -o "$elf" >"$OUT/${name}_compile.log" 2>&1; then
    echo "FAIL: probe '$name' did not compile"
    tail -40 "$OUT/${name}_compile.log" || true
    exit 1
  fi
  chmod +x "$elf"
  set +e
  "$elf" >"$OUT/${name}_run.log" 2>&1
  PROBE_RC=$?
  set -e
}

PANIC_RC=1

is_refusal() {
  [ "$PROBE_RC" -eq "$PANIC_RC" ] && ! grep -q 'KIN_REFUSAL_ESCAPED' "$OUT/$1_run.log"
}

refused() {
  local name="$1"
  run_probe "$name"
  if ! is_refusal "$name"; then
    echo "FAIL: probe '$name' was NOT refused by a panic (rc $PROBE_RC, want $PANIC_RC and no sentinel)"
    cat "$OUT/${name}_run.log" || true
    exit 1
  fi
  echo "  refused: $name (rc $PROBE_RC)"
}

# Self-check of the classifier.
cat > "$OUT/kin_refusal_sab_exit7.sio" <<'PROBE'
//@ run-pass
fn main() -> i32 with IO {
    return 7
}
PROBE
run_probe sab_exit7
if is_refusal sab_exit7; then echo "FAIL: gate classified exit $PROBE_RC as a panic refusal"; exit 1; fi
echo "  self-check ok: exit $PROBE_RC is not a refusal"
printf '#!/bin/sh\nkill -SEGV $$\n' > "$OUT/kin_refusal_sab_segv.elf"
chmod +x "$OUT/kin_refusal_sab_segv.elf"
set +e
{ "$OUT/kin_refusal_sab_segv.elf"; PROBE_RC=$?; } >"$OUT/sab_segv_run.log" 2>&1
set -e
if is_refusal sab_segv; then echo "FAIL: gate classified a SIGSEGV (rc $PROBE_RC) as a panic refusal"; exit 1; fi
echo "  self-check ok: SIGSEGV (rc $PROBE_RC) is not a refusal"

# Control: every guard satisfied.
write_probe control 3 2 "[5, 10, 10, 20]" 0
run_probe control
if [ "$PROBE_RC" -ne 0 ] || ! grep -q 'KIN_REFUSAL_ESCAPED' "$OUT/control_run.log"; then
  echo "FAIL: control probe did not run to completion (rc $PROBE_RC)"
  cat "$OUT/control_run.log" || true
  exit 1
fi
echo "  control ok: valid call accepted"

write_probe cp_zero       3 2 "[0, 10, 10, 20]" 0;  refused cp_zero
write_probe cp_decreasing 3 2 "[5, 10, 8, 20]"  0;  refused cp_decreasing
write_probe species_neg   3 2 "[5, 10, 10, 20]" -1; refused species_neg
write_probe species_oob   3 2 "[5, 10, 10, 20]" 3;  refused species_oob
write_probe nsp_nine      9 2 "[5, 10, 10, 20]" 0 9 2; refused nsp_nine
write_probe nrxn_nine     3 9 "[5, 10, 10, 20]" 0 3 9; refused nrxn_nine
write_probe dims_mismatch 3 1 "[5, 10, 10, 20]" 0;  refused dims_mismatch

write_direct_probe direct_control 3 2 10
run_probe direct_control
if [ "$PROBE_RC" -ne 0 ] || ! grep -q 'KIN_REFUSAL_ESCAPED' "$OUT/direct_control_run.log"; then
  echo "FAIL: direct control probe did not run to completion (rc $PROBE_RC)"
  cat "$OUT/direct_control_run.log" || true
  exit 1
fi
echo "  control ok: simulate_general_crn direct call accepted"

write_direct_probe direct_nsp_zero     0 2 10; refused direct_nsp_zero
write_direct_probe direct_dims_mismatch 3 1 10; refused direct_dims_mismatch

echo "KINETICS_REFUSAL_GATE_OK"
