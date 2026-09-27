#!/usr/bin/env bash
# PBPK28 multi-drug driver refusals that a run-pass fixture cannot observe.
#
# The v2 harness has no run-fail mode: a panic is an exit 1 with no message on
# both engines, so a run-pass file cannot assert "this call is refused". Each
# probe below is compiled and run here and must exit with EXACTLY the panic
# status WITHOUT printing its escape sentinel; a control must exit 0 AND print
# the sentinel. The panic status is 1 on both engines (measured 2026-09-27:
# `panic("boom")` in main exits 1 under Madaros and under lean_single). Any
# other non-zero status is a failure, not a refusal: a segfault (139), a
# floating-point trap (136) or a stray exit code would otherwise pass as the
# expected panic (PR #2695 review). The gate checks this about itself before
# any probe: a probe that returns 7 and a probe that dies of SIGSEGV must both
# be classified as NOT refused.
#
# Probes (PR #2695 review):
#   oral_unreg   md_oral_dose_ok on an unregistered slot below capacity. It
#                answered "ok" from zero-filled storage before it validated the
#                index with md_check_drug, like every other per-drug function.
#   bolus_unreg  md_bolus_dose_ok on the same slot. It answered "not ok"
#                (5 mg divided by a zero volume), equally from empty storage.
#   control      both predicates on the registered drug return normally.
#   load_inspect md_factor_now when the inhibitor load overflows: ext_load =
#                1e308 plus one inhibitor contribution C_u/Ki of 1e308 is +Inf,
#                and the factor 1/(1 + Inf) came back as exactly 0 -- a victim
#                sink silently switched off.
#   load_step    md_step on the same state: the production path shares the one
#                load implementation (md_site_load) and must refuse as well;
#                it used to step with the victim's sink at 0.
#   load_control the same state with ext_load = 1e307, so that the sum
#                (1.1e308) is finite: the step and the inspection run.
#   oral_neg/nan/inf, bolus_neg/nan/inf
#                md_dose_oral / md_dose_iv_bolus with mg = -1, NaN, +Inf: the
#                refusal side of "the predicate is false exactly when the
#                call panics" (the accepting side is V14 of the multidrug gates).
#
# Engine: bin/souc (Madaros) by default; set SOUNIO_SOUC_ENGINE=lean_single to
# run the same probes on the bootstrap engine.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# Always pin this worktree's stdlib (never inherit a foreign SOUNIO_STDLIB_PATH).
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
SOUC="${SOUC:-$ROOT/bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT

echo "== pbpk28_refusal_gate (engine: ${SOUNIO_SOUC_ENGINE:-default}) =="

# write_probe NAME BODY: a one-drug driver, then BODY, then the sentinel.
write_probe() {
  local name="$1" body="$2"
  cat > "$OUT/pbpk28_refusal_${name}.sio" <<PROBE
//@ run-pass
use darwin_pbpk::core::pbpk28_params::*
use darwin_pbpk::ddi::multidrug28::*
fn main() -> i32 with Mut, Div, Panic, IO {
    var md = md_new(true)
    var p = pbpk28_params_rapamycin()
    p.cl_central = 1.0
    let a = md_add_drug(&!md, p, 0.1, 1.0, 1.0)
    ${body}
    println("PBPK28_REFUSAL_ESCAPED")
    return 0
}
PROBE
}

run_probe() {
  local name="$1" src="$OUT/pbpk28_refusal_$1.sio" elf="$OUT/pbpk28_refusal_$1.elf"
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

# is_refusal NAME: true only for the panic status and no escape sentinel.
is_refusal() {
  [ "$PROBE_RC" -eq "$PANIC_RC" ] && ! grep -q 'PBPK28_REFUSAL_ESCAPED' "$OUT/$1_run.log"
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

# control NAME: must exit 0 and print the sentinel.
control() {
  local name="$1"
  run_probe "$name"
  if [ "$PROBE_RC" -ne 0 ] || ! grep -q 'PBPK28_REFUSAL_ESCAPED' "$OUT/${name}_run.log"; then
    echo "FAIL: control probe '$name' did not run to completion (rc $PROBE_RC)"
    cat "$OUT/${name}_run.log" || true
    exit 1
  fi
}

# Self-check of the classifier: an ordinary non-zero exit and a crash must not
# count as a refusal.
write_probe sab_exit7 "return 7"
run_probe sab_exit7
if is_refusal sab_exit7; then echo "FAIL: gate classified exit $PROBE_RC as a panic refusal"; exit 1; fi
echo "  self-check ok: exit $PROBE_RC is not a refusal"
printf '#!/bin/sh\nkill -SEGV $$\n' > "$OUT/pbpk28_refusal_sab_segv.elf"
chmod +x "$OUT/pbpk28_refusal_sab_segv.elf"
set +e
{ "$OUT/pbpk28_refusal_sab_segv.elf"; PROBE_RC=$?; } >"$OUT/sab_segv_run.log" 2>&1
set -e
if is_refusal sab_segv; then echo "FAIL: gate classified a SIGSEGV (rc $PROBE_RC) as a panic refusal"; exit 1; fi
echo "  self-check ok: SIGSEGV (rc $PROBE_RC) is not a refusal"

write_probe control "let o = md_oral_dose_ok(&md, a, 5.0)
    let b = md_bolus_dose_ok(&md, a, 5.0)
    if !(o && b) { return 2 }"
control control
echo "  control ok: registered drug accepted"

write_probe oral_unreg "if md_oral_dose_ok(&md, 3, 5.0) { println(\"ok\") } else { println(\"not ok\") }"
refused oral_unreg
write_probe bolus_unreg "if md_bolus_dose_ok(&md, 3, 5.0) { println(\"ok\") } else { println(\"not ok\") }"
refused bolus_unreg

# Load overflow: a second drug x (Ki at CYP3A4) is dosed and run for an hour,
# then Ki is set so that its liver contribution fu*C_t/(Kp*Ki) is 1e308 at the
# current state, and ext_load is added on top of it.
load_body() {
  cat <<BODY
md_set_clint(&!md, a, md_site_liver(), md_cyp3a4(), 100.0)
    var q = pbpk28_params_rapamycin()
    q.cl_central = 1.0
    let x = md_add_drug(&!md, q, 0.1, 1.0, 1.0)
    md_set_ki(&!md, x, md_cyp3a4(), 1.0)
    md_dose_iv_bolus(&!md, a, 5.0)
    md_dose_iv_bolus(&!md, x, 5.0)
    md_run_to(&!md, 1.0, 0.1)
    let c = md.ct[x * 14 + 1]
    md_set_ki(&!md, x, md_cyp3a4(), md.fu_site[x * 2] * c / md.kp[x * 14 + 1] / 1.0e308)
    md_set_ext_load(&!md, md_site_liver(), md_cyp3a4(), $1)
    $2
BODY
}
write_probe load_control "$(load_body 1.0e307 'let f = md_factor_now(&md, a, md_site_liver(), md_cyp3a4())
    md_step(&!md, 0.1)
    if !(f > 0.0) { return 2 }')"
control load_control
echo "  control ok: finite inhibitor load accepted"
write_probe load_inspect "$(load_body 1.0e308 'let f = md_factor_now(&md, a, md_site_liver(), md_cyp3a4())
    print("factor ") println(f)')"
refused load_inspect
write_probe load_step "$(load_body 1.0e308 'md_step(&!md, 0.1)')"
refused load_step

# Dose domain: each dose the predicates reject (-1, NaN, +Inf) must make the
# dosing call panic, so that "*_dose_ok is false" and "the call is refused"
# coincide (the accepted doses are checked in the multidrug gates, V14).
for d in "neg:0.0 - 1.0" "nan:0.0 * (1.0e308 * 10.0)" "inf:1.0e308 * 10.0"; do
  tag="${d%%:*}"; expr="${d#*:}"
  write_probe "oral_${tag}" "md_dose_oral(&!md, a, ${expr})"
  refused "oral_${tag}"
  write_probe "bolus_${tag}" "md_dose_iv_bolus(&!md, a, ${expr})"
  refused "bolus_${tag}"
done

echo "PBPK28_REFUSAL_GATE_OK"
