#!/usr/bin/env bash
# PBPK28 multi-drug driver refusals that a run-pass fixture cannot observe.
#
# The v2 harness has no run-fail mode: a panic is an exit 1 with no message on
# both engines, so a run-pass file cannot assert "this call is refused". Each
# probe below is compiled and run here and must ABORT (non-zero exit) WITHOUT
# printing its escape sentinel; a control must run to completion. Same pattern
# as scripts/ci/ep_gum_covariance_gate.sh.
#
# Probes (PR #2695 review):
#   oral_unreg   md_oral_dose_ok on an unregistered slot below capacity. It
#                answered "ok" from zero-filled storage before it validated the
#                index with md_check_drug, like every other per-drug function.
#   bolus_unreg  md_bolus_dose_ok on the same slot. It answered "not ok"
#                (5 mg divided by a zero volume), equally from empty storage.
#   control      both predicates on the registered drug return normally.
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

refused() {
  local name="$1"
  run_probe "$name"
  if [ "$PROBE_RC" -eq 0 ] || grep -q 'PBPK28_REFUSAL_ESCAPED' "$OUT/${name}_run.log"; then
    echo "FAIL: probe '$name' was NOT refused"
    cat "$OUT/${name}_run.log" || true
    exit 1
  fi
  echo "  refused: $name (rc $PROBE_RC)"
}

write_probe control "let o = md_oral_dose_ok(&md, a, 5.0)
    let b = md_bolus_dose_ok(&md, a, 5.0)
    if !(o && b) { return 2 }"
run_probe control
if [ "$PROBE_RC" -ne 0 ] || ! grep -q 'PBPK28_REFUSAL_ESCAPED' "$OUT/control_run.log"; then
  echo "FAIL: control probe did not run to completion (rc $PROBE_RC)"
  cat "$OUT/control_run.log" || true
  exit 1
fi
echo "  control ok: registered drug accepted"

write_probe oral_unreg "if md_oral_dose_ok(&md, 3, 5.0) { println(\"ok\") } else { println(\"not ok\") }"
refused oral_unreg
write_probe bolus_unreg "if md_bolus_dose_ok(&md, 3, 5.0) { println(\"ok\") } else { println(\"not ok\") }"
refused bolus_unreg

echo "PBPK28_REFUSAL_GATE_OK"
