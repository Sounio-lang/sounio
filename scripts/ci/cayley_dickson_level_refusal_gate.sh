#!/usr/bin/env bash
# Cayley-Dickson level refusals that a run-pass fixture cannot observe.
#
# CDElement holds 2048 = 2^11 coefficients and its private `bits` level is set
# only by cd_zero, which panics on a level outside [0, 11]. Before that check,
# out-of-range levels ran to completion with exit 0:
#   eq_12        cd_eq(cd_zero(12), cd_zero(12)) walked 4096 slots of c.
#   split_12     cd_split(cd_zero(12)) indexed c[2048].
#   embed_neg    cd_embed(cd_zero(-1), 0) shifted by a negative count.
# join_low/join_high exercise cd_join's half-level guard directly: every
# element is built through cd_zero, so the three probes above all panic in
# the constructor and never reach the tower checks. Here both halves are
# valid elements and only one half's level disagrees with the target, so a
# regression that accepts a mismatched low or high half would escape.
# The v2 harness has no run-fail mode (a panic is exit 1 with no message), so,
# like scripts/ci/pbpk28_refusal_gate.sh, each probe is compiled and run here
# and must exit with EXACTLY the panic status (1) WITHOUT printing its escape
# sentinel. Any other status -- a crash or a stray exit code -- is a failure,
# and the gate checks that about itself first. The control (level 11, the
# largest valid level) must exit 0 and print the sentinel.
#
# Engine: bin/souc (Madaros) by default; set SOUNIO_SOUC_ENGINE=lean_single to
# run the same probes on the bootstrap engine.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
SOUC="${SOUC:-$ROOT/bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
PANIC_RC=1

echo "== cayley_dickson_level_refusal_gate (engine: ${SOUNIO_SOUC_ENGINE:-default}) =="

write_probe() {
  cat > "$OUT/cd_refusal_$1.sio" <<PROBE
//@ run-pass
use algebra::cayley_dickson::*
use algebra::cayley_dickson_tower::*
fn main() with Mut, Div, Panic, IO {
    $2
    println("CD_REFUSAL_ESCAPED")
}
PROBE
}

run_probe() {
  local name="$1" elf="$OUT/cd_refusal_$1.elf"
  if ! "$SOUC" compile "$OUT/cd_refusal_$name.sio" -o "$elf" >"$OUT/${name}_compile.log" 2>&1; then
    echo "FAIL: probe '$name' did not compile"; tail -40 "$OUT/${name}_compile.log" || true; exit 1
  fi
  chmod +x "$elf"
  set +e; "$elf" >"$OUT/${name}_run.log" 2>&1; PROBE_RC=$?; set -e
}

is_refusal() { [ "$PROBE_RC" -eq "$PANIC_RC" ] && ! grep -q 'CD_REFUSAL_ESCAPED' "$OUT/$1_run.log"; }

refused() {
  run_probe "$1"
  if ! is_refusal "$1"; then
    echo "FAIL: probe '$1' was NOT refused by a panic (rc $PROBE_RC, want $PANIC_RC and no sentinel)"
    cat "$OUT/$1_run.log" || true; exit 1
  fi
  echo "  refused: $1 (rc $PROBE_RC)"
}

# Self-check: an ordinary non-zero exit and a crash must not count as refusals.
# (Stand-in executables: lean_single refuses an early `return` in these probes'
# main, so the stray-exit case is a shell stub, like the crash case.)
printf '#!/bin/sh\nexit 7\n' > "$OUT/sab_exit7.sh"; chmod +x "$OUT/sab_exit7.sh"
set +e; "$OUT/sab_exit7.sh" >"$OUT/sab_exit7_run.log" 2>&1; PROBE_RC=$?; set -e
if is_refusal sab_exit7; then echo "FAIL: gate classified exit $PROBE_RC as a panic refusal"; exit 1; fi
echo "  self-check ok: exit $PROBE_RC is not a refusal"
printf '#!/bin/sh\nkill -SEGV $$\n' > "$OUT/sab_segv.sh"; chmod +x "$OUT/sab_segv.sh"
set +e; "$OUT/sab_segv.sh" >"$OUT/sab_segv_run.log" 2>&1; PROBE_RC=$?; set -e
if is_refusal sab_segv; then echo "FAIL: gate classified a SIGSEGV (rc $PROBE_RC) as a panic refusal"; exit 1; fi
echo "  self-check ok: SIGSEGV (rc $PROBE_RC) is not a refusal"

write_probe control "let a = cd_zero(11)
    let h = cd_split(cd_one(11))
    let e = cd_embed(cd_one(10), 11)
    let ok = cd_eq(a, cd_zero(11)) && cd_eq(e, cd_one(11)) && cd_eq(cd_join(h.low, h.high, 11), cd_one(11))
    if ok { println(\"CD_LEVEL_11_OK\") }"
run_probe control
if [ "$PROBE_RC" -ne 0 ] || ! grep -q 'CD_REFUSAL_ESCAPED' "$OUT/control_run.log" || ! grep -q 'CD_LEVEL_11_OK' "$OUT/control_run.log"; then
  echo "FAIL: level-11 control did not run to completion (rc $PROBE_RC)"; cat "$OUT/control_run.log" || true; exit 1
fi
echo "  control ok: level 11 accepted"

write_probe eq_12 "let r = cd_eq(cd_zero(12), cd_zero(12))"
refused eq_12
write_probe split_12 "let r = cd_split(cd_zero(12))"
refused split_12
write_probe embed_neg "let r = cd_embed(cd_zero(0 - 1), 0)"
refused embed_neg
write_probe join_low_mismatch "let r = cd_join(cd_zero(2), cd_zero(3), 4)"
refused join_low_mismatch
write_probe join_high_mismatch "let r = cd_join(cd_zero(3), cd_zero(2), 4)"
refused join_high_mismatch

echo "CD_LEVEL_REFUSAL_GATE_PASS probes=5 controls=1"
