#!/usr/bin/env bash
# #2813: bin/madaros must refuse, with a diagnostic, to run the compiler when it
# cannot reserve the stack the compiler needs -- never let it SIGSEGV.
#
# The launcher reserves MADAROS_STACK_KB (default 512 MiB) with `ulimit -s`.
# That used to be best-effort (`|| true`): under a hard limit below it (for
# example `ulimit -Hs 8192`, common in containers) the raw compiler ran on the
# 8 MiB default and crashed with rc=139 compiling large programs (measured on
# a program that imports native::frame: "Segmentation fault ... compiler exited
# with status 139").
#
# Checks, all under `ulimit -Ss 8192; ulimit -Hs 8192` unless noted:
#   1. `madaros build` exits 75 and names the limits; no SIGSEGV, no ELF.
#   2. `madaros run` and `madaros check` refuse the same way.
#   3. `madaros --version` and `madaros info` still work (they run no compile).
#   4. MADAROS_STACK_KB=8192 is the explicit opt-in: hello.sio builds and runs.
#   5. Control, default limits: hello.sio builds and runs (no refusal).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
TAG="madaros_stack_reservation_gate"
fail() { echo "$TAG: FAIL $*" >&2; exit 1; }

MADAROS="$ROOT/bin/madaros"
SRC="$ROOT/tests/run-pass/hello.sio"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-stack-gate.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
unset MADAROS_STACK_KB

if [[ "$(ulimit -H -s)" != "unlimited" ]] && (( $(ulimit -H -s) < 524288 )); then
  echo "$TAG: SKIP: this runner's hard stack limit ($(ulimit -H -s) KiB) is itself below the reservation; checks 4-5 cannot be run as the control" >&2
  exit 0
fi

limited() {  # limited <cmd...>: run under a hard 8 MiB stack
  bash -c 'ulimit -Ss 8192 && ulimit -Hs 8192 && exec "$@"' _ "$@"
}

# 1. build refuses
rc=0; limited "$MADAROS" build "$SRC" -o "$WORK/a.elf" >"$WORK/1.log" 2>&1 || rc=$?
[[ $rc -eq 75 ]] || { cat "$WORK/1.log" >&2; fail "build under a hard 8 MiB stack exited $rc, expected 75"; }
grep -q "cannot reserve the compiler stack: need 524288 KiB, soft limit is 8192 KiB (hard limit 8192 KiB)" "$WORK/1.log" \
  || { cat "$WORK/1.log" >&2; fail "refusal does not name the limits"; }
! grep -qi "segmentation fault\|status 139" "$WORK/1.log" || fail "compiler still crashed"
[[ ! -e "$WORK/a.elf" ]] || fail "an ELF was written despite the refusal"
echo "$TAG: PASS build refuses (rc=75): $(grep -m1 'cannot reserve' "$WORK/1.log")"

# 2. run and check refuse the same way
for sub in run check; do
  rc=0; limited "$MADAROS" "$sub" "$SRC" >"$WORK/2.$sub.log" 2>&1 || rc=$?
  [[ $rc -eq 75 ]] || { cat "$WORK/2.$sub.log" >&2; fail "$sub under a hard 8 MiB stack exited $rc, expected 75"; }
done
echo "$TAG: PASS run/check refuse (rc=75)"

# 3. no-compile commands are not gated
limited "$MADAROS" --version >"$WORK/3.log" 2>&1 || { cat "$WORK/3.log" >&2; fail "--version refused"; }
limited "$MADAROS" info >"$WORK/3i.log" 2>&1 || { cat "$WORK/3i.log" >&2; fail "info refused"; }
echo "$TAG: PASS --version / info unaffected"

# 4. explicit opt-in
rc=0; MADAROS_STACK_KB=8192 limited "$MADAROS" build "$SRC" -o "$WORK/b.elf" >"$WORK/4.log" 2>&1 || rc=$?
[[ $rc -eq 0 && -x "$WORK/b.elf" ]] || { cat "$WORK/4.log" >&2; fail "MADAROS_STACK_KB=8192 opt-in did not build hello.sio (rc=$rc)"; }
"$WORK/b.elf" | grep -q . || fail "opt-in ELF printed nothing"
echo "$TAG: PASS MADAROS_STACK_KB=8192 opt-in builds and runs hello.sio"

# 5. control
rc=0; "$MADAROS" build "$SRC" -o "$WORK/c.elf" >"$WORK/5.log" 2>&1 || rc=$?
[[ $rc -eq 0 && -x "$WORK/c.elf" ]] || { cat "$WORK/5.log" >&2; fail "default build of hello.sio failed (rc=$rc)"; }
echo "$TAG: PASS default limits build hello.sio"

echo "MADAROS_STACK_RESERVATION_GATE_OK"
