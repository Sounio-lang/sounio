#!/usr/bin/env bash
# The committed Madaros prebuilt must compile the bootstrap compiler to the seed.
#
# S' = self-hosted/compiler/lean_single.sio. The committed seed
# bin/souc-lean-single-x86_64 is its byte-identical fixed point
# (canonical_compiler_gate.sh). So a correct Madaros M has a checkable property:
# M(S') is a lean_single compiler, and M(S')(S') must be the seed, byte for byte.
#
# WHY THIS EXISTS. Measured 2026-11-22: the prebuilt then committed, built from
# 0ed977cec, was 309 compiler commits behind self-hosted/. It miscompiled S' in
# two ways that source had already fixed:
#   1. write_file on a packed global [i8; N] read every 8th byte, 32 bytes late
#      (#2966, fixed by f078496f87). S' writes its output ELF that way, so
#      M(S') wrote a file that was not an ELF.
#   2. `0xc3 as i8` was the identity, 195, instead of -61 (fixed by 4066c302f5).
# Every gate that runs bin/madaros-linux-x86_64 passed, because none of them
# asked it to compile a program as large as its own bootstrap seed. The refresh
# workflow kept proposing a newer build (PR #3440) that never merged, and nothing
# turned red.
#
# Two stages:
#   1. cheap: two fixtures that discriminate those two defects, compiled by the
#      prebuilt and checked against their expected output, then the E221
#      witnesses of #2720 (no main, bodyless extern "C" fn main: refused with
#      `error[E221]: no main` and no ELF; a real main still builds and runs);
#   2. full: M(S') -> MX, MX(S') -> Y, cmp Y seed. About 80 s and 7 GB of RAM
#      for the first compile; never run two at once.
#
# madaros_fixed_point_gate.sh is not this property: it compiles main.sio with
# Madaros (Madaros on itself), never S'. canonical_compiler_gate.sh runs the seed
# on S', never Madaros. This gate is the one that puts the shipped Madaros and the
# shipped seed on the same input.
#
#   scripts/ci/madaros_prebuilt_agrees_with_seed_gate.sh [--cheap-only]
#   SOUNIO_MADAROS_BIN=<elf>  test another Madaros instead of the prebuilt
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
SEED="$ROOT/bin/souc-lean-single-x86_64"
MADAROS="${SOUNIO_MADAROS_BIN:-$ROOT/bin/madaros-linux-x86_64}"
SRC="$ROOT/self-hosted/compiler/lean_single.sio"
FIXDIR="$ROOT/tests/compiler/madaros_prebuilt_agrees_with_seed"
CHEAP_ONLY=0
[[ "${1:-}" == "--cheap-only" ]] && CHEAP_ONLY=1

W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT

fail() { echo "[madaros-prebuilt-agrees-with-seed] FAIL: $*" >&2; exit 1; }
say() { echo "[madaros-prebuilt-agrees-with-seed] $*"; }

unset SOUC_BIN SOUNIO_SOUC_BIN
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
ulimit -s unlimited 2>/dev/null || true

if [[ -z "${SOUNIO_MADAROS_BIN:-}" ]]; then
  bash "$ROOT/scripts/lib/materialize_madaros_prebuilt.sh" --verify \
    || fail "bin/madaros-linux-x86_64.gz does not materialize to the ELF its .sha256 names"
fi
[[ -x "$SEED" ]] || fail "missing seed $SEED"
[[ -x "$MADAROS" ]] || fail "missing Madaros $MADAROS"
[[ "$(head -c4 "$MADAROS" | od -An -tx1 | tr -d ' ')" == "7f454c46" ]] \
  || fail "$MADAROS is not an ELF"
say "madaros: $MADAROS ($(sha256sum "$MADAROS" | cut -c1-12))"
say "seed:    $SEED ($(sha256sum "$SEED" | cut -c1-12))"

# compile_run <fixture> -> stdout of the compiled program, run in $W
compile_run() {
  local name="$1" elf="$W/$1.elf"
  "$MADAROS" compile "$FIXDIR/$name.sio" -o "$elf" >"$W/$name.compile.log" 2>&1 \
    || { tail -20 "$W/$name.compile.log" >&2; fail "Madaros refused $name.sio"; }
  chmod +x "$elf"
  (cd "$W" && "$elf") || fail "$name.sio: compiled program exited rc=$?"
}

# Stage 1: the two discriminating fixtures.
out="$(compile_run cast_i8_sign_extends)"
[[ "$out" == "-61 -61 EQ" ]] \
  || fail "cast_i8_sign_extends printed '$out', expected '-61 -61 EQ' (\`0xc3 as i8\` must be -61)"
say "stage 1: cast_i8_sign_extends -> $out"

rm -f "$W/write_file_packed_global.out"
out="$(compile_run write_file_packed_global)"
[[ "$out" == "16" ]] || fail "write_file_packed_global printed '$out', expected 16"
[[ -f "$W/write_file_packed_global.out" ]] || fail "write_file_packed_global wrote no file"
got="$(cat "$W/write_file_packed_global.out")"
[[ "$got" == "ABCDEFGHIJKLMNOP" ]] \
  || fail "write_file on a packed global wrote '$(od -An -tx1 "$W/write_file_packed_global.out" | tr -s ' ')', expected ABCDEFGHIJKLMNOP"
say "stage 1: write_file_packed_global -> $got"

# Stage 1b: E221 (#2720). An entry file with no main, or with only a bodyless
# `extern "C" fn main()`, must be refused with the exact diagnostic and must not
# leave an ELF behind; a prebuilt built before #2720 wrote one that died with
# SIGILL. The positive control with a real main must still build and run, and
# the no-main file must stay a valid library for `check` and `--emit-obj`.
for neg in e221_no_main e221_extern_main; do
  rc=0
  "$MADAROS" compile "$FIXDIR/$neg.sio" -o "$W/$neg.elf" >"$W/$neg.log" 2>&1 || rc=$?
  [[ $rc -ne 0 ]] || fail "$neg.sio compiled (rc 0); expected error[E221]: no main"
  grep -qF 'error[E221]: no main' "$W/$neg.log" \
    || { tail -10 "$W/$neg.log" >&2; fail "$neg.sio refused (rc $rc) without error[E221]: no main"; }
  [[ ! -e "$W/$neg.elf" ]] || fail "$neg.sio was refused but left an ELF at $W/$neg.elf"
  say "stage 1: $neg -> rc $rc, error[E221]: no main, no ELF"
done
out="$(compile_run e221_real_main)"
[[ "$out" == "E221_REAL_MAIN_OK" ]] || fail "e221_real_main printed '$out', expected E221_REAL_MAIN_OK"
say "stage 1: e221_real_main -> $out"
"$MADAROS" check "$FIXDIR/e221_no_main.sio" >"$W/lib.check.log" 2>&1 \
  || { tail -10 "$W/lib.check.log" >&2; fail "check refused the no-main library e221_no_main.sio"; }
"$MADAROS" --emit-obj "$FIXDIR/e221_no_main.sio" -o "$W/lib.o" >"$W/lib.obj.log" 2>&1 \
  || { tail -10 "$W/lib.obj.log" >&2; fail "--emit-obj refused the no-main library e221_no_main.sio"; }
[[ -s "$W/lib.o" ]] || fail "--emit-obj on e221_no_main.sio wrote no object"
say "stage 1: e221_no_main as a library -> check OK, --emit-obj OK"

if [[ $CHEAP_ONLY -eq 1 ]]; then
  echo "MADAROS_PREBUILT_AGREES_WITH_SEED_CHEAP_OK"
  exit 0
fi

# Stage 2: Madaros(S') is a lean_single compiler; it must rebuild the seed exactly.
t0=$(date +%s)
"$MADAROS" compile "$SRC" -o "$W/MX" >"$W/mx.log" 2>&1 \
  || { tail -20 "$W/mx.log" >&2; fail "Madaros refused self-hosted/compiler/lean_single.sio"; }
chmod +x "$W/MX"
[[ "$(head -c4 "$W/MX" | od -An -tx1 | tr -d ' ')" == "7f454c46" ]] || fail "Madaros(S') is not an ELF"
t1=$(date +%s)
"$W/MX" "$SRC" "$W/Y" >"$W/y.log" 2>&1 \
  || { tail -20 "$W/y.log" >&2; fail "Madaros(S') refused self-hosted/compiler/lean_single.sio"; }
t2=$(date +%s)
[[ -s "$W/Y" ]] || fail "Madaros(S')(S') produced no output"
say "stage 2: Madaros(S') $((t1 - t0)) s, Madaros(S')(S') $((t2 - t1)) s"
say "  Madaros(S')      $(sha256sum "$W/MX" | awk '{print $1}')"
say "  Madaros(S')(S')  $(sha256sum "$W/Y" | awk '{print $1}')"
say "  seed             $(sha256sum "$SEED" | awk '{print $1}')"
if ! cmp "$W/Y" "$SEED"; then
  fail "Madaros(S')(S') differs from bin/souc-lean-single-x86_64: the prebuilt miscompiles the bootstrap compiler. Rebuild it from main (make build-madaros) and refresh bin/madaros-linux-x86_64.gz"
fi

echo "MADAROS_PREBUILT_AGREES_WITH_SEED_OK"
