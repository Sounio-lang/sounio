#!/usr/bin/env bash
# A function named like a native float builtin (sqrt, exp, log, sin, cos, pow)
# is called from its own module and from modules that import it, and nowhere
# else. Before the fix, defining one anywhere in the module closure made it the
# builtin of the whole program: main.sio printed sqrt(4.0) = 104 without
# importing it, and `use math::pure::{exp}` made a bare sqrt require Panic.
# Compiles and RUNS each fixture and compares the output to expected.txt.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIX="$ROOT_DIR/tests/multimodule/builtin_math_shadow"
TAG="[madaros-builtin-math-shadow]"

fail() {
  echo "$TAG FAIL: $*" >&2
  exit 1
}

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "$TAG SKIP: Linux-only gate" >&2
  exit 0
fi
case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *) echo "$TAG SKIP: x86-64 Linux-only gate" >&2; exit 0 ;;
esac

WORK="$(mktemp -d /tmp/sounio-madaros-builtin-math-shadow.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

RAW="${MADAROS_RAW_BIN:-}"
if [[ -z "$RAW" ]]; then
  RAW="$WORK/madaros-from-source.elf"
  echo "$TAG no MADAROS_RAW_BIN; building Madaros from source"
  if ! bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$RAW" >"$WORK/build.log" 2>&1; then
    tail -n 40 "$WORK/build.log" >&2 || true
    fail "could not build Madaros from source"
  fi
fi
[[ -x "$RAW" ]] || fail "Madaros is missing or not executable: $RAW"

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
ulimit -s unlimited 2>/dev/null || ulimit -s 1048576 2>/dev/null || true

: >"$WORK/got.txt"
for t in main imports_user pure_exp_only; do
  echo "$t:" >>"$WORK/got.txt"
  if ! "$RAW" compile "$FIX/$t.sio" -o "$WORK/$t.elf" >"$WORK/$t.log" 2>&1 || [[ ! -s "$WORK/$t.elf" ]]; then
    tail -n 30 "$WORK/$t.log" >&2 || true
    fail "$t: does not compile"
  fi
  chmod +x "$WORK/$t.elf"
  rc=0
  "$WORK/$t.elf" >>"$WORK/got.txt" 2>&1 || rc=$?
  [[ "$rc" -eq 0 ]] || fail "$t: exited $rc"
done
if ! diff -u "$FIX/expected.txt" "$WORK/got.txt"; then
  fail "output differs from expected.txt"
fi
echo "$TAG PASS"
