#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIX="$ROOT_DIR/tests/multimodule/duplicate_main_collapse/basic"
TAG="[madaros-duplicate-main-collapse]"

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

if [[ -n "${SOUNIO_MADAROS_DUP_MAIN_GATE_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_DUP_MAIN_GATE_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir "$WORK" || fail "could not create gate directory: $WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-duplicate-main.XXXXXX)"
fi
# Compiler invocation changes to the fixture directory. Resolve caller-provided
# relative overrides now so every later artifact path names the same directory.
WORK="$(cd "$WORK" && pwd -P)"
if [[ -z "${SOUNIO_MADAROS_DUP_MAIN_GATE_KEEP:-}" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

RAW="${SOUNIO_MADAROS_DUP_MAIN_GATE_BIN:-${MADAROS_RAW_BIN:-}}"
if [[ -z "$RAW" ]]; then
  RAW="$WORK/madaros-from-source.elf"
  if ! bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$RAW" >"$WORK/build.log" 2>&1; then
    tail -n 40 "$WORK/build.log" >&2 || true
    fail "could not build Madaros from source"
  fi
fi
[[ -x "$RAW" ]] || fail "Madaros is missing or not executable: $RAW"
[[ "$(head -c 2 "$RAW")" != '#!' ]] || fail "not a raw ELF: $RAW"
case "$RAW" in /*) ;; *) RAW="$PWD/$RAW" ;; esac

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
if ! (cd "$FIX" && "$RAW" --native-compile main.sio -o "$WORK/probe.elf") >"$WORK/compile.log" 2>&1; then
  tail -n 40 "$WORK/compile.log" >&2 || true
  fail "specialized-collapse fixture did not compile"
fi
grep -Fq "imported_compile: specialized_collapse lower_count=1" "$WORK/compile.log" || {
  tail -n 40 "$WORK/compile.log" >&2 || true
  fail "fixture did not exercise the specialized-collapse lowering path"
}
if grep -Fq "imported_compile: specialized lower failed (" "$WORK/compile.log"; then
  tail -n 40 "$WORK/compile.log" >&2 || true
  fail "specialized lowering failed and fell back to the ordinary multi-module path"
fi
[[ -s "$WORK/probe.elf" ]] || fail "compiler did not emit an ELF"
chmod +x "$WORK/probe.elf"
if ! timeout 30 "$WORK/probe.elf" >"$WORK/actual.txt" 2>&1; then
  cat "$WORK/actual.txt" >&2 || true
  fail "compiled fixture did not return success"
fi
if ! diff -u "$FIX/expected.txt" "$WORK/actual.txt" >"$WORK/output.diff"; then
  cat "$WORK/output.diff" >&2
  fail "wrong entry point or output; DEP_MAIN must be absent"
fi

echo "$TAG PASS: USER_MAIN is the sole executable entry on specialized collapse"

BAD_FIX="$ROOT_DIR/tests/multimodule/duplicate_main_collapse/typecheck_error"
BAD_ELF="$WORK/typecheck-error.elf"
if (cd "$BAD_FIX" && "$RAW" --native-compile main.sio -o "$BAD_ELF") >"$WORK/typecheck-error.log" 2>&1; then
  tail -n 40 "$WORK/typecheck-error.log" >&2 || true
  fail "dependency main semantic error was not typechecked"
fi
if [[ -s "$BAD_ELF" ]]; then
  fail "refused dependency-main fixture still emitted an ELF"
fi
grep -Fq "this binding expects a different type" "$WORK/typecheck-error.log" || {
  tail -n 40 "$WORK/typecheck-error.log" >&2 || true
  fail "dependency-main fixture was refused without the expected type mismatch"
}

echo "$TAG PASS: dependency main remains in the typecheck merge"
