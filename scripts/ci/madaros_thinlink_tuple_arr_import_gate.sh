#!/usr/bin/env bash
# Copilot review (PR #2516), comment 4113449699: an IMPORTED tuple-array-
# returning callee, through the real thin-link multi-module path
# (module_loader.sio's thin_build_compiled_unit), not a lowerer probe entry
# point. See tests/multimodule/thinlink_tuple_arr_import/README.md for what
# the `basic` fixture pins and does not pin.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIX="$ROOT_DIR/tests/multimodule/thinlink_tuple_arr_import"
TAG="[madaros-thinlink-tuple-arr-import]"
WORK="${SOUNIO_MADAROS_THINLINK_TUPLE_ARR_IMPORT_GATE_DIR:-$(mktemp -d /tmp/sounio-madaros-thinlink-tuple-arr-import.XXXXXX)}"

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

mkdir -p "$WORK"
if [[ -z "${SOUNIO_MADAROS_THINLINK_TUPLE_ARR_IMPORT_GATE_KEEP:-}" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

RAW="${SOUNIO_MADAROS_THINLINK_TUPLE_ARR_IMPORT_GATE_BIN:-${MADAROS_RAW_BIN:-}}"
if [[ -z "$RAW" ]]; then
  RAW="$WORK/madaros-from-source.elf"
  echo "$TAG no MADAROS_RAW_BIN; building Madaros from source"
  # build_modular_madaros.sh takes the global build lock itself; do not wrap it.
  if ! bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$RAW" >"$WORK/build.log" 2>&1; then
    tail -n 40 "$WORK/build.log" >&2 || true
    fail "could not build Madaros from source"
  fi
fi
[[ -x "$RAW" ]] || fail "Madaros is missing or not executable: $RAW"
[[ "$(head -c 2 "$RAW")" != '#!' ]] || fail "not a raw ELF (a wrapper script?): $RAW"

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

compile_and_run() {
  local label="$1" src="$2"
  local log="$WORK/$label.log" elf="$WORK/$label.elf" out="$WORK/$label.out"
  [[ -f "$src" ]] || fail "$label: missing fixture $src"
  if ! "$RAW" --native-compile "$src" -o "$elf" >"$log" 2>&1; then
    tail -n 40 "$log" >&2 || true
    fail "$label: did not compile"
  fi
  [[ -s "$elf" ]] || fail "$label: compiler did not emit an ELF"
  chmod +x "$elf"
  if ! timeout 30 "$elf" >"$out" 2>&1; then
    cat "$out" >&2 || true
    fail "$label: compiled program did not run to completion"
  fi
}

expect_output() {
  local label="$1" expected="$2"
  if ! diff -u "$expected" "$WORK/$label.out" >"$WORK/$label.diff"; then
    cat "$WORK/$label.diff" >&2
    fail "$label: program output differs from expected.txt (imported tuple-array element misclassified as integer, or same-name unit collision)"
  fi
}

# --- basic: imported tuple-array callee + same-named helper across units ---
compile_and_run basic "$FIX/basic/main.sio"
expect_output basic "$FIX/basic/expected.txt"
echo "$TAG PASS(basic): imported (f64, [f64; 3])-returning callee classifies correctly; same-named helper across two units does not cross-contaminate"

echo "$TAG PASS: imported tuple-array callee through the real thin-link unit builder"
