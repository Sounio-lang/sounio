#!/usr/bin/env bash
# Copilot review (PR #2516), comment 4113449699: an IMPORTED tuple-array-
# returning callee, through the public native-compile multi-module path,
# not a lowerer probe. The current CLI uses module_native_driver's full IR
# route, not module_loader's legacy thin-link unit builder. The fixture name
# is historical; its assertions still pin tuple-array return classification.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIX="$ROOT_DIR/tests/multimodule/thinlink_tuple_arr_import"
TAG="[madaros-thinlink-tuple-arr-import]"
# Copilot review (PR #2516), comment 4116326226: match the sibling gates'
# KEEP semantics (e.g. madaros_tuple_arr_capacity_boundary_gate.sh:34,55) --
# only "1" retains the work directory; every other value, including an
# explicit "0", still cleans up. The old `-z` check inverted that: setting
# KEEP=0 was falsy-but-nonempty, so `-z` was false and the cleanup trap was
# skipped, unexpectedly retaining the directory.
KEEP_WORK="${SOUNIO_MADAROS_THINLINK_TUPLE_ARR_IMPORT_GATE_KEEP:-0}"

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

# Copilot review (PR #2516), comment 4113800060: an override that names an
# EXISTING directory would otherwise get `rm -rf`'d by the EXIT trap below
# once KEEP is unset -- a CI/local config typo (or a path meant for
# something else entirely) must not silently become a delete. Same guard
# as madaros_tuple_arr_capacity_boundary_gate.sh's own SOUNIO_MADAROS_
# TUPLE_ARR_CAP_GATE_DIR: refuse an override that already exists and
# create it ourselves, so only a directory THIS run made can ever be the
# trap's target.
if [[ -n "${SOUNIO_MADAROS_THINLINK_TUPLE_ARR_IMPORT_GATE_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_THINLINK_TUPLE_ARR_IMPORT_GATE_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir "$WORK" || fail "could not create gate directory: $WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-thinlink-tuple-arr-import.XXXXXX)"
fi
if [[ "$KEEP_WORK" != "1" ]]; then
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
# Exercise the shipped default, even if the caller exported the A/B opt-out.
unset SOUNIO_NO_REGION_RECLAIM

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

# --- basic: imported tuple-array callee vs. an unrelated same-named private fn ---
compile_and_run basic "$FIX/basic/main.sio"
expect_output basic "$FIX/basic/expected.txt"
echo "$TAG PASS(basic): imported (i64, [f64; 2])-returning callee classifies correctly, not clobbered by an unrelated, uncalled, same-named private fn collected first"

# --- zero_mask: selected zero mask must overwrite a later namesake mask ---
compile_and_run zero_mask "$FIX/zero_mask/main.sio"
expect_output zero_mask "$FIX/zero_mask/expected.txt"
echo "$TAG PASS(zero_mask): selected integer-array tuple return records an authoritative zero mask, not a later namesake f64-array mask"

echo "$TAG PASS: imported tuple-array callee through the default modular native path"
