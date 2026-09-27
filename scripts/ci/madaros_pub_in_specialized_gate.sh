#!/usr/bin/env bash
# Copilot review (PR #2515), comment 4113458034: check_items_verdict_boot4_
# with_module_map (the specialized multi-module typecheck a generic
# instantiation anywhere routes the whole program through) stamped
# current_module_id per item but left current_module fixed at empty_path(),
# so a `pub(in path)` access was refused unconditionally through this path,
# for every caller, regardless of what the restriction actually named. See
# tests/multimodule/thinlink_pub_in_specialized/README.md for the fixtures
# and why each is invoked with a bare, zero-slash filename.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIX="$ROOT_DIR/tests/multimodule/thinlink_pub_in_specialized"
TAG="[madaros-pub-in-specialized]"

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

# Copilot review (PR #2515), comment 4114195565: `mkdir -p` accepts an
# override that names an existing directory, and the EXIT trap below then
# recursively deletes that caller-owned directory once KEEP is unset. Same
# guard as madaros_tuple_arr_capacity_boundary_gate.sh's own override:
# refuse an override that already exists and create it ourselves, so only
# a directory THIS run made can ever be the trap's target.
if [[ -n "${SOUNIO_MADAROS_PUB_IN_SPECIALIZED_GATE_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_PUB_IN_SPECIALIZED_GATE_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir "$WORK" || fail "could not create gate directory: $WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-pub-in-specialized.XXXXXX)"
fi
if [[ -z "${SOUNIO_MADAROS_PUB_IN_SPECIALIZED_GATE_KEEP:-}" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

RAW="${SOUNIO_MADAROS_PUB_IN_SPECIALIZED_GATE_BIN:-${MADAROS_RAW_BIN:-}}"
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
case "$RAW" in
  /*) ;;
  *) RAW="$PWD/$RAW" ;;
esac

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

# compile_and_run <label> <fixture-dir>: invoked with CWD set to
# <fixture-dir> and a BARE "main.sio" argument (zero slashes), so
# file_path_to_module_path derives the bare module name ("main") instead of
# embedding this fixture directory's own name as a parent-directory prefix
# -- see the README for why that distinction matters for pub(in path).
compile_and_run() {
  local label="$1" dir="$2"
  local log="$WORK/$label.log" elf="$WORK/$label.elf" out="$WORK/$label.out"
  [[ -f "$dir/main.sio" ]] || fail "$label: missing fixture $dir/main.sio"
  if ! (cd "$dir" && "$RAW" --native-compile main.sio -o "$elf") >"$log" 2>&1; then
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
    fail "$label: program output differs from expected.txt (pub(in path) access wrongly refused, or wrongly accepted)"
  fi
}

# expect_refusal <label> <fixture-dir>: same invocation shape, but the
# compile must FAIL closed with error[E175] (report_private_fn) and write
# no ELF.
expect_refusal() {
  local label="$1" dir="$2"
  local log="$WORK/$label.log" elf="$WORK/$label.elf"
  [[ -f "$dir/main.sio" ]] || fail "$label: missing fixture $dir/main.sio"
  if (cd "$dir" && "$RAW" --native-compile main.sio -o "$elf") >"$log" 2>&1; then
    tail -n 25 "$log" >&2 || true
    fail "$label: compiled, but this access should have been refused (a known-wrong executable)"
  fi
  grep -Fq "error[E175]" "$log" || {
    tail -n 25 "$log" >&2 || true
    fail "$label: refused, but without the expected error[E175] diagnostic"
  }
  if [[ -s "$elf" ]]; then
    fail "$label: the compile was refused but an ELF was still written"
  fi
}

# --- basic: pub(main), accessed from the module literally named `main` -----
compile_and_run basic "$FIX/basic"
expect_output basic "$FIX/basic/expected.txt"
echo "$TAG PASS(basic): pub(main) access accepted through the specialized multi-module path"

# --- wrong_module: pub(nobody), a module nothing here is named -------------
expect_refusal wrong_module "$FIX/wrong_module"
echo "$TAG PASS(wrong_module): pub(nobody) access still refused -- a real check, not always-true"

echo "$TAG PASS: pub(in path) visibility correct through the specialized multi-module checker"
