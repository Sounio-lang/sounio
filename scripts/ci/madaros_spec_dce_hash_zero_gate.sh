#!/usr/bin/env bash
# scripts/ci/madaros_spec_dce_hash_zero_gate.sh
#
# self-hosted/check/specializer.sio's cross-module dead-code-elimination
# reachability marker (spec_dce_hash_insert / spec_dce_hash_find) used
# `if h == 0 { return false }` as a shortcut, indistinguishable from the
# open-addressing `marks[]` table's own empty-slot sentinel (marks[slot] == 0).
# ast_name_hash's own clamp (`if hash < 0 { 0 - hash } else { hash }`) makes
# every real hash >= 0, so a legitimate name CAN hash to exactly 0
# (`fZOXITBAFRX_E` does, confirmed both via this compiler's own ast_name_hash
# and independently in Python). A call site to such a function could never
# mark it reachable -- spec_dce_hash_insert silently refused to insert the
# hash-0 mark at all -- so spec_dce_filter_with_global_marks dropped the
# still-called FnDef from the item list outright.
#
# Measured pre-fix, on an UNMODIFIED single-module, no-import, no-collision
# program (tests/multimodule/spec_dce_hash_zero/hashzero/main.sio):
#
#     error[E137] in <main>::main at ...: use of undeclared variable
#        = name fZOXITBAFRX_E
#     IR lowering failed during merge: epistemic_export_failed
#     Compilation failed!
#
# So this defect happens to fail CLOSED today (a hard compile error, not a
# silently wrong binary) because check_program_epistemic_into's re-typecheck
# of the DCE-filtered item list still sees the dangling call. That is
# incidental to this program's exact shape, not a guarantee -- a differently
# shaped program could plausibly let this silently drop a real function
# instead. See docs/audit/MADAROS_SPEC_DCE_HASH_ZERO_2026-09-22.md.
#
# Fixed the same way as the identical bug class in a different hash table,
# self-hosted/compiler/private_fn_identity.sio's pfi_census_note/
# pfi_census_mods (2026-09-22): store/compare `key = h + 1` in marks[]
# instead of the raw hash, and drop the `h == 0` early-outs entirely, so 0
# stays an unambiguous empty-slot sentinel for every possible hash, including
# 0 itself.
#
# This gate does not build: in CI it runs against the ELF that an earlier
# step built and kept (MADAROS_RAW_BIN / SOUNIO_MADAROS_BIN, same env the
# bin/madaros launcher and madaros_private_fn_identity_gate.sh honour).
# Standalone, with no ELF named, it builds Madaros from source.
#
# usage:  MADAROS_RAW_BIN=/path/to/madaros scripts/ci/madaros_spec_dce_hash_zero_gate.sh

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "madaros_spec_dce_hash_zero"

case "$(uname -s 2>/dev/null || echo unknown)/$(uname -m 2>/dev/null || echo unknown)" in
  Linux/x86_64|Linux/amd64) ;;
  *) gate_skip "x86-64 Linux only" ;;
esac

FIX="$ROOT_DIR/tests/multimodule/spec_dce_hash_zero"
WORK="${SOUNIO_MADAROS_SPEC_DCE_HASH_ZERO_GATE_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/madaros-spec-dce-hash-zero.XXXXXX")}"
mkdir -p "$WORK"
if [[ -z "${SOUNIO_MADAROS_SPEC_DCE_HASH_ZERO_GATE_KEEP:-}" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

RAW="${SOUNIO_MADAROS_SPEC_DCE_HASH_ZERO_GATE_BIN:-${MADAROS_RAW_BIN:-${SOUNIO_MADAROS_BIN:-}}}"
if [[ -z "$RAW" ]]; then
  RAW="$WORK/madaros-from-source.elf"
  echo "[madaros-spec-dce-hash-zero] no MADAROS_RAW_BIN; building Madaros from source"
  # build_modular_madaros.sh takes the global build lock itself; do not wrap it.
  if ! bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$RAW" >"$WORK/build.log" 2>&1; then
    tail -n 40 "$WORK/build.log" >&2 || true
    gate_fail "could not build Madaros from source"
  fi
fi
require_executable "$RAW" "Madaros is missing or not executable: $RAW"
[[ "$(head -c 2 "$RAW")" != '#!' ]] || gate_fail "not a raw ELF (a wrapper script?): $RAW"

# The compiler process needs a large soft stack (same measured floor as
# madaros_private_fn_identity_gate.sh / madaros_imported_call_arity_13_gate.sh).
stack_kb="${SOUNIO_MADAROS_SPEC_DCE_HASH_ZERO_STACK_KB:-524288}"
[[ "$stack_kb" =~ ^[1-9][0-9]*$ ]] || gate_fail "invalid stack size: $stack_kb"
stack_before="$(ulimit -S -s 2>/dev/null)" || gate_fail "soft stack limit is unavailable"
if [[ "$stack_before" != "unlimited" ]] && ((stack_before < stack_kb)); then
  ulimit -S -s "$stack_kb" 2>/dev/null || gate_fail "could not raise soft stack limit to ${stack_kb} KiB"
fi

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

# compile_and_run <label> <source>  ->  $WORK/<label>.{log,elf,out}
#
# Positional `<src> -o <out>`, NOT `--native-compile <src> -o <out>`: the
# latter refuses a single-module, no-import program outright ("native
# compile disabled for single-module streaming lane"), which is exactly this
# fixture's shape. Positional is what bin/madaros itself invokes under
# `madaros compile` / `madaros build` (bin/madaros:_compile_source_to_artifact).
compile_and_run() {
  local label="$1" src="$2"
  local log="$WORK/$label.log" elf="$WORK/$label.elf" out="$WORK/$label.out"
  require_file "$src" "$label: missing fixture $src"
  if ! "$RAW" "$src" -o "$elf" >"$log" 2>&1; then
    tail -n 40 "$log" >&2 || true
    gate_fail "$label: did not compile"
  fi
  require_elf "$elf" "$label: compiler reported success but did not emit a native ELF"
  chmod +x "$elf"
  if ! timeout 30 "$elf" >"$out" 2>&1; then
    cat "$out" >&2 || true
    gate_fail "$label: compiled program did not run to completion"
  fi
}

expect_output() {
  local label="$1" expected="$2"
  if ! diff -u "$expected" "$WORK/$label.out" >"$WORK/$label.diff"; then
    cat "$WORK/$label.diff" >&2
    gate_fail "$label: program output differs from $(basename "$(dirname "$expected")")/expected.txt"
  fi
}

# --- hashzero: the reported repro --------------------------------------------
# fZOXITBAFRX_E's ast_name_hash is exactly 0. Before the fix this failed
# CLOSED with error[E137]/epistemic_export_failed (see header); it must now
# compile and run, printing the real return value.
compile_and_run hashzero "$FIX/hashzero/main.sio"
expect_output hashzero "$FIX/hashzero/expected.txt"
echo "[madaros-spec-dce-hash-zero] PASS(hashzero): a live, called fn whose name hashes to 0 survives DCE and runs"

# --- control: identical shape, non-zero hash ---------------------------------
# Proves the harness and fixture shape are sound independent of the hash-0
# defect -- if this ever fails, something else broke, not the sentinel fix.
compile_and_run control "$FIX/control/main.sio"
expect_output control "$FIX/control/expected.txt"
echo "[madaros-spec-dce-hash-zero] PASS(control): identical shape, non-zero-hash fn compiles and runs"

gate_pass "a fn whose ast_name_hash is exactly 0 is not dropped by cross-module DCE"
