#!/usr/bin/env bash
# Same-named PRIVATE functions in different modules must keep separate identities.
#
# The merged IR identified functions by UNQUALIFIED name: two modules that each
# defined a private `fn helper` shared one body, the first-loaded module won, every
# caller in every module ran it, and the program compiled clean. The type checker
# already kept them apart (fn_sig_table_find_prefer_module), so this was accepted
# and then miscompiled; swapping two `use` lines swapped the winner.
# self-hosted/compiler/private_fn_identity.sio now gives each such function a
# module-qualified name before anything keys on it.
#
# What this gate pins, against the fixtures in tests/multimodule/private_fn_identity/:
#   basic  the reported repro (a=1 b=20), in BOTH import orders
#   rich   3 modules; colliding private fns incl. recursion, fn-as-value, an impl
#          method, an identical benign copy, and a pub fn that must keep its name
#   skip   a shape the pass cannot prove safe must be REPORTED, not guessed
# and a positive control: with the pass switched off
# (SOUNIO_DISABLE_PRIVATE_FN_IDENTITY=1) the repro must NOT come out right, so a
# green run cannot be a program that never exercised the collision.
#
# In CI this gate does not build: it runs against the ELF that the
# madaros_current_source_f64_lowering_gate.sh step built and kept
# (MADAROS_RAW_BIN). Standalone, with no ELF named, it builds Madaros from source.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIX="$ROOT_DIR/tests/multimodule/private_fn_identity"
TAG="[madaros-private-fn-identity]"
WORK="${SOUNIO_MADAROS_PRIVATE_FN_IDENTITY_GATE_DIR:-$(mktemp -d /tmp/sounio-madaros-private-fn-identity.XXXXXX)}"

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
if [[ -z "${SOUNIO_MADAROS_PRIVATE_FN_IDENTITY_GATE_KEEP:-}" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

RAW="${SOUNIO_MADAROS_PRIVATE_FN_IDENTITY_GATE_BIN:-${MADAROS_RAW_BIN:-}}"
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

# The compiler process itself needs a large soft stack (see
# madaros_imported_call_arity_13_gate.sh for the measurements).
stack_kb="${SOUNIO_MADAROS_PRIVATE_FN_IDENTITY_STACK_KB:-524288}"
[[ "$stack_kb" =~ ^[1-9][0-9]*$ && ${#stack_kb} -le 9 ]] || fail "invalid stack size: $stack_kb"
stack_before="$(ulimit -S -s 2>/dev/null)" || fail "soft stack limit is unavailable"
if [[ "$stack_before" != "unlimited" ]] && ((stack_before < stack_kb)); then
  ulimit -S -s "$stack_kb" 2>/dev/null || fail "could not raise soft stack limit to ${stack_kb} KiB"
fi
stack_after="$(ulimit -S -s 2>/dev/null)" || fail "soft stack limit is unavailable after update"
if [[ "$stack_after" != "unlimited" ]] && ((stack_after < stack_kb)); then
  fail "soft stack limit remained below ${stack_kb} KiB: $stack_after"
fi
echo "$TAG stack_kb before=$stack_before after=$stack_after requested=$stack_kb"

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

# compile_case <label> <source> [ENV=VAL ...]  ->  $WORK/<label>.{log,elf,out}
compile_and_run() {
  local label="$1" src="$2"
  shift 2
  local log="$WORK/$label.log" elf="$WORK/$label.elf" out="$WORK/$label.out"
  [[ -f "$src" ]] || fail "$label: missing fixture $src"
  if ! env "$@" "$RAW" --native-compile "$src" -o "$elf" >"$log" 2>&1; then
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

# expect_output <label> <expected-file>: exact stdout, no other lines.
expect_output() {
  local label="$1" expected="$2"
  if ! diff -u "$expected" "$WORK/$label.out" >"$WORK/$label.diff"; then
    cat "$WORK/$label.diff" >&2
    fail "$label: program output differs from $(basename "$(dirname "$expected")")/expected.txt (same-named private functions were merged)"
  fi
}

expect_log() {
  local label="$1" pattern="$2" what="$3"
  grep -Fq -- "$pattern" "$WORK/$label.log" || {
    tail -n 25 "$WORK/$label.log" >&2 || true
    fail "$label: compile log lacks $what (expected: $pattern)"
  }
}

# --- basic: the reported repro, both import orders --------------------------
compile_and_run basic "$FIX/basic/main.sio"
expect_output basic "$FIX/basic/expected.txt"
expect_log basic "private_fn_identity: renamed 2 same-named private fn(s) in 2 module(s)" "the rename receipt"

compile_and_run basic_swapped "$FIX/basic/main_swapped.sio"
expect_output basic_swapped "$FIX/basic/expected.txt"
echo "$TAG PASS(basic): a=1 b=20 in both import orders"

# --- rich -------------------------------------------------------------------
compile_and_run rich "$FIX/rich/main.sio"
expect_output rich "$FIX/rich/expected.txt"
expect_log rich "private_fn_identity: renamed" "the rename receipt"
echo "$TAG PASS(rich): colliding private fns, recursion, fn values, impl method, pub/private mix"

# --- skip: unprovable shape is reported, not guessed ------------------------
compile_and_run skip "$FIX/skip/main.sio"
expect_output skip "$FIX/skip/expected.txt"
expect_log skip 'warning[private_fn_identity]: private fn `helper` in module #1' "the skipped-rename warning"
echo "$TAG PASS(skip): unprovable rename reported; the provable one still applied"

# --- positive control: pass off => the repro is NOT right -------------------
# A green gate must be able to go red. With the pass disabled the merged IR is
# back to first-loaded-wins, so the repro must not print the expected line. If it
# does, these fixtures no longer exercise the collision (or the bug has been fixed
# some other way and this control -- not the fix -- is what needs updating).
compile_and_run control "$FIX/basic/main.sio" SOUNIO_DISABLE_PRIVATE_FN_IDENTITY=1
if diff -q "$FIX/basic/expected.txt" "$WORK/control.out" >/dev/null 2>&1; then
  fail "control: with SOUNIO_DISABLE_PRIVATE_FN_IDENTITY=1 the repro still printed the correct answer; the gate cannot tell fixed from broken"
fi
if grep -Fq "private_fn_identity: renamed" "$WORK/control.log"; then
  fail "control: the pass ran although SOUNIO_DISABLE_PRIVATE_FN_IDENTITY=1"
fi
echo "$TAG PASS(control): pass disabled reproduces the defect ($(tr -d '\n' <"$WORK/control.out"))"

echo "$TAG PASS: same-named private fns keep separate identities across modules"
