#!/usr/bin/env bash
# Same-named definitions in different modules must not share a global-initialiser fold.
#
# The parser folds pure-fn calls and global references in global initialisers
# (`var G: i64 = helper()`) through a side table (GLOBAL_VAR_INIT_*, parser/ast.sio)
# that is keyed by the BARE name and deliberately accumulates across modules, so
# imported constants survive to lower time. Two modules that each define a private
# `fn helper` therefore shared one key: the second module's fn was never recorded,
# and `var G = helper()` in that module folded the FIRST module's body (no
# diagnostic; the import order picked the winner). private_fn_identity.sio renames
# functions AFTER parsing, so it cannot reach this. Every table word now carries the
# module that recorded it, and parse-time lookups resolve a name in one module.
#
# What this gate pins, against tests/multimodule/global_init_identity/:
#   basic      the reported repro (a=1 b=20 g=20), in BOTH import orders
#   fold       same-named private paramful fns and an element-list global
#   globals    same-named private globals referenced by an initialiser
#   import     control: another module's importable fn/global still folds; a foreign
#              fn body is evaluated in ITS module; an imported f64 still lands in BSS
#   shadow     a module defining `helper` itself must not adopt another's `helper`
#   ambiguous  a name two other modules define is REPORTED and left unfolded
# and positive controls: with SOUNIO_DISABLE_GLOBAL_INIT_MODULE_SCOPE=1 (the old
# bare-name table) the repros must NOT come out right and the ambiguity warning must
# not appear, so a green run cannot be a program that never collided.
#
# In CI this gate does not build: it runs against the ELF that the
# madaros_current_source_f64_lowering_gate.sh step built and kept
# (MADAROS_RAW_BIN). Standalone, with no ELF named, it builds Madaros from source.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIX="$ROOT_DIR/tests/multimodule/global_init_identity"
TAG="[madaros-global-init-identity]"
WORK="${SOUNIO_MADAROS_GLOBAL_INIT_IDENTITY_GATE_DIR:-$(mktemp -d /tmp/sounio-madaros-global-init-identity.XXXXXX)}"
OFF="SOUNIO_DISABLE_GLOBAL_INIT_MODULE_SCOPE=1"
AMBIG_WARN='warning[global_init_ambiguous]'

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
if [[ -z "${SOUNIO_MADAROS_GLOBAL_INIT_IDENTITY_GATE_KEEP:-}" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

RAW="${SOUNIO_MADAROS_GLOBAL_INIT_IDENTITY_GATE_BIN:-${MADAROS_RAW_BIN:-}}"
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
stack_kb="${SOUNIO_MADAROS_GLOBAL_INIT_IDENTITY_STACK_KB:-524288}"
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
# A control must differ from the fixed run in the switch alone.
unset SOUNIO_DISABLE_GLOBAL_INIT_MODULE_SCOPE SOUNIO_DISABLE_PRIVATE_FN_IDENTITY

# compile_and_run <label> <source> [ENV=VAL ...]  ->  $WORK/<label>.{log,elf,out}
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
    fail "$label: program output differs from $(basename "$(dirname "$expected")")/expected.txt (a same-named definition in another module leaked into this module's global initialiser)"
  fi
}

# expect_no_ambiguity <label>: only the ambiguous case may print the warning.
expect_no_ambiguity() {
  if grep -Fq -- "$AMBIG_WARN" "$WORK/$1.log"; then
    grep -F -- "$AMBIG_WARN" "$WORK/$1.log" >&2 || true
    fail "$1: printed $AMBIG_WARN although no name is defined by two other modules"
  fi
}

# expect_control_defect <label> <expected-file> <line>...: with the table scoping off
# the run must differ from the fixed output AND contain every listed defect line, so
# the control fails for the reason under test and not for some other reason.
expect_control_defect() {
  local label="$1" expected="$2"
  shift 2
  if diff -q "$expected" "$WORK/$label.out" >/dev/null 2>&1; then
    fail "$label: with $OFF the output still equals expected.txt; the fixture no longer exercises the collision (or it was fixed some other way -- update this control, not the fix)"
  fi
  local line
  for line in "$@"; do
    grep -Fxq -- "$line" "$WORK/$label.out" || {
      cat "$WORK/$label.out" >&2 || true
      fail "$label: with $OFF the output lacks the expected defect line '$line'"
    }
  done
  expect_no_ambiguity "$label"
}

# --- basic: the reported repro, both import orders ---------------------------
compile_and_run basic "$FIX/basic/main.sio"
expect_output basic "$FIX/basic/expected.txt"
expect_no_ambiguity basic
compile_and_run basic_swapped "$FIX/basic/main_swapped.sio"
expect_output basic_swapped "$FIX/basic/expected.txt"
echo "$TAG PASS(basic): a=1 b=20 g=20 in both import orders"

# --- fold ---------------------------------------------------------------------
compile_and_run fold "$FIX/fold/main.sio"
expect_output fold "$FIX/fold/expected.txt"
expect_no_ambiguity fold
echo "$TAG PASS(fold): same-named paramful fns and an element-list global fold per module"

# --- globals ------------------------------------------------------------------
compile_and_run globals "$FIX/globals/main.sio"
expect_output globals "$FIX/globals/expected.txt"
expect_no_ambiguity globals
echo "$TAG PASS(globals): same-named globals in an initialiser resolve per module"

# --- import: nothing that used to fold across modules may stop folding --------
compile_and_run import "$FIX/import/main.sio"
expect_output import "$FIX/import/expected.txt"
expect_no_ambiguity import
echo "$TAG PASS(import): importable fn/global still fold; foreign body evaluated in its module; imported f64 intact"

# --- shadow -------------------------------------------------------------------
compile_and_run shadow "$FIX/shadow/main.sio"
expect_output shadow "$FIX/shadow/expected.txt"
expect_no_ambiguity shadow
echo "$TAG PASS(shadow): a module's own helper is never replaced by another module's"

# --- ambiguous: reported, not folded -----------------------------------------
compile_and_run ambiguous "$FIX/ambiguous/main.sio"
expect_output ambiguous "$FIX/ambiguous/expected.txt"
grep -Fq -- "$AMBIG_WARN" "$WORK/ambiguous.log" || {
  tail -n 25 "$WORK/ambiguous.log" >&2 || true
  fail "ambiguous: compile log lacks $AMBIG_WARN"
}
echo "$TAG PASS(ambiguous): a name two other modules define is reported and left unfolded"

# --- positive controls: scoping off => the defects are back -------------------
# A green gate must be able to go red. With the old bare-name table the fixtures
# above must fail in exactly the way they were measured to before the fix.
compile_and_run control_basic "$FIX/basic/main.sio" "$OFF"
expect_control_defect control_basic "$FIX/basic/expected.txt" "a=1" "b=20" "g=1"
compile_and_run control_fold "$FIX/fold/main.sio" "$OFF"
expect_control_defect control_fold "$FIX/fold/expected.txt" "sb=6" "bb=6" "tb0=2"
compile_and_run control_globals "$FIX/globals/main.sio" "$OFF"
expect_control_defect control_globals "$FIX/globals/expected.txt" "ha=6" "hb=0"
compile_and_run control_import "$FIX/import/main.sio" "$OFF"
expect_control_defect control_import "$FIX/import/expected.txt" "l=3"
compile_and_run control_shadow "$FIX/shadow/main.sio" "$OFF"
expect_control_defect control_shadow "$FIX/shadow/expected.txt" "x_is_foreign=1" "x2_is_foreign=1"
compile_and_run control_ambiguous "$FIX/ambiguous/main.sio" "$OFF"
expect_no_ambiguity control_ambiguous
echo "$TAG PASS(control): scoping disabled reproduces every defect ($(tr '\n' ' ' <"$WORK/control_basic.out"))"

echo "$TAG PASS: same-named definitions in different modules keep separate global-initialiser folds"
