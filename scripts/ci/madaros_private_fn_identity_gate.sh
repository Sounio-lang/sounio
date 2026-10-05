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
#   basic     the reported repro (a=1 b=20), in BOTH import orders
#   rich      3 modules; colliding private fns incl. recursion, fn-as-value, an impl
#             method, an identical benign copy, and a pub fn that must keep its name
#   hashcoll  names are compared EXACTLY: `bA` shares ast_name_hash with `ab`, and a
#             reference to it must not be rewritten because `ab` was renamed
#   symcoll   a private free fn named like a method's emitted symbol (`Type_method`)
#   reserved  a generated `name__m<N>` must not reuse a symbol that already exists
#   reservedglobal  ...including a module GLOBAL (an ItemFn with no body)
#   hashadversarial  a generated-name AVAILABILITY check that trusted the hash
#             census alone falsely refused a program with two unrelated hash-
#             colliding decoys and no real collision
#   genericcollapse  the SAME collision, but through the specialized-collapse
#             pipeline (a generic instantiation anywhere routes the whole
#             program through module_frontend_specialized_prepare instead of
#             the ordinary path every other case here exercises), both import
#             orders
#   capacity  70 colliding private fns in one module (the old fixed table held 64)
#   skip      a shape the pass cannot prove safe is skipped -- and that is only
#             accepted because the other module was renamed and no collision remains
#   genshadow  the same, but the parameter shadows the GENERATED name
#             (`helper__m1`) rather than the original one -- pre-fix this was
#             renamed anyway and crashed during lowering instead of refusing
#   unresolved  the same shape in EVERY colliding module: the compile must be
#             REFUSED with error[private_fn_identity], not emitted with a warning
#   restricted  two pub(crate) fns of one name: exported, so unrenamable -> REFUSED
#   restrictedmix  a pub(crate) fn met first + a private one: the private is renamed,
#             the exported one is then unique, and it compiles
#   capacity boundary  512 colliding names are accepted, 513 are REFUSED (the
#             per-module table limit), generated here rather than committed
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

# expect_census_detected <label> <main.sio> <pattern> <what>: proves the
# CENSUS recorded a collision and drove a rename -- nothing more. Unlike
# compile_and_run, the exit code is deliberately not checked: a hash-0
# collision (see hashzero below) can compile, fail cleanly, or crash the
# compiler process itself downstream of private_fn_identity, because of a
# SEPARATE, pre-existing bug outside this pass's scope. Checking rc here
# would make this gate depend on that other bug's behavior.
expect_census_detected() {
  local label="$1" src="$2" pattern="$3" what="$4"
  local log="$WORK/$label.log" elf="$WORK/$label.elf"
  "$RAW" --native-compile "$src" -o "$elf" >"$log" 2>&1 || true
  grep -Fq -- "$pattern" "$log" || {
    tail -n 25 "$log" >&2 || true
    fail "$label: compile log lacks $what (expected: $pattern)"
  }
}

# --- basic: the reported repro, both import orders --------------------------
compile_and_run basic "$FIX/basic/main.sio"
expect_output basic "$FIX/basic/expected.txt"
# Minimal by construction: modules are processed in load order and the exact
# "is this name still defined elsewhere?" scan sees the programs as they now are, so
# once module 1 is renamed module 2's `helper` is unique and keeps its bare name.
expect_log basic "private_fn_identity: renamed 1 same-named private fn(s) in 1 module(s)" "the rename receipt"

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
expect_log skip "private_fn_identity: left 1 fn(s) unrenamed; no collision remains" "the skipped-rename note"
if grep -Fq "error[private_fn_identity]" "$WORK/skip.log"; then
  fail "skip: refused a compile whose collision was fully resolved"
fi
echo "$TAG PASS(skip): unprovable rename skipped; no collision remains, so it compiles"

# --- genshadow: a parameter shadows the GENERATED name, not the original ----
# skip (above) catches a parameter spelled like the ORIGINAL name. This
# catches a parameter spelled like the name the pass would GENERATE
# (`helper__m1`): renaming module 1's `helper` to `helper__m1` would make its
# own call site resolve to that parameter instead, silently. Pre-fix this was
# renamed anyway and crashed during lowering instead of being refused.
compile_and_run genshadow "$FIX/genshadow/main.sio"
expect_output genshadow "$FIX/genshadow/expected.txt"
expect_log genshadow "private_fn_identity: left 1 fn(s) unrenamed; no collision remains" "the skipped-rename note"
if grep -Fq "error[private_fn_identity]" "$WORK/genshadow.log"; then
  fail "genshadow: refused a compile whose collision was fully resolved"
fi
echo "$TAG PASS(genshadow): a generated-name shadow is skipped, not renamed into a crash"

# --- exact identity, symbol reservation, capacity -----------------------------
compile_and_run hashcoll "$FIX/hashcoll/main.sio"
expect_output hashcoll "$FIX/hashcoll/expected.txt"
echo "$TAG PASS(hashcoll): ab / bA (same ast_name_hash) are not confused"

compile_and_run symcoll "$FIX/symcoll/main.sio"
expect_output symcoll "$FIX/symcoll/expected.txt"
echo "$TAG PASS(symcoll): a private fn named like a method symbol is kept apart"

compile_and_run reserved "$FIX/reserved/main.sio"
expect_output reserved "$FIX/reserved/expected.txt"
echo "$TAG PASS(reserved): a generated name never reuses an existing symbol"

compile_and_run reservedglobal "$FIX/reservedglobal/main.sio"
expect_output reservedglobal "$FIX/reservedglobal/expected.txt"
echo "$TAG PASS(reservedglobal): a generated name never reuses a module global"

# A generated-name AVAILABILITY check that trusted the hash census alone (no
# exact follow-up) falsely marked BOTH helper__m1 and helper__m2 unsafe here
# (confirmed: they collide with the unrelated helper__lR / helper__lS under
# djb2) and refused a program with no real generated-name collision at all
# (measured on the pre-fix build: compile rc=1). The fix scans exactly.
compile_and_run hashadversarial "$FIX/hashadversarial/main.sio"
expect_output hashadversarial "$FIX/hashadversarial/expected.txt"
echo "$TAG PASS(hashadversarial): two unrelated hash-colliding decoys do not block a real rename"

# `fZOXITBAFRX_E` hashes to exactly 0 under this compiler's own djb2
# ast_name_hash (confirmed directly; ast_name_hash's own clamp makes every
# real hash >= 0, so a legitimate name CAN land on 0). Before the h+1 key
# shift in pfi_census_note/pfi_census_mods, `if h == 0 { return true }` /
# `{ return 0 }` meant this exact collision was never recorded at all --
# PFI_DUP_ANY stayed unset and the original first-loaded-wins miscompile was
# intact (measured pre-fix: SIGILL, not merely a wrong printed value).
#
# This case can only pin the CENSUS half of the fix (the rename receipt
# below), not full run-to-completion like every other case here: a call to a
# function whose name hashes to 0 hits a SEPARATE, pre-existing bug in this
# compiler's dead-code-elimination reachability marker
# (self-hosted/check/specializer.sio's spec_dce_hash_insert /
# spec_dce_hash_query, the identical `h == 0` sentinel mistake in a
# different hash table), which is out of scope for private_fn_identity.sio
# and tracked separately -- see
# docs/audit/MADAROS_PRIVATE_FN_IDENTITY_2026-09-21.md.
expect_census_detected hashzero "$FIX/hashzero/main.sio" \
  "private_fn_identity: renamed 1 same-named private fn(s) in 1 module(s)" "the rename receipt"
echo "$TAG PASS(hashzero): a collision whose ast_name_hash is exactly 0 is detected and renamed by the census"

# Every other case here goes through the ORDINARY multi-module pipeline. A
# collision-bearing program that also instantiates a generic anywhere routes
# through the SEPARATE specialized-collapse pipeline instead
# (module_frontend_specialized_prepare merges every module into one item list
# before lowering) -- confirmed by the "specialized_collapse" log line below,
# so a silently-skipped collapse can't pass this by accident.
compile_and_run genericcollapse "$FIX/genericcollapse/main.sio"
expect_output genericcollapse "$FIX/genericcollapse/expected.txt"
expect_log genericcollapse "specialized_collapse" "confirmation the specialized-collapse pipeline actually ran"
compile_and_run genericcollapse_swapped "$FIX/genericcollapse/main_swapped.sio"
expect_output genericcollapse_swapped "$FIX/genericcollapse/expected.txt"
echo "$TAG PASS(genericcollapse): colliding private GENERIC helpers, both import orders, through the specialized-collapse pipeline"

compile_and_run capacity "$FIX/capacity/main.sio"
expect_output capacity "$FIX/capacity/expected.txt"
expect_log capacity "private_fn_identity: renamed 70 same-named private fn(s) in 1 module(s)" "all 70 renames (the second module is then unique)"
echo "$TAG PASS(capacity): 70 colliding private fns per module, none left behind"

# --- refusals ------------------------------------------------------------------
# expect_refusal <label> <main.sio> <diagnostic-substring> [second-substring]
# The compile must EXIT NON-ZERO, print the diagnostic, and write no ELF.
expect_refusal() {
  local label="$1" src="$2" want="$3" want2="${4:-}"
  local log="$WORK/$label.log" elf="$WORK/$label.elf"
  if "$RAW" --native-compile "$src" -o "$elf" >"$log" 2>&1; then
    tail -n 25 "$log" >&2 || true
    fail "$label: compiled, but the collision cannot be resolved (a known-wrong executable)"
  fi
  grep -Fq -- "$want" "$log" || {
    tail -n 25 "$log" >&2 || true
    fail "$label: refused, but without the expected diagnostic ($want)"
  }
  if [[ -n "$want2" ]]; then
    grep -Fq -- "$want2" "$log" || {
      tail -n 25 "$log" >&2 || true
      fail "$label: the diagnostic does not state the real reason ($want2)"
    }
  fi
  if [[ -s "$elf" ]]; then
    fail "$label: the compile was refused but an ELF was still written"
  fi
}

# Every colliding module has a parameter spelled like the fn, so none can be
# renamed and the two bodies would share one slot. The compile must STOP -- and
# say the shadowing reason, not some other one.
expect_refusal unresolved "$FIX/unresolved/main.sio" \
  'error[private_fn_identity]: fn `helper` (module #1)' "a local, parameter or pattern uses that name"
echo "$TAG PASS(unresolved): unrenamable collision refused with a diagnostic, no executable"

# pub(crate) is exported, so it cannot be renamed; two of one name are refused, and
# the diagnostic must give THAT reason (not a shadowing one).
expect_refusal restricted "$FIX/restricted/main.sio" \
  'error[private_fn_identity]: fn `helper` (module #1)' "it is exported (pub(crate)"
echo "$TAG PASS(restricted): two pub(crate) fns of one name refused, with the exported reason"

compile_and_run restrictedmix "$FIX/restrictedmix/main.sio"
expect_output restrictedmix "$FIX/restrictedmix/expected.txt"
expect_log restrictedmix "private_fn_identity: left 1 fn(s) unrenamed; no collision remains" "the skipped-rename note"
echo "$TAG PASS(restrictedmix): a private fn is renamed away from a pub(crate) one"

# --- capacity boundary: 512 accepted, 513 refused ------------------------------
# The per-module table holds 512 names (PFI_T_MAX). A regression that silently
# dropped entries again at the boundary would pass a 70-name test, so pin both
# sides of it. Generated, not committed: ~1000 lines per side.
gen_cap_case() {  # gen_cap_case <dir> <n>
  local dir="$1" n="$2" i side off
  mkdir -p "$dir"
  for side in a b; do
    off=0; [[ "$side" == b ]] && off=1000
    {
      for ((i = 0; i < n; i++)); do echo "fn f$i() -> i64 { $((i + off)) }"; done
      echo "pub fn ${side}_sum() -> i64 {"
      echo "    var t: i64 = 0"
      for ((i = 0; i < n; i++)); do echo "    t = t + f$i()"; done
      echo "    t"
      echo "}"
    } >"$dir/pfi_cb_$side.sio"
  done
  cat >"$dir/main.sio" <<'SIO'
use pfi_cb_a::{a_sum}
use pfi_cb_b::{b_sum}

fn main() -> i32 with IO, Mut, Panic {
    print("a=")
    print_int(a_sum())
    print(" b=")
    print_int(b_sum())
    println("")
    0
}
SIO
}

gen_cap_case "$WORK/cap512" 512
sa=$((512 * 511 / 2)); sb=$((sa + 1000 * 512))
printf 'a=%d b=%d\n' "$sa" "$sb" >"$WORK/cap512.expected"
compile_and_run cap512 "$WORK/cap512/main.sio"
expect_output cap512 "$WORK/cap512.expected"
expect_log cap512 "private_fn_identity: renamed 512 same-named private fn(s) in 1 module(s)" "all 512 renames"
echo "$TAG PASS(cap512): exactly 512 colliding names accepted and all renamed"

gen_cap_case "$WORK/cap513" 513
expect_refusal cap513 "$WORK/cap513/main.sio" \
  'error[private_fn_identity]: more than 512 same-named private fns in one module'
echo "$TAG PASS(cap513): 513 colliding names refused, no executable"

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
