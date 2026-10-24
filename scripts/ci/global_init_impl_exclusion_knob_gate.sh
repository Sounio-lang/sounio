#!/usr/bin/env bash
# #2661: SOUNIO_DISABLE_GLOBAL_INIT_IMPL_EXCLUSION=1 is the attribution knob for
# the second #2603 delta: impl methods are not recorded under their bare name
# in the parser's pure-fn constant-fold table (self-hosted/parser/items.sio,
# items_maybe_record_pure_fn_const). The knob restores ONLY that recording, in
# single- and multi-module compiles alike, independently of
# SOUNIO_DISABLE_GLOBAL_INIT_MODULE_SCOPE (the module-scoping knob).
#
# Witness: `impl S { fn helper() -> i64 { 7 } }` parsed before a free
# `fn helper() -> i64 { 42 }`, and `var G: i64 = helper()`. With the exclusion
# (default) the free fn is the one recorded and G = 42. With the knob the
# method occupies the bare name first and G folds to 7.
# Multi-module: defs.sio's `pub` impl method `helper` (7) and main.sio's own
# free `helper` (42). Measured 42 under all four settings: in this shape
# neither knob lets another module's method reach the importer's global
# initialiser, and the gate pins that so a change to either knob that leaks it
# is caught.
# Control: a program with no method/free-fn name collision prints the same
# value under every knob setting.
#
# Values measured with a Madaros built from this branch (#2661).
#
# The dump cases at the end show the knob flipping the recorded owner of
# `helper` in the table itself (SOUNIO_DUMP_GLOBAL_INIT_TABLE=1).
#
# Engine: Madaros (bin/souc's default). The table is a Madaros parser
# structure; lean_single has no equivalent.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-gvi-impl-knob.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
ulimit -s unlimited 2>/dev/null || true
TAG="[gvi-impl-knob]"

fail() { echo "$TAG FAIL: $*" >&2; exit 1; }

cat >"$WORK/single.sio" <<'EOF'
struct S { v: i64 }
impl S {
    fn helper() -> i64 { 7 }
}
fn helper() -> i64 { 42 }
var G: i64 = helper()
fn main() -> i32 with IO {
    print_int(G)
    print("\n")
    return 0
}
EOF

mkdir -p "$WORK/multi"
cat >"$WORK/multi/defs.sio" <<'EOF'
pub struct S { v: i64 }
impl S {
    pub fn helper() -> i64 { 7 }
}
pub fn other() -> i64 { 3 }
EOF
cat >"$WORK/multi/main.sio" <<'EOF'
use defs::{S, other}
fn helper() -> i64 { 42 }
var G: i64 = helper()
fn main() -> i32 with IO {
    print_int(G + other() - 3)
    print("\n")
    return 0
}
EOF

cat >"$WORK/control.sio" <<'EOF'
struct S { v: i64 }
impl S {
    fn method_only() -> i64 { 7 }
}
fn helper() -> i64 { 42 }
var G: i64 = helper()
fn main() -> i32 with IO {
    print_int(G)
    print("\n")
    return 0
}
EOF

run_case() {
  local label="$1" src="$2" scope_off="$3" excl_off="$4" want="$5" out rc
  set +e
  out="$(cd "$(dirname "$src")" && env -u SOUNIO_DISABLE_GLOBAL_INIT_MODULE_SCOPE -u SOUNIO_DISABLE_GLOBAL_INIT_IMPL_EXCLUSION \
        ${scope_off:+SOUNIO_DISABLE_GLOBAL_INIT_MODULE_SCOPE=1} ${excl_off:+SOUNIO_DISABLE_GLOBAL_INIT_IMPL_EXCLUSION=1} \
        timeout 900 "$ROOT_DIR/bin/souc" run "$src" 2>"$WORK/$label.log" | tail -n 1)"
  rc=$?
  set -e
  if [[ $rc -ne 0 || "$out" != "$want" ]]; then
    tail -n 20 "$WORK/$label.log" >&2 || true
    fail "$label: rc=$rc out='$out' want='$want'"
  fi
  echo "$TAG PASS $label -> $out"
}

# Single module: the module-scope knob is inert (serial 0 either way); the
# impl-exclusion knob decides.
run_case single.scope_on.excl_on   "$WORK/single.sio" ""  ""  42
run_case single.scope_off.excl_on  "$WORK/single.sio" "1" ""  42
run_case single.scope_on.excl_off  "$WORK/single.sio" ""  "1" 7
run_case single.scope_off.excl_off "$WORK/single.sio" "1" "1" 7

# Multi module, the full 2x2 (no setting leaks the other module's method).
run_case multi.scope_on.excl_on    "$WORK/multi/main.sio" ""  ""  42
run_case multi.scope_off.excl_on   "$WORK/multi/main.sio" "1" ""  42
run_case multi.scope_on.excl_off   "$WORK/multi/main.sio" ""  "1" 42
run_case multi.scope_off.excl_off  "$WORK/multi/main.sio" "1" "1" 42

# Control: no method shares the free fn's name, so no knob moves the value.
run_case control.excl_on           "$WORK/control.sio" ""  ""  42
run_case control.excl_off          "$WORK/control.sio" ""  "1" 42
run_case control.both_off          "$WORK/control.sio" "1" "1" 42

# #2661 attribution dump: SOUNIO_DUMP_GLOBAL_INIT_TABLE=1 prints every word the
# parser records in the constant-fold table, with its name and the kind of
# definition that recorded it (parser/ast.sio ast_gvi_dump_note). The knob must
# flip WHICH definition owns the bare name `helper`: the free fn (42) by
# default, the impl method (7) with SOUNIO_DISABLE_GLOBAL_INIT_IMPL_EXCLUSION=1,
# and G follows. The dump is printed by the compiler, so it is read from the
# combined compile output, not from the program's stdout.
dump_case() {
  local label="$1" excl_off="$2" rc
  shift 2
  set +e
  (cd "$WORK" && env -u SOUNIO_DISABLE_GLOBAL_INIT_MODULE_SCOPE -u SOUNIO_DISABLE_GLOBAL_INIT_IMPL_EXCLUSION \
      SOUNIO_DUMP_GLOBAL_INIT_TABLE=1 ${excl_off:+SOUNIO_DISABLE_GLOBAL_INIT_IMPL_EXCLUSION=1} \
      timeout 900 "$ROOT_DIR/bin/souc" run "$WORK/single.sio") >"$WORK/$label.dump" 2>&1
  rc=$?
  set -e
  [[ $rc -eq 0 ]] || { tail -n 20 "$WORK/$label.dump" >&2; fail "$label: rc=$rc"; }
  local want
  for want in "$@"; do
    case "$want" in
      !*) if grep -qF -- "${want#!}" "$WORK/$label.dump"; then
            grep -F '[gvi-dump]' "$WORK/$label.dump" >&2 || true
            fail "$label: dump has '${want#!}'"
          fi ;;
      *)  if ! grep -qF -- "$want" "$WORK/$label.dump"; then
            grep -F '[gvi-dump]' "$WORK/$label.dump" >&2 || true
            fail "$label: dump lacks '$want'"
          fi ;;
    esac
  done
  echo "$TAG PASS $label -> $(grep -F 'name=helper' "$WORK/$label.dump" | sed 's/.*origin=//' | tr '\n' ' ')"
}

dump_case dump.excl_on  "" \
  "origin=fn name=helper word=42" "origin=global name=G word=42" "!origin=impl-method"
dump_case dump.excl_off "1" \
  "origin=impl-method name=helper word=7" "origin=global name=G word=7" "!origin=fn name=helper"

# Off by default: no dump line without the variable (combined output).
set +e
(cd "$WORK" && env -u SOUNIO_DUMP_GLOBAL_INIT_TABLE timeout 900 "$ROOT_DIR/bin/souc" run "$WORK/single.sio") >"$WORK/nodump.out" 2>&1
nodump_rc=$?
set -e
[[ $nodump_rc -eq 0 ]] || { tail -n 20 "$WORK/nodump.out" >&2; fail "nodump: rc=$nodump_rc"; }
if grep -qF '[gvi-dump]' "$WORK/nodump.out"; then
  fail "dump printed without SOUNIO_DUMP_GLOBAL_INIT_TABLE"
fi
echo "$TAG PASS nodump -> no [gvi-dump] line"

echo "$TAG PASS"
