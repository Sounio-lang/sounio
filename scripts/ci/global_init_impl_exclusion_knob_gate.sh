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

echo "$TAG PASS"
