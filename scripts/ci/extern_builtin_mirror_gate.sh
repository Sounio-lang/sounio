#!/usr/bin/env bash
# Keep the checker's extern allow-list identical to the native backend's.
#
# #1622: an `extern "C"` declaration reaches codegen as an instr_count == 0 stub
# (ir/lower.sio flushes it that way). native_v2_builtin_id_for_func_ref in
# self-hosted/native/codegen_x86_linux.sio maps such a stub to a builtin id
# 1..27 and returns 0 -- "emit nothing" -- for every other name, so a call to an
# unlisted extern reads whatever is in rax. check/check.sio refuses that call at
# type-check time (E250) using name_is_native_backend_builtin.
#
# Two lists, one truth. If the backend gains a builtin and the checker does not
# hear about it, the checker rejects a name that now works -- a false refusal,
# strictly worse than the bug E250 was written to kill. If the backend LOSES one,
# the checker accepts a call that silently returns 0 again. This gate diffs them.
#
# Track A added a second route to the same backend. For twelve names the parser
# (self-hosted/parser/items.sio, extern_name_has_ffi_intrinsic) rewrites the
# declaration into a wrapper forwarding to an ffi_<name> intrinsic, ids 39..50.
# That makes four lists that must agree, and each disagreement has its own
# failure:
#
#   backend ffi_* arms  vs  parser rewrite list
#       a wrapper forwarding to an intrinsic nobody emits calls into nothing;
#       and, the reason the third list exists at all, wrapping a name with NO
#       intrinsic moves the refusal from the call to the declaration (E137 on
#       the unbound ffi_<name>), so declaring a binding without calling it
#       stops compiling -- see tests/run-pass/ffi_declared_never_called_is_legal.sio
#   backend ffi_* arms  vs  checker ffi_* bindings
#       an unbound ffi_<name> makes every wrapper body fail to type-check
#   backend plain arms + parser rewrite list  vs  name_is_native_backend_builtin
#       the original two-list check, widened: a name is callable if the backend
#       answers it by its own name OR through its wrapper. floor, ceil, pow and
#       tgamma are callable only through the wrapper and have no plain arm.
#
# Usage: scripts/ci/extern_builtin_mirror_gate.sh

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 1

CODEGEN="self-hosted/native/codegen_x86_linux.sio"
CHECK="self-hosted/check/check.sio"
PARSER="self-hosted/parser/items.sio"

fail() { echo "[extern-mirror] FAIL: $*" >&2; exit 1; }

[ -f "$CODEGEN" ] || fail "missing $CODEGEN"
[ -f "$CHECK" ]   || fail "missing $CHECK"
[ -f "$PARSER" ]  || fail "missing $PARSER"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/extern-mirror.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# --- backend side ----------------------------------------------------------
# The body of native_v2_builtin_id_for_func_ref, which is the authority. Each
# arm is `if <pred>((*func).name) { return <id> }` or a name_ref_is_* variant;
# the predicate name carries the builtin name after the is_ prefix.
awk '
    /^fn native_v2_builtin_id_for_func_ref/ { inside = 1; next }
    inside && /^}/                         { inside = 0 }
    inside                                 { print }
' "$CODEGEN" \
  | grep -oE '(name_is|name_ref_is)_[a-z_0-9]+' \
  | sed -E 's/^name_ref_is_//; s/^name_is_//' \
  | sort -u > "$WORK/backend_raw.txt"

# Drop predicates that match a name no source file can spell.
#
# The convention above -- "the predicate name carries the builtin name after
# the is_ prefix" -- holds only while the builtin name is a valid identifier.
# An arity-retagged builtin like `str_slice.3` is written into the IR by
# ir_module_ensure_builtin_call_targets and never appears in source; the dot is
# deliberate, so that no user identifier can collide with it. Its predicate is
# therefore spelled name_is_str_slice3, and the extraction yields `str_slice3`
# -- a name that is in neither list and belongs in neither.
#
# The mirror exists so the CHECKER admits an `extern "C"` the backend
# implements. A name containing a dot cannot be an extern declaration, so it
# can never need mirroring, and adding it would let the checker admit an extern
# literally named `str_slice3` that the backend does not have.
#
# Detected from the predicate's own body -- it compares a byte against 46, the
# dot -- rather than from its comment, so a renamed helper or a reworded
# comment does not change the answer.
: > "$WORK/backend_all.txt"
while read -r _n; do
  _pred=""
  for _cand in "name_is_$_n" "name_ref_is_$_n"; do
    grep -q "^fn $_cand(" "$CODEGEN" && _pred="$_cand"
  done
  if [[ -n "$_pred" ]] \
     && awk -v f="fn $_pred(" 'index($0,f)==1{i=1;next} i&&/^}/{exit} i' "$CODEGEN" \
        | grep -qE '\b46 as i8\b'; then
    echo "[extern-mirror] internal-only builtin (dotted name, not source-spellable): $_n" >&2
    continue
  fi
  printf '%s\n' "$_n"
done < "$WORK/backend_raw.txt" | sort -u > "$WORK/backend_all.txt"

# Two kinds of arm live in that registry and they answer different questions.
#
#   plain arms (sqrt, exp, getpid, ...) fire on the DECLARED name, because an
#   `extern "C"` declaration reaches codegen as an instr_count == 0 stub.
#   ffi_* arms (ffi_sqrt .. ffi_tgamma, ids 39..50) fire on the name inside a
#   forwarding wrapper the PARSER builds for the declaration instead.
#
# So the checker's allow-list must equal the union, not the plain half: floor,
# ceil, pow and tgamma are callable purely through their wrappers and have no
# plain arm at all.
grep    -E '^ffi_' "$WORK/backend_all.txt" | sed -E 's/^ffi_//' | sort -u > "$WORK/backend_ffi.txt"
grep -v -E '^ffi_' "$WORK/backend_all.txt"                      | sort -u > "$WORK/backend_plain.txt"

# --- parser side -----------------------------------------------------------
# extern_name_has_ffi_intrinsic decides which declarations get a forwarding
# body. A name it wraps without a backend arm compiles to a call into nothing;
# a name it does NOT wrap while the checker binds ffi_<name> is a dead binding.
# Worse, and the reason this arm exists: wrapping a name with no intrinsic used
# to move the refusal from the call to the DECLARATION (E137 on the unbound
# ffi_<name>), so declaring a binding without calling it stopped compiling.
awk '
    /^fn extern_name_has_ffi_intrinsic/ { inside = 1; next }
    inside && /^}/                      { inside = 0 }
    inside                              { print }
' "$PARSER" \
  | grep -oE 'make_name\("[a-z_0-9]+"\)' \
  | sed -E 's/^make_name\("//; s/"\)$//' \
  | sort -u > "$WORK/parser_ffi.txt"

# --- checker ffi binding side ----------------------------------------------
# The wrapper body only resolves if checker_collect_runtime_builtins_inplace
# binds ffi_<name>.
awk '
    /^fn checker_collect_runtime_builtins_inplace/ { inside = 1; next }
    inside && /^}/                                 { inside = 0 }
    inside                                         { print }
' "$CHECK" \
  | grep -oE 'make_name\("ffi_[a-z_0-9]+"\)' \
  | sed -E 's/^make_name\("ffi_//; s/"\)$//' \
  | sort -u > "$WORK/checker_ffi.txt"

# --- checker side ----------------------------------------------------------
awk '
    /^fn name_is_native_backend_builtin/ { inside = 1; next }
    inside && /^}/                       { inside = 0 }
    inside                               { print }
' "$CHECK" \
  | grep -oE 'make_name\("[a-z_0-9]+"\)' \
  | sed -E 's/^make_name\("//; s/"\)$//' \
  | sort -u > "$WORK/checker.txt"

sort -u "$WORK/backend_plain.txt" "$WORK/parser_ffi.txt" > "$WORK/backend.txt"

BACKEND_N=$(wc -l < "$WORK/backend.txt")
CHECKER_N=$(wc -l < "$WORK/checker.txt")
BACKEND_FFI_N=$(wc -l < "$WORK/backend_ffi.txt")
PARSER_FFI_N=$(wc -l < "$WORK/parser_ffi.txt")
CHECKER_FFI_N=$(wc -l < "$WORK/checker_ffi.txt")

# An empty extraction means the source moved and the gate is measuring nothing.
# That must fail loudly, not pass vacuously -- the exact class this repository
# keeps rediscovering.
[ "$BACKEND_N" -ge 20 ] || fail \
  "extracted only $BACKEND_N names from native_v2_builtin_id_for_func_ref in $CODEGEN.
  The function was renamed or restructured; this gate is no longer measuring it."
[ "$CHECKER_N" -ge 20 ] || fail \
  "extracted only $CHECKER_N names from name_is_native_backend_builtin in $CHECK.
  The function was renamed or restructured; this gate is no longer measuring it."
[ "$BACKEND_FFI_N" -ge 8 ] || fail \
  "extracted only $BACKEND_FFI_N ffi_* arms from native_v2_builtin_id_for_func_ref in $CODEGEN.
  The registry was renamed or restructured; this gate is no longer measuring it."
[ "$PARSER_FFI_N" -ge 8 ] || fail \
  "extracted only $PARSER_FFI_N names from extern_name_has_ffi_intrinsic in $PARSER.
  The predicate was renamed or restructured; this gate is no longer measuring it."
[ "$CHECKER_FFI_N" -ge 8 ] || fail \
  "extracted only $CHECKER_FFI_N ffi_* bindings from checker_collect_runtime_builtins_inplace in $CHECK.
  The binding block was renamed or restructured; this gate is no longer measuring it."

if ! diff -u "$WORK/backend_ffi.txt" "$WORK/parser_ffi.txt" > "$WORK/diff_parser.txt"; then
    echo "[extern-mirror] backend ffi_* arms ($BACKEND_FFI_N) vs parser rewrite list ($PARSER_FFI_N):" >&2
    sed -n '3,$p' "$WORK/diff_parser.txt" >&2
    echo >&2
    echo "  -name  the backend has ffi_$name; the parser never builds the wrapper that reaches it" >&2
    echo "  +name  the parser forwards to ffi_$name; nothing emits it, so the call goes nowhere" >&2
    fail "the parser rewrite list and the backend ffi_* registry have drifted.
  Authority: native_v2_builtin_id_for_func_ref ($CODEGEN)
  Mirror:    extern_name_has_ffi_intrinsic ($PARSER)"
fi

if ! diff -u "$WORK/backend_ffi.txt" "$WORK/checker_ffi.txt" > "$WORK/diff_bind.txt"; then
    echo "[extern-mirror] backend ffi_* arms ($BACKEND_FFI_N) vs checker ffi_* bindings ($CHECKER_FFI_N):" >&2
    sed -n '3,$p' "$WORK/diff_bind.txt" >&2
    echo >&2
    echo "  -name  the backend emits ffi_$name; the checker leaves it unbound, so the wrapper body E137s" >&2
    echo "  +name  the checker binds ffi_$name; no backend arm emits it" >&2
    fail "the checker ffi_* bindings and the backend ffi_* registry have drifted.
  Authority: native_v2_builtin_id_for_func_ref ($CODEGEN)
  Mirror:    checker_collect_runtime_builtins_inplace ($CHECK)"
fi

if ! diff -u "$WORK/backend.txt" "$WORK/checker.txt" > "$WORK/diff.txt"; then
    echo "[extern-mirror] backend ($BACKEND_N names) vs checker ($CHECKER_N names):" >&2
    sed -n '3,$p' "$WORK/diff.txt" >&2
    echo >&2
    echo "  -name  the backend implements it; the checker would reject a working call" >&2
    echo "  +name  the checker allows it; calls compile to an empty stub reading 0" >&2
    fail "the two builtin lists have drifted.
  Authority: native_v2_builtin_id_for_func_ref plain arms + extern_name_has_ffi_intrinsic
             ($CODEGEN, $PARSER)
  Mirror:    name_is_native_backend_builtin ($CHECK)"
fi

echo "[extern-mirror] PASS: $BACKEND_N callable builtin names ($BACKEND_FFI_N via ffi_* wrappers), checker, parser and backend agree"
