#!/usr/bin/env bash
# Keep the checker's extern allow-list identical to the native backend's.
#
# #1622: an `extern "C"` declaration reaches codegen as an instr_count == 0 stub
# (ir/lower.sio flushes it that way). native_v2_builtin_id_for_func_ref in
# self-hosted/native/codegen_x86_linux.sio maps such a stub to a builtin id
# 1..27 and returns 0 -- "emit nothing" -- for every other name, so a call to an
# unlisted extern reads whatever is in rax. check/check.sio refuses that call at
# type-check time (E219) using name_is_native_backend_builtin.
#
# Two lists, one truth. If the backend gains a builtin and the checker does not
# hear about it, the checker rejects a name that now works -- a false refusal,
# strictly worse than the bug E219 was written to kill. If the backend LOSES one,
# the checker accepts a call that silently returns 0 again. This gate diffs them.
#
# Usage: scripts/ci/extern_builtin_mirror_gate.sh

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 1

CODEGEN="self-hosted/native/codegen_x86_linux.sio"
CHECK="self-hosted/check/check.sio"

fail() { echo "[extern-mirror] FAIL: $*" >&2; exit 1; }

[ -f "$CODEGEN" ] || fail "missing $CODEGEN"
[ -f "$CHECK" ]   || fail "missing $CHECK"

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
  | sort -u > "$WORK/backend.txt"

# --- checker side ----------------------------------------------------------
awk '
    /^fn name_is_native_backend_builtin/ { inside = 1; next }
    inside && /^}/                       { inside = 0 }
    inside                               { print }
' "$CHECK" \
  | grep -oE 'make_name\("[a-z_0-9]+"\)' \
  | sed -E 's/^make_name\("//; s/"\)$//' \
  | sort -u > "$WORK/checker.txt"

BACKEND_N=$(wc -l < "$WORK/backend.txt")
CHECKER_N=$(wc -l < "$WORK/checker.txt")

# An empty extraction means the source moved and the gate is measuring nothing.
# That must fail loudly, not pass vacuously -- the exact class this repository
# keeps rediscovering.
[ "$BACKEND_N" -ge 20 ] || fail \
  "extracted only $BACKEND_N names from native_v2_builtin_id_for_func_ref in $CODEGEN.
  The function was renamed or restructured; this gate is no longer measuring it."
[ "$CHECKER_N" -ge 20 ] || fail \
  "extracted only $CHECKER_N names from name_is_native_backend_builtin in $CHECK.
  The function was renamed or restructured; this gate is no longer measuring it."

if ! diff -u "$WORK/backend.txt" "$WORK/checker.txt" > "$WORK/diff.txt"; then
    echo "[extern-mirror] backend ($BACKEND_N names) vs checker ($CHECKER_N names):" >&2
    sed -n '3,$p' "$WORK/diff.txt" >&2
    echo >&2
    echo "  -name  the backend implements it; the checker would reject a working call" >&2
    echo "  +name  the checker allows it; calls compile to an empty stub reading 0" >&2
    fail "the two builtin lists have drifted.
  Authority: native_v2_builtin_id_for_func_ref ($CODEGEN)
  Mirror:    name_is_native_backend_builtin ($CHECK)"
fi

echo "[extern-mirror] PASS: $BACKEND_N builtin names, checker and backend agree"
