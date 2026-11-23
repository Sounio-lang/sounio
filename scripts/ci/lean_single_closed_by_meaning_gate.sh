#!/usr/bin/env bash
# lean_single.sio must mean only what its text says.
#
# Two checks, both of which failed before the fix that introduced this gate:
#
# 1. Madaros accepts the source of the bootstrap compiler. Madaros enforces block
#    scope (E137) and branch typing (E007); lean_single.sio used to read five
#    variables outside the block that declared them (dovf/hsig in lex_all,
#    rhs_ty/rhs_ty_hash in compile_stmt, q in epistemic_infer_pass), which the
#    seed lowers to reads of whatever the stack held, and had 73 statement `if`s
#    whose branches disagreed in type.
# 2. The committed seed no longer acts on such a stale slot: an assignment to a
#    global after an f64 local assignment used to draw a spurious
#    "assignment type mismatch" (tests/compiler/lean_single_closed_by_meaning/).
#
#   scripts/ci/lean_single_closed_by_meaning_gate.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEED="$ROOT/bin/souc-lean-single-x86_64"
MADAROS="${SOUNIO_MADAROS_BIN:-$ROOT/bin/madaros-linux-x86_64}"
SRC="$ROOT/self-hosted/compiler/lean_single.sio"
FIX="$ROOT/tests/compiler/lean_single_closed_by_meaning/global_assign_after_f64_local.sio"
W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT

fail() { echo "[lean-single-closed-by-meaning] FAIL: $*" >&2; exit 1; }

[[ -x "$SEED" ]] || fail "missing seed $SEED"
[[ -x "$MADAROS" ]] || fail "missing Madaros $MADAROS"
ulimit -s unlimited 2>/dev/null || true

# 1. Madaros check of the bootstrap compiler's source.
if ! "$MADAROS" check "$SRC" > "$W/madaros.log" 2>&1; then
  grep -E '^error\[' "$W/madaros.log" | sort | uniq -c | sort -rn | head -20 >&2 || true
  fail "Madaros refuses self-hosted/compiler/lean_single.sio"
fi
grep -q '^check: OK' "$W/madaros.log" || fail "Madaros check did not report OK"

# 2. The stale-slot fixture: compiles without the spurious warning, prints 5.
"$SEED" "$FIX" "$W/fix.elf" > "$W/fix.log" 2>&1 || { cat "$W/fix.log" >&2; fail "seed refused the fixture"; }
if grep -q 'assignment type mismatch' "$W/fix.log"; then
  cat "$W/fix.log" >&2
  fail "seed reports a type mismatch on a well-typed global assignment (stale rhs_ty)"
fi
chmod +x "$W/fix.elf"
out="$("$W/fix.elf")"
[[ "$out" == "5" ]] || fail "fixture printed '$out', expected 5"

echo "LEAN_SINGLE_CLOSED_BY_MEANING_OK"
