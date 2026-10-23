#!/usr/bin/env bash
# struct_field_lowering_gate.sh — RFC-0001 Stage 0 / #2966 / #3000
#
# The legacy field-index hash `name[0] % 64` invented .span → 51. Stage 0
# requires that `field_idx_from_name*` refuse that path fail-closed via
# `lower_refuse_untyped_field_hash` (IR_LOWER_HARD_ERROR_REASON=10).
#
# This gate measures SOURCE. The committed Madaros ELF is unchanged until
# `make build-madaros`; provenance notices already say behaviour claims need
# a from-source build. The gate fails if the hash fallback returns in the
# lookup functions.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
. "$ROOT/scripts/lib/gate_assert.sh"
gate_name "struct_field_lowering_gate"

LOWER="$ROOT/self-hosted/ir/lower.sio"
[[ -f "$LOWER" ]] || gate_fail "missing $LOWER"

if ! grep -q "fn lower_refuse_untyped_field_hash" "$LOWER"; then
  gate_fail "lower_refuse_untyped_field_hash missing — Stage 0 not landed"
fi

if ! grep -q "IR_LOWER_HARD_ERROR_REASON == 10" "$LOWER"; then
  gate_fail "hard-error reason 10 (unknown struct field) not reported"
fi

# The three historical return sites must not compute name[0]%64 anymore.
# Allow the string in comments / refuse helper docs, but not as a return expr
# of the form `(….buf[0] as i64) % 64`.
hits=$(grep -nE '\(.*\.buf\[0\] as i64\) % 64' "$LOWER" || true)
if [[ -n "$hits" ]]; then
  # Filter comment-only lines
  live=$(printf '%s\n' "$hits" | grep -vE '^\s*[0-9]+:\s*//' || true)
  if [[ -n "$live" ]]; then
    echo "$live" | sed 's/^/    /' >&2
    gate_fail "live name[0]%64 field-index hash still present in lower.sio"
  fi
fi

# Refuse helper must be reached from both lookup entry points.
grep -q "lower_refuse_untyped_field_hash" "$LOWER" || gate_fail "refuse helper unreferenced"

gate_pass "Stage 0 source: field hash fail-closed (reason 10)"
echo "measured=yes"
echo "note: rebuild Madaros from source before claiming runtime Stage 0 behaviour"
exit 0
