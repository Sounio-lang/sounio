#!/usr/bin/env bash
# field_name_hash_refused_gate.sh — ADR-026 amendment (2026-11-09), RFC-0001 Stage 0
#
# A field is resolved by the type that owns it, never by a hash of its name.
# The old `(name.buf[0] as i64) % 64` fallback in self-hosted/ir/lower.sio
# invented slot 51 for `.span` and slot 37 for `epsilon` (#1499). This gate
# measures SOURCE: no live return of that hash may remain, and the field-index
# lookups must reach `lower_refuse_untyped_field_hash`, which raises hard-error
# reason 10. Behaviour of a binary needs a from-source Madaros build.
#
# Negative control: the same check run on a copy of lower.sio with one hash
# return restored must fail, so a check that cannot see the hash is red.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
. "$ROOT/scripts/lib/gate_assert.sh"
gate_name "field_name_hash_refused_gate"

LOWER="$ROOT/self-hosted/ir/lower.sio"
[[ -f "$LOWER" ]] || gate_fail "missing $LOWER"

# A live hash return: `% 64` applied to the first byte of a name, on a line
# that is not a comment.
has_live_hash() {
  grep -E '\.buf\[0\] as i64\) % 64' "$1" | grep -vqE '^[[:space:]]*//'
}

check_source() {
  local f="$1"
  if has_live_hash "$f"; then
    return 1
  fi
  grep -q '^fn lower_refuse_untyped_field_hash(field_name: Name) -> i64' "$f" || return 2
  grep -q 'IR_LOWER_HARD_ERROR_REASON = 10' "$f" || return 3
  grep -q 'IR_LOWER_HARD_ERROR_REASON == 10' "$f" || return 4
  grep -q 'return lower_refuse_untyped_field_hash(field_name)' "$f" || return 5
  grep -q 'lower_refuse_untyped_field_hash(n)' "$f" || return 6
  return 0
}

set +e
check_source "$LOWER"
rc=$?
set -e
case "$rc" in
  0) ;;
  1) gate_fail "a live name[0] % 64 field-index hash is back in lower.sio" ;;
  2) gate_fail "lower_refuse_untyped_field_hash is missing" ;;
  3) gate_fail "the refusal does not set hard-error reason 10" ;;
  4) gate_fail "hard-error reason 10 has no diagnostic in lower_hard_error_report" ;;
  5) gate_fail "field_idx_from_name does not reach the refusal" ;;
  6) gate_fail "field_idx_from_name_simple does not reach the refusal" ;;
  *) gate_fail "unexpected check status $rc" ;;
esac

TMP="$(mktemp -d "${TMPDIR:-/tmp}/field-hash-gate.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
sed 's/            lower_refuse_untyped_field_hash(n)/            (n.buf[0] as i64) % 64/' "$LOWER" > "$TMP/lower.sio"
set +e
check_source "$TMP/lower.sio"
neg=$?
set -e
[[ "$neg" -ne 0 ]] || gate_fail "negative control: a restored hash return was not detected"

gate_pass "no name-hash field index in lower.sio; refusal reason 10 wired (negative control red: status $neg)"
echo "measured=yes"
exit 0
