#!/usr/bin/env bash
# Two checker spines, one language. This gate refuses to let them drift apart
# in silence.
#
# WHY THIS EXISTS. self-hosted/check/check.sio carries the type checker twice:
# a by-value spine of `impl Checker` methods taking `self`, and a `*mut
# Checker` transcription that exists so the modular compiler (Madaros, the
# default engine) does not copy a 172-field Checker on every call. Madaros
# runs the transcription.
#
# When an arm exists in one spine and not the other, THE LANGUAGE FEATURE
# BEHIND IT GOES INERT ON THE DEFAULT ENGINE, silently: no error, no warning,
# no diagnostic. The declaration parses, the collect pass skips it, every
# annotation naming it falls through to an opaque named type, and the feature
# simply does not happen. Found four times so far:
#
#   2026-09-01  ItemAlgebra missing from the *mut collect. Every
#               `algebra ... { }` declaration was inert; the registry was
#               never populated. Found by instrumenting a source build.
#   2026-09-09  ItemUnit missing from the *mut collect, AND the units branch
#               missing from checker_lower_named_type_mut. Units of measure
#               were non-functional on the default engine: 11 of the 14 unit
#               fixtures in tests/run-pass failed there while all 14 passed
#               on lean_single. Found the same way.
#   2026-09-09  checker_finish_binary_units_inplace did not transcribe its
#               by-value twin at all -- it dereferenced the Checker and called
#               it. That line had never executed, because the guard above it
#               returned early while unit_id was always -1. Wiring units up
#               reached it for the first time and segfaulted the compiler.
#   2026-09-09  The generic identity gate in the *mut binary path exempted
#               hypercomplex and IndepKnowledge operands from a
#               types_compatible check but not unit-branded ones, so
#               `distance / time` was rejected before any unit logic ran.
#
# Four instances, one shape. A fifth will happen. This makes it fail here
# instead of in someone's model.
#
# WHAT IT CHECKS
#   A. ItemKind arm parity, ratcheted by name against checker_spine_parity.frozen.
#   B. No NEW by-value Checker bridge inside a *mut function, ratcheted
#      against checker_spine_bridges.frozen.
#
# NO PYTHON. Extraction is awk (scripts/ci/checker_spine_extract.awk). This
# tree's rule is that Python is a binding or a marshaller and never
# computation of our own; a gate's analysis is computation of our own.
#
# ON EMPTINESS. Both extractions are floored, so a refactor that renames what
# this gate reads FAILS rather than reporting a clean zero -- the failure mode
# scripts/lib/gate_assert.sh exists to prevent.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CHECK="$ROOT/self-hosted/check/check.sio"
EXTRACT_AWK="$ROOT/scripts/ci/checker_spine_extract.awk"
LEDGER="$ROOT/scripts/ci/checker_spine_parity.frozen"
BRIDGE_LEDGER="$ROOT/scripts/ci/checker_spine_bridges.frozen"
cd "$ROOT"
. "$ROOT/scripts/lib/gate_assert.sh"
gate_name "checker_spine_parity"

require_tool awk
require_file "$CHECK"
require_file "$EXTRACT_AWK"
require_file "$LEDGER"
require_file "$BRIDGE_LEDGER"

RAW=$(awk -f "$EXTRACT_AWK" "$CHECK")
require_nonempty "$RAW" "the awk extraction produced nothing at all -- it broke, this is not a measured zero"

BYVAL=$(grep '^BYVAL ' <<<"$RAW" | awk '{print $2}' | sort -u)
MUT=$(grep '^MUT ' <<<"$RAW" | awk '{print $2}' | sort -u)
RETCK=$(grep '^RETCK ' <<<"$RAW" | awk '{print $2}' | sort -u)

require_nonempty "$BYVAL" "no ItemKind arms found in the by-value collect_item -- it was renamed or reshaped"
require_nonempty "$MUT"   "no ItemKind arms found in checker_collect_item_inplace -- same"
require_nonempty "$RETCK" "no by-value Checker-returning methods found -- the signature scan broke"

BYVAL_N=$(grep -c . <<<"$BYVAL")
MUT_N=$(grep -c . <<<"$MUT")

# Anti-vacuity floors. Not the real assertion: these exist so a broken
# extraction cannot present itself as parity.
[[ "$BYVAL_N" -ge 20 ]] || gate_fail "by-value collect_item yielded only $BYVAL_N ItemKind arms (floor 20). The extraction is broken, not the code."
[[ "$MUT_N"   -ge 5  ]] || gate_fail "*mut collect yielded only $MUT_N ItemKind arms (floor 5). The extraction is broken, not the code."

echo "checker_spine_parity: by-value handles $BYVAL_N ItemKinds, *mut handles $MUT_N"

# --- A: ItemKind divergence, ratcheted by name ---
LIVE_A=$(comm -23 <(echo "$BYVAL") <(echo "$MUT"))
FROZEN_A=$(grep -vE '^[[:space:]]*(#|$)' "$LEDGER" | awk '{print $1}' | sort -u)
require_nonempty "$FROZEN_A" "the frozen ItemKind ledger has no entries -- it is unreadable, not empty"

NEW_A=$(comm -23 <(echo "$LIVE_A") <(echo "$FROZEN_A") || true)
GONE_A=$(comm -13 <(echo "$LIVE_A") <(echo "$FROZEN_A") || true)

if [[ -n "${NEW_A//[[:space:]]/}" ]]; then
  {
    echo "FAIL: declaration kinds handled by the by-value collect pass and NOT by the"
    echo "      *mut one, and not recorded in $LEDGER:"
    while read -r k; do [[ -n "$k" ]] && echo "        $k"; done <<<"$NEW_A"
    echo ""
    echo "      Each is a language feature that is INERT on the default engine."
    echo "      Wire it into checker_collect_item_inplace, or add it to the ledger"
    echo "      with a one-line reason saying where it is handled instead."
  } >&2
  exit 1
fi
if [[ -n "${GONE_A//[[:space:]]/}" ]]; then
  echo "NOTE: these ledger entries are no longer divergent -- remove them from $LEDGER:"
  while read -r k; do [[ -n "$k" ]] && echo "        $k"; done <<<"$GONE_A"
fi
echo "checker_spine_parity: divergent ItemKinds live=$(grep -c . <<<"$LIVE_A") frozen=$(grep -c . <<<"$FROZEN_A")"

# --- B: by-value Checker bridges inside *mut functions, ratcheted ---
#
# Some bridges are deliberate and the tree names them so (checker_*_bridge).
# They are still a cost -- each copies a 172-field Checker in and out -- but
# they are load-bearing today and removing them is separate work. What must
# not happen is a NEW one appearing unnoticed, which is exactly how
# checker_finish_binary_units_inplace came to hold a line that took the
# compiler down the first time control reached it.
# Filter raw (*c).method( hits down to those whose target really is a
# by-value method returning a Checker. The `|| true` is not decoration: the
# loop's last iteration is a grep that fails whenever the final candidate is
# not a bridge, and under `set -e` that killed this gate SILENTLY between its
# two checks -- the exact failure this gate exists to make loud.
LIVE_B=$(grep '^BRIDGE ' <<<"$RAW" | awk '{print $2}' | sort -u \
         | while IFS= read -r b; do
             m="${b##*->}"
             if grep -qx "$m" <<<"$RETCK"; then echo "$b"; fi
           done | sort -u || true)
require_nonempty "$LIVE_B" "the bridge scan found nothing at all -- the extraction broke, this is not a measured zero"

FROZEN_B=$(grep -vE '^[[:space:]]*(#|$)' "$BRIDGE_LEDGER" | awk '{print $1}' | sort -u)
require_nonempty "$FROZEN_B" "the frozen bridge ledger has no entries -- it is unreadable, not empty"

NEW_B=$(comm -23 <(echo "$LIVE_B") <(echo "$FROZEN_B") || true)
GONE_B=$(comm -13 <(echo "$LIVE_B") <(echo "$FROZEN_B") || true)

if [[ -n "${NEW_B//[[:space:]]/}" ]]; then
  {
    echo "FAIL: a *mut function newly reaches a by-value Checker method through (*c)."
    echo "      That copies the whole Checker in and out on every call, which is the"
    echo "      entire thing the *mut transcription exists to avoid."
    while read -r b; do [[ -n "$b" ]] && echo "        $b"; done <<<"$NEW_B"
  } >&2
  exit 1
fi
if [[ -n "${GONE_B//[[:space:]]/}" ]]; then
  echo "NOTE: these bridges are gone -- remove them from $BRIDGE_LEDGER:"
  while read -r b; do [[ -n "$b" ]] && echo "        $b"; done <<<"$GONE_B"
fi
echo "checker_spine_parity: by-value bridges live=$(grep -c . <<<"$LIVE_B") frozen=$(grep -c . <<<"$FROZEN_B")"

gate_pass "CHECKER_SPINE_PARITY_OK"
