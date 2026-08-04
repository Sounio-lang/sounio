#!/usr/bin/env bash
# A gate that can pass on no data is not a gate.
#
# Census, 2026-08-04, over the 417 *_gate.{sh,py} in scripts/ci and scripts/dev:
# 258 (67%) had no emptiness guard of any kind. The failure shape is always the
# same — a value is extracted from a tool, the extraction comes back empty
# because the tool changed, moved or broke, and the empty value is then compared
# as though it meant something. Measured instances on that date:
#
#   run_pass_output_gate.sh          "PASS (strict): all 0 ... tests"
#   reproduce_artifact.sh            "PASS: test suite" while the suite crashed
#   check_doc_snippets.sh            "0 pass, 0 fail, 0 total" -> exit 0
#   eisa_bridge_conformance_gate.sh  "PASS anti-vacuity" on an empty extraction
#   madaros_readiness_status.sh      a failing `gh` call -> status[...]=pass
#
# This gate flags the pattern. It is RATCHETED against a baseline because 258
# scripts cannot be fixed at once — but the count may only fall, and a NEW gate
# must be clean. That is the difference between a debt and a leak.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9
. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "gate_vacuity_gate"

BASELINE="$ROOT_DIR/scripts/ci/fixtures/gate_vacuity_baseline.txt"

mapfile -t GATES < <(git ls-files 'scripts/ci/*_gate.sh' 'scripts/dev/*_gate.sh' 2>/dev/null | sort -u)
require_min_count "${#GATES[@]}" 300 "gate scripts"

# A script is FLAGGED when it extracts a value from a tool and carries no
# emptiness guard anywhere. Both halves matter: extraction without a guard is
# the defect; a script that extracts nothing has nothing to guard.
EXTRACTS='\$\((grep|awk|sed|cut|jq|python3|wc|head|tail|tr|rg)[^)]*\)|\$\([^)]*\|[^)]*(grep|awk|sed|jq|wc)[^)]*\)'
# A numeric FLOOR on an extracted count is a guard too — `[[ "$n" -ge 8 ]] || fail`
# refuses an empty extraction just as surely as `[[ -n "$n" ]]` does, and it
# refuses a partial one as well. Omitting it made this gate reject a guard that
# was genuinely there, which is its own kind of wrong answer: a checker that
# only recognises one spelling of correctness pushes people toward the spelling
# rather than the property.
GUARDS='\[\[? *-[nz] *"?\$|\$\{#[A-Za-z_]+\[@\]\} *-(gt|ge)|-s +"?\$|-(ge|gt|eq|lt|le) +[0-9]|require_nonempty|require_min_count|gate_fail|nothing to measure|vacuous'

: >"${TMPDIR:-/tmp}/gate_vacuity_flagged.$$"
trap 'rm -f "${TMPDIR:-/tmp}/gate_vacuity_flagged.$$"' EXIT
FLAGGED="${TMPDIR:-/tmp}/gate_vacuity_flagged.$$"

scanned=0
for g in "${GATES[@]}"; do
  [[ -f "$g" ]] || continue
  scanned=$((scanned + 1))
  if grep -Eq "$EXTRACTS" "$g" 2>/dev/null && ! grep -Eq "$GUARDS" "$g" 2>/dev/null; then
    printf '%s\n' "$g" >>"$FLAGGED"
  fi
done
require_min_count "$scanned" 300 "gate scripts actually read"

sort -u -o "$FLAGGED" "$FLAGGED"
flagged_n=$(grep -c . "$FLAGGED" || true)
echo "  scanned $scanned gates, flagged ${flagged_n:-0} with an unguarded extraction"

if [[ ! -f "$BASELINE" ]]; then
  echo "  no baseline at $BASELINE. Current set:"
  sed 's/^/    /' "$FLAGGED"
  gate_fail "seed the baseline from THIS output and commit it"
fi

grep -vE '^\s*(#|$)' "$BASELINE" | sort -u >"$FLAGGED.baseline"

NEW=$(comm -23 "$FLAGGED" "$FLAGGED.baseline")
FIXED=$(comm -13 "$FLAGGED" "$FLAGGED.baseline")

if [[ -n "$NEW" ]]; then
  echo "  NEW unguarded gate(s) — every value pulled from a tool needs an emptiness check:"
  sed 's/^/    /' <<<"$NEW"
  echo
  echo "  Source scripts/lib/gate_assert.sh and use require_nonempty / require_min_count /"
  echo "  count_matches. count_matches is the one that separates 'no match' from 'the tool"
  echo "  broke'; \`grep -c ... || true\` reports both as 0."
  gate_fail "$(grep -c . <<<"$NEW") gate(s) added without an emptiness guard"
fi

if [[ -n "$FIXED" ]]; then
  echo "  these are in the baseline but now carry a guard — remove them from $BASELINE:"
  sed 's/^/    /' <<<"$FIXED"
  gate_fail "the baseline may only shrink, and it is now out of date"
fi

gate_pass "${flagged_n:-0} known-unguarded, all in the baseline; the list may only shrink"
