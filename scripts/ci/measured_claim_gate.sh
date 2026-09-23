#!/usr/bin/env bash
# measured_claim_gate.sh — a number this repository states about itself must
# equal the number the tree yields.
#
# WHY. A backlog sweep on 2026-08-27 rebased 32 open pull requests. Nearly every
# one asserted a figure that had been true when written and was false by then:
#
#   a ratchet said 471, the tree said 468
#   a census said 1832, the corpus held 1842
#   a ladder said i256/i512 were "absent entirely" -- they ship with run-pass witnesses
#   a doc said the committed ELF prints FAIL -- the required gate was green
#   a note said four `sorry` remain in formal/lean4 -- there are none
#
# None of that is caught by review or by a rebase: the conflicts were textual,
# the merges were clean, and CI was green. A stale number reads exactly like a
# fresh one. This gate re-derives the ones that can be re-derived.
#
# SCOPE, deliberately narrow. Only claims with a command that produces them. A
# sentence of prose is not admitted, however wrong it may be -- a gate that
# guessed at prose would fail honest pull requests and, worse, would itself be
# machinery asserting more than it can check. See scripts/ci/fixtures/
# measured_claims.tsv for the admission rule.
#
# WHAT WOULD KILL THIS GATE. It compares committed text against a command. If a
# claim_cmd silently returns empty -- file renamed, JSON key gone -- an
# unguarded gate would compare "" to "" and pass. Both sides are therefore
# required non-empty and strictly formatted (numeric or SHA-256 digest), and
# the selftest drives that path directly.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9
. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "measured_claim_gate"

CLAIMS="${SOUNIO_MEASURED_CLAIMS:-$ROOT_DIR/scripts/ci/fixtures/measured_claims.tsv}"
BASELINE="${SOUNIO_MEASURED_CLAIMS_BASELINE:-$ROOT_DIR/scripts/ci/fixtures/measured_claims_baseline.txt}"

require_file "$CLAIMS" "the claims table is missing — this gate checks nothing without it"

baselined() {
  [[ -f "$BASELINE" ]] || return 1
  grep -qx -- "$1" "$BASELINE" 2>/dev/null
}

run_side() {  # run_side <label> <id> <cmd>
  local out
  out="$(bash -o pipefail -c "$3" 2>/dev/null | tr -d '[:space:]')"
  if [[ -z "$out" ]]; then
    echo "  $2: $1 command produced NOTHING — the claim cannot be read." >&2
    echo "      cmd: $3" >&2
    echo "      A gate that compares two empty strings passes while checking nothing." >&2
    return 1
  fi
  if [[ ! "$out" =~ ^-?[0-9]+$ && ! "$out" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "  $2: $1 command produced neither a number nor a sha256 digest: '$out'" >&2
    echo "      cmd: $3" >&2
    return 1
  fi
  printf '%s' "$out"
}

rows=0; agreed=0; drifted=0; excused=0; broken=0
declare -a FAILED=()

while IFS=$'\t' read -r id desc claim_cmd measure_cmd fix_cmd; do
  [[ -z "${id:-}" || "$id" == \#* || "$id" == "id" ]] && continue
  rows=$((rows + 1))

  claimed="$(run_side claim "$id" "$claim_cmd")"   || { broken=$((broken+1)); FAILED+=("$id"); continue; }
  measured="$(run_side measure "$id" "$measure_cmd")" || { broken=$((broken+1)); FAILED+=("$id"); continue; }

  if [[ "$claimed" == "$measured" ]]; then
    agreed=$((agreed + 1))
    if baselined "$id"; then
      echo "  FIXED     $id — claimed $claimed, measured $measured. It agrees now; drop it from the baseline."
    else
      echo "  ok        $id — $claimed"
    fi
    continue
  fi

  if baselined "$id"; then
    excused=$((excused + 1))
    echo "  baselined $id — claimed $claimed, measured $measured ($desc)"
  else
    drifted=$((drifted + 1))
    FAILED+=("$id")
    echo "  DRIFTED   $id — claimed $claimed, measured $measured" >&2
    echo "            $desc" >&2
    echo "            claim:   $claim_cmd" >&2
    echo "            measure: $measure_cmd" >&2
    if [[ -n "${fix_cmd:-}" ]]; then
      echo "            FIX:     $fix_cmd" >&2
    else
      echo "            FIX:     (no fix_cmd recorded for this row — add one)" >&2
    fi
  fi
done < "$CLAIMS"

[[ $rows -gt 0 ]] || gate_fail "the claims table parsed to zero rows — it is present but says nothing"

echo "[measured-claim] rows=$rows agreed=$agreed drifted=$drifted baselined=$excused unreadable=$broken"

if [[ $broken -gt 0 ]]; then
  gate_fail "$broken claim(s) could not be read: ${FAILED[*]}. Fix the command or remove the row — an unreadable claim is not a passing one."
fi

if [[ $drifted -gt 0 ]]; then
  echo >&2
  echo "  A number this repository states about itself no longer matches the tree." >&2
  echo "  Run the FIX command printed above and commit the result. Baselining is" >&2
  echo "  for debt that predates your change, not for debt your change introduces." >&2
  gate_fail "$drifted claim(s) drifted: ${FAILED[*]}"
fi

echo "MEASURED_CLAIM_GATE_OK: every re-derivable claim matches the tree ($agreed agreed, $excused baselined)"
