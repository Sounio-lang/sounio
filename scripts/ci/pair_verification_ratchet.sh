#!/usr/bin/env bash
# A claim is not a finding until someone who did not produce it returns a verdict.
#
# Why this exists
# ---------------
# Investigations in this repository land as prose. Prose cannot be verified
# claim by claim, so it is verified as a whole, which in practice means it is
# believed as a whole. Three retractions in one night came out of that: a PBPK
# result reported as structurally invalid and withdrawn, a field-resolution rule
# that turned out to be an artefact of the prober's own initialiser, and a
# conflict count of zero produced by a grep that matched nothing.
#
# Every one of those would have been caught by a single question -- what command
# produced this, and what is the OTHER route to it -- asked by someone who did
# not write it.
#
# So findings become numbered atomic claims, and each carries a verdict from a
# second party:
#
#   ## CLAIM-n -- <one falsifiable statement>
#   - produced-by: <command or method>
#   - **VERDICT-n: CONFIRMED | REFUTED | UNMEASURED** by <who>
#   - via: <a DIFFERENT route than the one that produced it>
#
# UNMEASURED is a first-class verdict, not a failure to finish. An investigation
# with five confirmed claims and one honest UNMEASURED is worth more than six
# confident ones, and the third category is the one every review in this
# repository has been missing.
#
# What this gate blocks on
# ------------------------
# Claims with NO verdict at all. Not REFUTED (a refuted claim is a working
# protocol) and not UNMEASURED (that is the honest state). Only silence.
#
# The ceiling is frozen and lowered by editing the line, which puts each
# reduction in a diff next to the work that produced it -- the same shape as
# witness_declares_its_sabotage_gate.sh, and for the same reason: a debt that can
# grow while the gate stays green is not a debt.
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9
. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "pair_verification_ratchet"

CLAIMS_DIR="docs/audit/claims"
UNVERIFIED_CEILING="${SOUNIO_PAIR_UNVERIFIED_CEILING:-0}"

ART_DIR="$ROOT_DIR/artifacts/gates"; mkdir -p "$ART_DIR"
ART="$ART_DIR/pair_verification.json"

if [[ ! -d "$CLAIMS_DIR" ]]; then
  gate_fail "no $CLAIMS_DIR -- the claims directory is the gate's entire input"
fi

mapfile -t FILES < <(find "$CLAIMS_DIR" -name '*.md' -type f | sort)
if [[ ${#FILES[@]} -eq 0 ]]; then
  # Non-vacuity: an empty corpus is a missing corpus, not a clean one. This gate
  # exists because measuring nothing reads as success, so it may not do that.
  gate_fail "inspected ZERO claim files -- $CLAIMS_DIR is empty or unreadable"
fi

total=0; confirmed=0; refuted=0; unmeasured=0; unverified=0; noroute=0
for f in "${FILES[@]}"; do
  while IFS= read -r n; do
    total=$((total + 1))
    # The verdict for CLAIM-n must name the same n, so a stray verdict cannot
    # cover a claim it does not belong to.
    v="$(grep -oE "\*\*VERDICT-${n}: *(CONFIRMED|REFUTED|UNMEASURED)" "$f" | head -1 | grep -oE '(CONFIRMED|REFUTED|UNMEASURED)')"
    case "$v" in
      CONFIRMED)  confirmed=$((confirmed + 1)) ;;
      REFUTED)    refuted=$((refuted + 1)) ;;
      UNMEASURED) unmeasured=$((unmeasured + 1)) ;;
      *)          echo "  UNVERIFIED  $f CLAIM-$n has no verdict"; unverified=$((unverified + 1)); continue ;;
    esac
    # A CONFIRMED or REFUTED verdict must state the other route. Without it the
    # verifier may simply have re-run the producer's command, which reproduces
    # the producer's errors and certifies them.
    if [[ "$v" != "UNMEASURED" ]]; then
      blk="$(awk -v n="$n" '$0 ~ "^\\*\\*VERDICT-"n"|VERDICT-"n":" {p=1} p&&/^- via:/{print; exit}' "$f")"
      [[ -z "$blk" ]] && blk="$(grep -A3 -E "VERDICT-${n}:" "$f" | grep -m1 '^- via:')"
      if [[ -z "$blk" ]]; then
        echo "  NO-ROUTE    $f CLAIM-$n is $v with no 'via:' -- the verifier may have re-run the producer's command"
        noroute=$((noroute + 1))
      fi
    fi
  done < <(grep -oE '^## CLAIM-[0-9]+' "$f" | grep -oE '[0-9]+')
done

status=$([[ $unverified -le $UNVERIFIED_CEILING && $noroute -eq 0 ]] && echo pass || echo fail)
printf '{"status":"%s","metrics":{"total":%d,"passed":%d,"failed":%d,"not_run":%d},"verdicts":{"confirmed":%d,"refuted":%d,"unmeasured":%d,"unverified":%d,"no_route":%d},"files":%d}\n' \
  "$status" "$total" "$((confirmed + refuted + unmeasured))" "$unverified" "$unmeasured" \
  "$confirmed" "$refuted" "$unmeasured" "$unverified" "$noroute" "${#FILES[@]}" | gate_write_artifact "$ART"

echo "pair_verification_ratchet: status=$status files=${#FILES[@]} claims=$total"
echo "  confirmed=$confirmed refuted=$refuted unmeasured=$unmeasured unverified=$unverified (ceiling $UNVERIFIED_CEILING) no_route=$noroute"
if [[ $noroute -ne 0 ]]; then
  gate_fail "$noroute verdict(s) state no independent route"
fi
if [[ $unverified -gt $UNVERIFIED_CEILING ]]; then
  gate_fail "$unverified claim(s) carry no verdict; ceiling is $UNVERIFIED_CEILING"
fi
gate_pass "$total claims, all carrying a verdict ($confirmed confirmed, $refuted refuted, $unmeasured unmeasured)"
exit 0
