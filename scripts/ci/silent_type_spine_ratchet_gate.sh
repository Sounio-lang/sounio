#!/usr/bin/env bash
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/gate_artifact.sh"
# Parameter-position uses of type kinds the *mut lowering spine does not handle.
#
# GATE_CONTRACT: v0
# GATE_ID: silent_type_spine_ratchet
# GATE_CLAIMS: the scaffolding that lets unlowered type kinds cross a parameter may only shrink
# GATE_ENGINE: source census (no compiler invocation)
# GATE_RESULT_ON_SKIP: fail
#
# WHY.
#
# self-hosted/check/check.sio has two type-lowering spines. Parameter types go
# through checker_lower_type_expr_mut, whose match handles 34 kinds; the other
# spine handles 54. The twenty in the difference fall to:
#
#     _ => { checker_note_type_error_mut(c)  ty_error() }
#
# which counts an error and prints nothing, so `check: OK` and rc=0 while the
# checker has recorded a failure. The founder's account: that silence is
# scaffolding — it is what let those twenty families be written at all while the
# compiler stayed self-hosting.
#
# The twenty are not incidental. They are the ZD surgical family, the proof types,
# the causal-inference types, differential privacy, aleatoric uncertainty and
# session types — the whole ambitious surface. The ordinary kinds (Named, Unit,
# Never, Infer, Knowledge, Model, Policy) are on both spines.
#
# Scaffolding is not a defect. Invisible scaffolding is. This gate makes it
# legible and shrink-only: as each family is brought across, the count falls, and
# when it reaches zero the `_ =>` can become a hard error by exhaustion rather
# than by decision.
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
FROZEN_FILE="scripts/ci/silent_type_spine.frozen"
OUT="${GATE_ARTIFACT:-artifacts/gates/silent_type_spine_ratchet.json}"

# ZD surgical family (Forgettable ExactlyPrivate Editable CapabilityGated
# Composable Audited Revivable Interpretable) now lowers on the *mut spine.
# Remaining scaffolding: proof, causal, privacy, aleatoric, sessions.
KINDS='Axiom|Lemma|Proof|CausalEffect|Counterfactual|Intervention|PotentialOutcome|DiffPrivate|DPBudget|Aleatoric|Chan|Session'
PAT="fn [a-zA-Z_0-9]+\([^)]*: *(${KINDS})[<\[]"

frozen=$(tr -dc '0-9' < "$FROZEN_FILE" 2>/dev/null)
[[ -n "$frozen" ]] || { echo "silent_type_spine: FAIL: missing or unreadable $FROZEN_FILE" >&2; exit 1; }

# NEGATIVE CONTROL 1 — the kind list must still match the spine difference. If
# check.sio stops naming these kinds at all, the census is measuring a corpus
# against a vocabulary that no longer exists.
# `grep -c` prints 0 AND returns 1 when nothing matches, so `|| echo 0` appends a
# second line and the arithmetic test below silently never fires. Measured: the
# negative control for this very guard did not trip until the pipe was fixed.
spine_hits=$(grep -cE "TypeExprKind::Type(${KINDS})" self-hosted/check/check.sio 2>/dev/null | head -1)
[[ -n "$spine_hits" ]] || spine_hits=0
if [[ "$spine_hits" -eq 0 ]]; then
  echo "silent_type_spine: FAIL: check.sio names none of the tracked kinds" >&2
  echo "  the vocabulary moved; re-derive the list from the two spines before trusting a count" >&2
  exit 1
fi

mapfile -t hits < <(git ls-files -z '*.sio' 2>/dev/null | xargs -0 grep -HnoE "$PAT" 2>/dev/null | sort)
measured="${#hits[@]}"

# NEGATIVE CONTROL 2 — a census that finds nothing is not a pass while the frozen
# count is above zero. Reaching zero must be earned by the spine, not by a broken
# pattern.
if [[ "$measured" -eq 0 && "$frozen" -gt 0 ]]; then
  echo "silent_type_spine: FAIL: census found 0 while frozen at $frozen" >&2
  echo "  either twenty type families left the corpus at once, or the pattern is broken" >&2
  exit 1
fi

printf '%s\n' "${hits[@]}" | sed 's/^/SILENT_SPINE_CROSSING /'

mkdir -p "$(dirname "$OUT")"
status=pass; rc=0
if (( measured > frozen )); then
  status=fail; rc=1
  echo "REFUSE: parameter-position crossings of unlowered kinds rose ${frozen} -> ${measured}." >&2
  echo "  These cross a spine that counts an error and prints nothing. Adding one is free" >&2
  echo "  today and invisible tomorrow. Lower the kind onto checker_lower_type_expr_mut," >&2
  echo "  or say in the PR why this one must cross." >&2
elif (( measured < frozen )); then
  echo "OK: crossings fell ${frozen} -> ${measured}. Lower the frozen count:"
  echo "  printf '%s\\n' ${measured} > ${FROZEN_FILE}"
else
  echo "OK: crossings hold at ${frozen} (each named above)."
fi

cat <<JSON | gate_write_artifact "$OUT"
{
  "gate": "silent_type_spine_ratchet",
  "status": "${status}",
  "claims": "parameter-position uses of kinds the *mut spine does not lower may only shrink",
  "frozen": ${frozen},
  "measured": ${measured},
  "metrics": { "total": ${measured}, "passed": $(( measured <= frozen ? measured : 0 )), "failed": $(( measured > frozen ? 1 : 0 )), "not_run": 0 }
}
JSON
exit "$rc"
