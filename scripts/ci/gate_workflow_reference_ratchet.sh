#!/usr/bin/env bash
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/gate_artifact.sh"
# gate_workflow_reference_ratchet.sh — freeze the number of gate scripts that no
# workflow names, so the count cannot grow.
#
# WHY THIS EXISTS
#
# scripts/dev/check_workflow_script_refs.sh already enforces one direction:
# every `scripts/...` path named in a workflow must exist and be executable. It
# protects the WORKFLOW AUTHOR from a broken reference.
#
# The reverse is unenforced. A gate script may be added, committed, and reviewed
# without any workflow naming it, and nothing breaks — which is precisely how the
# unnamed set reached its present size. The obligation sat on nobody.
# (SOUNIO-EFFORT-LOCATION: move the effort from the reader to the actor.)
#
# WHAT THIS MEASURES, AND WHAT IT DOES NOT
#
# It measures DIRECT INVOCATION: whether a workflow names the script. It does
# NOT measure COVERAGE — a script no workflow names may still be run by a parent
# that a workflow does name. #1972 measured exactly that difference and found 45
# such scripts. A number carries how it was measured, or it is not evidence.
#
# So: a rise here means "one more gate that no workflow names", not "one more
# gate that never runs".
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
# Shared assertions rather than hand-rolled ones — gate_vacuity_gate.sh requires
# them. Its reason is the one this gate is about: an extraction that silently
# returns nothing must be red, not green.
. "scripts/lib/gate_assert.sh"
gate_name "gate_workflow_reference_ratchet"

REF="scripts/ci/gate_workflow_reference.frozen"
OUT="${GATE_ARTIFACT:-artifacts/gates/gate_workflow_reference_ratchet.json}"

listar_gates()   { git ls-files 'scripts/ci/*.sh' 'scripts/dev/*gate*.sh' | sort; }
listar_nomeados() {
  grep -rhoE '(\./)?scripts/[A-Za-z0-9_./-]+' .github/workflows/ 2>/dev/null \
    | sed 's|^\./||' | sort -u
}
nao_nomeados() { comm -23 <(listar_gates) <(listar_nomeados); }

selftest() {
  local rc=0 n
  # POSITIVE: a gate known to be named by a workflow must NOT appear as unnamed.
  if nao_nomeados | grep -qx 'scripts/ci/concept_status_gate.sh'; then
    echo "  FALHA POSITIVO: um gate nomeado num workflow saiu como nao-nomeado"; rc=1
  else echo "  ok   POSITIVO: gate nomeado num workflow nao aparece como nao-nomeado"; fi
  # NEGATIVE 1: the named list must be non-empty — an empty list would make every
  # gate look unnamed and the number would be noise.
  n=$(listar_nomeados | wc -l | tr -d ' ')
  if [ "$n" -lt 20 ]; then echo "  FALHA NEGATIVO 1: so $n referencias em workflows — instrumento morto"; rc=1
  else echo "  ok   NEGATIVO 1: lista de referencias nao vazia ($n)"; fi
  # NEGATIVE 2: the gate list must be non-empty for the same reason.
  n=$(listar_gates | wc -l | tr -d ' ')
  if [ "$n" -lt 50 ]; then echo "  FALHA NEGATIVO 2: so $n gates encontrados — instrumento morto"; rc=1
  else echo "  ok   NEGATIVO 2: lista de gates nao vazia ($n)"; fi
  # NEGATIVE 3: a path named with a leading ./ in a workflow must still count as
  # named — otherwise the ratchet inflates on a formatting difference.
  if printf './scripts/ci/x.sh\n' | sed 's|^\./||' | grep -qx 'scripts/ci/x.sh'; then
    echo "  ok   NEGATIVO 3: prefixo ./ normalizado"
  else echo "  FALHA NEGATIVO 3: prefixo ./ nao normalizado"; rc=1; fi
  echo "falhas: $rc"
  return $rc
}

[ "${1:-}" = "--selftest" ] && { selftest; exit $?; }
[ "${1:-}" = "--list" ] && { nao_nomeados; exit 0; }

selftest >/dev/null 2>&1 || {
  echo "ABORT: the gate's own controls fail — its number would be noise, not evidence."
  selftest; exit 2
}

# Anti-vacuity, in both lists. If the workflow reference list came back empty
# every gate would look unnamed; if the gate list came back empty none would.
# Either way the number would be noise. This is the same failure the gate exists
# to catch, one level up.
n_ref=$(listar_nomeados | wc -l | tr -d ' ')
n_gate=$(listar_gates | wc -l | tr -d ' ')
require_min_count "$n_ref" 20 "scripts/ references in workflows"
require_min_count "$n_gate" 50 "versioned gate scripts"

atual=$(nao_nomeados | wc -l | tr -d ' ')
require_nonempty "$atual" "the unnamed-gate count came back empty"
[ -f "$REF" ] || printf '%s\n' "$atual" | gate_write_artifact "$REF"
congelado=$(head -1 "$REF" | tr -d ' ')

mkdir -p "$(dirname "$OUT")"
estado=pass; falhou=0
if [ "$atual" -gt "$congelado" ]; then
  estado=fail; falhou=1
  echo "REFUSE: gate scripts that no workflow names rose ${congelado} -> ${atual}."
  echo "Adding a gate that no workflow names costs nothing today and is invisible"
  echo "tomorrow. Name it in a workflow, or say in the PR why it must not run."
  echo "Newly unnamed:"
  comm -13 <(sort "$REF.list" 2>/dev/null || true) <(nao_nomeados) | sed 's/^/  /'
elif [ "$atual" -lt "$congelado" ]; then
  echo "OK: unnamed gates fell ${congelado} -> ${atual}. Lower the frozen count:"
  echo "  printf '%s\\n' ${atual} > ${REF} && bash $0 --list > ${REF}.list"
else
  echo "OK: unnamed gates hold at ${congelado}."
fi
# The measured list is a diagnostic for humans; nothing here reads it back. It
# used to be written next to the frozen list, inside scripts/ci/ — a critical
# path — so every CI run left the checkout dirty and the worktree governance gate
# later in the same job refused it (unallowed_critical_dirty=1 exceeds max=0).
# A gate must not dirty the tree the next gate inspects.
mkdir -p "$(dirname "$OUT")"
nao_nomeados > "$(dirname "$OUT")/gate_workflow_reference.measured.list"

cat <<JSON | gate_write_artifact "$OUT"
{
  "gate": "gate_workflow_reference_ratchet",
  "status": "${estado}",
  "measures": "direct invocation by a workflow, not coverage",
  "frozen": ${congelado},
  "measured": ${atual},
  "metrics": { "total": ${atual}, "passed": $(( atual - falhou )), "failed": ${falhou}, "not_run": 0 }
}
JSON
exit "${falhou}"
