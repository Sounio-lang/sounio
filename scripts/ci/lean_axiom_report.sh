#!/usr/bin/env bash
# lean_axiom_report.sh -- print `#print axioms` for the headline theorems.
#
# Runs formal/AxiomReport.lean (package formal/, toolchain formal/lean-toolchain)
# and formal/lean4/AxiomReport.lean (package formal/lean4/). The output is the
# authoritative list that formal/AXIOM_INVENTORY.md refers to; the inventory
# generator (scripts/dev/gen_axiom_inventory.sh) does not run Lean.
#
# Fails when: a module does not build, a report does not run, a theorem named in
# a report produces no `#print axioms` line, or any footprint contains `sorryAx`.
# It does NOT fail on a non-standard axiom: the point is to show the footprint,
# and formal/ declares 61 axioms on purpose (see AXIOM_INVENTORY.md).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -d "${HOME}/.elan/bin" ]]; then export PATH="${HOME}/.elan/bin:${PATH}"; fi
command -v lake >/dev/null 2>&1 || { echo "FAIL: lake not found" >&2; exit 1; }

fail() { echo "FAIL: $*" >&2; exit 1; }
REPORT="$(mktemp)"
trap 'rm -f "$REPORT"' EXIT

run_report() {
  local dir="$1"; shift
  local log
  log="$(mktemp)"
  echo "== ${dir#"$REPO_ROOT"/}: lake build $*"
  (cd "$dir" && lake build "$@") >"$log" 2>&1 || { cat "$log" >&2; rm -f "$log"; fail "lake build $* in $dir"; }
  (cd "$dir" && lake env lean AxiomReport.lean) >"$log" 2>&1 || { cat "$log" >&2; rm -f "$log"; fail "lake env lean AxiomReport.lean in $dir"; }
  cat "$log" | tee -a "$REPORT"
  local want got
  want="$(grep -c '^#print axioms ' "$dir/AxiomReport.lean")"
  got="$(grep -cE "depends on axioms|does not depend on any axioms" "$log" || true)"
  rm -f "$log"
  [[ "$want" == "$got" ]] || fail "$dir/AxiomReport.lean: $want #print axioms commands, $got results"
}

run_report "$REPO_ROOT/formal" OntologyELPlusClosureComplete ElfLinker TypeChecker
run_report "$REPO_ROOT/formal/lean4" SounioHydrogenPbox SounioGradedModal

if grep -q 'sorryAx' "$REPORT"; then
  grep 'sorryAx' "$REPORT" >&2
  fail "a headline theorem depends on sorryAx"
fi

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    echo "### #print axioms for headline theorems"
    echo '```'
    cat "$REPORT"
    echo '```'
  } >>"$GITHUB_STEP_SUMMARY"
fi
echo "PASS: axiom report produced for $(grep -cE 'depends on axioms|does not depend on any axioms' "$REPORT") theorems"
