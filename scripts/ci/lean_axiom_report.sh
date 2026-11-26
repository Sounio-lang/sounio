#!/usr/bin/env bash
# lean_axiom_report.sh -- print `#print axioms` for the headline theorems.
#
# Runs formal/AxiomReport.lean (package formal/, toolchain formal/lean-toolchain)
# and formal/lean4/AxiomReport.lean (package formal/lean4/); with --bootstrap, runs
# formal/Bootstrap/AxiomReport.lean instead (the CI Bootstrap Proofs job). The output is the
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
  local dir="$1" report="$2"; shift 2
  local log
  log="$(mktemp)"
  echo "== ${dir#"$REPO_ROOT"/}: lake build $*"
  (cd "$dir" && lake build "$@") >"$log" 2>&1 || { cat "$log" >&2; rm -f "$log"; fail "lake build $* in $dir"; }
  (cd "$dir" && lake env lean "$report") >"$log" 2>&1 || { cat "$log" >&2; rm -f "$log"; fail "lake env lean $report in $dir"; }
  cat "$log" | tee -a "$REPORT"
  local want got
  want="$(grep -c '^#print axioms ' "$dir/$report")"
  got="$(grep -cE "depends on axioms|does not depend on any axioms" "$log" || true)"
  rm -f "$log"
  [[ "$want" == "$got" ]] || fail "$dir/$report: $want #print axioms commands, $got results"
}

case "${1:-}" in
  --bootstrap)
    run_report "$REPO_ROOT/formal" Bootstrap/AxiomReport.lean Bootstrap.SporosX1Comp
    ;;
  "")
    run_report "$REPO_ROOT/formal" AxiomReport.lean OntologyELPlusClosureComplete ElfLinker TypeChecker TrustSemiring EpistemicSafety EpistemicStat EpistemicJoint SCCore
    run_report "$REPO_ROOT/formal/lean4" AxiomReport.lean SounioHydrogenPbox SounioGradedModal
    ;;
  *) fail "usage: $0 [--bootstrap]" ;;
esac

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
