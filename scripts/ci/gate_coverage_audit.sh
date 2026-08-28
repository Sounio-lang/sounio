#!/usr/bin/env bash
set -euo pipefail

# Gate Coverage Audit -- checks which TypeKind variants are exercised by tests.
# For each non-trivial TypeKind, verifies at least one compile-fail or frontend
# test references a related keyword or type annotation.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# TypeKind variants that need test coverage (excluding primitives/internal)
EPISTEMIC_TYPES=(
  "TyKnowledge:knowledge,Knowledge,measure"
  "TyModel:model,Model"
  "TyModelFamily:model_family,ModelFamily"
  "TyPolicy:policy,Policy"
  "TyAcquisitionPolicy:acquisition_policy,AcquisitionPolicy,plan_acquisition"
  "TyAcquisitionPlan:acquisition_plan,AcquisitionPlan"
  "TyRecoursePolicy:recourse_policy,RecoursePolicy,plan_recourse"
  "TyRecoursePlan:recourse_plan,RecoursePlan"
  "TyAlternativePolicy:alternative_policy,AlternativePolicy"
  "TyAlternativeSet:alternative_set,AlternativeSet"
  "TyAlternativeOption:alternative_option,AlternativeOption"
  "TyTransitionPolicy:transition_policy,TransitionPolicy,commit_alternative"
  "TyTransitionPlan:transition_plan,TransitionPlan,transition_reason"
  "TyDecisionPolicy:decision_policy,DecisionPolicy"
  "TyDeferralPolicy:deferral_policy,DeferralPolicy"
  "TyContest:contest,Contest,lift_knowledge"
  "TyRobust:robust,Robust,prove_robust"
  "TyHyper:hyper,Hyper"
  "TyIntervention:intervention,Intervention"
  "TyCounterfactual:counterfactual,Counterfactual"
  "TyValidated:validated,Validated,validate_manifest"
  "TyAdmissible:admissible,Admissible"
  "TyDeferred:deferred,Deferred,defer_action"
  "TyValidation:validation,Validation"
  "TyEffectVar:effect_var,EffectVar"
  "TyEffectRow:effect_row,EffectRow"
  "TyAleatoric:aleatoric,Aleatoric"
  "TyEpistemic:epistemic,Epistemic"
  "TyCausalEffect:causal_effect,CausalEffect,causal"
  "TyGradedEffect:graded_effect,GradedEffect"
  "TySession:session,Session"
  "TyChan:chan,Chan"
  "TySessionEnd:session_end,SessionEnd"
  "TySingleton:singleton,Singleton"
  "TyScopedEffect:scoped_effect,ScopedEffect"
  "TyCondIndep:cond_indep,CondIndep,conditional_independence"
  "TyPotentialOutcome:potential_outcome,PotentialOutcome"
  "TyATE:ate,ATE,average_treatment"
  "TyDiffPrivate:diff_private,DiffPrivate,dp_mechanism"
  "TyDPBudget:dp_budget,DPBudget,privacy_budget"
  "TyTransportable:transportable,Transportable,transport"
  "TySelectionDiagram:selection_diagram,SelectionDiagram"
  "TyFairPrediction:fair_prediction,FairPrediction,fairness"
  "TyFairnessGap:fairness_gap,FairnessGap"
  "TyDistribution:distribution,Distribution"
  "TySample:sample,Sample"
  "TyConditionalDist:conditional_dist,ConditionalDist"
  "TyEntropic:entropic,Entropic,entropy"
  "TyMutualInfo:mutual_info,MutualInfo"
  "TyKLBounded:kl_bounded,KLBounded"
  "TyVecShaped:vec_shaped,VecShaped"
  "TyMatrixShaped:matrix_shaped,MatrixShaped"
  "TyBroadcastable:broadcastable,Broadcastable"
  "TyELBO:elbo,ELBO"
  "TyVariationalFamily:variational_family,VariationalFamily"
  "TyDifferentiable:differentiable,Differentiable"
  "TyGradient:gradient,Gradient"
  "TyJacobian:jacobian,Jacobian"
  "TyMarkovChain:markov,MarkovChain"
  "TySDE:sde,SDE,stochastic"
  "TyMartingale:martingale,Martingale"
  "TyStationaryDist:stationary,StationaryDist"
  "TyBigO:bigo,BigO,complexity"
  "TyAmortized:amortized,Amortized"
  "TyProof:proof,Proof"
  "TyLemma:lemma,Lemma"
  "TyAxiom:axiom,Axiom"
)

total=${#EPISTEMIC_TYPES[@]}
covered=0
uncovered_list=()

for entry in "${EPISTEMIC_TYPES[@]}"; do
  type_name="${entry%%:*}"
  keywords="${entry#*:}"

  found=0
  IFS=',' read -ra KW_ARRAY <<< "$keywords"
  for kw in "${KW_ARRAY[@]}"; do
    if grep -rql "$kw" "$ROOT_DIR/tests/compile-fail/" "$ROOT_DIR/tests/frontend/" 2>/dev/null; then
      found=1
      break
    fi
  done

  if [[ "$found" -eq 1 ]]; then
    covered=$((covered + 1))
  else
    uncovered_list+=("$type_name")
  fi
done

pct=$((covered * 100 / total))

echo "=== Gate Coverage Audit ==="
echo "Total epistemic TypeKind variants: $total"
echo "Covered by tests: $covered"
echo "Uncovered: $((total - covered))"
echo "Coverage: ${pct}%"
echo ""

if [[ ${#uncovered_list[@]} -gt 0 ]]; then
  echo "Uncovered variants:"
  for v in "${uncovered_list[@]}"; do
    echo "  - $v"
  done
fi

echo ""
echo "gate_coverage_audit: ${pct}% (${covered}/${total})"
exit 0
