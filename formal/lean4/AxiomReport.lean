/-
Axiom report for the headline theorems of the `formal/lean4/` package.

Not a library root: it is run by `scripts/ci/lean_axiom_report.sh` in the CI
`lean-proofs` job (after `lake build`, via `lake env lean AxiomReport.lean`),
so the job log carries the real `#print axioms` output. formal/AXIOM_INVENTORY.md
points here instead of restating a result it cannot regenerate.
-/
import SounioHydrogenPbox
import SounioGradedModal

-- Hydrogen p-box: monotone event equivalence (no independence assumption).
#print axioms Sounio.HydrogenPbox.monotone_event_equiv

-- EGC graded soundness.
#print axioms SounioGradedModal.egc_graded_soundness
