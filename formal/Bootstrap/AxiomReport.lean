/-
Axiom report for the headline theorems of the bootstrap model built in CI.

Not a library root: it is run by `scripts/ci/lean_axiom_report.sh --bootstrap` in the
CI `bootstrap-proofs` job (`lake build <modules>` then `lake env lean
Bootstrap/AxiomReport.lean`), so the job log carries the real `#print axioms` output.
The heavier bootstrap theorems are audited by their own scripts on the lab cluster:
`bootstrap/model/seed_proofs.sh`, `sporos_stage1_proofs.sh`, `sporos_vm_proofs.sh`.
-/
import Bootstrap.SporosX1Comp

-- Sporos stage-1 checker (Root A decisions 31/34): accept direction from check_start.
-- (`stage1_oracle_exists_target`, `stage1_final_target`, and `stage1_final_exit2_target`
-- in `BootstrapSporosStage1` are built -j1 and axiom-checked by
-- `bootstrap/model/sporos_stage1_proofs.sh`, matching `BootstrapSeeds`.)
#print axioms SporosX1.checker_accepts_from_start
