<!-- docs:meta
topic_id: repo.examples.neuroreceptor-pet.results.readme
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#repo.examples.neuroreceptor-pet.results.readme
-->

# results/

Generated numerical outputs from `pet_2tcm_epistemic.sio` (audit) and `pet_2tcm_export.sio` (TAC curve).

All files in this directory are **deterministic** and **reproducible** from the current compiler artifact. Regenerate with:

```bash
cd /workspace/sounio
export SOUC_BIN="$(pwd)/bin/souc"
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"

"$SOUC_BIN" run examples/neuroreceptor_pet/pet_2tcm_epistemic.sio \
    > examples/neuroreceptor_pet/results/audit_output.txt

"$SOUC_BIN" run examples/neuroreceptor_pet/pet_2tcm_export.sio \
    > examples/neuroreceptor_pet/results/tac_curve.csv
```

## Files

| File | Description |
|------|-------------|
| `audit_output.txt` | Full stdout of the 12-test numerical audit (priors, Cp(t) check, GUM results, finite-difference derivatives vs analytic, sensitivity fractions, pass/fail) |
| `tac_curve.csv` | Time-activity curve sampled at 1-minute intervals over 0–60 min. Columns: `t, Cp, C1, C2, CT`. Produced by fixed-step RK4 with dt=0.05 |

## Quick Sanity Checks on the CSV

```
head -n 1   results/tac_curve.csv   → t,Cp,C1,C2,CT
head -n 2   results/tac_curve.csv   → 0.00, Cp(0)=1.0, zeros
grep "^5\." results/tac_curve.csv   → Cp(5) ≈ 0.3679
grep "^60"  results/tac_curve.csv   → Cp(60) ≈ 6e-6, CT ≈ 0.056
```

Current values (as of 2026-04-28):
- CT peak ≈ **0.308** at t ≈ 7 min
- CT at t=60 ≈ **0.056**
- Cp(60) ≈ **6.1e-6**

## Policy

No fabricated data. All numerical outputs must trace to a specific compiler artifact and commit. Plots derived from `tac_curve.csv` (via Python/R/Julia) should be saved alongside with the plotting script for full reproducibility.
