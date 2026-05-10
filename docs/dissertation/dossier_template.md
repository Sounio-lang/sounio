<!-- docs:meta
topic_id: repo.docs.dissertation.dossier-template
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.dossier-template
-->

# PBPK Dossier — Template

This is the human-readable section skeleton emitted by
`scripts/dissertation/dossier_generator.sio`. The generator is
driven from a `DossierInput` struct (no file I/O) and prints
Markdown matching this layout. The smoke test in
`tests/run-pass/dossier_smoke.sio` shows a representative input.

Claim boundary: generated dossiers should be read with
`docs/dissertation/pbpk_claim_truth_table.md`. The current defensible wording is
that Sounio demonstrates GUM-through-ODE in the PBPK14 stdlib and
GPU-validated K-AXI kernels for narrower PBPK/GUM witnesses; PBPK14 GPU-first
Tsit5 and GPU speedup claims require later gates.

## §1. Subject of submission

- Drug: `<drug_name>`
- Model: `<model_name>`

## §2. Model card

Compartmental PBPK system: `<model_name>`

## §3. Parameter priors

| name | units | value | rel_u | confidence |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

## §4. Numerical method

- Integrator: `<method_name>` (Tsit5 adaptive in current Sounio stdlib)
- abs_tol: `<abs_tol>`
- rel_tol: `<rel_tol>`
- h_min: `<h_min>`
- h_max: `<h_max>`

## §5. ISO 17025 GUM budget (1st order)

Pre-parsed from `benchmarks/pbpk/gum_budget.csv` (Phase Y).

| source | A/B | u_i | contribution |
|---|---|---|---|
| ... | ... | ... | ... |

- combined u_c: `<u_c_o1>`
- expanded U_95: `<u95_o1>`

## §6. ISO 17025 GUM budget (2nd order)

Optional. Pre-parsed from `benchmarks/pbpk/hessian_budget.csv` if Lane 8a's
Hessian propagator has run; otherwise the section emits an explicit
"_(Hessian budget not available in this run.)_" line.

## §7. Confidence gate evidence (Phase J)

- min_conf threshold: `<phase_j_min_conf>`
- observed confidence: `<phase_j_observed>`
- verdict: ADMIT | REFUSE

## §8. Clinical validation

Read from `stdlib/darwin_pbpk/validation/rapamycin_clinical.sio` outputs.

| endpoint | units | sounio | reference | pct_diff |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

## §9. Audit trail

- commit_sha: `<commit_sha>`
- generated_at_utc: `<generated_at_utc>`
- sounio_version: `<sounio_version>`
- claim_control: `docs/dissertation/pbpk_claim_truth_table.md`

## PDF rendering

The generator emits Markdown only. To produce a PDF dossier suitable for
regulatory submission, pipe the output through `pandoc`:

```bash
./bin/souc compile tests/run-pass/dossier_smoke.sio -o /tmp/dossier && \
  /tmp/dossier | pandoc -f markdown -o dossier.pdf
```
