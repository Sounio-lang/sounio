# PBPK Dossier — Template

This is the human-readable section skeleton emitted by
`scripts/dissertation/dossier_generator.sio`. The generator is
driven from a `DossierInput` struct (no file I/O) and prints
Markdown matching this layout. The smoke test in
`tests/run-pass/dossier_smoke.sio` shows a representative input.

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

## §10. Live interactive viewer

A 3D, browser-native interactive view of this model is hosted at:

> **https://www.souniolang.org/dissertation/**

The viewer renders the same 14 compartments, the Cypher elution timeline,
the GUM uncertainty cone, the Phase J confidence-gate verdict, and the
2nd-order Hessian heatmap reported above — all driven by an in-browser
RK4 integrator that is parity-locked to the Sounio `tsit5_pbpk14`
reference solver by `scripts/ci/dissertation_frontend_parity_gate.sh`
(< 1% RMSE per compartment).

Three narrated tours are available from the side panel:

1. **Cypher → blood → liver** (30 s) — drug-release kinetics
2. **BBB closeup** (20 s) — Kp_brain = 0.10, P-gp efflux
3. **GUM cone widening under CL_hep variability** (20 s) — direct
   visual proof of contribution #1 (GUM-through-ODE)

For committee handouts the viewer's `📸 Snapshot PNG` button (or the
`S` keyboard shortcut) exports a print-quality PNG of the current frame.

## PDF rendering

The generator emits Markdown only. To produce a PDF dossier suitable for
regulatory submission, pipe the output through `pandoc`:

```bash
./bin/souc compile tests/run-pass/dossier_smoke.sio -o /tmp/dossier && \
  /tmp/dossier | pandoc -f markdown -o dossier.pdf
```
