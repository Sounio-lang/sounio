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

## §10. Live interactive viewer — multi-drug A/B (Stage G)

A 3D, browser-native interactive view of this model is hosted at:

> **https://www.souniolang.org/dissertation/**

The Stage-G viewer is **drug-agnostic**: a chip-row selector at the top
of the canvas (or the `D` keyboard shortcut) toggles between
**rapamycin** (Cypher coronary stent — Higuchi diffusion release) and
**semaglutide** (subcutaneous depot — first-order absorption,
`k_a = ln 2 / 60 h`, `F = 0.89` after Overgaard 2019 / Carlsson 2020).

What swaps with the active drug:

| panel | rapamycin | semaglutide |
|---|---|---|
| release-source visual | Cypher coronary cylinder | lower-abdomen SC depot ellipsoid |
| receptor-occupancy bars | FKBP12 / mTORC1 in **liver, heart, gut** | GLP-1R in **brain, gut, pancreas** |
| PD readout panel | mTORC1 active fraction + neointimal index | plasma glucose (ΔG) + insulin (ΔI), linearised Bergman |
| Phase J evidence band | Ferron 1997 (n=24, CL CV=58%) + Cordis 2003 IFU | Overgaard 2019 (n=72, CL CV=15%) + Carlsson 2020 |
| patient-profile dropdown | typical / low CL / high CL / lean / obese | typical / slow CL / fast CL / lean / obese |
| release-scale slider | Higuchi K_H multiplier | SC dose multiplier (on 1 mg weekly) |
| snapshot PNG filename | `dissertation-rapamycin-<ts>.png` | `dissertation-semaglutide-<ts>.png` |

The PBPK kernel itself is **PBPK28** — permeability-limited, every organ
split into `(C_v, C_t)` with a PS coupling — plus per-organ TMDD blocks
(Mager 2004) and the drug's PD ODE. The fully-coupled
Crank-Nicolson step on the 27-state arrow matrix is parity-locked
(≤ 1 % RMSE per organ) against the Sounio reference solver by
`scripts/ci/dissertation_pbpk28_parity_gate.sh`, which runs **nine
hard cases** in a single pass:

1. Node ↔ Sounio rapamycin PBPK28 (organ-averages).
2. PBPK28-degenerate ↔ 1-state QSS analytical (reporting).
3. PBPK28-literature ↔ PBPK14 well-stirred (reporting — feeds the
   Type B model-form contribution into the §5 GUM budget per JCGM
   100:2008 §4.3).
4. Total-mass monotonic decay (12 / 12 samples).
5. Rapamycin TMDD parity at liver, heart, gut on `(R_free, DR)`.
6. Rapamycin PD parity on `(A, N)` at heart.
7. Node ↔ Sounio semaglutide PBPK28.
8. Semaglutide GLP-1R TMDD parity at brain, gut, pancreas.
9. Semaglutide glucose-insulin PD parity on `(ΔG, ΔI)` at pancreas.

### Tours (six total — three per drug)

Press `T` to cycle within the active drug; `D` cycles drug and resets
the tour selection.

**Rapamycin**:
1. **Cypher → blood → liver** (30 s) — Higuchi release kinetics and
   the Kp = 5.4 hepatic accumulation.
2. **BBB closeup — Kp = 0.10** (20 s) — P-gp efflux, why the BPR
   stays below 0.15 in the dissertation gate.
3. **GUM cone widening under CL_hep variability** (20 s) — direct
   visual proof of contribution #1 (GUM-through-ODE) under
   CYP3A4 poor-metaboliser perturbations.

**Semaglutide**:
4. **SC depot → blood → pancreas** (30 s) — `F · Dose · k_a · exp(-k_a·t)`
   absorption from the abdominal depot, slow systemic distribution.
5. **GLP-1R occupancy — brain, gut, pancreas** (24 s) — TMDD bars
   filling at three sites; appetite, satiety, and insulinotropic
   pathways narrated in turn.
6. **Bergman PD — ΔG falls as ΔI rises** (22 s) — pancreatic
   occupancy drives insulin secretion which suppresses plasma
   glucose; switches mid-tour to "slow CL" so the committee can see
   the PD GUM cone widen under exposure perturbation.

For committee handouts the viewer's `📸 Snapshot PNG` button (or the
`S` keyboard shortcut) exports a print-quality PNG of the current
frame, file-named with the active drug so multi-drug demos remain
unambiguous.

## PDF rendering

The generator emits Markdown only. To produce a PDF dossier suitable for
regulatory submission, pipe the output through `pandoc`:

```bash
./bin/souc compile tests/run-pass/dossier_smoke.sio -o /tmp/dossier && \
  /tmp/dossier | pandoc -f markdown -o dossier.pdf
```
