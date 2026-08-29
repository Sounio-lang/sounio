<!-- docs:meta
topic_id: repo.docs.audit.pbpk14-identifiability-2026-06-14
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.pbpk14-identifiability-2026-06-14
-->

# PBPK-14 structural identifiability from plasma (future-work / Contribution #2)

**Date:** 2026-06-14
**Artifact:** `examples/ode/pbpk14_identifiability.sio` (self-contained, runs on
`bin/souc-linux-x86_64 <src> <out>`)
**Closes:** future-work item "identifiability"; supplies the formal motivation for
dissertation **Contribution #2** (type-enforced `Knowledge<T>` epistemic priors).

## Result

The 14-compartment PBPK is **structurally non-identifiable** from a plasma (venous)
concentration curve. The unbound fraction `fu` enters the dynamics **only** through the
elimination terms

```
hepatic:  cl_int * fu * c_liver      renal:  gfr * fu * c_art
```

so the reparameterisation `fu → a·fu, cl_int → cl_int/a, gfr → gfr/a` leaves both
products `(cl_int·fu)` and `(gfr·fu)` — hence the **entire** system matrix `A` and every
trajectory — invariant. Only the two products are identifiable; `fu`, `cl_int`, `gfr`
individually are not.

## Evidence (reproducible, on the stiff backward-Euler solver, real flows)

| test | result | meaning |
| --- | --- | --- |
| A. null-direction reparam (fu·2, cl_int/2, gfr/2) | `max|Δplasma| = 0.000000` | curves identical to machine precision ⇒ non-identifiable combo |
| B. identifiable direction (cl_int +20%) | `max|Δplasma| = 0.36` | the product `cl_int·fu` is identifiable |
| C. Fisher info for (ln fu, ln cl_int, ln gfr) | `det(FIM)=5.6e-12`, `trace=30.8` | `det/trace³ ≈ 1e-16` ⇒ **rank-deficient** |
| C. null-direction kernel check | `|FIM·(+1,−1,−1)|₁ = 5e-4 ≈ 0` | null direction lies in the FIM kernel |
| C. identifiable 2×2 block | `det = 0.032 ≫ 0` | the remaining directions are well-conditioned |

The Fisher information is built from central log-sensitivities of the 13-point plasma
curve (`F = SᵀS`); it has rank 2 (two identifiable products) and a 1-D null space (the
`fu`/`cl_int`/`gfr` trade-off).

## Why this matters for the dissertation

Identifiability is a property of the **model + observation**, not of the optimiser: no
amount of plasma data can separate `fu` from the clearances. The missing dimension must
come from **prior information**. This is the formal justification for the type-enforced
epistemic priors (`Knowledge<T>`, PBPK28) — they are *mandatory*, not decorative, because
the data structurally cannot supply the missing constraint. It also bounds honest
claims: a fitted `fu` (or `cl_int`) reported without a prior is meaningless; only the
products, or a prior-pinned value, are defensible.

## Scope / caveats

- Demonstrated for the `fu ↔ (cl_int, gfr)` trade-off, the cleanest and provable
  structural non-identifiability. Other near-non-identifiabilities (tissue Kp vs volume,
  flow vs partition) are *practical* (data-precision-limited) rather than exactly
  structural and are not claimed here.
- The plasma observation is venous mass/volume; richer observations (tissue biopsy,
  multiple matrices) would change the identifiable set — itself a useful design point.

Companion: `PBPK14_MODELFORM_STIFF_REPAIR_2026-06-14.md` (the stiff integrator this
analysis runs on).
