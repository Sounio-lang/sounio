<!-- docs:meta
topic_id: repo.docs.audit.a-mu-wp25-dd-hvp-lo-absent-2026-08-23
authority: repo_only
audience: users
last_validated: 2026-08-23
validated_by: grok-cli2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.a-mu-wp25-dd-hvp-lo-absent-2026-08-23
-->

# WP25 data-driven HVP LO — typed absence, not a fabricated pin

```text
Semantic-Lane-ID: wp25-dd-hvp-lo-absent-20260823
Owner: grok-cli2
Concept-IDs: SOUNIO-EPISTEMIC-NUMERIC-VALUE
Intent-Preserved: uncertainty is not ignorance; a missing citation is not
  a latent f64; Sounio does not compute lattice HVP
Transformation: WP25 data-driven HVP LO is a distinct type
  Wp25DdHvpLoAbsent { provided: false } rather than Epistemic; Table 5
  is a bool citation; the hybrid lattice+e⁺e⁻ path is unchanged
Types-Changed: Wp25DdHvpLoAbsent added
Effects-Changed: none
IR-Changed: none
Claims-Introduced: Sounio can hold "WP25 has no data-driven HVP LO" without
  inventing 6931, 7132, or 7045; Madaros rejects the slot as f64 (E008)
  and as a locally-resolved pin-shaped struct (E009)
Claims-Forbidden: Sounio computed HVP; new physics; the tension is
  resolved; 0.0 is the WP25 data-driven LO; Unobserved<f64> is inhabited
  for this slot under Madaros; imported Epistemic is a closed type
  (it is a pre-existing Unknown sink: even `1.0` typechecks as
  `-> Epistemic`); lean_single is a claim oracle
Assumptions: TI WP25 Table 5 "Estimates not provided at this point" is
  the citation for absence. WP25 Eq. (9.1) hybrid pins are unchanged.
  ADR-008: Madaros is the only semantic clock for this claim.
Write-Set: stdlib/particle_physics/ew_precision.sio,
  stdlib/particle_physics/mod.sio,
  examples/particle_physics/a_mu_wp25_dd_hvp_lo_absent.sio,
  tests/run-pass/a_mu_wp25_dd_hvp_lo_absent.sio,
  tests/compile-fail/a_mu_wp25_dd_hvp_lo_observe_is_not_a_float.sio,
  tests/compile-fail/a_mu_wp25_dd_hvp_lo_cannot_be_a_pin.sio,
  this file
Read-Set: docs/audit/A_MU_GUM_SPLIT_2026-08-23.md,
  docs/decisions/adr-008-claim-oracle-semantic-clock.md
Positive-Witness: Madaros souc run prints WP25_DD_HVP_LO_PROVIDED 0 and
  HVP_ASSEMBLED 7044.8; VERDICT_ABSENT 1; VERDICT_HYBRID_INTACT 1
Negative-Witness: Madaros E008 collapsing the slot to f64; Madaros E009
  feeding the slot to a local pin-shaped struct (Epistemic field layout)
Acceptance-Gate: run-pass exit 0 under Madaros; both Madaros typecheck-fail
  tests fail with pinned error-pattern Wp25DdHvpLoAbsent
Integration-Target: origin/main
Authoritative-Only-If: produced by Madaros running Sounio
```

## Why not `Unobserved<f64>` in stdlib

Madaros `souc check` of `ew_precision.sio` rejects

```
pub fn a_mu_hvp_lo_datadriven_wp25() -> Unobserved<f64> { 0.0 }
```

with E008 (expected Unobserved, found f64). Putting 0.0, 6931, 7132, or
7045 inside Unobserved would also have been a fake central. lean_single
is the bootstrap seed, not a claim oracle for this lane.

The Madaros-authoritative slot is a struct with **no HVP number field**.
Collapsing it to f64 is E008. Feeding it to a locally-resolved pin-shaped
struct (`LocalPin { val, variance, confidence }`) is E009.

Imported `Epistemic` is a **pre-existing Unknown sink** under Madaros:
`fn forge() -> Epistemic { 1.0 }` typechecks. That is not special to this
slot. The live fix site is `self-hosted/check/check.sio` type-position
lookup (imported names bound as TyUnknown). That file is claimed by
another lane (`ns-wire`); this PR does not touch it.

## Receipts (2026-08-23)

Control ELF `/workspace/sounio/bin/madaros-linux-x86_64` (100902241 B).
All receipts below are Madaros.

`souc check` of `ew_precision.sio`, the example, and the run-pass:
**verdict=0**.

`souc run examples/particle_physics/a_mu_wp25_dd_hvp_lo_absent.sio`: **rc=0**

```
WP25_DD_HVP_LO_PROVIDED 0
HVP_ASSEMBLED 7044.799999
HVP_PUBLISHED 7045.000000
HVP_LO_WP20 6931.000000
HVP_LO_LATTICE_WP25 7132.000000
VERDICT_ABSENT 1
VERDICT_HYBRID_INTACT 1
```

`souc run tests/run-pass/a_mu_wp25_dd_hvp_lo_absent.sio`: **rc=0**.

Madaros `souc check` of
`tests/compile-fail/a_mu_wp25_dd_hvp_lo_observe_is_not_a_float.sio`:
**verdict=1**, `expected f64 found Wp25DdHvpLoAbsent`.

Madaros `souc check` of
`tests/compile-fail/a_mu_wp25_dd_hvp_lo_cannot_be_a_pin.sio`:
**verdict=1**, `expected LocalPin found Wp25DdHvpLoAbsent` (E009).

LLM-offload math-review:

- Audit file: xAI grok-4.3 **NO MATHEMATICAL CONTENT TO REVIEW** (type-level
  absence, no new formula). Z.AI quota skip until 2026-08-25.
- `ew_precision.sio` whole-file fan-out (`/tmp/llm-offload-6XxE9b`):
  `[OK]` on `a_mu_pull` and `a_mu_exp_combined_var`. Other `[WRONG]` /
  `[OVERREACH]` flags are pre-existing oblique / \(M_W\) / PDG-pin comments
  **not edited this lane** and are not accepted as this patch's defect.
  Outcome: **PASS_SINGLE_PROVIDER_DEGRADED** for the absence claim.
  Shared `.claude/llm_offload_log.md` was held by another active claim.
