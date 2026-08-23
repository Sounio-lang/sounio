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
  inventing 6931, 7132, or 7045; collapsing the slot to f64 is E008
Claims-Forbidden: Sounio computed HVP; new physics; the tension is
  resolved; 0.0 is the WP25 data-driven LO; Madaros wraps literals into
  Unobserved<T> in this stdlib module; returning the slot as Epistemic
  is rejected (it is not, under current Madaros)
Assumptions: TI WP25 Table 5 "Estimates not provided at this point" is
  the citation for absence. WP25 Eq. (9.1) hybrid pins are unchanged.
Write-Set: stdlib/particle_physics/ew_precision.sio,
  stdlib/particle_physics/mod.sio,
  examples/particle_physics/a_mu_wp25_dd_hvp_lo_absent.sio,
  tests/run-pass/a_mu_wp25_dd_hvp_lo_absent.sio,
  tests/compile-fail/a_mu_wp25_dd_hvp_lo_observe_is_not_a_float.sio,
  tests/compile-fail/a_mu_wp25_dd_hvp_lo_unobserved_not_a_pin.sio,
  this file
Read-Set: docs/audit/A_MU_GUM_SPLIT_2026-08-23.md,
  tests/compile-fail/observe_return_boundary.sio
Positive-Witness: souc run prints WP25_DD_HVP_LO_PROVIDED 0 and
  HVP_ASSEMBLED 7044.8; VERDICT_ABSENT 1; VERDICT_HYBRID_INTACT 1
Negative-Witness: collapsing the slot to f64 is E008 (Madaros);
  Unobserved<f64> cannot return as f64 without Observe (lean_single)
Acceptance-Gate: run-pass exit 0 under Madaros; both compile-fail tests
  fail with their pinned error-pattern
Integration-Target: origin/main
Authoritative-Only-If: the bool and the hybrid sum are produced by
  Madaros running Sounio; the Unobserved compile-fail is lean_single
```

## Why not `Unobserved<f64>` in stdlib

Madaros `souc check` of `ew_precision.sio` rejects

```
pub fn a_mu_hvp_lo_datadriven_wp25() -> Unobserved<f64> { 0.0 }
```

with E008 (expected Unobserved, found f64). The implicit wrap that
`tests/run-pass/unobserved_basic.sio` uses is lean_single. Putting 0.0
inside Unobserved would also have been a fake central.

So the Madaros-authoritative slot is a struct with **no HVP number
field**. The Unobserved observation-boundary is a self-contained
lean_single compile-fail, not a stdlib pin.

Madaros currently **accepts** `fn as_pin(x: Wp25DdHvpLoAbsent) -> Epistemic { x }`.
This lane does not claim that hole is closed. It claims the slot is not
a float and Table 5 is `provided = false`.

## Receipts (2026-08-23)

Control ELF `/workspace/sounio/bin/madaros-linux-x86_64` (100902241 B).

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

lean_single `souc check` of
`tests/compile-fail/a_mu_wp25_dd_hvp_lo_unobserved_not_a_pin.sio`:
**typecheck failed**, `cannot return Unobserved<T> as non-Unobserved type`.

LLM-offload math-review:

- Audit file: xAI grok-4.3 **NO MATHEMATICAL CONTENT TO REVIEW** (type-level
  absence, no new formula). Z.AI quota skip until 2026-08-25.
- `ew_precision.sio` whole-file fan-out (`/tmp/llm-offload-6XxE9b`):
  `[OK]` on `a_mu_pull` and `a_mu_exp_combined_var`. Other `[WRONG]` /
  `[OVERREACH]` flags are pre-existing oblique / \(M_W\) / PDG-pin comments
  **not edited this lane** and are not accepted as this patch's defect.
  Outcome: **PASS_SINGLE_PROVIDER_DEGRADED** for the absence claim.
  Shared `.claude/llm_offload_log.md` was held by another active claim.
