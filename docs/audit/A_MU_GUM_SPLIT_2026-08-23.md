<!-- docs:meta
topic_id: repo.docs.audit.a-mu-gum-split-2026-08-23
authority: repo_only
audience: users
last_validated: 2026-08-23
validated_by: grok-cli2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.a-mu-gum-split-2026-08-23
-->

# Muon \(a_\mu\) GUM split — two SM citations, not one float

```text
Semantic-Lane-ID: a-mu-gum-split-20260823
Owner: grok-cli2
Concept-IDs: SOUNIO-EPISTEMIC-NUMERIC-VALUE
Intent-Preserved: measurement uncertainty is not numerical error;
  two incompatible Standard Model predictions must not be averaged
  into one f64; Sounio does not compute lattice HVP
Transformation: pin Fermilab 2025, TI WP20, TI WP25 as Epistemic
  (units 10⁻¹¹); GUM pull via ep_sub; keep g_minus_2_muon_leading
  as the collapsed-f64 negative witness
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: Sounio can hold two SM a_μ citations without
  collapsing them; GUM pulls of those pins are computed in Sounio
Claims-Forbidden: Sounio computed HVP; new physics; the tension is
  resolved; there is one SM a_μ; raising precision solves it
Assumptions: published centrals and quoted σ are citations (Fermilab
  3 June 2025; TI WP20 2020; TI WP25 May 2025). Experimental stat,
  syst, and external pieces are combined in quadrature in Sounio.
  The two SM citations are treated as independent of each other and
  of the experiment for Var(X−Y)=Var(X)+Var(Y).
Write-Set: stdlib/particle_physics/ew_precision.sio,
  stdlib/particle_physics/mod.sio,
  examples/particle_physics/a_mu_gum_split.sio,
  tests/run-pass/a_mu_gum_split.sio, this file
Read-Set: stdlib/epistemic/knowledge.sio,
  stdlib/particle_physics/sm_params.sio,
  docs/decisions/adr-008-claim-oracle-semantic-clock.md
Positive-Witness: souc run of a_mu_gum_split.sio prints three pulls
  and named verdicts; WP20 ≠ WP25
Negative-Witness: g_minus_2_muon_leading still returns a single f64
  with exact 693e-10 hadronic VP
Acceptance-Gate: tests/run-pass/a_mu_gum_split.sio exit 0 under
  Madaros; example exit 0
Integration-Target: origin/main
Authoritative-Only-If: the pulls are produced by Madaros running
  Sounio, not by a peer runtime
```

## What was already there

`g_minus_2_qed_schwinger_ep` is the honest fragment: \(a_\mu^{(1)}=\alpha/2\pi\)
with GUM on \(\alpha\). `g_minus_2_muon_leading` is the hole: hadronic VP is
the exact constant `693.0e-10`.

## What this lane does

It does not compute HVP. It pins three published numbers as `Epistemic` and
lets Sounio compute the GUM pulls. Averaging WP20 with WP25 is a semantic
error; the functions exist so that averaging is a choice the caller would
have to write, not the default return.

## Falsifier

If after GUM the WP20–WP25 difference is consistent with 0 at k95 (`|pull| ≤
1.96`), the split claim is dead. The test encodes citation *order* and
*relative* pull size, not a fitted sigma from another language.

## Receipts (2026-08-23)

Madaros `souc check` of `ew_precision.sio`, the example, and the test:
**verdict=0**. Default Madaros `souc run` of the imported example **SIGSEGV
139** (imported-module native residual; same family as the particle expN
gates). Execution oracle for this lane is **lean_single**, matching
`scripts/ci/particle_exp7_gum_transfer_gate.sh`.

lean_single `souc run examples/particle_physics/a_mu_gum_split.sio` rc=0:

```
A_MU_EXP_VAL 116592070.500000   STD 14.737049
A_MU_WP20_VAL 116591810.000000  STD 43.000000
A_MU_WP25_VAL 116592033.000000  STD 62.000000
A_MU_PULL_EXP_WP20 5.730910
A_MU_PULL_EXP_WP25 0.588444
A_MU_PULL_WP25_WP20 2.955523
VERDICT_EXP_DISAGREES_WP20 1
VERDICT_EXP_AGREES_WP25 1
VERDICT_SPLIT_SURVIVES 1
```

`tests/run-pass/a_mu_gum_split.sio` lean_single rc=0.

`delta_alpha_had` needed `with Mut, Div, Panic` so Madaros could check the
module at all (E035 on `measured`). That is one signature, not a mass migrate.

Computed `f64` arguments to `Epistemic::measured` come back as σ=0 under
lean_single. Combined experimental σ is therefore a **literal**
(`14.7370485`) whose square is checked against `11.4²+9.1²+2.1²` in Sounio.

```text
Semantic-Outcome: two SM a_μ citations are Epistemic values; GUM pulls
  are computed in Sounio; they are not averaged. Madaros typechecks.
  Execution is lean_single until the imported native path stops
  SIGSEGVing this module graph.
```
