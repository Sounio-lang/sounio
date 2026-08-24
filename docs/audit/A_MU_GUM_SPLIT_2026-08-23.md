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
**verdict=0**.

**Re-run 2026-08-23 after Madaros native fix** (control ELF
`bin/madaros-linux-x86_64` 100902241 B). Default Madaros `souc run`
of the imported example and of `tests/run-pass/a_mu_gum_split.sio`:
**rc=0**. Combined experimental σ is now `sqrt(stat²+syst²+ext²)`
passed into `Epistemic::measured` — a computed f64, not a decimal
literal. Probe: `measured(central, newton_sqrt(11.4²+9.1²+2.1²))`
keeps σ = 14.737028 on both the Newton value and `ep_std`.

```
A_MU_EXP_VAL 116592070.500000   STD 14.737028
A_MU_WP20_VAL 116591810.000000  STD 43.000000
A_MU_WP25_VAL 116592033.000000  STD 62.000000
A_MU_PULL_EXP_WP20 5.730909
A_MU_PULL_EXP_WP25 0.588443
A_MU_PULL_WP25_WP20 2.955522
VERDICT_EXP_DISAGREES_WP20 1
VERDICT_EXP_AGREES_WP25 1
VERDICT_SPLIT_SURVIVES 1
```

`delta_alpha_had` needed `with Mut, Div, Panic` so Madaros could check the
module at all (E035 on `measured`). That is one signature, not a mass migrate.

This does **not** claim the imported-module native path is closed for every
graph. It claims this four-module a_μ graph runs.

## Sector budget (Madaros run, same ELF)

Sounio *sums* published QED, EW, HVP, HLbL. It does not compute HVP.

```
WP20  QED 116584718.931  EW 153.600  HVP 6845.100  HLbL 92.000  SUM 116591809.631
WP25  QED 116584718.800  EW 154.400  HVP 7045.000  HLbL 115.500 SUM 116592033.700
shift QED −0.131  EW +0.800  HVP +199.900  HLbL +23.500  SUM +224.069
published total shift +223.000
```

The fork is **mostly HVP**, not only HVP. HLbL moved 23.5 units. Claims-forbidden
still includes “Sounio computed lattice HVP” and “the tension is resolved”.

## WP25 HVP is a hybrid (Madaros, same run)

Eq. (9.1): lattice LO + e⁺e⁻ NLO + e⁺e⁻ NNLO. There is **no** WP25
data-driven LO HVP (Table 5: estimates not provided).

```
HVP_LO_LATTICE 7132.000
HVP_NLO_EPEM   −99.600
HVP_NNLO_EPEM   12.400
HVP_ASSEMBLED 7044.800
HVP_PUBLISHED 7045.000
HVP_LO_WP20    6931.000
PULL_HVP_LO_LATTICE_VS_WP20 2.755
VERDICT_HVP_HYBRID 1
VERDICT_HVP_LO_SPLIT 1
```

```text
Semantic-Outcome: two SM a_μ citations are Epistemic values; GUM pulls
  are computed in Sounio under Madaros; they are not averaged. The SM
  is a sum of sectors; the WP20→WP25 shift is largest in HVP and not
  zero in HLbL.
```
