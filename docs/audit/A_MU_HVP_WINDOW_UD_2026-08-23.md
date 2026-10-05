<!-- docs:meta
topic_id: repo.docs.audit.a-mu-hvp-window-ud-2026-08-23
authority: repo_only
audience: users
last_validated: 2026-08-23
validated_by: grok-cli2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.a-mu-hvp-window-ud-2026-08-23
-->

# Intermediate window \(a_\mu^W(ud)\) — lattice vs KNT, same quantity

```text
Semantic-Lane-ID: a-mu-hvp-window-ud-20260823
Owner: grok-cli2
Concept-IDs: SOUNIO-EPISTEMIC-NUMERIC-VALUE
Intent-Preserved: two evaluations of the same Euclidean window must not
  be averaged into one HVP LO; a missing WP25 data-driven LO is not a
  latent f64; Sounio does not compute lattice QCD
Transformation: pin WP25 Table 8 last-row lattice a_μ^W(ud) and
  Eq. (2.43) pre-CMD-3 KNT a_μ^W(ud) as Epistemic; GUM pull via ep_sub
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: Sounio can hold the lattice and KNT window-ud pins
  without collapsing them; the GUM pull of those pins is computed in
  Sounio
Claims-Forbidden: Sounio computed HVP; new physics; the tension is
  resolved; 2069.7 is full HVP LO; this is WP25 data-driven HVP LO;
  the exploratory CMD-3 window replacement is a TI combination;
  Madaros is fixed-point-verified
Assumptions: published centrals and quoted σ are citations (Aliberti
  et al., Phys. Rep. 1143 (2025) 1). Lattice: Table 8 last row
  206.97(41)×10⁻¹⁰ in isoQCD. Data-driven: Eq. (2.43) 199.0(1.1)×10⁻¹⁰
  pre-CMD-3 purely KNT-based (Benton et al. 2025). Stored here as
  10⁻¹¹ (×10) to match the other a_μ pins. Same RBC/UKQCD window
  (t0=0.4 fm, t1=1.0 fm, Δ=0.15 fm), isospin-limit ud-connected.
  Independent GUM: the methods do not share a dataset. The CMD-3
  replacement in the same paragraph is labelled exploratory and is
  not pinned.
Write-Set: stdlib/particle_physics/ew_precision.sio,
  stdlib/particle_physics/mod.sio,
  examples/particle_physics/a_mu_hvp_window_ud.sio,
  tests/run-pass/a_mu_hvp_window_ud.sio, this file
Read-Set: stdlib/epistemic/knowledge.sio,
  docs/audit/A_MU_GUM_SPLIT_2026-08-23.md,
  docs/audit/A_MU_CMD3_2PI_SPLIT_2026-08-23.md,
  docs/decisions/adr-008-claim-oracle-semantic-clock.md
Positive-Witness: souc run prints pull > k95 and WP25_DD_HVP_LO_PROVIDED 0
Negative-Witness: no combined window function; full lattice LO remains 7132
Acceptance-Gate: tests/run-pass/a_mu_hvp_window_ud.sio exit 0 under Madaros
Integration-Target: origin/main
Authoritative-Only-If: the pull is produced by Madaros running Sounio
```

## Why this quantity

WP25 will not quote a combined data-driven *full* LO HVP. The Euclidean
intermediate window is the quantity both methods actually publish for a
common kernel. The split lives in the light-quark connected piece, not
in the short-distance window.

| pin | WP25 (10⁻¹⁰) | stored (10⁻¹¹) |
|---|---:|---:|
| `a_mu_hvp_window_ud_lattice_wp25` | 206.97(41) Table 8 last row | 2069.7(4.1) |
| `a_mu_hvp_window_ud_knt_pre_cmd3` | 199.0(1.1) Eq. (2.43) | 1990.0(11.0) |

Independent GUM: \(\sigma=\sqrt{4.1^2+11.0^2}=\sqrt{137.81}\),
pull \(=79.7/\sigma>5\). k95 = 1.96.

Not Eq. (3.26) full-flavour \(a_\mu^W=236.58(43)\times 10^{-10}\). That
includes s, c, disconnected, and IB. Adding 2069.7 to 7132 is a
category error.

## Receipts (2026-08-23)

Control ELF `/workspace/sounio/bin/madaros-linux-x86_64` (100902241 B).

Madaros `souc check` of `ew_precision.sio`, the example, and the test:
**verdict=0**.

Madaros `souc run` of the example and the test: **rc=0**.

```
A_MU_UNIT 1e-11
A_MU_W_UD_LAT_VAL 2069.699999
A_MU_W_UD_LAT_STD 4.099999
A_MU_W_UD_KNT_VAL 1990.000000
A_MU_W_UD_KNT_STD 11.000000
A_MU_W_UD_PULL_LAT_KNT 6.789189
A_MU_K95 1.960000
WP25_DD_HVP_LO_PROVIDED 0
VERDICT_WINDOW_UD_SPLIT_SURVIVES 1
VERDICT_WP25_DD_LO_STILL_ABSENT 1
```

\(\sqrt{4.1^2+11.0^2}=\sqrt{137.81}\approx 11.739\); \(79.7/11.739=6.789189\).
Sounio computed the pull. Sounio did not compute the window.

The printed 2069.699999 is the f64 encoding of the citation 2069.7.
The test matches the citation at \(10^{-9}\) relative to the stored literal.

LLM-offload math-review (`/tmp/llm-offload-NUWA8K`):

- xAI grok-4.3: four `[OK]` (difference, σ, pull, category vs 7132 / Eq. 3.26).
- Z.AI: weekly quota exhausted until 2026-08-25.
- Outcome: **PASS_SINGLE_PROVIDER_DEGRADED**.
- `.claude/llm_offload_log.md` is claimed by ns-wire; this lane does not write it.
