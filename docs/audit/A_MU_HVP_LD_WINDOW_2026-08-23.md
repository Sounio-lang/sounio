<!-- docs:meta
topic_id: repo.docs.audit.a-mu-hvp-ld-window-2026-08-23
authority: repo_only
audience: users
last_validated: 2026-08-23
validated_by: grok-cli2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.a-mu-hvp-ld-window-2026-08-23
-->

# Long-distance window \(a_\mu^{\mathrm{LD}}(ud)\) — split smaller than W, larger than SD

```text
Semantic-Lane-ID: a-mu-hvp-ld-window-20260823
Owner: grok-cli2
Concept-IDs: SOUNIO-EPISTEMIC-NUMERIC-VALUE
Intent-Preserved: GUM pulls of published same-kernel pairs may be
  ordered; Sounio does not compute lattice QCD; WP25 data-driven HVP
  LO remains absent
Transformation: pin WP25 Eq. (3.27) lattice a_μ^LD(ud) and Eq. (2.43)
  KNT a_μ^LD(ud) as Epistemic; require |pull_LD| > k95 and
  |pull_W| > |pull_LD| > |pull_SD|
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: Sounio can hold the LD lattice and KNT pins without
  collapsing them; their GUM pull is computed in Sounio and exceeds
  k95; it is smaller than the W(ud) pull and larger than the SD pull
Claims-Forbidden: Sounio computed HVP; new physics; the tension is
  resolved; 4060 is full HVP LO; LD agreement; W is the only split;
  Madaros is fixed-point-verified
Assumptions: published centrals and quoted σ are citations (Aliberti
  et al., Phys. Rep. 1143 (2025) 1). Lattice: Eq. (3.27)
  406.0(4.9)×10⁻¹⁰; FLAG average with χ² rescaling 1.6 plus extra
  3.1×10⁻¹⁰ systematic; w0 scheme uncertainty is not in (4.9).
  Data-driven: Eq. (2.43) 389.9(1.7)×10⁻¹⁰ pre-CMD-3 KNT (Benton et
  al. 2025). Stored as 10⁻¹¹ (×10). W and LD pins are ud-connected;
  SD pins are full-flavour. Pull ordering is across those published
  pairs. Independent GUM. CMD-3 window replacement not pinned.
Write-Set: stdlib/particle_physics/ew_precision.sio,
  stdlib/particle_physics/mod.sio,
  examples/particle_physics/a_mu_hvp_ld_window.sio,
  tests/run-pass/a_mu_hvp_ld_window.sio, this file
Read-Set: docs/audit/A_MU_HVP_WINDOW_UD_2026-08-23.md,
  docs/audit/A_MU_HVP_SD_CONTROL_2026-08-23.md,
  docs/decisions/adr-008-claim-oracle-semantic-clock.md
Positive-Witness: souc run prints |pull_LD| > 1.96 and
  |pull_W| > |pull_LD| > |pull_SD|
Negative-Witness: full lattice LO remains 7132; DD LO still Absent
Acceptance-Gate: tests/run-pass/a_mu_hvp_ld_window.sio exit 0 under
  Madaros
Integration-Target: origin/main
Authoritative-Only-If: the pulls are produced by Madaros running Sounio
```

## Why LD

SD agrees (pull 1.24). W(ud) splits (pull 6.79). If LD also agreed, the
discrepancy would be only the intermediate window. It does not: Eq.
(3.27) vs Eq. (2.43) is a GUM split, smaller than W, larger than SD.

| pin | WP25 (10⁻¹⁰) | stored (10⁻¹¹) |
|---|---:|---:|
| `a_mu_hvp_ld_ud_lattice_wp25` | Eq. (3.27) 406.0(4.9) | 4060.0(49.0) |
| `a_mu_hvp_ld_ud_knt_pre_cmd3` | Eq. (2.43) 389.9(1.7) | 3899.0(17.0) |

Independent GUM: \(\sigma=\sqrt{49^2+17^2}=\sqrt{2690}\),
pull \(=161/\sigma\approx 3.10>1.96\).

Falsifier: if \(|\mathrm{pull_{LD}}|\le 1.96\), the LD-split claim is
dead. If \(|\mathrm{pull_W}|\le|\mathrm{pull_{LD}}|\), the “largest in
W” ordering is dead.

## Receipts (2026-08-23)

Control ELF `/workspace/sounio/bin/madaros-linux-x86_64` (100902241 B).

Madaros `souc check` verdict=0. Example and test **rc=0**.

```
A_MU_LD_UD_LAT_VAL 4060.000000
A_MU_LD_UD_KNT_VAL 3899.000000
A_MU_LD_UD_PULL_LAT_KNT 3.104200
A_MU_W_UD_PULL_LAT_KNT 6.789189
A_MU_SD_PULL_LAT_RR 1.242103
VERDICT_LD_SPLIT_SURVIVES 1
VERDICT_PULL_ORDER_W_LD_SD 1
WP25_DD_HVP_LO_PROVIDED 0
```

\(\sqrt{49^2+17^2}=\sqrt{2690}\approx 51.865\); \(161/51.865=3.104200\).

LLM-offload math-review (`/tmp/llm-offload-erbDnh`):

- xAI grok-4.3: three `[OK]` (σ and pull, W>LD>SD order, independent GUM).
- Z.AI: weekly quota exhausted until 2026-08-25.
- Outcome: **PASS_SINGLE_PROVIDER_DEGRADED**.
