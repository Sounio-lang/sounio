<!-- docs:meta
topic_id: repo.docs.audit.a-mu-cmd3-2pi-split-2026-08-23
authority: repo_only
audience: users
last_validated: 2026-08-23
validated_by: grok-cli2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.a-mu-cmd3-2pi-split-2026-08-23
-->

# CMD-3 vs pre-CMD-3 2π LO HVP — the split WP25 refused to average

```text
Semantic-Lane-ID: a-mu-cmd3-2pi-split-20260823
Owner: grok-cli2
Concept-IDs: SOUNIO-EPISTEMIC-NUMERIC-VALUE
Intent-Preserved: two incompatible 2π evaluations must not be averaged
  into one data-driven HVP LO; a missing WP25 citation is not a latent
  f64; Sounio does not integrate σ(e⁺e⁻→π⁺π⁻)
Transformation: pin the two 2π numbers published in the CMD-3 Letter as
  Epistemic; GUM pull via ep_sub; keep Wp25DdHvpLoAbsent unchanged
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: Sounio can hold CMD-3 2π and the pre-CMD-3 2π average
  without collapsing them; the GUM pull of those pins is computed in
  Sounio; WP25 data-driven HVP LO remains absent
Claims-Forbidden: Sounio computed HVP; new physics; the tension is
  resolved; CMD-3 is the WP25 data-driven LO; 5260 is full HVP LO;
  averaging 5260 with 5060 yields a TI combination; one SM a_μ;
  Madaros is fixed-point-verified
Assumptions: published centrals and quoted σ are citations
  (Ignatov et al., PRL 132, 231903 (2024), arXiv:2309.12910).
  CMD-3 5260(42) uses CMD-3 data for √s = 0.327–1.2 GeV and the
  average of other measurements outside that window. The 42 is
  systematics-dominated. The 5060(34) is the same Letter's citation
  of the WP20 average of previous measurements with χ² inflation
  (Aoyama et al. 2020). Independent GUM overstates σ_diff because
  the pins share some out-of-window data; a surviving k95 split is
  therefore conservative. Units 10⁻¹¹.
Write-Set: stdlib/particle_physics/ew_precision.sio,
  stdlib/particle_physics/mod.sio,
  examples/particle_physics/a_mu_cmd3_2pi_split.sio,
  tests/run-pass/a_mu_cmd3_2pi_split.sio, this file
Read-Set: stdlib/epistemic/knowledge.sio,
  docs/audit/A_MU_GUM_SPLIT_2026-08-23.md,
  docs/audit/A_MU_WP25_DD_HVP_LO_ABSENT_2026-08-23.md,
  docs/decisions/adr-008-claim-oracle-semantic-clock.md
Positive-Witness: souc run of a_mu_cmd3_2pi_split.sio prints pull
  > k95 and WP25_DD_HVP_LO_PROVIDED 0
Negative-Witness: no a_mu_hvp_lo_2pi_combined; g_minus_2_muon_leading
  still returns a collapsed f64
Acceptance-Gate: tests/run-pass/a_mu_cmd3_2pi_split.sio exit 0 under
  Madaros; example exit 0
Integration-Target: origin/main
Authoritative-Only-If: the pull is produced by Madaros running Sounio,
  not by a peer runtime
```

## What this is

WP25 (Aliberti et al., Phys. Rep. 1143 (2025) 1) will not quote a combined
data-driven LO HVP. The reason it gives is the CMD-3 \(e^+e^-\to\pi^+\pi^-\)
measurement, which "has increased the tensions among data-driven
dispersive evaluations of the LO HVP contribution to a level that makes
it impossible to combine the results in a meaningful way."

This lane pins the comparison **as published by CMD-3**, not a WP25
combination (there is none) and not a Sounio integral.

| pin | value (\(10^{-11}\)) | source |
|---|---:|---|
| `a_mu_hvp_lo_2pi_cmd3` | 5260(42) | CMD-3 PRL 132, 231903 (2024) |
| `a_mu_hvp_lo_2pi_pre_cmd3` | 5060(34) | same Letter, citing WP20 + χ² inflation |

Independent GUM: \(\sigma=\sqrt{42^2+34^2}=\sqrt{2920}\), pull \(=200/\sigma>3\).
k95 = 1.96. The split is the falsifier: if \(|\mathrm{pull}|\le 1.96\), the
claim is dead.

## What this is not

- Full HVP LO. Lattice WP25 LO is 7132(61) for the **full** channel sum.
  Adding 5260 to 7132 is a category error; the test refuses a 100-unit
  coincidence.
- WP25 data-driven HVP LO. That slot is still `Wp25DdHvpLoAbsent`.
- A BaBar-vs-KLOE table. WP25 Fig. 26 / Table 5 (section 2.11) illustrates
  the spread across experiments and methods in units \(10^{-10}\) and
  **does not** derive a global \(e^+e^-\) number. Those per-experiment
  rows were not extracted as machine-readable text here; inventing them
  would be fabrication. The CMD-3 Letter's own pair is the citation.

## Receipts (2026-08-23)

Control ELF `/workspace/sounio/bin/madaros-linux-x86_64` (100902241 B),
`SOUNIO_MADAROS_BIN` override. Worktree committed ELF is 99964676 B and
is not this receipt.

Madaros `souc check` of `ew_precision.sio`, the example, and the test:
**verdict=0**.

Madaros `souc run` of `examples/particle_physics/a_mu_cmd3_2pi_split.sio`
and `tests/run-pass/a_mu_cmd3_2pi_split.sio`: **rc=0**.

```
A_MU_UNIT 1e-11
A_MU_2PI_CMD3_VAL 5260.000000
A_MU_2PI_CMD3_STD 42.000000
A_MU_2PI_PRE_VAL 5060.000000
A_MU_2PI_PRE_STD 34.000000
A_MU_2PI_PULL_CMD3_PRE 3.701166
A_MU_K95 1.960000
WP25_DD_HVP_LO_PROVIDED 0
VERDICT_2PI_SPLIT_SURVIVES 1
VERDICT_WP25_DD_LO_STILL_ABSENT 1
```

\(\sqrt{42^2+34^2}=\sqrt{2920}\approx 54.037\); \(200/54.037=3.701166\).
Sounio computed the pull. Sounio did not compute the cross section.

This does not claim the imported-module native path is closed for every
graph. It claims this four-module 2π graph runs.

LLM-offload math-review (`bin/llm-offload -t math-review -i` this file,
`/tmp/llm-offload-n3fN28`):

- xAI grok-4.3: four `[OK]` (σ_diff, pull, conservative independent-GUM
  caveat, no WP25 combination). No `[WRONG]`.
- Z.AI: weekly quota exhausted until 2026-08-25 06:34:36 (code 1310).
- Outcome: **PASS_SINGLE_PROVIDER_DEGRADED**.
- `.claude/llm_offload_log.md` is claimed by ns-wire; this lane does not
  write it.
