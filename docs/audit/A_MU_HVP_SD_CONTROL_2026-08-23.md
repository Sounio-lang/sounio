<!-- docs:meta
topic_id: repo.docs.audit.a-mu-hvp-sd-control-2026-08-23
authority: repo_only
audience: users
last_validated: 2026-08-23
validated_by: grok-cli2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.a-mu-hvp-sd-control-2026-08-23
-->

# Short-distance window — vanishing control for the W(ud) split

```text
Semantic-Lane-ID: a-mu-hvp-sd-control-20260823
Owner: grok-cli2
Concept-IDs: SOUNIO-EPISTEMIC-NUMERIC-VALUE
Intent-Preserved: a GUM split that fires on every pair is not a
  diagnostic; Sounio does not compute lattice QCD; WP25 data-driven
  HVP LO remains absent
Transformation: pin WP25 Eq. (3.22) lattice a_μ^SD and the
  Colangelo 2022d R-ratio a_μ^SD as Epistemic; require |pull| ≤ k95
  and |pull_W(ud)| > |pull_SD|
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: Sounio can hold the SD lattice and R-ratio pins
  without collapsing them; their GUM pull is computed in Sounio and
  is consistent with 0 at k95; the W(ud) split is larger
Claims-Forbidden: Sounio computed HVP; new physics; the tension is
  resolved; 691.0 is full HVP LO; SD agreement implies W agreement;
  Madaros is fixed-point-verified
Assumptions: published centrals and quoted σ are citations (Aliberti
  et al., Phys. Rep. 1143 (2025) 1). Lattice: Eq. (3.22)
  69.10(26)×10⁻¹⁰. Data-driven: 68.4(5)×10⁻¹⁰ from Colangelo et al.
  2022d in the pre-CMD-3 scenario, as quoted by WP25; (5) means
  ±0.5 in 10⁻¹⁰. Stored here as 10⁻¹¹ (×10). Independent GUM.
Write-Set: stdlib/particle_physics/ew_precision.sio,
  stdlib/particle_physics/mod.sio,
  examples/particle_physics/a_mu_hvp_sd_control.sio,
  tests/run-pass/a_mu_hvp_sd_control.sio, this file
Read-Set: docs/audit/A_MU_HVP_WINDOW_UD_2026-08-23.md,
  docs/decisions/adr-008-claim-oracle-semantic-clock.md
Positive-Witness: souc run prints |pull_SD| ≤ 1.96 and
  |pull_W(ud)| > |pull_SD|
Negative-Witness: full lattice LO remains 7132; DD LO still Absent
Acceptance-Gate: tests/run-pass/a_mu_hvp_sd_control.sio exit 0 under
  Madaros
Integration-Target: origin/main
Authoritative-Only-If: the pulls are produced by Madaros running Sounio
```

## Why a control

The intermediate-window \(a_\mu^W(ud)\) split (PR #2077, pull 6.79) is
only diagnostic if the same GUM protocol does not fire on a window WP25
says is compatible. Short-distance is that window.

| pin | WP25 (10⁻¹⁰) | stored (10⁻¹¹) |
|---|---:|---:|
| `a_mu_hvp_sd_lattice_wp25` | Eq. (3.22) 69.10(26) | 691.0(2.6) |
| `a_mu_hvp_sd_rratio_pre_cmd3` | 68.4(5) Colangelo 2022d | 684.0(5.0) |

Independent GUM: \(\sigma=\sqrt{2.6^2+5.0^2}=\sqrt{31.76}\),
pull \(=7.0/\sigma\approx 1.24\le 1.96\).

Falsifier: if \(|\mathrm{pull_{SD}}|>1.96\), the “SD agrees” control is
dead. If \(|\mathrm{pull_W}|\le|\mathrm{pull_{SD}}|\), the claim that the
split lives in W is dead.

## Receipts (2026-08-23)

Control ELF `/workspace/sounio/bin/madaros-linux-x86_64` (100902241 B).

Madaros `souc check` verdict=0. Example and test **rc=0**.

```
A_MU_SD_LAT_VAL 691.000000
A_MU_SD_RR_VAL 684.000000
A_MU_SD_PULL_LAT_RR 1.242103
A_MU_W_UD_PULL_LAT_KNT 6.789189
A_MU_K95 1.960000
VERDICT_SD_AGREES 1
VERDICT_W_SPLIT_LARGER_THAN_SD 1
WP25_DD_HVP_LO_PROVIDED 0
```

\(\sqrt{2.6^2+5.0^2}=\sqrt{31.76}\approx 5.635\); \(7.0/5.635=1.242103\le 1.96\).

LLM-offload math-review (`/tmp/llm-offload-ShmQ67`):

- xAI grok-4.3: five `[OK]` (σ, sqrt, pull, k95, W>SD).
- Z.AI: weekly quota exhausted until 2026-08-25.
- Outcome: **PASS_SINGLE_PROVIDER_DEGRADED**.
