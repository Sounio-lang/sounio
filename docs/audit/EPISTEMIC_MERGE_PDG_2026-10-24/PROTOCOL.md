<!-- docs:meta
topic_id: repo.docs.audit.epistemic-merge-pdg-2026-10-24.protocol
authority: repo_only
audience: users
last_validated: 2026-10-24
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.epistemic-merge-pdg-2026-10-24.protocol
-->

# Protocol: retrospective audit of ADR-016 `meet` on the PDG 2025 averages

**Status**: registered before any result was computed. This file is committed alone; the results are committed afterwards and must not change it.

## Question

How often does a best-practice compilation of physical measurements combine inputs that the default `meet` of ADR-016 would refuse, so that an explicit consistency policy would have been required?

The Particle Data Group documents its combinations: when inputs disagree it inflates the uncertainty by a scale factor `S` and publishes `S`. The audit does not claim the PDG combines silently. It measures the prevalence of inconsistent fusion in the most careful corpus available, which bounds from below what less careful pipelines do, and it records what a downstream user inherits when only the average is quoted.

## Corpus

- File: `pdgall-2025-v0.2.3.sqlite`, downloaded on 2026-10-24 from `https://pdg.lbl.gov/2025/api/`.
- SHA-256: `1a042ee01d3aad5d92a31a5209f5ff6720f7a60ee27fd9dbfd0d743127189a1d`.
- Units of analysis: every row of `pdgdata` with `edition = '2025'` and `value_type = 'AC'` (average computed by the PDG). Global fits (`FC`) are excluded because their inputs are correlated and the covariances are not in the database.

## Inputs of each average

The rows of `pdgmeasurement_values` joined to `pdgmeasurement` on the same `pdgid` with `column_name = 'VALUE'`, `used_in_average = 1`, no `limit_type`, and non-null, strictly positive `error_positive` and `error_negative`.

- Standard uncertainty of input *i*: `u_i = (e⁺_i + e⁻_i) / 2`.
- Inclusion: at least 2 inputs.
- Independence between inputs is assumed. The database carries no covariances; this is a stated limitation, and the PDG makes the same assumption for most averages.

## Statistics

- Inverse-variance weighted mean `x̄`, `χ² = Σ (x_i − x̄)² / u_i²`, Birge ratio `R_B = √(χ² / (N − 1))`.
- Pairwise normalised error with coverage factor *k* = 2: `E_n = |x_i − x_j| / (2 √(u_i² + u_j²))`.

## Decision rule (primary)

The default policy of ADR-016 is `E_n ≤ 1` at *k* = 2 for two sources. Its exact *N*-source generalisation used here is the χ² test at the same tail probability: refuse iff

`P(χ²_{N−1} ≥ χ²) < p₂`, with `p₂ = P(χ²_1 ≥ 4) = 0.04550`.

For *N* = 2 this is identical to `E_n > 1`.

## Outcomes

- **Primary**: the proportion of included averages that the rule refuses, with a Wilson 95 % interval.
- **Secondary**:
  1. the proportion with at least one input pair at `E_n > 1`;
  2. a 2 × 2 table of the rule against the PDG's own inflation (`scale_factor > 1`);
  3. the distribution of the PDG `S` among refused averages;
  4. extraction validation: among averages with PDG `S > 1`, the median relative difference between our `R_B` and the PDG `S`. The two are not expected to coincide exactly, because the PDG excludes inputs with δ_i > 3√N δ_x̄ from the computation of `S`.
- **Sensitivity**: (a) excluding averages with any input asymmetry `|e⁺ − e⁻| / (e⁺ + e⁻) > 0.2`; (b) restricting to *N* ≥ 3.

No quantitative hypothesis is stated. The outcomes are reported whatever they are.

## Computation

Extraction by SQL to CSV (`extract.sql`); every computation in C++ (`audit.cpp`), with the χ² tail from the regularised upper incomplete gamma function, implemented in the file and checked against `p₂`.
