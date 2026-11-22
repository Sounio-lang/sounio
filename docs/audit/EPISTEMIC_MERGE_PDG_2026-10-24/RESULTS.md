<!-- docs:meta
topic_id: repo.docs.audit.epistemic-merge-pdg-2026-10-24.results
authority: repo_only
audience: users
last_validated: 2026-10-24
validated_by: audit.cpp; output.txt
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.epistemic-merge-pdg-2026-10-24.results
-->

# Results: ADR-016 `meet` on the PDG 2025 averages

Protocol: `PROTOCOL.md`, registered in commit `34a73f9e96` (2026-10-24 08:23:59 −03:00) before any result was computed, and unchanged since. Reproduce with `./run.sh`; the raw output is `output.txt`, per-average results are in `results.csv`.

## Results

**Corpus.** The 2025 edition holds 7068 computed averages. 6764 have at least one eligible input; 4315 of these have exactly one and are excluded. **2449 averages** with *N* ≥ 2 inputs enter the analysis.

**Checks.** The χ² tail used by the rule gives `p₂ = P(χ²_1 ≥ 4) = 0.045500`, equal to `erfc(√2)`, and `P(χ²_10 ≥ 18.307) = 0.05000`.

**Primary outcome.** The default `meet` refuses **248 of 2449 averages (10.13 %, 95 % CI 8.99–11.38 %)**.

**Secondary outcomes.**

1. At least one input pair has `E_n > 1`: 417 / 2449 (17.03 %, 95 % CI 15.59–18.57 %).
2. Rule against PDG inflation:

   | | PDG `S > 1` | PDG `S = 1` |
   |---|---|---|
   | refused | 228 | 20 |
   | accepted | 580 | 1621 |

3. Among the refused averages, the PDG scale factor has median 2.12 (IQR 1.78–2.54), maximum 8.80; 152 of the 248 have `S ≥ 2`.
4. Extraction validation: among the 808 averages with PDG `S > 1`, our Birge ratio differs from the PDG `S` by a median of 0.61 % (IQR 0.00–9.75 %).

**Sensitivity.** (a) Excluding averages with an input asymmetry above 0.2: 209 / 2267 (9.22 %, 95 % CI 8.10–10.48 %). (b) *N* ≥ 3 only: 176 / 1259 (13.98 %, 95 % CI 12.17–16.00 %).

**Post-hoc analyses** (not in the protocol; added after the corresponding observation, and labelled in `audit.cpp`).

- (c) Some inputs cite a PDG edition as their reference (derived values). Excluding averages with such an input: 248 / 2448 (10.13 %). The one excluded average was not refused.
- (d) All **20 / 20** averages that the rule refuses while the PDG reports `S = 1` are explained by the PDG rule for `S`, which excludes inputs with `u_i > 3√N · u_x̄` before computing `S`: with that exclusion applied, `S = 1` in every one of them.
- (e) Re-computing `S` with the PDG exclusion rule reproduces the published `S` with a median relative difference of 0.01 % (IQR 0.00–1.42 %) over the 808 inflated averages.

Example for (d): the muon lifetime (`S004T`). The PDG reports `S = 1.0`. Its eight inputs give `χ² = 16.09` on 7 degrees of freedom (`p = 0.024`, `R_B = 1.52`); BARCZYK 2008 is 2.9 σ from TISHCHENKO 2013. TISHCHENKO 2013 is about ten times more precise than every other input, so the exclusion rule removes all of them, and `S` is computed from a single input.

## Discussion

**What the primary result says.** One in ten multi-input averages in the most carefully curated compilation of particle-physics measurements combines inputs that are mutually inconsistent at the ISO 13528 default. Under ADR-016 none of these would compile as a plain `meet`. Each would need an explicit choice, recorded in provenance: carry the conflict as `Contest<T>`, or declare a Birge-type policy.

**Agreement with the PDG.** For 228 of the 248 the PDG already acted, by inflating the uncertainty (median `S` = 2.12). This corresponds to the `consistency: Birge` policy of ADR-016, applied by default rather than on request. The PDG is explicit about it, so these combinations are not silent at the source. What a downstream user receives, when only `value ± error` is quoted, is a number whose error has been more than doubled to absorb a disagreement, with the disagreement itself no longer visible. Under ADR-016 that information travels with the value.

**The PDG inflates beyond detection.** In 580 averages the PDG inflates (`S > 1`) where the χ² test does not reject. The PDG convention inflates whenever `R_B > 1`, which includes ordinary statistical fluctuation. This is conservative, and ADR-016 does not forbid it as a declared policy.

**A blind spot in the convention.** The 20 discordant cases share one mechanism. When one input dominates, the PDG exclusion rule discards the others before `S` is computed, so `S = 1` however strongly they disagree with the dominant input. The muon lifetime, which fixes the Fermi constant, is one of them. A consistency test applied to all inputs, as `meet` does, has no such blind spot. The exclusion rule exists for a good reason (low-weight inputs should not drive `S`), and these cases show what it costs: the average is dominated by one measurement whose agreement with the rest is never checked.

**Limitations.**
- Independence of inputs is assumed; the database carries no covariances, so correlated inputs (shared systematics, the same collaboration) can make the test either too strict or too lenient.
- Asymmetric uncertainties are symmetrised; sensitivity (a) shows this moves the primary rate by less than one percentage point.
- The rule is the χ² test at the tail probability of the two-source default; ADR-016 leaves the *N*-source policy to be declared, and other declared policies give other rates (secondary outcome 1 gives the pairwise one).
- The PDG is a best-practice corpus. The audit says nothing directly about less curated pipelines, and nothing here concerns the below-quantification-limit (censored) data of decision 10, for which no public corpus was used.

## Claims forbidden

- That the PDG combines inconsistent data silently: it publishes `S`.
- That 10 % of PDG averages are wrong: the rule detects inconsistency between inputs, not an error in the average.
- That the result generalises to other fields without an audit of their data.
