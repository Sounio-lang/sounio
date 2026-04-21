### 1. Top 5 concerns, ranked by threat to the main claim.

The central claim is that "the sedenion associator norm spikes at ictal onset", as stated in the title and abstract ("the ictal-onset spike in the associator norm is therefore a pre-registered, sign-consistent, and spatially robust finding"). But the evidence is a weak statistical signal in a tiny cohort (N=24), not a robust spike that supports a novel biomarker. Ranked by severity:

1. The effect size is trivial and inconsistent: cohort median spike is only +22.7% from PRE5 to IC (per "Results" section), but per-patient values in Appendix Table show negative spikes in 9/24 patients (e.g., patient chb01: spike=-0.12), contradicting "sign-consistent". This undermines the claim of a reliable "spike" versus random fluctuation in a small sample.

2. Overstated robustness: T3 claims "100/100 sign preservation" but that's on cohort medians, not individual patients; the figure shows medians barely positive (e.g., spike ~0.227), and with N=24, this is just noise amplification through resampling, not evidence of a true onset mechanism.

3. The claim ignores the refuted dip (T1 p=0.599), yet the abstract frames the spike as standalone success; the introduction's question ("Does the sedenion associator norm change at seizure onset...") is cherry-picked to highlight T2 while downplaying that half the hypothesis (pre-ictal change) failed.

4. Small N=24 from a single enriched pediatric dataset (CHB-MIT) can't support "spikes at ictal onset" as a general phenomenon; the discussion admits "we cannot claim the associator spike as a general onset marker outside this population", but the title and conclusion assert it without caveat, inflating a cohort-specific correlation to a biomarker.

5. Algebraic novelty is unproven: the "spike" might just be generic nonlinearity in EEG (e.g., any cubic term), not sedenion-specific, weakening the claim that non-associative algebra reveals a "geometric property" of seizures (per abstract).

### 2. Statistical / pre-registration audit.

The BH-FDR family is defined as {T1, T2, T4} at q=0.05, per the abstract and Table 1, but it's mishandled: T2's p=0.0123 and T4's p=0.0163 survive, but with only three tests, BH is overly permissive (cuts at 0.0167, 0.0333, 0.05), and the family excludes T3/T5 despite their use in bolstering claims (e.g., abstract calls T3 "spatially robust"). The three nulls (iid, circular, block) are treated as independent evidence, with separate p-values reported (e.g., T2: iid 1.23e-2, circular 8.99e-5, block 3.07e-2) and discussed as "robust across three null models" in Results, but this multiplies testing without correction, inflating significance; the protocol (per "PROTOCOL.md" mention) specifies iid as primary, but the paper elevates the others post-hoc, violating pre-registration. LOO AUC=0.642 with CI [0.513, 0.764] is reported honestly, but the width (0.251) is absurdly large for N=24—expected for binomial variance in small samples, yet the paper claims "strictly above 0.5" as success without noting it's barely (0.013) over chance, consistent with noise. The pilot Hessian study leaked: the introduction says it "proposed the off-diagonal L1 norm... as a candidate biomarker" but was abandoned for instability, yet it motivated the associator norm ("we therefore abandoned it in favour of"), so the statistic choice wasn't blind—pre-registration happened after pilot insights, per "the Hessian pilot is summarised here only to explain why we did not report its numbers".

### 3. Biomarker construction and algebraic choices.

The sedenion associator norm is arbitrary and undermotivated: the introduction claims it "records the failure of associativity" as a "geometric property" of seizures, but it's just a contrived scalar from embedding 16-channel EEG into R^16 with Cayley-Dickson product, averaged over 80 timesteps—no empirical justification why sedenions capture "order-sensitive recurrence" beyond vague citations (e.g., Baez on octonions, unrelated to EEG). A simpler nonlinearity like cubic coupling (e.g., mean of |a*b*c| for whitened channels) or quaternion tensor associator would likely produce similar "spikes" due to EEG's inherent non-stationarity at onset, not algebraic depth; the paper doesn't test alternatives, so the sedenion choice feels ornamental. The algebra isn't load-bearing: the discussion admits it's "an empirical, data-level index of how far the recent recurrence structure departs from an octonion subalgebra", but with zero divisors and non-alternativity hyped (e.g., "sedenions are the minimal Cayley--Dickson algebra whose associator has no alternative-law cancellation"), yet no ablation shows if octonions or even complexes fail—it's algebraic window dressing on basic signal variance.

### 4. Clinical validity and scope.

With scalp EEG from 24 pediatric patients in the single-center CHB-MIT dataset (enriched for intractable focal epilepsy, per "Data" section), the paper can't honestly call this "evidence" of a generic pre-ictal/ictal mechanism—the cohort is biased toward severe cases, lacks controls for medication or etiology, and ignores inter-individual variability (e.g., 9/24 negative spikes). The abstract's "pre-registered evidence that the sedenion associator norm spikes at ictal onset" overreaches; it's at best a weak correlation in one small, non-representative group. The right caveat is: "This modest statistical association in a pediatric intractable cohort does not imply a universal seizure mechanism and requires validation in diverse adult populations with intracranial EEG to rule out artifacts from scalp filtering or dataset specifics."

### 5. Recommendation: accept / major revisions / reject, with the single most important change required before submission.

Reject. The single most important change is to remove all claims of a "spike" as a biomarker and reframe as exploratory correlation in a small cohort, with mandatory comparisons to simpler baselines (e.g., signal variance or MSE, which the pipeline already computes but ignores).

Passages that would trigger methodological-review desk rejection if left as-is:
- Abstract: "The ictal-onset spike in the associator norm is therefore a pre-registered, sign-consistent, and spatially robust finding" — overstates weak, inconsistent effect.
- Results: "T2 survives all three nulls" — treats multiple nulls as independent without correction, violating multiplicity control.
- Discussion: "The sedenion associator norm spikes at ictal onset in the CHB-MIT cohort, pre-registered, 100k-permutation, BH-FDR-controlled" — ignores small effect size and refuted hypotheses.
- Conclusion: "A pre-registered... analysis... finds that the sedenion associator norm... spikes at seizure onset" — unsubstantiated generalization from N=24.
