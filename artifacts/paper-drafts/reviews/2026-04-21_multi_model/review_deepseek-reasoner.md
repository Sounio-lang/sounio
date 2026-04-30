## 1. Top 5 concerns, ranked by threat to the main claim.

**1. Minimal effect size, inflated by logarithmic presentation.** The central claim of a "spike" is supported by a cohort median increase of 22.7% from PRE5 to IC. This is a small effect, visually exaggerated in Figure 1 by a logarithmic vertical axis. The paper's own LOO classifier achieves an AUC of 0.642, which is considered a "small" discriminative effect in clinical machine learning. The claim of a "spike" is therefore a substantial overstatement of what is, at best, a modest, cohort-wide statistical deviation.

**2. N=24 is underpowered for a novel, high-dimensional biomarker.** The entire inference rests on 24 patients from a single, heavily pre-processed public dataset. The LOO AUC 95% CI of [0.513, 0.764] has a width of 0.251, which is enormous for a diagnostic claim. The lower bound barely clears chance. With this sample size, the study is only capable of detecting a very large effect, which it did not find. The "pre-registered" label does not compensate for a fundamental lack of power.

**3. The primary p-value (T2, p=0.0123) is weak and potentially cherry-picked from three null models.** The protocol specifies three null models but designates iid as primary. The iid null yields the least significant p-value (1.23e-2), while the circular shift null yields a far more significant one (8.99e-5). The authors then present all three as convergent "robustness" evidence. This is a classic example of within-family selective reporting: if the iid null had been non-significant, the circular shift result would likely have been promoted to primary evidence. The pre-registration does not guard against this form of selection.

**4. The biomarker is an uninterpretable black box with no link to physiology.** The "sedenion associator norm" is a scalar derived from a highly non-linear, arbitrary algebraic mapping of 16 EEG channels into a 16-dimensional non-associative algebra. The paper provides no mechanistic or empirical justification for why this particular construct should be sensitive to ictal onset beyond vague hand-waving about "order-sensitivity" and "geometric events." Any complex, non-linear function of 16 channels might produce a statistically significant deviation in a small cohort by chance. The paper fails to rule this out.

**5. The refutation of the pre-ictal dip (T1) undermines the proposed narrative.** The original hypothesis predicted a two-phase signature (dip then spike). The dip is conclusively refuted (p=0.599). The authors then retreat to claiming only the "ictal-onset spike." This is a major post-hoc narrowing of the claim, indicating the initial theoretical framework was flawed. The surviving "spike" is now an isolated, post-hoc observation, severely weakening its pre-registered status.

## 2. Statistical / pre-registration audit.

**BH-FDR family definition:** The family {T1, T2, T4} is correctly defined. However, T4 (LOO AUC) is not a direct test of the primary claim ("spikes at ictal onset"); it is a separate classifier-based assessment. Including it in the primary family is defensible but dilutes the focus.

**Treatment of multiple nulls:** This is the critical flaw. The protocol states "the iid null is the primary per-pre-registration; the other two are reported as robustness checks." However, in the Results and Abstract, the p-values from all three nulls are presented as parallel, supporting evidence for the same hypothesis (T2). This is statistically illegitimate. If three tests of the same hypothesis are performed, the Type I error rate is inflated unless they are formally combined or the most conservative is chosen. Presenting the most significant from a basket of nulls (8.99e-5) in the abstract alongside the primary (1.23e-2) is misleading. The pre-registration does not specify a plan for combining or reporting these, allowing for selective emphasis.

**LOO AUC 95% CI honesty:** The CI is reported as [0.513, 0.764]. Given N=24 and a LOO procedure, this interval is implausibly narrow. Standard DeLong or bootstrap CIs for LOO AUC with N=24 typically have widths >0.35. The reported width of 0.251 suggests the bootstrap procedure may not have fully accounted for the dependence induced by leave-*patient*-out resampling or used an inappropriate method. This merits immediate verification.

**Pilot Hessian study leakage:** The preprint states the Hessian pilot was "unstable" and "abandoned." However, its existence reveals a prior exploration of model-based, non-linear biomarkers. The decision to switch to the "data-level" associator norm was almost certainly informed by the pilot's failure. This constitutes a form of indirect data leakage: the cohort's (or a subset's) inability to support a Hessian-based biomarker influenced the choice of a different biomarker. The pre-registration of the final protocol does not erase this iterative, data-influenced development history, which biases the entire study.

## 3. Biomarker construction and algebraic choices.

**Motivation is arbitrary and post-hoc rationalized.** The justification is purely algebraic: sedenions are the "minimal Cayley–Dickson algebra whose associator has no alternative-law cancellation." There is zero biological or clinical motivation for modeling 16 EEG channels as basis elements of a sedenion algebra. The mapping of channel *i* to basis element *e_{i-1}* is completely arbitrary; any permutation would yield a different associator norm. The claim that this measures "order-sensitivity" of recurrence is a narrative draped over an arbitrary computation.

**Simpler alternatives would almost certainly capture the same signal.** The associator norm, for the chosen embedding, is a specific cubic polynomial in the 16 input channels. A systematic exploration of all cubic interactions (or even a simpler measure like the norm of the bispectrum) would likely reveal similar effects. The sedenion algebra is **ornamental**. It provides a complex mathematical backstory but no unique analytical power. The paper provides no control analyses showing that the sedenion construction outperforms a simple, interpretable nonlinear measure of multi-channel interaction.

**The algebra is not load-bearing.** The entire computation could be described as: 1) Take three consecutive time-lagged vectors of 16 pre-processed EEG samples. 2) Compute a specific trilinear form defined by the Cayley-Dickson multiplication table. 3) Take its Euclidean norm. The connection to non-associative algebra is a matter of notation, not necessity. The "sedenion" framing is a sophistication bias that obscures the simplicity (and arbitrariness) of the actual computed statistic.

## 4. Clinical validity and scope.

The paper cannot honestly call this "evidence" of a generic mechanism. The correct caveat is: **This is an underpowered, exploratory analysis of a mathematically arbitrary feature in a small, homogeneous, pediatric cohort with intractable epilepsy, showing a marginal statistical effect that may be a false positive or a dataset-specific artifact.**

CHB-MIT is a single-center, retrospective dataset enriched with severe cases. Findings do not generalize to adult epilepsy, other etiologies, or non-intractable cases. The use of scalp EEG further limits mechanistic interpretation, as the signal is far removed from underlying cortical dynamics. The paper's own Limitations section (ii) acknowledges the modest p-value and wide CI but does not go far enough in dismissing the clinical relevance of the finding. The title and abstract actively mislead by implying a generalizable "evidence" for a "spike at ictal onset in scalp EEG."

## 5. Recommendation: reject.

The single most important change required—a complete re-evaluation of the biomarker's motivation and validation in a much larger, independent cohort—is beyond the scope of a revision. The study is fundamentally flawed by its arbitrary biomarker, small sample size, and questionable statistical practices regarding multiple nulls.

**Recommendation: Reject.**

The core claim is not supported by the evidence presented. The methodological concerns (especially the selective use of multiple nulls and the uninterpretable biomarker) are fatal. The work is better suited as a technical report or a conference proceeding on mathematical signal processing, not a clinical biomarker paper in a Q1 journal.

---

### Passages Triggering Methodological Desk Rejection:

1.  **Abstract:** "The ictal-onset spike in the associator norm is therefore a pre-registered, sign-consistent, and spatially robust finding; it is not an artefact..." This is a definitive conclusion unsupported by the weak evidence and ignores the threat of arbitrary biomarker construction.
2.  **Results, Section 3.2:** "T2 ictal spike is robust across three null models" and the accompanying figure. This presentation treats the three p-values as independent, convergent evidence without statistical correction, a serious misuse of sensitivity analyses.
3.  **Discussion, Opening:** "The sedenion associator norm spikes at ictal onset in the CHB-MIT cohort, pre-registered, 100k-permutation, BH-FDR-controlled." This conflates procedural rigor (pre-registration) with clinical significance and overstates the strength of the finding.
4.  **Background, Sedenion Lifting:** The entire description of mapping channels to sedenion basis elements presents an arbitrary choice as a principled methodological step, with no justification for why this mapping is physiologically or mathematically privileged.
5.  **Methods, Null Models:** The statement that "the iid null is the primary per-pre-registration; the other two are reported as robustness checks" is a red flag. It sets up a scenario where the least favorable result is officially "primary," but the more favorable results from other nulls are still highlighted to strengthen the narrative, a form of *p*-hacking by proxy.
