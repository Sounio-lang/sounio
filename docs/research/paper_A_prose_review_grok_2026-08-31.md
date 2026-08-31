<!-- docs:meta
topic_id: repo.docs.research.paper-a-prose-review-grok-2026-08-31
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.paper-a-prose-review-grok-2026-08-31
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Paper A — programme-committee-style review of the PROSE (Grok, 2026-08-31)

**Dispatch:** `bin/llm-offload -t review -p xai` (the `review` task routes to Grok 4.3), packet =
the full closed draft `paper_A_MERGED_2026-08-25.md` (as of `7d007e374b`) + PLDI/OOPSLA reviewer
instructions (summary, strengths, weaknesses, per-section comments, questions, merit, three
changes). This is the first review of the *paper text*; the four earlier reviews were of the Lean
file. **Merit 2 / expertise 3.** Every substantive point is acted on or answered below.

---

## Review (verbatim)

**Summary**

The paper argues that uncertainty-propagation libraries silently understate variance when operands are correlated because they implement the independence special case of the GUM delta-method rules without checking the assumption. It recasts the problem as a Blackwell anti-garbling, proposes a type system that tracks noise-symbol source sets in the type `Knowledge⟨T,N⟩`, rejects independence-assuming operators on non-disjoint sets (E230), and claims a kernel-checked Lean mechanization of the resulting first-order soundness theorem together with a production implementation in the Sounio compiler evaluated on a vancomycin PBPK model.

**Strengths**

- The core observation (mul vs. square, add vs. sub asymmetry) is concrete and reproducible from the cited library fragment.
- The framing as an anti-garbling is a coherent, if narrow, application of an existing information-flow order.

**Weaknesses**

The central claims rest on a gap between what is mechanized and what is asserted to be deployed, on an extremely narrow soundness statement that excludes the nonlinear operators actually used in the clinical example, and on an evaluation whose "wired compiler" results are presented without reproducible artifacts or comparison to existing static analyses that already track affine correlations. The paper repeatedly asserts that the E230 rule is "now wired into the production checker and verified from source" while simultaneously marking interprocedural summaries—the component required for the rule to be usable—as future work. The clinical "34.2 % of true WARNs silenced" figure is obtained on a synthetic cohort whose generation is not released and whose Monte-Carlo validation is summarized rather than supplied. Related-work coverage of Fluctuat/zonotopic domains and of existing correlation-tracking libraries (Measurements.jl's derivative graph, Uncertain⟨T⟩) is cursory and does not engage with why those systems' design choices were rejected. The type system is presented as novel for lifting noise symbols into types, yet the paper never shows a non-trivial program that type-checks under the new discipline but would have been rejected by a simple intraprocedural escape analysis augmented with the same source-set lattice.

**Detailed comments by section**

- §2.1–2.2: The library fragment is useful, but the claim that "nothing routes x·x to the sound one" is an engineering observation, not a type-system result; the paper never demonstrates that the same defect would survive a modest refactor or a simple purity/alias analysis.
- §4.4 and §6.5: The honesty about "disjoint support ⇒ zero covariance but not conversely" is welcome, yet the soundness theorem is stated only for admitted operators. The completeness gap is therefore not a side remark but the central limitation of the approach; it should be quantified on a realistic corpus rather than dismissed to an escape valve whose implementation cost is left to §10.
- §6.2: The mechanized calculus reduces through the defective `gAddMeta`/`gMulMeta`. This is presented as a virtue ("we prove type safety of the unsound system"), but it means the theorem only shows that the source-set discipline prevents certain reductions from being reached; it does not show that the scalar operators themselves become sound.
- §6.4: Theorem 6.4 is claimed to be fully mechanized, yet the text lists six explicit "not mechanized" items, including the correspondence between the core calculus and the production E230 rule. The claim "kernel-checked, Mathlib-free" is therefore accurate only for a fragment whose relation to the deployed checker is itself unproven.
- §7.3: Phases N1–N4 are described as "landed," but the only pending item listed is precisely the interprocedural summaries required for any realistic program. The evaluation in §8 therefore measures a checker that is still intraprocedural; the reported false-positive counts are lower bounds.
- §8.4: The two-compartment flip-rate experiment is the only quantitative clinical claim. The 34.2 % figure is given for one specific sum under ρ=1; the phase-decomposition case produces the opposite error. The paper does not report how often real PBPK models perform the former versus the latter, nor whether the measured rate survives the correlation-aware operators that the type system itself encourages.

**Questions for the authors**

1. Which of the six "not mechanized" items in §6.4 would have to be discharged before the mechanization claim could be considered to cover the production checker?
2. The escape valve for overlapping-but-orthogonal operands requires either a proved-disjoint certificate or an explicit `add_correlated` call. Has either facility been implemented and exercised on a program larger than the micro-benchmarks in §8.3?
3. The Monte-Carlo adequacy check reports Var_MC/Var_T = 0.999 on the two-compartment model. Was this performed with the same source-set labels that the type system sees, or with an oracle that knows the true generating sources?

**Overall merit:** 2
**Reviewer expertise:** 3 (type systems and static analysis; limited domain knowledge in clinical pharmacokinetics)

**Three changes that would most raise the score**

1. Replace the claim of a "fully mechanized" soundness theorem with a precise statement of what has been kernel-checked versus what remains a manual correspondence argument between the Lean model and the production E230 rule.
2. Provide a reproducible artifact containing the wired Madaros binary (or build instructions), the exact commit of the two-compartment model, and the LCG seed used for the 5 000-patient cohort so that the 34.2 % figure can be independently regenerated.
3. Either implement and evaluate the interprocedural summaries or restrict all claims and measurements to the intraprocedural fragment that is actually deployed, making the limitations section match the evaluation scope.

---

## Disposition (same day)

| point | action |
|---|---|
| Change 1 — "fully mechanized" vs six non-mechanized items | **Done.** Contribution 3 and the opening of §6.4's status now say precisely: the *core calculus* (type safety, `Exact`, Theorem 6.4) is kernel-checked; the correspondence to the production E230 rule is a manual argument backed by the four controls and the sabotage gate; interprocedural, second order and the modelling axioms are not mechanized. "We do not claim the deployed checker is verified; we claim the rule it implements is." |
| Change 2 — reproducible artifact (binary/build, commit, seed) | **Done.** `docs/research/paper_A_artifact/` — `README.md` + `reproduce.sh`: Lean gate, the three Sounio prototypes, RQ4 cohort (LCG seed 20260831, both scenarios), Monte Carlo adequacy, affine tests, the engine-portable TDM receipt; every program prints a verdict line the script compares against the recorded values; compiler identity checked with `SOUNIO_REQUIRE_COMMITTED_MADAROS=1` (#2333). The wired E230 checker itself is on the rebased NS-wire branch (PR pending codex-2 review) and the artifact says so rather than pretending it is on `main`. |
| Change 3 — restrict claims to the intraprocedural fragment | **Done.** §8 opens with the scope statement (all measurements are of the intraprocedural checker; §8.3 counts are lower bounds; call-crossing programs are out of scope); the §7.3 "landed" language is unchanged because N1–N4 *are* landed on the integration branch — the scope sentence makes the limitation explicit where the numbers are. |
| §8.4 — prevalence of the two sums in real PK code; effect of correlation-aware operators | **Stated as a limitation** in §10: the claim is the mechanism and its two signs on one synthetic cohort, not a population prevalence. |
| Q1 | The correspondence calculus↔checker (a manual argument) and the interprocedural extension; the modelling axioms are boundaries, not gaps. Answered in §6.4's scope paragraph. |
| Q2 | `add_correlated` exists in-tree (`gum_supplement1.sio`) and is exercised only by unit tests; the certificate path is not implemented. The paper already says "orphaned"; the artifact README repeats it. The *affine* type (`stdlib/epistemic/affine`, #2322) is the implemented escape valve — it computes the exact sum instead of asking for ρ — and the artifact runs it on the clinical chain. |
| Q3 | The Monte Carlo draws from exactly the four sources the type system sees (weight, SCr, Q, Vp) with the same uncertainties; there is no oracle beyond the model's own labels. Stated in `paper_A_rq4_mc_adequacy_2026-08-31.md`. |
| §6.2 — "the scalar operators do not become sound" | Correct reading, and it is the paper's design: soundness is `Exact`, an invariant carried by typing over the defective operators; §6.1 says so. Left as is. |
| §2 — would the defect survive a refactor / alias analysis? | Not addressed in this round; noted for the related-work pass (an alias analysis sees variables, not measurement sources — `x` and `ident(x)` bound to different names share a source; that is the §8.2 `ident(x)+x` control). |
| Related work: Fluctuat/zonotopes, Measurements.jl's derivative graph | Cited in §9; the "why rejected" argument (external analyser vs type; graph vs set) needs a paragraph — **open**, next pass. |
