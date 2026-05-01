<!-- docs:meta
topic_id: repo.docs.research.knightian-operator-consensus-2026-04-30
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.knightian-operator-consensus-2026-04-30
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Knightian Operator — Consensus Fan-Out Review (2026-04-30)

**Subject under review:** [knightian_operator_choice.md](knightian_operator_choice.md) — author selected Ferson p-box over Walley credal sets and Klibanoff smooth ambiguity for the vancomycin Knightian-uncertainty thrust.

**Method:** Multi-provider consensus fan-out via `bin/llm-offload --raw <prompt> deepseek xai gemini qwen`.
- Responses obtained: **DeepSeek** (DeepSeek-Coder), **Grok 4.1 fast reasoning** (xai)
- Failed (OpenRouter credits exhausted at the time of run): Gemini 2.5 Pro, Qwen 3 235B
- Failed (invalid key): Groq Llama 3.3 70B
- Mistral via OpenRouter: same credits issue

Raw transcripts archived locally under `/tmp/llm-offload-SSJN5i/` (deepseek.md, grok.md). For the audit log, see `.claude/llm_offload_log.md`.

> Two providers is below the four-way bar the author originally set; the result is a **partial consensus**, but the two responses are independent (different orgs, different model families, different training data) and **substantively convergent**. Treat the result as a strong directional signal that warrants a documented response, not as a final ruling.

---

## 1. Where the two responses converge

Both DeepSeek and Grok return **NO** on operator soundness for the *vancomycin PBPK use case specifically*, with the same root cause:

> **Joint (Vc, CL) dependence is the killer omission.** P-boxes are univariate; they bound marginal CDFs only. The vancomycin Cmin steady-state map is a **non-monotone**, non-linear function of (Vc, CL), and the two parameters are empirically **correlated** (population-PK literature reports r ≈ 0.3–0.7, varying with renal function, obesity, critical-illness state). The correlation structure itself is Knightian.

Concrete consequences both providers identify:
- A naive p-box-on-marginals approach **assumes independence (or a fixed copula)**.
- Under positive Vc/CL correlation: predicted Cmin band is **too wide** → over-conservative refusal.
- Under negative correlation: predicted Cmin band is **too narrow** → unsafe acceptance of toxic doses.
- *Either* outcome is unacceptable for a clinical-decision-support gate.

Both providers note that the decision document acknowledges "discards information not encoded in marginal CDF" but treats it as a "con" rather than a **functional-correctness blocker**.

## 2. Where the two responses diverge

| Question | DeepSeek | Grok 4.1 |
|---|---|---|
| Right alternative? | Walley credal sets restricted to **neighborhood / contamination models** (finite-parametric envelope around a nominal joint distribution from popPK) | Walley credal sets directly via lower/upper expectations + moment bounds |
| Lean line-count for minimal sound formalization | 800–1200 lines for p-box arithmetic; ~3000–4200 lines total with the safety theorem (measure theory required for convolution) | 250 lines for univariate p-box arithmetic; 80 lines for univariate safety theorem |
| Clinical defensibility | NO — regulators recognize p-boxes only for univariate UQ, not multivariate clinical PK | NO — pharma regulators expect popPK / NLME / Bayesian Monte Carlo, not interval UQ |
| GUM compatibility | Identifies it as a *weakness*: GUM is itself criticized for ignoring correlation (JCGM 100:2008 §F.1.2.3 note); inheriting that limitation is not a feature | Treats as a non-rebuttal in the clinical context |

The Lean estimate divergence is notable: Grok's estimate covers only the univariate, GUM-style p-box; DeepSeek's includes the convolution-and-safety-theorem cost over a 2-compartment PBPK function and is closer to a full sound mechanization. **Both estimates exceed the original decision document's implicit "Lean-tractable in 4 weeks" framing.**

## 3. Where the consensus is *not* a refutation

Neither response argues that p-box is *unsound everywhere* — both agree it is the right call for **univariate** Knightian UQ (assay bias, single-parameter sensitivity). The pushback is specific to the **multivariate, non-monotone, correlation-uncertain** clinical-PBPK setting.

Walley with neighborhood models is **not strictly easier** than p-box; it just *forces* the joint-distribution question into the type. Klibanoff stays out of the running for both providers.

## 4. Author response (2026-04-30)

**Acknowledged.** The joint-dependence omission is real and the consensus is correct on this point. The current M2/M3 implementation (`stdlib/epistemic/knightian.sio`, `stdlib/clinical/vancomycin_pbpk.sio`) treats Vc and CL as independent univariate p-boxes; this is unsound for the clinical use case as the consensus describes it.

**Decision (committed):**

1. **Do not roll back M2/M3.** The univariate p-box is still useful as a *building block* and as a regression target; the work is shipped under `claude/approx-effect`.
2. **Open M2.5 — Fréchet-bound enclosure.** Add a soundness wrapper that, given univariate p-boxes on Vc and CL, produces a **copula-free upper enclosure** of the joint Cmin band by taking the worst-case copula (Fréchet–Hoeffding bounds on the joint CDF). This makes the existing p-box infrastructure sound at the cost of additional conservatism — provably an over-approximation under any joint distribution with the given marginals.
3. **Open M3.5 — Walley neighborhood model.** Implement a parametric Walley credal set as `pub struct CredalSet { nominal: Joint(Vc, CL), epsilon: f64, family: ContaminationClass }` and provide a soundness-preserving conversion `credal_to_pbox: CredalSet -> PBox` that goes through Fréchet bounds. Keep the p-box as the operator at the propagation surface; lift to Walley only at the elicitation surface.
4. **Sensitivity test.** Add a `tests/stdlib/clinical/test_vancomycin_correlation_sensitivity.sio` that re-runs the Cmin propagation over a grid of Vc/CL correlations ∈ {-0.7, -0.3, 0.0, 0.3, 0.7} and verifies that the Fréchet-enclosed band contains every correlation-specific point estimate.
5. **Update Lean estimate.** Revise the budget in `knightian_operator_choice.md` to track DeepSeek's 800–1200-line lower bound for the univariate sound case and 3000+ for the safety theorem; document the Fréchet-bound proof obligation as an explicit milestone, not a cleanup task.
6. **Disclose in papers.** The PL paper (`docs/papers/vancomycin_pl_paper_outline.md`) and clinical paper (`docs/papers/vancomycin_clinical_paper_outline.md`) must explicitly state the joint-dependence assumption and the Fréchet-bound enclosure as the soundness mechanism. No silent assumption of independence in any clinical claim.

The consensus did **not** kill the project; it surfaced a real soundness gap that would have killed the *clinical paper* on first review. Better here than there.

## 5. Outstanding follow-ups

- Re-run the consensus once OpenRouter credits are restored, to bring Gemini and Qwen into the loop. Update this document with a v2 if either provider materially shifts the picture.
- Cite Ferson, Joslyn, Helton (2003) and Williamson & Downs (1990) on Fréchet-bound p-box arithmetic as the foundation for M2.5.
- Cite Walley (1991) §2 and Augustin et al. *Introduction to Imprecise Probabilities* (Wiley 2014) §10 for the neighborhood-model construction.
- File issues for M2.5 and M3.5 in the milestone tracker; tie both to the audit-log row that records this consensus review.
