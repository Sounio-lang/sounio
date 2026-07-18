<!-- docs:meta
topic_id: repo.docs.papers.techrxiv-submission
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.techrxiv-submission
-->

# TechRxiv Preprint Submission — Checklist

Upload at: https://www.techrxiv.org/submit

---

## Paper A

> ⚠️ **CORRECTION NOTICE — DO NOT SUBMIT THIS ABSTRACT AS-IS (2026-07-16).**
> The quantitative O-SSM results in the abstract below were computed on an octonion
> multiplication table that carried a sign error in the `e2·e5` product
> (`-a2*b5+a5*b2` instead of `+a2*b5-a5*b2`). That table **fails alternativity and
> composition** — it is *not* the octonion algebra (nor any composition algebra), so the
> non-associative claims it produced are table artifacts, not octonion results.
> Specifically affected: "O-SSM wins 12" (of 15), "sorting 69.5% → 72.5% / diagonal at
> chance 32.5%", "ListOps 26% vs 15%", "Morse 44.5% vs 14%". On the corrected algebra the
> two paper-cited benchmarks that actually use an octonion table do **not** show these
> wins (`multihead_unit_oct`: sorting-like octonion ≈32.8% vs diagonal ≈54.0% — reversed;
> `listops`: octonion 13.0% = H-SSM 13.0% vs diagonal 20.5%). This matches the repo's own
> corrected record (NeuroDyn A/B re-audit NEGATIVE, ABIDE associator null, `zd_bptt`
> ZD-advantage → +0.00pp).
> **Corpus fix: PR #1024. Corrected framing to write up: representational-capacity claim
> (cf. PR #907), not an ML-benchmark-win claim.** The author must revise the abstract
> before submission — this notice deliberately does not rewrite it.

**PDF**: `docs/papers/paper_a_ossm.pdf` (11 pages, 239KB)

**Title**: Non-Associative State Space Models: Octonion Dynamics for Path-Dependent Sequence Modeling

**Authors**:
1. Demetrios Chiuratto Agourakis (ORCID: 0009-0001-8671-8878)
2. Dionisio Chiuratto Agourakis

**Abstract** (copy-paste):
Structured state space models (SSMs) such as S4 and Mamba rely on associative matrix operations to enable efficient parallel scans over sequences. We propose O-SSM, a state space model whose hidden state evolves via octonion multiplication in R^8, deliberately exploiting the non-associativity of the octonion algebra. Among the 7^3 = 343 basis triples of the imaginary octonions, exactly 168 = |PSL(2,7)| produce nonzero associators, creating 168 directions in which sequential state products depend on parenthesization order. Across 15 benchmarks spanning order-dependent, hierarchical, symmetry, and temporal tasks, O-SSM wins 12, including sorting (69.5% vs 35%), LRA-style ListOps (26% vs 15%), and Morse decoding (44.5% vs 14%). Multi-head scaling (4 heads x 8-dim = 32-dim hidden, 640 parameters) further improves sorting accuracy to 72.5% while diagonal SSMs remain at random chance (32.5%). O-SSM also outperforms S4D-Inv initialized diagonal SSMs by 11% on next-token prediction. The composition algebra property |xy| = |x|.|y| (Hurwitz's theorem) guarantees norm preservation through time. O-SSM is uniquely positioned at the Cayley-Dickson boundary: the maximal algebra combining non-commutativity, non-associativity, and norm preservation.

**Keywords**: state space models, octonions, non-associative algebra, sequence modeling, Fano plane, PSL(2,7), Hurwitz theorem, long-range dependencies

**Category**: Computer Science > Machine Learning

**License**: CC BY 4.0

---

## Paper B

**PDF**: `docs/papers/paper_b_ekan.pdf` (10 pages, 229KB)

**Title**: E-KAN: Analytical Uncertainty Propagation in Kolmogorov-Arnold Networks via the Guide to the Expression of Uncertainty in Measurement

**Authors**:
1. Demetrios Chiuratto Agourakis (ORCID: 0009-0001-8671-8878)
2. Dionisio Chiuratto Agourakis

**Abstract** (copy-paste):
Uncertainty quantification in neural networks typically requires expensive ensemble methods or approximate Bayesian inference. We show that Kolmogorov-Arnold Networks (KANs) with piecewise-linear hat-basis edge activations admit exact first-order uncertainty propagation under the Guide to the Expression of Uncertainty in Measurement (GUM, JCGM 100:2008), the international metrological standard. Our method, E-KAN (Epistemic KAN), propagates coefficient standard uncertainties analytically through each layer using the law of propagation of uncertainty (LPU), producing calibrated confidence intervals in a single forward pass -- with no sampling, no ensembles, and no posterior approximation. On three UCI regression benchmarks, E-KAN GUM achieves 90-100% coverage at the 95% confidence level where 5-model deep ensembles achieve 0-76%. Validated against N=2,000 Monte Carlo trials on both a pharmacokinetic ODE system (sigma-ratio 0.986, coverage 94.85%) and the E-KAN network itself (coverage 99.8%), GUM propagation is 20x faster than ensembles and provides ISO-traceable uncertainty budgets. We characterize failure modes: GUM breaks on feature interactions (Friedman-1: 10%), out-of-distribution inputs, and heteroscedastic noise.

**Keywords**: uncertainty quantification, Kolmogorov-Arnold Networks, GUM, JCGM 100:2008, measurement uncertainty, piecewise-linear, pharmacokinetics, epistemic uncertainty

**Category**: Computer Science > Machine Learning

**License**: CC BY 4.0

---

## Upload Steps

1. Go to https://www.techrxiv.org/submit
2. Sign in with IEEE account (create one if needed)
3. Click "Submit a Preprint"
4. Upload PDF
5. Fill in metadata (title, authors, abstract, keywords) from above
6. Select category and license
7. Submit
8. Repeat for second paper
9. Save the DOI links for NeurIPS submission
