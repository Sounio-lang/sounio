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

## Paper C

**PDF**: `docs/papers/paper_c_san_net_fpga.pdf` (7 pages, 67KB), compiled
from `docs/papers/paper_c_san_net_fpga.tex` with `tectonic` (no LaTeX
distribution installed on this host — `tectonic` is a self-contained,
no-sudo static binary that fetches TeX packages on demand). Recompile with
`tectonic docs/papers/paper_c_san_net_fpga.tex`. Prose source of record
remains `docs/papers/paper_c_san_net_fpga.md`.

**Title**: Network-Attached FPGA Inference Scanning: A Bit-Exact SAN
Catastrophe Scan over 100G Ethernet, and a Cautionary Tale About ARP

**Authors**:
1. Demetrios Chiuratto Agourakis (ORCID: 0009-0001-8671-8878)

**Abstract** (copy-paste):
We port a previously-accepted, production FPGA kernel -- a catastrophe-exit scan for early-exit inference architectures (SAN-ResNet-50, SAN-ViT-large), validated bit-exact over PCIe DMA at 511 Msamples/s on an AMD Alveo U250 -- to receive its input directly from a 100G Ethernet fibre instead of host DMA. We report two results. First, a correctness result: the network-attached kernel reproduces the DMA-attached kernel's output bit-for-bit across cohorts up to 4,000,003 samples, confirmed in a batch of ten consecutive, independent hardware runs with zero data loss. Second, a systems-debugging result: an initial throughput measurement attempt produced what appeared to be hardware instability -- a third-party network IP core that would intermittently stop accepting traffic regardless of the offered send rate, recoverable only by a full bitstream reload. We show this was not a hardware fault: it was ARP cache staleness in the host's Linux IP stack, silently discarding or delaying the first datagram of a burst sent after any idle period. A three-packet ICMP warm-up immediately before each data burst eliminates the failure entirely (10/10 clean runs post-fix, including a deliberately adversarial run with the ARP entry forcibly deleted). We report this debugging narrative in full because the failure signature -- a third-party accelerator that "just needs a reset" under unpredictable conditions -- is a common and expensive misdiagnosis in FPGA/network co-design, and the actual root cause was one layer up the stack from where we first looked.

**Keywords**: FPGA, Alveo U250, 100G Ethernet, RoCE fabric, network-attached accelerators, early-exit inference, catastrophe scan, VNx, HLS, bit-exact validation, ARP, systems debugging

**Category**: Computer Science > Distributed, Parallel, and Cluster Computing

**License**: CC BY 4.0

**Before uploading**: this paper reports a throughput floor (8 Gbit/s,
confirmed reproducible), not a ceiling — Section 5.3 of the draft is
explicit that the rate sweep needs re-running with the ARP fix in place
before a stronger throughput claim can be made. Consider whether to wait
for that follow-up measurement before submitting, or submit this as-is
with the limitation stated (already written into the draft).

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
