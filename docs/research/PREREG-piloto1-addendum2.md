<!-- docs:meta
topic_id: repo.docs.research.prereg-piloto1-addendum2
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.prereg-piloto1-addendum2
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Addendum 2 to PREREG-piloto1 — legible verdict: NO BARRIER (positive control + strong LM)

**Date:** 2026-07-20. Closes [addendum 1](PREREG-piloto1-addendum1.md) §3 (the work that had to be done
before the negative was legible). Pre-registration: [PREREG-piloto1-semantic-barriers.md](PREREG-piloto1-semantic-barriers.md).

## Verdict

**The barrier claim is falsified in real semantic fields.** No disconnected sublevel above chance in the
field that carries semantic rupture (PMI), in a single-subject corpus (A = Barb Sanders) **and** an
across-generator aggregate (B = Hall/VdC Norms), under a weak LM (GPT-2) **and** a strong modern LM
(Qwen2.5-Coder-1.5B), with the design **demonstrably able** to detect a real barrier of magnitude ≥~1.5σ.

The pre-registered *strong* falsifier fires: all sublevels connected. The necessary/gratuitous-suffering
distinction (§A.2/§A.3 of Mercyful Learning) — already vacuous in 𝕊 by the connectivity theorem — has **no
empirical referent in real semantic fields either**. The mountain-pass core must be rewritten without it.

## Results — corrected permutation p (c+k swept); p<0.05 = barrier

| field | A (Barb, 46 471) k=5/10/20 | B (Norms, 7 886) k=5/10/20 |
|---|---|---|
| **PMI · GPT-2** (primary; exogenous to the embedding graph) | 1.00 / 0.99 / 1.00 | 0.80 / 1.00 / 1.00 |
| **PMI · Qwen2.5-Coder-1.5B** (modern LM, vocab 151 936) | 0.89 / 1.00 / 1.00 | 0.80 / 0.94 / 0.87 |
| jump (trivial negative control) | 1.00 / 1.00 / 0.99 | 1.00 / 0.99 / 0.88 |
| Ollivier–Ricci curvature (endogenous — see caveat) | 0.005 / 0.005 / 0.005 | 0.005 / 0.005 / 0.005 |

Every PMI p is ≫ 0.05; the k=5-only blips of the first pass (addendum 1 §2) are confirmed null by the
corrected p. The trivial control is null as designed.

## Positive control — the sensitivity floor δ\* (addendum 1 §3.1; also the O-SSM injected-structure recovery)

Inject a barrier of magnitude δ (in units of the field's σ) on a Fiedler vertex-separator of the real
mutual-kNN graph and sweep δ. Detection = corrected permutation p < 0.05.

| δ (σ) | A: p | B: p |
|---|---|---|
| 4.0 | 0.005 | 0.005 |
| 2.0 | 0.005 | 0.025 |
| 1.5 | 0.005 | 0.139 |
| 1.0 | 0.114 | 0.463 |

**The design reliably detects an injected barrier of ≥~1.5σ (A) / ~2σ (B) and fails below ~1σ.** So the
negative reads as: *no barrier, and one of magnitude ≥~1.5σ would have been caught.* This simultaneously
discharges the O-SSM injected-structure-recovery debt — the instrument recovers structure planted by
construction.

## The curvature field lights up — and it is circular, not a barrier

Ollivier–Ricci gives p = 0.005, k-stable, in both samples. This is **not** independent evidence. The field
`s_curv = −κ` is high *exactly at graph bottlenecks* (negative curvature = bridge), and the test asks whether
removing high-s nodes disconnects the graph — which removing bridges does *by construction*. The field is
**endogenous** to the graph; testing it against that graph's connectivity is near-tautological. It shows
that MiniLM sentence embeddings of dream reports form modular clusters with thin bridges (a property of the
embedding geometry), and it serves as a *second* positive control confirming the test has power — but it is
**not** a semantic barrier. The two fields that could carry semantic rupture exogenously to the graph —
PMI (GPT-2 and Qwen) — are negative.

## Scope (unchanged from the pre-reg §8, reaffirmed)

Not about psychosis (dream ≠ psychosis). Not about suffering (measures informational composition failure).
Does not establish that real trajectories cross a barrier. What it establishes: the geometric premise of the
Mercyful-Learning formal core has **no** empirical referent in these real semantic fields, individual or
aggregate, robust to the LM. The person-specific escape (addendum 1 §1) did **not** open — A is negative.

## Reproduction / infra

- Fields/analysis: `docs/research/pc_core.py` (positive control, corrected p), `node_all_v2.py`
  (3 fields + PC), `node_qwen.py` (strong-LM PMI). Data: dreambank.net, CC BY-NC-SA 4.0.
- Compute: GPT-2 PMI + curvature on BEAGLE `cpu-ops` (32 dedicated cores). Qwen-1.5B PMI on a **DGX Spark
  (`spark-3c59`, GB10 Blackwell sm_121, torch 2.11.0+cu128)** — the BEAGLE GPUs were shared/OOM; the DGX GPU
  was dedicated and free. Models: `gpt2`, `Qwen/Qwen2.5-Coder-1.5B`; embeddings `all-MiniLM-L6-v2`;
  transformers 5.14.1.
- Infra note carried from addendum 1 §4: the pip SSL-verify bypass was **BEAGLE-nodes-only** (broken system
  CA bundle); the DGX Sparks have valid CA bundles and needed no bypass. Not a lab default either way.
