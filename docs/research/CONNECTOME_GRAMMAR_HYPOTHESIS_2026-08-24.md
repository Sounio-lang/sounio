<!-- docs:meta
topic_id: repo.docs.research.connectome-grammar-hypothesis-2026-08-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.connectome-grammar-hypothesis-2026-08-24
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Hypothesis: a generative grammar for mental states / connectomics

**Date:** 2026-08-24 · **Status:** HYPOTHESIS (bridge from the preservation-algebra
grammar to connectomics). NOT a result. Companion to DERIVATION_GRAMMAR_2026-08-24.md
and the ECSS / ORC / melancholia-attractor program.

## The bridge

The derivation grammar (seed = a zero-divisor `z`; typed functorial rules
ker→stab→Der/TKK→H²; causal type `c(z)=sign Q(z)` gating productions) is a candidate
*template* for a generative mathematics of mental states, under one identification:

| grammar object | connectome / mental-state reading |
|---|---|
| zero-divisor `z` (annihilation locus) | a **gating/inhibition state** — a configuration that can exactly silence certain contents |
| `ker L_z` (what `z` annihilates) | the **suppressed subspace** (inhibited contents) |
| `P_z` (preservation algebra) | the operations that **keep the suppression intact** — what cognition can do while holding something silenced |
| causal type `c(z)` | the **stability class** of the state |
| — spacelike (Euclidean, rigid, `H²=0`) | a **stable attractor** — clean composition, no anomaly |
| — **null (Carrollian/BMS, `H²=3`)** | a **critical/transition** state — on the light-cone, the anomaly turns on |
| — timelike (Lorentzian/de Sitter) | a **dynamical/evolving** state |
| the grammar itself | the **developmental/dynamical rules** generating reachable mental configurations (cf. the ECSS latent SSM) |

## The one falsifiable export

The tower's sharpest structural fact is **null-exclusivity of the anomaly**: central
charges (`H²(g;ℝ)>0`) appear ONLY at null loci; spacelike/timelike are anomaly-free.
Transported:

> **HYPOTHESIS.** A connectome invariant analogous to the central charge (an
> "anomaly" of the state's preservation symmetry) is nonzero ONLY at *critical /
> transitional* mental states, and vanishes at stable attractors.

This is directly testable against existing data, and it *resonates* with prior
findings that were NOT designed to look for it:
- ORC depression: the **subclinical minimum is most hyperbolic** (non-monotonic) —
  a criticality signature at a "boundary" state, matching a null-locus anomaly.
- **Melancholia as a painful attractor** — a critical basin where the anomaly-
  carrying structure would concentrate.

## Honest risk (the prior)

This program's history is full of *honest nulls* when beautiful hypercomplex
structure met data: the G₂ bridge (closed null, `z≈2` combinatorial artifact),
affect-network curvature EWS (reduced to network density), KEC-α (replicated null).
The base rate says elegant algebra often does not survive contact with connectomes.
So this is a hypothesis with a falsifier, not a claim — its value is that it is
*specific* (null-exclusive anomaly = criticality) and testable on data already in
hand, and its failure would be as informative as its success.

## Concrete next step (if pursued)

1. Define the connectome "zero-divisor" (a gating state) and its "central charge"
   (a cohomological / anomaly invariant of its preservation symmetry) precisely on
   the octonion/sedenion-labeled graph.
2. Compute it across the ORC depression-severity trajectory.
3. Test: does the anomaly light up ONLY at the critical (subclinical-minimum /
   melancholic) states? Null-exclusivity is the pass/fail.

---

## Test run on the ORC depression data (2026-08-24) — SUGGESTIVE, honestly weak

Ran the null-exclusive-anomaly test against the real semantic-ORC depression data
(SWOW-EN graph, `examples/semantic_orc/depression_semantic_orc.sio`, values from
`results/depression_curvature_details.json`). Mapping: the **GCI** (graph curvature
inhomogeneity = fraction of edges with `|κ|>2σ`) is the data-analog of my
"anomaly / degeneracy" (the null locus is characterized by a *degenerate*, rank-
collapsed preservation form — GCI measures exactly that inhomogeneity).

| band | mean κ | σ | GCI (anomaly analog) |
|---|---|---|---|
| minimum (subclinical) | −0.127 | 0.238 | **0.412** ← peak |
| mild | −0.071 | 0.206 | 0.287 |
| moderate | −0.065 | 0.222 | 0.261 |
| severe | −0.074 | 0.207 | 0.298 |

**Prediction shape vs data shape:** the hypothesis says the anomaly peaks at the
critical/null state; the GCI peaks at the subclinical minimum (0.412 vs 0.26–0.30),
which is also the most-hyperbolic (transition-onset) state. **The shapes match.**

**Why this is WEAK, stated in full (the honesty is the point):**
1. **n = 4 underpowered bands, overlapping CIs.** The source file itself says
   "minimum and mild CIs overlap — groups not fully separated; requires larger n."
   No inference is established — it is a pattern in 4 noisy points.
2. **Re-reading, not fresh computation.** The κ/GCI are hardcoded prior results;
   this checks whether an existing result matches my prediction's *shape*, it does
   not compute a new anomaly invariant.
3. **Post-hoc mapping.** I chose "GCI = anomaly analog" after seeing it peaks at the
   minimum. That is the post-hoc-bridge trap named explicitly.
4. **Strong prior against.** The same program's KEC-α structural metric on the DCT
   was a clean, replicated NULL (only valence_bias carried signal). Structural
   signals in this program mostly do not survive contact with data.

**Verdict.** The ORC data is *consistent with* the predicted shape (anomaly peaks
at criticality) — it survived first contact, unlike KEC-α — but this is a weak,
post-hoc, underpowered qualitative match, **nowhere near confirmation**. Its only
real value: it justifies a *pre-registered* test and names it exactly.

**The pre-registered test that would make it real:** fix, before looking, on a
larger cohort (e.g. LEMON N≈227, already in the prereg pipeline): (a) a proper
cohomological anomaly invariant of the preservation symmetry (not GCI, which is a
proxy), (b) the causal-type→band mapping, (c) the hypothesis "anomaly is maximal at
the transition band and lower at stable bands," with a permutation null. Pass/fail
on that — not on this post-hoc shape-match — is the actual science.
