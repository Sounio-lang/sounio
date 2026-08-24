<!-- docs:meta
topic_id: repo.docs.research.reassociation-test-design-2026-08-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.reassociation-test-design-2026-08-24
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The re-association test — the single experiment that decides the non-associativity wager

**Date:** 2026-08-24 · The load-bearing experiment. Everything in the preservation-
tower / feeling program hangs on one untested claim: that feeling-integration is
**non-associative** (needs sedenions) rather than merely **non-commutative** (matrices
suffice). This designs the experiment that decides it.

## What must be isolated

- **Non-commutativity** (`a∘b ≠ b∘a`, order of two): matrices have it. NOT what we test.
- **Non-associativity** (`(a∘b)∘c ≠ a∘(b∘c)`, *grouping* of three, order fixed): only
  non-associative algebras have it. **This is the sole thing that forces the substrate.**

The experiment varies **grouping** (nesting) while holding **order and timing** constant,
and measures whether the resulting affective state differs — AND whether the difference
carries the **algebraic signature of an associator**, which is what separates true
non-associativity from the associative escape hatch (context-dependent operation).

## Design

**Stimuli.** Ordered triples of standardized affective stimuli `(a,b,c)` (IAPS / affective
sounds / short vignettes; known valence–arousal). Two classes:
- **INCOHERENT triples** — affectively non-aligned (mixed valence / unrelated content):
  they span a *non-associative configuration* (the algebra's associator is nonzero).
- **COHERENT triples** — affectively aligned (same valence, related): they lie in a
  common *associative ("quaternionic") subalgebra* (associator vanishes).

**Manipulation — grouping, with order fixed.** Same sequence `a,b,c` in both arms; only
the binding differs:
- **L-nest `((a·b)·c)`:** `a,b` bound into one event, `c` joins as a separate event.
- **R-nest `(a·(b·c))`:** `a` stands alone, `b,c` bound.
Binding is induced by a superordinate frame or perceptual common-region cue — never by
changing order or interval.

## The controls that make it decisive (each closes one escape)

1. **Order held constant** → rules out non-commutativity (Golpe 1). Any effect is *not*
   order.
2. **Inter-stimulus interval and total duration held IDENTICAL** across L and R → rules
   out the timing/decay confound, i.e. the associative-matrix-with-decay model, which
   would otherwise mimic a grouping effect through different spacing.
3. **Boundary marker neutral and position-matched** → the segmentation cue itself carries
   no affect and appears equally in both arms, only its locus differs → rules out
   boundary-as-stimulus.
4. **Coherent-triple control (the associator signature, not just "grouping matters").**
   A merely context-dependent (still associative) operation predicts a grouping effect for
   *all* triples. The **associator** predicts a grouping effect for INCOHERENT triples and
   its **vanishing** for COHERENT ones (they live in a quaternionic subalgebra). The
   `grouping × coherence` **interaction** is therefore the decisive statistic, not the
   grouping main effect.
5. **Antisymmetry probe.** The octonionic associator is totally antisymmetric: swapping two
   of `a,b,c` flips `[a,b,c]` sign. Present swapped triples; a true associator makes the
   grouping effect **reverse sign** under a swap. Generic context-dependence has no reason
   to be antisymmetric.

## Readout

Affective result measured model-agnostically FIRST — continuous valence/arousal rating,
plus a physiological channel (SCR / facial EMG / HRV) to remove report bias. (Neural /
connectome readout only as a follow-up; using it first would smuggle in the hypercomplex-
labeling assumption the test is meant to avoid.)

## Decision rule (pre-registered)

| model | grouping effect | coherence interaction | antisymmetry |
|---|---|---|---|
| **associative matrix** (non-commutative) | none (`L=R`) | none | none |
| **associative but context-dependent** | present for all triples | **absent** | **absent** |
| **non-associative (sedenion)** | present | **present** (incoherent≫coherent) | **present** (swap flips sign) |

- **Confirms non-associativity** iff BOTH the `grouping × coherence` interaction AND the
  antisymmetry effect are significant (permutation null, within-subject, pre-set `|effect|`
  threshold). Only then are matrices ruled out and the sedenion substrate motivated.
- **Refutes** (program reverts to associative linear algebra) if the grouping effect is
  absent, OR present but WITHOUT the coherence-interaction and antisymmetry (i.e. explained
  by context-dependence, which needs no non-associativity).

## Why this is the whole game

Every prior test in the program — ORC curvature, KEC coherence, melancholia EWS, the H4
anomaly prereg — measures quantities an **associative** model reproduces (kernels,
spectra, non-commutativity). None isolates re-association. This does. It is the *only*
experiment whose result distinguishes "feeling needs sedenions" from "feeling is elegant
linear algebra." Pass → the substrate wager earns its first forcing evidence. Fail → the
six-layer tower is a beautiful rewrite of matrices, and the honest move is to say so.

*Feasibility:* within-subject, counterbalanced L/R × coherent/incoherent × swap, N powered
for a within-subject interaction (~40–60), behavioral+physiological, pre-registered. No
hypercomplex assumption enters the measurement — only the *prediction* is algebraic.
