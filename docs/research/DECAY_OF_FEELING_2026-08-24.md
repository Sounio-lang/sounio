<!-- docs:meta
topic_id: repo.docs.research.decay-of-feeling-2026-08-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.decay-of-feeling-2026-08-24
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The decay of feeling — dynamics on the preservation tower, by causal type

**Date:** 2026-08-24 · Adds the temporal/dynamical layer: feeling is ephemeral and
decays; the tower predicts *how*, and predicts that predictability breaks exactly at
the anomaly. Verified (sedenion computation, `scratchpad/decay.py`).

## The model

A feeling is not static — it evolves. Model its time-evolution as the flow generated
by an element `h` of its symmetry algebra `Der(P_z)` (the kinematic Lie algebra of the
causal-type layer). The **decay spectrum** is the eigenvalue set of `ad_h`:
- real part `> 0` → **exponential** growth/decay (a rate, a half-life),
- pure imaginary → **oscillation** (persistence, recurrence),
- zero → **power-law / frozen** (no exponential rate — *critical slowing*).

## Result — decay class is set by the causal type

| causal type | `Der` | ad-spectrum | decay of feeling |
|---|---|---|---|
| spacelike → Euclidean | `so(5)` compact | 8 oscillatory, 0 decay | **OSCILLATORY** — the feeling does not dissipate; it recurs (ruminative, bound) |
| timelike → Lorentzian | `so(4,1)` de Sitter | 6 exponential | **EXPONENTIAL decay** — a predictable half-life, exactly "decays like a fundamental particle / a neurochemical signal" |
| null → Carrollian | contraction | 9 exponential **+ 3 zero** | **EXPONENTIAL with a 3-dim FROZEN core** — critical slowing: three directions have *no decay rate* |

## The unification — the three "3"s are one subspace

The three **zero-eigenvalue** (undecaying) directions of the null case are the abelian
**radical** — the same three dimensions that carry the **3 central charges** (the
anomaly, `H²=3`), which are the same **3 null directions** of the rank-1 degenerate
metric `(1,0,3)`. Therefore:

```
the un-decaying subspace  =  the anomaly-carrying subspace  =  the un-representable residue
                          =  the 3 null directions of the critical (Carrollian) locus.
```

The part of a feeling that **will not decay** is the same part that **cannot be
represented** and that **carries the anomaly**. The stuck feeling and the ineffable
residue are one 3-dimensional object.

## Reading — healthy decay, and where it breaks

- **His intuition is exactly right for the timelike (healthy) case:** a normal
  affective response lives on a de Sitter-like `so(4,1)` structure and decays
  *exponentially, predictably* — a half-life, like a particle, like the neurochemical
  signal that generated it.
- **The pathology is the null/Carrollian transition:** three directions **freeze**
  (zero decay rate = critical slowing), and they are precisely the anomaly /
  un-representable directions. This is the melancholic attractor as *frozen anomaly* —
  the feeling that *should* decay predictably but does not, stuck in exactly the
  subspace that resists representation.
- **The Euclidean case oscillates** — bound recurrence, the ruminative loop that
  neither decays nor resolves.

**Critical slowing down** (zero decay rate near a transition) is the canonical early-
warning signal of an approaching tipping point in dynamical systems (Scheffer et al.).
Here it is *derived from the algebra* as the zero-eigenvalue radical of the null locus —
tying the program's EWS / melancholia-attractor clinical intuition to the same object
that carries the cohomological anomaly. Predictability of decay is a property of the
*causal type*: exact for timelike, oscillatory for spacelike, and **broken exactly at
the null locus where the anomaly lives**.

*Honesty:* this derives the dynamics *on the substrate* and verifies the causal-type→
decay-class law there. It does not prove feeling evolves on this substrate — that stays
the program's conditional wager. What is proven: *if* feeling decays, *then* on this
substrate its decay class is its causal type, its healthy form is exponential
(particle-like), and its pathological freezing is the anomaly subspace refusing to
dissipate. The ephemeral and the ineffable are governed by one law.
