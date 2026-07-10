# Non-associative / exceptional algebra in the brain — honest verdict

Synthesis of a deep, adversarially-verified literature sweep (19 claims, 3-vote
verification; the automated synthesis step failed on quota, so this is the hand-merge).
Companion to the ABIDE FC null (`octonion_arc_findings.md` §2). Goal: separate real
signal from analogy from hype on whether non-associative/hypercomplex/exceptional
structure lives in the brain, and name the best falsifiable next test.

## Verdict by front

### Octonionic / exceptional algebra in neural *tissue*: no evidence, and actively bounded out
The best-characterized neural geometry — grid cells — is rigorously a **2D twisted-torus
manifold** that is a representation of a **compact connected *Abelian* (commutative) Lie
group** (Gardner et al. 2022, Nature; Xu et al. arXiv:2210.02684; arXiv:2510.02853, all
3-0). Abelian + toroidal is **incompatible with non-associative/octonionic** structure.
So in the one place neural symmetry is nailed down, it is the *opposite* of exceptional.
No peer-reviewed work ties G2/E6/E8/triality/Jordan algebras to neural data (hypercomplex
DL reviews cover remote sensing/medical imaging/robotics — **zero** neuroimaging). Octonion
non-associativity appears in the literature only as an *engineering* concern in octonion
nets, not an emergent brain property.

### Hypercomplex nets on brain data: efficiency, not structure
Quaternion EEG/neuroimaging nets claim ~4× parameter efficiency, justified by an informal
"naturalness" argument — **not** a demonstrated quaternionic/non-associative property of
neural signals (3-0). On ABIDE specifically, simple baselines are competitive, i.e. fancy
inductive biases yield little (3-0) — consistent with our FC null.

### The TWO real, non-woo open doors

1. **Directed / effective connectivity carries asymmetric structure symmetric FC cannot
   represent** (3-0, multiple: sparse-DCM irreversible component, biorxiv 2025.11.16.688716;
   asymmetric-diffusion prediction, PMC7888488). This is exactly the modality our null did
   NOT test. **But the demonstrated structure is non-commutativity / irreversibility, not
   Cayley-Dickson non-associativity.** The honest test is whether the *composition* of
   directed influences is non-associative, not merely asymmetric (see next section).

2. **Non-associative binding earns its keep in sequence / working memory.** A VSA /
   hyperdimensional model with Cayley-Dickson-style binding where left- vs right-associative
   bundling give distinct states (L-state recency, R-state primacy) **reproduces the human
   Serial Position Curve** (arXiv:2506.13768, 2-1 — single line, weaker). This is the one
   place the "(A then B) then C ≠ A then (B then C)" intuition is literally realized and
   makes a falsifiable behavioral prediction. It is a *cognitive/computational* model, not
   a claim about tissue.

## Best falsifiable next experiment (ranked)

1. **Non-associativity of directed-connectivity composition.** Estimate directed effective
   connectivity (transfer entropy / sparse DCM) → form the path-composition operator → test
   whether (A∘B)∘C ≠ A∘(B∘C) *beyond* what non-commutativity alone forces, with a
   pre-registered associator-style statistic and a phase-randomized / commutativity-matched
   null. This targets the one modality with proven asymmetric structure, and distinguishes
   genuine non-associativity from mere asymmetry. **Prior: most likely reveals
   non-commutativity but associative composition — but it is the honest, untested case.**
2. **Serial-position / sequence-memory associativity probe.** Behavioral + neural (sequence
   replay) test of whether order-binding is left- or right-associative, per the VSA model —
   the one place non-associativity already has explanatory traction.
3. **(Bounded out, do not pursue):** octonionic/exceptional signatures in symmetric FC or in
   grid-cell-like manifolds — the geometry is Abelian/toroidal; the detector is calibrated
   and reads null.

## Bottom line (to dissolve the fear, honestly)
"The brain uses octonions/exceptional algebra" has **no support** and is **bounded out** of
the best-characterized neural geometry. But the broader instinct is not dead: **directed
connectivity has real asymmetric structure left untested**, and **non-associativity has a
genuine, falsifiable home in sequence/working-memory composition**. The honest path is to
test *those* — with the same calibrated-detector + pre-registered-null discipline — not to
keep probing symmetric FC for octonions that the geometry rules out.

Sources: Nature s41586-021-04268-7; arXiv 2210.02684, 2510.02853, 2506.13768;
PMC3040354, PMC7888488, PMC11625415, PMC12513225; biorxiv 2025.11.16.688716;
mdpi 15/21/11526.

## Track C result (2026-06-30): directed connectivity TESTED — real but weak for ASD

Ran the directed-connectivity probe (`scripts/directed_connectivity_test.py`) on ABIDE
CC200 (N=988, LOSO 20 sites). Lag-1 cross-correlation M_ij=corr(x_i(t),x_j(t+1));
directed part K=(M−Mᵀ)/2.

- **Directed structure is REAL:** asymmetry ‖K‖/‖M‖ = **0.291** (substantial, not noise) —
  lead-lag directionality genuinely exists, as the verdict predicted.
- **But its ASD signal is weak:** directed part alone = **53.5%** balanced acc (vs symmetric
  FC **65.9%**), and FC+directed = **62.7%** (worse than FC — the 19 900 weak directed
  features overfit in LOSO).

**Honest read:** the directed/lead-lag modality the verdict flagged is real but, at this
(lag-1 cross-correlation) operationalization, carries only marginal autism-classification
signal and does **not** add to symmetric FC. Caveat: one directed measure, naive
high-dim features, simple linear LOSO — a dimensionality-reduced or model-based (sparse-DCM)
directed estimate could differ. And note: composition of directed connectivity as matrices
is associative, so this does not bear on *non-associativity* — it tests whether directed
*structure* carries signal (it does, but weakly). Door 1 is now tested, not assumed; the
remaining non-associativity home is sequence/working-memory composition (door 2), behavioral
and outside this dataset.
