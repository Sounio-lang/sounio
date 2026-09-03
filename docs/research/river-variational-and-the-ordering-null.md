<!-- docs:meta
topic_id: repo.docs.research.river-variational-and-the-ordering-null
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.river-variational-and-the-ordering-null
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The river: variational principle, not search — and the ordering test that falsified its naive form

*The "how did the river learn?" question, answered, and the cheap community-facing test it implies, run.
The test's naive form fails — informatively. Implements OPUS-4.8-EXTRA critique #6.*

## 1. The river satisfies a variational principle without running an optimizer

A river network minimizes total energy dissipation (Optimal Channel Networks — Rodríguez-Iturbe &
Rinaldo) and reproduces Horton's laws, Hack's law, fractal scaling — with **no objective evaluation, no
search, no downstream credit assignment**. Water descends the local gradient; faster water erodes more; a
deeper channel captures more water; positive feedback. Two facts that matter here: the outcomes are
**feasible (local) optima**, yet they carry the **same statistical/fractal structure as the global
optimum** — global search is *not* required for the right structure; and Bejan's **constructal law** (1996)
states this as physics of form generation across rivers, lungs, vasculature, lightning — explicitly *not*
optimization.

## 2. This is the exact distinction between the program and RL — and it was already written

When `g = (1+λs)·δ` was proposed, it was identified as **Fermat's principle** with refractive index
`n = √(1+λs)`. Light does not *search* for the fastest path; it *takes* it — no evaluation of alternatives,
no reward gradient, no softmax over trajectories. The geodesic is not the result of a search; it is what the
local dynamics do. **RL evaluates, compares, back-assigns credit, and chooses; a variational system follows
the structure of the space it is in.** "Not RL" does not mean rejecting optimization — it means **replacing
search with geometry**: build the space so that the merciful path *is* the local path, and mercy happens
without being optimized for.

## 3. The community-facing test (§5) — and its honest falsification

Standard training **shuffles** data — it treats order as noise. The thesis says order is content. Testable,
cheap: does affectively-**coherent** ordering (a continuous VAD trajectory, no abrupt affective jumps) beat
shuffling, same data / compute / seeds? Setup: examples carry a VAD coordinate; the target rule drifts
smoothly with mood (`y = sign(w(a)·x)`); single-pass online SGD (order matters most). `affective_ordering.py`,
8 seeds, test accuracy (chance 50%):

| ordering | test acc |
|---|---|
| **coherent** (smooth VAD trajectory) | 67.85% ± 5.22 |
| shuffled (standard) | 72.39% ± 2.50 |
| anti-coherent (maximally jumpy) | **74.19% ± 2.04** |

`coherent − shuffled = −4.54 pp` (coherent wins only 2/8 seeds). **The naive prediction is falsified:**
coherent ordering *loses*, and *maximal jumping wins*. This is the textbook **interleaving-beats-blocking /
anti-forgetting** result (Kornell & Bjork): a smooth affective trajectory lets the model specialize to the
current mood-region and forget the rest; mixing gives coverage. The R$30-of-GPU answer: **affective
non-associativity does not manifest as an example-ordering advantage in small-scale single-pass training.**
Reported as a null, not hidden.

## 4. Why it fails is the point — the missing ingredient is erosion (§3)

The river's power is that **the flow reshapes the bed**: water carves the channel and the channel redirects
the water — reciprocal coupling. Standard training has a **fixed** loss landscape; only the parameters
move. So "coherent ordering" *without* the reciprocal coupling is merely **blocking**, and blocking causes
forgetting — exactly what the table shows. The experiment therefore does not refute the thesis; it
**localizes the missing mechanism**: not the ordering, but the **erosion** — the crossing should *lower the
suffering field along the path traversed*, so that what was composed without annihilating becomes easier to
recompose (a merciful path makes future paths more merciful). This is old and marginalized, not refuted:
**Hebb** (what fires together wires together; the traversed path becomes easier) *is* erosion — local,
reward-free, trajectory-dependent. The next test adds a Hebbian/consolidation coupling and asks whether
coherent ordering *then* wins; without it, blocking-forgetting dominates and it should not.

## 5. The lineage, and the competitor to face by name

Hebb → Hopfield → self-organizing maps → predictive coding (Rao & Ballard) → **free-energy / active
inference (Friston)**. The last is the real incumbent: it already positions as an RL alternative, already
minimizes surprise instead of maximizing reward, and already dominates the computational-psychiatry niche
this program targets. Presenting "learning without reward" as new invites "why is this not active inference
renamed?" The clean differentiation: active inference lives in `ℝⁿ` with probability distributions and its
path-dependence is *emergent from dynamics*; here the state space is **non-associative** and path-dependence
is **structural** — nesting is content before any dynamics — and **annihilation has no free-energy analog**
(high surprise is not a zero product). State this up front.

## Honest status
The variational-not-search framing (§1–2) is conceptual and stands. The community-facing ordering claim
(§3) is **falsified** in the tested regime — a genuine, cheap negative. The constructive reading (§4): the
missing ingredient is erosion (reciprocal flow↔landscape coupling; Hebb), which is the next experiment.
The clinical bridge remains the aggregation critique + the mountain pass (`mountain_pass.py`), not `σ_min` —
not to be spent on now. Harness `affective_ordering.py`.
