<!-- docs:meta
topic_id: repo.docs.research.erosion-hebbian-result
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.erosion-hebbian-result
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Erosion (Hebbian consolidation): the predicted direction, insufficient magnitude

*Follow-on to `river-variational-and-the-ordering-null.md`. The ordering-null localized the missing
mechanism as **erosion** — the river's flow reshaping the bed (reciprocal coupling), which fixed-landscape
training lacks. This tests it directly: add Hebbian consolidation and ask whether coherent ordering then
beats shuffle. Honest outcome: the direction is confirmed, the magnitude is not enough.*

## The mechanism
A consolidated **bed** `θ_c` (slow EMA of the weights) plus a **Hebbian channel-depth** `Ω` (accumulated
squared gradient per weight — where the flow ran strong, the channel is deep), anchoring the fast weights
to the carved bed: `θ ← θ − lr·(g + μ·Ω·(θ − θ_c))`. Local, reward-free, trajectory-dependent — Hebb. Same
data / init / seeds as the ordering test.

## Result (10 seeds, single-pass online SGD, test accuracy)

| | coherent | shuffled | Δ (coh − shuf) |
|---|---|---|---|
| **no erosion** | 67.51 ± 4.72 | 71.78 ± 2.72 | **−4.27 pp** (2/10) |
| **with erosion** | 69.73 ± 3.26 | 72.40 ± 2.30 | **−2.67 pp** (2/10) |

Erosion helps **coherent more than shuffled** (+2.22 vs +0.62 pp); the gap narrows from −4.27 to −2.67.
**The §4 prediction is confirmed in direction:** erosion cures the forgetting that is specifically
coherent's problem (the consolidated bed protects carved channels). **But it is insufficient:** coherent
still *loses* (−2.67 pp, 2/10) — interleaving still wins at this scale.

## Honest reading
The erosion mechanism is **real and directional but small**. It does not overturn the
interleaving-beats-blocking advantage in small-scale single-pass training. We did **not** tune `μ, α`
upward to force a reversal — that would p-hack a win. So the standing empirical position is:
- the **variational-not-search** framing (conceptual) stands;
- the **order-is-content** claim (coherent beats shuffle) is **not** supported, even with erosion, at this
  scale;
- erosion moves the needle in the predicted direction, which is consistent with — but far from proof of —
  the reciprocal-coupling thesis.

What would legitimately flip it (not to be reached for by tuning): a task where order is **content**, not
curriculum — i.e. a genuinely non-associative composition where `(a·b)·c ≠ a·(b·c)` in the *data-generating
process*, so that shuffling destroys signal rather than removing nuisance correlation. That is the honest
next design, and it is the one place the algebra (not the metaphor) would finally enter the training claim.
Harness `erosion_hebbian.py`.
