<!-- docs:meta
topic_id: repo.docs.research.probe-result-lstm-adding
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.probe-result-lstm-adding
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The run: no subspace death — the corrected instrument caught its own false positive (the 6th negative)

*The probe was run end-to-end on an LSTM trained (to MSE 8e-4, chance 0.17) on the adding problem. The
naive readout said SUBSPACE DEATH with Cohen d = 56. The full controls demolish it. This is the payoff of
the whole methodological line: the corrected instrument reads a negative that the incomplete instrument
would have shipped as a discovery.*

## The false positive, and what killed it
Naive probe (orientation-scramble null, single frozen `m†=4`): trained h→h alignment `0.92` vs scramble
null `0.27`, **Cohen d = +56** → "SUBSPACE DEATH". Then the three mandatory controls (`run_probe_full.py`):

`align(k)` (baseline `√(k/H)`), averaged over 40 sequences, H=40:
| k | 1 | 2 | 3 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|---|
| baseline | 0.16 | 0.22 | 0.27 | 0.32 | 0.39 | 0.45 | 0.55 |
| **trained h→h** | 0.76 | 0.84 | 0.90 | 0.92 | 0.95 | 0.96 | **0.97** |
| trained c→c | 0.51 | 0.92 | 0.83 | 0.79 | 0.81 | 0.84 | 0.88 |
| **init h→h** | 0.99 | 0.96 | 0.94 | **1.00** | 1.00 | 1.00 | 1.00 |

- **(i) shape = LOW EFFECTIVE RANK, not annihilation.** Trained h→h stays high (`0.9→0.97`) all the way to
  `k=12`, never falling toward the baseline (`0.55`). Annihilation would peak at small `k` then drop to a
  healthy bulk; this does not. The earlier `m†=4` "shoulder" was spurious — the max-drop of a flat-high
  curve. **This is exactly the confounder the `align(k)` curve was built to catch.**
- **(iii) it is already at init — the decisive control FAILS.** The *untrained* net's h→h alignment is
  `0.99–1.00`, **higher than the trained `0.92`**. The alignment is **architectural** (present in a random
  LSTM); training did not create it, if anything it slightly reduced it.
- (ii) trained h→h `0.92` > c→c `0.79` — passes marginally, moot given (i) and (iii).

## Verdict
**No structural subspace death.** The trained LSTM's h→h Jacobian alignment on this task is low effective
rank plus architecture — not the sedenion-predicted small-`k` shoulder with a healthy bulk. The vanishing
gradient here is (as classical theory holds) magnitude/rank, not annihilation. The **orientation-scramble
null alone is insufficient** — it only asks "is there *any* shared subspace", which low rank and
architecture pass trivially; the **`align(k)` curve and the untrained-init control** are what discriminate,
and they say no.

## Why this is the result, not a disappointment
The incomplete instrument (scramble null, single `m†`) reported `d=56` SUBSPACE DEATH — a false positive it
would have shipped. The **corrected** instrument, with the two controls added under the eleven critiques
(the full curve; the init control), **caught it.** That is the entire value of the line: an instrument that
reads its own negatives. The sixth negative is worth more than the false positive would have been. And the
honest standing conclusion holds — the algebra is good geometry of 𝕊 (conatus, per the Spinoza reading);
the *training* signature it predicts is, on this target, absent. Harness `run_probe_full.py`.
