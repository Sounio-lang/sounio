<!-- docs:meta
topic_id: repo.docs.gpu.bracketing-task
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.gpu.bracketing-task
-->

# A task where non-associativity *matters* — evaluation-order (bracketing) on realistic symbolic inputs

ABIDE showed a real dataset where non-associativity carries no signal (`ABIDE_ASSOCIATOR_NULL.md`). The
complementary question (novelty map §6.1): is there a task with **non-toy inputs** where non-associativity
is *required*? Yes — when the generating process is itself non-associative. This is the cleanest honest
instance: **evaluation-order (bracketing) discrimination.**

## The task
- **Inputs** — a realistic symbolic distribution: a **Zipfian** vocabulary of 64 symbols and length-4
  token sequences (the statistics of language, not random octonions).
- **Label** — for a sequence `s=(s1,s2,s3,s4)`, `y = 1[⟨w*, r1 − r2⟩ > 0]` where
  `r1 = ((s1·s2)·(s3·s4))` (balanced bracketing) and `r2 = (((s1·s2)·s3)·s4)` (left bracketing), and `·`
  is octonion (non-associative) multiplication of fixed teacher embeddings.

The discriminant `r1 − r2` is a **bracketing associator** — identically **0 for any associative algebra**.
So the label is *undecidable* to an associative model and *decidable* to a non-associative one, by
construction. Median-thresholded to a 50/50 split; train 6000 / test 2000.

## Ablation (all trained; embeddings fixed & shared, each model learns its head)
| Model | test acc | note |
|---|---|---|
| **OCT** — bracketing associator `r1−r2` (non-assoc) + logistic head | **95.9%** | reads the label |
| **QUAT** — the *same* feature in the associative 4-dim subalgebra | 49.9% | `‖r1−r2‖ = 2.2e-16` → structurally zero, blind |
| LINEAR — logistic on raw concatenated token embeddings | 51.6% | chance (label is quartic) |
| MLP 32→64→1 on raw token embeddings | 58.4% | partial — capacity/data baseline |

**OCT and QUAT differ only in the algebra** (non-associative vs the associative subalgebra), same feature
map, same head, same data. The 95.9 vs 49.9 gap is attributable **entirely to non-associativity**: the
associative version of the identical computation is zero to machine precision and therefore blind. The
manual octonion-product backward is finite-difference checked (`gradient check: PASS`).

## Honest scope
Semi-synthetic: the **inputs are a realistic symbolic distribution**, the **label is constructed** to be
a bracketing associator. This is exactly what novelty §6.1 asked for — *"non-associativity is the label
by construction, yet the inputs are a real symbolic distribution."* It demonstrates the mechanism matters
on non-toy inputs; it is **not** a claim that a *natural* dataset was found in which non-associativity was
*discovered* (the A∞/higher-homotopy and exceptional-physics doors remain open for that). Together with
`NONASSOC_HEADTOHEAD.md` (constructed) and `ABIDE_ASSOCIATOR_NULL.md` (natural, null), this bounds the
empirical claim precisely: **when evaluation order under a non-associative operation is the signal, the
octonion associator reads it and every associative model is blind — and that condition is what to look for
in real data.** Harness `bracketing_task.py`.
