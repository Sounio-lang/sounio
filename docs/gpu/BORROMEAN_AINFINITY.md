<!-- docs:meta
topic_id: repo.docs.gpu.borromean-ainfinity
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.gpu.borromean-ainfinity
-->

# A∞ / higher-homotopy: Borromean triple-linking where the higher invariant is *required*

The novelty map (§6.1) named A∞ / higher-homotopy as the door where non-associativity is *intrinsic*
rather than injected: associativity holds only up to a homotopy whose first obstruction is the associator
(the A∞ operation `m₃` = the Massey product). This experiment realizes it on a **genuine mathematical
structure** and a **real ML data type** — the **Borromean** configuration read through **path
signatures / iterated integrals** (rough-path features, used in time-series ML).

## The structure
Borromean rings: three components **pairwise unlinked** (all pairwise linking = 0) yet globally linked.
For paths, the level-2 invariants are the **Lévy areas** `A₁₂,A₁₃,A₂₃` (associative, shuffle-reducible);
the triple linking is a **level-3 iterated-area** invariant `μ_k = ∫ A_ij(t) dX^k` — the linking of
component *k* with the area swept by the other two. This is the Massey `m₃` term. On the **pairwise-trivial
slice** (|Lévy areas| ≈ 0 — the Borromean regime), the associative invariants are **blind by
construction**; only the higher invariant carries the signal.

Computation self-checked: the discrete Lévy area of a unit circle is `π` to 1e-2; the triple invariant
is genuinely nonzero on the slice (std ≈ 9.5, not machine noise — an earlier fully-antisymmetric
"signed-volume" candidate was identically zero, a real fact: the genuine invariants live in the free Lie
algebra / log-signature, not in Λ³).

## Ablation (test accuracy on the pairwise-trivial slice; chance = 50%)
| Feature | acc | reading |
|---|---|---|
| ENDPT — level-1 displacement (associative) | 48.6% | blind |
| ASSOC — level-2 Lévy areas `A₁₂,A₁₃,A₂₃` (associative) | 48.4% | **blind by construction** |
| OCT — octonion associator of the 3 coordinate increments | 48.9% | **the bridge does NOT capture it** |
| **HIGHER — level-3 iterated area `μ` (Massey m₃)** | **99.8%** | defines the label |
| MLP — on the raw path `T×3` | 56.1% | unstructured baseline barely learns it |

## Two honest findings
1. **The A∞ door is real.** On a genuine topological structure (Borromean paths), in the regime where
   associative invariants vanish, the higher-homotopy (Massey) invariant is *necessary and sufficient*,
   and an unstructured MLP on the raw path reaches only 56%. This is the strongest "non-associativity is
   intrinsic, not injected" instance in the trilogy — the invariant is real mathematics, computation
   verified.
2. **The octonion associator, naively applied, is NOT this invariant** (48.9%, chance). The path's Massey
   product and the octonion associator are **distinct** non-associative structures: our tensor-core
   primitive computes the latter, and a static triple of coordinate increments does not encode the
   *sequential/temporal* iterated structure the Massey product measures. A faithful bridge — an
   **octonion-valued path signature** whose higher term is an iterated non-associative bracket — is left
   as the honest open follow-up. We do **not** engineer the embedding until OCT crosses chance.

## Where this leaves the empirical program
Three experiments now bound the claim precisely:
- `NONASSOC_HEADTOHEAD.md` — non-associativity *constructed* as the signal → octonion associator solves it.
- `BRACKETING_TASK.md` — non-associativity *definitional* (evaluation order) on realistic symbolic inputs
  → octonion associator reads it (95.9%), associative blind.
- **this** — non-associativity *intrinsic* (A∞/Massey) on a genuine topological structure → the higher
  invariant is required (99.8%), associative blind, **and our octonion primitive is a different
  non-associative object** (honest negative bridge).
- `ABIDE_ASSOCIATOR_NULL.md` — a *natural clinical* dataset where non-associativity is absent (positive
  control at 63.9%).

The condition for the octonion tool to pay off is now sharp: the signal must be a **static** ternary
(Cayley-Dickson-type) associator, not merely *any* higher-homotopy obstruction. Harness
`borromean_signature.py`.
