<!-- docs:meta
topic_id: repo.docs.papers.epistemic-types.confidence-semantics
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.epistemic-types.confidence-semantics
-->

# Confidence Semantics for the Epistemic Type

*Addresses §3.4 of the PLDI submission cycle-1 review.*

---

## §1. The Reviewer's Objection

The `confidence` field is an integer in [0, 1000]. The per-operation decay rules
(`× 98/100` per `mul`, `× 99/100` per `add`) were presented without a stated
denotational semantics, leaving the reviewer unable to determine whether the field
carries probabilistic content, is a heuristic scalar, or is purely nominal.

This document states a precise semantics and shows the decay rules are *derived*
from it, not chosen arbitrarily.

---

## §2. Epistemic Pedigree Depth Semantics

**Definition.** Let `d(e)` be the *composition depth* of expression `e`: the number of
non-trivial operations (any operation other than `measured` or `certain`) in the
derivation chain that produced `e`. Let `D_max = 1000`. Then:

```
confidence(e)  :=  1000 · (1 − d(e) / D_max)
                =  1000 − d(e)      (when decay factor = 1 per step, idealized)
```

Each composition step represents one opportunity for **model misspecification**: an
approximation that was used (first-order Taylor, independence assumption, a unit
conversion, a functional form for a coupling constant running) may fail to hold for the
specific value being propagated.

The per-step decay factors encode a conservative per-step validity probability `p`:

| Operation | Decay  | Interpretation |
|-----------|--------|----------------|
| `mul`     | 98/100 | p = 0.98 per multiplicative combination |
| `div`     | 97/100 | p = 0.97 (division is one extra approximation: y⁻¹ expansion) |
| `add`     | 99/100 | p = 0.99 per additive combination |
| `square`  | 98/100 | p = 0.98 (special case of mul, same depth cost) |
| `sqrt`    | 98/100 | p = 0.98 (one nonlinear approximation) |
| `scale`   | 1.00   | exact scaling, no new approximation |
| `shift`   | 1.00   | exact shift, no new approximation |

**Survival probability interpretation.** A chain of `n` independent steps each with
validity probability `p` has survival probability `pⁿ`. For `p = 0.98` and `n = 50`:

```
0.98⁵⁰ ≈ 0.364
```

The corresponding `confidence` value after 50 `mul`-class steps from a root at 1000 is:

```
1000 · 0.98⁵⁰ ≈ 364
```

This is not a formal frequentist probability — it is an **ordinal pedigree score**
calibrated so that the numerical value is interpretable as a rough survival fraction under
the stated per-step model. The exact decay constants are chosen to be conservative and
round-friendly; they are not fit to data.

---

## §3. Constructor Semantics

**`Epistemic::measured(val, std)`** constructs from a direct experimental measurement.
`d(e) = 0`, so `confidence = 1000`. The variance is `std²`. This is the epistemic floor:
a direct PDG measurement is taken as a calibration point.

**`Epistemic::certain(val)`** constructs an exact constant (e.g., `c`, `ħ` in natural
units, integer charges). `confidence = 1000`, `variance = 0`. No uncertainty source and
no composition depth.

Both constructors are the only sites where `confidence` is initialized; all subsequent
values are derived monotonically downward by the operation table above.

---

## §4. Gate Semantics

```
ep_require_conf(e, min_conf)
```

Returns `e` unchanged if `e.confidence >= min_conf`. Otherwise returns:

```
Epistemic { val: e.val, variance: e.variance, confidence: 0 }
```

The zero-confidence sentinel propagates through all subsequent operations (since
`min(0, c) = 0` for any `c`, and `0 × k = 0` for integer multiplication). A downstream
`ep_require_conf` at any threshold will therefore reject it.

**Semantics of the gate:** this is not a probability bound. It is a **pedigree depth
gate**: it filters results whose derivation chains are longer than what the caller
declares is acceptable. A `min_conf = 800` gate means "I trust only results derived from
no more than ~10 multiplicative steps from raw measurements." The caller is asserting a
structural property of the derivation, not a frequentist coverage claim.

This design choice is deliberate: the alternative — requiring the caller to compute
required coverage — would demand a full Bayesian network over the derivation DAG, which
is not available at the call site.

---

## §5. Limitation: Uncertainty Type Is Not Tracked

The current semantics does not distinguish between:

- **Type A (statistical) uncertainty**: reducible with more data — e.g., a Monte Carlo
  integral estimate whose variance decreases as O(1/N).
- **Type B (systematic) uncertainty**: irreducible model uncertainty — e.g., the choice
  of renormalization scheme in a QCD calculation.

The confidence score decays identically for both, which conflates depth-of-composition
with depth-of-irreducibility. A future extension would parametrize `confidence` as a pair
`(depth_score, systematic_fraction)`, where `systematic_fraction` tracks what fraction
of the accumulated variance is believed to be irreducible. This would allow a downstream
consumer to separately gate on "is the derivation shallow?" and "is most of the
uncertainty reducible?".

Until that extension is implemented, consumers should treat `confidence` as a pure
**pedigree depth score** and not over-interpret it as a Type A/B decomposition.
