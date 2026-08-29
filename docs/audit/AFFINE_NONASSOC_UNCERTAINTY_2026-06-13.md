<!-- docs:meta
topic_id: repo.docs.audit.affine-nonassoc-uncertainty-2026-06-13
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.affine-nonassoc-uncertainty-2026-06-13
-->

# Affine arithmetic over a non-associative algebra: non-associativity as a noise symbol

Status: **frontier opened — core mechanism designed, N=3 case proven (paper + execution), novelty boundary established against prior art.**
Branch: `feat/affine-nonassoc-uncertainty`. Demonstrator: `examples/epistemic/affine_nonassoc_demo.sio` (self-contained; verified green via `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run ...` on 2026-06-30).

## The frontier target

The reject-defaults map named the next default to reject: **"independence is a lie" generalized from the perturbation-DAG to every `+` and `*`.** The existing uncertainty substrate already folds non-associativity into a *scalar* variance budget — `UncertainOct.mul3` and `PerturbGraph.pg_mul` add `κ·‖[a,b,c]‖²`. But both assume **leaf independence**: `pg_mul` combines operands as `‖j‖²·Var(i) + ‖i‖²·Var(j)`, with no representation of shared error sources. That independence assumption is exactly the default to reject.

**Affine arithmetic (AA)** is the correct substrate for that rejection. A value is
```
x = x0 + Σ_k  d_k · ε_k
```
where `ε_k ∈ [−1,1]` are shared scalar noise symbols and the coefficients `x0, d_k` are octonion-valued (`[f64;8]`). Two values that share a symbol are *correlated through it*; a symbol with equal-and-opposite coefficients cancels exactly. Correlation is structural, not a separate covariance matrix. The one budget is `Var(x) = Σ_k ‖d_k‖²`.

## The crux that forks the design (resolved)

The naive pitch — "a non-associative product injects one scalar structural symbol `κ‖[a,b,c]‖²`" — is **false**, and the falsity is the whole point. Take `a = a0 + a1·ε`, `b = b0`, `c = c0` (b, c exact):

| association | ε-coefficient |
|---|---|
| `(a·b)·c` | `(a1·b0)·c0` |
| `a·(b·c)` | `a1·(b0·c0)` |
| **difference** | **`[a1, b0, c0]`** (the associator) |

Non-associativity perturbs **every shared symbol's coefficient**, and the perturbation stays *correlated* with whatever owns that symbol. The order-ambiguity does not live in a separate independent scalar add-on — it is distributed across the symbol row. The scalar `κ‖[a,b,c]‖²` model is a lossy projection of this onto an independent budget.

This forked two designs:
- **Path 1 (leaf-correlation only):** shared symbols at leaves for additive correlation; keep the norm-rule + scalar structural term for products. Correct and shippable, but this is *textbook AA applied* — not frontier.
- **Path 2 (octonion-coefficient affine forms propagated through products):** confronts the per-symbol associator spread. **This is the research and the novelty.** N=3 is closed and buildable (matches the already-proven "DAG order-safe IFF N≤3"); N≥4 needs the associahedron machinery (`pentagon_variance`, commit `73fa72b91`) because there the spread is over 5 associations (Catalan C₃).

We are on **Path 2**.

## Verified result (N=3)

`examples/epistemic/affine_nonassoc_demo.sio` implements octonion-coefficient affine forms with a first-order product rule `∂/∂ε_k = a0·d_k(b) + d_k(a)·b0` (octonion order preserved, hence non-associativity preserved). It asserts three claims numerically and **runs green on the verified lean_single lane**:

1. **Exact conditioning = zero covariance.** `x − x` → affine `Var = 0`; the independence-assuming scalar model is forced to report `2·Var(x) = 0.08`. (Realizes the exact-conditioning/zero-covariance lead, arXiv:2312.17141, as the canonical AA cancellation.)
2. **Correlation honesty.** `Var(x+y)` = `4‖d‖²` when `x,y` share a symbol vs `2‖d‖²` when independent — tracked, not assumed.
3. **Non-associativity is a noise-symbol perturbation.** For `[e1,e2,e4]` (non-Fano, `‖associator‖² = 4`), the ε-coefficient spread between `((ab)c)` and `(a(bc))` equals `[a1,b0,c0]` to machine zero. The order-ambiguity rides the *same symbol* as `a`'s input uncertainty.

The N=3 hand-calc is therefore proven both on paper and by execution.

## Novelty boundary (honest — do not round up)

A focused prior-art sweep (9 queries) places the boundary precisely:
- **Fusing roundoff + epistemic/input uncertainty in one affine form is textbook** — Stolfi & de Figueiredo. The fresh-symbol-per-nonlinear-op *is* the roundoff term. Not novel.
- **"Associator as uncertainty" exists as a concept** in non-associative quantum mechanics (arXiv:1411.3710; PRL 121.201602) — but as a *bound on simultaneous measurability of observables*, not a propagated variance carried inside a value representation. Adjacent, different direction.
- **No prior art** for: AA operating *over a non-associative algebra*, with the associator `[a,b,c]` injected as a *first-class affine noise symbol* so product order-ambiguity becomes propagated variance, fused with roundoff and epistemic symbols — nor for the **PL/type-system framing** of it. The closest computational analog (ZC502 Isaac-Sim physical-consistency plugin, Feb 2026) independently mechanizes the octonion associator as a numerical-drift diagnostic but makes the *opposite* modeling choice (structural signal, explicitly *not* noise, no AA).

**The novel residue is the narrow fused mechanization + its language framing, not any single ingredient.** Claims must be scoped to that.

### Key references
- Stolfi & de Figueiredo, *Affine Arithmetic: Concepts and Applications* — roundoff + input uncertainty fused via shared noise symbols.
- arXiv:1411.3710 / PRL 121.201602 — associator bounds an uncertainty (concept-level prior art, different sense).
- ZC502, *Isaac-Sim Physical-consistency plugin* v0.4.2 (2026) — associator as drift diagnostic, opposite choice, no AA.
- Goubault & Putot, perturbed affine arithmetic / zonotopes (arXiv:0807.2961) — AA-as-error-tracking over ℝ (associative), the machinery this extends.

## Roadmap

1. **[done]** Self-contained N=3 demonstrator, runs green on `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run ...`.
2. **Promote to a stdlib module** `stdlib/epistemic/affine_octonion.sio` — arena-backed (mirror `perturbation_graph.sio`'s flat-arena, bare-`&!`-deref discipline to avoid the large-boxed-struct codegen hazard), `NSYM` symbols, second-order remainder lumped into a fresh roundoff symbol with a documented magnitude bound.
3. **Upgrade `PerturbGraph` to correlation-honest** — replace scalar `pvar[n]` with a noise-symbol row; the current independent combination becomes the special case of disjoint symbol sets. This retires the leaf-independence assumption in the live substrate.
4. **[done 2026-06-15]** N≥4 spread — `examples/epistemic/affine_nonassoc_n4_demo.sio` wires the ε0-coefficient rows of all 5 affine parenthesizations of w·x·y·z into `pentagon_variance`. Claim 4: max component-wise diff (affine coeff row vs plain pentagon on w1,x0,y0,z0) = 0 machine-exact; variance = 0.96 (×1e6, genuinely non-associative non-Fano quad e1,e2,e4,e1). Claim 5: all-real quad → variance = 0 (reals associate). ALL CLAIMS VERIFIED by live run on 2026-06-15. `pentagon_variance` is the correct path-independent order-ambiguity measure when N>3.
5. **Type-system framing** — surface `NonAssoc` (already an effect) as the carrier: a non-associative product *is* a variance-producing effect, and the affine form is its propagated witness. This is the PL-novel half.

## Why this is the right "big step"

It advances the telos (uncertainty-as-default-substrate, "my mind and the world") rather than the toolchain that serves it, it is falsifiable and already falsified-green at N=3 on the verified lean_single lane, and the prior-art sweep shows the narrow mechanization is unworked. It does **not** block on the arm64 self-host, so it proceeds in parallel with closing that out.
