<!-- docs:meta
topic_id: repo.docs.compiler.sc-grammar-and-typing
authority: repo_only
audience: contributors
last_validated: 2026-11-26
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.sc-grammar-and-typing
-->

# Sounio Core (SC): grammar and typing rules — ADR-032 step 1

This is the step-1 artifact of [ADR-032](../decisions/adr-032-epistemic-typed-core.md). The ADR names three parts: this document, the Lean fragment [`formal/SCCore.lean`](../../formal/SCCore.lean), and the step-1 witness programs. §1–§4 specify SC. §5 states what the Lean fragment covers and proves. §6 lists the obligations, each with its status.

Claim labels follow the project's convention: OBSERVED (seen in a run), ESTABLISHED (proved or measured with a stated model), NOT ESTABLISHED.

## 1. Shape: a graph with explicit effect edges

ADR-032 decision 1 (2026-11-25) fixes the shape. An SC module is a directed graph.

- **Nodes** carry a canonical id (§3) and one of the constructors in §2.
- **Data edges** run from an operand to its use. A value shared by two uses is one node with two outgoing data edges. In tree notation this is a `let`: sharing is expressed by binding, never by duplicating a subtree.
- **Effect edges** order the nodes that perform effects (`IO`, `Mut`, `Alloc`, `Panic`, `Div`, `Epistemic`). They form a chain per function body, so the lint, the semantics and the lowerer all read the order from the same structure (Bračevac-style).
- **Source edges** run from each Knowledge-typed node to the Knightian sources (§3) its value may carry. Statically they appear as the K set of the node's type (§4).

The tree notation below is the textual form of this graph, and it is the form the Lean fragment uses.

## 2. Grammar

```
e ::= n                          integer / f64 literal
    | knight(s, lo, hi)          named Knightian source s with declared bounds
    | gauss(s, sd)               named aleatory source s (GUM standard uncertainty)
    | x                          variable
    | let x = e in e             sharing (one node, many uses)
    | e + e | e - e              linear
    | c * e                      scaling by a constant
    | e *ₙ e                     bilinear at node n (may create a residue)
    | f@ₛ(e)                     call of an analysed function at call site s
    | opaque f(e)                call summarised as Unknown (recursion, extern)
    | require(e)                 precise requirement: no Knightian source (E230/E313/E314)
    | requireExact(e)            exact requirement: no source at all
    | (e, e)                     aggregate (pairs in the Lean fragment)
    | e.fᵢ                       field projection (e.0, e.1 in the Lean fragment)
```

Function bodies are SC expressions whose parameters are variables. Every function is defined before its callers, a rule Sounio's surface syntax already enforces, so analysed calls are well-founded. A function that calls itself, directly or through others, or whose body is unavailable, is reached only through `opaque`.

## 3. Sources and canonical identity

```
src ::= named(s)                     declared in source text; identified by name s
      | fresh(n, π)                  residue of bilinear node n under call path π
π   ::= [] | s :: π                  call path, innermost call site first
```

- `named(s)` is the same source wherever it appears: in any module, and in either import order. Its bounds are part of its declaration. Two declarations of `s` with different bounds, or with `lo > hi`, are refused (E314).
- `fresh(n, π)` is new **per instance**. The residue that node `n` creates in the body of `f` differs between call sites `s₁` and `s₂`, because `π` differs. This is what makes "two calls that create fresh residues do not cancel" hold.
- Canonical ids (ADR-032 decision 3) are what step 2 must produce: the node id `n` from the parser and the call site `s`. They must not depend on module load order.

## 4. Types and typing rules

```
S ::= Known(S)   (S a finite set of sources)   |   Unknown
τ ::= num | Knowledge[A; K] | (τ, τ)
```

A Knowledge type carries two sets of the lattice ordered `Known(S) ⊑ Known(S ∪ T) ⊑ Unknown`:
- **A**, the aleatory sources, which GUM propagates;
- **K**, the Knightian sources, which are unknown but bounded.

The join `⊔` is set union on `Known`, and `Unknown` absorbs. `Known(∅)` means no source of that kind.

The judgement is `Γ; π ⊢ e : τ`. The call path `π` enters the judgement because fresh sources are instantiated per call path. Below, `K[a;k]` abbreviates `Knowledge[a; k]`.

| Rule | Premises | Conclusion |
|---|---|---|
| Lit | | `Γ; π ⊢ n : num` |
| Knight | `lo ≤ hi` | `Γ; π ⊢ knight(s,lo,hi) : K[∅; {named s}]` |
| Gauss | | `Γ; π ⊢ gauss(s,sd) : K[{named s}; ∅]` |
| Var | `Γ(x) = τ` | `Γ; π ⊢ x : τ` |
| Let | `Γ; π ⊢ e₁ : τ₁`, `Γ,x:τ₁; π ⊢ e₂ : τ` | `Γ; π ⊢ let x = e₁ in e₂ : τ` |
| Add/Sub | `τ₁, τ₂` | `num ⊕ num = num`; `num ⊕ K[a;k] = K[a;k]`; `K[a;k] ⊕ K[a';k'] = K[a ⊔ a'; k ⊔ k']` |
| Scale | `Γ; π ⊢ e : τ`, `τ` not a pair | `Γ; π ⊢ c * e : τ` |
| Mul | `τ₁, τ₂` | as Add, except `K[a;k] ⊗ₙ K[a';k'] = K[a ⊔ a' ⊔ r; k ⊔ k' ⊔ r]` with `r = {fresh(n,π)}`. The `⊔ r` on K is dropped when `k = k' = Known(∅)`, because then no Knightian term reaches the product. |
| Pair | `Γ; π ⊢ e₁ : τ₁`, `Γ; π ⊢ e₂ : τ₂` | `Γ; π ⊢ (e₁, e₂) : (τ₁, τ₂)` |
| Proj | `Γ; π ⊢ e : (τ₁, τ₂)` | `Γ; π ⊢ e.0 : τ₁`, `Γ; π ⊢ e.1 : τ₂` |
| Call | `Γ; π ⊢ e : τₐ`, `x:τₐ; s::π ⊢ body_f : τ` | `Γ; π ⊢ f@ₛ(e) : τ` |
| Opaque | `Γ; π ⊢ e : τₐ` | `Γ; π ⊢ opaque f(e) : K[Unknown; Unknown]` |
| Require | `Γ; π ⊢ e : num` or `K[a; ∅]` | `Γ; π ⊢ require(e) : num` |
| RequireExact | `Γ; π ⊢ e : num` or `K[∅; ∅]` | `Γ; π ⊢ requireExact(e) : num` |

`require` has no rule for a non-empty or `Unknown` K set. `requireExact` has none for a non-empty or `Unknown` A or K set. The checker refuses those, with E230, E313 or E314 depending on the surface construct.

**Summaries.** The Call rule types the callee's body under the extended path `s::π`. A summary is a memoised form of that rule: the parameter sets it forwards, the named sources it declares, and the fresh nodes that the call site instantiates with `s`. The Lean fragment uses the rule itself. Proving the summary-as-data form equal to it is NOT ESTABLISHED (§5).

**Projection.** A projection's type is the field's own type, not the join over the aggregate. This is ADR-032's per-field projection, and it is proved for pairs in the Lean fragment (`proj_carries_field_set`, §5).

## 5. The Lean fragment (`formal/SCCore.lean`)

**Covered.**
- Literals, variables and `let`.
- Two named sources: `knight` (Knightian, with declared bounds) and `gauss` (aleatory).
- Arithmetic: `+`, `-`, constant scaling, and bilinear `*ₙ` with a fresh residue.
- Pairs with per-field projection (`fst`, `snd`).
- Calls to earlier-defined functions, instantiated per call path, and `opaque` calls.
- `require` (no Knightian source) and `requireExact` (no source at all).

Runtime values are affine forms: a centre plus terms `(src, lo, hi, coefficient, kind)`, where kind is aleatory or Knightian, plus pairs of values. The types carry two sets per Knowledge value, A and K. A runtime term must lie in the static set of its own kind.

The residue of a bilinear node always enters A. It enters K as well, unless both operands' K sets are `Known(∅)`, in which case no Knightian term can reach the product. Evaluation is big-step with fuel. A runtime `conflict` check refuses a term list that holds two Knightian terms of one named source with different bounds.

**Not covered** (each stated, none hidden):
- **Summaries as data.** The fragment types a call by instantiating the callee's body under the call path. A summary is a memoised form of that rule, but the bilinear rule branches on whether the argument's K set is empty. A summary therefore has to carry that condition symbolically. Proving it equal to body instantiation is open: NOT ESTABLISHED.
- Core Lint.
- Semantic refinement (theorem d).
- f64 rounding: coefficients are integers, so the algebra is exact.

**Theorems.** The prediction, written before each check, was no `sorry` and axioms ⊆ {propext, Quot.sound, Classical.choice}. Both checks were OBSERVED on 2026-11-26 with Lean 4.33.0, first for the K-only fragment (#3745) and then for this one. The prediction held both times. No `Classical.choice` appears.

| Theorem | Statement | `#print axioms` |
|---|---|---|
| `sc_sound` — (a) | If `typeOf … e = some τ` and the environment's values have their types, then `eval` either runs out of fuel or returns `v` with `HasTy v τ`: every runtime term lies in the static set of its kind. So evaluation is never `stuck`, never `imprecise` and never `conflict`. | propext, Quot.sound |
| `require_never_imprecise` — (b, E230) | A `require` the checker types never fails at run time as imprecise. | propext, Quot.sound |
| `requireExact_never_imprecise` — (b) | Neither does a typed `requireExact`. | propext, Quot.sound |
| `consistent_never_conflict` — (b, E314) | If the program's named-bound table is consistent, no runtime term list holds two Knightian terms of one named source with different bounds. | propext, Quot.sound |
| `opaque_refused_at_require` — (b, Unknown) | `require(opaque f(e))` is never typed. | propext |
| `proj_carries_field_set` | The type of `fst e` (resp. `snd e`) is exactly the first (second) field's type, not a join over the aggregate. | propext |
| `consistentB_sound` | The decidable E314 check implies the consistency hypothesis. | propext, Quot.sound |
| `consistent_perm` | Consistency is invariant under permutation of the declarations, so import order is irrelevant to E314. | propext |

`sc_sound` rests on an invariant with two parts:
- A value's runtime sources of each kind are contained in that kind's static set (`HasTy`).
- Every runtime named Knightian term carries bounds that appear in the program's declaration table (`NamedOK`).

This static/dynamic correspondence is what makes the refusals sound.

Cancellation is not modelled statically: `k - k` has type `Known{named k}`. The checker is therefore **sound, not complete**. It may refuse a value whose sources cancel at run time, and it never accepts a value that carries one. An aleatory value passes `require`, because a GUM uncertainty is a precise bound. It does not pass `requireExact`.

Run it: `cd formal && lake build SCCore && lake env lean AxiomReport.lean`, or the CI step `scripts/ci/lean_axiom_report.sh`.

## 6. Step-1 obligations (ADR-032 §Plan, item 1)

| Obligation | Status |
|---|---|
| Repeated identity calls preserve one shared source | ESTABLISHED in the fragment: a kernel-checked `example` (`id(k) - id(k)` carries two terms of `named 3`, coefficients `+1` and `−1`). In the compiler: #3748's `ident_repeated_id_e230` and #3741's `epi006_knight_same`. |
| Two calls that create fresh residues do not cancel | ESTABLISHED in the fragment (`fresh 7 [1] ≠ fresh 7 [2]`; residues are instantiated per call path). In the compiler: OBSERVED on #3748, `ident_fresh_per_call_no_cancel`. |
| Imported named source: same bounds agree, conflicting bounds refuse (E314) | ESTABLISHED: the `consistentB` examples plus `consistent_never_conflict`. Surface witness: `epi006_knight_import_conflict` (#3741). |
| `Unknown` (recursion) does not satisfy a precise requirement | ESTABLISHED: `opaque_refused_at_require`. |
| Reordering imports leaves identities and results unchanged | ESTABLISHED for E314 (`consistent_perm`), and by construction for `named(s)`. For `fresh(n,π)` it depends on canonical node ids that do not depend on load order (step 2). In the compiler: OBSERVED on #3748, `ident_import_order_ab/ba`. |
| A field projection carries the field's set | ESTABLISHED in the fragment: `proj_carries_field_set`, plus the witness `require (snd ⟨knight 3, 7⟩)` accepted while `require (fst …)` is refused. In the compiler: OBSERVED on #3748, `ident_struct_field_projection_precise`. |
| A and K are kept apart; an aleatory value meets `require` but not `requireExact` | ESTABLISHED in the fragment: the `typeOf` examples for `k3 + g5`, `require g5`, `requireExact g5`, and the square of an aleatory value (whose residue stays in A). |
| A type-correct, value-changing rewrite passes Core Lint and fails semantic refinement | Needs Core Lint and theorem (d). NOT ESTABLISHED; depends on step 2. |

Decision 3 of ADR-032 asks for (a) and (b) to be closed before step 3. Both are closed **for this fragment**, including fields, which step 3 lowers first. Summaries as data are the remaining extension before step 3's call construct.
