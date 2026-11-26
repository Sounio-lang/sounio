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
    | x                          variable
    | let x = e in e             sharing (one node, many uses)
    | e + e | e - e              linear
    | c * e                      scaling by a constant
    | e *ₙ e                     bilinear at node n (may create a residue)
    | f@ₛ(e)                     call of an analysed function at call site s
    | opaque f(e)                call summarised as Unknown (recursion, extern)
    | require(e)                 precise requirement (E230/E313/E314 boundary)
    | e.fᵢ                       field projection            (not in the Lean fragment)
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
K ::= Known(S)   (S a finite set of sources)   |   Unknown
τ ::= num | Knowledge[K]
```

The lattice is ordered `Known(S) ⊑ Known(S ∪ T) ⊑ Unknown`. Its join `⊔` is set union on `Known`, and `Unknown` absorbs. `Known(∅)` means precise. The aleatoric set A follows the same rules and is not repeated here.

The judgement is `Γ; π ⊢ e : τ`. The call path `π` enters the judgement because fresh sources are instantiated per call path.

| Rule | Premises | Conclusion |
|---|---|---|
| Lit | | `Γ; π ⊢ n : num` |
| Knight | `lo ≤ hi` | `Γ; π ⊢ knight(s,lo,hi) : Knowledge[Known{named s}]` |
| Var | `Γ(x) = τ` | `Γ; π ⊢ x : τ` |
| Let | `Γ; π ⊢ e₁ : τ₁`, `Γ,x:τ₁; π ⊢ e₂ : τ` | `Γ; π ⊢ let x = e₁ in e₂ : τ` |
| Add/Sub | `τ₁, τ₂` | `num ⊕ num = num`; `num ⊕ K[k] = K[k]`; `K[k] ⊕ K[k'] = K[k ⊔ k']` |
| Scale | `Γ; π ⊢ e : τ` | `Γ; π ⊢ c * e : τ` |
| Mul | `τ₁, τ₂` | as Add, except `K[k] ⊗ₙ K[k'] = K[k ⊔ k' ⊔ Known{fresh(n,π)}]` |
| Call | `Γ; π ⊢ e : τₐ`, `x:τₐ; s::π ⊢ body_f : τ` | `Γ; π ⊢ f@ₛ(e) : τ` |
| Opaque | `Γ; π ⊢ e : τₐ` | `Γ; π ⊢ opaque f(e) : Knowledge[Unknown]` |
| Require | `Γ; π ⊢ e : num` or `Knowledge[Known(∅)]` | `Γ; π ⊢ require(e) : num` |

`require` has no rule for `Knowledge[Known(S)]` with `S ≠ ∅`, or for `Knowledge[Unknown]`. The checker refuses those, with E230/E313/E314 depending on the surface construct.

**Summaries.** The Call rule types the callee's body under the extended path `s::π`. A summary is a memoised form of that rule: forwarded parameter sets, named sources, and fresh nodes that the call site instantiates with `s`. The Lean fragment uses the rule itself, so the summary-as-data form is a refinement still to be proved equal to it (NOT ESTABLISHED; §6).

**Projection.** `e.fᵢ : Knowledge[Kᵢ]`, where `Kᵢ` is the set of field `i`, not the join over the aggregate. This is ADR-032's per-field projection. It is specified here and is not yet in the Lean fragment.

## 5. The Lean fragment (`formal/SCCore.lean`)

**Covered.** Literals; `knight` with bounds; variables; `let`; `+`, `-`, constant scaling; bilinear `*ₙ` with a fresh residue; calls to earlier-defined functions, instantiated per call path; `opaque`; `require`. Runtime values are affine forms: a centre plus terms `(src, lo, hi, coefficient)`. Evaluation is big-step with fuel. A runtime `conflict` check refuses a term list that holds two terms of one named source with different bounds.

**Not covered** (each stated, none hidden):
- the A set;
- fields and projection;
- Core Lint;
- semantic refinement (theorem d);
- summaries as data;
- f64 rounding (coefficients are integers, so the algebra is exact).

**Theorems.** Predicted before checking: no `sorry`, and axioms ⊆ {propext, Quot.sound, Classical.choice}. OBSERVED 2026-11-26 with Lean 4.33.0. The prediction held.

| Theorem | Statement | `#print axioms` |
|---|---|---|
| `sc_sound` — (a) | If `typeOf … e = some τ` and the environment's values have their types, then `eval` either runs out of fuel or returns `v` with `HasTy v τ`. So it is never `stuck`, never `imprecise` and never `conflict`. | propext, Quot.sound |
| `require_never_imprecise` — (b, E230) | A `require` the checker types never fails at run time as imprecise. | propext, Quot.sound |
| `consistent_never_conflict` — (b, E314) | If the program's named-bound table is consistent, no runtime term list holds two terms of one named source with different bounds. | propext, Quot.sound |
| `opaque_refused_at_require` — (b, Unknown) | `require(opaque f(e))` is never typed. | propext |
| `consistentB_sound` | The decidable E314 check implies the consistency hypothesis. | propext, Quot.sound |
| `consistent_perm` | Consistency is invariant under permutation of the declarations, so import order is irrelevant to E314. | propext |

The invariant behind `sc_sound`: a value's runtime sources are contained in its static set `S` (`HasTy`), and every runtime named term carries bounds that appear in the program's declaration table (`NamedOK`). This is the static/dynamic correspondence that makes the refusals sound. Cancellation is not modelled statically (`k - k` has type `Known{named k}`), so the checker is **sound, not complete**: it may refuse a value whose sources cancel at run time, and it never accepts a value that carries one.

Run it: `cd formal && lake build SCCore && lake env lean AxiomReport.lean`, or the CI step `scripts/ci/lean_axiom_report.sh`.

## 6. Step-1 obligations (ADR-032 §Plan, item 1)

| Obligation | Status |
|---|---|
| Repeated identity calls preserve one shared source | ESTABLISHED in the fragment: kernel-checked `example` (`id(k) - id(k)` carries two terms of `named 3`, coefficients `+1` and `−1`). Surface witnesses: #3741's `epi006_knight_*`. |
| Two calls that create fresh residues do not cancel | ESTABLISHED in the fragment: `sq(k)` at sites 1 and 2 types as `{…, fresh(7,[1]), …, fresh(7,[2])}`, and `fresh 7 [1] ≠ fresh 7 [2]`. In the compiler: NOT ESTABLISHED. Today's Madaros has no per-call-site instantiation; that is the source-identity lane (`feat/source-identity-calls-fields`). |
| Imported named source: same bounds agree, conflicting bounds refuse (E314) | ESTABLISHED: `consistentB` examples plus `consistent_never_conflict`. Surface witness: `epi006_knight_import_conflict` (#3741). |
| `Unknown` (recursion) does not satisfy a precise requirement | ESTABLISHED: `opaque_refused_at_require`. |
| Reordering imports leaves identities and results unchanged | ESTABLISHED for E314 (`consistent_perm`), and by construction for `named(s)`, which is identified by name. For `fresh(n,π)`: depends on canonical node ids that do not depend on load order, which is step 2 (NOT ESTABLISHED). |
| A field projection carries the field's set | Specified (§4). Not in the fragment. NOT ESTABLISHED. |
| A type-correct, value-changing rewrite passes Core Lint and fails semantic refinement | Needs Core Lint and theorem (d). NOT ESTABLISHED; depends on step 2. |

Decision 3 of ADR-032 asks for (a) and (b) to be closed before step 3. Both are closed **for this fragment**. Fields and summaries-as-data extend the fragment before step 3's field and call constructs land.
