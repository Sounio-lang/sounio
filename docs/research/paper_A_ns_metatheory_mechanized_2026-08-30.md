<!-- docs:meta
topic_id: repo.docs.research.paper-a-ns-metatheory-mechanized-2026-08-30
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.paper-a-ns-metatheory-mechanized-2026-08-30
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Paper A — the NS metatheory is mechanized (2026-08-30)

**Closes** the two `◻` rows of §6.4's mechanization table (Lemma 2 "argued, not
mechanized"; NS-extended preservation "[pending wire]") **and** the theorem itself, which
the closed draft claimed "at paper level". Artifact: `formal/lean4/EpistemicEffectsNS.lean`
(~1400 lines, Lean 4.33.1, Mathlib-free, sorry-free). Gate:
`bash scripts/ci/ns_metatheory_lean_gate.sh` → `NS_METATHEORY_LEAN_GATE_PASS` (C1–C5).

Nothing here touches the compiler; the wire (`4ac63da51f`) and this proof are independent
artifacts about the same rule. The paper files were updated in place (§6.3, §6.4 table and
closing paragraph, contribution 3, abstract, top note; `paper_A_MERGED_2026-08-25.md`,
`paper_A_section6_draft_2026-08-25.md`, `paper_A.html`, `paper_A_README.md`). Those paper files
live on the research lane `lane/fable-1/p0f-ffi-takeover`, not on `main`; this note is the
`main`-side pointer.

---

## 1. The one design decision that made it provable

The closed draft's Theorem 6.4 composes three things — the base calculus's type safety,
Lemma 1 (covariance-exactness) and Lemma 2 (support over-approximation) — with a prose
"by preservation the premise is carried along". To *mechanize* that composition the
calculus has to say what "true variance" and "true support" **are**. So:

> A runtime Knowledge value is `kraw v m a` — payload `v`, the scalar metadata it
> **reports** (`m : KMeta`, exactly V2's `{gumVar, conf}`), and its **true first-order
> affine form** `a : Aff := List (Nat × Int)` — a formal sum of monomials `c·ε_s` over
> measurement sources `s`.

Everything else follows:

| Question | Answer in the calculus |
|---|---|
| true variance | `trueVar a := ⟨a,a⟩`, `inner a b := Σ_{(s,c)∈a} c · coeff b s` |
| true support | the sources occurring in `a` |
| what `measure` does | seeds `[(s, c)]`, reports `c²` (`meas_red`) |
| what `certain` does | seeds `[]`, reports `0` |
| what `kadd` does to the truth | `a ++ b` — exact affine addition (duplicates allowed; `coeff` sums them, so no canonical form is needed) |
| what `kadd` **reports** | `gAddMeta ma mb` = `ep_add` — `var_a + var_b`, **no covariance term** |
| what `kmul` does to the truth | `scale y a ++ scale x b` — the delta-method linearisation at `(x, y)` |
| what `kmul` reports | `gMulMeta` = `ep_mul` — `y²var_a + x²var_b` |

The operational semantics is **deliberately the defective one** (this is §6.2's sharp
point, kept). The discipline lives entirely in the types:

```
t_kadd : Γ ⊢ a : tknow treal Na ! E₁ → Γ ⊢ b : tknow treal Nb ! E₂ →
         nsDisjoint Na Nb = true →                              -- E230 when it fails
         Γ ⊢ kadd a b : tknow treal (nsUnion Na Nb) ! E₁ ∪ E₂
t_kraw : Γ ⊢ v : T ! ∅ → IsValue v → kvalid m → Covers N a →   -- Lemma 2 as an invariant
         Γ ⊢ kraw v m a : tknow T N ! ∅
```

`NS := Option (List Nat)`, `none = ⊤`; `nsDisjoint` is a **decidable Bool** (`⊤` is never
disjoint — `nsDisjoint_top_left/right` are `rfl`), which is what makes the E230 check a
*checked precondition* and the witnesses `decide`-able.

## 2. Soundness is a separate invariant, not a stronger typing judgment

§6.1's thesis is that type safety ≠ soundness. The mechanization keeps that separation
literal:

- `Exact e` — every `kraw` in `e` satisfies `m.gumVar = trueVar a` — is a predicate on
  terms, **not** part of `HasTy`.
- `preservation` (typing, with `N`) and `exact_preservation` (soundness) are two theorems.
  The second needs the first's premise: at `kadd_red` the goal is
  `ma.gumVar + mb.gumVar = trueVar (a ++ b)`; `trueVar_append` rewrites the right side to
  `trueVar a + trueVar b + 2⟨a,b⟩`; `inner_zero_of_ns (Covers Na a) (Covers Nb b)
  (nsDisjoint Na Nb = true)` kills the `2⟨a,b⟩`. That single line **is** Theorem 6.4's
  composition — Lemma 2 → disjoint true supports → Lemma 1 → exact.

One could instead have baked `m.gumVar = trueVar a` into `t_kraw` and called preservation
"soundness". That would have hidden exactly the distinction the paper is built on.

## 3. Theorem map

| Paper | Lean (`Sounio.EpistemicEffectsNS`) | Note |
|---|---|---|
| Lemma 1, general form | `trueVar_append : trueVar (a ++ b) = trueVar a + trueVar b + 2 * inner a b` | all affine forms; `SounioAntiGarblingModel.lean` had Int witnesses only and said the general form "needs `ring`/Mathlib" — it does not (`inner_comm` + `omega`) |
| Lemma 1, products | `trueVar_mul : trueVar (scale y a ++ scale x b) = y²·trueVar a + x²·trueVar b + 2xy·inner a b` | first-order, matches `gMulMeta` up to the covariance term |
| DISJ ⟹ zero cov | `inner_disjoint`, `coeff_absent` | |
| lattice `L` | `NS`, `nsUnion` (⊤ absorbing), `nsMem`, `nsDisjoint` (Bool), `nsDisjoint_sound`, `nsDisjoint_of_shared` | a shared member refutes disjointness for **every** annotation incl. ⊤ |
| Lemma 2 (abstraction) | `Covers`, `covers_single`, `covers_empty`, `covers_union`, `covers_scale`, `support_over_approx`, `covers_coeff` (nonzero-coefficient form) | transfer soundness per operator; extraction from any derivation |
| Crux #1 composed | `covers_disjoint`, `inner_zero_of_ns` | NS-disjoint + covered ⟹ `⟨a,b⟩ = 0` |
| NS type safety | `progress`, `preservation` (+ `weakening`, `substClosed`, `value_emptyE`, canonical forms) | port of V2's proofs with the `N` index and the `Covers` obligation at every `kraw` construction |
| Soundness | `Exact`, `exact_shift`, `exact_subst`, `exact_preservation` | |
| Theorem 6.4 | `AGFree`, `typed_agfree` (no reduction needed), `soundness_star` (along `⇒*`) | "no **reached** operator is an anti-garbling" |
| §8.2 controls | `x_plus_x_steps` (the defective semantics *does* step), `x_plus_x_understates` (2 ≠ 4), `x_plus_x_gap` (= 2⟨x,x⟩), `x_plus_x_untypable` (∀ Γ T E, ∀ annotation), `measure_plus_measure_untypable` (source level, via `invMeasure`), `let_x_plus_x_untypable` (the §8.2 shared-variable `let x = measure s in x + x`, via `invLet` + `invVar`), `x_plus_top_untypable` (`x + opaque(y)`: ⊤ clause in isolation, via `invOpaque`; `opaque_y_typable`), `x_plus_y_typable`, `x_plus_y_exact`, `x_times_y_exact`, `x_times_x_understates` | the sabotage witness in the kernel; `opaque e` = the paper's `opaque_knowledge()` fixture, typed at ⊤ |

Axiom footprint (`#print axioms`, captured by the gate):

```
trueVar_append, inner_zero_of_ns, progress, exact_preservation, typed_agfree,
x_plus_x_untypable            : [propext, Quot.sound]
preservation, soundness_star  : [propext, Classical.choice, Quot.sound]   (by_cases in substClosed)
x_plus_x_understates          : [propext]
```

No `sorryAx`. The module imports only the effect lattice (`Effect`, `EffectSet`, `⊆ₑ`)
from `EpistemicEffects.lean`; that file's own open obligation (`:378`, `sorry`) does not
leak — the footprint proves it.

## 4. What is still *not* mechanized (now the honest residual of §6.4)

1. **Calculus ↔ checker correspondence.** `HasTy` is not proven equivalent to the
   production E230 rule in `noise_sets.sio`/`check.sio`. The wire is source-verified and
   sabotage-gated (`ns_antigarbling_gate.sh`), which is evidence, not proof.
2. **Interprocedural summaries (§5.6).** The calculus has `lam`/`app` but every Knowledge
   value crossing a call is a concrete `kraw` with a concrete `N`; the *parametric*
   call-summary of §5.6 is not modelled. (The conservative default — `⊤` at boundaries —
   is sound in this calculus by `covers_top`.)
3. **Second order (§6.5).** `scale y a ++ scale x b` is the linearisation; the theorem is
   first-order by construction, as the paper states.
4. **The escape valve (§5.5).** `add_correlated(a, b, ρ)` is not in the calculus. Its
   exactness would be a *hypothesis* (the claimed `ρ` equals the true `⟨a,b⟩`), not a
   theorem — which is the paper's point about it being a typed claim.
5. **Honest labelling (the noise-symbol axiom).** Distinct `measure` labels are distinct
   physical sources. **Assumed, not proved** — with dishonest labels the calculus
   under-approximates covariance (Grok 4.6 round 3 forced this to be said plainly). The type system *tracks* sources; it does not *discover* them — two
   physically correlated measurements given different labels type as disjoint. This is the
   modelling axiom of every noise-symbol system (Comba–Stolfi, Fluctuat) and is stated, not
   discharged. (xai review 2026-08-30, item 3 — `paper_A_ns_metatheory_xai_review_2026-08-30.md`.)

6. **Algebraic, not distributional.** `trueVar a = ⟨a,a⟩` is the variance of `Σ c_s ε_s` under
   independent unit-variance symbols *by definition*; no sampling/distributional adequacy is
   modelled or claimed (Grok 4.6 rounds 2–3).

## 5. Reproduce

```bash
bash scripts/ci/ns_metatheory_lean_gate.sh
# C1 compile · C2 sorry-free · C3 axioms ⊆ {propext, Quot.sound, Classical.choice} ·
# C4 theorem names · C5 lakefile @[default_target]
```

On hosts where `lake` cannot spawn threads (this pod: `failed to create thread`) the gate
falls back to `lean --threads=1` with the dependency built into a temp dir; CI's
`lean-proofs` job uses `lake build` as for `EpistemicEffectsV2`.
