# Paper A — §6 *Metatheory* (full draft, 2026-08-25)

> Draft prose for the metatheory. Mechanized results are grounded in
> `formal/lean4/EpistemicEffectsV2.lean` (629 lines; progress + preservation, Lean
> 4.33.1) and `docs/research/lean/SounioAntiGarblingModel.lean` (Lemma 1, kernel-checked,
> axiom-free). Claims that the NS extension's re-run of preservation is not yet mechanized
> are labelled **[pending wire]** — we state the metatheorem at paper level and mark
> exactly which lemmas carry a machine proof today.

---

## 6. Metatheory

We establish that a well-typed program contains no first-order anti-garbling: at every
independence-assuming operator, the operands' true covariance is zero, so the propagated
variance is exact rather than understated (§4, Lemma 1). The argument has three parts —
a mechanized type-safety substrate (§6.2), a sound source-set abstraction (§6.3), and a
local soundness criterion already proven (§4.3) — composed in §6.4. Two boundaries are
carried as explicit hypotheses of the theorem rather than hidden (§6.5).

### 6.1 What type safety alone does *not* give

It is worth being precise about the gap the discipline must close, because a conventional
type-safety result does **not** close it. Our core calculus already enjoys full type safety
(§6.2), and that is not enough: a program can be perfectly well-typed, never get stuck,
preserve its types under reduction, and *still* report an understated variance. Type safety
guarantees the metadata stays *valid* (a non-negative variance, a bounded confidence); it
says nothing about whether that variance is the *correct* one. Anti-garbling is a soundness
property one level above type safety, and it needs the source-set the base calculus does not
carry. §6.2 makes this concrete by showing that the mechanized dynamic semantics *is* the
defective one.

### 6.2 The mechanized substrate: type safety, and why it is not soundness

`EpistemicEffectsV2.lean` formalizes a core epistemic-effects calculus: a Knowledge type
`tknow T`, a `measure`/`kraw` form carrying scalar GUM metadata `KMeta = {gumVar, conf}`,
an effect row with sub-effecting (`⊆ₑ`), and the arithmetic operators `kadd`, `kmul` typed
at `tknow treal` (`HasTy`, `:59–89`). The full type-safety pair is mechanized, Lean 4.33.1:

- **Progress** — `progress'` (`:223`), `effect_progress` (`:301`): a closed well-typed term
  is a value or steps.
- **Preservation** — `preservation'` (`:559`), `preservation` (`:626`): typing and effect
  rows are preserved under `Step`, with the usual supporting infrastructure (weakening
  `:420`, closed substitution `:504`, canonical forms `:199–223`).
- **Metadata validity** — `gAddMeta_valid` (`:324`), `gMulMeta_valid` (`:342`): the metadata
  combinators preserve `kvalid m := 0 ≤ m.gumVar ∧ 0 ≤ m.conf ∧ m.conf ≤ 1000`.

Here is the sharp point. The operational combinator the calculus reduces `kadd` through is

```lean
def gAddMeta (ma mb : KMeta) : KMeta :=
  { gumVar := ma.gumVar + mb.gumVar, conf := if ma.conf ≤ mb.conf then ma.conf else mb.conf }
                                                                       -- EpistemicEffectsV2.lean:92
def gMulMeta (x : Int) (ma : KMeta) (y : Int) (mb : KMeta) : KMeta :=
  { gumVar := y * y * ma.gumVar + x * x * mb.gumVar, ... }              -- :94
```

`gAddMeta` is `ep_add` and `gMulMeta` is `ep_mul` — **the very operators of §2**,
`var_a + var_b` and `y²·var_a + x²·var_b`, with no covariance term. So the mechanized
semantics faithfully implements the defect, and `gAddMeta_valid` proves that the defective
add *preserves validity*: `0 ≤ gumVar` is maintained even as the variance is understated.
This is the formal statement of §6.1 — **validity is preserved; soundness is not** — and it
is exactly why a source-set discipline, not a stronger type-safety proof, is required. The
substrate is sound *as a type system* and silent *as an uncertainty accountant*.

### 6.3 The source-set analysis is a sound abstract interpretation

The NS extension of §5 adds a source-set `N` to the Knowledge type and a disjointness
premise to `t_kadd`/`t_kmul`. Its metatheoretic contribution is one abstraction-soundness
fact:

> **Lemma 2 (support over-approximation).** For every value, the tracked noise-set `N`
> over-approximates the value's true noise-symbol support: every source that actually
> contributes to the value's uncertainty is a member of `N` (with `⊤` the trivially-safe
> over-approximation).

*Argument.* The transfer functions of §5.3 are the abstract counterparts of the concrete
dependency: `measure` introduces a fresh symbol (the true support of a new measurement);
`copy` preserves support; `kadd`/`kmul` produce a value whose true support is contained in
the union of the operands' supports, which the abstract transfer `∪` computes exactly, and
`⊤` absorbs the unknown case. Each transfer is monotone on the lattice `L = (𝒫(S) ∪ {⊤},
⊑)` (§5.1), so the analysis has a least fixpoint (Kildall), and monotonicity plus the
local containment at each node give the global over-approximation by the standard
abstract-interpretation soundness schema. The engine is realized in `ns_dataflow.sio`
(`nsg_propagate`, the monotone fixpoint) with the escape analyzer's proof obligations,
lattice boolean→set.

Over-approximation is what makes the conservative choices of §5 *sound*, not merely
cautious: because `N` can only *over*-state the true support, `⊤`-is-never-disjoint and the
assume-sharing interprocedural default (§5.6) can never mistake a correlated pair for a
disjoint one. They can only err toward rejecting a sound program — the completeness side,
not the soundness side.

### 6.4 The soundness theorem

> **Theorem (no first-order anti-garbling).** Let `e` be a program well-typed under the
> NS-extended system (§5). Then at every independence-assuming operator (`kadd`/`kmul`)
> reached during evaluation of `e`, the operands have zero covariance, and the propagated
> first-order variance equals the true first-order variance. Equivalently: no reachable
> `kadd`/`kmul` in a well-typed `e` is an anti-garbling.

*Proof (composition).* Take any independence-assuming operator in `e`. It type-checks, so
by (Add-Indep)/(Mul-Indep) its operand source-sets are provably disjoint: `Nₐ ∩ N_b = ∅`,
both `≠ ⊤`. By Lemma 2 the true supports are contained in `Nₐ, N_b`, hence the true
supports are disjoint, hence the true covariance `⟨a,b⟩ = 0` (disjoint support ⟹ zero
covariance, §4.4). By Lemma 1 (`SounioAntiGarblingModel`, kernel-checked) zero covariance
makes the scalar `gAddMeta`/`gMulMeta` variance *exact* — no understatement. Preservation
(`EpistemicEffectsV2.preservation`) carries the typing, hence the disjointness premise,
along each reduction step, so the guarantee holds not just at the source term but at every
operator reached during evaluation. ∎

**Mechanization status, stated honestly:**

| Ingredient | Status |
|---|---|
| Base calculus progress + preservation | ✅ mechanized — `EpistemicEffectsV2.lean` (Lean 4.33.1) |
| `gAddMeta`/`gMulMeta` = the §2 operators; validity preserved | ✅ mechanized — `gAddMeta_valid`, `gMulMeta_valid` |
| Local criterion: disjoint support ⟹ zero cov ⟹ exact (Lemma 1) | ✅ kernel-checked, axiom-free — `SounioAntiGarblingModel.lean` |
| Analysis soundness: `N` over-approximates true support (Lemma 2) | ◻ argued here; abstraction-soundness schema, not yet mechanized |
| NS-extended preservation (disjointness premise preserved under Step) | ◻ **[pending wire]** — extends the mechanized `preservation'` to the `N`-annotated `tknow` |

So the two *numeric* facts the theorem turns on — the defect operators and the local
soundness criterion — carry machine proofs today; what remains to mechanize is the
extension of the existing preservation proof to the source-set-annotated Knowledge type,
which is the same artifact as the N-phase compiler wire (§7.3). We claim the theorem at
paper level and the two load-bearing lemmas at kernel level, and we do not claim the
NS-extended preservation is mechanized until it is.

### 6.5 Two boundaries carried as hypotheses

The theorem is deliberately scoped. Both limits are stated as part of the guarantee, not
discovered by a referee:

- **Conservative, not complete.** The guarantee is *soundness* (no admitted operator is an
  anti-garbling), not *completeness* (not every sound operator is admitted). The rule keys
  on disjoint support, which is sufficient but not necessary for zero covariance (§4.4), so
  it rejects the overlapping-but-orthogonal case. That case is recovered by the escape valve
  (§5.5), never by unsound admission. Formally: the theorem quantifies over *admitted*
  operators; it makes no claim that the admitted set is maximal.

- **First-order only.** The soundness criterion (Lemma 1) is exact for the *linear* fragment
  (`kadd`, and `kmul` to first order in the delta method). The nonlinear operators
  (`ep_mul`, `ep_div`, `ep_square`, `ep_sqrt`) are delta-method approximations that drop
  second-order terms, so even under disjoint support a residual second-order discrepancy
  remains. The theorem therefore guarantees the absence of the *first-order covariance*
  anti-garbling — the entire defect class of §2 — and explicitly not the truncation error of
  the delta method, which is a separate, symmetric (non-directional) approximation error and
  not an anti-garbling. Extending the guarantee to second order is future work (it needs the
  Hessian/second-moment terms the current metadata does not carry).
