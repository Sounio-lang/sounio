<!-- docs:meta
topic_id: repo.docs.research.paper-a-sections4-5-draft-2026-08-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.paper-a-sections4-5-draft-2026-08-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Paper A — §4 *Anti-garbling as the soundness criterion* + §5 *The type system* (full draft, 2026-08-25)

> Draft prose for the technical core. §4 is grounded in
> `docs/research/lean/SounioAntiGarblingModel.lean` (kernel-checked, axiom-free,
> Lean 4.33.1); §5 in `docs/research/sounio/ns_contract.sio` and `ns_dataflow.sio`
> (souc-green prototypes) and the authorized compiler wire (E230, `noise_sets.sio`).
> Notation continues §2: an epistemic value has mean `m` and variance `v`; `Cov(X,Y)`
> is the covariance of the underlying random variables.

---

## 4. Anti-garbling as the soundness criterion

§2 showed a family of operations that understate variance on correlated operands. To
turn "understate" into a checkable property we need a criterion that says, of an
uncertainty operation, whether it is *allowed to produce the answer it produces*. That
criterion is not application-specific; it is the information-monotonicity law already
standard in quantitative information flow, read in the variance channel.

### 4.1 Sound operations lose information; anti-garbling creates it

An uncertain quantity is, operationally, an *experiment*: a channel from the unknown
true value to an observation. Blackwell's informativeness order compares two such
experiments — `A` is *more informative* than `B` (written `B ⪯ A`) iff `B` can be
obtained from `A` by post-processing through a stochastic channel, a **garbling**
(Blackwell 1953). Garbling can only discard information; the data-processing inequality
is exactly this monotonicity. The quantitative-information-flow community adopts the
same order as its refinement order and its soundness backbone: a program transformation
is admissible iff it *refines* (is a garbling of) the original — McIver, Morgan, Smith
et al. (POST 2014); Alvim et al., *The Science of Quantitative Information Flow* (2020).

We instantiate that discipline for numeric uncertainty propagation:

> **Soundness criterion (anti-garbling).** An uncertainty operation is *sound* iff its
> output experiment is a garbling of the true joint experiment on its operands — i.e.
> it never reports *more* information (less uncertainty) than the operands contain. An
> operation that reports a variance smaller than the truth is an **anti-garbling**: it
> manufactures information, and no correct program may contain one.

This reframes §2 precisely. `ep_sub(&x,&x)` overstating variance is a *garbling* — it
throws information away, which the criterion permits (it is merely conservative).
`ep_add(&x,&x)` and `ep_mul(&x,&x)` understating variance are *anti-garblings* — they
are forbidden. The add/sub asymmetry of §2.2 is the anti-garbling criterion read
directly off the sign of the dropped covariance term.

*Scope of this paper.* The full Blackwell order is defined on distributions, and the
general "reassociation-is-a-garbling" theorem in that setting is developed separately
(companion work on non-associative composition). Paper A needs only the **second-moment
shadow** of the criterion — the variance channel — because the defect class of §2 lives
entirely there: every operation in question propagates *variance*, and the unsoundness
is a variance understatement. We therefore state and enforce the criterion on variance,
and prove exactly that fragment.

### 4.2 The scalar operators as an independence claim

Each `ep_*` operation implements the GUM propagation law under an unstated hypothesis.
For addition, the general law is `Var(X+Y) = Var(X) + Var(Y) + 2·Cov(X,Y)`; `ep_add`
computes `Var(X) + Var(Y)`, which equals the truth **iff `Cov(X,Y) = 0`**. The scalar
representation `(m, v)` cannot express `Cov`, so the operator cannot condition on it: it
asserts independence unconditionally. The defect of §2 is not a wrong formula — each
formula is the correct *independent-case* law — but an **unguarded precondition**: the
operator is sound on the sub-domain `Cov = 0` and is applied on all of it.

Making the criterion checkable therefore reduces to making `Cov = 0` a property the type
can carry and the compiler can verify. §4.3 pins down exactly what must be verified; §5
carries it in the type.

### 4.3 The core lemma (kernel-checked)

We model an uncertain value as an **affine form** over independent unit-variance noise
symbols `ε₁, ε₂, …` — `x = Σᵢ cᵢ εᵢ` — the representation of affine arithmetic
(Comba–Stolfi 1993). The coefficient vector *is* the value's source identity: two values
share a source iff they share a nonzero coefficient on the same `εᵢ`. In this model the
true variance is `‖c‖²`, covariance is the inner product `⟨a,b⟩ = Σᵢ aᵢbᵢ`, and addition
is componentwise (correlation handled by construction). The scalar `ep_add` variance is
`‖a‖² + ‖b‖²`, which forgets shared coefficients.

**Lemma 1 (understatement = twice covariance).** For affine forms `a, b`,

    trueAddVar(a,b) − naiveAddVar(a,b) = 2·⟨a,b⟩,

hence `naiveAddVar(a,b) = trueAddVar(a,b) ⟺ ⟨a,b⟩ = 0`.

*Proof.* `trueAddVar(a,b) = ‖a+b‖² = ‖a‖² + ‖b‖² + 2⟨a,b⟩` and `naiveAddVar(a,b) =
‖a‖² + ‖b‖²`; subtract. The general identity is a polynomial identity (`ring`); the
Mathlib-free artifact discharges representative integer witnesses by `decide`. ∎

This is checked, not asserted. `SounioAntiGarblingModel.lean` (Lean 4.33.1, `#print
axioms` = none, `sorry = 0`) proves, on witnesses `x = (1,0)`, `y = (0,1)`, `z = (2,1)`:

- `anti_garbling_x_plus_x` — `naiveAddVar x x < trueAddVar x x` (2 < 4): the §2 `x+x`
  understatement, kernel-checked.
- `anti_garbling_gap_x`, `anti_garbling_gap_z` — the gap is exactly `2·⟨·,·⟩` at two
  coefficient scales.
- `sound_under_disjoint` — `naiveAddVar x y = trueAddVar x y`: the scalar add is *exact*
  when the sources do not overlap.
- `gap_zero_iff_disjoint_witness` — the gap vanishes iff `⟨x,y⟩ = 0` on these witnesses.

**Corollary (the checkable condition).** `ep_add` (and, mutatis mutandis, `ep_mul`) is
sound on operands `a, b` iff `⟨a,b⟩ = 0` — zero covariance. The type system's job is to
certify this condition, or reject the operation.

### 4.4 What the type can actually certify — conservative, not exact

Lemma 1 makes zero *covariance* the exact soundness condition, and covariance is a
numeric quantity the type does not know. What the type *can* decide is a **structural**
proxy: whether the operands' noise-symbol **supports are disjoint** (share no `εᵢ`).
The two are not the same, and honesty about the gap is load-bearing:

> **Disjoint support ⟹ zero covariance, but not conversely.** If `a` and `b` share no
> coefficient then `⟨a,b⟩ = Σᵢ aᵢbᵢ = 0`. The converse fails: `a = (1,1)`, `b = (1,−1)`
> have overlapping support yet `⟨a,b⟩ = 0`. (This corrects an earlier "sound ⟺ disjoint"
> phrasing; the necessary-and-sufficient condition is zero covariance, and disjoint
> support is sufficient only — codex review, 2026-08-22.)

Consequently the type check is **conservatively sound**: it admits an
independence-assuming operator only when it can prove disjoint support, which *implies*
`⟨a,b⟩ = 0`, so every admitted operation is genuinely sound (Lemma 1). It may *reject*
some sound operations — the coincidentally-orthogonal-but-overlapping case — which §5.5
handles with an explicit escape valve rather than by unsound admission. This is the
standard soundness/completeness trade of a static discipline, and we take the sound side:
a rejected sound program is a nuisance; an admitted anti-garbling is the bug we exist to
prevent.

---

## 5. The type system

The criterion of §4 asks for one fact at every independence-assuming operator: are the
operand source-supports provably disjoint? We answer it by carrying the source-support in
the type and checking disjointness at the operator. The machinery is a set-valued
dataflow — the source-identity idea of affine arithmetic (Comba–Stolfi; Goubault–Putot's
Fluctuat), but lifted **into the type** and used to **reject** rather than to enclose.

### 5.1 Types carry a noise-symbol source-set

An epistemic type is `Knowledge⟨T, N⟩`, where `N` is a **noise-set** drawn from the
lattice

    L = (𝒫(S) ∪ {⊤}, ⊑),   ∅ ⊑ every finite set ⊑ ⊤,   join = ∪ (⊤ absorbing),

`S` the set of measurement sources. The implementation (`ns_contract.sio`) represents `N`
as a 3-state handle: `−1 = ⊤` (unknown), `0 = ∅` (deterministic / no measured source),
`>0` an interned nonempty set. The one rule that makes the lattice sound for our purpose:

> **`⊤` is never disjoint from anything.** An unknown source-set is treated as
> potentially sharing every source — the conservative top, not a convenient "assume
> independent" default (`ns_contract.sio: ns_disjoint`, `ns_union`).

### 5.2 Formation

    ─────────────────────────────────────  (Measure, s fresh)
    Γ ⊢ measure(v, σ) : Knowledge⟨T, {s}⟩

    ────────────────────────────────  (Exact)
    Γ ⊢ certain(v) : Knowledge⟨T, ∅⟩

A measurement seeds a fresh singleton source; an exact constant carries the empty set.
(`ns_measure` seeds `bit(id)`; the sabotage knob of §8.2 replaces this with `∅`.)

### 5.3 Transfer

Copy inherits; independence-assuming binary operators union the operand sets, with `⊤`
absorbing:

    Γ ⊢ a : Knowledge⟨T, N⟩
    ─────────────────────────────  (Copy / Ident)
    Γ ⊢ copy(a) : Knowledge⟨T, N⟩

Union is the join of `L`: `ns_union(a,b) = if a=⊤ ∨ b=⊤ then ⊤ else a ∪ b`. This is a
monotone transfer over the value graph; iterated to a least fixpoint it is the standard
Kildall dataflow — realised in `ns_dataflow.sio` as `nsg_propagate`, whose lattice is
`set-of-sources` where the in-tree escape analyzer's is `boolean`. Same graph, same
fixpoint engine, different lattice: the source-identity analysis is the escape analysis
with `∪` for reachability.

### 5.4 The checked precondition (the heart)

An independence-assuming operator is well-typed **only** if the operand supports are
provably disjoint:

    Γ ⊢ a : Knowledge⟨T, Nₐ⟩    Γ ⊢ b : Knowledge⟨T, N_b⟩    disjoint(Nₐ, N_b)
    ──────────────────────────────────────────────────────────────────────────  (Add-Indep)
    Γ ⊢ ep_add(a, b) : Knowledge⟨T, Nₐ ∪ N_b⟩

where `disjoint(Nₐ, N_b) := Nₐ ≠ ⊤ ∧ N_b ≠ ⊤ ∧ Nₐ ∩ N_b = ∅`. `ep_mul` has the identical
side condition (§4's corollary). When the premise `disjoint(Nₐ, N_b)` fails — overlapping
supports, or either operand `⊤` — the operator is **rejected**:

> **E230 — anti-garbling: independence-assuming operation over non-disjoint / unknown
> noise-symbol sets.**

`ns_contract.sio: add_flagged` is exactly this predicate (`if ns_disjoint(a,b) return
false else return true`), validated by the acceptance controls: `x+x` flagged (shared
source), `x+y` accepted (disjoint), `x + unknown` flagged (`⊤` conservative), and
`ident(x) + x` still flagged (identity survives a copy). E230 is deliberately a distinct
diagnostic from `E222` (R-ORIGIN provenance) so the two rules stay causally separable
(§8.2, §5.7).

### 5.5 The escape valve — a proved-disjoint certificate or an explicit covariance

Because the check is conservative (§4.4), some sound programs are rejected. Two admissions
recover them without weakening soundness:

1. **A proved-disjoint certificate.** Where the programmer (or an oracle pass) can
   establish `⟨a,b⟩ = 0` for overlapping-but-orthogonal operands, a certificate discharges
   the premise of (Add-Indep) directly — the type admits the operation on the strength of
   the proof, not the support test.
2. **An explicit correlation-aware operator.** For genuinely correlated operands, the
   sound path is not to suppress the check but to *take the covariance as an argument*:
   `gum_s1_add_correlated(a, b, ρ)` (in-tree, `gum_supplement1.sio`, currently orphaned)
   propagates `Var(a) + Var(b) + 2ρ√(Var(a)Var(b))`. Its typing rule unions the supports
   with **no** disjointness premise, because it does not assume independence:

    Γ ⊢ a : Knowledge⟨T, Nₐ⟩    Γ ⊢ b : Knowledge⟨T, N_b⟩
    ────────────────────────────────────────────────────────  (Add-Corr)
    Γ ⊢ add_correlated(a, b, ρ) : Knowledge⟨T, Nₐ ∪ N_b⟩

The type discipline thus does not forbid correlated arithmetic; it forbids doing it with
the operator that *assumes* independence. The programmer's choice of operator becomes a
typed claim about correlation, checked against the tracked supports.

### 5.6 Interprocedural summaries — the load-bearing dependency

The transfer of §5.3 is intraprocedural. Across a call, the source-set of a returned value
depends on the callee's body and on which caller-supports flow into which parameters. Two
options, and only one is sound: dropping to `⊤` at every call boundary (sound but so
conservative it rejects almost everything), or **parametric call-summaries** that
substitute caller supports into a callee's abstract source-set. We take the latter; it is
the same summary machinery the compiler's memory-reclamation analysis independently
requires (the escape analyzer is intraprocedural today for the same reason), so the
engineering cost is shared, not doubled. This is the principal implementation dependency
and we flag it as such rather than understate it: without interprocedural summaries the
conservative default must be **assume-sharing** — the exact opposite of the library's
assume-independence — which is sound but noisy, and the summaries are what make the
discipline usable.

### 5.7 NS and provenance are siblings, not the same rule

The source-set discipline (NS) sits beside the language's provenance discipline
(R-ORIGIN, which asks *measured vs derived*), and the two must not be conflated:

| Aspect | R-ORIGIN (provenance) | NS (noise-symbol) |
|---|---|---|
| Question | *where* did the value come from? | *which sources'* uncertainty does it carry? |
| Lattice | scalar origin-kind | source-**set** (powerset, `∪`) |
| Violation | laundering: computed value claims *measured* | anti-garbling: independence assumed between correlated operands |
| Diagnostic | E222 | **E230** |
| Soundness anchor | no-laundering witnesses | `SounioAntiGarblingModel` (Lemma 1) |

They share the dataflow substrate (§5.3) and the `TypeEntry` tag mechanism (a trailing
`noise_set_id` field beside `provenance_id`), but they are distinct abstract domains with
distinct diagnostics, kept causally separable precisely so the evaluation of §8.2 can show
that disabling NS removes E230 refusals while leaving E222 refusals intact.
