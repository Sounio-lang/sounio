<!-- docs:meta
topic_id: repo.docs.research.antigarbling-fusion-theorem-2026-09-01
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.antigarbling-fusion-theorem-2026-09-01
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Anti-Garbling Fusion Theorem: two certificates, one calculus — and a third axis

**Date:** 2026-09-01 · **Author:** fable-1 (agent=claude), lane/fable-1 ·
**Status:** THEOREM, machine-checked. `formal/lean4/EpistemicEffectsNSA.lean`
(1414 lines, Lean 4.33.1, `lean --threads=1`, exit 0, zero warnings, zero `sorry`,
zero `native_decide`, axioms ⊆ {propext, Quot.sound, Classical.choice}).
Gate: `scripts/ci/antigarbling_fusion_lean_gate.sh` (named in `ci.yml`, job Lean Proofs).
**Supersedes in part:** `ANTIGARBLING_COMPLETENESS_2026-08-23.md` and its proof companion —
the "exactly two" claim is **corrected** (§4); the two-certificate soundness is **lifted**
from a toy model to a typed calculus over every Cayley–Dickson algebra (§2–3).
**Extends:** `EpistemicEffectsNS.lean` (Paper A §6, scalar carrier) — carrier generalized to
`CD(n)`, n = 0 ℝ, 1 ℂ, 2 ℍ, 3 𝕆, 4 𝕊, …

---

## 0. The claim in one sentence

> In an uncertainty-typed language over a non-associative algebra, **the two identities a
> compiler is tempted to impose — independence of operands and associativity of the
> product — are guarded by two certificates of the *same shape* (a support
> over-approximation in the type + a decidable predicate), living on two *orthogonal
> supports* (noise symbols vs. basis elements) and attached to two *different relations*
> (the typing judgment vs. the program-equivalence the optimizer may use).** A certified
> re-association is garbling-free: same value, same true form, same reported variance.
> Drop either certificate and exactly one of those fails. And there is a **third**
> structural garbling — imposing norm-multiplicativity — invisible on ℝ,ℂ,ℍ,𝕆 (Hurwitz)
> and real on the sedenions: the GUM variance shortcut understates by a factor of 2 on a
> kernel-checked witness with disjoint sources and no re-association.

---

## 1. Why the toy model was not the theorem

The 08-23 note proved the two certificate lemmas and a "dimension count" in
`SounioAntiGarblingModel.lean` — a model with no typing judgment, no reduction, no program.
`EpistemicEffectsNS.lean` (08-30) IS a typed calculus with Theorem 6.4 — but its carrier is
`Int`, which is associative, so axis 1 is *vacuously* absent there. Neither file contains the
fusion. Worse, the framing was subtly wrong about *where* order garbling lives:

**Observation.** In a program with a fixed AST, the parenthesization *is* the true one.
Reduction can never order-garble. Order garbling is born only when someone **rewrites** the
program — an optimizer reassociates, a reduction `Π` picks an order, an equivalence
`(xy)z ≡ x(yz)` is used. Hence:

- **Axis 2 (support) is a property of the typing judgment** `Γ ⊢ e : Know⟨N, Q⟩` — checked at
  every combining site (`kadd`/`kmul`).
- **Axis 1 (order) is a property of the program-equivalence relation** the compiler is
  licensed to use — checked at every re-association site.

The two certificates therefore cannot interfere *syntactically*: they guard different
relations. This is what the 08-23 "orthogonal by construction" wanted to say and did not
have the objects to say. It is now a theorem (`nsDisjoint_reassoc_invariant`: the NS
premises of `(xy)z` and `x(yz)` are logically equivalent — re-association neither creates
nor destroys a support certificate; and `assocCert` mentions no `N`).

This connects the anti-garbling programme to the **equivalence theory A-1 /
`Invariant<T,G>`** line (f64 reassociation gated off by default): an uncertified
re-association is a claim of invariance under a transformation the value does not have.

---

## 2. The calculus (`§F` of the Lean file)

**Carrier.** `CD(n)` on integer coordinates `Vec := ℕ → ℤ` (live indices `< 2ⁿ`), product
`(a·b)_k = Σ_{i<2ⁿ} σ(i, i⊕k)·a_i·b_{i⊕k}` with the Cayley–Dickson sign `cdSigma` written by
*structural* recursion on the bit-width (so the kernel can evaluate it under `decide`; the
`termination_by` form in `SounioCayleyDickson.lean` is opaque to `decide`, and `native_decide`
cannot run on this host — §7). Proven: biadditivity, ℤ-homogeneity, congruence on live
coordinates, `‖a+b‖² = ‖a‖² + ‖b‖² + 2⟨a,b⟩`.

**Affine forms with vector coefficients.** `Aff := List (ℕ × Vec)`; `coeff a s` sums the
coefficients on source `s`; `innerA`, `trueVar a := innerA a a = Σ_s ‖∂_s‖²` (Lemma 0). Ports
of the kernel lemmas: `trueVar_append`, `innerA_disjoint`, `innerA_zero_of_ns`. Leibniz
scalings `scaleR a y = [(s, ∂_s·y)]`, `scaleL x b = [(s, x·∂_s)]`.

**Types.** `Ty := ⟨N : NS, Q : QS⟩`, both `Option (List ℕ)` with `none = ⊤`.
`Covers N a` (every source of the true form is tracked) is the axis-2 invariant of values.
Its **twin** `qCovers n Q v := ∀ k < 2ⁿ, v_k ≠ 0 → k ∈ Q` (every live basis element of the
payload is tracked) and `qCoversAff n Q a` (…of every sensitivity vector) are the axis-1
invariant. `Q` propagates by `qUnion` under `+` and by the XOR-product
`qProd Qa Qb = {i ⊕ j}` under `·` (`qCovers_cdMul`: the product's support lies in the XOR
set — the exact analogue of `covers_union`).

**Typing.** `t_kadd`/`t_kmul` keep the NS premise `nsDisjoint Na Nb = true` (E230);
`t_kraw` demands `Covers N a ∧ qCovers Q v ∧ qCoversAff Q a`. No rule mentions the
associator.

**Reduction.** As in the NS kernel, carrying true forms. The product's *reported* variance
is the **sensitivity (affine) propagator** `gMulMeta = trueVar(scaleR a y) + trueVar(scaleL x b)`
— the two Leibniz halves, cross term dropped. This uses **no norm identity**. The **GUM
variance shortcut** `gMulShortcut = ‖y‖²·Var x + ‖x‖²·Var y` (which *is*
`EpistemicEffectsNS.gMulMeta` at n = 0) is defined alongside for §4.

**Axis 2 lifted to every `CD(n)`.** `typed_agfree`, `preservation` (the `⟨N,Q⟩` type is
preserved), `exact_preservation`, `soundness_star`. The proof is the kernel's proof with
`Int` replaced by `Vec`; nothing about associativity or norm multiplicativity is needed —
which is *why* the sensitivity propagator is the honest one (§4).

---

## 3. Axis 1 and the fusion (`§B, §E, §H`)

**The certificate.** `assocCert n Qx Qy Qz : Bool` is true when `n ≤ 2` (ℝ, ℂ, ℍ — mirrors
`SounioCayleyDickson.canReassociate` and `ir_can_reassociate_triple`), or when `n = 3` and
`Qx ∪ Qy ∪ Qz ⊆ {0} ∪ ℓ` for one of the seven Fano lines `ℓ` (a quaternionic subalgebra).
⊤ is never certified beyond `n ≤ 2`.

**Certificate soundness = trilinearity by support induction** (the real work):
- `assoc_add{1,2,3}`, `assoc_smul{1,2,3}`: the associator is ℤ-trilinear.
- `peel_decomp`: `v = v_i·e_i + v'` with `v'_i = 0`; `qCoversL_peel`: the remainder is
  supported in the tail.
- `assoc_slot{1,2,3}`: induction on the support list — if the associator vanishes on the basis
  elements of `L` (other slots fixed), it vanishes on every vector supported in `L`.
- `assoc_zero_of_qCoversL`: compose the three slots ⇒ **an associative basis-index set is an
  associative subalgebra.** This is the axis-1 analogue of `inner_zero_of_ns`: the syntactic
  cover discharges the semantic condition.
- `fano_lines_assoc` (7 × 64 basis triples), `quaternion_assoc`, `complex_assoc`,
  `real_assoc`: kernel-`decide`d. `non_fano_124`: `[e₁,e₂,e₄] = 2e₇ ≠ 0`.
- `assoc_zero_of_cert`: certified *types* ⇒ vanishing associator of every covered triple of
  *values*.

**The exact gaps, in the calculus.**
- `reassoc_payload_gap`: `(xy)z − x(yz) = [x,y,z]` (by definition; recorded).
- `reassoc_sensitivity_gap`: per source `s`,
  `∂_s((xy)z) − ∂_s(x(yz)) = [∂_s x, y, z] + [x, ∂_s y, z] + [x, y, ∂_s z]`
  — the 08-23 §3B identity, now about the true forms the calculus actually carries.

**THE FUSION THEOREM** (`reassoc_sound`). Let `X, Y, Z` be well-typed values at
`⟨Nx,Qx⟩, ⟨Ny,Qy⟩, ⟨Nz,Qz⟩`, exact, with the source `(XY)Z` passing its two NS checks, and
`assocCert n Qx Qy Qz = true`. Then the target `X(YZ)`:
1. is well-typed at `⟨Nx ∪ (Ny ∪ Nz), Qx ⊗ (Qy ⊗ Qz)⟩` (via `nsDisjoint_reassoc_invariant`);
2. evaluates to the **same payload** on live coordinates;
3. evaluates to a **pairwise-equal true form** (`reassoc_forms_eq`: each entry differs by one
   covered associator, which the certificate kills);
4. **reports the same variance**, and both reports are exact.

Proof of (4) is the fusion in one line: `Var_src = trueVar(src form)` by axis-2 exactness,
`trueVar(src form) = trueVar(tgt form)` by axis-1 form equality, `= Var_tgt` by axis-2
exactness again. **Neither certificate alone suffices:**

| witness | axis 2 (NS) | axis 1 (assoc) | what breaks |
|---|---|---|---|
| **W1** `e₁,e₂,e₄` on sources 0,1,2 (n=3) | clean — `w1_typable` | `assocCert = false` — `w1_cert_refused` | value changes by `2e₇` (`w1_reassoc_changes_value`); source-0 sensitivity changes too (`w1_sensitivity_changes`) |
| **W1′** `e₁,e₂,e₃` on 0,1,2 (n=3) | clean | `true` — `w1'_cert` | nothing — `reassoc_sound` instantiated (`w1'_reassoc_sound`) |
| **W2** `x·x`, shared source 0 (n=0) | E230 — `w2_untypable` | `true` unconditionally — `assocCert_level0` | reported 200, true 400 (`w2_understates`) |

W1 is the pair the 08-23 note described in prose ("octonion basis elements have disjoint
sources yet a non-vanishing associator") — now a program the type system admits and the
rewrite certificate refuses, with the value change kernel-checked.

---

## 4. The third axis — correction to "exactly two" (`§J`)

The 08-23 Proposition 2 counted the inputs to the **sensitivity vector**: the symbol
assignment and the parenthesization. That count is correct — and `exact_preservation`
proves it for every `CD(n)` for the propagator that *carries* sensitivities.

But the scalar `ep_mul` — and GUM in general — does **not** carry sensitivities. It carries
variances and uses `Var(x·y) ≈ y²·Var x + x²·Var y`, which for a vector carrier is
`‖y‖²·Var x + ‖x‖²·Var y`. Comparing with the true `Σ_s ‖∂_s x · y‖²`, the shortcut has
silently imposed a third identity the object may not satisfy:

> **‖d·y‖² = ‖d‖²·‖y‖²** — norm multiplicativity.

That identity holds exactly in the composition algebras ℝ, ℂ, ℍ, 𝕆 (Hurwitz) and **fails in
the sedenions**. It is an identity-imposition in the precise sense of the 08-23 §0
definition, so it is a *structural* garbling, not curvature. Kernel-checked
(`sed_shortcut_understates`): at n = 4, `X` with sensitivity `d = e₁ + e₁₀` on source 0,
`Y = e₄ + e₁₅` exact — disjoint sources, one product, no re-association, both certificates
hold — the shortcut reports `‖Y‖²·Var X = 2·2 = 4`; the true first-order variance is
`‖d·Y‖² = 8`. **Understatement by a factor of 2 with a clean bill from both certificates.**
The sensitivity propagator reports 8 (exact). Controls: the octonion analogue
(`oct_shortcut_exact`: 4 = 4, Hurwitz), and a zero-divisor pair where the shortcut
*over*-states (`sed_shortcut_overstates`: 4 vs 0 — non-multiplicative norms garble in both
directions, so the shortcut is not even conservative). At n = 0 the shortcut *is* the
sensitivity propagator (`shortcut_eq_sensitivity_level0`), which is why the NS kernel never
met this axis.

**Corrected statement of completeness (C′).** For first-order bilinear propagation:
- a propagator that **carries per-source sensitivity vectors** has exactly two structural
  garblings — support and order — and both certificates together give exactness on every
  `CD(n)` (`exact_preservation` + `reassoc_sound`);
- a propagator that **carries variances only** has a **third** obligation — norm
  multiplicativity of the carrier — discharged for free iff the carrier is a composition
  algebra (`n ≤ 3`), and **false** from the sedenions up.

The three axes are three identities: `⟨εᵢ,εⱼ⟩ = δᵢⱼ` (measure), `(xy)z = x(yz)`
(multiplication), `‖xy‖ = ‖x‖‖y‖` (norm). Each is the kernel of a projection GUM performs
silently. The first two are certified by `N` and `Q`; the third is certified by the
**level** `n ≤ 3` — which is already a type in Sounio (`Sedenion` vs `Octonion`).

---

## 5. Honest scope

- **Operator fragment.** The Lean calculus has values, `measure`, `certain`, `opaque`,
  `kadd`, `kmul`; no λ/let. The λ-metatheory (β, substitution, weakening, progress) is in
  `EpistemicEffectsNS.lean` and is carrier-agnostic; it was not re-ported. The fusion content
  lives entirely in the operator fragment.
- **Re-association is stated at the redex on values**, not as a congruence-closed rewrite
  relation on open terms. The theorem is about the two programs `(XY)Z` and `X(YZ)`; lifting
  to contexts is boilerplate, not insight.
- **The n = 3 certificate recognises only Fano-line subalgebras** (`{0} ∪ ℓ`), not every
  associative subset (e.g. `{0, i}` ≅ ℂ, or a full ℍ in a non-standard basis). It is sound,
  and conservative — like NS's disjoint-support test versus the exact zero-covariance
  condition (08-25 skeleton §4.4). Sharpening it is a decidable-predicate change; the
  soundness theorem's shape does not move.
- **`Q` under products is the XOR list**, which can carry duplicates and grows; a canonical
  set representation is an engineering matter.
- **The third axis is exhibited, not classified**: I show norm-multiplicativity is a third
  identity the variance shortcut imposes and that it fails at n = 4; I do not prove there is
  no fourth for variance-only propagators. For the sensitivity propagator, `exact_preservation`
  *is* the completeness proof — nothing else is assumed.

## 6. Falsifiers

- **F1′** (kills `reassoc_sound`): a certified triple whose two parenthesizations evaluate to
  different live payloads or different reported variances. Impossible by the theorem; a
  counterexample would locate a bug in `assoc_zero_of_cert` or the support induction.
- **F7** (kills the third-axis reading): show `‖d·y‖² = ‖d‖²‖y‖²` holds for all sedenions —
  refuted by `sed_shortcut_understates` (8 ≠ 4).
- **F8** (kills the level-3 exemption): an octonion pair with `‖d·y‖² ≠ ‖d‖²‖y‖²`. Hurwitz
  says none exists; 2000 random integer pairs (this session) found none;
  `oct_shortcut_exact` is one kernel-checked instance, not the theorem — a Mathlib-free proof
  of octonion norm multiplicativity over `cdMul 3` is a natural next lemma.

## 7. Lessons (host + toolchain)

- `native_decide` **cannot run on this pod** (`failed to create thread` from the native
  compiler even with `--threads=1`); `SounioCayleyDickson.lean` therefore does not build here.
  Any module meant to be gate-checked on this host must be kernel-`decide`d only.
- `decide` cannot unfold a `termination_by` (well-founded) definition. Rewriting `cdSigma` by
  structural recursion on the bit-width made `decide` work; 7 × 64 basis-triple associators
  plus the calculus witnesses decide in ~25 s.
- `omega` atomises non-linear products: distribute with `Int.mul_add`/`Int.add_mul` first,
  then `omega` closes the linear residue (same pattern as the 08-23 toy model).

## 8. What this unlocks (items 1–2 done the same day — §10; 3 open)

1. **Wire `Q` into the checker** next to `noise_set_id`: the machinery already exists in
   `ir_can_reassociate_triple` / `cd_sigma_ct`; the type-level `Q` tag makes the e-graph
   reassociation rule (`beta5` draft: `reassociate: fano_selective`) a *checked*
   precondition — the first E230-family rejection for order fabrication.
2. **Fail-close the variance shortcut above level 3.** Any `ep_mul` on a sedenion-typed
   `Knowledge` must either carry sensitivities or be rejected — the third axis says the
   scalar formula is unsound there even with perfect NS hygiene.
3. **Paper A**: §4 gains the fusion theorem and the corrected (C′); the sedenion witness is a
   one-paragraph result no competing system can state.

## 10. The compiler wire (same day) — what the theorem changed in `self-hosted/check`

Measured first, on the committed Madaros (`bin/madaros-linux-x86_64`, md5 `ff69dae4`): both
fail-opens below returned `check: OK`.

**E251 — axis 3 at the product.** `Knowledge<Hyper<A,_>> * Knowledge<Hyper<A,_>>` with
`A` of algebra kind ≥ 4 (Sedenion, Clifford) is refused. The site is the E245 exemption for
`TyHyper` inners (`check.sio`, Knowledge×Knowledge binary), whose variance propagation is
`epsilon_combine_relative` — the shortcut. Octonion products and sedenion *sums* stay
admitted (`octonion_shortcut_exact`; no norm identity in `+`).

**E252 — axis 1 at the declaration.** An `algebra … { reassociate: … }` clause is a claim
of order freedom; it is licensed only when the declared law backs it, mirroring `assocCert`:
`free` needs an associative product (else E252 — the W1 licence); `fano_selective` needs the
octonion level (on Sedenion, E252). **Omitted clause:** the parser now yields −1 and the
checker derives the most permissive certified strategy (associative → free; alternative at
level 3 → fano_selective; otherwise blocked). Before, an omitted clause was silently `free` —
an alternative algebra received free reassociation.

**Two findings about the algebra registry, both measured on source builds.**
(i) In the by-value spine, `collect_algebra_def` walked the op declarations into
`entries[idx]` and then overwrote the entry with `info` (defaults: associative) — the registry
never recorded `alternative`. Fixed by ordering. (ii) More consequential: that spine is **dead
for Madaros**. The live collector is the `*mut` if-chain `checker_collect_item_inplace`, which
had **no `ItemAlgebra` arm** — every `algebra … { }` declaration was inert in the Madaros
checker (registry empty; `check_hyper_binary` fell back to hard-coded predicates; the IR
algebra table saw only name-derived defaults). Proven by an env-gated trace on a source build
(Slurm job 11422, ELF `25a43cbb`): the by-value collector never ran, so the first E252 build
was silent. `checker_collect_algebra_def_inplace` now exists, registers the declaration
(tag, laws) and carries E252 and the fail-closed default. A related parser slip surfaced on
the same build: the branch for an explicit `reassociate: free` never assigned, so with the new
−1 default an explicit `free` read as omitted — fixed.

**Measurement discipline that made this visible.** Every refusal is paired with an accepted
control on the same compiler; E251 is measured with NS on (E230 also fires: ⊤-parameter
operands are refused by design) and with `SOUNIO_NS_DISABLE=1` (E230 vanishes, E251 must
survive — causal separability, the same move the NS gate makes for E245). The octonion
control is therefore a gate-only fixture (`tests/fixtures/antigarbling/`), not a suite test.

**What was NOT changed, and why.** The small e-graph's `fano_selective` gate reads
`EgNode.value` as a basis index, but `opt_cleanup.sio` creates VAR nodes with IR *register*
ids. The gate therefore compares register numbers against Fano lines. It is inert in
practice — octonion/sedenion products are single IR instructions (`IrHyperMulO/S`) that never
enter the FADD/FMUL e-graph, whose FMULs are scalar f64 (associative in ℝ) — so nothing is
unsound today, but the "Q certificate" it pretends to be does not exist there. Making it real
needs a basis-support mask on e-graph nodes fed from the IR; that is the `Q` propagation of
§2 (`qUnion`/`qProd`) as engineering, left explicitly open.

**Measured (Slurm job 11425, source build of `b1775894cc`, ELF `fca28454`):**
`antigarbling_third_axis_gate.sh` 8/8 PASS — E251 refused with NS on and surviving
`SOUNIO_NS_DISABLE=1`; octonion product + sedenion sum accepted; `free` on alternative and
`fano_selective` on Sedenion refused with E252; omitted clause on alternative accepted;
`algebra_decl_basic` and `octonion_hessian_fano_annotated` still accepted. `ns_antigarbling_gate.sh`
(E230) still OK. The hyper/algebra check sweep has 6 failures (E015/E004/E008/E036), all
identical on the committed compiler `ff69dae4` — pre-existing, not regressions.

Gate: `scripts/ci/antigarbling_third_axis_gate.sh` (refusals paired with accepted controls
on the same compiler), named in `ci.yml`. Fixtures: `tests/compile-fail/e251_*.sio`,
`e252_*.sio`; controls in `tests/run-pass/`. Catalogue rows E251/E252 with explanations.

## 9. Theorem map

| claim | Lean |
|---|---|
| CD product biadditive / homogeneous / live-congruent | `cdMul_add_left/right`, `cdMul_smul_left/right`, `cdMul_congr_left/right` |
| `‖a+b‖²` expansion | `normSq_add` |
| associator trilinear | `assoc_add{1,2,3}`, `assoc_smul{1,2,3}` |
| support induction | `peel_decomp`, `qCoversL_peel`, `assoc_slot{1,2,3}` |
| associative index set ⇒ associative subalgebra | `assoc_zero_of_qCoversL` |
| Fano lines / ℍ / ℂ / ℝ associative; (1,2,4) not | `fano_lines_assoc`, `quaternion_assoc`, `complex_assoc`, `real_assoc`, `non_fano_124(_value)` |
| axis-1 certificate soundness | `assoc_zero_of_cert` |
| axis-2 certificate soundness (vector coefficients) | `innerA_zero_of_ns`, `innerA_disjoint`, `trueVar_append` |
| `Q` propagation | `qCovers_vadd`, `qCovers_cdMul`, `qCoversAff_scaleR/L`, `qCovers_coeff` |
| syntactic orthogonality | `nsDisjoint_reassoc_invariant` |
| axis 2 on every `CD(n)` | `typed_agfree`, `preservation`, `exact_preservation`, `soundness_star` |
| the two exact gaps | `reassoc_payload_gap`, `reassoc_sensitivity_gap` |
| certified forms equal | `reassoc_forms_eq` |
| **FUSION** | `reassoc_sound` |
| orthogonality witnesses | `w1_typable`, `w1_cert_refused`, `w1_reassoc_changes_value`, `w1_sensitivity_changes`, `w1'_cert`, `w1'_reassoc_sound`, `assocCert_level0`, `w2_untypable`, `w2_understates` |
| third axis | `sed_shortcut_understates`, `sed_x_typable`, `oct_shortcut_exact`, `sed_shortcut_overstates`, `shortcut_eq_sensitivity_level0` |
