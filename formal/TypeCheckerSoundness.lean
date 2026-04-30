/-!
# Sounio.TypeCheckerSoundness — Phase 8.4 Formal Verification

Operational soundness for `Knowledge<Float>` type-check against a small
big-step evaluation semantics.

**Context.**  `formal/TypeChecker.lean:307-317` previously carried:
```lean
theorem check_implies_infer ... : True → ∃ T' : Ty, Sub T' T := by
  intro _; exact ⟨T, Sub.refl T⟩
```
— a witness-only existential with no operational content.  Kimi 2.6's
forensic review flagged this trivial proof as the meta-gap Phase 3 would
close.

This file provides the first substantive operational soundness proof for
Sounio's epistemic calculus:

  * Extends type-level subtyping with an **epistemic bound**
    `EpBound { radius, conf_floor }` — a wrapper over `Sub` capturing the
    uncertainty envelope a program is asserted to respect.
  * Defines a minimal epistemic calculus (`KLit`, `KAdd`) with big-step
    evaluation semantics that mirror `stdlib/epistemic/gum.sio`
    (`gum_add` delta-method: `ε_sum = ε_a + ε_b`; combined confidence
    `= min(c_a, c_b)`).
  * Proves `check_ep_implies_eval_bounded`:  if `Check e b` holds and
    `e ⇓ v`, then `v.eps ≤ b.radius` and `v.conf ≥ b.conf_floor`.

**Scope.**  The calculus covers literals and addition.  Multiplication
is a documented open-work item: the GUM multiplication formula
(`ε = |b|·ε_a + |a|·ε_b`) requires the Check rule to commit to the
operand magnitudes ex ante; a `MagnitudeCert` sub-judgement can close
this gap in a follow-up phase.  The `Check`+`Eval` semantics for
addition already demonstrates the proof pattern.

Self-contained.  No Mathlib.  No dependency on `TypeChecker.lean`.
Zero sorry.  Three Float axioms (parallel to `SecondOrderGUM.lean §5`).
-/

namespace Sounio.TypeCheckerSoundness

-- ---------------------------------------------------------------------------
-- §1. Float axioms (parallel to SecondOrderGUM.lean §5)
-- ---------------------------------------------------------------------------

/-- Float `≤` is reflexive. -/
axiom float_le_refl (a : Float) : a ≤ a

/-- Float `≤` is transitive. -/
axiom float_le_trans (a b c : Float) : a ≤ b → b ≤ c → a ≤ c

/-- Float addition is monotone under `≤`. -/
axiom float_add_le_add (a b c d : Float) : a ≤ b → c ≤ d → a + c ≤ b + d

-- ---------------------------------------------------------------------------
-- §2. Minimal types and epistemic bounds
-- ---------------------------------------------------------------------------

/-- The subset of `Ty` this file reasons about. -/
inductive Ty : Type where
  | float      : Ty
  | knowledge  : Ty → Ty
  deriving DecidableEq, Repr

/-- Epistemic bound: the user-asserted upper bound on uncertainty radius
    and lower bound on confidence. -/
structure EpBound where
  radius      : Float
  conf_floor  : Float
  deriving Repr

/-- Tightening either the radius or the confidence floor produces a
    subtype (the wrapper over `Sub` that Phase 3 installs). -/
def SubEp (t1 t2 : Ty) (b1 b2 : EpBound) : Prop :=
  t1 = t2 ∧ b1.radius ≤ b2.radius ∧ b2.conf_floor ≤ b1.conf_floor

-- ---------------------------------------------------------------------------
-- §3. Structural lemmas over `SubEp`
-- ---------------------------------------------------------------------------

theorem subEp_refl (t : Ty) (b : EpBound) : SubEp t t b b := by
  refine ⟨rfl, ?_, ?_⟩
  · exact float_le_refl _
  · exact float_le_refl _

theorem subEp_trans (t1 t2 t3 : Ty) (b1 b2 b3 : EpBound)
    (h12 : SubEp t1 t2 b1 b2) (h23 : SubEp t2 t3 b2 b3) :
    SubEp t1 t3 b1 b3 := by
  obtain ⟨ht12, hr12, hc12⟩ := h12
  obtain ⟨ht23, hr23, hc23⟩ := h23
  refine ⟨ht12.trans ht23, ?_, ?_⟩
  · exact float_le_trans _ _ _ hr12 hr23
  · exact float_le_trans _ _ _ hc23 hc12

/-- Knowledge covariance: if `t1 = t2` and `b1 ≤ b2`,
    then `Knowledge t1 @ b1 ≤ Knowledge t2 @ b2`. -/
theorem subEp_knowledge_cov (t1 t2 : Ty) (b1 b2 : EpBound)
    (ht : t1 = t2)
    (hr : b1.radius ≤ b2.radius)
    (hc : b2.conf_floor ≤ b1.conf_floor) :
    SubEp (Ty.knowledge t1) (Ty.knowledge t2) b1 b2 := by
  refine ⟨?_, hr, hc⟩
  congr 1

-- ---------------------------------------------------------------------------
-- §4. The epistemic calculus
-- ---------------------------------------------------------------------------

/-- Minimal epistemic expression language.  Mirrors the runtime
    `Knowledge<f64>` arithmetic in `stdlib/epistemic/knowledge.sio`. -/
inductive Expr : Type where
  | kLit (val eps conf : Float) : Expr
  | kAdd (e1 e2 : Expr)         : Expr
  deriving Repr

/-- Values are Knowledge literals. -/
inductive IsValue : Expr → Prop where
  | litIsVal (val eps conf : Float) : IsValue (Expr.kLit val eps conf)

-- ---------------------------------------------------------------------------
-- §5. Big-step semantics
-- ---------------------------------------------------------------------------

/-- Big-step evaluation.  `Eval e v ε c` means `e` reduces to the
    Knowledge value with point estimate `v`, uncertainty radius `ε`,
    and confidence floor `c`.

    Addition composes ε linearly (delta-method first-order) and takes
    the minimum confidence via an `if-then-else` encoding of `min`. -/
inductive Eval : Expr → Float → Float → Float → Prop where
  | evLit (v ε c : Float) :
      Eval (Expr.kLit v ε c) v ε c
  | evAdd (e1 e2 : Expr) (v1 ε1 c1 v2 ε2 c2 : Float)
      (h1 : Eval e1 v1 ε1 c1) (h2 : Eval e2 v2 ε2 c2) :
      Eval (Expr.kAdd e1 e2) (v1 + v2) (ε1 + ε2)
           (if c1 ≤ c2 then c1 else c2)

-- ---------------------------------------------------------------------------
-- §6. The `Check` relation
-- ---------------------------------------------------------------------------

/-- `Check e b` certifies that every value of `e` (reachable by big-step
    evaluation) has uncertainty radius `≤ b.radius` and confidence
    floor `≥ b.conf_floor`. -/
inductive Check : Expr → EpBound → Prop where
  | chkLit (v ε c : Float) (b : EpBound)
      (hε : ε ≤ b.radius) (hc : b.conf_floor ≤ c) :
      Check (Expr.kLit v ε c) b
  | chkAdd (e1 e2 : Expr) (b1 b2 b : EpBound)
      (h1 : Check e1 b1) (h2 : Check e2 b2)
      (hradd : b1.radius + b2.radius ≤ b.radius)
      (hcon1 : b.conf_floor ≤ b1.conf_floor)
      (hcon2 : b.conf_floor ≤ b2.conf_floor) :
      Check (Expr.kAdd e1 e2) b

-- ---------------------------------------------------------------------------
-- §7. Main operational soundness theorem
-- ---------------------------------------------------------------------------

/-- Helper: `c ≤ (if a ≤ b then a else b)` whenever `c ≤ a` and `c ≤ b`. -/
private theorem ite_le_of_le (a b c : Float) (ha : c ≤ a) (hb : c ≤ b) :
    c ≤ (if a ≤ b then a else b) := by
  by_cases h : a ≤ b
  · simp [h]; exact ha
  · simp [h]; exact hb

/-- **Operational soundness (Phase 3's key theorem).**

    If `Check e b` holds, every big-step evaluation of `e` yields a value
    whose uncertainty radius does not exceed `b.radius` and whose
    confidence floor is no less than `b.conf_floor`.

    Replaces the trivial `check_implies_infer := ⟨T, Sub.refl T⟩` in
    `TypeChecker.lean:307` with a real bridge between the type-level
    bound and the operational evaluation.

    Proof is by structural induction on the `Check` derivation, using
    the `Eval` big-step rules and the three Float axioms to trace
    through each arithmetic operation. -/
theorem check_ep_implies_eval_bounded
    (e : Expr) (b : EpBound) (v ε c : Float)
    (hchk : Check e b) (hev : Eval e v ε c) :
    ε ≤ b.radius ∧ b.conf_floor ≤ c := by
  induction hchk generalizing v ε c with
  | chkLit v0 ε0 c0 b hε hc =>
    cases hev with
    | evLit => exact ⟨hε, hc⟩
  | chkAdd e1 e2 b1 b2 b h1 h2 hradd hcon1 hcon2 ih1 ih2 =>
    cases hev with
    | evAdd _ _ v1 ε1 c1 v2 ε2 c2 h_e1 h_e2 =>
      obtain ⟨hε1, hc1⟩ := ih1 v1 ε1 c1 h_e1
      obtain ⟨hε2, hc2⟩ := ih2 v2 ε2 c2 h_e2
      refine ⟨?_, ?_⟩
      · -- ε1 + ε2 ≤ b1.radius + b2.radius ≤ b.radius
        have add_mono : ε1 + ε2 ≤ b1.radius + b2.radius :=
          float_add_le_add _ _ _ _ hε1 hε2
        exact float_le_trans _ _ _ add_mono hradd
      · -- b.conf_floor ≤ min(c1, c2)  ⇐  b.conf_floor ≤ b1.conf_floor ≤ c1
        have c1_ok : b.conf_floor ≤ c1 := float_le_trans _ _ _ hcon1 hc1
        have c2_ok : b.conf_floor ≤ c2 := float_le_trans _ _ _ hcon2 hc2
        exact ite_le_of_le c1 c2 b.conf_floor c1_ok c2_ok

-- ---------------------------------------------------------------------------
-- §8. Summary
-- ---------------------------------------------------------------------------

/-!
## Verified Properties

| Property                                                   | Status  | Location                        |
|------------------------------------------------------------|---------|---------------------------------|
| SubEp reflexivity                                          | Proved  | `subEp_refl`                    |
| SubEp transitivity                                         | Proved  | `subEp_trans`                   |
| SubEp Knowledge covariance                                 | Proved  | `subEp_knowledge_cov`           |
| Operational soundness (`Check e b ∧ e ⇓ v ⟹ v ≤ b`)       | Proved  | `check_ep_implies_eval_bounded` |

## Compared with the prior trivial proof

`formal/TypeChecker.lean:307-317` previously had:
```lean
theorem check_implies_infer ... : True → ∃ T' : Ty, Sub T' T := by
  intro _; exact ⟨T, Sub.refl T⟩
```
— a vacuous existential witness, no operational content.

This file's `check_ep_implies_eval_bounded` quantifies over every
evaluation and proves the runtime uncertainty radius stays within the
type-asserted bound.  The proof is by induction on `Check`, using
`Eval` to trace each arithmetic step.

## Axioms

Three Float axioms (`float_le_refl`, `float_le_trans`, `float_add_le_add`),
parallel to `SecondOrderGUM.lean` and `Epistemic.lean`.

## Scope and future work

- Addition covered.  Multiplication is deferred: the first-order GUM
  formula `ε = |b|·ε_a + |a|·ε_b` requires the Check rule to commit to
  operand magnitudes; a `MagnitudeCert` sub-judgement can close this
  in a follow-up phase.  The proof pattern established here extends
  naturally (same induction schema, one extra arithmetic step).
- Soundness against the actual Sounio `check/types.sio` codegen — the
  theorem is over an abstract `Check` relation, not the x86-emitting
  `.sio` checker.  This disclosure applies to every proof in `formal/`.
- Completeness is not claimed: there are well-bounded programs for which
  no `Check` derivation exists.
-/

end Sounio.TypeCheckerSoundness
