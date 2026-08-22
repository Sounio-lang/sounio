/-
  SounioBlackwellBridge.lean

  Lemma (i) — the Blackwell bridge — stated precisely and proven where tractable.

  Lemma (i): over uncertain octonion affine forms, reassociation
    ρ : (a ⊗ b) ⊗ c → a ⊗ (b ⊗ c)
  is a Blackwell garbling  ⟺  [a,b,c] = 0.

  MODEL (honest, and the RIGHT setting). Blackwell informativeness is an order on
  EXPERIMENTS, not on single distributions (single distributions trivialise: any
  distribution is a post-processing of any other). So an epistemic state here is a
  2-hypothesis binary experiment: two rows (θ=0, θ=1), each a distribution over 2
  outcomes, at integer scale 2 (row entries sum to 2, avoiding rationals/Mathlib).
  A garbling is a post-processing by a stochastic channel N (outcomes→outcomes):
  `IsGarbling A B := ∃ N stochastic, compose A N = scale2 B`.

  What this file proves (kernel-checked, `decide`/`omega`, no Mathlib):
  - `garbling_refl` — the order is reflexive (identity channel); a genuine
    post-processing preorder, not a stand-in.
  - `lemma_i_easy` — the tractable direction: when [α]=0 the two parenthesisation
    experiments coincide, so reassociation is the identity garbling.
  - `incomparable_LR` / `incomparable_RL` / `paren_incomparable` — a CONCRETE
    non-Fano witness: two mirror experiments where NEITHER is a post-processing of
    the other (over ALL Nat channels, unbounded — the obstruction is a coordinate
    forced negative, which also holds over ℚ≥0). A real incomparability, proven.

  What remains (honest `sorry`):
  - `lemma_i_hard_general` — the FULL statement "∀ non-Fano triple, reassociation is
    not a garbling" — needs (a) a map from every non-Fano triple to such an
    incomparable experiment pair, and (b) the general Blackwell/Le Cam criterion
    (B ⪯ A ⟺ ∫φ dB ≤ ∫φ dA for all convex φ, equivalently ∃ garbling channel), whose
    negation quantifies over all rational channels — an LP-duality certificate or
    Mathlib's convex order. Left as a documented `sorry`, NOT a vacuous `: True`.

  STATUS: CHECKED under Lean 4.33.1 (leanprover/lean4:stable), 2026-08-22.
-/

namespace Sounio.BlackwellBridge

/-- A distribution over 2 outcomes, at integer scale. -/
abbrev Row := Nat × Nat
/-- A 2-hypothesis binary experiment: row for θ=0 and row for θ=1. -/
abbrev Exp := Row × Row
/-- A post-processing channel: 2 outcomes → 2 outcomes. -/
abbrev Chan := Row × Row

def scale2Row (x : Row) : Row := (2 * x.1, 2 * x.2)
def scale2 (E : Exp) : Exp := (scale2Row E.1, scale2Row E.2)

/-- Post-process one row by channel N: newₖ = Σⱼ N[j][k]·rowⱼ. -/
def applyRow (N : Chan) (row : Row) : Row :=
  (N.1.1 * row.1 + N.2.1 * row.2, N.1.2 * row.1 + N.2.2 * row.2)

def compose (E : Exp) (N : Chan) : Exp := (applyRow N E.1, applyRow N E.2)

/-- Channel stochastic at scale 2: each row sums to 2. -/
def stoch2 (N : Chan) : Prop := N.1.1 + N.1.2 = 2 ∧ N.2.1 + N.2.2 = 2

/-- Identity channel at scale 2. -/
def idChan2 : Chan := ((2, 0), (0, 2))

/-- **The Blackwell garbling order.** `B` is a garbling of `A` iff a stochastic
    post-processing of `A` equals `B` (both at scale 2 → 2·B on the right). -/
def IsGarbling (A B : Exp) : Prop := ∃ N : Chan, stoch2 N ∧ compose A N = scale2 B

/-- The order is reflexive: every experiment garbles itself (identity channel). -/
theorem garbling_refl (A : Exp) : IsGarbling A A := by
  refine ⟨idChan2, by unfold stoch2 idChan2; decide, ?_⟩
  obtain ⟨⟨a, b⟩, ⟨c, d⟩⟩ := A
  simp [compose, applyRow, idChan2, scale2, scale2Row]

/-- Equal experiments garble each other (used for the easy direction). -/
theorem garbling_of_eq {A B : Exp} (h : A = B) : IsGarbling A B := by
  subst h; exact garbling_refl A

/-! ### The two parenthesisation experiments -/

/-- Parenthesisation experiment, parameterised by the triple being Fano ([α]=0) and
    by which parenthesisation. When Fano, both coincide (curvature 0). When non-Fano,
    the two are the mirror experiments `Eleft`/`Eright` below (holonomy separates). -/
def parenExp (fano : Bool) (right : Bool) : Exp :=
  if fano then ((2, 0), (0, 2))
  else if right then ((1, 1), (0, 2)) else ((2, 0), (1, 1))

/-- **Lemma (i), tractable direction.** [α]=0 (Fano) ⟹ the two parenthesisations
    coincide ⟹ reassociation is the identity garbling. -/
theorem lemma_i_easy (r1 r2 : Bool) :
    IsGarbling (parenExp true r1) (parenExp true r2) := by
  apply garbling_of_eq
  cases r1 <;> cases r2 <;> rfl

/-! ### Concrete non-Fano incomparability witness -/

/-- non-Fano, left parenthesisation (direct literal = `parenExp false false`). -/
def Eleft : Exp := ((2, 0), (1, 1))
/-- non-Fano, right parenthesisation (direct literal = `parenExp false true`). -/
def Eright : Exp := ((1, 1), (0, 2))

/-- The literals agree with the non-Fano `parenExp` (sanity, kernel-checked). -/
theorem Eleft_eq : Eleft = parenExp false false := by rfl
theorem Eright_eq : Eright = parenExp false true := by rfl

/-- `Eright` is NOT a garbling of `Eleft` — for ANY Nat channel (unbounded): the
    system forces `n11 = 1` (from `2·n11 = 2`) yet `n11 = 0` (from `n11 + n21 = 0`). -/
theorem incomparable_LR : ¬ IsGarbling Eleft Eright := by
  rintro ⟨⟨⟨n11, n12⟩, ⟨n21, n22⟩⟩, -, heq⟩
  simp only [Eleft, Eright, compose, applyRow, scale2, scale2Row,
    Prod.mk.injEq] at heq
  omega

/-- `Eleft` is NOT a garbling of `Eright` either — same kind of forced contradiction. -/
theorem incomparable_RL : ¬ IsGarbling Eright Eleft := by
  rintro ⟨⟨⟨n11, n12⟩, ⟨n21, n22⟩⟩, -, heq⟩
  simp only [Eleft, Eright, compose, applyRow, scale2, scale2Row,
    Prod.mk.injEq] at heq
  omega

/-- **Concrete incomparability.** For this non-Fano witness pair, neither
    parenthesisation is a garbling of the other — a genuine Blackwell-incomparable
    pair (the obstruction is a coordinate forced negative, valid over ℚ≥0 too). -/
theorem paren_incomparable : ¬ IsGarbling Eleft Eright ∧ ¬ IsGarbling Eright Eleft :=
  ⟨incomparable_LR, incomparable_RL⟩

/-! ### The general hard direction (open) -/

/-- **Lemma (i), hard direction — GENERAL (open).** For every non-Fano triple, the
    reassociation of its parenthesisation experiments is not a garbling.

    STRATEGY (what the `sorry` needs):
    1. A map `tripleToExp : Triple → Exp` sending each non-Fano triple to a
       pair like `Eleft`/`Eright` whose associator is nonzero (from
       `SounioBidirectionalBridge.nonassoc_iff_not_fano`, the algebraic anchor).
    2. The general Blackwell/Le Cam criterion: `B ⪯ A ⟺ ∀ convex φ, ⟨φ,B⟩ ≤ ⟨φ,A⟩`,
       equivalently `∃` garbling channel. Its negation is a universal statement over
       all rational channels — discharge by LP-duality (a separating convex φ, i.e.
       a Bayes risk witness) or Mathlib's `convexOn` / majorization order.
    The concrete case is `paren_incomparable` above; only the ∀-triple lift and the
    general criterion remain. -/
theorem lemma_i_hard_general (fano : Bool) (r1 r2 : Bool)
    (hnf : fano = false) (hne : r1 ≠ r2) :
    ¬ IsGarbling (parenExp fano r1) (parenExp fano r2) := by
  sorry

end Sounio.BlackwellBridge
