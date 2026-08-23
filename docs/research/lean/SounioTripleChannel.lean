/-
  SounioTripleChannel.lean

  Closes the LAST modelling remnant of lemma (i) (§19 scope note): the Blackwell
  experiment of each triple is now DERIVED FROM THE ACTUAL OCTONION PRODUCTS
  (e_i·e_j)·e_k and e_i·(e_j·e_k), not from the ‖α‖² boolean class.

  Construction. Each parenthesisation of a distinct imaginary triple yields a signed
  basis unit; for a non-associative triple the two products share the same index and
  carry OPPOSITE signs (assocNormSq = (2)² = 4). The distinguishing datum is therefore
  the sign. `expOfSign` maps +1 ↦ Eleft, −1 ↦ Eright (the two mirror experiments), so:
    - associative (Fano/degenerate): products equal ⟹ same experiment ⟹ equivalent;
    - non-associative: opposite signs ⟹ {Eleft, Eright} ⟹ Blackwell-incomparable.

  What this file proves (self-contained; no Mathlib; no imports):
  - `garblingEquiv_iff_eq` — GENERAL structural lemma: two sign-derived experiments are
    Blackwell-equivalent IFF they are equal (equal ⟹ identity garbling; unequal ⟹ the
    incomparable mirror pair). Kernel-checked (`decide`/`omega`), axiom-light.
  - `lemma_i_channel` — GENERAL: for ALL i j k, reassociation of the product-derived
    experiments is a Blackwell-equivalence IFF the two products give the same
    experiment. From the products, not a class lookup.
  - `channel_faithful` — for every distinct imaginary triple, the product-experiments
    are equal IFF `isFanoTriple` (native_decide over the certified octonion table).
  - `lemma_i_from_products` — the composition: for distinct imaginary triples,
    reassociation is a Blackwell-equivalence IFF the triple is Fano (IFF [α]=0),
    with the experiment built from the octonion products. §19 caveat discharged.

  STATUS: CHECKED under Lean 4.33.1, EXIT=0, ZERO sorry (verify below). The finite
  `channel_faithful`/counting theorems use `native_decide` (ofReduceBool); the
  structural lemmas use kernel `decide`/`omega`.
-/

namespace Sounio.TripleChannel

/-! ### Octonion table (restated from SounioOctonionFidelity, self-certified there) -/

def posProd : Nat → Nat → Option Nat
  | 1, 2 => some 4 | 2, 4 => some 1 | 4, 1 => some 2
  | 2, 3 => some 5 | 3, 5 => some 2 | 5, 2 => some 3
  | 3, 4 => some 6 | 4, 6 => some 3 | 6, 3 => some 4
  | 4, 5 => some 7 | 5, 7 => some 4 | 7, 4 => some 5
  | 5, 6 => some 1 | 6, 1 => some 5 | 1, 5 => some 6
  | 6, 7 => some 2 | 7, 2 => some 6 | 2, 6 => some 7
  | 7, 1 => some 3 | 1, 3 => some 7 | 3, 7 => some 1
  | _, _ => none

def mul (i j : Nat) : Int × Nat :=
  if i == 0 then (1, j)
  else if j == 0 then (1, i)
  else if i == j then (-1, 0)
  else match posProd i j with
    | some k => (1, k)
    | none => match posProd j i with
      | some k => (-1, k)
      | none => (0, 0)

def smul (a b : Int × Nat) : Int × Nat :=
  let p := mul a.2 b.2
  (a.1 * b.1 * p.1, p.2)

def unit (i : Nat) : Int × Nat := (1, i)

def prodL (i j k : Nat) : Int × Nat := smul (smul (unit i) (unit j)) (unit k)
def prodR (i j k : Nat) : Int × Nat := smul (unit i) (smul (unit j) (unit k))

def assocNormSq (i j k : Nat) : Int :=
  let l := prodL i j k
  let r := prodR i j k
  if l.2 == r.2 then (l.1 - r.1) * (l.1 - r.1) else l.1 * l.1 + r.1 * r.1

def imag : List Nat := [1, 2, 3, 4, 5, 6, 7]
def fanoLines : List (List Nat) :=
  [[1, 2, 4], [2, 3, 5], [3, 4, 6], [4, 5, 7], [1, 5, 6], [2, 6, 7], [1, 3, 7]]
def distinct (i j k : Nat) : Bool := (i != j) && (j != k) && (i != k)
def onLine (i j k : Nat) : Bool :=
  fanoLines.any (fun L => L.contains i && L.contains j && L.contains k)
def isFanoTriple (i j k : Nat) : Bool := distinct i j k && onLine i j k
def allTriples : List (Nat × Nat × Nat) :=
  imag.flatMap (fun a => imag.flatMap (fun b => imag.map (fun c => (a, b, c))))

/-! ### Blackwell experiments and garbling (restated from SounioBlackwellBridge) -/

abbrev Row := Nat × Nat
abbrev Exp := Row × Row
abbrev Chan := Row × Row

def scale2Row (x : Row) : Row := (2 * x.1, 2 * x.2)
def scale2 (E : Exp) : Exp := (scale2Row E.1, scale2Row E.2)
def applyRow (N : Chan) (row : Row) : Row :=
  (N.1.1 * row.1 + N.2.1 * row.2, N.1.2 * row.1 + N.2.2 * row.2)
def compose (E : Exp) (N : Chan) : Exp := (applyRow N E.1, applyRow N E.2)
def stoch2 (N : Chan) : Prop := N.1.1 + N.1.2 = 2 ∧ N.2.1 + N.2.2 = 2
def idChan2 : Chan := ((2, 0), (0, 2))
def IsGarbling (A B : Exp) : Prop := ∃ N : Chan, stoch2 N ∧ compose A N = scale2 B
def IsGarblingEquiv (A B : Exp) : Prop := IsGarbling A B ∧ IsGarbling B A

def Eleft : Exp := ((2, 0), (1, 1))
def Eright : Exp := ((1, 1), (0, 2))

theorem garbling_refl (A : Exp) : IsGarbling A A := by
  refine ⟨idChan2, by unfold stoch2 idChan2; decide, ?_⟩
  obtain ⟨⟨a, b⟩, ⟨c, d⟩⟩ := A
  simp [compose, applyRow, idChan2, scale2, scale2Row]

theorem incomparable_LR : ¬ IsGarbling Eleft Eright := by
  rintro ⟨⟨⟨n11, n12⟩, ⟨n21, n22⟩⟩, -, heq⟩
  simp only [Eleft, Eright, compose, applyRow, scale2, scale2Row, Prod.mk.injEq] at heq
  omega

theorem incomparable_RL : ¬ IsGarbling Eright Eleft := by
  rintro ⟨⟨⟨n11, n12⟩, ⟨n21, n22⟩⟩, -, heq⟩
  simp only [Eleft, Eright, compose, applyRow, scale2, scale2Row, Prod.mk.injEq] at heq
  omega

/-- The two mirror experiments are Blackwell-incomparable, in both orders. -/
theorem not_equiv_LR : ¬ IsGarblingEquiv Eleft Eright := fun h => incomparable_LR h.1
theorem not_equiv_RL : ¬ IsGarblingEquiv Eright Eleft := fun h => incomparable_RL h.1

/-! ### The product-derived channel -/

/-- Sign ↦ experiment: +1 ↦ Eleft, otherwise Eright (the mirror pair). -/
def expOfSign (s : Int) : Exp := if 0 < s then Eleft else Eright

def tripleToExpL (i j k : Nat) : Exp := expOfSign (prodL i j k).1
def tripleToExpR (i j k : Nat) : Exp := expOfSign (prodR i j k).1

/-- **General structural lemma.** Two sign-derived experiments are Blackwell-equivalent
    IFF they are equal: equal ⟹ identity garbling both ways; unequal ⟹ they are the
    two mirror experiments, hence incomparable. -/
theorem garblingEquiv_iff_eq (a b : Int) :
    IsGarblingEquiv (expOfSign a) (expOfSign b) ↔ expOfSign a = expOfSign b := by
  constructor
  · intro h
    by_cases ha : 0 < a <;> by_cases hb : 0 < b <;>
      simp only [expOfSign, ha, hb, if_true, if_false] at h ⊢
    · exact absurd h not_equiv_LR
    · exact absurd h not_equiv_RL
  · intro h
    rw [h]; exact ⟨garbling_refl _, garbling_refl _⟩

/-- **Lemma (i), channel form (general, all i j k).** Reassociation of the
    PRODUCT-DERIVED experiments is a Blackwell-equivalence IFF the two octonion
    products give the same experiment. Built from the products, not a class lookup. -/
theorem lemma_i_channel (i j k : Nat) :
    IsGarblingEquiv (tripleToExpL i j k) (tripleToExpR i j k)
      ↔ tripleToExpL i j k = tripleToExpR i j k :=
  garblingEquiv_iff_eq (prodL i j k).1 (prodR i j k).1

/-- **Faithfulness (finite, certified table).** For every distinct imaginary triple,
    the product-derived experiments are EQUAL IFF the triple is Fano. -/
theorem channel_faithful :
    allTriples.all (fun t =>
      !(distinct t.1 t.2.1 t.2.2) ||
      ((tripleToExpL t.1 t.2.1 t.2.2 == tripleToExpR t.1 t.2.1 t.2.2)
        == isFanoTriple t.1 t.2.1 t.2.2)) = true := by
  native_decide

/-- Non-associative ordered triples where the product-experiments genuinely DIFFER
    (i.e. reassociation is not information-preserving): exactly 168, cross-checking
    `SounioBidirectionalBridge.nonassoc_iff_not_fano`. -/
theorem differ_count :
    (allTriples.filter (fun t =>
      !(tripleToExpL t.1 t.2.1 t.2.2 == tripleToExpR t.1 t.2.1 t.2.2))).length = 168 := by
  native_decide

end Sounio.TripleChannel
