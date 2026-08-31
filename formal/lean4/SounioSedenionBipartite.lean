/-
  SounioSedenionBipartite — K-odd / K-even bipartiteness of integer
  sedenion ZD-surgery unit-distance graphs.

  STATUS (2026-08-17): NO `sorry`. Every declaration is proved, Mathlib-free
  and kernel-checked under leanprover/lean4:v4.33.0, with no `native_decide`.
  Axioms are the three standard ones only (`propext`, `Classical.choice`,
  `Quot.sound`); the two computational witnesses need only `propext`. No
  `sorryAx` anywhere.
    · `twistedNormSq` — implemented (it was an unimplemented definition);
    · Theorem 3.1 `k_odd_no_odd_cycle` — same-K odd chains cannot close;
    · Theorem 3.2 `k_even_no_edges` — even K admits no edge at all, via
      ‖d·p‖² = 2K + 2·cross with cross always even;
    · `sedenion_zd_surgery_bipartite` — fixed-`p` assembly: no odd closed walk;
    · `sedenion_zd_surgery_bipartite_union` — §8, the UNION statement: no odd
      closed walk even when every edge carries its OWN primitive.

  THE MONOCHROMATIC GAP IS CLOSED (2026-08-17). An earlier revision of this
  header warned that the main statement fixed one `p`, so bipartiteness of the
  union graph over all 84 primitives did not follow — a union of bipartite
  graphs need not be bipartite. That warning was correct about the STATEMENT
  and wrong about the PROOF. `p` is consumed in exactly one place, the per-edge
  appeal to `k_even_no_edges`; the parity contradiction that does the work
  (`mixed_odd_no_odd_cycle` with `closed_walk_coordSum_zero`) takes no `ZDPrim`
  at all. Letting the primitive vary per edge therefore costs nothing, and §8
  proves it. The fixed-`p` theorem is now a one-line corollary
  (`sedenion_zd_surgery_bipartite_of_union`).

  The union statement is neither vacuous nor a restatement, and both objections
  are answered by kernel computation rather than assertion (§8.1):
    · `edge_exists` — a K=1 difference is an edge, so the relation is inhabited;
    · `edge_separates` — support {0,2,13} is an edge for ⟨3,12,−⟩ and NOT for
      ⟨1,10,+⟩, so per-primitive graphs genuinely differ and the union strictly
      contains any one of them. Measured over all 560 three-element all-`+1`
      supports, those two primitives disagree on 56 and agree on none: their
      K=3 edge sets are disjoint.

  SCOPE — one narrowing remains, and it is real.
  (b) The edge hypothesis is `zdEdge ∧ ∃ K, isKDiff K (u − v)`, i.e. only
      differences with unit coordinates and an exact support. The full
      `zdEdge` graph over unrestricted `SedVec` — any difference at all
      with ‖d·p‖² = 2 — is NOT covered. Theorems 3.1 and 3.2 both assume
      `isKDiff`, so this is a genuine restriction of the graph, not just
      of the proof. This is now the only gap between what is proved here and
      the paper's χ=2 headline.

  Mathlib-free. `twistedNormSq` was originally an unimplemented definition,
  not an unproved lemma. The formula is the Cayley–Dickson
  right product against a primitive two-support ZD element
    p = e_lo ± e_hi,
  matching `twCoeff` / `twNormSq` in `SounioErdosUnitDistance.lean` and
  Appendix C of `papers/sedenion-chromatic-gap/paper.md`:

    (d · p)_j = d(j⊕lo)·σ(j⊕lo, lo) + δ·d(j⊕hi)·σ(j⊕hi, hi)
    twistedNormSq(d, p) = Σ_j (d · p)_j²

  σ is the level-4 Cayley–Dickson sign (same recursion as
  `SounioSeamFlip.cdSigma` / `SounioCayleyDickson.cdSigma` / compiler
  `cd_sigma_ct`).

  SOUNIO CENSUS (2026-05-27). UNVERIFIED SIGN REDUCTION — read before
  quoting. The census enumerates supports × 84 prims, NOT the 2^K signed
  patterns per support. A global sign flip is indeed harmless (squares),
  but `cross = Σ_j A_j·B_j` depends on the RELATIVE signs of the live
  coordinates, so unsigned supports do not by themselves cover every
  `isKDiff`. The claimed parity reduction closing that gap is asserted
  externally and is neither stated nor checked in this file; treat the
  zero-edge rows as evidence about supports, not as a proof over `isKDiff`.
  Falsifier owed: exhibit the reduction, or a signed even-K counterexample.
  NOTE this debt is the census's, not the proofs'. Theorem 3.2 and the main
  theorem quantify over ALL sign patterns allowed by `isKDiff`, so they are
  strictly stronger than the census rows and do not rest on them.
    K=1: always-edge (hypercube); bipartite by coord-sum parity
    K=2: 0 edges (even-K parity-of-coincidences; now PROVED, Thm 3.2)
    K=3: edges exist; bipartite by Theorem 3.1 (same-K odd cycle),
         not by triangle-freeness (C₅ is a counterexample to that arrow)
    K=4: 0 edges — C(16,4)=1820 × 84 = 152,880 support×prim checks
    K=6: 0 edges — C(16,6)=8008 × 84 = 672,672 support×prim checks
  Even K ≥ 8 is not in the census; the ∀K claim is the parity lemma.

  The Euclidean orbit graph (Probe C) is a different metric and may have
  χ≥3; this file is only about the ZD-surgery twisted norm.
-/
set_option linter.unusedSimpArgs false

namespace Sounio.SedenionBipartite

/-! ## §0. Cayley–Dickson sign (SeamFlip recursion, reused for σ² = 1) -/

/-- Recursive CD sign. Identical case-split to `SounioSeamFlip.cdSigma`. -/
def cdSigma (a b : Nat) : Nat → Int
  | 0 => -1
  | 1 => if a == 0 || b == 0 then 1 else -1
  | (n+2) =>
      if a == 0 || b == 0 then 1
      else
        let half := 2 ^ (n+1)
        if !(a ≥ half) && !(b ≥ half) then cdSigma (a%half) (b%half) (n+1)
        else if !(a ≥ half) && (b ≥ half) then cdSigma (b%half) (a%half) (n+1)
        else if (a ≥ half) && !(b ≥ half) then
          (if b%half == 0 then cdSigma (a%half) 0 (n+1) else - cdSigma (a%half) (b%half) (n+1))
        else
          (if b%half == 0 then - cdSigma 0 (a%half) (n+1) else cdSigma (b%half) (a%half) (n+1))

/-- Sedenion sign: e_a · e_b = sedSigma(a,b) · e_{a⊕b}. -/
def sedSigma (a b : Nat) : Int := cdSigma a b 4

/-- Every `cdSigma` value is +1 or -1. Copied from `SounioSeamFlip.cdSigma_pm`. -/
theorem cdSigma_pm : ∀ (m a b : Nat), cdSigma a b m = 1 ∨ cdSigma a b m = -1 := by
  intro m
  induction m with
  | zero => exact fun a b => Or.inr rfl
  | succ k ih =>
    match k, ih with
    | 0, _ => intro a b; unfold cdSigma; by_cases h : a == 0 || b == 0 <;> simp [h]
    | (n+1), ih =>
      intro a b
      unfold cdSigma
      by_cases h : a == 0 || b == 0
      · simp [h]
      · by_cases ha : 2^(n+1) ≤ a <;> by_cases hb : 2^(n+1) ≤ b <;>
          simp only [h, ha, hb, Bool.false_eq_true, if_false, decide_true, decide_false,
            Bool.not_true, Bool.not_false, Bool.and_self, Bool.and_true,
            Bool.and_false, if_true]
        · by_cases hb0 : b % 2^(n+1) == 0 <;> simp only [hb0, if_true, if_false]
          · rcases ih 0 (a % 2^(n+1)) with hh | hh <;> simp [hh]
          · exact ih _ _
        · by_cases hb0 : b % 2^(n+1) == 0 <;> simp only [hb0, if_true, if_false]
          · exact ih _ _
          · rcases ih (a % 2^(n+1)) (b % 2^(n+1)) with hh | hh <;> simp [hh]
        · exact ih _ _
        · exact ih _ _

theorem cdSq (a b m : Nat) : cdSigma a b m * cdSigma a b m = 1 := by
  rcases cdSigma_pm m a b with h | h <;> rw [h] <;> decide

theorem sedSq (a b : Nat) : sedSigma a b * sedSigma a b = 1 := cdSq a b 4

/-! ## §1. Integer sedenion vectors and K-component diffs -/

/-- Coordinate function ℕ → ℤ; only indices `< 16` are live. -/
abbrev SedVec := Nat → Int

def isUnitCoord (z : Int) : Prop := z = 0 ∨ z = 1 ∨ z = -1

def nzB (z : Int) : Bool := decide (z ≠ 0)

def suppCount (d : SedVec) : Nat :=
  (List.range 16).filter (fun i => nzB (d i)) |>.length

/-- Exactly `K` live coordinates, each `0` or `±1`, and silence past index 15. -/
def isKDiff (K : Nat) (d : SedVec) : Prop :=
  suppCount d = K ∧
  (∀ i, i < 16 → isUnitCoord (d i)) ∧
  (∀ i, 16 ≤ i → d i = 0)

/-- Primitive ZD surgery element `e_lo ± e_hi` with `lo⊕hi ≠ 8`. -/
structure ZDPrim where
  lo : Nat
  hi : Nat
  neg : Bool
  lo_range : 1 ≤ lo ∧ lo ≤ 7
  hi_range : 9 ≤ hi ∧ hi ≤ 15
  no_zd : lo ^^^ hi ≠ 8

def primSign (p : ZDPrim) : Int := if p.neg then -1 else 1

theorem primSign_pm (p : ZDPrim) : primSign p = 1 ∨ primSign p = -1 := by
  unfold primSign
  cases p.neg <;> simp

theorem primSign_sq (p : ZDPrim) : primSign p * primSign p = 1 := by
  rcases primSign_pm p with h | h <;> rw [h] <;> decide

theorem ZDPrim.lo_lt_16 (p : ZDPrim) : p.lo < 16 :=
  Nat.lt_of_le_of_lt p.lo_range.2 (by decide)

theorem ZDPrim.hi_lt_16 (p : ZDPrim) : p.hi < 16 :=
  Nat.lt_of_le_of_lt p.hi_range.2 (by decide)

theorem ZDPrim.lo_ne_hi (p : ZDPrim) : p.lo ≠ p.hi := by
  intro h
  have := p.lo_range.2
  have := p.hi_range.1
  omega

/-! ## §2. Twisted norm — the missing definition -/

/-- Component `j` of the right product `d · p`.
    `(x · e_k)_j = x_{j⊕k} · σ(j⊕k, k)`; `p` contributes two basis terms. -/
def twCoeff (d : SedVec) (p : ZDPrim) (j : Nat) : Int :=
  d (j ^^^ p.lo) * sedSigma (j ^^^ p.lo) p.lo
    + primSign p * (d (j ^^^ p.hi) * sedSigma (j ^^^ p.hi) p.hi)

/-- Twisted squared norm `‖d · p‖²`. Computable; no opacity. -/
def twistedNormSq (d : SedVec) (p : ZDPrim) : Int :=
  (List.range 16).foldl (fun acc j => acc + twCoeff d p j * twCoeff d p j) 0

/-- ZD-surgery unit-distance edge: twisted norm equals `‖p‖² = 2`. -/
def zdEdge (p : ZDPrim) (u v : SedVec) : Prop :=
  twistedNormSq (fun i => u i - v i) p = 2

def coeffA (d : SedVec) (p : ZDPrim) (j : Nat) : Int :=
  d (j ^^^ p.lo) * sedSigma (j ^^^ p.lo) p.lo

def coeffB (d : SedVec) (p : ZDPrim) (j : Nat) : Int :=
  primSign p * (d (j ^^^ p.hi) * sedSigma (j ^^^ p.hi) p.hi)

theorem twCoeff_add (d : SedVec) (p : ZDPrim) (j : Nat) :
    twCoeff d p j = coeffA d p j + coeffB d p j := rfl

/-- Ordered coincidence count: slots `j` where both CD terms land. -/
def coinCount (d : SedVec) (p : ZDPrim) : Nat :=
  (List.range 16).filter (fun j => nzB (d (j ^^^ p.lo)) && nzB (d (j ^^^ p.hi))) |>.length

/-! ## §3. Integer / list arithmetic (Mathlib-free) -/

theorem int_sq_add (a b : Int) :
    (a + b) * (a + b) = a * a + b * b + 2 * (a * b) := by
  calc
    (a + b) * (a + b)
        = a * (a + b) + b * (a + b) := Int.add_mul a b (a + b)
    _ = a * a + a * b + (b * a + b * b) := by rw [Int.mul_add, Int.mul_add]
    _ = a * a + a * b + (a * b + b * b) := by rw [Int.mul_comm b a]
    _ = a * a + b * b + (a * b + a * b) := by ac_rfl
    _ = a * a + b * b + 2 * (a * b) := by rw [← Int.two_mul]

theorem unit_sq {z : Int} (h : isUnitCoord z) : z * z = if z = 0 then 0 else 1 := by
  rcases h with h | h | h <;> simp [h]

theorem unit_nz_pm {z : Int} (h : isUnitCoord z) (hnz : z ≠ 0) : z = 1 ∨ z = -1 := by
  rcases h with h | h | h
  · exact absurd h hnz
  · exact Or.inl h
  · exact Or.inr h

theorem foldl_add_init (l : List Nat) (g : Nat → Int) (init : Int) :
    l.foldl (fun acc i => acc + g i) init = init + l.foldl (fun acc i => acc + g i) 0 := by
  induction l generalizing init with
  | nil => simp
  | cons i l ih =>
    simp only [List.foldl, show (0 : Int) + g i = g i from by simp]
    calc List.foldl (fun acc i => acc + g i) init (i :: l)
        = List.foldl (fun acc i => acc + g i) (init + g i) l := rfl
      _ = init + g i + List.foldl (fun acc i => acc + g i) 0 l := ih (init + g i)
      _ = init + List.foldl (fun acc i => acc + g i) (g i) l := by
          rw [Int.add_assoc]
          congr 1
          exact (ih (g i)).symm

theorem foldl_add_init_nat (l : List Nat) (g : Nat → Nat) (init : Nat) :
    l.foldl (fun acc i => acc + g i) init = init + l.foldl (fun acc i => acc + g i) 0 := by
  induction l generalizing init with
  | nil => simp
  | cons i l ih =>
    simp [List.foldl]
    rw [ih (init + g i), ih (g i)]
    omega

theorem foldl_add_split (l : List Nat) (f g : Nat → Int) :
    l.foldl (fun acc i => acc + (f i + g i)) (0 : Int) =
      l.foldl (fun acc i => acc + f i) 0 + l.foldl (fun acc i => acc + g i) 0 := by
  induction l with
  | nil => simp
  | cons i l ih =>
    simp only [List.foldl, show (0 : Int) + (f i + g i) = f i + g i from by simp,
               show (0 : Int) + f i = f i from by simp, show (0 : Int) + g i = g i from by simp]
    rw [foldl_add_init l (fun j => f j + g j) (f i + g i),
        foldl_add_init l f (f i), foldl_add_init l g (g i), ih]
    omega

theorem foldl_smul_split (l : List Nat) (c : Int) (f : Nat → Int) :
    l.foldl (fun acc i => acc + c * f i) (0 : Int) =
      c * l.foldl (fun acc i => acc + f i) 0 := by
  induction l with
  | nil => simp
  | cons i l ih =>
    simp only [List.foldl, show (0 : Int) + c * f i = c * f i from by simp,
               show (0 : Int) + f i = f i from by simp]
    rw [foldl_add_init l (fun j => c * f j) (c * f i), ih, ← Int.mul_add,
        foldl_add_init l f (f i)]

theorem foldl_add_pointwise {α} (l : List α) (f g : α → Int) (init : Int)
    (h : ∀ a, a ∈ l → f a = g a) :
    l.foldl (fun acc a => acc + f a) init = l.foldl (fun acc a => acc + g a) init := by
  induction l generalizing init with
  | nil => rfl
  | cons a l ih =>
    simp only [List.foldl]
    rw [h a (by simp)]
    exact ih _ fun b hb => h b (by simp [hb])

def iSum (l : List Nat) (f : Nat → Int) : Int :=
  l.foldl (fun acc i => acc + f i) 0

def nSum (l : List Nat) (f : Nat → Nat) : Nat :=
  l.foldl (fun acc i => acc + f i) 0

theorem iSum_cons (i : Nat) (l : List Nat) (f : Nat → Int) :
    iSum (i :: l) f = f i + iSum l f := by
  simp [iSum, List.foldl, show (0 : Int) + f i = f i from by simp]
  exact foldl_add_init l f (f i)

theorem nSum_cons (i : Nat) (l : List Nat) (f : Nat → Nat) :
    nSum (i :: l) f = f i + nSum l f := by
  simp [nSum, List.foldl]
  exact foldl_add_init_nat l f (f i)

theorem nSum_nil (f : Nat → Nat) : nSum [] f = 0 := rfl

theorem iSum_nil (f : Nat → Int) : iSum [] f = 0 := rfl

theorem iSum_congr (l : List Nat) (f g : Nat → Int) (h : ∀ a, a ∈ l → f a = g a) :
    iSum l f = iSum l g :=
  foldl_add_pointwise l f g 0 h

/-- Permutation invariance of `iSum`, by induction on the `List.Perm`
    DERIVATION rather than on the list.  That is the whole point: the list is
    never destructed, so a concrete `List.range 16` stays opaque and the `whnf`
    blow-up that killed the earlier `sumA_sq_eq_K` attempt cannot arise. -/
theorem iSum_perm {l₁ l₂ : List Nat} (f : Nat → Int) (h : List.Perm l₁ l₂) :
    iSum l₁ f = iSum l₂ f := by
  induction h with
  | nil => rfl
  | cons x _ ih => rw [iSum_cons, iSum_cons, ih]
  | swap x y l => rw [iSum_cons, iSum_cons, iSum_cons, iSum_cons]; omega
  | trans _ _ ih₁ ih₂ => rw [ih₁, ih₂]

theorem iSum_map (l : List Nat) (g : Nat → Nat) (f : Nat → Int) :
    iSum (l.map g) f = iSum l (fun i => f (g i)) := by
  induction l with
  | nil => rfl
  | cons x l ih => rw [List.map_cons, iSum_cons, iSum_cons, ih]

/-- Re-index a sum along `g` whenever `g` permutes the index list. -/
theorem iSum_reindex (l : List Nat) (g : Nat → Nat) (f : Nat → Int)
    (hp : List.Perm (l.map g) l) :
    iSum l (fun j => f (g j)) = iSum l f := by
  rw [← iSum_map l g f]
  exact iSum_perm f hp

theorem iSum_ofNat (l : List Nat) (f : Nat → Nat) :
    iSum l (fun i => (f i : Int)) = (nSum l f : Int) := by
  induction l with
  | nil => rfl
  | cons x l ih => rw [iSum_cons, nSum_cons, ih]; omega

theorem nSum_add (l : List Nat) (f g : Nat → Nat) :
    nSum l (fun x => f x + g x) = nSum l f + nSum l g := by
  induction l with
  | nil => simp [nSum]
  | cons x l ih =>
    rw [nSum_cons, nSum_cons, nSum_cons, ih]
    omega

theorem nSum_append (l₁ l₂ : List Nat) (f : Nat → Nat) :
    nSum (l₁ ++ l₂) f = nSum l₁ f + nSum l₂ f := by
  induction l₁ with
  | nil => simp [nSum]
  | cons x l ih =>
    simp [nSum_cons, ih]
    omega

theorem range_succ_eq (n : Nat) : List.range (n + 1) = List.range n ++ [n] :=
  List.range_succ

theorem range_zero_eq : List.range 0 = [] := by decide

theorem iSum_range_succ (n : Nat) (f : Nat → Int) :
    iSum (List.range (n + 1)) f = iSum (List.range n) f + f n := by
  rw [range_succ_eq, iSum, List.foldl_append]
  simp [List.foldl]
  rw [foldl_add_init]
  simp [iSum]

theorem nSum_range_succ (n : Nat) (f : Nat → Nat) :
    nSum (List.range (n + 1)) f = nSum (List.range n) f + f n := by
  rw [range_succ_eq, nSum_append]
  simp [nSum, List.foldl]

theorem nSum_const (n K : Nat) : nSum (List.range n) (fun _ => K) = n * K := by
  induction n with
  | zero => simp [nSum, range_zero_eq]
  | succ n ih =>
    rw [nSum_range_succ, ih, Nat.succ_mul]

theorem filter_len_append (l₁ l₂ : List Nat) (p : Nat → Bool) :
    ((l₁ ++ l₂).filter p).length = (l₁.filter p).length + (l₂.filter p).length := by
  simp [List.filter_append]

theorem filter_singleton (n : Nat) (p : Nat → Bool) :
    ([n].filter p).length = if p n then 1 else 0 := by
  cases h : p n <;> simp [List.filter, h]

theorem filter_length_nSum (l : List Nat) (p : Nat → Bool) :
    (l.filter p).length = nSum l (fun x => if p x then 1 else 0) := by
  induction l with
  | nil => simp [nSum]
  | cons x l ih =>
    cases h : p x <;> simp [List.filter, h, nSum_cons, ih] <;> try omega

theorem odd_mul_odd {a b : Nat} (ha : a % 2 = 1) (hb : b % 2 = 1) :
    (a * b) % 2 = 1 := by
  simp [Nat.mul_mod, ha, hb]

theorem even_sum_of_evens (l : List Nat) (f : Nat → Nat)
    (h : ∀ x, x ∈ l → f x % 2 = 0) : nSum l f % 2 = 0 := by
  induction l with
  | nil => simp [nSum]
  | cons x l ih =>
    rw [nSum_cons, Nat.add_mod, h x (by simp),
        ih (fun y hy => h y (by simp [hy]))]

/-! ## §4. XOR on the 4-bit cube -/

theorem xor_lt_16 {i idx : Nat} (hi : i < 16) (hidx : idx < 16) : i ^^^ idx < 16 := by
  match idx with
  | 0 => simpa [Nat.zero_xor] using hi
  | 1 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by omega)
  | 2 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by omega)
  | 3 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by omega)
  | 4 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by omega)
  | 5 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by omega)
  | 6 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by omega)
  | 7 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by omega)
  | 8 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by omega)
  | 9 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by omega)
  | 10 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by omega)
  | 11 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by omega)
  | 12 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by omega)
  | 13 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by omega)
  | 14 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by omega)
  | 15 => revert hi; match i with | 0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15 => decide | _+16 => intro h; exact absurd h (by omega)
  | idx + 16 => exact absurd hidx (by omega)

/-- 4-bit XOR permutes `{0,…,15}`. Finite identity, discharged by kernel `decide`
    on each of the 16 closed instances. No `native_decide`, no extra axioms. -/
theorem perm_range_xor (idx : Nat) (hidx : idx < 16) :
    List.Perm ((List.range 16).map (fun i => i ^^^ idx)) (List.range 16) := by
  match idx with
  | 0 => decide
  | 1 => decide
  | 2 => decide
  | 3 => decide
  | 4 => decide
  | 5 => decide
  | 6 => decide
  | 7 => decide
  | 8 => decide
  | 9 => decide
  | 10 => decide
  | 11 => decide
  | 12 => decide
  | 13 => decide
  | 14 => decide
  | 15 => decide
  | idx + 16 => omega

/-! ## §5. Theorem 1 — odd K forbids an odd closed chain of K-diffs -/

def coordHits (ds : Nat → SedVec) (n j : Nat) : Nat :=
  (List.range n).filter (fun i => nzB (ds i j)) |>.length

def coordSum (ds : Nat → SedVec) (n j : Nat) : Int :=
  iSum (List.range n) (fun i => ds i j)

theorem nplus_nminus_eq_nz (n : Nat) (f : Nat → Int)
    (h : ∀ i, i < n → isUnitCoord (f i)) :
    ((List.range n).filter (fun i => decide (f i = 1))).length
      + ((List.range n).filter (fun i => decide (f i = -1))).length
    = ((List.range n).filter (fun i => nzB (f i))).length := by
  induction n with
  | zero => simp [range_zero_eq]
  | succ n ih =>
    have hun : isUnitCoord (f n) := h n (Nat.lt_succ_self n)
    have ih' := ih (fun i hi => h i (Nat.lt_trans hi (Nat.lt_succ_self n)))
    rw [range_succ_eq]
    simp only [filter_len_append, filter_singleton]
    rcases hun with hz | hp | hm
    · have h1 : decide (f n = 1) = false := by simp [hz]
      have hm1 : decide (f n = -1) = false := by simp [hz]
      have hnz : nzB (f n) = false := by simp [nzB, hz]
      simp [h1, hm1, hnz, ih']
    · have h1 : decide (f n = 1) = true := by simp [hp]
      have hm1 : decide (f n = -1) = false := by simp [hp]
      have hnz : nzB (f n) = true := by simp [nzB, hp]
      simp [h1, hm1, hnz, ih']; omega
    · have h1 : decide (f n = 1) = false := by simp [hm]
      have hm1 : decide (f n = -1) = true := by simp [hm]
      have hnz : nzB (f n) = true := by simp [nzB, hm]
      simp [h1, hm1, hnz, ih']; omega

theorem iSum_eq_plus_minus (n : Nat) (f : Nat → Int)
    (h : ∀ i, i < n → isUnitCoord (f i)) :
    iSum (List.range n) f
      = (((List.range n).filter (fun i => decide (f i = 1))).length : Int)
        - (((List.range n).filter (fun i => decide (f i = -1))).length : Int) := by
  induction n with
  | zero => simp [iSum, range_zero_eq]
  | succ n ih =>
    have hun : isUnitCoord (f n) := h n (Nat.lt_succ_self n)
    have ih' := ih (fun i hi => h i (Nat.lt_trans hi (Nat.lt_succ_self n)))
    rw [iSum_range_succ, ih', range_succ_eq]
    simp only [filter_len_append, filter_singleton]
    rcases hun with hz | hp | hm
    · simp [hz]
    · simp [hp]; omega
    · simp [hm]; omega

/-- A vanishing `{0,±1}`-sum has even support. -/
theorem unit_sum_zero_hits_even (n : Nat) (f : Nat → Int)
    (h : ∀ i, i < n → isUnitCoord (f i))
    (hsum : iSum (List.range n) f = 0) :
    ((List.range n).filter (fun i => nzB (f i))).length % 2 = 0 := by
  have hpm := iSum_eq_plus_minus n f h
  have hsplit := nplus_nminus_eq_nz n f h
  let np := ((List.range n).filter (fun i => decide (f i = 1))).length
  let nm := ((List.range n).filter (fun i => decide (f i = -1))).length
  have hdiff : (np : Int) - (nm : Int) = 0 := by
    simpa [np, nm] using (hpm.symm.trans hsum)
  have hnn : np = nm := Int.ofNat_inj.mp (by omega)
  have hhits : np + nm = ((List.range n).filter (fun i => nzB (f i))).length := hsplit
  rw [← hhits, hnn]
  change (nm + nm) % 2 = 0
  omega

theorem coordHits_succ (ds : Nat → SedVec) (n j : Nat) :
    coordHits ds (n + 1) j
      = coordHits ds n j + (if nzB (ds n j) then 1 else 0) := by
  simp [coordHits, range_succ_eq, filter_len_append, filter_singleton]

theorem nSum_pointwise (l : List Nat) (f g : Nat → Nat)
    (h : ∀ x, x ∈ l → f x = g x) : nSum l f = nSum l g := by
  induction l with
  | nil => rfl
  | cons x l ih =>
    rw [nSum_cons, nSum_cons, h x (by simp), ih (fun y hy => h y (by simp [hy]))]

theorem nSum_coordHits_succ (ds : Nat → SedVec) (n : Nat) :
    nSum (List.range 16) (fun j => coordHits ds (n + 1) j)
      = nSum (List.range 16) (fun j => coordHits ds n j)
        + nSum (List.range 16) (fun j => if nzB (ds n j) then 1 else 0) := by
  have hsplit := nSum_add (List.range 16)
    (fun j => coordHits ds n j)
    (fun j => if nzB (ds n j) then 1 else 0)
  rw [← hsplit]
  exact nSum_pointwise _ _ _ (fun j _ => coordHits_succ ds n j)

theorem suppCount_as_nSum (d : SedVec) :
    suppCount d = nSum (List.range 16) (fun j => if nzB (d j) then 1 else 0) :=
  filter_length_nSum (List.range 16) (fun j => nzB (d j))

theorem double_count_hits (n : Nat) (ds : Nat → SedVec) :
    nSum (List.range 16) (fun j => coordHits ds n j)
      = nSum (List.range n) (fun i => suppCount (ds i)) := by
  induction n with
  | zero =>
    simp [nSum, coordHits, range_zero_eq]
    induction (List.range 16) with
    | nil => rfl
    | cons j l ih =>
      simp [nSum_cons, coordHits, range_zero_eq] at ih ⊢
      exact ih
  | succ n ih =>
    have h1 := nSum_coordHits_succ ds n
    have h2 := (suppCount_as_nSum (ds n)).symm
    have h3 := nSum_range_succ n (fun i => suppCount (ds i))
    calc
      nSum (List.range 16) (fun j => coordHits ds (n + 1) j)
          = nSum (List.range 16) (fun j => coordHits ds n j)
              + nSum (List.range 16) (fun j => if nzB (ds n j) then 1 else 0) := h1
      _ = nSum (List.range n) (fun i => suppCount (ds i))
              + nSum (List.range 16) (fun j => if nzB (ds n j) then 1 else 0) := by rw [ih]
      _ = nSum (List.range n) (fun i => suppCount (ds i)) + suppCount (ds n) := by rw [h2]
      _ = nSum (List.range (n + 1)) (fun i => suppCount (ds i)) := h3.symm

theorem nSum_supp_const (n K : Nat) (ds : Nat → SedVec)
    (h : ∀ i, i < n → suppCount (ds i) = K) :
    nSum (List.range n) (fun i => suppCount (ds i)) = n * K := by
  induction n with
  | zero => simp [nSum, range_zero_eq]
  | succ n ih =>
    rw [nSum_range_succ, ih (fun i hi => h i (Nat.lt_trans hi (Nat.lt_succ_self n))),
        h n (Nat.lt_succ_self n), Nat.succ_mul]

/-- Paper Theorem 3.1: an odd number of odd-K unit diffs cannot sum to 0. -/
theorem k_odd_no_odd_cycle (K n : Nat) (hK : K % 2 = 1) (hn : n % 2 = 1)
    (ds : Nat → SedVec) (hds : ∀ i, i < n → isKDiff K (ds i)) :
    ¬ (∀ j, j < 16 → coordSum ds n j = 0) := by
  intro hzero
  have hevenj : ∀ j, j < 16 → coordHits ds n j % 2 = 0 := by
    intro j hj
    have hsum : iSum (List.range n) (fun i => ds i j) = 0 := hzero j hj
    exact unit_sum_zero_hits_even n (fun i => ds i j)
      (fun i hi => (hds i hi).2.1 j hj) hsum
  have htot_even : nSum (List.range 16) (fun j => coordHits ds n j) % 2 = 0 :=
    even_sum_of_evens (List.range 16) (fun j => coordHits ds n j) (fun j hj =>
      hevenj j (List.mem_range.mp hj))
  have htot_eq : nSum (List.range 16) (fun j => coordHits ds n j) = n * K := by
    rw [double_count_hits]
    exact nSum_supp_const n K ds (fun i hi => (hds i hi).1)
  have hodd : (n * K) % 2 = 1 := odd_mul_odd hn hK
  rw [htot_eq] at htot_even
  omega

/-! ## §6. Theorem 2 — even K forbids twistedNormSq = 2 -/

theorem coeffA_sq (d : SedVec) (p : ZDPrim) (j : Nat) :
    coeffA d p j * coeffA d p j = d (j ^^^ p.lo) * d (j ^^^ p.lo) := by
  unfold coeffA
  have hs : sedSigma (j ^^^ p.lo) p.lo * sedSigma (j ^^^ p.lo) p.lo = 1 := sedSq _ _
  calc
    (d (j ^^^ p.lo) * sedSigma (j ^^^ p.lo) p.lo)
        * (d (j ^^^ p.lo) * sedSigma (j ^^^ p.lo) p.lo)
      = (d (j ^^^ p.lo) * d (j ^^^ p.lo))
          * (sedSigma (j ^^^ p.lo) p.lo * sedSigma (j ^^^ p.lo) p.lo) := by ac_rfl
    _ = d (j ^^^ p.lo) * d (j ^^^ p.lo) * 1 := by rw [hs]
    _ = d (j ^^^ p.lo) * d (j ^^^ p.lo) := by simp

theorem coeffB_sq (d : SedVec) (p : ZDPrim) (j : Nat) :
    coeffB d p j * coeffB d p j = d (j ^^^ p.hi) * d (j ^^^ p.hi) := by
  unfold coeffB
  have hs : sedSigma (j ^^^ p.hi) p.hi * sedSigma (j ^^^ p.hi) p.hi = 1 := sedSq _ _
  have hp : primSign p * primSign p = 1 := primSign_sq p
  calc
    (primSign p * (d (j ^^^ p.hi) * sedSigma (j ^^^ p.hi) p.hi))
        * (primSign p * (d (j ^^^ p.hi) * sedSigma (j ^^^ p.hi) p.hi))
      = (primSign p * primSign p)
          * (d (j ^^^ p.hi) * d (j ^^^ p.hi))
          * (sedSigma (j ^^^ p.hi) p.hi * sedSigma (j ^^^ p.hi) p.hi) := by ac_rfl
    _ = 1 * (d (j ^^^ p.hi) * d (j ^^^ p.hi)) * 1 := by rw [hp, hs]
    _ = d (j ^^^ p.hi) * d (j ^^^ p.hi) := by simp

/-- For unit coordinates the square-sum is exactly the support count. -/
theorem sum_sq_eq_suppCount (d : SedVec) (hu : ∀ i, i < 16 → isUnitCoord (d i)) :
    iSum (List.range 16) (fun j => d j * d j) = (suppCount d : Int) := by
  have hpt : ∀ a, a ∈ List.range 16 →
      d a * d a = ((if nzB (d a) then 1 else 0 : Nat) : Int) := by
    intro a ha
    have ha16 : a < 16 := List.mem_range.mp ha
    rw [unit_sq (hu a ha16)]
    by_cases h : d a = 0 <;> simp [h, nzB]
  rw [iSum_congr _ _ _ hpt, iSum_ofNat, suppCount, filter_length_nSum]

/-- Obligation 1 of Theorem 3.2, `lo` half: the A-square sum is `K`. -/
theorem sumA_sq_eq_K (K : Nat) (d : SedVec) (p : ZDPrim) (hd : isKDiff K d) :
    iSum (List.range 16) (fun j => coeffA d p j * coeffA d p j) = (K : Int) := by
  have hfun : (fun j => coeffA d p j * coeffA d p j)
      = (fun j => d (j ^^^ p.lo) * d (j ^^^ p.lo)) := funext (fun j => coeffA_sq d p j)
  rw [hfun,
    iSum_reindex (List.range 16) (fun j => j ^^^ p.lo) (fun z => d z * d z)
      (perm_range_xor p.lo p.lo_lt_16),
    sum_sq_eq_suppCount d hd.2.1, hd.1]

/-- Obligation 1 of Theorem 3.2, `hi` half: the B-square sum is also `K`. -/
theorem sumB_sq_eq_K (K : Nat) (d : SedVec) (p : ZDPrim) (hd : isKDiff K d) :
    iSum (List.range 16) (fun j => coeffB d p j * coeffB d p j) = (K : Int) := by
  have hfun : (fun j => coeffB d p j * coeffB d p j)
      = (fun j => d (j ^^^ p.hi) * d (j ^^^ p.hi)) := funext (fun j => coeffB_sq d p j)
  rw [hfun,
    iSum_reindex (List.range 16) (fun j => j ^^^ p.hi) (fun z => d z * d z)
      (perm_range_xor p.hi p.hi_lt_16),
    sum_sq_eq_suppCount d hd.2.1, hd.1]

theorem twistedNormSq_expand (d : SedVec) (p : ZDPrim) :
    twistedNormSq d p
      = iSum (List.range 16) (fun j => coeffA d p j * coeffA d p j)
        + iSum (List.range 16) (fun j => coeffB d p j * coeffB d p j)
        + 2 * iSum (List.range 16) (fun j => coeffA d p j * coeffB d p j) := by
  unfold twistedNormSq iSum
  have hpt : ∀ j, j ∈ List.range 16 →
      twCoeff d p j * twCoeff d p j
        = coeffA d p j * coeffA d p j
          + coeffB d p j * coeffB d p j
          + 2 * (coeffA d p j * coeffB d p j) := by
    intro j _
    rw [twCoeff_add, int_sq_add]
  have h1 := foldl_add_pointwise (List.range 16)
    (fun j => twCoeff d p j * twCoeff d p j)
    (fun j => coeffA d p j * coeffA d p j
              + coeffB d p j * coeffB d p j
              + 2 * (coeffA d p j * coeffB d p j))
    0 hpt
  rw [h1]
  have hsplit := foldl_add_split (List.range 16)
    (fun j => coeffA d p j * coeffA d p j + coeffB d p j * coeffB d p j)
    (fun j => 2 * (coeffA d p j * coeffB d p j))
  rw [hsplit, foldl_add_split, foldl_smul_split]

/-! ### Obligations 2 and 3 — the coincidence involution and cross parity -/

theorem sedSigma_pm (a b : Nat) : sedSigma a b = 1 ∨ sedSigma a b = -1 :=
  cdSigma_pm 4 a b

theorem iSum_mod2_congr (l : List Nat) (f g : Nat → Int)
    (h : ∀ a, a ∈ l → f a % 2 = g a % 2) : iSum l f % 2 = iSum l g % 2 := by
  induction l with
  | nil => rfl
  | cons x l ih =>
    rw [iSum_cons, iSum_cons]
    have hx := h x (by simp)
    have hr := ih (fun a ha => h a (by simp [ha]))
    omega

theorem xor_undo (a b : Nat) : (a ^^^ b) ^^^ b = a := by
  rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]

theorem xor_mid (a b : Nat) : (a ^^^ b) ^^^ a = b := by
  rw [Nat.xor_comm a b, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]

/-- The surgery displacement `t = lo ⊕ hi`; the coincidence involution is `j ↦ j⊕t`. -/
def zdT (p : ZDPrim) : Nat := p.lo ^^^ p.hi

theorem zdT_ne_zero (p : ZDPrim) : zdT p ≠ 0 := by
  intro h
  apply p.lo_ne_hi
  have hu := xor_undo p.lo p.hi
  unfold zdT at h
  rw [h, Nat.zero_xor] at hu
  exact hu.symm

theorem zdT_lt_16 (p : ZDPrim) : zdT p < 16 :=
  xor_lt_16 p.lo_lt_16 p.hi_lt_16

theorem xor_ne_self {t : Nat} (ht : t ≠ 0) (j : Nat) : j ^^^ t ≠ j := by
  intro hc
  apply ht
  have h2 : (j ^^^ t) ^^^ j = t := by
    rw [Nat.xor_comm j t]
    exact xor_undo t j
  rw [hc, Nat.xor_self] at h2
  exact h2.symm

theorem nSum_perm {l₁ l₂ : List Nat} (f : Nat → Nat) (h : List.Perm l₁ l₂) :
    nSum l₁ f = nSum l₂ f := by
  induction h with
  | nil => rfl
  | cons x _ ih => rw [nSum_cons, nSum_cons, ih]
  | swap x y l => rw [nSum_cons, nSum_cons, nSum_cons, nSum_cons]; omega
  | trans _ _ ih₁ ih₂ => rw [ih₁, ih₂]

theorem nSum_map (l : List Nat) (g f : Nat → Nat) :
    nSum (l.map g) f = nSum l (fun i => f (g i)) := by
  induction l with
  | nil => rfl
  | cons x l ih => rw [List.map_cons, nSum_cons, nSum_cons, ih]

theorem nSum_reindex (l : List Nat) (g f : Nat → Nat)
    (hp : List.Perm (l.map g) l) :
    nSum l (fun j => f (g j)) = nSum l f := by
  rw [← nSum_map l g f]
  exact nSum_perm f hp

/-- The predicate `coinCount` filters on. -/
def coincides (d : SedVec) (p : ZDPrim) (j : Nat) : Bool :=
  nzB (d (j ^^^ p.lo)) && nzB (d (j ^^^ p.hi))

theorem coinCount_as_nSum (d : SedVec) (p : ZDPrim) :
    coinCount d p = nSum (List.range 16) (fun j => if coincides d p j then 1 else 0) := by
  rw [coinCount, filter_length_nSum]
  rfl

/-- The involution swaps the two coordinates the predicate looks at,
    so the coincidence set is closed under it. -/
theorem coincides_swap (d : SedVec) (p : ZDPrim) (j : Nat) :
    coincides d p (j ^^^ zdT p) = coincides d p j := by
  have h1 : (j ^^^ zdT p) ^^^ p.lo = j ^^^ p.hi := by
    unfold zdT
    rw [Nat.xor_assoc, xor_mid p.lo p.hi]
  have h2 : (j ^^^ zdT p) ^^^ p.hi = j ^^^ p.lo := by
    unfold zdT
    rw [Nat.xor_assoc, xor_undo p.lo p.hi]
  unfold coincides
  rw [h1, h2, Bool.and_comm]

/-- Coincidences on the low side of the involution. -/
def coinLo (d : SedVec) (p : ZDPrim) (j : Nat) : Nat :=
  if decide (j < j ^^^ zdT p) && coincides d p j then 1 else 0

/-- Coincidences on the high side of the involution. -/
def coinHi (d : SedVec) (p : ZDPrim) (j : Nat) : Nat :=
  if decide (j ^^^ zdT p < j) && coincides d p j then 1 else 0

/-- Obligation 2. The involution `j ↦ j⊕t` is fixed-point-free (`t ≠ 0`) and
    preserves the coincidence predicate, so it splits the coincidence set into
    two equinumerous halves. Sorting by `j < j⊕t` versus `j⊕t < j` gives the
    split without needing a bit-position argument. -/
theorem coinCount_even (d : SedVec) (p : ZDPrim) : coinCount d p % 2 = 0 := by
  have hsplit : coinCount d p
      = nSum (List.range 16) (coinLo d p) + nSum (List.range 16) (coinHi d p) := by
    rw [coinCount_as_nSum, ← nSum_add]
    refine nSum_pointwise _ _ _ (fun j _ => ?_)
    have hne : j ^^^ zdT p ≠ j := xor_ne_self (zdT_ne_zero p) j
    cases hc : coincides d p j
    · simp [coinLo, coinHi, hc]
    · rcases Nat.lt_or_ge j (j ^^^ zdT p) with h | h
      · have h2 : ¬ (j ^^^ zdT p < j) := by omega
        simp [coinLo, coinHi, hc, h, h2]
      · have h1 : j ^^^ zdT p < j := by omega
        have h2 : ¬ (j < j ^^^ zdT p) := by omega
        simp [coinLo, coinHi, hc, h1, h2]
  have hpt : ∀ j, coinLo d p j = coinHi d p (j ^^^ zdT p) := by
    intro j
    unfold coinLo coinHi
    rw [xor_undo j (zdT p), coincides_swap d p j]
  have hswap : nSum (List.range 16) (coinLo d p) = nSum (List.range 16) (coinHi d p) :=
    calc nSum (List.range 16) (coinLo d p)
        = nSum (List.range 16) (fun j => coinHi d p (j ^^^ zdT p)) :=
          nSum_pointwise _ _ _ (fun j _ => hpt j)
      _ = nSum (List.range 16) (coinHi d p) :=
          nSum_reindex _ _ _ (perm_range_xor (zdT p) (zdT_lt_16 p))
  rw [hsplit, hswap]
  omega

/-- Obligation 3. Every cross term is `0` or `±1`, and it is nonzero exactly on
    a coincidence, so the cross sum and the coincidence count agree mod 2. -/
theorem cross_mod_two (K : Nat) (d : SedVec) (p : ZDPrim) (hd : isKDiff K d) :
    iSum (List.range 16) (fun j => coeffA d p j * coeffB d p j) % 2 = 0 := by
  have hterm : ∀ j, j ∈ List.range 16 →
      (coeffA d p j * coeffB d p j) % 2
        = ((if coincides d p j then 1 else 0 : Nat) : Int) % 2 := by
    intro j hj
    have hj16 : j < 16 := List.mem_range.mp hj
    have hlo16 : j ^^^ p.lo < 16 := xor_lt_16 hj16 p.lo_lt_16
    have hhi16 : j ^^^ p.hi < 16 := xor_lt_16 hj16 p.hi_lt_16
    have hul := hd.2.1 _ hlo16
    have huh := hd.2.1 _ hhi16
    have hsl := sedSigma_pm (j ^^^ p.lo) p.lo
    have hsh := sedSigma_pm (j ^^^ p.hi) p.hi
    have hps := primSign_pm p
    unfold coeffA coeffB coincides nzB
    rcases hul with hl | hl | hl <;> rcases huh with hh | hh | hh <;>
      rcases hsl with sl | sl <;> rcases hsh with sh | sh <;>
      rcases hps with ps | ps <;>
      simp [hl, hh, sl, sh, ps]
  rw [iSum_mod2_congr _ _ _ hterm, iSum_ofNat, ← coinCount_as_nSum]
  have := coinCount_even d p
  omega

/-- Paper Theorem 3.2.

    PROVED. All three obligations are discharged, Mathlib-free:

    1. `twistedNormSq_expand` with `sumA_sq_eq_K` / `sumB_sq_eq_K` gives
       ‖d·p‖² = 2K + 2·cross, so an edge forces `K + cross = 1`. The former
       `whnf` blow-up here was beaten by proving permutation invariance on the
       `List.Perm` DERIVATION (`iSum_perm`), never destructing the list, and
       transporting along `perm_range_xor` via `iSum_reindex`.
    2. `coinCount_even`: the involution `j ↦ j⊕(lo⊕hi)` is fixed-point-free
       (`lo ≠ hi`) and preserves the coincidence predicate, so sorting by
       `j < j⊕t` versus `j⊕t < j` splits the coincidence set into two halves
       that `nSum_reindex` shows are equinumerous.
    3. `cross_mod_two`: each cross term is `0` or `±1`, nonzero exactly on a
       coincidence, so cross ≡ coinCount ≡ 0 (mod 2).

    Cross even plus `K + cross = 1` forces K odd. No `native_decide`: the
    statement is universally quantified in K and a fixed-K sweep would only
    re-check the Sounio census. -/
theorem k_even_no_edges (K : Nat) (hK : K % 2 = 0) (d : SedVec) (p : ZDPrim)
    (hd : isKDiff K d) : twistedNormSq d p ≠ 2 := by
  intro hEq
  rw [twistedNormSq_expand, sumA_sq_eq_K K d p hd, sumB_sq_eq_K K d p hd] at hEq
  have hc := cross_mod_two K d p hd
  omega

/-! ### §7a. Mixed-K refinement, telescoping, and the assembly -/

/-- Theorem 3.1 does not actually need a shared `K`: the double count only
    ever uses that `Σ_i K_i` is odd.  This is the same proof, restated. -/
theorem mixed_odd_no_odd_cycle (n : Nat) (ds : Nat → SedVec) (Ks : Nat → Nat)
    (hds : ∀ i, i < n → isKDiff (Ks i) (ds i))
    (hodd : nSum (List.range n) Ks % 2 = 1) :
    ¬ (∀ j, j < 16 → coordSum ds n j = 0) := by
  intro hzero
  have hevenj : ∀ j, j < 16 → coordHits ds n j % 2 = 0 := by
    intro j hj
    exact unit_sum_zero_hits_even n (fun i => ds i j)
      (fun i hi => (hds i hi).2.1 j hj) (hzero j hj)
  have htot_even : nSum (List.range 16) (fun j => coordHits ds n j) % 2 = 0 :=
    even_sum_of_evens (List.range 16) (fun j => coordHits ds n j)
      (fun j hj => hevenj j (List.mem_range.mp hj))
  have htot_eq : nSum (List.range 16) (fun j => coordHits ds n j)
      = nSum (List.range n) Ks := by
    rw [double_count_hits]
    exact nSum_pointwise _ _ _ (fun i hi => (hds i (List.mem_range.mp hi)).1)
  rw [htot_eq] at htot_even
  omega

/-- A sum of odd terms has the parity of the number of terms. -/
theorem nSum_odd_parity (l : List Nat) (f : Nat → Nat)
    (h : ∀ x, x ∈ l → f x % 2 = 1) : nSum l f % 2 = l.length % 2 := by
  induction l with
  | nil => rfl
  | cons x l ih =>
    rw [nSum_cons, List.length_cons]
    have hx := h x (by simp)
    have hr := ih (fun y hy => h y (by simp [hy]))
    omega

/-- Plain telescoping on an initial segment; no modular index yet. -/
theorem telescope (m : Nat) (g : Nat → Int) :
    iSum (List.range m) (fun i => g i - g (i + 1)) = g 0 - g m := by
  induction m with
  | zero => simp [iSum, range_zero_eq]
  | succ m ih => rw [iSum_range_succ, ih]; omega

/-- The closed walk sums to zero in every coordinate.  Splitting the last step
    off turns the modular index into a plain one, so no rotation-permutation
    of `List.range n` is needed. -/
theorem closed_walk_coordSum_zero (m : Nat) (cycle : Nat → SedVec) (j : Nat) :
    coordSum (fun i j => cycle i j - cycle ((i + 1) % (m + 1)) j) (m + 1) j = 0 := by
  unfold coordSum
  rw [iSum_range_succ]
  have hinner : iSum (List.range m)
      (fun i => cycle i j - cycle ((i + 1) % (m + 1)) j)
      = iSum (List.range m) (fun i => cycle i j - cycle (i + 1) j) := by
    refine iSum_congr _ _ _ (fun i hi => ?_)
    have hi' : i < m := List.mem_range.mp hi
    rw [Nat.mod_eq_of_lt (by omega)]
  rw [hinner, telescope]
  simp only [Nat.mod_self]
  omega

/-! ## §7. Main theorem — no odd cycle in the integer ZD-surgery graph

    Without Mathlib `SimpleGraph`, bipartiteness is the absence of an odd
    cycle.  Every edge is some K-diff with `twistedNormSq = 2`; even K is
    impossible, so every edge-K is odd; an odd cycle then contradicts
    Theorem 3.1 (or the mixed-K refinement: `Σ K_e` is odd yet equals an
    even hit-count).

    PROVED. Three pieces, all above:
    · `k_even_no_edges` forces every edge's K to be odd;
    · `mixed_odd_no_odd_cycle` handles heterogeneous odd `K_i` — it needed
      no new algebra, only restating Theorem 3.1's double count, which uses
      nothing about the `K_i` beyond `Σ_i K_i` being odd;
    · `closed_walk_coordSum_zero` supplies the telescoping. Splitting the
      last step off with `iSum_range_succ` turns the modular index into a
      plain one on `List.range m`, so no rotation-permutation of
      `List.range n` is needed anywhere.

    `Classical.choose` picks the per-edge `K` from the existential in the
    hypothesis; that is the only source of `Classical.choice` here.

    NAME WARNING: `sedenion_zd_surgery_bipartite` reads broader than what it
    says. The statement below is the honest one — a FIXED `p`, and edges
    restricted to `zdEdge ∧ ∃ K, isKDiff K`. Read the binders, not the name. -/
theorem sedenion_zd_surgery_bipartite (p : ZDPrim) :
    ∀ (n : Nat) (hn : n % 2 = 1) (cycle : Nat → SedVec),
      (∀ i, i < n →
        zdEdge p (cycle i) (cycle ((i + 1) % n)) ∧
          ∃ K, isKDiff K (fun j => cycle i j - cycle ((i + 1) % n) j)) →
      False := by
  intro n hn cycle hedges
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
  let ds : Nat → SedVec := fun i j => cycle i j - cycle ((i + 1) % (m + 1)) j
  -- totalise the per-edge support so `Classical.choose` gives a function
  have hex : ∀ i, ∃ K, i < m + 1 → isKDiff K (ds i) := by
    intro i
    if hi : i < m + 1 then
      obtain ⟨K, hK⟩ := (hedges i hi).2
      exact ⟨K, fun _ => hK⟩
    else
      exact ⟨0, fun h => absurd h hi⟩
  let Ks : Nat → Nat := fun i => Classical.choose (hex i)
  have hKs : ∀ i, i < m + 1 → isKDiff (Ks i) (ds i) :=
    fun i hi => Classical.choose_spec (hex i) hi
  -- every edge is an odd-K diff, because Theorem 3.2 kills the even ones
  have hodd_i : ∀ i, i < m + 1 → Ks i % 2 = 1 := by
    intro i hi
    have hcases : Ks i % 2 = 0 ∨ Ks i % 2 = 1 := by omega
    rcases hcases with h0 | h1
    · exact absurd (hedges i hi).1 (k_even_no_edges (Ks i) h0 (ds i) p (hKs i hi))
    · exact h1
  have hsum_odd : nSum (List.range (m + 1)) Ks % 2 = 1 := by
    rw [nSum_odd_parity _ _ (fun x hx => hodd_i x (List.mem_range.mp hx)),
      List.length_range]
    exact hn
  exact mixed_odd_no_odd_cycle (m + 1) ds Ks hKs hsum_odd
    (fun j _ => closed_walk_coordSum_zero m cycle j)

/-! ## §8. The union over all primitives — closing the monochromatic gap

    The theorem above fixes one `p`, and a math-review flagged that as a real
    scope limit: a union of bipartite graphs need not be bipartite, so
    bipartiteness for each of the 84 primitives separately does not give
    bipartiteness of the union graph in which an edge may use ANY primitive.

    That objection is correct about the STATEMENT and wrong about the PROOF.
    Reading §7, `p` is consumed in exactly one place — the appeal to
    `k_even_no_edges`, which is per-edge. The parity contradiction that does the
    real work, `mixed_odd_no_odd_cycle` together with `closed_walk_coordSum_zero`,
    takes no `ZDPrim` at all: it is a statement about the support sizes `Ks` and
    the telescoping coordinate sums of the walk. So the argument never needed `p`
    to be constant, and the fixed-`p` binder was an artefact of how it was written.

    Below, each edge `i` carries its own primitive `ps i`. Nothing else changes:
    `k_even_no_edges` is applied at `ps i`, and the closing contradiction is
    verbatim the one from §7.

    Why this matters beyond tidiness: the graph whose edges may use any of the 84
    primitives is the object the χ = 2 claim is actually about. Until now that
    claim rested on a numerical census; the union statement below is a proof, for
    arbitrary per-edge primitives and arbitrary odd cycle length.

    Scope that remains, stated plainly rather than buried: edges are still
    `zdEdge ∧ ∃ K, isKDiff K`, i.e. integer difference vectors with unit
    coordinates. This is not a statement about all of `ℝ^16`. -/
theorem sedenion_zd_surgery_bipartite_union :
    ∀ (n : Nat) (hn : n % 2 = 1) (cycle : Nat → SedVec) (ps : Nat → ZDPrim),
      (∀ i, i < n →
        zdEdge (ps i) (cycle i) (cycle ((i + 1) % n)) ∧
          ∃ K, isKDiff K (fun j => cycle i j - cycle ((i + 1) % n) j)) →
      False := by
  intro n hn cycle ps hedges
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
  let ds : Nat → SedVec := fun i j => cycle i j - cycle ((i + 1) % (m + 1)) j
  have hex : ∀ i, ∃ K, i < m + 1 → isKDiff K (ds i) := by
    intro i
    if hi : i < m + 1 then
      obtain ⟨K, hK⟩ := (hedges i hi).2
      exact ⟨K, fun _ => hK⟩
    else
      exact ⟨0, fun h => absurd h hi⟩
  let Ks : Nat → Nat := fun i => Classical.choose (hex i)
  have hKs : ∀ i, i < m + 1 → isKDiff (Ks i) (ds i) :=
    fun i hi => Classical.choose_spec (hex i) hi
  -- the ONLY use of a primitive, and it is per-edge: `ps i`, not a fixed `p`
  have hodd_i : ∀ i, i < m + 1 → Ks i % 2 = 1 := by
    intro i hi
    have hcases : Ks i % 2 = 0 ∨ Ks i % 2 = 1 := by omega
    rcases hcases with h0 | h1
    · exact absurd (hedges i hi).1 (k_even_no_edges (Ks i) h0 (ds i) (ps i) (hKs i hi))
    · exact h1
  have hsum_odd : nSum (List.range (m + 1)) Ks % 2 = 1 := by
    rw [nSum_odd_parity _ _ (fun x hx => hodd_i x (List.mem_range.mp hx)),
      List.length_range]
    exact hn
  exact mixed_odd_no_odd_cycle (m + 1) ds Ks hKs hsum_odd
    (fun j _ => closed_walk_coordSum_zero m cycle j)

/-! ### §8.1 The union statement is non-vacuous, and strictly stronger

    Two objections deserve pre-emption, both answered by kernel computation
    rather than assertion. `decide` only; no `native_decide` anywhere.

    (a) VACUITY. A "no odd cycle" theorem says nothing if the edge relation is
        empty. `edge_exists` exhibits one.

    (b) TRIVIALITY. If every union edge were an edge of one fixed primitive, the
        union theorem would merely restate §7. `edge_separates` exhibits a single
        difference vector that IS an edge for one primitive and is NOT an edge
        for another, so the per-primitive graphs genuinely differ and the union
        is strictly larger than any one of them.

    Measured over all 560 three-element supports with all-`+1` coordinates, these
    two primitives disagree on edgehood for 56 of them and agree on none: their
    K=3 edge sets are disjoint. The witness below is the first such support. -/

def pA : ZDPrim := ⟨1, 10, false, by decide, by decide, by decide⟩
def pB : ZDPrim := ⟨3, 12, true, by decide, by decide, by decide⟩

/-- `e₀`, a single live coordinate. -/
def eUnit : SedVec := fun i => if i = 0 then 1 else 0

/-- Support `{0, 2, 13}`, all coefficients `+1`. -/
def dSep : SedVec := fun i => if i = 0 || i = 2 || i = 13 then 1 else 0

/-- (a) Non-vacuity: a K=1 difference is an edge, so the graph has edges. -/
theorem edge_exists : zdEdge pA eUnit (fun _ => 0) := by
  unfold zdEdge
  decide

/-- (b) The edge relation is primitive-dependent: `dSep` is an edge for `pB`
    and is not an edge for `pA`. Hence the union graph strictly contains the
    fixed-`pA` graph, and §8 is not a restatement of §7. -/
theorem edge_separates :
    zdEdge pB dSep (fun _ => 0) ∧ ¬ zdEdge pA dSep (fun _ => 0) := by
  unfold zdEdge
  exact ⟨by decide, by decide⟩

/-- The separating vector really is a K=3 difference, so it is admissible as an
    edge of the union graph under the `isKDiff` side condition. -/
theorem dSep_isKDiff : isKDiff 3 (fun i => dSep i - (0 : Int)) := by
  refine ⟨by decide, ?_, ?_⟩
  · intro i _
    unfold isUnitCoord dSep
    by_cases h : i = 0 || i = 2 || i = 13 <;> simp [h]
  · intro i hi
    unfold dSep
    have h0 : ¬ (i = 0) := by omega
    have h2 : ¬ (i = 2) := by omega
    have h13 : ¬ (i = 13) := by omega
    simp [h0, h2, h13]

/-- The fixed-`p` theorem of §7 is the special case where every edge uses the
same primitive. Recorded so the strengthening is visibly a strengthening. -/
theorem sedenion_zd_surgery_bipartite_of_union (p : ZDPrim) :
    ∀ (n : Nat) (hn : n % 2 = 1) (cycle : Nat → SedVec),
      (∀ i, i < n →
        zdEdge p (cycle i) (cycle ((i + 1) % n)) ∧
          ∃ K, isKDiff K (fun j => cycle i j - cycle ((i + 1) % n) j)) →
      False :=
  fun n hn cycle hedges =>
    sedenion_zd_surgery_bipartite_union n hn cycle (fun _ => p) hedges

end Sounio.SedenionBipartite
