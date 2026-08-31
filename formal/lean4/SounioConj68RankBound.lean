/-
  SounioConj68RankBound — Lean leg of the rank-bound half of the attack on
  Conjecture 6.8 (Guterman–Zhilina, arXiv:2608.26903): the commutativity graph
  of the sedenions restricted to zero-divisor imaginary parts has diameter 3.

  Companion to tests/run-pass/sedenion_conj68_basis_probe.sio,
  examples/research/conj68_rank_structure.sio and
  docs/research/conj68_attack_log_2026-08-31.md.

  THE STRUCTURE THEOREM (measured 145/145 pairs, proved in sketch below):
  for zero divisors x = (a,b), x' = (a',b') of the sedenions, the bilinear map
      T : Im C(x) × Im C(x') → Im 𝕊,   T(u,w) = [u,w]
  has  im(T) ⊥ span{ x, x', x̃, x̃' },  where x̃ = (b,−a) is the hexagon
  companion (Lemma 3.6 of the paper), hence rank(T) ≤ 15 − 4 = 11.

  Proof sketch (general case):
    (i)  φ(u,w,v) := ⟨[u,w],v⟩ is cyclic: φ(u,w,v) = φ(w,v,u), from the
         inner-product identities ⟨a,bc⟩ = ⟨ac̄,b⟩ = ⟨b̄a,c⟩ (Schafer / paper
         Lemma 4.13) applied to pure elements.
    (ii) x ⊥ im T:  φ(u,w,x) = φ(x,u,w) = ⟨[x,u],w⟩ = 0 since u ∈ C(x).
         Symmetrically x' ⊥ im T.
    (iii) x̃ ⊥ im T:  KEY LEMMA (Lemma B below): [x̃, u] ∈ ℝ·(0,1) = ℝ·e₈
         for every u ∈ Im C(x) — from the double-hexagon multiplication table
         (Table 2 of the paper): [x̃,x] = 4·f̃₀ and x̃ commutes with all four
         generators of O(x).  Every element of Im C(x') is a zero divisor or a
         real multiple of one, hence DOUBLY PURE, hence ⊥ e₈.  So
         φ(u,w,x̃) = φ(x̃,u,w) = ⟨[x̃,u],w⟩ = 0.  Symmetrically x̃' ⊥ im T.

  This file verifies, by kernel evaluation (native_decide, Mathlib-free,
  no sorry), the canonical instances over ℤ:
    C1  the U-frame of x₀ = (e₁,e₂) annihilates x₀ bilaterally,
    C2  Lemma B: [x̃₀, u] is supported only at coordinate 8 for all u in the
        U-frame,
    C3  all frame vectors (both sides) are doubly pure,
    C4  the four complement vectors are ⊥ all 25 commutators of the frames of
        the pair (e₁,e₂) vs (e₂,e₃) — the hardest ("class 3") pair of the
        basis sweep.
  The general statement follows from these by Aut(𝕆)-transitivity on ZD pairs
  (Khalil–Yiu) plus bilinearity; the transitivity reduction is future work.
-/

namespace SounioConj68RankBound

def cdSigma (a b : Nat) : Nat → Int
  | 0 => -1
  | 1 => if a == 0 || b == 0 then 1 else -1
  | (n+2) =>
      if a == 0 || b == 0 then 1
      else
        let half := 2 ^ (n+1)
        let aHi := a ≥ half; let bHi := b ≥ half; let aLo := a % half; let bLo := b % half
        if !aHi && !bHi then cdSigma aLo bLo (n+1)
        else if !aHi && bHi then cdSigma bLo aLo (n+1)
        else if aHi && !bHi then (if bLo == 0 then cdSigma aLo 0 (n+1) else - cdSigma aLo bLo (n+1))
        else (if bLo == 0 then - cdSigma 0 aLo (n+1) else cdSigma bLo aLo (n+1))

def sedSigma (a b : Nat) : Int := cdSigma a b 4

abbrev Vec := List Int

def coord (x : Vec) (k : Nat) : Int := x.getD k 0

/-- Sedenion product coordinate k: with j = i ^^^ k unique, sum over i. -/
def mulC (x y : Vec) (k : Nat) : Int :=
  (List.range 16).foldl (fun acc i => acc + sedSigma i (i ^^^ k) * coord x i * coord y (i ^^^ k)) 0

def mul (x y : Vec) : Vec := (List.range 16).map (mulC x y)

def vsub (x y : Vec) : Vec := (List.range 16).map (fun k => coord x k - coord y k)
def vadd (x y : Vec) : Vec := (List.range 16).map (fun k => coord x k + coord y k)

def comm (x y : Vec) : Vec := vsub (mul x y) (mul y x)

def isZero (v : Vec) : Bool := (List.range 16).all (fun k => coord v k == 0)

def dot (x y : Vec) : Int := (List.range 16).foldl (fun acc k => acc + coord x k * coord y k) 0

def e (i : Nat) : Vec := (List.range 16).map (fun k => if k == i then (1 : Int) else 0)

/-- x₀ = (e₁, e₂) as the sedenion e₁ + e₁₀. -/
def x0 : Vec := vadd (e 1) (e 10)

/-- Hexagon companion x̃₀ = (b, −a) = (e₂, −e₁) = e₂ − e₉. -/
def xt0 : Vec := vsub (e 2) (e 9)

/-- U-frame: x₀ and its four basis partners (double-hexagon neighbours). -/
def uFrame : List Vec :=
  [x0, vsub (e 4) (e 15), vadd (e 5) (e 14), vsub (e 6) (e 13), vadd (e 7) (e 12)]

/-- The hard pair: x' = (e₂, e₃) = e₂ + e₁₁ and its companion (e₃,−e₂). -/
def xp : Vec := vadd (e 2) (e 11)
def xtp : Vec := vsub (e 3) (e 10)

/-- V-frame of x': x' and its four basis partners. -/
def vFrame : List Vec :=
  [xp, vsub (e 4) (e 13), vadd (e 5) (e 12), vadd (e 6) (e 15), vsub (e 7) (e 14)]

/-- C1: the four U-frame partners annihilate x₀ on both sides (they generate
    O(x₀)), and the V-frame partners annihilate x'. -/
def framesAnnihilate : Bool :=
  (uFrame.drop 1).all (fun u => isZero (mul x0 u) && isZero (mul u x0)) &&
  (vFrame.drop 1).all (fun w => isZero (mul xp w) && isZero (mul w xp))

theorem frames_annihilate : framesAnnihilate = true := by native_decide

/-- C2 (Lemma B, canonical): [x̃₀, u] is supported only at coordinate 8
    (= the direction (0,1)) for every u in the U-frame. -/
def lemmaB : Bool :=
  uFrame.all (fun u => (List.range 16).all (fun k => k == 8 || coord (comm xt0 u) k == 0))

theorem lemmaB_canonical : lemmaB = true := by native_decide

/-- C3: every frame vector on both sides is doubly pure (coords 0 and 8 zero).
    This is what makes Lemma B kill φ(u,w,x̃): w ⊥ e₈ for all w ∈ Im C(x'). -/
def doublyPure : Bool :=
  (uFrame ++ vFrame).all (fun v => coord v 0 == 0 && coord v 8 == 0)

theorem frames_doubly_pure : doublyPure = true := by native_decide

/-- C4: the four complement directions {x₀, x', x̃₀, x̃'} are orthogonal to all
    25 commutators [uᵢ, wⱼ] of the pair (e₁,e₂) vs (e₂,e₃).  With the four
    vectors linearly independent (immediate from supports), this exhibits
    rank(T) ≤ 11 for the canonical hard pair. -/
def complementOrthogonal : Bool :=
  uFrame.all (fun u => vFrame.all (fun w =>
    let c := comm u w
    dot c x0 == 0 && dot c xp == 0 && dot c xt0 == 0 && dot c xtp == 0))

theorem complement_orthogonal_canonical : complementOrthogonal = true := by native_decide

/-- The four complement vectors are linearly independent: the 4×16 coordinate
    matrix contains the identity-like minor on coordinates (1,9),(11,10)… —
    checked here by a determinant-free pigeonhole: each vector has a coordinate
    where it is the unique nonzero one among the four. -/
def complementIndependent : Bool :=
  let vs := [x0, xp, xt0, xtp]
  -- x0 unique at 1? x0=e1+e10, xtp=e3−e10 … use coords 1, 11, 9, 3:
  (coord x0 1 != 0 && coord xp 1 == 0 && coord xt0 1 == 0 && coord xtp 1 == 0) &&
  (coord xp 11 != 0 && coord x0 11 == 0 && coord xt0 11 == 0 && coord xtp 11 == 0) &&
  (coord xt0 9 != 0 && coord x0 9 == 0 && coord xp 9 == 0 && coord xtp 9 == 0) &&
  (coord xtp 3 != 0 && coord x0 3 == 0 && coord xp 3 == 0 && coord xt0 3 == 0) &&
  vs.length == 4

theorem complement_independent : complementIndependent = true := by native_decide

/-! ## The sector (O × O) ghost law — Rodada 6

For the O(x) × O(x') sector the image drops two MORE dimensions:
    im(T|O×O) ⊥ (n(A')·z + n(A)·z', 0)  and  (0, n(A')·z + n(A)·z'),
z = ab, z' = a'b' — the norm-normalized sum of the Γ_O-invariants.  Found by
the dictionary-kernel pipeline (mod-p nullspace + rational reconstruction +
exact ℤ verification, 25/25 generic pairs, VERIFY_FAIL = 0); explains the
(z+z') stratum seen earlier as the n(A) = n(A') case.  Hence rank(T|₄ₓ₄) ≤ 9
(measured tight).  Canonical instance: pair (e₁,e₂) vs (e₂,e₃) has
n = n' = 1, z = e₃, z' = e₁, so the ghosts are e₁+e₃ and e₉+e₁₁. -/

def ghost1 : Vec := vadd (e 1) (e 3)
def ghost2 : Vec := vadd (e 9) (e 11)

def ghostOrthogonal : Bool :=
  (uFrame.drop 1).all (fun u => (vFrame.drop 1).all (fun w =>
    let c := comm u w
    dot c ghost1 == 0 && dot c ghost2 == 0))

theorem ghost_orthogonal_canonical : ghostOrthogonal = true := by native_decide

/-! ## PAPER PROOF of the two ghost identities (Rodada 7)

Normalize n(a)=n(b)=n(a')=n(b')=1; frames u = (c, cz) with c ⊥ {1,a,b,z},
w = (g, gz') with g ⊥ {1,a',b',z'}; φ(u,w,v) = ⟨[u,w],v⟩ cyclic.

GHOST 1, Z = (z+z', 0):  φ(u,w,Z) = ⟨[Z,u],w⟩.
  [(z,0),u]  = ([z,c], -2c) = (-2cz, -2c)      (z ⊥ c so zc = -cz; (cz)z = -c)
  [(z',0),u] = ([z',c], 2(cz)z').
  ⟨(cz)z', gz'⟩ = ⟨cz,g⟩ n(z') = ⟨cz,g⟩       (right mult by unit = isometry)
  ⟨c, gz'⟩ = ⟨c z̄', g⟩ = -⟨cz', g⟩.
  Sum = -2⟨cz,g⟩ - 2⟨c,gz'⟩ + ⟨z'c - cz', g⟩ + 2⟨cz,g⟩
      = 2⟨cz',g⟩ + ⟨z'c,g⟩ - ⟨cz',g⟩ = ⟨cz' + z'c, g⟩ = -2⟨c,z'⟩ Re(g) = 0
  since g is PURE.  ∎

GHOST 2, Z₂ = (0, z+z'):  φ(u,w,Z₂) = ⟨[Z₂,u],w⟩.
  [(0,z),u]  = (-2c, 2cz)
  [(0,z'),u] = ([cz,z'], -2z'c)               (flexibility for z'(cz))
  ⟨(cz)z', g⟩ = -⟨cz, gz'⟩;  ⟨z'(cz), g⟩ = -⟨cz, z'g⟩
  ⟨cz, gz' + z'g⟩ = -2⟨g,z'⟩ Re(cz) = 0       (cz pure)
  ⟨z'c, gz'⟩ = -⟨z'cz', g⟩ = -⟨c,g⟩ + 2⟨c,z'⟩⟨z',g⟩   (z'cz' = c - 2⟨c,z'⟩z')
  Sum = -2⟨c,g⟩ + 2⟨cz,gz'⟩ - ⟨cz,gz'⟩ + ⟨cz,z'g⟩ - 2⟨z'c,gz'⟩
      = -4⟨c,z'⟩⟨z',g⟩ = 0   since g ⊥ z'.  ∎

Unnormalized scaling gives the measured law (n(A')·z + n(A)·z').
Below: kernel verification on a SECOND canonical pair with n(A)=1, n(A')=2,
pinning the normalized coefficients.  x'' = (e₂+e₄, -e₃-e₅): z'' = -2e₁,
ghost = 2z + z'' = 2e₃ - 2e₁ ∝ e₃ - e₁. -/

def xpp : Vec := vsub (vsub (vadd (e 2) (e 4)) (e 11)) (e 13)

def vFrame2 : List Vec :=
  [xpp,
   vsub (e 6) (e 15),
   vadd (e 7) (e 14),
   vsub (vadd (vsub (e 2) (e 4)) (e 11)) (e 13),
   vsub (vadd (vsub (e 3) (e 5)) (e 12)) (e 10)]

def frames2Annihilate : Bool :=
  (vFrame2.drop 1).all (fun w => isZero (mul xpp w) && isZero (mul w xpp))

theorem frames2_annihilate : frames2Annihilate = true := by native_decide

def ghost1n : Vec := vsub (e 3) (e 1)
def ghost2n : Vec := vsub (e 11) (e 9)

/-- The NORMALIZED ghost law at n(A)=1, n(A')=2: the direction
    n(A')·z + n(A)·z' = 2e₃ - 2e₁ (embedded in both slots) is orthogonal to
    every sector commutator of the pair (e₁,e₂) vs (e₂+e₄, -e₃-e₅). -/
def ghostNormalized : Bool :=
  (uFrame.drop 1).all (fun u => (vFrame2.drop 1).all (fun w =>
    let c := comm u w
    dot c ghost1n == 0 && dot c ghost2n == 0))

theorem ghost_normalized_n2 : ghostNormalized = true := by native_decide

end SounioConj68RankBound
