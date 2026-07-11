/-
  SounioCDRecursiveSeam -- a full-box geometric zero-divisor predicate.

  `offSeam` describes one fixed lower x upper Cayley-Dickson cut.  For an
  arbitrary pair, `recursiveOffSeam` removes common leading doubling bits until
  the pair first crosses a cut, then applies `offSeam` at that local level.
  The recursion stops at the octonions: levels <= 3 have no zero divisors.

  Mathlib-free.  The scale-independent theorems below contain no
  `native_decide` and no `sorry`.
-/
import SounioCDConverse

namespace SounioCDRecursiveSeam

open SounioCDTowerSeam
open SounioCDConverse

/-- Geometric seam classification for the full Cayley-Dickson box.  Equal
    high/low tags recurse after deleting the shared leading bit.  At the first
    mixed cut, orient the lower index first and reuse the local `offSeam` test. -/
def recursiveOffSeam : Nat -> Nat -> Nat -> Bool
  | 0, _, _ => false
  | bits + 1, l, u =>
      if (l == 0) || (u == 0) || (l == u) then false
      else if bits < 3 then false
      else
        let top := 2 ^ bits
        if (l < top) == (u < top) then
          recursiveOffSeam bits (l % top) (u % top)
        else if l < top then
          offSeam (bits + 1) l u
        else
          offSeam (bits + 1) u l

/-- Finite checker used only to discharge the division-algebra base levels in
    the kernel.  It is not part of the scale-independent induction step. -/
def recursiveAgreement (bits : Nat) : Bool :=
  let N := 2 ^ bits
  (List.range N).all (fun l => (List.range N).all (fun u =>
    (l == 0) || (u == 0) || (l == u)
      || (recursiveOffSeam bits l u == hasXorAnnih bits l u)))

theorem recursive_agreement_0 : recursiveAgreement 0 = true := by decide
theorem recursive_agreement_1 : recursiveAgreement 1 = true := by decide
theorem recursive_agreement_2 : recursiveAgreement 2 = true := by decide
theorem recursive_agreement_3 : recursiveAgreement 3 = true := by decide

/-- Extract a pointwise result from a finite agreement checker. -/
theorem eq_of_recursiveAgreement (bits l u : Nat)
    (hA : recursiveAgreement bits = true)
    (hl1 : 1 <= l) (hl : l < 2 ^ bits)
    (hu1 : 1 <= u) (hu : u < 2 ^ bits) (hne : l ≠ u) :
    recursiveOffSeam bits l u = hasXorAnnih bits l u := by
  unfold recursiveAgreement at hA
  rw [List.all_eq_true] at hA
  have hlrow := hA l (List.mem_range.mpr hl)
  rw [List.all_eq_true] at hlrow
  have hcell := hlrow u (List.mem_range.mpr hu)
  have hl0 : (l == 0) = false := beq_eq_false_iff_ne.mpr (by omega)
  have hu0 : (u == 0) = false := beq_eq_false_iff_ne.mpr (by omega)
  have hlu : (l == u) = false := beq_eq_false_iff_ne.mpr hne
  rw [hl0, hu0, hlu] at hcell
  exact beq_iff_eq.mp hcell

/-- The XOR-winner predicate is symmetric in its two basis indices. -/
theorem hasXorAnnih_comm (bits l u : Nat) :
    hasXorAnnih bits l u = hasXorAnnih bits u l := by
  have forward : forall x y,
      hasXorAnnih bits x y = true -> hasXorAnnih bits y x = true := by
    intro x y h
    unfold hasXorAnnih at h |-
    rw [List.any_eq_true] at h |-
    obtain ⟨a, ha, hp⟩ := h
    refine ⟨a, ha, ?_⟩
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hp |-
    obtain ⟨⟨ha1, had⟩, hP⟩ := hp
    refine ⟨⟨ha1, ?_⟩, ?_⟩
    . rwa [Nat.xor_comm]
    . rw [Nat.xor_comm y x]
      rcases cdSigma_pm bits x a with h1 | h1 <;>
      rcases cdSigma_pm bits y a with h2 | h2 <;>
      rcases cdSigma_pm bits x (a ^^^ (x ^^^ y)) with h3 | h3 <;>
      rcases cdSigma_pm bits y (a ^^^ (x ^^^ y)) with h4 | h4 <;>
        rw [h1, h2, h3, h4] at hP |- <;> first | exact hP | exact absurd hP (by decide)
  apply Bool.eq_iff_iff.mpr
  exact ⟨forward l u, forward u l⟩

/-- Winner value on the XOR orbit anchored at `a`. -/
def winnerVal (bits l u a : Nat) : Int :=
  fVal l u a bits * fVal l u (a ^^^ (l ^^^ u)) bits

/-- Transposing all four nonzero basis-unit sign factors preserves the winner
    value.  The exceptional orbits `{0,l⊕u}` and `{l,u}` are handled
    explicitly; the generic case has four antisymmetry signs, which cancel. -/
theorem winner_transpose (bits l u r : Nat)
    (hl1 : 1 <= l) (hl : l < 2 ^ bits)
    (hu1 : 1 <= u) (hu : u < 2 ^ bits) (hne : l ≠ u)
    (hr : r < 2 ^ bits) :
    cdSigma r l bits * cdSigma r u bits
        * cdSigma (r ^^^ (l ^^^ u)) l bits * cdSigma (r ^^^ (l ^^^ u)) u bits
      = cdSigma l r bits * cdSigma u r bits
        * cdSigma l (r ^^^ (l ^^^ u)) bits * cdSigma u (r ^^^ (l ^^^ u)) bits := by
  have hd : l ^^^ u < 2 ^ bits := Nat.xor_lt_two_pow hl hu
  have hq : r ^^^ (l ^^^ u) < 2 ^ bits := Nat.xor_lt_two_pow hr hd
  have hbits : 1 <= bits := by
    cases bits with
    | zero => rw [Nat.pow_zero] at hl; omega
    | succ _ => omega
  have hd0 : l ^^^ u ≠ 0 := fun h => hne (xor_eq_zero_of l u h)
  have hd1 : 1 <= l ^^^ u := Nat.one_le_iff_ne_zero.mpr hd0
  have hdl : l ^^^ u ≠ l := by
    intro h
    have e : l ^^^ u = l ^^^ 0 := by simpa using h
    have := xor_cancel_left l u 0 e
    omega
  have hdu : l ^^^ u ≠ u := by
    intro h
    have e : l ^^^ u = 0 ^^^ u := by simpa using h
    have := xor_cancel_right l 0 u e
    omega
  by_cases hr0 : r = 0
  . subst r
    rw [Nat.zero_xor,
      cdSigma_zero_left bits l hbits, cdSigma_zero_left bits u hbits,
      cdSigma_zero_right bits l hbits, cdSigma_zero_right bits u hbits]
    rw [cdAntisym_all bits (l ^^^ u) l hd1 hd hl1 hl hdl,
        cdAntisym_all bits (l ^^^ u) u hd1 hd hu1 hu hdu]
    rcases cdSigma_pm bits l (l ^^^ u) with h1 | h1 <;>
    rcases cdSigma_pm bits u (l ^^^ u) with h2 | h2 <;> rw [h1, h2] <;> decide
  by_cases hrd : r = l ^^^ u
  . subst r
    have hdd : (l ^^^ u) ^^^ (l ^^^ u) = 0 := Nat.xor_self _
    rw [hdd,
      cdSigma_zero_left bits l hbits, cdSigma_zero_left bits u hbits,
      cdSigma_zero_right bits l hbits, cdSigma_zero_right bits u hbits]
    rw [cdAntisym_all bits (l ^^^ u) l hd1 hd hl1 hl hdl,
        cdAntisym_all bits (l ^^^ u) u hd1 hd hu1 hu hdu]
    rcases cdSigma_pm bits l (l ^^^ u) with h1 | h1 <;>
    rcases cdSigma_pm bits u (l ^^^ u) with h2 | h2 <;> rw [h1, h2] <;> decide
  by_cases hrl : r = l
  . subst r
    have hqeq : l ^^^ (l ^^^ u) = u := by
      rw [← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor]
    rw [hqeq]
    rcases cdSigma_pm bits l l with h1 | h1 <;>
    rcases cdSigma_pm bits l u with h2 | h2 <;>
    rcases cdSigma_pm bits u l with h3 | h3 <;>
    rcases cdSigma_pm bits u u with h4 | h4 <;> rw [h1, h2, h3, h4] <;> decide
  by_cases hru : r = u
  . subst r
    have hqeq : u ^^^ (l ^^^ u) = l := by
      rw [xor_left_comm u l u, Nat.xor_self, Nat.xor_zero]
    rw [hqeq]
    rcases cdSigma_pm bits u l with h1 | h1 <;>
    rcases cdSigma_pm bits u u with h2 | h2 <;>
    rcases cdSigma_pm bits l l with h3 | h3 <;>
    rcases cdSigma_pm bits l u with h4 | h4 <;> rw [h1, h2, h3, h4] <;> decide
  have hr1 : 1 <= r := by omega
  have hlxd : l ^^^ (l ^^^ u) = u := by
    rw [← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor]
  have huxd : u ^^^ (l ^^^ u) = l := by
    rw [xor_left_comm u l u, Nat.xor_self, Nat.xor_zero]
  have hql : r ^^^ (l ^^^ u) ≠ l := by
    intro h
    apply hru
    apply xor_cancel_right r u (l ^^^ u)
    rw [h, huxd]
  have hqu : r ^^^ (l ^^^ u) ≠ u := by
    intro h
    apply hrl
    apply xor_cancel_right r l (l ^^^ u)
    rw [h, hlxd]
  have hq0 : r ^^^ (l ^^^ u) ≠ 0 := by
    intro h
    apply hrd
    exact xor_eq_zero_of r (l ^^^ u) h
  have hq1 : 1 <= r ^^^ (l ^^^ u) := Nat.one_le_iff_ne_zero.mpr hq0
  rw [cdAntisym_all bits r l hr1 hr hl1 hl hrl,
      cdAntisym_all bits r u hr1 hr hu1 hu hru,
      cdAntisym_all bits (r ^^^ (l ^^^ u)) l hq1 hq hl1 hl hql,
      cdAntisym_all bits (r ^^^ (l ^^^ u)) u hq1 hq hu1 hu hqu]
  rcases cdSigma_pm bits l r with h1 | h1 <;>
  rcases cdSigma_pm bits u r with h2 | h2 <;>
  rcases cdSigma_pm bits l (r ^^^ (l ^^^ u)) with h3 | h3 <;>
  rcases cdSigma_pm bits u (r ^^^ (l ^^^ u)) with h4 | h4 <;>
    rw [h1, h2, h3, h4] <;> decide

/-- A high representative of a both-low pair carries exactly the downstairs
    winner value. -/
theorem winner_high_both_low (n l u r : Nat)
    (hl1 : 1 <= l) (hl : l < 2 ^ (n + 1))
    (hu1 : 1 <= u) (hu : u < 2 ^ (n + 1)) (hne : l ≠ u)
    (hr : r < 2 ^ (n + 1)) :
    winnerVal (n + 2) l u (2 ^ (n + 1) + r) = winnerVal (n + 1) l u r := by
  have hd : l ^^^ u < 2 ^ (n + 1) := Nat.xor_lt_two_pow hl hu
  have hq : r ^^^ (l ^^^ u) < 2 ^ (n + 1) := Nat.xor_lt_two_pow hr hd
  have hidx : (2 ^ (n + 1) + r) ^^^ (l ^^^ u)
      = 2 ^ (n + 1) + (r ^^^ (l ^^^ u)) := by
    calc
      (2 ^ (n + 1) + r) ^^^ (l ^^^ u)
          = (2 ^ (n + 1) ^^^ r) ^^^ (l ^^^ u) := by
              rw [two_pow_xor_eq_add (n + 1) r hr]
      _ = 2 ^ (n + 1) ^^^ (r ^^^ (l ^^^ u)) := Nat.xor_assoc _ _ _
      _ = 2 ^ (n + 1) + (r ^^^ (l ^^^ u)) :=
            two_pow_xor_eq_add (n + 1) _ hq
  unfold winnerVal fVal
  rw [hidx,
    cdSigma_lo_hi n r l hr hl1 hl,
    cdSigma_lo_hi n r u hr hu1 hu,
    cdSigma_lo_hi n (r ^^^ (l ^^^ u)) l hq hl1 hl,
    cdSigma_lo_hi n (r ^^^ (l ^^^ u)) u hq hu1 hu]
  simpa [Int.mul_assoc] using winner_transpose (n + 1) l u r hl1 hl hu1 hu hne hr

theorem winnerVal_zero_neg (bits l u : Nat)
    (hl1 : 1 <= l) (hl : l < 2 ^ bits)
    (hu1 : 1 <= u) (hu : u < 2 ^ bits) (hne : l ≠ u) :
    winnerVal bits l u 0 = -1 := by
  have h := P0_neg_general bits l u hl1 hl hu1 hu hne
  simpa [winnerVal, fVal, Nat.zero_xor, Int.mul_assoc] using h

theorem winnerVal_orbit (bits l u a : Nat) :
    winnerVal bits l u (a ^^^ (l ^^^ u)) = winnerVal bits l u a := by
  unfold winnerVal
  have h := P_orbit_inv l u a bits
  exact h.symm

/-- Embedding a nonzero distinct both-low pair into the next CD level neither
    creates nor destroys an XOR-winner. -/
theorem hasXorAnnih_both_low (n l u : Nat)
    (hl1 : 1 <= l) (hl : l < 2 ^ (n + 1))
    (hu1 : 1 <= u) (hu : u < 2 ^ (n + 1)) (hne : l ≠ u) :
    hasXorAnnih (n + 2) l u = hasXorAnnih (n + 1) l u := by
  have hpow : 2 ^ (n + 2) = 2 ^ (n + 1) * 2 := by
    rw [show n + 2 = (n + 1) + 1 by omega, Nat.pow_succ]
  have hd : l ^^^ u < 2 ^ (n + 1) := Nat.xor_lt_two_pow hl hu
  have hltup : l < 2 ^ (n + 2) := by omega
  have huup : u < 2 ^ (n + 2) := by omega
  apply Bool.eq_iff_iff.mpr
  constructor
  . intro h
    unfold hasXorAnnih at h
    rw [List.any_eq_true] at h
    obtain ⟨a, ha_mem, hp⟩ := h
    rw [List.mem_range] at ha_mem
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hp
    obtain ⟨⟨ha1, had⟩, hP⟩ := hp
    rw [P_eq_fVal] at hP
    change winnerVal (n + 2) l u a = 1 at hP
    by_cases ha : a < 2 ^ (n + 1)
    . unfold hasXorAnnih
      refine List.any_eq_true.mpr ⟨a, List.mem_range.mpr ha, ?_⟩
      rw [Bool.and_eq_true, Bool.and_eq_true, decide_eq_true_eq,
        decide_eq_true_eq, beq_iff_eq, P_eq_fVal]
      refine ⟨⟨ha1, had⟩, ?_⟩
      have hs := P_stable_low (n + 1) l u a hl hu ha
      change winnerVal (n + 2) l u a = winnerVal (n + 1) l u a at hs
      rw [hP] at hs
      exact hs.symm
    . obtain ⟨r, hr, har⟩ : exists r, r < 2 ^ (n + 1) ∧
          a = 2 ^ (n + 1) + r := ⟨a - 2 ^ (n + 1), by omega, by omega⟩
      subst a
      have hdown := winner_high_both_low n l u r hl1 hl hu1 hu hne hr
      rw [hP] at hdown
      have hPr : winnerVal (n + 1) l u r = 1 := hdown.symm
      have hr0 : r ≠ 0 := by
        intro h0
        subst r
        have hn := winnerVal_zero_neg (n + 1) l u hl1 hl hu1 hu hne
        rw [hPr] at hn
        exact absurd hn (by decide)
      have hrd : r ≠ l ^^^ u := by
        intro he
        subst r
        have ho := winnerVal_orbit (n + 1) l u 0
        rw [Nat.zero_xor, winnerVal_zero_neg (n + 1) l u hl1 hl hu1 hu hne, hPr] at ho
        exact absurd ho (by decide)
      unfold hasXorAnnih
      refine List.any_eq_true.mpr ⟨r, List.mem_range.mpr hr, ?_⟩
      rw [Bool.and_eq_true, Bool.and_eq_true, decide_eq_true_eq,
        decide_eq_true_eq, beq_iff_eq, P_eq_fVal]
      exact ⟨⟨by omega, hrd⟩, hPr⟩
  . intro h
    unfold hasXorAnnih at h
    rw [List.any_eq_true] at h
    obtain ⟨a, ha_mem, hp⟩ := h
    rw [List.mem_range] at ha_mem
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hp
    obtain ⟨⟨ha1, had⟩, hP⟩ := hp
    rw [P_eq_fVal] at hP
    change winnerVal (n + 1) l u a = 1 at hP
    unfold hasXorAnnih
    refine List.any_eq_true.mpr ⟨a, List.mem_range.mpr (by omega), ?_⟩
    rw [Bool.and_eq_true, Bool.and_eq_true, decide_eq_true_eq,
      decide_eq_true_eq, beq_iff_eq, P_eq_fVal]
    refine ⟨⟨ha1, had⟩, ?_⟩
    have hs := P_stable_low (n + 1) l u a hl hu ha_mem
    change winnerVal (n + 2) l u a = winnerVal (n + 1) l u a at hs
    rw [hP] at hs
    exact hs

/-- Low representatives of a both-high pair reduce directly to the pair of
    low residues. -/
theorem winner_low_both_high (n l u r : Nat)
    (hl : l < 2 ^ (n + 1)) (hu : u < 2 ^ (n + 1))
    (hr : r < 2 ^ (n + 1)) :
    winnerVal (n + 2) (2 ^ (n + 1) + l) (2 ^ (n + 1) + u) r
      = winnerVal (n + 1) l u r := by
  have hd : l ^^^ u < 2 ^ (n + 1) := Nat.xor_lt_two_pow hl hu
  have hq : r ^^^ (l ^^^ u) < 2 ^ (n + 1) := Nat.xor_lt_two_pow hr hd
  have hpair : (2 ^ (n + 1) + l) ^^^ (2 ^ (n + 1) + u) = l ^^^ u := by
    rw [← two_pow_xor_eq_add (n + 1) l hl,
      ← two_pow_xor_eq_add (n + 1) u hu, xor_cross_cancel]
  unfold winnerVal
  rw [hpair,
    fVal_high_stable (n + 1) l u r hl hu hr,
    fVal_high_stable (n + 1) l u (r ^^^ (l ^^^ u)) hl hu hq]

/-- High representatives of a both-high pair also reduce to the downstairs
    winner value.  Pure-seam right factors (`r=0` or `r=l⊕u`) use the
    dedicated `cdSigma_hi_pow` branch; all others use `cdSigma_hi_hi`. -/
theorem winner_high_both_high (n l u r : Nat)
    (hl1 : 1 <= l) (hl : l < 2 ^ (n + 1))
    (hu1 : 1 <= u) (hu : u < 2 ^ (n + 1)) (hne : l ≠ u)
    (hr : r < 2 ^ (n + 1)) :
    winnerVal (n + 2) (2 ^ (n + 1) + l) (2 ^ (n + 1) + u)
        (2 ^ (n + 1) + r)
      = winnerVal (n + 1) l u r := by
  have hd : l ^^^ u < 2 ^ (n + 1) := Nat.xor_lt_two_pow hl hu
  have hd0 : l ^^^ u ≠ 0 := fun h => hne (xor_eq_zero_of l u h)
  have hd1 : 1 <= l ^^^ u := Nat.one_le_iff_ne_zero.mpr hd0
  have hq : r ^^^ (l ^^^ u) < 2 ^ (n + 1) := Nat.xor_lt_two_pow hr hd
  have hpair : (2 ^ (n + 1) + l) ^^^ (2 ^ (n + 1) + u) = l ^^^ u := by
    rw [← two_pow_xor_eq_add (n + 1) l hl,
      ← two_pow_xor_eq_add (n + 1) u hu, xor_cross_cancel]
  have hidx : (2 ^ (n + 1) + r) ^^^ (l ^^^ u)
      = 2 ^ (n + 1) + (r ^^^ (l ^^^ u)) := by
    calc
      (2 ^ (n + 1) + r) ^^^ (l ^^^ u)
          = (2 ^ (n + 1) ^^^ r) ^^^ (l ^^^ u) := by
              rw [two_pow_xor_eq_add (n + 1) r hr]
      _ = 2 ^ (n + 1) ^^^ (r ^^^ (l ^^^ u)) := Nat.xor_assoc _ _ _
      _ = 2 ^ (n + 1) + (r ^^^ (l ^^^ u)) :=
            two_pow_xor_eq_add (n + 1) _ hq
  by_cases hr0 : r = 0
  . subst r
    have ht := winner_transpose (n + 1) l u 0 hl1 hl hu1 hu hne (by omega)
    unfold winnerVal fVal
    rw [hpair, hidx]
    simp only [Nat.zero_xor, Nat.add_zero]
    rw [
      cdSigma_hi_pow n l hl, cdSigma_hi_pow n u hu,
      cdSigma_hi_hi n l (l ^^^ u) hl hd1 hd,
      cdSigma_hi_hi n u (l ^^^ u) hu hd1 hd]
    simpa [Nat.zero_xor, Int.mul_assoc,
      cdSigma_zero_left (n + 1) l (by omega),
      cdSigma_zero_left (n + 1) u (by omega),
      cdSigma_zero_right (n + 1) l (by omega),
      cdSigma_zero_right (n + 1) u (by omega)] using ht
  by_cases hrd : r = l ^^^ u
  . have hq0 : r ^^^ (l ^^^ u) = 0 := by rw [hrd, Nat.xor_self]
    have ht := winner_transpose (n + 1) l u r hl1 hl hu1 hu hne hr
    unfold winnerVal fVal
    rw [hpair, hidx, hq0]
    simp only [Nat.add_zero]
    rw [
      cdSigma_hi_hi n l r hl (by omega) hr,
      cdSigma_hi_hi n u r hu (by omega) hr,
      cdSigma_hi_pow n l hl, cdSigma_hi_pow n u hu]
    simpa [hq0, Int.mul_assoc,
      cdSigma_zero_left (n + 1) l (by omega),
      cdSigma_zero_left (n + 1) u (by omega),
      cdSigma_zero_right (n + 1) l (by omega),
      cdSigma_zero_right (n + 1) u (by omega)] using ht
  have hr1 : 1 <= r := by omega
  have hq0 : r ^^^ (l ^^^ u) ≠ 0 := by
    intro h
    apply hrd
    exact xor_eq_zero_of r (l ^^^ u) h
  have hq1 : 1 <= r ^^^ (l ^^^ u) := Nat.one_le_iff_ne_zero.mpr hq0
  unfold winnerVal fVal
  rw [hpair, hidx,
    cdSigma_hi_hi n l r hl hr1 hr,
    cdSigma_hi_hi n u r hu hr1 hr,
    cdSigma_hi_hi n l (r ^^^ (l ^^^ u)) hl hq1 hq,
    cdSigma_hi_hi n u (r ^^^ (l ^^^ u)) hu hq1 hq]
  simpa [Int.mul_assoc] using winner_transpose (n + 1) l u r hl1 hl hu1 hu hne hr

/-- Removing a shared high CD bit from two nonzero distinct residues preserves
    the XOR-winner predicate. -/
theorem hasXorAnnih_both_high (n l u : Nat)
    (hl1 : 1 <= l) (hl : l < 2 ^ (n + 1))
    (hu1 : 1 <= u) (hu : u < 2 ^ (n + 1)) (hne : l ≠ u) :
    hasXorAnnih (n + 2) (2 ^ (n + 1) + l) (2 ^ (n + 1) + u)
      = hasXorAnnih (n + 1) l u := by
  have hpow : 2 ^ (n + 2) = 2 ^ (n + 1) * 2 := by
    rw [show n + 2 = (n + 1) + 1 by omega, Nat.pow_succ]
  have hpair : (2 ^ (n + 1) + l) ^^^ (2 ^ (n + 1) + u) = l ^^^ u := by
    rw [← two_pow_xor_eq_add (n + 1) l hl,
      ← two_pow_xor_eq_add (n + 1) u hu, xor_cross_cancel]
  apply Bool.eq_iff_iff.mpr
  constructor
  . intro h
    unfold hasXorAnnih at h
    rw [List.any_eq_true] at h
    obtain ⟨a, ha_mem, hp⟩ := h
    rw [List.mem_range] at ha_mem
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hp
    obtain ⟨⟨ha1, had⟩, hP⟩ := hp
    rw [P_eq_fVal] at hP
    change winnerVal (n + 2) (2 ^ (n + 1) + l) (2 ^ (n + 1) + u) a = 1 at hP
    by_cases ha : a < 2 ^ (n + 1)
    . have hs := winner_low_both_high n l u a hl hu ha
      rw [hP] at hs
      have hPa : winnerVal (n + 1) l u a = 1 := hs.symm
      unfold hasXorAnnih
      refine List.any_eq_true.mpr ⟨a, List.mem_range.mpr ha, ?_⟩
      rw [Bool.and_eq_true, Bool.and_eq_true, decide_eq_true_eq,
        decide_eq_true_eq, beq_iff_eq, P_eq_fVal]
      exact ⟨⟨ha1, by rwa [hpair] at had⟩, hPa⟩
    . obtain ⟨r, hr, har⟩ : exists r, r < 2 ^ (n + 1) ∧
          a = 2 ^ (n + 1) + r := ⟨a - 2 ^ (n + 1), by omega, by omega⟩
      subst a
      have hs := winner_high_both_high n l u r hl1 hl hu1 hu hne hr
      rw [hP] at hs
      have hPr : winnerVal (n + 1) l u r = 1 := hs.symm
      have hr0 : r ≠ 0 := by
        intro h0
        subst r
        have hn := winnerVal_zero_neg (n + 1) l u hl1 hl hu1 hu hne
        rw [hPr] at hn
        exact absurd hn (by decide)
      have hrd : r ≠ l ^^^ u := by
        intro he
        subst r
        have ho := winnerVal_orbit (n + 1) l u 0
        rw [Nat.zero_xor, winnerVal_zero_neg (n + 1) l u hl1 hl hu1 hu hne, hPr] at ho
        exact absurd ho (by decide)
      unfold hasXorAnnih
      refine List.any_eq_true.mpr ⟨r, List.mem_range.mpr hr, ?_⟩
      rw [Bool.and_eq_true, Bool.and_eq_true, decide_eq_true_eq,
        decide_eq_true_eq, beq_iff_eq, P_eq_fVal]
      exact ⟨⟨by omega, hrd⟩, hPr⟩
  . intro h
    unfold hasXorAnnih at h
    rw [List.any_eq_true] at h
    obtain ⟨a, ha_mem, hp⟩ := h
    rw [List.mem_range] at ha_mem
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hp
    obtain ⟨⟨ha1, had⟩, hP⟩ := hp
    rw [P_eq_fVal] at hP
    change winnerVal (n + 1) l u a = 1 at hP
    unfold hasXorAnnih
    refine List.any_eq_true.mpr ⟨a, List.mem_range.mpr (by omega), ?_⟩
    rw [Bool.and_eq_true, Bool.and_eq_true, decide_eq_true_eq,
      decide_eq_true_eq, beq_iff_eq, P_eq_fVal]
    refine ⟨⟨ha1, by rwa [hpair]⟩, ?_⟩
    have hs := winner_low_both_high n l u a hl hu ha_mem
    rw [hP] at hs
    exact hs

/-- Every low orbit of the pure high-edge pair `(H,H+m)` loses. -/
theorem winner_low_high_edge_neg (n m r : Nat)
    (hm1 : 1 <= m) (hm : m < 2 ^ (n + 1)) (hr : r < 2 ^ (n + 1)) :
    winnerVal (n + 2) (2 ^ (n + 1)) (2 ^ (n + 1) + m) r = -1 := by
  have hq : r ^^^ m < 2 ^ (n + 1) := Nat.xor_lt_two_pow hr hm
  have hpair : (2 ^ (n + 1)) ^^^ (2 ^ (n + 1) + m) = m := by
    rw [← two_pow_xor_eq_add (n + 1) m hm,
      ← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor]
  have fs1 := fVal_high_stable (n + 1) 0 m r (Nat.two_pow_pos _) hm hr
  have fs2 := fVal_high_stable (n + 1) 0 m (r ^^^ m) (Nat.two_pow_pos _) hm hq
  rw [Nat.add_zero, show n + 1 + 1 = n + 2 by omega] at fs1 fs2
  unfold winnerVal
  rw [hpair, fs1, fs2]
  unfold fVal
  rw [cdSigma_zero_left (n + 1) r (by omega),
    cdSigma_zero_left (n + 1) (r ^^^ m) (by omega), Int.one_mul, Int.one_mul,
    Nat.xor_comm r m]
  exact cdSigma_cocycle' (n + 1) m r hm hr (by omega)

/-- Every high orbit of `(H,H+m)` loses as well. -/
theorem winner_high_high_edge_neg (n m r : Nat)
    (hm1 : 1 <= m) (hm : m < 2 ^ (n + 1)) (hr : r < 2 ^ (n + 1)) :
    winnerVal (n + 2) (2 ^ (n + 1)) (2 ^ (n + 1) + m)
        (2 ^ (n + 1) + r) = -1 := by
  have hq : r ^^^ m < 2 ^ (n + 1) := Nat.xor_lt_two_pow hr hm
  have hpair : (2 ^ (n + 1)) ^^^ (2 ^ (n + 1) + m) = m := by
    rw [← two_pow_xor_eq_add (n + 1) m hm,
      ← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor]
  have hidx : (2 ^ (n + 1) + r) ^^^ m = 2 ^ (n + 1) + (r ^^^ m) := by
    calc
      (2 ^ (n + 1) + r) ^^^ m = (2 ^ (n + 1) ^^^ r) ^^^ m := by
        rw [two_pow_xor_eq_add (n + 1) r hr]
      _ = 2 ^ (n + 1) ^^^ (r ^^^ m) := Nat.xor_assoc _ _ _
      _ = 2 ^ (n + 1) + (r ^^^ m) := two_pow_xor_eq_add (n + 1) _ hq
  by_cases hr0 : r = 0
  . subst r
    have hp0 := cdSigma_hi_pow n 0 (Nat.two_pow_pos (n + 1))
    have hh0m := cdSigma_hi_hi n 0 m (Nat.two_pow_pos (n + 1)) hm1 hm
    rw [Nat.add_zero] at hp0 hh0m
    unfold winnerVal fVal
    rw [hpair, hidx]
    simp only [Nat.zero_xor, Nat.add_zero]
    rw [hp0, cdSigma_hi_pow n m hm,
      hh0m,
      cdSigma_hi_hi n m m hm hm1 hm,
      cdSigma_zero_right (n + 1) m (by omega),
      cdSigma_diag (n + 1) m hm1 hm]
    decide
  by_cases hrm : r = m
  . subst r
    have hp0 := cdSigma_hi_pow n 0 (Nat.two_pow_pos (n + 1))
    have hh0m := cdSigma_hi_hi n 0 m (Nat.two_pow_pos (n + 1)) hm1 hm
    rw [Nat.add_zero] at hp0 hh0m
    unfold winnerVal fVal
    rw [hpair, hidx, Nat.xor_self]
    simp only [Nat.add_zero]
    rw [hh0m,
      cdSigma_hi_hi n m m hm hm1 hm,
      hp0, cdSigma_hi_pow n m hm,
      cdSigma_zero_right (n + 1) m (by omega),
      cdSigma_diag (n + 1) m hm1 hm]
    decide
  have hr1 : 1 <= r := by omega
  have hq0 : r ^^^ m ≠ 0 := fun h => hrm (xor_eq_zero_of r m h)
  have hq1 : 1 <= r ^^^ m := Nat.one_le_iff_ne_zero.mpr hq0
  have hh0r := cdSigma_hi_hi n 0 r (Nat.two_pow_pos (n + 1)) hr1 hr
  have hh0q := cdSigma_hi_hi n 0 (r ^^^ m) (Nat.two_pow_pos (n + 1)) hq1 hq
  rw [Nat.add_zero] at hh0r hh0q
  unfold winnerVal fVal
  rw [hpair, hidx,
    hh0r,
    cdSigma_hi_hi n m r hm hr1 hr,
    hh0q,
    cdSigma_hi_hi n m (r ^^^ m) hm hq1 hq,
    cdSigma_zero_right (n + 1) r (by omega),
    cdSigma_zero_right (n + 1) (r ^^^ m) (by omega)]
  rw [cdAntisym_all (n + 1) r m hr1 hr hm1 hm hrm,
    cdAntisym_all (n + 1) (r ^^^ m) m hq1 hq hm1 hm (by
      intro h
      apply hr0
      apply xor_cancel_right r 0 m
      rw [h, Nat.zero_xor])]
  have hc := cdSigma_cocycle' (n + 1) m r hm hr (by omega)
  rw [Nat.xor_comm m r] at hc
  rcases cdSigma_pm (n + 1) m r with h1 | h1 <;>
  rcases cdSigma_pm (n + 1) m (r ^^^ m) with h2 | h2 <;>
    rw [h1, h2] at hc |- <;> first | decide | exact absurd hc (by decide)

theorem hasXorAnnih_high_edge_false (n m : Nat)
    (hm1 : 1 <= m) (hm : m < 2 ^ (n + 1)) :
    hasXorAnnih (n + 2) (2 ^ (n + 1)) (2 ^ (n + 1) + m) = false := by
  have hpow : 2 ^ (n + 2) = 2 ^ (n + 1) * 2 := by
    rw [show n + 2 = (n + 1) + 1 by omega, Nat.pow_succ]
  rw [Bool.eq_false_iff]
  intro h
  unfold hasXorAnnih at h
  rw [List.any_eq_true] at h
  obtain ⟨a, ha_mem, hp⟩ := h
  rw [List.mem_range] at ha_mem
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hp
  obtain ⟨_, hP⟩ := hp
  rw [P_eq_fVal] at hP
  change winnerVal (n + 2) (2 ^ (n + 1)) (2 ^ (n + 1) + m) a = 1 at hP
  by_cases ha : a < 2 ^ (n + 1)
  . have hn := winner_low_high_edge_neg n m a hm1 hm ha
    rw [hP] at hn
    exact absurd hn (by decide)
  . obtain ⟨r, hr, har⟩ : exists r, r < 2 ^ (n + 1) ∧
        a = 2 ^ (n + 1) + r := ⟨a - 2 ^ (n + 1), by omega, by omega⟩
    subst a
    have hn := winner_high_high_edge_neg n m r hm1 hm hr
    rw [hP] at hn
    exact absurd hn (by decide)

def RecursiveCorrect (bits : Nat) : Prop :=
  forall l u, 1 <= l -> l < 2 ^ bits -> 1 <= u -> u < 2 ^ bits -> l ≠ u ->
    recursiveOffSeam bits l u = hasXorAnnih bits l u

theorem recursiveCorrect_0 : RecursiveCorrect 0 := by
  intro l u hl1 hl hu1 hu hne
  exact eq_of_recursiveAgreement 0 l u recursive_agreement_0 hl1 hl hu1 hu hne

theorem recursiveCorrect_1 : RecursiveCorrect 1 := by
  intro l u hl1 hl hu1 hu hne
  exact eq_of_recursiveAgreement 1 l u recursive_agreement_1 hl1 hl hu1 hu hne

theorem recursiveCorrect_2 : RecursiveCorrect 2 := by
  intro l u hl1 hl hu1 hu hne
  exact eq_of_recursiveAgreement 2 l u recursive_agreement_2 hl1 hl hu1 hu hne

theorem recursiveCorrect_3 : RecursiveCorrect 3 := by
  intro l u hl1 hl hu1 hu hne
  exact eq_of_recursiveAgreement 3 l u recursive_agreement_3 hl1 hl hu1 hu hne

/-- One geometric recursion step. -/
theorem recursiveCorrect_step (n : Nat) (hb : 3 <= n + 1)
    (ih : RecursiveCorrect (n + 1)) : RecursiveCorrect (n + 2) := by
  intro l u hl1 hl hu1 hu hne
  let H := 2 ^ (n + 1)
  have hpow : 2 ^ (n + 2) = H * 2 := by
    dsimp only [H]
    rw [show n + 2 = (n + 1) + 1 by omega, Nat.pow_succ]
  have hguard : ((l == 0) || (u == 0) || (l == u)) = false := by
    rw [beq_eq_false_iff_ne.mpr (by omega : l ≠ 0),
      beq_eq_false_iff_ne.mpr (by omega : u ≠ 0),
      beq_eq_false_iff_ne.mpr hne]
    decide
  have hguardN : ¬ (((l == 0) || (u == 0) || (l == u)) = true) := by
    rw [hguard]
    decide
  have hbase : ¬ n + 1 < 3 := by omega
  by_cases hlH : l < H
  . by_cases huH : u < H
    . have hmodl : l % H = l := Nat.mod_eq_of_lt hlH
      have hmodu : u % H = u := Nat.mod_eq_of_lt huH
      have htag : ((l < H) == (u < H)) = true := by simp [hlH, huH]
      rw [recursiveOffSeam, if_neg hguardN, if_neg hbase, if_pos htag, hmodl, hmodu]
      calc
        recursiveOffSeam (n + 1) l u
            = hasXorAnnih (n + 1) l u := ih l u hl1 hlH hu1 huH hne
        _ = hasXorAnnih (n + 2) l u :=
            (hasXorAnnih_both_low n l u hl1 hlH hu1 huH hne).symm
    . have huHi : H <= u := by omega
      have htag : ((l < H) == (u < H)) = false := by simp [hlH, huH]
      have htagN : ¬ (((l < H) == (u < H)) = true) := by rw [htag]; decide
      rw [recursiveOffSeam, if_neg hguardN, if_neg hbase, if_neg htagN, if_pos hlH]
      exact (seam_coincidence (n + 2) l u (by omega) hl1 hlH huHi hu).2.1.symm
  . have hlHi : H <= l := by omega
    by_cases huH : u < H
    . have htag : ((l < H) == (u < H)) = false := by simp [hlH, huH]
      have htagN : ¬ (((l < H) == (u < H)) = true) := by rw [htag]; decide
      rw [recursiveOffSeam, if_neg hguardN, if_neg hbase, if_neg htagN, if_neg hlH]
      calc
        offSeam (n + 2) u l
            = hasXorAnnih (n + 2) u l :=
                (seam_coincidence (n + 2) u l (by omega) hu1 huH hlHi hl).2.1.symm
        _ = hasXorAnnih (n + 2) l u := (hasXorAnnih_comm (n + 2) l u).symm
    . have huHi : H <= u := by omega
      obtain ⟨lr, hlr, hleq⟩ : exists r, r < H ∧ l = H + r :=
        ⟨l - H, by omega, by omega⟩
      obtain ⟨ur, hur, hueq⟩ : exists r, r < H ∧ u = H + r :=
        ⟨u - H, by omega, by omega⟩
      subst l
      subst u
      have hmodl : (H + lr) % H = lr := by
        rw [Nat.add_mod_left, Nat.mod_eq_of_lt hlr]
      have hmodu : (H + ur) % H = ur := by
        rw [Nat.add_mod_left, Nat.mod_eq_of_lt hur]
      have hHl : ¬ H + lr < H := by omega
      have hHu : ¬ H + ur < H := by omega
      have htag : (((H + lr) < H) == ((H + ur) < H)) = true := by simp [hHl, hHu]
      have hneR : lr ≠ ur := by omega
      rw [recursiveOffSeam, if_neg hguardN, if_neg hbase, if_pos htag, hmodl, hmodu]
      by_cases hlr0 : lr = 0
      . subst lr
        have hur1 : 1 <= ur := by omega
        have hzero : recursiveOffSeam (n + 1) 0 ur = false := by
          have hg : (((0 : Nat) == 0) || (ur == 0) || ((0 : Nat) == ur)) = true := by
            rw [show ((0 : Nat) == 0) = true from rfl, Bool.true_or, Bool.true_or]
          rw [recursiveOffSeam, if_pos hg]
        rw [hzero]
        dsimp only [H]
        exact (hasXorAnnih_high_edge_false n ur hur1 hur).symm
      by_cases hur0 : ur = 0
      . subst ur
        have hlr1 : 1 <= lr := by omega
        have hzero : recursiveOffSeam (n + 1) lr 0 = false := by
          have hg : ((lr == 0) || ((0 : Nat) == 0) || (lr == 0)) = true := by
            rw [show ((0 : Nat) == 0) = true from rfl, Bool.or_true, Bool.true_or]
          rw [recursiveOffSeam, if_pos hg]
        rw [hzero]
        calc
          false = hasXorAnnih (n + 2) H (H + lr) :=
            by
              dsimp only [H]
              exact (hasXorAnnih_high_edge_false n lr hlr1 hlr).symm
          _ = hasXorAnnih (n + 2) (H + lr) H :=
            (hasXorAnnih_comm (n + 2) (H + lr) H).symm
      . have hlr1 : 1 <= lr := by omega
        have hur1 : 1 <= ur := by omega
        calc
          recursiveOffSeam (n + 1) lr ur
              = hasXorAnnih (n + 1) lr ur := ih lr ur hlr1 hlr hur1 hur hneR
          _ = hasXorAnnih (n + 2) (H + lr) (H + ur) := by
            dsimp only [H]
            exact (hasXorAnnih_both_high n lr ur hlr1 hlr hur1 hur hneR).symm

/-- The geometric classification is correct at every level `k+3`. -/
theorem recursiveCorrect_ge3 : forall k, RecursiveCorrect (k + 3)
  | 0 => recursiveCorrect_3
  | k + 1 => recursiveCorrect_step (k + 2) (by omega) (recursiveCorrect_ge3 k)

/-- Full-box geometric/XOR characterization, for every CD level. -/
theorem recursiveOffSeam_eq_hasXorAnnih_full (bits l u : Nat)
    (hl1 : 1 <= l) (hl : l < 2 ^ bits)
    (hu1 : 1 <= u) (hu : u < 2 ^ bits) (hne : l ≠ u) :
    recursiveOffSeam bits l u = hasXorAnnih bits l u := by
  cases bits with
  | zero => exact recursiveCorrect_0 l u hl1 hl hu1 hu hne
  | succ b =>
    cases b with
    | zero => exact recursiveCorrect_1 l u hl1 hl hu1 hu hne
    | succ b =>
      cases b with
      | zero => exact recursiveCorrect_2 l u hl1 hl hu1 hu hne
      | succ k => exact recursiveCorrect_ge3 k l u hl1 hl hu1 hu hne

/-- Full-box geometric zero-divisor characterization. -/
theorem recursiveOffSeam_eq_isZD_full (bits l u : Nat)
    (hl1 : 1 <= l) (hl : l < 2 ^ bits)
    (hu1 : 1 <= u) (hu : u < 2 ^ bits) (hne : l ≠ u) :
    recursiveOffSeam bits l u = isZD bits l u := by
  calc
    recursiveOffSeam bits l u = hasXorAnnih bits l u :=
      recursiveOffSeam_eq_hasXorAnnih_full bits l u hl1 hl hu1 hu hne
    _ = isZD bits l u :=
      (isZD_eq_hasXorAnnih_full bits l u hl1 hl hu1 hu hne).symm

/-- The operator criterion is the complement of the recursive geometry. -/
theorem anti0_eq_not_recursiveOffSeam_full (bits l u : Nat)
    (hl1 : 1 <= l) (hl : l < 2 ^ bits)
    (hu1 : 1 <= u) (hu : u < 2 ^ bits) (hne : l ≠ u) :
    anti0 bits l u = ! recursiveOffSeam bits l u := by
  rw [recursiveOffSeam_eq_isZD_full bits l u hl1 hl hu1 hu hne]
  exact anti0_eq_not_isZD_full bits l u hl1 hl hu1 hu hne

/-- Symmetry of the recursive geometry on its natural domain. -/
theorem recursiveOffSeam_comm (bits l u : Nat)
    (hl1 : 1 <= l) (hl : l < 2 ^ bits)
    (hu1 : 1 <= u) (hu : u < 2 ^ bits) (hne : l ≠ u) :
    recursiveOffSeam bits l u = recursiveOffSeam bits u l := by
  rw [recursiveOffSeam_eq_hasXorAnnih_full bits l u hl1 hl hu1 hu hne,
    recursiveOffSeam_eq_hasXorAnnih_full bits u l hu1 hu hl1 hl (fun h => hne h.symm)]
  exact hasXorAnnih_comm bits l u

/-- On the original lower x upper locus, the recursive predicate is exactly
    the existing local `offSeam`. -/
theorem recursiveOffSeam_eq_offSeam_loHi (bits l u : Nat) (hb : 4 <= bits)
    (hl1 : 1 <= l) (hl : l < 2 ^ (bits - 1))
    (hu1 : 2 ^ (bits - 1) <= u) (hu : u < 2 ^ bits) :
    recursiveOffSeam bits l u = offSeam bits l u := by
  have hle : 2 ^ (bits - 1) <= 2 ^ bits := Nat.pow_le_pow_right (by decide) (by omega)
  have hlt : l < 2 ^ bits := by omega
  have hne : l ≠ u := by omega
  calc
    recursiveOffSeam bits l u = isZD bits l u :=
      recursiveOffSeam_eq_isZD_full bits l u hl1 hlt (by omega) hu hne
    _ = offSeam bits l u := isZD_eq_offSeam bits l u hb hl1 hl hu1 hu

-- Fixed-level executable regressions; the forall chain above does not depend
-- on these native anchors.
theorem recursive_agreement_16 : recursiveAgreement 4 = true := by native_decide
theorem recursive_agreement_32 : recursiveAgreement 5 = true := by native_decide
theorem recursive_agreement_64 : recursiveAgreement 6 = true := by native_decide

end SounioCDRecursiveSeam
