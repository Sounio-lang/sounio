/-!
Two-mode solution of the existing ZD difference transfer.
Pure integer algebra; the concrete Cayley-Dickson bridge is a separate module.
This is a corollary of the existing transfer, not a novelty or full-spectrum claim.
-/
namespace Sounio.ZDTwoMode

def Transfer (D C : Nat → Int) : Prop :=
  ∀ n, D (n+1) = 8 * D n + 24 * C n ∧ C (n+1) = 4 * C n

theorem modes (D C : Nat → Int) (h : Transfer D C) (n : Nat) :
    C n = (4 : Int)^n * C 0 ∧
    D n + 6 * C n = (8 : Int)^n * (D 0 + 6 * C 0) := by
  induction n with
  | zero => simp
  | succ n ih =>
    obtain ⟨hd, hc⟩ := h n
    constructor
    · rw [hc, ih.1, Int.pow_succ]; grind
    · rw [hd, hc, Int.pow_succ]
      have hh := ih.2
      grind

theorem closed (D C : Nat → Int) (h : Transfer D C) (n : Nat) :
    D n = (8 : Int)^n * (D 0 + 6 * C 0) - 6 * (4 : Int)^n * C 0 := by
  obtain ⟨hc, hd⟩ := modes D C h n
  rw [hc] at hd
  grind

theorem homogeneous_of_initial_eq (D C : Nat → Int) (h : Transfer D C)
    (hzero : C 0 = 0) (n : Nat) : C n = 0 ∧ D n = (8 : Int)^n * D 0 := by
  obtain ⟨hc, hd⟩ := modes D C h n
  simp only [hzero, Int.mul_zero, Int.add_zero] at hc hd
  rw [hc] at hd
  simp only [Int.mul_zero, Int.add_zero] at hd
  exact ⟨hc, hd⟩

/-- Observing D at two consecutive levels determines the initial C exactly.
No integer division is needed, so the statement has no rounding convention. -/
theorem recover_initial (D C : Nat → Int) (h : Transfer D C) :
    24 * C 0 = D 1 - 8 * D 0 := by
  have hd := (h 0).1
  simp only [Nat.zero_add] at hd
  omega

/-- Pure eightfold scaling at the FIRST step already forces the initial C to vanish. -/
theorem homogeneous_iff (D C : Nat → Int) (h : Transfer D C) :
    D 1 = 8 * D 0 ↔ C 0 = 0 := by
  have hd := (h 0).1
  simp only [Nat.zero_add] at hd
  constructor <;> intro hz <;> omega

#print axioms modes
#print axioms closed
#print axioms homogeneous_of_initial_eq
#print axioms recover_initial
#print axioms homogeneous_iff
end Sounio.ZDTwoMode
