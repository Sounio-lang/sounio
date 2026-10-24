import Std.Tactic

/-! Structural invariants corresponding to numeric_buffer.sio and the ODE
checked contracts. These theorems concern natural-number arithmetic and
abstract state transitions, not IEEE arithmetic, generated code or allocation.
The checked implementation must separately exclude machine overflow. -/
namespace SounioNumericBuffer

theorem affine_index_bounded (rows cols row col rs cs offset bound : Nat)
    (hr : row < rows) (hc : col < cols)
    (hmax : offset + (rows - 1) * rs + (cols - 1) * cs < bound) :
    offset + row * rs + col * cs < bound := by
  have hrow : row ≤ rows - 1 := by omega
  have hcol : col ≤ cols - 1 := by omega
  have hmulrow := Nat.mul_le_mul_right rs hrow
  have hmulcol := Nat.mul_le_mul_right cs hcol
  omega

theorem payload_bytes_bounded (n budget : Nat)
    (h : n ≤ budget / 8) : n * 8 ≤ budget := by
  have hmul := Nat.mul_le_mul_right 8 h
  have hdiv := Nat.div_mul_le_self budget 8
  omega

structure OwnerState where
  size : Nat
  capacity : Nat
  allocation : Nat
deriving DecidableEq

def reserveResult (before : OwnerState) (requested : Nat)
    (allocationSucceeded : Bool) (newAllocation : Nat) : OwnerState :=
  if requested ≤ before.capacity then before
  else if allocationSucceeded then
    { before with capacity := requested, allocation := newAllocation }
  else before

theorem failed_reserve_preserves_owner (before : OwnerState)
    (requested newAllocation : Nat) :
    reserveResult before requested false newAllocation = before := by
  simp [reserveResult]

def acceptedTime (before proposed : Nat) (accepted : Bool) : Nat :=
  if accepted then proposed else before

theorem rejected_step_preserves_time (before proposed : Nat) :
    acceptedTime before proposed false = before := by
  rfl

theorem accepted_time_monotone (before proposed : Nat)
    (h : before < proposed) :
    before < acceptedTime before proposed true := by
  exact h

end SounioNumericBuffer
