/-
TMRExposure: the exposure-weighted vote bound asked for by decision 50 of the
root-A stage-1 design, and the precise statement that decision 48 left owing.

`TMR.tri_err` bounds one voted step: vote every step, and a word is wrong with
probability at most `3·q²`. Decision 48 does not vote every word every step —
it votes only what the step wrote, so a word the interpreter merely reads is
left unvoted, and the silicon can still corrupt it (`X86Int.memFlips` hits any
mapped byte, not only the one the current instruction targets). Decision 50
says the bound must then carry how long the word went unwritten.

`idleK C k s`: one copy's distribution after `k` steps of exposure — the
channel's noisy step applied `k` times, with no vote in between. `exposure_err`
is the generalisation of `tri_err` this needs: a word idle for `k` steps and
then written is wrong, at the write's vote, with probability at most
`3·((k+1)·q)²`. At `k = 0` (voted every step) this is exactly `tri_err`'s
`3·q²`; the constant grows with the square of how long the word lived between
writes, matching the Σ_words (exposure_w · p)² shape decision 50 asked for.

What this module does not yet do: instantiate `Channel` with the real per-word
projection of `X86Int.St` (one channel for a Sporos register, one for a memory
byte, under `Rates.reg`/`Rates.mem`), and account for the counter that tracks
`exposure_w` itself running on the same faulty silicon (the risk decision 50
flagged). Both are owed before this is a statement about the interpreter
rather than about an abstract `Channel`.
-/
import Bootstrap.TMR

namespace TMR

open PDist

variable {σ : Type} [DecidableEq σ]

/-- One copy, left unvoted for `k` steps: the channel's noisy step composed
`k` times, with no correction in between. -/
def idleK (C : Channel σ) : Nat → σ → PDF σ
  | 0, s => pureF s
  | k + 1, s => bindF (C.step s) (idleK C k)

/-- `iter` from the other end: one more application lands on the outside,
not just the inside. Needed because `exposed` applies the channel's step
after the idle run, so its clean value is reached from the outside. -/
theorem iter_succ (f : σ → σ) : ∀ n s, iter f (n + 1) s = f (iter f n s) := by
  intro n
  induction n with
  | zero => intro s; rfl
  | succ n ih =>
    intro s
    show iter f (n + 1) (f s) = f (iter f (n + 1) s)
    rw [ih (f s)]
    rfl

theorem idleK_ok (C : Channel σ) : ∀ k s, (idleK C k s).ok := by
  intro k
  induction k with
  | zero => intro s; exact pureF_ok s
  | succ k ih => intro s; exact bindF_ok (C.ok s) ih

/-- Without any fault, `k` idle steps land exactly on the architectural `k`-fold
iterate — the same `f` the vote would have certified at every step. -/
theorem faultFree_idleK (C : Channel σ) : ∀ k s, faultFree (idleK C k s) = [iter C.f k s] := by
  intro k
  induction k with
  | zero => intro s; rfl
  | succ k ih =>
    intro s
    show faultFree (bindF (C.step s) (idleK C k)) = [iter C.f (k + 1) s]
    rw [faultFree_bindF, C.clean]
    simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, ih]
    rfl

/-- `k` idle steps fault with probability at most `k·q`: faults accumulate
additively with no vote to reset them, exactly as an unhardened run does
(`X86Int.baseline`). -/
theorem fm_idleK_le (C : Channel σ) : ∀ k s, fm (idleK C k s) ≤ (k : Rat) * C.q := by
  intro k
  induction k with
  | zero =>
    intro s
    show fm (pureF s) ≤ ((0 : Nat) : Rat) * C.q
    rw [fm_pureF, show ((0 : Nat) : Rat) = 0 from rfl, Rat.zero_mul]
    exact Rat.le_refl
  | succ k ih =>
    intro s
    have hk : 0 ≤ (k : Rat) * C.q := Rat.mul_nonneg Rat.natCast_nonneg C.q0
    have h1 : fm (bindF (C.step s) (idleK C k)) ≤ fm (C.step s) + (k : Rat) * C.q :=
      fm_bindF_le' (C.ok s) hk (idleK_ok C k) ih
    have h2 : fm (bindF (C.step s) (idleK C k)) ≤ C.q + (k : Rat) * C.q :=
      Rat.le_trans h1 (add_le_add' (C.le s) Rat.le_refl)
    have hn : (((k + 1 : Nat) : Rat)) * C.q = C.q + (k : Rat) * C.q := by
      rw [Rat.natCast_add, Rat.add_mul, show ((1 : Nat) : Rat) = 1 from rfl, Rat.one_mul, Rat.add_comm]
    rw [hn]; exact h2

/-- One copy's distribution after `k` idle steps followed by the write that
finally touches the word. Its clean branch is the correct value after `k + 1`
architectural steps; the point of the lemma below is its fault mass. -/
def exposed (C : Channel σ) (k : Nat) (s : σ) : PDF σ := bindF (idleK C k s) C.step

theorem exposed_ok (C : Channel σ) (k : Nat) (s : σ) : (exposed C k s).ok :=
  bindF_ok (idleK_ok C k s) C.ok

theorem faultFree_exposed (C : Channel σ) (k : Nat) (s : σ) :
    faultFree (exposed C k s) = [iter C.f (k + 1) s] := by
  show faultFree (bindF (idleK C k s) C.step) = [iter C.f (k + 1) s]
  rw [faultFree_bindF, faultFree_idleK]
  simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, C.clean]
  rw [iter_succ C.f k s]

theorem fm_exposed_le (C : Channel σ) (k : Nat) (s : σ) :
    fm (exposed C k s) ≤ ((k + 1 : Nat) : Rat) * C.q := by
  have h0 : 0 ≤ C.q := C.q0
  have h1 : fm (exposed C k s) ≤ fm (idleK C k s) + C.q :=
    fm_bindF_le' (idleK_ok C k s) h0 C.ok C.le
  have h2 : fm (exposed C k s) ≤ (k : Rat) * C.q + C.q := Rat.le_trans h1 (add_le_add' (fm_idleK_le C k s) Rat.le_refl)
  have hn : (((k + 1 : Nat) : Rat)) * C.q = (k : Rat) * C.q + C.q := by
    rw [Rat.natCast_add, Rat.add_mul, show ((1 : Nat) : Rat) = 1 from rfl, Rat.one_mul]
  rw [hn]; exact h2

/-- **Decision 50, made precise.** A word left unwritten (and therefore
unvoted) for `k` steps, then written: the vote at that write fails to correct
it with probability at most `3·((k+1)·q)²`, where `q` bounds a single copy's
per-step fault rate. At `k = 0` — voted every step — this is `tri_err`'s
`3·q²` exactly. The constant is the square of the word's own exposure count,
the number decision 50 asks the interpreter to report alongside its output. -/
theorem exposure_err (C : Channel σ) (k : Nat) (s : σ) :
    fm (tri (exposed C k s) (iter C.f (k + 1) s)) ≤
      3 * ((((k + 1 : Nat) : Rat) * C.q) * (((k + 1 : Nat) : Rat) * C.q)) := by
  have h0 := fm_nonneg (exposed_ok C k s).1
  refine Rat.le_trans (fm_tri_le (exposed_ok C k s) (faultFree_exposed C k s)) ?_
  have := sq_le h0 (fm_exposed_le C k s)
  grind

/-- `exposure_err` at zero exposure is exactly `tri_err`. -/
theorem exposure_err_zero (C : Channel σ) (s : σ) :
    fm (tri (exposed C 0 s) (iter C.f 1 s)) ≤ 3 * (C.q * C.q) := by
  have h := exposure_err C 0 s
  have hc : (((0 + 1 : Nat) : Rat)) * C.q = C.q := by
    rw [show ((0 + 1 : Nat) : Rat) = 1 from rfl, Rat.one_mul]
  rwa [hc] at h

end TMR
