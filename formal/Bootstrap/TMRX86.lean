/-
TMRX86: the TMR bounds of `Bootstrap.TMR` and `Bootstrap.TMRExposure`, stated
for the real x86-64 model instead of an abstract channel.

`X86Int.pstep` is already a channel in the sense of `TMR.Channel`: its
zero-fault branch is the architectural `step` (`faultFree_pstep`) and its fault
mass is at most `pf + F.q` (`fm_pstep`), where `pf` is the fetch-fault rate and
`F` the fault model on retired states (`Rates.state`: register, memory and flag
bit flips). `machine` packages it over `Out`, with `exit` and `stuck` absorbing.

With it, `tri_err`, `trun_err` and `exposure_err` become statements about the
X86Int machine: three copies of the whole process state, voted every step, are
wrong at a vote with probability at most `3·(pf + F.q)²`; left unvoted for `k`
steps and then voted, at most `3·((k + 1)·(pf + F.q))²`. The clean outcome the
vote is compared against is the architectural run, `run im (k + 1)`.

What this does not do yet: the per-word projection decision 50 asks for, a
separate channel for one Sporos register or one memory word with that word's
own rate. That needs a coupling lemma (a fault elsewhere, in an address register
say, can write the word), which is not a property of `TMR.Channel`; see the
stage-1 design note.
-/
import Bootstrap.TMRExposure
import Bootstrap.X86Int

namespace TMRX86

open PDist TMR X86Int

/-- One physical step on `Out`: a running state takes a physical step; `exit`
and `stuck` stay where they are. -/
def ostep (alts : Alts) (F : Fault) (pf : Rat) (im : Nat → Option (Ins × Nat)) :
    Out → PDF Out
  | .ok s => pstep alts F pf im s
  | o => pureF o

/-- The architectural step on `Out`, the zero-fault branch of `ostep`. -/
def ofun (im : Nat → Option (Ins × Nat)) : Out → Out
  | .ok s => step im s
  | o => o

/-- The X86Int machine as a TMR channel, at fetch-fault rate `pf` and retire
fault model `F`. -/
def machine (alts : Alts) (F : Fault) {pf : Rat} (hpf : Rate pf)
    (im : Nat → Option (Ins × Nat)) : Channel Out where
  f := ofun im
  step := ostep alts F pf im
  q := pf + F.q
  q0 := nonneg_add hpf.1 F.q0
  ok o := by
    cases o with
    | ok s => exact pstep_ok alts F hpf im s
    | exit c => exact pureF_ok _
    | stuck => exact pureF_ok _
  clean o := by
    cases o with
    | ok s => exact faultFree_pstep alts F pf im s
    | exit c => rfl
    | stuck => rfl
  le o := by
    cases o with
    | ok s => exact fm_pstep alts F hpf im s
    | exit c => show fm (pureF (Out.exit c)) ≤ _; rw [fm_pureF]; exact nonneg_add hpf.1 F.q0
    | stuck => show fm (pureF Out.stuck) ≤ _; rw [fm_pureF]; exact nonneg_add hpf.1 F.q0

/-- The iterate of the channel's clean step is the architectural run. -/
theorem iter_ofun (im : Nat → Option (Ins × Nat)) : ∀ n o, iter (ofun im) n o = run im n o := by
  intro n
  induction n with
  | zero => intro o; rfl
  | succ n ih =>
    intro o
    cases o with
    | ok s => exact ih (step im s)
    | exit c =>
      show iter (ofun im) n (Out.exit c) = Out.exit c
      rw [ih]; cases n <;> rfl
    | stuck =>
      show iter (ofun im) n Out.stuck = Out.stuck
      rw [ih]; cases n <;> rfl

/-- One voted step of three X86Int copies is wrong with probability at most
`3·(pf + F.q)²`. -/
theorem machine_tri_err (alts : Alts) (F : Fault) {pf : Rat} (hpf : Rate pf)
    (im : Nat → Option (Ins × Nat)) (s : St) :
    fm (tri (pstep alts F pf im s) (step im s)) ≤ 3 * ((pf + F.q) * (pf + F.q)) :=
  tri_err (machine alts F hpf im) (.ok s)

/-- `n` steps of three X86Int copies, voted every step: the voted run goes
wrong with probability at most `n·3·(pf + F.q)²`, and its clean branch is the
architectural run. -/
theorem machine_trun_err (alts : Alts) (F : Fault) {pf : Rat} (hpf : Rate pf)
    (im : Nat → Option (Ins × Nat)) (n : Nat) (s : St) :
    fm (trun (machine alts F hpf im) n (some (.ok s))) ≤
      n * (3 * ((pf + F.q) * (pf + F.q))) ∧
    ∀ y ∈ faultFree (trun (machine alts F hpf im) n (some (.ok s))),
      y = some (run im n (.ok s)) := by
  have hf : (machine alts F hpf im).f = ofun im := rfl
  refine ⟨trun_err _ n _, fun y hy => ?_⟩
  rw [trun_clean _ n _ y hy, hf, iter_ofun]

/-- Decision 50 on the X86Int machine: three copies left unvoted for `k` steps
and then voted are wrong at that vote with probability at most
`3·((k + 1)·(pf + F.q))²`; the clean outcome is `run im (k + 1)`. -/
theorem machine_exposure_err (alts : Alts) (F : Fault) {pf : Rat} (hpf : Rate pf)
    (im : Nat → Option (Ins × Nat)) (k : Nat) (s : St) :
    fm (tri (exposed (machine alts F hpf im) k (.ok s)) (run im (k + 1) (.ok s))) ≤
      3 * ((((k + 1 : Nat) : Rat) * (pf + F.q)) * (((k + 1 : Nat) : Rat) * (pf + F.q))) := by
  have hf : (machine alts F hpf im).f = ofun im := rfl
  have h := exposure_err (machine alts F hpf im) k (.ok s)
  rw [hf, iter_ofun] at h
  exact h

/-- At `k = 0` the exposure bound is the voted-every-step bound. -/
example (alts : Alts) (F : Fault) {pf : Rat} (hpf : Rate pf)
    (im : Nat → Option (Ins × Nat)) (s : St) :
    fm (tri (exposed (machine alts F hpf im) 0 (.ok s)) (run im 1 (.ok s))) ≤
      3 * ((pf + F.q) * (pf + F.q)) := by
  have h := machine_exposure_err alts F hpf im 0 s
  simpa [Rat.one_mul] using h

end TMRX86
