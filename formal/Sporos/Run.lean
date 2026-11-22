/-
Whole runs of Sporos, and what the checker guarantees about them (`run_safe`):
a checked program, started on any input, never fails for a reason the checker
rules out, however many steps it takes.
-/
import Sporos.SafeStep

namespace Sporos

/-- Read-only data laid out from address 0, followed by the heap. -/
def loadData (data : List (List Nat)) : Nat → Val := fun a =>
  match (data.flatten)[a]? with
  | some b => ⟨b % 256, []⟩
  | none => ⟨0, []⟩

def init (pr : Prog) (msize : Nat) (inp : List Nat) : St :=
  let locals := match pr.words[pr.main]? with | some m => m.locals | none => 0
  { ds := [], aux := [], rs := [⟨pr.main, 0, List.replicate locals ⟨0, []⟩, 0, 0⟩],
    mem := loadData pr.data, msize := msize, hp := pr.data.flatten.length, tabs := [],
    inp := inp, ipos := 0, out := [], pc := [], step := 0 }

theorem init_inv (pr : Prog) (hp : checkProg pr = true) (msize : Nat) (inp : List Nat) :
    Inv pr (init pr msize inp) := by
  have hp' := hp
  simp only [checkProg, Bool.and_eq_true] at hp'
  cases hm : pr.words[pr.main]? with
  | none => rw [hm] at hp'; simp at hp'
  | some m =>
    rw [hm] at hp'
    have h0 : m.ins = 0 := by simpa using hp'.2
    obtain ⟨hpos, hent⟩ := checkWord_entry (checkProg_word hp hm)
    unfold Inv init
    simp only [hm]
    refine ⟨⟨m, ⟨m.ins, 0⟩, hm, hent, hpos, by simp [h0], rfl, by simp⟩, trivial⟩

/-- `n` steps, or the end of the run. -/
def run (pr : Prog) (ph : Phys) : Nat → St → Res
  | 0, s => .next s
  | n + 1, s =>
    match step pr ph s with
    | .next t => run pr ph n t
    | r => r

theorem run_safe_from (pr : Prog) (ph : Phys) (hp : checkProg pr = true) :
    ∀ (n : Nat) (s : St), Inv pr s → Safe pr (run pr ph n s) := by
  intro n
  induction n with
  | zero => intro s h; exact h
  | succ n ih =>
    intro s h
    have hs := step_safe pr ph hp s h
    simp only [run]
    revert hs
    cases step pr ph s with
    | next t => intro hs; exact ih t hs
    | halt c o => intro _; trivial
    | fail e => intro hs; exact hs

/-- A checked program never underflows a stack, reads a bad local, jumps outside
its word or calls a missing word, on any input, at any step. -/
theorem run_safe (pr : Prog) (ph : Phys) (hp : checkProg pr = true) (msize : Nat) (inp : List Nat)
    (n : Nat) (e : Err) (h : run pr ph n (init pr msize inp) = .fail e) : e.static = false := by
  have := run_safe_from pr ph hp n _ (init_inv pr hp msize inp)
  rw [h] at this; exact this

/-! ### A first program: copy input to output -/

/-- `echo`: `getb dup -1 eq jnz 7 putb jmp 0` then `drop 0 exit`. -/
def echo : Prog where
  words := [{
    ins := 0, outs := 0, locals := 0,
    code := [.prim .getb, .prim .dup, .lit (M - 1), .prim .eq, .jnz 7, .prim .putb, .jmp 0,
             .prim .drop, .lit 0, .prim .exit],
    ann := [(0, ⟨0, 0⟩), (7, ⟨1, 0⟩)] }]
  data := []
  main := 0

theorem echo_checks : checkProg echo = true := by decide

def noPhys : Phys := ⟨fun _ => 0, fun _ => 0, fun _ _ => 0, fun _ => 0⟩

/-- On input "hi", echo writes "hi" and exits 0; byte `h` carries the origin of
input position 0. -/
theorem echo_hi :
    (match run echo noPhys 40 (init echo 0 [104, 105]) with
      | .halt c o => c == 0 && o.reverse.map Prod.fst == [104, 105] &&
          (o.reverse.head?.map fun x => x.2.any fun g => g.inp == some 0) == some true
      | _ => false) = true := by decide

end Sporos
