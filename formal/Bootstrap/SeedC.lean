/-
seed-c on every listing. For every input `L` of bytes (`∀ c ∈ L, c < 256`),

  (∃ F, exec F seedC L = .exit 0 bs) ↔ BinC L bs     (`seedC_correct`)

and `seed-c` exits 1 on every listing that is not valid format C (`seedC_refuses`).

Proof structure:
  - `seed-c` holds its loop-head state in `x19` (accumulator), `x20` (digit
    counter 0..7), and `x21` (comment flag 0/1).
  - At `n = 0` (`com = false` and `com = true`), `acc = 0`: checked over all 256
    input bytes via `decide +kernel`.
  - At `1 ≤ n ≤ 7` (`com = false`):
    - Any byte outside `{'0', '1', ':', ' ', '\n', '\t'}` is refused without
      reading either `x19` or `x20`; one row (`iter_nonsep`) covers all `n` and `a`.
    - Separators `[':', ' ', '\n', '\t']` are refused without reading `x19`;
      one compact check (`iter_sep`, 7 × 4 cases) covers all `a`.
    - Only `'0'` (48) and `'1'` (49) read `x19`, checked for every `a < 2^n`
      via `decide +kernel` (`iter_bits_1`..`iter_bits_7`, 508 cases total).
-/
import Bootstrap.AArch64Min
import Bootstrap.FormatsC

open AArch64Min Bootstrap.FormatsC

namespace AArch64Min.SeedC

def seedC_0 : List Nat := [127, 69, 76, 70, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 183, 0, 1, 0, 0, 0, 120, 0, 64, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 56, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 84, 1, 0, 0, 0, 0, 0, 0, 84, 1, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 19, 0, 128, 210, 20, 0, 128, 210, 21, 0, 128, 210, 232, 7, 128, 210, 0, 0, 128, 210, 225, 3, 0, 145, 34, 0, 128, 210, 1, 0, 0, 212, 31, 0, 0, 241, 107, 5, 0, 84, 192, 4, 0, 84, 224, 3, 64, 57, 181, 0, 0, 180, 31, 40, 0, 113, 161, 254, 255, 84, 21, 0, 128, 210, 243, 255, 255, 23, 31, 232, 0, 113, 32, 3, 0, 84, 31, 128, 0, 113, 224, 2, 0, 84, 31, 40, 0, 113, 160, 2, 0, 84, 31, 36, 0, 113, 96, 2, 0, 84, 0, 192, 0, 81, 31, 4, 0, 113, 40, 3, 0, 84, 19, 4, 19, 139, 148, 6, 0, 145, 159, 34, 0, 241, 129, 252, 255, 84, 243, 3, 0, 57, 8, 8, 128, 210]
def seedC_1 : List Nat := [32, 0, 128, 210, 225, 3, 0, 145, 34, 0, 128, 210, 1, 0, 0, 212, 31, 4, 0, 241, 161, 1, 0, 84, 19, 0, 128, 210, 20, 0, 128, 210, 217, 255, 255, 23, 52, 1, 0, 181, 31, 232, 0, 113, 193, 250, 255, 84, 53, 0, 128, 210, 212, 255, 255, 23, 148, 0, 0, 181, 168, 11, 128, 210, 0, 0, 128, 210, 1, 0, 0, 212, 168, 11, 128, 210, 32, 0, 128, 210, 1, 0, 0, 212]
def seedC : List Nat := seedC_0 ++ seedC_1

private def noneT1 : Tree (Option Ins) := .node (.leaf none) (.leaf none)
private def noneT2 : Tree (Option Ins) := .node noneT1 noneT1
private def noneT3 : Tree (Option Ins) := .node noneT2 noneT2
private def noneT4 : Tree (Option Ins) := .node noneT3 noneT3
private def noneT5 : Tree (Option Ins) := .node noneT4 noneT4
private def noneT6 : Tree (Option Ins) := .node noneT5 noneT5
private def noneT7 : Tree (Option Ins) := .node noneT6 noneT6

private def imCTree : Tree (Option Ins) :=
  Tree.node (Tree.node (Tree.node (Tree.node noneT4 (Tree.node noneT3 (Tree.node noneT2 (Tree.node noneT1 (Tree.node (Tree.leaf (some (Ins.movWide 2 64 19 0 0))) (Tree.leaf (some (Ins.movWide 2 64 20 0 0)))))))) (Tree.node (Tree.node (Tree.node (Tree.node (Tree.node (Tree.leaf (some (Ins.movWide 2 64 21 0 0))) (Tree.leaf (some (Ins.movWide 2 64 8 63 0)))) (Tree.node (Tree.leaf (some (Ins.movWide 2 64 0 0 0))) (Tree.leaf (some (Ins.addSubImm false false 64 1 31 0))))) (Tree.node (Tree.node (Tree.leaf (some (Ins.movWide 2 64 2 1 0))) (Tree.leaf (some (Ins.svc 0)))) (Tree.node (Tree.leaf (some (Ins.addSubImm true true 64 31 0 0))) (Tree.leaf (some (Ins.bcond 11 172)))))) (Tree.node (Tree.node (Tree.node (Tree.leaf (some (Ins.bcond 0 152))) (Tree.leaf (some (Ins.ldr (MemSize.b1) 0 31 0)))) (Tree.node (Tree.leaf (some (Ins.cbz false 64 21 20))) (Tree.leaf (some (Ins.addSubImm true true 32 31 0 10))))) (Tree.node (Tree.node (Tree.leaf (some (Ins.bcond 1 18446744073709551572))) (Tree.leaf (some (Ins.movWide 2 64 21 0 0)))) (Tree.node (Tree.leaf (some (Ins.b false 18446744073709551564))) (Tree.leaf (some (Ins.addSubImm true true 32 31 0 58))))))) (Tree.node (Tree.node (Tree.node (Tree.node (Tree.leaf (some (Ins.bcond 0 100))) (Tree.leaf (some (Ins.addSubImm true true 32 31 0 32)))) (Tree.node (Tree.leaf (some (Ins.bcond 0 92))) (Tree.leaf (some (Ins.addSubImm true true 32 31 0 10))))) (Tree.node (Tree.node (Tree.leaf (some (Ins.bcond 0 84))) (Tree.leaf (some (Ins.addSubImm true true 32 31 0 9)))) (Tree.node (Tree.leaf (some (Ins.bcond 0 76))) (Tree.leaf (some (Ins.addSubImm true false 32 0 0 48)))))) (Tree.node (Tree.node (Tree.node (Tree.leaf (some (Ins.addSubImm true true 32 31 0 1))) (Tree.leaf (some (Ins.bcond 8 100)))) (Tree.node (Tree.leaf (some (Ins.addSubReg false false 64 19 0 19 (ShiftKind.lsl) 1))) (Tree.leaf (some (Ins.addSubImm false false 64 20 20 1))))) (Tree.node (Tree.node (Tree.leaf (some (Ins.addSubImm true true 64 31 20 8))) (Tree.leaf (some (Ins.bcond 1 18446744073709551504)))) (Tree.node (Tree.leaf (some (Ins.str (MemSize.b1) 19 31 0))) (Tree.leaf (some (Ins.movWide 2 64 8 64 0))))))))) (Tree.node (Tree.node (Tree.node (Tree.node (Tree.node (Tree.node (Tree.leaf (some (Ins.movWide 2 64 0 1 0))) (Tree.leaf (some (Ins.addSubImm false false 64 1 31 0)))) (Tree.node (Tree.leaf (some (Ins.movWide 2 64 2 1 0))) (Tree.leaf (some (Ins.svc 0))))) (Tree.node (Tree.node (Tree.leaf (some (Ins.addSubImm true true 64 31 0 1))) (Tree.leaf (some (Ins.bcond 1 52)))) (Tree.node (Tree.leaf (some (Ins.movWide 2 64 19 0 0))) (Tree.leaf (some (Ins.movWide 2 64 20 0 0)))))) (Tree.node (Tree.node (Tree.node (Tree.leaf (some (Ins.b false 18446744073709551460))) (Tree.leaf (some (Ins.cbz true 64 20 36)))) (Tree.node (Tree.leaf (some (Ins.addSubImm true true 32 31 0 58))) (Tree.leaf (some (Ins.bcond 1 18446744073709551448))))) (Tree.node (Tree.node (Tree.leaf (some (Ins.movWide 2 64 21 1 0))) (Tree.leaf (some (Ins.b false 18446744073709551440)))) (Tree.node (Tree.leaf (some (Ins.cbz true 64 20 16))) (Tree.leaf (some (Ins.movWide 2 64 8 93 0))))))) (Tree.node (Tree.node (Tree.node (Tree.node (Tree.leaf (some (Ins.movWide 2 64 0 0 0))) (Tree.leaf (some (Ins.svc 0)))) (Tree.node (Tree.leaf (some (Ins.movWide 2 64 8 93 0))) (Tree.leaf (some (Ins.movWide 2 64 0 1 0))))) (Tree.node (Tree.node (Tree.leaf (some (Ins.svc 0))) (Tree.leaf none)) noneT1)) noneT3)) noneT5)) noneT7

def imC : Img := ⟨seedC, 0x400000, 340, imCTree⟩

def sp : Nat := 0x7ffffffde000

private def spLo : Nat := 0x7ffffffddfe0

private def none64 : List (Option Nat) := [none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none]

private def none31 : List (Option Nat) := [none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none]

def s0 : St :=
  { regs := none31, sp := some sp, pc := 0x400078, fl := .undef,
    mem := none64, lo := spLo, inp := [], out := [] }

theorem load_eq : load seedC sp [] = some (imC, s0) := by decide +kernel

/-- `loop:` in `seed-c.bin` (offset 0x084). -/
def head : Nat := 0x400084

def mkAbs (acc n com : Option Nat) : St :=
  { regs := [none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, none, acc, n, com, none, none, none, none, none, none, none, none, none],
    sp := some sp, pc := head, fl := .undef,
    mem := none64, lo := spLo, inp := [], out := [] }

def absC (com : Bool) (n : Nat) (acc : Option Nat) : St :=
  mkAbs acc (some n) (some (if com then 1 else 0))

def absCMid (n acc : Option Nat) : St :=
  mkAbs acc n (some 0)

def stepC (com : Bool) (n a c : Nat) : Option (Bool × Nat × Nat × List Nat) :=
  if com then some (c != 10, n, a, [])
  else if (c = 58 || isSep c) = true then
    (if n ≠ 0 then none else if c = 58 then some (true, 0, a, []) else some (false, 0, a, []))
  else match binVal c with
    | none => none
    | some v =>
      if n + 1 = 8 then some (false, 0, 0, [a * 2 + v])
      else some (false, n + 1, a * 2 + v, [])

def iterCheck (R : Sum St Outcome) (com : Bool) (n a c : Nat) : Bool :=
  match R with
  | .inl t => match stepC com n a c with
    | some (com', n', a', e) => leB [] e (absC com' n' (some a')) t
    | none => false
  | .inr (.exit st _) => st == 1 && (stepC com n a c).isNone
  | _ => false

def iterOK (com : Bool) (n a c : Nat) : Bool :=
  iterCheck (runU true imC head 80 { absC com n (some a) with inp := [c] }) com n a c

def iterBitPair (n a : Nat) : Bool :=
  iterOK false n a 48 && iterOK false n a 49


def exit1Check (R : Sum St Outcome) : Bool :=
  match R with
  | .inr (.exit st x) => st == 1 && x == []
  | _ => false

def iterNonSepOK (c : Nat) : Bool :=
  if c == 48 || c == 49 || c == 58 || isSep c then true
  else exit1Check (runU true imC head 80 { absCMid none none with inp := [c] })

def iterSepOK (n c : Nat) : Bool :=
  exit1Check (runU true imC head 80 { absCMid (some n) none with inp := [c] })

def eofCheck (R : Sum St Outcome) (n : Nat) : Bool :=
  match R with
  | .inr (.exit st x) => x == [] && st == (if n = 0 then 0 else 1)
  | _ => false

def eofOK (com : Bool) (n : Nat) (acc : Option Nat) : Bool :=
  eofCheck (runU false imC head 80 (absC com n acc)) n

def eofMidOK (n : Nat) : Bool :=
  eofCheck (runU false imC head 80 (absCMid (some n) none)) n

def entryCheck (R : Sum St Outcome) : Bool :=
  match R with
  | .inl t => leB [] [] (absC false 0 (some 0)) t
  | _ => false

def entryOK : Bool := entryCheck (runU true imC head 10 s0)

def validC (com : Bool) (n a : Nat) : Prop :=
  (n = 0 ∧ a = 0) ∨ (com = false ∧ 1 ≤ n ∧ n ≤ 7 ∧ a < 2 ^ n)

/-! ### Kernel enumeration (13 compact declarations) -/

theorem iter_false_0 : (List.range 256).all (iterOK false 0 0) = true := by decide +kernel
theorem iter_true_0 : (List.range 256).all (iterOK true 0 0) = true := by decide +kernel
theorem iter_nonsep : (List.range 256).all iterNonSepOK = true := by decide +kernel
theorem iter_sep : ([1, 2, 3, 4, 5, 6, 7].all fun n => [58, 32, 10, 9].all (iterSepOK n)) = true := by
  decide +kernel

theorem iter_bits_1 : (List.range 2).all (iterBitPair 1) = true := by decide +kernel
theorem iter_bits_2 : (List.range 4).all (iterBitPair 2) = true := by decide +kernel
theorem iter_bits_3 : (List.range 8).all (iterBitPair 3) = true := by decide +kernel
theorem iter_bits_4 : (List.range 16).all (iterBitPair 4) = true := by decide +kernel
theorem iter_bits_5 : (List.range 32).all (iterBitPair 5) = true := by decide +kernel
theorem iter_bits_6 : (List.range 64).all (iterBitPair 6) = true := by decide +kernel
theorem iter_bits_7 : (List.range 128).all (iterBitPair 7) = true := by decide +kernel

theorem eof_facts :
    (eofOK false 0 (some 0) && eofOK true 0 (some 0) &&
     ([1, 2, 3, 4, 5, 6, 7].all eofMidOK)) = true := by decide +kernel

theorem entry_ok : entryOK = true := by decide +kernel

/-! ### Lifting to all inputs -/

theorem mkAbs_mono {a1 a2 n1 n2 c1 c2 : Option Nat}
    (ha : OLe a1 a2) (hn : OLe n1 n2) (hc : OLe c1 c2) :
    StLe [] [] (mkAbs a1 n1 c1) (mkAbs a2 n2 c2) where
  reg i := by
    match i with
    | 19 => exact ha
    | 20 => exact hn
    | 21 => exact hc
    | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18
    | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | _ + 31 => exact OLe.refl _
  rlen := rfl
  sp := OLe.refl _
  pc := rfl
  fl := FlLe.refl _
  byte _ := OLe.refl _
  mlen := rfl
  lo := rfl
  inp := rfl
  out := rfl

theorem absC_weaken_acc {r o : List Nat} {n a : Nat} {s : St}
    (h : StLe r o (absC false n (some a)) s) : StLe r o (absCMid (some n) none) s := by
  have hm : StLe [] [] (absCMid (some n) none) (absC false n (some a)) :=
    mkAbs_mono (OLe.none _) (OLe.refl _) (OLe.refl _)
  simpa using StLe.trans hm h

theorem absC_weaken_both {r o : List Nat} {n a : Nat} {s : St}
    (h : StLe r o (absC false n (some a)) s) : StLe r o (absCMid none none) s := by
  have hm : StLe [] [] (absCMid none none) (absC false n (some a)) :=
    mkAbs_mono (OLe.none _) (OLe.none _) (OLe.refl _)
  simpa using StLe.trans hm h

theorem iter_zero_all (com : Bool) (c : Nat) (hc : c < 256) : iterOK com 0 0 c = true := by
  cases com
  · exact List.all_eq_true.mp iter_false_0 c (List.mem_range.mpr hc)
  · exact List.all_eq_true.mp iter_true_0 c (List.mem_range.mpr hc)

theorem iter_nonsep_all (c : Nat) (hc : c < 256) : iterNonSepOK c = true :=
  List.all_eq_true.mp iter_nonsep c (List.mem_range.mpr hc)

theorem iter_sep_all (n c : Nat) (hn1 : 1 ≤ n) (hn7 : n ≤ 7) (hc : c = 58 ∨ isSep c = true) :
    iterSepOK n c = true := by
  have h1 := List.all_eq_true.mp iter_sep n (by
    match n, hn1, hn7 with
    | 1, _, _ | 2, _, _ | 3, _, _ | 4, _, _ | 5, _, _ | 6, _, _ | 7, _, _ => simp
    | _ + 8, _, _ | 0, _, _ => omega)
  have hcm : c ∈ [58, 32, 10, 9] := by
    rcases hc with rfl | hs
    · simp
    · unfold isSep at hs; simp at hs; rcases hs with (rfl | rfl) | rfl <;> simp
  exact List.all_eq_true.mp h1 c hcm

theorem iter_bit_all (n a : Nat) (hn1 : 1 ≤ n) (hn7 : n ≤ 7) (ha : a < 2 ^ n)
    {c : Nat} (hc : c = 48 ∨ c = 49) : iterOK false n a c = true := by
  have hmem := List.mem_range.mpr ha
  have hpair : iterBitPair n a = true := by
    match n, hn1, hn7 with
    | 1, _, _ => exact List.all_eq_true.mp iter_bits_1 a hmem
    | 2, _, _ => exact List.all_eq_true.mp iter_bits_2 a hmem
    | 3, _, _ => exact List.all_eq_true.mp iter_bits_3 a hmem
    | 4, _, _ => exact List.all_eq_true.mp iter_bits_4 a hmem
    | 5, _, _ => exact List.all_eq_true.mp iter_bits_5 a hmem
    | 6, _, _ => exact List.all_eq_true.mp iter_bits_6 a hmem
    | 7, _, _ => exact List.all_eq_true.mp iter_bits_7 a hmem
  unfold iterBitPair at hpair
  simp only [Bool.and_eq_true] at hpair
  rcases hc with rfl | rfl
  · exact hpair.1
  · exact hpair.2

theorem eof_zero_all (com : Bool) : eofOK com 0 (some 0) = true := by
  have h := eof_facts
  simp only [Bool.and_eq_true] at h
  cases com
  · exact h.1.1
  · exact h.1.2

theorem eof_mid_all (n : Nat) (hn1 : 1 ≤ n) (hn7 : n ≤ 7) : eofMidOK n = true := by
  have h := eof_facts
  simp only [Bool.and_eq_true] at h
  have hnmem : n ∈ [1, 2, 3, 4, 5, 6, 7] := by
    match n, hn1, hn7 with
    | 1, _, _ | 2, _, _ | 3, _, _ | 4, _, _ | 5, _, _ | 6, _, _ | 7, _, _ => simp
    | _ + 8, _, _ | 0, _, _ => omega
  exact List.all_eq_true.mp h.2 n hnmem

theorem binVal_le {c v : Nat} (h : binVal c = some v) : v ≤ 1 := by
  unfold binVal at h
  split at h
  · cases h; omega
  · split at h
    · cases h; omega
    · cases h

theorem stepC_valid {com : Bool} {n a c : Nat} {com' : Bool} {n' a' : Nat} {e : List Nat}
    (hs : stepC com n a c = some (com', n', a', e)) (hv : validC com n a) : validC com' n' a' := by
  unfold stepC at hs
  split at hs
  · cases hs
    rcases hv with ⟨hn, ha⟩ | ⟨hc, _⟩
    · exact .inl ⟨hn, ha⟩
    · subst hc; contradiction
  split at hs
  · split at hs
    · cases hs
    · rename_i hn
      have ha0 : a = 0 := by
        rcases hv with ⟨_, ha⟩ | ⟨_, hn1, _⟩
        · exact ha
        · omega
      split at hs <;> (cases hs; exact .inl ⟨rfl, ha0⟩)
  · rename_i hcom _
    have hc : com = false := by cases com <;> simp_all
    cases hx : binVal c with
    | none => rw [hx] at hs; cases hs
    | some v =>
      rw [hx] at hs; dsimp only at hs
      have hv1 := binVal_le hx
      split at hs
      · cases hs; exact .inl ⟨rfl, rfl⟩
      · cases hs
        rename_i h8
        rcases hv with ⟨hn, ha⟩ | ⟨_, hn1, hn7, ha⟩
        · subst hn ha
          exact .inr ⟨rfl, by omega, by omega, by simp; omega⟩
        · have hlt : n + 1 < 8 := by omega
          have hp : 2 ^ (n + 1) = 2 ^ n * 2 := Nat.pow_succ ..
          exact .inr ⟨rfl, by omega, by omega, by omega⟩

theorem decC_cons (c : Nat) (l : List Nat) (com : Bool) (n a : Nat) (acc : List Nat) :
    decC (c :: l) com n a acc = match stepC com n a c with
      | some (com', n', a', e) => decC l com' n' a' (e ++ acc)
      | none => none := by
  cases com
  · simp only [decC, stepC, Bool.false_eq_true, if_false]
    split
    · split
      · rfl
      · split <;> rfl
    · cases binVal c with
      | none => rfl
      | some v => dsimp only; split <;> rfl
  · simp only [decC, stepC, if_true]
    by_cases h10 : c = 10
    · simp [h10]
    · have : (c != 10) = true := by simp [h10]
      simp [h10, this]

theorem stepC_nonbit_none {n a c : Nat} (hn : 1 ≤ n) (hc48 : c ≠ 48) (hc49 : c ≠ 49) :
    stepC false n a c = none := by
  unfold stepC
  simp only [Bool.false_eq_true, if_false]
  split
  · have hn0 : n ≠ 0 := by omega
    simp [hn0]
  · have hb : binVal c = none := by
      unfold binVal
      simp [hc48, hc49]
    simp [hb]

theorem loop : ∀ (L : List Nat) (com : Bool) (n a : Nat) (acc : List Nat) (s : St),
    (∀ c ∈ L, c < 256) → validC com n a → StLe L acc (absC com n (some a)) s →
    match decC L com n a acc with
    | some bs => ∃ k, run imC k s = .exit 0 bs
    | none => ∃ k x, run imC k s = .exit 1 x
  | [], com, n, a, acc, s, _, hv, h => by
    rcases hv with ⟨hn, ha⟩ | ⟨hc, hn1, hn7, ha⟩
    · subst hn ha
      have he : eofCheck (runU false imC head 80 (absC com 0 (some 0))) 0 = true := eof_zero_all com
      have hm := runU_mono false imC head (.inl rfl) 80 h
      generalize runU false imC head 80 (absC com 0 (some 0)) = R at he hm
      cases R with
      | inl _ => cases he
      | inr oc =>
        cases oc with
        | exit st x =>
          simp only [eofCheck, Bool.and_eq_true, beq_iff_eq] at he
          obtain ⟨hx, hst⟩ := he
          obtain ⟨k, hk⟩ := runU_exit 80 (hm.2 st x rfl)
          subst hx; simp at hst; subst hst
          simp only [decC, if_true]; rw [List.append_nil] at hk; exact ⟨k, hk⟩
        | stuck _ _ => cases he
        | outOfFuel => cases he
    · subst hc
      have he : eofCheck (runU false imC head 80 (absCMid (some n) none)) n = true := eof_mid_all n hn1 hn7
      have hm := runU_mono false imC head (.inl rfl) 80 (absC_weaken_acc h)
      generalize runU false imC head 80 (absCMid (some n) none) = R at he hm
      cases R with
      | inl _ => cases he
      | inr oc =>
        cases oc with
        | exit st x =>
          simp only [eofCheck, Bool.and_eq_true, beq_iff_eq] at he
          obtain ⟨hx, hst⟩ := he
          obtain ⟨k, hk⟩ := runU_exit 80 (hm.2 st x rfl)
          subst hx
          have hn0 : n ≠ 0 := by omega
          simp [hn0] at hst; subst hst
          simp only [decC, hn0, if_false]; exact ⟨k, _, hk⟩
        | stuck _ _ => cases he
        | outOfFuel => cases he
  | c :: l, com, n, a, acc, s, hL, hv, h => by
    have hc := hL c (by simp)
    rw [decC_cons]
    by_cases hbit : com = false ∧ 1 ≤ n ∧ n ≤ 7 ∧ c ≠ 48 ∧ c ≠ 49
    · obtain ⟨hcom, hn1, hn7, hc48, hc49⟩ := hbit
      subst hcom
      by_cases hsep : c = 58 ∨ isSep c = true
      · have hr := iter_sep_all n c hn1 hn7 hsep
        unfold iterSepOK at hr
        have h' : StLe l acc { absCMid (some n) none with inp := [c] } s :=
          StLe.shift_inp (x := [c]) (absC_weaken_acc h)
        have hm := runU_mono true imC head (.inr rfl) 80 h'
        generalize runU true imC head 80 { absCMid (some n) none with inp := [c] } = R at hr hm
        cases R with
        | inl _ => cases hr
        | inr oc =>
          cases oc with
          | exit st x =>
            simp only [exit1Check, Bool.and_eq_true, beq_iff_eq] at hr
            obtain ⟨rfl, rfl⟩ := hr
            obtain ⟨k, hk⟩ := runU_exit 80 (hm.2 _ _ rfl)
            rw [stepC_nonbit_none hn1 hc48 hc49]
            exact ⟨k, _, hk⟩
          | stuck _ _ => cases hr
          | outOfFuel => cases hr
      · have hnotsep : c ≠ 58 ∧ isSep c = false := by
          constructor
          · intro e; exact hsep (.inl e)
          · cases hs : isSep c with
            | false => rfl
            | true => exact absurd (.inr hs) hsep
        have hr := iter_nonsep_all c hc
        unfold iterNonSepOK at hr
        have hcnb : (c == 48 || c == 49 || c == 58 || isSep c) = false := by
          simp [hc48, hc49, hnotsep.1, hnotsep.2]
        rw [hcnb] at hr; rw [if_neg (by decide)] at hr
        have h' : StLe l acc { absCMid none none with inp := [c] } s :=
          StLe.shift_inp (x := [c]) (absC_weaken_both h)
        have hm := runU_mono true imC head (.inr rfl) 80 h'
        generalize runU true imC head 80 { absCMid none none with inp := [c] } = R at hr hm
        cases R with
        | inl _ => cases hr
        | inr oc =>
          cases oc with
          | exit st x =>
            simp only [exit1Check, Bool.and_eq_true, beq_iff_eq] at hr
            obtain ⟨rfl, rfl⟩ := hr
            obtain ⟨k, hk⟩ := runU_exit 80 (hm.2 _ _ rfl)
            rw [stepC_nonbit_none hn1 hc48 hc49]
            exact ⟨k, _, hk⟩
          | stuck _ _ => cases hr
          | outOfFuel => cases hr
    · have hi : iterOK com n a c = true := by
        rcases hv with ⟨hn, ha⟩ | ⟨hcom, hn1, hn7, ha⟩
        · subst hn ha; exact iter_zero_all com c hc
        · subst hcom
          have hcbit : c = 48 ∨ c = 49 := by by_cases h48 : c = 48 <;> by_cases h49 : c = 49 <;> simp_all
          exact iter_bit_all n a hn1 hn7 ha hcbit
      have h' : StLe l acc { absC com n (some a) with inp := [c] } s := StLe.shift_inp (x := [c]) h
      have hm := runU_mono true imC head (.inr rfl) 80 h'
      unfold iterOK at hi
      generalize runU true imC head 80 { absC com n (some a) with inp := [c] } = R at hi hm
      cases R with
      | inl t =>
        simp only [iterCheck] at hi
        cases hs : stepC com n a c with
        | none => rw [hs] at hi; cases hi
        | some q =>
          obtain ⟨com', n', a', e⟩ := q
          rw [hs] at hi; dsimp only at hi ⊢
          obtain ⟨t', ht', htt⟩ := hm.1 t rfl
          have h2 := StLe.trans (leB_sound hi) htt
          obtain ⟨k, hk⟩ := runU_run 80 ht'
          have ih := loop l com' n' a' (e ++ acc) t' (fun c hc => hL c (by simp [hc]))
            (stepC_valid hs hv) (by rw [List.nil_append] at h2; exact h2)
          cases hd : decC l com' n' a' (e ++ acc) with
          | some bs => rw [hd] at ih; obtain ⟨k2, hk2⟩ := ih; exact ⟨k + k2, by rw [hk]; exact hk2⟩
          | none => rw [hd] at ih; obtain ⟨k2, x, hk2⟩ := ih; exact ⟨k + k2, x, by rw [hk]; exact hk2⟩
      | inr oc =>
        cases oc with
        | exit st x =>
          simp only [iterCheck, Bool.and_eq_true, beq_iff_eq] at hi
          obtain ⟨rfl, hnone⟩ := hi
          obtain ⟨k, hk⟩ := runU_exit 80 (hm.2 _ _ rfl)
          cases hs : stepC com n a c with
          | none => exact ⟨k, _, hk⟩
          | some q => rw [hs] at hnone; cases hnone
        | stuck _ _ => cases hi
        | outOfFuel => cases hi

theorem load_C (L : List Nat) : load seedC 0x7ffffffde000 L = some (imC, { s0 with inp := L }) := by
  rw [load_inp, show (0x7ffffffde000 : Nat) = sp from rfl, load_eq]; rfl

/-- What `exec` does on every byte listing `L`: if valid format C, exit 0 with
the decoded bytes; otherwise exit 1. -/
theorem exec_decC (L : List Nat) (hL : ∀ c ∈ L, c < 256) :
    match decC L false 0 0 [] with
    | some bs => ∃ F, exec F seedC L = .exit 0 bs
    | none => ∃ F x, exec F seedC L = .exit 1 x := by
  have he : entryCheck (runU true imC head 10 s0) = true := entry_ok
  have h0 : StLe L [] s0 { s0 with inp := L } := StLe.app_self (r := L) (o := []) s0
  have hm := runU_mono true imC head (.inr rfl) 10 h0
  generalize runU true imC head 10 s0 = R at he hm
  cases R with
  | inr _ => cases he
  | inl t =>
    simp only [entryCheck] at he
    obtain ⟨t', ht', htt⟩ := hm.1 t rfl
    obtain ⟨k, hk⟩ := runU_run 10 ht'
    have hloop := loop L false 0 0 [] t' hL (.inl ⟨rfl, rfl⟩)
      (by have hs := StLe.trans (leB_sound he) htt; rw [List.nil_append, List.nil_append] at hs; exact hs)
    simp only [exec_of_load (load_C L)]
    cases hd : decC L false 0 0 [] with
    | some bs => rw [hd] at hloop; obtain ⟨k2, hk2⟩ := hloop; exact ⟨k + k2, by rw [hk]; exact hk2⟩
    | none => rw [hd] at hloop; obtain ⟨k2, x, hk2⟩ := hloop; exact ⟨k + k2, x, by rw [hk]; exact hk2⟩

/-- `seed-c` exits 0 with output `bs` iff `bs` is what listing `L` describes in format C. -/
theorem seedC_correct (L bs : List Nat) (hL : ∀ c ∈ L, c < 256) :
    (∃ F, exec F seedC L = .exit 0 bs) ↔ BinC L bs := by
  rw [← decC_iff]
  have h := exec_decC L hL
  simp only [exec_of_load (load_C L)] at h ⊢
  cases hd : decC L false 0 0 [] with
  | some bs' =>
    rw [hd] at h
    obtain ⟨F, hF⟩ := h
    constructor
    · rintro ⟨G, hG⟩; rw [(run_det hF hG).2]
    · intro e; cases e; exact ⟨F, hF⟩
  | none =>
    rw [hd] at h
    obtain ⟨F, x, hF⟩ := h
    constructor
    · rintro ⟨G, hG⟩; cases (run_det hF hG).1
    · intro e; cases e

/-- On any listing not valid in format C, `seed-c` exits 1. -/
theorem seedC_refuses (L : List Nat) (hL : ∀ c ∈ L, c < 256) (h : ∀ bs, ¬ BinC L bs) :
    ∃ F x, exec F seedC L = .exit 1 x := by
  have he := exec_decC L hL
  cases hd : decC L false 0 0 [] with
  | some bs => exact absurd ((decC_iff L bs).mp hd) (h bs)
  | none => rw [hd] at he; exact he

/-! ### Fast concrete ground witnesses -/

theorem c_empty : exec 200 seedC [] = .exit 0 [] := by rw [exec_of_load (load_C _)]; decide +kernel

theorem c_bytes :
    exec 1000 seedC
      [48, 49, 48, 48, 48, 48, 48, 49, 32, 58, 32, 50, 10, 49, 49, 49, 49, 49, 49, 49, 49]
      = .exit 0 [65, 255] := by rw [exec_of_load (load_C _)]; decide +kernel

theorem c_refuses_digit2 :
    exec 1000 seedC [48, 48, 48, 48, 48, 48, 48, 50] = .exit 1 [] := by rw [exec_of_load (load_C _)]; decide +kernel

theorem c_refuses_split :
    exec 1000 seedC [48, 32, 48, 48, 48, 48, 48, 48, 48] = .exit 1 [] := by rw [exec_of_load (load_C _)]; decide +kernel

theorem c_refuses_colon_in_group :
    exec 1000 seedC [48, 58, 10] = .exit 1 [] := by rw [exec_of_load (load_C _)]; decide +kernel

theorem c_refuses_short :
    exec 1000 seedC [48, 49, 48, 48, 48, 48, 48, 49, 48, 49] = .exit 1 [65] := by rw [exec_of_load (load_C _)]; decide +kernel

theorem c_refuses_cr :
    exec 1000 seedC [13] = .exit 1 [] := by rw [exec_of_load (load_C _)]; decide +kernel

end AArch64Min.SeedC
