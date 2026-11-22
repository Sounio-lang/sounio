/-
Core verification layer for `sporos_vm` (`bootstrap/model/sporos_vm`).

Root A framing:
- Root A is MUSE's hand-written checker (`bootstrap/roots/sporos/sporos-stage1.hex`).
- `spr1_check` is an independent second check (not Root A or Root A's trust base).
- `sporos_vm` is the post-admission native x86-64 bytecode VM paired with
  `Sporos.decode` + `Sporos.checkProg`, running `SporosProvLen.lstep` /
  `Sporos.step _ noPhys` with TMR protection over 3 read-only copies (`code1`,
  `code2`, `code3`) of the flat word/data-meta/instruction table and the
  canonical 38-byte TMR majority voter at `SporosVmBin.L_vm_vote`.

This module establishes:
1. Binary layout constants (`baseAddr`, `ctxBase`, `ctxSize`, `hdrLen`, `vmImage`, `vmAlts`).
2. Kernel-checked decode soundness `table_ok` over `SporosVmBin.vmTable` using
   linear slice walking (`walkSlice`) + `decide +kernel`.
3. Kernel-checked rodata tables (`ro_effIn_ok`, `ro_effOut_ok`, `ro_jtab_ok`,
   `ro_primJtab_ok`, `ro_errPtrs_ok`, `ro_errLens_ok`).
4. Disjoint-region memory lemmas (`Mem.wf`, `CodeMem`, `mem_rdN_of_mem`,
   `mem_wrN_region`, `CodeMem.preserve_*`) and symbolic execution combinators
   (`Reaches`, `Ends`, `step_of_table`, `step_jtab_of_table`).
-/
import Bootstrap.SporosVmBinData
import Bootstrap.SporosProvLen

namespace SporosVmCore

open X86Env
open X86Int
open X86Vm

def W : Nat := X86Env.W
def baseAddr : Nat := 0x400000
def ctxBase : Nat := 0x600000
def ctxSize : Nat := 0x1000
def hdrLen : Nat := 0xb0
def vmCodeEnd : Nat := SporosVmBin.L_code_end

def vmImage : Nat → Option (X86Vm.Ins × Nat) :=
  X86Vm.imageOf SporosVmBin.vmBytes baseAddr

def vmAlts : X86Vm.Alts :=
  X86Vm.imageAlts SporosVmBin.vmBytes baseAddr

/-- The read-only ELF PT_LOAD 0 region of `sporos_vm`. -/
def codeRegion : Region :=
  ⟨baseAddr, SporosVmBin.vmBytes, false⟩

/-- `CodeMem m` states that `m` is well-formed and contains the exact read-only ELF image. -/
def CodeMem (m : Mem) : Prop :=
  codeRegion ∈ m ∧ m.wf

/-! ### Decode verification over `vmBytes` -/

theorem hdrBytes_len : SporosVmBin.hdrBytes.length = hdrLen := by
  decide +kernel

theorem vmBytes_len_pos : 0 < SporosVmBin.vmBytes.length := by
  decide +kernel

theorem vmBytes_len_eq : SporosVmBin.vmBytes.length = 12945 := by
  decide +kernel

theorem code_end_sub_base : vmCodeEnd - baseAddr = 0x2c79 := by
  decide +kernel

theorem code_start_sub_base : SporosVmBin.L_code_start - baseAddr = hdrLen := by
  decide +kernel

theorem codeBytes_len : SporosVmBin.codeBytes.length = vmCodeEnd - (baseAddr + hdrLen) := by
  decide +kernel

theorem vmBytes_drop_hdr :
    SporosVmBin.vmBytes.drop hdrLen = SporosVmBin.codeBytes ++ SporosVmBin.rodataBytes := by
  unfold SporosVmBin.vmBytes
  rw [List.append_assoc, List.drop_left' hdrBytes_len]

theorem vmBytes_drop_ro :
    SporosVmBin.vmBytes.drop (vmCodeEnd - baseAddr) = SporosVmBin.rodataBytes := by
  unfold SporosVmBin.vmBytes
  have hlen : (SporosVmBin.hdrBytes ++ SporosVmBin.codeBytes).length = vmCodeEnd - baseAddr := by
    rw [List.length_append, hdrBytes_len, codeBytes_len]
    decide +kernel
  rw [List.drop_left' hlen]

/-- Linear cursor check matching `(vmBytes.drop (a - baseAddr))` against `vmTable`. -/
def walkSlice (bs : List Nat) (a : Nat) : List (Nat × X86Vm.Ins × Nat) → Bool
  | [] => true
  | (addr, ins, len) :: rest =>
    a == addr && 0 < len &&
    (X86Vm.decode (fun i => bs[i]?) a == some (ins, len)) &&
    walkSlice (bs.drop len) (a + len) rest

theorem walkSlice_sound :
    ∀ (tbl : List (Nat × X86Vm.Ins × Nat)) (k a : Nat),
      walkSlice (SporosVmBin.vmBytes.drop k) a tbl = true →
      a = baseAddr + k →
      ∀ e ∈ tbl, vmImage e.1 = some e.2
  | [], _, _, _, _, e, he => nomatch he
  | (addr, ins, len) :: rest, k, a, hwalk, ha, e, he => by
    simp only [walkSlice, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hwalk
    obtain ⟨⟨⟨had, hlen⟩, hdec⟩, hrest⟩ := hwalk
    subst had
    rw [List.mem_cons] at he
    rcases he with rfl | he
    · unfold vmImage X86Vm.imageOf
      have hge : ¬ (a < baseAddr) := by omega
      rw [if_neg hge]
      have hsub : a - baseAddr = k := by omega
      rw [hsub]
      simpa only [List.getElem?_drop] using hdec
    · have hdrop : (SporosVmBin.vmBytes.drop k).drop len = SporosVmBin.vmBytes.drop (k + len) := by
        rw [List.drop_drop, Nat.add_comm]
      rw [hdrop] at hrest
      exact walkSlice_sound rest (k + len) (a + len) hrest (by omega) e he

theorem walk_vmTable_ok :
    walkSlice (SporosVmBin.codeBytes ++ SporosVmBin.rodataBytes) SporosVmBin.L_code_start SporosVmBin.vmTable = true := by
  decide +kernel

theorem table_ok : ∀ e ∈ SporosVmBin.vmTable, vmImage e.1 = some e.2 := by
  have hwalk : walkSlice (SporosVmBin.vmBytes.drop hdrLen) SporosVmBin.L_code_start SporosVmBin.vmTable = true := by
    rw [vmBytes_drop_hdr]
    exact walk_vmTable_ok
  exact walkSlice_sound SporosVmBin.vmTable hdrLen SporosVmBin.L_code_start hwalk
    (by decide +kernel)

/-! ### Rodata verification -/

def sliceAt (addr n : Nat) : List Nat :=
  (SporosVmBin.vmBytes.drop (addr - baseAddr)).take n

def takeQwords : Nat → List Nat → List Nat
  | 0, _ => []
  | n + 1, b0 :: b1 :: b2 :: b3 :: b4 :: b5 :: b6 :: b7 :: rest =>
    X86Int.leVal [b0, b1, b2, b3, b4, b5, b6, b7] :: takeQwords n rest
  | _, _ => []

def roSlice (addr n : Nat) : List Nat :=
  (SporosVmBin.rodataBytes.drop (addr - vmCodeEnd)).take n

def roQwords (addr count : Nat) : List Nat :=
  takeQwords count (SporosVmBin.rodataBytes.drop (addr - vmCodeEnd))

theorem takeQwords_getElem? (n i : Nat) (bs : List Nat) (hi : i < n) (hlen : 8 * n ≤ bs.length) :
    (takeQwords n bs)[i]? = some (X86Int.leVal ((bs.drop (8 * i)).take 8)) := by
  induction n generalizing i bs with
  | zero => exact absurd hi (Nat.not_lt_zero _)
  | succ n ih =>
    match bs with
    | b0 :: b1 :: b2 :: b3 :: b4 :: b5 :: b6 :: b7 :: rest =>
      cases i with
      | zero => simp [takeQwords]
      | succ i =>
        have hlen' : 8 * n ≤ rest.length := by simp at hlen; omega
        have hih := ih i rest (by omega) hlen'
        have hdrop : (b0 :: b1 :: b2 :: b3 :: b4 :: b5 :: b6 :: b7 :: rest).drop (8 * (i + 1)) =
            rest.drop (8 * i) := by
          have : 8 * (i + 1) = 8 + 8 * i := by omega
          rw [this, ← List.drop_drop]
          rfl
        simpa [takeQwords, hdrop] using hih
    | [] | [_] | [_, _] | [_, _, _] | [_, _, _, _] | [_, _, _, _, _]
    | [_, _, _, _, _, _] | [_, _, _, _, _, _, _] =>
      simp only [List.length_nil, List.length_cons] at hlen
      omega

theorem roSlice_eq_sliceAt {addr n : Nat} (hlo : vmCodeEnd ≤ addr) :
    roSlice addr n = sliceAt addr n := by
  unfold roSlice sliceAt
  rw [← vmBytes_drop_ro, List.drop_drop]
  have hadd : (vmCodeEnd - baseAddr) + (addr - vmCodeEnd) = addr - baseAddr := by
    have := code_end_sub_base; omega
  rw [hadd]

theorem ro_effIn_ok :
    roSlice SporosVmBin.L_prim_eff_in 57 = Sporos.Prim.all.map (fun p => p.eff.1) := by
  decide +kernel

theorem ro_effOut_ok :
    roSlice SporosVmBin.L_prim_eff_out 57 = Sporos.Prim.all.map (fun p => p.eff.2) := by
  decide +kernel

def expectedJtab : List Nat :=
  List.replicate 57 SporosVmBin.L_prim_prologue ++
  [ SporosVmBin.O_lit, SporosVmBin.O_str, SporosVmBin.O_pick, SporosVmBin.O_tor,
    SporosVmBin.O_fromr, SporosVmBin.O_rfetch, SporosVmBin.O_locget, SporosVmBin.O_locset,
    SporosVmBin.O_jmp, SporosVmBin.O_jz, SporosVmBin.O_jnz, SporosVmBin.O_call,
    SporosVmBin.O_ret ]

def expectedPrimJtab : List Nat :=
  [ SporosVmBin.P_dup, SporosVmBin.P_drop, SporosVmBin.P_swap, SporosVmBin.P_over,
    SporosVmBin.P_rot, SporosVmBin.P_nip, SporosVmBin.P_tuck,
    SporosVmBin.P_add, SporosVmBin.P_sub, SporosVmBin.P_mul, SporosVmBin.P_divu,
    SporosVmBin.P_modu, SporosVmBin.P_divs, SporosVmBin.P_mods, SporosVmBin.P_neg,
    SporosVmBin.P_band, SporosVmBin.P_bor, SporosVmBin.P_bxor, SporosVmBin.P_bnot,
    SporosVmBin.P_shl, SporosVmBin.P_shr, SporosVmBin.P_sar,
    SporosVmBin.P_eq, SporosVmBin.P_ne, SporosVmBin.P_ltu, SporosVmBin.P_lts,
    SporosVmBin.P_leu, SporosVmBin.P_les,
    SporosVmBin.P_ld8, SporosVmBin.P_st8, SporosVmBin.P_ld16, SporosVmBin.P_st16,
    SporosVmBin.P_ld32, SporosVmBin.P_st32, SporosVmBin.P_ld64, SporosVmBin.P_st64,
    SporosVmBin.P_copy, SporosVmBin.P_fill, SporosVmBin.P_mcmp,
    SporosVmBin.P_alloc, SporosVmBin.P_mark, SporosVmBin.P_release,
    SporosVmBin.P_streq, SporosVmBin.P_strhash,
    SporosVmBin.P_tnew, SporosVmBin.P_tput, SporosVmBin.P_tget,
    SporosVmBin.P_getb, SporosVmBin.P_putb, SporosVmBin.P_putn, SporosVmBin.P_exit,
    SporosVmBin.P_zero_mk1, SporosVmBin.P_zero_mk1, SporosVmBin.P_zero_mk1,
    SporosVmBin.P_provget, SporosVmBin.P_confget, SporosVmBin.P_demand ]

theorem ro_jtab_ok : roQwords SporosVmBin.L_jtab 70 = expectedJtab := by
  decide +kernel

theorem ro_primJtab_ok : roQwords SporosVmBin.L_prim_jtab 57 = expectedPrimJtab := by
  decide +kernel

theorem ro_errPtrs_ok :
    roQwords SporosVmBin.L_err_ptrs 10 =
      [ SporosVmBin.L_err_s_0, SporosVmBin.L_err_s_1, SporosVmBin.L_err_s_2,
        SporosVmBin.L_err_s_3, SporosVmBin.L_err_s_4, SporosVmBin.L_err_s_5,
        SporosVmBin.L_err_s_6, SporosVmBin.L_err_s_7, SporosVmBin.L_err_s_8,
        SporosVmBin.L_err_s_9 ] := by
  decide +kernel

theorem ro_errLens_ok :
    roQwords SporosVmBin.L_err_lens 10 = [9, 13, 9, 8, 8, 8, 6, 8, 9, 6] := by
  decide +kernel

theorem ro_bounds_ok :
    SporosVmBin.L_rodata = vmCodeEnd ∧
    SporosVmBin.L_err_lens + 80 ≤ baseAddr + SporosVmBin.vmBytes.length ∧
    baseAddr ≤ SporosVmBin.L_prim_eff_in ∧
    SporosVmBin.L_prim_eff_in + 57 ≤ SporosVmBin.L_prim_eff_out ∧
    SporosVmBin.L_prim_eff_out + 57 ≤ SporosVmBin.L_jtab ∧
    SporosVmBin.L_jtab + 560 ≤ SporosVmBin.L_prim_jtab ∧
    SporosVmBin.L_prim_jtab + 456 ≤ SporosVmBin.L_str_usage := by
  decide +kernel

/-! ### Disjoint-region memory lemmas -/

theorem mem_find_of_mem : ∀ (m : Mem) (r : Region) (a idx : Nat),
    m.Pairwise (fun x y => x.hi ≤ y.base ∨ y.hi ≤ x.base) →
    r ∈ m → r.base ≤ a → a < r.hi →
    ∃ i, Mem.find m a idx = some (idx + i, r)
  | [], _, _, _, _, hmem, _, _ => nomatch hmem
  | x :: xs, r, a, idx, hpw, hmem, hlo, hhi => by
    rw [List.mem_cons] at hmem
    rcases hmem with rfl | hmem
    · refine ⟨0, ?_⟩
      simp [Mem.find, hlo, hhi]
    · have hdisj := (List.pairwise_cons.mp hpw).1 r hmem
      have hnot : ¬ (x.base ≤ a ∧ a < x.hi) := by
        rcases hdisj with h | h <;> omega
      obtain ⟨i, hi⟩ := mem_find_of_mem xs r a (idx + 1) (List.pairwise_cons.mp hpw).2 hmem hlo hhi
      refine ⟨i + 1, ?_⟩
      simp only [Mem.find, if_neg hnot]
      simpa [Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using hi

theorem mem_rd_of_mem {m : Mem} {r : Region} {a : Nat}
    (hw : m.wf) (hm : r ∈ m) (hlo : r.base ≤ a) (hhi : a < r.hi) :
    m.rd a = some (r.data[a - r.base]'(by simp [Region.hi] at hhi; omega)) := by
  obtain ⟨i, hf⟩ := mem_find_of_mem m r a 0 hw.2 hm hlo hhi
  have hlt : a - r.base < r.data.length := by simp [Region.hi] at hhi; omega
  simp [Mem.rd, hf, List.getElem?_eq_getElem hlt]

theorem list_take_drop_succ {α : Type} (l : List α) (k n : Nat) (hk : k < l.length) :
    l[k]'hk :: (l.drop (k + 1)).take n = (l.drop k).take (n + 1) := by
  induction k generalizing l with
  | zero =>
    cases l with
    | nil => simp at hk
    | cons x xs => simp
  | succ k ih =>
    cases l with
    | nil => simp at hk
    | cons x xs =>
      simp only [List.getElem_cons_succ, List.drop_succ_cons]
      exact ih xs (by simp at hk; omega)

theorem mem_rdN_of_mem (m : Mem) (r : Region) (a n : Nat)
    (hw : m.wf) (hm : r ∈ m) (hlo : r.base ≤ a) (hhi : a + n ≤ r.hi) :
    m.rdN a n = some ((r.data.drop (a - r.base)).take n) := by
  induction n generalizing a with
  | zero => simp [Mem.rdN]
  | succ n ih =>
    have ha : a < r.hi := by omega
    have hrd := mem_rd_of_mem hw hm hlo ha
    have hih := ih (a + 1) (by omega) (by omega)
    have hsub : a + 1 - r.base = (a - r.base) + 1 := by omega
    have hlt : a - r.base < r.data.length := by simp [Region.hi] at ha; omega
    simp only [Mem.rdN, hrd, hih, hsub, bind, Option.bind, pure]
    congr 1
    exact list_take_drop_succ r.data (a - r.base) n hlt

theorem mem_rdN_ext {m : Mem} {r : Region} {a n : Nat} {bs : List Nat}
    (hw : m.wf) (hm : r ∈ m) (hlo : r.base ≤ a) (hhi : a + n ≤ r.hi)
    (hbs : (r.data.drop (a - r.base)).take n = bs) :
    m.rdN a n = some bs := by
  rw [mem_rdN_of_mem m r a n hw hm hlo hhi, hbs]

theorem rd64_of_mem {m : Mem} {r : Region} {a v : Nat}
    (hw : m.wf) (hm : r ∈ m) (hlo : r.base ≤ a) (hhi : a + 8 ≤ r.hi)
    (hv : leVal ((r.data.drop (a - r.base)).take 8) = v) :
    rd64 m a = some v := by
  simp [rd64, mem_rdN_of_mem m r a 8 hw hm hlo hhi, hv]

theorem rd32_of_mem {m : Mem} {r : Region} {a v : Nat}
    (hw : m.wf) (hm : r ∈ m) (hlo : r.base ≤ a) (hhi : a + 4 ≤ r.hi)
    (hv : leVal ((r.data.drop (a - r.base)).take 4) = v) :
    rd32 m a = some v := by
  simp [rd32, mem_rdN_of_mem m r a 4 hw hm hlo hhi, hv]

theorem rd16_of_mem {m : Mem} {r : Region} {a v : Nat}
    (hw : m.wf) (hm : r ∈ m) (hlo : r.base ≤ a) (hhi : a + 2 ≤ r.hi)
    (hv : leVal ((r.data.drop (a - r.base)).take 2) = v) :
    rd16 m a = some v := by
  simp [rd16, mem_rdN_of_mem m r a 2 hw hm hlo hhi, hv]

theorem rdMem_w8_of_mem {m : Mem} {r : Region} {a : Nat}
    (hw : m.wf) (hm : r ∈ m) (hlo : r.base ≤ a) (hhi : a < r.hi) :
    rdMem .w8 m a = some (r.data[a - r.base]'(by simp [Region.hi] at hhi; omega)) := by
  simp [rdMem, mem_rd_of_mem hw hm hlo hhi]

theorem rdMem_w64_of_mem {m : Mem} {r : Region} {a v : Nat}
    (hw : m.wf) (hm : r ∈ m) (hlo : r.base ≤ a) (hhi : a + 8 ≤ r.hi)
    (hv : leVal ((r.data.drop (a - r.base)).take 8) = v) :
    rdMem .w64 m a = some v :=
  rd64_of_mem hw hm hlo hhi hv

theorem CodeMem.rd64_eq {m : Mem} (hc : CodeMem m) {a v : Nat}
    (hlo : baseAddr ≤ a) (hhi : a + 8 ≤ baseAddr + SporosVmBin.vmBytes.length)
    (hv : leVal (sliceAt a 8) = v) :
    rd64 m a = some v := by
  have hlo' : codeRegion.base ≤ a := hlo
  have hhi' : a + 8 ≤ codeRegion.hi := by dsimp [codeRegion, Region.hi]; omega
  have hv' : leVal ((codeRegion.data.drop (a - codeRegion.base)).take 8) = v := hv
  exact rd64_of_mem hc.2 hc.1 hlo' hhi' hv'

theorem CodeMem.rd8_eq {m : Mem} (hc : CodeMem m) {a b : Nat}
    (hlo : baseAddr ≤ a) (hhi : a < baseAddr + SporosVmBin.vmBytes.length)
    (hb : SporosVmBin.vmBytes[a - baseAddr]'(by omega) = b) :
    rdMem .w8 m a = some b := by
  have hlo' : codeRegion.base ≤ a := hlo
  have hhi' : a < codeRegion.hi := by dsimp [codeRegion, Region.hi]; omega
  have h := rdMem_w8_of_mem hc.2 hc.1 hlo' hhi'
  have hget : codeRegion.data[a - codeRegion.base]'(by dsimp [codeRegion, Region.hi] at hhi' ⊢; omega) = b := hb
  rw [hget] at h
  exact h

/-- Writing a single byte inside a writable region updates only that region and preserves `Mem.wf`. -/
theorem mem_find_idx_lt : ∀ (m : Mem) (a idx i : Nat) (r : Region),
    Mem.find m a idx = some (i, r) → idx ≤ i ∧ i - idx < m.length ∧ m[i - idx]? = some r
  | [], _, _, _, _, hf => by simp [Mem.find] at hf
  | x :: xs, a, idx, i, r, hf => by
    simp only [Mem.find] at hf
    split at hf
    · cases hf
      refine ⟨Nat.le_refl _, ?_, ?_⟩ <;> simp [Nat.sub_self]
    · obtain ⟨h1, h2, h3⟩ := mem_find_idx_lt xs a (idx + 1) i r hf
      refine ⟨by omega, by simp; omega, ?_⟩
      have hne : i - idx ≠ 0 := by omega
      obtain ⟨j, hj⟩ := Nat.exists_eq_succ_of_ne_zero hne
      have hj' : i - (idx + 1) = j := by omega
      simpa [hj, hj'] using h3

theorem mem_pairwise_disj {m : Mem}
    (hpw : m.Pairwise (fun x y => x.hi ≤ y.base ∨ y.hi ≤ x.base))
    {x y : Region} (hx : x ∈ m) (hy : y ∈ m) (hne : x ≠ y) :
    x.hi ≤ y.base ∨ y.hi ≤ x.base := by
  induction m with
  | nil => nomatch hx
  | cons z zs ih =>
    obtain ⟨hz, hzs⟩ := List.pairwise_cons.mp hpw
    rw [List.mem_cons] at hx hy
    rcases hx with rfl | hx <;> rcases hy with rfl | hy
    · exact absurd rfl hne
    · exact hz y hy
    · cases hz x hx with
      | inl h => exact Or.inr h
      | inr h => exact Or.inl h
    · exact ih hzs hx hy

theorem mem_find_bounds : ∀ (m : Mem) (a idx : Nat) {j : Nat} {r : Region},
    Mem.find m a idx = some (j, r) → r.base ≤ a ∧ a < r.hi
  | [], _, _, _, _, hf => by simp [Mem.find] at hf
  | x :: xs, a, idx, j, r, hf => by
    simp only [Mem.find] at hf
    split at hf
    · cases hf; assumption
    · exact mem_find_bounds xs a (idx + 1) hf

theorem mem_find_unique_base (m : Mem) (r : Region) (a idx : Nat)
    (hlo : r.base ≤ a) (hhi : a < r.hi)
    {j : Nat} {s : Region} (hf : Mem.find m a idx = some (j, s))
    (hs : s ∈ m) (hr : r ∈ m) (hpw : m.Pairwise (fun x y => x.hi ≤ y.base ∨ y.hi ≤ x.base)) :
    s = r := by
  have hb := mem_find_bounds m a idx hf
  by_cases hEq : s = r
  · exact hEq
  · have hdisj := mem_pairwise_disj hpw hs hr hEq
    rcases hdisj with h | h <;> omega

theorem wf_mem_same_base_unique {m : Mem} (hw : m.wf) {x y : Region}
    (hx : x ∈ m) (hy : y ∈ m) (hbase : x.base = y.base)
    (hnx : 0 < x.data.length) (hny : 0 < y.data.length) :
    x = y := by
  by_cases hEq : x = y
  · exact hEq
  · have hdisj := mem_pairwise_disj hw.2 hx hy hEq
    simp only [Region.hi] at hdisj
    rcases hdisj with h | h <;> omega

/-- Updating region `r`'s data at offset `k` keeps the exact shape and `Mem.wf`. -/
theorem mem_wr_region {m : Mem} {r : Region} {a v : Nat}
    (hw : m.wf) (hm : r ∈ m) (hrw : r.rw = true) (hlo : r.base ≤ a) (hhi : a < r.hi) :
    ∃ (i : Nat) (m' : Mem),
      m.wr a v = some m' ∧
      m[i]? = some r ∧
      m' = m.set i { r with data := r.data.set (a - r.base) (v % 256) } ∧
      shape m' = shape m ∧
      m'.wf := by
  obtain ⟨i, hf⟩ := mem_find_of_mem m r a 0 hw.2 hm hlo hhi
  simp only [Nat.zero_add] at hf
  have ⟨_, hilt, hget⟩ := mem_find_idx_lt m a 0 i r hf
  let r' : Region := { r with data := r.data.set (a - r.base) (v % 256) }
  let m' : Mem := m.set i r'
  have hwr : m.wr a v = some m' := by simp [Mem.wr, hf, hrw, r', m']
  have hsh : shape m' = shape m := wr_shape hwr
  exact ⟨i, m', hwr, hget, rfl, hsh, wf_of_shape hsh hw⟩

/-- Writing inside a region `r` preserves all other regions `s ≠ r` in `m`. -/
theorem mem_set_other_mem {m : Mem} {i : Nat} {r' s : Region}
    (hget : m[i]? = some r') (hs : s ∈ m) (hne : s.base ≠ r'.base) :
    s ∈ m.set i r' := by
  rw [List.mem_iff_getElem] at hs
  obtain ⟨j, hj, rfl⟩ := hs
  rw [List.mem_iff_getElem]
  refine ⟨j, by simpa using hj, ?_⟩
  by_cases hji : j = i
  · subst hji
    have : m[j] = r' := by simpa [List.getElem?_eq_getElem hj] using hget
    exact absurd (congrArg Region.base this) hne
  · simp [List.getElem_set_ne (Ne.symm hji)]

/-- Helper: erasing index `i` from `m` preserves any member whose base differs from `m[i]`. -/
theorem mem_eraseIdx_other {m : Mem} {i : Nat} {r s : Region}
    (hget : m[i]? = some r) (hs : s ∈ m) (hne : s.base ≠ r.base) :
    s ∈ m.eraseIdx i := by
  obtain ⟨cidx, hclt, hceq⟩ := List.mem_iff_getElem.mp hs
  have hci : cidx ≠ i := by
    intro he
    subst he
    have : m[cidx] = r := by simpa [List.getElem?_eq_getElem hclt] using hget
    exact hne (by rw [← hceq, this])
  rw [List.mem_iff_getElem, List.eraseIdx_eq_take_drop_succ]
  rcases Nat.lt_or_gt_of_ne hci with hlt | hgt
  · refine ⟨cidx, by simp [List.length_take, List.length_drop]; omega, ?_⟩
    rw [List.getElem_append_left (by simp [List.length_take]; omega), List.getElem_take, hceq]
  · refine ⟨cidx - 1, by simp [List.length_take, List.length_drop]; omega, ?_⟩
    have htake : (m.take i).length = i := List.length_take_of_le (by omega)
    rw [List.getElem_append_right (by omega), List.getElem_drop]
    have : i + 1 + (cidx - 1 - i) = cidx := by omega
    simpa [htake, this] using hceq

theorem CodeMem.wr_preserve {m m' : Mem} {a v : Nat}
    (hc : CodeMem m) (hwr : m.wr a v = some m') :
    CodeMem m' := by
  simp only [Mem.wr, bind, Option.bind] at hwr
  cases hf : Mem.find m a 0 with
  | none => rw [hf] at hwr; cases hwr
  | some p =>
    rcases p with ⟨i, r⟩
    rw [hf] at hwr
    dsimp only at hwr
    by_cases hrw : r.rw = true
    · rw [if_pos hrw] at hwr
      cases hwr
      have ⟨_, hilt, hget⟩ := mem_find_idx_lt m a 0 i r hf
      have hrm : r ∈ m := List.mem_of_getElem? hget
      have hfb := mem_find_bounds m a 0 hf
      have hrz : 0 < r.data.length := by simp [Region.hi] at hfb; omega
      have hne : codeRegion.base ≠ r.base := by
        intro heq
        have heq_r := wf_mem_same_base_unique hc.2 hc.1 hrm heq vmBytes_len_pos hrz
        rw [← heq_r] at hrw
        cases hrw
      have hcode' : codeRegion ∈ m.set i { r with data := r.data.set (a - r.base) (v % 256) } := by
        obtain ⟨cidx, hclt, hceq⟩ := List.mem_iff_getElem.mp hc.1
        rw [List.mem_iff_getElem]
        refine ⟨cidx, by simpa using hclt, ?_⟩
        by_cases hci : cidx = i
        · subst hci
          have hmr : m[cidx] = r := by simpa [List.getElem?_eq_getElem hclt] using hget
          have hbase : codeRegion.base = r.base := by rw [← hceq, hmr]
          exact absurd hbase hne
        · rw [List.getElem_set_ne (Ne.symm hci), hceq]
      have hwr' : m.wr a v = some (m.set i { r with data := r.data.set (a - r.base) (v % 256) }) := by
        simp [Mem.wr, hf, hrw]
      exact ⟨hcode', wf_of_shape (wr_shape hwr') hc.2⟩
    · rw [if_neg hrw] at hwr; cases hwr

theorem CodeMem.wrN_preserve {m m' : Mem} {a : Nat} {bs : List Nat}
    (hc : CodeMem m) (hwr : m.wrN a bs = some m') :
    CodeMem m' := by
  induction bs generalizing m a with
  | nil => simp [Mem.wrN] at hwr; subst hwr; exact hc
  | cons b bs ih =>
    simp only [Mem.wrN, bind, Option.bind] at hwr
    cases hw1 : m.wr a b with
    | none => rw [hw1] at hwr; cases hwr
    | some m1 =>
      rw [hw1] at hwr
      exact ih (hc.wr_preserve hw1) hwr

theorem CodeMem.wr64_preserve {m m' : Mem} {a v : Nat}
    (hc : CodeMem m) (hwr : wr64 m a v = some m') :
    CodeMem m' :=
  hc.wrN_preserve hwr

theorem CodeMem.wrMem_preserve {m m' : Mem} {mw : MemW} {a v : Nat}
    (hc : CodeMem m) (hwr : wrMem mw m a v = some m') :
    CodeMem m' := by
  cases mw <;> simp [wrMem, wr16, wr32, wr64] at hwr
  · exact hc.wr_preserve hwr
  · exact hc.wrN_preserve hwr
  · exact hc.wrN_preserve hwr
  · exact hc.wrN_preserve hwr

theorem CodeMem.syscall_preserve {k k' : Kern} {m m' : Mem} {nr a1 a2 a3 a4 a5 a6 rax : Nat}
    (hc : CodeMem m) (hne_munmap : nr = 11 → a1 ≠ baseAddr)
    (hs : X86Env.syscall k m nr a1 a2 a3 a4 a5 a6 = some (.ret rax k' m')) :
    CodeMem m' := by
  have hwf' : m'.wf := syscall_wf hc.2 hs
  refine ⟨?_, hwf'⟩
  unfold X86Env.syscall at hs
  split at hs
  · -- read
    split at hs
    · simp only [bind, Option.bind, pure] at hs
      cases hwr : m.wrN a2 _ with
      | none => rw [hwr] at hs; simp at hs
      | some m1 => rw [hwr] at hs; simp only [Option.some.injEq, SysRes.ret.injEq] at hs; obtain ⟨_, _, rfl⟩ := hs; exact (hc.wrN_preserve hwr).1
    · split at hs
      · cases hm : k.fds[a1 - 3]? with
        | none => rw [hm] at hs; simp at hs; obtain ⟨_, _, rfl⟩ := hs; exact hc.1
        | some v =>
          rw [hm] at hs
          cases v with
          | none => simp at hs; obtain ⟨_, _, rfl⟩ := hs; exact hc.1
          | some fd =>
            simp only [bind, Option.bind, pure] at hs
            cases hwr : m.wrN a2 _ with
            | none => rw [hwr] at hs; simp at hs
            | some m1 => rw [hwr] at hs; simp only [Option.some.injEq, SysRes.ret.injEq] at hs; obtain ⟨_, _, rfl⟩ := hs; exact (hc.wrN_preserve hwr).1
      · simp at hs; obtain ⟨_, _, rfl⟩ := hs; exact hc.1
  · -- write
    split at hs
    · simp only [bind, Option.bind, pure] at hs
      cases hrd : m.rdN a2 _ with
      | none => rw [hrd] at hs; simp at hs
      | some bs => rw [hrd] at hs; simp only [Option.some.injEq, SysRes.ret.injEq] at hs; obtain ⟨_, _, rfl⟩ := hs; exact hc.1
    · simp at hs; obtain ⟨_, _, rfl⟩ := hs; exact hc.1
  · -- open
    split at hs
    · cases hs
    · simp only [bind, Option.bind, pure] at hs
      cases hcstr : m.cstr a1 4096 with
      | none => rw [hcstr] at hs; simp at hs
      | some nm =>
        rw [hcstr] at hs; dsimp only at hs
        split at hs <;> (simp only [Option.some.injEq, SysRes.ret.injEq] at hs; obtain ⟨_, _, rfl⟩ := hs; exact hc.1)
  · -- close
    split at hs <;> (cases hs; exact hc.1)
  · -- mmap
    split at hs
    · cases hs
    · split at hs
      · cases hs; exact hc.1
      · simp only [bind, Option.bind, pure] at hs
        cases hpl : place m (k.next).1 (pages a2) with
        | none => rw [hpl] at hs; simp at hs
        | some o =>
          rw [hpl] at hs
          cases o with
          | none => simp at hs; obtain ⟨_, _, rfl⟩ := hs; exact hc.1
          | some a => simp at hs; obtain ⟨_, _, rfl⟩ := hs; exact List.mem_append_left _ hc.1
  · -- munmap
    cases hf : m.find a1 0 with
    | none => rw [hf] at hs; simp at hs
    | some p =>
      rcases p with ⟨i, r⟩
      rw [hf] at hs; dsimp only at hs
      split at hs
      · rename_i hcond
        cases hs
        have hne : a1 ≠ baseAddr := hne_munmap rfl
        have hne_r : codeRegion.base ≠ r.base := by
          have : r.base = a1 := hcond.1
          rw [this]; exact Ne.symm hne
        have ⟨_, _, hget⟩ := mem_find_idx_lt m a1 0 i r hf
        exact mem_eraseIdx_other hget hc.1 hne_r
      · cases hs
  · -- mremap
    split at hs
    · cases hs
    · cases hf : m.find a1 0 with
      | none => rw [hf] at hs; simp at hs
      | some p =>
        rcases p with ⟨i, r⟩
        rw [hf] at hs; dsimp only at hs
        split at hs
        · cases hs
        · rename_i hcond
          split at hs
          · cases hs
          · simp only [bind, Option.bind, pure] at hs
            cases hpl : place (m.eraseIdx i) (k.next).1 (pages a3) with
            | none => rw [hpl] at hs; simp at hs
            | some o =>
              rw [hpl] at hs
              cases o with
              | none => simp at hs; obtain ⟨_, _, rfl⟩ := hs; exact hc.1
              | some a =>
                simp only [Option.some.injEq, SysRes.ret.injEq] at hs
                obtain ⟨_, _, rfl⟩ := hs
                have hrw : r.rw = true := by
                  by_cases hrwt : r.rw = true
                  · exact hrwt
                  · exact False.elim (hcond (Or.inr (Or.inr hrwt)))
                have hne_r : codeRegion.base ≠ r.base := by
                  intro heq
                  have ⟨_, _, hget⟩ := mem_find_idx_lt m a1 0 i r hf
                  have hrm : r ∈ m := List.mem_of_getElem? hget
                  have hfb := mem_find_bounds m a1 0 hf
                  have hrz : 0 < r.data.length := by simp [Region.hi] at hfb; omega
                  have heq_r := wf_mem_same_base_unique hc.2 hc.1 hrm heq vmBytes_len_pos hrz
                  rw [← heq_r] at hrw
                  cases hrw
                have ⟨_, _, hget⟩ := mem_find_idx_lt m a1 0 i r hf
                exact List.mem_append_left _ (mem_eraseIdx_other hget hc.1 hne_r)
  · -- exit
    cases hs
  · -- unknown
    cases hs

/-! ### Symbolic execution combinators -/

def Reaches (im : Nat → Option (X86Vm.Ins × Nat)) (s₀ s₁ : X86Vm.St) : Prop :=
  ∃ k, X86Vm.run im k (.ok s₀) = .ok s₁

def Ends (im : Nat → Option (X86Vm.Ins × Nat)) (s₀ : X86Vm.St) (o : X86Vm.Out) : Prop :=
  ∃ k, X86Vm.run im k (.ok s₀) = o

theorem Reaches.refl (im : Nat → Option (X86Vm.Ins × Nat)) (s : X86Vm.St) : Reaches im s s :=
  ⟨0, rfl⟩

theorem Reaches.trans {im : Nat → Option (X86Vm.Ins × Nat)} {s₀ s₁ s₂ : X86Vm.St}
    (h₁ : Reaches im s₀ s₁) (h₂ : Reaches im s₁ s₂) : Reaches im s₀ s₂ := by
  obtain ⟨k₁, hk₁⟩ := h₁
  obtain ⟨k₂, hk₂⟩ := h₂
  refine ⟨k₁ + k₂, ?_⟩
  rw [X86Vm.run_add, hk₁, hk₂]

theorem Reaches.step {im : Nat → Option (X86Vm.Ins × Nat)} {s₀ s₁ : X86Vm.St}
    (h : X86Vm.step im s₀ = .ok s₁) : Reaches im s₀ s₁ :=
  ⟨1, by simpa [X86Vm.run] using h⟩

theorem Ends.step {im : Nat → Option (X86Vm.Ins × Nat)} {s₀ : X86Vm.St} {o : X86Vm.Out}
    (h : X86Vm.step im s₀ = o) : Ends im s₀ o :=
  ⟨1, by simpa [X86Vm.run] using h⟩

theorem Ends.of_reaches {im : Nat → Option (X86Vm.Ins × Nat)} {s₀ s₁ : X86Vm.St} {o : X86Vm.Out}
    (h₁ : Reaches im s₀ s₁) (h₂ : Ends im s₁ o) : Ends im s₀ o := by
  obtain ⟨k₁, hk₁⟩ := h₁
  obtain ⟨k₂, hk₂⟩ := h₂
  refine ⟨k₁ + k₂, ?_⟩
  rw [X86Vm.run_add, hk₁, hk₂]

theorem step_of_table {s : X86Vm.St} {ins : X86Vm.Ins} {len : Nat}
    (hmem : (s.rip, ins, len) ∈ SporosVmBin.vmTable) :
    X86Vm.step vmImage s = X86Vm.exec ins len s := by
  simp [X86Vm.step, table_ok _ hmem]

end SporosVmCore
