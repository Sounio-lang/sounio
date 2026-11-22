/-
Symbolic execution and mutable region memory layer for `sporos_vm`.

Root A framing:
- Root A is MUSE's hand-written checker (`bootstrap/roots/sporos/sporos-stage1.hex`).
- `spr1_check` is an independent second check (not Root A or Root A's trust base).
- `sporos_vm` is the post-admission native x86-64 bytecode VM paired with
  `Sporos.decode` + `Sporos.checkProg`.

This module provides:
1. Chunked `vmTable` lookup (`vmTable_eq_chunks_flatten`, `step_of_chunk`).
2. Clean single-step rewrite lemmas (`exec_*`) for every `X86Vm.Ins` form used in
   `sporos_vm`, plus arithmetic/flag bridges (`aluF_*`, `cond_sub_*`,
   `toSigned64_lt_iff`, `toSigned64_le_iff`, `leVal_bytes*`).
3. Mutable byte-array updates (`writeBytes`, `mem_wrN_at_idx`, `mem_wrN_region`)
   preserving `Mem.wf`, `shape`, `CodeMem`, and disjoint regions.
4. Machine specifications for the VM runtime helpers (`fn_mmap_spec`,
   `fn_mremap_spec`, `fn_write_stderr_spec`, `fn_write_stderr_cstr_spec`).
-/
import Bootstrap.SporosVmCore

namespace SporosVmExec

open X86Env
open X86Int
open X86Vm
open SporosVmCore

/-! ### Chunked instruction table lookup -/

@[simp] theorem st_set_mem (s : X86Int.St) (r : Nat) (v : Option Nat) :
    (s.set r v).mem = s.mem := rfl

@[simp] theorem st_set_k (s : X86Int.St) (r : Nat) (v : Option Nat) :
    (s.set r v).k = s.k := rfl

theorem vmTable_eq_chunks_flatten :
    SporosVmBin.vmTable = (SporosVmBin.vmTableChunks.map Prod.snd).flatten := by
  unfold SporosVmBin.vmTable SporosVmBin.vmTableChunks
  dsimp only [List.map]
  simp only [List.flatten_cons, List.flatten_nil, List.append_nil, List.append_assoc]

theorem mem_vmTable_of_chunk (k : Nat) {a : Nat} {c : List (Nat × X86Vm.Ins × Nat)}
    (hc : SporosVmBin.vmTableChunks[k]? = some (a, c))
    {e : Nat × X86Vm.Ins × Nat} (he : e ∈ c) :
    e ∈ SporosVmBin.vmTable := by
  rw [vmTable_eq_chunks_flatten, List.mem_flatten]
  refine ⟨c, ?_, he⟩
  rw [List.mem_map]
  exact ⟨(a, c), List.mem_of_getElem? hc, rfl⟩

theorem step_of_chunk (k : Nat) {a : Nat} {c : List (Nat × X86Vm.Ins × Nat)}
    (hc : SporosVmBin.vmTableChunks[k]? = some (a, c))
    {s : X86Vm.St} {ins : X86Vm.Ins} {len : Nat}
    (he : (s.rip, ins, len) ∈ c) :
    X86Vm.step vmImage s = X86Vm.exec ins len s :=
  step_of_table (mem_vmTable_of_chunk k hc he)

/-! ### Little-endian byte conversions -/

theorem leVal_bytes2 (v : Nat) : leVal (bytes2 v) = v % 2 ^ 16 := by
  simp [bytes2, leVal]
  omega

theorem leVal_bytes4 (v : Nat) : leVal (bytes4 v) = v % 2 ^ 32 := by
  simp [bytes4, leVal, List.range_succ, List.range_zero]
  omega

theorem leVal_bytes8 (v : Nat) : leVal (bytes8 v) = v % X86Env.W := by
  simp [bytes8, leVal, List.range_succ, List.range_zero, X86Env.W]
  omega

theorem bytes2_length (v : Nat) : (bytes2 v).length = 2 := by simp [bytes2]
theorem bytes4_length (v : Nat) : (bytes4 v).length = 4 := by simp [bytes4]
theorem bytes8_length (v : Nat) : (bytes8 v).length = 8 := by simp [bytes8]

theorem bytes2_map_mod (v : Nat) : (bytes2 v).map (· % 256) = bytes2 v := by
  simp [bytes2]

theorem bytes4_map_mod (v : Nat) : (bytes4 v).map (· % 256) = bytes4 v := by
  simp [bytes4, List.range_succ, List.range_zero]

theorem bytes8_map_mod (v : Nat) : (bytes8 v).map (· % 256) = bytes8 v := by
  simp [bytes8, List.range_succ, List.range_zero]

/-! ### Single-step execution rewrites -/

theorem exec_movI {d imm len : Nat} {s : X86Vm.St} :
    X86Vm.exec (.ofInt (.movI d imm)) len s = .ok { s.set d (some (imm % X86Env.W)) with rip := s.rip + len } :=
  rfl

theorem exec_mov {d r len v : Nat} {s : X86Vm.St} (hr : s.reg r = some v) :
    X86Vm.exec (.ofInt (.mov d r)) len s = .ok { s.set d (some v) with rip := s.rip + len } := by
  simp [X86Vm.exec, X86Int.exec, hr, X86Int.lift, bind, Option.bind, pure]

theorem exec_alu {op : X86Int.Alu} {d r len a b : Nat} {s : X86Vm.St}
    (ha : s.reg d = some a) (hb : s.reg r = some b) :
    X86Vm.exec (.ofInt (.alu op d r)) len s =
      let (v, f) := X86Int.aluF op a b
      .ok { (s.set d (some v)) with fl := f, rip := s.rip + len } := by
  simp [X86Vm.exec, X86Int.exec, ha, hb, X86Int.lift, bind, Option.bind, pure]

theorem exec_aluI {op : X86Int.Alu} {d imm len a : Nat} {s : X86Vm.St} (ha : s.reg d = some a) :
    X86Vm.exec (.ofInt (.aluI op d imm)) len s =
      let (v, f) := X86Int.aluF op a (imm % X86Env.W)
      .ok { (s.set d (some v)) with fl := f, rip := s.rip + len } := by
  simp [X86Vm.exec, X86Int.exec, ha, X86Int.lift, bind, Option.bind, pure]

theorem exec_cmp {a b len x y : Nat} {s : X86Vm.St} (hx : s.reg a = some x) (hy : s.reg b = some y) :
    X86Vm.exec (.ofInt (.cmp a b)) len s =
      .ok { s with fl := (X86Int.aluF .sub x y).2, rip := s.rip + len } := by
  simp [X86Vm.exec, X86Int.exec, hx, hy, X86Int.lift, bind, Option.bind, pure]

theorem exec_cmpI {a imm len x : Nat} {s : X86Vm.St} (hx : s.reg a = some x) :
    X86Vm.exec (.ofInt (.cmpI a imm)) len s =
      .ok { s with fl := (X86Int.aluF .sub x (imm % X86Env.W)).2, rip := s.rip + len } := by
  simp [X86Vm.exec, X86Int.exec, hx, X86Int.lift, bind, Option.bind, pure]

theorem exec_test {a b len x y : Nat} {s : X86Vm.St} (hx : s.reg a = some x) (hy : s.reg b = some y) :
    X86Vm.exec (.ofInt (.test a b)) len s =
      .ok { s with fl := (X86Int.aluF .and x y).2, rip := s.rip + len } := by
  simp [X86Vm.exec, X86Int.exec, hx, hy, X86Int.lift, bind, Option.bind, pure]

theorem exec_testI {a imm len x : Nat} {s : X86Vm.St} (hx : s.reg a = some x) :
    X86Vm.exec (.testI a imm) len s =
      .ok { s with fl := (X86Int.aluF .and (x % X86Env.W) (imm % X86Env.W)).2, rip := s.rip + len } := by
  simp [X86Vm.exec, hx, X86Int.lift, bind, Option.bind, pure]

theorem exec_shlI {d c len a : Nat} {s : X86Vm.St} (ha : s.reg d = some a) :
    X86Vm.exec (.ofInt (.shlI d c)) len s =
      .ok { (s.set d (some (a * 2 ^ (c % 64) % X86Env.W))) with fl := Flags.undef, rip := s.rip + len } := by
  simp [X86Vm.exec, X86Int.exec, ha, X86Int.lift, bind, Option.bind, pure]

theorem exec_shrI {d c len a : Nat} {s : X86Vm.St} (ha : s.reg d = some a) :
    X86Vm.exec (.ofInt (.shrI d c)) len s =
      .ok { (s.set d (some (a / 2 ^ (c % 64)))) with fl := Flags.undef, rip := s.rip + len } := by
  simp [X86Vm.exec, X86Int.exec, ha, X86Int.lift, bind, Option.bind, pure]

theorem exec_shlR {d len a c : Nat} {s : X86Vm.St} (ha : s.reg d = some a) (hc : s.reg 1 = some c) :
    X86Vm.exec (.shlR d) len s =
      .ok { (s.set d (some ((a % X86Env.W) * 2 ^ (c % 64) % X86Env.W))) with fl := Flags.undef, rip := s.rip + len } := by
  simp [X86Vm.exec, ha, hc, X86Int.lift, bind, Option.bind, pure]

theorem exec_shrR {d len a c : Nat} {s : X86Vm.St} (ha : s.reg d = some a) (hc : s.reg 1 = some c) :
    X86Vm.exec (.shrR d) len s =
      .ok { (s.set d (some ((a % X86Env.W) / 2 ^ (c % 64)))) with fl := Flags.undef, rip := s.rip + len } := by
  simp [X86Vm.exec, ha, hc, X86Int.lift, bind, Option.bind, pure]

theorem exec_sarR {d len a c : Nat} {s : X86Vm.St} (ha : s.reg d = some a) (hc : s.reg 1 = some c) :
    X86Vm.exec (.sarR d) len s =
      .ok { (s.set d (some (sar64 a c))) with fl := Flags.undef, rip := s.rip + len } := by
  simp [X86Vm.exec, ha, hc, X86Int.lift, bind, Option.bind, pure]

theorem exec_neg {d len a : Nat} {s : X86Vm.St} (ha : s.reg d = some a) :
    X86Vm.exec (.neg d) len s =
      let (v, f) := X86Int.aluF .sub 0 (a % X86Env.W)
      .ok { (s.set d (some v)) with fl := f, rip := s.rip + len } := by
  simp [X86Vm.exec, ha, X86Int.lift, bind, Option.bind, pure]

theorem exec_not {d len a : Nat} {s : X86Vm.St} (ha : s.reg d = some a) :
    X86Vm.exec (.not d) len s =
      .ok { (s.set d (some (X86Env.W - 1 - a % X86Env.W))) with rip := s.rip + len } := by
  simp [X86Vm.exec, ha, X86Int.lift, bind, Option.bind, pure]

theorem exec_imul {d src len a b : Nat} {s : X86Vm.St} (ha : s.reg d = some a) (hb : s.reg src = some b) :
    X86Vm.exec (.imul d src) len s =
      let res := ((a % X86Env.W) * (b % X86Env.W)) % X86Env.W
      let ov := toSigned64 a * toSigned64 b != toSigned64 res
      .ok { (s.set d (some res)) with fl := ⟨some ov, none, none, some ov⟩, rip := s.rip + len } := by
  simp [X86Vm.exec, ha, hb, X86Int.lift, bind, Option.bind, pure]

theorem exec_ld {d b disp len a v : Nat} {s : X86Vm.St}
    (hb : s.reg b = some a) (hm : rd64 s.mem (ea a disp) = some v) :
    X86Vm.exec (.ofInt (.ld d b disp)) len s = .ok { s.set d (some v) with rip := s.rip + len } := by
  simp [X86Vm.exec, X86Int.exec, hb, hm, X86Int.lift, bind, Option.bind, pure]

theorem exec_ld32 {d b disp len a v : Nat} {s : X86Vm.St}
    (hb : s.reg b = some a) (hm : rd32 s.mem (ea a disp) = some v) :
    X86Vm.exec (.ofInt (.ld32 d b disp)) len s = .ok { s.set d (some v) with rip := s.rip + len } := by
  simp [X86Vm.exec, X86Int.exec, hb, hm, X86Int.lift, bind, Option.bind, pure]

theorem exec_ld16 {d b disp len a v : Nat} {s : X86Vm.St}
    (hb : s.reg b = some a) (hm : rd16 s.mem (ea a disp) = some v) :
    X86Vm.exec (.ld16 d b disp) len s = .ok { s.set d (some v) with rip := s.rip + len } := by
  simp [X86Vm.exec, hb, hm, X86Int.lift, bind, Option.bind, pure]

theorem exec_st {b disp r len a v : Nat} {m' : Mem} {s : X86Vm.St}
    (hb : s.reg b = some a) (hr : s.reg r = some v) (hm : wr64 s.mem (ea a disp) v = some m') :
    X86Vm.exec (.ofInt (.st b disp r)) len s = .ok { s with mem := m', rip := s.rip + len } := by
  simp [X86Vm.exec, X86Int.exec, hb, hr, hm, X86Int.lift, bind, Option.bind, pure]

theorem exec_st32 {b disp r len a v : Nat} {m' : Mem} {s : X86Vm.St}
    (hb : s.reg b = some a) (hr : s.reg r = some v) (hm : wr32 s.mem (ea a disp) v = some m') :
    X86Vm.exec (.st32 b disp r) len s = .ok { s with mem := m', rip := s.rip + len } := by
  simp [X86Vm.exec, hb, hr, hm, X86Int.lift, bind, Option.bind, pure]

theorem exec_st16 {b disp r len a v : Nat} {m' : Mem} {s : X86Vm.St}
    (hb : s.reg b = some a) (hr : s.reg r = some v) (hm : wr16 s.mem (ea a disp) v = some m') :
    X86Vm.exec (.st16 b disp r) len s = .ok { s with mem := m', rip := s.rip + len } := by
  simp [X86Vm.exec, hb, hr, hm, X86Int.lift, bind, Option.bind, pure]

theorem exec_lea {d b disp len a : Nat} {s : X86Vm.St} (hb : s.reg b = some a) :
    X86Vm.exec (.ofInt (.lea d b disp)) len s = .ok { s.set d (some (ea a disp)) with rip := s.rip + len } := by
  simp [X86Vm.exec, X86Int.exec, hb, X86Int.lift, bind, Option.bind, pure]

theorem exec_ldIdx {w : MemW} {d b i scale disp len xb xi v : Nat} {s : X86Vm.St}
    (hb : s.reg b = some xb) (hi : s.reg i = some xi) (hm : rdMem w s.mem (eaIdx xb xi scale disp) = some v) :
    X86Vm.exec (.ldIdx w d b i scale disp) len s = .ok { s.set d (some v) with rip := s.rip + len } := by
  simp [X86Vm.exec, hb, hi, hm, X86Int.lift, bind, Option.bind, pure]

theorem exec_stIdx {w : MemW} {b i scale disp r len xb xi v : Nat} {m' : Mem} {s : X86Vm.St}
    (hb : s.reg b = some xb) (hi : s.reg i = some xi) (hr : s.reg r = some v)
    (hm : wrMem w s.mem (eaIdx xb xi scale disp) v = some m') :
    X86Vm.exec (.stIdx w b i scale disp r) len s = .ok { s with mem := m', rip := s.rip + len } := by
  simp [X86Vm.exec, hb, hi, hr, hm, X86Int.lift, bind, Option.bind, pure]

theorem exec_leaIdx {d b i scale disp len xb xi : Nat} {s : X86Vm.St}
    (hb : s.reg b = some xb) (hi : s.reg i = some xi) :
    X86Vm.exec (.leaIdx d b i scale disp) len s =
      .ok { s.set d (some (eaIdx xb xi scale disp)) with rip := s.rip + len } := by
  simp [X86Vm.exec, hb, hi, X86Int.lift, bind, Option.bind, pure]

theorem exec_setcc {cc d len old : Nat} {c : Bool} {s : X86Vm.St}
    (hc : X86Int.cond s.fl cc = some c) (hd : s.reg d = some old) :
    X86Vm.exec (.setcc cc d) len s =
      let old' := old % X86Env.W
      .ok { s.set d (some (old' - old' % 256 + if c then 1 else 0)) with rip := s.rip + len } := by
  simp [X86Vm.exec, hc, hd, X86Int.lift, bind, Option.bind, pure]

theorem exec_cmovcc {cc d src len v old : Nat} {c : Bool} {s : X86Vm.St}
    (hc : X86Int.cond s.fl cc = some c) (hs : s.reg src = some v) (hd : s.reg d = some old) :
    X86Vm.exec (.cmovcc cc d src) len s =
      .ok { s.set d (some (if c then v else old)) with rip := s.rip + len } := by
  simp [X86Vm.exec, hc, hs, hd, X86Int.lift, bind, Option.bind, pure]

theorem exec_jmp {t len : Nat} {s : X86Vm.St} :
    X86Vm.exec (.ofInt (.jmp t)) len s = .ok { s with rip := t } := rfl

theorem exec_jcc {cc t len : Nat} {c : Bool} {s : X86Vm.St} (hc : X86Int.cond s.fl cc = some c) :
    X86Vm.exec (.ofInt (.jcc cc t)) len s = .ok { s with rip := if c then t else s.rip + len } := by
  simp [X86Vm.exec, X86Int.exec, hc, X86Int.lift, bind, Option.bind, pure]

theorem exec_jtab {b i disp len x y t : Nat} {s : X86Vm.St}
    (hx : s.reg b = some x) (hy : s.reg i = some y)
    (ht : rd64 s.mem ((x + 8 * y + disp) % X86Env.W) = some t) :
    X86Vm.exec (.ofInt (.jtab b i disp)) len s = .ok { s with rip := t } := by
  simp [X86Vm.exec, X86Int.exec, hx, hy, ht, X86Int.lift, bind, Option.bind, pure]

theorem exec_call {t len sp : Nat} {m' : Mem} {s : X86Vm.St}
    (hsp : s.reg 4 = some sp)
    (hm : wr64 s.mem ((sp + X86Env.W - 8) % X86Env.W) (s.rip + len) = some m') :
    X86Vm.exec (.ofInt (.call t)) len s =
      .ok { s.set 4 (some ((sp + X86Env.W - 8) % X86Env.W)) with mem := m', rip := t } := by
  simp [X86Vm.exec, X86Int.exec, hsp, hm, X86Int.lift, bind, Option.bind, pure]

theorem exec_ret {len sp t : Nat} {s : X86Vm.St}
    (hsp : s.reg 4 = some sp) (hm : rd64 s.mem sp = some t) :
    X86Vm.exec (.ofInt .ret) len s =
      .ok { s.set 4 (some ((sp + 8) % X86Env.W)) with rip := t } := by
  simp [X86Vm.exec, X86Int.exec, hsp, hm, X86Int.lift, bind, Option.bind, pure]

theorem exec_syscall_ret {len nr a1 a2 a3 a4 a5 a6 rax : Nat} {k' : Kern} {m' : Mem} {s : X86Vm.St}
    (h0 : s.reg 0 = some nr) (h7 : s.reg 7 = some a1) (h6 : s.reg 6 = some a2)
    (h2 : s.reg 2 = some a3) (h10 : s.reg 10 = some a4) (h8 : s.reg 8 = some a5)
    (h9 : s.reg 9 = some a6)
    (hs : X86Env.syscall s.k s.mem nr a1 a2 a3 a4 a5 a6 = some (.ret rax k' m')) :
    X86Vm.exec (.ofInt .syscall) len s =
      .ok { ((s.set 0 (some rax)).set 1 (some (s.rip + len))).set 11 none with
            k := k', mem := m', rip := s.rip + len } := by
  simp [X86Vm.exec, X86Int.exec, h0, h7, h6, h2, h10, h8, h9, hs]

theorem exec_syscall_exit {len nr a1 a2 a3 a4 a5 a6 c : Nat} {s : X86Vm.St}
    (h0 : s.reg 0 = some nr) (h7 : s.reg 7 = some a1) (h6 : s.reg 6 = some a2)
    (h2 : s.reg 2 = some a3) (h10 : s.reg 10 = some a4) (h8 : s.reg 8 = some a5)
    (h9 : s.reg 9 = some a6)
    (hs : X86Env.syscall s.k s.mem nr a1 a2 a3 a4 a5 a6 = some (.exit c)) :
    X86Vm.exec (.ofInt .syscall) len s = .exit c := by
  simp [X86Vm.exec, X86Int.exec, h0, h7, h6, h2, h10, h8, h9, hs]

/-! ### Arithmetic & condition code bridges -/

theorem cond_aluF_sub_e (x y : Nat) (hx : x < X86Env.W) (hy : y < X86Env.W) :
    X86Int.cond (X86Int.aluF .sub x y).2 4 = some (decide (x = y)) := by
  simp [X86Int.cond, X86Int.aluF]
  by_cases hxy : x = y
  · subst hxy; simp
  · have hnz : (x + X86Env.W - y) % X86Env.W ≠ 0 := by
      intro hz
      rcases Nat.dvd_of_mod_eq_zero hz with ⟨k, hk⟩
      rcases k with _ | _ | k
      · omega
      · omega
      · have h2 : 2 * X86Env.W ≤ X86Env.W * (k + 1 + 1) := by
          rw [Nat.mul_comm 2 X86Env.W]
          exact Nat.mul_le_mul_left X86Env.W (by omega)
        omega
    simp [hnz, hxy]

theorem cond_aluF_sub_ne (x y : Nat) (hx : x < X86Env.W) (hy : y < X86Env.W) :
    X86Int.cond (X86Int.aluF .sub x y).2 5 = some (decide (x ≠ y)) := by
  have he := cond_aluF_sub_e x y hx hy
  simp only [X86Int.cond, X86Int.aluF, bind, Option.bind, pure] at he ⊢
  simp only [Option.some.injEq] at he ⊢
  have h4 : (4 % 2 == 0) = true := rfl
  have h5 : (5 % 2 == 0) = false := rfl
  rw [h4] at he; rw [h5]
  simp only [↓reduceIte, Bool.false_eq_true] at he ⊢
  rw [he]
  by_cases hxy : x = y <;> simp [hxy]

theorem cond_aluF_sub_b (x y : Nat) :
    X86Int.cond (X86Int.aluF .sub x y).2 2 = some (decide (x < y)) := by
  simp [X86Int.cond, X86Int.aluF]

theorem cond_aluF_sub_ae (x y : Nat) :
    X86Int.cond (X86Int.aluF .sub x y).2 3 = some (decide (y ≤ x)) := by
  simp [X86Int.cond, X86Int.aluF]
  by_cases h : x < y <;> simp [h] <;> omega

theorem cond_aluF_sub_be (x y : Nat) (hx : x < X86Env.W) (hy : y < X86Env.W) :
    X86Int.cond (X86Int.aluF .sub x y).2 6 = some (decide (x ≤ y)) := by
  have he := cond_aluF_sub_e x y hx hy
  simp only [X86Int.cond, X86Int.aluF, bind, Option.bind, pure, Option.some.injEq] at he ⊢
  have h4 : (4 % 2 == 0) = true := rfl
  have h6 : (6 % 2 == 0) = true := rfl
  rw [h4] at he; rw [h6]
  simp only [↓reduceIte] at he ⊢
  rw [he]
  by_cases hxy : x ≤ y
  · have : (decide (x < y) || decide (x = y)) = true := by
      rcases Nat.lt_or_eq_of_le hxy with h | h <;> simp [h]
    simp [hxy, this]
  · have hlt : ¬ x < y := by omega
    have hne : x ≠ y := by omega
    simp [hxy, hlt, hne]

theorem cond_aluF_sub_a (x y : Nat) (hx : x < X86Env.W) (hy : y < X86Env.W) :
    X86Int.cond (X86Int.aluF .sub x y).2 7 = some (decide (y < x)) := by
  have hbe := cond_aluF_sub_be x y hx hy
  simp only [X86Int.cond, X86Int.aluF, bind, Option.bind, pure, Option.some.injEq] at hbe ⊢
  have h6 : (6 % 2 == 0) = true := rfl
  have h7 : (7 % 2 == 0) = false := rfl
  rw [h6] at hbe; rw [h7]
  simp only [↓reduceIte, Bool.false_eq_true] at hbe ⊢
  rw [hbe]
  by_cases h : x ≤ y
  · have hnot : ¬ y < x := by omega
    simp [h, hnot]
  · have hlt : y < x := by omega
    simp [h, hlt]

theorem cond_aluF_and_e (x y : Nat) :
    X86Int.cond (X86Int.aluF .and x y).2 4 = some (decide (Nat.land x y = 0)) := by
  have hland : Nat.land x y = x &&& y := rfl
  rw [hland]
  by_cases h : x &&& y = 0 <;> simp [X86Int.cond, X86Int.aluF, h]

theorem cond_aluF_and_ne (x y : Nat) :
    X86Int.cond (X86Int.aluF .and x y).2 5 = some (decide (Nat.land x y ≠ 0)) := by
  have hland : Nat.land x y = x &&& y := rfl
  rw [hland]
  by_cases h : x &&& y = 0 <;> simp [X86Int.cond, X86Int.aluF, h]

theorem cond_aluF_and_self_s (x : Nat) :
    X86Int.cond (X86Int.aluF .and x x).2 8 = some (X86Int.msb x) := by
  simp [X86Int.cond, X86Int.aluF]

theorem cond_aluF_and_self_e (x : Nat) :
    X86Int.cond (X86Int.aluF .and x x).2 4 = some (decide (x = 0)) := by
  by_cases h : x = 0 <;> simp [X86Int.cond, X86Int.aluF, h]

theorem cond_aluF_and_self_ne (x : Nat) :
    X86Int.cond (X86Int.aluF .and x x).2 5 = some (decide (x ≠ 0)) := by
  by_cases h : x = 0 <;> simp [X86Int.cond, X86Int.aluF, h]

theorem toSigned64_lt_iff (a b : Nat) (ha : a < X86Env.W) (hb : b < X86Env.W) :
    let f := (X86Int.aluF .sub a b).2
    X86Int.cond f 12 = some (decide (toSigned64 a < toSigned64 b)) := by
  dsimp only
  simp [X86Int.cond, X86Int.aluF, toSigned64, Nat.mod_eq_of_lt ha, Nat.mod_eq_of_lt hb]
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hmsb : ∀ v < X86Env.W, X86Int.msb v = decide (2 ^ 63 ≤ v) := by
    intro v hv
    unfold X86Int.msb
    by_cases h : v < 2 ^ 63
    · have : v / 2 ^ 63 = 0 := Nat.div_eq_of_lt h
      simp [this, h]
    · have hge : 2 ^ 63 ≤ v := Nat.le_of_not_lt h
      have : v / 2 ^ 63 = 1 := by
        have : X86Env.W = 2 * 2 ^ 63 := by decide
        omega
      simp [this, hge]
  have hres_lt : (a + X86Env.W - b) % X86Env.W < X86Env.W := Nat.mod_lt _ (by decide)
  rw [hmsb a ha, hmsb b hb, hmsb _ hres_lt]
  by_cases ha0 : a < 2 ^ 63 <;> by_cases hb0 : b < 2 ^ 63
  · have hno : (2 ^ 63 ≤ a) = false := by simp [ha0]
    have hnb : (2 ^ 63 ≤ b) = false := by simp [hb0]
    simp only [hno, hnb, bne_self_eq_false, Bool.false_and, Bool.bne_false]
    by_cases hab : a < b
    · have hm : (a + X86Env.W - b) % X86Env.W = a + X86Env.W - b :=
        Nat.mod_eq_of_lt (by omega)
      have hge : 2 ^ 63 ≤ a + X86Env.W - b := by
        have : X86Env.W = 2 * 2 ^ 63 := by decide
        omega
      simp [hm, hge]; omega
    · have hm : (a + X86Env.W - b) % X86Env.W = a - b := by
        have : a + X86Env.W - b = (a - b) + X86Env.W := by omega
        rw [this, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
      have hlt : ¬ (2 ^ 63 ≤ a - b) := by omega
      simp [hm, hlt]; omega
  · have hno : (2 ^ 63 ≤ a) = false := by simp [ha0]
    have hnb : (2 ^ 63 ≤ b) = true := by simp [Nat.le_of_not_lt hb0]
    simp [hno, hnb, ha0, hb0]; omega
  · have hno : (2 ^ 63 ≤ a) = true := by simp [Nat.le_of_not_lt ha0]
    have hnb : (2 ^ 63 ≤ b) = false := by simp [hb0]
    simp [hno, hnb, ha0, hb0]; omega
  · have hno : (2 ^ 63 ≤ a) = true := by simp [Nat.le_of_not_lt ha0]
    have hnb : (2 ^ 63 ≤ b) = true := by simp [Nat.le_of_not_lt hb0]
    simp only [hno, hnb, bne_self_eq_false, Bool.false_and, Bool.bne_false]
    by_cases hab : a < b
    · have hm : (a + X86Env.W - b) % X86Env.W = a + X86Env.W - b :=
        Nat.mod_eq_of_lt (by omega)
      have hge : 2 ^ 63 ≤ a + X86Env.W - b := by
        have : X86Env.W = 2 * 2 ^ 63 := by decide
        omega
      simp [hm, hge, ha0, hb0]; omega
    · have hm : (a + X86Env.W - b) % X86Env.W = a - b := by
        have : a + X86Env.W - b = (a - b) + X86Env.W := by omega
        rw [this, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
      have hlt : ¬ (2 ^ 63 ≤ a - b) := by omega
      simp [hm, hlt, ha0, hb0]; omega

theorem toSigned64_le_iff (a b : Nat) (ha : a < X86Env.W) (hb : b < X86Env.W) :
    let f := (X86Int.aluF .sub a b).2
    X86Int.cond f 14 = some (decide (toSigned64 a ≤ toSigned64 b)) := by
  have he := cond_aluF_sub_e a b ha hb
  have hl := toSigned64_lt_iff a b ha hb
  dsimp only at hl ⊢
  simp only [X86Int.cond, X86Int.aluF, bind, Option.bind, pure, Option.some.injEq] at he hl ⊢
  have h4 : (4 % 2 == 0) = true := rfl
  have h12 : (12 % 2 == 0) = true := rfl
  have h14 : (14 % 2 == 0) = true := rfl
  rw [h4] at he; rw [h12] at hl; rw [h14]
  simp only [↓reduceIte] at he hl ⊢
  rw [he, hl]
  have hinj : toSigned64 a = toSigned64 b → a = b := by
    intro h
    unfold toSigned64 at h
    dsimp only at h
    rw [Nat.mod_eq_of_lt ha, Nat.mod_eq_of_lt hb] at h
    by_cases ha0 : a < 2 ^ 63 <;> by_cases hb0 : b < 2 ^ 63 <;> simp [ha0, hb0] at h <;> omega
  by_cases hlt : toSigned64 a < toSigned64 b <;> by_cases heq : a = b
  · subst heq; exact absurd hlt (Int.lt_irrefl _)
  · have hle : toSigned64 a ≤ toSigned64 b := Int.le_of_lt hlt
    simp [hlt, heq, hle]
  · subst heq; simp
  · have hnot : ¬ (toSigned64 a ≤ toSigned64 b) := by
      intro hle
      rcases Int.lt_or_eq_of_le hle with h | h
      · exact hlt h
      · exact heq (hinj h)
    simp [hlt, heq, hnot]

/-! ### Mutable byte-array updates over disjoint `Mem` regions -/

def writeBytes (d : List Nat) (off : Nat) : List Nat → List Nat
  | [] => d
  | b :: bs => writeBytes (d.set off (b % 256)) (off + 1) bs

theorem writeBytes_length (d : List Nat) (off : Nat) (bs : List Nat) :
    (writeBytes d off bs).length = d.length := by
  induction bs generalizing d off with
  | nil => rfl
  | cons b bs ih => simp [writeBytes, ih, List.length_set]

theorem writeBytes_getElem?_lt (d : List Nat) (off j : Nat) (bs : List Nat) (hj : j < off) :
    (writeBytes d off bs)[j]? = d[j]? := by
  induction bs generalizing d off with
  | nil => rfl
  | cons b bs ih =>
    simp only [writeBytes]
    rw [ih (d.set off (b % 256)) (off + 1) (by omega), List.getElem?_set_ne (by omega)]

theorem writeBytes_getElem?_ge (d : List Nat) (off j : Nat) (bs : List Nat)
    (hj : off + bs.length ≤ j) :
    (writeBytes d off bs)[j]? = d[j]? := by
  induction bs generalizing d off with
  | nil => rfl
  | cons b bs ih =>
    simp only [writeBytes, List.length_cons] at hj ⊢
    rw [ih (d.set off (b % 256)) (off + 1) (by omega), List.getElem?_set_ne (by omega)]

theorem writeBytes_getElem?_inside (d : List Nat) (off k : Nat) (bs : List Nat)
    (hk : k < bs.length) (hd : off + bs.length ≤ d.length) :
    (writeBytes d off bs)[off + k]? = some (bs[k] % 256) := by
  induction bs generalizing d off k with
  | nil => exact absurd hk (Nat.not_lt_zero _)
  | cons b bs ih =>
    cases k with
    | zero =>
      simp only [writeBytes, Nat.add_zero]
      rw [writeBytes_getElem?_lt _ (off + 1) off _ (by omega)]
      simp [List.getElem?_set_self (show off < d.length by simp at hd; omega)]
    | succ k =>
      simp only [writeBytes, List.length_cons] at hk hd ⊢
      have hsub : off + (k + 1) = (off + 1) + k := by omega
      have hd' : off + 1 + bs.length ≤ (d.set off (b % 256)).length := by
        simp [List.length_set]; omega
      rw [hsub]
      exact ih (d.set off (b % 256)) (off + 1) k (by omega) hd'

theorem list_take_drop_eq_of_getElem? (l₁ l₂ : List Nat)
    (h : ∀ (j : Nat) (hj : j < l₂.length), l₁[j]? = some (l₂[j]'hj))
    (hlen : l₂.length ≤ l₁.length) :
    l₁.take l₂.length = l₂ := by
  apply List.ext_getElem
  · simp [List.length_take]; omega
  · intro i h1 h2
    simp only [List.length_take] at h1
    rw [List.getElem_take]
    have hg := h i h2
    rw [List.getElem?_eq_getElem (by omega), Option.some.injEq] at hg
    exact hg

theorem writeBytes_drop_take (d : List Nat) (off : Nat) (bs : List Nat)
    (hd : off + bs.length ≤ d.length) :
    ((writeBytes d off bs).drop off).take bs.length = bs.map (· % 256) := by
  have htake : ((writeBytes d off bs).drop off).take (bs.map (· % 256)).length = bs.map (· % 256) := by
    apply list_take_drop_eq_of_getElem?
    · intro j hj
      simp only [List.length_map] at hj
      rw [List.getElem?_drop, writeBytes_getElem?_inside d off j bs hj hd]
      simp [List.getElem_map]
    · simp [List.length_drop, writeBytes_length]; omega
  simpa [List.length_map] using htake

theorem mem_find_at_idx (m : Mem) (i : Nat) (r : Region) (a : Nat)
    (hw : m.wf) (hget : m[i]? = some r) (hlo : r.base ≤ a) (hhi : a < r.hi) :
    Mem.find m a 0 = some (i, r) := by
  have hm : r ∈ m := List.mem_of_getElem? hget
  obtain ⟨k, hfk⟩ := mem_find_of_mem m r a 0 hw.2 hm hlo hhi
  simp only [Nat.zero_add] at hfk
  have ⟨_, hilt, hgetk⟩ := mem_find_idx_lt m a 0 k r hfk
  simp only [Nat.sub_zero] at hilt hgetk
  have hi_lt : i < m.length := by
    cases h : m[i]? with
    | none => rw [h] at hget; cases hget
    | some _ => exact List.getElem?_eq_some_iff.mp hget |>.1
  by_cases hki : k = i
  · subst hki; exact hfk
  · have hdisj := List.pairwise_iff_getElem.mp hw.2
    have hmi : m[i] = r := by simpa [List.getElem?_eq_getElem hi_lt] using hget
    have hmk : m[k] = r := by simpa [List.getElem?_eq_getElem hilt] using hgetk
    have hnz : 0 < r.data.length := by simp [Region.hi] at hhi; omega
    rcases Nat.lt_or_gt_of_ne hki with hlt | hgt
    · have hd := hdisj k i hilt hi_lt hlt
      rw [hmk, hmi] at hd
      simp [Region.hi] at hd
      omega
    · have hd := hdisj i k hi_lt hilt hgt
      rw [hmi, hmk] at hd
      simp [Region.hi] at hd
      omega

theorem mem_set_other_mem_base {m : Mem} {i : Nat} {r r' s : Region}
    (hget : m[i]? = some r) (hs : s ∈ m) (hne : s.base ≠ r.base) :
    s ∈ m.set i r' := by
  rw [List.mem_iff_getElem] at hs ⊢
  obtain ⟨j, hj, rfl⟩ := hs
  refine ⟨j, by simpa using hj, ?_⟩
  by_cases hji : j = i
  · subst hji
    have : m[j] = r := by simpa [List.getElem?_eq_getElem hj] using hget
    exact absurd (congrArg Region.base this) hne
  · simp [List.getElem_set_ne (Ne.symm hji)]

theorem mem_wrN_at_idx : ∀ (bs : List Nat) (m : Mem) (i : Nat) (r : Region) (a : Nat),
    m.wf → m[i]? = some r → r.rw = true → r.base ≤ a → a + bs.length ≤ r.hi →
    m.wrN a bs = some (m.set i { r with data := writeBytes r.data (a - r.base) bs })
  | [], m, i, r, a, _, hget, _, _, _ => by
    have hi_lt : i < m.length := (List.getElem?_eq_some_iff.mp hget).1
    have hmi : m[i] = r := by simpa [List.getElem?_eq_getElem hi_lt] using hget
    have hreg : ({ r with data := r.data } : Region) = r := by cases r; rfl
    simp only [Mem.wrN, writeBytes]
    rw [hreg, ← hmi, List.set_getElem_self]
  | b :: bs, m, i, r, a, hw, hget, hrw, hlo, hhi => by
    have hf := mem_find_at_idx m i r a hw hget hlo (by simp [Region.hi] at hhi ⊢; omega)
    let r₁ : Region := { r with data := r.data.set (a - r.base) (b % 256) }
    let m₁ : Mem := m.set i r₁
    have hwr1 : m.wr a b = some m₁ := by simp [Mem.wr, hf, hrw, r₁, m₁]
    have hw1 : m₁.wf := wf_of_shape (wr_shape hwr1) hw
    have hi_lt : i < m.length := (List.getElem?_eq_some_iff.mp hget).1
    have hget1 : m₁[i]? = some r₁ := by simp [m₁, List.getElem?_set_self hi_lt]
    have hhi1 : a + 1 + bs.length ≤ r₁.hi := by
      have hrlen : r₁.data.length = r.data.length := by dsimp [r₁]; simp [List.length_set]
      dsimp only [Region.hi] at hhi ⊢
      rw [hrlen]
      have : r₁.base = r.base := rfl
      rw [this]
      simp [List.length_cons] at hhi
      omega
    have hlo1 : r₁.base ≤ a + 1 := by dsimp [r₁]; omega
    have hih := mem_wrN_at_idx bs m₁ i r₁ (a + 1) hw1 hget1 hrw hlo1 hhi1
    have hsub : a + 1 - r₁.base = (a - r.base) + 1 := by dsimp [r₁]; omega
    simp only [Mem.wrN, hwr1, bind, Option.bind, hih, m₁, List.set_set, writeBytes, hsub]
    rfl

theorem mem_wrN_region {m : Mem} {r : Region} {a : Nat} {bs : List Nat}
    (hw : m.wf) (hm : r ∈ m) (hrw : r.rw = true)
    (hlo : r.base ≤ a) (hhi : a + bs.length ≤ r.hi) :
    ∃ (i : Nat) (m' : Mem),
      m.wrN a bs = some m' ∧
      m[i]? = some r ∧
      m' = m.set i { r with data := writeBytes r.data (a - r.base) bs } ∧
      shape m' = shape m ∧
      m'.wf ∧
      { r with data := writeBytes r.data (a - r.base) bs } ∈ m' ∧
      (∀ s ∈ m, s.base ≠ r.base → s ∈ m') := by
  obtain ⟨i, hi, hceq⟩ := List.mem_iff_getElem.mp hm
  have hget : m[i]? = some r := by simp [List.getElem?_eq_getElem hi, hceq]
  have hwr := mem_wrN_at_idx bs m i r a hw hget hrw hlo hhi
  have hsh : shape (m.set i { r with data := writeBytes r.data (a - r.base) bs }) = shape m := by
    apply List.ext_getElem
    · simp [shape]
    · intro j h1 h2
      simp only [shape, List.length_map] at h1 h2
      by_cases hji : j = i
      · subst hji
        simp [shape, List.getElem_set_self, Region.hi, writeBytes_length, hceq]
      · simp [shape, List.getElem_set_ne (Ne.symm hji)]
  refine ⟨i, m.set i { r with data := writeBytes r.data (a - r.base) bs }, hwr, hget, rfl, hsh, wf_of_shape hsh hw, ?_, ?_⟩
  · rw [List.mem_iff_getElem]
    exact ⟨i, by simpa using hi, by simp [List.getElem_set_self]⟩
  · intro s hs hne
    exact mem_set_other_mem_base hget hs hne

theorem wr64_region {m : Mem} {r : Region} {a v : Nat}
    (hw : m.wf) (hm : r ∈ m) (hrw : r.rw = true) (hlo : r.base ≤ a) (hhi : a + 8 ≤ r.hi) :
    ∃ (i : Nat) (m' : Mem),
      wr64 m a v = some m' ∧
      m[i]? = some r ∧
      m' = m.set i { r with data := writeBytes r.data (a - r.base) (bytes8 v) } ∧
      m'.wf ∧
      { r with data := writeBytes r.data (a - r.base) (bytes8 v) } ∈ m' ∧
      (∀ s ∈ m, s.base ≠ r.base → s ∈ m') := by
  have h8 : a + (bytes8 v).length ≤ r.hi := by simp [bytes8_length]; exact hhi
  obtain ⟨i, m', hwr, hget, hm', _, hwf', hr', hoth⟩ := mem_wrN_region hw hm hrw hlo h8
  exact ⟨i, m', hwr, hget, hm', hwf', hr', hoth⟩

theorem wrMem_w8_region {m : Mem} {r : Region} {a v : Nat}
    (hw : m.wf) (hm : r ∈ m) (hrw : r.rw = true) (hlo : r.base ≤ a) (hhi : a + 1 ≤ r.hi) :
    ∃ (i : Nat) (m' : Mem),
      wrMem .w8 m a v = some m' ∧
      m[i]? = some r ∧
      m' = m.set i { r with data := writeBytes r.data (a - r.base) [v % 256] } ∧
      m'.wf ∧
      { r with data := writeBytes r.data (a - r.base) [v % 256] } ∈ m' ∧
      (∀ s ∈ m, s.base ≠ r.base → s ∈ m') := by
  have h1 : a + [v % 256].length ≤ r.hi := by simpa using hhi
  obtain ⟨i, m', hwr, hget, hm', _, hwf', hr', hoth⟩ := mem_wrN_region hw hm hrw hlo h1
  have hwr8 : wrMem .w8 m a v = some m' := by
    simp only [Mem.wrN, bind, Option.bind] at hwr
    cases hw1 : m.wr a (v % 256) with
    | none => rw [hw1] at hwr; cases hwr
    | some m1 =>
      rw [hw1] at hwr
      simp only [Option.some.injEq] at hwr
      subst hwr
      simpa [wrMem] using hw1
  exact ⟨i, m', hwr8, hget, hm', hwf', hr', hoth⟩

theorem wrMem_w16_region {m : Mem} {r : Region} {a v : Nat}
    (hw : m.wf) (hm : r ∈ m) (hrw : r.rw = true) (hlo : r.base ≤ a) (hhi : a + 2 ≤ r.hi) :
    ∃ (i : Nat) (m' : Mem),
      wrMem .w16 m a v = some m' ∧
      m[i]? = some r ∧
      m' = m.set i { r with data := writeBytes r.data (a - r.base) (bytes2 v) } ∧
      m'.wf ∧
      { r with data := writeBytes r.data (a - r.base) (bytes2 v) } ∈ m' ∧
      (∀ s ∈ m, s.base ≠ r.base → s ∈ m') := by
  have h2 : a + (bytes2 v).length ≤ r.hi := by simp [bytes2_length]; exact hhi
  obtain ⟨i, m', hwr, hget, hm', _, hwf', hr', hoth⟩ := mem_wrN_region hw hm hrw hlo h2
  exact ⟨i, m', by simpa [wrMem, wr16] using hwr, hget, hm', hwf', hr', hoth⟩

theorem wrMem_w32_region {m : Mem} {r : Region} {a v : Nat}
    (hw : m.wf) (hm : r ∈ m) (hrw : r.rw = true) (hlo : r.base ≤ a) (hhi : a + 4 ≤ r.hi) :
    ∃ (i : Nat) (m' : Mem),
      wrMem .w32 m a v = some m' ∧
      m[i]? = some r ∧
      m' = m.set i { r with data := writeBytes r.data (a - r.base) (bytes4 v) } ∧
      m'.wf ∧
      { r with data := writeBytes r.data (a - r.base) (bytes4 v) } ∈ m' ∧
      (∀ s ∈ m, s.base ≠ r.base → s ∈ m') := by
  have h4 : a + (bytes4 v).length ≤ r.hi := by simp [bytes4_length]; exact hhi
  obtain ⟨i, m', hwr, hget, hm', _, hwf', hr', hoth⟩ := mem_wrN_region hw hm hrw hlo h4
  exact ⟨i, m', by simpa [wrMem, wr32] using hwr, hget, hm', hwf', hr', hoth⟩

theorem wrMem_w64_region {m : Mem} {r : Region} {a v : Nat}
    (hw : m.wf) (hm : r ∈ m) (hrw : r.rw = true) (hlo : r.base ≤ a) (hhi : a + 8 ≤ r.hi) :
    ∃ (i : Nat) (m' : Mem),
      wrMem .w64 m a v = some m' ∧
      m[i]? = some r ∧
      m' = m.set i { r with data := writeBytes r.data (a - r.base) (bytes8 v) } ∧
      m'.wf ∧
      { r with data := writeBytes r.data (a - r.base) (bytes8 v) } ∈ m' ∧
      (∀ s ∈ m, s.base ≠ r.base → s ∈ m') :=
  wr64_region hw hm hrw hlo hhi

/-! ### Rodata read helpers -/

theorem roQwords_rd64 {m : Mem} (hc : CodeMem m) {addr count i v : Nat}
    (hlo : vmCodeEnd ≤ addr) (hhi : addr + 8 * count ≤ baseAddr + SporosVmBin.vmBytes.length)
    (hi : i < count)
    (hget : (roQwords addr count)[i]? = some v) :
    rd64 m (addr + 8 * i) = some v := by
  have hro_len : 8 * count ≤ (SporosVmBin.rodataBytes.drop (addr - vmCodeEnd)).length := by
    rw [List.length_drop, ← vmBytes_drop_ro, List.length_drop]
    have := code_end_sub_base
    omega
  have hq := takeQwords_getElem? count i (SporosVmBin.rodataBytes.drop (addr - vmCodeEnd)) hi hro_len
  rw [show roQwords addr count = takeQwords count (SporosVmBin.rodataBytes.drop (addr - vmCodeEnd)) from rfl] at hget
  rw [hq, Option.some.injEq] at hget
  have hdrop : ((SporosVmBin.rodataBytes.drop (addr - vmCodeEnd)).drop (8 * i)).take 8 =
      roSlice (addr + 8 * i) 8 := by
    unfold roSlice
    rw [List.drop_drop]
    have hadd : (addr - vmCodeEnd) + 8 * i = addr + 8 * i - vmCodeEnd := by omega
    rw [hadd]
  rw [hdrop, roSlice_eq_sliceAt (by omega)] at hget
  have hbv : baseAddr ≤ vmCodeEnd := by decide
  exact hc.rd64_eq (by omega) (by omega) hget

theorem roSlice_rd8 {m : Mem} (hc : CodeMem m) {addr n i b : Nat}
    (hlo : vmCodeEnd ≤ addr) (hhi : addr + n ≤ baseAddr + SporosVmBin.vmBytes.length)
    (hi : i < n)
    (hget : (roSlice addr n)[i]? = some b) :
    rdMem .w8 m (addr + i) = some b := by
  have hslice := roSlice_eq_sliceAt hlo (n := n)
  rw [hslice] at hget
  unfold sliceAt at hget
  have hbv : baseAddr ≤ vmCodeEnd := by decide
  have hlt : addr + i - baseAddr < SporosVmBin.vmBytes.length := by omega
  have hsub : addr - baseAddr + i = addr + i - baseAddr := by omega
  rw [List.getElem?_take, if_pos hi, List.getElem?_drop, hsub, List.getElem?_eq_getElem hlt, Option.some.injEq] at hget
  exact hc.rd8_eq (by omega) (by omega) hget

/-! ### Register-file well-formedness & step helpers -/

def RegsOk (s : X86Vm.St) : Prop :=
  s.regs.length = 16 ∧ ∀ r, r < 16 → r ≠ 11 → (s.reg r).isSome

theorem regsOk_set {s : X86Vm.St} (hd : RegsOk s) {r : Nat} (hr : r < 16) (v : Nat) :
    RegsOk (s.set r (some v)) := by
  refine ⟨by simp [X86Int.St.set, hd.1], ?_⟩
  intro r' hr' hne
  by_cases hrr : r' = r
  · subst hrr
    simp [X86Int.St.set, X86Int.St.reg, List.getElem?_set_self (by rw [hd.1]; exact hr')]
  · simp [X86Int.St.set, X86Int.St.reg, List.getElem?_set_ne (Ne.symm hrr)]
    exact hd.2 r' hr' hne

theorem regsOk_syscall_regs {s : X86Vm.St} (hd : RegsOk s) :
    (∃ v0, s.reg 0 = some v0) ∧ (∃ v7, s.reg 7 = some v7) ∧
    (∃ v6, s.reg 6 = some v6) ∧ (∃ v2, s.reg 2 = some v2) ∧
    (∃ v10, s.reg 10 = some v10) ∧ (∃ v8, s.reg 8 = some v8) ∧
    (∃ v9, s.reg 9 = some v9) := by
  have mk : ∀ r, r < 16 → r ≠ 11 → ∃ v, s.reg r = some v := by
    intro r hr hne
    have hs := hd.2 r hr hne
    cases h : s.reg r with
    | none => simp [h] at hs
    | some v => exact ⟨v, rfl⟩
  exact ⟨mk 0 (by decide) (by decide), mk 7 (by decide) (by decide),
    mk 6 (by decide) (by decide), mk 2 (by decide) (by decide),
    mk 10 (by decide) (by decide), mk 8 (by decide) (by decide),
    mk 9 (by decide) (by decide)⟩

theorem regsOk_with_rip {s : X86Vm.St} (hd : RegsOk s) (r' : Nat) :
    RegsOk { s with rip := r' } := hd

theorem regsOk_with_fl {s : X86Vm.St} (hd : RegsOk s) (f' : X86Int.Flags) :
    RegsOk { s with fl := f' } := hd

theorem regsOk_with_mem {s : X86Vm.St} (hd : RegsOk s) (m' : Mem) :
    RegsOk { s with mem := m' } := hd

theorem regsOk_with_k {s : X86Vm.St} (hd : RegsOk s) (k' : Kern) :
    RegsOk { s with k := k' } := hd

theorem regsOk_set_none_11 {s : X86Vm.St} (hd : RegsOk s) :
    RegsOk (s.set 11 none) := by
  refine ⟨by simp [X86Int.St.set, hd.1], ?_⟩
  intro r hr hne
  simp [X86Int.St.set, X86Int.St.reg, List.getElem?_set_ne (Ne.symm hne)]
  exact hd.2 r hr hne

theorem reg_set_self {s : X86Vm.St} {r : Nat} (hr : r < s.regs.length) (v : Option Nat) :
    (s.set r v).reg r = v := by
  simp [X86Int.St.set, X86Int.St.reg, List.getElem?_set_self hr]

theorem reg_set_ne {s : X86Vm.St} {r r' : Nat} (hne : r ≠ r') (v : Option Nat) :
    (s.set r v).reg r' = s.reg r' := by
  simp [X86Int.St.set, X86Int.St.reg, List.getElem?_set_ne hne]

theorem regs_len_set {s : X86Vm.St} {r : Nat} {v : Option Nat} :
    (s.set r v).regs.length = s.regs.length := by
  simp [X86Int.St.set]

theorem step_at_chunk (k : Nat) {a : Nat} {c : List (Nat × X86Vm.Ins × Nat)}
    {s : X86Vm.St} {rip : Nat} {ins : X86Vm.Ins} {len : Nat}
    (hrip : s.rip = rip)
    (hc : SporosVmBin.vmTableChunks[k]? = some (a, c))
    (he : (rip, ins, len) ∈ c) :
    X86Vm.step vmImage s = X86Vm.exec ins len s := by
  rw [← hrip] at he
  exact step_of_chunk k hc he

theorem aluF_add_val (a b : Nat) : (X86Int.aluF .add a b).1 = (a + b) % X86Env.W := rfl
theorem aluF_sub_val (a b : Nat) : (X86Int.aluF .sub a b).1 = (a + X86Env.W - b) % X86Env.W := rfl
theorem aluF_and_val (a b : Nat) : (X86Int.aluF .and a b).1 = Nat.land a b := rfl
theorem aluF_or_val (a b : Nat) : (X86Int.aluF .or a b).1 = Nat.lor a b := rfl
theorem aluF_xor_val (a b : Nat) : (X86Int.aluF .xor a b).1 = Nat.xor a b := rfl

theorem aluF_xor_self_val (a : Nat) : (X86Int.aluF .xor a a).1 = 0 := by
  simp [X86Int.aluF]

theorem msb_lt_2p63 {a : Nat} (ha : a < 2 ^ 63) : X86Int.msb a = false := by
  simp [X86Int.msb, Nat.div_eq_of_lt ha]

theorem msb_neg_enomem : X86Int.msb (X86Env.neg X86Env.ENOMEM) = true := by
  decide

theorem exec_movI_lt {d imm len : Nat} {s : X86Vm.St} (hi : imm < X86Env.W) :
    X86Vm.exec (.ofInt (.movI d imm)) len s = .ok { s.set d (some imm) with rip := s.rip + len } := by
  rw [exec_movI, Nat.mod_eq_of_lt hi]

theorem exec_shlI_lt {d c len a : Nat} {s : X86Vm.St} (ha : s.reg d = some a)
    (hsh : a * 2 ^ (c % 64) < X86Env.W) :
    X86Vm.exec (.ofInt (.shlI d c)) len s =
      .ok { s.set d (some (a * 2 ^ (c % 64))) with fl := X86Int.Flags.undef, rip := s.rip + len } := by
  rw [exec_shlI ha, Nat.mod_eq_of_lt hsh]

theorem exec_aluI_add_lt {d imm len a : Nat} {s : X86Vm.St} (ha : s.reg d = some a)
    (hi : imm < X86Env.W) :
    X86Vm.exec (.ofInt (.aluI .add d imm)) len s =
      .ok { s.set d (some ((a + imm) % X86Env.W)) with
            fl := (X86Int.aluF .add a imm).2, rip := s.rip + len } := by
  simp [X86Vm.exec, X86Int.exec, ha, Nat.mod_eq_of_lt hi, X86Int.lift, bind, Option.bind, pure,
    aluF_add_val]

theorem exec_alu_xor_self {d len a : Nat} {s : X86Vm.St} (ha : s.reg d = some a) :
    X86Vm.exec (.ofInt (.alu .xor d d)) len s =
      .ok { s.set d (some 0) with fl := (X86Int.aluF .xor a a).2, rip := s.rip + len } := by
  simp [X86Vm.exec, X86Int.exec, ha, X86Int.lift, bind, Option.bind, pure, aluF_xor_self_val]

theorem exec_shlI_1_lt {d len a : Nat} {s : X86Vm.St} (ha : s.reg d = some a)
    (hsh : a * 2 < X86Env.W) :
    X86Vm.exec (.ofInt (.shlI d 1)) len s =
      .ok { s.set d (some (a * 2)) with fl := X86Int.Flags.undef, rip := s.rip + len } := by
  simpa [show 2 ^ (1 % 64) = 2 from by decide] using exec_shlI_lt (c := 1) ha (by simpa using hsh)

theorem exec_shlI_4_lt {d len a : Nat} {s : X86Vm.St} (ha : s.reg d = some a)
    (hsh : a * 16 < X86Env.W) :
    X86Vm.exec (.ofInt (.shlI d 4)) len s =
      .ok { s.set d (some (a * 16)) with fl := X86Int.Flags.undef, rip := s.rip + len } := by
  simpa [show 2 ^ (4 % 64) = 16 from by decide] using exec_shlI_lt (c := 4) ha (by simpa using hsh)

theorem getElem?_set_getD_self {l : List (Option Nat)} {r : Nat} {v : Option Nat} (h : r < l.length) :
    ((l.set r v)[r]?).getD none = v := by
  rw [List.getElem?_set_self h]; rfl

theorem Reaches.step_trans {im : Nat → Option (X86Vm.Ins × Nat)} {s₀ s₁ s₂ : X86Vm.St}
    (hstep : X86Vm.step im s₀ = .ok s₁) (hrest : Reaches im s₁ s₂) : Reaches im s₀ s₂ :=
  Reaches.trans (Reaches.step hstep) hrest

theorem Ends.step_trans {im : Nat → Option (X86Vm.Ins × Nat)} {s₀ s₁ : X86Vm.St} {o : X86Vm.Out}
    (hstep : X86Vm.step im s₀ = .ok s₁) (hrest : Ends im s₁ o) : Ends im s₀ o :=
  Ends.of_reaches (Reaches.step hstep) hrest

/-! ### Additional `writeBytes` and region read/write helpers -/

theorem writeBytes_getElem?_disjoint (d : List Nat) (off j : Nat) (bs : List Nat)
    (hdisj : j < off ∨ off + bs.length ≤ j) :
    (writeBytes d off bs)[j]? = d[j]? := by
  rcases hdisj with h | h
  · exact writeBytes_getElem?_lt d off j bs h
  · exact writeBytes_getElem?_ge d off j bs h

theorem writeBytes_drop_take_disjoint (d : List Nat) (off off2 n : Nat) (bs : List Nat)
    (hdisj : off2 + n ≤ off ∨ off + bs.length ≤ off2)
    (hlen : off2 + n ≤ d.length) :
    ((writeBytes d off bs).drop off2).take n = (d.drop off2).take n := by
  have h1 : (((writeBytes d off bs).drop off2).take n).length = n := by
    simp [List.length_take, List.length_drop, writeBytes_length]; omega
  have h2 : ((d.drop off2).take n).length = n := by
    simp [List.length_take, List.length_drop]; omega
  apply List.ext_getElem (by omega)
  intro i hi1 hi2
  rw [List.getElem_take, List.getElem_drop, List.getElem_take, List.getElem_drop]
  have hget := writeBytes_getElem?_disjoint d off (off2 + i) bs (by
    rcases hdisj with h | h
    · left; omega
    · right; omega)
  have hl1 : off2 + i < (writeBytes d off bs).length := by
    rw [writeBytes_length]; omega
  have hl2 : off2 + i < d.length := by omega
  rw [List.getElem?_eq_getElem hl1, List.getElem?_eq_getElem hl2, Option.some.injEq] at hget
  exact hget

theorem leVal_writeBytes_same (d : List Nat) (off v : Nat) (hlen : off + 8 ≤ d.length) :
    leVal (((writeBytes d off (bytes8 v)).drop off).take 8) = v % X86Env.W := by
  have h8 : (bytes8 v).length = 8 := bytes8_length v
  rw [← h8, writeBytes_drop_take d off (bytes8 v) (by rw [h8]; exact hlen),
    bytes8_map_mod, leVal_bytes8]

theorem rd64_of_writeBytes_same {r : Region} {m' : Mem} {a v : Nat}
    (hm' : m'.wf) (hr' : { r with data := writeBytes r.data (a - r.base) (bytes8 v) } ∈ m')
    (hlo : r.base ≤ a) (hhi : a + 8 ≤ r.hi) :
    rd64 m' a = some (v % X86Env.W) := by
  have hlen : a - r.base + 8 ≤ r.data.length := by simp [Region.hi] at hhi; omega
  refine rd64_of_mem hm' hr' hlo (by simp [Region.hi, writeBytes_length]; simpa [Region.hi] using hhi) ?_
  simpa using leVal_writeBytes_same r.data (a - r.base) v hlen

theorem mem_eraseIdx_ne_base : ∀ (m : Mem) (idx : Nat) {r u : Region},
    m.wf → m[idx]? = some r → (∀ v ∈ m, 0 < v.data.length) →
    u ∈ m.eraseIdx idx → u.base ≠ r.base
  | [], _, _, _, _, _, _, hu => by simp at hu
  | x :: xs, 0, r, u, hw, hget, hpos, hu => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at hget
    subst hget
    simp only [List.eraseIdx_cons_zero] at hu
    have hrpos := hpos x (by simp)
    have hupos := hpos u (by simp [hu])
    have hdisj := (List.pairwise_cons.mp hw.2).1 u hu
    intro hb
    rcases hdisj with h1 | h1 <;> simp [Region.hi] at h1 <;> omega
  | x :: xs, idx + 1, r, u, hw, hget, hpos, hu => by
    simp only [List.eraseIdx_cons_succ, List.mem_cons] at hu
    simp only [List.getElem?_cons_succ] at hget
    rcases hu with rfl | hu'
    · have hupos := hpos u (by simp)
      have hrpos := hpos r (List.mem_cons_of_mem _ (List.mem_of_getElem? hget))
      have hdisj := (List.pairwise_cons.mp hw.2).1 r (List.mem_of_getElem? hget)
      intro hb
      rcases hdisj with h1 | h1 <;> simp [Region.hi] at h1 <;> omega
    · exact mem_eraseIdx_ne_base xs idx ⟨fun a ha => hw.1 a (by simp [ha]), (List.pairwise_cons.mp hw.2).2⟩
        hget (fun v hv => hpos v (by simp [hv])) hu'

theorem mem_eraseIdx_free {m : Mem} {idx : Nat} {r : Region} {a n : Nat}
    (hw : m.wf) (hget : m[idx]? = some r)
    (hpos : ∀ u ∈ m, 0 < u.data.length)
    (hfree : ∀ u ∈ m, u.base ≠ r.base → a + n ≤ u.base ∨ u.hi ≤ a) :
    X86Env.Mem.free (m.eraseIdx idx) a n = true := by
  rw [X86Env.Mem.free, List.all_eq_true]
  intro u hu
  have hum : u ∈ m := List.mem_of_mem_eraseIdx hu
  have hne : u.base ≠ r.base := mem_eraseIdx_ne_base m idx hw hget hpos hu
  rcases hfree u hum hne with h1 | h2
  · simp [h1]
  · simp [h2]

theorem rd64_stack_push_preserve {m m' : Mem} {stk : Region} {sp v : Nat}
    (hw : m.wf) (hstk : stk ∈ m) (hrw : stk.rw = true)
    (h8 : 8 ≤ sp) (hspW : sp < X86Env.W)
    (hlo : stk.base ≤ sp - 8) (hhi : sp ≤ stk.hi)
    (hwr : wr64 m ((sp + X86Env.W - 8) % X86Env.W) v = some m') :
    ∃ stk' ∈ m',
      stk'.base = stk.base ∧ stk'.rw = true ∧ stk'.data.length = stk.data.length ∧
      (∀ j, j < sp - 8 - stk.base ∨ sp - 8 - stk.base + 8 ≤ j → stk'.data[j]? = stk.data[j]?) ∧
      m'.wf ∧
      (∀ u ∈ m, u.base ≠ stk.base → u ∈ m') ∧
      rd64 m' ((sp + X86Env.W - 8) % X86Env.W) = some (v % X86Env.W) := by
  have hea : (sp + X86Env.W - 8) % X86Env.W = sp - 8 := by
    have : sp + X86Env.W - 8 = (sp - 8) + X86Env.W := by omega
    rw [this, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  rw [hea] at hwr ⊢
  obtain ⟨i, m1, hwr1, _, hm1, hwf1, hr1, hoth1⟩ :=
    wr64_region hw hstk hrw hlo (by omega : sp - 8 + 8 ≤ stk.hi) (v := v)
  rw [hwr1, Option.some.injEq] at hwr
  subst hwr
  let stk' := { stk with data := writeBytes stk.data (sp - 8 - stk.base) (bytes8 v) }
  refine ⟨stk', hr1, rfl, hrw, writeBytes_length _ _ _, ?_, hwf1, hoth1, ?_⟩
  · intro j hj
    apply writeBytes_getElem?_disjoint
    rw [bytes8_length]
    exact hj
  · exact rd64_of_writeBytes_same hwf1 hr1 hlo (by omega)

theorem mem_append_wf {m : Mem} {a n : Nat}
    (hw : m.wf) (ha : a % page = 0) (hf : m.free a n = true) :
    ((m ++ [(⟨a, List.replicate n 0, true⟩ : Region)]) : Mem).wf := by
  have hlen : (⟨a, List.replicate n 0, true⟩ : Region).data.length = n := List.length_replicate
  exact X86Env.wf_append hw ha (by simpa [hlen] using hf)

theorem mem_mremap_wf {m : Mem} {idx n : Nat} {r : Region} {a : Nat}
    (hw : m.wf) (ha : a % page = 0)
    (hf : X86Env.Mem.free (m.eraseIdx idx) a n = true)
    (hle : r.data.length ≤ n) :
    X86Env.Mem.wf ((m.eraseIdx idx) ++ [(⟨a, r.data ++ List.replicate (n - r.data.length) 0, true⟩ : Region)]) := by
  have hlen : (⟨a, r.data ++ List.replicate (n - r.data.length) 0, true⟩ : Region).data.length = n := by
    simp [List.length_append, List.length_replicate]; omega
  rw [← hlen] at hf
  exact X86Env.wf_append (X86Env.wf_erase hw idx) ha hf

theorem rd64_of_mem_transfer {m m' : Mem} {r : Region} {a v : Nat}
    (hw : m.wf) (hw' : m'.wf) (hr : r ∈ m) (hr' : r ∈ m')
    (hlo : r.base ≤ a) (hhi : a + 8 ≤ r.hi) (hrd : rd64 m a = some v) :
    rd64 m' a = some v := by
  obtain ⟨u, hu⟩ := show ∃ u, leVal ((r.data.drop (a - r.base)).take 8) = u from ⟨_, rfl⟩
  have hold := rd64_of_mem hw hr hlo hhi hu
  rw [hrd, Option.some.injEq] at hold
  subst hold
  exact rd64_of_mem hw' hr' hlo hhi hu

private theorem ctxSize_pos : 0 < ctxSize := by decide
private theorem ctxSize_ge_8 : 8 ≤ ctxSize := by decide
private theorem ctxSize_ge_16 : 16 ≤ ctxSize := by decide
private theorem ctxSize_ge_24 : 24 ≤ ctxSize := by decide
private theorem vm_lim_lt_W : 2 ^ 47 + 16 ≤ X86Env.W := by decide

/-! ### Runtime helper machine specifications (`sporos_vm`) -/

theorem fn_write_stderr_spec {s : X86Vm.St} {sp retAddr : Nat}
    (hrip : s.rip = SporosVmBin.fn_write_stderr)
    (hregs : RegsOk s)
    (hsp : s.reg 4 = some sp)
    (hret : rd64 s.mem sp = some retAddr) :
    ∃ s', Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.mem = s.mem ∧ s'.k = s.k ∧
      RegsOk s' ∧ s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 4 → r ≠ 7 → r ≠ 11 → s'.reg r = s.reg r) := by
  have h01 : X86Vm.step vmImage s =
      .ok { s.set 7 (some 2) with rip := 4201400 } := by
    rw [step_at_chunk 23 hrip rfl (by decide : (SporosVmBin.fn_write_stderr, (.ofInt (.movI 7 2)), 5) ∈ SporosVmBin.vmTable_23),
      exec_movI_lt (by decide), hrip]
    rfl
  let s1 := { s.set 7 (some 2) with rip := 4201400 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) 2
  have h02 : X86Vm.step vmImage s1 =
      .ok { s1.set 0 (some 1) with rip := 4201405 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201400, (.ofInt (.movI 0 1)), 5) ∈ SporosVmBin.vmTable_23),
      exec_movI_lt (by decide)]
  let s2 := { s1.set 0 (some 1) with rip := 4201405 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) 1
  obtain ⟨⟨v0, hv0⟩, ⟨v7, hv7⟩, ⟨v6, hv6⟩, ⟨v2, hv2⟩, ⟨v10, hv10⟩, ⟨v8, hv8⟩, ⟨v9, hv9⟩⟩ :=
    regsOk_syscall_regs hr2
  have hv0' : v0 = 1 := by
    have hs : s2.reg 0 = some 1 := by
      simp only [s2, X86Int.St.reg, X86Int.St.set]
      rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
    rw [hs] at hv0; cases hv0; rfl
  have hv7' : v7 = 2 := by
    have hs : s2.reg 7 = some 2 := by
      simp only [s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 7)]
      rw [getElem?_set_getD_self (by rw [hregs.1]; decide)]
    rw [hs] at hv7; cases hv7; rfl
  subst hv0' hv7'
  have hsys : X86Env.syscall s2.k s2.mem 1 2 v6 v2 v10 v8 v9 =
      some (.ret (X86Env.neg X86Env.EBADF) s2.k s2.mem) := by
    simp [X86Env.syscall]
  have h03 : X86Vm.step vmImage s2 =
      .ok { ((s2.set 0 (some (X86Env.neg X86Env.EBADF))).set 1 (some 4201407)).set 11 none with
            k := s2.k, mem := s2.mem, rip := 4201407 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201405, (.ofInt .syscall), 2) ∈ SporosVmBin.vmTable_23),
      exec_syscall_ret hv0 hv7 hv6 hv2 hv10 hv8 hv9 hsys]
  let s3 := { ((s2.set 0 (some (X86Env.neg X86Env.EBADF))).set 1 (some 4201407)).set 11 none with
                k := s2.k, mem := s2.mem, rip := 4201407 }
  have hr3 : RegsOk s3 :=
    regsOk_set_none_11 (regsOk_set (regsOk_set hr2 (by decide) _) (by decide) 4201407)
  have hsp3 : s3.reg 4 = some sp := by
    simp only [s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 11 ≠ 4),
      List.getElem?_set_ne (by decide : 1 ≠ 4), List.getElem?_set_ne (by decide : 0 ≠ 4),
      List.getElem?_set_ne (by decide : 7 ≠ 4)]
    exact hsp
  have hm3 : rd64 s3.mem sp = some retAddr := hret
  have h04 : X86Vm.step vmImage s3 =
      .ok { s3.set 4 (some ((sp + 8) % X86Env.W)) with rip := retAddr } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201407, (.ofInt .ret), 1) ∈ SporosVmBin.vmTable_23),
      exec_ret hsp3 hm3]
  let s4 := { s3.set 4 (some ((sp + 8) % X86Env.W)) with rip := retAddr }
  refine ⟨s4, Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step h04))),
    rfl, rfl, rfl, regsOk_set hr3 (by decide) _, ?_, ?_⟩
  · simp only [s4, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
  · intro r h0 h1 h4 h7 h11
    simp only [s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne h4.symm, List.getElem?_set_ne h11.symm,
      List.getElem?_set_ne h1.symm, List.getElem?_set_ne h0.symm,
      List.getElem?_set_ne h7.symm]

private theorem fn_write_stderr_cstr_loop_spec :
    ∀ (rem : Nat) (s : X86Vm.St) {sp retAddr ptr off : Nat} {bs : List Nat},
      rem = bs.length - off → off ≤ bs.length → bs.length < X86Env.W →
      s.rip = 4201413 → RegsOk s → s.reg 4 = some sp → rd64 s.mem sp = some retAddr →
      s.reg 6 = some ptr → s.reg 2 = some off →
      (∀ i < bs.length, rdMem .w8 s.mem (eaIdx ptr i 1 0) = some (bs[i]!) ∧ bs[i]! ≠ 0 ∧ bs[i]! < 256) →
      rdMem .w8 s.mem (eaIdx ptr bs.length 1 0) = some 0 →
      ∃ s', Reaches vmImage s s' ∧
        s'.rip = retAddr ∧ s'.mem = s.mem ∧ s'.k = s.k ∧
        RegsOk s' ∧ s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
        (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 4 → r ≠ 6 → r ≠ 7 → r ≠ 11 → s'.reg r = s.reg r)
  | 0, s, sp, retAddr, ptr, off, bs, hrem, hle, _hW, hrip, hregs, hsp, hret, hr6, hr2, _hbs, hnul => by
    have hoff : off = bs.length := by omega
    subst hoff
    have h01 : X86Vm.step vmImage s =
        .ok { s.set 0 (some 0) with rip := 4201417 } := by
      rw [step_at_chunk 23 hrip rfl (by decide : (4201413, (.ldIdx .w8 0 6 2 1 0), 4) ∈ SporosVmBin.vmTable_23),
        exec_ldIdx hr6 hr2 hnul, hrip]
    let s1 := { s.set 0 (some 0) with rip := 4201417 }
    have hr1 : RegsOk s1 := regsOk_set hregs (by decide) 0
    have hs1_0 : s1.reg 0 = some 0 := by
      simp only [s1, X86Int.St.reg, X86Int.St.set]
      rw [getElem?_set_getD_self (by rw [hregs.1]; decide)]
    have h02 : X86Vm.step vmImage s1 =
        .ok { s1 with fl := (X86Int.aluF .and 0 0).2, rip := 4201420 } := by
      rw [step_at_chunk 23 rfl rfl (by decide : (4201417, (.ofInt (.test 0 0)), 3) ∈ SporosVmBin.vmTable_23),
        exec_test hs1_0 hs1_0]
    let s2 := { s1 with fl := (X86Int.aluF .and 0 0).2, rip := 4201420 }
    have hr2' : RegsOk s2 := hr1
    have h03 : X86Vm.step vmImage s2 = .ok { s2 with rip := 4201428 } := by
      rw [step_at_chunk 23 rfl rfl (by decide : (4201420, (.ofInt (.jcc 4 4201428)), 2) ∈ SporosVmBin.vmTable_23),
        exec_jcc (c := true) (cond_aluF_and_self_e 0)]
      rfl
    let s3 := { s2 with rip := 4201428 }
    have hr3 : RegsOk s3 := hr2'
    have hs3_2 : s3.reg 2 = some bs.length := by
      simp only [s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 2)]
      exact hr2
    have h04 : X86Vm.step vmImage s3 =
        .ok { s3 with fl := (X86Int.aluF .and bs.length bs.length).2, rip := 4201431 } := by
      rw [step_at_chunk 23 rfl rfl (by decide : (4201428, (.ofInt (.test 2 2)), 3) ∈ SporosVmBin.vmTable_23),
        exec_test hs3_2 hs3_2]
    let s4 := { s3 with fl := (X86Int.aluF .and bs.length bs.length).2, rip := 4201431 }
    have hr4 : RegsOk s4 := hr3
    by_cases hlen0 : bs.length = 0
    · have hc4 : X86Int.cond s4.fl 4 = some true := by
        dsimp [s4]; rw [cond_aluF_and_self_e, decide_eq_true hlen0]
      have h05 : X86Vm.step vmImage s4 = .ok { s4 with rip := 4201445 } := by
        rw [step_at_chunk 23 rfl rfl (by decide : (4201431, (.ofInt (.jcc 4 4201445)), 2) ∈ SporosVmBin.vmTable_23),
          exec_jcc hc4]
        rfl
      let s5 := { s4 with rip := 4201445 }
      have hr5 : RegsOk s5 := hr4
      have hsp5 : s5.reg 4 = some sp := by
        simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 4)]
        exact hsp
      have h06 : X86Vm.step vmImage s5 =
          .ok { s5.set 4 (some ((sp + 8) % X86Env.W)) with rip := retAddr } := by
        rw [step_at_chunk 23 rfl rfl (by decide : (4201445, (.ofInt .ret), 1) ∈ SporosVmBin.vmTable_23),
          exec_ret hsp5 hret]
      let s6 := { s5.set 4 (some ((sp + 8) % X86Env.W)) with rip := retAddr }
      refine ⟨s6,
        Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
          (Reaches.step_trans h05 (Reaches.step h06))))),
        rfl, rfl, rfl, regsOk_set hr5 (by decide) _, ?_, ?_⟩
      · simp only [s6, X86Int.St.reg, X86Int.St.set]
        rw [getElem?_set_getD_self (by rw [hr5.1]; decide)]
      · intro r h0 _ _ h4 _ _ _
        simp only [s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
          List.getElem?_set_ne h4.symm, List.getElem?_set_ne h0.symm]
    · have hc4 : X86Int.cond s4.fl 4 = some false := by
        dsimp [s4]; rw [cond_aluF_and_self_e, decide_eq_false hlen0]
      have h05 : X86Vm.step vmImage s4 = .ok { s4 with rip := 4201433 } := by
        rw [step_at_chunk 23 rfl rfl (by decide : (4201431, (.ofInt (.jcc 4 4201445)), 2) ∈ SporosVmBin.vmTable_23),
          exec_jcc hc4]
        rfl
      let s5 := { s4 with rip := 4201433 }
      have hr5 : RegsOk s5 := hr4
      have h06 : X86Vm.step vmImage s5 =
          .ok { s5.set 7 (some 2) with rip := 4201438 } := by
        rw [step_at_chunk 23 rfl rfl (by decide : (4201433, (.ofInt (.movI 7 2)), 5) ∈ SporosVmBin.vmTable_23),
          exec_movI_lt (by decide)]
      let s6 := { s5.set 7 (some 2) with rip := 4201438 }
      have hr6' : RegsOk s6 := regsOk_set hr5 (by decide) 2
      have h07 : X86Vm.step vmImage s6 =
          .ok { s6.set 0 (some 1) with rip := 4201443 } := by
        rw [step_at_chunk 23 rfl rfl (by decide : (4201438, (.ofInt (.movI 0 1)), 5) ∈ SporosVmBin.vmTable_23),
          exec_movI_lt (by decide)]
      let s7 := { s6.set 0 (some 1) with rip := 4201443 }
      have hr7' : RegsOk s7 := regsOk_set hr6' (by decide) 1
      obtain ⟨⟨v0, hv0⟩, ⟨v7, hv7⟩, ⟨v6, hv6⟩, ⟨v2, hv2⟩, ⟨v10, hv10⟩, ⟨v8, hv8⟩, ⟨v9, hv9⟩⟩ :=
        regsOk_syscall_regs hr7'
      have hv0' : v0 = 1 := by
        have hs : s7.reg 0 = some 1 := by
          simp only [s7, X86Int.St.reg, X86Int.St.set]
          rw [getElem?_set_getD_self (by rw [hr6'.1]; decide)]
        rw [hs] at hv0; cases hv0; rfl
      have hv7' : v7 = 2 := by
        have hs : s7.reg 7 = some 2 := by
          simp only [s7, s6, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 7)]
          rw [getElem?_set_getD_self (by rw [hr5.1]; decide)]
        rw [hs] at hv7; cases hv7; rfl
      subst hv0' hv7'
      have hsys : X86Env.syscall s7.k s7.mem 1 2 v6 v2 v10 v8 v9 =
          some (.ret (X86Env.neg X86Env.EBADF) s7.k s7.mem) := by
        simp [X86Env.syscall]
      have h08 : X86Vm.step vmImage s7 =
          .ok { ((s7.set 0 (some (X86Env.neg X86Env.EBADF))).set 1 (some 4201445)).set 11 none with
                k := s7.k, mem := s7.mem, rip := 4201445 } := by
        rw [step_at_chunk 23 rfl rfl (by decide : (4201443, (.ofInt .syscall), 2) ∈ SporosVmBin.vmTable_23),
          exec_syscall_ret hv0 hv7 hv6 hv2 hv10 hv8 hv9 hsys]
      let s8 := { ((s7.set 0 (some (X86Env.neg X86Env.EBADF))).set 1 (some 4201445)).set 11 none with
                    k := s7.k, mem := s7.mem, rip := 4201445 }
      have hr8' : RegsOk s8 :=
        regsOk_set_none_11 (regsOk_set (regsOk_set hr7' (by decide) _) (by decide) 4201445)
      have hsp8 : s8.reg 4 = some sp := by
        simp only [s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
          List.getElem?_set_ne (by decide : 11 ≠ 4), List.getElem?_set_ne (by decide : 1 ≠ 4),
          List.getElem?_set_ne (by decide : 0 ≠ 4), List.getElem?_set_ne (by decide : 7 ≠ 4)]
        exact hsp
      have h09 : X86Vm.step vmImage s8 =
          .ok { s8.set 4 (some ((sp + 8) % X86Env.W)) with rip := retAddr } := by
        rw [step_at_chunk 23 rfl rfl (by decide : (4201445, (.ofInt .ret), 1) ∈ SporosVmBin.vmTable_23),
          exec_ret hsp8 hret]
      let s9 := { s8.set 4 (some ((sp + 8) % X86Env.W)) with rip := retAddr }
      refine ⟨s9,
        Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
          (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step_trans h08
            (Reaches.step h09)))))))),
        rfl, rfl, rfl, regsOk_set hr8' (by decide) _, ?_, ?_⟩
      · simp only [s9, X86Int.St.reg, X86Int.St.set]
        rw [getElem?_set_getD_self (by rw [hr8'.1]; decide)]
      · intro r h0 h1 _ h4 _ h7 h11
        simp only [s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
          List.getElem?_set_ne h4.symm, List.getElem?_set_ne h11.symm,
          List.getElem?_set_ne h1.symm, List.getElem?_set_ne h0.symm,
          List.getElem?_set_ne h7.symm]
  | rem + 1, s, sp, retAddr, ptr, off, bs, hrem, hle, hW, hrip, hregs, hsp, hret, hr6, hr2, hbs, hnul => by
    have hoff_lt : off < bs.length := by omega
    obtain ⟨hm8, hnz, _⟩ := hbs off hoff_lt
    have h01 : X86Vm.step vmImage s =
        .ok { s.set 0 (some (bs[off]!)) with rip := 4201417 } := by
      rw [step_at_chunk 23 hrip rfl (by decide : (4201413, (.ldIdx .w8 0 6 2 1 0), 4) ∈ SporosVmBin.vmTable_23),
        exec_ldIdx hr6 hr2 hm8, hrip]
    let s1 := { s.set 0 (some (bs[off]!)) with rip := 4201417 }
    have hr1 : RegsOk s1 := regsOk_set hregs (by decide) _
    have hs1_0 : s1.reg 0 = some (bs[off]!) := by
      simp only [s1, X86Int.St.reg, X86Int.St.set]
      rw [getElem?_set_getD_self (by rw [hregs.1]; decide)]
    have h02 : X86Vm.step vmImage s1 =
        .ok { s1 with fl := (X86Int.aluF .and (bs[off]!) (bs[off]!)).2, rip := 4201420 } := by
      rw [step_at_chunk 23 rfl rfl (by decide : (4201417, (.ofInt (.test 0 0)), 3) ∈ SporosVmBin.vmTable_23),
        exec_test hs1_0 hs1_0]
    let s2 := { s1 with fl := (X86Int.aluF .and (bs[off]!) (bs[off]!)).2, rip := 4201420 }
    have hr2' : RegsOk s2 := hr1
    have hc4 : X86Int.cond s2.fl 4 = some false := by
      dsimp [s2]; rw [cond_aluF_and_self_e, decide_eq_false hnz]
    have h03 : X86Vm.step vmImage s2 = .ok { s2 with rip := 4201422 } := by
      rw [step_at_chunk 23 rfl rfl (by decide : (4201420, (.ofInt (.jcc 4 4201428)), 2) ∈ SporosVmBin.vmTable_23),
        exec_jcc hc4]
      rfl
    let s3 := { s2 with rip := 4201422 }
    have hr3 : RegsOk s3 := hr2'
    have hs3_2 : s3.reg 2 = some off := by
      simp only [s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 2)]
      exact hr2
    have h04 : X86Vm.step vmImage s3 =
        .ok { s3.set 2 (some (off + 1)) with
              fl := (X86Int.aluF .add off 1).2, rip := 4201426 } := by
      rw [step_at_chunk 23 rfl rfl (by decide : (4201422, (.ofInt (.aluI .add 2 1)), 4) ∈ SporosVmBin.vmTable_23),
        exec_aluI_add_lt hs3_2 (by decide), Nat.mod_eq_of_lt (by omega)]
    let s4 := { s3.set 2 (some (off + 1)) with fl := (X86Int.aluF .add off 1).2, rip := 4201426 }
    have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) _
    have h05 : X86Vm.step vmImage s4 = .ok { s4 with rip := 4201413 } := by
      rw [step_at_chunk 23 rfl rfl (by decide : (4201426, (.ofInt (.jmp 4201413)), 2) ∈ SporosVmBin.vmTable_23),
        exec_jmp]
    let s5 := { s4 with rip := 4201413 }
    have hr5 : RegsOk s5 := hr4
    have hsp5 : s5.reg 4 = some sp := by
      simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 2 ≠ 4), List.getElem?_set_ne (by decide : 0 ≠ 4)]
      exact hsp
    have hr5_6 : s5.reg 6 = some ptr := by
      simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 2 ≠ 6), List.getElem?_set_ne (by decide : 0 ≠ 6)]
      exact hr6
    have hr5_2 : s5.reg 2 = some (off + 1) := by
      simp only [s5, s4, X86Int.St.reg, X86Int.St.set]
      rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
    obtain ⟨sf, hreach, hripf, hmemf, hkf, hregsf, hspf, hothf⟩ :=
      fn_write_stderr_cstr_loop_spec rem s5 (sp := sp) (retAddr := retAddr) (ptr := ptr)
        (off := off + 1) (bs := bs) (by omega) (by omega) hW rfl hr5 hsp5 hret
        hr5_6 hr5_2 hbs hnul
    refine ⟨sf,
      Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
        (Reaches.step_trans h05 hreach)))),
      hripf, hmemf, hkf, hregsf, hspf, ?_⟩
    intro r h0 h1 h2 h4 h6 h7 h11
    rw [hothf r h0 h1 h2 h4 h6 h7 h11]
    simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne h2.symm, List.getElem?_set_ne h0.symm]

theorem fn_write_stderr_cstr_spec {s : X86Vm.St} {sp retAddr ptr : Nat} {bs : List Nat}
    (hrip : s.rip = SporosVmBin.fn_write_stderr_cstr)
    (hregs : RegsOk s)
    (hsp : s.reg 4 = some sp)
    (hret : rd64 s.mem sp = some retAddr)
    (hr7 : s.reg 7 = some ptr)
    (hW : bs.length < X86Env.W)
    (hbs : ∀ i < bs.length, rdMem .w8 s.mem (eaIdx ptr i 1 0) = some (bs[i]!) ∧ bs[i]! ≠ 0 ∧ bs[i]! < 256)
    (hnul : rdMem .w8 s.mem (eaIdx ptr bs.length 1 0) = some 0) :
    ∃ s', Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.mem = s.mem ∧ s'.k = s.k ∧
      RegsOk s' ∧ s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 4 → r ≠ 6 → r ≠ 7 → r ≠ 11 → s'.reg r = s.reg r) := by
  have h01 : X86Vm.step vmImage s =
      .ok { s.set 6 (some ptr) with rip := 4201411 } := by
    rw [step_at_chunk 23 hrip rfl (by decide : (SporosVmBin.fn_write_stderr_cstr, (.ofInt (.mov 6 7)), 3) ∈ SporosVmBin.vmTable_23),
      exec_mov hr7, hrip]
    rfl
  let s1 := { s.set 6 (some ptr) with rip := 4201411 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) ptr
  obtain ⟨v2, hv2⟩ : ∃ v2, s1.reg 2 = some v2 := by
    have hs := hr1.2 2 (by decide) (by decide)
    cases h : s1.reg 2 with
    | none => simp [h] at hs
    | some v => exact ⟨v, rfl⟩
  have h02 : X86Vm.step vmImage s1 =
      .ok { s1.set 2 (some 0) with fl := (X86Int.aluF .xor v2 v2).2, rip := 4201413 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201411, (.ofInt (.alu .xor 2 2)), 2) ∈ SporosVmBin.vmTable_23),
      exec_alu_xor_self hv2]
  let s2 := { s1.set 2 (some 0) with fl := (X86Int.aluF .xor v2 v2).2, rip := 4201413 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) 0
  have hs2_4 : s2.reg 4 = some sp := by
    simp only [s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 2 ≠ 4), List.getElem?_set_ne (by decide : 6 ≠ 4)]
    exact hsp
  have hs2_6 : s2.reg 6 = some ptr := by
    simp only [s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 6)]
    rw [getElem?_set_getD_self (by rw [hregs.1]; decide)]
  have hs2_2 : s2.reg 2 = some 0 := by
    simp only [s2, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  obtain ⟨sf, hreach, hripf, hmemf, hkf, hregsf, hspf, hothf⟩ :=
    fn_write_stderr_cstr_loop_spec bs.length s2 (sp := sp) (retAddr := retAddr) (ptr := ptr)
      (off := 0) (bs := bs) (by simp) (Nat.zero_le _) hW rfl hr2 hs2_4 hret
      hs2_6 hs2_2 hbs hnul
  refine ⟨sf, Reaches.step_trans h01 (Reaches.step_trans h02 hreach),
    hripf, hmemf, hkf, hregsf, hspf, ?_⟩
  intro r h0 h1 h2 h4 h6 h7 h11
  rw [hothf r h0 h1 h2 h4 h6 h7 h11]
  simp only [s2, s1, X86Int.St.reg, X86Int.St.set,
    List.getElem?_set_ne h2.symm, List.getElem?_set_ne h6.symm]

theorem L_exit_nomem_spec {s : X86Vm.St} {sp : Nat} {stk : Region}
    (hrip : s.rip = SporosVmBin.L_exit_nomem)
    (hregs : RegsOk s)
    (hsp : s.reg 4 = some sp)
    (h8 : 8 ≤ sp) (hspW : sp < X86Env.W)
    (hmwf : s.mem.wf)
    (hstk : stk ∈ s.mem) (hrw : stk.rw = true) (hlo : stk.base ≤ sp - 8) (hhi : sp ≤ stk.hi) :
    Ends vmImage s (.exit 2) := by
  have h01 : X86Vm.step vmImage s = .ok { s.set 6 (some 4206959) with rip := 4201295 } := by
    rw [step_at_chunk 23 hrip rfl (by decide : (SporosVmBin.L_exit_nomem, (.ofInt (.movI 6 4206959)), 5) ∈ SporosVmBin.vmTable_23),
      exec_movI_lt (by decide), hrip]
    rfl
  let s1 := { s.set 6 (some 4206959) with rip := 4201295 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) 4206959
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 2 (some 25) with rip := 4201300 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201295, (.ofInt (.movI 2 25)), 5) ∈ SporosVmBin.vmTable_23),
      exec_movI_lt (by decide)]
  let s2 := { s1.set 2 (some 25) with rip := 4201300 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) 25
  have hea : (sp + X86Env.W - 8) % X86Env.W = sp - 8 := by
    have : sp + X86Env.W - 8 = (sp - 8) + X86Env.W := by omega
    rw [this, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  obtain ⟨i, m3, hwr3, _, _, hwf3, hr3_stk, _⟩ :=
    wr64_region hmwf hstk hrw hlo (by omega : sp - 8 + 8 ≤ stk.hi) (v := 4201305)
  have hwr3' : wr64 s2.mem ((sp + X86Env.W - 8) % X86Env.W) 4201305 = some m3 := by
    simpa [s2, s1, hea] using hwr3
  have hs2_4 : s2.reg 4 = some sp := by
    simp only [s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 2 ≠ 4), List.getElem?_set_ne (by decide : 6 ≠ 4)]
    exact hsp
  have h03 : X86Vm.step vmImage s2 =
      .ok { s2.set 4 (some (sp - 8)) with mem := m3, rip := 4201395 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201300, (.ofInt (.call 4201395)), 5) ∈ SporosVmBin.vmTable_23),
      exec_call hs2_4 hwr3']
    simp [s2, hea]
  let s3 := { s2.set 4 (some (sp - 8)) with mem := m3, rip := 4201395 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) _
  have hs3_4 : s3.reg 4 = some (sp - 8) := by
    simp only [s3, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
  have hm3_ret : rd64 s3.mem (sp - 8) = some 4201305 := by
    have hr := rd64_of_writeBytes_same hwf3 hr3_stk hlo (by omega : sp - 8 + 8 ≤ stk.hi)
    simpa [s3, show (4201305 % X86Env.W) = 4201305 from by decide] using hr
  obtain ⟨s4, hreach, hrip4, _, _, hregs4, _, _⟩ :=
    fn_write_stderr_spec (s := s3) (sp := sp - 8) (retAddr := 4201305) rfl hr3 hs3_4 hm3_ret
  have h05 : X86Vm.step vmImage s4 = .ok { s4.set 7 (some 2) with rip := 4201310 } := by
    rw [step_at_chunk 23 hrip4 rfl (by decide : (4201305, (.ofInt (.movI 7 2)), 5) ∈ SporosVmBin.vmTable_23),
      exec_movI_lt (by decide), hrip4]
  let s5 := { s4.set 7 (some 2) with rip := 4201310 }
  have hr5 : RegsOk s5 := regsOk_set hregs4 (by decide) 2
  have h06 : X86Vm.step vmImage s5 = .ok { s5.set 0 (some 60) with rip := 4201315 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201310, (.ofInt (.movI 0 60)), 5) ∈ SporosVmBin.vmTable_23),
      exec_movI_lt (by decide)]
  let s6 := { s5.set 0 (some 60) with rip := 4201315 }
  have hr6 : RegsOk s6 := regsOk_set hr5 (by decide) 60
  obtain ⟨⟨v0, hv0⟩, ⟨v7, hv7⟩, ⟨v6, hv6⟩, ⟨v2, hv2⟩, ⟨v10, hv10⟩, ⟨v8, hv8⟩, ⟨v9, hv9⟩⟩ :=
    regsOk_syscall_regs hr6
  have hv0' : v0 = 60 := by
    have hs : s6.reg 0 = some 60 := by
      simp only [s6, X86Int.St.reg, X86Int.St.set]
      rw [getElem?_set_getD_self (by rw [hr5.1]; decide)]
    rw [hs] at hv0; cases hv0; rfl
  have hv7' : v7 = 2 := by
    have hs : s6.reg 7 = some 2 := by
      simp only [s6, s5, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 7)]
      rw [getElem?_set_getD_self (by rw [hregs4.1]; decide)]
    rw [hs] at hv7; cases hv7; rfl
  subst hv0' hv7'
  have h07 : X86Vm.step vmImage s6 = .exit 2 := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201315, (.ofInt .syscall), 2) ∈ SporosVmBin.vmTable_23),
      exec_syscall_exit (c := 2) hv0 hv7 hv6 hv2 hv10 hv8 hv9 (by simp [X86Env.syscall])]
  exact Ends.step_trans h01 (Ends.step_trans h02 (Ends.step_trans h03
    (Ends.of_reaches hreach (Ends.step_trans h05 (Ends.step_trans h06 (Ends.step h07))))))

theorem fn_mmap_ok_spec {s : X86Vm.St} {sp retAddr len a : Nat} {stk : Region}
    (hrip : s.rip = SporosVmBin.fn_mmap)
    (hregs : RegsOk s)
    (hsp : s.reg 4 = some sp)
    (hstk : stk ∈ s.mem) (hmwf : s.mem.wf) (hlo : stk.base ≤ sp) (hhi : sp + 8 ≤ stk.hi)
    (hret : rd64 s.mem sp = some retAddr)
    (hr7 : s.reg 7 = some len)
    (hlen : 0 < len)
    (hor : (s.k.next).1 = some (.at a))
    (ha_pg : a % X86Env.page = 0) (ha_pos : 0 < a) (ha_lim : a + X86Env.pages len ≤ 2 ^ 47)
    (hfree : s.mem.free a (X86Env.pages len) = true) :
    let k1 := (s.k.next).2
    let m1 : Mem := s.mem ++ [⟨a, List.replicate (X86Env.pages len) 0, true⟩]
    ∃ s', Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.k = k1 ∧ s'.mem = m1 ∧ m1.wf ∧
      RegsOk s' ∧ s'.reg 0 = some a ∧ s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 4 → r ≠ 6 → r ≠ 7 → r ≠ 8 → r ≠ 9 → r ≠ 10 → r ≠ 11 → s'.reg r = s.reg r) := by
  have h01 : X86Vm.step vmImage s = .ok { s.set 6 (some len) with rip := 4201320 } := by
    rw [step_at_chunk 23 hrip rfl (by decide : (SporosVmBin.fn_mmap, (.ofInt (.mov 6 7)), 3) ∈ SporosVmBin.vmTable_23),
      exec_mov hr7, hrip]
    rfl
  let s1 := { s.set 6 (some len) with rip := 4201320 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) len
  have hs1_7 : s1.reg 7 = some len := by
    simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 6 ≠ 7)]
    exact hr7
  have h02 : X86Vm.step vmImage s1 =
      .ok { s1.set 7 (some 0) with fl := (X86Int.aluF .xor len len).2, rip := 4201322 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201320, (.ofInt (.alu .xor 7 7)), 2) ∈ SporosVmBin.vmTable_23),
      exec_alu_xor_self hs1_7]
  let s2 := { s1.set 7 (some 0) with fl := (X86Int.aluF .xor len len).2, rip := 4201322 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) 0
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 2 (some 3) with rip := 4201327 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201322, (.ofInt (.movI 2 3)), 5) ∈ SporosVmBin.vmTable_23),
      exec_movI_lt (by decide)]
  let s3 := { s2.set 2 (some 3) with rip := 4201327 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) 3
  have h04 : X86Vm.step vmImage s3 = .ok { s3.set 10 (some 34) with rip := 4201333 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201327, (.ofInt (.movI 10 34)), 6) ∈ SporosVmBin.vmTable_23),
      exec_movI_lt (by decide)]
  let s4 := { s3.set 10 (some 34) with rip := 4201333 }
  have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) 34
  have h05 : X86Vm.step vmImage s4 =
      .ok { s4.set 8 (some (X86Env.W - 1)) with rip := 4201343 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201333, (.ofInt (.movI 8 18446744073709551615)), 10) ∈ SporosVmBin.vmTable_23),
      exec_movI, show 18446744073709551615 % X86Env.W = X86Env.W - 1 from by decide]
  let s5 := { s4.set 8 (some (X86Env.W - 1)) with rip := 4201343 }
  have hr5 : RegsOk s5 := regsOk_set hr4 (by decide) _
  obtain ⟨v9, hv9⟩ : ∃ v9, s5.reg 9 = some v9 := by
    have hs := hr5.2 9 (by decide) (by decide)
    cases h : s5.reg 9 with
    | none => simp [h] at hs
    | some v => exact ⟨v, rfl⟩
  have h06 : X86Vm.step vmImage s5 =
      .ok { s5.set 9 (some 0) with fl := (X86Int.aluF .xor v9 v9).2, rip := 4201346 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201343, (.ofInt (.alu .xor 9 9)), 3) ∈ SporosVmBin.vmTable_23),
      exec_alu_xor_self hv9]
  let s6 := { s5.set 9 (some 0) with fl := (X86Int.aluF .xor v9 v9).2, rip := 4201346 }
  have hr6 : RegsOk s6 := regsOk_set hr5 (by decide) 0
  have h07 : X86Vm.step vmImage s6 = .ok { s6.set 0 (some 9) with rip := 4201351 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201346, (.ofInt (.movI 0 9)), 5) ∈ SporosVmBin.vmTable_23),
      exec_movI_lt (by decide)]
  let s7 := { s6.set 0 (some 9) with rip := 4201351 }
  have hr7' : RegsOk s7 := regsOk_set hr6 (by decide) 9
  have hs7_0 : s7.reg 0 = some 9 := by
    simp only [s7, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr6.1]; decide)]
  have hs7_7 : s7.reg 7 = some 0 := by
    simp only [s7, s6, s5, s4, s3, s2, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 7), List.getElem?_set_ne (by decide : 9 ≠ 7),
      List.getElem?_set_ne (by decide : 8 ≠ 7), List.getElem?_set_ne (by decide : 10 ≠ 7),
      List.getElem?_set_ne (by decide : 2 ≠ 7)]
    rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  have hs7_6 : s7.reg 6 = some len := by
    simp only [s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 6), List.getElem?_set_ne (by decide : 9 ≠ 6),
      List.getElem?_set_ne (by decide : 8 ≠ 6), List.getElem?_set_ne (by decide : 10 ≠ 6),
      List.getElem?_set_ne (by decide : 2 ≠ 6), List.getElem?_set_ne (by decide : 7 ≠ 6)]
    rw [getElem?_set_getD_self (by rw [hregs.1]; decide)]
  have hs7_2 : s7.reg 2 = some 3 := by
    simp only [s7, s6, s5, s4, s3, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 2), List.getElem?_set_ne (by decide : 9 ≠ 2),
      List.getElem?_set_ne (by decide : 8 ≠ 2), List.getElem?_set_ne (by decide : 10 ≠ 2)]
    rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
  have hs7_10 : s7.reg 10 = some 34 := by
    simp only [s7, s6, s5, s4, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 10), List.getElem?_set_ne (by decide : 9 ≠ 10),
      List.getElem?_set_ne (by decide : 8 ≠ 10)]
    rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
  have hs7_8 : s7.reg 8 = some (X86Env.W - 1) := by
    simp only [s7, s6, s5, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 8), List.getElem?_set_ne (by decide : 9 ≠ 8)]
    rw [getElem?_set_getD_self (by rw [hr4.1]; decide)]
  have hs7_9 : s7.reg 9 = some 0 := by
    simp only [s7, s6, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 9)]
    rw [getElem?_set_getD_self (by rw [hr5.1]; decide)]
  have hlen_ne : len ≠ 0 := by omega
  have hplace : X86Env.place s.mem (s.k.next).1 (X86Env.pages len) = some (some a) := by
    rw [hor]
    simp [X86Env.place, ha_pg, ha_pos, ha_lim, hfree]
  have hsys : X86Env.syscall s7.k s7.mem 9 0 len 3 34 (X86Env.W - 1) 0 =
      some (.ret a (s.k.next).2 (s.mem ++ [(⟨a, List.replicate (X86Env.pages len) 0, true⟩ : Region)])) := by
    change X86Env.syscall s.k s.mem 9 0 len 3 34 (X86Env.W - 1) 0 = _
    simp [X86Env.syscall, X86Env.PROT_RW, X86Env.MAP_PRIVATE_ANON, hlen_ne]
    cases hn : s.k.next with
    | mk c k' =>
      have hc : c = some (.at a) := by simpa [hn] using hor
      subst hc
      have hpl' : X86Env.place s.mem (some (.at a)) (X86Env.pages len) = some (some a) := by
        simpa [hn] using hplace
      simp [hpl']
  let k1 := (s.k.next).2
  let m1 : Mem := s.mem ++ [⟨a, List.replicate (X86Env.pages len) 0, true⟩]
  have h08 : X86Vm.step vmImage s7 =
      .ok { ((s7.set 0 (some a)).set 1 (some 4201353)).set 11 none with
            k := k1, mem := m1, rip := 4201353 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201351, (.ofInt .syscall), 2) ∈ SporosVmBin.vmTable_23),
      exec_syscall_ret hs7_0 hs7_7 hs7_6 hs7_2 hs7_10 hs7_8 hs7_9 hsys]
  let s8 := { ((s7.set 0 (some a)).set 1 (some 4201353)).set 11 none with
                k := k1, mem := m1, rip := 4201353 }
  have hr8 : RegsOk s8 :=
    regsOk_set_none_11 (regsOk_set (regsOk_set hr7' (by decide) a) (by decide) 4201353)
  have hs8_0 : s8.reg 0 = some a := by
    simp only [s8, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 11 ≠ 0), List.getElem?_set_ne (by decide : 1 ≠ 0)]
    rw [getElem?_set_getD_self (by rw [hr7'.1]; decide)]
  have h09 : X86Vm.step vmImage s8 =
      .ok { s8 with fl := (X86Int.aluF .and a a).2, rip := 4201356 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201353, (.ofInt (.test 0 0)), 3) ∈ SporosVmBin.vmTable_23),
      exec_test hs8_0 hs8_0]
  let s9 := { s8 with fl := (X86Int.aluF .and a a).2, rip := 4201356 }
  have hr9 : RegsOk s9 := hr8
  have hmsb : X86Int.msb a = false := msb_lt_2p63 (by omega)
  have hc8 : X86Int.cond s9.fl 8 = some false := by
    simpa [s9, hmsb] using cond_aluF_and_self_s a
  have h10 : X86Vm.step vmImage s9 = .ok { s9 with rip := 4201362 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201356, (.ofInt (.jcc 8 4201290)), 6) ∈ SporosVmBin.vmTable_23),
      exec_jcc hc8]
    rfl
  let s10 := { s9 with rip := 4201362 }
  have hr10 : RegsOk s10 := hr9
  have hm1wf : m1.wf := mem_append_wf hmwf ha_pg hfree
  have hret_m1 : rd64 m1 sp = some retAddr :=
    rd64_of_mem_transfer hmwf hm1wf hstk (List.mem_append_left _ hstk) hlo hhi hret
  have hs10_4 : s10.reg 4 = some sp := by
    simp only [s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 11 ≠ 4), List.getElem?_set_ne (by decide : 1 ≠ 4),
      List.getElem?_set_ne (by decide : 0 ≠ 4), List.getElem?_set_ne (by decide : 9 ≠ 4),
      List.getElem?_set_ne (by decide : 8 ≠ 4), List.getElem?_set_ne (by decide : 10 ≠ 4),
      List.getElem?_set_ne (by decide : 2 ≠ 4), List.getElem?_set_ne (by decide : 7 ≠ 4),
      List.getElem?_set_ne (by decide : 6 ≠ 4)]
    exact hsp
  have h11 : X86Vm.step vmImage s10 =
      .ok { s10.set 4 (some ((sp + 8) % X86Env.W)) with rip := retAddr } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201362, (.ofInt .ret), 1) ∈ SporosVmBin.vmTable_23),
      exec_ret hs10_4 hret_m1]
  let s11 := { s10.set 4 (some ((sp + 8) % X86Env.W)) with rip := retAddr }
  refine ⟨s11,
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
      (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step_trans h08
        (Reaches.step_trans h09 (Reaches.step_trans h10 (Reaches.step h11)))))))))),
    rfl, rfl, rfl, hm1wf, regsOk_set hr10 (by decide) _, ?_, ?_, ?_⟩
  · simp only [s11, s10, s9, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 4 ≠ 0)]
    exact hs8_0
  · simp only [s11, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr10.1]; decide)]
  · intro r h0 h1 h2 h4 h6 h7 h8 h9 h10' h11'
    simp only [s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne h4.symm, List.getElem?_set_ne h11'.symm, List.getElem?_set_ne h1.symm,
      List.getElem?_set_ne h0.symm, List.getElem?_set_ne h9.symm, List.getElem?_set_ne h8.symm,
      List.getElem?_set_ne h10'.symm, List.getElem?_set_ne h2.symm, List.getElem?_set_ne h7.symm,
      List.getElem?_set_ne h6.symm]

theorem fn_mmap_nomem_spec {s : X86Vm.St} {sp len : Nat} {stk : Region}
    (hrip : s.rip = SporosVmBin.fn_mmap)
    (hregs : RegsOk s)
    (hsp : s.reg 4 = some sp)
    (h8 : 8 ≤ sp) (hspW : sp < X86Env.W)
    (hmwf : s.mem.wf)
    (hstk : stk ∈ s.mem) (hrw : stk.rw = true) (hlo : stk.base ≤ sp - 8) (hhi : sp ≤ stk.hi)
    (hr7 : s.reg 7 = some len)
    (hlen : 0 < len)
    (hor : (s.k.next).1 = some .nomem) :
    Ends vmImage s (.exit 2) := by
  have h01 : X86Vm.step vmImage s = .ok { s.set 6 (some len) with rip := 4201320 } := by
    rw [step_at_chunk 23 hrip rfl (by decide : (SporosVmBin.fn_mmap, (.ofInt (.mov 6 7)), 3) ∈ SporosVmBin.vmTable_23),
      exec_mov hr7, hrip]
    rfl
  let s1 := { s.set 6 (some len) with rip := 4201320 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) len
  have hs1_7 : s1.reg 7 = some len := by
    simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 6 ≠ 7)]
    exact hr7
  have h02 : X86Vm.step vmImage s1 =
      .ok { s1.set 7 (some 0) with fl := (X86Int.aluF .xor len len).2, rip := 4201322 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201320, (.ofInt (.alu .xor 7 7)), 2) ∈ SporosVmBin.vmTable_23),
      exec_alu_xor_self hs1_7]
  let s2 := { s1.set 7 (some 0) with fl := (X86Int.aluF .xor len len).2, rip := 4201322 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) 0
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 2 (some 3) with rip := 4201327 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201322, (.ofInt (.movI 2 3)), 5) ∈ SporosVmBin.vmTable_23),
      exec_movI_lt (by decide)]
  let s3 := { s2.set 2 (some 3) with rip := 4201327 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) 3
  have h04 : X86Vm.step vmImage s3 = .ok { s3.set 10 (some 34) with rip := 4201333 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201327, (.ofInt (.movI 10 34)), 6) ∈ SporosVmBin.vmTable_23),
      exec_movI_lt (by decide)]
  let s4 := { s3.set 10 (some 34) with rip := 4201333 }
  have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) 34
  have h05 : X86Vm.step vmImage s4 =
      .ok { s4.set 8 (some (X86Env.W - 1)) with rip := 4201343 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201333, (.ofInt (.movI 8 18446744073709551615)), 10) ∈ SporosVmBin.vmTable_23),
      exec_movI, show 18446744073709551615 % X86Env.W = X86Env.W - 1 from by decide]
  let s5 := { s4.set 8 (some (X86Env.W - 1)) with rip := 4201343 }
  have hr5 : RegsOk s5 := regsOk_set hr4 (by decide) _
  obtain ⟨v9, hv9⟩ : ∃ v9, s5.reg 9 = some v9 := by
    have hs := hr5.2 9 (by decide) (by decide)
    cases h : s5.reg 9 with
    | none => simp [h] at hs
    | some v => exact ⟨v, rfl⟩
  have h06 : X86Vm.step vmImage s5 =
      .ok { s5.set 9 (some 0) with fl := (X86Int.aluF .xor v9 v9).2, rip := 4201346 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201343, (.ofInt (.alu .xor 9 9)), 3) ∈ SporosVmBin.vmTable_23),
      exec_alu_xor_self hv9]
  let s6 := { s5.set 9 (some 0) with fl := (X86Int.aluF .xor v9 v9).2, rip := 4201346 }
  have hr6 : RegsOk s6 := regsOk_set hr5 (by decide) 0
  have h07 : X86Vm.step vmImage s6 = .ok { s6.set 0 (some 9) with rip := 4201351 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201346, (.ofInt (.movI 0 9)), 5) ∈ SporosVmBin.vmTable_23),
      exec_movI_lt (by decide)]
  let s7 := { s6.set 0 (some 9) with rip := 4201351 }
  have hr7' : RegsOk s7 := regsOk_set hr6 (by decide) 9
  have hs7_0 : s7.reg 0 = some 9 := by
    simp only [s7, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr6.1]; decide)]
  have hs7_7 : s7.reg 7 = some 0 := by
    simp only [s7, s6, s5, s4, s3, s2, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 7), List.getElem?_set_ne (by decide : 9 ≠ 7),
      List.getElem?_set_ne (by decide : 8 ≠ 7), List.getElem?_set_ne (by decide : 10 ≠ 7),
      List.getElem?_set_ne (by decide : 2 ≠ 7)]
    rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  have hs7_6 : s7.reg 6 = some len := by
    simp only [s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 6), List.getElem?_set_ne (by decide : 9 ≠ 6),
      List.getElem?_set_ne (by decide : 8 ≠ 6), List.getElem?_set_ne (by decide : 10 ≠ 6),
      List.getElem?_set_ne (by decide : 2 ≠ 6), List.getElem?_set_ne (by decide : 7 ≠ 6)]
    rw [getElem?_set_getD_self (by rw [hregs.1]; decide)]
  have hs7_2 : s7.reg 2 = some 3 := by
    simp only [s7, s6, s5, s4, s3, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 2), List.getElem?_set_ne (by decide : 9 ≠ 2),
      List.getElem?_set_ne (by decide : 8 ≠ 2), List.getElem?_set_ne (by decide : 10 ≠ 2)]
    rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
  have hs7_10 : s7.reg 10 = some 34 := by
    simp only [s7, s6, s5, s4, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 10), List.getElem?_set_ne (by decide : 9 ≠ 10),
      List.getElem?_set_ne (by decide : 8 ≠ 10)]
    rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
  have hs7_8 : s7.reg 8 = some (X86Env.W - 1) := by
    simp only [s7, s6, s5, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 8), List.getElem?_set_ne (by decide : 9 ≠ 8)]
    rw [getElem?_set_getD_self (by rw [hr4.1]; decide)]
  have hs7_9 : s7.reg 9 = some 0 := by
    simp only [s7, s6, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 9)]
    rw [getElem?_set_getD_self (by rw [hr5.1]; decide)]
  have hlen_ne : len ≠ 0 := by omega
  have hsys : X86Env.syscall s7.k s7.mem 9 0 len 3 34 (X86Env.W - 1) 0 =
      some (.ret (X86Env.neg X86Env.ENOMEM) (s.k.next).2 s.mem) := by
    change X86Env.syscall s.k s.mem 9 0 len 3 34 (X86Env.W - 1) 0 = _
    simp [X86Env.syscall, X86Env.PROT_RW, X86Env.MAP_PRIVATE_ANON, hlen_ne]
    cases hn : s.k.next with
    | mk c k' =>
      have hc : c = some .nomem := by simpa [hn] using hor
      subst hc
      simp [X86Env.place]
  have h08 : X86Vm.step vmImage s7 =
      .ok { ((s7.set 0 (some (X86Env.neg X86Env.ENOMEM))).set 1 (some 4201353)).set 11 none with
            k := (s.k.next).2, mem := s.mem, rip := 4201353 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201351, (.ofInt .syscall), 2) ∈ SporosVmBin.vmTable_23),
      exec_syscall_ret hs7_0 hs7_7 hs7_6 hs7_2 hs7_10 hs7_8 hs7_9 hsys]
  let s8 := { ((s7.set 0 (some (X86Env.neg X86Env.ENOMEM))).set 1 (some 4201353)).set 11 none with
                k := (s.k.next).2, mem := s.mem, rip := 4201353 }
  have hr8 : RegsOk s8 :=
    regsOk_set_none_11 (regsOk_set (regsOk_set hr7' (by decide) _) (by decide) 4201353)
  have hs8_0 : s8.reg 0 = some (X86Env.neg X86Env.ENOMEM) := by
    simp only [s8, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 11 ≠ 0), List.getElem?_set_ne (by decide : 1 ≠ 0)]
    rw [getElem?_set_getD_self (by rw [hr7'.1]; decide)]
  have h09 : X86Vm.step vmImage s8 =
      .ok { s8 with fl := (X86Int.aluF .and (X86Env.neg X86Env.ENOMEM) (X86Env.neg X86Env.ENOMEM)).2,
                    rip := 4201356 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201353, (.ofInt (.test 0 0)), 3) ∈ SporosVmBin.vmTable_23),
      exec_test hs8_0 hs8_0]
  let s9 := { s8 with fl := (X86Int.aluF .and (X86Env.neg X86Env.ENOMEM) (X86Env.neg X86Env.ENOMEM)).2,
                      rip := 4201356 }
  have hr9 : RegsOk s9 := hr8
  have hc8 : X86Int.cond s9.fl 8 = some true := by
    simpa [s9, msb_neg_enomem] using cond_aluF_and_self_s (X86Env.neg X86Env.ENOMEM)
  have h10 : X86Vm.step vmImage s9 = .ok { s9 with rip := 4201290 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201356, (.ofInt (.jcc 8 4201290)), 6) ∈ SporosVmBin.vmTable_23),
      exec_jcc hc8]
    rfl
  let s10 := { s9 with rip := 4201290 }
  have hr10 : RegsOk s10 := hr9
  have hs10_4 : s10.reg 4 = some sp := by
    simp only [s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 11 ≠ 4), List.getElem?_set_ne (by decide : 1 ≠ 4),
      List.getElem?_set_ne (by decide : 0 ≠ 4), List.getElem?_set_ne (by decide : 9 ≠ 4),
      List.getElem?_set_ne (by decide : 8 ≠ 4), List.getElem?_set_ne (by decide : 10 ≠ 4),
      List.getElem?_set_ne (by decide : 2 ≠ 4), List.getElem?_set_ne (by decide : 7 ≠ 4),
      List.getElem?_set_ne (by decide : 6 ≠ 4)]
    exact hsp
  have h_exit := L_exit_nomem_spec (s := s10) rfl hr10 hs10_4 h8 hspW hmwf hstk hrw hlo hhi
  exact Ends.step_trans h01 (Ends.step_trans h02 (Ends.step_trans h03 (Ends.step_trans h04
    (Ends.step_trans h05 (Ends.step_trans h06 (Ends.step_trans h07 (Ends.step_trans h08
      (Ends.step_trans h09 (Ends.step_trans h10 h_exit)))))))))

theorem fn_mremap_ok_spec {s : X86Vm.St} {sp retAddr oldPtr oldLen newLen idx a : Nat} {oldReg stk : Region}
    (hrip : s.rip = SporosVmBin.fn_mremap)
    (hregs : RegsOk s)
    (hsp : s.reg 4 = some sp)
    (hstk : stk ∈ s.mem) (hstkne : stk.base ≠ oldPtr) (hlo : stk.base ≤ sp) (hhi : sp + 8 ≤ stk.hi)
    (hret : rd64 s.mem sp = some retAddr)
    (hmwf : s.mem.wf)
    (hr7 : s.reg 7 = some oldPtr) (hr6 : s.reg 6 = some oldLen) (hr2 : s.reg 2 = some newLen)
    (hfind : s.mem.find oldPtr 0 = some (idx, oldReg))
    (hbase : oldReg.base = oldPtr) (holen : oldReg.data.length = X86Env.pages oldLen) (hrw : oldReg.rw = true)
    (hnlen : oldReg.data.length ≤ X86Env.pages newLen)
    (hor : (s.k.next).1 = some (.at a))
    (ha_pg : a % X86Env.page = 0) (ha_pos : 0 < a) (ha_lim : a + X86Env.pages newLen ≤ 2 ^ 47)
    (hfree : X86Env.Mem.free (s.mem.eraseIdx idx) a (X86Env.pages newLen) = true) :
    let k1 := (s.k.next).2
    let m1 : Mem := (s.mem.eraseIdx idx) ++ [(⟨a, oldReg.data ++ List.replicate (X86Env.pages newLen - oldReg.data.length) 0, true⟩ : Region)]
    ∃ s', Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.k = k1 ∧ s'.mem = m1 ∧ m1.wf ∧
      RegsOk s' ∧ s'.reg 0 = some a ∧ s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 4 → r ≠ 6 → r ≠ 7 → r ≠ 8 → r ≠ 9 → r ≠ 10 → r ≠ 11 → s'.reg r = s.reg r) := by
  have h01 : X86Vm.step vmImage s = .ok { s.set 10 (some 1) with rip := 4201369 } := by
    rw [step_at_chunk 23 hrip rfl (by decide : (SporosVmBin.fn_mremap, (.ofInt (.movI 10 1)), 6) ∈ SporosVmBin.vmTable_23),
      exec_movI_lt (by decide), hrip]
    rfl
  let s1 := { s.set 10 (some 1) with rip := 4201369 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) 1
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 8 (some 0) with rip := 4201375 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201369, (.ofInt (.movI 8 0)), 6) ∈ SporosVmBin.vmTable_23),
      exec_movI_lt (by decide)]
  let s2 := { s1.set 8 (some 0) with rip := 4201375 }
  have hr2' : RegsOk s2 := regsOk_set hr1 (by decide) 0
  obtain ⟨v9, hv9⟩ : ∃ v9, s2.reg 9 = some v9 := by
    have hs := hr2'.2 9 (by decide) (by decide)
    cases h : s2.reg 9 with
    | none => simp [h] at hs
    | some v => exact ⟨v, rfl⟩
  have h03 : X86Vm.step vmImage s2 =
      .ok { s2.set 9 (some 0) with fl := (X86Int.aluF .xor v9 v9).2, rip := 4201378 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201375, (.ofInt (.alu .xor 9 9)), 3) ∈ SporosVmBin.vmTable_23),
      exec_alu_xor_self hv9]
  let s3 := { s2.set 9 (some 0) with fl := (X86Int.aluF .xor v9 v9).2, rip := 4201378 }
  have hr3 : RegsOk s3 := regsOk_set hr2' (by decide) 0
  have h04 : X86Vm.step vmImage s3 = .ok { s3.set 0 (some 25) with rip := 4201383 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201378, (.ofInt (.movI 0 25)), 5) ∈ SporosVmBin.vmTable_23),
      exec_movI_lt (by decide)]
  let s4 := { s3.set 0 (some 25) with rip := 4201383 }
  have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) 25
  have hs4_0 : s4.reg 0 = some 25 := by
    simp only [s4, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
  have hs4_7 : s4.reg 7 = some oldPtr := by
    simp only [s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 7), List.getElem?_set_ne (by decide : 9 ≠ 7),
      List.getElem?_set_ne (by decide : 8 ≠ 7), List.getElem?_set_ne (by decide : 10 ≠ 7)]
    exact hr7
  have hs4_6 : s4.reg 6 = some oldLen := by
    simp only [s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 6), List.getElem?_set_ne (by decide : 9 ≠ 6),
      List.getElem?_set_ne (by decide : 8 ≠ 6), List.getElem?_set_ne (by decide : 10 ≠ 6)]
    exact hr6
  have hs4_2 : s4.reg 2 = some newLen := by
    simp only [s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 2), List.getElem?_set_ne (by decide : 9 ≠ 2),
      List.getElem?_set_ne (by decide : 8 ≠ 2), List.getElem?_set_ne (by decide : 10 ≠ 2)]
    exact hr2
  have hs4_10 : s4.reg 10 = some 1 := by
    simp only [s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 10), List.getElem?_set_ne (by decide : 9 ≠ 10),
      List.getElem?_set_ne (by decide : 8 ≠ 10)]
    rw [getElem?_set_getD_self (by rw [hregs.1]; decide)]
  have hs4_8 : s4.reg 8 = some 0 := by
    simp only [s4, s3, s2, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 8), List.getElem?_set_ne (by decide : 9 ≠ 8)]
    rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  have hs4_9 : s4.reg 9 = some 0 := by
    simp only [s4, s3, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 9)]
    rw [getElem?_set_getD_self (by rw [hr2'.1]; decide)]
  have hplace : X86Env.place (s.mem.eraseIdx idx) (s.k.next).1 (X86Env.pages newLen) = some (some a) := by
    rw [hor]
    simp [X86Env.place, ha_pg, ha_pos, ha_lim, hfree]
  have hnotlt' : X86Env.pages oldLen ≤ X86Env.pages newLen := by rw [← holen]; exact hnlen
  let k1 := (s.k.next).2
  let m1 : Mem := (s.mem.eraseIdx idx) ++ [⟨a, oldReg.data ++ List.replicate (X86Env.pages newLen - oldReg.data.length) 0, true⟩]
  have hsys : X86Env.syscall s4.k s4.mem 25 oldPtr oldLen newLen 1 0 0 = some (.ret a k1 m1) := by
    change X86Env.syscall s.k s.mem 25 oldPtr oldLen newLen 1 0 0 = _
    dsimp only [k1, m1]
    simp [X86Env.syscall, X86Env.MREMAP_MAYMOVE, hfind, hbase, holen, hrw, hnotlt']
    cases hn : s.k.next with
    | mk c k' =>
      have hc : c = some (.at a) := by simpa [hn] using hor
      subst hc
      have hpl' : X86Env.place (s.mem.eraseIdx idx) (some (.at a)) (X86Env.pages newLen) = some (some a) := by
        simpa [hn] using hplace
      simp [hpl']
  have h05 : X86Vm.step vmImage s4 =
      .ok { ((s4.set 0 (some a)).set 1 (some 4201385)).set 11 none with
            k := k1, mem := m1, rip := 4201385 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201383, (.ofInt .syscall), 2) ∈ SporosVmBin.vmTable_23),
      exec_syscall_ret hs4_0 hs4_7 hs4_6 hs4_2 hs4_10 hs4_8 hs4_9 hsys]
  let s5 := { ((s4.set 0 (some a)).set 1 (some 4201385)).set 11 none with
                k := k1, mem := m1, rip := 4201385 }
  have hr5 : RegsOk s5 :=
    regsOk_set_none_11 (regsOk_set (regsOk_set hr4 (by decide) a) (by decide) 4201385)
  have hs5_0 : s5.reg 0 = some a := by
    simp only [s5, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 11 ≠ 0), List.getElem?_set_ne (by decide : 1 ≠ 0)]
    rw [getElem?_set_getD_self (by rw [hr4.1]; decide)]
  have h06 : X86Vm.step vmImage s5 =
      .ok { s5 with fl := (X86Int.aluF .and a a).2, rip := 4201388 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201385, (.ofInt (.test 0 0)), 3) ∈ SporosVmBin.vmTable_23),
      exec_test hs5_0 hs5_0]
  let s6 := { s5 with fl := (X86Int.aluF .and a a).2, rip := 4201388 }
  have hr6' : RegsOk s6 := hr5
  have hmsb : X86Int.msb a = false := msb_lt_2p63 (by omega)
  have hc8 : X86Int.cond s6.fl 8 = some false := by
    simpa [s6, hmsb] using cond_aluF_and_self_s a
  have h07 : X86Vm.step vmImage s6 = .ok { s6 with rip := 4201394 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201388, (.ofInt (.jcc 8 4201290)), 6) ∈ SporosVmBin.vmTable_23),
      exec_jcc hc8]
    rfl
  let s7 := { s6 with rip := 4201394 }
  have hr7' : RegsOk s7 := hr6'
  have hm1wf : m1.wf := mem_mremap_wf hmwf ha_pg hfree hnlen
  have hstk_m1 : stk ∈ m1 := by
    have ⟨_, _, hget⟩ := mem_find_idx_lt s.mem oldPtr 0 idx oldReg hfind
    exact List.mem_append_left _ (mem_eraseIdx_other hget hstk (by rw [hbase]; exact hstkne))
  have hret_m1 : rd64 m1 sp = some retAddr :=
    rd64_of_mem_transfer hmwf hm1wf hstk hstk_m1 hlo hhi hret
  have hs7_4 : s7.reg 4 = some sp := by
    simp only [s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 11 ≠ 4), List.getElem?_set_ne (by decide : 1 ≠ 4),
      List.getElem?_set_ne (by decide : 0 ≠ 4), List.getElem?_set_ne (by decide : 9 ≠ 4),
      List.getElem?_set_ne (by decide : 8 ≠ 4), List.getElem?_set_ne (by decide : 10 ≠ 4)]
    exact hsp
  have h08 : X86Vm.step vmImage s7 =
      .ok { s7.set 4 (some ((sp + 8) % X86Env.W)) with rip := retAddr } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201394, (.ofInt .ret), 1) ∈ SporosVmBin.vmTable_23),
      exec_ret hs7_4 hret_m1]
  let s8 := { s7.set 4 (some ((sp + 8) % X86Env.W)) with rip := retAddr }
  refine ⟨s8,
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
      (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step h08))))))),
    rfl, rfl, rfl, hm1wf, regsOk_set hr7' (by decide) _, ?_, ?_, ?_⟩
  · simp only [s8, s7, s6, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 4 ≠ 0)]
    exact hs5_0
  · simp only [s8, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr7'.1]; decide)]
  · intro r h0 h1 h2 h4 h6 h7 h8 h9 h10' h11'
    simp only [s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne h4.symm, List.getElem?_set_ne h11'.symm, List.getElem?_set_ne h1.symm,
      List.getElem?_set_ne h0.symm, List.getElem?_set_ne h9.symm, List.getElem?_set_ne h8.symm,
      List.getElem?_set_ne h10'.symm]

theorem fn_mremap_nomem_spec {s : X86Vm.St} {sp oldPtr oldLen newLen idx : Nat} {oldReg stk : Region}
    (hrip : s.rip = SporosVmBin.fn_mremap)
    (hregs : RegsOk s)
    (hsp : s.reg 4 = some sp)
    (h8 : 8 ≤ sp) (hspW : sp < X86Env.W)
    (hmwf : s.mem.wf)
    (hstk : stk ∈ s.mem) (hstkrw : stk.rw = true) (hlo : stk.base ≤ sp - 8) (hhi : sp ≤ stk.hi)
    (hr7 : s.reg 7 = some oldPtr) (hr6 : s.reg 6 = some oldLen) (hr2 : s.reg 2 = some newLen)
    (hfind : s.mem.find oldPtr 0 = some (idx, oldReg))
    (hbase : oldReg.base = oldPtr) (holen : oldReg.data.length = X86Env.pages oldLen) (hrw : oldReg.rw = true)
    (hnlen : oldReg.data.length ≤ X86Env.pages newLen)
    (hor : (s.k.next).1 = some .nomem) :
    Ends vmImage s (.exit 2) := by
  have h01 : X86Vm.step vmImage s = .ok { s.set 10 (some 1) with rip := 4201369 } := by
    rw [step_at_chunk 23 hrip rfl (by decide : (SporosVmBin.fn_mremap, (.ofInt (.movI 10 1)), 6) ∈ SporosVmBin.vmTable_23),
      exec_movI_lt (by decide), hrip]
    rfl
  let s1 := { s.set 10 (some 1) with rip := 4201369 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) 1
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 8 (some 0) with rip := 4201375 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201369, (.ofInt (.movI 8 0)), 6) ∈ SporosVmBin.vmTable_23),
      exec_movI_lt (by decide)]
  let s2 := { s1.set 8 (some 0) with rip := 4201375 }
  have hr2' : RegsOk s2 := regsOk_set hr1 (by decide) 0
  obtain ⟨v9, hv9⟩ : ∃ v9, s2.reg 9 = some v9 := by
    have hs := hr2'.2 9 (by decide) (by decide)
    cases h : s2.reg 9 with
    | none => simp [h] at hs
    | some v => exact ⟨v, rfl⟩
  have h03 : X86Vm.step vmImage s2 =
      .ok { s2.set 9 (some 0) with fl := (X86Int.aluF .xor v9 v9).2, rip := 4201378 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201375, (.ofInt (.alu .xor 9 9)), 3) ∈ SporosVmBin.vmTable_23),
      exec_alu_xor_self hv9]
  let s3 := { s2.set 9 (some 0) with fl := (X86Int.aluF .xor v9 v9).2, rip := 4201378 }
  have hr3 : RegsOk s3 := regsOk_set hr2' (by decide) 0
  have h04 : X86Vm.step vmImage s3 = .ok { s3.set 0 (some 25) with rip := 4201383 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201378, (.ofInt (.movI 0 25)), 5) ∈ SporosVmBin.vmTable_23),
      exec_movI_lt (by decide)]
  let s4 := { s3.set 0 (some 25) with rip := 4201383 }
  have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) 25
  have hs4_0 : s4.reg 0 = some 25 := by
    simp only [s4, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
  have hs4_7 : s4.reg 7 = some oldPtr := by
    simp only [s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 7), List.getElem?_set_ne (by decide : 9 ≠ 7),
      List.getElem?_set_ne (by decide : 8 ≠ 7), List.getElem?_set_ne (by decide : 10 ≠ 7)]
    exact hr7
  have hs4_6 : s4.reg 6 = some oldLen := by
    simp only [s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 6), List.getElem?_set_ne (by decide : 9 ≠ 6),
      List.getElem?_set_ne (by decide : 8 ≠ 6), List.getElem?_set_ne (by decide : 10 ≠ 6)]
    exact hr6
  have hs4_2 : s4.reg 2 = some newLen := by
    simp only [s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 2), List.getElem?_set_ne (by decide : 9 ≠ 2),
      List.getElem?_set_ne (by decide : 8 ≠ 2), List.getElem?_set_ne (by decide : 10 ≠ 2)]
    exact hr2
  have hs4_10 : s4.reg 10 = some 1 := by
    simp only [s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 10), List.getElem?_set_ne (by decide : 9 ≠ 10),
      List.getElem?_set_ne (by decide : 8 ≠ 10)]
    rw [getElem?_set_getD_self (by rw [hregs.1]; decide)]
  have hs4_8 : s4.reg 8 = some 0 := by
    simp only [s4, s3, s2, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 8), List.getElem?_set_ne (by decide : 9 ≠ 8)]
    rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  have hs4_9 : s4.reg 9 = some 0 := by
    simp only [s4, s3, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 9)]
    rw [getElem?_set_getD_self (by rw [hr2'.1]; decide)]
  have hnotlt' : X86Env.pages oldLen ≤ X86Env.pages newLen := by rw [← holen]; exact hnlen
  have hsys : X86Env.syscall s4.k s4.mem 25 oldPtr oldLen newLen 1 0 0 =
      some (.ret (X86Env.neg X86Env.ENOMEM) (s.k.next).2 s.mem) := by
    change X86Env.syscall s.k s.mem 25 oldPtr oldLen newLen 1 0 0 = _
    simp [X86Env.syscall, X86Env.MREMAP_MAYMOVE, hfind, hbase, holen, hrw, hnotlt']
    cases hn : s.k.next with
    | mk c k' =>
      have hc : c = some .nomem := by simpa [hn] using hor
      subst hc
      simp [X86Env.place]
  have h05 : X86Vm.step vmImage s4 =
      .ok { ((s4.set 0 (some (X86Env.neg X86Env.ENOMEM))).set 1 (some 4201385)).set 11 none with
            k := (s.k.next).2, mem := s.mem, rip := 4201385 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201383, (.ofInt .syscall), 2) ∈ SporosVmBin.vmTable_23),
      exec_syscall_ret hs4_0 hs4_7 hs4_6 hs4_2 hs4_10 hs4_8 hs4_9 hsys]
  let s5 := { ((s4.set 0 (some (X86Env.neg X86Env.ENOMEM))).set 1 (some 4201385)).set 11 none with
                k := (s.k.next).2, mem := s.mem, rip := 4201385 }
  have hr5 : RegsOk s5 :=
    regsOk_set_none_11 (regsOk_set (regsOk_set hr4 (by decide) _) (by decide) 4201385)
  have hs5_0 : s5.reg 0 = some (X86Env.neg X86Env.ENOMEM) := by
    simp only [s5, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 11 ≠ 0), List.getElem?_set_ne (by decide : 1 ≠ 0)]
    rw [getElem?_set_getD_self (by rw [hr4.1]; decide)]
  have h06 : X86Vm.step vmImage s5 =
      .ok { s5 with fl := (X86Int.aluF .and (X86Env.neg X86Env.ENOMEM) (X86Env.neg X86Env.ENOMEM)).2,
                    rip := 4201388 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201385, (.ofInt (.test 0 0)), 3) ∈ SporosVmBin.vmTable_23),
      exec_test hs5_0 hs5_0]
  let s6 := { s5 with fl := (X86Int.aluF .and (X86Env.neg X86Env.ENOMEM) (X86Env.neg X86Env.ENOMEM)).2,
                      rip := 4201388 }
  have hr6' : RegsOk s6 := hr5
  have hc8 : X86Int.cond s6.fl 8 = some true := by
    simpa [s6, msb_neg_enomem] using cond_aluF_and_self_s (X86Env.neg X86Env.ENOMEM)
  have h07 : X86Vm.step vmImage s6 = .ok { s6 with rip := 4201290 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201388, (.ofInt (.jcc 8 4201290)), 6) ∈ SporosVmBin.vmTable_23),
      exec_jcc hc8]
    rfl
  let s7 := { s6 with rip := 4201290 }
  have hr7' : RegsOk s7 := hr6'
  have hs7_4 : s7.reg 4 = some sp := by
    simp only [s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 11 ≠ 4), List.getElem?_set_ne (by decide : 1 ≠ 4),
      List.getElem?_set_ne (by decide : 0 ≠ 4), List.getElem?_set_ne (by decide : 9 ≠ 4),
      List.getElem?_set_ne (by decide : 8 ≠ 4), List.getElem?_set_ne (by decide : 10 ≠ 4)]
    exact hsp
  have h_exit := L_exit_nomem_spec (s := s7) rfl hr7' hs7_4 h8 hspW hmwf hstk hstkrw hlo hhi
  exact Ends.step_trans h01 (Ends.step_trans h02 (Ends.step_trans h03 (Ends.step_trans h04
    (Ends.step_trans h05 (Ends.step_trans h06 (Ends.step_trans h07 h_exit))))))

theorem fn_ds_push_join_spec {s : X86Vm.St} {sp retAddr ctxBase dsBase dsLen w p : Nat}
    {ctx dsReg stk : Region}
    (hrip : s.rip = 4201502)
    (hregs : RegsOk s)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr12 : s.reg 12 = some w) (hr13 : s.reg 13 = some p)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_rw : ctx.rw = true)
    (hctx_len : ctxSize ≤ ctx.data.length)
    (hds : dsReg ∈ s.mem) (hds_base : dsReg.base = dsBase) (hds_rw : dsReg.rw = true)
    (hne : dsBase ≠ ctxBase)
    (hstk : stk ∈ s.mem) (hstkne_ds : stk.base ≠ dsBase) (hstkne_ctx : stk.base ≠ ctxBase)
    (hstk_lo : stk.base ≤ sp) (hstk_hi : sp + 8 ≤ stk.hi)
    (hds_hi : dsLen * 16 + 16 ≤ dsReg.data.length)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W) (hdsW : dsBase + dsLen * 16 + 16 ≤ X86Env.W)
    (hdsLenW : dsLen * 16 < X86Env.W)
    (hmwf : s.mem.wf)
    (hret : rd64 s.mem sp = some retAddr)
    (hc0 : leVal ((ctx.data.drop 0).take 8) = dsBase)
    (hc8 : leVal ((ctx.data.drop 8).take 8) = dsLen) :
    let dsData' := writeBytes (writeBytes dsReg.data (dsLen * 16) (bytes8 w)) (dsLen * 16 + 8) (bytes8 p)
    let ctxData' := writeBytes ctx.data 8 (bytes8 (dsLen + 1))
    ∃ m' s', Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.k = s.k ∧ s'.mem = m' ∧ m'.wf ∧ RegsOk s' ∧
      s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      { dsReg with data := dsData' } ∈ m' ∧
      { ctx with data := ctxData' } ∈ m' ∧
      (∀ u ∈ s.mem, u.base ≠ dsBase → u.base ≠ ctxBase → u ∈ m') ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 4 → s'.reg r = s.reg r) := by
  have hctxBaseW : ctxBase < X86Env.W := by have := ctxSize_pos; omega
  have hea_c0 : ea ctxBase 0 = ctxBase := by simp [X86Int.ea, Nat.mod_eq_of_lt hctxBaseW]
  have hea_c8 : ea ctxBase 8 = ctxBase + 8 := by
    have : ctxBase + 8 < X86Env.W := by have := ctxSize_ge_24; omega
    simp [X86Int.ea, Nat.mod_eq_of_lt this]
  have hrd_c0 : rd64 s.mem (ea ctxBase 0) = some dsBase := by
    rw [hea_c0]
    exact rd64_of_mem hmwf hctx (by simp [hctx_base])
      (by simp [Region.hi, hctx_base]; have := ctxSize_ge_8; omega)
      (by simpa [hctx_base] using hc0)
  have h01 : X86Vm.step vmImage s = .ok { s.set 0 (some dsBase) with rip := 4201505 } := by
    rw [step_at_chunk 24 hrip rfl (by decide : (4201502, (.ofInt (.ld 0 15 0)), 3) ∈ SporosVmBin.vmTable_24),
      exec_ld hr15 hrd_c0, hrip]
  let s1 := { s.set 0 (some dsBase) with rip := 4201505 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) dsBase
  have hs1_15 : s1.reg 15 = some ctxBase := by
    simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 15)]
    exact hr15
  have hrd_c8 : rd64 s1.mem (ea ctxBase 8) = some dsLen := by
    rw [hea_c8]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; have := ctxSize_ge_16; omega)
      (by simpa [hctx_base] using hc8)
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 1 (some dsLen) with rip := 4201509 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201505, (.ofInt (.ld 1 15 8)), 4) ∈ SporosVmBin.vmTable_24),
      exec_ld hs1_15 hrd_c8]
  let s2 := { s1.set 1 (some dsLen) with rip := 4201509 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) dsLen
  have hs2_1 : s2.reg 1 = some dsLen := by
    simp only [s2, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 2 (some dsLen) with rip := 4201512 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201509, (.ofInt (.mov 2 1)), 3) ∈ SporosVmBin.vmTable_24),
      exec_mov hs2_1]
  let s3 := { s2.set 2 (some dsLen) with rip := 4201512 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) dsLen
  have hs3_2 : s3.reg 2 = some dsLen := by
    simp only [s3, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
  have h04 : X86Vm.step vmImage s3 =
      .ok { s3.set 2 (some (dsLen * 16)) with fl := X86Int.Flags.undef, rip := 4201516 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201512, (.ofInt (.shlI 2 4)), 4) ∈ SporosVmBin.vmTable_24),
      exec_shlI_4_lt hs3_2 hdsLenW]
  let s4 := { s3.set 2 (some (dsLen * 16)) with fl := X86Int.Flags.undef, rip := 4201516 }
  have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) _
  have hs4_0 : s4.reg 0 = some dsBase := by
    simp only [s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 2 ≠ 0), List.getElem?_set_ne (by decide : 1 ≠ 0)]
    rw [getElem?_set_getD_self (by rw [hregs.1]; decide)]
  have hs4_2 : s4.reg 2 = some (dsLen * 16) := by
    simp only [s4, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
  have hs4_12 : s4.reg 12 = some w := by
    simp only [s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 2 ≠ 12), List.getElem?_set_ne (by decide : 1 ≠ 12),
      List.getElem?_set_ne (by decide : 0 ≠ 12)]
    exact hr12
  have hea_d0 : eaIdx dsBase (dsLen * 16) 1 0 = dsBase + dsLen * 16 := by
    simp [eaIdx, Nat.mod_eq_of_lt (by omega : dsBase + dsLen * 16 < X86Env.W)]
  obtain ⟨i1, m5, hwr5, _, _, hwf5, hr5_ds, hoth5⟩ :=
    wrMem_w64_region hmwf hds hds_rw (a := dsBase + dsLen * 16) (v := w)
      (by rw [hds_base]; omega) (by simp [Region.hi, hds_base]; omega)
  have hwr5' : wrMem .w64 s4.mem (eaIdx dsBase (dsLen * 16) 1 0) w = some m5 := by
    simpa [s4, s3, s2, s1, hea_d0] using hwr5
  have h05 : X86Vm.step vmImage s4 = .ok { s4 with mem := m5, rip := 4201520 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201516, (.stIdx .w64 0 2 1 0 12), 4) ∈ SporosVmBin.vmTable_24),
      exec_stIdx hs4_0 hs4_2 hs4_12 hwr5']
  let s5 := { s4 with mem := m5, rip := 4201520 }
  have hr5 : RegsOk s5 := hr4
  have hs5_0 : s5.reg 0 = some dsBase := hs4_0
  have hs5_2 : s5.reg 2 = some (dsLen * 16) := hs4_2
  have hs5_13 : s5.reg 13 = some p := by
    simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 2 ≠ 13), List.getElem?_set_ne (by decide : 1 ≠ 13),
      List.getElem?_set_ne (by decide : 0 ≠ 13)]
    exact hr13
  have hea_d8 : eaIdx dsBase (dsLen * 16) 1 8 = dsBase + dsLen * 16 + 8 := by
    simp [eaIdx, Nat.mod_eq_of_lt (by omega : dsBase + dsLen * 16 + 8 < X86Env.W)]
  let ds1 := { dsReg with data := writeBytes dsReg.data (dsBase + dsLen * 16 - dsReg.base) (bytes8 w) }
  have hds1_hi : dsBase + dsLen * 16 + 16 ≤ ds1.hi := by
    simp [ds1, Region.hi, writeBytes_length, hds_base]; omega
  obtain ⟨i2, m6, hwr6, _, _, hwf6, hr6_ds, hoth6⟩ :=
    wrMem_w64_region hwf5 hr5_ds hds_rw (a := dsBase + dsLen * 16 + 8) (v := p)
      (by simp [hds_base]; omega) hds1_hi
  have hwr6' : wrMem .w64 s5.mem (eaIdx dsBase (dsLen * 16) 1 8) p = some m6 := by
    simpa [s5, hea_d8] using hwr6
  have h06 : X86Vm.step vmImage s5 = .ok { s5 with mem := m6, rip := 4201525 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201520, (.stIdx .w64 0 2 1 8 13), 5) ∈ SporosVmBin.vmTable_24),
      exec_stIdx hs5_0 hs5_2 hs5_13 hwr6']
  let s6 := { s5 with mem := m6, rip := 4201525 }
  have hr6 : RegsOk s6 := hr5
  have hs6_1 : s6.reg 1 = some dsLen := by
    simp only [s6, s5, s4, s3, s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 1)]
    rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  have h07 : X86Vm.step vmImage s6 =
      .ok { s6.set 1 (some ((dsLen + 1) % X86Env.W)) with
            fl := (X86Int.aluF .add dsLen 1).2, rip := 4201529 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201525, (.ofInt (.aluI .add 1 1)), 4) ∈ SporosVmBin.vmTable_24),
      exec_aluI_add_lt hs6_1 (by decide)]
  let s7 := { s6.set 1 (some ((dsLen + 1) % X86Env.W)) with
                fl := (X86Int.aluF .add dsLen 1).2, rip := 4201529 }
  have hr7' : RegsOk s7 := regsOk_set hr6 (by decide) _
  have hs7_15 : s7.reg 15 = some ctxBase := by
    simp only [s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 15), List.getElem?_set_ne (by decide : 2 ≠ 15),
      List.getElem?_set_ne (by decide : 0 ≠ 15)]
    exact hr15
  have hs7_1 : s7.reg 1 = some ((dsLen + 1) % X86Env.W) := by
    simp only [s7, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr6.1]; decide)]
  have hctx_m5 : ctx ∈ m5 := hoth5 ctx hctx (by rw [hctx_base, hds_base]; exact Ne.symm hne)
  have hctx_m6 : ctx ∈ m6 := hoth6 ctx hctx_m5 (by simp [hctx_base, hds_base]; exact Ne.symm hne)
  obtain ⟨i3, m7, hwr7, _, _, hwf7, hr7_ctx, hoth7⟩ :=
    wr64_region hwf6 hctx_m6 hctx_rw (a := ctxBase + 8) (v := (dsLen + 1) % X86Env.W)
      (by rw [hctx_base]; omega) (by simp [Region.hi, hctx_base]; have := ctxSize_ge_16; omega)
  have hwr7' : wr64 s7.mem (ea ctxBase 8) ((dsLen + 1) % X86Env.W) = some m7 := by
    simpa [s7, s6, hea_c8] using hwr7
  have h08 : X86Vm.step vmImage s7 = .ok { s7 with mem := m7, rip := 4201533 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201529, (.ofInt (.st 15 8 1)), 4) ∈ SporosVmBin.vmTable_24),
      exec_st hs7_15 hs7_1 hwr7']
  let s8 := { s7 with mem := m7, rip := 4201533 }
  have hr8 : RegsOk s8 := hr7'
  have hs8_4 : s8.reg 4 = some sp := by
    simp only [s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 4), List.getElem?_set_ne (by decide : 2 ≠ 4),
      List.getElem?_set_ne (by decide : 0 ≠ 4)]
    exact hsp
  have hstk_m5 : stk ∈ m5 := hoth5 stk hstk (by rw [hds_base]; exact hstkne_ds)
  have hstk_m6 : stk ∈ m6 := hoth6 stk hstk_m5 (by simp [hds_base]; exact hstkne_ds)
  have hstk_m7 : stk ∈ m7 := hoth7 stk hstk_m6 (by rw [hctx_base]; exact hstkne_ctx)
  have hret_m7 : rd64 s8.mem sp = some retAddr :=
    rd64_of_mem_transfer hmwf hwf7 hstk hstk_m7 hstk_lo hstk_hi hret
  have h09 : X86Vm.step vmImage s8 =
      .ok { s8.set 4 (some ((sp + 8) % X86Env.W)) with rip := retAddr } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201533, (.ofInt .ret), 1) ∈ SporosVmBin.vmTable_24),
      exec_ret hs8_4 hret_m7]
  let s9 := { s8.set 4 (some ((sp + 8) % X86Env.W)) with rip := retAddr }
  refine ⟨m7, s9,
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
      (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step_trans h08
        (Reaches.step h09)))))))),
    rfl, rfl, rfl, hwf7, regsOk_set hr8 (by decide) _, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [s9, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr8.1]; decide)]
  · have hds2_m7 :
        { ds1 with data := writeBytes ds1.data (dsBase + dsLen * 16 + 8 - ds1.base) (bytes8 p) } ∈ m7 :=
      hoth7 _ hr6_ds (by simp [ds1, hds_base, hctx_base]; exact hne)
    simpa [ds1, hds_base, show dsBase + dsLen * 16 - dsBase = dsLen * 16 from by omega,
      show dsBase + dsLen * 16 + 8 - dsBase = dsLen * 16 + 8 from by omega] using hds2_m7
  · simpa [hctx_base, show ctxBase + 8 - ctxBase = 8 from by omega,
      Nat.mod_eq_of_lt (by omega : dsLen + 1 < X86Env.W)] using hr7_ctx
  · intro u hu huds huctx
    have hu5 := hoth5 u hu (by rw [hds_base]; exact huds)
    have hu6 := hoth6 u hu5 (by simp [hds_base]; exact huds)
    exact hoth7 u hu6 (by rw [hctx_base]; exact huctx)
  · intro r h0 h1 h2 h4
    simp only [s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne h4.symm, List.getElem?_set_ne h1.symm,
      List.getElem?_set_ne h2.symm, List.getElem?_set_ne h0.symm]

theorem fn_ds_push_nogrow_spec {s : X86Vm.St} {sp retAddr ctxBase dsBase dsLen dsCap w p : Nat}
    {ctx dsReg stk : Region}
    (hrip : s.rip = SporosVmBin.fn_ds_push)
    (hregs : RegsOk s)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr7 : s.reg 7 = some w) (hr6 : s.reg 6 = some p)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_rw : ctx.rw = true)
    (hctx_len : ctxSize ≤ ctx.data.length)
    (hds : dsReg ∈ s.mem) (hds_base : dsReg.base = dsBase) (hds_rw : dsReg.rw = true)
    (hne : dsBase ≠ ctxBase)
    (hstk : stk ∈ s.mem) (hstkne_ds : stk.base ≠ dsBase) (hstkne_ctx : stk.base ≠ ctxBase)
    (hstk_lo : stk.base ≤ sp) (hstk_hi : sp + 8 ≤ stk.hi)
    (hlt : dsLen < dsCap)
    (hds_hi : dsCap * 16 ≤ dsReg.data.length)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W) (hdsW : dsBase + dsCap * 16 ≤ X86Env.W)
    (hdsCapW : dsCap * 16 < X86Env.W)
    (hmwf : s.mem.wf)
    (hret : rd64 s.mem sp = some retAddr)
    (hc0 : leVal ((ctx.data.drop 0).take 8) = dsBase)
    (hc8 : leVal ((ctx.data.drop 8).take 8) = dsLen)
    (hc16 : leVal ((ctx.data.drop 16).take 8) = dsCap) :
    let dsData' := writeBytes (writeBytes dsReg.data (dsLen * 16) (bytes8 w)) (dsLen * 16 + 8) (bytes8 p)
    let ctxData' := writeBytes ctx.data 8 (bytes8 (dsLen + 1))
    ∃ m' s', Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.k = s.k ∧ s'.mem = m' ∧ m'.wf ∧ RegsOk s' ∧
      s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      { dsReg with data := dsData' } ∈ m' ∧
      { ctx with data := ctxData' } ∈ m' ∧
      (∀ u ∈ s.mem, u.base ≠ dsBase → u.base ≠ ctxBase → u ∈ m') ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 4 → r ≠ 12 → r ≠ 13 → s'.reg r = s.reg r) := by
  have hctxBaseW : ctxBase < X86Env.W := by have := ctxSize_pos; omega
  have hea_c8 : ea ctxBase 8 = ctxBase + 8 := by
    have : ctxBase + 8 < X86Env.W := by have := ctxSize_ge_24; omega
    simp [X86Int.ea, Nat.mod_eq_of_lt this]
  have hea_c16 : ea ctxBase 16 = ctxBase + 16 := by
    have : ctxBase + 16 < X86Env.W := by have := ctxSize_ge_24; omega
    simp [X86Int.ea, Nat.mod_eq_of_lt this]
  have h01 : X86Vm.step vmImage s = .ok { s.set 12 (some w) with rip := 4201449 } := by
    rw [step_at_chunk 23 hrip rfl (by decide : (SporosVmBin.fn_ds_push, (.ofInt (.mov 12 7)), 3) ∈ SporosVmBin.vmTable_23),
      exec_mov hr7, hrip]
    rfl
  let s1 := { s.set 12 (some w) with rip := 4201449 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) w
  have hs1_6 : s1.reg 6 = some p := by
    simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 12 ≠ 6)]
    exact hr6
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 13 (some p) with rip := 4201452 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201449, (.ofInt (.mov 13 6)), 3) ∈ SporosVmBin.vmTable_23),
      exec_mov hs1_6]
  let s2 := { s1.set 13 (some p) with rip := 4201452 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) p
  have hs2_15 : s2.reg 15 = some ctxBase := by
    simp only [s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 13 ≠ 15), List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hrd_c8 : rd64 s2.mem (ea ctxBase 8) = some dsLen := by
    rw [hea_c8]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; have := ctxSize_ge_16; omega)
      (by simpa [hctx_base] using hc8)
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 0 (some dsLen) with rip := 4201456 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201452, (.ofInt (.ld 0 15 8)), 4) ∈ SporosVmBin.vmTable_23),
      exec_ld hs2_15 hrd_c8]
  let s3 := { s2.set 0 (some dsLen) with rip := 4201456 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) dsLen
  have hs3_15 : s3.reg 15 = some ctxBase := by
    simp only [s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 15), List.getElem?_set_ne (by decide : 13 ≠ 15),
      List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hrd_c16 : rd64 s3.mem (ea ctxBase 16) = some dsCap := by
    rw [hea_c16]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; have := ctxSize_ge_24; omega)
      (by simpa [hctx_base] using hc16)
  have h04 : X86Vm.step vmImage s3 = .ok { s3.set 1 (some dsCap) with rip := 4201460 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201456, (.ofInt (.ld 1 15 16)), 4) ∈ SporosVmBin.vmTable_23),
      exec_ld hs3_15 hrd_c16]
  let s4 := { s3.set 1 (some dsCap) with rip := 4201460 }
  have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) dsCap
  have hs4_0 : s4.reg 0 = some dsLen := by
    simp only [s4, s3, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0)]
    rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
  have hs4_1 : s4.reg 1 = some dsCap := by
    simp only [s4, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
  have h05 : X86Vm.step vmImage s4 =
      .ok { s4 with fl := (X86Int.aluF .sub dsLen dsCap).2, rip := 4201463 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201460, (.ofInt (.cmp 0 1)), 3) ∈ SporosVmBin.vmTable_23),
      exec_cmp hs4_0 hs4_1]
  let s5 := { s4 with fl := (X86Int.aluF .sub dsLen dsCap).2, rip := 4201463 }
  have hr5 : RegsOk s5 := hr4
  have hcb : X86Int.cond s5.fl 2 = some true := by
    dsimp [s5]; rw [cond_aluF_sub_b, decide_eq_true hlt]
  have h06 : X86Vm.step vmImage s5 = .ok { s5 with rip := 4201502 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201463, (.ofInt (.jcc 2 4201502)), 2) ∈ SporosVmBin.vmTable_23),
      exec_jcc hcb]
    rfl
  let s6 := { s5 with rip := 4201502 }
  have hr6' : RegsOk s6 := hr5
  have hs6_4 : s6.reg 4 = some sp := by
    simp only [s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 4), List.getElem?_set_ne (by decide : 0 ≠ 4),
      List.getElem?_set_ne (by decide : 13 ≠ 4), List.getElem?_set_ne (by decide : 12 ≠ 4)]
    exact hsp
  have hs6_15 : s6.reg 15 = some ctxBase := by
    simp only [s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 15), List.getElem?_set_ne (by decide : 0 ≠ 15),
      List.getElem?_set_ne (by decide : 13 ≠ 15), List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hs6_12 : s6.reg 12 = some w := by
    simp only [s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 12), List.getElem?_set_ne (by decide : 0 ≠ 12),
      List.getElem?_set_ne (by decide : 13 ≠ 12)]
    rw [getElem?_set_getD_self (by rw [hregs.1]; decide)]
  have hs6_13 : s6.reg 13 = some p := by
    simp only [s6, s5, s4, s3, s2, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 13), List.getElem?_set_ne (by decide : 0 ≠ 13)]
    rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  obtain ⟨m', sf, hreach, hripf, hkf, hmemf, hwf', hregsf, hspf, hdsf, hctxf, hothf, hregoth⟩ :=
    fn_ds_push_join_spec (s := s6) (sp := sp) (retAddr := retAddr) (ctxBase := ctxBase)
      (dsBase := dsBase) (dsLen := dsLen) (w := w) (p := p) (ctx := ctx) (dsReg := dsReg) (stk := stk)
      rfl hr6' hs6_4 hs6_15 hs6_12 hs6_13 hctx hctx_base hctx_rw hctx_len hds hds_base hds_rw hne
      hstk hstkne_ds hstkne_ctx hstk_lo hstk_hi (by omega) hctxW (by omega) (by omega) hmwf
      hret hc0 hc8
  refine ⟨m', sf,
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
      (Reaches.step_trans h05 (Reaches.step_trans h06 hreach))))),
    hripf, hkf, hmemf, hwf', hregsf, hspf, hdsf, hctxf, hothf, ?_⟩
  intro r h0 h1 h2 h4 h12 h13
  rw [hregoth r h0 h1 h2 h4]
  simp only [s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
    List.getElem?_set_ne h1.symm, List.getElem?_set_ne h0.symm,
    List.getElem?_set_ne h13.symm, List.getElem?_set_ne h12.symm]

theorem pages_of_dvd {n : Nat} (hdvd : n % X86Env.page = 0) : X86Env.pages n = n := by
  have hp : X86Env.page = 4096 := rfl
  rw [X86Env.pages, hp] at *
  have hmod : n % 4096 = 0 := hdvd
  have hdiv : (n + 4096 - 1) / 4096 = n / 4096 := by omega
  rw [hdiv]
  omega

theorem fn_ds_push_grow_spec {s : X86Vm.St} {sp retAddr ctxBase dsBase dsLen dsCap w p a : Nat}
    {ctx dsReg stk : Region}
    (hrip : s.rip = SporosVmBin.fn_ds_push)
    (hregs : RegsOk s)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr7 : s.reg 7 = some w) (hr6 : s.reg 6 = some p)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_rw : ctx.rw = true)
    (hctx_len : ctxSize ≤ ctx.data.length)
    (hds : dsReg ∈ s.mem) (hds_base : dsReg.base = dsBase) (hds_rw : dsReg.rw = true)
    (hne : dsBase ≠ ctxBase) (hane : a ≠ ctxBase) (hastk : a ≠ stk.base)
    (hstk : stk ∈ s.mem) (hstkrw : stk.rw = true) (hstkne_ds : stk.base ≠ dsBase) (hstkne_ctx : stk.base ≠ ctxBase)
    (h8 : 8 ≤ sp) (hspW : sp < X86Env.W)
    (hstk_lo : stk.base ≤ sp - 8) (hstk_hi : sp + 8 ≤ stk.hi)
    (hge : dsCap ≤ dsLen) (hle : dsLen ≤ dsCap) (hcap_pos : 0 < dsCap)
    (hds_eq : dsReg.data.length = dsCap * 16)
    (hds_pg : (dsCap * 16) % X86Env.page = 0)
    (hdsCapW : dsCap * 32 < X86Env.W)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W)
    (hmwf : s.mem.wf) (hpos : ∀ u ∈ s.mem, 0 < u.data.length)
    (hret : rd64 s.mem sp = some retAddr)
    (hc0 : leVal ((ctx.data.drop 0).take 8) = dsBase)
    (hc8 : leVal ((ctx.data.drop 8).take 8) = dsLen)
    (hc16 : leVal ((ctx.data.drop 16).take 8) = dsCap)
    (hor : (s.k.next).1 = some (.at a))
    (ha_pg : a % X86Env.page = 0) (ha_pos : 0 < a) (ha_lim : a + dsCap * 32 ≤ 2 ^ 47)
    (hfree : ∀ u ∈ s.mem, u.base ≠ dsBase → a + dsCap * 32 ≤ u.base ∨ u.hi ≤ a) :
    let dsData' := writeBytes (writeBytes (dsReg.data ++ List.replicate (dsCap * 16) 0) (dsLen * 16) (bytes8 w))
      (dsLen * 16 + 8) (bytes8 p)
    let ctxData' := writeBytes (writeBytes (writeBytes ctx.data 0 (bytes8 a)) 16 (bytes8 (dsCap * 2))) 8 (bytes8 (dsLen + 1))
    ∃ m' s' stk', Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.k = (s.k.next).2 ∧ s'.mem = m' ∧ m'.wf ∧ RegsOk s' ∧
      s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      (⟨a, dsData', true⟩ : Region) ∈ m' ∧
      { ctx with data := ctxData' } ∈ m' ∧
      stk' ∈ m' ∧ stk'.base = stk.base ∧ stk'.rw = true ∧ stk'.data.length = stk.data.length ∧
      (∀ j, j < sp - 8 - stk.base ∨ sp - stk.base ≤ j → stk'.data[j]? = stk.data[j]?) ∧
      (∀ u ∈ s.mem, u.base ≠ dsBase → u.base ≠ ctxBase → u.base ≠ stk.base → u ∈ m') ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 4 → r ≠ 6 → r ≠ 7 → r ≠ 8 → r ≠ 9 → r ≠ 10 → r ≠ 11 → r ≠ 12 → r ≠ 13 → s'.reg r = s.reg r) := by
  have hctxBaseW : ctxBase < X86Env.W := by have := ctxSize_pos; omega
  have hea_c0 : ea ctxBase 0 = ctxBase := by simp [X86Int.ea, Nat.mod_eq_of_lt hctxBaseW]
  have hea_c8 : ea ctxBase 8 = ctxBase + 8 := by
    have : ctxBase + 8 < X86Env.W := by have := ctxSize_ge_24; omega
    simp [X86Int.ea, Nat.mod_eq_of_lt this]
  have hea_c16 : ea ctxBase 16 = ctxBase + 16 := by
    have : ctxBase + 16 < X86Env.W := by have := ctxSize_ge_24; omega
    simp [X86Int.ea, Nat.mod_eq_of_lt this]
  have h01 : X86Vm.step vmImage s = .ok { s.set 12 (some w) with rip := 4201449 } := by
    rw [step_at_chunk 23 hrip rfl (by decide : (SporosVmBin.fn_ds_push, (.ofInt (.mov 12 7)), 3) ∈ SporosVmBin.vmTable_23),
      exec_mov hr7, hrip]
    rfl
  let s1 := { s.set 12 (some w) with rip := 4201449 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) w
  have hs1_6 : s1.reg 6 = some p := by
    simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 12 ≠ 6)]
    exact hr6
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 13 (some p) with rip := 4201452 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201449, (.ofInt (.mov 13 6)), 3) ∈ SporosVmBin.vmTable_23),
      exec_mov hs1_6]
  let s2 := { s1.set 13 (some p) with rip := 4201452 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) p
  have hs2_15 : s2.reg 15 = some ctxBase := by
    simp only [s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 13 ≠ 15), List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hrd_c8 : rd64 s2.mem (ea ctxBase 8) = some dsLen := by
    rw [hea_c8]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; have := ctxSize_ge_16; omega)
      (by simpa [hctx_base] using hc8)
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 0 (some dsLen) with rip := 4201456 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201452, (.ofInt (.ld 0 15 8)), 4) ∈ SporosVmBin.vmTable_23),
      exec_ld hs2_15 hrd_c8]
  let s3 := { s2.set 0 (some dsLen) with rip := 4201456 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) dsLen
  have hs3_15 : s3.reg 15 = some ctxBase := by
    simp only [s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 15), List.getElem?_set_ne (by decide : 13 ≠ 15),
      List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hrd_c16 : rd64 s3.mem (ea ctxBase 16) = some dsCap := by
    rw [hea_c16]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; have := ctxSize_ge_24; omega)
      (by simpa [hctx_base] using hc16)
  have h04 : X86Vm.step vmImage s3 = .ok { s3.set 1 (some dsCap) with rip := 4201460 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201456, (.ofInt (.ld 1 15 16)), 4) ∈ SporosVmBin.vmTable_23),
      exec_ld hs3_15 hrd_c16]
  let s4 := { s3.set 1 (some dsCap) with rip := 4201460 }
  have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) dsCap
  have hs4_0 : s4.reg 0 = some dsLen := by
    simp only [s4, s3, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0)]
    rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
  have hs4_1 : s4.reg 1 = some dsCap := by
    simp only [s4, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
  have h05 : X86Vm.step vmImage s4 =
      .ok { s4 with fl := (X86Int.aluF .sub dsLen dsCap).2, rip := 4201463 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201460, (.ofInt (.cmp 0 1)), 3) ∈ SporosVmBin.vmTable_23),
      exec_cmp hs4_0 hs4_1]
  let s5 := { s4 with fl := (X86Int.aluF .sub dsLen dsCap).2, rip := 4201463 }
  have hr5 : RegsOk s5 := hr4
  have hnotlt : ¬ dsLen < dsCap := by omega
  have hcb : X86Int.cond s5.fl 2 = some false := by
    dsimp [s5]; rw [cond_aluF_sub_b, decide_eq_false hnotlt]
  have h06 : X86Vm.step vmImage s5 = .ok { s5 with rip := 4201465 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201463, (.ofInt (.jcc 2 4201502)), 2) ∈ SporosVmBin.vmTable_23),
      exec_jcc hcb]
    rfl
  let s6 := { s5 with rip := 4201465 }
  have hr6' : RegsOk s6 := hr5
  have hs6_15 : s6.reg 15 = some ctxBase := by
    simp only [s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 15), List.getElem?_set_ne (by decide : 0 ≠ 15),
      List.getElem?_set_ne (by decide : 13 ≠ 15), List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hrd_c0 : rd64 s6.mem (ea ctxBase 0) = some dsBase := by
    rw [hea_c0]
    exact rd64_of_mem hmwf hctx (by simp [hctx_base])
      (by simp [Region.hi, hctx_base]; have := ctxSize_ge_8; omega)
      (by simpa [hctx_base] using hc0)
  have h07 : X86Vm.step vmImage s6 = .ok { s6.set 7 (some dsBase) with rip := 4201468 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201465, (.ofInt (.ld 7 15 0)), 3) ∈ SporosVmBin.vmTable_23),
      exec_ld hs6_15 hrd_c0]
  let s7 := { s6.set 7 (some dsBase) with rip := 4201468 }
  have hr7' : RegsOk s7 := regsOk_set hr6' (by decide) dsBase
  have hs7_1 : s7.reg 1 = some dsCap := by
    simp only [s7, s6, s5, s4, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 7 ≠ 1)]
    rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
  have h08 : X86Vm.step vmImage s7 = .ok { s7.set 6 (some dsCap) with rip := 4201471 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201468, (.ofInt (.mov 6 1)), 3) ∈ SporosVmBin.vmTable_23),
      exec_mov hs7_1]
  let s8 := { s7.set 6 (some dsCap) with rip := 4201471 }
  have hr8 : RegsOk s8 := regsOk_set hr7' (by decide) dsCap
  have hs8_6 : s8.reg 6 = some dsCap := by
    simp only [s8, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr7'.1]; decide)]
  have h09 : X86Vm.step vmImage s8 =
      .ok { s8.set 6 (some (dsCap * 16)) with fl := X86Int.Flags.undef, rip := 4201475 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201471, (.ofInt (.shlI 6 4)), 4) ∈ SporosVmBin.vmTable_23),
      exec_shlI_4_lt hs8_6 (by omega)]
  let s9 := { s8.set 6 (some (dsCap * 16)) with fl := X86Int.Flags.undef, rip := 4201475 }
  have hr9 : RegsOk s9 := regsOk_set hr8 (by decide) _
  have hs9_6 : s9.reg 6 = some (dsCap * 16) := by
    simp only [s9, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr8.1]; decide)]
  have h10 : X86Vm.step vmImage s9 = .ok { s9.set 2 (some (dsCap * 16)) with rip := 4201478 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201475, (.ofInt (.mov 2 6)), 3) ∈ SporosVmBin.vmTable_23),
      exec_mov hs9_6]
  let s10 := { s9.set 2 (some (dsCap * 16)) with rip := 4201478 }
  have hr10 : RegsOk s10 := regsOk_set hr9 (by decide) _
  have hs10_2 : s10.reg 2 = some (dsCap * 16) := by
    simp only [s10, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr9.1]; decide)]
  have h11 : X86Vm.step vmImage s10 =
      .ok { s10.set 2 (some (dsCap * 32)) with fl := X86Int.Flags.undef, rip := 4201482 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201478, (.ofInt (.shlI 2 1)), 4) ∈ SporosVmBin.vmTable_23),
      exec_shlI_1_lt hs10_2 (by omega), Nat.mul_assoc]
  let s11 := { s10.set 2 (some (dsCap * 32)) with fl := X86Int.Flags.undef, rip := 4201482 }
  have hr11 : RegsOk s11 := regsOk_set hr10 (by decide) _
  have hs11_4 : s11.reg 4 = some sp := by
    simp only [s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 2 ≠ 4), List.getElem?_set_ne (by decide : 6 ≠ 4),
      List.getElem?_set_ne (by decide : 7 ≠ 4), List.getElem?_set_ne (by decide : 1 ≠ 4),
      List.getElem?_set_ne (by decide : 0 ≠ 4), List.getElem?_set_ne (by decide : 13 ≠ 4),
      List.getElem?_set_ne (by decide : 12 ≠ 4)]
    exact hsp
  have hea_sp : (sp + X86Env.W - 8) % X86Env.W = sp - 8 := by
    have : sp + X86Env.W - 8 = (sp - 8) + X86Env.W := by omega
    rw [this, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  obtain ⟨istk, m12, hwr12, hget12, hm12eq, hwf12, hstk12, hoth12⟩ :=
    wr64_region hmwf hstk hstkrw hstk_lo (by omega : sp - 8 + 8 ≤ stk.hi) (v := 4201487)
  have hwr12' : wr64 s11.mem ((sp + X86Env.W - 8) % X86Env.W) 4201487 = some m12 := by
    simpa [s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, hea_sp] using hwr12
  have h12 : X86Vm.step vmImage s11 =
      .ok { s11.set 4 (some (sp - 8)) with mem := m12, rip := 4201363 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201482, (.ofInt (.call 4201363)), 5) ∈ SporosVmBin.vmTable_23),
      exec_call hs11_4 hwr12']
    simp [s11, hea_sp]
  let s12 := { s11.set 4 (some (sp - 8)) with mem := m12, rip := 4201363 }
  have hr12' : RegsOk s12 := regsOk_set hr11 (by decide) _
  let stk1 := { stk with data := writeBytes stk.data (sp - 8 - stk.base) (bytes8 4201487) }
  have hstk1_mem : stk1 ∈ m12 := hstk12
  have hstk1_hi : stk1.hi = stk.hi := by simp [stk1, Region.hi, writeBytes_length]
  have hret12 : rd64 s12.mem (sp - 8) = some 4201487 := by
    have hr := rd64_of_writeBytes_same hwf12 hstk12 hstk_lo (by omega : sp - 8 + 8 ≤ stk.hi)
    simpa [s12, show (4201487 % X86Env.W) = 4201487 from by decide] using hr
  have hds_m12 : dsReg ∈ m12 := hoth12 dsReg hds (by rw [hds_base]; exact hstkne_ds.symm)
  have hctx_m12 : ctx ∈ m12 := hoth12 ctx hctx (by rw [hctx_base]; exact hstkne_ctx.symm)
  obtain ⟨idx, hfind12⟩ := mem_find_of_mem m12 dsReg dsBase 0 hwf12.2 hds_m12
    (by simp [hds_base]) (by simp [Region.hi, hds_base, hds_eq]; omega)
  have hfind12' : m12.find dsBase 0 = some (idx, dsReg) := by simpa using hfind12
  have hget_idx := (mem_find_idx_lt m12 dsBase 0 idx dsReg hfind12').2.2
  simp only [Nat.sub_zero] at hget_idx
  have hpos12 : ∀ u ∈ m12, 0 < u.data.length := by
    intro u hu
    rw [hm12eq] at hu
    rcases List.mem_or_eq_of_mem_set hu with hu0 | rfl
    · exact hpos u hu0
    · simpa [writeBytes_length] using hpos stk hstk
  have hpg_16 : X86Env.pages (dsCap * 16) = dsCap * 16 := pages_of_dvd hds_pg
  have hds_pg32 : (dsCap * 32) % X86Env.page = 0 := by
    have : dsCap * 32 = (dsCap * 16) * 2 := by omega
    rw [this, Nat.mul_mod, hds_pg]; decide
  have hpg_32 : X86Env.pages (dsCap * 32) = dsCap * 32 := pages_of_dvd hds_pg32
  have hfree12 : X86Env.Mem.free (m12.eraseIdx idx) a (X86Env.pages (dsCap * 32)) = true := by
    rw [hpg_32]
    refine mem_eraseIdx_free hwf12 hget_idx hpos12 ?_
    intro u hu hub
    have hne_ds : u.base ≠ dsBase := by rw [← hds_base]; exact hub
    rw [hm12eq] at hu
    rcases List.mem_or_eq_of_mem_set hu with hu0 | rfl
    · exact hfree u hu0 hne_ds
    · simp only [Region.hi, writeBytes_length]
      exact hfree stk hstk hstkne_ds
  have hs12_4 : s12.reg 4 = some (sp - 8) := by
    simp only [s12, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr11.1]; decide)]
  have hs12_7 : s12.reg 7 = some dsBase := by
    simp only [s12, s11, s10, s9, s8, s7, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 4 ≠ 7), List.getElem?_set_ne (by decide : 2 ≠ 7),
      List.getElem?_set_ne (by decide : 6 ≠ 7)]
    rw [getElem?_set_getD_self (by rw [hr6'.1]; decide)]
  have hs12_6 : s12.reg 6 = some (dsCap * 16) := by
    simp only [s12, s11, s10, s9, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 4 ≠ 6), List.getElem?_set_ne (by decide : 2 ≠ 6)]
    rw [getElem?_set_getD_self (by rw [hr8.1]; decide)]
  have hs12_2 : s12.reg 2 = some (dsCap * 32) := by
    simp only [s12, s11, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 4 ≠ 2)]
    rw [getElem?_set_getD_self (by rw [hr10.1]; decide)]
  obtain ⟨s13, hrm, hrip13, hk13, hmem13, hm13wf, hr13', hs13_0, hs13_4, hoth13⟩ :=
    fn_mremap_ok_spec (s := s12) (sp := sp - 8) (retAddr := 4201487)
      (oldPtr := dsBase) (oldLen := dsCap * 16) (newLen := dsCap * 32)
      (idx := idx) (a := a) (oldReg := dsReg) (stk := stk1)
      rfl hr12' hs12_4 hstk1_mem (by simp [stk1]; exact hstkne_ds) hstk_lo (by rw [hstk1_hi]; omega)
      hret12 hwf12 hs12_7 hs12_6 hs12_2 hfind12' hds_base (by rw [hpg_16, hds_eq]) hds_rw
      (by rw [hpg_32, hds_eq]; omega) hor ha_pg ha_pos (by rw [hpg_32]; exact ha_lim) hfree12
  have hm13_eq_rep : X86Env.pages (dsCap * 32) - dsReg.data.length = dsCap * 16 := by
    rw [hpg_32, hds_eq]; omega
  let dsNew : Region := ⟨a, dsReg.data ++ List.replicate (dsCap * 16) 0, true⟩
  have hdsNew_m13 : dsNew ∈ s13.mem := by
    rw [hmem13]
    refine List.mem_append_right _ ?_
    simp [dsNew, hm13_eq_rep]
  have hctx_m13 : ctx ∈ s13.mem := by
    rw [hmem13]
    exact List.mem_append_left _ (mem_eraseIdx_other hget_idx hctx_m12 (by rw [hds_base, hctx_base]; exact hne.symm))
  have hstk1_m13 : stk1 ∈ s13.mem := by
    rw [hmem13]
    exact List.mem_append_left _ (mem_eraseIdx_other hget_idx hstk1_mem (by simp [stk1, hds_base]; exact hstkne_ds))
  have hs13_4' : s13.reg 4 = some sp := by
    have : (sp - 8 + 8) % X86Env.W = sp := by rw [Nat.sub_add_cancel h8, Nat.mod_eq_of_lt hspW]
    rwa [this] at hs13_4
  have hs13_15 : s13.reg 15 = some ctxBase := by
    rw [hoth13 15 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    simp only [s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 4 ≠ 15), List.getElem?_set_ne (by decide : 2 ≠ 15),
      List.getElem?_set_ne (by decide : 6 ≠ 15), List.getElem?_set_ne (by decide : 7 ≠ 15),
      List.getElem?_set_ne (by decide : 1 ≠ 15), List.getElem?_set_ne (by decide : 0 ≠ 15),
      List.getElem?_set_ne (by decide : 13 ≠ 15), List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hs13_12 : s13.reg 12 = some w := by
    rw [hoth13 12 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    simp only [s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 4 ≠ 12), List.getElem?_set_ne (by decide : 2 ≠ 12),
      List.getElem?_set_ne (by decide : 6 ≠ 12), List.getElem?_set_ne (by decide : 7 ≠ 12),
      List.getElem?_set_ne (by decide : 1 ≠ 12), List.getElem?_set_ne (by decide : 0 ≠ 12),
      List.getElem?_set_ne (by decide : 13 ≠ 12)]
    rw [getElem?_set_getD_self (by rw [hregs.1]; decide)]
  have hs13_13 : s13.reg 13 = some p := by
    rw [hoth13 13 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    simp only [s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 4 ≠ 13), List.getElem?_set_ne (by decide : 2 ≠ 13),
      List.getElem?_set_ne (by decide : 6 ≠ 13), List.getElem?_set_ne (by decide : 7 ≠ 13),
      List.getElem?_set_ne (by decide : 1 ≠ 13), List.getElem?_set_ne (by decide : 0 ≠ 13)]
    rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  have hm13wf' : s13.mem.wf := by rw [hmem13]; exact hm13wf
  obtain ⟨_, m14, hwr14, _, _, hwf14, hctx14, hoth14⟩ :=
    wr64_region hm13wf' hctx_m13 hctx_rw (a := ctxBase + 0) (v := a)
      (by simp [hctx_base]) (by simp [Region.hi, hctx_base]; have := ctxSize_ge_8; omega)
  have hwr14' : wr64 s13.mem (ea ctxBase 0) a = some m14 := by simpa [hea_c0] using hwr14
  have h14 : X86Vm.step vmImage s13 = .ok { s13 with mem := m14, rip := 4201490 } := by
    rw [step_at_chunk 24 hrip13 rfl (by decide : (4201487, (.ofInt (.st 15 0 0)), 3) ∈ SporosVmBin.vmTable_24),
      exec_st hs13_15 hs13_0 hwr14', hrip13]
  let s14 := { s13 with mem := m14, rip := 4201490 }
  have hr14 : RegsOk s14 := hr13'
  have hs14_15 : s14.reg 15 = some ctxBase := hs13_15
  let ctx14 := { ctx with data := writeBytes ctx.data (ctxBase + 0 - ctx.base) (bytes8 a) }
  have hctx14_hc16 : leVal ((ctx14.data.drop 16).take 8) = dsCap := by
    simp only [ctx14, hctx_base, Nat.sub_self, Nat.add_zero]
    rw [writeBytes_drop_take_disjoint ctx.data 0 16 8 (bytes8 a) (by right; rw [bytes8_length]; omega)
      (by have := ctxSize_ge_24; omega)]
    exact hc16
  have hrd14_c16 : rd64 s14.mem (ea ctxBase 16) = some dsCap := by
    rw [hea_c16]
    exact rd64_of_mem hwf14 hctx14 (by simp [hctx_base])
      (by simp [Region.hi, writeBytes_length, hctx_base]; have := ctxSize_ge_24; omega)
      (by simpa [ctx14, hctx_base] using hctx14_hc16)
  have h15 : X86Vm.step vmImage s14 = .ok { s14.set 1 (some dsCap) with rip := 4201494 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201490, (.ofInt (.ld 1 15 16)), 4) ∈ SporosVmBin.vmTable_24),
      exec_ld hs14_15 hrd14_c16]
  let s15 := { s14.set 1 (some dsCap) with rip := 4201494 }
  have hr15' : RegsOk s15 := regsOk_set hr14 (by decide) dsCap
  have hs15_1 : s15.reg 1 = some dsCap := by
    simp only [s15, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr14.1]; decide)]
  have h16 : X86Vm.step vmImage s15 =
      .ok { s15.set 1 (some (dsCap * 2)) with fl := X86Int.Flags.undef, rip := 4201498 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201494, (.ofInt (.shlI 1 1)), 4) ∈ SporosVmBin.vmTable_24),
      exec_shlI_1_lt hs15_1 (by omega)]
  let s16 := { s15.set 1 (some (dsCap * 2)) with fl := X86Int.Flags.undef, rip := 4201498 }
  have hr16 : RegsOk s16 := regsOk_set hr15' (by decide) _
  have hs16_15 : s16.reg 15 = some ctxBase := by
    simp only [s16, s15, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 15)]
    exact hs14_15
  have hs16_1 : s16.reg 1 = some (dsCap * 2) := by
    simp only [s16, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr15'.1]; decide)]
  obtain ⟨_, m17, hwr17, _, _, hwf17, hctx17, hoth17⟩ :=
    wr64_region hwf14 hctx14 hctx_rw (a := ctxBase + 16) (v := dsCap * 2)
      (by simp [hctx_base])
      (by simp [Region.hi, writeBytes_length, hctx_base]; have := ctxSize_ge_24; omega)
  have hwr17' : wr64 s16.mem (ea ctxBase 16) (dsCap * 2) = some m17 := by simpa [s16, s15, hea_c16] using hwr17
  have h17 : X86Vm.step vmImage s16 = .ok { s16 with mem := m17, rip := 4201502 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201498, (.ofInt (.st 15 16 1)), 4) ∈ SporosVmBin.vmTable_24),
      exec_st hs16_15 hs16_1 hwr17']
  let s17 := { s16 with mem := m17, rip := 4201502 }
  have hr17 : RegsOk s17 := hr16
  have hdsNew_m14 : dsNew ∈ m14 := hoth14 dsNew hdsNew_m13 (by simp [dsNew, hctx_base]; exact hane)
  have hdsNew_m17 : dsNew ∈ m17 := hoth17 dsNew hdsNew_m14 (by simp [dsNew, hctx_base]; exact hane)
  have hstk1_m14 : stk1 ∈ m14 := hoth14 stk1 hstk1_m13 (by simp [stk1, hctx_base]; exact hstkne_ctx)
  have hstk1_m17 : stk1 ∈ m17 := hoth17 stk1 hstk1_m14 (by simp [stk1, hctx_base]; exact hstkne_ctx)
  let ctx17 := { ctx14 with data := writeBytes ctx14.data (ctxBase + 16 - ctx14.base) (bytes8 (dsCap * 2)) }
  have hctx17_hc0 : leVal ((ctx17.data.drop 0).take 8) = a := by
    simp only [ctx17, ctx14, hctx_base, Nat.add_zero, Nat.sub_self, Nat.add_sub_cancel_left]
    rw [writeBytes_drop_take_disjoint _ 16 0 8 (bytes8 (dsCap * 2)) (by left; omega) (by rw [writeBytes_length]; have := ctxSize_ge_24; omega)]
    rw [leVal_writeBytes_same _ 0 a (by have := ctxSize_ge_8; omega)]
    exact Nat.mod_eq_of_lt (by have := vm_lim_lt_W; omega)
  have hctx17_hc8 : leVal ((ctx17.data.drop 8).take 8) = dsLen := by
    simp only [ctx17, ctx14, hctx_base, Nat.add_zero, Nat.sub_self, Nat.add_sub_cancel_left]
    rw [writeBytes_drop_take_disjoint _ 16 8 8 (bytes8 (dsCap * 2)) (by left; omega) (by rw [writeBytes_length]; have := ctxSize_ge_24; omega)]
    rw [writeBytes_drop_take_disjoint ctx.data 0 8 8 (bytes8 a) (by right; rw [bytes8_length]; omega) (by have := ctxSize_ge_16; omega)]
    exact hc8
  have hret_stk1 : rd64 s17.mem sp = some retAddr := by
    have hstk1_drop : (stk1.data.drop (sp - stk1.base)).take 8 = (stk.data.drop (sp - stk.base)).take 8 := by
      simp only [stk1]
      apply writeBytes_drop_take_disjoint
      · right; rw [bytes8_length]; omega
      · simp [Region.hi] at hstk_hi; omega
    obtain ⟨v, hv⟩ := show ∃ v, leVal ((stk.data.drop (sp - stk.base)).take 8) = v from ⟨_, rfl⟩
    have hold := rd64_of_mem hmwf hstk (by omega) hstk_hi hv
    rw [hret, Option.some.injEq] at hold
    subst hold
    exact rd64_of_mem hwf17 hstk1_m17 (by simp [stk1]; omega)
      (by simp [stk1, Region.hi, writeBytes_length] at hstk_hi ⊢; omega) (by rw [hstk1_drop, hv])
  have hdsCapW16 : dsCap * 16 ≤ 2 ^ 47 := by omega
  have hctx17_hi : ctxSize ≤ ctx17.data.length := by
    simp [ctx17, ctx14, writeBytes_length]; exact hctx_len
  have hs17_4 : s17.reg 4 = some sp := by
    simp only [s17, s16, s15, s14, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 4)]
    exact hs13_4'
  have hs17_12 : s17.reg 12 = some w := by
    simp only [s17, s16, s15, s14, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 12)]
    exact hs13_12
  have hs17_13 : s17.reg 13 = some p := by
    simp only [s17, s16, s15, s14, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 13)]
    exact hs13_13
  obtain ⟨m', sf, hreach_join, hripf, hkf, hmemf, hwf', hregsf, hspf, hdsf, hctxf, hothf, hregoth⟩ :=
    fn_ds_push_join_spec (s := s17) (sp := sp) (retAddr := retAddr) (ctxBase := ctxBase)
      (dsBase := a) (dsLen := dsLen) (w := w) (p := p) (ctx := ctx17) (dsReg := dsNew) (stk := stk1)
      rfl hr17 hs17_4 hs16_15 hs17_12 hs17_13
      (by simpa [s17, ctx17, ctx14] using hctx17) (by simp [ctx17, ctx14, hctx_base]) hctx_rw hctx17_hi
      (by simpa [s17] using hdsNew_m17) rfl rfl hane
      (by simpa [s17] using hstk1_m17) (by simp [stk1]; exact hastk.symm)
      (by simp [stk1]; exact hstkne_ctx) (by simp [stk1]; omega)
      (by simp [stk1, Region.hi, writeBytes_length] at hstk_hi ⊢; omega)
      (by simp [dsNew, List.length_append, List.length_replicate, hds_eq]; omega)
      hctxW (by have := vm_lim_lt_W; omega)
      (by omega) hwf17 hret_stk1 hctx17_hc0 hctx17_hc8
  have hstk1_m' : stk1 ∈ m' := by
    apply hothf stk1 (by simpa [s17] using hstk1_m17)
    · simp [stk1]; exact hastk.symm
    · simp [stk1]; exact hstkne_ctx
  refine ⟨m', sf, stk1,
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
      (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step_trans h08
        (Reaches.step_trans h09 (Reaches.step_trans h10 (Reaches.step_trans h11 (Reaches.step_trans h12
          (Reaches.trans hrm (Reaches.step_trans h14 (Reaches.step_trans h15 (Reaches.step_trans h16
            (Reaches.step_trans h17 hreach_join)))))))))))))))),
    hripf, by simpa [s17, s16, s15, s14, hk13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1] using hkf, hmemf, hwf', hregsf, hspf,
    by simpa [dsNew] using hdsf,
    by simpa [ctx17, ctx14, hctx_base, Nat.add_zero, Nat.sub_self, Nat.add_sub_cancel_left] using hctxf,
    hstk1_m', rfl, hstkrw, writeBytes_length _ _ _, ?_, ?_, ?_⟩
  · intro j hj
    apply writeBytes_getElem?_disjoint
    rw [bytes8_length]
    rcases hj with h | h
    · exact Or.inl h
    · exact Or.inr (by omega)
  · intro u hu huds huctx hustk
    have hu12 : u ∈ m12 := hoth12 u hu hustk
    have hu13 : u ∈ s13.mem := by
      rw [hmem13]
      exact List.mem_append_left _ (mem_eraseIdx_other hget_idx hu12 (by rw [hds_base]; exact huds))
    have hu14 : u ∈ m14 := hoth14 u hu13 (by rw [hctx_base]; exact huctx)
    have hu17 : u ∈ m17 := hoth17 u hu14 (by simp [hctx_base]; exact huctx)
    have hua : u.base ≠ a := by
      intro hba
      have hupos := hpos u hu
      rcases hfree u hu huds with h1 | h2
      · omega
      · simp [Region.hi] at h2; omega
    exact hothf u (by simpa [s17] using hu17) hua huctx
  · intro r h0 h1 h2 h4 h6 h7 h8 h9 h10' h11' h12' h13'
    have h17_13 : s17.reg r = s13.reg r := by
      simp only [s17, s16, s15, s14, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne h1.symm]
    rw [hregoth r h0 h1 h2 h4, h17_13,
      hoth13 r h0 h1 h2 h4 h6 h7 h8 h9 h10' h11']
    simp only [s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne h4.symm, List.getElem?_set_ne h2.symm, List.getElem?_set_ne h6.symm,
      List.getElem?_set_ne h7.symm, List.getElem?_set_ne h1.symm, List.getElem?_set_ne h0.symm,
      List.getElem?_set_ne h13'.symm, List.getElem?_set_ne h12'.symm]

theorem fn_ds_push_nomem_spec {s : X86Vm.St} {sp ctxBase dsBase dsLen dsCap w p : Nat}
    {ctx dsReg stk : Region}
    (hrip : s.rip = SporosVmBin.fn_ds_push)
    (hregs : RegsOk s)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr7 : s.reg 7 = some w) (hr6 : s.reg 6 = some p)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_len : ctxSize ≤ ctx.data.length)
    (hds : dsReg ∈ s.mem) (hds_base : dsReg.base = dsBase) (hds_rw : dsReg.rw = true)
    (hstk : stk ∈ s.mem) (hstkrw : stk.rw = true) (hstkne_ds : stk.base ≠ dsBase)
    (h16 : 16 ≤ sp) (hspW : sp < X86Env.W)
    (hstk_lo : stk.base ≤ sp - 16) (hstk_hi : sp ≤ stk.hi)
    (hge : dsCap ≤ dsLen) (hle : dsLen ≤ dsCap) (hcap_pos : 0 < dsCap)
    (hds_eq : dsReg.data.length = dsCap * 16)
    (hds_pg : (dsCap * 16) % X86Env.page = 0)
    (hdsCapW : dsCap * 32 < X86Env.W)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W)
    (hmwf : s.mem.wf)
    (hc0 : leVal ((ctx.data.drop 0).take 8) = dsBase)
    (hc8 : leVal ((ctx.data.drop 8).take 8) = dsLen)
    (hc16 : leVal ((ctx.data.drop 16).take 8) = dsCap)
    (hor : (s.k.next).1 = some .nomem) :
    Ends vmImage s (.exit 2) := by
  have hctxBaseW : ctxBase < X86Env.W := by have := ctxSize_pos; omega
  have hea_c0 : ea ctxBase 0 = ctxBase := by simp [X86Int.ea, Nat.mod_eq_of_lt hctxBaseW]
  have hea_c8 : ea ctxBase 8 = ctxBase + 8 := by
    have : ctxBase + 8 < X86Env.W := by have := ctxSize_ge_24; omega
    simp [X86Int.ea, Nat.mod_eq_of_lt this]
  have hea_c16 : ea ctxBase 16 = ctxBase + 16 := by
    have : ctxBase + 16 < X86Env.W := by have := ctxSize_ge_24; omega
    simp [X86Int.ea, Nat.mod_eq_of_lt this]
  have h01 : X86Vm.step vmImage s = .ok { s.set 12 (some w) with rip := 4201449 } := by
    rw [step_at_chunk 23 hrip rfl (by decide : (SporosVmBin.fn_ds_push, (.ofInt (.mov 12 7)), 3) ∈ SporosVmBin.vmTable_23),
      exec_mov hr7, hrip]
    rfl
  let s1 := { s.set 12 (some w) with rip := 4201449 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) w
  have hs1_6 : s1.reg 6 = some p := by
    simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 12 ≠ 6)]
    exact hr6
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 13 (some p) with rip := 4201452 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201449, (.ofInt (.mov 13 6)), 3) ∈ SporosVmBin.vmTable_23),
      exec_mov hs1_6]
  let s2 := { s1.set 13 (some p) with rip := 4201452 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) p
  have hs2_15 : s2.reg 15 = some ctxBase := by
    simp only [s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 13 ≠ 15), List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hrd_c8 : rd64 s2.mem (ea ctxBase 8) = some dsLen := by
    rw [hea_c8]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; have := ctxSize_ge_16; omega)
      (by simpa [hctx_base] using hc8)
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 0 (some dsLen) with rip := 4201456 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201452, (.ofInt (.ld 0 15 8)), 4) ∈ SporosVmBin.vmTable_23),
      exec_ld hs2_15 hrd_c8]
  let s3 := { s2.set 0 (some dsLen) with rip := 4201456 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) dsLen
  have hs3_15 : s3.reg 15 = some ctxBase := by
    simp only [s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 15), List.getElem?_set_ne (by decide : 13 ≠ 15),
      List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hrd_c16 : rd64 s3.mem (ea ctxBase 16) = some dsCap := by
    rw [hea_c16]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; have := ctxSize_ge_24; omega)
      (by simpa [hctx_base] using hc16)
  have h04 : X86Vm.step vmImage s3 = .ok { s3.set 1 (some dsCap) with rip := 4201460 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201456, (.ofInt (.ld 1 15 16)), 4) ∈ SporosVmBin.vmTable_23),
      exec_ld hs3_15 hrd_c16]
  let s4 := { s3.set 1 (some dsCap) with rip := 4201460 }
  have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) dsCap
  have hs4_0 : s4.reg 0 = some dsLen := by
    simp only [s4, s3, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0)]
    rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
  have hs4_1 : s4.reg 1 = some dsCap := by
    simp only [s4, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
  have h05 : X86Vm.step vmImage s4 =
      .ok { s4 with fl := (X86Int.aluF .sub dsLen dsCap).2, rip := 4201463 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201460, (.ofInt (.cmp 0 1)), 3) ∈ SporosVmBin.vmTable_23),
      exec_cmp hs4_0 hs4_1]
  let s5 := { s4 with fl := (X86Int.aluF .sub dsLen dsCap).2, rip := 4201463 }
  have hr5 : RegsOk s5 := hr4
  have hnotlt : ¬ dsLen < dsCap := by omega
  have hcb : X86Int.cond s5.fl 2 = some false := by
    dsimp [s5]; rw [cond_aluF_sub_b, decide_eq_false hnotlt]
  have h06 : X86Vm.step vmImage s5 = .ok { s5 with rip := 4201465 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201463, (.ofInt (.jcc 2 4201502)), 2) ∈ SporosVmBin.vmTable_23),
      exec_jcc hcb]
    rfl
  let s6 := { s5 with rip := 4201465 }
  have hr6' : RegsOk s6 := hr5
  have hs6_15 : s6.reg 15 = some ctxBase := by
    simp only [s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 15), List.getElem?_set_ne (by decide : 0 ≠ 15),
      List.getElem?_set_ne (by decide : 13 ≠ 15), List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hrd_c0 : rd64 s6.mem (ea ctxBase 0) = some dsBase := by
    rw [hea_c0]
    exact rd64_of_mem hmwf hctx (by simp [hctx_base])
      (by simp [Region.hi, hctx_base]; have := ctxSize_ge_8; omega)
      (by simpa [hctx_base] using hc0)
  have h07 : X86Vm.step vmImage s6 = .ok { s6.set 7 (some dsBase) with rip := 4201468 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201465, (.ofInt (.ld 7 15 0)), 3) ∈ SporosVmBin.vmTable_23),
      exec_ld hs6_15 hrd_c0]
  let s7 := { s6.set 7 (some dsBase) with rip := 4201468 }
  have hr7' : RegsOk s7 := regsOk_set hr6' (by decide) dsBase
  have hs7_1 : s7.reg 1 = some dsCap := by
    simp only [s7, s6, s5, s4, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 7 ≠ 1)]
    rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
  have h08 : X86Vm.step vmImage s7 = .ok { s7.set 6 (some dsCap) with rip := 4201471 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201468, (.ofInt (.mov 6 1)), 3) ∈ SporosVmBin.vmTable_23),
      exec_mov hs7_1]
  let s8 := { s7.set 6 (some dsCap) with rip := 4201471 }
  have hr8 : RegsOk s8 := regsOk_set hr7' (by decide) dsCap
  have hs8_6 : s8.reg 6 = some dsCap := by
    simp only [s8, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr7'.1]; decide)]
  have h09 : X86Vm.step vmImage s8 =
      .ok { s8.set 6 (some (dsCap * 16)) with fl := X86Int.Flags.undef, rip := 4201475 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201471, (.ofInt (.shlI 6 4)), 4) ∈ SporosVmBin.vmTable_23),
      exec_shlI_4_lt hs8_6 (by omega)]
  let s9 := { s8.set 6 (some (dsCap * 16)) with fl := X86Int.Flags.undef, rip := 4201475 }
  have hr9 : RegsOk s9 := regsOk_set hr8 (by decide) _
  have hs9_6 : s9.reg 6 = some (dsCap * 16) := by
    simp only [s9, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr8.1]; decide)]
  have h10 : X86Vm.step vmImage s9 = .ok { s9.set 2 (some (dsCap * 16)) with rip := 4201478 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201475, (.ofInt (.mov 2 6)), 3) ∈ SporosVmBin.vmTable_23),
      exec_mov hs9_6]
  let s10 := { s9.set 2 (some (dsCap * 16)) with rip := 4201478 }
  have hr10 : RegsOk s10 := regsOk_set hr9 (by decide) _
  have hs10_2 : s10.reg 2 = some (dsCap * 16) := by
    simp only [s10, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr9.1]; decide)]
  have h11 : X86Vm.step vmImage s10 =
      .ok { s10.set 2 (some (dsCap * 32)) with fl := X86Int.Flags.undef, rip := 4201482 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201478, (.ofInt (.shlI 2 1)), 4) ∈ SporosVmBin.vmTable_23),
      exec_shlI_1_lt hs10_2 (by omega), Nat.mul_assoc]
  let s11 := { s10.set 2 (some (dsCap * 32)) with fl := X86Int.Flags.undef, rip := 4201482 }
  have hr11 : RegsOk s11 := regsOk_set hr10 (by decide) _
  have hs11_4 : s11.reg 4 = some sp := by
    simp only [s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 2 ≠ 4), List.getElem?_set_ne (by decide : 6 ≠ 4),
      List.getElem?_set_ne (by decide : 7 ≠ 4), List.getElem?_set_ne (by decide : 1 ≠ 4),
      List.getElem?_set_ne (by decide : 0 ≠ 4), List.getElem?_set_ne (by decide : 13 ≠ 4),
      List.getElem?_set_ne (by decide : 12 ≠ 4)]
    exact hsp
  have hea_sp : (sp + X86Env.W - 8) % X86Env.W = sp - 8 := by
    have : sp + X86Env.W - 8 = (sp - 8) + X86Env.W := by omega
    rw [this, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  obtain ⟨istk, m12, hwr12, _, _, hwf12, hstk12, hoth12⟩ :=
    wr64_region hmwf hstk hstkrw (by omega : stk.base ≤ sp - 8) (by omega : sp - 8 + 8 ≤ stk.hi) (v := 4201487)
  have hwr12' : wr64 s11.mem ((sp + X86Env.W - 8) % X86Env.W) 4201487 = some m12 := by
    simpa [s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, hea_sp] using hwr12
  have h12 : X86Vm.step vmImage s11 =
      .ok { s11.set 4 (some (sp - 8)) with mem := m12, rip := 4201363 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201482, (.ofInt (.call 4201363)), 5) ∈ SporosVmBin.vmTable_23),
      exec_call hs11_4 hwr12']
    simp [s11, hea_sp]
  let s12 := { s11.set 4 (some (sp - 8)) with mem := m12, rip := 4201363 }
  have hr12' : RegsOk s12 := regsOk_set hr11 (by decide) _
  let stk1 := { stk with data := writeBytes stk.data (sp - 8 - stk.base) (bytes8 4201487) }
  have hstk1_hi : stk1.hi = stk.hi := by simp [stk1, Region.hi, writeBytes_length]
  have hds_m12 : dsReg ∈ m12 := hoth12 dsReg hds (by rw [hds_base]; exact hstkne_ds.symm)
  obtain ⟨idx, hfind12⟩ := mem_find_of_mem m12 dsReg dsBase 0 hwf12.2 hds_m12
    (by simp [hds_base]) (by simp [Region.hi, hds_base, hds_eq]; omega)
  have hfind12' : m12.find dsBase 0 = some (idx, dsReg) := by simpa using hfind12
  have hpg_16 : X86Env.pages (dsCap * 16) = dsCap * 16 := pages_of_dvd hds_pg
  have hds_pg32 : (dsCap * 32) % X86Env.page = 0 := by
    have : dsCap * 32 = (dsCap * 16) * 2 := by omega
    rw [this, Nat.mul_mod, hds_pg]; decide
  have hpg_32 : X86Env.pages (dsCap * 32) = dsCap * 32 := pages_of_dvd hds_pg32
  have hs12_4 : s12.reg 4 = some (sp - 8) := by
    simp only [s12, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr11.1]; decide)]
  have hs12_7 : s12.reg 7 = some dsBase := by
    simp only [s12, s11, s10, s9, s8, s7, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 4 ≠ 7), List.getElem?_set_ne (by decide : 2 ≠ 7),
      List.getElem?_set_ne (by decide : 6 ≠ 7)]
    rw [getElem?_set_getD_self (by rw [hr6'.1]; decide)]
  have hs12_6 : s12.reg 6 = some (dsCap * 16) := by
    simp only [s12, s11, s10, s9, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 4 ≠ 6), List.getElem?_set_ne (by decide : 2 ≠ 6)]
    rw [getElem?_set_getD_self (by rw [hr8.1]; decide)]
  have hs12_2 : s12.reg 2 = some (dsCap * 32) := by
    simp only [s12, s11, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 4 ≠ 2)]
    rw [getElem?_set_getD_self (by rw [hr10.1]; decide)]
  have h_exit := fn_mremap_nomem_spec (s := s12) (sp := sp - 8)
    (oldPtr := dsBase) (oldLen := dsCap * 16) (newLen := dsCap * 32)
    (idx := idx) (oldReg := dsReg) (stk := stk1)
    rfl hr12' hs12_4 (by omega) (by omega) hwf12 hstk12 hstkrw
    (by simp [stk1]; omega) (by rw [hstk1_hi]; omega)
    hs12_7 hs12_6 hs12_2 hfind12' hds_base (by rw [hpg_16, hds_eq]) hds_rw
    (by rw [hpg_32, hds_eq]; omega) hor
  exact Ends.step_trans h01 (Ends.step_trans h02 (Ends.step_trans h03 (Ends.step_trans h04
    (Ends.step_trans h05 (Ends.step_trans h06 (Ends.step_trans h07 (Ends.step_trans h08
      (Ends.step_trans h09 (Ends.step_trans h10 (Ends.step_trans h11 (Ends.step_trans h12 h_exit)))))))))))

theorem fn_aux_push_join_spec {s : X86Vm.St} {sp retAddr ctxBase auxBase auxLen w p : Nat}
    {ctx auxReg stk : Region}
    (hrip : s.rip = 4201592)
    (hregs : RegsOk s)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr12 : s.reg 12 = some w) (hr13 : s.reg 13 = some p)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_rw : ctx.rw = true)
    (hctx_len : ctxSize ≤ ctx.data.length)
    (haux : auxReg ∈ s.mem) (haux_base : auxReg.base = auxBase) (haux_rw : auxReg.rw = true)
    (hne : auxBase ≠ ctxBase)
    (hstk : stk ∈ s.mem) (hstkne_aux : stk.base ≠ auxBase) (hstkne_ctx : stk.base ≠ ctxBase)
    (hstk_lo : stk.base ≤ sp) (hstk_hi : sp + 8 ≤ stk.hi)
    (haux_hi : auxLen * 16 + 16 ≤ auxReg.data.length)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W) (hauxW : auxBase + auxLen * 16 + 16 ≤ X86Env.W)
    (hauxLenW : auxLen * 16 < X86Env.W)
    (hmwf : s.mem.wf)
    (hret : rd64 s.mem sp = some retAddr)
    (hc24 : leVal ((ctx.data.drop 24).take 8) = auxBase)
    (hc32 : leVal ((ctx.data.drop 32).take 8) = auxLen) :
    let auxData' := writeBytes (writeBytes auxReg.data (auxLen * 16) (bytes8 w)) (auxLen * 16 + 8) (bytes8 p)
    let ctxData' := writeBytes ctx.data 32 (bytes8 (auxLen + 1))
    ∃ m' s', Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.k = s.k ∧ s'.mem = m' ∧ m'.wf ∧ RegsOk s' ∧
      s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      { auxReg with data := auxData' } ∈ m' ∧
      { ctx with data := ctxData' } ∈ m' ∧
      (∀ u ∈ s.mem, u.base ≠ auxBase → u.base ≠ ctxBase → u ∈ m') ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 4 → s'.reg r = s.reg r) := by
  have hcs : ctxSize = 4096 := rfl
  have hea_c24 : ea ctxBase 24 = ctxBase + 24 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 24 < X86Env.W)]
  have hea_c32 : ea ctxBase 32 = ctxBase + 32 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 32 < X86Env.W)]
  have hrd_c24 : rd64 s.mem (ea ctxBase 24) = some auxBase := by
    rw [hea_c24]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc24)
  have h01 : X86Vm.step vmImage s = .ok { s.set 0 (some auxBase) with rip := 4201596 } := by
    rw [step_at_chunk 24 hrip rfl (by decide : (4201592, (.ofInt (.ld 0 15 24)), 4) ∈ SporosVmBin.vmTable_24),
      exec_ld hr15 hrd_c24, hrip]
  let s1 := { s.set 0 (some auxBase) with rip := 4201596 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) auxBase
  have hs1_15 : s1.reg 15 = some ctxBase := by
    simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 15)]
    exact hr15
  have hrd_c32 : rd64 s1.mem (ea ctxBase 32) = some auxLen := by
    rw [hea_c32]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc32)
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 1 (some auxLen) with rip := 4201600 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201596, (.ofInt (.ld 1 15 32)), 4) ∈ SporosVmBin.vmTable_24),
      exec_ld hs1_15 hrd_c32]
  let s2 := { s1.set 1 (some auxLen) with rip := 4201600 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) auxLen
  have hs2_1 : s2.reg 1 = some auxLen := by
    simp only [s2, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 2 (some auxLen) with rip := 4201603 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201600, (.ofInt (.mov 2 1)), 3) ∈ SporosVmBin.vmTable_24),
      exec_mov hs2_1]
  let s3 := { s2.set 2 (some auxLen) with rip := 4201603 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) auxLen
  have hs3_2 : s3.reg 2 = some auxLen := by
    simp only [s3, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
  have h04 : X86Vm.step vmImage s3 =
      .ok { s3.set 2 (some (auxLen * 16)) with fl := X86Int.Flags.undef, rip := 4201607 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201603, (.ofInt (.shlI 2 4)), 4) ∈ SporosVmBin.vmTable_24),
      exec_shlI_4_lt hs3_2 hauxLenW]
  let s4 := { s3.set 2 (some (auxLen * 16)) with fl := X86Int.Flags.undef, rip := 4201607 }
  have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) _
  have hs4_0 : s4.reg 0 = some auxBase := by
    simp only [s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 2 ≠ 0), List.getElem?_set_ne (by decide : 1 ≠ 0)]
    rw [getElem?_set_getD_self (by rw [hregs.1]; decide)]
  have hs4_2 : s4.reg 2 = some (auxLen * 16) := by
    simp only [s4, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
  have hs4_12 : s4.reg 12 = some w := by
    simp only [s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 2 ≠ 12), List.getElem?_set_ne (by decide : 1 ≠ 12),
      List.getElem?_set_ne (by decide : 0 ≠ 12)]
    exact hr12
  have hea_d0 : eaIdx auxBase (auxLen * 16) 1 0 = auxBase + auxLen * 16 := by
    simp [eaIdx, Nat.mod_eq_of_lt (by omega : auxBase + auxLen * 16 < X86Env.W)]
  obtain ⟨i1, m5, hwr5, _, _, hwf5, hr5_aux, hoth5⟩ :=
    wrMem_w64_region hmwf haux haux_rw (a := auxBase + auxLen * 16) (v := w)
      (by rw [haux_base]; omega) (by simp [Region.hi, haux_base]; omega)
  have hwr5' : wrMem .w64 s4.mem (eaIdx auxBase (auxLen * 16) 1 0) w = some m5 := by
    simpa [s4, s3, s2, s1, hea_d0] using hwr5
  have h05 : X86Vm.step vmImage s4 = .ok { s4 with mem := m5, rip := 4201611 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201607, (.stIdx .w64 0 2 1 0 12), 4) ∈ SporosVmBin.vmTable_24),
      exec_stIdx hs4_0 hs4_2 hs4_12 hwr5']
  let s5 := { s4 with mem := m5, rip := 4201611 }
  have hr5 : RegsOk s5 := hr4
  have hs5_0 : s5.reg 0 = some auxBase := hs4_0
  have hs5_2 : s5.reg 2 = some (auxLen * 16) := hs4_2
  have hs5_13 : s5.reg 13 = some p := by
    simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 2 ≠ 13), List.getElem?_set_ne (by decide : 1 ≠ 13),
      List.getElem?_set_ne (by decide : 0 ≠ 13)]
    exact hr13
  have hea_d8 : eaIdx auxBase (auxLen * 16) 1 8 = auxBase + auxLen * 16 + 8 := by
    simp [eaIdx, Nat.mod_eq_of_lt (by omega : auxBase + auxLen * 16 + 8 < X86Env.W)]
  let aux1 := { auxReg with data := writeBytes auxReg.data (auxBase + auxLen * 16 - auxReg.base) (bytes8 w) }
  have haux1_hi : auxBase + auxLen * 16 + 16 ≤ aux1.hi := by
    simp [aux1, Region.hi, writeBytes_length, haux_base]; omega
  obtain ⟨i2, m6, hwr6, _, _, hwf6, hr6_aux, hoth6⟩ :=
    wrMem_w64_region hwf5 hr5_aux haux_rw (a := auxBase + auxLen * 16 + 8) (v := p)
      (by simp [haux_base]; omega) haux1_hi
  have hwr6' : wrMem .w64 s5.mem (eaIdx auxBase (auxLen * 16) 1 8) p = some m6 := by
    simpa [s5, hea_d8] using hwr6
  have h06 : X86Vm.step vmImage s5 = .ok { s5 with mem := m6, rip := 4201616 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201611, (.stIdx .w64 0 2 1 8 13), 5) ∈ SporosVmBin.vmTable_24),
      exec_stIdx hs5_0 hs5_2 hs5_13 hwr6']
  let s6 := { s5 with mem := m6, rip := 4201616 }
  have hr6 : RegsOk s6 := hr5
  have hs6_1 : s6.reg 1 = some auxLen := by
    simp only [s6, s5, s4, s3, s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 1)]
    rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  have h07 : X86Vm.step vmImage s6 =
      .ok { s6.set 1 (some ((auxLen + 1) % X86Env.W)) with
            fl := (X86Int.aluF .add auxLen 1).2, rip := 4201620 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201616, (.ofInt (.aluI .add 1 1)), 4) ∈ SporosVmBin.vmTable_24),
      exec_aluI_add_lt hs6_1 (by decide)]
  let s7 := { s6.set 1 (some ((auxLen + 1) % X86Env.W)) with
                fl := (X86Int.aluF .add auxLen 1).2, rip := 4201620 }
  have hr7' : RegsOk s7 := regsOk_set hr6 (by decide) _
  have hs7_15 : s7.reg 15 = some ctxBase := by
    simp only [s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 15), List.getElem?_set_ne (by decide : 2 ≠ 15),
      List.getElem?_set_ne (by decide : 0 ≠ 15)]
    exact hr15
  have hs7_1 : s7.reg 1 = some ((auxLen + 1) % X86Env.W) := by
    simp only [s7, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr6.1]; decide)]
  have hctx_m5 : ctx ∈ m5 := hoth5 ctx hctx (by rw [hctx_base, haux_base]; exact Ne.symm hne)
  have hctx_m6 : ctx ∈ m6 := hoth6 ctx hctx_m5 (by simp [hctx_base, haux_base]; exact Ne.symm hne)
  obtain ⟨i3, m7, hwr7, _, _, hwf7, hr7_ctx, hoth7⟩ :=
    wr64_region hwf6 hctx_m6 hctx_rw (a := ctxBase + 32) (v := (auxLen + 1) % X86Env.W)
      (by rw [hctx_base]; omega) (by simp [Region.hi, hctx_base]; omega)
  have hwr7' : wr64 s7.mem (ea ctxBase 32) ((auxLen + 1) % X86Env.W) = some m7 := by
    simpa [s7, s6, hea_c32] using hwr7
  have h08 : X86Vm.step vmImage s7 = .ok { s7 with mem := m7, rip := 4201624 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201620, (.ofInt (.st 15 32 1)), 4) ∈ SporosVmBin.vmTable_24),
      exec_st hs7_15 hs7_1 hwr7']
  let s8 := { s7 with mem := m7, rip := 4201624 }
  have hr8 : RegsOk s8 := hr7'
  have hs8_4 : s8.reg 4 = some sp := by
    simp only [s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 4), List.getElem?_set_ne (by decide : 2 ≠ 4),
      List.getElem?_set_ne (by decide : 0 ≠ 4)]
    exact hsp
  have hstk_m5 : stk ∈ m5 := hoth5 stk hstk (by rw [haux_base]; exact hstkne_aux)
  have hstk_m6 : stk ∈ m6 := hoth6 stk hstk_m5 (by simp [haux_base]; exact hstkne_aux)
  have hstk_m7 : stk ∈ m7 := hoth7 stk hstk_m6 (by rw [hctx_base]; exact hstkne_ctx)
  have hret_m7 : rd64 s8.mem sp = some retAddr :=
    rd64_of_mem_transfer hmwf hwf7 hstk hstk_m7 hstk_lo hstk_hi hret
  have h09 : X86Vm.step vmImage s8 =
      .ok { s8.set 4 (some ((sp + 8) % X86Env.W)) with rip := retAddr } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201624, (.ofInt .ret), 1) ∈ SporosVmBin.vmTable_24),
      exec_ret hs8_4 hret_m7]
  let s9 := { s8.set 4 (some ((sp + 8) % X86Env.W)) with rip := retAddr }
  refine ⟨m7, s9,
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
      (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step_trans h08
        (Reaches.step h09)))))))),
    rfl, rfl, rfl, hwf7, regsOk_set hr8 (by decide) _, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [s9, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr8.1]; decide)]
  · have haux2_m7 :
        { aux1 with data := writeBytes aux1.data (auxBase + auxLen * 16 + 8 - aux1.base) (bytes8 p) } ∈ m7 :=
      hoth7 _ hr6_aux (by simp [aux1, haux_base, hctx_base]; exact hne)
    simpa [aux1, haux_base, show auxBase + auxLen * 16 - auxBase = auxLen * 16 from by omega,
      show auxBase + auxLen * 16 + 8 - auxBase = auxLen * 16 + 8 from by omega] using haux2_m7
  · simpa [hctx_base, show ctxBase + 32 - ctxBase = 32 from by omega,
      Nat.mod_eq_of_lt (by omega : auxLen + 1 < X86Env.W)] using hr7_ctx
  · intro u hu huaux huctx
    have hu5 := hoth5 u hu (by rw [haux_base]; exact huaux)
    have hu6 := hoth6 u hu5 (by simp [haux_base]; exact huaux)
    exact hoth7 u hu6 (by rw [hctx_base]; exact huctx)
  · intro r h0 h1 h2 h4
    simp only [s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne h4.symm, List.getElem?_set_ne h1.symm,
      List.getElem?_set_ne h2.symm, List.getElem?_set_ne h0.symm]

theorem fn_aux_push_nogrow_spec {s : X86Vm.St} {sp retAddr ctxBase auxBase auxLen auxCap w p : Nat}
    {ctx auxReg stk : Region}
    (hrip : s.rip = SporosVmBin.fn_aux_push)
    (hregs : RegsOk s)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr7 : s.reg 7 = some w) (hr6 : s.reg 6 = some p)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_rw : ctx.rw = true)
    (hctx_len : ctxSize ≤ ctx.data.length)
    (haux : auxReg ∈ s.mem) (haux_base : auxReg.base = auxBase) (haux_rw : auxReg.rw = true)
    (hne : auxBase ≠ ctxBase)
    (hstk : stk ∈ s.mem) (hstkne_aux : stk.base ≠ auxBase) (hstkne_ctx : stk.base ≠ ctxBase)
    (hstk_lo : stk.base ≤ sp) (hstk_hi : sp + 8 ≤ stk.hi)
    (hlt : auxLen < auxCap)
    (haux_hi : auxCap * 16 ≤ auxReg.data.length)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W) (hauxW : auxBase + auxCap * 16 ≤ X86Env.W)
    (hauxCapW : auxCap * 16 < X86Env.W)
    (hmwf : s.mem.wf)
    (hret : rd64 s.mem sp = some retAddr)
    (hc24 : leVal ((ctx.data.drop 24).take 8) = auxBase)
    (hc32 : leVal ((ctx.data.drop 32).take 8) = auxLen)
    (hc40 : leVal ((ctx.data.drop 40).take 8) = auxCap) :
    let auxData' := writeBytes (writeBytes auxReg.data (auxLen * 16) (bytes8 w)) (auxLen * 16 + 8) (bytes8 p)
    let ctxData' := writeBytes ctx.data 32 (bytes8 (auxLen + 1))
    ∃ m' s', Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.k = s.k ∧ s'.mem = m' ∧ m'.wf ∧ RegsOk s' ∧
      s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      { auxReg with data := auxData' } ∈ m' ∧
      { ctx with data := ctxData' } ∈ m' ∧
      (∀ u ∈ s.mem, u.base ≠ auxBase → u.base ≠ ctxBase → u ∈ m') ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 4 → r ≠ 12 → r ≠ 13 → s'.reg r = s.reg r) := by
  have hcs : ctxSize = 4096 := rfl
  have hea_c32 : ea ctxBase 32 = ctxBase + 32 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 32 < X86Env.W)]
  have hea_c40 : ea ctxBase 40 = ctxBase + 40 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 40 < X86Env.W)]
  have h01 : X86Vm.step vmImage s = .ok { s.set 12 (some w) with rip := 4201537 } := by
    rw [step_at_chunk 24 hrip rfl (by decide : (SporosVmBin.fn_aux_push, (.ofInt (.mov 12 7)), 3) ∈ SporosVmBin.vmTable_24),
      exec_mov hr7, hrip]
    rfl
  let s1 := { s.set 12 (some w) with rip := 4201537 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) w
  have hs1_6 : s1.reg 6 = some p := by
    simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 12 ≠ 6)]
    exact hr6
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 13 (some p) with rip := 4201540 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201537, (.ofInt (.mov 13 6)), 3) ∈ SporosVmBin.vmTable_24),
      exec_mov hs1_6]
  let s2 := { s1.set 13 (some p) with rip := 4201540 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) p
  have hs2_15 : s2.reg 15 = some ctxBase := by
    simp only [s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 13 ≠ 15), List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hrd_c32 : rd64 s2.mem (ea ctxBase 32) = some auxLen := by
    rw [hea_c32]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc32)
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 0 (some auxLen) with rip := 4201544 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201540, (.ofInt (.ld 0 15 32)), 4) ∈ SporosVmBin.vmTable_24),
      exec_ld hs2_15 hrd_c32]
  let s3 := { s2.set 0 (some auxLen) with rip := 4201544 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) auxLen
  have hs3_15 : s3.reg 15 = some ctxBase := by
    simp only [s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 15), List.getElem?_set_ne (by decide : 13 ≠ 15),
      List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hrd_c40 : rd64 s3.mem (ea ctxBase 40) = some auxCap := by
    rw [hea_c40]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc40)
  have h04 : X86Vm.step vmImage s3 = .ok { s3.set 1 (some auxCap) with rip := 4201548 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201544, (.ofInt (.ld 1 15 40)), 4) ∈ SporosVmBin.vmTable_24),
      exec_ld hs3_15 hrd_c40]
  let s4 := { s3.set 1 (some auxCap) with rip := 4201548 }
  have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) auxCap
  have hs4_0 : s4.reg 0 = some auxLen := by
    simp only [s4, s3, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0)]
    rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
  have hs4_1 : s4.reg 1 = some auxCap := by
    simp only [s4, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
  have h05 : X86Vm.step vmImage s4 =
      .ok { s4 with fl := (X86Int.aluF .sub auxLen auxCap).2, rip := 4201551 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201548, (.ofInt (.cmp 0 1)), 3) ∈ SporosVmBin.vmTable_24),
      exec_cmp hs4_0 hs4_1]
  let s5 := { s4 with fl := (X86Int.aluF .sub auxLen auxCap).2, rip := 4201551 }
  have hr5 : RegsOk s5 := hr4
  have hcb : X86Int.cond s5.fl 2 = some true := by
    dsimp [s5]; rw [cond_aluF_sub_b, decide_eq_true hlt]
  have h06 : X86Vm.step vmImage s5 = .ok { s5 with rip := 4201592 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201551, (.ofInt (.jcc 2 4201592)), 2) ∈ SporosVmBin.vmTable_24),
      exec_jcc hcb]
    rfl
  let s6 := { s5 with rip := 4201592 }
  have hr6' : RegsOk s6 := hr5
  have hs6_4 : s6.reg 4 = some sp := by
    simp only [s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 4), List.getElem?_set_ne (by decide : 0 ≠ 4),
      List.getElem?_set_ne (by decide : 13 ≠ 4), List.getElem?_set_ne (by decide : 12 ≠ 4)]
    exact hsp
  have hs6_15 : s6.reg 15 = some ctxBase := by
    simp only [s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 15), List.getElem?_set_ne (by decide : 0 ≠ 15),
      List.getElem?_set_ne (by decide : 13 ≠ 15), List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hs6_12 : s6.reg 12 = some w := by
    simp only [s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 12), List.getElem?_set_ne (by decide : 0 ≠ 12),
      List.getElem?_set_ne (by decide : 13 ≠ 12)]
    rw [getElem?_set_getD_self (by rw [hregs.1]; decide)]
  have hs6_13 : s6.reg 13 = some p := by
    simp only [s6, s5, s4, s3, s2, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 13), List.getElem?_set_ne (by decide : 0 ≠ 13)]
    rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  obtain ⟨m', sf, hreach, hripf, hkf, hmemf, hwf', hregsf, hspf, hauxf, hctxf, hothf, hregoth⟩ :=
    fn_aux_push_join_spec (s := s6) (sp := sp) (retAddr := retAddr) (ctxBase := ctxBase)
      (auxBase := auxBase) (auxLen := auxLen) (w := w) (p := p) (ctx := ctx) (auxReg := auxReg) (stk := stk)
      rfl hr6' hs6_4 hs6_15 hs6_12 hs6_13 hctx hctx_base hctx_rw hctx_len haux haux_base haux_rw hne
      hstk hstkne_aux hstkne_ctx hstk_lo hstk_hi (by omega) hctxW (by omega) (by omega) hmwf
      hret hc24 hc32
  refine ⟨m', sf,
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
      (Reaches.step_trans h05 (Reaches.step_trans h06 hreach))))),
    hripf, hkf, hmemf, hwf', hregsf, hspf, hauxf, hctxf, hothf, ?_⟩
  intro r h0 h1 h2 h4 h12 h13
  rw [hregoth r h0 h1 h2 h4]
  simp only [s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
    List.getElem?_set_ne h1.symm, List.getElem?_set_ne h0.symm,
    List.getElem?_set_ne h13.symm, List.getElem?_set_ne h12.symm]

theorem fn_aux_push_grow_spec {s : X86Vm.St} {sp retAddr ctxBase auxBase auxLen auxCap w p a : Nat}
    {ctx auxReg stk : Region}
    (hrip : s.rip = SporosVmBin.fn_aux_push)
    (hregs : RegsOk s)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr7 : s.reg 7 = some w) (hr6 : s.reg 6 = some p)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_rw : ctx.rw = true)
    (hctx_len : ctxSize ≤ ctx.data.length)
    (haux : auxReg ∈ s.mem) (haux_base : auxReg.base = auxBase) (haux_rw : auxReg.rw = true)
    (hne : auxBase ≠ ctxBase) (hane : a ≠ ctxBase) (hastk : a ≠ stk.base)
    (hstk : stk ∈ s.mem) (hstkrw : stk.rw = true) (hstkne_aux : stk.base ≠ auxBase) (hstkne_ctx : stk.base ≠ ctxBase)
    (h8 : 8 ≤ sp) (hspW : sp < X86Env.W)
    (hstk_lo : stk.base ≤ sp - 8) (hstk_hi : sp + 8 ≤ stk.hi)
    (hge : auxCap ≤ auxLen) (hle : auxLen ≤ auxCap) (hcap_pos : 0 < auxCap)
    (haux_eq : auxReg.data.length = auxCap * 16)
    (haux_pg : (auxCap * 16) % X86Env.page = 0)
    (hauxCapW : auxCap * 32 < X86Env.W)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W)
    (hmwf : s.mem.wf) (hpos : ∀ u ∈ s.mem, 0 < u.data.length)
    (hret : rd64 s.mem sp = some retAddr)
    (hc24 : leVal ((ctx.data.drop 24).take 8) = auxBase)
    (hc32 : leVal ((ctx.data.drop 32).take 8) = auxLen)
    (hc40 : leVal ((ctx.data.drop 40).take 8) = auxCap)
    (hor : (s.k.next).1 = some (.at a))
    (ha_pg : a % X86Env.page = 0) (ha_pos : 0 < a) (ha_lim : a + auxCap * 32 ≤ 2 ^ 47)
    (hfree : ∀ u ∈ s.mem, u.base ≠ auxBase → a + auxCap * 32 ≤ u.base ∨ u.hi ≤ a) :
    let auxData' := writeBytes (writeBytes (auxReg.data ++ List.replicate (auxCap * 16) 0) (auxLen * 16) (bytes8 w))
      (auxLen * 16 + 8) (bytes8 p)
    let ctxData' := writeBytes (writeBytes (writeBytes ctx.data 24 (bytes8 a)) 40 (bytes8 (auxCap * 2))) 32 (bytes8 (auxLen + 1))
    ∃ m' s' stk', Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.k = (s.k.next).2 ∧ s'.mem = m' ∧ m'.wf ∧ RegsOk s' ∧
      s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      (⟨a, auxData', true⟩ : Region) ∈ m' ∧
      { ctx with data := ctxData' } ∈ m' ∧
      stk' ∈ m' ∧ stk'.base = stk.base ∧ stk'.rw = true ∧ stk'.data.length = stk.data.length ∧
      (∀ j, j < sp - 8 - stk.base ∨ sp - stk.base ≤ j → stk'.data[j]? = stk.data[j]?) ∧
      (∀ u ∈ s.mem, u.base ≠ auxBase → u.base ≠ ctxBase → u.base ≠ stk.base → u ∈ m') ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 4 → r ≠ 6 → r ≠ 7 → r ≠ 8 → r ≠ 9 → r ≠ 10 → r ≠ 11 → r ≠ 12 → r ≠ 13 → s'.reg r = s.reg r) := by
  have hcs : ctxSize = 4096 := rfl
  have hea_c24 : ea ctxBase 24 = ctxBase + 24 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 24 < X86Env.W)]
  have hea_c32 : ea ctxBase 32 = ctxBase + 32 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 32 < X86Env.W)]
  have hea_c40 : ea ctxBase 40 = ctxBase + 40 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 40 < X86Env.W)]
  have h01 : X86Vm.step vmImage s = .ok { s.set 12 (some w) with rip := 4201537 } := by
    rw [step_at_chunk 24 hrip rfl (by decide : (SporosVmBin.fn_aux_push, (.ofInt (.mov 12 7)), 3) ∈ SporosVmBin.vmTable_24),
      exec_mov hr7, hrip]
    rfl
  let s1 := { s.set 12 (some w) with rip := 4201537 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) w
  have hs1_6 : s1.reg 6 = some p := by
    simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 12 ≠ 6)]
    exact hr6
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 13 (some p) with rip := 4201540 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201537, (.ofInt (.mov 13 6)), 3) ∈ SporosVmBin.vmTable_24),
      exec_mov hs1_6]
  let s2 := { s1.set 13 (some p) with rip := 4201540 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) p
  have hs2_15 : s2.reg 15 = some ctxBase := by
    simp only [s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 13 ≠ 15), List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hrd_c32 : rd64 s2.mem (ea ctxBase 32) = some auxLen := by
    rw [hea_c32]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc32)
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 0 (some auxLen) with rip := 4201544 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201540, (.ofInt (.ld 0 15 32)), 4) ∈ SporosVmBin.vmTable_24),
      exec_ld hs2_15 hrd_c32]
  let s3 := { s2.set 0 (some auxLen) with rip := 4201544 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) auxLen
  have hs3_15 : s3.reg 15 = some ctxBase := by
    simp only [s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 15), List.getElem?_set_ne (by decide : 13 ≠ 15),
      List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hrd_c40 : rd64 s3.mem (ea ctxBase 40) = some auxCap := by
    rw [hea_c40]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc40)
  have h04 : X86Vm.step vmImage s3 = .ok { s3.set 1 (some auxCap) with rip := 4201548 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201544, (.ofInt (.ld 1 15 40)), 4) ∈ SporosVmBin.vmTable_24),
      exec_ld hs3_15 hrd_c40]
  let s4 := { s3.set 1 (some auxCap) with rip := 4201548 }
  have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) auxCap
  have hs4_0 : s4.reg 0 = some auxLen := by
    simp only [s4, s3, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0)]
    rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
  have hs4_1 : s4.reg 1 = some auxCap := by
    simp only [s4, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
  have h05 : X86Vm.step vmImage s4 =
      .ok { s4 with fl := (X86Int.aluF .sub auxLen auxCap).2, rip := 4201551 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201548, (.ofInt (.cmp 0 1)), 3) ∈ SporosVmBin.vmTable_24),
      exec_cmp hs4_0 hs4_1]
  let s5 := { s4 with fl := (X86Int.aluF .sub auxLen auxCap).2, rip := 4201551 }
  have hr5 : RegsOk s5 := hr4
  have hnotlt : ¬ auxLen < auxCap := by omega
  have hcb : X86Int.cond s5.fl 2 = some false := by
    dsimp [s5]; rw [cond_aluF_sub_b, decide_eq_false hnotlt]
  have h06 : X86Vm.step vmImage s5 = .ok { s5 with rip := 4201553 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201551, (.ofInt (.jcc 2 4201592)), 2) ∈ SporosVmBin.vmTable_24),
      exec_jcc hcb]
    rfl
  let s6 := { s5 with rip := 4201553 }
  have hr6' : RegsOk s6 := hr5
  have hs6_15 : s6.reg 15 = some ctxBase := by
    simp only [s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 15), List.getElem?_set_ne (by decide : 0 ≠ 15),
      List.getElem?_set_ne (by decide : 13 ≠ 15), List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hrd_c24 : rd64 s6.mem (ea ctxBase 24) = some auxBase := by
    rw [hea_c24]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc24)
  have h07 : X86Vm.step vmImage s6 = .ok { s6.set 7 (some auxBase) with rip := 4201557 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201553, (.ofInt (.ld 7 15 24)), 4) ∈ SporosVmBin.vmTable_24),
      exec_ld hs6_15 hrd_c24]
  let s7 := { s6.set 7 (some auxBase) with rip := 4201557 }
  have hr7' : RegsOk s7 := regsOk_set hr6' (by decide) auxBase
  have hs7_1 : s7.reg 1 = some auxCap := by
    simp only [s7, s6, s5, s4, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 7 ≠ 1)]
    rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
  have h08 : X86Vm.step vmImage s7 = .ok { s7.set 6 (some auxCap) with rip := 4201560 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201557, (.ofInt (.mov 6 1)), 3) ∈ SporosVmBin.vmTable_24),
      exec_mov hs7_1]
  let s8 := { s7.set 6 (some auxCap) with rip := 4201560 }
  have hr8 : RegsOk s8 := regsOk_set hr7' (by decide) auxCap
  have hs8_6 : s8.reg 6 = some auxCap := by
    simp only [s8, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr7'.1]; decide)]
  have h09 : X86Vm.step vmImage s8 =
      .ok { s8.set 6 (some (auxCap * 16)) with fl := X86Int.Flags.undef, rip := 4201564 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201560, (.ofInt (.shlI 6 4)), 4) ∈ SporosVmBin.vmTable_24),
      exec_shlI_4_lt hs8_6 (by omega)]
  let s9 := { s8.set 6 (some (auxCap * 16)) with fl := X86Int.Flags.undef, rip := 4201564 }
  have hr9 : RegsOk s9 := regsOk_set hr8 (by decide) _
  have hs9_6 : s9.reg 6 = some (auxCap * 16) := by
    simp only [s9, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr8.1]; decide)]
  have h10 : X86Vm.step vmImage s9 = .ok { s9.set 2 (some (auxCap * 16)) with rip := 4201567 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201564, (.ofInt (.mov 2 6)), 3) ∈ SporosVmBin.vmTable_24),
      exec_mov hs9_6]
  let s10 := { s9.set 2 (some (auxCap * 16)) with rip := 4201567 }
  have hr10 : RegsOk s10 := regsOk_set hr9 (by decide) _
  have hs10_2 : s10.reg 2 = some (auxCap * 16) := by
    simp only [s10, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr9.1]; decide)]
  have h11 : X86Vm.step vmImage s10 =
      .ok { s10.set 2 (some (auxCap * 32)) with fl := X86Int.Flags.undef, rip := 4201571 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201567, (.ofInt (.shlI 2 1)), 4) ∈ SporosVmBin.vmTable_24),
      exec_shlI_1_lt hs10_2 (by omega), Nat.mul_assoc]
  let s11 := { s10.set 2 (some (auxCap * 32)) with fl := X86Int.Flags.undef, rip := 4201571 }
  have hr11 : RegsOk s11 := regsOk_set hr10 (by decide) _
  have hs11_4 : s11.reg 4 = some sp := by
    simp only [s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 2 ≠ 4), List.getElem?_set_ne (by decide : 6 ≠ 4),
      List.getElem?_set_ne (by decide : 7 ≠ 4), List.getElem?_set_ne (by decide : 1 ≠ 4),
      List.getElem?_set_ne (by decide : 0 ≠ 4), List.getElem?_set_ne (by decide : 13 ≠ 4),
      List.getElem?_set_ne (by decide : 12 ≠ 4)]
    exact hsp
  have hea_sp : (sp + X86Env.W - 8) % X86Env.W = sp - 8 := by
    have : sp + X86Env.W - 8 = (sp - 8) + X86Env.W := by omega
    rw [this, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  obtain ⟨istk, m12, hwr12, hget12, hm12eq, hwf12, hstk12, hoth12⟩ :=
    wr64_region hmwf hstk hstkrw hstk_lo (by omega : sp - 8 + 8 ≤ stk.hi) (v := 4201576)
  have hwr12' : wr64 s11.mem ((sp + X86Env.W - 8) % X86Env.W) 4201576 = some m12 := by
    simpa [s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, hea_sp] using hwr12
  have h12 : X86Vm.step vmImage s11 =
      .ok { s11.set 4 (some (sp - 8)) with mem := m12, rip := 4201363 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201571, (.ofInt (.call 4201363)), 5) ∈ SporosVmBin.vmTable_24),
      exec_call hs11_4 hwr12']
    simp [s11, hea_sp]
  let s12 := { s11.set 4 (some (sp - 8)) with mem := m12, rip := 4201363 }
  have hr12' : RegsOk s12 := regsOk_set hr11 (by decide) _
  let stk1 := { stk with data := writeBytes stk.data (sp - 8 - stk.base) (bytes8 4201576) }
  have hstk1_mem : stk1 ∈ m12 := hstk12
  have hstk1_hi : stk1.hi = stk.hi := by simp [stk1, Region.hi, writeBytes_length]
  have hret12 : rd64 s12.mem (sp - 8) = some 4201576 := by
    have hr := rd64_of_writeBytes_same hwf12 hstk12 hstk_lo (by omega : sp - 8 + 8 ≤ stk.hi)
    simpa [s12, show (4201576 % X86Env.W) = 4201576 from by decide] using hr
  have haux_m12 : auxReg ∈ m12 := hoth12 auxReg haux (by rw [haux_base]; exact hstkne_aux.symm)
  have hctx_m12 : ctx ∈ m12 := hoth12 ctx hctx (by rw [hctx_base]; exact hstkne_ctx.symm)
  obtain ⟨idx, hfind12⟩ := mem_find_of_mem m12 auxReg auxBase 0 hwf12.2 haux_m12
    (by simp [haux_base]) (by simp [Region.hi, haux_base, haux_eq]; omega)
  have hfind12' : m12.find auxBase 0 = some (idx, auxReg) := by simpa using hfind12
  have hget_idx := (mem_find_idx_lt m12 auxBase 0 idx auxReg hfind12').2.2
  simp only [Nat.sub_zero] at hget_idx
  have hpos12 : ∀ u ∈ m12, 0 < u.data.length := by
    intro u hu
    rw [hm12eq] at hu
    rcases List.mem_or_eq_of_mem_set hu with hu0 | rfl
    · exact hpos u hu0
    · simpa [writeBytes_length] using hpos stk hstk
  have hpg_16 : X86Env.pages (auxCap * 16) = auxCap * 16 := pages_of_dvd haux_pg
  have haux_pg32 : (auxCap * 32) % X86Env.page = 0 := by
    have : auxCap * 32 = (auxCap * 16) * 2 := by omega
    rw [this, Nat.mul_mod, haux_pg]; decide
  have hpg_32 : X86Env.pages (auxCap * 32) = auxCap * 32 := pages_of_dvd haux_pg32
  have hfree12 : X86Env.Mem.free (m12.eraseIdx idx) a (X86Env.pages (auxCap * 32)) = true := by
    rw [hpg_32]
    refine mem_eraseIdx_free hwf12 hget_idx hpos12 ?_
    intro u hu hub
    have hne_aux : u.base ≠ auxBase := by rw [← haux_base]; exact hub
    rw [hm12eq] at hu
    rcases List.mem_or_eq_of_mem_set hu with hu0 | rfl
    · exact hfree u hu0 hne_aux
    · simp only [Region.hi, writeBytes_length]
      exact hfree stk hstk hstkne_aux
  have hs12_4 : s12.reg 4 = some (sp - 8) := by
    simp only [s12, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr11.1]; decide)]
  have hs12_7 : s12.reg 7 = some auxBase := by
    simp only [s12, s11, s10, s9, s8, s7, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 4 ≠ 7), List.getElem?_set_ne (by decide : 2 ≠ 7),
      List.getElem?_set_ne (by decide : 6 ≠ 7)]
    rw [getElem?_set_getD_self (by rw [hr6'.1]; decide)]
  have hs12_6 : s12.reg 6 = some (auxCap * 16) := by
    simp only [s12, s11, s10, s9, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 4 ≠ 6), List.getElem?_set_ne (by decide : 2 ≠ 6)]
    rw [getElem?_set_getD_self (by rw [hr8.1]; decide)]
  have hs12_2 : s12.reg 2 = some (auxCap * 32) := by
    simp only [s12, s11, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 4 ≠ 2)]
    rw [getElem?_set_getD_self (by rw [hr10.1]; decide)]
  obtain ⟨s13, hrm, hrip13, hk13, hmem13, hm13wf, hr13', hs13_0, hs13_4, hoth13⟩ :=
    fn_mremap_ok_spec (s := s12) (sp := sp - 8) (retAddr := 4201576)
      (oldPtr := auxBase) (oldLen := auxCap * 16) (newLen := auxCap * 32)
      (idx := idx) (a := a) (oldReg := auxReg) (stk := stk1)
      rfl hr12' hs12_4 hstk1_mem (by simp [stk1]; exact hstkne_aux) hstk_lo (by rw [hstk1_hi]; omega)
      hret12 hwf12 hs12_7 hs12_6 hs12_2 hfind12' haux_base (by rw [hpg_16, haux_eq]) haux_rw
      (by rw [hpg_32, haux_eq]; omega) hor ha_pg ha_pos (by rw [hpg_32]; exact ha_lim) hfree12
  have hm13_eq_rep : X86Env.pages (auxCap * 32) - auxReg.data.length = auxCap * 16 := by
    rw [hpg_32, haux_eq]; omega
  let auxNew : Region := ⟨a, auxReg.data ++ List.replicate (auxCap * 16) 0, true⟩
  have hauxNew_m13 : auxNew ∈ s13.mem := by
    rw [hmem13]
    refine List.mem_append_right _ ?_
    simp [auxNew, hm13_eq_rep]
  have hctx_m13 : ctx ∈ s13.mem := by
    rw [hmem13]
    exact List.mem_append_left _ (mem_eraseIdx_other hget_idx hctx_m12 (by rw [haux_base, hctx_base]; exact hne.symm))
  have hstk1_m13 : stk1 ∈ s13.mem := by
    rw [hmem13]
    exact List.mem_append_left _ (mem_eraseIdx_other hget_idx hstk1_mem (by simp [stk1, haux_base]; exact hstkne_aux))
  have hs13_4' : s13.reg 4 = some sp := by
    have : (sp - 8 + 8) % X86Env.W = sp := by rw [Nat.sub_add_cancel h8, Nat.mod_eq_of_lt hspW]
    rwa [this] at hs13_4
  have hs13_15 : s13.reg 15 = some ctxBase := by
    rw [hoth13 15 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    simp only [s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 4 ≠ 15), List.getElem?_set_ne (by decide : 2 ≠ 15),
      List.getElem?_set_ne (by decide : 6 ≠ 15), List.getElem?_set_ne (by decide : 7 ≠ 15),
      List.getElem?_set_ne (by decide : 1 ≠ 15), List.getElem?_set_ne (by decide : 0 ≠ 15),
      List.getElem?_set_ne (by decide : 13 ≠ 15), List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hs13_12 : s13.reg 12 = some w := by
    rw [hoth13 12 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    simp only [s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 4 ≠ 12), List.getElem?_set_ne (by decide : 2 ≠ 12),
      List.getElem?_set_ne (by decide : 6 ≠ 12), List.getElem?_set_ne (by decide : 7 ≠ 12),
      List.getElem?_set_ne (by decide : 1 ≠ 12), List.getElem?_set_ne (by decide : 0 ≠ 12),
      List.getElem?_set_ne (by decide : 13 ≠ 12)]
    rw [getElem?_set_getD_self (by rw [hregs.1]; decide)]
  have hs13_13 : s13.reg 13 = some p := by
    rw [hoth13 13 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    simp only [s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 4 ≠ 13), List.getElem?_set_ne (by decide : 2 ≠ 13),
      List.getElem?_set_ne (by decide : 6 ≠ 13), List.getElem?_set_ne (by decide : 7 ≠ 13),
      List.getElem?_set_ne (by decide : 1 ≠ 13), List.getElem?_set_ne (by decide : 0 ≠ 13)]
    rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  have hm13wf' : s13.mem.wf := by rw [hmem13]; exact hm13wf
  obtain ⟨_, m14, hwr14, _, _, hwf14, hctx14, hoth14⟩ :=
    wr64_region hm13wf' hctx_m13 hctx_rw (a := ctxBase + 24) (v := a)
      (by rw [hctx_base]; omega) (by simp [Region.hi, hctx_base]; omega)
  have hwr14' : wr64 s13.mem (ea ctxBase 24) a = some m14 := by simpa [hea_c24] using hwr14
  have h14 : X86Vm.step vmImage s13 = .ok { s13 with mem := m14, rip := 4201580 } := by
    rw [step_at_chunk 24 hrip13 rfl (by decide : (4201576, (.ofInt (.st 15 24 0)), 4) ∈ SporosVmBin.vmTable_24),
      exec_st hs13_15 hs13_0 hwr14', hrip13]
  let s14 := { s13 with mem := m14, rip := 4201580 }
  have hr14 : RegsOk s14 := hr13'
  have hs14_15 : s14.reg 15 = some ctxBase := hs13_15
  let ctx14 := { ctx with data := writeBytes ctx.data (ctxBase + 24 - ctx.base) (bytes8 a) }
  have hctx14_hc40 : leVal ((ctx14.data.drop 40).take 8) = auxCap := by
    simp only [ctx14, hctx_base, Nat.add_sub_cancel_left]
    rw [writeBytes_drop_take_disjoint ctx.data 24 40 8 (bytes8 a) (by right; rw [bytes8_length]; omega) (by omega)]
    exact hc40
  have hrd14_c40 : rd64 s14.mem (ea ctxBase 40) = some auxCap := by
    rw [hea_c40]
    exact rd64_of_mem hwf14 hctx14 (by simp [hctx_base])
      (by simp [Region.hi, writeBytes_length, hctx_base]; omega)
      (by simpa [ctx14, hctx_base] using hctx14_hc40)
  have h15 : X86Vm.step vmImage s14 = .ok { s14.set 1 (some auxCap) with rip := 4201584 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201580, (.ofInt (.ld 1 15 40)), 4) ∈ SporosVmBin.vmTable_24),
      exec_ld hs14_15 hrd14_c40]
  let s15 := { s14.set 1 (some auxCap) with rip := 4201584 }
  have hr15' : RegsOk s15 := regsOk_set hr14 (by decide) auxCap
  have hs15_1 : s15.reg 1 = some auxCap := by
    simp only [s15, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr14.1]; decide)]
  have h16 : X86Vm.step vmImage s15 =
      .ok { s15.set 1 (some (auxCap * 2)) with fl := X86Int.Flags.undef, rip := 4201588 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201584, (.ofInt (.shlI 1 1)), 4) ∈ SporosVmBin.vmTable_24),
      exec_shlI_1_lt hs15_1 (by omega)]
  let s16 := { s15.set 1 (some (auxCap * 2)) with fl := X86Int.Flags.undef, rip := 4201588 }
  have hr16 : RegsOk s16 := regsOk_set hr15' (by decide) _
  have hs16_15 : s16.reg 15 = some ctxBase := by
    simp only [s16, s15, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 15)]
    exact hs14_15
  have hs16_1 : s16.reg 1 = some (auxCap * 2) := by
    simp only [s16, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr15'.1]; decide)]
  obtain ⟨_, m17, hwr17, _, _, hwf17, hctx17, hoth17⟩ :=
    wr64_region hwf14 hctx14 hctx_rw (a := ctxBase + 40) (v := auxCap * 2)
      (by simp [hctx_base])
      (by simp [Region.hi, writeBytes_length, hctx_base]; omega)
  have hwr17' : wr64 s16.mem (ea ctxBase 40) (auxCap * 2) = some m17 := by simpa [s16, s15, hea_c40] using hwr17
  have h17 : X86Vm.step vmImage s16 = .ok { s16 with mem := m17, rip := 4201592 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201588, (.ofInt (.st 15 40 1)), 4) ∈ SporosVmBin.vmTable_24),
      exec_st hs16_15 hs16_1 hwr17']
  let s17 := { s16 with mem := m17, rip := 4201592 }
  have hr17 : RegsOk s17 := hr16
  have hauxNew_m14 : auxNew ∈ m14 := hoth14 auxNew hauxNew_m13 (by simp [auxNew, hctx_base]; exact hane)
  have hauxNew_m17 : auxNew ∈ m17 := hoth17 auxNew hauxNew_m14 (by simp [auxNew, hctx_base]; exact hane)
  have hstk1_m14 : stk1 ∈ m14 := hoth14 stk1 hstk1_m13 (by simp [stk1, hctx_base]; exact hstkne_ctx)
  have hstk1_m17 : stk1 ∈ m17 := hoth17 stk1 hstk1_m14 (by simp [stk1, hctx_base]; exact hstkne_ctx)
  let ctx17 := { ctx14 with data := writeBytes ctx14.data (ctxBase + 40 - ctx14.base) (bytes8 (auxCap * 2)) }
  have hctx17_hc24 : leVal ((ctx17.data.drop 24).take 8) = a := by
    simp only [ctx17, ctx14, hctx_base, Nat.add_sub_cancel_left]
    rw [writeBytes_drop_take_disjoint _ 40 24 8 (bytes8 (auxCap * 2)) (by left; omega) (by rw [writeBytes_length]; omega)]
    rw [leVal_writeBytes_same _ 24 a (by omega)]
    exact Nat.mod_eq_of_lt (by have := vm_lim_lt_W; omega)
  have hctx17_hc32 : leVal ((ctx17.data.drop 32).take 8) = auxLen := by
    simp only [ctx17, ctx14, hctx_base, Nat.add_sub_cancel_left]
    rw [writeBytes_drop_take_disjoint _ 40 32 8 (bytes8 (auxCap * 2)) (by left; omega) (by rw [writeBytes_length]; omega)]
    rw [writeBytes_drop_take_disjoint ctx.data 24 32 8 (bytes8 a) (by right; rw [bytes8_length]; omega) (by omega)]
    exact hc32
  have hret_stk1 : rd64 s17.mem sp = some retAddr := by
    have hstk1_drop : (stk1.data.drop (sp - stk1.base)).take 8 = (stk.data.drop (sp - stk.base)).take 8 := by
      simp only [stk1]
      apply writeBytes_drop_take_disjoint
      · right; rw [bytes8_length]; omega
      · simp [Region.hi] at hstk_hi; omega
    obtain ⟨v, hv⟩ := show ∃ v, leVal ((stk.data.drop (sp - stk.base)).take 8) = v from ⟨_, rfl⟩
    have hold := rd64_of_mem hmwf hstk (by omega) hstk_hi hv
    rw [hret, Option.some.injEq] at hold
    subst hold
    exact rd64_of_mem hwf17 hstk1_m17 (by simp [stk1]; omega)
      (by simp [stk1, Region.hi, writeBytes_length] at hstk_hi ⊢; omega) (by rw [hstk1_drop, hv])
  have hauxCapW16 : auxCap * 16 ≤ 2 ^ 47 := by omega
  have hctx17_hi : ctxSize ≤ ctx17.data.length := by
    simp [ctx17, ctx14, writeBytes_length]; exact hctx_len
  have hs17_4 : s17.reg 4 = some sp := by
    simp only [s17, s16, s15, s14, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 4)]
    exact hs13_4'
  have hs17_12 : s17.reg 12 = some w := by
    simp only [s17, s16, s15, s14, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 12)]
    exact hs13_12
  have hs17_13 : s17.reg 13 = some p := by
    simp only [s17, s16, s15, s14, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 13)]
    exact hs13_13
  obtain ⟨m', sf, hreach_join, hripf, hkf, hmemf, hwf', hregsf, hspf, hauxf, hctxf, hothf, hregoth⟩ :=
    fn_aux_push_join_spec (s := s17) (sp := sp) (retAddr := retAddr) (ctxBase := ctxBase)
      (auxBase := a) (auxLen := auxLen) (w := w) (p := p) (ctx := ctx17) (auxReg := auxNew) (stk := stk1)
      rfl hr17 hs17_4 hs16_15 hs17_12 hs17_13
      (by simpa [s17, ctx17, ctx14] using hctx17) (by simp [ctx17, ctx14, hctx_base]) hctx_rw hctx17_hi
      (by simpa [s17] using hauxNew_m17) rfl rfl hane
      (by simpa [s17] using hstk1_m17) (by simp [stk1]; exact hastk.symm)
      (by simp [stk1]; exact hstkne_ctx) (by simp [stk1]; omega)
      (by simp [stk1, Region.hi, writeBytes_length] at hstk_hi ⊢; omega)
      (by simp [auxNew, List.length_append, List.length_replicate, haux_eq]; omega)
      hctxW (by have := vm_lim_lt_W; omega) (by omega) hwf17 hret_stk1 hctx17_hc24 hctx17_hc32
  have hstk1_m' : stk1 ∈ m' := by
    apply hothf stk1 (by simpa [s17] using hstk1_m17)
    · simp [stk1]; exact hastk.symm
    · simp [stk1]; exact hstkne_ctx
  refine ⟨m', sf, stk1,
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
      (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step_trans h08
        (Reaches.step_trans h09 (Reaches.step_trans h10 (Reaches.step_trans h11 (Reaches.step_trans h12
          (Reaches.trans hrm (Reaches.step_trans h14 (Reaches.step_trans h15 (Reaches.step_trans h16
            (Reaches.step_trans h17 hreach_join)))))))))))))))),
    hripf, by simpa [s17, s16, s15, s14, hk13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1] using hkf, hmemf, hwf', hregsf, hspf,
    by simpa [auxNew] using hauxf,
    by simpa [ctx17, ctx14, hctx_base, Nat.add_sub_cancel_left] using hctxf,
    hstk1_m', rfl, hstkrw, writeBytes_length _ _ _, ?_, ?_, ?_⟩
  · intro j hj
    apply writeBytes_getElem?_disjoint
    rw [bytes8_length]
    rcases hj with h | h
    · exact Or.inl h
    · exact Or.inr (by omega)
  · intro u hu huaux huctx hustk
    have hu12 : u ∈ m12 := hoth12 u hu hustk
    have hu13 : u ∈ s13.mem := by
      rw [hmem13]
      exact List.mem_append_left _ (mem_eraseIdx_other hget_idx hu12 (by rw [haux_base]; exact huaux))
    have hu14 : u ∈ m14 := hoth14 u hu13 (by rw [hctx_base]; exact huctx)
    have hu17 : u ∈ m17 := hoth17 u hu14 (by simp [hctx_base]; exact huctx)
    have hua : u.base ≠ a := by
      intro hba
      have hupos := hpos u hu
      rcases hfree u hu huaux with h1 | h2
      · omega
      · simp [Region.hi] at h2; omega
    exact hothf u (by simpa [s17] using hu17) hua huctx
  · intro r h0 h1 h2 h4 h6 h7 h8 h9 h10' h11' h12' h13'
    have h17_13 : s17.reg r = s13.reg r := by
      simp only [s17, s16, s15, s14, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne h1.symm]
    rw [hregoth r h0 h1 h2 h4, h17_13,
      hoth13 r h0 h1 h2 h4 h6 h7 h8 h9 h10' h11']
    simp only [s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne h4.symm, List.getElem?_set_ne h2.symm, List.getElem?_set_ne h6.symm,
      List.getElem?_set_ne h7.symm, List.getElem?_set_ne h1.symm, List.getElem?_set_ne h0.symm,
      List.getElem?_set_ne h13'.symm, List.getElem?_set_ne h12'.symm]

theorem fn_aux_push_nomem_spec {s : X86Vm.St} {sp ctxBase auxBase auxLen auxCap w p : Nat}
    {ctx auxReg stk : Region}
    (hrip : s.rip = SporosVmBin.fn_aux_push)
    (hregs : RegsOk s)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr7 : s.reg 7 = some w) (hr6 : s.reg 6 = some p)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_len : ctxSize ≤ ctx.data.length)
    (haux : auxReg ∈ s.mem) (haux_base : auxReg.base = auxBase) (haux_rw : auxReg.rw = true)
    (hstk : stk ∈ s.mem) (hstkrw : stk.rw = true) (hstkne_aux : stk.base ≠ auxBase)
    (h16 : 16 ≤ sp) (hspW : sp < X86Env.W)
    (hstk_lo : stk.base ≤ sp - 16) (hstk_hi : sp ≤ stk.hi)
    (hge : auxCap ≤ auxLen) (hle : auxLen ≤ auxCap) (hcap_pos : 0 < auxCap)
    (haux_eq : auxReg.data.length = auxCap * 16)
    (haux_pg : (auxCap * 16) % X86Env.page = 0)
    (hauxCapW : auxCap * 32 < X86Env.W)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W)
    (hmwf : s.mem.wf)
    (hc24 : leVal ((ctx.data.drop 24).take 8) = auxBase)
    (hc32 : leVal ((ctx.data.drop 32).take 8) = auxLen)
    (hc40 : leVal ((ctx.data.drop 40).take 8) = auxCap)
    (hor : (s.k.next).1 = some .nomem) :
    Ends vmImage s (.exit 2) := by
  have hcs : ctxSize = 4096 := rfl
  have hea_c24 : ea ctxBase 24 = ctxBase + 24 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 24 < X86Env.W)]
  have hea_c32 : ea ctxBase 32 = ctxBase + 32 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 32 < X86Env.W)]
  have hea_c40 : ea ctxBase 40 = ctxBase + 40 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 40 < X86Env.W)]
  have h01 : X86Vm.step vmImage s = .ok { s.set 12 (some w) with rip := 4201537 } := by
    rw [step_at_chunk 24 hrip rfl (by decide : (SporosVmBin.fn_aux_push, (.ofInt (.mov 12 7)), 3) ∈ SporosVmBin.vmTable_24),
      exec_mov hr7, hrip]
    rfl
  let s1 := { s.set 12 (some w) with rip := 4201537 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) w
  have hs1_6 : s1.reg 6 = some p := by
    simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 12 ≠ 6)]
    exact hr6
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 13 (some p) with rip := 4201540 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201537, (.ofInt (.mov 13 6)), 3) ∈ SporosVmBin.vmTable_24),
      exec_mov hs1_6]
  let s2 := { s1.set 13 (some p) with rip := 4201540 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) p
  have hs2_15 : s2.reg 15 = some ctxBase := by
    simp only [s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 13 ≠ 15), List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hrd_c32 : rd64 s2.mem (ea ctxBase 32) = some auxLen := by
    rw [hea_c32]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc32)
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 0 (some auxLen) with rip := 4201544 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201540, (.ofInt (.ld 0 15 32)), 4) ∈ SporosVmBin.vmTable_24),
      exec_ld hs2_15 hrd_c32]
  let s3 := { s2.set 0 (some auxLen) with rip := 4201544 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) auxLen
  have hs3_15 : s3.reg 15 = some ctxBase := by
    simp only [s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 15), List.getElem?_set_ne (by decide : 13 ≠ 15),
      List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hrd_c40 : rd64 s3.mem (ea ctxBase 40) = some auxCap := by
    rw [hea_c40]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc40)
  have h04 : X86Vm.step vmImage s3 = .ok { s3.set 1 (some auxCap) with rip := 4201548 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201544, (.ofInt (.ld 1 15 40)), 4) ∈ SporosVmBin.vmTable_24),
      exec_ld hs3_15 hrd_c40]
  let s4 := { s3.set 1 (some auxCap) with rip := 4201548 }
  have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) auxCap
  have hs4_0 : s4.reg 0 = some auxLen := by
    simp only [s4, s3, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0)]
    rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
  have hs4_1 : s4.reg 1 = some auxCap := by
    simp only [s4, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
  have h05 : X86Vm.step vmImage s4 =
      .ok { s4 with fl := (X86Int.aluF .sub auxLen auxCap).2, rip := 4201551 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201548, (.ofInt (.cmp 0 1)), 3) ∈ SporosVmBin.vmTable_24),
      exec_cmp hs4_0 hs4_1]
  let s5 := { s4 with fl := (X86Int.aluF .sub auxLen auxCap).2, rip := 4201551 }
  have hr5 : RegsOk s5 := hr4
  have hnotlt : ¬ auxLen < auxCap := by omega
  have hcb : X86Int.cond s5.fl 2 = some false := by
    dsimp [s5]; rw [cond_aluF_sub_b, decide_eq_false hnotlt]
  have h06 : X86Vm.step vmImage s5 = .ok { s5 with rip := 4201553 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201551, (.ofInt (.jcc 2 4201592)), 2) ∈ SporosVmBin.vmTable_24),
      exec_jcc hcb]
    rfl
  let s6 := { s5 with rip := 4201553 }
  have hr6' : RegsOk s6 := hr5
  have hs6_15 : s6.reg 15 = some ctxBase := by
    simp only [s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 15), List.getElem?_set_ne (by decide : 0 ≠ 15),
      List.getElem?_set_ne (by decide : 13 ≠ 15), List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hrd_c24 : rd64 s6.mem (ea ctxBase 24) = some auxBase := by
    rw [hea_c24]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc24)
  have h07 : X86Vm.step vmImage s6 = .ok { s6.set 7 (some auxBase) with rip := 4201557 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201553, (.ofInt (.ld 7 15 24)), 4) ∈ SporosVmBin.vmTable_24),
      exec_ld hs6_15 hrd_c24]
  let s7 := { s6.set 7 (some auxBase) with rip := 4201557 }
  have hr7' : RegsOk s7 := regsOk_set hr6' (by decide) auxBase
  have hs7_1 : s7.reg 1 = some auxCap := by
    simp only [s7, s6, s5, s4, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 7 ≠ 1)]
    rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
  have h08 : X86Vm.step vmImage s7 = .ok { s7.set 6 (some auxCap) with rip := 4201560 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201557, (.ofInt (.mov 6 1)), 3) ∈ SporosVmBin.vmTable_24),
      exec_mov hs7_1]
  let s8 := { s7.set 6 (some auxCap) with rip := 4201560 }
  have hr8 : RegsOk s8 := regsOk_set hr7' (by decide) auxCap
  have hs8_6 : s8.reg 6 = some auxCap := by
    simp only [s8, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr7'.1]; decide)]
  have h09 : X86Vm.step vmImage s8 =
      .ok { s8.set 6 (some (auxCap * 16)) with fl := X86Int.Flags.undef, rip := 4201564 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201560, (.ofInt (.shlI 6 4)), 4) ∈ SporosVmBin.vmTable_24),
      exec_shlI_4_lt hs8_6 (by omega)]
  let s9 := { s8.set 6 (some (auxCap * 16)) with fl := X86Int.Flags.undef, rip := 4201564 }
  have hr9 : RegsOk s9 := regsOk_set hr8 (by decide) _
  have hs9_6 : s9.reg 6 = some (auxCap * 16) := by
    simp only [s9, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr8.1]; decide)]
  have h10 : X86Vm.step vmImage s9 = .ok { s9.set 2 (some (auxCap * 16)) with rip := 4201567 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201564, (.ofInt (.mov 2 6)), 3) ∈ SporosVmBin.vmTable_24),
      exec_mov hs9_6]
  let s10 := { s9.set 2 (some (auxCap * 16)) with rip := 4201567 }
  have hr10 : RegsOk s10 := regsOk_set hr9 (by decide) _
  have hs10_2 : s10.reg 2 = some (auxCap * 16) := by
    simp only [s10, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr9.1]; decide)]
  have h11 : X86Vm.step vmImage s10 =
      .ok { s10.set 2 (some (auxCap * 32)) with fl := X86Int.Flags.undef, rip := 4201571 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201567, (.ofInt (.shlI 2 1)), 4) ∈ SporosVmBin.vmTable_24),
      exec_shlI_1_lt hs10_2 (by omega), Nat.mul_assoc]
  let s11 := { s10.set 2 (some (auxCap * 32)) with fl := X86Int.Flags.undef, rip := 4201571 }
  have hr11 : RegsOk s11 := regsOk_set hr10 (by decide) _
  have hs11_4 : s11.reg 4 = some sp := by
    simp only [s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 2 ≠ 4), List.getElem?_set_ne (by decide : 6 ≠ 4),
      List.getElem?_set_ne (by decide : 7 ≠ 4), List.getElem?_set_ne (by decide : 1 ≠ 4),
      List.getElem?_set_ne (by decide : 0 ≠ 4), List.getElem?_set_ne (by decide : 13 ≠ 4),
      List.getElem?_set_ne (by decide : 12 ≠ 4)]
    exact hsp
  have hea_sp : (sp + X86Env.W - 8) % X86Env.W = sp - 8 := by
    have : sp + X86Env.W - 8 = (sp - 8) + X86Env.W := by omega
    rw [this, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  obtain ⟨istk, m12, hwr12, _, _, hwf12, hstk12, hoth12⟩ :=
    wr64_region hmwf hstk hstkrw (by omega : stk.base ≤ sp - 8) (by omega : sp - 8 + 8 ≤ stk.hi) (v := 4201576)
  have hwr12' : wr64 s11.mem ((sp + X86Env.W - 8) % X86Env.W) 4201576 = some m12 := by
    simpa [s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, hea_sp] using hwr12
  have h12 : X86Vm.step vmImage s11 =
      .ok { s11.set 4 (some (sp - 8)) with mem := m12, rip := 4201363 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201571, (.ofInt (.call 4201363)), 5) ∈ SporosVmBin.vmTable_24),
      exec_call hs11_4 hwr12']
    simp [s11, hea_sp]
  let s12 := { s11.set 4 (some (sp - 8)) with mem := m12, rip := 4201363 }
  have hr12' : RegsOk s12 := regsOk_set hr11 (by decide) _
  let stk1 := { stk with data := writeBytes stk.data (sp - 8 - stk.base) (bytes8 4201576) }
  have hstk1_hi : stk1.hi = stk.hi := by simp [stk1, Region.hi, writeBytes_length]
  have haux_m12 : auxReg ∈ m12 := hoth12 auxReg haux (by rw [haux_base]; exact hstkne_aux.symm)
  obtain ⟨idx, hfind12⟩ := mem_find_of_mem m12 auxReg auxBase 0 hwf12.2 haux_m12
    (by simp [haux_base]) (by simp [Region.hi, haux_base, haux_eq]; omega)
  have hfind12' : m12.find auxBase 0 = some (idx, auxReg) := by simpa using hfind12
  have hpg_16 : X86Env.pages (auxCap * 16) = auxCap * 16 := pages_of_dvd haux_pg
  have haux_pg32 : (auxCap * 32) % X86Env.page = 0 := by
    have : auxCap * 32 = (auxCap * 16) * 2 := by omega
    rw [this, Nat.mul_mod, haux_pg]; decide
  have hpg_32 : X86Env.pages (auxCap * 32) = auxCap * 32 := pages_of_dvd haux_pg32
  have hs12_4 : s12.reg 4 = some (sp - 8) := by
    simp only [s12, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr11.1]; decide)]
  have hs12_7 : s12.reg 7 = some auxBase := by
    simp only [s12, s11, s10, s9, s8, s7, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 4 ≠ 7), List.getElem?_set_ne (by decide : 2 ≠ 7),
      List.getElem?_set_ne (by decide : 6 ≠ 7)]
    rw [getElem?_set_getD_self (by rw [hr6'.1]; decide)]
  have hs12_6 : s12.reg 6 = some (auxCap * 16) := by
    simp only [s12, s11, s10, s9, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 4 ≠ 6), List.getElem?_set_ne (by decide : 2 ≠ 6)]
    rw [getElem?_set_getD_self (by rw [hr8.1]; decide)]
  have hs12_2 : s12.reg 2 = some (auxCap * 32) := by
    simp only [s12, s11, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 4 ≠ 2)]
    rw [getElem?_set_getD_self (by rw [hr10.1]; decide)]
  have h_exit := fn_mremap_nomem_spec (s := s12) (sp := sp - 8)
    (oldPtr := auxBase) (oldLen := auxCap * 16) (newLen := auxCap * 32)
    (idx := idx) (oldReg := auxReg) (stk := stk1)
    rfl hr12' hs12_4 (by omega) (by omega) hwf12 hstk12 hstkrw
    (by simp [stk1]; omega) (by rw [hstk1_hi]; omega)
    hs12_7 hs12_6 hs12_2 hfind12' haux_base (by rw [hpg_16, haux_eq]) haux_rw
    (by rw [hpg_32, haux_eq]; omega) hor
  exact Ends.step_trans h01 (Ends.step_trans h02 (Ends.step_trans h03 (Ends.step_trans h04
    (Ends.step_trans h05 (Ends.step_trans h06 (Ends.step_trans h07 (Ends.step_trans h08
      (Ends.step_trans h09 (Ends.step_trans h10 (Ends.step_trans h11 (Ends.step_trans h12 h_exit)))))))))))

theorem fn_out_push_byte_join_spec {s : X86Vm.St} {sp retAddr ctxBase outBase outLen b : Nat}
    {ctx outReg stk : Region}
    (hrip : s.rip = 4201694)
    (hregs : RegsOk s)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr12 : s.reg 12 = some b)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_rw : ctx.rw = true)
    (hctx_len : ctxSize ≤ ctx.data.length)
    (hout : outReg ∈ s.mem) (hout_base : outReg.base = outBase) (hout_rw : outReg.rw = true)
    (hne : outBase ≠ ctxBase)
    (hstk : stk ∈ s.mem) (hstkne_out : stk.base ≠ outBase) (hstkne_ctx : stk.base ≠ ctxBase)
    (hstk_lo : stk.base ≤ sp) (hstk_hi : sp + 8 ≤ stk.hi)
    (hout_hi : outLen + 1 ≤ outReg.data.length)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W) (houtW : outBase + outLen + 1 ≤ X86Env.W)
    (houtLenW : outLen + 1 < X86Env.W)
    (hmwf : s.mem.wf)
    (hret : rd64 s.mem sp = some retAddr)
    (hc200 : leVal ((ctx.data.drop 200).take 8) = outBase)
    (hc208 : leVal ((ctx.data.drop 208).take 8) = outLen) :
    let outData' := writeBytes outReg.data outLen [b % 256]
    let ctxData' := writeBytes ctx.data 208 (bytes8 (outLen + 1))
    ∃ m' s', Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.k = s.k ∧ s'.mem = m' ∧ m'.wf ∧ RegsOk s' ∧
      s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      { outReg with data := outData' } ∈ m' ∧
      { ctx with data := ctxData' } ∈ m' ∧
      (∀ u ∈ s.mem, u.base ≠ outBase → u.base ≠ ctxBase → u ∈ m') ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 4 → s'.reg r = s.reg r) := by
  have hcs : ctxSize = 4096 := rfl
  have hea_c200 : ea ctxBase 200 = ctxBase + 200 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 200 < X86Env.W)]
  have hea_c208 : ea ctxBase 208 = ctxBase + 208 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 208 < X86Env.W)]
  have hrd_c200 : rd64 s.mem (ea ctxBase 200) = some outBase := by
    rw [hea_c200]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc200)
  have h01 : X86Vm.step vmImage s = .ok { s.set 0 (some outBase) with rip := 4201701 } := by
    rw [step_at_chunk 24 hrip rfl (by decide : (4201694, (.ofInt (.ld 0 15 200)), 7) ∈ SporosVmBin.vmTable_24),
      exec_ld hr15 hrd_c200, hrip]
  let s1 := { s.set 0 (some outBase) with rip := 4201701 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) outBase
  have hs1_15 : s1.reg 15 = some ctxBase := by
    simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 15)]
    exact hr15
  have hrd_c208 : rd64 s1.mem (ea ctxBase 208) = some outLen := by
    rw [hea_c208]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc208)
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 1 (some outLen) with rip := 4201708 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201701, (.ofInt (.ld 1 15 208)), 7) ∈ SporosVmBin.vmTable_24),
      exec_ld hs1_15 hrd_c208]
  let s2 := { s1.set 1 (some outLen) with rip := 4201708 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) outLen
  have hs2_0 : s2.reg 0 = some outBase := by
    simp only [s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0)]
    rw [getElem?_set_getD_self (by rw [hregs.1]; decide)]
  have hs2_1 : s2.reg 1 = some outLen := by
    simp only [s2, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  have hs2_12 : s2.reg 12 = some b := by
    simp only [s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 12), List.getElem?_set_ne (by decide : 0 ≠ 12)]
    exact hr12
  have hea_d0 : eaIdx outBase outLen 1 0 = outBase + outLen := by
    simp [eaIdx, Nat.mod_eq_of_lt (by omega : outBase + outLen < X86Env.W)]
  obtain ⟨i1, m3, hwr3, _, _, hwf3, hr3_out, hoth3⟩ :=
    wrMem_w8_region hmwf hout hout_rw (a := outBase + outLen) (v := b)
      (by rw [hout_base]; omega) (by simp [Region.hi, hout_base]; omega)
  have hwr3' : wrMem .w8 s2.mem (eaIdx outBase outLen 1 0) b = some m3 := by
    simpa [s2, s1, hea_d0] using hwr3
  have h03 : X86Vm.step vmImage s2 = .ok { s2 with mem := m3, rip := 4201712 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201708, (.stIdx .w8 0 1 1 0 12), 4) ∈ SporosVmBin.vmTable_24),
      exec_stIdx hs2_0 hs2_1 hs2_12 hwr3']
  let s3 := { s2 with mem := m3, rip := 4201712 }
  have hr3 : RegsOk s3 := hr2
  have hs3_1 : s3.reg 1 = some outLen := hs2_1
  have h04 : X86Vm.step vmImage s3 =
      .ok { s3.set 1 (some ((outLen + 1) % X86Env.W)) with
            fl := (X86Int.aluF .add outLen 1).2, rip := 4201716 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201712, (.ofInt (.aluI .add 1 1)), 4) ∈ SporosVmBin.vmTable_24),
      exec_aluI_add_lt hs3_1 (by decide)]
  let s4 := { s3.set 1 (some ((outLen + 1) % X86Env.W)) with
                fl := (X86Int.aluF .add outLen 1).2, rip := 4201716 }
  have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) _
  have hs4_15 : s4.reg 15 = some ctxBase := by
    simp only [s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 15), List.getElem?_set_ne (by decide : 0 ≠ 15)]
    exact hr15
  have hs4_1 : s4.reg 1 = some ((outLen + 1) % X86Env.W) := by
    simp only [s4, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
  have hctx_m3 : ctx ∈ m3 := hoth3 ctx hctx (by rw [hctx_base, hout_base]; exact Ne.symm hne)
  obtain ⟨i2, m4, hwr4, _, _, hwf4, hr4_ctx, hoth4⟩ :=
    wr64_region hwf3 hctx_m3 hctx_rw (a := ctxBase + 208) (v := (outLen + 1) % X86Env.W)
      (by rw [hctx_base]; omega) (by simp [Region.hi, hctx_base]; omega)
  have hwr4' : wr64 s4.mem (ea ctxBase 208) ((outLen + 1) % X86Env.W) = some m4 := by
    simpa [s4, s3, hea_c208] using hwr4
  have h05 : X86Vm.step vmImage s4 = .ok { s4 with mem := m4, rip := 4201723 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201716, (.ofInt (.st 15 208 1)), 7) ∈ SporosVmBin.vmTable_24),
      exec_st hs4_15 hs4_1 hwr4']
  let s5 := { s4 with mem := m4, rip := 4201723 }
  have hr5 : RegsOk s5 := hr4
  have hs5_4 : s5.reg 4 = some sp := by
    simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 4), List.getElem?_set_ne (by decide : 0 ≠ 4)]
    exact hsp
  have hstk_m3 : stk ∈ m3 := hoth3 stk hstk (by rw [hout_base]; exact hstkne_out)
  have hstk_m4 : stk ∈ m4 := hoth4 stk hstk_m3 (by rw [hctx_base]; exact hstkne_ctx)
  have hret_m4 : rd64 s5.mem sp = some retAddr :=
    rd64_of_mem_transfer hmwf hwf4 hstk hstk_m4 hstk_lo hstk_hi hret
  have h06 : X86Vm.step vmImage s5 =
      .ok { s5.set 4 (some ((sp + 8) % X86Env.W)) with rip := retAddr } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201723, (.ofInt .ret), 1) ∈ SporosVmBin.vmTable_24),
      exec_ret hs5_4 hret_m4]
  let s6 := { s5.set 4 (some ((sp + 8) % X86Env.W)) with rip := retAddr }
  refine ⟨m4, s6,
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
      (Reaches.step_trans h05 (Reaches.step h06))))),
    rfl, rfl, rfl, hwf4, regsOk_set hr5 (by decide) _, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [s6, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr5.1]; decide)]
  · have hout1_m4 := hoth4 _ hr3_out (by simp [hout_base, hctx_base]; exact hne)
    simpa [hout_base, show outBase + outLen - outBase = outLen from by omega] using hout1_m4
  · simpa [hctx_base, show ctxBase + 208 - ctxBase = 208 from by omega,
      Nat.mod_eq_of_lt (by omega : outLen + 1 < X86Env.W)] using hr4_ctx
  · intro u hu huout huctx
    have hu3 := hoth3 u hu (by rw [hout_base]; exact huout)
    exact hoth4 u hu3 (by rw [hctx_base]; exact huctx)
  · intro r h0 h1 h4
    simp only [s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne h4.symm, List.getElem?_set_ne h1.symm, List.getElem?_set_ne h0.symm]

theorem fn_out_push_byte_nogrow_spec {s : X86Vm.St} {sp retAddr ctxBase outBase outLen outCap b : Nat}
    {ctx outReg stk : Region}
    (hrip : s.rip = SporosVmBin.fn_out_push_byte)
    (hregs : RegsOk s)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr7 : s.reg 7 = some b)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_rw : ctx.rw = true)
    (hctx_len : ctxSize ≤ ctx.data.length)
    (hout : outReg ∈ s.mem) (hout_base : outReg.base = outBase) (hout_rw : outReg.rw = true)
    (hne : outBase ≠ ctxBase)
    (hstk : stk ∈ s.mem) (hstkne_out : stk.base ≠ outBase) (hstkne_ctx : stk.base ≠ ctxBase)
    (hstk_lo : stk.base ≤ sp) (hstk_hi : sp + 8 ≤ stk.hi)
    (hlt : outLen < outCap)
    (hout_hi : outCap ≤ outReg.data.length)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W) (houtW : outBase + outCap ≤ X86Env.W)
    (houtCapW : outCap < X86Env.W)
    (hmwf : s.mem.wf)
    (hret : rd64 s.mem sp = some retAddr)
    (hc200 : leVal ((ctx.data.drop 200).take 8) = outBase)
    (hc208 : leVal ((ctx.data.drop 208).take 8) = outLen)
    (hc216 : leVal ((ctx.data.drop 216).take 8) = outCap) :
    let outData' := writeBytes outReg.data outLen [b % 256]
    let ctxData' := writeBytes ctx.data 208 (bytes8 (outLen + 1))
    ∃ m' s', Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.k = s.k ∧ s'.mem = m' ∧ m'.wf ∧ RegsOk s' ∧
      s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      { outReg with data := outData' } ∈ m' ∧
      { ctx with data := ctxData' } ∈ m' ∧
      (∀ u ∈ s.mem, u.base ≠ outBase → u.base ≠ ctxBase → u ∈ m') ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 4 → r ≠ 12 → s'.reg r = s.reg r) := by
  have hcs : ctxSize = 4096 := rfl
  have hea_c208 : ea ctxBase 208 = ctxBase + 208 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 208 < X86Env.W)]
  have hea_c216 : ea ctxBase 216 = ctxBase + 216 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 216 < X86Env.W)]
  have h01 : X86Vm.step vmImage s = .ok { s.set 12 (some b) with rip := 4201628 } := by
    rw [step_at_chunk 24 hrip rfl (by decide : (SporosVmBin.fn_out_push_byte, (.ofInt (.mov 12 7)), 3) ∈ SporosVmBin.vmTable_24),
      exec_mov hr7, hrip]
    rfl
  let s1 := { s.set 12 (some b) with rip := 4201628 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) b
  have hs1_15 : s1.reg 15 = some ctxBase := by
    simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hrd_c208 : rd64 s1.mem (ea ctxBase 208) = some outLen := by
    rw [hea_c208]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc208)
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 0 (some outLen) with rip := 4201635 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201628, (.ofInt (.ld 0 15 208)), 7) ∈ SporosVmBin.vmTable_24),
      exec_ld hs1_15 hrd_c208]
  let s2 := { s1.set 0 (some outLen) with rip := 4201635 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) outLen
  have hs2_15 : s2.reg 15 = some ctxBase := by
    simp only [s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 15), List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hrd_c216 : rd64 s2.mem (ea ctxBase 216) = some outCap := by
    rw [hea_c216]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc216)
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 1 (some outCap) with rip := 4201642 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201635, (.ofInt (.ld 1 15 216)), 7) ∈ SporosVmBin.vmTable_24),
      exec_ld hs2_15 hrd_c216]
  let s3 := { s2.set 1 (some outCap) with rip := 4201642 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) outCap
  have hs3_0 : s3.reg 0 = some outLen := by
    simp only [s3, s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0)]
    rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  have hs3_1 : s3.reg 1 = some outCap := by
    simp only [s3, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
  have h04 : X86Vm.step vmImage s3 =
      .ok { s3 with fl := (X86Int.aluF .sub outLen outCap).2, rip := 4201645 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201642, (.ofInt (.cmp 0 1)), 3) ∈ SporosVmBin.vmTable_24),
      exec_cmp hs3_0 hs3_1]
  let s4 := { s3 with fl := (X86Int.aluF .sub outLen outCap).2, rip := 4201645 }
  have hr4 : RegsOk s4 := hr3
  have hcb : X86Int.cond s4.fl 2 = some true := by
    dsimp [s4]; rw [cond_aluF_sub_b, decide_eq_true hlt]
  have h05 : X86Vm.step vmImage s4 = .ok { s4 with rip := 4201694 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201645, (.ofInt (.jcc 2 4201694)), 2) ∈ SporosVmBin.vmTable_24),
      exec_jcc hcb]
    rfl
  let s5 := { s4 with rip := 4201694 }
  have hr5 : RegsOk s5 := hr4
  have hs5_4 : s5.reg 4 = some sp := by
    simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 4), List.getElem?_set_ne (by decide : 0 ≠ 4),
      List.getElem?_set_ne (by decide : 12 ≠ 4)]
    exact hsp
  have hs5_15 : s5.reg 15 = some ctxBase := by
    simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 15), List.getElem?_set_ne (by decide : 0 ≠ 15),
      List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hs5_12 : s5.reg 12 = some b := by
    simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 12), List.getElem?_set_ne (by decide : 0 ≠ 12)]
    rw [getElem?_set_getD_self (by rw [hregs.1]; decide)]
  obtain ⟨m', sf, hreach, hripf, hkf, hmemf, hwf', hregsf, hspf, houtf, hctxf, hothf, hregoth⟩ :=
    fn_out_push_byte_join_spec (s := s5) (sp := sp) (retAddr := retAddr) (ctxBase := ctxBase)
      (outBase := outBase) (outLen := outLen) (b := b) (ctx := ctx) (outReg := outReg) (stk := stk)
      rfl hr5 hs5_4 hs5_15 hs5_12 hctx hctx_base hctx_rw hctx_len hout hout_base hout_rw hne
      hstk hstkne_out hstkne_ctx hstk_lo hstk_hi (by omega) hctxW (by omega) (by omega) hmwf
      hret hc200 hc208
  refine ⟨m', sf,
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
      (Reaches.step_trans h05 hreach)))),
    hripf, hkf, hmemf, hwf', hregsf, hspf, houtf, hctxf, hothf, ?_⟩
  intro r h0 h1 h4 h12
  rw [hregoth r h0 h1 h4]
  simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
    List.getElem?_set_ne h1.symm, List.getElem?_set_ne h0.symm, List.getElem?_set_ne h12.symm]

theorem fn_out_push_byte_grow_spec {s : X86Vm.St} {sp retAddr ctxBase outBase outLen outCap b a : Nat}
    {ctx outReg stk : Region}
    (hrip : s.rip = SporosVmBin.fn_out_push_byte)
    (hregs : RegsOk s)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr7 : s.reg 7 = some b)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_rw : ctx.rw = true)
    (hctx_len : ctxSize ≤ ctx.data.length)
    (hout : outReg ∈ s.mem) (hout_base : outReg.base = outBase) (hout_rw : outReg.rw = true)
    (hne : outBase ≠ ctxBase) (hane : a ≠ ctxBase) (hastk : a ≠ stk.base)
    (hstk : stk ∈ s.mem) (hstkrw : stk.rw = true) (hstkne_out : stk.base ≠ outBase) (hstkne_ctx : stk.base ≠ ctxBase)
    (h8 : 8 ≤ sp) (hspW : sp < X86Env.W)
    (hstk_lo : stk.base ≤ sp - 8) (hstk_hi : sp + 8 ≤ stk.hi)
    (hge : outCap ≤ outLen) (hle : outLen ≤ outCap) (hcap_pos : 0 < outCap)
    (hout_eq : outReg.data.length = outCap)
    (hout_pg : outCap % X86Env.page = 0)
    (houtCapW : outCap * 2 < X86Env.W)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W)
    (hmwf : s.mem.wf) (hpos : ∀ u ∈ s.mem, 0 < u.data.length)
    (hret : rd64 s.mem sp = some retAddr)
    (hc200 : leVal ((ctx.data.drop 200).take 8) = outBase)
    (hc208 : leVal ((ctx.data.drop 208).take 8) = outLen)
    (hc216 : leVal ((ctx.data.drop 216).take 8) = outCap)
    (hor : (s.k.next).1 = some (.at a))
    (ha_pg : a % X86Env.page = 0) (ha_pos : 0 < a) (ha_lim : a + outCap * 2 ≤ 2 ^ 47)
    (hfree : ∀ u ∈ s.mem, u.base ≠ outBase → a + outCap * 2 ≤ u.base ∨ u.hi ≤ a) :
    let outData' := writeBytes (outReg.data ++ List.replicate outCap 0) outLen [b % 256]
    let ctxData' := writeBytes (writeBytes (writeBytes ctx.data 200 (bytes8 a)) 216 (bytes8 (outCap * 2))) 208 (bytes8 (outLen + 1))
    ∃ m' s' stk', Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.k = (s.k.next).2 ∧ s'.mem = m' ∧ m'.wf ∧ RegsOk s' ∧
      s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      (⟨a, outData', true⟩ : Region) ∈ m' ∧
      { ctx with data := ctxData' } ∈ m' ∧
      stk' ∈ m' ∧ stk'.base = stk.base ∧ stk'.rw = true ∧ stk'.data.length = stk.data.length ∧
      (∀ j, j < sp - 8 - stk.base ∨ sp - stk.base ≤ j → stk'.data[j]? = stk.data[j]?) ∧
      (∀ u ∈ s.mem, u.base ≠ outBase → u.base ≠ ctxBase → u.base ≠ stk.base → u ∈ m') ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 4 → r ≠ 6 → r ≠ 7 → r ≠ 8 → r ≠ 9 → r ≠ 10 → r ≠ 11 → r ≠ 12 → s'.reg r = s.reg r) := by
  have hcs : ctxSize = 4096 := rfl
  have hea_c200 : ea ctxBase 200 = ctxBase + 200 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 200 < X86Env.W)]
  have hea_c208 : ea ctxBase 208 = ctxBase + 208 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 208 < X86Env.W)]
  have hea_c216 : ea ctxBase 216 = ctxBase + 216 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 216 < X86Env.W)]
  have h01 : X86Vm.step vmImage s = .ok { s.set 12 (some b) with rip := 4201628 } := by
    rw [step_at_chunk 24 hrip rfl (by decide : (SporosVmBin.fn_out_push_byte, (.ofInt (.mov 12 7)), 3) ∈ SporosVmBin.vmTable_24),
      exec_mov hr7, hrip]
    rfl
  let s1 := { s.set 12 (some b) with rip := 4201628 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) b
  have hs1_15 : s1.reg 15 = some ctxBase := by
    simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hrd_c208 : rd64 s1.mem (ea ctxBase 208) = some outLen := by
    rw [hea_c208]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc208)
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 0 (some outLen) with rip := 4201635 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201628, (.ofInt (.ld 0 15 208)), 7) ∈ SporosVmBin.vmTable_24),
      exec_ld hs1_15 hrd_c208]
  let s2 := { s1.set 0 (some outLen) with rip := 4201635 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) outLen
  have hs2_15 : s2.reg 15 = some ctxBase := by
    simp only [s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 15), List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hrd_c216 : rd64 s2.mem (ea ctxBase 216) = some outCap := by
    rw [hea_c216]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc216)
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 1 (some outCap) with rip := 4201642 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201635, (.ofInt (.ld 1 15 216)), 7) ∈ SporosVmBin.vmTable_24),
      exec_ld hs2_15 hrd_c216]
  let s3 := { s2.set 1 (some outCap) with rip := 4201642 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) outCap
  have hs3_0 : s3.reg 0 = some outLen := by
    simp only [s3, s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0)]
    rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  have hs3_1 : s3.reg 1 = some outCap := by
    simp only [s3, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
  have h04 : X86Vm.step vmImage s3 =
      .ok { s3 with fl := (X86Int.aluF .sub outLen outCap).2, rip := 4201645 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201642, (.ofInt (.cmp 0 1)), 3) ∈ SporosVmBin.vmTable_24),
      exec_cmp hs3_0 hs3_1]
  let s4 := { s3 with fl := (X86Int.aluF .sub outLen outCap).2, rip := 4201645 }
  have hr4 : RegsOk s4 := hr3
  have hnotlt : ¬ outLen < outCap := by omega
  have hcb : X86Int.cond s4.fl 2 = some false := by
    dsimp [s4]; rw [cond_aluF_sub_b, decide_eq_false hnotlt]
  have h05 : X86Vm.step vmImage s4 = .ok { s4 with rip := 4201647 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201645, (.ofInt (.jcc 2 4201694)), 2) ∈ SporosVmBin.vmTable_24),
      exec_jcc hcb]
    rfl
  let s5 := { s4 with rip := 4201647 }
  have hr5 : RegsOk s5 := hr4
  have hs5_15 : s5.reg 15 = some ctxBase := by
    simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 15), List.getElem?_set_ne (by decide : 0 ≠ 15),
      List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hrd_c200 : rd64 s5.mem (ea ctxBase 200) = some outBase := by
    rw [hea_c200]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc200)
  have h06 : X86Vm.step vmImage s5 = .ok { s5.set 7 (some outBase) with rip := 4201654 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201647, (.ofInt (.ld 7 15 200)), 7) ∈ SporosVmBin.vmTable_24),
      exec_ld hs5_15 hrd_c200]
  let s6 := { s5.set 7 (some outBase) with rip := 4201654 }
  have hr6' : RegsOk s6 := regsOk_set hr5 (by decide) outBase
  have hs6_1 : s6.reg 1 = some outCap := by
    simp only [s6, s5, s4, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 7 ≠ 1)]
    exact hs3_1
  have h07 : X86Vm.step vmImage s6 = .ok { s6.set 6 (some outCap) with rip := 4201657 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201654, (.ofInt (.mov 6 1)), 3) ∈ SporosVmBin.vmTable_24),
      exec_mov hs6_1]
  let s7 := { s6.set 6 (some outCap) with rip := 4201657 }
  have hr7' : RegsOk s7 := regsOk_set hr6' (by decide) outCap
  have hs7_1 : s7.reg 1 = some outCap := by
    simp only [s7, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 6 ≠ 1)]
    exact hs6_1
  have h08 : X86Vm.step vmImage s7 = .ok { s7.set 2 (some outCap) with rip := 4201660 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201657, (.ofInt (.mov 2 1)), 3) ∈ SporosVmBin.vmTable_24),
      exec_mov hs7_1]
  let s8 := { s7.set 2 (some outCap) with rip := 4201660 }
  have hr8 : RegsOk s8 := regsOk_set hr7' (by decide) outCap
  have hs8_2 : s8.reg 2 = some outCap := by
    simp only [s8, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr7'.1]; decide)]
  have h09 : X86Vm.step vmImage s8 =
      .ok { s8.set 2 (some (outCap * 2)) with fl := X86Int.Flags.undef, rip := 4201664 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201660, (.ofInt (.shlI 2 1)), 4) ∈ SporosVmBin.vmTable_24),
      exec_shlI_1_lt hs8_2 (by omega)]
  let s9 := { s8.set 2 (some (outCap * 2)) with fl := X86Int.Flags.undef, rip := 4201664 }
  have hr9 : RegsOk s9 := regsOk_set hr8 (by decide) _
  have hs9_4 : s9.reg 4 = some sp := by
    simp only [s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 2 ≠ 4), List.getElem?_set_ne (by decide : 6 ≠ 4),
      List.getElem?_set_ne (by decide : 7 ≠ 4), List.getElem?_set_ne (by decide : 1 ≠ 4),
      List.getElem?_set_ne (by decide : 0 ≠ 4), List.getElem?_set_ne (by decide : 12 ≠ 4)]
    exact hsp
  have hea_sp : (sp + X86Env.W - 8) % X86Env.W = sp - 8 := by
    have : sp + X86Env.W - 8 = (sp - 8) + X86Env.W := by omega
    rw [this, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  obtain ⟨istk, m10, hwr10, hget10, hm10eq, hwf10, hstk10, hoth10⟩ :=
    wr64_region hmwf hstk hstkrw hstk_lo (by omega : sp - 8 + 8 ≤ stk.hi) (v := 4201669)
  have hwr10' : wr64 s9.mem ((sp + X86Env.W - 8) % X86Env.W) 4201669 = some m10 := by
    simpa [s9, s8, s7, s6, s5, s4, s3, s2, s1, hea_sp] using hwr10
  have h10 : X86Vm.step vmImage s9 =
      .ok { s9.set 4 (some (sp - 8)) with mem := m10, rip := 4201363 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201664, (.ofInt (.call 4201363)), 5) ∈ SporosVmBin.vmTable_24),
      exec_call hs9_4 hwr10']
    simp [s9, hea_sp]
  let s10 := { s9.set 4 (some (sp - 8)) with mem := m10, rip := 4201363 }
  have hr10 : RegsOk s10 := regsOk_set hr9 (by decide) _
  let stk1 := { stk with data := writeBytes stk.data (sp - 8 - stk.base) (bytes8 4201669) }
  have hstk1_mem : stk1 ∈ m10 := hstk10
  have hstk1_hi : stk1.hi = stk.hi := by simp [stk1, Region.hi, writeBytes_length]
  have hret10 : rd64 s10.mem (sp - 8) = some 4201669 := by
    have hr := rd64_of_writeBytes_same hwf10 hstk10 hstk_lo (by omega : sp - 8 + 8 ≤ stk.hi)
    simpa [s10, show (4201669 % X86Env.W) = 4201669 from by decide] using hr
  have hout_m10 : outReg ∈ m10 := hoth10 outReg hout (by rw [hout_base]; exact hstkne_out.symm)
  have hctx_m10 : ctx ∈ m10 := hoth10 ctx hctx (by rw [hctx_base]; exact hstkne_ctx.symm)
  obtain ⟨idx, hfind10⟩ := mem_find_of_mem m10 outReg outBase 0 hwf10.2 hout_m10
    (by simp [hout_base]) (by simp [Region.hi, hout_base, hout_eq]; omega)
  have hfind10' : m10.find outBase 0 = some (idx, outReg) := by simpa using hfind10
  have hget_idx := (mem_find_idx_lt m10 outBase 0 idx outReg hfind10').2.2
  simp only [Nat.sub_zero] at hget_idx
  have hpos10 : ∀ u ∈ m10, 0 < u.data.length := by
    intro u hu
    rw [hm10eq] at hu
    rcases List.mem_or_eq_of_mem_set hu with hu0 | rfl
    · exact hpos u hu0
    · simpa [writeBytes_length] using hpos stk hstk
  have hpg_1 : X86Env.pages outCap = outCap := pages_of_dvd hout_pg
  have hout_pg2 : (outCap * 2) % X86Env.page = 0 := by
    rw [Nat.mul_mod, hout_pg]; decide
  have hpg_2 : X86Env.pages (outCap * 2) = outCap * 2 := pages_of_dvd hout_pg2
  have hfree10 : X86Env.Mem.free (m10.eraseIdx idx) a (X86Env.pages (outCap * 2)) = true := by
    rw [hpg_2]
    refine mem_eraseIdx_free hwf10 hget_idx hpos10 ?_
    intro u hu hub
    have hne_out : u.base ≠ outBase := by rw [← hout_base]; exact hub
    rw [hm10eq] at hu
    rcases List.mem_or_eq_of_mem_set hu with hu0 | rfl
    · exact hfree u hu0 hne_out
    · simp only [Region.hi, writeBytes_length]
      exact hfree stk hstk hstkne_out
  have hs10_4 : s10.reg 4 = some (sp - 8) := by
    simp only [s10, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr9.1]; decide)]
  have hs10_7 : s10.reg 7 = some outBase := by
    simp only [s10, s9, s8, s7, s6, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 4 ≠ 7), List.getElem?_set_ne (by decide : 2 ≠ 7),
      List.getElem?_set_ne (by decide : 6 ≠ 7)]
    rw [getElem?_set_getD_self (by rw [hr5.1]; decide)]
  have hs10_6 : s10.reg 6 = some outCap := by
    simp only [s10, s9, s8, s7, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 4 ≠ 6), List.getElem?_set_ne (by decide : 2 ≠ 6)]
    rw [getElem?_set_getD_self (by rw [hr6'.1]; decide)]
  have hs10_2 : s10.reg 2 = some (outCap * 2) := by
    simp only [s10, s9, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 4 ≠ 2)]
    rw [getElem?_set_getD_self (by rw [hr8.1]; decide)]
  obtain ⟨s11, hrm, hrip11, hk11, hmem11, hm11wf, hr11', hs11_0, hs11_4, hoth11⟩ :=
    fn_mremap_ok_spec (s := s10) (sp := sp - 8) (retAddr := 4201669)
      (oldPtr := outBase) (oldLen := outCap) (newLen := outCap * 2)
      (idx := idx) (a := a) (oldReg := outReg) (stk := stk1)
      rfl hr10 hs10_4 hstk1_mem (by simp [stk1]; exact hstkne_out) hstk_lo (by rw [hstk1_hi]; omega)
      hret10 hwf10 hs10_7 hs10_6 hs10_2 hfind10' hout_base (by rw [hpg_1, hout_eq]) hout_rw
      (by rw [hpg_2, hout_eq]; omega) hor ha_pg ha_pos (by rw [hpg_2]; exact ha_lim) hfree10
  have hm11_eq_rep : X86Env.pages (outCap * 2) - outReg.data.length = outCap := by
    rw [hpg_2, hout_eq]; omega
  let outNew : Region := ⟨a, outReg.data ++ List.replicate outCap 0, true⟩
  have houtNew_m11 : outNew ∈ s11.mem := by
    rw [hmem11]
    refine List.mem_append_right _ ?_
    simp [outNew, hm11_eq_rep]
  have hctx_m11 : ctx ∈ s11.mem := by
    rw [hmem11]
    exact List.mem_append_left _ (mem_eraseIdx_other hget_idx hctx_m10 (by rw [hout_base, hctx_base]; exact hne.symm))
  have hstk1_m11 : stk1 ∈ s11.mem := by
    rw [hmem11]
    exact List.mem_append_left _ (mem_eraseIdx_other hget_idx hstk1_mem (by simp [stk1, hout_base]; exact hstkne_out))
  have hs11_4' : s11.reg 4 = some sp := by
    have : (sp - 8 + 8) % X86Env.W = sp := by rw [Nat.sub_add_cancel h8, Nat.mod_eq_of_lt hspW]
    rwa [this] at hs11_4
  have hs11_15 : s11.reg 15 = some ctxBase := by
    rw [hoth11 15 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    simp only [s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 4 ≠ 15), List.getElem?_set_ne (by decide : 2 ≠ 15),
      List.getElem?_set_ne (by decide : 6 ≠ 15), List.getElem?_set_ne (by decide : 7 ≠ 15),
      List.getElem?_set_ne (by decide : 1 ≠ 15), List.getElem?_set_ne (by decide : 0 ≠ 15),
      List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hs11_12 : s11.reg 12 = some b := by
    rw [hoth11 12 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    simp only [s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 4 ≠ 12), List.getElem?_set_ne (by decide : 2 ≠ 12),
      List.getElem?_set_ne (by decide : 6 ≠ 12), List.getElem?_set_ne (by decide : 7 ≠ 12),
      List.getElem?_set_ne (by decide : 1 ≠ 12), List.getElem?_set_ne (by decide : 0 ≠ 12)]
    rw [getElem?_set_getD_self (by rw [hregs.1]; decide)]
  have hm11wf' : s11.mem.wf := by rw [hmem11]; exact hm11wf
  obtain ⟨_, m12, hwr12, _, _, hwf12, hctx12, hoth12⟩ :=
    wr64_region hm11wf' hctx_m11 hctx_rw (a := ctxBase + 200) (v := a)
      (by rw [hctx_base]; omega) (by simp [Region.hi, hctx_base]; omega)
  have hwr12' : wr64 s11.mem (ea ctxBase 200) a = some m12 := by simpa [hea_c200] using hwr12
  have h12 : X86Vm.step vmImage s11 = .ok { s11 with mem := m12, rip := 4201676 } := by
    rw [step_at_chunk 24 hrip11 rfl (by decide : (4201669, (.ofInt (.st 15 200 0)), 7) ∈ SporosVmBin.vmTable_24),
      exec_st hs11_15 hs11_0 hwr12', hrip11]
  let s12 := { s11 with mem := m12, rip := 4201676 }
  have hr12 : RegsOk s12 := hr11'
  have hs12_15 : s12.reg 15 = some ctxBase := hs11_15
  let ctx12 := { ctx with data := writeBytes ctx.data (ctxBase + 200 - ctx.base) (bytes8 a) }
  have hctx12_hc216 : leVal ((ctx12.data.drop 216).take 8) = outCap := by
    simp only [ctx12, hctx_base, Nat.add_sub_cancel_left]
    rw [writeBytes_drop_take_disjoint ctx.data 200 216 8 (bytes8 a) (by right; rw [bytes8_length]; omega) (by omega)]
    exact hc216
  have hrd12_c216 : rd64 s12.mem (ea ctxBase 216) = some outCap := by
    rw [hea_c216]
    exact rd64_of_mem hwf12 hctx12 (by simp [hctx_base])
      (by simp [Region.hi, writeBytes_length, hctx_base]; omega)
      (by simpa [ctx12, hctx_base] using hctx12_hc216)
  have h13 : X86Vm.step vmImage s12 = .ok { s12.set 1 (some outCap) with rip := 4201683 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201676, (.ofInt (.ld 1 15 216)), 7) ∈ SporosVmBin.vmTable_24),
      exec_ld hs12_15 hrd12_c216]
  let s13 := { s12.set 1 (some outCap) with rip := 4201683 }
  have hr13' : RegsOk s13 := regsOk_set hr12 (by decide) outCap
  have hs13_1 : s13.reg 1 = some outCap := by
    simp only [s13, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr12.1]; decide)]
  have h14 : X86Vm.step vmImage s13 =
      .ok { s13.set 1 (some (outCap * 2)) with fl := X86Int.Flags.undef, rip := 4201687 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201683, (.ofInt (.shlI 1 1)), 4) ∈ SporosVmBin.vmTable_24),
      exec_shlI_1_lt hs13_1 (by omega)]
  let s14 := { s13.set 1 (some (outCap * 2)) with fl := X86Int.Flags.undef, rip := 4201687 }
  have hr14 : RegsOk s14 := regsOk_set hr13' (by decide) _
  have hs14_15 : s14.reg 15 = some ctxBase := by
    simp only [s14, s13, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 15)]
    exact hs12_15
  have hs14_1 : s14.reg 1 = some (outCap * 2) := by
    simp only [s14, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr13'.1]; decide)]
  obtain ⟨_, m15, hwr15, _, _, hwf15, hctx15, hoth15⟩ :=
    wr64_region hwf12 hctx12 hctx_rw (a := ctxBase + 216) (v := outCap * 2)
      (by simp [hctx_base])
      (by simp [Region.hi, writeBytes_length, hctx_base]; omega)
  have hwr15' : wr64 s14.mem (ea ctxBase 216) (outCap * 2) = some m15 := by simpa [s14, s13, hea_c216] using hwr15
  have h15 : X86Vm.step vmImage s14 = .ok { s14 with mem := m15, rip := 4201694 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201687, (.ofInt (.st 15 216 1)), 7) ∈ SporosVmBin.vmTable_24),
      exec_st hs14_15 hs14_1 hwr15']
  let s15 := { s14 with mem := m15, rip := 4201694 }
  have hr15' : RegsOk s15 := hr14
  have houtNew_m12 : outNew ∈ m12 := hoth12 outNew houtNew_m11 (by simp [outNew, hctx_base]; exact hane)
  have houtNew_m15 : outNew ∈ m15 := hoth15 outNew houtNew_m12 (by simp [outNew, hctx_base]; exact hane)
  have hstk1_m12 : stk1 ∈ m12 := hoth12 stk1 hstk1_m11 (by simp [stk1, hctx_base]; exact hstkne_ctx)
  have hstk1_m15 : stk1 ∈ m15 := hoth15 stk1 hstk1_m12 (by simp [stk1, hctx_base]; exact hstkne_ctx)
  let ctx15 := { ctx12 with data := writeBytes ctx12.data (ctxBase + 216 - ctx12.base) (bytes8 (outCap * 2)) }
  have hctx15_hc200 : leVal ((ctx15.data.drop 200).take 8) = a := by
    simp only [ctx15, ctx12, hctx_base, Nat.add_sub_cancel_left]
    rw [writeBytes_drop_take_disjoint _ 216 200 8 (bytes8 (outCap * 2)) (by left; omega) (by rw [writeBytes_length]; omega)]
    rw [leVal_writeBytes_same _ 200 a (by omega)]
    exact Nat.mod_eq_of_lt (by have := vm_lim_lt_W; omega)
  have hctx15_hc208 : leVal ((ctx15.data.drop 208).take 8) = outLen := by
    simp only [ctx15, ctx12, hctx_base, Nat.add_sub_cancel_left]
    rw [writeBytes_drop_take_disjoint _ 216 208 8 (bytes8 (outCap * 2)) (by left; omega) (by rw [writeBytes_length]; omega)]
    rw [writeBytes_drop_take_disjoint ctx.data 200 208 8 (bytes8 a) (by right; rw [bytes8_length]; omega) (by omega)]
    exact hc208
  have hret_stk1 : rd64 s15.mem sp = some retAddr := by
    have hstk1_drop : (stk1.data.drop (sp - stk1.base)).take 8 = (stk.data.drop (sp - stk.base)).take 8 := by
      simp only [stk1]
      apply writeBytes_drop_take_disjoint
      · right; rw [bytes8_length]; omega
      · simp [Region.hi] at hstk_hi; omega
    obtain ⟨v, hv⟩ := show ∃ v, leVal ((stk.data.drop (sp - stk.base)).take 8) = v from ⟨_, rfl⟩
    have hold := rd64_of_mem hmwf hstk (by omega) hstk_hi hv
    rw [hret, Option.some.injEq] at hold
    subst hold
    exact rd64_of_mem hwf15 hstk1_m15 (by simp [stk1]; omega)
      (by simp [stk1, Region.hi, writeBytes_length] at hstk_hi ⊢; omega) (by rw [hstk1_drop, hv])
  have hctx15_hi : ctxSize ≤ ctx15.data.length := by
    simp [ctx15, ctx12, writeBytes_length]; exact hctx_len
  have hs15_4 : s15.reg 4 = some sp := by
    simp only [s15, s14, s13, s12, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 4)]
    exact hs11_4'
  have hs15_12 : s15.reg 12 = some b := by
    simp only [s15, s14, s13, s12, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 12)]
    exact hs11_12
  obtain ⟨m', sf, hreach_join, hripf, hkf, hmemf, hwf', hregsf, hspf, houtf, hctxf, hothf, hregoth⟩ :=
    fn_out_push_byte_join_spec (s := s15) (sp := sp) (retAddr := retAddr) (ctxBase := ctxBase)
      (outBase := a) (outLen := outLen) (b := b) (ctx := ctx15) (outReg := outNew) (stk := stk1)
      rfl hr15' hs15_4 hs14_15 hs15_12
      (by simpa [ctx15] using hctx15) (by simp [ctx15, ctx12, hctx_base]) hctx_rw hctx15_hi
      (by simpa [s15] using houtNew_m15) rfl rfl hane
      (by simpa [s15] using hstk1_m15) (by simp [stk1]; exact hastk.symm)
      (by simp [stk1]; exact hstkne_ctx) (by simp [stk1]; omega)
      (by simp [stk1, Region.hi, writeBytes_length] at hstk_hi ⊢; omega)
      (by simp [outNew, List.length_append, List.length_replicate, hout_eq]; omega)
      hctxW (by have := vm_lim_lt_W; omega) (by have := vm_lim_lt_W; omega) hwf15 hret_stk1 hctx15_hc200 hctx15_hc208
  have hstk1_m' : stk1 ∈ m' := by
    apply hothf stk1 (by simpa [s15] using hstk1_m15)
    · simp [stk1]; exact hastk.symm
    · simp [stk1]; exact hstkne_ctx
  refine ⟨m', sf, stk1,
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
      (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step_trans h08
        (Reaches.step_trans h09 (Reaches.step_trans h10
          (Reaches.trans hrm (Reaches.step_trans h12 (Reaches.step_trans h13 (Reaches.step_trans h14
            (Reaches.step_trans h15 hreach_join)))))))))))))),
    hripf, by simpa [s15, s14, s13, s12, hk11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1] using hkf, hmemf, hwf', hregsf, hspf,
    by simpa [outNew] using houtf,
    by simpa [ctx15, ctx12, hctx_base, Nat.add_sub_cancel_left] using hctxf,
    hstk1_m', rfl, hstkrw, writeBytes_length _ _ _, ?_, ?_, ?_⟩
  · intro j hj
    apply writeBytes_getElem?_disjoint
    rw [bytes8_length]
    rcases hj with h | h
    · exact Or.inl h
    · exact Or.inr (by omega)
  · intro u hu huout huctx hustk
    have hu10 : u ∈ m10 := hoth10 u hu hustk
    have hu11 : u ∈ s11.mem := by
      rw [hmem11]
      exact List.mem_append_left _ (mem_eraseIdx_other hget_idx hu10 (by rw [hout_base]; exact huout))
    have hu12 : u ∈ m12 := hoth12 u hu11 (by rw [hctx_base]; exact huctx)
    have hu15 : u ∈ m15 := hoth15 u hu12 (by simp [hctx_base]; exact huctx)
    have hua : u.base ≠ a := by
      intro hba
      have hupos := hpos u hu
      rcases hfree u hu huout with h1 | h2
      · omega
      · simp [Region.hi] at h2; omega
    exact hothf u (by simpa [s15] using hu15) hua huctx
  · intro r h0 h1 h2 h4 h6 h7 h8 h9 h10' h11' h12'
    have h15_11 : s15.reg r = s11.reg r := by
      simp only [s15, s14, s13, s12, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne h1.symm]
    rw [hregoth r h0 h1 h4, h15_11,
      hoth11 r h0 h1 h2 h4 h6 h7 h8 h9 h10' h11']
    simp only [s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne h4.symm, List.getElem?_set_ne h2.symm, List.getElem?_set_ne h6.symm,
      List.getElem?_set_ne h7.symm, List.getElem?_set_ne h1.symm, List.getElem?_set_ne h0.symm,
      List.getElem?_set_ne h12'.symm]

theorem fn_out_push_byte_nomem_spec {s : X86Vm.St} {sp ctxBase outBase outLen outCap b : Nat}
    {ctx outReg stk : Region}
    (hrip : s.rip = SporosVmBin.fn_out_push_byte)
    (hregs : RegsOk s)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr7 : s.reg 7 = some b)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_len : ctxSize ≤ ctx.data.length)
    (hout : outReg ∈ s.mem) (hout_base : outReg.base = outBase) (hout_rw : outReg.rw = true)
    (hstk : stk ∈ s.mem) (hstkrw : stk.rw = true) (hstkne_out : stk.base ≠ outBase)
    (h16 : 16 ≤ sp) (hspW : sp < X86Env.W)
    (hstk_lo : stk.base ≤ sp - 16) (hstk_hi : sp ≤ stk.hi)
    (hge : outCap ≤ outLen) (hle : outLen ≤ outCap) (hcap_pos : 0 < outCap)
    (hout_eq : outReg.data.length = outCap)
    (hout_pg : outCap % X86Env.page = 0)
    (houtCapW : outCap * 2 < X86Env.W)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W)
    (hmwf : s.mem.wf)
    (hc200 : leVal ((ctx.data.drop 200).take 8) = outBase)
    (hc208 : leVal ((ctx.data.drop 208).take 8) = outLen)
    (hc216 : leVal ((ctx.data.drop 216).take 8) = outCap)
    (hor : (s.k.next).1 = some .nomem) :
    Ends vmImage s (.exit 2) := by
  have hcs : ctxSize = 4096 := rfl
  have hea_c200 : ea ctxBase 200 = ctxBase + 200 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 200 < X86Env.W)]
  have hea_c208 : ea ctxBase 208 = ctxBase + 208 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 208 < X86Env.W)]
  have hea_c216 : ea ctxBase 216 = ctxBase + 216 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 216 < X86Env.W)]
  have h01 : X86Vm.step vmImage s = .ok { s.set 12 (some b) with rip := 4201628 } := by
    rw [step_at_chunk 24 hrip rfl (by decide : (SporosVmBin.fn_out_push_byte, (.ofInt (.mov 12 7)), 3) ∈ SporosVmBin.vmTable_24),
      exec_mov hr7, hrip]
    rfl
  let s1 := { s.set 12 (some b) with rip := 4201628 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) b
  have hs1_15 : s1.reg 15 = some ctxBase := by
    simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hrd_c208 : rd64 s1.mem (ea ctxBase 208) = some outLen := by
    rw [hea_c208]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc208)
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 0 (some outLen) with rip := 4201635 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201628, (.ofInt (.ld 0 15 208)), 7) ∈ SporosVmBin.vmTable_24),
      exec_ld hs1_15 hrd_c208]
  let s2 := { s1.set 0 (some outLen) with rip := 4201635 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) outLen
  have hs2_15 : s2.reg 15 = some ctxBase := by
    simp only [s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 15), List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hrd_c216 : rd64 s2.mem (ea ctxBase 216) = some outCap := by
    rw [hea_c216]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc216)
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 1 (some outCap) with rip := 4201642 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201635, (.ofInt (.ld 1 15 216)), 7) ∈ SporosVmBin.vmTable_24),
      exec_ld hs2_15 hrd_c216]
  let s3 := { s2.set 1 (some outCap) with rip := 4201642 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) outCap
  have hs3_0 : s3.reg 0 = some outLen := by
    simp only [s3, s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0)]
    rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  have hs3_1 : s3.reg 1 = some outCap := by
    simp only [s3, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
  have h04 : X86Vm.step vmImage s3 =
      .ok { s3 with fl := (X86Int.aluF .sub outLen outCap).2, rip := 4201645 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201642, (.ofInt (.cmp 0 1)), 3) ∈ SporosVmBin.vmTable_24),
      exec_cmp hs3_0 hs3_1]
  let s4 := { s3 with fl := (X86Int.aluF .sub outLen outCap).2, rip := 4201645 }
  have hr4 : RegsOk s4 := hr3
  have hnotlt : ¬ outLen < outCap := by omega
  have hcb : X86Int.cond s4.fl 2 = some false := by
    dsimp [s4]; rw [cond_aluF_sub_b, decide_eq_false hnotlt]
  have h05 : X86Vm.step vmImage s4 = .ok { s4 with rip := 4201647 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201645, (.ofInt (.jcc 2 4201694)), 2) ∈ SporosVmBin.vmTable_24),
      exec_jcc hcb]
    rfl
  let s5 := { s4 with rip := 4201647 }
  have hr5 : RegsOk s5 := hr4
  have hs5_15 : s5.reg 15 = some ctxBase := by
    simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 15), List.getElem?_set_ne (by decide : 0 ≠ 15),
      List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hrd_c200 : rd64 s5.mem (ea ctxBase 200) = some outBase := by
    rw [hea_c200]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc200)
  have h06 : X86Vm.step vmImage s5 = .ok { s5.set 7 (some outBase) with rip := 4201654 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201647, (.ofInt (.ld 7 15 200)), 7) ∈ SporosVmBin.vmTable_24),
      exec_ld hs5_15 hrd_c200]
  let s6 := { s5.set 7 (some outBase) with rip := 4201654 }
  have hr6' : RegsOk s6 := regsOk_set hr5 (by decide) outBase
  have hs6_1 : s6.reg 1 = some outCap := by
    simp only [s6, s5, s4, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 7 ≠ 1)]
    exact hs3_1
  have h07 : X86Vm.step vmImage s6 = .ok { s6.set 6 (some outCap) with rip := 4201657 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201654, (.ofInt (.mov 6 1)), 3) ∈ SporosVmBin.vmTable_24),
      exec_mov hs6_1]
  let s7 := { s6.set 6 (some outCap) with rip := 4201657 }
  have hr7' : RegsOk s7 := regsOk_set hr6' (by decide) outCap
  have hs7_1 : s7.reg 1 = some outCap := by
    simp only [s7, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 6 ≠ 1)]
    exact hs6_1
  have h08 : X86Vm.step vmImage s7 = .ok { s7.set 2 (some outCap) with rip := 4201660 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201657, (.ofInt (.mov 2 1)), 3) ∈ SporosVmBin.vmTable_24),
      exec_mov hs7_1]
  let s8 := { s7.set 2 (some outCap) with rip := 4201660 }
  have hr8 : RegsOk s8 := regsOk_set hr7' (by decide) outCap
  have hs8_2 : s8.reg 2 = some outCap := by
    simp only [s8, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr7'.1]; decide)]
  have h09 : X86Vm.step vmImage s8 =
      .ok { s8.set 2 (some (outCap * 2)) with fl := X86Int.Flags.undef, rip := 4201664 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201660, (.ofInt (.shlI 2 1)), 4) ∈ SporosVmBin.vmTable_24),
      exec_shlI_1_lt hs8_2 (by omega)]
  let s9 := { s8.set 2 (some (outCap * 2)) with fl := X86Int.Flags.undef, rip := 4201664 }
  have hr9 : RegsOk s9 := regsOk_set hr8 (by decide) _
  have hs9_4 : s9.reg 4 = some sp := by
    simp only [s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 2 ≠ 4), List.getElem?_set_ne (by decide : 6 ≠ 4),
      List.getElem?_set_ne (by decide : 7 ≠ 4), List.getElem?_set_ne (by decide : 1 ≠ 4),
      List.getElem?_set_ne (by decide : 0 ≠ 4), List.getElem?_set_ne (by decide : 12 ≠ 4)]
    exact hsp
  have hea_sp : (sp + X86Env.W - 8) % X86Env.W = sp - 8 := by
    have : sp + X86Env.W - 8 = (sp - 8) + X86Env.W := by omega
    rw [this, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  obtain ⟨istk, m10, hwr10, _, _, hwf10, hstk10, hoth10⟩ :=
    wr64_region hmwf hstk hstkrw (by omega : stk.base ≤ sp - 8) (by omega : sp - 8 + 8 ≤ stk.hi) (v := 4201669)
  have hwr10' : wr64 s9.mem ((sp + X86Env.W - 8) % X86Env.W) 4201669 = some m10 := by
    simpa [s9, s8, s7, s6, s5, s4, s3, s2, s1, hea_sp] using hwr10
  have h10 : X86Vm.step vmImage s9 =
      .ok { s9.set 4 (some (sp - 8)) with mem := m10, rip := 4201363 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201664, (.ofInt (.call 4201363)), 5) ∈ SporosVmBin.vmTable_24),
      exec_call hs9_4 hwr10']
    simp [s9, hea_sp]
  let s10 := { s9.set 4 (some (sp - 8)) with mem := m10, rip := 4201363 }
  have hr10 : RegsOk s10 := regsOk_set hr9 (by decide) _
  let stk1 := { stk with data := writeBytes stk.data (sp - 8 - stk.base) (bytes8 4201669) }
  have hstk1_hi : stk1.hi = stk.hi := by simp [stk1, Region.hi, writeBytes_length]
  have hout_m10 : outReg ∈ m10 := hoth10 outReg hout (by rw [hout_base]; exact hstkne_out.symm)
  obtain ⟨idx, hfind10⟩ := mem_find_of_mem m10 outReg outBase 0 hwf10.2 hout_m10
    (by simp [hout_base]) (by simp [Region.hi, hout_base, hout_eq]; omega)
  have hfind10' : m10.find outBase 0 = some (idx, outReg) := by simpa using hfind10
  have hpg_1 : X86Env.pages outCap = outCap := pages_of_dvd hout_pg
  have hout_pg2 : (outCap * 2) % X86Env.page = 0 := by
    rw [Nat.mul_mod, hout_pg]; decide
  have hpg_2 : X86Env.pages (outCap * 2) = outCap * 2 := pages_of_dvd hout_pg2
  have hs10_4 : s10.reg 4 = some (sp - 8) := by
    simp only [s10, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr9.1]; decide)]
  have hs10_7 : s10.reg 7 = some outBase := by
    simp only [s10, s9, s8, s7, s6, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 4 ≠ 7), List.getElem?_set_ne (by decide : 2 ≠ 7),
      List.getElem?_set_ne (by decide : 6 ≠ 7)]
    rw [getElem?_set_getD_self (by rw [hr5.1]; decide)]
  have hs10_6 : s10.reg 6 = some outCap := by
    simp only [s10, s9, s8, s7, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 4 ≠ 6), List.getElem?_set_ne (by decide : 2 ≠ 6)]
    rw [getElem?_set_getD_self (by rw [hr6'.1]; decide)]
  have hs10_2 : s10.reg 2 = some (outCap * 2) := by
    simp only [s10, s9, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 4 ≠ 2)]
    rw [getElem?_set_getD_self (by rw [hr8.1]; decide)]
  have h_exit := fn_mremap_nomem_spec (s := s10) (sp := sp - 8)
    (oldPtr := outBase) (oldLen := outCap) (newLen := outCap * 2)
    (idx := idx) (oldReg := outReg) (stk := stk1)
    rfl hr10 hs10_4 (by omega) (by omega) hwf10 hstk10 hstkrw
    (by simp [stk1]; omega) (by rw [hstk1_hi]; omega)
    hs10_7 hs10_6 hs10_2 hfind10' hout_base (by rw [hpg_1, hout_eq]) hout_rw
    (by rw [hpg_2, hout_eq]; omega) hor
  exact Ends.step_trans h01 (Ends.step_trans h02 (Ends.step_trans h03 (Ends.step_trans h04
    (Ends.step_trans h05 (Ends.step_trans h06 (Ends.step_trans h07 (Ends.step_trans h08
      (Ends.step_trans h09 (Ends.step_trans h10 h_exit)))))))))

theorem fn_out_push_u64_join_spec {s : X86Vm.St} {sp retAddr ctxBase outBase outLen v : Nat}
    {ctx outReg stk : Region}
    (hrip : s.rip = 4201799)
    (hregs : RegsOk s)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr12 : s.reg 12 = some v)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_rw : ctx.rw = true)
    (hctx_len : ctxSize ≤ ctx.data.length)
    (hout : outReg ∈ s.mem) (hout_base : outReg.base = outBase) (hout_rw : outReg.rw = true)
    (hne : outBase ≠ ctxBase)
    (hstk : stk ∈ s.mem) (hstkne_out : stk.base ≠ outBase) (hstkne_ctx : stk.base ≠ ctxBase)
    (hstk_lo : stk.base ≤ sp) (hstk_hi : sp + 8 ≤ stk.hi)
    (hout_hi : outLen + 8 ≤ outReg.data.length)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W) (houtW : outBase + outLen + 8 ≤ X86Env.W)
    (houtLenW : outLen + 8 < X86Env.W)
    (hmwf : s.mem.wf)
    (hret : rd64 s.mem sp = some retAddr)
    (hc200 : leVal ((ctx.data.drop 200).take 8) = outBase)
    (hc208 : leVal ((ctx.data.drop 208).take 8) = outLen) :
    let outData' := writeBytes outReg.data outLen (bytes8 v)
    let ctxData' := writeBytes ctx.data 208 (bytes8 (outLen + 8))
    ∃ m' s', Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.k = s.k ∧ s'.mem = m' ∧ m'.wf ∧ RegsOk s' ∧
      s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      { outReg with data := outData' } ∈ m' ∧
      { ctx with data := ctxData' } ∈ m' ∧
      (∀ u ∈ s.mem, u.base ≠ outBase → u.base ≠ ctxBase → u ∈ m') ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 4 → s'.reg r = s.reg r) := by
  have hcs : ctxSize = 4096 := rfl
  have hea_c200 : ea ctxBase 200 = ctxBase + 200 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 200 < X86Env.W)]
  have hea_c208 : ea ctxBase 208 = ctxBase + 208 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 208 < X86Env.W)]
  have hrd_c200 : rd64 s.mem (ea ctxBase 200) = some outBase := by
    rw [hea_c200]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc200)
  have h01 : X86Vm.step vmImage s = .ok { s.set 0 (some outBase) with rip := 4201806 } := by
    rw [step_at_chunk 25 hrip rfl (by decide : (4201799, (.ofInt (.ld 0 15 200)), 7) ∈ SporosVmBin.vmTable_25),
      exec_ld hr15 hrd_c200, hrip]
  let s1 := { s.set 0 (some outBase) with rip := 4201806 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) outBase
  have hs1_15 : s1.reg 15 = some ctxBase := by
    simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 15)]
    exact hr15
  have hrd_c208 : rd64 s1.mem (ea ctxBase 208) = some outLen := by
    rw [hea_c208]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc208)
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 1 (some outLen) with rip := 4201813 } := by
    rw [step_at_chunk 25 rfl rfl (by decide : (4201806, (.ofInt (.ld 1 15 208)), 7) ∈ SporosVmBin.vmTable_25),
      exec_ld hs1_15 hrd_c208]
  let s2 := { s1.set 1 (some outLen) with rip := 4201813 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) outLen
  have hs2_0 : s2.reg 0 = some outBase := by
    simp only [s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0)]
    rw [getElem?_set_getD_self (by rw [hregs.1]; decide)]
  have hs2_1 : s2.reg 1 = some outLen := by
    simp only [s2, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  have hs2_12 : s2.reg 12 = some v := by
    simp only [s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 12), List.getElem?_set_ne (by decide : 0 ≠ 12)]
    exact hr12
  have hea_d0 : eaIdx outBase outLen 1 0 = outBase + outLen := by
    simp [eaIdx, Nat.mod_eq_of_lt (by omega : outBase + outLen < X86Env.W)]
  obtain ⟨i1, m3, hwr3, _, _, hwf3, hr3_out, hoth3⟩ :=
    wrMem_w64_region hmwf hout hout_rw (a := outBase + outLen) (v := v)
      (by rw [hout_base]; omega) (by simp [Region.hi, hout_base]; omega)
  have hwr3' : wrMem .w64 s2.mem (eaIdx outBase outLen 1 0) v = some m3 := by
    simpa [s2, s1, hea_d0] using hwr3
  have h03 : X86Vm.step vmImage s2 = .ok { s2 with mem := m3, rip := 4201817 } := by
    rw [step_at_chunk 25 rfl rfl (by decide : (4201813, (.stIdx .w64 0 1 1 0 12), 4) ∈ SporosVmBin.vmTable_25),
      exec_stIdx hs2_0 hs2_1 hs2_12 hwr3']
  let s3 := { s2 with mem := m3, rip := 4201817 }
  have hr3 : RegsOk s3 := hr2
  have hs3_1 : s3.reg 1 = some outLen := hs2_1
  have h04 : X86Vm.step vmImage s3 =
      .ok { s3.set 1 (some ((outLen + 8) % X86Env.W)) with
            fl := (X86Int.aluF .add outLen 8).2, rip := 4201821 } := by
    rw [step_at_chunk 25 rfl rfl (by decide : (4201817, (.ofInt (.aluI .add 1 8)), 4) ∈ SporosVmBin.vmTable_25),
      exec_aluI_add_lt hs3_1 (by decide)]
  let s4 := { s3.set 1 (some ((outLen + 8) % X86Env.W)) with
                fl := (X86Int.aluF .add outLen 8).2, rip := 4201821 }
  have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) _
  have hs4_15 : s4.reg 15 = some ctxBase := by
    simp only [s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 15), List.getElem?_set_ne (by decide : 0 ≠ 15)]
    exact hr15
  have hs4_1 : s4.reg 1 = some ((outLen + 8) % X86Env.W) := by
    simp only [s4, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
  have hctx_m3 : ctx ∈ m3 := hoth3 ctx hctx (by rw [hctx_base, hout_base]; exact Ne.symm hne)
  obtain ⟨i2, m4, hwr4, _, _, hwf4, hr4_ctx, hoth4⟩ :=
    wr64_region hwf3 hctx_m3 hctx_rw (a := ctxBase + 208) (v := (outLen + 8) % X86Env.W)
      (by rw [hctx_base]; omega) (by simp [Region.hi, hctx_base]; omega)
  have hwr4' : wr64 s4.mem (ea ctxBase 208) ((outLen + 8) % X86Env.W) = some m4 := by
    simpa [s4, s3, hea_c208] using hwr4
  have h05 : X86Vm.step vmImage s4 = .ok { s4 with mem := m4, rip := 4201828 } := by
    rw [step_at_chunk 25 rfl rfl (by decide : (4201821, (.ofInt (.st 15 208 1)), 7) ∈ SporosVmBin.vmTable_25),
      exec_st hs4_15 hs4_1 hwr4']
  let s5 := { s4 with mem := m4, rip := 4201828 }
  have hr5 : RegsOk s5 := hr4
  have hs5_4 : s5.reg 4 = some sp := by
    simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 4), List.getElem?_set_ne (by decide : 0 ≠ 4)]
    exact hsp
  have hstk_m3 : stk ∈ m3 := hoth3 stk hstk (by rw [hout_base]; exact hstkne_out)
  have hstk_m4 : stk ∈ m4 := hoth4 stk hstk_m3 (by rw [hctx_base]; exact hstkne_ctx)
  have hret_m4 : rd64 s5.mem sp = some retAddr :=
    rd64_of_mem_transfer hmwf hwf4 hstk hstk_m4 hstk_lo hstk_hi hret
  have h06 : X86Vm.step vmImage s5 =
      .ok { s5.set 4 (some ((sp + 8) % X86Env.W)) with rip := retAddr } := by
    rw [step_at_chunk 25 rfl rfl (by decide : (4201828, (.ofInt .ret), 1) ∈ SporosVmBin.vmTable_25),
      exec_ret hs5_4 hret_m4]
  let s6 := { s5.set 4 (some ((sp + 8) % X86Env.W)) with rip := retAddr }
  refine ⟨m4, s6,
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
      (Reaches.step_trans h05 (Reaches.step h06))))),
    rfl, rfl, rfl, hwf4, regsOk_set hr5 (by decide) _, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [s6, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr5.1]; decide)]
  · have hout1_m4 := hoth4 _ hr3_out (by simp [hout_base, hctx_base]; exact hne)
    simpa [hout_base, show outBase + outLen - outBase = outLen from by omega] using hout1_m4
  · simpa [hctx_base, show ctxBase + 208 - ctxBase = 208 from by omega,
      Nat.mod_eq_of_lt (by omega : outLen + 8 < X86Env.W)] using hr4_ctx
  · intro u hu huout huctx
    have hu3 := hoth3 u hu (by rw [hout_base]; exact huout)
    exact hoth4 u hu3 (by rw [hctx_base]; exact huctx)
  · intro r h0 h1 h4
    simp only [s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne h4.symm, List.getElem?_set_ne h1.symm, List.getElem?_set_ne h0.symm]

private theorem fn_out_push_u64_loop_done_spec {s : X86Vm.St} {sp retAddr ctxBase outBase outLen outCap v : Nat}
    {ctx outReg stk : Region}
    (hrip : s.rip = 4201727)
    (hregs : RegsOk s)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr12 : s.reg 12 = some v)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_rw : ctx.rw = true)
    (hctx_len : ctxSize ≤ ctx.data.length)
    (hout : outReg ∈ s.mem) (hout_base : outReg.base = outBase) (hout_rw : outReg.rw = true)
    (hne : outBase ≠ ctxBase)
    (hstk : stk ∈ s.mem) (hstkne_out : stk.base ≠ outBase) (hstkne_ctx : stk.base ≠ ctxBase)
    (hstk_lo : stk.base ≤ sp) (hstk_hi : sp + 8 ≤ stk.hi)
    (hle : outLen + 8 ≤ outCap)
    (hout_hi : outCap ≤ outReg.data.length)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W) (houtW : outBase + outCap ≤ X86Env.W)
    (houtCapW : outCap < X86Env.W)
    (hmwf : s.mem.wf)
    (hret : rd64 s.mem sp = some retAddr)
    (hc200 : leVal ((ctx.data.drop 200).take 8) = outBase)
    (hc208 : leVal ((ctx.data.drop 208).take 8) = outLen)
    (hc216 : leVal ((ctx.data.drop 216).take 8) = outCap) :
    let outData' := writeBytes outReg.data outLen (bytes8 v)
    let ctxData' := writeBytes ctx.data 208 (bytes8 (outLen + 8))
    ∃ m' s', Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.k = s.k ∧ s'.mem = m' ∧ m'.wf ∧ RegsOk s' ∧
      s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      { outReg with data := outData' } ∈ m' ∧
      { ctx with data := ctxData' } ∈ m' ∧
      (∀ u ∈ s.mem, u.base ≠ outBase → u.base ≠ ctxBase → u ∈ m') ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 4 → s'.reg r = s.reg r) := by
  have hcs : ctxSize = 4096 := rfl
  have hea_c208 : ea ctxBase 208 = ctxBase + 208 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 208 < X86Env.W)]
  have hea_c216 : ea ctxBase 216 = ctxBase + 216 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 216 < X86Env.W)]
  have hrd_c208 : rd64 s.mem (ea ctxBase 208) = some outLen := by
    rw [hea_c208]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc208)
  have h01 : X86Vm.step vmImage s = .ok { s.set 0 (some outLen) with rip := 4201734 } := by
    rw [step_at_chunk 24 hrip rfl (by decide : (4201727, (.ofInt (.ld 0 15 208)), 7) ∈ SporosVmBin.vmTable_24),
      exec_ld hr15 hrd_c208, hrip]
  let s1 := { s.set 0 (some outLen) with rip := 4201734 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) outLen
  have hs1_0 : s1.reg 0 = some outLen := by
    simp only [s1, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hregs.1]; decide)]
  have h02 : X86Vm.step vmImage s1 =
      .ok { s1.set 0 (some (outLen + 8)) with
            fl := (X86Int.aluF .add outLen 8).2, rip := 4201738 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201734, (.ofInt (.aluI .add 0 8)), 4) ∈ SporosVmBin.vmTable_24),
      exec_aluI_add_lt hs1_0 (by decide), Nat.mod_eq_of_lt (by omega : outLen + 8 < X86Env.W)]
  let s2 := { s1.set 0 (some (outLen + 8)) with
                fl := (X86Int.aluF .add outLen 8).2, rip := 4201738 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) _
  have hs2_15 : s2.reg 15 = some ctxBase := by
    simp only [s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 15)]
    exact hr15
  have hrd_c216 : rd64 s2.mem (ea ctxBase 216) = some outCap := by
    rw [hea_c216]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc216)
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 1 (some outCap) with rip := 4201745 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201738, (.ofInt (.ld 1 15 216)), 7) ∈ SporosVmBin.vmTable_24),
      exec_ld hs2_15 hrd_c216]
  let s3 := { s2.set 1 (some outCap) with rip := 4201745 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) outCap
  have hs3_0 : s3.reg 0 = some (outLen + 8) := by
    simp only [s3, s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0)]
    rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  have hs3_1 : s3.reg 1 = some outCap := by
    simp only [s3, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
  have h04 : X86Vm.step vmImage s3 =
      .ok { s3 with fl := (X86Int.aluF .sub (outLen + 8) outCap).2, rip := 4201748 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201745, (.ofInt (.cmp 0 1)), 3) ∈ SporosVmBin.vmTable_24),
      exec_cmp hs3_0 hs3_1]
  let s4 := { s3 with fl := (X86Int.aluF .sub (outLen + 8) outCap).2, rip := 4201748 }
  have hr4 : RegsOk s4 := hr3
  have hcb : X86Int.cond s4.fl 6 = some true := by
    dsimp [s4]
    rw [cond_aluF_sub_be _ _ (by omega : outLen + 8 < X86Env.W) houtCapW, decide_eq_true hle]
  have h05 : X86Vm.step vmImage s4 = .ok { s4 with rip := 4201799 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201748, (.ofInt (.jcc 6 4201799)), 2) ∈ SporosVmBin.vmTable_24),
      exec_jcc hcb]
    rfl
  let s5 := { s4 with rip := 4201799 }
  have hr5 : RegsOk s5 := hr4
  have hs5_4 : s5.reg 4 = some sp := by
    simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 4), List.getElem?_set_ne (by decide : 0 ≠ 4)]
    exact hsp
  have hs5_15 : s5.reg 15 = some ctxBase := by
    simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 15), List.getElem?_set_ne (by decide : 0 ≠ 15)]
    exact hr15
  have hs5_12 : s5.reg 12 = some v := by
    simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 12), List.getElem?_set_ne (by decide : 0 ≠ 12)]
    exact hr12
  obtain ⟨m', sf, hreach, hripf, hkf, hmemf, hwf', hregsf, hspf, houtf, hctxf, hothf, hregoth⟩ :=
    fn_out_push_u64_join_spec (s := s5) (sp := sp) (retAddr := retAddr) (ctxBase := ctxBase)
      (outBase := outBase) (outLen := outLen) (v := v) (ctx := ctx) (outReg := outReg) (stk := stk)
      rfl hr5 hs5_4 hs5_15 hs5_12 hctx hctx_base hctx_rw hctx_len hout hout_base hout_rw hne
      hstk hstkne_out hstkne_ctx hstk_lo hstk_hi (by omega) hctxW (by omega) (by omega) hmwf
      hret hc200 hc208
  refine ⟨m', sf,
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
      (Reaches.step_trans h05 hreach)))),
    hripf, hkf, hmemf, hwf', hregsf, hspf, houtf, hctxf, hothf, ?_⟩
  intro r h0 h1 h4
  rw [hregoth r h0 h1 h4]
  simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
    List.getElem?_set_ne h1.symm, List.getElem?_set_ne h0.symm]

theorem fn_out_push_u64_nogrow_spec {s : X86Vm.St} {sp retAddr ctxBase outBase outLen outCap v : Nat}
    {ctx outReg stk : Region}
    (hrip : s.rip = SporosVmBin.fn_out_push_u64)
    (hregs : RegsOk s)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr7 : s.reg 7 = some v)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_rw : ctx.rw = true)
    (hctx_len : ctxSize ≤ ctx.data.length)
    (hout : outReg ∈ s.mem) (hout_base : outReg.base = outBase) (hout_rw : outReg.rw = true)
    (hne : outBase ≠ ctxBase)
    (hstk : stk ∈ s.mem) (hstkne_out : stk.base ≠ outBase) (hstkne_ctx : stk.base ≠ ctxBase)
    (hstk_lo : stk.base ≤ sp) (hstk_hi : sp + 8 ≤ stk.hi)
    (hle : outLen + 8 ≤ outCap)
    (hout_hi : outCap ≤ outReg.data.length)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W) (houtW : outBase + outCap ≤ X86Env.W)
    (houtCapW : outCap < X86Env.W)
    (hmwf : s.mem.wf)
    (hret : rd64 s.mem sp = some retAddr)
    (hc200 : leVal ((ctx.data.drop 200).take 8) = outBase)
    (hc208 : leVal ((ctx.data.drop 208).take 8) = outLen)
    (hc216 : leVal ((ctx.data.drop 216).take 8) = outCap) :
    let outData' := writeBytes outReg.data outLen (bytes8 v)
    let ctxData' := writeBytes ctx.data 208 (bytes8 (outLen + 8))
    ∃ m' s', Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.k = s.k ∧ s'.mem = m' ∧ m'.wf ∧ RegsOk s' ∧
      s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      { outReg with data := outData' } ∈ m' ∧
      { ctx with data := ctxData' } ∈ m' ∧
      (∀ u ∈ s.mem, u.base ≠ outBase → u.base ≠ ctxBase → u ∈ m') ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 4 → r ≠ 12 → s'.reg r = s.reg r) := by
  have h01 : X86Vm.step vmImage s = .ok { s.set 12 (some v) with rip := 4201727 } := by
    rw [step_at_chunk 24 hrip rfl (by decide : (SporosVmBin.fn_out_push_u64, (.ofInt (.mov 12 7)), 3) ∈ SporosVmBin.vmTable_24),
      exec_mov hr7, hrip]
    rfl
  let s1 := { s.set 12 (some v) with rip := 4201727 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) v
  have hs1_4 : s1.reg 4 = some sp := by
    simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 12 ≠ 4)]
    exact hsp
  have hs1_15 : s1.reg 15 = some ctxBase := by
    simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hs1_12 : s1.reg 12 = some v := by
    simp only [s1, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hregs.1]; decide)]
  obtain ⟨m', sf, hreach, hripf, hkf, hmemf, hwf', hregsf, hspf, houtf, hctxf, hothf, hregoth⟩ :=
    fn_out_push_u64_loop_done_spec (s := s1) (sp := sp) (retAddr := retAddr) (ctxBase := ctxBase)
      (outBase := outBase) (outLen := outLen) (outCap := outCap) (v := v) (ctx := ctx) (outReg := outReg) (stk := stk)
      rfl hr1 hs1_4 hs1_15 hs1_12 hctx hctx_base hctx_rw hctx_len hout hout_base hout_rw hne
      hstk hstkne_out hstkne_ctx hstk_lo hstk_hi hle hout_hi hctxW houtW houtCapW hmwf hret hc200 hc208 hc216
  refine ⟨m', sf, Reaches.step_trans h01 hreach, hripf, hkf, hmemf, hwf', hregsf, hspf, houtf, hctxf, hothf, ?_⟩
  intro r h0 h1 h4 h12
  rw [hregoth r h0 h1 h4]
  simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne h12.symm]

theorem fn_out_push_u64_loop_ok_spec {s : X86Vm.St} {sp retAddr ctxBase outBase outLen outCap v a : Nat}
    {ctx outReg stk : Region}
    (hrip : s.rip = 4201727)
    (hregs : RegsOk s)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr12 : s.reg 12 = some v)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_rw : ctx.rw = true)
    (hctx_len : ctxSize ≤ ctx.data.length)
    (hout : outReg ∈ s.mem) (hout_base : outReg.base = outBase) (hout_rw : outReg.rw = true)
    (hne : outBase ≠ ctxBase) (hane : a ≠ ctxBase) (hastk : a ≠ stk.base)
    (hstk : stk ∈ s.mem) (hstkrw : stk.rw = true) (hstkne_out : stk.base ≠ outBase) (hstkne_ctx : stk.base ≠ ctxBase)
    (h8 : 8 ≤ sp) (hspW : sp < X86Env.W)
    (hstk_lo : stk.base ≤ sp - 8) (hstk_hi : sp + 8 ≤ stk.hi)
    (hgt : outCap < outLen + 8) (hle2 : outLen + 8 ≤ outCap * 2) (hcap_pos : 0 < outCap)
    (hout_eq : outReg.data.length = outCap)
    (hout_pg : outCap % X86Env.page = 0)
    (houtCapW : outCap * 2 < X86Env.W)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W)
    (hmwf : s.mem.wf) (hpos : ∀ u ∈ s.mem, 0 < u.data.length)
    (hret : rd64 s.mem sp = some retAddr)
    (hc200 : leVal ((ctx.data.drop 200).take 8) = outBase)
    (hc208 : leVal ((ctx.data.drop 208).take 8) = outLen)
    (hc216 : leVal ((ctx.data.drop 216).take 8) = outCap)
    (hor : (s.k.next).1 = some (.at a))
    (ha_pg : a % X86Env.page = 0) (ha_pos : 0 < a) (ha_lim : a + outCap * 2 ≤ 2 ^ 47)
    (hfree : ∀ u ∈ s.mem, u.base ≠ outBase → a + outCap * 2 ≤ u.base ∨ u.hi ≤ a) :
    let outData' := writeBytes (outReg.data ++ List.replicate outCap 0) outLen (bytes8 v)
    let ctxData' := writeBytes (writeBytes (writeBytes ctx.data 200 (bytes8 a)) 216 (bytes8 (outCap * 2))) 208 (bytes8 (outLen + 8))
    ∃ m' s' stk', Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.k = (s.k.next).2 ∧ s'.mem = m' ∧ m'.wf ∧ RegsOk s' ∧
      s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      (⟨a, outData', true⟩ : Region) ∈ m' ∧
      { ctx with data := ctxData' } ∈ m' ∧
      stk' ∈ m' ∧ stk'.base = stk.base ∧ stk'.rw = true ∧ stk'.data.length = stk.data.length ∧
      (∀ j, j < sp - 8 - stk.base ∨ sp - stk.base ≤ j → stk'.data[j]? = stk.data[j]?) ∧
      (∀ u ∈ s.mem, u.base ≠ outBase → u.base ≠ ctxBase → u.base ≠ stk.base → u ∈ m') ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 4 → r ≠ 6 → r ≠ 7 → r ≠ 8 → r ≠ 9 → r ≠ 10 → r ≠ 11 → s'.reg r = s.reg r) := by
  have hcs : ctxSize = 4096 := rfl
  have hea_c200 : ea ctxBase 200 = ctxBase + 200 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 200 < X86Env.W)]
  have hea_c208 : ea ctxBase 208 = ctxBase + 208 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 208 < X86Env.W)]
  have hea_c216 : ea ctxBase 216 = ctxBase + 216 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 216 < X86Env.W)]
  have hrd_c208 : rd64 s.mem (ea ctxBase 208) = some outLen := by
    rw [hea_c208]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc208)
  have h01 : X86Vm.step vmImage s = .ok { s.set 0 (some outLen) with rip := 4201734 } := by
    rw [step_at_chunk 24 hrip rfl (by decide : (4201727, (.ofInt (.ld 0 15 208)), 7) ∈ SporosVmBin.vmTable_24),
      exec_ld hr15 hrd_c208, hrip]
  let s1 := { s.set 0 (some outLen) with rip := 4201734 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) outLen
  have hs1_0 : s1.reg 0 = some outLen := by
    simp only [s1, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hregs.1]; decide)]
  have h02 : X86Vm.step vmImage s1 =
      .ok { s1.set 0 (some (outLen + 8)) with
            fl := (X86Int.aluF .add outLen 8).2, rip := 4201738 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201734, (.ofInt (.aluI .add 0 8)), 4) ∈ SporosVmBin.vmTable_24),
      exec_aluI_add_lt hs1_0 (by decide), Nat.mod_eq_of_lt (by omega : outLen + 8 < X86Env.W)]
  let s2 := { s1.set 0 (some (outLen + 8)) with
                fl := (X86Int.aluF .add outLen 8).2, rip := 4201738 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) _
  have hs2_15 : s2.reg 15 = some ctxBase := by
    simp only [s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 15)]
    exact hr15
  have hrd_c216 : rd64 s2.mem (ea ctxBase 216) = some outCap := by
    rw [hea_c216]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc216)
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 1 (some outCap) with rip := 4201745 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201738, (.ofInt (.ld 1 15 216)), 7) ∈ SporosVmBin.vmTable_24),
      exec_ld hs2_15 hrd_c216]
  let s3 := { s2.set 1 (some outCap) with rip := 4201745 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) outCap
  have hs3_0 : s3.reg 0 = some (outLen + 8) := by
    simp only [s3, s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0)]
    rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  have hs3_1 : s3.reg 1 = some outCap := by
    simp only [s3, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
  have h04 : X86Vm.step vmImage s3 =
      .ok { s3 with fl := (X86Int.aluF .sub (outLen + 8) outCap).2, rip := 4201748 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201745, (.ofInt (.cmp 0 1)), 3) ∈ SporosVmBin.vmTable_24),
      exec_cmp hs3_0 hs3_1]
  let s4 := { s3 with fl := (X86Int.aluF .sub (outLen + 8) outCap).2, rip := 4201748 }
  have hr4 : RegsOk s4 := hr3
  have hnotle : ¬ outLen + 8 ≤ outCap := by omega
  have hcb : X86Int.cond s4.fl 6 = some false := by
    dsimp [s4]
    rw [cond_aluF_sub_be _ _ (by omega : outLen + 8 < X86Env.W) (by omega : outCap < X86Env.W), decide_eq_false hnotle]
  have h05 : X86Vm.step vmImage s4 = .ok { s4 with rip := 4201750 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201748, (.ofInt (.jcc 6 4201799)), 2) ∈ SporosVmBin.vmTable_24),
      exec_jcc hcb]
    rfl
  let s5 := { s4 with rip := 4201750 }
  have hr5 : RegsOk s5 := hr4
  have hs5_15 : s5.reg 15 = some ctxBase := by
    simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 15), List.getElem?_set_ne (by decide : 0 ≠ 15)]
    exact hr15
  have hrd_c200 : rd64 s5.mem (ea ctxBase 200) = some outBase := by
    rw [hea_c200]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc200)
  have h06 : X86Vm.step vmImage s5 = .ok { s5.set 7 (some outBase) with rip := 4201757 } := by
    rw [step_at_chunk 25 rfl rfl (by decide : (4201750, (.ofInt (.ld 7 15 200)), 7) ∈ SporosVmBin.vmTable_25),
      exec_ld hs5_15 hrd_c200]
  let s6 := { s5.set 7 (some outBase) with rip := 4201757 }
  have hr6' : RegsOk s6 := regsOk_set hr5 (by decide) outBase
  have hs6_1 : s6.reg 1 = some outCap := by
    simp only [s6, s5, s4, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 7 ≠ 1)]
    exact hs3_1
  have h07 : X86Vm.step vmImage s6 = .ok { s6.set 6 (some outCap) with rip := 4201760 } := by
    rw [step_at_chunk 25 rfl rfl (by decide : (4201757, (.ofInt (.mov 6 1)), 3) ∈ SporosVmBin.vmTable_25),
      exec_mov hs6_1]
  let s7 := { s6.set 6 (some outCap) with rip := 4201760 }
  have hr7' : RegsOk s7 := regsOk_set hr6' (by decide) outCap
  have hs7_1 : s7.reg 1 = some outCap := by
    simp only [s7, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 6 ≠ 1)]
    exact hs6_1
  have h08 : X86Vm.step vmImage s7 = .ok { s7.set 2 (some outCap) with rip := 4201763 } := by
    rw [step_at_chunk 25 rfl rfl (by decide : (4201760, (.ofInt (.mov 2 1)), 3) ∈ SporosVmBin.vmTable_25),
      exec_mov hs7_1]
  let s8 := { s7.set 2 (some outCap) with rip := 4201763 }
  have hr8 : RegsOk s8 := regsOk_set hr7' (by decide) outCap
  have hs8_2 : s8.reg 2 = some outCap := by
    simp only [s8, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr7'.1]; decide)]
  have h09 : X86Vm.step vmImage s8 =
      .ok { s8.set 2 (some (outCap * 2)) with fl := X86Int.Flags.undef, rip := 4201767 } := by
    rw [step_at_chunk 25 rfl rfl (by decide : (4201763, (.ofInt (.shlI 2 1)), 4) ∈ SporosVmBin.vmTable_25),
      exec_shlI_1_lt hs8_2 (by omega)]
  let s9 := { s8.set 2 (some (outCap * 2)) with fl := X86Int.Flags.undef, rip := 4201767 }
  have hr9 : RegsOk s9 := regsOk_set hr8 (by decide) _
  have hs9_4 : s9.reg 4 = some sp := by
    simp only [s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 2 ≠ 4), List.getElem?_set_ne (by decide : 6 ≠ 4),
      List.getElem?_set_ne (by decide : 7 ≠ 4), List.getElem?_set_ne (by decide : 1 ≠ 4),
      List.getElem?_set_ne (by decide : 0 ≠ 4)]
    exact hsp
  have hea_sp : (sp + X86Env.W - 8) % X86Env.W = sp - 8 := by
    have : sp + X86Env.W - 8 = (sp - 8) + X86Env.W := by omega
    rw [this, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  obtain ⟨istk, m10, hwr10, hget10, hm10eq, hwf10, hstk10, hoth10⟩ :=
    wr64_region hmwf hstk hstkrw hstk_lo (by omega : sp - 8 + 8 ≤ stk.hi) (v := 4201772)
  have hwr10' : wr64 s9.mem ((sp + X86Env.W - 8) % X86Env.W) 4201772 = some m10 := by
    simpa [s9, s8, s7, s6, s5, s4, s3, s2, s1, hea_sp] using hwr10
  have h10 : X86Vm.step vmImage s9 =
      .ok { s9.set 4 (some (sp - 8)) with mem := m10, rip := 4201363 } := by
    rw [step_at_chunk 25 rfl rfl (by decide : (4201767, (.ofInt (.call 4201363)), 5) ∈ SporosVmBin.vmTable_25),
      exec_call hs9_4 hwr10']
    simp [s9, hea_sp]
  let s10 := { s9.set 4 (some (sp - 8)) with mem := m10, rip := 4201363 }
  have hr10 : RegsOk s10 := regsOk_set hr9 (by decide) _
  let stk1 := { stk with data := writeBytes stk.data (sp - 8 - stk.base) (bytes8 4201772) }
  have hstk1_mem : stk1 ∈ m10 := hstk10
  have hstk1_hi : stk1.hi = stk.hi := by simp [stk1, Region.hi, writeBytes_length]
  have hret10 : rd64 s10.mem (sp - 8) = some 4201772 := by
    have hr := rd64_of_writeBytes_same hwf10 hstk10 hstk_lo (by omega : sp - 8 + 8 ≤ stk.hi)
    simpa [s10, show (4201772 % X86Env.W) = 4201772 from by decide] using hr
  have hout_m10 : outReg ∈ m10 := hoth10 outReg hout (by rw [hout_base]; exact hstkne_out.symm)
  have hctx_m10 : ctx ∈ m10 := hoth10 ctx hctx (by rw [hctx_base]; exact hstkne_ctx.symm)
  obtain ⟨idx, hfind10⟩ := mem_find_of_mem m10 outReg outBase 0 hwf10.2 hout_m10
    (by simp [hout_base]) (by simp [Region.hi, hout_base, hout_eq]; omega)
  have hfind10' : m10.find outBase 0 = some (idx, outReg) := by simpa using hfind10
  have hget_idx := (mem_find_idx_lt m10 outBase 0 idx outReg hfind10').2.2
  simp only [Nat.sub_zero] at hget_idx
  have hpos10 : ∀ u ∈ m10, 0 < u.data.length := by
    intro u hu
    rw [hm10eq] at hu
    rcases List.mem_or_eq_of_mem_set hu with hu0 | rfl
    · exact hpos u hu0
    · simpa [writeBytes_length] using hpos stk hstk
  have hpg_1 : X86Env.pages outCap = outCap := pages_of_dvd hout_pg
  have hout_pg2 : (outCap * 2) % X86Env.page = 0 := by
    rw [Nat.mul_mod, hout_pg]; decide
  have hpg_2 : X86Env.pages (outCap * 2) = outCap * 2 := pages_of_dvd hout_pg2
  have hfree10 : X86Env.Mem.free (m10.eraseIdx idx) a (X86Env.pages (outCap * 2)) = true := by
    rw [hpg_2]
    refine mem_eraseIdx_free hwf10 hget_idx hpos10 ?_
    intro u hu hub
    have hne_out : u.base ≠ outBase := by rw [← hout_base]; exact hub
    rw [hm10eq] at hu
    rcases List.mem_or_eq_of_mem_set hu with hu0 | rfl
    · exact hfree u hu0 hne_out
    · simp only [Region.hi, writeBytes_length]
      exact hfree stk hstk hstkne_out
  have hs10_4 : s10.reg 4 = some (sp - 8) := by
    simp only [s10, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr9.1]; decide)]
  have hs10_7 : s10.reg 7 = some outBase := by
    simp only [s10, s9, s8, s7, s6, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 4 ≠ 7), List.getElem?_set_ne (by decide : 2 ≠ 7),
      List.getElem?_set_ne (by decide : 6 ≠ 7)]
    rw [getElem?_set_getD_self (by rw [hr5.1]; decide)]
  have hs10_6 : s10.reg 6 = some outCap := by
    simp only [s10, s9, s8, s7, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 4 ≠ 6), List.getElem?_set_ne (by decide : 2 ≠ 6)]
    rw [getElem?_set_getD_self (by rw [hr6'.1]; decide)]
  have hs10_2 : s10.reg 2 = some (outCap * 2) := by
    simp only [s10, s9, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 4 ≠ 2)]
    rw [getElem?_set_getD_self (by rw [hr8.1]; decide)]
  obtain ⟨s11, hrm, hrip11, hk11, hmem11, hm11wf, hr11', hs11_0, hs11_4, hoth11⟩ :=
    fn_mremap_ok_spec (s := s10) (sp := sp - 8) (retAddr := 4201772)
      (oldPtr := outBase) (oldLen := outCap) (newLen := outCap * 2)
      (idx := idx) (a := a) (oldReg := outReg) (stk := stk1)
      rfl hr10 hs10_4 hstk1_mem (by simp [stk1]; exact hstkne_out) hstk_lo (by rw [hstk1_hi]; omega)
      hret10 hwf10 hs10_7 hs10_6 hs10_2 hfind10' hout_base (by rw [hpg_1, hout_eq]) hout_rw
      (by rw [hpg_2, hout_eq]; omega) hor ha_pg ha_pos (by rw [hpg_2]; exact ha_lim) hfree10
  have hm11_eq_rep : X86Env.pages (outCap * 2) - outReg.data.length = outCap := by
    rw [hpg_2, hout_eq]; omega
  let outNew : Region := ⟨a, outReg.data ++ List.replicate outCap 0, true⟩
  have houtNew_m11 : outNew ∈ s11.mem := by
    rw [hmem11]
    refine List.mem_append_right _ ?_
    simp [outNew, hm11_eq_rep]
  have hctx_m11 : ctx ∈ s11.mem := by
    rw [hmem11]
    exact List.mem_append_left _ (mem_eraseIdx_other hget_idx hctx_m10 (by rw [hout_base, hctx_base]; exact hne.symm))
  have hstk1_m11 : stk1 ∈ s11.mem := by
    rw [hmem11]
    exact List.mem_append_left _ (mem_eraseIdx_other hget_idx hstk1_mem (by simp [stk1, hout_base]; exact hstkne_out))
  have hs11_4' : s11.reg 4 = some sp := by
    have : (sp - 8 + 8) % X86Env.W = sp := by rw [Nat.sub_add_cancel h8, Nat.mod_eq_of_lt hspW]
    rwa [this] at hs11_4
  have hs11_15 : s11.reg 15 = some ctxBase := by
    rw [hoth11 15 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    simp only [s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 4 ≠ 15), List.getElem?_set_ne (by decide : 2 ≠ 15),
      List.getElem?_set_ne (by decide : 6 ≠ 15), List.getElem?_set_ne (by decide : 7 ≠ 15),
      List.getElem?_set_ne (by decide : 1 ≠ 15), List.getElem?_set_ne (by decide : 0 ≠ 15)]
    exact hr15
  have hs11_12 : s11.reg 12 = some v := by
    rw [hoth11 12 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
    simp only [s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 4 ≠ 12), List.getElem?_set_ne (by decide : 2 ≠ 12),
      List.getElem?_set_ne (by decide : 6 ≠ 12), List.getElem?_set_ne (by decide : 7 ≠ 12),
      List.getElem?_set_ne (by decide : 1 ≠ 12), List.getElem?_set_ne (by decide : 0 ≠ 12)]
    exact hr12
  have hm11wf' : s11.mem.wf := by rw [hmem11]; exact hm11wf
  obtain ⟨_, m12, hwr12, _, _, hwf12, hctx12, hoth12⟩ :=
    wr64_region hm11wf' hctx_m11 hctx_rw (a := ctxBase + 200) (v := a)
      (by rw [hctx_base]; omega) (by simp [Region.hi, hctx_base]; omega)
  have hwr12' : wr64 s11.mem (ea ctxBase 200) a = some m12 := by simpa [hea_c200] using hwr12
  have h12 : X86Vm.step vmImage s11 = .ok { s11 with mem := m12, rip := 4201779 } := by
    rw [step_at_chunk 25 hrip11 rfl (by decide : (4201772, (.ofInt (.st 15 200 0)), 7) ∈ SporosVmBin.vmTable_25),
      exec_st hs11_15 hs11_0 hwr12', hrip11]
  let s12 := { s11 with mem := m12, rip := 4201779 }
  have hr12' : RegsOk s12 := hr11'
  have hs12_15 : s12.reg 15 = some ctxBase := hs11_15
  let ctx12 := { ctx with data := writeBytes ctx.data (ctxBase + 200 - ctx.base) (bytes8 a) }
  have hctx12_hc216 : leVal ((ctx12.data.drop 216).take 8) = outCap := by
    simp only [ctx12, hctx_base, Nat.add_sub_cancel_left]
    rw [writeBytes_drop_take_disjoint ctx.data 200 216 8 (bytes8 a) (by right; rw [bytes8_length]; omega) (by omega)]
    exact hc216
  have hrd12_c216 : rd64 s12.mem (ea ctxBase 216) = some outCap := by
    rw [hea_c216]
    exact rd64_of_mem hwf12 hctx12 (by simp [hctx_base])
      (by simp [Region.hi, writeBytes_length, hctx_base]; omega)
      (by simpa [ctx12, hctx_base] using hctx12_hc216)
  have h13 : X86Vm.step vmImage s12 = .ok { s12.set 1 (some outCap) with rip := 4201786 } := by
    rw [step_at_chunk 25 rfl rfl (by decide : (4201779, (.ofInt (.ld 1 15 216)), 7) ∈ SporosVmBin.vmTable_25),
      exec_ld hs12_15 hrd12_c216]
  let s13 := { s12.set 1 (some outCap) with rip := 4201786 }
  have hr13' : RegsOk s13 := regsOk_set hr12' (by decide) outCap
  have hs13_1 : s13.reg 1 = some outCap := by
    simp only [s13, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr12'.1]; decide)]
  have h14 : X86Vm.step vmImage s13 =
      .ok { s13.set 1 (some (outCap * 2)) with fl := X86Int.Flags.undef, rip := 4201790 } := by
    rw [step_at_chunk 25 rfl rfl (by decide : (4201786, (.ofInt (.shlI 1 1)), 4) ∈ SporosVmBin.vmTable_25),
      exec_shlI_1_lt hs13_1 (by omega)]
  let s14 := { s13.set 1 (some (outCap * 2)) with fl := X86Int.Flags.undef, rip := 4201790 }
  have hr14 : RegsOk s14 := regsOk_set hr13' (by decide) _
  have hs14_15 : s14.reg 15 = some ctxBase := by
    simp only [s14, s13, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 15)]
    exact hs12_15
  have hs14_1 : s14.reg 1 = some (outCap * 2) := by
    simp only [s14, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr13'.1]; decide)]
  obtain ⟨_, m15, hwr15, _, _, hwf15, hctx15, hoth15⟩ :=
    wr64_region hwf12 hctx12 hctx_rw (a := ctxBase + 216) (v := outCap * 2)
      (by simp [hctx_base])
      (by simp [Region.hi, writeBytes_length, hctx_base]; omega)
  have hwr15' : wr64 s14.mem (ea ctxBase 216) (outCap * 2) = some m15 := by simpa [s14, s13, hea_c216] using hwr15
  have h15 : X86Vm.step vmImage s14 = .ok { s14 with mem := m15, rip := 4201797 } := by
    rw [step_at_chunk 25 rfl rfl (by decide : (4201790, (.ofInt (.st 15 216 1)), 7) ∈ SporosVmBin.vmTable_25),
      exec_st hs14_15 hs14_1 hwr15']
  let s15 := { s14 with mem := m15, rip := 4201797 }
  have hr15' : RegsOk s15 := hr14
  have h16 : X86Vm.step vmImage s15 = .ok { s15 with rip := 4201727 } := by
    rw [step_at_chunk 25 rfl rfl (by decide : (4201797, (.ofInt (.jmp 4201727)), 2) ∈ SporosVmBin.vmTable_25),
      exec_jmp]
  let s16 := { s15 with rip := 4201727 }
  have hr16 : RegsOk s16 := hr15'
  have houtNew_m12 : outNew ∈ m12 := hoth12 outNew houtNew_m11 (by simp [outNew, hctx_base]; exact hane)
  have houtNew_m15 : outNew ∈ m15 := hoth15 outNew houtNew_m12 (by simp [outNew, hctx_base]; exact hane)
  have hstk1_m12 : stk1 ∈ m12 := hoth12 stk1 hstk1_m11 (by simp [stk1, hctx_base]; exact hstkne_ctx)
  have hstk1_m15 : stk1 ∈ m15 := hoth15 stk1 hstk1_m12 (by simp [stk1, hctx_base]; exact hstkne_ctx)
  let ctx15 := { ctx12 with data := writeBytes ctx12.data (ctxBase + 216 - ctx12.base) (bytes8 (outCap * 2)) }
  have hctx15_hc200 : leVal ((ctx15.data.drop 200).take 8) = a := by
    simp only [ctx15, ctx12, hctx_base, Nat.add_sub_cancel_left]
    rw [writeBytes_drop_take_disjoint _ 216 200 8 (bytes8 (outCap * 2)) (by left; omega) (by rw [writeBytes_length]; omega)]
    rw [leVal_writeBytes_same _ 200 a (by omega)]
    exact Nat.mod_eq_of_lt (by have := vm_lim_lt_W; omega)
  have hctx15_hc208 : leVal ((ctx15.data.drop 208).take 8) = outLen := by
    simp only [ctx15, ctx12, hctx_base, Nat.add_sub_cancel_left]
    rw [writeBytes_drop_take_disjoint _ 216 208 8 (bytes8 (outCap * 2)) (by left; omega) (by rw [writeBytes_length]; omega)]
    rw [writeBytes_drop_take_disjoint ctx.data 200 208 8 (bytes8 a) (by right; rw [bytes8_length]; omega) (by omega)]
    exact hc208
  have hctx15_hc216 : leVal ((ctx15.data.drop 216).take 8) = outCap * 2 := by
    simp only [ctx15, ctx12, hctx_base, Nat.add_sub_cancel_left]
    rw [leVal_writeBytes_same _ 216 (outCap * 2) (by rw [writeBytes_length]; omega)]
    exact Nat.mod_eq_of_lt houtCapW
  have hret_stk1 : rd64 s16.mem sp = some retAddr := by
    have hstk1_drop : (stk1.data.drop (sp - stk1.base)).take 8 = (stk.data.drop (sp - stk.base)).take 8 := by
      simp only [stk1]
      apply writeBytes_drop_take_disjoint
      · right; rw [bytes8_length]; omega
      · simp [Region.hi] at hstk_hi; omega
    obtain ⟨rv, hrv⟩ := show ∃ rv, leVal ((stk.data.drop (sp - stk.base)).take 8) = rv from ⟨_, rfl⟩
    have hold := rd64_of_mem hmwf hstk (by omega) hstk_hi hrv
    rw [hret, Option.some.injEq] at hold
    subst hold
    exact rd64_of_mem hwf15 hstk1_m15 (by simp [stk1]; omega)
      (by simp [stk1, Region.hi, writeBytes_length] at hstk_hi ⊢; omega) (by rw [hstk1_drop, hrv])
  have hctx15_hi : ctxSize ≤ ctx15.data.length := by
    simp [ctx15, ctx12, writeBytes_length]; exact hctx_len
  have hs16_4 : s16.reg 4 = some sp := by
    simp only [s16, s15, s14, s13, s12, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 4)]
    exact hs11_4'
  have hs16_15 : s16.reg 15 = some ctxBase := hs14_15
  have hs16_12 : s16.reg 12 = some v := by
    simp only [s16, s15, s14, s13, s12, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 12)]
    exact hs11_12
  obtain ⟨m', sf, hreach_done, hripf, hkf, hmemf, hwf', hregsf, hspf, houtf, hctxf, hothf, hregoth⟩ :=
    fn_out_push_u64_loop_done_spec (s := s16) (sp := sp) (retAddr := retAddr) (ctxBase := ctxBase)
      (outBase := a) (outLen := outLen) (outCap := outCap * 2) (v := v) (ctx := ctx15) (outReg := outNew) (stk := stk1)
      rfl hr16 hs16_4 hs16_15 hs16_12
      (by simpa [ctx15] using hctx15) (by simp [ctx15, ctx12, hctx_base]) hctx_rw hctx15_hi
      (by simpa [s16, s15] using houtNew_m15) rfl rfl hane
      (by simpa [s16, s15] using hstk1_m15) (by simp [stk1]; exact hastk.symm)
      (by simp [stk1]; exact hstkne_ctx) (by simp [stk1]; omega)
      (by simp [stk1, Region.hi, writeBytes_length] at hstk_hi ⊢; omega)
      hle2 (by simp [outNew, List.length_append, List.length_replicate, hout_eq]; omega)
      hctxW (by have := vm_lim_lt_W; omega) houtCapW hwf15 hret_stk1 hctx15_hc200 hctx15_hc208 hctx15_hc216
  have hstk1_m' : stk1 ∈ m' := by
    apply hothf stk1 (by simpa [s16, s15] using hstk1_m15)
    · simp [stk1]; exact hastk.symm
    · simp [stk1]; exact hstkne_ctx
  refine ⟨m', sf, stk1,
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
      (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step_trans h08
        (Reaches.step_trans h09 (Reaches.step_trans h10
          (Reaches.trans hrm (Reaches.step_trans h12 (Reaches.step_trans h13 (Reaches.step_trans h14
            (Reaches.step_trans h15 (Reaches.step_trans h16 hreach_done))))))))))))))),
    hripf, by simpa [s16, s15, s14, s13, s12, hk11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1] using hkf, hmemf, hwf', hregsf, hspf,
    by simpa [outNew] using houtf,
    by simpa [ctx15, ctx12, hctx_base, Nat.add_sub_cancel_left] using hctxf,
    hstk1_m', rfl, hstkrw, writeBytes_length _ _ _, ?_, ?_, ?_⟩
  · intro j hj
    apply writeBytes_getElem?_disjoint
    rw [bytes8_length]
    rcases hj with h | h
    · exact Or.inl h
    · exact Or.inr (by omega)
  · intro u hu huout huctx hustk
    have hu10 : u ∈ m10 := hoth10 u hu hustk
    have hu11 : u ∈ s11.mem := by
      rw [hmem11]
      exact List.mem_append_left _ (mem_eraseIdx_other hget_idx hu10 (by rw [hout_base]; exact huout))
    have hu12 : u ∈ m12 := hoth12 u hu11 (by rw [hctx_base]; exact huctx)
    have hu15 : u ∈ m15 := hoth15 u hu12 (by simp [hctx_base]; exact huctx)
    have hua : u.base ≠ a := by
      intro hba
      have hupos := hpos u hu
      rcases hfree u hu huout with h1 | h2
      · omega
      · simp [Region.hi] at h2; omega
    exact hothf u (by simpa [s16, s15] using hu15) hua huctx
  · intro r h0 h1 h2 h4 h6 h7 h8 h9 h10' h11'
    have h16_11 : s16.reg r = s11.reg r := by
      simp only [s16, s15, s14, s13, s12, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne h1.symm]
    rw [hregoth r h0 h1 h4, h16_11,
      hoth11 r h0 h1 h2 h4 h6 h7 h8 h9 h10' h11']
    simp only [s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne h4.symm, List.getElem?_set_ne h2.symm, List.getElem?_set_ne h6.symm,
      List.getElem?_set_ne h7.symm, List.getElem?_set_ne h1.symm, List.getElem?_set_ne h0.symm]

theorem fn_out_push_u64_spec {s : X86Vm.St} {sp retAddr ctxBase outBase outLen outCap v a : Nat}
    {ctx outReg stk : Region}
    (hrip : s.rip = SporosVmBin.fn_out_push_u64)
    (hregs : RegsOk s)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr7 : s.reg 7 = some v)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_rw : ctx.rw = true)
    (hctx_len : ctxSize ≤ ctx.data.length)
    (hout : outReg ∈ s.mem) (hout_base : outReg.base = outBase) (hout_rw : outReg.rw = true)
    (hne : outBase ≠ ctxBase) (hane : a ≠ ctxBase) (hastk : a ≠ stk.base)
    (hstk : stk ∈ s.mem) (hstkrw : stk.rw = true) (hstkne_out : stk.base ≠ outBase) (hstkne_ctx : stk.base ≠ ctxBase)
    (h8 : 8 ≤ sp) (hspW : sp < X86Env.W)
    (hstk_lo : stk.base ≤ sp - 8) (hstk_hi : sp + 8 ≤ stk.hi)
    (hgt : outCap < outLen + 8) (hle2 : outLen + 8 ≤ outCap * 2) (hcap_pos : 0 < outCap)
    (hout_eq : outReg.data.length = outCap)
    (hout_pg : outCap % X86Env.page = 0)
    (houtCapW : outCap * 2 < X86Env.W)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W)
    (hmwf : s.mem.wf) (hpos : ∀ u ∈ s.mem, 0 < u.data.length)
    (hret : rd64 s.mem sp = some retAddr)
    (hc200 : leVal ((ctx.data.drop 200).take 8) = outBase)
    (hc208 : leVal ((ctx.data.drop 208).take 8) = outLen)
    (hc216 : leVal ((ctx.data.drop 216).take 8) = outCap)
    (hor : (s.k.next).1 = some (.at a))
    (ha_pg : a % X86Env.page = 0) (ha_pos : 0 < a) (ha_lim : a + outCap * 2 ≤ 2 ^ 47)
    (hfree : ∀ u ∈ s.mem, u.base ≠ outBase → a + outCap * 2 ≤ u.base ∨ u.hi ≤ a) :
    let outData' := writeBytes (outReg.data ++ List.replicate outCap 0) outLen (bytes8 v)
    let ctxData' := writeBytes (writeBytes (writeBytes ctx.data 200 (bytes8 a)) 216 (bytes8 (outCap * 2))) 208 (bytes8 (outLen + 8))
    ∃ m' s' stk', Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.k = (s.k.next).2 ∧ s'.mem = m' ∧ m'.wf ∧ RegsOk s' ∧
      s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      (⟨a, outData', true⟩ : Region) ∈ m' ∧
      { ctx with data := ctxData' } ∈ m' ∧
      stk' ∈ m' ∧ stk'.base = stk.base ∧ stk'.rw = true ∧ stk'.data.length = stk.data.length ∧
      (∀ j, j < sp - 8 - stk.base ∨ sp - stk.base ≤ j → stk'.data[j]? = stk.data[j]?) ∧
      (∀ u ∈ s.mem, u.base ≠ outBase → u.base ≠ ctxBase → u.base ≠ stk.base → u ∈ m') ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 4 → r ≠ 6 → r ≠ 7 → r ≠ 8 → r ≠ 9 → r ≠ 10 → r ≠ 11 → r ≠ 12 → s'.reg r = s.reg r) := by
  have h01 : X86Vm.step vmImage s = .ok { s.set 12 (some v) with rip := 4201727 } := by
    rw [step_at_chunk 24 hrip rfl (by decide : (SporosVmBin.fn_out_push_u64, (.ofInt (.mov 12 7)), 3) ∈ SporosVmBin.vmTable_24),
      exec_mov hr7, hrip]
    rfl
  let s1 := { s.set 12 (some v) with rip := 4201727 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) v
  have hs1_4 : s1.reg 4 = some sp := by
    simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 12 ≠ 4)]
    exact hsp
  have hs1_15 : s1.reg 15 = some ctxBase := by
    simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  have hs1_12 : s1.reg 12 = some v := by
    simp only [s1, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hregs.1]; decide)]
  obtain ⟨m', sf, stk', hreach, hripf, hkf, hmemf, hwf', hregsf, hspf, houtf, hctxf,
      hstkm, hstkb, hstkrw', hstkl, hstkp, hothf, hregoth⟩ :=
    fn_out_push_u64_loop_ok_spec (s := s1) (sp := sp) (retAddr := retAddr) (ctxBase := ctxBase)
      (outBase := outBase) (outLen := outLen) (outCap := outCap) (v := v) (a := a)
      (ctx := ctx) (outReg := outReg) (stk := stk)
      rfl hr1 hs1_4 hs1_15 hs1_12 hctx hctx_base hctx_rw hctx_len hout hout_base hout_rw
      hne hane hastk hstk hstkrw hstkne_out hstkne_ctx h8 hspW hstk_lo hstk_hi
      hgt hle2 hcap_pos hout_eq hout_pg houtCapW hctxW hmwf hpos hret hc200 hc208 hc216
      hor ha_pg ha_pos ha_lim hfree
  refine ⟨m', sf, stk', Reaches.step_trans h01 hreach, hripf, hkf, hmemf, hwf', hregsf, hspf,
    houtf, hctxf, hstkm, hstkb, hstkrw', hstkl, hstkp, hothf, ?_⟩
  intro r h0 h1 h2 h4 h6 h7 h8' h9 h10 h11 h12
  rw [hregoth r h0 h1 h2 h4 h6 h7 h8' h9 h10 h11]
  simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne h12.symm]

theorem fn_out_push_u64_loop_nomem_spec {s : X86Vm.St} {sp ctxBase outBase outLen outCap : Nat}
    {ctx outReg stk : Region}
    (hrip : s.rip = 4201727)
    (hregs : RegsOk s)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_len : ctxSize ≤ ctx.data.length)
    (hout : outReg ∈ s.mem) (hout_base : outReg.base = outBase) (hout_rw : outReg.rw = true)
    (hstk : stk ∈ s.mem) (hstkrw : stk.rw = true) (hstkne_out : stk.base ≠ outBase)
    (h16 : 16 ≤ sp) (hspW : sp < X86Env.W)
    (hstk_lo : stk.base ≤ sp - 16) (hstk_hi : sp ≤ stk.hi)
    (hgt : outCap < outLen + 8) (hle2 : outLen + 8 ≤ outCap * 2) (hcap_pos : 0 < outCap)
    (hout_eq : outReg.data.length = outCap)
    (hout_pg : outCap % X86Env.page = 0)
    (houtCapW : outCap * 2 < X86Env.W)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W)
    (hmwf : s.mem.wf)
    (hc200 : leVal ((ctx.data.drop 200).take 8) = outBase)
    (hc208 : leVal ((ctx.data.drop 208).take 8) = outLen)
    (hc216 : leVal ((ctx.data.drop 216).take 8) = outCap)
    (hor : (s.k.next).1 = some .nomem) :
    Ends vmImage s (.exit 2) := by
  have hcs : ctxSize = 4096 := rfl
  have hea_c200 : ea ctxBase 200 = ctxBase + 200 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 200 < X86Env.W)]
  have hea_c208 : ea ctxBase 208 = ctxBase + 208 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 208 < X86Env.W)]
  have hea_c216 : ea ctxBase 216 = ctxBase + 216 := by
    simp [X86Int.ea, Nat.mod_eq_of_lt (by omega : ctxBase + 216 < X86Env.W)]
  have hrd_c208 : rd64 s.mem (ea ctxBase 208) = some outLen := by
    rw [hea_c208]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc208)
  have h01 : X86Vm.step vmImage s = .ok { s.set 0 (some outLen) with rip := 4201734 } := by
    rw [step_at_chunk 24 hrip rfl (by decide : (4201727, (.ofInt (.ld 0 15 208)), 7) ∈ SporosVmBin.vmTable_24),
      exec_ld hr15 hrd_c208, hrip]
  let s1 := { s.set 0 (some outLen) with rip := 4201734 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) outLen
  have hs1_0 : s1.reg 0 = some outLen := by
    simp only [s1, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hregs.1]; decide)]
  have h02 : X86Vm.step vmImage s1 =
      .ok { s1.set 0 (some (outLen + 8)) with
            fl := (X86Int.aluF .add outLen 8).2, rip := 4201738 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201734, (.ofInt (.aluI .add 0 8)), 4) ∈ SporosVmBin.vmTable_24),
      exec_aluI_add_lt hs1_0 (by decide), Nat.mod_eq_of_lt (by omega : outLen + 8 < X86Env.W)]
  let s2 := { s1.set 0 (some (outLen + 8)) with
                fl := (X86Int.aluF .add outLen 8).2, rip := 4201738 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) _
  have hs2_15 : s2.reg 15 = some ctxBase := by
    simp only [s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 15)]
    exact hr15
  have hrd_c216 : rd64 s2.mem (ea ctxBase 216) = some outCap := by
    rw [hea_c216]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc216)
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 1 (some outCap) with rip := 4201745 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201738, (.ofInt (.ld 1 15 216)), 7) ∈ SporosVmBin.vmTable_24),
      exec_ld hs2_15 hrd_c216]
  let s3 := { s2.set 1 (some outCap) with rip := 4201745 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) outCap
  have hs3_0 : s3.reg 0 = some (outLen + 8) := by
    simp only [s3, s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0)]
    rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  have hs3_1 : s3.reg 1 = some outCap := by
    simp only [s3, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
  have h04 : X86Vm.step vmImage s3 =
      .ok { s3 with fl := (X86Int.aluF .sub (outLen + 8) outCap).2, rip := 4201748 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201745, (.ofInt (.cmp 0 1)), 3) ∈ SporosVmBin.vmTable_24),
      exec_cmp hs3_0 hs3_1]
  let s4 := { s3 with fl := (X86Int.aluF .sub (outLen + 8) outCap).2, rip := 4201748 }
  have hr4 : RegsOk s4 := hr3
  have hnotle : ¬ outLen + 8 ≤ outCap := by omega
  have hcb : X86Int.cond s4.fl 6 = some false := by
    dsimp [s4]
    rw [cond_aluF_sub_be _ _ (by omega : outLen + 8 < X86Env.W) (by omega : outCap < X86Env.W), decide_eq_false hnotle]
  have h05 : X86Vm.step vmImage s4 = .ok { s4 with rip := 4201750 } := by
    rw [step_at_chunk 24 rfl rfl (by decide : (4201748, (.ofInt (.jcc 6 4201799)), 2) ∈ SporosVmBin.vmTable_24),
      exec_jcc hcb]
    rfl
  let s5 := { s4 with rip := 4201750 }
  have hr5 : RegsOk s5 := hr4
  have hs5_15 : s5.reg 15 = some ctxBase := by
    simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 15), List.getElem?_set_ne (by decide : 0 ≠ 15)]
    exact hr15
  have hrd_c200 : rd64 s5.mem (ea ctxBase 200) = some outBase := by
    rw [hea_c200]
    exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
      (by simp [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc200)
  have h06 : X86Vm.step vmImage s5 = .ok { s5.set 7 (some outBase) with rip := 4201757 } := by
    rw [step_at_chunk 25 rfl rfl (by decide : (4201750, (.ofInt (.ld 7 15 200)), 7) ∈ SporosVmBin.vmTable_25),
      exec_ld hs5_15 hrd_c200]
  let s6 := { s5.set 7 (some outBase) with rip := 4201757 }
  have hr6' : RegsOk s6 := regsOk_set hr5 (by decide) outBase
  have hs6_1 : s6.reg 1 = some outCap := by
    simp only [s6, s5, s4, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 7 ≠ 1)]
    exact hs3_1
  have h07 : X86Vm.step vmImage s6 = .ok { s6.set 6 (some outCap) with rip := 4201760 } := by
    rw [step_at_chunk 25 rfl rfl (by decide : (4201757, (.ofInt (.mov 6 1)), 3) ∈ SporosVmBin.vmTable_25),
      exec_mov hs6_1]
  let s7 := { s6.set 6 (some outCap) with rip := 4201760 }
  have hr7' : RegsOk s7 := regsOk_set hr6' (by decide) outCap
  have hs7_1 : s7.reg 1 = some outCap := by
    simp only [s7, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 6 ≠ 1)]
    exact hs6_1
  have h08 : X86Vm.step vmImage s7 = .ok { s7.set 2 (some outCap) with rip := 4201763 } := by
    rw [step_at_chunk 25 rfl rfl (by decide : (4201760, (.ofInt (.mov 2 1)), 3) ∈ SporosVmBin.vmTable_25),
      exec_mov hs7_1]
  let s8 := { s7.set 2 (some outCap) with rip := 4201763 }
  have hr8 : RegsOk s8 := regsOk_set hr7' (by decide) outCap
  have hs8_2 : s8.reg 2 = some outCap := by
    simp only [s8, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr7'.1]; decide)]
  have h09 : X86Vm.step vmImage s8 =
      .ok { s8.set 2 (some (outCap * 2)) with fl := X86Int.Flags.undef, rip := 4201767 } := by
    rw [step_at_chunk 25 rfl rfl (by decide : (4201763, (.ofInt (.shlI 2 1)), 4) ∈ SporosVmBin.vmTable_25),
      exec_shlI_1_lt hs8_2 (by omega)]
  let s9 := { s8.set 2 (some (outCap * 2)) with fl := X86Int.Flags.undef, rip := 4201767 }
  have hr9 : RegsOk s9 := regsOk_set hr8 (by decide) _
  have hs9_4 : s9.reg 4 = some sp := by
    simp only [s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 2 ≠ 4), List.getElem?_set_ne (by decide : 6 ≠ 4),
      List.getElem?_set_ne (by decide : 7 ≠ 4), List.getElem?_set_ne (by decide : 1 ≠ 4),
      List.getElem?_set_ne (by decide : 0 ≠ 4)]
    exact hsp
  have hea_sp : (sp + X86Env.W - 8) % X86Env.W = sp - 8 := by
    have : sp + X86Env.W - 8 = (sp - 8) + X86Env.W := by omega
    rw [this, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  obtain ⟨istk, m10, hwr10, _, _, hwf10, hstk10, hoth10⟩ :=
    wr64_region hmwf hstk hstkrw (by omega : stk.base ≤ sp - 8) (by omega : sp - 8 + 8 ≤ stk.hi) (v := 4201772)
  have hwr10' : wr64 s9.mem ((sp + X86Env.W - 8) % X86Env.W) 4201772 = some m10 := by
    simpa [s9, s8, s7, s6, s5, s4, s3, s2, s1, hea_sp] using hwr10
  have h10 : X86Vm.step vmImage s9 =
      .ok { s9.set 4 (some (sp - 8)) with mem := m10, rip := 4201363 } := by
    rw [step_at_chunk 25 rfl rfl (by decide : (4201767, (.ofInt (.call 4201363)), 5) ∈ SporosVmBin.vmTable_25),
      exec_call hs9_4 hwr10']
    simp [s9, hea_sp]
  let s10 := { s9.set 4 (some (sp - 8)) with mem := m10, rip := 4201363 }
  have hr10 : RegsOk s10 := regsOk_set hr9 (by decide) _
  let stk1 := { stk with data := writeBytes stk.data (sp - 8 - stk.base) (bytes8 4201772) }
  have hstk1_hi : stk1.hi = stk.hi := by simp [stk1, Region.hi, writeBytes_length]
  have hout_m10 : outReg ∈ m10 := hoth10 outReg hout (by rw [hout_base]; exact hstkne_out.symm)
  obtain ⟨idx, hfind10⟩ := mem_find_of_mem m10 outReg outBase 0 hwf10.2 hout_m10
    (by simp [hout_base]) (by simp [Region.hi, hout_base, hout_eq]; omega)
  have hfind10' : m10.find outBase 0 = some (idx, outReg) := by simpa using hfind10
  have hpg_1 : X86Env.pages outCap = outCap := pages_of_dvd hout_pg
  have hout_pg2 : (outCap * 2) % X86Env.page = 0 := by
    rw [Nat.mul_mod, hout_pg]; decide
  have hpg_2 : X86Env.pages (outCap * 2) = outCap * 2 := pages_of_dvd hout_pg2
  have hs10_4 : s10.reg 4 = some (sp - 8) := by
    simp only [s10, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr9.1]; decide)]
  have hs10_7 : s10.reg 7 = some outBase := by
    simp only [s10, s9, s8, s7, s6, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 4 ≠ 7), List.getElem?_set_ne (by decide : 2 ≠ 7),
      List.getElem?_set_ne (by decide : 6 ≠ 7)]
    rw [getElem?_set_getD_self (by rw [hr5.1]; decide)]
  have hs10_6 : s10.reg 6 = some outCap := by
    simp only [s10, s9, s8, s7, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 4 ≠ 6), List.getElem?_set_ne (by decide : 2 ≠ 6)]
    rw [getElem?_set_getD_self (by rw [hr6'.1]; decide)]
  have hs10_2 : s10.reg 2 = some (outCap * 2) := by
    simp only [s10, s9, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 4 ≠ 2)]
    rw [getElem?_set_getD_self (by rw [hr8.1]; decide)]
  have h_exit := fn_mremap_nomem_spec (s := s10) (sp := sp - 8)
    (oldPtr := outBase) (oldLen := outCap) (newLen := outCap * 2)
    (idx := idx) (oldReg := outReg) (stk := stk1)
    rfl hr10 hs10_4 (by omega) (by omega) hwf10 hstk10 hstkrw
    (by simp [stk1]; omega) (by rw [hstk1_hi]; omega)
    hs10_7 hs10_6 hs10_2 hfind10' hout_base (by rw [hpg_1, hout_eq]) hout_rw
    (by rw [hpg_2, hout_eq]; omega) hor
  exact Ends.step_trans h01 (Ends.step_trans h02 (Ends.step_trans h03 (Ends.step_trans h04
    (Ends.step_trans h05 (Ends.step_trans h06 (Ends.step_trans h07 (Ends.step_trans h08
      (Ends.step_trans h09 (Ends.step_trans h10 h_exit)))))))))

theorem fn_out_push_u64_nomem_spec {s : X86Vm.St} {sp ctxBase outBase outLen outCap v : Nat}
    {ctx outReg stk : Region}
    (hrip : s.rip = SporosVmBin.fn_out_push_u64)
    (hregs : RegsOk s)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr7 : s.reg 7 = some v)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_len : ctxSize ≤ ctx.data.length)
    (hout : outReg ∈ s.mem) (hout_base : outReg.base = outBase) (hout_rw : outReg.rw = true)
    (hstk : stk ∈ s.mem) (hstkrw : stk.rw = true) (hstkne_out : stk.base ≠ outBase)
    (h16 : 16 ≤ sp) (hspW : sp < X86Env.W)
    (hstk_lo : stk.base ≤ sp - 16) (hstk_hi : sp ≤ stk.hi)
    (hgt : outCap < outLen + 8) (hle2 : outLen + 8 ≤ outCap * 2) (hcap_pos : 0 < outCap)
    (hout_eq : outReg.data.length = outCap)
    (hout_pg : outCap % X86Env.page = 0)
    (houtCapW : outCap * 2 < X86Env.W)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W)
    (hmwf : s.mem.wf)
    (hc200 : leVal ((ctx.data.drop 200).take 8) = outBase)
    (hc208 : leVal ((ctx.data.drop 208).take 8) = outLen)
    (hc216 : leVal ((ctx.data.drop 216).take 8) = outCap)
    (hor : (s.k.next).1 = some .nomem) :
    Ends vmImage s (.exit 2) := by
  have h01 : X86Vm.step vmImage s = .ok { s.set 12 (some v) with rip := 4201727 } := by
    rw [step_at_chunk 24 hrip rfl (by decide : (SporosVmBin.fn_out_push_u64, (.ofInt (.mov 12 7)), 3) ∈ SporosVmBin.vmTable_24),
      exec_mov hr7, hrip]
    rfl
  let s1 := { s.set 12 (some v) with rip := 4201727 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) v
  have hs1_4 : s1.reg 4 = some sp := by
    simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 12 ≠ 4)]
    exact hsp
  have hs1_15 : s1.reg 15 = some ctxBase := by
    simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact hr15
  exact Ends.step_trans h01
    (fn_out_push_u64_loop_nomem_spec (s := s1) (sp := sp) (ctxBase := ctxBase)
      (outBase := outBase) (outLen := outLen) (outCap := outCap)
      (ctx := ctx) (outReg := outReg) (stk := stk)
      rfl hr1 hs1_4 hs1_15 hctx hctx_base hctx_len hout hout_base hout_rw hstk hstkrw hstkne_out
      h16 hspW hstk_lo hstk_hi hgt hle2 hcap_pos hout_eq hout_pg houtCapW hctxW hmwf
      hc200 hc208 hc216 hor)

end SporosVmExec
