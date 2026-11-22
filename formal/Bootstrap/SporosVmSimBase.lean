/-
Simulation infrastructure and core control-flow specs for `sporos_vm`.

Root A framing:
- Root A is MUSE's hand-written checker (`bootstrap/roots/sporos/sporos-stage1.hex`).
- `spr1_check` is an independent second check (not Root A or Root A's trust base).
- `sporos_vm` is the post-admission native x86-64 bytecode VM paired with
  `Sporos.decode` + `Sporos.checkProg`.

This module defines:
1. Extra `X86Vm.exec` rewrites (`exec_mul`, `exec_div`, `exec_idiv`, `exec_cqo`, `exec_sarI`)
   and `Sporos` <-> `X86Vm` arithmetic/word/region helpers.
2. `flatProgQwords` / `encodeFlatProg` / `FlatProgFit` / `FlatCodeOk` and
   exact qword read theorems for the triplicated program tables.
3. Arena invariants (`PairListRel`, `RsLocRel`, `MemRel`, `ChainRel`, `TabsRel`,
   `InpRel`, `OutRel`) bundled in `VmCtx` and `VmRel`.
4. `AtMemSyscall` / `OracleLegalAt` / `OracleAdmits` / `HitsNomem` matching
   `X86Env.syscall` (`mmap`/`mremap`).
5. Core VM block specs: `L_vm_halt_spec`, `L_fail_spec`, `L_vm_loop_fetch_spec`,
   `L_vm_dispatch_spec`, `L_prim_prologue_spec`, `L_prim_epilogue_mk1_spec`,
   `L_prim_epilogue_spec`, plus oracle-aware wrappers for `fn_mremap`,
   `fn_ds_push`, `fn_aux_push`, `fn_out_push_byte`, and `fn_out_push_u64`.
-/
import Bootstrap.SporosVmExec
import Bootstrap.SporosProvLen

namespace SporosVmSimBase

open X86Env
open X86Int
open X86Vm
open SporosVmCore
open SporosVmExec

/-! ### Extra single-step rewrites and arithmetic/byte bridges -/

theorem M_eq_W : Sporos.M = X86Env.W := rfl

theorem exec_mul {src len a b : Nat} {s : X86Vm.St}
    (ha : s.reg 0 = some a) (hb : s.reg src = some b) :
    let p := (a % X86Env.W) * (b % X86Env.W)
    let lo := p % X86Env.W
    let hi := p / X86Env.W
    let ov := hi != 0
    X86Vm.exec (.mul src) len s =
      .ok { (s.set 0 (some lo)).set 2 (some hi) with
            fl := ⟨some ov, none, none, some ov⟩, rip := s.rip + len } := by
  simp [X86Vm.exec, ha, hb, X86Int.lift, bind, Option.bind, pure]

theorem exec_div {src len lo hi d : Nat} {s : X86Vm.St}
    (hlo : s.reg 0 = some lo) (hhi : s.reg 2 = some hi) (hd : s.reg src = some d)
    (hnz : d % X86Env.W ≠ 0)
    (hlt : ((hi % X86Env.W) * X86Env.W + lo % X86Env.W) / (d % X86Env.W) < X86Env.W) :
    let num := (hi % X86Env.W) * X86Env.W + lo % X86Env.W
    let den := d % X86Env.W
    X86Vm.exec (.div src) len s =
      .ok { (s.set 0 (some (num / den))).set 2 (some (num % den)) with
            fl := Flags.undef, rip := s.rip + len } := by
  have hle : ¬ (X86Env.W ≤ ((hi % X86Env.W) * X86Env.W + lo % X86Env.W) / (d % X86Env.W)) := by
    omega
  simp [X86Vm.exec, hlo, hhi, hd, hnz, hle, X86Int.lift, bind, Option.bind, pure]

theorem exec_cqo {len a : Nat} {s : X86Vm.St} (ha : s.reg 0 = some a) :
    let hi := if X86Int.msb (a % X86Env.W) then X86Env.W - 1 else 0
    X86Vm.exec .cqo len s = .ok { s.set 2 (some hi) with rip := s.rip + len } := by
  simp [X86Vm.exec, ha, X86Int.lift, bind, Option.bind, pure]

theorem exec_idiv {src len lo hi d : Nat} {s : X86Vm.St}
    (hlo : s.reg 0 = some lo) (hhi : s.reg 2 = some hi) (hd : s.reg src = some d)
    (hnz : toSigned64 d ≠ 0)
    (hmin : -((2 ^ 63 : Nat) : Int) ≤
      Int.tdiv (toSigned64 hi * (X86Env.W : Int) + ((lo % X86Env.W : Nat) : Int)) (toSigned64 d))
    (hlim : Int.tdiv (toSigned64 hi * (X86Env.W : Int) + ((lo % X86Env.W : Nat) : Int)) (toSigned64 d) <
      ((2 ^ 63 : Nat) : Int)) :
    let num : Int := toSigned64 hi * (X86Env.W : Int) + ((lo % X86Env.W : Nat) : Int)
    let den : Int := toSigned64 d
    X86Vm.exec (.idiv src) len s =
      .ok { (s.set 0 (some (ofSigned64 (Int.tdiv num den)))).set 2
              (some (ofSigned64 (Int.tmod num den))) with
            fl := Flags.undef, rip := s.rip + len } := by
  have hmod : (lo : Int) % (X86Env.W : Int) = ((lo % X86Env.W : Nat) : Int) := by
    unfold X86Env.W; omega
  have hmin' :
      (-9223372036854775808 : Int) ≤
        (toSigned64 hi * (X86Env.W : Int) + (lo : Int) % (X86Env.W : Int)).tdiv (toSigned64 d) := by
    rw [hmod]; exact hmin
  have hlim' :
      (toSigned64 hi * (X86Env.W : Int) + (lo : Int) % (X86Env.W : Int)).tdiv (toSigned64 d) <
        (9223372036854775808 : Int) := by
    rw [hmod]; exact hlim
  simp [X86Vm.exec, hlo, hhi, hd, X86Int.lift, hnz, hmin', hlim']

theorem exec_sarI {d c len a : Nat} {s : X86Vm.St} (ha : s.reg d = some a) :
    X86Vm.exec (.sarI d c) len s =
      .ok { s.set d (some (sar64 a c)) with fl := Flags.undef, rip := s.rip + len } := by
  simp [X86Vm.exec, ha, X86Int.lift, bind, Option.bind, pure]

theorem nat_land_self (x : Nat) : Nat.land x x = x :=
  Nat.and_self x

theorem cond_aluF_and_self_le {x : Nat} (hx0 : 0 < x) (hx63 : x < 2 ^ 63) :
    X86Int.cond (X86Int.aluF .and x x).2 14 = some false := by
  have h2p63 : (2 ^ 63 : Nat) < X86Env.W := by decide
  have hxW : x < X86Env.W := Nat.lt_trans hx63 h2p63
  have h1 : Nat.land x x = x := nat_land_self x
  have hmsb : X86Int.msb x = false := msb_lt_2p63 hx63
  have hz : (x == 0) = false := beq_false_of_ne (by omega)
  simp [X86Int.cond, X86Int.aluF, hmsb, hz]

theorem toSigned_eq_toSigned64 {x : Nat} (hx : x < X86Env.W) :
    Sporos.toSigned x = toSigned64 x := by
  simp [Sporos.toSigned, toSigned64, Nat.mod_eq_of_lt hx, M_eq_W]

theorem ofInt_eq_ofSigned64 (i : Int) :
    Sporos.ofInt i = ofSigned64 i := by
  simp [Sporos.ofInt, ofSigned64, M_eq_W]

theorem regs_get_of_ok {s : X86Vm.St} (hr : RegsOk s) (r : Nat) (h1 : r < 16) (h2 : r ≠ 11) :
    ∃ v, s.reg r = some v := by
  have hne := hr.2 r h1 h2
  cases h : s.reg r with
  | none => simp [h] at hne
  | some v => exact ⟨v, rfl⟩

theorem list_foldl_add (l : List Nat) (z : Nat) :
    l.foldl (· + ·) z = z + l.foldl (· + ·) 0 := by
  induction l generalizing z with
  | nil => simp
  | cons x xs ih =>
    rw [List.foldl_cons, List.foldl_cons, Nat.zero_add, ih (z + x), ih x]
    omega

theorem list_range_foldl_add_eq_foldr (n : Nat) (f : Nat → Nat) :
    (List.range n).foldl (fun acc j => acc + f j) 0 =
      (List.range n).foldr (fun j acc => f j + acc) 0 := by
  induction n with
  | zero => rfl
  | succ k ih =>
    rw [List.range_succ, List.foldl_append, List.foldr_append]
    simp only [List.foldl_cons, List.foldl_nil, List.foldr_cons, List.foldr_nil, Nat.add_zero]
    have hfoldr : ∀ (l : List Nat) (z : Nat),
        l.foldr (fun j acc => f j + acc) z = l.foldr (fun j acc => f j + acc) 0 + z := by
      intro l z
      induction l with
      | nil => simp
      | cons a t iht => simp only [List.foldr_cons, iht, Nat.add_assoc]
    rw [ih, hfoldr (List.range k) (f k)]

/-! ### Byte-list and region read/write helpers -/

local instance : Inhabited Sporos.Word := ⟨⟨0, 0, 0, [], []⟩⟩
local instance : Inhabited Sporos.Instr := ⟨.ret⟩
local instance : Inhabited SporosProvLen.LVal := ⟨⟨0, 0⟩⟩
local instance : Inhabited SporosProvLen.LFrame := ⟨⟨0, 0, [], 0, 0⟩⟩
local instance : Inhabited Region := ⟨⟨0, [], false⟩⟩

def BytesLt (m : Mem) : Prop :=
  ∀ r ∈ m, ∀ b ∈ r.data, b < 256

theorem bytes2_lt {v : Nat} : ∀ b ∈ bytes2 v, b < 256 := by
  intro b hb; simp [bytes2] at hb; rcases hb with rfl | rfl <;> omega

theorem bytes4_lt {v : Nat} : ∀ b ∈ bytes4 v, b < 256 := by
  intro b hb
  simp [bytes4, List.range_succ, List.range_zero] at hb
  rcases hb with rfl | rfl | rfl | rfl <;> omega

theorem bytes8_lt {v : Nat} : ∀ b ∈ bytes8 v, b < 256 := by
  intro b hb
  simp [bytes8, List.range_succ, List.range_zero] at hb
  rcases hb with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> omega

theorem list_set_lt {d : List Nat} (hd : ∀ x ∈ d, x < 256) (off v : Nat) (hv : v < 256) :
    ∀ x ∈ d.set off v, x < 256 := by
  intro x hx
  rw [List.mem_iff_getElem?] at hx
  obtain ⟨j, hj⟩ := hx
  by_cases heq : j = off
  · subst heq
    by_cases hoff : j < d.length
    · rw [List.getElem?_set_self hoff] at hj; cases hj; exact hv
    · rw [List.getElem?_eq_none (by simp [List.length_set]; omega)] at hj; cases hj
  · rw [List.getElem?_set_ne (Ne.symm heq)] at hj
    exact hd x (List.mem_of_getElem? hj)

theorem writeBytes_bytes_lt {d : List Nat} (hd : ∀ b ∈ d, b < 256) (off : Nat) (bs : List Nat) :
    ∀ b ∈ writeBytes d off bs, b < 256 := by
  induction bs generalizing d off with
  | nil => simpa [writeBytes] using hd
  | cons y ys ih =>
    simp only [writeBytes]
    exact ih (list_set_lt hd off (y % 256) (Nat.mod_lt y (by decide))) (off + 1)

theorem mem_set_mem_cases {α : Type} {l : List α} {i : Nat} {x u : α}
    (hu : u ∈ l.set i x) : u ∈ l ∨ u = x := by
  rw [List.mem_iff_getElem?] at hu
  obtain ⟨j, hj⟩ := hu
  by_cases heq : j = i
  · subst heq
    by_cases hi : j < l.length
    · rw [List.getElem?_set_self hi] at hj; cases hj; exact Or.inr rfl
    · rw [List.getElem?_eq_none (by simp [List.length_set]; omega)] at hj; cases hj
  · rw [List.getElem?_set_ne (Ne.symm heq)] at hj
    exact Or.inl (List.mem_of_getElem? hj)

theorem bytesLt_set_wr {m : Mem} {i : Nat} {r : Region} {d' : List Nat}
    (hbl : BytesLt m) (_hm : m[i]? = some r) (hd' : ∀ b ∈ d', b < 256) :
    BytesLt (m.set i { r with data := d' }) := by
  intro u hu b hb
  rcases mem_set_mem_cases hu with hu0 | rfl
  · exact hbl u hu0 b hb
  · exact hd' b hb

theorem bytesLt_wrMem {m m' : Mem} {w : MemW} {a v : Nat} {r : Region}
    (hbl : BytesLt m) (hw : m.wf) (hm : r ∈ m) (hrw : r.rw = true)
    (hlo : r.base ≤ a) (hhi : a + (match w with | .w8 => 1 | .w16 => 2 | .w32 => 4 | .w64 => 8) ≤ r.hi)
    (hwr : wrMem w m a v = some m') : BytesLt m' := by
  cases w with
  | w8 =>
    obtain ⟨i, m1, h1, hget, hm1, _, _, _⟩ := wrMem_w8_region hw hm hrw hlo hhi
    rw [h1] at hwr; cases hwr; subst hm1
    exact bytesLt_set_wr hbl hget (writeBytes_bytes_lt (hbl r hm) _ _)
  | w16 =>
    obtain ⟨i, m1, h1, hget, hm1, _, _, _⟩ := wrMem_w16_region hw hm hrw hlo hhi
    rw [h1] at hwr; cases hwr; subst hm1
    exact bytesLt_set_wr hbl hget (writeBytes_bytes_lt (hbl r hm) _ _)
  | w32 =>
    obtain ⟨i, m1, h1, hget, hm1, _, _, _⟩ := wrMem_w32_region hw hm hrw hlo hhi
    rw [h1] at hwr; cases hwr; subst hm1
    exact bytesLt_set_wr hbl hget (writeBytes_bytes_lt (hbl r hm) _ _)
  | w64 =>
    obtain ⟨i, m1, h1, hget, hm1, _, _, _⟩ := wrMem_w64_region hw hm hrw hlo hhi
    rw [h1] at hwr; cases hwr; subst hm1
    exact bytesLt_set_wr hbl hget (writeBytes_bytes_lt (hbl r hm) _ _)

theorem leVal_take8_lt_W {bs : List Nat} (hb : ∀ x ∈ bs, x < 256) :
    leVal (bs.take 8) < X86Env.W := by
  have go : ∀ (k : Nat) (l : List Nat), l.length ≤ k → (∀ x ∈ l, x < 256) →
      leVal l < 256 ^ k := by
    intro k
    induction k with
    | zero =>
      intro l hl _
      cases l with
      | nil => simp [leVal]
      | cons _ _ => simp at hl
    | succ k ih =>
      intro l hl hbl
      cases l with
      | nil => simp [leVal]; exact Nat.pow_pos (by decide)
      | cons x xs =>
        simp only [List.length_cons] at hl
        have hx : x < 256 := hbl x (by simp)
        have hxs : leVal xs < 256 ^ k :=
          ih xs (by omega) (fun y hy => hbl y (by simp [hy]))
        simp only [leVal]
        have hp : 1 ≤ 256 ^ k := Nat.one_le_pow _ _ (by decide)
        have hpow : 256 ^ (k + 1) = 256 * 256 ^ k := by rw [Nat.pow_succ]; omega
        omega
  have ht := go 8 (bs.take 8) (List.length_take_le _ _) (fun x hx => hb x (List.mem_of_mem_take hx))
  simpa [X86Env.W] using ht

theorem ea_add_eq {a disp : Nat} (h : a + disp < X86Env.W) : ea a disp = a + disp := by
  simp [X86Int.ea, Nat.mod_eq_of_lt h]

theorem eaIdx_add_eq {xb xi scale disp : Nat} (h : xb + scale * xi + disp < X86Env.W) :
    eaIdx xb xi scale disp = xb + scale * xi + disp := by
  simp [X86Vm.eaIdx, Nat.mod_eq_of_lt h]

theorem rdMem_w8_at {m : Mem} {r : Region} {a b : Nat}
    (hw : m.wf) (hm : r ∈ m) (hlo : r.base ≤ a) (hhi : a + 1 ≤ r.hi)
    (hb : r.data[a - r.base]? = some b) :
    rdMem .w8 m a = some b := by
  obtain ⟨k, hfk⟩ := mem_find_of_mem m r a 0 hw.2 hm hlo hhi
  simp only [Nat.zero_add] at hfk
  simp only [rdMem, Mem.rd, hfk, bind, Option.bind]
  exact hb

theorem rdMem_w16_at {m : Mem} {r : Region} {a v : Nat}
    (hw : m.wf) (hm : r ∈ m) (hlo : r.base ≤ a) (hhi : a + 2 ≤ r.hi)
    (hv : leVal ((r.data.drop (a - r.base)).take 2) = v) :
    rdMem .w16 m a = some v := by
  simp only [rdMem, rd16, mem_rdN_of_mem m r a 2 hw hm hlo hhi, Option.map, hv]

theorem rdMem_w32_at {m : Mem} {r : Region} {a v : Nat}
    (hw : m.wf) (hm : r ∈ m) (hlo : r.base ≤ a) (hhi : a + 4 ≤ r.hi)
    (hv : leVal ((r.data.drop (a - r.base)).take 4) = v) :
    rdMem .w32 m a = some v := rd32_of_mem hw hm hlo hhi hv

theorem rdMem_w64_at {m : Mem} {r : Region} {a v : Nat}
    (hw : m.wf) (hm : r ∈ m) (hlo : r.base ≤ a) (hhi : a + 8 ≤ r.hi)
    (hv : leVal ((r.data.drop (a - r.base)).take 8) = v) :
    rdMem .w64 m a = some v := rdMem_w64_of_mem hw hm hlo hhi hv

theorem mem_set_self_get {α : Type} (l : List α) (i : Nat) (x : α) (hi : i < l.length) :
    (l.set i x)[i]? = some x := by
  simp [hi]

theorem mem_set_ne_get {α : Type} (l : List α) {i j : Nat} (x : α) (hne : i ≠ j) :
    (l.set i x)[j]? = l[j]? :=
  List.getElem?_set_ne hne

theorem writeBytes_at_self (d : List Nat) (off : Nat) (b : Nat) (hb : b < 256) (hoff : off < d.length) :
    (writeBytes d off [b])[off]? = some b := by
  simp only [writeBytes, Nat.mod_eq_of_lt hb, List.getElem?_set_self hoff]

theorem list_take_drop_range_leVal {n : Nat} (hpos : n ≤ 8) (d : List Nat) (off v : Nat)
    (hlen : off + n ≤ d.length) (hv : ∀ i < n, d[off + i]? = some (v / 256 ^ i % 256)) :
    leVal ((d.drop off).take n) = v % 256 ^ n := by
  induction n generalizing off v with
  | zero => simp [leVal, Nat.mod_one]
  | succ k ih =>
    have h0 : d[off]? = some (v % 256) := by
      have := hv 0 (by omega)
      simpa using this
    have hval : d[off]'(by omega) = v % 256 := by
      have hget := List.getElem?_eq_getElem (by omega : off < d.length)
      rw [hget] at h0
      injection h0
    have hdrop : (d.drop off).take (k + 1) = (v % 256) :: ((d.drop (off + 1)).take k) := by
      rw [List.drop_eq_getElem_cons (by omega : off < d.length), hval, List.take_succ_cons]
    rw [hdrop, leVal]
    have hrest : leVal ((d.drop (off + 1)).take k) = (v / 256) % 256 ^ k := by
      apply ih (by omega) (off + 1) (v / 256) (by omega)
      intro i hi
      have hdv := hv (i + 1) (by omega)
      have hpow : (v / 256) / 256 ^ i = v / 256 ^ (i + 1) := by
        rw [Nat.div_div_eq_div_mul, Nat.mul_comm 256, ← Nat.pow_succ]
      rw [hpow, show off + 1 + i = off + (i + 1) by omega]
      exact hdv
    rw [hrest]
    have hpow : 256 ^ (k + 1) = 256 * 256 ^ k := by rw [Nat.pow_succ]; omega
    rw [hpow]
    have hq := Nat.div_add_mod (v / 256) (256 ^ k)
    have hvr := Nat.div_add_mod v 256
    have hlt_rem : v % 256 + 256 * ((v / 256) % 256 ^ k) < 256 * 256 ^ k := by
      have hm1 : v % 256 < 256 := Nat.mod_lt _ (by decide)
      have hm2 : (v / 256) % 256 ^ k < 256 ^ k := Nat.mod_lt _ (Nat.pow_pos (by decide))
      omega
    have hv_eq : v = 256 * 256 ^ k * ((v / 256) / 256 ^ k) + (v % 256 + 256 * ((v / 256) % 256 ^ k)) := by
      have hmul : 256 * (v / 256) = 256 * 256 ^ k * ((v / 256) / 256 ^ k) + 256 * ((v / 256) % 256 ^ k) := by
        symm
        rw [Nat.mul_assoc, ← Nat.mul_add, hq]
      omega
    have hmod_step : (256 * 256 ^ k * ((v / 256) / 256 ^ k) + (v % 256 + 256 * ((v / 256) % 256 ^ k))) %
        (256 * 256 ^ k) = v % 256 + 256 * ((v / 256) % 256 ^ k) := by
      rw [Nat.mul_add_mod, Nat.mod_eq_of_lt hlt_rem]
    rw [← hv_eq] at hmod_step
    exact hmod_step.symm

theorem wrMem_w8_eq {m : Mem} {r : Region} {a v : Nat}
    (hw : m.wf) (hm : r ∈ m) (hrw : r.rw = true) (hlo : r.base ≤ a) (hhi : a + 1 ≤ r.hi) :
    ∃ (i : Nat) (m' : Mem),
      wrMem .w8 m a v = some m' ∧ m[i]? = some r ∧
      m' = m.set i { r with data := writeBytes r.data (a - r.base) [v % 256] } ∧
      m'.wf ∧ { r with data := writeBytes r.data (a - r.base) [v % 256] } ∈ m' ∧
      (∀ s ∈ m, s.base ≠ r.base → s ∈ m') :=
  wrMem_w8_region hw hm hrw hlo hhi

theorem mem_mremap_free_ne {m : Mem} {idx n a : Nat} {old u : Region}
    (_hmwf : m.wf) (hget : m[idx]? = some old) (hu : u ∈ m) (hne : u.base ≠ old.base)
    (hn : 0 < n) (hul : 0 < u.data.length)
    (hfree : Mem.free (m.eraseIdx idx) a n = true) :
    a ≠ u.base := by
  have hu' : u ∈ m.eraseIdx idx := mem_eraseIdx_other hget hu hne
  simp only [Mem.free, List.all_eq_true, Bool.or_eq_true, decide_eq_true_eq] at hfree
  rcases hfree u hu' with h1 | h2
  · omega
  · simp only [Region.hi] at h2; omega

theorem mem_nodup_replace_eraseIdx {m : Mem} {rs : List Region} {idx : Nat} {old newR : Region}
    (hmwf : m.wf) (hget : m[idx]? = some old)
    (hnodup : (rs.map Region.base).Nodup)
    (hsub : ∀ r ∈ rs, r ∈ m) (hpos : ∀ u ∈ m, 0 < u.data.length)
    (hnew : 0 < newR.data.length) (hfree : Mem.free (m.eraseIdx idx) newR.base newR.data.length = true) :
    ((rs.map fun r => if r.base == old.base then newR else r).map Region.base).Nodup := by
  have hdisj : ∀ r ∈ rs, r.base ≠ old.base → newR.base ≠ r.base := by
    intro r hr hne
    have hrm := hsub r hr
    exact mem_mremap_free_ne hmwf hget hrm hne hnew (hpos r hrm) hfree
  clear hmwf hget hfree hpos
  induction rs with
  | nil => simp
  | cons x xs ih =>
    simp only [List.map_cons, List.nodup_cons] at hnodup ⊢
    constructor
    · intro hmem
      rw [List.mem_map] at hmem
      rcases hmem with ⟨y, hy, hyb⟩
      rw [List.mem_map] at hy
      rcases hy with ⟨r, hr, rfl⟩
      by_cases hx : x.base == old.base <;> by_cases hrx : r.base == old.base
      · have h1 : x.base = old.base := by simpa using hx
        have h2 : r.base = old.base := by simpa using hrx
        apply hnodup.1
        rw [List.mem_map]
        exact ⟨r, hr, by rw [h1, h2]⟩
      · simp only [hx, ↓reduceIte, hrx, Bool.false_eq_true] at hyb
        have hne : r.base ≠ old.base := by simpa using hrx
        exact hdisj r (by simp [hr]) hne hyb.symm
      · simp only [hx, Bool.false_eq_true, ↓reduceIte, hrx] at hyb
        have hne : x.base ≠ old.base := by simpa using hx
        exact hdisj x (by simp) hne hyb
      · simp only [hx, Bool.false_eq_true, ↓reduceIte, hrx] at hyb
        apply hnodup.1
        rw [List.mem_map]
        exact ⟨r, hr, hyb⟩
    · exact ih hnodup.2 (fun r hr => hsub r (by simp [hr]))
        (fun r hr => hdisj r (by simp [hr]))

/-! ### Flat program encoding (`code1`, `code2`, `code3`) -/

def primCode (p : Sporos.Prim) : Nat :=
  Sporos.Prim.all.idxOf p

theorem primCode_lt (p : Sporos.Prim) : primCode p < 57 := by
  unfold primCode
  have hm := Sporos.Prim.mem_all p
  have hl : Sporos.Prim.all.length = 57 := by decide
  rw [← hl]
  exact List.idxOf_lt_length_of_mem hm

theorem primAll_get_primCode (p : Sporos.Prim) :
    Sporos.Prim.all[primCode p]'(by rw [show Sporos.Prim.all.length = 57 from by decide]; exact primCode_lt p) = p := by
  cases p <;> decide

def instrPair : Sporos.Instr → Nat × Nat
  | .prim p => (primCode p, 0)
  | .lit n => (57, n % X86Env.W)
  | .str i => (58, i % X86Env.W)
  | .pick n => (59, n % X86Env.W)
  | .tor => (60, 0)
  | .fromr => (61, 0)
  | .rfetch => (62, 0)
  | .locget i => (63, i % X86Env.W)
  | .locset i => (64, i % X86Env.W)
  | .jmp t => (65, t % X86Env.W)
  | .jz t => (66, t % X86Env.W)
  | .jnz t => (67, t % X86Env.W)
  | .call w => (68, w % X86Env.W)
  | .ret => (69, 0)

def dataMetaQwords (acc : Nat) : List (List Nat) → List Nat
  | [] => []
  | d :: ds => acc :: d.length :: dataMetaQwords (acc + d.length) ds

def wordMetaQwords (off : Nat) : List Sporos.Word → List Nat
  | [] => []
  | wd :: ws =>
    wd.ins :: wd.outs :: wd.locals :: wd.code.length :: off :: 0 :: 0 :: 0 ::
      wordMetaQwords (off + wd.code.length * 16) ws

def codePairsQwords (ws : List Sporos.Word) : List Nat :=
  ws.flatMap fun wd => wd.code.flatMap fun ins =>
    let (op, arg) := instrPair ins
    [op, arg]

def flatProgHeader (pr : Sporos.Prog) : List Nat :=
  let tot := (pr.words.map (·.code.length)).foldl (· + ·) 0
  [pr.main, pr.words.length, pr.data.length, tot]

def flatProgBaseOff (pr : Sporos.Prog) : Nat :=
  32 + pr.data.length * 16 + pr.words.length * 64

def flatProgQwords (pr : Sporos.Prog) : List Nat :=
  flatProgHeader pr ++
  dataMetaQwords 0 pr.data ++
  wordMetaQwords (flatProgBaseOff pr) pr.words ++
  codePairsQwords pr.words

def encodeFlatProg (pr : Sporos.Prog) : List Nat :=
  (flatProgQwords pr).flatMap bytes8

def FlatProgFit (pr : Sporos.Prog) : Prop :=
  pr.main < X86Env.W ∧
  pr.words.length < X86Env.W ∧
  pr.data.length < X86Env.W ∧
  (flatProgQwords pr).length * 8 < 2 ^ 47 ∧
  pr.data.flatten.length < X86Env.W ∧
  (∀ wd ∈ pr.words, wd.ins < X86Env.W ∧ wd.outs < X86Env.W ∧
    wd.locals < 2 ^ 44 ∧ wd.code.length < X86Env.W ∧
    (∀ ins ∈ wd.code, match ins with
      | .lit n => n < X86Env.W
      | .str i => i < X86Env.W
      | .pick n => n < X86Env.W
      | .locget i => i < X86Env.W
      | .locset i => i < X86Env.W
      | .jmp t => t < X86Env.W
      | .jz t => t < X86Env.W
      | .jnz t => t < X86Env.W
      | .call w => w < X86Env.W
      | _ => True))

theorem dataMetaQwords_len (acc : Nat) (ds : List (List Nat)) :
    (dataMetaQwords acc ds).length = 2 * ds.length := by
  induction ds generalizing acc with
  | nil => rfl
  | cons d ds ih => simp [dataMetaQwords, ih, Nat.mul_succ]

theorem wordMetaQwords_len (off : Nat) (ws : List Sporos.Word) :
    (wordMetaQwords off ws).length = 8 * ws.length := by
  induction ws generalizing off with
  | nil => rfl
  | cons w ws ih => simp [wordMetaQwords, ih, Nat.mul_succ]

theorem dataMetaQwords_get (ds : List (List Nat)) :
    ∀ (acc i : Nat), i < ds.length →
      (dataMetaQwords acc ds)[2 * i]? = some (acc + ((ds.take i).map List.length).foldl (· + ·) 0) ∧
      (dataMetaQwords acc ds)[2 * i + 1]? = some ds[i]!.length := by
  induction ds with
  | nil => intro acc i hi; simp at hi
  | cons d ds ih =>
    intro acc i hi
    cases i with
    | zero =>
      simp [dataMetaQwords]
    | succ k =>
      have hk : k < ds.length := by simp at hi; omega
      have hrec := ih (acc + d.length) k hk
      have hstep : ∀ (a b : Nat) (l : List Nat) (idx : Nat),
          (a :: b :: l)[idx + 2]? = l[idx]? := fun _ _ _ _ => rfl
      simp only [dataMetaQwords, List.take_succ_cons, List.map_cons,
        List.foldl_cons, Nat.zero_add]
      rw [list_foldl_add]
      constructor
      · rw [show 2 * (k + 1) = 2 * k + 2 by omega, hstep]
        have h1 := hrec.1
        rw [Nat.add_assoc] at h1
        exact h1
      · rw [show 2 * (k + 1) + 1 = (2 * k + 1) + 2 by omega, hstep]
        simpa [List.getElem!_eq_getElem?_getD] using hrec.2

theorem wordMetaQwords_get (ws : List Sporos.Word) :
    ∀ (off w : Nat), w < ws.length →
      let wd := ws[w]!
      let woff := off + ((ws.take w).map fun x => x.code.length * 16).foldl (· + ·) 0
      (wordMetaQwords off ws)[8 * w]? = some wd.ins ∧
      (wordMetaQwords off ws)[8 * w + 2]? = some wd.locals ∧
      (wordMetaQwords off ws)[8 * w + 3]? = some wd.code.length ∧
      (wordMetaQwords off ws)[8 * w + 4]? = some woff := by
  induction ws with
  | nil => intro off w hw; simp at hw
  | cons wd ws ih =>
    intro off w hw
    cases w with
    | zero =>
      simp [wordMetaQwords]
    | succ k =>
      have hk : k < ws.length := by simp at hw; omega
      have hrec := ih (off + wd.code.length * 16) k hk
      have hdrop : ∀ (a0 a1 a2 a3 a4 a5 a6 a7 : Nat) (l : List Nat) (idx : Nat),
          (a0 :: a1 :: a2 :: a3 :: a4 :: a5 :: a6 :: a7 :: l)[idx + 8]? = l[idx]? :=
        fun _ _ _ _ _ _ _ _ _ _ => rfl
      simp only [wordMetaQwords, List.take_succ_cons, List.map_cons, List.foldl_cons, Nat.zero_add,
        List.getElem!_eq_getElem?_getD]
      rw [list_foldl_add]
      refine ⟨?_, ?_, ?_, ?_⟩
      · rw [show 8 * (k + 1) = 8 * k + 8 by omega, hdrop]
        simpa [List.getElem!_eq_getElem?_getD] using hrec.1
      · rw [show 8 * (k + 1) + 2 = (8 * k + 2) + 8 by omega, hdrop]
        simpa [List.getElem!_eq_getElem?_getD] using hrec.2.1
      · rw [show 8 * (k + 1) + 3 = (8 * k + 3) + 8 by omega, hdrop]
        simpa [List.getElem!_eq_getElem?_getD] using hrec.2.2.1
      · rw [show 8 * (k + 1) + 4 = (8 * k + 4) + 8 by omega, hdrop]
        simpa [Nat.add_assoc, List.getElem!_eq_getElem?_getD] using hrec.2.2.2

theorem codePairsQwords_len (ws : List Sporos.Word) :
    (codePairsQwords ws).length =
      2 * (ws.map (·.code.length)).foldl (· + ·) 0 := by
  induction ws with
  | nil => simp [codePairsQwords]
  | cons wd ws ih =>
    simp only [codePairsQwords, List.flatMap_cons, List.length_append, List.map_cons,
      List.foldl_cons, Nat.zero_add] at ih ⊢
    have hwd : (wd.code.flatMap fun ins => let (op, arg) := instrPair ins; [op, arg]).length =
        2 * wd.code.length := by
      induction wd.code with
      | nil => simp
      | cons i cs ihc =>
        simp [List.flatMap_cons, ihc]
        omega
    have hfl := list_foldl_add (ws.map (·.code.length)) wd.code.length
    rw [hwd, ih, hfl, Nat.mul_add]

theorem codePairsQwords_get (ws : List Sporos.Word) :
    ∀ (w pc : Nat), w < ws.length → pc < ws[w]!.code.length →
      let baseQ := 2 * ((ws.take w).map (·.code.length)).foldl (· + ·) 0
      let ins := ws[w]!.code[pc]!
      (codePairsQwords ws)[baseQ + 2 * pc]? = some (instrPair ins).1 ∧
      (codePairsQwords ws)[baseQ + 2 * pc + 1]? = some (instrPair ins).2 := by
  have hword : ∀ (cs : List Sporos.Instr) (pc : Nat), pc < cs.length →
      (cs.flatMap fun ins => let (op, arg) := instrPair ins; [op, arg])[2 * pc]? =
        some (instrPair cs[pc]!).1 ∧
      (cs.flatMap fun ins => let (op, arg) := instrPair ins; [op, arg])[2 * pc + 1]? =
        some (instrPair cs[pc]!).2 := by
    intro cs
    induction cs with
    | nil => intro pc hpc; simp at hpc
    | cons i cs ihc =>
      intro pc hpc
      cases pc with
      | zero => simp [List.flatMap_cons]
      | succ k =>
        have hk : k < cs.length := by simp at hpc; omega
        have hr := ihc k hk
        have hstep2 : ∀ (a b : Nat) (l : List Nat) (idx : Nat),
            ([a, b] ++ l)[idx + 2]? = l[idx]? := fun _ _ _ _ => rfl
        simp only [List.flatMap_cons, List.getElem!_eq_getElem?_getD]
        constructor
        · rw [show 2 * (k + 1) = 2 * k + 2 by omega, hstep2]
          simpa [List.getElem!_eq_getElem?_getD] using hr.1
        · rw [show 2 * (k + 1) + 1 = (2 * k + 1) + 2 by omega, hstep2]
          simpa [List.getElem!_eq_getElem?_getD] using hr.2
  have hword_len : ∀ (cs : List Sporos.Instr),
      (cs.flatMap fun ins => let (op, arg) := instrPair ins; [op, arg]).length = 2 * cs.length := by
    intro cs
    induction cs with
    | nil => simp
    | cons i cs ihc =>
      simp [List.flatMap_cons, ihc]
      omega
  induction ws with
  | nil => intro w _ hw; simp at hw
  | cons wd ws ih =>
    intro w pc hw hpc
    cases w with
    | zero =>
      simp only [codePairsQwords, List.flatMap_cons, List.take_zero, List.map_nil, List.foldl_nil,
        Nat.mul_zero, Nat.zero_add, List.getElem!_eq_getElem?_getD, List.getElem?_cons_zero,
        Option.getD_some] at hpc ⊢
      have hl1 : 2 * pc < (wd.code.flatMap fun ins =>
          let (op, arg) := instrPair ins; [op, arg]).length := by
        rw [hword_len]; omega
      have hl2 : 2 * pc + 1 < (wd.code.flatMap fun ins =>
          let (op, arg) := instrPair ins; [op, arg]).length := by
        rw [hword_len]; omega
      have hr := hword wd.code pc hpc
      constructor
      · rw [List.getElem?_append_left hl1]
        simpa [List.getElem!_eq_getElem?_getD] using hr.1
      · rw [List.getElem?_append_left hl2]
        simpa [List.getElem!_eq_getElem?_getD] using hr.2
    | succ k =>
      have hk : k < ws.length := by simp at hw; omega
      have hpc' : pc < ws[k]!.code.length := by
        simpa [List.getElem!_eq_getElem?_getD] using hpc
      have hr := ih k pc hk hpc'
      simp only [codePairsQwords, List.flatMap_cons, List.take_succ_cons, List.map_cons,
        List.foldl_cons, Nat.zero_add]
      rw [list_foldl_add]
      have hlen1 : (wd.code.flatMap fun ins =>
          let (op, arg) := instrPair ins; [op, arg]).length = 2 * wd.code.length :=
        hword_len wd.code
      have hmul : 2 * (wd.code.length + ((ws.take k).map (·.code.length)).foldl (· + ·) 0) =
          2 * wd.code.length + 2 * ((ws.take k).map (·.code.length)).foldl (· + ·) 0 := by
        omega
      rw [hmul, Nat.add_assoc]
      have hge1 : (wd.code.flatMap fun ins =>
          let (op, arg) := instrPair ins; [op, arg]).length ≤
          2 * wd.code.length + (2 * ((ws.take k).map (·.code.length)).foldl (· + ·) 0 + 2 * pc) := by
        rw [hlen1]; omega
      have hge2 : (wd.code.flatMap fun ins =>
          let (op, arg) := instrPair ins; [op, arg]).length ≤
          2 * wd.code.length + (2 * ((ws.take k).map (·.code.length)).foldl (· + ·) 0 + 2 * pc) + 1 := by
        rw [hlen1]; omega
      rw [List.getElem?_append_right hge1]
      have hsub1 :
          2 * wd.code.length + (2 * ((ws.take k).map (·.code.length)).foldl (· + ·) 0 + 2 * pc) -
            (wd.code.flatMap fun ins => let (op, arg) := instrPair ins; [op, arg]).length =
            2 * ((ws.take k).map (·.code.length)).foldl (· + ·) 0 + 2 * pc := by
        rw [hlen1]; omega
      rw [hsub1]
      constructor
      · simpa [codePairsQwords, List.getElem!_eq_getElem?_getD] using hr.1
      · rw [show 2 * wd.code.length + (2 * ((ws.take k).map (·.code.length)).foldl (· + ·) 0 + 2 * pc) + 1 =
            2 * wd.code.length + (2 * ((ws.take k).map (·.code.length)).foldl (· + ·) 0 + 2 * pc + 1) by omega]
        rw [List.getElem?_append_right (by rw [hlen1]; omega)]
        have hsub2 :
            2 * wd.code.length + (2 * ((ws.take k).map (·.code.length)).foldl (· + ·) 0 + 2 * pc + 1) -
              (wd.code.flatMap fun ins => let (op, arg) := instrPair ins; [op, arg]).length =
              2 * ((ws.take k).map (·.code.length)).foldl (· + ·) 0 + 2 * pc + 1 := by
          rw [hlen1]; omega
        rw [hsub2]
        simpa [codePairsQwords, List.getElem!_eq_getElem?_getD] using hr.2

theorem flatMap_bytes8_drop_take (qs : List Nat) (i : Nat) (hi : i < qs.length)
    (hlt : qs[i]! < X86Env.W) :
    leVal (((qs.flatMap bytes8).drop (i * 8)).take 8) = qs[i]! := by
  induction qs generalizing i with
  | nil => simp at hi
  | cons q qs ih =>
    cases i with
    | zero =>
      simp only [List.flatMap_cons, Nat.zero_mul, List.drop_zero, List.getElem!_eq_getElem?_getD,
        List.getElem?_cons_zero, Option.getD_some] at hlt ⊢
      have htake : ((bytes8 q ++ qs.flatMap bytes8).take 8) = bytes8 q := by
        rw [List.take_append_of_le_length (by rw [bytes8_length]; exact Nat.le_refl 8),
          List.take_of_length_le (by rw [bytes8_length]; exact Nat.le_refl 8)]
      rw [htake, leVal_bytes8, Nat.mod_eq_of_lt hlt]
    | succ k =>
      have hk : k < qs.length := by simp at hi; omega
      have hlt' : qs[k]! < X86Env.W := by simpa [List.getElem!_eq_getElem?_getD] using hlt
      have hdrop :
          ((q :: qs).flatMap bytes8).drop ((k + 1) * 8) =
            (qs.flatMap bytes8).drop (k * 8) := by
        simp only [List.flatMap_cons]
        rw [List.drop_append, List.drop_eq_nil_of_le (by rw [bytes8_length]; omega),
          List.nil_append, bytes8_length]
        simp [Nat.succ_mul]
      rw [hdrop]
      simpa [List.getElem!_eq_getElem?_getD] using ih k hk hlt'

theorem flatMap_bytes8_length (qs : List Nat) :
    (qs.flatMap bytes8).length = qs.length * 8 := by
  induction qs with
  | nil => rfl
  | cons q qs ih =>
    simp only [List.flatMap_cons, List.length_append, bytes8_length, ih, List.length_cons]
    omega

theorem flatProgQwords_get_lt (pr : Sporos.Prog) (hfit : FlatProgFit pr) :
    ∀ j < (flatProgQwords pr).length, (flatProgQwords pr)[j]! < X86Env.W := by
  have hsz : (flatProgQwords pr).length * 8 < 2 ^ 47 := hfit.2.2.2.1
  have hlt247 : ∀ n, n * 8 < 2 ^ 47 → n < X86Env.W := by
    intro n hn; have h47 : 2 ^ 47 < X86Env.W := by decide
    omega
  have hdm : ∀ (ds : List (List Nat)) (acc : Nat),
      acc + ds.flatten.length < X86Env.W →
      ∀ x ∈ dataMetaQwords acc ds, x < X86Env.W := by
    intro ds
    induction ds with
    | nil => intro _ _ x hx; simp [dataMetaQwords] at hx
    | cons d ds ih =>
      intro acc hsum x hx
      simp only [dataMetaQwords, List.mem_cons] at hx
      simp only [List.flatten_cons, List.length_append] at hsum
      rcases hx with rfl | rfl | hx
      · omega
      · omega
      · exact ih (acc + d.length) (by omega) x hx
  have hwm : ∀ (ws : List Sporos.Word) (off : Nat),
      (∀ wd ∈ ws, wd.ins < X86Env.W ∧ wd.outs < X86Env.W ∧ wd.locals < 2 ^ 44 ∧ wd.code.length < X86Env.W) →
      off + (ws.map (·.code.length * 16)).foldl (· + ·) 0 < X86Env.W →
      ∀ x ∈ wordMetaQwords off ws, x < X86Env.W := by
    intro ws
    induction ws with
    | nil => intro _ _ _ x hx; simp [wordMetaQwords] at hx
    | cons w ws ih =>
      intro off hw hsum x hx
      simp only [List.map_cons, List.foldl_cons, Nat.zero_add] at hsum
      rw [list_foldl_add] at hsum
      have hw0 := hw w (by simp)
      simp only [wordMetaQwords, List.mem_cons] at hx
      have h44 : 2 ^ 44 < X86Env.W := by decide
      have hlocW : w.locals < X86Env.W := Nat.lt_trans hw0.2.2.1 h44
      have hW0 : 0 < X86Env.W := by decide
      rcases hx with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | hx
      · exact hw0.1
      · exact hw0.2.1
      · exact hlocW
      · exact hw0.2.2.2
      · omega
      · exact hW0
      · exact hW0
      · exact hW0
      · exact ih (off + w.code.length * 16) (fun wd hwd => hw wd (by simp [hwd])) (by omega) x hx
  have htot : (flatProgQwords pr).length =
      4 + 2 * pr.data.length + 8 * pr.words.length + 2 * (pr.words.map (·.code.length)).foldl (· + ·) 0 := by
    simp [flatProgQwords, flatProgHeader, dataMetaQwords_len, wordMetaQwords_len, codePairsQwords_len]
    omega
  have htotW : flatProgBaseOff pr +
      (pr.words.map (fun x => x.code.length * 16)).foldl (· + ·) 0 < X86Env.W := by
    have hmap : (pr.words.map (fun x => x.code.length * 16)).foldl (· + ·) 0 =
        16 * (pr.words.map (·.code.length)).foldl (· + ·) 0 := by
      have go : ∀ (ws : List Sporos.Word) (z : Nat),
          (ws.map (fun x => x.code.length * 16)).foldl (· + ·) (16 * z) =
            16 * (ws.map (·.code.length)).foldl (· + ·) z := by
        intro ws
        induction ws with
        | nil => intro z; simp
        | cons x xs ih =>
          intro z
          simp only [List.map_cons, List.foldl_cons]
          rw [show 16 * z + x.code.length * 16 = 16 * (z + x.code.length) by omega]
          exact ih (z + x.code.length)
      simpa using go pr.words 0
    have hflat8 : (flatProgQwords pr).length * 8 < 2 ^ 47 := hsz
    have h47 : 2 ^ 47 < X86Env.W := by decide
    simp only [flatProgBaseOff]
    omega
  intro j hj
  have hmem : (flatProgQwords pr)[j]! ∈ flatProgQwords pr := by
    rw [List.getElem!_eq_getElem?_getD]
    have hget := List.getElem?_eq_getElem hj
    rw [hget]
    exact List.getElem_mem hj
  have hmem_all : ∀ y ∈ flatProgQwords pr, y < X86Env.W := by
    intro y hy
    simp only [flatProgQwords, List.mem_append] at hy
    rcases hy with ((hy | hy) | hy) | hy
    · simp only [flatProgHeader, List.mem_cons, List.not_mem_nil, or_false] at hy
      rcases hy with rfl | rfl | rfl | rfl
      · exact hfit.1
      · exact hfit.2.1
      · exact hfit.2.2.1
      · exact hlt247 _ (by omega)
    · exact hdm pr.data 0 (by simpa using hfit.2.2.2.2.1) _ hy
    · exact hwm pr.words (flatProgBaseOff pr)
        (fun wd hwd => let h := hfit.2.2.2.2.2 wd hwd; ⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1⟩)
        htotW _ hy
    · simp only [codePairsQwords, List.mem_flatMap] at hy
      rcases hy with ⟨wd, hwd, ins, hins, hpair⟩
      have hwd' := hfit.2.2.2.2.2 wd hwd
      have hins' := hwd'.2.2.2.2 ins hins
      have hW70 : 70 < X86Env.W := by decide
      have hW0 : 0 < X86Env.W := by decide
      cases ins with
      | prim p =>
        have h57 := primCode_lt p
        simp only [instrPair, List.mem_cons, List.not_mem_nil, or_false] at hpair
        rcases hpair with rfl | rfl <;> omega
      | tor =>
        simp only [instrPair, List.mem_cons, List.not_mem_nil, or_false] at hpair
        rcases hpair with rfl | rfl <;> omega
      | fromr =>
        simp only [instrPair, List.mem_cons, List.not_mem_nil, or_false] at hpair
        rcases hpair with rfl | rfl <;> omega
      | rfetch =>
        simp only [instrPair, List.mem_cons, List.not_mem_nil, or_false] at hpair
        rcases hpair with rfl | rfl <;> omega
      | ret =>
        simp only [instrPair, List.mem_cons, List.not_mem_nil, or_false] at hpair
        rcases hpair with rfl | rfl <;> omega
      | lit n =>
        simp only [instrPair, List.mem_cons, List.not_mem_nil, or_false] at hpair
        rcases hpair with rfl | rfl
        · omega
        · exact Nat.mod_lt _ hW0
      | str n =>
        simp only [instrPair, List.mem_cons, List.not_mem_nil, or_false] at hpair
        rcases hpair with rfl | rfl
        · omega
        · exact Nat.mod_lt _ hW0
      | pick n =>
        simp only [instrPair, List.mem_cons, List.not_mem_nil, or_false] at hpair
        rcases hpair with rfl | rfl
        · omega
        · exact Nat.mod_lt _ hW0
      | locget n =>
        simp only [instrPair, List.mem_cons, List.not_mem_nil, or_false] at hpair
        rcases hpair with rfl | rfl
        · omega
        · exact Nat.mod_lt _ hW0
      | locset n =>
        simp only [instrPair, List.mem_cons, List.not_mem_nil, or_false] at hpair
        rcases hpair with rfl | rfl
        · omega
        · exact Nat.mod_lt _ hW0
      | jmp n =>
        simp only [instrPair, List.mem_cons, List.not_mem_nil, or_false] at hpair
        rcases hpair with rfl | rfl
        · omega
        · exact Nat.mod_lt _ hW0
      | jz n =>
        simp only [instrPair, List.mem_cons, List.not_mem_nil, or_false] at hpair
        rcases hpair with rfl | rfl
        · omega
        · exact Nat.mod_lt _ hW0
      | jnz n =>
        simp only [instrPair, List.mem_cons, List.not_mem_nil, or_false] at hpair
        rcases hpair with rfl | rfl
        · omega
        · exact Nat.mod_lt _ hW0
      | call n =>
        simp only [instrPair, List.mem_cons, List.not_mem_nil, or_false] at hpair
        rcases hpair with rfl | rfl
        · omega
        · exact Nat.mod_lt _ hW0
  exact hmem_all _ hmem

def FlatCodeOk (pr : Sporos.Prog) (r : Region) : Prop :=
  (encodeFlatProg pr).length ≤ r.data.length ∧
  r.hi ≤ 2 ^ 47 ∧
  r.data.take (encodeFlatProg pr).length = encodeFlatProg pr

theorem rd64_flat_at {pr : Sporos.Prog} {r : Region} {m : Mem}
    (hmwf : m.wf) (hm : r ∈ m) (hflat : FlatCodeOk pr r) (hfit : FlatProgFit pr)
    {qIdx : Nat} (hq : qIdx < (flatProgQwords pr).length) :
    rd64 m (r.base + qIdx * 8) = some (flatProgQwords pr)[qIdx]! := by
  have hlen : (encodeFlatProg pr).length = (flatProgQwords pr).length * 8 :=
    flatMap_bytes8_length _
  have hbound : qIdx * 8 + 8 ≤ r.data.length := by
    have := hflat.1
    rw [hlen] at this
    omega
  have hlo : r.base ≤ r.base + qIdx * 8 := by omega
  have hhi : r.base + qIdx * 8 + 8 ≤ r.hi := by
    simp only [Region.hi]; omega
  apply rd64_of_mem hmwf hm hlo hhi
  have hsub : r.base + qIdx * 8 - r.base = qIdx * 8 := by omega
  rw [hsub]
  have hdt : ((r.data.drop (qIdx * 8)).take 8) =
      (((encodeFlatProg pr).drop (qIdx * 8)).take 8) := by
    have htake : r.data.take (encodeFlatProg pr).length = encodeFlatProg pr := hflat.2.2
    calc (r.data.drop (qIdx * 8)).take 8
        = ((r.data.take (qIdx * 8 + 8)).drop (qIdx * 8)) := by
            rw [List.take_drop, Nat.add_comm]
      _ = (((r.data.take (encodeFlatProg pr).length).take (qIdx * 8 + 8)).drop (qIdx * 8)) := by
            rw [List.take_take, Nat.min_eq_left (by rw [hlen]; omega)]
      _ = (((encodeFlatProg pr).take (qIdx * 8 + 8)).drop (qIdx * 8)) := by rw [htake]
      _ = ((encodeFlatProg pr).drop (qIdx * 8)).take 8 := by
            rw [List.take_drop, Nat.add_comm]
  rw [hdt]
  exact flatMap_bytes8_drop_take (flatProgQwords pr) qIdx hq (flatProgQwords_get_lt pr hfit qIdx hq)

theorem rd64_flat_nw {pr : Sporos.Prog} {r : Region} {m : Mem}
    (hmwf : m.wf) (hm : r ∈ m) (hflat : FlatCodeOk pr r) (hfit : FlatProgFit pr) :
    rd64 m (ea r.base 8) = some pr.words.length := by
  have hltW : r.base + 8 < X86Env.W := by
    have h1 := hflat.1; have h2 := hflat.2.1
    have hlen : 32 ≤ (encodeFlatProg pr).length := by
      unfold encodeFlatProg
      rw [flatMap_bytes8_length]
      simp [flatProgQwords, flatProgHeader]
      omega
    have h47 : 2 ^ 47 < X86Env.W := by decide
    simp only [Region.hi] at h2; omega
  rw [ea_add_eq hltW]
  simpa [flatProgQwords, flatProgHeader] using
    rd64_flat_at hmwf hm hflat hfit (qIdx := 1) (by simp [flatProgQwords, flatProgHeader])

theorem rd64_flat_nd {pr : Sporos.Prog} {r : Region} {m : Mem}
    (hmwf : m.wf) (hm : r ∈ m) (hflat : FlatCodeOk pr r) (hfit : FlatProgFit pr) :
    rd64 m (ea r.base 16) = some pr.data.length := by
  have hltW : r.base + 16 < X86Env.W := by
    have h1 := hflat.1; have h2 := hflat.2.1
    have hlen : 32 ≤ (encodeFlatProg pr).length := by
      unfold encodeFlatProg
      rw [flatMap_bytes8_length]
      simp [flatProgQwords, flatProgHeader]
      omega
    have h47 : 2 ^ 47 < X86Env.W := by decide
    simp only [Region.hi] at h2; omega
  rw [ea_add_eq hltW]
  simpa [flatProgQwords, flatProgHeader] using
    rd64_flat_at hmwf hm hflat hfit (qIdx := 2) (by simp [flatProgQwords, flatProgHeader])

theorem rd64_flat_data {pr : Sporos.Prog} {r : Region} {m : Mem}
    (hmwf : m.wf) (hm : r ∈ m) (hflat : FlatCodeOk pr r) (hfit : FlatProgFit pr)
    {i : Nat} (hi : i < pr.data.length) :
    let off := ((pr.data.take i).map List.length).foldl (· + ·) 0
    let boff := i * 16 + 32
    rd64 m (eaIdx r.base boff 1 0) = some off ∧
    rd64 m (eaIdx r.base boff 1 8) = some pr.data[i]!.length := by
  have hlen := flatMap_bytes8_length (flatProgQwords pr)
  have hfqlen : (flatProgQwords pr).length =
      4 + 2 * pr.data.length + 8 * pr.words.length + 2 * (pr.words.map (·.code.length)).foldl (· + ·) 0 := by
    simp [flatProgQwords, flatProgHeader, dataMetaQwords_len, wordMetaQwords_len, codePairsQwords_len]
    omega
  have h47 : 2 ^ 47 < X86Env.W := by decide
  have hrhi : r.base + (flatProgQwords pr).length * 8 < X86Env.W := by
    have h1 := hflat.1; have h2 := hflat.2.1
    unfold encodeFlatProg at h1
    rw [hlen] at h1; simp only [Region.hi] at h2; omega
  have hdm := dataMetaQwords_get pr.data 0 i hi
  simp only [Nat.zero_add] at hdm
  have hq0 : (flatProgQwords pr)[4 + 2 * i]? =
      some (((pr.data.take i).map List.length).foldl (· + ·) 0) := by
    simp only [flatProgQwords, List.append_assoc]
    rw [List.getElem?_append_right (by have : (flatProgHeader pr).length = 4 := rfl; omega),
      show (flatProgHeader pr).length = 4 from rfl,
      show 4 + 2 * i - 4 = 2 * i by omega]
    rw [List.getElem?_append_left (by rw [dataMetaQwords_len]; omega)]
    exact hdm.1
  have hq1 : (flatProgQwords pr)[4 + 2 * i + 1]? = some pr.data[i]!.length := by
    simp only [flatProgQwords, List.append_assoc]
    rw [List.getElem?_append_right (by have : (flatProgHeader pr).length = 4 := rfl; omega),
      show (flatProgHeader pr).length = 4 from rfl,
      show 4 + 2 * i + 1 - 4 = 2 * i + 1 by omega]
    rw [List.getElem?_append_left (by rw [dataMetaQwords_len]; omega)]
    exact hdm.2
  have hlt0 : 4 + 2 * i < (flatProgQwords pr).length := by rw [hfqlen]; omega
  have hlt1 : 4 + 2 * i + 1 < (flatProgQwords pr).length := by rw [hfqlen]; omega
  have h0 := rd64_flat_at hmwf hm hflat hfit hlt0
  have h1 := rd64_flat_at hmwf hm hflat hfit hlt1
  have hget0 : (flatProgQwords pr)[4 + 2 * i]! = ((pr.data.take i).map List.length).foldl (· + ·) 0 := by
    rw [List.getElem!_eq_getElem?_getD, hq0]; rfl
  have hget1 : (flatProgQwords pr)[4 + 2 * i + 1]! = pr.data[i]!.length := by
    rw [List.getElem!_eq_getElem?_getD, hq1]; rfl
  rw [hget0] at h0; rw [hget1] at h1
  constructor
  · rw [eaIdx_add_eq (by omega), show r.base + 1 * (i * 16 + 32) + 0 = r.base + (4 + 2 * i) * 8 by omega]
    exact h0
  · rw [eaIdx_add_eq (by omega), show r.base + 1 * (i * 16 + 32) + 8 = r.base + (4 + 2 * i + 1) * 8 by omega]
    exact h1

theorem rd64_flat_word {pr : Sporos.Prog} {r : Region} {m : Mem}
    (hmwf : m.wf) (hm : r ∈ m) (hflat : FlatCodeOk pr r) (hfit : FlatProgFit pr)
    {w : Nat} (hw : w < pr.words.length) :
    let boff := w * 64 + pr.data.length * 16 + 32
    let woff := flatProgBaseOff pr + ((pr.words.take w).map fun x => x.code.length * 16).foldl (· + ·) 0
    rd64 m (eaIdx r.base boff 1 0) = some pr.words[w]!.ins ∧
    rd64 m (eaIdx r.base boff 1 16) = some pr.words[w]!.locals ∧
    rd64 m (eaIdx r.base boff 1 24) = some pr.words[w]!.code.length ∧
    rd64 m (eaIdx r.base boff 1 32) = some woff := by
  have hlen := flatMap_bytes8_length (flatProgQwords pr)
  have hfqlen : (flatProgQwords pr).length =
      4 + 2 * pr.data.length + 8 * pr.words.length + 2 * (pr.words.map (·.code.length)).foldl (· + ·) 0 := by
    simp [flatProgQwords, flatProgHeader, dataMetaQwords_len, wordMetaQwords_len, codePairsQwords_len]
    omega
  have h47 : 2 ^ 47 < X86Env.W := by decide
  have hrhi : r.base + (flatProgQwords pr).length * 8 < X86Env.W := by
    have h1 := hflat.1; have h2 := hflat.2.1
    unfold encodeFlatProg at h1
    rw [hlen] at h1; simp only [Region.hi] at h2; omega
  have hwm := wordMetaQwords_get pr.words (flatProgBaseOff pr) w hw
  have hpre : (flatProgHeader pr ++ dataMetaQwords 0 pr.data).length = 4 + 2 * pr.data.length := by
    simp [flatProgHeader, dataMetaQwords_len]; omega
  have hget : ∀ k ∈ [0, 2, 3, 4],
      (flatProgQwords pr)[4 + 2 * pr.data.length + 8 * w + k]? =
        (wordMetaQwords (flatProgBaseOff pr) pr.words)[8 * w + k]? := by
    intro k hk
    have hkl : k ≤ 4 := by simp at hk; rcases hk with rfl | rfl | rfl | rfl <;> omega
    have h1 : 4 + 2 * pr.data.length ≤ 4 + 2 * pr.data.length + 8 * w + k := by omega
    have h2 : 8 * w + k < (wordMetaQwords (flatProgBaseOff pr) pr.words).length := by
      rw [wordMetaQwords_len]; omega
    simp only [flatProgQwords]
    rw [List.append_assoc]
    rw [List.getElem?_append_right (by rw [hpre]; exact h1)]
    rw [hpre, show 4 + 2 * pr.data.length + 8 * w + k - (4 + 2 * pr.data.length) = 8 * w + k by omega]
    rw [List.getElem?_append_left h2]
  have stepAt : ∀ k val, k ∈ [0, 2, 3, 4] →
      (wordMetaQwords (flatProgBaseOff pr) pr.words)[8 * w + k]? = some val →
      rd64 m (r.base + (4 + 2 * pr.data.length + 8 * w + k) * 8) = some val := by
    intro k val hk hkv
    have hkl : k ≤ 4 := by simp at hk; rcases hk with rfl | rfl | rfl | rfl <;> omega
    have hlt : 4 + 2 * pr.data.length + 8 * w + k < (flatProgQwords pr).length := by
      rw [hfqlen]; omega
    have hr := rd64_flat_at hmwf hm hflat hfit hlt
    have hq := hget k hk
    rw [hkv] at hq
    have hget' : (flatProgQwords pr)[4 + 2 * pr.data.length + 8 * w + k]! = val := by
      rw [List.getElem!_eq_getElem?_getD, hq]; rfl
    rwa [hget'] at hr
  dsimp only at hwm
  refine ⟨?_, ?_, ?_, ?_⟩
  · have := stepAt 0 _ (by simp) hwm.1
    rw [eaIdx_add_eq (by omega), show r.base + 1 * (w * 64 + pr.data.length * 16 + 32) + 0 =
      r.base + (4 + 2 * pr.data.length + 8 * w + 0) * 8 by omega]
    exact this
  · have := stepAt 2 _ (by simp) hwm.2.1
    rw [eaIdx_add_eq (by omega), show r.base + 1 * (w * 64 + pr.data.length * 16 + 32) + 16 =
      r.base + (4 + 2 * pr.data.length + 8 * w + 2) * 8 by omega]
    exact this
  · have := stepAt 3 _ (by simp) hwm.2.2.1
    rw [eaIdx_add_eq (by omega), show r.base + 1 * (w * 64 + pr.data.length * 16 + 32) + 24 =
      r.base + (4 + 2 * pr.data.length + 8 * w + 3) * 8 by omega]
    exact this
  · have := stepAt 4 _ (by simp) hwm.2.2.2
    rw [eaIdx_add_eq (by omega), show r.base + 1 * (w * 64 + pr.data.length * 16 + 32) + 32 =
      r.base + (4 + 2 * pr.data.length + 8 * w + 4) * 8 by omega]
    exact this

theorem rd64_flat_insn {pr : Sporos.Prog} {r : Region} {m : Mem}
    (hmwf : m.wf) (hm : r ∈ m) (hflat : FlatCodeOk pr r) (hfit : FlatProgFit pr)
    {w pc : Nat} (hw : w < pr.words.length) (hpc : pc < pr.words[w]!.code.length) :
    let woff := flatProgBaseOff pr + ((pr.words.take w).map fun x => x.code.length * 16).foldl (· + ·) 0
    let boff := pc * 16 + woff
    let ins := pr.words[w]!.code[pc]!
    rd64 m (eaIdx r.base boff 1 0) = some (instrPair ins).1 ∧
    rd64 m (eaIdx r.base boff 1 8) = some (instrPair ins).2 := by
  have hlen := flatMap_bytes8_length (flatProgQwords pr)
  have hfqlen : (flatProgQwords pr).length =
      4 + 2 * pr.data.length + 8 * pr.words.length + 2 * (pr.words.map (·.code.length)).foldl (· + ·) 0 := by
    simp [flatProgQwords, flatProgHeader, dataMetaQwords_len, wordMetaQwords_len, codePairsQwords_len]
    omega
  have h47 : 2 ^ 47 < X86Env.W := by decide
  have hrhi : r.base + (flatProgQwords pr).length * 8 < X86Env.W := by
    have h1 := hflat.1; have h2 := hflat.2.1
    unfold encodeFlatProg at h1
    rw [hlen] at h1; simp only [Region.hi] at h2; omega
  have hcp := codePairsQwords_get pr.words w pc hw hpc
  have hsum_le :
      ((pr.words.take w).map (·.code.length)).foldl (· + ·) 0 + pr.words[w]!.code.length ≤
        (pr.words.map (·.code.length)).foldl (· + ·) 0 := by
    have go : ∀ (ws : List Sporos.Word) (idx : Nat), idx < ws.length →
        ((ws.take idx).map (·.code.length)).foldl (· + ·) 0 + ws[idx]!.code.length ≤
          (ws.map (·.code.length)).foldl (· + ·) 0 := by
      intro ws
      induction ws with
      | nil => intro idx hi; simp at hi
      | cons a l ih =>
        intro idx hi
        cases idx with
        | zero =>
          simp only [List.take_zero, List.map_nil, List.foldl_nil, Nat.zero_add,
            List.getElem!_eq_getElem?_getD, List.getElem?_cons_zero, Option.getD_some,
            List.map_cons, List.foldl_cons]
          rw [list_foldl_add]
          omega
        | succ k =>
          have hk : k < l.length := by simp at hi; omega
          have hr := ih k hk
          simp only [List.take_succ_cons, List.map_cons, List.foldl_cons, Nat.zero_add,
            List.getElem!_eq_getElem?_getD, List.getElem?_cons_succ] at hr ⊢
          rw [list_foldl_add, list_foldl_add (l.map (·.code.length))]
          omega
    exact go pr.words w hw
  have hmul_off :
      ((pr.words.take w).map fun x => x.code.length * 16).foldl (· + ·) 0 =
        16 * ((pr.words.take w).map (·.code.length)).foldl (· + ·) 0 := by
    have go : ∀ (ws : List Sporos.Word) (z : Nat),
        (ws.map (fun x => x.code.length * 16)).foldl (· + ·) (16 * z) =
          16 * (ws.map (·.code.length)).foldl (· + ·) z := by
      intro ws
      induction ws with
      | nil => intro z; simp
      | cons x xs ih =>
        intro z
        simp only [List.map_cons, List.foldl_cons]
        rw [show 16 * z + x.code.length * 16 = 16 * (z + x.code.length) by omega]
        exact ih (z + x.code.length)
    simpa using go (pr.words.take w) 0
  have hpre : (flatProgHeader pr ++ dataMetaQwords 0 pr.data ++
      wordMetaQwords (flatProgBaseOff pr) pr.words).length =
      4 + 2 * pr.data.length + 8 * pr.words.length := by
    simp [flatProgHeader, dataMetaQwords_len, wordMetaQwords_len, Nat.add_assoc]
    omega
  have hpre8 : (4 + 2 * pr.data.length + 8 * pr.words.length) * 8 = flatProgBaseOff pr := by
    simp [flatProgBaseOff]; omega
  let baseQ := 4 + 2 * pr.data.length + 8 * pr.words.length +
    2 * ((pr.words.take w).map (·.code.length)).foldl (· + ·) 0
  have hlt0 : baseQ + 2 * pc < (flatProgQwords pr).length := by
    rw [hfqlen]; dsimp [baseQ]; omega
  have hlt1 : baseQ + 2 * pc + 1 < (flatProgQwords pr).length := by
    rw [hfqlen]; dsimp [baseQ]; omega
  have hq0 : (flatProgQwords pr)[baseQ + 2 * pc]? = some (instrPair pr.words[w]!.code[pc]!).1 := by
    simp only [flatProgQwords, baseQ]
    rw [List.getElem?_append_right (by rw [hpre]; omega), hpre]
    rw [show 4 + 2 * pr.data.length + 8 * pr.words.length +
        2 * ((pr.words.take w).map (·.code.length)).foldl (· + ·) 0 + 2 * pc -
        (4 + 2 * pr.data.length + 8 * pr.words.length) =
        2 * ((pr.words.take w).map (·.code.length)).foldl (· + ·) 0 + 2 * pc by omega]
    exact hcp.1
  have hq1 : (flatProgQwords pr)[baseQ + 2 * pc + 1]? = some (instrPair pr.words[w]!.code[pc]!).2 := by
    simp only [flatProgQwords, baseQ]
    rw [List.getElem?_append_right (by rw [hpre]; omega), hpre]
    rw [show 4 + 2 * pr.data.length + 8 * pr.words.length +
        2 * ((pr.words.take w).map (·.code.length)).foldl (· + ·) 0 + 2 * pc + 1 -
        (4 + 2 * pr.data.length + 8 * pr.words.length) =
        2 * ((pr.words.take w).map (·.code.length)).foldl (· + ·) 0 + 2 * pc + 1 by omega]
    exact hcp.2
  have h0 := rd64_flat_at hmwf hm hflat hfit hlt0
  have h1 := rd64_flat_at hmwf hm hflat hfit hlt1
  have hget0 : (flatProgQwords pr)[baseQ + 2 * pc]! = (instrPair pr.words[w]!.code[pc]!).1 := by
    rw [List.getElem!_eq_getElem?_getD, hq0]; rfl
  have hget1 : (flatProgQwords pr)[baseQ + 2 * pc + 1]! = (instrPair pr.words[w]!.code[pc]!).2 := by
    rw [List.getElem!_eq_getElem?_getD, hq1]; rfl
  rw [hget0] at h0; rw [hget1] at h1
  have hboff : r.base + 1 * (pc * 16 + (flatProgBaseOff pr +
      ((pr.words.take w).map fun x => x.code.length * 16).foldl (· + ·) 0)) =
      r.base + (baseQ + 2 * pc) * 8 := by
    rw [hmul_off, ← hpre8]; dsimp [baseQ]; omega
  constructor
  · rw [eaIdx_add_eq (by rw [Nat.add_zero, hboff]; omega), hboff, Nat.add_zero]
    exact h0
  · rw [eaIdx_add_eq (by rw [hboff]; omega), hboff, Nat.add_assoc,
      show (baseQ + 2 * pc) * 8 + 8 = (baseQ + 2 * pc + 1) * 8 by omega]
    exact h1

/-! ### Arena content relations (`ds`, `aux`, `rs`, `loc`, `mem`, `tabs`, `inp`, `out`) -/

def PairListRel (data : List Nat) (len : Nat) (xs : List SporosProvLen.LVal) : Prop :=
  xs.length = len ∧ len * 16 ≤ data.length ∧
  ∀ i, i < xs.length →
    leVal ((data.drop (i * 16)).take 8) = xs.reverse[i]!.w ∧
    leVal ((data.drop (i * 16 + 8)).take 8) = xs.reverse[i]!.plen

theorem pairListRel_get_top {data : List Nat} {len : Nat} {xs : List SporosProvLen.LVal}
    (hr : PairListRel data len xs) {k : Nat} (hk : k < xs.length) :
    leVal ((data.drop ((len - 1 - k) * 16)).take 8) = xs[k]!.w ∧
    leVal ((data.drop ((len - 1 - k) * 16 + 8)).take 8) = xs[k]!.plen := by
  have hi : len - 1 - k < xs.length := by rw [← hr.1]; omega
  have hrev := hr.2.2 (len - 1 - k) hi
  have hidx : xs.reverse[len - 1 - k]! = xs[k]! := by
    have hl : len = xs.length := hr.1.symm
    subst hl
    rw [List.getElem!_eq_getElem?_getD, List.getElem!_eq_getElem?_getD]
    rw [List.getElem?_reverse (by omega)]
    have hsub : xs.length - 1 - (xs.length - 1 - k) = k := by omega
    rw [hsub]
  simpa [hidx] using hrev

theorem pairListRel_push {data : List Nat} {len : Nat} {xs : List SporosProvLen.LVal}
    (hr : PairListRel data len xs) (v : SporosProvLen.LVal)
    (hv_w : v.w < X86Env.W) (hv_p : v.plen < X86Env.W)
    (hlen : len * 16 + 16 ≤ data.length) :
    PairListRel (writeBytes (writeBytes data (len * 16) (bytes8 v.w)) (len * 16 + 8) (bytes8 v.plen))
      (len + 1) (v :: xs) := by
  refine ⟨by simp [hr.1], by rw [writeBytes_length, writeBytes_length]; omega, ?_⟩
  intro i hi
  simp only [List.length_cons, List.reverse_cons] at hi ⊢
  by_cases hlt : i < xs.length
  · have hi_len : i < len := by rw [← hr.1]; exact hlt
    have hdisj0 := writeBytes_drop_take_disjoint (writeBytes data (len * 16) (bytes8 v.w))
      (len * 16 + 8) (i * 16) 8 (bytes8 v.plen) (by rw [bytes8_length]; omega)
      (by rw [writeBytes_length]; omega)
    have hdisj1 := writeBytes_drop_take_disjoint data (len * 16) (i * 16) 8 (bytes8 v.w)
      (by rw [bytes8_length]; omega) (by omega)
    have hdisj2 := writeBytes_drop_take_disjoint (writeBytes data (len * 16) (bytes8 v.w))
      (len * 16 + 8) (i * 16 + 8) 8 (bytes8 v.plen) (by rw [bytes8_length]; omega)
      (by rw [writeBytes_length]; omega)
    have hdisj3 := writeBytes_drop_take_disjoint data (len * 16) (i * 16 + 8) 8 (bytes8 v.w)
      (by rw [bytes8_length]; omega) (by omega)
    have hr_i := hr.2.2 i hlt
    have hget : (xs.reverse ++ [v])[i]! = xs.reverse[i]! := by
      rw [List.getElem!_eq_getElem?_getD, List.getElem!_eq_getElem?_getD]
      rw [List.getElem?_append_left (by simp [hlt])]
    rw [hdisj0, hdisj1, hdisj2, hdisj3, hget]
    exact hr_i
  · have heq : i = xs.length := by omega
    subst heq
    rw [← hr.1]
    have hdisj := writeBytes_drop_take_disjoint (writeBytes data (xs.length * 16) (bytes8 v.w))
      (xs.length * 16 + 8) (xs.length * 16) 8 (bytes8 v.plen) (by rw [bytes8_length]; omega)
      (by rw [writeBytes_length, hr.1]; omega)
    have hw0 := leVal_writeBytes_same data (xs.length * 16) v.w (by rw [hr.1]; omega)
    have hw1 := leVal_writeBytes_same (writeBytes data (xs.length * 16) (bytes8 v.w)) (xs.length * 16 + 8) v.plen
      (by rw [writeBytes_length, hr.1]; omega)
    have hget : (xs.reverse ++ [v])[xs.length]! = v := by
      rw [List.getElem!_eq_getElem?_getD]
      rw [List.getElem?_append_right (by simp), List.length_reverse, Nat.sub_self]
      rfl
    rw [hdisj, hw0, hw1, hget, Nat.mod_eq_of_lt hv_w, Nat.mod_eq_of_lt hv_p]
    exact ⟨rfl, rfl⟩

theorem pairListRel_drop {data : List Nat} {len k : Nat} {xs : List SporosProvLen.LVal}
    (hr : PairListRel data len xs) (hk : k ≤ xs.length) :
    PairListRel data (len - k) (xs.drop k) := by
  refine ⟨by simp [hr.1], Nat.le_trans (Nat.mul_le_mul_right 16 (Nat.sub_le len k)) hr.2.1, ?_⟩
  intro i hi
  simp only [List.length_drop] at hi
  have hlt : i < xs.length := by omega
  have hr_i := hr.2.2 i hlt
  have hrev : (xs.drop k).reverse[i]! = xs.reverse[i]! := by
    rw [List.getElem!_eq_getElem?_getD, List.getElem!_eq_getElem?_getD]
    rw [List.getElem?_reverse (by simp [hi]), List.getElem?_reverse hlt]
    rw [List.length_drop, List.getElem?_drop]
    have hidx : k + (xs.length - k - 1 - i) = xs.length - 1 - i := by omega
    rw [hidx]
  rw [hrev]
  exact hr_i

theorem pairListRel_extend_zero {data : List Nat} {len ext : Nat} {xs : List SporosProvLen.LVal}
    (hr : PairListRel data len xs) (hlen : len * 16 ≤ data.length) :
    PairListRel (data ++ List.replicate ext 0) len xs := by
  refine ⟨hr.1, by rw [List.length_append, List.length_replicate]; exact Nat.le_trans hlen (Nat.le_add_right _ _), ?_⟩
  intro i hi
  have hi_len : i < len := by rw [← hr.1]; exact hi
  have h1 : ((data ++ List.replicate ext 0).drop (i * 16)).take 8 = (data.drop (i * 16)).take 8 := by
    rw [List.drop_append_of_le_length (by omega), List.take_append_of_le_length (by rw [List.length_drop]; omega)]
  have h2 : ((data ++ List.replicate ext 0).drop (i * 16 + 8)).take 8 = (data.drop (i * 16 + 8)).take 8 := by
    rw [List.drop_append_of_le_length (by omega), List.take_append_of_le_length (by rw [List.length_drop]; omega)]
  rw [h1, h2]
  exact hr.2.2 i hi

def rsLocBase (rs : List SporosProvLen.LFrame) : Nat :=
  (rs.map (·.locs.length)).foldl (· + ·) 0

def RsLocRel (rsData locData : List Nat) (locLen : Nat) : List SporosProvLen.LFrame → Prop
  | [] => locLen = 0
  | f :: rest =>
    let lb := rsLocBase rest
    locLen = lb + f.locs.length ∧
    (rest.length + 1) * 32 ≤ rsData.length ∧
    (lb + f.locs.length) * 16 ≤ locData.length ∧
    leVal ((rsData.drop (rest.length * 32)).take 8) = f.word ∧
    leVal ((rsData.drop (rest.length * 32 + 8)).take 8) = f.pc ∧
    leVal ((rsData.drop (rest.length * 32 + 16)).take 8) = lb ∧
    leVal ((rsData.drop (rest.length * 32 + 24)).take 8) = f.locs.length ∧
    (∀ j, j < f.locs.length →
      leVal ((locData.drop ((lb + j) * 16)).take 8) = f.locs[j]!.w ∧
      leVal ((locData.drop ((lb + j) * 16 + 8)).take 8) = f.locs[j]!.plen) ∧
    RsLocRel rsData locData lb rest

theorem rsLocRel_write_rs_after {rsData locData : List Nat} {locLen off sz : Nat} {bs : List Nat}
    {rs : List SporosProvLen.LFrame} (hr : RsLocRel rsData locData locLen rs)
    (hoff : rs.length * 32 ≤ off) (hbs : bs.length = sz) :
    RsLocRel (writeBytes rsData off bs) locData locLen rs := by
  induction rs generalizing locLen with
  | nil => exact hr
  | cons f rest ih =>
    simp only [RsLocRel] at hr ⊢
    rcases hr with ⟨hl, hrsLen, hlocLen, hw, hpc, hlb, hn, hlocs, hrest⟩
    have hrlen : (f :: rest).length = rest.length + 1 := rfl
    rw [hrlen] at hoff
    refine ⟨hl, by rw [writeBytes_length]; exact hrsLen, hlocLen, ?_, ?_, ?_, ?_, hlocs, ih hrest (by omega)⟩
    · rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [hbs]; omega) (by omega)]; exact hw
    · rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [hbs]; omega) (by omega)]; exact hpc
    · rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [hbs]; omega) (by omega)]; exact hlb
    · rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [hbs]; omega) (by omega)]; exact hn

theorem rsLocRel_write_loc_after {rsData locData : List Nat} {locLen off : Nat} {bs : List Nat}
    {rs : List SporosProvLen.LFrame} (hr : RsLocRel rsData locData locLen rs)
    (hoff : locLen * 16 ≤ off) :
    RsLocRel rsData (writeBytes locData off bs) locLen rs := by
  induction rs generalizing locLen with
  | nil => exact hr
  | cons f rest ih =>
    simp only [RsLocRel] at hr ⊢
    rcases hr with ⟨hl, hrsLen, hlocLen, hw, hpc, hlb, hn, hlocs, hrest⟩
    refine ⟨hl, hrsLen, by rw [writeBytes_length]; exact hlocLen, hw, hpc, hlb, hn, ?_, ih hrest (by omega)⟩
    intro j hj
    have h1 := hlocs j hj
    rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by omega) (by omega),
      writeBytes_drop_take_disjoint _ _ _ _ _ (by omega) (by omega)]
    exact h1

theorem rsLocRel_extend_rs_zero {rsData locData : List Nat} {locLen ext : Nat}
    {rs : List SporosProvLen.LFrame} (hr : RsLocRel rsData locData locLen rs)
    (_hlen : rs.length * 32 ≤ rsData.length) :
    RsLocRel (rsData ++ List.replicate ext 0) locData locLen rs := by
  induction rs generalizing locLen with
  | nil => exact hr
  | cons f rest ih =>
    simp only [RsLocRel] at hr ⊢
    rcases hr with ⟨hl, hrsLen, hlocLen, hw, hpc, hlb, hn, hlocs, hrest⟩
    have htake : ∀ a, a + 8 ≤ (rest.length + 1) * 32 →
        ((rsData ++ List.replicate ext 0).drop a).take 8 = (rsData.drop a).take 8 := by
      intro a ha
      rw [List.drop_append_of_le_length (by omega), List.take_append_of_le_length (by rw [List.length_drop]; omega)]
    refine ⟨hl, by rw [List.length_append, List.length_replicate]; omega, hlocLen,
      ?_, ?_, ?_, ?_, hlocs, ih hrest (by omega)⟩
    · rw [htake _ (by omega)]; exact hw
    · rw [htake _ (by omega)]; exact hpc
    · rw [htake _ (by omega)]; exact hlb
    · rw [htake _ (by omega)]; exact hn

theorem rsLocRel_extend_loc_zero {rsData locData : List Nat} {locLen ext : Nat}
    {rs : List SporosProvLen.LFrame} (hr : RsLocRel rsData locData locLen rs)
    (_hlen : locLen * 16 ≤ locData.length) :
    RsLocRel rsData (locData ++ List.replicate ext 0) locLen rs := by
  induction rs generalizing locLen with
  | nil => exact hr
  | cons f rest ih =>
    simp only [RsLocRel] at hr ⊢
    rcases hr with ⟨hl, hrsLen, hlocLen, hw, hpc, hlb, hn, hlocs, hrest⟩
    refine ⟨hl, hrsLen, by rw [List.length_append, List.length_replicate]; omega,
      hw, hpc, hlb, hn, ?_, ih hrest (by omega)⟩
    intro j hj
    have htake : ∀ a, a + 8 ≤ (rsLocBase rest + f.locs.length) * 16 →
        ((locData ++ List.replicate ext 0).drop a).take 8 = (locData.drop a).take 8 := by
      intro a ha
      rw [List.drop_append_of_le_length (by omega), List.take_append_of_le_length (by rw [List.length_drop]; omega)]
    have h1 := hlocs j hj
    rw [htake _ (by omega), htake _ (by omega)]
    exact h1

theorem rsLocRel_adv_top {rsData locData : List Nat} {locLen : Nat}
    {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hr : RsLocRel rsData locData locLen (f :: rest))
    (hlen : (rest.length + 1) * 32 ≤ rsData.length) (hpc : f.pc + 1 < X86Env.W) :
    RsLocRel (writeBytes rsData (rest.length * 32 + 8) (bytes8 (f.pc + 1))) locData locLen
      ({ f with pc := f.pc + 1 } :: rest) := by
  simp only [RsLocRel] at hr ⊢
  rcases hr with ⟨hl, hrsLen, hlocLen, hw, hpc0, hlb, hn, hlocs, hrest⟩
  refine ⟨hl, by rw [writeBytes_length]; exact hrsLen, hlocLen,
    ?_, ?_, ?_, ?_, hlocs, rsLocRel_write_rs_after hrest (by omega) (bytes8_length _)⟩
  · rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by omega)]; exact hw
  · rw [leVal_writeBytes_same _ _ _ (by omega), Nat.mod_eq_of_lt hpc]
  · rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by omega)]; exact hlb
  · rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by omega)]; exact hn

theorem rsLocRel_setTop_pc {rsData locData : List Nat} {locLen t : Nat}
    {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hr : RsLocRel rsData locData locLen (f :: rest))
    (hlen : (rest.length + 1) * 32 ≤ rsData.length) (ht : t < X86Env.W) :
    RsLocRel (writeBytes rsData (rest.length * 32 + 8) (bytes8 t)) locData locLen
      ({ f with pc := t } :: rest) := by
  simp only [RsLocRel] at hr ⊢
  rcases hr with ⟨hl, hrsLen, hlocLen, hw, _, hlb, hn, hlocs, hrest⟩
  refine ⟨hl, by rw [writeBytes_length]; exact hrsLen, hlocLen,
    ?_, ?_, ?_, ?_, hlocs, rsLocRel_write_rs_after hrest (by omega) (bytes8_length _)⟩
  · rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by omega)]; exact hw
  · rw [leVal_writeBytes_same _ _ _ (by omega), Nat.mod_eq_of_lt ht]
  · rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by omega)]; exact hlb
  · rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by omega)]; exact hn

theorem rsLocRel_locset {rsData locData : List Nat} {locLen i : Nat} {v : SporosProvLen.LVal}
    {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hr : RsLocRel rsData locData locLen (f :: rest))
    (hi : i < f.locs.length) (hv_w : v.w < X86Env.W) (hv_p : v.plen < X86Env.W)
    (hadv_rs : (rest.length + 1) * 32 ≤ rsData.length) (hpc : f.pc + 1 < X86Env.W)
    (hloc_len : locLen * 16 ≤ locData.length) :
    let lb := rsLocBase rest
    let locData' := writeBytes (writeBytes locData ((lb + i) * 16) (bytes8 v.w))
      ((lb + i) * 16 + 8) (bytes8 v.plen)
    let rsData' := writeBytes rsData (rest.length * 32 + 8) (bytes8 (f.pc + 1))
    RsLocRel rsData' locData' locLen ({ f with pc := f.pc + 1, locs := f.locs.set i v } :: rest) := by
  simp only [RsLocRel] at hr ⊢
  rcases hr with ⟨hl, hrsLen, hlocLen, hw, _, hlb, hn, hlocs, hrest⟩
  refine ⟨by simp [hl], by rw [writeBytes_length]; exact hrsLen,
    by simp only [List.length_set]; rw [writeBytes_length, writeBytes_length]; exact hlocLen,
    ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by omega)]; exact hw
  · rw [leVal_writeBytes_same _ _ _ (by omega), Nat.mod_eq_of_lt hpc]
  · rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by omega)]; exact hlb
  · simp only [List.length_set]
    rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by omega)]; exact hn
  · intro j hj
    simp only [List.length_set] at hj
    by_cases heq : j = i
    · subst heq
      have hdisj := writeBytes_drop_take_disjoint
        (writeBytes locData ((rsLocBase rest + j) * 16) (bytes8 v.w))
        ((rsLocBase rest + j) * 16 + 8) ((rsLocBase rest + j) * 16) 8 (bytes8 v.plen)
        (by rw [bytes8_length]; omega) (by rw [writeBytes_length]; omega)
      have hw0 := leVal_writeBytes_same locData ((rsLocBase rest + j) * 16) v.w (by omega)
      have hw1 := leVal_writeBytes_same (writeBytes locData ((rsLocBase rest + j) * 16) (bytes8 v.w))
        ((rsLocBase rest + j) * 16 + 8) v.plen (by rw [writeBytes_length]; omega)
      have hget : (f.locs.set j v)[j]! = v := by
        rw [List.getElem!_eq_getElem?_getD, mem_set_self_get _ j v hi]; rfl
      rw [hdisj, hw0, hw1, hget, Nat.mod_eq_of_lt hv_w, Nat.mod_eq_of_lt hv_p]
      exact ⟨rfl, rfl⟩
    · have hget : (f.locs.set i v)[j]! = f.locs[j]! := by
        rw [List.getElem!_eq_getElem?_getD, List.getElem!_eq_getElem?_getD,
          mem_set_ne_get _ v (Ne.symm heq)]
      rw [hget]
      have hdisj0 : ((writeBytes (writeBytes locData ((rsLocBase rest + i) * 16) (bytes8 v.w))
          ((rsLocBase rest + i) * 16 + 8) (bytes8 v.plen)).drop ((rsLocBase rest + j) * 16)).take 8 =
          (locData.drop ((rsLocBase rest + j) * 16)).take 8 := by
        rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega)
            (by rw [writeBytes_length]; omega),
          writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by omega)]
      have hdisj1 : ((writeBytes (writeBytes locData ((rsLocBase rest + i) * 16) (bytes8 v.w))
          ((rsLocBase rest + i) * 16 + 8) (bytes8 v.plen)).drop ((rsLocBase rest + j) * 16 + 8)).take 8 =
          (locData.drop ((rsLocBase rest + j) * 16 + 8)).take 8 := by
        rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega)
            (by rw [writeBytes_length]; omega),
          writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; omega) (by omega)]
      rw [hdisj0, hdisj1]
      exact hlocs j hj
  · exact rsLocRel_write_rs_after
      (rsLocRel_write_loc_after (rsLocRel_write_loc_after hrest (by omega)) (by omega))
      (by omega) (bytes8_length _)

def MemRel (mwData mpData : List Nat) (msize : Nat) (mem : SporosProvLen.LMem) : Prop :=
  msize ≤ mwData.length ∧ msize * 8 ≤ mpData.length ∧
  ∀ x, x < msize →
    mwData[x]? = some ((mem x).w % 256) ∧
    leVal ((mpData.drop (x * 8)).take 8) = (mem x).plen % X86Env.W

inductive ChainRel (kv : List Nat) (kvLen : Nat) : Nat → List (Nat × SporosProvLen.LVal) → Prop
  | nil : ChainRel kv kvLen (X86Env.W - 1) []
  | cons {idx next k : Nat} {v : SporosProvLen.LVal} {rest : List (Nat × SporosProvLen.LVal)} :
      idx < kvLen →
      idx * 32 + 32 ≤ kv.length →
      leVal ((kv.drop (idx * 32)).take 8) = k →
      leVal ((kv.drop (idx * 32 + 8)).take 8) = v.w →
      leVal ((kv.drop (idx * 32 + 16)).take 8) = v.plen →
      leVal ((kv.drop (idx * 32 + 24)).take 8) = next →
      ChainRel kv kvLen next rest →
      ChainRel kv kvLen idx ((k, v) :: rest)

theorem chainRel_write_after {kv : List Nat} {kvLen off : Nat} {bs : List Nat}
    {idx : Nat} {l : List (Nat × SporosProvLen.LVal)}
    (hc : ChainRel kv kvLen idx l) (hoff : kvLen * 32 ≤ off) :
    ChainRel (writeBytes kv off bs) kvLen idx l := by
  induction hc with
  | nil => exact .nil
  | @cons idx next k v rest hidx hlen hk hw hp hnx hrest ih =>
    refine .cons hidx (by rw [writeBytes_length]; exact hlen) ?_ ?_ ?_ ?_ ih
    · rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by omega) (by omega)]; exact hk
    · rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by omega) (by omega)]; exact hw
    · rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by omega) (by omega)]; exact hp
    · rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by omega) (by omega)]; exact hnx

theorem chainRel_extend_zero {kv : List Nat} {kvLen ext : Nat}
    {idx : Nat} {l : List (Nat × SporosProvLen.LVal)}
    (hc : ChainRel kv kvLen idx l) (_hlen : kvLen * 32 ≤ kv.length) :
    ChainRel (kv ++ List.replicate ext 0) kvLen idx l := by
  induction hc with
  | nil => exact .nil
  | @cons idx next k v rest hidx hlen hk hw hp hnx hrest ih =>
    have htake : ∀ a, a + 8 ≤ idx * 32 + 32 →
        ((kv ++ List.replicate ext 0).drop a).take 8 = (kv.drop a).take 8 := by
      intro a ha
      rw [List.drop_append_of_le_length (by omega), List.take_append_of_le_length (by rw [List.length_drop]; omega)]
    refine .cons hidx (by rw [List.length_append, List.length_replicate]; omega) ?_ ?_ ?_ ?_ ih
    · rw [htake _ (by omega)]; exact hk
    · rw [htake _ (by omega)]; exact hw
    · rw [htake _ (by omega)]; exact hp
    · rw [htake _ (by omega)]; exact hnx

def TabsRel (tabsData kvData : List Nat) (kvLen : Nat)
    (tabs : List (List (Nat × SporosProvLen.LVal))) : Prop :=
  tabs.length * 8 ≤ tabsData.length ∧
  ∀ t, t < tabs.length → ChainRel kvData kvLen (leVal ((tabsData.drop (t * 8)).take 8)) tabs[t]!

def InpRel (inpData : List Nat) (ipos inpLen : Nat) (inp : List Nat) : Prop :=
  ipos + inp.length = inpLen ∧ inpLen ≤ inpData.length ∧
  ∀ j, j < inp.length → inpData[ipos + j]? = some (inp[j]! % 256)

def OutRel (outData : List Nat) (outLen : Nat) (out : List Nat) : Prop :=
  out.length = outLen ∧ outLen ≤ outData.length ∧
  ∀ j, j < out.length → outData[j]? = some out.reverse[j]!

theorem outRel_extend_zero {outData : List Nat} {outLen ext : Nat} {out : List Nat}
    (ho : OutRel outData outLen out) :
    OutRel (outData ++ List.replicate ext 0) outLen out := by
  refine ⟨ho.1, by rw [List.length_append, List.length_replicate]; exact Nat.le_trans ho.2.1 (Nat.le_add_right _ _), ?_⟩
  intro j hj
  rw [List.getElem?_append_left (by have := ho.1; have := ho.2.1; omega)]
  exact ho.2.2 j hj

theorem outRel_push_byte {outData : List Nat} {outLen b : Nat} {out : List Nat}
    (ho : OutRel outData outLen out) (hlen : outLen + 1 ≤ outData.length) :
    OutRel (writeBytes outData outLen [b % 256]) (outLen + 1) ((b % 256) :: out) := by
  refine ⟨by simp [ho.1], by rw [writeBytes_length]; exact hlen, ?_⟩
  intro j hj
  simp only [List.length_cons, List.reverse_cons] at hj ⊢
  by_cases hlt : j < out.length
  · have hjl : j < outLen := by rw [← ho.1]; exact hlt
    rw [writeBytes_getElem?_lt _ _ _ _ hjl]
    have hget : (out.reverse ++ [b % 256])[j]! = out.reverse[j]! := by
      rw [List.getElem!_eq_getElem?_getD, List.getElem!_eq_getElem?_getD,
        List.getElem?_append_left (by simp [hlt])]
    rw [hget]
    exact ho.2.2 j hlt
  · have heq : j = out.length := by omega
    subst heq
    have hj' : out.length = outLen := ho.1
    rw [hj', writeBytes_at_self _ _ _ (Nat.mod_lt _ (by decide)) (by omega)]
    have hget : (out.reverse ++ [b % 256])[out.length]! = b % 256 := by
      rw [List.getElem!_eq_getElem?_getD, List.getElem?_append_right (by simp),
        List.length_reverse, Nat.sub_self]
      rfl
    rw [← hj', hget]

theorem outRel_push_u64 {outData : List Nat} {outLen v : Nat} {out : List Nat}
    (ho : OutRel outData outLen out) (hlen : outLen + 8 ≤ outData.length) :
    OutRel (writeBytes outData outLen (bytes8 v)) (outLen + 8) ((bytes8 v).reverse ++ out) := by
  refine ⟨by simp [ho.1, bytes8_length]; omega, by rw [writeBytes_length]; exact hlen, ?_⟩
  intro j hj
  simp only [List.length_append, List.length_reverse, bytes8_length, List.reverse_append,
    List.reverse_reverse] at hj ⊢
  by_cases hlt : j < out.length
  · have hjl : j < outLen := by rw [← ho.1]; exact hlt
    rw [writeBytes_getElem?_lt _ _ _ _ hjl]
    have hget : (out.reverse ++ bytes8 v)[j]! = out.reverse[j]! := by
      rw [List.getElem!_eq_getElem?_getD, List.getElem!_eq_getElem?_getD,
        List.getElem?_append_left (by simp [hlt])]
    rw [hget]
    exact ho.2.2 j hlt
  · have hk : j - out.length < 8 := by omega
    have hge : outLen ≤ j := by have := ho.1; omega
    have hins := writeBytes_getElem?_inside outData outLen (j - outLen) (bytes8 v)
      (by rw [bytes8_length, ← ho.1]; exact hk) (by rw [bytes8_length]; exact hlen)
    rw [show outLen + (j - outLen) = j by omega] at hins
    rw [hins]
    have hsub : j - outLen = j - out.length := by rw [ho.1]
    have hblt : (bytes8 v)[j - outLen]'(by rw [bytes8_length, ← ho.1]; exact hk) < 256 :=
      bytes8_lt _ (List.getElem_mem _)
    have hel : some ((bytes8 v)[j - outLen]'(by rw [bytes8_length, ← ho.1]; exact hk)) =
        (bytes8 v)[j - outLen]? :=
      (List.getElem?_eq_getElem (by rw [bytes8_length, ← ho.1]; exact hk)).symm
    have hget : (out.reverse ++ bytes8 v)[j]! = (bytes8 v)[j - out.length]! := by
      rw [List.getElem!_eq_getElem?_getD, List.getElem!_eq_getElem?_getD,
        List.getElem?_append_right (by simp; omega), List.length_reverse]
    rw [Nat.mod_eq_of_lt hblt, hel, hsub, hget, List.getElem!_eq_getElem?_getD]
    rw [List.getElem?_eq_getElem (by rw [bytes8_length]; exact hk), Option.getD_some]

/-! ### VM context, state invariant (`VmRel`), and oracle discipline -/

structure VmCtx where
  stk : Region
  sp : Nat
  ctxReg : Region
  code1 : Region
  code2 : Region
  code3 : Region
  dsReg : Region
  dsCap : Nat
  auxReg : Region
  auxCap : Nat
  rsReg : Region
  rsCap : Nat
  locReg : Region
  locCap : Nat
  memwReg : Region
  mempReg : Region
  tabsReg : Region
  tabsCap : Nat
  kvReg : Region
  kvLen : Nat
  kvCap : Nat
  inpReg : Region
  inpLen : Nat
  outReg : Region
  outCap : Nat

def VmCtx.regions (vc : VmCtx) : List Region :=
  [codeRegion, vc.stk, vc.ctxReg, vc.code1, vc.code2, vc.code3,
   vc.dsReg, vc.auxReg, vc.rsReg, vc.locReg, vc.memwReg, vc.mempReg,
   vc.tabsReg, vc.kvReg, vc.inpReg, vc.outReg]

def VmCtx.wf (vc : VmCtx) (pr : Sporos.Prog) (ls : SporosProvLen.LSt) (s : X86Vm.St) : Prop :=
  CodeMem s.mem ∧
  BytesLt s.mem ∧
  (∀ u ∈ s.mem, 0 < u.data.length) ∧
  RegsOk s ∧
  s.reg 4 = some vc.sp ∧
  s.reg 15 = some ctxBase ∧
  (vc.regions.map Region.base).Nodup ∧
  (∀ r ∈ vc.regions, r ∈ s.mem) ∧
  vc.stk.rw = true ∧ vc.stk.base + 32 ≤ vc.sp ∧ vc.sp + 8 ≤ vc.stk.hi ∧ vc.sp < X86Env.W ∧
  vc.ctxReg.base = ctxBase ∧ vc.ctxReg.rw = true ∧ ctxSize ≤ vc.ctxReg.data.length ∧
    vc.ctxReg.hi ≤ 2 ^ 47 ∧
  FlatCodeOk pr vc.code1 ∧ FlatCodeOk pr vc.code2 ∧ FlatCodeOk pr vc.code3 ∧
  vc.dsReg.rw = true ∧ 4 ≤ vc.dsCap ∧ vc.dsReg.data.length = vc.dsCap * 16 ∧
    (vc.dsCap * 16) % X86Env.page = 0 ∧ vc.dsReg.hi ≤ 2 ^ 47 ∧
  vc.auxReg.rw = true ∧ 4 ≤ vc.auxCap ∧ vc.auxReg.data.length = vc.auxCap * 16 ∧
    (vc.auxCap * 16) % X86Env.page = 0 ∧ vc.auxReg.hi ≤ 2 ^ 47 ∧
  vc.rsReg.rw = true ∧ 0 < vc.rsCap ∧ vc.rsReg.data.length = vc.rsCap * 32 ∧
    (vc.rsCap * 32) % X86Env.page = 0 ∧ vc.rsReg.hi ≤ 2 ^ 47 ∧
  vc.locReg.rw = true ∧ 0 < vc.locCap ∧ vc.locReg.data.length = vc.locCap * 16 ∧
    (vc.locCap * 16) % X86Env.page = 0 ∧ vc.locReg.hi ≤ 2 ^ 47 ∧
  vc.memwReg.rw = true ∧ ls.msize < 2 ^ 44 ∧
    vc.memwReg.data.length = X86Env.pages (max ls.msize 1) ∧ vc.memwReg.hi ≤ 2 ^ 47 ∧
  vc.mempReg.rw = true ∧
    vc.mempReg.data.length = X86Env.pages (8 * max ls.msize 1) ∧ vc.mempReg.hi ≤ 2 ^ 47 ∧
  vc.tabsReg.rw = true ∧ 0 < vc.tabsCap ∧ vc.tabsReg.data.length = vc.tabsCap * 8 ∧
    (vc.tabsCap * 8) % X86Env.page = 0 ∧ vc.tabsReg.hi ≤ 2 ^ 47 ∧
  vc.kvReg.rw = true ∧ 0 < vc.kvCap ∧ vc.kvReg.data.length = vc.kvCap * 32 ∧
    (vc.kvCap * 32) % X86Env.page = 0 ∧ vc.kvReg.hi ≤ 2 ^ 47 ∧
    vc.kvLen ≤ vc.kvCap ∧
  vc.inpReg.hi ≤ 2 ^ 47 ∧
  vc.outReg.rw = true ∧ 8 ≤ vc.outCap ∧ vc.outReg.data.length = vc.outCap ∧
    vc.outCap % X86Env.page = 0 ∧ vc.outReg.hi ≤ 2 ^ 47 ∧
  -- Context slots
  leVal ((vc.ctxReg.data.drop 0).take 8) = vc.dsReg.base ∧
  leVal ((vc.ctxReg.data.drop 8).take 8) = ls.ds.length ∧
  leVal ((vc.ctxReg.data.drop 16).take 8) = vc.dsCap ∧
  leVal ((vc.ctxReg.data.drop 24).take 8) = vc.auxReg.base ∧
  leVal ((vc.ctxReg.data.drop 32).take 8) = ls.aux.length ∧
  leVal ((vc.ctxReg.data.drop 40).take 8) = vc.auxCap ∧
  leVal ((vc.ctxReg.data.drop 48).take 8) = vc.rsReg.base ∧
  leVal ((vc.ctxReg.data.drop 56).take 8) = ls.rs.length ∧
  leVal ((vc.ctxReg.data.drop 64).take 8) = vc.rsCap ∧
  leVal ((vc.ctxReg.data.drop 72).take 8) = vc.locReg.base ∧
  leVal ((vc.ctxReg.data.drop 80).take 8) = rsLocBase ls.rs ∧
  leVal ((vc.ctxReg.data.drop 88).take 8) = vc.locCap ∧
  leVal ((vc.ctxReg.data.drop 96).take 8) = vc.memwReg.base ∧
  leVal ((vc.ctxReg.data.drop 104).take 8) = vc.mempReg.base ∧
  leVal ((vc.ctxReg.data.drop 112).take 8) = ls.msize ∧
  leVal ((vc.ctxReg.data.drop 120).take 8) = ls.hp ∧
  leVal ((vc.ctxReg.data.drop 128).take 8) = vc.tabsReg.base ∧
  leVal ((vc.ctxReg.data.drop 136).take 8) = ls.tabs.length ∧
  leVal ((vc.ctxReg.data.drop 144).take 8) = vc.tabsCap ∧
  leVal ((vc.ctxReg.data.drop 152).take 8) = vc.kvReg.base ∧
  leVal ((vc.ctxReg.data.drop 160).take 8) = vc.kvLen ∧
  leVal ((vc.ctxReg.data.drop 168).take 8) = vc.kvCap ∧
  leVal ((vc.ctxReg.data.drop 176).take 8) = vc.inpReg.base ∧
  leVal ((vc.ctxReg.data.drop 184).take 8) = vc.inpLen ∧
  leVal ((vc.ctxReg.data.drop 192).take 8) = ls.ipos ∧
  leVal ((vc.ctxReg.data.drop 200).take 8) = vc.outReg.base ∧
  leVal ((vc.ctxReg.data.drop 208).take 8) = ls.out.length ∧
  leVal ((vc.ctxReg.data.drop 216).take 8) = vc.outCap ∧
  leVal ((vc.ctxReg.data.drop 224).take 8) = ls.pc ∧
  leVal ((vc.ctxReg.data.drop 232).take 8) = vc.code1.base ∧
  leVal ((vc.ctxReg.data.drop 240).take 8) = vc.code2.base ∧
  leVal ((vc.ctxReg.data.drop 248).take 8) = vc.code3.base ∧
  -- Arena content invariants
  PairListRel vc.dsReg.data ls.ds.length ls.ds ∧ ls.ds.length ≤ vc.dsCap ∧
  PairListRel vc.auxReg.data ls.aux.length ls.aux ∧ ls.aux.length ≤ vc.auxCap ∧
  RsLocRel vc.rsReg.data vc.locReg.data (rsLocBase ls.rs) ls.rs ∧
    ls.rs.length ≤ vc.rsCap ∧ rsLocBase ls.rs ≤ vc.locCap ∧
  MemRel vc.memwReg.data vc.mempReg.data ls.msize ls.mem ∧
  TabsRel vc.tabsReg.data vc.kvReg.data vc.kvLen ls.tabs ∧ ls.tabs.length ≤ vc.tabsCap ∧
  InpRel vc.inpReg.data ls.ipos vc.inpLen ls.inp ∧
  OutRel vc.outReg.data ls.out.length ls.out ∧ ls.out.length ≤ vc.outCap ∧
  ls.WordsLt ∧ FlatProgFit pr

def VmRel (pr : Sporos.Prog) (ls : SporosProvLen.LSt) (s : X86Vm.St) : Prop :=
  s.rip = SporosVmBin.L_vm_loop ∧ ∃ vc : VmCtx, vc.wf pr ls s

/-- The state `s` is at an `mmap` or `mremap` syscall. -/
def AtMemSyscall (s : X86Vm.St) : Prop :=
  vmImage s.rip = some (.ofInt .syscall, 2) ∧ (s.reg 0 = some 9 ∨ s.reg 0 = some 25)

/-- The oracle's choice at `s` makes the `mmap`/`mremap` syscall non-stuck. -/
def OracleLegalAt (s : X86Vm.St) : Prop :=
  AtMemSyscall s →
    ∃ nr a1 a2 a3 a4 a5 a6 rax k' m',
      s.reg 0 = some nr ∧ s.reg 7 = some a1 ∧ s.reg 6 = some a2 ∧
      s.reg 2 = some a3 ∧ s.reg 10 = some a4 ∧ s.reg 8 = some a5 ∧
      s.reg 9 = some a6 ∧
      X86Env.syscall s.k s.mem nr a1 a2 a3 a4 a5 a6 = some (.ret rax k' m')

/-- Every reachable memory syscall from `s0` meets a legal oracle choice. -/
def OracleAdmits (s0 : X86Vm.St) : Prop :=
  ∀ k s, X86Vm.run vmImage k (.ok s0) = .ok s → OracleLegalAt s

/-- The run reaches a memory syscall where the oracle refuses memory (`.nomem`). -/
def HitsNomem (s0 : X86Vm.St) : Prop :=
  ∃ k s, X86Vm.run vmImage k (.ok s0) = .ok s ∧ AtMemSyscall s ∧
    s.k.oracle.head? = some .nomem

theorem oracleAdmits_of_reaches {s0 s1 : X86Vm.St}
    (ho : OracleAdmits s0) (hr : Reaches vmImage s0 s1) : OracleAdmits s1 := by
  intro k s hk
  obtain ⟨k0, hk0⟩ := hr
  apply ho (k0 + k) s
  rw [X86Vm.run_add, hk0, hk]

theorem hitsNomem_of_reaches {s0 s1 : X86Vm.St}
    (hr : Reaches vmImage s0 s1) (hh : HitsNomem s1) : HitsNomem s0 := by
  obtain ⟨k0, hk0⟩ := hr
  obtain ⟨k, s, hk, hat, hhd⟩ := hh
  exact ⟨k0 + k, s, by rw [X86Vm.run_add, hk0, hk], hat, hhd⟩

/-! ### `VmCtx.wf` region extraction & preservation helpers -/

namespace VmCtx

theorem mem_stk {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : vc.stk ∈ s.mem := h.2.2.2.2.2.2.2.1 vc.stk (by simp [regions])
theorem mem_ctx {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : vc.ctxReg ∈ s.mem := h.2.2.2.2.2.2.2.1 vc.ctxReg (by simp [regions])
theorem mem_code1 {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : vc.code1 ∈ s.mem := h.2.2.2.2.2.2.2.1 vc.code1 (by simp [regions])
theorem mem_code2 {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : vc.code2 ∈ s.mem := h.2.2.2.2.2.2.2.1 vc.code2 (by simp [regions])
theorem mem_code3 {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : vc.code3 ∈ s.mem := h.2.2.2.2.2.2.2.1 vc.code3 (by simp [regions])
theorem mem_ds {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : vc.dsReg ∈ s.mem := h.2.2.2.2.2.2.2.1 vc.dsReg (by simp [regions])
theorem mem_aux {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : vc.auxReg ∈ s.mem := h.2.2.2.2.2.2.2.1 vc.auxReg (by simp [regions])
theorem mem_rs {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : vc.rsReg ∈ s.mem := h.2.2.2.2.2.2.2.1 vc.rsReg (by simp [regions])
theorem mem_loc {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : vc.locReg ∈ s.mem := h.2.2.2.2.2.2.2.1 vc.locReg (by simp [regions])
theorem mem_memw {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : vc.memwReg ∈ s.mem := h.2.2.2.2.2.2.2.1 vc.memwReg (by simp [regions])
theorem mem_memp {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : vc.mempReg ∈ s.mem := h.2.2.2.2.2.2.2.1 vc.mempReg (by simp [regions])
theorem mem_tabs {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : vc.tabsReg ∈ s.mem := h.2.2.2.2.2.2.2.1 vc.tabsReg (by simp [regions])
theorem mem_kv {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : vc.kvReg ∈ s.mem := h.2.2.2.2.2.2.2.1 vc.kvReg (by simp [regions])
theorem mem_inp {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : vc.inpReg ∈ s.mem := h.2.2.2.2.2.2.2.1 vc.inpReg (by simp [regions])
theorem mem_out {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : vc.outReg ∈ s.mem := h.2.2.2.2.2.2.2.1 vc.outReg (by simp [regions])

theorem ne_bases {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) {i j : Nat} (hi : i < vc.regions.length) (hj : j < vc.regions.length) (hne : i ≠ j) :
    vc.regions[i].base ≠ vc.regions[j].base := by
  have hnd := h.2.2.2.2.2.2.1
  have go : ∀ (rs : List Region) (a b : Nat) (ha : a < rs.length) (hb : b < rs.length),
      a ≠ b → (rs.map Region.base).Nodup → rs[a].base ≠ rs[b].base := by
    intro rs
    induction rs with
    | nil => intro _ _ ha; simp at ha
    | cons x xs ih =>
      intro a b ha hb hab hndp
      simp only [List.map_cons, List.nodup_cons, List.mem_map] at hndp
      cases a with
      | zero =>
        cases b with
        | zero => exact absurd rfl hab
        | succ b' =>
          intro heq
          apply hndp.1
          exact ⟨xs[b']'(by simp only [List.length_cons] at hb; omega),
            List.getElem_mem (by simp only [List.length_cons] at hb; omega), heq.symm⟩
      | succ a' =>
        cases b with
        | zero =>
          intro heq
          apply hndp.1
          exact ⟨xs[a']'(by simp only [List.length_cons] at ha; omega),
            List.getElem_mem (by simp only [List.length_cons] at ha; omega), heq⟩
        | succ b' =>
          exact ih a' b' (by simp only [List.length_cons] at ha; omega)
            (by simp only [List.length_cons] at hb; omega) (by omega) hndp.2
  exact go vc.regions i j hi hj hne hnd

end VmCtx

/-! ### Output drain (`L_vm_halt`) and error reporting (`L_fail_*` -> `L_vm_fail`) -/

theorem L_vm_halt_loop_spec (s₀ : X86Vm.St) {outReg : Region} {written outLen exitCode : Nat}
    (hrip : s₀.rip = 4200972)
    (hcm : CodeMem s₀.mem)
    (hregs : RegsOk s₀)
    (hout : outReg ∈ s₀.mem) (houtLen : outLen ≤ outReg.data.length)
    (houtHi : outReg.hi ≤ 2 ^ 47)
    (_hbytes : ∀ b ∈ outReg.data, b < 256)
    (hwle : written ≤ outLen)
    (h15 : s₀.reg 15 = some ctxBase)
    (h12 : s₀.reg 12 = some outLen)
    (h13 : s₀.reg 13 = some written)
    (hctx : ∃ ctx ∈ s₀.mem, ctx.base = ctxBase ∧ ctxSize ≤ ctx.data.length ∧
      leVal ((ctx.data.drop 200).take 8) = outReg.base ∧
      leVal ((ctx.data.drop 448).take 8) = exitCode) :
    ∃ s_exit, Reaches vmImage s₀ s_exit ∧
      X86Vm.step vmImage s_exit = .exit (exitCode % 256) ∧
      s_exit.k.out = s₀.k.out ++ ((outReg.data.drop written).take (outLen - written)) := by
  obtain ⟨ctx, hctxm, hctxb, hctxl, h200, h448⟩ := hctx
  have hcb : ctxBase = 0x600000 := rfl
  have hcs : ctxSize = 4096 := rfl
  have hW47 : 2 ^ 47 < X86Env.W := by decide
  have hctxW : ctxBase + ctxSize ≤ X86Env.W := by omega
  have houtW : outReg.hi < X86Env.W := by omega
  suffices gen : ∀ rem s, s.rip = 4200972 → CodeMem s.mem → RegsOk s →
      outReg ∈ s.mem → s.reg 15 = some ctxBase → s.reg 12 = some outLen →
      s.reg 13 = some (outLen - rem) → rem ≤ outLen →
      ctx ∈ s.mem →
      ∃ s_exit, Reaches vmImage s s_exit ∧
        X86Vm.step vmImage s_exit = .exit (exitCode % 256) ∧
        s_exit.k.out = s.k.out ++ ((outReg.data.drop (outLen - rem)).take rem) by
    have hrem := gen (outLen - written) s₀ hrip hcm hregs hout h15 h12
      (by rwa [show outLen - (outLen - written) = written by omega]) (by omega) hctxm
    simpa [show outLen - (outLen - written) = written by omega] using hrem
  intro rem
  induction rem using Nat.strongRecOn with
  | ind rem ih =>
    intro s hsrip hscm hsregs hsout hs15 hs12 hs13 hle hsctx
    have hW247 : 2 ^ 47 < X86Env.W := by decide
    have hwW : outLen - rem < X86Env.W := by simp only [Region.hi] at houtHi; omega
    have hlW : outLen < X86Env.W := by simp only [Region.hi] at houtHi; omega
    have h01 : X86Vm.step vmImage s =
        .ok { s with fl := (X86Int.aluF .sub (outLen - rem) outLen).2, rip := 4200975 } := by
      rw [step_at_chunk 22 hsrip rfl (by decide : (4200972, (.ofInt (.cmp 13 12)), 3) ∈ SporosVmBin.vmTable_22),
        exec_cmp hs13 hs12, hsrip]
    let s1 := { s with fl := (X86Int.aluF .sub (outLen - rem) outLen).2, rip := 4200975 }
    have hr1 : RegsOk s1 := hsregs
    cases rem with
    | zero =>
      have heq : outLen - 0 = outLen := by omega
      have hcond : X86Int.cond s1.fl 3 = some true := by
        rw [cond_aluF_sub_ae (outLen - 0) outLen]
        simp [heq]
      have h02 : X86Vm.step vmImage s1 = .ok { s1 with rip := 4201027 } := by
        rw [step_at_chunk 22 rfl rfl (by decide : (4200975, (.ofInt (.jcc 3 4201027)), 2) ∈ SporosVmBin.vmTable_22),
          exec_jcc hcond]
        simp [s1]
      let s2 := { s1 with rip := 4201027 }
      have hr2 : RegsOk s2 := regsOk_with_rip hr1 _
      have hs2_15 : s2.reg 15 = some ctxBase := hs15
      have hs2_mem : s2.mem = s.mem := rfl
      have hsub448 : ctxBase + 448 - ctx.base = 448 := by
        rw [hctxb]; exact Nat.add_sub_cancel_left ctxBase 448
      have hctx_hi448 : ctxBase + 448 + 8 ≤ ctx.hi := by
        change ctxBase + 448 + 8 ≤ ctx.base + ctx.data.length
        rw [hctxb, Nat.add_assoc]
        exact Nat.add_le_add_left (Nat.le_trans (by decide : 448 + 8 ≤ 4096) (by simpa [hcs] using hctxl)) ctxBase
      have hea448 : ea ctxBase 448 = ctxBase + 448 :=
        ea_add_eq (Nat.lt_of_lt_of_le (Nat.add_lt_add_left (by decide : 448 < ctxSize) ctxBase) hctxW)
      have hrd448 : rd64 s2.mem (ea ctxBase 448) = some exitCode := by
        rw [hea448, hs2_mem]
        exact rd64_of_mem hscm.2 hsctx (by rw [hctxb]; exact Nat.le_add_right ctxBase 448)
          hctx_hi448 (by rw [hsub448]; exact h448)
      have h03 : X86Vm.step vmImage s2 = .ok { s2.set 7 (some exitCode) with rip := 4201034 } := by
        rw [step_at_chunk 22 rfl rfl (by decide : (4201027, (.ofInt (.ld 7 15 448)), 7) ∈ SporosVmBin.vmTable_22),
          exec_ld hs2_15 hrd448]
      let s3 := { s2.set 7 (some exitCode) with rip := 4201034 }
      have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) exitCode
      have hs3_7 : s3.reg 7 = some exitCode := by
        simp only [s3, X86Int.St.reg, X86Int.St.set]
        rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
      have h04 : X86Vm.step vmImage s3 =
          .ok { s3.set 7 (some (Nat.land exitCode 255)) with
                fl := (X86Int.aluF .and exitCode 255).2, rip := 4201041 } := by
        rw [step_at_chunk 22 rfl rfl (by decide : (4201034, (.ofInt (.aluI .and 7 255)), 7) ∈ SporosVmBin.vmTable_22),
          exec_aluI hs3_7, show 255 % X86Env.W = 255 from by decide]
        have hvf : X86Int.aluF .and exitCode 255 =
            (Nat.land exitCode 255, (X86Int.aluF .and exitCode 255).2) :=
          Prod.ext (aluF_and_val _ _) rfl
        rw [hvf]
      let s4 := { s3.set 7 (some (Nat.land exitCode 255)) with
                    fl := (X86Int.aluF .and exitCode 255).2, rip := 4201041 }
      have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) _
      have h05 : X86Vm.step vmImage s4 = .ok { s4.set 0 (some 60) with rip := 4201046 } := by
        rw [step_at_chunk 22 rfl rfl (by decide : (4201041, (.ofInt (.movI 0 60)), 5) ∈ SporosVmBin.vmTable_22),
          exec_movI_lt (by decide)]
      let s5 := { s4.set 0 (some 60) with rip := 4201046 }
      have hr5 : RegsOk s5 := regsOk_set hr4 (by decide) 60
      obtain ⟨⟨v0, hv0⟩, ⟨v7, hv7⟩, ⟨v6, hv6⟩, ⟨v2, hv2⟩, ⟨v10, hv10⟩, ⟨v8, hv8⟩, ⟨v9, hv9⟩⟩ :=
        regsOk_syscall_regs hr5
      have hv0' : v0 = 60 := by
        have hs : s5.reg 0 = some 60 := by
          simp only [s5, X86Int.St.reg, X86Int.St.set]
          rw [getElem?_set_getD_self (by rw [hr4.1]; decide)]
        rw [hs] at hv0; cases hv0; rfl
      have hv7' : v7 = Nat.land exitCode 255 := by
        have hs : s5.reg 7 = some (Nat.land exitCode 255) := by
          simp only [s5, s4, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 7)]
          rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
        rw [hs] at hv7; cases hv7; rfl
      subst hv0' hv7'
      have hland : Nat.land exitCode 255 % 256 = exitCode % 256 := by
        change (exitCode &&& (2 ^ 8 - 1)) % 256 = exitCode % 256
        rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_mod]
      have hsys : X86Env.syscall s5.k s5.mem 60 (Nat.land exitCode 255) v6 v2 v10 v8 v9 =
          some (.exit (exitCode % 256)) := by
        simp [X86Env.syscall]
        simpa using hland
      have h06 : X86Vm.step vmImage s5 = .exit (exitCode % 256) := by
        rw [step_at_chunk 22 rfl rfl (by decide : (4201046, (.ofInt .syscall), 2) ∈ SporosVmBin.vmTable_22),
          exec_syscall_exit hv0 hv7 hv6 hv2 hv10 hv8 hv9 hsys]
      have hkout5 : s5.k.out = s.k.out := rfl
      refine ⟨s5, Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03
        (Reaches.step_trans h04 (Reaches.step h05)))), h06, by simp [hkout5]⟩
    | succ rem' =>
      have hlt_rem : outLen - (rem' + 1) < outLen := by omega
      have hcond : X86Int.cond s1.fl 3 = some false := by
        rw [cond_aluF_sub_ae (outLen - (rem' + 1)) outLen]
        simp [hlt_rem]
      have h02 : X86Vm.step vmImage s1 = .ok { s1 with rip := 4200977 } := by
        rw [step_at_chunk 22 rfl rfl (by decide : (4200975, (.ofInt (.jcc 3 4201027)), 2) ∈ SporosVmBin.vmTable_22),
          exec_jcc hcond]
        simp [s1]
      let s2 := { s1 with rip := 4200977 }
      have hr2 : RegsOk s2 := regsOk_with_rip hr1 _
      have hs2_15 : s2.reg 15 = some ctxBase := hs15
      have hea200 : ea ctxBase 200 = ctxBase + 200 := ea_add_eq (by omega)
      have hrd200 : rd64 s2.mem (ea ctxBase 200) = some outReg.base := by
        rw [hea200]
        exact rd64_of_mem hscm.2 hsctx (by rw [hctxb]; omega)
          (by simp only [Region.hi, hctxb]; omega) (by simpa [hctxb] using h200)
      have h03 : X86Vm.step vmImage s2 = .ok { s2.set 6 (some outReg.base) with rip := 4200984 } := by
        rw [step_at_chunk 22 rfl rfl (by decide : (4200977, (.ofInt (.ld 6 15 200)), 7) ∈ SporosVmBin.vmTable_22),
          exec_ld hs2_15 hrd200]
      let s3 := { s2.set 6 (some outReg.base) with rip := 4200984 }
      have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) outReg.base
      have hs3_6 : s3.reg 6 = some outReg.base := by
        simp only [s3, X86Int.St.reg, X86Int.St.set]
        rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
      have hs3_13 : s3.reg 13 = some (outLen - (rem' + 1)) := by
        simp only [s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 6 ≠ 13)]
        exact hs13
      have hptrW : outReg.base + (outLen - (rem' + 1)) < X86Env.W := by
        simp only [Region.hi] at houtHi; omega
      have hs3_rip : s3.rip = 4200984 := rfl
      have h04 : X86Vm.step vmImage s3 =
          .ok { s3.set 6 (some (outReg.base + (outLen - (rem' + 1)))) with
                fl := (X86Int.aluF .add outReg.base (outLen - (rem' + 1))).2, rip := 4200987 } := by
        rw [step_at_chunk 22 rfl rfl (by decide : (4200984, (.ofInt (.alu .add 6 13)), 3) ∈ SporosVmBin.vmTable_22),
          exec_alu hs3_6 hs3_13, hs3_rip]
        simp [aluF_add_val, Nat.mod_eq_of_lt hptrW]
      let s4 := { s3.set 6 (some (outReg.base + (outLen - (rem' + 1)))) with
                    fl := (X86Int.aluF .add outReg.base (outLen - (rem' + 1))).2, rip := 4200987 }
      have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) _
      have hs4_12 : s4.reg 12 = some outLen := by
        simp only [s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 6 ≠ 12)]
        exact hs12
      have h05 : X86Vm.step vmImage s4 = .ok { s4.set 2 (some outLen) with rip := 4200990 } := by
        rw [step_at_chunk 22 rfl rfl (by decide : (4200987, (.ofInt (.mov 2 12)), 3) ∈ SporosVmBin.vmTable_22),
          exec_mov hs4_12]
      let s5 := { s4.set 2 (some outLen) with rip := 4200990 }
      have hr5 : RegsOk s5 := regsOk_set hr4 (by decide) outLen
      have hs5_2 : s5.reg 2 = some outLen := by
        simp only [s5, X86Int.St.reg, X86Int.St.set]
        rw [getElem?_set_getD_self (by rw [hr4.1]; decide)]
      have hs5_13 : s5.reg 13 = some (outLen - (rem' + 1)) := by
        simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
          List.getElem?_set_ne (by decide : 2 ≠ 13), List.getElem?_set_ne (by decide : 6 ≠ 13)]
        exact hs13
      have hsub_val : (X86Int.aluF .sub outLen (outLen - (rem' + 1))).1 = rem' + 1 := by
        rw [aluF_sub_val, show outLen + X86Env.W - (outLen - (rem' + 1)) = (rem' + 1) + X86Env.W by omega]
        rw [Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
      have hs5_rip : s5.rip = 4200990 := rfl
      have h06 : X86Vm.step vmImage s5 =
          .ok { s5.set 2 (some (rem' + 1)) with
                fl := (X86Int.aluF .sub outLen (outLen - (rem' + 1))).2, rip := 4200993 } := by
        rw [step_at_chunk 22 rfl rfl (by decide : (4200990, (.ofInt (.alu .sub 2 13)), 3) ∈ SporosVmBin.vmTable_22),
          exec_alu hs5_2 hs5_13, hs5_rip]
        simp [hsub_val]
      let s6 := { s5.set 2 (some (rem' + 1)) with
                    fl := (X86Int.aluF .sub outLen (outLen - (rem' + 1))).2, rip := 4200993 }
      have hr6 : RegsOk s6 := regsOk_set hr5 (by decide) _
      have h07 : X86Vm.step vmImage s6 = .ok { s6.set 7 (some 1) with rip := 4200998 } := by
        rw [step_at_chunk 22 rfl rfl (by decide : (4200993, (.ofInt (.movI 7 1)), 5) ∈ SporosVmBin.vmTable_22),
          exec_movI_lt (by decide)]
      let s7 := { s6.set 7 (some 1) with rip := 4200998 }
      have hr7 : RegsOk s7 := regsOk_set hr6 (by decide) 1
      have h08 : X86Vm.step vmImage s7 = .ok { s7.set 0 (some 1) with rip := 4201003 } := by
        rw [step_at_chunk 22 rfl rfl (by decide : (4200998, (.ofInt (.movI 0 1)), 5) ∈ SporosVmBin.vmTable_22),
          exec_movI_lt (by decide)]
      let s8 := { s7.set 0 (some 1) with rip := 4201003 }
      have hr8 : RegsOk s8 := regsOk_set hr7 (by decide) 1
      obtain ⟨⟨v0, hv0⟩, ⟨v7, hv7⟩, ⟨v6, hv6⟩, ⟨v2, hv2⟩, ⟨v10, hv10⟩, ⟨v8, hv8⟩, ⟨v9, hv9⟩⟩ :=
        regsOk_syscall_regs hr8
      have hv0' : v0 = 1 := by
        have hs : s8.reg 0 = some 1 := by
          simp only [s8, X86Int.St.reg, X86Int.St.set]
          rw [getElem?_set_getD_self (by rw [hr7.1]; decide)]
        rw [hs] at hv0; cases hv0; rfl
      have hv7' : v7 = 1 := by
        have hs : s8.reg 7 = some 1 := by
          simp only [s8, s7, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 7)]
          rw [getElem?_set_getD_self (by rw [hr6.1]; decide)]
        rw [hs] at hv7; cases hv7; rfl
      have hv6' : v6 = outReg.base + (outLen - (rem' + 1)) := by
        have hs : s8.reg 6 = some (outReg.base + (outLen - (rem' + 1))) := by
          simp only [s8, s7, s6, s5, s4, X86Int.St.reg, X86Int.St.set,
            List.getElem?_set_ne (by decide : 0 ≠ 6), List.getElem?_set_ne (by decide : 7 ≠ 6),
            List.getElem?_set_ne (by decide : 2 ≠ 6)]
          rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
        rw [hs] at hv6; cases hv6; rfl
      have hv2' : v2 = rem' + 1 := by
        have hs : s8.reg 2 = some (rem' + 1) := by
          simp only [s8, s7, s6, X86Int.St.reg, X86Int.St.set,
            List.getElem?_set_ne (by decide : 0 ≠ 2), List.getElem?_set_ne (by decide : 7 ≠ 2)]
          rw [getElem?_set_getD_self (by rw [hr5.1]; decide)]
        rw [hs] at hv2; cases hv2; rfl
      subst hv0' hv7' hv6' hv2'
      have hcnt_pos_le :
          0 < (s8.k.count (rem' + 1) (rem' + 1)).1 ∧ (s8.k.count (rem' + 1) (rem' + 1)).1 ≤ rem' + 1 := by
        unfold X86Env.Kern.count
        simp only [Nat.min_self]
        rcases hnx : s8.k.next with ⟨oc, k'⟩
        cases oc with
        | none => simp
        | some c =>
          cases c with
          | count m =>
            by_cases hbm : 0 < m ∧ m ≤ rem' + 1
            · simp [hbm]
            · simp [hbm]
          | «at» _ => simp
          | nomem => simp
      let n := (s8.k.count (rem' + 1) (rem' + 1)).1
      let k1 := (s8.k.count (rem' + 1) (rem' + 1)).2
      have hn_pos : 0 < n := hcnt_pos_le.1
      have hn_le : n ≤ rem' + 1 := hcnt_pos_le.2
      have hlo_out : outReg.base ≤ outReg.base + (outLen - (rem' + 1)) := by omega
      have hhi_out : outReg.base + (outLen - (rem' + 1)) + n ≤ outReg.hi := by
        simp only [Region.hi]; omega
      have hrdN : s8.mem.rdN (outReg.base + (outLen - (rem' + 1))) n =
          some ((outReg.data.drop (outLen - (rem' + 1))).take n) := by
        have hm := mem_rdN_of_mem s8.mem outReg (outReg.base + (outLen - (rem' + 1))) n
          hscm.2 hsout hlo_out hhi_out
        simpa [show outReg.base + (outLen - (rem' + 1)) - outReg.base = outLen - (rem' + 1) by omega] using hm
      have hsys : X86Env.syscall s8.k s8.mem 1 1 (outReg.base + (outLen - (rem' + 1))) (rem' + 1)
          v10 v8 v9 =
          some (.ret n { k1 with out := s8.k.out ++ ((outReg.data.drop (outLen - (rem' + 1))).take n) } s8.mem) := by
        simp only [X86Env.syscall, ↓reduceIte, n, k1, bind, Option.bind, pure]
        rw [hrdN]
      have h09 : X86Vm.step vmImage s8 =
          .ok { ((s8.set 0 (some n)).set 1 (some 4201005)).set 11 none with
                k := { k1 with out := s8.k.out ++ ((outReg.data.drop (outLen - (rem' + 1))).take n) },
                mem := s8.mem, rip := 4201005 } := by
        rw [step_at_chunk 22 rfl rfl (by decide : (4201003, (.ofInt .syscall), 2) ∈ SporosVmBin.vmTable_22),
          exec_syscall_ret hv0 hv7 hv6 hv2 hv10 hv8 hv9 hsys]
      let s9 := { ((s8.set 0 (some n)).set 1 (some 4201005)).set 11 none with
                    k := { k1 with out := s8.k.out ++ ((outReg.data.drop (outLen - (rem' + 1))).take n) },
                    mem := s8.mem, rip := 4201005 }
      have hr9 : RegsOk s9 := regsOk_set_none_11 (regsOk_set (regsOk_set hr8 (by decide) n) (by decide) 4201005)
      have hs9_0 : s9.reg 0 = some n := by
        simp only [s9, X86Int.St.reg, X86Int.St.set,
          List.getElem?_set_ne (by decide : 11 ≠ 0), List.getElem?_set_ne (by decide : 1 ≠ 0)]
        rw [getElem?_set_getD_self (by rw [hr8.1]; decide)]
      have h010 : X86Vm.step vmImage s9 =
          .ok { s9 with fl := (X86Int.aluF .and n n).2, rip := 4201008 } := by
        rw [step_at_chunk 22 rfl rfl (by decide : (4201005, (.ofInt (.test 0 0)), 3) ∈ SporosVmBin.vmTable_22),
          exec_test hs9_0 hs9_0]
      let s10 := { s9 with fl := (X86Int.aluF .and n n).2, rip := 4201008 }
      have hr10 : RegsOk s10 := hr9
      have hn63 : n < 2 ^ 63 := by
        have : outLen ≤ 2 ^ 47 := by simp only [Region.hi] at houtHi; omega
        have : 2 ^ 47 < 2 ^ 63 := by decide
        omega
      have hcond_le : X86Int.cond s10.fl 14 = some false := cond_aluF_and_self_le hn_pos hn63
      have h011 : X86Vm.step vmImage s10 = .ok { s10 with rip := 4201010 } := by
        rw [step_at_chunk 22 rfl rfl (by decide : (4201008, (.ofInt (.jcc 14 4201015)), 2) ∈ SporosVmBin.vmTable_22),
          exec_jcc hcond_le]
        simp [s10]
      let s11 := { s10 with rip := 4201010 }
      have hr11 : RegsOk s11 := regsOk_with_rip hr10 _
      have hs11_13 : s11.reg 13 = some (outLen - (rem' + 1)) := by
        simp only [s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
          List.getElem?_set_ne (by decide : 11 ≠ 13), List.getElem?_set_ne (by decide : 1 ≠ 13),
          List.getElem?_set_ne (by decide : 0 ≠ 13), List.getElem?_set_ne (by decide : 7 ≠ 13),
          List.getElem?_set_ne (by decide : 2 ≠ 13), List.getElem?_set_ne (by decide : 6 ≠ 13)]
        exact hs13
      have hs11_0 : s11.reg 0 = some n := hs9_0
      have hsum_rem : outLen - (rem' + 1) + n = outLen - (rem' + 1 - n) := by omega
      have hsumW : outLen - (rem' + 1 - n) < X86Env.W := by
        have : outLen ≤ 2 ^ 47 := by simp only [Region.hi] at houtHi; omega
        omega
      have hs11_rip : s11.rip = 4201010 := rfl
      have h012 : X86Vm.step vmImage s11 =
          .ok { s11.set 13 (some (outLen - (rem' + 1 - n))) with
                fl := (X86Int.aluF .add (outLen - (rem' + 1)) n).2, rip := 4201013 } := by
        rw [step_at_chunk 22 rfl rfl (by decide : (4201010, (.ofInt (.alu .add 13 0)), 3) ∈ SporosVmBin.vmTable_22),
          exec_alu hs11_13 hs11_0, hs11_rip]
        simp [aluF_add_val, hsum_rem, Nat.mod_eq_of_lt hsumW]
      let s12 := { s11.set 13 (some (outLen - (rem' + 1 - n))) with
                     fl := (X86Int.aluF .add (outLen - (rem' + 1)) n).2, rip := 4201013 }
      have hr12 : RegsOk s12 := regsOk_set hr11 (by decide) _
      have h013 : X86Vm.step vmImage s12 = .ok { s12 with rip := 4200972 } := by
        rw [step_at_chunk 22 rfl rfl (by decide : (4201013, (.ofInt (.jmp 4200972)), 2) ∈ SporosVmBin.vmTable_22),
          exec_jmp]
      let s13 := { s12 with rip := 4200972 }
      have hr13 : RegsOk s13 := regsOk_with_rip hr12 _
      have hs13_15 : s13.reg 15 = some ctxBase := by
        simp only [s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
          List.getElem?_set_ne (by decide : 13 ≠ 15), List.getElem?_set_ne (by decide : 11 ≠ 15),
          List.getElem?_set_ne (by decide : 1 ≠ 15), List.getElem?_set_ne (by decide : 0 ≠ 15),
          List.getElem?_set_ne (by decide : 7 ≠ 15), List.getElem?_set_ne (by decide : 2 ≠ 15),
          List.getElem?_set_ne (by decide : 6 ≠ 15)]
        exact hs15
      have hs13_12 : s13.reg 12 = some outLen := by
        simp only [s13, s12, s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
          List.getElem?_set_ne (by decide : 13 ≠ 12), List.getElem?_set_ne (by decide : 11 ≠ 12),
          List.getElem?_set_ne (by decide : 1 ≠ 12), List.getElem?_set_ne (by decide : 0 ≠ 12),
          List.getElem?_set_ne (by decide : 7 ≠ 12), List.getElem?_set_ne (by decide : 2 ≠ 12),
          List.getElem?_set_ne (by decide : 6 ≠ 12)]
        exact hs12
      have hs13_13 : s13.reg 13 = some (outLen - (rem' + 1 - n)) := by
        simp only [s13, s12, X86Int.St.reg, X86Int.St.set]
        rw [getElem?_set_getD_self (by rw [hr11.1]; decide)]
      obtain ⟨s_exit, hreach, hexit, hkout⟩ := ih (rem' + 1 - n) (by omega) s13 rfl hscm hr13 hsout
        hs13_15 hs13_12 hs13_13 (by omega) hsctx
      have hreach_tot : Reaches vmImage s s_exit :=
        Reaches.trans
          (Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03
            (Reaches.step_trans h04 (Reaches.step_trans h05 (Reaches.step_trans h06
              (Reaches.step_trans h07 (Reaches.step_trans h08 (Reaches.step_trans h09
                (Reaches.step_trans h010 (Reaches.step_trans h011 (Reaches.step_trans h012
                  (Reaches.step h013))))))))))))) hreach
      refine ⟨s_exit, hreach_tot, hexit, ?_⟩
      rw [hkout]
      have hk13_9 : s13.k.out = s9.k.out := rfl
      have hk9_8 : s9.k.out = s8.k.out ++ ((outReg.data.drop (outLen - (rem' + 1))).take n) := rfl
      have hk8_0 : s8.k.out = s.k.out := rfl
      have hk13 : s13.k.out = s.k.out ++ ((outReg.data.drop (outLen - (rem' + 1))).take n) := by
        rw [hk13_9, hk9_8, hk8_0]
      rw [hk13, List.append_assoc, hsum_rem.symm]
      have hsplit : (outReg.data.drop (outLen - (rem' + 1))).take (rem' + 1) =
          (outReg.data.drop (outLen - (rem' + 1))).take n ++
          (outReg.data.drop (outLen - (rem' + 1) + n)).take (rem' + 1 - n) := by
        rw [show rem' + 1 = n + (rem' + 1 - n) by omega, List.take_add, List.drop_drop,
          Nat.add_sub_cancel_left]
      rw [hsplit]

theorem L_vm_halt_spec {vc : VmCtx} {_pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    {exitCode : Nat}
    (hrip : s.rip = SporosVmBin.L_vm_halt)
    (hcm : CodeMem s.mem) (hbl : BytesLt s.mem) (hregs : RegsOk s)
    (h15 : s.reg 15 = some ctxBase)
    (hctx : vc.ctxReg ∈ s.mem) (hctxb : vc.ctxReg.base = ctxBase) (hctxl : ctxSize ≤ vc.ctxReg.data.length)
    (hout : vc.outReg ∈ s.mem) (houtHi : vc.outReg.hi ≤ 2 ^ 47)
    (hc200 : leVal ((vc.ctxReg.data.drop 200).take 8) = vc.outReg.base)
    (hc208 : leVal ((vc.ctxReg.data.drop 208).take 8) = ls.out.length)
    (hc448 : leVal ((vc.ctxReg.data.drop 448).take 8) = exitCode)
    (horel : OutRel vc.outReg.data ls.out.length ls.out) :
    ∃ s_exit, Reaches vmImage s s_exit ∧
      X86Vm.step vmImage s_exit = .exit (exitCode % 256) ∧
      s_exit.k.out = s.k.out ++ ls.out.reverse := by
  have hea208 : ea ctxBase 208 = ctxBase + 208 := ea_add_eq (by
    have hcb : ctxBase = 0x600000 := rfl
    have hcs : ctxSize = 4096 := rfl
    have h47 : 2 ^ 47 < X86Env.W := by decide
    omega)
  have hrd208 : rd64 s.mem (ea ctxBase 208) = some ls.out.length := by
    rw [hea208]
    have hcb : ctxBase = 0x600000 := rfl
    have hcs : ctxSize = 4096 := rfl
    exact rd64_of_mem hcm.2 hctx (by rw [hctxb]; omega)
      (by simp only [Region.hi, hctxb]; omega) (by simpa [hctxb] using hc208)
  have h01 : X86Vm.step vmImage s = .ok { s.set 12 (some ls.out.length) with rip := 4200969 } := by
    rw [step_at_chunk 22 hrip rfl (by decide : (SporosVmBin.L_vm_halt, (.ofInt (.ld 12 15 208)), 7) ∈ SporosVmBin.vmTable_22),
      exec_ld h15 hrd208, hrip]
    rfl
  let s1 := { s.set 12 (some ls.out.length) with rip := 4200969 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) ls.out.length
  obtain ⟨v13, hv13⟩ := regs_get_of_ok hr1 13 (by decide) (by decide)
  have h02 : X86Vm.step vmImage s1 =
      .ok { s1.set 13 (some 0) with fl := (X86Int.aluF .xor v13 v13).2, rip := 4200972 } := by
    rw [step_at_chunk 22 rfl rfl (by decide : (4200969, (.ofInt (.alu .xor 13 13)), 3) ∈ SporosVmBin.vmTable_22),
      exec_alu_xor_self hv13]
  let s2 := { s1.set 13 (some 0) with fl := (X86Int.aluF .xor v13 v13).2, rip := 4200972 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) 0
  have hs2_15 : s2.reg 15 = some ctxBase := by
    simp only [s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 13 ≠ 15), List.getElem?_set_ne (by decide : 12 ≠ 15)]
    exact h15
  have hs2_12 : s2.reg 12 = some ls.out.length := by
    simp only [s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 13 ≠ 12)]
    rw [getElem?_set_getD_self (by rw [hregs.1]; decide)]
  have hs2_13 : s2.reg 13 = some 0 := by
    simp only [s2, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  obtain ⟨s_exit, hreach, hexit, hkout⟩ := L_vm_halt_loop_spec s2 rfl hcm hr2 hout
    (by have := horel.1; have := horel.2.1; omega) houtHi (hbl vc.outReg hout) (Nat.zero_le _)
    hs2_15 hs2_12 hs2_13 ⟨vc.ctxReg, hctx, hctxb, hctxl, hc200, hc448⟩
  refine ⟨s_exit, Reaches.trans (Reaches.step_trans h01 (Reaches.step h02)) hreach, hexit, ?_⟩
  rw [hkout]
  have hs2k : s2.k.out = s.k.out := rfl
  rw [hs2k, List.drop_zero, Nat.sub_zero]
  have htake : vc.outReg.data.take ls.out.length = ls.out.reverse := by
    apply List.ext_getElem?
    intro j
    by_cases hj : j < ls.out.length
    · rw [List.getElem?_take_of_lt hj, horel.2.2 j hj,
        List.getElem!_eq_getElem?_getD,
        List.getElem?_eq_getElem (by simpa only [List.length_reverse] using hj),
        Option.getD_some]
    · have hj1 : (vc.outReg.data.take ls.out.length).length ≤ j := by
        have := horel.1; have := horel.2.1
        rw [List.length_take]; omega
      have hj2 : ls.out.reverse.length ≤ j := by simp; omega
      rw [List.getElem?_eq_none hj1, List.getElem?_eq_none hj2]
  rw [htake]

/-! ### Universal `BytesLt` and non-empty region preservation over `X86Vm.step` -/

theorem bytesLt_replicate_zero (n : Nat) : ∀ b ∈ List.replicate n 0, b < 256 := by
  intro b hb
  rw [List.mem_replicate] at hb
  omega

theorem bytesLt_wr {m m' : Mem} {a v : Nat}
    (hb : BytesLt m) (hwr : m.wr a v = some m') : BytesLt m' := by
  simp only [Mem.wr, bind, Option.bind, pure] at hwr
  cases hf : Mem.find m a 0 with
  | none => simp [hf] at hwr
  | some p =>
    rcases p with ⟨i, r⟩
    simp only [hf] at hwr
    split at hwr
    · simp only [Option.some.injEq] at hwr
      subst hwr
      have hget := (mem_find_idx_lt m a 0 i r hf).2.2
      have hr : r ∈ m := List.mem_of_getElem? hget
      exact bytesLt_set_wr hb hget (list_set_lt (hb r hr) _ _ (Nat.mod_lt _ (by decide)))
    · simp at hwr

theorem bytesLt_wrN {m m' : Mem} {a : Nat} {bs : List Nat}
    (hb : BytesLt m) (hwr : m.wrN a bs = some m') : BytesLt m' := by
  induction bs generalizing m a with
  | nil => simp [Mem.wrN] at hwr; subst hwr; exact hb
  | cons b rest ih =>
    simp only [Mem.wrN, bind, Option.bind] at hwr
    cases hw : m.wr a b with
    | none => simp [hw] at hwr
    | some m1 => rw [hw] at hwr; exact ih (bytesLt_wr hb hw) hwr

theorem bytesLt_wr64 {m m' : Mem} {a v : Nat}
    (hb : BytesLt m) (hwr : wr64 m a v = some m') : BytesLt m' :=
  bytesLt_wrN hb hwr

private theorem bytesLt_wrMem_raw {m m' : Mem} {w : MemW} {a v : Nat}
    (hb : BytesLt m) (hwr : wrMem w m a v = some m') : BytesLt m' := by
  cases w <;> simp [wrMem, wr16, wr32, wr64] at hwr
  · exact bytesLt_wr hb hwr
  · exact bytesLt_wrN hb hwr
  · exact bytesLt_wrN hb hwr
  · exact bytesLt_wrN hb hwr

theorem len_pos_wr {m m' : Mem} {a v : Nat}
    (hp : ∀ u ∈ m, 0 < u.data.length) (hwr : m.wr a v = some m') :
    ∀ u ∈ m', 0 < u.data.length := by
  simp only [Mem.wr, bind, Option.bind, pure] at hwr
  cases hf : Mem.find m a 0 with
  | none => simp [hf] at hwr
  | some p =>
    rcases p with ⟨i, r⟩
    simp only [hf] at hwr
    split at hwr
    · simp only [Option.some.injEq] at hwr
      subst hwr
      have hr : r ∈ m := List.mem_of_getElem? (mem_find_idx_lt m a 0 i r hf).2.2
      intro u hu
      rcases mem_set_mem_cases hu with hu' | rfl
      · exact hp u hu'
      · simpa using hp r hr
    · simp at hwr

theorem len_pos_wrN {m m' : Mem} {a : Nat} {bs : List Nat}
    (hp : ∀ u ∈ m, 0 < u.data.length) (hwr : m.wrN a bs = some m') :
    ∀ u ∈ m', 0 < u.data.length := by
  induction bs generalizing m a with
  | nil => simp [Mem.wrN] at hwr; subst hwr; exact hp
  | cons b rest ih =>
    simp only [Mem.wrN, bind, Option.bind] at hwr
    cases hw : m.wr a b with
    | none => simp [hw] at hwr
    | some m1 => rw [hw] at hwr; exact ih (len_pos_wr hp hw) hwr

theorem len_pos_wrMem {m m' : Mem} {w : MemW} {a v : Nat}
    (hp : ∀ u ∈ m, 0 < u.data.length) (hwr : wrMem w m a v = some m') :
    ∀ u ∈ m', 0 < u.data.length := by
  cases w <;> simp [wrMem, wr16, wr32, wr64] at hwr
  · exact len_pos_wr hp hwr
  · exact len_pos_wrN hp hwr
  · exact len_pos_wrN hp hwr
  · exact len_pos_wrN hp hwr

theorem pages_pos {n : Nat} (hn : 0 < n) : 0 < X86Env.pages n := by
  have hp : X86Env.page = 4096 := rfl
  simp only [X86Env.pages, hp]
  omega

theorem bytesLt_syscall {k k' : Kern} {m m' : Mem} {nr a1 a2 a3 a4 a5 a6 rax : Nat}
    (hb : BytesLt m) (hp : ∀ u ∈ m, 0 < u.data.length)
    (hs : X86Env.syscall k m nr a1 a2 a3 a4 a5 a6 = some (.ret rax k' m')) :
    BytesLt m' ∧ ∀ u ∈ m', 0 < u.data.length := by
  unfold X86Env.syscall at hs
  split at hs
  · split at hs
    · simp only [bind, Option.bind, pure] at hs
      cases hw : m.wrN a2 _ with
      | none => rw [hw] at hs; simp at hs
      | some m1 =>
        rw [hw] at hs; simp only [Option.some.injEq, SysRes.ret.injEq] at hs
        obtain ⟨_, _, rfl⟩ := hs
        exact ⟨bytesLt_wrN hb hw, len_pos_wrN hp hw⟩
    · split at hs
      · cases hf : k.fds[a1 - 3]? with
        | none => rw [hf] at hs; simp at hs; obtain ⟨_, _, rfl⟩ := hs; exact ⟨hb, hp⟩
        | some v =>
          rw [hf] at hs
          cases v with
          | none => simp at hs; obtain ⟨_, _, rfl⟩ := hs; exact ⟨hb, hp⟩
          | some fd =>
            simp only [bind, Option.bind, pure] at hs
            cases hw : m.wrN a2 _ with
            | none => rw [hw] at hs; simp at hs
            | some m1 =>
              rw [hw] at hs; simp only [Option.some.injEq, SysRes.ret.injEq] at hs
              obtain ⟨_, _, rfl⟩ := hs
              exact ⟨bytesLt_wrN hb hw, len_pos_wrN hp hw⟩
      · simp at hs; obtain ⟨_, _, rfl⟩ := hs; exact ⟨hb, hp⟩
  · split at hs
    · simp only [bind, Option.bind, pure] at hs
      cases hr : m.rdN a2 _ with
      | none => rw [hr] at hs; simp at hs
      | some bs =>
        rw [hr] at hs; simp only [Option.some.injEq, SysRes.ret.injEq] at hs
        obtain ⟨_, _, rfl⟩ := hs; exact ⟨hb, hp⟩
    · simp at hs; obtain ⟨_, _, rfl⟩ := hs; exact ⟨hb, hp⟩
  · split at hs
    · simp at hs
    · cases hc : m.cstr a1 4096 with
      | none => rw [hc] at hs; simp at hs
      | some name =>
        simp only [hc, bind, Option.bind, pure] at hs
        split at hs <;> simp at hs <;> obtain ⟨_, _, rfl⟩ := hs <;> exact ⟨hb, hp⟩
  · split at hs <;> simp at hs <;> obtain ⟨_, _, rfl⟩ := hs <;> exact ⟨hb, hp⟩
  · split at hs
    · simp at hs
    · split at hs
      · simp at hs; obtain ⟨_, _, rfl⟩ := hs; exact ⟨hb, hp⟩
      · rename_i ha2
        have ha2_pos : 0 < a2 := Nat.pos_of_ne_zero ha2
        simp only [bind, Option.bind, pure] at hs
        cases hpl : X86Env.place m (k.next).1 (X86Env.pages a2) with
        | none => rw [hpl] at hs; simp at hs
        | some o =>
          rw [hpl] at hs
          cases o with
          | none => simp at hs; obtain ⟨_, _, rfl⟩ := hs; exact ⟨hb, hp⟩
          | some a =>
            simp only [Option.some.injEq, SysRes.ret.injEq] at hs
            obtain ⟨_, _, rfl⟩ := hs
            refine ⟨?_, ?_⟩
            · intro u hu
              rw [List.mem_append] at hu
              rcases hu with hu | hu
              · exact hb u hu
              · simp only [List.mem_singleton] at hu; subst hu
                exact bytesLt_replicate_zero _
            · intro u hu
              rw [List.mem_append] at hu
              rcases hu with hu | hu
              · exact hp u hu
              · simp only [List.mem_singleton] at hu; subst hu
                simp [List.length_replicate, pages_pos ha2_pos]
  · cases hf : Mem.find m a1 0 with
    | none => rw [hf] at hs; simp at hs
    | some p =>
      rw [hf] at hs; dsimp only at hs
      split at hs
      · dsimp only [pure] at hs
        cases hs
        exact ⟨fun u hu => hb u (List.mem_of_mem_eraseIdx hu),
          fun u hu => hp u (List.mem_of_mem_eraseIdx hu)⟩
      · simp at hs
  · split at hs
    · simp at hs
    · cases hf : Mem.find m a1 0 with
      | none => rw [hf] at hs; simp at hs
      | some p =>
        rcases p with ⟨i, r⟩
        rw [hf] at hs; dsimp only at hs
        have hr_mem : r ∈ m := List.mem_of_getElem? (mem_find_idx_lt m a1 0 i r hf).2.2
        split at hs
        · simp at hs
        · split at hs
          · simp at hs
          · simp only [bind, Option.bind, pure] at hs
            cases hpl : X86Env.place (m.eraseIdx i) (k.next).1 (X86Env.pages a3) with
            | none => rw [hpl] at hs; simp at hs
            | some o =>
              rw [hpl] at hs
              cases o with
              | none => simp at hs; obtain ⟨_, _, rfl⟩ := hs; exact ⟨hb, hp⟩
              | some a =>
                simp only [Option.some.injEq, SysRes.ret.injEq] at hs
                obtain ⟨_, _, rfl⟩ := hs
                refine ⟨?_, ?_⟩
                · intro u hu
                  rw [List.mem_append] at hu
                  rcases hu with hu | hu
                  · exact hb u (List.mem_of_mem_eraseIdx hu)
                  · simp only [List.mem_singleton] at hu; subst hu
                    intro b hmem
                    rw [List.mem_append] at hmem
                    rcases hmem with hmem | hmem
                    · exact hb r hr_mem b hmem
                    · exact bytesLt_replicate_zero _ b hmem
                · intro u hu
                  rw [List.mem_append] at hu
                  rcases hu with hu | hu
                  · exact hp u (List.mem_of_mem_eraseIdx hu)
                  · simp only [List.mem_singleton] at hu; subst hu
                    simp only [List.length_append]
                    have := hp r hr_mem; omega
  · cases hs
  · cases hs

private theorem lift_ok_eq {o : Option X86Int.St} {s' : X86Vm.St}
    (h : X86Int.lift o = .ok s') : o = some s' := by
  cases o with
  | none => simp [X86Int.lift] at h
  | some s1 => simp [X86Int.lift] at h; subst h; rfl

private theorem int_exec_bytesLt_len_pos {ii : X86Int.Ins} {len : Nat} {s s' : X86Vm.St}
    (hb : BytesLt s.mem) (hp : ∀ u ∈ s.mem, 0 < u.data.length)
    (hs : X86Int.exec ii len s = .ok s') :
    BytesLt s'.mem ∧ ∀ u ∈ s'.mem, 0 < u.data.length := by
  cases ii with
  | st b disp r =>
    have ho := lift_ok_eq hs
    simp only [bind, Option.bind, pure] at ho
    cases h1 : s.reg b <;> cases h2 : s.reg r <;> simp [h1, h2] at ho
    cases h3 : wr64 s.mem (ea _ disp) _ with
    | none => rw [h3] at ho; simp at ho
    | some m' =>
      rw [h3] at ho; cases ho
      exact ⟨bytesLt_wr64 hb h3, len_pos_wrN hp h3⟩
  | st8 b disp r =>
    have ho := lift_ok_eq hs
    simp only [bind, Option.bind, pure] at ho
    cases h1 : s.reg b <;> cases h2 : s.reg r <;> simp [h1, h2] at ho
    cases h3 : s.mem.wr (ea _ disp) (_ % 256) with
    | none => rw [h3] at ho; simp at ho
    | some m' =>
      rw [h3] at ho; cases ho
      exact ⟨bytesLt_wr hb h3, len_pos_wr hp h3⟩
  | pushI imm =>
    have ho := lift_ok_eq hs
    simp only [bind, Option.bind, pure] at ho
    cases h1 : s.reg 4 <;> simp [h1] at ho
    cases h3 : wr64 s.mem ((_ + X86Env.W - 8) % X86Env.W) _ with
    | none => rw [h3] at ho; simp at ho
    | some m' =>
      rw [h3] at ho; cases ho
      exact ⟨bytesLt_wr64 hb h3, len_pos_wrN hp h3⟩
  | call t =>
    have ho := lift_ok_eq hs
    simp only [bind, Option.bind, pure] at ho
    cases h1 : s.reg 4 <;> simp [h1] at ho
    cases h3 : wr64 s.mem ((_ + X86Env.W - 8) % X86Env.W) _ with
    | none => rw [h3] at ho; simp at ho
    | some m' =>
      rw [h3] at ho; cases ho
      exact ⟨bytesLt_wr64 hb h3, len_pos_wrN hp h3⟩
  | push r =>
    have ho := lift_ok_eq hs
    simp only [bind, Option.bind, pure] at ho
    cases h1 : s.reg r <;> cases h2 : s.reg 4 <;> simp [h1, h2] at ho
    cases h3 : wr64 s.mem ((_ + X86Env.W - 8) % X86Env.W) _ with
    | none => rw [h3] at ho; simp at ho
    | some m' =>
      rw [h3] at ho; cases ho
      exact ⟨bytesLt_wr64 hb h3, len_pos_wrN hp h3⟩
  | syscall =>
    simp only [X86Int.exec] at hs
    cases h0 : s.reg 0 <;> cases h7 : s.reg 7 <;> cases h6 : s.reg 6 <;>
      cases h2 : s.reg 2 <;> cases h10 : s.reg 10 <;> cases h8 : s.reg 8 <;>
      cases h9 : s.reg 9 <;> simp [h0, h7, h6, h2, h10, h8, h9] at hs
    cases hsy : X86Env.syscall s.k s.mem _ _ _ _ _ _ _ with
    | none => rw [hsy] at hs; simp at hs
    | some sr =>
      cases sr with
      | exit _ => rw [hsy] at hs; simp at hs
      | ret rax k' m' =>
        rw [hsy] at hs; cases hs
        exact bytesLt_syscall hb hp hsy
  | movI _ _ | jmp _ =>
    cases hs; exact ⟨hb, hp⟩
  | mov _ _ | mov32 _ _ | alu _ _ _ | aluI _ _ _ | cmp _ _ | cmpI _ _ | test _ _
  | shlI _ _ | shrI _ _ | ld _ _ _ | ld32 _ _ _ | ld8 _ _ _ | lea _ _ _
  | jcc _ _ | jtab _ _ _ | jmpR _ | ret | pop _ =>
    have ho := lift_ok_eq hs
    simp only [bind, Option.bind, pure] at ho
    repeat' (split at ho <;> simp at ho)
    cases ho; exact ⟨hb, hp⟩

theorem step_bytesLt_len_pos {im : Nat → Option (X86Vm.Ins × Nat)} {s s' : X86Vm.St}
    (hb : BytesLt s.mem) (hp : ∀ u ∈ s.mem, 0 < u.data.length)
    (hs : X86Vm.step im s = .ok s') :
    BytesLt s'.mem ∧ ∀ u ∈ s'.mem, 0 < u.data.length := by
  simp only [X86Vm.step] at hs
  cases him : im s.rip with
  | none => rw [him] at hs; simp at hs
  | some p =>
    rcases p with ⟨ins, len⟩
    rw [him] at hs; dsimp only [X86Vm.exec] at hs
    cases ins with
    | ofInt ii => exact int_exec_bytesLt_len_pos hb hp hs
    | st16 b disp r =>
      have ho := lift_ok_eq hs
      simp only [bind, Option.bind, pure] at ho
      cases h1 : s.reg b <;> cases h2 : s.reg r <;> simp [h1, h2] at ho
      cases h3 : wr16 s.mem (ea _ disp) _ with
      | none => rw [h3] at ho; simp at ho
      | some m' =>
        rw [h3] at ho; cases ho
        exact ⟨bytesLt_wrN hb h3, len_pos_wrN hp h3⟩
    | st32 b disp r =>
      have ho := lift_ok_eq hs
      simp only [bind, Option.bind, pure] at ho
      cases h1 : s.reg b <;> cases h2 : s.reg r <;> simp [h1, h2] at ho
      cases h3 : wr32 s.mem (ea _ disp) _ with
      | none => rw [h3] at ho; simp at ho
      | some m' =>
        rw [h3] at ho; cases ho
        exact ⟨bytesLt_wrN hb h3, len_pos_wrN hp h3⟩
    | stIdx w b i scale disp r =>
      have ho := lift_ok_eq hs
      simp only [bind, Option.bind, pure] at ho
      cases h1 : s.reg b <;> cases h2 : s.reg i <;> cases h3 : s.reg r <;> simp [h1, h2, h3] at ho
      cases h4 : wrMem w s.mem (eaIdx _ _ scale disp) _ with
      | none => rw [h4] at ho; simp at ho
      | some m' =>
        rw [h4] at ho; cases ho
        exact ⟨bytesLt_wrMem_raw hb h4, len_pos_wrMem hp h4⟩
    | div _ | idiv _ =>
      have ho := lift_ok_eq hs
      simp only [bind, Option.bind, pure] at ho
      repeat' (split at ho <;> simp at ho)
      rcases ho with ⟨_, _, ho⟩
      rw [← ho]; exact ⟨hb, hp⟩
    | mul _ | imul _ _ | cqo | neg _ | not _ | shlR _ | shrR _ | sarR _
    | sarI _ _ | ld16 _ _ _ | ldIdx _ _ _ _ _ _ | leaIdx _ _ _ _ _ | testI _ _ | cmpAlI _
    | testAlI _ | setcc _ _ | cmovcc _ _ _ =>
      have ho := lift_ok_eq hs
      simp only [bind, Option.bind, pure] at ho
      repeat' (split at ho <;> simp at ho)
      rw [← ho]; exact ⟨hb, hp⟩

theorem reaches_bytesLt_len_pos {im : Nat → Option (X86Vm.Ins × Nat)} {s s' : X86Vm.St}
    (hb : BytesLt s.mem) (hp : ∀ u ∈ s.mem, 0 < u.data.length)
    (hr : Reaches im s s') :
    BytesLt s'.mem ∧ ∀ u ∈ s'.mem, 0 < u.data.length := by
  obtain ⟨k, hk⟩ := hr
  induction k generalizing s with
  | zero =>
    simp [X86Vm.run] at hk
    subst hk
    exact ⟨hb, hp⟩
  | succ k ih =>
    simp only [X86Vm.run] at hk
    cases hst : X86Vm.step im s with
    | ok s1 =>
      rw [hst] at hk
      have ⟨hb1, hp1⟩ := step_bytesLt_len_pos hb hp hst
      exact ih hb1 hp1 hk
    | exit _ =>
      rw [hst] at hk
      cases k <;> simp [X86Vm.run] at hk
    | stuck =>
      rw [hst] at hk
      cases k <;> simp [X86Vm.run] at hk

private theorem reg_set_self_ok {s : X86Vm.St} (hr : RegsOk s) {r : Nat} (hlt : r < 16) (v : Option Nat) :
    (s.set r v).reg r = v :=
  reg_set_self (by rw [hr.1]; exact hlt) v

private theorem reg_with_rip_set_ne {s : X86Vm.St} {r r' rip' : Nat} {v : Option Nat} (hne : r ≠ r') :
    ({ s.set r v with rip := rip' } : X86Vm.St).reg r' = s.reg r' :=
  reg_set_ne hne v

private theorem reg_with_rip_set_self {s : X86Vm.St} (hr : RegsOk s) {r rip' : Nat} (hlt : r < 16) (v : Option Nat) :
    ({ s.set r v with rip := rip' } : X86Vm.St).reg r = v :=
  reg_set_self_ok hr hlt v

private theorem reg_with_mem_rip_set_ne {s : X86Vm.St} {m' : Mem} {r r' rip' : Nat} {v : Option Nat} (hne : r ≠ r') :
    ({ s.set r v with mem := m', rip := rip' } : X86Vm.St).reg r' = s.reg r' :=
  reg_set_ne hne v

private theorem reg_with_fl_rip_set_ne {s : X86Vm.St} {fl' : X86Int.Flags} {r r' rip' : Nat} {v : Option Nat} (hne : r ≠ r') :
    ({ s.set r v with fl := fl', rip := rip' } : X86Vm.St).reg r' = s.reg r' :=
  reg_set_ne hne v

private theorem reg_with_fl_rip_set_self {s : X86Vm.St} (hr : RegsOk s) {fl' : X86Int.Flags} {r rip' : Nat} (hlt : r < 16) (v : Option Nat) :
    ({ s.set r v with fl := fl', rip := rip' } : X86Vm.St).reg r = v :=
  reg_set_self_ok hr hlt v

/-! ### `VmCtx.wf` preservation and context slot read/write helpers -/

namespace VmCtx

theorem sp_bounds {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) :
    vc.stk.base + 32 ≤ vc.sp ∧ vc.sp + 8 ≤ vc.stk.hi ∧ vc.sp < X86Env.W ∧
    8 ≤ vc.sp ∧ 16 ≤ vc.sp ∧ 32 ≤ vc.sp ∧
    (vc.sp + X86Env.W - 8) % X86Env.W = vc.sp - 8 := by
  have h1 := h.2.2.2.2.2.2.2.2.2.1
  have h2 := h.2.2.2.2.2.2.2.2.2.2.1
  have h3 := h.2.2.2.2.2.2.2.2.2.2.2.1
  refine ⟨h1, h2, h3, by omega, by omega, by omega, ?_⟩
  have hsub : vc.sp + X86Env.W - 8 = (vc.sp - 8) + X86Env.W := by omega
  rw [hsub, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]

theorem ctx_bounds {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) :
    vc.ctxReg.base = ctxBase ∧ vc.ctxReg.rw = true ∧ ctxSize ≤ vc.ctxReg.data.length ∧
    vc.ctxReg.hi ≤ 2 ^ 47 ∧ ctxBase + ctxSize ≤ X86Env.W ∧
    vc.ctxReg.hi < X86Env.W := by
  have h1 := h.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h2 := h.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h3 := h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h4 := h.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hcb : ctxBase = 0x600000 := rfl
  have hcs : ctxSize = 4096 := rfl
  have h47 : 2 ^ 47 < X86Env.W := by decide
  exact ⟨h1, h2, h3, h4, by omega, by omega⟩

theorem wf_step_misc {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s s' : X86Vm.St}
    (h : vc.wf pr ls s)
    (hmem : s'.mem = s.mem)
    (hregs : RegsOk s')
    (hsp : s'.reg 4 = some vc.sp)
    (h15 : s'.reg 15 = some ctxBase) :
    vc.wf pr ls s' := by
  rcases h with ⟨hcm, hbl, hpos, _, _, _, hnd, hmem_all, rest⟩
  rw [← hmem] at hcm hbl hpos hmem_all
  exact ⟨hcm, hbl, hpos, hregs, hsp, h15, hnd, hmem_all, rest⟩

theorem wf_write_ctx_scratch {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s s' : X86Vm.St} {idx off : Nat} {bs : List Nat}
    (h : vc.wf pr ls s)
    (hoff : 256 ≤ off)
    (hidx : s.mem[idx]? = some vc.ctxReg)
    (hm' : s'.mem = s.mem.set idx { vc.ctxReg with data := writeBytes vc.ctxReg.data off bs })
    (hcm' : CodeMem s'.mem)
    (hbl' : BytesLt s'.mem)
    (hpos' : ∀ u ∈ s'.mem, 0 < u.data.length)
    (hregs' : RegsOk s')
    (hsp' : s'.reg 4 = some vc.sp)
    (h15' : s'.reg 15 = some ctxBase) :
    let ctx' := { vc.ctxReg with data := writeBytes vc.ctxReg.data off bs }
    { vc with ctxReg := ctx' }.wf pr ls s' := by
  dsimp only
  let ctx' := { vc.ctxReg with data := writeBytes vc.ctxReg.data off bs }
  have hregions_eq : ({ vc with ctxReg := ctx' } : VmCtx).regions.map Region.base =
      vc.regions.map Region.base := by
    simp [VmCtx.regions, ctx']
  have hmem_ctx' : ctx' ∈ s'.mem := by
    have hidx_lt : idx < s.mem.length := (List.getElem?_eq_some_iff.mp hidx).1
    rw [hm']
    exact List.mem_of_getElem? (by rw [List.getElem?_set_self hidx_lt])
  have hmem_all : ∀ r ∈ ({ vc with ctxReg := ctx' } : VmCtx).regions, r ∈ s'.mem := by
    intro r hr
    simp only [VmCtx.regions, List.mem_cons, List.not_mem_nil, or_false] at hr
    have hlen16 : vc.regions.length = 16 := rfl
    have hnot : ∀ (j : Nat) (hj : j < vc.regions.length), j ≠ 2 → vc.regions[j] = r → r ∈ s'.mem := by
      intro j hj hne heq
      have hnb : r.base ≠ vc.ctxReg.base := by
        rw [← heq]
        exact vc.ne_bases h hj (by rw [hlen16]; decide) hne
      have hrm : r ∈ s.mem := h.2.2.2.2.2.2.2.1 r (by rw [← heq]; exact List.getElem_mem hj)
      rw [hm']
      obtain ⟨k, hk⟩ := List.mem_iff_getElem?.mp hrm
      have hki : k ≠ idx := by
        intro heqi; subst heqi; rw [hidx] at hk; cases hk; exact hnb rfl
      exact List.mem_of_getElem? (by rw [List.getElem?_set_ne (Ne.symm hki), hk])
    have hlt : ∀ k : Nat, k < 16 → k < vc.regions.length := fun k hk => by rw [hlen16]; exact hk
    rcases hr with rfl | rfl | hr_ctx | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · exact hnot 0 (hlt 0 (by decide)) (by decide) rfl
    · exact hnot 1 (hlt 1 (by decide)) (by decide) rfl
    · rw [hr_ctx]; exact hmem_ctx'
    · exact hnot 3 (hlt 3 (by decide)) (by decide) rfl
    · exact hnot 4 (hlt 4 (by decide)) (by decide) rfl
    · exact hnot 5 (hlt 5 (by decide)) (by decide) rfl
    · exact hnot 6 (hlt 6 (by decide)) (by decide) rfl
    · exact hnot 7 (hlt 7 (by decide)) (by decide) rfl
    · exact hnot 8 (hlt 8 (by decide)) (by decide) rfl
    · exact hnot 9 (hlt 9 (by decide)) (by decide) rfl
    · exact hnot 10 (hlt 10 (by decide)) (by decide) rfl
    · exact hnot 11 (hlt 11 (by decide)) (by decide) rfl
    · exact hnot 12 (hlt 12 (by decide)) (by decide) rfl
    · exact hnot 13 (hlt 13 (by decide)) (by decide) rfl
    · exact hnot 14 (hlt 14 (by decide)) (by decide) rfl
    · exact hnot 15 (hlt 15 (by decide)) (by decide) rfl
  rcases h with ⟨_, _, _, _, _, _, hnd, _,
    hs1, hs2, hs3, hs4,
    hc1, hc2, hc3, hc4,
    hf1, hf2, hf3,
    hd1, hd2, hd3, hd4, hd5,
    ha1, ha2, ha3, ha4, ha5,
    hr1, hr2, hr3, hr4, hr5,
    hl1, hl2, hl3, hl4, hl5,
    hm1, hm2, hm3, hm4,
    hp1, hp2, hp3,
    ht1, ht2, ht3, ht4, ht5,
    hk1, hk2, hk3, hk4, hk5, hk6,
    hi1,
    ho1, ho2, ho3, ho4, ho5,
    s0, s8, s16, s24, s32, s40, s48, s56, s64, s72, s80, s88,
    s96, s104, s112, s120, s128, s136, s144, s152, s160, s168,
    s176, s184, s192, s200, s208, s216, s224, s232, s240, s248,
    harena⟩
  have hslot : ∀ j, j + 8 ≤ 256 →
      ((ctx'.data.drop j).take 8) = ((vc.ctxReg.data.drop j).take 8) := by
    intro j hj
    exact writeBytes_drop_take_disjoint vc.ctxReg.data off j 8 bs (Or.inl (by omega))
      (by have := hc3; have hcs : ctxSize = 4096 := rfl; omega)
  refine ⟨hcm', hbl', hpos', hregs', hsp', h15', ?_, hmem_all,
    hs1, hs2, hs3, hs4,
    hc1, hc2, ?_, ?_,
    hf1, hf2, hf3,
    hd1, hd2, hd3, hd4, hd5,
    ha1, ha2, ha3, ha4, ha5,
    hr1, hr2, hr3, hr4, hr5,
    hl1, hl2, hl3, hl4, hl5,
    hm1, hm2, hm3, hm4,
    hp1, hp2, hp3,
    ht1, ht2, ht3, ht4, ht5,
    hk1, hk2, hk3, hk4, hk5, hk6,
    hi1,
    ho1, ho2, ho3, ho4, ho5,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    harena⟩
  · rw [hregions_eq]; exact hnd
  · rw [writeBytes_length]; exact hc3
  · simpa [Region.hi, writeBytes_length, ctx'] using hc4
  · rw [hslot 0 (by decide)]; exact s0
  · rw [hslot 8 (by decide)]; exact s8
  · rw [hslot 16 (by decide)]; exact s16
  · rw [hslot 24 (by decide)]; exact s24
  · rw [hslot 32 (by decide)]; exact s32
  · rw [hslot 40 (by decide)]; exact s40
  · rw [hslot 48 (by decide)]; exact s48
  · rw [hslot 56 (by decide)]; exact s56
  · rw [hslot 64 (by decide)]; exact s64
  · rw [hslot 72 (by decide)]; exact s72
  · rw [hslot 80 (by decide)]; exact s80
  · rw [hslot 88 (by decide)]; exact s88
  · rw [hslot 96 (by decide)]; exact s96
  · rw [hslot 104 (by decide)]; exact s104
  · rw [hslot 112 (by decide)]; exact s112
  · rw [hslot 120 (by decide)]; exact s120
  · rw [hslot 128 (by decide)]; exact s128
  · rw [hslot 136 (by decide)]; exact s136
  · rw [hslot 144 (by decide)]; exact s144
  · rw [hslot 152 (by decide)]; exact s152
  · rw [hslot 160 (by decide)]; exact s160
  · rw [hslot 168 (by decide)]; exact s168
  · rw [hslot 176 (by decide)]; exact s176
  · rw [hslot 184 (by decide)]; exact s184
  · rw [hslot 192 (by decide)]; exact s192
  · rw [hslot 200 (by decide)]; exact s200
  · rw [hslot 208 (by decide)]; exact s208
  · rw [hslot 216 (by decide)]; exact s216
  · rw [hslot 224 (by decide)]; exact s224
  · rw [hslot 232 (by decide)]; exact s232
  · rw [hslot 240 (by decide)]; exact s240
  · rw [hslot 248 (by decide)]; exact s248

theorem wr64_ctx_scratch {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} (h : vc.wf pr ls s) (off v : Nat)
    (hoff : 256 ≤ off) (hlim : off + 8 ≤ ctxSize) :
    ∃ (i : Nat) (m' : Mem),
      wr64 s.mem (ea ctxBase off) v = some m' ∧
      s.mem[i]? = some vc.ctxReg ∧
      m' = s.mem.set i { vc.ctxReg with data := writeBytes vc.ctxReg.data off (bytes8 v) } ∧
      CodeMem m' ∧ BytesLt m' ∧ (∀ u ∈ m', 0 < u.data.length) ∧
      leVal (({ vc.ctxReg with data := writeBytes vc.ctxReg.data off (bytes8 v) }.data.drop off).take 8) = v % X86Env.W := by
  have hb := ctx_bounds h
  have hcb := hb.1; have hrw := hb.2.1; have hcl := hb.2.2.1; have hW := hb.2.2.2.2.1
  have hea : ea ctxBase off = ctxBase + off := ea_add_eq (by omega)
  rw [hea]
  obtain ⟨i, m', hwr, hidx, hm', _, hmem', _⟩ :=
    wrMem_w64_region h.1.2 (mem_ctx h) hrw (a := ctxBase + off)
      (by rw [hcb]; omega) (by simp only [Region.hi, hcb]; omega) (v := v)
  have hsub : ctxBase + off - vc.ctxReg.base = off := by rw [hcb]; omega
  rw [hsub] at hm' hmem'
  have hcm' : CodeMem m' := h.1.wr64_preserve hwr
  have hbl' : BytesLt m' := bytesLt_wr64 h.2.1 hwr
  have hpos' : ∀ u ∈ m', 0 < u.data.length := len_pos_wrN h.2.2.1 hwr
  have hval : leVal (({ vc.ctxReg with data := writeBytes vc.ctxReg.data off (bytes8 v) }.data.drop off).take 8) =
      v % X86Env.W :=
    leVal_writeBytes_same vc.ctxReg.data off v (by omega)
  exact ⟨i, m', hwr, hidx, hm', hcm', hbl', hpos', hval⟩

theorem rd64_ctx_of_eq {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} (h : vc.wf pr ls s) {off v : Nat} (hlim : off + 8 ≤ ctxSize)
    (hval : leVal ((vc.ctxReg.data.drop off).take 8) = v) :
    rd64 s.mem (ea ctxBase off) = some v := by
  have hb := ctx_bounds h
  rw [ea_add_eq (by omega)]
  exact rd64_of_mem h.1.2 (mem_ctx h) (by rw [hb.1]; omega)
    (by simp only [Region.hi, hb.1]; omega) (by rw [hb.1, Nat.add_sub_cancel_left]; exact hval)

end VmCtx

/-! ### Runtime failure paths (`L_fail_*` -> `L_vm_fail`) -/

def errId : Sporos.Err → Nat
  | .underflow => 0
  | .auxUnderflow => 1
  | .badLocal => 2
  | .badJump => 3
  | .badWord => 4
  | .noFrame => 5
  | .bounds => 6
  | .divZero => 7
  | .badTable => 8
  | .demand => 9

def errEntry : Sporos.Err → Nat
  | .underflow => SporosVmBin.L_fail_underflow
  | .auxUnderflow => SporosVmBin.L_fail_auxUnderflow
  | .badLocal => SporosVmBin.L_fail_badLocal
  | .badJump => SporosVmBin.L_fail_badJump
  | .badWord => SporosVmBin.L_fail_badWord
  | .noFrame => SporosVmBin.L_fail_noFrame
  | .bounds => SporosVmBin.L_fail_bounds
  | .divZero => SporosVmBin.L_fail_divZero
  | .badTable => SporosVmBin.L_fail_badTable
  | .demand => SporosVmBin.L_fail_demand

theorem errId_lt_10 (e : Sporos.Err) : errId e < 10 := by
  cases e <;> decide

private theorem errTriple_mov (e : Sporos.Err) :
    (errEntry e, (.ofInt (.movI 0 (errId e))), 5) ∈ SporosVmBin.vmTable_22 := by
  cases e <;> decide

private theorem errTriple_st (e : Sporos.Err) :
    (errEntry e + 5, (.ofInt (.st 15 456 0)), 7) ∈ SporosVmBin.vmTable_22 := by
  cases e <;> decide

private theorem errTriple_jmp (e : Sporos.Err) :
    (errEntry e + 12, (.ofInt (.jmp SporosVmBin.L_vm_fail)), 5) ∈ SporosVmBin.vmTable_22 := by
  cases e <;> decide

private theorem errEntry_ok_bounds : SporosVmBin.vmTableChunks[22]? = some (0, SporosVmBin.vmTable_22) ∨ True :=
  Or.inr trivial

theorem L_vm_fail_spec {s : X86Vm.St} {sp err : Nat} {stk ctx : Region}
    (hrip : s.rip = SporosVmBin.L_vm_fail)
    (hcm : CodeMem s.mem) (hregs : RegsOk s)
    (hsp : s.reg 4 = some sp) (h15 : s.reg 15 = some ctxBase)
    (hstk : stk ∈ s.mem) (hstkrw : stk.rw = true) (hstkne : stk.base ≠ ctxBase)
    (hlo : stk.base ≤ sp - 8) (hhi : sp ≤ stk.hi) (h8 : 8 ≤ sp) (hspW : sp < X86Env.W)
    (hctx : ctx ∈ s.mem) (hctxb : ctx.base = ctxBase) (hctxl : ctxSize ≤ ctx.data.length)
    (herr : err < 10)
    (hc456 : leVal ((ctx.data.drop 456).take 8) = err) :
    ∃ s_exit, Reaches vmImage s s_exit ∧
      X86Vm.step vmImage s_exit = .exit 1 ∧
      s_exit.k = s.k := by
  have hcb : ctxBase = 0x600000 := rfl
  have hcs : ctxSize = 4096 := rfl
  have hea456 : ea ctxBase 456 = ctxBase + 456 := ea_add_eq (by decide)
  have hea_sp : (sp + X86Env.W - 8) % X86Env.W = sp - 8 := by
    have : sp + X86Env.W - 8 = (sp - 8) + X86Env.W := by omega
    rw [this, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have hret_sp : ((sp - 8) + 8) % X86Env.W = sp := by
    rw [Nat.sub_add_cancel h8, Nat.mod_eq_of_lt hspW]
  -- 4201218: movI 6 4206984; 4201223: movI 2 20; 4201228: call 4201395
  have h01 : X86Vm.step vmImage s = .ok { s.set 6 (some 4206984) with rip := 4201223 } := by
    rw [step_at_chunk 22 hrip rfl (by decide : (SporosVmBin.L_vm_fail, (.ofInt (.movI 6 4206984)), 5) ∈ SporosVmBin.vmTable_22),
      exec_movI_lt (by decide), hrip]; rfl
  let s1 := { s.set 6 (some 4206984) with rip := 4201223 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) 4206984
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 2 (some 20) with rip := 4201228 } := by
    rw [step_at_chunk 22 rfl rfl (by decide : (4201223, (.ofInt (.movI 2 20)), 5) ∈ SporosVmBin.vmTable_22),
      exec_movI_lt (by decide)]
  let s2 := { s1.set 2 (some 20) with rip := 4201228 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) 20
  have hs2_4 : s2.reg 4 = some sp := by
    simp only [s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 2 ≠ 4), List.getElem?_set_ne (by decide : 6 ≠ 4)]
    exact hsp
  have hs2_15 : s2.reg 15 = some ctxBase := by
    simp only [s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 2 ≠ 15), List.getElem?_set_ne (by decide : 6 ≠ 15)]
    exact h15
  obtain ⟨_, m3, hwr3, _, _, hwf3, hstk3, hoth3⟩ :=
    wr64_region hcm.2 hstk hstkrw hlo (by omega : sp - 8 + 8 ≤ stk.hi) (v := 4201233)
  have hwr3' : wr64 s2.mem ((sp + X86Env.W - 8) % X86Env.W) (4201228 + 5) = some m3 := by
    simpa [s2, s1, hea_sp] using hwr3
  have h03 : X86Vm.step vmImage s2 =
      .ok { s2.set 4 (some (sp - 8)) with mem := m3, rip := 4201395 } := by
    rw [step_at_chunk 22 rfl rfl (by decide : (4201228, (.ofInt (.call 4201395)), 5) ∈ SporosVmBin.vmTable_22),
      exec_call hs2_4 hwr3']
    simp [s2, hea_sp]
  let s3 := { s2.set 4 (some (sp - 8)) with mem := m3, rip := 4201395 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) (sp - 8)
  have hs3_4 : s3.reg 4 = some (sp - 8) := by
    dsimp only [s3]; exact reg_set_self_ok hr2 (by decide) _
  have hs3_15 : s3.reg 15 = some ctxBase := by
    dsimp only [s3]; rw [reg_with_mem_rip_set_ne (by decide : 4 ≠ 15)]; exact hs2_15
  have hret3 : rd64 s3.mem (sp - 8) = some 4201233 := by
    have hr := rd64_of_writeBytes_same hwf3 hstk3 hlo (by omega : sp - 8 + 8 ≤ stk.hi)
    simpa [s3, show (4201233 % X86Env.W) = 4201233 from by decide] using hr
  obtain ⟨s4, hreach4, hs4_rip, hs4_mem, hs4_k, hr4, hs4_4, hs4_oth⟩ :=
    fn_write_stderr_spec (s := s3) (sp := sp - 8) (retAddr := 4201233) rfl hr3 hs3_4 hret3
  have hcm3 : CodeMem m3 := hcm.wr64_preserve (by simpa [hea_sp] using hwr3)
  have hcm4 : CodeMem s4.mem := by rw [hs4_mem]; exact hcm3
  have hs4_4' : s4.reg 4 = some sp := by rw [hs4_4, hret_sp]
  have hs4_15 : s4.reg 15 = some ctxBase := by
    rw [hs4_oth 15 (by decide) (by decide) (by decide) (by decide) (by decide)]
    exact hs3_15
  have hctx4 : ctx ∈ s4.mem := by
    rw [hs4_mem]
    exact hoth3 ctx hctx (by rw [hctxb]; exact Ne.symm hstkne)
  -- 4201233: ld 0 15 456
  have hrd456 : rd64 s4.mem (ea ctxBase 456) = some err := by
    rw [hea456]
    have h464 : 456 + 8 ≤ ctx.data.length :=
      Nat.le_trans (by decide : 456 + 8 ≤ 4096) (by simpa [ctxSize] using hctxl)
    have hctx4_hi : ctxBase + 456 + 8 ≤ ctx.hi := by
      rw [Region.hi, hctxb]; exact Nat.add_le_add_left h464 ctxBase
    exact rd64_of_mem hcm4.2 hctx4 (by rw [hctxb]; exact Nat.le_add_right ctxBase 456) hctx4_hi
      (by rw [hctxb, Nat.add_sub_cancel_left]; exact hc456)
  have h05 : X86Vm.step vmImage s4 = .ok { s4.set 0 (some err) with rip := 4201240 } := by
    rw [step_at_chunk 22 hs4_rip rfl (by decide : (4201233, (.ofInt (.ld 0 15 456)), 7) ∈ SporosVmBin.vmTable_22),
      exec_ld hs4_15 hrd456, hs4_rip]
  let s5 := { s4.set 0 (some err) with rip := 4201240 }
  have hr5 : RegsOk s5 := regsOk_set hr4 (by decide) err
  -- 4201240: movI 1 4207089
  have h06 : X86Vm.step vmImage s5 = .ok { s5.set 1 (some 4207089) with rip := 4201245 } := by
    rw [step_at_chunk 22 rfl rfl (by decide : (4201240, (.ofInt (.movI 1 4207089)), 5) ∈ SporosVmBin.vmTable_22),
      exec_movI_lt (by decide)]
  let s6 := { s5.set 1 (some 4207089) with rip := 4201245 }
  have hr6 : RegsOk s6 := regsOk_set hr5 (by decide) 4207089
  have hs6_1 : s6.reg 1 = some 4207089 := by
    dsimp only [s6]; exact reg_with_rip_set_self hr5 (by decide) _
  have hs6_0 : s6.reg 0 = some err := by
    dsimp only [s6, s5]; rw [reg_with_rip_set_ne (by decide : 1 ≠ 0)]
    exact reg_with_rip_set_self hr4 (by decide) _
  -- 4201245: ldIdx .w64 6 1 0 8 0
  have hptr_ex : ∃ ptr, (roQwords SporosVmBin.L_err_ptrs 10)[err]? = some ptr :=
    ⟨(roQwords SporosVmBin.L_err_ptrs 10)[err]'(by rw [ro_errPtrs_ok]; simp; exact herr),
      List.getElem?_eq_getElem _⟩
  obtain ⟨errPtr, herrPtr⟩ := hptr_ex
  have hrd_ptr : rdMem .w64 s6.mem (eaIdx 4207089 err 8 0) = some errPtr := by
    have h47 : 4207089 + 8 * err < X86Env.W := by have : X86Env.W = 2 ^ 64 := rfl; omega
    have hro_lim : SporosVmBin.L_err_ptrs + 8 * 10 ≤ baseAddr + SporosVmBin.vmBytes.length := by
      have hb := ro_bounds_ok.2.1
      have hptr_lens : SporosVmBin.L_err_ptrs + 80 ≤ SporosVmBin.L_err_lens := by decide
      omega
    rw [eaIdx_add_eq h47]
    exact roQwords_rd64 hcm4 (by decide : vmCodeEnd ≤ SporosVmBin.L_err_ptrs) hro_lim herr herrPtr
  have h07 : X86Vm.step vmImage s6 = .ok { s6.set 6 (some errPtr) with rip := 4201249 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201245, (.ldIdx .w64 6 1 0 8 0), 4) ∈ SporosVmBin.vmTable_23),
      exec_ldIdx hs6_1 hs6_0 hrd_ptr]
  let s7 := { s6.set 6 (some errPtr) with rip := 4201249 }
  have hr7 : RegsOk s7 := regsOk_set hr6 (by decide) errPtr
  -- 4201249: movI 1 4207169
  have h08 : X86Vm.step vmImage s7 = .ok { s7.set 1 (some 4207169) with rip := 4201254 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201249, (.ofInt (.movI 1 4207169)), 5) ∈ SporosVmBin.vmTable_23),
      exec_movI_lt (by decide)]
  let s8 := { s7.set 1 (some 4207169) with rip := 4201254 }
  have hr8 : RegsOk s8 := regsOk_set hr7 (by decide) 4207169
  have hs8_1 : s8.reg 1 = some 4207169 := by
    dsimp only [s8]; exact reg_with_rip_set_self hr7 (by decide) _
  have hs8_0 : s8.reg 0 = some err := by
    dsimp only [s8, s7]; rw [reg_with_rip_set_ne (by decide : 1 ≠ 0), reg_with_rip_set_ne (by decide : 6 ≠ 0)]; exact hs6_0
  -- 4201254: ldIdx .w64 2 1 0 8 0
  have hlen_ex : ∃ len, (roQwords SporosVmBin.L_err_lens 10)[err]? = some len :=
    ⟨(roQwords SporosVmBin.L_err_lens 10)[err]'(by rw [ro_errLens_ok]; simp; exact herr),
      List.getElem?_eq_getElem _⟩
  obtain ⟨errLen, herrLen⟩ := hlen_ex
  have hrd_len : rdMem .w64 s8.mem (eaIdx 4207169 err 8 0) = some errLen := by
    have h47 : 4207169 + 8 * err < X86Env.W := by have : X86Env.W = 2 ^ 64 := rfl; omega
    have hlens_eq : SporosVmBin.L_err_lens = 4207169 := by decide
    have hro_lim : SporosVmBin.L_err_lens + 8 * 10 ≤ baseAddr + SporosVmBin.vmBytes.length := by
      have := ro_bounds_ok.2.1; omega
    rw [eaIdx_add_eq h47, ← hlens_eq]
    exact roQwords_rd64 hcm4 (by decide : vmCodeEnd ≤ SporosVmBin.L_err_lens)
      hro_lim herr herrLen
  have h09 : X86Vm.step vmImage s8 = .ok { s8.set 2 (some errLen) with rip := 4201258 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201254, (.ldIdx .w64 2 1 0 8 0), 4) ∈ SporosVmBin.vmTable_23),
      exec_ldIdx hs8_1 hs8_0 hrd_len]
  let s9 := { s8.set 2 (some errLen) with rip := 4201258 }
  have hr9 : RegsOk s9 := regsOk_set hr8 (by decide) errLen
  have hs9_4 : s9.reg 4 = some sp := by
    dsimp only [s9, s8, s7, s6, s5]
    rw [reg_with_rip_set_ne (by decide : 2 ≠ 4), reg_with_rip_set_ne (by decide : 1 ≠ 4),
        reg_with_rip_set_ne (by decide : 6 ≠ 4), reg_with_rip_set_ne (by decide : 1 ≠ 4),
        reg_with_rip_set_ne (by decide : 0 ≠ 4)]
    exact hs4_4'
  have hstk3_mem : { stk with data := writeBytes stk.data (sp - 8 - stk.base) (bytes8 4201233) } ∈ s9.mem := by
    change _ ∈ s4.mem
    rw [hs4_mem]
    exact hstk3
  let stk3 := { stk with data := writeBytes stk.data (sp - 8 - stk.base) (bytes8 4201233) }
  have hstk3_b : stk3.base = stk.base := rfl
  have hstk3_rw : stk3.rw = true := hstkrw
  have hstk3_hi : stk3.hi = stk.hi := by simp [stk3, Region.hi, writeBytes_length]
  obtain ⟨_, m10, hwr10, _, _, hwf10, hstk10, _⟩ :=
    wr64_region hcm4.2 hstk3_mem hstk3_rw (by rw [hstk3_b]; exact hlo)
      (by omega : sp - 8 + 8 ≤ stk3.hi) (v := 4201263)
  have hwr10' : wr64 s9.mem ((sp + X86Env.W - 8) % X86Env.W) (4201258 + 5) = some m10 := by
    simpa [s9, s8, s7, s6, s5, hea_sp] using hwr10
  have h10 : X86Vm.step vmImage s9 =
      .ok { s9.set 4 (some (sp - 8)) with mem := m10, rip := 4201395 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201258, (.ofInt (.call 4201395)), 5) ∈ SporosVmBin.vmTable_23),
      exec_call hs9_4 hwr10']
    simp [s9, hea_sp]
  let s10 := { s9.set 4 (some (sp - 8)) with mem := m10, rip := 4201395 }
  have hr10 : RegsOk s10 := regsOk_set hr9 (by decide) (sp - 8)
  have hs10_4 : s10.reg 4 = some (sp - 8) := by
    dsimp only [s10]; exact reg_set_self_ok hr9 (by decide) _
  have hret10 : rd64 s10.mem (sp - 8) = some 4201263 := by
    have hr := rd64_of_writeBytes_same hwf10 hstk10 (by simpa [hstk3_b] using hlo)
      (by omega : sp - 8 + 8 ≤ stk3.hi)
    simpa [s10, show (4201263 % X86Env.W) = 4201263 from by decide] using hr
  obtain ⟨s11, hreach11, hs11_rip, hs11_mem, hs11_k, hr11, hs11_4, _⟩ :=
    fn_write_stderr_spec (s := s10) (sp := sp - 8) (retAddr := 4201263) rfl hr10 hs10_4 hret10
  have hcm10 : CodeMem m10 := hcm4.wr64_preserve (by simpa [s9, s8, s7, s6, s5, hea_sp] using hwr10)
  have hcm11 : CodeMem s11.mem := by rw [hs11_mem]; exact hcm10
  have hs11_4' : s11.reg 4 = some sp := by rw [hs11_4, hret_sp]
  -- 4201263: movI 6 4207004; 4201268: movI 2 1; 4201273: call 4201395
  have h12 : X86Vm.step vmImage s11 = .ok { s11.set 6 (some 4207004) with rip := 4201268 } := by
    rw [step_at_chunk 23 hs11_rip rfl (by decide : (4201263, (.ofInt (.movI 6 4207004)), 5) ∈ SporosVmBin.vmTable_23),
      exec_movI_lt (by decide), hs11_rip]
  let s12 := { s11.set 6 (some 4207004) with rip := 4201268 }
  have hr12 : RegsOk s12 := regsOk_set hr11 (by decide) 4207004
  have h13 : X86Vm.step vmImage s12 = .ok { s12.set 2 (some 1) with rip := 4201273 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201268, (.ofInt (.movI 2 1)), 5) ∈ SporosVmBin.vmTable_23),
      exec_movI_lt (by decide)]
  let s13 := { s12.set 2 (some 1) with rip := 4201273 }
  have hr13 : RegsOk s13 := regsOk_set hr12 (by decide) 1
  have hs13_4 : s13.reg 4 = some sp := by
    dsimp only [s13, s12]
    rw [reg_with_rip_set_ne (by decide : 2 ≠ 4), reg_with_rip_set_ne (by decide : 6 ≠ 4)]
    exact hs11_4'
  have hstk10_mem : { stk3 with data := writeBytes stk3.data (sp - 8 - stk3.base) (bytes8 4201263) } ∈ s13.mem := by
    change _ ∈ s11.mem
    rw [hs11_mem]
    exact hstk10
  let stk10 := { stk3 with data := writeBytes stk3.data (sp - 8 - stk3.base) (bytes8 4201263) }
  have hstk10_b : stk10.base = stk.base := rfl
  have hstk10_rw : stk10.rw = true := hstkrw
  have hstk10_hi : stk10.hi = stk.hi := by simp [stk10, stk3, Region.hi, writeBytes_length]
  obtain ⟨_, m14, hwr14, _, _, hwf14, hstk14, _⟩ :=
    wr64_region hcm11.2 hstk10_mem hstk10_rw (by change stk.base ≤ sp - 8; exact hlo)
      (by omega : sp - 8 + 8 ≤ stk10.hi) (v := 4201278)
  have hwr14' : wr64 s13.mem ((sp + X86Env.W - 8) % X86Env.W) (4201273 + 5) = some m14 := by
    simpa [s13, s12, hea_sp] using hwr14
  have h14 : X86Vm.step vmImage s13 =
      .ok { s13.set 4 (some (sp - 8)) with mem := m14, rip := 4201395 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201273, (.ofInt (.call 4201395)), 5) ∈ SporosVmBin.vmTable_23),
      exec_call hs13_4 hwr14']
    simp [s13, hea_sp]
  let s14 := { s13.set 4 (some (sp - 8)) with mem := m14, rip := 4201395 }
  have hr14 : RegsOk s14 := regsOk_set hr13 (by decide) (sp - 8)
  have hs14_4 : s14.reg 4 = some (sp - 8) := by
    simp only [s14, X86Int.St.reg, X86Int.St.set]
    rw [getElem?_set_getD_self (by rw [hr13.1]; decide)]
  have hret14 : rd64 s14.mem (sp - 8) = some 4201278 := by
    have hr := rd64_of_writeBytes_same hwf14 hstk14 (by simpa [hstk10_b] using hlo)
      (by omega : sp - 8 + 8 ≤ stk10.hi)
    simpa [s14, show (4201278 % X86Env.W) = 4201278 from by decide] using hr
  obtain ⟨s15, hreach15, hs15_rip, _, hs15_k, hr15, _, _⟩ :=
    fn_write_stderr_spec (s := s14) (sp := sp - 8) (retAddr := 4201278) rfl hr14 hs14_4 hret14
  -- 4201278: movI 7 1; 4201283: movI 0 60; 4201288: syscall
  have h16 : X86Vm.step vmImage s15 = .ok { s15.set 7 (some 1) with rip := 4201283 } := by
    rw [step_at_chunk 23 hs15_rip rfl (by decide : (4201278, (.ofInt (.movI 7 1)), 5) ∈ SporosVmBin.vmTable_23),
      exec_movI_lt (by decide), hs15_rip]
  let s16 := { s15.set 7 (some 1) with rip := 4201283 }
  have hr16 : RegsOk s16 := regsOk_set hr15 (by decide) 1
  have h17 : X86Vm.step vmImage s16 = .ok { s16.set 0 (some 60) with rip := 4201288 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201283, (.ofInt (.movI 0 60)), 5) ∈ SporosVmBin.vmTable_23),
      exec_movI_lt (by decide)]
  let s17 := { s16.set 0 (some 60) with rip := 4201288 }
  have hr17 : RegsOk s17 := regsOk_set hr16 (by decide) 60
  obtain ⟨⟨v0, hv0⟩, ⟨v7, hv7⟩, ⟨v6, hv6⟩, ⟨v2, hv2⟩, ⟨v10, hv10⟩, ⟨v8, hv8⟩, ⟨v9, hv9⟩⟩ :=
    regsOk_syscall_regs hr17
  have hv0' : v0 = 60 := by
    have hs : s17.reg 0 = some 60 := by
      simp only [s17, X86Int.St.reg, X86Int.St.set]
      rw [getElem?_set_getD_self (by rw [hr16.1]; decide)]
    rw [hs] at hv0; cases hv0; rfl
  have hv7' : v7 = 1 := by
    have hs : s17.reg 7 = some 1 := by
      simp only [s17, s16, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 7)]
      rw [getElem?_set_getD_self (by rw [hr15.1]; decide)]
    rw [hs] at hv7; cases hv7; rfl
  subst hv0' hv7'
  have hexit : X86Vm.step vmImage s17 = .exit 1 := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201288, (.ofInt .syscall), 2) ∈ SporosVmBin.vmTable_23),
      exec_syscall_exit (c := 1) hv0 hv7 hv6 hv2 hv10 hv8 hv9 (by simp [X86Env.syscall])]
  have hreach_all : Reaches vmImage s s17 :=
    Reaches.trans (Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 hreach4)))
      (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step_trans h08
        (Reaches.step_trans h09 (Reaches.trans (Reaches.step_trans h10 hreach11)
          (Reaches.step_trans h12 (Reaches.step_trans h13 (Reaches.trans (Reaches.step_trans h14 hreach15)
            (Reaches.step_trans h16 (Reaches.step h17)))))))))))
  refine ⟨s17, hreach_all, hexit, ?_⟩
  have hks15 : s15.k = s.k := by
    change s15.k = s.k
    rw [hs15_k]
    change s14.k = s.k
    change s11.k = s.k
    rw [hs11_k]
    change s4.k = s.k
    rw [hs4_k]
    rfl
  exact hks15

theorem L_fail_spec {e : Sporos.Err} {s : X86Vm.St} {sp : Nat} {stk ctx : Region}
    (hrip : s.rip = errEntry e)
    (hcm : CodeMem s.mem) (hregs : RegsOk s)
    (hsp : s.reg 4 = some sp) (h15 : s.reg 15 = some ctxBase)
    (hstk : stk ∈ s.mem) (hstkrw : stk.rw = true) (hstkne : stk.base ≠ ctxBase)
    (hlo : stk.base ≤ sp - 8) (hhi : sp ≤ stk.hi) (h8 : 8 ≤ sp) (hspW : sp < X86Env.W)
    (hctx : ctx ∈ s.mem) (hctxb : ctx.base = ctxBase) (hctxrw : ctx.rw = true)
    (hctxl : ctxSize ≤ ctx.data.length) :
    ∃ s_exit, Reaches vmImage s s_exit ∧
      X86Vm.step vmImage s_exit = .exit 1 ∧
      s_exit.k = s.k := by
  have hcb : ctxBase = 0x600000 := rfl
  have hcs : ctxSize = 4096 := rfl
  have herr_lt : errId e < X86Env.W := Nat.lt_trans (errId_lt_10 e) (by decide)
  have h01 : X86Vm.step vmImage s =
      .ok { s.set 0 (some (errId e)) with rip := errEntry e + 5 } := by
    rw [step_at_chunk 22 hrip rfl (errTriple_mov e), exec_movI_lt herr_lt, hrip]
  let s1 := { s.set 0 (some (errId e)) with rip := errEntry e + 5 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) (errId e)
  have hs1_15 : s1.reg 15 = some ctxBase := by
    dsimp only [s1]; rw [reg_with_rip_set_ne (by decide : 0 ≠ 15)]; exact h15
  have hs1_0 : s1.reg 0 = some (errId e) := by
    dsimp only [s1]; exact reg_with_rip_set_self hregs (by decide) _
  have hea456 : ea ctxBase 456 = ctxBase + 456 := ea_add_eq (by decide)
  have h464 : 456 + 8 ≤ ctx.data.length :=
    Nat.le_trans (by decide : 456 + 8 ≤ 4096) (by simpa [ctxSize] using hctxl)
  have hctx_hi456 : ctxBase + 456 + 8 ≤ ctx.hi := by
    rw [Region.hi, hctxb]; exact Nat.add_le_add_left h464 ctxBase
  obtain ⟨_, m2, hwr2, _, _, hwf2, hctx2, hoth2⟩ :=
    wrMem_w64_region hcm.2 hctx hctxrw (a := ctxBase + 456)
      (by rw [hctxb]; exact Nat.le_add_right ctxBase 456)
      hctx_hi456 (v := errId e)
  have hsub : ctxBase + 456 - ctx.base = 456 := by rw [hctxb, Nat.add_sub_cancel_left]
  rw [hsub] at hctx2
  have hwr2' : wr64 s1.mem (ea ctxBase 456) (errId e) = some m2 := by
    simpa [s1, wr64, wrMem, hea456] using hwr2
  have h02 : X86Vm.step vmImage s1 = .ok { s1 with mem := m2, rip := errEntry e + 12 } := by
    rw [step_at_chunk 22 rfl rfl (errTriple_st e), exec_st hs1_15 hs1_0 hwr2']
  let s2 := { s1 with mem := m2, rip := errEntry e + 12 }
  have h03 : X86Vm.step vmImage s2 = .ok { s2 with rip := SporosVmBin.L_vm_fail } := by
    rw [step_at_chunk 22 rfl rfl (errTriple_jmp e), exec_jmp]
  let s3 := { s2 with rip := SporosVmBin.L_vm_fail }
  have hcm2 : CodeMem m2 := hcm.wr64_preserve hwr2
  have hstk3 : stk ∈ s3.mem := hoth2 stk hstk (by rw [hctxb]; exact hstkne)
  let ctx2 := { ctx with data := writeBytes ctx.data 456 (bytes8 (errId e)) }
  have hc456 : leVal ((ctx2.data.drop 456).take 8) = errId e := by
    rw [leVal_writeBytes_same _ _ _ (by omega), Nat.mod_eq_of_lt herr_lt]
  have hs3_4 : s3.reg 4 = some sp := by
    simp only [s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 4)]
    exact hsp
  have hs3_15 : s3.reg 15 = some ctxBase := hs1_15
  obtain ⟨s_exit, hreach, hexit, hk⟩ := L_vm_fail_spec (s := s3) (sp := sp) (err := errId e)
    (stk := stk) (ctx := ctx2) rfl hcm2 hr1 hs3_4 hs3_15 hstk3 hstkrw hstkne hlo hhi h8 hspW
    hctx2 hctxb (by simpa [ctx2, writeBytes_length] using hctxl) (errId_lt_10 e) hc456
  exact ⟨s_exit, Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 hreach)), hexit, hk⟩

theorem L_fail_from_vc_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {e : Sporos.Err} {s : X86Vm.St}
    (hrip : s.rip = errEntry e) (h : vc.wf pr ls s) :
    ∃ s_exit, Reaches vmImage s s_exit ∧
      X86Vm.step vmImage s_exit = .exit 1 ∧
      s_exit.k = s.k := by
  have hspb := vc.sp_bounds h
  have hcb := vc.ctx_bounds h
  have hlt1 : 1 < vc.regions.length := by rw [show vc.regions.length = 16 from rfl]; decide
  have hlt2 : 2 < vc.regions.length := by rw [show vc.regions.length = 16 from rfl]; decide
  have hne : vc.stk.base ≠ ctxBase := by
    rw [← hcb.1]
    simpa [VmCtx.regions] using vc.ne_bases h hlt1 hlt2 (by decide)
  exact L_fail_spec hrip h.1 h.2.2.2.1 h.2.2.2.2.1 h.2.2.2.2.2.1 (vc.mem_stk h)
    h.2.2.2.2.2.2.2.2.1 hne (by omega) (by omega) hspb.2.2.2.1 hspb.2.2.1 (vc.mem_ctx h)
    hcb.1 hcb.2.1 hcb.2.2.1

/-! ### Oracle extraction and wrapper lemmas for `mremap` & growable pushes -/

theorem mem_eraseIdx_of_ne_base : ∀ (m : Mem) (idx : Nat) {r u : Region},
    m[idx]? = some r → u ∈ m → u.base ≠ r.base → u ∈ m.eraseIdx idx
  | [], _, _, _, hget, _, _ => by simp at hget
  | _ :: xs, 0, r, u, hget, hu, hne => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at hget
    subst hget
    rw [List.mem_cons] at hu
    rcases hu with rfl | hu
    · exact absurd rfl hne
    · simp [hu]
  | _ :: xs, idx + 1, r, u, hget, hu, hne => by
    simp only [List.getElem?_cons_succ] at hget
    rw [List.mem_cons] at hu
    rcases hu with rfl | hu
    · simp
    · simp only [List.eraseIdx_cons_succ, List.mem_cons]
      exact Or.inr (mem_eraseIdx_of_ne_base xs idx hget hu hne)

theorem mem_free_of_eraseIdx_same_bounds {m : Mem} {idx a n : Nat} {r u : Region}
    (hget : m[idx]? = some r)
    (hfree : X86Env.Mem.free (m.eraseIdx idx) a n = true)
    (hu : u ∈ m) (hne : u.base ≠ r.base) :
    a + n ≤ u.base ∨ u.hi ≤ a := by
  have hue : u ∈ m.eraseIdx idx := mem_eraseIdx_of_ne_base m idx hget hu hne
  rw [X86Env.Mem.free, List.all_eq_true] at hfree
  have h1 := hfree u hue
  simp only [Bool.or_eq_true, decide_eq_true_eq] at h1
  exact h1

theorem mem_free_ne_base {a n : Nat} {u : Region}
    (hn : 0 < n) (hupos : 0 < u.data.length) (hb : a + n ≤ u.base ∨ u.hi ≤ a) :
    a ≠ u.base := by
  intro heq
  subst heq
  simp only [Region.hi] at hb
  omega

theorem at_mremap_oracle_spec {s : X86Vm.St} {oldPtr oldLen newLen idx : Nat} {oldReg : Region}
    (hrip : s.rip = SporosVmBin.fn_mremap)
    (hregs : RegsOk s) (hor : OracleAdmits s)
    (hr7 : s.reg 7 = some oldPtr) (hr6 : s.reg 6 = some oldLen) (hr2 : s.reg 2 = some newLen)
    (hfind : s.mem.find oldPtr 0 = some (idx, oldReg))
    (hbase : oldReg.base = oldPtr) (holen : oldReg.data.length = X86Env.pages oldLen) (hrw : oldReg.rw = true)
    (hnlen : oldReg.data.length ≤ X86Env.pages newLen) :
    ((s.k.next).1 = some .nomem ∧ HitsNomem s) ∨
    (∃ a, (s.k.next).1 = some (.at a) ∧
      a % X86Env.page = 0 ∧ 0 < a ∧ a + X86Env.pages newLen ≤ 2 ^ 47 ∧
      X86Env.Mem.free (s.mem.eraseIdx idx) a (X86Env.pages newLen) = true) := by
  have h01 : X86Vm.step vmImage s = .ok { s.set 10 (some 1) with rip := 4201369 } := by
    rw [step_at_chunk 23 hrip rfl (by decide : (SporosVmBin.fn_mremap, (.ofInt (.movI 10 1)), 6) ∈ SporosVmBin.vmTable_23),
      exec_movI_lt (by decide), hrip]; rfl
  let s1 := { s.set 10 (some 1) with rip := 4201369 }
  have hr1 : RegsOk s1 := regsOk_set hregs (by decide) 1
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 8 (some 0) with rip := 4201375 } := by
    rw [step_at_chunk 23 rfl rfl (by decide : (4201369, (.ofInt (.movI 8 0)), 6) ∈ SporosVmBin.vmTable_23),
      exec_movI_lt (by decide)]
  let s2 := { s1.set 8 (some 0) with rip := 4201375 }
  have hr2' : RegsOk s2 := regsOk_set hr1 (by decide) 0
  obtain ⟨v9, hv9⟩ := regs_get_of_ok hr2' 9 (by decide) (by decide)
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
  have hs4_k : s4.k = s.k := rfl
  have hs4_m : s4.mem = s.mem := rfl
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
    dsimp only [s4, s3]
    rw [reg_with_rip_set_ne (by decide : 0 ≠ 9)]
    exact reg_with_fl_rip_set_self hr2' (by decide) _
  have hreach4 : Reaches vmImage s s4 :=
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step h04)))
  obtain ⟨k4, hk4⟩ := hreach4
  have hs4_rip : s4.rip = 4201383 := rfl
  have hat4 : AtMemSyscall s4 :=
    ⟨by
        rw [hs4_rip]
        exact table_ok (4201383, (.ofInt .syscall), 2)
          (mem_vmTable_of_chunk 23 rfl (by decide : (4201383, (.ofInt .syscall), 2) ∈ SporosVmBin.vmTable_23)),
      Or.inr hs4_0⟩
  obtain ⟨nr, a1, a2, a3, a4, a5, a6, rax, k', m', hn0, hn7, hn6, hn2, hn10, hn8, hn9, hsys⟩ :=
    hor k4 s4 hk4 hat4
  rw [hs4_0, Option.some.injEq] at hn0; subst hn0
  rw [hs4_7, Option.some.injEq] at hn7; subst hn7
  rw [hs4_6, Option.some.injEq] at hn6; subst hn6
  rw [hs4_2, Option.some.injEq] at hn2; subst hn2
  rw [hs4_10, Option.some.injEq] at hn10; subst hn10
  rw [hs4_8, Option.some.injEq] at hn8; subst hn8
  rw [hs4_9, Option.some.injEq] at hn9; subst hn9
  rw [hs4_k, hs4_m] at hsys
  have hnotlt : ¬ X86Env.pages newLen < oldReg.data.length := by omega
  have heqb : ¬ oldReg.base ≠ oldPtr := fun hc => hc hbase
  have heql : ¬ oldReg.data.length ≠ X86Env.pages oldLen := fun hc => hc holen
  have hnrw : ¬ ¬ oldReg.rw = true := fun hc => hc hrw
  simp only [X86Env.syscall, X86Env.MREMAP_MAYMOVE, ne_self_iff_false, hfind, heqb, heql, hnrw,
    or_self, if_neg (by decide : ¬ False), show ¬ (X86Env.pages newLen < oldReg.data.length) from hnotlt,
    bind, Option.bind] at hsys
  cases hnext : s.k.next with
  | mk c kr =>
    rw [hnext] at hsys
    dsimp only at hsys
    cases c with
    | none => simp [X86Env.place] at hsys
    | some ch =>
      cases ch with
      | «at» a =>
        by_cases hok : a % X86Env.page = 0 ∧ 0 < a ∧ a + X86Env.pages newLen ≤ 2 ^ 47 ∧
            X86Env.Mem.free (s.mem.eraseIdx idx) a (X86Env.pages newLen) = true
        · exact Or.inr ⟨a, rfl, hok.1, hok.2.1, hok.2.2.1, hok.2.2.2⟩
        · simp [X86Env.place, hok] at hsys
      | nomem =>
        have hhd : s4.k.oracle.head? = some .nomem := by
          rw [hs4_k]
          cases horac : s.k.oracle with
          | nil => simp [X86Env.Kern.next, horac] at hnext
          | cons hd tl =>
            simp only [X86Env.Kern.next, horac, Prod.mk.injEq, Option.some.injEq] at hnext
            simpa using hnext.1
        exact Or.inl ⟨rfl, ⟨k4, s4, hk4, hat4, hhd⟩⟩
      | count _ => simp [X86Env.place] at hsys

theorem call_mremap_oracle_spec {s : X86Vm.St} {sp oldPtr oldLen newLen idx : Nat} {oldReg stk : Region}
    (hrip : s.rip = SporosVmBin.fn_mremap)
    (hregs : RegsOk s) (hor : OracleAdmits s)
    (hsp : s.reg 4 = some sp)
    (h8 : 8 ≤ sp) (hspW : sp < X86Env.W)
    (hmwf : s.mem.wf)
    (hstk : stk ∈ s.mem) (hstkrw : stk.rw = true) (hstkne : stk.base ≠ oldPtr)
    (hlo : stk.base ≤ sp - 8) (hhi : sp + 8 ≤ stk.hi)
    {retAddr : Nat} (hret : rd64 s.mem sp = some retAddr)
    (hr7 : s.reg 7 = some oldPtr) (hr6 : s.reg 6 = some oldLen) (hr2 : s.reg 2 = some newLen)
    (hfind : s.mem.find oldPtr 0 = some (idx, oldReg))
    (hbase : oldReg.base = oldPtr) (holen : oldReg.data.length = X86Env.pages oldLen) (hrw : oldReg.rw = true)
    (hnlen : oldReg.data.length ≤ X86Env.pages newLen) :
    (∃ a s', Reaches vmImage s s' ∧
      (s.k.next).1 = some (.at a) ∧
      a % X86Env.page = 0 ∧ 0 < a ∧ a + X86Env.pages newLen ≤ 2 ^ 47 ∧
      X86Env.Mem.free (s.mem.eraseIdx idx) a (X86Env.pages newLen) = true ∧
      s'.rip = retAddr ∧ s'.k = (s.k.next).2 ∧
      s'.mem = (s.mem.eraseIdx idx) ++ [(⟨a, oldReg.data ++ List.replicate (X86Env.pages newLen - oldReg.data.length) 0, true⟩ : Region)] ∧
      s'.mem.wf ∧ RegsOk s' ∧ s'.reg 0 = some a ∧ s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      OracleAdmits s' ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 4 → r ≠ 6 → r ≠ 7 → r ≠ 8 → r ≠ 9 → r ≠ 10 → r ≠ 11 → s'.reg r = s.reg r)) ∨
    (HitsNomem s ∧ Ends vmImage s (.exit 2)) := by
  rcases at_mremap_oracle_spec hrip hregs hor hr7 hr6 hr2 hfind hbase holen hrw hnlen with
    ⟨hnomem, hhit⟩ | ⟨a, hat, hpg, hpos, hlim, hfree⟩
  · exact Or.inr ⟨hhit, fn_mremap_nomem_spec hrip hregs hsp h8 hspW hmwf hstk hstkrw hlo
      (by omega) hr7 hr6 hr2 hfind hbase holen hrw hnlen hnomem⟩
  · obtain ⟨s', hreach, hrip', hk', hmem', hwf', hr', h0', h4', hoth⟩ :=
      fn_mremap_ok_spec hrip hregs hsp hstk hstkne (by omega) hhi hret hmwf hr7 hr6 hr2 hfind
        hbase holen hrw hnlen hat hpg hpos hlim hfree
    exact Or.inl ⟨a, s', hreach, hat, hpg, hpos, hlim, hfree, hrip', hk', hmem', by rw [hmem']; exact hwf', hr', h0', h4',
      oracleAdmits_of_reaches hor hreach, hoth⟩

theorem take8_drop_eq_of_disjoint {l1 l2 : List Nat} {lo hi off : Nat}
    (_hlen : l1.length = l2.length)
    (heq : ∀ j, (j < lo ∨ hi ≤ j) → l1[j]? = l2[j]?)
    (hdisj : off + 8 ≤ lo ∨ hi ≤ off) :
    (l1.drop off).take 8 = (l2.drop off).take 8 := by
  apply List.ext_getElem?
  intro k
  by_cases hk : k < 8
  · rw [List.getElem?_take_of_lt hk, List.getElem?_take_of_lt hk,
      List.getElem?_drop, List.getElem?_drop]
    exact heq (off + k) (by rcases hdisj with h | h <;> omega)
  · have hk8 : 8 ≤ k := Nat.le_of_not_lt hk
    rw [List.getElem?_eq_none (by rw [List.length_take]; omega),
        List.getElem?_eq_none (by rw [List.length_take]; omega)]

theorem mem_eq_of_same_base {m : Mem} (hmwf : m.wf) (hpos : ∀ u ∈ m, 0 < u.data.length)
    {x y : Region} (hx : x ∈ m) (hy : y ∈ m) (hb : x.base = y.base) : x = y := by
  exact Classical.byContradiction fun hne => by
    have hdisj := mem_pairwise_disj hmwf.2 hx hy hne
    have hxpos := hpos x hx
    have hypos := hpos y hy
    simp only [Region.hi] at hdisj
    omega

theorem mremap_find_after_stack_write {m m' : Mem} {stk stk' r : Region} {base idx a n : Nat}
    (hmwf : m.wf)
    (hstk : stk ∈ m) (hstk_b : stk'.base = stk.base) (hstk_hi : stk'.hi = stk.hi)
    (hstk_ne : stk.base ≠ base)
    (hstk_in : stk' ∈ m') (hoth : ∀ u ∈ m, u.base ≠ stk.base → u ∈ m')
    (hget' : m'[idx]? = some r) (hrb : r.base = base)
    (hfree : X86Env.Mem.free (m'.eraseIdx idx) a n = true)
    (hnpos : 0 < n) (hpos : ∀ u ∈ m, 0 < u.data.length) :
    a ≠ stk.base ∧ (∀ u ∈ m, u.base ≠ base → a + n ≤ u.base ∨ u.hi ≤ a) := by
  have hmem_of : ∀ u ∈ m, u.base ≠ base → a + n ≤ u.base ∨ u.hi ≤ a := by
    intro u hu hne
    by_cases hus : u.base = stk.base
    · have heq := mem_eq_of_same_base hmwf hpos hu hstk hus
      subst heq
      have h1 := mem_free_of_eraseIdx_same_bounds hget' hfree hstk_in (by rw [hstk_b, hrb]; exact hstk_ne)
      simpa [hstk_b, hstk_hi] using h1
    · exact mem_free_of_eraseIdx_same_bounds hget' hfree (hoth u hu hus) (by rw [hrb]; exact hne)
  exact ⟨mem_free_ne_base hnpos (hpos stk hstk) (hmem_of stk hstk hstk_ne), hmem_of⟩

private theorem call_ds_push_nogrow_oracle_spec {s : X86Vm.St} {sp retAddr ctxBase dsBase dsLen dsCap w p : Nat}
    {ctx dsReg stk : Region}
    (hrip : s.rip = SporosVmBin.fn_ds_push)
    (hregs : RegsOk s) (hor : OracleAdmits s)
    (hbl : BytesLt s.mem) (hpos : ∀ u ∈ s.mem, 0 < u.data.length)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr7 : s.reg 7 = some w) (hr6 : s.reg 6 = some p)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_rw : ctx.rw = true)
    (hctx_len : ctxSize ≤ ctx.data.length)
    (hds : dsReg ∈ s.mem) (hds_base : dsReg.base = dsBase) (hds_rw : dsReg.rw = true)
    (hds_hi : dsReg.hi ≤ 2 ^ 47)
    (hne : dsBase ≠ ctxBase)
    (hstk : stk ∈ s.mem) (_hstkrw : stk.rw = true) (hstkne_ds : stk.base ≠ dsBase) (hstkne_ctx : stk.base ≠ ctxBase)
    (h16 : 16 ≤ sp) (_hspW : sp < X86Env.W)
    (hstk_lo : stk.base ≤ sp - 16) (hstk_hi : sp + 8 ≤ stk.hi)
    (_hle : dsLen ≤ dsCap) (hcap_pos : 0 < dsCap)
    (hds_eq : dsReg.data.length = dsCap * 16)
    (hds_pg : (dsCap * 16) % X86Env.page = 0)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W)
    (hmwf : s.mem.wf)
    (hret : rd64 s.mem sp = some retAddr)
    (hc0 : leVal ((ctx.data.drop 0).take 8) = dsBase)
    (hc8 : leVal ((ctx.data.drop 8).take 8) = dsLen)
    (hc16 : leVal ((ctx.data.drop 16).take 8) = dsCap)
    (hlt : dsLen < dsCap) :
    (∃ dsReg' ctx' stk' m' s' dsCap',
      Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.mem = m' ∧ m'.wf ∧ BytesLt m' ∧ (∀ u ∈ m', 0 < u.data.length) ∧
      RegsOk s' ∧ OracleAdmits s' ∧ s'.k.out = s.k.out ∧
      s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      dsCap ≤ dsCap' ∧ dsLen + 1 ≤ dsCap' ∧ (dsCap' * 16) % X86Env.page = 0 ∧
      dsReg' ∈ m' ∧ dsReg'.rw = true ∧ dsReg'.hi ≤ 2 ^ 47 ∧
      dsReg'.data = writeBytes (writeBytes (dsReg.data ++ List.replicate ((dsCap' - dsCap) * 16) 0)
        (dsLen * 16) (bytes8 w)) (dsLen * 16 + 8) (bytes8 p) ∧
      ctx' ∈ m' ∧ ctx'.base = ctx.base ∧ ctx'.rw = ctx.rw ∧ ctx'.data.length = ctx.data.length ∧
      leVal ((ctx'.data.drop 0).take 8) = dsReg'.base ∧
      leVal ((ctx'.data.drop 8).take 8) = dsLen + 1 ∧
      leVal ((ctx'.data.drop 16).take 8) = dsCap' ∧
      (∀ j, (j < 0 ∨ 24 ≤ j) → ctx'.data[j]? = ctx.data[j]?) ∧
      stk' ∈ m' ∧ stk'.base = stk.base ∧ stk'.rw = stk.rw ∧ stk'.data.length = stk.data.length ∧
      (∀ u ∈ s.mem, u.base ≠ dsBase → u.base ≠ dsReg'.base) ∧
      (∀ u ∈ s.mem, u.base ≠ dsBase → u.base ≠ ctxBase → u.base ≠ stk.base → u ∈ m') ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 4 → r ≠ 6 → r ≠ 7 → r ≠ 8 → r ≠ 9 → r ≠ 10 → r ≠ 11 → r ≠ 12 → r ≠ 13 → s'.reg r = s.reg r)) ∨
    (HitsNomem s ∧ Ends vmImage s (.exit 2)) := by
    have hW : X86Env.W = 2 ^ 64 := rfl
    have hdsW : dsBase + dsCap * 16 ≤ X86Env.W := by
      simp only [Region.hi, hds_base, hds_eq] at hds_hi; omega
    have hdsCapW : dsCap * 32 < X86Env.W := by
      simp only [Region.hi, hds_eq] at hds_hi; omega
    obtain ⟨m', s', hreach, hrip', hk', hm', hwf', hr', hsp', hds', hctx', hoth', hregoth⟩ :=
      fn_ds_push_nogrow_spec hrip hregs hsp hr15 hr7 hr6 hctx hctx_base hctx_rw hctx_len
        hds hds_base hds_rw hne hstk hstkne_ds hstkne_ctx (by omega) hstk_hi hlt (by omega) hctxW hdsW
        (by omega) hmwf hret hc0 hc8 hc16
    have ⟨hbl', hpos'⟩ := reaches_bytesLt_len_pos hbl hpos hreach
    have hstk' := hoth' stk hstk hstkne_ds hstkne_ctx
    have hcs : ctxSize = 4096 := rfl
    have hc0' : leVal ((({ ctx with data := writeBytes ctx.data 8 (bytes8 (dsLen + 1)) }.data).drop 0).take 8) = dsReg.base := by
      dsimp only
      rw [writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inl (by decide)) (by omega), hc0, hds_base]
    have hc8' : leVal ((({ ctx with data := writeBytes ctx.data 8 (bytes8 (dsLen + 1)) }.data).drop 8).take 8) = dsLen + 1 := by
      dsimp only
      rw [leVal_writeBytes_same _ _ _ (by omega), Nat.mod_eq_of_lt (by omega)]
    have hc16' : leVal ((({ ctx with data := writeBytes ctx.data 8 (bytes8 (dsLen + 1)) }.data).drop 16).take 8) = dsCap := by
      dsimp only
      rw [writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inr (by rw [bytes8_length]; decide)) (by omega), hc16]
    have hrest : ∀ j, (j < 0 ∨ 24 ≤ j) → (writeBytes ctx.data 8 (bytes8 (dsLen + 1)))[j]? = ctx.data[j]? := by
      intro j hj
      apply writeBytes_getElem?_disjoint
      rw [bytes8_length]
      omega
    refine Or.inl ⟨{ dsReg with data := _ }, { ctx with data := writeBytes ctx.data 8 (bytes8 (dsLen + 1)) },
      stk, m', s', dsCap, hreach, hrip', hm', hwf', by rw [← hm']; exact hbl', by rw [← hm']; exact hpos', hr', oracleAdmits_of_reaches hor hreach,
      by rw [hk'], hsp', Nat.le_refl _, by omega, hds_pg, hds', hds_rw,
      by simp only [Region.hi, writeBytes_length, hds_eq] at hds_hi ⊢; exact hds_hi,
      by simp [Nat.sub_self], hctx', rfl, rfl, writeBytes_length _ _ _,
      hc0', hc8', hc16', hrest, hstk', rfl, rfl, rfl,
      fun _ _ hd => by rw [hds_base]; exact hd,
      fun u hu hd hc _ => hoth' u hu hd hc,
      fun r h0 h1 h2 h4 _ _ _ _ _ _ h12 h13 => hregoth r h0 h1 h2 h4 h12 h13⟩

private theorem call_ds_push_grow_oracle_spec {s : X86Vm.St} {sp retAddr ctxBase dsBase dsLen dsCap w p : Nat}
    {ctx dsReg stk : Region}
    (hrip : s.rip = SporosVmBin.fn_ds_push)
    (hregs : RegsOk s) (hor : OracleAdmits s)
    (hbl : BytesLt s.mem) (hpos : ∀ u ∈ s.mem, 0 < u.data.length)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr7 : s.reg 7 = some w) (hr6 : s.reg 6 = some p)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_rw : ctx.rw = true)
    (hctx_len : ctxSize ≤ ctx.data.length)
    (hds : dsReg ∈ s.mem) (hds_base : dsReg.base = dsBase) (hds_rw : dsReg.rw = true)
    (hds_hi : dsReg.hi ≤ 2 ^ 47)
    (hne : dsBase ≠ ctxBase)
    (hstk : stk ∈ s.mem) (hstkrw : stk.rw = true) (hstkne_ds : stk.base ≠ dsBase) (hstkne_ctx : stk.base ≠ ctxBase)
    (h16 : 16 ≤ sp) (hspW : sp < X86Env.W)
    (hstk_lo : stk.base ≤ sp - 16) (hstk_hi : sp + 8 ≤ stk.hi)
    (hle : dsLen ≤ dsCap) (hcap_pos : 0 < dsCap)
    (hds_eq : dsReg.data.length = dsCap * 16)
    (hds_pg : (dsCap * 16) % X86Env.page = 0)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W)
    (hmwf : s.mem.wf)
    (hret : rd64 s.mem sp = some retAddr)
    (hc0 : leVal ((ctx.data.drop 0).take 8) = dsBase)
    (hc8 : leVal ((ctx.data.drop 8).take 8) = dsLen)
    (hc16 : leVal ((ctx.data.drop 16).take 8) = dsCap)
    (hlt : ¬ dsLen < dsCap) :
    (∃ dsReg' ctx' stk' m' s' dsCap',
      Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.mem = m' ∧ m'.wf ∧ BytesLt m' ∧ (∀ u ∈ m', 0 < u.data.length) ∧
      RegsOk s' ∧ OracleAdmits s' ∧ s'.k.out = s.k.out ∧
      s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      dsCap ≤ dsCap' ∧ dsLen + 1 ≤ dsCap' ∧ (dsCap' * 16) % X86Env.page = 0 ∧
      dsReg' ∈ m' ∧ dsReg'.rw = true ∧ dsReg'.hi ≤ 2 ^ 47 ∧
      dsReg'.data = writeBytes (writeBytes (dsReg.data ++ List.replicate ((dsCap' - dsCap) * 16) 0)
        (dsLen * 16) (bytes8 w)) (dsLen * 16 + 8) (bytes8 p) ∧
      ctx' ∈ m' ∧ ctx'.base = ctx.base ∧ ctx'.rw = ctx.rw ∧ ctx'.data.length = ctx.data.length ∧
      leVal ((ctx'.data.drop 0).take 8) = dsReg'.base ∧
      leVal ((ctx'.data.drop 8).take 8) = dsLen + 1 ∧
      leVal ((ctx'.data.drop 16).take 8) = dsCap' ∧
      (∀ j, (j < 0 ∨ 24 ≤ j) → ctx'.data[j]? = ctx.data[j]?) ∧
      stk' ∈ m' ∧ stk'.base = stk.base ∧ stk'.rw = stk.rw ∧ stk'.data.length = stk.data.length ∧
      (∀ u ∈ s.mem, u.base ≠ dsBase → u.base ≠ dsReg'.base) ∧
      (∀ u ∈ s.mem, u.base ≠ dsBase → u.base ≠ ctxBase → u.base ≠ stk.base → u ∈ m') ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 4 → r ≠ 6 → r ≠ 7 → r ≠ 8 → r ≠ 9 → r ≠ 10 → r ≠ 11 → r ≠ 12 → r ≠ 13 → s'.reg r = s.reg r)) ∨
    (HitsNomem s ∧ Ends vmImage s (.exit 2)) := by
    have hW : X86Env.W = 2 ^ 64 := rfl
    have hdsCapW : dsCap * 32 < X86Env.W := by
      simp only [Region.hi, hds_eq] at hds_hi; omega
    have hge : dsCap ≤ dsLen := Nat.le_of_not_lt hlt
    have h01 : X86Vm.step vmImage s = .ok { s.set 12 (some w) with rip := 4201449 } := by
      rw [step_at_chunk 23 hrip rfl (by decide : (SporosVmBin.fn_ds_push, (.ofInt (.mov 12 7)), 3) ∈ SporosVmBin.vmTable_23),
        exec_mov hr7, hrip]; rfl
    let s1 := { s.set 12 (some w) with rip := 4201449 }
    have hr1 : RegsOk s1 := regsOk_set hregs (by decide) w
    have hs1_6 : s1.reg 6 = some p := by
      simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 12 ≠ 6)]; exact hr6
    have h02 : X86Vm.step vmImage s1 = .ok { s1.set 13 (some p) with rip := 4201452 } := by
      rw [step_at_chunk 23 rfl rfl (by decide : (4201449, (.ofInt (.mov 13 6)), 3) ∈ SporosVmBin.vmTable_23),
        exec_mov hs1_6]
    let s2 := { s1.set 13 (some p) with rip := 4201452 }
    have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) p
    have hs2_15 : s2.reg 15 = some ctxBase := by
      simp only [s2, s1, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 13 ≠ 15), List.getElem?_set_ne (by decide : 12 ≠ 15)]; exact hr15
    have hcs : ctxSize = 4096 := rfl
    have hea_c0 : ea ctxBase 0 = ctxBase := ea_add_eq (by omega)
    have hea_c8 : ea ctxBase 8 = ctxBase + 8 := ea_add_eq (by omega)
    have hea_c16 : ea ctxBase 16 = ctxBase + 16 := ea_add_eq (by omega)
    have hrd_c0 : rd64 s2.mem (ea ctxBase 0) = some dsBase := by
      rw [hea_c0]; exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
        (by simp only [Region.hi, hctx_base]; omega) (by simpa [hctx_base] using hc0)
    have hrd_c8 : rd64 s2.mem (ea ctxBase 8) = some dsLen := by
      rw [hea_c8]; exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
        (by simp only [Region.hi, hctx_base]; omega) (by rw [hctx_base, Nat.add_sub_cancel_left]; exact hc8)
    have hrd_c16 : rd64 s2.mem (ea ctxBase 16) = some dsCap := by
      rw [hea_c16]; exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
        (by simp only [Region.hi, hctx_base]; omega) (by rw [hctx_base, Nat.add_sub_cancel_left]; exact hc16)
    have h03 : X86Vm.step vmImage s2 = .ok { s2.set 0 (some dsLen) with rip := 4201456 } := by
      rw [step_at_chunk 23 rfl rfl (by decide : (4201452, (.ofInt (.ld 0 15 8)), 4) ∈ SporosVmBin.vmTable_23),
        exec_ld hs2_15 hrd_c8]
    let s3 := { s2.set 0 (some dsLen) with rip := 4201456 }
    have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) dsLen
    have hs3_15 : s3.reg 15 = some ctxBase := by
      simp only [s3, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 15)]; exact hs2_15
    have hrd_c16_3 : rd64 s3.mem (ea ctxBase 16) = some dsCap := hrd_c16
    have h04 : X86Vm.step vmImage s3 = .ok { s3.set 1 (some dsCap) with rip := 4201460 } := by
      rw [step_at_chunk 23 rfl rfl (by decide : (4201456, (.ofInt (.ld 1 15 16)), 4) ∈ SporosVmBin.vmTable_23),
        exec_ld hs3_15 hrd_c16_3]
    let s4 := { s3.set 1 (some dsCap) with rip := 4201460 }
    have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) dsCap
    have hs4_0 : s4.reg 0 = some dsLen := by
      simp only [s4, s3, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0)]
      rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
    have hs4_1 : s4.reg 1 = some dsCap := by
      simp only [s4, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
    have h05 : X86Vm.step vmImage s4 = .ok { s4 with fl := (X86Int.aluF .sub dsLen dsCap).2, rip := 4201463 } := by
      rw [step_at_chunk 23 rfl rfl (by decide : (4201460, (.ofInt (.cmp 0 1)), 3) ∈ SporosVmBin.vmTable_23),
        exec_cmp hs4_0 hs4_1]
    let s5 := { s4 with fl := (X86Int.aluF .sub dsLen dsCap).2, rip := 4201463 }
    have h06 : X86Vm.step vmImage s5 = .ok { s5 with rip := 4201465 } := by
      rw [step_at_chunk 23 rfl rfl (by decide : (4201463, (.ofInt (.jcc 2 4201502)), 2) ∈ SporosVmBin.vmTable_23),
        exec_jcc (cond_aluF_sub_b dsLen dsCap)]
      simp [s5, hlt]
    let s6 := { s5 with rip := 4201465 }
    have hs6_15 : s6.reg 15 = some ctxBase := by
      simp only [s6, s5, s4, s3, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 1 ≠ 15), List.getElem?_set_ne (by decide : 0 ≠ 15)]
      exact hs2_15
    have h07 : X86Vm.step vmImage s6 = .ok { s6.set 7 (some dsBase) with rip := 4201468 } := by
      rw [step_at_chunk 23 rfl rfl (by decide : (4201465, (.ofInt (.ld 7 15 0)), 3) ∈ SporosVmBin.vmTable_23),
        exec_ld hs6_15 hrd_c0]
    let s7 := { s6.set 7 (some dsBase) with rip := 4201468 }
    have hr7' : RegsOk s7 := regsOk_set hr4 (by decide) dsBase
    have hs7_1 : s7.reg 1 = some dsCap := by
      simp only [s7, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 7 ≠ 1)]
      exact hs4_1
    have h08 : X86Vm.step vmImage s7 = .ok { s7.set 6 (some dsCap) with rip := 4201471 } := by
      rw [step_at_chunk 23 rfl rfl (by decide : (4201468, (.ofInt (.mov 6 1)), 3) ∈ SporosVmBin.vmTable_23),
        exec_mov hs7_1]
    let s8 := { s7.set 6 (some dsCap) with rip := 4201471 }
    have hr8 : RegsOk s8 := regsOk_set hr7' (by decide) dsCap
    have hs8_6 : s8.reg 6 = some dsCap := by
      simp only [s8, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr7'.1]; decide)]
    have h09 : X86Vm.step vmImage s8 =
        .ok { s8.set 6 (some (dsCap * 16)) with fl := X86Int.Flags.undef, rip := 4201475 } := by
      rw [step_at_chunk 23 rfl rfl (by decide : (4201471, (.ofInt (.shlI 6 4)), 4) ∈ SporosVmBin.vmTable_23),
        exec_shlI_4_lt hs8_6 (by omega)]
    let s9 := { s8.set 6 (some (dsCap * 16)) with fl := X86Int.Flags.undef, rip := 4201475 }
    have hr9 : RegsOk s9 := regsOk_set hr8 (by decide) (dsCap * 16)
    have hs9_6 : s9.reg 6 = some (dsCap * 16) := by
      simp only [s9, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr8.1]; decide)]
    have h10 : X86Vm.step vmImage s9 = .ok { s9.set 2 (some (dsCap * 16)) with rip := 4201478 } := by
      rw [step_at_chunk 23 rfl rfl (by decide : (4201475, (.ofInt (.mov 2 6)), 3) ∈ SporosVmBin.vmTable_23),
        exec_mov hs9_6]
    let s10 := { s9.set 2 (some (dsCap * 16)) with rip := 4201478 }
    have hr10 : RegsOk s10 := regsOk_set hr9 (by decide) (dsCap * 16)
    have hs10_2 : s10.reg 2 = some (dsCap * 16) := by
      simp only [s10, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr9.1]; decide)]
    have h11 : X86Vm.step vmImage s10 =
        .ok { s10.set 2 (some (dsCap * 32)) with fl := X86Int.Flags.undef, rip := 4201482 } := by
      rw [step_at_chunk 23 rfl rfl (by decide : (4201478, (.ofInt (.shlI 2 1)), 4) ∈ SporosVmBin.vmTable_23),
        exec_shlI_1_lt hs10_2 (by omega : dsCap * 16 * 2 < X86Env.W)]
      simp [s10, show dsCap * 16 * 2 = dsCap * 32 by omega]
    let s11 := { s10.set 2 (some (dsCap * 32)) with fl := X86Int.Flags.undef, rip := 4201482 }
    have hr11 : RegsOk s11 := regsOk_set hr10 (by decide) (dsCap * 32)
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
    obtain ⟨_, m12, hwr12, _, _, hwf12, hstk12, hoth12⟩ :=
      wr64_region hmwf hstk hstkrw (by omega) (by omega : sp - 8 + 8 ≤ stk.hi) (v := 4201487)
    have hwr12' : wr64 s11.mem ((sp + X86Env.W - 8) % X86Env.W) (4201482 + 5) = some m12 := by
      simpa [s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, hea_sp] using hwr12
    have h12 : X86Vm.step vmImage s11 =
        .ok { s11.set 4 (some (sp - 8)) with mem := m12, rip := SporosVmBin.fn_mremap } := by
      rw [step_at_chunk 23 rfl rfl (by decide : (4201482, (.ofInt (.call 4201363)), 5) ∈ SporosVmBin.vmTable_23),
        exec_call hs11_4 hwr12']
      simp [s11, hea_sp, SporosVmBin.fn_mremap]
    let s12 := { s11.set 4 (some (sp - 8)) with mem := m12, rip := SporosVmBin.fn_mremap }
    have hr12 : RegsOk s12 := regsOk_set hr11 (by decide) (sp - 8)
    have hreach12 : Reaches vmImage s s12 :=
      Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
        (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step_trans h08
          (Reaches.step_trans h09 (Reaches.step_trans h10 (Reaches.step_trans h11 (Reaches.step h12)))))))))))
    have hds12 : dsReg ∈ s12.mem := hoth12 dsReg hds (by rw [hds_base]; exact Ne.symm hstkne_ds)
    have hds12_hi : dsBase < dsReg.hi := by simp only [Region.hi, hds_base, hds_eq]; omega
    obtain ⟨idx, hfind12⟩ := mem_find_of_mem s12.mem dsReg dsBase 0 hwf12.2 hds12 (by rw [hds_base]; exact Nat.le_refl _) hds12_hi
    have hs12_7 : s12.reg 7 = some dsBase := by
      simp only [s12, s11, s10, s9, s8, s7, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 4 ≠ 7), List.getElem?_set_ne (by decide : 2 ≠ 7),
        List.getElem?_set_ne (by decide : 6 ≠ 7)]
      rw [getElem?_set_getD_self (by rw [hr4.1]; decide)]
    have hs12_6 : s12.reg 6 = some (dsCap * 16) := by
      simp only [s12, s11, s10, s9, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 4 ≠ 6), List.getElem?_set_ne (by decide : 2 ≠ 6)]
      exact hs9_6
    have hs12_2 : s12.reg 2 = some (dsCap * 32) := by
      simp only [s12, s11, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 4 ≠ 2)]
      rw [getElem?_set_getD_self (by rw [hr10.1]; decide)]
    have hpg16 : X86Env.pages (dsCap * 16) = dsCap * 16 := pages_of_dvd hds_pg
    have hdvd32 : (dsCap * 32) % X86Env.page = 0 := by
      have hmul : dsCap * 32 = 2 * (dsCap * 16) := by omega
      rw [hmul, Nat.mul_mod, hds_pg]; simp
    have hpg32 : X86Env.pages (dsCap * 32) = dsCap * 32 := pages_of_dvd hdvd32
    rcases at_mremap_oracle_spec rfl hr12 (oracleAdmits_of_reaches hor hreach12)
        hs12_7 hs12_6 hs12_2 hfind12 hds_base (by rw [hds_eq, hpg16]) hds_rw
        (by rw [hds_eq, hpg32]; omega) with ⟨hnomem, hhit12⟩ | ⟨a, hat, ha_pg, ha_pos, ha_lim, hfree⟩
    · have hs12_k : s12.k = s.k := rfl
      rw [hs12_k] at hnomem
      have hits0 := hitsNomem_of_reaches hreach12 hhit12
      exact Or.inr ⟨hits0, fn_ds_push_nomem_spec hrip hregs hsp hr15 hr7 hr6 hctx hctx_base hctx_len
        hds hds_base hds_rw hstk hstkrw hstkne_ds h16 hspW hstk_lo (by omega) hge hle hcap_pos
        hds_eq hds_pg hdsCapW hctxW hmwf hc0 hc8 hc16 hnomem⟩
    · have hs12_k : s12.k = s.k := rfl
      rw [hs12_k] at hat
      rw [hpg32] at ha_lim hfree
      rw [Nat.zero_add] at hfind12 hfree
      have hget12 : m12[idx]? = some dsReg := by
        simpa using (mem_find_idx_lt s12.mem dsBase 0 idx dsReg hfind12).2.2
      change X86Env.Mem.free (m12.eraseIdx idx) a (dsCap * 32) = true at hfree
      have hstk12_b : { stk with data := writeBytes stk.data (sp - 8 - stk.base) (bytes8 4201487) }.base = stk.base := rfl
      have hstk12_hi : { stk with data := writeBytes stk.data (sp - 8 - stk.base) (bytes8 4201487) }.hi = stk.hi := by
        simp [Region.hi, writeBytes_length]
      obtain ⟨hastk, hfree_all⟩ := mremap_find_after_stack_write (n := dsCap * 32) hmwf hstk hstk12_b hstk12_hi
        hstkne_ds hstk12 hoth12 hget12 hds_base hfree (by omega) hpos
      have hane : a ≠ ctxBase := by
        rw [← hctx_base]
        exact mem_free_ne_base (by omega) (hpos ctx hctx) (hfree_all ctx hctx (by rw [hctx_base]; exact Ne.symm hne))
      obtain ⟨m', s', stk', hreach_g, hrip', hk', hm', hwf', hr', hsp', hds', hctx', hstk', hstk'_b, hstk'_rw, hstk'_len, _, hoth', hregoth⟩ :=
        fn_ds_push_grow_spec hrip hregs hsp hr15 hr7 hr6 hctx hctx_base hctx_rw hctx_len
          hds hds_base hds_rw hne hane hastk hstk hstkrw hstkne_ds hstkne_ctx (by omega) hspW
          (by omega) hstk_hi hge hle hcap_pos hds_eq hds_pg hdsCapW hctxW hmwf hpos hret
          hc0 hc8 hc16 hat ha_pg ha_pos ha_lim hfree_all
      have ⟨hbl', hpos'⟩ := reaches_bytesLt_len_pos hbl hpos hreach_g
      let ctxData' := writeBytes (writeBytes (writeBytes ctx.data 0 (bytes8 a)) 16 (bytes8 (dsCap * 2))) 8 (bytes8 (dsLen + 1))
      let dsData' := writeBytes (writeBytes (dsReg.data ++ List.replicate (dsCap * 16) 0) (dsLen * 16) (bytes8 w))
        (dsLen * 16 + 8) (bytes8 p)
      have hcs' : ({ ctx with data := ctxData' } : Region).data.length = ctx.data.length := by
        dsimp only [ctxData']; rw [writeBytes_length, writeBytes_length, writeBytes_length]
      have hc0' : leVal (({ ctx with data := ctxData' }.data.drop 0).take 8) = a := by
        dsimp only [ctxData']
        rw [writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inl (by decide)) (by rw [writeBytes_length, writeBytes_length]; omega),
            writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inl (by decide)) (by rw [writeBytes_length]; omega),
            leVal_writeBytes_same _ _ _ (by omega), Nat.mod_eq_of_lt (by omega)]
      have hc8' : leVal (({ ctx with data := ctxData' }.data.drop 8).take 8) = dsLen + 1 := by
        dsimp only [ctxData']
        rw [leVal_writeBytes_same _ _ _ (by rw [writeBytes_length, writeBytes_length]; omega),
            Nat.mod_eq_of_lt (by omega)]
      have hc16' : leVal (({ ctx with data := ctxData' }.data.drop 16).take 8) = dsCap * 2 := by
        dsimp only [ctxData']
        rw [writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inr (by rw [bytes8_length]; decide)) (by rw [writeBytes_length, writeBytes_length]; omega),
            leVal_writeBytes_same _ _ _ (by rw [writeBytes_length]; omega), Nat.mod_eq_of_lt (by omega)]
      have hrest : ∀ j, (j < 0 ∨ 24 ≤ j) → ({ ctx with data := ctxData' } : Region).data[j]? = ctx.data[j]? := by
        intro j hj
        dsimp only [ctxData']
        rw [writeBytes_getElem?_disjoint _ _ _ _ (by rw [bytes8_length]; omega),
            writeBytes_getElem?_disjoint _ _ _ _ (by rw [bytes8_length]; omega),
            writeBytes_getElem?_disjoint _ _ _ _ (by rw [bytes8_length]; omega)]
      have hk_out : s'.k.out = s.k.out := by
        rw [hk']; cases horac : s.k.oracle <;> simp [X86Env.Kern.next, horac]
      have hdvd_cap : (dsCap * 2 * 16) % X86Env.page = 0 := by
        rw [show dsCap * 2 * 16 = dsCap * 32 from by omega]; exact hdvd32
      have hds_hi' : (⟨a, dsData', true⟩ : Region).hi ≤ 2 ^ 47 := by
        simp [dsData', Region.hi, writeBytes_length, List.length_append, List.length_replicate, hds_eq]; omega
      have hds_data' : dsData' = writeBytes (writeBytes (dsReg.data ++ List.replicate ((dsCap * 2 - dsCap) * 16) 0) (dsLen * 16) (bytes8 w))
          (dsLen * 16 + 8) (bytes8 p) := by
        rw [show (dsCap * 2 - dsCap) * 16 = dsCap * 16 by omega]
      refine Or.inl ⟨⟨a, dsData', true⟩, { ctx with data := ctxData' }, stk', m', s', dsCap * 2,
        hreach_g, hrip', hm', hwf', by rw [← hm']; exact hbl', by rw [← hm']; exact hpos', hr',
        oracleAdmits_of_reaches hor hreach_g, hk_out, hsp', by omega, by omega, hdvd_cap, hds', rfl,
        hds_hi', hds_data', hctx', rfl, rfl, ?_⟩
      refine ⟨?_, ?_, ?_, ?_, ?_, hstk', hstk'_b, by rw [hstk'_rw, hstkrw], hstk'_len,
        fun u hu hd => Ne.symm (mem_free_ne_base (by omega) (hpos u hu) (hfree_all u hu hd)), hoth', hregoth⟩
      · exact hcs'
      · exact hc0'
      · exact hc8'
      · exact hc16'
      · exact hrest

theorem call_ds_push_oracle_spec {s : X86Vm.St} {sp retAddr ctxBase dsBase dsLen dsCap w p : Nat}
    {ctx dsReg stk : Region}
    (hrip : s.rip = SporosVmBin.fn_ds_push)
    (hregs : RegsOk s) (hor : OracleAdmits s)
    (hbl : BytesLt s.mem) (hpos : ∀ u ∈ s.mem, 0 < u.data.length)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr7 : s.reg 7 = some w) (hr6 : s.reg 6 = some p)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_rw : ctx.rw = true)
    (hctx_len : ctxSize ≤ ctx.data.length)
    (hds : dsReg ∈ s.mem) (hds_base : dsReg.base = dsBase) (hds_rw : dsReg.rw = true)
    (hds_hi : dsReg.hi ≤ 2 ^ 47)
    (hne : dsBase ≠ ctxBase)
    (hstk : stk ∈ s.mem) (hstkrw : stk.rw = true) (hstkne_ds : stk.base ≠ dsBase) (hstkne_ctx : stk.base ≠ ctxBase)
    (h16 : 16 ≤ sp) (hspW : sp < X86Env.W)
    (hstk_lo : stk.base ≤ sp - 16) (hstk_hi : sp + 8 ≤ stk.hi)
    (hle : dsLen ≤ dsCap) (hcap_pos : 0 < dsCap)
    (hds_eq : dsReg.data.length = dsCap * 16)
    (hds_pg : (dsCap * 16) % X86Env.page = 0)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W)
    (hmwf : s.mem.wf)
    (hret : rd64 s.mem sp = some retAddr)
    (hc0 : leVal ((ctx.data.drop 0).take 8) = dsBase)
    (hc8 : leVal ((ctx.data.drop 8).take 8) = dsLen)
    (hc16 : leVal ((ctx.data.drop 16).take 8) = dsCap) :
    (∃ dsReg' ctx' stk' m' s' dsCap',
      Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.mem = m' ∧ m'.wf ∧ BytesLt m' ∧ (∀ u ∈ m', 0 < u.data.length) ∧
      RegsOk s' ∧ OracleAdmits s' ∧ s'.k.out = s.k.out ∧
      s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      dsCap ≤ dsCap' ∧ dsLen + 1 ≤ dsCap' ∧ (dsCap' * 16) % X86Env.page = 0 ∧
      dsReg' ∈ m' ∧ dsReg'.rw = true ∧ dsReg'.hi ≤ 2 ^ 47 ∧
      dsReg'.data = writeBytes (writeBytes (dsReg.data ++ List.replicate ((dsCap' - dsCap) * 16) 0)
        (dsLen * 16) (bytes8 w)) (dsLen * 16 + 8) (bytes8 p) ∧
      ctx' ∈ m' ∧ ctx'.base = ctx.base ∧ ctx'.rw = ctx.rw ∧ ctx'.data.length = ctx.data.length ∧
      leVal ((ctx'.data.drop 0).take 8) = dsReg'.base ∧
      leVal ((ctx'.data.drop 8).take 8) = dsLen + 1 ∧
      leVal ((ctx'.data.drop 16).take 8) = dsCap' ∧
      (∀ j, (j < 0 ∨ 24 ≤ j) → ctx'.data[j]? = ctx.data[j]?) ∧
      stk' ∈ m' ∧ stk'.base = stk.base ∧ stk'.rw = stk.rw ∧ stk'.data.length = stk.data.length ∧
      (∀ u ∈ s.mem, u.base ≠ dsBase → u.base ≠ dsReg'.base) ∧
      (∀ u ∈ s.mem, u.base ≠ dsBase → u.base ≠ ctxBase → u.base ≠ stk.base → u ∈ m') ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 4 → r ≠ 6 → r ≠ 7 → r ≠ 8 → r ≠ 9 → r ≠ 10 → r ≠ 11 → r ≠ 12 → r ≠ 13 → s'.reg r = s.reg r)) ∨
    (HitsNomem s ∧ Ends vmImage s (.exit 2)) := by
  by_cases hlt : dsLen < dsCap
  · exact call_ds_push_nogrow_oracle_spec hrip hregs hor hbl hpos hsp hr15 hr7 hr6 hctx hctx_base
      hctx_rw hctx_len hds hds_base hds_rw hds_hi hne hstk hstkrw hstkne_ds hstkne_ctx h16 hspW
      hstk_lo hstk_hi hle hcap_pos hds_eq hds_pg hctxW hmwf hret hc0 hc8 hc16 hlt
  · exact call_ds_push_grow_oracle_spec hrip hregs hor hbl hpos hsp hr15 hr7 hr6 hctx hctx_base
      hctx_rw hctx_len hds hds_base hds_rw hds_hi hne hstk hstkrw hstkne_ds hstkne_ctx h16 hspW
      hstk_lo hstk_hi hle hcap_pos hds_eq hds_pg hctxW hmwf hret hc0 hc8 hc16 hlt

private theorem call_aux_push_nogrow_oracle_spec {s : X86Vm.St} {sp retAddr ctxBase auxBase auxLen auxCap w p : Nat}
    {ctx auxReg stk : Region}
    (hrip : s.rip = SporosVmBin.fn_aux_push)
    (hregs : RegsOk s) (hor : OracleAdmits s)
    (hbl : BytesLt s.mem) (hpos : ∀ u ∈ s.mem, 0 < u.data.length)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr7 : s.reg 7 = some w) (hr6 : s.reg 6 = some p)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_rw : ctx.rw = true)
    (hctx_len : ctxSize ≤ ctx.data.length)
    (haux : auxReg ∈ s.mem) (haux_base : auxReg.base = auxBase) (haux_rw : auxReg.rw = true)
    (haux_hi : auxReg.hi ≤ 2 ^ 47)
    (hne : auxBase ≠ ctxBase)
    (hstk : stk ∈ s.mem) (_hstkrw : stk.rw = true) (hstkne_aux : stk.base ≠ auxBase) (hstkne_ctx : stk.base ≠ ctxBase)
    (h16 : 16 ≤ sp) (_hspW : sp < X86Env.W)
    (hstk_lo : stk.base ≤ sp - 16) (hstk_hi : sp + 8 ≤ stk.hi)
    (_hle : auxLen ≤ auxCap) (hcap_pos : 0 < auxCap)
    (haux_eq : auxReg.data.length = auxCap * 16)
    (haux_pg : (auxCap * 16) % X86Env.page = 0)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W)
    (hmwf : s.mem.wf)
    (hret : rd64 s.mem sp = some retAddr)
    (hc24 : leVal ((ctx.data.drop 24).take 8) = auxBase)
    (hc32 : leVal ((ctx.data.drop 32).take 8) = auxLen)
    (hc40 : leVal ((ctx.data.drop 40).take 8) = auxCap)
    (hlt : auxLen < auxCap) :
    (∃ auxReg' ctx' stk' m' s' auxCap',
      Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.mem = m' ∧ m'.wf ∧ BytesLt m' ∧ (∀ u ∈ m', 0 < u.data.length) ∧
      RegsOk s' ∧ OracleAdmits s' ∧ s'.k.out = s.k.out ∧
      s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      auxCap ≤ auxCap' ∧ auxLen + 1 ≤ auxCap' ∧ (auxCap' * 16) % X86Env.page = 0 ∧
      auxReg' ∈ m' ∧ auxReg'.rw = true ∧ auxReg'.hi ≤ 2 ^ 47 ∧
      auxReg'.data = writeBytes (writeBytes (auxReg.data ++ List.replicate ((auxCap' - auxCap) * 16) 0)
        (auxLen * 16) (bytes8 w)) (auxLen * 16 + 8) (bytes8 p) ∧
      ctx' ∈ m' ∧ ctx'.base = ctx.base ∧ ctx'.rw = ctx.rw ∧ ctx'.data.length = ctx.data.length ∧
      leVal ((ctx'.data.drop 24).take 8) = auxReg'.base ∧
      leVal ((ctx'.data.drop 32).take 8) = auxLen + 1 ∧
      leVal ((ctx'.data.drop 40).take 8) = auxCap' ∧
      (∀ j, (j < 24 ∨ 48 ≤ j) → ctx'.data[j]? = ctx.data[j]?) ∧
      stk' ∈ m' ∧ stk'.base = stk.base ∧ stk'.rw = stk.rw ∧ stk'.data.length = stk.data.length ∧
      (∀ u ∈ s.mem, u.base ≠ auxBase → u.base ≠ auxReg'.base) ∧
      (∀ u ∈ s.mem, u.base ≠ auxBase → u.base ≠ ctxBase → u.base ≠ stk.base → u ∈ m') ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 4 → r ≠ 6 → r ≠ 7 → r ≠ 8 → r ≠ 9 → r ≠ 10 → r ≠ 11 → r ≠ 12 → r ≠ 13 → s'.reg r = s.reg r)) ∨
    (HitsNomem s ∧ Ends vmImage s (.exit 2)) := by
    have hW : X86Env.W = 2 ^ 64 := rfl
    have hauxW : auxBase + auxCap * 16 ≤ X86Env.W := by
      simp only [Region.hi, haux_base, haux_eq] at haux_hi; omega
    have hauxCapW : auxCap * 32 < X86Env.W := by
      simp only [Region.hi, haux_eq] at haux_hi; omega
    obtain ⟨m', s', hreach, hrip', hk', hm', hwf', hr', hsp', haux', hctx', hoth', hregoth⟩ :=
      fn_aux_push_nogrow_spec hrip hregs hsp hr15 hr7 hr6 hctx hctx_base hctx_rw hctx_len
        haux haux_base haux_rw hne hstk hstkne_aux hstkne_ctx (by omega) hstk_hi hlt (by omega) hctxW hauxW
        (by omega) hmwf hret hc24 hc32 hc40
    have ⟨hbl', hpos'⟩ := reaches_bytesLt_len_pos hbl hpos hreach
    have hstk' := hoth' stk hstk hstkne_aux hstkne_ctx
    have hcs : ctxSize = 4096 := rfl
    have hc24' : leVal ((({ ctx with data := writeBytes ctx.data 32 (bytes8 (auxLen + 1)) }.data).drop 24).take 8) = auxReg.base := by
      dsimp only
      rw [writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inl (by decide)) (by omega), hc24, haux_base]
    have hc32' : leVal ((({ ctx with data := writeBytes ctx.data 32 (bytes8 (auxLen + 1)) }.data).drop 32).take 8) = auxLen + 1 := by
      dsimp only
      rw [leVal_writeBytes_same _ _ _ (by omega), Nat.mod_eq_of_lt (by omega)]
    have hc40' : leVal ((({ ctx with data := writeBytes ctx.data 32 (bytes8 (auxLen + 1)) }.data).drop 40).take 8) = auxCap := by
      dsimp only
      rw [writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inr (by rw [bytes8_length]; decide)) (by omega), hc40]
    have hrest : ∀ j, (j < 24 ∨ 48 ≤ j) → (writeBytes ctx.data 32 (bytes8 (auxLen + 1)))[j]? = ctx.data[j]? := by
      intro j hj
      apply writeBytes_getElem?_disjoint
      rw [bytes8_length]
      omega
    refine Or.inl ⟨{ auxReg with data := _ }, { ctx with data := writeBytes ctx.data 32 (bytes8 (auxLen + 1)) },
      stk, m', s', auxCap, hreach, hrip', hm', hwf', by rw [← hm']; exact hbl', by rw [← hm']; exact hpos', hr', oracleAdmits_of_reaches hor hreach,
      by rw [hk'], hsp', Nat.le_refl _, by omega, haux_pg, haux', haux_rw,
      by simp only [Region.hi, writeBytes_length, haux_eq] at haux_hi ⊢; exact haux_hi,
      by simp [Nat.sub_self], hctx', rfl, rfl, writeBytes_length _ _ _,
      hc24', hc32', hc40', hrest, hstk', rfl, rfl, rfl,
      fun _ _ ha => by rw [haux_base]; exact ha,
      fun u hu ha hc _ => hoth' u hu ha hc,
      fun r h0 h1 h2 h4 _ _ _ _ _ _ h12 h13 => hregoth r h0 h1 h2 h4 h12 h13⟩

private theorem call_aux_push_grow_oracle_spec {s : X86Vm.St} {sp retAddr ctxBase auxBase auxLen auxCap w p : Nat}
    {ctx auxReg stk : Region}
    (hrip : s.rip = SporosVmBin.fn_aux_push)
    (hregs : RegsOk s) (hor : OracleAdmits s)
    (hbl : BytesLt s.mem) (hpos : ∀ u ∈ s.mem, 0 < u.data.length)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr7 : s.reg 7 = some w) (hr6 : s.reg 6 = some p)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_rw : ctx.rw = true)
    (hctx_len : ctxSize ≤ ctx.data.length)
    (haux : auxReg ∈ s.mem) (haux_base : auxReg.base = auxBase) (haux_rw : auxReg.rw = true)
    (haux_hi : auxReg.hi ≤ 2 ^ 47)
    (hne : auxBase ≠ ctxBase)
    (hstk : stk ∈ s.mem) (hstkrw : stk.rw = true) (hstkne_aux : stk.base ≠ auxBase) (hstkne_ctx : stk.base ≠ ctxBase)
    (h16 : 16 ≤ sp) (hspW : sp < X86Env.W)
    (hstk_lo : stk.base ≤ sp - 16) (hstk_hi : sp + 8 ≤ stk.hi)
    (hle : auxLen ≤ auxCap) (hcap_pos : 0 < auxCap)
    (haux_eq : auxReg.data.length = auxCap * 16)
    (haux_pg : (auxCap * 16) % X86Env.page = 0)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W)
    (hmwf : s.mem.wf)
    (hret : rd64 s.mem sp = some retAddr)
    (hc24 : leVal ((ctx.data.drop 24).take 8) = auxBase)
    (hc32 : leVal ((ctx.data.drop 32).take 8) = auxLen)
    (hc40 : leVal ((ctx.data.drop 40).take 8) = auxCap)
    (hlt : ¬ auxLen < auxCap) :
    (∃ auxReg' ctx' stk' m' s' auxCap',
      Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.mem = m' ∧ m'.wf ∧ BytesLt m' ∧ (∀ u ∈ m', 0 < u.data.length) ∧
      RegsOk s' ∧ OracleAdmits s' ∧ s'.k.out = s.k.out ∧
      s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      auxCap ≤ auxCap' ∧ auxLen + 1 ≤ auxCap' ∧ (auxCap' * 16) % X86Env.page = 0 ∧
      auxReg' ∈ m' ∧ auxReg'.rw = true ∧ auxReg'.hi ≤ 2 ^ 47 ∧
      auxReg'.data = writeBytes (writeBytes (auxReg.data ++ List.replicate ((auxCap' - auxCap) * 16) 0)
        (auxLen * 16) (bytes8 w)) (auxLen * 16 + 8) (bytes8 p) ∧
      ctx' ∈ m' ∧ ctx'.base = ctx.base ∧ ctx'.rw = ctx.rw ∧ ctx'.data.length = ctx.data.length ∧
      leVal ((ctx'.data.drop 24).take 8) = auxReg'.base ∧
      leVal ((ctx'.data.drop 32).take 8) = auxLen + 1 ∧
      leVal ((ctx'.data.drop 40).take 8) = auxCap' ∧
      (∀ j, (j < 24 ∨ 48 ≤ j) → ctx'.data[j]? = ctx.data[j]?) ∧
      stk' ∈ m' ∧ stk'.base = stk.base ∧ stk'.rw = stk.rw ∧ stk'.data.length = stk.data.length ∧
      (∀ u ∈ s.mem, u.base ≠ auxBase → u.base ≠ auxReg'.base) ∧
      (∀ u ∈ s.mem, u.base ≠ auxBase → u.base ≠ ctxBase → u.base ≠ stk.base → u ∈ m') ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 4 → r ≠ 6 → r ≠ 7 → r ≠ 8 → r ≠ 9 → r ≠ 10 → r ≠ 11 → r ≠ 12 → r ≠ 13 → s'.reg r = s.reg r)) ∨
    (HitsNomem s ∧ Ends vmImage s (.exit 2)) := by
    have hW : X86Env.W = 2 ^ 64 := rfl
    have hauxCapW : auxCap * 32 < X86Env.W := by
      simp only [Region.hi, haux_eq] at haux_hi; omega
    have hge : auxCap ≤ auxLen := Nat.le_of_not_lt hlt
    have h01 : X86Vm.step vmImage s = .ok { s.set 12 (some w) with rip := 4201537 } := by
      rw [step_at_chunk 24 hrip rfl (by decide : (SporosVmBin.fn_aux_push, (.ofInt (.mov 12 7)), 3) ∈ SporosVmBin.vmTable_24),
        exec_mov hr7, hrip]; rfl
    let s1 := { s.set 12 (some w) with rip := 4201537 }
    have hr1 : RegsOk s1 := regsOk_set hregs (by decide) w
    have hs1_6 : s1.reg 6 = some p := by
      simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 12 ≠ 6)]; exact hr6
    have h02 : X86Vm.step vmImage s1 = .ok { s1.set 13 (some p) with rip := 4201540 } := by
      rw [step_at_chunk 24 rfl rfl (by decide : (4201537, (.ofInt (.mov 13 6)), 3) ∈ SporosVmBin.vmTable_24),
        exec_mov hs1_6]
    let s2 := { s1.set 13 (some p) with rip := 4201540 }
    have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) p
    have hs2_15 : s2.reg 15 = some ctxBase := by
      simp only [s2, s1, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 13 ≠ 15), List.getElem?_set_ne (by decide : 12 ≠ 15)]; exact hr15
    have hcs : ctxSize = 4096 := rfl
    have hea_c24 : ea ctxBase 24 = ctxBase + 24 := ea_add_eq (by omega)
    have hea_c32 : ea ctxBase 32 = ctxBase + 32 := ea_add_eq (by omega)
    have hea_c40 : ea ctxBase 40 = ctxBase + 40 := ea_add_eq (by omega)
    have hrd_c24 : rd64 s2.mem (ea ctxBase 24) = some auxBase := by
      rw [hea_c24]; exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
        (by simp only [Region.hi, hctx_base]; omega) (by rw [hctx_base, Nat.add_sub_cancel_left]; exact hc24)
    have hrd_c32 : rd64 s2.mem (ea ctxBase 32) = some auxLen := by
      rw [hea_c32]; exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
        (by simp only [Region.hi, hctx_base]; omega) (by rw [hctx_base, Nat.add_sub_cancel_left]; exact hc32)
    have hrd_c40 : rd64 s2.mem (ea ctxBase 40) = some auxCap := by
      rw [hea_c40]; exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
        (by simp only [Region.hi, hctx_base]; omega) (by rw [hctx_base, Nat.add_sub_cancel_left]; exact hc40)
    have h03 : X86Vm.step vmImage s2 = .ok { s2.set 0 (some auxLen) with rip := 4201544 } := by
      rw [step_at_chunk 24 rfl rfl (by decide : (4201540, (.ofInt (.ld 0 15 32)), 4) ∈ SporosVmBin.vmTable_24),
        exec_ld hs2_15 hrd_c32]
    let s3 := { s2.set 0 (some auxLen) with rip := 4201544 }
    have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) auxLen
    have hs3_15 : s3.reg 15 = some ctxBase := by
      simp only [s3, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 15)]; exact hs2_15
    have h04 : X86Vm.step vmImage s3 = .ok { s3.set 1 (some auxCap) with rip := 4201548 } := by
      rw [step_at_chunk 24 rfl rfl (by decide : (4201544, (.ofInt (.ld 1 15 40)), 4) ∈ SporosVmBin.vmTable_24),
        exec_ld hs3_15 hrd_c40]
    let s4 := { s3.set 1 (some auxCap) with rip := 4201548 }
    have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) auxCap
    have hs4_0 : s4.reg 0 = some auxLen := by
      simp only [s4, s3, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0)]
      rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
    have hs4_1 : s4.reg 1 = some auxCap := by
      simp only [s4, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
    have h05 : X86Vm.step vmImage s4 = .ok { s4 with fl := (X86Int.aluF .sub auxLen auxCap).2, rip := 4201551 } := by
      rw [step_at_chunk 24 rfl rfl (by decide : (4201548, (.ofInt (.cmp 0 1)), 3) ∈ SporosVmBin.vmTable_24),
        exec_cmp hs4_0 hs4_1]
    let s5 := { s4 with fl := (X86Int.aluF .sub auxLen auxCap).2, rip := 4201551 }
    have h06 : X86Vm.step vmImage s5 = .ok { s5 with rip := 4201553 } := by
      rw [step_at_chunk 24 rfl rfl (by decide : (4201551, (.ofInt (.jcc 2 4201592)), 2) ∈ SporosVmBin.vmTable_24),
        exec_jcc (cond_aluF_sub_b auxLen auxCap)]
      simp [s5, hlt]
    let s6 := { s5 with rip := 4201553 }
    have hs6_15 : s6.reg 15 = some ctxBase := by
      simp only [s6, s5, s4, s3, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 1 ≠ 15), List.getElem?_set_ne (by decide : 0 ≠ 15)]
      exact hs2_15
    have h07 : X86Vm.step vmImage s6 = .ok { s6.set 7 (some auxBase) with rip := 4201557 } := by
      rw [step_at_chunk 24 rfl rfl (by decide : (4201553, (.ofInt (.ld 7 15 24)), 4) ∈ SporosVmBin.vmTable_24),
        exec_ld hs6_15 hrd_c24]
    let s7 := { s6.set 7 (some auxBase) with rip := 4201557 }
    have hr7' : RegsOk s7 := regsOk_set hr4 (by decide) auxBase
    have hs7_1 : s7.reg 1 = some auxCap := by
      simp only [s7, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 7 ≠ 1)]
      exact hs4_1
    have h08 : X86Vm.step vmImage s7 = .ok { s7.set 6 (some auxCap) with rip := 4201560 } := by
      rw [step_at_chunk 24 rfl rfl (by decide : (4201557, (.ofInt (.mov 6 1)), 3) ∈ SporosVmBin.vmTable_24),
        exec_mov hs7_1]
    let s8 := { s7.set 6 (some auxCap) with rip := 4201560 }
    have hr8 : RegsOk s8 := regsOk_set hr7' (by decide) auxCap
    have hs8_6 : s8.reg 6 = some auxCap := by
      simp only [s8, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr7'.1]; decide)]
    have h09 : X86Vm.step vmImage s8 =
        .ok { s8.set 6 (some (auxCap * 16)) with fl := X86Int.Flags.undef, rip := 4201564 } := by
      rw [step_at_chunk 24 rfl rfl (by decide : (4201560, (.ofInt (.shlI 6 4)), 4) ∈ SporosVmBin.vmTable_24),
        exec_shlI_4_lt hs8_6 (by omega)]
    let s9 := { s8.set 6 (some (auxCap * 16)) with fl := X86Int.Flags.undef, rip := 4201564 }
    have hr9 : RegsOk s9 := regsOk_set hr8 (by decide) (auxCap * 16)
    have hs9_6 : s9.reg 6 = some (auxCap * 16) := by
      simp only [s9, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr8.1]; decide)]
    have h10 : X86Vm.step vmImage s9 = .ok { s9.set 2 (some (auxCap * 16)) with rip := 4201567 } := by
      rw [step_at_chunk 24 rfl rfl (by decide : (4201564, (.ofInt (.mov 2 6)), 3) ∈ SporosVmBin.vmTable_24),
        exec_mov hs9_6]
    let s10 := { s9.set 2 (some (auxCap * 16)) with rip := 4201567 }
    have hr10 : RegsOk s10 := regsOk_set hr9 (by decide) (auxCap * 16)
    have hs10_2 : s10.reg 2 = some (auxCap * 16) := by
      simp only [s10, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr9.1]; decide)]
    have h11 : X86Vm.step vmImage s10 =
        .ok { s10.set 2 (some (auxCap * 32)) with fl := X86Int.Flags.undef, rip := 4201571 } := by
      rw [step_at_chunk 24 rfl rfl (by decide : (4201567, (.ofInt (.shlI 2 1)), 4) ∈ SporosVmBin.vmTable_24),
        exec_shlI_1_lt hs10_2 (by omega : auxCap * 16 * 2 < X86Env.W)]
      simp [s10, show auxCap * 16 * 2 = auxCap * 32 by omega]
    let s11 := { s10.set 2 (some (auxCap * 32)) with fl := X86Int.Flags.undef, rip := 4201571 }
    have hr11 : RegsOk s11 := regsOk_set hr10 (by decide) (auxCap * 32)
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
    obtain ⟨_, m12, hwr12, _, _, hwf12, hstk12, hoth12⟩ :=
      wr64_region hmwf hstk hstkrw (by omega) (by omega : sp - 8 + 8 ≤ stk.hi) (v := 4201576)
    have hwr12' : wr64 s11.mem ((sp + X86Env.W - 8) % X86Env.W) (4201571 + 5) = some m12 := by
      simpa [s11, s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, hea_sp] using hwr12
    have h12 : X86Vm.step vmImage s11 =
        .ok { s11.set 4 (some (sp - 8)) with mem := m12, rip := SporosVmBin.fn_mremap } := by
      rw [step_at_chunk 24 rfl rfl (by decide : (4201571, (.ofInt (.call 4201363)), 5) ∈ SporosVmBin.vmTable_24),
        exec_call hs11_4 hwr12']
      simp [s11, hea_sp, SporosVmBin.fn_mremap]
    let s12 := { s11.set 4 (some (sp - 8)) with mem := m12, rip := SporosVmBin.fn_mremap }
    have hr12 : RegsOk s12 := regsOk_set hr11 (by decide) (sp - 8)
    have hreach12 : Reaches vmImage s s12 :=
      Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
        (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step_trans h08
          (Reaches.step_trans h09 (Reaches.step_trans h10 (Reaches.step_trans h11 (Reaches.step h12)))))))))))
    have haux12 : auxReg ∈ s12.mem := hoth12 auxReg haux (by rw [haux_base]; exact Ne.symm hstkne_aux)
    have haux12_hi : auxBase < auxReg.hi := by simp only [Region.hi, haux_base, haux_eq]; omega
    obtain ⟨idx, hfind12⟩ := mem_find_of_mem s12.mem auxReg auxBase 0 hwf12.2 haux12 (by rw [haux_base]; exact Nat.le_refl _) haux12_hi
    have hs12_7 : s12.reg 7 = some auxBase := by
      simp only [s12, s11, s10, s9, s8, s7, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 4 ≠ 7), List.getElem?_set_ne (by decide : 2 ≠ 7),
        List.getElem?_set_ne (by decide : 6 ≠ 7)]
      rw [getElem?_set_getD_self (by rw [hr4.1]; decide)]
    have hs12_6 : s12.reg 6 = some (auxCap * 16) := by
      simp only [s12, s11, s10, s9, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 4 ≠ 6), List.getElem?_set_ne (by decide : 2 ≠ 6)]
      exact hs9_6
    have hs12_2 : s12.reg 2 = some (auxCap * 32) := by
      simp only [s12, s11, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 4 ≠ 2)]
      rw [getElem?_set_getD_self (by rw [hr10.1]; decide)]
    have hpg16 : X86Env.pages (auxCap * 16) = auxCap * 16 := pages_of_dvd haux_pg
    have hdvd32 : (auxCap * 32) % X86Env.page = 0 := by
      have hmul : auxCap * 32 = 2 * (auxCap * 16) := by omega
      rw [hmul, Nat.mul_mod, haux_pg]; simp
    have hpg32 : X86Env.pages (auxCap * 32) = auxCap * 32 := pages_of_dvd hdvd32
    rcases at_mremap_oracle_spec rfl hr12 (oracleAdmits_of_reaches hor hreach12)
        hs12_7 hs12_6 hs12_2 hfind12 haux_base (by rw [haux_eq, hpg16]) haux_rw
        (by rw [haux_eq, hpg32]; omega) with ⟨hnomem, hhit12⟩ | ⟨a, hat, ha_pg, ha_pos, ha_lim, hfree⟩
    · have hs12_k : s12.k = s.k := rfl
      rw [hs12_k] at hnomem
      exact Or.inr ⟨hitsNomem_of_reaches hreach12 hhit12,
        fn_aux_push_nomem_spec hrip hregs hsp hr15 hr7 hr6 hctx hctx_base hctx_len
          haux haux_base haux_rw hstk hstkrw hstkne_aux h16 hspW hstk_lo (by omega) hge hle hcap_pos
          haux_eq haux_pg hauxCapW hctxW hmwf hc24 hc32 hc40 hnomem⟩
    · have hs12_k : s12.k = s.k := rfl
      rw [hs12_k] at hat
      rw [hpg32] at ha_lim hfree
      rw [Nat.zero_add] at hfind12 hfree
      have hget12 : m12[idx]? = some auxReg := by
        simpa using (mem_find_idx_lt s12.mem auxBase 0 idx auxReg hfind12).2.2
      change X86Env.Mem.free (m12.eraseIdx idx) a (auxCap * 32) = true at hfree
      have hstk12_b : { stk with data := writeBytes stk.data (sp - 8 - stk.base) (bytes8 4201576) }.base = stk.base := rfl
      have hstk12_hi : { stk with data := writeBytes stk.data (sp - 8 - stk.base) (bytes8 4201576) }.hi = stk.hi := by
        simp [Region.hi, writeBytes_length]
      obtain ⟨hastk, hfree_all⟩ := mremap_find_after_stack_write (n := auxCap * 32) hmwf hstk hstk12_b hstk12_hi
        hstkne_aux hstk12 hoth12 hget12 haux_base hfree (by omega) hpos
      have hane : a ≠ ctxBase := by
        rw [← hctx_base]
        exact mem_free_ne_base (by omega) (hpos ctx hctx) (hfree_all ctx hctx (by rw [hctx_base]; exact Ne.symm hne))
      obtain ⟨m', s', stk', hreach_g, hrip', hk', hm', hwf', hr', hsp', haux', hctx', hstk', hstk'_b, hstk'_rw, hstk'_len, _, hoth', hregoth⟩ :=
        fn_aux_push_grow_spec hrip hregs hsp hr15 hr7 hr6 hctx hctx_base hctx_rw hctx_len
          haux haux_base haux_rw hne hane hastk hstk hstkrw hstkne_aux hstkne_ctx (by omega) hspW
          (by omega) hstk_hi hge hle hcap_pos haux_eq haux_pg hauxCapW hctxW hmwf hpos hret
          hc24 hc32 hc40 hat ha_pg ha_pos ha_lim hfree_all
      have ⟨hbl', hpos'⟩ := reaches_bytesLt_len_pos hbl hpos hreach_g
      let ctxData' := writeBytes (writeBytes (writeBytes ctx.data 24 (bytes8 a)) 40 (bytes8 (auxCap * 2))) 32 (bytes8 (auxLen + 1))
      let auxData' := writeBytes (writeBytes (auxReg.data ++ List.replicate (auxCap * 16) 0) (auxLen * 16) (bytes8 w))
        (auxLen * 16 + 8) (bytes8 p)
      have hcs' : ({ ctx with data := ctxData' } : Region).data.length = ctx.data.length := by
        dsimp only [ctxData']; rw [writeBytes_length, writeBytes_length, writeBytes_length]
      have hc24' : leVal (({ ctx with data := ctxData' }.data.drop 24).take 8) = a := by
        dsimp only [ctxData']
        rw [writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inl (by decide)) (by rw [writeBytes_length, writeBytes_length]; omega),
            writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inl (by decide)) (by rw [writeBytes_length]; omega),
            leVal_writeBytes_same _ _ _ (by omega), Nat.mod_eq_of_lt (by omega)]
      have hc32' : leVal (({ ctx with data := ctxData' }.data.drop 32).take 8) = auxLen + 1 := by
        dsimp only [ctxData']
        rw [leVal_writeBytes_same _ _ _ (by rw [writeBytes_length, writeBytes_length]; omega),
            Nat.mod_eq_of_lt (by omega)]
      have hc40' : leVal (({ ctx with data := ctxData' }.data.drop 40).take 8) = auxCap * 2 := by
        dsimp only [ctxData']
        rw [writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inr (by rw [bytes8_length]; decide)) (by rw [writeBytes_length, writeBytes_length]; omega),
            leVal_writeBytes_same _ _ _ (by rw [writeBytes_length]; omega), Nat.mod_eq_of_lt (by omega)]
      have hrest : ∀ j, (j < 24 ∨ 48 ≤ j) → ({ ctx with data := ctxData' } : Region).data[j]? = ctx.data[j]? := by
        intro j hj
        dsimp only [ctxData']
        rw [writeBytes_getElem?_disjoint _ _ _ _ (by rw [bytes8_length]; omega),
            writeBytes_getElem?_disjoint _ _ _ _ (by rw [bytes8_length]; omega),
            writeBytes_getElem?_disjoint _ _ _ _ (by rw [bytes8_length]; omega)]
      have hk_out : s'.k.out = s.k.out := by
        rw [hk']; cases horac : s.k.oracle <;> simp [X86Env.Kern.next, horac]
      have hdvd_cap : (auxCap * 2 * 16) % X86Env.page = 0 := by
        rw [show auxCap * 2 * 16 = auxCap * 32 from by omega]; exact hdvd32
      have haux_hi' : (⟨a, auxData', true⟩ : Region).hi ≤ 2 ^ 47 := by
        simp [auxData', Region.hi, writeBytes_length, List.length_append, List.length_replicate, haux_eq]; omega
      have haux_data' : auxData' = writeBytes (writeBytes (auxReg.data ++ List.replicate ((auxCap * 2 - auxCap) * 16) 0) (auxLen * 16) (bytes8 w))
          (auxLen * 16 + 8) (bytes8 p) := by
        rw [show (auxCap * 2 - auxCap) * 16 = auxCap * 16 by omega]
      refine Or.inl ⟨⟨a, auxData', true⟩, { ctx with data := ctxData' }, stk', m', s', auxCap * 2,
        hreach_g, hrip', hm', hwf', by rw [← hm']; exact hbl', by rw [← hm']; exact hpos', hr',
        oracleAdmits_of_reaches hor hreach_g, hk_out, hsp', by omega, by omega, hdvd_cap, haux', rfl,
        haux_hi', haux_data', hctx', rfl, rfl, ?_⟩
      refine ⟨?_, ?_, ?_, ?_, ?_, hstk', hstk'_b, by rw [hstk'_rw, hstkrw], hstk'_len,
        fun u hu ha => Ne.symm (mem_free_ne_base (by omega) (hpos u hu) (hfree_all u hu ha)), hoth', hregoth⟩
      · exact hcs'
      · exact hc24'
      · exact hc32'
      · exact hc40'
      · exact hrest

theorem call_aux_push_oracle_spec {s : X86Vm.St} {sp retAddr ctxBase auxBase auxLen auxCap w p : Nat}
    {ctx auxReg stk : Region}
    (hrip : s.rip = SporosVmBin.fn_aux_push)
    (hregs : RegsOk s) (hor : OracleAdmits s)
    (hbl : BytesLt s.mem) (hpos : ∀ u ∈ s.mem, 0 < u.data.length)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr7 : s.reg 7 = some w) (hr6 : s.reg 6 = some p)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_rw : ctx.rw = true)
    (hctx_len : ctxSize ≤ ctx.data.length)
    (haux : auxReg ∈ s.mem) (haux_base : auxReg.base = auxBase) (haux_rw : auxReg.rw = true)
    (haux_hi : auxReg.hi ≤ 2 ^ 47)
    (hne : auxBase ≠ ctxBase)
    (hstk : stk ∈ s.mem) (hstkrw : stk.rw = true) (hstkne_aux : stk.base ≠ auxBase) (hstkne_ctx : stk.base ≠ ctxBase)
    (h16 : 16 ≤ sp) (hspW : sp < X86Env.W)
    (hstk_lo : stk.base ≤ sp - 16) (hstk_hi : sp + 8 ≤ stk.hi)
    (hle : auxLen ≤ auxCap) (hcap_pos : 0 < auxCap)
    (haux_eq : auxReg.data.length = auxCap * 16)
    (haux_pg : (auxCap * 16) % X86Env.page = 0)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W)
    (hmwf : s.mem.wf)
    (hret : rd64 s.mem sp = some retAddr)
    (hc24 : leVal ((ctx.data.drop 24).take 8) = auxBase)
    (hc32 : leVal ((ctx.data.drop 32).take 8) = auxLen)
    (hc40 : leVal ((ctx.data.drop 40).take 8) = auxCap) :
    (∃ auxReg' ctx' stk' m' s' auxCap',
      Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.mem = m' ∧ m'.wf ∧ BytesLt m' ∧ (∀ u ∈ m', 0 < u.data.length) ∧
      RegsOk s' ∧ OracleAdmits s' ∧ s'.k.out = s.k.out ∧
      s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      auxCap ≤ auxCap' ∧ auxLen + 1 ≤ auxCap' ∧ (auxCap' * 16) % X86Env.page = 0 ∧
      auxReg' ∈ m' ∧ auxReg'.rw = true ∧ auxReg'.hi ≤ 2 ^ 47 ∧
      auxReg'.data = writeBytes (writeBytes (auxReg.data ++ List.replicate ((auxCap' - auxCap) * 16) 0)
        (auxLen * 16) (bytes8 w)) (auxLen * 16 + 8) (bytes8 p) ∧
      ctx' ∈ m' ∧ ctx'.base = ctx.base ∧ ctx'.rw = ctx.rw ∧ ctx'.data.length = ctx.data.length ∧
      leVal ((ctx'.data.drop 24).take 8) = auxReg'.base ∧
      leVal ((ctx'.data.drop 32).take 8) = auxLen + 1 ∧
      leVal ((ctx'.data.drop 40).take 8) = auxCap' ∧
      (∀ j, (j < 24 ∨ 48 ≤ j) → ctx'.data[j]? = ctx.data[j]?) ∧
      stk' ∈ m' ∧ stk'.base = stk.base ∧ stk'.rw = stk.rw ∧ stk'.data.length = stk.data.length ∧
      (∀ u ∈ s.mem, u.base ≠ auxBase → u.base ≠ auxReg'.base) ∧
      (∀ u ∈ s.mem, u.base ≠ auxBase → u.base ≠ ctxBase → u.base ≠ stk.base → u ∈ m') ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 4 → r ≠ 6 → r ≠ 7 → r ≠ 8 → r ≠ 9 → r ≠ 10 → r ≠ 11 → r ≠ 12 → r ≠ 13 → s'.reg r = s.reg r)) ∨
    (HitsNomem s ∧ Ends vmImage s (.exit 2)) := by
  by_cases hlt : auxLen < auxCap
  · exact call_aux_push_nogrow_oracle_spec hrip hregs hor hbl hpos hsp hr15 hr7 hr6 hctx hctx_base
      hctx_rw hctx_len haux haux_base haux_rw haux_hi hne hstk hstkrw hstkne_aux hstkne_ctx h16 hspW
      hstk_lo hstk_hi hle hcap_pos haux_eq haux_pg hctxW hmwf hret hc24 hc32 hc40 hlt
  · exact call_aux_push_grow_oracle_spec hrip hregs hor hbl hpos hsp hr15 hr7 hr6 hctx hctx_base
      hctx_rw hctx_len haux haux_base haux_rw haux_hi hne hstk hstkrw hstkne_aux hstkne_ctx h16 hspW
      hstk_lo hstk_hi hle hcap_pos haux_eq haux_pg hctxW hmwf hret hc24 hc32 hc40 hlt

private theorem call_out_push_byte_nogrow_oracle_spec {s : X86Vm.St} {sp retAddr ctxBase outBase outLen outCap b : Nat}
    {ctx outReg stk : Region}
    (hrip : s.rip = SporosVmBin.fn_out_push_byte)
    (hregs : RegsOk s) (hor : OracleAdmits s)
    (hbl : BytesLt s.mem) (hpos : ∀ u ∈ s.mem, 0 < u.data.length)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr7 : s.reg 7 = some b)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_rw : ctx.rw = true)
    (hctx_len : ctxSize ≤ ctx.data.length)
    (hout : outReg ∈ s.mem) (hout_base : outReg.base = outBase) (hout_rw : outReg.rw = true)
    (hout_hi : outReg.hi ≤ 2 ^ 47)
    (hne : outBase ≠ ctxBase)
    (hstk : stk ∈ s.mem) (_hstkrw : stk.rw = true) (hstkne_out : stk.base ≠ outBase) (hstkne_ctx : stk.base ≠ ctxBase)
    (h16 : 16 ≤ sp) (_hspW : sp < X86Env.W)
    (hstk_lo : stk.base ≤ sp - 16) (hstk_hi : sp + 8 ≤ stk.hi)
    (_hle : outLen ≤ outCap) (hcap_pos : 0 < outCap)
    (hout_eq : outReg.data.length = outCap)
    (hout_pg : outCap % X86Env.page = 0)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W)
    (hmwf : s.mem.wf)
    (hret : rd64 s.mem sp = some retAddr)
    (hc200 : leVal ((ctx.data.drop 200).take 8) = outBase)
    (hc208 : leVal ((ctx.data.drop 208).take 8) = outLen)
    (hc216 : leVal ((ctx.data.drop 216).take 8) = outCap)
    (hlt : outLen < outCap) :
    (∃ outReg' ctx' stk' m' s' outCap',
      Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.mem = m' ∧ m'.wf ∧ BytesLt m' ∧ (∀ u ∈ m', 0 < u.data.length) ∧
      RegsOk s' ∧ OracleAdmits s' ∧ s'.k.out = s.k.out ∧
      s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      outCap ≤ outCap' ∧ outLen + 1 ≤ outCap' ∧ outCap' % X86Env.page = 0 ∧
      outReg' ∈ m' ∧ outReg'.rw = true ∧ outReg'.hi ≤ 2 ^ 47 ∧
      outReg'.data = writeBytes (outReg.data ++ List.replicate (outCap' - outCap) 0) outLen [b % 256] ∧
      ctx' ∈ m' ∧ ctx'.base = ctx.base ∧ ctx'.rw = ctx.rw ∧ ctx'.data.length = ctx.data.length ∧
      leVal ((ctx'.data.drop 200).take 8) = outReg'.base ∧
      leVal ((ctx'.data.drop 208).take 8) = outLen + 1 ∧
      leVal ((ctx'.data.drop 216).take 8) = outCap' ∧
      (∀ j, (j < 200 ∨ 224 ≤ j) → ctx'.data[j]? = ctx.data[j]?) ∧
      stk' ∈ m' ∧ stk'.base = stk.base ∧ stk'.rw = stk.rw ∧ stk'.data.length = stk.data.length ∧
      (∀ u ∈ s.mem, u.base ≠ outBase → u.base ≠ outReg'.base) ∧
      (∀ u ∈ s.mem, u.base ≠ outBase → u.base ≠ ctxBase → u.base ≠ stk.base → u ∈ m') ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 4 → r ≠ 6 → r ≠ 7 → r ≠ 8 → r ≠ 9 → r ≠ 10 → r ≠ 11 → r ≠ 12 → s'.reg r = s.reg r)) ∨
    (HitsNomem s ∧ Ends vmImage s (.exit 2)) := by
    have hW : X86Env.W = 2 ^ 64 := rfl
    have houtW : outBase + outCap ≤ X86Env.W := by
      simp only [Region.hi, hout_base, hout_eq] at hout_hi; omega
    have houtCapW : outCap * 2 < X86Env.W := by
      simp only [Region.hi, hout_eq] at hout_hi; omega
    obtain ⟨m', s', hreach, hrip', hk', hm', hwf', hr', hsp', hout', hctx', hoth', hregoth⟩ :=
      fn_out_push_byte_nogrow_spec hrip hregs hsp hr15 hr7 hctx hctx_base hctx_rw hctx_len
        hout hout_base hout_rw hne hstk hstkne_out hstkne_ctx (by omega) hstk_hi hlt (by omega) hctxW houtW
        (by omega) hmwf hret hc200 hc208 hc216
    have ⟨hbl', hpos'⟩ := reaches_bytesLt_len_pos hbl hpos hreach
    have hstk' := hoth' stk hstk hstkne_out hstkne_ctx
    have hcs : ctxSize = 4096 := rfl
    have hc200' : leVal ((({ ctx with data := writeBytes ctx.data 208 (bytes8 (outLen + 1)) }.data).drop 200).take 8) = outReg.base := by
      dsimp only
      rw [writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inl (by decide)) (by omega), hc200, hout_base]
    have hc208' : leVal ((({ ctx with data := writeBytes ctx.data 208 (bytes8 (outLen + 1)) }.data).drop 208).take 8) = outLen + 1 := by
      dsimp only
      rw [leVal_writeBytes_same _ _ _ (by omega), Nat.mod_eq_of_lt (by omega)]
    have hc216' : leVal ((({ ctx with data := writeBytes ctx.data 208 (bytes8 (outLen + 1)) }.data).drop 216).take 8) = outCap := by
      dsimp only
      rw [writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inr (by rw [bytes8_length]; decide)) (by omega), hc216]
    have hrest : ∀ j, (j < 200 ∨ 224 ≤ j) → (writeBytes ctx.data 208 (bytes8 (outLen + 1)))[j]? = ctx.data[j]? := by
      intro j hj
      apply writeBytes_getElem?_disjoint
      rw [bytes8_length]
      omega
    refine Or.inl ⟨{ outReg with data := _ }, { ctx with data := writeBytes ctx.data 208 (bytes8 (outLen + 1)) },
      stk, m', s', outCap, hreach, hrip', hm', hwf', by rw [← hm']; exact hbl', by rw [← hm']; exact hpos', hr', oracleAdmits_of_reaches hor hreach,
      by rw [hk'], hsp', Nat.le_refl _, by omega, hout_pg, hout', hout_rw,
      by simp only [Region.hi, writeBytes_length, hout_eq] at hout_hi ⊢; exact hout_hi,
      by simp [Nat.sub_self], hctx', rfl, rfl, writeBytes_length _ _ _,
      hc200', hc208', hc216', hrest, hstk', rfl, rfl, rfl,
      fun _ _ ho => by rw [hout_base]; exact ho,
      fun u hu ho hc _ => hoth' u hu ho hc,
      fun r h0 h1 _ h4 _ _ _ _ _ _ h12 => hregoth r h0 h1 h4 h12⟩

private theorem call_out_push_byte_grow_oracle_spec {s : X86Vm.St} {sp retAddr ctxBase outBase outLen outCap b : Nat}
    {ctx outReg stk : Region}
    (hrip : s.rip = SporosVmBin.fn_out_push_byte)
    (hregs : RegsOk s) (hor : OracleAdmits s)
    (hbl : BytesLt s.mem) (hpos : ∀ u ∈ s.mem, 0 < u.data.length)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr7 : s.reg 7 = some b)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_rw : ctx.rw = true)
    (hctx_len : ctxSize ≤ ctx.data.length)
    (hout : outReg ∈ s.mem) (hout_base : outReg.base = outBase) (hout_rw : outReg.rw = true)
    (hout_hi : outReg.hi ≤ 2 ^ 47)
    (hne : outBase ≠ ctxBase)
    (hstk : stk ∈ s.mem) (hstkrw : stk.rw = true) (hstkne_out : stk.base ≠ outBase) (hstkne_ctx : stk.base ≠ ctxBase)
    (h16 : 16 ≤ sp) (hspW : sp < X86Env.W)
    (hstk_lo : stk.base ≤ sp - 16) (hstk_hi : sp + 8 ≤ stk.hi)
    (hle : outLen ≤ outCap) (hcap_pos : 0 < outCap)
    (hout_eq : outReg.data.length = outCap)
    (hout_pg : outCap % X86Env.page = 0)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W)
    (hmwf : s.mem.wf)
    (hret : rd64 s.mem sp = some retAddr)
    (hc200 : leVal ((ctx.data.drop 200).take 8) = outBase)
    (hc208 : leVal ((ctx.data.drop 208).take 8) = outLen)
    (hc216 : leVal ((ctx.data.drop 216).take 8) = outCap)
    (hlt : ¬ outLen < outCap) :
    (∃ outReg' ctx' stk' m' s' outCap',
      Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.mem = m' ∧ m'.wf ∧ BytesLt m' ∧ (∀ u ∈ m', 0 < u.data.length) ∧
      RegsOk s' ∧ OracleAdmits s' ∧ s'.k.out = s.k.out ∧
      s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      outCap ≤ outCap' ∧ outLen + 1 ≤ outCap' ∧ outCap' % X86Env.page = 0 ∧
      outReg' ∈ m' ∧ outReg'.rw = true ∧ outReg'.hi ≤ 2 ^ 47 ∧
      outReg'.data = writeBytes (outReg.data ++ List.replicate (outCap' - outCap) 0) outLen [b % 256] ∧
      ctx' ∈ m' ∧ ctx'.base = ctx.base ∧ ctx'.rw = ctx.rw ∧ ctx'.data.length = ctx.data.length ∧
      leVal ((ctx'.data.drop 200).take 8) = outReg'.base ∧
      leVal ((ctx'.data.drop 208).take 8) = outLen + 1 ∧
      leVal ((ctx'.data.drop 216).take 8) = outCap' ∧
      (∀ j, (j < 200 ∨ 224 ≤ j) → ctx'.data[j]? = ctx.data[j]?) ∧
      stk' ∈ m' ∧ stk'.base = stk.base ∧ stk'.rw = stk.rw ∧ stk'.data.length = stk.data.length ∧
      (∀ u ∈ s.mem, u.base ≠ outBase → u.base ≠ outReg'.base) ∧
      (∀ u ∈ s.mem, u.base ≠ outBase → u.base ≠ ctxBase → u.base ≠ stk.base → u ∈ m') ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 4 → r ≠ 6 → r ≠ 7 → r ≠ 8 → r ≠ 9 → r ≠ 10 → r ≠ 11 → r ≠ 12 → s'.reg r = s.reg r)) ∨
    (HitsNomem s ∧ Ends vmImage s (.exit 2)) := by
    have hW : X86Env.W = 2 ^ 64 := rfl
    have houtCapW : outCap * 2 < X86Env.W := by
      simp only [Region.hi, hout_eq] at hout_hi; omega
    have hge : outCap ≤ outLen := Nat.le_of_not_lt hlt
    have h01 : X86Vm.step vmImage s = .ok { s.set 12 (some b) with rip := 4201628 } := by
      rw [step_at_chunk 24 hrip rfl (by decide : (SporosVmBin.fn_out_push_byte, (.ofInt (.mov 12 7)), 3) ∈ SporosVmBin.vmTable_24),
        exec_mov hr7, hrip]; rfl
    let s1 := { s.set 12 (some b) with rip := 4201628 }
    have hr1 : RegsOk s1 := regsOk_set hregs (by decide) b
    have hs1_15 : s1.reg 15 = some ctxBase := by
      simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 12 ≠ 15)]; exact hr15
    have hcs : ctxSize = 4096 := rfl
    have hea_c200 : ea ctxBase 200 = ctxBase + 200 := ea_add_eq (by omega)
    have hea_c208 : ea ctxBase 208 = ctxBase + 208 := ea_add_eq (by omega)
    have hea_c216 : ea ctxBase 216 = ctxBase + 216 := ea_add_eq (by omega)
    have hrd_c200 : rd64 s1.mem (ea ctxBase 200) = some outBase := by
      rw [hea_c200]; exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
        (by simp only [Region.hi, hctx_base]; omega) (by rw [hctx_base, Nat.add_sub_cancel_left]; exact hc200)
    have hrd_c208 : rd64 s1.mem (ea ctxBase 208) = some outLen := by
      rw [hea_c208]; exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
        (by simp only [Region.hi, hctx_base]; omega) (by rw [hctx_base, Nat.add_sub_cancel_left]; exact hc208)
    have hrd_c216 : rd64 s1.mem (ea ctxBase 216) = some outCap := by
      rw [hea_c216]; exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
        (by simp only [Region.hi, hctx_base]; omega) (by rw [hctx_base, Nat.add_sub_cancel_left]; exact hc216)
    have h02 : X86Vm.step vmImage s1 = .ok { s1.set 0 (some outLen) with rip := 4201635 } := by
      rw [step_at_chunk 24 rfl rfl (by decide : (4201628, (.ofInt (.ld 0 15 208)), 7) ∈ SporosVmBin.vmTable_24),
        exec_ld hs1_15 hrd_c208]
    let s2 := { s1.set 0 (some outLen) with rip := 4201635 }
    have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) outLen
    have hs2_15 : s2.reg 15 = some ctxBase := by
      simp only [s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 15)]; exact hs1_15
    have h03 : X86Vm.step vmImage s2 = .ok { s2.set 1 (some outCap) with rip := 4201642 } := by
      rw [step_at_chunk 24 rfl rfl (by decide : (4201635, (.ofInt (.ld 1 15 216)), 7) ∈ SporosVmBin.vmTable_24),
        exec_ld hs2_15 hrd_c216]
    let s3 := { s2.set 1 (some outCap) with rip := 4201642 }
    have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) outCap
    have hs3_0 : s3.reg 0 = some outLen := by
      simp only [s3, s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0)]
      rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
    have hs3_1 : s3.reg 1 = some outCap := by
      simp only [s3, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
    have h04 : X86Vm.step vmImage s3 = .ok { s3 with fl := (X86Int.aluF .sub outLen outCap).2, rip := 4201645 } := by
      rw [step_at_chunk 24 rfl rfl (by decide : (4201642, (.ofInt (.cmp 0 1)), 3) ∈ SporosVmBin.vmTable_24),
        exec_cmp hs3_0 hs3_1]
    let s4 := { s3 with fl := (X86Int.aluF .sub outLen outCap).2, rip := 4201645 }
    have h05 : X86Vm.step vmImage s4 = .ok { s4 with rip := 4201647 } := by
      rw [step_at_chunk 24 rfl rfl (by decide : (4201645, (.ofInt (.jcc 2 4201694)), 2) ∈ SporosVmBin.vmTable_24),
        exec_jcc (cond_aluF_sub_b outLen outCap)]
      simp [s4, hlt]
    let s5 := { s4 with rip := 4201647 }
    have hs5_15 : s5.reg 15 = some ctxBase := by
      simp only [s5, s4, s3, s2, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 1 ≠ 15), List.getElem?_set_ne (by decide : 0 ≠ 15)]
      exact hs1_15
    have h06 : X86Vm.step vmImage s5 = .ok { s5.set 7 (some outBase) with rip := 4201654 } := by
      rw [step_at_chunk 24 rfl rfl (by decide : (4201647, (.ofInt (.ld 7 15 200)), 7) ∈ SporosVmBin.vmTable_24),
        exec_ld hs5_15 hrd_c200]
    let s6 := { s5.set 7 (some outBase) with rip := 4201654 }
    have hr6 : RegsOk s6 := regsOk_set hr3 (by decide) outBase
    have hs6_1 : s6.reg 1 = some outCap := by
      simp only [s6, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 7 ≠ 1)]
      exact hs3_1
    have h07 : X86Vm.step vmImage s6 = .ok { s6.set 6 (some outCap) with rip := 4201657 } := by
      rw [step_at_chunk 24 rfl rfl (by decide : (4201654, (.ofInt (.mov 6 1)), 3) ∈ SporosVmBin.vmTable_24),
        exec_mov hs6_1]
    let s7 := { s6.set 6 (some outCap) with rip := 4201657 }
    have hr7' : RegsOk s7 := regsOk_set hr6 (by decide) outCap
    have hs7_1 : s7.reg 1 = some outCap := by
      simp only [s7, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 6 ≠ 1)]
      exact hs6_1
    have h08 : X86Vm.step vmImage s7 = .ok { s7.set 2 (some outCap) with rip := 4201660 } := by
      rw [step_at_chunk 24 rfl rfl (by decide : (4201657, (.ofInt (.mov 2 1)), 3) ∈ SporosVmBin.vmTable_24),
        exec_mov hs7_1]
    let s8 := { s7.set 2 (some outCap) with rip := 4201660 }
    have hr8 : RegsOk s8 := regsOk_set hr7' (by decide) outCap
    have hs8_2 : s8.reg 2 = some outCap := by
      simp only [s8, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr7'.1]; decide)]
    have h09 : X86Vm.step vmImage s8 =
        .ok { s8.set 2 (some (outCap * 2)) with fl := X86Int.Flags.undef, rip := 4201664 } := by
      rw [step_at_chunk 24 rfl rfl (by decide : (4201660, (.ofInt (.shlI 2 1)), 4) ∈ SporosVmBin.vmTable_24),
        exec_shlI_1_lt hs8_2 houtCapW]
    let s9 := { s8.set 2 (some (outCap * 2)) with fl := X86Int.Flags.undef, rip := 4201664 }
    have hr9 : RegsOk s9 := regsOk_set hr8 (by decide) (outCap * 2)
    have hs9_4 : s9.reg 4 = some sp := by
      simp only [s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 2 ≠ 4), List.getElem?_set_ne (by decide : 6 ≠ 4),
        List.getElem?_set_ne (by decide : 7 ≠ 4), List.getElem?_set_ne (by decide : 1 ≠ 4),
        List.getElem?_set_ne (by decide : 0 ≠ 4), List.getElem?_set_ne (by decide : 12 ≠ 4)]
      exact hsp
    have hea_sp : (sp + X86Env.W - 8) % X86Env.W = sp - 8 := by
      have : sp + X86Env.W - 8 = (sp - 8) + X86Env.W := by omega
      rw [this, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
    obtain ⟨_, m10, hwr10, _, _, hwf10, hstk10, hoth10⟩ :=
      wr64_region hmwf hstk hstkrw (by omega) (by omega : sp - 8 + 8 ≤ stk.hi) (v := 4201669)
    have hwr10' : wr64 s9.mem ((sp + X86Env.W - 8) % X86Env.W) (4201664 + 5) = some m10 := by
      simpa [s9, s8, s7, s6, s5, s4, s3, s2, s1, hea_sp] using hwr10
    have h10 : X86Vm.step vmImage s9 =
        .ok { s9.set 4 (some (sp - 8)) with mem := m10, rip := SporosVmBin.fn_mremap } := by
      rw [step_at_chunk 24 rfl rfl (by decide : (4201664, (.ofInt (.call 4201363)), 5) ∈ SporosVmBin.vmTable_24),
        exec_call hs9_4 hwr10']
      simp [s9, hea_sp, SporosVmBin.fn_mremap]
    let s10 := { s9.set 4 (some (sp - 8)) with mem := m10, rip := SporosVmBin.fn_mremap }
    have hr10 : RegsOk s10 := regsOk_set hr9 (by decide) (sp - 8)
    have hreach10 : Reaches vmImage s s10 :=
      Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
        (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step_trans h08
          (Reaches.step_trans h09 (Reaches.step h10)))))))))
    have hout10 : outReg ∈ s10.mem := hoth10 outReg hout (by rw [hout_base]; exact Ne.symm hstkne_out)
    have hout10_hi : outBase < outReg.hi := by simp only [Region.hi, hout_base, hout_eq]; omega
    obtain ⟨idx, hfind10⟩ := mem_find_of_mem s10.mem outReg outBase 0 hwf10.2 hout10 (by rw [hout_base]; exact Nat.le_refl _) hout10_hi
    have hs10_7 : s10.reg 7 = some outBase := by
      simp only [s10, s9, s8, s7, s6, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 4 ≠ 7), List.getElem?_set_ne (by decide : 2 ≠ 7),
        List.getElem?_set_ne (by decide : 6 ≠ 7)]
      rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
    have hs10_6 : s10.reg 6 = some outCap := by
      simp only [s10, s9, s8, s7, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 4 ≠ 6), List.getElem?_set_ne (by decide : 2 ≠ 6)]
      rw [getElem?_set_getD_self (by rw [hr6.1]; decide)]
    have hs10_2 : s10.reg 2 = some (outCap * 2) := by
      simp only [s10, s9, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 4 ≠ 2)]
      rw [getElem?_set_getD_self (by rw [hr8.1]; decide)]
    have hpg1 : X86Env.pages outCap = outCap := pages_of_dvd hout_pg
    have hdvd2 : (outCap * 2) % X86Env.page = 0 := by
      have hmul : outCap * 2 = 2 * outCap := by omega
      rw [hmul, Nat.mul_mod, hout_pg]; simp
    have hpg2 : X86Env.pages (outCap * 2) = outCap * 2 := pages_of_dvd hdvd2
    rcases at_mremap_oracle_spec rfl hr10 (oracleAdmits_of_reaches hor hreach10)
        hs10_7 hs10_6 hs10_2 hfind10 hout_base (by rw [hout_eq, hpg1]) hout_rw
        (by rw [hout_eq, hpg2]; omega) with ⟨hnomem, hhit10⟩ | ⟨a, hat, ha_pg, ha_pos, ha_lim, hfree⟩
    · have hs10_k : s10.k = s.k := rfl
      rw [hs10_k] at hnomem
      exact Or.inr ⟨hitsNomem_of_reaches hreach10 hhit10,
        fn_out_push_byte_nomem_spec hrip hregs hsp hr15 hr7 hctx hctx_base hctx_len
          hout hout_base hout_rw hstk hstkrw hstkne_out h16 hspW hstk_lo (by omega) hge hle hcap_pos
          hout_eq hout_pg houtCapW hctxW hmwf hc200 hc208 hc216 hnomem⟩
    · have hs10_k : s10.k = s.k := rfl
      rw [hs10_k] at hat
      rw [hpg2] at ha_lim hfree
      rw [Nat.zero_add] at hfind10 hfree
      have hget10 : m10[idx]? = some outReg := by
        simpa using (mem_find_idx_lt s10.mem outBase 0 idx outReg hfind10).2.2
      change X86Env.Mem.free (m10.eraseIdx idx) a (outCap * 2) = true at hfree
      have hstk10_b : { stk with data := writeBytes stk.data (sp - 8 - stk.base) (bytes8 4201669) }.base = stk.base := rfl
      have hstk10_hi : { stk with data := writeBytes stk.data (sp - 8 - stk.base) (bytes8 4201669) }.hi = stk.hi := by
        simp [Region.hi, writeBytes_length]
      obtain ⟨hastk, hfree_all⟩ := mremap_find_after_stack_write (n := outCap * 2) hmwf hstk hstk10_b hstk10_hi
        hstkne_out hstk10 hoth10 hget10 hout_base hfree (by omega) hpos
      have hane : a ≠ ctxBase := by
        rw [← hctx_base]
        exact mem_free_ne_base (by omega) (hpos ctx hctx) (hfree_all ctx hctx (by rw [hctx_base]; exact Ne.symm hne))
      obtain ⟨m', s', stk', hreach_g, hrip', hk', hm', hwf', hr', hsp', hout', hctx', hstk', hstk'_b, hstk'_rw, hstk'_len, _, hoth', hregoth⟩ :=
        fn_out_push_byte_grow_spec hrip hregs hsp hr15 hr7 hctx hctx_base hctx_rw hctx_len
          hout hout_base hout_rw hne hane hastk hstk hstkrw hstkne_out hstkne_ctx (by omega) hspW
          (by omega) hstk_hi hge hle hcap_pos hout_eq hout_pg houtCapW hctxW hmwf hpos hret
          hc200 hc208 hc216 hat ha_pg ha_pos ha_lim hfree_all
      have ⟨hbl', hpos'⟩ := reaches_bytesLt_len_pos hbl hpos hreach_g
      let ctxData' := writeBytes (writeBytes (writeBytes ctx.data 200 (bytes8 a)) 216 (bytes8 (outCap * 2))) 208 (bytes8 (outLen + 1))
      let outData' := writeBytes (outReg.data ++ List.replicate outCap 0) outLen [b % 256]
      have hcs' : ({ ctx with data := ctxData' } : Region).data.length = ctx.data.length := by
        dsimp only [ctxData']; rw [writeBytes_length, writeBytes_length, writeBytes_length]
      have hc200' : leVal (({ ctx with data := ctxData' }.data.drop 200).take 8) = a := by
        dsimp only [ctxData']
        rw [writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inl (by decide)) (by rw [writeBytes_length, writeBytes_length]; omega),
            writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inl (by decide)) (by rw [writeBytes_length]; omega),
            leVal_writeBytes_same _ _ _ (by omega), Nat.mod_eq_of_lt (by omega)]
      have hc208' : leVal (({ ctx with data := ctxData' }.data.drop 208).take 8) = outLen + 1 := by
        dsimp only [ctxData']
        rw [leVal_writeBytes_same _ _ _ (by rw [writeBytes_length, writeBytes_length]; omega),
            Nat.mod_eq_of_lt (by omega)]
      have hc216' : leVal (({ ctx with data := ctxData' }.data.drop 216).take 8) = outCap * 2 := by
        dsimp only [ctxData']
        rw [writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inr (by rw [bytes8_length]; decide)) (by rw [writeBytes_length, writeBytes_length]; omega),
            leVal_writeBytes_same _ _ _ (by rw [writeBytes_length]; omega), Nat.mod_eq_of_lt (by omega)]
      have hrest : ∀ j, (j < 200 ∨ 224 ≤ j) → ({ ctx with data := ctxData' } : Region).data[j]? = ctx.data[j]? := by
        intro j hj
        dsimp only [ctxData']
        rw [writeBytes_getElem?_disjoint _ _ _ _ (by rw [bytes8_length]; omega),
            writeBytes_getElem?_disjoint _ _ _ _ (by rw [bytes8_length]; omega),
            writeBytes_getElem?_disjoint _ _ _ _ (by rw [bytes8_length]; omega)]
      have hk_out : s'.k.out = s.k.out := by
        rw [hk']; cases horac : s.k.oracle <;> simp [X86Env.Kern.next, horac]
      have hout_hi' : (⟨a, outData', true⟩ : Region).hi ≤ 2 ^ 47 := by
        simp [outData', Region.hi, writeBytes_length, List.length_append, List.length_replicate, hout_eq]; omega
      have hout_data' : outData' = writeBytes (outReg.data ++ List.replicate (outCap * 2 - outCap) 0) outLen [b % 256] := by
        rw [show outCap * 2 - outCap = outCap by omega]
      refine Or.inl ⟨⟨a, outData', true⟩, { ctx with data := ctxData' }, stk', m', s', outCap * 2,
        hreach_g, hrip', hm', hwf', by rw [← hm']; exact hbl', by rw [← hm']; exact hpos', hr',
        oracleAdmits_of_reaches hor hreach_g, hk_out, hsp', by omega, by omega, hdvd2, hout', rfl,
        hout_hi', hout_data', hctx', rfl, rfl, ?_⟩
      refine ⟨?_, ?_, ?_, ?_, ?_, hstk', hstk'_b, by rw [hstk'_rw, hstkrw], hstk'_len,
        fun u hu ho => Ne.symm (mem_free_ne_base (by omega) (hpos u hu) (hfree_all u hu ho)), hoth', hregoth⟩
      · exact hcs'
      · exact hc200'
      · exact hc208'
      · exact hc216'
      · exact hrest

theorem call_out_push_byte_oracle_spec {s : X86Vm.St} {sp retAddr ctxBase outBase outLen outCap b : Nat}
    {ctx outReg stk : Region}
    (hrip : s.rip = SporosVmBin.fn_out_push_byte)
    (hregs : RegsOk s) (hor : OracleAdmits s)
    (hbl : BytesLt s.mem) (hpos : ∀ u ∈ s.mem, 0 < u.data.length)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr7 : s.reg 7 = some b)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_rw : ctx.rw = true)
    (hctx_len : ctxSize ≤ ctx.data.length)
    (hout : outReg ∈ s.mem) (hout_base : outReg.base = outBase) (hout_rw : outReg.rw = true)
    (hout_hi : outReg.hi ≤ 2 ^ 47)
    (hne : outBase ≠ ctxBase)
    (hstk : stk ∈ s.mem) (hstkrw : stk.rw = true) (hstkne_out : stk.base ≠ outBase) (hstkne_ctx : stk.base ≠ ctxBase)
    (h16 : 16 ≤ sp) (hspW : sp < X86Env.W)
    (hstk_lo : stk.base ≤ sp - 16) (hstk_hi : sp + 8 ≤ stk.hi)
    (hle : outLen ≤ outCap) (hcap_pos : 0 < outCap)
    (hout_eq : outReg.data.length = outCap)
    (hout_pg : outCap % X86Env.page = 0)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W)
    (hmwf : s.mem.wf)
    (hret : rd64 s.mem sp = some retAddr)
    (hc200 : leVal ((ctx.data.drop 200).take 8) = outBase)
    (hc208 : leVal ((ctx.data.drop 208).take 8) = outLen)
    (hc216 : leVal ((ctx.data.drop 216).take 8) = outCap) :
    (∃ outReg' ctx' stk' m' s' outCap',
      Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.mem = m' ∧ m'.wf ∧ BytesLt m' ∧ (∀ u ∈ m', 0 < u.data.length) ∧
      RegsOk s' ∧ OracleAdmits s' ∧ s'.k.out = s.k.out ∧
      s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      outCap ≤ outCap' ∧ outLen + 1 ≤ outCap' ∧ outCap' % X86Env.page = 0 ∧
      outReg' ∈ m' ∧ outReg'.rw = true ∧ outReg'.hi ≤ 2 ^ 47 ∧
      outReg'.data = writeBytes (outReg.data ++ List.replicate (outCap' - outCap) 0) outLen [b % 256] ∧
      ctx' ∈ m' ∧ ctx'.base = ctx.base ∧ ctx'.rw = ctx.rw ∧ ctx'.data.length = ctx.data.length ∧
      leVal ((ctx'.data.drop 200).take 8) = outReg'.base ∧
      leVal ((ctx'.data.drop 208).take 8) = outLen + 1 ∧
      leVal ((ctx'.data.drop 216).take 8) = outCap' ∧
      (∀ j, (j < 200 ∨ 224 ≤ j) → ctx'.data[j]? = ctx.data[j]?) ∧
      stk' ∈ m' ∧ stk'.base = stk.base ∧ stk'.rw = stk.rw ∧ stk'.data.length = stk.data.length ∧
      (∀ u ∈ s.mem, u.base ≠ outBase → u.base ≠ outReg'.base) ∧
      (∀ u ∈ s.mem, u.base ≠ outBase → u.base ≠ ctxBase → u.base ≠ stk.base → u ∈ m') ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 4 → r ≠ 6 → r ≠ 7 → r ≠ 8 → r ≠ 9 → r ≠ 10 → r ≠ 11 → r ≠ 12 → s'.reg r = s.reg r)) ∨
    (HitsNomem s ∧ Ends vmImage s (.exit 2)) := by
  by_cases hlt : outLen < outCap
  · exact call_out_push_byte_nogrow_oracle_spec hrip hregs hor hbl hpos hsp hr15 hr7 hctx hctx_base
      hctx_rw hctx_len hout hout_base hout_rw hout_hi hne hstk hstkrw hstkne_out hstkne_ctx h16 hspW
      hstk_lo hstk_hi hle hcap_pos hout_eq hout_pg hctxW hmwf hret hc200 hc208 hc216 hlt
  · exact call_out_push_byte_grow_oracle_spec hrip hregs hor hbl hpos hsp hr15 hr7 hctx hctx_base
      hctx_rw hctx_len hout hout_base hout_rw hout_hi hne hstk hstkrw hstkne_out hstkne_ctx h16 hspW
      hstk_lo hstk_hi hle hcap_pos hout_eq hout_pg hctxW hmwf hret hc200 hc208 hc216 hlt

private theorem call_out_push_u64_nogrow_oracle_spec {s : X86Vm.St} {sp retAddr ctxBase outBase outLen outCap v : Nat}
    {ctx outReg stk : Region}
    (hrip : s.rip = SporosVmBin.fn_out_push_u64)
    (hregs : RegsOk s) (hor : OracleAdmits s)
    (hbl : BytesLt s.mem) (hpos : ∀ u ∈ s.mem, 0 < u.data.length)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr7 : s.reg 7 = some v)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_rw : ctx.rw = true)
    (hctx_len : ctxSize ≤ ctx.data.length)
    (hout : outReg ∈ s.mem) (hout_base : outReg.base = outBase) (hout_rw : outReg.rw = true)
    (hout_hi : outReg.hi ≤ 2 ^ 47)
    (hne : outBase ≠ ctxBase)
    (hstk : stk ∈ s.mem) (_hstkrw : stk.rw = true) (hstkne_out : stk.base ≠ outBase) (hstkne_ctx : stk.base ≠ ctxBase)
    (h16 : 16 ≤ sp) (_hspW : sp < X86Env.W)
    (hstk_lo : stk.base ≤ sp - 16) (hstk_hi : sp + 8 ≤ stk.hi)
    (_hle : outLen ≤ outCap) (hcap_ge : 8 ≤ outCap)
    (hout_eq : outReg.data.length = outCap)
    (hout_pg : outCap % X86Env.page = 0)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W)
    (hmwf : s.mem.wf)
    (hret : rd64 s.mem sp = some retAddr)
    (hc200 : leVal ((ctx.data.drop 200).take 8) = outBase)
    (hc208 : leVal ((ctx.data.drop 208).take 8) = outLen)
    (hc216 : leVal ((ctx.data.drop 216).take 8) = outCap)
    (hfit : outLen + 8 ≤ outCap) :
    (∃ outReg' ctx' stk' m' s' outCap',
      Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.mem = m' ∧ m'.wf ∧ BytesLt m' ∧ (∀ u ∈ m', 0 < u.data.length) ∧
      RegsOk s' ∧ OracleAdmits s' ∧ s'.k.out = s.k.out ∧
      s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      outCap ≤ outCap' ∧ outLen + 8 ≤ outCap' ∧ outCap' % X86Env.page = 0 ∧
      outReg' ∈ m' ∧ outReg'.rw = true ∧ outReg'.hi ≤ 2 ^ 47 ∧
      outReg'.data = writeBytes (outReg.data ++ List.replicate (outCap' - outCap) 0) outLen (bytes8 v) ∧
      ctx' ∈ m' ∧ ctx'.base = ctx.base ∧ ctx'.rw = ctx.rw ∧ ctx'.data.length = ctx.data.length ∧
      leVal ((ctx'.data.drop 200).take 8) = outReg'.base ∧
      leVal ((ctx'.data.drop 208).take 8) = outLen + 8 ∧
      leVal ((ctx'.data.drop 216).take 8) = outCap' ∧
      (∀ j, (j < 200 ∨ 224 ≤ j) → ctx'.data[j]? = ctx.data[j]?) ∧
      stk' ∈ m' ∧ stk'.base = stk.base ∧ stk'.rw = stk.rw ∧ stk'.data.length = stk.data.length ∧
      (∀ u ∈ s.mem, u.base ≠ outBase → u.base ≠ outReg'.base) ∧
      (∀ u ∈ s.mem, u.base ≠ outBase → u.base ≠ ctxBase → u.base ≠ stk.base → u ∈ m') ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 4 → r ≠ 6 → r ≠ 7 → r ≠ 8 → r ≠ 9 → r ≠ 10 → r ≠ 11 → r ≠ 12 → s'.reg r = s.reg r)) ∨
    (HitsNomem s ∧ Ends vmImage s (.exit 2)) := by
    have hW : X86Env.W = 2 ^ 64 := rfl
    have houtW : outBase + outCap ≤ X86Env.W := by
      simp only [Region.hi, hout_base, hout_eq] at hout_hi; omega
    have houtCapW : outCap * 2 < X86Env.W := by
      simp only [Region.hi, hout_eq] at hout_hi; omega
    obtain ⟨m', s', hreach, hrip', hk', hm', hwf', hr', hsp', hout', hctx', hoth', hregoth⟩ :=
      fn_out_push_u64_nogrow_spec hrip hregs hsp hr15 hr7 hctx hctx_base hctx_rw hctx_len
        hout hout_base hout_rw hne hstk hstkne_out hstkne_ctx (by omega) hstk_hi hfit (by omega) hctxW houtW
        (by omega) hmwf hret hc200 hc208 hc216
    have ⟨hbl', hpos'⟩ := reaches_bytesLt_len_pos hbl hpos hreach
    have hstk' := hoth' stk hstk hstkne_out hstkne_ctx
    have hcs : ctxSize = 4096 := rfl
    have hc200' : leVal ((({ ctx with data := writeBytes ctx.data 208 (bytes8 (outLen + 8)) }.data).drop 200).take 8) = outReg.base := by
      dsimp only
      rw [writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inl (by decide)) (by omega), hc200, hout_base]
    have hc208' : leVal ((({ ctx with data := writeBytes ctx.data 208 (bytes8 (outLen + 8)) }.data).drop 208).take 8) = outLen + 8 := by
      dsimp only
      rw [leVal_writeBytes_same _ _ _ (by omega), Nat.mod_eq_of_lt (by omega)]
    have hc216' : leVal ((({ ctx with data := writeBytes ctx.data 208 (bytes8 (outLen + 8)) }.data).drop 216).take 8) = outCap := by
      dsimp only
      rw [writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inr (by rw [bytes8_length]; decide)) (by omega), hc216]
    have hrest : ∀ j, (j < 200 ∨ 224 ≤ j) → (writeBytes ctx.data 208 (bytes8 (outLen + 8)))[j]? = ctx.data[j]? := by
      intro j hj
      apply writeBytes_getElem?_disjoint
      rw [bytes8_length]
      omega
    refine Or.inl ⟨{ outReg with data := _ }, { ctx with data := writeBytes ctx.data 208 (bytes8 (outLen + 8)) },
      stk, m', s', outCap, hreach, hrip', hm', hwf', by rw [← hm']; exact hbl', by rw [← hm']; exact hpos', hr', oracleAdmits_of_reaches hor hreach,
      by rw [hk'], hsp', Nat.le_refl _, hfit, hout_pg, hout', hout_rw,
      by simp only [Region.hi, writeBytes_length, hout_eq] at hout_hi ⊢; exact hout_hi,
      by simp [Nat.sub_self], hctx', rfl, rfl, writeBytes_length _ _ _,
      hc200', hc208', hc216', hrest, hstk', rfl, rfl, rfl,
      fun _ _ ho => by rw [hout_base]; exact ho,
      fun u hu ho hc _ => hoth' u hu ho hc,
      fun r h0 h1 _ h4 _ _ _ _ _ _ h12 => hregoth r h0 h1 h4 h12⟩

private theorem call_out_push_u64_grow_oracle_spec {s : X86Vm.St} {sp retAddr ctxBase outBase outLen outCap v : Nat}
    {ctx outReg stk : Region}
    (hrip : s.rip = SporosVmBin.fn_out_push_u64)
    (hregs : RegsOk s) (hor : OracleAdmits s)
    (hbl : BytesLt s.mem) (hpos : ∀ u ∈ s.mem, 0 < u.data.length)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr7 : s.reg 7 = some v)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_rw : ctx.rw = true)
    (hctx_len : ctxSize ≤ ctx.data.length)
    (hout : outReg ∈ s.mem) (hout_base : outReg.base = outBase) (hout_rw : outReg.rw = true)
    (hout_hi : outReg.hi ≤ 2 ^ 47)
    (hne : outBase ≠ ctxBase)
    (hstk : stk ∈ s.mem) (hstkrw : stk.rw = true) (hstkne_out : stk.base ≠ outBase) (hstkne_ctx : stk.base ≠ ctxBase)
    (h16 : 16 ≤ sp) (hspW : sp < X86Env.W)
    (hstk_lo : stk.base ≤ sp - 16) (hstk_hi : sp + 8 ≤ stk.hi)
    (hle : outLen ≤ outCap) (hcap_ge : 8 ≤ outCap)
    (hout_eq : outReg.data.length = outCap)
    (hout_pg : outCap % X86Env.page = 0)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W)
    (hmwf : s.mem.wf)
    (hret : rd64 s.mem sp = some retAddr)
    (hc200 : leVal ((ctx.data.drop 200).take 8) = outBase)
    (hc208 : leVal ((ctx.data.drop 208).take 8) = outLen)
    (hc216 : leVal ((ctx.data.drop 216).take 8) = outCap)
    (hfit : ¬ outLen + 8 ≤ outCap) :
    (∃ outReg' ctx' stk' m' s' outCap',
      Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.mem = m' ∧ m'.wf ∧ BytesLt m' ∧ (∀ u ∈ m', 0 < u.data.length) ∧
      RegsOk s' ∧ OracleAdmits s' ∧ s'.k.out = s.k.out ∧
      s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      outCap ≤ outCap' ∧ outLen + 8 ≤ outCap' ∧ outCap' % X86Env.page = 0 ∧
      outReg' ∈ m' ∧ outReg'.rw = true ∧ outReg'.hi ≤ 2 ^ 47 ∧
      outReg'.data = writeBytes (outReg.data ++ List.replicate (outCap' - outCap) 0) outLen (bytes8 v) ∧
      ctx' ∈ m' ∧ ctx'.base = ctx.base ∧ ctx'.rw = ctx.rw ∧ ctx'.data.length = ctx.data.length ∧
      leVal ((ctx'.data.drop 200).take 8) = outReg'.base ∧
      leVal ((ctx'.data.drop 208).take 8) = outLen + 8 ∧
      leVal ((ctx'.data.drop 216).take 8) = outCap' ∧
      (∀ j, (j < 200 ∨ 224 ≤ j) → ctx'.data[j]? = ctx.data[j]?) ∧
      stk' ∈ m' ∧ stk'.base = stk.base ∧ stk'.rw = stk.rw ∧ stk'.data.length = stk.data.length ∧
      (∀ u ∈ s.mem, u.base ≠ outBase → u.base ≠ outReg'.base) ∧
      (∀ u ∈ s.mem, u.base ≠ outBase → u.base ≠ ctxBase → u.base ≠ stk.base → u ∈ m') ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 4 → r ≠ 6 → r ≠ 7 → r ≠ 8 → r ≠ 9 → r ≠ 10 → r ≠ 11 → r ≠ 12 → s'.reg r = s.reg r)) ∨
    (HitsNomem s ∧ Ends vmImage s (.exit 2)) := by
    have hW : X86Env.W = 2 ^ 64 := rfl
    have houtCapW : outCap * 2 < X86Env.W := by
      simp only [Region.hi, hout_eq] at hout_hi; omega
    have hgt : outCap < outLen + 8 := Nat.lt_of_not_ge hfit
    have hle2 : outLen + 8 ≤ outCap * 2 := by omega
    have h01 : X86Vm.step vmImage s = .ok { s.set 12 (some v) with rip := 4201727 } := by
      rw [step_at_chunk 24 hrip rfl (by decide : (SporosVmBin.fn_out_push_u64, (.ofInt (.mov 12 7)), 3) ∈ SporosVmBin.vmTable_24),
        exec_mov hr7, hrip]; rfl
    let s1 := { s.set 12 (some v) with rip := 4201727 }
    have hr1 : RegsOk s1 := regsOk_set hregs (by decide) v
    have hs1_15 : s1.reg 15 = some ctxBase := by
      simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 12 ≠ 15)]; exact hr15
    have hcs : ctxSize = 4096 := rfl
    have hea_c200 : ea ctxBase 200 = ctxBase + 200 := ea_add_eq (by omega)
    have hea_c208 : ea ctxBase 208 = ctxBase + 208 := ea_add_eq (by omega)
    have hea_c216 : ea ctxBase 216 = ctxBase + 216 := ea_add_eq (by omega)
    have hrd_c200 : rd64 s1.mem (ea ctxBase 200) = some outBase := by
      rw [hea_c200]; exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
        (by simp only [Region.hi, hctx_base]; omega) (by rw [hctx_base, Nat.add_sub_cancel_left]; exact hc200)
    have hrd_c208 : rd64 s1.mem (ea ctxBase 208) = some outLen := by
      rw [hea_c208]; exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
        (by simp only [Region.hi, hctx_base]; omega) (by rw [hctx_base, Nat.add_sub_cancel_left]; exact hc208)
    have hrd_c216 : rd64 s1.mem (ea ctxBase 216) = some outCap := by
      rw [hea_c216]; exact rd64_of_mem hmwf hctx (by rw [hctx_base]; omega)
        (by simp only [Region.hi, hctx_base]; omega) (by rw [hctx_base, Nat.add_sub_cancel_left]; exact hc216)
    have h02 : X86Vm.step vmImage s1 = .ok { s1.set 0 (some outLen) with rip := 4201734 } := by
      rw [step_at_chunk 24 rfl rfl (by decide : (4201727, (.ofInt (.ld 0 15 208)), 7) ∈ SporosVmBin.vmTable_24),
        exec_ld hs1_15 hrd_c208]
    let s2 := { s1.set 0 (some outLen) with rip := 4201734 }
    have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) outLen
    have hs2_0 : s2.reg 0 = some outLen := by
      simp only [s2, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
    have h03 : X86Vm.step vmImage s2 =
        .ok { s2.set 0 (some (outLen + 8)) with fl := (X86Int.aluF .add outLen 8).2, rip := 4201738 } := by
      rw [step_at_chunk 24 rfl rfl (by decide : (4201734, (.ofInt (.aluI .add 0 8)), 4) ∈ SporosVmBin.vmTable_24),
        exec_aluI_add_lt hs2_0 (by decide), Nat.mod_eq_of_lt (by omega)]
    let s3 := { s2.set 0 (some (outLen + 8)) with fl := (X86Int.aluF .add outLen 8).2, rip := 4201738 }
    have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) (outLen + 8)
    have hs3_15 : s3.reg 15 = some ctxBase := by
      simp only [s3, s2, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 0 ≠ 15)]; exact hs1_15
    have h04 : X86Vm.step vmImage s3 = .ok { s3.set 1 (some outCap) with rip := 4201745 } := by
      rw [step_at_chunk 24 rfl rfl (by decide : (4201738, (.ofInt (.ld 1 15 216)), 7) ∈ SporosVmBin.vmTable_24),
        exec_ld hs3_15 hrd_c216]
    let s4 := { s3.set 1 (some outCap) with rip := 4201745 }
    have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) outCap
    have hs4_0 : s4.reg 0 = some (outLen + 8) := by
      simp only [s4, s3, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0)]
      rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
    have hs4_1 : s4.reg 1 = some outCap := by
      simp only [s4, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
    have h05 : X86Vm.step vmImage s4 = .ok { s4 with fl := (X86Int.aluF .sub (outLen + 8) outCap).2, rip := 4201748 } := by
      rw [step_at_chunk 24 rfl rfl (by decide : (4201745, (.ofInt (.cmp 0 1)), 3) ∈ SporosVmBin.vmTable_24),
        exec_cmp hs4_0 hs4_1]
    let s5 := { s4 with fl := (X86Int.aluF .sub (outLen + 8) outCap).2, rip := 4201748 }
    have h06 : X86Vm.step vmImage s5 = .ok { s5 with rip := 4201750 } := by
      rw [step_at_chunk 24 rfl rfl (by decide : (4201748, (.ofInt (.jcc 6 4201799)), 2) ∈ SporosVmBin.vmTable_24),
        exec_jcc (cond_aluF_sub_be (outLen + 8) outCap (by omega) (by omega))]
      simp [s5, hfit]
    let s6 := { s5 with rip := 4201750 }
    have hs6_15 : s6.reg 15 = some ctxBase := by
      change s4.reg 15 = some ctxBase
      simp only [s4, s3, s2, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 1 ≠ 15), List.getElem?_set_ne (by decide : 0 ≠ 15)]; exact hs1_15
    have h07 : X86Vm.step vmImage s6 = .ok { s6.set 7 (some outBase) with rip := 4201757 } := by
      rw [step_at_chunk 25 rfl rfl (by decide : (4201750, (.ofInt (.ld 7 15 200)), 7) ∈ SporosVmBin.vmTable_25),
        exec_ld hs6_15 hrd_c200]
    let s7 := { s6.set 7 (some outBase) with rip := 4201757 }
    have hr7' : RegsOk s7 := regsOk_set hr4 (by decide) outBase
    have hs7_1 : s7.reg 1 = some outCap := by
      simp only [s7, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 7 ≠ 1)]; exact hs4_1
    have h08 : X86Vm.step vmImage s7 = .ok { s7.set 6 (some outCap) with rip := 4201760 } := by
      rw [step_at_chunk 25 rfl rfl (by decide : (4201757, (.ofInt (.mov 6 1)), 3) ∈ SporosVmBin.vmTable_25),
        exec_mov hs7_1]
    let s8 := { s7.set 6 (some outCap) with rip := 4201760 }
    have hr8 : RegsOk s8 := regsOk_set hr7' (by decide) outCap
    have hs8_1 : s8.reg 1 = some outCap := by
      simp only [s8, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 6 ≠ 1)]; exact hs7_1
    have h09 : X86Vm.step vmImage s8 = .ok { s8.set 2 (some outCap) with rip := 4201763 } := by
      rw [step_at_chunk 25 rfl rfl (by decide : (4201760, (.ofInt (.mov 2 1)), 3) ∈ SporosVmBin.vmTable_25),
        exec_mov hs8_1]
    let s9 := { s8.set 2 (some outCap) with rip := 4201763 }
    have hr9 : RegsOk s9 := regsOk_set hr8 (by decide) outCap
    have hs9_2 : s9.reg 2 = some outCap := by
      simp only [s9, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr8.1]; decide)]
    have h10 : X86Vm.step vmImage s9 =
        .ok { s9.set 2 (some (outCap * 2)) with fl := X86Int.Flags.undef, rip := 4201767 } := by
      rw [step_at_chunk 25 rfl rfl (by decide : (4201763, (.ofInt (.shlI 2 1)), 4) ∈ SporosVmBin.vmTable_25),
        exec_shlI_1_lt hs9_2 houtCapW]
    let s10 := { s9.set 2 (some (outCap * 2)) with fl := X86Int.Flags.undef, rip := 4201767 }
    have hr10 : RegsOk s10 := regsOk_set hr9 (by decide) (outCap * 2)
    have hs10_4 : s10.reg 4 = some sp := by
      simp only [s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 2 ≠ 4), List.getElem?_set_ne (by decide : 6 ≠ 4),
        List.getElem?_set_ne (by decide : 7 ≠ 4), List.getElem?_set_ne (by decide : 1 ≠ 4),
        List.getElem?_set_ne (by decide : 0 ≠ 4), List.getElem?_set_ne (by decide : 12 ≠ 4)]
      exact hsp
    have hea_sp : (sp + X86Env.W - 8) % X86Env.W = sp - 8 := by
      have : sp + X86Env.W - 8 = (sp - 8) + X86Env.W := by omega
      rw [this, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
    obtain ⟨_, m11, hwr11, _, _, hwf11, hstk11, hoth11⟩ :=
      wr64_region hmwf hstk hstkrw (by omega) (by omega : sp - 8 + 8 ≤ stk.hi) (v := 4201772)
    have hwr11' : wr64 s10.mem ((sp + X86Env.W - 8) % X86Env.W) (4201767 + 5) = some m11 := by
      simpa [s10, s9, s8, s7, s6, s5, s4, s3, s2, s1, hea_sp] using hwr11
    have h11 : X86Vm.step vmImage s10 =
        .ok { s10.set 4 (some (sp - 8)) with mem := m11, rip := SporosVmBin.fn_mremap } := by
      rw [step_at_chunk 25 rfl rfl (by decide : (4201767, (.ofInt (.call 4201363)), 5) ∈ SporosVmBin.vmTable_25),
        exec_call hs10_4 hwr11']
      simp [s10, hea_sp, SporosVmBin.fn_mremap]
    let s11 := { s10.set 4 (some (sp - 8)) with mem := m11, rip := SporosVmBin.fn_mremap }
    have hr11 : RegsOk s11 := regsOk_set hr10 (by decide) (sp - 8)
    have hreach11 : Reaches vmImage s s11 :=
      Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
        (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step_trans h08
          (Reaches.step_trans h09 (Reaches.step_trans h10 (Reaches.step h11))))))))))
    have hout11 : outReg ∈ s11.mem := hoth11 outReg hout (by rw [hout_base]; exact Ne.symm hstkne_out)
    have hout11_hi : outBase < outReg.hi := by simp only [Region.hi, hout_base, hout_eq]; omega
    obtain ⟨idx, hfind11⟩ := mem_find_of_mem s11.mem outReg outBase 0 hwf11.2 hout11 (by rw [hout_base]; exact Nat.le_refl _) hout11_hi
    have hs11_7 : s11.reg 7 = some outBase := by
      simp only [s11, s10, s9, s8, s7, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 4 ≠ 7), List.getElem?_set_ne (by decide : 2 ≠ 7),
        List.getElem?_set_ne (by decide : 6 ≠ 7)]
      rw [getElem?_set_getD_self (by rw [hr4.1]; decide)]
    have hs11_6 : s11.reg 6 = some outCap := by
      simp only [s11, s10, s9, s8, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 4 ≠ 6), List.getElem?_set_ne (by decide : 2 ≠ 6)]
      rw [getElem?_set_getD_self (by rw [hr7'.1]; decide)]
    have hs11_2 : s11.reg 2 = some (outCap * 2) := by
      simp only [s11, s10, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 4 ≠ 2)]
      rw [getElem?_set_getD_self (by rw [hr9.1]; decide)]
    have hpg1 : X86Env.pages outCap = outCap := pages_of_dvd hout_pg
    have hdvd2 : (outCap * 2) % X86Env.page = 0 := by
      have hmul : outCap * 2 = 2 * outCap := by omega
      rw [hmul, Nat.mul_mod, hout_pg]; simp
    have hpg2 : X86Env.pages (outCap * 2) = outCap * 2 := pages_of_dvd hdvd2
    rcases at_mremap_oracle_spec rfl hr11 (oracleAdmits_of_reaches hor hreach11)
        hs11_7 hs11_6 hs11_2 hfind11 hout_base (by rw [hout_eq, hpg1]) hout_rw
        (by rw [hout_eq, hpg2]; omega) with ⟨hnomem, hhit11⟩ | ⟨a, hat, ha_pg, ha_pos, ha_lim, hfree⟩
    · have hs11_k : s11.k = s.k := rfl
      rw [hs11_k] at hnomem
      exact Or.inr ⟨hitsNomem_of_reaches hreach11 hhit11,
        fn_out_push_u64_nomem_spec hrip hregs hsp hr15 hr7 hctx hctx_base hctx_len
          hout hout_base hout_rw hstk hstkrw hstkne_out h16 hspW hstk_lo (by omega) hgt hle2 (by omega)
          hout_eq hout_pg houtCapW hctxW hmwf hc200 hc208 hc216 hnomem⟩
    · have hs11_k : s11.k = s.k := rfl
      rw [hs11_k] at hat
      rw [hpg2] at ha_lim hfree
      rw [Nat.zero_add] at hfind11 hfree
      have hget11 : m11[idx]? = some outReg := by
        simpa using (mem_find_idx_lt s11.mem outBase 0 idx outReg hfind11).2.2
      change X86Env.Mem.free (m11.eraseIdx idx) a (outCap * 2) = true at hfree
      have hstk11_b : { stk with data := writeBytes stk.data (sp - 8 - stk.base) (bytes8 4201772) }.base = stk.base := rfl
      have hstk11_hi : { stk with data := writeBytes stk.data (sp - 8 - stk.base) (bytes8 4201772) }.hi = stk.hi := by
        simp [Region.hi, writeBytes_length]
      obtain ⟨hastk, hfree_all⟩ := mremap_find_after_stack_write (n := outCap * 2) hmwf hstk hstk11_b hstk11_hi
        hstkne_out hstk11 hoth11 hget11 hout_base hfree (by omega) hpos
      have hane : a ≠ ctxBase := by
        rw [← hctx_base]
        exact mem_free_ne_base (by omega) (hpos ctx hctx) (hfree_all ctx hctx (by rw [hctx_base]; exact Ne.symm hne))
      obtain ⟨m', s', stk', hreach_g, hrip', hk', hm', hwf', hr', hsp', hout', hctx', hstk', hstk'_b, hstk'_rw, hstk'_len, _, hoth', hregoth⟩ :=
        fn_out_push_u64_spec hrip hregs hsp hr15 hr7 hctx hctx_base hctx_rw hctx_len
          hout hout_base hout_rw hne hane hastk hstk hstkrw hstkne_out hstkne_ctx (by omega) hspW
          (by omega) hstk_hi hgt hle2 (by omega) hout_eq hout_pg houtCapW hctxW hmwf hpos hret
          hc200 hc208 hc216 hat ha_pg ha_pos ha_lim hfree_all
      have ⟨hbl', hpos'⟩ := reaches_bytesLt_len_pos hbl hpos hreach_g
      let ctxData' := writeBytes (writeBytes (writeBytes ctx.data 200 (bytes8 a)) 216 (bytes8 (outCap * 2))) 208 (bytes8 (outLen + 8))
      let outData' := writeBytes (outReg.data ++ List.replicate outCap 0) outLen (bytes8 v)
      have hcs' : ({ ctx with data := ctxData' } : Region).data.length = ctx.data.length := by
        dsimp only [ctxData']; rw [writeBytes_length, writeBytes_length, writeBytes_length]
      have hc200' : leVal (({ ctx with data := ctxData' }.data.drop 200).take 8) = a := by
        dsimp only [ctxData']
        rw [writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inl (by decide)) (by rw [writeBytes_length, writeBytes_length]; omega),
            writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inl (by decide)) (by rw [writeBytes_length]; omega),
            leVal_writeBytes_same _ _ _ (by omega), Nat.mod_eq_of_lt (by omega)]
      have hc208' : leVal (({ ctx with data := ctxData' }.data.drop 208).take 8) = outLen + 8 := by
        dsimp only [ctxData']
        rw [leVal_writeBytes_same _ _ _ (by rw [writeBytes_length, writeBytes_length]; omega),
            Nat.mod_eq_of_lt (by omega)]
      have hc216' : leVal (({ ctx with data := ctxData' }.data.drop 216).take 8) = outCap * 2 := by
        dsimp only [ctxData']
        rw [writeBytes_drop_take_disjoint _ _ _ _ _ (Or.inr (by rw [bytes8_length]; decide)) (by rw [writeBytes_length, writeBytes_length]; omega),
            leVal_writeBytes_same _ _ _ (by rw [writeBytes_length]; omega), Nat.mod_eq_of_lt (by omega)]
      have hrest : ∀ j, (j < 200 ∨ 224 ≤ j) → ({ ctx with data := ctxData' } : Region).data[j]? = ctx.data[j]? := by
        intro j hj
        dsimp only [ctxData']
        rw [writeBytes_getElem?_disjoint _ _ _ _ (by rw [bytes8_length]; omega),
            writeBytes_getElem?_disjoint _ _ _ _ (by rw [bytes8_length]; omega),
            writeBytes_getElem?_disjoint _ _ _ _ (by rw [bytes8_length]; omega)]
      have hk_out : s'.k.out = s.k.out := by
        rw [hk']; cases horac : s.k.oracle <;> simp [X86Env.Kern.next, horac]
      have hout_hi' : (⟨a, outData', true⟩ : Region).hi ≤ 2 ^ 47 := by
        simp [outData', Region.hi, writeBytes_length, List.length_append, List.length_replicate, hout_eq]; omega
      have hout_data' : outData' = writeBytes (outReg.data ++ List.replicate (outCap * 2 - outCap) 0) outLen (bytes8 v) := by
        rw [show outCap * 2 - outCap = outCap by omega]
      refine Or.inl ⟨⟨a, outData', true⟩, { ctx with data := ctxData' }, stk', m', s', outCap * 2,
        hreach_g, hrip', hm', hwf', by rw [← hm']; exact hbl', by rw [← hm']; exact hpos', hr',
        oracleAdmits_of_reaches hor hreach_g, hk_out, hsp', by omega, by omega, hdvd2, hout', rfl,
        hout_hi', hout_data', hctx', rfl, rfl, ?_⟩
      refine ⟨?_, ?_, ?_, ?_, ?_, hstk', hstk'_b, by rw [hstk'_rw, hstkrw], hstk'_len,
        fun u hu ho => Ne.symm (mem_free_ne_base (by omega) (hpos u hu) (hfree_all u hu ho)), hoth', hregoth⟩
      · exact hcs'
      · exact hc200'
      · exact hc208'
      · exact hc216'
      · exact hrest

theorem call_out_push_u64_oracle_spec {s : X86Vm.St} {sp retAddr ctxBase outBase outLen outCap v : Nat}
    {ctx outReg stk : Region}
    (hrip : s.rip = SporosVmBin.fn_out_push_u64)
    (hregs : RegsOk s) (hor : OracleAdmits s)
    (hbl : BytesLt s.mem) (hpos : ∀ u ∈ s.mem, 0 < u.data.length)
    (hsp : s.reg 4 = some sp)
    (hr15 : s.reg 15 = some ctxBase) (hr7 : s.reg 7 = some v)
    (hctx : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase) (hctx_rw : ctx.rw = true)
    (hctx_len : ctxSize ≤ ctx.data.length)
    (hout : outReg ∈ s.mem) (hout_base : outReg.base = outBase) (hout_rw : outReg.rw = true)
    (hout_hi : outReg.hi ≤ 2 ^ 47)
    (hne : outBase ≠ ctxBase)
    (hstk : stk ∈ s.mem) (hstkrw : stk.rw = true) (hstkne_out : stk.base ≠ outBase) (hstkne_ctx : stk.base ≠ ctxBase)
    (h16 : 16 ≤ sp) (hspW : sp < X86Env.W)
    (hstk_lo : stk.base ≤ sp - 16) (hstk_hi : sp + 8 ≤ stk.hi)
    (hle : outLen ≤ outCap) (hcap_ge : 8 ≤ outCap)
    (hout_eq : outReg.data.length = outCap)
    (hout_pg : outCap % X86Env.page = 0)
    (hctxW : ctxBase + ctxSize ≤ X86Env.W)
    (hmwf : s.mem.wf)
    (hret : rd64 s.mem sp = some retAddr)
    (hc200 : leVal ((ctx.data.drop 200).take 8) = outBase)
    (hc208 : leVal ((ctx.data.drop 208).take 8) = outLen)
    (hc216 : leVal ((ctx.data.drop 216).take 8) = outCap) :
    (∃ outReg' ctx' stk' m' s' outCap',
      Reaches vmImage s s' ∧
      s'.rip = retAddr ∧ s'.mem = m' ∧ m'.wf ∧ BytesLt m' ∧ (∀ u ∈ m', 0 < u.data.length) ∧
      RegsOk s' ∧ OracleAdmits s' ∧ s'.k.out = s.k.out ∧
      s'.reg 4 = some ((sp + 8) % X86Env.W) ∧
      outCap ≤ outCap' ∧ outLen + 8 ≤ outCap' ∧ outCap' % X86Env.page = 0 ∧
      outReg' ∈ m' ∧ outReg'.rw = true ∧ outReg'.hi ≤ 2 ^ 47 ∧
      outReg'.data = writeBytes (outReg.data ++ List.replicate (outCap' - outCap) 0) outLen (bytes8 v) ∧
      ctx' ∈ m' ∧ ctx'.base = ctx.base ∧ ctx'.rw = ctx.rw ∧ ctx'.data.length = ctx.data.length ∧
      leVal ((ctx'.data.drop 200).take 8) = outReg'.base ∧
      leVal ((ctx'.data.drop 208).take 8) = outLen + 8 ∧
      leVal ((ctx'.data.drop 216).take 8) = outCap' ∧
      (∀ j, (j < 200 ∨ 224 ≤ j) → ctx'.data[j]? = ctx.data[j]?) ∧
      stk' ∈ m' ∧ stk'.base = stk.base ∧ stk'.rw = stk.rw ∧ stk'.data.length = stk.data.length ∧
      (∀ u ∈ s.mem, u.base ≠ outBase → u.base ≠ outReg'.base) ∧
      (∀ u ∈ s.mem, u.base ≠ outBase → u.base ≠ ctxBase → u.base ≠ stk.base → u ∈ m') ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 4 → r ≠ 6 → r ≠ 7 → r ≠ 8 → r ≠ 9 → r ≠ 10 → r ≠ 11 → r ≠ 12 → s'.reg r = s.reg r)) ∨
    (HitsNomem s ∧ Ends vmImage s (.exit 2)) := by
  by_cases hfit : outLen + 8 ≤ outCap
  · exact call_out_push_u64_nogrow_oracle_spec hrip hregs hor hbl hpos hsp hr15 hr7 hctx hctx_base
      hctx_rw hctx_len hout hout_base hout_rw hout_hi hne hstk hstkrw hstkne_out hstkne_ctx h16 hspW
      hstk_lo hstk_hi hle hcap_ge hout_eq hout_pg hctxW hmwf hret hc200 hc208 hc216 hfit
  · exact call_out_push_u64_grow_oracle_spec hrip hregs hor hbl hpos hsp hr15 hr7 hctx hctx_base
      hctx_rw hctx_len hout hout_base hout_rw hout_hi hne hstk hstkrw hstkne_out hstkne_ctx h16 hspW
      hstk_lo hstk_hi hle hcap_ge hout_eq hout_pg hctxW hmwf hret hc200 hc208 hc216 hfit

/-! ### Additional `VmCtx` extraction & scratch-slot helpers -/

namespace VmCtx

theorem arena_bounds {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) :
    (vc.dsReg.rw = true ∧ 4 ≤ vc.dsCap ∧ vc.dsReg.data.length = vc.dsCap * 16 ∧
      (vc.dsCap * 16) % X86Env.page = 0 ∧ vc.dsReg.hi ≤ 2 ^ 47 ∧
      PairListRel vc.dsReg.data ls.ds.length ls.ds ∧ ls.ds.length ≤ vc.dsCap) ∧
    (vc.auxReg.rw = true ∧ 4 ≤ vc.auxCap ∧ vc.auxReg.data.length = vc.auxCap * 16 ∧
      (vc.auxCap * 16) % X86Env.page = 0 ∧ vc.auxReg.hi ≤ 2 ^ 47 ∧
      PairListRel vc.auxReg.data ls.aux.length ls.aux ∧ ls.aux.length ≤ vc.auxCap) ∧
    (vc.rsReg.rw = true ∧ 0 < vc.rsCap ∧ vc.rsReg.data.length = vc.rsCap * 32 ∧
      (vc.rsCap * 32) % X86Env.page = 0 ∧ vc.rsReg.hi ≤ 2 ^ 47 ∧
      vc.locReg.rw = true ∧ 0 < vc.locCap ∧ vc.locReg.data.length = vc.locCap * 16 ∧
      (vc.locCap * 16) % X86Env.page = 0 ∧ vc.locReg.hi ≤ 2 ^ 47 ∧
      RsLocRel vc.rsReg.data vc.locReg.data (rsLocBase ls.rs) ls.rs ∧
      ls.rs.length ≤ vc.rsCap ∧ rsLocBase ls.rs ≤ vc.locCap) := by
  rcases h with ⟨_, _, _, _, _, _, _, _,
    _, _, _, _,
    _, _, _, _,
    _, _, _,
    hd1, hd2, hd3, hd4, hd5,
    ha1, ha2, ha3, ha4, ha5,
    hr1, hr2, hr3, hr4, hr5,
    hl1, hl2, hl3, hl4, hl5,
    _, _, _, _,
    _, _, _,
    _, _, _, _, _,
    _, _, _, _, _, _,
    _,
    _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    hpds, hdsle, hpaux, hauxle, hrsrel, hrsle, hlocle, _⟩
  exact ⟨⟨hd1, hd2, hd3, hd4, hd5, hpds, hdsle⟩,
    ⟨ha1, ha2, ha3, ha4, ha5, hpaux, hauxle⟩,
    ⟨hr1, hr2, hr3, hr4, hr5, hl1, hl2, hl3, hl4, hl5, hrsrel, hrsle, hlocle⟩⟩

theorem code_flat {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) :
    FlatCodeOk pr vc.code1 ∧ FlatCodeOk pr vc.code2 ∧ FlatCodeOk pr vc.code3 ∧
    FlatProgFit pr ∧ ls.WordsLt := by
  rcases h with ⟨_, _, _, _, _, _, _, _,
    _, _, _, _,
    _, _, _, _,
    hf1, hf2, hf3,
    _, _, _, _, _,
    _, _, _, _, _,
    _, _, _, _, _,
    _, _, _, _, _,
    _, _, _, _,
    _, _, _,
    _, _, _, _, _,
    _, _, _, _, _, _,
    _,
    _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, hwlt, hfit⟩
  exact ⟨hf1, hf2, hf3, hfit, hwlt⟩

private theorem wf_ctx_slots {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) :
    leVal ((vc.ctxReg.data.drop 0).take 8) = vc.dsReg.base ∧
    leVal ((vc.ctxReg.data.drop 8).take 8) = ls.ds.length ∧
    leVal ((vc.ctxReg.data.drop 16).take 8) = vc.dsCap ∧
    leVal ((vc.ctxReg.data.drop 48).take 8) = vc.rsReg.base ∧
    leVal ((vc.ctxReg.data.drop 56).take 8) = ls.rs.length ∧
    leVal ((vc.ctxReg.data.drop 64).take 8) = vc.rsCap ∧
    leVal ((vc.ctxReg.data.drop 224).take 8) = ls.pc ∧
    leVal ((vc.ctxReg.data.drop 232).take 8) = vc.code1.base ∧
    leVal ((vc.ctxReg.data.drop 240).take 8) = vc.code2.base ∧
    leVal ((vc.ctxReg.data.drop 248).take 8) = vc.code3.base := by
  obtain ⟨_, _, _, _, _, _, _, _,
    _, _, _, _,  _, _, _, _,  _, _, _,
    _, _, _, _, _,  _, _, _, _, _,  _, _, _, _, _,  _, _, _, _, _,
    _, _, _, _,  _, _, _,  _, _, _, _, _,  _, _, _, _, _, _,  _,  _, _, _, _, _,
    s0, s8, s16, _, _, _, s48, s56, s64, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, s224, s232, s240, s248, _⟩ := h
  exact ⟨s0, s8, s16, s48, s56, s64, s224, s232, s240, s248⟩

theorem rd_ctx_dsBase {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : rd64 s.mem (ea ctxBase 0) = some vc.dsReg.base :=
  rd64_ctx_of_eq h (by decide) (wf_ctx_slots h).1

theorem rd_ctx_dsLen {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : rd64 s.mem (ea ctxBase 8) = some ls.ds.length :=
  rd64_ctx_of_eq h (by decide) (wf_ctx_slots h).2.1

theorem rd_ctx_dsCap {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : rd64 s.mem (ea ctxBase 16) = some vc.dsCap :=
  rd64_ctx_of_eq h (by decide) (wf_ctx_slots h).2.2.1

theorem rd_ctx_rs_code_slots {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) :
    rd64 s.mem (ea ctxBase 48) = some vc.rsReg.base ∧
    rd64 s.mem (ea ctxBase 56) = some ls.rs.length ∧
    rd64 s.mem (ea ctxBase 64) = some vc.rsCap ∧
    rd64 s.mem (ea ctxBase 224) = some ls.pc ∧
    rd64 s.mem (ea ctxBase 232) = some vc.code1.base ∧
    rd64 s.mem (ea ctxBase 240) = some vc.code2.base ∧
    rd64 s.mem (ea ctxBase 248) = some vc.code3.base := by
  have ⟨_, _, _, s48, s56, s64, s224, s232, s240, s248⟩ := wf_ctx_slots h
  exact ⟨rd64_ctx_of_eq h (by decide) s48,
    rd64_ctx_of_eq h (by decide) s56,
    rd64_ctx_of_eq h (by decide) s64,
    rd64_ctx_of_eq h (by decide) s224,
    rd64_ctx_of_eq h (by decide) s232,
    rd64_ctx_of_eq h (by decide) s240,
    rd64_ctx_of_eq h (by decide) s248⟩

theorem wr64_ctx_scratch_slot {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} (h : vc.wf pr ls s) (off v rip' : Nat)
    (hoff : 256 ≤ off) (hlim : off + 8 ≤ ctxSize)
    {r : Nat} (hr : s.reg r = some v)
    (hstep : ∀ {m' : Mem},
      wr64 s.mem (ea ctxBase off) v = some m' →
      X86Vm.step vmImage s = .ok { s with mem := m', rip := rip' }) :
    let ctx' := { vc.ctxReg with data := writeBytes vc.ctxReg.data off (bytes8 v) }
    ∃ idx m',
      let s' := { s with mem := m', rip := rip' }
      s.mem[idx]? = some vc.ctxReg ∧
      m' = s.mem.set idx ctx' ∧
      X86Vm.step vmImage s = .ok s' ∧
      { vc with ctxReg := ctx' }.wf pr ls s' ∧
      leVal ((ctx'.data.drop off).take 8) = v % X86Env.W ∧
      (∀ k, (k + 8 ≤ off ∨ off + 8 ≤ k) → k + 8 ≤ ctxSize →
        leVal ((ctx'.data.drop k).take 8) = leVal ((vc.ctxReg.data.drop k).take 8)) := by
  obtain ⟨idx, m', hwr, hidx, hm', hcm', hbl', hpos', hval⟩ := wr64_ctx_scratch h off v hoff hlim
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have _ := hr
  have hwf' : { vc with ctxReg := { vc.ctxReg with data := writeBytes vc.ctxReg.data off (bytes8 v) } }.wf
      pr ls { s with mem := m', rip := rip' } :=
    wf_write_ctx_scratch h hoff hidx hm' hcm' hbl' hpos' h.2.2.2.1 h.2.2.2.2.1 h15
  refine ⟨idx, m', hidx, hm', hstep hwr, hwf', hval, ?_⟩
  intro k hk hklim
  have hcl := (ctx_bounds h).2.2.1
  rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; exact hk) (by omega)]

end VmCtx

/-! ### TMR inline voter & loop fetch/dispatch simulation -/

theorem tmr_inline_vote_eq_spec {a done v : Nat} {s : X86Vm.St}
    (hrip : s.rip = a)
    (hm1 : (a, (.ofInt (.cmp 0 3)), 3) ∈ SporosVmBin.vmTable)
    (hm2 : (a + 3, (.ofInt (.jcc 4 done)), 2) ∈ SporosVmBin.vmTable)
    (h0 : s.reg 0 = some v) (h3 : s.reg 3 = some v) (hv : v < X86Env.W) :
    Reaches vmImage s { s with fl := (X86Int.aluF .sub v v).2, rip := done } := by
  have h01 : X86Vm.step vmImage s =
      .ok { s with fl := (X86Int.aluF .sub v v).2, rip := a + 3 } := by
    have hm1' : (s.rip, (.ofInt (.cmp 0 3)), 3) ∈ SporosVmBin.vmTable := by rwa [hrip]
    rw [step_of_table hm1', exec_cmp h0 h3, hrip]
  let s1 := { s with fl := (X86Int.aluF .sub v v).2, rip := a + 3 }
  have h02 : X86Vm.step vmImage s1 = .ok { s1 with rip := done } := by
    have hm2' : (s1.rip, (.ofInt (.jcc 4 done)), 2) ∈ SporosVmBin.vmTable := hm2
    rw [step_of_table hm2', exec_jcc (cond_aluF_sub_e v v hv hv)]
    simp [s1]
  exact Reaches.step_trans h01 (Reaches.step h02)

theorem tmr_ld_vote_eq_spec {a done disp c1 c2 c3 v : Nat} {s : X86Vm.St}
    (hrip : s.rip = a) (hr : RegsOk s) (h15 : s.reg 15 = some ctxBase)
    (hm1 : (a, (.ofInt (.ld 2 15 232)), 7) ∈ SporosVmBin.vmTable)
    (hm2 : (a + 7, (.ofInt (.ld 0 2 disp)), 4) ∈ SporosVmBin.vmTable)
    (hm3 : (a + 11, (.ofInt (.ld 2 15 240)), 7) ∈ SporosVmBin.vmTable)
    (hm4 : (a + 18, (.ofInt (.ld 3 2 disp)), 4) ∈ SporosVmBin.vmTable)
    (hm5 : (a + 22, (.ofInt (.ld 2 15 248)), 7) ∈ SporosVmBin.vmTable)
    (hm6 : (a + 29, (.ofInt (.ld 1 2 disp)), 4) ∈ SporosVmBin.vmTable)
    (hm7 : (a + 33, (.ofInt (.cmp 0 3)), 3) ∈ SporosVmBin.vmTable)
    (hm8 : (a + 36, (.ofInt (.jcc 4 done)), 2) ∈ SporosVmBin.vmTable)
    (hc1 : rd64 s.mem (ea ctxBase 232) = some c1)
    (hc2 : rd64 s.mem (ea ctxBase 240) = some c2)
    (hc3 : rd64 s.mem (ea ctxBase 248) = some c3)
    (hv1 : rd64 s.mem (ea c1 disp) = some v)
    (hv2 : rd64 s.mem (ea c2 disp) = some v)
    (hv3 : rd64 s.mem (ea c3 disp) = some v)
    (hv : v < X86Env.W) :
    ∃ s', Reaches vmImage s s' ∧
      s'.rip = done ∧ s'.mem = s.mem ∧ s'.k = s.k ∧ RegsOk s' ∧
      s'.reg 0 = some v ∧ s'.reg 1 = some v ∧ s'.reg 3 = some v ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 3 → s'.reg r = s.reg r) := by
  have h01 : X86Vm.step vmImage s = .ok { s.set 2 (some c1) with rip := a + 7 } := by
    have hm1' : (s.rip, (.ofInt (.ld 2 15 232)), 7) ∈ SporosVmBin.vmTable := by rwa [hrip]
    rw [step_of_table hm1', exec_ld h15 hc1, hrip]
  let s1 := { s.set 2 (some c1) with rip := a + 7 }
  have hr1 : RegsOk s1 := regsOk_set hr (by decide) c1
  have hs1_2 : s1.reg 2 = some c1 := by
    simp only [s1, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr.1]; decide)]
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 0 (some v) with rip := a + 11 } := by
    rw [step_of_table hm2, exec_ld hs1_2 hv1]
  let s2 := { s1.set 0 (some v) with rip := a + 11 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) v
  have hs2_15 : s2.reg 15 = some ctxBase := by
    simp only [s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 15), List.getElem?_set_ne (by decide : 2 ≠ 15)]; exact h15
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 2 (some c2) with rip := a + 18 } := by
    rw [step_of_table hm3, exec_ld hs2_15 hc2]
  let s3 := { s2.set 2 (some c2) with rip := a + 18 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) c2
  have hs3_2 : s3.reg 2 = some c2 := by
    simp only [s3, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
  have h04 : X86Vm.step vmImage s3 = .ok { s3.set 3 (some v) with rip := a + 22 } := by
    rw [step_of_table hm4, exec_ld hs3_2 hv2]
  let s4 := { s3.set 3 (some v) with rip := a + 22 }
  have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) v
  have hs4_15 : s4.reg 15 = some ctxBase := by
    simp only [s4, s3, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 3 ≠ 15), List.getElem?_set_ne (by decide : 2 ≠ 15)]; exact hs2_15
  have h05 : X86Vm.step vmImage s4 = .ok { s4.set 2 (some c3) with rip := a + 29 } := by
    rw [step_of_table hm5, exec_ld hs4_15 hc3]
  let s5 := { s4.set 2 (some c3) with rip := a + 29 }
  have hr5 : RegsOk s5 := regsOk_set hr4 (by decide) c3
  have hs5_2 : s5.reg 2 = some c3 := by
    simp only [s5, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr4.1]; decide)]
  have h06 : X86Vm.step vmImage s5 = .ok { s5.set 1 (some v) with rip := a + 33 } := by
    rw [step_of_table hm6, exec_ld hs5_2 hv3]
  let s6 := { s5.set 1 (some v) with rip := a + 33 }
  have hr6 : RegsOk s6 := regsOk_set hr5 (by decide) v
  have hs6_0 : s6.reg 0 = some v := by
    simp only [s6, s5, s4, s3, s2, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 0), List.getElem?_set_ne (by decide : 2 ≠ 0),
      List.getElem?_set_ne (by decide : 3 ≠ 0)]
    rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  have hs6_3 : s6.reg 3 = some v := by
    simp only [s6, s5, s4, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 3), List.getElem?_set_ne (by decide : 2 ≠ 3)]
    rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
  have hs6_1 : s6.reg 1 = some v := by
    simp only [s6, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr5.1]; decide)]
  have hvote := tmr_inline_vote_eq_spec rfl hm7 hm8 hs6_0 hs6_3 hv
  let s7 := { s6 with fl := (X86Int.aluF .sub v v).2, rip := done }
  have hreach : Reaches vmImage s s7 :=
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03
      (Reaches.step_trans h04 (Reaches.step_trans h05 (Reaches.step_trans h06 hvote)))))
  refine ⟨s7, hreach, rfl, rfl, rfl, hr6, hs6_0, hs6_1, hs6_3, ?_⟩
  intro r h0 h1 h2 h3
  simp only [s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
    List.getElem?_set_ne (Ne.symm h1), List.getElem?_set_ne (Ne.symm h2),
    List.getElem?_set_ne (Ne.symm h3), List.getElem?_set_ne (Ne.symm h0)]

theorem tmr_ldIdx_vote_eq_spec {a done disp c1 c2 c3 r6v v : Nat} {s : X86Vm.St}
    (hrip : s.rip = a) (hr : RegsOk s) (h15 : s.reg 15 = some ctxBase) (hr6 : s.reg 6 = some r6v)
    (hm1 : (a, (.ofInt (.ld 2 15 232)), 7) ∈ SporosVmBin.vmTable)
    (hm2 : (a + 7, (.ldIdx .w64 0 2 6 1 disp), 5) ∈ SporosVmBin.vmTable)
    (hm3 : (a + 12, (.ofInt (.ld 2 15 240)), 7) ∈ SporosVmBin.vmTable)
    (hm4 : (a + 19, (.ldIdx .w64 3 2 6 1 disp), 5) ∈ SporosVmBin.vmTable)
    (hm5 : (a + 24, (.ofInt (.ld 2 15 248)), 7) ∈ SporosVmBin.vmTable)
    (hm6 : (a + 31, (.ldIdx .w64 1 2 6 1 disp), 5) ∈ SporosVmBin.vmTable)
    (hm7 : (a + 36, (.ofInt (.cmp 0 3)), 3) ∈ SporosVmBin.vmTable)
    (hm8 : (a + 39, (.ofInt (.jcc 4 done)), 2) ∈ SporosVmBin.vmTable)
    (hc1 : rd64 s.mem (ea ctxBase 232) = some c1)
    (hc2 : rd64 s.mem (ea ctxBase 240) = some c2)
    (hc3 : rd64 s.mem (ea ctxBase 248) = some c3)
    (hv1 : rd64 s.mem (eaIdx c1 r6v 1 disp) = some v)
    (hv2 : rd64 s.mem (eaIdx c2 r6v 1 disp) = some v)
    (hv3 : rd64 s.mem (eaIdx c3 r6v 1 disp) = some v)
    (hv : v < X86Env.W) :
    ∃ s', Reaches vmImage s s' ∧
      s'.rip = done ∧ s'.mem = s.mem ∧ s'.k = s.k ∧ RegsOk s' ∧
      s'.reg 0 = some v ∧ s'.reg 1 = some v ∧ s'.reg 3 = some v ∧
      (∀ r, r ≠ 0 → r ≠ 1 → r ≠ 2 → r ≠ 3 → s'.reg r = s.reg r) := by
  have h01 : X86Vm.step vmImage s = .ok { s.set 2 (some c1) with rip := a + 7 } := by
    have hm1' : (s.rip, (.ofInt (.ld 2 15 232)), 7) ∈ SporosVmBin.vmTable := by rwa [hrip]
    rw [step_of_table hm1', exec_ld h15 hc1, hrip]
  let s1 := { s.set 2 (some c1) with rip := a + 7 }
  have hr1 : RegsOk s1 := regsOk_set hr (by decide) c1
  have hs1_2 : s1.reg 2 = some c1 := by
    simp only [s1, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr.1]; decide)]
  have hs1_6 : s1.reg 6 = some r6v := by
    simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 6)]; exact hr6
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 0 (some v) with rip := a + 12 } := by
    rw [step_of_table hm2, exec_ldIdx hs1_2 hs1_6 (show rdMem .w64 s1.mem (eaIdx c1 r6v 1 disp) = some v from hv1)]
  let s2 := { s1.set 0 (some v) with rip := a + 12 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) v
  have hs2_15 : s2.reg 15 = some ctxBase := by
    simp only [s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 15), List.getElem?_set_ne (by decide : 2 ≠ 15)]; exact h15
  have hs2_6 : s2.reg 6 = some r6v := by
    simp only [s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 6)]; exact hs1_6
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 2 (some c2) with rip := a + 19 } := by
    rw [step_of_table hm3, exec_ld hs2_15 hc2]
  let s3 := { s2.set 2 (some c2) with rip := a + 19 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) c2
  have hs3_2 : s3.reg 2 = some c2 := by
    simp only [s3, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
  have hs3_6 : s3.reg 6 = some r6v := by
    simp only [s3, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 6)]; exact hs2_6
  have h04 : X86Vm.step vmImage s3 = .ok { s3.set 3 (some v) with rip := a + 24 } := by
    rw [step_of_table hm4, exec_ldIdx hs3_2 hs3_6 (show rdMem .w64 s3.mem (eaIdx c2 r6v 1 disp) = some v from hv2)]
  let s4 := { s3.set 3 (some v) with rip := a + 24 }
  have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) v
  have hs4_15 : s4.reg 15 = some ctxBase := by
    simp only [s4, s3, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 3 ≠ 15), List.getElem?_set_ne (by decide : 2 ≠ 15)]; exact hs2_15
  have hs4_6 : s4.reg 6 = some r6v := by
    simp only [s4, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 3 ≠ 6)]; exact hs3_6
  have h05 : X86Vm.step vmImage s4 = .ok { s4.set 2 (some c3) with rip := a + 31 } := by
    rw [step_of_table hm5, exec_ld hs4_15 hc3]
  let s5 := { s4.set 2 (some c3) with rip := a + 31 }
  have hr5 : RegsOk s5 := regsOk_set hr4 (by decide) c3
  have hs5_2 : s5.reg 2 = some c3 := by
    simp only [s5, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr4.1]; decide)]
  have hs5_6 : s5.reg 6 = some r6v := by
    simp only [s5, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 6)]; exact hs4_6
  have h06 : X86Vm.step vmImage s5 = .ok { s5.set 1 (some v) with rip := a + 36 } := by
    rw [step_of_table hm6, exec_ldIdx hs5_2 hs5_6 (show rdMem .w64 s5.mem (eaIdx c3 r6v 1 disp) = some v from hv3)]
  let s6 := { s5.set 1 (some v) with rip := a + 36 }
  have hr6' : RegsOk s6 := regsOk_set hr5 (by decide) v
  have hs6_0 : s6.reg 0 = some v := by
    simp only [s6, s5, s4, s3, s2, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 0), List.getElem?_set_ne (by decide : 2 ≠ 0),
      List.getElem?_set_ne (by decide : 3 ≠ 0)]
    rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  have hs6_3 : s6.reg 3 = some v := by
    simp only [s6, s5, s4, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 3), List.getElem?_set_ne (by decide : 2 ≠ 3)]
    rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
  have hs6_1 : s6.reg 1 = some v := by
    simp only [s6, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr5.1]; decide)]
  have hvote := tmr_inline_vote_eq_spec rfl hm7 hm8 hs6_0 hs6_3 hv
  let s7 := { s6 with fl := (X86Int.aluF .sub v v).2, rip := done }
  have hreach : Reaches vmImage s s7 :=
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03
      (Reaches.step_trans h04 (Reaches.step_trans h05 (Reaches.step_trans h06 hvote)))))
  refine ⟨s7, hreach, rfl, rfl, rfl, hr6', hs6_0, hs6_1, hs6_3, ?_⟩
  intro r h0 h1 h2 h3
  simp only [s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
    List.getElem?_set_ne (Ne.symm h1), List.getElem?_set_ne (Ne.symm h2),
    List.getElem?_set_ne (Ne.symm h3), List.getElem?_set_ne (Ne.symm h0)]

theorem L_vm_vote_eq_spec {v : Nat} {s : X86Vm.St}
    (hrip : s.rip = SporosVmBin.L_vm_vote) (hr : RegsOk s)
    (h0 : s.reg 0 = some v) (h3 : s.reg 3 = some v) (hv : v < X86Env.W) :
    ∃ s', Reaches vmImage s s' ∧
      s'.rip = SporosVmBin.L_vm_dispatch ∧ s'.mem = s.mem ∧ s'.k = s.k ∧ RegsOk s' ∧
      s'.reg 0 = some v ∧ s'.reg 1 = some v ∧ s'.reg 3 = some v ∧
      (∀ r, r ≠ 1 → r ≠ 3 → s'.reg r = s.reg r) := by
  have hvote := tmr_inline_vote_eq_spec hrip
    (mem_vmTable_of_chunk 1 rfl (by decide : (4194890, (.ofInt (.cmp 0 3)), 3) ∈ SporosVmBin.vmTable_1))
    (mem_vmTable_of_chunk 1 rfl (by decide : (4194893, (.ofInt (.jcc 4 4194908)), 2) ∈ SporosVmBin.vmTable_1))
    h0 h3 hv
  let s1 := { s with fl := (X86Int.aluF .sub v v).2, rip := 4194908 }
  have hs1_0 : s1.reg 0 = some v := h0
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 3 (some v) with rip := 4194911 } := by
    rw [step_at_chunk 1 rfl rfl (by decide : (4194908, (.ofInt (.mov 3 0)), 3) ∈ SporosVmBin.vmTable_1),
      exec_mov hs1_0]
  let s2 := { s1.set 3 (some v) with rip := 4194911 }
  have hr2 : RegsOk s2 := regsOk_set hr (by decide) v
  have hs2_0 : s2.reg 0 = some v := by
    simp only [s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 3 ≠ 0)]; exact hs1_0
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 1 (some v) with rip := 4194914 } := by
    rw [step_of_table (mem_vmTable_of_chunk 1 rfl (by decide : (4194911, (.ofInt (.mov 1 0)), 3) ∈ SporosVmBin.vmTable_1)),
      exec_mov hs2_0]
  let s3 := { s2.set 1 (some v) with rip := 4194914 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) v
  have h04 : X86Vm.step vmImage s3 = .ok { s3 with rip := SporosVmBin.L_vm_dispatch } := by
    rw [step_of_table (mem_vmTable_of_chunk 1 rfl (by decide : (4194914, (.ofInt (.jmp 4194928)), 2) ∈ SporosVmBin.vmTable_1)),
      exec_jmp]; rfl
  let s4 := { s3 with rip := SporosVmBin.L_vm_dispatch }
  have hs4_0 : s4.reg 0 = some v := by
    simp only [s4, s3, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0)]; exact hs2_0
  have hs4_1 : s4.reg 1 = some v := by
    simp only [s4, s3, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
  have hs4_3 : s4.reg 3 = some v := by
    simp only [s4, s3, s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 3)]
    rw [getElem?_set_getD_self (by rw [hr.1]; decide)]
  refine ⟨s4, Reaches.trans hvote (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step h04))),
    rfl, rfl, rfl, hr3, hs4_0, hs4_1, hs4_3, ?_⟩
  intro r hr1 hr3'
  simp only [s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
    List.getElem?_set_ne (Ne.symm hr1), List.getElem?_set_ne (Ne.symm hr3')]

private theorem flat_boff_lt {pr : Sporos.Prog} (hfit : FlatProgFit pr) :
    pr.data.length * 16 < 2 ^ 47 ∧
    pr.words.length * 64 < 2 ^ 47 ∧
    flatProgBaseOff pr < 2 ^ 47 ∧
    (∀ {w : Nat}, w < pr.words.length →
      w * 64 + pr.data.length * 16 + 32 + 32 < 2 ^ 47 ∧
      ∀ {pc : Nat}, pc < pr.words[w]!.code.length →
        pc * 16 + (flatProgBaseOff pr + ((pr.words.take w).map fun x => x.code.length * 16).foldl (· + ·) 0) + 16 ≤ 2 ^ 47) := by
  have hfqlen : (flatProgQwords pr).length =
      4 + 2 * pr.data.length + 8 * pr.words.length + 2 * (pr.words.map (·.code.length)).foldl (· + ·) 0 := by
    simp [flatProgQwords, flatProgHeader, dataMetaQwords_len, wordMetaQwords_len, codePairsQwords_len]
    omega
  have hlim := hfit.2.2.2.1
  rw [hfqlen] at hlim
  have hbase_eq : flatProgBaseOff pr = (4 + 2 * pr.data.length + 8 * pr.words.length) * 8 := by
    simp [flatProgBaseOff]; omega
  refine ⟨by omega, by omega, by omega, ?_⟩
  intro w hw
  refine ⟨by omega, ?_⟩
  intro pc hpc
  have hsum_le :
      ((pr.words.take w).map (·.code.length)).foldl (· + ·) 0 + pr.words[w]!.code.length ≤
        (pr.words.map (·.code.length)).foldl (· + ·) 0 := by
    have go : ∀ (ws : List Sporos.Word) (idx : Nat), idx < ws.length →
        ((ws.take idx).map (·.code.length)).foldl (· + ·) 0 + ws[idx]!.code.length ≤
          (ws.map (·.code.length)).foldl (· + ·) 0 := by
      intro ws
      induction ws with
      | nil => intro idx hi; simp at hi
      | cons a l ih =>
        intro idx hi
        cases idx with
        | zero =>
          simp only [List.take_zero, List.map_nil, List.foldl_nil, Nat.zero_add,
            List.getElem!_eq_getElem?_getD, List.getElem?_cons_zero, Option.getD_some,
            List.map_cons, List.foldl_cons]
          rw [list_foldl_add]
          omega
        | succ k =>
          have hk : k < l.length := by simp at hi; omega
          have hr := ih k hk
          simp only [List.take_succ_cons, List.map_cons, List.foldl_cons, Nat.zero_add,
            List.getElem!_eq_getElem?_getD, List.getElem?_cons_succ] at hr ⊢
          rw [list_foldl_add, list_foldl_add (l.map (·.code.length))]
          omega
    exact go pr.words w hw
  have hmul_off :
      ((pr.words.take w).map fun x => x.code.length * 16).foldl (· + ·) 0 =
        16 * ((pr.words.take w).map (·.code.length)).foldl (· + ·) 0 := by
    have go : ∀ (ws : List Sporos.Word) (z : Nat),
        (ws.map (fun x => x.code.length * 16)).foldl (· + ·) (16 * z) =
          16 * (ws.map (·.code.length)).foldl (· + ·) z := by
      intro ws
      induction ws with
      | nil => intro z; simp
      | cons x xs ih =>
        intro z
        simp only [List.map_cons, List.foldl_cons]
        rw [show 16 * z + x.code.length * 16 = 16 * (z + x.code.length) by omega]
        exact ih (z + x.code.length)
    simpa using go (pr.words.take w) 0
  rw [hmul_off, hbase_eq]
  omega

theorem instrPair_snd_lt_W {pr : Sporos.Prog} (hfit : FlatProgFit pr) {w pc : Nat}
    (hw : w < pr.words.length) (hpc : pc < pr.words[w]!.code.length) :
    (instrPair pr.words[w]!.code[pc]!).2 < X86Env.W := by
  have hwd : pr.words[w]! ∈ pr.words := by
    rw [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem hw]
    exact List.getElem_mem hw
  have hweq : pr.words[w]! = pr.words[w] := by
    simp only [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem hw, Option.getD_some]
  have hwd : pr.words[w]! ∈ pr.words := hweq ▸ List.getElem_mem hw
  have hwd_fit := hfit.2.2.2.2.2 pr.words[w]! hwd
  have hpc' : pc < pr.words[w].code.length := by rwa [← hweq]
  have hpceq : pr.words[w]!.code[pc]! = pr.words[w].code[pc] := by
    rw [hweq]
    simp only [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem hpc', Option.getD_some]
  have hins : pr.words[w]!.code[pc]! ∈ pr.words[w]!.code := by
    rw [hpceq, hweq]
    exact List.getElem_mem hpc'
  have hins_fit := hwd_fit.2.2.2.2 pr.words[w]!.code[pc]! hins
  have hW : 0 < X86Env.W := by decide
  cases h : pr.words[w]!.code[pc]! <;> simp only [instrPair]
  all_goals first | exact Nat.mod_lt _ hW | exact hW | decide

theorem L_vm_loop_fetch_words_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s₀ : X86Vm.St} (hrip : s₀.rip = SporosVmBin.L_vm_loop) (h : vc.wf pr ls s₀) :
    (match ls.rs with
    | [] => ∃ s_exit, Reaches vmImage s₀ s_exit ∧ X86Vm.step vmImage s_exit = .exit 1 ∧ s_exit.k = s₀.k
    | f :: rest =>
      if _hnw : f.word < pr.words.length then
        let wd := pr.words[f.word]!
        if _hpc : f.pc < wd.code.length then
          ∃ s', Reaches vmImage s₀ s' ∧
            s'.rip = 4194722 ∧ s'.mem = s₀.mem ∧ s'.k = s₀.k ∧ vc.wf pr (f :: rest |>.head? |> fun _ => ls) s' ∧
            s'.reg 12 = some f.word ∧ s'.reg 13 = some f.pc ∧
            s'.reg 6 = some (f.word * 64 + pr.data.length * 16 + 32)
        else ∃ s_exit, Reaches vmImage s₀ s_exit ∧ X86Vm.step vmImage s_exit = .exit 1 ∧ s_exit.k = s₀.k
      else ∃ s_exit, Reaches vmImage s₀ s_exit ∧ X86Vm.step vmImage s_exit = .exit 1 ∧ s_exit.k = s₀.k) := by
  have hr := h.2.2.2.1
  have hsp := h.2.2.2.2.1
  have h15 := h.2.2.2.2.2.1
  have hmem := h.1.2
  have ⟨hrs_rw, _, _, _, hrs_hi, _, _, _, _, _, hrs_rel, hrs_le, _⟩ := (vc.arena_bounds h).2.2
  have ⟨hf1, hf2, hf3, hfit, _⟩ := vc.code_flat h
  have ⟨hrsBase, hrsLen, _, _, hc1, hc2, hc3⟩ := vc.rd_ctx_rs_code_slots h
  have h47 : 2 ^ 47 < X86Env.W := by decide
  have h01 : X86Vm.step vmImage s₀ = .ok { s₀.set 0 (some ls.rs.length) with rip := 4194484 } := by
    rw [step_at_chunk 0 hrip rfl (by decide : (4194480, (.ofInt (.ld 0 15 56)), 4) ∈ SporosVmBin.vmTable_0),
      exec_ld h15 hrsLen, hrip]; rfl
  let s1 := { s₀.set 0 (some ls.rs.length) with rip := 4194484 }
  have hr1 : RegsOk s1 := regsOk_set hr (by decide) ls.rs.length
  have hs1_0 : s1.reg 0 = some ls.rs.length := by
    simp only [s1, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr.1]; decide)]
  have h02 : X86Vm.step vmImage s1 =
      .ok { s1 with fl := (X86Int.aluF .and ls.rs.length ls.rs.length).2, rip := 4194487 } := by
    rw [step_at_chunk 0 rfl rfl (by decide : (4194484, (.ofInt (.test 0 0)), 3) ∈ SporosVmBin.vmTable_0),
      exec_test hs1_0 hs1_0]
  let s2 := { s1 with fl := (X86Int.aluF .and ls.rs.length ls.rs.length).2, rip := 4194487 }
  cases hrs : ls.rs with
  | nil =>
    have h03 : X86Vm.step vmImage s2 = .ok { s2 with rip := 4201133 } := by
      rw [step_at_chunk 0 rfl rfl (by decide : (4194487, (.ofInt (.jcc 4 4201133)), 6) ∈ SporosVmBin.vmTable_0),
        exec_jcc (cond_aluF_and_self_e ls.rs.length)]
      simp [s2, hrs]
    let s3 := { s2 with rip := 4201133 }
    have hwf3 : vc.wf pr ls s3 := vc.wf_step_misc h rfl hr1 (by
      simp only [s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 4)]; exact hsp) (by
      simp only [s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 15)]; exact h15)
    obtain ⟨se, hre, hexe, hke⟩ := L_fail_from_vc_spec (e := .noFrame) rfl hwf3
    exact ⟨se, Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 hre)), hexe, hke⟩
  | cons f rest =>
    dsimp only
    rw [hrs] at hrs_rel hrs_le
    simp only [RsLocRel] at hrs_rel
    rcases hrs_rel with ⟨_, hrl_len, _, hfw_lt, hfpc_lt, _, _, _, _⟩
    have h03 : X86Vm.step vmImage s2 = .ok { s2 with rip := 4194493 } := by
      rw [step_at_chunk 0 rfl rfl (by decide : (4194487, (.ofInt (.jcc 4 4201133)), 6) ∈ SporosVmBin.vmTable_0),
        exec_jcc (cond_aluF_and_self_e ls.rs.length)]
      simp [s2, hrs]
    let s3 := { s2 with rip := 4194493 }
    have hs3_0 : s3.reg 0 = some (rest.length + 1) := by simpa [s3, s2, X86Int.St.reg, hrs] using hs1_0
    have hlenW : rest.length + 1 < X86Env.W := by
      have : (rest.length + 1) * 32 ≤ vc.rsReg.data.length := hrl_len
      simp only [Region.hi] at hrs_hi; omega
    have hsub1 : (rest.length + 1 + X86Env.W - 1) % X86Env.W = rest.length := by
      have : rest.length + 1 + X86Env.W - 1 = rest.length + X86Env.W := by omega
      rw [this, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
    have h04 : X86Vm.step vmImage s3 =
        .ok { s3.set 0 (some rest.length) with fl := (X86Int.aluF .sub (rest.length + 1) 1).2, rip := 4194497 } := by
      rw [step_at_chunk 0 rfl rfl (by decide : (4194493, (.ofInt (.aluI .sub 0 1)), 4) ∈ SporosVmBin.vmTable_0),
        exec_aluI hs3_0, Nat.mod_eq_of_lt (by decide)]
      dsimp only
      rw [aluF_sub_val, hsub1]
    let s4 := { s3.set 0 (some rest.length) with fl := (X86Int.aluF .sub (rest.length + 1) 1).2, rip := 4194497 }
    have hr4 : RegsOk s4 := regsOk_set hr1 (by decide) rest.length
    have hs4_0 : s4.reg 0 = some rest.length := by
      simp only [s4, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
    have hsh5W : rest.length * 32 < X86Env.W := by simp only [Region.hi] at hrs_hi; omega
    have h05 : X86Vm.step vmImage s4 =
        .ok { s4.set 0 (some (rest.length * 32)) with fl := X86Int.Flags.undef, rip := 4194501 } := by
      rw [step_at_chunk 0 rfl rfl (by decide : (4194497, (.ofInt (.shlI 0 5)), 4) ∈ SporosVmBin.vmTable_0),
        exec_shlI_lt (c := 5) hs4_0 (by simpa [show 2 ^ (5 % 64) = 32 from by decide] using hsh5W)]
    let s5 := { s4.set 0 (some (rest.length * 32)) with fl := X86Int.Flags.undef, rip := 4194501 }
    have hr5 : RegsOk s5 := regsOk_set hr4 (by decide) (rest.length * 32)
    have hs5_15 : s5.reg 15 = some ctxBase := by
      simp only [s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 0 ≠ 15)]; exact h15
    have h06 : X86Vm.step vmImage s5 = .ok { s5.set 2 (some vc.rsReg.base) with rip := 4194505 } := by
      rw [step_at_chunk 0 rfl rfl (by decide : (4194501, (.ofInt (.ld 2 15 48)), 4) ∈ SporosVmBin.vmTable_0),
        exec_ld hs5_15 hrsBase]
    let s6 := { s5.set 2 (some vc.rsReg.base) with rip := 4194505 }
    have hr6 : RegsOk s6 := regsOk_set hr5 (by decide) vc.rsReg.base
    have hs6_0 : s6.reg 0 = some (rest.length * 32) := by
      simp only [s6, s5, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 0)]
      rw [getElem?_set_getD_self (by rw [hr4.1]; decide)]
    have hs6_2 : s6.reg 2 = some vc.rsReg.base := by
      simp only [s6, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr5.1]; decide)]
    have hrs_off_W : rest.length * 32 + vc.rsReg.base + 16 < X86Env.W := by
      simp only [Region.hi] at hrs_hi; omega
    have h07 : X86Vm.step vmImage s6 =
        .ok { s6.set 0 (some (rest.length * 32 + vc.rsReg.base)) with
          fl := (X86Int.aluF .add (rest.length * 32) vc.rsReg.base).2, rip := 4194508 } := by
      rw [step_at_chunk 0 rfl rfl (by decide : (4194505, (.ofInt (.alu .add 0 2)), 3) ∈ SporosVmBin.vmTable_0),
        exec_alu hs6_0 hs6_2]
      dsimp only
      rw [aluF_add_val, Nat.mod_eq_of_lt (by omega)]
    let s7 := { s6.set 0 (some (rest.length * 32 + vc.rsReg.base)) with
      fl := (X86Int.aluF .add (rest.length * 32) vc.rsReg.base).2, rip := 4194508 }
    have hr7 : RegsOk s7 := regsOk_set hr6 (by decide) (rest.length * 32 + vc.rsReg.base)
    have hs7_0 : s7.reg 0 = some (rest.length * 32 + vc.rsReg.base) := by
      simp only [s7, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr6.1]; decide)]
    have hmem_rs : vc.rsReg ∈ s₀.mem := vc.mem_rs h
    have hrd_w : rd64 s7.mem (ea (rest.length * 32 + vc.rsReg.base) 0) = some f.word := by
      rw [ea_add_eq (by omega)]
      refine rd64_of_mem hmem hmem_rs (by omega)
        (by simp only [Region.hi] at hrs_hi ⊢; omega) ?_
      rw [show rest.length * 32 + vc.rsReg.base + 0 - vc.rsReg.base = rest.length * 32 by omega]
      exact hfw_lt
    have h08 : X86Vm.step vmImage s7 = .ok { s7.set 12 (some f.word) with rip := 4194511 } := by
      rw [step_at_chunk 0 rfl rfl (by decide : (4194508, (.ofInt (.ld 12 0 0)), 3) ∈ SporosVmBin.vmTable_0),
        exec_ld hs7_0 hrd_w]
    let s8 := { s7.set 12 (some f.word) with rip := 4194511 }
    have hr8 : RegsOk s8 := regsOk_set hr7 (by decide) f.word
    have hs8_0 : s8.reg 0 = some (rest.length * 32 + vc.rsReg.base) := by
      simp only [s8, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 12 ≠ 0)]; exact hs7_0
    have hrd_pc : rd64 s8.mem (ea (rest.length * 32 + vc.rsReg.base) 8) = some f.pc := by
      rw [ea_add_eq (by omega)]
      refine rd64_of_mem hmem hmem_rs (by omega)
        (by simp only [Region.hi] at hrs_hi ⊢; omega) ?_
      rw [show rest.length * 32 + vc.rsReg.base + 8 - vc.rsReg.base = rest.length * 32 + 8 by omega]
      exact hfpc_lt
    have h09 : X86Vm.step vmImage s8 = .ok { s8.set 13 (some f.pc) with rip := 4194515 } := by
      rw [step_at_chunk 0 rfl rfl (by decide : (4194511, (.ofInt (.ld 13 0 8)), 4) ∈ SporosVmBin.vmTable_0),
        exec_ld hs8_0 hrd_pc]
    let s9 := { s8.set 13 (some f.pc) with rip := 4194515 }
    have hr9 : RegsOk s9 := regsOk_set hr8 (by decide) f.pc
    have hs9_15 : s9.reg 15 = some ctxBase := by
      simp only [s9, s8, s7, s6, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 13 ≠ 15), List.getElem?_set_ne (by decide : 12 ≠ 15),
        List.getElem?_set_ne (by decide : 0 ≠ 15), List.getElem?_set_ne (by decide : 2 ≠ 15)]
      exact hs5_15
    have hs9_4 : s9.reg 4 = some vc.sp := by
      simp only [s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 13 ≠ 4), List.getElem?_set_ne (by decide : 12 ≠ 4),
        List.getElem?_set_ne (by decide : 0 ≠ 4), List.getElem?_set_ne (by decide : 2 ≠ 4)]
      exact hsp
    have hs9_12 : s9.reg 12 = some f.word := by
      simp only [s9, s8, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 13 ≠ 12)]
      rw [getElem?_set_getD_self (by rw [hr7.1]; decide)]
    have hs9_13 : s9.reg 13 = some f.pc := by
      simp only [s9, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr8.1]; decide)]
    have hnw1 := rd64_flat_nw hmem (vc.mem_code1 h) hf1 hfit
    have hnw2 := rd64_flat_nw hmem (vc.mem_code2 h) hf2 hfit
    have hnw3 := rd64_flat_nw hmem (vc.mem_code3 h) hf3 hfit
    obtain ⟨s10, hrch10, hrip10, hmem10, hk10, hr10, hs10_0, _, _, hs10_oth⟩ :=
      tmr_ld_vote_eq_spec (s := s9) rfl hr9 hs9_15
        (mem_vmTable_of_chunk 0 rfl (by decide : (4194515, (.ofInt (.ld 2 15 232)), 7) ∈ SporosVmBin.vmTable_0))
        (mem_vmTable_of_chunk 0 rfl (by decide : (4194522, (.ofInt (.ld 0 2 8)), 4) ∈ SporosVmBin.vmTable_0))
        (mem_vmTable_of_chunk 0 rfl (by decide : (4194526, (.ofInt (.ld 2 15 240)), 7) ∈ SporosVmBin.vmTable_0))
        (mem_vmTable_of_chunk 0 rfl (by decide : (4194533, (.ofInt (.ld 3 2 8)), 4) ∈ SporosVmBin.vmTable_0))
        (mem_vmTable_of_chunk 0 rfl (by decide : (4194537, (.ofInt (.ld 2 15 248)), 7) ∈ SporosVmBin.vmTable_0))
        (mem_vmTable_of_chunk 0 rfl (by decide : (4194544, (.ofInt (.ld 1 2 8)), 4) ∈ SporosVmBin.vmTable_0))
        (mem_vmTable_of_chunk 0 rfl (by decide : (4194548, (.ofInt (.cmp 0 3)), 3) ∈ SporosVmBin.vmTable_0))
        (mem_vmTable_of_chunk 0 rfl (by decide : (4194551, (.ofInt (.jcc 4 4194571)), 2) ∈ SporosVmBin.vmTable_0))
        hc1 hc2 hc3 hnw1 hnw2 hnw3 hfit.2.1
    have hs10_12 : s10.reg 12 = some f.word := by rw [hs10_oth 12 (by decide) (by decide) (by decide) (by decide)]; exact hs9_12
    have hs10_13 : s10.reg 13 = some f.pc := by rw [hs10_oth 13 (by decide) (by decide) (by decide) (by decide)]; exact hs9_13
    have hs10_15 : s10.reg 15 = some ctxBase := by rw [hs10_oth 15 (by decide) (by decide) (by decide) (by decide)]; exact hs9_15
    have hs10_4 : s10.reg 4 = some vc.sp := by rw [hs10_oth 4 (by decide) (by decide) (by decide) (by decide)]; exact hs9_4
    have h11 : X86Vm.step vmImage s10 =
        .ok { s10 with fl := (X86Int.aluF .sub f.word pr.words.length).2, rip := 4194574 } := by
      rw [step_at_chunk 0 hrip10 rfl (by decide : (4194571, (.ofInt (.cmp 12 0)), 3) ∈ SporosVmBin.vmTable_0),
        exec_cmp hs10_12 hs10_0, hrip10]
    let s11 := { s10 with fl := (X86Int.aluF .sub f.word pr.words.length).2, rip := 4194574 }
    have hrch04 : Reaches vmImage s₀ s4 :=
      Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step h04)))
    have hrch49 : Reaches vmImage s4 s9 :=
      Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step_trans h08 (Reaches.step h09))))
    have hreach11 : Reaches vmImage s₀ s11 :=
      Reaches.trans hrch04 (Reaches.trans hrch49 (Reaches.trans hrch10 (Reaches.step h11)))
    by_cases hnw : f.word < pr.words.length
    · rw [dif_pos hnw]
      have h12 : X86Vm.step vmImage s11 = .ok { s11 with rip := 4194580 } := by
        rw [step_at_chunk 0 rfl rfl (by decide : (4194574, (.ofInt (.jcc 3 4201099)), 6) ∈ SporosVmBin.vmTable_0),
          exec_jcc (cond_aluF_sub_ae f.word pr.words.length)]
        simp [s11, show ¬ (pr.words.length ≤ f.word) by omega]
      let s12 := { s11 with rip := 4194580 }
      have hs12_15 : s12.reg 15 = some ctxBase := hs10_15
      have hc1_12 : rd64 s12.mem (ea ctxBase 232) = some vc.code1.base := by rw [show s12.mem = s₀.mem from hmem10]; exact hc1
      have hc2_12 : rd64 s12.mem (ea ctxBase 240) = some vc.code2.base := by rw [show s12.mem = s₀.mem from hmem10]; exact hc2
      have hc3_12 : rd64 s12.mem (ea ctxBase 248) = some vc.code3.base := by rw [show s12.mem = s₀.mem from hmem10]; exact hc3
      have hnd1 := rd64_flat_nd hmem (vc.mem_code1 h) hf1 hfit
      have hnd2 := rd64_flat_nd hmem (vc.mem_code2 h) hf2 hfit
      have hnd3 := rd64_flat_nd hmem (vc.mem_code3 h) hf3 hfit
      rw [show s₀.mem = s12.mem from hmem10.symm] at hnd1 hnd2 hnd3
      obtain ⟨s13, hrch13, hrip13, hmem13, hk13, hr13, hs13_0, _, _, hs13_oth⟩ :=
        tmr_ld_vote_eq_spec (s := s12) rfl hr10 hs12_15
          (mem_vmTable_of_chunk 0 rfl (by decide : (4194580, (.ofInt (.ld 2 15 232)), 7) ∈ SporosVmBin.vmTable_0))
          (mem_vmTable_of_chunk 0 rfl (by decide : (4194587, (.ofInt (.ld 0 2 16)), 4) ∈ SporosVmBin.vmTable_0))
          (mem_vmTable_of_chunk 0 rfl (by decide : (4194591, (.ofInt (.ld 2 15 240)), 7) ∈ SporosVmBin.vmTable_0))
          (mem_vmTable_of_chunk 0 rfl (by decide : (4194598, (.ofInt (.ld 3 2 16)), 4) ∈ SporosVmBin.vmTable_0))
          (mem_vmTable_of_chunk 0 rfl (by decide : (4194602, (.ofInt (.ld 2 15 248)), 7) ∈ SporosVmBin.vmTable_0))
          (mem_vmTable_of_chunk 0 rfl (by decide : (4194609, (.ofInt (.ld 1 2 16)), 4) ∈ SporosVmBin.vmTable_0))
          (mem_vmTable_of_chunk 0 rfl (by decide : (4194613, (.ofInt (.cmp 0 3)), 3) ∈ SporosVmBin.vmTable_0))
          (mem_vmTable_of_chunk 0 rfl (by decide : (4194616, (.ofInt (.jcc 4 4194636)), 2) ∈ SporosVmBin.vmTable_0))
          hc1_12 hc2_12 hc3_12 hnd1 hnd2 hnd3 hfit.2.2.1
      have hs13_12 : s13.reg 12 = some f.word := by rw [hs13_oth 12 (by decide) (by decide) (by decide) (by decide)]; exact hs10_12
      have hs13_13 : s13.reg 13 = some f.pc := by rw [hs13_oth 13 (by decide) (by decide) (by decide) (by decide)]; exact hs10_13
      have hs13_15 : s13.reg 15 = some ctxBase := by rw [hs13_oth 15 (by decide) (by decide) (by decide) (by decide)]; exact hs12_15
      have hs13_4 : s13.reg 4 = some vc.sp := by rw [hs13_oth 4 (by decide) (by decide) (by decide) (by decide)]; exact hs10_4
      have ⟨hnd16_lt, hnw64_lt, _, hboff_lt⟩ := flat_boff_lt hfit
      have hboff_w := (hboff_lt hnw).1
      have h14 : X86Vm.step vmImage s13 =
          .ok { s13.set 0 (some (pr.data.length * 16)) with fl := X86Int.Flags.undef, rip := 4194640 } := by
        rw [step_at_chunk 0 hrip13 rfl (by decide : (4194636, (.ofInt (.shlI 0 4)), 4) ∈ SporosVmBin.vmTable_0),
          exec_shlI_4_lt hs13_0 (by omega), hrip13]
      let s14 := { s13.set 0 (some (pr.data.length * 16)) with fl := X86Int.Flags.undef, rip := 4194640 }
      have hr14 : RegsOk s14 := regsOk_set hr13 (by decide) (pr.data.length * 16)
      have hs14_12 : s14.reg 12 = some f.word := by
        simp only [s14, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 12)]; exact hs13_12
      have h15' : X86Vm.step vmImage s14 = .ok { s14.set 6 (some f.word) with rip := 4194643 } := by
        rw [step_at_chunk 0 rfl rfl (by decide : (4194640, (.ofInt (.mov 6 12)), 3) ∈ SporosVmBin.vmTable_0),
          exec_mov hs14_12]
      let s15 := { s14.set 6 (some f.word) with rip := 4194643 }
      have hr15 : RegsOk s15 := regsOk_set hr14 (by decide) f.word
      have hs15_6 : s15.reg 6 = some f.word := by
        simp only [s15, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr14.1]; decide)]
      have h16 : X86Vm.step vmImage s15 =
          .ok { s15.set 6 (some (f.word * 64)) with fl := X86Int.Flags.undef, rip := 4194647 } := by
        rw [step_at_chunk 0 rfl rfl (by decide : (4194643, (.ofInt (.shlI 6 6)), 4) ∈ SporosVmBin.vmTable_0),
          exec_shlI_lt (c := 6) hs15_6 (by rw [show 2 ^ (6 % 64) = 64 from by decide]; omega)]
      let s16 := { s15.set 6 (some (f.word * 64)) with fl := X86Int.Flags.undef, rip := 4194647 }
      have hr16 : RegsOk s16 := regsOk_set hr15 (by decide) (f.word * 64)
      have hs16_6 : s16.reg 6 = some (f.word * 64) := by
        simp only [s16, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr15.1]; decide)]
      have hs16_0 : s16.reg 0 = some (pr.data.length * 16) := by
        simp only [s16, s15, s14, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 6 ≠ 0)]
        rw [getElem?_set_getD_self (by rw [hr13.1]; decide)]
      have h17 : X86Vm.step vmImage s16 =
          .ok { s16.set 6 (some (f.word * 64 + pr.data.length * 16)) with
            fl := (X86Int.aluF .add (f.word * 64) (pr.data.length * 16)).2, rip := 4194650 } := by
        rw [step_at_chunk 0 rfl rfl (by decide : (4194647, (.ofInt (.alu .add 6 0)), 3) ∈ SporosVmBin.vmTable_0),
          exec_alu hs16_6 hs16_0]
        dsimp only
        rw [aluF_add_val, Nat.mod_eq_of_lt (by omega)]
      let s17 := { s16.set 6 (some (f.word * 64 + pr.data.length * 16)) with
        fl := (X86Int.aluF .add (f.word * 64) (pr.data.length * 16)).2, rip := 4194650 }
      have hr17 : RegsOk s17 := regsOk_set hr16 (by decide) (f.word * 64 + pr.data.length * 16)
      have hs17_6 : s17.reg 6 = some (f.word * 64 + pr.data.length * 16) := by
        simp only [s17, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr16.1]; decide)]
      let boff := f.word * 64 + pr.data.length * 16 + 32
      have h18 : X86Vm.step vmImage s17 =
          .ok { s17.set 6 (some boff) with fl := (X86Int.aluF .add (f.word * 64 + pr.data.length * 16) 32).2, rip := 4194654 } := by
        rw [step_at_chunk 0 rfl rfl (by decide : (4194650, (.ofInt (.aluI .add 6 32)), 4) ∈ SporosVmBin.vmTable_0),
          exec_aluI_add_lt hs17_6 (by omega), Nat.mod_eq_of_lt (by omega)]
      let s18 := { s17.set 6 (some boff) with fl := (X86Int.aluF .add (f.word * 64 + pr.data.length * 16) 32).2, rip := 4194654 }
      have hr18 : RegsOk s18 := regsOk_set hr17 (by decide) boff
      have hs18_6 : s18.reg 6 = some boff := by
        simp only [s18, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr17.1]; decide)]
      have hs18_15 : s18.reg 15 = some ctxBase := by
        simp only [s18, s17, s16, s15, s14, X86Int.St.reg, X86Int.St.set,
          List.getElem?_set_ne (by decide : 6 ≠ 15), List.getElem?_set_ne (by decide : 0 ≠ 15)]; exact hs13_15
      have hs18_12 : s18.reg 12 = some f.word := by
        simp only [s18, s17, s16, s15, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 6 ≠ 12)]
        exact hs14_12
      have hs18_13 : s18.reg 13 = some f.pc := by
        simp only [s18, s17, s16, s15, s14, X86Int.St.reg, X86Int.St.set,
          List.getElem?_set_ne (by decide : 6 ≠ 13), List.getElem?_set_ne (by decide : 0 ≠ 13)]; exact hs13_13
      have hs18_4 : s18.reg 4 = some vc.sp := by
        simp only [s18, s17, s16, s15, s14, X86Int.St.reg, X86Int.St.set,
          List.getElem?_set_ne (by decide : 6 ≠ 4), List.getElem?_set_ne (by decide : 0 ≠ 4)]; exact hs13_4
      have hmem18 : s18.mem = s₀.mem := hmem13.trans hmem10
      have hc1_18 : rd64 s18.mem (ea ctxBase 232) = some vc.code1.base := by rw [hmem18]; exact hc1
      have hc2_18 : rd64 s18.mem (ea ctxBase 240) = some vc.code2.base := by rw [hmem18]; exact hc2
      have hc3_18 : rd64 s18.mem (ea ctxBase 248) = some vc.code3.base := by rw [hmem18]; exact hc3
      have hw1 := (rd64_flat_word hmem (vc.mem_code1 h) hf1 hfit hnw).2.2.1
      have hw2 := (rd64_flat_word hmem (vc.mem_code2 h) hf2 hfit hnw).2.2.1
      have hw3 := (rd64_flat_word hmem (vc.mem_code3 h) hf3 hfit hnw).2.2.1
      rw [show s₀.mem = s18.mem from hmem18.symm] at hw1 hw2 hw3
      have hwd : pr.words[f.word]! ∈ pr.words := by
        rw [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem hnw]; exact List.getElem_mem hnw
      have hwd_code_lt : pr.words[f.word]!.code.length < X86Env.W := (hfit.2.2.2.2.2 _ hwd).2.2.2.1
      obtain ⟨s19, hrch19, hrip19, hmem19, hk19, hr19, hs19_0, _, _, hs19_oth⟩ :=
        tmr_ldIdx_vote_eq_spec (s := s18) rfl hr18 hs18_15 hs18_6
          (mem_vmTable_of_chunk 0 rfl (by decide : (4194654, (.ofInt (.ld 2 15 232)), 7) ∈ SporosVmBin.vmTable_0))
          (mem_vmTable_of_chunk 0 rfl (by decide : (4194661, (.ldIdx .w64 0 2 6 1 24), 5) ∈ SporosVmBin.vmTable_0))
          (mem_vmTable_of_chunk 0 rfl (by decide : (4194666, (.ofInt (.ld 2 15 240)), 7) ∈ SporosVmBin.vmTable_0))
          (mem_vmTable_of_chunk 0 rfl (by decide : (4194673, (.ldIdx .w64 3 2 6 1 24), 5) ∈ SporosVmBin.vmTable_0))
          (mem_vmTable_of_chunk 0 rfl (by decide : (4194678, (.ofInt (.ld 2 15 248)), 7) ∈ SporosVmBin.vmTable_0))
          (mem_vmTable_of_chunk 0 rfl (by decide : (4194685, (.ldIdx .w64 1 2 6 1 24), 5) ∈ SporosVmBin.vmTable_0))
          (mem_vmTable_of_chunk 0 rfl (by decide : (4194690, (.ofInt (.cmp 0 3)), 3) ∈ SporosVmBin.vmTable_0))
          (mem_vmTable_of_chunk 0 rfl (by decide : (4194693, (.ofInt (.jcc 4 4194713)), 2) ∈ SporosVmBin.vmTable_0))
          hc1_18 hc2_18 hc3_18 hw1 hw2 hw3 hwd_code_lt
      have hs19_12 : s19.reg 12 = some f.word := by rw [hs19_oth 12 (by decide) (by decide) (by decide) (by decide)]; exact hs18_12
      have hs19_13 : s19.reg 13 = some f.pc := by rw [hs19_oth 13 (by decide) (by decide) (by decide) (by decide)]; exact hs18_13
      have hs19_15 : s19.reg 15 = some ctxBase := by rw [hs19_oth 15 (by decide) (by decide) (by decide) (by decide)]; exact hs18_15
      have hs19_4 : s19.reg 4 = some vc.sp := by rw [hs19_oth 4 (by decide) (by decide) (by decide) (by decide)]; exact hs18_4
      have hs19_6 : s19.reg 6 = some boff := by rw [hs19_oth 6 (by decide) (by decide) (by decide) (by decide)]; exact hs18_6
      have h20 : X86Vm.step vmImage s19 =
          .ok { s19 with fl := (X86Int.aluF .sub f.pc pr.words[f.word]!.code.length).2, rip := 4194716 } := by
        rw [step_at_chunk 0 hrip19 rfl (by decide : (4194713, (.ofInt (.cmp 13 0)), 3) ∈ SporosVmBin.vmTable_0),
          exec_cmp hs19_13 hs19_0, hrip19]
      let s20 := { s19 with fl := (X86Int.aluF .sub f.pc pr.words[f.word]!.code.length).2, rip := 4194716 }
      have hreach20 : Reaches vmImage s₀ s20 :=
        Reaches.trans hreach11 (Reaches.step_trans h12 (Reaches.trans hrch13
          (Reaches.step_trans h14 (Reaches.step_trans h15' (Reaches.step_trans h16
            (Reaches.step_trans h17 (Reaches.step_trans h18 (Reaches.trans hrch19 (Reaches.step h20)))))))))
      have hmem20 : s20.mem = s₀.mem := hmem19.trans hmem18
      have hk20 : s20.k = s₀.k := hk19.trans (hk13.trans hk10)
      by_cases hpc : f.pc < pr.words[f.word]!.code.length
      · rw [dif_pos hpc]
        have h21 : X86Vm.step vmImage s20 = .ok { s20 with rip := 4194722 } := by
          rw [step_at_chunk 0 rfl rfl (by decide : (4194716, (.ofInt (.jcc 3 4201099)), 6) ∈ SporosVmBin.vmTable_0),
            exec_jcc (cond_aluF_sub_ae f.pc pr.words[f.word]!.code.length)]
          dsimp only [s20]
          rw [if_neg (by rw [decide_eq_false (by omega)]; decide)]
        let s21 := { s20 with rip := 4194722 }
        have hwf21 : vc.wf pr ls s21 := vc.wf_step_misc h hmem20 hr19 hs19_4 hs19_15
        exact ⟨s21, Reaches.trans hreach20 (Reaches.step h21), rfl, hmem20, hk20, hwf21, hs19_12, hs19_13, hs19_6⟩
      · rw [dif_neg hpc]
        have h21 : X86Vm.step vmImage s20 = .ok { s20 with rip := 4201099 } := by
          rw [step_at_chunk 0 rfl rfl (by decide : (4194716, (.ofInt (.jcc 3 4201099)), 6) ∈ SporosVmBin.vmTable_0),
            exec_jcc (cond_aluF_sub_ae f.pc pr.words[f.word]!.code.length)]
          dsimp only [s20]
          rw [if_pos (decide_eq_true (by omega))]
        let s21 := { s20 with rip := 4201099 }
        have hwf21 : vc.wf pr ls s21 := vc.wf_step_misc h hmem20 hr19 hs19_4 hs19_15
        obtain ⟨se, hre, hexe, hke⟩ := L_fail_from_vc_spec (e := .badJump) rfl hwf21
        exact ⟨se, Reaches.trans hreach20 (Reaches.step_trans h21 hre), hexe, by rw [hke]; exact hk20⟩
    · rw [dif_neg hnw]
      have h12 : X86Vm.step vmImage s11 = .ok { s11 with rip := 4201099 } := by
        rw [step_at_chunk 0 rfl rfl (by decide : (4194574, (.ofInt (.jcc 3 4201099)), 6) ∈ SporosVmBin.vmTable_0),
          exec_jcc (cond_aluF_sub_ae f.word pr.words.length)]
        simp [s11, show pr.words.length ≤ f.word by omega]
      let s12 := { s11 with rip := 4201099 }
      have hwf12 : vc.wf pr ls s12 := vc.wf_step_misc h hmem10 hr10 hs10_4 hs10_15
      obtain ⟨se, hre, hexe, hke⟩ := L_fail_from_vc_spec (e := .badJump) rfl hwf12
      exact ⟨se, Reaches.trans hreach11 (Reaches.step_trans h12 hre), hexe, by rw [hke]; exact hk10⟩

theorem L_vm_loop_fetch_insn_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {w pc : Nat}
    (hrip : s.rip = 4194722) (h : vc.wf pr ls s)
    (_ : s.reg 12 = some w) (hr13 : s.reg 13 = some pc)
    (hr6 : s.reg 6 = some (w * 64 + pr.data.length * 16 + 32))
    (hw : w < pr.words.length) (hpc : pc < pr.words[w]!.code.length) :
    let ins := pr.words[w]!.code[pc]!
    let op := (instrPair ins).1
    let arg := (instrPair ins).2
    let ctx1 := { vc.ctxReg with data := writeBytes vc.ctxReg.data 264 (bytes8 arg) }
    ∃ idx1 m1 s',
      s.mem[idx1]? = some vc.ctxReg ∧
      m1 = s.mem.set idx1 ctx1 ∧
      Reaches vmImage s s' ∧
      s'.rip = SporosVmBin.L_vm_dispatch ∧ s'.mem = m1 ∧ s'.k = s.k ∧
      { vc with ctxReg := ctx1 }.wf pr ls s' ∧
      s'.reg 0 = some op ∧ s'.reg 1 = some op ∧ s'.reg 3 = some op ∧
      leVal ((ctx1.data.drop 264).take 8) = arg := by
  have hr := h.2.2.2.1
  have h15 := h.2.2.2.2.2.1
  have hmem := h.1.2
  have ⟨hf1, hf2, hf3, hfit, _⟩ := vc.code_flat h
  have ⟨_, _, _, _, hc1, hc2, hc3⟩ := vc.rd_ctx_rs_code_slots h
  have ⟨_, _, _, hboff_lt⟩ := flat_boff_lt hfit
  have hinsn_lim := (hboff_lt hw).2 hpc
  have h47 : 2 ^ 47 < X86Env.W := by decide
  let woff := flatProgBaseOff pr + ((pr.words.take w).map fun x => x.code.length * 16).foldl (· + ·) 0
  have hwoff_W : woff < X86Env.W := by dsimp [woff]; omega
  have hwo1 := (rd64_flat_word hmem (vc.mem_code1 h) hf1 hfit hw).2.2.2
  have hwo2 := (rd64_flat_word hmem (vc.mem_code2 h) hf2 hfit hw).2.2.2
  have hwo3 := (rd64_flat_word hmem (vc.mem_code3 h) hf3 hfit hw).2.2.2
  obtain ⟨s1, hrch1, hrip1, hmem1, hk1, hr1, hs1_0, _, _, hs1_oth⟩ :=
    tmr_ldIdx_vote_eq_spec hrip hr h15 hr6
      (mem_vmTable_of_chunk 0 rfl (by decide : (4194722, (.ofInt (.ld 2 15 232)), 7) ∈ SporosVmBin.vmTable_0))
      (mem_vmTable_of_chunk 0 rfl (by decide : (4194729, (.ldIdx .w64 0 2 6 1 32), 5) ∈ SporosVmBin.vmTable_0))
      (mem_vmTable_of_chunk 0 rfl (by decide : (4194734, (.ofInt (.ld 2 15 240)), 7) ∈ SporosVmBin.vmTable_0))
      (mem_vmTable_of_chunk 0 rfl (by decide : (4194741, (.ldIdx .w64 3 2 6 1 32), 5) ∈ SporosVmBin.vmTable_0))
      (mem_vmTable_of_chunk 1 rfl (by decide : (4194746, (.ofInt (.ld 2 15 248)), 7) ∈ SporosVmBin.vmTable_1))
      (mem_vmTable_of_chunk 1 rfl (by decide : (4194753, (.ldIdx .w64 1 2 6 1 32), 5) ∈ SporosVmBin.vmTable_1))
      (mem_vmTable_of_chunk 1 rfl (by decide : (4194758, (.ofInt (.cmp 0 3)), 3) ∈ SporosVmBin.vmTable_1))
      (mem_vmTable_of_chunk 1 rfl (by decide : (4194761, (.ofInt (.jcc 4 4194781)), 2) ∈ SporosVmBin.vmTable_1))
      hc1 hc2 hc3 hwo1 hwo2 hwo3 hwoff_W
  have hs1_13 : s1.reg 13 = some pc := by rw [hs1_oth 13 (by decide) (by decide) (by decide) (by decide)]; exact hr13
  have hs1_15 : s1.reg 15 = some ctxBase := by rw [hs1_oth 15 (by decide) (by decide) (by decide) (by decide)]; exact h15
  have hs1_4 : s1.reg 4 = some vc.sp := by rw [hs1_oth 4 (by decide) (by decide) (by decide) (by decide)]; exact h.2.2.2.2.1
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 6 (some pc) with rip := 4194784 } := by
    rw [step_at_chunk 1 hrip1 rfl (by decide : (4194781, (.ofInt (.mov 6 13)), 3) ∈ SporosVmBin.vmTable_1),
      exec_mov hs1_13, hrip1]
  let s2 := { s1.set 6 (some pc) with rip := 4194784 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) pc
  have hs2_6 : s2.reg 6 = some pc := by
    simp only [s2, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  have h03 : X86Vm.step vmImage s2 =
      .ok { s2.set 6 (some (pc * 16)) with fl := X86Int.Flags.undef, rip := 4194788 } := by
    rw [step_at_chunk 1 rfl rfl (by decide : (4194784, (.ofInt (.shlI 6 4)), 4) ∈ SporosVmBin.vmTable_1),
      exec_shlI_4_lt hs2_6 (by dsimp [woff] at hinsn_lim; omega)]
  let s3 := { s2.set 6 (some (pc * 16)) with fl := X86Int.Flags.undef, rip := 4194788 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) (pc * 16)
  have hs3_6 : s3.reg 6 = some (pc * 16) := by
    simp only [s3, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
  have hs3_0 : s3.reg 0 = some woff := by
    simp only [s3, s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 6 ≠ 0)]; exact hs1_0
  let iboff := pc * 16 + woff
  have h04 : X86Vm.step vmImage s3 =
      .ok { s3.set 6 (some iboff) with fl := (X86Int.aluF .add (pc * 16) woff).2, rip := 4194791 } := by
    rw [step_at_chunk 1 rfl rfl (by decide : (4194788, (.ofInt (.alu .add 6 0)), 3) ∈ SporosVmBin.vmTable_1),
      exec_alu hs3_6 hs3_0]
    dsimp only
    rw [aluF_add_val, Nat.mod_eq_of_lt (by dsimp [iboff, woff] at hinsn_lim ⊢; omega)]
  let s4 := { s3.set 6 (some iboff) with fl := (X86Int.aluF .add (pc * 16) woff).2, rip := 4194791 }
  have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) iboff
  have hs4_6 : s4.reg 6 = some iboff := by
    simp only [s4, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
  have hs4_15 : s4.reg 15 = some ctxBase := by
    simp only [s4, s3, s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 6 ≠ 15)]; exact hs1_15
  have hs4_4 : s4.reg 4 = some vc.sp := by
    simp only [s4, s3, s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 6 ≠ 4)]; exact hs1_4
  have hc1_4 : rd64 s4.mem (ea ctxBase 232) = some vc.code1.base := by rw [show s4.mem = s.mem from hmem1]; exact hc1
  have hc2_4 : rd64 s4.mem (ea ctxBase 240) = some vc.code2.base := by rw [show s4.mem = s.mem from hmem1]; exact hc2
  have hc3_4 : rd64 s4.mem (ea ctxBase 248) = some vc.code3.base := by rw [show s4.mem = s.mem from hmem1]; exact hc3
  have hi1 := rd64_flat_insn hmem (vc.mem_code1 h) hf1 hfit hw hpc
  have hi2 := rd64_flat_insn hmem (vc.mem_code2 h) hf2 hfit hw hpc
  have hi3 := rd64_flat_insn hmem (vc.mem_code3 h) hf3 hfit hw hpc
  have hi1_2 := hi1.2; have hi2_2 := hi2.2; have hi3_2 := hi3.2
  rw [← hmem1] at hi1_2 hi2_2 hi3_2
  have hargW := instrPair_snd_lt_W hfit hw hpc
  obtain ⟨s5, hrch5, hrip5, hmem5, hk5, hr5, hs5_0, _, _, hs5_oth⟩ :=
    tmr_ldIdx_vote_eq_spec (s := s4) rfl hr4 hs4_15 hs4_6
      (mem_vmTable_of_chunk 1 rfl (by decide : (4194791, (.ofInt (.ld 2 15 232)), 7) ∈ SporosVmBin.vmTable_1))
      (mem_vmTable_of_chunk 1 rfl (by decide : (4194798, (.ldIdx .w64 0 2 6 1 8), 5) ∈ SporosVmBin.vmTable_1))
      (mem_vmTable_of_chunk 1 rfl (by decide : (4194803, (.ofInt (.ld 2 15 240)), 7) ∈ SporosVmBin.vmTable_1))
      (mem_vmTable_of_chunk 1 rfl (by decide : (4194810, (.ldIdx .w64 3 2 6 1 8), 5) ∈ SporosVmBin.vmTable_1))
      (mem_vmTable_of_chunk 1 rfl (by decide : (4194815, (.ofInt (.ld 2 15 248)), 7) ∈ SporosVmBin.vmTable_1))
      (mem_vmTable_of_chunk 1 rfl (by decide : (4194822, (.ldIdx .w64 1 2 6 1 8), 5) ∈ SporosVmBin.vmTable_1))
      (mem_vmTable_of_chunk 1 rfl (by decide : (4194827, (.ofInt (.cmp 0 3)), 3) ∈ SporosVmBin.vmTable_1))
      (mem_vmTable_of_chunk 1 rfl (by decide : (4194830, (.ofInt (.jcc 4 4194850)), 2) ∈ SporosVmBin.vmTable_1))
      hc1_4 hc2_4 hc3_4 hi1_2 hi2_2 hi3_2 hargW
  have hs5_15 : s5.reg 15 = some ctxBase := by rw [hs5_oth 15 (by decide) (by decide) (by decide) (by decide)]; exact hs4_15
  have hs5_6 : s5.reg 6 = some iboff := by rw [hs5_oth 6 (by decide) (by decide) (by decide) (by decide)]; exact hs4_6
  have hs5_4 : s5.reg 4 = some vc.sp := by rw [hs5_oth 4 (by decide) (by decide) (by decide) (by decide)]; exact hs4_4
  have hmem5_eq : s5.mem = s.mem := by rw [hmem5]; exact hmem1
  have hwf5 : vc.wf pr ls s5 := vc.wf_step_misc h hmem5_eq hr5 hs5_4 hs5_15
  obtain ⟨idx1, m1, hidx1, hm1_eq, hstep6, hwf6, hc264, hrest6⟩ :=
    vc.wr64_ctx_scratch_slot hwf5 264 _ 4194857 (by decide) (by decide) hs5_0
      (fun {m'} hwr => by
        rw [step_at_chunk 1 hrip5 rfl (by decide : (4194850, (.ofInt (.st 15 264 0)), 7) ∈ SporosVmBin.vmTable_1),
          exec_st hs5_15 hs5_0 hwr, hrip5])
  let s6 := { s5 with mem := m1, rip := 4194857 }
  have hc264' : leVal (({ vc.ctxReg with data := writeBytes vc.ctxReg.data 264 (bytes8 (instrPair pr.words[w]!.code[pc]!).2) }.data.drop 264).take 8) =
      (instrPair pr.words[w]!.code[pc]!).2 := by
    rw [hc264, Nat.mod_eq_of_lt hargW]
  have hs6_15 : s6.reg 15 = some ctxBase := hs5_15
  have hs6_6 : s6.reg 6 = some iboff := hs5_6
  have hs6_4 : s6.reg 4 = some vc.sp := hs5_4
  have hr6' : RegsOk s6 := hr5
  let vc1 := { vc with ctxReg := { vc.ctxReg with data := writeBytes vc.ctxReg.data 264 (bytes8 (instrPair pr.words[w]!.code[pc]!).2) } }
  have hwf6' : vc1.wf pr ls s6 := hwf6
  have ⟨_, _, _, _, hc1_6, hc2_6, hc3_6⟩ := vc1.rd_ctx_rs_code_slots hwf6'
  have hop1 := hi1.1; have hop2 := hi2.1; have hop3 := hi3.1
  have hm1_code1 : vc.code1 ∈ m1 := hwf6.2.2.2.2.2.2.2.1 vc.code1 (by simp [VmCtx.regions])
  have hm1_code2 : vc.code2 ∈ m1 := hwf6.2.2.2.2.2.2.2.1 vc.code2 (by simp [VmCtx.regions])
  have hm1_code3 : vc.code3 ∈ m1 := hwf6.2.2.2.2.2.2.2.1 vc.code3 (by simp [VmCtx.regions])
  have hop1' := (rd64_flat_insn hwf6.1.2 hm1_code1 hf1 hfit hw hpc).1
  have hop2' := (rd64_flat_insn hwf6.1.2 hm1_code2 hf2 hfit hw hpc).1
  have hop3' := (rd64_flat_insn hwf6.1.2 hm1_code3 hf3 hfit hw hpc).1
  -- Steps 4194857..4194890
  have h07 : X86Vm.step vmImage s6 = .ok { s6.set 2 (some vc.code1.base) with rip := 4194864 } := by
    rw [step_at_chunk 1 rfl rfl (by decide : (4194857, (.ofInt (.ld 2 15 232)), 7) ∈ SporosVmBin.vmTable_1),
      exec_ld hs6_15 hc1_6]
  let s7 := { s6.set 2 (some vc.code1.base) with rip := 4194864 }
  have hr7 : RegsOk s7 := regsOk_set hr6' (by decide) vc.code1.base
  have hs7_2 : s7.reg 2 = some vc.code1.base := by
    simp only [s7, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr6'.1]; decide)]
  have hs7_6 : s7.reg 6 = some iboff := by
    simp only [s7, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 6)]; exact hs6_6
  have h08 : X86Vm.step vmImage s7 = .ok { s7.set 0 (some (instrPair pr.words[w]!.code[pc]!).1) with rip := 4194868 } := by
    rw [step_at_chunk 1 rfl rfl (by decide : (4194864, (.ldIdx .w64 0 2 6 1 0), 4) ∈ SporosVmBin.vmTable_1),
      exec_ldIdx hs7_2 hs7_6 (show rdMem .w64 s7.mem (eaIdx vc.code1.base iboff 1 0) = some _ from hop1')]
  let s8 := { s7.set 0 (some (instrPair pr.words[w]!.code[pc]!).1) with rip := 4194868 }
  have hr8 : RegsOk s8 := regsOk_set hr7 (by decide) (instrPair pr.words[w]!.code[pc]!).1
  have hs8_15 : s8.reg 15 = some ctxBase := by
    simp only [s8, s7, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 0 ≠ 15), List.getElem?_set_ne (by decide : 2 ≠ 15)]; exact hs6_15
  have hs8_6 : s8.reg 6 = some iboff := by
    simp only [s8, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 6)]; exact hs7_6
  have h09 : X86Vm.step vmImage s8 = .ok { s8.set 2 (some vc.code2.base) with rip := 4194875 } := by
    rw [step_at_chunk 1 rfl rfl (by decide : (4194868, (.ofInt (.ld 2 15 240)), 7) ∈ SporosVmBin.vmTable_1),
      exec_ld hs8_15 hc2_6]
  let s9 := { s8.set 2 (some vc.code2.base) with rip := 4194875 }
  have hr9 : RegsOk s9 := regsOk_set hr8 (by decide) vc.code2.base
  have hs9_2 : s9.reg 2 = some vc.code2.base := by
    simp only [s9, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr8.1]; decide)]
  have hs9_6 : s9.reg 6 = some iboff := by
    simp only [s9, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 6)]; exact hs8_6
  have h10 : X86Vm.step vmImage s9 = .ok { s9.set 3 (some (instrPair pr.words[w]!.code[pc]!).1) with rip := 4194879 } := by
    rw [step_at_chunk 1 rfl rfl (by decide : (4194875, (.ldIdx .w64 3 2 6 1 0), 4) ∈ SporosVmBin.vmTable_1),
      exec_ldIdx hs9_2 hs9_6 (show rdMem .w64 s9.mem (eaIdx vc.code2.base iboff 1 0) = some _ from hop2')]
  let s10 := { s9.set 3 (some (instrPair pr.words[w]!.code[pc]!).1) with rip := 4194879 }
  have hr10 : RegsOk s10 := regsOk_set hr9 (by decide) (instrPair pr.words[w]!.code[pc]!).1
  have hs10_15 : s10.reg 15 = some ctxBase := by
    simp only [s10, s9, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 3 ≠ 15), List.getElem?_set_ne (by decide : 2 ≠ 15)]; exact hs8_15
  have hs10_6 : s10.reg 6 = some iboff := by
    simp only [s10, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 3 ≠ 6)]; exact hs9_6
  have h11 : X86Vm.step vmImage s10 = .ok { s10.set 2 (some vc.code3.base) with rip := 4194886 } := by
    rw [step_at_chunk 1 rfl rfl (by decide : (4194879, (.ofInt (.ld 2 15 248)), 7) ∈ SporosVmBin.vmTable_1),
      exec_ld hs10_15 hc3_6]
  let s11 := { s10.set 2 (some vc.code3.base) with rip := 4194886 }
  have hr11 : RegsOk s11 := regsOk_set hr10 (by decide) vc.code3.base
  have hs11_2 : s11.reg 2 = some vc.code3.base := by
    simp only [s11, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr10.1]; decide)]
  have hs11_6 : s11.reg 6 = some iboff := by
    simp only [s11, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 6)]; exact hs10_6
  have h12 : X86Vm.step vmImage s11 =
      .ok { s11.set 1 (some (instrPair pr.words[w]!.code[pc]!).1) with rip := SporosVmBin.L_vm_vote } := by
    rw [step_at_chunk 1 rfl rfl (by decide : (4194886, (.ldIdx .w64 1 2 6 1 0), 4) ∈ SporosVmBin.vmTable_1),
      exec_ldIdx hs11_2 hs11_6 (show rdMem .w64 s11.mem (eaIdx vc.code3.base iboff 1 0) = some _ from hop3')]; rfl
  let s12 := { s11.set 1 (some (instrPair pr.words[w]!.code[pc]!).1) with rip := SporosVmBin.L_vm_vote }
  have hr12' : RegsOk s12 := regsOk_set hr11 (by decide) (instrPair pr.words[w]!.code[pc]!).1
  have hs12_0 : s12.reg 0 = some (instrPair pr.words[w]!.code[pc]!).1 := by
    simp only [s12, s11, s10, s9, s8, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 0), List.getElem?_set_ne (by decide : 2 ≠ 0),
      List.getElem?_set_ne (by decide : 3 ≠ 0)]
    rw [getElem?_set_getD_self (by rw [hr7.1]; decide)]
  have hs12_3 : s12.reg 3 = some (instrPair pr.words[w]!.code[pc]!).1 := by
    simp only [s12, s11, s10, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 3), List.getElem?_set_ne (by decide : 2 ≠ 3)]
    rw [getElem?_set_getD_self (by rw [hr9.1]; decide)]
  have hopW : (instrPair pr.words[w]!.code[pc]!).1 < X86Env.W := by
    cases pr.words[w]!.code[pc]! with
    | prim p => exact Nat.lt_trans (primCode_lt p) (by decide)
    | _ => simp [instrPair]; decide
  obtain ⟨s13, hrch13, hrip13, hmem13, hk13, hr13', hs13_0, hs13_1, hs13_3, hs13_oth⟩ :=
    L_vm_vote_eq_spec rfl hr12' hs12_0 hs12_3 hopW
  have hs13_15 : s13.reg 15 = some ctxBase := by
    rw [hs13_oth 15 (by decide) (by decide)]
    simp only [s12, s11, s10, s9, s8, s7, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 15), List.getElem?_set_ne (by decide : 2 ≠ 15)]
    exact hs10_15
  have hs13_4 : s13.reg 4 = some vc.sp := by
    rw [hs13_oth 4 (by decide) (by decide)]
    simp only [s12, s11, s10, s9, s8, s7, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 4), List.getElem?_set_ne (by decide : 2 ≠ 4),
      List.getElem?_set_ne (by decide : 3 ≠ 4), List.getElem?_set_ne (by decide : 0 ≠ 4)]
    exact hs6_4
  have hwf13 : vc1.wf pr ls s13 := vc1.wf_step_misc hwf6 hmem13 hr13' hs13_4 hs13_15
  have hreach_all : Reaches vmImage s s13 :=
    Reaches.trans hrch1 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
      (Reaches.trans hrch5 (Reaches.step_trans hstep6 (Reaches.step_trans h07 (Reaches.step_trans h08
        (Reaches.step_trans h09 (Reaches.step_trans h10 (Reaches.step_trans h11
          (Reaches.step_trans h12 hrch13)))))))))))
  have hm_eq : s13.mem = m1 := hmem13
  have hk_eq : s13.k = s.k := by rw [hk13]; exact hk5.trans hk1
  rw [hmem5_eq] at hidx1 hm1_eq
  exact ⟨idx1, m1, s13, hidx1, hm1_eq, hreach_all, hrip13, hm_eq, hk_eq, hwf13,
    hs13_0, hs13_1, hs13_3, hc264'⟩

def instrEntry : Sporos.Instr → Nat
  | .prim _ => SporosVmBin.L_prim_prologue
  | .lit _ => SporosVmBin.O_lit
  | .str _ => SporosVmBin.O_str
  | .pick _ => SporosVmBin.O_pick
  | .tor => SporosVmBin.O_tor
  | .fromr => SporosVmBin.O_fromr
  | .rfetch => SporosVmBin.O_rfetch
  | .locget _ => SporosVmBin.O_locget
  | .locset _ => SporosVmBin.O_locset
  | .jmp _ => SporosVmBin.O_jmp
  | .jz _ => SporosVmBin.O_jz
  | .jnz _ => SporosVmBin.O_jnz
  | .call _ => SporosVmBin.O_call
  | .ret => SporosVmBin.O_ret

theorem instrPair_fst_lt_70 (ins : Sporos.Instr) : (instrPair ins).1 < 70 := by
  cases ins with
  | prim p => exact Nat.lt_trans (primCode_lt p) (by decide)
  | lit _ | str _ | pick _ | tor | fromr | rfetch | locget _ | locset _ | jmp _ | jz _ | jnz _ | call _ | ret =>
    simp [instrPair]

theorem expectedJtab_instr (ins : Sporos.Instr) :
    expectedJtab[(instrPair ins).1]? = some (instrEntry ins) := by
  cases ins with
  | prim p =>
    have hp := primCode_lt p
    rw [instrPair, instrEntry, expectedJtab, List.getElem?_append_left (by simp [hp]),
      List.getElem?_replicate]
    simp [hp]
  | lit _ | str _ | pick _ | tor | fromr | rfetch | locget _ | locset _ | jmp _ | jz _ | jnz _ | call _ | ret =>
    dsimp only [instrPair, instrEntry, expectedJtab]
    rw [List.getElem?_append_right (by rw [List.length_replicate]; decide), List.length_replicate]
    rfl

private theorem jtab_bounds_ok :
    SporosVmBin.L_jtab + 8 * 70 ≤ baseAddr + SporosVmBin.vmBytes.length := by
  rw [vmBytes_len_eq]; decide

theorem L_vm_dispatch_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} (ins : Sporos.Instr)
    (hrip : s.rip = SporosVmBin.L_vm_dispatch) (h : vc.wf pr ls s)
    (hr0 : s.reg 0 = some (instrPair ins).1) :
    let op := (instrPair ins).1
    let ctx1 := { vc.ctxReg with data := writeBytes vc.ctxReg.data 256 (bytes8 op) }
    ∃ idx1 m1 s',
      s.mem[idx1]? = some vc.ctxReg ∧
      m1 = s.mem.set idx1 ctx1 ∧
      Reaches vmImage s s' ∧
      s'.rip = instrEntry ins ∧ s'.mem = m1 ∧ s'.k = s.k ∧
      { vc with ctxReg := ctx1 }.wf pr ls s' ∧
      s'.reg 0 = some op ∧
      leVal ((ctx1.data.drop 256).take 8) = op ∧
      (∀ k, (k + 8 ≤ 256 ∨ 264 ≤ k) → k + 8 ≤ ctxSize →
        leVal ((ctx1.data.drop k).take 8) = leVal ((vc.ctxReg.data.drop k).take 8)) := by
  have hop70 := instrPair_fst_lt_70 ins
  have hopW : (instrPair ins).1 < X86Env.W := Nat.lt_trans hop70 (by decide)
  obtain ⟨idx1, m1, hidx1, hm1_eq, h01, hwf1, hc256, hrest⟩ :=
    vc.wr64_ctx_scratch_slot h 256 _ 4194935 (by decide) (by decide) hr0
      (fun {m'} hwr => by
        have hm1' : (SporosVmBin.L_vm_dispatch, (.ofInt (.st 15 256 0)), 7) ∈ SporosVmBin.vmTable_1 := by decide
        rw [step_at_chunk 1 hrip rfl hm1', exec_st h.2.2.2.2.2.1 hr0 hwr, hrip]
        rfl)
  let s1 := { s with mem := m1, rip := 4194935 }
  have hs1_0 : s1.reg 0 = some (instrPair ins).1 := hr0
  have hr1 : RegsOk s1 := h.2.2.2.1
  have h02 : X86Vm.step vmImage s1 =
      .ok { s1 with fl := (X86Int.aluF .sub (instrPair ins).1 70).2, rip := 4194939 } := by
    rw [step_at_chunk 1 rfl rfl (by decide : (4194935, (.ofInt (.cmpI 0 70)), 4) ∈ SporosVmBin.vmTable_1),
      exec_cmpI hs1_0, Nat.mod_eq_of_lt (by decide)]
  let s2 := { s1 with fl := (X86Int.aluF .sub (instrPair ins).1 70).2, rip := 4194939 }
  have h03 : X86Vm.step vmImage s2 = .ok { s2 with rip := 4194945 } := by
    rw [step_at_chunk 1 rfl rfl (by decide : (4194939, (.ofInt (.jcc 3 4201099)), 6) ∈ SporosVmBin.vmTable_1),
      exec_jcc (cond_aluF_sub_ae (instrPair ins).1 70)]
    simp [s2, show ¬ (70 ≤ (instrPair ins).1) by omega]
  let s3 := { s2 with rip := 4194945 }
  have h04 : X86Vm.step vmImage s3 = .ok { s3.set 2 (some 4205803) with rip := 4194950 } := by
    rw [step_at_chunk 1 rfl rfl (by decide : (4194945, (.ofInt (.movI 2 4205803)), 5) ∈ SporosVmBin.vmTable_1),
      exec_movI_lt (by decide)]
  let s4 := { s3.set 2 (some 4205803) with rip := 4194950 }
  have hr4 : RegsOk s4 := regsOk_set hr1 (by decide) 4205803
  have hs4_2 : s4.reg 2 = some 4205803 := by
    simp only [s4, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  have hs4_0 : s4.reg 0 = some (instrPair ins).1 := by
    simp only [s4, s3, s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 0)]; exact hs1_0
  have hjtab_get : (roQwords SporosVmBin.L_jtab 70)[(instrPair ins).1]? = some (instrEntry ins) := by
    rw [ro_jtab_ok]; exact expectedJtab_instr ins
  have hrd_t : rd64 s4.mem ((4205803 + 8 * (instrPair ins).1 + 0) % X86Env.W) = some (instrEntry ins) := by
    have heq : 4205803 = SporosVmBin.L_jtab := rfl
    have h47 : (SporosVmBin.L_jtab + 8 * (instrPair ins).1 + 0) < X86Env.W := by
      have : SporosVmBin.L_jtab = 4205803 := rfl
      have : X86Env.W = 2 ^ 64 := rfl
      omega
    rw [heq, Nat.mod_eq_of_lt h47, Nat.add_zero]
    exact roQwords_rd64 hwf1.1 (by decide : vmCodeEnd ≤ SporosVmBin.L_jtab) jtab_bounds_ok hop70 hjtab_get
  have h05 : X86Vm.step vmImage s4 = .ok { s4 with rip := instrEntry ins } := by
    rw [step_at_chunk 1 rfl rfl (by decide : (4194950, (.ofInt (.jtab 2 0 0)), 3) ∈ SporosVmBin.vmTable_1),
      exec_jtab hs4_2 hs4_0 hrd_t]
  let s5 := { s4 with rip := instrEntry ins }
  have hs5_15 : s5.reg 15 = some ctxBase := by
    simp only [s5, s4, s3, s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 15)]
    exact h.2.2.2.2.2.1
  have hs5_4 : s5.reg 4 = some vc.sp := by
    simp only [s5, s4, s3, s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 4)]
    exact h.2.2.2.2.1
  let ctx1 := { vc.ctxReg with data := writeBytes vc.ctxReg.data 256 (bytes8 (instrPair ins).1) }
  have hwf5 : { vc with ctxReg := ctx1 }.wf pr ls s5 :=
    ({ vc with ctxReg := ctx1 }).wf_step_misc hwf1 rfl hr4 hs5_4 hs5_15
  have hreach : Reaches vmImage s s5 :=
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04 (Reaches.step h05))))
  have hc256' : leVal ((ctx1.data.drop 256).take 8) = (instrPair ins).1 := by
    rw [hc256, Nat.mod_eq_of_lt hopW]
  exact ⟨idx1, m1, s5, hidx1, hm1_eq, hreach, rfl, rfl, rfl, hwf5, hs4_0, hc256', hrest⟩

theorem L_vm_loop_dispatch_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s₀ : X86Vm.St} (hrel : VmRel pr ls s₀) (hvc : vc.wf pr ls s₀) :
    match ls.rs with
    | [] => ∃ s_exit, Reaches vmImage s₀ s_exit ∧ X86Vm.step vmImage s_exit = .exit 1 ∧ s_exit.k = s₀.k
    | f :: _ =>
      if _hnw : f.word < pr.words.length then
        let wd := pr.words[f.word]!
        if _hpc : f.pc < wd.code.length then
          let ins := wd.code[f.pc]!
          let ctx' := { vc.ctxReg with data := writeBytes (writeBytes vc.ctxReg.data 264 (bytes8 (instrPair ins).2)) 256 (bytes8 (instrPair ins).1) }
          ∃ s', Reaches vmImage s₀ s' ∧
            s'.rip = instrEntry ins ∧ s'.k = s₀.k ∧
            { vc with ctxReg := ctx' }.wf pr ls s' ∧
            s'.reg 0 = some (instrPair ins).1 ∧
            leVal ((ctx'.data.drop 256).take 8) = (instrPair ins).1 ∧
            leVal ((ctx'.data.drop 264).take 8) = (instrPair ins).2
        else ∃ s_exit, Reaches vmImage s₀ s_exit ∧ X86Vm.step vmImage s_exit = .exit 1 ∧ s_exit.k = s₀.k
      else ∃ s_exit, Reaches vmImage s₀ s_exit ∧ X86Vm.step vmImage s_exit = .exit 1 ∧ s_exit.k = s₀.k := by
  have hfw := L_vm_loop_fetch_words_spec hrel.1 hvc
  cases hrs : ls.rs with
  | nil => rw [hrs] at hfw; exact hfw
  | cons f rest =>
    rw [hrs] at hfw
    dsimp only at hfw ⊢
    by_cases hnw : f.word < pr.words.length
    · rw [dif_pos hnw] at hfw ⊢
      by_cases hpc : f.pc < pr.words[f.word]!.code.length
      · rw [dif_pos hpc] at hfw ⊢
        obtain ⟨s1, hrch1, hrip1, hmem1, hk1, hwf1, _, hr13, hr6⟩ := hfw
        have hwf1' : vc.wf pr ls s1 := by simpa [hrs] using hwf1
        obtain ⟨idx1, m1, s2, hidx1, hm1_eq, hrch2, hrip2, hmem2, hk2, hwf2, hr2_0, _, _, hc264⟩ :=
          L_vm_loop_fetch_insn_spec hrip1 hwf1' ‹_› hr13 hr6 hnw hpc
        let ins := pr.words[f.word]!.code[f.pc]!
        let vc1 := { vc with ctxReg := { vc.ctxReg with data := writeBytes vc.ctxReg.data 264 (bytes8 (instrPair ins).2) } }
        obtain ⟨idx2, m2, s3, _, _, hrch3, hrip3, _, hk3, hwf3, hr3_0, hc256, hrest⟩ :=
          L_vm_dispatch_spec ins hrip2 hwf2 hr2_0
        have hc264' : leVal ((({ vc.ctxReg with data := writeBytes (writeBytes vc.ctxReg.data 264 (bytes8 (instrPair ins).2)) 256 (bytes8 (instrPair ins).1) }.data).drop 264).take 8) =
            (instrPair ins).2 := by
          rw [hrest 264 (Or.inr (by decide)) (by decide)]
          exact hc264
        exact ⟨s3, Reaches.trans hrch1 (Reaches.trans hrch2 hrch3), hrip3, by rw [hk3, hk2, hk1],
          hwf3, hr3_0, hc256, hc264'⟩
      · rw [dif_neg hpc] at hfw ⊢; exact hfw
    · rw [dif_neg hnw] at hfw ⊢; exact hfw

/-! ### Part 2 Tail: Shared primitive prologue, epilogue, and PC-advance simulation -/

def primEntry : Sporos.Prim → Nat
  | .dup => SporosVmBin.P_dup
  | .drop => SporosVmBin.P_drop
  | .swap => SporosVmBin.P_swap
  | .over => SporosVmBin.P_over
  | .rot => SporosVmBin.P_rot
  | .nip => SporosVmBin.P_nip
  | .tuck => SporosVmBin.P_tuck
  | .add => SporosVmBin.P_add
  | .sub => SporosVmBin.P_sub
  | .mul => SporosVmBin.P_mul
  | .divu => SporosVmBin.P_divu
  | .modu => SporosVmBin.P_modu
  | .divs => SporosVmBin.P_divs
  | .mods => SporosVmBin.P_mods
  | .neg => SporosVmBin.P_neg
  | .band => SporosVmBin.P_band
  | .bor => SporosVmBin.P_bor
  | .bxor => SporosVmBin.P_bxor
  | .bnot => SporosVmBin.P_bnot
  | .shl => SporosVmBin.P_shl
  | .shr => SporosVmBin.P_shr
  | .sar => SporosVmBin.P_sar
  | .eq => SporosVmBin.P_eq
  | .ne => SporosVmBin.P_ne
  | .ltu => SporosVmBin.P_ltu
  | .lts => SporosVmBin.P_lts
  | .leu => SporosVmBin.P_leu
  | .les => SporosVmBin.P_les
  | .ld8 => SporosVmBin.P_ld8
  | .st8 => SporosVmBin.P_st8
  | .ld16 => SporosVmBin.P_ld16
  | .st16 => SporosVmBin.P_st16
  | .ld32 => SporosVmBin.P_ld32
  | .st32 => SporosVmBin.P_st32
  | .ld64 => SporosVmBin.P_ld64
  | .st64 => SporosVmBin.P_st64
  | .copy => SporosVmBin.P_copy
  | .fill => SporosVmBin.P_fill
  | .mcmp => SporosVmBin.P_mcmp
  | .alloc => SporosVmBin.P_alloc
  | .mark => SporosVmBin.P_mark
  | .release => SporosVmBin.P_release
  | .streq => SporosVmBin.P_streq
  | .strhash => SporosVmBin.P_strhash
  | .tnew => SporosVmBin.P_tnew
  | .tput => SporosVmBin.P_tput
  | .tget => SporosVmBin.P_tget
  | .getb => SporosVmBin.P_getb
  | .putb => SporosVmBin.P_putb
  | .putn => SporosVmBin.P_putn
  | .exit => SporosVmBin.P_exit
  | .clock | .core | .sensor => SporosVmBin.P_zero_mk1
  | .provget => SporosVmBin.P_provget
  | .confget => SporosVmBin.P_confget
  | .demand => SporosVmBin.P_demand

theorem prim_eff_le_4 (p : Sporos.Prim) : p.eff.1 ≤ 4 ∧ p.eff.2 ≤ 3 := by
  cases p <;> decide

theorem expectedPrimJtab_prim (p : Sporos.Prim) :
    expectedPrimJtab[primCode p]? = some (primEntry p) := by
  cases p <;> rfl

theorem effIn_prim_get (p : Sporos.Prim) :
    (roSlice SporosVmBin.L_prim_eff_in 57)[primCode p]? = some p.eff.1 := by
  rw [ro_effIn_ok]; cases p <;> rfl

theorem effOut_prim_get (p : Sporos.Prim) :
    (roSlice SporosVmBin.L_prim_eff_out 57)[primCode p]? = some p.eff.2 := by
  rw [ro_effOut_ok]; cases p <;> rfl

private theorem effIn_bounds_ok :
    SporosVmBin.L_prim_eff_in + 57 ≤ baseAddr + SporosVmBin.vmBytes.length := by
  rw [vmBytes_len_eq]; decide

private theorem effOut_bounds_ok :
    SporosVmBin.L_prim_eff_out + 57 ≤ baseAddr + SporosVmBin.vmBytes.length := by
  rw [vmBytes_len_eq]; decide

private theorem primJtab_bounds_ok :
    SporosVmBin.L_prim_jtab + 8 * 57 ≤ baseAddr + SporosVmBin.vmBytes.length := by
  rw [vmBytes_len_eq]; decide

private theorem lq_take_step_mod (xs : List SporosProvLen.LVal)
    (k : Nat) (hk : k < xs.length) :
    (xs.take (k + 1)).foldr (fun v acc => v.plen + acc) 0 =
      (xs.take k).foldr (fun v acc => v.plen + acc) 0 + xs[k]!.plen := by
  rw [List.take_add_one, List.getElem?_eq_getElem hk, Option.toList_some, List.foldr_append]
  simp only [List.foldr, Nat.add_zero]
  induction xs.take k with
  | nil => simp [hk]
  | cons y ys ih => simp only [List.foldr_cons, ih, Nat.add_assoc]

private theorem lq_mod_step (acc plen : Nat) :
    ((acc % X86Env.W + plen) % X86Env.W) = (acc + plen) % X86Env.W := by
  rw [Nat.add_mod, Nat.mod_mod, ← Nat.add_mod]

private theorem ds_elem_bounds {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) :
    ∀ v ∈ ls.ds, v.w < X86Env.W ∧ v.plen < X86Env.W := by
  obtain ⟨_, _, _, _, _, _, _, _,
    _, _, _, _,  _, _, _, _,  _, _, _,
    _, _, _, _, _,  _, _, _, _, _,  _, _, _, _, _,  _, _, _, _, _,
    _, _, _, _,  _, _, _,  _, _, _, _, _,  _, _, _, _, _, _,  _,  _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, hwlt, _⟩ := h
  simpa [M_eq_W] using hwlt.1

private theorem prologue_arg_rd64 {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) {k : Nat} (hk : k < ls.ds.length) :
    ls.ds[k]!.w < X86Env.W ∧ ls.ds[k]!.plen < X86Env.W ∧
    (ls.ds.length - (k + 1)) * 16 + 16 ≤ vc.dsReg.data.length ∧
    vc.dsReg.base + (ls.ds.length - (k + 1)) * 16 + 16 < X86Env.W ∧
    rd64 s.mem (eaIdx vc.dsReg.base ((ls.ds.length - (k + 1)) * 16) 1 0) = some ls.ds[k]!.w ∧
    rd64 s.mem (eaIdx vc.dsReg.base ((ls.ds.length - (k + 1)) * 16) 1 8) = some ls.ds[k]!.plen := by
  have ⟨⟨_, _, hdsLen, _, hdsHi, hpds, _⟩, _, _⟩ := vc.arena_bounds h
  have hmem := h.1.2
  have hmem_ds := vc.mem_ds h
  have hmemk : ls.ds[k]! ∈ ls.ds := by
    simp only [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem hk, Option.getD_some]
    exact List.getElem_mem hk
  have ⟨hwW, hpW⟩ := ds_elem_bounds h ls.ds[k]! hmemk
  have hlim : (ls.ds.length - (k + 1)) * 16 + 16 ≤ vc.dsReg.data.length := by
    have : ls.ds.length - (k + 1) + 1 ≤ ls.ds.length := by omega
    rw [hdsLen]; omega
  have h47 : 2 ^ 47 < X86Env.W := by decide
  have hbaseW : vc.dsReg.base + (ls.ds.length - (k + 1)) * 16 + 16 < X86Env.W := by
    simp only [Region.hi] at hdsHi; omega
  have ⟨hw0, hp0⟩ := pairListRel_get_top hpds hk
  have hrw : rd64 s.mem (eaIdx vc.dsReg.base ((ls.ds.length - (k + 1)) * 16) 1 0) = some ls.ds[k]!.w := by
    rw [eaIdx_add_eq (by omega)]
    refine rd64_of_mem hmem hmem_ds (by omega) (by simp only [Region.hi] at hdsHi ⊢; omega) ?_
    rw [show vc.dsReg.base + 1 * ((ls.ds.length - (k + 1)) * 16) + 0 - vc.dsReg.base =
      (ls.ds.length - 1 - k) * 16 by omega]
    exact hw0
  have hrp : rd64 s.mem (eaIdx vc.dsReg.base ((ls.ds.length - (k + 1)) * 16) 1 8) = some ls.ds[k]!.plen := by
    rw [eaIdx_add_eq (by omega)]
    refine rd64_of_mem hmem hmem_ds (by omega) (by simp only [Region.hi] at hdsHi ⊢; omega) ?_
    rw [show vc.dsReg.base + 1 * ((ls.ds.length - (k + 1)) * 16) + 8 - vc.dsReg.base =
      (ls.ds.length - 1 - k) * 16 + 8 by omega]
    exact hp0
  exact ⟨hwW, hpW, hlim, hbaseW, hrw, hrp⟩

private theorem prologue_load_slot_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {a offW offP k qa : Nat}
    (hrip : s.rip = a) (h : vc.wf pr ls s) (hk : k < ls.ds.length)
    (hoffW : 256 ≤ offW) (hoffWlim : offW + 8 ≤ ctxSize)
    (hoffP : 256 ≤ offP) (hoffPlim : offP + 8 ≤ ctxSize)
    (hdisj : offW + 8 ≤ offP)
    (hr1 : s.reg 1 = some ls.ds.length) (hr6 : s.reg 6 = some vc.dsReg.base)
    (hr10 : s.reg 10 = some (qa % X86Env.W))
    (hm1 : (a, (.ofInt (.mov 2 1)), 3) ∈ SporosVmBin.vmTable_2)
    (hm2 : (a + 3, (.ofInt (.aluI .sub 2 (k + 1))), 4) ∈ SporosVmBin.vmTable_2)
    (hm3 : (a + 7, (.ofInt (.shlI 2 4)), 4) ∈ SporosVmBin.vmTable_2)
    (hm4 : (a + 11, (.ldIdx .w64 11 6 2 1 0), 4) ∈ SporosVmBin.vmTable_2)
    (hm5 : (a + 15, (.ofInt (.st 15 offW 11)), 7) ∈ SporosVmBin.vmTable_2)
    (hm6 : (a + 22, (.ldIdx .w64 11 6 2 1 8), 5) ∈ SporosVmBin.vmTable_2)
    (hm7 : (a + 27, (.ofInt (.st 15 offP 11)), 7) ∈ SporosVmBin.vmTable_2)
    (hm8 : (a + 34, (.ofInt (.alu .add 10 11)), 3) ∈ SporosVmBin.vmTable_2) :
    ∃ ctxData',
      let ctx' := { vc.ctxReg with data := ctxData' }
      ∃ s', Reaches vmImage s s' ∧
        s'.rip = a + 37 ∧ s'.k = s.k ∧
        { vc with ctxReg := ctx' }.wf pr ls s' ∧
        ctxData'.length = vc.ctxReg.data.length ∧
        s'.reg 1 = some ls.ds.length ∧ s'.reg 6 = some vc.dsReg.base ∧
        s'.reg 10 = some ((qa + ls.ds[k]!.plen) % X86Env.W) ∧
        (∀ r, r ≠ 2 → r ≠ 10 → r ≠ 11 → s'.reg r = s.reg r) ∧
        leVal ((ctxData'.drop offW).take 8) = ls.ds[k]!.w ∧
        leVal ((ctxData'.drop offP).take 8) = ls.ds[k]!.plen ∧
        (∀ m0, (m0 + 8 ≤ offW ∨ (offW + 8 ≤ m0 ∧ m0 + 8 ≤ offP) ∨ offP + 8 ≤ m0) → m0 + 8 ≤ ctxSize →
          leVal ((ctxData'.drop m0).take 8) = leVal ((vc.ctxReg.data.drop m0).take 8)) := by
  have ⟨hwW, hpW, hlim, hbw, hrw, hrp⟩ := prologue_arg_rd64 h hk
  have hr := h.2.2.2.1
  have h15 := h.2.2.2.2.2.1
  have hsp := h.2.2.2.2.1
  have h47 : 2 ^ 47 < X86Env.W := by decide
  have hk_W : k + 1 < X86Env.W := by
    have ⟨⟨_, _, hdsLen, _, hdsHi, _, _⟩, _, _⟩ := vc.arena_bounds h
    simp only [Region.hi] at hdsHi; omega
  have hsub1 : (ls.ds.length + X86Env.W - (k + 1) % X86Env.W) % X86Env.W = ls.ds.length - (k + 1) := by
    rw [Nat.mod_eq_of_lt hk_W]
    have : ls.ds.length + X86Env.W - (k + 1) = (ls.ds.length - (k + 1)) + X86Env.W := by omega
    rw [this, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have h01 : X86Vm.step vmImage s = .ok { s.set 2 (some ls.ds.length) with rip := a + 3 } := by
    rw [step_at_chunk 2 hrip rfl hm1, exec_mov hr1, hrip]
  let s1 := { s.set 2 (some ls.ds.length) with rip := a + 3 }
  have hr1' : RegsOk s1 := regsOk_set hr (by decide) ls.ds.length
  have hs1_2 : s1.reg 2 = some ls.ds.length := by
    simp only [s1, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr.1]; decide)]
  have h02 : X86Vm.step vmImage s1 =
      .ok { s1.set 2 (some (ls.ds.length - (k + 1))) with
        fl := (X86Int.aluF .sub ls.ds.length ((k + 1) % X86Env.W)).2, rip := a + 7 } := by
    rw [step_at_chunk 2 rfl rfl hm2, exec_aluI hs1_2]
    dsimp only
    rw [aluF_sub_val, hsub1]
  let s2 := { s1.set 2 (some (ls.ds.length - (k + 1))) with
    fl := (X86Int.aluF .sub ls.ds.length ((k + 1) % X86Env.W)).2, rip := a + 7 }
  have hr2' : RegsOk s2 := regsOk_set hr1' (by decide) (ls.ds.length - (k + 1))
  have hs2_2 : s2.reg 2 = some (ls.ds.length - (k + 1)) := by
    simp only [s2, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr1'.1]; decide)]
  have h03 : X86Vm.step vmImage s2 =
      .ok { s2.set 2 (some ((ls.ds.length - (k + 1)) * 16)) with
        fl := X86Int.Flags.undef, rip := a + 11 } := by
    have h16W : (ls.ds.length - (k + 1)) * 16 < X86Env.W := by
      have ⟨⟨_, _, hdsLen, _, hdsHi, _, _⟩, _, _⟩ := vc.arena_bounds h
      simp only [Region.hi] at hdsHi; omega
    rw [step_at_chunk 2 rfl rfl hm3, exec_shlI_4_lt hs2_2 (by omega)]
  let s3 := { s2.set 2 (some ((ls.ds.length - (k + 1)) * 16)) with
    fl := X86Int.Flags.undef, rip := a + 11 }
  have hr3' : RegsOk s3 := regsOk_set hr2' (by decide) ((ls.ds.length - (k + 1)) * 16)
  have hs3_2 : s3.reg 2 = some ((ls.ds.length - (k + 1)) * 16) := by
    simp only [s3, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr2'.1]; decide)]
  have hs3_6 : s3.reg 6 = some vc.dsReg.base := by
    simp only [s3, s2, s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 6)]; exact hr6
  have h04 : X86Vm.step vmImage s3 = .ok { s3.set 11 (some ls.ds[k]!.w) with rip := a + 15 } := by
    rw [step_at_chunk 2 rfl rfl hm4,
      exec_ldIdx hs3_6 hs3_2 (show rdMem .w64 s3.mem _ = some _ from hrw)]
  let s4 := { s3.set 11 (some ls.ds[k]!.w) with rip := a + 15 }
  have hr4' : RegsOk s4 := regsOk_set hr3' (by decide) ls.ds[k]!.w
  have hs4_11 : s4.reg 11 = some ls.ds[k]!.w := by
    simp only [s4, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr3'.1]; decide)]
  have hs4_15 : s4.reg 15 = some ctxBase := by
    simp only [s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 11 ≠ 15), List.getElem?_set_ne (by decide : 2 ≠ 15)]; exact h15
  have hs4_4 : s4.reg 4 = some vc.sp := by
    simp only [s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 11 ≠ 4), List.getElem?_set_ne (by decide : 2 ≠ 4)]; exact hsp
  have hwf4 : vc.wf pr ls s4 := vc.wf_step_misc h rfl hr4' hs4_4 hs4_15
  obtain ⟨_, m5, _, _, hstep5, hwf5, hcW, hrest5⟩ :=
    vc.wr64_ctx_scratch_slot hwf4 offW _ (a + 22) hoffW hoffWlim hs4_11
      (fun {m'} hwr => by
        rw [step_at_chunk 2 rfl rfl hm5, exec_st hs4_15 hs4_11 hwr])
  let s5 := { s4 with mem := m5, rip := a + 22 }
  let vc1 := { vc with ctxReg := { vc.ctxReg with data := writeBytes vc.ctxReg.data offW (bytes8 ls.ds[k]!.w) } }
  have hs5_6 : s5.reg 6 = some vc.dsReg.base := by
    simp only [s5, s4, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 11 ≠ 6)]; exact hs3_6
  have hs5_2 : s5.reg 2 = some ((ls.ds.length - (k + 1)) * 16) := by
    simp only [s5, s4, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 11 ≠ 2)]; exact hs3_2
  have hmem5_ds : vc.dsReg ∈ m5 := hwf5.2.2.2.2.2.2.2.1 vc.dsReg (by simp [VmCtx.regions])
  have hrp5 : rd64 m5 (eaIdx vc.dsReg.base ((ls.ds.length - (k + 1)) * 16) 1 8) = some ls.ds[k]!.plen := by
    have ⟨⟨_, _, _, _, hdsHi, hpds, _⟩, _, _⟩ := vc1.arena_bounds hwf5
    have ⟨_, hp0⟩ := pairListRel_get_top hpds hk
    rw [eaIdx_add_eq (by omega)]
    refine rd64_of_mem hwf5.1.2 hmem5_ds (by omega)
      (by simp only [Region.hi] at hdsHi ⊢; omega) ?_
    rw [show vc.dsReg.base + 1 * ((ls.ds.length - (k + 1)) * 16) + 8 - vc.dsReg.base =
      (ls.ds.length - 1 - k) * 16 + 8 by omega]
    exact hp0
  have h06 : X86Vm.step vmImage s5 = .ok { s5.set 11 (some ls.ds[k]!.plen) with rip := a + 27 } := by
    rw [step_at_chunk 2 rfl rfl hm6,
      exec_ldIdx hs5_6 hs5_2 (show rdMem .w64 s5.mem _ = some _ from hrp5)]
  let s6 := { s5.set 11 (some ls.ds[k]!.plen) with rip := a + 27 }
  have hr6' : RegsOk s6 := regsOk_set hr4' (by decide) ls.ds[k]!.plen
  have hs6_11 : s6.reg 11 = some ls.ds[k]!.plen := by
    simp only [s6, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr4'.1]; decide)]
  have hs6_15 : s6.reg 15 = some ctxBase := by
    simp only [s6, s5, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 11 ≠ 15)]; exact hs4_15
  have hs6_4 : s6.reg 4 = some vc.sp := by
    simp only [s6, s5, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 11 ≠ 4)]; exact hs4_4
  have hwf6 : vc1.wf pr ls s6 := vc1.wf_step_misc hwf5 rfl hr6' hs6_4 hs6_15
  obtain ⟨_, m7, _, _, hstep7, hwf7, hcP, hrest7⟩ :=
    vc1.wr64_ctx_scratch_slot hwf6 offP _ (a + 34) hoffP hoffPlim hs6_11
      (fun {m'} hwr => by
        rw [step_at_chunk 2 rfl rfl hm7, exec_st hs6_15 hs6_11 hwr])
  let s7 := { s6 with mem := m7, rip := a + 34 }
  have hs7_10 : s7.reg 10 = some (qa % X86Env.W) := by
    simp only [s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 11 ≠ 10), List.getElem?_set_ne (by decide : 2 ≠ 10)]; exact hr10
  have hs7_11 : s7.reg 11 = some ls.ds[k]!.plen := hs6_11
  have h08 : X86Vm.step vmImage s7 =
      .ok { s7.set 10 (some ((qa + ls.ds[k]!.plen) % X86Env.W)) with
        fl := (X86Int.aluF .add (qa % X86Env.W) ls.ds[k]!.plen).2, rip := a + 37 } := by
    rw [step_at_chunk 2 rfl rfl hm8, exec_alu hs7_10 hs7_11]
    dsimp only
    rw [aluF_add_val, lq_mod_step]
  let s8 := { s7.set 10 (some ((qa + ls.ds[k]!.plen) % X86Env.W)) with
    fl := (X86Int.aluF .add (qa % X86Env.W) ls.ds[k]!.plen).2, rip := a + 37 }
  have hr8' : RegsOk s8 := regsOk_set hr6' (by decide) ((qa + ls.ds[k]!.plen) % X86Env.W)
  have hs8_15 : s8.reg 15 = some ctxBase := by
    simp only [s8, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 10 ≠ 15)]; exact hs6_15
  have hs8_4 : s8.reg 4 = some vc.sp := by
    simp only [s8, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 10 ≠ 4)]; exact hs6_4
  let d1 := writeBytes vc.ctxReg.data offW (bytes8 ls.ds[k]!.w)
  let cd' := writeBytes d1 offP (bytes8 ls.ds[k]!.plen)
  have hlen1 : d1.length = vc.ctxReg.data.length := by
    dsimp [d1]; rw [writeBytes_length]
  have hlen' : cd'.length = vc.ctxReg.data.length := by
    dsimp [cd']; rw [writeBytes_length]; exact hlen1
  have hwf8 : { vc with ctxReg := { vc.ctxReg with data := cd' } }.wf pr ls s8 :=
    ({ vc with ctxReg := { vc.ctxReg with data := cd' } }).wf_step_misc hwf7 rfl hr8' hs8_4 hs8_15
  have hreach : Reaches vmImage s s8 :=
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
      (Reaches.step_trans hstep5 (Reaches.step_trans h06 (Reaches.step_trans hstep7 (Reaches.step h08)))))))
  have hcW' : leVal ((cd'.drop offW).take 8) = ls.ds[k]!.w := by
    rw [hrest7 offW (Or.inl hdisj) hoffWlim, hcW, Nat.mod_eq_of_lt hwW]
  have hcP' : leVal ((cd'.drop offP).take 8) = ls.ds[k]!.plen := by
    rw [hcP, Nat.mod_eq_of_lt hpW]
  refine ⟨cd', s8, hreach, rfl, rfl, hwf8, hlen', ?_, ?_, ?_, ?_, hcW', hcP', ?_⟩
  · simp only [s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 10 ≠ 1), List.getElem?_set_ne (by decide : 11 ≠ 1),
      List.getElem?_set_ne (by decide : 2 ≠ 1)]; exact hr1
  · simp only [s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 10 ≠ 6), List.getElem?_set_ne (by decide : 11 ≠ 6),
      List.getElem?_set_ne (by decide : 2 ≠ 6)]; exact hr6
  · simp only [s8, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr6'.1]; decide)]
  · intro r hr2 hr10' hr11
    simp only [s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (Ne.symm hr10'), List.getElem?_set_ne (Ne.symm hr11),
      List.getElem?_set_ne (Ne.symm hr2)]
  · intro m0 hm0 hm0lim
    rcases hm0 with hltW | ⟨hgeW, hltP⟩ | hgeP
    · rw [hrest7 m0 (Or.inl (by omega)) hm0lim, hrest5 m0 (Or.inl hltW) hm0lim]
    · rw [hrest7 m0 (Or.inl hltP) hm0lim, hrest5 m0 (Or.inr hgeW) hm0lim]
    · rw [hrest7 m0 (Or.inr hgeP) hm0lim, hrest5 m0 (Or.inr (by omega)) hm0lim]

private theorem pp_q_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} (p : Sporos.Prim) {qa : Nat}
    (hrip : s.rip = 4195181) (h : vc.wf pr ls s)
    (hr0 : s.reg 0 = some (primCode p)) (hr10 : s.reg 10 = some (qa % X86Env.W)) :
    let q := (qa + ls.pc + 1) % X86Env.W
    ∃ ctxData',
      let ctx' := { vc.ctxReg with data := ctxData' }
      ∃ s', Reaches vmImage s s' ∧
        s'.rip = primEntry p ∧ s'.k = s.k ∧
        { vc with ctxReg := ctx' }.wf pr ls s' ∧
        ctxData'.length = vc.ctxReg.data.length ∧
        leVal ((ctxData'.drop 272).take 8) = q ∧
        (∀ m0, (m0 + 8 ≤ 272 ∨ 280 ≤ m0) → m0 + 8 ≤ ctxSize →
          leVal ((ctxData'.drop m0).take 8) = leVal ((vc.ctxReg.data.drop m0).take 8)) := by
  have hr := h.2.2.2.1
  have h15 := h.2.2.2.2.2.1
  have hsp := h.2.2.2.2.1
  have hpc := (vc.rd_ctx_rs_code_slots h).2.2.2.1
  have h01 : X86Vm.step vmImage s = .ok { s.set 11 (some ls.pc) with rip := 4195188 } := by
    rw [step_at_chunk 2 hrip rfl (by decide : (4195181, (.ofInt (.ld 11 15 224)), 7) ∈ SporosVmBin.vmTable_2),
      exec_ld h15 hpc, hrip]
  let s1 := { s.set 11 (some ls.pc) with rip := 4195188 }
  have hr1 : RegsOk s1 := regsOk_set hr (by decide) ls.pc
  have hs1_11 : s1.reg 11 = some ls.pc := by
    simp only [s1, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr.1]; decide)]
  have hs1_10 : s1.reg 10 = some (qa % X86Env.W) := by
    simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 11 ≠ 10)]; exact hr10
  have h02 : X86Vm.step vmImage s1 =
      .ok { s1.set 10 (some ((qa + ls.pc) % X86Env.W)) with
        fl := (X86Int.aluF .add (qa % X86Env.W) ls.pc).2, rip := 4195191 } := by
    rw [step_at_chunk 2 rfl rfl (by decide : (4195188, (.ofInt (.alu .add 10 11)), 3) ∈ SporosVmBin.vmTable_2),
      exec_alu hs1_10 hs1_11]
    dsimp only
    rw [aluF_add_val, lq_mod_step]
  let s2 := { s1.set 10 (some ((qa + ls.pc) % X86Env.W)) with
    fl := (X86Int.aluF .add (qa % X86Env.W) ls.pc).2, rip := 4195191 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) ((qa + ls.pc) % X86Env.W)
  have hs2_10 : s2.reg 10 = some ((qa + ls.pc) % X86Env.W) := by
    simp only [s2, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  have h03 : X86Vm.step vmImage s2 =
      .ok { s2.set 10 (some ((qa + ls.pc + 1) % X86Env.W)) with
        fl := (X86Int.aluF .add ((qa + ls.pc) % X86Env.W) 1).2, rip := 4195195 } := by
    rw [step_at_chunk 2 rfl rfl (by decide : (4195191, (.ofInt (.aluI .add 10 1)), 4) ∈ SporosVmBin.vmTable_2),
      exec_aluI hs2_10, Nat.mod_eq_of_lt (by decide : 1 < X86Env.W)]
    dsimp only
    rw [aluF_add_val, lq_mod_step]
  let s3 := { s2.set 10 (some ((qa + ls.pc + 1) % X86Env.W)) with
    fl := (X86Int.aluF .add ((qa + ls.pc) % X86Env.W) 1).2, rip := 4195195 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) ((qa + ls.pc + 1) % X86Env.W)
  have hs3_10 : s3.reg 10 = some ((qa + ls.pc + 1) % X86Env.W) := by
    simp only [s3, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
  have hs3_15 : s3.reg 15 = some ctxBase := by
    simp only [s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 10 ≠ 15), List.getElem?_set_ne (by decide : 11 ≠ 15)]; exact h15
  have hs3_4 : s3.reg 4 = some vc.sp := by
    simp only [s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 10 ≠ 4), List.getElem?_set_ne (by decide : 11 ≠ 4)]; exact hsp
  have hwf3 : vc.wf pr ls s3 := vc.wf_step_misc h rfl hr3 hs3_4 hs3_15
  obtain ⟨_, m4, _, _, h04, hwf4, hc272, hrest4⟩ :=
    vc.wr64_ctx_scratch_slot hwf3 272 _ 4195202 (by decide) (by decide) hs3_10
      (fun {m'} hwr => by
        rw [step_at_chunk 2 rfl rfl (by decide : (4195195, (.ofInt (.st 15 272 10)), 7) ∈ SporosVmBin.vmTable_2),
          exec_st hs3_15 hs3_10 hwr])
  let s4 := { s3 with mem := m4, rip := 4195202 }
  have hs4_0 : s4.reg 0 = some (primCode p) := by
    simp only [s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 10 ≠ 0), List.getElem?_set_ne (by decide : 11 ≠ 0)]; exact hr0
  have h05 : X86Vm.step vmImage s4 = .ok { s4.set 2 (some 4206363) with rip := 4195207 } := by
    rw [step_at_chunk 2 rfl rfl (by decide : (4195202, (.ofInt (.movI 2 4206363)), 5) ∈ SporosVmBin.vmTable_2),
      exec_movI_lt (by decide)]
  let s5 := { s4.set 2 (some 4206363) with rip := 4195207 }
  have hr5 : RegsOk s5 := regsOk_set hr3 (by decide) 4206363
  have hs5_2 : s5.reg 2 = some 4206363 := by
    simp only [s5, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
  have hs5_0 : s5.reg 0 = some (primCode p) := by
    simp only [s5, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 0)]; exact hs4_0
  have hpj_get : (roQwords SporosVmBin.L_prim_jtab 57)[primCode p]? = some (primEntry p) := by
    rw [ro_primJtab_ok]; exact expectedPrimJtab_prim p
  have hp_lt := primCode_lt p
  have hrd_t : rd64 s5.mem ((4206363 + 8 * primCode p + 0) % X86Env.W) = some (primEntry p) := by
    have heq : 4206363 = SporosVmBin.L_prim_jtab := rfl
    have hW : (SporosVmBin.L_prim_jtab + 8 * primCode p + 0) < X86Env.W := by
      have : SporosVmBin.L_prim_jtab = 4206363 := rfl
      have : X86Env.W = 2 ^ 64 := rfl
      omega
    rw [heq, Nat.mod_eq_of_lt hW, Nat.add_zero]
    exact roQwords_rd64 hwf4.1 (by decide : vmCodeEnd ≤ SporosVmBin.L_prim_jtab) primJtab_bounds_ok hp_lt hpj_get
  have h06 : X86Vm.step vmImage s5 = .ok { s5 with rip := primEntry p } := by
    rw [step_at_chunk 2 rfl rfl (by decide : (4195207, (.ofInt (.jtab 2 0 0)), 3) ∈ SporosVmBin.vmTable_2),
      exec_jtab hs5_2 hs5_0 hrd_t]
  let s6 := { s5 with rip := primEntry p }
  have hs6_15 : s6.reg 15 = some ctxBase := by
    simp only [s6, s5, s4, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 15)]; exact hs3_15
  have hs6_4 : s6.reg 4 = some vc.sp := by
    simp only [s6, s5, s4, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 4)]; exact hs3_4
  let cd' := writeBytes vc.ctxReg.data 272 (bytes8 ((qa + ls.pc + 1) % X86Env.W))
  let ctx' := { vc.ctxReg with data := cd' }
  have hlen' : cd'.length = vc.ctxReg.data.length := by
    dsimp [cd']; rw [writeBytes_length]
  have hwf6 : { vc with ctxReg := ctx' }.wf pr ls s6 :=
    ({ vc with ctxReg := ctx' }).wf_step_misc hwf4 rfl hr5 hs6_4 hs6_15
  have hreach : Reaches vmImage s s6 :=
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
      (Reaches.step_trans h05 (Reaches.step h06)))))
  have hc272' : leVal ((cd'.drop 272).take 8) = (qa + ls.pc + 1) % X86Env.W := by
    rw [hc272, Nat.mod_mod]
  exact ⟨cd', s6, hreach, rfl, rfl, hwf6, hlen', hc272', hrest4⟩

theorem L_prim_prologue_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} (p : Sporos.Prim)
    (hrip : s.rip = SporosVmBin.L_prim_prologue) (h : vc.wf pr ls s)
    (hr0 : s.reg 0 = some (primCode p)) :
    if ls.ds.length < p.eff.1 then
      ∃ s_exit, Reaches vmImage s s_exit ∧ X86Vm.step vmImage s_exit = .exit 1 ∧ s_exit.k = s.k
    else
      ∃ ctxData',
        let ctx' := { vc.ctxReg with data := ctxData' }
        let vc' := { vc with ctxReg := ctx' }
        ∃ s', Reaches vmImage s s' ∧
          s'.rip = primEntry p ∧ s'.k = s.k ∧ vc'.wf pr ls s' ∧
          ctxData'.length = vc.ctxReg.data.length ∧
          rd64 s'.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take p.eff.1) ls.pc) ∧
          rd64 s'.mem (ea ctxBase 280) = some p.eff.1 ∧
          rd64 s'.mem (ea ctxBase 288) = some p.eff.2 ∧
          (1 ≤ p.eff.1 → rd64 s'.mem (ea ctxBase 296) = some ls.ds[0]!.w ∧ rd64 s'.mem (ea ctxBase 304) = some ls.ds[0]!.plen) ∧
          (2 ≤ p.eff.1 → rd64 s'.mem (ea ctxBase 312) = some ls.ds[1]!.w ∧ rd64 s'.mem (ea ctxBase 320) = some ls.ds[1]!.plen) ∧
          (3 ≤ p.eff.1 → rd64 s'.mem (ea ctxBase 328) = some ls.ds[2]!.w ∧ rd64 s'.mem (ea ctxBase 336) = some ls.ds[2]!.plen) ∧
          (4 ≤ p.eff.1 → rd64 s'.mem (ea ctxBase 344) = some ls.ds[3]!.w ∧ rd64 s'.mem (ea ctxBase 352) = some ls.ds[3]!.plen) ∧
          (∀ m0, (m0 + 8 ≤ 272 ∨ 360 ≤ m0) → m0 + 8 ≤ ctxSize →
            leVal ((ctxData'.drop m0).take 8) = leVal ((vc.ctxReg.data.drop m0).take 8)) := by
  have hr := h.2.2.2.1
  have h15 := h.2.2.2.2.2.1
  have hsp := h.2.2.2.2.1
  have hp_lt := primCode_lt p
  have ⟨he_in_le, he_out_le⟩ := prim_eff_le_4 p
  have h4W : 4 < X86Env.W := by decide
  have he_in_W : p.eff.1 < X86Env.W := by omega
  have he_out_W : p.eff.2 < X86Env.W := by omega
  have h01 : X86Vm.step vmImage s = .ok { s.set 2 (some 4205689) with rip := 4194958 } := by
    rw [step_at_chunk 1 hrip rfl (by decide : (4194953, (.ofInt (.movI 2 4205689)), 5) ∈ SporosVmBin.vmTable_1),
      exec_movI_lt (by decide), hrip]
    rfl
  let s1 := { s.set 2 (some 4205689) with rip := 4194958 }
  have hr1 : RegsOk s1 := regsOk_set hr (by decide) 4205689
  have hs1_2 : s1.reg 2 = some 4205689 := by
    simp only [s1, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr.1]; decide)]
  have hs1_0 : s1.reg 0 = some (primCode p) := by
    simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 0)]; exact hr0
  have hrd_ein : rdMem .w8 s1.mem (eaIdx 4205689 (primCode p) 1 0) = some p.eff.1 := by
    have heq : 4205689 = SporosVmBin.L_prim_eff_in := rfl
    have hW : 4205689 + 1 * primCode p + 0 < X86Env.W := by have : X86Env.W = 2 ^ 64 := rfl; omega
    rw [eaIdx_add_eq hW, heq]
    simpa [s1, Nat.add_zero] using
      roSlice_rd8 h.1 (by decide : vmCodeEnd ≤ SporosVmBin.L_prim_eff_in) effIn_bounds_ok hp_lt (effIn_prim_get p)
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 8 (some p.eff.1) with rip := 4194963 } := by
    rw [step_at_chunk 1 rfl rfl (by decide : (4194958, (.ldIdx .w8 8 2 0 1 0), 5) ∈ SporosVmBin.vmTable_1),
      exec_ldIdx hs1_2 hs1_0 hrd_ein]
  let s2 := { s1.set 8 (some p.eff.1) with rip := 4194963 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) p.eff.1
  have hs2_0 : s2.reg 0 = some (primCode p) := by
    simp only [s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 8 ≠ 0)]; exact hs1_0
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 2 (some 4205746) with rip := 4194968 } := by
    rw [step_at_chunk 1 rfl rfl (by decide : (4194963, (.ofInt (.movI 2 4205746)), 5) ∈ SporosVmBin.vmTable_1),
      exec_movI_lt (by decide)]
  let s3 := { s2.set 2 (some 4205746) with rip := 4194968 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) 4205746
  have hs3_2 : s3.reg 2 = some 4205746 := by
    simp only [s3, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
  have hs3_0 : s3.reg 0 = some (primCode p) := by
    simp only [s3, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 0)]; exact hs2_0
  have hrd_eout : rdMem .w8 s3.mem (eaIdx 4205746 (primCode p) 1 0) = some p.eff.2 := by
    have heq : 4205746 = SporosVmBin.L_prim_eff_out := rfl
    have hW : 4205746 + 1 * primCode p + 0 < X86Env.W := by have : X86Env.W = 2 ^ 64 := rfl; omega
    rw [eaIdx_add_eq hW, heq]
    simpa [s3, s2, s1, Nat.add_zero] using
      roSlice_rd8 h.1 (by decide : vmCodeEnd ≤ SporosVmBin.L_prim_eff_out) effOut_bounds_ok hp_lt (effOut_prim_get p)
  have h04 : X86Vm.step vmImage s3 = .ok { s3.set 9 (some p.eff.2) with rip := 4194973 } := by
    rw [step_at_chunk 1 rfl rfl (by decide : (4194968, (.ldIdx .w8 9 2 0 1 0), 5) ∈ SporosVmBin.vmTable_1),
      exec_ldIdx hs3_2 hs3_0 hrd_eout]
  let s4 := { s3.set 9 (some p.eff.2) with rip := 4194973 }
  have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) p.eff.2
  have hs4_8 : s4.reg 8 = some p.eff.1 := by
    simp only [s4, s3, s2, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 9 ≠ 8), List.getElem?_set_ne (by decide : 2 ≠ 8)]
    rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  have hs4_9 : s4.reg 9 = some p.eff.2 := by
    simp only [s4, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
  have hs4_15 : s4.reg 15 = some ctxBase := by
    simp only [s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 9 ≠ 15), List.getElem?_set_ne (by decide : 2 ≠ 15),
      List.getElem?_set_ne (by decide : 8 ≠ 15)]; exact h15
  have hs4_4 : s4.reg 4 = some vc.sp := by
    simp only [s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 9 ≠ 4), List.getElem?_set_ne (by decide : 2 ≠ 4),
      List.getElem?_set_ne (by decide : 8 ≠ 4)]; exact hsp
  have hwf4 : vc.wf pr ls s4 := vc.wf_step_misc h rfl hr4 hs4_4 hs4_15
  obtain ⟨_, m5, _, _, h05, hwf5, hc280, hrest5⟩ :=
    vc.wr64_ctx_scratch_slot hwf4 280 _ 4194980 (by decide) (by decide) hs4_8
      (fun {m'} hwr => by
        rw [step_at_chunk 1 rfl rfl (by decide : (4194973, (.ofInt (.st 15 280 8)), 7) ∈ SporosVmBin.vmTable_1),
          exec_st hs4_15 hs4_8 hwr])
  let s5 := { s4 with mem := m5, rip := 4194980 }
  let vc5 := { vc with ctxReg := { vc.ctxReg with data := writeBytes vc.ctxReg.data 280 (bytes8 p.eff.1) } }
  obtain ⟨_, m6, _, _, h06, hwf6, hc288, hrest6⟩ :=
    vc5.wr64_ctx_scratch_slot hwf5 288 _ 4194987 (by decide) (by decide) (show s5.reg 9 = some p.eff.2 from hs4_9)
      (fun {m'} hwr => by
        rw [step_at_chunk 1 rfl rfl (by decide : (4194980, (.ofInt (.st 15 288 9)), 7) ∈ SporosVmBin.vmTable_1),
          exec_st (show s5.reg 15 = some ctxBase from hs4_15) (show s5.reg 9 = some p.eff.2 from hs4_9) hwr])
  let s6 := { s5 with mem := m6, rip := 4194987 }
  let vc6 := { vc5 with ctxReg := { vc5.ctxReg with data := writeBytes vc5.ctxReg.data 288 (bytes8 p.eff.2) } }
  have hs6_15 : s6.reg 15 = some ctxBase := hs4_15
  have hdsLen6 := vc6.rd_ctx_dsLen hwf6
  have h07 : X86Vm.step vmImage s6 = .ok { s6.set 1 (some ls.ds.length) with rip := 4194991 } := by
    rw [step_at_chunk 1 rfl rfl (by decide : (4194987, (.ofInt (.ld 1 15 8)), 4) ∈ SporosVmBin.vmTable_1),
      exec_ld hs6_15 hdsLen6]
  let s7 := { s6.set 1 (some ls.ds.length) with rip := 4194991 }
  have hr7 : RegsOk s7 := regsOk_set hr4 (by decide) ls.ds.length
  have hs7_1 : s7.reg 1 = some ls.ds.length := by
    simp only [s7, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr4.1]; decide)]
  have hs7_8 : s7.reg 8 = some p.eff.1 := by
    simp only [s7, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 8)]; exact hs4_8
  have h08 : X86Vm.step vmImage s7 =
      .ok { s7 with fl := (X86Int.aluF .sub ls.ds.length p.eff.1).2, rip := 4194994 } := by
    rw [step_at_chunk 1 rfl rfl (by decide : (4194991, (.ofInt (.cmp 1 8)), 3) ∈ SporosVmBin.vmTable_1),
      exec_cmp hs7_1 hs7_8]
  let s8 := { s7 with fl := (X86Int.aluF .sub ls.ds.length p.eff.1).2, rip := 4194994 }
  have hrch08 : Reaches vmImage s s8 :=
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
      (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step h08)))))))
  by_cases huf : ls.ds.length < p.eff.1
  · rw [if_pos huf]
    have h09 : X86Vm.step vmImage s8 = .ok { s8 with rip := 4201048 } := by
      rw [step_at_chunk 1 rfl rfl (by decide : (4194994, (.ofInt (.jcc 2 4201048)), 6) ∈ SporosVmBin.vmTable_1),
        exec_jcc (cond_aluF_sub_b ls.ds.length p.eff.1)]
      dsimp only [s8]
      rw [if_pos (decide_eq_true huf)]
    let s9 := { s8 with rip := 4201048 }
    have hs9_15 : s9.reg 15 = some ctxBase := by
      simp only [s9, s8, s7, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 15)]; exact hs4_15
    have hs9_4 : s9.reg 4 = some vc.sp := by
      simp only [s9, s8, s7, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 4)]; exact hs4_4
    have hwf9 : vc6.wf pr ls s9 := vc6.wf_step_misc hwf6 rfl hr7 hs9_4 hs9_15
    obtain ⟨se, hre, hexe, hke⟩ := L_fail_from_vc_spec (e := .underflow) rfl hwf9
    exact ⟨se, Reaches.trans hrch08 (Reaches.step_trans h09 hre), hexe, hke⟩
  · rw [if_neg huf]
    have h09 : X86Vm.step vmImage s8 = .ok { s8 with rip := 4195000 } := by
      rw [step_at_chunk 1 rfl rfl (by decide : (4194994, (.ofInt (.jcc 2 4201048)), 6) ∈ SporosVmBin.vmTable_1),
        exec_jcc (cond_aluF_sub_b ls.ds.length p.eff.1)]
      dsimp only [s8]
      rw [if_neg (by rw [decide_eq_false huf]; decide)]
    let s9 := { s8 with rip := 4195000 }
    obtain ⟨r10v, hr10v⟩ := Option.isSome_iff_exists.mp (hr7.2 10 (by decide) (by decide))
    have hs9_10 : s9.reg 10 = some r10v := hr10v
    have h10 : X86Vm.step vmImage s9 =
        .ok { s9.set 10 (some 0) with fl := (X86Int.aluF .xor r10v r10v).2, rip := 4195003 } := by
      rw [step_at_chunk 1 rfl rfl (by decide : (4195000, (.ofInt (.alu .xor 10 10)), 3) ∈ SporosVmBin.vmTable_1),
        exec_alu hs9_10 hs9_10]
      dsimp only
      have hxor : r10v.xor r10v = 0 := Nat.xor_self r10v
      rw [aluF_xor_val, hxor]
    let s10 := { s9.set 10 (some 0) with fl := (X86Int.aluF .xor r10v r10v).2, rip := 4195003 }
    have hr10 : RegsOk s10 := regsOk_set hr7 (by decide) 0
    have hs10_15 : s10.reg 15 = some ctxBase := by
      simp only [s10, s9, s8, s7, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 10 ≠ 15), List.getElem?_set_ne (by decide : 1 ≠ 15)]; exact hs4_15
    have hs10_4 : s10.reg 4 = some vc.sp := by
      simp only [s10, s9, s8, s7, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 10 ≠ 4), List.getElem?_set_ne (by decide : 1 ≠ 4)]; exact hs4_4
    have hwf10 : vc6.wf pr ls s10 := vc6.wf_step_misc hwf6 rfl hr10 hs10_4 hs10_15
    have hdsBase10 := vc6.rd_ctx_dsBase hwf10
    have h11 : X86Vm.step vmImage s10 = .ok { s10.set 6 (some vc.dsReg.base) with rip := 4195006 } := by
      rw [step_at_chunk 1 rfl rfl (by decide : (4195003, (.ofInt (.ld 6 15 0)), 3) ∈ SporosVmBin.vmTable_1),
        exec_ld hs10_15 hdsBase10]
    let s11 := { s10.set 6 (some vc.dsReg.base) with rip := 4195006 }
    have hr11 : RegsOk s11 := regsOk_set hr10 (by decide) vc.dsReg.base
    have hs11_8 : s11.reg 8 = some p.eff.1 := by
      simp only [s11, s10, s9, s8, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 6 ≠ 8), List.getElem?_set_ne (by decide : 10 ≠ 8)]; exact hs7_8
    have h12 : X86Vm.step vmImage s11 =
        .ok { s11 with fl := (X86Int.aluF .and p.eff.1 p.eff.1).2, rip := 4195009 } := by
      rw [step_at_chunk 1 rfl rfl (by decide : (4195006, (.ofInt (.test 8 8)), 3) ∈ SporosVmBin.vmTable_1),
        exec_test hs11_8 hs11_8]
    let s12 := { s11 with fl := (X86Int.aluF .and p.eff.1 p.eff.1).2, rip := 4195009 }
    have hrch0_12 : Reaches vmImage s s12 :=
      Reaches.trans hrch08 (Reaches.step_trans h09 (Reaches.step_trans h10 (Reaches.step_trans h11 (Reaches.step h12))))
    have hs12_0 : s12.reg 0 = some (primCode p) := by
      simp only [s12, s11, s10, s9, s8, s7, s6, s5, s4, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 6 ≠ 0), List.getElem?_set_ne (by decide : 10 ≠ 0),
        List.getElem?_set_ne (by decide : 1 ≠ 0), List.getElem?_set_ne (by decide : 9 ≠ 0)]; exact hs3_0
    have hs12_1 : s12.reg 1 = some ls.ds.length := by
      simp only [s12, s11, s10, s9, s8, X86Int.St.reg, X86Int.St.set,
        List.getElem?_set_ne (by decide : 6 ≠ 1), List.getElem?_set_ne (by decide : 10 ≠ 1)]; exact hs7_1
    have hs12_6 : s12.reg 6 = some vc.dsReg.base := by
      simp only [s12, s11, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr10.1]; decide)]
    have hs12_8 : s12.reg 8 = some p.eff.1 := hs11_8
    have hs12_10 : s12.reg 10 = some 0 := by
      simp only [s12, s11, s10, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 6 ≠ 10)]
      rw [getElem?_set_getD_self (by rw [hr7.1]; decide)]
    have hs12_15 : s12.reg 15 = some ctxBase := by
      simp only [s12, s11, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 6 ≠ 15)]; exact hs10_15
    have hs12_4 : s12.reg 4 = some vc.sp := by
      simp only [s12, s11, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 6 ≠ 4)]; exact hs10_4
    have hwf12 : vc6.wf pr ls s12 := vc6.wf_step_misc hwf6 rfl hr11 hs12_4 hs12_15
    have hvc6_len : vc6.ctxReg.data.length = vc.ctxReg.data.length := by
      have h5cl : vc5.ctxReg.data.length = vc.ctxReg.data.length := by
        dsimp [vc5]; rw [writeBytes_length]
      dsimp [vc6]; rw [writeBytes_length]; exact h5cl
    have hvc6_280 : leVal ((vc6.ctxReg.data.drop 280).take 8) = p.eff.1 := by
      rw [hrest6 280 (Or.inl (by decide)) (by decide), hc280, Nat.mod_eq_of_lt he_in_W]
    have hvc6_288 : leVal ((vc6.ctxReg.data.drop 288).take 8) = p.eff.2 := by
      rw [hc288, Nat.mod_eq_of_lt he_out_W]
    have hvc6_rest : ∀ m0, (m0 + 8 ≤ 280 ∨ 296 ≤ m0) → m0 + 8 ≤ ctxSize →
        leVal ((vc6.ctxReg.data.drop m0).take 8) = leVal ((vc.ctxReg.data.drop m0).take 8) := by
      intro m0 hm0 hm0lim
      rcases hm0 with hm0 | hm0
      · rw [hrest6 m0 (Or.inl (by omega)) hm0lim, hrest5 m0 (Or.inl hm0) hm0lim]
      · rw [hrest6 m0 (Or.inr hm0) hm0lim, hrest5 m0 (Or.inr (by omega)) hm0lim]
    have finish_pp_q : ∀ (cdA : List Nat) (sA : X86Vm.St) (qa : Nat),
        Reaches vmImage s sA → sA.rip = 4195181 → sA.k = s.k →
        let vcA := { vc with ctxReg := { vc.ctxReg with data := cdA } }
        vcA.wf pr ls sA → cdA.length = vc.ctxReg.data.length →
        sA.reg 0 = some (primCode p) → sA.reg 10 = some (qa % X86Env.W) →
        (qa + ls.pc + 1) % X86Env.W = SporosProvLen.lq (ls.ds.take p.eff.1) ls.pc →
        leVal ((cdA.drop 280).take 8) = p.eff.1 →
        leVal ((cdA.drop 288).take 8) = p.eff.2 →
        (1 ≤ p.eff.1 → leVal ((cdA.drop 296).take 8) = ls.ds[0]!.w ∧ leVal ((cdA.drop 304).take 8) = ls.ds[0]!.plen) →
        (2 ≤ p.eff.1 → leVal ((cdA.drop 312).take 8) = ls.ds[1]!.w ∧ leVal ((cdA.drop 320).take 8) = ls.ds[1]!.plen) →
        (3 ≤ p.eff.1 → leVal ((cdA.drop 328).take 8) = ls.ds[2]!.w ∧ leVal ((cdA.drop 336).take 8) = ls.ds[2]!.plen) →
        (4 ≤ p.eff.1 → leVal ((cdA.drop 344).take 8) = ls.ds[3]!.w ∧ leVal ((cdA.drop 352).take 8) = ls.ds[3]!.plen) →
        (∀ m0, (m0 + 8 ≤ 272 ∨ 360 ≤ m0) → m0 + 8 ≤ ctxSize →
          leVal ((cdA.drop m0).take 8) = leVal ((vc.ctxReg.data.drop m0).take 8)) →
        ∃ ctxData',
          let ctx' := { vc.ctxReg with data := ctxData' }
          let vc' := { vc with ctxReg := ctx' }
          ∃ s', Reaches vmImage s s' ∧
            s'.rip = primEntry p ∧ s'.k = s.k ∧ vc'.wf pr ls s' ∧
            ctxData'.length = vc.ctxReg.data.length ∧
            rd64 s'.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take p.eff.1) ls.pc) ∧
            rd64 s'.mem (ea ctxBase 280) = some p.eff.1 ∧
            rd64 s'.mem (ea ctxBase 288) = some p.eff.2 ∧
            (1 ≤ p.eff.1 → rd64 s'.mem (ea ctxBase 296) = some ls.ds[0]!.w ∧ rd64 s'.mem (ea ctxBase 304) = some ls.ds[0]!.plen) ∧
            (2 ≤ p.eff.1 → rd64 s'.mem (ea ctxBase 312) = some ls.ds[1]!.w ∧ rd64 s'.mem (ea ctxBase 320) = some ls.ds[1]!.plen) ∧
            (3 ≤ p.eff.1 → rd64 s'.mem (ea ctxBase 328) = some ls.ds[2]!.w ∧ rd64 s'.mem (ea ctxBase 336) = some ls.ds[2]!.plen) ∧
            (4 ≤ p.eff.1 → rd64 s'.mem (ea ctxBase 344) = some ls.ds[3]!.w ∧ rd64 s'.mem (ea ctxBase 352) = some ls.ds[3]!.plen) ∧
            (∀ m0, (m0 + 8 ≤ 272 ∨ 360 ≤ m0) → m0 + 8 ≤ ctxSize →
              leVal ((ctxData'.drop m0).take 8) = leVal ((vc.ctxReg.data.drop m0).take 8)) := by
      intro cdA sA qa hrA hripA hkA vcA hwfA hlenA hA0 hA10 hqeq h280 h288 h0a h1a h2a h3a hAr
      obtain ⟨cd, sB, hrchB, hripB, hkB, hwfB, hcd_len0, hc272, hrestB⟩ :=
        pp_q_spec p (qa := qa) hripA hwfA hA0 hA10
      let vcB := { vc with ctxReg := { vc.ctxReg with data := cd } }
      have hwfB' : vcB.wf pr ls sB := hwfB
      have hcd_len : cd.length = vc.ctxReg.data.length := hcd_len0.trans hlenA
      have hpair : ∀ {off v : Nat}, 280 ≤ off → off + 8 ≤ ctxSize →
          leVal ((cdA.drop off).take 8) = v → rd64 sB.mem (ea ctxBase off) = some v := by
        intro off v hoff hofflim heq
        exact vcB.rd64_ctx_of_eq hwfB' hofflim (by rw [hrestB off (Or.inr hoff) hofflim, heq])
      refine ⟨cd, sB, Reaches.trans hrA hrchB, hripB, hkB.trans hkA, hwfB', hcd_len,
        vcB.rd64_ctx_of_eq hwfB' (by decide) (by rw [hc272, hqeq]),
        hpair (by decide) (by decide) h280,
        hpair (by decide) (by decide) h288,
        fun h1 => ⟨hpair (by decide) (by decide) (h0a h1).1, hpair (by decide) (by decide) (h0a h1).2⟩,
        fun h2 => ⟨hpair (by decide) (by decide) (h1a h2).1, hpair (by decide) (by decide) (h1a h2).2⟩,
        fun h3 => ⟨hpair (by decide) (by decide) (h2a h3).1, hpair (by decide) (by decide) (h2a h3).2⟩,
        fun h4 => ⟨hpair (by decide) (by decide) (h3a h4).1, hpair (by decide) (by decide) (h3a h4).2⟩, ?_⟩
      intro m0 hm0 hm0lim
      rcases hm0 with hm0 | hm0
      · rw [hrestB m0 (Or.inl hm0) hm0lim, hAr m0 (Or.inl hm0) hm0lim]
      · rw [hrestB m0 (Or.inr (by omega)) hm0lim, hAr m0 (Or.inr hm0) hm0lim]
    have prologue_cmp_jcc : ∀ {cdX : List Nat} {sX : X86Vm.St} (aCmp k : Nat),
        k + 1 < X86Env.W → sX.rip = aCmp →
        let vcX := { vc with ctxReg := { vc.ctxReg with data := cdX } }
        vcX.wf pr ls sX → sX.reg 8 = some p.eff.1 →
        (aCmp, (.ofInt (.cmpI 8 (k + 1))), 4) ∈ SporosVmBin.vmTable_2 →
        (aCmp + 4, (.ofInt (.jcc 4 4195181)), 2) ∈ SporosVmBin.vmTable_2 →
        (p.eff.1 = k + 1 → ∃ s', Reaches vmImage sX s' ∧ s'.rip = 4195181 ∧ s'.k = sX.k ∧ vcX.wf pr ls s' ∧ (∀ r, s'.reg r = sX.reg r)) ∧
        (p.eff.1 ≠ k + 1 → ∃ s', Reaches vmImage sX s' ∧ s'.rip = aCmp + 6 ∧ s'.k = sX.k ∧ vcX.wf pr ls s' ∧ (∀ r, s'.reg r = sX.reg r)) := by
      intro cdX sX aCmp k hkW hripX vcX hwfX hr8X hmCmp hmJcc
      have hrX := hwfX.2.2.2.1
      have h15X := hwfX.2.2.2.2.2.1
      have hspX := hwfX.2.2.2.2.1
      have h0c : X86Vm.step vmImage sX =
          .ok { sX with fl := (X86Int.aluF .sub p.eff.1 (k + 1)).2, rip := aCmp + 4 } := by
        rw [step_at_chunk 2 hripX rfl hmCmp, exec_cmpI hr8X, Nat.mod_eq_of_lt hkW, hripX]
      let sC := { sX with fl := (X86Int.aluF .sub p.eff.1 (k + 1)).2, rip := aCmp + 4 }
      refine ⟨fun heq => ?_, fun hne => ?_⟩
      · have h0j : X86Vm.step vmImage sC = .ok { sC with rip := 4195181 } := by
          rw [step_at_chunk 2 rfl rfl hmJcc, exec_jcc (cond_aluF_sub_e p.eff.1 (k + 1) he_in_W hkW)]
          dsimp only [sC]; rw [if_pos (decide_eq_true heq)]
        exact ⟨{ sC with rip := 4195181 }, Reaches.step_trans h0c (Reaches.step h0j),
          rfl, rfl, vcX.wf_step_misc hwfX rfl hrX hspX h15X, fun _ => rfl⟩
      · have h0j : X86Vm.step vmImage sC = .ok { sC with rip := aCmp + 6 } := by
          rw [step_at_chunk 2 rfl rfl hmJcc, exec_jcc (cond_aluF_sub_e p.eff.1 (k + 1) he_in_W hkW)]
          dsimp only [sC]; rw [if_neg (by rw [decide_eq_false hne]; decide)]
        exact ⟨{ sC with rip := aCmp + 6 }, Reaches.step_trans h0c (Reaches.step h0j),
          rfl, rfl, vcX.wf_step_misc hwfX rfl hrX hspX h15X, fun _ => rfl⟩
    by_cases he0 : p.eff.1 = 0
    · have h13 : X86Vm.step vmImage s12 = .ok { s12 with rip := 4195181 } := by
        rw [step_at_chunk 2 rfl rfl (by decide : (4195009, (.ofInt (.jcc 4 4195181)), 6) ∈ SporosVmBin.vmTable_2),
          exec_jcc (cond_aluF_and_self_e p.eff.1)]
        dsimp only [s12]; rw [if_pos (decide_eq_true he0)]
      let s13 := { s12 with rip := 4195181 }
      have hwf13 : vc6.wf pr ls s13 := vc6.wf_step_misc hwf12 rfl hr11 hs12_4 hs12_15
      exact finish_pp_q vc6.ctxReg.data s13 0 (Reaches.trans hrch0_12 (Reaches.step h13)) rfl rfl hwf13 hvc6_len
        hs12_0 hs12_10 (by simp [he0, SporosProvLen.lq, M_eq_W]) hvc6_280 hvc6_288
        (fun h => by omega) (fun h => by omega) (fun h => by omega) (fun h => by omega)
        (fun m0 hm0 lim => by
          rcases hm0 with hm0 | hm0
          · exact hvc6_rest m0 (Or.inl (by omega)) lim
          · exact hvc6_rest m0 (Or.inr (by omega)) lim)
    · have h13 : X86Vm.step vmImage s12 = .ok { s12 with rip := 4195015 } := by
        rw [step_at_chunk 2 rfl rfl (by decide : (4195009, (.ofInt (.jcc 4 4195181)), 6) ∈ SporosVmBin.vmTable_2),
          exec_jcc (cond_aluF_and_self_e p.eff.1)]
        dsimp only [s12]; rw [if_neg (by rw [decide_eq_false he0]; decide)]
      let s13 := { s12 with rip := 4195015 }
      have hwf13 : vc6.wf pr ls s13 := vc6.wf_step_misc hwf12 rfl hr11 hs12_4 hs12_15
      have hk0 : 0 < ls.ds.length := by omega
      obtain ⟨cd1, s14, hrch14, hrip14, hk14, hwf14, hcd1_len0, hs14_1, hs14_6, hs14_10, hs14_oth, hc296, hc304, hrest14⟩ :=
        prologue_load_slot_spec (a := 4195015) (offW := 296) (offP := 304) (k := 0) (qa := 0)
          rfl hwf13 hk0 (by decide) (by decide) (by decide) (by decide) (by decide) hs12_1 hs12_6 hs12_10
          (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      have hcd1_len : cd1.length = vc.ctxReg.data.length := hcd1_len0.trans hvc6_len
      have hcd1_280 : leVal ((cd1.drop 280).take 8) = p.eff.1 := by
        rw [hrest14 280 (Or.inl (by decide)) (by decide), hvc6_280]
      have hcd1_288 : leVal ((cd1.drop 288).take 8) = p.eff.2 := by
        rw [hrest14 288 (Or.inl (by decide)) (by decide), hvc6_288]
      have hcd1_rest : ∀ m0, (m0 + 8 ≤ 272 ∨ 312 ≤ m0) → m0 + 8 ≤ ctxSize →
          leVal ((cd1.drop m0).take 8) = leVal ((vc.ctxReg.data.drop m0).take 8) := by
        intro m0 hm0 lim
        rcases hm0 with hm0 | hm0
        · rw [hrest14 m0 (Or.inl (by omega)) lim, hvc6_rest m0 (Or.inl (by omega)) lim]
        · rw [hrest14 m0 (Or.inr (Or.inr hm0)) lim, hvc6_rest m0 (Or.inr (by omega)) lim]
      have hs14_0 : s14.reg 0 = some (primCode p) := by rw [hs14_oth 0 (by decide) (by decide) (by decide)]; exact hs12_0
      have hs14_8 : s14.reg 8 = some p.eff.1 := by rw [hs14_oth 8 (by decide) (by decide) (by decide)]; exact hs12_8
      have hcmp1 := prologue_cmp_jcc (cdX := cd1) 4195052 0 (by decide) hrip14 hwf14 hs14_8 (by decide) (by decide)
      by_cases he1 : p.eff.1 = 1
      · obtain ⟨s15, hrch15, hrip15, hk15, hwf15, hs15_eq⟩ := hcmp1.1 he1
        have hfold1 : (ls.ds.take 1).foldr (fun v acc => v.plen + acc) 0 = ls.ds[0]!.plen := by
          simpa using lq_take_step_mod ls.ds 0 hk0
        have hq1 : (0 + ls.ds[0]!.plen + ls.pc + 1) % X86Env.W = SporosProvLen.lq (ls.ds.take p.eff.1) ls.pc := by
          dsimp only [SporosProvLen.lq]
          rw [he1, hfold1]
          simp [M_eq_W]
        exact finish_pp_q cd1 s15 (0 + ls.ds[0]!.plen)
          (Reaches.trans hrch0_12 (Reaches.step_trans h13 (Reaches.trans hrch14 hrch15)))
          hrip15 (hk15.trans hk14) hwf15 hcd1_len
          (by rw [hs15_eq]; exact hs14_0) (by rw [hs15_eq]; exact hs14_10) hq1
          hcd1_280 hcd1_288 (fun _ => ⟨hc296, hc304⟩) (fun h => by omega) (fun h => by omega) (fun h => by omega)
          (fun m0 hm0 lim => by
            rcases hm0 with hm0 | hm0
            · exact hcd1_rest m0 (Or.inl hm0) lim
            · exact hcd1_rest m0 (Or.inr (by omega)) lim)
      · obtain ⟨s15, hrch15, hrip15, hk15, hwf15, hs15_eq⟩ := hcmp1.2 he1
        have hk1 : 1 < ls.ds.length := by omega
        obtain ⟨cd2, s16, hrch16, hrip16, hk16, hwf16, hcd2_len0, hs16_1, hs16_6, hs16_10, hs16_oth, hc312, hc320, hrest16⟩ :=
          prologue_load_slot_spec (a := 4195058) (offW := 312) (offP := 320) (k := 1) (qa := 0 + ls.ds[0]!.plen)
            hrip15 hwf15 hk1 (by decide) (by decide) (by decide) (by decide) (by decide)
            (by rw [hs15_eq]; exact hs14_1) (by rw [hs15_eq]; exact hs14_6) (by rw [hs15_eq]; exact hs14_10)
            (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
        have hcd2_len : cd2.length = vc.ctxReg.data.length := hcd2_len0.trans hcd1_len
        have hcd2_280 : leVal ((cd2.drop 280).take 8) = p.eff.1 := by rw [hrest16 280 (Or.inl (by decide)) (by decide), hcd1_280]
        have hcd2_288 : leVal ((cd2.drop 288).take 8) = p.eff.2 := by rw [hrest16 288 (Or.inl (by decide)) (by decide), hcd1_288]
        have hcd2_296 : leVal ((cd2.drop 296).take 8) = ls.ds[0]!.w := by rw [hrest16 296 (Or.inl (by decide)) (by decide), hc296]
        have hcd2_304 : leVal ((cd2.drop 304).take 8) = ls.ds[0]!.plen := by rw [hrest16 304 (Or.inl (by decide)) (by decide), hc304]
        have hcd2_rest : ∀ m0, (m0 + 8 ≤ 272 ∨ 328 ≤ m0) → m0 + 8 ≤ ctxSize →
            leVal ((cd2.drop m0).take 8) = leVal ((vc.ctxReg.data.drop m0).take 8) := by
          intro m0 hm0 lim
          rcases hm0 with hm0 | hm0
          · rw [hrest16 m0 (Or.inl (by omega)) lim, hcd1_rest m0 (Or.inl hm0) lim]
          · rw [hrest16 m0 (Or.inr (Or.inr hm0)) lim, hcd1_rest m0 (Or.inr (by omega)) lim]
        have hs16_0 : s16.reg 0 = some (primCode p) := by
          rw [hs16_oth 0 (by decide) (by decide) (by decide), hs15_eq]; exact hs14_0
        have hs16_8 : s16.reg 8 = some p.eff.1 := by
          rw [hs16_oth 8 (by decide) (by decide) (by decide), hs15_eq]; exact hs14_8
        have hcmp2 := prologue_cmp_jcc (cdX := cd2) 4195095 1 (by decide) hrip16 hwf16 hs16_8 (by decide) (by decide)
        by_cases he2 : p.eff.1 = 2
        · obtain ⟨s17, hrch17, hrip17, hk17, hwf17, hs17_eq⟩ := hcmp2.1 he2
          have hfold2 : (ls.ds.take 2).foldr (fun v acc => v.plen + acc) 0 = ls.ds[0]!.plen + ls.ds[1]!.plen := by
            rw [lq_take_step_mod ls.ds 1 hk1, lq_take_step_mod ls.ds 0 hk0]; simp
          have hq2 : ((0 + ls.ds[0]!.plen + ls.ds[1]!.plen) + ls.pc + 1) % X86Env.W =
              SporosProvLen.lq (ls.ds.take p.eff.1) ls.pc := by
            dsimp only [SporosProvLen.lq]
            rw [he2, hfold2]
            simp [M_eq_W, Nat.add_assoc]
          exact finish_pp_q cd2 s17 (0 + ls.ds[0]!.plen + ls.ds[1]!.plen)
            (Reaches.trans hrch0_12 (Reaches.step_trans h13 (Reaches.trans hrch14 (Reaches.trans hrch15 (Reaches.trans hrch16 hrch17)))))
            hrip17 (hk17.trans (hk16.trans (hk15.trans hk14))) hwf17 hcd2_len
            (by rw [hs17_eq]; exact hs16_0) (by rw [hs17_eq]; exact hs16_10) hq2
            hcd2_280 hcd2_288 (fun _ => ⟨hcd2_296, hcd2_304⟩) (fun _ => ⟨hc312, hc320⟩)
            (fun h => by omega) (fun h => by omega)
            (fun m0 hm0 lim => by
              rcases hm0 with hm0 | hm0
              · exact hcd2_rest m0 (Or.inl hm0) lim
              · exact hcd2_rest m0 (Or.inr (by omega)) lim)
        · obtain ⟨s17, hrch17, hrip17, hk17, hwf17, hs17_eq⟩ := hcmp2.2 he2
          have hk2 : 2 < ls.ds.length := Nat.lt_of_lt_of_le
            (match hev : p.eff.1 with
            | 0 => absurd hev he0
            | 1 => absurd hev he1
            | 2 => absurd hev he2
            | n + 3 => hev ▸ Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ n)))
            (Nat.le_of_not_lt huf)
          obtain ⟨cd3, s18, hrch18, hrip18, hk18, hwf18, hcd3_len0, hs18_1, hs18_6, hs18_10, hs18_oth, hc328, hc336, hrest18⟩ :=
            prologue_load_slot_spec (a := 4195101) (offW := 328) (offP := 336) (k := 2) (qa := 0 + ls.ds[0]!.plen + ls.ds[1]!.plen)
              hrip17 hwf17 hk2 (by decide) (by decide) (by decide) (by decide) (by decide)
              (by rw [hs17_eq]; exact hs16_1) (by rw [hs17_eq]; exact hs16_6) (by rw [hs17_eq]; exact hs16_10)
              (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
          have hcd3_len : cd3.length = vc.ctxReg.data.length := hcd3_len0.trans hcd2_len
          have hcd3_280 : leVal ((cd3.drop 280).take 8) = p.eff.1 := by rw [hrest18 280 (Or.inl (by decide)) (by decide), hcd2_280]
          have hcd3_288 : leVal ((cd3.drop 288).take 8) = p.eff.2 := by rw [hrest18 288 (Or.inl (by decide)) (by decide), hcd2_288]
          have hcd3_296 : leVal ((cd3.drop 296).take 8) = ls.ds[0]!.w := by rw [hrest18 296 (Or.inl (by decide)) (by decide), hcd2_296]
          have hcd3_304 : leVal ((cd3.drop 304).take 8) = ls.ds[0]!.plen := by rw [hrest18 304 (Or.inl (by decide)) (by decide), hcd2_304]
          have hcd3_312 : leVal ((cd3.drop 312).take 8) = ls.ds[1]!.w := by rw [hrest18 312 (Or.inl (by decide)) (by decide), hc312]
          have hcd3_320 : leVal ((cd3.drop 320).take 8) = ls.ds[1]!.plen := by rw [hrest18 320 (Or.inl (by decide)) (by decide), hc320]
          have hcd3_rest : ∀ m0, (m0 + 8 ≤ 272 ∨ 344 ≤ m0) → m0 + 8 ≤ ctxSize →
              leVal ((cd3.drop m0).take 8) = leVal ((vc.ctxReg.data.drop m0).take 8) := by
            intro m0 hm0 lim
            rcases hm0 with hm0 | hm0
            · rw [hrest18 m0 (Or.inl (Nat.le_trans hm0 (by decide))) lim, hcd2_rest m0 (Or.inl hm0) lim]
            · rw [hrest18 m0 (Or.inr (Or.inr hm0)) lim, hcd2_rest m0 (Or.inr (Nat.le_trans (by decide) hm0)) lim]
          have hs18_0 : s18.reg 0 = some (primCode p) := by
            rw [hs18_oth 0 (by decide) (by decide) (by decide), hs17_eq]; exact hs16_0
          have hs18_8 : s18.reg 8 = some p.eff.1 := by
            rw [hs18_oth 8 (by decide) (by decide) (by decide), hs17_eq]; exact hs16_8
          have hcmp3 := prologue_cmp_jcc (cdX := cd3) 4195138 2 (by decide) hrip18 hwf18 hs18_8 (by decide) (by decide)
          by_cases he3 : p.eff.1 = 3
          · obtain ⟨s19, hrch19, hrip19, hk19, hwf19, hs19_eq⟩ := hcmp3.1 he3
            have hfold3 : (ls.ds.take 3).foldr (fun v acc => v.plen + acc) 0 =
                ls.ds[0]!.plen + ls.ds[1]!.plen + ls.ds[2]!.plen := by
              rw [lq_take_step_mod ls.ds 2 hk2, lq_take_step_mod ls.ds 1 hk1, lq_take_step_mod ls.ds 0 hk0]; simp
            have hq3 : ((0 + ls.ds[0]!.plen + ls.ds[1]!.plen + ls.ds[2]!.plen) + ls.pc + 1) % X86Env.W =
                SporosProvLen.lq (ls.ds.take p.eff.1) ls.pc := by
              dsimp only [SporosProvLen.lq]
              rw [he3, hfold3]
              simp [M_eq_W, Nat.add_assoc]
            exact finish_pp_q cd3 s19 (0 + ls.ds[0]!.plen + ls.ds[1]!.plen + ls.ds[2]!.plen)
              (Reaches.trans hrch0_12 (Reaches.step_trans h13 (Reaches.trans hrch14 (Reaches.trans hrch15 (Reaches.trans hrch16 (Reaches.trans hrch17 (Reaches.trans hrch18 hrch19)))))))
              hrip19 (hk19.trans (hk18.trans (hk17.trans (hk16.trans (hk15.trans hk14))))) hwf19 hcd3_len
              (by rw [hs19_eq]; exact hs18_0) (by rw [hs19_eq]; exact hs18_10) hq3
              hcd3_280 hcd3_288 (fun _ => ⟨hcd3_296, hcd3_304⟩) (fun _ => ⟨hcd3_312, hcd3_320⟩)
              (fun _ => ⟨hc328, hc336⟩) (fun h => by omega)
              (fun m0 hm0 lim => by
                rcases hm0 with hm0 | hm0
                · exact hcd3_rest m0 (Or.inl hm0) lim
                · exact hcd3_rest m0 (Or.inr (Nat.le_trans (by decide) hm0)) lim)
          · obtain ⟨s19, hrch19, hrip19, hk19, hwf19, hs19_eq⟩ := hcmp3.2 he3
            have hle4 : p.eff.1 ≤ 4 := by first | assumption | exact Nat.le_of_lt_succ (by assumption)
            have he4 : p.eff.1 = 4 := match hev : p.eff.1 with
              | 0 => absurd hev he0
              | 1 => absurd hev he1
              | 2 => absurd hev he2
              | 3 => absurd hev he3
              | 4 => rfl
              | n + 5 => False.elim (Nat.lt_irrefl 4 (Nat.lt_of_lt_of_le
                  (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.zero_lt_succ n)))))
                  (hev ▸ hle4)))
            have hk3 : 3 < ls.ds.length := by omega
            obtain ⟨cd4, s20, hrch20, hrip20, hk20, hwf20, hcd4_len0, _, _, hs20_10, hs20_oth, hc344, hc352, hrest20⟩ :=
              prologue_load_slot_spec (a := 4195144) (offW := 344) (offP := 352) (k := 3) (qa := 0 + ls.ds[0]!.plen + ls.ds[1]!.plen + ls.ds[2]!.plen)
                hrip19 hwf19 hk3 (by decide) (by decide) (by decide) (by decide) (by decide)
                (by rw [hs19_eq]; exact hs18_1) (by rw [hs19_eq]; exact hs18_6) (by rw [hs19_eq]; exact hs18_10)
                (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
            have hcd4_len : cd4.length = vc.ctxReg.data.length := hcd4_len0.trans hcd3_len
            have hcd4_280 : leVal ((cd4.drop 280).take 8) = p.eff.1 := by rw [hrest20 280 (Or.inl (by decide)) (by decide), hcd3_280]
            have hcd4_288 : leVal ((cd4.drop 288).take 8) = p.eff.2 := by rw [hrest20 288 (Or.inl (by decide)) (by decide), hcd3_288]
            have hcd4_296 : leVal ((cd4.drop 296).take 8) = ls.ds[0]!.w := by rw [hrest20 296 (Or.inl (by decide)) (by decide), hcd3_296]
            have hcd4_304 : leVal ((cd4.drop 304).take 8) = ls.ds[0]!.plen := by rw [hrest20 304 (Or.inl (by decide)) (by decide), hcd3_304]
            have hcd4_312 : leVal ((cd4.drop 312).take 8) = ls.ds[1]!.w := by rw [hrest20 312 (Or.inl (by decide)) (by decide), hcd3_312]
            have hcd4_320 : leVal ((cd4.drop 320).take 8) = ls.ds[1]!.plen := by rw [hrest20 320 (Or.inl (by decide)) (by decide), hcd3_320]
            have hcd4_328 : leVal ((cd4.drop 328).take 8) = ls.ds[2]!.w := by rw [hrest20 328 (Or.inl (by decide)) (by decide), hc328]
            have hcd4_336 : leVal ((cd4.drop 336).take 8) = ls.ds[2]!.plen := by rw [hrest20 336 (Or.inl (by decide)) (by decide), hc336]
            have hcd4_rest : ∀ m0, (m0 + 8 ≤ 272 ∨ 360 ≤ m0) → m0 + 8 ≤ ctxSize →
                leVal ((cd4.drop m0).take 8) = leVal ((vc.ctxReg.data.drop m0).take 8) := by
              intro m0 hm0 lim
              rcases hm0 with hm0 | hm0
              · rw [hrest20 m0 (Or.inl (Nat.le_trans hm0 (by decide))) lim, hcd3_rest m0 (Or.inl hm0) lim]
              · rw [hrest20 m0 (Or.inr (Or.inr hm0)) lim, hcd3_rest m0 (Or.inr (Nat.le_trans (by decide) hm0)) lim]
            have hs20_0 : s20.reg 0 = some (primCode p) := by
              rw [hs20_oth 0 (by decide) (by decide) (by decide), hs19_eq]; exact hs18_0
            have hfold4 : (ls.ds.take 4).foldr (fun v acc => v.plen + acc) 0 =
                ls.ds[0]!.plen + ls.ds[1]!.plen + ls.ds[2]!.plen + ls.ds[3]!.plen := by
              rw [lq_take_step_mod ls.ds 3 hk3, lq_take_step_mod ls.ds 2 hk2,
                lq_take_step_mod ls.ds 1 hk1, lq_take_step_mod ls.ds 0 hk0]
              simp
            have hq4 : ((0 + ls.ds[0]!.plen + ls.ds[1]!.plen + ls.ds[2]!.plen + ls.ds[3]!.plen) + ls.pc + 1) % X86Env.W =
                SporosProvLen.lq (ls.ds.take p.eff.1) ls.pc := by
              dsimp only [SporosProvLen.lq]
              rw [he4, hfold4]
              simp [M_eq_W, Nat.add_assoc]
            exact finish_pp_q cd4 s20 (0 + ls.ds[0]!.plen + ls.ds[1]!.plen + ls.ds[2]!.plen + ls.ds[3]!.plen)
              (Reaches.trans hrch0_12 (Reaches.step_trans h13 (Reaches.trans hrch14 (Reaches.trans hrch15 (Reaches.trans hrch16 (Reaches.trans hrch17 (Reaches.trans hrch18 (Reaches.trans hrch19 hrch20))))))))
              hrip20 (hk20.trans (hk19.trans (hk18.trans (hk17.trans (hk16.trans (hk15.trans hk14)))))) hwf20 hcd4_len
              hs20_0 hs20_10 hq4
              hcd4_280 hcd4_288 (fun _ => ⟨hcd4_296, hcd4_304⟩) (fun _ => ⟨hcd4_312, hcd4_320⟩)
              (fun _ => ⟨hcd4_328, hcd4_336⟩) (fun _ => ⟨hc344, hc352⟩) hcd4_rest

theorem ctx_leVal_of_rd64 {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) {off v : Nat} (hoff : off + 8 ≤ ctxSize)
    (hrd : rd64 s.mem (ea ctxBase off) = some v) :
    leVal ((vc.ctxReg.data.drop off).take 8) = v := by
  have ⟨hbase, _, _hcl, _, _hlo, _hhi⟩ := vc.ctx_bounds h
  have hmem_ctx := vc.mem_ctx h
  rw [ea_add_eq (by omega)] at hrd
  rw [rd64, mem_rdN_of_mem s.mem vc.ctxReg (ctxBase + off) 8 h.1.2 hmem_ctx (by omega)
    (by simp only [Region.hi]; omega)] at hrd
  dsimp only [Option.map, Option.bind] at hrd
  injection hrd with hrd
  rw [hbase, Nat.add_sub_cancel_left] at hrd
  exact hrd

theorem L_prim_epilogue_mk1_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {w q : Nat}
    (hrip : s.rip = SporosVmBin.L_prim_epilogue_mk1) (h : vc.wf pr ls s)
    (hr0 : s.reg 0 = some w) (hrq : rd64 s.mem (ea ctxBase 272) = some q) (hqW : q < X86Env.W) :
    ∃ ctxData',
      let ctx' := { vc.ctxReg with data := ctxData' }
      let vc' := { vc with ctxReg := ctx' }
      ∃ s', Reaches vmImage s s' ∧
        s'.rip = SporosVmBin.L_prim_epilogue ∧ s'.k = s.k ∧ vc'.wf pr ls s' ∧
        ctxData'.length = vc.ctxReg.data.length ∧
        rd64 s'.mem (ea ctxBase 360) = some (w % X86Env.W) ∧
        rd64 s'.mem (ea ctxBase 368) = some q ∧
        (∀ m0, (m0 + 8 ≤ 360 ∨ 376 ≤ m0) → m0 + 8 ≤ ctxSize →
          leVal ((ctxData'.drop m0).take 8) = leVal ((vc.ctxReg.data.drop m0).take 8)) := by
  have hr := h.2.2.2.1
  have h15 := h.2.2.2.2.2.1
  have hsp := h.2.2.2.2.1
  have hq_old := ctx_leVal_of_rd64 h (by decide : 272 + 8 ≤ ctxSize) hrq
  obtain ⟨_, m1, _, _, h01, hwf1, hc360, hrest1⟩ :=
    vc.wr64_ctx_scratch_slot h 360 _ 4195217 (by decide) (by decide) hr0
      (fun {m'} hwr => by
        rw [step_at_chunk 2 hrip rfl (by decide : (4195210, (.ofInt (.st 15 360 0)), 7) ∈ SporosVmBin.vmTable_2),
          exec_st h15 hr0 hwr, hrip]; rfl)
  let s1 := { s with mem := m1, rip := 4195217 }
  let vc1 := { vc with ctxReg := { vc.ctxReg with data := writeBytes vc.ctxReg.data 360 (bytes8 w) } }
  have hrq1 : rd64 s1.mem (ea ctxBase 272) = some q :=
    vc1.rd64_ctx_of_eq hwf1 (by decide) (by rw [hrest1 272 (Or.inl (by decide)) (by decide), hq_old])
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 2 (some q) with rip := 4195224 } := by
    rw [step_at_chunk 2 rfl rfl (by decide : (4195217, (.ofInt (.ld 2 15 272)), 7) ∈ SporosVmBin.vmTable_2),
      exec_ld (show s1.reg 15 = some ctxBase from h15) hrq1]
  let s2 := { s1.set 2 (some q) with rip := 4195224 }
  have hr2 : RegsOk s2 := regsOk_set hr (by decide) q
  have hs2_2 : s2.reg 2 = some q := by
    simp only [s2, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr.1]; decide)]
  have hs2_15 : s2.reg 15 = some ctxBase := by
    simp only [s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 15)]; exact h15
  have hs2_4 : s2.reg 4 = some vc.sp := by
    simp only [s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 2 ≠ 4)]; exact hsp
  have hwf2 : vc1.wf pr ls s2 := vc1.wf_step_misc hwf1 rfl hr2 hs2_4 hs2_15
  obtain ⟨_, m3, _, _, h03, hwf3, hc368, hrest3⟩ :=
    vc1.wr64_ctx_scratch_slot hwf2 368 _ 4195231 (by decide) (by decide) hs2_2
      (fun {m'} hwr => by
        rw [step_at_chunk 2 rfl rfl (by decide : (4195224, (.ofInt (.st 15 368 2)), 7) ∈ SporosVmBin.vmTable_2),
          exec_st hs2_15 hs2_2 hwr])
  let s3 := { s2 with mem := m3, rip := 4195231 }
  let cd := writeBytes (writeBytes vc.ctxReg.data 360 (bytes8 w)) 368 (bytes8 q)
  let vc3 := { vc with ctxReg := { vc.ctxReg with data := cd } }
  have h04 : X86Vm.step vmImage s3 = .ok { s3 with rip := SporosVmBin.L_prim_epilogue } := by
    rw [step_at_chunk 2 rfl rfl (by decide : (4195231, (.ofInt (.jmp 4195233)), 2) ∈ SporosVmBin.vmTable_2),
      exec_jmp]; rfl
  let s4 := { s3 with rip := SporosVmBin.L_prim_epilogue }
  have hwf4 : vc3.wf pr ls s4 := vc3.wf_step_misc hwf3 rfl hr2 hs2_4 hs2_15
  have hcl := (vc.ctx_bounds h).2.2.1
  have hlen1 : vc1.ctxReg.data.length = vc.ctxReg.data.length := by
    dsimp [vc1]; rw [writeBytes_length]
  have hlen3 : cd.length = vc.ctxReg.data.length := by
    dsimp [cd]; rw [writeBytes_length]; exact hlen1
  have hc360' : leVal ((cd.drop 360).take 8) = w % X86Env.W := by
    rw [hrest3 360 (Or.inl (by decide)) (by decide), hc360]
  have hc368' : leVal ((cd.drop 368).take 8) = q := by
    rw [hc368, Nat.mod_eq_of_lt hqW]
  refine ⟨cd, s4, Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step h04))),
    rfl, rfl, hwf4, hlen3, vc3.rd64_ctx_of_eq hwf4 (by decide) hc360',
    vc3.rd64_ctx_of_eq hwf4 (by decide) hc368', ?_⟩
  intro m0 hm0 hm0lim
  rcases hm0 with hm0 | hm0
  · rw [hrest3 m0 (Or.inl (by omega)) hm0lim, hrest1 m0 (Or.inl hm0) hm0lim]
  · rw [hrest3 m0 (Or.inr hm0) hm0lim, hrest1 m0 (Or.inr (by omega)) hm0lim]

theorem wf_write_rs_top_pc {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s s' : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    {m' : Mem}
    (h : vc.wf pr ls s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (hwr : wr64 s.mem (ea vc.rsReg.base (rest.length * 32 + 8)) (f.pc + 1) = some m')
    (hrs'_mem : { vc.rsReg with data := writeBytes vc.rsReg.data (rest.length * 32 + 8) (bytes8 (f.pc + 1)) } ∈ m')
    (hoth : ∀ u ∈ s.mem, u.base ≠ vc.rsReg.base → u ∈ m')
    (hs'_mem : s'.mem = m') (hs'_hr : RegsOk s') (hs'_sp : s'.reg 4 = some vc.sp) (hs'_15 : s'.reg 15 = some ctxBase) :
    let rs' := { vc.rsReg with data := writeBytes vc.rsReg.data (rest.length * 32 + 8) (bytes8 (f.pc + 1)) }
    { vc with rsReg := rs' }.wf pr { ls with rs := { f with pc := f.pc + 1 } :: rest } s' := by
  obtain ⟨hcm, hblt, hpos, _, _, _, hnodup, hmem,
    hstk1, hstk2, hstk3, hstk4, hctx1, hctx2, hctx3, hctx4, hf1, hf2, hf3,
    hds1, hds2, hds3, hds4, hds5, haux1, haux2, haux3, haux4, haux5,
    hrs1, hrs2, hrs3, hrs4, hrs5, hloc1, hloc2, hloc3, hloc4, hloc5,
    hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3,
    htb1, htb2, htb3, htb4, htb5, hkv1, hkv2, hkv3, hkv4, hkv5, hkv6,
    hinp1, hout1, hout2, hout3, hout4, hout5,
    hsl0, hsl1, hsl2, hsl3, hsl4, hsl5, hsl6, hsl7,
    hsl8, hsl9, hsl10, hsl11, hsl12, hsl13, hsl14, hsl15,
    hsl16, hsl17, hsl18, hsl19, hsl20, hsl21, hsl22, hsl23,
    hsl24, hsl25, hsl26, hsl27, hsl28, hsl29, hsl30, hsl31,
    hpds, hdsle, hpaux, hauxle, hrsrel, hrsle, hlocle,
    hmemrel, htabsrel, htabsle, hinprel, houtrel, houtle, hwlt, hfit⟩ := h
  rw [hrs] at hrsrel hrsle hlocle hsl7 hsl10
  let rs' := { vc.rsReg with data := writeBytes vc.rsReg.data (rest.length * 32 + 8) (bytes8 (f.pc + 1)) }
  have hcm' : CodeMem m' := hcm.wr64_preserve hwr
  have hblt' : BytesLt m' := bytesLt_wr64 hblt hwr
  have hpos' : ∀ u ∈ m', 0 < u.data.length := len_pos_wrN hpos hwr
  have hnodup' : (List.map Region.base ({ vc with rsReg := rs' }).regions).Nodup := hnodup
  have hmem' : ∀ r ∈ ({ vc with rsReg := rs' }).regions, r ∈ m' := by
    intro r hr_mem
    have hnodup0 : (List.map Region.base vc.regions).Nodup := hnodup
    simp only [VmCtx.regions, List.map_cons, List.map_nil] at hnodup0
    simp only [VmCtx.regions, List.mem_cons, List.not_mem_nil, or_false] at hr_mem
    rcases hr_mem with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals first
      | exact hrs'_mem
      | exact hoth _ (hmem _ (by simp [VmCtx.regions]))
          (by intro heq; rw [heq] at hnodup0; simp [List.nodup_cons] at hnodup0)
  have hrs_len_bound : rest.length * 32 + 16 ≤ vc.rsReg.data.length := by
    have : (f :: rest).length ≤ vc.rsCap := hrsle
    simp only [List.length_cons] at this; omega
  have hrs3' : rs'.data.length = vc.rsCap * 32 := by
    dsimp [rs']; rw [writeBytes_length]; exact hrs3
  have hrs5' : rs'.hi ≤ 2 ^ 47 := by
    simpa [Region.hi, rs', hrs3', ← hrs3] using hrs5
  have hrsrel' : RsLocRel rs'.data vc.locReg.data (rsLocBase ({ f with pc := f.pc + 1 } :: rest))
      ({ f with pc := f.pc + 1 } :: rest) := by
    have hloc_eq : rsLocBase ({ f with pc := f.pc + 1 } :: rest) = rsLocBase (f :: rest) := by
      simp [rsLocBase]
    rw [hloc_eq]
    exact rsLocRel_adv_top hrsrel (Nat.le_trans (Nat.mul_le_mul_right 32 (by simpa only [List.length_cons] using hrsle)) (Nat.le_of_eq hrs3.symm)) hpcW
  have hlocle' : rsLocBase ({ f with pc := f.pc + 1 } :: rest) ≤ vc.locCap := by
    simpa [rsLocBase] using hlocle
  have hwlt' : ({ ls with rs := { f with pc := f.pc + 1 } :: rest } : SporosProvLen.LSt).WordsLt := by
    have hrs_w := hwlt.2.2.1
    rw [hrs] at hrs_w
    refine ⟨hwlt.1, hwlt.2.1, ?_, hwlt.2.2.2.1, hwlt.2.2.2.2.1, hwlt.2.2.2.2.2⟩
    intro g hg v hv
    simp only [List.mem_cons] at hg
    rcases hg with rfl | hg
    · exact hrs_w f (by simp) v hv
    · exact hrs_w g (by simp [hg]) v hv
  refine ⟨hs'_mem ▸ hcm', hs'_mem ▸ hblt', hs'_mem ▸ hpos', hs'_hr, hs'_sp, hs'_15, hnodup', ?_,
    hstk1, hstk2, hstk3, hstk4, hctx1, hctx2, hctx3, hctx4, hf1, hf2, hf3,
    hds1, hds2, hds3, hds4, hds5, haux1, haux2, haux3, haux4, haux5,
    hrs1, hrs2, hrs3', hrs4, hrs5', hloc1, hloc2, hloc3, hloc4, hloc5,
    hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3,
    htb1, htb2, htb3, htb4, htb5, hkv1, hkv2, hkv3, hkv4, hkv5, hkv6,
    hinp1, hout1, hout2, hout3, hout4, hout5,
    hsl0, hsl1, hsl2, hsl3, hsl4, hsl5, hsl6, by simpa using hsl7,
    hsl8, hsl9, by simpa [rsLocBase] using hsl10, hsl11, hsl12, hsl13, hsl14, hsl15,
    hsl16, hsl17, hsl18, hsl19, hsl20, hsl21, hsl22, hsl23,
    hsl24, hsl25, hsl26, hsl27, hsl28, hsl29, hsl30, hsl31,
    hpds, hdsle, hpaux, hauxle, hrsrel', by simpa using hrsle, hlocle',
    hmemrel, htabsrel, htabsle, hinprel, houtrel, houtle, hwlt', hfit⟩
  simpa [hs'_mem] using hmem'

theorem adv_pc_loop_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {a : Nat} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = a) (h : vc.wf pr ls s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (hm1 : (a, (.ofInt (.ld 0 15 48)), 4) ∈ SporosVmBin.vmTable)
    (hm2 : (a + 4, (.ofInt (.ld 1 15 56)), 4) ∈ SporosVmBin.vmTable)
    (hm3 : (a + 8, (.ofInt (.aluI .sub 1 1)), 4) ∈ SporosVmBin.vmTable)
    (hm4 : (a + 12, (.ofInt (.shlI 1 5)), 4) ∈ SporosVmBin.vmTable)
    (hm5 : (a + 16, (.ofInt (.alu .add 0 1)), 3) ∈ SporosVmBin.vmTable)
    (hm6 : (a + 19, (.ofInt (.ld 1 0 8)), 4) ∈ SporosVmBin.vmTable)
    (hm7 : (a + 23, (.ofInt (.aluI .add 1 1)), 4) ∈ SporosVmBin.vmTable)
    (hm8 : (a + 27, (.ofInt (.st 0 8 1)), 4) ∈ SporosVmBin.vmTable)
    (hm9 : (a + 31, (.ofInt (.jmp 4194480)), 5) ∈ SporosVmBin.vmTable) :
    let rs' := { vc.rsReg with data := writeBytes vc.rsReg.data (rest.length * 32 + 8) (bytes8 (f.pc + 1)) }
    let ls' := { ls with rs := { f with pc := f.pc + 1 } :: rest }
    ∃ s', Reaches vmImage s s' ∧ s'.rip = SporosVmBin.L_vm_loop ∧ s'.k = s.k ∧
      { vc with rsReg := rs' }.wf pr ls' s' ∧ VmRel pr ls' s' := by
  have hr := h.2.2.2.1
  have hsp := h.2.2.2.2.1
  have h15 := h.2.2.2.2.2.1
  have hmem := h.1.2
  have hmem_rs := vc.mem_rs h
  have ⟨hrsRw, _, _, _, hrsHi, _, _, _, _, _, hrs_rel, hrs_le, _⟩ := (vc.arena_bounds h).2.2
  have ⟨hrsBase, hrsLen, _, _, _, _, _⟩ := vc.rd_ctx_rs_code_slots h
  rw [hrs] at hrsLen hrs_rel hrs_le
  simp only [RsLocRel] at hrs_rel
  rcases hrs_rel with ⟨_, hrl_len, _, _, hfpc_lt, _, _, _, _⟩
  have h01 : X86Vm.step vmImage s = .ok { s.set 0 (some vc.rsReg.base) with rip := a + 4 } := by
    have hm1' : (s.rip, (.ofInt (.ld 0 15 48)), 4) ∈ SporosVmBin.vmTable := by rwa [hrip]
    rw [step_of_table hm1', exec_ld h15 hrsBase, hrip]
  let s1 := { s.set 0 (some vc.rsReg.base) with rip := a + 4 }
  have hr1 : RegsOk s1 := regsOk_set hr (by decide) vc.rsReg.base
  have hs1_0 : s1.reg 0 = some vc.rsReg.base := by
    simp only [s1, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr.1]; decide)]
  have hs1_15 : s1.reg 15 = some ctxBase := by
    simp only [s1, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 0 ≠ 15)]; exact h15
  have hrsLen' : rd64 s1.mem (ea ctxBase 56) = some (rest.length + 1) := by
    simpa [s1, List.length_cons] using hrsLen
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 1 (some (rest.length + 1)) with rip := a + 8 } := by
    rw [step_of_table hm2, exec_ld hs1_15 hrsLen']
  let s2 := { s1.set 1 (some (rest.length + 1)) with rip := a + 8 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) (rest.length + 1)
  have hs2_1 : s2.reg 1 = some (rest.length + 1) := by
    simp only [s2, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr1.1]; decide)]
  have h47 : 2 ^ 47 < X86Env.W := by decide
  have hlenW : rest.length + 1 < X86Env.W := by
    simp only [Region.hi] at hrsHi; omega
  have hsub1 : (rest.length + 1 + X86Env.W - 1) % X86Env.W = rest.length := by
    have : rest.length + 1 + X86Env.W - 1 = rest.length + X86Env.W := by omega
    rw [this, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have h03 : X86Vm.step vmImage s2 =
      .ok { s2.set 1 (some rest.length) with fl := (X86Int.aluF .sub (rest.length + 1) 1).2, rip := a + 12 } := by
    rw [step_of_table hm3, exec_aluI hs2_1]
    dsimp only; rw [Nat.mod_eq_of_lt (by decide : 1 < X86Env.W), aluF_sub_val, hsub1]
  let s3 := { s2.set 1 (some rest.length) with fl := (X86Int.aluF .sub (rest.length + 1) 1).2, rip := a + 12 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) rest.length
  have hs3_1 : s3.reg 1 = some rest.length := by
    simp only [s3, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr2.1]; decide)]
  have hshW : rest.length * 32 < X86Env.W := by
    simp only [Region.hi] at hrsHi; omega
  have h04 : X86Vm.step vmImage s3 =
      .ok { s3.set 1 (some (rest.length * 32)) with fl := X86Int.Flags.undef, rip := a + 16 } := by
    rw [step_of_table hm4, exec_shlI_lt (c := 5) hs3_1 (by rw [show 2 ^ (5 % 64) = 32 from by decide]; exact hshW)]
  let s4 := { s3.set 1 (some (rest.length * 32)) with fl := X86Int.Flags.undef, rip := a + 16 }
  have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) (rest.length * 32)
  have hs4_0 : s4.reg 0 = some vc.rsReg.base := by
    simp only [s4, s3, s2, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0)]; exact hs1_0
  have hs4_1 : s4.reg 1 = some (rest.length * 32) := by
    simp only [s4, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr3.1]; decide)]
  have hrs_off_W : vc.rsReg.base + rest.length * 32 + 16 < X86Env.W := by
    simp only [Region.hi] at hrsHi; omega
  have h05 : X86Vm.step vmImage s4 =
      .ok { s4.set 0 (some (vc.rsReg.base + rest.length * 32)) with
        fl := (X86Int.aluF .add vc.rsReg.base (rest.length * 32)).2, rip := a + 19 } := by
    rw [step_of_table hm5, exec_alu hs4_0 hs4_1]
    dsimp only; rw [aluF_add_val, Nat.mod_eq_of_lt (by omega)]
  let s5 := { s4.set 0 (some (vc.rsReg.base + rest.length * 32)) with
    fl := (X86Int.aluF .add vc.rsReg.base (rest.length * 32)).2, rip := a + 19 }
  have hr5 : RegsOk s5 := regsOk_set hr4 (by decide) (vc.rsReg.base + rest.length * 32)
  have hs5_0 : s5.reg 0 = some (vc.rsReg.base + rest.length * 32) := by
    simp only [s5, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr4.1]; decide)]
  have hrd_pc : rd64 s5.mem (ea (vc.rsReg.base + rest.length * 32) 8) = some f.pc := by
    rw [ea_add_eq (by omega)]
    refine rd64_of_mem hmem hmem_rs (by omega)
      (by simp only [Region.hi] at hrsHi ⊢; omega) ?_
    rw [show vc.rsReg.base + rest.length * 32 + 8 - vc.rsReg.base = rest.length * 32 + 8 by omega]
    exact hfpc_lt
  have h06 : X86Vm.step vmImage s5 = .ok { s5.set 1 (some f.pc) with rip := a + 23 } := by
    rw [step_of_table hm6, exec_ld hs5_0 hrd_pc]
  let s6 := { s5.set 1 (some f.pc) with rip := a + 23 }
  have hr6 : RegsOk s6 := regsOk_set hr5 (by decide) f.pc
  have hs6_1 : s6.reg 1 = some f.pc := by
    simp only [s6, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr5.1]; decide)]
  have h07 : X86Vm.step vmImage s6 =
      .ok { s6.set 1 (some (f.pc + 1)) with fl := (X86Int.aluF .add f.pc 1).2, rip := a + 27 } := by
    rw [step_of_table hm7, exec_aluI_add_lt hs6_1 (by omega), Nat.mod_eq_of_lt hpcW]
  let s7 := { s6.set 1 (some (f.pc + 1)) with fl := (X86Int.aluF .add f.pc 1).2, rip := a + 27 }
  have hr7 : RegsOk s7 := regsOk_set hr6 (by decide) (f.pc + 1)
  have hs7_0 : s7.reg 0 = some (vc.rsReg.base + rest.length * 32) := by
    simp only [s7, s6, X86Int.St.reg, X86Int.St.set, List.getElem?_set_ne (by decide : 1 ≠ 0)]; exact hs5_0
  have hs7_1 : s7.reg 1 = some (f.pc + 1) := by
    simp only [s7, X86Int.St.reg, X86Int.St.set]; rw [getElem?_set_getD_self (by rw [hr6.1]; decide)]
  have hwr_goal := wr64_region (m := s7.mem) (r := vc.rsReg) (a := ea (vc.rsReg.base + rest.length * 32) 8)
    (v := f.pc + 1) hmem hmem_rs hrsRw (by rw [ea_add_eq (by omega)]; omega)
    (by rw [ea_add_eq (by omega)]; simp only [Region.hi] at hrsHi ⊢; omega)
  obtain ⟨_, m', hwr, _, _, _, hrs'_mem, hoth⟩ := hwr_goal
  have hwr' : wr64 s7.mem (ea vc.rsReg.base (rest.length * 32 + 8)) (f.pc + 1) = some m' := by
    rw [ea_add_eq (by omega)] at hwr ⊢
    simpa [Nat.add_assoc] using hwr
  have hrs'_mem' : { vc.rsReg with data := writeBytes vc.rsReg.data (rest.length * 32 + 8) (bytes8 (f.pc + 1)) } ∈ m' := by
    rw [ea_add_eq (by omega), show vc.rsReg.base + rest.length * 32 + 8 - vc.rsReg.base = rest.length * 32 + 8 by omega] at hrs'_mem
    exact hrs'_mem
  have h08 : X86Vm.step vmImage s7 = .ok { s7 with mem := m', rip := a + 31 } := by
    rw [step_of_table hm8, exec_st hs7_0 hs7_1 hwr]
  let s8 := { s7 with mem := m', rip := a + 31 }
  have h09 : X86Vm.step vmImage s8 = .ok { s8 with rip := SporosVmBin.L_vm_loop } := by
    rw [step_of_table hm9, exec_jmp]; rfl
  let s9 := { s8 with rip := SporosVmBin.L_vm_loop }
  have hs9_sp : s9.reg 4 = some vc.sp := by
    simp only [s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 4), List.getElem?_set_ne (by decide : 0 ≠ 4)]; exact hsp
  have hs9_15 : s9.reg 15 = some ctxBase := by
    simp only [s9, s8, s7, s6, s5, s4, s3, s2, s1, X86Int.St.reg, X86Int.St.set,
      List.getElem?_set_ne (by decide : 1 ≠ 15), List.getElem?_set_ne (by decide : 0 ≠ 15)]; exact h15
  have hwf9 := wf_write_rs_top_pc (s' := s9) h hrs hpcW hwr' hrs'_mem' hoth rfl hr7 hs9_sp hs9_15
  have hreach : Reaches vmImage s s9 :=
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
      (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step_trans h07 (Reaches.step_trans h08 (Reaches.step h09))))))))
  exact ⟨s9, hreach, rfl, rfl, hwf9, rfl, _, hwf9⟩

theorem L_adv_pc_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = 4195385) (h : vc.wf pr ls s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W) :
    let rs' := { vc.rsReg with data := writeBytes vc.rsReg.data (rest.length * 32 + 8) (bytes8 (f.pc + 1)) }
    let ls' := { ls with rs := { f with pc := f.pc + 1 } :: rest }
    ∃ s', Reaches vmImage s s' ∧ s'.rip = SporosVmBin.L_vm_loop ∧ s'.k = s.k ∧
      { vc with rsReg := rs' }.wf pr ls' s' ∧ VmRel pr ls' s' :=
  adv_pc_loop_spec hrip h hrs hpcW
    (mem_vmTable_of_chunk 3 rfl (by decide : (4195385, (.ofInt (.ld 0 15 48)), 4) ∈ SporosVmBin.vmTable_3))
    (mem_vmTable_of_chunk 3 rfl (by decide : (4195389, (.ofInt (.ld 1 15 56)), 4) ∈ SporosVmBin.vmTable_3))
    (mem_vmTable_of_chunk 3 rfl (by decide : (4195393, (.ofInt (.aluI .sub 1 1)), 4) ∈ SporosVmBin.vmTable_3))
    (mem_vmTable_of_chunk 3 rfl (by decide : (4195397, (.ofInt (.shlI 1 5)), 4) ∈ SporosVmBin.vmTable_3))
    (mem_vmTable_of_chunk 3 rfl (by decide : (4195401, (.ofInt (.alu .add 0 1)), 3) ∈ SporosVmBin.vmTable_3))
    (mem_vmTable_of_chunk 3 rfl (by decide : (4195404, (.ofInt (.ld 1 0 8)), 4) ∈ SporosVmBin.vmTable_3))
    (mem_vmTable_of_chunk 3 rfl (by decide : (4195408, (.ofInt (.aluI .add 1 1)), 4) ∈ SporosVmBin.vmTable_3))
    (mem_vmTable_of_chunk 3 rfl (by decide : (4195412, (.ofInt (.st 0 8 1)), 4) ∈ SporosVmBin.vmTable_3))
    (mem_vmTable_of_chunk 3 rfl (by decide : (4195416, (.ofInt (.jmp 4194480)), 5) ∈ SporosVmBin.vmTable_3))

private theorem nodup_replace_val {α : Type} [DecidableEq α] (old new : α) :
    ∀ {l : List α}, l.Nodup → (∀ x ∈ l, x ≠ old → x ≠ new) →
      (l.map (fun x => if x = old then new else x)).Nodup
  | [], _, _ => List.nodup_nil
  | x :: xs, hn, hne => by
      rw [List.nodup_cons] at hn
      rw [List.map_cons, List.nodup_cons]
      refine ⟨?_, nodup_replace_val old new hn.2 (fun y hy => hne y (List.mem_cons_of_mem x hy))⟩
      intro hmem
      rw [List.mem_map] at hmem
      obtain ⟨y, hy, hy_eq⟩ := hmem
      by_cases hxo : x = old <;> by_cases hyo : y = old
      · exact hn.1 (hxo ▸ hyo ▸ hy)
      · simp [hxo, hyo] at hy_eq
        exact hne y (List.mem_cons_of_mem x hy) hyo hy_eq
      · simp [hxo, hyo] at hy_eq
        exact hne x (List.mem_cons_self ..) hxo hy_eq.symm
      · simp [hxo, hyo] at hy_eq
        exact hn.1 (hy_eq ▸ hy)

private theorem ne_a_of_disj {a n : Nat} {u : Region} (hn : 0 < n) (hul : 0 < u.data.length)
    (h : a + n ≤ u.base ∨ u.hi ≤ a) : u.base ≠ a := by
  simp only [Region.hi] at h; omega

theorem wf_epilogue_finish {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s s' : X86Vm.St} {m' : Mem} {dsReg' : Region} {newDs : List SporosProvLen.LVal}
    (hcm' : CodeMem m') (hbl' : BytesLt m') (hpos' : ∀ u ∈ m', 0 < u.data.length)
    (hs'_mem : s'.mem = m') (hs'_hr : RegsOk s')
    (hs'_sp : s'.reg 4 = some vc.sp) (hs'_15 : s'.reg 15 = some ctxBase)
    (h : vc.wf pr ls s)
    (hds_base : dsReg'.base = vc.dsReg.base) (hds_rw : dsReg'.rw = true)
    (hds_len : dsReg'.data.length = vc.dsCap * 16) (hds_hi : dsReg'.hi ≤ 2 ^ 47)
    (hpds' : PairListRel dsReg'.data newDs.length newDs) (hle' : newDs.length ≤ vc.dsCap)
    (hwlt_ds' : ∀ v ∈ newDs, v.w < X86Env.W ∧ v.plen < X86Env.W)
    (hds'_mem : dsReg' ∈ m')
    (hctx'_mem : { vc.ctxReg with data := writeBytes vc.ctxReg.data 8 (bytes8 newDs.length) } ∈ m')
    (hoth : ∀ u ∈ s.mem, u.base ≠ vc.dsReg.base → u.base ≠ ctxBase → u ∈ m') :
    let ctx' := { vc.ctxReg with data := writeBytes vc.ctxReg.data 8 (bytes8 newDs.length) }
    { vc with ctxReg := ctx', dsReg := dsReg' }.wf pr { ls with ds := newDs } s' := by
  obtain ⟨_hcm, _hblt, _hpos, _, _, _, hnodup, hmem,
    hstk1, hstk2, hstk3, hstk4, hctx1, hctx2, hctx3, hctx4, hf1, hf2, hf3,
    _hds1, hds2, _hds3, hds4, _hds5, haux1, haux2, haux3, haux4, haux5,
    hrs1, hrs2, hrs3, hrs4, hrs5, hloc1, hloc2, hloc3, hloc4, hloc5,
    hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3,
    htb1, htb2, htb3, htb4, htb5, hkv1, hkv2, hkv3, hkv4, hkv5, hkv6,
    hinp1, hout1, hout2, hout3, hout4, hout5,
    hsl0, _hsl1, hsl2, hsl3, hsl4, hsl5, hsl6, hsl7,
    hsl8, hsl9, hsl10, hsl11, hsl12, hsl13, hsl14, hsl15,
    hsl16, hsl17, hsl18, hsl19, hsl20, hsl21, hsl22, hsl23,
    hsl24, hsl25, hsl26, hsl27, hsl28, hsl29, hsl30, hsl31,
    _hpds, _hdsle, hpaux, hauxle, hrsrel, hrsle, hlocle,
    hmemrel, htabsrel, htabsle, hinprel, houtrel, houtle, hwlt, hfit⟩ := h
  let ctx' := { vc.ctxReg with data := writeBytes vc.ctxReg.data 8 (bytes8 newDs.length) }
  have hcm'' : CodeMem s'.mem := hs'_mem ▸ hcm'
  have hbl'' : BytesLt s'.mem := hs'_mem ▸ hbl'
  have hpos'' : ∀ u ∈ s'.mem, 0 < u.data.length := hs'_mem ▸ hpos'
  have hnodup0 : (List.map Region.base vc.regions).Nodup := hnodup
  simp only [VmCtx.regions, List.map_cons, List.map_nil] at hnodup0
  have hnodup' : (List.map Region.base ({ vc with ctxReg := ctx', dsReg := dsReg' }).regions).Nodup := by
    simpa only [VmCtx.regions, List.map_cons, List.map_nil, hds_base, ctx'] using hnodup
  have hmem' : ∀ r ∈ ({ vc with ctxReg := ctx', dsReg := dsReg' }).regions, r ∈ s'.mem := by
    rw [hs'_mem]
    intro r hr_mem
    simp only [VmCtx.regions, List.mem_cons, List.not_mem_nil, or_false] at hr_mem
    rcases hr_mem with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals first
      | exact hds'_mem
      | exact hctx'_mem
      | exact hoth _ (hmem _ (by simp [VmCtx.regions]))
          (by intro heq; rw [heq] at hnodup0; simp [List.nodup_cons] at hnodup0)
          (by intro heq; rw [← hctx1] at heq; rw [heq] at hnodup0; simp [List.nodup_cons] at hnodup0)
  have hctx3_n : 4096 ≤ vc.ctxReg.data.length := hctx3
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hds_hi' : dsReg'.base + dsReg'.data.length ≤ 2 ^ 47 := hds_hi
  have hctx3' : ctxSize ≤ ctx'.data.length := by
    dsimp [ctx']; rw [writeBytes_length]; exact hctx3
  have hctx4' : ctx'.hi ≤ 2 ^ 47 := by
    simpa only [Region.hi, ctx', writeBytes_length] using hctx4
  have hsl0' : leVal ((ctx'.data.drop 0).take 8) = dsReg'.base := by
    change leVal (List.take 8 (List.drop 0 (writeBytes vc.ctxReg.data 8 (bytes8 newDs.length)))) = dsReg'.base
    rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; decide) (by omega), hds_base]
    exact hsl0
  have hdsCapW : newDs.length < X86Env.W := by omega
  have hsl1' : leVal ((ctx'.data.drop 8).take 8) = newDs.length := by
    dsimp [ctx']; rw [leVal_writeBytes_same _ _ _ (by omega), Nat.mod_eq_of_lt hdsCapW]
  have hdisj_slot : ∀ off, 16 ≤ off → off + 8 ≤ ctxSize →
      leVal (List.take 8 (List.drop off ctx'.data)) = leVal (List.take 8 (List.drop off vc.ctxReg.data)) := by
    intro off hoff lim
    dsimp only [ctxSize] at lim
    dsimp [ctx']
    rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; exact Or.inr hoff) (by omega)]
  have hwlt' : ({ ls with ds := newDs } : SporosProvLen.LSt).WordsLt :=
    ⟨hwlt_ds', hwlt.2.1, hwlt.2.2.1, hwlt.2.2.2.1, hwlt.2.2.2.2.1, hwlt.2.2.2.2.2⟩
  refine ⟨hcm'', hbl'', hpos'', hs'_hr, hs'_sp, hs'_15, hnodup', hmem',
    hstk1, hstk2, hstk3, hstk4, hctx1, hctx2, hctx3', hctx4', hf1, hf2, hf3,
    hds_rw, hds2, hds_len, hds4, hds_hi, haux1, haux2, haux3, haux4, haux5,
    hrs1, hrs2, hrs3, hrs4, hrs5, hloc1, hloc2, hloc3, hloc4, hloc5,
    hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3,
    htb1, htb2, htb3, htb4, htb5, hkv1, hkv2, hkv3, hkv4, hkv5, hkv6,
    hinp1, hout1, hout2, hout3, hout4, hout5,
    hsl0', hsl1',
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl2,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl3,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl4,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl5,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl6,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl7,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl8,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl9,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl10,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl11,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl12,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl13,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl14,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl15,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl16,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl17,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl18,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl19,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl20,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl21,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl22,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl23,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl24,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl25,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl26,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl27,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl28,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl29,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl30,
    by rw [hdisj_slot _ (by decide) (by decide)]; exact hsl31,
    hpds', hle', hpaux, hauxle, hrsrel, hrsle, hlocle,
    hmemrel, htabsrel, htabsle, hinprel, houtrel, houtle, hwlt', hfit⟩

theorem wf_write_ctx8_drop {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s s' : X86Vm.St} {m' : Mem} {e_in : Nat}
    (h : vc.wf pr ls s) (hein : e_in ≤ ls.ds.length)
    (hwr : wr64 s.mem (ea ctxBase 8) (ls.ds.length - e_in) = some m')
    (hctx'_mem : { vc.ctxReg with data := writeBytes vc.ctxReg.data 8 (bytes8 (ls.ds.length - e_in)) } ∈ m')
    (hoth : ∀ u ∈ s.mem, u.base ≠ ctxBase → u ∈ m')
    (hs'_mem : s'.mem = m') (hs'_hr : RegsOk s')
    (hs'_sp : s'.reg 4 = some vc.sp) (hs'_15 : s'.reg 15 = some ctxBase) :
    let ctx' := { vc.ctxReg with data := writeBytes vc.ctxReg.data 8 (bytes8 (ls.ds.length - e_in)) }
    { vc with ctxReg := ctx' }.wf pr { ls with ds := ls.ds.drop e_in } s' := by
  have ⟨hcm, hblt, hpos, _, _, _, hnodup, hmem,
    _, _, _, _, hctx1, _, _, _, _, _, _,
    hds1, _, hds3, _, hds5, _⟩ := h
  have hab := VmCtx.arena_bounds h
  have hpds : PairListRel vc.dsReg.data ls.ds.length ls.ds := hab.1.2.2.2.2.2.1
  have hwlt : ls.WordsLt := by
    have ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
      _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
      _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
      _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
      _, _, _, _, _, _, _, _, _, _, _, _, _, hwlt, _⟩ := h
    exact hwlt
  have hcm' := hcm.wr64_preserve hwr
  have hbl' := bytesLt_wr64 hblt hwr
  have hpos' := len_pos_wrN hpos hwr
  have hpds' : PairListRel vc.dsReg.data (ls.ds.drop e_in).length (ls.ds.drop e_in) := by
    rw [List.length_drop]; exact pairListRel_drop hpds hein
  have hds_ne_ctx : vc.dsReg.base ≠ ctxBase := by
    simp only [VmCtx.regions, List.map_cons, List.map_nil] at hnodup
    intro heq; rw [← hctx1] at heq; rw [heq] at hnodup; simp [List.nodup_cons] at hnodup
  simpa [List.length_drop] using wf_epilogue_finish hcm' hbl' hpos' hs'_mem hs'_hr hs'_sp hs'_15 h
    rfl hds1 hds3 hds5 hpds' (by have := hab.1.2.2.2.2.2.2; rw [List.length_drop]; omega)
    (fun v hv => hwlt.1 v (List.mem_of_mem_drop hv))
    (hoth _ (hmem _ (by simp [VmCtx.regions])) hds_ne_ctx)
    (by simpa [List.length_drop] using hctx'_mem)
    (fun u hu _ hnc => hoth u hu hnc)

private theorem eraseIdx_mem_subset {α : Type} :
    ∀ (l : List α) (idx : Nat) {x : α}, x ∈ List.eraseIdx l idx → x ∈ l
  | [], _, _, hx => hx
  | _ :: xs, 0, _, hx => List.mem_cons_of_mem _ hx
  | y :: xs, idx + 1, x, hx => by
      simp only [List.eraseIdx_cons_succ, List.mem_cons] at hx ⊢
      rcases hx with rfl | hx
      · exact Or.inl rfl
      · exact Or.inr (eraseIdx_mem_subset xs idx hx)

private theorem wf_grow_ds_arena {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s s' : X86Vm.St} {m_mr m' : Mem} {stk' : Region} {a : Nat}
    (h : vc.wf pr ls s)
    (hstk'_base : stk'.base = vc.stk.base) (hstk'_rw : stk'.rw = vc.stk.rw)
    (hstk'_len : stk'.data.length = vc.stk.data.length)
    (hs'_mem : s'.mem = m') (hs'_hr : RegsOk s')
    (hs'_sp : s'.reg 4 = some vc.sp) (hs'_15 : s'.reg 15 = some ctxBase)
    (ha_hi : a + vc.dsCap * 32 ≤ 2 ^ 47)
    (ha_disj : a ≠ vc.stk.base ∧ ∀ u ∈ s.mem, u.base ≠ vc.dsReg.base → a + vc.dsCap * 32 ≤ u.base ∨ u.hi ≤ a)
    (hm'_stk : stk' ∈ m')
    (hm'_ds : { base := a, data := vc.dsReg.data ++ List.replicate (vc.dsCap * 16) 0, rw := true } ∈ m')
    (hm'_ctx : { vc.ctxReg with data := writeBytes (writeBytes vc.ctxReg.data 0 (bytes8 a)) 16 (bytes8 (2 * vc.dsCap)) } ∈ m')
    (hm'_oth : ∀ u ∈ m_mr, u.base ≠ ctxBase → u ∈ m')
    (hmr_oth : ∀ u ∈ s.mem, u.base ≠ vc.stk.base → u.base ≠ vc.dsReg.base → u ∈ m_mr)
    (hbl' : BytesLt m') (hpos' : ∀ u ∈ m', 0 < u.data.length) (hcm' : CodeMem m') :
    let dsReg' := { base := a, data := vc.dsReg.data ++ List.replicate (vc.dsCap * 16) 0, rw := true }
    let ctx' := { vc.ctxReg with data := writeBytes (writeBytes vc.ctxReg.data 0 (bytes8 a)) 16 (bytes8 (2 * vc.dsCap)) }
    { vc with stk := stk', ctxReg := ctx', dsReg := dsReg', dsCap := 2 * vc.dsCap }.wf pr ls s' := by
  obtain ⟨_hcm, _hblt, hpos, _, _, _, hnodup, hmem,
    hstk1, hstk2, hstk3, hstk4, hctx1, hctx2, hctx3, hctx4, hf1, hf2, hf3,
    _hds1, hds2, hds3, hds4, hds5, haux1, haux2, haux3, haux4, haux5,
    hrs1, hrs2, hrs3, hrs4, hrs5, hloc1, hloc2, hloc3, hloc4, hloc5,
    hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3,
    htb1, htb2, htb3, htb4, htb5, hkv1, hkv2, hkv3, hkv4, hkv5, hkv6,
    hinp1, hout1, hout2, hout3, hout4, hout5,
    _hsl0, hsl1, _hsl2, hsl3, hsl4, hsl5, hsl6, hsl7,
    hsl8, hsl9, hsl10, hsl11, hsl12, hsl13, hsl14, hsl15,
    hsl16, hsl17, hsl18, hsl19, hsl20, hsl21, hsl22, hsl23,
    hsl24, hsl25, hsl26, hsl27, hsl28, hsl29, hsl30, hsl31,
    hpds, hdsle, hpaux, hauxle, hrsrel, hrsle, hlocle,
    hmemrel, htabsrel, htabsle, hinprel, houtrel, houtle, hwlt, hfit⟩ := h
  let dsReg' : Region := { base := a, data := vc.dsReg.data ++ List.replicate (vc.dsCap * 16) 0, rw := true }
  let ctx1 : Region := { vc.ctxReg with data := writeBytes vc.ctxReg.data 0 (bytes8 a) }
  let ctx' : Region := { vc.ctxReg with data := writeBytes ctx1.data 16 (bytes8 (2 * vc.dsCap)) }
  have hctx3_n : 4096 ≤ vc.ctxReg.data.length := hctx3
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hds_hi' : vc.dsReg.base + vc.dsReg.data.length ≤ 2 ^ 47 := hds5
  have hnodup0 : (List.map Region.base vc.regions).Nodup := hnodup
  simp only [VmCtx.regions, List.map_cons, List.map_nil, List.nodup_cons, List.mem_cons,
    List.not_mem_nil, or_false, not_or] at hnodup0
  have hnodup' : (List.map Region.base
      ({ vc with stk := stk', ctxReg := ctx', dsReg := dsReg', dsCap := 2 * vc.dsCap } : VmCtx).regions).Nodup := by
    have hmap_eq : List.map Region.base
        ({ vc with stk := stk', ctxReg := ctx', dsReg := dsReg', dsCap := 2 * vc.dsCap } : VmCtx).regions =
        (List.map Region.base vc.regions).map (fun x => if x = vc.dsReg.base then a else x) := by
      simp only [VmCtx.regions, List.map_cons, List.map_nil, hstk'_base, ctx', dsReg']
      simp only [↓reduceIte]
      repeat rw [if_neg (by intro heq; simp only [heq, not_true_eq_false, false_and, and_false] at hnodup0)]
    rw [hmap_eq]
    refine nodup_replace_val vc.dsReg.base a hnodup ?_
    intro x hx hxne
    rw [List.mem_map] at hx
    obtain ⟨u, hu, rfl⟩ := hx
    exact ne_a_of_disj (by omega) (hpos u (hmem u hu)) (ha_disj.2 u (hmem u hu) hxne)
  have hmem' : ∀ r ∈ ({ vc with stk := stk', ctxReg := ctx', dsReg := dsReg', dsCap := 2 * vc.dsCap } : VmCtx).regions,
      r ∈ s'.mem := by
    rw [hs'_mem]
    intro r hr_mem
    simp only [VmCtx.regions, List.mem_cons, List.not_mem_nil, or_false] at hr_mem
    rcases hr_mem with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals first
      | exact hm'_stk
      | exact hm'_ctx
      | exact hm'_ds
      | exact hm'_oth _ (hmr_oth _ (hmem _ (by simp [VmCtx.regions]))
            (by intro heq; simp only [heq, not_true_eq_false, false_and, and_false] at hnodup0)
            (by intro heq; simp only [heq, not_true_eq_false, false_and, and_false] at hnodup0))
            (by intro heq; rw [← hctx1] at heq; simp only [heq, not_true_eq_false, false_and, and_false] at hnodup0)
  have hstk1' : stk'.rw = true := hstk'_rw ▸ hstk1
  have hstk2' : stk'.base + 32 ≤ vc.sp := hstk'_base ▸ hstk2
  have hstk3' : vc.sp + 8 ≤ stk'.hi := by
    simpa only [Region.hi, hstk'_base, hstk'_len] using hstk3
  have hctx3' : ctxSize ≤ ctx'.data.length := by
    dsimp [ctx', ctx1]; rw [writeBytes_length, writeBytes_length]; exact hctx3
  have hctx4' : ctx'.hi ≤ 2 ^ 47 := by
    simpa only [Region.hi, ctx', ctx1, writeBytes_length] using hctx4
  have hds3' : dsReg'.data.length = (2 * vc.dsCap) * 16 := by
    dsimp [dsReg']; rw [List.length_append, List.length_replicate, hds3]; omega
  have hds4' : (2 * vc.dsCap) * 16 % X86Env.page = 0 := by
    rw [show (2 * vc.dsCap) * 16 = 2 * (vc.dsCap * 16) by omega, Nat.mul_mod, hds4, Nat.mul_zero, Nat.zero_mod]
  have hds5' : dsReg'.hi ≤ 2 ^ 47 := by
    simpa only [Region.hi, dsReg', List.length_append, List.length_replicate, hds3,
      show vc.dsCap * 16 + vc.dsCap * 16 = vc.dsCap * 32 by omega] using ha_hi
  have hsl0' : leVal ((ctx'.data.drop 0).take 8) = a := by
    change leVal (List.take 8 (List.drop 0 (writeBytes (writeBytes vc.ctxReg.data 0 (bytes8 a)) 16 (bytes8 (2 * vc.dsCap))))) = a
    rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; decide) (by rw [writeBytes_length]; omega)]
    rw [leVal_writeBytes_same _ _ _ (by omega), Nat.mod_eq_of_lt (by omega)]
  have hctx1_disj8 : leVal ((ctx1.data.drop 8).take 8) = leVal ((vc.ctxReg.data.drop 8).take 8) := by
    dsimp [ctx1]; rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; decide) (by omega)]
  have hsl1' : leVal ((ctx'.data.drop 8).take 8) = ls.ds.length := by
    dsimp [ctx']
    rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; decide) (by dsimp [ctx1]; rw [writeBytes_length]; omega)]
    rw [hctx1_disj8]; exact hsl1
  have h2capW : 2 * vc.dsCap < X86Env.W := by omega
  have hsl2' : leVal ((ctx'.data.drop 16).take 8) = 2 * vc.dsCap := by
    dsimp [ctx']
    rw [leVal_writeBytes_same _ _ _ (by dsimp [ctx1]; rw [writeBytes_length]; omega), Nat.mod_eq_of_lt h2capW]
  have hdisj_slot : ∀ off, 24 ≤ off → off + 8 ≤ ctxSize →
      leVal (List.take 8 (List.drop off ctx'.data)) = leVal (List.take 8 (List.drop off vc.ctxReg.data)) := by
    intro off hoff lim
    dsimp only [ctxSize] at lim
    dsimp [ctx', ctx1]
    rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; exact Or.inr (by omega))
      (by rw [writeBytes_length]; omega)]
    rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; exact Or.inr (by omega)) (by omega)]
  have hpds' : PairListRel dsReg'.data ls.ds.length ls.ds := by
    dsimp [dsReg']; exact pairListRel_extend_zero hpds (by rw [hds3]; omega)
  have hsl_keep : ∀ {off val : Nat}, 24 ≤ off → off + 8 ≤ ctxSize →
      leVal (List.take 8 (List.drop off vc.ctxReg.data)) = val →
      leVal (List.take 8 (List.drop off ctx'.data)) = val := fun h1 h2 h =>
    (hdisj_slot _ h1 h2).trans h
  have hcm'' : CodeMem s'.mem := hs'_mem ▸ hcm'
  have hbl'' : BytesLt s'.mem := hs'_mem ▸ hbl'
  have hpos'' : ∀ u ∈ s'.mem, 0 < u.data.length := hs'_mem ▸ hpos'
  have hds2' : 4 ≤ 2 * vc.dsCap := Nat.le_trans hds2 (Nat.le_mul_of_pos_left vc.dsCap (by decide))
  have hdsle' : ls.ds.length ≤ 2 * vc.dsCap := Nat.le_trans hdsle (Nat.le_mul_of_pos_left vc.dsCap (by decide))
  refine ⟨hcm'', hbl'', hpos'', hs'_hr, hs'_sp, hs'_15, hnodup', hmem',
    hstk1', hstk2', hstk3', hstk4, hctx1, hctx2, hctx3', hctx4', hf1, hf2, hf3,
    rfl, hds2', hds3', hds4', hds5', haux1, haux2, haux3, haux4, haux5,
    hrs1, hrs2, hrs3, hrs4, hrs5, hloc1, hloc2, hloc3, hloc4, hloc5,
    hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3,
    htb1, htb2, htb3, htb4, htb5, hkv1, hkv2, hkv3, hkv4, hkv5, hkv6,
    hinp1, hout1, hout2, hout3, hout4, hout5,
    hsl0', hsl1', hsl2',
    hsl_keep (by decide) (by decide) hsl3,
    hsl_keep (by decide) (by decide) hsl4,
    hsl_keep (by decide) (by decide) hsl5,
    hsl_keep (by decide) (by decide) hsl6,
    hsl_keep (by decide) (by decide) hsl7,
    hsl_keep (by decide) (by decide) hsl8,
    hsl_keep (by decide) (by decide) hsl9,
    hsl_keep (by decide) (by decide) hsl10,
    hsl_keep (by decide) (by decide) hsl11,
    hsl_keep (by decide) (by decide) hsl12,
    hsl_keep (by decide) (by decide) hsl13,
    hsl_keep (by decide) (by decide) hsl14,
    hsl_keep (by decide) (by decide) hsl15,
    hsl_keep (by decide) (by decide) hsl16,
    hsl_keep (by decide) (by decide) hsl17,
    hsl_keep (by decide) (by decide) hsl18,
    hsl_keep (by decide) (by decide) hsl19,
    hsl_keep (by decide) (by decide) hsl20,
    hsl_keep (by decide) (by decide) hsl21,
    hsl_keep (by decide) (by decide) hsl22,
    hsl_keep (by decide) (by decide) hsl23,
    hsl_keep (by decide) (by decide) hsl24,
    hsl_keep (by decide) (by decide) hsl25,
    hsl_keep (by decide) (by decide) hsl26,
    hsl_keep (by decide) (by decide) hsl27,
    hsl_keep (by decide) (by decide) hsl28,
    hsl_keep (by decide) (by decide) hsl29,
    hsl_keep (by decide) (by decide) hsl30,
    hsl_keep (by decide) (by decide) hsl31,
    hpds', hdsle', hpaux, hauxle, hrsrel, hrsle, hlocle,
    hmemrel, htabsrel, htabsle, hinprel, houtrel, houtle, hwlt, hfit⟩

private theorem epilogue_push_load_spec {rem' : Nat} {s : X86Vm.St}
    {sp ctxBase dsBase dsCap curLen : Nat} {ctx dsReg : Region} {outs : List SporosProvLen.LVal}
    (hrip : s.rip = 4195329) (hcm : CodeMem s.mem) (hreg : RegsOk s)
    (hs15 : s.reg 15 = some ctxBase) (hs4 : s.reg 4 = some sp)
    (hs12 : s.reg 12 = some dsBase) (hs13 : s.reg 13 = some curLen) (hs3 : s.reg 3 = some (rem' + 1))
    (hrem_le : rem' + 1 ≤ outs.length) (houts_le : outs.length ≤ 3)
    (hctx_mem : ctx ∈ s.mem) (hctx_base : ctx.base = ctxBase)
    (hctx_len : ctxSize ≤ ctx.data.length) (hctx_W : ctxBase + ctxSize ≤ X86Env.W)
    (hctx_slots : ∀ (i : Nat) (_ : i < outs.length),
      leVal ((ctx.data.drop (360 + i * 16)).take 8) = outs[i].w ∧
      leVal ((ctx.data.drop (368 + i * 16)).take 8) = outs[i].plen)
    (hds_len : dsReg.data.length = dsCap * 16) (hds_hi : dsReg.hi ≤ 2 ^ 47)
    (hcap_bound : curLen + (rem' + 1) ≤ dsCap) :
    let hv_lt : rem' < outs.length := Nat.lt_of_lt_of_le (Nat.lt_succ_self rem') hrem_le
    ∃ s7, Reaches vmImage s s7 ∧ s7.rip = 4195363 ∧ s7.mem = s.mem ∧ s7.k = s.k ∧ RegsOk s7 ∧
      s7.reg 15 = some ctxBase ∧ s7.reg 4 = some sp ∧ s7.reg 12 = some dsBase ∧
      s7.reg 13 = some curLen ∧ s7.reg 3 = some rem' ∧ s7.reg 6 = some (curLen * 16) ∧
      s7.reg 0 = some outs[rem'].w ∧ s7.reg 1 = some outs[rem'].plen := by
  have hm_wf : s.mem.wf := hcm.2
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hctx_len' : 4096 ≤ ctx.data.length := hctx_len
  have hctx_W' : ctxBase + 4096 ≤ X86Env.W := hctx_W
  have hds_hi' : dsReg.base + dsReg.data.length ≤ 2 ^ 47 := hds_hi
  have hdsCapW : dsCap * 16 < X86Env.W := by omega
  have hcurW : curLen + rem' + 1 < X86Env.W := by omega
  have hv_lt : rem' < outs.length := by omega
  have hsub1 : (X86Int.aluF .sub (rem' + 1) (1 % X86Env.W)).1 = rem' := by
    rw [Nat.mod_eq_of_lt (by decide), aluF_sub_val]
    rw [show rem' + 1 + X86Env.W - 1 = rem' + X86Env.W by omega, Nat.add_mod_right]
    exact Nat.mod_eq_of_lt (by omega)
  have hsub1_pair : X86Int.aluF .sub (rem' + 1) (1 % X86Env.W) =
      (rem', (X86Int.aluF .sub (rem' + 1) (1 % X86Env.W)).2) := Prod.ext hsub1 rfl
  have h01 : X86Vm.step vmImage s =
      .ok { s.set 3 (some rem') with fl := (X86Int.aluF .sub (rem' + 1) (1 % X86Env.W)).2, rip := 4195333 } := by
    rw [step_at_chunk 3 hrip rfl (by decide : (4195329, (.ofInt (.aluI .sub 3 1)), 4) ∈ SporosVmBin.vmTable_3),
      exec_aluI hs3, hsub1_pair, hrip]
  let s1 := { s.set 3 (some rem') with fl := (X86Int.aluF .sub (rem' + 1) (1 % X86Env.W)).2, rip := 4195333 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) rem'
  have hs1_3 : s1.reg 3 = some rem' := reg_set_self_ok hreg (by decide) (some rem')
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 2 (some rem') with rip := 4195336 } := by
    rw [step_at_chunk 3 rfl rfl (by decide : (4195333, (.ofInt (.mov 2 3)), 3) ∈ SporosVmBin.vmTable_3),
      exec_mov hs1_3]
  let s2 := { s1.set 2 (some rem') with rip := 4195336 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) rem'
  have hs2_2 : s2.reg 2 = some rem' := reg_set_self_ok hr1 (by decide) (some rem')
  have hshl2 : rem' * 2 ^ (4 % 64) % X86Env.W = rem' * 16 := by
    rw [show 2 ^ (4 % 64) = 16 by decide, Nat.mod_eq_of_lt (by omega)]
  have h03 : X86Vm.step vmImage s2 =
      .ok { s2.set 2 (some (rem' * 16)) with fl := X86Int.Flags.undef, rip := 4195340 } := by
    rw [step_at_chunk 3 rfl rfl (by decide : (4195336, (.ofInt (.shlI 2 4)), 4) ∈ SporosVmBin.vmTable_3),
      exec_shlI hs2_2, hshl2]
  let s3 := { s2.set 2 (some (rem' * 16)) with fl := X86Int.Flags.undef, rip := 4195340 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) (rem' * 16)
  have hs3_15 : s3.reg 15 = some ctxBase := by
    change (s2.set 2 (some (rem' * 16))).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s1.set 2 (some rem')).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s.set 3 (some rem')).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs15
  have hs3_2 : s3.reg 2 = some (rem' * 16) := reg_set_self_ok hr2 (by decide) (some (rem' * 16))
  have hea_w : X86Vm.eaIdx ctxBase (rem' * 16) 1 360 = ctxBase + (360 + rem' * 16) := by
    simp only [X86Vm.eaIdx, Nat.one_mul, show ctxBase + rem' * 16 + 360 = ctxBase + (360 + rem' * 16) by omega]
    exact Nat.mod_eq_of_lt (by omega)
  have hrd_w : X86Vm.rdMem .w64 s3.mem (X86Vm.eaIdx ctxBase (rem' * 16) 1 360) = some outs[rem'].w := by
    rw [hea_w]
    exact rdMem_w64_of_mem hm_wf hctx_mem (by rw [hctx_base]; omega) (by simp only [Region.hi, hctx_base]; omega)
      (by rw [hctx_base, Nat.add_sub_cancel_left]; exact (hctx_slots rem' hv_lt).1)
  have h04 : X86Vm.step vmImage s3 = .ok { s3.set 0 (some outs[rem'].w) with rip := 4195348 } := by
    rw [step_at_chunk 3 rfl rfl (by decide : (4195340, (.ldIdx .w64 0 15 2 1 360), 8) ∈ SporosVmBin.vmTable_3),
      exec_ldIdx hs3_15 hs3_2 hrd_w]
  let s4 := { s3.set 0 (some outs[rem'].w) with rip := 4195348 }
  have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) outs[rem'].w
  have hs4_15 : s4.reg 15 = some ctxBase := by
    change (s3.set 0 (some outs[rem'].w)).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs3_15
  have hs4_2 : s4.reg 2 = some (rem' * 16) := by
    change (s3.set 0 (some outs[rem'].w)).reg 2 = some (rem' * 16); rw [reg_set_ne (by decide)]; exact hs3_2
  have hea_p : X86Vm.eaIdx ctxBase (rem' * 16) 1 368 = ctxBase + (368 + rem' * 16) := by
    simp only [X86Vm.eaIdx, Nat.one_mul, show ctxBase + rem' * 16 + 368 = ctxBase + (368 + rem' * 16) by omega]
    exact Nat.mod_eq_of_lt (by omega)
  have hrd_p : X86Vm.rdMem .w64 s4.mem (X86Vm.eaIdx ctxBase (rem' * 16) 1 368) = some outs[rem'].plen := by
    rw [hea_p]
    exact rdMem_w64_of_mem hm_wf hctx_mem (by rw [hctx_base]; omega) (by simp only [Region.hi, hctx_base]; omega)
      (by rw [hctx_base, Nat.add_sub_cancel_left]; exact (hctx_slots rem' hv_lt).2)
  have h05 : X86Vm.step vmImage s4 = .ok { s4.set 1 (some outs[rem'].plen) with rip := 4195356 } := by
    rw [step_at_chunk 3 rfl rfl (by decide : (4195348, (.ldIdx .w64 1 15 2 1 368), 8) ∈ SporosVmBin.vmTable_3),
      exec_ldIdx hs4_15 hs4_2 hrd_p]
  let s5 := { s4.set 1 (some outs[rem'].plen) with rip := 4195356 }
  have hr5 : RegsOk s5 := regsOk_set hr4 (by decide) outs[rem'].plen
  have hs5_13 : s5.reg 13 = some curLen := by
    change (s4.set 1 (some outs[rem'].plen)).reg 13 = some curLen; rw [reg_set_ne (by decide)]
    change (s3.set 0 (some outs[rem'].w)).reg 13 = some curLen; rw [reg_set_ne (by decide)]
    change (s2.set 2 (some (rem' * 16))).reg 13 = some curLen; rw [reg_set_ne (by decide)]
    change (s1.set 2 (some rem')).reg 13 = some curLen; rw [reg_set_ne (by decide)]
    change (s.set 3 (some rem')).reg 13 = some curLen; rw [reg_set_ne (by decide)]; exact hs13
  have h06 : X86Vm.step vmImage s5 = .ok { s5.set 6 (some curLen) with rip := 4195359 } := by
    rw [step_at_chunk 3 rfl rfl (by decide : (4195356, X86Vm.Ins.ofInt (X86Int.Ins.mov 6 13), 3) ∈ SporosVmBin.vmTable_3),
      exec_mov hs5_13]
  let s6 := { s5.set 6 (some curLen) with rip := 4195359 }
  have hr6 : RegsOk s6 := regsOk_set hr5 (by decide) curLen
  have hs6_6 : s6.reg 6 = some curLen := reg_set_self_ok hr5 (by decide) (some curLen)
  have hshl6 : curLen * 2 ^ (4 % 64) % X86Env.W = curLen * 16 := by
    rw [show 2 ^ (4 % 64) = 16 by decide, Nat.mod_eq_of_lt (by omega)]
  have h07 : X86Vm.step vmImage s6 =
      .ok { s6.set 6 (some (curLen * 16)) with fl := X86Int.Flags.undef, rip := 4195363 } := by
    rw [step_at_chunk 3 rfl rfl (by decide : (4195359, X86Vm.Ins.ofInt (X86Int.Ins.shlI 6 4), 4) ∈ SporosVmBin.vmTable_3),
      exec_shlI hs6_6, hshl6]
  let s7 := { s6.set 6 (some (curLen * 16)) with fl := X86Int.Flags.undef, rip := 4195363 }
  have hr7 : RegsOk s7 := regsOk_set hr6 (by decide) (curLen * 16)
  have hs7_15 : s7.reg 15 = some ctxBase := by
    change (s6.set 6 (some (curLen * 16))).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s5.set 6 (some curLen)).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s4.set 1 (some outs[rem'].plen)).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    exact hs4_15
  have hs7_4 : s7.reg 4 = some sp := by
    change (s6.set 6 (some (curLen * 16))).reg 4 = some sp; rw [reg_set_ne (by decide)]
    change (s5.set 6 (some curLen)).reg 4 = some sp; rw [reg_set_ne (by decide)]
    change (s4.set 1 (some outs[rem'].plen)).reg 4 = some sp; rw [reg_set_ne (by decide)]
    change (s3.set 0 (some outs[rem'].w)).reg 4 = some sp; rw [reg_set_ne (by decide)]
    change (s2.set 2 (some (rem' * 16))).reg 4 = some sp; rw [reg_set_ne (by decide)]
    change (s1.set 2 (some rem')).reg 4 = some sp; rw [reg_set_ne (by decide)]
    change (s.set 3 (some rem')).reg 4 = some sp; rw [reg_set_ne (by decide)]; exact hs4
  have hs7_12 : s7.reg 12 = some dsBase := by
    change (s6.set 6 (some (curLen * 16))).reg 12 = some dsBase; rw [reg_set_ne (by decide)]
    change (s5.set 6 (some curLen)).reg 12 = some dsBase; rw [reg_set_ne (by decide)]
    change (s4.set 1 (some outs[rem'].plen)).reg 12 = some dsBase; rw [reg_set_ne (by decide)]
    change (s3.set 0 (some outs[rem'].w)).reg 12 = some dsBase; rw [reg_set_ne (by decide)]
    change (s2.set 2 (some (rem' * 16))).reg 12 = some dsBase; rw [reg_set_ne (by decide)]
    change (s1.set 2 (some rem')).reg 12 = some dsBase; rw [reg_set_ne (by decide)]
    change (s.set 3 (some rem')).reg 12 = some dsBase; rw [reg_set_ne (by decide)]; exact hs12
  have hs7_13 : s7.reg 13 = some curLen := by
    change (s6.set 6 (some (curLen * 16))).reg 13 = some curLen; rw [reg_set_ne (by decide)]
    change (s5.set 6 (some curLen)).reg 13 = some curLen; rw [reg_set_ne (by decide)]; exact hs5_13
  have hs7_3 : s7.reg 3 = some rem' := by
    change (s6.set 6 (some (curLen * 16))).reg 3 = some rem'; rw [reg_set_ne (by decide)]
    change (s5.set 6 (some curLen)).reg 3 = some rem'; rw [reg_set_ne (by decide)]
    change (s4.set 1 (some outs[rem'].plen)).reg 3 = some rem'; rw [reg_set_ne (by decide)]
    change (s3.set 0 (some outs[rem'].w)).reg 3 = some rem'; rw [reg_set_ne (by decide)]
    change (s2.set 2 (some (rem' * 16))).reg 3 = some rem'; rw [reg_set_ne (by decide)]
    change (s1.set 2 (some rem')).reg 3 = some rem'; rw [reg_set_ne (by decide)]; exact hs1_3
  have hs7_6 : s7.reg 6 = some (curLen * 16) := reg_set_self_ok hr6 (by decide) (some (curLen * 16))
  have hs7_0 : s7.reg 0 = some outs[rem'].w := by
    change (s6.set 6 (some (curLen * 16))).reg 0 = some outs[rem'].w; rw [reg_set_ne (by decide)]
    change (s5.set 6 (some curLen)).reg 0 = some outs[rem'].w; rw [reg_set_ne (by decide)]
    change (s4.set 1 (some outs[rem'].plen)).reg 0 = some outs[rem'].w; rw [reg_set_ne (by decide)]
    exact reg_set_self_ok hr3 (by decide) (some outs[rem'].w)
  have hs7_1 : s7.reg 1 = some outs[rem'].plen := by
    change (s6.set 6 (some (curLen * 16))).reg 1 = some outs[rem'].plen; rw [reg_set_ne (by decide)]
    change (s5.set 6 (some curLen)).reg 1 = some outs[rem'].plen; rw [reg_set_ne (by decide)]
    exact reg_set_self_ok hr4 (by decide) (some outs[rem'].plen)
  refine ⟨s7, Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
    (Reaches.step_trans h05 (Reaches.step_trans h06 (Reaches.step h07)))))),
    rfl, rfl, rfl, hr7, hs7_15, hs7_4, hs7_12, hs7_13, hs7_3, hs7_6, hs7_0, hs7_1⟩

private theorem epilogue_push_store_w_spec {rem' : Nat} {s7 : X86Vm.St}
    {sp ctxBase dsBase dsCap curLen : Nat} {dsReg : Region} {outs : List SporosProvLen.LVal}
    (hrip7 : s7.rip = 4195363) (hcm : CodeMem s7.mem) (hbl : BytesLt s7.mem)
    (hpos : ∀ u ∈ s7.mem, 0 < u.data.length) (hr7 : RegsOk s7)
    (hs7_15 : s7.reg 15 = some ctxBase) (hs7_4 : s7.reg 4 = some sp)
    (hs7_12 : s7.reg 12 = some dsBase) (hs7_13 : s7.reg 13 = some curLen)
    (hs7_3 : s7.reg 3 = some rem') (hs7_6 : s7.reg 6 = some (curLen * 16))
    (hv_lt : rem' < outs.length)
    (hs7_0 : s7.reg 0 = some outs[rem'].w) (hs7_1 : s7.reg 1 = some outs[rem'].plen)
    (hds_mem : dsReg ∈ s7.mem) (hds_base : dsReg.base = dsBase) (hds_rw : dsReg.rw = true)
    (hds_len : dsReg.data.length = dsCap * 16) (hds_hi : dsReg.hi ≤ 2 ^ 47)
    (hcap_bound : curLen + (rem' + 1) ≤ dsCap) :
    ∃ dsReg1 s8,
      X86Vm.step vmImage s7 = .ok s8 ∧ s8.rip = 4195367 ∧ s8.k = s7.k ∧
      CodeMem s8.mem ∧ BytesLt s8.mem ∧ (∀ u ∈ s8.mem, 0 < u.data.length) ∧ RegsOk s8 ∧
      s8.reg 15 = some ctxBase ∧ s8.reg 4 = some sp ∧ s8.reg 12 = some dsBase ∧
      s8.reg 13 = some curLen ∧ s8.reg 3 = some rem' ∧
      s8.reg 6 = some (curLen * 16) ∧ s8.reg 1 = some outs[rem'].plen ∧
      dsReg1.base = dsBase ∧ dsReg1.rw = true ∧ dsReg1.data.length = dsCap * 16 ∧ dsReg1.hi ≤ 2 ^ 47 ∧
      dsReg1.data = writeBytes dsReg.data (curLen * 16) (bytes8 outs[rem'].w) ∧
      dsReg1 ∈ s8.mem ∧ (∀ u ∈ s7.mem, u.base ≠ dsBase → u ∈ s8.mem) := by
  have hm_wf : s7.mem.wf := hcm.2
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hds_hi' : dsReg.base + dsCap * 16 ≤ 2 ^ 47 := by simpa only [Region.hi, hds_len] using hds_hi
  have hea_ds0 : X86Vm.eaIdx dsBase (curLen * 16) 1 0 = dsBase + curLen * 16 := by
    simp only [X86Vm.eaIdx, Nat.one_mul, Nat.add_zero]; exact Nat.mod_eq_of_lt (by rw [hds_base] at hds_hi'; omega)
  obtain ⟨_i1, m1, hwr1, _hm1_idx, _hm1_set, _hm1_wf, hds1_mem, hm1_oth⟩ :=
    wrMem_w64_region hm_wf hds_mem hds_rw (by rw [hds_base]; omega)
      (by simp only [Region.hi, hds_base, hds_len]; omega)
      (a := dsBase + curLen * 16) (v := outs[rem'].w)
  have hwr1' : X86Vm.wrMem .w64 s7.mem (X86Vm.eaIdx dsBase (curLen * 16) 1 0) outs[rem'].w = some m1 := by
    rw [hea_ds0]; exact hwr1
  have h08 : X86Vm.step vmImage s7 = .ok { s7 with rip := 4195367, mem := m1 } := by
    rw [step_at_chunk 3 hrip7 rfl (by decide : (4195363, (.stIdx .w64 12 6 1 0 0), 4) ∈ SporosVmBin.vmTable_3),
      exec_stIdx hs7_12 hs7_6 hs7_0 hwr1', hrip7]
  let s8 := { s7 with rip := 4195367, mem := m1 }
  have hcm1 : CodeMem m1 := hcm.wrMem_preserve hwr1'
  have hbl1 : BytesLt m1 :=
    bytesLt_wrMem hbl hm_wf hds_mem hds_rw (by rw [hds_base]; omega)
      (by dsimp only; simp only [Region.hi, hds_base, hds_len]; omega) hwr1'
  have hpos1 : ∀ u ∈ m1, 0 < u.data.length := len_pos_wrMem hpos hwr1'
  let dsReg1 : Region :=
    { base := dsBase, data := writeBytes dsReg.data (curLen * 16) (bytes8 outs[rem'].w), rw := true }
  have hds1_eq : dsReg1 =
      { base := dsReg.base, data := writeBytes dsReg.data (dsBase + curLen * 16 - dsReg.base) (bytes8 outs[rem'].w), rw := dsReg.rw } := by
    dsimp [dsReg1]; rw [hds_base, hds_rw, show dsBase + curLen * 16 - dsBase = curLen * 16 by omega]
  have hds1_mem' : dsReg1 ∈ m1 := by rw [hds1_eq]; exact hds1_mem
  have hds1_len : dsReg1.data.length = dsCap * 16 := by dsimp [dsReg1]; rw [writeBytes_length]; exact hds_len
  have hds1_hi : dsReg1.hi ≤ 2 ^ 47 := by
    simp only [Region.hi, dsReg1, writeBytes_length]; rw [hds_len]; exact by rw [← hds_base]; exact hds_hi'
  have hm1_oth' : ∀ u ∈ s7.mem, u.base ≠ dsBase → u ∈ s8.mem := fun u hu hnu =>
    hm1_oth u hu (by rw [hds_base]; exact hnu)
  exact ⟨dsReg1, s8, h08, rfl, rfl, hcm1, hbl1, hpos1, hr7, hs7_15, hs7_4, hs7_12, hs7_13, hs7_3,
    hs7_6, hs7_1, rfl, rfl, hds1_len, hds1_hi, rfl, hds1_mem', hm1_oth'⟩

private theorem epilogue_push_store_mem_spec {rem' : Nat} {s7 : X86Vm.St}
    {sp ctxBase dsBase dsCap curLen : Nat} {dsReg : Region} {outs curDs : List SporosProvLen.LVal}
    (hrip7 : s7.rip = 4195363) (hcm : CodeMem s7.mem) (hbl : BytesLt s7.mem)
    (hpos : ∀ u ∈ s7.mem, 0 < u.data.length) (hr7 : RegsOk s7)
    (hs7_15 : s7.reg 15 = some ctxBase) (hs7_4 : s7.reg 4 = some sp)
    (hs7_12 : s7.reg 12 = some dsBase) (hs7_13 : s7.reg 13 = some curLen)
    (hs7_3 : s7.reg 3 = some rem') (hs7_6 : s7.reg 6 = some (curLen * 16))
    (hv_lt : rem' < outs.length)
    (hs7_0 : s7.reg 0 = some outs[rem'].w) (hs7_1 : s7.reg 1 = some outs[rem'].plen)
    (hvW : outs[rem'].w < X86Env.W ∧ outs[rem'].plen < X86Env.W)
    (hds_mem : dsReg ∈ s7.mem) (hds_base : dsReg.base = dsBase) (hds_rw : dsReg.rw = true)
    (hds_len : dsReg.data.length = dsCap * 16) (hds_hi : dsReg.hi ≤ 2 ^ 47)
    (hcap_bound : curLen + (rem' + 1) ≤ dsCap) (hpds : PairListRel dsReg.data curLen curDs) :
    ∃ dsReg2 s9,
      Reaches vmImage s7 s9 ∧ s9.rip = 4195372 ∧ s9.k = s7.k ∧
      CodeMem s9.mem ∧ BytesLt s9.mem ∧ (∀ u ∈ s9.mem, 0 < u.data.length) ∧ RegsOk s9 ∧
      s9.reg 15 = some ctxBase ∧ s9.reg 4 = some sp ∧ s9.reg 12 = some dsBase ∧
      s9.reg 13 = some curLen ∧ s9.reg 3 = some rem' ∧
      dsReg2.base = dsBase ∧ dsReg2.rw = true ∧ dsReg2.data.length = dsCap * 16 ∧ dsReg2.hi ≤ 2 ^ 47 ∧
      dsReg2 ∈ s9.mem ∧ (∀ u ∈ s7.mem, u.base ≠ dsBase → u ∈ s9.mem) ∧
      PairListRel dsReg2.data (curLen + 1) (outs[rem'] :: curDs) := by
  obtain ⟨dsReg1, s8, h08, hrip8, hk8, hcm1, hbl1, hpos1, hr8, hs8_15, hs8_4, hs8_12, hs8_13, hs8_3,
    hs8_6, hs8_1, hds1_base, hds1_rw, hds1_len, hds1_hi, hds1_data, hds1_mem, hm1_oth⟩ :=
    epilogue_push_store_w_spec hrip7 hcm hbl hpos hr7 hs7_15 hs7_4 hs7_12 hs7_13 hs7_3 hs7_6
      hv_lt hs7_0 hs7_1 hds_mem hds_base hds_rw hds_len hds_hi hcap_bound
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hds1_hi' : dsReg1.base + dsCap * 16 ≤ 2 ^ 47 := by simpa only [Region.hi, hds1_len] using hds1_hi
  have hea_ds8 : X86Vm.eaIdx dsBase (curLen * 16) 1 8 = dsBase + (curLen * 16 + 8) := by
    simp only [X86Vm.eaIdx, Nat.one_mul, Nat.add_assoc]; exact Nat.mod_eq_of_lt (by rw [hds1_base] at hds1_hi'; omega)
  obtain ⟨_i2, m2, hwr2, _hm2_idx, _hm2_set, _hm2_wf, hds2_mem, hm2_oth⟩ :=
    wrMem_w64_region hcm1.2 hds1_mem hds1_rw (by rw [hds1_base]; omega)
      (by simp only [Region.hi, hds1_base, hds1_len]; omega)
      (a := dsBase + (curLen * 16 + 8)) (v := outs[rem'].plen)
  have hwr2' : X86Vm.wrMem .w64 s8.mem (X86Vm.eaIdx dsBase (curLen * 16) 1 8) outs[rem'].plen = some m2 := by
    rw [hea_ds8]; exact hwr2
  have h09 : X86Vm.step vmImage s8 = .ok { s8 with rip := 4195372, mem := m2 } := by
    rw [step_at_chunk 3 hrip8 rfl (by decide : (4195367, (.stIdx .w64 12 6 1 8 1), 5) ∈ SporosVmBin.vmTable_3),
      exec_stIdx hs8_12 hs8_6 hs8_1 hwr2', hrip8]
  let s9 := { s8 with rip := 4195372, mem := m2 }
  have hcm2 : CodeMem m2 := hcm1.wrMem_preserve hwr2'
  have hbl2 : BytesLt m2 :=
    bytesLt_wrMem hbl1 hcm1.2 hds1_mem hds1_rw (by rw [hds1_base]; omega)
      (by dsimp only; simp only [Region.hi, hds1_base, hds1_len]; omega) hwr2'
  have hpos2 : ∀ u ∈ m2, 0 < u.data.length := len_pos_wrMem hpos1 hwr2'
  let dsReg2 : Region :=
    { base := dsBase, data := writeBytes dsReg1.data (curLen * 16 + 8) (bytes8 outs[rem'].plen), rw := true }
  have hds2_eq : dsReg2 =
      { base := dsReg1.base, data := writeBytes dsReg1.data (dsBase + (curLen * 16 + 8) - dsReg1.base) (bytes8 outs[rem'].plen), rw := dsReg1.rw } := by
    dsimp [dsReg2]; rw [hds1_base, hds1_rw, show dsBase + (curLen * 16 + 8) - dsBase = curLen * 16 + 8 by omega]
  have hds2_mem' : dsReg2 ∈ s9.mem := by show dsReg2 ∈ m2; rw [hds2_eq]; exact hds2_mem
  have hds2_len : dsReg2.data.length = dsCap * 16 := by dsimp [dsReg2]; rw [writeBytes_length]; exact hds1_len
  have hds2_hi : dsReg2.hi ≤ 2 ^ 47 := by
    simp only [Region.hi, dsReg2, writeBytes_length]; rw [hds1_len, ← hds1_base]; exact hds1_hi'
  have hm2_oth' : ∀ u ∈ s7.mem, u.base ≠ dsBase → u ∈ s9.mem := fun u hu hnu => by
    show u ∈ m2
    exact hm2_oth u (hm1_oth u hu hnu) (by rw [hds1_base]; exact hnu)
  have hpds2_raw := pairListRel_push hpds outs[rem'] hvW.1 hvW.2 (by rw [hds_len]; omega)
  have hpds2 : PairListRel dsReg2.data (curLen + 1) (outs[rem'] :: curDs) := by
    dsimp only [dsReg2]; rw [hds1_data]; exact hpds2_raw
  exact ⟨dsReg2, s9, Reaches.step_trans h08 (Reaches.step h09), rfl, hk8, hcm2, hbl2, hpos2, hr8,
    hs8_15, hs8_4, hs8_12, hs8_13, hs8_3, rfl, rfl, hds2_len, hds2_hi, hds2_mem', hm2_oth', hpds2⟩

private theorem epilogue_push_store_spec {rem' : Nat} {s7 : X86Vm.St}
    {sp ctxBase dsBase dsCap curLen : Nat} {dsReg : Region} {outs curDs : List SporosProvLen.LVal}
    (hrip7 : s7.rip = 4195363) (hcm : CodeMem s7.mem) (hbl : BytesLt s7.mem)
    (hpos : ∀ u ∈ s7.mem, 0 < u.data.length) (hr7 : RegsOk s7)
    (hs7_15 : s7.reg 15 = some ctxBase) (hs7_4 : s7.reg 4 = some sp)
    (hs7_12 : s7.reg 12 = some dsBase) (hs7_13 : s7.reg 13 = some curLen)
    (hs7_3 : s7.reg 3 = some rem') (hs7_6 : s7.reg 6 = some (curLen * 16))
    (hv_lt : rem' < outs.length)
    (hs7_0 : s7.reg 0 = some outs[rem'].w) (hs7_1 : s7.reg 1 = some outs[rem'].plen)
    (hvW : outs[rem'].w < X86Env.W ∧ outs[rem'].plen < X86Env.W)
    (hds_mem : dsReg ∈ s7.mem) (hds_base : dsReg.base = dsBase) (hds_rw : dsReg.rw = true)
    (hds_len : dsReg.data.length = dsCap * 16) (hds_hi : dsReg.hi ≤ 2 ^ 47)
    (hcap_bound : curLen + (rem' + 1) ≤ dsCap) (hpds : PairListRel dsReg.data curLen curDs) :
    ∃ dsReg2 s11,
      Reaches vmImage s7 s11 ∧ s11.rip = 4195379 ∧ s11.k = s7.k ∧
      s11.fl = (X86Int.aluF .and rem' rem').2 ∧
      CodeMem s11.mem ∧ BytesLt s11.mem ∧ (∀ u ∈ s11.mem, 0 < u.data.length) ∧ RegsOk s11 ∧
      s11.reg 15 = some ctxBase ∧ s11.reg 4 = some sp ∧ s11.reg 12 = some dsBase ∧
      s11.reg 13 = some (curLen + 1) ∧ s11.reg 3 = some rem' ∧
      dsReg2.base = dsBase ∧ dsReg2.rw = true ∧ dsReg2.data.length = dsCap * 16 ∧ dsReg2.hi ≤ 2 ^ 47 ∧
      dsReg2 ∈ s11.mem ∧ (∀ u ∈ s7.mem, u.base ≠ dsBase → u ∈ s11.mem) ∧
      PairListRel dsReg2.data (curLen + 1) (outs[rem'] :: curDs) := by
  obtain ⟨dsReg2, s9, hrch9, hrip9, hk9, hcm2, hbl2, hpos2, hr9, hs9_15, hs9_4, hs9_12, hs9_13, hs9_3,
    hds2_base, hds2_rw, hds2_len, hds2_hi, hds2_mem, hm2_oth, hpds2⟩ :=
    epilogue_push_store_mem_spec hrip7 hcm hbl hpos hr7 hs7_15 hs7_4 hs7_12 hs7_13 hs7_3 hs7_6
      hv_lt hs7_0 hs7_1 hvW hds_mem hds_base hds_rw hds_len hds_hi hcap_bound hpds
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hds_hi' : dsReg.base + dsCap * 16 ≤ 2 ^ 47 := by simpa only [Region.hi, hds_len] using hds_hi
  have hcurW : curLen + rem' + 1 < X86Env.W := by omega
  have hadd1 : (X86Int.aluF .add curLen (1 % X86Env.W)).1 = curLen + 1 := by
    rw [Nat.mod_eq_of_lt (by decide), aluF_add_val]; exact Nat.mod_eq_of_lt (by omega)
  have hadd1_pair : X86Int.aluF .add curLen (1 % X86Env.W) =
      (curLen + 1, (X86Int.aluF .add curLen (1 % X86Env.W)).2) := Prod.ext hadd1 rfl
  have h10 : X86Vm.step vmImage s9 =
      .ok { s9.set 13 (some (curLen + 1)) with fl := (X86Int.aluF .add curLen (1 % X86Env.W)).2, rip := 4195376 } := by
    rw [step_at_chunk 3 hrip9 rfl (by decide : (4195372, (.ofInt (.aluI .add 13 1)), 4) ∈ SporosVmBin.vmTable_3),
      exec_aluI hs9_13, hadd1_pair, hrip9]
  let s10 := { s9.set 13 (some (curLen + 1)) with fl := (X86Int.aluF .add curLen (1 % X86Env.W)).2, rip := 4195376 }
  have hr10 : RegsOk s10 := regsOk_set hr9 (by decide) (curLen + 1)
  have hs10_3 : s10.reg 3 = some rem' := by
    change (s9.set 13 (some (curLen + 1))).reg 3 = some rem'; rw [reg_set_ne (by decide)]; exact hs9_3
  have h11 : X86Vm.step vmImage s10 =
      .ok { s10 with fl := (X86Int.aluF .and rem' rem').2, rip := 4195379 } := by
    rw [step_at_chunk 3 rfl rfl (by decide : (4195376, (.ofInt (.test 3 3)), 3) ∈ SporosVmBin.vmTable_3),
      exec_test hs10_3 hs10_3]
  let s11 := { s10 with fl := (X86Int.aluF .and rem' rem').2, rip := 4195379 }
  have hs11_15 : s11.reg 15 = some ctxBase := by
    change (s9.set 13 (some (curLen + 1))).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs9_15
  have hs11_4 : s11.reg 4 = some sp := by
    change (s9.set 13 (some (curLen + 1))).reg 4 = some sp; rw [reg_set_ne (by decide)]; exact hs9_4
  have hs11_12 : s11.reg 12 = some dsBase := by
    change (s9.set 13 (some (curLen + 1))).reg 12 = some dsBase; rw [reg_set_ne (by decide)]; exact hs9_12
  have hs11_13 : s11.reg 13 = some (curLen + 1) := reg_set_self_ok hr9 (by decide) (some (curLen + 1))
  exact ⟨dsReg2, s11, Reaches.trans hrch9 (Reaches.step_trans h10 (Reaches.step h11)),
    rfl, hk9, rfl, hcm2, hbl2, hpos2, hr10, hs11_15, hs11_4, hs11_12, hs11_13, hs10_3,
    hds2_base, hds2_rw, hds2_len, hds2_hi, hds2_mem, hm2_oth, hpds2⟩

theorem epilogue_push_loop_spec :
    ∀ (rem : Nat) {s : X86Vm.St} {sp ctxBase dsBase dsCap curLen : Nat}
      {ctx dsReg : Region} {outs curDs : List SporosProvLen.LVal},
    s.rip = 4195329 → CodeMem s.mem → BytesLt s.mem → (∀ u ∈ s.mem, 0 < u.data.length) → RegsOk s →
    s.reg 15 = some ctxBase → s.reg 4 = some sp →
    s.reg 12 = some dsBase → s.reg 13 = some curLen → s.reg 3 = some rem →
    0 < rem → rem ≤ outs.length → outs.length ≤ 3 →
    (∀ v ∈ outs, v.w < X86Env.W ∧ v.plen < X86Env.W) →
    ctx ∈ s.mem → ctx.base = ctxBase → ctx.rw = true →
    ctxSize ≤ ctx.data.length → ctxBase + ctxSize ≤ X86Env.W →
    (∀ (i : Nat) (_ : i < outs.length),
      leVal ((ctx.data.drop (360 + i * 16)).take 8) = outs[i].w ∧
      leVal ((ctx.data.drop (368 + i * 16)).take 8) = outs[i].plen) →
    dsReg ∈ s.mem → dsReg.base = dsBase → dsReg.rw = true →
    dsReg.data.length = dsCap * 16 → dsReg.hi ≤ 2 ^ 47 →
    dsBase ≠ ctxBase → curLen + rem ≤ dsCap → PairListRel dsReg.data curLen curDs →
    ∃ dsReg' s',
      Reaches vmImage s s' ∧ s'.rip = 4195381 ∧ s'.k = s.k ∧
      CodeMem s'.mem ∧ BytesLt s'.mem ∧ (∀ u ∈ s'.mem, 0 < u.data.length) ∧ RegsOk s' ∧
      s'.reg 15 = some ctxBase ∧ s'.reg 4 = some sp ∧ s'.reg 13 = some (curLen + rem) ∧
      dsReg'.base = dsBase ∧ dsReg'.rw = true ∧ dsReg'.data.length = dsCap * 16 ∧ dsReg'.hi ≤ 2 ^ 47 ∧
      dsReg' ∈ s'.mem ∧ (∀ u ∈ s.mem, u.base ≠ dsBase → u ∈ s'.mem) ∧
      PairListRel dsReg'.data (curLen + rem) (outs.take rem ++ curDs)
  | 0, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, hrem_pos, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ =>
      absurd hrem_pos (Nat.lt_irrefl 0)
  | rem' + 1, s, sp, ctxBase, dsBase, dsCap, curLen, ctx, dsReg, outs, curDs,
    hrip, hcm, hbl, hpos, hreg, hs15, hs4, hs12, hs13, hs3,
    _hrem_pos, hrem_le, houts_le, houtsW, hctx_mem, hctx_base, hctx_rw, hctx_len, hctx_W, hctx_slots,
    hds_mem, hds_base, hds_rw, hds_len, hds_hi, hne, hcap_bound, hpds => by
      have hv_lt : rem' < outs.length := Nat.lt_of_lt_of_le (Nat.lt_succ_self rem') hrem_le
      have hvW := houtsW outs[rem'] (List.getElem_mem hv_lt)
      obtain ⟨s7, hrch7, hrip7, hmem7, hk7, hr7, hs7_15, hs7_4, hs7_12, hs7_13, hs7_3, hs7_6, hs7_0, hs7_1⟩ :=
        epilogue_push_load_spec hrip hcm hreg hs15 hs4 hs12 hs13 hs3 hrem_le houts_le
          hctx_mem hctx_base hctx_len hctx_W hctx_slots hds_len hds_hi hcap_bound
      obtain ⟨dsReg2, s11, hrch11, hrip11, hk11, hfl11, hcm2, hbl2, hpos2, hr10,
        hs11_15, hs11_4, hs11_12, hs11_13, hs11_3, hds2_base, hds2_rw, hds2_len, hds2_hi, hds2_mem, hm2_oth, hpds2⟩ :=
        epilogue_push_store_spec hrip7 (hmem7.symm ▸ hcm) (hmem7.symm ▸ hbl) (hmem7.symm ▸ hpos) hr7
          hs7_15 hs7_4 hs7_12 hs7_13 hs7_3 hs7_6 hv_lt hs7_0 hs7_1 hvW
          (hmem7.symm ▸ hds_mem) hds_base hds_rw hds_len hds_hi hcap_bound hpds
      have hrch_tot11 : Reaches vmImage s s11 := Reaches.trans hrch7 hrch11
      have hk11' : s11.k = s.k := hk11.trans hk7
      have hm2_oth' : ∀ u ∈ s.mem, u.base ≠ dsBase → u ∈ s11.mem := fun u hu hnu => hm2_oth u (hmem7.symm ▸ hu) hnu
      cases hrem' : rem' with
      | zero =>
        have hcond11 : X86Int.cond s11.fl 5 = some false := by
          rw [hfl11, hrem', cond_aluF_and_self_ne 0]; rfl
        have h12 : X86Vm.step vmImage s11 = .ok { s11 with rip := 4195381 } := by
          rw [step_at_chunk 3 hrip11 rfl (by decide : (4195379, (.ofInt (.jcc 5 4195329)), 2) ∈ SporosVmBin.vmTable_3),
            exec_jcc hcond11, hrip11]
          rfl
        let s12 := { s11 with rip := 4195381 }
        refine ⟨dsReg2, s12, Reaches.trans hrch_tot11 (Reaches.step h12), rfl, hk11', hcm2, hbl2, hpos2, hr10,
          hs11_15, hs11_4, hs11_13, hds2_base, hds2_rw, hds2_len, hds2_hi, hds2_mem, hm2_oth', ?_⟩
        · have htake1 : outs.take 1 = [outs[0]] := by
            rw [show (1 : Nat) = 0 + 1 by rfl, List.take_add_one, List.take_zero,
              List.getElem?_eq_getElem (by omega)]
            rfl
          rw [show curLen + (0 + 1) = curLen + 1 by omega, show (0 + 1 : Nat) = 1 by rfl, htake1,
            List.singleton_append]
          simpa [hrem'] using hpds2
      | succ k =>
        have hcond11 : X86Int.cond s11.fl 5 = some true := by
          rw [hfl11, hrem', cond_aluF_and_self_ne (k + 1), decide_eq_true (Nat.succ_ne_zero k)]
        have h12 : X86Vm.step vmImage s11 = .ok { s11 with rip := 4195329 } := by
          rw [step_at_chunk 3 hrip11 rfl (by decide : (4195379, (.ofInt (.jcc 5 4195329)), 2) ∈ SporosVmBin.vmTable_3),
            exec_jcc hcond11]
          rfl
        let s12 := { s11 with rip := 4195329 }
        have hs12_3 : s12.reg 3 = some (k + 1) := by rw [← hrem']; exact hs11_3
        obtain ⟨dsReg', s', hrch', hrip', hk', hcm', hbl', hpos', hreg', hs'_15, hs'_4, hs'_13,
          hds'_base, hds'_rw, hds'_len, hds'_hi, hds'_mem, hoth', hpds'⟩ :=
          epilogue_push_loop_spec (k + 1) (s := s12) (curDs := outs[rem'] :: curDs)
            rfl hcm2 hbl2 hpos2 hr10
            hs11_15 hs11_4 hs11_12 hs11_13 hs12_3
            (Nat.zero_lt_succ k) (by omega) houts_le houtsW
            (hm2_oth' ctx hctx_mem (by rw [hctx_base]; exact hne.symm))
            hctx_base hctx_rw hctx_len hctx_W hctx_slots
            hds2_mem hds2_base hds2_rw hds2_len hds2_hi hne (by omega) hpds2
        have htake_eq : outs.take (k + 1) ++ outs[rem'] :: curDs = outs.take (k + 1 + 1) ++ curDs := by
          rw [show k + 1 = rem' from hrem'.symm, List.take_add_one (l := outs) (i := rem'),
            List.getElem?_eq_getElem hv_lt]
          dsimp [Option.toList]
          rw [List.append_assoc]
          rfl
        refine ⟨dsReg', s', Reaches.trans (Reaches.trans hrch_tot11 (Reaches.step h12)) hrch', hrip', hk'.trans hk11',
          hcm', hbl', hpos', hreg', hs'_15, hs'_4,
          (by rw [show curLen + (k + 1 + 1) = curLen + 1 + (k + 1) by omega]; exact hs'_13),
          hds'_base, hds'_rw, hds'_len, hds'_hi, hds'_mem,
          fun u hu hnu => hoth' u (hm2_oth' u hu hnu) hnu,
          (by rw [show curLen + (k + 1 + 1) = curLen + 1 + (k + 1) by omega, ← htake_eq]; exact hpds')⟩

private theorem epilogue_push_from_4195315 {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {outs : List SporosProvLen.LVal}
    {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = 4195315) (h : vc.wf pr ls s)
    (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (houts_pos : 0 < outs.length) (houts_le : outs.length ≤ 3)
    (houtsW : ∀ v ∈ outs, v.w < X86Env.W ∧ v.plen < X86Env.W)
    (hcap : ls.ds.length + outs.length ≤ vc.dsCap)
    (h288 : rd64 s.mem (ea ctxBase 288) = some outs.length)
    (hslots : ∀ (i : Nat) (_ : i < outs.length),
      rd64 s.mem (ea ctxBase (360 + i * 16)) = some outs[i].w ∧
      rd64 s.mem (ea ctxBase (368 + i * 16)) = some outs[i].plen) :
    let ls' := { ls with ds := outs ++ ls.ds, rs := { f with pc := f.pc + 1 } :: rest }
    ∃ s', Reaches vmImage s s' ∧ s'.k = s.k ∧ VmRel pr ls' s' := by
  have ⟨hcm, hbl, hpos, hreg, hsp, h15, hnodup, hmem, _, _, _, _,
    hctx1, hctx2, hctx3, hctx4, _, _, _, hds1, _, hds3, _, hds5, _⟩ := h
  have hpds := (VmCtx.arena_bounds h).1.2.2.2.2.2.1
  have hwlt_ds : ∀ v ∈ ls.ds, v.w < X86Env.W ∧ v.plen < X86Env.W := by
    have ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
      _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
      _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
      _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
      _, _, _, _, _, _, _, _, _, _, _, _, _, hwlt, _⟩ := h
    exact hwlt.1
  have hctxW : ctxBase + ctxSize ≤ X86Env.W := (vc.ctx_bounds h).2.2.2.2.1
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hctx3_n : 4096 ≤ vc.ctxReg.data.length := hctx3
  have hctxW_n : ctxBase + 4096 ≤ X86Env.W := hctxW
  have hds_hi_n : vc.dsReg.base + vc.dsReg.data.length ≤ 2 ^ 47 := hds5
  have hds_ne_ctx : vc.dsReg.base ≠ ctxBase := by
    simp only [VmCtx.regions, List.map_cons, List.map_nil] at hnodup
    intro heq; rw [← hctx1] at heq; rw [heq] at hnodup; simp [List.nodup_cons] at hnodup
  have h01 : X86Vm.step vmImage s = .ok { s.set 12 (some vc.dsReg.base) with rip := 4195318 } := by
    rw [step_at_chunk 3 hrip rfl (by decide : (4195315, (.ofInt (.ld 12 15 0)), 3) ∈ SporosVmBin.vmTable_3),
      exec_ld h15 (vc.rd_ctx_dsBase h), hrip]
  let s1 := { s.set 12 (some vc.dsReg.base) with rip := 4195318 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) vc.dsReg.base
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 12 (some vc.dsReg.base)).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hs1_len : rd64 s1.mem (ea ctxBase 8) = some ls.ds.length := vc.rd_ctx_dsLen (s := s) h
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 13 (some ls.ds.length) with rip := 4195322 } := by
    rw [step_at_chunk 3 rfl rfl (by decide : (4195318, (.ofInt (.ld 13 15 8)), 4) ∈ SporosVmBin.vmTable_3),
      exec_ld hs1_15 hs1_len]
  let s2 := { s1.set 13 (some ls.ds.length) with rip := 4195322 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) ls.ds.length
  have hs2_15 : s2.reg 15 = some ctxBase := by
    change (s1.set 13 (some ls.ds.length)).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15
  have hs2_288 : rd64 s2.mem (ea ctxBase 288) = some outs.length := h288
  have h03 : X86Vm.step vmImage s2 = .ok { s2.set 3 (some outs.length) with rip := 4195329 } := by
    rw [step_at_chunk 3 rfl rfl (by decide : (4195322, (.ofInt (.ld 3 15 288)), 7) ∈ SporosVmBin.vmTable_3),
      exec_ld hs2_15 hs2_288]
  let s3 := { s2.set 3 (some outs.length) with rip := 4195329 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) outs.length
  have hs3_15 : s3.reg 15 = some ctxBase := by
    change (s2.set 3 (some outs.length)).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs2_15
  have hs3_4 : s3.reg 4 = some vc.sp := by
    change (s2.set 3 (some outs.length)).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s1.set 13 (some ls.ds.length)).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s.set 12 (some vc.dsReg.base)).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hsp
  have hs3_12 : s3.reg 12 = some vc.dsReg.base := by
    change (s2.set 3 (some outs.length)).reg 12 = some vc.dsReg.base; rw [reg_set_ne (by decide)]
    change (s1.set 13 (some ls.ds.length)).reg 12 = some vc.dsReg.base; rw [reg_set_ne (by decide)]
    exact reg_set_self_ok hreg (by decide) (some vc.dsReg.base)
  have hs3_13 : s3.reg 13 = some ls.ds.length := by
    change (s2.set 3 (some outs.length)).reg 13 = some ls.ds.length; rw [reg_set_ne (by decide)]
    exact reg_set_self_ok hr1 (by decide) (some ls.ds.length)
  have hs3_3 : s3.reg 3 = some outs.length := reg_set_self_ok hr2 (by decide) (some outs.length)
  have hctx_slots : ∀ (i : Nat) (_ : i < outs.length),
      leVal ((vc.ctxReg.data.drop (360 + i * 16)).take 8) = outs[i].w ∧
      leVal ((vc.ctxReg.data.drop (368 + i * 16)).take 8) = outs[i].plen := by
    intro i hi
    have hsi := hslots i hi
    exact ⟨ctx_leVal_of_rd64 h (by dsimp only [ctxSize]; omega) hsi.1,
      ctx_leVal_of_rd64 h (by dsimp only [ctxSize]; omega) hsi.2⟩
  have hs3_ctx : vc.ctxReg ∈ s3.mem := vc.mem_ctx (s := s) h
  have hs3_ds : vc.dsReg ∈ s3.mem := hmem vc.dsReg (by simp [VmCtx.regions])
  obtain ⟨dsReg', s4, hrch4, hrip4, hk4, hcm4, hbl4, hpos4, hr4, hs4_15, hs4_4, hs4_13,
    hds'_base, hds'_rw, hds'_len, hds'_hi, hds'_mem, hoth4, hpds4⟩ :=
    epilogue_push_loop_spec outs.length (s := s3) rfl hcm hbl hpos hr3
      hs3_15 hs3_4 hs3_12 hs3_13 hs3_3 houts_pos (Nat.le_refl _) houts_le houtsW
      hs3_ctx hctx1 hctx2 hctx3 hctxW hctx_slots
      hs3_ds rfl hds1 hds3 hds5 hds_ne_ctx hcap hpds
  have hctx_mem4 : vc.ctxReg ∈ s4.mem := hoth4 vc.ctxReg (vc.mem_ctx (s := s) h) (by rw [hctx1]; exact hds_ne_ctx.symm)
  have hnewLen_eq : (outs ++ ls.ds).length = ls.ds.length + outs.length := by
    rw [List.length_append]; omega
  obtain ⟨_i5, m5, hwr5, _hm5_idx, _hm5_set, _hm5_wf, hctx5_mem, hm5_oth⟩ :=
    wr64_region hcm4.2 hctx_mem4 hctx2 (by rw [hctx1]; exact Nat.le_add_right ctxBase 8)
      (by simp only [Region.hi, hctx1]; exact Nat.add_le_add_left (Nat.le_trans (by decide : 16 ≤ 4096) hctx3_n) ctxBase)
      (a := ctxBase + 8) (v := ls.ds.length + outs.length)
  have hwr5' : wr64 s4.mem (ea ctxBase 8) (ls.ds.length + outs.length) = some m5 := by
    rw [ea_add_eq (by omega)]; exact hwr5
  have h05 : X86Vm.step vmImage s4 = .ok { s4 with rip := 4195385, mem := m5 } := by
    rw [step_at_chunk 3 hrip4 rfl (by decide : (4195381, (.ofInt (.st 15 8 13)), 4) ∈ SporosVmBin.vmTable_3),
      exec_st hs4_15 hs4_13 hwr5', hrip4]
  let s5 := { s4 with rip := 4195385, mem := m5 }
  have hcm5 : CodeMem m5 := hcm4.wr64_preserve hwr5'
  have hbl5 : BytesLt m5 := bytesLt_wr64 hbl4 hwr5'
  have hpos5 : ∀ u ∈ m5, 0 < u.data.length := len_pos_wrN hpos4 hwr5'
  have hctx5_mem' : { vc.ctxReg with data := writeBytes vc.ctxReg.data 8 (bytes8 (outs ++ ls.ds).length) } ∈ m5 := by
    simpa only [hctx1, show ctxBase + 8 - ctxBase = 8 by omega, hnewLen_eq] using hctx5_mem
  have hds5_mem' : dsReg' ∈ m5 := hm5_oth dsReg' hds'_mem (by rw [hds'_base, hctx1]; exact hds_ne_ctx)
  have hoth5 : ∀ u ∈ s.mem, u.base ≠ vc.dsReg.base → u.base ≠ ctxBase → u ∈ m5 :=
    fun u hu hnds hnctx => hm5_oth u (hoth4 u hu hnds) (by rw [← hctx1] at hnctx; exact hnctx)
  have hwlt_ds' : ∀ v ∈ outs ++ ls.ds, v.w < X86Env.W ∧ v.plen < X86Env.W := by
    intro v hv
    rcases List.mem_append.mp hv with hv | hv
    · exact houtsW v hv
    · exact hwlt_ds v hv
  have hpds5 : PairListRel dsReg'.data (outs ++ ls.ds).length (outs ++ ls.ds) := by
    rw [hnewLen_eq]; simpa only [List.take_length] using hpds4
  have hwf5 := wf_epilogue_finish hcm5 hbl5 hpos5 (s' := s5) rfl hr4 hs4_4 hs4_15 h
    hds'_base hds'_rw hds'_len hds'_hi hpds5 (by rw [hnewLen_eq]; exact hcap)
    hwlt_ds' hds5_mem' hctx5_mem' hoth5
  obtain ⟨s6, hrch6, hrip6, hk6, hwf6, hrel6⟩ :=
    L_adv_pc_spec (s := s5) rfl hwf5 hrs hpcW
  exact ⟨s6, Reaches.trans (Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03
    (Reaches.trans hrch4 (Reaches.step h05))))) hrch6, hk6.trans hk4, hrel6⟩

private theorem epilogue_grow_setup_spec {vc4 : VmCtx} {pr : Sporos.Prog} {ls4 : SporosProvLen.LSt}
    {s12 : X86Vm.St}
    (hrip12 : s12.rip = 4195278) (hwf12 : vc4.wf pr ls4 s12)
    (hs12_1 : s12.reg 1 = some vc4.dsCap) :
    ∃ s17,
      Reaches vmImage s12 s17 ∧ s17.rip = 4195295 ∧ s17.mem = s12.mem ∧ s17.k = s12.k ∧
      RegsOk s17 ∧ vc4.wf pr ls4 s17 ∧
      s17.reg 7 = some vc4.dsReg.base ∧
      s17.reg 6 = some (vc4.dsCap * 16) ∧
      s17.reg 2 = some (vc4.dsCap * 32) := by
  have hr12 : RegsOk s12 := hwf12.2.2.2.1
  have h12_4 : s12.reg 4 = some vc4.sp := hwf12.2.2.2.2.1
  have h12_15 : s12.reg 15 = some ctxBase := hwf12.2.2.2.2.2.1
  have hab := VmCtx.arena_bounds hwf12
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hdsLen_eq : vc4.dsReg.data.length = vc4.dsCap * 16 := hab.1.2.2.1
  have hdsHi47 : vc4.dsReg.hi ≤ 2 ^ 47 := hab.1.2.2.2.2.1
  have hds_hi_n : vc4.dsReg.base + vc4.dsReg.data.length ≤ 2 ^ 47 := hdsHi47
  have hs12_base : rd64 s12.mem (ea ctxBase 0) = some vc4.dsReg.base := vc4.rd_ctx_dsBase hwf12
  have h13 : X86Vm.step vmImage s12 = .ok { s12.set 7 (some vc4.dsReg.base) with rip := 4195281 } := by
    rw [step_at_chunk 2 hrip12 rfl (by decide : (4195278, (.ofInt (.ld 7 15 0)), 3) ∈ SporosVmBin.vmTable_2),
      exec_ld h12_15 hs12_base, hrip12]
  let s13 := { s12.set 7 (some vc4.dsReg.base) with rip := 4195281 }
  have hr13 : RegsOk s13 := regsOk_set hr12 (by decide) vc4.dsReg.base
  have hs13_1 : s13.reg 1 = some vc4.dsCap := by
    change (s12.set 7 (some vc4.dsReg.base)).reg 1 = some vc4.dsCap
    rw [reg_set_ne (by decide)]; exact hs12_1
  have h14 : X86Vm.step vmImage s13 = .ok { s13.set 6 (some vc4.dsCap) with rip := 4195284 } := by
    rw [step_at_chunk 2 rfl rfl (by decide : (4195281, (.ofInt (.mov 6 1)), 3) ∈ SporosVmBin.vmTable_2),
      exec_mov hs13_1]
  let s14 := { s13.set 6 (some vc4.dsCap) with rip := 4195284 }
  have hr14 : RegsOk s14 := regsOk_set hr13 (by decide) vc4.dsCap
  have hs14_6 : s14.reg 6 = some vc4.dsCap := reg_set_self_ok hr13 (by decide) (some vc4.dsCap)
  have hshl14 : vc4.dsCap * 2 ^ (4 % 64) % X86Env.W = vc4.dsCap * 16 := by
    rw [show 2 ^ (4 % 64) = 16 by decide, Nat.mod_eq_of_lt (by omega)]
  have h15_s : X86Vm.step vmImage s14 =
      .ok { s14.set 6 (some (vc4.dsCap * 16)) with fl := X86Int.Flags.undef, rip := 4195288 } := by
    rw [step_at_chunk 2 rfl rfl (by decide : (4195284, (.ofInt (.shlI 6 4)), 4) ∈ SporosVmBin.vmTable_2),
      exec_shlI hs14_6, hshl14]
  let s15 := { s14.set 6 (some (vc4.dsCap * 16)) with fl := X86Int.Flags.undef, rip := 4195288 }
  have hr15 : RegsOk s15 := regsOk_set hr14 (by decide) (vc4.dsCap * 16)
  have hs15_6 : s15.reg 6 = some (vc4.dsCap * 16) := reg_set_self_ok hr14 (by decide) (some (vc4.dsCap * 16))
  have h16 : X86Vm.step vmImage s15 = .ok { s15.set 2 (some (vc4.dsCap * 16)) with rip := 4195291 } := by
    rw [step_at_chunk 3 rfl rfl (by decide : (4195288, (.ofInt (.mov 2 6)), 3) ∈ SporosVmBin.vmTable_3),
      exec_mov hs15_6]
  let s16 := { s15.set 2 (some (vc4.dsCap * 16)) with rip := 4195291 }
  have hr16 : RegsOk s16 := regsOk_set hr15 (by decide) (vc4.dsCap * 16)
  have hs16_2 : s16.reg 2 = some (vc4.dsCap * 16) := reg_set_self_ok hr15 (by decide) (some (vc4.dsCap * 16))
  have hshl16 : (vc4.dsCap * 16) * 2 ^ (1 % 64) % X86Env.W = vc4.dsCap * 32 := by
    rw [show 2 ^ (1 % 64) = 2 by decide, show (vc4.dsCap * 16) * 2 = vc4.dsCap * 32 by omega,
      Nat.mod_eq_of_lt (by omega)]
  have h17 : X86Vm.step vmImage s16 =
      .ok { s16.set 2 (some (vc4.dsCap * 32)) with fl := X86Int.Flags.undef, rip := 4195295 } := by
    rw [step_at_chunk 3 rfl rfl (by decide : (4195291, (.ofInt (.shlI 2 1)), 4) ∈ SporosVmBin.vmTable_3),
      exec_shlI hs16_2, hshl16]
  let s17 := { s16.set 2 (some (vc4.dsCap * 32)) with fl := X86Int.Flags.undef, rip := 4195295 }
  have hr17 : RegsOk s17 := regsOk_set hr16 (by decide) (vc4.dsCap * 32)
  have hs17_4 : s17.reg 4 = some vc4.sp := by
    change (s16.set 2 (some (vc4.dsCap * 32))).reg 4 = some vc4.sp; rw [reg_set_ne (by decide)]
    change (s15.set 2 (some (vc4.dsCap * 16))).reg 4 = some vc4.sp; rw [reg_set_ne (by decide)]
    change (s14.set 6 (some (vc4.dsCap * 16))).reg 4 = some vc4.sp; rw [reg_set_ne (by decide)]
    change (s13.set 6 (some vc4.dsCap)).reg 4 = some vc4.sp; rw [reg_set_ne (by decide)]
    change (s12.set 7 (some vc4.dsReg.base)).reg 4 = some vc4.sp; rw [reg_set_ne (by decide)]
    exact h12_4
  have hs17_15 : s17.reg 15 = some ctxBase := by
    change (s16.set 2 (some (vc4.dsCap * 32))).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s15.set 2 (some (vc4.dsCap * 16))).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s14.set 6 (some (vc4.dsCap * 16))).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s13.set 6 (some vc4.dsCap)).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s12.set 7 (some vc4.dsReg.base)).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    exact h12_15
  have hs17_7 : s17.reg 7 = some vc4.dsReg.base := by
    change (s16.set 2 (some (vc4.dsCap * 32))).reg 7 = some vc4.dsReg.base; rw [reg_set_ne (by decide)]
    change (s15.set 2 (some (vc4.dsCap * 16))).reg 7 = some vc4.dsReg.base; rw [reg_set_ne (by decide)]
    change (s14.set 6 (some (vc4.dsCap * 16))).reg 7 = some vc4.dsReg.base; rw [reg_set_ne (by decide)]
    change (s13.set 6 (some vc4.dsCap)).reg 7 = some vc4.dsReg.base; rw [reg_set_ne (by decide)]
    exact reg_set_self_ok hr12 (by decide) (some vc4.dsReg.base)
  have hs17_6 : s17.reg 6 = some (vc4.dsCap * 16) := by
    change (s16.set 2 (some (vc4.dsCap * 32))).reg 6 = some (vc4.dsCap * 16); rw [reg_set_ne (by decide)]
    change (s15.set 2 (some (vc4.dsCap * 16))).reg 6 = some (vc4.dsCap * 16); rw [reg_set_ne (by decide)]
    exact reg_set_self_ok hr14 (by decide) (some (vc4.dsCap * 16))
  have hs17_2 : s17.reg 2 = some (vc4.dsCap * 32) :=
    reg_set_self_ok hr16 (by decide) (some (vc4.dsCap * 32))
  exact ⟨s17, Reaches.step_trans h13 (Reaches.step_trans h14 (Reaches.step_trans h15_s (Reaches.step_trans h16 (Reaches.step h17)))),
    rfl, rfl, rfl, hr17, vc4.wf_step_misc hwf12 rfl hr17 hs17_4 hs17_15, hs17_7, hs17_6, hs17_2⟩

private theorem epilogue_grow_call_spec {vc4 : VmCtx} {pr : Sporos.Prog} {ls4 : SporosProvLen.LSt}
    {s17 : X86Vm.St}
    (hrip17 : s17.rip = 4195295) (hwf17 : vc4.wf pr ls4 s17) (hor17 : OracleAdmits s17)
    (hs17_7 : s17.reg 7 = some vc4.dsReg.base)
    (hs17_6 : s17.reg 6 = some (vc4.dsCap * 16))
    (hs17_2 : s17.reg 2 = some (vc4.dsCap * 32)) :
    (∃ (a : Nat) (dsReg' : Region) (s19 : X86Vm.St) (stk' : Region),
      Reaches vmImage s17 s19 ∧ s19.rip = 4195300 ∧ s19.k.out = s17.k.out ∧ OracleAdmits s19 ∧
      CodeMem s19.mem ∧ BytesLt s19.mem ∧ (∀ u ∈ s19.mem, 0 < u.data.length) ∧ RegsOk s19 ∧
      s19.reg 0 = some a ∧ s19.reg 4 = some vc4.sp ∧ s19.reg 15 = some ctxBase ∧
      dsReg' = { base := a, data := vc4.dsReg.data ++ List.replicate (vc4.dsCap * 16) 0, rw := true } ∧
      dsReg' ∈ s19.mem ∧
      stk'.base = vc4.stk.base ∧ stk'.rw = vc4.stk.rw ∧ stk'.data.length = vc4.stk.data.length ∧
      stk' ∈ s19.mem ∧ vc4.ctxReg ∈ s19.mem ∧
      (∀ u ∈ s17.mem, u.base ≠ vc4.stk.base → u.base ≠ vc4.dsReg.base → u ∈ s19.mem) ∧
      a + vc4.dsCap * 32 ≤ 2 ^ 47 ∧
      (a ≠ vc4.stk.base ∧ ∀ u ∈ s17.mem, u.base ≠ vc4.dsReg.base → a + vc4.dsCap * 32 ≤ u.base ∨ u.hi ≤ a)) ∨
    (HitsNomem s17 ∧ Ends vmImage s17 (.exit 2)) := by
  have ⟨hcm17, hbl17, hpos17, hr17, hs17_4, hs17_15, hnodup, _,
    hstk1, hstk2, hstk3, hstk4, hctx1, _, _, _⟩ := hwf17
  have hab := VmCtx.arena_bounds hwf17
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hpg : X86Env.page = 4096 := rfl
  have hdsLen_eq : vc4.dsReg.data.length = vc4.dsCap * 16 := hab.1.2.2.1
  have hdsHi47 : vc4.dsReg.hi ≤ 2 ^ 47 := hab.1.2.2.2.2.1
  have hds_hi_n : vc4.dsReg.base + vc4.dsCap * 16 ≤ 2 ^ 47 := by
    simpa only [Region.hi, hdsLen_eq] using hdsHi47
  have hdsCap_ge4 : 4 ≤ vc4.dsCap := hab.1.2.1
  have hdsPg : vc4.dsCap * 16 % X86Env.page = 0 := hab.1.2.2.2.1
  have hdsCapW : vc4.dsCap < X86Env.W := by omega
  have hsp_sub8 : (vc4.sp + X86Env.W - 8) % X86Env.W = vc4.sp - 8 := by
    rw [show vc4.sp + X86Env.W - 8 = (vc4.sp - 8) + X86Env.W by omega, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have hstk_lo : vc4.stk.base ≤ vc4.sp - 8 := by omega
  have hstk_hi : vc4.sp - 8 + 8 ≤ vc4.stk.hi := by simp only [Region.hi] at hstk3 ⊢; omega
  obtain ⟨_i_stk, m_call, hwr_stk, _hstk_idx, _hstk_set, hm_call_wf, hstk'_mem, hcall_oth⟩ :=
    wr64_region hcm17.2 (vc4.mem_stk hwf17) hstk1 hstk_lo hstk_hi (a := vc4.sp - 8) (v := 4195300)
  have hwr_stk' : wr64 s17.mem ((vc4.sp + X86Env.W - 8) % X86Env.W) (4195295 + 5) = some m_call := by
    rw [hsp_sub8]; exact hwr_stk
  have hwr_stk'' : wr64 s17.mem ((vc4.sp + X86Env.W - 8) % X86Env.W) (s17.rip + 5) = some m_call := by
    rw [hrip17]; exact hwr_stk'
  have h18 : X86Vm.step vmImage s17 =
      .ok { s17.set 4 (some ((vc4.sp + X86Env.W - 8) % X86Env.W)) with rip := 4201363, mem := m_call } := by
    rw [step_at_chunk 3 hrip17 rfl (by decide : (4195295, (.ofInt (.call 4201363)), 5) ∈ SporosVmBin.vmTable_3),
      exec_call hs17_4 hwr_stk'']
  let s18 := { s17.set 4 (some ((vc4.sp + X86Env.W - 8) % X86Env.W)) with rip := 4201363, mem := m_call }
  have hr18 : RegsOk s18 := regsOk_set hr17 (by decide) ((vc4.sp + X86Env.W - 8) % X86Env.W)
  let stk' : Region :=
    { base := vc4.stk.base, data := writeBytes vc4.stk.data (vc4.sp - 8 - vc4.stk.base) (bytes8 4195300), rw := vc4.stk.rw }
  have hstk'_mem' : stk' ∈ m_call := hstk'_mem
  have hnodup0 : (List.map Region.base vc4.regions).Nodup := hnodup
  simp only [VmCtx.regions, List.map_cons, List.map_nil] at hnodup0
  have hstk_ne_ds : vc4.stk.base ≠ vc4.dsReg.base := by
    intro heq; rw [heq] at hnodup0; simp [List.nodup_cons] at hnodup0
  have hstk_ne_ctx : vc4.stk.base ≠ ctxBase := by
    intro heq; rw [← hctx1] at heq; rw [heq] at hnodup0; simp [List.nodup_cons] at hnodup0
  have hds_ne_ctx : vc4.dsReg.base ≠ ctxBase := by
    intro heq; rw [← hctx1] at heq; rw [heq] at hnodup0; simp [List.nodup_cons] at hnodup0
  have hds_mem_call : vc4.dsReg ∈ m_call := hcall_oth vc4.dsReg (vc4.mem_ds hwf17) hstk_ne_ds.symm
  obtain ⟨idx_ds, hidx_ds_lt, hidx_ds_eq⟩ := List.mem_iff_getElem.mp hds_mem_call
  have hidx_ds_get : m_call[idx_ds]? = some vc4.dsReg :=
    (List.getElem?_eq_getElem hidx_ds_lt).trans (by rw [hidx_ds_eq])
  have hfind_ds : m_call.find vc4.dsReg.base 0 = some (idx_ds, vc4.dsReg) :=
    mem_find_at_idx m_call idx_ds vc4.dsReg vc4.dsReg.base hm_call_wf hidx_ds_get (Nat.le_refl _)
      (by simp only [Region.hi]; omega)
  have hrd_ret : rd64 m_call (vc4.sp - 8) = some 4195300 := by
    have := rd64_of_writeBytes_same hm_call_wf hstk'_mem' (by omega)
      (by simp only [Region.hi] at hstk3 ⊢; omega) (v := 4195300)
    simpa [Nat.mod_eq_of_lt (by decide : 4195300 < X86Env.W)] using this
  have hs18_4 : s18.reg 4 = some (vc4.sp - 8) := by
    rw [← hsp_sub8]
    exact reg_set_self_ok hr17 (by decide) (some ((vc4.sp + X86Env.W - 8) % X86Env.W))
  have hs18_15 : s18.reg 15 = some ctxBase := by
    change (s17.set 4 (some ((vc4.sp + X86Env.W - 8) % X86Env.W))).reg 15 = some ctxBase
    rw [reg_set_ne (by decide)]; exact hs17_15
  have hs18_7 : s18.reg 7 = some vc4.dsReg.base := by
    change (s17.set 4 (some ((vc4.sp + X86Env.W - 8) % X86Env.W))).reg 7 = some vc4.dsReg.base
    rw [reg_set_ne (by decide)]; exact hs17_7
  have hs18_6 : s18.reg 6 = some (vc4.dsCap * 16) := by
    change (s17.set 4 (some ((vc4.sp + X86Env.W - 8) % X86Env.W))).reg 6 = some (vc4.dsCap * 16)
    rw [reg_set_ne (by decide)]; exact hs17_6
  have hs18_2 : s18.reg 2 = some (vc4.dsCap * 32) := by
    change (s17.set 4 (some ((vc4.sp + X86Env.W - 8) % X86Env.W))).reg 2 = some (vc4.dsCap * 32)
    rw [reg_set_ne (by decide)]; exact hs17_2
  have hpages_old : X86Env.pages (vc4.dsCap * 16) = vc4.dsCap * 16 := by
    dsimp [X86Env.pages]; rw [hpg] at hdsPg ⊢; omega
  have hpages_new : X86Env.pages (vc4.dsCap * 32) = vc4.dsCap * 32 := by
    dsimp [X86Env.pages]; rw [hpg] at hdsPg ⊢; omega
  have hmr := call_mremap_oracle_spec (s := s18) (sp := vc4.sp - 8) (oldPtr := vc4.dsReg.base)
    (oldLen := vc4.dsCap * 16) (newLen := vc4.dsCap * 32) (idx := idx_ds) (oldReg := vc4.dsReg) (stk := stk')
    rfl hr18 (oracleAdmits_of_reaches hor17 (Reaches.step h18)) hs18_4 (by omega) (by omega) hm_call_wf hstk'_mem'
    hstk1 hstk_ne_ds (by dsimp [stk']; omega) (by simp only [Region.hi, stk', writeBytes_length] at hstk3 ⊢; omega)
    hrd_ret hs18_7 hs18_6 hs18_2 hfind_ds rfl
    (by rw [hpages_old, hdsLen_eq]) hab.1.1 (by rw [hpages_new, hdsLen_eq]; omega)
  rcases hmr with ⟨a, s19, hrch19, hnext, _ha_pg, _ha_pos, ha_hi, ha_free, hrip19, hk19, hm19,
    hm19_wf, hr19, hs19_0, hs19_4, _hor19, hs19_oth⟩ | ⟨hhit, hends⟩
  · have hfree' : X86Env.Mem.free (List.eraseIdx m_call idx_ds) a (vc4.dsCap * 32) = true := by
      simpa only [hpages_new] using ha_free
    have ha_hi' : a + vc4.dsCap * 32 ≤ 2 ^ 47 := by simpa only [hpages_new] using ha_hi
    have hbl_call : BytesLt m_call := bytesLt_wr64 hbl17 hwr_stk'
    have ha_disj := mremap_find_after_stack_write hcm17.2 (vc4.mem_stk hwf17)
      (rfl : stk'.base = vc4.stk.base) (by simp only [Region.hi, stk', writeBytes_length])
      hstk_ne_ds hstk'_mem' hcall_oth hidx_ds_get rfl hfree' (by omega) hpos17
    let dsReg' : Region :=
      { base := a, data := vc4.dsReg.data ++ List.replicate (vc4.dsCap * 16) 0, rw := true }
    have hm19_eq : s19.mem = List.eraseIdx m_call idx_ds ++ [dsReg'] := by
      simpa only [dsReg', hpages_new, hdsLen_eq, show vc4.dsCap * 32 - vc4.dsCap * 16 = vc4.dsCap * 16 by omega] using hm19
    have hds'_mem19 : dsReg' ∈ s19.mem := by rw [hm19_eq]; simp
    have hm19_oth : ∀ u ∈ m_call, u.base ≠ vc4.dsReg.base → u ∈ s19.mem := fun u hu hnu => by
      rw [hm19_eq]; exact List.mem_append_left _ (mem_eraseIdx_of_ne_base m_call idx_ds hidx_ds_get hu hnu)
    have hctx4_mem19 : vc4.ctxReg ∈ s19.mem :=
      hm19_oth vc4.ctxReg (hcall_oth vc4.ctxReg (vc4.mem_ctx hwf17) (by rw [hctx1]; exact hstk_ne_ctx.symm))
        (by rw [hctx1]; exact hds_ne_ctx.symm)
    have hstk'_mem19 : stk' ∈ s19.mem := hm19_oth stk' hstk'_mem' hstk_ne_ds
    have hcm19 : CodeMem s19.mem := by
      refine ⟨?_, hm19_wf⟩
      have helf : ({ base := baseAddr, data := SporosVmBin.vmBytes, rw := false } : Region) ∈ s17.mem := hcm17.1
      have helf_ne_stk : baseAddr ≠ vc4.stk.base := by
        intro heq
        have heqf := mem_eq_of_same_base hcm17.2 hpos17 helf (vc4.mem_stk hwf17) heq
        have : (false : Bool) = true := by rw [← hstk1, ← heqf]
        exact Bool.false_ne_true this
      have helf_ne_ds : baseAddr ≠ vc4.dsReg.base := by
        intro heq
        have heqf := mem_eq_of_same_base hcm17.2 hpos17 helf (vc4.mem_ds hwf17) heq
        have : (false : Bool) = true := by rw [← hab.1.1, ← heqf]
        exact Bool.false_ne_true this
      exact hm19_oth _ (hcall_oth _ helf helf_ne_stk) helf_ne_ds
    have hbl19 : BytesLt s19.mem := by
      intro u hu b hb
      rw [hm19_eq, List.mem_append] at hu
      rcases hu with hu | hu
      · exact hbl_call u (eraseIdx_mem_subset m_call idx_ds hu) b hb
      · rw [List.mem_singleton] at hu; subst hu
        dsimp [dsReg'] at hb
        rcases List.mem_append.mp hb with hb | hb
        · exact hbl17 vc4.dsReg (vc4.mem_ds hwf17) b hb
        · exact bytesLt_replicate_zero _ b hb
    have hpos19 : ∀ u ∈ s19.mem, 0 < u.data.length := by
      intro u hu
      rw [hm19_eq, List.mem_append] at hu
      rcases hu with hu | hu
      · exact len_pos_wrN hpos17 hwr_stk' u (eraseIdx_mem_subset m_call idx_ds hu)
      · rw [List.mem_singleton] at hu; subst hu; dsimp [dsReg']; rw [List.length_append, hdsLen_eq]; omega
    have hs19_15 : s19.reg 15 = some ctxBase := by
      rw [hs19_oth 15 (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)]
      exact hs18_15
    have hs19_4' : s19.reg 4 = some vc4.sp := by
      have hs19_4_eq : (vc4.sp - 8 + 8) % X86Env.W = vc4.sp := by
        rw [show vc4.sp - 8 + 8 = vc4.sp by omega, Nat.mod_eq_of_lt hstk4]
      rw [hs19_4_eq] at hs19_4
      exact hs19_4
    have hk_out : s19.k.out = s17.k.out := by
      rw [hk19]
      cases hk : s18.k.oracle with
      | nil => simp [X86Env.Kern.next, hk] at hnext
      | cons c rest_k => simp only [X86Env.Kern.next, hk]; rfl
    have h18_s : X86Vm.step vmImage s17 = .ok s18 := h18
    have hrch19' : Reaches vmImage s17 s19 := Reaches.step_trans h18_s hrch19
    refine Or.inl ⟨a, dsReg', s19, stk', hrch19', hrip19, hk_out, oracleAdmits_of_reaches hor17 hrch19',
      hcm19, hbl19, hpos19, hr19, hs19_0, hs19_4', hs19_15, rfl, hds'_mem19, rfl, rfl, ?_, hstk'_mem19, hctx4_mem19,
      fun u hu hnstk hnds => hm19_oth u (hcall_oth u hu hnstk) hnds, ha_hi', ha_disj⟩
    · dsimp [stk']; exact writeBytes_length vc4.stk.data (vc4.sp - 8 - vc4.stk.base) (bytes8 4195300)
  · exact Or.inr ⟨hitsNomem_of_reaches (Reaches.step h18) hhit, Ends.of_reaches (Reaches.step h18) hends⟩

private theorem epilogue_post_mremap_spec {vc4 : VmCtx} {pr : Sporos.Prog} {ls4 : SporosProvLen.LSt}
    {s17 s19 : X86Vm.St} {a : Nat} {dsReg' stk' : Region} {outs : List SporosProvLen.LVal}
    {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hwf17 : vc4.wf pr ls4 s17)
    (hrip19 : s19.rip = 4195300) (hor19 : OracleAdmits s19)
    (hcm19 : CodeMem s19.mem) (hbl19 : BytesLt s19.mem) (hpos19 : ∀ u ∈ s19.mem, 0 < u.data.length)
    (hr19 : RegsOk s19) (hs19_0 : s19.reg 0 = some a)
    (hs19_4' : s19.reg 4 = some vc4.sp) (hs19_15 : s19.reg 15 = some ctxBase)
    (hds'_eq : dsReg' = { base := a, data := vc4.dsReg.data ++ List.replicate (vc4.dsCap * 16) 0, rw := true })
    (hds'_mem19 : dsReg' ∈ s19.mem)
    (hstk'_b : stk'.base = vc4.stk.base) (hstk'_rw : stk'.rw = vc4.stk.rw)
    (hstk'_len : stk'.data.length = vc4.stk.data.length)
    (hstk'_mem19 : stk' ∈ s19.mem) (hctx4_mem19 : vc4.ctxReg ∈ s19.mem)
    (hm19_from_s17 : ∀ u ∈ s17.mem, u.base ≠ vc4.stk.base → u.base ≠ vc4.dsReg.base → u ∈ s19.mem)
    (ha_hi' : a + vc4.dsCap * 32 ≤ 2 ^ 47)
    (ha_disj : a ≠ vc4.stk.base ∧ ∀ u ∈ s17.mem, u.base ≠ vc4.dsReg.base → a + vc4.dsCap * 32 ≤ u.base ∨ u.hi ≤ a)
    (hrs : ls4.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (houts_pos : 0 < outs.length) (heout : outs.length ≤ 3)
    (houtsW : ∀ v ∈ outs, v.w < X86Env.W ∧ v.plen < X86Env.W)
    (hcap_new : ls4.ds.length + outs.length ≤ 2 * vc4.dsCap)
    (h4_288 : rd64 s17.mem (ea ctxBase 288) = some outs.length)
    (h4_slots : ∀ (i : Nat) (_ : i < outs.length),
      rd64 s17.mem (ea ctxBase (360 + i * 16)) = some outs[i].w ∧
      rd64 s17.mem (ea ctxBase (368 + i * 16)) = some outs[i].plen) :
    ∃ s_fin,
      Reaches vmImage s19 s_fin ∧ s_fin.k.out = s19.k.out ∧ OracleAdmits s_fin ∧
      VmRel pr { ls4 with ds := outs ++ ls4.ds, rs := { f with pc := f.pc + 1 } :: rest } s_fin := by
  have ⟨_, _, _, _, _, _, hnodup, _, _, _, _, _, hctx1, hctx2, hctx3, hctx4, _⟩ := hwf17
  have hab := VmCtx.arena_bounds hwf17
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hctxW : ctxBase + ctxSize ≤ X86Env.W := (vc4.ctx_bounds hwf17).2.2.2.2.1
  have hctx3_n : 4096 ≤ vc4.ctxReg.data.length := hctx3
  have hctxW_n : ctxBase + 4096 ≤ X86Env.W := hctxW
  have hdsHi47 : vc4.dsReg.hi ≤ 2 ^ 47 := hab.1.2.2.2.2.1
  have hdsLen_eq : vc4.dsReg.data.length = vc4.dsCap * 16 := hab.1.2.2.1
  have hds_hi_n : vc4.dsReg.base + vc4.dsCap * 16 ≤ 2 ^ 47 := by
    simpa only [Region.hi, hdsLen_eq] using hdsHi47
  have hdsCap_2_W : 2 * vc4.dsCap < X86Env.W := by omega
  have hnodup0 : (List.map Region.base vc4.regions).Nodup := hnodup
  simp only [VmCtx.regions, List.map_cons, List.map_nil] at hnodup0
  have hstk_ne_ctx : vc4.stk.base ≠ ctxBase := by
    intro heq; rw [← hctx1] at heq; rw [heq] at hnodup0; simp [List.nodup_cons] at hnodup0
  have hds_ne_ctx : vc4.dsReg.base ≠ ctxBase := by
    intro heq; rw [← hctx1] at heq; rw [heq] at hnodup0; simp [List.nodup_cons] at hnodup0
  obtain ⟨_i20, m20, hwr20, _hm20_idx, _hm20_set, hm20_wf, hctx20_mem, hm20_oth⟩ :=
    wr64_region hcm19.2 hctx4_mem19 hctx2 (by rw [hctx1]; exact Nat.le_add_right ctxBase 0)
      (by simp only [Region.hi, hctx1]; exact Nat.add_le_add_left (Nat.le_trans (by decide : 8 ≤ 4096) hctx3_n) ctxBase)
      (a := ctxBase + 0) (v := a)
  have hwr20' : wr64 s19.mem (ea ctxBase 0) a = some m20 := by rw [ea_add_eq (by omega)]; exact hwr20
  have h20 : X86Vm.step vmImage s19 = .ok { s19 with rip := 4195303, mem := m20 } := by
    rw [step_at_chunk 3 hrip19 rfl (by decide : (4195300, (.ofInt (.st 15 0 0)), 3) ∈ SporosVmBin.vmTable_3),
      exec_st hs19_15 hs19_0 hwr20', hrip19]
  let s20 := { s19 with rip := 4195303, mem := m20 }
  let ctx20 := { vc4.ctxReg with data := writeBytes vc4.ctxReg.data 0 (bytes8 a) }
  have hctx20_mem' : ctx20 ∈ m20 := by
    simpa only [ctx20, hctx1, Nat.sub_self, Nat.add_zero] using hctx20_mem
  have hs20_15 : s20.reg 15 = some ctxBase := hs19_15
  have hrd20_16 : rd64 s20.mem (ea ctxBase 16) = some vc4.dsCap := by
    show rd64 m20 (ea ctxBase 16) = some vc4.dsCap
    rw [ea_add_eq (by omega), rd64, mem_rdN_of_mem m20 ctx20 (ctxBase + 16) 8 hm20_wf hctx20_mem'
      (by dsimp [ctx20]; rw [hctx1]; exact Nat.le_add_right ctxBase 16)
      (by simp only [Region.hi, ctx20, hctx1, writeBytes_length]
          exact Nat.add_le_add_left (Nat.le_trans (by decide : 24 ≤ 4096) hctx3_n) ctxBase)]
    dsimp only [Option.map, ctx20]
    rw [show ctxBase + 16 - vc4.ctxReg.base = 16 by rw [hctx1]; omega,
      writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; exact Or.inr (by decide))
        (Nat.le_trans (by decide : 24 ≤ 4096) hctx3_n),
      ctx_leVal_of_rd64 hwf17 (by decide) (vc4.rd_ctx_dsCap hwf17)]
  have h21 : X86Vm.step vmImage s20 = .ok { s20.set 1 (some vc4.dsCap) with rip := 4195307 } := by
    rw [step_at_chunk 3 rfl rfl (by decide : (4195303, (.ofInt (.ld 1 15 16)), 4) ∈ SporosVmBin.vmTable_3),
      exec_ld hs20_15 hrd20_16]
  let s21 := { s20.set 1 (some vc4.dsCap) with rip := 4195307 }
  have hr21 : RegsOk s21 := regsOk_set hr19 (by decide) vc4.dsCap
  have hs21_1 : s21.reg 1 = some vc4.dsCap := reg_set_self_ok hr19 (by decide) (some vc4.dsCap)
  have h2capW : vc4.dsCap * 2 ^ (1 % 64) % X86Env.W = 2 * vc4.dsCap := by
    rw [show 2 ^ (1 % 64) = 2 by decide, Nat.mul_comm vc4.dsCap 2, Nat.mod_eq_of_lt (by omega)]
  have h22 : X86Vm.step vmImage s21 =
      .ok { s21.set 1 (some (2 * vc4.dsCap)) with fl := X86Int.Flags.undef, rip := 4195311 } := by
    rw [step_at_chunk 3 rfl rfl (by decide : (4195307, (.ofInt (.shlI 1 1)), 4) ∈ SporosVmBin.vmTable_3),
      exec_shlI hs21_1, h2capW]
  let s22 := { s21.set 1 (some (2 * vc4.dsCap)) with fl := X86Int.Flags.undef, rip := 4195311 }
  have hr22 : RegsOk s22 := regsOk_set hr21 (by decide) (2 * vc4.dsCap)
  have hs22_15 : s22.reg 15 = some ctxBase := by
    change (s21.set 1 (some (2 * vc4.dsCap))).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s20.set 1 (some vc4.dsCap)).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs20_15
  have hs22_1 : s22.reg 1 = some (2 * vc4.dsCap) := reg_set_self_ok hr21 (by decide) (some (2 * vc4.dsCap))
  have hctx20_len : 4096 ≤ ctx20.data.length := by dsimp [ctx20]; rw [writeBytes_length]; exact hctx3_n
  obtain ⟨_i23, m23, hwr23, _hm23_idx, _hm23_set, _hm23_wf, hctx23_mem, hm23_oth⟩ :=
    wr64_region hm20_wf hctx20_mem' (by dsimp [ctx20]; exact hctx2)
      (by dsimp [ctx20]; rw [hctx1]; exact Nat.le_add_right ctxBase 16)
      (by simp only [Region.hi, ctx20, hctx1]
          exact Nat.add_le_add_left (Nat.le_trans (by decide : 24 ≤ 4096) hctx20_len) ctxBase)
      (a := ctxBase + 16) (v := 2 * vc4.dsCap)
  have hwr23' : wr64 s22.mem (ea ctxBase 16) (2 * vc4.dsCap) = some m23 := by rw [ea_add_eq (by omega)]; exact hwr23
  have h23 : X86Vm.step vmImage s22 = .ok { s22 with rip := 4195315, mem := m23 } := by
    rw [step_at_chunk 3 rfl rfl (by decide : (4195311, (.ofInt (.st 15 16 1)), 4) ∈ SporosVmBin.vmTable_3),
      exec_st hs22_15 hs22_1 hwr23']
  let s23 := { s22 with rip := 4195315, mem := m23 }
  have hs23_4 : s23.reg 4 = some vc4.sp := by
    change (s21.set 1 (some (2 * vc4.dsCap))).reg 4 = some vc4.sp; rw [reg_set_ne (by decide)]
    change (s20.set 1 (some vc4.dsCap)).reg 4 = some vc4.sp; rw [reg_set_ne (by decide)]; exact hs19_4'
  have hcm20 := hcm19.wr64_preserve hwr20'
  have hcm23 := hcm20.wr64_preserve hwr23'
  have hbl20 := bytesLt_wr64 hbl19 hwr20'
  have hbl23 := bytesLt_wr64 hbl20 hwr23'
  have hpos20 := len_pos_wrN hpos19 hwr20'
  have hpos23 := len_pos_wrN hpos20 hwr23'
  let ctx23 := { vc4.ctxReg with data := writeBytes (writeBytes vc4.ctxReg.data 0 (bytes8 a)) 16 (bytes8 (2 * vc4.dsCap)) }
  have hctx23_mem' : ctx23 ∈ m23 := by
    simpa only [ctx23, ctx20, hctx1, show ctxBase + 16 - ctxBase = 16 by omega] using hctx23_mem
  have hm23_from_19 : ∀ u ∈ s19.mem, u.base ≠ ctxBase → u ∈ m23 := fun u hu hnu =>
    hm23_oth u (hm20_oth u hu (by rw [hctx1]; exact hnu)) (by dsimp [ctx20]; rw [hctx1]; exact hnu)
  have ha_ne_ctx : a ≠ ctxBase := by
    have hcl4 : 0 < vc4.ctxReg.data.length := by omega
    have := ne_a_of_disj (by omega) hcl4 (ha_disj.2 vc4.ctxReg (vc4.mem_ctx hwf17) (by rw [hctx1]; exact hds_ne_ctx.symm))
    rw [hctx1] at this; exact this.symm
  have hm23_ds : { base := a, data := vc4.dsReg.data ++ List.replicate (vc4.dsCap * 16) 0, rw := true } ∈ m23 := by
    rw [← hds'_eq]; exact hm23_from_19 dsReg' hds'_mem19 (by rw [hds'_eq]; exact ha_ne_ctx)
  have hm23_stk : stk' ∈ m23 := hm23_from_19 stk' hstk'_mem19 (by rw [hstk'_b]; exact hstk_ne_ctx)
  have hwf23 := wf_grow_ds_arena hwf17 (s' := s23) (m_mr := s19.mem) (m' := m23)
    hstk'_b hstk'_rw hstk'_len rfl hr22 hs23_4 hs22_15 ha_hi' ha_disj
    hm23_stk hm23_ds hctx23_mem' hm23_from_19 hm19_from_s17 hbl23 hpos23 hcm23
  let vc23 : VmCtx :=
    { vc4 with stk := stk', ctxReg := ctx23,
               dsReg := { base := a, data := vc4.dsReg.data ++ List.replicate (vc4.dsCap * 16) 0, rw := true },
               dsCap := 2 * vc4.dsCap }
  have h23_disj : ∀ off, 256 ≤ off → off + 8 ≤ ctxSize →
      leVal (List.take 8 (List.drop off ctx23.data)) = leVal (List.take 8 (List.drop off vc4.ctxReg.data)) := by
    intro off hoff lim
    have lim' : off + 8 ≤ 4096 := lim
    dsimp [ctx23]
    rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; exact Or.inr (by omega))
      (by rw [writeBytes_length]; exact Nat.le_trans lim' hctx3_n)]
    rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; exact Or.inr (by omega))
      (Nat.le_trans lim' hctx3_n)]
  have hctx23_len : 4096 ≤ ctx23.data.length := by dsimp [ctx23]; rw [writeBytes_length, writeBytes_length]; exact hctx3_n
  have hrd23_of_17 : ∀ off v, 256 ≤ off → off + 8 ≤ ctxSize →
      rd64 s17.mem (ea ctxBase off) = some v → rd64 s23.mem (ea ctxBase off) = some v := by
    intro off v hoff lim hrd17
    have lim' : off + 8 ≤ 4096 := lim
    rw [ea_add_eq (by omega)]
    rw [rd64, mem_rdN_of_mem s23.mem ctx23 (ctxBase + off) 8 hwf23.1.2 (vc23.mem_ctx hwf23)
      (by dsimp [ctx23]; rw [hctx1]; exact Nat.le_add_right ctxBase off)
      (by simp only [Region.hi, ctx23, hctx1]
          exact Nat.add_le_add_left (Nat.le_trans lim' hctx23_len) ctxBase)]
    dsimp only [Option.map]
    rw [show ctxBase + off - ctx23.base = off by simp [ctx23, hctx1], h23_disj off hoff lim,
      ctx_leVal_of_rd64 hwf17 lim hrd17]
  have h23_288 : rd64 s23.mem (ea ctxBase 288) = some outs.length := hrd23_of_17 288 outs.length (by decide) (by decide) h4_288
  have h23_slots : ∀ (i : Nat) (_ : i < outs.length),
      rd64 s23.mem (ea ctxBase (360 + i * 16)) = some outs[i].w ∧
      rd64 s23.mem (ea ctxBase (368 + i * 16)) = some outs[i].plen := fun i hi =>
    ⟨hrd23_of_17 (360 + i * 16) outs[i].w (by omega) (by dsimp only [ctxSize]; omega) (h4_slots i hi).1,
     hrd23_of_17 (368 + i * 16) outs[i].plen (by omega) (by dsimp only [ctxSize]; omega) (h4_slots i hi).2⟩
  obtain ⟨s_fin, hrch_fin, hk_fin, hrel_fin⟩ :=
    epilogue_push_from_4195315 (s := s23) rfl hwf23 hrs hpcW houts_pos heout houtsW hcap_new h23_288 h23_slots
  have hrch_tot : Reaches vmImage s19 s_fin :=
    Reaches.trans (Reaches.step_trans h20 (Reaches.step_trans h21 (Reaches.step_trans h22 (Reaches.step h23)))) hrch_fin
  exact ⟨s_fin, hrch_tot, by rw [hk_fin]; rfl, oracleAdmits_of_reaches hor19 hrch_tot, hrel_fin⟩

theorem L_prim_epilogue_oracle_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {e_in : Nat} {outs : List SporosProvLen.LVal}
    {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.L_prim_epilogue) (h : vc.wf pr ls s) (hor : OracleAdmits s)
    (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (hein : e_in ≤ ls.ds.length) (heout : outs.length ≤ 3)
    (houtsW : ∀ v ∈ outs, v.w < X86Env.W ∧ v.plen < X86Env.W)
    (h280 : rd64 s.mem (ea ctxBase 280) = some e_in)
    (h288 : rd64 s.mem (ea ctxBase 288) = some outs.length)
    (hslots : ∀ (i : Nat) (_ : i < outs.length),
      rd64 s.mem (ea ctxBase (360 + i * 16)) = some outs[i].w ∧
      rd64 s.mem (ea ctxBase (368 + i * 16)) = some outs[i].plen) :
    let ls' := { ls with ds := outs ++ ls.ds.drop e_in, rs := { f with pc := f.pc + 1 } :: rest }
    (∃ s', Reaches vmImage s s' ∧ s'.k.out = s.k.out ∧ OracleAdmits s' ∧ VmRel pr ls' s') ∨
      (HitsNomem s ∧ Ends vmImage s (.exit 2)) := by
  have ⟨hcm, hbl, hpos, hreg, hsp, h15, hnodup, hmem, hstk1, hstk2, hstk3, hstk4, hctx1, hctx2, hctx3, _⟩ := h
  have hab := VmCtx.arena_bounds h
  have hW : X86Env.W = 2 ^ 64 := rfl
  have hpg : X86Env.page = 4096 := rfl
  have hdsLen_eq : vc.dsReg.data.length = vc.dsCap * 16 := hab.1.2.2.1
  have hdsHi47 : vc.dsReg.hi ≤ 2 ^ 47 := hab.1.2.2.2.2.1
  have hds_hi_n : vc.dsReg.base + vc.dsReg.data.length ≤ 2 ^ 47 := hdsHi47
  have hdsCap_ge4 : 4 ≤ vc.dsCap := hab.1.2.1
  have hdsPg : vc.dsCap * 16 % X86Env.page = 0 := hab.1.2.2.2.1
  have hdsCapW : vc.dsCap < X86Env.W := by omega
  have hdsLenW : ls.ds.length < X86Env.W := by have := hab.1.2.2.2.2.2.2; omega
  have hctxW : ctxBase + ctxSize ≤ X86Env.W := (vc.ctx_bounds h).2.2.2.2.1
  have hctx3_n : 4096 ≤ vc.ctxReg.data.length := hctx3
  have hctxW_n : ctxBase + 4096 ≤ X86Env.W := hctxW
  have hrip0 : s.rip = 4195233 := hrip
  have h01 : X86Vm.step vmImage s = .ok { s.set 0 (some ls.ds.length) with rip := 4195237 } := by
    rw [step_at_chunk 2 hrip0 rfl (by decide : (4195233, (.ofInt (.ld 0 15 8)), 4) ∈ SporosVmBin.vmTable_2),
      exec_ld h15 (vc.rd_ctx_dsLen h), hrip0]
  let s1 := { s.set 0 (some ls.ds.length) with rip := 4195237 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) ls.ds.length
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 0 (some ls.ds.length)).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hs1_280 : rd64 s1.mem (ea ctxBase 280) = some e_in := h280
  have h02 : X86Vm.step vmImage s1 = .ok { s1.set 1 (some e_in) with rip := 4195244 } := by
    rw [step_at_chunk 2 rfl rfl (by decide : (4195237, (.ofInt (.ld 1 15 280)), 7) ∈ SporosVmBin.vmTable_2),
      exec_ld hs1_15 hs1_280]
  let s2 := { s1.set 1 (some e_in) with rip := 4195244 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) e_in
  have hs2_0 : s2.reg 0 = some ls.ds.length := by
    change (s1.set 1 (some e_in)).reg 0 = some ls.ds.length; rw [reg_set_ne (by decide)]
    exact reg_set_self_ok hreg (by decide) (some ls.ds.length)
  have hs2_1 : s2.reg 1 = some e_in := reg_set_self_ok hr1 (by decide) (some e_in)
  have hsub_val : (X86Int.aluF .sub ls.ds.length e_in).1 = ls.ds.length - e_in := by
    rw [aluF_sub_val, show ls.ds.length + X86Env.W - e_in = (ls.ds.length - e_in) + X86Env.W by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have hsub_pair : X86Int.aluF .sub ls.ds.length e_in =
      (ls.ds.length - e_in, (X86Int.aluF .sub ls.ds.length e_in).2) := Prod.ext hsub_val rfl
  have h03 : X86Vm.step vmImage s2 =
      .ok { s2.set 0 (some (ls.ds.length - e_in)) with fl := (X86Int.aluF .sub ls.ds.length e_in).2, rip := 4195247 } := by
    rw [step_at_chunk 2 rfl rfl (by decide : (4195244, (.ofInt (.alu .sub 0 1)), 3) ∈ SporosVmBin.vmTable_2),
      exec_alu hs2_0 hs2_1, hsub_pair]
  let s3 := { s2.set 0 (some (ls.ds.length - e_in)) with fl := (X86Int.aluF .sub ls.ds.length e_in).2, rip := 4195247 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) (ls.ds.length - e_in)
  have hs3_15 : s3.reg 15 = some ctxBase := by
    change (s2.set 0 (some (ls.ds.length - e_in))).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s1.set 1 (some e_in)).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15
  have hs3_0 : s3.reg 0 = some (ls.ds.length - e_in) := reg_set_self_ok hr2 (by decide) (some (ls.ds.length - e_in))
  obtain ⟨_i4, m4, hwr4, _hm4_idx, _hm4_set, hm4_wf, hctx4_mem, hm4_oth⟩ :=
    wr64_region hcm.2 (vc.mem_ctx h) hctx2 (by rw [hctx1]; exact Nat.le_add_right ctxBase 8)
      (by simp only [Region.hi, hctx1]; exact Nat.add_le_add_left (Nat.le_trans (by decide : 16 ≤ 4096) hctx3_n) ctxBase)
      (a := ctxBase + 8) (v := ls.ds.length - e_in)
  have hwr4' : wr64 s3.mem (ea ctxBase 8) (ls.ds.length - e_in) = some m4 := by
    rw [ea_add_eq (by omega)]; exact hwr4
  have h04 : X86Vm.step vmImage s3 = .ok { s3 with rip := 4195251, mem := m4 } := by
    rw [step_at_chunk 2 rfl rfl (by decide : (4195247, (.ofInt (.st 15 8 0)), 4) ∈ SporosVmBin.vmTable_2),
      exec_st hs3_15 hs3_0 hwr4']
  let s4 := { s3 with rip := 4195251, mem := m4 }
  have hs4_4 : s4.reg 4 = some vc.sp := by
    change (s2.set 0 (some (ls.ds.length - e_in))).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s1.set 1 (some e_in)).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s.set 0 (some ls.ds.length)).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hsp
  have hs4_15 : s4.reg 15 = some ctxBase := hs3_15
  let ctx4 := { vc.ctxReg with data := writeBytes vc.ctxReg.data 8 (bytes8 (ls.ds.length - e_in)) }
  let vc4 := { vc with ctxReg := ctx4 }
  let ls4 := { ls with ds := ls.ds.drop e_in }
  have hctx4_mem' : ctx4 ∈ m4 := by
    simpa only [ctx4, hctx1, show ctxBase + 8 - ctxBase = 8 by omega] using hctx4_mem
  have hwf4 : vc4.wf pr ls4 s4 :=
    wf_write_ctx8_drop h hein hwr4' hctx4_mem' (fun u hu hnu => hm4_oth u hu (by rw [hctx1]; exact hnu))
      rfl hr3 hs4_4 hs3_15
  have hs4_disj : ∀ off, 256 ≤ off → off + 8 ≤ ctxSize →
      leVal (List.take 8 (List.drop off ctx4.data)) = leVal (List.take 8 (List.drop off vc.ctxReg.data)) := by
    intro off hoff lim
    dsimp only [ctxSize] at lim
    dsimp [ctx4]
    rw [writeBytes_drop_take_disjoint _ _ _ _ _ (by rw [bytes8_length]; exact Or.inr (by omega)) (by omega)]
  have hctx4_len : 4096 ≤ ctx4.data.length := by dsimp [ctx4]; rw [writeBytes_length]; exact hctx3_n
  have hrd4_of_old : ∀ off v, 256 ≤ off → off + 8 ≤ ctxSize →
      rd64 s.mem (ea ctxBase off) = some v → rd64 s4.mem (ea ctxBase off) = some v := by
    intro off v hoff lim hrd_old
    have lim' : off + 8 ≤ 4096 := lim
    rw [ea_add_eq (by omega)]
    rw [rd64, mem_rdN_of_mem s4.mem ctx4 (ctxBase + off) 8 hwf4.1.2 (vc4.mem_ctx hwf4)
      (by dsimp [ctx4]; rw [hctx1]; exact Nat.le_add_right ctxBase off)
      (by simp only [Region.hi, ctx4, hctx1]; exact Nat.add_le_add_left (Nat.le_trans lim' hctx4_len) ctxBase)]
    dsimp only [Option.map]
    rw [show ctxBase + off - ctx4.base = off by simp [ctx4, hctx1], hs4_disj off hoff lim,
      ctx_leVal_of_rd64 h lim hrd_old]
  have h4_288 : rd64 s4.mem (ea ctxBase 288) = some outs.length := hrd4_of_old 288 outs.length (by decide) (by decide) h288
  have h4_slots : ∀ (i : Nat) (_ : i < outs.length),
      rd64 s4.mem (ea ctxBase (360 + i * 16)) = some outs[i].w ∧
      rd64 s4.mem (ea ctxBase (368 + i * 16)) = some outs[i].plen := fun i hi =>
    ⟨hrd4_of_old (360 + i * 16) outs[i].w (by omega) (by dsimp only [ctxSize]; omega) (hslots i hi).1,
     hrd4_of_old (368 + i * 16) outs[i].plen (by omega) (by dsimp only [ctxSize]; omega) (hslots i hi).2⟩
  have h05 : X86Vm.step vmImage s4 = .ok { s4.set 3 (some outs.length) with rip := 4195258 } := by
    rw [step_at_chunk 2 rfl rfl (by decide : (4195251, (.ofInt (.ld 3 15 288)), 7) ∈ SporosVmBin.vmTable_2),
      exec_ld hs4_15 h4_288]
  let s5 := { s4.set 3 (some outs.length) with rip := 4195258 }
  have hr5 : RegsOk s5 := regsOk_set hr3 (by decide) outs.length
  have hs5_3 : s5.reg 3 = some outs.length := reg_set_self_ok hr3 (by decide) (some outs.length)
  have h06 : X86Vm.step vmImage s5 =
      .ok { s5 with fl := (X86Int.aluF .and outs.length outs.length).2, rip := 4195261 } := by
    rw [step_at_chunk 2 rfl rfl (by decide : (4195258, (.ofInt (.test 3 3)), 3) ∈ SporosVmBin.vmTable_2),
      exec_test hs5_3 hs5_3]
  let s6 := { s5 with fl := (X86Int.aluF .and outs.length outs.length).2, rip := 4195261 }
  have hrch6 : Reaches vmImage s s6 :=
    Reaches.step_trans h01 (Reaches.step_trans h02 (Reaches.step_trans h03 (Reaches.step_trans h04
      (Reaches.step_trans h05 (Reaches.step h06)))))
  have hs6_15 : s6.reg 15 = some ctxBase := by
    change (s4.set 3 (some outs.length)).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs3_15
  have hs6_4 : s6.reg 4 = some vc.sp := by
    change (s4.set 3 (some outs.length)).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs4_4
  have hs6_0 : s6.reg 0 = some (ls.ds.length - e_in) := by
    change (s4.set 3 (some outs.length)).reg 0 = some (ls.ds.length - e_in); rw [reg_set_ne (by decide)]; exact hs3_0
  have hwf6 : vc4.wf pr ls4 s6 := vc4.wf_step_misc hwf4 rfl hr5 hs6_4 hs6_15
  by_cases hzero : outs.length = 0
  · have houts_nil : outs = [] := List.eq_nil_of_length_eq_zero hzero
    have h07 : X86Vm.step vmImage s6 = .ok { s6 with rip := 4195385 } := by
      rw [step_at_chunk 2 rfl rfl (by decide : (4195261, (.ofInt (.jcc 4 4195385)), 2) ∈ SporosVmBin.vmTable_2),
        exec_jcc (cond_aluF_and_self_e outs.length)]
      dsimp only [s6]
      simp only [hzero, decide_true, ↓reduceIte]
    let s7 := { s6 with rip := 4195385 }
    have hwf7 : vc4.wf pr ls4 s7 := vc4.wf_step_misc hwf6 rfl hr5 hs6_4 hs6_15
    obtain ⟨s8, hrch8, hrip8, hk8, hwf8, hrel8⟩ :=
      L_adv_pc_spec (s := s7) rfl hwf7 hrs hpcW
    have hrch_tot : Reaches vmImage s s8 := Reaches.trans (Reaches.trans hrch6 (Reaches.step h07)) hrch8
    refine Or.inl ⟨s8, hrch_tot, by rw [hk8]; rfl, oracleAdmits_of_reaches hor hrch_tot, ?_⟩
    simpa [houts_nil, ls4] using hrel8
  · have h07 : X86Vm.step vmImage s6 = .ok { s6 with rip := 4195263 } := by
      rw [step_at_chunk 2 rfl rfl (by decide : (4195261, (.ofInt (.jcc 4 4195385)), 2) ∈ SporosVmBin.vmTable_2),
        exec_jcc (cond_aluF_and_self_e outs.length)]
      dsimp only [s6]
      simp only [hzero, decide_false, Bool.false_eq_true, ↓reduceIte]
    let s7 := { s6 with rip := 4195263 }
    have hs7_0 : s7.reg 0 = some (ls.ds.length - e_in) := hs6_0
    have h08 : X86Vm.step vmImage s7 = .ok { s7.set 12 (some (ls.ds.length - e_in)) with rip := 4195266 } := by
      rw [step_at_chunk 2 rfl rfl (by decide : (4195263, (.ofInt (.mov 12 0)), 3) ∈ SporosVmBin.vmTable_2),
        exec_mov hs7_0]
    let s8 := { s7.set 12 (some (ls.ds.length - e_in)) with rip := 4195266 }
    have hr8 : RegsOk s8 := regsOk_set hr5 (by decide) (ls.ds.length - e_in)
    have hs8_12 : s8.reg 12 = some (ls.ds.length - e_in) := reg_set_self_ok hr5 (by decide) _
    have hs8_3 : s8.reg 3 = some outs.length := by
      change (s7.set 12 (some (ls.ds.length - e_in))).reg 3 = some outs.length
      rw [reg_set_ne (by decide)]
      exact hs5_3
    have hadd12 : (X86Int.aluF .add (ls.ds.length - e_in) outs.length).1 = (ls.ds.length - e_in) + outs.length := by
      rw [aluF_add_val]; exact Nat.mod_eq_of_lt (by omega)
    have hadd12_pair : X86Int.aluF .add (ls.ds.length - e_in) outs.length =
        ((ls.ds.length - e_in) + outs.length, (X86Int.aluF .add (ls.ds.length - e_in) outs.length).2) :=
      Prod.ext hadd12 rfl
    have h09 : X86Vm.step vmImage s8 =
        .ok { s8.set 12 (some (ls.ds.length - e_in + outs.length)) with
                fl := (X86Int.aluF .add (ls.ds.length - e_in) outs.length).2, rip := 4195269 } := by
      rw [step_at_chunk 2 rfl rfl (by decide : (4195266, (.ofInt (.alu .add 12 3)), 3) ∈ SporosVmBin.vmTable_2),
        exec_alu hs8_12 hs8_3, hadd12_pair]
    let s9 := { s8.set 12 (some (ls.ds.length - e_in + outs.length)) with
                  fl := (X86Int.aluF .add (ls.ds.length - e_in) outs.length).2, rip := 4195269 }
    have hr9 : RegsOk s9 := regsOk_set hr8 (by decide) (ls.ds.length - e_in + outs.length)
    have hs9_15 : s9.reg 15 = some ctxBase := by
      change (s8.set 12 (some (ls.ds.length - e_in + outs.length))).reg 15 = some ctxBase
      rw [reg_set_ne (by decide)]
      change (s7.set 12 (some (ls.ds.length - e_in))).reg 15 = some ctxBase
      rw [reg_set_ne (by decide)]
      exact hs6_15
    have hs9_cap : rd64 s9.mem (ea ctxBase 16) = some vc.dsCap := by
      show rd64 s6.mem (ea ctxBase 16) = some vc.dsCap
      exact vc4.rd_ctx_dsCap hwf6
    have h10 : X86Vm.step vmImage s9 = .ok { s9.set 1 (some vc.dsCap) with rip := 4195273 } := by
      rw [step_at_chunk 2 rfl rfl (by decide : (4195269, (.ofInt (.ld 1 15 16)), 4) ∈ SporosVmBin.vmTable_2),
        exec_ld hs9_15 hs9_cap]
    let s10 := { s9.set 1 (some vc.dsCap) with rip := 4195273 }
    have hr10 : RegsOk s10 := regsOk_set hr9 (by decide) vc.dsCap
    have hs10_12 : s10.reg 12 = some (ls.ds.length - e_in + outs.length) := by
      change (s9.set 1 (some vc.dsCap)).reg 12 = some (ls.ds.length - e_in + outs.length)
      rw [reg_set_ne (by decide)]
      exact reg_set_self_ok hr8 (by decide) _
    have hs10_1 : s10.reg 1 = some vc.dsCap := reg_set_self_ok hr9 (by decide) _
    have h11 : X86Vm.step vmImage s10 =
        .ok { s10 with fl := (X86Int.aluF .sub (ls.ds.length - e_in + outs.length) vc.dsCap).2, rip := 4195276 } := by
      rw [step_at_chunk 2 rfl rfl (by decide : (4195273, (.ofInt (.cmp 12 1)), 3) ∈ SporosVmBin.vmTable_2),
        exec_cmp hs10_12 hs10_1]
    let s11 := { s10 with fl := (X86Int.aluF .sub (ls.ds.length - e_in + outs.length) vc.dsCap).2, rip := 4195276 }
    have hrch11 : Reaches vmImage s s11 :=
      Reaches.trans hrch6 (Reaches.step_trans h07 (Reaches.step_trans h08 (Reaches.step_trans h09
        (Reaches.step_trans h10 (Reaches.step h11)))))
    have hs11_15 : s11.reg 15 = some ctxBase := by
      change (s9.set 1 (some vc.dsCap)).reg 15 = some ctxBase
      rw [reg_set_ne (by decide)]
      exact hs9_15
    have hs11_4 : s11.reg 4 = some vc.sp := by
      change (s9.set 1 (some vc.dsCap)).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
      change (s8.set 12 (some (ls.ds.length - e_in + outs.length))).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
      change (s7.set 12 (some (ls.ds.length - e_in))).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
      exact hs6_4
    by_cases hcap_ok : ls.ds.length - e_in + outs.length ≤ vc.dsCap
    · have h12 : X86Vm.step vmImage s11 = .ok { s11 with rip := 4195315 } := by
        rw [step_at_chunk 2 rfl rfl (by decide : (4195276, (.ofInt (.jcc 6 4195315)), 2) ∈ SporosVmBin.vmTable_2),
          exec_jcc (cond_aluF_sub_be _ _ (by omega) hdsCapW)]
        dsimp only [s11]
        simp only [hcap_ok, decide_true, ↓reduceIte]
      let s12 := { s11 with rip := 4195315 }
      have hwf12 : vc4.wf pr ls4 s12 := vc4.wf_step_misc hwf6 rfl hr10 hs11_4 hs11_15
      obtain ⟨s_fin, hrch_fin, hk_fin, hrel_fin⟩ :=
        epilogue_push_from_4195315 (s := s12) rfl hwf12 hrs hpcW (Nat.pos_of_ne_zero hzero) heout houtsW
          (by simp [ls4, List.length_drop]; exact hcap_ok) h4_288 h4_slots
      have hrch_tot : Reaches vmImage s s_fin := Reaches.trans (Reaches.trans hrch11 (Reaches.step h12)) hrch_fin
      exact Or.inl ⟨s_fin, hrch_tot, by rw [hk_fin]; rfl, oracleAdmits_of_reaches hor hrch_tot, hrel_fin⟩
    · have h12 : X86Vm.step vmImage s11 = .ok { s11 with rip := 4195278 } := by
        rw [step_at_chunk 2 rfl rfl (by decide : (4195276, (.ofInt (.jcc 6 4195315)), 2) ∈ SporosVmBin.vmTable_2),
          exec_jcc (cond_aluF_sub_be _ _ (by omega) hdsCapW)]
        dsimp only [s11]
        simp only [hcap_ok, decide_false, Bool.false_eq_true, ↓reduceIte]
      let s12 := { s11 with rip := 4195278 }
      have hwf12 : vc4.wf pr ls4 s12 := vc4.wf_step_misc hwf6 rfl hr10 hs11_4 hs11_15
      obtain ⟨s17, hrch17, hrip17, hmem17, hk17, hr17, hwf17, hs17_7, hs17_6, hs17_2⟩ :=
        epilogue_grow_setup_spec (s12 := s12) rfl hwf12 hs10_1
      have hrch_s17 : Reaches vmImage s s17 := Reaches.trans (Reaches.trans hrch11 (Reaches.step h12)) hrch17
      have hor17 : OracleAdmits s17 := oracleAdmits_of_reaches hor hrch_s17
      rcases epilogue_grow_call_spec hrip17 hwf17 hor17 hs17_7 hs17_6 hs17_2 with
        ⟨a, dsReg', s19, stk', hrch19, hrip19, hk19_out, hor19, hcm19, hbl19, hpos19, hr19,
          hs19_0, hs19_4', hs19_15, hds'_eq, hds'_mem19, hstk'_b, hstk'_rw, hstk'_len, hstk'_mem19,
          hctx4_mem19, hm19_from_s17, ha_hi', ha_disj⟩ | ⟨hhit, hends⟩
      · have hcap_new : ls4.ds.length + outs.length ≤ 2 * vc4.dsCap := by
          simp only [ls4, List.length_drop]
          have h_le : ls.ds.length - e_in ≤ vc.dsCap := Nat.le_trans (Nat.sub_le _ _) hab.1.2.2.2.2.2.2
          have h4cap : 4 ≤ vc4.dsCap := hab.1.2.1
          change ls.ds.length - e_in + outs.length ≤ 2 * vc.dsCap
          omega
        obtain ⟨s_fin, hrch_fin, hk_fin, hor_fin, hrel_fin⟩ :=
          epilogue_post_mremap_spec hwf17 hrip19 hor19 hcm19 hbl19 hpos19 hr19 hs19_0 hs19_4' hs19_15
            hds'_eq hds'_mem19 hstk'_b hstk'_rw hstk'_len hstk'_mem19 hctx4_mem19 hm19_from_s17
            ha_hi' ha_disj hrs hpcW (Nat.pos_of_ne_zero hzero) heout houtsW hcap_new
            (by rw [hmem17]; exact h4_288) (by rw [hmem17]; exact h4_slots)
        have hrch_tot : Reaches vmImage s s_fin := Reaches.trans (Reaches.trans hrch_s17 hrch19) hrch_fin
        have hk_out : s_fin.k.out = s.k.out := by
          rw [hk_fin, hk19_out, hk17]
          rfl
        exact Or.inl ⟨s_fin, hrch_tot, hk_out, hor_fin, hrel_fin⟩
      · exact Or.inr ⟨hitsNomem_of_reaches hrch_s17 hhit, Ends.of_reaches hrch_s17 hends⟩

end SporosVmSimBase
