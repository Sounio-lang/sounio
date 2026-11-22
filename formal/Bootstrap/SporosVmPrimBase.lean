/-
Shared primitive simulation infrastructure and stack/arithmetic/logic/control
primitives (37 of 57 primitives) for `SporosVm`.

Root A framing:
Root A is MUSE's hand-written checker (`bootstrap/roots/sporos/sporos-stage1.hex`),
`spr1_check` is an independent second check, and `sporos_vm` is the post-admission
native x86-64 VM paired with `Sporos.decode` + `Sporos.checkProg`.
-/
import Bootstrap.SporosVmSimBase

namespace SporosVmPrimBase

open X86Env X86Int X86Vm SporosVmCore SporosVmExec SporosVmSimBase

local instance : Inhabited SporosProvLen.LVal := ⟨⟨0, 0⟩⟩
local instance : Inhabited SporosProvLen.LFrame := ⟨⟨0, 0, [], 0, 0⟩⟩

/-! ### General state, register, and bit/integer bridges -/

theorem reg_set_self_ok' {s : X86Vm.St} (hr : RegsOk s) {r : Nat} (hlt : r < 16) (v : Option Nat) :
    (s.set r v).reg r = v :=
  reg_set_self (by rw [hr.1]; exact hlt) v

theorem reg_of_ok {s : X86Vm.St} (hr : RegsOk s) (r : Nat) (h1 : r < 16) (h2 : r ≠ 11) :
    ∃ v, s.reg r = some v :=
  regs_get_of_ok hr r h1 h2

theorem getElem!_mem_of_lt {α : Type} [Inhabited α] {l : List α} {i : Nat} (hi : i < l.length) :
    l[i]! ∈ l := by
  rw [getElem!_pos l i hi]
  exact List.getElem_mem hi

theorem M_eq_W' : Sporos.M = X86Env.W := rfl

theorem wrap_eq_mod (x : Nat) : Sporos.wrap x = x % X86Env.W := rfl

theorem lt_W_of_mod (x : Nat) : x % X86Env.W < X86Env.W := Nat.mod_lt _ (by decide)

theorem land_lt_W {a b : Nat} (ha : a < X86Env.W) : a.land b < X86Env.W := by
  have : a.land b ≤ a := by
    change a &&& b ≤ a
    exact Nat.and_le_left
  exact Nat.lt_of_le_of_lt this ha

theorem lor_lt_W {a b : Nat} (ha : a < X86Env.W) (hb : b < X86Env.W) : a.lor b < X86Env.W :=
  Nat.or_lt_two_pow ha hb

theorem lxor_lt_W {a b : Nat} (ha : a < X86Env.W) (hb : b < X86Env.W) : a.xor b < X86Env.W :=
  Nat.xor_lt_two_pow ha hb

theorem add_pc_1_mod_eq {p pc : Nat} :
    ((p + pc) % X86Env.W + (1 % X86Env.W)) % X86Env.W = (p + pc + 1) % Sporos.M := by
  rw [M_eq_W', ← Nat.add_mod]

/-! ### Signed 64-bit arithmetic bridges matching `SporosProvLen.lexecPrim` -/

theorem toSigned64_hi_num {a : Nat} (ha : a < X86Env.W) :
    let hi := if X86Int.msb a = true then X86Env.W - 1 else 0
    X86Vm.toSigned64 hi * ↑X86Env.W + ↑a = X86Vm.toSigned64 a := by
  have hW_int : (X86Env.W : Int) = 18446744073709551616 := rfl
  have hm1 : X86Vm.toSigned64 (X86Env.W - 1) = -1 := by decide
  have hz : X86Vm.toSigned64 0 = 0 := by decide
  have hge : ∀ x, x < X86Env.W → (X86Int.msb x = true ↔ 2 ^ 63 ≤ x) := by
    intro x hx
    unfold X86Int.msb
    have hW2 : X86Env.W = 18446744073709551616 := rfl
    by_cases h : x < 2 ^ 63
    · have : x / 2 ^ 63 = 0 := Nat.div_eq_of_lt h
      simp [this, h]
    · have : x / 2 ^ 63 = 1 := by omega
      simp [this]; omega
  dsimp only
  by_cases hm : X86Int.msb a = true
  · rw [if_pos hm, hm1]
    have hge_a : 2 ^ 63 ≤ a := (hge a ha).mp hm
    dsimp only [X86Vm.toSigned64]
    rw [Nat.mod_eq_of_lt ha, if_neg (by omega)]
    omega
  · rw [if_neg hm, hz]
    have hlt_a : a < 2 ^ 63 := Nat.lt_of_not_le (mt (hge a ha).mpr hm)
    dsimp only [X86Vm.toSigned64]
    rw [Nat.mod_eq_of_lt ha, if_pos hlt_a]
    omega

theorem toSigned64_range {x : Nat} (hx : x < X86Env.W) :
    -(2 ^ 63 : Int) ≤ X86Vm.toSigned64 x ∧ X86Vm.toSigned64 x < (2 ^ 63 : Int) := by
  have hW_int : (X86Env.W : Int) = 18446744073709551616 := rfl
  dsimp only [X86Vm.toSigned64]
  rw [Nat.mod_eq_of_lt hx]
  split <;> omega

theorem toSigned64_zero_iff {x : Nat} (hx : x < X86Env.W) :
    X86Vm.toSigned64 x = 0 ↔ x = 0 := by
  have hW_int : (X86Env.W : Int) = 18446744073709551616 := rfl
  dsimp only [X86Vm.toSigned64]
  rw [Nat.mod_eq_of_lt hx]
  split <;> omega

theorem toSigned64_neg1_iff {x : Nat} (hx : x < X86Env.W) :
    X86Vm.toSigned64 x = -1 ↔ x = X86Env.W - 1 := by
  have hW_int : (X86Env.W : Int) = 18446744073709551616 := rfl
  have hW_nat : X86Env.W = 18446744073709551616 := rfl
  dsimp only [X86Vm.toSigned64]
  rw [Nat.mod_eq_of_lt hx]
  split <;> omega

theorem toSigned64_min_iff {x : Nat} (hx : x < X86Env.W) :
    X86Vm.toSigned64 x = -(2 ^ 63 : Int) ↔ x = 2 ^ 63 := by
  have hW_int : (X86Env.W : Int) = 18446744073709551616 := rfl
  dsimp only [X86Vm.toSigned64]
  rw [Nat.mod_eq_of_lt hx]
  split <;> omega

theorem idiv_bounds_of_ne_min_neg1 {a b : Nat} (ha : a < X86Env.W) (hb : b < X86Env.W)
    (hb0 : b ≠ 0) (hne : ¬ (a = 2 ^ 63 ∧ b = X86Env.W - 1)) :
    let num := X86Vm.toSigned64 a
    let den := X86Vm.toSigned64 b
    den ≠ 0 ∧ -(2 ^ 63 : Int) ≤ num.tdiv den ∧ num.tdiv den < (2 ^ 63 : Int) := by
  have hden0 : X86Vm.toSigned64 b ≠ 0 := by
    intro h; exact hb0 ((toSigned64_zero_iff hb).mp h)
  have ⟨ha_lo, ha_hi⟩ := toSigned64_range ha
  refine ⟨hden0, ?_⟩
  by_cases habs1 : (X86Vm.toSigned64 b).natAbs = 1
  · have hden_pm : X86Vm.toSigned64 b = 1 ∨ X86Vm.toSigned64 b = -1 := by omega
    rcases hden_pm with hd1 | hdm1
    · rw [hd1, Int.tdiv_one]; exact ⟨ha_lo, ha_hi⟩
    · have hb_eq : b = X86Env.W - 1 := (toSigned64_neg1_iff hb).mp hdm1
      have ha_ne : a ≠ 2 ^ 63 := fun h => hne ⟨h, hb_eq⟩
      have ha_gt : -(2 ^ 63 : Int) < X86Vm.toSigned64 a := by
        have hne_a : X86Vm.toSigned64 a ≠ -(2 ^ 63 : Int) := fun h => ha_ne ((toSigned64_min_iff ha).mp h)
        omega
      rw [hdm1, Int.tdiv_neg, Int.tdiv_one]
      omega
  · have hna : (X86Vm.toSigned64 a).natAbs ≤ 2 ^ 63 := by omega
    have hnb : 2 ≤ (X86Vm.toSigned64 b).natAbs := by omega
    have hnat_div : (X86Vm.toSigned64 a).natAbs / (X86Vm.toSigned64 b).natAbs ≤ 2 ^ 62 := by
      have : (X86Vm.toSigned64 a).natAbs / (X86Vm.toSigned64 b).natAbs ≤ (2 ^ 63) / 2 :=
        Nat.le_trans (Nat.div_le_div_right hna) (Nat.div_le_div_left hnb (by decide))
      omega
    have hq_abs : ((X86Vm.toSigned64 a).tdiv (X86Vm.toSigned64 b)).natAbs < 2 ^ 63 := by
      rw [Int.natAbs_tdiv]
      exact Nat.lt_of_le_of_lt hnat_div (by decide)
    rcases Int.natAbs_eq ((X86Vm.toSigned64 a).tdiv (X86Vm.toSigned64 b)) with hpos | hneg <;> omega

theorem divs_min_neg1_eq :
    Sporos.ofInt ((Sporos.toSigned (2 ^ 63)).tdiv (Sporos.toSigned (X86Env.W - 1))) = 2 ^ 63 := by
  decide

theorem mods_neg1_eq (a : Nat) :
    Sporos.ofInt ((Sporos.toSigned a).tmod (Sporos.toSigned (X86Env.W - 1))) = 0 := by
  have h1 : Sporos.toSigned (X86Env.W - 1) = -1 := by decide
  rw [h1, Int.tmod_neg, Int.tmod_one]
  rfl

theorem sar64_eq_sporos {a b : Nat} (ha : a < X86Env.W) :
    X86Vm.sar64 a b = Sporos.ofInt (Sporos.toSigned a / ↑(2 ^ (b % 64))) := by
  rw [X86Vm.sar64_eq_ofSigned64, toSigned_eq_toSigned64 ha, ofInt_eq_ofSigned64]
  rfl

/-! ### Context slot accessors from `VmCtx.wf` -/

theorem rd_ctx_msize {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : rd64 s.mem (ea ctxBase 112) = some ls.msize := by
  have ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, hsl14, _⟩ := h
  exact vc.rd64_ctx_of_eq h (by decide) hsl14

theorem rd_ctx_hp {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : rd64 s.mem (ea ctxBase 120) = some ls.hp := by
  have ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, hsl15, _⟩ := h
  exact vc.rd64_ctx_of_eq h (by decide) hsl15

theorem rd_ctx_pc {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : rd64 s.mem (ea ctxBase 224) = some ls.pc := by
  have ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, hsl28, _⟩ := h
  exact vc.rd64_ctx_of_eq h (by decide) hsl28

theorem wf_pc_lt_W {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : ls.pc < X86Env.W := by
  have ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, hwlt, _⟩ := h
  rw [← M_eq_W']
  exact hwlt.2.2.2.2.2

theorem wf_ds_lt_W {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) {v : SporosProvLen.LVal} (hv : v ∈ ls.ds) :
    v.w < X86Env.W ∧ v.plen < X86Env.W := by
  have ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, hwlt, _⟩ := h
  exact hwlt.1 v hv

theorem wf_msize_lt {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : ls.msize < 2 ^ 44 := by
  have ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, hmw2, _⟩ := h
  exact hmw2

/-! ### Shared simulation outcome predicate and lifting -/

def PrimPostOutcome (pr : Sporos.Prog) (ls : SporosProvLen.LSt)
    (f : SporosProvLen.LFrame) (rest : List SporosProvLen.LFrame) (s : X86Vm.St) (p : Sporos.Prim) : Prop :=
  match SporosProvLen.lexecPrim p (ls.ds.take p.eff.1) ls with
  | .ok res ls' =>
    let ls'' := { ls' with ds := res ++ ls.ds.drop p.eff.1, rs := { f with pc := f.pc + 1 } :: rest }
    (∃ s', Reaches vmImage s s' ∧ s'.k.out = s.k.out ∧ OracleAdmits s' ∧ VmRel pr ls'' s') ∨
      (HitsNomem s ∧ Ends vmImage s (.exit 2))
  | .halt code =>
    ∃ s_exit, Reaches vmImage s s_exit ∧ X86Vm.step vmImage s_exit = .exit (code % 256) ∧
      s_exit.k.out = s.k.out ++ ls.out.reverse
  | .fail _e =>
    ∃ s_exit, Reaches vmImage s s_exit ∧ X86Vm.step vmImage s_exit = .exit 1 ∧ s_exit.k = s.k

def PrimOutcome (pr : Sporos.Prog) (ls : SporosProvLen.LSt)
    (f : SporosProvLen.LFrame) (rest : List SporosProvLen.LFrame) (s : X86Vm.St) (p : Sporos.Prim) : Prop :=
  if ls.ds.length < p.eff.1 then
    ∃ s_exit, Reaches vmImage s s_exit ∧ X86Vm.step vmImage s_exit = .exit 1 ∧ s_exit.k = s.k
  else
    PrimPostOutcome pr ls f rest s p

theorem lift_next_outcome {pr : Sporos.Prog} {ls'' : SporosProvLen.LSt} {s₀ s₁ : X86Vm.St}
    (hrch : Reaches vmImage s₀ s₁) (hk : s₁.k = s₀.k) :
    ((∃ s', Reaches vmImage s₁ s' ∧ s'.k.out = s₁.k.out ∧ OracleAdmits s' ∧ VmRel pr ls'' s') ∨
      (HitsNomem s₁ ∧ Ends vmImage s₁ (.exit 2))) →
    ((∃ s', Reaches vmImage s₀ s' ∧ s'.k.out = s₀.k.out ∧ OracleAdmits s' ∧ VmRel pr ls'' s') ∨
      (HitsNomem s₀ ∧ Ends vmImage s₀ (.exit 2)))
  | Or.inl ⟨s', hr', hk', hor', hrel'⟩ =>
    Or.inl ⟨s', Reaches.trans hrch hr', by rw [hk', hk], hor', hrel'⟩
  | Or.inr ⟨hhit, hends⟩ =>
    Or.inr ⟨hitsNomem_of_reaches hrch hhit, Ends.of_reaches hrch hends⟩

theorem lift_fail_outcome {s₀ s₁ : X86Vm.St} (hrch : Reaches vmImage s₀ s₁) (hk : s₁.k = s₀.k)
    (hfail : ∃ s_exit, Reaches vmImage s₁ s_exit ∧ X86Vm.step vmImage s_exit = .exit 1 ∧ s_exit.k = s₁.k) :
    ∃ s_exit, Reaches vmImage s₀ s_exit ∧ X86Vm.step vmImage s_exit = .exit 1 ∧ s_exit.k = s₀.k := by
  obtain ⟨sx, hr, hsx, hks⟩ := hfail
  exact ⟨sx, Reaches.trans hrch hr, hsx, hks.trans hk⟩

theorem lift_halt_outcome {s₀ s₁ : X86Vm.St} {lout : List Nat} {code : Nat}
    (hrch : Reaches vmImage s₀ s₁) (hk : s₁.k = s₀.k)
    (hhalt : ∃ s_exit, Reaches vmImage s₁ s_exit ∧ X86Vm.step vmImage s_exit = .exit code ∧
      s_exit.k.out = s₁.k.out ++ lout) :
    ∃ s_exit, Reaches vmImage s₀ s_exit ∧ X86Vm.step vmImage s_exit = .exit code ∧
      s_exit.k.out = s₀.k.out ++ lout := by
  obtain ⟨sx, hr, hsx, hks⟩ := hhalt
  exact ⟨sx, Reaches.trans hrch hr, hsx, by rw [hks, hk]⟩

/-! ### Shared scratch-writer and prologue/epilogue combinators -/

/-- Execute `st 15 off r` into scratch `ctx[off]` (`256 ≤ off`), preserving all disjoint ctx slots. -/
theorem step_wr_ctx_scratch {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {len off r v : Nat}
    (hstep : X86Vm.step vmImage s = X86Vm.exec (.ofInt (.st 15 off r)) len s)
    (h : vc.wf pr ls s) (hr : s.reg r = some v) (hoff : 256 ≤ off) (hlim : off + 8 ≤ ctxSize) :
    let ctx' := { vc.ctxReg with data := writeBytes vc.ctxReg.data off (bytes8 v) }
    let vc' := { vc with ctxReg := ctx' }
    ∃ s',
      X86Vm.step vmImage s = .ok s' ∧
      s'.rip = s.rip + len ∧ s'.k = s.k ∧ s'.regs = s.regs ∧ s'.fl = s.fl ∧
      vc'.wf pr ls s' ∧
      rd64 s'.mem (ea ctxBase off) = some (v % X86Env.W) ∧
      (∀ m0 u, (m0 + 8 ≤ off ∨ off + 8 ≤ m0) → m0 + 8 ≤ ctxSize →
        rd64 s.mem (ea ctxBase m0) = some u → rd64 s'.mem (ea ctxBase m0) = some u) := by
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hreg : RegsOk s := h.2.2.2.1
  obtain ⟨idx, m', hwr, hidx, hm', hcm', hbl', hpos', _hle_self⟩ := vc.wr64_ctx_scratch h off v hoff hlim
  let s' : X86Vm.St := { regs := s.regs, rip := s.rip + len, fl := s.fl, mem := m', k := s.k }
  have hs : X86Vm.step vmImage s = .ok s' := by
    rw [hstep, exec_st h15 hr hwr]
  let ctx' := { vc.ctxReg with data := writeBytes vc.ctxReg.data off (bytes8 v) }
  let vc' := { vc with ctxReg := ctx' }
  have hwf' : vc'.wf pr ls s' :=
    vc.wf_write_ctx_scratch h hoff hidx hm' hcm' hbl' hpos' hreg h4 h15
  have ⟨_, _, _, _, _, _, _, _, _, _, _, _, hctx1, _, hctx3, _⟩ := h
  have hctxW := (vc.ctx_bounds h).2.2.2.2.1
  have hctx3' : 4096 ≤ vc.ctxReg.data.length := hctx3
  have hcb : ctxBase = 0x600000 := rfl
  have hcs : ctxSize = 4096 := rfl
  have hW_nat : X86Env.W = 18446744073709551616 := rfl
  have hrd_self : rd64 s'.mem (ea ctxBase off) = some (v % X86Env.W) := by
    have hctxs : ctxSize ≤ ctx'.data.length := by
      dsimp [ctx']; rw [writeBytes_length]; exact hctx3
    have hmem_ctx : ctx' ∈ s'.mem := vc'.mem_ctx hwf'
    rw [ea_add_eq (by omega)]
    rw [rd64, mem_rdN_of_mem s'.mem ctx' (ctxBase + off) 8 hwf'.1.2 hmem_ctx
      (by dsimp [ctx']; rw [hctx1]; exact Nat.le_add_right ctxBase off)
      (by simp only [Region.hi, ctx', hctx1, writeBytes_length]; omega)]
    dsimp only [Option.map, ctx']
    rw [show ctxBase + off - vc.ctxReg.base = off by rw [hctx1]; omega]
    exact congrArg some _hle_self
  have hrd_oth : ∀ m0 u, (m0 + 8 ≤ off ∨ off + 8 ≤ m0) → m0 + 8 ≤ ctxSize →
      rd64 s.mem (ea ctxBase m0) = some u → rd64 s'.mem (ea ctxBase m0) = some u := by
    intro m0 u hdisj hm0lim hrd0
    have hdisj' : m0 + 8 ≤ off ∨ off + (bytes8 v).length ≤ m0 := by
      rw [bytes8_length]; exact hdisj
    have hmem_ctx : ctx' ∈ s'.mem := vc'.mem_ctx hwf'
    have hm0lim' : m0 + 8 ≤ 4096 := hm0lim
    rw [ea_add_eq (by omega)]
    rw [rd64, mem_rdN_of_mem s'.mem ctx' (ctxBase + m0) 8 hwf'.1.2 hmem_ctx
      (by dsimp [ctx']; rw [hctx1]; exact Nat.le_add_right ctxBase m0)
      (by simp only [Region.hi, ctx', hctx1, writeBytes_length]; omega)]
    dsimp only [Option.map, ctx']
    rw [show ctxBase + m0 - vc.ctxReg.base = m0 by rw [hctx1]; omega,
      writeBytes_drop_take_disjoint _ _ _ _ _ hdisj' (Nat.le_trans hm0lim' hctx3'),
      ctx_leVal_of_rd64 h hm0lim hrd0]
  exact ⟨s', hs, rfl, rfl, rfl, rfl, hwf', hrd_self, hrd_oth⟩

/-- Compute `(pv + ls.pc + 1) % M` via the standard 5-instruction scratch tag sequence
    `ld 0 15 pOff; ld 1 15 224; add 0 1; addI 0 1; st 15 qOff 0` and store to `ctx[qOff]`. -/
theorem step_wr_ctx_tag_slot {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {pOff qOff pv nextAddr : Nat}
    (hstep0 : ∀ t : X86Vm.St, t.rip = s.rip → X86Vm.step vmImage t = X86Vm.exec (.ofInt (.ld 0 15 pOff)) 7 t)
    (hstep1 : ∀ t : X86Vm.St, t.rip = s.rip + 7 → X86Vm.step vmImage t = X86Vm.exec (.ofInt (.ld 1 15 224)) 7 t)
    (hstep2 : ∀ t : X86Vm.St, t.rip = s.rip + 14 → X86Vm.step vmImage t = X86Vm.exec (.ofInt (.alu .add 0 1)) 3 t)
    (hstep3 : ∀ t : X86Vm.St, t.rip = s.rip + 17 → X86Vm.step vmImage t = X86Vm.exec (.ofInt (.aluI .add 0 1)) 4 t)
    (hstep4 : ∀ t : X86Vm.St, t.rip = s.rip + 21 → X86Vm.step vmImage t = X86Vm.exec (.ofInt (.st 15 qOff 0)) 7 t)
    (hnext : s.rip + 28 = nextAddr)
    (h : vc.wf pr ls s) (hpv : rd64 s.mem (ea ctxBase pOff) = some pv)
    (hq : 256 ≤ qOff) (hqlim : qOff + 8 ≤ ctxSize) :
    let ctx' := { vc.ctxReg with data := writeBytes vc.ctxReg.data qOff (bytes8 ((pv + ls.pc + 1) % Sporos.M)) }
    let vc' := { vc with ctxReg := ctx' }
    ∃ s',
      Reaches vmImage s s' ∧ s'.rip = nextAddr ∧ s'.k = s.k ∧
      vc'.wf pr ls s' ∧
      rd64 s'.mem (ea ctxBase qOff) = some ((pv + ls.pc + 1) % Sporos.M) ∧
      (∀ m0 u, (m0 + 8 ≤ qOff ∨ qOff + 8 ≤ m0) → m0 + 8 ≤ ctxSize →
        rd64 s.mem (ea ctxBase m0) = some u → rd64 s'.mem (ea ctxBase m0) = some u) := by
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hreg : RegsOk s := h.2.2.2.1
  have h0 : X86Vm.step vmImage s = .ok { s.set 0 (some pv) with rip := s.rip + 7 } := by
    rw [hstep0 s rfl, exec_ld h15 hpv]
  let t1 := { s.set 0 (some pv) with rip := s.rip + 7 }
  have hr1 : RegsOk t1 := regsOk_set hreg (by decide) pv
  have ht1_15 : t1.reg 15 = some ctxBase := by
    change (s.set 0 (some pv)).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have ht1_0 : t1.reg 0 = some pv := reg_set_self_ok' hreg (by decide) (some pv)
  have hpc : rd64 t1.mem (ea ctxBase 224) = some ls.pc := (rd_ctx_pc h : _)
  have h1 : X86Vm.step vmImage t1 = .ok { t1.set 1 (some ls.pc) with rip := s.rip + 14 } := by
    rw [hstep1 t1 rfl, exec_ld ht1_15 hpc, show s.rip + 7 + 7 = s.rip + 14 by omega]
  let t2 := { t1.set 1 (some ls.pc) with rip := s.rip + 14 }
  have hr2 : RegsOk t2 := regsOk_set hr1 (by decide) ls.pc
  have ht2_0 : t2.reg 0 = some pv := by
    change (t1.set 1 (some ls.pc)).reg 0 = some pv; rw [reg_set_ne (by decide)]; exact ht1_0
  have ht2_1 : t2.reg 1 = some ls.pc := reg_set_self_ok' hr1 (by decide) (some ls.pc)
  have hadd0 : (X86Int.aluF .add pv ls.pc).1 = (pv + ls.pc) % X86Env.W := aluF_add_val pv ls.pc
  have hadd0_pair : X86Int.aluF .add pv ls.pc = ((pv + ls.pc) % X86Env.W, (X86Int.aluF .add pv ls.pc).2) :=
    Prod.ext hadd0 rfl
  have h2 : X86Vm.step vmImage t2 =
      .ok { t2.set 0 (some ((pv + ls.pc) % X86Env.W)) with fl := (X86Int.aluF .add pv ls.pc).2, rip := s.rip + 17 } := by
    rw [hstep2 t2 rfl, exec_alu ht2_0 ht2_1, hadd0_pair, show s.rip + 14 + 3 = s.rip + 17 by omega]
  let t3 := { t2.set 0 (some ((pv + ls.pc) % X86Env.W)) with fl := (X86Int.aluF .add pv ls.pc).2, rip := s.rip + 17 }
  have hr3 : RegsOk t3 := regsOk_set hr2 (by decide) ((pv + ls.pc) % X86Env.W)
  have ht3_0 : t3.reg 0 = some ((pv + ls.pc) % X86Env.W) := reg_set_self_ok' hr2 (by decide) _
  have hadd1 : (X86Int.aluF .add ((pv + ls.pc) % X86Env.W) (1 % X86Env.W)).1 = (pv + ls.pc + 1) % Sporos.M := by
    rw [aluF_add_val]; exact add_pc_1_mod_eq
  have hadd1_pair : X86Int.aluF .add ((pv + ls.pc) % X86Env.W) (1 % X86Env.W) =
      ((pv + ls.pc + 1) % Sporos.M, (X86Int.aluF .add ((pv + ls.pc) % X86Env.W) (1 % X86Env.W)).2) :=
    Prod.ext hadd1 rfl
  have h3 : X86Vm.step vmImage t3 =
      .ok { t3.set 0 (some ((pv + ls.pc + 1) % Sporos.M)) with
              fl := (X86Int.aluF .add ((pv + ls.pc) % X86Env.W) (1 % X86Env.W)).2, rip := s.rip + 21 } := by
    rw [hstep3 t3 rfl, exec_aluI ht3_0, hadd1_pair, show s.rip + 17 + 4 = s.rip + 21 by omega]
  let t4 := { t3.set 0 (some ((pv + ls.pc + 1) % Sporos.M)) with
                fl := (X86Int.aluF .add ((pv + ls.pc) % X86Env.W) (1 % X86Env.W)).2, rip := s.rip + 21 }
  have hr4 : RegsOk t4 := regsOk_set hr3 (by decide) ((pv + ls.pc + 1) % Sporos.M)
  have ht4_4 : t4.reg 4 = some vc.sp := by
    change (t3.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (t2.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (t1.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have ht4_15 : t4.reg 15 = some ctxBase := by
    change (t3.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (t2.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (t1.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact ht1_15
  have ht4_0 : t4.reg 0 = some ((pv + ls.pc + 1) % Sporos.M) := reg_set_self_ok' hr3 (by decide) _
  have hwf4 : vc.wf pr ls t4 := vc.wf_step_misc h rfl hr4 ht4_4 ht4_15
  obtain ⟨s', h4s, hrip', hk', _, _, hwf', hrd', hoth'⟩ :=
    step_wr_ctx_scratch (hstep4 t4 rfl) hwf4 ht4_0 hq hqlim
  have hmod_lt : ((pv + ls.pc + 1) % Sporos.M) % X86Env.W = (pv + ls.pc + 1) % Sporos.M := by
    rw [M_eq_W', Nat.mod_mod]
  rw [hmod_lt] at hrd'
  refine ⟨s', Reaches.step_trans h0 (Reaches.step_trans h1 (Reaches.step_trans h2 (Reaches.step_trans h3 (Reaches.step h4s)))),
    by rw [hrip']; dsimp [t4]; omega, hk', hwf', hrd', hoth'⟩

/-- Finish a single-output primitive via `L_prim_epilogue_mk1` (`4195210`). -/
theorem finish_mk1_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {e_in w q : Nat} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.L_prim_epilogue_mk1) (h : vc.wf pr ls s) (hor : OracleAdmits s)
    (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (hein : e_in ≤ ls.ds.length) (hr0 : s.reg 0 = some w)
    (h272 : rd64 s.mem (ea ctxBase 272) = some q) (hqW : q < X86Env.W)
    (h280 : rd64 s.mem (ea ctxBase 280) = some e_in)
    (h288 : rd64 s.mem (ea ctxBase 288) = some 1) :
    let ls' := { ls with ds := ⟨w % X86Env.W, q⟩ :: ls.ds.drop e_in, rs := { f with pc := f.pc + 1 } :: rest }
    (∃ s', Reaches vmImage s s' ∧ s'.k.out = s.k.out ∧ OracleAdmits s' ∧ VmRel pr ls' s') ∨
      (HitsNomem s ∧ Ends vmImage s (.exit 2)) := by
  obtain ⟨ctxData', s1, hrch1, hrip1, hk1, hwf1, _, h360, h368, hpres⟩ :=
    L_prim_epilogue_mk1_spec hrip h hr0 h272 hqW
  let vc' := { vc with ctxReg := { vc.ctxReg with data := ctxData' } }
  have h280' : rd64 s1.mem (ea ctxBase 280) = some e_in :=
    vc'.rd64_ctx_of_eq hwf1 (by decide)
      (by rw [hpres 280 (Or.inl (by decide)) (by decide), ctx_leVal_of_rd64 h (by decide) h280])
  have h288' : rd64 s1.mem (ea ctxBase 288) = some 1 :=
    vc'.rd64_ctx_of_eq hwf1 (by decide)
      (by rw [hpres 288 (Or.inl (by decide)) (by decide), ctx_leVal_of_rd64 h (by decide) h288])
  have hslots : ∀ (i : Nat) (_ : i < ([({ w := w % X86Env.W, plen := q } : SporosProvLen.LVal)]).length),
      rd64 s1.mem (ea ctxBase (360 + i * 16)) = some ([({ w := w % X86Env.W, plen := q } : SporosProvLen.LVal)][i]).w ∧
      rd64 s1.mem (ea ctxBase (368 + i * 16)) = some ([({ w := w % X86Env.W, plen := q } : SporosProvLen.LVal)][i]).plen := by
    intro i hi
    have hi0 : i = 0 := by simpa using hi
    subst hi0
    exact ⟨h360, h368⟩
  have hepi := L_prim_epilogue_oracle_spec
    (outs := [({ w := w % X86Env.W, plen := q } : SporosProvLen.LVal)])
    hrip1 hwf1 (oracleAdmits_of_reaches hor hrch1)
    hrs hpcW hein (by simp) (by
      intro v hv
      simp only [List.mem_singleton] at hv; subst hv; exact ⟨lt_W_of_mod w, hqW⟩)
    h280' h288' hslots
  exact lift_next_outcome hrch1 hk1 hepi

/-- Finish via a `jmp 4195210` (`L_prim_epilogue_mk1`) instruction. -/
theorem jmp_mk1_finish_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {e_in w q : Nat} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hstep : X86Vm.step vmImage s = X86Vm.exec (.ofInt (.jmp SporosVmBin.L_prim_epilogue_mk1)) 5 s)
    (h : vc.wf pr ls s) (hor : OracleAdmits s)
    (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (hein : e_in ≤ ls.ds.length) (hr0 : s.reg 0 = some w)
    (h272 : rd64 s.mem (ea ctxBase 272) = some q) (hqW : q < X86Env.W)
    (h280 : rd64 s.mem (ea ctxBase 280) = some e_in)
    (h288 : rd64 s.mem (ea ctxBase 288) = some 1) :
    let ls' := { ls with ds := ⟨w % X86Env.W, q⟩ :: ls.ds.drop e_in, rs := { f with pc := f.pc + 1 } :: rest }
    (∃ s', Reaches vmImage s s' ∧ s'.k.out = s.k.out ∧ OracleAdmits s' ∧ VmRel pr ls' s') ∨
      (HitsNomem s ∧ Ends vmImage s (.exit 2)) := by
  have hj : X86Vm.step vmImage s = .ok { s with rip := SporosVmBin.L_prim_epilogue_mk1 } := by
    rw [hstep, exec_jmp]
  let s1 := { s with rip := SporosVmBin.L_prim_epilogue_mk1 }
  have hwf1 : vc.wf pr ls s1 := vc.wf_step_misc h rfl h.2.2.2.1 h.2.2.2.2.1 h.2.2.2.2.2.1
  exact lift_next_outcome (Reaches.step hj) rfl
    (finish_mk1_spec rfl hwf1 (oracleAdmits_of_reaches hor (Reaches.step hj)) hrs hpcW hein hr0 h272 hqW h280 h288)

/-- Finish via a `jmp 4195233` (`L_prim_epilogue`) instruction with explicit `outs`. -/
theorem jmp_epilogue0_finish_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {e_in : Nat} {outs : List SporosProvLen.LVal}
    {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hstep : X86Vm.step vmImage s = X86Vm.exec (.ofInt (.jmp SporosVmBin.L_prim_epilogue)) 5 s)
    (h : vc.wf pr ls s) (hor : OracleAdmits s)
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
  have hj : X86Vm.step vmImage s = .ok { s with rip := SporosVmBin.L_prim_epilogue } := by
    rw [hstep, exec_jmp]
  let s1 := { s with rip := SporosVmBin.L_prim_epilogue }
  have hwf1 : vc.wf pr ls s1 := vc.wf_step_misc h rfl h.2.2.2.1 h.2.2.2.2.1 h.2.2.2.2.2.1
  exact lift_next_outcome (Reaches.step hj) rfl
    (L_prim_epilogue_oracle_spec rfl hwf1 (oracleAdmits_of_reaches hor (Reaches.step hj))
      hrs hpcW hein heout houtsW h280 h288 hslots)

/-! ### Shared arithmetic/unary/shift/compare primitive combinators -/

/-- Shared proof for 2-operand pure ALU primitives (`add`, `sub`, `band`, `bor`, `bxor`). -/
theorem alu_bin_mk1_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    {p : Sporos.Prim} {op : X86Int.Alu} {c0 c1 c2 c3 : Nat} {a0 : Nat} {resFn : Nat → Nat → Nat}
    (hrip : s.rip = a0)
    (hs0 : (a0, (.ofInt (.ld 0 15 312)), 7) ∈ (SporosVmBin.vmTableChunks[c0]?.getD (0, [])).2)
    (hs1 : (a0 + 7, (.ofInt (.ld 1 15 296)), 7) ∈ (SporosVmBin.vmTableChunks[c1]?.getD (0, [])).2)
    (hs2 : (a0 + 14, (.ofInt (.alu op 0 1)), 3) ∈ (SporosVmBin.vmTableChunks[c2]?.getD (0, [])).2)
    (hs3 : (a0 + 17, (.ofInt (.jmp SporosVmBin.L_prim_epilogue_mk1)), 5) ∈ (SporosVmBin.vmTableChunks[c3]?.getD (0, [])).2)
    (hpe : p.eff = (2, 1))
    (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s)
    (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2)
    (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w)
    (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w)
    (hval : ∀ a b, a < X86Env.W → b < X86Env.W →
      (X86Int.aluF op a b).1 % X86Env.W = resFn a b)
    (hlex : SporosProvLen.lexecPrim p (ls.ds.take 2) ls =
      .ok [⟨resFn ls.ds[1]!.w ls.ds[0]!.w, SporosProvLen.lq (ls.ds.take 2) ls.pc⟩] ls) :
    PrimPostOutcome pr ls f rest s p := by
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hreg : RegsOk s := h.2.2.2.1
  have ha0 : ls.ds[1]! ∈ ls.ds := getElem!_mem_of_lt (by omega)
  have hb0 : ls.ds[0]! ∈ ls.ds := getElem!_mem_of_lt (by omega)
  have haW : ls.ds[1]!.w < X86Env.W := (wf_ds_lt_W h ha0).1
  have hbW : ls.ds[0]!.w < X86Env.W := (wf_ds_lt_W h hb0).1
  have hch0 : ∃ b0 d0, SporosVmBin.vmTableChunks[c0]? = some (b0, d0) := by
    cases hk : SporosVmBin.vmTableChunks[c0]? with
    | none => simp [hk] at hs0
    | some p => exact ⟨p.1, p.2, rfl⟩
  have hch1 : ∃ b1 d1, SporosVmBin.vmTableChunks[c1]? = some (b1, d1) := by
    cases hk : SporosVmBin.vmTableChunks[c1]? with
    | none => simp [hk] at hs1
    | some p => exact ⟨p.1, p.2, rfl⟩
  have hch2 : ∃ b2 d2, SporosVmBin.vmTableChunks[c2]? = some (b2, d2) := by
    cases hk : SporosVmBin.vmTableChunks[c2]? with
    | none => simp [hk] at hs2
    | some p => exact ⟨p.1, p.2, rfl⟩
  have hch3 : ∃ b3 d3, SporosVmBin.vmTableChunks[c3]? = some (b3, d3) := by
    cases hk : SporosVmBin.vmTableChunks[c3]? with
    | none => simp [hk] at hs3
    | some p => exact ⟨p.1, p.2, rfl⟩
  obtain ⟨b0, d0, hc0⟩ := hch0
  obtain ⟨b1, d1, hc1⟩ := hch1
  obtain ⟨b2, d2, hc2⟩ := hch2
  obtain ⟨b3, d3, hc3⟩ := hch3
  rw [hc0] at hs0; rw [hc1] at hs1; rw [hc2] at hs2; rw [hc3] at hs3
  have h0 : X86Vm.step vmImage s = .ok { s.set 0 (some ls.ds[1]!.w) with rip := a0 + 7 } := by
    rw [step_at_chunk c0 hrip hc0 hs0, exec_ld h15 h312, hrip]
  let s1 := { s.set 0 (some ls.ds[1]!.w) with rip := a0 + 7 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) ls.ds[1]!.w
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hs1_0 : s1.reg 0 = some ls.ds[1]!.w := reg_set_self_ok' hreg (by decide) _
  have h1 : X86Vm.step vmImage s1 = .ok { s1.set 1 (some ls.ds[0]!.w) with rip := a0 + 14 } := by
    rw [step_at_chunk c1 rfl hc1 hs1, exec_ld hs1_15 h296, show a0 + 7 + 7 = a0 + 14 by omega]
  let s2 := { s1.set 1 (some ls.ds[0]!.w) with rip := a0 + 14 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) ls.ds[0]!.w
  have hs2_0 : s2.reg 0 = some ls.ds[1]!.w := by
    change (s1.set 1 _).reg 0 = some ls.ds[1]!.w; rw [reg_set_ne (by decide)]; exact hs1_0
  have hs2_1 : s2.reg 1 = some ls.ds[0]!.w := reg_set_self_ok' hr1 (by decide) _
  let af := X86Int.aluF op ls.ds[1]!.w ls.ds[0]!.w
  have h2 : X86Vm.step vmImage s2 = .ok { s2.set 0 (some af.1) with fl := af.2, rip := a0 + 17 } := by
    rw [step_at_chunk c2 rfl hc2 hs2, exec_alu hs2_0 hs2_1, show a0 + 14 + 3 = a0 + 17 by omega]
  let s3 := { s2.set 0 (some af.1) with fl := af.2, rip := a0 + 17 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) af.1
  have hs3_4 : s3.reg 4 = some vc.sp := by
    change (s2.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s1.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hs3_15 : s3.reg 15 = some ctxBase := by
    change (s2.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s1.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15
  have hs3_0 : s3.reg 0 = some af.1 := reg_set_self_ok' hr2 (by decide) _
  have hwf3 : vc.wf pr ls s3 := vc.wf_step_misc h rfl hr3 hs3_4 hs3_15
  have hqW : SporosProvLen.lq (ls.ds.take 2) ls.pc < X86Env.W := by
    dsimp [SporosProvLen.lq]; rw [← M_eq_W']; exact lt_W_of_mod _
  have hrch3 : Reaches vmImage s s3 := Reaches.step_trans h0 (Reaches.step_trans h1 (Reaches.step h2))
  have hfin := jmp_mk1_finish_spec (step_at_chunk c3 rfl hc3 hs3) hwf3
    (oracleAdmits_of_reaches hor hrch3) hrs hpcW hle hs3_0 h272 hqW h280 h288
  have hres := lift_next_outcome hrch3 rfl hfin
  dsimp only [PrimPostOutcome]
  rw [hpe]
  rw [hlex]
  dsimp only [af] at hres
  simpa only [hpe, List.singleton_append, hval ls.ds[1]!.w ls.ds[0]!.w haW hbW] using hres

/-- Shared proof for 2-operand comparison primitives (`eq`, `ne`, `ltu`, `lts`, `leu`, `les`). -/
theorem cmp_bin_mk1_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    {p : Sporos.Prim} {cc : Nat} {a0 : Nat} {cmpBool : Nat → Nat → Bool}
    (hrip : s.rip = a0)
    (hs0 : (a0, (.ofInt (.alu .xor 2 2)), 2) ∈ SporosVmBin.vmTable_6)
    (hs1 : (a0 + 2, (.ofInt (.ld 0 15 312)), 7) ∈ SporosVmBin.vmTable_6)
    (hs2 : (a0 + 9, (.ofInt (.ld 1 15 296)), 7) ∈ SporosVmBin.vmTable_6)
    (hs3 : (a0 + 16, (.ofInt (.cmp 0 1)), 3) ∈ SporosVmBin.vmTable_6)
    (hs4 : (a0 + 19, (.setcc cc 2), 3) ∈ SporosVmBin.vmTable_6)
    (hs5 : (a0 + 22, (.ofInt (.mov 0 2)), 3) ∈ SporosVmBin.vmTable_6)
    (hs6 : (a0 + 25, (.ofInt (.jmp SporosVmBin.L_prim_epilogue_mk1)), 5) ∈ SporosVmBin.vmTable_6)
    (hpe : p.eff = (2, 1)) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s)
    (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2)
    (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w)
    (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w)
    (hcond : ∀ a b, a < X86Env.W → b < X86Env.W →
      X86Int.cond (X86Int.aluF .sub a b).2 cc = some (cmpBool a b))
    (hlex : SporosProvLen.lexecPrim p (ls.ds.take 2) ls =
      .ok [⟨Sporos.bool (cmpBool ls.ds[1]!.w ls.ds[0]!.w), SporosProvLen.lq (ls.ds.take 2) ls.pc⟩] ls) :
    PrimPostOutcome pr ls f rest s p := by
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hreg : RegsOk s := h.2.2.2.1
  have haW : ls.ds[1]!.w < X86Env.W := (wf_ds_lt_W h (getElem!_mem_of_lt (by omega))).1
  have hbW : ls.ds[0]!.w < X86Env.W := (wf_ds_lt_W h (getElem!_mem_of_lt (by omega))).1
  obtain ⟨r2, hr2_0⟩ := reg_of_ok hreg 2 (by decide) (by decide)
  have hxor0 : (X86Int.aluF .xor r2 r2).1 = 0 := by rw [aluF_xor_val]; exact Nat.xor_self r2
  have hxor_pair : X86Int.aluF .xor r2 r2 = (0, (X86Int.aluF .xor r2 r2).2) := Prod.ext hxor0 rfl
  have h0 : X86Vm.step vmImage s =
      .ok { s.set 2 (some 0) with fl := (X86Int.aluF .xor r2 r2).2, rip := a0 + 2 } := by
    rw [step_at_chunk 6 hrip rfl hs0, exec_alu hr2_0 hr2_0, hxor_pair, hrip]
  let s1 := { s.set 2 (some 0) with fl := (X86Int.aluF .xor r2 r2).2, rip := a0 + 2 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) 0
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hs1_2 : s1.reg 2 = some 0 := reg_set_self_ok' hreg (by decide) _
  have h1 : X86Vm.step vmImage s1 = .ok { s1.set 0 (some ls.ds[1]!.w) with rip := a0 + 9 } := by
    rw [step_at_chunk 6 rfl rfl hs1, exec_ld hs1_15 h312, show a0 + 2 + 7 = a0 + 9 by omega]
  let s2 := { s1.set 0 (some ls.ds[1]!.w) with rip := a0 + 9 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) ls.ds[1]!.w
  have hs2_15 : s2.reg 15 = some ctxBase := by
    change (s1.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15
  have hs2_0 : s2.reg 0 = some ls.ds[1]!.w := reg_set_self_ok' hr1 (by decide) _
  have hs2_2 : s2.reg 2 = some 0 := by
    change (s1.set 0 _).reg 2 = some 0; rw [reg_set_ne (by decide)]; exact hs1_2
  have h2 : X86Vm.step vmImage s2 = .ok { s2.set 1 (some ls.ds[0]!.w) with rip := a0 + 16 } := by
    rw [step_at_chunk 6 rfl rfl hs2, exec_ld hs2_15 h296, show a0 + 9 + 7 = a0 + 16 by omega]
  let s3 := { s2.set 1 (some ls.ds[0]!.w) with rip := a0 + 16 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) ls.ds[0]!.w
  have hs3_0 : s3.reg 0 = some ls.ds[1]!.w := by
    change (s2.set 1 _).reg 0 = some ls.ds[1]!.w; rw [reg_set_ne (by decide)]; exact hs2_0
  have hs3_1 : s3.reg 1 = some ls.ds[0]!.w := reg_set_self_ok' hr2 (by decide) _
  have hs3_2 : s3.reg 2 = some 0 := by
    change (s2.set 1 _).reg 2 = some 0; rw [reg_set_ne (by decide)]; exact hs2_2
  have h3 : X86Vm.step vmImage s3 =
      .ok { s3 with fl := (X86Int.aluF .sub ls.ds[1]!.w ls.ds[0]!.w).2, rip := a0 + 19 } := by
    rw [step_at_chunk 6 rfl rfl hs3, exec_cmp hs3_0 hs3_1, show a0 + 16 + 3 = a0 + 19 by omega]
  let s4 := { s3 with fl := (X86Int.aluF .sub ls.ds[1]!.w ls.ds[0]!.w).2, rip := a0 + 19 }
  have hs4_2 : s4.reg 2 = some 0 := hs3_2
  let b := cmpBool ls.ds[1]!.w ls.ds[0]!.w
  have h4s : X86Vm.step vmImage s4 =
      .ok { s4.set 2 (some (if b = true then 1 else 0)) with rip := a0 + 22 } := by
    rw [step_at_chunk 6 rfl rfl hs4, exec_setcc (hcond ls.ds[1]!.w ls.ds[0]!.w haW hbW) hs4_2]
    dsimp only
    rw [show 0 % X86Env.W - 0 % X86Env.W % 256 = 0 by decide, Nat.zero_add, show a0 + 19 + 3 = a0 + 22 by omega]
  let s5 := { s4.set 2 (some (if b = true then 1 else 0)) with rip := a0 + 22 }
  have hr5 : RegsOk s5 := regsOk_set hr3 (by decide) (if b = true then 1 else 0)
  have hs5_2 : s5.reg 2 = some (if b = true then 1 else 0) := reg_set_self_ok' hr3 (by decide) _
  have h5 : X86Vm.step vmImage s5 =
      .ok { s5.set 0 (some (if b = true then 1 else 0)) with rip := a0 + 25 } := by
    rw [step_at_chunk 6 rfl rfl hs5, exec_mov hs5_2, show a0 + 22 + 3 = a0 + 25 by omega]
  let s6 := { s5.set 0 (some (if b = true then 1 else 0)) with rip := a0 + 25 }
  have hr6 : RegsOk s6 := regsOk_set hr5 (by decide) (if b = true then 1 else 0)
  have hs6_4 : s6.reg 4 = some vc.sp := by
    change (s5.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s4.set 2 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s2.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s1.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s.set 2 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hs6_15 : s6.reg 15 = some ctxBase := by
    change (s5.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s4.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s2.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs2_15
  have hs6_0 : s6.reg 0 = some (if b = true then 1 else 0) := reg_set_self_ok' hr5 (by decide) _
  have hwf6 : vc.wf pr ls s6 := vc.wf_step_misc h rfl hr6 hs6_4 hs6_15
  have hqW : SporosProvLen.lq (ls.ds.take 2) ls.pc < X86Env.W := by
    dsimp [SporosProvLen.lq]; rw [← M_eq_W']; exact lt_W_of_mod _
  have hrch6 : Reaches vmImage s s6 :=
    Reaches.step_trans h0 (Reaches.step_trans h1 (Reaches.step_trans h2
      (Reaches.step_trans h3 (Reaches.step_trans h4s (Reaches.step h5)))))
  have hfin := jmp_mk1_finish_spec (step_at_chunk 6 rfl rfl hs6) hwf6
    (oracleAdmits_of_reaches hor hrch6) hrs hpcW hle hs6_0 h272 hqW h280 h288
  have hres := lift_next_outcome hrch6 rfl hfin
  have hmod_bool : (if b = true then 1 else 0) % X86Env.W = Sporos.bool b := by
    dsimp only [Sporos.bool]; split <;> decide
  dsimp only [PrimPostOutcome]
  rw [hpe]
  rw [hlex]
  simpa only [hpe, hmod_bool, List.singleton_append] using hres

/-- Shared proof for `P_clock`, `P_core`, `P_sensor` sharing `P_zero_mk1` (`4198923`). -/
theorem zero_mk1_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    {p : Sporos.Prim} {e_in : Nat}
    (hrip : s.rip = SporosVmBin.P_zero_mk1)
    (hpe : p.eff = (e_in, 1)) (hle : e_in ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s)
    (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take e_in) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some e_in)
    (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (hlex : SporosProvLen.lexecPrim p (ls.ds.take e_in) ls =
      .ok [⟨0, SporosProvLen.lq (ls.ds.take e_in) ls.pc⟩] ls) :
    PrimPostOutcome pr ls f rest s p := by
  have hrip' : s.rip = 4198923 := hrip
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hreg : RegsOk s := h.2.2.2.1
  obtain ⟨r0, hr0⟩ := reg_of_ok hreg 0 (by decide) (by decide)
  have hxor0 : (X86Int.aluF .xor r0 r0).1 = 0 := by rw [aluF_xor_val]; exact Nat.xor_self r0
  have hxor_pair : X86Int.aluF .xor r0 r0 = (0, (X86Int.aluF .xor r0 r0).2) := Prod.ext hxor0 rfl
  have h0 : X86Vm.step vmImage s =
      .ok { s.set 0 (some 0) with fl := (X86Int.aluF .xor r0 r0).2, rip := 4198925 } := by
    rw [step_at_chunk 14 hrip' rfl (by decide : (4198923, (.ofInt (.alu .xor 0 0)), 2) ∈ SporosVmBin.vmTable_14),
      exec_alu hr0 hr0, hxor_pair, hrip']
  let s1 := { s.set 0 (some 0) with fl := (X86Int.aluF .xor r0 r0).2, rip := 4198925 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) 0
  have hs1_4 : s1.reg 4 = some vc.sp := by
    change (s.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hs1_0 : s1.reg 0 = some 0 := reg_set_self_ok' hreg (by decide) _
  have hwf1 : vc.wf pr ls s1 := vc.wf_step_misc h rfl hr1 hs1_4 hs1_15
  have hqW : SporosProvLen.lq (ls.ds.take e_in) ls.pc < X86Env.W := by
    dsimp [SporosProvLen.lq]; rw [← M_eq_W']; exact lt_W_of_mod _
  have hfin := jmp_mk1_finish_spec
    (step_at_chunk 14 rfl rfl (by decide : (4198925, (.ofInt (.jmp 4195210)), 5) ∈ SporosVmBin.vmTable_14))
    hwf1 (oracleAdmits_of_reaches hor (Reaches.step h0)) hrs hpcW hle hs1_0 h272 hqW h280 h288
  have hres := lift_next_outcome (Reaches.step h0) rfl hfin
  dsimp only [PrimPostOutcome]
  rw [hpe]
  rw [hlex]
  simpa only [hpe, show (0 : Nat) % X86Env.W = 0 by decide, List.singleton_append] using hres

/-! ### Stack primitives (`dup`, `drop`, `swap`, `over`, `rot`, `nip`, `tuck`) -/

private theorem lq_take1 {a : SporosProvLen.LVal} {pc : Nat} :
    SporosProvLen.lq [a] pc = (a.plen + pc + 1) % Sporos.M := by
  simp only [SporosProvLen.lq, List.foldr, Nat.add_zero]

theorem P_dup_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_dup) (hle : 1 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h280 : rd64 s.mem (ea ctxBase 280) = some 1)
    (h288 : rd64 s.mem (ea ctxBase 288) = some 2)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w)
    (h304 : rd64 s.mem (ea ctxBase 304) = some ls.ds[0]!.plen) :
    PrimPostOutcome pr ls f rest s .dup := by
  have hrip' : s.rip = 4195421 := hrip
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hreg : RegsOk s := h.2.2.2.1
  have haW := wf_ds_lt_W h (getElem!_mem_of_lt (by omega : 0 < ls.ds.length))
  -- step 0: ld 0 15 296
  have h0 : X86Vm.step vmImage s = .ok { s.set 0 (some ls.ds[0]!.w) with rip := 4195428 } := by
    rw [step_at_chunk 3 hrip' rfl (by decide : (4195421, (.ofInt (.ld 0 15 296)), 7) ∈ SporosVmBin.vmTable_3),
      exec_ld h15 h296, hrip']
  let s1 := { s.set 0 (some ls.ds[0]!.w) with rip := 4195428 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) ls.ds[0]!.w
  have hs1_4 : s1.reg 4 = some vc.sp := by
    change (s.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hs1_0 : s1.reg 0 = some ls.ds[0]!.w := reg_set_self_ok' hreg (by decide) _
  have hwf1 : vc.wf pr ls s1 := vc.wf_step_misc h rfl hr1 hs1_4 hs1_15
  -- step 1: st 15 360 0
  obtain ⟨s2, hs2, hrip2, hk2, _, _, hwf2, hrd360, hoth2⟩ :=
    step_wr_ctx_scratch (step_at_chunk 3 rfl rfl (by decide : (4195428, (.ofInt (.st 15 360 0)), 7) ∈ SporosVmBin.vmTable_3))
      hwf1 hs1_0 (by decide) (by decide)
  rw [Nat.mod_eq_of_lt haW.1] at hrd360
  -- steps 2..6: tag slot 304 -> 368
  have h304_2 : rd64 s2.mem (ea ctxBase 304) = some ls.ds[0]!.plen :=
    hoth2 304 _ (Or.inl (by decide)) (by decide) h304
  obtain ⟨s7, hrch7, hrip7, hk7, hwf7, hrd368, hoth7⟩ :=
    step_wr_ctx_tag_slot
      (fun t ht => step_at_chunk 3 (by rw [ht, hrip2]) rfl (by decide : (4195435, (.ofInt (.ld 0 15 304)), 7) ∈ SporosVmBin.vmTable_3))
      (fun t ht => step_at_chunk 3 (by rw [ht, hrip2]) rfl (by decide : (4195442, (.ofInt (.ld 1 15 224)), 7) ∈ SporosVmBin.vmTable_3))
      (fun t ht => step_at_chunk 3 (by rw [ht, hrip2]) rfl (by decide : (4195449, (.ofInt (.alu .add 0 1)), 3) ∈ SporosVmBin.vmTable_3))
      (fun t ht => step_at_chunk 3 (by rw [ht, hrip2]) rfl (by decide : (4195452, (.ofInt (.aluI .add 0 1)), 4) ∈ SporosVmBin.vmTable_3))
      (fun t ht => step_at_chunk 3 (by rw [ht, hrip2]) rfl (by decide : (4195456, (.ofInt (.st 15 368 0)), 7) ∈ SporosVmBin.vmTable_3))
      (nextAddr := 4195463) (by rw [hrip2]) hwf2 h304_2 (by decide) (by decide)
  have hrd360_7 : rd64 s7.mem (ea ctxBase 360) = some ls.ds[0]!.w :=
    hoth7 360 _ (Or.inl (by decide)) (by decide) hrd360
  -- copy 360 -> 376 and 368 -> 384
  have h7_15 : s7.reg 15 = some ctxBase := hwf7.2.2.2.2.2.1
  have h7_4 : s7.reg 4 = some vc.sp := hwf7.2.2.2.2.1
  have hr7 : RegsOk s7 := hwf7.2.2.2.1
  have h7 : X86Vm.step vmImage s7 = .ok { s7.set 0 (some ls.ds[0]!.w) with rip := 4195470 } := by
    rw [step_at_chunk 3 hrip7 rfl (by decide : (4195463, (.ofInt (.ld 0 15 360)), 7) ∈ SporosVmBin.vmTable_3),
      exec_ld h7_15 hrd360_7, hrip7]
  let s8 := { s7.set 0 (some ls.ds[0]!.w) with rip := 4195470 }
  have hr8 : RegsOk s8 := regsOk_set hr7 (by decide) ls.ds[0]!.w
  have hs8_4 : s8.reg 4 = some vc.sp := by
    change (s7.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h7_4
  have hs8_15 : s8.reg 15 = some ctxBase := by
    change (s7.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h7_15
  have hs8_0 : s8.reg 0 = some ls.ds[0]!.w := reg_set_self_ok' hr7 (by decide) _
  have hwf8 := VmCtx.wf_step_misc (s' := s8) hwf7 rfl hr8 hs8_4 hs8_15
  obtain ⟨s9, hs9, hrip9, hk9, _, _, hwf9, hrd376, hoth9⟩ :=
    step_wr_ctx_scratch (step_at_chunk 3 rfl rfl (by decide : (4195470, (.ofInt (.st 15 376 0)), 7) ∈ SporosVmBin.vmTable_3))
      hwf8 hs8_0 (by decide) (by decide)
  rw [Nat.mod_eq_of_lt haW.1] at hrd376
  have hrd368_9 : rd64 s9.mem (ea ctxBase 368) = some ((ls.ds[0]!.plen + ls.pc + 1) % Sporos.M) :=
    hoth9 368 _ (Or.inl (by decide)) (by decide) hrd368
  have h9_15 : s9.reg 15 = some ctxBase := hwf9.2.2.2.2.2.1
  have h9_4 : s9.reg 4 = some vc.sp := hwf9.2.2.2.2.1
  have hr9 : RegsOk s9 := hwf9.2.2.2.1
  have h9 : X86Vm.step vmImage s9 =
      .ok { s9.set 0 (some ((ls.ds[0]!.plen + ls.pc + 1) % Sporos.M)) with rip := 4195484 } := by
    rw [step_at_chunk 3 hrip9 rfl (by decide : (4195477, (.ofInt (.ld 0 15 368)), 7) ∈ SporosVmBin.vmTable_3),
      exec_ld h9_15 hrd368_9, hrip9]
  let s10 := { s9.set 0 (some ((ls.ds[0]!.plen + ls.pc + 1) % Sporos.M)) with rip := 4195484 }
  have hr10 : RegsOk s10 := regsOk_set hr9 (by decide) _
  have hs10_4 : s10.reg 4 = some vc.sp := by
    change (s9.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h9_4
  have hs10_15 : s10.reg 15 = some ctxBase := by
    change (s9.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h9_15
  have hs10_0 : s10.reg 0 = some ((ls.ds[0]!.plen + ls.pc + 1) % Sporos.M) := reg_set_self_ok' hr9 (by decide) _
  have hwf10 := VmCtx.wf_step_misc (s' := s10) hwf9 rfl hr10 hs10_4 hs10_15
  obtain ⟨s11, hs11, hrip11, hk11, _, _, hwf11, hrd384, hoth11⟩ :=
    step_wr_ctx_scratch (step_at_chunk 3 rfl rfl (by decide : (4195484, (.ofInt (.st 15 384 0)), 7) ∈ SporosVmBin.vmTable_3))
      hwf10 hs10_0 (by decide) (by decide)
  rw [M_eq_W', Nat.mod_mod] at hrd384
  have hrch11 : Reaches vmImage s s11 :=
    Reaches.trans (Reaches.step_trans h0 (Reaches.step hs2))
      (Reaches.trans hrch7 (Reaches.step_trans h7 (Reaches.step_trans hs9 (Reaches.step_trans h9 (Reaches.step hs11)))))
  have hk_tot : s11.k = s.k := by
    dsimp only [s10, s8, s1, X86Int.St.set] at hk11 hk9 hk7 hk2 ⊢
    rw [hk11, hk9, hk7, hk2]
  have h280_11 : rd64 s11.mem (ea ctxBase 280) = some 1 :=
    hoth11 280 _ (Or.inl (by decide)) (by decide) (hoth9 280 _ (Or.inl (by decide)) (by decide)
      (hoth7 280 _ (Or.inl (by decide)) (by decide) (hoth2 280 _ (Or.inl (by decide)) (by decide) h280)))
  have h288_11 : rd64 s11.mem (ea ctxBase 288) = some 2 :=
    hoth11 288 _ (Or.inl (by decide)) (by decide) (hoth9 288 _ (Or.inl (by decide)) (by decide)
      (hoth7 288 _ (Or.inl (by decide)) (by decide) (hoth2 288 _ (Or.inl (by decide)) (by decide) h288)))
  have hrd360_11 : rd64 s11.mem (ea ctxBase 360) = some ls.ds[0]!.w :=
    hoth11 360 _ (Or.inl (by decide)) (by decide) (hoth9 360 _ (Or.inl (by decide)) (by decide) hrd360_7)
  have hrd368_11 : rd64 s11.mem (ea ctxBase 368) = some ((ls.ds[0]!.plen + ls.pc + 1) % Sporos.M) :=
    hoth11 368 _ (Or.inl (by decide)) (by decide) hrd368_9
  have hrd376_11 : rd64 s11.mem (ea ctxBase 376) = some ls.ds[0]!.w :=
    hoth11 376 _ (Or.inl (by decide)) (by decide) hrd376
  let ta := SporosProvLen.ltag ls.pc ls.ds[0]!
  have houtsW : ∀ v ∈ ([ta, ta] : List SporosProvLen.LVal), v.w < X86Env.W ∧ v.plen < X86Env.W := by
    intro v hv
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hv
    rcases hv with rfl | rfl <;> exact ⟨haW.1, by dsimp [ta, SporosProvLen.ltag]; rw [M_eq_W']; exact lt_W_of_mod _⟩
  have hslots : ∀ (i : Nat) (_ : i < ([ta, ta] : List SporosProvLen.LVal).length),
      rd64 s11.mem (ea ctxBase (360 + i * 16)) = some ([ta, ta][i]).w ∧
      rd64 s11.mem (ea ctxBase (368 + i * 16)) = some ([ta, ta][i]).plen := by
    intro i hi
    rcases i with _ | _ | i
    · exact ⟨hrd360_11, hrd368_11⟩
    · exact ⟨hrd376_11, hrd384⟩
    · simp only [List.length_cons, List.length_nil] at hi; omega
  have hfin := jmp_epilogue0_finish_spec
    (step_at_chunk 3 hrip11 rfl (by decide : (4195491, (.ofInt (.jmp 4195233)), 5) ∈ SporosVmBin.vmTable_3))
    hwf11 (oracleAdmits_of_reaches hor hrch11) hrs hpcW hle (by simp) houtsW h280_11 h288_11 hslots
  have hres := lift_next_outcome hrch11 hk_tot hfin
  rcases hds_eq : ls.ds with _ | ⟨a, r⟩
  · simp [hds_eq] at hle
  · dsimp only [PrimPostOutcome]
    simp only [Sporos.Prim.eff, List.take_succ_cons, List.take_zero, SporosProvLen.lexecPrim, hds_eq, List.drop_succ_cons, List.drop_zero]
    simpa only [ta, hds_eq, List.getElem!_cons_zero, List.drop_succ_cons, List.drop_zero] using hres

theorem P_drop_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_drop) (hle : 1 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h280 : rd64 s.mem (ea ctxBase 280) = some 1)
    (h288 : rd64 s.mem (ea ctxBase 288) = some 0) :
    PrimPostOutcome pr ls f rest s .drop := by
  have hrip' : s.rip = 4195496 := hrip
  have hfin := jmp_epilogue0_finish_spec (outs := [])
    (step_at_chunk 3 hrip' rfl (by decide : (4195496, (.ofInt (.jmp 4195233)), 5) ∈ SporosVmBin.vmTable_3))
    h hor hrs hpcW hle (by decide) (fun _ hv => by simp at hv) h280 h288 (fun _ hi => by simp at hi)
  have hle0 := hle
  rcases hds_eq : ls.ds with _ | ⟨a, r⟩
  · simp [hds_eq] at hle0
  · dsimp only [PrimPostOutcome]
    simp only [Sporos.Prim.eff, List.take_succ_cons, List.take_zero, SporosProvLen.lexecPrim, hds_eq]
    simpa [hds_eq] using hfin

theorem P_swap_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_swap) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2)
    (h288 : rd64 s.mem (ea ctxBase 288) = some 2)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w)
    (h304 : rd64 s.mem (ea ctxBase 304) = some ls.ds[0]!.plen)
    (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w)
    (h320 : rd64 s.mem (ea ctxBase 320) = some ls.ds[1]!.plen) :
    PrimPostOutcome pr ls f rest s .swap := by
  have hrip' : s.rip = 4195501 := hrip
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hreg : RegsOk s := h.2.2.2.1
  have hbW := wf_ds_lt_W h (getElem!_mem_of_lt (by omega : 0 < ls.ds.length))
  have haW := wf_ds_lt_W h (getElem!_mem_of_lt (by omega : 1 < ls.ds.length))
  have h0 : X86Vm.step vmImage s = .ok { s.set 0 (some ls.ds[1]!.w) with rip := 4195508 } := by
    rw [step_at_chunk 3 hrip' rfl (by decide : (4195501, (.ofInt (.ld 0 15 312)), 7) ∈ SporosVmBin.vmTable_3),
      exec_ld h15 h312, hrip']
  let s1 := { s.set 0 (some ls.ds[1]!.w) with rip := 4195508 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) ls.ds[1]!.w
  have hs1_4 : s1.reg 4 = some vc.sp := by
    change (s.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hs1_0 : s1.reg 0 = some ls.ds[1]!.w := reg_set_self_ok' hreg (by decide) _
  have hwf1 : vc.wf pr ls s1 := vc.wf_step_misc h rfl hr1 hs1_4 hs1_15
  obtain ⟨s2, hs2, hrip2, hk2, _, _, hwf2, hrd360, hoth2⟩ :=
    step_wr_ctx_scratch (step_at_chunk 3 rfl rfl (by decide : (4195508, (.ofInt (.st 15 360 0)), 7) ∈ SporosVmBin.vmTable_3))
      hwf1 hs1_0 (by decide) (by decide)
  rw [Nat.mod_eq_of_lt haW.1] at hrd360
  have h320_2 := hoth2 320 _ (Or.inl (by decide)) (by decide) h320
  obtain ⟨s7, hrch7, hrip7, hk7, hwf7, hrd368, hoth7⟩ :=
    step_wr_ctx_tag_slot
      (fun t ht => step_at_chunk 3 (by rw [ht, hrip2]) rfl (by decide : (4195515, (.ofInt (.ld 0 15 320)), 7) ∈ SporosVmBin.vmTable_3))
      (fun t ht => step_at_chunk 3 (by rw [ht, hrip2]) rfl (by decide : (4195522, (.ofInt (.ld 1 15 224)), 7) ∈ SporosVmBin.vmTable_3))
      (fun t ht => step_at_chunk 3 (by rw [ht, hrip2]) rfl (by decide : (4195529, (.ofInt (.alu .add 0 1)), 3) ∈ SporosVmBin.vmTable_3))
      (fun t ht => step_at_chunk 3 (by rw [ht, hrip2]) rfl (by decide : (4195532, (.ofInt (.aluI .add 0 1)), 4) ∈ SporosVmBin.vmTable_3))
      (fun t ht => step_at_chunk 3 (by rw [ht, hrip2]) rfl (by decide : (4195536, (.ofInt (.st 15 368 0)), 7) ∈ SporosVmBin.vmTable_3))
      (nextAddr := 4195543) (by rw [hrip2]) hwf2 h320_2 (by decide) (by decide)
  have h296_7 := hoth7 296 _ (Or.inl (by decide)) (by decide) (hoth2 296 _ (Or.inl (by decide)) (by decide) h296)
  have h7_15 : s7.reg 15 = some ctxBase := hwf7.2.2.2.2.2.1
  have h7_4 : s7.reg 4 = some vc.sp := hwf7.2.2.2.2.1
  have hr7 : RegsOk s7 := hwf7.2.2.2.1
  have h7 : X86Vm.step vmImage s7 = .ok { s7.set 0 (some ls.ds[0]!.w) with rip := 4195550 } := by
    rw [step_at_chunk 3 hrip7 rfl (by decide : (4195543, (.ofInt (.ld 0 15 296)), 7) ∈ SporosVmBin.vmTable_3),
      exec_ld h7_15 h296_7, hrip7]
  let s8 := { s7.set 0 (some ls.ds[0]!.w) with rip := 4195550 }
  have hr8 : RegsOk s8 := regsOk_set hr7 (by decide) ls.ds[0]!.w
  have hs8_4 : s8.reg 4 = some vc.sp := by
    change (s7.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h7_4
  have hs8_15 : s8.reg 15 = some ctxBase := by
    change (s7.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h7_15
  have hs8_0 : s8.reg 0 = some ls.ds[0]!.w := reg_set_self_ok' hr7 (by decide) _
  have hwf8 := VmCtx.wf_step_misc (s' := s8) hwf7 rfl hr8 hs8_4 hs8_15
  obtain ⟨s9, hs9, hrip9, hk9, _, _, hwf9, hrd376, hoth9⟩ :=
    step_wr_ctx_scratch (step_at_chunk 3 rfl rfl (by decide : (4195550, (.ofInt (.st 15 376 0)), 7) ∈ SporosVmBin.vmTable_3))
      hwf8 hs8_0 (by decide) (by decide)
  rw [Nat.mod_eq_of_lt hbW.1] at hrd376
  have h304_9 := hoth9 304 _ (Or.inl (by decide)) (by decide)
    (hoth7 304 _ (Or.inl (by decide)) (by decide) (hoth2 304 _ (Or.inl (by decide)) (by decide) h304))
  obtain ⟨s14, hrch14, hrip14, hk14, hwf14, hrd384, hoth14⟩ :=
    step_wr_ctx_tag_slot
      (fun t ht => step_at_chunk 3 (by rw [ht, hrip9]) rfl (by decide : (4195557, (.ofInt (.ld 0 15 304)), 7) ∈ SporosVmBin.vmTable_3))
      (fun t ht => step_at_chunk 3 (by rw [ht, hrip9]) rfl (by decide : (4195564, (.ofInt (.ld 1 15 224)), 7) ∈ SporosVmBin.vmTable_3))
      (fun t ht => step_at_chunk 3 (by rw [ht, hrip9]) rfl (by decide : (4195571, (.ofInt (.alu .add 0 1)), 3) ∈ SporosVmBin.vmTable_3))
      (fun t ht => step_at_chunk 3 (by rw [ht, hrip9]) rfl (by decide : (4195574, (.ofInt (.aluI .add 0 1)), 4) ∈ SporosVmBin.vmTable_3))
      (fun t ht => step_at_chunk 3 (by rw [ht, hrip9]) rfl (by decide : (4195578, (.ofInt (.st 15 384 0)), 7) ∈ SporosVmBin.vmTable_3))
      (nextAddr := 4195585) (by rw [hrip9]) hwf9 h304_9 (by decide) (by decide)
  have hrch14_tot : Reaches vmImage s s14 :=
    Reaches.trans (Reaches.step_trans h0 (Reaches.step hs2))
      (Reaches.trans hrch7 (Reaches.trans (Reaches.step_trans h7 (Reaches.step hs9)) hrch14))
  have hk_tot : s14.k = s.k := by
    dsimp only [s8, s1, X86Int.St.set] at hk14 hk9 hk7 hk2 ⊢
    rw [hk14, hk9, hk7, hk2]
  have h280_14 : rd64 s14.mem (ea ctxBase 280) = some 2 :=
    hoth14 280 _ (Or.inl (by decide)) (by decide) (hoth9 280 _ (Or.inl (by decide)) (by decide)
      (hoth7 280 _ (Or.inl (by decide)) (by decide) (hoth2 280 _ (Or.inl (by decide)) (by decide) h280)))
  have h288_14 : rd64 s14.mem (ea ctxBase 288) = some 2 :=
    hoth14 288 _ (Or.inl (by decide)) (by decide) (hoth9 288 _ (Or.inl (by decide)) (by decide)
      (hoth7 288 _ (Or.inl (by decide)) (by decide) (hoth2 288 _ (Or.inl (by decide)) (by decide) h288)))
  have hrd360_14 : rd64 s14.mem (ea ctxBase 360) = some ls.ds[1]!.w :=
    hoth14 360 _ (Or.inl (by decide)) (by decide) (hoth9 360 _ (Or.inl (by decide)) (by decide)
      (hoth7 360 _ (Or.inl (by decide)) (by decide) hrd360))
  have hrd368_14 : rd64 s14.mem (ea ctxBase 368) = some ((ls.ds[1]!.plen + ls.pc + 1) % Sporos.M) :=
    hoth14 368 _ (Or.inl (by decide)) (by decide) (hoth9 368 _ (Or.inl (by decide)) (by decide) hrd368)
  have hrd376_14 : rd64 s14.mem (ea ctxBase 376) = some ls.ds[0]!.w :=
    hoth14 376 _ (Or.inl (by decide)) (by decide) hrd376
  let ta := SporosProvLen.ltag ls.pc ls.ds[1]!
  let tb := SporosProvLen.ltag ls.pc ls.ds[0]!
  have houtsW : ∀ v ∈ ([ta, tb] : List SporosProvLen.LVal), v.w < X86Env.W ∧ v.plen < X86Env.W := by
    intro v hv; simp only [List.mem_cons, List.not_mem_nil, or_false] at hv
    rcases hv with rfl | rfl
    · exact ⟨haW.1, by dsimp [ta, SporosProvLen.ltag]; rw [M_eq_W']; exact lt_W_of_mod _⟩
    · exact ⟨hbW.1, by dsimp [tb, SporosProvLen.ltag]; rw [M_eq_W']; exact lt_W_of_mod _⟩
  have hslots : ∀ (i : Nat) (_ : i < ([ta, tb] : List SporosProvLen.LVal).length),
      rd64 s14.mem (ea ctxBase (360 + i * 16)) = some ([ta, tb][i]).w ∧
      rd64 s14.mem (ea ctxBase (368 + i * 16)) = some ([ta, tb][i]).plen := by
    intro i hi; rcases i with _ | _ | i
    · exact ⟨hrd360_14, hrd368_14⟩
    · exact ⟨hrd376_14, hrd384⟩
    · simp only [List.length_cons, List.length_nil] at hi; omega
  have hfin := jmp_epilogue0_finish_spec
    (step_at_chunk 3 hrip14 rfl (by decide : (4195585, (.ofInt (.jmp 4195233)), 5) ∈ SporosVmBin.vmTable_3))
    hwf14 (oracleAdmits_of_reaches hor hrch14_tot) hrs hpcW hle (by simp) houtsW h280_14 h288_14 hslots
  have hres := lift_next_outcome hrch14_tot hk_tot hfin
  rcases hds_eq : ls.ds with _ | ⟨b, _ | ⟨a, r⟩⟩ <;> try (simp [hds_eq] at hle)
  dsimp only [PrimPostOutcome]
  simp only [Sporos.Prim.eff, List.take_succ_cons, List.take_zero, SporosProvLen.lexecPrim, hds_eq, List.drop_succ_cons, List.drop_zero]
  simpa only [ta, tb, hds_eq, List.getElem!_cons_zero, List.getElem!_cons_succ, List.drop_succ_cons, List.drop_zero] using hres

theorem P_over_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_over) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2)
    (h288 : rd64 s.mem (ea ctxBase 288) = some 3)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w)
    (h304 : rd64 s.mem (ea ctxBase 304) = some ls.ds[0]!.plen)
    (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w)
    (h320 : rd64 s.mem (ea ctxBase 320) = some ls.ds[1]!.plen) :
    PrimPostOutcome pr ls f rest s .over := by
  have hrip' : s.rip = 4195590 := hrip
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hreg : RegsOk s := h.2.2.2.1
  have hbW := wf_ds_lt_W h (getElem!_mem_of_lt (by omega : 0 < ls.ds.length))
  have haW := wf_ds_lt_W h (getElem!_mem_of_lt (by omega : 1 < ls.ds.length))
  have h0 : X86Vm.step vmImage s = .ok { s.set 0 (some ls.ds[1]!.w) with rip := 4195597 } := by
    rw [step_at_chunk 3 hrip' rfl (by decide : (4195590, (.ofInt (.ld 0 15 312)), 7) ∈ SporosVmBin.vmTable_3),
      exec_ld h15 h312, hrip']
  let s1 := { s.set 0 (some ls.ds[1]!.w) with rip := 4195597 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) ls.ds[1]!.w
  have hs1_4 : s1.reg 4 = some vc.sp := by
    change (s.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hs1_0 : s1.reg 0 = some ls.ds[1]!.w := reg_set_self_ok' hreg (by decide) _
  have hwf1 : vc.wf pr ls s1 := vc.wf_step_misc h rfl hr1 hs1_4 hs1_15
  obtain ⟨s2, hs2, hrip2, hk2, _, _, hwf2, hrd360, hoth2⟩ :=
    step_wr_ctx_scratch (step_at_chunk 3 rfl rfl (by decide : (4195597, (.ofInt (.st 15 360 0)), 7) ∈ SporosVmBin.vmTable_3))
      hwf1 hs1_0 (by decide) (by decide)
  rw [Nat.mod_eq_of_lt haW.1] at hrd360
  have h320_2 := hoth2 320 _ (Or.inl (by decide)) (by decide) h320
  obtain ⟨s7, hrch7, hrip7, hk7, hwf7, hrd368, hoth7⟩ :=
    step_wr_ctx_tag_slot
      (fun t ht => step_at_chunk 3 (by rw [ht, hrip2]) rfl (by decide : (4195604, (.ofInt (.ld 0 15 320)), 7) ∈ SporosVmBin.vmTable_3))
      (fun t ht => step_at_chunk 3 (by rw [ht, hrip2]) rfl (by decide : (4195611, (.ofInt (.ld 1 15 224)), 7) ∈ SporosVmBin.vmTable_3))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip2]) rfl (by decide : (4195618, (.ofInt (.alu .add 0 1)), 3) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip2]) rfl (by decide : (4195621, (.ofInt (.aluI .add 0 1)), 4) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip2]) rfl (by decide : (4195625, (.ofInt (.st 15 368 0)), 7) ∈ SporosVmBin.vmTable_4))
      (nextAddr := 4195632) (by rw [hrip2]) hwf2 h320_2 (by decide) (by decide)
  have h296_7 := hoth7 296 _ (Or.inl (by decide)) (by decide) (hoth2 296 _ (Or.inl (by decide)) (by decide) h296)
  have h7_15 : s7.reg 15 = some ctxBase := hwf7.2.2.2.2.2.1
  have h7_4 : s7.reg 4 = some vc.sp := hwf7.2.2.2.2.1
  have hr7 : RegsOk s7 := hwf7.2.2.2.1
  have h7 : X86Vm.step vmImage s7 = .ok { s7.set 0 (some ls.ds[0]!.w) with rip := 4195639 } := by
    rw [step_at_chunk 4 hrip7 rfl (by decide : (4195632, (.ofInt (.ld 0 15 296)), 7) ∈ SporosVmBin.vmTable_4),
      exec_ld h7_15 h296_7, hrip7]
  let s8 := { s7.set 0 (some ls.ds[0]!.w) with rip := 4195639 }
  have hr8 : RegsOk s8 := regsOk_set hr7 (by decide) ls.ds[0]!.w
  have hs8_4 : s8.reg 4 = some vc.sp := by
    change (s7.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h7_4
  have hs8_15 : s8.reg 15 = some ctxBase := by
    change (s7.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h7_15
  have hs8_0 : s8.reg 0 = some ls.ds[0]!.w := reg_set_self_ok' hr7 (by decide) _
  have hwf8 := VmCtx.wf_step_misc (s' := s8) hwf7 rfl hr8 hs8_4 hs8_15
  obtain ⟨s9, hs9, hrip9, hk9, _, _, hwf9, hrd376, hoth9⟩ :=
    step_wr_ctx_scratch (step_at_chunk 4 rfl rfl (by decide : (4195639, (.ofInt (.st 15 376 0)), 7) ∈ SporosVmBin.vmTable_4))
      hwf8 hs8_0 (by decide) (by decide)
  rw [Nat.mod_eq_of_lt hbW.1] at hrd376
  have h304_9 := hoth9 304 _ (Or.inl (by decide)) (by decide)
    (hoth7 304 _ (Or.inl (by decide)) (by decide) (hoth2 304 _ (Or.inl (by decide)) (by decide) h304))
  obtain ⟨s14, hrch14, hrip14, hk14, hwf14, hrd384, hoth14⟩ :=
    step_wr_ctx_tag_slot
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip9]) rfl (by decide : (4195646, (.ofInt (.ld 0 15 304)), 7) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip9]) rfl (by decide : (4195653, (.ofInt (.ld 1 15 224)), 7) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip9]) rfl (by decide : (4195660, (.ofInt (.alu .add 0 1)), 3) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip9]) rfl (by decide : (4195663, (.ofInt (.aluI .add 0 1)), 4) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip9]) rfl (by decide : (4195667, (.ofInt (.st 15 384 0)), 7) ∈ SporosVmBin.vmTable_4))
      (nextAddr := 4195674) (by rw [hrip9]) hwf9 h304_9 (by decide) (by decide)
  -- copy 360 -> 392 and 368 -> 400
  have hrd360_14 := hoth14 360 _ (Or.inl (by decide)) (by decide)
    (hoth9 360 _ (Or.inl (by decide)) (by decide) (hoth7 360 _ (Or.inl (by decide)) (by decide) hrd360))
  have hrd368_14 := hoth14 368 _ (Or.inl (by decide)) (by decide)
    (hoth9 368 _ (Or.inl (by decide)) (by decide) hrd368)
  have h14_15 : s14.reg 15 = some ctxBase := hwf14.2.2.2.2.2.1
  have h14_4 : s14.reg 4 = some vc.sp := hwf14.2.2.2.2.1
  have hr14 : RegsOk s14 := hwf14.2.2.2.1
  have h14 : X86Vm.step vmImage s14 = .ok { s14.set 0 (some ls.ds[1]!.w) with rip := 4195681 } := by
    rw [step_at_chunk 4 hrip14 rfl (by decide : (4195674, (.ofInt (.ld 0 15 360)), 7) ∈ SporosVmBin.vmTable_4),
      exec_ld h14_15 hrd360_14, hrip14]
  let s15 := { s14.set 0 (some ls.ds[1]!.w) with rip := 4195681 }
  have hr15 : RegsOk s15 := regsOk_set hr14 (by decide) ls.ds[1]!.w
  have hs15_4 : s15.reg 4 = some vc.sp := by
    change (s14.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h14_4
  have hs15_15 : s15.reg 15 = some ctxBase := by
    change (s14.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h14_15
  have hs15_0 : s15.reg 0 = some ls.ds[1]!.w := reg_set_self_ok' hr14 (by decide) _
  have hwf15 := VmCtx.wf_step_misc (s' := s15) hwf14 rfl hr15 hs15_4 hs15_15
  obtain ⟨s16, hs16, hrip16, hk16, _, _, hwf16, hrd392, hoth16⟩ :=
    step_wr_ctx_scratch (step_at_chunk 4 rfl rfl (by decide : (4195681, (.ofInt (.st 15 392 0)), 7) ∈ SporosVmBin.vmTable_4))
      hwf15 hs15_0 (by decide) (by decide)
  rw [Nat.mod_eq_of_lt haW.1] at hrd392
  have hrd368_16 := hoth16 368 _ (Or.inl (by decide)) (by decide) hrd368_14
  have h16_15 : s16.reg 15 = some ctxBase := hwf16.2.2.2.2.2.1
  have h16_4 : s16.reg 4 = some vc.sp := hwf16.2.2.2.2.1
  have hr16 : RegsOk s16 := hwf16.2.2.2.1
  have h16 : X86Vm.step vmImage s16 =
      .ok { s16.set 0 (some ((ls.ds[1]!.plen + ls.pc + 1) % Sporos.M)) with rip := 4195695 } := by
    rw [step_at_chunk 4 hrip16 rfl (by decide : (4195688, (.ofInt (.ld 0 15 368)), 7) ∈ SporosVmBin.vmTable_4),
      exec_ld h16_15 hrd368_16, hrip16]
  let s17 := { s16.set 0 (some ((ls.ds[1]!.plen + ls.pc + 1) % Sporos.M)) with rip := 4195695 }
  have hr17 : RegsOk s17 := regsOk_set hr16 (by decide) _
  have hs17_4 : s17.reg 4 = some vc.sp := by
    change (s16.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h16_4
  have hs17_15 : s17.reg 15 = some ctxBase := by
    change (s16.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h16_15
  have hs17_0 : s17.reg 0 = some ((ls.ds[1]!.plen + ls.pc + 1) % Sporos.M) := reg_set_self_ok' hr16 (by decide) _
  have hwf17 := VmCtx.wf_step_misc (s' := s17) hwf16 rfl hr17 hs17_4 hs17_15
  obtain ⟨s18, hs18, hrip18, hk18, _, _, hwf18, hrd400, hoth18⟩ :=
    step_wr_ctx_scratch (step_at_chunk 4 rfl rfl (by decide : (4195695, (.ofInt (.st 15 400 0)), 7) ∈ SporosVmBin.vmTable_4))
      hwf17 hs17_0 (by decide) (by decide)
  rw [M_eq_W', Nat.mod_mod] at hrd400
  have hrch18 : Reaches vmImage s s18 :=
    Reaches.trans (Reaches.step_trans h0 (Reaches.step hs2))
      (Reaches.trans hrch7 (Reaches.trans (Reaches.step_trans h7 (Reaches.step hs9))
        (Reaches.trans hrch14 (Reaches.step_trans h14 (Reaches.step_trans hs16 (Reaches.step_trans h16 (Reaches.step hs18)))))))
  have hk_tot : s18.k = s.k := by
    dsimp only [s17, s15, s8, s1, X86Int.St.set] at hk18 hk16 hk14 hk9 hk7 hk2 ⊢
    rw [hk18, hk16, hk14, hk9, hk7, hk2]
  have h280_18 := hoth18 280 _ (Or.inl (by decide)) (by decide) (hoth16 280 _ (Or.inl (by decide)) (by decide)
    (hoth14 280 _ (Or.inl (by decide)) (by decide) (hoth9 280 _ (Or.inl (by decide)) (by decide)
      (hoth7 280 _ (Or.inl (by decide)) (by decide) (hoth2 280 _ (Or.inl (by decide)) (by decide) h280)))))
  have h288_18 := hoth18 288 _ (Or.inl (by decide)) (by decide) (hoth16 288 _ (Or.inl (by decide)) (by decide)
    (hoth14 288 _ (Or.inl (by decide)) (by decide) (hoth9 288 _ (Or.inl (by decide)) (by decide)
      (hoth7 288 _ (Or.inl (by decide)) (by decide) (hoth2 288 _ (Or.inl (by decide)) (by decide) h288)))))
  have hrd360_18 := hoth18 360 _ (Or.inl (by decide)) (by decide) (hoth16 360 _ (Or.inl (by decide)) (by decide) hrd360_14)
  have hrd368_18 := hoth18 368 _ (Or.inl (by decide)) (by decide) hrd368_16
  have hrd376_18 := hoth18 376 _ (Or.inl (by decide)) (by decide) (hoth16 376 _ (Or.inl (by decide)) (by decide)
    (hoth14 376 _ (Or.inl (by decide)) (by decide) hrd376))
  have hrd384_18 := hoth18 384 _ (Or.inl (by decide)) (by decide) (hoth16 384 _ (Or.inl (by decide)) (by decide) hrd384)
  have hrd392_18 := hoth18 392 _ (Or.inl (by decide)) (by decide) hrd392
  let ta := SporosProvLen.ltag ls.pc ls.ds[1]!
  let tb := SporosProvLen.ltag ls.pc ls.ds[0]!
  have houtsW : ∀ v ∈ ([ta, tb, ta] : List SporosProvLen.LVal), v.w < X86Env.W ∧ v.plen < X86Env.W := by
    intro v hv; simp only [List.mem_cons, List.not_mem_nil, or_false] at hv
    rcases hv with rfl | rfl | rfl
    · exact ⟨haW.1, by dsimp [ta, SporosProvLen.ltag]; rw [M_eq_W']; exact lt_W_of_mod _⟩
    · exact ⟨hbW.1, by dsimp [tb, SporosProvLen.ltag]; rw [M_eq_W']; exact lt_W_of_mod _⟩
    · exact ⟨haW.1, by dsimp [ta, SporosProvLen.ltag]; rw [M_eq_W']; exact lt_W_of_mod _⟩
  have hslots : ∀ (i : Nat) (_ : i < ([ta, tb, ta] : List SporosProvLen.LVal).length),
      rd64 s18.mem (ea ctxBase (360 + i * 16)) = some ([ta, tb, ta][i]).w ∧
      rd64 s18.mem (ea ctxBase (368 + i * 16)) = some ([ta, tb, ta][i]).plen := by
    intro i hi; rcases i with _ | _ | _ | i
    · exact ⟨hrd360_18, hrd368_18⟩
    · exact ⟨hrd376_18, hrd384_18⟩
    · exact ⟨hrd392_18, hrd400⟩
    · simp only [List.length_cons, List.length_nil] at hi; omega
  have hfin := jmp_epilogue0_finish_spec
    (step_at_chunk 4 hrip18 rfl (by decide : (4195702, (.ofInt (.jmp 4195233)), 5) ∈ SporosVmBin.vmTable_4))
    hwf18 (oracleAdmits_of_reaches hor hrch18) hrs hpcW hle (by simp) houtsW h280_18 h288_18 hslots
  have hres := lift_next_outcome hrch18 hk_tot hfin
  rcases hds_eq : ls.ds with _ | ⟨b, _ | ⟨a, r⟩⟩ <;> try (simp [hds_eq] at hle)
  dsimp only [PrimPostOutcome]
  simp only [Sporos.Prim.eff, List.take_succ_cons, List.take_zero, SporosProvLen.lexecPrim, hds_eq, List.drop_succ_cons, List.drop_zero]
  simpa only [ta, tb, hds_eq, List.getElem!_cons_zero, List.getElem!_cons_succ, List.drop_succ_cons, List.drop_zero] using hres

theorem P_rot_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_rot) (hle : 3 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h280 : rd64 s.mem (ea ctxBase 280) = some 3)
    (h288 : rd64 s.mem (ea ctxBase 288) = some 3)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w)
    (h304 : rd64 s.mem (ea ctxBase 304) = some ls.ds[0]!.plen)
    (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w)
    (h320 : rd64 s.mem (ea ctxBase 320) = some ls.ds[1]!.plen)
    (h328 : rd64 s.mem (ea ctxBase 328) = some ls.ds[2]!.w)
    (h336 : rd64 s.mem (ea ctxBase 336) = some ls.ds[2]!.plen) :
    PrimPostOutcome pr ls f rest s .rot := by
  have hrip' : s.rip = 4195707 := hrip
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hreg : RegsOk s := h.2.2.2.1
  have hcW := wf_ds_lt_W h (getElem!_mem_of_lt (by omega : 0 < ls.ds.length))
  have hbW := wf_ds_lt_W h (getElem!_mem_of_lt (by omega : 1 < ls.ds.length))
  have haW := wf_ds_lt_W h (getElem!_mem_of_lt (by omega : 2 < ls.ds.length))
  -- 328 -> 360, tag 336 -> 368
  have h0 : X86Vm.step vmImage s = .ok { s.set 0 (some ls.ds[2]!.w) with rip := 4195714 } := by
    rw [step_at_chunk 4 hrip' rfl (by decide : (4195707, (.ofInt (.ld 0 15 328)), 7) ∈ SporosVmBin.vmTable_4),
      exec_ld h15 h328, hrip']
  let s1 := { s.set 0 (some ls.ds[2]!.w) with rip := 4195714 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) ls.ds[2]!.w
  have hs1_4 : s1.reg 4 = some vc.sp := by
    change (s.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hs1_0 : s1.reg 0 = some ls.ds[2]!.w := reg_set_self_ok' hreg (by decide) _
  have hwf1 : vc.wf pr ls s1 := vc.wf_step_misc h rfl hr1 hs1_4 hs1_15
  obtain ⟨s2, hs2, hrip2, hk2, _, _, hwf2, hrd360, hoth2⟩ :=
    step_wr_ctx_scratch (step_at_chunk 4 rfl rfl (by decide : (4195714, (.ofInt (.st 15 360 0)), 7) ∈ SporosVmBin.vmTable_4))
      hwf1 hs1_0 (by decide) (by decide)
  rw [Nat.mod_eq_of_lt haW.1] at hrd360
  have h336_2 := hoth2 336 _ (Or.inl (by decide)) (by decide) h336
  obtain ⟨s7, hrch7, hrip7, hk7, hwf7, hrd368, hoth7⟩ :=
    step_wr_ctx_tag_slot
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip2]) rfl (by decide : (4195721, (.ofInt (.ld 0 15 336)), 7) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip2]) rfl (by decide : (4195728, (.ofInt (.ld 1 15 224)), 7) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip2]) rfl (by decide : (4195735, (.ofInt (.alu .add 0 1)), 3) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip2]) rfl (by decide : (4195738, (.ofInt (.aluI .add 0 1)), 4) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip2]) rfl (by decide : (4195742, (.ofInt (.st 15 368 0)), 7) ∈ SporosVmBin.vmTable_4))
      (nextAddr := 4195749) (by rw [hrip2]) hwf2 h336_2 (by decide) (by decide)
  -- 296 -> 376, tag 304 -> 384
  have h296_7 := hoth7 296 _ (Or.inl (by decide)) (by decide) (hoth2 296 _ (Or.inl (by decide)) (by decide) h296)
  have h7_15 : s7.reg 15 = some ctxBase := hwf7.2.2.2.2.2.1
  have h7_4 : s7.reg 4 = some vc.sp := hwf7.2.2.2.2.1
  have hr7 : RegsOk s7 := hwf7.2.2.2.1
  have h7 : X86Vm.step vmImage s7 = .ok { s7.set 0 (some ls.ds[0]!.w) with rip := 4195756 } := by
    rw [step_at_chunk 4 hrip7 rfl (by decide : (4195749, (.ofInt (.ld 0 15 296)), 7) ∈ SporosVmBin.vmTable_4),
      exec_ld h7_15 h296_7, hrip7]
  let s8 := { s7.set 0 (some ls.ds[0]!.w) with rip := 4195756 }
  have hr8 : RegsOk s8 := regsOk_set hr7 (by decide) ls.ds[0]!.w
  have hs8_4 : s8.reg 4 = some vc.sp := by
    change (s7.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h7_4
  have hs8_15 : s8.reg 15 = some ctxBase := by
    change (s7.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h7_15
  have hs8_0 : s8.reg 0 = some ls.ds[0]!.w := reg_set_self_ok' hr7 (by decide) _
  have hwf8 := VmCtx.wf_step_misc (s' := s8) hwf7 rfl hr8 hs8_4 hs8_15
  obtain ⟨s9, hs9, hrip9, hk9, _, _, hwf9, hrd376, hoth9⟩ :=
    step_wr_ctx_scratch (step_at_chunk 4 rfl rfl (by decide : (4195756, (.ofInt (.st 15 376 0)), 7) ∈ SporosVmBin.vmTable_4))
      hwf8 hs8_0 (by decide) (by decide)
  rw [Nat.mod_eq_of_lt hcW.1] at hrd376
  have h304_9 := hoth9 304 _ (Or.inl (by decide)) (by decide)
    (hoth7 304 _ (Or.inl (by decide)) (by decide) (hoth2 304 _ (Or.inl (by decide)) (by decide) h304))
  obtain ⟨s14, hrch14, hrip14, hk14, hwf14, hrd384, hoth14⟩ :=
    step_wr_ctx_tag_slot
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip9]) rfl (by decide : (4195763, (.ofInt (.ld 0 15 304)), 7) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip9]) rfl (by decide : (4195770, (.ofInt (.ld 1 15 224)), 7) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip9]) rfl (by decide : (4195777, (.ofInt (.alu .add 0 1)), 3) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip9]) rfl (by decide : (4195780, (.ofInt (.aluI .add 0 1)), 4) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip9]) rfl (by decide : (4195784, (.ofInt (.st 15 384 0)), 7) ∈ SporosVmBin.vmTable_4))
      (nextAddr := 4195791) (by rw [hrip9]) hwf9 h304_9 (by decide) (by decide)
  -- 312 -> 392, tag 320 -> 400
  have h312_14 := hoth14 312 _ (Or.inl (by decide)) (by decide)
    (hoth9 312 _ (Or.inl (by decide)) (by decide)
      (hoth7 312 _ (Or.inl (by decide)) (by decide) (hoth2 312 _ (Or.inl (by decide)) (by decide) h312)))
  have h14_15 : s14.reg 15 = some ctxBase := hwf14.2.2.2.2.2.1
  have h14_4 : s14.reg 4 = some vc.sp := hwf14.2.2.2.2.1
  have hr14 : RegsOk s14 := hwf14.2.2.2.1
  have h14 : X86Vm.step vmImage s14 = .ok { s14.set 0 (some ls.ds[1]!.w) with rip := 4195798 } := by
    rw [step_at_chunk 4 hrip14 rfl (by decide : (4195791, (.ofInt (.ld 0 15 312)), 7) ∈ SporosVmBin.vmTable_4),
      exec_ld h14_15 h312_14, hrip14]
  let s15 := { s14.set 0 (some ls.ds[1]!.w) with rip := 4195798 }
  have hr15 : RegsOk s15 := regsOk_set hr14 (by decide) ls.ds[1]!.w
  have hs15_4 : s15.reg 4 = some vc.sp := by
    change (s14.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h14_4
  have hs15_15 : s15.reg 15 = some ctxBase := by
    change (s14.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h14_15
  have hs15_0 : s15.reg 0 = some ls.ds[1]!.w := reg_set_self_ok' hr14 (by decide) _
  have hwf15 := VmCtx.wf_step_misc (s' := s15) hwf14 rfl hr15 hs15_4 hs15_15
  obtain ⟨s16, hs16, hrip16, hk16, _, _, hwf16, hrd392, hoth16⟩ :=
    step_wr_ctx_scratch (step_at_chunk 4 rfl rfl (by decide : (4195798, (.ofInt (.st 15 392 0)), 7) ∈ SporosVmBin.vmTable_4))
      hwf15 hs15_0 (by decide) (by decide)
  rw [Nat.mod_eq_of_lt hbW.1] at hrd392
  have h320_16 := hoth16 320 _ (Or.inl (by decide)) (by decide)
    (hoth14 320 _ (Or.inl (by decide)) (by decide)
      (hoth9 320 _ (Or.inl (by decide)) (by decide)
        (hoth7 320 _ (Or.inl (by decide)) (by decide) (hoth2 320 _ (Or.inl (by decide)) (by decide) h320))))
  obtain ⟨s21, hrch21, hrip21, hk21, hwf21, hrd400, hoth21⟩ :=
    step_wr_ctx_tag_slot
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip16]) rfl (by decide : (4195805, (.ofInt (.ld 0 15 320)), 7) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip16]) rfl (by decide : (4195812, (.ofInt (.ld 1 15 224)), 7) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip16]) rfl (by decide : (4195819, (.ofInt (.alu .add 0 1)), 3) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip16]) rfl (by decide : (4195822, (.ofInt (.aluI .add 0 1)), 4) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip16]) rfl (by decide : (4195826, (.ofInt (.st 15 400 0)), 7) ∈ SporosVmBin.vmTable_4))
      (nextAddr := 4195833) (by rw [hrip16]) hwf16 h320_16 (by decide) (by decide)
  have hrch21_tot : Reaches vmImage s s21 :=
    Reaches.trans (Reaches.step_trans h0 (Reaches.step hs2))
      (Reaches.trans hrch7 (Reaches.trans (Reaches.step_trans h7 (Reaches.step hs9))
        (Reaches.trans hrch14 (Reaches.trans (Reaches.step_trans h14 (Reaches.step hs16)) hrch21))))
  have hk_tot : s21.k = s.k := by
    dsimp only [s15, s8, s1, X86Int.St.set] at hk21 hk16 hk14 hk9 hk7 hk2 ⊢
    rw [hk21, hk16, hk14, hk9, hk7, hk2]
  have h280_21 := hoth21 280 _ (Or.inl (by decide)) (by decide) (hoth16 280 _ (Or.inl (by decide)) (by decide)
    (hoth14 280 _ (Or.inl (by decide)) (by decide) (hoth9 280 _ (Or.inl (by decide)) (by decide)
      (hoth7 280 _ (Or.inl (by decide)) (by decide) (hoth2 280 _ (Or.inl (by decide)) (by decide) h280)))))
  have h288_21 := hoth21 288 _ (Or.inl (by decide)) (by decide) (hoth16 288 _ (Or.inl (by decide)) (by decide)
    (hoth14 288 _ (Or.inl (by decide)) (by decide) (hoth9 288 _ (Or.inl (by decide)) (by decide)
      (hoth7 288 _ (Or.inl (by decide)) (by decide) (hoth2 288 _ (Or.inl (by decide)) (by decide) h288)))))
  have hrd360_21 := hoth21 360 _ (Or.inl (by decide)) (by decide) (hoth16 360 _ (Or.inl (by decide)) (by decide)
    (hoth14 360 _ (Or.inl (by decide)) (by decide) (hoth9 360 _ (Or.inl (by decide)) (by decide)
      (hoth7 360 _ (Or.inl (by decide)) (by decide) hrd360))))
  have hrd368_21 := hoth21 368 _ (Or.inl (by decide)) (by decide) (hoth16 368 _ (Or.inl (by decide)) (by decide)
    (hoth14 368 _ (Or.inl (by decide)) (by decide) (hoth9 368 _ (Or.inl (by decide)) (by decide) hrd368)))
  have hrd376_21 := hoth21 376 _ (Or.inl (by decide)) (by decide) (hoth16 376 _ (Or.inl (by decide)) (by decide)
    (hoth14 376 _ (Or.inl (by decide)) (by decide) hrd376))
  have hrd384_21 := hoth21 384 _ (Or.inl (by decide)) (by decide) (hoth16 384 _ (Or.inl (by decide)) (by decide) hrd384)
  have hrd392_21 := hoth21 392 _ (Or.inl (by decide)) (by decide) hrd392
  let ta := SporosProvLen.ltag ls.pc ls.ds[2]!
  let tc := SporosProvLen.ltag ls.pc ls.ds[0]!
  let tb := SporosProvLen.ltag ls.pc ls.ds[1]!
  have houtsW : ∀ v ∈ ([ta, tc, tb] : List SporosProvLen.LVal), v.w < X86Env.W ∧ v.plen < X86Env.W := by
    intro v hv; simp only [List.mem_cons, List.not_mem_nil, or_false] at hv
    rcases hv with rfl | rfl | rfl
    · exact ⟨haW.1, by dsimp [ta, SporosProvLen.ltag]; rw [M_eq_W']; exact lt_W_of_mod _⟩
    · exact ⟨hcW.1, by dsimp [tc, SporosProvLen.ltag]; rw [M_eq_W']; exact lt_W_of_mod _⟩
    · exact ⟨hbW.1, by dsimp [tb, SporosProvLen.ltag]; rw [M_eq_W']; exact lt_W_of_mod _⟩
  have hslots : ∀ (i : Nat) (_ : i < ([ta, tc, tb] : List SporosProvLen.LVal).length),
      rd64 s21.mem (ea ctxBase (360 + i * 16)) = some ([ta, tc, tb][i]).w ∧
      rd64 s21.mem (ea ctxBase (368 + i * 16)) = some ([ta, tc, tb][i]).plen := by
    intro i hi; rcases i with _ | _ | _ | i
    · exact ⟨hrd360_21, hrd368_21⟩
    · exact ⟨hrd376_21, hrd384_21⟩
    · exact ⟨hrd392_21, hrd400⟩
    · simp only [List.length_cons, List.length_nil] at hi; omega
  have hfin := jmp_epilogue0_finish_spec
    (step_at_chunk 4 hrip21 rfl (by decide : (4195833, (.ofInt (.jmp 4195233)), 5) ∈ SporosVmBin.vmTable_4))
    hwf21 (oracleAdmits_of_reaches hor hrch21_tot) hrs hpcW hle (by simp) houtsW h280_21 h288_21 hslots
  have hres := lift_next_outcome hrch21_tot hk_tot hfin
  rcases hds_eq : ls.ds with _ | ⟨c, _ | ⟨b, _ | ⟨a, r⟩⟩⟩ <;> try (simp [hds_eq] at hle)
  dsimp only [PrimPostOutcome]
  simp only [Sporos.Prim.eff, List.take_succ_cons, List.take_zero, SporosProvLen.lexecPrim, hds_eq, List.drop_succ_cons, List.drop_zero]
  simpa only [ta, tc, tb, hds_eq, List.getElem!_cons_zero, List.getElem!_cons_succ, List.drop_succ_cons, List.drop_zero] using hres

theorem P_nip_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_nip) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2)
    (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w)
    (h304 : rd64 s.mem (ea ctxBase 304) = some ls.ds[0]!.plen) :
    PrimPostOutcome pr ls f rest s .nip := by
  have hrip' : s.rip = 4195838 := hrip
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hreg : RegsOk s := h.2.2.2.1
  have hbW := wf_ds_lt_W h (getElem!_mem_of_lt (by omega : 0 < ls.ds.length))
  have h0 : X86Vm.step vmImage s = .ok { s.set 0 (some ls.ds[0]!.w) with rip := 4195845 } := by
    rw [step_at_chunk 4 hrip' rfl (by decide : (4195838, (.ofInt (.ld 0 15 296)), 7) ∈ SporosVmBin.vmTable_4),
      exec_ld h15 h296, hrip']
  let s1 := { s.set 0 (some ls.ds[0]!.w) with rip := 4195845 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) ls.ds[0]!.w
  have hs1_4 : s1.reg 4 = some vc.sp := by
    change (s.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hs1_0 : s1.reg 0 = some ls.ds[0]!.w := reg_set_self_ok' hreg (by decide) _
  have hwf1 : vc.wf pr ls s1 := vc.wf_step_misc h rfl hr1 hs1_4 hs1_15
  obtain ⟨s2, hs2, hrip2, hk2, _, _, hwf2, hrd360, hoth2⟩ :=
    step_wr_ctx_scratch (step_at_chunk 4 rfl rfl (by decide : (4195845, (.ofInt (.st 15 360 0)), 7) ∈ SporosVmBin.vmTable_4))
      hwf1 hs1_0 (by decide) (by decide)
  rw [Nat.mod_eq_of_lt hbW.1] at hrd360
  have h304_2 := hoth2 304 _ (Or.inl (by decide)) (by decide) h304
  obtain ⟨s7, hrch7, hrip7, hk7, hwf7, hrd368, hoth7⟩ :=
    step_wr_ctx_tag_slot
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip2]) rfl (by decide : (4195852, (.ofInt (.ld 0 15 304)), 7) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip2]) rfl (by decide : (4195859, (.ofInt (.ld 1 15 224)), 7) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip2]) rfl (by decide : (4195866, (.ofInt (.alu .add 0 1)), 3) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip2]) rfl (by decide : (4195869, (.ofInt (.aluI .add 0 1)), 4) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip2]) rfl (by decide : (4195873, (.ofInt (.st 15 368 0)), 7) ∈ SporosVmBin.vmTable_4))
      (nextAddr := 4195880) (by rw [hrip2]) hwf2 h304_2 (by decide) (by decide)
  have hrch7_tot : Reaches vmImage s s7 := Reaches.trans (Reaches.step_trans h0 (Reaches.step hs2)) hrch7
  have hk_tot : s7.k = s.k := hk7.trans hk2
  have h280_7 := hoth7 280 _ (Or.inl (by decide)) (by decide) (hoth2 280 _ (Or.inl (by decide)) (by decide) h280)
  have h288_7 := hoth7 288 _ (Or.inl (by decide)) (by decide) (hoth2 288 _ (Or.inl (by decide)) (by decide) h288)
  have hrd360_7 := hoth7 360 _ (Or.inl (by decide)) (by decide) hrd360
  let tb := SporosProvLen.ltag ls.pc ls.ds[0]!
  have houtsW : ∀ v ∈ ([tb] : List SporosProvLen.LVal), v.w < X86Env.W ∧ v.plen < X86Env.W := by
    intro v hv; simp only [List.mem_singleton] at hv; subst hv
    exact ⟨hbW.1, by dsimp [tb, SporosProvLen.ltag]; rw [M_eq_W']; exact lt_W_of_mod _⟩
  have hslots : ∀ (i : Nat) (_ : i < ([tb] : List SporosProvLen.LVal).length),
      rd64 s7.mem (ea ctxBase (360 + i * 16)) = some ([tb][i]).w ∧
      rd64 s7.mem (ea ctxBase (368 + i * 16)) = some ([tb][i]).plen := by
    intro i hi
    rcases i with _ | i
    · simpa using ⟨hrd360_7, hrd368⟩
    · simp at hi
  have hfin := jmp_epilogue0_finish_spec
    (step_at_chunk 4 hrip7 rfl (by decide : (4195880, (.ofInt (.jmp 4195233)), 5) ∈ SporosVmBin.vmTable_4))
    hwf7 (oracleAdmits_of_reaches hor hrch7_tot) hrs hpcW hle (by simp) houtsW h280_7 h288_7 hslots
  have hres := lift_next_outcome hrch7_tot hk_tot hfin
  rcases hds_eq : ls.ds with _ | ⟨b, _ | ⟨a, r⟩⟩ <;> try (simp [hds_eq] at hle)
  dsimp only [PrimPostOutcome]
  simp only [Sporos.Prim.eff, List.take_succ_cons, List.take_zero, SporosProvLen.lexecPrim, hds_eq, List.drop_succ_cons, List.drop_zero]
  simpa only [tb, hds_eq, List.getElem!_cons_zero, List.drop_succ_cons, List.drop_zero] using hres

theorem P_tuck_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_tuck) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2)
    (h288 : rd64 s.mem (ea ctxBase 288) = some 3)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w)
    (h304 : rd64 s.mem (ea ctxBase 304) = some ls.ds[0]!.plen)
    (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w)
    (h320 : rd64 s.mem (ea ctxBase 320) = some ls.ds[1]!.plen) :
    PrimPostOutcome pr ls f rest s .tuck := by
  have hrip' : s.rip = 4195885 := hrip
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hreg : RegsOk s := h.2.2.2.1
  have hbW := wf_ds_lt_W h (getElem!_mem_of_lt (by omega : 0 < ls.ds.length))
  have haW := wf_ds_lt_W h (getElem!_mem_of_lt (by omega : 1 < ls.ds.length))
  have h0 : X86Vm.step vmImage s = .ok { s.set 0 (some ls.ds[0]!.w) with rip := 4195892 } := by
    rw [step_at_chunk 4 hrip' rfl (by decide : (4195885, (.ofInt (.ld 0 15 296)), 7) ∈ SporosVmBin.vmTable_4),
      exec_ld h15 h296, hrip']
  let s1 := { s.set 0 (some ls.ds[0]!.w) with rip := 4195892 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) ls.ds[0]!.w
  have hs1_4 : s1.reg 4 = some vc.sp := by
    change (s.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hs1_0 : s1.reg 0 = some ls.ds[0]!.w := reg_set_self_ok' hreg (by decide) _
  have hwf1 : vc.wf pr ls s1 := vc.wf_step_misc h rfl hr1 hs1_4 hs1_15
  obtain ⟨s2, hs2, hrip2, hk2, _, _, hwf2, hrd360, hoth2⟩ :=
    step_wr_ctx_scratch (step_at_chunk 4 rfl rfl (by decide : (4195892, (.ofInt (.st 15 360 0)), 7) ∈ SporosVmBin.vmTable_4))
      hwf1 hs1_0 (by decide) (by decide)
  rw [Nat.mod_eq_of_lt hbW.1] at hrd360
  have h304_2 := hoth2 304 _ (Or.inl (by decide)) (by decide) h304
  obtain ⟨s7, hrch7, hrip7, hk7, hwf7, hrd368, hoth7⟩ :=
    step_wr_ctx_tag_slot
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip2]) rfl (by decide : (4195899, (.ofInt (.ld 0 15 304)), 7) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip2]) rfl (by decide : (4195906, (.ofInt (.ld 1 15 224)), 7) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip2]) rfl (by decide : (4195913, (.ofInt (.alu .add 0 1)), 3) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip2]) rfl (by decide : (4195916, (.ofInt (.aluI .add 0 1)), 4) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip2]) rfl (by decide : (4195920, (.ofInt (.st 15 368 0)), 7) ∈ SporosVmBin.vmTable_4))
      (nextAddr := 4195927) (by rw [hrip2]) hwf2 h304_2 (by decide) (by decide)
  have h312_7 := hoth7 312 _ (Or.inl (by decide)) (by decide) (hoth2 312 _ (Or.inl (by decide)) (by decide) h312)
  have h7_15 : s7.reg 15 = some ctxBase := hwf7.2.2.2.2.2.1
  have h7_4 : s7.reg 4 = some vc.sp := hwf7.2.2.2.2.1
  have hr7 : RegsOk s7 := hwf7.2.2.2.1
  have h7 : X86Vm.step vmImage s7 = .ok { s7.set 0 (some ls.ds[1]!.w) with rip := 4195934 } := by
    rw [step_at_chunk 4 hrip7 rfl (by decide : (4195927, (.ofInt (.ld 0 15 312)), 7) ∈ SporosVmBin.vmTable_4),
      exec_ld h7_15 h312_7, hrip7]
  let s8 := { s7.set 0 (some ls.ds[1]!.w) with rip := 4195934 }
  have hr8 : RegsOk s8 := regsOk_set hr7 (by decide) ls.ds[1]!.w
  have hs8_4 : s8.reg 4 = some vc.sp := by
    change (s7.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h7_4
  have hs8_15 : s8.reg 15 = some ctxBase := by
    change (s7.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h7_15
  have hs8_0 : s8.reg 0 = some ls.ds[1]!.w := reg_set_self_ok' hr7 (by decide) _
  have hwf8 := VmCtx.wf_step_misc (s' := s8) hwf7 rfl hr8 hs8_4 hs8_15
  obtain ⟨s9, hs9, hrip9, hk9, _, _, hwf9, hrd376, hoth9⟩ :=
    step_wr_ctx_scratch (step_at_chunk 4 rfl rfl (by decide : (4195934, (.ofInt (.st 15 376 0)), 7) ∈ SporosVmBin.vmTable_4))
      hwf8 hs8_0 (by decide) (by decide)
  rw [Nat.mod_eq_of_lt haW.1] at hrd376
  have h320_9 := hoth9 320 _ (Or.inl (by decide)) (by decide)
    (hoth7 320 _ (Or.inl (by decide)) (by decide) (hoth2 320 _ (Or.inl (by decide)) (by decide) h320))
  obtain ⟨s14, hrch14, hrip14, hk14, hwf14, hrd384, hoth14⟩ :=
    step_wr_ctx_tag_slot
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip9]) rfl (by decide : (4195941, (.ofInt (.ld 0 15 320)), 7) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip9]) rfl (by decide : (4195948, (.ofInt (.ld 1 15 224)), 7) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip9]) rfl (by decide : (4195955, (.ofInt (.alu .add 0 1)), 3) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip9]) rfl (by decide : (4195958, (.ofInt (.aluI .add 0 1)), 4) ∈ SporosVmBin.vmTable_4))
      (fun t ht => step_at_chunk 4 (by rw [ht, hrip9]) rfl (by decide : (4195962, (.ofInt (.st 15 384 0)), 7) ∈ SporosVmBin.vmTable_4))
      (nextAddr := 4195969) (by rw [hrip9]) hwf9 h320_9 (by decide) (by decide)
  -- copy 360 -> 392 and 368 -> 400
  have hrd360_14 := hoth14 360 _ (Or.inl (by decide)) (by decide)
    (hoth9 360 _ (Or.inl (by decide)) (by decide) (hoth7 360 _ (Or.inl (by decide)) (by decide) hrd360))
  have hrd368_14 := hoth14 368 _ (Or.inl (by decide)) (by decide)
    (hoth9 368 _ (Or.inl (by decide)) (by decide) hrd368)
  have h14_15 : s14.reg 15 = some ctxBase := hwf14.2.2.2.2.2.1
  have h14_4 : s14.reg 4 = some vc.sp := hwf14.2.2.2.2.1
  have hr14 : RegsOk s14 := hwf14.2.2.2.1
  have h14 : X86Vm.step vmImage s14 = .ok { s14.set 0 (some ls.ds[0]!.w) with rip := 4195976 } := by
    rw [step_at_chunk 4 hrip14 rfl (by decide : (4195969, (.ofInt (.ld 0 15 360)), 7) ∈ SporosVmBin.vmTable_4),
      exec_ld h14_15 hrd360_14, hrip14]
  let s15 := { s14.set 0 (some ls.ds[0]!.w) with rip := 4195976 }
  have hr15 : RegsOk s15 := regsOk_set hr14 (by decide) ls.ds[0]!.w
  have hs15_4 : s15.reg 4 = some vc.sp := by
    change (s14.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h14_4
  have hs15_15 : s15.reg 15 = some ctxBase := by
    change (s14.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h14_15
  have hs15_0 : s15.reg 0 = some ls.ds[0]!.w := reg_set_self_ok' hr14 (by decide) _
  have hwf15 := VmCtx.wf_step_misc (s' := s15) hwf14 rfl hr15 hs15_4 hs15_15
  obtain ⟨s16, hs16, hrip16, hk16, _, _, hwf16, hrd392, hoth16⟩ :=
    step_wr_ctx_scratch (step_at_chunk 4 rfl rfl (by decide : (4195976, (.ofInt (.st 15 392 0)), 7) ∈ SporosVmBin.vmTable_4))
      hwf15 hs15_0 (by decide) (by decide)
  rw [Nat.mod_eq_of_lt hbW.1] at hrd392
  have hrd368_16 := hoth16 368 _ (Or.inl (by decide)) (by decide) hrd368_14
  have h16_15 : s16.reg 15 = some ctxBase := hwf16.2.2.2.2.2.1
  have h16_4 : s16.reg 4 = some vc.sp := hwf16.2.2.2.2.1
  have hr16 : RegsOk s16 := hwf16.2.2.2.1
  have h16 : X86Vm.step vmImage s16 =
      .ok { s16.set 0 (some ((ls.ds[0]!.plen + ls.pc + 1) % Sporos.M)) with rip := 4195990 } := by
    rw [step_at_chunk 4 hrip16 rfl (by decide : (4195983, (.ofInt (.ld 0 15 368)), 7) ∈ SporosVmBin.vmTable_4),
      exec_ld h16_15 hrd368_16, hrip16]
  let s17 := { s16.set 0 (some ((ls.ds[0]!.plen + ls.pc + 1) % Sporos.M)) with rip := 4195990 }
  have hr17 : RegsOk s17 := regsOk_set hr16 (by decide) _
  have hs17_4 : s17.reg 4 = some vc.sp := by
    change (s16.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h16_4
  have hs17_15 : s17.reg 15 = some ctxBase := by
    change (s16.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h16_15
  have hs17_0 : s17.reg 0 = some ((ls.ds[0]!.plen + ls.pc + 1) % Sporos.M) := reg_set_self_ok' hr16 (by decide) _
  have hwf17 := VmCtx.wf_step_misc (s' := s17) hwf16 rfl hr17 hs17_4 hs17_15
  obtain ⟨s18, hs18, hrip18, hk18, _, _, hwf18, hrd400, hoth18⟩ :=
    step_wr_ctx_scratch (step_at_chunk 4 rfl rfl (by decide : (4195990, (.ofInt (.st 15 400 0)), 7) ∈ SporosVmBin.vmTable_4))
      hwf17 hs17_0 (by decide) (by decide)
  rw [M_eq_W', Nat.mod_mod] at hrd400
  have hrch18 : Reaches vmImage s s18 :=
    Reaches.trans (Reaches.step_trans h0 (Reaches.step hs2))
      (Reaches.trans hrch7 (Reaches.trans (Reaches.step_trans h7 (Reaches.step hs9))
        (Reaches.trans hrch14 (Reaches.step_trans h14 (Reaches.step_trans hs16 (Reaches.step_trans h16 (Reaches.step hs18)))))))
  have hk_tot : s18.k = s.k := by
    dsimp only [s17, s15, s8, s1, X86Int.St.set] at hk18 hk16 hk14 hk9 hk7 hk2 ⊢
    rw [hk18, hk16, hk14, hk9, hk7, hk2]
  have h280_18 := hoth18 280 _ (Or.inl (by decide)) (by decide) (hoth16 280 _ (Or.inl (by decide)) (by decide)
    (hoth14 280 _ (Or.inl (by decide)) (by decide) (hoth9 280 _ (Or.inl (by decide)) (by decide)
      (hoth7 280 _ (Or.inl (by decide)) (by decide) (hoth2 280 _ (Or.inl (by decide)) (by decide) h280)))))
  have h288_18 := hoth18 288 _ (Or.inl (by decide)) (by decide) (hoth16 288 _ (Or.inl (by decide)) (by decide)
    (hoth14 288 _ (Or.inl (by decide)) (by decide) (hoth9 288 _ (Or.inl (by decide)) (by decide)
      (hoth7 288 _ (Or.inl (by decide)) (by decide) (hoth2 288 _ (Or.inl (by decide)) (by decide) h288)))))
  have hrd360_18 := hoth18 360 _ (Or.inl (by decide)) (by decide) (hoth16 360 _ (Or.inl (by decide)) (by decide) hrd360_14)
  have hrd368_18 := hoth18 368 _ (Or.inl (by decide)) (by decide) hrd368_16
  have hrd376_18 := hoth18 376 _ (Or.inl (by decide)) (by decide) (hoth16 376 _ (Or.inl (by decide)) (by decide)
    (hoth14 376 _ (Or.inl (by decide)) (by decide) hrd376))
  have hrd384_18 := hoth18 384 _ (Or.inl (by decide)) (by decide) (hoth16 384 _ (Or.inl (by decide)) (by decide) hrd384)
  have hrd392_18 := hoth18 392 _ (Or.inl (by decide)) (by decide) hrd392
  let tb := SporosProvLen.ltag ls.pc ls.ds[0]!
  let ta := SporosProvLen.ltag ls.pc ls.ds[1]!
  have houtsW : ∀ v ∈ ([tb, ta, tb] : List SporosProvLen.LVal), v.w < X86Env.W ∧ v.plen < X86Env.W := by
    intro v hv; simp only [List.mem_cons, List.not_mem_nil, or_false] at hv
    rcases hv with rfl | rfl | rfl
    · exact ⟨hbW.1, by dsimp [tb, SporosProvLen.ltag]; rw [M_eq_W']; exact lt_W_of_mod _⟩
    · exact ⟨haW.1, by dsimp [ta, SporosProvLen.ltag]; rw [M_eq_W']; exact lt_W_of_mod _⟩
    · exact ⟨hbW.1, by dsimp [tb, SporosProvLen.ltag]; rw [M_eq_W']; exact lt_W_of_mod _⟩
  have hslots : ∀ (i : Nat) (_ : i < ([tb, ta, tb] : List SporosProvLen.LVal).length),
      rd64 s18.mem (ea ctxBase (360 + i * 16)) = some ([tb, ta, tb][i]).w ∧
      rd64 s18.mem (ea ctxBase (368 + i * 16)) = some ([tb, ta, tb][i]).plen := by
    intro i hi; rcases i with _ | _ | _ | i
    · exact ⟨hrd360_18, hrd368_18⟩
    · exact ⟨hrd376_18, hrd384_18⟩
    · exact ⟨hrd392_18, hrd400⟩
    · simp only [List.length_cons, List.length_nil] at hi; omega
  have hfin := jmp_epilogue0_finish_spec
    (step_at_chunk 4 hrip18 rfl (by decide : (4195997, (.ofInt (.jmp 4195233)), 5) ∈ SporosVmBin.vmTable_4))
    hwf18 (oracleAdmits_of_reaches hor hrch18) hrs hpcW hle (by simp) houtsW h280_18 h288_18 hslots
  have hres := lift_next_outcome hrch18 hk_tot hfin
  rcases hds_eq : ls.ds with _ | ⟨b, _ | ⟨a, r⟩⟩ <;> try (simp [hds_eq] at hle)
  dsimp only [PrimPostOutcome]
  simp only [Sporos.Prim.eff, List.take_succ_cons, List.take_zero, SporosProvLen.lexecPrim, hds_eq, List.drop_succ_cons, List.drop_zero]
  simpa only [tb, ta, hds_eq, List.getElem!_cons_zero, List.getElem!_cons_succ, List.drop_succ_cons, List.drop_zero] using hres

/-! ### Arithmetic, division, bitwise, shift, and comparison primitives -/

theorem P_add_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_add) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w) :
    PrimPostOutcome pr ls f rest s .add := by
  have hle0 := hle
  rcases hds_eq : ls.ds with _ | ⟨b, _ | ⟨a, r⟩⟩ <;> try (simp [hds_eq] at hle0)
  exact alu_bin_mk1_spec (op := .add) (c0 := 5) (c1 := 5) (c2 := 5) (c3 := 5) (resFn := fun x y => Sporos.wrap (x + y)) hrip
    (by decide) (by decide) (by decide) (by decide) rfl hle h hor hrs hpcW h272 h280 h288 h296 h312
    (fun x y _ _ => by (try dsimp only); rw [aluF_add_val, wrap_eq_mod, Nat.mod_mod])
    (by simp [hds_eq, SporosProvLen.lexecPrim])

theorem P_sub_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_sub) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w) :
    PrimPostOutcome pr ls f rest s .sub := by
  have hle0 := hle
  rcases hds_eq : ls.ds with _ | ⟨b, _ | ⟨a, r⟩⟩ <;> try (simp [hds_eq] at hle0)
  exact alu_bin_mk1_spec (op := .sub) (c0 := 5) (c1 := 5) (c2 := 5) (c3 := 5)
    (resFn := fun x y => Sporos.wrap (x + Sporos.M - y % Sporos.M)) hrip
    (by decide) (by decide) (by decide) (by decide) rfl hle h hor hrs hpcW h272 h280 h288 h296 h312
    (fun x y _ hy => by (try dsimp only); rw [aluF_sub_val, wrap_eq_mod, M_eq_W', Nat.mod_mod, Nat.mod_eq_of_lt hy])
    (by simp [hds_eq, SporosProvLen.lexecPrim])

theorem P_mul_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_mul) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w) :
    PrimPostOutcome pr ls f rest s .mul := by
  have hrip' : s.rip = 4196046 := hrip
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hreg : RegsOk s := h.2.2.2.1
  have haW := (wf_ds_lt_W h (getElem!_mem_of_lt (by omega : 1 < ls.ds.length))).1
  have hbW := (wf_ds_lt_W h (getElem!_mem_of_lt (by omega : 0 < ls.ds.length))).1
  have h0 : X86Vm.step vmImage s = .ok { s.set 0 (some ls.ds[1]!.w) with rip := 4196053 } := by
    rw [step_at_chunk 5 hrip' rfl (by decide : (4196046, (.ofInt (.ld 0 15 312)), 7) ∈ SporosVmBin.vmTable_5),
      exec_ld h15 h312, hrip']
  let s1 := { s.set 0 (some ls.ds[1]!.w) with rip := 4196053 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) ls.ds[1]!.w
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hs1_0 : s1.reg 0 = some ls.ds[1]!.w := reg_set_self_ok' hreg (by decide) _
  have h1 : X86Vm.step vmImage s1 = .ok { s1.set 1 (some ls.ds[0]!.w) with rip := 4196060 } := by
    rw [step_at_chunk 5 rfl rfl (by decide : (4196053, (.ofInt (.ld 1 15 296)), 7) ∈ SporosVmBin.vmTable_5),
      exec_ld hs1_15 h296]
  let s2 := { s1.set 1 (some ls.ds[0]!.w) with rip := 4196060 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) ls.ds[0]!.w
  have hs2_0 : s2.reg 0 = some ls.ds[1]!.w := by
    change (s1.set 1 _).reg 0 = some ls.ds[1]!.w; rw [reg_set_ne (by decide)]; exact hs1_0
  have hs2_1 : s2.reg 1 = some ls.ds[0]!.w := reg_set_self_ok' hr1 (by decide) _
  let res := ls.ds[1]!.w % X86Env.W * (ls.ds[0]!.w % X86Env.W) % X86Env.W
  let ov := X86Vm.toSigned64 ls.ds[1]!.w * X86Vm.toSigned64 ls.ds[0]!.w != X86Vm.toSigned64 res
  have h2 : X86Vm.step vmImage s2 =
      .ok { s2.set 0 (some res) with fl := { cf := some ov, zf := none, sf := none, of := some ov }, rip := 4196064 } := by
    rw [step_at_chunk 5 rfl rfl (by decide : (4196060, (.imul 0 1), 4) ∈ SporosVmBin.vmTable_5),
      exec_imul hs2_0 hs2_1]
  let s3 := { s2.set 0 (some res) with fl := { cf := some ov, zf := none, sf := none, of := some ov }, rip := 4196064 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) res
  have hs3_4 : s3.reg 4 = some vc.sp := by
    change (s2.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s1.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hs3_15 : s3.reg 15 = some ctxBase := by
    change (s2.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s1.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15
  have hs3_0 : s3.reg 0 = some res := reg_set_self_ok' hr2 (by decide) _
  have hwf3 : vc.wf pr ls s3 := vc.wf_step_misc h rfl hr3 hs3_4 hs3_15
  have hqW : SporosProvLen.lq (ls.ds.take 2) ls.pc < X86Env.W := by
    dsimp [SporosProvLen.lq]; rw [← M_eq_W']; exact lt_W_of_mod _
  have hrch3 := Reaches.step_trans h0 (Reaches.step_trans h1 (Reaches.step h2))
  have hfin := jmp_mk1_finish_spec
    (step_at_chunk 5 rfl rfl (by decide : (4196064, (.ofInt (.jmp 4195210)), 5) ∈ SporosVmBin.vmTable_5))
    hwf3 (oracleAdmits_of_reaches hor hrch3) hrs hpcW hle hs3_0 h272 hqW h280 h288
  have hres := lift_next_outcome hrch3 rfl hfin
  have hle0 := hle
  rcases hds_eq : ls.ds with _ | ⟨b, _ | ⟨a, r⟩⟩ <;> try (simp [hds_eq] at hle0)
  have haW' : a.w < X86Env.W := by simpa [hds_eq] using haW
  have hbW' : b.w < X86Env.W := by simpa [hds_eq] using hbW
  have hres_eq : res % X86Env.W = Sporos.wrap (a.w * b.w) := by
    dsimp [res]; simp [hds_eq, Nat.mod_eq_of_lt haW', Nat.mod_eq_of_lt hbW', wrap_eq_mod]
  dsimp only [PrimPostOutcome]
  simp only [Sporos.Prim.eff, List.take_succ_cons, List.take_zero, SporosProvLen.lexecPrim, hds_eq]
  simpa only [hres_eq, hds_eq, List.take_succ_cons, List.take_zero, List.singleton_append] using hres

theorem P_divu_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_divu) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w) :
    PrimPostOutcome pr ls f rest s .divu := by
  have hrip' : s.rip = 4196069 := hrip
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hreg : RegsOk s := h.2.2.2.1
  have haW := (wf_ds_lt_W h (getElem!_mem_of_lt (by omega : 1 < ls.ds.length))).1
  have hbW := (wf_ds_lt_W h (getElem!_mem_of_lt (by omega : 0 < ls.ds.length))).1
  have h0 : X86Vm.step vmImage s = .ok { s.set 1 (some ls.ds[0]!.w) with rip := 4196076 } := by
    rw [step_at_chunk 5 hrip' rfl (by decide : (4196069, (.ofInt (.ld 1 15 296)), 7) ∈ SporosVmBin.vmTable_5),
      exec_ld h15 h296, hrip']
  let s1 := { s.set 1 (some ls.ds[0]!.w) with rip := 4196076 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) ls.ds[0]!.w
  have hs1_1 : s1.reg 1 = some ls.ds[0]!.w := reg_set_self_ok' hreg (by decide) _
  have h1 : X86Vm.step vmImage s1 = .ok { s1 with fl := (X86Int.aluF .and ls.ds[0]!.w ls.ds[0]!.w).2, rip := 4196079 } := by
    rw [step_at_chunk 5 rfl rfl (by decide : (4196076, (.ofInt (.test 1 1)), 3) ∈ SporosVmBin.vmTable_5),
      exec_test hs1_1 hs1_1]
  let s2 := { s1 with fl := (X86Int.aluF .and ls.ds[0]!.w ls.ds[0]!.w).2, rip := 4196079 }
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hs1_4 : s1.reg 4 = some vc.sp := by
    change (s.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hle0 := hle
  rcases hds_eq : ls.ds with _ | ⟨b, _ | ⟨a, r⟩⟩ <;> try (simp [hds_eq] at hle0)
  have haW' : a.w < X86Env.W := by simpa [hds_eq] using haW
  have hbW' : b.w < X86Env.W := by simpa [hds_eq] using hbW
  by_cases hbz : b.w = 0
  · have h2 : X86Vm.step vmImage s2 = .ok { s2 with rip := 4201167 } := by
      rw [step_at_chunk 5 rfl rfl (by decide : (4196079, (.ofInt (.jcc 4 4201167)), 6) ∈ SporosVmBin.vmTable_5),
        exec_jcc (cond_aluF_and_self_e ls.ds[0]!.w)]
      dsimp only [s2]; simp [hds_eq, hbz]
    let s3 := { s2 with rip := 4201167 }
    have hwf3 : vc.wf pr ls s3 := vc.wf_step_misc h rfl hr1 hs1_4 hs1_15
    have hfail := L_fail_from_vc_spec (e := .divZero) (s := s3) rfl hwf3
    have hrch3 := Reaches.step_trans h0 (Reaches.step_trans h1 (Reaches.step h2))
    dsimp only [PrimPostOutcome]
    simp only [Sporos.Prim.eff, List.take_succ_cons, List.take_zero, SporosProvLen.lexecPrim, hds_eq, hbz, Nat.zero_mod, ↓reduceIte]
    exact lift_fail_outcome hrch3 rfl hfail
  · have h2 : X86Vm.step vmImage s2 = .ok { s2 with rip := 4196085 } := by
      rw [step_at_chunk 5 rfl rfl (by decide : (4196079, (.ofInt (.jcc 4 4201167)), 6) ∈ SporosVmBin.vmTable_5),
        exec_jcc (cond_aluF_and_self_e ls.ds[0]!.w)]
      dsimp only [s2]; simp [hds_eq, hbz]
    let s3 := { s2 with rip := 4196085 }
    have hs3_15 : s3.reg 15 = some ctxBase := hs1_15
    have h3 : X86Vm.step vmImage s3 = .ok { s3.set 0 (some ls.ds[1]!.w) with rip := 4196092 } := by
      rw [step_at_chunk 5 rfl rfl (by decide : (4196085, (.ofInt (.ld 0 15 312)), 7) ∈ SporosVmBin.vmTable_5),
        exec_ld hs3_15 h312]
    let s4 := { s3.set 0 (some ls.ds[1]!.w) with rip := 4196092 }
    have hr4 : RegsOk s4 := regsOk_set hr1 (by decide) ls.ds[1]!.w
    have hs4_0 : s4.reg 0 = some ls.ds[1]!.w := reg_set_self_ok' hr1 (by decide) _
    have hs4_1 : s4.reg 1 = some ls.ds[0]!.w := by
      change (s3.set 0 _).reg 1 = some ls.ds[0]!.w; rw [reg_set_ne (by decide)]; exact hs1_1
    obtain ⟨r2, hr2⟩ := reg_of_ok hr4 2 (by decide) (by decide)
    have hxor0 : (X86Int.aluF .xor r2 r2).1 = 0 := by rw [aluF_xor_val]; exact Nat.xor_self r2
    have hxor_pair : X86Int.aluF .xor r2 r2 = (0, (X86Int.aluF .xor r2 r2).2) := Prod.ext hxor0 rfl
    have h4s : X86Vm.step vmImage s4 =
        .ok { s4.set 2 (some 0) with fl := (X86Int.aluF .xor r2 r2).2, rip := 4196094 } := by
      rw [step_at_chunk 5 rfl rfl (by decide : (4196092, (.ofInt (.alu .xor 2 2)), 2) ∈ SporosVmBin.vmTable_5),
        exec_alu hr2 hr2, hxor_pair]
    let s5 := { s4.set 2 (some 0) with fl := (X86Int.aluF .xor r2 r2).2, rip := 4196094 }
    have hr5 : RegsOk s5 := regsOk_set hr4 (by decide) 0
    have hs5_0 : s5.reg 0 = some ls.ds[1]!.w := by
      change (s4.set 2 _).reg 0 = some ls.ds[1]!.w; rw [reg_set_ne (by decide)]; exact hs4_0
    have hs5_2 : s5.reg 2 = some 0 := reg_set_self_ok' hr4 (by decide) _
    have hs5_1 : s5.reg 1 = some ls.ds[0]!.w := by
      change (s4.set 2 _).reg 1 = some ls.ds[0]!.w; rw [reg_set_ne (by decide)]; exact hs4_1
    have hnum_eq : 0 % X86Env.W * X86Env.W + ls.ds[1]!.w % X86Env.W = a.w := by
      simp [hds_eq, Nat.mod_eq_of_lt haW']
    have hden_eq : ls.ds[0]!.w % X86Env.W = b.w := by simp [hds_eq, Nat.mod_eq_of_lt hbW']
    have hq_lt : (0 % X86Env.W * X86Env.W + ls.ds[1]!.w % X86Env.W) / (ls.ds[0]!.w % X86Env.W) < X86Env.W := by
      rw [hnum_eq, hden_eq]; exact Nat.lt_of_le_of_lt (Nat.div_le_self _ _) haW'
    have h5 : X86Vm.step vmImage s5 =
        .ok { (s5.set 0 (some (a.w / b.w))).set 2 (some (a.w % b.w)) with fl := X86Int.Flags.undef, rip := 4196097 } := by
      rw [step_at_chunk 5 rfl rfl (by decide : (4196094, (.div 1), 3) ∈ SporosVmBin.vmTable_5),
        exec_div hs5_0 hs5_2 hs5_1 (by rw [hden_eq]; exact hbz) hq_lt, hnum_eq, hden_eq]
    let s6 := { (s5.set 0 (some (a.w / b.w))).set 2 (some (a.w % b.w)) with fl := X86Int.Flags.undef, rip := 4196097 }
    have hr6 : RegsOk s6 := regsOk_set (regsOk_set hr5 (by decide) (a.w / b.w)) (by decide) (a.w % b.w)
    have hs6_4 : s6.reg 4 = some vc.sp := by
      change ((s5.set 0 _).set 2 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide), reg_set_ne (by decide)]
      change (s4.set 2 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
      change (s3.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs1_4
    have hs6_15 : s6.reg 15 = some ctxBase := by
      change ((s5.set 0 _).set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide), reg_set_ne (by decide)]
      change (s4.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
      change (s3.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15
    have hs6_0 : s6.reg 0 = some (a.w / b.w) := by
      change ((s5.set 0 _).set 2 _).reg 0 = some (a.w / b.w); rw [reg_set_ne (by decide)]
      exact reg_set_self_ok' hr5 (by decide) _
    have hwf6 : vc.wf pr ls s6 := vc.wf_step_misc h rfl hr6 hs6_4 hs6_15
    have hqW : SporosProvLen.lq (ls.ds.take 2) ls.pc < X86Env.W := by
      dsimp [SporosProvLen.lq]; rw [← M_eq_W']; exact lt_W_of_mod _
    have hrch6 := Reaches.step_trans h0 (Reaches.step_trans h1 (Reaches.step_trans h2
      (Reaches.step_trans h3 (Reaches.step_trans h4s (Reaches.step h5)))))
    have hfin := jmp_mk1_finish_spec
      (step_at_chunk 5 rfl rfl (by decide : (4196097, (.ofInt (.jmp 4195210)), 5) ∈ SporosVmBin.vmTable_5))
      hwf6 (oracleAdmits_of_reaches hor hrch6) hrs hpcW hle hs6_0 h272 hqW h280 h288
    have hres := lift_next_outcome hrch6 rfl hfin
    have hbw_mod : b.w % Sporos.M ≠ 0 := by rw [M_eq_W', Nat.mod_eq_of_lt hbW']; exact hbz
    dsimp only [PrimPostOutcome]
    simp only [Sporos.Prim.eff, List.take_succ_cons, List.take_zero, SporosProvLen.lexecPrim, hds_eq, if_neg hbw_mod]
    simpa only [wrap_eq_mod, M_eq_W', hds_eq, List.take_succ_cons, List.take_zero, List.singleton_append] using hres

theorem P_modu_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_modu) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w) :
    PrimPostOutcome pr ls f rest s .modu := by
  have hrip' : s.rip = 4196102 := hrip
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hreg : RegsOk s := h.2.2.2.1
  have haW := (wf_ds_lt_W h (getElem!_mem_of_lt (by omega : 1 < ls.ds.length))).1
  have hbW := (wf_ds_lt_W h (getElem!_mem_of_lt (by omega : 0 < ls.ds.length))).1
  have h0 : X86Vm.step vmImage s = .ok { s.set 1 (some ls.ds[0]!.w) with rip := 4196109 } := by
    rw [step_at_chunk 5 hrip' rfl (by decide : (4196102, (.ofInt (.ld 1 15 296)), 7) ∈ SporosVmBin.vmTable_5),
      exec_ld h15 h296, hrip']
  let s1 := { s.set 1 (some ls.ds[0]!.w) with rip := 4196109 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) ls.ds[0]!.w
  have hs1_1 : s1.reg 1 = some ls.ds[0]!.w := reg_set_self_ok' hreg (by decide) _
  have h1 : X86Vm.step vmImage s1 = .ok { s1 with fl := (X86Int.aluF .and ls.ds[0]!.w ls.ds[0]!.w).2, rip := 4196112 } := by
    rw [step_at_chunk 5 rfl rfl (by decide : (4196109, (.ofInt (.test 1 1)), 3) ∈ SporosVmBin.vmTable_5),
      exec_test hs1_1 hs1_1]
  let s2 := { s1 with fl := (X86Int.aluF .and ls.ds[0]!.w ls.ds[0]!.w).2, rip := 4196112 }
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hs1_4 : s1.reg 4 = some vc.sp := by
    change (s.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hle0 := hle
  rcases hds_eq : ls.ds with _ | ⟨b, _ | ⟨a, r⟩⟩ <;> try (simp [hds_eq] at hle0)
  have haW' : a.w < X86Env.W := by simpa [hds_eq] using haW
  have hbW' : b.w < X86Env.W := by simpa [hds_eq] using hbW
  by_cases hbz : b.w = 0
  · have h2 : X86Vm.step vmImage s2 = .ok { s2 with rip := 4201167 } := by
      rw [step_at_chunk 5 rfl rfl (by decide : (4196112, (.ofInt (.jcc 4 4201167)), 6) ∈ SporosVmBin.vmTable_5),
        exec_jcc (cond_aluF_and_self_e ls.ds[0]!.w)]
      dsimp only [s2]; simp [hds_eq, hbz]
    let s3 := { s2 with rip := 4201167 }
    have hwf3 : vc.wf pr ls s3 := vc.wf_step_misc h rfl hr1 hs1_4 hs1_15
    have hfail := L_fail_from_vc_spec (e := .divZero) (s := s3) rfl hwf3
    have hrch3 := Reaches.step_trans h0 (Reaches.step_trans h1 (Reaches.step h2))
    dsimp only [PrimPostOutcome]
    simp only [Sporos.Prim.eff, List.take_succ_cons, List.take_zero, SporosProvLen.lexecPrim, hds_eq, hbz, Nat.zero_mod, ↓reduceIte]
    exact lift_fail_outcome hrch3 rfl hfail
  · have h2 : X86Vm.step vmImage s2 = .ok { s2 with rip := 4196118 } := by
      rw [step_at_chunk 5 rfl rfl (by decide : (4196112, (.ofInt (.jcc 4 4201167)), 6) ∈ SporosVmBin.vmTable_5),
        exec_jcc (cond_aluF_and_self_e ls.ds[0]!.w)]
      dsimp only [s2]; simp [hds_eq, hbz]
    let s3 := { s2 with rip := 4196118 }
    have hs3_15 : s3.reg 15 = some ctxBase := hs1_15
    have h3 : X86Vm.step vmImage s3 = .ok { s3.set 0 (some ls.ds[1]!.w) with rip := 4196125 } := by
      rw [step_at_chunk 5 rfl rfl (by decide : (4196118, (.ofInt (.ld 0 15 312)), 7) ∈ SporosVmBin.vmTable_5),
        exec_ld hs3_15 h312]
    let s4 := { s3.set 0 (some ls.ds[1]!.w) with rip := 4196125 }
    have hr4 : RegsOk s4 := regsOk_set hr1 (by decide) ls.ds[1]!.w
    have hs4_0 : s4.reg 0 = some ls.ds[1]!.w := reg_set_self_ok' hr1 (by decide) _
    have hs4_1 : s4.reg 1 = some ls.ds[0]!.w := by
      change (s3.set 0 _).reg 1 = some ls.ds[0]!.w; rw [reg_set_ne (by decide)]; exact hs1_1
    obtain ⟨r2, hr2⟩ := reg_of_ok hr4 2 (by decide) (by decide)
    have hxor0 : (X86Int.aluF .xor r2 r2).1 = 0 := by rw [aluF_xor_val]; exact Nat.xor_self r2
    have hxor_pair : X86Int.aluF .xor r2 r2 = (0, (X86Int.aluF .xor r2 r2).2) := Prod.ext hxor0 rfl
    have h4s : X86Vm.step vmImage s4 =
        .ok { s4.set 2 (some 0) with fl := (X86Int.aluF .xor r2 r2).2, rip := 4196127 } := by
      rw [step_at_chunk 5 rfl rfl (by decide : (4196125, (.ofInt (.alu .xor 2 2)), 2) ∈ SporosVmBin.vmTable_5),
        exec_alu hr2 hr2, hxor_pair]
    let s5 := { s4.set 2 (some 0) with fl := (X86Int.aluF .xor r2 r2).2, rip := 4196127 }
    have hr5 : RegsOk s5 := regsOk_set hr4 (by decide) 0
    have hs5_0 : s5.reg 0 = some ls.ds[1]!.w := by
      change (s4.set 2 _).reg 0 = some ls.ds[1]!.w; rw [reg_set_ne (by decide)]; exact hs4_0
    have hs5_2 : s5.reg 2 = some 0 := reg_set_self_ok' hr4 (by decide) _
    have hs5_1 : s5.reg 1 = some ls.ds[0]!.w := by
      change (s4.set 2 _).reg 1 = some ls.ds[0]!.w; rw [reg_set_ne (by decide)]; exact hs4_1
    have hnum_eq : 0 % X86Env.W * X86Env.W + ls.ds[1]!.w % X86Env.W = a.w := by
      simp [hds_eq, Nat.mod_eq_of_lt haW']
    have hden_eq : ls.ds[0]!.w % X86Env.W = b.w := by simp [hds_eq, Nat.mod_eq_of_lt hbW']
    have hq_lt : (0 % X86Env.W * X86Env.W + ls.ds[1]!.w % X86Env.W) / (ls.ds[0]!.w % X86Env.W) < X86Env.W := by
      rw [hnum_eq, hden_eq]; exact Nat.lt_of_le_of_lt (Nat.div_le_self _ _) haW'
    have h5 : X86Vm.step vmImage s5 =
        .ok { (s5.set 0 (some (a.w / b.w))).set 2 (some (a.w % b.w)) with fl := X86Int.Flags.undef, rip := 4196130 } := by
      rw [step_at_chunk 5 rfl rfl (by decide : (4196127, (.div 1), 3) ∈ SporosVmBin.vmTable_5),
        exec_div hs5_0 hs5_2 hs5_1 (by rw [hden_eq]; exact hbz) hq_lt, hnum_eq, hden_eq]
    let s6 := { (s5.set 0 (some (a.w / b.w))).set 2 (some (a.w % b.w)) with fl := X86Int.Flags.undef, rip := 4196130 }
    have hr6 : RegsOk s6 := regsOk_set (regsOk_set hr5 (by decide) (a.w / b.w)) (by decide) (a.w % b.w)
    have hs6_2 : s6.reg 2 = some (a.w % b.w) := reg_set_self_ok' (regsOk_set hr5 (by decide) _) (by decide) _
    have h6 : X86Vm.step vmImage s6 = .ok { s6.set 0 (some (a.w % b.w)) with rip := 4196133 } := by
      rw [step_at_chunk 5 rfl rfl (by decide : (4196130, (.ofInt (.mov 0 2)), 3) ∈ SporosVmBin.vmTable_5),
        exec_mov hs6_2]
    let s7 := { s6.set 0 (some (a.w % b.w)) with rip := 4196133 }
    have hr7 : RegsOk s7 := regsOk_set hr6 (by decide) (a.w % b.w)
    have hs7_4 : s7.reg 4 = some vc.sp := by
      change (s6.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
      change ((s5.set 0 _).set 2 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide), reg_set_ne (by decide)]
      change (s4.set 2 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
      change (s3.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs1_4
    have hs7_15 : s7.reg 15 = some ctxBase := by
      change (s6.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
      change ((s5.set 0 _).set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide), reg_set_ne (by decide)]
      change (s4.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
      change (s3.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15
    have hs7_0 : s7.reg 0 = some (a.w % b.w) := reg_set_self_ok' hr6 (by decide) _
    have hwf7 : vc.wf pr ls s7 := vc.wf_step_misc h rfl hr7 hs7_4 hs7_15
    have hqW : SporosProvLen.lq (ls.ds.take 2) ls.pc < X86Env.W := by
      dsimp [SporosProvLen.lq]; rw [← M_eq_W']; exact lt_W_of_mod _
    have hrch7 := Reaches.step_trans h0 (Reaches.step_trans h1 (Reaches.step_trans h2
      (Reaches.step_trans h3 (Reaches.step_trans h4s (Reaches.step_trans h5 (Reaches.step h6))))))
    have hfin := jmp_mk1_finish_spec
      (step_at_chunk 5 rfl rfl (by decide : (4196133, (.ofInt (.jmp 4195210)), 5) ∈ SporosVmBin.vmTable_5))
      hwf7 (oracleAdmits_of_reaches hor hrch7) hrs hpcW hle hs7_0 h272 hqW h280 h288
    have hres := lift_next_outcome hrch7 rfl hfin
    have hbw_mod : b.w % Sporos.M ≠ 0 := by rw [M_eq_W', Nat.mod_eq_of_lt hbW']; exact hbz
    dsimp only [PrimPostOutcome]
    simp only [Sporos.Prim.eff, List.take_succ_cons, List.take_zero, SporosProvLen.lexecPrim, hds_eq, if_neg hbw_mod]
    simpa only [wrap_eq_mod, M_eq_W', hds_eq, List.take_succ_cons, List.take_zero, List.singleton_append] using hres

theorem P_divs_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_divs) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w) :
    PrimPostOutcome pr ls f rest s .divs := by
  have hrip' : s.rip = 4196138 := hrip
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hreg : RegsOk s := h.2.2.2.1
  have haW := (wf_ds_lt_W h (getElem!_mem_of_lt (by omega : 1 < ls.ds.length))).1
  have hbW := (wf_ds_lt_W h (getElem!_mem_of_lt (by omega : 0 < ls.ds.length))).1
  have h0 : X86Vm.step vmImage s = .ok { s.set 1 (some ls.ds[0]!.w) with rip := 4196145 } := by
    rw [step_at_chunk 5 hrip' rfl (by decide : (4196138, (.ofInt (.ld 1 15 296)), 7) ∈ SporosVmBin.vmTable_5),
      exec_ld h15 h296, hrip']
  let s1 := { s.set 1 (some ls.ds[0]!.w) with rip := 4196145 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) ls.ds[0]!.w
  have hs1_1 : s1.reg 1 = some ls.ds[0]!.w := reg_set_self_ok' hreg (by decide) _
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hs1_4 : s1.reg 4 = some vc.sp := by
    change (s.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have h1 : X86Vm.step vmImage s1 = .ok { s1 with fl := (X86Int.aluF .and ls.ds[0]!.w ls.ds[0]!.w).2, rip := 4196148 } := by
    rw [step_at_chunk 5 rfl rfl (by decide : (4196145, (.ofInt (.test 1 1)), 3) ∈ SporosVmBin.vmTable_5),
      exec_test hs1_1 hs1_1]
  let s2 := { s1 with fl := (X86Int.aluF .and ls.ds[0]!.w ls.ds[0]!.w).2, rip := 4196148 }
  have hle0 := hle
  rcases hds_eq : ls.ds with _ | ⟨b, _ | ⟨a, r⟩⟩ <;> try (simp [hds_eq] at hle0)
  have haW' : a.w < X86Env.W := by simpa [hds_eq] using haW
  have hbW' : b.w < X86Env.W := by simpa [hds_eq] using hbW
  have hqW : SporosProvLen.lq ([b, a]) ls.pc < X86Env.W := by
    dsimp [SporosProvLen.lq]; rw [← M_eq_W']; exact lt_W_of_mod _
  by_cases hbz : b.w = 0
  · have h2 : X86Vm.step vmImage s2 = .ok { s2 with rip := 4201167 } := by
      rw [step_at_chunk 5 rfl rfl (by decide : (4196148, (.ofInt (.jcc 4 4201167)), 6) ∈ SporosVmBin.vmTable_5),
        exec_jcc (cond_aluF_and_self_e ls.ds[0]!.w)]
      dsimp only [s2]; simp [hds_eq, hbz]
    let s3 := { s2 with rip := 4201167 }
    have hwf3 : vc.wf pr ls s3 := vc.wf_step_misc h rfl hr1 hs1_4 hs1_15
    have hfail := L_fail_from_vc_spec (e := .divZero) (s := s3) rfl hwf3
    have hrch3 := Reaches.step_trans h0 (Reaches.step_trans h1 (Reaches.step h2))
    dsimp only [PrimPostOutcome]
    simp only [Sporos.Prim.eff, List.take_succ_cons, List.take_zero, SporosProvLen.lexecPrim, hds_eq, hbz, Nat.zero_mod, ↓reduceIte]
    exact lift_fail_outcome hrch3 rfl hfail
  · have h2 : X86Vm.step vmImage s2 = .ok { s2 with rip := 4196154 } := by
      rw [step_at_chunk 5 rfl rfl (by decide : (4196148, (.ofInt (.jcc 4 4201167)), 6) ∈ SporosVmBin.vmTable_5),
        exec_jcc (cond_aluF_and_self_e ls.ds[0]!.w)]
      dsimp only [s2]; simp [hds_eq, hbz]
    let s3 := { s2 with rip := 4196154 }
    have hs3_15 : s3.reg 15 = some ctxBase := hs1_15
    have h3 : X86Vm.step vmImage s3 = .ok { s3.set 0 (some a.w) with rip := 4196161 } := by
      rw [step_at_chunk 5 rfl rfl (by decide : (4196154, (.ofInt (.ld 0 15 312)), 7) ∈ SporosVmBin.vmTable_5),
        exec_ld hs3_15 h312]; simp [hds_eq, s3]
    let s4 := { s3.set 0 (some a.w) with rip := 4196161 }
    have hr4 : RegsOk s4 := regsOk_set hr1 (by decide) a.w
    have hs4_0 : s4.reg 0 = some a.w := reg_set_self_ok' hr1 (by decide) _
    have hs4_1 : s4.reg 1 = some b.w := by
      change (s3.set 0 _).reg 1 = some b.w; rw [reg_set_ne (by decide)]
      change s1.reg 1 = some b.w
      simpa [hds_eq] using hs1_1
    have h4s : X86Vm.step vmImage s4 = .ok { s4.set 2 (some (9223372036854775808 % X86Env.W)) with rip := 4196171 } := by
      rw [step_at_chunk 5 rfl rfl (by decide : (4196161, (.ofInt (.movI 2 9223372036854775808)), 10) ∈ SporosVmBin.vmTable_5),
        exec_movI]
    let s5 := { s4.set 2 (some (9223372036854775808 % X86Env.W)) with rip := 4196171 }
    have hr5 : RegsOk s5 := regsOk_set hr4 (by decide) _
    have hs5_0 : s5.reg 0 = some a.w := by
      change (s4.set 2 _).reg 0 = some a.w; rw [reg_set_ne (by decide)]; exact hs4_0
    have hs5_2 : s5.reg 2 = some (2 ^ 63) := by
      change (s4.set 2 _).reg 2 = some (2 ^ 63)
      rw [show (2 ^ 63 : Nat) = 9223372036854775808 % X86Env.W by decide]
      exact reg_set_self_ok' hr4 (by decide) _
    have hs5_1 : s5.reg 1 = some b.w := by
      change (s4.set 2 _).reg 1 = some b.w; rw [reg_set_ne (by decide)]; exact hs4_1
    have hs5_4 : s5.reg 4 = some vc.sp := by
      change (s4.set 2 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
      change (s3.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs1_4
    have hs5_15 : s5.reg 15 = some ctxBase := by
      change (s4.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
      change (s3.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15
    have h5 : X86Vm.step vmImage s5 =
        .ok { s5 with fl := (X86Int.aluF .sub a.w (2 ^ 63)).2, rip := 4196174 } := by
      rw [step_at_chunk 5 rfl rfl (by decide : (4196171, (.ofInt (.cmp 0 2)), 3) ∈ SporosVmBin.vmTable_5),
        exec_cmp hs5_0 hs5_2]
    let s6 := { s5 with fl := (X86Int.aluF .sub a.w (2 ^ 63)).2, rip := 4196174 }
    have hrch6 : Reaches vmImage s s6 :=
      Reaches.step_trans h0 (Reaches.step_trans h1 (Reaches.step_trans h2
        (Reaches.step_trans h3 (Reaches.step_trans h4s (Reaches.step h5)))))
    have finish_cqo_idiv : ∀ (t : X86Vm.St) (htrip : t.rip = 4196186) (htr : RegsOk t)
        (ht4 : t.reg 4 = some vc.sp) (ht15 : t.reg 15 = some ctxBase)
        (ht0 : t.reg 0 = some a.w) (ht1 : t.reg 1 = some b.w)
        (hrch_t : Reaches vmImage s t) (htm : t.mem = s.mem) (htk : t.k = s.k)
        (hne : ¬ (a.w = 2 ^ 63 ∧ b.w = X86Env.W - 1)),
        PrimPostOutcome pr ls f rest s .divs := by
      intro t htrip htr ht4 ht15 ht0 ht1 hrch_t htm htk hne
      let hi := if X86Int.msb (a.w % X86Env.W) = true then X86Env.W - 1 else 0
      have hc : X86Vm.step vmImage t = .ok { t.set 2 (some hi) with rip := 4196188 } := by
        rw [step_at_chunk 5 htrip rfl (by decide : (4196186, .cqo, 2) ∈ SporosVmBin.vmTable_5),
          exec_cqo ht0, htrip]
      let t1 := { t.set 2 (some hi) with rip := 4196188 }
      have htr1 : RegsOk t1 := regsOk_set htr (by decide) hi
      have ht1_0 : t1.reg 0 = some a.w := by
        change (t.set 2 _).reg 0 = some a.w; rw [reg_set_ne (by decide)]; exact ht0
      have ht1_2 : t1.reg 2 = some hi := reg_set_self_ok' htr (by decide) _
      have ht1_1 : t1.reg 1 = some b.w := by
        change (t.set 2 _).reg 1 = some b.w; rw [reg_set_ne (by decide)]; exact ht1
      have ha_mod : a.w % X86Env.W = a.w := Nat.mod_eq_of_lt haW'
      have hnum : X86Vm.toSigned64 hi * ↑X86Env.W + ↑(a.w % X86Env.W) = X86Vm.toSigned64 a.w := by
        dsimp only [hi]; rw [ha_mod]; exact toSigned64_hi_num haW'
      obtain ⟨hden0, hlo, hhi⟩ := idiv_bounds_of_ne_min_neg1 haW' hbW' hbz hne
      have hd_exec : X86Vm.step vmImage t1 =
          .ok { (t1.set 0 (some (X86Vm.ofSigned64 ((X86Vm.toSigned64 a.w).tdiv (X86Vm.toSigned64 b.w))))).set 2
                  (some (X86Vm.ofSigned64 ((X86Vm.toSigned64 a.w).tmod (X86Vm.toSigned64 b.w))))
                with fl := X86Int.Flags.undef, rip := 4196191 } := by
        have ht1_rip : t1.rip = 4196188 := rfl
        rw [step_at_chunk 5 rfl rfl (by decide : (4196188, (.idiv 1), 3) ∈ SporosVmBin.vmTable_5),
          exec_idiv ht1_0 ht1_2 ht1_1 hden0 (by rw [hnum]; exact hlo) (by rw [hnum]; exact hhi)]
        simp only [hnum, ht1_rip]
      let t2 := { (t1.set 0 (some (X86Vm.ofSigned64 ((X86Vm.toSigned64 a.w).tdiv (X86Vm.toSigned64 b.w))))).set 2
                    (some (X86Vm.ofSigned64 ((X86Vm.toSigned64 a.w).tmod (X86Vm.toSigned64 b.w))))
                  with fl := X86Int.Flags.undef, rip := 4196191 }
      have htr2 : RegsOk t2 := regsOk_set (regsOk_set htr1 (by decide) _) (by decide) _
      have ht2_4 : t2.reg 4 = some vc.sp := by
        change ((t1.set 0 _).set 2 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide), reg_set_ne (by decide)]
        change (t.set 2 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact ht4
      have ht2_15 : t2.reg 15 = some ctxBase := by
        change ((t1.set 0 _).set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide), reg_set_ne (by decide)]
        change (t.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact ht15
      have ht2_0 : t2.reg 0 = some (X86Vm.ofSigned64 ((X86Vm.toSigned64 a.w).tdiv (X86Vm.toSigned64 b.w))) := by
        change ((t1.set 0 _).set 2 _).reg 0 = _; rw [reg_set_ne (by decide)]; exact reg_set_self_ok' htr1 (by decide) _
      have hwf2 : vc.wf pr ls t2 := vc.wf_step_misc h htm htr2 ht2_4 ht2_15
      have hrch2 : Reaches vmImage s t2 :=
        Reaches.trans hrch_t (Reaches.step_trans hc (Reaches.step hd_exec))
      have hfin := jmp_mk1_finish_spec
        (step_at_chunk 5 rfl rfl (by decide : (4196191, (.ofInt (.jmp 4195210)), 5) ∈ SporosVmBin.vmTable_5))
        hwf2 (oracleAdmits_of_reaches hor hrch2) hrs hpcW hle ht2_0
        (by dsimp only [t2, t1, X86Int.St.set]; simpa [htm, hds_eq] using h272) hqW
        (by dsimp only [t2, t1, X86Int.St.set]; simpa [htm] using h280)
        (by dsimp only [t2, t1, X86Int.St.set]; simpa [htm] using h288)
      have hres := lift_next_outcome hrch2 htk hfin
      dsimp only [PrimPostOutcome]
      simp only [Sporos.Prim.eff, List.take_succ_cons, List.take_zero, SporosProvLen.lexecPrim, hds_eq,
        Sporos.wrap, M_eq_W', Nat.mod_eq_of_lt hbW', if_neg hbz, toSigned_eq_toSigned64 haW',
        toSigned_eq_toSigned64 hbW', ofInt_eq_ofSigned64] at hres ⊢
      exact hres
    by_cases ha_min : a.w = 2 ^ 63
    · have h6 : X86Vm.step vmImage s6 = .ok { s6 with rip := 4196176 } := by
        rw [step_at_chunk 5 rfl rfl (by decide : (4196174, (.ofInt (.jcc 5 4196186)), 2) ∈ SporosVmBin.vmTable_5),
          exec_jcc (cond_aluF_sub_ne a.w (2 ^ 63) haW' (by decide))]
        dsimp only [s6]; simp [ha_min]
      let s7 := { s6 with rip := 4196176 }
      have hs7_1 : s7.reg 1 = some b.w := hs5_1
      have h7 : X86Vm.step vmImage s7 =
          .ok { s7 with fl := (X86Int.aluF .sub b.w (X86Env.W - 1)).2, rip := 4196180 } := by
        rw [step_at_chunk 5 rfl rfl (by decide : (4196176, (.ofInt (.cmpI 1 18446744073709551615)), 4) ∈ SporosVmBin.vmTable_5),
          exec_cmpI hs7_1, show (18446744073709551615 : Nat) % X86Env.W = X86Env.W - 1 by decide]
      let s8 := { s7 with fl := (X86Int.aluF .sub b.w (X86Env.W - 1)).2, rip := 4196180 }
      have hrch8 : Reaches vmImage s s8 :=
        Reaches.trans hrch6 (Reaches.step_trans h6 (Reaches.step h7))
      by_cases hb_neg1 : b.w = X86Env.W - 1
      · have h8 : X86Vm.step vmImage s8 = .ok { s8 with rip := 4195210 } := by
          rw [step_at_chunk 5 rfl rfl (by decide : (4196180, (.ofInt (.jcc 4 4195210)), 6) ∈ SporosVmBin.vmTable_5),
            exec_jcc (cond_aluF_sub_e b.w (X86Env.W - 1) hbW' (by decide))]
          dsimp only [s8]; simp [hb_neg1]
        let s9 := { s8 with rip := 4195210 }
        have hrch9 : Reaches vmImage s s9 := Reaches.trans hrch8 (Reaches.step h8)
        have hs9_0 : s9.reg 0 = some (2 ^ 63) := by change s5.reg 0 = _; simpa [ha_min] using hs5_0
        have hwf9 : vc.wf pr ls s9 := vc.wf_step_misc h rfl hr5 hs5_4 hs5_15
        have hfin := finish_mk1_spec (s := s9) rfl hwf9 (oracleAdmits_of_reaches hor hrch9) hrs hpcW hle hs9_0
          (by rw [show s9.mem = s.mem from rfl]; simpa [hds_eq] using h272) hqW h280 h288
        have hres := lift_next_outcome hrch9 rfl hfin
        dsimp only [PrimPostOutcome]
        have hdiv_eq : Sporos.wrap (Sporos.ofInt ((Sporos.toSigned a.w).tdiv (Sporos.toSigned b.w))) =
            (2 ^ 63) % X86Env.W := by
          rw [ha_min, hb_neg1, divs_min_neg1_eq]; decide
        simp only [Sporos.Prim.eff, List.take_succ_cons, List.take_zero, SporosProvLen.lexecPrim, hds_eq,
          M_eq_W', Nat.mod_eq_of_lt hbW', if_neg hbz, hdiv_eq] at hres ⊢
        exact hres
      · have h8 : X86Vm.step vmImage s8 = .ok { s8 with rip := 4196186 } := by
          rw [step_at_chunk 5 rfl rfl (by decide : (4196180, (.ofInt (.jcc 4 4195210)), 6) ∈ SporosVmBin.vmTable_5),
            exec_jcc (cond_aluF_sub_e b.w (X86Env.W - 1) hbW' (by decide))]
          dsimp only [s8]; simp [hb_neg1]
        let s9 := { s8 with rip := 4196186 }
        have hrch9 : Reaches vmImage s s9 := Reaches.trans hrch8 (Reaches.step h8)
        exact finish_cqo_idiv s9 rfl hr5 hs5_4 hs5_15 hs5_0 hs5_1 hrch9 rfl rfl (fun ⟨_, h⟩ => hb_neg1 h)
    · have h6 : X86Vm.step vmImage s6 = .ok { s6 with rip := 4196186 } := by
        rw [step_at_chunk 5 rfl rfl (by decide : (4196174, (.ofInt (.jcc 5 4196186)), 2) ∈ SporosVmBin.vmTable_5),
          exec_jcc (cond_aluF_sub_ne a.w (2 ^ 63) haW' (by decide))]
        dsimp only [s6]; simp [ha_min]
      let s7 := { s6 with rip := 4196186 }
      have hrch7 : Reaches vmImage s s7 := Reaches.trans hrch6 (Reaches.step h6)
      exact finish_cqo_idiv s7 rfl hr5 hs5_4 hs5_15 hs5_0 hs5_1 hrch7 rfl rfl (fun ⟨h, _⟩ => ha_min h)

theorem P_mods_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_mods) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w) :
    PrimPostOutcome pr ls f rest s .mods := by
  have hrip' : s.rip = 4196196 := hrip
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hreg : RegsOk s := h.2.2.2.1
  have haW := (wf_ds_lt_W h (getElem!_mem_of_lt (by omega : 1 < ls.ds.length))).1
  have hbW := (wf_ds_lt_W h (getElem!_mem_of_lt (by omega : 0 < ls.ds.length))).1
  have h0 : X86Vm.step vmImage s = .ok { s.set 1 (some ls.ds[0]!.w) with rip := 4196203 } := by
    rw [step_at_chunk 5 hrip' rfl (by decide : (4196196, (.ofInt (.ld 1 15 296)), 7) ∈ SporosVmBin.vmTable_5),
      exec_ld h15 h296, hrip']
  let s1 := { s.set 1 (some ls.ds[0]!.w) with rip := 4196203 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) ls.ds[0]!.w
  have hs1_1 : s1.reg 1 = some ls.ds[0]!.w := reg_set_self_ok' hreg (by decide) _
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hs1_4 : s1.reg 4 = some vc.sp := by
    change (s.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have h1 : X86Vm.step vmImage s1 = .ok { s1 with fl := (X86Int.aluF .and ls.ds[0]!.w ls.ds[0]!.w).2, rip := 4196206 } := by
    rw [step_at_chunk 5 rfl rfl (by decide : (4196203, (.ofInt (.test 1 1)), 3) ∈ SporosVmBin.vmTable_5),
      exec_test hs1_1 hs1_1]
  let s2 := { s1 with fl := (X86Int.aluF .and ls.ds[0]!.w ls.ds[0]!.w).2, rip := 4196206 }
  have hle0 := hle
  rcases hds_eq : ls.ds with _ | ⟨b, _ | ⟨a, r⟩⟩ <;> try (simp [hds_eq] at hle0)
  have haW' : a.w < X86Env.W := by simpa [hds_eq] using haW
  have hbW' : b.w < X86Env.W := by simpa [hds_eq] using hbW
  have hqW : SporosProvLen.lq ([b, a]) ls.pc < X86Env.W := by
    dsimp [SporosProvLen.lq]; rw [← M_eq_W']; exact lt_W_of_mod _
  by_cases hbz : b.w = 0
  · have h2 : X86Vm.step vmImage s2 = .ok { s2 with rip := 4201167 } := by
      rw [step_at_chunk 5 rfl rfl (by decide : (4196206, (.ofInt (.jcc 4 4201167)), 6) ∈ SporosVmBin.vmTable_5),
        exec_jcc (cond_aluF_and_self_e ls.ds[0]!.w)]
      dsimp only [s2]; simp [hds_eq, hbz]
    let s3 := { s2 with rip := 4201167 }
    have hwf3 : vc.wf pr ls s3 := vc.wf_step_misc h rfl hr1 hs1_4 hs1_15
    have hfail := L_fail_from_vc_spec (e := .divZero) (s := s3) rfl hwf3
    have hrch3 := Reaches.step_trans h0 (Reaches.step_trans h1 (Reaches.step h2))
    dsimp only [PrimPostOutcome]
    simp only [Sporos.Prim.eff, List.take_succ_cons, List.take_zero, SporosProvLen.lexecPrim, hds_eq, hbz, Nat.zero_mod, ↓reduceIte]
    exact lift_fail_outcome hrch3 rfl hfail
  · have h2 : X86Vm.step vmImage s2 = .ok { s2 with rip := 4196212 } := by
      rw [step_at_chunk 5 rfl rfl (by decide : (4196206, (.ofInt (.jcc 4 4201167)), 6) ∈ SporosVmBin.vmTable_5),
        exec_jcc (cond_aluF_and_self_e ls.ds[0]!.w)]
      dsimp only [s2]; simp [hds_eq, hbz]
    let s3 := { s2 with rip := 4196212 }
    have hs3_1 : s3.reg 1 = some b.w := by change s1.reg 1 = _; simpa [hds_eq] using hs1_1
    have h3 : X86Vm.step vmImage s3 =
        .ok { s3 with fl := (X86Int.aluF .sub b.w (X86Env.W - 1)).2, rip := 4196216 } := by
      rw [step_at_chunk 5 rfl rfl (by decide : (4196212, (.ofInt (.cmpI 1 18446744073709551615)), 4) ∈ SporosVmBin.vmTable_5),
        exec_cmpI hs3_1, show (18446744073709551615 : Nat) % X86Env.W = X86Env.W - 1 by decide]
    let s4 := { s3 with fl := (X86Int.aluF .sub b.w (X86Env.W - 1)).2, rip := 4196216 }
    have hrch4 : Reaches vmImage s s4 :=
      Reaches.step_trans h0 (Reaches.step_trans h1 (Reaches.step_trans h2 (Reaches.step h3)))
    by_cases hbn : b.w = X86Env.W - 1
    · have h4s : X86Vm.step vmImage s4 = .ok { s4 with rip := 4196218 } := by
        rw [step_at_chunk 5 rfl rfl (by decide : (4196216, (.ofInt (.jcc 5 4196225)), 2) ∈ SporosVmBin.vmTable_5),
          exec_jcc (cond_aluF_sub_ne b.w (X86Env.W - 1) hbW' (by decide))]
        dsimp only [s4]; simp [hbn]
      let s5 := { s4 with rip := 4196218 }
      obtain ⟨r0, hr0⟩ := reg_of_ok hr1 0 (by decide) (by decide)
      have hs5_0 : s5.reg 0 = some r0 := hr0
      have hxor0 : (X86Int.aluF .xor r0 r0).1 = 0 := by rw [aluF_xor_val]; exact Nat.xor_self r0
      have hxor_pair : X86Int.aluF .xor r0 r0 = (0, (X86Int.aluF .xor r0 r0).2) := Prod.ext hxor0 rfl
      have h5 : X86Vm.step vmImage s5 =
          .ok { s5.set 0 (some 0) with fl := (X86Int.aluF .xor r0 r0).2, rip := 4196220 } := by
        rw [step_at_chunk 5 rfl rfl (by decide : (4196218, (.ofInt (.alu .xor 0 0)), 2) ∈ SporosVmBin.vmTable_5),
          exec_alu hs5_0 hs5_0, hxor_pair]
      let s6 := { s5.set 0 (some 0) with fl := (X86Int.aluF .xor r0 r0).2, rip := 4196220 }
      have hr6 : RegsOk s6 := regsOk_set hr1 (by decide) 0
      have hs6_4 : s6.reg 4 = some vc.sp := by
        change (s5.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs1_4
      have hs6_15 : s6.reg 15 = some ctxBase := by
        change (s5.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15
      have hs6_0 : s6.reg 0 = some 0 := reg_set_self_ok' hr1 (by decide) _
      have hwf6 : vc.wf pr ls s6 := vc.wf_step_misc h rfl hr6 hs6_4 hs6_15
      have hrch6 : Reaches vmImage s s6 :=
        Reaches.trans hrch4 (Reaches.step_trans h4s (Reaches.step h5))
      have hfin := jmp_mk1_finish_spec
        (step_at_chunk 5 rfl rfl (by decide : (4196220, (.ofInt (.jmp 4195210)), 5) ∈ SporosVmBin.vmTable_5))
        hwf6 (oracleAdmits_of_reaches hor hrch6) hrs hpcW hle hs6_0
        (by rw [show s6.mem = s.mem from rfl]; simpa [hds_eq] using h272) hqW h280 h288
      have hres := lift_next_outcome hrch6 rfl hfin
      dsimp only [PrimPostOutcome]
      have hmod_eq : Sporos.wrap (Sporos.ofInt ((Sporos.toSigned a.w).tmod (Sporos.toSigned b.w))) = 0 % X86Env.W := by
        rw [hbn, mods_neg1_eq]; rfl
      simp only [Sporos.Prim.eff, List.take_succ_cons, List.take_zero, SporosProvLen.lexecPrim, hds_eq,
        M_eq_W', Nat.mod_eq_of_lt hbW', if_neg hbz, hmod_eq] at hres ⊢
      exact hres
    · have h4s : X86Vm.step vmImage s4 = .ok { s4 with rip := 4196225 } := by
        rw [step_at_chunk 5 rfl rfl (by decide : (4196216, (.ofInt (.jcc 5 4196225)), 2) ∈ SporosVmBin.vmTable_5),
          exec_jcc (cond_aluF_sub_ne b.w (X86Env.W - 1) hbW' (by decide))]
        dsimp only [s4]; simp [hbn]
      let s5 := { s4 with rip := 4196225 }
      have hs5_15 : s5.reg 15 = some ctxBase := hs1_15
      have h5 : X86Vm.step vmImage s5 = .ok { s5.set 0 (some a.w) with rip := 4196232 } := by
        rw [step_at_chunk 5 rfl rfl (by decide : (4196225, (.ofInt (.ld 0 15 312)), 7) ∈ SporosVmBin.vmTable_5),
          exec_ld hs5_15 h312]; simp [hds_eq, s5]
      let s6 := { s5.set 0 (some a.w) with rip := 4196232 }
      have hr6 : RegsOk s6 := regsOk_set hr1 (by decide) a.w
      have hs6_0 : s6.reg 0 = some a.w := reg_set_self_ok' hr1 (by decide) _
      have hs6_1 : s6.reg 1 = some b.w := by
        change (s5.set 0 _).reg 1 = some b.w; rw [reg_set_ne (by decide)]; exact hs3_1
      let hi := if X86Int.msb (a.w % X86Env.W) = true then X86Env.W - 1 else 0
      have h6 : X86Vm.step vmImage s6 = .ok { s6.set 2 (some hi) with rip := 4196234 } := by
        rw [step_at_chunk 5 rfl rfl (by decide : (4196232, .cqo, 2) ∈ SporosVmBin.vmTable_5),
          exec_cqo hs6_0]
      let s7 := { s6.set 2 (some hi) with rip := 4196234 }
      have hr7 : RegsOk s7 := regsOk_set hr6 (by decide) hi
      have hs7_0 : s7.reg 0 = some a.w := by
        change (s6.set 2 _).reg 0 = some a.w; rw [reg_set_ne (by decide)]; exact hs6_0
      have hs7_2 : s7.reg 2 = some hi := reg_set_self_ok' hr6 (by decide) _
      have hs7_1 : s7.reg 1 = some b.w := by
        change (s6.set 2 _).reg 1 = some b.w; rw [reg_set_ne (by decide)]; exact hs6_1
      have ha_mod : a.w % X86Env.W = a.w := Nat.mod_eq_of_lt haW'
      have hnum : X86Vm.toSigned64 hi * ↑X86Env.W + ↑(a.w % X86Env.W) = X86Vm.toSigned64 a.w := by
        dsimp only [hi]; rw [ha_mod]; exact toSigned64_hi_num haW'
      obtain ⟨hden0, hlo, hhi⟩ := idiv_bounds_of_ne_min_neg1 haW' hbW' hbz (fun ⟨_, h⟩ => hbn h)
      have h7 : X86Vm.step vmImage s7 =
          .ok { (s7.set 0 (some (X86Vm.ofSigned64 ((X86Vm.toSigned64 a.w).tdiv (X86Vm.toSigned64 b.w))))).set 2
                  (some (X86Vm.ofSigned64 ((X86Vm.toSigned64 a.w).tmod (X86Vm.toSigned64 b.w))))
                with fl := X86Int.Flags.undef, rip := 4196237 } := by
        have hs7_rip : s7.rip = 4196234 := rfl
        rw [step_at_chunk 5 rfl rfl (by decide : (4196234, (.idiv 1), 3) ∈ SporosVmBin.vmTable_5),
          exec_idiv hs7_0 hs7_2 hs7_1 hden0 (by rw [hnum]; exact hlo) (by rw [hnum]; exact hhi)]
        simp only [hnum, hs7_rip]
      let s8 := { (s7.set 0 (some (X86Vm.ofSigned64 ((X86Vm.toSigned64 a.w).tdiv (X86Vm.toSigned64 b.w))))).set 2
                    (some (X86Vm.ofSigned64 ((X86Vm.toSigned64 a.w).tmod (X86Vm.toSigned64 b.w))))
                  with fl := X86Int.Flags.undef, rip := 4196237 }
      have hr8 : RegsOk s8 := regsOk_set (regsOk_set hr7 (by decide) _) (by decide) _
      have hs8_2 : s8.reg 2 = some (X86Vm.ofSigned64 ((X86Vm.toSigned64 a.w).tmod (X86Vm.toSigned64 b.w))) :=
        reg_set_self_ok' (regsOk_set hr7 (by decide) _) (by decide) _
      have h8 : X86Vm.step vmImage s8 =
          .ok { s8.set 0 (some (X86Vm.ofSigned64 ((X86Vm.toSigned64 a.w).tmod (X86Vm.toSigned64 b.w)))) with rip := 4196240 } := by
        rw [step_at_chunk 5 rfl rfl (by decide : (4196237, (.ofInt (.mov 0 2)), 3) ∈ SporosVmBin.vmTable_5),
          exec_mov hs8_2]
      let s9 := { s8.set 0 (some (X86Vm.ofSigned64 ((X86Vm.toSigned64 a.w).tmod (X86Vm.toSigned64 b.w)))) with rip := 4196240 }
      have hr9 : RegsOk s9 := regsOk_set hr8 (by decide) _
      have hs9_4 : s9.reg 4 = some vc.sp := by
        change (s8.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
        change ((s7.set 0 _).set 2 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide), reg_set_ne (by decide)]
        change (s6.set 2 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
        change (s5.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs1_4
      have hs9_15 : s9.reg 15 = some ctxBase := by
        change (s8.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
        change ((s7.set 0 _).set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide), reg_set_ne (by decide)]
        change (s6.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
        change (s5.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15
      have hs9_0 : s9.reg 0 = some (X86Vm.ofSigned64 ((X86Vm.toSigned64 a.w).tmod (X86Vm.toSigned64 b.w))) :=
        reg_set_self_ok' hr8 (by decide) _
      have hwf9 : vc.wf pr ls s9 := vc.wf_step_misc h rfl hr9 hs9_4 hs9_15
      have hrch9 : Reaches vmImage s s9 :=
        Reaches.trans hrch4 (Reaches.step_trans h4s (Reaches.step_trans h5
          (Reaches.step_trans h6 (Reaches.step_trans h7 (Reaches.step h8)))))
      have hfin := jmp_mk1_finish_spec
        (step_at_chunk 5 rfl rfl (by decide : (4196240, (.ofInt (.jmp 4195210)), 5) ∈ SporosVmBin.vmTable_5))
        hwf9 (oracleAdmits_of_reaches hor hrch9) hrs hpcW hle hs9_0
        (by rw [show s9.mem = s.mem from rfl]; simpa [hds_eq] using h272) hqW h280 h288
      have hres := lift_next_outcome hrch9 rfl hfin
      dsimp only [PrimPostOutcome]
      simp only [Sporos.Prim.eff, List.take_succ_cons, List.take_zero, SporosProvLen.lexecPrim, hds_eq,
        Sporos.wrap, M_eq_W', Nat.mod_eq_of_lt hbW', if_neg hbz, toSigned_eq_toSigned64 haW',
        toSigned_eq_toSigned64 hbW', ofInt_eq_ofSigned64] at hres ⊢
      exact hres

theorem P_neg_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_neg) (hle : 1 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 1) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 1) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) :
    PrimPostOutcome pr ls f rest s .neg := by
  have hrip' : s.rip = 4196245 := hrip
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hreg : RegsOk s := h.2.2.2.1
  have h0 : X86Vm.step vmImage s = .ok { s.set 0 (some ls.ds[0]!.w) with rip := 4196252 } := by
    rw [step_at_chunk 5 hrip' rfl (by decide : (4196245, (.ofInt (.ld 0 15 296)), 7) ∈ SporosVmBin.vmTable_5),
      exec_ld h15 h296, hrip']
  let s1 := { s.set 0 (some ls.ds[0]!.w) with rip := 4196252 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) ls.ds[0]!.w
  have hs1_0 : s1.reg 0 = some ls.ds[0]!.w := reg_set_self_ok' hreg (by decide) _
  have hs1_4 : s1.reg 4 = some vc.sp := by
    change (s.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  let vf := X86Int.aluF .sub 0 (ls.ds[0]!.w % X86Env.W)
  have h1 : X86Vm.step vmImage s1 = .ok { s1.set 0 (some vf.1) with fl := vf.2, rip := 4196255 } := by
    rw [step_at_chunk 5 rfl rfl (by decide : (4196252, (.neg 0), 3) ∈ SporosVmBin.vmTable_5),
      exec_neg hs1_0]
  let s2 := { s1.set 0 (some vf.1) with fl := vf.2, rip := 4196255 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) vf.1
  have hs2_4 : s2.reg 4 = some vc.sp := by
    change (s1.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs1_4
  have hs2_15 : s2.reg 15 = some ctxBase := by
    change (s1.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15
  have hs2_0 : s2.reg 0 = some vf.1 := reg_set_self_ok' hr1 (by decide) _
  have hwf2 : vc.wf pr ls s2 := vc.wf_step_misc h rfl hr2 hs2_4 hs2_15
  have hqW : SporosProvLen.lq (ls.ds.take 1) ls.pc < X86Env.W := by
    dsimp [SporosProvLen.lq]; rw [← M_eq_W']; exact lt_W_of_mod _
  have hrch2 := Reaches.step_trans h0 (Reaches.step h1)
  have hfin := jmp_mk1_finish_spec
    (step_at_chunk 5 rfl rfl (by decide : (4196255, (.ofInt (.jmp 4195210)), 5) ∈ SporosVmBin.vmTable_5))
    hwf2 (oracleAdmits_of_reaches hor hrch2) hrs hpcW hle hs2_0 h272 hqW h280 h288
  have hres := lift_next_outcome hrch2 rfl hfin
  have hle0 := hle
  rcases hds_eq : ls.ds with _ | ⟨a, r⟩ <;> try (simp [hds_eq] at hle0)
  have hv_eq : vf.1 % X86Env.W = Sporos.wrap (Sporos.M - a.w % Sporos.M) := by
    have hvf1 : vf.1 = (0 + X86Env.W - a.w % X86Env.W) % X86Env.W := by
      dsimp [vf]; rw [aluF_sub_val, hds_eq]; rfl
    rw [hvf1, wrap_eq_mod, M_eq_W', Nat.zero_add, Nat.mod_mod]
  dsimp only [PrimPostOutcome]
  simp only [Sporos.Prim.eff, List.take_succ_cons, List.take_zero, SporosProvLen.lexecPrim, hds_eq]
  simpa only [hds_eq, hv_eq, List.take_succ_cons, List.take_zero, List.singleton_append] using hres

theorem P_band_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_band) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w) :
    PrimPostOutcome pr ls f rest s .band := by
  have hle0 := hle
  rcases hds_eq : ls.ds with _ | ⟨b, _ | ⟨a, r⟩⟩ <;> try (simp [hds_eq] at hle0)
  exact alu_bin_mk1_spec (op := .and) (c0 := 5) (c1 := 5) (c2 := 5) (c3 := 5)
    (resFn := fun x y => Sporos.wrap (Nat.land x y))
    hrip (by decide) (by decide) (by decide) (by decide) rfl hle h hor hrs hpcW
    h272 h280 h288 h296 h312
    (fun x _ hx _ => by
      (try dsimp only); rw [aluF_and_val, wrap_eq_mod, Nat.mod_eq_of_lt (land_lt_W hx)])
    (by simp [hds_eq, SporosProvLen.lexecPrim])

theorem P_bor_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_bor) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w) :
    PrimPostOutcome pr ls f rest s .bor := by
  have hle0 := hle
  rcases hds_eq : ls.ds with _ | ⟨b, _ | ⟨a, r⟩⟩ <;> try (simp [hds_eq] at hle0)
  exact alu_bin_mk1_spec (op := .or) (c0 := 5) (c1 := 5) (c2 := 5) (c3 := 5)
    (resFn := fun x y => Sporos.wrap (Nat.lor x y))
    hrip (by decide) (by decide) (by decide) (by decide) rfl hle h hor hrs hpcW
    h272 h280 h288 h296 h312
    (fun x y hx hy => by
      (try dsimp only); rw [aluF_or_val, wrap_eq_mod, Nat.mod_eq_of_lt (lor_lt_W hx hy)])
    (by simp [hds_eq, SporosProvLen.lexecPrim])

theorem P_bxor_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_bxor) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w) :
    PrimPostOutcome pr ls f rest s .bxor := by
  have hle0 := hle
  rcases hds_eq : ls.ds with _ | ⟨b, _ | ⟨a, r⟩⟩ <;> try (simp [hds_eq] at hle0)
  exact alu_bin_mk1_spec (op := .xor) (c0 := 5) (c1 := 5) (c2 := 6) (c3 := 6)
    (resFn := fun x y => Sporos.wrap (Nat.xor x y))
    hrip (by decide) (by decide) (by decide) (by decide) rfl hle h hor hrs hpcW
    h272 h280 h288 h296 h312
    (fun x y hx hy => by
      (try dsimp only); rw [aluF_xor_val, wrap_eq_mod, Nat.mod_eq_of_lt (lxor_lt_W hx hy)])
    (by simp [hds_eq, SporosProvLen.lexecPrim])

theorem P_bnot_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_bnot) (hle : 1 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 1) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 1) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) :
    PrimPostOutcome pr ls f rest s .bnot := by
  have hrip' : s.rip = 4196326 := hrip
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hreg : RegsOk s := h.2.2.2.1
  have h0 : X86Vm.step vmImage s = .ok { s.set 0 (some ls.ds[0]!.w) with rip := 4196333 } := by
    rw [step_at_chunk 6 hrip' rfl (by decide : (4196326, (.ofInt (.ld 0 15 296)), 7) ∈ SporosVmBin.vmTable_6),
      exec_ld h15 h296, hrip']
  let s1 := { s.set 0 (some ls.ds[0]!.w) with rip := 4196333 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) ls.ds[0]!.w
  have hs1_0 : s1.reg 0 = some ls.ds[0]!.w := reg_set_self_ok' hreg (by decide) _
  have hs1_4 : s1.reg 4 = some vc.sp := by
    change (s.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  let res := X86Env.W - 1 - ls.ds[0]!.w % X86Env.W
  have h1 : X86Vm.step vmImage s1 = .ok { s1.set 0 (some res) with rip := 4196336 } := by
    rw [step_at_chunk 6 rfl rfl (by decide : (4196333, (.not 0), 3) ∈ SporosVmBin.vmTable_6),
      exec_not hs1_0]
  let s2 := { s1.set 0 (some res) with rip := 4196336 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) res
  have hs2_4 : s2.reg 4 = some vc.sp := by
    change (s1.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs1_4
  have hs2_15 : s2.reg 15 = some ctxBase := by
    change (s1.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15
  have hs2_0 : s2.reg 0 = some res := reg_set_self_ok' hr1 (by decide) _
  have hwf2 : vc.wf pr ls s2 := vc.wf_step_misc h rfl hr2 hs2_4 hs2_15
  have hqW : SporosProvLen.lq (ls.ds.take 1) ls.pc < X86Env.W := by
    dsimp [SporosProvLen.lq]; rw [← M_eq_W']; exact lt_W_of_mod _
  have hrch2 := Reaches.step_trans h0 (Reaches.step h1)
  have hfin := jmp_mk1_finish_spec
    (step_at_chunk 6 rfl rfl (by decide : (4196336, (.ofInt (.jmp 4195210)), 5) ∈ SporosVmBin.vmTable_6))
    hwf2 (oracleAdmits_of_reaches hor hrch2) hrs hpcW hle hs2_0 h272 hqW h280 h288
  have hres := lift_next_outcome hrch2 rfl hfin
  have hle0 := hle
  rcases hds_eq : ls.ds with _ | ⟨a, r⟩ <;> try (simp [hds_eq] at hle0)
  have hres_eq : res % X86Env.W = Sporos.wrap (Sporos.M - 1 - a.w % Sporos.M) := by
    dsimp [res]; rw [wrap_eq_mod, M_eq_W', hds_eq]; rfl
  dsimp only [PrimPostOutcome]
  simp only [Sporos.Prim.eff, List.take_succ_cons, List.take_zero, SporosProvLen.lexecPrim, hds_eq]
  simpa only [hds_eq, hres_eq, List.take_succ_cons, List.take_zero, List.singleton_append] using hres

theorem P_shl_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_shl) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w) :
    PrimPostOutcome pr ls f rest s .shl := by
  have hrip' : s.rip = 4196341 := hrip
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hreg : RegsOk s := h.2.2.2.1
  have haW := (wf_ds_lt_W h (getElem!_mem_of_lt (by omega : 1 < ls.ds.length))).1
  have h0 : X86Vm.step vmImage s = .ok { s.set 0 (some ls.ds[1]!.w) with rip := 4196348 } := by
    rw [step_at_chunk 6 hrip' rfl (by decide : (4196341, (.ofInt (.ld 0 15 312)), 7) ∈ SporosVmBin.vmTable_6),
      exec_ld h15 h312, hrip']
  let s1 := { s.set 0 (some ls.ds[1]!.w) with rip := 4196348 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) ls.ds[1]!.w
  have hs1_0 : s1.reg 0 = some ls.ds[1]!.w := reg_set_self_ok' hreg (by decide) _
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hs1_4 : s1.reg 4 = some vc.sp := by
    change (s.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have h1 : X86Vm.step vmImage s1 = .ok { s1.set 1 (some ls.ds[0]!.w) with rip := 4196355 } := by
    rw [step_at_chunk 6 rfl rfl (by decide : (4196348, (.ofInt (.ld 1 15 296)), 7) ∈ SporosVmBin.vmTable_6),
      exec_ld hs1_15 h296]
  let s2 := { s1.set 1 (some ls.ds[0]!.w) with rip := 4196355 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) ls.ds[0]!.w
  have hs2_0 : s2.reg 0 = some ls.ds[1]!.w := by
    change (s1.set 1 _).reg 0 = some ls.ds[1]!.w; rw [reg_set_ne (by decide)]; exact hs1_0
  have hs2_1 : s2.reg 1 = some ls.ds[0]!.w := reg_set_self_ok' hr1 (by decide) _
  let res := (ls.ds[1]!.w % X86Env.W) * 2 ^ (ls.ds[0]!.w % 64) % X86Env.W
  have h2 : X86Vm.step vmImage s2 = .ok { s2.set 0 (some res) with fl := X86Int.Flags.undef, rip := 4196358 } := by
    rw [step_at_chunk 6 rfl rfl (by decide : (4196355, (.shlR 0), 3) ∈ SporosVmBin.vmTable_6),
      exec_shlR hs2_0 hs2_1]
  let s3 := { s2.set 0 (some res) with fl := X86Int.Flags.undef, rip := 4196358 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) res
  have hs3_4 : s3.reg 4 = some vc.sp := by
    change (s2.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s1.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs1_4
  have hs3_15 : s3.reg 15 = some ctxBase := by
    change (s2.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s1.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15
  have hs3_0 : s3.reg 0 = some res := reg_set_self_ok' hr2 (by decide) _
  have hwf3 : vc.wf pr ls s3 := vc.wf_step_misc h rfl hr3 hs3_4 hs3_15
  have hqW : SporosProvLen.lq (ls.ds.take 2) ls.pc < X86Env.W := by
    dsimp [SporosProvLen.lq]; rw [← M_eq_W']; exact lt_W_of_mod _
  have hrch3 := Reaches.step_trans h0 (Reaches.step_trans h1 (Reaches.step h2))
  have hfin := jmp_mk1_finish_spec
    (step_at_chunk 6 rfl rfl (by decide : (4196358, (.ofInt (.jmp 4195210)), 5) ∈ SporosVmBin.vmTable_6))
    hwf3 (oracleAdmits_of_reaches hor hrch3) hrs hpcW hle hs3_0 h272 hqW h280 h288
  have hres := lift_next_outcome hrch3 rfl hfin
  have hle0 := hle
  rcases hds_eq : ls.ds with _ | ⟨b, _ | ⟨a, r⟩⟩ <;> try (simp [hds_eq] at hle0)
  have haW' : a.w < X86Env.W := by simpa [hds_eq] using haW
  have hres_eq : res % X86Env.W = Sporos.wrap (a.w * 2 ^ (b.w % 64)) := by
    dsimp [res]; rw [hds_eq, wrap_eq_mod]; simp [Nat.mod_eq_of_lt haW']
  dsimp only [PrimPostOutcome]
  simp only [Sporos.Prim.eff, List.take_succ_cons, List.take_zero, SporosProvLen.lexecPrim, hds_eq]
  simpa only [hds_eq, hres_eq, List.take_succ_cons, List.take_zero, List.singleton_append] using hres

theorem P_shr_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_shr) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w) :
    PrimPostOutcome pr ls f rest s .shr := by
  have hrip' : s.rip = 4196363 := hrip
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hreg : RegsOk s := h.2.2.2.1
  have haW := (wf_ds_lt_W h (getElem!_mem_of_lt (by omega : 1 < ls.ds.length))).1
  have h0 : X86Vm.step vmImage s = .ok { s.set 0 (some ls.ds[1]!.w) with rip := 4196370 } := by
    rw [step_at_chunk 6 hrip' rfl (by decide : (4196363, (.ofInt (.ld 0 15 312)), 7) ∈ SporosVmBin.vmTable_6),
      exec_ld h15 h312, hrip']
  let s1 := { s.set 0 (some ls.ds[1]!.w) with rip := 4196370 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) ls.ds[1]!.w
  have hs1_0 : s1.reg 0 = some ls.ds[1]!.w := reg_set_self_ok' hreg (by decide) _
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hs1_4 : s1.reg 4 = some vc.sp := by
    change (s.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have h1 : X86Vm.step vmImage s1 = .ok { s1.set 1 (some ls.ds[0]!.w) with rip := 4196377 } := by
    rw [step_at_chunk 6 rfl rfl (by decide : (4196370, (.ofInt (.ld 1 15 296)), 7) ∈ SporosVmBin.vmTable_6),
      exec_ld hs1_15 h296]
  let s2 := { s1.set 1 (some ls.ds[0]!.w) with rip := 4196377 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) ls.ds[0]!.w
  have hs2_0 : s2.reg 0 = some ls.ds[1]!.w := by
    change (s1.set 1 _).reg 0 = some ls.ds[1]!.w; rw [reg_set_ne (by decide)]; exact hs1_0
  have hs2_1 : s2.reg 1 = some ls.ds[0]!.w := reg_set_self_ok' hr1 (by decide) _
  let res := (ls.ds[1]!.w % X86Env.W) / 2 ^ (ls.ds[0]!.w % 64)
  have h2 : X86Vm.step vmImage s2 = .ok { s2.set 0 (some res) with fl := X86Int.Flags.undef, rip := 4196380 } := by
    rw [step_at_chunk 6 rfl rfl (by decide : (4196377, (.shrR 0), 3) ∈ SporosVmBin.vmTable_6),
      exec_shrR hs2_0 hs2_1]
  let s3 := { s2.set 0 (some res) with fl := X86Int.Flags.undef, rip := 4196380 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) res
  have hs3_4 : s3.reg 4 = some vc.sp := by
    change (s2.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s1.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs1_4
  have hs3_15 : s3.reg 15 = some ctxBase := by
    change (s2.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s1.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15
  have hs3_0 : s3.reg 0 = some res := reg_set_self_ok' hr2 (by decide) _
  have hwf3 : vc.wf pr ls s3 := vc.wf_step_misc h rfl hr3 hs3_4 hs3_15
  have hqW : SporosProvLen.lq (ls.ds.take 2) ls.pc < X86Env.W := by
    dsimp [SporosProvLen.lq]; rw [← M_eq_W']; exact lt_W_of_mod _
  have hrch3 := Reaches.step_trans h0 (Reaches.step_trans h1 (Reaches.step h2))
  have hfin := jmp_mk1_finish_spec
    (step_at_chunk 6 rfl rfl (by decide : (4196380, (.ofInt (.jmp 4195210)), 5) ∈ SporosVmBin.vmTable_6))
    hwf3 (oracleAdmits_of_reaches hor hrch3) hrs hpcW hle hs3_0 h272 hqW h280 h288
  have hres := lift_next_outcome hrch3 rfl hfin
  have hle0 := hle
  rcases hds_eq : ls.ds with _ | ⟨b, _ | ⟨a, r⟩⟩ <;> try (simp [hds_eq] at hle0)
  have haW' : a.w < X86Env.W := by simpa [hds_eq] using haW
  have hres_eq : res % X86Env.W = Sporos.wrap (a.w / 2 ^ (b.w % 64)) := by
    dsimp [res]; rw [hds_eq, wrap_eq_mod]; simp [Nat.mod_eq_of_lt haW']
  dsimp only [PrimPostOutcome]
  simp only [Sporos.Prim.eff, List.take_succ_cons, List.take_zero, SporosProvLen.lexecPrim, hds_eq]
  simpa only [hds_eq, hres_eq, List.take_succ_cons, List.take_zero, List.singleton_append] using hres

theorem P_sar_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_sar) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w) :
    PrimPostOutcome pr ls f rest s .sar := by
  have hrip' : s.rip = 4196385 := hrip
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hreg : RegsOk s := h.2.2.2.1
  have haW := (wf_ds_lt_W h (getElem!_mem_of_lt (by omega : 1 < ls.ds.length))).1
  have h0 : X86Vm.step vmImage s = .ok { s.set 0 (some ls.ds[1]!.w) with rip := 4196392 } := by
    rw [step_at_chunk 6 hrip' rfl (by decide : (4196385, (.ofInt (.ld 0 15 312)), 7) ∈ SporosVmBin.vmTable_6),
      exec_ld h15 h312, hrip']
  let s1 := { s.set 0 (some ls.ds[1]!.w) with rip := 4196392 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) ls.ds[1]!.w
  have hs1_0 : s1.reg 0 = some ls.ds[1]!.w := reg_set_self_ok' hreg (by decide) _
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hs1_4 : s1.reg 4 = some vc.sp := by
    change (s.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have h1 : X86Vm.step vmImage s1 = .ok { s1.set 1 (some ls.ds[0]!.w) with rip := 4196399 } := by
    rw [step_at_chunk 6 rfl rfl (by decide : (4196392, (.ofInt (.ld 1 15 296)), 7) ∈ SporosVmBin.vmTable_6),
      exec_ld hs1_15 h296]
  let s2 := { s1.set 1 (some ls.ds[0]!.w) with rip := 4196399 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) ls.ds[0]!.w
  have hs2_0 : s2.reg 0 = some ls.ds[1]!.w := by
    change (s1.set 1 _).reg 0 = some ls.ds[1]!.w; rw [reg_set_ne (by decide)]; exact hs1_0
  have hs2_1 : s2.reg 1 = some ls.ds[0]!.w := reg_set_self_ok' hr1 (by decide) _
  let res := X86Vm.sar64 ls.ds[1]!.w ls.ds[0]!.w
  have h2 : X86Vm.step vmImage s2 = .ok { s2.set 0 (some res) with fl := X86Int.Flags.undef, rip := 4196402 } := by
    rw [step_at_chunk 6 rfl rfl (by decide : (4196399, (.sarR 0), 3) ∈ SporosVmBin.vmTable_6),
      exec_sarR hs2_0 hs2_1]
  let s3 := { s2.set 0 (some res) with fl := X86Int.Flags.undef, rip := 4196402 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) res
  have hs3_4 : s3.reg 4 = some vc.sp := by
    change (s2.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
    change (s1.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs1_4
  have hs3_15 : s3.reg 15 = some ctxBase := by
    change (s2.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
    change (s1.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15
  have hs3_0 : s3.reg 0 = some res := reg_set_self_ok' hr2 (by decide) _
  have hwf3 : vc.wf pr ls s3 := vc.wf_step_misc h rfl hr3 hs3_4 hs3_15
  have hqW : SporosProvLen.lq (ls.ds.take 2) ls.pc < X86Env.W := by
    dsimp [SporosProvLen.lq]; rw [← M_eq_W']; exact lt_W_of_mod _
  have hrch3 := Reaches.step_trans h0 (Reaches.step_trans h1 (Reaches.step h2))
  have hfin := jmp_mk1_finish_spec
    (step_at_chunk 6 rfl rfl (by decide : (4196402, (.ofInt (.jmp 4195210)), 5) ∈ SporosVmBin.vmTable_6))
    hwf3 (oracleAdmits_of_reaches hor hrch3) hrs hpcW hle hs3_0 h272 hqW h280 h288
  have hres := lift_next_outcome hrch3 rfl hfin
  have hle0 := hle
  rcases hds_eq : ls.ds with _ | ⟨b, _ | ⟨a, r⟩⟩ <;> try (simp [hds_eq] at hle0)
  have haW' : a.w < X86Env.W := by simpa [hds_eq] using haW
  have hres_eq : res % X86Env.W = Sporos.wrap (Sporos.ofInt (Sporos.toSigned a.w / (2 ^ (b.w % 64) : Nat))) := by
    dsimp [res]; rw [hds_eq]; simp [wrap_eq_mod, sar64_eq_sporos haW']
  dsimp only [PrimPostOutcome]
  simp only [Sporos.Prim.eff, List.take_succ_cons, List.take_zero, SporosProvLen.lexecPrim, hds_eq]
  simpa only [hds_eq, hres_eq, List.take_succ_cons, List.take_zero, List.singleton_append] using hres

/-! ### Comparison primitives (`eq`, `ne`, `ltu`, `lts`, `leu`, `les`) -/

private theorem wrap_bool (b : Bool) : Sporos.wrap (Sporos.bool b) = Sporos.bool b := by
  cases b <;> decide

theorem P_eq_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_eq) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w) :
    PrimPostOutcome pr ls f rest s .eq := by
  have hle0 := hle
  rcases hds_eq : ls.ds with _ | ⟨b, _ | ⟨a, r⟩⟩ <;> try (simp [hds_eq] at hle0)
  exact cmp_bin_mk1_spec (cc := 4) (cmpBool := fun x y => decide (x = y))
    hrip (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    rfl hle h hor hrs hpcW h272 h280 h288 h296 h312
    (fun x y hx hy => cond_aluF_sub_e x y hx hy)
    (by simp [hds_eq, SporosProvLen.lexecPrim, wrap_bool])

theorem P_ne_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_ne) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w) :
    PrimPostOutcome pr ls f rest s .ne := by
  have hle0 := hle
  rcases hds_eq : ls.ds with _ | ⟨b, _ | ⟨a, r⟩⟩ <;> try (simp [hds_eq] at hle0)
  exact cmp_bin_mk1_spec (cc := 5) (cmpBool := fun x y => decide (x ≠ y))
    hrip (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    rfl hle h hor hrs hpcW h272 h280 h288 h296 h312
    (fun x y hx hy => cond_aluF_sub_ne x y hx hy)
    (by simp [hds_eq, SporosProvLen.lexecPrim, wrap_bool])

theorem P_ltu_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_ltu) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w) :
    PrimPostOutcome pr ls f rest s .ltu := by
  have hle0 := hle
  rcases hds_eq : ls.ds with _ | ⟨b, _ | ⟨a, r⟩⟩ <;> try (simp [hds_eq] at hle0)
  exact cmp_bin_mk1_spec (cc := 2) (cmpBool := fun x y => decide (x < y))
    hrip (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    rfl hle h hor hrs hpcW h272 h280 h288 h296 h312
    (fun x y _ _ => cond_aluF_sub_b x y)
    (by simp [hds_eq, SporosProvLen.lexecPrim, wrap_bool])

theorem P_lts_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_lts) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w) :
    PrimPostOutcome pr ls f rest s .lts := by
  have hle0 := hle
  rcases hds_eq : ls.ds with _ | ⟨b, _ | ⟨a, r⟩⟩ <;> try (simp [hds_eq] at hle0)
  have haW : a.w < X86Env.W := (wf_ds_lt_W h (by rw [hds_eq]; simp)).1
  have hbW : b.w < X86Env.W := (wf_ds_lt_W h (by rw [hds_eq]; simp)).1
  exact cmp_bin_mk1_spec (cc := 12) (cmpBool := fun x y => decide (X86Vm.toSigned64 x < X86Vm.toSigned64 y))
    hrip (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    rfl hle h hor hrs hpcW h272 h280 h288 h296 h312
    (fun x y hx hy => toSigned64_lt_iff x y hx hy)
    (by simp [hds_eq, SporosProvLen.lexecPrim, wrap_bool, toSigned_eq_toSigned64 haW, toSigned_eq_toSigned64 hbW])

theorem P_leu_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_leu) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w) :
    PrimPostOutcome pr ls f rest s .leu := by
  have hle0 := hle
  rcases hds_eq : ls.ds with _ | ⟨b, _ | ⟨a, r⟩⟩ <;> try (simp [hds_eq] at hle0)
  exact cmp_bin_mk1_spec (cc := 6) (cmpBool := fun x y => decide (x ≤ y))
    hrip (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    rfl hle h hor hrs hpcW h272 h280 h288 h296 h312
    (fun x y hx hy => cond_aluF_sub_be x y hx hy)
    (by simp [hds_eq, SporosProvLen.lexecPrim, wrap_bool])

theorem P_les_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_les) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w) :
    PrimPostOutcome pr ls f rest s .les := by
  have hle0 := hle
  rcases hds_eq : ls.ds with _ | ⟨b, _ | ⟨a, r⟩⟩ <;> try (simp [hds_eq] at hle0)
  have haW : a.w < X86Env.W := (wf_ds_lt_W h (by rw [hds_eq]; simp)).1
  have hbW : b.w < X86Env.W := (wf_ds_lt_W h (by rw [hds_eq]; simp)).1
  exact cmp_bin_mk1_spec (cc := 14) (cmpBool := fun x y => decide (X86Vm.toSigned64 x ≤ X86Vm.toSigned64 y))
    hrip (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    rfl hle h hor hrs hpcW h272 h280 h288 h296 h312
    (fun x y hx hy => toSigned64_le_iff x y hx hy)
    (by simp [hds_eq, SporosProvLen.lexecPrim, wrap_bool, toSigned_eq_toSigned64 haW, toSigned_eq_toSigned64 hbW])

/-! ### Exit, environment, provenance, and assertion primitives -/

private theorem land_255_eq_mod (x : Nat) : Nat.land x 255 = x % 256 := by
  change x &&& (2 ^ 8 - 1) = x % 2 ^ 8
  exact Nat.and_two_pow_sub_one_eq_mod x 8

theorem VmCtx.halt_of_wf {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {exitCode : Nat}
    (hrip : s.rip = SporosVmBin.L_vm_halt) (h : vc.wf pr ls s)
    (h448 : rd64 s.mem (ea ctxBase 448) = some exitCode) :
    ∃ s_exit, Reaches vmImage s s_exit ∧
      X86Vm.step vmImage s_exit = .exit (exitCode % 256) ∧
      s_exit.k.out = s.k.out ++ ls.out.reverse := by
  have hc448 : leVal ((vc.ctxReg.data.drop 448).take 8) = exitCode :=
    ctx_leVal_of_rd64 h (by decide) h448
  have hcb := vc.ctx_bounds h
  have ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, houtHi,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, hc200, hc208,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, horel, _⟩ := h
  exact L_vm_halt_spec (_pr := pr) hrip h.1 h.2.1 h.2.2.2.1 h.2.2.2.2.2.1 (vc.mem_ctx h)
    hcb.1 hcb.2.2.1 (vc.mem_out h) houtHi hc200 hc208 hc448 horel

theorem P_exit_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_exit) (hle : 1 ≤ ls.ds.length)
    (h : vc.wf pr ls s)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) :
    PrimPostOutcome pr ls f rest s .exit := by
  have hrip' : s.rip = 4198897 := hrip
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hreg : RegsOk s := h.2.2.2.1
  have h0 : X86Vm.step vmImage s = .ok { s.set 0 (some ls.ds[0]!.w) with rip := 4198904 } := by
    rw [step_at_chunk 14 hrip' rfl (by decide : (4198897, (.ofInt (.ld 0 15 296)), 7) ∈ SporosVmBin.vmTable_14),
      exec_ld h15 h296, hrip']
  let s1 := { s.set 0 (some ls.ds[0]!.w) with rip := 4198904 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) ls.ds[0]!.w
  have hs1_0 : s1.reg 0 = some ls.ds[0]!.w := reg_set_self_ok' hreg (by decide) _
  have hs1_4 : s1.reg 4 = some vc.sp := by
    change (s.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hand_val : (X86Int.aluF .and ls.ds[0]!.w (255 % X86Env.W)).1 = ls.ds[0]!.w % 256 := by
    rw [aluF_and_val, Nat.mod_eq_of_lt (by decide : 255 < X86Env.W), land_255_eq_mod]
  have hand_pair : X86Int.aluF .and ls.ds[0]!.w (255 % X86Env.W) =
      (ls.ds[0]!.w % 256, (X86Int.aluF .and ls.ds[0]!.w (255 % X86Env.W)).2) := Prod.ext hand_val rfl
  have h1 : X86Vm.step vmImage s1 =
      .ok { s1.set 0 (some (ls.ds[0]!.w % 256)) with fl := (X86Int.aluF .and ls.ds[0]!.w (255 % X86Env.W)).2, rip := 4198911 } := by
    rw [step_at_chunk 14 rfl rfl (by decide : (4198904, (.ofInt (.aluI .and 0 255)), 7) ∈ SporosVmBin.vmTable_14),
      exec_aluI hs1_0, hand_pair]
  let s2 := { s1.set 0 (some (ls.ds[0]!.w % 256)) with fl := (X86Int.aluF .and ls.ds[0]!.w (255 % X86Env.W)).2, rip := 4198911 }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) (ls.ds[0]!.w % 256)
  have hs2_4 : s2.reg 4 = some vc.sp := by
    change (s1.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs1_4
  have hs2_15 : s2.reg 15 = some ctxBase := by
    change (s1.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15
  have hs2_0 : s2.reg 0 = some (ls.ds[0]!.w % 256) := reg_set_self_ok' hr1 (by decide) _
  have hwf2 : vc.wf pr ls s2 := vc.wf_step_misc h rfl hr2 hs2_4 hs2_15
  obtain ⟨s3, h2, hrip3, hk3, _, _, hwf3, h448, _⟩ :=
    step_wr_ctx_scratch (step_at_chunk 14 rfl rfl (by decide : (4198911, (.ofInt (.st 15 448 0)), 7) ∈ SporosVmBin.vmTable_14))
      hwf2 hs2_0 (by decide) (by decide)
  have h3 : X86Vm.step vmImage s3 = .ok { s3 with rip := 4200962 } := by
    rw [step_at_chunk 14 (by simpa [s2] using hrip3) rfl
      (by decide : (4198918, (.ofInt (.jmp 4200962)), 5) ∈ SporosVmBin.vmTable_14), exec_jmp]
  let s4 := { s3 with rip := 4200962 }
  have hwf4 := VmCtx.wf_step_misc (s' := s4) hwf3 rfl hwf3.2.2.2.1 hwf3.2.2.2.2.1 hwf3.2.2.2.2.2.1
  have hmod_lt : (ls.ds[0]!.w % 256) % X86Env.W = ls.ds[0]!.w % 256 :=
    Nat.mod_eq_of_lt (Nat.lt_trans (Nat.mod_lt _ (by decide)) (by decide))
  rw [hmod_lt] at h448
  have hhalt := VmCtx.halt_of_wf rfl hwf4 h448
  have hrch4 := Reaches.step_trans h0 (Reaches.step_trans h1 (Reaches.step_trans h2 (Reaches.step h3)))
  have hres := lift_halt_outcome hrch4 hk3 hhalt
  have hle0 := hle
  rcases hds_eq : ls.ds with _ | ⟨c, r⟩ <;> try (simp [hds_eq] at hle0)
  dsimp only [PrimPostOutcome]
  simp only [Sporos.Prim.eff, SporosProvLen.lexecPrim] at hres ⊢
  simpa [hds_eq, Nat.mod_mod] using hres

theorem P_clock_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_zero_mk1)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 0) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 0) (h288 : rd64 s.mem (ea ctxBase 288) = some 1) :
    PrimPostOutcome pr ls f rest s .clock :=
  zero_mk1_spec hrip rfl (Nat.zero_le _) h hor hrs hpcW h272 h280 h288
    (by simp [SporosProvLen.lexecPrim, Sporos.wrap, M_eq_W', Nat.zero_mod])

theorem P_core_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_zero_mk1)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 0) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 0) (h288 : rd64 s.mem (ea ctxBase 288) = some 1) :
    PrimPostOutcome pr ls f rest s .core :=
  zero_mk1_spec hrip rfl (Nat.zero_le _) h hor hrs hpcW h272 h280 h288
    (by simp [SporosProvLen.lexecPrim, Sporos.wrap, M_eq_W', Nat.zero_mod])

theorem P_sensor_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_zero_mk1) (hle : 1 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 1) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 1) (h288 : rd64 s.mem (ea ctxBase 288) = some 1) :
    PrimPostOutcome pr ls f rest s .sensor := by
  have hle0 := hle
  rcases hds_eq : ls.ds with _ | ⟨a, r⟩ <;> try (simp [hds_eq] at hle0)
  exact zero_mk1_spec hrip rfl hle h hor hrs hpcW h272 h280 h288
    (by simp [hds_eq, SporosProvLen.lexecPrim, Sporos.wrap, M_eq_W', Nat.zero_mod])

theorem P_provget_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_provget) (hle : 1 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 1) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 1) (h288 : rd64 s.mem (ea ctxBase 288) = some 2)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) (h304 : rd64 s.mem (ea ctxBase 304) = some ls.ds[0]!.plen) :
    PrimPostOutcome pr ls f rest s .provget := by
  have hrip' : s.rip = 4198930 := hrip
  have hle0 := hle
  rcases hds_eq : ls.ds with _ | ⟨a, r⟩ <;> try (simp [hds_eq] at hle0)
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hreg : RegsOk s := h.2.2.2.1
  have haLt := wf_ds_lt_W h (by simp [hds_eq] : a ∈ ls.ds)
  have haW : a.w < X86Env.W := haLt.1
  have hapW : a.plen < X86Env.W := haLt.2
  have h296a : rd64 s.mem (ea ctxBase 296) = some a.w := by simpa [hds_eq] using h296
  have h304a : rd64 s.mem (ea ctxBase 304) = some a.plen := by simpa [hds_eq] using h304
  let q := SporosProvLen.lq [a] ls.pc
  have h272a : rd64 s.mem (ea ctxBase 272) = some q := by simpa [hds_eq, q] using h272
  have h0 : X86Vm.step vmImage s = .ok { s.set 0 (some a.plen) with rip := 4198937 } := by
    rw [step_at_chunk 14 hrip' rfl (by decide : (4198930, (.ofInt (.ld 0 15 304)), 7) ∈ SporosVmBin.vmTable_14),
      exec_ld h15 h304a, hrip']
  let s1 := { s.set 0 (some a.plen) with rip := 4198937 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) a.plen
  have hs1_0 : s1.reg 0 = some a.plen := reg_set_self_ok' hreg (by decide) _
  have hs1_4 : s1.reg 4 = some vc.sp := by
    change (s.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hwf1 : vc.wf pr ls s1 := vc.wf_step_misc h rfl hr1 hs1_4 hs1_15
  obtain ⟨s2, h1, hrip2, hk2, _, _, hwf2, h360_2, hoth2⟩ :=
    step_wr_ctx_scratch (step_at_chunk 14 rfl rfl (by decide : (4198937, (.ofInt (.st 15 360 0)), 7) ∈ SporosVmBin.vmTable_14))
      hwf1 hs1_0 (by decide) (by decide)
  rw [Nat.mod_eq_of_lt hapW] at h360_2
  have hs2_rip : s2.rip = 4198944 := by simpa [s1] using hrip2
  have h2 : X86Vm.step vmImage s2 = .ok { s2.set 2 (some q) with rip := 4198951 } := by
    rw [step_at_chunk 14 hs2_rip rfl (by decide : (4198944, (.ofInt (.ld 2 15 272)), 7) ∈ SporosVmBin.vmTable_14),
      exec_ld hwf2.2.2.2.2.2.1 (hoth2 272 q (Or.inl (by decide)) (by decide) h272a), hs2_rip]
  let s3 := { s2.set 2 (some q) with rip := 4198951 }
  have hr3 : RegsOk s3 := regsOk_set hwf2.2.2.2.1 (by decide) q
  have hs3_2 : s3.reg 2 = some q := reg_set_self_ok' hwf2.2.2.2.1 (by decide) _
  have hs3_4 : s3.reg 4 = some vc.sp := by
    change (s2.set 2 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hwf2.2.2.2.2.1
  have hs3_15 : s3.reg 15 = some ctxBase := by
    change (s2.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hwf2.2.2.2.2.2.1
  have hwf3 := VmCtx.wf_step_misc (s' := s3) hwf2 rfl hr3 hs3_4 hs3_15
  obtain ⟨s4, h3, hrip4, hk4, _, _, hwf4, h368_4, hoth4⟩ :=
    step_wr_ctx_scratch (step_at_chunk 14 rfl rfl (by decide : (4198951, (.ofInt (.st 15 368 2)), 7) ∈ SporosVmBin.vmTable_14))
      hwf3 hs3_2 (by decide) (by decide)
  have hqW : q < X86Env.W := by dsimp [q, SporosProvLen.lq]; rw [← M_eq_W']; exact lt_W_of_mod _
  rw [Nat.mod_eq_of_lt hqW] at h368_4
  have hs4_rip : s4.rip = 4198958 := by simpa [s3] using hrip4
  have h4 : X86Vm.step vmImage s4 = .ok { s4.set 0 (some a.w) with rip := 4198965 } := by
    rw [step_at_chunk 14 hs4_rip rfl (by decide : (4198958, (.ofInt (.ld 0 15 296)), 7) ∈ SporosVmBin.vmTable_14),
      exec_ld hwf4.2.2.2.2.2.1 (hoth4 296 a.w (Or.inl (by decide)) (by decide) (hoth2 296 a.w (Or.inl (by decide)) (by decide) h296a)),
      hs4_rip]
  let s5 := { s4.set 0 (some a.w) with rip := 4198965 }
  have hr5 : RegsOk s5 := regsOk_set hwf4.2.2.2.1 (by decide) a.w
  have hs5_0 : s5.reg 0 = some a.w := reg_set_self_ok' hwf4.2.2.2.1 (by decide) _
  have hs5_4 : s5.reg 4 = some vc.sp := by
    change (s4.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hwf4.2.2.2.2.1
  have hs5_15 : s5.reg 15 = some ctxBase := by
    change (s4.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hwf4.2.2.2.2.2.1
  have hwf5 := VmCtx.wf_step_misc (s' := s5) hwf4 rfl hr5 hs5_4 hs5_15
  obtain ⟨s6, h5, hrip6, hk6, _, _, hwf6, h376_6, hoth6⟩ :=
    step_wr_ctx_scratch (step_at_chunk 14 rfl rfl (by decide : (4198965, (.ofInt (.st 15 376 0)), 7) ∈ SporosVmBin.vmTable_14))
      hwf5 hs5_0 (by decide) (by decide)
  rw [Nat.mod_eq_of_lt haW] at h376_6
  have hs6_rip : s6.rip = 4198972 := by simpa [s5] using hrip6
  have h304_6 : rd64 s6.mem (ea ctxBase 304) = some a.plen :=
    hoth6 304 a.plen (Or.inl (by decide)) (by decide)
      (hoth4 304 a.plen (Or.inl (by decide)) (by decide) (hoth2 304 a.plen (Or.inl (by decide)) (by decide) h304a))
  obtain ⟨s7, hrch_tag, hrip7, hk7, hwf7, h384_7, hoth7⟩ :=
    step_wr_ctx_tag_slot (nextAddr := 4199000)
      (fun t ht => step_at_chunk 14 (by rw [ht, hs6_rip]) rfl (by decide : (4198972, (.ofInt (.ld 0 15 304)), 7) ∈ SporosVmBin.vmTable_14))
      (fun t ht => step_at_chunk 14 (by rw [ht, hs6_rip]) rfl (by decide : (4198979, (.ofInt (.ld 1 15 224)), 7) ∈ SporosVmBin.vmTable_14))
      (fun t ht => step_at_chunk 14 (by rw [ht, hs6_rip]) rfl (by decide : (4198986, (.ofInt (.alu .add 0 1)), 3) ∈ SporosVmBin.vmTable_14))
      (fun t ht => step_at_chunk 14 (by rw [ht, hs6_rip]) rfl (by decide : (4198989, (.ofInt (.aluI .add 0 1)), 4) ∈ SporosVmBin.vmTable_14))
      (fun t ht => step_at_chunk 14 (by rw [ht, hs6_rip]) rfl (by decide : (4198993, (.ofInt (.st 15 384 0)), 7) ∈ SporosVmBin.vmTable_14))
      (by omega) hwf6 h304_6 (by decide) (by decide)
  have hrch7 : Reaches vmImage s s7 :=
    Reaches.trans (Reaches.step_trans h0 (Reaches.step_trans h1 (Reaches.step_trans h2
      (Reaches.step_trans h3 (Reaches.step_trans h4 (Reaches.step h5)))))) hrch_tag
  have hk7_s : s7.k = s.k := hk7.trans (hk6.trans (hk4.trans hk2))
  have h280_7 : rd64 s7.mem (ea ctxBase 280) = some 1 :=
    hoth7 280 1 (Or.inl (by decide)) (by decide)
      (hoth6 280 1 (Or.inl (by decide)) (by decide)
        (hoth4 280 1 (Or.inl (by decide)) (by decide) (hoth2 280 1 (Or.inl (by decide)) (by decide) h280)))
  have h288_7 : rd64 s7.mem (ea ctxBase 288) = some 2 :=
    hoth7 288 2 (Or.inl (by decide)) (by decide)
      (hoth6 288 2 (Or.inl (by decide)) (by decide)
        (hoth4 288 2 (Or.inl (by decide)) (by decide) (hoth2 288 2 (Or.inl (by decide)) (by decide) h288)))
  have h360_7 : rd64 s7.mem (ea ctxBase 360) = some a.plen :=
    hoth7 360 a.plen (Or.inl (by decide)) (by decide)
      (hoth6 360 a.plen (Or.inl (by decide)) (by decide)
        (hoth4 360 a.plen (Or.inl (by decide)) (by decide) h360_2))
  have h368_7 : rd64 s7.mem (ea ctxBase 368) = some q :=
    hoth7 368 q (Or.inl (by decide)) (by decide)
      (hoth6 368 q (Or.inl (by decide)) (by decide) h368_4)
  have h376_7 : rd64 s7.mem (ea ctxBase 376) = some a.w :=
    hoth7 376 a.w (Or.inl (by decide)) (by decide) h376_6
  let outs : List SporosProvLen.LVal := [⟨a.plen, q⟩, SporosProvLen.ltag ls.pc a]
  have hqtW : (a.plen + ls.pc + 1) % Sporos.M < X86Env.W := by rw [← M_eq_W']; exact lt_W_of_mod _
  have hslots : ∀ i (_ : i < outs.length),
      rd64 s7.mem (ea ctxBase (360 + i * 16)) = some outs[i].w ∧
      rd64 s7.mem (ea ctxBase (368 + i * 16)) = some outs[i].plen := by
    intro i hi
    rcases i with _ | _ | i
    · exact ⟨h360_7, h368_7⟩
    · exact ⟨h376_7, h384_7⟩
    · simp only [outs, List.length_cons, List.length_nil] at hi; omega
  have houtsW : ∀ v ∈ outs, v.w < X86Env.W ∧ v.plen < X86Env.W := by
    intro v hv
    simp only [outs, List.mem_cons, List.not_mem_nil, or_false] at hv
    rcases hv with rfl | rfl
    · exact ⟨hapW, hqW⟩
    · exact ⟨haW, hqtW⟩
  have hfin := jmp_epilogue0_finish_spec
    (step_at_chunk 14 hrip7 rfl (by decide : (4199000, (.ofInt (.jmp 4195233)), 5) ∈ SporosVmBin.vmTable_14))
    hwf7 (oracleAdmits_of_reaches hor hrch7) hrs hpcW hle (by simp [outs]) houtsW h280_7 h288_7 hslots
  have hres := lift_next_outcome hrch7 hk7_s hfin
  dsimp only [PrimPostOutcome]
  simp only [Sporos.Prim.eff, hds_eq, List.take_succ_cons, List.take_zero, SporosProvLen.lexecPrim]
  simpa only [hds_eq, outs, Sporos.wrap, M_eq_W', Nat.mod_eq_of_lt hapW, q] using hres

theorem P_confget_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_confget) (hle : 1 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 1) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 1) (h288 : rd64 s.mem (ea ctxBase 288) = some 2)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) (h304 : rd64 s.mem (ea ctxBase 304) = some ls.ds[0]!.plen) :
    PrimPostOutcome pr ls f rest s .confget := by
  have hrip' : s.rip = 4199005 := hrip
  have hle0 := hle
  rcases hds_eq : ls.ds with _ | ⟨a, r⟩ <;> try (simp [hds_eq] at hle0)
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hreg : RegsOk s := h.2.2.2.1
  have haLt := wf_ds_lt_W h (by simp [hds_eq] : a ∈ ls.ds)
  have haW : a.w < X86Env.W := haLt.1
  have h296a : rd64 s.mem (ea ctxBase 296) = some a.w := by simpa [hds_eq] using h296
  have h304a : rd64 s.mem (ea ctxBase 304) = some a.plen := by simpa [hds_eq] using h304
  let q := SporosProvLen.lq [a] ls.pc
  have h272a : rd64 s.mem (ea ctxBase 272) = some q := by simpa [hds_eq, q] using h272
  obtain ⟨r0, hr0⟩ := reg_of_ok hreg 0 (by decide) (by decide)
  have hxor0 : (X86Int.aluF .xor r0 r0).1 = 0 := by rw [aluF_xor_val]; exact Nat.xor_self r0
  have hxor_pair : X86Int.aluF .xor r0 r0 = (0, (X86Int.aluF .xor r0 r0).2) := Prod.ext hxor0 rfl
  have h0 : X86Vm.step vmImage s =
      .ok { s.set 0 (some 0) with fl := (X86Int.aluF .xor r0 r0).2, rip := 4199007 } := by
    rw [step_at_chunk 14 hrip' rfl (by decide : (4199005, (.ofInt (.alu .xor 0 0)), 2) ∈ SporosVmBin.vmTable_14),
      exec_alu hr0 hr0, hxor_pair, hrip']
  let s1 := { s.set 0 (some 0) with fl := (X86Int.aluF .xor r0 r0).2, rip := 4199007 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) 0
  have hs1_0 : s1.reg 0 = some 0 := reg_set_self_ok' hreg (by decide) _
  have hs1_4 : s1.reg 4 = some vc.sp := by
    change (s.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hwf1 : vc.wf pr ls s1 := vc.wf_step_misc h rfl hr1 hs1_4 hs1_15
  obtain ⟨s2, h1, hrip2, hk2, _, _, hwf2, h360_2, hoth2⟩ :=
    step_wr_ctx_scratch (step_at_chunk 14 rfl rfl (by decide : (4199007, (.ofInt (.st 15 360 0)), 7) ∈ SporosVmBin.vmTable_14))
      hwf1 hs1_0 (by decide) (by decide)
  rw [Nat.mod_eq_of_lt (by decide : 0 < X86Env.W)] at h360_2
  have hs2_rip : s2.rip = 4199014 := by simpa [s1] using hrip2
  have h2 : X86Vm.step vmImage s2 = .ok { s2.set 2 (some q) with rip := 4199021 } := by
    rw [step_at_chunk 14 hs2_rip rfl (by decide : (4199014, (.ofInt (.ld 2 15 272)), 7) ∈ SporosVmBin.vmTable_14),
      exec_ld hwf2.2.2.2.2.2.1 (hoth2 272 q (Or.inl (by decide)) (by decide) h272a), hs2_rip]
  let s3 := { s2.set 2 (some q) with rip := 4199021 }
  have hr3 : RegsOk s3 := regsOk_set hwf2.2.2.2.1 (by decide) q
  have hs3_2 : s3.reg 2 = some q := reg_set_self_ok' hwf2.2.2.2.1 (by decide) _
  have hs3_4 : s3.reg 4 = some vc.sp := by
    change (s2.set 2 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hwf2.2.2.2.2.1
  have hs3_15 : s3.reg 15 = some ctxBase := by
    change (s2.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hwf2.2.2.2.2.2.1
  have hwf3 := VmCtx.wf_step_misc (s' := s3) hwf2 rfl hr3 hs3_4 hs3_15
  obtain ⟨s4, h3, hrip4, hk4, _, _, hwf4, h368_4, hoth4⟩ :=
    step_wr_ctx_scratch (step_at_chunk 14 rfl rfl (by decide : (4199021, (.ofInt (.st 15 368 2)), 7) ∈ SporosVmBin.vmTable_14))
      hwf3 hs3_2 (by decide) (by decide)
  have hqW : q < X86Env.W := by dsimp [q, SporosProvLen.lq]; rw [← M_eq_W']; exact lt_W_of_mod _
  rw [Nat.mod_eq_of_lt hqW] at h368_4
  have hs4_rip : s4.rip = 4199028 := by simpa [s3] using hrip4
  have h4 : X86Vm.step vmImage s4 = .ok { s4.set 0 (some a.w) with rip := 4199035 } := by
    rw [step_at_chunk 14 hs4_rip rfl (by decide : (4199028, (.ofInt (.ld 0 15 296)), 7) ∈ SporosVmBin.vmTable_14),
      exec_ld hwf4.2.2.2.2.2.1 (hoth4 296 a.w (Or.inl (by decide)) (by decide) (hoth2 296 a.w (Or.inl (by decide)) (by decide) h296a)),
      hs4_rip]
  let s5 := { s4.set 0 (some a.w) with rip := 4199035 }
  have hr5 : RegsOk s5 := regsOk_set hwf4.2.2.2.1 (by decide) a.w
  have hs5_0 : s5.reg 0 = some a.w := reg_set_self_ok' hwf4.2.2.2.1 (by decide) _
  have hs5_4 : s5.reg 4 = some vc.sp := by
    change (s4.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hwf4.2.2.2.2.1
  have hs5_15 : s5.reg 15 = some ctxBase := by
    change (s4.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hwf4.2.2.2.2.2.1
  have hwf5 := VmCtx.wf_step_misc (s' := s5) hwf4 rfl hr5 hs5_4 hs5_15
  obtain ⟨s6, h5, hrip6, hk6, _, _, hwf6, h376_6, hoth6⟩ :=
    step_wr_ctx_scratch (step_at_chunk 14 rfl rfl (by decide : (4199035, (.ofInt (.st 15 376 0)), 7) ∈ SporosVmBin.vmTable_14))
      hwf5 hs5_0 (by decide) (by decide)
  rw [Nat.mod_eq_of_lt haW] at h376_6
  have hs6_rip : s6.rip = 4199042 := by simpa [s5] using hrip6
  have h304_6 : rd64 s6.mem (ea ctxBase 304) = some a.plen :=
    hoth6 304 a.plen (Or.inl (by decide)) (by decide)
      (hoth4 304 a.plen (Or.inl (by decide)) (by decide) (hoth2 304 a.plen (Or.inl (by decide)) (by decide) h304a))
  obtain ⟨s7, hrch_tag, hrip7, hk7, hwf7, h384_7, hoth7⟩ :=
    step_wr_ctx_tag_slot (nextAddr := 4199070)
      (fun t ht => step_at_chunk 15 (by rw [ht, hs6_rip]) rfl (by decide : (4199042, (.ofInt (.ld 0 15 304)), 7) ∈ SporosVmBin.vmTable_15))
      (fun t ht => step_at_chunk 15 (by rw [ht, hs6_rip]) rfl (by decide : (4199049, (.ofInt (.ld 1 15 224)), 7) ∈ SporosVmBin.vmTable_15))
      (fun t ht => step_at_chunk 15 (by rw [ht, hs6_rip]) rfl (by decide : (4199056, (.ofInt (.alu .add 0 1)), 3) ∈ SporosVmBin.vmTable_15))
      (fun t ht => step_at_chunk 15 (by rw [ht, hs6_rip]) rfl (by decide : (4199059, (.ofInt (.aluI .add 0 1)), 4) ∈ SporosVmBin.vmTable_15))
      (fun t ht => step_at_chunk 15 (by rw [ht, hs6_rip]) rfl (by decide : (4199063, (.ofInt (.st 15 384 0)), 7) ∈ SporosVmBin.vmTable_15))
      (by omega) hwf6 h304_6 (by decide) (by decide)
  have hrch7 : Reaches vmImage s s7 :=
    Reaches.trans (Reaches.step_trans h0 (Reaches.step_trans h1 (Reaches.step_trans h2
      (Reaches.step_trans h3 (Reaches.step_trans h4 (Reaches.step h5)))))) hrch_tag
  have hk7_s : s7.k = s.k := hk7.trans (hk6.trans (hk4.trans hk2))
  have h280_7 : rd64 s7.mem (ea ctxBase 280) = some 1 :=
    hoth7 280 1 (Or.inl (by decide)) (by decide)
      (hoth6 280 1 (Or.inl (by decide)) (by decide)
        (hoth4 280 1 (Or.inl (by decide)) (by decide) (hoth2 280 1 (Or.inl (by decide)) (by decide) h280)))
  have h288_7 : rd64 s7.mem (ea ctxBase 288) = some 2 :=
    hoth7 288 2 (Or.inl (by decide)) (by decide)
      (hoth6 288 2 (Or.inl (by decide)) (by decide)
        (hoth4 288 2 (Or.inl (by decide)) (by decide) (hoth2 288 2 (Or.inl (by decide)) (by decide) h288)))
  have h360_7 : rd64 s7.mem (ea ctxBase 360) = some 0 :=
    hoth7 360 0 (Or.inl (by decide)) (by decide)
      (hoth6 360 0 (Or.inl (by decide)) (by decide)
        (hoth4 360 0 (Or.inl (by decide)) (by decide) h360_2))
  have h368_7 : rd64 s7.mem (ea ctxBase 368) = some q :=
    hoth7 368 q (Or.inl (by decide)) (by decide)
      (hoth6 368 q (Or.inl (by decide)) (by decide) h368_4)
  have h376_7 : rd64 s7.mem (ea ctxBase 376) = some a.w :=
    hoth7 376 a.w (Or.inl (by decide)) (by decide) h376_6
  let outs : List SporosProvLen.LVal := [⟨0, q⟩, SporosProvLen.ltag ls.pc a]
  have hqtW : (a.plen + ls.pc + 1) % Sporos.M < X86Env.W := by rw [← M_eq_W']; exact lt_W_of_mod _
  have hslots : ∀ i (_ : i < outs.length),
      rd64 s7.mem (ea ctxBase (360 + i * 16)) = some outs[i].w ∧
      rd64 s7.mem (ea ctxBase (368 + i * 16)) = some outs[i].plen := by
    intro i hi
    rcases i with _ | _ | i
    · exact ⟨h360_7, h368_7⟩
    · exact ⟨h376_7, h384_7⟩
    · simp only [outs, List.length_cons, List.length_nil] at hi; omega
  have houtsW : ∀ v ∈ outs, v.w < X86Env.W ∧ v.plen < X86Env.W := by
    intro v hv
    simp only [outs, List.mem_cons, List.not_mem_nil, or_false] at hv
    rcases hv with rfl | rfl
    · exact ⟨(by decide : (0 : Nat) < X86Env.W), hqW⟩
    · exact ⟨haW, hqtW⟩
  have hfin := jmp_epilogue0_finish_spec
    (step_at_chunk 15 hrip7 rfl (by decide : (4199070, (.ofInt (.jmp 4195233)), 5) ∈ SporosVmBin.vmTable_15))
    hwf7 (oracleAdmits_of_reaches hor hrch7) hrs hpcW hle (by simp [outs]) houtsW h280_7 h288_7 hslots
  have hres := lift_next_outcome hrch7 hk7_s hfin
  dsimp only [PrimPostOutcome]
  simp only [Sporos.Prim.eff, hds_eq, List.take_succ_cons, List.take_zero, SporosProvLen.lexecPrim]
  simpa only [hds_eq, outs, Sporos.wrap, M_eq_W', Nat.zero_mod, q] using hres

theorem P_demand_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_demand) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w)
    (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w)
    (h320 : rd64 s.mem (ea ctxBase 320) = some ls.ds[1]!.plen) :
    PrimPostOutcome pr ls f rest s .demand := by
  have hrip' : s.rip = 4199075 := hrip
  have hle0 := hle
  rcases hds_eq : ls.ds with _ | ⟨th, _ | ⟨a, r⟩⟩ <;> try (simp [hds_eq] at hle0)
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hreg : RegsOk s := h.2.2.2.1
  have haLt := wf_ds_lt_W h (by simp [hds_eq] : a ∈ ls.ds)
  have haW : a.w < X86Env.W := haLt.1
  have h296th : rd64 s.mem (ea ctxBase 296) = some th.w := by simpa [hds_eq] using h296
  have h312a : rd64 s.mem (ea ctxBase 312) = some a.w := by simpa [hds_eq] using h312
  have h320a : rd64 s.mem (ea ctxBase 320) = some a.plen := by simpa [hds_eq] using h320
  have h0 : X86Vm.step vmImage s = .ok { s.set 0 (some th.w) with rip := 4199082 } := by
    rw [step_at_chunk 15 hrip' rfl (by decide : (4199075, (.ofInt (.ld 0 15 296)), 7) ∈ SporosVmBin.vmTable_15),
      exec_ld h15 h296th, hrip']
  let s1 := { s.set 0 (some th.w) with rip := 4199082 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) th.w
  have hs1_0 : s1.reg 0 = some th.w := reg_set_self_ok' hreg (by decide) _
  have hs1_4 : s1.reg 4 = some vc.sp := by
    change (s.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have h1 : X86Vm.step vmImage s1 =
      .ok { s1 with fl := (X86Int.aluF .and th.w th.w).2, rip := 4199085 } := by
    rw [step_at_chunk 15 rfl rfl (by decide : (4199082, (.ofInt (.test 0 0)), 3) ∈ SporosVmBin.vmTable_15),
      exec_test hs1_0 hs1_0]
  let s2 := { s1 with fl := (X86Int.aluF .and th.w th.w).2, rip := 4199085 }
  by_cases hth : 0 < th.w
  · have h2 : X86Vm.step vmImage s2 = .ok { s2 with rip := 4199087 } := by
      rw [step_at_chunk 15 rfl rfl (by decide : (4199085, (.ofInt (.jcc 4 4199092)), 2) ∈ SporosVmBin.vmTable_15),
        exec_jcc (cond_aluF_and_self_e th.w)]
      dsimp only [s2]; simp [Nat.ne_of_gt hth]
    let s3 := { s2 with rip := 4199087 }
    have h3 : X86Vm.step vmImage s3 = .ok { s3 with rip := 4201201 } := by
      rw [step_at_chunk 15 rfl rfl (by decide : (4199087, (.ofInt (.jmp 4201201)), 5) ∈ SporosVmBin.vmTable_15),
        exec_jmp]
    let s4 := { s3 with rip := 4201201 }
    have hwf4 : vc.wf pr ls s4 := vc.wf_step_misc h rfl hr1 hs1_4 hs1_15
    have hfail := L_fail_from_vc_spec (e := .demand) (s := s4) rfl hwf4
    have hrch4 := Reaches.step_trans h0 (Reaches.step_trans h1 (Reaches.step_trans h2 (Reaches.step h3)))
    dsimp only [PrimPostOutcome]
    simp only [Sporos.Prim.eff, List.take_succ_cons, List.take_zero, SporosProvLen.lexecPrim, hds_eq, hth, ↓reduceIte]
    exact lift_fail_outcome hrch4 rfl hfail
  · have hth0 : th.w = 0 := by omega
    have h2 : X86Vm.step vmImage s2 = .ok { s2 with rip := 4199092 } := by
      rw [step_at_chunk 15 rfl rfl (by decide : (4199085, (.ofInt (.jcc 4 4199092)), 2) ∈ SporosVmBin.vmTable_15),
        exec_jcc (cond_aluF_and_self_e th.w)]
      dsimp only [s2]; simp [hth0]
    let s3 := { s2 with rip := 4199092 }
    have hs3_15 : s3.reg 15 = some ctxBase := hs1_15
    have h3 : X86Vm.step vmImage s3 = .ok { s3.set 0 (some a.w) with rip := 4199099 } := by
      rw [step_at_chunk 15 rfl rfl (by decide : (4199092, (.ofInt (.ld 0 15 312)), 7) ∈ SporosVmBin.vmTable_15),
        exec_ld hs3_15 h312a]
    let s4 := { s3.set 0 (some a.w) with rip := 4199099 }
    have hr4 : RegsOk s4 := regsOk_set hr1 (by decide) a.w
    have hs4_0 : s4.reg 0 = some a.w := reg_set_self_ok' hr1 (by decide) _
    have hs4_4 : s4.reg 4 = some vc.sp := by
      change (s3.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs1_4
    have hs4_15 : s4.reg 15 = some ctxBase := by
      change (s3.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15
    have hwf4 : vc.wf pr ls s4 := vc.wf_step_misc h rfl hr4 hs4_4 hs4_15
    obtain ⟨s5, h4s, hrip5, hk5, _, _, hwf5, h360_5, hoth5⟩ :=
      step_wr_ctx_scratch (step_at_chunk 15 rfl rfl (by decide : (4199099, (.ofInt (.st 15 360 0)), 7) ∈ SporosVmBin.vmTable_15))
        hwf4 hs4_0 (by decide) (by decide)
    rw [Nat.mod_eq_of_lt haW] at h360_5
    have hs5_rip : s5.rip = 4199106 := by simpa [s4] using hrip5
    have h320_5 : rd64 s5.mem (ea ctxBase 320) = some a.plen :=
      hoth5 320 a.plen (Or.inl (by decide)) (by decide) h320a
    obtain ⟨s6, hrch_tag, hrip6, hk6, hwf6, h368_6, hoth6⟩ :=
      step_wr_ctx_tag_slot (nextAddr := 4199134)
        (fun t ht => step_at_chunk 15 (by rw [ht, hs5_rip]) rfl (by decide : (4199106, (.ofInt (.ld 0 15 320)), 7) ∈ SporosVmBin.vmTable_15))
        (fun t ht => step_at_chunk 15 (by rw [ht, hs5_rip]) rfl (by decide : (4199113, (.ofInt (.ld 1 15 224)), 7) ∈ SporosVmBin.vmTable_15))
        (fun t ht => step_at_chunk 15 (by rw [ht, hs5_rip]) rfl (by decide : (4199120, (.ofInt (.alu .add 0 1)), 3) ∈ SporosVmBin.vmTable_15))
        (fun t ht => step_at_chunk 15 (by rw [ht, hs5_rip]) rfl (by decide : (4199123, (.ofInt (.aluI .add 0 1)), 4) ∈ SporosVmBin.vmTable_15))
        (fun t ht => step_at_chunk 15 (by rw [ht, hs5_rip]) rfl (by decide : (4199127, (.ofInt (.st 15 368 0)), 7) ∈ SporosVmBin.vmTable_15))
        (by omega) hwf5 h320_5 (by decide) (by decide)
    have hrch6 : Reaches vmImage s s6 :=
      Reaches.trans (Reaches.step_trans h0 (Reaches.step_trans h1 (Reaches.step_trans h2 (Reaches.step_trans h3 (Reaches.step h4s))))) hrch_tag
    have hk6_s : s6.k = s.k := hk6.trans hk5
    have h280_6 : rd64 s6.mem (ea ctxBase 280) = some 2 :=
      hoth6 280 2 (Or.inl (by decide)) (by decide) (hoth5 280 2 (Or.inl (by decide)) (by decide) h280)
    have h288_6 : rd64 s6.mem (ea ctxBase 288) = some 1 :=
      hoth6 288 1 (Or.inl (by decide)) (by decide) (hoth5 288 1 (Or.inl (by decide)) (by decide) h288)
    have h360_6 : rd64 s6.mem (ea ctxBase 360) = some a.w :=
      hoth6 360 a.w (Or.inl (by decide)) (by decide) h360_5
    let outs : List SporosProvLen.LVal := [SporosProvLen.ltag ls.pc a]
    have hqtW : (a.plen + ls.pc + 1) % Sporos.M < X86Env.W := by rw [← M_eq_W']; exact lt_W_of_mod _
    have hslots : ∀ i (_ : i < outs.length),
        rd64 s6.mem (ea ctxBase (360 + i * 16)) = some outs[i].w ∧
        rd64 s6.mem (ea ctxBase (368 + i * 16)) = some outs[i].plen := by
      intro i hi
      rcases i with _ | i
      · exact ⟨h360_6, h368_6⟩
      · simp [outs] at hi
    have houtsW : ∀ v ∈ outs, v.w < X86Env.W ∧ v.plen < X86Env.W := by
      intro v hv
      simp only [outs, List.mem_singleton] at hv; subst hv; exact ⟨haW, hqtW⟩
    have hfin := jmp_epilogue0_finish_spec
      (step_at_chunk 15 hrip6 rfl (by decide : (4199134, (.ofInt (.jmp 4195233)), 5) ∈ SporosVmBin.vmTable_15))
      hwf6 (oracleAdmits_of_reaches hor hrch6) hrs hpcW hle (by simp [outs]) houtsW h280_6 h288_6 hslots
    have hres := lift_next_outcome hrch6 hk6_s hfin
    dsimp only [PrimPostOutcome]
    simp only [Sporos.Prim.eff, hds_eq, List.take_succ_cons, List.take_zero, SporosProvLen.lexecPrim]
    simpa only [hds_eq, hth, Nat.not_lt_zero, ↓reduceIte, outs] using hres

end SporosVmPrimBase