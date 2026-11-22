import Bootstrap.SporosX1FinalRefuse

/-!
# Sporos stage 1: bounded refusal for outer word checks

Concrete decode facts (`ChkRefuseCode`) for the 26 fail-jumps in `annval`,
`codewalk_entry`, `fall_tail`, and opcode handlers of `sporos-stage1.hex`,
plus bounded `Out.exit 1` refusal lemmas for empty words, `annval` failures,
and `codewalk_entry` annotation-0 mismatches.
-/

open X86Env X86Int X86Dec Sporos

namespace SporosX1

/-- Decode facts for the 26 `jmp aFail1` failure branches inside the word
checker (`annval`, `codewalk_entry`, `fall_tail`, and opcode handlers). -/
structure ChkRefuseCode (im : Nat → Option (Ins × Nat)) : Prop where
  avDupD : im (aAnnval + 89) = some (.jmp aFail1, 5)
  avDupA : im (aAnnval + 111) = some (.jmp aFail1, 5)
  ckAnn0D : im (aCodewalk + 43) = some (.jmp aFail1, 5)
  ckAnn0A : im (aCodewalk + 63) = some (.jmp aFail1, 5)
  fallEnd : im (aFallTail + 13) = some (.jmp aFail1, 5)
  fallAnnD : im (aFallTail + 54) = some (.jmp aFail1, 5)
  fallAnnA : im (aFallTail + 74) = some (.jmp aFail1, 5)
  primUnd : im (aPrim + 25) = some (.jmp aFail1, 5)
  pickOob : im (aPick + 17) = some (.jmp aFail1, 5)
  torUnd : im (aTor + 6) = some (.jmp aFail1, 5)
  fromrUnd : im (aFromr + 6) = some (.jmp aFail1, 5)
  rfetchUnd : im (aRfetch + 6) = some (.jmp aFail1, 5)
  locgetOob : im (aLocget + 21) = some (.jmp aFail1, 5)
  locsetUnd : im (aLocset + 18) = some (.jmp aFail1, 5)
  locsetOob : im (aLocset + 32) = some (.jmp aFail1, 5)
  jmpUnann : im (aJmp + 25) = some (.jmp aFail1, 5)
  jmpAnnD : im (aJmp + 48) = some (.jmp aFail1, 5)
  jmpAnnA : im (aJmp + 68) = some (.jmp aFail1, 5)
  jzUnd : im (aJz + 18) = some (.jmp aFail1, 5)
  jzUnann : im (aJz + 36) = some (.jmp aFail1, 5)
  jzAnnD : im (aJz + 66) = some (.jmp aFail1, 5)
  jzAnnA : im (aJz + 86) = some (.jmp aFail1, 5)
  callOob : im (aCall + 17) = some (.jmp aFail1, 5)
  callUnd : im (aCall + 54) = some (.jmp aFail1, 5)
  retD : im (aRet + 9) = some (.jmp aFail1, 5)
  retA : im (aRet + 19) = some (.jmp aFail1, 5)

set_option maxHeartbeats 0 in
/-- All 26 refusal jumps decode as expected in `stage1Im`. -/
theorem stage1_chkRefuseCode : ChkRefuseCode stage1Im := by
  constructor <;> decide +kernel

theorem aluF_sub_zf_eq {a b : Nat} (ha : a < W) (hb : b < W) (heq : a = b) :
    (aluF .sub a b).2.zf = some true := by
  subst heq
  rw [aluF_zf, aluF_sub_fst, subW_eq_zero ha ha]
  simp

theorem aluF_sub_zf_ne {a b : Nat} (ha : a < W) (hb : b < W) (hne : a ≠ b) :
    (aluF .sub a b).2.zf = some false := by
  rw [aluF_zf, aluF_sub_fst, subW_eq_zero ha hb]
  simp [hne]

theorem lookup_some_of_mem :
    ∀ (l : List (Nat × Height)) {t : Nat} {h : Height},
      (t, h) ∈ l → ∃ h', l.lookup t = some h'
  | [], _, _, hmem => by cases hmem
  | (k, v) :: rest, t, h, hmem => by
    simp only [List.lookup]
    by_cases he : t = k
    · have hb : (t == k) = true := by simp [he]
      rw [hb]
      exact ⟨v, rfl⟩
    · have hb : (t == k) = false := by simp [he]
      rw [hb]
      cases hmem with
      | head _ => exact absurd rfl he
      | tail _ ht => exact lookup_some_of_mem rest ht

/-- Any state at `aFail1` with registers defined and one stack slot exits `1` in
at most 5 steps. -/
theorem exit1_of_fail1B {im : Nat → Option (Ins × Nat)}
    (rc : RefuseCode im) {sp : Nat} (hsp32 : 32 ≤ sp) (hspW : sp < W) :
    ∀ s, s.rip = aFail1 → s.reg 4 = some sp → RegsOk s → StackRoom s 1 →
      ∃ m, m ≤ 5 ∧ run im m (Out.ok s) = Out.exit 1 := by
  intro s hrip hr4 hregs hroom
  obtain ⟨v6, hv6⟩ := hregs 6 (by decide) (by decide)
  obtain ⟨v2, hv2⟩ := hregs 2 (by decide) (by decide)
  obtain ⟨v10, hv10⟩ := hregs 10 (by decide) (by decide)
  obtain ⟨v8, hv8⟩ := hregs 8 (by decide) (by decide)
  obtain ⟨v9, hv9⟩ := hregs 9 (by decide) (by decide)
  obtain ⟨m1, hm1s⟩ := Option.isSome_iff_exists.mp
    (hroom sp hr4 0 (by decide) 60)
  obtain ⟨m2, hm2s⟩ := Option.isSome_iff_exists.mp
    (wr64_some_of_wr64 hm1s (hroom sp hr4 0 (by decide) (1 % W)))
  obtain ⟨k, hk, hrun⟩ := Status_exitB rc.fail1 (by unfold W; decide) sp v6 v2 v10
    v8 v9 m1 m2 (by omega) hspW s hrip hr4 hv6 hv2 hv10 hv8 hv9 hm1s hm2s
  exact ⟨k, hk, by simpa using hrun⟩

/-- Any state at a `jmp aFail1` failure site with valid stack exits `1` in
at most 6 steps. -/
theorem jmp_fail1_exit1B {im : Nat → Option (Ins × Nat)}
    (rc : RefuseCode im) {site sp : Nat}
    (hjmp : im site = some (.jmp aFail1, 5))
    (hsp32 : 32 ≤ sp) (hspW : sp < W) :
    ∀ s, s.rip = site → s.reg 4 = some sp → RegsOk s → StackRoom s 1 →
      ∃ m, m ≤ 6 ∧ run im m (Out.ok s) = Out.exit 1 := by
  intro s hrip hr4 hregs hroom
  have e1 : X86Int.step im s = .ok { s with rip := aFail1 } :=
    step_jmp (by rw [hrip]; exact hjmp)
  obtain ⟨s1, hs1⟩ : ∃ u : X86Int.St, u = { s with rip := aFail1 } := ⟨_, rfl⟩
  have r1 : run im 1 (Out.ok s) = Out.ok s1 := by
    rw [run_succ_ok im 0 s, e1, ← hs1, run_zero]
  have hs1rip : s1.rip = aFail1 := by rw [hs1]
  have hs1r4 : s1.reg 4 = some sp := by rw [hs1]; exact hr4
  have hs1mem : s1.mem = s.mem := by rw [hs1]
  obtain ⟨k, hk, hrun⟩ := exit1_of_fail1B rc hsp32 hspW s1 hs1rip hs1r4
    (run_regs_ok r1 hregs)
    (StackRoom_mono (StackRoom_of_mem hs1mem (by rw [hs1r4, hr4]) hroom) (by decide))
  exact ⟨1 + k, (by omega), by rw [run_add im 1 k, r1]; exact hrun⟩

private theorem annval_range_le (k6 k7 len : Nat) (hk6 : k6 ≤ 4) (hk7 : k7 ≤ 6) :
    5 + k6 + 3 + 1 + k7 ≤ 60 + 20 * (len + 1) := by
  omega

private theorem annval_dup_d_le (k6 k11 k15 k19 k24 k27 len : Nat)
    (hk6 : k6 ≤ 4) (hk11 : k11 ≤ 5 + 20 * (len + 1))
    (hk15 : k15 ≤ 4) (hk19 : k19 ≤ 4) (hk24 : k24 ≤ 4) (hk27 : k27 ≤ 6) :
    5 + k6 + 3 + (1 + k11 + 3 + k15 + 3 + k19 + 4 + k24 + 1) + 1 + k27 ≤
      60 + 20 * (len + 1) := by
  omega

private theorem annval_dup_a_le (k6 k11 k15 k19 k24 k29 k32 len : Nat)
    (hk6 : k6 ≤ 4) (hk11 : k11 ≤ 5 + 20 * (len + 1))
    (hk15 : k15 ≤ 4) (hk19 : k19 ≤ 4) (hk24 : k24 ≤ 4) (hk29 : k29 ≤ 4)
    (hk32 : k32 ≤ 6) :
    5 + k6 + 3 + (1 + k11 + 3 + k15 + 3 + k19 + 4 + k24 + 1) + (3 + k29 + 2) + k32 ≤
      60 + 20 * (len + 1) := by
  omega

private theorem codewalk_refuse_d_le (k6 k12 k15 len : Nat)
    (hk6 : k6 ≤ 5 + 20 * (len + 1)) (hk12 : k12 ≤ 4) (hk15 : k15 ≤ 6) :
    5 + k6 + 5 + k12 + 2 + k15 ≤ 40 + 20 * (len + 1) := by
  omega

private theorem codewalk_refuse_a_le (k6 k12 k17 k20 len : Nat)
    (hk6 : k6 ≤ 5 + 20 * (len + 1)) (hk12 : k12 ≤ 4) (hk17 : k17 ≤ 4)
    (hk20 : k20 ≤ 6) :
    5 + k6 + 5 + k12 + 4 + k17 + 2 + k20 ≤ 40 + 20 * (len + 1) := by
  omega

/-- `check_word` refuses an empty word (`code.length = 0`) in at most 12 steps. -/
theorem CheckWord_empty_exit1B {im : Nat → Option (Ins × Nat)}
    (oc : OuterCode im) (rc : RefuseCode im) :
    ∀ (w n sp : Nat), w + 56 ≤ W → n ≠ 0 → 48 ≤ sp → sp < W →
    ∀ s, s.rip = aCheckWord → s.reg 5 = some n → s.reg 7 = some w →
      s.reg 4 = some sp → rd64 s.mem (w + 24) = some 0 →
      RegsOk s → StackRoom s 6 →
      ∃ m, m ≤ 12 ∧ run im m (Out.ok s) = Out.exit 1 := by
  intro w n sp hw56 hn hsp48 hspW s hrip hr5 hr7 hr4 hfnc hregs hroom
  have e1 : X86Int.step im s =
      .ok { s with fl := (aluF .and n n).2, rip := s.rip + 3 } :=
    step_test (by rw [hrip]; exact oc.cw0) hr5 hr5
  obtain ⟨a1, ha1⟩ : ∃ u : X86Int.St, u =
    { s with fl := (aluF .and n n).2, rip := s.rip + 3 } := ⟨_, rfl⟩
  have ha1rip : a1.rip = aCheckWord + 3 := by
    rw [ha1]; show s.rip + 3 = aCheckWord + 3; rw [hrip]
  have ha1fl : X86Int.cond a1.fl 5 = some true := by
    have hz : a1.fl.zf = some false := by
      rw [ha1]
      show (aluF .and n n).2.zf = some false
      rw [aluF_zf, aluF_and_fst_self]
      simp [hn]
    have := cond_cc5 hz
    simpa using this
  have e2 : X86Int.step im a1 = .ok { a1 with rip := aCheckWord + 10 } := by
    have hh := step_jcc (im := im) (s := a1) (by rw [ha1rip]; exact oc.cw1) ha1fl
    simpa [ha1rip] using hh
  obtain ⟨a2, ha2⟩ : ∃ u : X86Int.St, u = { a1 with rip := aCheckWord + 10 } :=
    ⟨_, rfl⟩
  have ha2rip : a2.rip = aCheckWord + 10 := by rw [ha2]
  have ha2r7 : a2.reg 7 = some w := by
    rw [ha2]; show ({ a1 with rip := aCheckWord + 10 }).reg 7 = some w
    rw [reg_with_rip]; rw [ha1]; exact hr7
  have ha2mem : a2.mem = s.mem := by
    rw [ha2]; show ({ a1 with rip := aCheckWord + 10 }).mem = s.mem
    rw [mem_with_rip]; rw [ha1]
  have hea24 : ea w 24 = w + 24 := Nat.mod_eq_of_lt (by omega)
  have e3 : X86Int.step im a2 = .ok { a2.set 0 (some 0) with rip := a2.rip + 4 } :=
    step_ld (by rw [ha2rip]; exact oc.cw3) ha2r7 (by rw [hea24, ha2mem]; exact hfnc)
  obtain ⟨a3, ha3⟩ : ∃ u : X86Int.St, u =
    { a2.set 0 (some 0) with rip := a2.rip + 4 } := ⟨_, rfl⟩
  have ha3rip : a3.rip = aCheckWord + 14 := by
    rw [ha3]; show a2.rip + 4 = aCheckWord + 14; rw [ha2rip]
  have ha3r0 : a3.reg 0 = some 0 := by
    rw [ha3]
    show (a2.set 0 (some 0)).reg 0 = some 0
    rw [reg_set_self (by
      have e : a2.regs.length = s.regs.length := by rw [ha2, ha1]
      have h5len : 5 < s.regs.length := reg_some_len hr5
      omega)]
  have e4 : X86Int.step im a3 =
      .ok { a3 with fl := (aluF .and 0 0).2, rip := a3.rip + 3 } :=
    step_test (by rw [ha3rip]; exact oc.cw4) ha3r0 ha3r0
  obtain ⟨a4, ha4⟩ : ∃ u : X86Int.St, u =
    { a3 with fl := (aluF .and 0 0).2, rip := a3.rip + 3 } := ⟨_, rfl⟩
  have ha4rip : a4.rip = aCheckWord + 17 := by
    rw [ha4]; show a3.rip + 3 = aCheckWord + 17; rw [ha3rip]
  have ha4fl : X86Int.cond a4.fl 5 = some false := by
    have hz : a4.fl.zf = some true := by
      rw [ha4]
      show (aluF .and 0 0).2.zf = some true
      rw [aluF_zf, aluF_and_fst_self]
      exact rfl
    have := cond_cc5 hz
    simpa using this
  have ga4 : ∀ q, q ≠ 0 → a4.reg q = s.reg q := by
    intro q hq
    rw [ha4]; show a3.reg q = s.reg q
    rw [ha3]; show (a2.set 0 (some 0)).reg q = s.reg q
    rw [reg_set_other hq, ha2]; show ({ a1 with rip := aCheckWord + 10 }).reg q = s.reg q
    rw [reg_with_rip, ha1]; exact rfl
  have ha4mem : a4.mem = s.mem := by
    rw [ha4]; show a3.mem = s.mem
    rw [ha3]; show (a2.set 0 (some 0)).mem = s.mem
    rw [mem_set]; exact ha2mem
  have e5 : X86Int.step im a4 = .ok { a4 with rip := aCheckWord + 19 } := by
    have hh := step_jcc (im := im) (s := a4) (by rw [ha4rip]; exact oc.cw5) ha4fl
    simpa [ha4rip] using hh
  obtain ⟨a5, ha5⟩ : ∃ u : X86Int.St, u = { a4 with rip := aCheckWord + 19 } :=
    ⟨_, rfl⟩
  have ha5rip : a5.rip = aCheckWord + 19 := by rw [ha5]
  have ha5mem : a5.mem = s.mem := by rw [ha5]; exact ha4mem
  have ha5r4 : a5.reg 4 = some sp := by
    rw [ha5]; show ({ a4 with rip := aCheckWord + 19 }).reg 4 = some sp
    rw [reg_with_rip, ga4 4 (by decide)]; exact hr4
  have r1 : run im 1 (Out.ok s) = Out.ok a1 := by
    rw [run_succ_ok im 0 s, e1, ← ha1, run_zero]
  have r2 : run im 1 (Out.ok a1) = Out.ok a2 := by
    rw [run_succ_ok im 0 a1, e2, ← ha2, run_zero]
  have r3 : run im 1 (Out.ok a2) = Out.ok a3 := by
    rw [run_succ_ok im 0 a2, e3, ← ha3, run_zero]
  have r4 : run im 1 (Out.ok a3) = Out.ok a4 := by
    rw [run_succ_ok im 0 a3, e4, ← ha4, run_zero]
  have r5 : run im 1 (Out.ok a4) = Out.ok a5 := by
    rw [run_succ_ok im 0 a4, e5, ← ha5, run_zero]
  have hrun5 : run im 5 (Out.ok s) = Out.ok a5 := by
    rw [run_add im 1 4, r1, run_add im 1 3, r2, run_add im 1 2, r3,
      run_add im 1 1, r4, r5]
  obtain ⟨k, hk, hrun⟩ := jmp_fail1_exit1B rc rc.cwEmpty (by omega) hspW a5
    ha5rip ha5r4 (run_regs_ok hrun5 hregs)
    (StackRoom_mono (StackRoom_of_mem ha5mem (by rw [ha5r4, hr4]) hroom) (by decide))
  exact ⟨5 + k, (by omega), by rw [run_add im 5 k, hrun5]; exact hrun⟩

/-- Bounded prefix of `annval` shared by all `annval` refusal branches:
pops index `t` and branches through `cmp rax, [rdi+24]`. If `t ≥ wd.code.length`
it exits `1` via `avRange`; otherwise it reaches `aAnnval + 37` with the `cb`
taken. -/
private theorem annval_step_head {im : Nat → Option (Ins × Nat)}
    (ac : AllCode im) (oc : OuterCode im)
    {wd : Sporos.Word} {w p sp : Nat} (st : AnnStatic wd w p sp)
    {t : Nat} {h : Height} {rest' : List (Nat × Height)}
    (hlen : rest'.length + 1 ≤ wd.ann.length) :
    ∀ s, AnnSt wd w p sp ((t, h) :: rest') s →
      AnnAt s (p + 12 * (wd.ann.length - (rest'.length + 1))) t h.d h.a →
      ∃ k6, k6 ≤ 4 ∧ ∃ b9 : X86Int.St,
        run im (5 + k6 + 3) (Out.ok s) = Out.ok b9 ∧
        b9.rip = aAnnval + 30 ∧
        b9.fl = (aluF .sub t wd.code.length).2 ∧
        b9.reg 0 = some t ∧
        b9.reg 2 = some wd.code.length ∧
        b9.reg 12 = some t ∧
        b9.reg 7 = some w ∧
        b9.reg 4 = some (sp - 16) ∧
        b9.reg 1 = some (rest'.length + 1) ∧
        b9.reg 6 = some (p + 12 * (wd.ann.length - (rest'.length + 1))) ∧
        (∀ q, q ≠ 0 → q ≠ 2 → q ≠ 4 → q ≠ 12 → b9.reg q = s.reg q) ∧
        (∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x → b9.mem.rdN x n = s.mem.rdN x n) ∧
        wshape b9.mem = wshape s.mem ∧
        b9.regs.length = s.regs.length ∧
        rd64 b9.mem (sp - 16) = some (p + 12 * (wd.ann.length - (rest'.length + 1))) ∧
        rd64 b9.mem (sp - 8) = some (rest'.length + 1) := by
  obtain ⟨hw56, hp0, hpb, hsp48, hspW, hncW, hhW, hdisjw, hdisjp⟩ := st
  intro s hs hcur
  obtain ⟨hrip, hr7, hr6, hr1, hr4, hf24, hf32, hf48, hlay, h12len, hroom⟩ := hs
  have hnrem : rest'.length + 1 ≠ 0 := by omega
  have hcurb : p + 12 * (wd.ann.length - (rest'.length + 1)) + 12 ≤
      p + 12 * wd.ann.length := by
    have : wd.ann.length - (rest'.length + 1) + 1 ≤ wd.ann.length := by omega
    omega
  have hannW : wd.ann.length < W := by omega
  have hslot8 : (sp + W - 8) % W = sp - 8 := by
    rw [show sp + W - 8 = sp + W - 8 * 1 by omega, slot_sub (by omega) hspW]
  have atOf : ∀ u : X86Int.St, (∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      u.mem.rdN x n = s.mem.rdN x n) →
      AnnAt u (p + 12 * (wd.ann.length - (rest'.length + 1))) t h.d h.a :=
    fun u hu => ⟨by
        rw [hu (p + 12 * (wd.ann.length - (rest'.length + 1))) 4 (by omega)]
        exact hcur.1,
      by
        rw [hu (p + 12 * (wd.ann.length - (rest'.length + 1)) + 4) 4 (by omega)]
        exact hcur.2.1,
      by
        rw [hu (p + 12 * (wd.ann.length - (rest'.length + 1)) + 8) 4 (by omega)]
        exact hcur.2.2⟩
  have fldOf : ∀ u : X86Int.St, (∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      u.mem.rdN x n = s.mem.rdN x n) →
      rd64 u.mem (w + 24) = some wd.code.length ∧
      rd64 u.mem (w + 32) = some wd.ann.length ∧
      rd64 u.mem (w + 48) = some p :=
    fun u hu => ⟨by unfold rd64; rw [hu (w + 24) 8 (by omega)]; exact hf24,
      by unfold rd64; rw [hu (w + 32) 8 (by omega)]; exact hf32,
      by unfold rd64; rw [hu (w + 48) 8 (by omega)]; exact hf48⟩
  have hfr0 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x → s.mem.rdN x n = s.mem.rdN x n :=
    fun x n _ => rfl
  have hr1' : s.reg 1 = some (rest'.length + 1) := hr1
  -- test rcx, rcx ; jnz taken
  have e1 : X86Int.step im s =
      .ok { s with fl := (aluF .and (rest'.length + 1) (rest'.length + 1)).2, rip := s.rip + 3 } :=
    step_test (by rw [hrip]; exact oc.av0) hr1' hr1'
  obtain ⟨b1, hb1⟩ : ∃ u : X86Int.St, u =
    { s with fl := (aluF .and (rest'.length + 1) (rest'.length + 1)).2, rip := s.rip + 3 } := ⟨_, rfl⟩
  have hb1rip : b1.rip = aAnnval + 3 := by rw [hb1]; show s.rip + 3 = aAnnval + 3; rw [hrip]
  have hb1fl : X86Int.cond b1.fl 5 = some true := by
    have hz : b1.fl.zf = some false := by
      rw [hb1]
      show (aluF .and (rest'.length + 1) (rest'.length + 1)).2.zf = some false
      rw [aluF_zf, aluF_and_fst_self]; simp
    have := cond_cc5 hz
    simpa using this
  have gb1 : ∀ q, b1.reg q = s.reg q := fun q => by rw [hb1]; exact rfl
  have hb1mem : b1.mem = s.mem := by rw [hb1]
  have e2 : X86Int.step im b1 = .ok { b1 with rip := aAnnval + 10 } := by
    have hh := step_jcc (im := im) (s := b1) (by rw [hb1rip]; exact oc.av1) hb1fl
    simpa using hh
  obtain ⟨b2, hb2⟩ : ∃ u : X86Int.St, u = { b1 with rip := aAnnval + 10 } := ⟨_, rfl⟩
  have hb2rip : b2.rip = aAnnval + 10 := by rw [hb2]
  have gb2 : ∀ q, b2.reg q = s.reg q := fun q => by rw [hb2]; exact gb1 q
  have hb2mem : b2.mem = s.mem := by rw [hb2]; exact hb1mem
  have hb2r4 : b2.reg 4 = some sp := by rw [gb2 4]; exact hr4
  have hb2r1 : b2.reg 1 = some (rest'.length + 1) := by rw [gb2 1]; exact hr1'
  have hlen2 : b2.regs.length = s.regs.length := by rw [hb2, hb1]
  -- push rcx
  have hw8 : ∀ v, (wr64 s.mem (sp - 8) v).isSome := by
    intro v
    have h := hroom sp hr4 0 (by decide) v
    have e : (sp + W - 8 * (0 + 1)) % W = sp - 8 := slot_sub (by omega) hspW
    rw [e] at h; exact h
  have hw16s : ∀ v, (wr64 s.mem (sp - 16) v).isSome := by
    intro v
    have h := hroom sp hr4 1 (by decide) v
    have e : (sp + W - 8 * (1 + 1)) % W = sp - 16 := slot_sub (by omega) hspW
    rw [e] at h; exact h
  obtain ⟨m3, hm3s⟩ := Option.isSome_iff_exists.mp (hw8 (rest'.length + 1))
  have hm3 : wr64 b2.mem ((sp + W - 8) % W) (rest'.length + 1) = some m3 := by
    rw [hb2mem, hslot8]; exact hm3s
  have e3 : X86Int.step im b2 = .ok { b2.set 4 (some ((sp + W - 8) % W)) with
      mem := m3, rip := b2.rip + 1 } :=
    step_push (by rw [hb2rip]; exact oc.av3) hb2r1 hb2r4 hm3
  obtain ⟨b3, hb3⟩ : ∃ u : X86Int.St, u =
    { b2.set 4 (some ((sp + W - 8) % W)) with mem := m3, rip := b2.rip + 1 } := ⟨_, rfl⟩
  have hb3rip : b3.rip = aAnnval + 11 := by rw [hb3]; show b2.rip + 1 = aAnnval + 11; rw [hb2rip]
  have hb3r4 : b3.reg 4 = some (sp - 8) := by
    rw [hb3]
    show (b2.set 4 (some ((sp + W - 8) % W))).reg 4 = some (sp - 8)
    rw [reg_set_self (by omega), hslot8]
  have gb3 : ∀ q, q ≠ 4 → b3.reg q = s.reg q := by
    intro q hq; rw [hb3]
    show (b2.set 4 _).reg q = s.reg q
    rw [reg_set_other hq]; exact gb2 q
  have hb3mem : b3.mem = m3 := by rw [hb3]
  have hfr3 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x → b3.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; rw [hb3mem]; exact rdN_wr64_other hm3s (by omega)
  have hw3 : wshape b3.mem = wshape s.mem := by rw [hb3mem]; exact wrN_wshape _ _ _ hm3s
  have hlen3 : b3.regs.length = s.regs.length := by
    rw [hb3]; show (b2.set 4 _).regs.length = s.regs.length
    exact List.length_set.trans hlen2
  have hslot8_3 : rd64 b3.mem (sp - 8) = some (rest'.length + 1) := by
    have h1 := rd64_wr64 hm3s
    rw [Nat.mod_eq_of_lt (by omega)] at h1; rw [hb3mem]; exact h1
  -- push rsi
  obtain ⟨m4, hm4s⟩ := Option.isSome_iff_exists.mp
    (wr64_some_of_wr64 hm3s (hw16s (p + 12 * (wd.ann.length - (rest'.length + 1)))))
  have hsp16 : ((sp - 8) + W - 8) % W = sp - 16 := by
    rw [show (sp - 8) + W - 8 = (sp - 8) + W - 8 * 1 by omega, slot_sub (by omega) (by omega)]
    omega
  have hm4 : wr64 b3.mem (((sp - 8) + W - 8) % W)
      (p + 12 * (wd.ann.length - (rest'.length + 1))) = some m4 := by
    rw [hb3mem, hsp16]; exact hm4s
  have hb3r6 : b3.reg 6 = some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [gb3 6 (by decide)]; exact hr6
  have e4 : X86Int.step im b3 = .ok { b3.set 4 (some (((sp - 8) + W - 8) % W))
      with mem := m4, rip := b3.rip + 1 } :=
    step_push (by rw [hb3rip]; exact oc.av4) hb3r6 hb3r4 hm4
  obtain ⟨b4, hb4⟩ : ∃ u : X86Int.St, u =
    { b3.set 4 (some (((sp - 8) + W - 8) % W)) with mem := m4, rip := b3.rip + 1 } := ⟨_, rfl⟩
  have hb4rip : b4.rip = aAnnval + 12 := by rw [hb4]; show b3.rip + 1 = aAnnval + 12; rw [hb3rip]
  have hb4r4 : b4.reg 4 = some (sp - 16) := by
    rw [hb4]; show (b3.set 4 _).reg 4 = some (sp - 16)
    rw [reg_set_self (by omega), hsp16]
  have gb4 : ∀ q, q ≠ 4 → b4.reg q = s.reg q := by
    intro q hq; rw [hb4]; show (b3.set 4 _).reg q = s.reg q
    rw [reg_set_other hq]; exact gb3 q hq
  have hb4mem : b4.mem = m4 := by rw [hb4]
  have hfr4 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x → b4.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; rw [hb4mem, rdN_wr64_other hm4s (by omega), ← hb3mem]; exact hfr3 x n hx
  have hw4 : wshape b4.mem = wshape s.mem := by
    rw [hb4mem]; exact (wrN_wshape _ _ _ hm4s).trans (by rw [← hb3mem]; exact hw3)
  have hlen4 : b4.regs.length = s.regs.length := by
    rw [hb4]; show (b3.set 4 _).regs.length = s.regs.length
    exact List.length_set.trans hlen3
  have hslot16_4 : rd64 b4.mem (sp - 16) =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    have h1 := rd64_wr64 hm4s
    rw [Nat.mod_eq_of_lt (by omega)] at h1; rw [hb4mem]; exact h1
  have hslot8_4 : rd64 b4.mem (sp - 8) = some (rest'.length + 1) := by
    rw [hb4mem, rd64_wr64_other hm4s (by omega), ← hb3mem]; exact hslot8_3
  -- mov rdx, rsi
  have hb4r6 : b4.reg 6 = some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [gb4 6 (by decide)]; exact hr6
  have e5 : X86Int.step im b4 = .ok { b4.set 2
      (some (p + 12 * (wd.ann.length - (rest'.length + 1)))) with rip := b4.rip + 3 } :=
    step_mov (by rw [hb4rip]; exact oc.av5) hb4r6
  obtain ⟨b5, hb5⟩ : ∃ u : X86Int.St, u = { b4.set 2
      (some (p + 12 * (wd.ann.length - (rest'.length + 1)))) with rip := b4.rip + 3 } := ⟨_, rfl⟩
  have hb5rip : b5.rip = aAnnval + 15 := by rw [hb5]; show b4.rip + 3 = aAnnval + 15; rw [hb4rip]
  have hb5r2 : b5.reg 2 = some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [hb5]; show (b4.set 2 _).reg 2 = _; rw [reg_set_self (by omega)]
  have gb5 : ∀ q, q ≠ 2 → b5.reg q = b4.reg q := by
    intro q hq; rw [hb5]; show (b4.set 2 _).reg q = b4.reg q; rw [reg_set_other hq]
  have hb5mem : b5.mem = b4.mem := by rw [hb5]; exact rfl
  have hb5r4 : b5.reg 4 = some (sp - 16) := by rw [gb5 4 (by decide)]; exact hb4r4
  have hlen5 : b5.regs.length = s.regs.length := by
    rw [hb5]; show (b4.set 2 _).regs.length = s.regs.length; exact List.length_set.trans hlen4
  have hfr5 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x → b5.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; rw [hb5mem]; exact hfr4 x n hx
  have hw5 : wshape b5.mem = wshape s.mem := by rw [hb5mem]; exact hw4
  have hb5r1 : b5.reg 1 = some (rest'.length + 1) := by
    rw [gb5 1 (by decide), gb4 1 (by decide)]; exact hr1'
  have hrd5 : (b5.mem.rdN (p + 12 * (wd.ann.length - (rest'.length + 1))) 4).map leVal = some t :=
    (atOf b5 hfr5).1
  obtain ⟨k6, b6, hk6, hrun6, hb6rip, hb6r0, hb6r4, gb6, hlen6, hfr6seg, hw6seg⟩ :=
    CallL32B ac.l32 ac.annl.dret oc.av6 (by omega) (by omega) (by omega)
      (by decide) (by omega) b5 hb5rip hb5r2 hb5r4 hrd5
      (StackRoom_shift hroom hr4 hb5r4 hw5 ⟨2, rfl, by omega⟩ hsp48 hspW)
  have hlen6s : b6.regs.length = s.regs.length := hlen6.trans hlen5
  have hfr6 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x → b6.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; exact (hfr6seg x n (by omega)).trans (hfr5 x n hx)
  have hw6 : wshape b6.mem = wshape s.mem := hw6seg.trans hw5
  have hb6r7 : b6.reg 7 = some w := by
    rw [gb6 7 (by decide) (by decide), gb5 7 (by decide), gb4 7 (by decide)]; exact hr7
  have hb6r1 : b6.reg 1 = some (rest'.length + 1) := by
    rw [gb6 1 (by decide) (by decide)]; exact hb5r1
  have hb6r6 : b6.reg 6 = some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [gb6 6 (by decide) (by decide), gb5 6 (by decide)]; exact hb4r6
  have hslot16_6 : rd64 b6.mem (sp - 16) =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    unfold rd64; rw [hfr6seg (sp - 16) 8 (by omega), hb5mem]; exact hslot16_4
  have hslot8_6 : rd64 b6.mem (sp - 8) = some (rest'.length + 1) := by
    unfold rd64; rw [hfr6seg (sp - 8) 8 (by omega), hb5mem]; exact hslot8_4
  -- mov r12, rax
  have e7 : X86Int.step im b6 = .ok { b6.set 12 (some t) with rip := b6.rip + 3 } :=
    step_mov (by rw [hb6rip]; exact oc.av7) hb6r0
  obtain ⟨b7, hb7⟩ : ∃ u : X86Int.St, u = { b6.set 12 (some t) with rip := b6.rip + 3 } := ⟨_, rfl⟩
  have hb7rip : b7.rip = aAnnval + 23 := by rw [hb7]; show b6.rip + 3 = aAnnval + 23; rw [hb6rip]
  have hb7r12 : b7.reg 12 = some t := by
    rw [hb7]; show (b6.set 12 (some t)).reg 12 = some t; rw [reg_set_self (by omega)]
  have gb7 : ∀ q, q ≠ 12 → b7.reg q = b6.reg q := by
    intro q hq; rw [hb7]; show (b6.set 12 (some t)).reg q = b6.reg q; rw [reg_set_other hq]
  have hb7mem : b7.mem = b6.mem := by rw [hb7]; exact rfl
  have hlen7 : b7.regs.length = s.regs.length := by
    rw [hb7]; show (b6.set 12 (some t)).regs.length = s.regs.length
    exact List.length_set.trans hlen6s
  have hfr7 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x → b7.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; rw [hb7mem]; exact hfr6 x n hx
  have hw7 : wshape b7.mem = wshape s.mem := by rw [hb7mem]; exact hw6
  have hb7r7 : b7.reg 7 = some w := by rw [gb7 7 (by decide)]; exact hb6r7
  -- mov rdx, [rdi+24]
  have hea24 : ea w 24 = w + 24 := Nat.mod_eq_of_lt (by omega)
  have e8 : X86Int.step im b7 = .ok { b7.set 2 (some wd.code.length) with rip := b7.rip + 4 } :=
    step_ld (by rw [hb7rip]; exact oc.av8) hb7r7 (by rw [hea24]; exact (fldOf b7 hfr7).1)
  obtain ⟨b8, hb8⟩ : ∃ u : X86Int.St, u = { b7.set 2 (some wd.code.length) with rip := b7.rip + 4 } := ⟨_, rfl⟩
  have hb8rip : b8.rip = aAnnval + 27 := by rw [hb8]; show b7.rip + 4 = aAnnval + 27; rw [hb7rip]
  have gb8 : ∀ q, q ≠ 2 → b8.reg q = b7.reg q := by
    intro q hq; rw [hb8]; show (b7.set 2 _).reg q = b7.reg q; rw [reg_set_other hq]
  have hb8r2 : b8.reg 2 = some wd.code.length := by
    rw [hb8]; show (b7.set 2 _).reg 2 = some wd.code.length; rw [reg_set_self (by omega)]
  have hb8r0 : b8.reg 0 = some t := by rw [gb8 0 (by decide), gb7 0 (by decide)]; exact hb6r0
  have hb8mem : b8.mem = b7.mem := by rw [hb8]; exact rfl
  have hlen8 : b8.regs.length = s.regs.length := by
    rw [hb8]; show (b7.set 2 _).regs.length = s.regs.length; exact List.length_set.trans hlen7
  have hfr8 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x → b8.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; rw [hb8mem]; exact hfr7 x n hx
  have hw8mem : wshape b8.mem = wshape s.mem := by rw [hb8mem]; exact hw7
  -- cmp rax, rdx
  have e9 : X86Int.step im b8 =
      .ok { b8 with fl := (aluF .sub t wd.code.length).2, rip := b8.rip + 3 } :=
    step_cmp (by rw [hb8rip]; exact oc.av9) hb8r0 hb8r2
  obtain ⟨b9, hb9⟩ : ∃ u : X86Int.St, u =
    { b8 with fl := (aluF .sub t wd.code.length).2, rip := b8.rip + 3 } := ⟨_, rfl⟩
  have rr1 : run im 1 (Out.ok s) = Out.ok b1 := by rw [run_succ_ok im 0 s, e1, ← hb1, run_zero]
  have rr2 : run im 1 (Out.ok b1) = Out.ok b2 := by rw [run_succ_ok im 0 b1, e2, ← hb2, run_zero]
  have rr3 : run im 1 (Out.ok b2) = Out.ok b3 := by rw [run_succ_ok im 0 b2, e3, ← hb3, run_zero]
  have rr4 : run im 1 (Out.ok b3) = Out.ok b4 := by rw [run_succ_ok im 0 b3, e4, ← hb4, run_zero]
  have rr5 : run im 1 (Out.ok b4) = Out.ok b5 := by rw [run_succ_ok im 0 b4, e5, ← hb5, run_zero]
  have rr7 : run im 1 (Out.ok b6) = Out.ok b7 := by rw [run_succ_ok im 0 b6, e7, ← hb7, run_zero]
  have rr8 : run im 1 (Out.ok b7) = Out.ok b8 := by rw [run_succ_ok im 0 b7, e8, ← hb8, run_zero]
  have rr9 : run im 1 (Out.ok b8) = Out.ok b9 := by rw [run_succ_ok im 0 b8, e9, ← hb9, run_zero]
  have hr5 : run im 5 (Out.ok s) = Out.ok b5 := by
    rw [run_add im 1 4, rr1, run_add im 1 3, rr2, run_add im 1 2, rr3, run_add im 1 1, rr4, rr5]
  have hr6 : run im (5 + k6) (Out.ok s) = Out.ok b6 := by
    rw [run_add im 5 k6, hr5, hrun6]
  have hr9 : run im (5 + k6 + 3) (Out.ok s) = Out.ok b9 := by
    rw [run_add im (5 + k6) 3, hr6, run_add im 1 2, rr7, run_add im 1 1, rr8, rr9]
  refine ⟨k6, hk6, b9, hr9, by rw [hb9]; show b8.rip + 3 = aAnnval + 30; rw [hb8rip],
    by rw [hb9], by rw [hb9]; exact hb8r0, by rw [hb9]; exact hb8r2,
    by rw [hb9]; show b8.reg 12 = some t; rw [gb8 12 (by decide)]; exact hb7r12,
    by rw [hb9]; show b8.reg 7 = some w; rw [gb8 7 (by decide)]; exact hb7r7,
    by rw [hb9]; show b8.reg 4 = some (sp - 16); rw [gb8 4 (by decide), gb7 4 (by decide)]; exact hb6r4,
    by rw [hb9]; show b8.reg 1 = some (rest'.length + 1); rw [gb8 1 (by decide), gb7 1 (by decide)]; exact hb6r1,
    by rw [hb9]; show b8.reg 6 = _; rw [gb8 6 (by decide), gb7 6 (by decide)]; exact hb6r6,
    ?_, by rw [hb9]; exact hfr8, by rw [hb9]; exact hw8mem, by rw [hb9]; exact hlen8,
    by rw [hb9, hb8mem, hb7mem]; exact hslot16_6,
    by rw [hb9, hb8mem, hb7mem]; exact hslot8_6⟩
  intro q h0 h2 h4 h12
  rw [hb9]; show b8.reg q = s.reg q
  rw [gb8 q h2, gb7 q h12, gb6 q h0 h4, gb5 q h2, gb4 q h4]

/-- One `annval` refusal: an entry with index `≥ wd.code.length` exits `1`. -/
theorem annval_range_exit1B {im : Nat → Option (Ins × Nat)}
    (ac : AllCode im) (oc : OuterCode im) (rc : RefuseCode im)
    {wd : Sporos.Word} {w p sp : Nat} (st : AnnStatic wd w p sp)
    {t : Nat} {h : Height} {rest' : List (Nat × Height)}
    (hlen : rest'.length + 1 ≤ wd.ann.length) (hge : wd.code.length ≤ t) :
    ∀ s, AnnSt wd w p sp ((t, h) :: rest') s →
      AnnAt s (p + 12 * (wd.ann.length - (rest'.length + 1))) t h.d h.a →
      RegsOk s →
      ∃ m, m ≤ 60 + 20 * (wd.ann.length + 1) ∧ run im m (Out.ok s) = Out.exit 1 := by
  intro s hs hcur hregs
  have hsp48 := st.2.2.2.1
  have hspW := st.2.2.2.2.1
  have hr4 := hs.2.2.2.2.1
  have hroom := hs.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k6, hk6, b9, hrun9, hb9rip, hb9fl, -, -, -, -, hb9r4, -, -, -, -, hwb9, -, -, -⟩ :=
    annval_step_head ac oc st hlen s hs hcur
  have hb9cond : X86Int.cond b9.fl 2 = some false := by
    have hcf : b9.fl.cf = some false := by rw [hb9fl]; exact aluF_sub_cf (by omega)
    exact cond_cc2 hcf
  have e10 : X86Int.step im b9 = .ok { b9 with rip := aAnnval + 32 } := by
    have hh := step_jcc (im := im) (s := b9) (by rw [hb9rip]; exact oc.av10) hb9cond
    simpa [hb9rip] using hh
  obtain ⟨b10, hb10⟩ : ∃ u : X86Int.St, u = { b9 with rip := aAnnval + 32 } := ⟨_, rfl⟩
  have r10 : run im 1 (Out.ok b9) = Out.ok b10 := by
    rw [run_succ_ok im 0 b9, e10, ← hb10, run_zero]
  have hrun10 : run im (5 + k6 + 3 + 1) (Out.ok s) = Out.ok b10 := by
    rw [run_add im (5 + k6 + 3) 1, hrun9, r10]
  have hb10rip : b10.rip = aAnnval + 32 := by rw [hb10]
  have hb10r4 : b10.reg 4 = some (sp - 16) := by rw [hb10]; exact hb9r4
  have hwb10 : wshape b10.mem = wshape s.mem := by rw [hb10]; exact hwb9
  obtain ⟨k7, hk7, hrun7⟩ := jmp_fail1_exit1B rc rc.avRange (by omega) (by omega)
    b10 hb10rip hb10r4 (run_regs_ok hrun10 hregs)
    (StackRoom_shift hroom hr4 hb10r4 hwb10 ⟨2, rfl, by omega⟩ hsp48 hspW)
  have hle : 5 + k6 + 3 + 1 + k7 ≤ 60 + 20 * (wd.ann.length + 1) :=
    annval_range_le k6 k7 wd.ann.length hk6 hk7
  exact ⟨5 + k6 + 3 + 1 + k7, hle, by
    rw [run_add im (5 + k6 + 3 + 1) k7, hrun10]; exact hrun7⟩

/-- Helper for `annval_dup_exit1B`: runs from `b9` (`aAnnval + 30`, `t < code.length`)
past `CallANNLB`, loads `h'.d` into `r8`, `h'.a` into `r9`, `h.d` into `rax`,
and stops at `aAnnval + 87` after `cmp rax, r8`. -/
private theorem annval_dup_mid {im : Nat → Option (Ins × Nat)}
    (ac : AllCode im) (oc : OuterCode im)
    {wd : Sporos.Word} {w p sp : Nat} (st : AnnStatic wd w p sp)
    {t : Nat} {h h' : Height} {rest' : List (Nat × Height)}
    (hlen : rest'.length + 1 ≤ wd.ann.length)
    (htlt : t < wd.code.length) (htW : t < W) (hlk : wd.ann.lookup t = some h')
    {s b9 : X86Int.St} (hs : AnnSt wd w p sp ((t, h) :: rest') s)
    (hcur : AnnAt s (p + 12 * (wd.ann.length - (rest'.length + 1))) t h.d h.a)
    (hb9rip : b9.rip = aAnnval + 30) (hb9fl : b9.fl = (aluF .sub t wd.code.length).2)
    (hb9r12 : b9.reg 12 = some t) (hb9r7 : b9.reg 7 = some w)
    (hb9r4 : b9.reg 4 = some (sp - 16)) (hb9r1 : b9.reg 1 = some (rest'.length + 1))
    (hb9r6 : b9.reg 6 = some (p + 12 * (wd.ann.length - (rest'.length + 1))))
    (hfr9 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x → b9.mem.rdN x n = s.mem.rdN x n)
    (hwb9 : wshape b9.mem = wshape s.mem) (hlen9 : b9.regs.length = s.regs.length)
    (hslot16_9 : rd64 b9.mem (sp - 16) = some (p + 12 * (wd.ann.length - (rest'.length + 1))))
    (hslot8_9 : rd64 b9.mem (sp - 8) = some (rest'.length + 1)) :
    ∃ k11 k15 k19 k24,
      k11 ≤ 5 + 20 * (wd.ann.length + 1) ∧ k15 ≤ 4 ∧ k19 ≤ 4 ∧ k24 ≤ 4 ∧
      ∃ b25 : X86Int.St,
        run im (1 + k11 + 3 + k15 + 3 + k19 + 4 + k24 + 1) (Out.ok b9) = Out.ok b25 ∧
        b25.rip = aAnnval + 87 ∧
        b25.fl = (aluF .sub h.d h'.d).2 ∧
        b25.reg 4 = some (sp - 8) ∧
        b25.reg 6 = some (p + 12 * (wd.ann.length - (rest'.length + 1))) ∧
        b25.reg 9 = some h'.a ∧
        (∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x → b25.mem.rdN x n = s.mem.rdN x n) ∧
        wshape b25.mem = wshape s.mem ∧
        b25.regs.length = s.regs.length := by
  obtain ⟨hw56, hp0, hpb, hsp48, hspW, hncW, hhW, hdisjw, hdisjp⟩ := st
  obtain ⟨_, _, _, _, hr4, hf24, hf32, hf48, hlay, h12len, hroom⟩ := hs
  have hcurb : p + 12 * (wd.ann.length - (rest'.length + 1)) + 12 ≤
      p + 12 * wd.ann.length := by
    have : wd.ann.length - (rest'.length + 1) + 1 ≤ wd.ann.length := by omega
    omega
  have layOf : ∀ u : X86Int.St, (∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      u.mem.rdN x n = s.mem.rdN x n) → AnnLayout u wd.ann p :=
    fun u hu => AnnLayout_of_frame hlay (fun x k hx1 hx2 => hu x k (by omega))
  have atOf : ∀ u : X86Int.St, (∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      u.mem.rdN x n = s.mem.rdN x n) →
      AnnAt u (p + 12 * (wd.ann.length - (rest'.length + 1))) t h.d h.a :=
    fun u hu => ⟨by
        rw [hu (p + 12 * (wd.ann.length - (rest'.length + 1))) 4 (by omega)]; exact hcur.1,
      by rw [hu (p + 12 * (wd.ann.length - (rest'.length + 1)) + 4) 4 (by omega)]; exact hcur.2.1,
      by rw [hu (p + 12 * (wd.ann.length - (rest'.length + 1)) + 8) 4 (by omega)]; exact hcur.2.2⟩
  have fldOf : ∀ u : X86Int.St, (∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      u.mem.rdN x n = s.mem.rdN x n) →
      rd64 u.mem (w + 24) = some wd.code.length ∧
      rd64 u.mem (w + 32) = some wd.ann.length ∧
      rd64 u.mem (w + 48) = some p :=
    fun u hu => ⟨by unfold rd64; rw [hu (w + 24) 8 (by omega)]; exact hf24,
      by unfold rd64; rw [hu (w + 32) 8 (by omega)]; exact hf32,
      by unfold rd64; rw [hu (w + 48) 8 (by omega)]; exact hf48⟩
  have hadd : ∀ (x k : Nat), k < W → x + k < W → (aluF .add x (k % W)).1 = x + k := by
    intro x k hkW hk
    rw [aluF_add_fst, Nat.mod_eq_of_lt hkW]
    exact Nat.mod_eq_of_lt hk
  -- jb taken
  have hb9cond : X86Int.cond b9.fl 2 = some true := by
    have hcf : b9.fl.cf = some true := by rw [hb9fl]; exact aluF_sub_cf_true htlt
    exact cond_cc2 hcf
  have e10 : X86Int.step im b9 = .ok { b9 with rip := aAnnval + 37 } := by
    have hh := step_jcc (im := im) (s := b9) (by rw [hb9rip]; exact oc.av10) hb9cond
    simpa using hh
  obtain ⟨b10, hb10⟩ : ∃ u : X86Int.St, u = { b9 with rip := aAnnval + 37 } := ⟨_, rfl⟩
  have hb10rip : b10.rip = aAnnval + 37 := by rw [hb10]
  have gb10 : ∀ q, b10.reg q = b9.reg q := fun q => by rw [hb10]; exact rfl
  have hb10mem : b10.mem = b9.mem := by rw [hb10]
  have hfr10 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x → b10.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; rw [hb10mem]; exact hfr9 x n hx
  have hw10 : wshape b10.mem = wshape s.mem := by rw [hb10mem]; exact hwb9
  have hlen10 : b10.regs.length = s.regs.length := by rw [hb10]; exact hlen9
  have hb10r12 : b10.reg 12 = some t := by rw [gb10 12]; exact hb9r12
  have hb10r7 : b10.reg 7 = some w := by rw [gb10 7]; exact hb9r7
  have hb10r4 : b10.reg 4 = some (sp - 16) := by rw [gb10 4]; exact hb9r4
  have hb10r1 : b10.reg 1 = some (rest'.length + 1) := by rw [gb10 1]; exact hb9r1
  have hb10r6 : b10.reg 6 = some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [gb10 6]; exact hb9r6
  have hslot16_10 : rd64 b10.mem (sp - 16) =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [hb10mem]; exact hslot16_9
  have hslot8_10 : rd64 b10.mem (sp - 8) = some (rest'.length + 1) := by
    rw [hb10mem]; exact hslot8_9
  -- call ANNL: rax = first entry e for index t
  obtain ⟨k11, b11, hk11, hrun11, hb11rip, hpost, hb11r1, hb11r4, hb11r12, hb11r7,
    hlay11, gb11, hfr11seg, hex0, hloc11, hw11seg, hlen11⟩ :=
    CallANNLB ac.annl ac.l32 (by decide) oc.av11 (by decide) htW hp0 hw56
      hpb (by omega) (by omega) b10 hb10rip hb10r12 hb10r7 hb10r4 hb10r1
      (fldOf b10 hfr10).2.2 (fldOf b10 hfr10).2.1 (layOf b10 hfr10)
      (StackRoom_shift hroom hr4 hb10r4 hw10 ⟨2, rfl, by omega⟩ hsp48 hspW)
      (by omega) (by omega)
  have hlen11s : b11.regs.length = s.regs.length := hlen11.trans hlen10
  have hfr11 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x → b11.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; exact (hfr11seg x n (by omega)).trans (hfr10 x n hx)
  have hw11 : wshape b11.mem = wshape s.mem := hw11seg.trans hw10
  have hslot16_11 : rd64 b11.mem (sp - 16) =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    unfold rd64; rw [hfr11seg (sp - 16) 8 (by omega)]; exact hslot16_10
  have hslot8_11 : rd64 b11.mem (sp - 8) = some (rest'.length + 1) := by
    unfold rd64; rw [hfr11seg (sp - 8) 8 (by omega)]; exact hslot8_10
  obtain ⟨e, hb11r0⟩ := hex0
  obtain ⟨he0, hat11, hd1, ha1⟩ : e ≠ 0 ∧ AnnAt b11 e t h'.d h'.a ∧
      p ≤ e ∧ e + 12 ≤ p + 12 * wd.ann.length := by
    cases hpost e hb11r0 with
    | inl hb => rw [hb.2] at hlk; simp at hlk
    | inr hb =>
      obtain ⟨d1, a1, hne0, hat, hlk1⟩ := hb
      have hhh : (⟨d1, a1⟩ : Height) = h' := Option.some.inj (hlk1.symm.trans hlk)
      have hdeq : d1 = h'.d := congrArg Height.d hhh
      have haeq : a1 = h'.a := congrArg Height.a hhh
      obtain ⟨hl1, hl2⟩ := hloc11 e hb11r0 hne0
      exact ⟨hne0, by rwa [hdeq, haeq] at hat, hl1, hl2⟩
  have hp24 : ((sp - 24) + 8) % W = sp - 16 := by
    rw [show sp - 24 + 8 = sp - 16 by omega]; exact Nat.mod_eq_of_lt (by omega)
  have hp16 : ((sp - 16) + 8) % W = sp - 8 := by
    rw [show sp - 16 + 8 = sp - 8 by omega]; exact Nat.mod_eq_of_lt (by omega)
  -- push rax
  have hroom11 : StackRoom b11 1 :=
    StackRoom_shift hroom hr4 hb11r4 hw11 ⟨2, rfl, by omega⟩ hsp48 hspW
  obtain ⟨m12, hm12s⟩ := Option.isSome_iff_exists.mp (hroom11 (sp - 16) hb11r4 0 (by decide) e)
  have hsp24 : ((sp - 16) + W - 8) % W = sp - 24 := by
    rw [show (sp - 16) + W - 8 = (sp - 16) + W - 8 * 1 by omega, slot_sub (by omega) (by omega)]
    omega
  have hm12 : wr64 b11.mem (sp - 24) e = some m12 := by
    have e' : (sp - 16 + W - 8 * (0 + 1)) % W = sp - 24 := slot_sub (by omega) (by omega)
    rw [e'] at hm12s; exact hm12s
  have hm12p : wr64 b11.mem (((sp - 16) + W - 8) % W) e = some m12 := by rw [hsp24]; exact hm12
  have e12 : X86Int.step im b11 = .ok { b11.set 4 (some (((sp - 16) + W - 8) % W))
      with mem := m12, rip := b11.rip + 1 } :=
    step_push (by rw [hb11rip]; exact oc.av12) hb11r0 hb11r4 hm12p
  obtain ⟨b12, hb12⟩ : ∃ u : X86Int.St, u =
    { b11.set 4 (some (((sp - 16) + W - 8) % W)) with mem := m12, rip := b11.rip + 1 } := ⟨_, rfl⟩
  have hb12rip : b12.rip = aAnnval + 43 := by rw [hb12]; show b11.rip + 1 = aAnnval + 43; rw [hb11rip]
  have hb12r4 : b12.reg 4 = some (sp - 24) := by
    rw [hb12]; show (b11.set 4 _).reg 4 = some (sp - 24); rw [reg_set_self (by omega), hsp24]
  have gb12 : ∀ q, q ≠ 4 → b12.reg q = b11.reg q := by
    intro q hq; rw [hb12]; show (b11.set 4 _).reg q = b11.reg q; rw [reg_set_other hq]
  have hb12mem : b12.mem = m12 := by rw [hb12]
  have hfr12 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x → b12.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; rw [hb12mem, rdN_wr64_other hm12 (by omega)]; exact hfr11 x n hx
  have hw12 : wshape b12.mem = wshape s.mem := by
    rw [hb12mem]; exact (wrN_wshape _ _ _ hm12).trans hw11
  have hlen12 : b12.regs.length = s.regs.length := by
    rw [hb12]; show (b11.set 4 _).regs.length = s.regs.length; exact List.length_set.trans hlen11s
  have hslot24_12 : rd64 b12.mem (sp - 24) = some e := by
    have h1 := rd64_wr64 hm12
    rw [Nat.mod_eq_of_lt (by omega)] at h1; rw [hb12mem]; exact h1
  have hslot16_12 : rd64 b12.mem (sp - 16) =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [hb12mem, rd64_wr64_other hm12 (by omega)]; exact hslot16_11
  have hslot8_12 : rd64 b12.mem (sp - 8) = some (rest'.length + 1) := by
    rw [hb12mem, rd64_wr64_other hm12 (by omega)]; exact hslot8_11
  have hat12 : AnnAt b12 e t h'.d h'.a :=
    ⟨by rw [hb12mem, rdN_wr64_other hm12 (by omega)]; exact hat11.1,
     by rw [hb12mem, rdN_wr64_other hm12 (by omega)]; exact hat11.2.1,
     by rw [hb12mem, rdN_wr64_other hm12 (by omega)]; exact hat11.2.2⟩
  -- mov rdx, rax ; add rdx, 4 ; call L32 -> first d = h'.d
  have hb12r0 : b12.reg 0 = some e := by rw [gb12 0 (by decide)]; exact hb11r0
  have e13 : X86Int.step im b12 = .ok { b12.set 2 (some e) with rip := b12.rip + 3 } :=
    step_mov (by rw [hb12rip]; exact oc.av13) hb12r0
  obtain ⟨b13, hb13⟩ : ∃ u : X86Int.St, u = { b12.set 2 (some e) with rip := b12.rip + 3 } := ⟨_, rfl⟩
  have hb13rip : b13.rip = aAnnval + 46 := by rw [hb13]; show b12.rip + 3 = aAnnval + 46; rw [hb12rip]
  have gb13 : ∀ q, q ≠ 2 → b13.reg q = b12.reg q := by
    intro q hq; rw [hb13]; show (b12.set 2 _).reg q = b12.reg q; rw [reg_set_other hq]
  have hb13r2 : b13.reg 2 = some e := by
    rw [hb13]; show (b12.set 2 _).reg 2 = some e; rw [reg_set_self (by omega)]
  have hb13mem : b13.mem = b12.mem := by rw [hb13]; exact rfl
  have e14 : X86Int.step im b13 = .ok { b13.set 2 (some (aluF .add e (4 % W)).1) with
      fl := (aluF .add e (4 % W)).2, rip := b13.rip + 4 } :=
    step_aluI_add (by rw [hb13rip]; exact oc.av14) hb13r2
  obtain ⟨b14, hb14⟩ : ∃ u : X86Int.St, u = { b13.set 2 (some (aluF .add e (4 % W)).1) with
      fl := (aluF .add e (4 % W)).2, rip := b13.rip + 4 } := ⟨_, rfl⟩
  have hb14rip : b14.rip = aAnnval + 50 := by rw [hb14]; show b13.rip + 4 = aAnnval + 50; rw [hb13rip]
  have hlen13 : b13.regs.length = s.regs.length := by
    rw [hb13]; show (b12.set 2 _).regs.length = s.regs.length; exact List.length_set.trans hlen12
  have hb14r2 : b14.reg 2 = some (e + 4) := by
    rw [hb14]; show (b13.set 2 _).reg 2 = some (e + 4)
    rw [reg_set_self (by omega), hadd e 4 (by decide) (by omega)]
  have gb14 : ∀ q, q ≠ 2 → b14.reg q = b13.reg q := by
    intro q hq; rw [hb14]; show (b13.set 2 _).reg q = b13.reg q; rw [reg_set_other hq]
  have hb14mem : b14.mem = b12.mem := by rw [hb14]; exact hb13mem
  have hb14r4 : b14.reg 4 = some (sp - 24) := by
    rw [gb14 4 (by decide), gb13 4 (by decide)]; exact hb12r4
  have hfr14 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x → b14.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; rw [hb14mem]; exact hfr12 x n hx
  have hw14 : wshape b14.mem = wshape s.mem := by rw [hb14mem]; exact hw12
  have hlen14 : b14.regs.length = s.regs.length := by
    rw [hb14]; show (b13.set 2 _).regs.length = s.regs.length; exact List.length_set.trans hlen13
  obtain ⟨k15, b15, hk15, hrun15, hb15rip, hb15r0, hb15r4, gb15, hlen15, hfr15seg, hw15seg⟩ :=
    CallL32B ac.l32 ac.annl.dret oc.av15 (by omega) (by omega) (by omega)
      (by decide) (by omega) b14 hb14rip hb14r2 hb14r4
      (by rw [hb14mem]; exact hat12.2.1)
      (StackRoom_shift hroom hr4 hb14r4 hw14 ⟨3, rfl, by omega⟩ hsp48 hspW)
  have hlen15s : b15.regs.length = s.regs.length := hlen15.trans hlen14
  have hfr15 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x → b15.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; exact (hfr15seg x n (by omega)).trans (hfr14 x n hx)
  have hw15 : wshape b15.mem = wshape s.mem := hw15seg.trans hw14
  have hslot24_15 : rd64 b15.mem (sp - 24) = some e := by
    unfold rd64; rw [hfr15seg (sp - 24) 8 (by omega), hb14mem]; exact hslot24_12
  have hslot16_15 : rd64 b15.mem (sp - 16) =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    unfold rd64; rw [hfr15seg (sp - 16) 8 (by omega), hb14mem]; exact hslot16_12
  have hslot8_15 : rd64 b15.mem (sp - 8) = some (rest'.length + 1) := by
    unfold rd64; rw [hfr15seg (sp - 8) 8 (by omega), hb14mem]; exact hslot8_12
  have hat15 : AnnAt b15 e t h'.d h'.a :=
    ⟨by rw [hfr15seg e 4 (by omega), hb14mem]; exact hat12.1,
     by rw [hfr15seg (e + 4) 4 (by omega), hb14mem]; exact hat12.2.1,
     by rw [hfr15seg (e + 8) 4 (by omega), hb14mem]; exact hat12.2.2⟩
  -- mov r8, rax ; pop rdx ; add rdx, 8 ; call L32 -> first a = h'.a
  have e16 : X86Int.step im b15 = .ok { b15.set 8 (some h'.d) with rip := b15.rip + 3 } :=
    step_mov (by rw [hb15rip]; exact oc.av16) hb15r0
  obtain ⟨b16, hb16⟩ : ∃ u : X86Int.St, u = { b15.set 8 (some h'.d) with rip := b15.rip + 3 } := ⟨_, rfl⟩
  have hb16rip : b16.rip = aAnnval + 58 := by rw [hb16]; show b15.rip + 3 = aAnnval + 58; rw [hb15rip]
  have hb16r8 : b16.reg 8 = some h'.d := by
    rw [hb16]; show (b15.set 8 _).reg 8 = some h'.d; rw [reg_set_self (by omega)]
  have gb16 : ∀ q, q ≠ 8 → b16.reg q = b15.reg q := by
    intro q hq; rw [hb16]; show (b15.set 8 _).reg q = b15.reg q; rw [reg_set_other hq]
  have hb16mem : b16.mem = b15.mem := by rw [hb16]; exact rfl
  have hb16r4 : b16.reg 4 = some (sp - 24) := by rw [gb16 4 (by decide)]; exact hb15r4
  have hlen16 : b16.regs.length = s.regs.length := by
    rw [hb16]; show (b15.set 8 _).regs.length = s.regs.length; exact List.length_set.trans hlen15s
  have e17 : X86Int.step im b16 = .ok
      { (b16.set 4 (some ((sp - 24 + 8) % W))).set 2 (some e) with rip := b16.rip + 1 } :=
    step_pop (by rw [hb16rip]; exact oc.av17) hb16r4 (by rw [hb16mem]; exact hslot24_15)
  obtain ⟨b17, hb17⟩ : ∃ u : X86Int.St, u =
    { (b16.set 4 (some ((sp - 24 + 8) % W))).set 2 (some e) with rip := b16.rip + 1 } := ⟨_, rfl⟩
  have hb17rip : b17.rip = aAnnval + 59 := by rw [hb17]; show b16.rip + 1 = aAnnval + 59; rw [hb16rip]
  have hlen16a : (b16.set 4 (some ((sp - 24 + 8) % W))).regs.length = s.regs.length :=
    List.length_set.trans hlen16
  have hb17r4 : b17.reg 4 = some (sp - 16) := by
    rw [hb17]; show ((b16.set 4 _).set 2 _).reg 4 = some (sp - 16)
    rw [reg_set_other (by decide), reg_set_self (by omega), hp24]
  have hb17r2 : b17.reg 2 = some e := by
    rw [hb17]; show ((b16.set 4 _).set 2 _).reg 2 = some e; rw [reg_set_self (by omega)]
  have gb17 : ∀ q, q ≠ 2 → q ≠ 4 → b17.reg q = b16.reg q := by
    intro q h2 h4; rw [hb17]; show ((b16.set 4 _).set 2 _).reg q = b16.reg q
    rw [reg_set_other h2, reg_set_other h4]
  have hb17mem : b17.mem = b15.mem := by rw [hb17]; exact hb16mem
  have hlen17 : b17.regs.length = s.regs.length := by
    rw [hb17]; exact List.length_set.trans hlen16a
  have hb17r8 : b17.reg 8 = some h'.d := by
    rw [gb17 8 (by decide) (by decide)]; exact hb16r8
  have e18 : X86Int.step im b17 = .ok { b17.set 2 (some (aluF .add e (8 % W)).1) with
      fl := (aluF .add e (8 % W)).2, rip := b17.rip + 4 } :=
    step_aluI_add (by rw [hb17rip]; exact oc.av18) hb17r2
  obtain ⟨b18, hb18⟩ : ∃ u : X86Int.St, u = { b17.set 2 (some (aluF .add e (8 % W)).1) with
      fl := (aluF .add e (8 % W)).2, rip := b17.rip + 4 } := ⟨_, rfl⟩
  have hb18rip : b18.rip = aAnnval + 63 := by rw [hb18]; show b17.rip + 4 = aAnnval + 63; rw [hb17rip]
  have hb18r2 : b18.reg 2 = some (e + 8) := by
    rw [hb18]; show (b17.set 2 _).reg 2 = some (e + 8)
    rw [reg_set_self (by omega), hadd e 8 (by decide) (by omega)]
  have gb18 : ∀ q, q ≠ 2 → b18.reg q = b17.reg q := by
    intro q hq; rw [hb18]; show (b17.set 2 _).reg q = b17.reg q; rw [reg_set_other hq]
  have hb18mem : b18.mem = b15.mem := by rw [hb18]; exact hb17mem
  have hb18r4 : b18.reg 4 = some (sp - 16) := by rw [gb18 4 (by decide)]; exact hb17r4
  have hfr18 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x → b18.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; rw [hb18mem]; exact hfr15 x n hx
  have hw18 : wshape b18.mem = wshape s.mem := by rw [hb18mem]; exact hw15
  have hlen18 : b18.regs.length = s.regs.length := by
    rw [hb18]; show (b17.set 2 _).regs.length = s.regs.length; exact List.length_set.trans hlen17
  have hb18r8 : b18.reg 8 = some h'.d := by rw [gb18 8 (by decide)]; exact hb17r8
  obtain ⟨k19, b19, hk19, hrun19, hb19rip, hb19r0, hb19r4, gb19, hlen19, hfr19seg, hw19seg⟩ :=
    CallL32B ac.l32 ac.annl.dret oc.av19 (by omega) (by omega) (by omega)
      (by decide) (by omega) b18 hb18rip hb18r2 hb18r4
      (by rw [hb18mem]; exact hat15.2.2)
      (StackRoom_shift hroom hr4 hb18r4 hw18 ⟨2, rfl, by omega⟩ hsp48 hspW)
  have hlen19s : b19.regs.length = s.regs.length := hlen19.trans hlen18
  have hfr19 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x → b19.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; exact (hfr19seg x n (by omega)).trans (hfr18 x n hx)
  have hw19 : wshape b19.mem = wshape s.mem := hw19seg.trans hw18
  have hb19r8 : b19.reg 8 = some h'.d := by rw [gb19 8 (by decide) (by decide)]; exact hb18r8
  have hslot16_19 : rd64 b19.mem (sp - 16) =
      some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    unfold rd64; rw [hfr19seg (sp - 16) 8 (by omega), hb18mem]; exact hslot16_15
  have hslot8_19 : rd64 b19.mem (sp - 8) = some (rest'.length + 1) := by
    unfold rd64; rw [hfr19seg (sp - 8) 8 (by omega), hb18mem]; exact hslot8_15
  -- mov r9, rax ; pop rsi ; mov rdx, rsi ; add rdx, 4 ; call L32 -> this d = h.d
  have e20 : X86Int.step im b19 = .ok { b19.set 9 (some h'.a) with rip := b19.rip + 3 } :=
    step_mov (by rw [hb19rip]; exact oc.av20) hb19r0
  obtain ⟨b20, hb20⟩ : ∃ u : X86Int.St, u = { b19.set 9 (some h'.a) with rip := b19.rip + 3 } := ⟨_, rfl⟩
  have hb20rip : b20.rip = aAnnval + 71 := by rw [hb20]; show b19.rip + 3 = aAnnval + 71; rw [hb19rip]
  have hb20r9 : b20.reg 9 = some h'.a := by
    rw [hb20]; show (b19.set 9 _).reg 9 = some h'.a; rw [reg_set_self (by omega)]
  have gb20 : ∀ q, q ≠ 9 → b20.reg q = b19.reg q := by
    intro q hq; rw [hb20]; show (b19.set 9 _).reg q = b19.reg q; rw [reg_set_other hq]
  have hb20mem : b20.mem = b19.mem := by rw [hb20]; exact rfl
  have hb20r4 : b20.reg 4 = some (sp - 16) := by rw [gb20 4 (by decide)]; exact hb19r4
  have hb20r8 : b20.reg 8 = some h'.d := by rw [gb20 8 (by decide)]; exact hb19r8
  have hlen20 : b20.regs.length = s.regs.length := by
    rw [hb20]; show (b19.set 9 _).regs.length = s.regs.length; exact List.length_set.trans hlen19s
  have e21 : X86Int.step im b20 = .ok
      { (b20.set 4 (some ((sp - 16 + 8) % W))).set 6
          (some (p + 12 * (wd.ann.length - (rest'.length + 1)))) with rip := b20.rip + 1 } :=
    step_pop (by rw [hb20rip]; exact oc.av21) hb20r4 (by rw [hb20mem]; exact hslot16_19)
  obtain ⟨b21, hb21⟩ : ∃ u : X86Int.St, u =
    { (b20.set 4 (some ((sp - 16 + 8) % W))).set 6
        (some (p + 12 * (wd.ann.length - (rest'.length + 1)))) with rip := b20.rip + 1 } := ⟨_, rfl⟩
  have hb21rip : b21.rip = aAnnval + 72 := by rw [hb21]; show b20.rip + 1 = aAnnval + 72; rw [hb20rip]
  have hlen20a : (b20.set 4 (some ((sp - 16 + 8) % W))).regs.length = s.regs.length :=
    List.length_set.trans hlen20
  have hb21r4 : b21.reg 4 = some (sp - 8) := by
    rw [hb21]; show ((b20.set 4 _).set 6 _).reg 4 = some (sp - 8)
    rw [reg_set_other (by decide), reg_set_self (by omega), hp16]
  have hb21r6 : b21.reg 6 = some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [hb21]; show ((b20.set 4 _).set 6 _).reg 6 = _; rw [reg_set_self (by omega)]
  have gb21 : ∀ q, q ≠ 6 → q ≠ 4 → b21.reg q = b20.reg q := by
    intro q h6 h4; rw [hb21]; show ((b20.set 4 _).set 6 _).reg q = b20.reg q
    rw [reg_set_other h6, reg_set_other h4]
  have hb21mem : b21.mem = b19.mem := by rw [hb21]; exact hb20mem
  have hlen21 : b21.regs.length = s.regs.length := by
    rw [hb21]; exact List.length_set.trans hlen20a
  have hb21r8 : b21.reg 8 = some h'.d := by rw [gb21 8 (by decide) (by decide)]; exact hb20r8
  have hb21r9 : b21.reg 9 = some h'.a := by rw [gb21 9 (by decide) (by decide)]; exact hb20r9
  have hfr21 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x → b21.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; rw [hb21mem]; exact hfr19 x n hx
  have hw21 : wshape b21.mem = wshape s.mem := by rw [hb21mem]; exact hw19
  have hslot8_21 : rd64 b21.mem (sp - 8) = some (rest'.length + 1) := by
    rw [hb21mem]; exact hslot8_19
  have e22 : X86Int.step im b21 = .ok { b21.set 2
      (some (p + 12 * (wd.ann.length - (rest'.length + 1)))) with rip := b21.rip + 3 } :=
    step_mov (by rw [hb21rip]; exact oc.av22) hb21r6
  obtain ⟨b22, hb22⟩ : ∃ u : X86Int.St, u = { b21.set 2
      (some (p + 12 * (wd.ann.length - (rest'.length + 1)))) with rip := b21.rip + 3 } := ⟨_, rfl⟩
  have hb22rip : b22.rip = aAnnval + 75 := by rw [hb22]; show b21.rip + 3 = aAnnval + 75; rw [hb21rip]
  have gb22 : ∀ q, q ≠ 2 → b22.reg q = b21.reg q := by
    intro q hq; rw [hb22]; show (b21.set 2 _).reg q = b21.reg q; rw [reg_set_other hq]
  have hb22r2 : b22.reg 2 = some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [hb22]; show (b21.set 2 _).reg 2 = _; rw [reg_set_self (by omega)]
  have hb22mem : b22.mem = b21.mem := by rw [hb22]; exact rfl
  have hlen22 : b22.regs.length = s.regs.length := by
    rw [hb22]; show (b21.set 2 _).regs.length = s.regs.length; exact List.length_set.trans hlen21
  have e23 : X86Int.step im b22 = .ok { b22.set 2
      (some (aluF .add (p + 12 * (wd.ann.length - (rest'.length + 1))) (4 % W)).1) with
      fl := (aluF .add (p + 12 * (wd.ann.length - (rest'.length + 1))) (4 % W)).2,
      rip := b22.rip + 4 } :=
    step_aluI_add (by rw [hb22rip]; exact oc.av23) hb22r2
  obtain ⟨b23, hb23⟩ : ∃ u : X86Int.St, u = { b22.set 2
      (some (aluF .add (p + 12 * (wd.ann.length - (rest'.length + 1))) (4 % W)).1) with
      fl := (aluF .add (p + 12 * (wd.ann.length - (rest'.length + 1))) (4 % W)).2,
      rip := b22.rip + 4 } := ⟨_, rfl⟩
  have hb23rip : b23.rip = aAnnval + 79 := by rw [hb23]; show b22.rip + 4 = aAnnval + 79; rw [hb22rip]
  have hb23r2 : b23.reg 2 = some (p + 12 * (wd.ann.length - (rest'.length + 1)) + 4) := by
    rw [hb23]; show (b22.set 2 _).reg 2 = _; rw [reg_set_self (by omega), hadd _ 4 (by decide) (by omega)]
  have gb23 : ∀ q, q ≠ 2 → b23.reg q = b22.reg q := by
    intro q hq; rw [hb23]; show (b22.set 2 _).reg q = b22.reg q; rw [reg_set_other hq]
  have hb23mem : b23.mem = b21.mem := by rw [hb23]; exact hb22mem
  have hb23r4 : b23.reg 4 = some (sp - 8) := by
    rw [gb23 4 (by decide), gb22 4 (by decide)]; exact hb21r4
  have hb23r8 : b23.reg 8 = some h'.d := by
    rw [gb23 8 (by decide), gb22 8 (by decide)]; exact hb21r8
  have hb23r9 : b23.reg 9 = some h'.a := by
    rw [gb23 9 (by decide), gb22 9 (by decide)]; exact hb21r9
  have hb23r6 : b23.reg 6 = some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [gb23 6 (by decide), gb22 6 (by decide)]; exact hb21r6
  have hfr23 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x → b23.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; rw [hb23mem]; exact hfr21 x n hx
  have hw23 : wshape b23.mem = wshape s.mem := by rw [hb23mem]; exact hw21
  have hlen23 : b23.regs.length = s.regs.length := by
    rw [hb23]; show (b22.set 2 _).regs.length = s.regs.length; exact List.length_set.trans hlen22
  obtain ⟨k24, b24, hk24, hrun24, hb24rip, hb24r0, hb24r4, gb24, hlen24, hfr24seg, hw24seg⟩ :=
    CallL32B ac.l32 ac.annl.dret oc.av24 (by omega) (by omega) (by omega)
      (by decide) (by omega) b23 hb23rip hb23r2 hb23r4
      (by rw [hb23mem]; exact (atOf b21 hfr21).2.1)
      (StackRoom_shift hroom hr4 hb23r4 hw23 ⟨1, by omega, by omega⟩ hsp48 hspW)
  have hlen24s : b24.regs.length = s.regs.length := hlen24.trans hlen23
  have hfr24 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x → b24.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; exact (hfr24seg x n (by omega)).trans (hfr23 x n hx)
  have hw24 : wshape b24.mem = wshape s.mem := hw24seg.trans hw23
  have hb24r8 : b24.reg 8 = some h'.d := by rw [gb24 8 (by decide) (by decide)]; exact hb23r8
  have hb24r9 : b24.reg 9 = some h'.a := by rw [gb24 9 (by decide) (by decide)]; exact hb23r9
  have hb24r6 : b24.reg 6 = some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
    rw [gb24 6 (by decide) (by decide)]; exact hb23r6
  -- cmp rax, r8
  have e25 : X86Int.step im b24 =
      .ok { b24 with fl := (aluF .sub h.d h'.d).2, rip := b24.rip + 3 } :=
    step_cmp (by rw [hb24rip]; exact oc.av25) hb24r0 hb24r8
  obtain ⟨b25, hb25⟩ : ∃ u : X86Int.St, u =
    { b24 with fl := (aluF .sub h.d h'.d).2, rip := b24.rip + 3 } := ⟨_, rfl⟩
  have rr10 : run im 1 (Out.ok b9) = Out.ok b10 := by rw [run_succ_ok im 0 b9, e10, ← hb10, run_zero]
  have rr12 : run im 1 (Out.ok b11) = Out.ok b12 := by rw [run_succ_ok im 0 b11, e12, ← hb12, run_zero]
  have rr13 : run im 1 (Out.ok b12) = Out.ok b13 := by rw [run_succ_ok im 0 b12, e13, ← hb13, run_zero]
  have rr14 : run im 1 (Out.ok b13) = Out.ok b14 := by rw [run_succ_ok im 0 b13, e14, ← hb14, run_zero]
  have rr16 : run im 1 (Out.ok b15) = Out.ok b16 := by rw [run_succ_ok im 0 b15, e16, ← hb16, run_zero]
  have rr17 : run im 1 (Out.ok b16) = Out.ok b17 := by rw [run_succ_ok im 0 b16, e17, ← hb17, run_zero]
  have rr18 : run im 1 (Out.ok b17) = Out.ok b18 := by rw [run_succ_ok im 0 b17, e18, ← hb18, run_zero]
  have rr20 : run im 1 (Out.ok b19) = Out.ok b20 := by rw [run_succ_ok im 0 b19, e20, ← hb20, run_zero]
  have rr21 : run im 1 (Out.ok b20) = Out.ok b21 := by rw [run_succ_ok im 0 b20, e21, ← hb21, run_zero]
  have rr22 : run im 1 (Out.ok b21) = Out.ok b22 := by rw [run_succ_ok im 0 b21, e22, ← hb22, run_zero]
  have rr23 : run im 1 (Out.ok b22) = Out.ok b23 := by rw [run_succ_ok im 0 b22, e23, ← hb23, run_zero]
  have rr25 : run im 1 (Out.ok b24) = Out.ok b25 := by rw [run_succ_ok im 0 b24, e25, ← hb25, run_zero]
  have hr11 : run im (1 + k11) (Out.ok b9) = Out.ok b11 := by
    rw [run_add im 1 k11, rr10, hrun11]
  have hr15 : run im (1 + k11 + 3 + k15) (Out.ok b9) = Out.ok b15 := by
    rw [run_add im (1 + k11 + 3) k15, run_add im (1 + k11) 3, hr11,
      run_add im 1 2, rr12, run_add im 1 1, rr13, rr14, hrun15]
  have hr19 : run im (1 + k11 + 3 + k15 + 3 + k19) (Out.ok b9) = Out.ok b19 := by
    rw [run_add im (1 + k11 + 3 + k15 + 3) k19, run_add im (1 + k11 + 3 + k15) 3, hr15,
      run_add im 1 2, rr16, run_add im 1 1, rr17, rr18, hrun19]
  have hr24 : run im (1 + k11 + 3 + k15 + 3 + k19 + 4 + k24) (Out.ok b9) = Out.ok b24 := by
    rw [run_add im (1 + k11 + 3 + k15 + 3 + k19 + 4) k24, run_add im (1 + k11 + 3 + k15 + 3 + k19) 4, hr19,
      run_add im 1 3, rr20, run_add im 1 2, rr21, run_add im 1 1, rr22, rr23, hrun24]
  have hrun : run im (1 + k11 + 3 + k15 + 3 + k19 + 4 + k24 + 1) (Out.ok b9) = Out.ok b25 := by
    rw [run_add im (1 + k11 + 3 + k15 + 3 + k19 + 4 + k24) 1, hr24, rr25]
  refine ⟨k11, k15, k19, k24, hk11, hk15, hk19, hk24, b25, hrun,
    by rw [hb25]; show b24.rip + 3 = aAnnval + 87; rw [hb24rip],
    by rw [hb25],
    by rw [hb25]; exact hb24r4, by rw [hb25]; exact hb24r6, by rw [hb25]; exact hb24r9,
    by rw [hb25]; exact hfr24, by rw [hb25]; exact hw24, by rw [hb25]; exact hlen24s⟩

/-- One `annval` refusal: a duplicate/conflicting annotation entry (`lookup t ≠ some h`)
branches to `avDupD` or `avDupA` and exits `1`. -/
theorem annval_dup_exit1B {im : Nat → Option (Ins × Nat)}
    (ac : AllCode im) (oc : OuterCode im) (rc : RefuseCode im) (cr : ChkRefuseCode im)
    {wd : Sporos.Word} {w p sp : Nat} (st : AnnStatic wd w p sp)
    {t : Nat} {h h' : Height} {rest' : List (Nat × Height)}
    (hlen : rest'.length + 1 ≤ wd.ann.length)
    (htlt : t < wd.code.length) (hmem : (t, h) ∈ wd.ann)
    (hlk : wd.ann.lookup t = some h') (hne : h' ≠ h) :
    ∀ s, AnnSt wd w p sp ((t, h) :: rest') s →
      AnnAt s (p + 12 * (wd.ann.length - (rest'.length + 1))) t h.d h.a →
      t < W → RegsOk s →
      ∃ m, m ≤ 60 + 20 * (wd.ann.length + 1) ∧ run im m (Out.ok s) = Out.exit 1 := by
  intro s hs hcur htW hregs
  have hw56 := st.1
  have hp0 := st.2.1
  have hpb := st.2.2.1
  have hsp48 := st.2.2.2.1
  have hspW := st.2.2.2.2.1
  have hhW := st.2.2.2.2.2.2.1
  have hdisjp := st.2.2.2.2.2.2.2.2
  have hr4 := hs.2.2.2.2.1
  have h12len := hs.2.2.2.2.2.2.2.2.2.1
  have hroom := hs.2.2.2.2.2.2.2.2.2.2
  obtain ⟨hdW, haW⟩ : h.d < W ∧ h.a < W := hhW (t, h) hmem
  obtain ⟨h'dW, h'aW⟩ : h'.d < W ∧ h'.a < W := hhW (t, h') (lookup_mem hlk)
  have hcurb : p + 12 * (wd.ann.length - (rest'.length + 1)) + 12 ≤
      p + 12 * wd.ann.length := by
    have : wd.ann.length - (rest'.length + 1) + 1 ≤ wd.ann.length := by omega
    omega
  obtain ⟨k6, hk6, b9, hrun9, hb9rip, hb9fl, -, -, hb9r12, hb9r7, hb9r4, hb9r1, hb9r6,
    -, hfr9, hwb9, hlen9, hslot16_9, hslot8_9⟩ :=
    annval_step_head ac oc st hlen s hs hcur
  obtain ⟨k11, k15, k19, k24, hk11, hk15, hk19, hk24, b25, hrun25, hb25rip, hb25fl,
    hb25r4, hb25r6, hb25r9, hfr25, hw25, hlen25⟩ :=
    annval_dup_mid ac oc st hlen htlt htW hlk hs hcur hb9rip hb9fl hb9r12 hb9r7 hb9r4
      hb9r1 hb9r6 hfr9 hwb9 hlen9 hslot16_9 hslot8_9
  have hrun_s_25 : run im ((5 + k6 + 3) + (1 + k11 + 3 + k15 + 3 + k19 + 4 + k24 + 1))
      (Out.ok s) = Out.ok b25 := by
    rw [run_add im (5 + k6 + 3), hrun9]; exact hrun25
  by_cases hd : h.d = h'.d
  · have ha : h.a ≠ h'.a := by
      intro heqa
      apply hne
      cases h; cases h'; simp at hd heqa; simp [hd, heqa]
    have hb25fl' : X86Int.cond b25.fl 4 = some true := by
      have hzf : b25.fl.zf = some true := by rw [hb25fl]; exact aluF_sub_zf_eq hdW h'dW hd
      exact cond_cc4 hzf
    have e26 : X86Int.step im b25 = .ok { b25 with rip := aAnnval + 94 } := by
      have hh := step_jcc (im := im) (s := b25) (by rw [hb25rip]; exact oc.av26) hb25fl'
      simpa using hh
    obtain ⟨b26, hb26⟩ : ∃ u : X86Int.St, u = { b25 with rip := aAnnval + 94 } := ⟨_, rfl⟩
    have hb26rip : b26.rip = aAnnval + 94 := by rw [hb26]
    have hb26r6 : b26.reg 6 = some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
      rw [hb26]; exact hb25r6
    have hb26r4 : b26.reg 4 = some (sp - 8) := by rw [hb26]; exact hb25r4
    have hb26r9 : b26.reg 9 = some h'.a := by rw [hb26]; exact hb25r9
    have hb26mem : b26.mem = b25.mem := by rw [hb26]
    have hlen26 : b26.regs.length = s.regs.length := by rw [hb26]; exact hlen25
    -- mov rdx, rsi ; add rdx, 8 ; call L32 -> this a = h.a
    have e27 : X86Int.step im b26 = .ok { b26.set 2
        (some (p + 12 * (wd.ann.length - (rest'.length + 1)))) with rip := b26.rip + 3 } :=
      step_mov (by rw [hb26rip]; exact oc.av27) hb26r6
    obtain ⟨b27, hb27⟩ : ∃ u : X86Int.St, u = { b26.set 2
        (some (p + 12 * (wd.ann.length - (rest'.length + 1)))) with rip := b26.rip + 3 } := ⟨_, rfl⟩
    have hb27rip : b27.rip = aAnnval + 97 := by rw [hb27]; show b26.rip + 3 = aAnnval + 97; rw [hb26rip]
    have gb27 : ∀ q, q ≠ 2 → b27.reg q = b26.reg q := by
      intro q hq; rw [hb27]; show (b26.set 2 _).reg q = b26.reg q; rw [reg_set_other hq]
    have hb27r2 : b27.reg 2 = some (p + 12 * (wd.ann.length - (rest'.length + 1))) := by
      rw [hb27]; show (b26.set 2 _).reg 2 = _; rw [reg_set_self (by omega)]
    have hb27mem : b27.mem = b26.mem := by rw [hb27]; exact rfl
    have hlen27 : b27.regs.length = s.regs.length := by
      rw [hb27]; show (b26.set 2 _).regs.length = s.regs.length; exact List.length_set.trans hlen26
    have hadd8 : (aluF .add (p + 12 * (wd.ann.length - (rest'.length + 1))) (8 % W)).1 =
        p + 12 * (wd.ann.length - (rest'.length + 1)) + 8 := by
      rw [aluF_add_fst, show (8 : Nat) % W = 8 from Nat.mod_eq_of_lt (by decide)]
      exact Nat.mod_eq_of_lt (by omega)
    have e28 : X86Int.step im b27 = .ok { b27.set 2
        (some (aluF .add (p + 12 * (wd.ann.length - (rest'.length + 1))) (8 % W)).1) with
        fl := (aluF .add (p + 12 * (wd.ann.length - (rest'.length + 1))) (8 % W)).2,
        rip := b27.rip + 4 } :=
      step_aluI_add (by rw [hb27rip]; exact oc.av28) hb27r2
    obtain ⟨b28, hb28⟩ : ∃ u : X86Int.St, u = { b27.set 2
        (some (aluF .add (p + 12 * (wd.ann.length - (rest'.length + 1))) (8 % W)).1) with
        fl := (aluF .add (p + 12 * (wd.ann.length - (rest'.length + 1))) (8 % W)).2,
        rip := b27.rip + 4 } := ⟨_, rfl⟩
    have hb28rip : b28.rip = aAnnval + 101 := by rw [hb28]; show b27.rip + 4 = aAnnval + 101; rw [hb27rip]
    have hb28r2 : b28.reg 2 = some (p + 12 * (wd.ann.length - (rest'.length + 1)) + 8) := by
      rw [hb28]; show (b27.set 2 _).reg 2 = _; rw [reg_set_self (by omega), hadd8]
    have gb28 : ∀ q, q ≠ 2 → b28.reg q = b27.reg q := by
      intro q hq; rw [hb28]; show (b27.set 2 _).reg q = b27.reg q; rw [reg_set_other hq]
    have hb28mem : b28.mem = b25.mem := by rw [hb28]; show b27.mem = b25.mem; rw [hb27mem]; exact hb26mem
    have hb28r4 : b28.reg 4 = some (sp - 8) := by
      rw [gb28 4 (by decide), gb27 4 (by decide)]; exact hb26r4
    have hb28r9 : b28.reg 9 = some h'.a := by
      rw [gb28 9 (by decide), gb27 9 (by decide)]; exact hb26r9
    have hw28 : wshape b28.mem = wshape s.mem := by rw [hb28mem]; exact hw25
    have hrd28a : (b28.mem.rdN (p + 12 * (wd.ann.length - (rest'.length + 1)) + 8) 4).map leVal =
        some h.a := by
      rw [hb28mem, hfr25 (p + 12 * (wd.ann.length - (rest'.length + 1)) + 8) 4 (by omega)]
      exact hcur.2.2
    obtain ⟨k29, b29, hk29, hrun29, hb29rip, hb29r0, hb29r4, gb29, -, -, hw29seg⟩ :=
      CallL32B ac.l32 ac.annl.dret oc.av29 (by omega) (by omega) (by omega)
        (by decide) (by omega) b28 hb28rip hb28r2 hb28r4 hrd28a
        (StackRoom_shift hroom hr4 hb28r4 hw28 ⟨1, by omega, by omega⟩ hsp48 hspW)
    have hw29 : wshape b29.mem = wshape s.mem := hw29seg.trans hw28
    have hb29r9 : b29.reg 9 = some h'.a := by rw [gb29 9 (by decide) (by decide)]; exact hb28r9
    -- cmp rax, r9 ; jz not taken -> avDupA
    have e30 : X86Int.step im b29 =
        .ok { b29 with fl := (aluF .sub h.a h'.a).2, rip := b29.rip + 3 } :=
      step_cmp (by rw [hb29rip]; exact oc.av30) hb29r0 hb29r9
    obtain ⟨b30, hb30⟩ : ∃ u : X86Int.St, u =
      { b29 with fl := (aluF .sub h.a h'.a).2, rip := b29.rip + 3 } := ⟨_, rfl⟩
    have hb30rip : b30.rip = aAnnval + 109 := by rw [hb30]; show b29.rip + 3 = aAnnval + 109; rw [hb29rip]
    have hb30fl : X86Int.cond b30.fl 4 = some false := by
      have hzf : b30.fl.zf = some false := by rw [hb30]; exact aluF_sub_zf_ne haW h'aW ha
      exact cond_cc4 hzf
    have e31 : X86Int.step im b30 = .ok { b30 with rip := aAnnval + 111 } := by
      have hh := step_jcc (im := im) (s := b30) (by rw [hb30rip]; exact oc.av31) hb30fl
      simpa [hb30rip] using hh
    obtain ⟨b31, hb31⟩ : ∃ u : X86Int.St, u = { b30 with rip := aAnnval + 111 } := ⟨_, rfl⟩
    have rr26 : run im 1 (Out.ok b25) = Out.ok b26 := by rw [run_succ_ok im 0 b25, e26, ← hb26, run_zero]
    have rr27 : run im 1 (Out.ok b26) = Out.ok b27 := by rw [run_succ_ok im 0 b26, e27, ← hb27, run_zero]
    have rr28 : run im 1 (Out.ok b27) = Out.ok b28 := by rw [run_succ_ok im 0 b27, e28, ← hb28, run_zero]
    have rr30 : run im 1 (Out.ok b29) = Out.ok b30 := by rw [run_succ_ok im 0 b29, e30, ← hb30, run_zero]
    have rr31 : run im 1 (Out.ok b30) = Out.ok b31 := by rw [run_succ_ok im 0 b30, e31, ← hb31, run_zero]
    have hrun28 : run im (((5 + k6 + 3) + (1 + k11 + 3 + k15 + 3 + k19 + 4 + k24 + 1)) + 3)
        (Out.ok s) = Out.ok b28 := by
      rw [run_add im _ 3, hrun_s_25, run_add im 1 2, rr26, run_add im 1 1, rr27, rr28]
    have hrun29 : run im (((5 + k6 + 3) + (1 + k11 + 3 + k15 + 3 + k19 + 4 + k24 + 1)) + 3 + k29)
        (Out.ok s) = Out.ok b29 := by
      rw [run_add im _ k29, hrun28, hrun29]
    have hrun31 : run im (((5 + k6 + 3) + (1 + k11 + 3 + k15 + 3 + k19 + 4 + k24 + 1)) +
        (3 + k29 + 2)) (Out.ok s) = Out.ok b31 := by
      rw [show (((5 + k6 + 3) + (1 + k11 + 3 + k15 + 3 + k19 + 4 + k24 + 1)) + (3 + k29 + 2)) =
          (((5 + k6 + 3) + (1 + k11 + 3 + k15 + 3 + k19 + 4 + k24 + 1)) + 3 + k29) + 2 by omega,
        run_add im _ 2, hrun29, run_add im 1 1, rr30, rr31]
    have hb31rip : b31.rip = aAnnval + 111 := by rw [hb31]
    have hb31r4 : b31.reg 4 = some (sp - 8) := by rw [hb31, hb30]; exact hb29r4
    have hwb31 : wshape b31.mem = wshape s.mem := by rw [hb31, hb30]; exact hw29
    obtain ⟨k32, hk32, hrun32⟩ := jmp_fail1_exit1B rc cr.avDupA (by omega) (by omega) b31
      hb31rip hb31r4 (run_regs_ok hrun31 hregs)
      (StackRoom_shift hroom hr4 hb31r4 hwb31 ⟨1, by omega, by omega⟩ hsp48 hspW)
    have hle := annval_dup_a_le k6 k11 k15 k19 k24 k29 k32 wd.ann.length
      hk6 hk11 hk15 hk19 hk24 hk29 hk32
    refine ⟨(((5 + k6 + 3) + (1 + k11 + 3 + k15 + 3 + k19 + 4 + k24 + 1)) + (3 + k29 + 2)) + k32,
      (by omega), by rw [run_add im _ k32, hrun31]; exact hrun32⟩
  · -- h.d ≠ h'.d: jz not taken -> avDupD
    have hb25fl' : X86Int.cond b25.fl 4 = some false := by
      have hzf : b25.fl.zf = some false := by rw [hb25fl]; exact aluF_sub_zf_ne hdW h'dW hd
      exact cond_cc4 hzf
    have e26 : X86Int.step im b25 = .ok { b25 with rip := aAnnval + 89 } := by
      have hh := step_jcc (im := im) (s := b25) (by rw [hb25rip]; exact oc.av26) hb25fl'
      simpa [hb25rip] using hh
    obtain ⟨b26, hb26⟩ : ∃ u : X86Int.St, u = { b25 with rip := aAnnval + 89 } := ⟨_, rfl⟩
    have r26 : run im 1 (Out.ok b25) = Out.ok b26 := by rw [run_succ_ok im 0 b25, e26, ← hb26, run_zero]
    have hrun26 : run im (((5 + k6 + 3) + (1 + k11 + 3 + k15 + 3 + k19 + 4 + k24 + 1)) + 1)
        (Out.ok s) = Out.ok b26 := by
      rw [run_add im _ 1, hrun_s_25, r26]
    have hb26rip : b26.rip = aAnnval + 89 := by rw [hb26]
    have hb26r4 : b26.reg 4 = some (sp - 8) := by rw [hb26]; exact hb25r4
    have hwb26 : wshape b26.mem = wshape s.mem := by rw [hb26]; exact hw25
    obtain ⟨k27, hk27, hrun27⟩ := jmp_fail1_exit1B rc cr.avDupD (by omega) (by omega) b26
      hb26rip hb26r4 (run_regs_ok hrun26 hregs)
      (StackRoom_shift hroom hr4 hb26r4 hwb26 ⟨1, by omega, by omega⟩ hsp48 hspW)
    have hle := annval_dup_d_le k6 k11 k15 k19 k24 k27 wd.ann.length
      hk6 hk11 hk15 hk19 hk24 hk27
    refine ⟨(((5 + k6 + 3) + (1 + k11 + 3 + k15 + 3 + k19 + 4 + k24 + 1)) + 1) + k27,
      (by omega), by rw [run_add im _ k27, hrun26]; exact hrun27⟩

private theorem annval_refuse_mid_le {rest_len ann_len k1 k2 : Nat}
    (hk1 : k1 ≤ 60 + 20 * (ann_len + 1))
    (hk2 : k2 ≤ (rest_len + 1) * (60 + 20 * (ann_len + 1))) :
    k1 + k2 ≤ (rest_len + 1 + 1) * (60 + 20 * (ann_len + 1)) := by
  have h1 : (rest_len + 1 + 1) * (60 + 20 * (ann_len + 1)) =
      (60 + 20 * (ann_len + 1)) + (rest_len + 1) * (60 + 20 * (ann_len + 1)) := by
    rw [Nat.add_mul, Nat.one_mul, Nat.add_comm]
  omega

/-- Whenever `Sporos.annOk wd = false`, the `annval` loop refuses the word
by exiting `1` within `(rest.length + 1) * (60 + 20 * (wd.ann.length + 1))` steps. -/
theorem annval_refuse_loopB {im : Nat → Option (Ins × Nat)}
    (ac : AllCode im) (oc : OuterCode im) (rc : RefuseCode im) (cr : ChkRefuseCode im)
    {wd : Sporos.Word} {w p sp : Nat} (st : AnnStatic wd w p sp)
    (hann : Sporos.annOk wd = false) :
    ∀ (rest pre : List (Nat × Height)),
      wd.ann = pre ++ rest →
      (∀ x ∈ pre, x.1 < wd.code.length ∧ wd.ann.lookup x.1 = some x.2) →
      ∀ s, AnnSt wd w p sp rest s → RegsOk s →
      ∃ m, m ≤ (rest.length + 1) * (60 + 20 * (wd.ann.length + 1)) ∧
        run im m (Out.ok s) = Out.exit 1 := by
  intro rest
  induction rest with
  | nil =>
    intro pre happ hpre s _ _
    exfalso
    suffices htrue : Sporos.annOk wd = true from by rw [htrue] at hann; cases hann
    unfold Sporos.annOk
    rw [List.all_eq_true]
    intro th hmem
    rw [happ, List.append_nil] at hmem
    obtain ⟨h1, h2⟩ := hpre th hmem
    simp [h1, h2]
  | cons th rest' ih =>
    obtain ⟨t, h⟩ := th
    intro pre happ hpre s hs hregs
    have hsp48 := st.2.2.2.1
    have hspW := st.2.2.2.2.1
    have hncW := st.2.2.2.2.2.1
    have hlay := hs.2.2.2.2.2.2.2.2.1
    have hlenann : wd.ann.length = pre.length + (rest'.length + 1) := by
      rw [happ, List.length_append, List.length_cons]
    have hmem : (t, h) ∈ wd.ann := by
      rw [happ]; exact List.mem_append_right _ List.mem_cons_self
    have hi0 : pre.length < wd.ann.length := by omega
    obtain ⟨hat, htW⟩ := hlay pre.length hi0
    have hget? : wd.ann[pre.length]? = some (t, h) := by
      rw [happ, List.getElem?_append_right (by simp)]; simp
    have hget : wd.ann[pre.length]'hi0 = (t, h) :=
      Option.some.inj ((List.getElem?_eq_getElem hi0).symm.trans hget?)
    have hcur : AnnAt s (p + 12 * (wd.ann.length - (rest'.length + 1))) t h.d h.a := by
      have e : p + 12 * pre.length = p + 12 * (wd.ann.length - (rest'.length + 1)) := by
        congr 1; congr 1; omega
      rw [e, hget] at hat; exact hat
    have htW' : t < W := by rw [hget] at htW; exact htW
    by_cases htlt : t < wd.code.length
    · obtain ⟨h', hlk⟩ := lookup_some_of_mem wd.ann hmem
      by_cases heqh : h' = h
      · have hlk' : wd.ann.lookup t = some h := heqh ▸ hlk
        obtain ⟨k1, s1, hk1, hrun1, hs1, -, -, -, -⟩ :=
          annval_stepB ac oc st (by omega) htlt hlk' s hs hcur htW'
        have happ' : wd.ann = (pre ++ [(t, h)]) ++ rest' := by
          rw [happ, List.append_assoc, List.singleton_append]
        have hpre' : ∀ x ∈ pre ++ [(t, h)],
            x.1 < wd.code.length ∧ wd.ann.lookup x.1 = some x.2 := by
          intro x hx
          rcases List.mem_append.mp hx with hx | hx
          · exact hpre x hx
          · have hx' : x = (t, h) := by simpa using hx
            rw [hx']; exact ⟨htlt, hlk'⟩
        obtain ⟨k2, hk2, hrun2⟩ := ih (pre ++ [(t, h)]) happ' hpre' s1 hs1 (run_regs_ok hrun1 hregs)
        have hle := annval_refuse_mid_le hk1 hk2
        exact ⟨k1 + k2, hle, by rw [run_add im k1 k2, hrun1]; exact hrun2⟩
      · obtain ⟨m, hm, hrun⟩ :=
          annval_dup_exit1B ac oc rc cr st (by omega) htlt hmem hlk heqh s hs hcur htW' hregs
        have h1le : 1 ≤ rest'.length + 1 + 1 := by omega
        have hmul := Nat.mul_le_mul_right (60 + 20 * (wd.ann.length + 1)) h1le
        simp only [Nat.one_mul, List.length_cons] at hmul ⊢
        exact ⟨m, Nat.le_trans hm hmul, hrun⟩
    · have hge : wd.code.length ≤ t := by omega
      obtain ⟨m, hm, hrun⟩ :=
        annval_range_exit1B ac oc rc st (by omega) hge s hs hcur hregs
      have h1le : 1 ≤ rest'.length + 1 + 1 := by omega
      have hmul := Nat.mul_le_mul_right (60 + 20 * (wd.ann.length + 1)) h1le
      simp only [Nat.one_mul, List.length_cons] at hmul ⊢
      exact ⟨m, Nat.le_trans hm hmul, hrun⟩

set_option maxHeartbeats 800000 in
/-- `codewalk_entry` refuses a word whose `wd.ann.lookup 0` entry differs from
`⟨wd.ins, 0⟩` by branching to `ckAnn0D` or `ckAnn0A` and exiting `1`. -/
theorem codewalk_entry_refuseB {im : Nat → Option (Ins × Nat)}
    (ac : AllCode im) (oc : OuterCode im) (rc : RefuseCode im) (cr : ChkRefuseCode im)
    {pr : Prog} {wd : Sporos.Word} {w p sp tbl cp : Nat}
    (st : WalkStatic pr wd w p sp tbl) (hsp48 : 48 ≤ sp)
    (h0 : (match wd.ann.lookup 0 with
      | some a => a == ⟨wd.ins, 0⟩
      | none => true) = false)
    (hinsW : wd.ins < W)
    (hhW : ∀ x ∈ wd.ann, x.2.d < W ∧ x.2.a < W) :
    ∀ s, s.rip = aCodewalk → s.reg 7 = some w → s.reg 4 = some sp →
      s.reg 1 = some 0 → (∃ v, s.reg 9 = some v) →
      (∃ v, s.reg 12 = some v) → 16 ≤ s.regs.length → RegsOk s →
      rd64 s.mem w = some wd.ins → rd64 s.mem (w + 32) = some wd.ann.length →
      rd64 s.mem (w + 40) = some cp → rd64 s.mem (w + 48) = some p →
      AnnLayout s wd.ann p → StackRoom s 6 →
      ∃ m, m ≤ 40 + 20 * (wd.ann.length + 1) ∧ run im m (Out.ok s) = Out.exit 1 := by
  intro s hrip hr7 hr4 hr1 h9ex h12ex h16 hregs hf0 hf32 hf40 hf48 hlay hroom
  have hw56 := st.w56
  have hspW := st.spW
  have hpann := st.pann
  cases hl0 : wd.ann.lookup 0 with
  | none => simp [hl0] at h0
  | some a =>
    have hne : a ≠ ⟨wd.ins, 0⟩ := by
      intro heq
      rw [hl0, heq] at h0
      simp at h0
    obtain ⟨adW, aaW⟩ : a.d < W ∧ a.a < W := hhW (0, a) (lookup_mem hl0)
    -- mov rsi, [rdi+40] ; xor ecx, ecx ; mov r8, [rdi] ; xor r9d, r9d ; xor r12d, r12d
    have hea40 : ea w 40 = w + 40 := Nat.mod_eq_of_lt (by omega)
    have e1 : X86Int.step im s = .ok { s.set 6 (some cp) with rip := s.rip + 4 } :=
      step_ld (by rw [hrip]; exact oc.ck0) hr7 (by rw [hea40]; exact hf40)
    obtain ⟨b1, hb1⟩ : ∃ u : X86Int.St, u = { s.set 6 (some cp) with rip := s.rip + 4 } := ⟨_, rfl⟩
    have hb1rip : b1.rip = aCodewalk + 4 := by rw [hb1]; show s.rip + 4 = aCodewalk + 4; rw [hrip]
    have gb1 : ∀ q, q ≠ 6 → b1.reg q = s.reg q := by
      intro q hq; rw [hb1]; show (s.set 6 _).reg q = s.reg q; rw [reg_set_other hq]
    have hb1mem : b1.mem = s.mem := by rw [hb1]; exact rfl
    have hlen1 : b1.regs.length = s.regs.length := by
      rw [hb1]; show (s.set 6 _).regs.length = s.regs.length; exact List.length_set
    have hb1r1 : b1.reg 1 = some 0 := by rw [gb1 1 (by decide)]; exact hr1
    have e2 : X86Int.step im b1 = .ok { b1.set 1 (some (aluF .xor 0 0).1) with
        fl := (aluF .xor 0 0).2, rip := b1.rip + 2 } :=
      step_alu_xor (by rw [hb1rip]; exact oc.ck1) hb1r1 hb1r1
    obtain ⟨b2, hb2⟩ : ∃ u : X86Int.St, u =
      { b1.set 1 (some (aluF .xor 0 0).1) with fl := (aluF .xor 0 0).2, rip := b1.rip + 2 } := ⟨_, rfl⟩
    have hb2rip : b2.rip = aCodewalk + 6 := by rw [hb2]; show b1.rip + 2 = aCodewalk + 6; rw [hb1rip]
    have gb2 : ∀ q, q ≠ 1 → b2.reg q = b1.reg q := by
      intro q hq; rw [hb2]; show (b1.set 1 _).reg q = b1.reg q; rw [reg_set_other hq]
    have hb2r1 : b2.reg 1 = some 0 := by
      rw [hb2]; show (b1.set 1 _).reg 1 = some 0; rw [reg_set_self (by omega), aluF_xor_fst_self]
    have hb2mem : b2.mem = s.mem := by rw [hb2]; exact hb1mem
    have hlen2 : b2.regs.length = s.regs.length := by
      rw [hb2]; show (b1.set 1 _).regs.length = s.regs.length; exact List.length_set.trans hlen1
    have hb2r7 : b2.reg 7 = some w := by rw [gb2 7 (by decide), gb1 7 (by decide)]; exact hr7
    have hea0 : ea w 0 = w := by unfold ea; rw [Nat.add_zero]; exact Nat.mod_eq_of_lt (by omega)
    have e3 : X86Int.step im b2 = .ok { b2.set 8 (some wd.ins) with rip := b2.rip + 3 } :=
      step_ld (by rw [hb2rip]; exact oc.ck2) hb2r7 (by rw [hea0, hb2mem]; exact hf0)
    obtain ⟨b3, hb3⟩ : ∃ u : X86Int.St, u = { b2.set 8 (some wd.ins) with rip := b2.rip + 3 } := ⟨_, rfl⟩
    have hb3rip : b3.rip = aCodewalk + 9 := by rw [hb3]; show b2.rip + 3 = aCodewalk + 9; rw [hb2rip]
    have gb3 : ∀ q, q ≠ 8 → b3.reg q = b2.reg q := by
      intro q hq; rw [hb3]; show (b2.set 8 _).reg q = b2.reg q; rw [reg_set_other hq]
    have hb3r8 : b3.reg 8 = some wd.ins := by
      rw [hb3]; show (b2.set 8 _).reg 8 = some wd.ins; rw [reg_set_self (by omega)]
    have hb3mem : b3.mem = s.mem := by rw [hb3]; exact hb2mem
    have hlen3 : b3.regs.length = s.regs.length := by
      rw [hb3]; show (b2.set 8 _).regs.length = s.regs.length; exact List.length_set.trans hlen2
    obtain ⟨v9, hb3r9⟩ : ∃ v, b3.reg 9 = some v := by
      obtain ⟨v, hv⟩ := h9ex
      exact ⟨v, by rw [gb3 9 (by decide), gb2 9 (by decide), gb1 9 (by decide)]; exact hv⟩
    have e4 : X86Int.step im b3 = .ok { b3.set 9 (some (aluF .xor v9 v9).1) with
        fl := (aluF .xor v9 v9).2, rip := b3.rip + 3 } :=
      step_alu_xor (by rw [hb3rip]; exact oc.ck3) hb3r9 hb3r9
    obtain ⟨b4, hb4⟩ : ∃ u : X86Int.St, u =
      { b3.set 9 (some (aluF .xor v9 v9).1) with fl := (aluF .xor v9 v9).2, rip := b3.rip + 3 } := ⟨_, rfl⟩
    have hb4rip : b4.rip = aCodewalk + 12 := by rw [hb4]; show b3.rip + 3 = aCodewalk + 12; rw [hb3rip]
    have gb4 : ∀ q, q ≠ 9 → b4.reg q = b3.reg q := by
      intro q hq; rw [hb4]; show (b3.set 9 _).reg q = b3.reg q; rw [reg_set_other hq]
    have hb4r9 : b4.reg 9 = some 0 := by
      rw [hb4]; show (b3.set 9 _).reg 9 = some 0; rw [reg_set_self (by omega), aluF_xor_fst_self]
    have hb4mem : b4.mem = s.mem := by rw [hb4]; exact hb3mem
    have hlen4 : b4.regs.length = s.regs.length := by
      rw [hb4]; show (b3.set 9 _).regs.length = s.regs.length; exact List.length_set.trans hlen3
    obtain ⟨v12, hb4r12⟩ : ∃ v, b4.reg 12 = some v := by
      obtain ⟨v, hv⟩ := h12ex
      exact ⟨v, by rw [gb4 12 (by decide), gb3 12 (by decide), gb2 12 (by decide),
        gb1 12 (by decide)]; exact hv⟩
    have e5 : X86Int.step im b4 = .ok { b4.set 12 (some (aluF .xor v12 v12).1) with
        fl := (aluF .xor v12 v12).2, rip := b4.rip + 3 } :=
      step_alu_xor (by rw [hb4rip]; exact oc.ck4) hb4r12 hb4r12
    obtain ⟨b5, hb5⟩ : ∃ u : X86Int.St, u =
      { b4.set 12 (some (aluF .xor v12 v12).1) with fl := (aluF .xor v12 v12).2, rip := b4.rip + 3 } := ⟨_, rfl⟩
    have hb5rip : b5.rip = aCodewalk + 15 := by rw [hb5]; show b4.rip + 3 = aCodewalk + 15; rw [hb4rip]
    have gb5 : ∀ q, q ≠ 12 → b5.reg q = b4.reg q := by
      intro q hq; rw [hb5]; show (b4.set 12 _).reg q = b4.reg q; rw [reg_set_other hq]
    have hb5r12 : b5.reg 12 = some 0 := by
      rw [hb5]; show (b4.set 12 _).reg 12 = some 0; rw [reg_set_self (by omega), aluF_xor_fst_self]
    have hb5mem : b5.mem = s.mem := by rw [hb5]; exact hb4mem
    have hlen5 : b5.regs.length = s.regs.length := by
      rw [hb5]; show (b4.set 12 _).regs.length = s.regs.length; exact List.length_set.trans hlen4
    have hb5r7 : b5.reg 7 = some w := by
      rw [gb5 7 (by decide), gb4 7 (by decide), gb3 7 (by decide)]; exact hb2r7
    have hb5r4 : b5.reg 4 = some sp := by
      rw [gb5 4 (by decide), gb4 4 (by decide), gb3 4 (by decide),
        gb2 4 (by decide), gb1 4 (by decide)]; exact hr4
    have hb5r1 : b5.reg 1 = some 0 := by
      rw [gb5 1 (by decide), gb4 1 (by decide), gb3 1 (by decide)]; exact hb2r1
    have hb5r8 : b5.reg 8 = some wd.ins := by
      rw [gb5 8 (by decide), gb4 8 (by decide)]; exact hb3r8
    have hb5r9 : b5.reg 9 = some 0 := by rw [gb5 9 (by decide)]; exact hb4r9
    -- call ANNL at 0
    obtain ⟨k6, b6, hk6, hrun6, hb6rip, hpost, -, hb6r4, -, -, -, gb6,
      hfr6seg, hex0, hloc6, hw6seg, hlen6⟩ :=
      CallANNLB ac.annl ac.l32 (by decide) oc.ck5 (by decide) (by decide) st.p0
        st.w56 st.pann (by omega) st.spW b5 hb5rip hb5r12 hb5r7 hb5r4 hb5r1
        (by rw [hb5mem]; exact hf48) (by rw [hb5mem]; exact hf32)
        (AnnLayout_of_mem hb5mem hlay)
        (StackRoom_of_mem hb5mem (by rw [hb5r4, hr4]) (StackRoom_weaken (StackRoom_weaken hroom)))
        st.disjw st.disjp
    have hw6 : wshape b6.mem = wshape s.mem := by rw [hw6seg, hb5mem]
    have hlen6s : b6.regs.length = s.regs.length := hlen6.trans hlen5
    have hb6r8 : b6.reg 8 = some wd.ins := by
      rw [gb6 8 (by decide) (by decide) (by decide) (by decide)]; exact hb5r8
    have hb6r9 : b6.reg 9 = some 0 := by
      rw [gb6 9 (by decide) (by decide) (by decide) (by decide)]; exact hb5r9
    obtain ⟨e, hb6r0⟩ := hex0
    have rr1 : run im 1 (Out.ok s) = Out.ok b1 := by rw [run_succ_ok im 0 s, e1, ← hb1, run_zero]
    have rr2 : run im 1 (Out.ok b1) = Out.ok b2 := by rw [run_succ_ok im 0 b1, e2, ← hb2, run_zero]
    have rr3 : run im 1 (Out.ok b2) = Out.ok b3 := by rw [run_succ_ok im 0 b2, e3, ← hb3, run_zero]
    have rr4 : run im 1 (Out.ok b3) = Out.ok b4 := by rw [run_succ_ok im 0 b3, e4, ← hb4, run_zero]
    have rr5 : run im 1 (Out.ok b4) = Out.ok b5 := by rw [run_succ_ok im 0 b4, e5, ← hb5, run_zero]
    have hrun06 : run im (5 + k6) (Out.ok s) = Out.ok b6 := by
      rw [run_add im 5 k6, run_add im 1 4, rr1, run_add im 1 3, rr2,
        run_add im 1 2, rr3, run_add im 1 1, rr4, rr5, hrun6]
    obtain ⟨he0, hat6⟩ : e ≠ 0 ∧ AnnAt b6 e 0 a.d a.a := by
      cases hpost e hb6r0 with
      | inl hb => rw [hb.2] at hl0; simp at hl0
      | inr hb =>
        obtain ⟨d1, a1, hne0, hat, hlk1⟩ := hb
        have hhh : (⟨d1, a1⟩ : Height) = a := Option.some.inj (hlk1.symm.trans hl0)
        have hdeq : d1 = a.d := congrArg Height.d hhh
        have haeq : a1 = a.a := congrArg Height.a hhh
        exact ⟨hne0, by rwa [hdeq, haeq] at hat⟩
    obtain ⟨hel, heh⟩ := hloc6 e hb6r0 he0
    have hdisjp := st.disjp
    have hadd : ∀ (x k : Nat), k < W → x + k < W → (aluF .add x (k % W)).1 = x + k := by
      intro x k hkW hk
      rw [aluF_add_fst, Nat.mod_eq_of_lt hkW]
      exact Nat.mod_eq_of_lt hk
    have hslot8 : (sp + W - 8) % W = sp - 8 := by
      rw [show sp + W - 8 = sp + W - 8 * 1 by omega, slot_sub (by omega) hspW]
    -- test rax, rax ; jz not taken ; push rax ; mov rdx, rax ; add rdx, 4 ; call L32 -> a.d
    have e7 : X86Int.step im b6 =
        .ok { b6 with fl := (aluF .and e e).2, rip := b6.rip + 3 } :=
      step_test (by rw [hb6rip]; exact oc.ck6) hb6r0 hb6r0
    obtain ⟨b7, hb7⟩ : ∃ u : X86Int.St, u =
      { b6 with fl := (aluF .and e e).2, rip := b6.rip + 3 } := ⟨_, rfl⟩
    have hb7rip : b7.rip = aCodewalk + 23 := by rw [hb7]; show b6.rip + 3 = aCodewalk + 23; rw [hb6rip]
    have hb7fl : X86Int.cond b7.fl 4 = some false := by
      have hz : b7.fl.zf = some false := by
        rw [hb7]; show (aluF .and e e).2.zf = some false; rw [aluF_zf, aluF_and_fst_self]; simp [he0]
      exact cond_cc4 hz
    have gb7 : ∀ q, b7.reg q = b6.reg q := fun q => by rw [hb7]; exact rfl
    have hb7mem : b7.mem = b6.mem := by rw [hb7]
    have e8 : X86Int.step im b7 = .ok { b7 with rip := aCodewalk + 25 } := by
      have hh := step_jcc (im := im) (s := b7) (by rw [hb7rip]; exact oc.ck7) hb7fl
      simpa [hb7rip] using hh
    obtain ⟨b8, hb8⟩ : ∃ u : X86Int.St, u = { b7 with rip := aCodewalk + 25 } := ⟨_, rfl⟩
    have hb8rip : b8.rip = aCodewalk + 25 := by rw [hb8]
    have gb8 : ∀ q, b8.reg q = b6.reg q := fun q => by rw [hb8]; exact gb7 q
    have hb8mem : b8.mem = b6.mem := by rw [hb8]; exact hb7mem
    have hb8r0 : b8.reg 0 = some e := by rw [gb8 0]; exact hb6r0
    have hb8r4 : b8.reg 4 = some sp := by rw [gb8 4]; exact hb6r4
    have hlen8 : b8.regs.length = s.regs.length := by rw [hb8, hb7]; exact hlen6s
    have hw8 : wshape b8.mem = wshape s.mem := by rw [hb8mem]; exact hw6
    have hroom8 : StackRoom b8 1 :=
      StackRoom_shift (L := 0) hroom hr4 hb8r4 hw8 ⟨0, rfl, by omega⟩ hsp48 hspW
    obtain ⟨m9, hm9s⟩ := Option.isSome_iff_exists.mp (hroom8 sp hb8r4 0 (by decide) e)
    have hm9 : wr64 b8.mem (sp - 8) e = some m9 := by
      have e' : (sp + W - 8 * (0 + 1)) % W = sp - 8 := slot_sub (by omega) hspW
      rw [e'] at hm9s; exact hm9s
    have hm9p : wr64 b8.mem ((sp + W - 8) % W) e = some m9 := by rw [hslot8]; exact hm9
    have e9 : X86Int.step im b8 = .ok { b8.set 4 (some ((sp + W - 8) % W)) with
        mem := m9, rip := b8.rip + 1 } :=
      step_push (by rw [hb8rip]; exact oc.ck8) hb8r0 hb8r4 hm9p
    obtain ⟨b9, hb9⟩ : ∃ u : X86Int.St, u =
      { b8.set 4 (some ((sp + W - 8) % W)) with mem := m9, rip := b8.rip + 1 } := ⟨_, rfl⟩
    have hb9rip : b9.rip = aCodewalk + 26 := by rw [hb9]; show b8.rip + 1 = aCodewalk + 26; rw [hb8rip]
    have hb9r4 : b9.reg 4 = some (sp - 8) := by
      rw [hb9]; show (b8.set 4 _).reg 4 = some (sp - 8); rw [reg_set_self (by omega), hslot8]
    have gb9 : ∀ q, q ≠ 4 → b9.reg q = b8.reg q := by
      intro q hq; rw [hb9]; show (b8.set 4 _).reg q = b8.reg q; rw [reg_set_other hq]
    have hb9mem : b9.mem = m9 := by rw [hb9]
    have hw9 : wshape b9.mem = wshape s.mem := by
      rw [hb9mem]; exact (wrN_wshape _ _ _ hm9).trans hw8
    have hlen9 : b9.regs.length = s.regs.length := by
      rw [hb9]; show (b8.set 4 _).regs.length = s.regs.length; exact List.length_set.trans hlen8
    have hslot9 : rd64 b9.mem (sp - 8) = some e := by
      have h1 := rd64_wr64 hm9; rw [Nat.mod_eq_of_lt (by omega)] at h1; rw [hb9mem]; exact h1
    have hat9 : AnnAt b9 e 0 a.d a.a :=
      ⟨by rw [hb9mem, rdN_wr64_other hm9 (by omega), hb8mem]; exact hat6.1,
       by rw [hb9mem, rdN_wr64_other hm9 (by omega), hb8mem]; exact hat6.2.1,
       by rw [hb9mem, rdN_wr64_other hm9 (by omega), hb8mem]; exact hat6.2.2⟩
    have hb9r0 : b9.reg 0 = some e := by rw [gb9 0 (by decide)]; exact hb8r0
    have e10 : X86Int.step im b9 = .ok { b9.set 2 (some e) with rip := b9.rip + 3 } :=
      step_mov (by rw [hb9rip]; exact oc.ck9) hb9r0
    obtain ⟨b10, hb10⟩ : ∃ u : X86Int.St, u = { b9.set 2 (some e) with rip := b9.rip + 3 } := ⟨_, rfl⟩
    have hb10rip : b10.rip = aCodewalk + 29 := by rw [hb10]; show b9.rip + 3 = aCodewalk + 29; rw [hb9rip]
    have gb10 : ∀ q, q ≠ 2 → b10.reg q = b9.reg q := by
      intro q hq; rw [hb10]; show (b9.set 2 _).reg q = b9.reg q; rw [reg_set_other hq]
    have hb10r2 : b10.reg 2 = some e := by
      rw [hb10]; show (b9.set 2 _).reg 2 = some e; rw [reg_set_self (by omega)]
    have hb10mem : b10.mem = b9.mem := by rw [hb10]; exact rfl
    have hlen10 : b10.regs.length = s.regs.length := by
      rw [hb10]; show (b9.set 2 _).regs.length = s.regs.length; exact List.length_set.trans hlen9
    have e11 : X86Int.step im b10 = .ok { b10.set 2 (some (aluF .add e (4 % W)).1) with
        fl := (aluF .add e (4 % W)).2, rip := b10.rip + 4 } :=
      step_aluI_add (by rw [hb10rip]; exact oc.ck10) hb10r2
    obtain ⟨b11, hb11⟩ : ∃ u : X86Int.St, u = { b10.set 2 (some (aluF .add e (4 % W)).1) with fl := (aluF .add e (4 % W)).2, rip := b10.rip + 4 } := ⟨_, rfl⟩
    have hb11rip : b11.rip = aCodewalk + 33 := by rw [hb11]; show b10.rip + 4 = aCodewalk + 33; rw [hb10rip]
    have hb11r2 : b11.reg 2 = some (e + 4) := by
      rw [hb11]; show (b10.set 2 _).reg 2 = some (e + 4)
      rw [reg_set_self (by omega), hadd e 4 (by decide) (by omega)]
    have gb11 : ∀ q, q ≠ 2 → b11.reg q = b10.reg q := by
      intro q hq; rw [hb11]; show (b10.set 2 _).reg q = b10.reg q; rw [reg_set_other hq]
    have hb11mem : b11.mem = b9.mem := by rw [hb11]; exact hb10mem
    have hb11r4 : b11.reg 4 = some (sp - 8) := by
      rw [gb11 4 (by decide), gb10 4 (by decide)]; exact hb9r4
    have hw11 : wshape b11.mem = wshape s.mem := by rw [hb11mem]; exact hw9
    have hlen11 : b11.regs.length = s.regs.length := by
      rw [hb11]; show (b10.set 2 _).regs.length = s.regs.length; exact List.length_set.trans hlen10
    obtain ⟨k12, b12, hk12, hrun12, hb12rip, hb12r0, hb12r4, gb12, hlen12, hfr12seg, hw12seg⟩ :=
      CallL32B ac.l32 ac.annl.dret oc.ck11 (by omega) (by omega) (by omega)
        (by decide) (by omega) b11 hb11rip hb11r2 hb11r4
        (by rw [hb11mem]; exact hat9.2.1)
        (StackRoom_shift hroom hr4 hb11r4 hw11 ⟨1, by omega, by omega⟩ hsp48 hspW)
    have hw12 : wshape b12.mem = wshape s.mem := hw12seg.trans hw11
    have hlen12s : b12.regs.length = s.regs.length := hlen12.trans hlen11
    have hslot12 : rd64 b12.mem (sp - 8) = some e := by
      unfold rd64; rw [hfr12seg (sp - 8) 8 (by omega), hb11mem]; exact hslot9
    have hat12a : (b12.mem.rdN (e + 8) 4).map leVal = some a.a := by
      rw [hfr12seg (e + 8) 4 (by omega), hb11mem]; exact hat9.2.2
    have hb12r8 : b12.reg 8 = some wd.ins := by
      rw [gb12 8 (by decide) (by decide), gb11 8 (by decide), gb10 8 (by decide),
        gb9 8 (by decide), gb8 8]; exact hb6r8
    have hb12r9 : b12.reg 9 = some 0 := by
      rw [gb12 9 (by decide) (by decide), gb11 9 (by decide), gb10 9 (by decide),
        gb9 9 (by decide), gb8 9]; exact hb6r9
    -- cmp rax, r8
    have e13 : X86Int.step im b12 =
        .ok { b12 with fl := (aluF .sub a.d wd.ins).2, rip := b12.rip + 3 } :=
      step_cmp (by rw [hb12rip]; exact oc.ck12) hb12r0 hb12r8
    obtain ⟨b13, hb13⟩ : ∃ u : X86Int.St, u =
      { b12 with fl := (aluF .sub a.d wd.ins).2, rip := b12.rip + 3 } := ⟨_, rfl⟩
    have hb13rip : b13.rip = aCodewalk + 41 := by rw [hb13]; show b12.rip + 3 = aCodewalk + 41; rw [hb12rip]
    have rr7 : run im 1 (Out.ok b6) = Out.ok b7 := by rw [run_succ_ok im 0 b6, e7, ← hb7, run_zero]
    have rr8 : run im 1 (Out.ok b7) = Out.ok b8 := by rw [run_succ_ok im 0 b7, e8, ← hb8, run_zero]
    have rr9 : run im 1 (Out.ok b8) = Out.ok b9 := by rw [run_succ_ok im 0 b8, e9, ← hb9, run_zero]
    have rr10 : run im 1 (Out.ok b9) = Out.ok b10 := by rw [run_succ_ok im 0 b9, e10, ← hb10, run_zero]
    have rr11 : run im 1 (Out.ok b10) = Out.ok b11 := by rw [run_succ_ok im 0 b10, e11, ← hb11, run_zero]
    have rr13 : run im 1 (Out.ok b12) = Out.ok b13 := by rw [run_succ_ok im 0 b12, e13, ← hb13, run_zero]
    have hrun11 : run im (5 + k6 + 5) (Out.ok s) = Out.ok b11 := by
      rw [run_add im (5 + k6) 5, hrun06, run_add im 1 4, rr7, run_add im 1 3, rr8,
        run_add im 1 2, rr9, run_add im 1 1, rr10, rr11]
    have hrun12 : run im (5 + k6 + 5 + k12) (Out.ok s) = Out.ok b12 := by
      rw [run_add im (5 + k6 + 5) k12, hrun11, hrun12]
    have hrun13 : run im (5 + k6 + 5 + k12 + 1) (Out.ok s) = Out.ok b13 := by
      rw [run_add im (5 + k6 + 5 + k12) 1, hrun12, rr13]
    by_cases hd : a.d = wd.ins
    · have ha : a.a ≠ 0 := by
        intro ha0; apply hne; cases a; simp at hd ha0; simp [hd, ha0]
      have hb13fl : X86Int.cond b13.fl 4 = some true := by
        have hzf : b13.fl.zf = some true := by rw [hb13]; exact aluF_sub_zf_eq adW hinsW hd
        exact cond_cc4 hzf
      have e14 : X86Int.step im b13 = .ok { b13 with rip := aCodewalk + 48 } := by
        have hh := step_jcc (im := im) (s := b13) (by rw [hb13rip]; exact oc.ck13) hb13fl
        simpa using hh
      obtain ⟨b14, hb14⟩ : ∃ u : X86Int.St, u = { b13 with rip := aCodewalk + 48 } := ⟨_, rfl⟩
      have hb14rip : b14.rip = aCodewalk + 48 := by rw [hb14]
      have hb14mem : b14.mem = b12.mem := by rw [hb14, hb13]
      have hb14r4 : b14.reg 4 = some (sp - 8) := by rw [hb14, hb13]; exact hb12r4
      have hb14r9 : b14.reg 9 = some 0 := by rw [hb14, hb13]; exact hb12r9
      have hp8 : ((sp - 8) + 8) % W = sp := slot_add8 (by omega) hspW
      have e15 : X86Int.step im b14 = .ok
          { (b14.set 4 (some ((sp - 8 + 8) % W))).set 2 (some e) with rip := b14.rip + 1 } :=
        step_pop (by rw [hb14rip]; exact oc.ck14) hb14r4 (by rw [hb14mem]; exact hslot12)
      obtain ⟨b15, hb15⟩ : ∃ u : X86Int.St, u =
        { (b14.set 4 (some ((sp - 8 + 8) % W))).set 2 (some e) with rip := b14.rip + 1 } := ⟨_, rfl⟩
      have hb15rip : b15.rip = aCodewalk + 49 := by rw [hb15]; show b14.rip + 1 = aCodewalk + 49; rw [hb14rip]
      have hlen14 : b14.regs.length = s.regs.length := by rw [hb14, hb13]; exact hlen12s
      have hlen14a : (b14.set 4 (some ((sp - 8 + 8) % W))).regs.length = s.regs.length :=
        List.length_set.trans hlen14
      have hb15r4 : b15.reg 4 = some sp := by
        rw [hb15]; show ((b14.set 4 _).set 2 _).reg 4 = some sp
        rw [reg_set_other (by decide), reg_set_self (by omega), hp8]
      have hb15r2 : b15.reg 2 = some e := by
        rw [hb15]; show ((b14.set 4 _).set 2 _).reg 2 = some e; rw [reg_set_self (by omega)]
      have gb15 : ∀ q, q ≠ 2 → q ≠ 4 → b15.reg q = b14.reg q := by
        intro q h2 h4; rw [hb15]; show ((b14.set 4 _).set 2 _).reg q = b14.reg q
        rw [reg_set_other h2, reg_set_other h4]
      have hb15mem : b15.mem = b12.mem := by rw [hb15]; exact hb14mem
      have hlen15 : b15.regs.length = s.regs.length := by
        rw [hb15]; exact List.length_set.trans hlen14a
      have e16 : X86Int.step im b15 = .ok { b15.set 2 (some (aluF .add e (8 % W)).1) with
          fl := (aluF .add e (8 % W)).2, rip := b15.rip + 4 } :=
        step_aluI_add (by rw [hb15rip]; exact oc.ck15) hb15r2
      obtain ⟨b16, hb16⟩ : ∃ u : X86Int.St, u = { b15.set 2 (some (aluF .add e (8 % W)).1) with fl := (aluF .add e (8 % W)).2, rip := b15.rip + 4 } := ⟨_, rfl⟩
      have hb16rip : b16.rip = aCodewalk + 53 := by rw [hb16]; show b15.rip + 4 = aCodewalk + 53; rw [hb15rip]
      have hb16r2 : b16.reg 2 = some (e + 8) := by
        rw [hb16]; show (b15.set 2 _).reg 2 = some (e + 8)
        rw [reg_set_self (by omega), hadd e 8 (by decide) (by omega)]
      have gb16 : ∀ q, q ≠ 2 → b16.reg q = b15.reg q := by
        intro q hq; rw [hb16]; show (b15.set 2 _).reg q = b15.reg q; rw [reg_set_other hq]
      have hb16mem : b16.mem = b12.mem := by rw [hb16]; exact hb15mem
      have hb16r4 : b16.reg 4 = some sp := by rw [gb16 4 (by decide)]; exact hb15r4
      have hw16 : wshape b16.mem = wshape s.mem := by rw [hb16mem]; exact hw12
      have hb16r9 : b16.reg 9 = some 0 := by
        rw [gb16 9 (by decide), gb15 9 (by decide) (by decide)]; exact hb14r9
      obtain ⟨k17, b17, hk17, hrun17, hb17rip, hb17r0, hb17r4, gb17, -, -, hw17seg⟩ :=
        CallL32B ac.l32 ac.annl.dret oc.ck16 (by omega) (by omega) (by omega)
          (by decide) (by omega) b16 hb16rip hb16r2 hb16r4
          (by rw [hb16mem]; exact hat12a)
          (StackRoom_shift (L := 0) hroom hr4 hb16r4 hw16 ⟨0, rfl, by omega⟩ hsp48 hspW)
      have hw17 : wshape b17.mem = wshape s.mem := hw17seg.trans hw16
      have hb17r9 : b17.reg 9 = some 0 := by rw [gb17 9 (by decide) (by decide)]; exact hb16r9
      -- cmp rax, r9 ; jz not taken -> ckAnn0A
      have e18 : X86Int.step im b17 =
          .ok { b17 with fl := (aluF .sub a.a 0).2, rip := b17.rip + 3 } :=
        step_cmp (by rw [hb17rip]; exact oc.ck17) hb17r0 hb17r9
      obtain ⟨b18, hb18⟩ : ∃ u : X86Int.St, u =
        { b17 with fl := (aluF .sub a.a 0).2, rip := b17.rip + 3 } := ⟨_, rfl⟩
      have hb18rip : b18.rip = aCodewalk + 61 := by rw [hb18]; show b17.rip + 3 = aCodewalk + 61; rw [hb17rip]
      have hb18fl : X86Int.cond b18.fl 4 = some false := by
        have hzf : b18.fl.zf = some false := by rw [hb18]; exact aluF_sub_zf_ne aaW (by decide) ha
        exact cond_cc4 hzf
      have e19 : X86Int.step im b18 = .ok { b18 with rip := aCodewalk + 63 } := by
        have hh := step_jcc (im := im) (s := b18) (by rw [hb18rip]; exact oc.ck18) hb18fl
        simpa [hb18rip] using hh
      obtain ⟨b19, hb19⟩ : ∃ u : X86Int.St, u = { b18 with rip := aCodewalk + 63 } := ⟨_, rfl⟩
      have rr14 : run im 1 (Out.ok b13) = Out.ok b14 := by rw [run_succ_ok im 0 b13, e14, ← hb14, run_zero]
      have rr15 : run im 1 (Out.ok b14) = Out.ok b15 := by rw [run_succ_ok im 0 b14, e15, ← hb15, run_zero]
      have rr16 : run im 1 (Out.ok b15) = Out.ok b16 := by rw [run_succ_ok im 0 b15, e16, ← hb16, run_zero]
      have rr18 : run im 1 (Out.ok b17) = Out.ok b18 := by rw [run_succ_ok im 0 b17, e18, ← hb18, run_zero]
      have rr19 : run im 1 (Out.ok b18) = Out.ok b19 := by rw [run_succ_ok im 0 b18, e19, ← hb19, run_zero]
      have hrun16 : run im ((5 + k6 + 5 + k12 + 1) + 3) (Out.ok s) = Out.ok b16 := by
        rw [run_add im _ 3, hrun13, run_add im 1 2, rr14, run_add im 1 1, rr15, rr16]
      have hrun17 : run im ((5 + k6 + 5 + k12 + 1) + 3 + k17) (Out.ok s) = Out.ok b17 := by
        rw [run_add im _ k17, hrun16, hrun17]
      have hrun19 : run im ((5 + k6 + 5 + k12 + 1) + (3 + k17 + 2)) (Out.ok s) = Out.ok b19 := by
        rw [show ((5 + k6 + 5 + k12 + 1) + (3 + k17 + 2)) =
            ((5 + k6 + 5 + k12 + 1) + 3 + k17) + 2 by omega,
          run_add im _ 2, hrun17, run_add im 1 1, rr18, rr19]
      have hb19rip : b19.rip = aCodewalk + 63 := by rw [hb19]
      have hb19r4 : b19.reg 4 = some sp := by rw [hb19, hb18]; exact hb17r4
      have hwb19 : wshape b19.mem = wshape s.mem := by rw [hb19, hb18]; exact hw17
      obtain ⟨k20, hk20, hrun20⟩ := jmp_fail1_exit1B rc cr.ckAnn0A (by omega) hspW b19
        hb19rip hb19r4 (run_regs_ok hrun19 hregs)
        (StackRoom_shift (L := 0) hroom hr4 hb19r4 hwb19 ⟨0, rfl, by omega⟩ hsp48 hspW)
      have hle := codewalk_refuse_a_le k6 k12 k17 k20 wd.ann.length hk6 hk12 hk17 hk20
      refine ⟨((5 + k6 + 5 + k12 + 1) + (3 + k17 + 2)) + k20, (by omega),
        by rw [run_add im _ k20, hrun19]; exact hrun20⟩
    · -- a.d ≠ wd.ins: jz not taken -> ckAnn0D
      have hb13fl : X86Int.cond b13.fl 4 = some false := by
        have hzf : b13.fl.zf = some false := by rw [hb13]; exact aluF_sub_zf_ne adW hinsW hd
        exact cond_cc4 hzf
      have e14 : X86Int.step im b13 = .ok { b13 with rip := aCodewalk + 43 } := by
        have hh := step_jcc (im := im) (s := b13) (by rw [hb13rip]; exact oc.ck13) hb13fl
        simpa [hb13rip] using hh
      obtain ⟨b14, hb14⟩ : ∃ u : X86Int.St, u = { b13 with rip := aCodewalk + 43 } := ⟨_, rfl⟩
      have rr14 : run im 1 (Out.ok b13) = Out.ok b14 := by rw [run_succ_ok im 0 b13, e14, ← hb14, run_zero]
      have hrun14 : run im ((5 + k6 + 5 + k12 + 1) + 1) (Out.ok s) = Out.ok b14 := by
        rw [run_add im _ 1, hrun13, rr14]
      have hb14rip : b14.rip = aCodewalk + 43 := by rw [hb14]
      have hb14r4 : b14.reg 4 = some (sp - 8) := by rw [hb14, hb13]; exact hb12r4
      have hwb14 : wshape b14.mem = wshape s.mem := by rw [hb14, hb13]; exact hw12
      obtain ⟨k15, hk15, hrun15⟩ := jmp_fail1_exit1B rc cr.ckAnn0D (by omega) (by omega) b14
        hb14rip hb14r4 (run_regs_ok hrun14 hregs)
        (StackRoom_shift hroom hr4 hb14r4 hwb14 ⟨1, by omega, by omega⟩ hsp48 hspW)
      have hle := codewalk_refuse_d_le k6 k12 k15 wd.ann.length hk6 hk12 hk15
      refine ⟨((5 + k6 + 5 + k12 + 1) + 1) + k15, (by omega),
        by rw [run_add im _ k15, hrun14]; exact hrun15⟩

end SporosX1
