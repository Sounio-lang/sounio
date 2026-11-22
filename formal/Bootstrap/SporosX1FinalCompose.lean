import Bootstrap.SporosX1FinalRefuse
import Bootstrap.SporosX1FinalWalkRefuse

/-!
# Sporos stage 1: refusal composition and final equivalence proof

Composes the accept and refusal directions of the single-pass stage-1 checker
(`stage1_accept_exit0` and `stage1_refuse_exit1`) with determinism (`run_exit_det`)
and the cubic time fit (`prefix_plus_checker_le_stage1Bound`) into `stage1_final_target_proof`.
-/

open X86Env X86Int X86Dec Sporos

namespace SporosX1

/-- Per-word step cost bound used across both accept and refusal checker loops. -/
abbrev wordCost (wd : Sporos.Word) : Nat :=
  ((7 + (wd.ann.length + 1) * (60 + 20 * (wd.ann.length + 1)) +
    (30 + 20 * (wd.ann.length + 1))) +
    (wd.code.length + 1) * (200 + 60 * (wd.ann.length + 1))) + 3

private theorem wordCost_ge_12 (wd : Sporos.Word) : 12 ≤ wordCost wd := by
  have h1 : 1 * 200 ≤ (wd.code.length + 1) * (200 + 60 * (wd.ann.length + 1)) :=
    Nat.mul_le_mul (by omega) (by omega)
  unfold wordCost
  omega

private theorem wordCost_ge_cw (a c k1 k2 k3 : Nat)
    (h1 : k1 ≤ 7) (h2 : k2 ≤ (a + 1) * (60 + 20 * (a + 1)))
    (h3 : k3 ≤ 40 + 20 * (a + 1)) :
    k1 + k2 + k3 ≤ ((7 + (a + 1) * (60 + 20 * (a + 1)) + (30 + 20 * (a + 1))) +
      (c + 1) * (200 + 60 * (a + 1))) + 3 := by
  have hc : 1 * 200 ≤ (c + 1) * (200 + 60 * (a + 1)) :=
    Nat.mul_le_mul (by omega) (by omega)
  omega

/-- Single-word step lemma parameterized by `checkWord1 pr wd = true`. -/
theorem word_step_oneB {im : Nat → Option (Ins × Nat)}
    (ac : AllCode im) (oc : OuterCode im)
    {pr : Prog} {tbl sp : Nat} (hsp48 : 48 ≤ sp) (hspW : sp < W)
    (htblW : tbl + 64 * pr.words.length < W)
    (htab : ∀ j, j < pr.words.length →
      sp ≤ tbl + 64 * j ∨ tbl + 64 * j + 64 ≤ sp - 48)
    (hdisjEff48 : aEff + 114 ≤ sp - 48 ∨ sp ≤ aEff) :
    ∀ (wd : Sporos.Word) (rest : List Sporos.Word)
      (pre : List Sporos.Word), pr.words = pre ++ (wd :: rest) →
      Sporos.checkWord1 pr wd = true →
      ∀ s, OuterSt pr tbl sp (wd :: rest) s →
      ∃ m s', m ≤ wordCost wd ∧ run im m (Out.ok s) = Out.ok s' ∧
        OuterSt pr tbl sp rest s' := by
  intro wd rest pre happ hwd s hs
  obtain ⟨hrip, hr5, hr7, hr13, hr14, hr4, h16, hregs, hroom, heff, htabS⟩ := hs
  have hlenann : pr.words.length = pre.length + (rest.length + 1) := by
    rw [happ, List.length_append, List.length_cons]
  have hj0 : pre.length < pr.words.length := by omega
  have hget? : pr.words[pre.length]? = some wd := by
    rw [happ, List.getElem?_append_right (by simp)]
    simp
  have hget : pr.words[pre.length]'hj0 = wd :=
    Option.some.inj ((List.getElem?_eq_getElem hj0).symm.trans hget?)
  obtain ⟨cp, p, hf0, hf8, hf16, hf24, hf32, hf40, hf48, hcsq, hlay, hok⟩ :=
    htabS pre.length hj0
  rw [hget] at hf0 hf8 hf16 hf24 hf32 hcsq hlay hok
  obtain ⟨st, hinsW, hhW, himm, hhb, hdisjC48, hcbound, hdisjw48, hdisjp48⟩ := hok
  have hr7w : s.reg 7 = some (tbl + 64 * pre.length) := by
    have hw : tbl + 64 * (pr.words.length - (wd :: rest).length) =
        tbl + 64 * pre.length := by
      rw [List.length_cons]
      congr 1; congr 1; omega
    rw [hw] at hr7
    exact hr7
  have hwd2 := hwd
  simp only [Sporos.checkWord1, Bool.and_eq_true] at hwd2
  have hne := hwd2.1.1.1
  have h0clause := hwd2.1.1.2
  have hann := hwd2.1.2
  have hwalk := hwd2.2
  have hnc : wd.code.length ≠ 0 := by
    intro hz
    have hc : wd.code = [] := List.eq_nil_of_length_eq_zero hz
    rw [hc] at hne
    simp at hne
  have hlenW : pr.words.length < W := by
    have h1 : pr.words.length ≤ 64 * pr.words.length :=
      Nat.le_mul_of_pos_left _ (by decide)
    omega
  obtain ⟨k1, s1, hk1, hrun1, hs1rip, hs1r6, hs1r1, -, hs1r5, hs1r7, gb1, hs1mem, hlen1⟩ :=
    CheckWord_headB oc (tbl + 64 * pre.length) (rest.length + 1)
      wd.code.length wd.ann.length p st.w56 (by omega) hnc s hrip hr5 hr7w
      hf24 hf48 hf32
  have hs1r4 : s1.reg 4 = some sp := by
    rw [gb1 4 (by decide) (by decide) (by decide)]; exact hr4
  have hregs1 : RegsOk s1 := run_regs_ok hrun1 hregs
  have hs1r13 : s1.reg 13 = some pr.words.length := by
    rw [gb1 13 (by decide) (by decide) (by decide)]; exact hr13
  have hs1r14 : s1.reg 14 = some tbl := by
    rw [gb1 14 (by decide) (by decide) (by decide)]; exact hr14
  have hannst : AnnSt wd (tbl + 64 * pre.length) p sp wd.ann s1 := by
    refine ⟨hs1rip, hs1r7, ?_, hs1r1, hs1r4,
      by rw [hs1mem]; exact hf24, by rw [hs1mem]; exact hf32,
      by rw [hs1mem]; exact hf48, AnnLayout_of_mem hs1mem hlay, by omega,
      StackRoom_of_mem hs1mem (by rw [hs1r4, hr4]) hroom⟩
    simpa using hs1r6
  have hannstatic : AnnStatic wd (tbl + 64 * pre.length) p sp :=
    ⟨st.w56, st.p0, st.pann, hsp48, hspW, st.clenW, hhW, hdisjw48, hdisjp48⟩
  obtain ⟨k2, s2, hk2, hrun2, hs2rip, hs2r7, hs2r4, hs2r6, hs2r1, hf24_2, hf32_2,
    hf48_2, hlay2, hroom2, h12len2, gb2, hfr2, hw2, hlen2⟩ :=
    annval_loopB ac oc hannstatic hann wd.ann [] (by simp) s1 hannst
  have hrun12 : run im (k1 + k2) (Out.ok s) = Out.ok s2 := by
    rw [run_add im k1 k2, hrun1, hrun2]
  have hregs2 : RegsOk s2 := run_regs_ok hrun12 hregs
  have hs2r13 : s2.reg 13 = some pr.words.length := by
    rw [gb2 13 (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide)]; exact hs1r13
  have hs2r14 : s2.reg 14 = some tbl := by
    rw [gb2 14 (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide)]; exact hs1r14
  have hs2r5 : s2.reg 5 = some (rest.length + 1) := by
    rw [gb2 5 (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide)]; exact hs1r5
  have hfr02 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      s2.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; rw [hfr2 x n hx, hs1mem]
  have hfld2 : ∀ (off v : Nat), off + 8 ≤ 64 →
      rd64 s.mem (tbl + 64 * pre.length + off) = some v →
      rd64 s2.mem (tbl + 64 * pre.length + off) = some v := by
    intro off v hoff hv; unfold rd64
    rw [hfr02 (tbl + 64 * pre.length + off) 8 (by
      have hd := htab pre.length hj0; omega)]
    exact hv
  have heff2 : EffTable s2 := by
    intro q
    rw [hfr02 (aEff + 2 * q.code) 2 (by
      have hc := Prim_code_lt q
      cases hdisjEff48 with
      | inl hh => exact Or.inl (by omega)
      | inr hh => exact Or.inr (by omega))]
    exact heff q
  have hcsq2 : CodeSeq s2.mem cp wd.code :=
    CodeSeq_of_frame hcsq (fun x k _ _ => hfr02 x k (by omega))
  have hwords2 : ∀ n (hn : n < pr.words.length),
      rd64 s2.mem (tbl + n * 64) = some (pr.words[n]).ins ∧
      rd64 s2.mem (tbl + n * 64 + 8) = some (pr.words[n]).outs := by
    intro n hn
    obtain ⟨cpn, pn, g0, g8, -, -, -, -, -, -, -, -⟩ := htabS n hn
    have hd := htab n hn
    refine ⟨?_, ?_⟩
    · have e : tbl + n * 64 = tbl + 64 * n := by rw [Nat.mul_comm]
      rw [e]; unfold rd64; rw [hfr02 (tbl + 64 * n) 8 (by omega)]; exact g0
    · have e : tbl + n * 64 + 8 = tbl + 64 * n + 8 := by rw [Nat.mul_comm]
      rw [e]; unfold rd64; rw [hfr02 (tbl + 64 * n + 8) 8 (by omega)]; exact g8
  have hdisjC32 : cp + codeBytes wd.code ≤ sp - 32 ∨ sp ≤ cp := by
    cases hdisjC48 with
    | inl hh => exact Or.inl (by omega)
    | inr hh => exact Or.inr hh
  obtain ⟨k3, s3, hk3, hrun3, hinv3, gb3, hfr3, hw3, hlen3⟩ :=
    codewalk_entryB ac oc st hsp48 h0clause hinsW s2 hs2rip hs2r7 hs2r4
      hs2r1 (hregs2 9 (by decide) (by decide))
      (hregs2 12 (by decide) (by decide)) hs2r13 hs2r14 (by omega)
      (hfld2 0 _ (by omega) (by simpa using hf0))
      (hfld2 8 _ (by omega) hf8) (hfld2 16 _ (by omega) hf16) hf24_2 hf32_2
      (hfld2 40 _ (by omega) hf40) hf48_2 hlay2 hroom2 heff2 hcsq2 hwords2
      hdisjC32 hcbound himm hhb
  have hs3r5 : s3.reg 5 = some (rest.length + 1) := by
    rw [gb3 5 (by decide) (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide) (by decide)]; exact hs2r5
  have hfr03 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      s3.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; exact (hfr3 x n (by omega)).trans (hfr02 x n hx)
  have hw02 : wshape s2.mem = wshape s.mem := by rw [hw2, hs1mem]
  have hlen02 : s2.regs.length = s.regs.length := hlen2.trans hlen1
  have hw03 : wshape s3.mem = wshape s.mem := hw3.trans hw02
  have hlen03 : s3.regs.length = s.regs.length := hlen3.trans hlen02
  obtain ⟨k4, s4, hk4, hrun4, hs4rip, hs4r1, hs4r6, hs4r7, hs4r4, hfr4, hw4,
    hs4r5, hs4r13, hs4r14⟩ :=
    walk_loopB ac st wd.code 0 cp ⟨wd.ins, 0⟩ ⟨wd.ins, 0⟩ s3 hinv3 hwalk
  have hlen4 : s4.regs.length = s.regs.length :=
    (run_regs_length k4 hrun4).trans hlen03
  have hfr04 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      s4.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; exact (hfr4 x n (by omega)).trans (hfr03 x n hx)
  have hw04 : wshape s4.mem = wshape s.mem := hw4.trans hw03
  have hs4r5' : s4.reg 5 = some (rest.length + 1) := by rw [hs4r5]; exact hs3r5
  have hs4r13' : s4.reg 13 = some pr.words.length := by rw [hs4r13]; exact hinv3.r13
  have hs4r14' : s4.reg 14 = some tbl := by rw [hs4r14]; exact hinv3.r14
  obtain ⟨k5, s5, hk5, hrun5, hs5rip, hs5r7, hs5r5, gb5, hs5mem, hlen5⟩ :=
    WordDone_tripleB oc (tbl + 64 * pre.length) (rest.length + 1) (by omega)
      (by omega)
      (show tbl + 64 * pre.length + 64 < W by
        have : 64 * pre.length + 64 = 64 * (pre.length + 1) := by
          rw [Nat.mul_add, Nat.mul_one]
        omega)
      s4 hs4rip hs4r7 hs4r5'
  have hrunT : run im (k1 + k2 + k3 + k4 + k5) (Out.ok s) = Out.ok s5 := by
    rw [run_add im (k1 + k2 + k3 + k4) k5,
      run_add im (k1 + k2 + k3) k4, run_add im (k1 + k2) k3,
      run_add im k1 k2, hrun1, hrun2, hrun3, hrun4, hrun5]
  have hfrT : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
      s5.mem.rdN x n = s.mem.rdN x n := by
    intro x n hx; rw [hs5mem]; exact hfr04 x n hx
  have hwT : wshape s5.mem = wshape s.mem := by rw [hs5mem]; exact hw04
  have hlenT : s5.regs.length = s.regs.length := hlen5.trans hlen4
  have hs5r4 : s5.reg 4 = some sp := by rw [gb5 4 (by decide) (by decide)]; exact hs4r4
  have hs5r13 : s5.reg 13 = some pr.words.length := by
    rw [gb5 13 (by decide) (by decide)]; exact hs4r13'
  have hs5r14 : s5.reg 14 = some tbl := by
    rw [gb5 14 (by decide) (by decide)]; exact hs4r14'
  have hs5r5' : s5.reg 5 = some rest.length := by rw [hs5r5]; congr 1
  have hs5r7' : s5.reg 7 =
      some (tbl + 64 * (pr.words.length - rest.length)) := by
    rw [hs5r7]; congr 1
    have e : pr.words.length - rest.length = pre.length + 1 := by omega
    rw [e, Nat.mul_add, Nat.mul_one, Nat.add_assoc]
  have heff5 : EffTable s5 := by
    intro q
    rw [hfrT (aEff + 2 * q.code) 2 (by
      have hc := Prim_code_lt q
      cases hdisjEff48 with
      | inl hh => exact Or.inl (by omega)
      | inr hh => exact Or.inr (by omega))]
    exact heff q
  have hle_ws : k1 + k2 + k3 + k4 + k5 ≤ wordCost wd :=
    Nat.add_le_add (Nat.add_le_add (Nat.add_le_add (Nat.add_le_add hk1 hk2) hk3) hk4) hk5
  refine ⟨k1 + k2 + k3 + k4 + k5, s5, hle_ws, hrunT, hs5rip, hs5r5', hs5r7', hs5r13,
    hs5r14, hs5r4, by omega, run_regs_ok hrunT hregs,
    StackRoom_of_wshape hwT (by rw [hs5r4, hr4]) hroom, heff5,
    WordTab_of_frame htabS htab hfrT⟩

/-- Bounded refusal for a single word whenever `checkWord1 pr wd = false`. -/
theorem word_refuseB {im : Nat → Option (Ins × Nat)}
    (ac : AllCode im) (oc : OuterCode im) (rc : RefuseCode im) (cr : ChkRefuseCode im)
    {pr : Prog} {tbl sp : Nat} (hsp48 : 48 ≤ sp) (hspW : sp < W)
    (htblW : tbl + 64 * pr.words.length < W)
    (htab : ∀ j, j < pr.words.length →
      sp ≤ tbl + 64 * j ∨ tbl + 64 * j + 64 ≤ sp - 48)
    (hdisjEff48 : aEff + 114 ≤ sp - 48 ∨ sp ≤ aEff)
    {wd : Sporos.Word} (hwb : WordBound wd)
    (hbad : Sporos.checkWord1 pr wd = false) :
    ∀ (rest pre : List Sporos.Word), pr.words = pre ++ (wd :: rest) →
      ∀ s, OuterSt pr tbl sp (wd :: rest) s →
      ∃ m, m ≤ wordCost wd ∧ run im m (Out.ok s) = Out.exit 1 := by
  intro rest pre happ s hs
  obtain ⟨hrip, hr5, hr7, hr13, hr14, hr4, h16, hregs, hroom, heff, htabS⟩ := hs
  have hlenann : pr.words.length = pre.length + (rest.length + 1) := by
    rw [happ, List.length_append, List.length_cons]
  have hj0 : pre.length < pr.words.length := by omega
  have hget? : pr.words[pre.length]? = some wd := by
    rw [happ, List.getElem?_append_right (by simp)]
    simp
  have hget : pr.words[pre.length]'hj0 = wd :=
    Option.some.inj ((List.getElem?_eq_getElem hj0).symm.trans hget?)
  obtain ⟨cp, p, hf0, hf8, hf16, hf24, hf32, hf40, hf48, hcsq, hlay, hok⟩ :=
    htabS pre.length hj0
  rw [hget] at hf0 hf8 hf16 hf24 hf32 hcsq hlay hok
  obtain ⟨st, hinsW, hhW, himm, hhb, hdisjC48, hcbound, hdisjw48, hdisjp48⟩ := hok
  have hr7w : s.reg 7 = some (tbl + 64 * pre.length) := by
    have hw : tbl + 64 * (pr.words.length - (wd :: rest).length) =
        tbl + 64 * pre.length := by
      rw [List.length_cons]
      congr 1; congr 1; omega
    rw [hw] at hr7
    exact hr7
  have houtW : wd.outs < W := by
    have h1 := hwb.2.1
    have h2 : (256 : Nat) ^ 4 < W := by decide
    omega
  by_cases hnc : wd.code.length ≠ 0
  case neg =>
    have hcl0 : wd.code.length = 0 := by omega
    have hr5' : s.reg 5 = some (rest.length + 1) := by
      simpa only [List.length_cons] using hr5
    obtain ⟨m, hm, hrun⟩ := CheckWord_empty_exit1B oc rc (tbl + 64 * pre.length)
      (rest.length + 1) sp st.w56 (by omega) hsp48 hspW s hrip hr5' hr7w hr4
      (by rw [← hcl0]; exact hf24) hregs hroom
    exact ⟨m, Nat.le_trans hm (wordCost_ge_12 wd), hrun⟩
  case pos =>
    have hlenW : pr.words.length < W := by
      have h1 : pr.words.length ≤ 64 * pr.words.length :=
        Nat.le_mul_of_pos_left _ (by decide)
      omega
    obtain ⟨k1, s1, hk1, hrun1, hs1rip, hs1r6, hs1r1, -, hs1r5, hs1r7, gb1, hs1mem, hlen1⟩ :=
      CheckWord_headB oc (tbl + 64 * pre.length) (rest.length + 1)
        wd.code.length wd.ann.length p st.w56 (by omega) hnc s hrip hr5 hr7w
        hf24 hf48 hf32
    have hs1r4 : s1.reg 4 = some sp := by
      rw [gb1 4 (by decide) (by decide) (by decide)]; exact hr4
    have hregs1 : RegsOk s1 := run_regs_ok hrun1 hregs
    have hs1r13 : s1.reg 13 = some pr.words.length := by
      rw [gb1 13 (by decide) (by decide) (by decide)]; exact hr13
    have hs1r14 : s1.reg 14 = some tbl := by
      rw [gb1 14 (by decide) (by decide) (by decide)]; exact hr14
    have hannst : AnnSt wd (tbl + 64 * pre.length) p sp wd.ann s1 := by
      refine ⟨hs1rip, hs1r7, ?_, hs1r1, hs1r4,
        by rw [hs1mem]; exact hf24, by rw [hs1mem]; exact hf32,
        by rw [hs1mem]; exact hf48, AnnLayout_of_mem hs1mem hlay, by omega,
        StackRoom_of_mem hs1mem (by rw [hs1r4, hr4]) hroom⟩
      simpa using hs1r6
    have hannstatic : AnnStatic wd (tbl + 64 * pre.length) p sp :=
      ⟨st.w56, st.p0, st.pann, hsp48, hspW, st.clenW, hhW, hdisjw48, hdisjp48⟩
    cases hann : Sporos.annOk wd with
    | false =>
      obtain ⟨k2, hk2, hrun2⟩ :=
        annval_refuse_loopB ac oc rc cr hannstatic hann wd.ann [] (by simp)
          (fun _ hx => by simp at hx) s1 hannst hregs1
      exact ⟨k1 + k2,
        Nat.le_trans (Nat.add_le_add hk1 hk2)
          (Nat.le_trans (Nat.le_add_right _ _)
            (Nat.le_trans (Nat.le_add_right _ _) (Nat.le_add_right _ _))),
        by rw [run_add im k1 k2, hrun1]; exact hrun2⟩
    | true =>
      obtain ⟨k2, s2, hk2, hrun2, hs2rip, hs2r7, hs2r4, hs2r6, hs2r1, hf24_2, hf32_2,
        hf48_2, hlay2, hroom2, h12len2, gb2, hfr2, hw2, hlen2⟩ :=
        annval_loopB ac oc hannstatic hann wd.ann [] (by simp) s1 hannst
      have hrun12 : run im (k1 + k2) (Out.ok s) = Out.ok s2 := by
        rw [run_add im k1 k2, hrun1, hrun2]
      have hregs2 : RegsOk s2 := run_regs_ok hrun12 hregs
      have hs2r13 : s2.reg 13 = some pr.words.length := by
        rw [gb2 13 (by decide) (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide)]; exact hs1r13
      have hs2r14 : s2.reg 14 = some tbl := by
        rw [gb2 14 (by decide) (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide) (by decide)]; exact hs1r14
      have hfr02 : ∀ x n, x + n ≤ sp - 48 ∨ sp ≤ x →
          s2.mem.rdN x n = s.mem.rdN x n := by
        intro x n hx; rw [hfr2 x n hx, hs1mem]
      have hfld2 : ∀ (off v : Nat), off + 8 ≤ 64 →
          rd64 s.mem (tbl + 64 * pre.length + off) = some v →
          rd64 s2.mem (tbl + 64 * pre.length + off) = some v := by
        intro off v hoff hv; unfold rd64
        rw [hfr02 (tbl + 64 * pre.length + off) 8 (by
          have hd := htab pre.length hj0; omega)]
        exact hv
      have heff2 : EffTable s2 := by
        intro q
        rw [hfr02 (aEff + 2 * q.code) 2 (by
          have hc := Prim_code_lt q
          cases hdisjEff48 with
          | inl hh => exact Or.inl (by omega)
          | inr hh => exact Or.inr (by omega))]
        exact heff q
      have hcsq2 : CodeSeq s2.mem cp wd.code :=
        CodeSeq_of_frame hcsq (fun x k _ _ => hfr02 x k (by omega))
      have hwords2 : ∀ n (hn : n < pr.words.length),
          rd64 s2.mem (tbl + n * 64) = some (pr.words[n]).ins ∧
          rd64 s2.mem (tbl + n * 64 + 8) = some (pr.words[n]).outs := by
        intro n hn
        obtain ⟨cpn, pn, g0, g8, -, -, -, -, -, -, -, -⟩ := htabS n hn
        have hd := htab n hn
        refine ⟨?_, ?_⟩
        · have e : tbl + n * 64 = tbl + 64 * n := by rw [Nat.mul_comm]
          rw [e]; unfold rd64; rw [hfr02 (tbl + 64 * n) 8 (by omega)]; exact g0
        · have e : tbl + n * 64 + 8 = tbl + 64 * n + 8 := by rw [Nat.mul_comm]
          rw [e]; unfold rd64; rw [hfr02 (tbl + 64 * n + 8) 8 (by omega)]; exact g8
      have hdisjC32 : cp + codeBytes wd.code ≤ sp - 32 ∨ sp ≤ cp := by
        cases hdisjC48 with
        | inl hh => exact Or.inl (by omega)
        | inr hh => exact Or.inr hh
      cases h0clause : (match wd.ann.lookup 0 with
          | some a => a == (⟨wd.ins, 0⟩ : Height) | none => true) with
      | false =>
        have hlen2' : 16 ≤ s2.regs.length := by rw [hlen2, hlen1]; exact h16
        obtain ⟨k3, hk3, hrun3⟩ :=
          codewalk_entry_refuseB ac oc rc cr st hsp48 h0clause hinsW hhW s2 hs2rip hs2r7
            hs2r4 hs2r1 (hregs2 9 (by decide) (by decide))
            (hregs2 12 (by decide) (by decide)) hlen2' hregs2
            (hfld2 0 _ (by omega) (by simpa using hf0)) hf32_2
            (hfld2 40 _ (by omega) hf40) hf48_2 hlay2 hroom2
        exact ⟨k1 + k2 + k3, wordCost_ge_cw wd.ann.length wd.code.length k1 k2 k3 hk1 hk2 hk3,
          by rw [run_add im (k1 + k2) k3, hrun12]; exact hrun3⟩
      | true =>
        have hwalk : Sporos.walk pr wd wd.code 0 ⟨wd.ins, 0⟩ = false := by
          have hne : (!wd.code.isEmpty) = true := by
            cases hc : wd.code with
            | nil => simp [hc] at hnc
            | cons _ _ => rfl
          have h1 := hbad
          unfold Sporos.checkWord1 at h1
          cases hl0 : wd.ann.lookup 0 with
          | none =>
            rw [hne, hl0, hann] at h1
            exact h1
          | some a =>
            have heqa : a = ⟨wd.ins, 0⟩ := height_eq_of_beq (by simpa [hl0] using h0clause)
            subst heqa
            rw [hne, hl0, hann] at h1
            dsimp only at h1
            have hb : (({ d := wd.ins, a := 0 } : Height) == { d := wd.ins, a := 0 }) = true := by
              simp [BEq.beq]
            rw [hb] at h1
            exact h1
        obtain ⟨k3, s3, hk3, hrun3, hinv3, -, -, -, -⟩ :=
          codewalk_entryB ac oc st hsp48 h0clause hinsW s2 hs2rip hs2r7 hs2r4
            hs2r1 (hregs2 9 (by decide) (by decide))
            (hregs2 12 (by decide) (by decide)) hs2r13 hs2r14 (by omega)
            (hfld2 0 _ (by omega) (by simpa using hf0))
            (hfld2 8 _ (by omega) hf8) (hfld2 16 _ (by omega) hf16) hf24_2 hf32_2
            (hfld2 40 _ (by omega) hf40) hf48_2 hlay2 hroom2 heff2 hcsq2 hwords2
            hdisjC32 hcbound himm hhb
        have hrun123 : run im (k1 + k2 + k3) (Out.ok s) = Out.ok s3 := by
          rw [run_add im (k1 + k2) k3, hrun12, hrun3]
        have hregs3 : RegsOk s3 := run_regs_ok hrun3 hregs2
        obtain ⟨k4, hk4, hrun4⟩ :=
          walk_refuseB ac rc cr st hsp48 hhW houtW wd.code 0 cp ⟨wd.ins, 0⟩ ⟨wd.ins, 0⟩
            s3 hinv3 hregs3 hwalk
        exact ⟨k1 + k2 + k3 + k4,
          Nat.le_trans (Nat.add_le_add (Nat.add_le_add (Nat.add_le_add hk1 hk2) hk3) hk4)
            (Nat.le_add_right _ 3),
          by rw [run_add im (k1 + k2 + k3) k4, hrun123]; exact hrun4⟩

/-- Bounded refusal loop over `pr.words`: executes valid prefix words with `word_step_oneB`
and stops on the first failing word with `word_refuseB`. -/
theorem words_refuse_loopB {im : Nat → Option (Ins × Nat)}
    (ac : AllCode im) (oc : OuterCode im) (rc : RefuseCode im) (cr : ChkRefuseCode im)
    {pr : Prog} {tbl sp : Nat} (hsp48 : 48 ≤ sp) (hspW : sp < W)
    (htblW : tbl + 64 * pr.words.length < W)
    (htab : ∀ j, j < pr.words.length →
      sp ≤ tbl + 64 * j ∨ tbl + 64 * j + 64 ≤ sp - 48)
    (hdisjEff48 : aEff + 114 ≤ sp - 48 ∨ sp ≤ aEff)
    (hwb : ∀ wd ∈ pr.words, WordBound wd) :
    ∀ (rest pre : List Sporos.Word),
      pr.words = pre ++ rest →
      (∃ wd ∈ rest, Sporos.checkWord1 pr wd = false) →
      ∀ s, OuterSt pr tbl sp rest s →
      ∃ m, m ≤ (rest.map wordCost).sum ∧ run im m (Out.ok s) = Out.exit 1 := by
  intro rest
  induction rest with
  | nil =>
    intro _ _ hex _ _
    simp at hex
  | cons wd rest' ih =>
    intro pre happ hex s hs
    cases hwd : Sporos.checkWord1 pr wd with
    | false =>
      have hwbwd : WordBound wd :=
        hwb wd (by rw [happ]; exact List.mem_append_right _ List.mem_cons_self)
      obtain ⟨m, hm, hrun⟩ :=
        word_refuseB ac oc rc cr hsp48 hspW htblW htab hdisjEff48 hwbwd hwd rest' pre happ s hs
      exact ⟨m, (by simp only [List.map_cons, List.sum_cons]; omega), hrun⟩
    | true =>
      obtain ⟨x, hx, hxf⟩ := hex
      have hx' : ∃ y ∈ rest', Sporos.checkWord1 pr y = false := by
        simp only [List.mem_cons] at hx
        rcases hx with rfl | hx
        · rw [hwd] at hxf; cases hxf
        · exact ⟨x, hx, hxf⟩
      obtain ⟨k1, s1, hk1, hrun1, hs1⟩ :=
        word_step_oneB ac oc hsp48 hspW htblW htab hdisjEff48 wd rest' pre happ hwd s hs
      have happ' : pr.words = (pre ++ [wd]) ++ rest' := by
        rw [happ, List.append_assoc, List.singleton_append]
      obtain ⟨k2, hk2, hrun2⟩ := ih (pre ++ [wd]) happ' hx' s1 hs1
      exact ⟨k1 + k2, (by simp only [List.map_cons, List.sum_cons]; omega), by
        rw [run_add im k1 k2, hrun1]; exact hrun2⟩

/-- Bounded refusal direction of the checker from `check_start`. -/
theorem checker_refuses_from_startB {im : Nat → Option (Ins × Nat)}
    (ac : AllCode im) (oc : OuterCode im) (rc : RefuseCode im) (cr : ChkRefuseCode im)
    {pr : Prog} {tbl sp : Nat} (hsp48 : 48 ≤ sp) (hspW : sp < W)
    (htblW : tbl + 64 * pr.words.length < W)
    (htab : ∀ j, j < pr.words.length →
      sp ≤ tbl + 64 * j ∨ tbl + 64 * j + 64 ≤ sp - 48)
    (hdisjEff48 : aEff + 114 ≤ sp - 48 ∨ sp ≤ aEff)
    (hwb : ∀ wd ∈ pr.words, WordBound wd)
    (hex : ∃ wd ∈ pr.words, Sporos.checkWord1 pr wd = false) :
    ∀ s, s.rip = aCheckStart → s.reg 14 = some tbl →
      s.reg 13 = some pr.words.length → s.reg 4 = some sp →
      16 ≤ s.regs.length → RegsOk s → StackRoom s 6 → EffTable s →
      WordTab pr tbl sp s →
      ∃ m, m ≤ (pr.words.map wordCost).sum + 10 ∧ run im m (Out.ok s) = Out.exit 1 := by
  intro s hrip hr14 hr13 hr4 h16 hregs hroom heff htabS
  obtain ⟨k1, s1, hk1, hrun1, hs1rip, hs1r7, hs1r5, gb1, hs1mem, hlen1⟩ :=
    CheckStart_tripleB oc s hrip hr14 hr13
  have hs1 : OuterSt pr tbl sp pr.words s1 := by
    refine ⟨hs1rip, hs1r5, by simpa using hs1r7,
      by rw [gb1 13 (by decide) (by decide)]; exact hr13,
      by rw [gb1 14 (by decide) (by decide)]; exact hr14,
      by rw [gb1 4 (by decide) (by decide)]; exact hr4, by omega,
      run_regs_ok hrun1 hregs,
      StackRoom_of_mem hs1mem (by rw [gb1 4 (by decide) (by decide)]) hroom,
      by intro q; rw [hs1mem]; exact heff q,
      by apply WordTab_of_frame htabS htab; intro x n _; rw [hs1mem]⟩
  obtain ⟨k2, hk2, hrun2⟩ :=
    words_refuse_loopB ac oc rc cr hsp48 hspW htblW htab hdisjEff48 hwb pr.words []
      (by simp) hex s1 hs1
  exact ⟨k1 + k2, by omega, by rw [run_add im k1 k2, hrun1]; exact hrun2⟩

set_option maxHeartbeats 3200000 in
set_option maxRecDepth 131072 in
/-- Prefix classification adding the `main`-word `ins = 0` invariant to the handoff tuple. -/
theorem stage1_prefix_classify_main {f path argv0 stdin : List Nat}
    {oracle : List Choice} {top : Nat}
    (hinit : Stage1InitOk f path argv0 top)
    (hadm : OracleAdmits
      (stage1Init f path argv0 stdin oracle top))
    (hnh : ¬ HitsNomem
      (stage1Init f path argv0 stdin oracle top)) :
    (∃ k, k ≤ 2000 + 2000 * f.length ∧
      run stage1Im k
        (Out.ok (stage1Init f path argv0 stdin oracle top)) =
        Out.exit 1) ∨
    (∃ (pr : Prog) (T sp : Nat) (s' : X86Int.St) (K : Nat),
      K ≤ 2000 + 2000 * f.length ∧
      Sporos.decode f = some pr ∧ ProgBound pr ∧ f = pr.enc ∧
      run stage1Im K
        (Out.ok (stage1Init f path argv0 stdin oracle top)) =
        Out.ok s' ∧
      s'.rip = base + 0x452 ∧ s'.reg 14 = some T ∧
      s'.reg 13 = some pr.words.length ∧ s'.reg 4 = some sp ∧
      48 ≤ sp ∧ sp < W ∧ T + 64 * pr.words.length < W ∧
      (∀ j, j < pr.words.length →
        sp ≤ T + 64 * j ∨ T + 64 * j + 64 ≤ sp - 48) ∧
      (aEff + 114 ≤ sp - 48 ∨ sp ≤ aEff) ∧
      StackRoom s' 6 ∧ RegsOk s' ∧ s'.regs.length = 16 ∧
      WordTab pr T sp s' ∧ EffTable s' ∧
      NoHitUpTo (stage1Init f path argv0 stdin oracle top) K ∧
      (∃ hj : pr.main < pr.words.length, (pr.words[pr.main]'hj).ins = 0)) := by
  have hinit0 := hinit
  obtain ⟨hfB, -, -, hpath0, -, hpathlen, -, htop47, hleInit,
    hstkhi⟩ := hinit
  have hc := stage1_parserCode
  have hW : W = 2 ^ 64 := rfl
  rcases loader_init_hit hinit0 hadm with hex | hld
  · exact absurd hex hnh
  · obtain ⟨kL, A', cap', c2, buf', s1, hkL, hrunL, hs1rip, hs1r3,
        hs1r12, hs1r13, hs1r14, hs1r4, hs1r0, hs1r10, hs1r8,
        hs1r9, hs1mem, hc2b, hc2hi0, hc2rw, hbufb, hbufhi, hbufrw,
        htake, hdisj1, hdisj2, hApage, hApos, hAbound, hcappage,
        hcapge, hs1fds, hs1len, hs1regs, hs1nh⟩ := hld
    have hsp0eq : (X86Env.initStack top [argv0, path]).2 =
        c2.base + 4096 := by
      have h1 := initStack_sp top argv0 path
      rw [← hc2b] at h1
      have hsr : X86Env.stackRoom = 4096 := rfl
      rw [hsr] at h1
      exact h1
    have hdatalen : 8192 ≤
        (X86Env.initStack top [argv0, path]).1.data.length := by
      rw [initStack_data_length]
      have hp : 4096 ≤ X86Env.pages
          (56 + ((argv0.length + 1) + (path.length + 1)) + 1) := by
        unfold X86Env.pages X86Env.page
        have h1 : 1 ≤ (56 + ((argv0.length + 1) +
            (path.length + 1)) + 1 + 4095) / 4096 :=
          Nat.div_pos (by omega) (by decide)
        omega
      have hsr : X86Env.stackRoom = 4096 := rfl
      omega
    have hc2hi : c2.hi = c2.base +
        (X86Env.initStack top [argv0, path]).1.data.length := by
      have e : (X86Env.initStack top [argv0, path]).1.hi =
          (X86Env.initStack top [argv0, path]).1.base +
            (X86Env.initStack top [argv0, path]).1.data.length :=
        rfl
      rw [hc2hi0, e, ← hc2b]
    have hc2top : c2.hi = top := by rw [hc2hi0]; exact hstkhi
    have hdisjInit : base + stage1Code.length ≤ c2.base := by
      have h := hleInit
      rw [phVaddr_eq, phMemsz_eq, ← hc2b] at h
      exact h
    have hspW : (X86Env.initStack top [argv0, path]).2 + 24 < W := by
      omega
    have hroom1 : StackRoom s1 10 :=
      stackRoom_of_c2 hs1mem hs1r4 hc2rw hdisjInit (by omega)
        (by omega) (by omega)
    have hcaplen : buf'.data.length = cap' := by
      have h := hbufhi
      unfold X86Env.Region.hi at h
      omega
    have hLcap : f.length ≤ cap' := by
      have h1 : (buf'.data.take f.length).length = f.length := by
        rw [htake]
      rw [List.length_take, hcaplen] at h1
      omega
    have hdisj1' :
        (⟨base, stage1Code, false⟩ : Region).hi ≤ A' ∨
          A' + cap' ≤ (⟨base, stage1Code, false⟩ : Region).base :=
      hdisj1
    have hrdNfile : s1.mem.rdN A' f.length = some f := by
      rw [hs1mem]
      have h := rdN_third_take hdisj1 hdisj2 hbufb
        (by omega : f.length ≤ buf'.data.length) hLcap
      rwa [htake] at h
    have hwin : ∃ bs, s1.mem.rdN A' 20 = some bs := by
      rw [hs1mem]
      exact ⟨buf'.data.take 20,
        rdN_third_take hdisj1 hdisj2 hbufb (by omega) (by omega)⟩
    have hrd : ∀ a, A' + 16 ≤ a → a + 4 ≤ A' + f.length →
        ∃ d, rd32 s1.mem a = some d := by
      intro a ha hb
      have hs := rdN_slice hrdNfile (o := a - A') (k := 4)
        (by omega)
      rw [show A' + (a - A') = a by omega] at hs
      exact ⟨leVal ((f.drop (a - A')).take 4), by
        show (s1.mem.rdN a 4).map leVal = _
        rw [hs]
        rfl⟩
    have hAe : A' + f.length < W := by omega
    rcases parser_front_classify_nh hs1rip hs1r3 hs1r13 hAe hwin
        hrd hs1len hs1r4 (by omega) hspW hs1regs
        (stackRoom_mono hroom1 (by decide)) with hfexit | hfprog
    · obtain ⟨kF, hkF, hrunF, hnhF⟩ := hfexit
      exact Or.inl ⟨kL + kF, by omega, by
        rw [run_add, hrunL, hrunF]⟩
    · obtain ⟨bs, ds, c, kF, s4, hkF, h20, hL16, hmagic, hdslen,
          hseq, hcL, hguard, hrunF, hs4rip, hs4r12, hs4r10, hnw0,
          hs4r1, hs4r6, hs4r15, hs4r3, hs4r13, hs4r4, hs4fr,
          hs4mem, hs4len, hs4k, hs4nh⟩ := hfprog
      have hrunLF : run stage1Im (kL + kF)
          (Out.ok (stage1Init f path argv0 stdin oracle top)) =
          Out.ok s4 := by
        rw [run_add, hrunL, hrunF]
      have hnhLF :
          NoHitUpTo (stage1Init f path argv0 stdin oracle top)
            (kL + kF) :=
        noHitUpTo_add hs1nh hrunL hs4nh
      have hnw32 : (leVal ((bs.drop 8).take 4) % 2 ^ 32) < 2 ^ 32 := Nat.mod_lt _ (by decide)
      have hs4regs : RegsOk s4 := run_regs_ok hrunF hs1regs
      have hs4room : StackRoom s4 2 :=
        StackRoom_of_mem hs4mem (by rw [hs4r4, hs1r4])
          (stackRoom_mono hroom1 (by decide))
      have hs4r7 : ∃ v, s4.reg 7 = some v :=
        hs4regs 7 (by decide) (by decide)
      have hs4r9 : ∃ v, s4.reg 9 = some v :=
        hs4regs 9 (by decide) (by decide)
      obtain ⟨m2a, t1, hrunt1, ht1rip, ht1r0, ht1r7, ht1r6, ht1r2,
          ht1r10, ht1r8, ht1r9, ht1mem, ht1sh, ht1k, ht1len⟩ :=
        parser_mmap1_to_syscall hs4rip hs4r6 hs4r10 hs4r4 hs4r7
          hs4r9 hs4room (by omega) hspW hnw32 hnw0 hs4len
      have hrunt1tot : run stage1Im (kL + kF + 16)
          (Out.ok (stage1Init f path argv0 stdin oracle top)) =
          Out.ok t1 := by
        rw [show kL + kF + 16 = (kL + kF) + 16 by omega, run_add,
          hrunLF, hrunt1]
      have ht1leg : OracleLegalAt t1 :=
        hadm _ _ hrunt1tot
      have ht1memsys : AtMemSyscall t1 :=
        ⟨by rw [ht1rip]; exact hc.p21e, Or.inl ht1r0⟩
      rcases head_cases_of_legal_mmap ht1leg ht1memsys ht1r0
          ht1r7 ht1r6 ht1r2 ht1r10 ht1r8 ht1r9
          (by omega) with hat1 | hnomem1
      · obtain ⟨T, rest, hor1⟩ := hat1
        have hs4head : s4.k.oracle.head? = some (.at T) := by
          rw [← ht1k, hor1]
          rfl
        obtain ⟨hTpage, hTpos, hTbound, hTfree⟩ :=
          place_facts_of_legal_at ht1leg ht1memsys ht1r0 ht1r7
            ht1r6 ht1r2 ht1r10 ht1r8 ht1r9 (by omega) hor1
        have hTfree4 : s4.mem.free T (X86Env.pages (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32))) =
            true := by
          rw [ht1mem] at hTfree
          rwa [free_of_wshape ht1sh] at hTfree
        obtain ⟨s5, m2, hrun5, hs5rip, hs5r14, hs5r4, hs5mem,
            hm2fr, hslot5, hs5fr, hs5len, hs5k, hm2sh, hs5nh⟩ :=
          parser_mmap1_ok_nh hs4rip hs4r6 hs4r10 hs4r4 hs4r7
            hs4r9 hs4room (by omega) hspW hnw32 hnw0 hs4head
            hTpage hTpos hTbound hTfree4 hs4len
        have hrunLF5 : run stage1Im (kL + kF + 20)
            (Out.ok (stage1Init f path argv0 stdin oracle top)) =
            Out.ok s5 := by
          rw [show kL + kF + 20 = (kL + kF) + 20 by omega,
            run_add, hrunLF, hrun5]
        have hnhLF5 :
            NoHitUpTo (stage1Init f path argv0 stdin oracle top)
              (kL + kF + 20) :=
          noHitUpTo_add hnhLF hrunLF hs5nh
        have hs5r : ∀ i v, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 →
            i ≠ 7 → i ≠ 8 → i ≠ 9 → i ≠ 10 → i ≠ 11 → i ≠ 14 →
            s4.reg i = some v → s5.reg i = some v := by
          intro i v h0 h1 h2 h4 h6 h7 h8 h9 h10 h11 h14 hv
          rw [hs5fr i h0 h1 h2 h4 h6 h7 h8 h9 h10 h11 h14]
          exact hv
        have hs5r13 : s5.reg 13 = some f.length :=
          hs5r 13 _ (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide) hs4r13
        have hs5r3 : s5.reg 3 = some A' :=
          hs5r 3 _ (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide) hs4r3
        have hs5r12 : s5.reg 12 = some (leVal ((bs.drop 4).take 4) % 2 ^ 32) :=
          hs5r 12 _ (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide) hs4r12
        have hs5r15 : s5.reg 15 = some (A' + f.length) :=
          hs5r 15 _ (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide) hs4r15
        have hs5regs : RegsOk s5 :=
          run_regs_ok hrun5 hs4regs
        have hs5r7 : ∃ v, s5.reg 7 = some v :=
          hs5regs 7 (by decide) (by decide)
        have hs5r9 : ∃ v, s5.reg 9 = some v :=
          hs5regs 9 (by decide) (by decide)
        have happ : ∀ (m : Mem) (r : Region),
            wshape (m ++ [r]) =
              wshape m ++ [(r.base, r.hi, r.rw)] := by
          intro m r
          simp [wshape]
        have hsh5 : wshape s5.mem =
            wshape ([⟨base, stage1Code, false⟩, c2, buf'] ++
              [⟨T, List.replicate (X86Env.pages (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32))) 0,
                true⟩]) := by
          rw [hs5mem, happ, hm2sh, hs4mem, hs1mem]
          simp [wshape]
        have hs5room : StackRoom s5 1 :=
          stackRoom_of_wshape_c2 hsh5 hs5r4 hc2rw hdisjInit
            (by omega) (by omega) (by omega)
        obtain ⟨m1a, t2, hrunt2, ht2rip, ht2r0, ht2r7, ht2r6,
            ht2r2, ht2r10, ht2r8, ht2r9, ht2r4, ht2mem, ht2sh,
            ht2k, ht2len⟩ :=
          parser_mmap2_to_syscall hs5rip hs5r13 hs5r4 hs5r7
            hs5r9 hs5room (by omega) (by omega) (by omega) hs5len
        have hrunt2tot : run stage1Im (kL + kF + 20 + 8)
            (Out.ok (stage1Init f path argv0 stdin oracle
              top)) = Out.ok t2 := by
          rw [show kL + kF + 20 + 8 = (kL + kF + 20) + 8 by omega,
            run_add, hrunLF5, hrunt2]
        have ht2leg : OracleLegalAt t2 := hadm _ _ hrunt2tot
        have ht2memsys : AtMemSyscall t2 :=
          ⟨by rw [ht2rip]; exact hc.p249, Or.inl ht2r0⟩
        rcases head_cases_of_legal_mmap ht2leg ht2memsys ht2r0
            ht2r7 ht2r6 ht2r2 ht2r10 ht2r8 ht2r9
            (by omega) with hat2 | hnomem2
        · obtain ⟨S, rest2, hor2⟩ := hat2
          have hs5head : s5.k.oracle.head? = some (.at S) := by
            rw [← ht2k, hor2]
            rfl
          obtain ⟨hSpage, hSpos, hSbound, hSfree⟩ :=
            place_facts_of_legal_at ht2leg ht2memsys ht2r0 ht2r7
              ht2r6 ht2r2 ht2r10 ht2r8 ht2r9 (by omega) hor2
          have hSfree5 : s5.mem.free S
              (X86Env.pages f.length) = true := by
            rw [ht2mem] at hSfree
            rwa [free_of_wshape ht2sh] at hSfree
          obtain ⟨s6, m1, hrun6, hs6rip, hs6r11, hs6r4, hs6mem,
              hm1fr, hs6fr, hs6len, hs6k, hm1sh, hs6nh⟩ :=
            parser_mmap2_ok_nh hs5rip hs5r13 hs5r4 hs5r7 hs5r9
              hs5room (by omega) (by omega) (by omega) hs5head
              hSpage hSpos hSbound hSfree5 hs5len
          have hrunLF6 : run stage1Im (kL + kF + 20 + 12)
              (Out.ok (stage1Init f path argv0 stdin oracle
                top)) = Out.ok s6 := by
            rw [show kL + kF + 20 + 12 = (kL + kF + 20) + 12 by
              omega, run_add, hrunLF5, hrun6]
          have hnhLF6 :
              NoHitUpTo (stage1Init f path argv0 stdin oracle
                top) (kL + kF + 20 + 12) :=
            noHitUpTo_add hnhLF5 hrunLF5 hs6nh
          have hsh6 : wshape s6.mem =
              wshape ([⟨base, stage1Code, false⟩, c2, buf'] ++
                [⟨T, List.replicate (X86Env.pages (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32))) 0,
                  true⟩,
                 ⟨S, List.replicate (X86Env.pages f.length) 0,
                  true⟩]) := by
            rw [hs6mem, happ, hm1sh, hs5mem, happ, hm2sh, hs4mem,
              hs1mem]
            simp [wshape]
          have hrdNfile6 : s6.mem.rdN A' f.length = some f := by
            have hdisjF : A' + f.length ≤
                (X86Env.initStack top [argv0, path]).2 + 24 - 16 ∨
                (X86Env.initStack top [argv0, path]).2 + 24 ≤
                  A' := by
              rcases hdisj2 with h | h
              · exact Or.inr (by omega)
              · exact Or.inl (by omega)
            have hm2rd : m2.rdN A' f.length = some f := by
              rw [hm2fr A' f.length hdisjF]
              rw [hs4mem]
              exact hrdNfile
            have hm1rd : m1.rdN A' f.length = some f := by
              have hdisjF1 : A' + f.length ≤
                  ((X86Env.initStack top [argv0, path]).2 + 24 -
                    8) - 8 ∨
                  (X86Env.initStack top [argv0, path]).2 + 24 -
                    8 ≤ A' := by omega
              rw [hm1fr A' f.length hdisjF1, hs5mem]
              exact rdN_append_left hm2rd
            rw [hs6mem]
            exact rdN_append_left hm1rd
          have h206 : s6.mem.rdN A' 20 = some bs := by
            have hdisj20 : A' + 20 ≤
                (X86Env.initStack top [argv0, path]).2 + 24 - 16 ∨
                (X86Env.initStack top [argv0, path]).2 + 24 ≤
                  A' := by
              rcases hdisj2 with h | h
              · exact Or.inr (by omega)
              · exact Or.inl (by omega)
            have hm2rd : m2.rdN A' 20 = some bs := by
              rw [hm2fr A' 20 hdisj20, hs4mem]
              exact h20
            have hm1rd : m1.rdN A' 20 = some bs := by
              have hdisj201 : A' + 20 ≤
                  ((X86Env.initStack top [argv0, path]).2 + 24 -
                    8) - 8 ∨
                  (X86Env.initStack top [argv0, path]).2 + 24 -
                    8 ≤ A' := by omega
              rw [hm1fr A' 20 hdisj201, hs5mem]
              exact rdN_append_left hm2rd
            rw [hs6mem]
            exact rdN_append_left hm1rd
          have hsplevel : (X86Env.initStack top [argv0, path]).2 +
              24 - 8 = c2.base + 4112 := by omega
          have hcur6 : rd64 s6.mem
              ((X86Env.initStack top [argv0, path]).2 + 24 - 8) =
              some c := by
            obtain ⟨bs8, hbs8, hle8⟩ : ∃ bs, s5.mem.rdN
                ((X86Env.initStack top [argv0, path]).2 + 24 - 8)
                8 = some bs ∧ leVal bs = c % W := by
              have h2 := hslot5
              unfold rd64 at h2
              exact Option.map_eq_some_iff.mp h2
            have h8 : s6.mem.rdN
                ((X86Env.initStack top [argv0, path]).2 + 24 - 8)
                8 = some bs8 := by
              rw [hs6mem]
              apply rdN_append_left
              rw [hm1fr _ _ (Or.inr (Nat.le_refl _))]
              exact hbs8
            unfold rd64
            rw [h8]
            simp only [Option.map_some, hle8]
            rw [Nat.mod_eq_of_lt (show c < W by omega)]
          have hnwv6 : ∃ v, rd64 s6.mem (A' + 8) = some v ∧
              v % 2 ^ 32 = (leVal ((bs.drop 8).take 4) % 2 ^ 32) := by
            have hs := rdN_slice h206 (o := 8) (k := 8)
              (by decide)
            have hbslen : bs.length = 20 := rdN_length h20
            have hlen8 : ((bs.drop 8).take 8).length = 8 := by
              simp [List.length_take, List.length_drop, hbslen]
            refine ⟨leVal ((bs.drop 8).take 8), ?_, ?_⟩
            · show (s6.mem.rdN (A' + 8) 8).map leVal =
                some (leVal ((bs.drop 8).take 8))
              simp [hs]
            · have h1 := leVal_first4_mod hlen8
              rw [h1]
              have htake : ((bs.drop 8).take 8).take 4 =
                  (bs.drop 8).take 4 := by
                rw [List.take_take]
                rfl
              rw [htake]
          have hrdstk6 : ∀ a : Nat, c2.base ≤ a → a < c2.hi →
              (s6.mem.rd a).isSome := by
            intro a h1 h2
            apply rd_isSome_of_find
            have href : ∃ ii r,
                Mem.find ([⟨base, stage1Code, false⟩, c2, buf'] ++
                  [⟨T, List.replicate (X86Env.pages (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32))) 0,
                    true⟩,
                   ⟨S, List.replicate (X86Env.pages f.length) 0,
                    true⟩]) a 0 = some (ii, r) ∧ r.rw = true :=
              ⟨1, c2, find_append_left _ _ a 0
                (find_c2_of_three hdisjInit h1 h2), hc2rw⟩
            obtain ⟨ii, r, hf, -⟩ := find_rw_of_wshape hsh6 href
            exact ⟨ii, r, hf⟩
          have hpop6 : ∀ off : Nat, off ≤ 48 →
              ∃ w, rd64 s6.mem
                ((X86Env.initStack top [argv0, path]).2 + 24 - 8 +
                  off) = some w := by
            intro off hoff
            refine rd64_some_of_forall_rd (fun j hj => ?_)
            apply hrdstk6 <;> omega
          have hwstk6 : ∀ a : Nat, c2.base ≤ a → a + 8 ≤ c2.hi →
              ∀ v, (wr64 s6.mem a v).isSome := by
            intro a h1 h2 v
            apply wr64_isSome_of_forall_wr
            intro j hj u
            have href : ∃ ii r,
                Mem.find ([⟨base, stage1Code, false⟩, c2, buf'] ++
                  [⟨T, List.replicate (X86Env.pages (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32))) 0,
                    true⟩,
                   ⟨S, List.replicate (X86Env.pages f.length) 0,
                    true⟩]) (a + j) 0 = some (ii, r) ∧
                  r.rw = true :=
              ⟨1, c2, find_append_left _ _ _ 0
                (find_c2_of_three hdisjInit (by omega)
                  (by omega)), hc2rw⟩
            obtain ⟨ii, r, hf, hrw⟩ := find_rw_of_wshape hsh6 href
            exact wr_isSome_iff.mpr ⟨ii, r, hf, hrw⟩
          have hstk6 : (∀ v, (wr64 s6.mem
                ((X86Env.initStack top [argv0, path]).2 + 24 - 8 +
                  48) v).isSome) ∧
              ∀ v, (wr64 s6.mem
                ((X86Env.initStack top [argv0, path]).2 + 24 - 8 +
                  40) v).isSome := by
            constructor <;> intro v
            · exact hwstk6 _ (by omega) (by omega) v
            · exact hwstk6 _ (by omega) (by omega) v
          have hfreeT : ∀ r ∈ ([⟨base, stage1Code, false⟩, c2,
                buf'] : Mem),
              T + X86Env.pages (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32)) ≤ r.base ∨ r.hi ≤ T := by
            have hf : Mem.free [⟨base, stage1Code, false⟩, c2,
                  buf'] T (X86Env.pages (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32))) = true := by
              have h := hTfree4
              rwa [hs4mem, hs1mem] at h
            intro r hr
            have h := List.all_eq_true.mp hf r hr
            simp only [Bool.or_eq_true, decide_eq_true_eq] at h
            exact h
          have hfreeS : ∀ r ∈ ([⟨base, stage1Code, false⟩, c2,
                buf'] ++
                [⟨T, List.replicate (X86Env.pages (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32))) 0,
                  true⟩] : Mem),
              S + X86Env.pages f.length ≤ r.base ∨ r.hi ≤ S := by
            have hf : Mem.free ([⟨base, stage1Code, false⟩, c2,
                  buf'] ++
                  [⟨T, List.replicate (X86Env.pages (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32))) 0,
                    true⟩]) S (X86Env.pages f.length) = true := by
              have h := hSfree5
              rwa [free_of_wshape hsh5] at h
            intro r hr
            have h := List.all_eq_true.mp hf r hr
            simp only [Bool.or_eq_true, decide_eq_true_eq] at h
            exact h
          have hpgnw : 64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32) ≤ X86Env.pages (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32)) :=
            pages_ge _
          have hpgL : f.length ≤ X86Env.pages f.length :=
            pages_ge _
          have hnonmemT : ∀ a, T ≤ a →
              a < T + X86Env.pages (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32)) →
              ∀ r ∈ ([⟨base, stage1Code, false⟩, c2, buf'] : Mem),
                ¬ (r.base ≤ a ∧ a < r.hi) := by
            intro a ha1 ha2 r hr hmem
            rcases hfreeT r hr with h | h <;> omega
          have hwrE6 : ∀ (off v : Nat), off + 8 ≤ 64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32) →
              (wr64 s6.mem (T + off) v).isSome := by
            intro off v hoff
            apply wr64_isSome_of_forall_wr
            intro j hj u
            have hwsh6mid : wshape s6.mem =
                wshape (([⟨base, stage1Code, false⟩, c2, buf'] ++
                  [⟨T, List.replicate (X86Env.pages (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32))) 0,
                    true⟩]) ++
                  [⟨S, List.replicate (X86Env.pages f.length) 0,
                    true⟩]) := by
              rw [hsh6]
              congr 1
            have htabhi : (⟨T, List.replicate
                  (X86Env.pages (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32))) 0, true⟩ : Region).hi =
                T + X86Env.pages (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32)) := by
              simp [X86Env.Region.hi]
            have hmem : (⟨T, List.replicate
                    (X86Env.pages (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32))) 0, true⟩ :
                  Region).base ≤ T + off + j ∧
                  T + off + j < (⟨T, List.replicate
                    (X86Env.pages (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32))) 0, true⟩ :
                    Region).hi := by
              refine ⟨by show T ≤ T + off + j; omega, ?_⟩
              rw [htabhi]
              omega
            exact wr_isSome_of_wshape_append_left hwsh6mid
              ⟨0 + ([⟨base, stage1Code, false⟩, c2, buf'] :
                  Mem).length,
                ⟨T, List.replicate (X86Env.pages (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32))) 0,
                  true⟩,
                find_append_last _ 0
                  (hnonmemT _ (by omega) (by omega)) hmem, rfl⟩
          have hwrS6 : ∀ a v, S + (c - A') ≤ a →
              a < S + (A' + f.length - A') →
              (s6.mem.wr a v).isSome := by
            intro a u ha1 ha2
            have hwsh6mid : wshape s6.mem =
                wshape (([⟨base, stage1Code, false⟩, c2, buf'] ++
                  [⟨T, List.replicate (X86Env.pages (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32))) 0,
                    true⟩]) ++
                  [⟨S, List.replicate (X86Env.pages f.length) 0,
                    true⟩]) := by
              rw [hsh6]
              congr 1
            have hscrhi : (⟨S, List.replicate
                  (X86Env.pages f.length) 0, true⟩ : Region).hi =
                S + X86Env.pages f.length := by
              simp [X86Env.Region.hi]
            apply wr_isSome_of_wshape_append_last hwsh6mid _ _ rfl
            · intro r hr
              rcases List.mem_append.mp hr with hr | hr
              · intro hmem
                rcases hfreeS r (List.mem_append_left _ hr) with
                  h | h <;> omega
              · simp only [List.mem_singleton] at hr
                rw [hr]
                intro hmem
                have htabhi : (⟨T, List.replicate
                      (X86Env.pages (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32))) 0, true⟩ :
                    Region).hi = T + X86Env.pages (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32)) := by
                  simp [X86Env.Region.hi]
                have hfr := hfreeS _
                  (List.mem_append_right _
                    (List.mem_singleton_self _))
                rw [htabhi] at hmem
                rcases hfr with h | h
                · omega
                · simp only [X86Env.Region.hi,
                    List.length_replicate] at h
                  omega
            · refine ⟨by show S ≤ a; omega, ?_⟩
              rw [hscrhi]
              omega
          have hAc : A' ≤ c := by
            have h := DataSeq.le hseq
            omega
          have hrdN6 : ∀ a l, c ≤ a → l ≤ A' + f.length - a →
              ∃ bs, s6.mem.rdN a l = some bs ∧
                ∀ b ∈ bs, b < 256 := by
            intro a l ha hl
            by_cases hl0 : l = 0
            · subst hl0
              exact ⟨[], rfl,
                fun b hb => (List.not_mem_nil hb).elim⟩
            have hs := rdN_slice hrdNfile6 (o := a - A')
              (k := l) (by omega)
            refine ⟨(f.drop (a - A')).take l, ?_, ?_⟩
            · rwa [show A' + (a - A') = a by omega] at hs
            · intro b hb
              exact hfB b
                (List.mem_of_mem_drop (List.mem_of_mem_take hb))
          have hmn32 : (leVal ((bs.drop 4).take 4) % 2 ^ 32) < 2 ^ 32 := Nat.mod_lt _ (by decide)
          have h47W : 2 ^ 47 < W := by rw [hW]; decide
          have hbndref : shapeBnd
              ([⟨base, stage1Code, false⟩, c2, buf'] ++
                [⟨T, List.replicate (X86Env.pages (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32))) 0,
                  true⟩,
                 ⟨S, List.replicate (X86Env.pages f.length) 0,
                  true⟩]) := by
            intro p hp
            obtain ⟨r, hr, hpr⟩ := List.mem_map.mp hp
            have hp2 : p.2 = X86Env.Region.hi r := by
              rw [← hpr]
            have hr2 : r = ⟨base, stage1Code, false⟩ ∨ r = c2 ∨
                r = buf' ∨
                r = ⟨T, List.replicate
                  (X86Env.pages
                    (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32)))
                  0, true⟩ ∨
                r = ⟨S, List.replicate (X86Env.pages f.length) 0,
                  true⟩ := by
              simpa using hr
            rw [hp2]
            rcases hr2 with h | h | h | h | h <;> rw [h]
            · show base + stage1Code.length ≤ 2 ^ 48
              have hlen : stage1Code.length ≤ 2 ^ 48 := by
                have hh := stage1Code_length
                omega
              unfold base
              omega
            · rw [hc2top]
              omega
            · rw [hbufhi]
              omega
            · simp only [X86Env.Region.hi, List.length_replicate]
              omega
            · simp only [X86Env.Region.hi, List.length_replicate]
              omega
          have hbnd6 : shapeBnd s6.mem :=
            shapeBnd_of_wshape hsh6 hbndref
          have hs6r : ∀ i v, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 →
              i ≠ 6 → i ≠ 7 → i ≠ 8 → i ≠ 9 → i ≠ 10 → i ≠ 11 →
              s5.reg i = some v → s6.reg i = some v := by
            intro i v h0 h1 h2 h4 h6 h7 h8 h9 h10 h11 hv
            rw [hs6fr i h0 h1 h2 h4 h6 h7 h8 h9 h10 h11]
            exact hv
          have hs6r14 : s6.reg 14 = some T :=
            hs6r 14 _ (by decide) (by decide) (by decide)
              (by decide) (by decide) (by decide) (by decide)
              (by decide) (by decide) (by decide) hs5r14
          have hs6r3 : s6.reg 3 = some A' :=
            hs6r 3 _ (by decide) (by decide) (by decide)
              (by decide) (by decide) (by decide) (by decide)
              (by decide) (by decide) (by decide) hs5r3
          have hs6r12 : s6.reg 12 = some (leVal ((bs.drop 4).take 4) % 2 ^ 32) :=
            hs6r 12 _ (by decide) (by decide) (by decide)
              (by decide) (by decide) (by decide) (by decide)
              (by decide) (by decide) (by decide) hs5r12
          have hs6r15 : s6.reg 15 = some (A' + f.length) :=
            hs6r 15 _ (by decide) (by decide) (by decide)
              (by decide) (by decide) (by decide) (by decide)
              (by decide) (by decide) (by decide) hs5r15
          have hs6regs : RegsOk s6 :=
            run_regs_ok hrun6 hs5regs
          rcases parser_back_full_nh hs6rip hs6r4 hs6r14
              hs6r3 hs6r12 hs6r15 hs6r11 (by omega) (by omega)
              hcur6 hnwv6 hpop6 hs6len hs6regs hbnd6 hstk6 hAc
              hcL hAe (by omega) (by omega) (by omega)
              (by omega)
              (by
                rcases hfreeT buf' (by simp) with h | h
                · exact Or.inl (by
                    have hb : buf'.base = A' := hbufb
                    omega)
                · exact Or.inr (by
                    have hb : buf'.hi = A' + cap' := hbufhi
                    omega))
              (by
                rcases hfreeT c2 (by simp) with h | h
                · exact Or.inl (by omega)
                · exact Or.inr (by omega))
              (by
                have h := hfreeS _
                  (List.mem_append_right _
                    (List.mem_singleton_self _))
                simp only [X86Env.Region.hi,
                  List.length_replicate] at h
                rcases h with h | h
                · exact Or.inl (by omega)
                · exact Or.inr (by omega))
              (by
                rcases hdisj2 with h | h
                · exact Or.inr (by omega)
                · exact Or.inl (by omega))
              (by
                rcases hfreeS buf' (by
                  apply List.mem_append_left
                  simp) with h | h
                · exact Or.inl (by omega)
                · exact Or.inr (by
                    have hb : buf'.hi = A' + cap' := hbufhi
                    omega))
              (by
                rcases hfreeS c2 (by
                  apply List.mem_append_left
                  simp) with h | h
                · exact Or.inl (by omega)
                · exact Or.inr (by omega))
              hwrE6 hwrS6 hrdN6
              (Or.inl (by omega))
              (by
                rcases hfreeT ⟨base, stage1Code, false⟩ (by
                  simp) with h | h
                · exact Or.inl (by
                    have hpg := pages_ge
                      (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32))
                    have h2 : T + X86Env.pages
                        (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32)) ≤
                        base := h
                    omega)
                · exact Or.inr h)
              (by
                rcases hfreeS ⟨base, stage1Code, false⟩ (by
                  apply List.mem_append_left
                  simp) with h | h
                · exact Or.inl (by
                    have hpg := pages_ge f.length
                    have h2 : S + X86Env.pages f.length ≤ base := h
                    omega)
                · exact Or.inr (by
                    have h2 : base + stage1Code.length ≤ S := h
                    omega))
              with hbexit | hbprog
          · obtain ⟨kB, hkB, hrunB, hnhB⟩ := hbexit
            exact Or.inl ⟨kL + kF + 20 + 12 + kB, by omega, by
              rw [show kL + kF + 20 + 12 + kB =
                (kL + kF + 20 + 12) + kB by omega, run_add,
                hrunLF6, hrunB]⟩
          · obtain ⟨wds, NB, s7, hNB, hwdslen, hrunB, hs7rip,
                hs7r4, hs7r14, hs7r13, hs7r12, hs7r0, hlay7, htab7,
                hbw7, hcurE7, hmnlt7, hins7, hs7k, hs7len, hs7bnd,
                hsh7, hcode7, hnhB⟩ := hbprog
            have hdisja : A' + f.length ≤
                (X86Env.initStack top [argv0, path]).2 + 24 - 16 ∨
                (X86Env.initStack top [argv0, path]).2 + 24 ≤
                  A' := by
              rcases hdisj2 with h | h
              · exact Or.inr (by omega)
              · exact Or.inl (by omega)
            have hrd32_6 : ∀ a d, rd32 s1.mem a = some d →
                A' + 16 ≤ a → a + 4 ≤ c →
                rd32 s6.mem a = some d := by
              intro a d hd ha hac
              obtain ⟨bsq, hbq, hlq⟩ : ∃ bs,
                  s1.mem.rdN a 4 = some bs ∧ leVal bs = d := by
                have h2 : rd32 s1.mem a =
                    (s1.mem.rdN a 4).map leVal := rfl
                rw [h2] at hd
                exact Option.map_eq_some_iff.mp hd
              have hm2q : m2.rdN a 4 = some bsq := by
                rw [hm2fr a 4 (by omega), hs4mem]
                exact hbq
              have hs5q : s5.mem.rdN a 4 = some bsq := by
                rw [hs5mem]
                exact rdN_append_left hm2q
              have hm1q : m1.rdN a 4 = some bsq := by
                rw [hm1fr a 4 (by omega)]
                exact hs5q
              have h6q : s6.mem.rdN a 4 = some bsq := by
                rw [hs6mem]
                exact rdN_append_left hm1q
              show (s6.mem.rdN a 4).map leVal = some d
              rw [h6q]
              simp [hlq]
            have hseq6 : DataSeq s6.mem (A' + 16) ds c :=
              DataSeq_of_rd32 hseq hrd32_6
            have hL20 : 20 ≤ f.length := by
              have hcle := DataSeq.le hseq
              omega
            obtain ⟨pr, hdec, hwords, hmain, hdatamap, hpb,
                hfenc⟩ :=
              decode_of_layouts hrdNfile6 hfB hL20 h206 hmagic
                hdslen hseq6 hwdslen hlay7 hbw7 hcurE7
            have hmain0 : ∃ hj : pr.main < pr.words.length,
                (pr.words[pr.main]'hj).ins = 0 := by
              obtain ⟨hj, hi⟩ := hins7
              refine ⟨by rw [hmain, hwords]; exact hj, ?_⟩
              simp only [hmain, hwords]
              exact hi
            have hruntot : run stage1Im (kL + kF + 20 + 12 + NB)
                (Out.ok
                  (stage1Init f path argv0 stdin oracle top)) =
                Out.ok s7 := by
              rw [show kL + kF + 20 + 12 + NB =
                (kL + kF + 20 + 12) + NB by omega, run_add,
                hrunLF6, hrunB]
            have hcode1 : s1.mem.rdN base stage1Code.length =
                some stage1Code := by
              have h := rdN_first_region stage1Code.length
                (r := ⟨base, stage1Code, false⟩) (t := [c2, buf'])
                (x := base) (Nat.le_refl _)
                (by simp [X86Env.Region.hi])
              have h2 : (stage1Code.drop (base - base)).take
                    stage1Code.length = stage1Code := by
                rw [Nat.sub_self, List.drop_zero]
                exact List.take_of_length_le (Nat.le_refl _)
              rw [h2] at h
              rwa [hs1mem]
            have hcode6 : s6.mem.rdN base stage1Code.length =
                some stage1Code := by
              have hm2c : m2.rdN base stage1Code.length =
                  some stage1Code := by
                rw [hm2fr base _ (Or.inl (by omega)), hs4mem]
                exact hcode1
              have hs5c : s5.mem.rdN base stage1Code.length =
                  some stage1Code := by
                rw [hs5mem]
                exact rdN_append_left hm2c
              have hm1c : m1.rdN base stage1Code.length =
                  some stage1Code := by
                rw [hm1fr base _ (Or.inl (by omega))]
                exact hs5c
              rw [hs6mem]
              exact rdN_append_left hm1c
            have htabD : ∀ j, j < pr.words.length →
                (X86Env.initStack top [argv0, path]).2 + 24 - 8 +
                    56 ≤ T + 64 * j ∨
                  T + 64 * j + 64 ≤
                    (X86Env.initStack top [argv0, path]).2 + 24 -
                      8 + 56 - 48 := by
              intro j hj
              rw [hwords, hwdslen] at hj
              rcases hfreeT c2 (by simp) with h | h
              · exact Or.inr (by
                  have hpg := pages_ge
                    (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32))
                  omega)
              · exact Or.inl (by omega)
            have hdisjEffX : aEff + 114 ≤
                  (X86Env.initStack top [argv0, path]).2 + 24 - 8 +
                    56 - 48 ∨
                (X86Env.initStack top [argv0, path]).2 + 24 - 8 +
                    56 ≤ aEff :=
              Or.inl (by
                have hcl := stage1Code_length
                unfold aEff
                omega)
            refine Or.inr ⟨pr, T,
              (X86Env.initStack top [argv0, path]).2 + 24 - 8 + 56,
              s7, kL + kF + 20 + 12 + NB, by omega, hdec, hpb,
              hfenc, hruntot, hs7rip, hs7r14, by
              rw [hwords, hwdslen]
              exact hs7r13, hs7r4, by omega, by omega, by
              rw [hwords, hwdslen]
              have hpg := pages_ge
                (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32))
              omega, htabD, hdisjEffX,
              stackRoom_of_wshape_c2 (hsh7.trans hsh6) hs7r4
                hc2rw hdisjInit (by omega) (by omega) (by omega),
              run_regs_ok hrunB hs6regs, hs7len,
              WordTab_of_TabLay (by rw [hwords]; exact htab7) (by
                rw [hwords]
                omega) (by omega) (by
                rw [hwords]
                exact hbw7) (by omega) (by omega) (by
                rw [hwords, hwdslen]
                have hpg := pages_ge
                  (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32))
                omega) (by
                rw [hwords, hwdslen]
                rcases hfreeT c2 (by simp) with h | h
                · exact Or.inr (by
                    have hpg := pages_ge
                      (64 * (leVal ((bs.drop 8).take 4) % 2 ^ 32))
                    omega)
                · exact Or.inl (by omega)) (by
                rw [hwords]
                rcases hdisj2 with h | h
                · exact Or.inl (by omega)
                · exact Or.inr (by omega)) hdisjEffX,
              EffTable_of_code (hcode7 hcode6),
              noHitUpTo_add hnhLF6 hrunLF6 hnhB,
              hmain0⟩
        · exfalso
          exact hnh (hitsNomem_at hrunt2tot ht2memsys hnomem2)
      · exfalso
        exact hnh (hitsNomem_at hrunt1tot ht1memsys hnomem1)

/-- Combined parser-prefix plus checker bound fits inside `stage1Bound`. -/
theorem prefix_plus_checker_le_stage1Bound {f : List Nat} {pr : Prog} {K m : Nat}
    (hfenc : f = pr.enc)
    (hK : K ≤ 2000 + 2000 * f.length)
    (hm : m ≤ (pr.words.map wordCost).sum + 10) :
    K + m ≤ 1000000 + 10000 * (f.length + 16) ^ 3 := by
  have hX : wordsBytes pr.words + 16 ≤ f.length := by
    have hlen := enc_length pr
    rw [← hfenc] at hlen
    omega
  have hfit : m ≤ 10000 * (wordsBytes pr.words + 16) ^ 3 :=
    Nat.le_trans hm (checker_cost_fit (wordsBytes pr.words) pr.words (Nat.le_refl _))
  let X := wordsBytes pr.words + 16
  let B := f.length + 16
  have hpX : X ^ 3 = (X * X) * X := by rw [Nat.pow_succ, Nat.pow_succ, Nat.pow_one]
  have hpB : B ^ 3 = (B * B) * f.length + (B * B) * 16 := by
    calc B ^ 3 = (B * B) * B := by rw [Nat.pow_succ, Nat.pow_succ, Nat.pow_one]
      _ = (B * B) * (f.length + 16) := rfl
      _ = (B * B) * f.length + (B * B) * 16 := Nat.mul_add (B * B) f.length 16
  have hm_part : 10000 * X ^ 3 ≤ 10000 * ((B * B) * f.length) := by
    rw [hpX]
    exact Nat.mul_le_mul (Nat.le_refl 10000)
      (Nat.mul_le_mul (Nat.mul_le_mul (by omega) (by omega)) hX)
  rw [hpX] at hm_part
  have hfit' : m ≤ 10000 * ((X * X) * X) := by
    change m ≤ 10000 * X ^ 3 at hfit
    rw [hpX] at hfit
    exact hfit
  have hfl_BB : f.length ≤ (B * B) * 16 := by
    have h1 : f.length ≤ B * 1 := by omega
    have h2 : B * 1 ≤ B * B := Nat.mul_le_mul (Nat.le_refl B) (by omega)
    have h3 : B * B ≤ (B * B) * 16 := Nat.le_mul_of_pos_right _ (by decide)
    exact Nat.le_trans h1 (Nat.le_trans h2 h3)
  have hK_part : 2000 * f.length ≤ 10000 * ((B * B) * 16) :=
    Nat.mul_le_mul (by decide) hfl_BB
  have htot : 10000 * (f.length + 16) ^ 3 =
      10000 * ((B * B) * f.length) + 10000 * ((B * B) * 16) := by
    change 10000 * B ^ 3 = _
    rw [hpB, Nat.left_distrib]
  omega

/-- Deterministic execution cannot exit with two distinct exit codes. -/
theorem run_exit_det (im : Nat → Option (Ins × Nat)) :
    ∀ {k1 k2 : Nat} {o : Out} {c1 c2 : Nat},
      run im k1 o = Out.exit c1 → run im k2 o = Out.exit c2 → c1 = c2 := by
  intro k1 k2 o c1 c2 h1 h2
  have hex : ∀ (d c : Nat), run im d (Out.exit c) = Out.exit c := by
    intro d c
    induction d with
    | zero => rfl
    | succ d ih =>
      rw [show d + 1 = 1 + d by omega, run_add im 1 d]
      have h1e : run im 1 (Out.exit c) = Out.exit c := rfl
      rw [h1e]
      exact ih
  rcases Nat.le_total k1 k2 with h | h
  · have hk : k2 = k1 + (k2 - k1) := by omega
    rw [hk, run_add, h1, hex] at h2
    cases h2; rfl
  · have hk : k1 = k2 + (k1 - k2) := by omega
    rw [hk, run_add, h2, hex] at h1
    cases h1; rfl

private theorem exists_false_of_all_false {α : Type} (p : α → Bool) :
    ∀ (xs : List α), xs.all p = false → ∃ x ∈ xs, p x = false
  | [], h => by simp at h
  | x :: xs, h => by
    cases hp : p x with
    | false => exact ⟨x, List.mem_cons_self, hp⟩
    | true =>
      simp only [List.all_cons, hp, Bool.true_and] at h
      obtain ⟨y, hy, hpy⟩ := exists_false_of_all_false p xs h
      exact ⟨y, List.mem_cons_of_mem x hy, hpy⟩

/-- Accept direction for the real stage-1 image within the cubic bound. -/
theorem stage1_accept_exit0
    (f path argv0 stdin : List Nat) (oracle : List Choice) (top : Nat)
    (hinit : Stage1InitOk f path argv0 top)
    (horacle : OracleAdmits (stage1Init f path argv0 stdin oracle top))
    (hnomem : ¬ HitsNomem (stage1Init f path argv0 stdin oracle top))
    (hacc : ∃ p : Prog, Sporos.decode f = some p ∧ Sporos.checkProg p = true) :
    ∃ k, k ≤ 1000000 + 10000 * (f.length + 16) ^ 3 ∧
      run stage1Im k (Out.ok (stage1Init f path argv0 stdin oracle top)) = Out.exit 0 := by
  obtain ⟨pr, hdec, hchk⟩ := hacc
  obtain ⟨hfenc, -⟩ := decode_inv hdec hinit.1
  obtain ⟨T, S, sp, s', K, hK, hrun, hrip, hr14, hr13, hr4, hsp48, hspW, htblW, htab,
    hdisjEff48, hroom, hregs, hlen16, htabS, heff, -⟩ :=
    stage1_prefix_accept hinit horacle hnomem hdec hchk
  obtain ⟨m, hm, hrunm⟩ :=
    checker_accepts_from_startB stage1_allCode stage1_outerCode
      hsp48 hspW htblW htab hdisjEff48 hchk s'
      (by rw [hrip]; rfl) hr14 hr13 hr4 (by omega) hregs hroom heff htabS
  refine ⟨K + m, prefix_plus_checker_le_stage1Bound hfenc hK hm, ?_⟩
  rw [run_add, hrun, hrunm]

/-- Refusal direction for the real stage-1 image within the cubic bound. -/
theorem stage1_refuse_exit1
    (f path argv0 stdin : List Nat) (oracle : List Choice) (top : Nat)
    (hinit : Stage1InitOk f path argv0 top)
    (horacle : OracleAdmits (stage1Init f path argv0 stdin oracle top))
    (hnomem : ¬ HitsNomem (stage1Init f path argv0 stdin oracle top))
    (hbad : ¬ ∃ p : Prog, Sporos.decode f = some p ∧ Sporos.checkProg p = true) :
    ∃ k, k ≤ 1000000 + 10000 * (f.length + 16) ^ 3 ∧
      run stage1Im k (Out.ok (stage1Init f path argv0 stdin oracle top)) = Out.exit 1 := by
  rcases stage1_prefix_classify_main hinit horacle hnomem with hpre | hhand
  · obtain ⟨k, hk, hrun⟩ := hpre
    have hlin3 : f.length + 16 ≤ (f.length + 16) ^ 3 := by
      calc f.length + 16 = (f.length + 16) * 1 := by rw [Nat.mul_one]
        _ ≤ (f.length + 16) * ((f.length + 16) * (f.length + 16)) :=
            Nat.mul_le_mul_left _ (Nat.mul_pos (by omega) (by omega))
        _ = (f.length + 16) ^ 3 := by
            rw [Nat.pow_succ, Nat.pow_succ, Nat.pow_succ, Nat.pow_zero, Nat.one_mul, Nat.mul_assoc]
    have hmul : 2000 * f.length ≤ 2000 * (f.length + 16) ^ 3 :=
      Nat.mul_le_mul (Nat.le_refl 2000) (by omega)
    exact ⟨k, by omega, hrun⟩
  · obtain ⟨pr, T, sp, s', K, hK, hdec, hpb, hfenc, hrun, hrip, hr14, hr13, hr4,
      hsp48, hspW, htblW, htab, hdisjEff48, hroom, hregs, hlen16, htabS, heff, -, hmain0⟩ :=
      hhand
    have hchk_ne : Sporos.checkProg pr ≠ true := fun ht => hbad ⟨pr, hdec, ht⟩
    have hchkF : Sporos.checkProg pr = false := Bool.eq_false_iff.mpr hchk_ne
    have hchk1F : Sporos.checkProg1 pr = false := by
      rw [Sporos.checkProg1_eq]; exact hchkF
    obtain ⟨hj, hins0⟩ := hmain0
    have hgetm : pr.words[pr.main]? = some (pr.words[pr.main]'hj) :=
      List.getElem?_eq_getElem hj
    have hallF : (pr.words.all fun wd => Sporos.checkWord1 pr wd) = false := by
      unfold Sporos.checkProg1 at hchk1F
      rw [hgetm] at hchk1F
      dsimp only at hchk1F
      simp only [hins0, beq_self_eq_true, Bool.and_true] at hchk1F
      exact hchk1F
    have hexBad : ∃ wd ∈ pr.words, Sporos.checkWord1 pr wd = false :=
      exists_false_of_all_false (Sporos.checkWord1 pr) pr.words hallF
    obtain ⟨m, hm, hrunm⟩ :=
      checker_refuses_from_startB stage1_allCode stage1_outerCode stage1_refuseCode
        stage1_chkRefuseCode hsp48 hspW htblW htab hdisjEff48 hpb.2.2.2.2 hexBad s'
        (by rw [hrip]; rfl) hr14 hr13 hr4 (by omega) hregs hroom heff htabS
    refine ⟨K + m, prefix_plus_checker_le_stage1Bound hfenc hK hm, ?_⟩
    rw [run_add, hrun, hrunm]

/-- Full biconditional + totality proof backing `stage1_final_target`. -/
theorem stage1_final_target_proof
    (f path argv0 stdin : List Nat) (oracle : List Choice) (top : Nat)
    (hinit : Stage1InitOk f path argv0 top)
    (horacle : OracleAdmits (stage1Init f path argv0 stdin oracle top))
    (hnomem : ¬ HitsNomem (stage1Init f path argv0 stdin oracle top)) :
    ((∃ k, k ≤ 1000000 + 10000 * (f.length + 16) ^ 3 ∧
        run stage1Im k
          (Out.ok (stage1Init f path argv0 stdin oracle top)) =
          Out.exit 0) ↔
      (∃ p : Prog, Sporos.decode f = some p ∧ Sporos.checkProg p = true)) ∧
    ((∃ k, k ≤ 1000000 + 10000 * (f.length + 16) ^ 3 ∧
        run stage1Im k
          (Out.ok (stage1Init f path argv0 stdin oracle top)) =
          Out.exit 1) ↔
      ¬ (∃ p : Prog, Sporos.decode f = some p ∧ Sporos.checkProg p = true)) ∧
    (∃ k, k ≤ 1000000 + 10000 * (f.length + 16) ^ 3 ∧
      (run stage1Im k
          (Out.ok (stage1Init f path argv0 stdin oracle top)) =
          Out.exit 0 ∨
       run stage1Im k
          (Out.ok (stage1Init f path argv0 stdin oracle top)) =
          Out.exit 1)) := by
  by_cases hacc : ∃ p : Prog, Sporos.decode f = some p ∧ Sporos.checkProg p = true
  · obtain ⟨k, hk, hrun⟩ := stage1_accept_exit0 f path argv0 stdin oracle top hinit horacle hnomem hacc
    have hno1 : ¬ ∃ k', k' ≤ 1000000 + 10000 * (f.length + 16) ^ 3 ∧
        run stage1Im k' (Out.ok (stage1Init f path argv0 stdin oracle top)) = Out.exit 1 := by
      intro ⟨k', _, hrun'⟩
      have h01 := run_exit_det stage1Im hrun hrun'
      cases h01
    exact ⟨Iff.intro (fun _ => hacc) (fun _ => ⟨k, hk, hrun⟩),
      Iff.intro (fun hex => absurd hex hno1) (fun hbad => absurd hacc hbad),
      ⟨k, hk, Or.inl hrun⟩⟩
  · obtain ⟨k, hk, hrun⟩ := stage1_refuse_exit1 f path argv0 stdin oracle top hinit horacle hnomem hacc
    have hno0 : ¬ ∃ k', k' ≤ 1000000 + 10000 * (f.length + 16) ^ 3 ∧
        run stage1Im k' (Out.ok (stage1Init f path argv0 stdin oracle top)) = Out.exit 0 := by
      intro ⟨k', _, hrun'⟩
      have h01 := run_exit_det stage1Im hrun' hrun
      cases h01
    exact ⟨Iff.intro (fun hex => absurd hex hno0) (fun h => absurd h hacc),
      Iff.intro (fun _ => hacc) (fun _ => ⟨k, hk, hrun⟩),
      ⟨k, hk, Or.inr hrun⟩⟩

end SporosX1
