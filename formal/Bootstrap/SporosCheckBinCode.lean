/-
SporosCheckBinCode: correctness proofs for `skipcodeT` and `skipwordsT`.
-/
import Bootstrap.SporosCheckBinLoops

namespace SporosCheckBin

open X86Env X86Int

local macro "pframe_tac" : tactic =>
  `(tactic| first
    | exact Frame.refl _ _
    | (refine Frame.of_mk ?_ (by simp);
       repeat (first | exact RegsFrame.refl _ _ | refine RegsFrame.set ?_ (by first | decide | simp) _)))

/-! ### `skipCode` (`skipcodeT`) -/

theorem skipcode_len {L p rej : Nat}
    (f14 : im (L + 14) = some (.ld8 14 p 0, 4))
    (f18 : im (L + 18) = some (.cmpI 14 57, 4))
    (f22 : im (L + 22) = some (.jcc 2 (L + 88), 2))
    (f24 : im (L + 24) = some (.cmpI 14 128, 7))
    (f31 : im (L + 31) = some (.jcc 2 rej, 6))
    (f37 : im (L + 37) = some (.jcc 4 (L + 80), 2))
    (f39 : im (L + 39) = some (.cmpI 14 140, 7))
    (f46 : im (L + 46) = some (.jcc 7 rej, 6))
    (f52 : im (L + 52) = some (.jcc 4 (L + 88), 2))
    (f54 : im (L + 54) = some (.cmpI 14 131, 7))
    (f61 : im (L + 61) = some (.jcc 2 (L + 72), 2))
    (f63 : im (L + 63) = some (.cmpI 14 134, 7))
    (f70 : im (L + 70) = some (.jcc 2 (L + 88), 2))
    (f72 : im (L + 72) = some (.movI 14 5, 6))
    (f78 : im (L + 78) = some (.jmp (L + 94), 2))
    (f80 : im (L + 80) = some (.movI 14 9, 6))
    (f86 : im (L + 86) = some (.jmp (L + 94), 2))
    (f88 : im (L + 88) = some (.movI 14 1, 6))
    {b pos : Nat} {bs : List Nat} {s : St}
    (he : Env s b bs) (hrip : s.rip = L + 14) (hrp : s.reg p = some (b + pos))
    (hpos : pos < bs.length) :
    Reaches s (fun s' => Frame [14] s s' ∧
      match instrLenA bs[pos] with
      | none => s'.rip = rej
      | some l => s'.rip = L + 94 ∧ s'.reg 14 = some l) := by
  have hl := he.len
  have hsm := he.small
  have hop : bs[pos] < 256 := he.bytes bs[pos] (List.getElem_mem hpos)
  have hrd : s.mem.rd (b + pos) = some bs[pos] := he.buf pos hpos
  xs f14 using hl, hrp, ea_small (show b + pos + 0 < 2 ^ 64 by omega), Nat.add_zero, hrd
  xs f18 using hl
  have hc57 := (cond_sub (show bs[pos] < W by rw [W_eq]; omega)
    (show 57 < W by rw [W_eq]; omega)).1
  by_cases h57 : bs[pos] < 57
  · xs f22 using hc57, h57
    xs f88
    refine Reaches.refl ⟨by frame_tac, ?_⟩
    rw [show instrLenA bs[pos] = some 1 by unfold instrLenA; simp [h57]]
    dsimp only
    exact ⟨by omega, by rsimp using hl⟩
  · xs f22 using hc57, h57
    xs f24 using hl
    have hc128 := cond_sub (show bs[pos] < W by rw [W_eq]; omega)
      (show 128 < W by rw [W_eq]; omega)
    by_cases h128 : bs[pos] < 128
    · xs f31 using hc128.1, h128
      exact Reaches.refl ⟨by frame_tac, by simp [instrLenA, h57, h128]⟩
    · xs f31 using hc128.1, h128
      by_cases he128 : bs[pos] = 128
      · xs f37 using hc128.2.2.1, he128
        xs f80
        xs f86
        refine Reaches.refl ⟨by frame_tac, ?_⟩
        rw [show instrLenA 128 = some 9 by decide]
        dsimp only
        exact ⟨rfl, by rsimp using hl⟩
      · xs f37 using hc128.2.2.1, he128
        xs f39 using hl
        have hc140 := cond_sub (show bs[pos] < W by rw [W_eq]; omega)
          (show 140 < W by rw [W_eq]; omega)
        by_cases h140 : 140 < bs[pos]
        · xs f46 using hc140.2.2.2.2.2, h140
          have hne : 0x8c < bs[pos] := h140
          exact Reaches.refl ⟨by frame_tac, by simp [instrLenA, h57, h128, he128, hne]⟩
        · xs f46 using hc140.2.2.2.2.2, h140
          by_cases he140 : bs[pos] = 140
          · xs f52 using hc140.2.2.1, he140
            xs f88
            refine Reaches.refl ⟨by frame_tac, ?_⟩
            rw [show instrLenA 140 = some 1 by decide]
            dsimp only
            exact ⟨by omega, by rsimp using hl⟩
          · xs f52 using hc140.2.2.1, he140
            xs f54 using hl
            have hc131 := (cond_sub (show bs[pos] < W by rw [W_eq]; omega)
              (show 131 < W by rw [W_eq]; omega)).1
            by_cases h131 : bs[pos] < 131
            · xs f61 using hc131, h131
              xs f72
              xs f78
              have hne : ¬ (0x8c < bs[pos]) := h140
              refine Reaches.refl ⟨by frame_tac, ?_⟩
              rw [show instrLenA bs[pos] = some 5 by unfold instrLenA; repeat' split <;> first | rfl | omega]
              dsimp only
              exact ⟨rfl, by rsimp using hl⟩
            · xs f61 using hc131, h131
              xs f63 using hl
              have hc134 := (cond_sub (show bs[pos] < W by rw [W_eq]; omega)
                (show 134 < W by rw [W_eq]; omega)).1
              by_cases h134 : bs[pos] < 134
              · xs f70 using hc134, h134
                xs f88
                have hne : ¬ (0x8c < bs[pos]) := h140
                refine Reaches.refl ⟨by frame_tac, ?_⟩
                rw [show instrLenA bs[pos] = some 1 by unfold instrLenA; repeat' split <;> first | rfl | omega]
                dsimp only
                exact ⟨by omega, by rsimp using hl⟩
              · xs f70 using hc134, h134
                xs f72
                xs f78
                have hne : ¬ (0x8c < bs[pos]) := h140
                refine Reaches.refl ⟨by frame_tac, ?_⟩
                rw [show instrLenA bs[pos] = some 5 by unfold instrLenA; repeat' split <;> first | rfl | omega]
                dsimp only
                exact ⟨rfl, by rsimp using hl⟩

theorem skipcode_tail {L p rej : Nat} (hp : p < 16) (hp5 : p ≠ 5) (hp11 : p ≠ 11)
    (f94 : im (L + 94) = some (.alu .add p 14, 3))
    (f97 : im (L + 97) = some (.cmp p 5, 3))
    (f100 : im (L + 100) = some (.jcc 7 rej, 6))
    (f106 : im (L + 106) = some (.aluI .sub 11 1, 4))
    (f110 : im (L + 110) = some (.jmp L, 2))
    {b pos n l : Nat} {bs : List Nat} {s : St}
    (he : Env s b bs) (hrip : s.rip = L + 94) (hrp : s.reg p = some (b + pos))
    (h14 : s.reg 14 = some l) (h11 : s.reg 11 = some (n + 1))
    (hpos : pos ≤ bs.length) (hl1 : l ≤ 9) (hn : n + 1 < 2 ^ 64) :
    Reaches s (fun s' => Frame [p, 11, 14] s s' ∧
      if bs.length < pos + l then s'.rip = rej
      else s'.rip = L ∧ s'.reg p = some (b + (pos + l)) ∧ s'.reg 11 = some n) := by
  have hl := he.len
  have h5 := he.rbp
  have hsm := he.small
  xs f94 using hl, hp, hrp, h14, add_small (show b + pos + l < 2 ^ 64 by omega)
  xs f97 using hl, hp, hp5, hp5.symm, h5
  have hc7 := (cond_sub (show b + pos + l < W by rw [W_eq]; omega)
    (show b + bs.length < W by rw [W_eq]; omega)).2.2.2.2.2
  by_cases hlt : bs.length < pos + l
  · xs f100 using hc7, show b + bs.length < b + pos + l by omega
    exact Reaches.refl ⟨by pframe_tac, by simp [hlt]⟩
  · xs f100 using hc7, show ¬ (b + bs.length < b + pos + l) by omega
    xs f106 using hl, hp11, hp11.symm, h11,
      sub_small (show n + 1 < 2 ^ 64 by omega) (show 1 ≤ n + 1 by omega), Nat.add_sub_cancel
    xs f110
    refine Reaches.refl ⟨by pframe_tac, ?_⟩
    simp only [hlt, reduceIte, true_and]
    refine ⟨?_, ?_⟩
    · simp only [reg_mk_set_ne, ne_eq, not_false_eq_true, hp11.symm, reg_mk_set_eq, hl, hp,
        Option.some.injEq]
      omega
    · rsimp using hl

theorem skipcode_spec {L p rej : Nat} (hc : CodeAt (skipcodeT L p rej))
    (hp : p < 16) (hp5 : p ≠ 5) (hp11 : p ≠ 11) (hp14 : p ≠ 14) {b : Nat} {bs : List Nat} :
    ∀ (n pos : Nat) (s : St), Env s b bs → s.rip = L → s.reg p = some (b + pos) →
      s.reg 11 = some n → pos ≤ bs.length → n < 2 ^ 64 →
      Reaches s (fun s' => Frame [p, 11, 14] s s' ∧
        match skipCodeA bs n pos with
        | some pos' => s'.rip = L + 112 ∧ s'.reg p = some (b + pos') ∧ s'.reg 11 = some 0
        | none => s'.rip = rej) := by
  simp only [skipcodeT, CodeAt_cons, CodeAt_nil, and_true] at hc
  obtain ⟨f0, f3, f5, f8, f14, f18, f22, f24, f31, f37, f39, f46, f52, f54, f61, f63, f70,
    f72, f78, f80, f86, f88, f94, f97, f100, f106, f110⟩ := hc
  intro n
  induction n with
  | zero =>
    intro pos s he hrip hrp h11 hpos hn
    xs f0 using h11
    xs f3 using cond_test_e
    exact Reaches.refl ⟨by pframe_tac, rfl, hrp, h11⟩
  | succ n ih =>
    intro pos s he hrip hrp h11 hpos hn
    have h5 := he.rbp
    have hl := he.len
    have hsm := he.small
    xs f0 using h11
    xs f3 using cond_test_e, Nat.add_one_ne_zero
    xs f5 using hrp, h5
    have hc3 := (cond_sub (show b + pos < W by rw [W_eq]; omega)
      (show b + bs.length < W by rw [W_eq]; omega)).2.1
    by_cases hlt : pos < bs.length
    · xs f8 using hc3, show ¬ (b + bs.length ≤ b + pos) by omega
      refine reach_frame (by frame_tac)
        ((skipcode_len (b := b) (bs := bs) f14 f18 f22 f24 f31 f37 f39 f46 f52 f54 f61 f63 f70
          f72 f78 f80 f86 f88 ?_ ?_ ?_ hlt).trans fun s1 ⟨hf1, hm1⟩ => ?_)
      · exact he.frame (S := [p, 11, 14]) (by frame_tac) (by simp [hp5.symm])
      · rsimp; omega
      · rsimp; exact hrp
      · have hf01 : Frame [p, 11, 14] _ s1 :=
          Frame.mono hf1 (fun x hx => by simp at hx; subst hx; simp)
        have he1 : Env s1 b bs := he.frame (Frame.trans_same (by frame_tac) hf01) (by simp [hp5.symm])
        cases hia : instrLenA bs[pos] with
        | none =>
          rw [hia] at hm1
          exact Reaches.refl ⟨hf01, by simp [skipCodeA, hlt, hia, hm1]⟩
        | some l =>
          rw [hia] at hm1
          have hlb := instrLenA_pos hia
          refine (skipcode_tail hp hp5 hp11 f94 f97 f100 f106 f110 he1 hm1.1
            (by rw [hf1.regs p (by simp [hp14])]; rsimp; exact hrp) hm1.2
            (by rw [hf1.regs 11 (by simp)]; rsimp; exact h11) hpos hlb.2 hn).trans fun s2 ⟨hf2, hm2⟩ => ?_
          have hf12 : Frame [p, 11, 14] _ s2 := Frame.trans_same hf01 hf2
          by_cases hbo : bs.length < pos + l
          · rw [if_pos hbo] at hm2
            exact Reaches.refl ⟨hf12, by simp [skipCodeA, hlt, hia, hbo, hm2]⟩
          · rw [if_neg hbo] at hm2
            have he2 : Env s2 b bs := he.frame (Frame.trans_same (by frame_tac) hf12) (by simp [hp5.symm])
            refine (ih (pos + l) s2 he2 hm2.1 hm2.2.1 hm2.2.2 (by omega) (by omega)).mono
              fun s3 ⟨hf3, hm3⟩ => ⟨Frame.trans_same hf12 hf3, ?_⟩
            simpa [skipCodeA, hlt, hia, hbo] using hm3
    · xs f8 using hc3, show b + bs.length ≤ b + pos by omega
      exact Reaches.refl ⟨by frame_tac, by simp [skipCodeA, hlt]⟩

/-! ### `skipWords` (`skipwordsT`) -/

theorem skipwords_spec {L rej : Nat} (hc : CodeAt (skipwordsT L rej)) {b : Nat} {bs : List Nat} :
    ∀ (n pos : Nat) (s : St), Env s b bs → s.rip = L → s.reg 10 = some (b + pos) →
      s.reg 15 = some n → pos ≤ bs.length → n < 2 ^ 32 →
      Reaches s (fun s' => Frame [10, 11, 14, 15] s s' ∧
        match skipWordsA bs n pos with
        | some pos' => s'.rip = L + 201 ∧ s'.reg 10 = some (b + pos') ∧ s'.reg 15 = some 0
        | none => s'.rip = rej) := by
  rw [skipwordsT, CodeAt_append, CodeAt_append] at hc
  obtain ⟨⟨hcPre, hcSc⟩, hcPost⟩ := hc
  simp only [CodeAt_cons, CodeAt_nil, and_true] at hcPre hcPost
  obtain ⟨f0, f3, f9, f12, f16, f19, f25, f29, f33, f36, f40⟩ := hcPre
  obtain ⟨f156, f159, f163, f166, f169, f173, f177, f180, f183, f186, f192, f196⟩ := hcPost
  intro n
  induction n with
  | zero =>
    intro pos s he hrip h10 h15 hpos hn
    xs f0 using h15
    xs f3 using cond_test_e
    exact Reaches.refl ⟨by frame_tac, rfl, h10, h15⟩
  | succ n ih =>
    intro pos s he hrip h10 h15 hpos hn
    have h5 := he.rbp
    have hl := he.len
    have hsm := he.small
    xs f0 using h15
    xs f3 using cond_test_e, Nat.add_one_ne_zero
    xs f9 using hl, h10
    xs f12 using hl, add_small (show b + pos + 20 < 2 ^ 64 by omega)
    xs f16 using hl, h5
    have hc7 := (cond_sub (show b + pos + 20 < W by rw [W_eq]; omega)
      (show b + bs.length < W by rw [W_eq]; omega)).2.2.2.2.2
    by_cases hlt : bs.length < pos + 20
    · xs f19 using hc7, show b + bs.length < b + pos + 20 by omega
      exact Reaches.refl ⟨by frame_tac, by simp [skipWordsA, hlt]⟩
    · xs f19 using hc7, show ¬ (b + bs.length < b + pos + 20) by omega
      have hpos20 : pos + 20 ≤ bs.length := by omega
      have hpos16 : pos + 16 + 4 ≤ bs.length := by omega
      have hpos12 : pos + 12 + 4 ≤ bs.length := by omega
      let na := u32 bs (pos + 16)
      let nc := u32 bs (pos + 12)
      have hna : na < 2 ^ 32 := u32_lt he.bytes (pos + 16)
      have hnc : nc < 2 ^ 32 := u32_lt he.bytes (pos + 12)
      xs f25 using hl, h10, ea_small (show b + pos + 16 < 2 ^ 64 by omega),
        show b + pos + 16 = b + (pos + 16) by omega, rd32_buf he.buf hpos16
      xs f29 using hl, show u32 bs (pos + 16) * 2 ^ 32 % 18446744073709551616 = u32 bs (pos + 16) * 2 ^ 32 by
        rw [Nat.mod_eq_of_lt]; change na * 2 ^ 32 < 2 ^ 64; omega
      xs f33 using hl, h15, add_small (show n + 1 + u32 bs (pos + 16) * 2 ^ 32 < 2 ^ 64 by
        change n + 1 + na * 2 ^ 32 < 2 ^ 64; omega)
      xs f36 using hl, h10, ea_small (show b + pos + 12 < 2 ^ 64 by omega),
        show b + pos + 12 = b + (pos + 12) by omega, rd32_buf he.buf hpos12
      xs f40 using hl, h10, add_small (show b + pos + 20 < 2 ^ 64 by omega)
      refine reach_frame (by frame_tac)
        ((skipcode_spec (b := b) (bs := bs) hcSc (by decide) (by decide) (by decide) (by decide)
          nc (pos + 20) _ ?_ ?_ ?_ ?_ hpos20 (by omega)).trans fun s1 ⟨hf1, hm1⟩ => ?_)
      · exact he.frame (S := [10, 11, 14, 15]) (by frame_tac) (by decide)
      · rsimp; omega
      · rsimp using hl; omega
      · rsimp using hl; rfl
      · have hf01 : Frame [10, 11, 14, 15] _ s1 := Frame.mono hf1 (by decide)
        have he1 : Env s1 b bs := he.frame (Frame.trans_same (by frame_tac) hf01) (by decide)
        have hl1 := he1.len
        have h5_1 := he1.rbp
        have h15_1 : s1.reg 15 = some (n + 1 + na * 2 ^ 32) := by
          rw [hf1.regs 15 (by decide)]; rsimp using hl; rfl
        cases hsca : skipCodeA bs nc (pos + 20) with
        | none =>
          rw [hsca] at hm1
          exact Reaches.refl ⟨hf01, by
            simp only [skipWordsA, hlt, reduceIte]
            rw [show skipCodeA bs (u32 bs (pos + 12)) (pos + 20) = none from hsca]
            exact hm1⟩
        | some p =>
          rw [hsca] at hm1
          have hp_le := skipCodeA_le hpos20 hsca
          xs f156 using hl1, h15_1
          xs f159 using hl1, show (n + 1 + na * 2 ^ 32) / 2 ^ 32 = na by omega
          xs f163 using hl1, h15_1, show (n + 1 + na * 2 ^ 32) % 2 ^ 32 = n + 1 by omega
          xs f166 using hl1
          xs f169 using hl1, show na * 8 % 18446744073709551616 = na * 8 by omega
          xs f173 using hl1, show na * 4 % 18446744073709551616 = na * 4 by omega
          xs f177 using hl1, add_small (show na * 8 + na * 4 < 2 ^ 64 by omega)
          xs f180 using hl1, hm1.2.1, add_small (show b + p + (na * 8 + na * 4) < 2 ^ 64 by omega)
          xs f183 using hl1, h5_1
          have hc7' := (cond_sub (show b + p + (na * 8 + na * 4) < W by rw [W_eq]; omega)
            (show b + bs.length < W by rw [W_eq]; omega)).2.2.2.2.2
          by_cases hlt2 : bs.length < p + 12 * na
          · xs f186 using hc7', show b + bs.length < b + p + (na * 8 + na * 4) by omega
            refine Reaches.refl ⟨Frame.trans_same hf01 (by frame_tac), ?_⟩
            simp only [skipWordsA, hlt, reduceIte]
            rw [show skipCodeA bs (u32 bs (pos + 12)) (pos + 20) = some p from hsca]
            simp [show bs.length < p + 12 * u32 bs (pos + 16) from hlt2]
          · xs f186 using hc7', show ¬ (b + bs.length < b + p + (na * 8 + na * 4)) by omega
            xs f192 using hl1, sub_small (show n + 1 < 2 ^ 64 by omega) (show 1 ≤ n + 1 by omega),
              Nat.add_sub_cancel
            xs f196
            refine reach_frame (Frame.trans_same hf01 (by frame_tac))
              ((ih (p + 12 * na) _ ?_ rfl ?_ ?_ (by omega) (by omega)).mono fun s3 ⟨hf3, hm3⟩ => ?_)
            · exact he1.frame (S := [10, 11, 14, 15]) (by frame_tac) (by decide)
            · rsimp using hl1; omega
            · rsimp using hl1
            · refine ⟨hf3, ?_⟩
              simp only [skipWordsA, hlt, reduceIte]
              rw [show skipCodeA bs (u32 bs (pos + 12)) (pos + 20) = some p from hsca]
              simp only [show ¬ bs.length < p + 12 * u32 bs (pos + 16) from hlt2, reduceIte]
              simpa [na] using hm3

end SporosCheckBin
