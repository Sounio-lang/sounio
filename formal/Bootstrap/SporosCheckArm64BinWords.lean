/-
SporosCheckArm64BinWords: annotation-check, word-loop, and full-pass binary
specifications (`anns_spec`, `checkWords_spec`, `pass_spec`, `p1_spec`, `p2_spec`, `p3_spec`)
for `spr1_check_arm64` over `AArch64Phys` / `AArch64Env`.
-/
import Bootstrap.SporosCheckArm64BinWalk

namespace SporosCheckArm64Bin

open AArch64Env AArch64Phys SporosCheckArm64Flat

private theorem modW_add {B k d : Nat} (hB : B + 2728 < 2 ^ 64) (hk : k + d ≤ 2728) :
    (B + k + d) % AArch64Env.W = B + (k + d) := by
  rw [W_eq, Nat.add_assoc]
  exact Nat.mod_eq_of_lt (Nat.lt_of_le_of_lt (Nat.add_le_add_left hk B) hB)

private theorem modW_back {B k off tgt : Nat} (hB : B + 2728 < 2 ^ 64)
    (h1 : k + off = 2 ^ 64 + tgt) (h2 : tgt ≤ 2728) :
    (B + k + off) % AArch64Env.W = B + tgt := by
  rw [W_eq, Nat.add_assoc, h1, show B + (2 ^ 64 + tgt) = (B + tgt) + 2 ^ 64 by omega,
    Nat.add_mod_right]
  exact Nat.mod_eq_of_lt (Nat.lt_of_le_of_lt (Nat.add_le_add_left h2 B) hB)

private theorem relOff_w {B k tgt : Nat} (hB : B + 2728 < 2 ^ 64)
    (hk : k ≤ 2728) (ht : tgt ≤ 2728) :
    (B + k + relOff (B + k) (B + tgt)) % AArch64Env.W = B + tgt :=
  relOff_tgt (Nat.lt_of_le_of_lt (Nat.add_le_add_left hk B) hB)
    (Nat.lt_of_le_of_lt (Nat.add_le_add_left ht B) hB)

private theorem rdMem_w4_at0 {m : Mem} {b off : Nat} {bs : List Nat}
    (hbuf : Buf m b bs) (hsm : b + bs.length < HMAX) (hw : off + 4 ≤ bs.length) :
    rdMem m (((b + off) % 2 ^ 64 + 0) % AArch64Env.W) .w4 = some (u32 bs off) := by
  have h1 : b + off < 2 ^ 64 := by unfold HMAX at hsm; omega
  rw [ea_mod_small h1 (by omega), Nat.add_zero]
  exact rdMem_w4_buf hbuf hw

private theorem anns_loop_spec {B adrImm R nxt b wordOff nc annOff na : Nat} {bs : List Nat} {m : Mem}
    (pb : PassBlocks B adrImm R nxt) (hB : B + 2728 < 2 ^ 64)
    (hbuf : Buf m b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < HMAX)
    (hw : wordOff + 20 ≤ bs.length) (hnc : u32 bs (wordOff + 12) = nc)
    (hna : u32 bs (wordOff + 16) = na) (hann : annOff + na * 12 ≤ bs.length) :
    ∀ (k off : Nat) (s : St),
      off + k * 12 = annOff + na * 12 →
      s.mem = m → s.regs.length = 31 → s.pc = B + 2544 →
      s.reg 6 = some (b + off) → s.reg 7 = some (b + (annOff + na * 12)) →
      s.reg 9 = some (b + annOff) → s.reg 12 = some (b + wordOff) →
      Reaches s fun s' => Frame [6, 10, 11, 12, 14, 15] s s' ∧
        if annsOK bs nc annOff na k off then
          s'.pc = B + 488 ∧ s'.regs.length = 31 ∧ s'.mem = m ∧ s'.reg 12 = some (b + (annOff + na * 12))
        else s'.pc = B + 2708
  | 0, off, s, heq, hm, hl, hpc, h6, h7, _, _ => by
    have hoff : off = annOff + na * 12 := by omega
    subst hoff
    have hcP := pb.cWdonePre; simp only [CodeAt_cons] at hcP; obtain ⟨_, _, _, _, f2544, f2548, _⟩ := hcP
    have hcE := pb.cAnnsEnd; simp only [CodeAt_cons] at hcE
    obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, f2660, f2664, _⟩ := hcE
    have hcs := cond_sub (show b + (annOff + na * 12) < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
      (show b + (annOff + na * 12) < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
    have hr2544 : (B + 2544 + 4) % AArch64Env.W = B + 2548 := modW_small (by omega)
    have hr2548 : (B + 2548 + 112) % AArch64Env.W = B + 2660 := modW_small (by omega)
    have hr2660 : (B + 2660 + 4) % AArch64Env.W = B + 2664 := modW_small (by omega)
    have hr2664 : (B + 2664 + 18446744073709549440) % AArch64Env.W = B + 488 := by rw [W_eq]; omega
    have hbm : (b + (annOff + na * 12)) % 2 ^ 64 = b + (annOff + na * 12) := Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega)
    xs f2544 using hl, h6, h7, hr2544
    have hfl : AArch64Phys.cond (addSubOp true 64 ((b + (annOff + na * 12)) % 2 ^ 64) ((b + (annOff + na * 12)) % 2 ^ 64)).2 0 = some true := by
      rw [hbm, hcs.1]; simp
    xs f2548 using hfl, hr2548
    xs f2660 using hl, h7, hbm, hr2660
    xs f2664 using hr2664
    exact Reaches.refl ⟨by frame_tac, rfl, by simp [hl], hm, by simp [hl]⟩
  | k + 1, off, s, heq, hm, hl, hpc, h6, h7, h9, h12 => by
    have hne : off ≠ annOff + na * 12 := by omega
    have hnc_lt := u32_lt hby (wordOff + 12); rw [hnc] at hnc_lt
    have ht_lt := u32_lt hby off
    have had_lt := u32_lt hby (off + 4)
    have haa_lt := u32_lt hby (off + 8)
    have hru0 := rdU32_eq_some_u32 hby (by omega : off + 4 ≤ bs.length)
    have hru4 := rdU32_eq_some_u32 hby (by omega : off + 4 + 4 ≤ bs.length)
    have hru8 := rdU32_eq_some_u32 hby (by omega : off + 8 + 4 ≤ bs.length)
    have hcP := pb.cWdonePre; simp only [CodeAt_cons] at hcP
    obtain ⟨_, _, _, _, f2544, f2548, f2552, f2556, f2560, f2564, _⟩ := hcP
    have hcE := pb.cAnnsEnd; simp only [CodeAt_cons] at hcE
    obtain ⟨f2612, f2616, f2620, f2624, f2628, f2632, f2636, f2640, f2644, f2648, f2652, f2656, _⟩ := hcE
    have hcs0 := cond_sub (show b + off < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
      (show b + (annOff + na * 12) < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
    have hrd_t := rdMem_w4_at0 hbuf hsm (by omega : off + 4 ≤ bs.length)
    have hrd_nc := rdMem_w4_at hbuf hsm (by omega : wordOff + 12 + 4 ≤ bs.length); rw [hnc] at hrd_nc
    xs f2544 using hl, h6, h7, modW_add hB (k := 2544) (d := 4) (by decide)
    have hfl0 : AArch64Phys.cond (addSubOp true 64 ((b + off) % 2 ^ 64)
        ((b + (annOff + na * 12)) % 2 ^ 64)).2 0 = some false := by
      rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega),
        Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs0.1]
      simp [hne]
    xs f2548 using hfl0, modW_add hB (k := 2548) (d := 4) (by decide)
    xs f2552 using hl, hm, h6, hrd_t, ht_lt, modW_add hB (k := 2552) (d := 4) (by decide)
    xs f2556 using hl, hm, h12, hrd_nc, hnc_lt, modW_add hB (k := 2556) (d := 4) (by decide)
    have hcs1 := cond_sub (show u32 bs off < pow2 64 by rw [pow2_64]; omega)
      (show nc < pow2 64 by rw [pow2_64]; omega)
    xs f2560 using hl, modW_add hB (k := 2560) (d := 4) (by decide)
    by_cases hge : nc ≤ u32 bs off
    · have hfl1 : AArch64Phys.cond (addSubOp true 64 (u32 bs off % 2 ^ 64) (nc % 2 ^ 64)).2 2 = some true := by
        rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega), hcs1.2.2.1]; simp [hge]
      xs f2564 using hfl1, relOff_w hB (k := 2564) (tgt := 2708) (by decide) (by decide)
      simp only [annsOK, hru0, hru4, hru8, show ¬ u32 bs off < nc by omega, decide_false, Bool.false_and,
        Bool.false_eq_true, ↓reduceIte]
      exact Reaches.refl ⟨by frame_tac, rfl⟩
    · have hfl1 : AArch64Phys.cond (addSubOp true 64 (u32 bs off % 2 ^ 64) (nc % 2 ^ 64)).2 2 = some false := by
        rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega), hcs1.2.2.1]; simp [hge]
      xs f2564 using hfl1, modW_add hB (k := 2564) (d := 4) (by decide)
      refine (lookup_spec pb.cLuAnns (by omega) (Or.inr rfl) (by simpa [hm] using hbuf) hby hsm hw hna hann
        (by omega : u32 bs off < 2 ^ 64) (by simp [hl]) rfl (by simp [h12]) (by simp [h9]) (by simp [hl])).trans
        fun s1 ⟨hf1, hpc1, hs1_11, hs1_len, hlu⟩ => ?_
      have hs1_mem : s1.mem = m := by rw [hf1.mem]; exact hm
      have hs1_6 : s1.reg 6 = some (b + off) := by rw [hf1.regs 6 (by decide)]; simp [h6]
      have hs1_7 : s1.reg 7 = some (b + (annOff + na * 12)) := by rw [hf1.regs 7 (by decide)]; simp [h7]
      have hs1_9 : s1.reg 9 = some (b + annOff) := by rw [hf1.regs 9 (by decide)]; simp [h9]
      have hs1_12 : s1.reg 12 = some (b + wordOff) := by rw [hf1.regs 12 (by decide)]; simp [h12]
      cases hlu_eq : lookup bs annOff na (u32 bs off) with
      | none =>
        rw [hlu_eq] at hlu
        have hcs2 := cond_sub (show b + (annOff + na * 12) < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
          (show b + (annOff + na * 12) < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
        xs f2612 using hs1_len, hlu, hs1_11, modW_add hB (k := 2612) (d := 4) (by decide)
        have hfl2 : AArch64Phys.cond (addSubOp true 64 ((b + (annOff + na * 12)) % 2 ^ 64)
            ((b + (annOff + na * 12)) % 2 ^ 64)).2 0 = some true := by
          rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs2.1]; simp
        xs f2616 using hfl2, relOff_w hB (k := 2616) (tgt := 2708) (by decide) (by decide)
        simp only [annsOK, hru0, hru4, hru8, hlu_eq,
          show (none == some (u32 bs (off + 4), u32 bs (off + 8))) = false by rfl,
          Bool.and_false, Bool.false_and, Bool.false_eq_true, ↓reduceIte]
        exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide)) (by frame_tac)), rfl⟩
      | some pa =>
        rcases pa with ⟨ad, aa⟩
        rw [hlu_eq] at hlu
        obtain ⟨q, _, hq12, hs1_10, hqne, rfl, rfl⟩ := hlu
        have hqad := u32_lt hby (q + 4); have hqaa := u32_lt hby (q + 8)
        have hcs2 := cond_sub (show b + q < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
          (show b + (annOff + na * 12) < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
        have hrd_q4 := rdMem_w4_at (m := s1.mem) (by simpa [hs1_mem] using hbuf) hsm (by omega : q + 4 + 4 ≤ bs.length)
        have hrd_o4 := rdMem_w4_at (m := s1.mem) (by simpa [hs1_mem] using hbuf) hsm (by omega : off + 4 + 4 ≤ bs.length)
        xs f2612 using hs1_len, hs1_10, hs1_11, modW_add hB (k := 2612) (d := 4) (by decide)
        have hfl2 : AArch64Phys.cond (addSubOp true 64 ((b + q) % 2 ^ 64)
            ((b + (annOff + na * 12)) % 2 ^ 64)).2 0 = some false := by
          rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega),
            Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs2.1]
          simp [hqne]
        xs f2616 using hfl2, modW_add hB (k := 2616) (d := 4) (by decide)
        xs f2620 using hs1_len, hs1_10, hrd_q4, hqad, modW_add hB (k := 2620) (d := 4) (by decide)
        xs f2624 using hs1_len, hs1_6, hrd_o4, had_lt, modW_add hB (k := 2624) (d := 4) (by decide)
        have hcs3 := cond_sub (show u32 bs (q + 4) < pow2 64 by rw [pow2_64]; omega)
          (show u32 bs (off + 4) < pow2 64 by rw [pow2_64]; omega)
        xs f2628 using hs1_len, modW_add hB (k := 2628) (d := 4) (by decide)
        by_cases heqd : u32 bs (q + 4) = u32 bs (off + 4)
        · have hfl3 : AArch64Phys.cond (addSubOp true 64 (u32 bs (q + 4) % 2 ^ 64)
              (u32 bs (off + 4) % 2 ^ 64)).2 1 = some false := by
            rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega), hcs3.2.1]; simp [heqd]
          have hrd_q8 := rdMem_w4_at (m := s1.mem) (by simpa [hs1_mem] using hbuf) hsm (by omega : q + 8 + 4 ≤ bs.length)
          have hrd_o8 := rdMem_w4_at (m := s1.mem) (by simpa [hs1_mem] using hbuf) hsm (by omega : off + 8 + 4 ≤ bs.length)
          xs f2632 using hfl3, modW_add hB (k := 2632) (d := 4) (by decide)
          xs f2636 using hs1_len, hs1_10, hrd_q8, hqaa, modW_add hB (k := 2636) (d := 4) (by decide)
          xs f2640 using hs1_len, hs1_6, hrd_o8, haa_lt, modW_add hB (k := 2640) (d := 4) (by decide)
          have hcs4 := cond_sub (show u32 bs (q + 8) < pow2 64 by rw [pow2_64]; omega)
            (show u32 bs (off + 8) < pow2 64 by rw [pow2_64]; omega)
          xs f2644 using hs1_len, modW_add hB (k := 2644) (d := 4) (by decide)
          by_cases heqa : u32 bs (q + 8) = u32 bs (off + 8)
          · have hfl4 : AArch64Phys.cond (addSubOp true 64 (u32 bs (q + 8) % 2 ^ 64)
                (u32 bs (off + 8) % 2 ^ 64)).2 1 = some false := by
              rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega), hcs4.2.1]; simp [heqa]
            have hoff12 : ((b + off) % 2 ^ 64 + 12) % 2 ^ 64 = b + (off + 12) := by
              unfold HMAX at hsm; omega
            xs f2648 using hfl4, modW_add hB (k := 2648) (d := 4) (by decide)
            xs f2652 using hs1_len, hs1_6, hoff12, modW_add hB (k := 2652) (d := 4) (by decide)
            xs f2656 using modW_back hB (k := 2656) (off := 18446744073709551504) (tgt := 2544) (by decide) (by decide)
            refine (anns_loop_spec pb hB hbuf hby hsm hw hnc hna hann k (off + 12) _ (by omega)
              (by simp [hs1_mem]) (by simp [hs1_len]) rfl (by simp [hs1_len]) (by simp [hs1_7]) (by simp [hs1_9])
              (by simp [hs1_12])).mono fun s2 ⟨hf2, hres2⟩ => ?_
            refine ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide))
              (Frame.trans_same (by frame_tac) hf2)), ?_⟩
            simpa [annsOK, hru0, hru4, hru8, show u32 bs off < nc by omega, hlu_eq, heqd, heqa] using hres2
          · have hfl4 : AArch64Phys.cond (addSubOp true 64 (u32 bs (q + 8) % 2 ^ 64)
                (u32 bs (off + 8) % 2 ^ 64)).2 1 = some true := by
              rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega), hcs4.2.1]; simp [heqa]
            xs f2648 using hfl4, relOff_w hB (k := 2648) (tgt := 2708) (by decide) (by decide)
            have hne_pair : (some (u32 bs (q + 4), u32 bs (q + 8)) == some (u32 bs (off + 4), u32 bs (off + 8))) = false := by
              simp [heqa]
            simp only [annsOK, hru0, hru4, hru8, hlu_eq, hne_pair, Bool.and_false, Bool.false_and,
              Bool.false_eq_true, ↓reduceIte]
            exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide)) (by frame_tac)), rfl⟩
        · have hfl3 : AArch64Phys.cond (addSubOp true 64 (u32 bs (q + 4) % 2 ^ 64)
              (u32 bs (off + 4) % 2 ^ 64)).2 1 = some true := by
            rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega), hcs3.2.1]; simp [heqd]
          xs f2632 using hfl3, relOff_w hB (k := 2632) (tgt := 2708) (by decide) (by decide)
          have hne_pair : (some (u32 bs (q + 4), u32 bs (q + 8)) == some (u32 bs (off + 4), u32 bs (off + 8))) = false := by
            simp [heqd]
          simp only [annsOK, hru0, hru4, hru8, hlu_eq, hne_pair, Bool.and_false, Bool.false_and,
            Bool.false_eq_true, ↓reduceIte]
          exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide)) (by frame_tac)), rfl⟩

theorem anns_spec {B adrImm R nxt b wordOff nc annOff na : Nat} {bs : List Nat} {m : Mem} {s : St}
    (pb : PassBlocks B adrImm R nxt) (hB : B + 2728 < 2 ^ 64)
    (hbuf : Buf m b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < HMAX)
    (hw : wordOff + 20 ≤ bs.length) (hnc : u32 bs (wordOff + 12) = nc)
    (hna : u32 bs (wordOff + 16) = na) (hann : annOff + na * 12 ≤ bs.length)
    (hm : s.mem = m) (hl : s.regs.length = 31) (hpc : s.pc = B + 2528)
    (h9 : s.reg 9 = some (b + annOff)) (h12 : s.reg 12 = some (b + wordOff)) :
    Reaches s fun s' => Frame [6, 7, 10, 11, 12, 14, 15] s s' ∧
      if annsOK bs nc annOff na na annOff then
        s'.pc = B + 488 ∧ s'.regs.length = 31 ∧ s'.mem = m ∧ s'.reg 12 = some (b + (annOff + na * 12))
      else s'.pc = B + 2708 := by
  have hcP := pb.cWdonePre; simp only [CodeAt_cons] at hcP; obtain ⟨f2528, f2532, f2536, f2540, _⟩ := hcP
  have hlt_na := u32_lt hby (wordOff + 16); rw [hna] at hlt_na
  have hrd := rdMem_w4_at hbuf hsm (by omega : wordOff + 16 + 4 ≤ bs.length); rw [hna] at hrd
  have hr2528 : (B + 2528 + 4) % AArch64Env.W = B + 2532 := modW_small (by omega)
  have hr2532 : (B + 2532 + 4) % AArch64Env.W = B + 2536 := modW_small (by omega)
  have hr2536 : (B + 2536 + 4) % AArch64Env.W = B + 2540 := modW_small (by omega)
  have hr2540 : (B + 2540 + 4) % AArch64Env.W = B + 2544 := modW_small (by omega)
  have h3na : (na % 2 ^ 64 + na % 2 ^ 64 * 2 ^ 1 % 2 ^ 64) % 2 ^ 64 = 3 * na := by omega
  have hbam : (b + annOff) % 2 ^ 64 = b + annOff := Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega)
  have h7_val : ((b + annOff) % 2 ^ 64 + (3 * na) % 2 ^ 64 * 2 ^ 2 % 2 ^ 64) % 2 ^ 64 =
      b + (annOff + na * 12) := by
    rw [hbam, Nat.mod_eq_of_lt (by omega : 3 * na < 2 ^ 64),
      Nat.mod_eq_of_lt (by omega : 3 * na * 2 ^ 2 < 2 ^ 64)]
    unfold HMAX at hsm; rw [Nat.mod_eq_of_lt (by omega)]; omega
  xs f2528 using hl, hm, h12, hrd, hlt_na, hr2528
  xs f2532 using hl, h3na, hr2532
  xs f2536 using hl, h9, h7_val, hr2536
  xs f2540 using hl, h9, hbam, hr2540
  refine (anns_loop_spec pb hB hbuf hby hsm hw hnc hna hann na annOff _ rfl
    (by simp [hm]) (by simp [hl]) rfl (by simp [hl]) (by simp [hl]) (by simp [h9]) (by simp [h12])).mono fun s' ⟨hf, hres⟩ => ?_
  exact ⟨Frame.trans_same (by frame_tac) (Frame.mono hf (by decide)), hres⟩

theorem checkWords_spec {B adrImm R nxt b ws nw : Nat} {bs : List Nat} {m : Mem}
    (pb : PassBlocks B adrImm R nxt) (hB : B + 2728 < 2 ^ 64)
    (hbuf : Buf m b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < HMAX)
    (hsw_all : skipWords bs nw ws = some bs.length) (hws_le : ws ≤ bs.length) (hnw : nw < 2 ^ 32) :
    ∀ (rem off : Nat) (s : St),
      skipWords bs rem off = some bs.length → off ≤ bs.length →
      s.mem = m → s.regs.length = 31 → s.pc = B + 488 →
      s.reg 1 = some nw → s.reg 12 = some (b + off) →
      s.reg 24 = some (b + bs.length) → s.reg 25 = some (b + ws) →
      Reaches s fun s' => Frame PassScratch s s' ∧
        match checkWords bs nw ws rem off with
        | .ok => s'.pc = B + 2668 ∧ s'.regs.length = 31 ∧ s'.mem = m
        | .reject => s'.pc = B + 2708
        | .overflow => s'.pc = B + 2688
  | 0, off, s, hsw, _, hm, hl, hpc, _, h12, h24, _ => by
    simp only [skipWords, Option.some.injEq] at hsw; subst hsw
    have hcM3 := pb.cMid3; simp only [CodeAt_cons] at hcM3; obtain ⟨_, _, _, f488, f492, _⟩ := hcM3
    have hcs := cond_sub (show b + bs.length < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
      (show b + bs.length < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
    have hr488 : (B + 488 + 4) % AArch64Env.W = B + 492 := modW_small (by omega)
    have hr492 : (B + 492 + relOff (B + 492) (B + 2668)) % AArch64Env.W = B + 2668 :=
      relOff_tgt (by omega) (by omega)
    xs f488 using hl, h12, h24, hr488
    have hfl : AArch64Phys.cond (addSubOp true 64 ((b + bs.length) % 2 ^ 64) ((b + bs.length) % 2 ^ 64)).2 0 = some true := by
      rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs.1]; simp
    xs f492 using hfl, hr492
    simp only [checkWords]
    exact Reaches.refl ⟨by frame_tac, rfl, hl, hm⟩
  | k + 1, off, s, hsw, _, hm, hl, hpc, h1, h12, h24, h25 => by
    obtain ⟨wm, hrw, _, h20_nxt, hle_nxt, hsw_rest⟩ := skipWords_succ_iff.1 hsw
    obtain ⟨h20, hins, houts, hlocs, hnc, hna, hcodeOff, hsc_wm, hann_le, hnextOff, hann_end⟩ :=
      readWordAt_fields hby hrw
    have hne : off ≠ bs.length := by omega
    have hnc_lt := u32_lt hby (off + 12); rw [hnc] at hnc_lt
    have hins_lt := u32_lt hby off; rw [hins] at hins_lt
    have hcM3 := pb.cMid3; simp only [CodeAt_cons] at hcM3; obtain ⟨_, _, _, f488, f492, f496, f500, f504, _⟩ := hcM3
    have hcMov := pb.cMovIdx0; simp only [CodeAt_cons] at hcMov; obtain ⟨f612, _⟩ := hcMov
    have hcW := pb.cWalk0; simp only [CodeAt_cons] at hcW
    obtain ⟨f660, f664, f668, f672, f676, f680, f684, f688, f692, f696, f700, _⟩ := hcW
    have hcs0 := cond_sub (show b + off < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
      (show b + bs.length < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
    have hrd_nc := rdMem_w4_at hbuf hsm (by omega : off + 12 + 4 ≤ bs.length); rw [hnc] at hrd_nc
    xs f488 using hl, h12, h24, modW_add hB (k := 488) (d := 4) (by decide)
    have hfl0 : AArch64Phys.cond (addSubOp true 64 ((b + off) % 2 ^ 64) ((b + bs.length) % 2 ^ 64)).2 0 = some false := by
      rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega),
        Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs0.1]; simp [hne]
    xs f492 using hfl0, modW_add hB (k := 492) (d := 4) (by decide)
    xs f496 using hl, hm, h12, hrd_nc, hnc_lt, modW_add hB (k := 496) (d := 4) (by decide)
    by_cases hnc0 : wm.nc = 0
    · xs f500 using hl, Nat.mod_eq_of_lt (by omega : wm.nc < 2 ^ 64), hnc0, relOff_w hB (k := 500) (tgt := 2708) (by decide) (by decide)
      simp only [checkWords, hrw, hnc0, beq_self_eq_true, ↓reduceIte]
      exact Reaches.refl ⟨by frame_tac, rfl⟩
    · have hbo20 : ((b + off) % 2 ^ 64 + 20 % 2 ^ 64) % 2 ^ 64 = b + (off + 20) := by
        unfold HMAX at hsm; omega
      xs f500 using hl, Nat.mod_eq_of_lt (by omega : wm.nc < 2 ^ 64), hnc0, modW_add hB (k := 500) (d := 4) (by decide)
      xs f504 using hl, h12, hbo20, modW_add hB (k := 504) (d := 4) (by decide)
      refine (skipcode_spec pb.cSc (Or.inl rfl) (by omega) (by omega) hbuf hby hsm wm.nc (off + 20) _
        (by simp [hm]) (by simp [hl]) rfl (by simp [hl]) (by simp [hl]) (by simp [h24]) h20 (by omega)).trans
        fun s1 ⟨hf1, hsc1⟩ => ?_
      rw [hsc_wm] at hsc1
      obtain ⟨hpc1, hs1_9, _, _⟩ := hsc1
      have hs1_len : s1.regs.length = 31 := by rw [hf1.len]; simp [hl]
      have hs1_mem : s1.mem = m := by rw [hf1.mem]; exact hm
      have hs1_1 : s1.reg 1 = some nw := by rw [hf1.regs 1 (by decide)]; simp [h1]
      have hs1_12 : s1.reg 12 = some (b + off) := by rw [hf1.regs 12 (by decide)]; simp [h12]
      have hs1_24 : s1.reg 24 = some (b + bs.length) := by rw [hf1.regs 24 (by decide)]; simp [h24]
      have hs1_25 : s1.reg 25 = some (b + ws) := by rw [hf1.regs 25 (by decide)]; simp [h25]
      xs f612 using hs1_len, modW_add hB (k := 612) (d := 4) (by decide)
      have hann_fit : wm.annOff + wm.na * 12 ≤ bs.length := by rw [← hnextOff]; exact hann_end
      refine (lookup_spec pb.cLu0 (by omega) (Or.inl rfl) (by simpa [hs1_mem] using hbuf) hby hsm h20 hna hann_fit
        (by decide : (0 : Nat) < 2 ^ 64) (by simp [hs1_len]) rfl (by simp [hs1_12]) (by simp [hs1_9])
        (by simp [hs1_len])).trans fun s2 ⟨hf2, hpc2, hs2_11, hs2_len, hlu0⟩ => ?_
      have hs2_mem : s2.mem = m := by rw [hf2.mem]; exact hs1_mem
      have hs2_1 : s2.reg 1 = some nw := by rw [hf2.regs 1 (by decide)]; simp [hs1_1]
      have hs2_8 : s2.reg 8 = some 0 := by rw [hf2.regs 8 (by decide)]; simp [hs1_len]
      have hs2_9 : s2.reg 9 = some (b + wm.annOff) := by rw [hf2.regs 9 (by decide)]; simp [hs1_9]
      have hs2_12 : s2.reg 12 = some (b + off) := by rw [hf2.regs 12 (by decide)]; simp [hs1_12]
      have hs2_24 : s2.reg 24 = some (b + bs.length) := by rw [hf2.regs 24 (by decide)]; simp [hs1_24]
      have hs2_25 : s2.reg 25 = some (b + ws) := by rw [hf2.regs 25 (by decide)]; simp [hs1_25]
      have hcont : ∀ (s3 : St), s3.pc = B + 692 → s3.regs.length = 31 → s3.mem = m →
          s3.reg 1 = some nw → s3.reg 8 = some 0 → s3.reg 9 = some (b + wm.annOff) →
          s3.reg 12 = some (b + off) → s3.reg 24 = some (b + bs.length) → s3.reg 25 = some (b + ws) →
          Reaches s3 fun s' => Frame PassScratch s3 s' ∧
            match (match walk bs nw ws wm.outs wm.locals wm.nc wm.annOff wm.na wm.nc 0 wm.codeOff wm.ins 0 with
              | .overflow => CheckRes.overflow
              | .reject => CheckRes.reject
              | .ok => if annsOK bs wm.nc wm.annOff wm.na wm.na wm.annOff then checkWords bs nw ws k wm.nextOff
                else CheckRes.reject) with
            | .ok => s'.pc = B + 2668 ∧ s'.regs.length = 31 ∧ s'.mem = m
            | .reject => s'.pc = B + 2708
            | .overflow => s'.pc = B + 2688 := by
        intro s3 hpc3 hs3_len hs3_mem hs3_1 hs3_8 hs3_9 hs3_12 hs3_24 hs3_25
        have hrd_ins := rdMem_w4_at0 (m := s3.mem) (by simpa [hs3_mem] using hbuf) hsm (by omega : off + 4 ≤ bs.length)
        rw [hins] at hrd_ins
        xs f692 using hs3_len, hs3_12, hrd_ins, hins_lt, modW_add hB (k := 692) (d := 4) (by decide)
        xs f696 using hs3_len, modW_add hB (k := 696) (d := 4) (by decide)
        xs f700 using hs3_len, hs3_12, hbo20, modW_add hB (k := 700) (d := 4) (by decide)
        rw [hcodeOff]
        refine (walk_spec pb hB hbuf hby hsm h20 houts hlocs hnc hna hann_fit hsw_all hws_le hnw
          wm.nc 0 (off + 20) wm.ins 0 _ (by omega) hsc_wm h20 (by simp [hs3_mem]) (by simp [hs3_len]) rfl
          (by simp [hs3_1]) (by simp [hs3_len]) (by simp [hs3_8]) (by simp [hs3_9]) (by simp [hs3_12])
          (by simp [hs3_24]) (by simp [hs3_25])
          ⟨fun _ => ⟨by simp [hs3_len], by simp [hs3_len]⟩, wm.ins, 0, by simp [hs3_len], by simp [hs3_len],
            by rw [← hins]; exact u32_lt_HMAX hby off, by decide⟩).trans fun s4 ⟨hf4, hw_res⟩ => ?_
        cases hwk : walk bs nw ws wm.outs wm.locals wm.nc wm.annOff wm.na wm.nc 0 (off + 20) wm.ins 0 with
        | overflow => rw [hwk] at hw_res; exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.mono hf4 (by decide)), hw_res⟩
        | reject => rw [hwk] at hw_res; exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.mono hf4 (by decide)), hw_res⟩
        | ok =>
          rw [hwk] at hw_res
          obtain ⟨hpc4, hs4_len, hs4_mem⟩ := hw_res
          have hs4_1 : s4.reg 1 = some nw := by rw [hf4.regs 1 (by decide)]; simp [hs3_1]
          have hs4_9 : s4.reg 9 = some (b + wm.annOff) := by rw [hf4.regs 9 (by decide)]; simp [hs3_9]
          have hs4_12 : s4.reg 12 = some (b + off) := by rw [hf4.regs 12 (by decide)]; simp [hs3_12]
          have hs4_24 : s4.reg 24 = some (b + bs.length) := by rw [hf4.regs 24 (by decide)]; simp [hs3_24]
          have hs4_25 : s4.reg 25 = some (b + ws) := by rw [hf4.regs 25 (by decide)]; simp [hs3_25]
          refine (anns_spec pb hB hbuf hby hsm h20 hnc hna hann_fit hs4_mem hs4_len hpc4 hs4_9 hs4_12).trans
            fun s5 ⟨hf5, ha_res⟩ => ?_
          cases ha_eq : annsOK bs wm.nc wm.annOff wm.na wm.na wm.annOff with
          | false =>
            rw [ha_eq] at ha_res
            exact Reaches.refl ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf4 (by decide))
              (Frame.mono hf5 (by decide))), ha_res⟩
          | true =>
            rw [ha_eq] at ha_res
            obtain ⟨hpc5, hs5_len, hs5_mem, hs5_12⟩ := ha_res
            rw [← hnextOff] at hs5_12
            have hs5_1 : s5.reg 1 = some nw := by rw [hf5.regs 1 (by decide)]; exact hs4_1
            have hs5_24 : s5.reg 24 = some (b + bs.length) := by rw [hf5.regs 24 (by decide)]; exact hs4_24
            have hs5_25 : s5.reg 25 = some (b + ws) := by rw [hf5.regs 25 (by decide)]; exact hs4_25
            refine (checkWords_spec pb hB hbuf hby hsm hsw_all hws_le hnw k wm.nextOff s5 hsw_rest hle_nxt
              hs5_mem hs5_len hpc5 hs5_1 hs5_12 hs5_24 hs5_25).mono fun s6 ⟨hf6, hres6⟩ => ?_
            exact ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf4 (by decide))
              (Frame.trans_same (Frame.mono hf5 (by decide)) hf6)), hres6⟩
      have hf02 : Frame PassScratch s s2 :=
        Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide))
          (Frame.trans_same (by frame_tac) (Frame.mono hf2 (by decide))))
      cases hlu0_eq : lookup bs wm.annOff wm.na 0 with
      | none =>
        rw [hlu0_eq] at hlu0
        have hcs1 := cond_sub (show b + (wm.annOff + wm.na * 12) < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
          (show b + (wm.annOff + wm.na * 12) < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
        xs f660 using hs2_len, hlu0, hs2_11, modW_add hB (k := 660) (d := 4) (by decide)
        have hfl1 : AArch64Phys.cond (addSubOp true 64 ((b + (wm.annOff + wm.na * 12)) % 2 ^ 64)
            ((b + (wm.annOff + wm.na * 12)) % 2 ^ 64)).2 0 = some true := by
          rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs1.1]; simp
        xs f664 using hfl1, modW_add hB (k := 664) (d := 28) (by decide)
        refine (hcont _ rfl (by simp [hs2_len]) (by simp [hs2_mem]) (by simp [hs2_1]) (by simp [hs2_8]) (by simp [hs2_9]) (by simp [hs2_12]) (by simp [hs2_24]) (by simp [hs2_25])).mono fun s' ⟨hf', hres'⟩ => ?_
        refine ⟨Frame.trans_same hf02 (Frame.trans_same (by frame_tac) hf'), ?_⟩
        cases hwlk : walk bs nw ws wm.outs wm.locals wm.nc wm.annOff wm.na wm.nc 0 wm.codeOff wm.ins 0 <;>
          simpa [checkWords, hrw, beq_false_of_ne hnc0, hlu0_eq, hwlk] using hres'
      | some p0 =>
        rcases p0 with ⟨d0, a0⟩
        rw [hlu0_eq] at hlu0
        obtain ⟨q, _, hq12, hs2_10, hqne, rfl, rfl⟩ := hlu0
        have hqd := u32_lt hby (q + 4); have hqa := u32_lt hby (q + 8)
        have hcs1 := cond_sub (show b + q < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
          (show b + (wm.annOff + wm.na * 12) < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
        have hrd_q4 := rdMem_w4_at (m := s2.mem) (by simpa [hs2_mem] using hbuf) hsm (by omega : q + 4 + 4 ≤ bs.length)
        have hrd_ins := rdMem_w4_at0 (m := s2.mem) (by simpa [hs2_mem] using hbuf) hsm (by omega : off + 4 ≤ bs.length)
        rw [hins] at hrd_ins
        xs f660 using hs2_len, hs2_10, hs2_11, modW_add hB (k := 660) (d := 4) (by decide)
        have hfl1 : AArch64Phys.cond (addSubOp true 64 ((b + q) % 2 ^ 64)
            ((b + (wm.annOff + wm.na * 12)) % 2 ^ 64)).2 0 = some false := by
          rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega),
            Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs1.1]
          simp [hqne]
        xs f664 using hfl1, modW_add hB (k := 664) (d := 4) (by decide)
        xs f668 using hs2_len, hs2_10, hrd_q4, hqd, modW_add hB (k := 668) (d := 4) (by decide)
        xs f672 using hs2_len, hs2_12, hrd_ins, hins_lt, modW_add hB (k := 672) (d := 4) (by decide)
        have hcs2 := cond_sub (show u32 bs (q + 4) < pow2 64 by rw [pow2_64]; omega)
          (show wm.ins < pow2 64 by rw [pow2_64]; omega)
        xs f676 using hs2_len, modW_add hB (k := 676) (d := 4) (by decide)
        by_cases heqd : u32 bs (q + 4) = wm.ins
        · have hfl2 : AArch64Phys.cond (addSubOp true 64 (u32 bs (q + 4) % 2 ^ 64) (wm.ins % 2 ^ 64)).2 1 = some false := by
            rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega), hcs2.2.1]; simp [heqd]
          have hrd_q8 := rdMem_w4_at (m := s2.mem) (by simpa [hs2_mem] using hbuf) hsm (by omega : q + 8 + 4 ≤ bs.length)
          xs f680 using hfl2, modW_add hB (k := 680) (d := 4) (by decide)
          xs f684 using hs2_len, hs2_10, hrd_q8, hqa, modW_add hB (k := 684) (d := 4) (by decide)
          by_cases ha0 : u32 bs (q + 8) = 0
          · xs f688 using hs2_len, Nat.mod_eq_of_lt (by omega : u32 bs (q + 8) < 2 ^ 64), ha0, modW_add hB (k := 688) (d := 4) (by decide)
            refine (hcont _ rfl (by simp [hs2_len]) (by simp [hs2_mem]) (by simp [hs2_1]) (by simp [hs2_8]) (by simp [hs2_9])
              (by simp [hs2_12]) (by simp [hs2_24]) (by simp [hs2_25])).mono fun s' ⟨hf', hres'⟩ => ?_
            refine ⟨Frame.trans_same hf02 (Frame.trans_same (by frame_tac) hf'), ?_⟩
            cases hwlk : walk bs nw ws wm.outs wm.locals wm.nc wm.annOff wm.na wm.nc 0 wm.codeOff wm.ins 0 <;>
              simpa [checkWords, hrw, beq_false_of_ne hnc0, hlu0_eq, heqd, ha0, hwlk] using hres'
          · xs f688 using hs2_len, Nat.mod_eq_of_lt (by omega : u32 bs (q + 8) < 2 ^ 64), ha0, relOff_w hB (k := 688) (tgt := 2708) (by decide) (by decide)
            simp only [checkWords, hrw, beq_false_of_ne hnc0, Bool.false_eq_true, ↓reduceIte, hlu0_eq,
              heqd, beq_self_eq_true, beq_false_of_ne ha0, Bool.and_false, Bool.not_false]
            exact Reaches.refl ⟨Frame.trans_same hf02 (by frame_tac), rfl⟩
        · have hfl2 : AArch64Phys.cond (addSubOp true 64 (u32 bs (q + 4) % 2 ^ 64) (wm.ins % 2 ^ 64)).2 1 = some true := by
            rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega), hcs2.2.1]; simp [heqd]
          xs f680 using hfl2, relOff_w hB (k := 680) (tgt := 2708) (by decide) (by decide)
          simp only [checkWords, hrw, beq_false_of_ne hnc0, Bool.false_eq_true, ↓reduceIte, hlu0_eq,
            beq_false_of_ne heqd, Bool.false_and, Bool.not_false]
          exact Reaches.refl ⟨Frame.trans_same hf02 (by frame_tac), rfl⟩

theorem pass_spec {B adrImm R nxt D b : Nat} {bs : List Nat} {m : Mem} {s : St}
    (hc : CodeAt (passCode B adrImm R nxt))
    (hB : B + 2728 < 2 ^ 64) (hnxt : nxt < 2 ^ 64) (hnxt_ge : B + 2728 ≤ nxt)
    (hD : D + 16 < 2 ^ 64) (hR : R < 31)
    (hadr : ((B / 4096) * 4096 + adrImm) % AArch64Env.W = D)
    (hD0 : rdMem m D .x8 = some b) (hD8 : rdMem m (D + 8) .x8 = some bs.length)
    (hbuf : Buf m b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < HMAX)
    (hl : s.regs.length = 31) (hpc : s.pc = B) (hm : s.mem = m) :
    Reaches s fun s' => Frame (R :: PassScratch) s s' ∧
      s'.pc = nxt ∧ s'.reg R = some (checkFlatWord bs) ∧ s'.regs.length = 31 ∧ s'.mem = m := by
  have pb := passBlocks_of_codeAt hc
  have hcbA := pb.cAccB; simp only [CodeAt_cons] at hcbA; obtain ⟨fAccB, _⟩ := hcbA
  have hcbO := pb.cOvfB; simp only [CodeAt_cons] at hcbO; obtain ⟨fOvfB, _⟩ := hcbO
  have hcbR := pb.cRejB; simp only [CodeAt_cons] at hcbR; obtain ⟨fRejB, _⟩ := hcbR
  have hall : bs.all (· < 256) = true := List.all_eq_true.2 fun x hx => decide_eq_true (hby x hx)
  have hltW : ¬ bs.length ≥ SporosCheckArm64Flat.W := by unfold SporosCheckArm64Flat.W; unfold HMAX at hsm; omega
  have hcf_res : ∀ r, checkFlatRes bs = some r →
      checkFlatWord bs = match r with | .ok => ACCEPT | .reject => REJECT | .overflow => OVERFLOW := by
    intro r hr; unfold checkFlatWord checkFlat; rw [hr]; cases r <;> rfl
  have finish_rej : ∀ s1, Frame PassScratch s s1 → s1.pc = B + 2708 → checkFlatRes bs = some .reject →
      Reaches s1 fun s' => Frame (R :: PassScratch) s s' ∧
        s'.pc = nxt ∧ s'.reg R = some (checkFlatWord bs) ∧ s'.regs.length = 31 ∧ s'.mem = m := by
    intro s1 hf1 hpc1 hres
    refine (verdict_spec pb.cRej fRejB (by omega) hnxt (by omega) (by decide) hR (by rw [hf1.len]; exact hl) hpc1).mono
      fun s2 ⟨hf2, hpc2, hR2, hl2⟩ => ?_
    rw [hcf_res .reject hres]
    exact ⟨Frame.trans_same (Frame.mono hf1 (by intro x hx; exact List.mem_cons_of_mem R hx))
      (Frame.mono hf2 (by intro x hx; simp only [List.mem_singleton] at hx; subst hx; simp)), hpc2, hR2, hl2, by rw [hf2.mem, hf1.mem, hm]⟩
  have finish_ovf : ∀ s1, Frame PassScratch s s1 → s1.pc = B + 2688 → checkFlatRes bs = some .overflow →
      Reaches s1 fun s' => Frame (R :: PassScratch) s s' ∧
        s'.pc = nxt ∧ s'.reg R = some (checkFlatWord bs) ∧ s'.regs.length = 31 ∧ s'.mem = m := by
    intro s1 hf1 hpc1 hres
    refine (verdict_spec pb.cOvf fOvfB (by omega) hnxt (by omega) (by decide) hR (by rw [hf1.len]; exact hl) hpc1).mono
      fun s2 ⟨hf2, hpc2, hR2, hl2⟩ => ?_
    rw [hcf_res .overflow hres]
    exact ⟨Frame.trans_same (Frame.mono hf1 (by intro x hx; exact List.mem_cons_of_mem R hx))
      (Frame.mono hf2 (by intro x hx; simp only [List.mem_singleton] at hx; subst hx; simp)), hpc2, hR2, hl2, by rw [hf2.mem, hf1.mem, hm]⟩
  have finish_acc : ∀ s1, Frame PassScratch s s1 → s1.pc = B + 2668 → checkFlatRes bs = some .ok →
      Reaches s1 fun s' => Frame (R :: PassScratch) s s' ∧
        s'.pc = nxt ∧ s'.reg R = some (checkFlatWord bs) ∧ s'.regs.length = 31 ∧ s'.mem = m := by
    intro s1 hf1 hpc1 hres
    refine (verdict_spec pb.cAcc fAccB (by omega) hnxt (by omega) (by decide) hR (by rw [hf1.len]; exact hl) hpc1).mono
      fun s2 ⟨hf2, hpc2, hR2, hl2⟩ => ?_
    rw [hcf_res .ok hres]
    exact ⟨Frame.trans_same (Frame.mono hf1 (by intro x hx; exact List.mem_cons_of_mem R hx))
      (Frame.mono hf2 (by intro x hx; simp only [List.mem_singleton] at hx; subst hx; simp)), hpc2, hR2, hl2, by rw [hf2.mem, hf1.mem, hm]⟩
  refine (parse_spec pb hB hD hadr hD0 hD8 hbuf hby hsm hl hpc hm).trans fun s1 ⟨hf1, hp⟩ => ?_
  cases hpe : parse bs with
  | none =>
    rw [hpe] at hp
    exact finish_rej s1 hf1 hp (by simp [checkFlatRes, hall, hltW, hpe])
  | some tup =>
    rcases tup with ⟨main, nw, ws⟩
    rw [hpe] at hp; dsimp only at hp
    by_cases hmg : nw ≤ main
    · rw [if_pos hmg] at hp
      exact finish_rej s1 hf1 hp (by simp [checkFlatRes, hall, hltW, hpe, hmg])
    · rw [if_neg hmg] at hp
      cases hwm : wordAt bs ws main with
      | none =>
        rw [hwm] at hp
        exact finish_rej s1 hf1 hp (by simp [checkFlatRes, hall, hltW, hpe, hmg, hwm])
      | some mainWm =>
        rw [hwm] at hp; dsimp only at hp
        by_cases hnz : mainWm.ins ≠ 0
        · rw [if_pos hnz] at hp
          exact finish_rej s1 hf1 hp (by simp [checkFlatRes, hall, hltW, hpe, hmg, hwm, bne_iff_ne, hnz])
        · rw [if_neg hnz] at hp
          obtain ⟨hpc1, hl1, hm1, _, h1, h12, h24, h25, hws_le, hnw, hsw⟩ := hp
          refine (checkWords_spec pb hB hbuf hby hsm hsw hws_le hnw nw ws s1 hsw hws_le hm1 hl1 hpc1
            h1 h12 h24 h25).trans fun s2 ⟨hf2, hcw⟩ => ?_
          have hf12 := Frame.trans_same hf1 hf2
          cases hcwe : checkWords bs nw ws nw ws with
          | ok =>
            rw [hcwe] at hcw
            exact finish_acc s2 hf12 hcw.1 (by simp [checkFlatRes, hall, hltW, hpe, hmg, hwm, bne_iff_ne, hnz, hcwe])
          | reject =>
            rw [hcwe] at hcw
            exact finish_rej s2 hf12 hcw (by simp [checkFlatRes, hall, hltW, hpe, hmg, hwm, bne_iff_ne, hnz, hcwe])
          | overflow =>
            rw [hcwe] at hcw
            exact finish_ovf s2 hf12 hcw (by simp [checkFlatRes, hall, hltW, hpe, hmg, hwm, bne_iff_ne, hnz, hcwe])

theorem p1_spec {b : Nat} {bs : List Nat} {m : Mem} {s : St}
    (hD0 : rdMem m 0x600000 .x8 = some b) (hD8 : rdMem m (0x600000 + 8) .x8 = some bs.length)
    (hbuf : Buf m b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < HMAX)
    (hl : s.regs.length = 31) (hpc : s.pc = L_p1) (hm : s.mem = m) :
    Reaches s fun s' => Frame (19 :: PassScratch) s s' ∧
      s'.pc = L_p2 ∧ s'.reg 19 = some (checkFlatWord bs) ∧ s'.regs.length = 31 ∧ s'.mem = m :=
  pass_spec p1_codeAt (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hD0 hD8 hbuf hby hsm hl hpc hm

theorem p2_spec {b : Nat} {bs : List Nat} {m : Mem} {s : St}
    (hD0 : rdMem m 0x601000 .x8 = some b) (hD8 : rdMem m (0x601000 + 8) .x8 = some bs.length)
    (hbuf : Buf m b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < HMAX)
    (hl : s.regs.length = 31) (hpc : s.pc = L_p2) (hm : s.mem = m) :
    Reaches s fun s' => Frame (20 :: PassScratch) s s' ∧
      s'.pc = L_p3 ∧ s'.reg 20 = some (checkFlatWord bs) ∧ s'.regs.length = 31 ∧ s'.mem = m :=
  pass_spec p2_codeAt (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hD0 hD8 hbuf hby hsm hl hpc hm

theorem p3_spec {b : Nat} {bs : List Nat} {m : Mem} {s : St}
    (hD0 : rdMem m 0x602000 .x8 = some b) (hD8 : rdMem m (0x602000 + 8) .x8 = some bs.length)
    (hbuf : Buf m b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < HMAX)
    (hl : s.regs.length = 31) (hpc : s.pc = L_p3) (hm : s.mem = m) :
    Reaches s fun s' => Frame (21 :: PassScratch) s s' ∧
      s'.pc = L_vote ∧ s'.reg 21 = some (checkFlatWord bs) ∧ s'.regs.length = 31 ∧ s'.mem = m :=
  pass_spec p3_codeAt (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hD0 hD8 hbuf hby hsm hl hpc hm

end SporosCheckArm64Bin
