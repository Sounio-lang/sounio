/-
SporosCheckBinPass: universal correctness proof for a single `checkFlat` pass (`passCode B D R`)
and instantiation at `p1`, `p2`, `p3`.
-/
import Bootstrap.SporosCheckBinCode

namespace SporosCheckBin

open X86Env X86Int

syntax "rsimpg" (" using " Lean.Parser.Tactic.simpArg,*)? : tactic
macro_rules
  | `(tactic| rsimpg) =>
    `(tactic| simp (config := { decide := true }) only [reg_set_eq, reg_set_ne, set_regs, set_regs_length,
        regs_mk, rip_mk, fl_mk, mem_mk, k_mk, reg_mk_self, reg_mk_set_eq, reg_mk_set_ne, List.length_set,
        set_rip, set_mem, set_k, set_fl, Option.some.injEq, W_eq, Nat.reducePow, Nat.reduceMod])
  | `(tactic| rsimpg using $hs,*) =>
    `(tactic| simp (config := { decide := true }) only [reg_set_eq, reg_set_ne, set_regs, set_regs_length,
        regs_mk, rip_mk, fl_mk, mem_mk, k_mk, reg_mk_self, reg_mk_set_eq, reg_mk_set_ne, List.length_set,
        set_rip, set_mem, set_k, set_fl, Option.some.injEq, W_eq, Nat.reducePow, Nat.reduceMod, $hs,*])

theorem instrLenA_range {op l : Nat} (h : instrLenA op = some l) :
    op < 57 ∨ (0x80 ≤ op ∧ op ≤ 0x8c) := by
  unfold instrLenA at h
  split at h; · exact Or.inl ‹_›
  · split at h; · simp at h
    · split at h; · omega
      · split at h; · simp at h
        · omega

/-! ### Generic helpers for `skipCodeA`, `Desc`, `xor`, and control-flow snippets -/

theorem skipCodeA_add {bs : List Nat} : ∀ (m n pos : Nat),
    skipCodeA bs (m + n) pos = (skipCodeA bs m pos).bind (skipCodeA bs n)
  | 0, n, pos => by simp [skipCodeA]
  | m + 1, n, pos => by
    rw [Nat.add_right_comm]
    simp only [skipCodeA]
    split
    · cases instrLenA bs[pos] with
      | none => rfl
      | some l =>
        simp only
        split
        · rfl
        · exact skipCodeA_add m n _
    · rfl

theorem alu_xor_self (x : Nat) : (aluF .xor x x).1 = 0 := by simp [aluF, Nat.xor_self]

theorem checkFlatA_header_ok {bs : List Nat} (hby : ∀ x ∈ bs, x < 256) (hsm : bs.length < HMAX) :
    (bs.all (· < 256) && bs.length < HMAX) = true := by
  have h1 : bs.all (· < 256) = true := List.all_eq_true.2 fun x hx => decide_eq_true (hby x hx)
  simp [h1, hsm]

structure Desc (m : Mem) (D b len : Nat) : Prop where
  ptr : rd64 m (ea D 0) = some b
  len : rd64 m (ea D 8) = some len

theorem cmpI15_jcc4 {a tgt k x : Nat} {s : St}
    (f1 : im a = some (.cmpI 15 k, 7)) (f2 : im (a + 7) = some (.jcc 4 tgt, 6))
    (hrip : s.rip = a) (h15 : s.reg 15 = some x) (hx : x < 256) (hk : k < 2 ^ 64) :
    Reaches s (fun s' => Frame [] s s' ∧ s'.rip = (if x = k then tgt else a + 13)) := by
  have hmod : k % 18446744073709551616 = k := Nat.mod_eq_of_lt hk
  xs f1 using h15, hmod
  by_cases h : x = k
  · have hc4 : X86Int.cond (aluF .sub k k).snd 4 = some true := by
      simpa using (cond_sub (show k < W by rw [W_eq]; exact hk) (show k < W by rw [W_eq]; exact hk)).2.2.1
    xs f2 using h, hc4
    exact Reaches.refl ⟨by frame_tac, rfl⟩
  · have hc4 := (cond_sub (show x < W by rw [W_eq]; omega) (show k < W by rw [W_eq]; exact hk)).2.2.1
    xs f2 using hc4, h
    exact Reaches.refl ⟨by frame_tac, by rsimp; omega⟩

/-- Checking `(u32(q+4), u32(q+8))` against `(r2, r7)` after `lookupT`. -/
theorem ann_check_r2_r7 {L ok rej : Nat}
    (h1 : CodeAt [(L, .cmp 10 11, 3), (L + 3, .jcc 4 rej, 6), (L + 9, .ld32 14 10 4, 4),
      (L + 13, .cmp 14 2, 3), (L + 16, .jcc 4 (L + 23), 2), (L + 18, .jmp rej, 5),
      (L + 23, .ld32 14 10 8, 4), (L + 27, .cmp 14 7, 3), (L + 30, .jcc 4 ok, 2),
      (L + 32, .jmp rej, 5)])
    {b idx na annOff annEnd d a : Nat} {bs : List Nat} {s : St}
    (hbuf : Buf s.mem b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < 2 ^ 48)
    (hl : s.regs.length = 16) (hrip : s.rip = L)
    (h11 : s.reg 11 = some (b + annEnd)) (h2 : s.reg 2 = some d) (h7 : s.reg 7 = some a)
    (hd : d < 2 ^ 64) (ha : a < 2 ^ 64)
    (hres : match lookupA bs idx na annOff with
      | some q' => s.reg 10 = some (b + q') ∧ q' + 12 ≤ annEnd
      | none => s.reg 10 = some (b + annEnd))
    (hann : annEnd ≤ bs.length) :
    Reaches s (fun s' => Frame [14] s s' ∧
      match lookupA bs idx na annOff with
      | none => s'.rip = rej
      | some q' => if u32 bs (q' + 4) = d ∧ u32 bs (q' + 8) = a then s'.rip = ok else s'.rip = rej) := by
  simp only [CodeAt_cons, CodeAt_nil, and_true] at h1
  obtain ⟨f0, f3, f9, f13, f16, f18, f23, f27, f30, f32⟩ := h1
  cases hlu : lookupA bs idx na annOff with
  | none =>
    rw [hlu] at hres
    xs f0 using hres, h11
    have hc4 := (cond_sub (show b + annEnd < W by rw [W_eq]; omega)
      (show b + annEnd < W by rw [W_eq]; omega)).2.2.1
    xs f3 using hc4
    exact Reaches.refl ⟨by frame_tac, rfl⟩
  | some q' =>
    rw [hlu] at hres
    obtain ⟨h10, hq12⟩ := hres
    xs f0 using h10, h11
    have hc4 := (cond_sub (show b + q' < W by rw [W_eq]; omega)
      (show b + annEnd < W by rw [W_eq]; omega)).2.2.1
    xs f3 using hc4, show b + q' ≠ b + annEnd by omega
    have hq4 : q' + 4 + 4 ≤ bs.length := by omega
    have hq8 : q' + 8 + 4 ≤ bs.length := by omega
    have hu4 := u32_lt hby (q' + 4)
    have hu8 := u32_lt hby (q' + 8)
    xs f9 using hl, h10, ea_small (show b + q' + 4 < 2 ^ 64 by omega),
      show b + q' + 4 = b + (q' + 4) by omega, rd32_buf hbuf hq4
    xs f13 using hl, h2
    by_cases h4eq : u32 bs (q' + 4) = d
    · have hc4a : X86Int.cond (aluF .sub d d).snd 4 = some true := by
        simpa using (cond_sub (show d < W by rw [W_eq]; exact hd) (show d < W by rw [W_eq]; exact hd)).2.2.1
      xs f16 using h4eq, hc4a
      xs f23 using hl, h10, ea_small (show b + q' + 8 < 2 ^ 64 by omega),
        show b + q' + 8 = b + (q' + 8) by omega, rd32_buf hbuf hq8
      xs f27 using hl, h7
      by_cases h8eq : u32 bs (q' + 8) = a
      · have hc4b : X86Int.cond (aluF .sub a a).snd 4 = some true := by
          simpa using (cond_sub (show a < W by rw [W_eq]; exact ha) (show a < W by rw [W_eq]; exact ha)).2.2.1
        xs f30 using h8eq, hc4b
        exact Reaches.refl ⟨by frame_tac, rfl⟩
      · have hc4b := (cond_sub (show u32 bs (q' + 8) < W by rw [W_eq]; omega)
          (show a < W by rw [W_eq]; exact ha)).2.2.1
        xs f30 using hc4b, h8eq
        xs f32
        exact Reaches.refl ⟨by frame_tac, rfl⟩
    · have hc4a := (cond_sub (show u32 bs (q' + 4) < W by rw [W_eq]; omega)
        (show d < W by rw [W_eq]; exact hd)).2.2.1
      xs f16 using hc4a, h4eq
      xs f18
      exact Reaches.refl ⟨by frame_tac, rfl⟩

/-! ### Unpacking `passCode` into logical blocks -/

structure PassBlocks (B D R : Nat) : Prop where
  cHdr : CodeAt [(B + 0, .movI 15 D, 6), (B + 6, .ld 7 15 0, 3), (B + 9, .ld 2 15 8, 4),
    (B + 13, .mov 5 7, 3), (B + 16, .alu .add 5 2, 3), (B + 19, .cmpI 2 16, 4),
    (B + 23, .jcc 2 (B + 3301), 6), (B + 29, .ld32 2 7 0, 2), (B + 31, .cmpI 2 827478099, 7),
    (B + 38, .jcc 4 (B + 45), 2), (B + 40, .jmp (B + 3301), 5), (B + 45, .ld32 12 7 4, 4),
    (B + 49, .ld32 6 7 8, 3), (B + 52, .ld32 15 7 12, 4), (B + 56, .lea 10 7 16, 4)]
  cSd : CodeAt (skipdataT (B + 60) (B + 3301))
  cMid1 : CodeAt [(B + 111, .mov 13 10, 3), (B + 114, .mov 15 6, 3)]
  cSw1 : CodeAt (skipwordsT (B + 117) (B + 3301))
  cMid2 : CodeAt [(B + 318, .cmp 10 5, 3), (B + 321, .jcc 4 (B + 328), 2), (B + 323, .jmp (B + 3301), 5),
    (B + 328, .cmp 12 6, 3), (B + 331, .jcc 3 (B + 3301), 6), (B + 337, .mov 10 13, 3), (B + 340, .mov 15 12, 3)]
  cSw2 : CodeAt (skipwordsT (B + 343) (B + 3301))
  cMid3 : CodeAt [(B + 544, .ld32 2 10 0, 3), (B + 547, .test 2 2, 3), (B + 550, .jcc 4 (B + 557), 2),
    (B + 552, .jmp (B + 3301), 5), (B + 557, .mov 12 13, 3), (B + 560, .cmp 12 5, 3),
    (B + 563, .jcc 4 (B + 3271), 6), (B + 569, .ld32 11 12 12, 5), (B + 574, .test 11 11, 3),
    (B + 577, .jcc 4 (B + 3301), 6), (B + 583, .lea 9 12 20, 5)]
  cSc : CodeAt (skipcodeT (B + 588) 9 (B + 3301))
  cXor : CodeAt [(B + 700, .alu .xor 8 8, 3)]
  cLu0 : CodeAt (lookupT (B + 703) 8)
  cWalk0 : CodeAt [(B + 747, .cmp 10 11, 3), (B + 750, .jcc 4 (B + 784), 2),
    (B + 752, .ld32 14 10 4, 4), (B + 756, .ld32 15 12 0, 4), (B + 760, .cmp 14 15, 3),
    (B + 763, .jcc 4 (B + 770), 2), (B + 765, .jmp (B + 3301), 5), (B + 770, .ld32 14 10 8, 4),
    (B + 774, .test 14 14, 3), (B + 777, .jcc 4 (B + 784), 2), (B + 779, .jmp (B + 3301), 5),
    (B + 784, .ld32 2 12 0, 4), (B + 788, .alu .xor 7 7, 2), (B + 790, .lea 6 12 20, 5),
    (B + 795, .ld32 15 12 12, 5), (B + 800, .cmp 8 15, 3), (B + 803, .jcc 4 (B + 3120), 6)]
  cLuI : CodeAt (lookupT (B + 809) 8)
  cDisp : CodeAt [(B + 853, .cmp 10 11, 3), (B + 856, .jcc 4 (B + 866), 2),
    (B + 858, .ld32 2 10 4, 4), (B + 862, .ld32 7 10 8, 4), (B + 866, .ld8 15 6 0, 4),
    (B + 870, .cmpI 15 57, 4), (B + 874, .jcc 2 (B + 1054), 6),
    (B + 880, .cmpI 15 128, 7), (B + 887, .jcc 4 (B + 2089), 6),
    (B + 893, .cmpI 15 129, 7), (B + 900, .jcc 4 (B + 2121), 6),
    (B + 906, .cmpI 15 130, 7), (B + 913, .jcc 4 (B + 2153), 6),
    (B + 919, .cmpI 15 131, 7), (B + 926, .jcc 4 (B + 2198), 6),
    (B + 932, .cmpI 15 132, 7), (B + 939, .jcc 4 (B + 2243), 6),
    (B + 945, .cmpI 15 133, 7), (B + 952, .jcc 4 (B + 2288), 6),
    (B + 958, .cmpI 15 134, 7), (B + 965, .jcc 4 (B + 2329), 6),
    (B + 971, .cmpI 15 135, 7), (B + 978, .jcc 4 (B + 2379), 6),
    (B + 984, .cmpI 15 136, 7), (B + 991, .jcc 4 (B + 2421), 6),
    (B + 997, .cmpI 15 137, 7), (B + 1004, .jcc 4 (B + 2517), 6),
    (B + 1010, .cmpI 15 138, 7), (B + 1017, .jcc 4 (B + 2517), 6),
    (B + 1023, .cmpI 15 139, 7), (B + 1030, .jcc 4 (B + 2626), 6),
    (B + 1036, .cmpI 15 140, 7), (B + 1043, .jcc 4 (B + 2906), 6),
    (B + 1049, .jmp (B + 3301), 5), (B + 1054, .cmpI 15 50, 4), (B + 1058, .jcc 4 (B + 2069), 6)]
  cEff : CodeAt (effT (B + 1064) (B + 2026))
  cEffT : CodeAt [(B + 2026, .cmp 2 10, 3), (B + 2029, .jcc 2 (B + 3301), 6),
    (B + 2035, .alu .sub 2 10, 3), (B + 2038, .alu .add 2 11, 3)]
  cChkEff : CodeAt (chkT (B + 2041) 2 (B + 3286))
  cLit : CodeAt [(B + 2058, .movI 15 1, 6), (B + 2064, .jmp (B + 2942), 5),
    (B + 2069, .test 2 2, 3), (B + 2072, .jcc 4 (B + 3301), 6),
    (B + 2078, .movI 15 1, 6), (B + 2084, .jmp (B + 3045), 5), (B + 2089, .aluI .add 2 1, 4)]
  cChk80 : CodeAt (chkT (B + 2093) 2 (B + 3286))
  cStr : CodeAt [(B + 2110, .movI 15 9, 6), (B + 2116, .jmp (B + 2942), 5), (B + 2121, .aluI .add 2 2, 4)]
  cChk81 : CodeAt (chkT (B + 2125) 2 (B + 3286))
  cPick : CodeAt [(B + 2142, .movI 15 5, 6), (B + 2148, .jmp (B + 2942), 5),
    (B + 2153, .ld32 10 6 1, 4), (B + 2157, .cmp 10 2, 3), (B + 2160, .jcc 3 (B + 3301), 6),
    (B + 2166, .aluI .add 2 1, 4)]
  cChk82 : CodeAt (chkT (B + 2170) 2 (B + 3286))
  cTor : CodeAt [(B + 2187, .movI 15 5, 6), (B + 2193, .jmp (B + 2942), 5),
    (B + 2198, .test 2 2, 3), (B + 2201, .jcc 4 (B + 3301), 6),
    (B + 2207, .aluI .sub 2 1, 4), (B + 2211, .aluI .add 7 1, 4)]
  cChk83 : CodeAt (chkT (B + 2215) 7 (B + 3286))
  cFromr : CodeAt [(B + 2232, .movI 15 1, 6), (B + 2238, .jmp (B + 2942), 5),
    (B + 2243, .test 7 7, 3), (B + 2246, .jcc 4 (B + 3301), 6),
    (B + 2252, .aluI .sub 7 1, 4), (B + 2256, .aluI .add 2 1, 4)]
  cChk84 : CodeAt (chkT (B + 2260) 2 (B + 3286))
  cRfetch : CodeAt [(B + 2277, .movI 15 1, 6), (B + 2283, .jmp (B + 2942), 5),
    (B + 2288, .test 7 7, 3), (B + 2291, .jcc 4 (B + 3301), 6), (B + 2297, .aluI .add 2 1, 4)]
  cChk85 : CodeAt (chkT (B + 2301) 2 (B + 3286))
  cLocget : CodeAt [(B + 2318, .movI 15 1, 6), (B + 2324, .jmp (B + 2942), 5),
    (B + 2329, .ld32 10 6 1, 4), (B + 2333, .ld32 11 12 8, 5),
    (B + 2338, .cmp 10 11, 3), (B + 2341, .jcc 3 (B + 3301), 6), (B + 2347, .aluI .add 2 1, 4)]
  cChk86 : CodeAt (chkT (B + 2351) 2 (B + 3286))
  cLocset : CodeAt [(B + 2368, .movI 15 5, 6), (B + 2374, .jmp (B + 2942), 5),
    (B + 2379, .test 2 2, 3), (B + 2382, .jcc 4 (B + 3301), 6),
    (B + 2388, .ld32 10 6 1, 4), (B + 2392, .ld32 11 12 8, 5),
    (B + 2397, .cmp 10 11, 3), (B + 2400, .jcc 3 (B + 3301), 6),
    (B + 2406, .aluI .sub 2 1, 4), (B + 2410, .movI 15 5, 6),
    (B + 2416, .jmp (B + 2942), 5), (B + 2421, .ld32 15 6 1, 4)]
  cLuJmp : CodeAt (lookupT (B + 2425) 15)
  cJmp : CodeAt [(B + 2469, .cmp 10 11, 3), (B + 2472, .jcc 4 (B + 3301), 6),
    (B + 2478, .ld32 14 10 4, 4), (B + 2482, .cmp 14 2, 3), (B + 2485, .jcc 4 (B + 2492), 2),
    (B + 2487, .jmp (B + 3301), 5), (B + 2492, .ld32 14 10 8, 4), (B + 2496, .cmp 14 7, 3),
    (B + 2499, .jcc 4 (B + 2506), 2), (B + 2501, .jmp (B + 3301), 5),
    (B + 2506, .movI 15 5, 6), (B + 2512, .jmp (B + 3045), 5),
    (B + 2517, .test 2 2, 3), (B + 2520, .jcc 4 (B + 3301), 6),
    (B + 2526, .aluI .sub 2 1, 4), (B + 2530, .ld32 15 6 1, 4)]
  cLuJz : CodeAt (lookupT (B + 2534) 15)
  cCall0 : CodeAt [(B + 2578, .cmp 10 11, 3), (B + 2581, .jcc 4 (B + 3301), 6),
    (B + 2587, .ld32 14 10 4, 4), (B + 2591, .cmp 14 2, 3), (B + 2594, .jcc 4 (B + 2601), 2),
    (B + 2596, .jmp (B + 3301), 5), (B + 2601, .ld32 14 10 8, 4), (B + 2605, .cmp 14 7, 3),
    (B + 2608, .jcc 4 (B + 2615), 2), (B + 2610, .jmp (B + 3301), 5),
    (B + 2615, .movI 15 5, 6), (B + 2621, .jmp (B + 2942), 5),
    (B + 2626, .ld32 15 6 1, 4), (B + 2630, .movI 14 D, 6), (B + 2636, .ld 14 14 0, 3),
    (B + 2639, .ld32 14 14 8, 4), (B + 2643, .cmp 15 14, 3), (B + 2646, .jcc 3 (B + 3301), 6),
    (B + 2652, .mov 10 13, 3)]
  cSwCall : CodeAt (skipwordsT (B + 2655) (B + 3301))
  cCall1 : CodeAt [(B + 2856, .ld32 14 10 0, 3), (B + 2859, .cmp 2 14, 3), (B + 2862, .jcc 2 (B + 3301), 6),
    (B + 2868, .alu .sub 2 14, 3), (B + 2871, .ld32 14 10 4, 4), (B + 2875, .alu .add 2 14, 3)]
  cChkCall : CodeAt (chkT (B + 2878) 2 (B + 3286))
  cFall : CodeAt [(B + 2895, .movI 15 5, 6), (B + 2901, .jmp (B + 2942), 5),
    (B + 2906, .ld32 14 12 4, 5), (B + 2911, .cmp 2 14, 3), (B + 2914, .jcc 4 (B + 2921), 2),
    (B + 2916, .jmp (B + 3301), 5), (B + 2921, .test 7 7, 3), (B + 2924, .jcc 4 (B + 2931), 2),
    (B + 2926, .jmp (B + 3301), 5), (B + 2931, .movI 15 1, 6), (B + 2937, .jmp (B + 3045), 5),
    (B + 2942, .aluI .add 8 1, 4), (B + 2946, .ld32 14 12 12, 5), (B + 2951, .cmp 8 14, 3),
    (B + 2954, .jcc 3 (B + 3301), 6)]
  cLuFall : CodeAt (lookupT (B + 2960) 8)
  cNofall : CodeAt [(B + 3004, .cmp 10 11, 3), (B + 3007, .jcc 4 (B + 3037), 2),
    (B + 3009, .ld32 14 10 4, 4), (B + 3013, .cmp 14 2, 3), (B + 3016, .jcc 4 (B + 3023), 2),
    (B + 3018, .jmp (B + 3301), 5), (B + 3023, .ld32 14 10 8, 4), (B + 3027, .cmp 14 7, 3),
    (B + 3030, .jcc 4 (B + 3037), 2), (B + 3032, .jmp (B + 3301), 5),
    (B + 3037, .alu .add 6 15, 3), (B + 3040, .jmp (B + 795), 5), (B + 3045, .aluI .add 8 1, 4)]
  cLuNof : CodeAt (lookupT (B + 3049) 8)
  cAnnsPre : CodeAt [(B + 3093, .cmp 10 11, 3), (B + 3096, .jcc 4 (B + 3111), 2),
    (B + 3098, .ld32 2 10 4, 4), (B + 3102, .ld32 7 10 8, 4), (B + 3106, .jmp (B + 3037), 5),
    (B + 3111, .alu .xor 2 2, 2), (B + 3113, .alu .xor 7 7, 2), (B + 3115, .jmp (B + 3037), 5),
    (B + 3120, .ld32 14 12 16, 5), (B + 3125, .mov 7 14, 3), (B + 3128, .shlI 7 3, 4),
    (B + 3132, .shlI 14 2, 4), (B + 3136, .alu .add 7 14, 3), (B + 3139, .alu .add 7 9, 3),
    (B + 3142, .mov 6 9, 3), (B + 3145, .cmp 6 7, 3), (B + 3148, .jcc 4 (B + 3263), 2),
    (B + 3150, .ld32 15 6 0, 3), (B + 3153, .ld32 14 12 12, 5), (B + 3158, .cmp 15 14, 3),
    (B + 3161, .jcc 3 (B + 3301), 6)]
  cLuAnns : CodeAt (lookupT (B + 3167) 15)
  cEnd : CodeAt [(B + 3211, .cmp 10 11, 3), (B + 3214, .jcc 4 (B + 3301), 6),
    (B + 3220, .ld32 14 10 4, 4), (B + 3224, .ld32 2 6 4, 3), (B + 3227, .cmp 14 2, 3),
    (B + 3230, .jcc 4 (B + 3237), 2), (B + 3232, .jmp (B + 3301), 5),
    (B + 3237, .ld32 14 10 8, 4), (B + 3241, .ld32 2 6 8, 3), (B + 3244, .cmp 14 2, 3),
    (B + 3247, .jcc 4 (B + 3254), 2), (B + 3249, .jmp (B + 3301), 5),
    (B + 3254, .aluI .add 6 12, 4), (B + 3258, .jmp (B + 3145), 5),
    (B + 3263, .mov 12 7, 3), (B + 3266, .jmp (B + 560), 5),
    (B + 3271, .movI R ACCEPT, 10), (B + 3281, .jmp (B + 3316), 5),
    (B + 3286, .movI R OVERFLOW, 10), (B + 3296, .jmp (B + 3316), 5),
    (B + 3301, .movI R REJECT, 10), (B + 3311, .jmp (B + 3316), 5)]

theorem passBlocks_of_codeAt {B D R : Nat} (hc : CodeAt (passCode B D R)) : PassBlocks B D R := by
  unfold passCode at hc
  repeat rw [CodeAt_append] at hc
  rcases hc with ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨
    c1, c2⟩, c3⟩, c4⟩, c5⟩, c6⟩, c7⟩, c8⟩, c9⟩, c10⟩, c11⟩, c12⟩, c13⟩, c14⟩, c15⟩, c16⟩, c17⟩,
    c18⟩, c19⟩, c20⟩, c21⟩, c22⟩, c23⟩, c24⟩, c25⟩, c26⟩, c27⟩, c28⟩, c29⟩, c30⟩, c31⟩, c32⟩,
    c33⟩, c34⟩, c35⟩, c36⟩, c37⟩, c38⟩, c39⟩, c40⟩, c41⟩, c42⟩, c43⟩, c44⟩, c45⟩
  exact ⟨c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19,
    c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37,
    c38, c39, c40, c41, c42, c43, c44, c45⟩

/-! ### `hstep` dispatch and branch specifications -/

def dispatchCode (a : Nat) : List (Nat × Nat) → List (Nat × Ins × Nat)
  | [] => []
  | (k, tgt) :: rest => (a, .cmpI 15 k, 7) :: (a + 7, .jcc 4 tgt, 6) :: dispatchCode (a + 13) rest

def dispatchTgt (op defTgt : Nat) : List (Nat × Nat) → Nat
  | [] => defTgt
  | (k, tgt) :: rest => if op = k then tgt else dispatchTgt op defTgt rest

theorem dispatch_loop {defTgt op : Nat} : ∀ (tbl : List (Nat × Nat)) (a : Nat) (s : St),
    CodeAt (dispatchCode a tbl) → im (a + 13 * tbl.length) = some (.jmp defTgt, 5) →
    (∀ p ∈ tbl, p.1 < 2 ^ 64) → s.rip = a → s.reg 15 = some op → op < 256 →
    Reaches s (fun s' => Frame [] s s' ∧ s'.rip = dispatchTgt op defTgt tbl)
  | [], a, s, _, fj, _, hr, _, _ => by
    simp only [List.length_nil, Nat.mul_zero, Nat.add_zero] at fj
    xs fj
    exact Reaches.refl ⟨by frame_tac, rfl⟩
  | (k, tgt) :: rest, a, s, hc, fj, hks, hr, h15, hop => by
    simp only [dispatchCode, CodeAt_cons] at hc
    obtain ⟨f1, f2, hcr⟩ := hc
    have hk : k < 2 ^ 64 := hks (k, tgt) (by simp)
    refine (cmpI15_jcc4 f1 f2 hr h15 hop hk).trans fun s1 ⟨hf1, hr1⟩ => ?_
    by_cases heq : op = k
    · rw [if_pos heq] at hr1
      exact Reaches.refl ⟨Frame.mono hf1 (by decide), by simp [dispatchTgt, heq, hr1]⟩
    · rw [if_neg heq] at hr1
      have fj' : im ((a + 13) + 13 * rest.length) = some (.jmp defTgt, 5) := by
        rw [Nat.add_assoc, show 13 + 13 * rest.length = 13 * (rest.length + 1) by omega]
        exact fj
      refine (dispatch_loop rest (a + 13) s1 hcr fj' (fun p hp => hks p (by simp [hp]))
        hr1 (by rw [hf1.regs 15 (by decide)]; exact h15) hop).mono fun s2 ⟨hf2, hr2⟩ => ?_
      exact ⟨Frame.trans_same (Frame.mono hf1 (by decide)) hf2, by simp [dispatchTgt, heq, hr2]⟩

theorem dispatchTgt_spec (op B rip : Nat) (h57 : ¬ op < 57) :
    rip = dispatchTgt op (B + 3301) [(128, B + 2089), (129, B + 2121), (130, B + 2153),
      (131, B + 2198), (132, B + 2243), (133, B + 2288), (134, B + 2329), (135, B + 2379),
      (136, B + 2421), (137, B + 2517), (138, B + 2517), (139, B + 2626), (140, B + 2906)] →
    if op < 57 then rip = B + 1054
    else if op = 0x80 then rip = B + 2089
    else if op = 0x81 then rip = B + 2121
    else if op = 0x82 then rip = B + 2153
    else if op = 0x83 then rip = B + 2198
    else if op = 0x84 then rip = B + 2243
    else if op = 0x85 then rip = B + 2288
    else if op = 0x86 then rip = B + 2329
    else if op = 0x87 then rip = B + 2379
    else if op = 0x88 then rip = B + 2421
    else if op = 0x89 ∨ op = 0x8a then rip = B + 2517
    else if op = 0x8b then rip = B + 2626
    else if op = 0x8c then rip = B + 2906
    else rip = B + 3301 := by
  intro h; subst rip
  simp only [dispatchTgt, h57, reduceIte]
  by_cases h128 : op = 128; · simp [h128]
  simp only [h128, reduceIte]
  by_cases h129 : op = 129; · simp [h129]
  simp only [h129, reduceIte]
  by_cases h130 : op = 130; · simp [h130]
  simp only [h130, reduceIte]
  by_cases h131 : op = 131; · simp [h131]
  simp only [h131, reduceIte]
  by_cases h132 : op = 132; · simp [h132]
  simp only [h132, reduceIte]
  by_cases h133 : op = 133; · simp [h133]
  simp only [h133, reduceIte]
  by_cases h134 : op = 134; · simp [h134]
  simp only [h134, reduceIte]
  by_cases h135 : op = 135; · simp [h135]
  simp only [h135, reduceIte]
  by_cases h136 : op = 136; · simp [h136]
  simp only [h136, reduceIte]
  by_cases h137 : op = 137; · simp [h137]
  simp only [h137, reduceIte]
  by_cases h138 : op = 138; · simp [h138]
  simp only [h138, reduceIte]
  by_cases h139 : op = 139; · simp [h139]
  simp only [h139, reduceIte]
  by_cases h140 : op = 140 <;> simp [h140]

theorem hstep_dispatch {B D R : Nat} (pb : PassBlocks B D R)
    {b p op : Nat} {bs : List Nat} {s : St}
    (hp : p < bs.length) (hbuf : Buf s.mem b bs) (hop_val : op = bs[p]) (hop : op < 256)
    (hsm : b + bs.length < 2 ^ 48) (hl : s.regs.length = 16)
    (hrip : s.rip = B + 866) (h6 : s.reg 6 = some (b + p)) :
    Reaches s (fun s' => Frame [15] s s' ∧ s'.reg 15 = some op ∧
      if op < 57 then s'.rip = B + 1054
      else if op = 0x80 then s'.rip = B + 2089
      else if op = 0x81 then s'.rip = B + 2121
      else if op = 0x82 then s'.rip = B + 2153
      else if op = 0x83 then s'.rip = B + 2198
      else if op = 0x84 then s'.rip = B + 2243
      else if op = 0x85 then s'.rip = B + 2288
      else if op = 0x86 then s'.rip = B + 2329
      else if op = 0x87 then s'.rip = B + 2379
      else if op = 0x88 then s'.rip = B + 2421
      else if op = 0x89 ∨ op = 0x8a then s'.rip = B + 2517
      else if op = 0x8b then s'.rip = B + 2626
      else if op = 0x8c then s'.rip = B + 2906
      else s'.rip = B + 3301) := by
  have hc := pb.cDisp
  simp only [CodeAt_cons, CodeAt_nil, and_true] at hc
  obtain ⟨_, _, _, _, d0, d4, d8, d14, d21, d27, d34, d40, d47, d53, d60, d66, d73, d79,
    d86, d92, d99, d105, d112, d118, d125, d131, d138, d144, d151, d157, d164, d170, d177,
    d183, _⟩ := hc
  have hrd : s.mem.rd (b + p) = some op := by rw [hop_val]; exact hbuf p hp
  xs d0 using hl, h6, ea_small (show b + p + 0 < 2 ^ 64 by omega), Nat.add_zero, hrd
  xs d4 using hl
  have hc57 := (cond_sub (show op < W by rw [W_eq]; omega) (show 57 < W by rw [W_eq]; omega)).1
  by_cases h57 : op < 57
  · xs d8 using hc57, h57
    exact Reaches.refl ⟨by frame_tac, (by simp only [reg_mk_set_eq, hl, show 15 < 16 by decide]), rfl⟩
  · xs d8 using hc57, h57
    let tbl : List (Nat × Nat) := [(128, B + 2089), (129, B + 2121), (130, B + 2153),
      (131, B + 2198), (132, B + 2243), (133, B + 2288), (134, B + 2329), (135, B + 2379),
      (136, B + 2421), (137, B + 2517), (138, B + 2517), (139, B + 2626), (140, B + 2906)]
    have hcd : CodeAt (dispatchCode (B + 880) tbl) := by
      simp only [tbl, dispatchCode, CodeAt_cons, CodeAt_nil, and_true]
      exact ⟨d14, d21, d27, d34, d40, d47, d53, d60, d66, d73, d79, d86, d92, d99, d105, d112,
        d118, d125, d131, d138, d144, d151, d157, d164, d170, d177⟩
    have hks : ∀ q ∈ tbl, q.1 < 2 ^ 64 := by
      rintro ⟨k, t⟩ hq
      simp only [tbl, List.mem_cons, List.not_mem_nil, or_false] at hq
      rcases hq with h|h|h|h|h|h|h|h|h|h|h|h|h <;> cases h <;> (dsimp only; decide)
    refine reach_frame (by frame_tac)
      ((dispatch_loop tbl (B + 880) _ hcd d183 hks (by rsimp; omega) (by rsimp using hl) hop).mono
        fun s' ⟨hf', hr'⟩ => ⟨Frame.mono hf' (by decide), (by rw [hf'.regs 15 (by decide)]; rsimp using hl), ?_⟩)
    simpa [h57] using dispatchTgt_spec op B s'.rip h57 hr'


theorem eff_le4 : (List.range 57).all (fun op =>
    decide ((primEffTree.eval op).1 ≤ 4 && (primEffTree.eval op).2 ≤ 4)) = true := by
  decide +kernel

theorem eff_bound {op : Nat} (h : op < 57) :
    (primEffTree.eval op).1 ≤ 4 ∧ (primEffTree.eval op).2 ≤ 4 := by
  have := List.all_eq_true.1 eff_le4 op (List.mem_range.2 h)
  simpa using this

theorem hstep_prim {B D R : Nat} (pb : PassBlocks B D R)
    {op d a : Nat} {s : St}
    (hl : s.regs.length = 16) (hrip : s.rip = B + 1054) (h15 : s.reg 15 = some op)
    (h2 : s.reg 2 = some d) (h7 : s.reg 7 = some a) (hop : op < 57) (hd : d < HMAX) :
    Reaches s (fun s' => Frame [2, 10, 11, 15] s s' ∧
      if op = 50 then
        if d = 0 then s'.rip = B + 3301
        else s'.rip = B + 3045 ∧ s'.reg 2 = some d ∧ s'.reg 7 = some a ∧ s'.reg 15 = some 1
      else
        let (eIn, eOut) := primEffTree.eval op
        if d < eIn then s'.rip = B + 3301
        else
          let d' := d - eIn + eOut
          if HMAX ≤ d' then s'.rip = B + 3286
          else s'.rip = B + 2942 ∧ s'.reg 2 = some d' ∧ s'.reg 7 = some a ∧
            s'.reg 15 = some 1) := by
  have hdisp := pb.cDisp
  simp only [CodeAt_cons, CodeAt_nil, and_true] at hdisp
  obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, p0, p4⟩ := hdisp
  have hefft := pb.cEffT
  simp only [CodeAt_cons, CodeAt_nil, and_true] at hefft
  obtain ⟨t0, t3, t9, t12⟩ := hefft
  have hlit := pb.cLit
  simp only [CodeAt_cons, CodeAt_nil, and_true] at hlit
  obtain ⟨l0, l6, l11, l14, l20, l26, _⟩ := hlit
  xs p0 using h15
  by_cases h50 : op = 50
  · have hc4 : X86Int.cond (aluF .sub 50 50).snd 4 = some true := by
      simpa using (cond_sub (show 50 < W by rw [W_eq]; decide) (show 50 < W by rw [W_eq]; decide)).2.2.1
    xs p4 using h50, hc4
    xs l11 using h2
    by_cases hd0 : d = 0
    · xs l14 using hd0, cond_test_e
      exact Reaches.refl ⟨by frame_tac, rfl⟩
    · xs l14 using hd0, cond_test_e
      xs l20
      xs l26
      exact Reaches.refl ⟨by frame_tac, ⟨rfl, (by rsimpg; exact h2), (by rsimpg; exact h7), (by rsimpg using hl)⟩⟩
  · have hc4 := (cond_sub (show op < W by rw [W_eq]; omega) (show 50 < W by rw [W_eq]; decide)).2.2.1
    xs p4 using hc4, h50
    refine reach_frame (by frame_tac)
      ((eff_spec pb.cEff ?_ ?_ ?_ hop).trans
        fun s1 ⟨hf1, hr1, h10_1, h11_1⟩ => ?_)
    · exact hl
    · rsimpg; omega
    · rsimpg; exact h15
    have heff := eff_bound hop
    have hl1 : s1.regs.length = 16 := by rw [hf1.len, hl]
    have h2_1 : s1.reg 2 = some d := by rw [hf1.regs 2 (by decide)]; exact h2
    have h7_1 : s1.reg 7 = some a := by rw [hf1.regs 7 (by decide)]; exact h7
    have he1 : (primEffTree.eval op).1 ≤ 4 := heff.1
    have he2 : (primEffTree.eval op).2 ≤ 4 := heff.2
    xs t0 using h2_1, h10_1
    have hc2 := (cond_sub (show d < W by rw [W_eq]; unfold HMAX at hd; omega)
      (show (primEffTree.eval op).1 < W by rw [W_eq]; omega)).1
    by_cases hlt : d < (primEffTree.eval op).1
    · xs t3 using hc2, hlt
      exact Reaches.refl ⟨Frame.trans_same (Frame.mono hf1 (by decide)) (by frame_tac), rfl⟩
    · xs t3 using hc2, hlt
      xs t9 using hl1, h2_1, h10_1,
        sub_small (show d < 2 ^ 64 by unfold HMAX at hd; omega) (show (primEffTree.eval op).1 ≤ d by omega)
      xs t12 using hl1, h11_1,
        add_small (show d - (primEffTree.eval op).1 + (primEffTree.eval op).2 < 2 ^ 64 by unfold HMAX at hd; omega)
      refine (reach_frame (by frame_tac)
        ((chk_spec pb.cChkEff (by simp only [List.length_set, hl1]) (by rsimpg; omega) (by rsimpg using hl1)
          (show d - (primEffTree.eval op).1 + (primEffTree.eval op).2 < 2 ^ 64 by unfold HMAX at hd; omega)).trans
          fun s2 ⟨hf2, hm2⟩ => ?_)).mono fun s2 ⟨hf12, hm2'⟩ =>
            ⟨Frame.trans_same (Frame.mono hf1 (by decide)) hf12, hm2'⟩
      have hf13 : Frame [2, 10, 11, 15] _ s2 := Frame.mono hf2 (by decide)
      by_cases hov : HMAX ≤ d - (primEffTree.eval op).1 + (primEffTree.eval op).2
      · rw [if_pos hov] at hm2
        exact Reaches.refl ⟨hf13, by simp [hov, hm2]⟩
      · rw [if_neg hov] at hm2
        xs l0 using hm2
        xs l6
        refine Reaches.refl ⟨Frame.trans_same hf13 (by frame_tac), ?_⟩
        simp only [hov, reduceIte, true_and]
        exact ⟨(by rsimpg; rw [hf2.regs 2 (by decide)]; rsimpg using hl1),
          (by rsimpg; rw [hf2.regs 7 (by decide)]; rsimpg; exact h7_1),
          (by have hl2 : s2.regs.length = 16 := by rw [hf2.len]; simp only [List.length_set, hl1]
              rsimpg using hl2)⟩

theorem hstep_push_rstack {B D R : Nat} (pb : PassBlocks B D R)
    {b p op d a : Nat} {bs : List Nat} {s : St}
    (hbuf : Buf s.mem b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < 2 ^ 48)
    (hl : s.regs.length = 16) (h6 : s.reg 6 = some (b + p))
    (h2 : s.reg 2 = some d) (h7 : s.reg 7 = some a) (hd : d < HMAX) (ha : a < HMAX)
    (hpre : op = 0x80 ∨ op = 0x81 ∨ (op = 0x82 ∧ p + 5 ≤ bs.length) ∨
      op = 0x83 ∨ op = 0x84 ∨ op = 0x85)
    (hrip : s.rip = if op = 0x80 then B + 2089
      else if op = 0x81 then B + 2121
      else if op = 0x82 then B + 2153
      else if op = 0x83 then B + 2198
      else if op = 0x84 then B + 2243
      else B + 2288) :
    Reaches s (fun s' => Frame [2, 7, 10, 15] s s' ∧
      if op = 0x80 then
        if HMAX ≤ d + 1 then s'.rip = B + 3286
        else s'.rip = B + 2942 ∧ s'.reg 2 = some (d + 1) ∧ s'.reg 7 = some a ∧ s'.reg 15 = some 9
      else if op = 0x81 then
        if HMAX ≤ d + 2 then s'.rip = B + 3286
        else s'.rip = B + 2942 ∧ s'.reg 2 = some (d + 2) ∧ s'.reg 7 = some a ∧ s'.reg 15 = some 5
      else if op = 0x82 then
        if u32 bs (p + 1) < d then
          if HMAX ≤ d + 1 then s'.rip = B + 3286
          else s'.rip = B + 2942 ∧ s'.reg 2 = some (d + 1) ∧ s'.reg 7 = some a ∧ s'.reg 15 = some 5
        else s'.rip = B + 3301
      else if op = 0x83 then
        if d = 0 then s'.rip = B + 3301
        else if HMAX ≤ a + 1 then s'.rip = B + 3286
        else s'.rip = B + 2942 ∧ s'.reg 2 = some (d - 1) ∧ s'.reg 7 = some (a + 1) ∧ s'.reg 15 = some 1
      else if op = 0x84 then
        if a = 0 then s'.rip = B + 3301
        else if HMAX ≤ d + 1 then s'.rip = B + 3286
        else s'.rip = B + 2942 ∧ s'.reg 2 = some (d + 1) ∧ s'.reg 7 = some (a - 1) ∧ s'.reg 15 = some 1
      else
        if a = 0 then s'.rip = B + 3301
        else if HMAX ≤ d + 1 then s'.rip = B + 3286
        else s'.rip = B + 2942 ∧ s'.reg 2 = some (d + 1) ∧ s'.reg 7 = some a ∧ s'.reg 15 = some 1) := by
  have hd48 : d < 2 ^ 48 := hd
  have ha48 : a < 2 ^ 48 := ha
  have hd64 : d < 2 ^ 64 := by omega
  have ha64 : a < 2 ^ 64 := by omega
  rcases hpre with rfl | rfl | ⟨rfl, hp5⟩ | rfl | rfl | rfl
  · simp (config := { decide := true }) only [reduceIte] at hrip ⊢
    have hl' := pb.cLit; simp only [CodeAt_cons, CodeAt_nil, and_true] at hl'
    obtain ⟨_, _, _, _, _, _, l80⟩ := hl'
    have hs' := pb.cStr; simp only [CodeAt_cons, CodeAt_nil, and_true] at hs'
    obtain ⟨s0, s6, _⟩ := hs'
    xs l80 using hl, h2, add_small (show d + 1 < 2 ^ 64 by omega)
    refine reach_frame (by frame_tac)
      ((chk_spec pb.cChk80 (by simp only [List.length_set, hl]) (by rsimpg; omega) (by rsimpg using hl)
        (show d + 1 < 2 ^ 64 by omega)).trans fun s1 ⟨hf1, hm1⟩ => ?_)
    have hf01 : Frame [2, 7, 10, 15] _ s1 := Frame.mono hf1 (by decide)
    by_cases hov : HMAX ≤ d + 1
    · rw [if_pos hov] at hm1; exact Reaches.refl ⟨hf01, by simp [hov, hm1]⟩
    · rw [if_neg hov] at hm1
      xs s0 using hm1; xs s6
      refine Reaches.refl ⟨Frame.trans_same hf01 (by frame_tac), ?_⟩
      simp only [hov, reduceIte, true_and]
      have hl1 : s1.regs.length = 16 := by rw [hf1.len]; simp only [List.length_set, hl]
      exact ⟨(by rsimpg; rw [hf1.regs 2 (by decide)]; rsimpg using hl),
        (by rsimpg; rw [hf1.regs 7 (by decide)]; rsimpg; exact h7), (by rsimpg using hl1)⟩
  · simp (config := { decide := true }) only [reduceIte] at hrip ⊢
    have hs' := pb.cStr; simp only [CodeAt_cons, CodeAt_nil, and_true] at hs'
    obtain ⟨_, _, s81⟩ := hs'
    have hpk := pb.cPick; simp only [CodeAt_cons, CodeAt_nil, and_true] at hpk
    obtain ⟨k0, k6, _, _, _, _⟩ := hpk
    xs s81 using hl, h2, add_small (show d + 2 < 2 ^ 64 by omega)
    refine reach_frame (by frame_tac)
      ((chk_spec pb.cChk81 (by simp only [List.length_set, hl]) (by rsimpg; omega) (by rsimpg using hl)
        (show d + 2 < 2 ^ 64 by omega)).trans fun s1 ⟨hf1, hm1⟩ => ?_)
    have hf01 : Frame [2, 7, 10, 15] _ s1 := Frame.mono hf1 (by decide)
    by_cases hov : HMAX ≤ d + 2
    · rw [if_pos hov] at hm1; exact Reaches.refl ⟨hf01, by simp [hov, hm1]⟩
    · rw [if_neg hov] at hm1
      xs k0 using hm1; xs k6
      refine Reaches.refl ⟨Frame.trans_same hf01 (by frame_tac), ?_⟩
      simp only [hov, reduceIte, true_and]
      have hl1 : s1.regs.length = 16 := by rw [hf1.len]; simp only [List.length_set, hl]
      exact ⟨(by rsimpg; rw [hf1.regs 2 (by decide)]; rsimpg using hl),
        (by rsimpg; rw [hf1.regs 7 (by decide)]; rsimpg; exact h7), (by rsimpg using hl1)⟩
  · simp (config := { decide := true }) only [reduceIte] at hrip ⊢
    have hpk := pb.cPick; simp only [CodeAt_cons, CodeAt_nil, and_true] at hpk
    obtain ⟨_, _, k0, k4, k7, k13⟩ := hpk
    have htr := pb.cTor; simp only [CodeAt_cons, CodeAt_nil, and_true] at htr
    obtain ⟨t0, t6, _, _, _, _⟩ := htr
    have hu := u32_lt hby (p + 1)
    xs k0 using hl, h6, ea_small (show b + p + 1 < 2 ^ 64 by omega),
      show b + p + 1 = b + (p + 1) by omega, rd32_buf hbuf (show p + 1 + 4 ≤ bs.length by omega)
    xs k4 using hl, h2
    have hc3 := (cond_sub (show u32 bs (p + 1) < W by rw [W_eq]; omega) (show d < W by rw [W_eq]; exact hd64)).2.1
    by_cases hlt : u32 bs (p + 1) < d
    · xs k7 using hc3, show ¬ (d ≤ u32 bs (p + 1)) by omega
      xs k13 using hl, h2, add_small (show d + 1 < 2 ^ 64 by omega)
      refine reach_frame (by frame_tac)
        ((chk_spec pb.cChk82 (by simp only [List.length_set, hl]) (by rsimpg; omega) (by rsimpg using hl)
          (show d + 1 < 2 ^ 64 by omega)).trans fun s1 ⟨hf1, hm1⟩ => ?_)
      have hf01 : Frame [2, 7, 10, 15] _ s1 := Frame.mono hf1 (by decide)
      by_cases hov : HMAX ≤ d + 1
      · rw [if_pos hov] at hm1; exact Reaches.refl ⟨hf01, by simp [hlt, hov, hm1]⟩
      · rw [if_neg hov] at hm1
        xs t0 using hm1; xs t6
        refine Reaches.refl ⟨Frame.trans_same hf01 (by frame_tac), ?_⟩
        simp only [hlt, hov, reduceIte, true_and]
        have hl1 : s1.regs.length = 16 := by rw [hf1.len]; simp only [List.length_set, hl]
        exact ⟨(by rsimpg; rw [hf1.regs 2 (by decide)]; rsimpg using hl),
          (by rsimpg; rw [hf1.regs 7 (by decide)]; rsimpg; exact h7), (by rsimpg using hl1)⟩
    · xs k7 using hc3, show d ≤ u32 bs (p + 1) by omega
      exact Reaches.refl ⟨by frame_tac, by simp [hlt]⟩
  · simp (config := { decide := true }) only [reduceIte] at hrip ⊢
    have htr := pb.cTor; simp only [CodeAt_cons, CodeAt_nil, and_true] at htr
    obtain ⟨_, _, r0, r3, r9, r13⟩ := htr
    have hfr := pb.cFromr; simp only [CodeAt_cons, CodeAt_nil, and_true] at hfr
    obtain ⟨f0, f6, _, _, _, _⟩ := hfr
    xs r0 using h2
    by_cases hd0 : d = 0
    · xs r3 using hd0, cond_test_e; exact Reaches.refl ⟨by frame_tac, rfl⟩
    · xs r3 using hd0, cond_test_e
      xs r9 using hl, h2, sub_small hd64 (show 1 ≤ d by omega)
      xs r13 using hl, h7, add_small (show a + 1 < 2 ^ 64 by omega)
      refine reach_frame (by frame_tac)
        ((chk_spec pb.cChk83 (by simp only [List.length_set, hl]) (by rsimpg; omega) (by rsimpg using hl)
          (show a + 1 < 2 ^ 64 by omega)).trans fun s1 ⟨hf1, hm1⟩ => ?_)
      have hf01 : Frame [2, 7, 10, 15] _ s1 := Frame.mono hf1 (by decide)
      by_cases hov : HMAX ≤ a + 1
      · rw [if_pos hov] at hm1; exact Reaches.refl ⟨hf01, by simp [hov, hm1]⟩
      · rw [if_neg hov] at hm1
        xs f0 using hm1; xs f6
        refine Reaches.refl ⟨Frame.trans_same hf01 (by frame_tac), ?_⟩
        simp only [hov, reduceIte, true_and]
        have hl1 : s1.regs.length = 16 := by rw [hf1.len]; simp only [List.length_set, hl]
        exact ⟨(by rsimpg; rw [hf1.regs 2 (by decide)]; rsimpg using hl),
          (by rsimpg; rw [hf1.regs 7 (by decide)]; rsimpg using hl), (by rsimpg using hl1)⟩
  · simp (config := { decide := true }) only [reduceIte] at hrip ⊢
    have hfr := pb.cFromr; simp only [CodeAt_cons, CodeAt_nil, and_true] at hfr
    obtain ⟨_, _, f0, f3, f9, f13⟩ := hfr
    have hrf := pb.cRfetch; simp only [CodeAt_cons, CodeAt_nil, and_true] at hrf
    obtain ⟨g0, g6, _, _, _⟩ := hrf
    xs f0 using h7
    by_cases ha0 : a = 0
    · xs f3 using ha0, cond_test_e; exact Reaches.refl ⟨by frame_tac, rfl⟩
    · xs f3 using ha0, cond_test_e
      xs f9 using hl, h7, sub_small ha64 (show 1 ≤ a by omega)
      xs f13 using hl, h2, add_small (show d + 1 < 2 ^ 64 by omega)
      refine reach_frame (by frame_tac)
        ((chk_spec pb.cChk84 (by simp only [List.length_set, hl]) (by rsimpg; omega) (by rsimpg using hl)
          (show d + 1 < 2 ^ 64 by omega)).trans fun s1 ⟨hf1, hm1⟩ => ?_)
      have hf01 : Frame [2, 7, 10, 15] _ s1 := Frame.mono hf1 (by decide)
      by_cases hov : HMAX ≤ d + 1
      · rw [if_pos hov] at hm1; exact Reaches.refl ⟨hf01, by simp [hov, hm1]⟩
      · rw [if_neg hov] at hm1
        xs g0 using hm1; xs g6
        refine Reaches.refl ⟨Frame.trans_same hf01 (by frame_tac), ?_⟩
        simp only [hov, reduceIte, true_and]
        have hl1 : s1.regs.length = 16 := by rw [hf1.len]; simp only [List.length_set, hl]
        exact ⟨(by rsimpg; rw [hf1.regs 2 (by decide)]; rsimpg using hl),
          (by rsimpg; rw [hf1.regs 7 (by decide)]; rsimpg using hl), (by rsimpg using hl1)⟩
  · simp (config := { decide := true }) only [reduceIte] at hrip ⊢
    have hrf := pb.cRfetch; simp only [CodeAt_cons, CodeAt_nil, and_true] at hrf
    obtain ⟨_, _, r0, r3, r9⟩ := hrf
    have hlg := pb.cLocget; simp only [CodeAt_cons, CodeAt_nil, and_true] at hlg
    obtain ⟨g0, g6, _, _, _, _, _⟩ := hlg
    xs r0 using h7
    by_cases ha0 : a = 0
    · xs r3 using ha0, cond_test_e; exact Reaches.refl ⟨by frame_tac, rfl⟩
    · xs r3 using ha0, cond_test_e
      xs r9 using hl, h2, add_small (show d + 1 < 2 ^ 64 by omega)
      refine reach_frame (by frame_tac)
        ((chk_spec pb.cChk85 (by simp only [List.length_set, hl]) (by rsimpg; omega) (by rsimpg using hl)
          (show d + 1 < 2 ^ 64 by omega)).trans fun s1 ⟨hf1, hm1⟩ => ?_)
      have hf01 : Frame [2, 7, 10, 15] _ s1 := Frame.mono hf1 (by decide)
      by_cases hov : HMAX ≤ d + 1
      · rw [if_pos hov] at hm1; exact Reaches.refl ⟨hf01, by simp [hov, hm1]⟩
      · rw [if_neg hov] at hm1
        xs g0 using hm1; xs g6
        refine Reaches.refl ⟨Frame.trans_same hf01 (by frame_tac), ?_⟩
        simp only [hov, reduceIte, true_and]
        have hl1 : s1.regs.length = 16 := by rw [hf1.len]; simp only [List.length_set, hl]
        exact ⟨(by rsimpg; rw [hf1.regs 2 (by decide)]; rsimpg using hl),
          (by rsimpg; rw [hf1.regs 7 (by decide)]; rsimpg; exact h7), (by rsimpg using hl1)⟩

private theorem hstep_86 {B D R : Nat} (pb : PassBlocks B D R)
    {b p wordOff d a : Nat} {bs : List Nat} {s : St}
    (hbuf : Buf s.mem b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < 2 ^ 48)
    (hl : s.regs.length = 16) (h6 : s.reg 6 = some (b + p)) (h12 : s.reg 12 = some (b + wordOff))
    (h2 : s.reg 2 = some d) (h7 : s.reg 7 = some a) (hd : d < HMAX)
    (hw20 : wordOff + 20 ≤ bs.length) (hp5 : p + 5 ≤ bs.length) (hrip : s.rip = B + 2329) :
    Reaches s (fun s' => Frame [2, 7, 10, 11, 14, 15] s s' ∧
      let res :=
        if u32 bs (p + 1) < u32 bs (wordOff + 8) then
          if HMAX ≤ d + 1 then StepResA.overflow else .fall (d + 1) a 5
        else .reject
      match res with
      | .reject => s'.rip = B + 3301
      | .overflow => s'.rip = B + 3286
      | .fall nd na' len => s'.rip = B + 2942 ∧ s'.reg 2 = some nd ∧ s'.reg 7 = some na' ∧ s'.reg 15 = some len
      | .nofall len => s'.rip = B + 3045 ∧ s'.reg 2 = some d ∧ s'.reg 7 = some a ∧ s'.reg 15 = some len) := by
  have hd48 : d < 2 ^ 48 := hd
  have hd64 : d < 2 ^ 64 := by omega
  have hlg := pb.cLocget; simp only [CodeAt_cons, CodeAt_nil, and_true] at hlg
  obtain ⟨_, _, g0, g4, g9, g12, g18⟩ := hlg
  have hls := pb.cLocset; simp only [CodeAt_cons, CodeAt_nil, and_true] at hls
  obtain ⟨l0, l6, _, _, _, _, _, _, _, _, _, _⟩ := hls
  have hi := u32_lt hby (p + 1)
  have hloc := u32_lt hby (wordOff + 8)
  xs g0 using hl, h6, ea_small (show b + p + 1 < 2 ^ 64 by omega),
    show b + p + 1 = b + (p + 1) by omega, rd32_buf hbuf (show p + 1 + 4 ≤ bs.length by omega)
  xs g4 using hl, h12, ea_small (show b + wordOff + 8 < 2 ^ 64 by omega),
    show b + wordOff + 8 = b + (wordOff + 8) by omega, rd32_buf hbuf (show wordOff + 8 + 4 ≤ bs.length by omega)
  xs g9 using hl
  have hc3 := (cond_sub (show u32 bs (p + 1) < W by rw [W_eq]; omega)
    (show u32 bs (wordOff + 8) < W by rw [W_eq]; omega)).2.1
  by_cases hlt : u32 bs (p + 1) < u32 bs (wordOff + 8)
  · xs g12 using hc3, show ¬ (u32 bs (wordOff + 8) ≤ u32 bs (p + 1)) by omega
    xs g18 using hl, h2, add_small (show d + 1 < 2 ^ 64 by unfold HMAX at hd; omega)
    refine reach_frame (by frame_tac)
      ((chk_spec pb.cChk86 (by simp only [List.length_set, hl]) (by rsimpg; omega) (by rsimpg using hl)
        (show d + 1 < 2 ^ 64 by unfold HMAX at hd; omega)).trans fun s1 ⟨hf1, hm1⟩ => ?_)
    have hf01 : Frame [2, 7, 10, 11, 14, 15] _ s1 := Frame.mono hf1 (by decide)
    by_cases hov : HMAX ≤ d + 1
    · rw [if_pos hov] at hm1; exact Reaches.refl ⟨hf01, by simp [hlt, hov, hm1]⟩
    · rw [if_neg hov] at hm1
      xs l0 using hm1; xs l6
      refine Reaches.refl ⟨Frame.trans_same hf01 (by frame_tac), ?_⟩
      simp only [hlt, hov, reduceIte, true_and]
      have hl1 : s1.regs.length = 16 := by rw [hf1.len]; simp only [List.length_set, hl]
      exact ⟨(by rsimpg; rw [hf1.regs 2 (by decide)]; rsimpg using hl),
        (by rsimpg; rw [hf1.regs 7 (by decide)]; rsimpg; exact h7), (by rsimpg using hl1)⟩
  · xs g12 using hc3, show u32 bs (wordOff + 8) ≤ u32 bs (p + 1) by omega
    exact Reaches.refl ⟨by frame_tac, by simp [hlt]⟩

private theorem hstep_87 {B D R : Nat} (pb : PassBlocks B D R)
    {b p wordOff d a : Nat} {bs : List Nat} {s : St}
    (hbuf : Buf s.mem b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < 2 ^ 48)
    (hl : s.regs.length = 16) (h6 : s.reg 6 = some (b + p)) (h12 : s.reg 12 = some (b + wordOff))
    (h2 : s.reg 2 = some d) (h7 : s.reg 7 = some a) (hd : d < HMAX)
    (hw20 : wordOff + 20 ≤ bs.length) (hp5 : p + 5 ≤ bs.length) (hrip : s.rip = B + 2379) :
    Reaches s (fun s' => Frame [2, 7, 10, 11, 14, 15] s s' ∧
      let res :=
        if d = 0 then StepResA.reject
        else if u32 bs (p + 1) < u32 bs (wordOff + 8) then .fall (d - 1) a 5 else .reject
      match res with
      | .reject => s'.rip = B + 3301
      | .overflow => s'.rip = B + 3286
      | .fall nd na' len => s'.rip = B + 2942 ∧ s'.reg 2 = some nd ∧ s'.reg 7 = some na' ∧ s'.reg 15 = some len
      | .nofall len => s'.rip = B + 3045 ∧ s'.reg 2 = some d ∧ s'.reg 7 = some a ∧ s'.reg 15 = some len) := by
  have hd48 : d < 2 ^ 48 := hd
  have hd64 : d < 2 ^ 64 := by omega
  have hls := pb.cLocset; simp only [CodeAt_cons, CodeAt_nil, and_true] at hls
  obtain ⟨_, _, s0, s3, s9, s13, s18, s21, s27, s31, s37, _⟩ := hls
  have hi := u32_lt hby (p + 1)
  have hloc := u32_lt hby (wordOff + 8)
  xs s0 using h2
  by_cases hd0 : d = 0
  · xs s3 using hd0, cond_test_e; exact Reaches.refl ⟨by frame_tac, rfl⟩
  · xs s3 using hd0, cond_test_e
    xs s9 using hl, h6, ea_small (show b + p + 1 < 2 ^ 64 by omega),
      show b + p + 1 = b + (p + 1) by omega, rd32_buf hbuf (show p + 1 + 4 ≤ bs.length by omega)
    xs s13 using hl, h12, ea_small (show b + wordOff + 8 < 2 ^ 64 by omega),
      show b + wordOff + 8 = b + (wordOff + 8) by omega, rd32_buf hbuf (show wordOff + 8 + 4 ≤ bs.length by omega)
    xs s18 using hl
    have hc3 := (cond_sub (show u32 bs (p + 1) < W by rw [W_eq]; omega)
      (show u32 bs (wordOff + 8) < W by rw [W_eq]; omega)).2.1
    by_cases hlt : u32 bs (p + 1) < u32 bs (wordOff + 8)
    · xs s21 using hc3, show ¬ (u32 bs (wordOff + 8) ≤ u32 bs (p + 1)) by omega
      xs s27 using hl, h2, sub_small hd64 (show 1 ≤ d by omega)
      xs s31 using hl
      xs s37
      refine Reaches.refl ⟨by frame_tac, ?_⟩
      simp only [hlt, reduceIte, true_and]
      exact ⟨(by rsimpg using hl), (by rsimpg; exact h7), (by rsimpg using hl)⟩
    · xs s21 using hc3, show u32 bs (wordOff + 8) ≤ u32 bs (p + 1) by omega
      exact Reaches.refl ⟨by frame_tac, by simp [hlt]⟩

private theorem hstep_88 {B D R : Nat} (pb : PassBlocks B D R)
    {b p wordOff annOff d a : Nat} {bs : List Nat} {s : St}
    (hbuf : Buf s.mem b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < 2 ^ 48)
    (hl : s.regs.length = 16) (h6 : s.reg 6 = some (b + p))
    (h9 : s.reg 9 = some (b + annOff)) (h12 : s.reg 12 = some (b + wordOff))
    (h2 : s.reg 2 = some d) (h7 : s.reg 7 = some a) (hd : d < HMAX) (ha : a < HMAX)
    (hw20 : wordOff + 20 ≤ bs.length) (hann : annOff + 12 * u32 bs (wordOff + 16) ≤ bs.length)
    (hp5 : p + 5 ≤ bs.length) (hrip : s.rip = B + 2421) :
    Reaches s (fun s' => Frame [2, 7, 10, 11, 14, 15] s s' ∧
      let res :=
        match lookupA bs (u32 bs (p + 1)) (u32 bs (wordOff + 16)) annOff with
        | none => StepResA.reject
        | some q => if u32 bs (q + 4) = d ∧ u32 bs (q + 8) = a then .nofall 5 else .reject
      match res with
      | .reject => s'.rip = B + 3301
      | .overflow => s'.rip = B + 3286
      | .fall nd na' len => s'.rip = B + 2942 ∧ s'.reg 2 = some nd ∧ s'.reg 7 = some na' ∧ s'.reg 15 = some len
      | .nofall len => s'.rip = B + 3045 ∧ s'.reg 2 = some d ∧ s'.reg 7 = some a ∧ s'.reg 15 = some len) := by
  have hd48 : d < 2 ^ 48 := hd
  have hd64 : d < 2 ^ 64 := by omega
  have ha64 : a < 2 ^ 64 := by unfold HMAX at ha; omega
  have hls := pb.cLocset; simp only [CodeAt_cons, CodeAt_nil, and_true] at hls
  obtain ⟨_, _, _, _, _, _, _, _, _, _, _, j0⟩ := hls
  have hjmp := pb.cJmp; simp only [CodeAt_cons, CodeAt_nil, and_true] at hjmp
  obtain ⟨j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11, j12, _, _, _, _⟩ := hjmp
  have ht := u32_lt hby (p + 1)
  xs j0 using hl, h6, ea_small (show b + p + 1 < 2 ^ 64 by omega),
    show b + p + 1 = b + (p + 1) by omega, rd32_buf hbuf (show p + 1 + 4 ≤ bs.length by omega)
  refine reach_frame (by frame_tac)
    ((lookup_spec pb.cLuJmp (by decide) (by decide) (by decide) (by decide)
      (s := _) (idx := u32 bs (p + 1)) ?_ hby hsm (by simp only [List.length_set, hl]) (by rsimpg; omega)
      (by rsimpg using hl; exact h12) (by rsimpg using hl; exact h9) (by rsimpg using hl)
      (by omega) hw20 hann).trans
      fun s1 ⟨hf1, hr1, h11_1, hm1⟩ => ?_)
  · exact hbuf
  · have hf01 : Frame [2, 7, 10, 11, 14, 15] _ s1 := Frame.mono hf1 (by decide)
    have hcAnn : CodeAt [(B + 2469, .cmp 10 11, 3), (B + 2472, .jcc 4 (B + 3301), 6),
      (B + 2478, .ld32 14 10 4, 4), (B + 2482, .cmp 14 2, 3), (B + 2485, .jcc 4 (B + 2492), 2),
      (B + 2487, .jmp (B + 3301), 5), (B + 2492, .ld32 14 10 8, 4), (B + 2496, .cmp 14 7, 3),
      (B + 2499, .jcc 4 (B + 2506), 2), (B + 2501, .jmp (B + 3301), 5)] := by
      simp only [CodeAt_cons, CodeAt_nil, and_true]
      exact ⟨j1, j2, j3, j4, j5, j6, j7, j8, j9, j10⟩
    refine (ann_check_r2_r7 (idx := u32 bs (p + 1)) (na := u32 bs (wordOff + 16))
      (annOff := annOff) (annEnd := annOff + 12 * u32 bs (wordOff + 16))
      hcAnn (by rw [hf1.mem]; exact hbuf) hby hsm (by rw [hf1.len]; simp only [List.length_set, hl]) hr1
      (by rw [← Nat.add_assoc]; exact h11_1)
      (by rw [hf1.regs 2 (by decide)]; rsimpg; exact h2)
      (by rw [hf1.regs 7 (by decide)]; rsimpg; exact h7)
      hd64 ha64 (by cases hlu : lookupA bs (u32 bs (p + 1)) (u32 bs (wordOff + 16)) annOff with
        | some q' => rw [hlu] at hm1; exact hm1
        | none => rw [hlu] at hm1; rw [← Nat.add_assoc]; exact hm1)
      hann).trans fun s2 ⟨hf2, hm2⟩ => ?_
    have hf02 : Frame [2, 7, 10, 11, 14, 15] _ s2 := Frame.trans_same hf01 (Frame.mono hf2 (by decide))
    cases hlu : lookupA bs (u32 bs (p + 1)) (u32 bs (wordOff + 16)) annOff with
    | none => rw [hlu] at hm2; exact Reaches.refl ⟨hf02, hm2⟩
    | some q =>
      rw [hlu] at hm2; dsimp only at hm2 ⊢
      by_cases heq : u32 bs (q + 4) = d ∧ u32 bs (q + 8) = a
      · rw [if_pos heq] at hm2
        xs j11 using hm2
        xs j12
        refine Reaches.refl ⟨Frame.trans_same hf02 (by frame_tac), ?_⟩
        simp only [heq, reduceIte, true_and]
        have hl2 : s2.regs.length = 16 := by rw [hf2.len, hf1.len]; simp only [List.length_set, hl]
        exact ⟨(by rsimpg; rw [hf2.regs 2 (by decide), hf1.regs 2 (by decide)]; rsimpg; exact h2),
          (by rsimpg; rw [hf2.regs 7 (by decide), hf1.regs 7 (by decide)]; rsimpg; exact h7),
          (by rsimpg using hl2)⟩
      · rw [if_neg heq] at hm2
        exact Reaches.refl ⟨hf02, by simp [heq, hm2]⟩

private theorem hstep_89a {B D R : Nat} (pb : PassBlocks B D R)
    {b p wordOff annOff d a : Nat} {bs : List Nat} {s : St}
    (hbuf : Buf s.mem b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < 2 ^ 48)
    (hl : s.regs.length = 16) (h6 : s.reg 6 = some (b + p))
    (h9 : s.reg 9 = some (b + annOff)) (h12 : s.reg 12 = some (b + wordOff))
    (h2 : s.reg 2 = some d) (h7 : s.reg 7 = some a) (hd : d < HMAX) (ha : a < HMAX)
    (hw20 : wordOff + 20 ≤ bs.length) (hann : annOff + 12 * u32 bs (wordOff + 16) ≤ bs.length)
    (hp5 : p + 5 ≤ bs.length) (hrip : s.rip = B + 2517) :
    Reaches s (fun s' => Frame [2, 7, 10, 11, 14, 15] s s' ∧
      let res :=
        if d = 0 then StepResA.reject
        else
          match lookupA bs (u32 bs (p + 1)) (u32 bs (wordOff + 16)) annOff with
          | none => .reject
          | some q => if u32 bs (q + 4) = d - 1 ∧ u32 bs (q + 8) = a then .fall (d - 1) a 5 else .reject
      match res with
      | .reject => s'.rip = B + 3301
      | .overflow => s'.rip = B + 3286
      | .fall nd na' len => s'.rip = B + 2942 ∧ s'.reg 2 = some nd ∧ s'.reg 7 = some na' ∧ s'.reg 15 = some len
      | .nofall len => s'.rip = B + 3045 ∧ s'.reg 2 = some d ∧ s'.reg 7 = some a ∧ s'.reg 15 = some len) := by
  have hd48 : d < 2 ^ 48 := hd
  have hd64 : d < 2 ^ 64 := by omega
  have ha64 : a < 2 ^ 64 := by unfold HMAX at ha; omega
  have hjmp := pb.cJmp; simp only [CodeAt_cons, CodeAt_nil, and_true] at hjmp
  obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, jz0, jz3, jz9, jz13⟩ := hjmp
  have hc0 := pb.cCall0; simp only [CodeAt_cons, CodeAt_nil, and_true] at hc0
  obtain ⟨k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, _, _, _, _, _, _, _⟩ := hc0
  have ht := u32_lt hby (p + 1)
  xs jz0 using h2
  by_cases hd0 : d = 0
  · xs jz3 using hd0, cond_test_e; exact Reaches.refl ⟨by frame_tac, rfl⟩
  · xs jz3 using hd0, cond_test_e
    xs jz9 using hl, h2, sub_small hd64 (show 1 ≤ d by omega)
    xs jz13 using hl, h6, ea_small (show b + p + 1 < 2 ^ 64 by omega),
      show b + p + 1 = b + (p + 1) by omega, rd32_buf hbuf (show p + 1 + 4 ≤ bs.length by omega)
    refine reach_frame (by frame_tac)
      ((lookup_spec pb.cLuJz (by decide) (by decide) (by decide) (by decide)
        (s := _) (idx := u32 bs (p + 1)) ?_ hby hsm (by simp only [List.length_set, hl]) (by rsimpg; omega)
        (by rsimpg using hl; exact h12) (by rsimpg using hl; exact h9) (by rsimpg using hl)
        (by omega) hw20 hann).trans
        fun s1 ⟨hf1, hr1, h11_1, hm1⟩ => ?_)
    · exact hbuf
    · have hf01 : Frame [2, 7, 10, 11, 14, 15] _ s1 := Frame.mono hf1 (by decide)
      have hcAnn : CodeAt [(B + 2578, .cmp 10 11, 3), (B + 2581, .jcc 4 (B + 3301), 6),
        (B + 2587, .ld32 14 10 4, 4), (B + 2591, .cmp 14 2, 3), (B + 2594, .jcc 4 (B + 2601), 2),
        (B + 2596, .jmp (B + 3301), 5), (B + 2601, .ld32 14 10 8, 4), (B + 2605, .cmp 14 7, 3),
        (B + 2608, .jcc 4 (B + 2615), 2), (B + 2610, .jmp (B + 3301), 5)] := by
        simp only [CodeAt_cons, CodeAt_nil, and_true]
        exact ⟨k1, k2, k3, k4, k5, k6, k7, k8, k9, k10⟩
      have h2_1 : s1.reg 2 = some (d - 1) := by rw [hf1.regs 2 (by decide)]; rsimpg using hl
      have h7_1 : s1.reg 7 = some a := by rw [hf1.regs 7 (by decide)]; rsimpg; exact h7
      refine (ann_check_r2_r7 (idx := u32 bs (p + 1)) (na := u32 bs (wordOff + 16))
        (annOff := annOff) (annEnd := annOff + 12 * u32 bs (wordOff + 16))
        hcAnn (by rw [hf1.mem]; exact hbuf) hby hsm (by rw [hf1.len]; simp only [List.length_set, hl]) hr1
        (by rw [← Nat.add_assoc]; exact h11_1)
        h2_1 h7_1 (by omega) ha64
        (by cases hlu : lookupA bs (u32 bs (p + 1)) (u32 bs (wordOff + 16)) annOff with
          | some q' => rw [hlu] at hm1; exact hm1
          | none => rw [hlu] at hm1; rw [← Nat.add_assoc]; exact hm1)
        hann).trans fun s2 ⟨hf2, hm2⟩ => ?_
      have hf02 : Frame [2, 7, 10, 11, 14, 15] _ s2 := Frame.trans_same hf01 (Frame.mono hf2 (by decide))
      cases hlu : lookupA bs (u32 bs (p + 1)) (u32 bs (wordOff + 16)) annOff with
      | none => rw [hlu] at hm2; exact Reaches.refl ⟨hf02, hm2⟩
      | some q =>
        rw [hlu] at hm2; dsimp only at hm2 ⊢
        by_cases heq : u32 bs (q + 4) = d - 1 ∧ u32 bs (q + 8) = a
        · rw [if_pos heq] at hm2
          xs k11 using hm2
          xs k12
          refine Reaches.refl ⟨Frame.trans_same hf02 (by frame_tac), ?_⟩
          simp only [heq, reduceIte, true_and]
          have hl2 : s2.regs.length = 16 := by rw [hf2.len, hf1.len]; simp only [List.length_set, hl]
          exact ⟨(by rsimpg; rw [hf2.regs 2 (by decide)]; exact h2_1),
            (by rsimpg; rw [hf2.regs 7 (by decide)]; exact h7_1), (by rsimpg using hl2)⟩
        · rw [if_neg heq] at hm2
          exact Reaches.refl ⟨hf02, by simp [heq, hm2]⟩

private theorem hstep_8b {B D R : Nat} (pb : PassBlocks B D R)
    {b p nw ws d a : Nat} {bs : List Nat} {s : St}
    (hbuf : Buf s.mem b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < 2 ^ 48)
    (hdesc : Desc s.mem D b bs.length) (hD : D + 8 < 2 ^ 64)
    (hsw : skipWordsA bs nw ws = some bs.length) (hnw : nw = u32 bs 8)
    (hl : s.regs.length = 16) (h5 : s.reg 5 = some (b + bs.length))
    (h6 : s.reg 6 = some (b + p)) (h13 : s.reg 13 = some (b + ws))
    (h2 : s.reg 2 = some d) (h7 : s.reg 7 = some a) (hd : d < HMAX)
    (hws : ws ≤ bs.length) (hbs16 : 16 ≤ bs.length) (hp5 : p + 5 ≤ bs.length)
    (hrip : s.rip = B + 2626) :
    Reaches s (fun s' => Frame [2, 7, 10, 11, 14, 15] s s' ∧
      let res :=
        if u32 bs (p + 1) < nw then
          match skipWordsA bs (u32 bs (p + 1)) ws with
          | none => StepResA.reject
          | some cw =>
            if d < u32 bs cw then .reject
            else if HMAX ≤ d - u32 bs cw + u32 bs (cw + 4) then .overflow else .fall (d - u32 bs cw + u32 bs (cw + 4)) a 5
        else .reject
      match res with
      | .reject => s'.rip = B + 3301
      | .overflow => s'.rip = B + 3286
      | .fall nd na' len => s'.rip = B + 2942 ∧ s'.reg 2 = some nd ∧ s'.reg 7 = some na' ∧ s'.reg 15 = some len
      | .nofall len => s'.rip = B + 3045 ∧ s'.reg 2 = some d ∧ s'.reg 7 = some a ∧ s'.reg 15 = some len) := by
  have hd48 : d < 2 ^ 48 := hd
  have hd64 : d < 2 ^ 64 := by omega
  have he : Env s b bs := ⟨hbuf, hby, hsm, hl, h5⟩
  have hc0 := pb.cCall0; simp only [CodeAt_cons, CodeAt_nil, and_true] at hc0
  obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, c0, c4, c10, c13, c17, c20, c26⟩ := hc0
  have hc1 := pb.cCall1; simp only [CodeAt_cons, CodeAt_nil, and_true] at hc1
  obtain ⟨c29, c32, c35, c41, c44, c48⟩ := hc1
  have hfl := pb.cFall; simp only [CodeAt_cons, CodeAt_nil, and_true] at hfl
  obtain ⟨f0, f6, _, _, _, _, _, _, _, _, _, _, _, _, _⟩ := hfl
  have hw := u32_lt hby (p + 1)
  have hnw32 := u32_lt hby 8
  xs c0 using hl, h6, ea_small (show b + p + 1 < 2 ^ 64 by omega),
    show b + p + 1 = b + (p + 1) by omega, rd32_buf hbuf (show p + 1 + 4 ≤ bs.length by omega)
  xs c4 using hl, show D % 18446744073709551616 = D from Nat.mod_eq_of_lt (by omega)
  xs c10 using hl, hdesc.ptr
  xs c13 using hl, ea_small (show b + 8 < 2 ^ 64 by omega),
    show b + 8 = b + 8 by rfl, rd32_buf hbuf (show 8 + 4 ≤ bs.length by omega)
  xs c17 using hl
  have hc3 := (cond_sub (show u32 bs (p + 1) < W by rw [W_eq]; omega)
    (show u32 bs 8 < W by rw [W_eq]; omega)).2.1
  by_cases hlt : u32 bs (p + 1) < nw
  · xs c20 using hc3, show ¬ (u32 bs 8 ≤ u32 bs (p + 1)) by omega
    xs c26 using hl, h13
    obtain ⟨cw, hcw, hcw20, _⟩ := skipWordsA_lt_step hlt hsw
    refine reach_frame (by frame_tac)
      ((skipwords_spec pb.cSwCall (u32 bs (p + 1)) ws (b := b) _ ?_ (by rsimpg; omega) (by rsimpg using hl)
        (by rsimpg using hl) hws (by omega)).trans fun s1 ⟨hf1, hm1⟩ => ?_)
    · exact he.frame (S := [2, 7, 10, 11, 14, 15]) (by frame_tac) (by decide)
    · rw [hcw] at hm1
      have hf01 : Frame [2, 7, 10, 11, 14, 15] _ s1 := Frame.mono hf1 (by decide)
      have hl1 : s1.regs.length = 16 := by rw [hf1.len]; simp only [List.length_set, hl]
      have h2_1 : s1.reg 2 = some d := by rw [hf1.regs 2 (by decide)]; rsimpg; exact h2
      have h7_1 : s1.reg 7 = some a := by rw [hf1.regs 7 (by decide)]; rsimpg; exact h7
      have hbuf1 : Buf s1.mem b bs := by rw [hf1.mem]; exact hbuf
      have hcins := u32_lt hby cw
      have hcouts := u32_lt hby (cw + 4)
      xs c29 using hl1, hm1.2.1, ea_small (show b + cw + 0 < 2 ^ 64 by omega),
        Nat.add_zero, rd32_buf hbuf1 (show cw + 4 ≤ bs.length by omega)
      xs c32 using hl1, h2_1
      have hc2 := (cond_sub (show d < W by rw [W_eq]; exact hd64)
        (show u32 bs cw < W by rw [W_eq]; omega)).1
      by_cases hlt2 : d < u32 bs cw
      · xs c35 using hc2, hlt2
        exact Reaches.refl ⟨Frame.trans_same hf01 (by frame_tac), by simp [hlt, hcw, hlt2]⟩
      · xs c35 using hc2, hlt2
        xs c41 using hl1, h2_1, sub_small hd64 (show u32 bs cw ≤ d by omega)
        xs c44 using hl1, hm1.2.1, ea_small (show b + cw + 4 < 2 ^ 64 by omega),
          show b + cw + 4 = b + (cw + 4) by omega, rd32_buf hbuf1 (show cw + 4 + 4 ≤ bs.length by omega)
        xs c48 using hl1, add_small (show d - u32 bs cw + u32 bs (cw + 4) < 2 ^ 64 by omega)
        refine (reach_frame (by frame_tac)
          ((chk_spec pb.cChkCall (by simp only [List.length_set, hl1]) (by rsimpg; omega)
            (by rsimpg using hl1) (show d - u32 bs cw + u32 bs (cw + 4) < 2 ^ 64 by omega)).trans
            fun s2 ⟨hf2, hm2⟩ => ?_)).mono fun s2 ⟨hf12, hm2'⟩ =>
              ⟨Frame.trans_same hf01 hf12, hm2'⟩
        have hf03 : Frame [2, 7, 10, 11, 14, 15] _ s2 := Frame.mono hf2 (by decide)
        by_cases hov : HMAX ≤ d - u32 bs cw + u32 bs (cw + 4)
        · rw [if_pos hov] at hm2
          exact Reaches.refl ⟨hf03, by simp [hlt, hcw, hlt2, hov, hm2]⟩
        · rw [if_neg hov] at hm2
          xs f0 using hm2
          xs f6
          refine Reaches.refl ⟨Frame.trans_same hf03 (by frame_tac), ?_⟩
          simp only [hlt, hcw, hlt2, hov, reduceIte, true_and]
          have hl2 : s2.regs.length = 16 := by rw [hf2.len]; simp only [List.length_set, hl1]
          exact ⟨(by rsimpg; rw [hf2.regs 2 (by decide)]; rsimpg using hl1),
            (by rsimpg; rw [hf2.regs 7 (by decide)]; rsimpg; exact h7_1), (by rsimpg using hl2)⟩
  · xs c20 using hc3, show u32 bs 8 ≤ u32 bs (p + 1) by omega
    exact Reaches.refl ⟨by frame_tac, by simp [hlt]⟩

private theorem hstep_8c {B D R : Nat} (pb : PassBlocks B D R)
    {b wordOff d a : Nat} {bs : List Nat} {s : St}
    (hbuf : Buf s.mem b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < 2 ^ 48)
    (hl : s.regs.length = 16) (h12 : s.reg 12 = some (b + wordOff))
    (h2 : s.reg 2 = some d) (h7 : s.reg 7 = some a) (hd : d < HMAX)
    (hw20 : wordOff + 20 ≤ bs.length) (hrip : s.rip = B + 2906) :
    Reaches s (fun s' => Frame [2, 7, 10, 11, 14, 15] s s' ∧
      let res := if d = u32 bs (wordOff + 4) ∧ a = 0 then StepResA.nofall 1 else .reject
      match res with
      | .reject => s'.rip = B + 3301
      | .overflow => s'.rip = B + 3286
      | .fall nd na' len => s'.rip = B + 2942 ∧ s'.reg 2 = some nd ∧ s'.reg 7 = some na' ∧ s'.reg 15 = some len
      | .nofall len => s'.rip = B + 3045 ∧ s'.reg 2 = some d ∧ s'.reg 7 = some a ∧ s'.reg 15 = some len) := by
  have hd48 : d < 2 ^ 48 := hd
  have hd64 : d < 2 ^ 64 := by omega
  have hfl := pb.cFall; simp only [CodeAt_cons, CodeAt_nil, and_true] at hfl
  obtain ⟨_, _, r0, r5, r8, r10, r15, r18, r20, r25, r31, _, _, _, _⟩ := hfl
  have houts := u32_lt hby (wordOff + 4)
  xs r0 using hl, h12, ea_small (show b + wordOff + 4 < 2 ^ 64 by omega),
    show b + wordOff + 4 = b + (wordOff + 4) by omega, rd32_buf hbuf (show wordOff + 4 + 4 ≤ bs.length by omega)
  xs r5 using hl, h2
  by_cases hdq : d = u32 bs (wordOff + 4)
  · have hc4 := (cond_sub (show u32 bs (wordOff + 4) < W by rw [W_eq]; omega)
      (show u32 bs (wordOff + 4) < W by rw [W_eq]; omega)).2.2.1
    simp only [decide_true] at hc4
    xs r8 using hdq, hc4
    xs r15 using h7
    by_cases ha0 : a = 0
    · xs r18 using ha0, cond_test_e
      xs r25
      xs r31
      refine Reaches.refl ⟨by frame_tac, rfl,
        (by rsimpg; rw [← hdq]; exact h2), (by rsimpg; rw [← ha0]; exact h7), (by rsimpg using hl)⟩
    · xs r18 using ha0, cond_test_e
      xs r20
      exact Reaches.refl ⟨by frame_tac, rfl⟩
  · have hc4 := (cond_sub (show d < W by rw [W_eq]; exact hd64)
      (show u32 bs (wordOff + 4) < W by rw [W_eq]; omega)).2.2.1
    xs r8 using hc4, hdq
    xs r10
    exact Reaches.refl ⟨by frame_tac, rfl⟩

theorem hstep_loc_jmp_call_ret {B D R : Nat} (pb : PassBlocks B D R)
    {b p wordOff annOff nw ws op d a : Nat} {bs : List Nat} {s : St}
    (hbuf : Buf s.mem b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < 2 ^ 48)
    (hdesc : Desc s.mem D b bs.length) (hD : D + 8 < 2 ^ 64)
    (hsw : skipWordsA bs nw ws = some bs.length) (hnw : nw = u32 bs 8)
    (hl : s.regs.length = 16) (h5 : s.reg 5 = some (b + bs.length))
    (h6 : s.reg 6 = some (b + p)) (h9 : s.reg 9 = some (b + annOff))
    (h12 : s.reg 12 = some (b + wordOff)) (h13 : s.reg 13 = some (b + ws))
    (h2 : s.reg 2 = some d) (h7 : s.reg 7 = some a)
    (hd : d < HMAX) (ha : a < HMAX)
    (hw20 : wordOff + 20 ≤ bs.length) (hann : annOff + 12 * u32 bs (wordOff + 16) ≤ bs.length)
    (hws : ws ≤ bs.length) (hbs16 : 16 ≤ bs.length)
    (hop : (op = 0x86 ∧ p + 5 ≤ bs.length) ∨ (op = 0x87 ∧ p + 5 ≤ bs.length) ∨
      (op = 0x88 ∧ p + 5 ≤ bs.length) ∨ ((op = 0x89 ∨ op = 0x8a) ∧ p + 5 ≤ bs.length) ∨
      (op = 0x8b ∧ p + 5 ≤ bs.length) ∨ op = 0x8c)
    (hrip : s.rip = if op = 0x86 then B + 2329
      else if op = 0x87 then B + 2379
      else if op = 0x88 then B + 2421
      else if op = 0x89 ∨ op = 0x8a then B + 2517
      else if op = 0x8b then B + 2626
      else B + 2906) :
    Reaches s (fun s' => Frame [2, 7, 10, 11, 14, 15] s s' ∧
      let res :=
        if op = 0x86 then
          let i := u32 bs (p + 1)
          if i < u32 bs (wordOff + 8) then
            let d' := d + 1
            if HMAX ≤ d' then StepResA.overflow else .fall d' a 5
          else .reject
        else if op = 0x87 then
          if d = 0 then .reject
          else if u32 bs (p + 1) < u32 bs (wordOff + 8) then .fall (d - 1) a 5
          else .reject
        else if op = 0x88 then
          let t := u32 bs (p + 1)
          match lookupA bs t (u32 bs (wordOff + 16)) annOff with
          | none => .reject
          | some q => if u32 bs (q + 4) = d ∧ u32 bs (q + 8) = a then .nofall 5 else .reject
        else if op = 0x89 ∨ op = 0x8a then
          if d = 0 then .reject
          else
            let d' := d - 1
            let t := u32 bs (p + 1)
            match lookupA bs t (u32 bs (wordOff + 16)) annOff with
            | none => .reject
            | some q => if u32 bs (q + 4) = d' ∧ u32 bs (q + 8) = a then .fall d' a 5 else .reject
        else if op = 0x8b then
          let w := u32 bs (p + 1)
          if w < nw then
            match skipWordsA bs w ws with
            | none => .reject
            | some cw =>
              let cins := u32 bs cw
              let couts := u32 bs (cw + 4)
              if d < cins then .reject
              else
                let d' := d - cins + couts
                if HMAX ≤ d' then .overflow else .fall d' a 5
          else .reject
        else
          if d = u32 bs (wordOff + 4) ∧ a = 0 then .nofall 1 else .reject
      match res with
      | .reject => s'.rip = B + 3301
      | .overflow => s'.rip = B + 3286
      | .fall nd na' len => s'.rip = B + 2942 ∧ s'.reg 2 = some nd ∧ s'.reg 7 = some na' ∧ s'.reg 15 = some len
      | .nofall len => s'.rip = B + 3045 ∧ s'.reg 2 = some d ∧ s'.reg 7 = some a ∧ s'.reg 15 = some len) := by
  rcases hop with ⟨rfl, hp5⟩ | ⟨rfl, hp5⟩ | ⟨rfl, hp5⟩ | ⟨h89a, hp5⟩ | ⟨rfl, hp5⟩ | rfl
  · simp (config := { decide := true }) only [reduceIte] at hrip ⊢
    exact hstep_86 pb hbuf hby hsm hl h6 h12 h2 h7 hd hw20 hp5 hrip
  · simp (config := { decide := true }) only [reduceIte] at hrip ⊢
    exact hstep_87 pb hbuf hby hsm hl h6 h12 h2 h7 hd hw20 hp5 hrip
  · simp (config := { decide := true }) only [reduceIte] at hrip ⊢
    exact hstep_88 pb hbuf hby hsm hl h6 h9 h12 h2 h7 hd ha hw20 hann hp5 hrip
  · have hno : op ≠ 0x86 ∧ op ≠ 0x87 ∧ op ≠ 0x88 := by rcases h89a with rfl | rfl <;> decide
    simp (config := { decide := true }) only [hno.1, hno.2.1, hno.2.2, h89a, reduceIte] at hrip ⊢
    exact hstep_89a pb hbuf hby hsm hl h6 h9 h12 h2 h7 hd ha hw20 hann hp5 hrip
  · simp (config := { decide := true }) only [reduceIte] at hrip ⊢
    exact hstep_8b pb hbuf hby hsm hdesc hD hsw hnw hl h5 h6 h13 h2 h7 hd hws hbs16 hp5 hrip
  · simp (config := { decide := true }) only [reduceIte] at hrip ⊢
    exact hstep_8c pb hbuf hby hsm hl h12 h2 h7 hd hw20 hrip

theorem hstep_spec {B D R : Nat} (pb : PassBlocks B D R)
    {b p wordOff annOff nw ws d a l : Nat} {bs : List Nat} {s : St}
    (hbuf : Buf s.mem b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < 2 ^ 48)
    (hdesc : Desc s.mem D b bs.length) (hD : D + 8 < 2 ^ 64)
    (hsw : skipWordsA bs nw ws = some bs.length) (hnw : nw = u32 bs 8)
    (hl : s.regs.length = 16) (h5 : s.reg 5 = some (b + bs.length))
    (hrip : s.rip = B + 866) (h6 : s.reg 6 = some (b + p))
    (h9 : s.reg 9 = some (b + annOff)) (h12 : s.reg 12 = some (b + wordOff))
    (h13 : s.reg 13 = some (b + ws)) (h2 : s.reg 2 = some d) (h7 : s.reg 7 = some a)
    (hd : d < HMAX) (ha : a < HMAX) (hp : p < bs.length)
    (hlen : instrLenA bs[p] = some l) (hpl : p + l ≤ bs.length)
    (hw20 : wordOff + 20 ≤ bs.length) (hann : annOff + 12 * u32 bs (wordOff + 16) ≤ bs.length)
    (hws : ws ≤ bs.length) (hbs16 : 16 ≤ bs.length) :
    Reaches s (fun s' => Frame [2, 7, 10, 11, 14, 15] s s' ∧
      match hstepA bs nw ws (u32 bs (wordOff + 4)) (u32 bs (wordOff + 8))
          (u32 bs (wordOff + 16)) annOff d a p with
      | .reject => s'.rip = B + 3301
      | .overflow => s'.rip = B + 3286
      | .fall nd na' len => s'.rip = B + 2942 ∧ s'.reg 2 = some nd ∧ s'.reg 7 = some na' ∧
          s'.reg 15 = some len ∧ len = l ∧ nd < HMAX ∧ na' < HMAX
      | .nofall len => s'.rip = B + 3045 ∧ s'.reg 2 = some d ∧ s'.reg 7 = some a ∧ s'.reg 15 = some len ∧ len = l) := by
  let op := bs[p]
  have hop : op < 256 := hby op (List.getElem_mem hp)
  have hopd : (bs[p]?).getD 255 = op := by simp [List.getElem?_eq_getElem hp, op]
  refine (hstep_dispatch pb hp hbuf rfl hop hsm hl hrip h6).trans fun s1 ⟨hf1, h15_1, hr1⟩ => ?_
  have hop_eq : bs[p] = op := rfl
  rw [hop_eq] at hr1 h15_1
  have hmem1 : s1.mem = s.mem := hf1.mem
  have hbuf1 : Buf s1.mem b bs := by rw [hmem1]; exact hbuf
  have hdesc1 : Desc s1.mem D b bs.length := ⟨by rw [hmem1]; exact hdesc.ptr, by rw [hmem1]; exact hdesc.len⟩
  have hl1 : s1.regs.length = 16 := by rw [hf1.len, hl]
  have h5_1 : s1.reg 5 = some (b + bs.length) := by rw [hf1.regs 5 (by decide)]; exact h5
  have h6_1 : s1.reg 6 = some (b + p) := by rw [hf1.regs 6 (by decide)]; exact h6
  have h9_1 : s1.reg 9 = some (b + annOff) := by rw [hf1.regs 9 (by decide)]; exact h9
  have h12_1 : s1.reg 12 = some (b + wordOff) := by rw [hf1.regs 12 (by decide)]; exact h12
  have h13_1 : s1.reg 13 = some (b + ws) := by rw [hf1.regs 13 (by decide)]; exact h13
  have h2_1 : s1.reg 2 = some d := by rw [hf1.regs 2 (by decide)]; exact h2
  have h7_1 : s1.reg 7 = some a := by rw [hf1.regs 7 (by decide)]; exact h7
  have hf01 : Frame [2, 7, 10, 11, 14, 15] s s1 := Frame.mono hf1 (by decide)
  by_cases h57 : op < 57
  · rw [if_pos h57] at hr1
    have hl_1 : l = 1 := by unfold instrLenA at hlen; simp [op, h57] at hlen; omega
    refine (hstep_prim pb hl1 hr1 h15_1 h2_1 h7_1 h57 hd).mono fun s2 ⟨hf2, hm2⟩ =>
      ⟨Frame.trans_same hf01 (Frame.mono hf2 (by decide)), ?_⟩
    simp only [hstepA, hopd, h57, reduceIte]
    by_cases h50 : op = 50
    · simp only [h50, reduceIte] at hm2 ⊢
      by_cases hd0 : d = 0
      · simp only [hd0, reduceIte] at hm2 ⊢; exact hm2
      · simp only [hd0, reduceIte] at hm2 ⊢; exact ⟨hm2.1, hm2.2.1, hm2.2.2.1, hm2.2.2.2, hl_1.symm⟩
    · simp only [h50, reduceIte] at hm2 ⊢
      by_cases hlt : d < (primEffTree.eval op).1
      · simp only [hlt, reduceIte] at hm2 ⊢; exact hm2
      · by_cases hov : HMAX ≤ d - (primEffTree.eval op).1 + (primEffTree.eval op).2
        · simp only [hlt, hov, reduceIte] at hm2 ⊢; exact hm2
        · simp only [hlt, hov, reduceIte] at hm2 ⊢
          exact ⟨hm2.1, hm2.2.1, hm2.2.2.1, hm2.2.2.2, hl_1.symm, by unfold HMAX at hov ⊢; omega, ha⟩
  · by_cases hpr : op = 0x80 ∨ op = 0x81 ∨ op = 0x82 ∨ op = 0x83 ∨ op = 0x84 ∨ op = 0x85
    · have hp5 : op = 0x82 → p + 5 ≤ bs.length := fun h => by
        unfold instrLenA at hlen; simp [op, h] at hlen; omega
      have hpre : op = 0x80 ∨ op = 0x81 ∨ (op = 0x82 ∧ p + 5 ≤ bs.length) ∨
          op = 0x83 ∨ op = 0x84 ∨ op = 0x85 := by
        rcases hpr with h | h | h | h | h | h <;> simp [h, hp5]
      have hr1' : s1.rip = if op = 0x80 then B + 2089
        else if op = 0x81 then B + 2121 else if op = 0x82 then B + 2153
        else if op = 0x83 then B + 2198 else if op = 0x84 then B + 2243 else B + 2288 := by
        rcases hpr with h | h | h | h | h | h <;>
          simp (config := { decide := true }) only [h, reduceIte] at hr1 ⊢ <;> exact hr1
      refine (hstep_push_rstack pb hbuf1 hby hsm hl1 h6_1 h2_1 h7_1 hd ha hpre hr1').mono
        fun s2 ⟨hf2, hm2⟩ => ⟨Frame.trans_same hf01 (Frame.mono hf2 (by decide)), ?_⟩
      rcases hpr with h | h | h | h | h | h
      · have hl9 : l = 9 := by unfold instrLenA at hlen; simp [op, h] at hlen; omega
        simp (config := { decide := true }) only [hstepA, hopd, h, reduceIte] at hm2 ⊢
        by_cases hov : HMAX ≤ d + 1
        · simp only [hov, reduceIte] at hm2 ⊢; exact hm2
        · simp only [hov, reduceIte] at hm2 ⊢
          exact ⟨hm2.1, hm2.2.1, hm2.2.2.1, hm2.2.2.2, hl9.symm, by unfold HMAX at hov ⊢; omega, ha⟩
      · have hl5 : l = 5 := by unfold instrLenA at hlen; simp [op, h] at hlen; omega
        simp (config := { decide := true }) only [hstepA, hopd, h, reduceIte] at hm2 ⊢
        by_cases hov : HMAX ≤ d + 2
        · simp only [hov, reduceIte] at hm2 ⊢; exact hm2
        · simp only [hov, reduceIte] at hm2 ⊢
          exact ⟨hm2.1, hm2.2.1, hm2.2.2.1, hm2.2.2.2, hl5.symm, by unfold HMAX at hov ⊢; omega, ha⟩
      · have hl5 : l = 5 := by unfold instrLenA at hlen; simp [op, h] at hlen; omega
        simp (config := { decide := true }) only [hstepA, hopd, h, reduceIte] at hm2 ⊢
        by_cases hlt : u32 bs (p + 1) < d
        · by_cases hov : HMAX ≤ d + 1
          · simp only [hlt, hov, reduceIte] at hm2 ⊢; exact hm2
          · simp only [hlt, hov, reduceIte] at hm2 ⊢
            exact ⟨hm2.1, hm2.2.1, hm2.2.2.1, hm2.2.2.2, hl5.symm, by unfold HMAX at hov ⊢; omega, ha⟩
        · simp only [hlt, reduceIte] at hm2 ⊢; exact hm2
      · have hl1 : l = 1 := by unfold instrLenA at hlen; simp [op, h] at hlen; omega
        simp (config := { decide := true }) only [hstepA, hopd, h, reduceIte] at hm2 ⊢
        by_cases hd0 : d = 0
        · simp only [hd0, reduceIte] at hm2 ⊢; exact hm2
        · by_cases hov : HMAX ≤ a + 1
          · simp only [hd0, hov, reduceIte] at hm2 ⊢; exact hm2
          · simp only [hd0, hov, reduceIte] at hm2 ⊢
            exact ⟨hm2.1, hm2.2.1, hm2.2.2.1, hm2.2.2.2, hl1.symm, by unfold HMAX at hd ⊢; omega, by unfold HMAX at hov ⊢; omega⟩
      · have hl1 : l = 1 := by unfold instrLenA at hlen; simp [op, h] at hlen; omega
        simp (config := { decide := true }) only [hstepA, hopd, h, reduceIte] at hm2 ⊢
        by_cases ha0 : a = 0
        · simp only [ha0, reduceIte] at hm2 ⊢; exact hm2
        · by_cases hov : HMAX ≤ d + 1
          · simp only [ha0, hov, reduceIte] at hm2 ⊢; exact hm2
          · simp only [ha0, hov, reduceIte] at hm2 ⊢
            exact ⟨hm2.1, hm2.2.1, hm2.2.2.1, hm2.2.2.2, hl1.symm, by unfold HMAX at hov ⊢; omega, by unfold HMAX at ha ⊢; omega⟩
      · have hl1 : l = 1 := by unfold instrLenA at hlen; simp [op, h] at hlen; omega
        simp (config := { decide := true }) only [hstepA, hopd, h, reduceIte] at hm2 ⊢
        by_cases ha0 : a = 0
        · simp only [ha0, reduceIte] at hm2 ⊢; exact hm2
        · by_cases hov : HMAX ≤ d + 1
          · simp only [ha0, hov, reduceIte] at hm2 ⊢; exact hm2
          · simp only [ha0, hov, reduceIte] at hm2 ⊢
            exact ⟨hm2.1, hm2.2.1, hm2.2.2.1, hm2.2.2.2, hl1.symm, by unfold HMAX at hov ⊢; omega, ha⟩
    · have hop_rng : op = 0x86 ∨ op = 0x87 ∨ op = 0x88 ∨ op = 0x89 ∨ op = 0x8a ∨ op = 0x8b ∨ op = 0x8c := by
        rcases instrLenA_range hlen with h1 | h1
        · rw [hop_eq] at h1; exact absurd h1 h57
        · rw [hop_eq] at h1; omega
      have hl_of : op ≠ 0x8c → l = 5 := fun hne => by
        unfold instrLenA at hlen
        rcases hop_rng with h|h|h|h|h|h|h <;> simp [op, h] at hlen hne <;> omega
      have hrest : (op = 0x86 ∧ p + 5 ≤ bs.length) ∨ (op = 0x87 ∧ p + 5 ≤ bs.length) ∨
          (op = 0x88 ∧ p + 5 ≤ bs.length) ∨ ((op = 0x89 ∨ op = 0x8a) ∧ p + 5 ≤ bs.length) ∨
          (op = 0x8b ∧ p + 5 ≤ bs.length) ∨ op = 0x8c := by
        rcases hop_rng with h | h | h | h | h | h | h
        · exact Or.inl ⟨h, by have := hl_of (by omega); omega⟩
        · exact Or.inr (Or.inl ⟨h, by have := hl_of (by omega); omega⟩)
        · exact Or.inr (Or.inr (Or.inl ⟨h, by have := hl_of (by omega); omega⟩))
        · exact Or.inr (Or.inr (Or.inr (Or.inl ⟨Or.inl h, by have := hl_of (by omega); omega⟩)))
        · exact Or.inr (Or.inr (Or.inr (Or.inl ⟨Or.inr h, by have := hl_of (by omega); omega⟩)))
        · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨h, by have := hl_of (by omega); omega⟩))))
        · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr h))))
      have hr1' : s1.rip = if op = 0x86 then B + 2329
        else if op = 0x87 then B + 2379
        else if op = 0x88 then B + 2421
        else if op = 0x89 ∨ op = 0x8a then B + 2517
        else if op = 0x8b then B + 2626
        else B + 2906 := by
        rcases hop_rng with h | h | h | h | h | h | h <;>
          simp (config := { decide := true }) only [h, reduceIte, true_or, or_true] at hr1 ⊢ <;> exact hr1
      refine (hstep_loc_jmp_call_ret pb hbuf1 hby hsm hdesc1 hD hsw hnw hl1 h5_1 h6_1 h9_1 h12_1 h13_1
        h2_1 h7_1 hd ha hw20 hann hws hbs16 hrest hr1').mono fun s2 ⟨hf2, hm2⟩ =>
        ⟨Frame.trans_same hf01 hf2, ?_⟩
      rcases hop_rng with h | h | h | h | h | h | h
      · have hl5 : l = 5 := hl_of (by omega)
        simp (config := { decide := true }) only [hstepA, hopd, h, reduceIte] at hm2 ⊢
        by_cases hlt : u32 bs (p + 1) < u32 bs (wordOff + 8)
        · by_cases hov : HMAX ≤ d + 1
          · simp only [hlt, hov, reduceIte] at hm2 ⊢; exact hm2
          · simp only [hlt, hov, reduceIte] at hm2 ⊢
            exact ⟨hm2.1, hm2.2.1, hm2.2.2.1, hm2.2.2.2, hl5.symm, by unfold HMAX at hov ⊢; omega, ha⟩
        · simp only [hlt, reduceIte] at hm2 ⊢; exact hm2
      · have hl5 : l = 5 := hl_of (by omega)
        simp (config := { decide := true }) only [hstepA, hopd, h, reduceIte] at hm2 ⊢
        by_cases hd0 : d = 0
        · simp only [hd0, reduceIte] at hm2 ⊢; exact hm2
        · by_cases hlt : u32 bs (p + 1) < u32 bs (wordOff + 8)
          · simp only [hd0, hlt, reduceIte] at hm2 ⊢
            exact ⟨hm2.1, hm2.2.1, hm2.2.2.1, hm2.2.2.2, hl5.symm, by unfold HMAX at hd ⊢; omega, ha⟩
          · simp only [hd0, hlt, reduceIte] at hm2 ⊢; exact hm2
      · have hl5 : l = 5 := hl_of (by omega)
        simp (config := { decide := true }) only [hstepA, hopd, h, reduceIte] at hm2 ⊢
        cases hlu : lookupA bs (u32 bs (p + 1)) (u32 bs (wordOff + 16)) annOff with
        | none => rw [hlu] at hm2; exact hm2
        | some q =>
          rw [hlu] at hm2; dsimp only at hm2 ⊢
          by_cases heq : u32 bs (q + 4) = d ∧ u32 bs (q + 8) = a
          · rw [if_pos heq] at hm2 ⊢; exact ⟨hm2.1, hm2.2.1, hm2.2.2.1, hm2.2.2.2, hl5.symm⟩
          · rw [if_neg heq] at hm2 ⊢; exact hm2
      · have hl5 : l = 5 := hl_of (by omega)
        simp (config := { decide := true }) only [hstepA, hopd, h, reduceIte] at hm2 ⊢
        by_cases hd0 : d = 0
        · simp only [hd0, reduceIte] at hm2 ⊢; exact hm2
        · simp only [hd0, reduceIte] at hm2 ⊢
          cases hlu : lookupA bs (u32 bs (p + 1)) (u32 bs (wordOff + 16)) annOff with
          | none => rw [hlu] at hm2; exact hm2
          | some q =>
            rw [hlu] at hm2; dsimp only at hm2 ⊢
            by_cases heq : u32 bs (q + 4) = d - 1 ∧ u32 bs (q + 8) = a
            · rw [if_pos heq] at hm2 ⊢
              exact ⟨hm2.1, hm2.2.1, hm2.2.2.1, hm2.2.2.2, hl5.symm, by unfold HMAX at hd ⊢; omega, ha⟩
            · rw [if_neg heq] at hm2 ⊢; exact hm2
      · have hl5 : l = 5 := hl_of (by omega)
        simp (config := { decide := true }) only [hstepA, hopd, h, reduceIte] at hm2 ⊢
        by_cases hd0 : d = 0
        · simp only [hd0, reduceIte] at hm2 ⊢; exact hm2
        · simp only [hd0, reduceIte] at hm2 ⊢
          cases hlu : lookupA bs (u32 bs (p + 1)) (u32 bs (wordOff + 16)) annOff with
          | none => rw [hlu] at hm2; exact hm2
          | some q =>
            rw [hlu] at hm2; dsimp only at hm2 ⊢
            by_cases heq : u32 bs (q + 4) = d - 1 ∧ u32 bs (q + 8) = a
            · rw [if_pos heq] at hm2 ⊢
              exact ⟨hm2.1, hm2.2.1, hm2.2.2.1, hm2.2.2.2, hl5.symm, by unfold HMAX at hd ⊢; omega, ha⟩
            · rw [if_neg heq] at hm2 ⊢; exact hm2
      · have hl5 : l = 5 := hl_of (by omega)
        simp (config := { decide := true }) only [hstepA, hopd, h, reduceIte] at hm2 ⊢
        by_cases hw : u32 bs (p + 1) < nw
        · simp only [hw, reduceIte] at hm2 ⊢
          cases hsw' : skipWordsA bs (u32 bs (p + 1)) ws with
          | none => rw [hsw'] at hm2; exact hm2
          | some cw =>
            rw [hsw'] at hm2
            dsimp only at hm2 ⊢
            by_cases hcins : d < u32 bs cw
            · simp only [hcins, reduceIte] at hm2 ⊢; exact hm2
            · by_cases hov : HMAX ≤ d - u32 bs cw + u32 bs (cw + 4)
              · simp only [hcins, hov, reduceIte] at hm2 ⊢; exact hm2
              · simp only [hcins, hov, reduceIte] at hm2 ⊢
                exact ⟨hm2.1, hm2.2.1, hm2.2.2.1, hm2.2.2.2, hl5.symm, by unfold HMAX at hov ⊢; omega, ha⟩
        · simp only [hw, reduceIte] at hm2 ⊢; exact hm2
      · have hl1 : l = 1 := by unfold instrLenA at hlen; simp [op, h] at hlen; omega
        simp (config := { decide := true }) only [hstepA, hopd, h, reduceIte] at hm2 ⊢
        by_cases heq : d = u32 bs (wordOff + 4) ∧ a = 0
        · rw [if_pos heq] at hm2 ⊢; exact ⟨hm2.1, hm2.2.1, hm2.2.2.1, hm2.2.2.2, hl1.symm⟩
        · rw [if_neg heq] at hm2 ⊢; exact hm2

/-! ### `walk_spec`, `anns_spec`, `checkWord_spec`, `checkWords_spec`, `parse_spec`, `pass_spec` -/

private theorem opt_cases {α : Type} (o : Option α) : o = none ∨ ∃ x, o = some x := by
  cases o <;> simp

theorem skipCodeA_succ {bs : List Nat} {n pos annOff : Nat}
    (h : skipCodeA bs (n + 1) pos = some annOff) :
    ∃ hp : pos < bs.length, ∃ l, instrLenA bs[pos] = some l ∧ pos + l ≤ bs.length ∧
      skipCodeA bs n (pos + l) = some annOff := by
  rw [skipCodeA.eq_def] at h
  dsimp only at h
  split at h
  · rename_i hp
    cases hl : instrLenA bs[pos] with
    | none => simp [hl] at h
    | some l =>
      simp only [hl] at h
      split at h
      · simp at h
      · exact ⟨hp, l, hl, by omega, h⟩
  · simp at h

theorem skipWordsA_one {bs : List Nat} {wordOff nextOff : Nat}
    (h : skipWordsA bs 1 wordOff = some nextOff) :
    wordOff + 20 ≤ bs.length ∧ ∃ annOff,
      skipCodeA bs (u32 bs (wordOff + 12)) (wordOff + 20) = some annOff ∧
      annOff + 12 * u32 bs (wordOff + 16) ≤ bs.length ∧
      nextOff = annOff + 12 * u32 bs (wordOff + 16) := by
  rw [skipWordsA.eq_def] at h
  dsimp only at h
  split at h
  · simp at h
  · cases hsc : skipCodeA bs (u32 bs (wordOff + 12)) (wordOff + 20) with
    | none => simp [hsc] at h
    | some annOff =>
      simp only [hsc] at h
      split at h
      · simp at h
      · simp only [skipWordsA.eq_def, Option.some.injEq] at h
        exact ⟨by omega, annOff, rfl, by omega, h.symm⟩

private def checkResRip (B : Nat) : CheckResA → Nat
  | .ok => B + 3271
  | .overflow => B + 3286
  | .reject => B + 3301

private def checkResVal : CheckResA → Nat
  | .ok => ACCEPT
  | .overflow => OVERFLOW
  | .reject => REJECT

private theorem walk_pre {B D R : Nat} (pb : PassBlocks B D R)
    {b wordOff annOff i curD curA : Nat} {bs : List Nat} {s : St}
    (hbuf : Buf s.mem b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < 2 ^ 48)
    (hl : s.regs.length = 16) (hrip : s.rip = B + 809)
    (h8 : s.reg 8 = some i) (h9 : s.reg 9 = some (b + annOff))
    (h12 : s.reg 12 = some (b + wordOff)) (h2 : s.reg 2 = some curD)
    (h7 : s.reg 7 = some curA) (hi : i < 2 ^ 64)
    (hw20 : wordOff + 20 ≤ bs.length) (hann : annOff + 12 * u32 bs (wordOff + 16) ≤ bs.length) :
    Reaches s (fun s' => Frame [2, 7, 10, 11, 14] s s' ∧ s'.rip = B + 866 ∧
      let h := match lookupA bs i (u32 bs (wordOff + 16)) annOff with
        | some q => (u32 bs (q + 4), u32 bs (q + 8))
        | none => (curD, curA)
      s'.reg 2 = some h.1 ∧ s'.reg 7 = some h.2) := by
  have hdisp := pb.cDisp; simp only [CodeAt_cons, CodeAt_nil, and_true] at hdisp
  obtain ⟨d0, d1, d2, d3, _⟩ := hdisp
  refine (lookup_spec pb.cLuI (by decide) (by decide) (by decide) (by decide)
    hbuf hby hsm hl hrip h12 h9 h8 hi hw20 hann).trans fun s1 ⟨hf1, hr1, h11_1, hm1⟩ => ?_
  have hl1 : s1.regs.length = 16 := by rw [hf1.len, hl]
  have h2_1 : s1.reg 2 = some curD := by rw [hf1.regs 2 (by decide)]; exact h2
  have h7_1 : s1.reg 7 = some curA := by rw [hf1.regs 7 (by decide)]; exact h7
  have hbuf1 : Buf s1.mem b bs := by rw [hf1.mem]; exact hbuf
  cases hlu : lookupA bs i (u32 bs (wordOff + 16)) annOff with
  | none =>
    rw [hlu] at hm1
    xs d0 using hm1, h11_1
    have hc4 := (cond_sub (show b + annOff + 12 * u32 bs (wordOff + 16) < W by rw [W_eq]; omega)
      (show b + annOff + 12 * u32 bs (wordOff + 16) < W by rw [W_eq]; omega)).2.2.1
    simp only [decide_true] at hc4
    xs d1 using hc4
    exact Reaches.refl ⟨Frame.trans_same (Frame.mono hf1 (by decide)) (by frame_tac),
      rfl, h2_1, h7_1⟩
  | some q =>
    rw [hlu] at hm1
    obtain ⟨h10_1, hq12⟩ := hm1
    xs d0 using h10_1, h11_1
    have hc4 := (cond_sub (show b + q < W by rw [W_eq]; omega)
      (show b + annOff + 12 * u32 bs (wordOff + 16) < W by rw [W_eq]; omega)).2.2.1
    xs d1 using hc4, show b + q ≠ b + annOff + 12 * u32 bs (wordOff + 16) by omega
    have hu4 := u32_lt hby (q + 4)
    have hu8 := u32_lt hby (q + 8)
    xs d2 using hl1, h10_1, ea_small (show b + q + 4 < 2 ^ 64 by omega),
      show b + q + 4 = b + (q + 4) by omega, rd32_buf hbuf1 (show q + 4 + 4 ≤ bs.length by omega)
    xs d3 using hl1, h10_1, ea_small (show b + q + 8 < 2 ^ 64 by omega),
      show b + q + 8 = b + (q + 8) by omega, rd32_buf hbuf1 (show q + 8 + 4 ≤ bs.length by omega)
    exact Reaches.refl ⟨Frame.trans_same (Frame.mono hf1 (by decide)) (by frame_tac),
      (by rw [hr1]), (by rsimpg using hl1), (by rsimpg using hl1)⟩

private theorem walk_ann_ok {B D R : Nat} (pb : PassBlocks B D R)
    {b idx na annOff annEnd d a : Nat} {bs : List Nat} {s : St}
    (hbuf : Buf s.mem b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < 2 ^ 48)
    (hl : s.regs.length = 16) (hrip : s.rip = B + 3004)
    (h11 : s.reg 11 = some (b + annEnd)) (h2 : s.reg 2 = some d) (h7 : s.reg 7 = some a)
    (hd : d < 2 ^ 64) (ha : a < 2 ^ 64)
    (hres : match lookupA bs idx na annOff with
      | some q' => s.reg 10 = some (b + q') ∧ q' + 12 ≤ annEnd
      | none => s.reg 10 = some (b + annEnd))
    (hann : annEnd ≤ bs.length) :
    Reaches s (fun s' => Frame [14] s s' ∧
      match lookupA bs idx na annOff with
      | none => s'.rip = B + 3037
      | some q => if u32 bs (q + 4) = d ∧ u32 bs (q + 8) = a then s'.rip = B + 3037 else s'.rip = B + 3301) := by
  have hnf := pb.cNofall; simp only [CodeAt_cons, CodeAt_nil, and_true] at hnf
  obtain ⟨n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, _⟩ := hnf
  cases hlu : lookupA bs idx na annOff with
  | none =>
    rw [hlu] at hres
    xs n0 using hres, h11
    have hc4 := (cond_sub (show b + annEnd < W by rw [W_eq]; omega)
      (show b + annEnd < W by rw [W_eq]; omega)).2.2.1
    simp only [decide_true] at hc4
    xs n1 using hc4
    exact Reaches.refl ⟨by frame_tac, rfl⟩
  | some q =>
    rw [hlu] at hres
    obtain ⟨h10, hq12⟩ := hres
    xs n0 using h10, h11
    have hc4 := (cond_sub (show b + q < W by rw [W_eq]; omega)
      (show b + annEnd < W by rw [W_eq]; omega)).2.2.1
    xs n1 using hc4, show b + q ≠ b + annEnd by omega
    have hu4 := u32_lt hby (q + 4)
    have hu8 := u32_lt hby (q + 8)
    xs n2 using hl, h10, ea_small (show b + q + 4 < 2 ^ 64 by omega),
      show b + q + 4 = b + (q + 4) by omega, rd32_buf hbuf (show q + 4 + 4 ≤ bs.length by omega)
    xs n3 using hl, h2
    by_cases h4eq : u32 bs (q + 4) = d
    · have hc4a := (cond_sub (show d < W by rw [W_eq]; exact hd)
        (show d < W by rw [W_eq]; exact hd)).2.2.1
      simp only [decide_true] at hc4a
      xs n4 using h4eq, hc4a
      xs n6 using hl, h10, ea_small (show b + q + 8 < 2 ^ 64 by omega),
        show b + q + 8 = b + (q + 8) by omega, rd32_buf hbuf (show q + 8 + 4 ≤ bs.length by omega)
      xs n7 using hl, h7
      by_cases h8eq : u32 bs (q + 8) = a
      · have hc4b := (cond_sub (show a < W by rw [W_eq]; exact ha)
          (show a < W by rw [W_eq]; exact ha)).2.2.1
        simp only [decide_true] at hc4b
        xs n8 using h8eq, hc4b
        exact Reaches.refl ⟨by frame_tac, rfl⟩
      · have hc4b := (cond_sub (show u32 bs (q + 8) < W by rw [W_eq]; omega)
          (show a < W by rw [W_eq]; exact ha)).2.2.1
        xs n8 using hc4b, h8eq
        xs n9
        exact Reaches.refl ⟨by frame_tac, rfl⟩
    · have hc4a := (cond_sub (show u32 bs (q + 4) < W by rw [W_eq]; omega)
        (show d < W by rw [W_eq]; exact hd)).2.2.1
      xs n4 using hc4a, h4eq
      xs n5
      exact Reaches.refl ⟨by frame_tac, rfl⟩

private theorem walk_nofall_nxt {B D R : Nat} (pb : PassBlocks B D R)
    {b wordOff annOff i d a : Nat} {bs : List Nat} {s : St}
    (hbuf : Buf s.mem b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < 2 ^ 48)
    (hl : s.regs.length = 16) (hrip : s.rip = B + 3045)
    (h8 : s.reg 8 = some i) (h9 : s.reg 9 = some (b + annOff))
    (h12 : s.reg 12 = some (b + wordOff)) (h2 : s.reg 2 = some d) (h7 : s.reg 7 = some a)
    (hi : i + 1 < 2 ^ 64) (hw20 : wordOff + 20 ≤ bs.length)
    (hann : annOff + 12 * u32 bs (wordOff + 16) ≤ bs.length) :
    Reaches s (fun s' => Frame [2, 7, 8, 10, 11, 14] s s' ∧ s'.rip = B + 3037 ∧
      s'.reg 8 = some (i + 1) ∧
      let nxt := match lookupA bs (i + 1) (u32 bs (wordOff + 16)) annOff with
        | some q => (u32 bs (q + 4), u32 bs (q + 8))
        | none => (0, 0)
      s'.reg 2 = some nxt.1 ∧ s'.reg 7 = some nxt.2) := by
  have hnf := pb.cNofall; simp only [CodeAt_cons, CodeAt_nil, and_true] at hnf
  obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, nf0⟩ := hnf
  have hap := pb.cAnnsPre; simp only [CodeAt_cons, CodeAt_nil, and_true] at hap
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7, _⟩ := hap
  xs nf0 using hl, h8, add_small hi
  refine reach_frame (by frame_tac)
    ((lookup_spec pb.cLuNof (by decide) (by decide) (by decide) (by decide)
      (s := _) (idx := i + 1) ?_ hby hsm (by simp only [List.length_set, hl]) (by rsimpg; omega)
      (by rsimpg; exact h12) (by rsimpg; exact h9) (by rsimpg using hl) hi hw20 hann).trans
      fun s1 ⟨hf1, hr1, h11_1, hm1⟩ => ?_)
  · exact hbuf
  · have hl1 : s1.regs.length = 16 := by rw [hf1.len]; simp only [List.length_set, hl]
    have h8_1 : s1.reg 8 = some (i + 1) := by rw [hf1.regs 8 (by decide)]; rsimpg using hl
    have h2_1 : s1.reg 2 = some d := by rw [hf1.regs 2 (by decide)]; rsimpg; exact h2
    have h7_1 : s1.reg 7 = some a := by rw [hf1.regs 7 (by decide)]; rsimpg; exact h7
    have hbuf1 : Buf s1.mem b bs := by rw [hf1.mem]; exact hbuf
    cases hlu : lookupA bs (i + 1) (u32 bs (wordOff + 16)) annOff with
    | none =>
      rw [hlu] at hm1
      xs a0 using hm1, h11_1
      have hc4 := (cond_sub (show b + annOff + 12 * u32 bs (wordOff + 16) < W by rw [W_eq]; omega)
        (show b + annOff + 12 * u32 bs (wordOff + 16) < W by rw [W_eq]; omega)).2.2.1
      simp only [decide_true] at hc4
      xs a1 using hc4
      xs a5 using hl1, h2_1, alu_xor_self
      xs a6 using hl1, h7_1, alu_xor_self
      xs a7
      exact Reaches.refl ⟨Frame.trans_same (Frame.mono hf1 (by decide)) (by frame_tac),
        rfl, (by rsimpg; exact h8_1), (by rsimpg using hl1), (by rsimpg using hl1)⟩
    | some q =>
      rw [hlu] at hm1
      obtain ⟨h10_1, hq12⟩ := hm1
      xs a0 using h10_1, h11_1
      have hc4 := (cond_sub (show b + q < W by rw [W_eq]; omega)
        (show b + annOff + 12 * u32 bs (wordOff + 16) < W by rw [W_eq]; omega)).2.2.1
      xs a1 using hc4, show b + q ≠ b + annOff + 12 * u32 bs (wordOff + 16) by omega
      have hu4 := u32_lt hby (q + 4)
      have hu8 := u32_lt hby (q + 8)
      xs a2 using hl1, h10_1, ea_small (show b + q + 4 < 2 ^ 64 by omega),
        show b + q + 4 = b + (q + 4) by omega, rd32_buf hbuf1 (show q + 4 + 4 ≤ bs.length by omega)
      xs a3 using hl1, h10_1, ea_small (show b + q + 8 < 2 ^ 64 by omega),
        show b + q + 8 = b + (q + 8) by omega, rd32_buf hbuf1 (show q + 8 + 4 ≤ bs.length by omega)
      xs a4
      exact Reaches.refl ⟨Frame.trans_same (Frame.mono hf1 (by decide)) (by frame_tac),
        rfl, (by rsimpg; exact h8_1), (by rsimpg using hl1), (by rsimpg using hl1)⟩

theorem walk_spec {B D R : Nat} (pb : PassBlocks B D R)
    {b wordOff annOff nw ws : Nat} {bs : List Nat}
    (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < 2 ^ 48)
    (hD : D + 8 < 2 ^ 64) (hsw : skipWordsA bs nw ws = some bs.length) (hnw : nw = u32 bs 8)
    (hw20 : wordOff + 20 ≤ bs.length) (hann : annOff + 12 * u32 bs (wordOff + 16) ≤ bs.length)
    (hws : ws ≤ bs.length) (hbs16 : 16 ≤ bs.length) :
    ∀ rem i p curD curA (s : St),
      Buf s.mem b bs → Desc s.mem D b bs.length → s.regs.length = 16 →
      s.rip = B + 795 → s.reg 5 = some (b + bs.length) →
      s.reg 6 = some (b + p) → s.reg 8 = some i →
      s.reg 9 = some (b + annOff) → s.reg 12 = some (b + wordOff) →
      s.reg 13 = some (b + ws) → s.reg 2 = some curD → s.reg 7 = some curA →
      rem + i = u32 bs (wordOff + 12) → skipCodeA bs rem p = some annOff →
      curD < HMAX → curA < HMAX →
      Reaches s (fun s' => Frame [2, 6, 7, 8, 10, 11, 14, 15] s s' ∧
        (∃ v8', s'.reg 8 = some v8') ∧
        let res := walkA bs nw ws (u32 bs (wordOff + 4)) (u32 bs (wordOff + 8))
          (u32 bs (wordOff + 12)) (u32 bs (wordOff + 16)) annOff rem i p curD curA
        match res with
        | .ok => s'.rip = B + 3120
        | .overflow => s'.rip = B + 3286
        | .reject => s'.rip = B + 3301)
  | 0, i, p, curD, curA, s, hbuf, hdesc, hl, hrip, h5, h6, h8, h9, h12, h13, h2, h7, hri, _, _, _ => by
    have hw0 := pb.cWalk0; simp only [CodeAt_cons, CodeAt_nil, and_true] at hw0
    obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, w14, w15, w16⟩ := hw0
    have hnc := u32_lt hby (wordOff + 12)
    have hinc : i = u32 bs (wordOff + 12) := by omega
    xs w14 using hl, h12, ea_small (show b + wordOff + 12 < 2 ^ 64 by omega),
      show b + wordOff + 12 = b + (wordOff + 12) by omega, rd32_buf hbuf (show wordOff + 12 + 4 ≤ bs.length by omega)
    xs w15 using hl, h8
    have hc4 := (cond_sub (show u32 bs (wordOff + 12) < W by rw [W_eq]; omega)
      (show u32 bs (wordOff + 12) < W by rw [W_eq]; omega)).2.2.1
    simp only [decide_true] at hc4
    xs w16 using hinc, hc4
    simp only [walkA.eq_def]
    exact Reaches.refl ⟨by frame_tac, ⟨i, by rsimpg; exact h8⟩, rfl⟩
  | rem + 1, i, p, curD, curA, s, hbuf, hdesc, hl, hrip, h5, h6, h8, h9, h12, h13, h2, h7, hri, hsk, hd, ha => by
    have hw0 := pb.cWalk0; simp only [CodeAt_cons, CodeAt_nil, and_true] at hw0
    obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, w14, w15, w16⟩ := hw0
    have hfl := pb.cFall; simp only [CodeAt_cons, CodeAt_nil, and_true] at hfl
    obtain ⟨_, _, _, _, _, _, _, _, _, _, _, f11, f12, f13, f14⟩ := hfl
    have hnf := pb.cNofall; simp only [CodeAt_cons, CodeAt_nil, and_true] at hnf
    obtain ⟨_, _, _, _, _, _, _, _, _, _, n10, n11, _⟩ := hnf
    have hnc := u32_lt hby (wordOff + 12)
    obtain ⟨hp, l, hlen, hpl, hsk'⟩ := skipCodeA_succ hsk
    have ⟨_, hann_le⟩ := skipCodeA_bounds hpl hsk'
    have hne : i ≠ u32 bs (wordOff + 12) := by omega
    xs w14 using hl, h12, ea_small (show b + wordOff + 12 < 2 ^ 64 by omega),
      show b + wordOff + 12 = b + (wordOff + 12) by omega, rd32_buf hbuf (show wordOff + 12 + 4 ≤ bs.length by omega)
    xs w15 using hl, h8
    have hc4 := (cond_sub (show i < W by rw [W_eq]; omega)
      (show u32 bs (wordOff + 12) < W by rw [W_eq]; omega)).2.2.1
    xs w16 using hc4, hne
    refine reach_frame (by frame_tac)
      ((walk_pre pb (b := b) (wordOff := wordOff) (annOff := annOff)
        (i := i) (curD := curD) (curA := curA) (bs := bs) (s := _)
        ?_ hby hsm (by simp only [List.length_set, hl]) (by rsimpg; rw [hrip])
        (by rsimpg; exact h8) (by rsimpg; exact h9) (by rsimpg; exact h12)
        (by rsimpg; exact h2) (by rsimpg; exact h7) (by omega) hw20 hann).trans
        fun s1 ⟨hf1, hr1, h2_1, h7_1⟩ => ?_)
    · exact hbuf
    let cur := match lookupA bs i (u32 bs (wordOff + 16)) annOff with
      | some q => (u32 bs (q + 4), u32 bs (q + 8)) | none => (curD, curA)
    have hcurD : cur.1 < HMAX := by
      dsimp only [cur]
      cases lookupA bs i (u32 bs (wordOff + 16)) annOff with
      | none => exact hd
      | some q => have := u32_lt hby (q + 4); simp only [HMAX]; omega
    have hcurA : cur.2 < HMAX := by
      dsimp only [cur]
      cases lookupA bs i (u32 bs (wordOff + 16)) annOff with
      | none => exact ha
      | some q => have := u32_lt hby (q + 8); simp only [HMAX]; omega
    have hl1 : s1.regs.length = 16 := by rw [hf1.len]; simp only [List.length_set, hl]
    have hbuf1 : Buf s1.mem b bs := by rw [hf1.mem]; exact hbuf
    have hdesc1 : Desc s1.mem D b bs.length := ⟨by rw [hf1.mem]; exact hdesc.ptr, by rw [hf1.mem]; exact hdesc.len⟩
    have h5_1 : s1.reg 5 = some (b + bs.length) := by rw [hf1.regs 5 (by decide)]; rsimpg; exact h5
    have h6_1 : s1.reg 6 = some (b + p) := by rw [hf1.regs 6 (by decide)]; rsimpg; exact h6
    have h8_1 : s1.reg 8 = some i := by rw [hf1.regs 8 (by decide)]; rsimpg; exact h8
    have h9_1 : s1.reg 9 = some (b + annOff) := by rw [hf1.regs 9 (by decide)]; rsimpg; exact h9
    have h12_1 : s1.reg 12 = some (b + wordOff) := by rw [hf1.regs 12 (by decide)]; rsimpg; exact h12
    have h13_1 : s1.reg 13 = some (b + ws) := by rw [hf1.regs 13 (by decide)]; rsimpg; exact h13
    have hf01 : Frame [2, 6, 7, 8, 10, 11, 14, 15] _ s1 := Frame.mono hf1 (by decide)
    refine (hstep_spec pb hbuf1 hby hsm hdesc1 hD hsw hnw hl1 h5_1 hr1 h6_1 h9_1 h12_1 h13_1
      h2_1 h7_1 hcurD hcurA hp hlen (by omega) hw20 hann hws hbs16).trans
      fun s2 ⟨hf2, hm2⟩ => ?_
    have hf02 : Frame [2, 6, 7, 8, 10, 11, 14, 15] _ s2 := Frame.trans_same hf01 (Frame.mono hf2 (by decide))
    cases hst : hstepA bs nw ws (u32 bs (wordOff + 4)) (u32 bs (wordOff + 8))
        (u32 bs (wordOff + 16)) annOff cur.1 cur.2 p with
    | reject =>
      dsimp only [cur] at hst; rw [hst] at hm2; dsimp only at hm2
      refine Reaches.refl ⟨hf02, ⟨i, by rw [hf2.regs 8 (by decide)]; exact h8_1⟩, ?_⟩
      rw [walkA.eq_def]; dsimp only; erw [hst]; exact hm2
    | overflow =>
      dsimp only [cur] at hst; rw [hst] at hm2; dsimp only at hm2
      refine Reaches.refl ⟨hf02, ⟨i, by rw [hf2.regs 8 (by decide)]; exact h8_1⟩, ?_⟩
      rw [walkA.eq_def]; dsimp only; erw [hst]; exact hm2
    | fall nd na' len =>
      dsimp only [cur] at hst; rw [hst] at hm2; dsimp only at hm2
      obtain ⟨hr2, h2_2, h7_2, h15_2, rfl, hnd, hna'⟩ := hm2
      have hl2 : s2.regs.length = 16 := by rw [hf2.len, hl1]
      have hbuf2 : Buf s2.mem b bs := by rw [hf2.mem]; exact hbuf1
      have hdesc2 : Desc s2.mem D b bs.length := ⟨by rw [hf2.mem]; exact hdesc1.ptr, by rw [hf2.mem]; exact hdesc1.len⟩
      have h5_2 : s2.reg 5 = some (b + bs.length) := by rw [hf2.regs 5 (by decide)]; exact h5_1
      have h6_2 : s2.reg 6 = some (b + p) := by rw [hf2.regs 6 (by decide)]; exact h6_1
      have h8_2 : s2.reg 8 = some i := by rw [hf2.regs 8 (by decide)]; exact h8_1
      have h9_2 : s2.reg 9 = some (b + annOff) := by rw [hf2.regs 9 (by decide)]; exact h9_1
      have h12_2 : s2.reg 12 = some (b + wordOff) := by rw [hf2.regs 12 (by decide)]; exact h12_1
      have h13_2 : s2.reg 13 = some (b + ws) := by rw [hf2.regs 13 (by decide)]; exact h13_1
      xs f11 using hl2, h8_2, add_small (show i + 1 < 2 ^ 64 by omega)
      xs f12 using hl2, h12_2, ea_small (show b + wordOff + 12 < 2 ^ 64 by omega),
        show b + wordOff + 12 = b + (wordOff + 12) by omega, rd32_buf hbuf2 (show wordOff + 12 + 4 ≤ bs.length by omega)
      xs f13 using hl2
      have hc3 := (cond_sub (show i + 1 < W by rw [W_eq]; omega)
        (show u32 bs (wordOff + 12) < W by rw [W_eq]; omega)).2.1
      by_cases hlt : i + 1 < u32 bs (wordOff + 12)
      · xs f14 using hc3, show ¬ (u32 bs (wordOff + 12) ≤ i + 1) by omega
        refine (reach_frame (by frame_tac)
          ((lookup_spec pb.cLuFall (by decide) (by decide) (by decide) (by decide)
            (s := _) (idx := i + 1) ?_ hby hsm (by simp only [List.length_set, hl2]) (by rsimpg; omega)
            (by rsimpg; exact h12_2) (by rsimpg; exact h9_2) (by rsimpg using hl2)
            (by omega) hw20 hann).trans
            fun s3 ⟨hf3, hr3, h11_3, hm3⟩ => ?_)).mono fun s' ⟨hf2', hm'⟩ =>
              ⟨Frame.trans_same hf02 hf2', hm'⟩
        · exact hbuf2
        · have hl3 : s3.regs.length = 16 := by rw [hf3.len]; simp only [List.length_set, hl2]
          have hbuf3 : Buf s3.mem b bs := by rw [hf3.mem]; exact hbuf2
          have hdesc3 : Desc s3.mem D b bs.length := ⟨by rw [hf3.mem]; exact hdesc2.ptr, by rw [hf3.mem]; exact hdesc2.len⟩
          have h5_3 : s3.reg 5 = some (b + bs.length) := by rw [hf3.regs 5 (by decide)]; rsimpg; exact h5_2
          have h6_3 : s3.reg 6 = some (b + p) := by rw [hf3.regs 6 (by decide)]; rsimpg; exact h6_2
          have h8_3 : s3.reg 8 = some (i + 1) := by rw [hf3.regs 8 (by decide)]; rsimpg using hl2
          have h9_3 : s3.reg 9 = some (b + annOff) := by rw [hf3.regs 9 (by decide)]; rsimpg; exact h9_2
          have h12_3 : s3.reg 12 = some (b + wordOff) := by rw [hf3.regs 12 (by decide)]; rsimpg; exact h12_2
          have h13_3 : s3.reg 13 = some (b + ws) := by rw [hf3.regs 13 (by decide)]; rsimpg; exact h13_2
          have h2_3 : s3.reg 2 = some nd := by rw [hf3.regs 2 (by decide)]; rsimpg; exact h2_2
          have h7_3 : s3.reg 7 = some na' := by rw [hf3.regs 7 (by decide)]; rsimpg; exact h7_2
          have h15_3 : s3.reg 15 = some len := by rw [hf3.regs 15 (by decide)]; rsimpg; exact h15_2
          refine (walk_ann_ok pb (idx := i + 1) (na := u32 bs (wordOff + 16)) (annOff := annOff)
            hbuf3 hby hsm hl3 hr3 (by rw [← Nat.add_assoc]; exact h11_3)
            h2_3 h7_3 (by unfold HMAX at hnd; omega) (by unfold HMAX at hna'; omega)
            (by cases hlu : lookupA bs (i + 1) (u32 bs (wordOff + 16)) annOff with
              | some q' => rw [hlu] at hm3; exact hm3
              | none => rw [hlu] at hm3; rw [← Nat.add_assoc]; exact hm3) hann).trans
            fun s4 ⟨hf4, hm4⟩ => ?_
          have hf34 : Frame [2, 6, 7, 8, 10, 11, 14, 15] _ s4 :=
            Frame.trans_same (Frame.mono hf3 (by decide)) (Frame.mono hf4 (by decide))
          have hl4 : s4.regs.length = 16 := by rw [hf4.len, hl3]
          have hbuf4 : Buf s4.mem b bs := by rw [hf4.mem]; exact hbuf3
          have hdesc4 : Desc s4.mem D b bs.length := ⟨by rw [hf4.mem]; exact hdesc3.ptr, by rw [hf4.mem]; exact hdesc3.len⟩
          have h5_4 : s4.reg 5 = some (b + bs.length) := by rw [hf4.regs 5 (by decide)]; exact h5_3
          have h6_4 : s4.reg 6 = some (b + p) := by rw [hf4.regs 6 (by decide)]; exact h6_3
          have h8_4 : s4.reg 8 = some (i + 1) := by rw [hf4.regs 8 (by decide)]; exact h8_3
          have h9_4 : s4.reg 9 = some (b + annOff) := by rw [hf4.regs 9 (by decide)]; exact h9_3
          have h12_4 : s4.reg 12 = some (b + wordOff) := by rw [hf4.regs 12 (by decide)]; exact h12_3
          have h13_4 : s4.reg 13 = some (b + ws) := by rw [hf4.regs 13 (by decide)]; exact h13_3
          have h2_4 : s4.reg 2 = some nd := by rw [hf4.regs 2 (by decide)]; exact h2_3
          have h7_4 : s4.reg 7 = some na' := by rw [hf4.regs 7 (by decide)]; exact h7_3
          have h15_4 : s4.reg 15 = some len := by rw [hf4.regs 15 (by decide)]; exact h15_3
          cases hlu : lookupA bs (i + 1) (u32 bs (wordOff + 16)) annOff with
          | none =>
            rw [hlu] at hm4; dsimp only at hm4
            xs n10 using hm4, hl4, h6_4, h15_4, add_small (show b + p + len < 2 ^ 64 by omega)
            xs n11
            refine (walk_spec pb (b := b) (wordOff := wordOff) (annOff := annOff) (nw := nw) (ws := ws) (bs := bs)
              hby hsm hD hsw hnw hw20 hann hws hbs16
              rem (i + 1) (p + len) nd na' _ ?_ ?_ (by simp only [List.length_set, hl4]) rfl
              (by rsimpg; exact h5_4) (by rsimpg using hl4; omega) (by rsimpg; exact h8_4)
              (by rsimpg; exact h9_4) (by rsimpg; exact h12_4) (by rsimpg; exact h13_4)
              (by rsimpg; exact h2_4) (by rsimpg; exact h7_4) (by omega) hsk' hnd hna').mono
              fun s5 ⟨hf5, hv8, hm5⟩ => ?_
            { exact hbuf4 }
            { exact hdesc4 }
            refine ⟨Frame.trans_same hf34 (Frame.trans_same (by frame_tac) hf5), hv8, ?_⟩
            rw [walkA.eq_def]; dsimp only; erw [hst]; dsimp only; rw [if_pos hlt, hlu]; exact hm5
          | some q =>
            rw [hlu] at hm4
            dsimp only at hm4
            by_cases heq : u32 bs (q + 4) = nd ∧ u32 bs (q + 8) = na'
            · rw [if_pos heq] at hm4
              xs n10 using hm4, hl4, h6_4, h15_4, add_small (show b + p + len < 2 ^ 64 by omega)
              xs n11
              refine (walk_spec pb (b := b) (wordOff := wordOff) (annOff := annOff) (nw := nw) (ws := ws) (bs := bs)
                hby hsm hD hsw hnw hw20 hann hws hbs16
                rem (i + 1) (p + len) nd na' _ ?_ ?_ (by simp only [List.length_set, hl4]) rfl
                (by rsimpg; exact h5_4) (by rsimpg using hl4; omega) (by rsimpg; exact h8_4)
                (by rsimpg; exact h9_4) (by rsimpg; exact h12_4) (by rsimpg; exact h13_4)
                (by rsimpg; exact h2_4) (by rsimpg; exact h7_4) (by omega) hsk' hnd hna').mono
                fun s5 ⟨hf5, hv8, hm5⟩ => ?_
              { exact hbuf4 }
              { exact hdesc4 }
              refine ⟨Frame.trans_same hf34 (Frame.trans_same (by frame_tac) hf5), hv8, ?_⟩
              rw [walkA.eq_def]; dsimp only; erw [hst]; dsimp only; rw [if_pos hlt, hlu]; dsimp only; rw [if_pos heq]; exact hm5
            · rw [if_neg heq] at hm4
              refine Reaches.refl ⟨hf34, ⟨i + 1, h8_4⟩, ?_⟩
              rw [walkA.eq_def]; dsimp only; erw [hst]; dsimp only; rw [if_pos hlt, hlu]; dsimp only; rw [if_neg heq]; exact hm4
      · xs f14 using hc3, show u32 bs (wordOff + 12) ≤ i + 1 by omega
        refine Reaches.refl ⟨Frame.trans_same hf02 (by frame_tac), ⟨i + 1, by rsimpg using hl2⟩, ?_⟩
        rw [walkA.eq_def]; dsimp only; erw [hst]; dsimp only; rw [if_neg hlt]; try rfl
    | nofall len =>
      dsimp only [cur] at hst; rw [hst] at hm2; dsimp only at hm2
      obtain ⟨hr2, h2_2, h7_2, h15_2, rfl⟩ := hm2
      have hl2 : s2.regs.length = 16 := by rw [hf2.len, hl1]
      have hbuf2 : Buf s2.mem b bs := by rw [hf2.mem]; exact hbuf1
      have hdesc2 : Desc s2.mem D b bs.length := ⟨by rw [hf2.mem]; exact hdesc1.ptr, by rw [hf2.mem]; exact hdesc1.len⟩
      have h5_2 : s2.reg 5 = some (b + bs.length) := by rw [hf2.regs 5 (by decide)]; exact h5_1
      have h6_2 : s2.reg 6 = some (b + p) := by rw [hf2.regs 6 (by decide)]; exact h6_1
      have h8_2 : s2.reg 8 = some i := by rw [hf2.regs 8 (by decide)]; exact h8_1
      have h9_2 : s2.reg 9 = some (b + annOff) := by rw [hf2.regs 9 (by decide)]; exact h9_1
      have h12_2 : s2.reg 12 = some (b + wordOff) := by rw [hf2.regs 12 (by decide)]; exact h12_1
      have h13_2 : s2.reg 13 = some (b + ws) := by rw [hf2.regs 13 (by decide)]; exact h13_1
      refine (walk_nofall_nxt pb hbuf2 hby hsm hl2 hr2 h8_2 h9_2 h12_2 h2_2 h7_2
        (by omega) hw20 hann).trans fun s3 ⟨hf3, hr3, h8_3, h2_3, h7_3⟩ => ?_
      have hf23 : Frame [2, 6, 7, 8, 10, 11, 14, 15] _ s3 :=
        Frame.trans_same hf02 (Frame.mono hf3 (by decide))
      have hl3 : s3.regs.length = 16 := by rw [hf3.len, hl2]
      have hbuf3 : Buf s3.mem b bs := by rw [hf3.mem]; exact hbuf2
      have hdesc3 : Desc s3.mem D b bs.length := ⟨by rw [hf3.mem]; exact hdesc2.ptr, by rw [hf3.mem]; exact hdesc2.len⟩
      have h5_3 : s3.reg 5 = some (b + bs.length) := by rw [hf3.regs 5 (by decide)]; exact h5_2
      have h6_3 : s3.reg 6 = some (b + p) := by rw [hf3.regs 6 (by decide)]; exact h6_2
      have h9_3 : s3.reg 9 = some (b + annOff) := by rw [hf3.regs 9 (by decide)]; exact h9_2
      have h12_3 : s3.reg 12 = some (b + wordOff) := by rw [hf3.regs 12 (by decide)]; exact h12_2
      have h13_3 : s3.reg 13 = some (b + ws) := by rw [hf3.regs 13 (by decide)]; exact h13_2
      have h15_3 : s3.reg 15 = some len := by rw [hf3.regs 15 (by decide)]; exact h15_2
      let nxt := match lookupA bs (i + 1) (u32 bs (wordOff + 16)) annOff with
        | some q => (u32 bs (q + 4), u32 bs (q + 8)) | none => (0, 0)
      have hnxt1 : nxt.1 < HMAX := by
        dsimp only [nxt]; cases lookupA bs (i + 1) (u32 bs (wordOff + 16)) annOff with
        | none => simp only [HMAX]; omega
        | some q => have := u32_lt hby (q + 4); simp only [HMAX]; omega
      have hnxt2 : nxt.2 < HMAX := by
        dsimp only [nxt]; cases lookupA bs (i + 1) (u32 bs (wordOff + 16)) annOff with
        | none => simp only [HMAX]; omega
        | some q => have := u32_lt hby (q + 8); simp only [HMAX]; omega
      xs n10 using hr3, hl3, h6_3, h15_3, add_small (show b + p + len < 2 ^ 64 by omega)
      xs n11
      refine (walk_spec pb (b := b) (wordOff := wordOff) (annOff := annOff) (nw := nw) (ws := ws) (bs := bs)
        hby hsm hD hsw hnw hw20 hann hws hbs16
        rem (i + 1) (p + len) nxt.1 nxt.2 _ ?_ ?_ (by simp only [List.length_set, hl3]) rfl
        (by rsimpg; exact h5_3) (by rsimpg using hl3; omega) (by rsimpg; exact h8_3)
        (by rsimpg; exact h9_3) (by rsimpg; exact h12_3) (by rsimpg; exact h13_3)
        (by rsimpg; exact h2_3) (by rsimpg; exact h7_3) (by omega) hsk' hnxt1 hnxt2).mono
        fun s4 ⟨hf4, hv8, hm4⟩ => ?_
      { exact hbuf3 }
      { exact hdesc3 }
      refine ⟨Frame.trans_same hf23 (Frame.trans_same (by frame_tac) hf4), hv8, ?_⟩
      rw [walkA.eq_def]; dsimp only; erw [hst]; exact hm4

private theorem anns_loop_spec {B D R : Nat} (pb : PassBlocks B D R)
    {b wordOff annOff : Nat} {bs : List Nat}
    (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < 2 ^ 48)
    (hw20 : wordOff + 20 ≤ bs.length) (hann : annOff + 12 * u32 bs (wordOff + 16) ≤ bs.length) :
    ∀ rem q (s : St),
      Buf s.mem b bs → s.regs.length = 16 → s.rip = B + 3145 →
      s.reg 6 = some (b + q) → s.reg 7 = some (b + (annOff + 12 * u32 bs (wordOff + 16))) →
      s.reg 9 = some (b + annOff) → s.reg 12 = some (b + wordOff) →
      q + 12 * rem = annOff + 12 * u32 bs (wordOff + 16) →
      Reaches s (fun s' => Frame [2, 6, 10, 11, 12, 14, 15] s s' ∧
        if annsOKA bs (u32 bs (wordOff + 12)) (u32 bs (wordOff + 16)) annOff rem q = true then
          s'.rip = B + 560 ∧ s'.reg 12 = some (b + (annOff + 12 * u32 bs (wordOff + 16)))
        else s'.rip = B + 3301)
  | 0, q, s, hbuf, hl, hrip, h6, h7, _, _, hrem => by
    have he := pb.cEnd; simp only [CodeAt_cons, CodeAt_nil, and_true] at he
    obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, e14, e15, _⟩ := he
    have hap := pb.cAnnsPre; simp only [CodeAt_cons, CodeAt_nil, and_true] at hap
    obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, a15, a16, _⟩ := hap
    have hq : q = annOff + 12 * u32 bs (wordOff + 16) := by omega
    xs a15 using h6, h7
    have hc4 := (cond_sub (show b + (annOff + 12 * u32 bs (wordOff + 16)) < W by rw [W_eq]; omega)
      (show b + (annOff + 12 * u32 bs (wordOff + 16)) < W by rw [W_eq]; omega)).2.2.1
    simp only [decide_true] at hc4
    xs a16 using hq, hc4
    xs e14 using hl, h7
    xs e15
    rw [annsOKA.eq_def]
    exact Reaches.refl ⟨by frame_tac, rfl, by rsimpg using hl⟩
  | rem + 1, q, s, hbuf, hl, hrip, h6, h7, h9, h12, hrem => by
    have hap := pb.cAnnsPre; simp only [CodeAt_cons, CodeAt_nil, and_true] at hap
    obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, a15, a16, a17, a18, a19, a20⟩ := hap
    have he := pb.cEnd; simp only [CodeAt_cons, CodeAt_nil, and_true] at he
    obtain ⟨e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, _⟩ := he
    have hq12 : q + 12 ≤ annOff + 12 * u32 bs (wordOff + 16) := by omega
    xs a15 using h6, h7
    have hc4 := (cond_sub (show b + q < W by rw [W_eq]; omega)
      (show b + (annOff + 12 * u32 bs (wordOff + 16)) < W by rw [W_eq]; omega)).2.2.1
    xs a16 using hc4, show b + q ≠ b + (annOff + 12 * u32 bs (wordOff + 16)) by omega
    have ht := u32_lt hby q
    have hnc := u32_lt hby (wordOff + 12)
    xs a17 using hl, h6, ea_small (show b + q + 0 < 2 ^ 64 by omega),
      Nat.add_zero, rd32_buf hbuf (show q + 4 ≤ bs.length by omega)
    xs a18 using hl, h12, ea_small (show b + wordOff + 12 < 2 ^ 64 by omega),
      show b + wordOff + 12 = b + (wordOff + 12) by omega, rd32_buf hbuf (show wordOff + 12 + 4 ≤ bs.length by omega)
    xs a19 using hl
    have hc3 := (cond_sub (show u32 bs q < W by rw [W_eq]; omega)
      (show u32 bs (wordOff + 12) < W by rw [W_eq]; omega)).2.1
    rw [annsOKA.eq_def]
    dsimp only
    by_cases hlt : u32 bs q < u32 bs (wordOff + 12)
    · xs a20 using hc3, show ¬ (u32 bs (wordOff + 12) ≤ u32 bs q) by omega
      refine reach_frame (by frame_tac)
        ((lookup_spec pb.cLuAnns (by decide) (by decide) (by decide) (by decide)
          (s := _) (idx := u32 bs q) ?_ hby hsm (by simp only [List.length_set, hl]) (by rsimpg; omega)
          (by rsimpg; exact h12) (by rsimpg; exact h9) (by rsimpg using hl)
          (by omega) hw20 hann).trans fun s1 ⟨hf1, hr1, h11_1, hm1⟩ => ?_)
      · exact hbuf
      · have hl1 : s1.regs.length = 16 := by rw [hf1.len]; simp only [List.length_set, hl]
        have h6_1 : s1.reg 6 = some (b + q) := by rw [hf1.regs 6 (by decide)]; rsimpg; exact h6
        have h7_1 : s1.reg 7 = some (b + (annOff + 12 * u32 bs (wordOff + 16))) := by
          rw [hf1.regs 7 (by decide)]; rsimpg; exact h7
        have h9_1 : s1.reg 9 = some (b + annOff) := by rw [hf1.regs 9 (by decide)]; rsimpg; exact h9
        have h12_1 : s1.reg 12 = some (b + wordOff) := by rw [hf1.regs 12 (by decide)]; rsimpg; exact h12
        have hbuf1 : Buf s1.mem b bs := by rw [hf1.mem]; exact hbuf
        have hf01 : Frame [2, 6, 10, 11, 12, 14, 15] _ s1 := Frame.mono hf1 (by decide)
        rcases opt_cases (lookupA bs (u32 bs q) (u32 bs (wordOff + 16)) annOff) with hlu | ⟨q', hlu⟩
        · rw [hlu] at hm1
          simp only [hlu, hlt, reduceIte]
          xs e0 using hr1, hm1, h11_1
          have hc4e := (cond_sub (show b + annOff + 12 * u32 bs (wordOff + 16) < W by rw [W_eq]; omega)
            (show b + annOff + 12 * u32 bs (wordOff + 16) < W by rw [W_eq]; omega)).2.2.1
          simp only [decide_true] at hc4e
          xs e1 using hc4e
          exact Reaches.refl ⟨Frame.trans_same hf01 (by frame_tac), rfl⟩
        · rw [hlu] at hm1
          simp only [hlu, hlt, reduceIte]
          dsimp only at hm1 ⊢
          obtain ⟨h10_1, hq'12⟩ := hm1
          xs e0 using hr1, h10_1, h11_1
          have hc4e := (cond_sub (show b + q' < W by rw [W_eq]; omega)
            (show b + annOff + 12 * u32 bs (wordOff + 16) < W by rw [W_eq]; omega)).2.2.1
          xs e1 using hc4e, show b + q' ≠ b + annOff + 12 * u32 bs (wordOff + 16) by omega
          have hqp4 := u32_lt hby (q' + 4)
          have hq4 := u32_lt hby (q + 4)
          have hqp8 := u32_lt hby (q' + 8)
          have hq8 := u32_lt hby (q + 8)
          xs e2 using hl1, h10_1, ea_small (show b + q' + 4 < 2 ^ 64 by omega),
            show b + q' + 4 = b + (q' + 4) by omega, rd32_buf hbuf1 (show q' + 4 + 4 ≤ bs.length by omega)
          xs e3 using hl1, h6_1, ea_small (show b + q + 4 < 2 ^ 64 by omega),
            show b + q + 4 = b + (q + 4) by omega, rd32_buf hbuf1 (show q + 4 + 4 ≤ bs.length by omega)
          xs e4 using hl1
          by_cases h4eq : u32 bs (q' + 4) = u32 bs (q + 4)
          · have hc4a := (cond_sub (show u32 bs (q + 4) < W by rw [W_eq]; omega)
              (show u32 bs (q + 4) < W by rw [W_eq]; omega)).2.2.1
            simp only [decide_true] at hc4a
            xs e5 using h4eq, hc4a
            xs e7 using hl1, h10_1, ea_small (show b + q' + 8 < 2 ^ 64 by omega),
              show b + q' + 8 = b + (q' + 8) by omega, rd32_buf hbuf1 (show q' + 8 + 4 ≤ bs.length by omega)
            xs e8 using hl1, h6_1, ea_small (show b + q + 8 < 2 ^ 64 by omega),
              show b + q + 8 = b + (q + 8) by omega, rd32_buf hbuf1 (show q + 8 + 4 ≤ bs.length by omega)
            xs e9 using hl1
            by_cases h8eq : u32 bs (q' + 8) = u32 bs (q + 8)
            · have hc4b := (cond_sub (show u32 bs (q + 8) < W by rw [W_eq]; omega)
                (show u32 bs (q + 8) < W by rw [W_eq]; omega)).2.2.1
              simp only [decide_true] at hc4b
              xs e10 using h8eq, hc4b
              xs e12 using hl1, h6_1, add_small (show b + q + 12 < 2 ^ 64 by omega)
              xs e13
              simp only [beq_self_eq_true, Bool.true_and]
              refine (anns_loop_spec pb hby hsm hw20 hann rem (q + 12) _
                ?_ (by simp only [List.length_set, hl1]) rfl (by rsimpg using hl1; omega)
                (by rsimpg; exact h7_1) (by rsimpg; exact h9_1) (by rsimpg; exact h12_1)
                (by omega)).mono fun s2 ⟨hf2, hm2⟩ => ?_
              · exact hbuf1
              · exact ⟨Frame.trans_same hf01 (Frame.trans_same (by frame_tac) hf2), hm2⟩
            · have hc4b := (cond_sub (show u32 bs (q' + 8) < W by rw [W_eq]; omega)
                (show u32 bs (q + 8) < W by rw [W_eq]; omega)).2.2.1
              xs e10 using hc4b, h8eq
              xs e11
              simp only [beq_eq_false_iff_ne.mpr h8eq, Bool.and_false, Bool.false_and, Bool.false_eq_true, reduceIte]
              exact Reaches.refl ⟨Frame.trans_same hf01 (by frame_tac), rfl⟩
          · have hc4a := (cond_sub (show u32 bs (q' + 4) < W by rw [W_eq]; omega)
              (show u32 bs (q + 4) < W by rw [W_eq]; omega)).2.2.1
            xs e5 using hc4a, h4eq
            xs e6
            simp only [beq_eq_false_iff_ne.mpr h4eq, Bool.false_and, Bool.false_eq_true, reduceIte]
            exact Reaches.refl ⟨Frame.trans_same hf01 (by frame_tac), rfl⟩
    · xs a20 using hc3, show u32 bs (wordOff + 12) ≤ u32 bs q by omega
      simp only [hlt, reduceIte]
      exact Reaches.refl ⟨by frame_tac, rfl⟩

private theorem anns_spec {B D R : Nat} (pb : PassBlocks B D R)
    {b wordOff annOff : Nat} {bs : List Nat} {s : St}
    (hbuf : Buf s.mem b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < 2 ^ 48)
    (hl : s.regs.length = 16) (hrip : s.rip = B + 3120)
    (h9 : s.reg 9 = some (b + annOff)) (h12 : s.reg 12 = some (b + wordOff))
    (hw20 : wordOff + 20 ≤ bs.length) (hann : annOff + 12 * u32 bs (wordOff + 16) ≤ bs.length) :
    Reaches s (fun s' => Frame [2, 6, 7, 10, 11, 12, 14, 15] s s' ∧
      if annsOKA bs (u32 bs (wordOff + 12)) (u32 bs (wordOff + 16)) annOff
          (u32 bs (wordOff + 16)) annOff = true then
        s'.rip = B + 560 ∧ s'.reg 12 = some (b + (annOff + 12 * u32 bs (wordOff + 16))) ∧
        s'.reg 7 = some (b + (annOff + 12 * u32 bs (wordOff + 16)))
      else s'.rip = B + 3301) := by
  have hap := pb.cAnnsPre; simp only [CodeAt_cons, CodeAt_nil, and_true] at hap
  obtain ⟨_, _, _, _, _, _, _, _, a8, a9, a10, a11, a12, a13, a14, _⟩ := hap
  have hna := u32_lt hby (wordOff + 16)
  xs a8 using hl, h12, ea_small (show b + wordOff + 16 < 2 ^ 64 by omega),
    show b + wordOff + 16 = b + (wordOff + 16) by omega, rd32_buf hbuf (show wordOff + 16 + 4 ≤ bs.length by omega)
  xs a9 using hl
  xs a10 using hl, show u32 bs (wordOff + 16) * 8 % 18446744073709551616 = u32 bs (wordOff + 16) * 8 by omega
  xs a11 using hl, show u32 bs (wordOff + 16) * 4 % 18446744073709551616 = u32 bs (wordOff + 16) * 4 by omega
  xs a12 using hl, add_small (show u32 bs (wordOff + 16) * 8 + u32 bs (wordOff + 16) * 4 < 2 ^ 64 by omega)
  xs a13 using hl, h9, add_small (show u32 bs (wordOff + 16) * 8 + u32 bs (wordOff + 16) * 4 + (b + annOff) < 2 ^ 64 by omega)
  xs a14 using hl, h9
  have h7eq : u32 bs (wordOff + 16) * 8 + u32 bs (wordOff + 16) * 4 + (b + annOff) =
      b + (annOff + 12 * u32 bs (wordOff + 16)) := by omega
  refine reach_frame (by frame_tac)
    ((anns_loop_spec pb hby hsm hw20 hann (u32 bs (wordOff + 16)) annOff _
      ?_ (by simp only [List.length_set, hl]) (by rsimpg; rw [hrip]) (by rsimpg using hl)
      (by rsimpg using hl; exact h7eq) (by rsimpg; exact h9) (by rsimpg; exact h12) rfl).trans
      fun s1 ⟨hf1, hm1⟩ => ?_)
  · exact hbuf
  · have hf01 : Frame [2, 6, 7, 10, 11, 12, 14, 15] _ s1 := Frame.mono hf1 (by decide)
    refine Reaches.refl ⟨hf01, ?_⟩
    split at hm1
    · next hok =>
      rw [if_pos hok]
      refine ⟨hm1.1, hm1.2, ?_⟩
      rw [hf1.regs 7 (by decide)]
      rsimpg using hl; exact h7eq
    · next hok =>
      rw [if_neg hok]
      exact hm1

private theorem checkWord_spec {B D R : Nat} (pb : PassBlocks B D R)
    {b nw ws wordOff nextOff v7 v8 : Nat} {bs : List Nat} {s : St}
    (hbuf : Buf s.mem b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < 2 ^ 48)
    (hdesc : Desc s.mem D b bs.length) (hD : D + 8 < 2 ^ 64)
    (hsw : skipWordsA bs nw ws = some bs.length) (hnw : nw = u32 bs 8)
    (hsw1 : skipWordsA bs 1 wordOff = some nextOff)
    (hws : ws ≤ bs.length) (hbs16 : 16 ≤ bs.length)
    (hl : s.regs.length = 16) (hrip : s.rip = B + 569)
    (h5 : s.reg 5 = some (b + bs.length)) (h12 : s.reg 12 = some (b + wordOff))
    (h13 : s.reg 13 = some (b + ws)) (h7 : s.reg 7 = some v7) (h8 : s.reg 8 = some v8) :
    Reaches s (fun s' => Frame [2, 6, 7, 8, 9, 10, 11, 12, 14, 15] s s' ∧
      match checkWordA bs nw ws wordOff with
      | .ok => s'.rip = B + 560 ∧ s'.reg 12 = some (b + nextOff) ∧
          (∃ v7', s'.reg 7 = some v7') ∧ (∃ v8', s'.reg 8 = some v8')
      | .overflow => s'.rip = B + 3286 ∧ (∃ v8', s'.reg 8 = some v8')
      | .reject => s'.rip = B + 3301 ∧ (∃ v8', s'.reg 8 = some v8')) := by
  have he : Env s b bs := ⟨hbuf, hby, hsm, hl, h5⟩
  have hm3 := pb.cMid3; simp only [CodeAt_cons, CodeAt_nil, and_true] at hm3
  obtain ⟨_, _, _, _, _, _, _, m7, m8, m9, m10⟩ := hm3
  have hxr := pb.cXor; simp only [CodeAt_cons, CodeAt_nil, and_true] at hxr
  have hw0 := pb.cWalk0; simp only [CodeAt_cons, CodeAt_nil, and_true] at hw0
  obtain ⟨w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, _⟩ := hw0
  obtain ⟨hw20, annOff, hsc, hann, hnext⟩ := skipWordsA_one hsw1
  have hnc := u32_lt hby (wordOff + 12)
  have hna := u32_lt hby (wordOff + 16)
  have hins := u32_lt hby wordOff
  xs m7 using hl, h12, ea_small (show b + wordOff + 12 < 2 ^ 64 by omega),
    show b + wordOff + 12 = b + (wordOff + 12) by omega, rd32_buf hbuf (show wordOff + 12 + 4 ≤ bs.length by omega)
  xs m8 using hl
  rw [checkWordA.eq_def]
  dsimp only
  by_cases hnc0 : u32 bs (wordOff + 12) = 0
  · xs m9 using hnc0, cond_test_e
    exact Reaches.refl ⟨by frame_tac, rfl, ⟨v8, by rsimpg; exact h8⟩⟩
  · xs m9 using hnc0, cond_test_e
    xs m10 using hl, h12, ea_small (show b + wordOff + 20 < 2 ^ 64 by omega)
    refine reach_frame (by frame_tac)
      ((skipcode_spec pb.cSc (by decide) (by decide) (by decide) (by decide)
        (u32 bs (wordOff + 12)) (wordOff + 20) _ (he.frame (S := [11, 9]) (by frame_tac) (by decide))
        (by rsimpg; rw [hrip]) (by rsimpg using hl; omega) (by rsimpg using hl) hw20 (by omega)).trans
        fun s1 ⟨hf1, hm1⟩ => ?_)
    rw [hsc] at hm1 ⊢
    dsimp only
    obtain ⟨hr1, h9_1, h11_1⟩ := hm1
    have hl1 : s1.regs.length = 16 := by rw [hf1.len]; simp only [List.length_set, hl]
    have hbuf1 : Buf s1.mem b bs := by rw [hf1.mem]; exact hbuf
    have hdesc1 : Desc s1.mem D b bs.length := ⟨by rw [hf1.mem]; exact hdesc.ptr, by rw [hf1.mem]; exact hdesc.len⟩
    have h5_1 : s1.reg 5 = some (b + bs.length) := by rw [hf1.regs 5 (by decide)]; rsimpg; exact h5
    have h12_1 : s1.reg 12 = some (b + wordOff) := by rw [hf1.regs 12 (by decide)]; rsimpg; exact h12
    have h13_1 : s1.reg 13 = some (b + ws) := by rw [hf1.regs 13 (by decide)]; rsimpg; exact h13
    have h7_1 : s1.reg 7 = some v7 := by rw [hf1.regs 7 (by decide)]; rsimpg; exact h7
    have h8_1 : s1.reg 8 = some v8 := by rw [hf1.regs 8 (by decide)]; rsimpg; exact h8
    have hf01 : Frame [2, 6, 7, 8, 9, 10, 11, 12, 14, 15] _ s1 := Frame.mono hf1 (by decide)
    xs hxr using hr1, h8_1, alu_xor_self
    refine reach_frame (Frame.trans_same hf01 (by frame_tac))
      ((lookup_spec pb.cLu0 (by decide) (by decide) (by decide) (by decide)
        (s := _) (idx := 0) ?_ hby hsm (by simp only [List.length_set, hl1]) rfl
        (by rsimpg; exact h12_1) (by rsimpg; exact h9_1) (by rsimpg using hl1)
        (by decide) hw20 hann).trans fun s2 ⟨hf2, hr2, h11_2, hm2⟩ => ?_)
    · exact hbuf1
    · have hl2 : s2.regs.length = 16 := by rw [hf2.len]; simp only [List.length_set, hl1]
      have hbuf2 : Buf s2.mem b bs := by rw [hf2.mem]; exact hbuf1
      have hdesc2 : Desc s2.mem D b bs.length := ⟨by rw [hf2.mem]; exact hdesc1.ptr, by rw [hf2.mem]; exact hdesc1.len⟩
      have h5_2 : s2.reg 5 = some (b + bs.length) := by rw [hf2.regs 5 (by decide)]; rsimpg; exact h5_1
      have h8_2 : s2.reg 8 = some 0 := by rw [hf2.regs 8 (by decide)]; rsimpg using hl1
      have h9_2 : s2.reg 9 = some (b + annOff) := by rw [hf2.regs 9 (by decide)]; rsimpg; exact h9_1
      have h12_2 : s2.reg 12 = some (b + wordOff) := by rw [hf2.regs 12 (by decide)]; rsimpg; exact h12_1
      have h13_2 : s2.reg 13 = some (b + ws) := by rw [hf2.regs 13 (by decide)]; rsimpg; exact h13_1
      have h7_2 : s2.reg 7 = some v7 := by rw [hf2.regs 7 (by decide)]; rsimpg; exact h7_1
      have hf02 : Frame [2, 6, 7, 8, 9, 10, 11, 12, 14, 15] _ s2 := Frame.mono hf2 (by decide)
      have hinit : Reaches s2 (fun s3 => Frame [2, 14, 15] s2 s3 ∧
          let initOK := match lookupA bs 0 (u32 bs (wordOff + 16)) annOff with
            | none => true
            | some q => u32 bs (q + 4) == u32 bs wordOff && u32 bs (q + 8) == 0
          if (!initOK) = true then s3.rip = B + 3301 else s3.rip = B + 784) := by
        cases hlu : lookupA bs 0 (u32 bs (wordOff + 16)) annOff with
        | none =>
          rw [hlu] at hm2
          xs w0 using hr2, hm2, h11_2
          have hc4 := (cond_sub (show b + annOff + 12 * u32 bs (wordOff + 16) < W by rw [W_eq]; omega)
            (show b + annOff + 12 * u32 bs (wordOff + 16) < W by rw [W_eq]; omega)).2.2.1
          simp only [decide_true] at hc4
          xs w1 using hc4
          exact Reaches.refl ⟨by frame_tac, rfl⟩
        | some q =>
          rw [hlu] at hm2
          obtain ⟨h10_2, hq12⟩ := hm2
          xs w0 using hr2, h10_2, h11_2
          have hc4 := (cond_sub (show b + q < W by rw [W_eq]; omega)
            (show b + annOff + 12 * u32 bs (wordOff + 16) < W by rw [W_eq]; omega)).2.2.1
          xs w1 using hc4, show b + q ≠ b + annOff + 12 * u32 bs (wordOff + 16) by omega
          have hq4 := u32_lt hby (q + 4)
          have hq8 := u32_lt hby (q + 8)
          xs w2 using hl2, h10_2, ea_small (show b + q + 4 < 2 ^ 64 by omega),
            show b + q + 4 = b + (q + 4) by omega, rd32_buf hbuf2 (show q + 4 + 4 ≤ bs.length by omega)
          xs w3 using hl2, h12_2, ea_small (show b + wordOff + 0 < 2 ^ 64 by omega),
            Nat.add_zero, rd32_buf hbuf2 (show wordOff + 4 ≤ bs.length by omega)
          xs w4 using hl2
          by_cases heq4 : u32 bs (q + 4) = u32 bs wordOff
          · have hc4a := (cond_sub (show u32 bs wordOff < W by rw [W_eq]; omega)
              (show u32 bs wordOff < W by rw [W_eq]; omega)).2.2.1
            simp only [decide_true] at hc4a
            xs w5 using heq4, hc4a
            xs w7 using hl2, h10_2, ea_small (show b + q + 8 < 2 ^ 64 by omega),
              show b + q + 8 = b + (q + 8) by omega, rd32_buf hbuf2 (show q + 8 + 4 ≤ bs.length by omega)
            xs w8 using hl2
            by_cases heq8 : u32 bs (q + 8) = 0
            · xs w9 using heq8, cond_test_e
              exact Reaches.refl ⟨by frame_tac, by simp only [beq_self_eq_true, Bool.and_self, Bool.not_true, Bool.false_eq_true, reduceIte]⟩
            · xs w9 using heq8, cond_test_e
              xs w10
              exact Reaches.refl ⟨by frame_tac, by simp only [beq_self_eq_true, beq_eq_false_iff_ne.mpr heq8, Bool.true_and, Bool.not_false, reduceIte]⟩
          · have hc4a := (cond_sub (show u32 bs (q + 4) < W by rw [W_eq]; omega)
              (show u32 bs wordOff < W by rw [W_eq]; omega)).2.2.1
            xs w5 using hc4a, heq4
            xs w6
            exact Reaches.refl ⟨by frame_tac, by simp only [beq_eq_false_iff_ne.mpr heq4, Bool.false_and, Bool.not_false, reduceIte]⟩
      refine hinit.trans fun s3 ⟨hf3, hm3⟩ => ?_
      have hf03 : Frame [2, 6, 7, 8, 9, 10, 11, 12, 14, 15] _ s3 :=
        Frame.trans_same hf02 (Frame.mono hf3 (by decide))
      have hl3 : s3.regs.length = 16 := by rw [hf3.len, hl2]
      have hbuf3 : Buf s3.mem b bs := by rw [hf3.mem]; exact hbuf2
      have hdesc3 : Desc s3.mem D b bs.length := ⟨by rw [hf3.mem]; exact hdesc2.ptr, by rw [hf3.mem]; exact hdesc2.len⟩
      have h5_3 : s3.reg 5 = some (b + bs.length) := by rw [hf3.regs 5 (by decide)]; exact h5_2
      have h8_3 : s3.reg 8 = some 0 := by rw [hf3.regs 8 (by decide)]; exact h8_2
      have h9_3 : s3.reg 9 = some (b + annOff) := by rw [hf3.regs 9 (by decide)]; exact h9_2
      have h12_3 : s3.reg 12 = some (b + wordOff) := by rw [hf3.regs 12 (by decide)]; exact h12_2
      have h13_3 : s3.reg 13 = some (b + ws) := by rw [hf3.regs 13 (by decide)]; exact h13_2
      have h7_3 : s3.reg 7 = some v7 := by rw [hf3.regs 7 (by decide)]; exact h7_2
      have hbranch : (s3.rip = B + 784 ∧ (lookupA bs 0 (u32 bs (wordOff + 16)) annOff = none ∨
            ∃ q, lookupA bs 0 (u32 bs (wordOff + 16)) annOff = some q ∧
              (u32 bs (q + 4) == u32 bs wordOff && u32 bs (q + 8) == 0) = true)) ∨
          (s3.rip = B + 3301 ∧ ∃ q, lookupA bs 0 (u32 bs (wordOff + 16)) annOff = some q ∧
            (u32 bs (q + 4) == u32 bs wordOff && u32 bs (q + 8) == 0) = false) := by
        dsimp only at hm3
        rcases opt_cases (lookupA bs 0 (u32 bs (wordOff + 16)) annOff) with hlu0 | ⟨q, hlu0⟩
        · rw [hlu0] at hm3; simp only [Bool.not_true, Bool.false_eq_true, reduceIte] at hm3
          exact Or.inl ⟨hm3, Or.inl hlu0⟩
        · rw [hlu0] at hm3; dsimp only at hm3
          by_cases hb : (u32 bs (q + 4) == u32 bs wordOff && u32 bs (q + 8) == 0) = true
          · simp only [hb, Bool.not_true, Bool.false_eq_true, reduceIte] at hm3
            exact Or.inl ⟨hm3, Or.inr ⟨q, hlu0, hb⟩⟩
          · have hbf : (u32 bs (q + 4) == u32 bs wordOff && u32 bs (q + 8) == 0) = false :=
              Bool.eq_false_iff.mpr hb
            simp only [hbf, Bool.not_false, reduceIte] at hm3
            exact Or.inr ⟨hm3, q, hlu0, hbf⟩
      rcases hbranch with ⟨hm3, hok0⟩ | ⟨hm3, q, hq, hb⟩
      · xs w11 using hm3, hl3, h12_3, ea_small (show b + wordOff + 0 < 2 ^ 64 by omega),
          Nat.add_zero, rd32_buf hbuf3 (show wordOff + 4 ≤ bs.length by omega)
        xs w12 using hl3, h7_3, alu_xor_self
        xs w13 using hl3, h12_3, ea_small (show b + wordOff + 20 < 2 ^ 64 by omega)
        refine reach_frame (Frame.trans_same hf03 (by frame_tac))
          ((walk_spec pb (b := b) (wordOff := wordOff) (annOff := annOff) (nw := nw) (ws := ws) (bs := bs)
            hby hsm hD hsw hnw hw20 hann hws hbs16
            (u32 bs (wordOff + 12)) 0 (wordOff + 20) (u32 bs wordOff) 0 _
            ?_ ?_ (by simp only [List.length_set, hl3]) (by dsimp only [rip_mk]; try omega)
            (by rsimpg; exact h5_3) (by rsimpg using hl3; omega) (by rsimpg; exact h8_3)
            (by rsimpg; exact h9_3) (by rsimpg; exact h12_3) (by rsimpg; exact h13_3)
            (by rsimpg using hl3) (by rsimpg using hl3) (by omega) hsc
            (by unfold HMAX; omega) (by unfold HMAX; decide)).trans
            fun s4 ⟨hf4, h8_ex, hm4⟩ => ?_)
        { exact hbuf3 }
        { exact hdesc3 }
        have hf34 : Frame [2, 6, 7, 8, 9, 10, 11, 12, 14, 15] _ s4 := Frame.mono hf4 (by decide)
        dsimp only at hm4
        cases hwk : walkA bs nw ws (u32 bs (wordOff + 4)) (u32 bs (wordOff + 8))
            (u32 bs (wordOff + 12)) (u32 bs (wordOff + 16)) annOff
            (u32 bs (wordOff + 12)) 0 (wordOff + 20) (u32 bs wordOff) 0 with
        | reject =>
          rw [hwk] at hm4
          refine Reaches.refl ⟨hf34, ?_⟩
          rcases hok0 with hnone | ⟨q, hq, hb⟩
          · simp only [hnone, Bool.not_true, Bool.false_eq_true, reduceIte]; exact ⟨hm4, h8_ex⟩
          · simp only [hq, hb, Bool.not_true, Bool.false_eq_true, reduceIte]; exact ⟨hm4, h8_ex⟩
        | overflow =>
          rw [hwk] at hm4
          refine Reaches.refl ⟨hf34, ?_⟩
          rcases hok0 with hnone | ⟨q, hq, hb⟩
          · simp only [hnone, Bool.not_true, Bool.false_eq_true, reduceIte]; exact ⟨hm4, h8_ex⟩
          · simp only [hq, hb, Bool.not_true, Bool.false_eq_true, reduceIte]; exact ⟨hm4, h8_ex⟩
        | ok =>
          rw [hwk] at hm4
          have hl4 : s4.regs.length = 16 := by rw [hf4.len]; simp only [List.length_set, hl3]
          have hbuf4 : Buf s4.mem b bs := by rw [hf4.mem]; exact hbuf3
          have h9_4 : s4.reg 9 = some (b + annOff) := by rw [hf4.regs 9 (by decide)]; rsimpg; exact h9_3
          have h12_4 : s4.reg 12 = some (b + wordOff) := by rw [hf4.regs 12 (by decide)]; rsimpg; exact h12_3
          refine (anns_spec pb hbuf4 hby hsm hl4 hm4 h9_4 h12_4 hw20 hann).mono
            fun s5 ⟨hf5, hm5⟩ => ?_
          have hf45 := Frame.trans_same hf34 (Frame.mono hf5 (by decide))
          have h8_5 : ∃ v8', s5.reg 8 = some v8' := by
            obtain ⟨v8', hv8⟩ := h8_ex; exact ⟨v8', by rw [hf5.regs 8 (by decide)]; exact hv8⟩
          refine ⟨hf45, ?_⟩
          split at hm5
          · next hok =>
            have hmain : s5.rip = B + 560 ∧ s5.reg 12 = some (b + nextOff) ∧
                (∃ v7', s5.reg 7 = some v7') ∧ ∃ v8', s5.reg 8 = some v8' := by
              refine ⟨hm5.1, ?_, ⟨_, hm5.2.2⟩, h8_5⟩; rw [hm5.2.1, hnext]
            rcases hok0 with hnone | ⟨q, hq, hb⟩
            · simp only [hnone, Bool.not_true, Bool.false_eq_true, reduceIte, hok]; exact hmain
            · simp only [hq, hb, Bool.not_true, Bool.false_eq_true, reduceIte, hok]; exact hmain
          · next hok =>
            rcases hok0 with hnone | ⟨q, hq, hb⟩
            · simp only [hnone, Bool.not_true, Bool.false_eq_true, reduceIte, hok]; exact ⟨hm5, h8_5⟩
            · simp only [hq, hb, Bool.not_true, Bool.false_eq_true, reduceIte, hok]; exact ⟨hm5, h8_5⟩
      · refine Reaches.refl ⟨hf03, ?_⟩
        simp only [hq, hb, Bool.not_false, reduceIte]
        exact ⟨hm3, 0, h8_3⟩

theorem checkWords_spec {B D R : Nat} (pb : PassBlocks B D R)
    {b nw ws : Nat} {bs : List Nat}
    (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < 2 ^ 48)
    (hD : D + 8 < 2 ^ 64) (hsw : skipWordsA bs nw ws = some bs.length) (hnw : nw = u32 bs 8)
    (hws : ws ≤ bs.length) (hbs16 : 16 ≤ bs.length) :
    ∀ rem wordOff v7 v8 (s : St),
      Buf s.mem b bs → Desc s.mem D b bs.length →
      skipWordsA bs rem wordOff = some bs.length → wordOff ≤ bs.length →
      s.regs.length = 16 → s.rip = B + 560 →
      s.reg 5 = some (b + bs.length) → s.reg 12 = some (b + wordOff) →
      s.reg 13 = some (b + ws) → s.reg 7 = some v7 → s.reg 8 = some v8 →
      Reaches s (fun s' => Frame [2, 6, 7, 8, 9, 10, 11, 12, 14, 15] s s' ∧
        let res := checkWordsA bs nw ws rem wordOff
        s'.rip = checkResRip B res ∧ ∃ v8', s'.reg 8 = some v8')
  | 0, wordOff, v7, v8, s, hbuf, _, hsw_rem, hwo_le, hl, hrip, h5, h12, _, _, h8 => by
    have hm3 := pb.cMid3; simp only [CodeAt_cons, CodeAt_nil, and_true] at hm3
    obtain ⟨_, _, _, _, _, m5, m6, _⟩ := hm3
    have hwo_eq : wordOff = bs.length := (skipWordsA_end_zero hwo_le hsw_rem).2 rfl
    xs m5 using h12, h5
    have hc4 := (cond_sub (show b + bs.length < W by rw [W_eq]; omega)
      (show b + bs.length < W by rw [W_eq]; omega)).2.2.1
    simp only [decide_true] at hc4
    xs m6 using hwo_eq, hc4
    rw [checkWordsA.eq_def]
    exact Reaches.refl ⟨by frame_tac, rfl, v8, h8⟩
  | rem + 1, wordOff, v7, v8, s, hbuf, hdesc, hsw_rem, hwo_le, hl, hrip, h5, h12, h13, h7, h8 => by
    have hm3 := pb.cMid3; simp only [CodeAt_cons, CodeAt_nil, and_true] at hm3
    obtain ⟨_, _, _, _, _, m5, m6, _⟩ := hm3
    have hwo_ne : wordOff ≠ bs.length := by
      intro heq; have hzero := (skipWordsA_end_zero hwo_le hsw_rem).1 heq; omega
    xs m5 using h12, h5
    have hc4 := (cond_sub (show b + wordOff < W by rw [W_eq]; omega)
      (show b + bs.length < W by rw [W_eq]; omega)).2.2.1
    xs m6 using hc4, decide_eq_false (show b + wordOff ≠ b + bs.length by omega)
    rw [checkWordsA.eq_def]
    dsimp only
    have hadd : skipWordsA bs (rem + 1) wordOff = (skipWordsA bs 1 wordOff).bind (skipWordsA bs rem) := by
      rw [show rem + 1 = 1 + rem by omega]; exact skipWordsA_add 1 rem wordOff
    rw [hadd] at hsw_rem
    cases hsw1 : skipWordsA bs 1 wordOff with
    | none => simp [hsw1] at hsw_rem
    | some nextOff =>
      simp only [hsw1, Option.bind_some] at hsw_rem
      have ⟨_, hnext_le⟩ := skipWordsA_bounds hwo_le hsw1
      refine reach_frame (by frame_tac)
        ((checkWord_spec pb (b := b) (nw := nw) (ws := ws) (wordOff := wordOff) (nextOff := nextOff)
          (v7 := v7) (v8 := v8) (bs := bs) (s := _)
          ?_ hby hsm ?_ hD hsw hnw hsw1 hws hbs16 ?_ (by dsimp only [rip_mk]; rw [hrip]; try omega) ?_ ?_ ?_ ?_ ?_).trans
          fun s1 ⟨hf1, hm1⟩ => ?_)
      { exact hbuf }
      { exact hdesc }
      { exact hl }
      { exact h5 }
      { exact h12 }
      { exact h13 }
      { exact h7 }
      { exact h8 }
      cases hcw : checkWordA bs nw ws wordOff with
      | reject => rw [hcw] at hm1; exact Reaches.refl ⟨hf1, hm1.1, hm1.2⟩
      | overflow => rw [hcw] at hm1; exact Reaches.refl ⟨hf1, hm1.1, hm1.2⟩
      | ok =>
        rw [hcw] at hm1
        obtain ⟨hr1, h12_1, ⟨v7', h7_1⟩, ⟨v8', h8_1⟩⟩ := hm1
        have hl1 : s1.regs.length = 16 := by rw [hf1.len, hl]
        have hbuf1 : Buf s1.mem b bs := by rw [hf1.mem]; exact hbuf
        have hdesc1 : Desc s1.mem D b bs.length := ⟨by rw [hf1.mem]; exact hdesc.ptr, by rw [hf1.mem]; exact hdesc.len⟩
        have h5_1 : s1.reg 5 = some (b + bs.length) := by rw [hf1.regs 5 (by decide)]; exact h5
        have h13_1 : s1.reg 13 = some (b + ws) := by rw [hf1.regs 13 (by decide)]; exact h13
        refine (checkWords_spec pb hby hsm hD hsw hnw hws hbs16 rem nextOff v7' v8' s1
          hbuf1 hdesc1 hsw_rem hnext_le hl1 hr1 h5_1 h12_1 h13_1 h7_1 h8_1).mono
          fun s2 ⟨hf2, hm2⟩ => ⟨Frame.trans_same hf1 hf2, hm2⟩

private theorem end_res_spec {B D R : Nat} (pb : PassBlocks B D R)
    {s : St} (r : CheckResA) (hl : s.regs.length = 16)
    (hR : R < 4) (hrip : s.rip = checkResRip B r) :
    Reaches s (fun s' => Frame [R] s s' ∧ s'.rip = B + passSize ∧
      s'.reg R = some (checkResVal r)) := by
  have he := pb.cEnd; simp only [CodeAt_cons, CodeAt_nil, and_true] at he
  obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, e16, e17, e18, e19, e20, e21⟩ := he
  cases r with
  | ok =>
    unfold checkResRip at hrip; dsimp only at hrip
    xs e16 using show ACCEPT % 18446744073709551616 = ACCEPT by decide
    xs e17
    refine Reaches.refl ⟨Frame.of_mk ((RegsFrame.refl _ _).set (by simp) _) (by simp only [List.length_set]),
      rfl, ?_⟩
    exact reg_mk_set_eq s.regs _ s.fl s.mem s.k (by rw [hl]; omega)
  | overflow =>
    unfold checkResRip at hrip; dsimp only at hrip
    xs e18 using show OVERFLOW % 18446744073709551616 = OVERFLOW by decide
    xs e19
    refine Reaches.refl ⟨Frame.of_mk ((RegsFrame.refl _ _).set (by simp) _) (by simp only [List.length_set]),
      rfl, ?_⟩
    exact reg_mk_set_eq s.regs _ s.fl s.mem s.k (by rw [hl]; omega)
  | reject =>
    unfold checkResRip at hrip; dsimp only at hrip
    xs e20 using show REJECT % 18446744073709551616 = REJECT by decide
    xs e21
    refine Reaches.refl ⟨Frame.of_mk ((RegsFrame.refl _ _).set (by simp) _) (by simp only [List.length_set]),
      rfl, ?_⟩
    exact reg_mk_set_eq s.regs _ s.fl s.mem s.k (by rw [hl]; omega)

theorem skipDataA_bounds {bs : List Nat} : ∀ {n pos pos' : Nat},
    pos ≤ bs.length → skipDataA bs n pos = some pos' → pos ≤ pos' ∧ pos' ≤ bs.length
  | 0, pos, pos', hp, h => by
    rw [skipDataA.eq_def] at h; simp only [Option.some.injEq] at h; subst h; exact ⟨Nat.le_refl _, hp⟩
  | n + 1, pos, pos', hp, h => by
    rw [skipDataA.eq_def] at h; dsimp only at h
    split at h; · simp at h
    split at h; · simp at h
    have ⟨h1, h2⟩ := skipDataA_bounds (n := n) (by omega) h
    exact ⟨by omega, h2⟩

theorem parse_spec {B D R : Nat} (pb : PassBlocks B D R)
    {b : Nat} {bs : List Nat} {s : St}
    (hbuf : Buf s.mem b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < 2 ^ 48)
    (hdesc : Desc s.mem D b bs.length) (hD : D + 8 < 2 ^ 64)
    (hl : s.regs.length = 16) (hrip : s.rip = B) :
    Reaches s (fun s' => Frame [2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15] s s' ∧
      match parseA bs with
      | none => s'.rip = B + 3301
      | some (main, nw, ws) =>
        if main < nw then
          match skipWordsA bs main ws with
          | none => s'.rip = B + 3301
          | some mainOff =>
            if u32 bs mainOff = 0 then
              s'.rip = B + 560 ∧ s'.reg 5 = some (b + bs.length) ∧
              s'.reg 7 = some b ∧ s'.reg 12 = some (b + ws) ∧
              s'.reg 13 = some (b + ws) ∧
              skipWordsA bs nw ws = some bs.length ∧ nw = u32 bs 8 ∧
              ws ≤ bs.length ∧ 16 ≤ bs.length
            else s'.rip = B + 3301
        else s'.rip = B + 3301) := by
  have hhdr := pb.cHdr; simp only [CodeAt_cons, CodeAt_nil, and_true] at hhdr
  obtain ⟨h0, h6, h9, h13, h16, h19, h23, h29, h31, h38, h40, h45, h49, h52, h56⟩ := hhdr
  have hm1 := pb.cMid1; simp only [CodeAt_cons, CodeAt_nil, and_true] at hm1
  obtain ⟨m111, m114⟩ := hm1
  have hm2 := pb.cMid2; simp only [CodeAt_cons, CodeAt_nil, and_true] at hm2
  obtain ⟨m318, m321, m323, m328, m331, m337, m340⟩ := hm2
  have hm3 := pb.cMid3; simp only [CodeAt_cons, CodeAt_nil, and_true] at hm3
  obtain ⟨m544, m547, m550, m552, m557, _⟩ := hm3
  xs h0 using show D % 18446744073709551616 = D from Nat.mod_eq_of_lt (by omega)
  xs h6 using hl, hdesc.ptr
  xs h9 using hl, hdesc.len
  xs h13 using hl
  xs h16 using hl, add_small (show b + bs.length < 2 ^ 64 by omega)
  xs h19 using hl
  have hc2 := (cond_sub (show bs.length < W by rw [W_eq]; omega)
    (show 16 < W by rw [W_eq]; decide)).1
  rw [parseA.eq_def]
  by_cases hlt16 : bs.length < 16
  · xs h23 using hc2, hlt16
    exact Reaches.refl ⟨by frame_tac, rfl⟩
  · xs h23 using hc2, hlt16
    have hu0 := u32_lt hby 0
    have hu4 := u32_lt hby 4
    have hu8 := u32_lt hby 8
    have hu12 := u32_lt hby 12
    xs h29 using hl, ea_small (show b + 0 < 2 ^ 64 by omega),
      (by simpa using rd32_buf hbuf (show 0 + 4 ≤ bs.length by omega))
    xs h31 using hl, show 827478099 % 18446744073709551616 = 827478099 by decide
    have hc4m := (cond_sub (show u32 bs 0 < W by rw [W_eq]; omega)
      (show 827478099 < W by rw [W_eq]; decide)).2.2.1
    by_cases hmag : u32 bs 0 = 827478099
    · have hc4m' : X86Int.cond (aluF .sub 827478099 827478099).snd 4 = some true := by
        simpa [hmag] using hc4m
      xs h38 using hmag, hc4m'
      xs h45 using hl, ea_small (show b + 4 < 2 ^ 64 by omega),
        show b + 4 = b + 4 by rfl, rd32_buf hbuf (show 8 ≤ bs.length by omega)
      xs h49 using hl, ea_small (show b + 8 < 2 ^ 64 by omega),
        show b + 8 = b + 8 by rfl, rd32_buf hbuf (show 12 ≤ bs.length by omega)
      xs h52 using hl, ea_small (show b + 12 < 2 ^ 64 by omega),
        show b + 12 = b + 12 by rfl, rd32_buf hbuf (show 16 ≤ bs.length by omega)
      xs h56 using hl, ea_small (show b + 16 < 2 ^ 64 by omega)
      refine reach_frame (by frame_tac)
        ((skipdata_spec pb.cSd (u32 bs 12) 16 (b := b) (bs := bs) _ ?_ (by dsimp only [rip_mk]; try omega) (by rsimpg using hl)
          (by rsimpg using hl) (by omega) hu12).trans fun s1 ⟨hf1, hm1⟩ => ?_)
      { exact ⟨hbuf, hby, hsm, by simp only [List.length_set, hl], by rsimpg using hl⟩ }
      cases hsd : skipDataA bs (u32 bs 12) 16 with
      | none =>
        rw [hsd] at hm1
        exact Reaches.refl ⟨Frame.mono hf1 (by decide), hm1⟩
      | some ws =>
        rw [hsd] at hm1
        obtain ⟨hr1, h10_1⟩ := hm1
        have ⟨_, hws_le⟩ := skipDataA_bounds (by omega) hsd
        have hl1 : s1.regs.length = 16 := by rw [hf1.len]; simp only [List.length_set, hl]
        have hbuf1 : Buf s1.mem b bs := by rw [hf1.mem]; exact hbuf
        have h5_1 : s1.reg 5 = some (b + bs.length) := by rw [hf1.regs 5 (by decide)]; rsimpg using hl
        have h6_1 : s1.reg 6 = some (u32 bs 8) := by rw [hf1.regs 6 (by decide)]; rsimpg using hl
        have h7_1 : s1.reg 7 = some b := by rw [hf1.regs 7 (by decide)]; rsimpg using hl
        have h12_1 : s1.reg 12 = some (u32 bs 4) := by rw [hf1.regs 12 (by decide)]; rsimpg using hl
        xs m111 using hr1, hl1, h10_1
        xs m114 using hl1, h6_1
        have hf01 : Frame [2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15] _ s1 := Frame.mono hf1 (by decide)
        refine (reach_frame (by frame_tac)
          ((skipwords_spec pb.cSw1 (u32 bs 8) ws (b := b) (bs := bs) _ ?_ (by dsimp only [rip_mk]; try omega) (by rsimpg; exact h10_1)
            (by rsimpg using hl1) hws_le hu8).trans fun s2 ⟨hf2, hm2⟩ => ?_)).mono
            fun s' ⟨hf', hm'⟩ => ⟨Frame.trans_same hf01 hf', hm'⟩
        { exact ⟨hbuf1, hby, hsm, by simp only [List.length_set, hl1], by rsimpg; exact h5_1⟩ }
        cases hsw : skipWordsA bs (u32 bs 8) ws with
        | none =>
          rw [hsw] at hm2
          exact Reaches.refl ⟨Frame.mono hf2 (by decide), hm2⟩
        | some endPos =>
          rw [hsw] at hm2
          obtain ⟨hr2, h10_2, _⟩ := hm2
          have ⟨_, hend_le⟩ := skipWordsA_bounds hws_le hsw
          have hl2 : s2.regs.length = 16 := by rw [hf2.len]; simp only [List.length_set, hl1]
          have hbuf2 : Buf s2.mem b bs := by rw [hf2.mem]; exact hbuf1
          have h5_2 : s2.reg 5 = some (b + bs.length) := by rw [hf2.regs 5 (by decide)]; rsimpg; exact h5_1
          have h6_2 : s2.reg 6 = some (u32 bs 8) := by rw [hf2.regs 6 (by decide)]; rsimpg; exact h6_1
          have h7_2 : s2.reg 7 = some b := by rw [hf2.regs 7 (by decide)]; rsimpg; exact h7_1
          have h12_2 : s2.reg 12 = some (u32 bs 4) := by rw [hf2.regs 12 (by decide)]; rsimpg; exact h12_1
          have h13_2 : s2.reg 13 = some (b + ws) := by rw [hf2.regs 13 (by decide)]; rsimpg using hl1
          have hf02 : Frame [2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15] _ s2 := Frame.mono hf2 (by decide)
          xs m318 using hr2, h10_2, h5_2
          have hc4e := (cond_sub (show b + endPos < W by rw [W_eq]; omega)
            (show b + bs.length < W by rw [W_eq]; omega)).2.2.1
          by_cases heq : endPos = bs.length
          · have hc4e' : X86Int.cond (aluF .sub (b + bs.length) (b + bs.length)).snd 4 = some true := by
              simpa [heq] using hc4e
            xs m321 using heq, hc4e'
            xs m328 using h12_2, h6_2
            have hc3 := (cond_sub (show u32 bs 4 < W by rw [W_eq]; omega)
              (show u32 bs 8 < W by rw [W_eq]; omega)).2.1
            by_cases hmn : u32 bs 4 < u32 bs 8
            · xs m331 using hc3, show ¬ (u32 bs 8 ≤ u32 bs 4) by omega
              xs m337 using hl2, h13_2
              xs m340 using hl2, h12_2
              have sw' : skipWordsA bs (u32 bs 8) ws = some bs.length := by rw [hsw, heq]
              obtain ⟨mainOff, hmo, hmo20, _⟩ := skipWordsA_lt_step hmn sw'
              refine reach_frame (Frame.trans_same hf02 (by frame_tac))
                ((skipwords_spec pb.cSw2 (u32 bs 4) ws (b := b) (bs := bs) _ ?_ (by dsimp only [rip_mk]; try omega) (by rsimpg using hl2)
                  (by rsimpg using hl2) hws_le hu4).trans fun s3 ⟨hf3, hm3⟩ => ?_)
              { exact ⟨hbuf2, hby, hsm, by simp only [List.length_set, hl2], by rsimpg; exact h5_2⟩ }
              rw [hmo] at hm3
              obtain ⟨hr3, h10_3, _⟩ := hm3
              have hl3 : s3.regs.length = 16 := by rw [hf3.len]; simp only [List.length_set, hl2]
              have hbuf3 : Buf s3.mem b bs := by rw [hf3.mem]; exact hbuf2
              have h5_3 : s3.reg 5 = some (b + bs.length) := by rw [hf3.regs 5 (by decide)]; rsimpg; exact h5_2
              have h7_3 : s3.reg 7 = some b := by rw [hf3.regs 7 (by decide)]; rsimpg; exact h7_2
              have h13_3 : s3.reg 13 = some (b + ws) := by rw [hf3.regs 13 (by decide)]; rsimpg; exact h13_2
              have hf03 : Frame [2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15] _ s3 := Frame.mono hf3 (by decide)
              have hum := u32_lt hby mainOff
              xs m544 using hr3, hl3, h10_3, ea_small (show b + mainOff + 0 < 2 ^ 64 by omega),
                Nat.add_zero, rd32_buf hbuf3 (show mainOff + 4 ≤ bs.length by omega)
              xs m547 using hl3
              simp only [hmn, hmo, reduceIte]
              by_cases hm0 : u32 bs mainOff = 0
              · xs m550 using hm0, cond_test_e
                xs m557 using hl3, h13_3
                exact Reaches.refl ⟨Frame.trans_same hf03 (by frame_tac), rfl,
                  (by rsimpg; exact h5_3), (by rsimpg; exact h7_3), (by rsimpg using hl3),
                  (by rsimpg; exact h13_3), sw', trivial, hws_le, by omega⟩
              · xs m550 using hm0, cond_test_e
                xs m552
                exact Reaches.refl ⟨Frame.trans_same hf03 (by frame_tac), rfl⟩
            · xs m331 using hc3, show u32 bs 8 ≤ u32 bs 4 by omega
              simp only [hmn, reduceIte]
              exact Reaches.refl ⟨Frame.trans_same hf02 (by frame_tac), rfl⟩
          · xs m321 using hc4e, show b + endPos ≠ b + bs.length by omega
            xs m323
            simp only [heq, reduceIte]
            exact Reaches.refl ⟨Frame.trans_same hf02 (by frame_tac), rfl⟩
    · xs h38 using hc4m, hmag
      xs h40
      exact Reaches.refl ⟨by frame_tac, rfl⟩

theorem pass_spec {B D R : Nat} (pb : PassBlocks B D R)
    {b v8 : Nat} {bs : List Nat} {s : St}
    (hbuf : Buf s.mem b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < 2 ^ 48)
    (hdesc : Desc s.mem D b bs.length) (hD : D + 8 < 2 ^ 64)
    (hl : s.regs.length = 16) (hrip : s.rip = B) (h8 : s.reg 8 = some v8)
    (hR : R < 4) (_hR2 : R ≠ 2) :
    Reaches s (fun s' => Frame [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, R] s s' ∧
      s'.rip = B + passSize ∧ s'.reg R = some (verdictWord (checkFlatA bs)) ∧
      ∃ v8', s'.reg 8 = some v8') := by
  have hok := checkFlatA_header_ok hby (by unfold HMAX; omega)
  refine (parse_spec pb hbuf hby hsm hdesc hD hl hrip).trans fun s1 ⟨hf1, hm1⟩ => ?_
  have hl1 : s1.regs.length = 16 := by rw [hf1.len, hl]
  have h8_1 : s1.reg 8 = some v8 := by rw [hf1.regs 8 (by decide)]; exact h8
  have hm_1 : ∀ x, x ∈ [2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15] →
      x ∈ [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, R] := by
    intro x hx; simp only [List.mem_cons, List.not_mem_nil, or_false] at hx ⊢
    rcases hx with h|h|h|h|h|h|h|h|h|h|h <;> simp [h]
  have hm_2 : ∀ x, x ∈ [2, 6, 7, 8, 9, 10, 11, 12, 14, 15] →
      x ∈ [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, R] := by
    intro x hx; simp only [List.mem_cons, List.not_mem_nil, or_false] at hx ⊢
    rcases hx with h|h|h|h|h|h|h|h|h|h <;> simp [h]
  have hm_3 : ∀ x, x ∈ [R] → x ∈ [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, R] := by
    intro x hx; simp only [List.mem_cons, List.not_mem_nil, or_false] at hx ⊢
    simp [hx]
  have h8R : 8 ∉ [R] := by simp only [List.mem_cons, List.not_mem_nil, or_false]; omega
  have hf01 : Frame [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, R] s s1 := Frame.mono hf1 hm_1
  rw [checkFlatA.eq_def]
  simp only [hok, Bool.not_true, Bool.false_eq_true, reduceIte]
  cases hp : parseA bs with
  | none =>
    rw [hp] at hm1
    refine (end_res_spec pb .reject hl1 hR hm1).mono fun s2 ⟨hf2, hr2, hR2_⟩ => ?_
    exact ⟨Frame.trans_same hf01 (Frame.mono hf2 hm_3), hr2, hR2_, v8, by rw [hf2.regs 8 h8R]; exact h8_1⟩
  | some p =>
    rcases p with ⟨main, nw, ws⟩
    rw [hp] at hm1
    dsimp only at hm1 ⊢
    by_cases hmn : main < nw
    · simp only [hmn, decide_true, Bool.not_true, Bool.false_eq_true, reduceIte] at hm1 ⊢
      cases hmo : skipWordsA bs main ws with
      | none =>
        rw [hmo] at hm1
        refine (end_res_spec pb .reject hl1 hR hm1).mono fun s2 ⟨hf2, hr2, hR2_⟩ => ?_
        exact ⟨Frame.trans_same hf01 (Frame.mono hf2 hm_3), hr2, hR2_, v8, by rw [hf2.regs 8 h8R]; exact h8_1⟩
      | some mainOff =>
        rw [hmo] at hm1
        dsimp only at hm1 ⊢
        by_cases hm0 : u32 bs mainOff = 0
        · simp only [hm0, ne_eq, not_true_eq_false, reduceIte] at hm1 ⊢
          obtain ⟨hr1, h5_1, h7_1, h12_1, h13_1, hsw, hnw, hws, hbs16⟩ := hm1
          have hbuf1 : Buf s1.mem b bs := by rw [hf1.mem]; exact hbuf
          have hdesc1 : Desc s1.mem D b bs.length := ⟨by rw [hf1.mem]; exact hdesc.ptr, by rw [hf1.mem]; exact hdesc.len⟩
          refine (checkWords_spec pb hby hsm hD hsw hnw hws hbs16
            nw ws b v8 s1 hbuf1 hdesc1 hsw hws hl1 hr1 h5_1 h12_1 h13_1 h7_1 h8_1).trans
            fun s2 ⟨hf2, hr2, hv8_2⟩ => ?_
          have hl2 : s2.regs.length = 16 := by rw [hf2.len, hl1]
          have hf02 : Frame [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, R] s s2 :=
            Frame.trans_same hf01 (Frame.mono hf2 hm_2)
          refine (end_res_spec pb (checkWordsA bs nw ws nw ws) hl2 hR hr2).mono
            fun s3 ⟨hf3, hr3, hR3⟩ => ?_
          have h8_3 : ∃ v8', s3.reg 8 = some v8' := by
            obtain ⟨v8', hv8'⟩ := hv8_2; exact ⟨v8', by rw [hf3.regs 8 h8R]; exact hv8'⟩
          have hfr : Frame [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, R] s s3 :=
            Frame.trans_same hf02 (Frame.mono hf3 hm_3)
          cases hcw : checkWordsA bs nw ws nw ws <;> rw [hcw] at hR3 <;>
            exact ⟨hfr, hr3, hR3, h8_3⟩
        · simp only [hm0, ne_eq, not_false_eq_true, reduceIte] at hm1 ⊢
          refine (end_res_spec pb .reject hl1 hR hm1).mono fun s2 ⟨hf2, hr2, hR2_⟩ => ?_
          exact ⟨Frame.trans_same hf01 (Frame.mono hf2 hm_3), hr2, hR2_, v8, by rw [hf2.regs 8 h8R]; exact h8_1⟩
    · simp only [hmn, decide_false, Bool.not_false, reduceIte] at hm1 ⊢
      refine (end_res_spec pb .reject hl1 hR hm1).mono fun s2 ⟨hf2, hr2, hR2_⟩ => ?_
      exact ⟨Frame.trans_same hf01 (Frame.mono hf2 hm_3), hr2, hR2_, v8, by rw [hf2.regs 8 h8R]; exact h8_1⟩

/-! ### Instantiation at `p1`, `p2`, `p3` -/

private theorem codeAt_of_core_sub {c : List (Nat × Ins × Nat)}
    (h : subOK c coreTable = true) : CodeAt c :=
  fun e he => table_ok e (List.mem_append_left initTable (subOK_sound c coreTable h e he))

set_option maxRecDepth 100000 in
theorem p1_blocks : PassBlocks L_p1 0x600000 0 :=
  passBlocks_of_codeAt (codeAt_of_core_sub (by decide +kernel))

set_option maxRecDepth 100000 in
theorem p2_blocks : PassBlocks L_p2 0x601000 3 :=
  passBlocks_of_codeAt (codeAt_of_core_sub (by decide +kernel))

set_option maxRecDepth 100000 in
theorem p3_blocks : PassBlocks L_p3 0x602000 1 :=
  passBlocks_of_codeAt (codeAt_of_core_sub (by decide +kernel))

theorem p1_spec {b v8 : Nat} {bs : List Nat} {s : St}
    (hbuf : Buf s.mem b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < 2 ^ 48)
    (hdesc : Desc s.mem 0x600000 b bs.length)
    (hl : s.regs.length = 16) (hrip : s.rip = L_p1) (h8 : s.reg 8 = some v8) :
    Reaches s (fun s' => Frame [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0] s s' ∧
      s'.rip = L_p2 ∧ s'.reg 0 = some (verdictWord (checkFlatA bs)) ∧ ∃ v8', s'.reg 8 = some v8') :=
  pass_spec p1_blocks hbuf hby hsm hdesc (by decide) hl hrip h8 (by decide) (by decide)

theorem p2_spec {b v8 : Nat} {bs : List Nat} {s : St}
    (hbuf : Buf s.mem b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < 2 ^ 48)
    (hdesc : Desc s.mem 0x601000 b bs.length)
    (hl : s.regs.length = 16) (hrip : s.rip = L_p2) (h8 : s.reg 8 = some v8) :
    Reaches s (fun s' => Frame [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 3] s s' ∧
      s'.rip = L_p3 ∧ s'.reg 3 = some (verdictWord (checkFlatA bs)) ∧ ∃ v8', s'.reg 8 = some v8') :=
  pass_spec p2_blocks hbuf hby hsm hdesc (by decide) hl hrip h8 (by decide) (by decide)

theorem p3_spec {b v8 : Nat} {bs : List Nat} {s : St}
    (hbuf : Buf s.mem b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < 2 ^ 48)
    (hdesc : Desc s.mem 0x602000 b bs.length)
    (hl : s.regs.length = 16) (hrip : s.rip = L_p3) (h8 : s.reg 8 = some v8) :
    Reaches s (fun s' => Frame [2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1] s s' ∧
      s'.rip = L_vote ∧ s'.reg 1 = some (verdictWord (checkFlatA bs)) ∧ ∃ v8', s'.reg 8 = some v8') :=
  pass_spec p3_blocks hbuf hby hsm hdesc (by decide) hl hrip h8 (by decide) (by decide)

end SporosCheckBin
