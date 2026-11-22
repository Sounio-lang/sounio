/-
SporosCheckBinFault: binary-level single-disturbance immunity for `spr1_check`
(`SporosCheckBin.im` and `spr1Alts` via `X86Dec.altsAt` on `spr1Bytes`).

Contents:
1. Byte-exact agreement between `spr1Bytes[L_vote - BASE ..]` (`im`) and
   `SporosCheckFlat.tailCode` (`tailImAt L_vote`), plus `decide +kernel` audit
   of `spr1Alts` over reachable pre-disturbance tail offsets (`preFaultOffs`).
2. `BinStackSafe`, `BinTailSafe`, `BinOneFaultRun`, and `pRunWithFaultAt`:
   complete four-channel (`fetch` via `spr1Alts`, `reg`, `mem`, `flag`)
   immunity over `SporosCheckBin.im` starting at `L_vote`.
3. `decide +kernel` control-flow closure and register-isolation audit of
   `p1Tbl`, `p2Tbl`, `p3Tbl` (`coreTable` `< L_vote`), submemory and
   register-agreement simulation (`run_pass_sim`), universal single-state-
   disturbance immunity (`reg`, `mem`, `flag`) from any `TripledState`
   across Passes 1–3 and tail (plus all 4 mechanisms in tail), and explicit
   non-vacuity witness `mkTripledState`.
-/
import Bootstrap.SporosCheckBinTail
import Bootstrap.SporosCheckFlat

namespace SporosCheckBin

open X86Env X86Int

/-! ### Part 1: Tail binary match (`im`, `spr1Alts`) vs `SporosCheckFlat.tailCode` -/

def spr1Alts : Alts := fun a =>
  if a < BASE then []
  else match im a with
    | some (_, len) => X86Dec.altsAt (fun i => spr1Bytes[a - BASE + i]?) a len
    | none => []

private def corePrefixList : List (List Nat) :=
  [core_0, core_1, core_2, core_3, core_4, core_5, core_6, core_7,
   core_8, core_9, core_10, core_11, core_12, core_13, core_14, core_15,
   core_16, core_17, core_18, core_19, core_20, core_21, core_22, core_23,
   core_24, core_25, core_26, core_27, core_28, core_29, core_30, core_31,
   core_32, core_33, core_34, core_35, core_36, core_37]

private def prefixBeforeTail : List Nat :=
  hdrBytes ++ List.replicate 480 0xf4 ++ corePrefixList.flatten

private def tailSlice : List Nat :=
  (core_38 ++ core_39) ++ List.replicate 17 0xf4

private theorem chunk_lengths_ok :
    (hdrBytes.length == 176 && corePrefixList.all (fun c => c.length == 256) &&
     corePrefixList.length == 38 && core_38.length == 256 && core_39.length == 107 &&
     init_0.length == 256 && init_1.length == 99) = true := by
  decide +kernel

private theorem flatten_len_256 : ∀ ls : List (List Nat),
    (∀ c ∈ ls, c.length = 256) → ls.flatten.length = 256 * ls.length
  | [], _ => by simp
  | c :: cs, h => by
    have hc : c.length = 256 := h c (List.mem_cons_self ..)
    have hcs := flatten_len_256 cs (fun x hx => h x (List.mem_cons_of_mem _ hx))
    simp only [List.flatten_cons, List.length_append, List.length_cons, hc, hcs]
    omega

private theorem corePrefix_eq :
    core_0 ++ core_1 ++ core_2 ++ core_3 ++ core_4 ++ core_5 ++ core_6 ++ core_7 ++
    core_8 ++ core_9 ++ core_10 ++ core_11 ++ core_12 ++ core_13 ++ core_14 ++ core_15 ++
    core_16 ++ core_17 ++ core_18 ++ core_19 ++ core_20 ++ core_21 ++ core_22 ++ core_23 ++
    core_24 ++ core_25 ++ core_26 ++ core_27 ++ core_28 ++ core_29 ++ core_30 ++ core_31 ++
    core_32 ++ core_33 ++ core_34 ++ core_35 ++ core_36 ++ core_37 = corePrefixList.flatten := by
  simp only [corePrefixList, List.flatten_cons, List.flatten_nil, List.append_nil, List.append_assoc]

private theorem spr1Bytes_split_eq :
    spr1Bytes = prefixBeforeTail ++ ((core_38 ++ core_39) ++ (List.replicate 5300 0xf4 ++ initBytes)) := by
  unfold spr1Bytes coreBytes prefixBeforeTail
  rw [← corePrefix_eq]
  simp only [List.append_assoc]

private theorem hdrBytes_len_eq : hdrBytes.length = 176 := by
  have h := chunk_lengths_ok
  simp only [Bool.and_eq_true, beq_iff_eq] at h
  exact h.1.1.1.1.1.1

private theorem core38_len_eq : core_38.length = 256 := by
  have h := chunk_lengths_ok
  simp only [Bool.and_eq_true, beq_iff_eq] at h
  exact h.1.1.1.2

private theorem core39_len_eq : core_39.length = 107 := by
  have h := chunk_lengths_ok
  simp only [Bool.and_eq_true, beq_iff_eq] at h
  exact h.1.1.2

private theorem init0_len_eq : init_0.length = 256 := by
  have h := chunk_lengths_ok
  simp only [Bool.and_eq_true, beq_iff_eq] at h
  exact h.1.2

private theorem init1_len_eq : init_1.length = 99 := by
  have h := chunk_lengths_ok
  simp only [Bool.and_eq_true, beq_iff_eq] at h
  exact h.2

private theorem prefixBeforeTail_len_eq : prefixBeforeTail.length = 10384 := by
  have h := chunk_lengths_ok
  simp only [Bool.and_eq_true, beq_iff_eq] at h
  have hflat : corePrefixList.flatten.length = 256 * corePrefixList.length :=
    flatten_len_256 corePrefixList (fun c hc => beq_iff_eq.1 (List.all_eq_true.1 h.1.1.1.1.1.2 c hc))
  simp only [prefixBeforeTail, List.length_append, List.length_replicate, hdrBytes_len_eq, hflat,
    h.1.1.1.1.2]

private theorem tailSlice_len_eq : tailSlice.length = 380 := by
  simp only [tailSlice, List.length_append, List.length_replicate, core38_len_eq, core39_len_eq]

private theorem tailSlice_bytes_eq :
    ((core_38 ++ core_39).drop 220).take 143 = SporosCheckFlat.tailCode := by
  decide +kernel

theorem L_vote_base_ok : SporosCheckFlat.base_ok L_vote := by
  unfold SporosCheckFlat.base_ok L_vote W
  decide

private theorem leB_congr {b1 b2 : Nat → Option Nat} :
    ∀ (n off : Nat), (∀ i, i < off + n → b1 i = b2 i) →
      X86Dec.leB b1 off n = X86Dec.leB b2 off n
  | 0, _, _ => rfl
  | n + 1, off, h => by
    simp only [X86Dec.leB, h off (by omega), leB_congr n (off + 1) (fun i hi => h i (by omega))]

private theorem memOp_congr_lt {b1 b2 : Nat → Option Nat} {off m bb : Nat} {x : Bool}
    (h0 : b1 off = b2 off) (h1 : b1 (off + 1) = b2 (off + 1))
    (h2 : X86Dec.leB b1 off 4 = X86Dec.leB b2 off 4)
    (h3 : X86Dec.leB b1 (off + 1) 4 = X86Dec.leB b2 (off + 1) 4) :
    X86Dec.memOp b1 off m bb x = X86Dec.memOp b2 off m bb x := by
  simp only [X86Dec.memOp]
  by_cases hmd3 : m / 64 == 3
  · simp [hmd3]
  · simp only [hmd3, Bool.false_eq_true, ↓reduceIte]
    by_cases hm4 : m % 8 == 4
    · simp only [hm4, ↓reduceIte, h0, bind, Option.bind]
      cases b2 off with
      | none => rfl
      | some s =>
        simp only []
        split
        · rfl
        · split
          · rfl
          · cases hmd : m / 64 with
            | zero => rfl
            | succ n => cases n with | zero => simp only [pure, h1] | succ _ => simp only [pure, h1, h3]
    · simp only [hm4, Bool.false_eq_true, ↓reduceIte, bind, Option.bind]
      split
      · rfl
      · cases hmd : m / 64 with
        | zero => rfl
        | succ n => cases n with | zero => simp only [pure, h0] | succ _ => simp only [pure, h0, h2]

private theorem decode_congr_12 {b1 b2 : Nat → Option Nat} (a : Nat)
    (h : ∀ i, i < 12 → b1 i = b2 i) :
    X86Dec.decode b1 a = X86Dec.decode b2 a := by
  have h0 : b1 0 = b2 0 := h 0 (by omega)
  simp only [X86Dec.decode, h0, bind, Option.bind]
  cases hb0 : b2 0 with
  | none => rfl
  | some b0 =>
    dsimp only []
    cases hrex : (0x40 ≤ b0 && b0 ≤ 0x4f)
    · have ha1 : b1 1 = b2 1 := h 1 (by omega)
      have ha2 : b1 2 = b2 2 := h 2 (by omega)
      have ha3 : b1 3 = b2 3 := h 3 (by omega)
      have hl1 : X86Dec.leB b1 1 8 = X86Dec.leB b2 1 8 := leB_congr 8 1 (fun i hi => h i (by omega))
      have hl2 : X86Dec.leB b1 1 4 = X86Dec.leB b2 1 4 := leB_congr 4 1 (fun i hi => h i (by omega))
      have hl3 : X86Dec.leB b1 2 4 = X86Dec.leB b2 2 4 := leB_congr 4 2 (fun i hi => h i (by omega))
      have hl4 : X86Dec.leB b1 3 4 = X86Dec.leB b2 3 4 := leB_congr 4 3 (fun i hi => h i (by omega))
      have hm1 : ∀ m bb x, X86Dec.memOp b1 2 m bb x = X86Dec.memOp b2 2 m bb x :=
        fun m bb x => memOp_congr_lt ha2 ha3 hl3 hl4
      have hm2 : ∀ m bb x, X86Dec.memOp b1 3 m bb x = X86Dec.memOp b2 3 m bb x :=
        fun m bb x => memOp_congr_lt ha3 (h 4 (by omega)) hl4 (leB_congr 4 4 (fun i hi => h i (by omega)))
      have h0s : b1 0 = some b0 := h0.trans hb0
      have h2s : b2 0 = some b0 := hb0
      simp only [Bool.false_eq_true, ↓reduceIte, Nat.zero_add, h0s, h2s,
        ha1, ha2, ha3, hl1, hl2, hl3, hl4, hm1, hm2]
    · have ha1 : b1 1 = b2 1 := h 1 (by omega)
      have ha2 : b1 2 = b2 2 := h 2 (by omega)
      have ha3 : b1 3 = b2 3 := h 3 (by omega)
      have ha4 : b1 4 = b2 4 := h 4 (by omega)
      have hl2 : X86Dec.leB b1 2 8 = X86Dec.leB b2 2 8 := leB_congr 8 2 (fun i hi => h i (by omega))
      have hl3 : X86Dec.leB b1 2 4 = X86Dec.leB b2 2 4 := leB_congr 4 2 (fun i hi => h i (by omega))
      have hl4 : X86Dec.leB b1 3 4 = X86Dec.leB b2 3 4 := leB_congr 4 3 (fun i hi => h i (by omega))
      have hl5 : X86Dec.leB b1 4 4 = X86Dec.leB b2 4 4 := leB_congr 4 4 (fun i hi => h i (by omega))
      have hm1 : ∀ m bb x, X86Dec.memOp b1 3 m bb x = X86Dec.memOp b2 3 m bb x :=
        fun m bb x => memOp_congr_lt ha3 ha4 hl4 hl5
      have hm2 : ∀ m bb x, X86Dec.memOp b1 4 m bb x = X86Dec.memOp b2 4 m bb x :=
        fun m bb x => memOp_congr_lt ha4 (h 5 (by omega)) hl5 (leB_congr 4 5 (fun i hi => h i (by omega)))
      simp only [↓reduceIte, Nat.reduceAdd]
      rw [ha1]
      cases hb1 : b2 1 with
      | none => rfl
      | some o =>
        dsimp only []
        simp only [ha2, ha3, ha4, hl2, hl3, hl4, hl5, hm1, hm2]

private theorem altsAt_congr_12 {b1 b2 : Nat → Option Nat} (a len : Nat)
    (h : ∀ i, i < 12 → b1 i = b2 i) :
    X86Dec.altsAt b1 a len = X86Dec.altsAt b2 a len := by
  simp only [X86Dec.altsAt]
  apply List.map_congr_left
  intro k _
  apply decode_congr_12
  intro i hi
  simp only [X86Dec.flipBit]
  split
  · rw [h i hi]
  · exact h i hi

private theorem spr1Bytes_len_eq : spr1Bytes.length = 16402 := by
  rw [spr1Bytes_split_eq]
  simp only [List.length_append, List.length_replicate, prefixBeforeTail_len_eq,
    core38_len_eq, core39_len_eq, initBytes, init0_len_eq, init1_len_eq]

private theorem spr1Bytes_pad0_f4 {a : Nat} (h1 : 0x4000b0 ≤ a) (h2 : a < L_tripled) :
    spr1Bytes[a - BASE]? = some 0xf4 := by
  have hsub : 176 ≤ a - BASE ∧ a - BASE < 656 := by unfold BASE L_tripled at *; omega
  rw [spr1Bytes_split_eq,
    List.getElem?_append_left (by simp only [prefixBeforeTail, List.length_append, List.length_replicate, hdrBytes_len_eq]; omega)]
  simp only [prefixBeforeTail]
  rw [List.getElem?_append_left (by simp only [List.length_append, List.length_replicate, hdrBytes_len_eq]; omega)]
  rw [List.getElem?_append_right (by rw [hdrBytes_len_eq]; omega)]
  rw [hdrBytes_len_eq, List.getElem?_replicate, if_pos (by omega)]

private theorem spr1Bytes_pad1_f4 {a : Nat} (h1 : L_core_end ≤ a) (h2 : a < L_start) :
    spr1Bytes[a - BASE]? = some 0xf4 := by
  have hsub : 10747 ≤ a - BASE ∧ a - BASE < 16047 := by unfold BASE L_core_end L_start at *; omega
  rw [spr1Bytes_split_eq,
    List.getElem?_append_right (by rw [prefixBeforeTail_len_eq]; omega), prefixBeforeTail_len_eq,
    List.getElem?_append_right (by simp only [List.length_append, core38_len_eq, core39_len_eq]; omega),
    List.length_append, core38_len_eq, core39_len_eq,
    List.getElem?_append_left (by rw [List.length_replicate]; omega),
    List.getElem?_replicate, if_pos (by omega)]

private theorem getElem?_append_left_exp (xs ys : List Nat) {k : Nat} (h : k < xs.length) :
    (xs ++ ys)[k]? = xs[k]? := List.getElem?_append_left h

private theorem getElem?_append_right_exp (xs ys : List Nat) {k : Nat} (h : xs.length ≤ k) :
    (xs ++ ys)[k]? = ys[k - xs.length]? := List.getElem?_append_right h

private theorem spr1Bytes_tailSlice_getElem? {k : Nat} (hk : k < 380) :
    spr1Bytes[10384 + k]? = tailSlice[k]? := by
  have hcore : (core_38 ++ core_39).length = 363 := by
    simp only [List.length_append, core38_len_eq, core39_len_eq]
  rw [spr1Bytes_split_eq,
    getElem?_append_right_exp prefixBeforeTail _ (by rw [prefixBeforeTail_len_eq]; omega),
    prefixBeforeTail_len_eq, Nat.add_sub_cancel_left]
  by_cases h363 : k < 363
  · rw [getElem?_append_left_exp (core_38 ++ core_39) _ (by rw [hcore]; omega)]
    rw [tailSlice, getElem?_append_left_exp (core_38 ++ core_39) _ (by rw [hcore]; omega)]
  · have h363' : 363 ≤ k := Nat.le_of_not_lt h363
    rw [getElem?_append_right_exp (core_38 ++ core_39) _ (by rw [hcore]; omega)]
    rw [getElem?_append_left_exp (List.replicate 5300 0xf4) initBytes (by rw [hcore, List.length_replicate]; omega)]
    rw [tailSlice, getElem?_append_right_exp (core_38 ++ core_39) _ (by rw [hcore]; omega)]
    rw [hcore, List.getElem?_replicate, List.getElem?_replicate, if_pos (by omega), if_pos (by omega)]

theorem tail_bytes_eq :
    (spr1Bytes.drop (L_vote - BASE)).take 143 = SporosCheckFlat.tailCode := by
  have hcore : (core_38 ++ core_39).length = 363 := by
    simp only [List.length_append, core38_len_eq, core39_len_eq]
  rw [← tailSlice_bytes_eq]
  apply List.ext_getElem?
  intro k
  by_cases hk : k < 143
  · rw [List.getElem?_take_of_lt hk, List.getElem?_drop,
      List.getElem?_take_of_lt hk, List.getElem?_drop,
      show L_vote - BASE + k = 10384 + (220 + k) by unfold L_vote BASE; omega,
      spr1Bytes_tailSlice_getElem? (by omega),
      tailSlice, getElem?_append_left_exp (core_38 ++ core_39) _ (by rw [hcore]; omega)]
  · rw [List.getElem?_take, List.getElem?_take, if_neg hk, if_neg hk]

private theorem im_tailSlice_eq {a : Nat}
    (h1 : BASE + 10384 ≤ a) (h2 : a ≤ BASE + 10747 + 5) :
    im a = X86Dec.decode (fun i => tailSlice[(a - (BASE + 10384)) + i]?) a := by
  unfold im X86Dec.imageOf
  rw [if_neg (by omega)]
  apply decode_congr_12
  intro i hi
  rw [show a - BASE + i = 10384 + ((a - (BASE + 10384)) + i) by omega]
  exact spr1Bytes_tailSlice_getElem? (by omega)

private theorem decode_f4_none (b : Nat → Option Nat) (a : Nat) (h0 : b 0 = some 0xf4) :
    X86Dec.decode b a = none := by
  simp only [X86Dec.decode, h0, bind, Option.bind]
  rw [show (if (0x40 ≤ (0xf4 : Nat) && (0xf4 : Nat) ≤ 0x4f) = true then 1 else 0) = 0 by decide]
  simp only [h0]
  rfl

private theorem im_eq_none_of_byte_f4 {a : Nat} (hge : BASE ≤ a)
    (hb : spr1Bytes[a - BASE]? = some 0xf4) : im a = none := by
  unfold im X86Dec.imageOf
  rw [if_neg (by omega)]
  have hb0 : spr1Bytes[a - BASE + 0]? = some 0xf4 := by
    rw [Nat.add_zero]; exact hb
  exact decode_f4_none _ a hb0

private def extraNoneTailAddrs : List Nat := [4204798, 4204818, 4204848, 4204906]

private theorem extraNoneTailAddrs_ok :
    extraNoneTailAddrs.all (fun a => (X86Dec.decode (fun i => tailSlice[(a - (BASE + 10384)) + i]?) a).isNone) = true := by
  decide +kernel

private def isNoneAddr (t : Nat) : Bool :=
  t < BASE || BASE + 16402 ≤ t ||
  (0x4000b0 ≤ t && t < L_tripled) ||
  (L_core_end ≤ t && t < L_start) ||
  extraNoneTailAddrs.contains t

private theorem decode_none_of_b0 (b : Nat → Option Nat) (a : Nat) (h0 : b 0 = none) :
    X86Dec.decode b a = none := by
  simp only [X86Dec.decode, h0, bind, Option.bind]

private theorem im_eq_none_of_isNoneAddr {t : Nat} (h : isNoneAddr t = true) : im t = none := by
  simp only [isNoneAddr, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at h
  rcases h with (((hlt | hge) | ⟨h0, h1⟩) | ⟨h0, h1⟩) | hex
  · unfold im X86Dec.imageOf; rw [if_pos hlt]
  · unfold im X86Dec.imageOf
    rw [if_neg (by omega)]
    have hn : spr1Bytes[t - BASE]? = none := List.getElem?_eq_none (by rw [spr1Bytes_len_eq]; omega)
    have hn0 : spr1Bytes[t - BASE + 0]? = none := by rw [Nat.add_zero]; exact hn
    exact decode_none_of_b0 _ t hn0
  · exact im_eq_none_of_byte_f4 (by unfold BASE at *; omega) (spr1Bytes_pad0_f4 h0 h1)
  · exact im_eq_none_of_byte_f4 (by unfold BASE L_core_end at *; omega) (spr1Bytes_pad1_f4 h0 h1)
  · have hm : t ∈ extraNoneTailAddrs := List.contains_iff_mem.1 hex
    have hbd : BASE + 10384 ≤ t ∧ t ≤ BASE + 10747 + 5 := by
      have hall : extraNoneTailAddrs.all (fun a => decide (BASE + 10384 ≤ a ∧ a ≤ BASE + 10747 + 5)) = true := by
        decide
      exact of_decide_eq_true (List.all_eq_true.1 hall t hm)
    rw [im_tailSlice_eq hbd.1 hbd.2]
    exact Option.isNone_iff_eq_none.1 (List.all_eq_true.1 extraNoneTailAddrs_ok t hm)

theorem im_0_none : im 0 = none := im_eq_none_of_isNoneAddr (by decide)

theorem im_core_end_none : im L_core_end = none := im_eq_none_of_isNoneAddr (by decide)

theorem im_4204869_spec :
    im 4204869 = some (.jcc 0 4204848, 2) ∧ im 4204871 = none ∧ im 4204848 = none := by
  refine ⟨?_, ?_, im_eq_none_of_isNoneAddr (by decide)⟩
  · rw [im_tailSlice_eq (by decide) (by decide)]; decide +kernel
  · rw [im_tailSlice_eq (by decide) (by decide)]; decide +kernel

def tailWindow : List Nat := SporosCheckFlat.tailCode ++ List.replicate 17 0xf4

private theorem tailSlice_window_eq :
    (tailSlice.drop 220).take 160 = tailWindow := by
  decide +kernel

private theorem tailWindow_eq :
    (spr1Bytes.drop (L_vote - BASE)).take 160 = tailWindow := by
  rw [← tailSlice_window_eq]
  apply List.ext_getElem?
  intro k
  by_cases hk : k < 160
  · rw [List.getElem?_take_of_lt hk, List.getElem?_drop,
      List.getElem?_take_of_lt hk, List.getElem?_drop,
      show L_vote - BASE + k = 10384 + (220 + k) by unfold L_vote BASE; omega]
    exact spr1Bytes_tailSlice_getElem? (by omega)
  · rw [List.getElem?_take, List.getElem?_take, if_neg hk, if_neg hk]

private theorem tailWindow_hi_im_offsets_eq :
    ((List.range 11).map (· + 132)).all (fun off =>
      X86Dec.decode (fun i => tailWindow[off + i]?) (L_vote + off) ==
        SporosCheckFlat.tailImAt L_vote (L_vote + off)) = true := by
  decide +kernel

private theorem tailWindow_im_offsets_eq :
    (List.range 143).all (fun off =>
      X86Dec.decode (fun i => tailWindow[off + i]?) (L_vote + off) ==
        SporosCheckFlat.tailImAt L_vote (L_vote + off)) = true := by
  rw [List.all_eq_true]
  intro off hoff
  rw [List.mem_range] at hoff
  by_cases h132 : off < 132
  · rw [beq_iff_eq]
    have hge : ¬ (L_vote + off < L_vote) := by omega
    unfold SporosCheckFlat.tailImAt X86Dec.imageOf
    rw [if_neg hge, Nat.add_sub_cancel_left]
    apply decode_congr_12
    intro i hi
    simp only [tailWindow]
    rw [List.getElem?_append_left (by rw [SporosCheckFlat.tailCode_length]; omega)]
  · have hmem : off ∈ (List.range 11).map (· + 132) := by
      rw [List.mem_map]
      exact ⟨off - 132, List.mem_range.2 (by omega), by omega⟩
    exact List.all_eq_true.1 tailWindow_hi_im_offsets_eq off hmem

theorem im_vote_eq_tailImAt {a : Nat} (h1 : L_vote ≤ a) (h2 : a < L_core_end) :
    im a = SporosCheckFlat.tailImAt L_vote a := by
  have hoff : a - L_vote < 143 := by unfold L_vote L_core_end at *; omega
  have hge : ¬ (a < BASE) := by unfold L_vote BASE at *; omega
  have hdec : im a = X86Dec.decode (fun i => tailWindow[(a - L_vote) + i]?) a := by
    unfold im X86Dec.imageOf
    rw [if_neg hge]
    apply decode_congr_12
    intro i hi
    rw [show a - BASE + i = (L_vote - BASE) + ((a - L_vote) + i) by unfold L_vote BASE at *; omega]
    rw [← List.getElem?_drop, ← List.getElem?_take_of_lt (by omega : (a - L_vote) + i < 160), tailWindow_eq]
  have heq := beq_iff_eq.1 (List.all_eq_true.1 tailWindow_im_offsets_eq (a - L_vote) (List.mem_range.2 hoff))
  rw [Nat.add_sub_cancel' h1] at heq
  exact hdec.trans heq

private def tailWindowAlts (off : Nat) : List (Option (Ins × Nat)) :=
  match SporosCheckFlat.tailImAt L_vote (L_vote + off) with
  | some (_, len) => X86Dec.altsAt (fun i => tailWindow[off + i]?) (L_vote + off) len
  | none => []

private theorem tailWindow_len_le_12 :
    SporosCheckFlat.preFaultOffs.all (fun off =>
      match SporosCheckFlat.tailImAt L_vote (L_vote + off) with
      | some (_, len) => decide (len ≤ 12)
      | none => true) = true := by
  decide +kernel

private theorem spr1Alts_eq_tailWindowAlts {off : Nat}
    (hmem : off ∈ SporosCheckFlat.preFaultOffs) :
    spr1Alts (L_vote + off) = tailWindowAlts off := by
  have hoff : off < 143 := by
    have ha : SporosCheckFlat.preFaultOffs.all (fun o => decide (o < 143)) = true := by decide
    exact of_decide_eq_true (List.all_eq_true.1 ha off hmem)
  have hge : ¬ (L_vote + off < BASE) := by unfold L_vote BASE; omega
  have him_eq : im (L_vote + off) = SporosCheckFlat.tailImAt L_vote (L_vote + off) :=
    im_vote_eq_tailImAt (by omega) (by unfold L_vote L_core_end; omega)
  simp only [spr1Alts, tailWindowAlts, if_neg hge, him_eq]
  cases hti : SporosCheckFlat.tailImAt L_vote (L_vote + off) with
  | none => rfl
  | some p =>
    rcases p with ⟨ins, len⟩
    apply altsAt_congr_12
    intro i hi
    rw [show L_vote + off - BASE + i = (L_vote - BASE) + (off + i) by unfold L_vote BASE at *; omega]
    rw [← List.getElem?_drop, ← List.getElem?_take_of_lt (by omega : off + i < 160), tailWindow_eq]

private theorem tailAllSpec_no_jump_out_and_no_ret_step :
    SporosCheckFlat.tailAllSpec.all (fun (off, ins, l) =>
      match ins with
      | .jmp t => decide (t < 143 ∧ t ∉ [0x14, 0x2e, 0x3d])
      | .jcc _ t => decide (t < 143 ∧ t ∉ [0x14, 0x2e, 0x3d] ∧ off + l < 143 ∧ off + l ∉ [0x14, 0x2e, 0x3d])
      | .call _ | .jtab _ _ _ | .jmpR _ => false
      | .ret => decide (off ∈ [0x14, 0x2e, 0x3d])
      | _ => decide (off + l ≤ 143 ∧ off + l ∉ [0x14, 0x2e, 0x3d])) = true := by
  decide +kernel

private theorem tailAlts_pre_0x88 :
    (tailWindowAlts 0x88 == SporosCheckFlat.tailAltsAt L_vote (L_vote + 0x88)) = true := by
  decide +kernel

private theorem tailAlts_pre_match :
    SporosCheckFlat.preFaultOffs.all (fun off =>
      off == 0x8d || tailWindowAlts off == SporosCheckFlat.tailAltsAt L_vote (L_vote + off)) = true := by
  have hbound : SporosCheckFlat.preFaultOffs.all (fun off =>
      off == 0x8d || off == 0x88 || off < 132) = true := by decide
  rw [List.all_eq_true]
  intro off hmem
  have hb := List.all_eq_true.1 hbound off hmem
  simp only [Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq] at hb ⊢
  rcases hb with (rfl | rfl) | h132
  · exact Or.inl rfl
  · exact Or.inr (by simpa using tailAlts_pre_0x88)
  · right
    have hge : ¬ (L_vote + off < L_vote) := by omega
    unfold tailWindowAlts SporosCheckFlat.tailAltsAt
    rw [if_neg hge]
    cases hti : SporosCheckFlat.tailImAt L_vote (L_vote + off) with
    | none => rfl
    | some p =>
      rcases p with ⟨ins, len⟩
      apply altsAt_congr_12
      intro i hi
      simp only [tailWindow, Nat.add_sub_cancel_left]
      rw [List.getElem?_append_left (by rw [SporosCheckFlat.tailCode_length]; omega)]

def noWriteSp : Ins → Bool
  | .ld d _ _ | .ld32 d _ _ | .ld8 d _ _ | .mov d _ | .mov32 d _ | .movI d _
  | .alu _ d _ | .aluI _ d _ | .shlI d _ | .shrI d _ | .lea d _ _ => d != 4
  | .pop _ | .push _ | .pushI _ | .call _ | .ret => false
  | _ => true

private theorem tailAlts_pre_exit_audit :
    SporosCheckFlat.preFaultOffs.all (fun off =>
      (tailWindowAlts off).all (fun opt =>
        match opt with
        | none => true
        | some (ins, l) =>
          match ins with
          | .ret | .jmpR _ | .call _ | .jtab _ _ _ | .pushI _ => false
          | .push _ => decide (off + l < 143) && (SporosCheckFlat.tailImAt L_vote (L_vote + off + l)).isNone
          | .jmp t =>
            decide (L_vote ≤ t ∧ t < L_core_end) || t == 4204869 || isNoneAddr t
          | .jcc _ t =>
            (decide (L_vote ≤ t ∧ t < L_core_end) || t == 4204869 || isNoneAddr t) &&
            (decide (L_vote ≤ L_vote + off + l ∧ L_vote + off + l < L_core_end) ||
              isNoneAddr (L_vote + off + l))
          | _ =>
            let nx := off + l
            let safeMem :=
              if !noWriteSp ins || (match ins with | .st _ _ _ | .st8 _ _ _ | .pop _ | .syscall => true | _ => false)
              then decide (nx ∉ [0x14, 0x2e, 0x3d])
              else true
            (decide (nx < 143) && safeMem) || isNoneAddr (L_vote + nx))) = true := by
  decide +kernel

private theorem tailAlts_8d_audit :
    (tailWindowAlts 0x8d).all (fun opt =>
      opt ∈ SporosCheckFlat.tailAltsAt L_vote (L_vote + 0x8d) ||
      match opt with
      | some (.jcc _ t, 6) => isNoneAddr t && isNoneAddr (L_vote + 0x8d + 6)
      | _ => false) = true := by
  decide +kernel

private theorem preFaultOff_mem_public {v : Nat} {s : St} {off : Nat}
    (h : SporosCheckFlat.PreFaultOff v s off) : off ∈ SporosCheckFlat.preFaultOffs := by
  dsimp only [SporosCheckFlat.PreFaultOff] at h
  split at h
  all_goals first | exact False.elim h | decide

private theorem preFaultOff_not_ret :
    SporosCheckFlat.preFaultOffs.all (fun off => decide (off ∉ [0x14, 0x2e, 0x3d])) = true := by
  decide +kernel

private theorem preFault_mem_of_base {base v : Nat} {s : St}
    (h : SporosCheckFlat.PreFault base v s) : s.rip - base ∈ SporosCheckFlat.preFaultOffs :=
  preFaultOff_mem_public h.2.2.2.2.2

private theorem preFault_0x7c_or_0x8d_reg0 {base v : Nat} {s : St}
    (h : SporosCheckFlat.PreFault base v s)
    (hx : s.rip - base = 0x7c ∨ s.rip - base = 0x8d) :
    s.reg 0 = some 60 := by
  have hpo := h.2.2.2.2.2
  rcases hx with heq | heq <;> (rw [heq] at hpo; exact hpo.2.1)

private theorem preFault_no_st_0 :
    SporosCheckFlat.preFaultOffs.all (fun off =>
      match SporosCheckFlat.tailImAt 0 off with
      | some (ins, _) =>
        (match ins with
         | .cmp _ _ | .jcc _ _ | .mov _ _ | .jmp _ | .movI _ _ | .alu _ _ _ | .syscall => true
         | _ => false) && noWriteSp ins
      | _ => false) = true := by
  decide +kernel

private theorem preFault_sys_only_0 :
    SporosCheckFlat.preFaultOffs.all (fun off =>
      match SporosCheckFlat.tailImAt 0 off with
      | some (.syscall, _) => decide (off = 0x7c ∨ off = 0x8d)
      | _ => true) = true := by
  decide +kernel

private def isNonStoreIns : Ins → Bool
  | .st _ _ _ | .st8 _ _ _ | .push _ | .pushI _ | .call _ | .ret | .pop _ | .syscall => false
  | _ => true

private theorem isNonStoreIns_spec {ins : Ins} (h : isNonStoreIns ins = true) :
    ∀ a b c, ins ≠ .st a b c ∧ ins ≠ .st8 a b c ∧ ins ≠ .push a ∧
      ins ≠ .pushI a ∧ ins ≠ .call a ∧ ins ≠ .ret ∧ ins ≠ .pop a ∧ ins ≠ .syscall := by
  intro a b c
  cases ins <;> simp [isNonStoreIns] at h ⊢

/-! ### Part 2: Instruction semantics helpers (`exec_rip_cases`, bounds, bytes) -/

def ripNextCond (ins : Ins) (len : Nat) (s s' : St) : Prop :=
  match ins with
  | .jmp t => s'.rip = t
  | .jcc _ t => s'.rip = t ∨ s'.rip = s.rip + len
  | .call t => s'.rip = t
  | .ret => ∃ sp t, s.reg 4 = some sp ∧ rd64 s.mem sp = some t ∧ s'.rip = t
  | .jtab b i disp =>
    ∃ x y t, s.reg b = some x ∧ s.reg i = some y ∧
      rd64 s.mem ((x + 8 * y + disp) % W) = some t ∧ s'.rip = t
  | .jmpR r => ∃ t, s.reg r = some t ∧ s'.rip = t
  | _ => s'.rip = s.rip + len

private theorem exec_rip_cases {ins : Ins} {len : Nat} {s s' : St} (h : exec ins len s = .ok s') :
    ripNextCond ins len s s' := by
  cases ins with
  | jmp t => simp [exec] at h; cases h; rfl
  | jcc cc t =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases hc : X86Int.cond s.fl cc with
    | none => simp [hc] at h
    | some c =>
      simp only [hc] at h
      cases h
      cases c <;> simp [ripNextCond]
  | call t =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases hsp : s.reg 4 with
    | none => simp [hsp] at h
    | some sp =>
      simp only [hsp] at h
      cases hwr : wr64 s.mem ((sp + W - 8) % W) (s.rip + len) with
      | none => simp [hwr] at h
      | some _ => simp [hwr] at h; cases h; rfl
  | ret =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases hsp : s.reg 4 with
    | none => simp [hsp] at h
    | some sp =>
      simp only [hsp] at h
      cases hrd : rd64 s.mem sp with
      | none => simp [hrd] at h
      | some t => simp [hrd] at h; cases h; exact ⟨sp, t, hsp, hrd, rfl⟩
  | jtab b i disp =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases hb : s.reg b with
    | none => simp [hb] at h
    | some x =>
      simp only [hb] at h
      cases hi : s.reg i with
      | none => simp [hi] at h
      | some y =>
        simp only [hi] at h
        cases hrd : rd64 s.mem ((x + 8 * y + disp) % W) with
        | none => simp [hrd] at h
        | some t => simp [hrd] at h; cases h; exact ⟨x, y, t, hb, hi, hrd, rfl⟩
  | jmpR r =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases hr : s.reg r with
    | none => simp [hr] at h
    | some t => simp [hr] at h; cases h; exact ⟨t, hr, rfl⟩
  | movI _ _ => simp [exec] at h; cases h; rfl
  | mov _ r =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases hr : s.reg r with
    | none => simp [hr] at h
    | some _ => simp [hr] at h; cases h; rfl
  | mov32 _ r =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases hr : s.reg r with
    | none => simp [hr] at h
    | some _ => simp [hr] at h; cases h; rfl
  | alu _ d r =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg d with
    | none => simp [ha] at h
    | some _ =>
      simp only [ha] at h
      cases hb : s.reg r with
      | none => simp [hb] at h
      | some _ => simp [hb] at h; cases h; rfl
  | aluI _ d _ =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg d with
    | none => simp [ha] at h
    | some _ => simp [ha] at h; cases h; rfl
  | cmp a b =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg a with
    | none => simp [ha] at h
    | some _ =>
      simp only [ha] at h
      cases hb : s.reg b with
      | none => simp [hb] at h
      | some _ => simp [hb] at h; cases h; rfl
  | cmpI a _ =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg a with
    | none => simp [ha] at h
    | some _ => simp [ha] at h; cases h; rfl
  | test a b =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg a with
    | none => simp [ha] at h
    | some _ =>
      simp only [ha] at h
      cases hb : s.reg b with
      | none => simp [hb] at h
      | some _ => simp [hb] at h; cases h; rfl
  | shlI d _ =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg d with
    | none => simp [ha] at h
    | some _ => simp [ha] at h; cases h; rfl
  | shrI d _ =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg d with
    | none => simp [ha] at h
    | some _ => simp [ha] at h; cases h; rfl
  | ld _ b disp =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg b with
    | none => simp [ha] at h
    | some av =>
      simp only [ha] at h
      cases hrd : rd64 s.mem (ea av disp) with
      | none => simp [hrd] at h
      | some _ => simp [hrd] at h; cases h; rfl
  | st b disp src =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg b with
    | none => simp [ha] at h
    | some av =>
      simp only [ha] at h
      cases hb : s.reg src with
      | none => simp [hb] at h
      | some bv =>
        simp only [hb] at h
        cases hwr : wr64 s.mem (ea av disp) bv with
        | none => simp [hwr] at h
        | some _ => simp [hwr] at h; cases h; rfl
  | ld32 _ b disp =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg b with
    | none => simp [ha] at h
    | some av =>
      simp only [ha] at h
      cases hrd : rd32 s.mem (ea av disp) with
      | none => simp [hrd] at h
      | some _ => simp [hrd] at h; cases h; rfl
  | ld8 _ b disp =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg b with
    | none => simp [ha] at h
    | some av =>
      simp only [ha] at h
      cases hrd : s.mem.rd (ea av disp) with
      | none => simp [hrd] at h
      | some _ => simp [hrd] at h; cases h; rfl
  | st8 b disp src =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg b with
    | none => simp [ha] at h
    | some av =>
      simp only [ha] at h
      cases hb : s.reg src with
      | none => simp [hb] at h
      | some bv =>
        simp only [hb] at h
        cases hwr : s.mem.wr (ea av disp) (bv % 256) with
        | none => simp [hwr] at h
        | some _ => simp [hwr] at h; cases h; rfl
  | lea _ b _ =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg b with
    | none => simp [ha] at h
    | some _ => simp [ha] at h; cases h; rfl
  | pushI imm =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg 4 with
    | none => simp [ha] at h
    | some sp =>
      simp only [ha] at h
      cases hwr : wr64 s.mem ((sp + W - 8) % W) (imm % W) with
      | none => simp [hwr] at h
      | some _ => simp [hwr] at h; cases h; rfl
  | push r =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases hv : s.reg r with
    | none => simp [hv] at h
    | some v =>
      simp only [hv] at h
      cases ha : s.reg 4 with
      | none => simp [ha] at h
      | some sp =>
        simp only [ha] at h
        cases hwr : wr64 s.mem ((sp + W - 8) % W) v with
        | none => simp [hwr] at h
        | some _ => simp [hwr] at h; cases h; rfl
  | pop _ =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg 4 with
    | none => simp [ha] at h
    | some sp =>
      simp only [ha] at h
      cases hrd : rd64 s.mem sp with
      | none => simp [hrd] at h
      | some _ => simp [hrd] at h; cases h; rfl
  | syscall =>
    simp only [exec] at h
    split at h
    · split at h
      · cases h; rfl
      · cases h
      · cases h
    · cases h

private theorem exec_preserves_mem_non_store {ins : Ins} {len : Nat} {s s' : St}
    (h : exec ins len s = .ok s')
    (hnost : ∀ a b c, ins ≠ .st a b c ∧ ins ≠ .st8 a b c ∧ ins ≠ .push a ∧
      ins ≠ .pushI a ∧ ins ≠ .call a ∧ ins ≠ .ret ∧ ins ≠ .pop a ∧ ins ≠ .syscall) :
    s'.mem = s.mem := by
  cases ins with
  | jmp _ => simp [exec] at h; cases h; rfl
  | jcc cc _ =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases hc : X86Int.cond s.fl cc with
    | none => simp [hc] at h
    | some _ => simp [hc] at h; cases h; rfl
  | movI _ _ => simp [exec] at h; cases h; rfl
  | mov _ r =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases hr : s.reg r with
    | none => simp [hr] at h
    | some _ => simp [hr] at h; cases h; rfl
  | mov32 _ r =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases hr : s.reg r with
    | none => simp [hr] at h
    | some _ => simp [hr] at h; cases h; rfl
  | alu _ d r =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg d with
    | none => simp [ha] at h
    | some _ =>
      simp only [ha] at h
      cases hb : s.reg r with
      | none => simp [hb] at h
      | some _ => simp [hb] at h; cases h; rfl
  | aluI _ d _ =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg d with
    | none => simp [ha] at h
    | some _ => simp [ha] at h; cases h; rfl
  | cmp a b =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg a with
    | none => simp [ha] at h
    | some _ =>
      simp only [ha] at h
      cases hb : s.reg b with
      | none => simp [hb] at h
      | some _ => simp [hb] at h; cases h; rfl
  | cmpI a _ =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg a with
    | none => simp [ha] at h
    | some _ => simp [ha] at h; cases h; rfl
  | test a b =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg a with
    | none => simp [ha] at h
    | some _ =>
      simp only [ha] at h
      cases hb : s.reg b with
      | none => simp [hb] at h
      | some _ => simp [hb] at h; cases h; rfl
  | shlI d _ =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg d with
    | none => simp [ha] at h
    | some _ => simp [ha] at h; cases h; rfl
  | shrI d _ =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg d with
    | none => simp [ha] at h
    | some _ => simp [ha] at h; cases h; rfl
  | ld _ b disp =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg b with
    | none => simp [ha] at h
    | some av =>
      simp only [ha] at h
      cases hrd : rd64 s.mem (ea av disp) with
      | none => simp [hrd] at h
      | some _ => simp [hrd] at h; cases h; rfl
  | ld32 _ b disp =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg b with
    | none => simp [ha] at h
    | some av =>
      simp only [ha] at h
      cases hrd : rd32 s.mem (ea av disp) with
      | none => simp [hrd] at h
      | some _ => simp [hrd] at h; cases h; rfl
  | ld8 _ b disp =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg b with
    | none => simp [ha] at h
    | some av =>
      simp only [ha] at h
      cases hrd : s.mem.rd (ea av disp) with
      | none => simp [hrd] at h
      | some _ => simp [hrd] at h; cases h; rfl
  | lea _ b _ =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg b with
    | none => simp [ha] at h
    | some _ => simp [ha] at h; cases h; rfl
  | jtab b i disp =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg b with
    | none => simp [ha] at h
    | some x =>
      simp only [ha] at h
      cases hb : s.reg i with
      | none => simp [hb] at h
      | some y =>
        simp only [hb] at h
        cases hrd : rd64 s.mem ((x + 8 * y + disp) % W) with
        | none => simp [hrd] at h
        | some _ => simp [hrd] at h; cases h; rfl
  | jmpR r =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg r with
    | none => simp [ha] at h
    | some _ => simp [ha] at h; cases h; rfl
  | st a b c => exact absurd rfl (hnost a b c).1
  | st8 a b c => exact absurd rfl (hnost a b c).2.1
  | push a => exact absurd rfl (hnost a 0 0).2.2.1
  | pushI a => exact absurd rfl (hnost a 0 0).2.2.2.1
  | call a => exact absurd rfl (hnost a 0 0).2.2.2.2.1
  | ret => exact absurd rfl (hnost 0 0 0).2.2.2.2.2.1
  | pop a => exact absurd rfl (hnost a 0 0).2.2.2.2.2.2.1
  | syscall => exact absurd rfl (hnost 0 0 0).2.2.2.2.2.2.2

private theorem exec_preserves_sp_of_ne4 {ins : Ins} {len : Nat} {s s' : St}
    (h : exec ins len s = .ok s') (hno4 : noWriteSp ins = true)
    (hnost : ∀ a b c, ins ≠ .st a b c ∧ ins ≠ .st8 a b c ∧ ins ≠ .push a ∧
      ins ≠ .pushI a ∧ ins ≠ .call a ∧ ins ≠ .ret ∧ ins ≠ .pop a ∧ ins ≠ .syscall) :
    s'.reg 4 = s.reg 4 := by
  have hdne : ∀ {d : Nat}, (d != 4) = true → d ≠ 4 := fun h => bne_iff_ne.1 h
  cases ins with
  | jmp _ => simp [exec] at h; cases h; rfl
  | jcc cc _ =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases hc : X86Int.cond s.fl cc with
    | none => simp [hc] at h
    | some _ => simp [hc] at h; cases h; rfl
  | movI d _ => simp [exec] at h; cases h; exact reg_set_ne (hdne hno4)
  | mov d r =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases hr : s.reg r with
    | none => simp [hr] at h
    | some _ => simp [hr] at h; cases h; exact reg_set_ne (hdne hno4)
  | mov32 d r =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases hr : s.reg r with
    | none => simp [hr] at h
    | some _ => simp [hr] at h; cases h; exact reg_set_ne (hdne hno4)
  | alu _ d r =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg d with
    | none => simp [ha] at h
    | some _ =>
      simp only [ha] at h
      cases hb : s.reg r with
      | none => simp [hb] at h
      | some _ => simp [hb] at h; cases h; exact reg_set_ne (hdne hno4)
  | aluI _ d _ =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg d with
    | none => simp [ha] at h
    | some _ => simp [ha] at h; cases h; exact reg_set_ne (hdne hno4)
  | cmp a b =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg a with
    | none => simp [ha] at h
    | some _ =>
      simp only [ha] at h
      cases hb : s.reg b with
      | none => simp [hb] at h
      | some _ => simp [hb] at h; cases h; rfl
  | cmpI a _ =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg a with
    | none => simp [ha] at h
    | some _ => simp [ha] at h; cases h; rfl
  | test a b =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg a with
    | none => simp [ha] at h
    | some _ =>
      simp only [ha] at h
      cases hb : s.reg b with
      | none => simp [hb] at h
      | some _ => simp [hb] at h; cases h; rfl
  | shlI d _ =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg d with
    | none => simp [ha] at h
    | some _ => simp [ha] at h; cases h; exact reg_set_ne (hdne hno4)
  | shrI d _ =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg d with
    | none => simp [ha] at h
    | some _ => simp [ha] at h; cases h; exact reg_set_ne (hdne hno4)
  | ld d b disp =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg b with
    | none => simp [ha] at h
    | some av =>
      simp only [ha] at h
      cases hrd : rd64 s.mem (ea av disp) with
      | none => simp [hrd] at h
      | some _ => simp [hrd] at h; cases h; exact reg_set_ne (hdne hno4)
  | ld32 d b disp =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg b with
    | none => simp [ha] at h
    | some av =>
      simp only [ha] at h
      cases hrd : rd32 s.mem (ea av disp) with
      | none => simp [hrd] at h
      | some _ => simp [hrd] at h; cases h; exact reg_set_ne (hdne hno4)
  | ld8 d b disp =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg b with
    | none => simp [ha] at h
    | some av =>
      simp only [ha] at h
      cases hrd : s.mem.rd (ea av disp) with
      | none => simp [hrd] at h
      | some _ => simp [hrd] at h; cases h; exact reg_set_ne (hdne hno4)
  | lea d b _ =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg b with
    | none => simp [ha] at h
    | some _ => simp [ha] at h; cases h; exact reg_set_ne (hdne hno4)
  | jtab b i disp =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg b with
    | none => simp [ha] at h
    | some x =>
      simp only [ha] at h
      cases hb : s.reg i with
      | none => simp [hb] at h
      | some y =>
        simp only [hb] at h
        cases hrd : rd64 s.mem ((x + 8 * y + disp) % W) with
        | none => simp [hrd] at h
        | some _ => simp [hrd] at h; cases h; rfl
  | jmpR r =>
    simp only [exec, lift, bind, Option.bind, pure] at h
    cases ha : s.reg r with
    | none => simp [ha] at h
    | some _ => simp [ha] at h; cases h; rfl
  | st a b c => exact absurd rfl (hnost a b c).1
  | st8 a b c => exact absurd rfl (hnost a b c).2.1
  | push a => exact absurd rfl (hnost a 0 0).2.2.1
  | pushI a => exact absurd rfl (hnost a 0 0).2.2.2.1
  | call a => exact absurd rfl (hnost a 0 0).2.2.2.2.1
  | ret => exact absurd rfl (hnost 0 0 0).2.2.2.2.2.1
  | pop a => exact absurd rfl (hnost a 0 0).2.2.2.2.2.2.1
  | syscall => exact absurd rfl (hnost 0 0 0).2.2.2.2.2.2.2

theorem regDef_flipReg {s : St} (hd : RegDefExcept11 s) (r b : Nat) :
    RegDefExcept11 (s.flipReg r b) := by
  refine ⟨by simp [St.flipReg, St.set, hd.1], fun j hj h11 => ?_⟩
  by_cases hjr : j = r
  · subst hjr
    simp only [St.flipReg, St.reg, St.set, List.getElem?_set_self (by rw [hd.1]; exact hj),
      Option.getD_some, Option.isSome_map]
    exact hd.2 j hj h11
  · simp only [St.flipReg, reg_set_ne (Ne.symm hjr)]
    exact hd.2 j hj h11

theorem regDef_flipMem {s : St} (hd : RegDefExcept11 s) (i j b : Nat) :
    RegDefExcept11 (s.flipMem i j b) := hd

theorem regDef_flagFlips {s s' : St} (hd : RegDefExcept11 s) (hfl : s' ∈ flagFlips s) :
    RegDefExcept11 s' := by
  simp only [flagFlips, List.mem_append, List.mem_map, Option.mem_toList] at hfl
  rcases hfl with (((⟨_, _, rfl⟩ | ⟨_, _, rfl⟩) | ⟨_, _, rfl⟩) | ⟨_, _, rfl⟩) <;> exact hd

def MemBytesLt256 (m : Mem) : Prop := ∀ r ∈ m, ∀ x ∈ r.data, x < 256

private theorem mem_rd_lt_256 : ∀ {m : Mem} {i0 a b : Nat},
    MemBytesLt256 m → (do let (_, r) ← Mem.find m a i0; r.data[a - r.base]?) = some b → b < 256
  | [], _, _, _, _, h => by simp [Mem.find] at h
  | r :: rest, i0, a, b, hm, h => by
    simp only [Mem.find] at h
    by_cases hcond : r.base ≤ a ∧ a < r.hi
    · rw [if_pos hcond] at h
      simp only [bind, Option.bind] at h
      exact hm r (List.mem_cons_self ..) b (List.mem_of_getElem? h)
    · rw [if_neg hcond] at h
      exact mem_rd_lt_256 (fun r' hr' => hm r' (List.mem_cons_of_mem _ hr')) h

private theorem leVal_lt_pow2 : ∀ (xs : List Nat), (∀ b ∈ xs, b < 256) → leVal xs < 256 ^ xs.length
  | [], _ => by simp [leVal]
  | b :: bs, hb => by
    have hb0 : b < 256 := hb b (List.mem_cons_self ..)
    have ih := leVal_lt_pow2 bs (fun x hx => hb x (List.mem_cons_of_mem _ hx))
    simp only [leVal, List.length_cons, Nat.pow_succ]
    omega

private theorem rdN_bytes_lt_256 : ∀ (m : Mem) (a n : Nat) (bs : List Nat),
    MemBytesLt256 m → m.rdN a n = some bs → (∀ b ∈ bs, b < 256) ∧ bs.length = n
  | _, _, 0, bs, _, h => by simp [Mem.rdN] at h; subst h; exact ⟨(fun _ hx => by cases hx), rfl⟩
  | m, a, n + 1, bs, hm, h => by
    simp only [Mem.rdN, bind, Option.bind, pure] at h
    cases hrd : m.rd a with
    | none => rw [hrd] at h; cases h
    | some b0 =>
      rw [hrd] at h
      cases hrest : m.rdN (a + 1) n with
      | none => rw [hrest] at h; cases h
      | some bs' =>
        rw [hrest] at h
        have hbs : bs = b0 :: bs' := by simpa using h.symm
        subst hbs
        have ih := rdN_bytes_lt_256 m (a + 1) n bs' hm hrest
        refine ⟨fun x hx => ?_, by simp [ih.2]⟩
        rcases List.mem_cons.1 hx with rfl | hx'
        · exact mem_rd_lt_256 hm hrd
        · exact ih.1 x hx'

theorem rd64_lt_W {m : Mem} {a v : Nat} (hm : MemBytesLt256 m) (h : rd64 m a = some v) : v < W := by
  simp only [rd64, Option.map_eq_some_iff] at h
  obtain ⟨bs, hbs, rfl⟩ := h
  have ⟨hby, hlen⟩ := rdN_bytes_lt_256 m a 8 bs hm hbs
  have hlt := leVal_lt_pow2 bs hby
  rw [hlen] at hlt
  have hpow : (256 : Nat) ^ 8 = W := by decide
  omega

theorem rd32_lt_W {m : Mem} {a v : Nat} (hm : MemBytesLt256 m) (h : rd32 m a = some v) : v < W := by
  simp only [rd32, Option.map_eq_some_iff] at h
  obtain ⟨bs, hbs, rfl⟩ := h
  have ⟨hby, hlen⟩ := rdN_bytes_lt_256 m a 4 bs hm hbs
  have hlt := leVal_lt_pow2 bs hby
  rw [hlen] at hlt
  have hpow : (256 : Nat) ^ 4 < W := by unfold W; decide
  omega

private theorem mem_modifyTailIdx_go {α : Type} (f : α → α) :
    ∀ (k : Nat) (l : List α) {x : α}, x ∈ List.modifyTailIdx.go (List.modifyHead f) k l →
      ∃ x0 ∈ l, x = x0 ∨ x = f x0
  | 0, [], _, hx => by cases hx
  | _ + 1, [], _, hx => by cases hx
  | 0, y :: ys, _, hx => by
    simp only [List.modifyTailIdx.go, List.modifyHead, List.mem_cons] at hx
    rcases hx with rfl | hy
    · exact ⟨_, List.mem_cons_self .., Or.inr rfl⟩
    · exact ⟨_, List.mem_cons_of_mem _ hy, Or.inl rfl⟩
  | k + 1, y :: ys, _, hx => by
    simp only [List.modifyTailIdx.go, List.mem_cons] at hx
    rcases hx with rfl | hy
    · exact ⟨_, List.mem_cons_self .., Or.inl rfl⟩
    · obtain ⟨x0, hx0, hres⟩ := mem_modifyTailIdx_go f k ys hy
      exact ⟨x0, List.mem_cons_of_mem _ hx0, hres⟩

private theorem mem_modify_list {α : Type} (f : α → α) (l : List α) (k : Nat) {x : α}
    (hx : x ∈ l.modify k f) : ∃ x0 ∈ l, x = x0 ∨ x = f x0 :=
  mem_modifyTailIdx_go f k l hx

private theorem list_set_of_le {α : Type} (v : α) :
    ∀ (l : List α) (i : Nat), l.length ≤ i → l.set i v = l
  | [], _, _ => rfl
  | _ :: _, 0, h => by simp at h
  | x :: xs, i + 1, h => by
    simp only [List.set, List.length_cons] at h ⊢
    congr 1; exact list_set_of_le v xs i (by omega)

theorem memBytesLt256_flipMem {m : Mem} (hm : MemBytesLt256 m) (i j b : Nat) (hb : b < 8) :
    MemBytesLt256 (m.modify i fun r => { r with data := r.data.modify j fun v => Nat.xor v (2 ^ b) }) := by
  intro r hr x hx
  obtain ⟨r0, hr0, rfl | hreq⟩ := mem_modify_list _ m i hr
  · exact hm _ hr0 x hx
  · subst hreq
    dsimp only at hx
    obtain ⟨x0, hx0m, rfl | hxeq⟩ := mem_modify_list _ r0.data j hx
    · exact hm r0 hr0 _ hx0m
    · subst hxeq
      have hx0lt : x0 < 2 ^ 8 := hm r0 hr0 x0 hx0m
      have h2b : 2 ^ b < 2 ^ 8 := Nat.pow_lt_pow_right (by decide) hb
      exact Nat.xor_lt_two_pow hx0lt h2b

def RegsLtW (s : St) : Prop := ∀ j v, s.reg j = some v → v < W

theorem regsLtW_set {s : St} (hs : RegsLtW s) (i : Nat) {v : Nat} (hv : v < W) :
    RegsLtW (s.set i (some v)) := by
  intro j w hw
  by_cases hij : i = j
  · subst hij
    by_cases hil : i < s.regs.length
    · rw [reg_set_eq hil] at hw; cases hw; exact hv
    · simp only [St.set, St.reg, list_set_of_le _ _ _ (Nat.le_of_not_lt hil)] at hw
      exact hs i w hw
  · rw [reg_set_ne hij] at hw
    exact hs j w hw

theorem regsLtW_flipReg {s : St} (hs : RegsLtW s) (r b : Nat) (hb : b < 64) :
    RegsLtW (s.flipReg r b) := by
  intro j w hw
  by_cases hrj : r = j
  · subst hrj
    by_cases hrl : r < s.regs.length
    · simp only [St.flipReg, reg_set_eq hrl, Option.map_eq_some_iff] at hw
      obtain ⟨v0, hv0, rfl⟩ := hw
      exact Nat.xor_lt_two_pow (hs r v0 hv0) (Nat.pow_lt_pow_right (by decide) hb)
    · simp only [St.flipReg, St.set, St.reg, list_set_of_le _ _ _ (Nat.le_of_not_lt hrl)] at hw
      exact hs r w hw
  · simp only [St.flipReg, reg_set_ne hrj] at hw
    exact hs j w hw

private theorem aluF_lt_W (op : Alu) (a b : Nat) (ha : a < W) (hb : b < W) : (aluF op a b).1 < W := by
  cases op with
  | add => simp only [aluF]; exact Nat.mod_lt _ (by unfold W; decide)
  | sub => simp only [aluF]; exact Nat.mod_lt _ (by unfold W; decide)
  | and => exact Nat.lt_of_le_of_lt Nat.and_le_left ha
  | or => exact Nat.or_lt_two_pow ha hb
  | xor => exact Nat.xor_lt_two_pow ha hb

private theorem exec_preserves_regsLtW {ins : Ins} {len : Nat} {s s' : St}
    (hs : RegsLtW s) (hm : MemBytesLt256 s.mem)
    (hnost : ∀ a b c, ins ≠ .st a b c ∧ ins ≠ .st8 a b c ∧ ins ≠ .push a ∧
      ins ≠ .pushI a ∧ ins ≠ .call a ∧ ins ≠ .ret ∧ ins ≠ .pop a ∧ ins ≠ .syscall)
    (he : exec ins len s = .ok s') : RegsLtW s' := by
  have hW_pos : 0 < W := by unfold W; decide
  have h232 : 2 ^ 32 < W := by unfold W; decide
  have h256 : 256 < W := by unfold W; decide
  cases ins with
  | jmp _ => simp [exec] at he; cases he; exact hs
  | jcc cc _ =>
    simp only [exec, lift, bind, Option.bind, pure] at he
    cases hc : X86Int.cond s.fl cc with
    | none => simp [hc] at he
    | some _ => simp [hc] at he; cases he; exact hs
  | movI d imm =>
    simp [exec] at he; cases he
    exact regsLtW_set hs d (Nat.mod_lt _ hW_pos)
  | mov d r =>
    simp only [exec, lift, bind, Option.bind, pure] at he
    cases hr : s.reg r with
    | none => simp [hr] at he
    | some _ => simp [hr] at he; cases he; exact regsLtW_set hs d (hs r _ hr)
  | mov32 d r =>
    simp only [exec, lift, bind, Option.bind, pure] at he
    cases hr : s.reg r with
    | none => simp [hr] at he
    | some _ =>
      simp [hr] at he; cases he
      exact regsLtW_set hs d (Nat.lt_trans (Nat.mod_lt _ (by decide)) h232)
  | alu op d r =>
    simp only [exec, lift, bind, Option.bind, pure] at he
    cases ha : s.reg d with
    | none => simp [ha] at he
    | some _ =>
      simp only [ha] at he
      cases hb : s.reg r with
      | none => simp [hb] at he
      | some _ =>
        simp [hb] at he; cases he
        exact regsLtW_set hs d (aluF_lt_W op _ _ (hs d _ ha) (hs r _ hb))
  | aluI op d imm =>
    simp only [exec, lift, bind, Option.bind, pure] at he
    cases ha : s.reg d with
    | none => simp [ha] at he
    | some _ =>
      simp [ha] at he; cases he
      exact regsLtW_set hs d (aluF_lt_W op _ _ (hs d _ ha) (Nat.mod_lt _ hW_pos))
  | cmp a b =>
    simp only [exec, lift, bind, Option.bind, pure] at he
    cases ha : s.reg a with
    | none => simp [ha] at he
    | some _ =>
      simp only [ha] at he
      cases hb : s.reg b with
      | none => simp [hb] at he
      | some _ => simp [hb] at he; cases he; exact hs
  | cmpI a _ =>
    simp only [exec, lift, bind, Option.bind, pure] at he
    cases ha : s.reg a with
    | none => simp [ha] at he
    | some _ => simp [ha] at he; cases he; exact hs
  | test a b =>
    simp only [exec, lift, bind, Option.bind, pure] at he
    cases ha : s.reg a with
    | none => simp [ha] at he
    | some _ =>
      simp only [ha] at he
      cases hb : s.reg b with
      | none => simp [hb] at he
      | some _ => simp [hb] at he; cases he; exact hs
  | shlI d _ =>
    simp only [exec, lift, bind, Option.bind, pure] at he
    cases ha : s.reg d with
    | none => simp [ha] at he
    | some _ => simp [ha] at he; cases he; exact regsLtW_set hs d (Nat.mod_lt _ hW_pos)
  | shrI d _ =>
    simp only [exec, lift, bind, Option.bind, pure] at he
    cases ha : s.reg d with
    | none => simp [ha] at he
    | some _ =>
      simp [ha] at he; cases he
      exact regsLtW_set hs d (Nat.lt_of_le_of_lt (Nat.div_le_self _ _) (hs d _ ha))
  | ld d b disp =>
    simp only [exec, lift, bind, Option.bind, pure] at he
    cases ha : s.reg b with
    | none => simp [ha] at he
    | some av =>
      simp only [ha] at he
      cases hrd : rd64 s.mem (ea av disp) with
      | none => simp [hrd] at he
      | some _ => simp [hrd] at he; cases he; exact regsLtW_set hs d (rd64_lt_W hm hrd)
  | ld32 d b disp =>
    simp only [exec, lift, bind, Option.bind, pure] at he
    cases ha : s.reg b with
    | none => simp [ha] at he
    | some av =>
      simp only [ha] at he
      cases hrd : rd32 s.mem (ea av disp) with
      | none => simp [hrd] at he
      | some _ => simp [hrd] at he; cases he; exact regsLtW_set hs d (rd32_lt_W hm hrd)
  | ld8 d b disp =>
    simp only [exec, lift, bind, Option.bind, pure] at he
    cases ha : s.reg b with
    | none => simp [ha] at he
    | some av =>
      simp only [ha] at he
      cases hrd : s.mem.rd (ea av disp) with
      | none => simp [hrd] at he
      | some _ =>
        simp [hrd] at he; cases he
        exact regsLtW_set hs d (Nat.lt_trans (mem_rd_lt_256 hm hrd) h256)
  | lea d b _ =>
    simp only [exec, lift, bind, Option.bind, pure] at he
    cases ha : s.reg b with
    | none => simp [ha] at he
    | some _ => simp [ha] at he; cases he; exact regsLtW_set hs d (Nat.mod_lt _ hW_pos)
  | jtab b i disp =>
    simp only [exec, lift, bind, Option.bind, pure] at he
    cases ha : s.reg b with
    | none => simp [ha] at he
    | some x =>
      simp only [ha] at he
      cases hb : s.reg i with
      | none => simp [hb] at he
      | some y =>
        simp only [hb] at he
        cases hrd : rd64 s.mem ((x + 8 * y + disp) % W) with
        | none => simp [hrd] at he
        | some _ => simp [hrd] at he; cases he; exact hs
  | jmpR r =>
    simp only [exec, lift, bind, Option.bind, pure] at he
    cases ha : s.reg r with
    | none => simp [ha] at he
    | some _ => simp [ha] at he; cases he; exact hs
  | st a b c => exact absurd rfl (hnost a b c).1
  | st8 a b c => exact absurd rfl (hnost a b c).2.1
  | push a => exact absurd rfl (hnost a 0 0).2.2.1
  | pushI a => exact absurd rfl (hnost a 0 0).2.2.2.1
  | call a => exact absurd rfl (hnost a 0 0).2.2.2.2.1
  | ret => exact absurd rfl (hnost 0 0 0).2.2.2.2.2.1
  | pop a => exact absurd rfl (hnost a 0 0).2.2.2.2.2.2.1
  | syscall => exact absurd rfl (hnost 0 0 0).2.2.2.2.2.2.2

/-! ### Part 3: `BinStackSafe`, `BinTailSafe`, and 4-channel tail immunity on `im` -/

def BinStackSafe (s : St) : Prop :=
  ∀ sp, s.reg 4 = some sp → ∀ t, rd64 s.mem sp = some t →
    ¬ (L_vote ≤ t ∧ t < L_core_end) ∧ im t = none

theorem BinStackSafe_to_TailStackSafe {s : St} (h : BinStackSafe s) :
    SporosCheckFlat.TailStackSafe L_vote s := by
  intro sp hsp t hrd
  have h1 := (h sp hsp t hrd).1
  have hce : L_core_end = L_vote + 143 := by decide
  rwa [← hce]

def BinTailSafe (s : St) : Prop :=
  s.rip = 4204869 ∨
  im s.rip = none ∨
  (L_vote ≤ s.rip ∧ s.rip < L_core_end ∧ SporosCheckFlat.TailSafe L_vote s ∧
    (s.rip - L_vote ∈ [0x14, 0x2e, 0x3d] → BinStackSafe s))

theorem BinTailSafe_step {s : St} (hs : BinTailSafe s) :
    match step im s with
    | .ok s' => BinTailSafe s'
    | .exit c => c ≠ 0
    | .stuck => True := by
  rcases hs with hr | hnone | ⟨hlo, hhi, hts, hstk⟩
  · have ⟨h0, h1, h2⟩ := im_4204869_spec
    simp only [step, hr, h0, exec, lift, bind, Option.bind, pure]
    cases hc : X86Int.cond s.fl 0 with
    | none => exact trivial
    | some c =>
      cases c with
      | true => exact Or.inr (Or.inl h2)
      | false =>
        simp only [Bool.false_eq_true, ↓reduceIte]
        have hrip' : (4204869 + 2 : Nat) = 4204871 := rfl
        rw [hrip']
        exact Or.inr (Or.inl h1)
  · simp only [step, hnone]
  · have him_eq : im s.rip = SporosCheckFlat.tailImAt L_vote s.rip := im_vote_eq_tailImAt hlo hhi
    have him_step : step im s = X86Int.step (SporosCheckFlat.tailImAt L_vote) s := by
      simp only [step, X86Int.step, him_eq]
    have htstep := SporosCheckFlat.TailSafe_step L_vote_base_ok hts
    rw [him_step]
    cases hst : X86Int.step (SporosCheckFlat.tailImAt L_vote) s with
    | stuck => exact trivial
    | exit c => rw [hst] at htstep; exact htstep
    | ok s' =>
      rw [hst] at htstep
      dsimp only at htstep ⊢
      have hsrip : s.rip = L_vote + (s.rip - L_vote) := (Nat.add_sub_cancel' hlo).symm
      have hoff : s.rip - L_vote < 143 := by unfold L_vote L_core_end at *; omega
      have hcov := List.all_eq_true.1 SporosCheckFlat.tailAllOffs_cover (s.rip - L_vote) (List.mem_range.2 hoff)
      simp only [Bool.or_eq_true, List.any_eq_true] at hcov
      rcases hcov with htnone | ⟨e, hemem, heeq⟩
      · have h0n : SporosCheckFlat.tailImAt 0 (s.rip - L_vote) = none := Option.isNone_iff_eq_none.1 htnone
        have hnb : SporosCheckFlat.tailImAt L_vote s.rip = none := by
          rw [hsrip, SporosCheckFlat.tailImAt_eq, h0n]; rfl
        simp [X86Int.step, hnb] at hst
      · have he1 : e.1 = s.rip - L_vote := beq_iff_eq.1 heeq
        have hspec_e := SporosCheckFlat.tailImAt_base_all (base := L_vote) hemem
        rw [he1, ← hsrip] at hspec_e
        simp only [X86Int.step, hspec_e] at hst
        have haud := List.all_eq_true.1 tailAllSpec_no_jump_out_and_no_ret_step e hemem
        have hrc := exec_rip_cases hst
        rcases e with ⟨eoff, eins, elen⟩
        dsimp only at he1 haud hspec_e
        have hce : L_core_end = L_vote + 143 := by decide
        have hW_v : L_vote + 143 < W := by unfold L_vote W; decide
        cases eins with
        | jmp t =>
          dsimp only [ripNextCond, SporosCheckFlat.shiftIns] at hrc haud
          simp only [decide_eq_true_eq] at haud
          have hm : (L_vote + t) % W = L_vote + t := Nat.mod_eq_of_lt (by omega)
          rw [hm] at hrc
          refine Or.inr (Or.inr ⟨by omega, by omega, htstep, fun hmem => ?_⟩)
          rw [hrc, Nat.add_sub_cancel_left] at hmem
          exact absurd hmem haud.2
        | jcc cc t =>
          dsimp only [ripNextCond, SporosCheckFlat.shiftIns] at hrc haud
          simp only [decide_eq_true_eq] at haud
          rcases hrc with hrc1 | hrc2
          · have hm : (L_vote + t) % W = L_vote + t := Nat.mod_eq_of_lt (by omega)
            rw [hm] at hrc1
            refine Or.inr (Or.inr ⟨by omega, by omega, htstep, fun hmem => ?_⟩)
            rw [hrc1, Nat.add_sub_cancel_left] at hmem
            exact absurd hmem haud.2.1
          · refine Or.inr (Or.inr ⟨by omega, by omega, htstep, fun hmem => ?_⟩)
            rw [hrc2, show s.rip + elen - L_vote = eoff + elen by omega] at hmem
            exact absurd hmem haud.2.2.2
        | call _ | jtab _ _ _ | jmpR _ => simp at haud
        | ret =>
          dsimp only [ripNextCond, SporosCheckFlat.shiftIns] at hrc haud
          simp only [decide_eq_true_eq] at haud
          rcases hrc with ⟨sp, t, hsp, hrd, hrip'⟩
          unfold BinTailSafe
          rw [hrip']
          exact Or.inr (Or.inl (hstk (by rw [← he1]; exact haud) sp hsp t hrd).2)
        | mov _ _ | mov32 _ _ | movI _ _ | alu _ _ _ | aluI _ _ _ | cmp _ _ | cmpI _ _
        | test _ _ | shlI _ _ | shrI _ _ | ld _ _ _ | st _ _ _ | ld32 _ _ _ | ld8 _ _ _
        | st8 _ _ _ | lea _ _ _ | pushI _ | push _ | pop _ | syscall =>
          dsimp only [ripNextCond, SporosCheckFlat.shiftIns] at hrc
          simp only [decide_eq_true_eq] at haud
          by_cases h143 : eoff + elen = 143
          · have : s'.rip = L_core_end := by omega
            unfold BinTailSafe
            rw [this]
            exact Or.inr (Or.inl im_core_end_none)
          · have hlt : eoff + elen < 143 := by omega
            refine Or.inr (Or.inr ⟨by omega, by omega, htstep, fun hmem => ?_⟩)
            rw [hrc, show s.rip + elen - L_vote = eoff + elen by omega] at hmem
            exact absurd hmem haud.2

theorem BinTailSafe_run : ∀ (m : Nat) {s : St}, BinTailSafe s → run im m (.ok s) ≠ .exit 0
  | 0, _, _ => by simp [run]
  | m + 1, s, hs => by
    have hst := BinTailSafe_step hs
    simp only [run]
    cases h0 : step im s with
    | stuck => rw [run_stuck]; intro h; cases h
    | exit c => rw [h0] at hst; rw [run_exit]; intro h; cases h; exact hst rfl
    | ok s' => rw [h0] at hst; exact BinTailSafe_run m hst

def BinPreFault (v : Nat) (s : St) : Prop :=
  SporosCheckFlat.PreFault L_vote v s ∧ BinStackSafe s

theorem BinPreFault_init (s0 : St) (hst : BinStackSafe s0) {v : Nat}
    (hv : SporosCheckFlat.SafeInvalid v) :
    BinPreFault v (SporosCheckFlat.tailInit L_vote s0 v v v) := by
  refine ⟨⟨hv, rfl, ?_, Nat.le_refl _, ?_, ?_⟩, ?_⟩
  · exact BinStackSafe_to_TailStackSafe hst
  · dsimp only [SporosCheckFlat.tailInit]; unfold L_vote; decide
  · change SporosCheckFlat.PreFaultOff v _ (L_vote - L_vote)
    rw [Nat.sub_self L_vote]
    exact ⟨rfl, rfl, rfl, (fun h => by cases h), (fun h => by cases h), (fun h => by cases h),
      (fun h => by simp at h)⟩
  · intro sp hsp t hrd
    have hsp0 : s0.reg 4 = some sp := by
      simpa [SporosCheckFlat.tailInit, TMRVoter.voterInit, St.reg, TMRVoter.voterRegs] using hsp
    exact hst sp hsp0 t hrd

private theorem isNonStoreIns_shiftIns (base : Nat) {ins : Ins} (h : isNonStoreIns ins = true) :
    isNonStoreIns (SporosCheckFlat.shiftIns base ins) = true := by
  cases ins <;> simp [SporosCheckFlat.shiftIns, isNonStoreIns] at h ⊢

private theorem noWriteSp_shiftIns (base : Nat) (ins : Ins) :
    noWriteSp (SporosCheckFlat.shiftIns base ins) = noWriteSp ins := by
  cases ins <;> rfl

private theorem exec_preserves_binStackSafe_of_pre {base : Nat} {ins0 : Ins} {len : Nat} {s s' : St}
    (hst' : exec (SporosCheckFlat.shiftIns base ins0) len s = .ok s')
    (h1 : isNonStoreIns ins0 = true)
    (h2 : noWriteSp ins0 = true) (hs2 : BinStackSafe s) : BinStackSafe s' := by
  have hnost_sh := isNonStoreIns_spec (isNonStoreIns_shiftIns base h1)
  have hno4_sh : noWriteSp (SporosCheckFlat.shiftIns base ins0) = true := by
    rw [noWriteSp_shiftIns]; exact h2
  have hm := exec_preserves_mem_non_store hst' hnost_sh
  have hsp := exec_preserves_sp_of_ne4 hst' hno4_sh hnost_sh
  intro sp hsp' t hrd
  rw [hsp] at hsp'; rw [hm] at hrd
  exact hs2 sp hsp' t hrd

private theorem exec_syscall_60_not_ok {len : Nat} {s s' : St}
    (h0 : s.reg 0 = some 60) (h : exec .syscall len s = .ok s') : False := by
  dsimp only [exec] at h
  split at h
  · rename_i nr a1 a2 a3 a4 a5 a6 he0 _ _ _ _ _ _
    rw [h0] at he0; injection he0 with hnr; subst hnr
    simp [syscall] at h
  · cases h

private theorem BinPreFault_step_gen {base v : Nat} (hb : SporosCheckFlat.base_ok base)
    (him : ∀ a, base ≤ a → a < base + 143 → im a = SporosCheckFlat.tailImAt base a)
    (hs1 : SporosCheckFlat.PreFault base v s) (hs2 : BinStackSafe s) :
    match step im s with
    | .ok s' => SporosCheckFlat.PreFault base v s' ∧ BinStackSafe s'
    | .exit c => c ≠ 0
    | .stuck => True := by
  have h1 := SporosCheckFlat.PreFault_step hb hs1
  have him_eq := him s.rip hs1.2.2.2.1 hs1.2.2.2.2.1
  have him_step : step im s = X86Int.step (SporosCheckFlat.tailImAt base) s := by
    simp only [step, X86Int.step, him_eq]
  rw [him_step]
  cases hst : X86Int.step (SporosCheckFlat.tailImAt base) s with
  | stuck => exact trivial
  | exit c => rw [hst] at h1; exact h1
  | ok s' =>
    rw [hst] at h1
    dsimp only at h1 ⊢
    refine ⟨h1, ?_⟩
    have hmem_pre := preFault_mem_of_base hs1
    have hlo := hs1.2.2.2.1
    have hsrip : s.rip = base + (s.rip - base) := (Nat.add_sub_cancel' hlo).symm
    have hst' : X86Int.step (SporosCheckFlat.tailImAt base) s = .ok s' := hst
    rw [X86Int.step, hsrip, SporosCheckFlat.tailImAt_eq] at hst'
    have hty := List.all_eq_true.1 preFault_no_st_0 (s.rip - base) hmem_pre
    cases him0 : SporosCheckFlat.tailImAt 0 (s.rip - base) with
    | none => simp [him0] at hst'
    | some p0 =>
      rcases p0 with ⟨ins0, len⟩
      simp only [him0, Option.map_some, Bool.and_eq_true] at hst' hty
      cases ins0 with
      | syscall =>
        have hs_p := List.all_eq_true.1 preFault_sys_only_0 (s.rip - base) hmem_pre
        simp only [him0, decide_eq_true_eq] at hs_p
        have h0_60 := preFault_0x7c_or_0x8d_reg0 hs1 hs_p
        dsimp only [SporosCheckFlat.shiftIns] at hst'
        exact False.elim (exec_syscall_60_not_ok h0_60 hst')
      | cmp _ _ | jcc _ _ | mov _ _ | jmp _ | movI _ _ | alu _ _ _ =>
        exact exec_preserves_binStackSafe_of_pre hst' rfl hty.2 hs2
      | _ => simp at hty

theorem BinPreFault_step {v : Nat} {s : St} (hs : BinPreFault v s) :
    match step im s with
    | .ok s' => BinPreFault v s'
    | .exit c => c ≠ 0
    | .stuck => True :=
  BinPreFault_step_gen L_vote_base_ok
    (fun a ha1 ha2 => im_vote_eq_tailImAt ha1 (by unfold L_vote L_core_end at *; omega))
    hs.1 hs.2

/-- Elementary step or physical disturbance against `SporosCheckBin.im` and `spr1Alts`. -/
inductive BinElemFaultStep (s : St) : Out → Prop where
  | clean : BinElemFaultStep s (step im s)
  | fetch {alt : Option (Ins × Nat)} :
      alt ∈ spr1Alts s.rip → BinElemFaultStep s (execO s alt)
  | reg {s1 s2 : St} :
      step im s = .ok s1 → s2 ∈ regFlips s1 → BinElemFaultStep s (.ok s2)
  | mem {s1 s2 : St} :
      step im s = .ok s1 → s2 ∈ memFlips s1 → BinElemFaultStep s (.ok s2)
  | flag {s1 s2 : St} :
      step im s = .ok s1 → s2 ∈ flagFlips s1 → BinElemFaultStep s (.ok s2)

def BinOutSafe : Out → Prop
  | .ok s' => BinTailSafe s'
  | .exit c => c ≠ 0
  | .stuck => True

private theorem execO_audit_nonStore_branch {s s' : St} {off len : Nat} {ins : Ins}
    (hsrip : s.rip = L_vote + off) (hstk : BinStackSafe s)
    (hex : exec ins len s = .ok s')
    (hrc : s'.rip = s.rip + len)
    (hns : isNonStoreIns ins = true)
    (hts : SporosCheckFlat.TailSafe L_vote s')
    (haud : (decide (off + len < 143) &&
      (if !noWriteSp ins || false then decide (off + len ∉ [0x14, 0x2e, 0x3d]) else true)) = true ∨
      isNoneAddr (L_vote + (off + len)) = true) : BinTailSafe s' := by
  unfold BinTailSafe
  rw [hrc, hsrip, Nat.add_assoc]
  rcases haud with haud1 | hn
  · simp only [Bool.and_eq_true, decide_eq_true_eq] at haud1
    by_cases hsp4 : noWriteSp ins = true
    · have hnost := isNonStoreIns_spec hns
      have hm := exec_preserves_mem_non_store hex hnost
      have hsp := exec_preserves_sp_of_ne4 hex hsp4 hnost
      exact Or.inr (Or.inr ⟨by unfold L_vote; omega, by unfold L_vote L_core_end; omega, hts,
        fun _ sp hsp' t hrd => by rw [hsp] at hsp'; rw [hm] at hrd; exact hstk sp hsp' t hrd⟩)
    · have hnot4 : (!noWriteSp ins || false) = true := by simp [hsp4]
      have hsafeMem := haud1.2
      simp only [hnot4, ↓reduceIte, decide_eq_true_eq] at hsafeMem
      exact Or.inr (Or.inr ⟨by unfold L_vote; omega, by unfold L_vote L_core_end; omega, hts, fun hm => by
        rw [Nat.add_sub_cancel_left] at hm
        exact absurd hm hsafeMem⟩)
  · exact Or.inr (Or.inl (im_eq_none_of_isNoneAddr hn))

private theorem execO_audit_to_binTailSafe {s s' : St} {off : Nat}
    (hoff_mem : off ∈ SporosCheckFlat.preFaultOffs)
    (hsrip : s.rip = L_vote + off) (hstk : BinStackSafe s)
    {alt : Option (Ins × Nat)} (halt : alt ∈ spr1Alts (L_vote + off))
    (hex : execO s alt = .ok s')
    (hts : SporosCheckFlat.TailSafe L_vote s') : BinTailSafe s' := by
  have halt' : alt ∈ tailWindowAlts off := by
    rw [← spr1Alts_eq_tailWindowAlts hoff_mem]; exact halt
  have haud := List.all_eq_true.1 (List.all_eq_true.1 tailAlts_pre_exit_audit off hoff_mem) alt halt'
  cases alt with
  | none => simp [execO] at hex
  | some p =>
    rcases p with ⟨ins, len⟩
    dsimp only [execO] at hex haud
    have hrc := exec_rip_cases hex
    have hce : L_core_end = L_vote + 143 := by decide
    cases ins with
    | ret | jmpR _ | call _ | jtab _ _ _ | pushI _ => simp at haud
    | push _ =>
      dsimp only [ripNextCond] at hrc
      simp only [Bool.and_eq_true, decide_eq_true_eq] at haud
      unfold BinTailSafe
      rw [hrc, hsrip]
      have hnone : im (L_vote + off + len) = none := by
        rw [Nat.add_assoc, im_vote_eq_tailImAt (by unfold L_vote; omega) (by unfold L_vote L_core_end; omega),
          ← Nat.add_assoc]
        exact Option.isNone_iff_eq_none.1 haud.2
      exact Or.inr (Or.inl hnone)
    | jmp t =>
      dsimp only [ripNextCond] at hrc
      simp only [Bool.or_eq_true, decide_eq_true_eq] at haud
      unfold BinTailSafe
      rw [hrc]
      rcases haud with (⟨h1, h2⟩ | h42) | hn
      · have hnost := isNonStoreIns_spec (ins := .jmp t) rfl
        have hm := exec_preserves_mem_non_store hex hnost
        have hsp := exec_preserves_sp_of_ne4 hex rfl hnost
        exact Or.inr (Or.inr ⟨h1, h2, hts, fun _ => fun sp hsp' t hrd => by
          rw [hsp] at hsp'; rw [hm] at hrd; exact hstk sp hsp' t hrd⟩)
      · exact Or.inl (beq_iff_eq.1 h42)
      · exact Or.inr (Or.inl (im_eq_none_of_isNoneAddr hn))
    | jcc cc t =>
      dsimp only [ripNextCond] at hrc
      simp only [Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq] at haud
      have hnost := isNonStoreIns_spec (ins := .jcc cc t) rfl
      have hm := exec_preserves_mem_non_store hex hnost
      have hsp := exec_preserves_sp_of_ne4 hex rfl hnost
      unfold BinTailSafe
      rcases hrc with hrc1 | hrc2
      · rw [hrc1]
        rcases haud.1 with (⟨h1, h2⟩ | h42) | hn
        · exact Or.inr (Or.inr ⟨h1, h2, hts, fun _ => fun sp hsp' t hrd => by
            rw [hsp] at hsp'; rw [hm] at hrd; exact hstk sp hsp' t hrd⟩)
        · exact Or.inl (beq_iff_eq.1 h42)
        · exact Or.inr (Or.inl (im_eq_none_of_isNoneAddr hn))
      · rw [hrc2, hsrip]
        rcases haud.2 with ⟨h1, h2⟩ | hn
        · exact Or.inr (Or.inr ⟨h1, h2, hts, fun _ => fun sp hsp' t hrd => by
            rw [hsp] at hsp'; rw [hm] at hrd; exact hstk sp hsp' t hrd⟩)
        · exact Or.inr (Or.inl (im_eq_none_of_isNoneAddr hn))
    | st _ _ _ | st8 _ _ _ | pop _ | syscall =>
      dsimp only [ripNextCond] at hrc
      simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at haud
      unfold BinTailSafe
      rw [hrc, hsrip]
      rcases haud with ⟨_, hnr⟩ | hn
      · simp only [or_true, ↓reduceIte, decide_eq_true_eq] at hnr
        exact Or.inr (Or.inr ⟨by unfold L_vote; omega, by unfold L_vote L_core_end; omega, hts, fun hm => by
          rw [Nat.add_assoc, Nat.add_sub_cancel_left] at hm
          exact absurd hm hnr⟩)
      · rw [Nat.add_assoc]
        exact Or.inr (Or.inl (im_eq_none_of_isNoneAddr hn))
    | mov _ _ | mov32 _ _ | movI _ _ | alu _ _ _ | aluI _ _ _ | cmp _ _ | cmpI _ _
    | test _ _ | shlI _ _ | shrI _ _ | ld _ _ _ | ld32 _ _ _ | ld8 _ _ _ | lea _ _ _ =>
      dsimp only [ripNextCond] at hrc
      rw [Bool.or_eq_true] at haud
      exact execO_audit_nonStore_branch hsrip hstk hex hrc rfl hts haud

theorem BinPreFault_elemFaultStep_safe {v : Nat} {s : St} (hs : BinPreFault v s)
    {o : Out} (hstep : BinElemFaultStep s o) : BinOutSafe o := by
  have hlo := hs.1.2.2.2.1
  have hhi : s.rip < L_core_end := by have := hs.1.2.2.2.2.1; unfold L_vote L_core_end at *; omega
  have him_eq := im_vote_eq_tailImAt hlo hhi
  have hmem_pre := preFault_mem_of_base hs.1
  cases hstep with
  | clean =>
    have hbp := BinPreFault_step hs
    cases hst : step im s with
    | stuck => exact trivial
    | exit c => rw [hst] at hbp; exact hbp
    | ok s' =>
      rw [hst] at hbp
      dsimp only at hbp
      have hts : SporosCheckFlat.TailSafe L_vote s' := by
        have hp' := SporosCheckFlat.PreFault_elemFaultStep_safe L_vote_base_ok hs.1
          SporosCheckFlat.ElemFaultStep.clean
        dsimp only [SporosCheckFlat.OutSafe] at hp'
        have him_step : step im s = X86Int.step (SporosCheckFlat.tailImAt L_vote) s := by
          simp only [step, X86Int.step, him_eq]
        rw [← him_step, hst] at hp'
        exact hp'
      exact Or.inr (Or.inr ⟨hbp.1.2.2.2.1, by have := hbp.1.2.2.2.2.1; unfold L_vote L_core_end at *; omega, hts,
        fun _ => hbp.2⟩)
  | @reg s1 s2 hclean hreg =>
    have hsp := hclean; rw [step, him_eq] at hsp
    have hts := SporosCheckFlat.PreFault_elemFaultStep_safe L_vote_base_ok hs.1
      (SporosCheckFlat.ElemFaultStep.reg hsp hreg)
    dsimp only [SporosCheckFlat.OutSafe] at hts
    have hbp := BinPreFault_step hs; rw [hclean] at hbp; dsimp only at hbp
    have hrip' : s1.rip - L_vote ∉ [0x14, 0x2e, 0x3d] :=
      of_decide_eq_true (List.all_eq_true.1 preFaultOff_not_ret _ (preFault_mem_of_base hbp.1))
    have hrip2 : s2.rip = s1.rip := by
      simp only [regFlips, List.mem_map] at hreg; rcases hreg with ⟨j, _, rfl⟩; rfl
    exact Or.inr (Or.inr ⟨by rw [hrip2]; exact hbp.1.2.2.2.1,
      by rw [hrip2]; have := hbp.1.2.2.2.2.1; unfold L_vote L_core_end at *; omega, hts,
      fun h => absurd (by rw [hrip2] at h; exact h) hrip'⟩)
  | @mem s1 s2 hclean hmem =>
    have hsp := hclean; rw [step, him_eq] at hsp
    have hts := SporosCheckFlat.PreFault_elemFaultStep_safe L_vote_base_ok hs.1
      (SporosCheckFlat.ElemFaultStep.mem hsp hmem)
    dsimp only [SporosCheckFlat.OutSafe] at hts
    have hbp := BinPreFault_step hs; rw [hclean] at hbp; dsimp only at hbp
    have hrip' : s1.rip - L_vote ∉ [0x14, 0x2e, 0x3d] :=
      of_decide_eq_true (List.all_eq_true.1 preFaultOff_not_ret _ (preFault_mem_of_base hbp.1))
    have hrip2 : s2.rip = s1.rip := by
      simp only [memFlips, List.mem_flatMap, List.mem_map] at hmem
      rcases hmem with ⟨_, _, _, _, _, _, rfl⟩; rfl
    exact Or.inr (Or.inr ⟨by rw [hrip2]; exact hbp.1.2.2.2.1,
      by rw [hrip2]; have := hbp.1.2.2.2.2.1; unfold L_vote L_core_end at *; omega, hts,
      fun h => absurd (by rw [hrip2] at h; exact h) hrip'⟩)
  | @flag s1 s2 hclean hfl =>
    have hsp := hclean; rw [step, him_eq] at hsp
    have hts := SporosCheckFlat.PreFault_elemFaultStep_safe L_vote_base_ok hs.1
      (SporosCheckFlat.ElemFaultStep.flag hsp hfl)
    dsimp only [SporosCheckFlat.OutSafe] at hts
    have hbp := BinPreFault_step hs; rw [hclean] at hbp; dsimp only at hbp
    have hrip' : s1.rip - L_vote ∉ [0x14, 0x2e, 0x3d] :=
      of_decide_eq_true (List.all_eq_true.1 preFaultOff_not_ret _ (preFault_mem_of_base hbp.1))
    have hrip2 : s2.rip = s1.rip := by
      simp only [flagFlips, List.mem_append, List.mem_map, Option.mem_toList] at hfl
      rcases hfl with (((⟨_, _, rfl⟩ | ⟨_, _, rfl⟩) | ⟨_, _, rfl⟩) | ⟨_, _, rfl⟩) <;> rfl
    exact Or.inr (Or.inr ⟨by rw [hrip2]; exact hbp.1.2.2.2.1,
      by rw [hrip2]; have := hbp.1.2.2.2.2.1; unfold L_vote L_core_end at *; omega, hts,
      fun h => absurd (by rw [hrip2] at h; exact h) hrip'⟩)
  | @fetch alt halt =>
    have hsrip : s.rip = L_vote + (s.rip - L_vote) := (Nat.add_sub_cancel' hlo).symm
    by_cases h8d : s.rip - L_vote = 0x8d
    · have h0x8d : 0x8d ∈ SporosCheckFlat.preFaultOffs := by rw [← h8d]; exact hmem_pre
      have halt' : alt ∈ tailWindowAlts 0x8d := by
        rw [← spr1Alts_eq_tailWindowAlts h0x8d, ← h8d, ← hsrip]; exact halt
      have haud := List.all_eq_true.1 tailAlts_8d_audit alt halt'
      simp only [Bool.or_eq_true, decide_eq_true_eq] at haud
      rcases haud with hmem_tail | hother
      · have halt_t : alt ∈ SporosCheckFlat.tailAltsAt L_vote s.rip := by
          rw [hsrip, h8d]; exact hmem_tail
        have hp := SporosCheckFlat.PreFault_elemFaultStep_safe L_vote_base_ok hs.1
          (SporosCheckFlat.ElemFaultStep.fetch halt_t)
        cases hex : execO s alt with
        | stuck => exact trivial
        | exit c => rw [hex] at hp; exact hp
        | ok s' =>
          rw [hex] at hp
          exact execO_audit_to_binTailSafe hmem_pre hsrip hs.2
            (by rw [← hsrip]; exact halt) hex hp
      · cases alt with
        | none => simp at hother
        | some p =>
          rcases p with ⟨ins, len⟩
          cases ins with
          | jcc cc t =>
            have hlen6 : len = 6 := by
              by_cases h6 : len = 6
              · exact h6
              · exfalso
                cases len with
                | zero => simp at hother
                | succ l1 => cases l1 with | zero => simp at hother | succ l2 =>
                  cases l2 with | zero => simp at hother | succ l3 =>
                  cases l3 with | zero => simp at hother | succ l4 =>
                  cases l4 with | zero => simp at hother | succ l5 =>
                  cases l5 with
                  | zero => simp at hother
                  | succ l6 => cases l6 with
                    | zero => exact h6 rfl
                    | succ _ => simp at hother
            subst hlen6
            simp only [Bool.and_eq_true] at hother
            cases hex : execO s (some (.jcc cc t, 6)) with
            | stuck => exact trivial
            | exit _ => simp [execO, exec, lift, bind, Option.bind, pure] at hex; split at hex <;> cases hex
            | ok s' =>
              dsimp only [execO] at hex
              have hrc := exec_rip_cases hex
              rcases hrc with hrc1 | hrc2
              · dsimp only [BinOutSafe]; unfold BinTailSafe; rw [hrc1]; exact Or.inr (Or.inl (im_eq_none_of_isNoneAddr hother.1))
              · dsimp only [BinOutSafe]; unfold BinTailSafe; rw [hrc2, hsrip, h8d]
                exact Or.inr (Or.inl (im_eq_none_of_isNoneAddr hother.2))
          | _ => simp at hother
    · have hmatch := List.all_eq_true.1 tailAlts_pre_match (s.rip - L_vote) hmem_pre
      simp only [Bool.or_eq_true, beq_iff_eq] at hmatch
      have halt_t : alt ∈ SporosCheckFlat.tailAltsAt L_vote s.rip := by
        rw [hsrip, ← (hmatch.resolve_left h8d), ← spr1Alts_eq_tailWindowAlts hmem_pre, ← hsrip]
        exact halt
      have hp := SporosCheckFlat.PreFault_elemFaultStep_safe L_vote_base_ok hs.1
        (SporosCheckFlat.ElemFaultStep.fetch halt_t)
      cases hex : execO s alt with
      | stuck => exact trivial
      | exit c => rw [hex] at hp; exact hp
      | ok s' =>
        rw [hex] at hp
        exact execO_audit_to_binTailSafe hmem_pre hsrip hs.2 (by rw [← hsrip]; exact halt) hex hp

inductive BinOneFaultRun : Bool → Nat → Out → Out → Prop where
  | zero (faulted : Bool) (o : Out) : BinOneFaultRun faulted 0 o o
  | step_exit (faulted : Bool) (n c : Nat) :
      BinOneFaultRun faulted (n + 1) (.exit c) (.exit c)
  | step_stuck (faulted : Bool) (n : Nat) :
      BinOneFaultRun faulted (n + 1) .stuck .stuck
  | step_clean (faulted : Bool) (n : Nat) (s : St) (o : Out) :
      BinOneFaultRun faulted n (step im s) o →
      BinOneFaultRun faulted (n + 1) (.ok s) o
  | step_fault (n : Nat) (s : St) (o1 o2 : Out) :
      BinElemFaultStep s o1 →
      BinOneFaultRun true n o1 o2 →
      BinOneFaultRun false (n + 1) (.ok s) o2

private theorem BinTailSafe_OneFaultRun_clean :
    ∀ {f : Bool} {n : Nat} {o1 o2 : Out},
      f = true → BinOneFaultRun f n o1 o2 → BinOutSafe o1 → o2 ≠ .exit 0 := by
  intro f n o1 o2 hf hrun hs1
  induction hrun with
  | zero f0 o =>
    cases o with
    | ok _ => intro h; cases h
    | exit c => intro h; cases h; exact hs1 rfl
    | stuck => intro h; cases h
  | step_exit => intro h; cases h; exact hs1 rfl
  | step_stuck => intro h; cases h
  | step_clean faulted m s' o _ ih =>
    apply ih hf
    have hstep := BinTailSafe_step hs1
    generalize ho : step im s' = o0 at hstep ⊢
    cases o0 <;> exact hstep
  | step_fault => cases hf

theorem BinPreFault_OneFaultRun {v : Nat} :
    ∀ {faulted : Bool} {n : Nat} {o0 o : Out},
      BinOneFaultRun faulted n o0 o → ∀ s, o0 = .ok s → BinPreFault v s → o ≠ .exit 0 := by
  intro faulted n o0 o hrun
  induction hrun with
  | zero _ _ => intro s h _; cases h; intro h1; cases h1
  | step_exit => intro s h; cases h
  | step_stuck => intro s h; cases h
  | step_clean faulted m s' o hnext ih =>
    intro s ho hs
    cases ho
    have hc := BinPreFault_step hs
    cases hst : step im s' with
    | ok s1 => rw [hst] at hc; exact ih s1 hst hc
    | exit c =>
      rw [hst] at hc hnext
      have : o = .exit c := by
        cases hnext with
        | zero => rfl
        | step_exit => rfl
      rw [this]; intro h; cases h; exact hc rfl
    | stuck =>
      rw [hst] at hnext
      have : o = .stuck := by
        cases hnext with
        | zero => rfl
        | step_stuck => rfl
      rw [this]; intro h; cases h
  | step_fault m s' o1 o2 helem hrest =>
    intro s ho hs
    cases ho
    exact BinTailSafe_OneFaultRun_clean rfl hrest (BinPreFault_elemFaultStep_safe hs helem)

/-! ### Direct `X86Int.pstep` / `X86Int.prun` on `im` and `spr1Alts` -/

private theorem rat_pos_mul_left {a b : Rat} (ha : 0 ≤ a) (_hb : 0 ≤ b) (h : 0 < a * b) : 0 < a := by
  by_cases hle : a ≤ 0
  · exfalso; have := Rat.le_antisymm hle ha; rw [this, Rat.zero_mul] at h; exact Rat.lt_irrefl h
  · exact Rat.not_le.1 hle

private theorem rat_pos_mul_right {a b : Rat} (_ha : 0 ≤ a) (hb : 0 ≤ b) (h : 0 < a * b) : 0 < b := by
  by_cases hle : b ≤ 0
  · exfalso; have := Rat.le_antisymm hle hb; rw [this, Rat.mul_zero] at h; exact Rat.lt_irrefl h
  · exact Rat.not_le.1 hle

private theorem mem_pureF_iff {α : Type} {a : α} {y : (α × Bool) × Rat} :
    y ∈ PDist.pureF a ↔ y = ((a, false), 1) := by
  simp [PDist.pureF]

private theorem mem_unif_pos {β : Type} {p : Rat} (hp : PDist.Rate p) {c : β} {xs : List β}
    {y : (β × Bool) × Rat} (hy : y ∈ X86Int.unif p c xs) (hpos : 0 < y.2) :
    y.1 = (c, false) ∨ (0 < p ∧ y.1.2 = true ∧ y.1.1 ∈ xs) := by
  cases xs with
  | nil =>
    simp only [X86Int.unif, PDist.pureF, List.mem_cons, List.not_mem_nil, or_false] at hy
    subst hy; exact Or.inl rfl
  | cons x r =>
    simp only [X86Int.unif, List.mem_cons, List.mem_map] at hy
    rcases hy with rfl | ⟨z, hz, rfl⟩
    · exact Or.inl rfl
    · refine Or.inr ⟨?_, rfl, ?_⟩
      · by_cases hp0 : p ≤ 0
        · exfalso
          have hp_eq : p = 0 := Rat.le_antisymm hp0 hp.1
          rw [hp_eq] at hpos
          simp [Rat.div_def] at hpos
        · exact Rat.not_le.1 hp0
      · simpa using hz

private theorem mem_relabel_pos {α β : Type} {g : α → β} {d : PDist.PDF α}
    {y : (β × Bool) × Rat} (hy : y ∈ X86Int.relabel g d) :
    ∃ x ∈ d, y = ((g x.1.1, x.1.2), x.2) := by
  simp only [X86Int.relabel, List.mem_map] at hy
  rcases hy with ⟨x, hx, rfl⟩
  exact ⟨x, hx, rfl⟩

private theorem mem_bindF_pos {α β : Type} {d : PDist.PDF α} {f : α → PDist.PDF β}
    (hd : ∀ x ∈ d, 0 ≤ x.2) (hf : ∀ a z, z ∈ f a → 0 ≤ z.2) {y : (β × Bool) × Rat}
    (hy : y ∈ PDist.bindF d f) (hpos : 0 < y.2) :
    ∃ x ∈ d, ∃ z ∈ f x.1.1,
      0 < x.2 ∧ 0 < z.2 ∧ y = ((z.1.1, x.1.2 || z.1.2), x.2 * z.2) := by
  induction d with
  | nil => simp [PDist.bindF] at hy
  | cons x rest ih =>
    rcases x with ⟨⟨a, tx⟩, wx⟩
    have hwx : 0 ≤ wx := hd ((a, tx), wx) (List.mem_cons_self ..)
    simp only [PDist.bindF, List.mem_append, PDist.tagW, List.mem_map] at hy
    rcases hy with ⟨z, hz, rfl⟩ | hrest
    · have hwz : 0 ≤ z.2 := hf a z hz
      exact ⟨((a, tx), wx), List.mem_cons_self .., z, hz,
        rat_pos_mul_left hwx hwz hpos, rat_pos_mul_right hwx hwz hpos, rfl⟩
    · rcases ih (fun u hu => hd u (List.mem_cons_of_mem _ hu)) hrest with ⟨u, hu, z, hz, hwu, hwz, heq⟩
      exact ⟨u, List.mem_cons_of_mem _ hu, z, hz, hwu, hwz, heq⟩

private theorem pstep_none_zero_pos {alts : Alts} {imFn : Nat → Option (Ins × Nat)}
    {s : St} {y : (Out × Bool) × Rat}
    (hy : y ∈ pstep alts .none 0 imFn s) (hpos : 0 < y.2) :
    y.1.1 = step imFn s := by
  have h0 : PDist.Rate (0 : Rat) := ⟨Rat.le_refl, by decide⟩
  have hf_ok := fetched_ok h0 alts imFn s
  rcases mem_bindF_pos hf_ok.1 (fun o _ ho => (retire_ok .none o).1 _ ho) hy hpos with
    ⟨⟨⟨of, tf⟩, wf⟩, hof, ⟨⟨or_, tr⟩, wr⟩, hor, hwf, _, rfl⟩
  dsimp only at hor hwf ⊢
  have heq_f : of = step imFn s := by
    rw [step]
    cases him : imFn s.rip with
    | none =>
      rw [him] at hof; rw [mem_pureF_iff] at hof; exact congrArg (·.1.1) hof
    | some p =>
      rcases p with ⟨ins, len⟩
      rw [him] at hof
      rcases mem_unif_pos h0 hof hwf with h1 | ⟨hlt, _⟩
      · exact congrArg (·.1) h1
      · exact absurd hlt (Rat.lt_irrefl)
  rw [heq_f] at hor
  cases hstep : step imFn s with
  | stuck => rw [hstep] at hor; rw [mem_pureF_iff] at hor; exact congrArg (·.1.1) hor
  | exit c => rw [hstep] at hor; rw [mem_pureF_iff] at hor; exact congrArg (·.1.1) hor
  | ok s1 =>
    rw [hstep] at hor
    rcases mem_relabel_pos hor with ⟨⟨⟨sr, tr'⟩, wr'⟩, hhit, hrel⟩
    simp only [Fault.none, mem_pureF_iff] at hhit
    cases hhit; cases hrel; rfl

private theorem prun_none_zero_pos {alts : Alts} {imFn : Nat → Option (Ins × Nat)} :
    ∀ n o {y : (Out × Bool) × Rat},
      y ∈ prun alts .none 0 imFn n o → 0 < y.2 → y.1.1 = run imFn n o
  | 0, o, y, hy, _ => by rw [prun, mem_pureF_iff] at hy; subst hy; rfl
  | _ + 1, .exit c, y, hy, _ => by dsimp only [prun] at hy; rw [mem_pureF_iff] at hy; subst hy; rfl
  | _ + 1, .stuck, y, hy, _ => by dsimp only [prun] at hy; rw [mem_pureF_iff] at hy; subst hy; rfl
  | n + 1, .ok s, y, hy, hpos => by
    have h0 : PDist.Rate (0 : Rat) := ⟨Rat.le_refl, by decide⟩
    rcases mem_bindF_pos (pstep_ok alts .none h0 imFn s).1
      (fun o _ ho => (prun_ok alts .none h0 imFn n o).1 _ ho) hy hpos with
      ⟨x, hx, z, hz, hwx, hwz, rfl⟩
    have hx_eq := pstep_none_zero_pos hx hwx
    have hz_eq := prun_none_zero_pos n x.1.1 hz hwz
    rw [hx_eq] at hz_eq
    exact hz_eq

private theorem pstep_singleMechanism_elem {r : Rates} (hr : r.ok)
    (hsing : SporosCheckFlat.singleMechanism r) {s : St} {y : (Out × Bool) × Rat}
    (hy : y ∈ pstep spr1Alts (r.state hr) r.fetch im s)
    (hpos : 0 < y.2) :
    BinElemFaultStep s y.1.1 := by
  have hf_ok := fetched_ok hr.1 spr1Alts im s
  rcases mem_bindF_pos hf_ok.1 (fun o _ ho => (retire_ok (r.state hr) o).1 _ ho) hy hpos with
    ⟨⟨⟨of, tf⟩, wf⟩, hof, ⟨⟨or_, tr⟩, wr⟩, hor, hwf, hwr, rfl⟩
  dsimp only at hof hor hwf hwr ⊢
  have hof_elem : of = step im s ∨
      (0 < r.fetch ∧ ∃ alt ∈ spr1Alts s.rip, of = execO s alt) := by
    cases him : im s.rip with
    | none =>
      rw [him] at hof; rw [mem_pureF_iff] at hof
      have hof' : of = .stuck := congrArg (·.1.1) hof
      exact Or.inl (by rw [hof']; simp [step, him])
    | some p =>
      rcases p with ⟨ins, len⟩
      rw [him] at hof
      rcases mem_unif_pos hr.1 hof hwf with h1 | ⟨hpf_pos, _, hmap⟩
      · have h1' : of = execO s (some (ins, len)) := congrArg (·.1) h1
        exact Or.inl (by rw [h1']; simp [step, him, execO])
      · simp only [List.mem_map] at hmap
        rcases hmap with ⟨alt, halt, halto⟩
        exact Or.inr ⟨hpf_pos, alt, halt, halto.symm⟩
  cases of with
  | stuck =>
    rw [mem_pureF_iff] at hor; cases hor
    rcases hof_elem with heq | ⟨_, alt, halt, halto⟩
    · rw [heq]; exact BinElemFaultStep.clean
    · rw [halto]; exact BinElemFaultStep.fetch halt
  | exit c =>
    rw [mem_pureF_iff] at hor; cases hor
    rcases hof_elem with heq | ⟨_, alt, halt, halto⟩
    · rw [heq]; exact BinElemFaultStep.clean
    · rw [halto]; exact BinElemFaultStep.fetch halt
  | ok s1 =>
    rcases mem_relabel_pos hor with ⟨⟨⟨sr, tr'⟩, wr'⟩, hhit, hrel⟩
    cases hrel; dsimp only
    dsimp only [Rates.state, Fault.seq, Fault.uniform] at hhit
    have hreg_ok := fun t => unif_ok hr.2.1 t (regFlips t)
    have hmem_ok := fun t => unif_ok hr.2.2.1 t (memFlips t)
    have hfl_ok := fun t => unif_ok hr.2.2.2 t (flagFlips t)
    have hmf_ok : ∀ t, (PDist.bindF (unif r.mem t (memFlips t))
        (fun x => unif r.flag x (flagFlips x))).ok :=
      fun t => PDist.bindF_ok (hmem_ok t) hfl_ok
    rcases mem_bindF_pos (hreg_ok s1).1 (fun a z hz => (hmf_ok a).1 z hz) hhit hwr with
      ⟨⟨⟨s2, t2⟩, w2⟩, hs2, ⟨⟨s4, t4⟩, w4⟩, hs4, hw2, hw4, heq4⟩
    rcases mem_bindF_pos (hmem_ok s2).1 (fun a z hz => (hfl_ok a).1 z hz) hs4 hw4 with
      ⟨⟨⟨s3, t3⟩, w3⟩, hs3, ⟨⟨s4', t4'⟩, w4'⟩, hs4', hw3, hw4', heq4'⟩
    cases heq4; cases heq4'; dsimp only at hs2 hs3 hs4' hw2 hw3 hw4' ⊢
    have hs2_cases := mem_unif_pos hr.2.1 hs2 hw2
    have hs3_cases := mem_unif_pos hr.2.2.1 hs3 hw3
    have hs4_cases := mem_unif_pos hr.2.2.2 hs4' hw4'
    rcases hsing with ⟨hreg0, hmem0, hfl0⟩ | ⟨hft0, hmem0, hfl0⟩ | ⟨hft0, hreg0, hfl0⟩ | ⟨hft0, hreg0, hmem0⟩
    · have hs2_eq : s2 = s1 := by
        rcases hs2_cases with h | ⟨hlt, _⟩; exact congrArg (·.1) h; rw [hreg0] at hlt; exact absurd hlt Rat.lt_irrefl
      have hs3_eq : s3 = s2 := by
        rcases hs3_cases with h | ⟨hlt, _⟩; exact congrArg (·.1) h; rw [hmem0] at hlt; exact absurd hlt Rat.lt_irrefl
      have hs4_eq : s4' = s3 := by
        rcases hs4_cases with h | ⟨hlt, _⟩; exact congrArg (·.1) h; rw [hfl0] at hlt; exact absurd hlt Rat.lt_irrefl
      rw [hs4_eq, hs3_eq, hs2_eq]
      rcases hof_elem with heq | ⟨_, alt, halt, halto⟩
      · rw [heq]; exact BinElemFaultStep.clean
      · rw [halto]; exact BinElemFaultStep.fetch halt
    · have hclean1 : step im s = .ok s1 := by
        rcases hof_elem with heq | ⟨hlt, _⟩; exact heq.symm; rw [hft0] at hlt; exact absurd hlt Rat.lt_irrefl
      have hs3_eq : s3 = s2 := by
        rcases hs3_cases with h | ⟨hlt, _⟩; exact congrArg (·.1) h; rw [hmem0] at hlt; exact absurd hlt Rat.lt_irrefl
      have hs4_eq : s4' = s3 := by
        rcases hs4_cases with h | ⟨hlt, _⟩; exact congrArg (·.1) h; rw [hfl0] at hlt; exact absurd hlt Rat.lt_irrefl
      rw [hs4_eq, hs3_eq]
      rcases hs2_cases with h | ⟨_, _, hmem_reg⟩
      · have hs2_eq : s2 = s1 := congrArg (·.1) h
        rw [hs2_eq, ← hclean1]; exact BinElemFaultStep.clean
      · exact BinElemFaultStep.reg hclean1 hmem_reg
    · have hclean1 : step im s = .ok s1 := by
        rcases hof_elem with heq | ⟨hlt, _⟩; exact heq.symm; rw [hft0] at hlt; exact absurd hlt Rat.lt_irrefl
      have hs2_eq : s2 = s1 := by
        rcases hs2_cases with h | ⟨hlt, _⟩; exact congrArg (·.1) h; rw [hreg0] at hlt; exact absurd hlt Rat.lt_irrefl
      have hs4_eq : s4' = s3 := by
        rcases hs4_cases with h | ⟨hlt, _⟩; exact congrArg (·.1) h; rw [hfl0] at hlt; exact absurd hlt Rat.lt_irrefl
      rw [hs4_eq]; rw [hs2_eq] at hs3_cases
      rcases hs3_cases with h | ⟨_, _, hmem_m⟩
      · have hs3_eq : s3 = s1 := congrArg (·.1) h
        rw [hs3_eq, ← hclean1]; exact BinElemFaultStep.clean
      · exact BinElemFaultStep.mem hclean1 hmem_m
    · have hclean1 : step im s = .ok s1 := by
        rcases hof_elem with heq | ⟨hlt, _⟩; exact heq.symm; rw [hft0] at hlt; exact absurd hlt Rat.lt_irrefl
      have hs2_eq : s2 = s1 := by
        rcases hs2_cases with h | ⟨hlt, _⟩; exact congrArg (·.1) h; rw [hreg0] at hlt; exact absurd hlt Rat.lt_irrefl
      have hs3_eq : s3 = s2 := by
        rcases hs3_cases with h | ⟨hlt, _⟩; exact congrArg (·.1) h; rw [hmem0] at hlt; exact absurd hlt Rat.lt_irrefl
      rw [hs3_eq, hs2_eq] at hs4_cases
      rcases hs4_cases with h | ⟨_, _, hmem_fl⟩
      · have hs4_eq : s4' = s1 := congrArg (·.1) h
        rw [hs4_eq, ← hclean1]; exact BinElemFaultStep.clean
      · exact BinElemFaultStep.flag hclean1 hmem_fl

private theorem BinPreFault_run {v : Nat} :
    ∀ (k : Nat) {s : St}, BinPreFault v s →
      match run im k (.ok s) with
      | .ok s' => BinPreFault v s'
      | .exit c => c ≠ 0
      | .stuck => True
  | 0, _, hs => hs
  | k + 1, s, hs => by
    have hc := BinPreFault_step hs
    simp only [run]
    cases hst : step im s with
    | stuck => rw [run_stuck]; exact trivial
    | exit c => rw [hst] at hc; rw [run_exit]; exact hc
    | ok s1 => rw [hst] at hc; exact BinPreFault_run k hc

/-- Physical execution on `SporosCheckBin.im` and `spr1Alts` with one disturbed step at `k`. -/
def pRunWithFaultAt (r : Rates) (hr : r.ok) (k m : Nat) (o : Out) : PDist.PDF Out :=
  PDist.bindF (prun spr1Alts .none 0 im k o) fun o1 =>
    PDist.bindF
      (match o1 with
       | .ok s1 => pstep spr1Alts (r.state hr) r.fetch im s1
       | o' => PDist.pureF o')
      (prun spr1Alts .none 0 im m)

theorem pRunWithFaultAt_ok (r : Rates) (hr : r.ok) (k m : Nat) (o : Out) :
    (pRunWithFaultAt r hr k m o).ok := by
  have h0 : PDist.Rate (0 : Rat) := ⟨Rat.le_refl, by decide⟩
  refine PDist.bindF_ok (prun_ok spr1Alts .none h0 im k o) ?_
  intro o1
  refine PDist.bindF_ok ?_ (prun_ok spr1Alts .none h0 im m)
  cases o1 with
  | ok s1 => exact pstep_ok spr1Alts (r.state hr) hr.1 im s1
  | exit _ => exact PDist.pureF_ok _
  | stuck => exact PDist.pureF_ok _

theorem binTailPRunWithFaultAt_pos_ne_exit0 {v : Nat} (s0 : St) (hst : BinStackSafe s0)
    (hv : SporosCheckFlat.SafeInvalid v)
    (r : Rates) (hr : r.ok) (hsing : SporosCheckFlat.singleMechanism r) (k m : Nat)
    {y : (Out × Bool) × Rat}
    (hy : y ∈ pRunWithFaultAt r hr k m (.ok (SporosCheckFlat.tailInit L_vote s0 v v v)))
    (hpos : 0 < y.2) :
    y.1.1 ≠ .exit 0 := by
  have h0 : PDist.Rate (0 : Rat) := ⟨Rat.le_refl, by decide⟩
  have hmid_ok : ∀ o1 : Out, (PDist.bindF
      (match o1 with
       | .ok s1 => pstep spr1Alts (r.state hr) r.fetch im s1
       | o' => PDist.pureF o')
      (prun spr1Alts .none 0 im m)).ok := fun o1 =>
    PDist.bindF_ok (by cases o1; exact pstep_ok _ _ hr.1 _ _; exact PDist.pureF_ok _; exact PDist.pureF_ok _)
      (prun_ok _ .none h0 _ m)
  rcases mem_bindF_pos (prun_ok spr1Alts .none h0 im k _).1
    (fun o1 z hz => (hmid_ok o1).1 z hz) hy hpos with
    ⟨x1, hx1, z2, hz2, hwx1, hwz2, rfl⟩
  have hx1_eq := prun_none_zero_pos k (.ok (SporosCheckFlat.tailInit L_vote s0 v v v)) hx1 hwx1
  have hpf_k := BinPreFault_run k (BinPreFault_init s0 hst hv)
  rcases mem_bindF_pos (by cases x1.1.1; exact (pstep_ok _ _ hr.1 _ _).1; exact (PDist.pureF_ok _).1; exact (PDist.pureF_ok _).1)
    (fun o2 z hz => (prun_ok spr1Alts .none h0 im m o2).1 z hz) hz2 hwz2 with
    ⟨x2, hx2, z3, hz3, hwx2, hwz3, heq3⟩
  have hz3_eq := prun_none_zero_pos m x2.1.1 hz3 hwz3
  have helem_safe : BinOutSafe x2.1.1 := by
    rw [hx1_eq] at hx2
    cases hrun_k : run im k (.ok (SporosCheckFlat.tailInit L_vote s0 v v v)) with
    | stuck => rw [hrun_k, mem_pureF_iff] at hx2; rw [congrArg (·.1.1) hx2]; exact trivial
    | exit c => rw [hrun_k] at hpf_k hx2; rw [mem_pureF_iff] at hx2; rw [congrArg (·.1.1) hx2]; exact hpf_k
    | ok s1 =>
      rw [hrun_k] at hpf_k hx2
      exact BinPreFault_elemFaultStep_safe hpf_k (pstep_singleMechanism_elem hr hsing hx2 hwx2)
  rw [heq3]; dsimp only; rw [hz3_eq]
  cases hx2o : x2.1.1 with
  | stuck => rw [hx2o] at helem_safe; rw [run_stuck]; intro h; cases h
  | exit c => rw [hx2o] at helem_safe; rw [run_exit]; intro h; cases h; exact helem_safe rfl
  | ok s' => rw [hx2o] at helem_safe; exact BinTailSafe_run m helem_safe

theorem binTailPRunWithFaultAt_exit0Prob_zero {v : Nat} (s0 : St) (hst : BinStackSafe s0)
    (hv : SporosCheckFlat.SafeInvalid v)
    (r : Rates) (hr : r.ok) (hsing : SporosCheckFlat.singleMechanism r) (k m : Nat) :
    SporosCheckFlat.exit0Prob (pRunWithFaultAt r hr k m (.ok (SporosCheckFlat.tailInit L_vote s0 v v v))) = 0 :=
  SporosCheckFlat.exit0Prob_eq_zero_of_forall (pRunWithFaultAt_ok r hr k m _).1
    (fun _ hy hpos => binTailPRunWithFaultAt_pos_ne_exit0 s0 hst hv r hr hsing k m hy hpos)

private theorem bin_run_ge_exit {k c n : Nat} {o : Out}
    (h : run im k o = .exit c) (hle : k ≤ n) : run im n o = .exit c := by
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le hle
  clear hle
  induction d with
  | zero => exact h
  | succ d ih => rw [← Nat.add_assoc, run_add, ih]; simp [run]

theorem bin_tail_threePass_single_fault {v a b c : Nat}
    (htri : SporosCheckFlat.SinglePassFaultTriple v (a, b, c))
    {s : St} (hrip : s.rip = L_vote) (hd : RegDefExcept11 s)
    (h0 : s.reg 0 = some a) (h3 : s.reg 3 = some b) (h1 : s.reg 1 = some c)
    (hv : SporosCheckFlat.SafeInvalid v) :
    ∀ n, run im n (.ok s) ≠ .exit 0 := by
  have ha := htri.1
  have hb := htri.2.1
  have hc := htri.2.2.1
  have hvote := SporosCheckFlat.SinglePassFaultTriple_vote htri
  have hex : Ends s (· = .exit (tailExit a b c)) := tail_spec ha hb hc hrip hd h0 h3 h1
  have hnz : tailExit a b c ≠ 0 := by
    simp only [tailExit, hvote]
    have hne : v ≠ ACCEPT := by simpa [ACCEPT, SporosCheckFlat.ACCEPT] using hv.2.1
    rw [if_neg hne]
    split <;> decide
  obtain ⟨N, hN⟩ := hex
  intro n hn
  by_cases hle : n ≤ N
  · have := bin_run_ge_exit hn hle; rw [hN] at this; injection this with h0eq; exact hnz h0eq
  · have := bin_run_ge_exit hN (Nat.le_of_not_le hle); rw [hn] at this; injection this with h0eq; exact hnz h0eq.symm

/-! ### Part 4: Passes 1–3 audit, submemory simulation, and `TripledState` -/

private def p1Pred (e : Nat × Ins × Nat) : Bool := L_p1 ≤ e.1 && e.1 < L_p2
private def p2Pred (e : Nat × Ins × Nat) : Bool := L_p2 ≤ e.1 && e.1 < L_p3
private def p3Pred (e : Nat × Ins × Nat) : Bool := L_p3 ≤ e.1 && e.1 < L_vote

def p1Tbl : List (Nat × Ins × Nat) := coreTable.filter p1Pred
def p2Tbl : List (Nat × Ins × Nat) := coreTable.filter p2Pred
def p3Tbl : List (Nat × Ins × Nat) := coreTable.filter p3Pred

private theorem filter_eq_self_of_all {α : Type} {p : α → Bool} :
    ∀ l : List α, l.all p = true → l.filter p = l
  | [], _ => rfl
  | x :: xs, h => by
    simp only [List.all_cons, Bool.and_eq_true] at h
    simp [h.1, filter_eq_self_of_all xs h.2]

private theorem filter_eq_nil_of_all_not {α : Type} {p : α → Bool} :
    ∀ l : List α, l.all (fun x => !p x) = true → l.filter p = []
  | [], _ => rfl
  | x :: xs, h => by
    simp only [List.all_cons, Bool.and_eq_true, Bool.not_eq_true'] at h
    simp [h.1, filter_eq_nil_of_all_not xs h.2]

private theorem filter_flatten_id {α : Type} (p : α → Bool) :
    ∀ ls : List (List α), (∀ l ∈ ls, l.all p = true) → ls.flatten.filter p = ls.flatten
  | [], _ => rfl
  | l :: ls, h => by
    simp only [List.flatten_cons, List.filter_append,
      filter_eq_self_of_all l (h l (List.mem_cons_self ..)),
      filter_flatten_id p ls (fun x hx => h x (List.mem_cons_of_mem _ hx))]

private theorem filter_flatten_nil {α : Type} (p : α → Bool) :
    ∀ ls : List (List α), (∀ l ∈ ls, l.all (fun x => !p x) = true) → ls.flatten.filter p = []
  | [], _ => rfl
  | l :: ls, h => by
    simp only [List.flatten_cons, List.filter_append,
      filter_eq_nil_of_all_not l (h l (List.mem_cons_self ..)),
      filter_flatten_nil p ls (fun x hx => h x (List.mem_cons_of_mem _ hx)), List.append_nil]

private def p1Chunks : List (List (Nat × Ins × Nat)) :=
  [coreTable_0, coreTable_1, coreTable_2, coreTable_3, coreTable_4, coreTable_5,
   coreTable_6, coreTable_7, coreTable_8, coreTable_9, coreTable_10, coreTable_11]
private def p1Rest : List (List (Nat × Ins × Nat)) :=
  [coreTable_13, coreTable_14, coreTable_15, coreTable_16, coreTable_17, coreTable_18,
   coreTable_19, coreTable_20, coreTable_21, coreTable_22, coreTable_23, coreTable_24,
   coreTable_25, coreTable_26, coreTable_27, coreTable_28, coreTable_29, coreTable_30,
   coreTable_31, coreTable_32, coreTable_33, coreTable_34, coreTable_35, coreTable_36]
private def p1List : List (List (Nat × Ins × Nat)) := p1Chunks ++ [coreTable_12.filter p1Pred]
private def p1Slice : List (Nat × Ins × Nat) := p1List.flatten

private def p2Pre : List (List (Nat × Ins × Nat)) := p1Chunks
private def p2Chunks : List (List (Nat × Ins × Nat)) :=
  [coreTable_13, coreTable_14, coreTable_15, coreTable_16, coreTable_17, coreTable_18,
   coreTable_19, coreTable_20, coreTable_21, coreTable_22, coreTable_23]
private def p2Rest : List (List (Nat × Ins × Nat)) :=
  [coreTable_25, coreTable_26, coreTable_27, coreTable_28, coreTable_29, coreTable_30,
   coreTable_31, coreTable_32, coreTable_33, coreTable_34, coreTable_35, coreTable_36]
private def p2List : List (List (Nat × Ins × Nat)) :=
  [coreTable_12.filter p2Pred] ++ p2Chunks ++ [coreTable_24.filter p2Pred]
private def p2Slice : List (Nat × Ins × Nat) := p2List.flatten

private def p3Pre : List (List (Nat × Ins × Nat)) := p2Pre ++ [coreTable_12] ++ p2Chunks
private def p3Chunks : List (List (Nat × Ins × Nat)) :=
  [coreTable_25, coreTable_26, coreTable_27, coreTable_28, coreTable_29, coreTable_30,
   coreTable_31, coreTable_32, coreTable_33, coreTable_34, coreTable_35]
private def p3List : List (List (Nat × Ins × Nat)) :=
  [coreTable_24.filter p3Pred] ++ p3Chunks ++ [coreTable_36.filter p3Pred]
private def p3Slice : List (Nat × Ins × Nat) := p3List.flatten

private theorem p1_chunks_ok :
    (p1Chunks.all (·.all p1Pred) && p1Rest.all (·.all fun e => !p1Pred e)) = true := by
  decide +kernel

private theorem p2_chunks_ok :
    (p2Pre.all (·.all fun e => !p2Pred e) &&
     p2Chunks.all (·.all p2Pred) &&
     p2Rest.all (·.all fun e => !p2Pred e)) = true := by
  decide +kernel

private theorem p3_chunks_ok :
    (p3Pre.all (·.all fun e => !p3Pred e) &&
     p3Chunks.all (·.all p3Pred)) = true := by
  decide +kernel

private theorem p1Tbl_eq_p1Slice : p1Tbl = p1Slice := by
  have h1 : coreTable = p1Chunks.flatten ++ coreTable_12 ++ p1Rest.flatten := by
    unfold coreTable p1Chunks p1Rest
    simp only [List.flatten_cons, List.flatten_nil, List.append_nil, List.append_assoc]
  have hok := p1_chunks_ok
  simp only [Bool.and_eq_true] at hok
  rw [p1Tbl, h1, List.filter_append, List.filter_append,
    filter_flatten_id p1Pred p1Chunks (List.all_eq_true.1 hok.1),
    filter_flatten_nil p1Pred p1Rest (List.all_eq_true.1 hok.2), List.append_nil]
  simp only [p1Slice, p1List, List.flatten_append, List.flatten_cons, List.flatten_nil, List.append_nil]

private theorem p2Tbl_eq_p2Slice : p2Tbl = p2Slice := by
  have h1 : coreTable = p2Pre.flatten ++ coreTable_12 ++ p2Chunks.flatten ++ coreTable_24 ++ p2Rest.flatten := by
    unfold coreTable p2Pre p1Chunks p2Chunks p2Rest
    simp only [List.flatten_cons, List.flatten_nil, List.append_nil,
      List.append_assoc]
  have hok := p2_chunks_ok
  simp only [Bool.and_eq_true] at hok
  rw [p2Tbl, h1, List.filter_append, List.filter_append, List.filter_append, List.filter_append,
    filter_flatten_nil p2Pred p2Pre (List.all_eq_true.1 hok.1.1),
    filter_flatten_id p2Pred p2Chunks (List.all_eq_true.1 hok.1.2),
    filter_flatten_nil p2Pred p2Rest (List.all_eq_true.1 hok.2),
    List.nil_append, List.append_nil]
  simp only [p2Slice, p2List, List.flatten_append, List.flatten_cons, List.flatten_nil, List.append_nil,
    List.append_assoc]

private theorem p3Tbl_eq_p3Slice : p3Tbl = p3Slice := by
  have h1 : coreTable = p3Pre.flatten ++ coreTable_24 ++ p3Chunks.flatten ++ coreTable_36 := by
    unfold coreTable p3Pre p2Pre p1Chunks p2Chunks p3Chunks
    simp only [List.flatten_append, List.flatten_cons, List.flatten_nil, List.append_nil, List.append_assoc]
  have hok := p3_chunks_ok
  simp only [Bool.and_eq_true] at hok
  rw [p3Tbl, h1, List.filter_append, List.filter_append, List.filter_append,
    filter_flatten_nil p3Pred p3Pre (List.all_eq_true.1 hok.1),
    filter_flatten_id p3Pred p3Chunks (List.all_eq_true.1 hok.2),
    List.nil_append]
  simp only [p3Slice, p3List, List.flatten_append, List.flatten_cons, List.flatten_nil, List.append_nil,
    List.append_assoc]

private def insWrites : Ins → List Nat
  | .ld d _ _ | .ld32 d _ _ | .ld8 d _ _ | .mov d _ | .mov32 d _ | .movI d _
  | .alu _ d _ | .aluI _ d _ | .shlI d _ | .shrI d _ | .lea d _ _ => [d]
  | .pop d => [4, d]
  | .push _ | .pushI _ | .call _ | .ret => [4]
  | .syscall => [0, 1, 11]
  | _ => []

private def insReads : Ins → List Nat
  | .mov _ r | .mov32 _ r | .jmpR r => [r]
  | .push r => [4, r]
  | .ld _ b _ | .ld32 _ b _ | .ld8 _ b _ | .lea _ b _ | .aluI _ b _ | .cmpI b _ | .shlI b _ | .shrI b _ => [b]
  | .alu _ a b | .cmp a b | .test a b | .st a _ b | .st8 a _ b | .jtab a b _ => [a, b]
  | .syscall => [0, 7, 6, 2, 10, 8, 9]
  | .call _ | .ret | .pushI _ | .pop _ => [4]
  | _ => []

private inductive AddrTree
  | leaf : AddrTree
  | node : Nat → AddrTree → AddrTree → AddrTree

private def treeHas (x : Nat) : AddrTree → Bool
  | .leaf => false
  | .node v l r => if x == v then true else if x < v then treeHas x l else treeHas x r

private def buildTreeFuel : Nat → List Nat → AddrTree
  | 0, _ => .leaf
  | _ + 1, [] => .leaf
  | fuel + 1, xs =>
    let m := xs.length / 2
    match xs[m]? with
    | none => .leaf
    | some v => .node v (buildTreeFuel fuel (xs.take m)) (buildTreeFuel fuel (xs.drop (m + 1)))

private theorem treeHas_mem : ∀ (fuel : Nat) (xs : List Nat) (x : Nat),
    treeHas x (buildTreeFuel fuel xs) = true → x ∈ xs
  | 0, _, _, h => by simp [buildTreeFuel, treeHas] at h
  | fuel + 1, xs, x, h => by
    cases xs with
    | nil => simp [buildTreeFuel, treeHas] at h
    | cons x0 xs0 =>
      dsimp only [buildTreeFuel] at h
      cases hm : (x0 :: xs0)[(x0 :: xs0).length / 2]? with
      | none => rw [hm] at h; simp [treeHas] at h
      | some v =>
        have hv : v ∈ x0 :: xs0 := List.mem_of_getElem? hm
        rw [hm] at h
        dsimp only [treeHas] at h
        split at h
        · rename_i heq; rw [beq_iff_eq.1 heq]; exact hv
        · split at h
          · exact List.mem_of_mem_take (treeHas_mem fuel _ x h)
          · exact List.mem_of_mem_drop (treeHas_mem fuel _ x h)

private def passEntryOK (lo hi : Nat) (forbidden addrs : List Nat) (e : Nat × Ins × Nat) : Bool :=
  let (a, ins, l) := e
  let okTgt (t : Nat) : Bool := t == hi || (lo ≤ t && t < hi && addrs.contains t)
  lo ≤ a && a < hi &&
  match ins with
  | .syscall | .ret | .jmpR _ | .call _ | .jtab _ _ _ | .st _ _ _ | .st8 _ _ _
  | .push _ | .pushI _ | .pop _ => false
  | .jmp t => forbidden.all (fun r => !(insWrites ins).contains r && !(insReads ins).contains r) && okTgt t
  | .jcc _ t => forbidden.all (fun r => !(insWrites ins).contains r && !(insReads ins).contains r) &&
      okTgt t && okTgt (a + l)
  | _ => forbidden.all (fun r => !(insWrites ins).contains r && !(insReads ins).contains r) &&
      okTgt (a + l)

private def passNext (endHi : Nat) : List (Nat × Ins × Nat) → Nat
  | [] => endHi
  | (a', _, _) :: _ => a'

private def passCheckTree (lo hi endHi : Nat) (forbidden : List Nat) (tree : AddrTree) :
    List (Nat × Ins × Nat) → Bool
  | [] => true
  | (a, ins, l) :: rest =>
    let nxt := passNext endHi rest
    let okTgt (t : Nat) : Bool := t == hi || (lo ≤ t && t < hi && treeHas t tree)
    let okSeq (t : Nat) : Bool := t == hi || (lo ≤ t && t < hi && (t == nxt || treeHas t tree))
    let okForb := forbidden.all (fun r => !(insWrites ins).contains r && !(insReads ins).contains r)
    lo ≤ a && a < hi && okForb &&
    (match ins with
     | .syscall | .ret | .jmpR _ | .call _ | .jtab _ _ _ | .st _ _ _ | .st8 _ _ _
     | .push _ | .pushI _ | .pop _ => false
     | .jmp t => okTgt t
     | .jcc _ t => okTgt t && okSeq (a + l)
     | _ => okSeq (a + l)) &&
    passCheckTree lo hi endHi forbidden tree rest

private theorem passNext_mem {hi endHi : Nat} {addrs : List Nat}
    (hend : endHi = hi ∨ endHi ∈ addrs) (rest : List (Nat × Ins × Nat))
    (hrest_mem : ∀ e ∈ rest, e.1 ∈ addrs) :
    passNext endHi rest = hi ∨ passNext endHi rest ∈ addrs := by
  cases rest with
  | nil => exact hend
  | cons hd _ => exact Or.inr (hrest_mem hd (List.mem_cons_self ..))

private theorem passCheckTree_sound (lo hi endHi : Nat) (forbidden addrs : List Nat) (tree : AddrTree)
    (htree : ∀ x, treeHas x tree = true → x ∈ addrs)
    (hend : endHi = hi ∨ endHi ∈ addrs) :
    ∀ (tbl : List (Nat × Ins × Nat)),
      passCheckTree lo hi endHi forbidden tree tbl = true →
      (∀ e ∈ tbl, e.1 ∈ addrs) →
      ∀ e ∈ tbl, passEntryOK lo hi forbidden addrs e = true
  | [], _, _, _, he => by cases he
  | (a, ins, l) :: rest, h, hmem, e, he => by
    simp only [passCheckTree, Bool.and_eq_true, and_assoc] at h
    obtain ⟨hlo, hhi, hforb, hbody, hrest⟩ := h
    have hrest_mem : ∀ e' ∈ rest, e'.1 ∈ addrs := fun e' he' => hmem e' (List.mem_cons_of_mem _ he')
    rcases List.mem_cons.1 he with rfl | he'
    · have hok_tgt : ∀ t, (t == hi || (lo ≤ t && t < hi && treeHas t tree)) = true →
          (t == hi || (lo ≤ t && t < hi && addrs.contains t)) = true := by
        intro t ht
        simp only [Bool.or_eq_true, Bool.and_eq_true, and_assoc, beq_iff_eq, decide_eq_true_eq] at ht ⊢
        rcases ht with rfl | ⟨h1, h2, h3⟩
        · exact Or.inl rfl
        · exact Or.inr ⟨h1, h2, List.contains_iff_mem.2 (htree t h3)⟩
      have hnxt_mem := passNext_mem hend rest hrest_mem
      have hok_seq : ∀ t, (t == hi || (lo ≤ t && t < hi &&
          (t == passNext endHi rest || treeHas t tree))) = true →
          (t == hi || (lo ≤ t && t < hi && addrs.contains t)) = true := by
        intro t ht
        simp only [Bool.or_eq_true, Bool.and_eq_true, and_assoc, beq_iff_eq, decide_eq_true_eq] at ht ⊢
        rcases ht with rfl | ⟨h1, h2, h3 | h3⟩
        · exact Or.inl rfl
        · rcases hnxt_mem with hnhi | hnmem
          · exact Or.inl (h3.trans hnhi)
          · exact Or.inr ⟨h1, h2, List.contains_iff_mem.2 (h3 ▸ hnmem)⟩
        · exact Or.inr ⟨h1, h2, List.contains_iff_mem.2 (htree t h3)⟩
      cases ins with
      | syscall | ret | jmpR _ | call _ | jtab _ _ _ | st _ _ _ | st8 _ _ _ | push _ | pushI _ | pop _ =>
        simp at hbody
      | jmp t =>
        simp only [passEntryOK, Bool.and_eq_true, hlo, hhi, true_and]
        exact ⟨hforb, hok_tgt t hbody⟩
      | jcc _ t =>
        simp only [passEntryOK, Bool.and_eq_true, and_assoc, hlo, hhi, true_and]
        simp only [Bool.and_eq_true] at hbody
        exact ⟨hforb, hok_tgt t hbody.1, hok_seq (a + l) hbody.2⟩
      | mov _ _ | mov32 _ _ | movI _ _ | alu _ _ _ | aluI _ _ _ | cmp _ _ | cmpI _ _
      | test _ _ | shlI _ _ | shrI _ _ | ld _ _ _ | ld32 _ _ _ | ld8 _ _ _ | lea _ _ _ =>
        simp only [passEntryOK, Bool.and_eq_true, hlo, hhi, true_and]
        exact ⟨hforb, hok_seq (a + l) hbody⟩
    · exact passCheckTree_sound lo hi endHi forbidden addrs tree htree hend rest hrest hrest_mem e he'

private def chunksHeadOr (hi : Nat) : List (List (Nat × Ins × Nat)) → Nat
  | [] => hi
  | [] :: cs => chunksHeadOr hi cs
  | ((a, _, _) :: _) :: _ => a

private theorem chunksHeadOr_mem {hi : Nat} {addrs : List Nat} :
    ∀ cs : List (List (Nat × Ins × Nat)),
      (∀ e ∈ cs.flatten, e.1 ∈ addrs) →
      chunksHeadOr hi cs = hi ∨ chunksHeadOr hi cs ∈ addrs
  | [], _ => Or.inl rfl
  | [] :: cs, h => chunksHeadOr_mem cs (by simpa using h)
  | ((a, ins, l) :: rest) :: _, h =>
    Or.inr (h (a, ins, l) (by simp))

private def passChunksCheck (lo hi : Nat) (forbidden : List Nat) (tree : AddrTree) :
    List (List (Nat × Ins × Nat)) → Bool
  | [] => true
  | c :: cs => passCheckTree lo hi (chunksHeadOr hi cs) forbidden tree c &&
      passChunksCheck lo hi forbidden tree cs

private theorem passChunksCheck_sound (lo hi : Nat) (forbidden addrs : List Nat) (tree : AddrTree)
    (htree : ∀ x, treeHas x tree = true → x ∈ addrs) :
    ∀ cs : List (List (Nat × Ins × Nat)),
      passChunksCheck lo hi forbidden tree cs = true →
      (∀ e ∈ cs.flatten, e.1 ∈ addrs) →
      ∀ e ∈ cs.flatten, passEntryOK lo hi forbidden addrs e = true
  | [], _, _, _, he => by cases he
  | c :: cs, h, hmem, e, he => by
    simp only [passChunksCheck, Bool.and_eq_true] at h
    have hcmem : ∀ e' ∈ c, e'.1 ∈ addrs := fun e' he' =>
      hmem e' (List.mem_flatten.2 ⟨c, List.mem_cons_self .., he'⟩)
    have hcsmem : ∀ e' ∈ cs.flatten, e'.1 ∈ addrs := fun e' he' => by
      obtain ⟨c', hc', hec'⟩ := List.mem_flatten.1 he'
      exact hmem e' (List.mem_flatten.2 ⟨c', List.mem_cons_of_mem _ hc', hec'⟩)
    have hend := chunksHeadOr_mem (hi := hi) cs hcsmem
    rcases List.mem_append.1 (by rw [List.flatten_cons] at he; exact he) with hec | hecs
    · exact passCheckTree_sound lo hi (chunksHeadOr hi cs) forbidden addrs tree htree hend c h.1 hcmem e hec
    · exact passChunksCheck_sound lo hi forbidden addrs tree htree cs h.2 hcsmem e hecs

private def p1Addrs : List Nat := p1List.flatten.map (·.1)
private def p2Addrs : List Nat := p2List.flatten.map (·.1)
private def p3Addrs : List Nat := p3List.flatten.map (·.1)

private def p1Tree : AddrTree := buildTreeFuel 11 p1Addrs
private def p2Tree : AddrTree := buildTreeFuel 11 p2Addrs
private def p3Tree : AddrTree := buildTreeFuel 11 p3Addrs

private theorem p1TreeCheck :
    passChunksCheck L_p1 L_p2 [4, 3, 1] p1Tree p1List = true := by
  decide +kernel

private theorem p2TreeCheck :
    passChunksCheck L_p2 L_p3 [4, 0, 1] p2Tree p2List = true := by
  decide +kernel

private theorem p3TreeCheck :
    passChunksCheck L_p3 L_vote [4, 0, 3] p3Tree p3List = true := by
  decide +kernel

private theorem p1Slice_def : p1Slice = p1List.flatten := by unfold p1Slice; rfl
private theorem p2Slice_def : p2Slice = p2List.flatten := by unfold p2Slice; rfl
private theorem p3Slice_def : p3Slice = p3List.flatten := by unfold p3Slice; rfl

private theorem p1Tree_mem : ∀ x, treeHas x p1Tree = true → x ∈ p1Addrs :=
  fun x hx => treeHas_mem 11 p1Addrs x hx

private theorem p2Tree_mem : ∀ x, treeHas x p2Tree = true → x ∈ p2Addrs :=
  fun x hx => treeHas_mem 11 p2Addrs x hx

private theorem p3Tree_mem : ∀ x, treeHas x p3Tree = true → x ∈ p3Addrs :=
  fun x hx => treeHas_mem 11 p3Addrs x hx

private theorem p1ListSafe : ∀ e ∈ p1List.flatten, passEntryOK L_p1 L_p2 [4, 3, 1] p1Addrs e = true :=
  passChunksCheck_sound L_p1 L_p2 [4, 3, 1] p1Addrs p1Tree p1Tree_mem p1List p1TreeCheck
    (fun e he => List.mem_map.2 ⟨e, he, rfl⟩)

private theorem p2ListSafe : ∀ e ∈ p2List.flatten, passEntryOK L_p2 L_p3 [4, 0, 1] p2Addrs e = true :=
  passChunksCheck_sound L_p2 L_p3 [4, 0, 1] p2Addrs p2Tree p2Tree_mem p2List p2TreeCheck
    (fun e he => List.mem_map.2 ⟨e, he, rfl⟩)

private theorem p3ListSafe : ∀ e ∈ p3List.flatten, passEntryOK L_p3 L_vote [4, 0, 3] p3Addrs e = true :=
  passChunksCheck_sound L_p3 L_vote [4, 0, 3] p3Addrs p3Tree p3Tree_mem p3List p3TreeCheck
    (fun e he => List.mem_map.2 ⟨e, he, rfl⟩)

private theorem p1Addrs_eq : p1Addrs = p1Tbl.map (·.1) := by
  dsimp only [p1Addrs]; rw [← p1Slice_def, ← p1Tbl_eq_p1Slice]

private theorem p2Addrs_eq : p2Addrs = p2Tbl.map (·.1) := by
  dsimp only [p2Addrs]; rw [← p2Slice_def, ← p2Tbl_eq_p2Slice]

private theorem p3Addrs_eq : p3Addrs = p3Tbl.map (·.1) := by
  dsimp only [p3Addrs]; rw [← p3Slice_def, ← p3Tbl_eq_p3Slice]

private theorem p1SafeCheck : ∀ e ∈ p1Tbl, passEntryOK L_p1 L_p2 [4, 3, 1] (p1Tbl.map (·.1)) e = true := by
  intro e he
  have h1 : e ∈ p1List.flatten := by rw [← p1Slice_def, ← p1Tbl_eq_p1Slice]; exact he
  have h2 := p1ListSafe e h1
  rwa [← p1Addrs_eq]

private theorem p2SafeCheck : ∀ e ∈ p2Tbl, passEntryOK L_p2 L_p3 [4, 0, 1] (p2Tbl.map (·.1)) e = true := by
  intro e he
  have h1 : e ∈ p2List.flatten := by rw [← p2Slice_def, ← p2Tbl_eq_p2Slice]; exact he
  have h2 := p2ListSafe e h1
  rwa [← p2Addrs_eq]

private theorem p3SafeCheck : ∀ e ∈ p3Tbl, passEntryOK L_p3 L_vote [4, 0, 3] (p3Tbl.map (·.1)) e = true := by
  intro e he
  have h1 : e ∈ p3List.flatten := by rw [← p3Slice_def, ← p3Tbl_eq_p3Slice]; exact he
  have h2 := p3ListSafe e h1
  rwa [← p3Addrs_eq]

private theorem p1Tbl_im (e : Nat × Ins × Nat) (h : e ∈ p1Tbl) : im e.1 = some e.2 :=
  table_ok e (List.mem_append_left initTable (List.mem_filter.1 h).1)
private theorem p2Tbl_im (e : Nat × Ins × Nat) (h : e ∈ p2Tbl) : im e.1 = some e.2 :=
  table_ok e (List.mem_append_left initTable (List.mem_filter.1 h).1)
private theorem p3Tbl_im (e : Nat × Ins × Nat) (h : e ∈ p3Tbl) : im e.1 = some e.2 :=
  table_ok e (List.mem_append_left initTable (List.mem_filter.1 h).1)

def MemExtends (m1 m2 : Mem) : Prop := ∀ a v, m1.rd a = some v → m2.rd a = some v

private theorem rdN_mono {m1 m2 : Mem} (hm : MemExtends m1 m2) :
    ∀ n a xs, m1.rdN a n = some xs → m2.rdN a n = some xs
  | 0, _, xs, h => by simp [Mem.rdN] at h; subst h; rfl
  | n + 1, a, xs, h => by
    simp only [Mem.rdN, bind, Option.bind, pure] at h
    cases hrd : m1.rd a with
    | none => rw [hrd] at h; cases h
    | some b0 =>
      rw [hrd] at h
      cases hrest : m1.rdN (a + 1) n with
      | none => rw [hrest] at h; cases h
      | some bs' =>
        rw [hrest] at h
        have h1 := hm a b0 hrd
        have h2 := rdN_mono hm n (a + 1) bs' hrest
        simp only [Mem.rdN, h1, h2, bind, Option.bind, pure] at h ⊢
        exact h

private theorem rd32_mono {m1 m2 : Mem} (hm : MemExtends m1 m2) {a v : Nat} (h : rd32 m1 a = some v) :
    rd32 m2 a = some v := by
  simp only [rd32, Option.map_eq_some_iff] at h ⊢
  obtain ⟨xs, hx, rfl⟩ := h
  exact ⟨xs, rdN_mono hm 4 a xs hx, rfl⟩

private theorem rd64_mono {m1 m2 : Mem} (hm : MemExtends m1 m2) {a v : Nat} (h : rd64 m1 a = some v) :
    rd64 m2 a = some v := by
  simp only [rd64, Option.map_eq_some_iff] at h ⊢
  obtain ⟨xs, hx, rfl⟩ := h
  exact ⟨xs, rdN_mono hm 8 a xs hx, rfl⟩

private theorem pass_no_store_of_safe {lo hi : Nat} {forb : List Nat} {tbl : List (Nat × Ins × Nat)}
    (hsafe : ∀ e ∈ tbl, passEntryOK lo hi forb (tbl.map (·.1)) e = true) {e : Nat × Ins × Nat} (he : e ∈ tbl) :
    ∀ a b c, e.2.1 ≠ .st a b c ∧ e.2.1 ≠ .st8 a b c ∧ e.2.1 ≠ .push a ∧
      e.2.1 ≠ .pushI a ∧ e.2.1 ≠ .call a ∧ e.2.1 ≠ .ret ∧ e.2.1 ≠ .pop a ∧ e.2.1 ≠ .syscall := by
  have h1 := hsafe e he
  simp only [passEntryOK, Bool.and_eq_true, and_assoc] at h1
  have h2 := h1.2.2
  apply isNonStoreIns_spec
  cases he_ins : e.2.1 <;> rw [he_ins] at h2 <;> first | rfl | (exfalso; simp at h2)

private theorem exec_pass_step_props {lo hi : Nat} {forb : List Nat} {tbl : List (Nat × Ins × Nat)}
    (hsafe : ∀ e ∈ tbl, passEntryOK lo hi forb (tbl.map (·.1)) e = true)
    (htbl : ∀ e ∈ tbl, im e.1 = some e.2)
    {s s' : St} (hin : s.rip ∈ tbl.map (·.1)) (hst : step im s = .ok s') :
    s'.mem = s.mem ∧ s'.regs.length = s.regs.length ∧
    (∀ r ∈ forb, s'.reg r = s.reg r) ∧
    (s'.rip ∈ tbl.map (·.1) ∨ s'.rip = hi) := by
  simp only [List.mem_map] at hin
  obtain ⟨e, he, herip⟩ := hin
  have him := htbl e he
  rw [herip] at him
  simp only [step, him] at hst
  have hnost := pass_no_store_of_safe hsafe he
  have hm := exec_preserves_mem_non_store hst hnost
  have hall := hsafe e he
  simp only [passEntryOK, Bool.and_eq_true, and_assoc] at hall
  obtain ⟨_, _, hbody⟩ := hall
  have okTgt_mem : ∀ {t : Nat},
      (t == hi || (lo ≤ t && t < hi && (tbl.map (·.1)).contains t)) = true →
      t ∈ tbl.map (·.1) ∨ t = hi := by
    intro t ht
    simp only [Bool.or_eq_true, Bool.and_eq_true, beq_iff_eq] at ht
    rcases ht with rfl | ⟨_, hcon⟩
    · exact Or.inr rfl
    · exact Or.inl (List.contains_iff_mem.1 hcon)
  have hrc := exec_rip_cases hst
  rcases e with ⟨eAddr, eins, elen⟩
  dsimp only at herip him hnost hbody hrc
  have hforb_ok : ∀ (d : Nat) (v : Option Nat),
      (forb.all fun r => !([d].contains r) && !(insReads eins).contains r) = true →
      ∀ r ∈ forb, (s.set d v).reg r = s.reg r := by
    intro d v hd r hr
    have hmem_r := List.all_eq_true.1 hd r hr
    simp only [Bool.and_eq_true, Bool.not_eq_true', List.contains_cons, List.contains_nil,
      Bool.or_false] at hmem_r
    exact reg_set_ne (Ne.symm (beq_eq_false_iff_ne.1 hmem_r.1))
  have hlen : s'.regs.length = s.regs.length := by
    cases eins with
    | call _ | jtab _ _ _ | jmpR _ | ret | st _ _ _ | st8 _ _ _ | push _ | pushI _ | pop _ | syscall =>
      exfalso; simp at hbody
    | jmp _ =>
      simp only [exec] at hst; cases hst; rfl
    | jcc cc _ =>
      simp only [exec, lift, bind, Option.bind, pure] at hst
      cases hc : X86Int.cond s.fl cc with
      | none => simp [hc] at hst
      | some c => simp [hc] at hst; cases hst; cases c <;> rfl
    | movI _ _ =>
      simp only [exec] at hst; cases hst; simp [St.set, List.length_set]
    | mov _ r | mov32 _ r =>
      simp only [exec, lift, bind, Option.bind, pure] at hst
      cases hr : s.reg r with
      | none => simp [hr] at hst
      | some _ => simp [hr] at hst; cases hst; simp [List.length_set]
    | alu _ d r =>
      simp only [exec, lift, bind, Option.bind, pure] at hst
      cases hd : s.reg d with
      | none => simp [hd] at hst
      | some _ =>
        simp [hd] at hst
        cases hr : s.reg r with
        | none => simp [hr] at hst
        | some _ => simp [hr] at hst; cases hst; simp [List.length_set]
    | aluI _ d _ | shlI d _ | shrI d _ =>
      simp only [exec, lift, bind, Option.bind, pure] at hst
      cases hd : s.reg d with
      | none => simp [hd] at hst
      | some _ => simp [hd] at hst; cases hst; simp [List.length_set]
    | cmp a b | test a b =>
      simp only [exec, lift, bind, Option.bind, pure] at hst
      cases ha : s.reg a with
      | none => simp [ha] at hst
      | some _ =>
        simp [ha] at hst
        cases hb : s.reg b with
        | none => simp [hb] at hst
        | some _ => simp [hb] at hst; cases hst; rfl
    | cmpI a _ =>
      simp only [exec, lift, bind, Option.bind, pure] at hst
      cases ha : s.reg a with
      | none => simp [ha] at hst
      | some _ => simp [ha] at hst; cases hst; rfl
    | ld _ b off =>
      simp only [exec, lift, bind, Option.bind, pure] at hst
      cases hb : s.reg b with
      | none => simp [hb] at hst
      | some bv =>
        simp [hb] at hst
        cases hm : rd64 s.mem (ea bv off) with
        | none => simp [hm] at hst
        | some _ => simp [hm] at hst; cases hst; simp [List.length_set]
    | ld32 _ b off =>
      simp only [exec, lift, bind, Option.bind, pure] at hst
      cases hb : s.reg b with
      | none => simp [hb] at hst
      | some bv =>
        simp [hb] at hst
        cases hm : rd32 s.mem (ea bv off) with
        | none => simp [hm] at hst
        | some _ => simp [hm] at hst; cases hst; simp [List.length_set]
    | ld8 _ b off =>
      simp only [exec, lift, bind, Option.bind, pure] at hst
      cases hb : s.reg b with
      | none => simp [hb] at hst
      | some bv =>
        simp [hb] at hst
        cases hm : s.mem.rd (ea bv off) with
        | none => simp [hm] at hst
        | some _ => simp [hm] at hst; cases hst; simp [List.length_set]
    | lea _ b _ =>
      simp only [exec, lift, bind, Option.bind, pure] at hst
      cases hb : s.reg b with
      | none => simp [hb] at hst
      | some _ => simp [hb] at hst; cases hst; simp [List.length_set]
  have hrip_ok : s'.rip ∈ tbl.map (·.1) ∨ s'.rip = hi := by
    dsimp only [ripNextCond] at hrc
    cases eins with
    | jmp t =>
      simp only [Bool.and_eq_true] at hbody
      rw [hrc]; exact okTgt_mem hbody.2
    | jcc cc t =>
      simp only [Bool.and_eq_true, and_assoc] at hbody
      rcases hrc with h1 | h2
      · rw [h1]; exact okTgt_mem hbody.2.1
      · rw [h2, ← herip]; exact okTgt_mem hbody.2.2
    | call _ | jtab _ _ _ | jmpR _ | ret | st _ _ _ | st8 _ _ _ | push _ | pushI _ | pop _ | syscall =>
      exfalso; simp at hbody
    | movI _ _ | mov _ _ | mov32 _ _ | alu _ _ _ | aluI _ _ _ | cmp _ _ | cmpI _ _
    | test _ _ | shlI _ _ | shrI _ _ | ld _ _ _ | ld32 _ _ _ | ld8 _ _ _ | lea _ _ _ =>
      simp only [Bool.and_eq_true] at hbody
      rw [hrc, ← herip]; exact okTgt_mem hbody.2
  have hforb_body : (forb.all fun r => !(insWrites eins).contains r && !(insReads eins).contains r) = true := by
    cases eins with
    | call _ | jtab _ _ _ | jmpR _ | ret | st _ _ _ | st8 _ _ _ | push _ | pushI _ | pop _ | syscall =>
      exfalso; simp at hbody
    | jmp _ | jcc _ _ | movI _ _ | mov _ _ | mov32 _ _ | alu _ _ _ | aluI _ _ _ | cmp _ _ | cmpI _ _
    | test _ _ | shlI _ _ | shrI _ _ | ld _ _ _ | ld32 _ _ _ | ld8 _ _ _ | lea _ _ _ =>
      simp only [Bool.and_eq_true, and_assoc] at hbody; exact hbody.1
  have hreg_ok : ∀ r ∈ forb, s'.reg r = s.reg r := by
    intro r hr
    cases eins with
    | jmp _ => simp [exec] at hst; cases hst; rfl
    | jcc cc _ =>
      simp only [exec, lift, bind, Option.bind, pure] at hst
      cases hc : X86Int.cond s.fl cc with
      | none => simp [hc] at hst
      | some _ => simp [hc] at hst; cases hst; rfl
    | cmp a b =>
      simp only [exec, lift, bind, Option.bind, pure] at hst
      cases ha : s.reg a with
      | none => simp [ha] at hst
      | some _ =>
        simp only [ha] at hst
        cases hb : s.reg b with
        | none => simp [hb] at hst
        | some _ => simp [hb] at hst; cases hst; rfl
    | cmpI a _ =>
      simp only [exec, lift, bind, Option.bind, pure] at hst
      cases ha : s.reg a with
      | none => simp [ha] at hst
      | some _ => simp [ha] at hst; cases hst; rfl
    | test a b =>
      simp only [exec, lift, bind, Option.bind, pure] at hst
      cases ha : s.reg a with
      | none => simp [ha] at hst
      | some _ =>
        simp only [ha] at hst
        cases hb : s.reg b with
        | none => simp [hb] at hst
        | some _ => simp [hb] at hst; cases hst; rfl
    | movI d _ =>
      simp only [exec] at hst; cases hst
      exact hforb_ok d _ hforb_body r hr
    | mov d s0 =>
      simp only [exec, lift, bind, Option.bind, pure] at hst
      cases hr0 : s.reg s0 with
      | none => simp [hr0] at hst
      | some _ =>
        simp [hr0] at hst; cases hst
        exact hforb_ok d _ hforb_body r hr
    | mov32 d s0 =>
      simp only [exec, lift, bind, Option.bind, pure] at hst
      cases hr0 : s.reg s0 with
      | none => simp [hr0] at hst
      | some _ =>
        simp [hr0] at hst; cases hst
        exact hforb_ok d _ hforb_body r hr
    | alu _ d s0 =>
      simp only [exec, lift, bind, Option.bind, pure] at hst
      cases ha : s.reg d with
      | none => simp [ha] at hst
      | some _ =>
        simp only [ha] at hst
        cases hb : s.reg s0 with
        | none => simp [hb] at hst
        | some _ =>
          simp [hb] at hst; cases hst
          exact hforb_ok d _ hforb_body r hr
    | aluI _ d _ =>
      simp only [exec, lift, bind, Option.bind, pure] at hst
      cases ha : s.reg d with
      | none => simp [ha] at hst
      | some _ =>
        simp [ha] at hst; cases hst
        exact hforb_ok d _ hforb_body r hr
    | shlI d _ =>
      simp only [exec, lift, bind, Option.bind, pure] at hst
      cases ha : s.reg d with
      | none => simp [ha] at hst
      | some _ =>
        simp [ha] at hst; cases hst
        exact hforb_ok d _ hforb_body r hr
    | shrI d _ =>
      simp only [exec, lift, bind, Option.bind, pure] at hst
      cases ha : s.reg d with
      | none => simp [ha] at hst
      | some _ =>
        simp [ha] at hst; cases hst
        exact hforb_ok d _ hforb_body r hr
    | ld d b disp =>
      simp only [exec, lift, bind, Option.bind, pure] at hst
      cases ha : s.reg b with
      | none => simp [ha] at hst
      | some av =>
        simp only [ha] at hst
        cases hrd : rd64 s.mem (ea av disp) with
        | none => simp [hrd] at hst
        | some _ =>
          simp [hrd] at hst; cases hst
          exact hforb_ok d _ hforb_body r hr
    | ld32 d b disp =>
      simp only [exec, lift, bind, Option.bind, pure] at hst
      cases ha : s.reg b with
      | none => simp [ha] at hst
      | some av =>
        simp only [ha] at hst
        cases hrd : rd32 s.mem (ea av disp) with
        | none => simp [hrd] at hst
        | some _ =>
          simp [hrd] at hst; cases hst
          exact hforb_ok d _ hforb_body r hr
    | ld8 d b disp =>
      simp only [exec, lift, bind, Option.bind, pure] at hst
      cases ha : s.reg b with
      | none => simp [ha] at hst
      | some av =>
        simp only [ha] at hst
        cases hrd : s.mem.rd (ea av disp) with
        | none => simp [hrd] at hst
        | some _ =>
          simp [hrd] at hst; cases hst
          exact hforb_ok d _ hforb_body r hr
    | lea d b _ =>
      simp only [exec, lift, bind, Option.bind, pure] at hst
      cases ha : s.reg b with
      | none => simp [ha] at hst
      | some _ =>
        simp [ha] at hst; cases hst
        exact hforb_ok d _ hforb_body r hr
    | call _ | jtab _ _ _ | jmpR _ | ret | st _ _ _ | st8 _ _ _ | push _ | pushI _ | pop _ | syscall =>
      exfalso; simp at hbody
  exact ⟨hm, hlen, hreg_ok, hrip_ok⟩

private theorem exec_pass_sim_step {lo hi : Nat} {forb : List Nat} {tbl : List (Nat × Ins × Nat)}
    (hsafe : ∀ e ∈ tbl, passEntryOK lo hi forb (tbl.map (·.1)) e = true)
    (htbl : ∀ e ∈ tbl, im e.1 = some e.2)
    {sA sB sA' : St}
    (hin : sA.rip ∈ tbl.map (·.1))
    (hrip : sA.rip = sB.rip) (hfl : sA.fl = sB.fl) (hm : MemExtends sA.mem sB.mem)
    (hlen : sA.regs.length = sB.regs.length)
    (hreg : ∀ j, j ∉ forb → sA.reg j = sB.reg j)
    (hstA : step im sA = .ok sA') :
    ∃ sB', step im sB = .ok sB' ∧ sA'.rip = sB'.rip ∧ sA'.fl = sB'.fl ∧
      sA'.mem = sA.mem ∧ sB'.mem = sB.mem ∧
      sA'.regs.length = sB'.regs.length ∧
      (∀ j, j ∉ forb → sA'.reg j = sB'.reg j) ∧
      (sA'.rip ∈ tbl.map (·.1) ∨ sA'.rip = hi) := by
  have hprops := exec_pass_step_props hsafe htbl hin hstA
  have hripA_in := hprops.2.2.2
  simp only [List.mem_map] at hin
  obtain ⟨e, he, herip⟩ := hin
  rcases e with ⟨eAddr, eins, elen⟩
  dsimp only at herip
  have him := htbl (eAddr, eins, elen) he
  rw [herip] at him
  simp only [step, him, ← hrip] at hstA ⊢
  have hall := hsafe (eAddr, eins, elen) he
  simp only [passEntryOK, Bool.and_eq_true, and_assoc] at hall
  obtain ⟨_, _, hbody⟩ := hall
  have hforb_check : (forb.all fun r => !(insWrites eins).contains r && !(insReads eins).contains r) = true := by
    cases eins with
    | call _ | jtab _ _ _ | jmpR _ | ret | st _ _ _ | st8 _ _ _ | push _ | pushI _ | pop _ | syscall =>
      simp at hbody
    | jmp _ | jcc _ _ | movI _ _ | mov _ _ | mov32 _ _ | alu _ _ _ | aluI _ _ _ | cmp _ _ | cmpI _ _
    | test _ _ | shlI _ _ | shrI _ _ | ld _ _ _ | ld32 _ _ _ | ld8 _ _ _ | lea _ _ _ =>
      simp only [Bool.and_eq_true, and_assoc] at hbody; exact hbody.1
  have hnotread : ∀ {r : Nat}, (insReads eins).contains r = true → sA.reg r = sB.reg r :=
    fun {r} hr => hreg r (fun hmf => by
      have ha := List.all_eq_true.1 hforb_check r hmf
      simp only [Bool.and_eq_true, Bool.not_eq_true'] at ha
      rw [hr] at ha; cases ha.2)
  have hset_eq : ∀ (d : Nat) (v : Option Nat) (j : Nat), j ∉ forb →
      (sA.set d v).reg j = (sB.set d v).reg j := by
    intro d v j hj
    by_cases hdj : d = j
    · subst hdj
      by_cases hdlA : d < sA.regs.length
      · have hdlB : d < sB.regs.length := hlen ▸ hdlA
        rw [reg_set_eq hdlA, reg_set_eq hdlB]
      · have hleA : sA.regs.length ≤ d := Nat.le_of_not_lt hdlA
        have hleB : sB.regs.length ≤ d := hlen ▸ hleA
        simp only [St.set, St.reg, list_set_of_le _ _ _ hleA, list_set_of_le _ _ _ hleB]
        exact hreg d hj
    · rw [reg_set_ne hdj, reg_set_ne hdj]; exact hreg j hj
  cases eins with
  | jmp t =>
    simp only [exec] at hstA
    cases hstA
    exact ⟨_, rfl, rfl, hfl, rfl, rfl, hlen, hreg, hripA_in⟩
  | jcc cc t =>
    simp only [exec, lift, bind, Option.bind, pure] at hstA
    cases hc : X86Int.cond sA.fl cc with
    | none => simp [hc] at hstA
    | some c =>
      simp [hc] at hstA
      cases hstA
      have hstB : exec (Ins.jcc cc t) elen sB =
          .ok { sB with rip := if c then t else sB.rip + elen } := by
        simp only [exec, lift, bind, Option.bind, pure, ← hfl, hc]
      refine ⟨_, hstB, ?_, hfl, rfl, rfl, hlen, hreg, hripA_in⟩
      cases c <;> simp [hrip]
  | movI d imm =>
    simp only [exec] at hstA
    cases hstA
    exact ⟨_, rfl, by simp [hrip], hfl, rfl, rfl,
      by simp [St.set, List.length_set, hlen],
      fun j hj => by simpa [St.reg] using hset_eq d _ j hj, hripA_in⟩
  | mov d r =>
    have hr := hnotread (r := r) (by simp [insReads])
    simp only [exec, lift, bind, Option.bind, pure] at hstA
    cases hrv : sA.reg r with
    | none => simp [hrv] at hstA
    | some rv =>
      simp [hrv] at hstA; cases hstA
      have hstB : exec (Ins.mov d r) elen sB =
          .ok { sB.set d (some rv) with rip := sB.rip + elen } := by
        simp only [exec, lift, bind, Option.bind, pure, ← hr, hrv]
      exact ⟨_, hstB, by simp [hrip], hfl, rfl, rfl,
        by simp [St.set, List.length_set, hlen],
        fun j hj => by simpa [St.reg] using hset_eq d _ j hj, hripA_in⟩
  | mov32 d r =>
    have hr := hnotread (r := r) (by simp [insReads])
    simp only [exec, lift, bind, Option.bind, pure] at hstA
    cases hrv : sA.reg r with
    | none => simp [hrv] at hstA
    | some rv =>
      simp [hrv] at hstA; cases hstA
      have hstB : exec (Ins.mov32 d r) elen sB =
          .ok { sB.set d (some (rv % 2 ^ 32)) with rip := sB.rip + elen } := by
        simp only [exec, lift, bind, Option.bind, pure, ← hr, hrv]
      exact ⟨_, hstB, by simp [hrip], hfl, rfl, rfl,
        by simp [St.set, List.length_set, hlen],
        fun j hj => by simpa [St.reg] using hset_eq d _ j hj, hripA_in⟩
  | alu op d r =>
    have hd := hnotread (r := d) (by simp [insReads])
    have hr := hnotread (r := r) (by simp [insReads])
    simp only [exec, lift, bind, Option.bind, pure] at hstA
    cases hdv : sA.reg d with
    | none => simp [hdv] at hstA
    | some dv =>
      simp [hdv] at hstA
      cases hrv : sA.reg r with
      | none => simp [hrv] at hstA
      | some rv =>
        simp [hrv] at hstA
        rcases hvf : aluF op dv rv with ⟨av, af⟩
        rw [hvf] at hstA; dsimp only at hstA; cases hstA
        have hstB : exec (Ins.alu op d r) elen sB =
            .ok { sB.set d (some av) with rip := sB.rip + elen, fl := af } := by
          simp only [exec, lift, bind, Option.bind, pure, ← hd, hdv, ← hr, hrv, hvf]
        exact ⟨_, hstB, by simp [hrip], rfl, rfl, rfl,
          by simp [St.set, List.length_set, hlen],
          fun j hj => by simpa [St.reg] using hset_eq d _ j hj, hripA_in⟩
  | aluI op d imm =>
    have hd := hnotread (r := d) (by simp [insReads])
    simp only [exec, lift, bind, Option.bind, pure] at hstA
    cases hdv : sA.reg d with
    | none => simp [hdv] at hstA
    | some dv =>
      simp [hdv] at hstA
      rcases hvf : aluF op dv (imm % W) with ⟨av, af⟩
      rw [hvf] at hstA; dsimp only at hstA; cases hstA
      have hstB : exec (Ins.aluI op d imm) elen sB =
          .ok { sB.set d (some av) with rip := sB.rip + elen, fl := af } := by
        simp only [exec, lift, bind, Option.bind, pure, ← hd, hdv, hvf]
      exact ⟨_, hstB, by simp [hrip], rfl, rfl, rfl,
        by simp [St.set, List.length_set, hlen],
        fun j hj => by simpa [St.reg] using hset_eq d _ j hj, hripA_in⟩
  | cmp a b =>
    have ha := hnotread (r := a) (by simp [insReads])
    have hb := hnotread (r := b) (by simp [insReads])
    simp only [exec, lift, bind, Option.bind, pure] at hstA
    cases hav : sA.reg a with
    | none => simp [hav] at hstA
    | some av =>
      simp [hav] at hstA
      cases hbv : sA.reg b with
      | none => simp [hbv] at hstA
      | some bv =>
        simp [hbv] at hstA; cases hstA
        have hstB : exec (Ins.cmp a b) elen sB =
            .ok { sB with rip := sB.rip + elen, fl := (aluF .sub av bv).2 } := by
          simp only [exec, lift, bind, Option.bind, pure, ← ha, hav, ← hb, hbv]
        exact ⟨_, hstB, by simp [hrip], rfl, rfl, rfl, hlen, hreg, hripA_in⟩
  | cmpI a imm =>
    have ha := hnotread (r := a) (by simp [insReads])
    simp only [exec, lift, bind, Option.bind, pure] at hstA
    cases hav : sA.reg a with
    | none => simp [hav] at hstA
    | some av =>
      simp [hav] at hstA; cases hstA
      have hstB : exec (Ins.cmpI a imm) elen sB =
          .ok { sB with rip := sB.rip + elen, fl := (aluF .sub av (imm % W)).2 } := by
        simp only [exec, lift, bind, Option.bind, pure, ← ha, hav]
      exact ⟨_, hstB, by simp [hrip], rfl, rfl, rfl, hlen, hreg, hripA_in⟩
  | test a b =>
    have ha := hnotread (r := a) (by simp [insReads])
    have hb := hnotread (r := b) (by simp [insReads])
    simp only [exec, lift, bind, Option.bind, pure] at hstA
    cases hav : sA.reg a with
    | none => simp [hav] at hstA
    | some av =>
      simp [hav] at hstA
      cases hbv : sA.reg b with
      | none => simp [hbv] at hstA
      | some bv =>
        simp [hbv] at hstA; cases hstA
        have hstB : exec (Ins.test a b) elen sB =
            .ok { sB with rip := sB.rip + elen, fl := (aluF .and av bv).2 } := by
          simp only [exec, lift, bind, Option.bind, pure, ← ha, hav, ← hb, hbv]
        exact ⟨_, hstB, by simp [hrip], rfl, rfl, rfl, hlen, hreg, hripA_in⟩
  | shlI d n =>
    have hd := hnotread (r := d) (by simp [insReads])
    simp only [exec, lift, bind, Option.bind, pure] at hstA
    cases hdv : sA.reg d with
    | none => simp [hdv] at hstA
    | some dv =>
      simp [hdv] at hstA; cases hstA
      have hstB : exec (Ins.shlI d n) elen sB =
          .ok { sB.set d (some (dv * 2 ^ (n % 64) % W)) with rip := sB.rip + elen, fl := Flags.undef } := by
        simp only [exec, lift, bind, Option.bind, pure, ← hd, hdv]
      exact ⟨_, hstB, by simp [hrip], rfl, rfl, rfl,
        by simp [St.set, List.length_set, hlen],
        fun j hj => by simpa [St.reg] using hset_eq d _ j hj, hripA_in⟩
  | shrI d n =>
    have hd := hnotread (r := d) (by simp [insReads])
    simp only [exec, lift, bind, Option.bind, pure] at hstA
    cases hdv : sA.reg d with
    | none => simp [hdv] at hstA
    | some dv =>
      simp [hdv] at hstA; cases hstA
      have hstB : exec (Ins.shrI d n) elen sB =
          .ok { sB.set d (some (dv / 2 ^ (n % 64))) with rip := sB.rip + elen, fl := Flags.undef } := by
        simp only [exec, lift, bind, Option.bind, pure, ← hd, hdv]
      exact ⟨_, hstB, by simp [hrip], rfl, rfl, rfl,
        by simp [St.set, List.length_set, hlen],
        fun j hj => by simpa [St.reg] using hset_eq d _ j hj, hripA_in⟩
  | ld d b off =>
    have hb := hnotread (r := b) (by simp [insReads])
    simp only [exec, lift, bind, Option.bind, pure] at hstA
    cases hbv : sA.reg b with
    | none => simp [hbv] at hstA
    | some bv =>
      simp [hbv] at hstA
      cases hrd : rd64 sA.mem (ea bv off) with
      | none => simp [hrd] at hstA
      | some mv =>
        simp [hrd] at hstA; cases hstA
        have hstB : exec (Ins.ld d b off) elen sB =
            .ok { sB.set d (some mv) with rip := sB.rip + elen } := by
          simp only [exec, lift, bind, Option.bind, pure, ← hb, hbv, rd64_mono hm hrd]
        exact ⟨_, hstB, by simp [hrip], hfl, rfl, rfl,
          by simp [St.set, List.length_set, hlen],
          fun j hj => by simpa [St.reg] using hset_eq d _ j hj, hripA_in⟩
  | ld32 d b off =>
    have hb := hnotread (r := b) (by simp [insReads])
    simp only [exec, lift, bind, Option.bind, pure] at hstA
    cases hbv : sA.reg b with
    | none => simp [hbv] at hstA
    | some bv =>
      simp [hbv] at hstA
      cases hrd : rd32 sA.mem (ea bv off) with
      | none => simp [hrd] at hstA
      | some mv =>
        simp [hrd] at hstA; cases hstA
        have hstB : exec (Ins.ld32 d b off) elen sB =
            .ok { sB.set d (some mv) with rip := sB.rip + elen } := by
          simp only [exec, lift, bind, Option.bind, pure, ← hb, hbv, rd32_mono hm hrd]
        exact ⟨_, hstB, by simp [hrip], hfl, rfl, rfl,
          by simp [St.set, List.length_set, hlen],
          fun j hj => by simpa [St.reg] using hset_eq d _ j hj, hripA_in⟩
  | ld8 d b off =>
    have hb := hnotread (r := b) (by simp [insReads])
    simp only [exec, lift, bind, Option.bind, pure] at hstA
    cases hbv : sA.reg b with
    | none => simp [hbv] at hstA
    | some bv =>
      simp [hbv] at hstA
      cases hrd : sA.mem.rd (ea bv off) with
      | none => simp [hrd] at hstA
      | some mv =>
        simp [hrd] at hstA; cases hstA
        have hstB : exec (Ins.ld8 d b off) elen sB =
            .ok { sB.set d (some mv) with rip := sB.rip + elen } := by
          simp only [exec, lift, bind, Option.bind, pure, ← hb, hbv, hm _ _ hrd]
        exact ⟨_, hstB, by simp [hrip], hfl, rfl, rfl,
          by simp [St.set, List.length_set, hlen],
          fun j hj => by simpa [St.reg] using hset_eq d _ j hj, hripA_in⟩
  | lea d b off =>
    have hb := hnotread (r := b) (by simp [insReads])
    simp only [exec, lift, bind, Option.bind, pure] at hstA
    cases hbv : sA.reg b with
    | none => simp [hbv] at hstA
    | some bv =>
      simp [hbv] at hstA; cases hstA
      have hstB : exec (Ins.lea d b off) elen sB =
          .ok { sB.set d (some (ea bv off)) with rip := sB.rip + elen } := by
        simp only [exec, lift, bind, Option.bind, pure, ← hb, hbv]
      exact ⟨_, hstB, by simp [hrip], hfl, rfl, rfl,
        by simp [St.set, List.length_set, hlen],
        fun j hj => by simpa [St.reg] using hset_eq d _ j hj, hripA_in⟩
  | call _ | jtab _ _ _ | jmpR _ | ret | st _ _ _ | st8 _ _ _ | push _ | pushI _ | pop _ | syscall =>
    exfalso; simp at hbody

private theorem tbl_bounds {lo hi : Nat} {forb : List Nat} {tbl : List (Nat × Ins × Nat)}
    (hsafe : ∀ e ∈ tbl, passEntryOK lo hi forb (tbl.map (·.1)) e = true)
    {a : Nat} (ha : a ∈ tbl.map (·.1)) : lo ≤ a ∧ a < hi := by
  simp only [List.mem_map] at ha
  obtain ⟨e, he, rfl⟩ := ha
  have h := hsafe e he
  simp only [passEntryOK, Bool.and_eq_true, and_assoc, decide_eq_true_eq] at h
  exact ⟨h.1, h.2.1⟩

private theorem step_ne_exit_of_ne_syscall {lo hi : Nat} {forb : List Nat} {tbl : List (Nat × Ins × Nat)}
    (hsafe : ∀ e ∈ tbl, passEntryOK lo hi forb (tbl.map (·.1)) e = true)
    (htbl : ∀ e ∈ tbl, im e.1 = some e.2)
    {s : St} (hin : s.rip ∈ tbl.map (·.1)) {c : Nat} : step im s ≠ .exit c := by
  simp only [List.mem_map] at hin
  obtain ⟨e, he, herip⟩ := hin
  have him := htbl e he
  rw [herip] at him
  have hall := hsafe e he
  simp only [passEntryOK, Bool.and_eq_true, and_assoc] at hall
  have hbody := hall.2.2
  simp only [step, him]
  rcases e with ⟨_, eins, _⟩
  dsimp only at hbody ⊢
  cases eins with
  | call _ | jtab _ _ _ | jmpR _ | ret | st _ _ _ | st8 _ _ _ | push _ | pushI _ | pop _ | syscall =>
    simp at hbody
  | jmp _ | movI _ _ =>
    simp only [exec]; intro h; cases h
  | jcc _ _ | mov _ _ | mov32 _ _ | alu _ _ _ | aluI _ _ _ | cmp _ _ | cmpI _ _
  | test _ _ | shlI _ _ | shrI _ _ | ld _ _ _ | ld32 _ _ _ | ld8 _ _ _ | lea _ _ _ =>
    simp only [exec, lift]
    split <;> intro h <;> cases h

private theorem pass_step_invs {lo hi : Nat} {forb : List Nat} {tbl : List (Nat × Ins × Nat)}
    (hsafe : ∀ e ∈ tbl, passEntryOK lo hi forb (tbl.map (·.1)) e = true)
    (htbl : ∀ e ∈ tbl, im e.1 = some e.2)
    {s s' : St} (hin : s.rip ∈ tbl.map (·.1)) (hst : step im s = .ok s') :
    s'.mem = s.mem ∧ s'.regs.length = s.regs.length ∧
    (∀ r ∈ forb, s'.reg r = s.reg r) ∧
    (s'.rip ∈ tbl.map (·.1) ∨ s'.rip = hi) ∧
    (RegDefExcept11 s → RegDefExcept11 s') ∧
    (RegsLtW s → MemBytesLt256 s.mem → RegsLtW s') := by
  have hp := exec_pass_step_props hsafe htbl hin hst
  refine ⟨hp.1, hp.2.1, hp.2.2.1, hp.2.2.2, ?_, ?_⟩
  · intro hd
    simp only [List.mem_map] at hin
    obtain ⟨e, he, herip⟩ := hin
    have him := htbl e he
    rw [herip] at him
    simp only [step, him] at hst
    exact exec_preserves_regDefExcept11 hd hst
  · intro hlt hm
    simp only [List.mem_map] at hin
    obtain ⟨e, he, herip⟩ := hin
    have him := htbl e he
    rw [herip] at him
    simp only [step, him] at hst
    exact exec_preserves_regsLtW hlt hm (pass_no_store_of_safe hsafe he) hst

private theorem pass_run_trichotomy {lo hi : Nat} {forb : List Nat} {tbl : List (Nat × Ins × Nat)}
    (hsafe : ∀ e ∈ tbl, passEntryOK lo hi forb (tbl.map (·.1)) e = true)
    (htbl : ∀ e ∈ tbl, im e.1 = some e.2) :
    ∀ (n : Nat) (s : St), s.rip ∈ tbl.map (·.1) →
      run im n (.ok s) = .stuck ∨
      (∃ s', run im n (.ok s) = .ok s' ∧ s'.rip ∈ tbl.map (·.1) ∧ s'.mem = s.mem ∧
        s'.regs.length = s.regs.length ∧ (∀ r ∈ forb, s'.reg r = s.reg r) ∧
        (RegDefExcept11 s → RegDefExcept11 s') ∧ (RegsLtW s → MemBytesLt256 s.mem → RegsLtW s')) ∨
      (∃ k s_hi, 0 < k ∧ k ≤ n ∧ run im k (.ok s) = .ok s_hi ∧ s_hi.rip = hi ∧
        s_hi.mem = s.mem ∧ s_hi.regs.length = s.regs.length ∧ (∀ r ∈ forb, s_hi.reg r = s.reg r) ∧
        (RegDefExcept11 s → RegDefExcept11 s_hi) ∧ (RegsLtW s → MemBytesLt256 s.mem → RegsLtW s_hi))
  | 0, s, hin =>
    Or.inr (Or.inl ⟨s, rfl, hin, rfl, rfl, fun _ _ => rfl, fun h => h, fun h _ => h⟩)
  | n + 1, s, hin => by
    simp only [run]
    cases hst : step im s with
    | stuck => rw [run_stuck]; exact Or.inl rfl
    | exit c => exact False.elim (step_ne_exit_of_ne_syscall hsafe htbl hin hst)
    | ok s1 =>
      have hi1 := pass_step_invs hsafe htbl hin hst
      rcases hi1.2.2.2.1 with hin1 | hhi1
      · rcases pass_run_trichotomy hsafe htbl n s1 hin1 with hstk | ⟨s', hrun', hrest⟩ | ⟨k, shi, hk1, hkn, hrun', hrest⟩
        · exact Or.inl hstk
        · exact Or.inr (Or.inl ⟨s', hrun', hrest.1, hrest.2.1.trans hi1.1, hrest.2.2.1.trans hi1.2.1,
            fun r hr => (hrest.2.2.2.1 r hr).trans (hi1.2.2.1 r hr),
            fun hd => hrest.2.2.2.2.1 (hi1.2.2.2.2.1 hd),
            fun hl hm => hrest.2.2.2.2.2 (hi1.2.2.2.2.2 hl hm) (by rw [hi1.1]; exact hm)⟩)
        · exact Or.inr (Or.inr ⟨k + 1, shi, Nat.zero_lt_succ k, Nat.succ_le_succ hkn,
            by rw [run, hst]; exact hrun', hrest.1, hrest.2.1.trans hi1.1, hrest.2.2.1.trans hi1.2.1,
            fun r hr => (hrest.2.2.2.1 r hr).trans (hi1.2.2.1 r hr),
            fun hd => hrest.2.2.2.2.1 (hi1.2.2.2.2.1 hd),
            fun hl hm => hrest.2.2.2.2.2 (hi1.2.2.2.2.2 hl hm) (by rw [hi1.1]; exact hm)⟩)
      · exact Or.inr (Or.inr ⟨1, s1, Nat.zero_lt_succ 0, Nat.succ_le_succ (Nat.zero_le n),
          by rw [run, hst]; rfl, hhi1, hi1.1, hi1.2.1, hi1.2.2.1, hi1.2.2.2.2.1, hi1.2.2.2.2.2⟩)

private theorem exec_pass_run_sim_all {lo hi : Nat} {forb : List Nat} {tbl : List (Nat × Ins × Nat)}
    (hsafe : ∀ e ∈ tbl, passEntryOK lo hi forb (tbl.map (·.1)) e = true)
    (htbl : ∀ e ∈ tbl, im e.1 = some e.2)
    (hhi_not : hi ∉ tbl.map (·.1))
    (hnoRet : ∀ {s : St} {n : Nat} {s' : St}, s.rip = hi → 0 < n → run im n (.ok s) = .ok s' → s'.rip ≠ hi) :
    ∀ (N : Nat) {sA sB sA' : St},
      sA.rip ∈ tbl.map (·.1) →
      sA.rip = sB.rip → sA.fl = sB.fl → MemExtends sA.mem sB.mem →
      sA.regs.length = sB.regs.length → (∀ j, j ∉ forb → sA.reg j = sB.reg j) →
      run im N (.ok sA) = .ok sA' → sA'.rip = hi →
      (∃ sB', run im N (.ok sB) = .ok sB' ∧ sA'.rip = sB'.rip ∧ sA'.fl = sB'.fl ∧
        sA'.mem = sA.mem ∧ sB'.mem = sB.mem ∧ sA'.regs.length = sB'.regs.length ∧
        (∀ j, j ∉ forb → sA'.reg j = sB'.reg j) ∧ (∀ r ∈ forb, sB'.reg r = sB.reg r)) ∧
      (∀ k, k < N → ∃ sAk sBk,
        run im k (.ok sA) = .ok sAk ∧ run im k (.ok sB) = .ok sBk ∧
        sAk.rip ∈ tbl.map (·.1) ∧ sAk.rip = sBk.rip ∧ sAk.fl = sBk.fl ∧
        sAk.mem = sA.mem ∧ sBk.mem = sB.mem ∧ sAk.regs.length = sBk.regs.length ∧
        (∀ j, j ∉ forb → sAk.reg j = sBk.reg j) ∧ (∀ r ∈ forb, sBk.reg r = sB.reg r))
  | 0, sA, sB, sA', hinA, _, _, _, _, _, hrun, hhi => by
    simp [run] at hrun; subst hrun; rw [hhi] at hinA; exact False.elim (hhi_not hinA)
  | N + 1, sA, sB, sA', hinA, hrip, hfl, hm, hlen, hreg, hrun, hhi => by
    cases hstA : step im sA with
    | stuck => rw [run, hstA, run_stuck] at hrun; cases hrun
    | exit c => exact False.elim (step_ne_exit_of_ne_syscall hsafe htbl hinA hstA)
    | ok sA1 =>
      rw [run, hstA] at hrun
      obtain ⟨sB1, hstB1, hrip1, hfl1, hmA1, hmB1, hlen1, hreg1, hA1_in⟩ :=
        exec_pass_sim_step hsafe htbl hinA hrip hfl hm hlen hreg hstA
      have hinB : sB.rip ∈ tbl.map (·.1) := by rw [← hrip]; exact hinA
      have hforbB1 := (exec_pass_step_props hsafe htbl hinB hstB1).2.2.1
      rcases hA1_in with hinA1 | hhiA1
      · have ih := exec_pass_run_sim_all hsafe htbl hhi_not hnoRet N hinA1 hrip1 hfl1
          (by rw [hmA1, hmB1]; exact hm) hlen1 hreg1 hrun hhi
        refine ⟨?_, ?_⟩
        · obtain ⟨sB', hrunB', hrip', hfl', hmA', hmB', hlen', hreg', hforb'⟩ := ih.1
          exact ⟨sB', by rw [run, hstB1]; exact hrunB', hrip', hfl',
            hmA'.trans hmA1, hmB'.trans hmB1, hlen', hreg',
            fun r hr => (hforb' r hr).trans (hforbB1 r hr)⟩
        · intro k hk
          cases k with
          | zero => exact ⟨sA, sB, rfl, rfl, hinA, hrip, hfl, rfl, rfl, hlen, hreg, fun _ _ => rfl⟩
          | succ k' =>
            obtain ⟨sAk, sBk, hrAk, hrBk, hk_in, hk_rip, hk_fl, hk_mA, hk_mB, hk_len, hk_reg, hk_forb⟩ :=
              ih.2 k' (Nat.lt_of_succ_lt_succ hk)
            exact ⟨sAk, sBk, by rw [run, hstA]; exact hrAk, by rw [run, hstB1]; exact hrBk,
              hk_in, hk_rip, hk_fl, hk_mA.trans hmA1, hk_mB.trans hmB1, hk_len, hk_reg,
              fun r hr => (hk_forb r hr).trans (hforbB1 r hr)⟩
      · cases N with
        | zero =>
          simp [run] at hrun; subst hrun
          refine ⟨⟨sB1, by rw [run, hstB1]; rfl, hrip1, hfl1, hmA1, hmB1, hlen1, hreg1, hforbB1⟩, ?_⟩
          intro k hk
          cases k with
          | zero => exact ⟨sA, sB, rfl, rfl, hinA, hrip, hfl, rfl, rfl, hlen, hreg, fun _ _ => rfl⟩
          | succ k' => omega
        | succ m => exact False.elim (hnoRet hhiA1 (Nat.zero_lt_succ m) hrun hhi)

private def tailSpecClosed : Bool :=
  SporosCheckFlat.tailSpec.all fun e =>
    decide (e.1 < 143) &&
    match e.2.1 with
    | .call _ | .ret | .jmpR _ | .jtab _ _ _ => false
    | .syscall =>
      decide (e.1 + e.2.2 = 143) ||
      SporosCheckFlat.tailSpec.any (fun e' => e'.1 == e.1 + e.2.2 && e'.1 != 0)
    | .jmp t =>
      SporosCheckFlat.tailSpec.any (fun e' => e'.1 == t && e'.1 != 0)
    | .jcc _ t =>
      SporosCheckFlat.tailSpec.any (fun e' => e'.1 == t && e'.1 != 0) &&
      SporosCheckFlat.tailSpec.any (fun e' => e'.1 == e.1 + e.2.2 && e'.1 != 0)
    | _ =>
      SporosCheckFlat.tailSpec.any (fun e' => e'.1 == e.1 + e.2.2 && e'.1 != 0)

private theorem tailSpecClosed_ok : tailSpecClosed = true := by decide +kernel

private theorem tailSpec_any_off {t : Nat}
    (h : SporosCheckFlat.tailSpec.any (fun e' => e'.1 == t && e'.1 != 0) = true) :
    t ∈ SporosCheckFlat.tailSpec.map (·.1) ∧ t ≠ 0 ∧ t < 143 := by
  simp only [List.any_eq_true, Bool.and_eq_true, beq_iff_eq, bne_iff_ne] at h
  obtain ⟨e', he', he1, hne0⟩ := h
  have hall := List.all_eq_true.1 tailSpecClosed_ok e' he'
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hall
  refine ⟨List.mem_map.2 ⟨e', he', he1⟩, ?_, ?_⟩
  · rw [← he1]; exact hne0
  · rw [← he1]; exact hall.1

private theorem tailSpec_step {s s' : St} {off : Nat}
    (hoff : off ∈ SporosCheckFlat.tailSpec.map (·.1))
    (hrip : s.rip = L_vote + off) (hst : step im s = .ok s') :
    s'.rip = L_core_end ∨
    ∃ off', off' ∈ SporosCheckFlat.tailSpec.map (·.1) ∧ off' ≠ 0 ∧ s'.rip = L_vote + off' := by
  simp only [List.mem_map] at hoff
  obtain ⟨⟨off', ins, len⟩, hmem, h0⟩ := hoff
  dsimp only at h0; rw [h0] at hmem
  have hall := List.all_eq_true.1 tailSpecClosed_ok (off, ins, len) hmem
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hall
  obtain ⟨hlt143, hbody⟩ := hall
  have him_eq : im s.rip = some (SporosCheckFlat.shiftIns L_vote ins, len) := by
    rw [hrip, im_vote_eq_tailImAt (by omega) (by unfold L_vote L_core_end at *; omega),
      SporosCheckFlat.tailImAt_base_spec hmem]
  simp only [step, him_eq] at hst
  have hrc := exec_rip_cases hst
  have hW : L_vote + 143 < W := by unfold L_vote W; decide
  cases ins with
  | call _ | ret | jmpR _ | jtab _ _ _ => simp at hbody
  | syscall =>
    dsimp only [SporosCheckFlat.shiftIns, ripNextCond] at hrc
    simp only [Bool.or_eq_true, decide_eq_true_eq] at hbody
    rcases hbody with h143 | hany
    · left; rw [hrc, hrip, Nat.add_assoc, h143]; unfold L_vote L_core_end; omega
    · obtain ⟨hm', hnz, _⟩ := tailSpec_any_off hany
      right; exact ⟨off + len, hm', hnz, by rw [hrc, hrip, Nat.add_assoc]⟩
  | jmp t =>
    dsimp only [SporosCheckFlat.shiftIns, ripNextCond] at hrc
    obtain ⟨hm', hnz, hlt⟩ := tailSpec_any_off hbody
    right; refine ⟨t, hm', hnz, ?_⟩
    rw [hrc, Nat.mod_eq_of_lt (by omega)]
  | jcc _ t =>
    dsimp only [SporosCheckFlat.shiftIns, ripNextCond] at hrc
    simp only [Bool.and_eq_true] at hbody
    rcases hrc with h1 | h2
    · obtain ⟨hm', hnz, hlt⟩ := tailSpec_any_off hbody.1
      right; refine ⟨t, hm', hnz, ?_⟩
      rw [h1, Nat.mod_eq_of_lt (by omega)]
    · obtain ⟨hm', hnz, _⟩ := tailSpec_any_off hbody.2
      right; exact ⟨off + len, hm', hnz, by rw [h2, hrip, Nat.add_assoc]⟩
  | movI _ _ | mov _ _ | mov32 _ _ | alu _ _ _ | aluI _ _ _ | cmp _ _ | cmpI _ _
  | test _ _ | shlI _ _ | shrI _ _ | ld _ _ _ | ld32 _ _ _ | ld8 _ _ _ | st _ _ _
  | st8 _ _ _ | lea _ _ _ | push _ | pushI _ | pop _ =>
    dsimp only [SporosCheckFlat.shiftIns, ripNextCond] at hrc
    obtain ⟨hm', hnz, _⟩ := tailSpec_any_off hbody
    right; exact ⟨off + len, hm', hnz, by rw [hrc, hrip, Nat.add_assoc]⟩

private theorem tailSpec_run_after :
    ∀ (n : Nat) {s s' : St},
      (s.rip = L_core_end ∨ ∃ off ∈ SporosCheckFlat.tailSpec.map (·.1), off ≠ 0 ∧ s.rip = L_vote + off) →
      run im n (.ok s) = .ok s' → L_vote < s'.rip ∧
      (s'.rip = L_core_end ∨ ∃ off ∈ SporosCheckFlat.tailSpec.map (·.1), off ≠ 0 ∧ s'.rip = L_vote + off)
  | 0, s, s', hs, hrun => by
    simp [run] at hrun; subst hrun
    refine ⟨?_, hs⟩
    rcases hs with hce | ⟨off, _, hnz, hrip⟩
    · rw [hce]; unfold L_vote L_core_end; omega
    · rw [hrip]; omega
  | n + 1, s, s', hs, hrun => by
    simp only [run] at hrun
    rcases hs with hce | ⟨off, hoff, _, hrip⟩
    · have hstk : step im s = .stuck := by simp [step, hce, im_core_end_none]
      rw [hstk, run_stuck] at hrun; cases hrun
    · cases hst : step im s with
      | stuck => rw [hst, run_stuck] at hrun; cases hrun
      | exit c => rw [hst, run_exit] at hrun; cases hrun
      | ok s1 =>
        rw [hst] at hrun
        rcases tailSpec_step hoff hrip hst with h1 | h1
        · exact tailSpec_run_after n (Or.inl h1) hrun
        · exact tailSpec_run_after n (Or.inr h1) hrun

private theorem no_return_Lvote {s : St} {n : Nat} {s' : St}
    (hs : s.rip = L_vote) (hn : 0 < n) (hrun : run im n (.ok s) = .ok s') :
    s'.rip ≠ L_vote ∧ L_vote < s'.rip := by
  cases n with
  | zero => omega
  | succ m =>
    simp only [run] at hrun
    cases hst : step im s with
    | stuck => rw [hst, run_stuck] at hrun; cases hrun
    | exit c => rw [hst, run_exit] at hrun; cases hrun
    | ok s1 =>
      rw [hst] at hrun
      have h0 : 0 ∈ SporosCheckFlat.tailSpec.map (·.1) := by decide
      have h1 := tailSpec_step h0 (by rw [hs]; rfl) hst
      have h2 := (tailSpec_run_after m h1 hrun).1
      exact ⟨by omega, h2⟩

private def p2NoRetHead : Bool :=
  p2List.all (·.all fun e => match e.2.1 with
    | .jmp t => t != L_p2
    | .jcc _ t => t != L_p2 && (e.1 + e.2.2 != L_p2)
    | _ => e.1 + e.2.2 != L_p2)

private def p3NoRetHead : Bool :=
  p3List.all (·.all fun e => match e.2.1 with
    | .jmp t => t != L_p3
    | .jcc _ t => t != L_p3 && (e.1 + e.2.2 != L_p3)
    | _ => e.1 + e.2.2 != L_p3)

private theorem p2NoRetHead_ok : p2NoRetHead = true := by decide +kernel
private theorem p3NoRetHead_ok : p3NoRetHead = true := by decide +kernel

private theorem p1_mem_Lp1 : L_p1 ∈ p1Tbl.map (·.1) := by
  have h : (p1Slice.map (·.1)).contains L_p1 = true := by decide +kernel
  rw [p1Tbl_eq_p1Slice]; exact List.contains_iff_mem.1 h

private theorem p2_mem_Lp2 : L_p2 ∈ p2Tbl.map (·.1) := by
  have h : (p2Slice.map (·.1)).contains L_p2 = true := by decide +kernel
  rw [p2Tbl_eq_p2Slice]; exact List.contains_iff_mem.1 h

private theorem p3_mem_Lp3 : L_p3 ∈ p3Tbl.map (·.1) := by
  have h : (p3Slice.map (·.1)).contains L_p3 = true := by decide +kernel
  rw [p3Tbl_eq_p3Slice]; exact List.contains_iff_mem.1 h

private theorem pass_no_head_step {lo hi : Nat} {forb : List Nat} {tbl : List (Nat × Ins × Nat)}
    (hsafe : ∀ e ∈ tbl, passEntryOK lo hi forb (tbl.map (·.1)) e = true)
    (htbl : ∀ e ∈ tbl, im e.1 = some e.2)
    (hno : ∀ e ∈ tbl, (match e.2.1 with
      | .jmp t => t != lo
      | .jcc _ t => t != lo && (e.1 + e.2.2 != lo)
      | _ => e.1 + e.2.2 != lo) = true)
    {s s' : St} (hin : s.rip ∈ tbl.map (·.1)) (hst : step im s = .ok s') :
    s'.rip ≠ lo := by
  simp only [List.mem_map] at hin
  obtain ⟨e, he, herip⟩ := hin
  have him := htbl e he; rw [herip] at him
  simp only [step, him] at hst
  have hrc := exec_rip_cases hst
  have hno_e := hno e he
  have hbody := (hsafe e he)
  simp only [passEntryOK, Bool.and_eq_true, and_assoc] at hbody
  rcases e with ⟨eAddr, eins, elen⟩
  dsimp only [ripNextCond] at hrc hno_e hbody
  cases eins with
  | jmp t => rw [hrc]; exact bne_iff_ne.1 hno_e
  | jcc cc t =>
    simp only [Bool.and_eq_true, bne_iff_ne] at hno_e
    rcases hrc with h1 | h2
    · rw [h1]; exact hno_e.1
    · rw [h2, ← herip]; exact hno_e.2
  | call _ | jtab _ _ _ | jmpR _ | ret | st _ _ _ | st8 _ _ _ | push _ | pushI _ | pop _ | syscall =>
    exfalso; simp at hbody
  | movI _ _ | mov _ _ | mov32 _ _ | alu _ _ _ | aluI _ _ _ | cmp _ _ | cmpI _ _
  | test _ _ | shlI _ _ | shrI _ _ | ld _ _ _ | ld32 _ _ _ | ld8 _ _ _ | lea _ _ _ =>
    rw [hrc, ← herip]; exact bne_iff_ne.1 hno_e

private theorem p2_no_head : ∀ e ∈ p2Tbl, (match e.2.1 with
    | .jmp t => t != L_p2
    | .jcc _ t => t != L_p2 && (e.1 + e.2.2 != L_p2)
    | _ => e.1 + e.2.2 != L_p2) = true := by
  intro e he
  rw [p2Tbl_eq_p2Slice, p2Slice_def] at he
  obtain ⟨ch, hch, hech⟩ := List.mem_flatten.1 he
  exact List.all_eq_true.1 (List.all_eq_true.1 p2NoRetHead_ok ch hch) e hech

private theorem p3_no_head : ∀ e ∈ p3Tbl, (match e.2.1 with
    | .jmp t => t != L_p3
    | .jcc _ t => t != L_p3 && (e.1 + e.2.2 != L_p3)
    | _ => e.1 + e.2.2 != L_p3) = true := by
  intro e he
  rw [p3Tbl_eq_p3Slice, p3Slice_def] at he
  obtain ⟨ch, hch, hech⟩ := List.mem_flatten.1 he
  exact List.all_eq_true.1 (List.all_eq_true.1 p3NoRetHead_ok ch hch) e hech

private theorem no_return_Lp3 :
    ∀ {s : St} {n : Nat} {s' : St}, s.rip = L_p3 → 0 < n → run im n (.ok s) = .ok s' →
      s'.rip ≠ L_p3 ∧ L_p3 < s'.rip := by
  have aux : ∀ (n : Nat) {s s' : St}, s.rip ∈ p3Tbl.map (·.1) → 0 < n →
      run im n (.ok s) = .ok s' → s'.rip ≠ L_p3 ∧ L_p3 < s'.rip := by
    intro n
    induction n with
    | zero => intro _ _ _ hn; omega
    | succ m ih =>
      intro s s' hin _ hrun
      simp only [run] at hrun
      cases hst : step im s with
      | stuck => rw [hst, run_stuck] at hrun; cases hrun
      | exit c => exact False.elim (step_ne_exit_of_ne_syscall p3SafeCheck p3Tbl_im hin hst)
      | ok s1 =>
        rw [hst] at hrun
        have hp := exec_pass_step_props p3SafeCheck p3Tbl_im hin hst
        have hnh := pass_no_head_step p3SafeCheck p3Tbl_im p3_no_head hin hst
        rcases hp.2.2.2 with hin1 | hv1
        · cases m with
          | zero =>
            simp [run] at hrun; subst hrun
            have hb := tbl_bounds p3SafeCheck hin1
            exact ⟨hnh, by omega⟩
          | succ m' => exact ih hin1 (Nat.zero_lt_succ m') hrun
        · have hv_gt : L_p3 < L_vote := by unfold L_p3 L_vote; omega
          cases m with
          | zero =>
            simp [run] at hrun; subst hrun
            exact ⟨by rw [hv1]; unfold L_p3 L_vote; omega, by rw [hv1]; exact hv_gt⟩
          | succ m' =>
            have hlv := no_return_Lvote hv1 (Nat.zero_lt_succ m') hrun
            exact ⟨by omega, by omega⟩
  intro s n s' hrip hn hrun
  exact aux n (by rw [hrip]; exact p3_mem_Lp3) hn hrun

private theorem no_return_Lp2 :
    ∀ {s : St} {n : Nat} {s' : St}, s.rip = L_p2 → 0 < n → run im n (.ok s) = .ok s' →
      s'.rip ≠ L_p2 ∧ L_p2 < s'.rip := by
  have aux : ∀ (n : Nat) {s s' : St}, s.rip ∈ p2Tbl.map (·.1) → 0 < n →
      run im n (.ok s) = .ok s' → s'.rip ≠ L_p2 ∧ L_p2 < s'.rip := by
    intro n
    induction n with
    | zero => intro _ _ _ hn; omega
    | succ m ih =>
      intro s s' hin _ hrun
      simp only [run] at hrun
      cases hst : step im s with
      | stuck => rw [hst, run_stuck] at hrun; cases hrun
      | exit c => exact False.elim (step_ne_exit_of_ne_syscall p2SafeCheck p2Tbl_im hin hst)
      | ok s1 =>
        rw [hst] at hrun
        have hp := exec_pass_step_props p2SafeCheck p2Tbl_im hin hst
        have hnh := pass_no_head_step p2SafeCheck p2Tbl_im p2_no_head hin hst
        rcases hp.2.2.2 with hin1 | hv1
        · cases m with
          | zero =>
            simp [run] at hrun; subst hrun
            have hb := tbl_bounds p2SafeCheck hin1
            exact ⟨hnh, by omega⟩
          | succ m' => exact ih hin1 (Nat.zero_lt_succ m') hrun
        · have hv_gt : L_p2 < L_p3 := by unfold L_p2 L_p3; omega
          cases m with
          | zero =>
            simp [run] at hrun; subst hrun
            exact ⟨by rw [hv1]; unfold L_p2 L_p3; omega, by rw [hv1]; exact hv_gt⟩
          | succ m' =>
            have hlp3 := no_return_Lp3 hv1 (Nat.zero_lt_succ m') hrun
            exact ⟨by omega, by omega⟩
  intro s n s' hrip hn hrun
  exact aux n (by rw [hrip]; exact p2_mem_Lp2) hn hrun

private theorem p1_hi_not_mem : L_p2 ∉ p1Tbl.map (·.1) := by
  intro h; have := tbl_bounds p1SafeCheck h; omega
private theorem p2_hi_not_mem : L_p3 ∉ p2Tbl.map (·.1) := by
  intro h; have := tbl_bounds p2SafeCheck h; omega
private theorem p3_hi_not_mem : L_vote ∉ p3Tbl.map (·.1) := by
  intro h; have := tbl_bounds p3SafeCheck h; omega

/-! ### Submemory isolation, single-byte `memFlips` locality, and `TripledState` -/

private theorem rdN_getElem? : ∀ {m : Mem} {a n : Nat} {xs : List Nat} {k : Nat},
    m.rdN a n = some xs → k < n → m.rd (a + k) = xs[k]?
  | _, _, 0, _, _, _, hk => by omega
  | m, a, n + 1, xs, k, h, hk => by
    simp only [Mem.rdN, bind, Option.bind, pure] at h
    cases hrd : m.rd a with
    | none => rw [hrd] at h; cases h
    | some b0 =>
      rw [hrd] at h
      cases hrest : m.rdN (a + 1) n with
      | none => rw [hrest] at h; cases h
      | some bs' =>
        rw [hrest] at h; injection h with heq; rw [← heq]
        cases k with
        | zero => simpa using hrd
        | succ k' =>
          have ih := rdN_getElem? hrest (by omega : k' < n)
          rw [show a + 1 + k' = a + (k' + 1) by omega] at ih
          exact ih

private theorem leVal_head_mod (x a : Nat) (hx : x < 256) : (x + 256 * a) % 256 = x := by omega

private theorem leVal_div_pow_mod (x a k : Nat) (hx : x < 256) :
    (x + 256 * a) / 256 ^ (k + 1) % 256 = a / 256 ^ k % 256 := by
  rw [Nat.pow_succ, Nat.mul_comm (256 ^ k) 256, ← Nat.div_div_eq_div_mul,
    Nat.add_mul_div_left _ _ (by decide), Nat.div_eq_of_lt hx, Nat.zero_add]

private theorem leVal_eq_bytes_at : ∀ (xs : List Nat), (∀ x ∈ xs, x < 256) → ∀ (k : Nat),
    k < xs.length → xs[k]? = some ((leVal xs / 256 ^ k) % 256)
  | [], _, k, hk => by simp at hk
  | x :: rest, hby, k, hk => by
    have hx : x < 256 := hby x (List.mem_cons_self ..)
    have hr : ∀ y ∈ rest, y < 256 := fun y hy => hby y (List.mem_cons_of_mem _ hy)
    simp only [leVal]
    cases k with
    | zero => simp [leVal_head_mod _ _ hx]
    | succ k' =>
      rw [List.getElem?_cons_succ, leVal_div_pow_mod x (leVal rest) k' hx]
      exact leVal_eq_bytes_at rest hr k' (by simp only [List.length_cons] at hk; omega)

private theorem rd64_bytes8_rd {m : Mem} (hm : MemBytesLt256 m) {a v : Nat}
    (h : rd64 m a = some v) {k : Nat} (hk : k < 8) :
    m.rd (a + k) = (bytes8 v)[k]? := by
  simp only [rd64, Option.map_eq_some_iff] at h
  obtain ⟨xs, hxs, rfl⟩ := h
  have ⟨hby, hlen⟩ := rdN_bytes_lt_256 m a 8 xs hm hxs
  rw [rdN_getElem? hxs hk, leVal_eq_bytes_at xs hby k (by omega)]
  simp [bytes8, hk]

private def passSubMem (D b : Nat) (bs : List Nat) : Mem :=
  [⟨D, bytes8 b ++ bytes8 bs.length, false⟩, ⟨b, bs, false⟩]

private theorem mem_rd_passSubMem_extends {D b : Nat} {bs : List Nat} {m : Mem}
    (hm : MemBytesLt256 m) (hbuf : Buf m b bs) (hdesc : Desc m D b bs.length)
    (hD : D + 8 < W) (_hdisj : D + 16 ≤ b ∨ b + bs.length ≤ D) :
    MemExtends (passSubMem D b bs) m := by
  have hD64 : D + 8 < 2 ^ 64 := by unfold W at hD; exact hD
  intro a v ha
  simp only [passSubMem, Mem.rd, Mem.find, Region.hi, List.length_append, bytes8_len,
    bind, Option.bind] at ha
  by_cases hD_le : D ≤ a ∧ a < D + 16
  · rw [show D + (8 + 8) = D + 16 by omega, if_pos hD_le] at ha
    dsimp only at ha
    by_cases ha8 : a - D < 8
    · have hget : (bytes8 b ++ bytes8 bs.length)[a - D]? = (bytes8 b)[a - D]? :=
        List.getElem?_append_left (by simp [bytes8_len]; omega)
      rw [hget] at ha
      have hrd := rd64_bytes8_rd hm hdesc.ptr ha8
      rw [ea_small (by omega)] at hrd
      rw [show D + 0 + (a - D) = a by omega] at hrd
      rw [hrd]; exact ha
    · have ha8' : a - D - 8 < 8 := by omega
      have hget : (bytes8 b ++ bytes8 bs.length)[a - D]? = (bytes8 bs.length)[a - D - 8]? := by
        rw [List.getElem?_append_right (by simp [bytes8_len]; omega), bytes8_len]
      rw [hget] at ha
      have hrd := rd64_bytes8_rd hm hdesc.len ha8'
      rw [ea_small hD64] at hrd
      rw [show D + 8 + (a - D - 8) = a by omega] at hrd
      rw [hrd]; exact ha
  · rw [show D + (8 + 8) = D + 16 by omega, if_neg hD_le] at ha
    by_cases hb1 : b ≤ a ∧ a < b + bs.length
    · simp only [hb1, and_self, ↓reduceIte] at ha
      have hbi : a - b < bs.length := by omega
      have hbuf_i := hbuf (a - b) hbi
      rw [show b + (a - b) = a by omega] at hbuf_i
      rw [List.getElem?_eq_getElem hbi] at ha
      cases ha; exact hbuf_i
    · simp only [hb1, ↓reduceIte] at ha
      cases ha

private theorem buf_passSubMem {D b : Nat} {bs : List Nat} (hdisj : D + 16 ≤ b ∨ b + bs.length ≤ D) :
    Buf (passSubMem D b bs) b bs := by
  intro i hi
  simp only [passSubMem, Mem.rd, Mem.find, Region.hi, List.length_append, bytes8_len,
    bind, Option.bind]
  have hnoD : ¬ (D ≤ b + i ∧ b + i < D + 16) := by omega
  rw [if_neg hnoD]
  have hin : b ≤ b + i ∧ b + i < b + bs.length := by omega
  simp only [hin, and_self, ↓reduceIte]
  rw [show b + i - b = i by omega, List.getElem?_eq_getElem hi]

private theorem rdN_passSubMem (D b : Nat) (bs : List Nat) :
    ∀ (k off : Nat), off + k ≤ 16 →
      (passSubMem D b bs).rdN (D + off) k = some (((bytes8 b ++ bytes8 bs.length).drop off).take k)
  | 0, off, _ => by simp [Mem.rdN]
  | k + 1, off, hk => by
    have hlt : off < (bytes8 b ++ bytes8 bs.length).length := by simp [bytes8_len]; omega
    have hrd : (passSubMem D b bs).rd (D + off) = some ((bytes8 b ++ bytes8 bs.length)[off]) := by
      simp only [passSubMem, Mem.rd, Mem.find, Region.hi, List.length_append, bytes8_len,
        bind, Option.bind]
      rw [if_pos (by omega)]
      dsimp only
      rw [show D + off - D = off by omega, List.getElem?_eq_getElem hlt]
    have ih := rdN_passSubMem D b bs k (off + 1) (by omega)
    rw [show D + (off + 1) = D + off + 1 by omega] at ih
    simp only [Mem.rdN, hrd, ih, bind, Option.bind, pure]
    rw [List.drop_eq_getElem_cons hlt]; rfl

private theorem desc_passSubMem {D b : Nat} {bs : List Nat}
    (hD : D + 8 < W) (hbW : b < W) (hlW : bs.length < W) :
    Desc (passSubMem D b bs) D b bs.length := by
  have hD64 : D + 8 < 2 ^ 64 := by unfold W at hD; exact hD
  refine ⟨?_, ?_⟩
  · rw [ea_small (by omega : D + 0 < 2 ^ 64), Nat.add_zero]
    simp only [rd64]
    have hr := rdN_passSubMem D b bs 8 0 (by decide)
    rw [Nat.add_zero] at hr; rw [hr]
    have htake : ((bytes8 b ++ bytes8 bs.length).drop 0).take 8 = bytes8 b := by
      simp [bytes8_len]
    rw [htake]; exact congrArg some (leVal_bytes8 hbW)
  · rw [ea_small hD64]
    simp only [rd64, rdN_passSubMem D b bs 8 8 (by decide)]
    have htake : ((bytes8 b ++ bytes8 bs.length).drop 8).take 8 = bytes8 bs.length := by
      simp [bytes8_len]; rw [← bytes8_len bs.length, List.take_length]
    rw [htake]; exact congrArg some (leVal_bytes8 hlW)

private theorem modifyTailIdx_go_length {α : Type} (f : α → α) :
    ∀ (k : Nat) (l : List α), (List.modifyTailIdx.go (List.modifyHead f) k l).length = l.length
  | k, [] => by cases k <;> rfl
  | 0, _ :: _ => rfl
  | k + 1, _ :: xs => by simp [List.modifyTailIdx.go, modifyTailIdx_go_length f k xs]

private theorem getElem?_modifyTailIdx_go_ne {α : Type} (f : α → α) :
    ∀ (k i : Nat) (l : List α), i ≠ k →
      (List.modifyTailIdx.go (List.modifyHead f) k l)[i]? = l[i]?
  | k, _, [], _ => by cases k <;> rfl
  | 0, 0, _ :: _, hne => absurd rfl hne
  | 0, i + 1, _ :: xs, _ => rfl
  | k + 1, 0, _ :: _, _ => rfl
  | k + 1, i + 1, _ :: xs, hne =>
    getElem?_modifyTailIdx_go_ne f k i xs (fun h => hne (congrArg Nat.succ h))

private theorem mem_rd_modifyTailIdx_go (j0 : Nat) (f : Nat → Nat) :
    ∀ (i0 : Nat) (m : Mem) (idx a : Nat),
      (∀ r0, m[i0]? = some r0 → a ≠ r0.base + j0) →
      (do let (_, r) ← Mem.find (List.modifyTailIdx.go (List.modifyHead fun r =>
            { r with data := r.data.modify j0 f }) i0 m) a idx; r.data[a - r.base]?) =
      (do let (_, r) ← Mem.find m a idx; r.data[a - r.base]?)
  | i0, [], _, _, _ => by cases i0 <;> rfl
  | 0, r :: rest, idx, a, hne => by
    have ha : a ≠ r.base + j0 := hne r rfl
    simp only [List.modifyTailIdx.go, List.modifyHead, Mem.find, Region.hi,
      List.modify, List.modifyTailIdx, modifyTailIdx_go_length]
    split
    · rename_i hcov
      simp only [bind, Option.bind, hcov, and_self, ↓reduceIte]
      exact getElem?_modifyTailIdx_go_ne f j0 (a - r.base) r.data (by omega)
    · rename_i hcov
      simp only [hcov, ↓reduceIte]
  | i0 + 1, r :: rest, idx, a, hne => by
    simp only [List.modifyTailIdx.go, Mem.find]
    split
    · rfl
    · exact mem_rd_modifyTailIdx_go j0 f i0 rest (idx + 1) a (fun r0 hr0 => hne r0 hr0)

theorem memFlips_one_addr {s1 s2 : St} (hm : s2 ∈ memFlips s1) :
    ∃ a0 : Nat, ∀ a, a ≠ a0 → s2.mem.rd a = s1.mem.rd a := by
  simp only [memFlips, List.mem_flatMap, List.mem_map] at hm
  obtain ⟨i0, _, j0, _, b, _, rfl⟩ := hm
  refine ⟨match s1.mem[i0]? with | some r0 => r0.base + j0 | none => 0, fun a ha => ?_⟩
  simp only [St.flipMem, Mem.rd, List.modify, List.modifyTailIdx]
  exact mem_rd_modifyTailIdx_go j0 (fun v => Nat.xor v (2 ^ b)) i0 s1.mem 0 a (fun r0 hr0 => by
    rw [hr0] at ha; exact ha)

def InPassFoot (bs : List Nat) (D b a : Nat) : Prop :=
  (D ≤ a ∧ a < D + 16) ∨ (b ≤ a ∧ a < b + bs.length)

private theorem memExtends_passSubMem_after_flip {D b : Nat} {bs : List Nat} {m1 m2 : Mem}
    {a0 : Nat} (ha0 : ¬ InPassFoot bs D b a0)
    (hext : MemExtends (passSubMem D b bs) m1)
    (hdiff : ∀ a, a ≠ a0 → m2.rd a = m1.rd a) :
    MemExtends (passSubMem D b bs) m2 := by
  intro a v ha
  have h1 := hext a v ha
  have ha_in : InPassFoot bs D b a := by
    simp only [passSubMem, Mem.rd, Mem.find, Region.hi, List.length_append, bytes8_len,
      bind, Option.bind] at ha
    by_cases h1' : D ≤ a ∧ a < D + 16
    · exact Or.inl h1'
    · by_cases h2' : b ≤ a ∧ a < b + bs.length
      · exact Or.inr h2'
      · exfalso
        rw [show D + (8 + 8) = D + 16 by omega, if_neg h1'] at ha
        simp only [h2', ↓reduceIte] at ha
        cases ha
  have hne : a ≠ a0 := fun h => ha0 (h ▸ ha_in)
  rw [hdiff a hne]; exact h1

private theorem regsLtW_of_mem_flip {s1 s2 : St} (hlt : RegsLtW s1) (hm : s2 ∈ memFlips s1) :
    RegsLtW s2 := by
  simp only [memFlips, List.mem_flatMap, List.mem_map] at hm
  obtain ⟨_, _, _, _, _, _, rfl⟩ := hm
  exact hlt

private theorem memLt_of_mem_flip {s1 s2 : St} (hm1 : MemBytesLt256 s1.mem) (hm : s2 ∈ memFlips s1) :
    MemBytesLt256 s2.mem := by
  simp only [memFlips, List.mem_flatMap, List.mem_map, List.mem_range] at hm
  obtain ⟨i, _, j, _, b, hb, rfl⟩ := hm
  exact memBytesLt256_flipMem hm1 i j b hb

private theorem regsLtW_of_flag_flip {s1 s2 : St} (hlt : RegsLtW s1) (hfl : s2 ∈ flagFlips s1) :
    RegsLtW s2 := by
  simp only [flagFlips, List.mem_append, List.mem_map, Option.mem_toList] at hfl
  rcases hfl with (((⟨_, _, rfl⟩ | ⟨_, _, rfl⟩) | ⟨_, _, rfl⟩) | ⟨_, _, rfl⟩) <;> exact hlt

private theorem mem_of_flag_flip {s1 s2 : St} (hfl : s2 ∈ flagFlips s1) : s2.mem = s1.mem := by
  simp only [flagFlips, List.mem_append, List.mem_map, Option.mem_toList] at hfl
  rcases hfl with (((⟨_, _, rfl⟩ | ⟨_, _, rfl⟩) | ⟨_, _, rfl⟩) | ⟨_, _, rfl⟩) <;> rfl

private theorem rip_of_flag_flip {s1 s2 : St} (hfl : s2 ∈ flagFlips s1) : s2.rip = s1.rip := by
  simp only [flagFlips, List.mem_append, List.mem_map, Option.mem_toList] at hfl
  rcases hfl with (((⟨_, _, rfl⟩ | ⟨_, _, rfl⟩) | ⟨_, _, rfl⟩) | ⟨_, _, rfl⟩) <;> rfl

private theorem reg_of_flag_flip {s1 s2 : St} (hfl : s2 ∈ flagFlips s1) (j : Nat) :
    s2.reg j = s1.reg j := by
  simp only [flagFlips, List.mem_append, List.mem_map, Option.mem_toList] at hfl
  rcases hfl with (((⟨_, _, rfl⟩ | ⟨_, _, rfl⟩) | ⟨_, _, rfl⟩) | ⟨_, _, rfl⟩) <;> rfl

private theorem regsLen_of_flag_flip {s1 s2 : St} (hfl : s2 ∈ flagFlips s1) :
    s2.regs.length = s1.regs.length := by
  simp only [flagFlips, List.mem_append, List.mem_map, Option.mem_toList] at hfl
  rcases hfl with (((⟨_, _, rfl⟩ | ⟨_, _, rfl⟩) | ⟨_, _, rfl⟩) | ⟨_, _, rfl⟩) <;> rfl

private theorem mem_of_reg_flip {s1 s2 : St} (hreg : s2 ∈ regFlips s1) : s2.mem = s1.mem := by
  simp only [regFlips, List.mem_map] at hreg; obtain ⟨_, _, rfl⟩ := hreg; rfl

private theorem rip_of_reg_flip {s1 s2 : St} (hreg : s2 ∈ regFlips s1) : s2.rip = s1.rip := by
  simp only [regFlips, List.mem_map] at hreg; obtain ⟨_, _, rfl⟩ := hreg; rfl

private theorem fl_of_reg_flip {s1 s2 : St} (hreg : s2 ∈ regFlips s1) : s2.fl = s1.fl := by
  simp only [regFlips, List.mem_map] at hreg; obtain ⟨_, _, rfl⟩ := hreg; rfl

private theorem regsLen_of_reg_flip {s1 s2 : St} (hreg : s2 ∈ regFlips s1) :
    s2.regs.length = s1.regs.length := by
  simp only [regFlips, List.mem_map] at hreg; obtain ⟨_, _, rfl⟩ := hreg; simp [St.flipReg, St.set]

private theorem mem_of_mem_flip_rip {s1 s2 : St} (hm : s2 ∈ memFlips s1) : s2.rip = s1.rip := by
  simp only [memFlips, List.mem_flatMap, List.mem_map] at hm
  obtain ⟨_, _, _, _, _, _, rfl⟩ := hm; rfl

private theorem mem_of_mem_flip_fl {s1 s2 : St} (hm : s2 ∈ memFlips s1) : s2.fl = s1.fl := by
  simp only [memFlips, List.mem_flatMap, List.mem_map] at hm
  obtain ⟨_, _, _, _, _, _, rfl⟩ := hm; rfl

private theorem mem_of_mem_flip_reg {s1 s2 : St} (hm : s2 ∈ memFlips s1) (j : Nat) :
    s2.reg j = s1.reg j := by
  simp only [memFlips, List.mem_flatMap, List.mem_map] at hm
  obtain ⟨_, _, _, _, _, _, rfl⟩ := hm; rfl

private theorem mem_of_mem_flip_regsLen {s1 s2 : St} (hm : s2 ∈ memFlips s1) :
    s2.regs.length = s1.regs.length := by
  simp only [memFlips, List.mem_flatMap, List.mem_map] at hm
  obtain ⟨_, _, _, _, _, _, rfl⟩ := hm; rfl

private theorem regsLtW_of_reg_flip {s1 s2 : St} (hlt : RegsLtW s1) (hreg : s2 ∈ regFlips s1) :
    RegsLtW s2 := by
  simp only [regFlips, List.mem_map, List.mem_range] at hreg
  obtain ⟨j, hj, rfl⟩ := hreg
  exact regsLtW_flipReg hlt (j / 64) (j % 64) (Nat.mod_lt _ (by decide))

private theorem binPreFault_of_vote {v : Nat} {s : St}
    (hv : SporosCheckFlat.SafeInvalid v) (hlen : s.regs.length = 16) (hst : BinStackSafe s)
    (hrip : s.rip = L_vote) (h0 : s.reg 0 = some v) (h3 : s.reg 3 = some v) (h1 : s.reg 1 = some v) :
    BinPreFault v s := by
  refine ⟨⟨hv, hlen, BinStackSafe_to_TailStackSafe hst, ?_, ?_, ?_⟩, hst⟩
  · rw [hrip]; exact Nat.le_refl _
  · rw [hrip]; unfold L_vote; decide
  · change SporosCheckFlat.PreFaultOff v s (s.rip - L_vote)
    rw [hrip, Nat.sub_self L_vote]
    exact ⟨h0, h1, h3, (fun h => by cases h), (fun h => by cases h), (fun h => by cases h),
      (fun h => by simp at h)⟩

theorem binPreFault_pRunWithFaultAt_pos_ne_exit0 {v : Nat} {s_vote : St}
    (hpf : BinPreFault v s_vote) (r : Rates) (hr : r.ok) (hsing : SporosCheckFlat.singleMechanism r)
    (k m : Nat) {y : (Out × Bool) × Rat}
    (hy : y ∈ pRunWithFaultAt r hr k m (.ok s_vote)) (hpos : 0 < y.2) :
    y.1.1 ≠ .exit 0 := by
  have h0 : PDist.Rate (0 : Rat) := ⟨Rat.le_refl, by decide⟩
  have hmid_ok : ∀ o1 : Out, (PDist.bindF
      (match o1 with
       | .ok s1 => pstep spr1Alts (r.state hr) r.fetch im s1
       | o' => PDist.pureF o')
      (prun spr1Alts .none 0 im m)).ok := fun o1 =>
    PDist.bindF_ok (by cases o1; exact pstep_ok _ _ hr.1 _ _; exact PDist.pureF_ok _; exact PDist.pureF_ok _)
      (prun_ok _ .none h0 _ m)
  rcases mem_bindF_pos (prun_ok spr1Alts .none h0 im k _).1
    (fun o1 z hz => (hmid_ok o1).1 z hz) hy hpos with
    ⟨x1, hx1, z2, hz2, hwx1, hwz2, heq1⟩
  have hx1_eq := prun_none_zero_pos k (.ok s_vote) hx1 hwx1
  have hpf_k := BinPreFault_run k hpf
  rcases mem_bindF_pos (by cases x1.1.1; exact (pstep_ok _ _ hr.1 _ _).1; exact (PDist.pureF_ok _).1; exact (PDist.pureF_ok _).1)
    (fun o2 z hz => (prun_ok spr1Alts .none h0 im m o2).1 z hz) hz2 hwz2 with
    ⟨x2, hx2, z3, hz3, hwx2, hwz3, heq3⟩
  have hz3_eq := prun_none_zero_pos m x2.1.1 hz3 hwz3
  have helem_safe : BinOutSafe x2.1.1 := by
    rw [hx1_eq] at hx2
    cases hrun_k : run im k (.ok s_vote) with
    | stuck => rw [hrun_k, mem_pureF_iff] at hx2; rw [congrArg (·.1.1) hx2]; exact trivial
    | exit c => rw [hrun_k] at hpf_k hx2; rw [mem_pureF_iff] at hx2; rw [congrArg (·.1.1) hx2]; exact hpf_k
    | ok s1 =>
      rw [hrun_k] at hpf_k hx2
      exact BinPreFault_elemFaultStep_safe hpf_k (pstep_singleMechanism_elem hr hsing hx2 hwx2)
  have hneq : y.1.1 = run im m x2.1.1 :=
    (congrArg (·.1.1) heq1).trans ((congrArg (·.1.1) heq3).trans hz3_eq)
  rw [hneq]
  cases hx2o : x2.1.1 with
  | stuck => rw [hx2o] at helem_safe; rw [run_stuck]; intro h; cases h
  | exit c => rw [hx2o] at helem_safe; rw [run_exit]; intro h; cases h; exact helem_safe rfl
  | ok s' => rw [hx2o] at helem_safe; exact BinTailSafe_run m helem_safe

theorem binPreFault_pRunWithFaultAt_exit0Prob_zero {v : Nat} {s_vote : St}
    (hpf : BinPreFault v s_vote) (r : Rates) (hr : r.ok) (hsing : SporosCheckFlat.singleMechanism r)
    (k m : Nat) :
    SporosCheckFlat.exit0Prob (pRunWithFaultAt r hr k m (.ok s_vote)) = 0 :=
  SporosCheckFlat.exit0Prob_eq_zero_of_forall (pRunWithFaultAt_ok r hr k m _).1
    (fun _ hy hpos => binPreFault_pRunWithFaultAt_pos_ne_exit0 hpf r hr hsing k m hy hpos)

private theorem pass_prefix_invs {lo hi : Nat} {forb : List Nat} {tbl : List (Nat × Ins × Nat)}
    (hsafe : ∀ e ∈ tbl, passEntryOK lo hi forb (tbl.map (·.1)) e = true)
    (htbl : ∀ e ∈ tbl, im e.1 = some e.2) :
    ∀ (k : Nat) {s sk : St},
      RegDefExcept11 s → RegsLtW s → MemBytesLt256 s.mem →
      (∀ j, j < k → ∃ sj, run im j (.ok s) = .ok sj ∧ sj.rip ∈ tbl.map (·.1)) →
      run im k (.ok s) = .ok sk →
      RegDefExcept11 sk ∧ RegsLtW sk ∧ sk.mem = s.mem
  | 0, s, sk, hd, hlt, _, _, hrun => by
    simp [run] at hrun; subst hrun; exact ⟨hd, hlt, rfl⟩
  | k + 1, s, sk, hd, hlt, hm, hpre, hrun => by
    obtain ⟨s0, hr0, hin0⟩ := hpre 0 (Nat.zero_lt_succ k)
    simp [run] at hr0; subst hr0
    simp only [run] at hrun
    cases hst : step im s with
    | stuck => rw [hst, run_stuck] at hrun; cases hrun
    | exit c => exact False.elim (step_ne_exit_of_ne_syscall hsafe htbl hin0 hst)
    | ok s1 =>
      rw [hst] at hrun
      have hi1 := pass_step_invs hsafe htbl hin0 hst
      have ih := pass_prefix_invs hsafe htbl k (hi1.2.2.2.2.1 hd) (hi1.2.2.2.2.2 hlt hm)
        (by rw [hi1.1]; exact hm)
        (fun j hj => by
          obtain ⟨sj, hrj, hinj⟩ := hpre (j + 1) (Nat.succ_lt_succ hj)
          rw [run, hst] at hrj
          exact ⟨sj, hrj, hinj⟩)
        hrun
      exact ⟨ih.1, ih.2.1, ih.2.2.trans hi1.1⟩

private theorem pass_clean_run {lo hi : Nat} {forb : List Nat} {tbl : List (Nat × Ins × Nat)}
    (hsafe : ∀ e ∈ tbl, passEntryOK lo hi forb (tbl.map (·.1)) e = true)
    (htbl : ∀ e ∈ tbl, im e.1 = some e.2)
    (hhi_not : hi ∉ tbl.map (·.1))
    (hnoRet : ∀ {s : St} {n : Nat} {s' : St}, s.rip = hi → 0 < n → run im n (.ok s) = .ok s' → s'.rip ≠ hi)
    {outReg D b v : Nat} (hout_not : outReg ∉ forb) {bs : List Nat} {s : St}
    (hrip : s.rip = lo) (hlo_mem : lo ∈ tbl.map (·.1))
    (hd : RegDefExcept11 s) (hlt : RegsLtW s) (hm : MemBytesLt256 s.mem)
    (hext : MemExtends (passSubMem D b bs) s.mem)
    (hspec : Reaches { s with mem := passSubMem D b bs } (fun s' => s'.rip = hi ∧ s'.reg outReg = some v)) :
    ∃ (N : Nat) (sA_end s_end : St),
      0 < N ∧
      run im N (.ok { s with mem := passSubMem D b bs }) = .ok sA_end ∧
      run im N (.ok s) = .ok s_end ∧
      sA_end.rip = hi ∧ sA_end.reg outReg = some v ∧
      s_end.rip = hi ∧ s_end.reg outReg = some v ∧
      s_end.mem = s.mem ∧ s_end.regs.length = s.regs.length ∧
      (∀ r ∈ forb, s_end.reg r = s.reg r) ∧
      RegDefExcept11 s_end ∧ RegsLtW s_end := by
  obtain ⟨N, sA_end, hrunA, hArip, hAout⟩ := hspec
  have hpos : 0 < N := by
    cases N with
    | zero => simp [run] at hrunA; subst hrunA; rw [hrip] at hArip; exact False.elim (hhi_not (hArip ▸ hlo_mem))
    | succ _ => exact Nat.zero_lt_succ _
  have hsim := exec_pass_run_sim_all hsafe htbl hhi_not hnoRet N
    (sA := { s with mem := passSubMem D b bs }) (sB := s)
    (by exact hrip ▸ hlo_mem) rfl rfl hext rfl (fun _ _ => rfl) hrunA hArip
  obtain ⟨s_end, hrunB, hripB, _, _, hmB, _, hregB, hforbB⟩ := hsim.1
  have hinvs := pass_prefix_invs hsafe htbl N hd hlt hm
    (fun j hj => by
      obtain ⟨sAj, sBj, _, hrBj, hinAj, hripj, _⟩ := hsim.2 j hj
      exact ⟨sBj, hrBj, hripj ▸ hinAj⟩)
    hrunB
  have hout_eq : s_end.reg outReg = some v := by rw [← hregB outReg hout_not]; exact hAout
  exact ⟨N, sA_end, s_end, hpos, hrunA, hrunB, hArip, hAout, hripB ▸ hArip, hout_eq,
    hmB, hinvs.1.1.trans hd.1.symm, hforbB, hinvs.1, hinvs.2.1⟩

private def PassPostOutcome (v outReg : Nat) (forb : List Nat) (D b : Nat) (bs : List Nat)
    (s s_end : St) : Prop :=
  RegDefExcept11 s_end ∧ RegsLtW s_end ∧ MemBytesLt256 s_end.mem ∧
  ((s_end.reg outReg = some v ∧
    ((∃ r ∈ forb, s_end.mem = s.mem ∧ ∀ r' ∈ forb, r' ≠ r → s_end.reg r' = s.reg r') ∨
     ((∀ r' ∈ forb, s_end.reg r' = s.reg r') ∧
       (s_end.mem = s.mem ∨
        ∃ a0, ¬ InPassFoot bs D b a0 ∧ ∀ a, a ≠ a0 → s_end.mem.rd a = s.mem.rd a)))) ∨
   ((∀ r' ∈ forb, s_end.reg r' = s.reg r') ∧
    (s_end.mem = s.mem ∨
     ∃ a0, InPassFoot bs D b a0 ∧ ∀ a, a ≠ a0 → s_end.mem.rd a = s.mem.rd a)))

private theorem pass_oneFault_summary {lo hi : Nat} {forb : List Nat} {tbl : List (Nat × Ins × Nat)}
    (hsafe : ∀ e ∈ tbl, passEntryOK lo hi forb (tbl.map (·.1)) e = true)
    (htbl : ∀ e ∈ tbl, im e.1 = some e.2)
    (hhi_not : hi ∉ tbl.map (·.1))
    (hnoRet : ∀ {s : St} {n : Nat} {s' : St}, s.rip = hi → 0 < n → run im n (.ok s) = .ok s' → s'.rip ≠ hi)
    {outReg D b v : Nat} (hout_not : outReg ∉ forb) {bs : List Nat} {s : St}
    (hrip : s.rip = lo) (hlo_mem : lo ∈ tbl.map (·.1))
    (hd : RegDefExcept11 s) (hlt : RegsLtW s) (hm : MemBytesLt256 s.mem)
    (hext : MemExtends (passSubMem D b bs) s.mem)
    {N : Nat} {sA_end : St}
    (hrunA : run im N (.ok { s with mem := passSubMem D b bs }) = .ok sA_end)
    (hArip : sA_end.rip = hi) (hAout : sA_end.reg outReg = some v)
    {k : Nat} (hk : k < N) {s_k s_k1 s2 : St}
    (hrunk : run im k (.ok s) = .ok s_k) (hstk : step im s_k = .ok s_k1)
    (hs2 : s2 ∈ regFlips s_k1 ∨ s2 ∈ memFlips s_k1 ∨ s2 ∈ flagFlips s_k1) (m : Nat) :
    run im m (.ok s2) ≠ .exit 0 ∨
    ∃ t s_end, t ≤ m ∧ run im t (.ok s2) = .ok s_end ∧ s_end.rip = hi ∧
      PassPostOutcome v outReg forb D b bs s s_end := by
  have hsim := exec_pass_run_sim_all hsafe htbl hhi_not hnoRet N
    (sA := { s with mem := passSubMem D b bs }) (sB := s)
    (by exact hrip ▸ hlo_mem) rfl rfl hext rfl (fun _ _ => rfl) hrunA hArip
  obtain ⟨sAk, sBk, hrunAk, hrunBk, hinAk, hripk, hflk, hmAk, hmBk, hlenk, hregk, hforbk⟩ := hsim.2 k hk
  have hsk_eq : s_k = sBk := by
    have h := hrunk.symm.trans hrunBk; injection h
  subst sBk
  have hinvs_k := pass_prefix_invs hsafe htbl k hd hlt hm
    (fun j hj => by
      obtain ⟨sAj, sBj, _, hrBj, hinAj, hripj, _⟩ := hsim.2 j (Nat.lt_trans hj hk)
      exact ⟨sBj, hrBj, hripj ▸ hinAj⟩)
    hrunk
  have hinBk : s_k.rip ∈ tbl.map (·.1) := by rw [← hripk]; exact hinAk
  have hinvs_k1 := pass_step_invs hsafe htbl hinBk hstk
  have hd_k1 : RegDefExcept11 s_k1 := hinvs_k1.2.2.2.2.1 hinvs_k.1
  have hlt_k1 : RegsLtW s_k1 := hinvs_k1.2.2.2.2.2 hinvs_k.2.1 (by rw [hinvs_k.2.2]; exact hm)
  have hm_k1 : s_k1.mem = s.mem := hinvs_k1.1.trans hinvs_k.2.2
  have hforb_k1 : ∀ r ∈ forb, s_k1.reg r = s.reg r :=
    fun r hr => (hinvs_k1.2.2.1 r hr).trans (hforbk r hr)
  have hAk_mem : sAk.mem = passSubMem D b bs := hmAk
  have hs2_invs : RegDefExcept11 s2 ∧ RegsLtW s2 ∧ MemBytesLt256 s2.mem := by
    rcases hs2 with hreg | hmem | hfl
    · refine ⟨?_, regsLtW_of_reg_flip hlt_k1 hreg, ?_⟩
      · simp only [regFlips, List.mem_map] at hreg; obtain ⟨idx, _, rfl⟩ := hreg
        exact regDef_flipReg hd_k1 (idx / 64) (idx % 64)
      · rw [mem_of_reg_flip hreg, hm_k1]; exact hm
    · refine ⟨?_, regsLtW_of_mem_flip hlt_k1 hmem, ?_⟩
      · simp only [memFlips, List.mem_flatMap, List.mem_map] at hmem
        obtain ⟨i, _, j, _, b0, _, rfl⟩ := hmem
        exact regDef_flipMem hd_k1 i j b0
      · exact memLt_of_mem_flip (by rw [hm_k1]; exact hm) hmem
    · refine ⟨regDef_flagFlips hd_k1 hfl, regsLtW_of_flag_flip hlt_k1 hfl, ?_⟩
      · rw [mem_of_flag_flip hfl, hm_k1]; exact hm
  by_cases hlast : k + 1 = N
  · have hrun_k1_A : run im (k + 1) (.ok { s with mem := passSubMem D b bs }) = .ok sA_end := by
      rw [hlast]; exact hrunA
    have hstepA_end : step im sAk = .ok sA_end := by
      have ha := run_add k 1 (.ok { s with mem := passSubMem D b bs })
      rw [hrunAk] at ha
      change run im (k + 1) (.ok { s with mem := passSubMem D b bs }) = step im sAk at ha
      rw [ha] at hrun_k1_A; exact hrun_k1_A
    obtain ⟨sB_k1, hstB_k1, hrip_k1, _, _, _, _, hreg_k1, _⟩ :=
      exec_pass_sim_step hsafe htbl hinAk hripk hflk (by rw [hmAk, hmBk]; exact hext) hlenk hregk hstepA_end
    have heq_k1 : s_k1 = sB_k1 := by have h := hstk.symm.trans hstB_k1; injection h
    subst sB_k1
    have hrip_s1 : s_k1.rip = hi := by rw [← hrip_k1]; exact hArip
    have hout_s1 : s_k1.reg outReg = some v := by rw [← hreg_k1 outReg hout_not]; exact hAout
    right; refine ⟨0, s2, Nat.zero_le m, rfl, ?_, hs2_invs.1, hs2_invs.2.1, hs2_invs.2.2, ?_⟩
    · rcases hs2 with hreg | hmem | hfl
      · rw [rip_of_reg_flip hreg, hrip_s1]
      · rw [mem_of_mem_flip_rip hmem, hrip_s1]
      · rw [rip_of_flag_flip hfl, hrip_s1]
    · rcases hs2 with hreg | hmem | hfl
      · have hreg' := hreg
        simp only [regFlips, List.mem_map] at hreg'
        obtain ⟨idx, _, rfl⟩ := hreg'
        by_cases hrf : idx / 64 ∈ forb
        · left
          refine ⟨?_, Or.inl ⟨idx / 64, hrf, by rw [mem_of_reg_flip hreg, hm_k1], fun r' hr' hne => ?_⟩⟩
          · rw [St.flipReg, reg_set_ne (fun h : idx / 64 = outReg => hout_not (h ▸ hrf))]; exact hout_s1
          · rw [St.flipReg, reg_set_ne (Ne.symm hne)]; exact hforb_k1 r' hr'
        · right
          refine ⟨fun r' hr' => ?_, Or.inl (by rw [mem_of_reg_flip hreg, hm_k1])⟩
          rw [St.flipReg, reg_set_ne (fun h : idx / 64 = r' => hrf (h ▸ hr'))]; exact hforb_k1 r' hr'
      · obtain ⟨a0, ha0⟩ := memFlips_one_addr hmem
        have ha0' : ∀ a, a ≠ a0 → s2.mem.rd a = s.mem.rd a := by
          intro a ha; rw [ha0 a ha, hm_k1]
        by_cases hfoot : InPassFoot bs D b a0
        · right
          exact ⟨fun r' hr' => (mem_of_mem_flip_reg hmem r').trans (hforb_k1 r' hr'),
            Or.inr ⟨a0, hfoot, ha0'⟩⟩
        · left
          refine ⟨by rw [mem_of_mem_flip_reg hmem outReg]; exact hout_s1,
            Or.inr ⟨fun r' hr' => (mem_of_mem_flip_reg hmem r').trans (hforb_k1 r' hr'),
              Or.inr ⟨a0, hfoot, ha0'⟩⟩⟩
      · right
        exact ⟨fun r' hr' => (reg_of_flag_flip hfl r').trans (hforb_k1 r' hr'),
          Or.inl ((mem_of_flag_flip hfl).trans hm_k1)⟩
  · have hkn : k + 1 < N := Nat.lt_of_le_of_ne hk hlast
    obtain ⟨sAk1, sBk1, hrunAk1, hrunBk1, hinAk1, hripk1, hflk1, hmAk1, hmBk1, hlenk1, hregk1, _⟩ :=
      hsim.2 (k + 1) hkn
    have hstepA1 : step im sAk = .ok sAk1 := by
      have ha := run_add k 1 (.ok { s with mem := passSubMem D b bs })
      rw [hrunAk] at ha
      change run im (k + 1) (.ok { s with mem := passSubMem D b bs }) = step im sAk at ha
      rw [ha] at hrunAk1; exact hrunAk1
    obtain ⟨sB_k1, hstB_k1, hrip_k1, hfl_k1, _, hmB_k1, hlen_k1, hreg_k1, _⟩ :=
      exec_pass_sim_step hsafe htbl hinAk hripk hflk (by rw [hmAk, hmBk]; exact hext) hlenk hregk hstepA1
    have heq_k1 : s_k1 = sB_k1 := by have h := hstk.symm.trans hstB_k1; injection h
    subst sB_k1
    have hrunA_rem : run im (N - (k + 1)) (.ok sAk1) = .ok sA_end := by
      have ha := run_add (k + 1) (N - (k + 1)) (.ok { s with mem := passSubMem D b bs })
      rw [show k + 1 + (N - (k + 1)) = N by omega, hrunAk1] at ha
      rw [← hrunA]; exact ha.symm
    have run_fallback : ∀ (hforb2 : ∀ r' ∈ forb, s2.reg r' = s.reg r')
        (hrip2 : s2.rip ∈ tbl.map (·.1))
        (hmem2 : s2.mem = s.mem ∨ ∃ a0, InPassFoot bs D b a0 ∧ ∀ a, a ≠ a0 → s2.mem.rd a = s.mem.rd a),
        run im m (.ok s2) ≠ .exit 0 ∨
        ∃ t s_end, t ≤ m ∧ run im t (.ok s2) = .ok s_end ∧ s_end.rip = hi ∧
          PassPostOutcome v outReg forb D b bs s s_end := by
      intro hforb2 hrip2 hmem2
      rcases pass_run_trichotomy hsafe htbl m s2 hrip2 with hstk2 | ⟨s', hrun', hin', _⟩ | ⟨t, s_end, _, htm, hrunt, hript, hm_t, _, hforb_t, hdt, hltt⟩
      · left; rw [hstk2]; intro h; cases h
      · left; rw [hrun']; intro h; cases h
      · right; refine ⟨t, s_end, htm, hrunt, hript, hdt hs2_invs.1, hltt hs2_invs.2.1 hs2_invs.2.2, ?_, ?_⟩
        · rw [hm_t]; exact hs2_invs.2.2
        · right; refine ⟨fun r' hr' => (hforb_t r' hr').trans (hforb2 r' hr'), ?_⟩
          rw [hm_t]; exact hmem2
    rcases hs2 with hreg | hmem | hfl
    · have hreg' := hreg
      simp only [regFlips, List.mem_map] at hreg'
      obtain ⟨idx, _, hidx_eq⟩ := hreg'
      by_cases hrf : idx / 64 ∈ forb
      · have hrip2 : sAk1.rip = s2.rip := by rw [rip_of_reg_flip hreg, ← hrip_k1]
        have hfl2 : sAk1.fl = s2.fl := by rw [fl_of_reg_flip hreg, ← hfl_k1]
        have hext2 : MemExtends sAk1.mem s2.mem := by
          rw [hmAk1, mem_of_reg_flip hreg, hm_k1]; exact hext
        have hlen2 : sAk1.regs.length = s2.regs.length := by
          rw [regsLen_of_reg_flip hreg, ← hlen_k1]
        have hreg2 : ∀ j, j ∉ forb → sAk1.reg j = s2.reg j := by
          intro j hj
          have hne : idx / 64 ≠ j := fun h => hj (h ▸ hrf)
          rw [← hidx_eq, St.flipReg, reg_set_ne hne]
          exact hreg_k1 j hj
        have hsim2 := exec_pass_run_sim_all hsafe htbl hhi_not hnoRet (N - (k + 1))
          hinAk1 hrip2 hfl2 hext2 hlen2 hreg2 hrunA_rem hArip
        by_cases hmrem : N - (k + 1) ≤ m
        · obtain ⟨s_end, hrun_end, hrip_end, _, _, hm_end, _, hreg_end, hforb_end⟩ := hsim2.1
          have hinvs_end := pass_prefix_invs hsafe htbl (N - (k + 1)) hs2_invs.1 hs2_invs.2.1 hs2_invs.2.2
            (fun j hj => by obtain ⟨sAj, sBj, _, hrBj, hinAj, hripj, _⟩ := hsim2.2 j hj; exact ⟨sBj, hrBj, hripj ▸ hinAj⟩)
            hrun_end
          right; refine ⟨N - (k + 1), s_end, hmrem, hrun_end, hrip_end ▸ hArip,
            hinvs_end.1, hinvs_end.2.1, by rw [hm_end]; exact hs2_invs.2.2, Or.inl ⟨?_, Or.inl ⟨idx / 64, hrf, ?_, ?_⟩⟩⟩
          · rw [← hreg_end outReg hout_not]; exact hAout
          · rw [hm_end, mem_of_reg_flip hreg, hm_k1]
          · intro r' hr' hne
            rw [hforb_end r' hr', ← hidx_eq, St.flipReg, reg_set_ne (Ne.symm hne)]
            exact hforb_k1 r' hr'
        · obtain ⟨_, sBm, _, hrun_m, _, _⟩ := hsim2.2 m (Nat.lt_of_not_le hmrem)
          left; rw [hrun_m]; intro h; cases h
      · refine run_fallback (fun r' hr' => ?_) (by rw [rip_of_reg_flip hreg, ← hrip_k1]; exact hinAk1)
          (Or.inl ((mem_of_reg_flip hreg).trans hm_k1))
        rw [← hidx_eq, St.flipReg, reg_set_ne (fun h : idx / 64 = r' => hrf (h ▸ hr'))]
        exact hforb_k1 r' hr'
    · obtain ⟨a0, ha0⟩ := memFlips_one_addr hmem
      have ha0' : ∀ a, a ≠ a0 → s2.mem.rd a = s.mem.rd a := by intro a ha; rw [ha0 a ha, hm_k1]
      by_cases hfoot : InPassFoot bs D b a0
      · exact run_fallback (fun r' hr' => (mem_of_mem_flip_reg hmem r').trans (hforb_k1 r' hr'))
          (by rw [mem_of_mem_flip_rip hmem, ← hrip_k1]; exact hinAk1) (Or.inr ⟨a0, hfoot, ha0'⟩)
      · have hrip2 : sAk1.rip = s2.rip := by rw [mem_of_mem_flip_rip hmem, ← hrip_k1]
        have hfl2 : sAk1.fl = s2.fl := by rw [mem_of_mem_flip_fl hmem, ← hfl_k1]
        have hext2 : MemExtends sAk1.mem s2.mem := by
          rw [hmAk1]
          exact memExtends_passSubMem_after_flip hfoot hext ha0'
        have hlen2 : sAk1.regs.length = s2.regs.length := by
          rw [mem_of_mem_flip_regsLen hmem, ← hlen_k1]
        have hreg2 : ∀ j, j ∉ forb → sAk1.reg j = s2.reg j := by
          intro j hj
          rw [mem_of_mem_flip_reg hmem j]
          exact hreg_k1 j hj
        have hsim2 := exec_pass_run_sim_all hsafe htbl hhi_not hnoRet (N - (k + 1))
          hinAk1 hrip2 hfl2 hext2 hlen2 hreg2 hrunA_rem hArip
        by_cases hmrem : N - (k + 1) ≤ m
        · obtain ⟨s_end, hrun_end, hrip_end, _, _, hm_end, _, hreg_end, hforb_end⟩ := hsim2.1
          have hinvs_end := pass_prefix_invs hsafe htbl (N - (k + 1)) hs2_invs.1 hs2_invs.2.1 hs2_invs.2.2
            (fun j hj => by obtain ⟨sAj, sBj, _, hrBj, hinAj, hripj, _⟩ := hsim2.2 j hj; exact ⟨sBj, hrBj, hripj ▸ hinAj⟩)
            hrun_end
          right; refine ⟨N - (k + 1), s_end, hmrem, hrun_end, hrip_end ▸ hArip,
            hinvs_end.1, hinvs_end.2.1, by rw [hm_end]; exact hs2_invs.2.2, Or.inl ⟨?_, Or.inr ⟨?_, ?_⟩⟩⟩
          · rw [← hreg_end outReg hout_not]; exact hAout
          · intro r' hr'; rw [hforb_end r' hr', mem_of_mem_flip_reg hmem r']; exact hforb_k1 r' hr'
          · right; refine ⟨a0, hfoot, fun a ha => by rw [hm_end]; exact ha0' a ha⟩
        · obtain ⟨_, sBm, _, hrun_m, _, _⟩ := hsim2.2 m (Nat.lt_of_not_le hmrem)
          left; rw [hrun_m]; intro h; cases h
    · exact run_fallback (fun r' hr' => (reg_of_flag_flip hfl r').trans (hforb_k1 r' hr'))
        (by rw [rip_of_flag_flip hfl, ← hrip_k1]; exact hinAk1)
        (Or.inl ((mem_of_flag_flip hfl).trans hm_k1))

structure TripledStateAt (bs : List Nat) (b1 b2 b3 : Nat) (s : St) : Prop where
  bytes : ∀ x ∈ bs, x < 256
  rip : s.rip = L_p1
  regDef : RegDefExcept11 s
  regsLtW : RegsLtW s
  memLt : MemBytesLt256 s.mem
  r0 : s.reg 0 = some REJECT
  r3 : s.reg 3 = some REJECT
  r1 : s.reg 1 = some REJECT
  buf1 : Buf s.mem b1 bs
  desc1 : Desc s.mem 0x600000 b1 bs.length
  small1 : b1 + bs.length < HMAX
  disjDesc1 : 0x600010 ≤ b1 ∨ b1 + bs.length ≤ 0x600000
  buf2 : Buf s.mem b2 bs
  desc2 : Desc s.mem 0x601000 b2 bs.length
  small2 : b2 + bs.length < HMAX
  disjDesc2 : 0x601010 ≤ b2 ∨ b2 + bs.length ≤ 0x601000
  buf3 : Buf s.mem b3 bs
  desc3 : Desc s.mem 0x602000 b3 bs.length
  small3 : b3 + bs.length < HMAX
  disjDesc3 : 0x602010 ≤ b3 ∨ b3 + bs.length ≤ 0x602000
  disj12 : ∀ a, InPassFoot bs 0x600000 b1 a → ¬ InPassFoot bs 0x601000 b2 a
  disj13 : ∀ a, InPassFoot bs 0x600000 b1 a → ¬ InPassFoot bs 0x602000 b3 a
  disj23 : ∀ a, InPassFoot bs 0x601000 b2 a → ¬ InPassFoot bs 0x602000 b3 a
  stkSafe : BinStackSafe s

def TripledState (bs : List Nat) (s : St) : Prop :=
  ∃ b1 b2 b3, TripledStateAt bs b1 b2 b3 s

theorem TripledState.bytes {bs : List Nat} {s : St} (h : TripledState bs s) :
    ∀ x ∈ bs, x < 256 :=
  let ⟨_, _, _, ht⟩ := h; ht.bytes

def mkTripledMem (bs : List Nat) : Mem :=
  [ ⟨0x600000, bytes8 0x100000000000 ++ bytes8 bs.length, true⟩
  , ⟨0x601000, bytes8 0x200000000000 ++ bytes8 bs.length, true⟩
  , ⟨0x602000, bytes8 0x300000000000 ++ bytes8 bs.length, true⟩
  , ⟨0x100000000000, bs, true⟩
  , ⟨0x200000000000, bs, true⟩
  , ⟨0x300000000000, bs, true⟩
  , ⟨0x7fffffffd000, List.replicate 0x2000 0, true⟩ ]

private def mkTripledRegs : List (Option Nat) :=
  ((((List.replicate 16 (some 0)).set 4 (some 0x7fffffffef00)).set 0 (some REJECT)).set 3 (some REJECT)).set 1 (some REJECT)

private theorem mkTripledRegs_def_ok :
    ((List.range 16).all fun j => (mkTripledRegs[j]?).getD none |>.isSome) = true := by
  decide

private theorem mkTripledRegs_ltW_ok :
    ((List.range 16).all fun j => match (mkTripledRegs[j]?).getD none with
      | some w => decide (w < W)
      | none => true) = true := by
  decide

def mkTripledState (bs : List Nat) : St :=
  { regs := ((((List.replicate 16 (some 0)).set 4 (some 0x7fffffffef00)).set 0 (some REJECT)).set 3 (some REJECT)).set 1 (some REJECT)
  , rip := L_p1
  , fl := Flags.undef
  , mem := mkTripledMem bs
  , k := ⟨[], [], [], [], []⟩ }

private theorem mkTripledMem_rdN_desc0 (bs : List Nat) :
    ∀ (k off : Nat), off + k ≤ 16 →
      (mkTripledMem bs).rdN (0x600000 + off) k = some (((bytes8 0x100000000000 ++ bytes8 bs.length).drop off).take k)
  | 0, _, _ => by simp [Mem.rdN]
  | k + 1, off, hk => by
    have hlt : off < (bytes8 0x100000000000 ++ bytes8 bs.length).length := by simp [bytes8_len]; omega
    have hhi0 : Region.hi ⟨0x600000, bytes8 0x100000000000 ++ bytes8 bs.length, true⟩ = 0x600010 := by
      simp [Region.hi, bytes8_len]
    have hrd : (mkTripledMem bs).rd (0x600000 + off) = some ((bytes8 0x100000000000 ++ bytes8 bs.length)[off]) := by
      dsimp only [mkTripledMem, Mem.rd, Mem.find, bind, Option.bind]
      rw [hhi0, if_pos (by omega)]
      dsimp only
      rw [show 0x600000 + off - 0x600000 = off by omega, List.getElem?_eq_getElem hlt]
    have ih := mkTripledMem_rdN_desc0 bs k (off + 1) (by omega)
    rw [show 0x600000 + (off + 1) = 0x600000 + off + 1 by omega] at ih
    simp only [Mem.rdN, hrd, ih, bind, Option.bind, pure]
    rw [List.drop_eq_getElem_cons hlt]; rfl

private theorem mkTripledMem_rdN_desc1 (bs : List Nat) :
    ∀ (k off : Nat), off + k ≤ 16 →
      (mkTripledMem bs).rdN (0x601000 + off) k = some (((bytes8 0x200000000000 ++ bytes8 bs.length).drop off).take k)
  | 0, _, _ => by simp [Mem.rdN]
  | k + 1, off, hk => by
    have hlt : off < (bytes8 0x200000000000 ++ bytes8 bs.length).length := by simp [bytes8_len]; omega
    have hhi0 : Region.hi ⟨0x600000, bytes8 0x100000000000 ++ bytes8 bs.length, true⟩ = 0x600010 := by
      simp [Region.hi, bytes8_len]
    have hhi1 : Region.hi ⟨0x601000, bytes8 0x200000000000 ++ bytes8 bs.length, true⟩ = 0x601010 := by
      simp [Region.hi, bytes8_len]
    have hrd : (mkTripledMem bs).rd (0x601000 + off) = some ((bytes8 0x200000000000 ++ bytes8 bs.length)[off]) := by
      dsimp only [mkTripledMem, Mem.rd, Mem.find, bind, Option.bind]
      rw [hhi0, if_neg (by omega), hhi1, if_pos (by omega)]
      dsimp only
      rw [show 0x601000 + off - 0x601000 = off by omega, List.getElem?_eq_getElem hlt]
    have ih := mkTripledMem_rdN_desc1 bs k (off + 1) (by omega)
    rw [show 0x601000 + (off + 1) = 0x601000 + off + 1 by omega] at ih
    simp only [Mem.rdN, hrd, ih, bind, Option.bind, pure]
    rw [List.drop_eq_getElem_cons hlt]; rfl

private theorem mkTripledMem_rdN_desc2 (bs : List Nat) :
    ∀ (k off : Nat), off + k ≤ 16 →
      (mkTripledMem bs).rdN (0x602000 + off) k = some (((bytes8 0x300000000000 ++ bytes8 bs.length).drop off).take k)
  | 0, _, _ => by simp [Mem.rdN]
  | k + 1, off, hk => by
    have hlt : off < (bytes8 0x300000000000 ++ bytes8 bs.length).length := by simp [bytes8_len]; omega
    have hhi0 : Region.hi ⟨0x600000, bytes8 0x100000000000 ++ bytes8 bs.length, true⟩ = 0x600010 := by
      simp [Region.hi, bytes8_len]
    have hhi1 : Region.hi ⟨0x601000, bytes8 0x200000000000 ++ bytes8 bs.length, true⟩ = 0x601010 := by
      simp [Region.hi, bytes8_len]
    have hhi2 : Region.hi ⟨0x602000, bytes8 0x300000000000 ++ bytes8 bs.length, true⟩ = 0x602010 := by
      simp [Region.hi, bytes8_len]
    have hrd : (mkTripledMem bs).rd (0x602000 + off) = some ((bytes8 0x300000000000 ++ bytes8 bs.length)[off]) := by
      dsimp only [mkTripledMem, Mem.rd, Mem.find, bind, Option.bind]
      rw [hhi0, if_neg (by omega), hhi1, if_neg (by omega), hhi2, if_pos (by omega)]
      dsimp only
      rw [show 0x602000 + off - 0x602000 = off by omega, List.getElem?_eq_getElem hlt]
    have ih := mkTripledMem_rdN_desc2 bs k (off + 1) (by omega)
    rw [show 0x602000 + (off + 1) = 0x602000 + off + 1 by omega] at ih
    simp only [Mem.rdN, hrd, ih, bind, Option.bind, pure]
    rw [List.drop_eq_getElem_cons hlt]; rfl

private theorem mkTripledMem_rdN_stk {bs : List Nat} (h44 : bs.length < 2 ^ 44) :
    ∀ (k off : Nat), off + k ≤ 8 →
      (mkTripledMem bs).rdN (0x7fffffffef00 + off) k = some (List.replicate k 0)
  | 0, _, _ => by simp [Mem.rdN]
  | k + 1, off, hk => by
    have hlt : 0x1f00 + off < 0x2000 := by omega
    have hrd : (mkTripledMem bs).rd (0x7fffffffef00 + off) = some 0 := by
      simp only [mkTripledMem, Mem.rd, Mem.find, Region.hi, List.length_append, bytes8_len,
        List.length_replicate, bind, Option.bind]
      have h1 : 0x100000000000 + bs.length < 0x200000000000 := by omega
      have h2 : 0x200000000000 + bs.length < 0x300000000000 := by omega
      have h3 : 0x300000000000 + bs.length < 0x7fffffffd000 := by omega
      rw [if_neg (by omega), if_neg (by omega), if_neg (by omega), if_neg (by omega),
        if_neg (by omega), if_neg (by omega), if_pos (by omega)]
      have hg : (List.replicate 0x2000 0)[0x7fffffffef00 + off - 0x7fffffffd000]? = some 0 := by
        rw [List.getElem?_replicate, if_pos (by omega)]
      exact hg
    have ih := mkTripledMem_rdN_stk h44 k (off + 1) (by omega)
    rw [show 0x7fffffffef00 + (off + 1) = 0x7fffffffef00 + off + 1 by omega] at ih
    simp only [Mem.rdN, hrd, ih, bind, Option.bind, pure, List.replicate_succ]

theorem mkTripledState_ok {bs : List Nat} (hby : ∀ x ∈ bs, x < 256) (h44 : bs.length < 2 ^ 44) :
    TripledState bs (mkTripledState bs) := by
  have hlW : bs.length < W := by unfold W; omega
  have hb1W : (0x100000000000 : Nat) < W := by unfold W; decide
  have hb2W : (0x200000000000 : Nat) < W := by unfold W; decide
  have hb3W : (0x300000000000 : Nat) < W := by unfold W; decide
  refine ⟨0x100000000000, 0x200000000000, 0x300000000000,
    hby, rfl, ?_, ?_, ?_, by simp [mkTripledState, St.reg],
    by simp [mkTripledState, St.reg], by simp [mkTripledState, St.reg],
    ?_, ?_, by unfold HMAX; omega, Or.inl (by decide),
    ?_, ?_, by unfold HMAX; omega, Or.inl (by decide),
    ?_, ?_, by unfold HMAX; omega, Or.inl (by decide),
    ?_, ?_, ?_, ?_⟩
  · refine ⟨by simp [mkTripledState], fun j hj _ => ?_⟩
    exact List.all_eq_true.1 mkTripledRegs_def_ok j (List.mem_range.2 hj)
  · intro j w hw
    have hlt16 : j < 16 := by
      by_cases h16 : j < 16
      · exact h16
      · exfalso
        have hlen : (mkTripledState bs).regs.length = 16 := by simp [mkTripledState]
        have hle : (mkTripledState bs).regs.length ≤ j := by rw [hlen]; exact Nat.le_of_not_lt h16
        rw [St.reg, List.getElem?_eq_none hle] at hw
        simp only [Option.getD] at hw; cases hw
    have hok := List.all_eq_true.1 mkTripledRegs_ltW_ok j (List.mem_range.2 hlt16)
    change (match (mkTripledState bs).reg j with | some w => decide (w < W) | none => true) = true at hok
    rw [hw] at hok
    exact of_decide_eq_true hok
  · intro r hr x hx
    simp only [mkTripledState, mkTripledMem, List.mem_cons, List.not_mem_nil, or_false] at hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · rcases List.mem_append.1 hx with h1 | h2
      · exact bytes8_lt_256 _ _ h1
      · exact bytes8_lt_256 _ _ h2
    · rcases List.mem_append.1 hx with h1 | h2
      · exact bytes8_lt_256 _ _ h1
      · exact bytes8_lt_256 _ _ h2
    · rcases List.mem_append.1 hx with h1 | h2
      · exact bytes8_lt_256 _ _ h1
      · exact bytes8_lt_256 _ _ h2
    · exact hby _ hx
    · exact hby _ hx
    · exact hby _ hx
    · rw [List.mem_replicate] at hx; rw [hx.2]; decide
  · intro i hi
    simp only [mkTripledState, mkTripledMem, Mem.rd, Mem.find, Region.hi, List.length_append,
      bytes8_len, List.length_replicate, bind, Option.bind]
    rw [if_neg (by omega), if_neg (by omega), if_neg (by omega), if_pos (by omega)]
    dsimp only
    rw [show 0x100000000000 + i - 0x100000000000 = i by omega, List.getElem?_eq_getElem hi]
  · refine ⟨?_, ?_⟩
    · dsimp only [mkTripledState]
      rw [ea_small (by decide), Nat.add_zero]
      have hr := mkTripledMem_rdN_desc0 bs 8 0 (by decide); rw [Nat.add_zero] at hr
      rw [rd64]
      dsimp only [Option.map]
      rw [hr, show ((bytes8 0x100000000000 ++ bytes8 bs.length).drop 0).take 8 = bytes8 0x100000000000 by simp [bytes8_len]]
      exact congrArg some (leVal_bytes8 hb1W)
    · dsimp only [mkTripledState]
      rw [ea_small (by decide), rd64]
      dsimp only [Option.map]
      rw [mkTripledMem_rdN_desc0 bs 8 8 (by decide),
        show ((bytes8 0x100000000000 ++ bytes8 bs.length).drop 8).take 8 = bytes8 bs.length by
          simp [bytes8_len]; rw [← bytes8_len bs.length, List.take_length]]
      exact congrArg some (leVal_bytes8 hlW)
  · intro i hi
    simp only [mkTripledState, mkTripledMem, Mem.rd, Mem.find, Region.hi, List.length_append,
      bytes8_len, List.length_replicate, bind, Option.bind]
    have h1 : 0x100000000000 + bs.length < 0x200000000000 := by omega
    rw [if_neg (by omega), if_neg (by omega), if_neg (by omega), if_neg (by omega), if_pos (by omega)]
    dsimp only
    rw [show 0x200000000000 + i - 0x200000000000 = i by omega, List.getElem?_eq_getElem hi]
  · refine ⟨?_, ?_⟩
    · dsimp only [mkTripledState]
      rw [ea_small (by decide), Nat.add_zero]
      have hr := mkTripledMem_rdN_desc1 bs 8 0 (by decide); rw [Nat.add_zero] at hr
      rw [rd64]
      dsimp only [Option.map]
      rw [hr, show ((bytes8 0x200000000000 ++ bytes8 bs.length).drop 0).take 8 = bytes8 0x200000000000 by simp [bytes8_len]]
      exact congrArg some (leVal_bytes8 hb2W)
    · dsimp only [mkTripledState]
      rw [ea_small (by decide), rd64]
      dsimp only [Option.map]
      rw [mkTripledMem_rdN_desc1 bs 8 8 (by decide),
        show ((bytes8 0x200000000000 ++ bytes8 bs.length).drop 8).take 8 = bytes8 bs.length by
          simp [bytes8_len]; rw [← bytes8_len bs.length, List.take_length]]
      exact congrArg some (leVal_bytes8 hlW)
  · intro i hi
    simp only [mkTripledState, mkTripledMem, Mem.rd, Mem.find, Region.hi, List.length_append,
      bytes8_len, List.length_replicate, bind, Option.bind]
    have h1 : 0x100000000000 + bs.length < 0x200000000000 := by omega
    have h2 : 0x200000000000 + bs.length < 0x300000000000 := by omega
    rw [if_neg (by omega), if_neg (by omega), if_neg (by omega), if_neg (by omega), if_neg (by omega), if_pos (by omega)]
    dsimp only
    rw [show 0x300000000000 + i - 0x300000000000 = i by omega, List.getElem?_eq_getElem hi]
  · refine ⟨?_, ?_⟩
    · dsimp only [mkTripledState]
      rw [ea_small (by decide), Nat.add_zero]
      have hr := mkTripledMem_rdN_desc2 bs 8 0 (by decide); rw [Nat.add_zero] at hr
      rw [rd64]
      dsimp only [Option.map]
      rw [hr, show ((bytes8 0x300000000000 ++ bytes8 bs.length).drop 0).take 8 = bytes8 0x300000000000 by simp [bytes8_len]]
      exact congrArg some (leVal_bytes8 hb3W)
    · dsimp only [mkTripledState]
      rw [ea_small (by decide), rd64]
      dsimp only [Option.map]
      rw [mkTripledMem_rdN_desc2 bs 8 8 (by decide),
        show ((bytes8 0x300000000000 ++ bytes8 bs.length).drop 8).take 8 = bytes8 bs.length by
          simp [bytes8_len]; rw [← bytes8_len bs.length, List.take_length]]
      exact congrArg some (leVal_bytes8 hlW)
  · intro a ha1 ha2; unfold InPassFoot at ha1 ha2; omega
  · intro a ha1 ha2; unfold InPassFoot at ha1 ha2; omega
  · intro a ha1 ha2; unfold InPassFoot at ha1 ha2; omega
  · intro sp hsp t hrd
    have hsp_eq : sp = 0x7fffffffef00 := by
      have : (mkTripledState bs).reg 4 = some 0x7fffffffef00 := by simp only [mkTripledState, St.reg]; decide
      rw [this] at hsp; cases hsp; rfl
    subst hsp_eq
    have hr := mkTripledMem_rdN_stk h44 8 0 (by decide); rw [Nat.add_zero] at hr
    dsimp only [mkTripledState, rd64] at hrd
    rw [hr] at hrd
    dsimp only [Option.map] at hrd
    have ht0 : t = 0 := by
      have : leVal (List.replicate 8 0) = 0 := by decide
      rw [this] at hrd; cases hrd; rfl
    subst ht0
    exact ⟨by unfold L_vote L_core_end; omega, im_0_none⟩

theorem mkTripledState_exists {bs : List Nat} (hby : ∀ x ∈ bs, x < 256) (h44 : bs.length < 2 ^ 44) :
    ∃ s0, TripledState bs s0 :=
  ⟨mkTripledState bs, mkTripledState_ok hby h44⟩

private theorem p1_clean_from {bs : List Nat} {b1 : Nat} {s : St}
    (hby : ∀ x ∈ bs, x < 256) (hsm : b1 + bs.length < HMAX)
    (hdisj : 0x600010 ≤ b1 ∨ b1 + bs.length ≤ 0x600000)
    (hrip : s.rip = L_p1) (hd : RegDefExcept11 s) (hlt : RegsLtW s) (hm : MemBytesLt256 s.mem)
    (hext : MemExtends (passSubMem 0x600000 b1 bs) s.mem) :
    ∃ N1 sA1 s1, 0 < N1 ∧
      run im N1 (.ok { s with mem := passSubMem 0x600000 b1 bs }) = .ok sA1 ∧
      run im N1 (.ok s) = .ok s1 ∧
      sA1.rip = L_p2 ∧ sA1.reg 0 = some (verdictWord (checkFlatA bs)) ∧
      s1.rip = L_p2 ∧ s1.reg 0 = some (verdictWord (checkFlatA bs)) ∧
      s1.mem = s.mem ∧ s1.regs.length = s.regs.length ∧
      (∀ r ∈ [4, 3, 1], s1.reg r = s.reg r) ∧ RegDefExcept11 s1 ∧ RegsLtW s1 := by
  have hbW : b1 < W := by unfold HMAX W at *; omega
  have hlW : bs.length < W := by unfold HMAX W at *; omega
  obtain ⟨v8, hv8⟩ := regDef_get hd 8
  have hspec := p1_spec (s := { s with mem := passSubMem 0x600000 b1 bs }) (buf_passSubMem hdisj) hby hsm
    (desc_passSubMem (D := 0x600000) (by unfold W; decide) hbW hlW) hd.1 hrip hv8
  exact pass_clean_run p1SafeCheck p1Tbl_im p1_hi_not_mem (fun {_ _ _} hs hn hr => (no_return_Lp2 hs hn hr).1)
    (outReg := 0) (by decide) hrip p1_mem_Lp1 hd hlt hm hext
    (hspec.mono fun _ ⟨_, hr, h0, _⟩ => ⟨hr, h0⟩)

private theorem p2_clean_from {bs : List Nat} {b2 : Nat} {s : St}
    (hby : ∀ x ∈ bs, x < 256) (hsm : b2 + bs.length < HMAX)
    (hdisj : 0x601010 ≤ b2 ∨ b2 + bs.length ≤ 0x601000)
    (hrip : s.rip = L_p2) (hd : RegDefExcept11 s) (hlt : RegsLtW s) (hm : MemBytesLt256 s.mem)
    (hext : MemExtends (passSubMem 0x601000 b2 bs) s.mem) :
    ∃ N2 sA2 s2, 0 < N2 ∧
      run im N2 (.ok { s with mem := passSubMem 0x601000 b2 bs }) = .ok sA2 ∧
      run im N2 (.ok s) = .ok s2 ∧
      sA2.rip = L_p3 ∧ sA2.reg 3 = some (verdictWord (checkFlatA bs)) ∧
      s2.rip = L_p3 ∧ s2.reg 3 = some (verdictWord (checkFlatA bs)) ∧
      s2.mem = s.mem ∧ s2.regs.length = s.regs.length ∧
      (∀ r ∈ [4, 0, 1], s2.reg r = s.reg r) ∧ RegDefExcept11 s2 ∧ RegsLtW s2 := by
  have hbW : b2 < W := by unfold HMAX W at *; omega
  have hlW : bs.length < W := by unfold HMAX W at *; omega
  obtain ⟨v8, hv8⟩ := regDef_get hd 8
  have hspec := p2_spec (s := { s with mem := passSubMem 0x601000 b2 bs }) (buf_passSubMem hdisj) hby hsm
    (desc_passSubMem (D := 0x601000) (by unfold W; decide) hbW hlW) hd.1 hrip hv8
  exact pass_clean_run p2SafeCheck p2Tbl_im p2_hi_not_mem (fun {_ _ _} hs hn hr => (no_return_Lp3 hs hn hr).1)
    (outReg := 3) (by decide) hrip p2_mem_Lp2 hd hlt hm hext
    (hspec.mono fun _ ⟨_, hr, h3, _⟩ => ⟨hr, h3⟩)

private theorem p3_clean_from {bs : List Nat} {b3 : Nat} {s : St}
    (hby : ∀ x ∈ bs, x < 256) (hsm : b3 + bs.length < HMAX)
    (hdisj : 0x602010 ≤ b3 ∨ b3 + bs.length ≤ 0x602000)
    (hrip : s.rip = L_p3) (hd : RegDefExcept11 s) (hlt : RegsLtW s) (hm : MemBytesLt256 s.mem)
    (hext : MemExtends (passSubMem 0x602000 b3 bs) s.mem) :
    ∃ N3 sA3 s3, 0 < N3 ∧
      run im N3 (.ok { s with mem := passSubMem 0x602000 b3 bs }) = .ok sA3 ∧
      run im N3 (.ok s) = .ok s3 ∧
      sA3.rip = L_vote ∧ sA3.reg 1 = some (verdictWord (checkFlatA bs)) ∧
      s3.rip = L_vote ∧ s3.reg 1 = some (verdictWord (checkFlatA bs)) ∧
      s3.mem = s.mem ∧ s3.regs.length = s.regs.length ∧
      (∀ r ∈ [4, 0, 3], s3.reg r = s.reg r) ∧ RegDefExcept11 s3 ∧ RegsLtW s3 := by
  have hbW : b3 < W := by unfold HMAX W at *; omega
  have hlW : bs.length < W := by unfold HMAX W at *; omega
  obtain ⟨v8, hv8⟩ := regDef_get hd 8
  have hspec := p3_spec (s := { s with mem := passSubMem 0x602000 b3 bs }) (buf_passSubMem hdisj) hby hsm
    (desc_passSubMem (D := 0x602000) (by unfold W; decide) hbW hlW) hd.1 hrip hv8
  exact pass_clean_run p3SafeCheck p3Tbl_im p3_hi_not_mem (fun {_ _ _} hs hn hr => (no_return_Lvote hs hn hr).1)
    (outReg := 1) (by decide) hrip p3_mem_Lp3 hd hlt hm hext
    (hspec.mono fun _ ⟨_, hr, h1, _⟩ => ⟨hr, h1⟩)

theorem safeInvalid_verdictWord {r : Option Bool} (h : r ≠ some true) :
    SporosCheckFlat.SafeInvalid (verdictWord r) := by
  cases r with
  | none => simpa [verdictWord, OVERFLOW, SporosCheckFlat.OVERFLOW] using SporosCheckFlat.SafeInvalid_overflow
  | some b => cases b with
    | false => simpa [verdictWord, REJECT, SporosCheckFlat.REJECT] using SporosCheckFlat.SafeInvalid_reject
    | true => exact absurd rfl h

private theorem postP3_no_exit0 {b3 : Nat} {bs : List Nat} {s2 : St}
    (hv : SporosCheckFlat.SafeInvalid (verdictWord (checkFlatA bs)))
    (hby : ∀ x ∈ bs, x < 256) (hsm3 : b3 + bs.length < HMAX)
    (hdisj3 : 0x602010 ≤ b3 ∨ b3 + bs.length ≤ 0x602000)
    (hrip : s2.rip = L_p3) (hd : RegDefExcept11 s2) (hlt : RegsLtW s2) (hm : MemBytesLt256 s2.mem)
    (hinv : (MemExtends (passSubMem 0x602000 b3 bs) s2.mem ∧
        (s2.reg 0 = some (verdictWord (checkFlatA bs)) ∨ s2.reg 3 = some (verdictWord (checkFlatA bs)))) ∨
      (s2.reg 0 = some (verdictWord (checkFlatA bs)) ∧ s2.reg 3 = some (verdictWord (checkFlatA bs)))) :
    ∀ m, run im m (.ok s2) ≠ .exit 0 := by
  intro m
  rcases hinv with ⟨hext3, h0 | h3⟩ | ⟨h0, h3⟩
  · obtain ⟨N3, _, s3, _, _, hrun3, _, _, hrip3, hr1_3, _, _, hforb3, hd3, hlt3⟩ :=
      p3_clean_from hby hsm3 hdisj3 hrip hd hlt hm hext3
    by_cases hm3 : N3 ≤ m
    · rw [show m = N3 + (m - N3) by omega, run_add, hrun3]
      obtain ⟨b, hb⟩ := regDef_get hd3 3
      have h0_3 : s3.reg 0 = some (verdictWord (checkFlatA bs)) := by rw [hforb3 0 (by decide)]; exact h0
      have htri : SporosCheckFlat.SinglePassFaultTriple (verdictWord (checkFlatA bs))
          (verdictWord (checkFlatA bs), b, verdictWord (checkFlatA bs)) :=
        ⟨hlt3 0 _ h0_3, hlt3 3 b hb, hlt3 1 _ hr1_3, Or.inr (Or.inl ⟨rfl, rfl⟩)⟩
      exact bin_tail_threePass_single_fault htri hrip3 hd3 h0_3 hb hr1_3 hv (m - N3)
    · rcases pass_run_trichotomy p3SafeCheck p3Tbl_im m s2 (by rw [hrip]; exact p3_mem_Lp3) with hstk | ⟨s', hr', _⟩ | ⟨t, shi, _, htm, hrunt, hipt, _⟩
      · rw [hstk]; intro h; cases h
      · rw [hr']; intro h; cases h
      · exfalso
        have hlt_m : t < N3 := by omega
        have hdiff : 0 < N3 - t := by omega
        have ha := run_add t (N3 - t) (.ok s2)
        rw [show t + (N3 - t) = N3 by omega, hrunt, hrun3] at ha
        exact (no_return_Lvote hipt hdiff ha.symm).1 hrip3
  · obtain ⟨N3, _, s3, _, _, hrun3, _, _, hrip3, hr1_3, _, _, hforb3, hd3, hlt3⟩ :=
      p3_clean_from hby hsm3 hdisj3 hrip hd hlt hm hext3
    by_cases hm3 : N3 ≤ m
    · rw [show m = N3 + (m - N3) by omega, run_add, hrun3]
      obtain ⟨a, ha⟩ := regDef_get hd3 0
      have h3_3 : s3.reg 3 = some (verdictWord (checkFlatA bs)) := by rw [hforb3 3 (by decide)]; exact h3
      have htri : SporosCheckFlat.SinglePassFaultTriple (verdictWord (checkFlatA bs))
          (a, verdictWord (checkFlatA bs), verdictWord (checkFlatA bs)) :=
        ⟨hlt3 0 a ha, hlt3 3 _ h3_3, hlt3 1 _ hr1_3, Or.inr (Or.inr ⟨rfl, rfl⟩)⟩
      exact bin_tail_threePass_single_fault htri hrip3 hd3 ha h3_3 hr1_3 hv (m - N3)
    · rcases pass_run_trichotomy p3SafeCheck p3Tbl_im m s2 (by rw [hrip]; exact p3_mem_Lp3) with hstk | ⟨s', hr', _⟩ | ⟨t, shi, _, htm, hrunt, hipt, _⟩
      · rw [hstk]; intro h; cases h
      · rw [hr']; intro h; cases h
      · exfalso
        have ha := run_add t (N3 - t) (.ok s2)
        rw [show t + (N3 - t) = N3 by omega, hrunt, hrun3] at ha
        exact (no_return_Lvote hipt (by omega) ha.symm).1 hrip3
  · rcases pass_run_trichotomy p3SafeCheck p3Tbl_im m s2 (by rw [hrip]; exact p3_mem_Lp3) with hstk | ⟨s', hr', _⟩ | ⟨t, shi, _, htm, hrunt, hipt, _, _, hforb3, hd3, hlt3⟩
    · rw [hstk]; intro h; cases h
    · rw [hr']; intro h; cases h
    · rw [show m = t + (m - t) by omega, run_add, hrunt]
      have hd3' := hd3 hd
      have hlt3' := hlt3 hlt hm
      obtain ⟨c, hc⟩ := regDef_get hd3' 1
      have h0_3 : shi.reg 0 = some (verdictWord (checkFlatA bs)) := by rw [hforb3 0 (by decide)]; exact h0
      have h3_3 : shi.reg 3 = some (verdictWord (checkFlatA bs)) := by rw [hforb3 3 (by decide)]; exact h3
      have htri : SporosCheckFlat.SinglePassFaultTriple (verdictWord (checkFlatA bs))
          (verdictWord (checkFlatA bs), verdictWord (checkFlatA bs), c) :=
        ⟨hlt3' 0 _ h0_3, hlt3' 3 _ h3_3, hlt3' 1 c hc, Or.inl ⟨rfl, rfl⟩⟩
      exact bin_tail_threePass_single_fault htri hipt hd3' h0_3 h3_3 hc hv (m - t)

private theorem postP2_no_exit0 {b2 b3 : Nat} {bs : List Nat} {s1 : St}
    (hv : SporosCheckFlat.SafeInvalid (verdictWord (checkFlatA bs)))
    (hby : ∀ x ∈ bs, x < 256) (hsm2 : b2 + bs.length < HMAX) (hsm3 : b3 + bs.length < HMAX)
    (hdisj2 : 0x601010 ≤ b2 ∨ b2 + bs.length ≤ 0x601000)
    (hdisj3 : 0x602010 ≤ b3 ∨ b3 + bs.length ≤ 0x602000)
    (hrip : s1.rip = L_p2) (hd : RegDefExcept11 s1) (hlt : RegsLtW s1) (hm : MemBytesLt256 s1.mem)
    (hinv : (MemExtends (passSubMem 0x601000 b2 bs) s1.mem ∧ MemExtends (passSubMem 0x602000 b3 bs) s1.mem) ∨
      (s1.reg 0 = some (verdictWord (checkFlatA bs)) ∧
        (MemExtends (passSubMem 0x601000 b2 bs) s1.mem ∨ MemExtends (passSubMem 0x602000 b3 bs) s1.mem))) :
    ∀ m, run im m (.ok s1) ≠ .exit 0 := by
  intro m
  rcases hinv with ⟨hext2, hext3⟩ | ⟨h0, hext2 | hext3⟩
  · obtain ⟨N2, _, s2, _, _, hrun2, _, _, hrip2, hr3_2, hm2, _, _, hd2, hlt2⟩ :=
      p2_clean_from hby hsm2 hdisj2 hrip hd hlt hm hext2
    by_cases hm2_le : N2 ≤ m
    · rw [show m = N2 + (m - N2) by omega, run_add, hrun2]
      exact postP3_no_exit0 hv hby hsm3 hdisj3 hrip2 hd2 hlt2 (by rw [hm2]; exact hm)
        (Or.inl ⟨by rw [hm2]; exact hext3, Or.inr hr3_2⟩) (m - N2)
    · rcases pass_run_trichotomy p2SafeCheck p2Tbl_im m s1 (by rw [hrip]; exact p2_mem_Lp2) with hstk | ⟨s', hr', _⟩ | ⟨t, _, _, htm, hrunt, hript, _⟩
      · rw [hstk]; intro h; cases h
      · rw [hr']; intro h; cases h
      · exfalso
        have ha := run_add t (N2 - t) (.ok s1)
        rw [show t + (N2 - t) = N2 by omega, hrunt, hrun2] at ha
        exact (no_return_Lp3 hript (by omega) ha.symm).1 hrip2
  · obtain ⟨N2, _, s2, _, _, hrun2, _, _, hrip2, hr3_2, hm2, _, hforb2, hd2, hlt2⟩ :=
      p2_clean_from hby hsm2 hdisj2 hrip hd hlt hm hext2
    by_cases hm2_le : N2 ≤ m
    · rw [show m = N2 + (m - N2) by omega, run_add, hrun2]
      have h0_2 : s2.reg 0 = some (verdictWord (checkFlatA bs)) := by rw [hforb2 0 (by decide)]; exact h0
      exact postP3_no_exit0 hv hby hsm3 hdisj3 hrip2 hd2 hlt2 (by rw [hm2]; exact hm)
        (Or.inr ⟨h0_2, hr3_2⟩) (m - N2)
    · rcases pass_run_trichotomy p2SafeCheck p2Tbl_im m s1 (by rw [hrip]; exact p2_mem_Lp2) with hstk | ⟨s', hr', _⟩ | ⟨t, _, _, htm, hrunt, hript, _⟩
      · rw [hstk]; intro h; cases h
      · rw [hr']; intro h; cases h
      · exfalso
        have ha := run_add t (N2 - t) (.ok s1)
        rw [show t + (N2 - t) = N2 by omega, hrunt, hrun2] at ha
        exact (no_return_Lp3 hript (by omega) ha.symm).1 hrip2
  · rcases pass_run_trichotomy p2SafeCheck p2Tbl_im m s1 (by rw [hrip]; exact p2_mem_Lp2) with hstk | ⟨s', hr', _⟩ | ⟨t, shi, _, htm, hrunt, hipt, hm_t, _, hforb2, hd2, hlt2⟩
    · rw [hstk]; intro h; cases h
    · rw [hr']; intro h; cases h
    · rw [show m = t + (m - t) by omega, run_add, hrunt]
      have h0_2 : shi.reg 0 = some (verdictWord (checkFlatA bs)) := by rw [hforb2 0 (by decide)]; exact h0
      exact postP3_no_exit0 hv hby hsm3 hdisj3 hipt (hd2 hd) (hlt2 hlt hm) (by rw [hm_t]; exact hm)
        (Or.inl ⟨by rw [hm_t]; exact hext3, Or.inl h0_2⟩) (m - t)

theorem tripledState_clean_trace {bs : List Nat} {s0 : St} (htri : TripledState bs s0) :
    ∃ N1 N2 N3 s_vote,
      0 < N1 ∧ 0 < N2 ∧ 0 < N3 ∧
      run im (N1 + N2 + N3) (.ok s0) = .ok s_vote ∧
      s_vote.rip = L_vote ∧ RegDefExcept11 s_vote ∧ RegsLtW s_vote ∧
      s_vote.reg 0 = some (verdictWord (checkFlatA bs)) ∧
      s_vote.reg 3 = some (verdictWord (checkFlatA bs)) ∧
      s_vote.reg 1 = some (verdictWord (checkFlatA bs)) ∧
      s_vote.mem = s0.mem ∧ BinStackSafe s_vote ∧
      Ends s0 (· = .exit (flatExit (checkFlatA bs))) := by
  rcases htri with ⟨b1, b2, b3, htri⟩
  have hext1 := mem_rd_passSubMem_extends (D := 0x600000) htri.memLt htri.buf1 htri.desc1 (by unfold W; decide) htri.disjDesc1
  have hext2 := mem_rd_passSubMem_extends (D := 0x601000) htri.memLt htri.buf2 htri.desc2 (by unfold W; decide) htri.disjDesc2
  have hext3 := mem_rd_passSubMem_extends (D := 0x602000) htri.memLt htri.buf3 htri.desc3 (by unfold W; decide) htri.disjDesc3
  obtain ⟨N1, _, s1, hp1, _, hrun1, _, _, hrip1, hr0_1, hm1, _, hforb1, hd1, hlt1⟩ :=
    p1_clean_from htri.bytes htri.small1 htri.disjDesc1 htri.rip htri.regDef htri.regsLtW htri.memLt hext1
  obtain ⟨N2, _, s2, hp2, _, hrun2, _, _, hrip2, hr3_2, hm2, _, hforb2, hd2, hlt2⟩ :=
    p2_clean_from htri.bytes htri.small2 htri.disjDesc2 hrip1 hd1 hlt1 (by rw [hm1]; exact htri.memLt)
      (by rw [hm1]; exact hext2)
  obtain ⟨N3, _, s3, hp3, _, hrun3, _, _, hrip3, hr1_3, hm3, _, hforb3, hd3, hlt3⟩ :=
    p3_clean_from htri.bytes htri.small3 htri.disjDesc3 hrip2 hd2 hlt2 (by rw [hm2, hm1]; exact htri.memLt)
      (by rw [hm2, hm1]; exact hext3)
  have hr0_3 : s3.reg 0 = some (verdictWord (checkFlatA bs)) := by
    rw [hforb3 0 (by decide), hforb2 0 (by decide)]; exact hr0_1
  have hr3_3 : s3.reg 3 = some (verdictWord (checkFlatA bs)) := by
    rw [hforb3 3 (by decide)]; exact hr3_2
  have hm_3 : s3.mem = s0.mem := hm3.trans (hm2.trans hm1)
  have hstk3 : BinStackSafe s3 := by
    intro sp hsp t hrd
    rw [hforb3 4 (by decide), hforb2 4 (by decide), hforb1 4 (by decide)] at hsp
    rw [hm_3] at hrd
    exact htri.stkSafe sp hsp t hrd
  have htot : run im (N1 + N2 + N3) (.ok s0) = .ok s3 := by
    rw [run_add, run_add, hrun1, hrun2, hrun3]
  have hvW := verdictWord_lt_W (checkFlatA bs)
  obtain ⟨Nt, hNt⟩ := tail_spec hvW hvW hvW hrip3 hd3 hr0_3 hr3_3 hr1_3
  rw [← flatExit_eq_tailExit] at hNt
  refine ⟨N1, N2, N3, s3, hp1, hp2, hp3, htot, hrip3, hd3, hlt3, hr0_3, hr3_3, hr1_3, hm_3, hstk3,
    ⟨N1 + N2 + N3 + Nt, by rw [run_add, htot]; exact hNt⟩⟩

theorem tripledState_oneStateFault_no_exit0 {bs : List Nat} {s0 : St}
    (htri : TripledState bs s0) (hbad : checkFlatA bs ≠ some true)
    (k m : Nat) {sk sk1 s2 : St}
    (hrunk : run im k (.ok s0) = .ok sk) (hstk : step im sk = .ok sk1)
    (hs2 : s2 ∈ regFlips sk1 ∨ s2 ∈ memFlips sk1 ∨ s2 ∈ flagFlips sk1) :
    run im m (.ok s2) ≠ .exit 0 := by
  rcases htri with ⟨b1, b2, b3, htri⟩
  have hv : SporosCheckFlat.SafeInvalid (verdictWord (checkFlatA bs)) := safeInvalid_verdictWord hbad
  have hext1 := mem_rd_passSubMem_extends (D := 0x600000) htri.memLt htri.buf1 htri.desc1 (by unfold W; decide) htri.disjDesc1
  have hext2 := mem_rd_passSubMem_extends (D := 0x601000) htri.memLt htri.buf2 htri.desc2 (by unfold W; decide) htri.disjDesc2
  have hext3 := mem_rd_passSubMem_extends (D := 0x602000) htri.memLt htri.buf3 htri.desc3 (by unfold W; decide) htri.disjDesc3
  obtain ⟨N1, sA1, s1, _, hrA1, hr1, hripA1, h0A1, hrip1, h0_1, hm1, _, hforb1, hd1, hlt1⟩ :=
    p1_clean_from htri.bytes htri.small1 htri.disjDesc1 htri.rip htri.regDef htri.regsLtW htri.memLt hext1
  by_cases hk1 : k < N1
  · have hsum1 := pass_oneFault_summary p1SafeCheck p1Tbl_im p1_hi_not_mem (fun {_ _ _} hs hn hr => (no_return_Lp2 hs hn hr).1)
      (outReg := 0) (by decide) htri.rip p1_mem_Lp1 htri.regDef htri.regsLtW htri.memLt hext1 hrA1 hripA1 h0A1
      hk1 hrunk hstk hs2 m
    rcases hsum1 with hno | ⟨t, s1', _, hrunt, hrip1', hd1', hlt1', hm1', hout1⟩
    · exact hno
    · rw [show m = t + (m - t) by omega, run_add, hrunt]
      rcases hout1 with ⟨h0_1', ⟨_, _, hm_eq, _⟩ | ⟨_, hm_eq | ⟨a0, _, ha0⟩⟩⟩ | ⟨_, hm_eq | ⟨a0, ha0_in, ha0⟩⟩
      · exact postP2_no_exit0 hv htri.bytes htri.small2 htri.small3 htri.disjDesc2 htri.disjDesc3
          hrip1' hd1' hlt1' hm1' (Or.inl ⟨by rw [hm_eq]; exact hext2, by rw [hm_eq]; exact hext3⟩) (m - t)
      · exact postP2_no_exit0 hv htri.bytes htri.small2 htri.small3 htri.disjDesc2 htri.disjDesc3
          hrip1' hd1' hlt1' hm1' (Or.inl ⟨by rw [hm_eq]; exact hext2, by rw [hm_eq]; exact hext3⟩) (m - t)
      · by_cases ha0_2 : InPassFoot bs 0x601000 b2 a0
        · exact postP2_no_exit0 hv htri.bytes htri.small2 htri.small3 htri.disjDesc2 htri.disjDesc3
            hrip1' hd1' hlt1' hm1' (Or.inr ⟨h0_1', Or.inr (memExtends_passSubMem_after_flip (htri.disj23 a0 ha0_2) hext3 ha0)⟩) (m - t)
        · exact postP2_no_exit0 hv htri.bytes htri.small2 htri.small3 htri.disjDesc2 htri.disjDesc3
            hrip1' hd1' hlt1' hm1' (Or.inr ⟨h0_1', Or.inl (memExtends_passSubMem_after_flip ha0_2 hext2 ha0)⟩) (m - t)
      · exact postP2_no_exit0 hv htri.bytes htri.small2 htri.small3 htri.disjDesc2 htri.disjDesc3
          hrip1' hd1' hlt1' hm1' (Or.inl ⟨by rw [hm_eq]; exact hext2, by rw [hm_eq]; exact hext3⟩) (m - t)
      · exact postP2_no_exit0 hv htri.bytes htri.small2 htri.small3 htri.disjDesc2 htri.disjDesc3
          hrip1' hd1' hlt1' hm1' (Or.inl ⟨memExtends_passSubMem_after_flip (htri.disj12 a0 ha0_in) hext2 ha0,
            memExtends_passSubMem_after_flip (htri.disj13 a0 ha0_in) hext3 ha0⟩) (m - t)
  · have hrun_k1 : run im (k - N1) (.ok s1) = .ok sk := by
      have ha := run_add N1 (k - N1) (.ok s0)
      rw [show N1 + (k - N1) = k by omega, hrunk, hr1] at ha; exact ha.symm
    have hext2_1 : MemExtends (passSubMem 0x601000 b2 bs) s1.mem := by rw [hm1]; exact hext2
    have hext3_1 : MemExtends (passSubMem 0x602000 b3 bs) s1.mem := by rw [hm1]; exact hext3
    have hm1_lt : MemBytesLt256 s1.mem := by rw [hm1]; exact htri.memLt
    obtain ⟨N2, sA2, s2', _, hrA2, hr2, hripA2, h3A2, hrip2, h3_2, hm2, _, hforb2, hd2, hlt2⟩ :=
      p2_clean_from htri.bytes htri.small2 htri.disjDesc2 hrip1 hd1 hlt1 hm1_lt hext2_1
    by_cases hk2 : k - N1 < N2
    · have hsum2 := pass_oneFault_summary p2SafeCheck p2Tbl_im p2_hi_not_mem (fun {_ _ _} hs hn hr => (no_return_Lp3 hs hn hr).1)
        (outReg := 3) (by decide) hrip1 p2_mem_Lp2 hd1 hlt1 hm1_lt hext2_1 hrA2 hripA2 h3A2
        hk2 hrun_k1 hstk hs2 m
      rcases hsum2 with hno | ⟨t, s2'', _, hrunt, hrip2', hd2', hlt2', hm2', hout2⟩
      · exact hno
      · rw [show m = t + (m - t) by omega, run_add, hrunt]
        rcases hout2 with ⟨h3_2', ⟨_, _, hm_eq, _⟩ | ⟨hforb_eq, hm_eq | ⟨a0, _, ha0⟩⟩⟩ | ⟨hforb_eq, hm_eq | ⟨a0, ha0_2, ha0⟩⟩
        · exact postP3_no_exit0 hv htri.bytes htri.small3 htri.disjDesc3 hrip2' hd2' hlt2' hm2'
            (Or.inl ⟨by rw [hm_eq]; exact hext3_1, Or.inr h3_2'⟩) (m - t)
        · exact postP3_no_exit0 hv htri.bytes htri.small3 htri.disjDesc3 hrip2' hd2' hlt2' hm2'
            (Or.inr ⟨by rw [hforb_eq 0 (by decide)]; exact h0_1, h3_2'⟩) (m - t)
        · exact postP3_no_exit0 hv htri.bytes htri.small3 htri.disjDesc3 hrip2' hd2' hlt2' hm2'
            (Or.inr ⟨by rw [hforb_eq 0 (by decide)]; exact h0_1, h3_2'⟩) (m - t)
        · exact postP3_no_exit0 hv htri.bytes htri.small3 htri.disjDesc3 hrip2' hd2' hlt2' hm2'
            (Or.inl ⟨by rw [hm_eq]; exact hext3_1, Or.inl (by rw [hforb_eq 0 (by decide)]; exact h0_1)⟩) (m - t)
        · exact postP3_no_exit0 hv htri.bytes htri.small3 htri.disjDesc3 hrip2' hd2' hlt2' hm2'
            (Or.inl ⟨memExtends_passSubMem_after_flip (htri.disj23 a0 ha0_2) hext3_1 ha0,
              Or.inl (by rw [hforb_eq 0 (by decide)]; exact h0_1)⟩) (m - t)
    · have hrun_k2 : run im (k - N1 - N2) (.ok s2') = .ok sk := by
        have ha := run_add N2 (k - N1 - N2) (.ok s1)
        rw [show N2 + (k - N1 - N2) = k - N1 by omega, hrun_k1, hr2] at ha; exact ha.symm
      have h0_2 : s2'.reg 0 = some (verdictWord (checkFlatA bs)) := by
        rw [hforb2 0 (by decide)]; exact h0_1
      have hext3_2 : MemExtends (passSubMem 0x602000 b3 bs) s2'.mem := by rw [hm2]; exact hext3_1
      have hm2_lt : MemBytesLt256 s2'.mem := by rw [hm2]; exact hm1_lt
      obtain ⟨N3, sA3, s3, _, hrA3, hr3, hripA3, h1A3, hrip3, h1_3, hm3, _, hforb3, hd3, hlt3⟩ :=
        p3_clean_from htri.bytes htri.small3 htri.disjDesc3 hrip2 hd2 hlt2 hm2_lt hext3_2
      by_cases hk3 : k - N1 - N2 < N3
      · have hsum3 := pass_oneFault_summary p3SafeCheck p3Tbl_im p3_hi_not_mem (fun {_ _ _} hs hn hr => (no_return_Lvote hs hn hr).1)
          (outReg := 1) (by decide) hrip2 p3_mem_Lp3 hd2 hlt2 hm2_lt hext3_2 hrA3 hripA3 h1A3
          hk3 hrun_k2 hstk hs2 m
        rcases hsum3 with hno | ⟨t, s3', _, hrunt, hrip3', hd3', hlt3', _, hout3⟩
        · exact hno
        · rw [show m = t + (m - t) by omega, run_add, hrunt]
          rcases hout3 with ⟨h1_3', ⟨r, hr, _, hforb_ne⟩ | ⟨hforb_eq, _⟩⟩ | ⟨hforb_eq, _⟩
          · obtain ⟨a, ha⟩ := regDef_get hd3' 0
            obtain ⟨b, hb⟩ := regDef_get hd3' 3
            simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
            rcases hr with rfl | rfl | rfl
            · have h0_3' : s3'.reg 0 = some (verdictWord (checkFlatA bs)) := by
                rw [hforb_ne 0 (by decide) (by decide)]; exact h0_2
              have h3_3' : s3'.reg 3 = some (verdictWord (checkFlatA bs)) := by
                rw [hforb_ne 3 (by decide) (by decide)]; exact h3_2
              exact bin_tail_threePass_single_fault
                ⟨hlt3' 0 (verdictWord (checkFlatA bs)) h0_3',
                 hlt3' 3 (verdictWord (checkFlatA bs)) h3_3',
                 hlt3' 1 (verdictWord (checkFlatA bs)) h1_3', Or.inl ⟨rfl, rfl⟩⟩
                hrip3' hd3' h0_3' h3_3' h1_3' hv (m - t)
            · have h3_3' : s3'.reg 3 = some (verdictWord (checkFlatA bs)) := by
                rw [hforb_ne 3 (by decide) (by decide)]; exact h3_2
              exact bin_tail_threePass_single_fault
                ⟨hlt3' 0 a ha,
                 hlt3' 3 (verdictWord (checkFlatA bs)) h3_3',
                 hlt3' 1 (verdictWord (checkFlatA bs)) h1_3', Or.inr (Or.inr ⟨rfl, rfl⟩)⟩
                hrip3' hd3' ha h3_3' h1_3' hv (m - t)
            · have h0_3' : s3'.reg 0 = some (verdictWord (checkFlatA bs)) := by
                rw [hforb_ne 0 (by decide) (by decide)]; exact h0_2
              exact bin_tail_threePass_single_fault
                ⟨hlt3' 0 (verdictWord (checkFlatA bs)) h0_3', hlt3' 3 b hb,
                 hlt3' 1 (verdictWord (checkFlatA bs)) h1_3', Or.inr (Or.inl ⟨rfl, rfl⟩)⟩
                hrip3' hd3' h0_3' hb h1_3' hv (m - t)
          · have h0_3' : s3'.reg 0 = some (verdictWord (checkFlatA bs)) := by
              rw [hforb_eq 0 (by decide)]; exact h0_2
            have h3_3' : s3'.reg 3 = some (verdictWord (checkFlatA bs)) := by
              rw [hforb_eq 3 (by decide)]; exact h3_2
            exact bin_tail_threePass_single_fault
              ⟨hlt3' 0 (verdictWord (checkFlatA bs)) h0_3',
               hlt3' 3 (verdictWord (checkFlatA bs)) h3_3',
               hlt3' 1 (verdictWord (checkFlatA bs)) h1_3', Or.inl ⟨rfl, rfl⟩⟩
              hrip3' hd3' h0_3' h3_3' h1_3' hv (m - t)
          · obtain ⟨c, hc⟩ := regDef_get hd3' 1
            have h0_3' : s3'.reg 0 = some (verdictWord (checkFlatA bs)) := by
              rw [hforb_eq 0 (by decide)]; exact h0_2
            have h3_3' : s3'.reg 3 = some (verdictWord (checkFlatA bs)) := by
              rw [hforb_eq 3 (by decide)]; exact h3_2
            exact bin_tail_threePass_single_fault
              ⟨hlt3' 0 (verdictWord (checkFlatA bs)) h0_3',
               hlt3' 3 (verdictWord (checkFlatA bs)) h3_3',
               hlt3' 1 c hc, Or.inl ⟨rfl, rfl⟩⟩
              hrip3' hd3' h0_3' h3_3' hc hv (m - t)
      · have hrun_k3 : run im (k - N1 - N2 - N3) (.ok s3) = .ok sk := by
          have ha := run_add N3 (k - N1 - N2 - N3) (.ok s2')
          rw [show N3 + (k - N1 - N2 - N3) = k - N1 - N2 by omega, hrun_k2, hr3] at ha; exact ha.symm
        have h0_3 : s3.reg 0 = some (verdictWord (checkFlatA bs)) := by
          rw [hforb3 0 (by decide)]; exact h0_2
        have h3_3 : s3.reg 3 = some (verdictWord (checkFlatA bs)) := by
          rw [hforb3 3 (by decide)]; exact h3_2
        have hstk3 : BinStackSafe s3 := by
          intro sp hsp t hrd
          rw [hforb3 4 (by decide), hforb2 4 (by decide), hforb1 4 (by decide)] at hsp
          rw [hm3.trans (hm2.trans hm1)] at hrd
          exact htri.stkSafe sp hsp t hrd
        have hpf3 := binPreFault_of_vote hv hd3.1 hstk3 hrip3 h0_3 h3_3 h1_3
        have hpf_k := BinPreFault_run (k - N1 - N2 - N3) hpf3
        rw [hrun_k3] at hpf_k
        have helem : BinElemFaultStep sk (.ok s2) := by
          rcases hs2 with hreg | hmem | hfl
          · exact BinElemFaultStep.reg hstk hreg
          · exact BinElemFaultStep.mem hstk hmem
          · exact BinElemFaultStep.flag hstk hfl
        have hts := BinPreFault_elemFaultStep_safe hpf_k helem
        exact BinTailSafe_run m hts

theorem tripledState_tail_fourChannel_no_exit0 {bs : List Nat} {s0 : St}
    (htri : TripledState bs s0) (hbad : checkFlatA bs ≠ some true) :
    ∃ N s_vote, run im N (.ok s0) = .ok s_vote ∧
      BinPreFault (verdictWord (checkFlatA bs)) s_vote ∧
      (∀ {n : Nat} {o : Out}, BinOneFaultRun false n (.ok s_vote) o → o ≠ .exit 0) ∧
      (∀ (r : Rates) (hr : r.ok), SporosCheckFlat.singleMechanism r → ∀ k m,
        SporosCheckFlat.exit0Prob (pRunWithFaultAt r hr k m (.ok s_vote)) = 0) := by
  have hv := safeInvalid_verdictWord hbad
  obtain ⟨N1, N2, N3, s_vote, _, _, _, htot, hrip, hd, _, hr0, hr3, hr1, _, hstk, _⟩ :=
    tripledState_clean_trace htri
  have hpf := binPreFault_of_vote hv hd.1 hstk hrip hr0 hr3 hr1
  exact ⟨N1 + N2 + N3, s_vote, htot, hpf,
    fun hrun => BinPreFault_OneFaultRun hrun _ rfl hpf,
    fun r hr hsing k m => binPreFault_pRunWithFaultAt_exit0Prob_zero hpf r hr hsing k m⟩

theorem tripledState_prun_noFetch_exit0Prob_zero {bs : List Nat} {s0 : St}
    (htri : TripledState bs s0) (hbad : checkFlatA bs ≠ some true)
    (r : Rates) (hr : r.ok) (hsing : SporosCheckFlat.singleMechanism r) (hnf : r.fetch = 0)
    (k m : Nat) :
    SporosCheckFlat.exit0Prob (pRunWithFaultAt r hr k m (.ok s0)) = 0 := by
  refine SporosCheckFlat.exit0Prob_eq_zero_of_forall (pRunWithFaultAt_ok r hr k m _).1 ?_
  intro y hy hpos
  have h0 : PDist.Rate (0 : Rat) := ⟨Rat.le_refl, by decide⟩
  have hmid_ok : ∀ o1 : Out, (PDist.bindF
      (match o1 with
       | .ok s1 => pstep spr1Alts (r.state hr) r.fetch im s1
       | o' => PDist.pureF o')
      (prun spr1Alts .none 0 im m)).ok := fun o1 =>
    PDist.bindF_ok (by cases o1; exact pstep_ok _ _ hr.1 _ _; exact PDist.pureF_ok _; exact PDist.pureF_ok _)
      (prun_ok _ .none h0 _ m)
  rcases mem_bindF_pos (prun_ok spr1Alts .none h0 im k _).1
    (fun o1 z hz => (hmid_ok o1).1 z hz) hy hpos with
    ⟨x1, hx1, z2, hz2, hwx1, hwz2, heq1⟩
  have hx1_eq := prun_none_zero_pos k (.ok s0) hx1 hwx1
  rcases mem_bindF_pos (by cases x1.1.1; exact (pstep_ok _ _ hr.1 _ _).1; exact (PDist.pureF_ok _).1; exact (PDist.pureF_ok _).1)
    (fun o2 z hz => (prun_ok spr1Alts .none h0 im m o2).1 z hz) hz2 hwz2 with
    ⟨x2, hx2, z3, hz3, hwx2, hwz3, heq3⟩
  have hz3_eq := prun_none_zero_pos m x2.1.1 hz3 hwz3
  have hy_eq : y.1.1 = run im m x2.1.1 :=
    (congrArg (·.1.1) heq1).trans ((congrArg (·.1.1) heq3).trans hz3_eq)
  rw [hy_eq]
  rw [hx1_eq] at hx2
  obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, hclean⟩ := tripledState_clean_trace htri
  obtain ⟨Mc, hMc⟩ := hclean
  have hflat_nz : flatExit (checkFlatA bs) ≠ 0 := by
    cases hcf : checkFlatA bs with
    | none => decide
    | some b => cases b with
      | false => decide
      | true => exact absurd hcf hbad
  have hclean_no0 : ∀ n, run im n (.ok s0) ≠ .exit 0 := by
    intro n hn
    by_cases hle : n ≤ Mc
    · have ha := bin_run_ge_exit hn hle; rw [hMc] at ha; injection ha with he; exact hflat_nz he
    · have ha := bin_run_ge_exit hMc (Nat.le_of_not_le hle); rw [hn] at ha; injection ha with he; exact hflat_nz he.symm
  cases hrunk : run im k (.ok s0) with
  | stuck =>
    rw [hrunk, mem_pureF_iff] at hx2
    have hstk : x2.1.1 = .stuck := congrArg (·.1.1) hx2
    rw [hstk, run_stuck]; intro h; cases h
  | exit c =>
    rw [hrunk, mem_pureF_iff] at hx2
    have hexc : x2.1.1 = .exit c := congrArg (·.1.1) hx2
    rw [hexc, run_exit]; intro h; cases h
    exact hclean_no0 k hrunk
  | ok sk =>
    rw [hrunk] at hx2
    have hf_ok := fetched_ok hr.1 spr1Alts im sk
    rcases mem_bindF_pos hf_ok.1 (fun o _ ho => (retire_ok (r.state hr) o).1 _ ho) hx2 hwx2 with
      ⟨⟨⟨of, tf⟩, wf⟩, hof, ⟨⟨or_, tr⟩, wr⟩, hor, hwf, hwr, heq2⟩
    dsimp only at hof hor hwf hwr heq2
    have hof_clean : of = step im sk := by
      cases him : im sk.rip with
      | none =>
        rw [him] at hof; rw [mem_pureF_iff] at hof
        exact (congrArg (·.1.1) hof).trans (by simp [step, him] : Out.stuck = step im sk)
      | some p =>
        rcases p with ⟨ins, len⟩
        rw [him] at hof
        rcases mem_unif_pos hr.1 hof hwf with h1 | ⟨hpf_pos, _⟩
        · exact (congrArg (·.1) h1).trans (by simp [step, him] : exec ins len sk = step im sk)
        · rw [hnf] at hpf_pos; exact absurd hpf_pos Rat.lt_irrefl
    rw [hof_clean] at hor
    cases hst : step im sk with
    | stuck =>
      rw [hst, mem_pureF_iff] at hor
      have hox : x2.1.1 = .stuck := by rw [heq2]; exact congrArg (·.1.1) hor
      rw [hox, run_stuck]; intro h; cases h
    | exit c =>
      rw [hst, mem_pureF_iff] at hor
      have hox : x2.1.1 = .exit c := by rw [heq2]; exact congrArg (·.1.1) hor
      rw [hox, run_exit]; intro h; cases h
      have hstep1 : run im (k + 1) (.ok s0) = .exit 0 := by
        rw [run_add k 1 (.ok s0), hrunk]; simp [run, hst]
      exact hclean_no0 (k + 1) hstep1
    | ok s1 =>
      rw [hst] at hor
      rcases mem_relabel_pos hor with ⟨⟨⟨sr, tr'⟩, wr'⟩, hhit, hrel⟩
      cases hrel; dsimp only at heq2
      have hox : x2.1.1 = .ok sr := by rw [heq2]
      rw [hox]
      dsimp only [Rates.state, Fault.seq, Fault.uniform] at hhit
      have hreg_ok := fun t => unif_ok hr.2.1 t (regFlips t)
      have hmem_ok := fun t => unif_ok hr.2.2.1 t (memFlips t)
      have hfl_ok := fun t => unif_ok hr.2.2.2 t (flagFlips t)
      have hmf_ok : ∀ t, (PDist.bindF (unif r.mem t (memFlips t))
          (fun x => unif r.flag x (flagFlips x))).ok :=
        fun t => PDist.bindF_ok (hmem_ok t) hfl_ok
      rcases mem_bindF_pos (hreg_ok s1).1 (fun a z hz => (hmf_ok a).1 z hz) hhit hwr with
        ⟨⟨⟨s2, t2⟩, w2⟩, hs2, ⟨⟨s4, t4⟩, w4⟩, hs4, hw2, hw4, heq4⟩
      rcases mem_bindF_pos (hmem_ok s2).1 (fun a z hz => (hfl_ok a).1 z hz) hs4 hw4 with
        ⟨⟨⟨s3, t3⟩, w3⟩, hs3, ⟨⟨s4', t4'⟩, w4'⟩, hs4', hw3, hw4', heq4'⟩
      cases heq4; cases heq4'; dsimp only at hs2 hs3 hs4' hw2 hw3 hw4' ⊢
      have hs2_cases := mem_unif_pos hr.2.1 hs2 hw2
      have hs3_cases := mem_unif_pos hr.2.2.1 hs3 hw3
      have hs4_cases := mem_unif_pos hr.2.2.2 hs4' hw4'
      rcases hsing with ⟨hreg0, hmem0, hfl0⟩ | ⟨hft0, hmem0, hfl0⟩ | ⟨hft0, hreg0, hfl0⟩ | ⟨hft0, hreg0, hmem0⟩
      · have hs2_eq : s2 = s1 := by
          rcases hs2_cases with h | ⟨hlt, _⟩; exact congrArg (·.1) h; rw [hreg0] at hlt; exact absurd hlt Rat.lt_irrefl
        have hs3_eq : s3 = s2 := by
          rcases hs3_cases with h | ⟨hlt, _⟩; exact congrArg (·.1) h; rw [hmem0] at hlt; exact absurd hlt Rat.lt_irrefl
        have hs4_eq : s4' = s3 := by
          rcases hs4_cases with h | ⟨hlt, _⟩; exact congrArg (·.1) h; rw [hfl0] at hlt; exact absurd hlt Rat.lt_irrefl
        rw [hs4_eq, hs3_eq, hs2_eq]
        have hcomb : run im m (.ok s1) = run im (k + 1 + m) (.ok s0) := by
          rw [run_add (k + 1) m (.ok s0), run_add k 1 (.ok s0), hrunk]; simp [run, hst]
        rw [hcomb]; exact hclean_no0 (k + 1 + m)
      · have hs3_eq : s3 = s2 := by
          rcases hs3_cases with h | ⟨hlt, _⟩; exact congrArg (·.1) h; rw [hmem0] at hlt; exact absurd hlt Rat.lt_irrefl
        have hs4_eq : s4' = s3 := by
          rcases hs4_cases with h | ⟨hlt, _⟩; exact congrArg (·.1) h; rw [hfl0] at hlt; exact absurd hlt Rat.lt_irrefl
        rw [hs4_eq, hs3_eq]
        rcases hs2_cases with h | ⟨_, _, hmem_reg⟩
        · rw [show s2 = s1 from congrArg Prod.fst h]
          have hcomb : run im m (.ok s1) = run im (k + 1 + m) (.ok s0) := by
            rw [run_add (k + 1) m (.ok s0), run_add k 1 (.ok s0), hrunk]; simp [run, hst]
          rw [hcomb]; exact hclean_no0 (k + 1 + m)
        · exact tripledState_oneStateFault_no_exit0 htri hbad k m hrunk hst (Or.inl hmem_reg)
      · have hs2_eq : s2 = s1 := by
          rcases hs2_cases with h | ⟨hlt, _⟩; exact congrArg (·.1) h; rw [hreg0] at hlt; exact absurd hlt Rat.lt_irrefl
        have hs4_eq : s4' = s3 := by
          rcases hs4_cases with h | ⟨hlt, _⟩; exact congrArg (·.1) h; rw [hfl0] at hlt; exact absurd hlt Rat.lt_irrefl
        rw [hs4_eq]; rw [hs2_eq] at hs3_cases
        rcases hs3_cases with h | ⟨_, _, hmem_m⟩
        · rw [show s3 = s1 from congrArg Prod.fst h]
          have hcomb : run im m (.ok s1) = run im (k + 1 + m) (.ok s0) := by
            rw [run_add (k + 1) m (.ok s0), run_add k 1 (.ok s0), hrunk]; simp [run, hst]
          rw [hcomb]; exact hclean_no0 (k + 1 + m)
        · exact tripledState_oneStateFault_no_exit0 htri hbad k m hrunk hst (Or.inr (Or.inl hmem_m))
      · have hs2_eq : s2 = s1 := by
          rcases hs2_cases with h | ⟨hlt, _⟩; exact congrArg (·.1) h; rw [hreg0] at hlt; exact absurd hlt Rat.lt_irrefl
        have hs3_eq : s3 = s2 := by
          rcases hs3_cases with h | ⟨hlt, _⟩; exact congrArg (·.1) h; rw [hmem0] at hlt; exact absurd hlt Rat.lt_irrefl
        rw [hs3_eq, hs2_eq] at hs4_cases
        rcases hs4_cases with h | ⟨_, _, hmem_fl⟩
        · rw [show s4' = s1 from congrArg Prod.fst h]
          have hcomb : run im m (.ok s1) = run im (k + 1 + m) (.ok s0) := by
            rw [run_add (k + 1) m (.ok s0), run_add k 1 (.ok s0), hrunk]; simp [run, hst]
          rw [hcomb]; exact hclean_no0 (k + 1 + m)
        · exact tripledState_oneStateFault_no_exit0 htri hbad k m hrunk hst (Or.inr (Or.inr hmem_fl))

end SporosCheckBin
