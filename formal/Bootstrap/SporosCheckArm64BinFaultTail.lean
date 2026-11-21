/-
SporosCheckArm64BinFaultTail: full-binary (`im` / `alts`) 4-channel single-disturbance
immunity across the `[L_vote, L_core_end)` voter + dual-`ACCEPT` guard tail, bridged
from `SporosCheckArm64Flat` via `AArch64Phys.fetch_fault_fixed_boundary` and the
`PAD0_UDF` / `PAD1_UDF` / `PAD2_UDF` isolation pads, plus `tail_entry_k01248_ne_exit0`
and the `prunWithFaultAt` -> `OneFaultRun` / `OneStateFaultRun` probabilistic unpacking lemmas.
-/
import Bootstrap.SporosCheckArm64BinTail

namespace SporosCheckArm64Bin

open AArch64Env AArch64Phys SporosCheckArm64Flat

/-! ### 3-channel (`reg`, `mem`, `nzcv`) single-step disturbance predicates -/

/-- Elementary 3-channel state disturbance (`clean`, `reg`, `mem`, `nzcv`; excludes `fetch`). -/
inductive ElemStateFaultStep (im : Img) (s : St) : Out → Prop
  | clean : ElemStateFaultStep im s (step im s)
  | reg (s1 : St) (i b : Nat) :
      step im s = .ok s1 → i < 32 → b < 64 →
      ElemStateFaultStep im s (.ok (s1.flipReg i b))
  | mem (s1 : St) (i j b : Nat) :
      step im s = .ok s1 →
      ElemStateFaultStep im s (.ok (s1.flipMem i j b))
  | nzcv (s1 s2 : St) :
      step im s = .ok s1 → s2 ∈ nzcvFlips s1 →
      ElemStateFaultStep im s (.ok s2)

theorem ElemStateFaultStep.toElemFaultStep {alts : Alts} {im : Img} {s : St} {o : Out} :
    ElemStateFaultStep im s o → ElemFaultStep alts im s o
  | .clean => .clean
  | .reg s1 i b hs hi hb => .reg s1 i b hs hi hb
  | .mem s1 i j b hs => .mem s1 i j b hs
  | .nzcv s1 s2 hs hnz => .nzcv s1 s2 hs hnz

/-- One-disturbance run over the 3 state channels (`reg`, `mem`, `nzcv`). -/
inductive OneStateFaultRun (im : Img) : Nat → Out → Out → Prop
  | of_clean (n : Nat) (o : Out) : OneStateFaultRun im n o (run im n o)
  | of_fault (n i : Nat) (o : Out) (si : St) (oi : Out) :
      i < n → run im i o = .ok si → ElemStateFaultStep im si oi →
      OneStateFaultRun im n o (run im (n - 1 - i) oi)

/-! ### Probabilistic `prunWithFaultAt` unpacking for arbitrary `(alts, im)` -/

private theorem zRate_ok : PDist.Rate 0 :=
  ⟨Rat.le_refl, show (0 : Rat) ≤ 1 from by decide⟩

private def zRates : Rates := ⟨0, 0, 0, 0⟩
private theorem zRates_ok : zRates.ok := ⟨zRate_ok, zRate_ok, zRate_ok, zRate_ok⟩

private theorem prunWFA_ok (alts : Alts) (r : Rates) (hr : r.ok) (ch : Fin 4) (im : Img) :
    ∀ fi n o, (prunWithFaultAt alts r hr ch im fi n o).ok := by
  intro fi n
  induction n generalizing fi with
  | zero => intro o; exact PDist.pureF_ok o
  | succ n ih =>
    intro o
    cases o with
    | exit c => exact PDist.pureF_ok (Out.exit c)
    | stuck => exact PDist.pureF_ok Out.stuck
    | ok s =>
      cases fi with
      | none =>
        exact PDist.bindF_ok (AArch64Phys.pstep_ok alts (zRates.state zRates_ok) zRates_ok.1 im s)
          fun _ => ih none _
      | some fk =>
        cases fk with
        | zero =>
          let sr := singleRates r hr ch
          exact PDist.bindF_ok (AArch64Phys.pstep_ok alts (sr.1.state sr.2) sr.2.1 im s)
            fun _ => ih none _
        | succ fk' =>
          exact PDist.bindF_ok (AArch64Phys.pstep_ok alts (zRates.state zRates_ok) zRates_ok.1 im s)
            fun _ => ih (some fk') _

private theorem pos_tagW {β : Type} {t : Bool} {w : Rat} {e : PDist.PDF β}
    (hw : 0 ≤ w) (he : ∀ x ∈ e, 0 ≤ x.2) {y : (β × Bool) × Rat} (hy : y ∈ PDist.tagW t w e) (hpos : 0 < y.2) :
    0 < w ∧ ∃ y0 ∈ e, y = ((y0.1.1, t || y0.1.2), w * y0.2) ∧ 0 < y0.2 := by
  unfold PDist.tagW at hy
  simp only [List.mem_map] at hy
  rcases hy with ⟨y0, hy0, rfl⟩
  have hwne : w ≠ 0 := fun h0 => by subst h0; rw [Rat.zero_mul] at hpos; exact Rat.lt_irrefl hpos
  have hy0ne : y0.2 ≠ 0 := fun h0 => by rw [h0, Rat.mul_zero] at hpos; exact Rat.lt_irrefl hpos
  exact ⟨Rat.lt_of_le_of_ne hw (Ne.symm hwne), y0, hy0, rfl,
    Rat.lt_of_le_of_ne (he y0 hy0) (Ne.symm hy0ne)⟩

private theorem pos_bindF {α β : Type} {d : PDist.PDF α}
    (hd : ∀ x ∈ d, 0 ≤ x.2) {k : α → PDist.PDF β} (hk : ∀ a, ∀ x ∈ k a, 0 ≤ x.2)
    {b : β} {tb_out : Bool} {wb_out : Rat}
    (hy : ((b, tb_out), wb_out) ∈ PDist.bindF d k) (hpos : 0 < wb_out) :
    ∃ a ta wa tb wb, ((a, ta), wa) ∈ d ∧ ((b, tb), wb) ∈ k a ∧ 0 < wa ∧ 0 < wb := by
  induction d with
  | nil => simp [PDist.bindF] at hy
  | cons x r ih =>
    obtain ⟨⟨a, ta⟩, wa⟩ := x
    simp only [PDist.bindF, List.mem_append] at hy
    rcases hy with hy1 | hy2
    · obtain ⟨_, y0, hy0, heq, hy0p⟩ := pos_tagW (hd _ List.mem_cons_self) (hk a) hy1 hpos
      obtain ⟨⟨z, tb⟩, wb⟩ := y0
      have hz : b = z := congrArg (fun x => x.1.1) heq
      subst hz
      have hwa_ne : wa ≠ 0 := fun h0 => by
        have hwb_eq : wb_out = wa * wb := congrArg Prod.snd heq
        rw [hwb_eq, h0, Rat.zero_mul] at hpos; exact Rat.lt_irrefl hpos
      exact ⟨a, ta, wa, tb, wb, List.mem_cons_self, hy0,
        Rat.lt_of_le_of_ne (hd _ List.mem_cons_self) (Ne.symm hwa_ne), hy0p⟩
    · rcases ih (fun z hz => hd z (List.mem_cons_of_mem _ hz)) hy2 with
        ⟨a', ta', wa', tb', wb', hmem, hka, hwa', hwb'⟩
      exact ⟨a', ta', wa', tb', wb', List.mem_cons_of_mem _ hmem, hka, hwa', hwb'⟩

private theorem unif_cases {β : Type} {p : Rat} (c : β) (xs : List β)
    {x : β} {t : Bool} {w : Rat} (hy : ((x, t), w) ∈ AArch64Phys.unif p c xs) :
    x = c ∨ x ∈ xs := by
  unfold AArch64Phys.unif at hy
  cases xs with
  | nil => simp [PDist.pureF] at hy; left; exact hy.1.1
  | cons z r =>
    simp only [List.mem_cons, List.mem_map] at hy
    rcases hy with hEq | ⟨u, hu, hzeq⟩
    · exact Or.inl (congrArg (fun p => p.1.1) hEq)
    · have hux : u = x := congrArg (fun p => p.1.1) hzeq
      exact Or.inr (hux ▸ List.mem_cons.2 hu)

private theorem unif_zero {β : Type} (c : β) (xs : List β)
    {x : β} {t : Bool} {w : Rat} (hy : ((x, t), w) ∈ AArch64Phys.unif 0 c xs) (hpos : 0 < w) :
    x = c := by
  unfold AArch64Phys.unif at hy
  cases xs with
  | nil => simp [PDist.pureF] at hy; exact hy.1.1
  | cons z r =>
    simp only [List.mem_cons, List.mem_map] at hy
    rcases hy with hEq | ⟨_, _, hzeq⟩
    · exact congrArg (fun p => p.1.1) hEq
    · have hwe : w = (0 : Rat) / (((z :: r).length : Nat) : Rat) := (congrArg Prod.snd hzeq).symm
      rw [hwe, Rat.div_def, Rat.zero_mul] at hpos
      exact absurd hpos Rat.lt_irrefl

private theorem fault_seq_hit (F G : AArch64Phys.Fault) (s : St) :
    (F.seq G).hit s = PDist.bindF (F.hit s) G.hit := rfl

private theorem fault_uniform_hit (alt : St → List St) (p : Rat) (hp : PDist.Rate p) (s : St) :
    (AArch64Phys.Fault.uniform alt p hp).hit s = AArch64Phys.unif p s (alt s) := rfl

private theorem retire_unpack {rc : Rates} (hrc : rc.ok) {s1 : St} {o : Out} {tb : Bool} {wb : Rat}
    (hret : ((o, tb), wb) ∈ AArch64Phys.relabel Out.ok ((rc.state hrc).hit s1)) (hpos : 0 < wb) :
    ∃ s_reg s_mem s_nzcv tr wr tm wm tn wn,
      o = .ok s_nzcv ∧
      ((s_reg, tr), wr) ∈ AArch64Phys.unif rc.reg s1 (regFlips s1) ∧ 0 < wr ∧
      ((s_mem, tm), wm) ∈ AArch64Phys.unif rc.mem s_reg (memFlips s_reg) ∧ 0 < wm ∧
      ((s_nzcv, tn), wn) ∈ AArch64Phys.unif rc.nzcv s_mem (nzcvFlips s_mem) ∧ 0 < wn := by
  simp only [AArch64Phys.relabel, List.mem_map] at hret
  rcases hret with ⟨⟨⟨s_f, tf⟩, wf⟩, hmem, heq⟩
  have heq_f : o = .ok s_f := (congrArg (fun x => x.1.1) heq).symm
  have hwf : wf = wb := congrArg Prod.snd heq
  subst heq_f; subst hwf
  rw [Rates.state, fault_seq_hit] at hmem
  obtain ⟨s_reg, tr, wr, tb2, wb2, hrg, hseq2, hwr, hwb2⟩ :=
    pos_bindF ((AArch64Phys.Fault.uniform regFlips rc.reg hrc.2.1).ok s1).1
      (fun sr => (((AArch64Phys.Fault.uniform memFlips rc.mem hrc.2.2.1).seq
        (AArch64Phys.Fault.uniform nzcvFlips rc.nzcv hrc.2.2.2)).ok sr).1) hmem hpos
  rw [fault_seq_hit] at hseq2
  obtain ⟨s_mem, tm, wm, tn, wn, hmm, hnz, hwm, hwn⟩ :=
    pos_bindF ((AArch64Phys.Fault.uniform memFlips rc.mem hrc.2.2.1).ok s_reg).1
      (fun sm => ((AArch64Phys.Fault.uniform nzcvFlips rc.nzcv hrc.2.2.2).ok sm).1) hseq2 hwb2
  rw [fault_uniform_hit] at hrg hmm hnz
  exact ⟨s_reg, s_mem, s_f, tr, wr, tm, wm, tn, wn, rfl, hrg, hwr, hmm, hwm, hnz, hwn⟩

private theorem pstep_zero_pos {alts : Alts} {im : Img} {s : St}
    {y : (Out × Bool) × Rat} (hy : y ∈ pstep alts (zRates.state zRates_ok) 0 im s)
    (hpos : 0 < y.2) : y.1.1 = step im s := by
  unfold pstep at hy
  have hf_ok := AArch64Phys.fetched_ok zRate_ok alts im s
  obtain ⟨⟨o_y, t_y⟩, w_y⟩ := y
  obtain ⟨a, ta, wa, tb, wb, ha, hret, hwa, hwb⟩ :=
    pos_bindF hf_ok.1 (fun o => (AArch64Phys.retire_ok (zRates.state zRates_ok) o).1) hy hpos
  have ha_eq : a = step im s := by
    unfold step
    cases him : im s.pc with
    | none => simp [him, PDist.pureF] at ha; exact ha.1.1
    | some p =>
      obtain ⟨ins, len⟩ := p; simp only [him] at ha
      exact unif_zero _ _ ha hwa
  cases a with
  | exit _ => simp [PDist.pureF] at hret; rw [← ha_eq]; exact hret.1.1
  | stuck => simp [PDist.pureF] at hret; rw [← ha_eq]; exact hret.1.1
  | ok s1 =>
    rcases retire_unpack zRates_ok hret hwb with
      ⟨s_r, s_m, s_n, _, wr, _, wm, _, wn, ho, hrg, hwr, hmm, hwm, hnz, hwn⟩
    rw [ho, unif_zero _ _ hnz hwn, unif_zero _ _ hmm hwm, unif_zero _ _ hrg hwr, ← ha_eq]

private theorem pstep_single_state_elem {alts : Alts} {r : Rates} (hr : r.ok) {ch : Fin 4}
    (hne0 : ch ≠ 0) {im : Img} {s : St} {y : (Out × Bool) × Rat}
    (hy : y ∈ pstep alts ((singleRates r hr ch).1.state (singleRates r hr ch).2)
        (singleRates r hr ch).1.fetch im s)
    (hpos : 0 < y.2) : ElemStateFaultStep im s y.1.1 := by
  let sr := singleRates r hr ch
  unfold pstep at hy
  have hf_ok := AArch64Phys.fetched_ok sr.2.1 alts im s
  obtain ⟨⟨o_y, t_y⟩, w_y⟩ := y
  obtain ⟨a, ta, wa, tb, wb, ha, hret, hwa, hwb⟩ :=
    pos_bindF hf_ok.1 (fun o => (AArch64Phys.retire_ok (sr.1.state sr.2) o).1) hy hpos
  have hft0 : sr.1.fetch = 0 := by
    rcases ch with ⟨_ | _ | _ | _ | _, hlt⟩
    · exact absurd rfl hne0
    · rfl
    · rfl
    · rfl
    · omega
  have ha_eq : a = step im s := by
    unfold step
    cases him : im s.pc with
    | none => simp [him, PDist.pureF] at ha; exact ha.1.1
    | some p =>
      obtain ⟨ins, len⟩ := p; simp only [him, hft0] at ha
      exact unif_zero _ _ ha hwa
  cases a with
  | exit _ =>
    simp [PDist.pureF] at hret
    rw [show o_y = step im s from hret.1.1.trans ha_eq]; exact .clean
  | stuck =>
    simp [PDist.pureF] at hret
    rw [show o_y = step im s from hret.1.1.trans ha_eq]; exact .clean
  | ok s1 =>
    have hstep : step im s = .ok s1 := ha_eq.symm
    rcases ch with ⟨_ | _ | _ | _ | _, hlt⟩
    · exact absurd rfl hne0
    · rcases retire_unpack sr.2 hret hwb with
        ⟨s_r, s_m, s_n, _, wr, _, wm, _, wn, ho, hrg, _, hmm, hwm, hnz, hwn⟩
      rw [ho, unif_zero _ _ hnz hwn, unif_zero _ _ hmm hwm]
      rcases unif_cases s1 _ hrg with rfl | hmem
      · rw [← hstep]; exact .clean
      · unfold AArch64Phys.regFlips at hmem
        simp only [List.mem_map, List.mem_range] at hmem
        rcases hmem with ⟨j, hj, rfl⟩
        exact .reg s1 (j / 64) (j % 64) hstep (by omega) (by omega)
    · rcases retire_unpack sr.2 hret hwb with
        ⟨s_r, s_m, s_n, _, wr, _, wm, _, wn, ho, hrg, hwr, hmm, _, hnz, hwn⟩
      have hr_eq := unif_zero s1 _ hrg hwr
      rw [ho, unif_zero _ _ hnz hwn]
      rcases unif_cases s_r _ hmm with heq1 | hmem
      · rw [heq1, hr_eq, ← hstep]; exact .clean
      · rw [hr_eq] at hmem
        unfold AArch64Phys.memFlips at hmem
        simp only [List.mem_flatMap, List.mem_map, List.mem_range] at hmem
        rcases hmem with ⟨i, _, j, _, b, _, rfl⟩
        exact .mem s1 i j b hstep
    · rcases retire_unpack sr.2 hret hwb with
        ⟨s_r, s_m, s_n, _, wr, _, wm, _, wn, ho, hrg, hwr, hmm, hwm, hnz, _⟩
      have hr_eq := unif_zero s1 _ hrg hwr
      have hm_eq := unif_zero s_r _ hmm hwm
      rw [ho]
      rcases unif_cases s_m _ hnz with heq1 | hmem
      · rw [heq1, hm_eq, hr_eq, ← hstep]; exact .clean
      · rw [hm_eq, hr_eq] at hmem; exact .nzcv s1 s_n hstep hmem
    · exfalso; omega

private theorem pstep_single_elem {alts : Alts} {r : Rates} (hr : r.ok) (ch : Fin 4)
    {im : Img} {s : St} {y : (Out × Bool) × Rat}
    (hy : y ∈ pstep alts ((singleRates r hr ch).1.state (singleRates r hr ch).2)
        (singleRates r hr ch).1.fetch im s)
    (hpos : 0 < y.2) : ElemFaultStep alts im s y.1.1 := by
  by_cases h0 : ch = 0
  · subst h0
    let sr := singleRates r hr 0
    unfold pstep at hy
    have hf_ok := AArch64Phys.fetched_ok sr.2.1 alts im s
    obtain ⟨⟨o_y, t_y⟩, w_y⟩ := y
    obtain ⟨a, ta, wa, tb, wb, ha, hret, _, hwb⟩ :=
      pos_bindF hf_ok.1 (fun o => (AArch64Phys.retire_ok (sr.1.state sr.2) o).1) hy hpos
    have hfa : a = step im s ∨
        ∃ ins len wAlt, im s.pc = some (ins, len) ∧ wAlt ∈ alts s.pc ∧ a = execO s wAlt := by
      unfold step
      cases him : im s.pc with
      | none => simp [him, PDist.pureF] at ha; exact Or.inl ha.1.1
      | some p =>
        obtain ⟨ins, len⟩ := p; simp only [him] at ha
        rcases unif_cases _ _ ha with heq | hmem
        · exact Or.inl heq
        · simp only [List.mem_map] at hmem
          rcases hmem with ⟨wAlt, hw, rfl⟩
          exact Or.inr ⟨ins, len, wAlt, rfl, hw, rfl⟩
    have hy_eq : o_y = a := by
      cases a with
      | exit _ => simp [PDist.pureF] at hret; exact hret.1.1
      | stuck => simp [PDist.pureF] at hret; exact hret.1.1
      | ok s1 =>
        rcases retire_unpack sr.2 hret hwb with
          ⟨s_r, s_m, s_n, _, wr, _, wm, _, wn, ho, hrg, hwr, hmm, hwm, hnz, hwn⟩
        rw [ho, unif_zero _ _ hnz hwn, unif_zero _ _ hmm hwm, unif_zero _ _ hrg hwr]
    rw [hy_eq]
    rcases hfa with rfl | ⟨ins, len, wAlt, him, hw, rfl⟩
    · exact .clean
    · exact .fetch ins len wAlt him hw
  · exact (pstep_single_state_elem hr h0 hy hpos).toElemFaultStep

private theorem run_exit_any (img : Img) (c : Nat) : ∀ n, run img n (.exit c) = .exit c
  | 0 => rfl
  | _ + 1 => rfl

private theorem run_stuck_any (img : Img) : ∀ n, run img n .stuck = .stuck
  | 0 => rfl
  | _ + 1 => rfl

private theorem prunWFA_none_pos {alts : Alts} {r : Rates} {hr : r.ok} {ch : Fin 4} {im : Img} :
    ∀ n o {y : (Out × Bool) × Rat},
      y ∈ prunWithFaultAt alts r hr ch im none n o → 0 < y.2 → y.1.1 = run im n o
  | 0, o, y, hy, _ => by simp [prunWithFaultAt, PDist.pureF] at hy; rw [hy, run]
  | n + 1, .exit c, y, hy, _ => by simp [prunWithFaultAt, PDist.pureF] at hy; rw [hy, run_exit_any]
  | n + 1, .stuck, y, hy, _ => by simp [prunWithFaultAt, PDist.pureF] at hy; rw [hy, run_stuck_any]
  | n + 1, .ok s, y, hy, hpos => by
    simp only [prunWithFaultAt] at hy
    have hp_ok := AArch64Phys.pstep_ok alts (zRates.state zRates_ok) zRates_ok.1 im s
    have hk_ok : ∀ a, ∀ x ∈ prunWithFaultAt alts r hr ch im none n a, 0 ≤ x.2 :=
      fun a => (prunWFA_ok alts r hr ch im none n a).1
    obtain ⟨a, _, _, tb, wb, ha, hrest, hwa, hwb⟩ := pos_bindF hp_ok.1 hk_ok hy hpos
    have ha_eq : a = step im s := pstep_zero_pos ha hwa
    have hrest_eq : y.1.1 = run im n a :=
      prunWFA_none_pos n a (y := ((y.1.1, tb), wb)) hrest hwb
    rw [hrest_eq, ha_eq]; rfl

theorem prunWithFaultAt_some_oneFault {alts : Alts} {r : Rates} {hr : r.ok} {ch : Fin 4} {im : Img} :
    ∀ n fi o {y : (Out × Bool) × Rat},
      y ∈ prunWithFaultAt alts r hr ch im (some fi) n o → 0 < y.2 → OneFaultRun alts im n o y.1.1
  | 0, _, o, y, hy, _ => by
    simp [prunWithFaultAt, PDist.pureF] at hy; rw [hy]; exact .of_clean 0 o
  | n + 1, _, .exit c, y, hy, _ => by
    simp [prunWithFaultAt, PDist.pureF] at hy
    rw [hy]; dsimp only; rw [← run_exit_any im c (n + 1)]; exact .of_clean (n + 1) (.exit c)
  | n + 1, _, .stuck, y, hy, _ => by
    simp [prunWithFaultAt, PDist.pureF] at hy
    rw [hy]; dsimp only; rw [← run_stuck_any im (n + 1)]; exact .of_clean (n + 1) .stuck
  | n + 1, 0, .ok s, y, hy, hpos => by
    simp only [prunWithFaultAt] at hy
    let sr := singleRates r hr ch
    have hp_ok := AArch64Phys.pstep_ok alts (sr.1.state sr.2) sr.2.1 im s
    have hk_ok : ∀ a, ∀ x ∈ prunWithFaultAt alts r hr ch im none n a, 0 ≤ x.2 :=
      fun a => (prunWFA_ok alts r hr ch im none n a).1
    obtain ⟨a, _, _, tb, wb, ha, hrest, hwa, hwb⟩ := pos_bindF hp_ok.1 hk_ok hy hpos
    have hel := pstep_single_elem hr ch ha hwa
    have hrun : y.1.1 = run im n a :=
      prunWFA_none_pos n a (y := ((y.1.1, tb), wb)) hrest hwb
    rw [hrun]; simpa using OneFaultRun.of_fault (n + 1) 0 (.ok s) s a (by omega) rfl hel
  | n + 1, fk + 1, .ok s, y, hy, hpos => by
    simp only [prunWithFaultAt] at hy
    have hp_ok := AArch64Phys.pstep_ok alts (zRates.state zRates_ok) zRates_ok.1 im s
    have hk_ok : ∀ a, ∀ x ∈ prunWithFaultAt alts r hr ch im (some fk) n a, 0 ≤ x.2 :=
      fun a => (prunWFA_ok alts r hr ch im (some fk) n a).1
    obtain ⟨a, _, _, tb, wb, ha, hrest, hwa, hwb⟩ := pos_bindF hp_ok.1 hk_ok hy hpos
    have ha_eq : a = step im s := pstep_zero_pos ha hwa
    have hsub := prunWithFaultAt_some_oneFault n fk a (y := ((y.1.1, tb), wb)) hrest hwb
    rw [ha_eq] at hsub
    generalize y.1.1 = res at hsub ⊢
    cases hsub with
    | of_clean _ => exact .of_clean (n + 1) (.ok s)
    | of_fault i _ si oi hi hri hel =>
      have hfin := OneFaultRun.of_fault (n + 1) (i + 1) (.ok s) si oi
        (by omega) (by simpa [run] using hri) hel
      rwa [show n + 1 - 1 - (i + 1) = n - 1 - i by omega] at hfin

theorem prunWithFaultAt_some_oneStateFault {alts : Alts} {r : Rates} {hr : r.ok}
    {ch : Fin 4} (hne0 : ch ≠ 0) {im : Img} :
    ∀ n fi o {y : (Out × Bool) × Rat},
      y ∈ prunWithFaultAt alts r hr ch im (some fi) n o → 0 < y.2 → OneStateFaultRun im n o y.1.1
  | 0, _, o, y, hy, _ => by
    simp [prunWithFaultAt, PDist.pureF] at hy; rw [hy]; exact .of_clean 0 o
  | n + 1, _, .exit c, y, hy, _ => by
    simp [prunWithFaultAt, PDist.pureF] at hy
    rw [hy]; dsimp only; rw [← run_exit_any im c (n + 1)]; exact .of_clean (n + 1) (.exit c)
  | n + 1, _, .stuck, y, hy, _ => by
    simp [prunWithFaultAt, PDist.pureF] at hy
    rw [hy]; dsimp only; rw [← run_stuck_any im (n + 1)]; exact .of_clean (n + 1) .stuck
  | n + 1, 0, .ok s, y, hy, hpos => by
    simp only [prunWithFaultAt] at hy
    let sr := singleRates r hr ch
    have hp_ok := AArch64Phys.pstep_ok alts (sr.1.state sr.2) sr.2.1 im s
    have hk_ok : ∀ a, ∀ x ∈ prunWithFaultAt alts r hr ch im none n a, 0 ≤ x.2 :=
      fun a => (prunWFA_ok alts r hr ch im none n a).1
    obtain ⟨a, _, _, tb, wb, ha, hrest, hwa, hwb⟩ := pos_bindF hp_ok.1 hk_ok hy hpos
    have hel := pstep_single_state_elem hr hne0 ha hwa
    have hrun : y.1.1 = run im n a :=
      prunWFA_none_pos n a (y := ((y.1.1, tb), wb)) hrest hwb
    rw [hrun]; simpa using OneStateFaultRun.of_fault (n + 1) 0 (.ok s) s a (by omega) rfl hel
  | n + 1, fk + 1, .ok s, y, hy, hpos => by
    simp only [prunWithFaultAt] at hy
    have hp_ok := AArch64Phys.pstep_ok alts (zRates.state zRates_ok) zRates_ok.1 im s
    have hk_ok : ∀ a, ∀ x ∈ prunWithFaultAt alts r hr ch im (some fk) n a, 0 ≤ x.2 :=
      fun a => (prunWFA_ok alts r hr ch im (some fk) n a).1
    obtain ⟨a, _, _, tb, wb, ha, hrest, hwa, hwb⟩ := pos_bindF hp_ok.1 hk_ok hy hpos
    have ha_eq : a = step im s := pstep_zero_pos ha hwa
    have hsub := prunWithFaultAt_some_oneStateFault hne0 n fk a (y := ((y.1.1, tb), wb)) hrest hwb
    rw [ha_eq] at hsub
    generalize y.1.1 = res at hsub ⊢
    cases hsub with
    | of_clean _ => exact .of_clean (n + 1) (.ok s)
    | of_fault i _ si oi hi hri hel =>
      have hfin := OneStateFaultRun.of_fault (n + 1) (i + 1) (.ok s) si oi
        (by omega) (by simpa [run] using hri) hel
      rwa [show n + 1 - 1 - (i + 1) = n - 1 - i by omega] at hfin

/-! ### `PAD0_UDF` / `PAD1_UDF` / `PAD2_UDF` isolation and full-image `[L_vote, L_core_end)` safety -/

private theorem decode_padWord_none : AArch64Phys.decode (fun i => padWordBytes[i]?) = none := by
  decide +kernel

private theorem im_pad_none {pref suff : List Nat} {n lo hi a : Nat}
    (hsplit : spr1Arm64Bytes = (pref ++ (List.replicate n padWordBytes).flatten) ++ suff)
    (hpref : pref.length = lo - BASE) (hhi_eq : 4 * n = hi - lo)
    (hlo_base : BASE ≤ lo) (hlo4 : lo % 4 = 0)
    (hlo : lo ≤ a) (hhi : a < hi) : im a = none := by
  unfold im imageOf
  split
  · next hc =>
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hc
    have ha4 : a % 4 = 0 := hc.2
    have hpad : (List.replicate n padWordBytes).flatten.length = hi - lo := by
      rw [replicate_pad_len]; exact hhi_eq
    have hk : (a - lo) / 4 < n := by omega
    have hbytes : ∀ i < 4, spr1Arm64Bytes[a - BASE + i]? = padWordBytes[i]? := by
      intro i hi4
      rw [hsplit]
      have hlt1 : a - BASE + i < (pref ++ (List.replicate n padWordBytes).flatten).length := by
        rw [List.length_append, hpref, hpad]; omega
      rw [List.getElem?_append_left hlt1]
      have hge1 : pref.length ≤ a - BASE + i := by rw [hpref]; omega
      rw [List.getElem?_append_right hge1, hpref]
      have hidx : a - BASE + i - (lo - BASE) = 4 * ((a - lo) / 4) + i := by omega
      rw [hidx]
      exact replicate_pad_getElem n ((a - lo) / 4) i hk hi4
    rw [AArch64Phys.decode_reads_only_4_bytes hbytes, decode_padWord_none]
  · rfl

theorem im_pad0_none {a : Nat} (hlo : BASE + 176 ≤ a) (hhi : a < L_p1) : im a = none :=
  im_pad_none (pref := hdrBytes) (suff := corePassBytes ++ pad1Bytes ++ tailBytes ++ pad2Bytes ++ initBytes) (n := 1044)
    (by simp only [spr1Arm64Bytes, pad0Bytes, List.append_assoc])
    (by rw [hdrBytes_len]; decide) (by decide) (by decide) (by decide) hlo hhi

theorem im_pad1_none {a : Nat} (hlo : L_p3_end ≤ a) (hhi : a < L_vote) : im a = none :=
  im_pad_none (pref := hdrBytes ++ pad0Bytes ++ corePassBytes) (suff := tailBytes ++ pad2Bytes ++ initBytes) (n := 4468)
    (by simp only [spr1Arm64Bytes, pad1Bytes, List.append_assoc])
    (by simp only [List.length_append, hdrBytes_len, pad0Bytes_len, corePassBytes_len]; decide)
    (by decide) (by decide) (by decide) hlo hhi

theorem im_pad2_none {a : Nat} (hlo : L_core_end ≤ a) (hhi : a < L_start) : im a = none :=
  im_pad_none (pref := hdrBytes ++ pad0Bytes ++ corePassBytes ++ pad1Bytes ++ tailBytes) (suff := initBytes) (n := 4160)
    (by simp only [spr1Arm64Bytes, pad2Bytes, List.append_assoc])
    (by simp only [List.length_append, hdrBytes_len, pad0Bytes_len, corePassBytes_len, pad1Bytes_len, tailBytes_len]; decide)
    (by decide) (by decide) (by decide) hlo hhi

theorem im_oob_none {a : Nat} (hoob : a < BASE ∨ BASE + 47636 < a + 4) : im a = none := by
  unfold im imageOf
  rw [spr1Arm64Bytes_len]
  have hf : (BASE ≤ a && a + 4 ≤ BASE + 47636 && BASE % 4 == 0 && a % 4 == 0) = false := by
    cases h : (BASE ≤ a && a + 4 ≤ BASE + 47636)
    · simp
    · simp only [Bool.and_eq_true, decide_eq_true_eq] at h
      rcases hoob with h1 | h2 <;> omega
  rw [hf]; rfl

def inTail (a : Nat) : Prop :=
  L_vote ≤ a ∧ a < L_core_end ∧ (a - L_vote) % 4 = 0

/-- Full-image tail safety invariant: `TailSafe L_vote s`, and any PC outside `[L_vote, L_core_end)`
decodes to `none` in the full `47,636`-byte ELF image `im`. -/
def FullTailSafe (s : St) : Prop :=
  TailSafe L_vote s ∧ (¬ inTail s.pc → im s.pc = none)

private def tailCleanBranchInRange : Bool :=
  (List.range 48).all fun k =>
    match decodeWord ((tailWords[k]?).getD 0) with
    | some (.b _ off) | some (.bcond _ off) | some (.cbz _ _ _ off) | some (.tbz _ _ _ off) =>
      let tgt := (L_vote + 4 * k + off) % AArch64Env.W
      L_vote ≤ tgt && tgt < L_core_end && (tgt - L_vote) % 4 == 0
    | some (.br _ _) | some (.ret _) => false
    | _ => true

private theorem tailCleanBranchInRange_ok : tailCleanBranchInRange = true := by decide +kernel

/-- Every 1-bit fetch perturbation at `k ∈ preFaultIndices` whose branch target leaves
`[L_vote, L_core_end)` lands in `PAD0_UDF`, `PAD1_UDF`, `PAD2_UDF`, or outside the
ELF image `[BASE, BASE + 47636)`, where `im tgt = none`. -/
def tailOobImNoneBool : Bool :=
  preFaultIndices.all fun k =>
    let w := (tailWords[k]?).getD 0
    (List.range 32).all fun b =>
      match decodeWord (Nat.xor w (2 ^ b)) with
      | none => true
      | some (.b _ off) | some (.bcond _ off) | some (.cbz _ _ _ off) | some (.tbz _ _ _ off) =>
        let tgt := (L_vote + 4 * k + off) % AArch64Env.W
        (L_vote ≤ tgt && tgt < L_core_end && (tgt - L_vote) % 4 == 0) ||
        (BASE + 176 ≤ tgt && tgt < L_p1) ||
        (L_p3_end ≤ tgt && tgt < L_vote) ||
        (L_core_end ≤ tgt && tgt < L_start) ||
        tgt < BASE || BASE + 47636 < tgt + 4
      | some (.br _ _) | some (.ret _) => false
      | _ => true

theorem tailOobImNone_audit : tailOobImNoneBool = true := by decide +kernel

private theorem applySys_ok_pc {s1 s' : St} {sr : Option SysRes}
    (h : applySys s1 sr = .ok s') : s'.pc = s1.pc := by
  cases sr with
  | none => cases h
  | some r =>
    cases r with
    | exit _ => simp [applySys] at h
    | ret _ _ _ => simp [applySys] at h; cases h; rfl

private theorem execSvc_ok_pc {s s' : St} {nx : Nat} (h : execSvc s nx = .ok s') : s'.pc = nx := by
  unfold execSvc at h; dsimp only at h
  split at h
  all_goals first
    | cases h
    | (split at h <;> first | cases h | exact applySys_ok_pc h)

theorem exec_pc_direct {ins : Ins} {s s' : St}
    (hnd : ins.isDirect = true) (hx : exec ins 4 s = .ok s') :
    s'.pc = (s.pc + 4) % AArch64Env.W ∨
    ∃ off, (ins = .b false off ∨ ins = .b true off ∨ (∃ cc, ins = .bcond cc off) ∨
      (∃ nz w rt, ins = .cbz nz w rt off) ∨ (∃ nz bit rt, ins = .tbz nz bit rt off)) ∧
      s'.pc = (s.pc + off) % AArch64Env.W := by
  have hlt_ok : ∀ {o : Option St}, lift o = .ok s' → o = some s' := by
    intro o ho; cases o with | none => cases ho | some t => cases ho; rfl
  have hwrZR : ∀ (t : St) w r v, (t.wrZR w r v).pc = t.pc := by intro t w r v; unfold St.wrZR; split <;> rfl
  have hwrSP : ∀ (t : St) w r v, (t.wrSP w r v).pc = t.pc := by intro t w r v; unfold St.wrSP; split <;> rfl
  cases ins with
  | b link off =>
    simp only [exec] at hx; cases hx
    cases link
    · exact Or.inr ⟨off, Or.inl rfl, rfl⟩
    · exact Or.inr ⟨off, Or.inr (Or.inl rfl), rfl⟩
  | bcond cc off =>
    simp only [exec] at hx; have h := hlt_ok hx
    cases hc : cond s.fl cc with
    | none => simp [hc] at h
    | some b =>
      simp only [hc, bind, Option.bind, pure, Option.some.injEq] at h
      cases b with
      | false => cases h; exact Or.inl rfl
      | true => cases h; exact Or.inr ⟨off, Or.inr (Or.inr (Or.inl ⟨cc, rfl⟩)), rfl⟩
  | cbz nz w rt off =>
    simp only [exec] at hx; have h := hlt_ok hx
    cases hr : s.rdZR w rt with
    | none => simp [hr] at h
    | some v =>
      simp only [hr, bind, Option.bind, pure, Option.some.injEq] at h
      cases ht : (if nz then v != 0 else v == 0)
      · rw [ht] at h; simp only [Bool.false_eq_true, ↓reduceIte] at h; cases h; exact Or.inl rfl
      · rw [ht] at h; simp only [↓reduceIte] at h; cases h; exact Or.inr ⟨off, Or.inr (Or.inr (Or.inr (Or.inl ⟨nz, w, rt, rfl⟩))), rfl⟩
  | tbz nz bit rt off =>
    simp only [exec] at hx; have h := hlt_ok hx
    cases hr : s.rdZR 64 rt with
    | none => simp [hr] at h
    | some v =>
      simp only [hr, bind, Option.bind, pure, Option.some.injEq] at h
      cases ht : (if nz then (v / pow2 bit) % 2 == 1 else !((v / pow2 bit) % 2 == 1))
      · rw [ht] at h; simp only [Bool.false_eq_true, ↓reduceIte] at h; cases h; exact Or.inl rfl
      · rw [ht] at h; simp only [↓reduceIte] at h; cases h; exact Or.inr ⟨off, Or.inr (Or.inr (Or.inr (Or.inr ⟨nz, bit, rt, rfl⟩))), rfl⟩
  | br _ _ => simp [Ins.isDirect] at hnd
  | ret _ => simp [Ins.isDirect] at hnd
  | svc imm =>
    simp only [exec] at hx; split at hx
    · cases hx
    · exact Or.inl (execSvc_ok_pc hx)
  | movWide kind w rd _ _ =>
    simp only [exec] at hx
    rcases kind with _ | _ | _ | _ | _ <;> try cases hx
    · exact Or.inl (hwrZR _ _ _ _)
    · exact Or.inl (hwrZR _ _ _ _)
    · have h := hlt_ok hx; cases hr : s.rdZR w rd <;> simp [hr] at h; cases h; exact Or.inl (hwrZR _ _ _ _)
  | addSubImm _ sf w rd rn _ =>
    simp only [exec] at hx; have h := hlt_ok hx
    cases hr : s.rdSP w rn <;> simp [hr] at h
    cases sf <;> cases h
    · exact Or.inl (hwrSP _ _ _ _)
    · exact Or.inl (hwrZR _ _ _ _)
  | addSubReg _ sf w rd rn rm _ _ =>
    simp only [exec] at hx; have h := hlt_ok hx
    cases hr1 : s.rdZR w rn <;> cases hr2 : s.rdZR w rm <;> simp [hr1, hr2] at h
    cases sf <;> cases h <;> exact Or.inl (hwrZR _ _ _ _)
  | logicReg op _ w rd rn rm _ _ =>
    simp only [exec] at hx; have h := hlt_ok hx
    cases hr1 : s.rdZR w rn <;> cases hr2 : s.rdZR w rm <;> simp [hr1, hr2] at h
    cases op <;> cases h <;> exact Or.inl (hwrZR _ _ _ _)
  | bitfield kind w rd rn _ _ =>
    simp only [exec] at hx; have h := hlt_ok hx
    cases hr1 : s.rdZR w rn <;> simp [hr1] at h
    cases kind with
    | sbfm | ubfm => cases h; exact Or.inl (hwrZR _ _ _ _)
    | bfm => cases hr2 : s.rdZR w rd <;> simp [hr2] at h; cases h; exact Or.inl (hwrZR _ _ _ _)
  | adr _ _ _ => simp only [exec] at hx; cases hx; exact Or.inl (hwrZR _ _ _ _)
  | ccmpReg _ w rn rm _ cc =>
    simp only [exec] at hx; have h := hlt_ok hx
    cases hr1 : s.rdZR w rn <;> cases hr2 : s.rdZR w rm <;> cases hc : cond s.fl cc <;> simp [hr1, hr2, hc] at h
    cases h; exact Or.inl rfl
  | csel _ w rd rn rm cc =>
    simp only [exec] at hx; have h := hlt_ok hx
    cases hr1 : s.rdZR w rn <;> cases hr2 : s.rdZR w rm <;> cases hc : cond s.fl cc <;> simp [hr1, hr2, hc] at h
    cases h; exact Or.inl (hwrZR _ _ _ _)
  | ldr sz rt rn off | ldur sz rt rn off =>
    simp only [exec] at hx; have h := hlt_ok hx
    cases hr : s.rdSP 64 rn with
    | none => simp [hr] at h
    | some base =>
      simp only [hr, bind, Option.bind, pure] at h
      split at h
      · cases h
      · cases hm : rdMem s.mem ((base + off) % AArch64Env.W) sz with
        | none => simp [hm] at h
        | some v => simp only [hm, Option.some.injEq] at h; cases h; exact Or.inl (hwrZR _ _ _ _)
  | str sz rt rn off | stur sz rt rn off =>
    simp only [exec] at hx; have h := hlt_ok hx
    cases hr : s.rdSP 64 rn with
    | none => simp [hr] at h
    | some base =>
      simp only [hr, bind, Option.bind, pure] at h
      split at h
      · cases h
      · cases hv : s.rdZR sz.width rt with
        | none => simp [hv] at h
        | some v =>
          simp only [hv] at h
          cases hm : wrMem s.mem ((base + off) % AArch64Env.W) v sz with
          | none => simp [hm] at h
          | some m' => simp only [hm, Option.some.injEq] at h; cases h; exact Or.inl rfl
  | ldrReg sz rt rn rm ext scaled =>
    simp only [exec] at hx; have h := hlt_ok hx
    cases hr : s.rdSP 64 rn with
    | none => simp [hr] at h
    | some base =>
      simp only [hr, bind, Option.bind, pure] at h
      split at h
      · cases h
      · cases hi : s.rdZR 64 rm with
        | none => simp [hi] at h
        | some idxRaw =>
          simp only [hi] at h
          split at h
          · cases h
          · simp only [Option.some.injEq] at h; cases h; exact Or.inl (hwrZR _ _ _ _)
  | strReg sz rt rn rm ext scaled =>
    simp only [exec] at hx; have h := hlt_ok hx
    cases hr : s.rdSP 64 rn with
    | none => simp [hr] at h
    | some base =>
      simp only [hr, bind, Option.bind, pure] at h
      split at h
      · cases h
      · cases hi : s.rdZR 64 rm with
        | none => simp [hi] at h
        | some idxRaw =>
          simp only [hi] at h
          cases hv : s.rdZR sz.width rt with
          | none => simp [hv] at h
          | some v =>
            simp only [hv] at h
            split at h
            · cases h
            · simp only [Option.some.injEq] at h; cases h; exact Or.inl rfl

private theorem fallthrough_inTail_or_none {k : Nat} (hk : k < 48) :
    inTail ((L_vote + 4 * k + 4) % AArch64Env.W) ∨ im ((L_vote + 4 * k + 4) % AArch64Env.W) = none := by
  have hmod : (L_vote + 4 * k + 4) % AArch64Env.W = L_vote + 4 * (k + 1) := by
    unfold L_vote AArch64Env.W; omega
  rw [hmod]
  by_cases h47 : k < 47
  · left; unfold inTail L_vote L_core_end; omega
  · right
    have heq : L_vote + 4 * (k + 1) = L_core_end := by unfold L_vote L_core_end; omega
    rw [heq]; exact im_pad2_none (Nat.le_refl _) (by decide)

private theorem step_eq_img {im1 im2 : Img} {s : St} (h : im1 s.pc = im2 s.pc) :
    step im1 s = step im2 s := by
  unfold step; rw [h]

theorem FullTailSafe_step {s : St} (ht : FullTailSafe s) :
    match step im s with
    | .ok s' => FullTailSafe s'
    | .exit c => c ≠ 0
    | .stuck => True := by
  rcases ht with ⟨hts, hout⟩
  by_cases hin : inTail s.pc
  · obtain ⟨hlo, hhi, hal⟩ := hin
    have hhi' : s.pc + 4 ≤ L_vote + 192 := by unfold L_vote L_core_end at *; omega
    have hstep_eq : step im s = step (tailImAt L_vote) s :=
      step_eq_img (im_eq_tailImAt hlo hhi')
    rw [hstep_eq]
    have hts' := TailSafe_step L_vote_base_ok hts
    cases hst : step (tailImAt L_vote) s with
    | stuck => trivial
    | exit c => rw [hst] at hts'; exact hts'
    | ok s' =>
      rw [hst] at hts'
      refine ⟨hts', ?_⟩
      intro hnot
      obtain ⟨k, hkeq⟩ : ∃ k, (s.pc - L_vote) / 4 = k := ⟨_, rfl⟩
      have hk : k < 48 := by unfold L_vote L_core_end at *; omega
      have hpc : s.pc = L_vote + 4 * k := by unfold L_vote at *; omega
      have him := tailImAt_word (base := L_vote) L_vote_base_ok.1 hk
      unfold tailStepAt at him
      cases hget_tw : tailWords[k]? with
      | none =>
        exfalso
        have hlen_tw : k < tailWords.length := by rw [tailWords_length]; exact hk
        rw [List.getElem?_eq_getElem hlen_tw] at hget_tw; cases hget_tw
      | some w =>
        rw [← hpc, hget_tw] at him
        simp only [Option.bind_some] at him
        have haud := List.all_eq_true.1 tailCleanBranchInRange_ok k (List.mem_range.2 hk)
        rw [hget_tw] at haud; simp only [Option.getD_some] at haud
        unfold step at hst; rw [him] at hst
        cases hdw : decodeWord w with
        | none => rw [hdw] at hst; cases hst
        | some ins =>
          rw [hdw] at hst
          have hx : exec ins 4 s = .ok s' := hst
          simp only [hdw] at haud
          have hdir : ins.isDirect = true := by
            cases ins <;> first | rfl | (simp at haud)
          rcases exec_pc_direct hdir hx with hfall | ⟨off, hbr, hpc'⟩
          · rw [hfall, hpc]
            rcases fallthrough_inTail_or_none hk with h1 | h2
            · exact absurd (by rw [hfall, hpc]; exact h1) hnot
            · exact h2
          · exfalso; apply hnot
            rw [hpc', hpc]
            rcases hbr with rfl | rfl | ⟨cc, rfl⟩ | ⟨nz, w', rt, rfl⟩ | ⟨nz, bit, rt, rfl⟩ <;>
              (simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at haud
               exact ⟨haud.1.1, haud.1.2, haud.2⟩)
  · have hnone := hout hin
    simp [step, hnone]

theorem FullTailSafe_run (n : Nat) {s : St} (ht : FullTailSafe s) :
    run im n (.ok s) ≠ .exit 0 := by
  induction n generalizing s with
  | zero => simp [run]
  | succ n ih =>
    simp only [run]
    have hs := FullTailSafe_step ht
    cases hst : step im s with
    | ok s' => rw [hst] at hs; exact ih hs
    | exit c => rw [hst] at hs; simp [run_exit, hs]
    | stuck => simp [run_stuck]

private theorem preFault_inTail {v : Nat} {s : St} (hp : PreFault L_vote v s) :
    inTail s.pc ∧ (s.pc - L_vote) / 4 ∈ preFaultIndices ∧ (s.pc - L_vote) / 4 < 48 := by
  rcases hp with ⟨_, _, _, _, k, hpc, hcase⟩
  have hk48 : k < 48 := by rcases hcase with ⟨_, _⟩ | ⟨_, _, _⟩ | ⟨_, _, _⟩ <;> omega
  have hk_eq : (s.pc - L_vote) / 4 = k := by rw [hpc]; omega
  refine ⟨⟨by rw [hpc]; omega, by rw [hpc]; unfold L_vote L_core_end; omega, by rw [hpc]; omega⟩, ?_, by omega⟩
  rw [hk_eq]
  rcases hcase with ⟨_, _⟩ | ⟨_, _, _⟩ | ⟨_, _, _⟩ <;>
    (simp only [preFaultIndices, List.mem_cons, List.not_mem_nil, or_false]; omega)

theorem PreFault_full_elemFaultStep {v : Nat} {s : St}
    (hp : PreFault L_vote v s) {o' : Out} (hst : ElemFaultStep alts im s o') :
    match o' with
    | .ok s' => FullTailSafe s'
    | .exit c => c ≠ 0
    | .stuck => True := by
  have ⟨hin, hkin, hk48⟩ := preFault_inTail hp
  obtain ⟨hlo, hhi, _⟩ := hin
  have hhi' : s.pc + 4 ≤ L_vote + 192 := by unfold L_vote L_core_end at *; omega
  have him_eq : im s.pc = tailImAt L_vote s.pc := im_eq_tailImAt hlo hhi'
  have halts_eq : alts s.pc = tailAltsAt L_vote s.pc := alts_eq_tailAltsAt hlo hhi'
  have hstep_eq : step im s = step (tailImAt L_vote) s := step_eq_img him_eq
  have htail_step : ElemFaultStep (tailAltsAt L_vote) (tailImAt L_vote) s o' := by
    cases hst with
    | clean => rw [hstep_eq]; exact .clean
    | fetch ins len wAlt him hw =>
      rw [him_eq] at him; rw [halts_eq] at hw; exact .fetch ins len wAlt him hw
    | reg s1 i b hs hi hb => rw [hstep_eq] at hs; exact .reg s1 i b hs hi hb
    | mem s1 i j b hs => rw [hstep_eq] at hs; exact .mem s1 i j b hs
    | nzcv s1 s2 hs hnz => rw [hstep_eq] at hs; exact .nzcv s1 s2 hs hnz
  have hts := PreFault_elemFaultStep_safe L_vote_base_ok hp htail_step
  have hok_out : ∀ s', o' = .ok s' → ¬ inTail s'.pc → im s'.pc = none := by
    intro s' hEq hnot
    have hpc_in : ∀ {s1 : St}, step (tailImAt L_vote) s = .ok s1 → inTail s1.pc := by
      intro s1 hs1
      have hps := PreFault_step L_vote_base_ok hp
      rw [hs1] at hps; exact (preFault_inTail hps).1
    cases hst with
    | clean =>
      rw [hstep_eq] at hEq; exact absurd (hpc_in hEq) hnot
    | reg s1 i b hs _ _ =>
      cases hEq; rw [hstep_eq] at hs
      have : (s1.flipReg i b).pc = s1.pc := by unfold St.flipReg; split <;> rfl
      exact absurd (this ▸ hpc_in hs) hnot
    | mem s1 i j b hs =>
      cases hEq; rw [hstep_eq] at hs
      exact absurd (hpc_in hs) hnot
    | nzcv s1 s2 hs hnz =>
      have hpc2 : s2.pc = s1.pc := by
        unfold AArch64Phys.nzcvFlips at hnz
        simp only [List.mem_append, List.mem_map] at hnz
        rcases hnz with ((⟨_, _, rfl⟩ | ⟨_, _, rfl⟩) | ⟨_, _, rfl⟩) | ⟨_, _, rfl⟩ <;> rfl
      cases hEq; rw [hstep_eq] at hs
      exact absurd (hpc2 ▸ hpc_in hs) hnot
    | fetch ins len wAlt _ hw =>
      obtain ⟨k, hkeq⟩ : ∃ k, (s.pc - L_vote) / 4 = k := ⟨_, rfl⟩
      have hk48' : k < 48 := hkeq ▸ hk48
      have hkin' : k ∈ preFaultIndices := hkeq ▸ hkin
      have hpc : s.pc = L_vote + 4 * k := by unfold L_vote at *; omega
      cases hget_tw : tailWords[k]? with
      | none =>
        exfalso
        have hlen_tw : k < tailWords.length := by rw [tailWords_length]; exact hk48'
        rw [List.getElem?_eq_getElem hlen_tw] at hget_tw; cases hget_tw
      | some w =>
        rw [halts_eq, hpc, tailAltsAt_word L_vote_base_ok.1 hk48'] at hw
        change wAlt ∈ (Option.map altsWord tailWords[k]?).getD [] at hw
        rw [hget_tw] at hw; simp only [Option.map_some, Option.getD_some] at hw
        unfold altsWord at hw
        simp only [List.mem_map, List.mem_range] at hw
        rcases hw with ⟨b, hb32, rfl⟩
        unfold execO at hEq
        cases hdw : decodeWord (Nat.xor w (2 ^ b)) with
        | none => rw [hdw] at hEq; cases hEq
        | some ins' =>
          rw [hdw] at hEq
          have hx' : exec ins' 4 s = .ok s' := hEq
          have haud := List.all_eq_true.1 (List.all_eq_true.1 tailOobImNone_audit k hkin') b (List.mem_range.2 hb32)
          rw [hget_tw] at haud; simp only [Option.getD_some] at haud
          simp only [hdw] at haud
          have hdir : ins'.isDirect = true := by
            cases ins' <;> first | rfl | (simp at haud)
          rcases exec_pc_direct hdir hx' with hfall | ⟨off, hbr, hpc'⟩
          · rw [hfall, hpc]
            rcases fallthrough_inTail_or_none hk48' with h1 | h2
            · exact absurd (by rw [hfall, hpc]; exact h1) hnot
            · exact h2
          · rw [hpc', hpc]
            rcases hbr with rfl | rfl | ⟨cc, rfl⟩ | ⟨nz, w', rt, rfl⟩ | ⟨nz, bit, rt, rfl⟩ <;>
              (simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at haud
               rcases haud with ((((h1 | h0) | h2) | h3) | h4) | h5
               · exact absurd (by rw [hpc', hpc]; exact ⟨h1.1.1, h1.1.2, h1.2⟩) hnot
               · exact im_pad0_none h0.1 h0.2
               · exact im_pad1_none h2.1 h2.2
               · exact im_pad2_none h3.1 h3.2
               · exact im_oob_none (Or.inl h4)
               · exact im_oob_none (Or.inr h5))
  cases o' with
  | stuck => trivial
  | exit c => exact hts
  | ok s' => exact ⟨hts, hok_out s' rfl⟩

private theorem inv_run_agree {im1 im2 : Img} {P : St → Prop}
    (hstep : ∀ s, P s → im1 s.pc = im2 s.pc ∧ ∀ s', step im2 s = .ok s' → P s') :
    ∀ {i : Nat} {s si : St}, P s → run im1 i (.ok s) = .ok si → P si
  | 0, _, _, hp, hri => by simp [run] at hri; cases hri; exact hp
  | i + 1, s, si, hp, hri => by
    simp only [run] at hri
    obtain ⟨him, hps⟩ := hstep s hp
    rw [step_eq_img him] at hri
    cases hs : step im2 s with
    | ok s1 => rw [hs] at hri; exact inv_run_agree hstep (hps s1 hs) hri
    | exit c => rw [hs, run_exit_any] at hri; cases hri
    | stuck => rw [hs, run_stuck_any] at hri; cases hri

/-- Full-binary 4-channel single-disturbance immunity over `[L_vote, L_core_end)` on the
complete `47,636`-byte ELF image `im` and its Hamming-1 fetch map `alts`. -/
theorem tail_full_oneFault_no_false_accept {v : Nat} {s : St}
    (hp : PreFault L_vote v s) :
    (∀ n o', OneFaultRun alts im n (.ok s) o' → o' ≠ .exit 0) ∧
    (∀ (r : Rates) (hr : r.ok) (ch : Fin 4) (fi n : Nat) (y : (Out × Bool) × Rat),
      y ∈ prunWithFaultAt alts r hr ch im (some fi) n (.ok s) → 0 < y.2 → y.1.1 ≠ .exit 0) := by
  have h1 : ∀ n o', OneFaultRun alts im n (.ok s) o' → o' ≠ .exit 0 := by
    intro n o' ho
    cases ho with
    | of_clean =>
      exact FullTailSafe_run n ⟨PreFault_to_TailSafe hp, fun hnot => absurd (preFault_inTail hp).1 hnot⟩
    | of_fault i _ si oi hi hri hst =>
      have hpi : PreFault L_vote v si :=
        inv_run_agree (im1 := im) (im2 := tailImAt L_vote) (P := PreFault L_vote v)
          (fun t ht =>
            let ⟨hlo, hhi, _⟩ := (preFault_inTail ht).1
            ⟨im_eq_tailImAt hlo (by unfold L_vote L_core_end at *; omega),
             fun s' hs' => by have h := PreFault_step L_vote_base_ok ht; rw [hs'] at h; exact h⟩) hp hri
      have hsaf := PreFault_full_elemFaultStep hpi hst
      cases oi with
      | ok s' => exact FullTailSafe_run (n - 1 - i) hsaf
      | exit c => simp [run_exit, hsaf]
      | stuck => simp [run_stuck]
  exact ⟨h1, fun r hr ch fi n y hy hpos => h1 n y.1.1 (prunWithFaultAt_some_oneFault n fi (.ok s) hy hpos)⟩

/-- Clean tail execution from any of the `P3` `b vote` 1-bit-perturbed entry offsets
`k ∈ {0, 1, 2, 4, 8}` with `x19 = x20 = v` (`valNotAcc v`) and `x21 = c < W`
never produces `.exit 0`. -/
theorem tail_entry_k01248_ne_exit0 {v c k : Nat} {s : St}
    (hk : k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 4 ∨ k = 8)
    (hpc : s.pc = L_vote + 4 * k) (hl : s.regs.length = 31)
    (h19 : s.reg 19 = some v) (h20 : s.reg 20 = some v) (h21 : s.reg 21 = some c)
    (hv : valNotAcc v) (hc : c < AArch64Env.W) :
    ∀ n, run im n (.ok s) ≠ .exit 0 := by
  have hmk : ∀ {t : St} {kt : Nat}, kt < 48 → t.pc = L_vote + 4 * kt → t.regs.length = 31 →
      tailSafeAt t kt → ∀ n, run im n (.ok t) ≠ .exit 0 := by
    intro t kt hkt hpct hlt hts n
    have hin : inTail t.pc := by
      rw [hpct]; unfold inTail L_vote L_core_end; omega
    have hTS : TailSafe L_vote t := by
      refine ⟨hlt, ?_⟩
      have hcond : L_vote ≤ t.pc ∧ t.pc < L_vote + 192 ∧ (t.pc - L_vote) % 4 = 0 := by
        rw [hpct]; omega
      rw [dif_pos hcond, show (t.pc - L_vote) / 4 = kt by rw [hpct]; omega]
      exact hts
    exact FullTailSafe_run n ⟨hTS, fun hnot => absurd hin hnot⟩
  have hk2_safe : ∀ {t : St}, t.pc = L_vote + 4 * 2 → t.regs.length = 31 →
      t.reg 19 = some v → t.reg 20 = some v → t.reg 21 = some c →
      ∀ n, run im n (.ok t) ≠ .exit 0 := by
    intro t hpct hlt ht19 ht20 ht21 n
    cases hz : t.fl.z with
    | none =>
      cases n with
      | zero => simp [run]
      | succ n' =>
        have him2 : im t.pc = some (.csel .csel 64 19 19 20 0, 4) := by
          rw [hpct]; exact table_ok (L_vote + 8, .csel .csel 64 19 19 20 0, 4)
            (List.mem_append.2 (Or.inl (List.mem_append.2 (Or.inr (by decide)))))
        have hst : step im t = .stuck := by
          simp [step, him2, exec, lift, bind, Option.bind, St.rdZR, ht19, ht20, AArch64Phys.cond, hz]
        simp [run, hst, run_stuck]
    | some b =>
      cases b with
      | true =>
        exact hmk (by decide) hpct hlt
          (Or.inr (Or.inr ⟨v, v, c, ht19, ht20, ht21, hv.1, hv.1, hc, Or.inl ⟨hz, hv⟩⟩)) n
      | false =>
        exact hmk (by decide) hpct hlt
          (Or.inr (Or.inr ⟨v, v, c, ht19, ht20, ht21, hv.1, hv.1, hc, Or.inr ⟨hz, fun _ => hv, fun _ => hv⟩⟩)) n
  rcases hk with rfl | rfl | rfl | rfl | rfl
  · exact hmk (by decide) hpc hl ⟨v, v, c, h19, h20, h21, hv.1, hv.1, hc, Or.inr ⟨v, by unfold TMR.vote; simp, hv⟩⟩
  · intro n
    cases n with
    | zero => simp [run]
    | succ n' =>
      have him1 : im s.pc = some (.ccmpReg true 64 19 21 6 1, 4) := by
        rw [hpc]; exact table_ok (L_vote + 4, .ccmpReg true 64 19 21 6 1, 4)
          (List.mem_append.2 (Or.inl (List.mem_append.2 (Or.inr (by decide)))))
      cases hz : s.fl.z with
      | none =>
        have hst : step im s = .stuck := by
          simp [step, him1, exec, lift, bind, Option.bind, St.rdZR, h19, h21, AArch64Phys.cond, hz]
        simp [run, hst, run_stuck]
      | some b =>
        have hpc1 : (s.pc + 4) % AArch64Env.W = L_vote + 4 * 2 := by rw [hpc]; decide
        have hst : ∃ fl', step im s = .ok { s with pc := L_vote + 4 * 2, fl := fl' } := by
          simp only [step, him1, exec, lift, bind, Option.bind, pure, St.rdZR,
            show (19 == 31) = false by decide, show (21 == 31) = false by decide,
            Bool.false_eq_true, ↓reduceIte, h19, h21, AArch64Phys.cond, hz, hpc1]
          cases b <;> exact ⟨_, rfl⟩
        obtain ⟨fl', hst⟩ := hst
        simp only [run, hst]
        exact hk2_safe (t := { s with pc := L_vote + 4 * 2, fl := fl' }) rfl hl h19 h20 h21 n'
  · exact hk2_safe hpc hl h19 h20 h21
  · intro n
    cases hz : s.fl.z with
    | none =>
      cases n with
      | zero => simp [run]
      | succ n' =>
        have him4 : im s.pc = some (.bcond 1 16, 4) := by
          rw [hpc]; exact table_ok (L_vote + 16, .bcond 1 16, 4)
            (List.mem_append.2 (Or.inl (List.mem_append.2 (Or.inr (by decide)))))
        have hst : step im s = .stuck := by
          simp [step, him4, exec, lift, bind, Option.bind, AArch64Phys.cond, hz]
        simp [run, hst, run_stuck]
    | some b =>
      cases b with
      | true => exact hmk (by decide) hpc hl (Or.inr (Or.inl ⟨hz, v, h19, hv⟩)) n
      | false => exact hmk (by decide) hpc hl (Or.inr (Or.inr hz)) n
  · exact hmk (by decide) hpc hl trivial

end SporosCheckArm64Bin
