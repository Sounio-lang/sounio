/-
SporosCheckArm64BinFaultIso: static pass-isolation audit over `corePassTable`
and `tailTable` (`decide +kernel`), exhaustive 1-bit fetch audit over all
2,046 instructions in `corePassBytes` (`decide +kernel`), generic pass-step
invariants (`VerdictRegsWf`, `MemSub` monotonicity, unread-register congruence,
strict-range progression, opcode-register bounds, non-`ACCEPT` clean invariant),
and `wrMem` locality helpers for `TripledState`.
-/
import Bootstrap.SporosCheckArm64BinFaultTail

namespace SporosCheckArm64Bin

open AArch64Env AArch64Phys SporosCheckArm64Flat

/-! ### Static pass-isolation and step-invariant audit over `corePassTable` + `tailTable` -/

def isOtherV (R r : Nat) : Bool :=
  (r == 19 || r == 20 || r == 21) && r != R

def wOk (w : Nat) : Bool := w == 32 || w == 64

/-- Offsets within a pass where `x14` is guaranteed to hold a byte `< 256` from `ldr.b1 x14`. -/
def inR14 (off : Nat) : Bool :=
  (160 < off && off ≤ 204) ||
  (352 < off && off ≤ 396) ||
  (520 < off && off ≤ 564) ||
  (2156 < off && off ≤ 2200)

/-- Offsets within a pass where `x15` is guaranteed to hold an opcode byte `< 256` from `ldr.b1 x15`. -/
def inR15 (off : Nat) : Bool :=
  776 < off && off < 1616

/-- Register written by a clean pass instruction, if any (returns `31` when discarding to XZR). -/
def insWriteReg : Ins → Option Nat
  | .movWide _ _ rd _ _ => some rd
  | .addSubImm _ _ _ rd _ _ => some rd
  | .addSubReg _ _ _ rd _ _ _ _ => some rd
  | .logicReg _ _ _ rd _ _ _ _ => some rd
  | .adr _ rd _ => some rd
  | .ldr _ rt _ _ | .ldur _ rt _ _ => some rt
  | _ => none

/-- Static check that clean step preserves `inR14`, `inR15`, and the `acc`-block / non-`ACCEPT` invariants. -/
def insStepExtraOk (lo R pc : Nat) (ins : Ins) : Bool :=
  let off := pc - lo
  let wr := insWriteReg ins
  let wrNe (r : Nat) : Bool := match wr with | none => true | some d => d != r
  let fallOk : Bool :=
    match ins with
    | .b false _ => true
    | _ =>
      let off' := off + 4
      let ok14 := if !inR14 off && inR14 off' then
          match ins with | .ldr .b1 14 _ _ => true | _ => false
        else if inR14 off then wrNe 14
        else true
      let ok15 := if !inR15 off && inR15 off' then
          match ins with | .ldr .b1 15 _ _ => true | _ => false
        else if inR15 off then wrNe 15
        else true
      let okAccFall := !(off == 2664)
      ok14 && ok15 && okAccFall
  let brOk : Bool :=
    match ins with
    | .b false offB | .bcond _ offB | .cbz _ _ _ offB =>
      let tgt := (pc + offB) % AArch64Env.W
      let toff := tgt - lo
      (if !inR14 off then !inR14 toff else true) &&
      (if !inR15 off then !inR15 toff else true) &&
      !(2668 < toff && toff < 2688)
    | _ => true
  let verdictOk : Bool :=
    if 2668 ≤ off && off < 2688 then true
    else if wrNe R then true
    else match ins with
      | .movWide 2 64 rd imm16 0 => rd == R && imm16 < 65536 && imm16 != 15510
      | .movWide 3 64 rd _ hw => rd == R && 1 ≤ hw && hw < 4
      | _ => false
  fallOk && brOk && verdictOk

/-- Static isolation check for a pass instruction at `pc ∈ [lo, hi)` owning verdict register `R`
and exiting to `nxt`: read-only on memory, no syscalls/indirect jumps, width in `{32, 64}`,
non-unconditional-branch instructions satisfy `pc + 4 < hi`, every direct branch targets
`(lo + 16, hi) ∪ {nxt}` 4-byte-aligned, neither reads nor writes any `r ∈ {19, 20, 21} \ {R}`,
and satisfies `insStepExtraOk`. -/
def insPassSafe (lo hi nxt R pc : Nat) (ins : Ins) : Bool :=
  let tgtOk (off : Nat) : Bool :=
    let tgt := (pc + off) % AArch64Env.W
    (lo + 16 < tgt && tgt < hi && (tgt - lo) % 4 == 0) || tgt == nxt
  let coreOk : Bool :=
    match ins with
    | .movWide _ w rd _ _ => pc + 4 < hi && wOk w && !isOtherV R rd
    | .addSubImm _ _ w rd rn _ => pc + 4 < hi && wOk w && !isOtherV R rd && !isOtherV R rn
    | .addSubReg _ _ w rd rn rm _ _ => pc + 4 < hi && wOk w && !isOtherV R rd && !isOtherV R rn && !isOtherV R rm
    | .logicReg _ _ w rd rn rm _ _ => pc + 4 < hi && wOk w && !isOtherV R rd && !isOtherV R rn && !isOtherV R rm
    | .adr _ rd _ => pc + 4 < hi && !isOtherV R rd
    | .ldr _ rt rn _ | .ldur _ rt rn _ => pc + 4 < hi && !isOtherV R rt && !isOtherV R rn
    | .b false off => tgtOk off
    | .bcond _ off => pc + 4 < hi && tgtOk off
    | .cbz _ w rt off => pc + 4 < hi && wOk w && !isOtherV R rt && tgtOk off
    | _ => false
  coreOk && insStepExtraOk lo R pc ins

/-- Tail clean-progression check: every direct branch in `[L_vote, L_core_end)` targets `(L_vote, L_core_end)`. -/
def insTailGtVote (pc : Nat) : Ins → Bool
  | .b _ off | .bcond _ off | .cbz _ _ _ off | .tbz _ _ _ off =>
    let tgt := (pc + off) % AArch64Env.W
    L_vote < tgt && tgt < L_core_end
  | .br _ _ | .ret _ => false
  | _ => true

private def tableAddrOk : Nat → List (Nat × Ins × Nat) → Bool
  | _, [] => true
  | a, (a', _, l) :: t => a == a' && l == 4 && tableAddrOk (a + 4) t

private def corePassAuditBool : Bool :=
  tableAddrOk L_p1 corePassTable &&
  (L_p1 + 4 * corePassTable.length == L_p3_end) &&
  tableAddrOk L_vote tailTable &&
  (L_vote + 4 * tailTable.length == L_core_end) &&
  corePassTable.all (fun e =>
    let pc := e.1; let ins := e.2.1
    if L_p1 ≤ pc && pc < L_p2 then insPassSafe L_p1 L_p2 L_p2 19 pc ins
    else if L_p2 ≤ pc && pc < L_p3 then insPassSafe L_p2 L_p3 L_p3 20 pc ins
    else insPassSafe L_p3 L_p3_end L_vote 21 pc ins) &&
  tailTable.all (fun e => insTailGtVote e.1 e.2.1)

private theorem corePassAudit_ok : corePassAuditBool = true := by decide +kernel

private theorem tableAddrOk_mem : ∀ (lo a : Nat) (tbl : List (Nat × Ins × Nat)),
    tableAddrOk lo tbl = true → lo ≤ a → a < lo + 4 * tbl.length → (a - lo) % 4 = 0 →
    ∃ ins, (a, ins, 4) ∈ tbl
  | _, _, [], _, _, hlt, _ => by simp at hlt; omega
  | lo, a, (a', ins, l) :: t, hok, hlo, hlt, hal => by
    simp only [tableAddrOk, Bool.and_eq_true, beq_iff_eq] at hok
    obtain ⟨⟨rfl, rfl⟩, ht⟩ := hok
    by_cases heq : a = lo
    · subst heq; exact ⟨ins, List.mem_cons_self⟩
    · have h4 : lo + 4 ≤ a := by omega
      have hlen : a < (lo + 4) + 4 * t.length := by simp only [List.length_cons] at hlt; omega
      obtain ⟨ins', hmem⟩ := tableAddrOk_mem (lo + 4) a t ht h4 hlen (by omega)
      exact ⟨ins', List.mem_cons_of_mem _ hmem⟩

abbrev PassSpec (lo hi nxt R : Nat) : Prop :=
  (lo = L_p1 ∧ hi = L_p2 ∧ nxt = L_p2 ∧ R = 19) ∨
  (lo = L_p2 ∧ hi = L_p3 ∧ nxt = L_p3 ∧ R = 20) ∨
  (lo = L_p3 ∧ hi = L_p3_end ∧ nxt = L_vote ∧ R = 21)

private theorem im_in_corePassTable {a len : Nat} {ins : Ins}
    (hlo : L_p1 ≤ a) (hhi : a < L_p3_end) (him : im a = some (ins, len)) :
    len = 4 ∧ (a, ins, 4) ∈ corePassTable := by
  have ⟨ha4, _⟩ := im_fixedWidth a ins len him
  have haud := corePassAudit_ok
  unfold corePassAuditBool at haud
  simp only [Bool.and_eq_true, beq_iff_eq] at haud
  obtain ⟨⟨⟨⟨⟨haddr, hce⟩, _⟩, _⟩, _⟩, _⟩ := haud
  have hal : (a - L_p1) % 4 = 0 := by unfold L_p1; omega
  obtain ⟨ins', hmem⟩ := tableAddrOk_mem L_p1 a corePassTable haddr hlo (by rw [hce]; exact hhi) hal
  have him' : im a = some (ins', 4) :=
    table_ok (a, ins', 4) (List.mem_append.2 (Or.inl (List.mem_append.2 (Or.inl hmem))))
  rw [him] at him'; cases him'
  exact ⟨rfl, hmem⟩

private theorem im_in_tailTable {a len : Nat} {ins : Ins}
    (hlo : L_vote ≤ a) (hhi : a < L_core_end) (him : im a = some (ins, len)) :
    len = 4 ∧ (a, ins, 4) ∈ tailTable := by
  have ⟨ha4, _⟩ := im_fixedWidth a ins len him
  have haud := corePassAudit_ok
  unfold corePassAuditBool at haud
  simp only [Bool.and_eq_true, beq_iff_eq] at haud
  obtain ⟨⟨⟨⟨⟨_, _⟩, haddr⟩, hce⟩, _⟩, _⟩ := haud
  have hal : (a - L_vote) % 4 = 0 := by unfold L_vote; omega
  obtain ⟨ins', hmem⟩ := tableAddrOk_mem L_vote a tailTable haddr hlo (by rw [hce]; exact hhi) hal
  have him' : im a = some (ins', 4) :=
    table_ok (a, ins', 4) (List.mem_append.2 (Or.inl (List.mem_append.2 (Or.inr hmem))))
  rw [him] at him'; cases him'
  exact ⟨rfl, hmem⟩

theorem im_in_pass_safe {lo hi nxt R a len : Nat} {ins : Ins}
    (hpass : PassSpec lo hi nxt R) (hlo : lo ≤ a) (hhi : a < hi) (him : im a = some (ins, len)) :
    len = 4 ∧ insPassSafe lo hi nxt R a ins = true := by
  have ha_ge : L_p1 ≤ a := by
    rcases hpass with ⟨rfl, rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl, rfl⟩ <;>
      (dsimp only [L_p1, L_p2, L_p3] at *; omega)
  have ha_lt : a < L_p3_end := by
    rcases hpass with ⟨rfl, rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl, rfl⟩ <;>
      (dsimp only [L_p1, L_p2, L_p3, L_p3_end] at *; omega)
  obtain ⟨rfl, hmem⟩ := im_in_corePassTable ha_ge ha_lt him
  have haud := corePassAudit_ok
  unfold corePassAuditBool at haud
  simp only [Bool.and_eq_true] at haud
  have he := List.all_eq_true.1 haud.1.2 (a, ins, 4) hmem
  dsimp only at he
  refine ⟨rfl, ?_⟩
  rcases hpass with ⟨rfl, rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl, rfl⟩
  · simpa [hlo, hhi] using he
  · have h1 : ¬ (L_p1 ≤ a ∧ a < L_p2) := by unfold L_p1 L_p2 at *; omega
    simpa [h1, hlo, hhi] using he
  · have h1 : ¬ (L_p1 ≤ a ∧ a < L_p2) := by unfold L_p1 L_p2 L_p3 at *; omega
    have h2 : ¬ (L_p2 ≤ a ∧ a < L_p3) := by unfold L_p2 L_p3 at *; omega
    simpa [h1, h2] using he

/-! ### `VerdictRegsWf`, `MemSub`, and generic pass-step properties -/

def VerdictRegsWf (s : St) : Prop :=
  ∀ r ∈ ([19, 20, 21] : List Nat), ∃ u < AArch64Env.W, s.reg r = some u

def MemSub (m1 m2 : Mem) : Prop :=
  ∀ a b, m1.rd a = some b → m2.rd a = some b

private theorem rdN_memSub {m1 m2 : Mem} (hsub : MemSub m1 m2) :
    ∀ a n bs, m1.rdN a n = some bs → m2.rdN a n = some bs
  | _, 0, _, h => h
  | a, n + 1, bs, h => by
    simp only [Mem.rdN, bind, Option.bind, pure] at h ⊢
    cases h1 : m1.rd a with
    | none => simp [h1] at h
    | some b =>
      simp only [h1] at h
      cases h2 : m1.rdN (a + 1) n with
      | none => simp [h2] at h
      | some rest =>
        simp only [h2] at h
        rw [hsub a b h1, rdN_memSub hsub (a + 1) n rest h2]; exact h

theorem rdMem_memSub {m1 m2 : Mem} (hsub : MemSub m1 m2) {a : Nat} {sz : MemSize} {v : Nat}
    (h : rdMem m1 a sz = some v) : rdMem m2 a sz = some v := by
  unfold rdMem at h ⊢
  cases hn : m1.rdN a sz.bytes with
  | none => simp [hn] at h
  | some bs => rw [hn] at h; rw [rdN_memSub hsub a sz.bytes bs hn]; exact h

private theorem reg_set_ne {s : St} {i j : Nat} (hne : i ≠ j) (v : Option Nat) :
    (s.setReg i v).reg j = s.reg j := by
  simp [St.reg, St.setReg, List.getElem?_set_ne hne]

private theorem lift_ne_exit (o : Option St) (c : Nat) : lift o ≠ .exit c := by
  cases o <;> simp [lift]

private theorem wrZR_withMem' (regs : List (Option Nat)) (sp : Option Nat) (pc : Nat) (fl : Flags) (m1 m2 : Mem) (k : Kern)
    (w rd val : Nat) :
    (St.mk regs sp pc fl m2 k).wrZR w rd val =
    { (St.mk regs sp pc fl m1 k).wrZR w rd val with mem := m2 } := by
  unfold St.wrZR; split <;> rfl

private theorem wrSP_withMem' (regs : List (Option Nat)) (sp : Option Nat) (pc : Nat) (fl : Flags) (m1 m2 : Mem) (k : Kern)
    (w rd val : Nat) :
    (St.mk regs sp pc fl m2 k).wrSP w rd val =
    { (St.mk regs sp pc fl m1 k).wrSP w rd val with mem := m2 } := by
  unfold St.wrSP; split <;> rfl

theorem verdictWf_wrZR (regs : List (Option Nat)) (sp : Option Nat) (pc : Nat) (fl : Flags) (mem : Mem) (k : Kern)
    {s : St} (hwf : VerdictRegsWf s) (hl : s.regs.length = 31) (hregs : regs = s.regs)
    {w : Nat} (hw : wOk w = true) (rd v : Nat) :
    VerdictRegsWf ((St.mk regs sp pc fl mem k).wrZR w rd v) := by
  have hw' : w = 32 ∨ w = 64 := by unfold wOk at hw; simp only [Bool.or_eq_true, beq_iff_eq] at hw; exact hw
  intro r hr
  unfold St.wrZR; split
  · obtain ⟨u, hu, hru⟩ := hwf r hr
    exact ⟨u, hu, by simp only [St.reg, hregs]; exact hru⟩
  · by_cases heq : rd = r
    · subst heq
      have hlt : rd < 31 := by
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
        rcases hr with rfl | rfl | rfl <;> decide
      have hpow : pow2 w ≤ AArch64Env.W := by rcases hw' with rfl | rfl <;> decide
      refine ⟨v % pow2 w, Nat.lt_of_lt_of_le (Nat.mod_lt v (by rcases hw' with rfl | rfl <;> decide)) hpow, ?_⟩
      simp [St.reg, St.setReg, hregs, hl, hlt]
    · rw [reg_set_ne heq]; obtain ⟨u, hu, hru⟩ := hwf r hr
      exact ⟨u, hu, by simp only [St.reg, hregs]; exact hru⟩

theorem verdictWf_wrSP (regs : List (Option Nat)) (sp : Option Nat) (pc : Nat) (fl : Flags) (mem : Mem) (k : Kern)
    {s : St} (hwf : VerdictRegsWf s) (hl : s.regs.length = 31) (hregs : regs = s.regs)
    {w : Nat} (hw : wOk w = true) (rd v : Nat) :
    VerdictRegsWf ((St.mk regs sp pc fl mem k).wrSP w rd v) := by
  have hw' : w = 32 ∨ w = 64 := by unfold wOk at hw; simp only [Bool.or_eq_true, beq_iff_eq] at hw; exact hw
  intro r hr
  unfold St.wrSP; split
  · obtain ⟨u, hu, hru⟩ := hwf r hr
    exact ⟨u, hu, by simp only [St.reg, hregs]; exact hru⟩
  · by_cases heq : rd = r
    · subst heq
      have hlt : rd < 31 := by
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
        rcases hr with rfl | rfl | rfl <;> decide
      have hpow : pow2 w ≤ AArch64Env.W := by rcases hw' with rfl | rfl <;> decide
      refine ⟨v % pow2 w, Nat.lt_of_lt_of_le (Nat.mod_lt v (by rcases hw' with rfl | rfl <;> decide)) hpow, ?_⟩
      simp [St.reg, St.setReg, hregs, hl, hlt]
    · rw [reg_set_ne heq]; obtain ⟨u, hu, hru⟩ := hwf r hr
      exact ⟨u, hu, by simp only [St.reg, hregs]; exact hru⟩

theorem sz_width_ok (sz : MemSize) : wOk sz.width = true := by
  cases sz <;> decide

private theorem setReg_wrZR_comm' (regs : List (Option Nat)) (sp : Option Nat) (pc : Nat) (fl : Flags) (mem : Mem) (k : Kern)
    (w rd val r : Nat) (v : Option Nat) (hne : rd ≠ r) :
    (St.mk (regs.set r v) sp pc fl mem k).wrZR w rd val =
    ((St.mk regs sp pc fl mem k).wrZR w rd val).setReg r v := by
  unfold St.wrZR; split
  · rfl
  · simp only [St.setReg, List.set_comm _ _ hne]

private theorem setReg_wrSP_comm' (regs : List (Option Nat)) (sp : Option Nat) (pc : Nat) (fl : Flags) (mem : Mem) (k : Kern)
    (w rd val r : Nat) (v : Option Nat) (hne : rd ≠ r) :
    (St.mk (regs.set r v) sp pc fl mem k).wrSP w rd val =
    ((St.mk regs sp pc fl mem k).wrSP w rd val).setReg r v := by
  unfold St.wrSP; split
  · rfl
  · simp only [St.setReg, List.set_comm _ _ hne]

private theorem setReg_rdZR_ne (t : St) (w rn r : Nat) (v : Option Nat) (hne : rn ≠ r) :
    (t.setReg r v).rdZR w rn = t.rdZR w rn := by
  unfold St.rdZR; split
  · rfl
  · simp [reg_set_ne hne.symm]

private theorem setReg_rdSP_ne (t : St) (w rn r : Nat) (v : Option Nat) (hne : rn ≠ r) :
    (t.setReg r v).rdSP w rn = t.rdSP w rn := by
  unfold St.rdSP; split
  · rfl
  · simp [reg_set_ne hne.symm]

private theorem wrZR_len_eq' (t : St) (w rd val : Nat) : (t.wrZR w rd val).regs.length = t.regs.length := by
  unfold St.wrZR; split <;> simp [St.setReg]

private theorem wrSP_len_eq' (t : St) (w rd val : Nat) : (t.wrSP w rd val).regs.length = t.regs.length := by
  unfold St.wrSP; split <;> simp [St.setReg]

private theorem wrZR_pc_eq' (t : St) (w rd val : Nat) : (t.wrZR w rd val).pc = t.pc := by
  unfold St.wrZR; split <;> rfl

private theorem wrSP_pc_eq' (t : St) (w rd val : Nat) : (t.wrSP w rd val).pc = t.pc := by
  unfold St.wrSP; split <;> rfl

private theorem wrZR_mem_eq' (t : St) (w rd val : Nat) : (t.wrZR w rd val).mem = t.mem := by
  unfold St.wrZR; split <;> rfl

private theorem wrSP_mem_eq' (t : St) (w rd val : Nat) : (t.wrSP w rd val).mem = t.mem := by
  unfold St.wrSP; split <;> rfl

private theorem wrZR_reg_ne' (t : St) (w rd val r : Nat) (hne : rd ≠ r) :
    (t.wrZR w rd val).reg r = t.reg r := by
  unfold St.wrZR; split
  · rfl
  · exact reg_set_ne hne _

private theorem wrSP_reg_ne' (t : St) (w rd val r : Nat) (hne : rd ≠ r) :
    (t.wrSP w rd val).reg r = t.reg r := by
  unfold St.wrSP; split
  · rfl
  · exact reg_set_ne hne _

private theorem ne_of_not_otherV {R x r : Nat} (hnot : (!isOtherV R x) = true)
    (hr : r ∈ ([19, 20, 21] : List Nat)) (hne : r ≠ R) : x ≠ r := by
  intro heq; subst heq
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  have : isOtherV R x = true := by
    unfold isOtherV; simp only [Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq, bne_iff_ne]
    exact ⟨by rcases hr with rfl | rfl | rfl <;> simp, hne⟩
  simp [this] at hnot

/-- Predicate for a valid PC inside pass `[lo, hi)` or at its exit `nxt`. -/
def inPassOrNxt (lo hi nxt pc : Nat) : Prop :=
  (lo ≤ pc ∧ pc < hi ∧ (pc - lo) % 4 = 0) ∨ pc = nxt

theorem exec_pass_props {lo hi nxt R : Nat} {ins : Ins} {s : St}
    (hsafe : insPassSafe lo hi nxt R s.pc ins = true) (hlo : lo ≤ s.pc) (_hhi : s.pc < hi)
    (hhiW : hi < AArch64Env.W) (hal : (s.pc - lo) % 4 = 0)
    (hl : s.regs.length = 31) (hwf : VerdictRegsWf s) :
    (∀ c, exec ins 4 s ≠ .exit c) ∧
    (∀ s', exec ins 4 s = .ok s' →
      s'.mem = s.mem ∧ s'.regs.length = 31 ∧
      ((lo < s'.pc ∧ s'.pc < hi ∧ (s'.pc - lo) % 4 = 0) ∨ s'.pc = nxt) ∧
      VerdictRegsWf s' ∧
      (∀ r ∈ ([19, 20, 21] : List Nat), r ≠ R → s'.reg r = s.reg r)) ∧
    (∀ r v, r ∈ ([19, 20, 21] : List Nat) → r ≠ R →
      exec ins 4 (s.setReg r v) = match exec ins 4 s with | .ok s' => .ok (s'.setReg r v) | o => o) ∧
    (∀ m2, MemSub s.mem m2 → ∀ s', exec ins 4 s = .ok s' →
      exec ins 4 { s with mem := m2 } = .ok { s' with mem := m2 }) := by
  unfold insPassSafe at hsafe
  simp only [Bool.and_eq_true] at hsafe
  have hsafe' := hsafe.1
  have hfall_ok : s.pc + 4 < hi →
      (lo < (s.pc + 4) % AArch64Env.W ∧ (s.pc + 4) % AArch64Env.W < hi ∧
        ((s.pc + 4) % AArch64Env.W - lo) % 4 = 0) ∨ (s.pc + 4) % AArch64Env.W = nxt := by
    intro hlt4; left; rw [Nat.mod_eq_of_lt (by omega : s.pc + 4 < AArch64Env.W)]; omega
  cases ins with
  | movWide kind w rd imm16 hw =>
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hsafe'
    obtain ⟨⟨hlt4, hwk⟩, hrd⟩ := hsafe'
    refine ⟨?_, ?_, ?_, ?_⟩
    · intro c hx; simp only [exec] at hx
      rcases kind with _ | _ | _ | _ | _ <;> try cases hx
      exact lift_ne_exit _ c hx
    · intro s' hx; simp only [exec] at hx
      rcases kind with _ | _ | _ | _ | _ <;> try cases hx
      · exact ⟨wrZR_mem_eq' _ _ _ _, by rw [wrZR_len_eq']; exact hl, by rw [wrZR_pc_eq']; exact hfall_ok hlt4,
          verdictWf_wrZR s.regs s.sp _ s.fl s.mem s.k hwf hl rfl hwk _ _,
          fun r hr hne => wrZR_reg_ne' _ _ _ _ r (ne_of_not_otherV hrd hr hne)⟩
      · exact ⟨wrZR_mem_eq' _ _ _ _, by rw [wrZR_len_eq']; exact hl, by rw [wrZR_pc_eq']; exact hfall_ok hlt4,
          verdictWf_wrZR s.regs s.sp _ s.fl s.mem s.k hwf hl rfl hwk _ _,
          fun r hr hne => wrZR_reg_ne' _ _ _ _ r (ne_of_not_otherV hrd hr hne)⟩
      · simp only [lift, bind, Option.bind, pure] at hx
        cases h1 : s.rdZR w rd with
        | none => simp [h1] at hx
        | some old =>
          simp only [h1] at hx; cases hx
          exact ⟨wrZR_mem_eq' _ _ _ _, by rw [wrZR_len_eq']; exact hl, by rw [wrZR_pc_eq']; exact hfall_ok hlt4,
            verdictWf_wrZR s.regs s.sp _ s.fl s.mem s.k hwf hl rfl hwk _ _,
            fun r hr hne => wrZR_reg_ne' _ _ _ _ r (ne_of_not_otherV hrd hr hne)⟩
    · intro r v hr hne
      have hne' := ne_of_not_otherV hrd hr hne
      simp only [exec]
      rcases kind with _ | _ | _ | _ | _ <;> try rfl
      · exact congrArg Out.ok (setReg_wrZR_comm' s.regs s.sp _ s.fl s.mem s.k w rd _ r v hne')
      · exact congrArg Out.ok (setReg_wrZR_comm' s.regs s.sp _ s.fl s.mem s.k w rd _ r v hne')
      · rw [setReg_rdZR_ne s w rd r v hne']
        cases h1 : s.rdZR w rd <;> simp [lift, bind, Option.bind, pure]
        exact setReg_wrZR_comm' s.regs s.sp _ s.fl s.mem s.k w rd _ r v hne'
    · intro m2 _ s' hx; simp only [exec] at hx ⊢
      rcases kind with _ | _ | _ | _ | _ <;> try cases hx
      · exact congrArg Out.ok (wrZR_withMem' s.regs s.sp _ s.fl s.mem m2 s.k w rd _)
      · exact congrArg Out.ok (wrZR_withMem' s.regs s.sp _ s.fl s.mem m2 s.k w rd _)
      · simp only [lift, bind, Option.bind, pure] at hx ⊢
        cases h1 : s.rdZR w rd with
        | none => simp [h1] at hx
        | some old =>
          have h1' : ({ s with mem := m2 }).rdZR w rd = some old := h1
          simp only [h1] at hx; cases hx; simp only [h1']
          exact congrArg Out.ok (wrZR_withMem' s.regs s.sp _ s.fl s.mem m2 s.k w rd _)
  | addSubImm isSub sf w rd rn imm =>
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hsafe'
    obtain ⟨⟨⟨hlt4, hwk⟩, hrd⟩, hrn⟩ := hsafe'
    refine ⟨fun c => lift_ne_exit _ c, ?_, ?_, ?_⟩
    · intro s' hx; simp only [exec, lift, bind, Option.bind, pure] at hx
      cases h1 : s.rdSP w rn with
      | none => simp [h1] at hx
      | some a =>
        simp only [h1] at hx
        cases sf <;> simp only [Bool.false_eq_true, ↓reduceIte] at hx <;> cases hx
        · exact ⟨wrSP_mem_eq' _ _ _ _, by rw [wrSP_len_eq']; exact hl, by rw [wrSP_pc_eq']; exact hfall_ok hlt4,
            verdictWf_wrSP s.regs s.sp _ s.fl s.mem s.k hwf hl rfl hwk _ _,
            fun r hr hne => wrSP_reg_ne' _ _ _ _ r (ne_of_not_otherV hrd hr hne)⟩
        · exact ⟨wrZR_mem_eq' _ _ _ _, by rw [wrZR_len_eq']; exact hl, by rw [wrZR_pc_eq']; exact hfall_ok hlt4,
            verdictWf_wrZR s.regs s.sp _ _ s.mem s.k hwf hl rfl hwk _ _,
            fun r hr hne => wrZR_reg_ne' _ _ _ _ r (ne_of_not_otherV hrd hr hne)⟩
    · intro r v hr hne
      have hne_d := ne_of_not_otherV hrd hr hne
      have hne_n := ne_of_not_otherV hrn hr hne
      simp only [exec, setReg_rdSP_ne s w rn r v hne_n]
      cases h1 : s.rdSP w rn <;> simp [lift, bind, Option.bind, pure]
      cases sf <;> simp only [Bool.false_eq_true, ↓reduceIte]
      · exact congrArg Out.ok (setReg_wrSP_comm' s.regs s.sp _ s.fl s.mem s.k w rd _ r v hne_d)
      · exact congrArg Out.ok (setReg_wrZR_comm' s.regs s.sp _ _ s.mem s.k w rd _ r v hne_d)
    · intro m2 _ s' hx; simp only [exec, lift, bind, Option.bind, pure] at hx ⊢
      cases h1 : s.rdSP w rn with
      | none => simp [h1] at hx
      | some a =>
        have h1' : ({ s with mem := m2 }).rdSP w rn = some a := h1
        simp only [h1] at hx; simp only [h1']
        cases sf <;> simp only [Bool.false_eq_true, ↓reduceIte] at hx ⊢ <;> cases hx
        · exact congrArg Out.ok (wrSP_withMem' s.regs s.sp _ s.fl s.mem m2 s.k w rd _)
        · exact congrArg Out.ok (wrZR_withMem' s.regs s.sp _ _ s.mem m2 s.k w rd _)
  | addSubReg isSub sf w rd rn rm sh amt =>
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hsafe'
    obtain ⟨⟨⟨⟨hlt4, hwk⟩, hrd⟩, hrn⟩, hrm⟩ := hsafe'
    refine ⟨fun c => lift_ne_exit _ c, ?_, ?_, ?_⟩
    · intro s' hx; simp only [exec, lift, bind, Option.bind, pure] at hx
      cases h1 : s.rdZR w rn <;> cases h2 : s.rdZR w rm <;> simp [h1, h2] at hx
      cases sf <;> simp only [Bool.false_eq_true, ↓reduceIte] at hx <;> cases hx <;>
        exact ⟨wrZR_mem_eq' _ _ _ _, by rw [wrZR_len_eq']; exact hl, by rw [wrZR_pc_eq']; exact hfall_ok hlt4,
          verdictWf_wrZR s.regs s.sp _ _ s.mem s.k hwf hl rfl hwk _ _,
          fun r hr hne => wrZR_reg_ne' _ _ _ _ r (ne_of_not_otherV hrd hr hne)⟩
    · intro r v hr hne
      simp only [exec, setReg_rdZR_ne s w rn r v (ne_of_not_otherV hrn hr hne),
        setReg_rdZR_ne s w rm r v (ne_of_not_otherV hrm hr hne)]
      cases h1 : s.rdZR w rn <;> cases h2 : s.rdZR w rm <;> simp [lift, bind, Option.bind, pure]
      cases sf <;> simp only [Bool.false_eq_true, ↓reduceIte]
      · exact setReg_wrZR_comm' s.regs s.sp _ s.fl s.mem s.k w rd _ r v (ne_of_not_otherV hrd hr hne)
      · exact setReg_wrZR_comm' s.regs s.sp _ _ s.mem s.k w rd _ r v (ne_of_not_otherV hrd hr hne)
    · intro m2 _ s' hx; simp only [exec, lift, bind, Option.bind, pure] at hx ⊢
      cases h1 : s.rdZR w rn <;> cases h2 : s.rdZR w rm <;> simp [h1, h2] at hx
      have h1' : ({ s with mem := m2 }).rdZR w rn = some _ := h1
      have h2' : ({ s with mem := m2 }).rdZR w rm = some _ := h2
      simp only [h1', h2']
      cases sf <;> simp only [Bool.false_eq_true, ↓reduceIte] at hx ⊢ <;> cases hx <;>
        exact congrArg Out.ok (wrZR_withMem' s.regs s.sp _ _ s.mem m2 s.k w rd _)
  | logicReg op inv w rd rn rm sh amt =>
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hsafe'
    obtain ⟨⟨⟨⟨hlt4, hwk⟩, hrd⟩, hrn⟩, hrm⟩ := hsafe'
    refine ⟨fun c => lift_ne_exit _ c, ?_, ?_, ?_⟩
    · intro s' hx; simp only [exec, lift, bind, Option.bind, pure] at hx
      cases h1 : s.rdZR w rn <;> cases h2 : s.rdZR w rm <;> simp [h1, h2] at hx
      cases op <;> simp only at hx <;> cases hx <;>
        exact ⟨wrZR_mem_eq' _ _ _ _, by rw [wrZR_len_eq']; exact hl, by rw [wrZR_pc_eq']; exact hfall_ok hlt4,
          verdictWf_wrZR s.regs s.sp _ _ s.mem s.k hwf hl rfl hwk _ _,
          fun r hr hne => wrZR_reg_ne' _ _ _ _ r (ne_of_not_otherV hrd hr hne)⟩
    · intro r v hr hne
      simp only [exec, setReg_rdZR_ne s w rn r v (ne_of_not_otherV hrn hr hne),
        setReg_rdZR_ne s w rm r v (ne_of_not_otherV hrm hr hne)]
      cases h1 : s.rdZR w rn <;> cases h2 : s.rdZR w rm <;> simp [lift, bind, Option.bind, pure]
      cases op <;> simp only [↓reduceIte] <;>
        exact congrArg Out.ok (setReg_wrZR_comm' s.regs s.sp _ _ s.mem s.k w rd _ r v (ne_of_not_otherV hrd hr hne))
    · intro m2 _ s' hx; simp only [exec, lift, bind, Option.bind, pure] at hx ⊢
      cases h1 : s.rdZR w rn <;> cases h2 : s.rdZR w rm <;> simp [h1, h2] at hx
      have h1' : ({ s with mem := m2 }).rdZR w rn = some _ := h1
      have h2' : ({ s with mem := m2 }).rdZR w rm = some _ := h2
      simp only [h1', h2']
      cases op <;> simp only at hx ⊢ <;> cases hx <;>
        exact congrArg Out.ok (wrZR_withMem' s.regs s.sp _ _ s.mem m2 s.k w rd _)
  | adr isPage rd imm =>
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hsafe'
    obtain ⟨hlt4, hrd⟩ := hsafe'
    refine ⟨fun c hx => by simp [exec] at hx, ?_, ?_, ?_⟩
    · intro s' hx; simp only [exec] at hx; cases hx
      exact ⟨wrZR_mem_eq' _ _ _ _, by rw [wrZR_len_eq']; exact hl, by rw [wrZR_pc_eq']; exact hfall_ok hlt4,
        verdictWf_wrZR s.regs s.sp _ s.fl s.mem s.k (w := 64) hwf hl rfl (by decide) _ _,
        fun r hr hne => wrZR_reg_ne' _ _ _ _ r (ne_of_not_otherV hrd hr hne)⟩
    · intro r v hr hne; simp only [exec]
      exact congrArg Out.ok (setReg_wrZR_comm' s.regs s.sp _ s.fl s.mem s.k 64 rd _ r v (ne_of_not_otherV hrd hr hne))
    · intro m2 _ s' hx; simp only [exec] at hx ⊢; cases hx
      exact congrArg Out.ok (wrZR_withMem' s.regs s.sp _ s.fl s.mem m2 s.k 64 rd _)
  | ldr sz rt rn off | ldur sz rt rn off =>
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hsafe'
    obtain ⟨⟨hlt4, hrt⟩, hrn⟩ := hsafe'
    refine ⟨fun c => lift_ne_exit _ c, ?_, ?_, ?_⟩
    · intro s' hx; simp only [exec, lift, bind, Option.bind, pure] at hx
      cases h1 : s.rdSP 64 rn with
      | none => simp [h1] at hx
      | some base =>
        simp only [h1] at hx
        by_cases hsp : (rn == 31 && base % 16 != 0) = true
        · simp [hsp] at hx
        · simp only [if_neg hsp] at hx
          cases h2 : rdMem s.mem ((base + off) % AArch64Env.W) sz with
          | none => simp [h2] at hx
          | some v =>
            simp only [h2] at hx; cases hx
            exact ⟨wrZR_mem_eq' _ _ _ _, by rw [wrZR_len_eq']; exact hl, by rw [wrZR_pc_eq']; exact hfall_ok hlt4,
              verdictWf_wrZR s.regs s.sp _ s.fl s.mem s.k hwf hl rfl (sz_width_ok sz) _ _,
              fun r hr hne => wrZR_reg_ne' _ _ _ _ r (ne_of_not_otherV hrt hr hne)⟩
    · intro r v hr hne
      simp only [exec, setReg_rdSP_ne s 64 rn r v (ne_of_not_otherV hrn hr hne)]
      cases h1 : s.rdSP 64 rn with
      | none => simp [lift, bind, Option.bind]
      | some base =>
        simp only [lift, bind, Option.bind, pure, St.setReg]
        by_cases hsp : (rn == 31 && base % 16 != 0) = true
        · simp [hsp]
        · simp only [if_neg hsp]
          cases h2 : rdMem s.mem ((base + off) % AArch64Env.W) sz with
          | none => rfl
          | some val =>
            exact congrArg Out.ok (setReg_wrZR_comm' s.regs s.sp _ s.fl s.mem s.k sz.width rt val r v (ne_of_not_otherV hrt hr hne))
    · intro m2 hsub s' hx; simp only [exec, lift, bind, Option.bind, pure] at hx ⊢
      cases h1 : s.rdSP 64 rn with
      | none => simp [h1] at hx
      | some base =>
        have h1' : ({ s with mem := m2 }).rdSP 64 rn = some base := h1
        simp only [h1] at hx; simp only [h1']
        by_cases hsp : (rn == 31 && base % 16 != 0) = true
        · simp [hsp] at hx
        · simp only [if_neg hsp] at hx ⊢
          cases h2 : rdMem s.mem ((base + off) % AArch64Env.W) sz with
          | none => simp [h2] at hx
          | some v =>
            simp only [h2] at hx; cases hx
            rw [rdMem_memSub hsub h2]
            exact congrArg Out.ok (wrZR_withMem' s.regs s.sp _ s.fl s.mem m2 s.k sz.width rt v)
  | b link off =>
    cases link
    · simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hsafe'
      refine ⟨fun c hx => by simp [exec] at hx, ?_, ?_, ?_⟩
      · intro s' hx; simp only [exec] at hx; cases hx
        have hpc' : (lo < (s.pc + off) % AArch64Env.W ∧ (s.pc + off) % AArch64Env.W < hi ∧
            ((s.pc + off) % AArch64Env.W - lo) % 4 = 0) ∨ (s.pc + off) % AArch64Env.W = nxt := by
          rcases hsafe' with ⟨⟨h1, h2⟩, h3⟩ | h4
          · exact Or.inl ⟨by omega, h2, h3⟩
          · exact Or.inr h4
        exact ⟨rfl, hl, hpc', hwf, fun _ _ _ => rfl⟩
      · intro r v _ _; rfl
      · intro m2 _ s' hx; simp only [exec] at hx ⊢; cases hx; rfl
    · cases hsafe'
  | bcond cc off =>
    simp only [Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq] at hsafe'
    obtain ⟨hlt4, htgt⟩ := hsafe'
    refine ⟨fun c => lift_ne_exit _ c, ?_, ?_, ?_⟩
    · intro s' hx; simp only [exec, lift, bind, Option.bind, pure] at hx
      cases hc : cond s.fl cc with
      | none => simp [hc] at hx
      | some c =>
        simp only [hc] at hx
        cases c <;> simp only [Bool.false_eq_true, ↓reduceIte] at hx <;> cases hx
        · exact ⟨rfl, hl, hfall_ok hlt4, hwf, fun _ _ _ => rfl⟩
        · have hpc' : (lo < (s.pc + off) % AArch64Env.W ∧ (s.pc + off) % AArch64Env.W < hi ∧
              ((s.pc + off) % AArch64Env.W - lo) % 4 = 0) ∨ (s.pc + off) % AArch64Env.W = nxt := by
            rcases htgt with ⟨⟨h1, h2⟩, h3⟩ | h4
            · exact Or.inl ⟨by omega, h2, h3⟩
            · exact Or.inr h4
          exact ⟨rfl, hl, hpc', hwf, fun _ _ _ => rfl⟩
    · intro r v _ _; simp only [exec, St.setReg]
      cases hc : cond s.fl cc with
      | none => simp [lift, bind, Option.bind]
      | some c => simp only [lift, bind, Option.bind, pure]; cases c <;> rfl
    · intro m2 _ s' hx; simp only [exec, lift, bind, Option.bind, pure] at hx ⊢
      cases hc : cond s.fl cc with
      | none => simp [hc] at hx
      | some c =>
        simp only [hc] at hx ⊢; cases c <;> simp only [Bool.false_eq_true, ↓reduceIte] at hx ⊢ <;> cases hx <;> rfl
  | cbz nz w rt off =>
    simp only [Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq] at hsafe'
    obtain ⟨⟨⟨hlt4, _⟩, hrt⟩, htgt⟩ := hsafe'
    refine ⟨fun c => lift_ne_exit _ c, ?_, ?_, ?_⟩
    · intro s' hx; simp only [exec, lift, bind, Option.bind, pure] at hx
      cases hr : s.rdZR w rt with
      | none => simp [hr] at hx
      | some v =>
        simp only [hr] at hx
        cases htk : (if nz then v != 0 else v == 0) <;>
          simp only [htk, Bool.false_eq_true, ↓reduceIte] at hx <;> cases hx
        · exact ⟨rfl, hl, hfall_ok hlt4, hwf, fun _ _ _ => rfl⟩
        · have hpc' : (lo < (s.pc + off) % AArch64Env.W ∧ (s.pc + off) % AArch64Env.W < hi ∧
              ((s.pc + off) % AArch64Env.W - lo) % 4 = 0) ∨ (s.pc + off) % AArch64Env.W = nxt := by
            rcases htgt with ⟨⟨h1, h2⟩, h3⟩ | h4
            · exact Or.inl ⟨by omega, h2, h3⟩
            · exact Or.inr h4
          exact ⟨rfl, hl, hpc', hwf, fun _ _ _ => rfl⟩
    · intro r v hr hne
      simp only [exec, setReg_rdZR_ne s w rt r v (ne_of_not_otherV hrt hr hne)]
      cases h1 : s.rdZR w rt with
      | none => simp [lift, bind, Option.bind]
      | some val =>
        simp only [lift, bind, Option.bind, pure]
        cases htk : (if nz then val != 0 else val == 0) <;>
          simp only [Bool.false_eq_true, ↓reduceIte] <;> rfl
    · intro m2 _ s' hx; simp only [exec, lift, bind, Option.bind, pure] at hx ⊢
      cases h1 : s.rdZR w rt with
      | none => simp [h1] at hx
      | some v =>
        have h1' : ({ s with mem := m2 }).rdZR w rt = some v := h1
        simp only [h1] at hx; simp only [h1']
        cases htk : (if nz then v != 0 else v == 0) <;>
          simp only [htk, Bool.false_eq_true, ↓reduceIte] at hx ⊢ <;> cases hx <;> rfl
  | _ => cases hsafe'

/-! ### Pass escape, `MemSub` replay, unread-register replay, and strict progression -/

theorem pass_clean_escape {lo hi nxt R : Nat} (hpass : PassSpec lo hi nxt R) :
    ∀ n (s : St), inPassOrNxt lo hi nxt s.pc →
      s.regs.length = 31 → VerdictRegsWf s →
      run im n (.ok s) = .exit 0 →
      ∃ t ≤ n, ∃ s', run im t (.ok s) = .ok s' ∧ s'.pc = nxt ∧ s'.mem = s.mem ∧
        s'.regs.length = 31 ∧ VerdictRegsWf s' ∧
        (∀ r ∈ ([19, 20, 21] : List Nat), r ≠ R → s'.reg r = s.reg r) ∧
        run im (n - t) (.ok s') = .exit 0
  | 0, _, _, _, _, h0 => by simp [run] at h0
  | n + 1, s, hin, hl, hwf, h0 => by
    by_cases heq : s.pc = nxt
    · exact ⟨0, Nat.zero_le _, s, rfl, heq, rfl, hl, hwf, fun _ _ _ => rfl, h0⟩
    · rcases hin with ⟨hlo, hlt, hal⟩ | heq'
      · have hhiW : hi < AArch64Env.W := by
          rcases hpass with ⟨rfl, rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl, rfl⟩ <;> decide
        simp only [run] at h0
        cases him : im s.pc with
        | none => simp [step, him, run_stuck] at h0
        | some p =>
          obtain ⟨ins, len⟩ := p
          obtain ⟨rfl, hsafe⟩ := im_in_pass_safe hpass hlo hlt him
          have ⟨hnoex, hprops, _, _⟩ := exec_pass_props hsafe hlo hlt hhiW hal hl hwf
          have hst_eq : step im s = exec ins 4 s := by simp [step, him]
          rw [hst_eq] at h0
          cases hx : exec ins 4 s with
          | exit c => exact absurd hx (hnoex c)
          | stuck => rw [hx, run_stuck] at h0; cases h0
          | ok s1 =>
            rw [hx] at h0
            obtain ⟨hm1, hl1, hin1, hwf1, hreg1⟩ := hprops s1 hx
            have hin1' : inPassOrNxt lo hi nxt s1.pc := by
              rcases hin1 with ⟨h1, h2, h3⟩ | h4
              · exact Or.inl ⟨by omega, h2, h3⟩
              · exact Or.inr h4
            obtain ⟨t, ht, s', hrt, hpc', hm', hl', hwf', hreg', hend⟩ :=
              pass_clean_escape hpass n s1 hin1' hl1 hwf1 h0
            refine ⟨t + 1, by omega, s', by simp [run, hst_eq, hx, hrt], hpc', hm'.trans hm1, hl', hwf',
              fun r hr hne => (hreg' r hr hne).trans (hreg1 r hr hne), ?_⟩
            rwa [show n + 1 - (t + 1) = n - t by omega]
      · exact absurd heq' heq

theorem pass_run_memSub {lo hi nxt R : Nat} (hpass : PassSpec lo hi nxt R) {m1 m2 : Mem} (hsub : MemSub m1 m2) :
    ∀ k (s s_end : St), s.mem = m1 → s.regs.length = 31 → VerdictRegsWf s →
      run im k (.ok s) = .ok s_end →
      (∀ t < k, ∃ u, run im t (.ok s) = .ok u ∧ lo ≤ u.pc ∧ u.pc < hi ∧ (u.pc - lo) % 4 = 0 ∧
        u.regs.length = 31 ∧ VerdictRegsWf u) →
      s_end.mem = m1 ∧ run im k (.ok { s with mem := m2 }) = .ok { s_end with mem := m2 }
  | 0, s, s_end, hm, _, _, hrun, _ => by
    simp [run] at hrun; subst hrun; exact ⟨hm, rfl⟩
  | k + 1, s, s_end, hm, hl, hwf, hrun, hstay => by
    have ⟨u0, hu0, hlo, hlt, hal, _, _⟩ := hstay 0 (Nat.zero_lt_succ k)
    simp [run] at hu0; subst hu0
    have hhiW : hi < AArch64Env.W := by
      rcases hpass with ⟨rfl, rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl, rfl⟩ <;> decide
    simp only [run] at hrun ⊢
    cases him : im s.pc with
    | none => simp [step, him, run_stuck] at hrun
    | some p =>
      obtain ⟨ins, len⟩ := p
      obtain ⟨rfl, hsafe⟩ := im_in_pass_safe hpass hlo hlt him
      have ⟨_, hprops, _, hmono⟩ := exec_pass_props hsafe hlo hlt hhiW hal hl hwf
      have hst1 : step im s = exec ins 4 s := by simp [step, him]
      have hst2 : step im { s with mem := m2 } = exec ins 4 { s with mem := m2 } := by simp [step, him]
      rw [hst1] at hrun; rw [hst2]
      cases hx : exec ins 4 s with
      | exit c => rw [hx, run_exit] at hrun; cases hrun
      | stuck => rw [hx, run_stuck] at hrun; cases hrun
      | ok s1 =>
        rw [hx] at hrun
        obtain ⟨hm1, hl1, _, hwf1, _⟩ := hprops s1 hx
        rw [hmono m2 (hm ▸ hsub) s1 hx]
        have hstay' : ∀ t < k, ∃ u, run im t (.ok s1) = .ok u ∧ lo ≤ u.pc ∧ u.pc < hi ∧
            (u.pc - lo) % 4 = 0 ∧ u.regs.length = 31 ∧ VerdictRegsWf u := by
          intro t ht
          have h := hstay (t + 1) (by omega)
          simpa [run, hst1, hx] using h
        exact pass_run_memSub hpass hsub k s1 s_end (hm1.trans hm) hl1 hwf1 hrun hstay'

theorem pass_run_unread_reg {lo hi nxt R r : Nat} (hpass : PassSpec lo hi nxt R)
    (hr : r ∈ ([19, 20, 21] : List Nat)) (hne : r ≠ R) (v : Option Nat) :
    ∀ k (s s_end : St), s.regs.length = 31 → VerdictRegsWf s →
      run im k (.ok s) = .ok s_end →
      (∀ t < k, ∃ u, run im t (.ok s) = .ok u ∧ lo ≤ u.pc ∧ u.pc < hi ∧ (u.pc - lo) % 4 = 0 ∧
        u.regs.length = 31 ∧ VerdictRegsWf u) →
      run im k (.ok (s.setReg r v)) = .ok (s_end.setReg r v)
  | 0, s, s_end, _, _, hrun, _ => by simp [run] at hrun; subst hrun; rfl
  | k + 1, s, s_end, hl, hwf, hrun, hstay => by
    have ⟨u0, hu0, hlo, hlt, hal, _, _⟩ := hstay 0 (Nat.zero_lt_succ k)
    simp [run] at hu0; subst hu0
    have hhiW : hi < AArch64Env.W := by
      rcases hpass with ⟨rfl, rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl, rfl⟩ <;> decide
    simp only [run] at hrun ⊢
    cases him : im s.pc with
    | none => simp [step, him, run_stuck] at hrun
    | some p =>
      obtain ⟨ins, len⟩ := p
      obtain ⟨rfl, hsafe⟩ := im_in_pass_safe hpass hlo hlt him
      have ⟨_, hprops, hcongr, _⟩ := exec_pass_props hsafe hlo hlt hhiW hal hl hwf
      have hst1 : step im s = exec ins 4 s := by simp [step, him]
      have hst2 : step im (s.setReg r v) = exec ins 4 (s.setReg r v) := by simp [step, him]
      rw [hst1] at hrun; rw [hst2, hcongr r v hr hne]
      cases hx : exec ins 4 s with
      | exit c => rw [hx, run_exit] at hrun; cases hrun
      | stuck => rw [hx, run_stuck] at hrun; cases hrun
      | ok s1 =>
        rw [hx] at hrun
        obtain ⟨_, hl1, _, hwf1, _⟩ := hprops s1 hx
        have hstay' : ∀ t < k, ∃ u, run im t (.ok s1) = .ok u ∧ lo ≤ u.pc ∧ u.pc < hi ∧
            (u.pc - lo) % 4 = 0 ∧ u.regs.length = 31 ∧ VerdictRegsWf u := by
          intro t ht; simpa [run, hst1, hx] using hstay (t + 1) (by omega)
        exact pass_run_unread_reg hpass hr hne v k s1 s_end hl1 hwf1 hrun hstay'

private def inCoreCode (a : Nat) : Prop :=
  (L_p1 ≤ a ∧ a < L_p3_end) ∨ (L_vote ≤ a ∧ a < L_core_end)

private theorem core_step_gt {b : Nat} (hb : b = L_p2 ∨ b = L_p3 ∨ b = L_vote)
    {s s' : St} (hge : b ≤ s.pc) (hin : inCoreCode s.pc) (hst : step im s = .ok s') :
    b < s'.pc ∧ (inCoreCode s'.pc ∨ im s'.pc = none) := by
  cases him : im s.pc with
  | none => simp [step, him] at hst
  | some p =>
    obtain ⟨ins, len⟩ := p
    have hx : exec ins len s = .ok s' := by simpa [step, him] using hst
    have haud := corePassAudit_ok
    unfold corePassAuditBool at haud; simp only [Bool.and_eq_true] at haud
    rcases hin with ⟨hp1, hp3e⟩ | ⟨hv, hce⟩
    · obtain ⟨rfl, hmem⟩ := im_in_corePassTable hp1 hp3e him
      have he := List.all_eq_true.1 haud.1.2 (s.pc, ins, 4) hmem
      dsimp only at he
      have hnot1 : ¬ (L_p1 ≤ s.pc ∧ s.pc < L_p2) := by
        rcases hb with rfl | rfl | rfl <;> (dsimp only [L_p1, L_p2, L_p3, L_vote] at *; omega)
      have hmod4 : (s.pc + 4) % AArch64Env.W = s.pc + 4 := by
        dsimp only [L_p3_end, AArch64Env.W] at *; omega
      by_cases h2 : s.pc < L_p3
      · have hge2 : L_p2 ≤ s.pc := by
          rcases hb with rfl | rfl | rfl <;> (dsimp only [L_p2, L_p3, L_vote] at *; omega)
        have hsafe : insPassSafe L_p2 L_p3 L_p3 20 s.pc ins = true := by
          simpa [hnot1, hge2, h2] using he
        have hdir : ins.isDirect = true := by
          cases ins <;> first | rfl | (simp [insPassSafe] at hsafe)
        rcases exec_pc_direct hdir hx with hf | ⟨off, hbr, hpc'⟩
        · rw [hf, hmod4]
          have hgt : b < s.pc + 4 := by rcases hb with rfl | rfl | rfl <;> (dsimp only [L_p2, L_p3, L_vote] at *; omega)
          exact ⟨hgt, Or.inl (Or.inl ⟨by dsimp only [L_p1, L_p2] at *; omega, by dsimp only [L_p3, L_p3_end] at *; omega⟩)⟩
        · rw [hpc']
          rcases hbr with rfl | rfl | ⟨cc, rfl⟩ | ⟨nz, w, rt, rfl⟩ | ⟨nz, bit, rt, rfl⟩ <;>
            (simp only [insPassSafe, Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq] at hsafe
             first
             | exact Bool.noConfusion hsafe.1
             | (have ht : ((L_p2 + 16 < (s.pc + off) % AArch64Env.W ∧ (s.pc + off) % AArch64Env.W < L_p3) ∧
                    ((s.pc + off) % AArch64Env.W - L_p2) % 4 = 0) ∨ (s.pc + off) % AArch64Env.W = L_p3 := by
                  first | exact hsafe.1.2 | exact hsafe.1
                rcases ht with ⟨⟨h1, h2'⟩, _⟩ | h3
                · exact ⟨by rcases hb with rfl | rfl | rfl <;> (dsimp only [L_p2, L_p3, L_p3_end, L_vote] at *; omega),
                    Or.inl (Or.inl ⟨by dsimp only [L_p1, L_p2] at *; omega, by dsimp only [L_p3, L_p3_end] at *; omega⟩)⟩
                · rw [h3]; exact ⟨by rcases hb with rfl | rfl | rfl <;> (dsimp only [L_p2, L_p3, L_p3_end, L_vote] at *; omega),
                    Or.inl (Or.inl ⟨by decide, by decide⟩)⟩))
      · have hge3 : L_p3 ≤ s.pc := by omega
        have hsafe : insPassSafe L_p3 L_p3_end L_vote 21 s.pc ins = true := by
          have hn2 : ¬ (L_p2 ≤ s.pc ∧ s.pc < L_p3) := by omega
          simpa [hnot1, hn2] using he
        have hdir : ins.isDirect = true := by
          cases ins <;> first | rfl | (simp [insPassSafe] at hsafe)
        rcases exec_pc_direct hdir hx with hf | ⟨off, hbr, hpc'⟩
        · rw [hf, hmod4]
          have hgt : b < s.pc + 4 := by rcases hb with rfl | rfl | rfl <;> (dsimp only [L_p2, L_p3, L_p3_end, L_vote] at *; omega)
          by_cases hlt' : s.pc + 4 < L_p3_end
          · exact ⟨hgt, Or.inl (Or.inl ⟨by dsimp only [L_p1, L_p3] at *; omega, hlt'⟩)⟩
          · have ⟨ha4, _⟩ := im_fixedWidth s.pc ins 4 him
            refine ⟨hgt, Or.inr ?_⟩
            rw [show s.pc + 4 = L_p3_end by dsimp only [L_p3_end] at *; omega]
            exact im_pad1_none (Nat.le_refl _) (by decide)
        · rw [hpc']
          rcases hbr with rfl | rfl | ⟨cc, rfl⟩ | ⟨nz, w, rt, rfl⟩ | ⟨nz, bit, rt, rfl⟩ <;>
            (simp only [insPassSafe, Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq] at hsafe
             first
             | exact Bool.noConfusion hsafe.1
             | (have ht : ((L_p3 + 16 < (s.pc + off) % AArch64Env.W ∧ (s.pc + off) % AArch64Env.W < L_p3_end) ∧
                    ((s.pc + off) % AArch64Env.W - L_p3) % 4 = 0) ∨ (s.pc + off) % AArch64Env.W = L_vote := by
                  first | exact hsafe.1.2 | exact hsafe.1
                rcases ht with ⟨⟨h1, h2'⟩, _⟩ | h3
                · exact ⟨by rcases hb with rfl | rfl | rfl <;> (dsimp only [L_p2, L_p3, L_p3_end, L_vote] at *; omega),
                    Or.inl (Or.inl ⟨by dsimp only [L_p1, L_p3] at *; omega, h2'⟩)⟩
                · rw [h3]; exact ⟨by rcases hb with rfl | rfl | rfl <;> (dsimp only [L_p2, L_p3, L_vote, L_p3_end] at *; omega),
                    Or.inl (Or.inr ⟨Nat.le_refl _, by decide⟩)⟩))
    · obtain ⟨rfl, hmem⟩ := im_in_tailTable hv hce him
      have hsafe := List.all_eq_true.1 haud.2 (s.pc, ins, 4) hmem
      dsimp only at hsafe
      have hdir : ins.isDirect = true := by
        cases ins <;> first | rfl | (simp [insTailGtVote] at hsafe)
      have hmod4 : (s.pc + 4) % AArch64Env.W = s.pc + 4 := by
        dsimp only [L_core_end, AArch64Env.W] at *; omega
      have hgt_b : b < s.pc + 4 := by
        rcases hb with rfl | rfl | rfl <;> (dsimp only [L_p2, L_p3, L_vote] at *; omega)
      rcases exec_pc_direct hdir hx with hf | ⟨off, hbr, hpc'⟩
      · rw [hf, hmod4]; refine ⟨hgt_b, ?_⟩
        by_cases hlt' : s.pc + 4 < L_core_end
        · exact Or.inl (Or.inr ⟨by omega, hlt'⟩)
        · have ⟨ha4, _⟩ := im_fixedWidth s.pc ins 4 him
          right; rw [show s.pc + 4 = L_core_end by dsimp only [L_core_end] at *; omega]
          exact im_pad2_none (Nat.le_refl _) (by decide)
      · rw [hpc']; rcases hbr with rfl | rfl | ⟨cc, rfl⟩ | ⟨nz, w, rt, rfl⟩ | ⟨nz, bit, rt, rfl⟩ <;>
          (simp only [insTailGtVote, Bool.and_eq_true, decide_eq_true_eq] at hsafe
           exact ⟨by rcases hb with rfl | rfl | rfl <;> (dsimp only [L_p2, L_p3, L_vote] at *; omega),
             Or.inl (Or.inr ⟨by omega, hsafe.2⟩)⟩)

theorem run_core_stays_gt {b : Nat} (hb : b = L_p2 ∨ b = L_p3 ∨ b = L_vote) :
    ∀ k (s s' : St), b ≤ s.pc → (inCoreCode s.pc ∨ im s.pc = none) →
      0 < k → run im k (.ok s) = .ok s' → b < s'.pc
  | 0, _, _, _, _, hk, _ => by omega
  | k + 1, s, s', hge, hbd, _, hrun => by
    simp only [run] at hrun
    rcases hbd with hin | hnone
    · cases hst : step im s with
      | exit _ => rw [hst, run_exit] at hrun; cases hrun
      | stuck => rw [hst, run_stuck] at hrun; cases hrun
      | ok s1 =>
        rw [hst] at hrun
        have ⟨hgt1, hbd1⟩ := core_step_gt hb hge hin hst
        cases k with
        | zero => simp [run] at hrun; subst hrun; exact hgt1
        | succ k' => exact run_core_stays_gt hb (k' + 1) s1 s' (by omega) hbd1 (Nat.zero_lt_succ _) hrun
    · simp [step, hnone, run_stuck] at hrun

/-- Any clean pass run from `lo` reaching `nxt` stays strictly inside `[lo, hi)` at every step `< N`. -/
theorem pass_stay_strict {lo hi nxt R : Nat} (hpass : PassSpec lo hi nxt R) :
    ∀ N (s s_end : St), inPassOrNxt lo hi nxt s.pc →
      s.regs.length = 31 → VerdictRegsWf s →
      run im N (.ok s) = .ok s_end → s_end.pc = nxt →
      ∀ t < N, ∃ u, run im t (.ok s) = .ok u ∧ lo ≤ u.pc ∧ u.pc < hi ∧
        (u.pc - lo) % 4 = 0 ∧ u.regs.length = 31 ∧ VerdictRegsWf u
  | 0, _, _, _, _, _, _, _, t, ht => by omega
  | N + 1, s, s_end, hin, hl, hwf, hrun, hend, t, ht => by
    have hb : nxt = L_p2 ∨ nxt = L_p3 ∨ nxt = L_vote := by
      rcases hpass with ⟨_, _, rfl, _⟩ | ⟨_, _, rfl, _⟩ | ⟨_, _, rfl, _⟩ <;> simp
    have hlt_in : lo ≤ s.pc ∧ s.pc < hi ∧ (s.pc - lo) % 4 = 0 := by
      rcases hin with h1 | heq
      · exact h1
      · exfalso
        have hin_core : inCoreCode s.pc := by
          rw [heq]; rcases hb with rfl | rfl | rfl
          · exact Or.inl ⟨by decide, by decide⟩
          · exact Or.inl ⟨by decide, by decide⟩
          · exact Or.inr ⟨Nat.le_refl _, by decide⟩
        have hgt := run_core_stays_gt hb (N + 1) s s_end (by omega) (Or.inl hin_core) (Nat.zero_lt_succ N) hrun
        omega
    obtain ⟨hlo, hlt, hal⟩ := hlt_in
    cases t with
    | zero => exact ⟨s, rfl, hlo, hlt, hal, hl, hwf⟩
    | succ t' =>
      have hhiW : hi < AArch64Env.W := by
        rcases hpass with ⟨_, rfl, _, _⟩ | ⟨_, rfl, _, _⟩ | ⟨_, rfl, _, _⟩ <;> decide
      simp only [run] at hrun ⊢
      cases him : im s.pc with
      | none => simp [step, him, run_stuck] at hrun
      | some p =>
        obtain ⟨ins, len⟩ := p
        obtain ⟨rfl, hsafe⟩ := im_in_pass_safe hpass hlo hlt him
        have ⟨_, hprops, _, _⟩ := exec_pass_props hsafe hlo hlt hhiW hal hl hwf
        have hst : step im s = exec ins 4 s := by simp [step, him]
        rw [hst] at hrun ⊢
        cases hx : exec ins 4 s with
        | exit _ => rw [hx, run_exit] at hrun; cases hrun
        | stuck => rw [hx, run_stuck] at hrun; cases hrun
        | ok s1 =>
          rw [hx] at hrun
          obtain ⟨_, hl1, hin1, hwf1, _⟩ := hprops s1 hx
          have hin1' : inPassOrNxt lo hi nxt s1.pc := by
            rcases hin1 with ⟨h1, h2, h3⟩ | h4
            · exact Or.inl ⟨by omega, h2, h3⟩
            · exact Or.inr h4
          exact pass_stay_strict hpass N s1 s_end hin1' hl1 hwf1 hrun hend t' (by omega)

/-! ### Clean-pass runtime invariants (`PassStepInv` and `clean_pass_never_acc`) -/

private def isPassInsKind : Ins → Prop
  | .movWide _ _ _ _ _ | .addSubImm _ _ _ _ _ _ | .addSubReg _ _ _ _ _ _ _ _
  | .logicReg _ _ _ _ _ _ _ _ | .adr _ _ _ | .ldr _ _ _ _ | .ldur _ _ _ _
  | .b false _ | .bcond _ _ | .cbz _ _ _ _ => True
  | _ => False

private theorem exec_insWriteReg_ne {ins : Ins} {s s' : St} {r : Nat}
    (hx : exec ins 4 s = .ok s')
    (hpass : isPassInsKind ins)
    (hne : match insWriteReg ins with | none => True | some d => d ≠ r) :
    s'.reg r = s.reg r := by
  cases ins with
  | movWide kind w rd imm16 hw =>
    simp only [insWriteReg] at hne; simp only [exec] at hx
    rcases kind with _ | _ | _ | _ | _ <;> try cases hx
    · exact wrZR_reg_ne' _ _ _ _ r hne
    · exact wrZR_reg_ne' _ _ _ _ r hne
    · simp only [lift, bind, Option.bind, pure] at hx
      cases h1 : s.rdZR w rd <;> simp [h1] at hx; cases hx
      exact wrZR_reg_ne' _ _ _ _ r hne
  | addSubImm isSub sf w rd rn imm =>
    simp only [insWriteReg] at hne; simp only [exec, lift, bind, Option.bind, pure] at hx
    cases h1 : s.rdSP w rn <;> simp [h1] at hx
    cases sf <;> simp only [Bool.false_eq_true, ↓reduceIte] at hx <;> cases hx
    · exact wrSP_reg_ne' _ _ _ _ r hne
    · exact wrZR_reg_ne' _ _ _ _ r hne
  | addSubReg isSub sf w rd rn rm sh amt =>
    simp only [insWriteReg] at hne; simp only [exec, lift, bind, Option.bind, pure] at hx
    cases h1 : s.rdZR w rn <;> cases h2 : s.rdZR w rm <;> simp [h1, h2] at hx
    cases sf <;> simp only [Bool.false_eq_true, ↓reduceIte] at hx <;> cases hx <;>
      exact wrZR_reg_ne' _ _ _ _ r hne
  | logicReg op inv w rd rn rm sh amt =>
    simp only [insWriteReg] at hne; simp only [exec, lift, bind, Option.bind, pure] at hx
    cases h1 : s.rdZR w rn <;> cases h2 : s.rdZR w rm <;> simp [h1, h2] at hx
    cases op <;> simp only at hx <;> cases hx <;> exact wrZR_reg_ne' _ _ _ _ r hne
  | adr isPage rd imm =>
    simp only [insWriteReg] at hne; simp only [exec] at hx; cases hx
    exact wrZR_reg_ne' _ _ _ _ r hne
  | ldr sz rt rn off | ldur sz rt rn off =>
    simp only [insWriteReg] at hne; simp only [exec, lift, bind, Option.bind, pure] at hx
    cases h1 : s.rdSP 64 rn with
    | none => simp [h1] at hx
    | some v =>
      simp only [h1] at hx
      by_cases hsp : (rn == 31 && v % 16 != 0) = true
      · simp [hsp] at hx
      · simp only [hsp] at hx; rw [if_neg (by simp)] at hx
        cases h2 : rdMem s.mem ((v + off) % AArch64Env.W) sz with
        | none => simp [h2] at hx
        | some x => simp only [h2] at hx; cases hx; exact wrZR_reg_ne' _ _ _ _ r hne
  | b link off =>
    cases link
    · simp only [exec] at hx; cases hx; rfl
    · exact False.elim hpass
  | bcond cc off =>
    simp only [exec, lift, bind, Option.bind, pure] at hx
    cases hc : cond s.fl cc <;> simp [hc] at hx
    cases ‹Bool› <;> simp at hx <;> cases hx <;> rfl
  | cbz nz w rt off =>
    simp only [exec, lift, bind, Option.bind, pure] at hx
    cases hr : s.rdZR w rt <;> simp [hr] at hx
    split at hx <;> cases hx <;> (split <;> rfl)
  | _ => exact False.elim hpass

private theorem insPassSafe_kind {lo hi nxt R pc : Nat} {ins : Ins}
    (hsafe : insPassSafe lo hi nxt R pc ins = true) :
    isPassInsKind ins := by
  cases ins with
  | b link _ => cases link <;> simp [insPassSafe, isPassInsKind] at hsafe ⊢
  | _ => first | trivial | (simp [insPassSafe] at hsafe)

private theorem insPassSafe_extra {lo hi nxt R pc : Nat} {ins : Ins}
    (hsafe : insPassSafe lo hi nxt R pc ins = true) :
    insStepExtraOk lo R pc ins = true := by
  unfold insPassSafe at hsafe; simp only [Bool.and_eq_true] at hsafe; exact hsafe.2

private theorem insPassSafe_br_gt16 {lo hi nxt R pc off : Nat} {ins : Ins}
    (hsafe : insPassSafe lo hi nxt R pc ins = true)
    (hbr : ins = .b false off ∨ (∃ cc, ins = .bcond cc off) ∨ (∃ nz w rt, ins = .cbz nz w rt off)) :
    lo + 16 < (pc + off) % AArch64Env.W ∨ (pc + off) % AArch64Env.W = nxt := by
  rcases hbr with rfl | ⟨cc, rfl⟩ | ⟨nz, w, rt, rfl⟩ <;>
    (simp only [insPassSafe, Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq] at hsafe
     have ht : ((lo + 16 < (pc + off) % AArch64Env.W ∧ (pc + off) % AArch64Env.W < hi) ∧
         ((pc + off) % AArch64Env.W - lo) % 4 = 0) ∨ (pc + off) % AArch64Env.W = nxt := by
       first | exact hsafe.1.2 | exact hsafe.1
     rcases ht with ⟨⟨h1, _⟩, _⟩ | h2
     · exact Or.inl h1
     · exact Or.inr h2)

/-- Runtime step invariant across a clean pass run on `m_k`: captures exact `x14 = bs.length`
at `lo + 16` (`cmp x14, #16`) and byte bounds `< 256` on `x14`/`x15` inside `inR14`/`inR15`. -/
def PassStepInv (lo D len : Nat) (u : St) : Prop :=
  (u.pc = lo + 4 ∨ u.pc = lo + 8 → u.reg 15 = some D) ∧
  (u.pc = lo + 12 ∨ u.pc = lo + 16 → u.reg 14 = some len) ∧
  (inR14 (u.pc - lo) = true → ∃ b < 256, u.reg 14 = some b) ∧
  (inR15 (u.pc - lo) = true → ∃ b < 256, u.reg 15 = some b)

private theorem rdMem_b1_lt_256 {m : Mem} (hby : ∀ a x, m.rd a = some x → x < 256)
    {a v : Nat} (h : rdMem m a .b1 = some v) : v < 256 := by
  unfold rdMem MemSize.bytes at h
  simp only [Mem.rdN, bind, Option.bind, pure] at h
  cases hr : m.rd a with
  | none => simp [hr] at h
  | some x =>
    simp only [hr, Option.map_some, Option.some.injEq, leVal, Nat.mul_zero, Nat.add_zero] at h
    subst h; exact hby a x hr

private theorem mem_corePassTable_of_chunk {e : Nat × Ins × Nat}
    (h : e ∈ corePassTable_0 ∨ e ∈ corePassTable_10 ∨ e ∈ corePassTable_21 ∨ e ∈ corePassTable_31) :
    e ∈ corePassTable := by
  unfold corePassTable
  rcases h with h | h | h | h <;> simp only [List.mem_append, h, true_or, or_true]

private theorem pass_head_ins {lo D : Nat}
    (h12 : (lo = L_p1 ∧ D = 0x600000) ∨ (lo = L_p2 ∧ D = 0x601000)) :
    (∃ imm, im lo = some (.adr true 15 imm, 4) ∧ (lo / 4096 * 4096 + imm) % AArch64Env.W = D) ∧
    im (lo + 4) = some (.ldr .x8 0 15 0, 4) ∧
    im (lo + 8) = some (.ldr .x8 14 15 8, 4) ∧
    im (lo + 12) = some (.addSubReg false false 64 24 0 14 .lsl 0, 4) := by
  have hmem_im : ∀ e, e ∈ corePassTable → im e.1 = some e.2 :=
    fun e he => table_ok e (List.mem_append.2 (Or.inl (List.mem_append.2 (Or.inl he))))
  rcases h12 with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
  · exact ⟨⟨2093056, hmem_im (4198656, .adr true 15 2093056, 4)
        (mem_corePassTable_of_chunk (Or.inl (List.mem_of_getElem? (i := 0) rfl))), by decide⟩,
      hmem_im (4198660, .ldr .x8 0 15 0, 4)
        (mem_corePassTable_of_chunk (Or.inl (List.mem_of_getElem? (i := 1) rfl))),
      hmem_im (4198664, .ldr .x8 14 15 8, 4)
        (mem_corePassTable_of_chunk (Or.inl (List.mem_of_getElem? (i := 2) rfl))),
      hmem_im (4198668, .addSubReg false false 64 24 0 14 .lsl 0, 4)
        (mem_corePassTable_of_chunk (Or.inl (List.mem_of_getElem? (i := 3) rfl)))⟩
  · exact ⟨⟨2097152, hmem_im (4201384, .adr true 15 2097152, 4)
        (mem_corePassTable_of_chunk (Or.inr (Or.inl (List.mem_of_getElem? (i := 42) rfl)))), by decide⟩,
      hmem_im (4201388, .ldr .x8 0 15 0, 4)
        (mem_corePassTable_of_chunk (Or.inr (Or.inl (List.mem_of_getElem? (i := 43) rfl)))),
      hmem_im (4201392, .ldr .x8 14 15 8, 4)
        (mem_corePassTable_of_chunk (Or.inr (Or.inl (List.mem_of_getElem? (i := 44) rfl)))),
      hmem_im (4201396, .addSubReg false false 64 24 0 14 .lsl 0, 4)
        (mem_corePassTable_of_chunk (Or.inr (Or.inl (List.mem_of_getElem? (i := 45) rfl))))⟩

private theorem insStepExtraOk_enter_r14 {lo R pc : Nat} {ins : Ins}
    (h : insStepExtraOk lo R pc ins = true) (hnotb : ∀ off, ins ≠ .b false off)
    (hc : inR14 (pc - lo) = false) (hn : inR14 (pc - lo + 4) = true) :
    ∃ rn offL, ins = .ldr .b1 14 rn offL := by
  unfold insStepExtraOk at h; simp only [Bool.and_eq_true] at h
  have h14 := h.1.1
  cases ins with
  | b link off =>
    cases link
    · exact absurd rfl (hnotb off)
    · simp [hc, hn] at h14
  | ldr sz rt rn offL =>
    have hok := h14.1.1
    simp only [hc, Bool.not_false, hn, and_self, ↓reduceIte] at hok
    cases sz with
    | b1 =>
      by_cases hrt : rt = 14
      · subst hrt; exact ⟨rn, offL, rfl⟩
      · exfalso; split at hok
        · next _ _ heq => injection heq with _ hrt'; exact hrt hrt'
        · cases hok
    | _ => simp at hok
  | _ => simp [hc, hn] at h14

private theorem insStepExtraOk_stay_r14 {lo R pc : Nat} {ins : Ins}
    (h : insStepExtraOk lo R pc ins = true) (hnotb : ∀ off, ins ≠ .b false off)
    (hc : inR14 (pc - lo) = true) :
    match insWriteReg ins with | none => True | some d => d ≠ 14 := by
  unfold insStepExtraOk at h; simp only [Bool.and_eq_true] at h
  have h14 := h.1.1
  cases ins with
  | b link off =>
    cases link
    · exact absurd rfl (hnotb off)
    · trivial
  | _ =>
    have hok := h14.1.1
    simp only [hc, Bool.not_true, Bool.false_eq_true, false_and, ↓reduceIte] at hok
    cases hwr : insWriteReg _ with
    | none => trivial
    | some d => rw [hwr] at hok; simpa using hok

private theorem insStepExtraOk_enter_r15 {lo R pc : Nat} {ins : Ins}
    (h : insStepExtraOk lo R pc ins = true) (hnotb : ∀ off, ins ≠ .b false off)
    (hc : inR15 (pc - lo) = false) (hn : inR15 (pc - lo + 4) = true) :
    ∃ rn offL, ins = .ldr .b1 15 rn offL := by
  unfold insStepExtraOk at h; simp only [Bool.and_eq_true] at h
  have h15 := h.1.1
  cases ins with
  | b link off =>
    cases link
    · exact absurd rfl (hnotb off)
    · simp [hc, hn] at h15
  | ldr sz rt rn offL =>
    have hok := h15.1.2
    simp only [hc, Bool.not_false, hn, and_self, ↓reduceIte] at hok
    cases sz with
    | b1 =>
      by_cases hrt : rt = 15
      · subst hrt; exact ⟨rn, offL, rfl⟩
      · exfalso; split at hok
        · next _ _ heq => injection heq with _ hrt'; exact hrt hrt'
        · cases hok
    | _ => simp at hok
  | _ => simp [hc, hn] at h15

private theorem insStepExtraOk_stay_r15 {lo R pc : Nat} {ins : Ins}
    (h : insStepExtraOk lo R pc ins = true) (hnotb : ∀ off, ins ≠ .b false off)
    (hc : inR15 (pc - lo) = true) :
    match insWriteReg ins with | none => True | some d => d ≠ 15 := by
  unfold insStepExtraOk at h; simp only [Bool.and_eq_true] at h
  have h15 := h.1.1
  cases ins with
  | b link off =>
    cases link
    · exact absurd rfl (hnotb off)
    · trivial
  | _ =>
    have hok := h15.1.2
    simp only [hc, Bool.not_true, Bool.false_eq_true, false_and, ↓reduceIte] at hok
    cases hwr : insWriteReg _ with
    | none => trivial
    | some d => rw [hwr] at hok; simpa using hok

theorem clean_pass_step_inv {lo hi nxt R D len : Nat}
    (h12 : (lo = L_p1 ∧ hi = L_p2 ∧ nxt = L_p2 ∧ R = 19 ∧ D = 0x600000) ∨
           (lo = L_p2 ∧ hi = L_p3 ∧ nxt = L_p3 ∧ R = 20 ∧ D = 0x601000))
    {s : St} (hpc0 : s.pc = lo) (hby : ∀ a x, s.mem.rd a = some x → x < 256)
    (hD8 : rdMem s.mem (D + 8) .x8 = some len) (hlen : len < AArch64Env.W) :
    ∀ t (u : St), run im t (.ok s) = .ok u →
      (∀ j < t, ∃ uj, run im j (.ok s) = .ok uj ∧ lo ≤ uj.pc ∧ uj.pc < hi ∧
        (uj.pc - lo) % 4 = 0 ∧ uj.regs.length = 31 ∧ VerdictRegsWf uj) →
      u.mem = s.mem ∧ PassStepInv lo D len u := by
  have hpass : PassSpec lo hi nxt R := by
    rcases h12 with ⟨rfl, rfl, rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl, rfl, rfl⟩
    · exact Or.inl ⟨rfl, rfl, rfl, rfl⟩
    · exact Or.inr (Or.inl ⟨rfl, rfl, rfl, rfl⟩)
  have hhiW : hi < AArch64Env.W := by
    rcases h12 with ⟨_, rfl, _, _, _⟩ | ⟨_, rfl, _, _, _⟩ <;> decide
  have hhiW4 : hi + 4 < AArch64Env.W := by
    rcases h12 with ⟨_, rfl, _, _, _⟩ | ⟨_, rfl, _, _, _⟩ <;> decide
  have hlo_lt_nxt : lo + 16 < nxt := by
    rcases h12 with ⟨rfl, _, rfl, _, _⟩ | ⟨rfl, _, rfl, _, _⟩ <;> decide
  have hD_lt : D + 8 < AArch64Env.W := by
    rcases h12 with ⟨_, _, _, _, rfl⟩ | ⟨_, _, _, _, rfl⟩ <;> decide
  have hhd := pass_head_ins (lo := lo) (D := D) (by
    rcases h12 with ⟨rfl, _, _, _, rfl⟩ | ⟨rfl, _, _, _, rfl⟩
    · exact Or.inl ⟨rfl, rfl⟩
    · exact Or.inr ⟨rfl, rfl⟩)
  intro t
  induction t with
  | zero =>
    intro u hu _
    simp only [run] at hu; cases hu
    refine ⟨rfl, fun h => by omega, fun h => by omega,
      fun h => by rw [hpc0, Nat.sub_self] at h; exact Bool.noConfusion h,
      fun h => by rw [hpc0, Nat.sub_self] at h; exact Bool.noConfusion h⟩
  | succ t ih =>
    intro u hu hstay
    obtain ⟨ut, hut, hlo_t, hlt_t, hal_t, hl_t, hwf_t⟩ := hstay t (Nat.lt_succ_self t)
    obtain ⟨hmem_t, hinv_t⟩ := ih ut hut (fun j hj => hstay j (by omega))
    rw [show t + 1 = t + 1 from rfl, run_add, hut] at hu
    simp only [run] at hu
    cases him : im ut.pc with
    | none => simp [step, him] at hu
    | some p =>
      obtain ⟨ins, lenI⟩ := p
      obtain ⟨rfl, hsafe⟩ := im_in_pass_safe hpass hlo_t hlt_t him
      have hx : exec ins 4 ut = .ok u := by simpa [step, him] using hu
      have ⟨_, hprops, _, _⟩ := exec_pass_props hsafe hlo_t hlt_t hhiW hal_t hl_t hwf_t
      obtain ⟨hm_u, _, _, _, _⟩ := hprops u hx
      have hkind := insPassSafe_kind hsafe
      have hextra0 := insPassSafe_extra hsafe
      have hdir : ins.isDirect = true := by
        cases ins <;> first | rfl | exact False.elim hkind
      have hmod4 : (ut.pc + 4) % AArch64Env.W = ut.pc + 4 := Nat.mod_eq_of_lt (by omega)
      refine ⟨hm_u.trans hmem_t, ?_, ?_, ?_, ?_⟩
      · -- u.pc = lo + 4 ∨ u.pc = lo + 8 -> u.reg 15 = some D
        intro hpc48
        rcases exec_pc_direct hdir hx with hfall | ⟨offB, hbr, hpc'⟩
        · rw [hfall, hmod4] at hpc48
          rcases hpc48 with h4 | h8
          · have hpc0' : ut.pc = lo := by omega
            obtain ⟨imm, him0, heqD⟩ := hhd.1
            rw [hpc0', him0] at him; cases him
            simp only [exec, hpc0', ↓reduceIte, heqD] at hx; cases hx
            simp [St.wrZR, St.setReg, St.reg, hl_t, pow2_64,
              Nat.mod_eq_of_lt (by omega : D < 18446744073709551616)]
          · have hpc4' : ut.pc = lo + 4 := by omega
            rw [hpc4', hhd.2.1] at him; cases him
            have h15 := hinv_t.1 (Or.inl hpc4')
            exact (exec_insWriteReg_ne hx hkind (by dsimp [insWriteReg]; decide)).trans h15
        · exfalso
          have hbr' : ins = .b false offB ∨ (∃ cc, ins = .bcond cc offB) ∨ (∃ nz w rt, ins = .cbz nz w rt offB) := by
            rcases hbr with rfl | rfl | ⟨cc, rfl⟩ | ⟨nz, w, rt, rfl⟩ | ⟨nz, bit, rt, rfl⟩
            · exact Or.inl rfl
            · exact False.elim hkind
            · exact Or.inr (Or.inl ⟨cc, rfl⟩)
            · exact Or.inr (Or.inr ⟨nz, w, rt, rfl⟩)
            · exact False.elim hkind
          have hgt := insPassSafe_br_gt16 hsafe hbr'
          rw [hpc'] at hpc48; rcases hgt with hgt | heqn <;> omega
      · -- u.pc = lo + 12 ∨ u.pc = lo + 16 -> u.reg 14 = some len
        intro hpc1216
        rcases exec_pc_direct hdir hx with hfall | ⟨offB, hbr, hpc'⟩
        · rw [hfall, hmod4] at hpc1216
          rcases hpc1216 with h12' | h16'
          · have hpc8' : ut.pc = lo + 8 := by omega
            rw [hpc8', hhd.2.2.1] at him; cases him
            have h15 := hinv_t.1 (Or.inr hpc8')
            have hDmod : D % pow2 64 = D := Nat.mod_eq_of_lt (by rw [pow2_64]; omega)
            have hD8m : (D + 8) % AArch64Env.W = D + 8 := Nat.mod_eq_of_lt hD_lt
            simp only [exec, lift, bind, Option.bind, pure, St.rdSP,
              show (15 == 31) = false by decide, Bool.false_eq_true, ↓reduceIte,
              h15, Option.map_some, hDmod, hD8m, hmem_t, hD8] at hx
            cases hx
            simp [St.wrZR, St.setReg, St.reg, hl_t, MemSize.width, pow2_64,
              Nat.mod_eq_of_lt (by unfold AArch64Env.W at hlen; exact hlen)]
          · have hpc12' : ut.pc = lo + 12 := by omega
            rw [hpc12', hhd.2.2.2] at him; cases him
            have h14 := hinv_t.2.1 (Or.inl hpc12')
            exact (exec_insWriteReg_ne hx hkind (by dsimp [insWriteReg]; decide)).trans h14
        · exfalso
          have hbr' : ins = .b false offB ∨ (∃ cc, ins = .bcond cc offB) ∨ (∃ nz w rt, ins = .cbz nz w rt offB) := by
            rcases hbr with rfl | rfl | ⟨cc, rfl⟩ | ⟨nz, w, rt, rfl⟩ | ⟨nz, bit, rt, rfl⟩
            · exact Or.inl rfl
            · exact False.elim hkind
            · exact Or.inr (Or.inl ⟨cc, rfl⟩)
            · exact Or.inr (Or.inr ⟨nz, w, rt, rfl⟩)
            · exact False.elim hkind
          have hgt := insPassSafe_br_gt16 hsafe hbr'
          rw [hpc'] at hpc1216; rcases hgt with hgt | heqn <;> omega
      · -- inR14 (u.pc - lo) -> ∃ b < 256, u.reg 14 = some b
        intro hr14'
        have hextra := hextra0
        unfold insStepExtraOk at hextra; simp only [Bool.and_eq_true] at hextra
        obtain ⟨⟨_, hbrE⟩, _⟩ := hextra
        rcases exec_pc_direct hdir hx with hfall | ⟨offB, hbr, hpc'⟩
        · have hoff_eq : u.pc - lo = ut.pc - lo + 4 := by rw [hfall, hmod4]; omega
          rw [hoff_eq] at hr14'
          by_cases hbis : ∃ offB, ins = .b false offB
          · obtain ⟨offB, rfl⟩ := hbis
            simp only [exec] at hx; cases hx
            cases h14_cur : inR14 (ut.pc - lo)
            · simp only [h14_cur, Bool.not_false, ↓reduceIte, Bool.and_eq_true, Bool.not_eq_true'] at hbrE
              rw [hoff_eq, hr14'] at hbrE; exact Bool.noConfusion hbrE.1.1
            · exact hinv_t.2.2.1 h14_cur
          · have hnotb : ∀ offB, ins ≠ .b false offB := fun offB heq => hbis ⟨offB, heq⟩
            cases h14_cur : inR14 (ut.pc - lo)
            · obtain ⟨rn, offL, rfl⟩ := insStepExtraOk_enter_r14 hextra0 hnotb h14_cur hr14'
              simp only [exec, lift, bind, Option.bind, pure] at hx
              cases h1 : ut.rdSP 64 rn with
              | none => simp [h1] at hx
              | some v =>
                simp only [h1] at hx
                by_cases hsp : (rn == 31 && v % 16 != 0) = true
                · simp [hsp] at hx
                · simp only [hsp] at hx; rw [if_neg (by simp)] at hx
                  cases h2 : rdMem ut.mem ((v + offL) % AArch64Env.W) .b1 with
                  | none => simp [h2] at hx
                  | some b =>
                    simp only [h2] at hx; cases hx
                    have hb256 : b < 256 := rdMem_b1_lt_256 (hmem_t ▸ hby) h2
                    refine ⟨b, hb256, ?_⟩
                    simp [St.wrZR, St.setReg, St.reg, hl_t, MemSize.width, pow2_32,
                      Nat.mod_eq_of_lt (by omega : b < 2 ^ 32)]
            · obtain ⟨b, hb, h14⟩ := hinv_t.2.2.1 h14_cur
              exact ⟨b, hb, (exec_insWriteReg_ne hx hkind (insStepExtraOk_stay_r14 hextra0 hnotb h14_cur)).trans h14⟩
        · cases h14_cur : inR14 (ut.pc - lo)
          · exfalso
            rcases hbr with rfl | rfl | ⟨cc, rfl⟩ | ⟨nz, w, rt, rfl⟩ | ⟨nz, bit, rt, rfl⟩
            · simp only [h14_cur, Bool.not_false, ↓reduceIte, Bool.and_eq_true, Bool.not_eq_true'] at hbrE
              rw [hpc'] at hr14'; rw [hr14'] at hbrE; exact Bool.noConfusion hbrE.1.1
            · exact False.elim hkind
            · simp only [h14_cur, Bool.not_false, ↓reduceIte, Bool.and_eq_true, Bool.not_eq_true'] at hbrE
              rw [hpc'] at hr14'; rw [hr14'] at hbrE; exact Bool.noConfusion hbrE.1.1
            · simp only [h14_cur, Bool.not_false, ↓reduceIte, Bool.and_eq_true, Bool.not_eq_true'] at hbrE
              rw [hpc'] at hr14'; rw [hr14'] at hbrE; exact Bool.noConfusion hbrE.1.1
            · exact False.elim hkind
          · obtain ⟨b, hb, h14⟩ := hinv_t.2.2.1 h14_cur
            have hne14 : match insWriteReg ins with | none => True | some d => d ≠ 14 := by
              rcases hbr with rfl | rfl | ⟨cc, rfl⟩ | ⟨nz, w, rt, rfl⟩ | ⟨nz, bit, rt, rfl⟩ <;> trivial
            exact ⟨b, hb, (exec_insWriteReg_ne hx hkind hne14).trans h14⟩
      · -- inR15 (u.pc - lo) -> ∃ b < 256, u.reg 15 = some b
        intro hr15'
        have hextra := hextra0
        unfold insStepExtraOk at hextra; simp only [Bool.and_eq_true] at hextra
        obtain ⟨⟨_, hbrE⟩, _⟩ := hextra
        rcases exec_pc_direct hdir hx with hfall | ⟨offB, hbr, hpc'⟩
        · have hoff_eq : u.pc - lo = ut.pc - lo + 4 := by rw [hfall, hmod4]; omega
          rw [hoff_eq] at hr15'
          by_cases hbis : ∃ offB, ins = .b false offB
          · obtain ⟨offB, rfl⟩ := hbis
            simp only [exec] at hx; cases hx
            cases h15_cur : inR15 (ut.pc - lo)
            · simp only [h15_cur, Bool.not_false, ↓reduceIte, Bool.and_eq_true, Bool.not_eq_true'] at hbrE
              rw [hoff_eq, hr15'] at hbrE; exact Bool.noConfusion hbrE.1.2
            · exact hinv_t.2.2.2 h15_cur
          · have hnotb : ∀ offB, ins ≠ .b false offB := fun offB heq => hbis ⟨offB, heq⟩
            cases h15_cur : inR15 (ut.pc - lo)
            · obtain ⟨rn, offL, rfl⟩ := insStepExtraOk_enter_r15 hextra0 hnotb h15_cur hr15'
              simp only [exec, lift, bind, Option.bind, pure] at hx
              cases h1 : ut.rdSP 64 rn with
              | none => simp [h1] at hx
              | some v =>
                simp only [h1] at hx
                by_cases hsp : (rn == 31 && v % 16 != 0) = true
                · simp [hsp] at hx
                · simp only [hsp] at hx; rw [if_neg (by simp)] at hx
                  cases h2 : rdMem ut.mem ((v + offL) % AArch64Env.W) .b1 with
                  | none => simp [h2] at hx
                  | some b =>
                    simp only [h2] at hx; cases hx
                    have hb256 : b < 256 := rdMem_b1_lt_256 (hmem_t ▸ hby) h2
                    refine ⟨b, hb256, ?_⟩
                    simp [St.wrZR, St.setReg, St.reg, hl_t, MemSize.width, pow2_32,
                      Nat.mod_eq_of_lt (by omega : b < 2 ^ 32)]
            · obtain ⟨b, hb, h15⟩ := hinv_t.2.2.2 h15_cur
              exact ⟨b, hb, (exec_insWriteReg_ne hx hkind (insStepExtraOk_stay_r15 hextra0 hnotb h15_cur)).trans h15⟩
        · cases h15_cur : inR15 (ut.pc - lo)
          · exfalso
            rcases hbr with rfl | rfl | ⟨cc, rfl⟩ | ⟨nz, w, rt, rfl⟩ | ⟨nz, bit, rt, rfl⟩
            · simp only [h15_cur, Bool.not_false, ↓reduceIte, Bool.and_eq_true, Bool.not_eq_true'] at hbrE
              rw [hpc'] at hr15'; rw [hr15'] at hbrE; exact Bool.noConfusion hbrE.1.2
            · exact False.elim hkind
            · simp only [h15_cur, Bool.not_false, ↓reduceIte, Bool.and_eq_true, Bool.not_eq_true'] at hbrE
              rw [hpc'] at hr15'; rw [hr15'] at hbrE; exact Bool.noConfusion hbrE.1.2
            · simp only [h15_cur, Bool.not_false, ↓reduceIte, Bool.and_eq_true, Bool.not_eq_true'] at hbrE
              rw [hpc'] at hr15'; rw [hr15'] at hbrE; exact Bool.noConfusion hbrE.1.2
            · exact False.elim hkind
          · obtain ⟨b, hb, h15⟩ := hinv_t.2.2.2 h15_cur
            have hne15 : match insWriteReg ins with | none => True | some d => d ≠ 15 := by
              rcases hbr with rfl | rfl | ⟨cc, rfl⟩ | ⟨nz, w, rt, rfl⟩ | ⟨nz, bit, rt, rfl⟩ <;> trivial
            exact ⟨b, hb, (exec_insWriteReg_ne hx hkind hne15).trans h15⟩

private theorem pass_acc_5steps {lo hi nxt R : Nat} (hpass : PassSpec lo hi nxt R)
    {u : St} (hpc : u.pc = lo + 2668) (hl : u.regs.length = 31) :
    ∃ u_acc, run im 5 (.ok u) = .ok u_acc ∧ u_acc.pc = nxt ∧ u_acc.reg R = some ACCEPT := by
  have hmem_im : ∀ e, e ∈ corePassTable → im e.1 = some e.2 :=
    fun e he => table_ok e (List.mem_append.2 (Or.inl (List.mem_append.2 (Or.inl he))))
  have h5 :
      im (lo + 2668) = some (.movWide 2 64 R 15510 0, 4) ∧
      im (lo + 2672) = some (.movWide 3 64 R 42330 1, 4) ∧
      im (lo + 2676) = some (.movWide 3 64 R 59760 2, 4) ∧
      im (lo + 2680) = some (.movWide 3 64 R 23244 3, 4) ∧
      ∃ off, im (lo + 2684) = some (.b false off, 4) ∧ (lo + 2684 + off) % AArch64Env.W = nxt := by
    rcases hpass with ⟨rfl, rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl, rfl⟩
    · exact ⟨hmem_im (4201324, .movWide 2 64 19 15510 0, 4) (mem_corePassTable_of_chunk (Or.inr (Or.inl (List.mem_of_getElem? (i := 27) rfl)))),
        hmem_im (4201328, .movWide 3 64 19 42330 1, 4) (mem_corePassTable_of_chunk (Or.inr (Or.inl (List.mem_of_getElem? (i := 28) rfl)))),
        hmem_im (4201332, .movWide 3 64 19 59760 2, 4) (mem_corePassTable_of_chunk (Or.inr (Or.inl (List.mem_of_getElem? (i := 29) rfl)))),
        hmem_im (4201336, .movWide 3 64 19 23244 3, 4) (mem_corePassTable_of_chunk (Or.inr (Or.inl (List.mem_of_getElem? (i := 30) rfl)))),
        44, hmem_im (4201340, .b false 44, 4) (mem_corePassTable_of_chunk (Or.inr (Or.inl (List.mem_of_getElem? (i := 31) rfl)))), by decide⟩
    · exact ⟨hmem_im (4204052, .movWide 2 64 20 15510 0, 4) (mem_corePassTable_of_chunk (Or.inr (Or.inr (Or.inl (List.mem_of_getElem? (i := 5) rfl))))),
        hmem_im (4204056, .movWide 3 64 20 42330 1, 4) (mem_corePassTable_of_chunk (Or.inr (Or.inr (Or.inl (List.mem_of_getElem? (i := 6) rfl))))),
        hmem_im (4204060, .movWide 3 64 20 59760 2, 4) (mem_corePassTable_of_chunk (Or.inr (Or.inr (Or.inl (List.mem_of_getElem? (i := 7) rfl))))),
        hmem_im (4204064, .movWide 3 64 20 23244 3, 4) (mem_corePassTable_of_chunk (Or.inr (Or.inr (Or.inl (List.mem_of_getElem? (i := 8) rfl))))),
        44, hmem_im (4204068, .b false 44, 4) (mem_corePassTable_of_chunk (Or.inr (Or.inr (Or.inl (List.mem_of_getElem? (i := 9) rfl))))), by decide⟩
    · exact ⟨hmem_im (4206780, .movWide 2 64 21 15510 0, 4) (mem_corePassTable_of_chunk (Or.inr (Or.inr (Or.inr (List.mem_of_getElem? (i := 47) rfl))))),
        hmem_im (4206784, .movWide 3 64 21 42330 1, 4) (mem_corePassTable_of_chunk (Or.inr (Or.inr (Or.inr (List.mem_of_getElem? (i := 48) rfl))))),
        hmem_im (4206788, .movWide 3 64 21 59760 2, 4) (mem_corePassTable_of_chunk (Or.inr (Or.inr (Or.inr (List.mem_of_getElem? (i := 49) rfl))))),
        hmem_im (4206792, .movWide 3 64 21 23244 3, 4) (mem_corePassTable_of_chunk (Or.inr (Or.inr (Or.inr (List.mem_of_getElem? (i := 50) rfl))))),
        17916, hmem_im (4206796, .b false 17916, 4) (mem_corePassTable_of_chunk (Or.inr (Or.inr (Or.inr (List.mem_of_getElem? (i := 51) rfl))))), by decide⟩
  obtain ⟨f0, f1, f2, f3, off, f4, hoff⟩ := h5
  have hR_lt : R < 31 := by
    rcases hpass with ⟨_, _, _, rfl⟩ | ⟨_, _, _, rfl⟩ | ⟨_, _, _, rfl⟩ <;> decide
  have hR_ne31 : R ≠ 31 := by omega
  have hloW : lo + 2688 < AArch64Env.W := by
    rcases hpass with ⟨rfl, _, _, _⟩ | ⟨rfl, _, _, _⟩ | ⟨rfl, _, _, _⟩ <;> decide
  have hpc0 : (lo + 2668 + 4) % AArch64Env.W = lo + 2672 := Nat.mod_eq_of_lt (by omega)
  have hpc1 : (lo + 2672 + 4) % AArch64Env.W = lo + 2676 := Nat.mod_eq_of_lt (by omega)
  have hpc2 : (lo + 2676 + 4) % AArch64Env.W = lo + 2680 := Nat.mod_eq_of_lt (by omega)
  have hpc3 : (lo + 2680 + 4) % AArch64Env.W = lo + 2684 := Nat.mod_eq_of_lt (by omega)
  have ha0 : 15510 * pow2 (0 * 16) % pow2 64 = 0x3c96 := by decide
  have ha1 : (0x3c96 % pow2 (1 * 16) + 42330 * pow2 (1 * 16) +
      0x3c96 / pow2 (1 * 16 + 16) * pow2 (1 * 16 + 16)) % pow2 64 = 0xa55a3c96 := by decide
  have ha2 : (0xa55a3c96 % pow2 (2 * 16) + 59760 * pow2 (2 * 16) +
      0xa55a3c96 / pow2 (2 * 16 + 16) * pow2 (2 * 16 + 16)) % pow2 64 = 0xe970a55a3c96 := by decide
  have ha3 : (0xe970a55a3c96 % pow2 48 + 23244 * pow2 48) % pow2 64 = ACCEPT := by decide
  let s1 := (u.withPc (lo + 2672)).setReg R (some 0x3c96)
  let s2 := (s1.withPc (lo + 2676)).setReg R (some 0xa55a3c96)
  let s3 := (s2.withPc (lo + 2680)).setReg R (some 0xe970a55a3c96)
  let s4 := (s3.withPc (lo + 2684)).setReg R (some ACCEPT)
  let s5 := s4.withPc nxt
  have hl1 : s1.regs.length = 31 := by simp [s1, hl]
  have hl2 : s2.regs.length = 31 := by simp [s2, hl1]
  have hl3 : s3.regs.length = 31 := by simp [s3, hl2]
  have hst0 : step im u = .ok s1 := by
    simp [step, hpc, f0, exec, wrZR_ne31 hR_ne31, hpc0, ha0, s1, St.withPc]
  have hst1 : step im s1 = .ok s2 := by
    have hr : s1.reg R = some 0x3c96 := reg_setReg_eq (s := u.withPc (lo + 2672)) (by simp [hl]; omega)
    simp [step, show s1.pc = lo + 2672 from rfl, f1, exec, lift, bind, Option.bind, pure,
      rdZR_of_reg hR_ne31 hr, show 0x3c96 % pow2 64 = 0x3c96 by decide,
      wrZR_ne31 hR_ne31, hpc1, ha1, s2, St.withPc]
  have hst2 : step im s2 = .ok s3 := by
    have hr : s2.reg R = some 0xa55a3c96 := reg_setReg_eq (s := s1.withPc (lo + 2676)) (by simp [hl1]; omega)
    simp [step, show s2.pc = lo + 2676 from rfl, f2, exec, lift, bind, Option.bind, pure,
      rdZR_of_reg hR_ne31 hr, show 0xa55a3c96 % pow2 64 = 0xa55a3c96 by decide,
      wrZR_ne31 hR_ne31, hpc2, ha2, s3, St.withPc]
  have hst3 : step im s3 = .ok s4 := by
    have hr : s3.reg R = some 0xe970a55a3c96 := reg_setReg_eq (s := s2.withPc (lo + 2680)) (by simp [hl2]; omega)
    simp [step, show s3.pc = lo + 2680 from rfl, f3, exec, lift, bind, Option.bind, pure,
      rdZR_of_reg hR_ne31 hr, show 0xe970a55a3c96 % pow2 64 = 0xe970a55a3c96 by decide,
      wrZR_ne31 hR_ne31, hpc3, ha3, s4, St.withPc]
  have hst4 : step im s4 = .ok s5 := by
    simp [step, show s4.pc = lo + 2684 from rfl, f4, exec, hoff, s5, St.withPc]
  refine ⟨s5, ?_, rfl, ?_⟩
  · rw [show (5 : Nat) = 1 + (1 + (1 + (1 + 1))) from rfl,
      run_add, show run im 1 (.ok u) = step im u from rfl, hst0,
      run_add, show run im 1 (.ok s1) = step im s1 from rfl, hst1,
      run_add, show run im 1 (.ok s2) = step im s2 from rfl, hst2,
      run_add, show run im 1 (.ok s3) = step im s3 from rfl, hst3,
      show run im 1 (.ok s4) = step im s4 from rfl, hst4]
  · exact reg_setReg_eq (s := s3.withPc (lo + 2684)) (by simp [hl3]; omega)

private theorem movk_preserves_mod65536 {old imm16 hw : Nat} (hhw : 1 ≤ hw) (hlt : hw < 4) :
    (old % pow2 (hw * 16) + imm16 * pow2 (hw * 16) +
      old / pow2 (hw * 16 + 16) * pow2 (hw * 16 + 16)) % pow2 64 % 65536 = old % 65536 := by
  have hd : 65536 ∣ pow2 64 := by decide
  rw [Nat.mod_mod_of_dvd _ hd]
  have h1 : hw = 1 ∨ hw = 2 ∨ hw = 3 := by omega
  rcases h1 with rfl | rfl | rfl <;> (unfold pow2; omega)

/-- On an invalid input (where the clean pass ends with `s_end.reg R = some v_end ≠ ACCEPT`),
the clean pass NEVER enters `[lo + 2668, lo + 2688)` and preserves `reg R ≠ ACCEPT` at every step `< N`. -/
theorem clean_pass_never_acc {lo hi nxt R N v_end : Nat} (hpass : PassSpec lo hi nxt R)
    {s s_end : St} (hpc0 : s.pc = lo) (hR0 : s.reg R = some REJECT)
    (hrunN : run im N (.ok s) = .ok s_end) (hend_pc : s_end.pc = nxt)
    (hend_R : s_end.reg R = some v_end) (hne_acc : v_end ≠ ACCEPT)
    (hstay : ∀ t < N, ∃ u, run im t (.ok s) = .ok u ∧ lo ≤ u.pc ∧ u.pc < hi ∧
      (u.pc - lo) % 4 = 0 ∧ u.regs.length = 31 ∧ VerdictRegsWf u) :
    ∀ t < N, ∀ u, run im t (.ok s) = .ok u →
      ¬ (lo + 2668 ≤ u.pc ∧ u.pc < lo + 2688) ∧
      ∃ vR < AArch64Env.W, u.reg R = some vR ∧ vR % 65536 ≠ ACCEPT % 65536 := by
  have hb : nxt = L_p2 ∨ nxt = L_p3 ∨ nxt = L_vote := by
    rcases hpass with ⟨_, _, rfl, _⟩ | ⟨_, _, rfl, _⟩ | ⟨_, _, rfl, _⟩ <;> simp
  have hin_nxt : inCoreCode nxt := by
    rcases hb with rfl | rfl | rfl
    · exact Or.inl ⟨by decide, by decide⟩
    · exact Or.inl ⟨by decide, by decide⟩
    · exact Or.inr ⟨Nat.le_refl _, by decide⟩
  have hhiW : hi < AArch64Env.W := by
    rcases hpass with ⟨_, rfl, _, _⟩ | ⟨_, rfl, _, _⟩ | ⟨_, rfl, _, _⟩ <;> decide
  have hhiW4 : hi + 4 < AArch64Env.W := by
    rcases hpass with ⟨_, rfl, _, _⟩ | ⟨_, rfl, _, _⟩ | ⟨_, rfl, _, _⟩ <;> decide
  have hR_lt : R < 31 := by
    rcases hpass with ⟨_, _, _, rfl⟩ | ⟨_, _, _, rfl⟩ | ⟨_, _, _, rfl⟩ <;> decide
  have hno_entry : ∀ t < N, ∀ u, run im t (.ok s) = .ok u → u.pc ≠ lo + 2668 := by
    intro t ht u hu heq
    obtain ⟨u', hu', _, _, _, hl_u, _⟩ := hstay t ht
    rw [hu] at hu'; cases hu'
    obtain ⟨u_acc, hr5, hpc_acc, hR_acc⟩ := pass_acc_5steps hpass heq hl_u
    have hrun5 : run im (t + 5) (.ok s) = .ok u_acc := by rw [run_add, hu, hr5]
    by_cases hcmp : t + 5 = N
    · rw [hcmp, hrunN] at hrun5; cases hrun5; rw [hend_R] at hR_acc; cases hR_acc; exact hne_acc rfl
    · by_cases hlt5 : t + 5 < N
      · have hr_rem : run im (N - (t + 5)) (.ok u_acc) = .ok s_end := by
          have := hrunN; rw [show N = (t + 5) + (N - (t + 5)) by omega, run_add, hrun5] at this; exact this
        have hgt := run_core_stays_gt hb (N - (t + 5)) u_acc s_end (by omega) (Or.inl (hpc_acc ▸ hin_nxt)) (by omega) hr_rem
        omega
      · have hr_rem : run im (t + 5 - N) (.ok s_end) = .ok u_acc := by
          have := hrun5; rw [show t + 5 = N + (t + 5 - N) by omega, run_add, hrunN] at this; exact this
        have hgt := run_core_stays_gt hb (t + 5 - N) s_end u_acc (by omega) (Or.inl (hend_pc ▸ hin_nxt)) (by omega) hr_rem
        omega
  intro t
  induction t with
  | zero =>
    intro ht u hu
    simp only [run] at hu; cases hu
    exact ⟨by omega, REJECT, by decide, hR0, by decide⟩
  | succ t ih =>
    intro ht u hu
    obtain ⟨ut, hut, hlo_t, hlt_t, hal_t, hl_t, hwf_t⟩ := hstay t (by omega)
    obtain ⟨hnot_acc_t, vR, hvR_lt, hR_t, hmod_t⟩ := ih (by omega) ut hut
    have hne2668 := hno_entry (t + 1) ht u hu
    rw [run_add, hut] at hu; simp only [run] at hu
    cases him : im ut.pc with
    | none => simp [step, him] at hu
    | some p =>
      obtain ⟨ins, lenI⟩ := p
      obtain ⟨rfl, hsafe⟩ := im_in_pass_safe hpass hlo_t hlt_t him
      have hx : exec ins 4 ut = .ok u := by simpa [step, him] using hu
      have hkind := insPassSafe_kind hsafe
      have hextra := insPassSafe_extra hsafe
      have hdir : ins.isDirect = true := by cases ins <;> first | rfl | exact False.elim hkind
      have hmod4 : (ut.pc + 4) % AArch64Env.W = ut.pc + 4 := Nat.mod_eq_of_lt (by omega)
      unfold insStepExtraOk at hextra; simp only [Bool.and_eq_true] at hextra
      obtain ⟨⟨_, hbrE⟩, hverE⟩ := hextra
      have hnot_acc_u : ¬ (lo + 2668 ≤ u.pc ∧ u.pc < lo + 2688) := by
        rintro ⟨h1, h2⟩
        rcases exec_pc_direct hdir hx with hfall | ⟨offB, hbr, hpc'⟩
        · rw [hfall, hmod4] at h1 h2 hne2668
          exact hnot_acc_t ⟨by omega, by omega⟩
        · rcases hbr with rfl | rfl | ⟨cc, rfl⟩ | ⟨nz, w, rt, rfl⟩ | ⟨nz, bit, rt, rfl⟩
          · simp only [Bool.and_eq_true, Bool.not_eq_true', Bool.and_eq_false_iff, decide_eq_false_iff_not] at hbrE
            rw [hpc'] at h1 h2 hne2668; rcases hbrE.2 with ha | hb <;> omega
          · exact False.elim hkind
          · simp only [Bool.and_eq_true, Bool.not_eq_true', Bool.and_eq_false_iff, decide_eq_false_iff_not] at hbrE
            rw [hpc'] at h1 h2 hne2668; rcases hbrE.2 with ha | hb <;> omega
          · simp only [Bool.and_eq_true, Bool.not_eq_true', Bool.and_eq_false_iff, decide_eq_false_iff_not] at hbrE
            rw [hpc'] at h1 h2 hne2668; rcases hbrE.2 with ha | hb <;> omega
          · exact False.elim hkind
      refine ⟨hnot_acc_u, ?_⟩
      have hcond_f : ¬ (decide (2668 ≤ ut.pc - lo) = true ∧ decide (ut.pc - lo < 2688) = true) := by
        simp only [decide_eq_true_eq]; exact fun ⟨h1, h2⟩ => hnot_acc_t ⟨by omega, by omega⟩
      rw [if_neg hcond_f] at hverE
      by_cases hwrR : (match insWriteReg ins with | none => true | some d => d != R) = true
      · have hneR : match insWriteReg ins with | none => True | some d => d ≠ R := by
          cases h : insWriteReg ins with
          | none => trivial
          | some d => rw [h] at hwrR; simpa using hwrR
        exact ⟨vR, hvR_lt, (exec_insWriteReg_ne hx hkind hneR).trans hR_t, hmod_t⟩
      · have hwrF : (match insWriteReg ins with | none => true | some d => d != R) = false :=
          Bool.eq_false_iff.2 hwrR
        simp only [hwrF, Bool.false_eq_true, ↓reduceIte] at hverE
        cases ins with
        | movWide kind w rd imm16 hw =>
          cases kind with
          | succ k1 =>
            cases k1 with
            | succ k2 =>
              cases k2 with
              | zero =>
                -- movz 64 R imm16 0
                by_cases hw64 : w = 64
                · subst hw64
                  by_cases hhw0 : hw = 0
                  · subst hhw0
                    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, bne_iff_ne] at hverE
                    obtain ⟨⟨rfl, hlt65⟩, himm⟩ := hverE
                    simp only [exec] at hx; cases hx
                    refine ⟨(imm16 * pow2 (0 * 16)) % pow2 64,
                      by unfold AArch64Env.W; rw [← pow2_64]; exact Nat.mod_lt _ (by decide),
                      by exact reg_wrZR_eq (s := { ut with pc := (ut.pc + 4) % AArch64Env.W }) hR_lt hl_t, ?_⟩
                    have hpow1 : pow2 (0 * 16) = 1 := rfl
                    have hpow64 : pow2 64 = 18446744073709551616 := by decide
                    have hacc : ACCEPT % 65536 = 15510 := by decide
                    rw [hpow1, Nat.mul_one, hpow64, hacc]
                    omega
                  · cases hw with | zero => exact absurd rfl hhw0 | succ _ => simp at hverE
                · split at hverE
                  · next _ _ heq => injection heq with _ hw'; exact absurd hw' hw64
                  · next _ _ _ heq => injection heq with hk; cases hk
                  · cases hverE
              | succ k3 =>
                cases k3 with
                | zero =>
                  -- movk 64 R imm16 hw with 1 ≤ hw < 4
                  by_cases hw64 : w = 64
                  · subst hw64
                    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hverE
                    obtain ⟨⟨rfl, hhw1⟩, hhw4⟩ := hverE
                    simp only [exec, lift, bind, Option.bind, pure] at hx
                    rw [rdZR_of_reg (by omega) hR_t] at hx; simp only at hx; cases hx
                    refine ⟨((vR % pow2 64) % pow2 (hw * 16) + imm16 * pow2 (hw * 16) +
                        (vR % pow2 64) / pow2 (hw * 16 + 16) * pow2 (hw * 16 + 16)) % pow2 64,
                      by unfold AArch64Env.W; rw [← pow2_64]; exact Nat.mod_lt _ (by decide),
                      by exact reg_wrZR_eq (s := { ut with pc := (ut.pc + 4) % AArch64Env.W }) hR_lt hl_t, ?_⟩
                    rw [movk_preserves_mod65536 hhw1 hhw4]
                    have hd : 65536 ∣ pow2 64 := by decide
                    rw [Nat.mod_mod_of_dvd vR hd]
                    exact hmod_t
                  · split at hverE
                    · next _ _ heq => injection heq with hk; cases hk
                    · next _ _ _ heq => injection heq with _ hw'; exact absurd hw' hw64
                    · cases hverE
                | succ _ => simp at hverE
            | zero => simp at hverE
          | zero => simp at hverE
        | _ => simp at hverE

/-! ### Exhaustive 1-bit fetch audit over `corePassBytes` (`decide +kernel`) -/

private theorem byte_add_256_testBit {b x i : Nat} (hb : b < 256) :
    (b + 256 * x).testBit i = if i < 8 then b.testBit i else x.testBit (i - 8) := by
  simp only [Nat.testBit, Nat.shiftRight_eq_div_pow]
  by_cases hi : i < 8
  · simp only [hi, ↓reduceIte]
    have h8 : 256 = 2 ^ i * 2 ^ (8 - i) := by
      rw [← Nat.pow_add, show i + (8 - i) = 8 by omega]
    have heven : 2 ^ (8 - i) = 2 * 2 ^ (7 - i) := by
      rw [show 8 - i = 1 + (7 - i) by omega, Nat.pow_add, Nat.pow_one]
    have hmul : b + 256 * x = b + 2 ^ i * (2 * (2 ^ (7 - i) * x)) := by
      rw [h8, heven, Nat.mul_assoc, Nat.mul_assoc]
    rw [hmul, Nat.add_mul_div_left _ _ (Nat.two_pow_pos i)]
    rw [Nat.and_comm 1 _, Nat.and_comm 1 _, Nat.and_one_is_mod, Nat.and_one_is_mod]
    have hmod : (b / 2 ^ i + 2 * (2 ^ (7 - i) * x)) % 2 = (b / 2 ^ i) % 2 := by omega
    rw [hmod]
  · simp only [hi, ↓reduceIte]
    have hi8 : 2 ^ i = 256 * 2 ^ (i - 8) := by
      rw [show (256 : Nat) = 2 ^ 8 from rfl, ← Nat.pow_add, show 8 + (i - 8) = i by omega]
    rw [hi8, ← Nat.div_div_eq_div_mul, show (b + 256 * x) / 256 = x by omega]

private theorem xor_256_pack {b0 b1 b2 b3 bit : Nat}
    (h0 : b0 < 256) (h1 : b1 < 256) (h2 : b2 < 256) (h3 : b3 < 256) (hbit : bit < 32) :
    let w := b0 + 256 * (b1 + 256 * (b2 + 256 * b3))
    let q := bit / 8
    let r := bit % 8
    let b0' := if q = 0 then Nat.xor b0 (2 ^ r) else b0
    let b1' := if q = 1 then Nat.xor b1 (2 ^ r) else b1
    let b2' := if q = 2 then Nat.xor b2 (2 ^ r) else b2
    let b3' := if q = 3 then Nat.xor b3 (2 ^ r) else b3
    b0' + 256 * (b1' + 256 * (b2' + 256 * b3')) = Nat.xor w (2 ^ bit) := by
  dsimp only
  have hxor : ∀ a b, Nat.xor a b = a ^^^ b := fun _ _ => rfl
  have hr : bit % 8 < 8 := Nat.mod_lt _ (by decide)
  have h2r : 2 ^ (bit % 8) < 2 ^ 8 := Nat.pow_lt_pow_right (by decide) hr
  have h0' : (if bit / 8 = 0 then Nat.xor b0 (2 ^ (bit % 8)) else b0) < 256 := by
    split
    · exact Nat.xor_lt_two_pow (n := 8) h0 h2r
    · exact h0
  have h1' : (if bit / 8 = 1 then Nat.xor b1 (2 ^ (bit % 8)) else b1) < 256 := by
    split
    · exact Nat.xor_lt_two_pow (n := 8) h1 h2r
    · exact h1
  have h2' : (if bit / 8 = 2 then Nat.xor b2 (2 ^ (bit % 8)) else b2) < 256 := by
    split
    · exact Nat.xor_lt_two_pow (n := 8) h2 h2r
    · exact h2
  have h3' : (if bit / 8 = 3 then Nat.xor b3 (2 ^ (bit % 8)) else b3) < 256 := by
    split
    · exact Nat.xor_lt_two_pow (n := 8) h3 h2r
    · exact h3
  apply Nat.eq_of_testBit_eq; intro i
  simp only [hxor] at h0' h1' h2' h3' ⊢
  rw [byte_add_256_testBit h0', Nat.testBit_xor, Nat.testBit_two_pow,
    byte_add_256_testBit h0]
  by_cases hi8 : i < 8
  · simp only [hi8, ↓reduceIte]
    by_cases hq0 : bit / 8 = 0
    · simp only [hq0, ↓reduceIte, Nat.testBit_xor, Nat.testBit_two_pow]
      have : decide (bit % 8 = i) = decide (bit = i) := by
        simp only [decide_eq_decide]; omega
      rw [this]
    · simp only [hq0, ↓reduceIte]
      have : decide (bit = i) = false := decide_eq_false (by omega)
      simp [this]
  · simp only [hi8, ↓reduceIte, byte_add_256_testBit h1', byte_add_256_testBit h1]
    by_cases hi16 : i - 8 < 8
    · simp only [hi16, ↓reduceIte]
      by_cases hq1 : bit / 8 = 1
      · simp only [hq1, ↓reduceIte, Nat.testBit_xor, Nat.testBit_two_pow]
        have : decide (bit % 8 = i - 8) = decide (bit = i) := by
          simp only [decide_eq_decide]; omega
        rw [this]
      · simp only [hq1, ↓reduceIte]
        have : decide (bit = i) = false := decide_eq_false (by omega)
        simp [this]
    · simp only [hi16, ↓reduceIte, byte_add_256_testBit h2', byte_add_256_testBit h2]
      by_cases hi24 : i - 8 - 8 < 8
      · simp only [hi24, ↓reduceIte]
        by_cases hq2 : bit / 8 = 2
        · simp only [hq2, ↓reduceIte, Nat.testBit_xor, Nat.testBit_two_pow]
          have : decide (bit % 8 = i - 8 - 8) = decide (bit = i) := by
            simp only [decide_eq_decide]; omega
          rw [this]
        · simp only [hq2, ↓reduceIte]
          have : decide (bit = i) = false := decide_eq_false (by omega)
          simp [this]
      · simp only [hi24, ↓reduceIte]
        by_cases hq3 : bit / 8 = 3
        · simp only [hq3, ↓reduceIte, Nat.testBit_xor, Nat.testBit_two_pow]
          by_cases hi32 : i < 32
          · have : decide (bit % 8 = i - 8 - 8 - 8) = decide (bit = i) := by
              simp only [decide_eq_decide]; omega
            rw [this]
          · have hlt1 : decide (bit % 8 = i - 8 - 8 - 8) = false := decide_eq_false (by omega)
            have hlt2 : decide (bit = i) = false := decide_eq_false (by omega)
            rw [hlt1, hlt2]
        · simp only [hq3, ↓reduceIte]
          have : decide (bit = i) = false := decide_eq_false (by omega)
          simp [this]

private theorem decode_flipBit_eq_xor {b : Nat → Option Nat} {b0 b1 b2 b3 bit : Nat}
    (hb0 : b 0 = some b0) (hb1 : b 1 = some b1) (hb2 : b 2 = some b2) (hb3 : b 3 = some b3)
    (h0 : b0 < 256) (h1 : b1 < 256) (h2 : b2 < 256) (h3 : b3 < 256) (hbit : bit < 32) :
    decode (flipBit b bit) =
      (decodeWord (Nat.xor (b0 + 256 * (b1 + 256 * (b2 + 256 * b3))) (2 ^ bit))).map (·, 4) := by
  have hpack := xor_256_pack h0 h1 h2 h3 hbit
  dsimp only at hpack
  have hle : le (flipBit b bit) 0 4 =
      some (Nat.xor (b0 + 256 * (b1 + 256 * (b2 + 256 * b3))) (2 ^ bit)) := by
    rw [← hpack]
    simp only [le, flipBit, Nat.add_zero, Nat.zero_add, hb0, hb1, hb2, hb3, bind, Option.bind, pure, Nat.mul_zero]
    by_cases hq0 : bit / 8 = 0
    · simp [hq0]
    · by_cases hq1 : bit / 8 = 1
      · simp [hq1]
      · by_cases hq2 : bit / 8 = 2
        · simp [hq2]
        · have hq3 : bit / 8 = 3 := by omega
          simp [hq3]
  unfold decode
  rw [hle]
  simp only [bind, Option.bind, pure]
  cases decodeWord _ <;> rfl

/-- Target address guaranteed to decode to `im tgt = none` (`PAD0`, `PAD1`, `PAD2`, or OOB). -/
def targetImNone (tgt : Nat) : Bool :=
  (BASE + 176 ≤ tgt && tgt < L_p1) ||
  (L_p3_end ≤ tgt && tgt < L_vote) ||
  (L_core_end ≤ tgt && tgt < L_start) ||
  tgt < BASE || BASE + 47636 < tgt + 4

theorem targetImNone_sound {tgt : Nat} (h : targetImNone tgt = true) : im tgt = none := by
  unfold targetImNone at h
  simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at h
  rcases h with (((h0 | h1) | h2) | h3) | h4
  · exact im_pad0_none h0.1 h0.2
  · exact im_pad1_none h1.1 h1.2
  · exact im_pad2_none h2.1 h2.2
  · exact im_oob_none (Or.inl h3)
  · exact im_oob_none (Or.inr h4)

/-- Target address aligned inside `[L_p1, L_p3_end)`. -/
def targetInPasses (tgt : Nat) : Bool :=
  L_p1 ≤ tgt && tgt < L_p3_end && (tgt - L_p1) % 4 == 0

/-- Target address in `TAIL` at one of the 5 proven-safe `P3 b vote` perturbation entries. -/
def targetInTailK01248 (tgt : Nat) : Bool :=
  tgt == L_vote || tgt == L_vote + 4 || tgt == L_vote + 8 || tgt == L_vote + 16 || tgt == L_vote + 32

/-- Exhaustive per-word 1-bit fetch safety check for `pc ∈ [L_p1, L_p3_end)`. -/
def passFetchAltOk (pc origW altW : Nat) : Bool :=
  let passIdx := if pc < L_p2 then 1 else if pc < L_p3 then 2 else 3
  let passBase := if pc < L_p2 then L_p1 else if pc < L_p3 then L_p2 else L_p3
  let off := pc - passBase
  let regWrSafe (rd : Nat) : Bool :=
    passIdx == 1 || (rd != 19 && (passIdx == 2 || rd != 20))
  let brTgtOk (offB : Nat) : Bool :=
    let tgt := (pc + offB) % AArch64Env.W
    targetInPasses tgt || targetImNone tgt || (passIdx == 3 && targetInTailK01248 tgt)
  match decodeWord altW with
  | none => true
  | some ins' =>
    match ins' with
    | .svc _ | .br _ _ | .ret _ | .strReg _ _ _ _ _ _ => false
    | .b _ offB => brTgtOk offB
    | .bcond _ offB => brTgtOk offB
    | .cbz _ w _ offB => wOk w && brTgtOk offB
    | .tbz _ _ _ offB => brTgtOk offB
    | .movWide kind w rd imm16 hw =>
      wOk w &&
      (regWrSafe rd ||
        (passIdx == 3 && rd == 20 && w == 64 && hw < 4 && imm16 < 65536 &&
          (kind == 0 || kind == 2 || kind == 3)))
    | .adr isPage rd imm =>
      regWrSafe rd ||
      (let base := if isPage then (pc / 4096) * 4096 else pc
       ((base + imm) % AArch64Env.W) != ACCEPT)
    | .addSubImm _ _ w rd _ _ => wOk w && regWrSafe rd
    | .addSubReg _ _ w rd _ _ _ _ => wOk w && regWrSafe rd
    | .logicReg _ _ w rd _ _ _ _ => wOk w && regWrSafe rd
    | .bitfield _ w rd _ _ _ => wOk w && regWrSafe rd
    | .csel _ w rd _ _ _ => wOk w && regWrSafe rd
    | .ccmpReg _ w _ _ _ _ => wOk w
    | .ldr _ rt _ _ | .ldur _ rt _ _ | .ldrReg _ rt _ _ _ _ => regWrSafe rt
    | .str sz _ rn offS =>
      if passIdx == 3 then true
      else match decodeWord origW with
        | some (.ldr sz0 _ rn0 off0) => sz == sz0 && rn == rn0 && offS == off0
        | some (.addSubImm true true 64 31 rn0 imm12) =>
          sz == .x8 && rn == rn0 && offS == imm12 * 8 && imm12 ≤ 140 &&
          ((off == 16 && rn == 14 && imm12 == 16) ||
           (inR14 off && rn == 14) ||
           (inR15 off && rn == 15))
        | _ => false
    | .stur sz _ rn offS =>
      if passIdx == 3 then true
      else match decodeWord origW with
        | some (.ldur sz0 _ rn0 off0) => sz == sz0 && rn == rn0 && offS == off0
        | _ => false

/-- Linear `O(N)` pass over `corePassBytes` verifying byte bounds `< 256` and all 32
Hamming-1 fetch perturbations for all 2,046 instructions in `[L_p1, L_p3_end)`. -/
def corePassBytesAudit : Nat → List Nat → Bool
  | _, [] => true
  | pc, b0 :: b1 :: b2 :: b3 :: rest =>
    b0 < 256 && b1 < 256 && b2 < 256 && b3 < 256 &&
    let w := b0 + 256 * (b1 + 256 * (b2 + 256 * b3))
    (List.range 32).all (fun bit => passFetchAltOk pc w (Nat.xor w (2 ^ bit))) &&
    corePassBytesAudit (pc + 4) rest
  | _, _ => false

private theorem corePassBytesAudit_append :
    ∀ (l1 l2 : List Nat) (a : Nat),
      l1.length % 4 = 0 →
      corePassBytesAudit a l1 = true →
      corePassBytesAudit (a + l1.length) l2 = true →
      corePassBytesAudit a (l1 ++ l2) = true
  | [], l2, a, _, _, h2 => by simpa using h2
  | [_], _, _, hlen, _, _ => by simp at hlen
  | [_, _], _, _, hlen, _, _ => by simp at hlen
  | [_, _, _], _, _, hlen, _, _ => by simp at hlen
  | b0 :: b1 :: b2 :: b3 :: rest, l2, a, hlen, h1, h2 => by
    simp only [List.cons_append, corePassBytesAudit, Bool.and_eq_true] at h1 ⊢
    obtain ⟨⟨⟨⟨hb0, hb1⟩, hb2⟩, hb3⟩, hall, hrest⟩ := h1
    have hlen' : rest.length % 4 = 0 := by simp only [List.length_cons] at hlen; omega
    have h2' : corePassBytesAudit (a + 4 + rest.length) l2 = true := by
      have hl : (b0 :: b1 :: b2 :: b3 :: rest).length = rest.length + 4 := by simp
      rwa [show a + 4 + rest.length = a + (b0 :: b1 :: b2 :: b3 :: rest).length by omega]
    exact ⟨⟨⟨⟨hb0, hb1⟩, hb2⟩, hb3⟩, hall,
      corePassBytesAudit_append rest l2 (a + 4) hlen' hrest h2'⟩

private def auditSeg0 : List Nat := corePass_0 ++ corePass_1 ++ corePass_2 ++ corePass_3
private def auditSeg1 : List Nat := corePass_4 ++ corePass_5 ++ corePass_6 ++ corePass_7
private def auditSeg2 : List Nat := corePass_8 ++ corePass_9 ++ corePass_10 ++ corePass_11
private def auditSeg3 : List Nat := corePass_12 ++ corePass_13 ++ corePass_14 ++ corePass_15
private def auditSeg4 : List Nat := corePass_16 ++ corePass_17 ++ corePass_18 ++ corePass_19
private def auditSeg5 : List Nat := corePass_20 ++ corePass_21 ++ corePass_22 ++ corePass_23
private def auditSeg6 : List Nat := corePass_24 ++ corePass_25 ++ corePass_26 ++ corePass_27
private def auditSeg7 : List Nat := corePass_28 ++ corePass_29 ++ corePass_30 ++ corePass_31

private theorem auditSeg0_len : auditSeg0.length = 1024 := by decide +kernel
private theorem auditSeg1_len : auditSeg1.length = 1024 := by decide +kernel
private theorem auditSeg2_len : auditSeg2.length = 1024 := by decide +kernel
private theorem auditSeg3_len : auditSeg3.length = 1024 := by decide +kernel
private theorem auditSeg4_len : auditSeg4.length = 1024 := by decide +kernel
private theorem auditSeg5_len : auditSeg5.length = 1024 := by decide +kernel
private theorem auditSeg6_len : auditSeg6.length = 1024 := by decide +kernel

private theorem auditSeg0_ok : corePassBytesAudit L_p1 auditSeg0 = true := by decide +kernel
private theorem auditSeg1_ok : corePassBytesAudit (L_p1 + 1024) auditSeg1 = true := by decide +kernel
private theorem auditSeg2_ok : corePassBytesAudit (L_p1 + 2048) auditSeg2 = true := by decide +kernel
private theorem auditSeg3_ok : corePassBytesAudit (L_p1 + 3072) auditSeg3 = true := by decide +kernel
private theorem auditSeg4_ok : corePassBytesAudit (L_p1 + 4096) auditSeg4 = true := by decide +kernel
private theorem auditSeg5_ok : corePassBytesAudit (L_p1 + 5120) auditSeg5 = true := by decide +kernel
private theorem auditSeg6_ok : corePassBytesAudit (L_p1 + 6144) auditSeg6 = true := by decide +kernel
private theorem auditSeg7_ok : corePassBytesAudit (L_p1 + 7168) auditSeg7 = true := by decide +kernel

private theorem re_seg32 (c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15
    c16 c17 c18 c19 c20 c21 c22 c23 c24 c25 c26 c27 c28 c29 c30 c31 : List Nat) :
    c0 ++ c1 ++ c2 ++ c3 ++ c4 ++ c5 ++ c6 ++ c7 ++
    c8 ++ c9 ++ c10 ++ c11 ++ c12 ++ c13 ++ c14 ++ c15 ++
    c16 ++ c17 ++ c18 ++ c19 ++ c20 ++ c21 ++ c22 ++ c23 ++
    c24 ++ c25 ++ c26 ++ c27 ++ c28 ++ c29 ++ c30 ++ c31 =
    (c0 ++ c1 ++ c2 ++ c3) ++ (c4 ++ c5 ++ c6 ++ c7) ++
    (c8 ++ c9 ++ c10 ++ c11) ++ (c12 ++ c13 ++ c14 ++ c15) ++
    (c16 ++ c17 ++ c18 ++ c19) ++ (c20 ++ c21 ++ c22 ++ c23) ++
    (c24 ++ c25 ++ c26 ++ c27) ++ (c28 ++ c29 ++ c30 ++ c31) := by
  simp only [List.append_assoc]

private theorem corePassBytes_eq_segs :
    corePassBytes =
      auditSeg0 ++ auditSeg1 ++ auditSeg2 ++ auditSeg3 ++
        auditSeg4 ++ auditSeg5 ++ auditSeg6 ++ auditSeg7 :=
  re_seg32 corePass_0 corePass_1 corePass_2 corePass_3
    corePass_4 corePass_5 corePass_6 corePass_7
    corePass_8 corePass_9 corePass_10 corePass_11
    corePass_12 corePass_13 corePass_14 corePass_15
    corePass_16 corePass_17 corePass_18 corePass_19
    corePass_20 corePass_21 corePass_22 corePass_23
    corePass_24 corePass_25 corePass_26 corePass_27
    corePass_28 corePass_29 corePass_30 corePass_31

theorem corePassBytesAudit_ok : corePassBytesAudit L_p1 corePassBytes = true := by
  rw [corePassBytes_eq_segs]
  have h01 : corePassBytesAudit L_p1 (auditSeg0 ++ auditSeg1) = true :=
    corePassBytesAudit_append auditSeg0 auditSeg1 L_p1 (by rw [auditSeg0_len]) auditSeg0_ok
      (by rw [auditSeg0_len]; exact auditSeg1_ok)
  have h02 : corePassBytesAudit L_p1 (auditSeg0 ++ auditSeg1 ++ auditSeg2) = true :=
    corePassBytesAudit_append _ auditSeg2 L_p1
      (by simp [List.length_append, auditSeg0_len, auditSeg1_len]) h01
      (by simp only [List.length_append, auditSeg0_len, auditSeg1_len]; exact auditSeg2_ok)
  have h03 : corePassBytesAudit L_p1 (auditSeg0 ++ auditSeg1 ++ auditSeg2 ++ auditSeg3) = true :=
    corePassBytesAudit_append _ auditSeg3 L_p1
      (by simp [List.length_append, auditSeg0_len, auditSeg1_len, auditSeg2_len]) h02
      (by simp only [List.length_append, auditSeg0_len, auditSeg1_len, auditSeg2_len]; exact auditSeg3_ok)
  have h04 : corePassBytesAudit L_p1 (auditSeg0 ++ auditSeg1 ++ auditSeg2 ++ auditSeg3 ++ auditSeg4) = true :=
    corePassBytesAudit_append _ auditSeg4 L_p1
      (by simp [List.length_append, auditSeg0_len, auditSeg1_len, auditSeg2_len, auditSeg3_len]) h03
      (by simp only [List.length_append, auditSeg0_len, auditSeg1_len, auditSeg2_len, auditSeg3_len]; exact auditSeg4_ok)
  have h05 : corePassBytesAudit L_p1 (auditSeg0 ++ auditSeg1 ++ auditSeg2 ++ auditSeg3 ++ auditSeg4 ++ auditSeg5) = true :=
    corePassBytesAudit_append _ auditSeg5 L_p1
      (by simp [List.length_append, auditSeg0_len, auditSeg1_len, auditSeg2_len, auditSeg3_len, auditSeg4_len]) h04
      (by simp only [List.length_append, auditSeg0_len, auditSeg1_len, auditSeg2_len, auditSeg3_len, auditSeg4_len]; exact auditSeg5_ok)
  have h06 : corePassBytesAudit L_p1 (auditSeg0 ++ auditSeg1 ++ auditSeg2 ++ auditSeg3 ++ auditSeg4 ++ auditSeg5 ++ auditSeg6) = true :=
    corePassBytesAudit_append _ auditSeg6 L_p1
      (by simp [List.length_append, auditSeg0_len, auditSeg1_len, auditSeg2_len, auditSeg3_len, auditSeg4_len, auditSeg5_len]) h05
      (by simp only [List.length_append, auditSeg0_len, auditSeg1_len, auditSeg2_len, auditSeg3_len, auditSeg4_len, auditSeg5_len]; exact auditSeg6_ok)
  exact corePassBytesAudit_append _ auditSeg7 L_p1
    (by simp [List.length_append, auditSeg0_len, auditSeg1_len, auditSeg2_len, auditSeg3_len, auditSeg4_len, auditSeg5_len, auditSeg6_len]) h06
    (by simp only [List.length_append, auditSeg0_len, auditSeg1_len, auditSeg2_len, auditSeg3_len, auditSeg4_len, auditSeg5_len, auditSeg6_len]; exact auditSeg7_ok)

private theorem corePassBytesAudit_sound :
    ∀ (bs : List Nat) (a pc : Nat),
      corePassBytesAudit a bs = true →
      a ≤ pc → pc < a + bs.length → (pc - a) % 4 = 0 →
      ∃ b0 b1 b2 b3,
        bs[pc - a]? = some b0 ∧ bs[pc - a + 1]? = some b1 ∧
        bs[pc - a + 2]? = some b2 ∧ bs[pc - a + 3]? = some b3 ∧
        b0 < 256 ∧ b1 < 256 ∧ b2 < 256 ∧ b3 < 256 ∧
        ∀ bit < 32,
          let w := b0 + 256 * (b1 + 256 * (b2 + 256 * b3))
          passFetchAltOk pc w (Nat.xor w (2 ^ bit)) = true
  | [], _, _, _, _, hlt, _ => by simp at hlt; omega
  | [_], _, _, h, _, _, _ => by simp [corePassBytesAudit] at h
  | [_, _], _, _, h, _, _, _ => by simp [corePassBytesAudit] at h
  | [_, _, _], _, _, h, _, _, _ => by simp [corePassBytesAudit] at h
  | b0 :: b1 :: b2 :: b3 :: rest, a, pc, h, hlo, hhi, hal => by
    simp only [corePassBytesAudit, Bool.and_eq_true, decide_eq_true_eq] at h
    obtain ⟨⟨⟨⟨h0, h1⟩, h2⟩, h3⟩, hall, hrest⟩ := h
    by_cases heq : pc = a
    · subst heq
      refine ⟨b0, b1, b2, b3, by simp, by simp, by simp, by simp, h0, h1, h2, h3, ?_⟩
      intro bit hbit
      exact List.all_eq_true.1 hall bit (List.mem_range.2 hbit)
    · have h4 : a + 4 ≤ pc := by omega
      have hlen : pc < (a + 4) + rest.length := by simp only [List.length_cons] at hhi; omega
      obtain ⟨c0, c1, c2, c3, g0, g1, g2, g3, hc0, hc1, hc2, hc3, hok⟩ :=
        corePassBytesAudit_sound rest (a + 4) pc hrest h4 hlen (by omega)
      have hidx : ∀ k, (b0 :: b1 :: b2 :: b3 :: rest)[pc - a + k]? = rest[pc - (a + 4) + k]? := by
        intro k
        have hk : pc - a + k = (pc - (a + 4) + k) + 1 + 1 + 1 + 1 := by omega
        rw [hk]; simp
      exact ⟨c0, c1, c2, c3, by simpa using hidx 0 |>.trans (by simpa using g0),
        (hidx 1).trans g1, (hidx 2).trans g2, (hidx 3).trans g3, hc0, hc1, hc2, hc3, hok⟩

private theorem le4_of_bytes {b : Nat → Option Nat} {b0 b1 b2 b3 : Nat}
    (hb0 : b 0 = some b0) (hb1 : b 1 = some b1) (hb2 : b 2 = some b2) (hb3 : b 3 = some b3) :
    le b 0 4 = some (b0 + 256 * (b1 + 256 * (b2 + 256 * b3))) := by
  simp only [le, Nat.add_zero, Nat.zero_add, hb0, hb1, hb2, hb3, bind, Option.bind, pure, Nat.mul_zero]

private theorem imageAlts_audit_of_bytes {img : List Nat} {pc b0 b1 b2 b3 : Nat}
    {wAlt : Option (Ins × Nat)}
    (hlen : img.length = 47636)
    (hlo : L_p1 ≤ pc) (hhi : pc < L_p3_end) (hal : (pc - L_p1) % 4 = 0)
    (hb0 : img[pc - BASE + 0]? = some b0) (hb1 : img[pc - BASE + 1]? = some b1)
    (hb2 : img[pc - BASE + 2]? = some b2) (hb3 : img[pc - BASE + 3]? = some b3)
    (h0 : b0 < 256) (h1 : b1 < 256) (h2 : b2 < 256) (h3 : b3 < 256)
    (hok : ∀ bit < 32,
      let w := b0 + 256 * (b1 + 256 * (b2 + 256 * b3))
      passFetchAltOk pc w (Nat.xor w (2 ^ bit)) = true)
    (hw : wAlt ∈ imageAlts img BASE pc) :
    ∃ w bit, bit < 32 ∧
      imageOf img BASE pc = (decodeWord w).map (·, 4) ∧
      wAlt = (decodeWord (Nat.xor w (2 ^ bit))).map (·, 4) ∧
      passFetchAltOk pc w (Nat.xor w (2 ^ bit)) = true := by
  have hbase_le : BASE ≤ pc := Nat.le_trans (by decide : BASE ≤ L_p1) hlo
  have hpc_le : pc + 4 ≤ BASE + 47636 :=
    Nat.le_trans (Nat.succ_le_of_lt (Nat.lt_of_lt_of_le (Nat.add_lt_add_right hhi 3) (by decide : L_p3_end + 3 ≤ BASE + 47636))) (Nat.le_refl _)
  have hpc_mod4 : pc % 4 = 0 := by
    have hp1 : L_p1 % 4 = 0 := by decide
    have hadd : L_p1 + (pc - L_p1) = pc := Nat.add_sub_of_le hlo
    rw [← hadd, Nat.add_mod, hp1, hal]
  have hc_img : (BASE ≤ pc && pc + 4 ≤ BASE + img.length && BASE % 4 == 0 && pc % 4 == 0) = true := by
    rw [hlen]
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq]
    exact ⟨⟨⟨hbase_le, hpc_le⟩, by decide⟩, hpc_mod4⟩
  have halts_eq : imageAlts img BASE pc = altsAt (fun i => img[pc - BASE + i]?) := by
    unfold imageAlts; rw [if_pos hc_img]
  rw [halts_eq] at hw
  simp only [altsAt, List.mem_map, List.mem_range] at hw
  obtain ⟨bit, hbit, rfl⟩ := hw
  let w := b0 + 256 * (b1 + 256 * (b2 + 256 * b3))
  refine ⟨w, bit, hbit, ?_, decode_flipBit_eq_xor hb0 hb1 hb2 hb3 h0 h1 h2 h3 hbit, hok bit hbit⟩
  have hle : le (fun i => img[pc - BASE + i]?) 0 4 = some w :=
    le4_of_bytes (b := fun i => img[pc - BASE + i]?) hb0 hb1 hb2 hb3
  unfold imageOf; rw [if_pos hc_img]
  unfold decode; rw [hle]
  simp only [bind, Option.bind, pure]
  cases decodeWord w <;> rfl

private theorem drop_slice_audit_of_bytes {img cp rest : List Nat} {pc : Nat}
    {wAlt : Option (Ins × Nat)}
    (himg : img.length = 47636) (hcp : cp.length = 8184)
    (hdrop : img.drop (L_tripled - BASE) = cp ++ rest)
    (haud : corePassBytesAudit L_p1 cp = true)
    (hlo : L_p1 ≤ pc) (hhi : pc < L_p3_end) (hal : (pc - L_p1) % 4 = 0)
    (hw : wAlt ∈ imageAlts img BASE pc) :
    ∃ w bit, bit < 32 ∧
      imageOf img BASE pc = (decodeWord w).map (·, 4) ∧
      wAlt = (decodeWord (Nat.xor w (2 ^ bit))).map (·, 4) ∧
      passFetchAltOk pc w (Nat.xor w (2 ^ bit)) = true := by
  have hpc_len : pc < L_p1 + cp.length := by
    rw [hcp]
    exact Nat.lt_of_lt_of_le hhi (by decide : L_p3_end ≤ L_p1 + 8184)
  obtain ⟨b0, b1, b2, b3, g0, g1, g2, g3, h0, h1, h2, h3, hok⟩ :=
    corePassBytesAudit_sound cp L_p1 pc haud hlo hpc_len hal
  have hbyte : ∀ k < 4, img[pc - BASE + k]? = cp[pc - L_p1 + k]? := by
    intro k hk
    have hd : (img.drop (L_tripled - BASE))[pc - L_p1 + k]? = (cp ++ rest)[pc - L_p1 + k]? :=
      congrArg (fun l => l[pc - L_p1 + k]?) hdrop
    have hlt_cp : pc - L_p1 + k < cp.length := by
      rw [hcp]
      have h1 : L_p1 + 8184 = L_p3_end := by decide
      omega
    rw [List.getElem?_drop, List.getElem?_append_left hlt_cp] at hd
    have hidx : L_tripled - BASE + (pc - L_p1 + k) = pc - BASE + k := by
      have htr : L_tripled = L_p1 := rfl
      have hbl : BASE ≤ L_p1 := by decide
      rw [htr]; omega
    rwa [hidx] at hd
  have g0' : cp[pc - L_p1 + 0]? = some b0 := by rw [Nat.add_zero]; exact g0
  exact imageAlts_audit_of_bytes himg hlo hhi hal
    ((hbyte 0 (by decide)).trans g0')
    ((hbyte 1 (by decide)).trans g1)
    ((hbyte 2 (by decide)).trans g2)
    ((hbyte 3 (by decide)).trans g3)
    h0 h1 h2 h3 hok hw

theorem alts_in_corePass_audit {pc : Nat} {wAlt : Option (Ins × Nat)}
    (hlo : L_p1 ≤ pc) (hhi : pc < L_p3_end) (hal : (pc - L_p1) % 4 = 0)
    (hw : wAlt ∈ alts pc) :
    ∃ w bit, bit < 32 ∧
      im pc = (decodeWord w).map (·, 4) ∧
      wAlt = (decodeWord (Nat.xor w (2 ^ bit))).map (·, 4) ∧
      passFetchAltOk pc w (Nat.xor w (2 ^ bit)) = true :=
  drop_slice_audit_of_bytes spr1Arm64Bytes_len corePassBytes_len drop_tripled_eq corePassBytesAudit_ok hlo hhi hal hw

/-! ### `movWide` non-`ACCEPT` helper and `wrMem` locality lemmas -/

theorem movWide_x20_ne_accept {v kind imm16 hw : Nat}
    (hv : v = REJECT ∨ v = OVERFLOW) (_hkind : kind = 0 ∨ kind = 2 ∨ kind = 3)
    (hhw : hw < 4) (himm : imm16 < 65536) :
    (kind = 0 → (pow2 64 - 1 - imm16 * pow2 (hw * 16)) % pow2 64 ≠ ACCEPT) ∧
    (kind = 2 → (imm16 * pow2 (hw * 16)) % pow2 64 ≠ ACCEPT) ∧
    (kind = 3 → (v % pow2 (hw * 16) + imm16 * pow2 (hw * 16) +
      v / pow2 (hw * 16 + 16) * pow2 (hw * 16 + 16)) % pow2 64 ≠ ACCEPT) := by
  have hhw4 : hw = 0 ∨ hw = 1 ∨ hw = 2 ∨ hw = 3 := by omega
  refine ⟨fun _ => ?_, fun _ => ?_, fun _ => ?_⟩
  · rcases hhw4 with rfl | rfl | rfl | rfl <;> (dsimp only [pow2, ACCEPT]; omega)
  · rcases hhw4 with rfl | rfl | rfl | rfl <;> (dsimp only [pow2, ACCEPT]; omega)
  · rcases hv with rfl | rfl <;> rcases hhw4 with rfl | rfl | rfl | rfl <;>
      (dsimp only [pow2, ACCEPT, REJECT, OVERFLOW]; omega)

private theorem mem_find_spec : ∀ (m : Mem) (a k i : Nat) (r : Region),
    Mem.find m a k = some (i, r) →
    r ∈ m ∧ r.base ≤ a ∧ a < r.hi ∧ k ≤ i ∧ m[i - k]? = some r
  | [], _, _, _, _, h => by simp [Mem.find] at h
  | r0 :: rest, a, k, i, r, h => by
    simp only [Mem.find] at h
    split at h
    · next hc =>
      cases h
      exact ⟨List.mem_cons_self, hc.1, hc.2, Nat.le_refl _, by simp⟩
    · obtain ⟨hmem, hlo, hhi, hle, hget⟩ := mem_find_spec rest a (k + 1) i r h
      refine ⟨List.mem_cons_of_mem _ hmem, hlo, hhi, by omega, ?_⟩
      have hsub : i - k = (i - (k + 1)) + 1 := by omega
      rw [hsub]; simpa using hget

theorem mem_wr_rd_ne {m m' : Mem} {a v : Nat} (hwf : m.wf) (hwr : m.wr a v = some m') :
    m'.wf ∧ ∀ a', a' ≠ a → m'.rd a' = m.rd a' := by
  have hwf' : m'.wf := wf_of_shape (wr_shape hwr) hwf
  refine ⟨hwf', fun a' hne => ?_⟩
  unfold Mem.wr at hwr
  cases hf : Mem.find m a 0 with
  | none => simp [hf] at hwr
  | some p =>
    obtain ⟨idx, r⟩ := p
    simp only [hf, bind, Option.bind] at hwr
    split at hwr
    · next hrw =>
      simp only [pure, Option.some.injEq] at hwr
      obtain ⟨_, hlo_a, hhi_a, _, hget⟩ := mem_find_spec m a 0 idx r hf
      simp only [Nat.sub_zero] at hget
      have hidx_lt : idx < m.length := (List.getElem?_eq_some_iff.1 hget).1
      let r' : Region := { r with data := r.data.set (a - r.base) (v % 256) }
      have hm'_eq : m' = m.set idx r' := hwr.symm
      cases hf' : Mem.find m a' 0 with
      | none =>
        have hnone' : Mem.find m' a' 0 = none := by
          cases hfm : Mem.find m' a' 0 with
          | none => rfl
          | some q =>
            exfalso
            obtain ⟨j, rj⟩ := q
            obtain ⟨_, hlo', hhi', _, hget'⟩ := mem_find_spec m' a' 0 j rj hfm
            simp only [Nat.sub_zero, hm'_eq] at hget'
            by_cases hj : idx = j
            · subst hj; rw [List.getElem?_set_self hidx_lt] at hget'; cases hget'
              have hhi_eq : r'.hi = r.hi := by simp [r', Region.hi]
              rw [hhi_eq] at hhi'
              have := Mem.find_at_idx m idx 0 a' hwf hget hlo' hhi'
              rw [this] at hf'; cases hf'
            · rw [List.getElem?_set_ne hj] at hget'
              have := Mem.find_at_idx m j 0 a' hwf hget' hlo' hhi'
              rw [this] at hf'; cases hf'
        simp [Mem.rd, hf', hnone']
      | some q =>
        obtain ⟨j, rj⟩ := q
        obtain ⟨_, hlo', hhi', _, hgetj⟩ := mem_find_spec m a' 0 j rj hf'
        simp only [Nat.sub_zero] at hgetj
        by_cases hj : idx = j
        · subst hj
          rw [hget] at hgetj; cases hgetj
          have hgetj' : m'[idx]? = some r' := by rw [hm'_eq, List.getElem?_set_self hidx_lt]
          have hhi_eq : r'.hi = r.hi := by simp [r', Region.hi]
          rw [Mem.rd_at_idx hwf' hgetj' hlo' (hhi_eq.symm ▸ hhi'),
              Mem.rd_at_idx hwf hget hlo' hhi']
          exact List.getElem?_set_ne (by omega : a - r.base ≠ a' - r.base)
        · have hgetj' : m'[j]? = some rj := by rw [hm'_eq, List.getElem?_set_ne hj]; exact hgetj
          rw [Mem.rd_at_idx hwf' hgetj' hlo' hhi', Mem.rd_at_idx hwf hgetj hlo' hhi']
    · cases hwr

theorem mem_wrN_rd_ne : ∀ (bs : List Nat) (m m' : Mem) (ea : Nat),
    m.wf → m.wrN ea bs = some m' →
    m'.wf ∧ ∀ a, (∀ j < bs.length, a ≠ ea + j) → m'.rd a = m.rd a
  | [], m, m', _, hwf, hwr => by simp [Mem.wrN] at hwr; subst hwr; exact ⟨hwf, fun _ _ => rfl⟩
  | b :: rest, m, m', ea, hwf, hwr => by
    simp only [Mem.wrN, bind, Option.bind] at hwr
    cases h1 : m.wr ea b with
    | none => simp [h1] at hwr
    | some m1 =>
      simp only [h1] at hwr
      obtain ⟨hwf1, hrd1⟩ := mem_wr_rd_ne hwf h1
      obtain ⟨hwf', hrd'⟩ := mem_wrN_rd_ne rest m1 m' (ea + 1) hwf1 hwr
      refine ⟨hwf', fun a hne => ?_⟩
      have hne0 : a ≠ ea := by simpa using hne 0 (by simp)
      have hner : ∀ j < rest.length, a ≠ ea + 1 + j := by
        intro j hj; have := hne (j + 1) (by simp; omega); omega
      exact (hrd' a hner).trans (hrd1 a hne0)

private theorem bytesN_len (v : Nat) : ∀ n, (bytesN v n).length = n
  | 0 => rfl
  | n + 1 => by simp [bytesN, bytesN_len _ n]

theorem wrMem_rd_ne {m m' : Mem} {ea v : Nat} {sz : MemSize}
    (hwf : m.wf) (hwr : wrMem m ea v sz = some m') :
    ∀ a, (∀ j < sz.bytes, a ≠ ea + j) → m'.rd a = m.rd a := by
  have ⟨_, hrd⟩ := mem_wrN_rd_ne (bytesN v sz.bytes) m m' ea hwf hwr
  exact fun a hne => hrd a fun j hj => hne j (by rwa [bytesN_len] at hj)

theorem wrMem_some_ge_4096 {m m' : Mem} {ea v : Nat}
    (hlow : ∀ r ∈ m, 4096 ≤ r.base) (hwr : wrMem m ea v .x8 = some m') :
    4096 ≤ ea := by
  unfold wrMem MemSize.bytes bytesN Mem.wrN at hwr
  simp only [bind, Option.bind] at hwr
  cases h1 : m.wr ea (v % 256) with
  | none => simp [h1] at hwr
  | some m1 =>
    unfold Mem.wr at h1
    cases hf : Mem.find m ea 0 with
    | none => simp [hf] at h1
    | some p =>
      obtain ⟨_, r⟩ := p
      obtain ⟨hmem, hle, _, _, _⟩ := mem_find_spec m ea 0 _ r hf
      exact Nat.le_trans (hlow r hmem) hle

end SporosCheckArm64Bin
