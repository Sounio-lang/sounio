/-
SporosCheckArm64BinFault: machine-level unconditional 4-channel single-disturbance
immunity of the committed `47,636`-byte `spr1_check_arm64` ELF binary over
`AArch64Phys` / `AArch64Env`.

Single-disturbance model and explicit scope boundary:
  1. **Unconditional 4-channel (`fetch`, `reg`, `mem`, `nzcv`) across the entire
     post-triplication core `[L_tripled, L_core_end)`** (`P1 ++ P2 ++ P3` and
     the voter/guard tail `[L_vote, L_core_end)`, separated from `hdrBytes`,
     from each other, and from `_start` by `PAD0_UDF`, `PAD1_UDF`, and `PAD2_UDF`,
     with `im a = none` everywhere outside code) over the full binary image `im`
     and Hamming-1 fetch alternatives `alts` (bridged via
     `AArch64Phys.fetch_fault_fixed_boundary` and `corePassBytesAudit_ok`):
     for any invalid input `bs` and any post-triplication state `TripledState bs s0`
     (`s0.pc = L_tripled`), no single disturbance on any of the 4 channels at any
     step can produce `.exit 0`.
  2. **Explicitly excluded disturbances**:
     - Pre-triplication disturbances during `_start` input acquisition
       (`[L_start, BASE + 47636)`), before three pairwise-disjoint copies of
       `bs` exist.
-/
import Bootstrap.SporosCheckArm64BinFaultIso

namespace SporosCheckArm64Bin

open AArch64Env AArch64Phys SporosCheckArm64Flat

/-! ### Determinism of `.exit` and two-of-three voter exit safety -/

theorem run_exit_unique {n1 n2 c1 c2 : Nat} {o : Out}
    (h1 : run im n1 o = .exit c1) (h2 : run im n2 o = .exit c2) : c1 = c2 := by
  by_cases hle : n1 ≤ n2
  · have h := congrArg (run im (n2 - n1)) h1
    rw [← run_add, show n1 + (n2 - n1) = n2 by omega, h2, run_exit] at h
    exact Out.exit.inj h.symm
  · have h := congrArg (run im (n1 - n2)) h2
    rw [← run_add, show n2 + (n1 - n2) = n1 by omega, h1, run_exit] at h
    exact Out.exit.inj h

private theorem checkFlatWord_rej_or_ovf {bs : List Nat} (hinv : checkFlat bs ≠ some true) :
    checkFlatWord bs = REJECT ∨ checkFlatWord bs = OVERFLOW := by
  unfold checkFlatWord
  cases hcf : checkFlat bs with
  | none => exact Or.inr rfl
  | some ok =>
    cases ok with
    | false => exact Or.inl rfl
    | true => exact absurd hcf hinv

private theorem checkFlatWord_ne_accept {bs : List Nat} (hinv : checkFlat bs ≠ some true) :
    checkFlatWord bs ≠ ACCEPT := by
  rcases checkFlatWord_rej_or_ovf hinv with h | h <;> (rw [h]; decide)

theorem vote_two_ne_accept {s : St} {a b c : Nat}
    (hl : s.regs.length = 31) (hpc : s.pc = L_vote)
    (h19 : s.reg 19 = some a) (h20 : s.reg 20 = some b) (h21 : s.reg 21 = some c)
    (ha : a < AArch64Env.W) (hb : b < AArch64Env.W) (hc : c < AArch64Env.W)
    (h2 : (a ≠ ACCEPT ∧ b ≠ ACCEPT) ∨ (a ≠ ACCEPT ∧ c ≠ ACCEPT) ∨ (b ≠ ACCEPT ∧ c ≠ ACCEPT)) :
    ∀ n, run im n (.ok s) ≠ .exit 0 := by
  have hne_acc : TMR.vote a b c ≠ some ACCEPT := by
    unfold TMR.vote
    rcases h2 with ⟨h1, h2⟩ | ⟨h1, h2⟩ | ⟨h1, h2⟩ <;>
      (by_cases hab : a = b <;> by_cases hac : a = c <;> by_cases hbc : b = c <;> simp_all)
  intro n h0
  cases hv : TMR.vote a b c with
  | none =>
    exact absurd (run_exit_unique (vote_guard_refuse hl hpc h19 h20 h21 ha hb hc hv) h0) (by decide)
  | some w =>
    have hwA : w ≠ ACCEPT := fun h => hne_acc (by rw [hv, h])
    by_cases hwO : w = OVERFLOW
    · subst hwO
      exact absurd (run_exit_unique (vote_guard_overflow hl hpc h19 h20 h21 ha hb hc hv) h0) (by decide)
    · exact absurd (run_exit_unique (vote_guard_reject hl hpc h19 h20 h21 ha hb hc hv hwA hwO) h0) (by decide)

/-! ### Per-pass minimal footprint memory (`passSubMem`) and memory-disturbance locality -/

/-- Minimal 2-region memory containing strictly the 16-byte descriptor at `D`
and the `bs.length`-byte buffer at `b`. -/
def passSubMem (m : Mem) (D b : Nat) (bs : List Nat) : Mem :=
  [⟨D, (List.range 16).map (fun i => (m.rd (D + i)).getD 0), false⟩, ⟨b, bs, false⟩]

private theorem rdN_some_each : ∀ (m : Mem) (a n : Nat) (vs : List Nat),
    m.rdN a n = some vs → ∀ j < n, ∃ x, m.rd (a + j) = some x
  | _, _, 0, _, _, _, hj => by cases hj
  | m, a, n + 1, vs, h, j, hj => by
    simp only [Mem.rdN, bind, Option.bind, pure] at h
    cases h0 : m.rd a with
    | none => simp [h0] at h
    | some x =>
      simp only [h0] at h
      cases hr : m.rdN (a + 1) n with
      | none => simp [hr] at h
      | some rest =>
        cases j with
        | zero => exact ⟨x, by simpa using h0⟩
        | succ j' =>
          have ⟨y, hy⟩ := rdN_some_each m (a + 1) n rest hr j' (by omega)
          exact ⟨y, by rwa [show a + 1 + j' = a + (j' + 1) by omega] at hy⟩

private theorem rdMem_some_each {m : Mem} {a v : Nat} {sz : MemSize} (h : rdMem m a sz = some v) :
    ∀ j < sz.bytes, ∃ x, m.rd (a + j) = some x := by
  unfold rdMem at h
  cases hr : m.rdN a sz.bytes with
  | none => simp [hr] at h
  | some vs => exact rdN_some_each m a sz.bytes vs hr

private theorem rdMem_x8_some_bytes {m : Mem} {a v : Nat} (h : rdMem m a .x8 = some v) :
    ∀ j < 8, ∃ x, m.rd (a + j) = some x :=
  rdMem_some_each h

private theorem passSubMem_rd_eq {m : Mem} {D b : Nat} {bs : List Nat}
    (hD0 : rdMem m D .x8 = some b) (hD8 : rdMem m (D + 8) .x8 = some bs.length)
    (hbuf : Buf m b bs) (_hdisj : b + bs.length ≤ D ∨ D + 16 ≤ b) :
    (∀ a, (passSubMem m D b bs).rd a =
      if D ≤ a ∧ a < D + 16 then m.rd a
      else if b ≤ a ∧ a < b + bs.length then m.rd a
      else none) := by
  intro a
  have hlen16 : ((List.range 16).map (fun i => (m.rd (D + i)).getD 0)).length = 16 := by simp
  have hLHS : (passSubMem m D b bs).rd a =
      if D ≤ a ∧ a < D + 16 then
        (((List.range 16).map (fun i => (m.rd (D + i)).getD 0))[a - D]?)
      else if b ≤ a ∧ a < b + bs.length then
        bs[a - b]?
      else none := by
    show (do let (_, r) ← (passSubMem m D b bs).find a 0; r.data[a - r.base]?) = _
    unfold passSubMem Mem.find Region.hi
    rw [hlen16]
    by_cases hD : D ≤ a ∧ a < D + 16
    · simp only [hD, and_self, ↓reduceIte, bind, Option.bind]
    · simp only [hD, ↓reduceIte]
      by_cases hB : b ≤ a ∧ a < b + bs.length
      · simp only [Mem.find, Region.hi, hB, and_self, ↓reduceIte, bind, Option.bind]
      · simp only [Mem.find, Region.hi, hB, ↓reduceIte, bind, Option.bind]
  rw [hLHS]
  by_cases hD : D ≤ a ∧ a < D + 16
  · simp only [hD, and_self, ↓reduceIte, List.getElem?_map,
      List.getElem?_range (by omega : a - D < 16), Option.map_some]
    have hex : ∃ x, m.rd (D + (a - D)) = some x := by
      by_cases h8 : a - D < 8
      · exact rdMem_x8_some_bytes hD0 (a - D) h8
      · have ⟨x, hx⟩ := rdMem_x8_some_bytes hD8 (a - D - 8) (by omega)
        exact ⟨x, by rwa [show D + 8 + (a - D - 8) = D + (a - D) by omega] at hx⟩
    obtain ⟨x, hx⟩ := hex
    rw [show D + (a - D) = a by omega] at hx ⊢
    rw [hx]; rfl
  · simp only [hD, ↓reduceIte]
    by_cases hB : b ≤ a ∧ a < b + bs.length
    · simp only [hB, and_self, ↓reduceIte]
      have h1 := hbuf (a - b) (by omega)
      rw [show b + (a - b) = a by omega] at h1
      rw [h1, List.getElem?_eq_getElem (by omega)]
    · simp only [hB, ↓reduceIte]

theorem passSubMem_spec {m : Mem} {D b : Nat} {bs : List Nat}
    (hD0 : rdMem m D .x8 = some b) (hD8 : rdMem m (D + 8) .x8 = some bs.length)
    (hbuf : Buf m b bs) (hdisj : b + bs.length ≤ D ∨ D + 16 ≤ b) :
    rdMem (passSubMem m D b bs) D .x8 = some b ∧
    rdMem (passSubMem m D b bs) (D + 8) .x8 = some bs.length ∧
    Buf (passSubMem m D b bs) b bs ∧
    (∀ m', (∀ a, ((D ≤ a ∧ a < D + 16) ∨ (b ≤ a ∧ a < b + bs.length)) → m'.rd a = m.rd a) →
      MemSub (passSubMem m D b bs) m') := by
  have hrd := passSubMem_rd_eq hD0 hD8 hbuf hdisj
  have hsub_eq : ∀ {m1 m2 : Mem} {a n : Nat},
      (∀ j < n, m1.rd (a + j) = m2.rd (a + j)) → m1.rdN a n = m2.rdN a n := by
    intro m1 m2 a n
    induction n generalizing a with
    | zero => intro _; rfl
    | succ k ih =>
      intro hag
      have h0 : m1.rd a = m2.rd a := by simpa using hag 0 (Nat.zero_lt_succ k)
      simp only [Mem.rdN, h0]
      congr 1; funext x
      rw [ih fun j hj => by rw [show a + 1 + j = a + (j + 1) by omega]; exact hag (j + 1) (by omega)]
  refine ⟨?_, ?_, ?_, ?_⟩
  · unfold rdMem MemSize.bytes at hD0 ⊢; dsimp only at hD0 ⊢
    rw [hsub_eq (m2 := m) fun j hj => by rw [hrd]; simp [show D ≤ D + j ∧ D + j < D + 16 by omega]]
    exact hD0
  · unfold rdMem MemSize.bytes at hD8 ⊢; dsimp only at hD8 ⊢
    rw [hsub_eq (m2 := m) fun j hj => by rw [hrd]; simp [show D ≤ D + 8 + j ∧ D + 8 + j < D + 16 by omega]]
    exact hD8
  · intro j hj
    rw [hrd]
    have hnotD : ¬ (D ≤ b + j ∧ b + j < D + 16) := by omega
    simp only [hnotD, ↓reduceIte, show b ≤ b + j ∧ b + j < b + bs.length by omega, and_self]
    exact hbuf j hj
  · intro m' hag a x hx
    rw [hrd] at hx
    by_cases hD : D ≤ a ∧ a < D + 16
    · simp only [hD, and_self, ↓reduceIte] at hx
      rw [hag a (Or.inl hD), hx]
    · simp only [hD, ↓reduceIte] at hx
      by_cases hB : b ≤ a ∧ a < b + bs.length
      · simp only [hB, and_self, ↓reduceIte] at hx
        rw [hag a (Or.inr hB), hx]
      · simp only [hB, ↓reduceIte] at hx; cases hx

private theorem passSubMem_bytes_lt_256 {m : Mem} {D b : Nat} {bs : List Nat}
    (hD0 : rdMem m D .x8 = some b) (hD8 : rdMem m (D + 8) .x8 = some bs.length)
    (hbuf : Buf m b bs) (hdisj : b + bs.length ≤ D ∨ D + 16 ≤ b)
    (hdesc : ∀ j < 16, ∀ x, m.rd (D + j) = some x → x < 256)
    (hby : ∀ x ∈ bs, x < 256) :
    ∀ a x, (passSubMem m D b bs).rd a = some x → x < 256 := by
  intro a x hx
  rw [passSubMem_rd_eq hD0 hD8 hbuf hdisj] at hx
  by_cases hD : D ≤ a ∧ a < D + 16
  · simp only [hD, and_self, ↓reduceIte] at hx
    exact hdesc (a - D) (by omega) x (by rwa [show D + (a - D) = a by omega])
  · simp only [hD, ↓reduceIte] at hx
    by_cases hB : b ≤ a ∧ a < b + bs.length
    · simp only [hB, and_self, ↓reduceIte] at hx
      have hb := hbuf (a - b) (by omega)
      rw [show b + (a - b) = a by omega] at hb
      rw [hb] at hx; cases hx
      exact hby _ (List.getElem_mem (by omega))
    · simp only [hB, ↓reduceIte] at hx; cases hx

private theorem list_modify_cons_zero {α : Type} (f : α → α) (a : α) (l : List α) :
    (a :: l).modify 0 f = f a :: l := rfl

private theorem list_modify_cons_succ {α : Type} (f : α → α) (a : α) (l : List α) (i : Nat) :
    (a :: l).modify (i + 1) f = a :: l.modify i f := rfl

private theorem getElem?_modify_ne {α : Type} (f : α → α) :
    ∀ (l : List α) (j k : Nat), k ≠ j → (l.modify j f)[k]? = l[k]?
  | [], _, _, _ => by simp
  | _ :: _, 0, 0, hne => absurd rfl hne
  | _ :: _, 0, _ + 1, _ => by simp
  | _ :: _, _ + 1, 0, _ => by simp
  | _ :: t, j + 1, k + 1, hne => by
    rw [list_modify_cons_succ, List.getElem?_cons_succ, List.getElem?_cons_succ]
    exact getElem?_modify_ne f t j k (by omega)

private theorem mem_modify_rd_ne (f : Nat → Nat) :
    ∀ (m : Mem) (i j k a : Nat),
    (∀ r, m[i]? = some r → a ≠ r.base + j) →
    (do let (_, r) ← Mem.find (m.modify i fun r => { r with data := r.data.modify j f }) a k; r.data[a - r.base]?) =
    (do let (_, r) ← Mem.find m a k; r.data[a - r.base]?)
  | [], _, _, _, _, _ => by simp
  | r :: rest, 0, j, k, a, hne => by
    rw [list_modify_cons_zero]
    simp only [Mem.find, Region.hi, List.length_modify]
    by_cases h : r.base ≤ a ∧ a < r.base + r.data.length
    · simp only [h, and_self, ↓reduceIte, bind, Option.bind]
      have ha_ne : a ≠ r.base + j := hne r rfl
      exact getElem?_modify_ne f r.data j (a - r.base) (by omega)
    · simp only [h, ↓reduceIte]
  | r :: rest, i + 1, j, k, a, hne => by
    rw [list_modify_cons_succ]
    simp only [Mem.find]
    by_cases h : r.base ≤ a ∧ a < r.hi
    · simp only [h, and_self, ↓reduceIte]
    · simp only [h, ↓reduceIte]
      exact mem_modify_rd_ne f rest i j (k + 1) a (fun r' hr' => hne r' hr')

theorem flipMem_at_most_one_addr {s : St} (_hwf : s.mem.wf) (i j bit : Nat) :
    ∃ aStar, ∀ a, a ≠ aStar → (s.flipMem i j bit).mem.rd a = s.mem.rd a := by
  cases hr : s.mem[i]? with
  | none =>
    refine ⟨0, fun a _ => ?_⟩
    exact mem_modify_rd_ne (fun v => Nat.xor v (2 ^ bit)) s.mem i j 0 a (fun r' hr' => by rw [hr] at hr'; cases hr')
  | some r =>
    refine ⟨r.base + j, fun a hne => ?_⟩
    exact mem_modify_rd_ne (fun v => Nat.xor v (2 ^ bit)) s.mem i j 0 a
      (fun r' hr' => by rw [hr] at hr'; cases hr'; exact hne)

private def inFootprint (D b len a : Nat) : Prop :=
  (D ≤ a ∧ a < D + 16) ∨ (b ≤ a ∧ a < b + len)

theorem flipMem_preserves_two_subMems {bs : List Nat} {s : St} (ht : TripledState bs s)
    (i j bit : Nat) :
    let m' := (s.flipMem i j bit).mem
    let m1 := passSubMem s.mem 0x600000 ht.b1 bs
    let m2 := passSubMem s.mem 0x601000 ht.b2 bs
    let m3 := passSubMem s.mem 0x602000 ht.b3 bs
    (MemSub m2 m' ∧ MemSub m3 m') ∨ (MemSub m1 m' ∧ MemSub m3 m') ∨ (MemSub m1 m' ∧ MemSub m2 m') := by
  obtain ⟨aStar, haStar⟩ := flipMem_at_most_one_addr ht.memWf i j bit
  have hs1 := passSubMem_spec ht.hD1_ptr ht.hD1_len ht.hBuf1 (ht.disjD1 0x600000 (by simp))
  have hs2 := passSubMem_spec ht.hD2_ptr ht.hD2_len ht.hBuf2 (ht.disjD2 0x601000 (by simp))
  have hs3 := passSubMem_spec ht.hD3_ptr ht.hD3_len ht.hBuf3 (ht.disjD3 0x602000 (by simp))
  have hsub1 : ¬ inFootprint 0x600000 ht.b1 bs.length aStar → MemSub (passSubMem s.mem 0x600000 ht.b1 bs) (s.flipMem i j bit).mem :=
    fun hnot => hs1.2.2.2 _ fun a ha => haStar a (fun heq => hnot (heq ▸ ha))
  have hsub2 : ¬ inFootprint 0x601000 ht.b2 bs.length aStar → MemSub (passSubMem s.mem 0x601000 ht.b2 bs) (s.flipMem i j bit).mem :=
    fun hnot => hs2.2.2.2 _ fun a ha => haStar a (fun heq => hnot (heq ▸ ha))
  have hsub3 : ¬ inFootprint 0x602000 ht.b3 bs.length aStar → MemSub (passSubMem s.mem 0x602000 ht.b3 bs) (s.flipMem i j bit).mem :=
    fun hnot => hs3.2.2.2 _ fun a ha => haStar a (fun heq => hnot (heq ▸ ha))
  by_cases h1 : inFootprint 0x600000 ht.b1 bs.length aStar
  · left
    have hnot2 : ¬ inFootprint 0x601000 ht.b2 bs.length aStar := by
      have hd12 := ht.disj12
      have hdD1 := ht.disjD1 0x601000 (by simp)
      have hdD2 := ht.disjD2 0x600000 (by simp)
      dsimp only [TripledState.b1, TripledState.b2, inFootprint] at h1 ⊢; omega
    have hnot3 : ¬ inFootprint 0x602000 ht.b3 bs.length aStar := by
      have hd13 := ht.disj13
      have hdD1 := ht.disjD1 0x602000 (by simp)
      have hdD3 := ht.disjD3 0x600000 (by simp)
      dsimp only [TripledState.b1, TripledState.b3, inFootprint] at h1 ⊢; omega
    exact ⟨hsub2 hnot2, hsub3 hnot3⟩
  · by_cases h2 : inFootprint 0x601000 ht.b2 bs.length aStar
    · right; left
      have hnot3 : ¬ inFootprint 0x602000 ht.b3 bs.length aStar := by
        have hd23 := ht.disj23
        have hdD2 := ht.disjD2 0x602000 (by simp)
        have hdD3 := ht.disjD3 0x601000 (by simp)
        dsimp only [TripledState.b2, TripledState.b3, inFootprint] at h2 ⊢; omega
      exact ⟨hsub1 h1, hsub3 hnot3⟩
    · right; right; exact ⟨hsub1 h1, hsub2 h2⟩

/-- If a clean pass load succeeded in `passSubMem s.mem D1 b1 bs` at `(ea, sz)`, then a fetch-flipped
store at the same `(ea, sz)` on `s.mem` preserves any pairwise-disjoint pass footprint `(D2, b2)`. -/
private theorem wrMem_passSubMem_preserves_other {m m' : Mem} {D1 b1 D2 b2 ea val v : Nat}
    {bs : List Nat} {sz : MemSize} (hwf : m.wf)
    (hD1p : rdMem m D1 .x8 = some b1) (hD1l : rdMem m (D1 + 8) .x8 = some bs.length)
    (hB1 : Buf m b1 bs) (hdisj1 : b1 + bs.length ≤ D1 ∨ D1 + 16 ≤ b1)
    (hD2p : rdMem m D2 .x8 = some b2) (hD2l : rdMem m (D2 + 8) .x8 = some bs.length)
    (hB2 : Buf m b2 bs) (hdisj2 : b2 + bs.length ≤ D2 ∨ D2 + 16 ≤ b2)
    (hdisj : ∀ a, inFootprint D1 b1 bs.length a → ¬ inFootprint D2 b2 bs.length a)
    (hrd : rdMem (passSubMem m D1 b1 bs) ea sz = some val)
    (hwr : wrMem m ea v sz = some m') :
    MemSub (passSubMem m D2 b2 bs) m' := by
  have hs2 := passSubMem_spec hD2p hD2l hB2 hdisj2
  have hrd1 := passSubMem_rd_eq hD1p hD1l hB1 hdisj1
  apply hs2.2.2.2
  intro a ha
  apply wrMem_rd_ne hwf hwr
  intro j hj heq
  obtain ⟨x, hx⟩ := rdMem_some_each hrd j hj
  rw [hrd1] at hx
  have hin1 : inFootprint D1 b1 bs.length (ea + j) := by
    unfold inFootprint
    by_cases hD : D1 ≤ ea + j ∧ ea + j < D1 + 16
    · exact Or.inl hD
    · simp only [hD, ↓reduceIte] at hx
      by_cases hB : b1 ≤ ea + j ∧ ea + j < b1 + bs.length
      · exact Or.inr hB
      · simp only [hB, ↓reduceIte] at hx; cases hx
  exact hdisj (ea + j) hin1 (heq ▸ ha)

/-- A fetch-flipped `str xzr, [x14, #128]` (from `cmp x14, #16` at pass offset `16`) writes
`[bs.length + 128, bs.length + 136)`. Because `4096 ≤ bs.length + 128`, `4096 ≤ b_k`, and the
three footprints are pairwise disjoint and disjoint from `[0x600000, 0x603000)`, this 8-byte
write can intersect at most one of `(F1, F2, F3)`, preserving at least two pass sub-memories. -/
private theorem wrMem_cmp16_preserves_two {bs : List Nat} {s : St} {m' : Mem} {v : Nat}
    (ht : TripledState bs s) (hwr : wrMem s.mem (bs.length + 128) v .x8 = some m') :
    16 ≤ bs.length ∧
    let m1 := passSubMem s.mem 0x600000 ht.b1 bs
    let m2 := passSubMem s.mem 0x601000 ht.b2 bs
    let m3 := passSubMem s.mem 0x602000 ht.b3 bs
    (MemSub m2 m' ∧ MemSub m3 m') ∨ (MemSub m1 m' ∧ MemSub m3 m') ∨ (MemSub m1 m' ∧ MemSub m2 m') := by
  have hge4096 : 4096 ≤ bs.length + 128 := wrMem_some_ge_4096 ht.noLowMem hwr
  refine ⟨by omega, ?_⟩
  have hs1 := passSubMem_spec ht.hD1_ptr ht.hD1_len ht.hBuf1 (ht.disjD1 0x600000 (by simp))
  have hs2 := passSubMem_spec ht.hD2_ptr ht.hD2_len ht.hBuf2 (ht.disjD2 0x601000 (by simp))
  have hs3 := passSubMem_spec ht.hD3_ptr ht.hD3_len ht.hBuf3 (ht.disjD3 0x602000 (by simp))
  let hits (D b : Nat) : Prop := ∃ j < 8, inFootprint D b bs.length (bs.length + 128 + j)
  have hsub1 : ¬ hits 0x600000 ht.b1 → MemSub (passSubMem s.mem 0x600000 ht.b1 bs) m' := fun hnot =>
    hs1.2.2.2 _ fun a ha => wrMem_rd_ne ht.memWf hwr a fun j hj heq => hnot ⟨j, hj, heq ▸ ha⟩
  have hsub2 : ¬ hits 0x601000 ht.b2 → MemSub (passSubMem s.mem 0x601000 ht.b2 bs) m' := fun hnot =>
    hs2.2.2.2 _ fun a ha => wrMem_rd_ne ht.memWf hwr a fun j hj heq => hnot ⟨j, hj, heq ▸ ha⟩
  have hsub3 : ¬ hits 0x602000 ht.b3 → MemSub (passSubMem s.mem 0x602000 ht.b3 bs) m' := fun hnot =>
    hs3.2.2.2 _ fun a ha => wrMem_rd_ne ht.memWf hwr a fun j hj heq => hnot ⟨j, hj, heq ▸ ha⟩
  have hpair : ∀ {Da ba Db bb : Nat},
      (Da = 0x600000 ∨ Da = 0x601000 ∨ Da = 0x602000) →
      (Db = 0x600000 ∨ Db = 0x601000 ∨ Db = 0x602000) → Da ≠ Db →
      4096 ≤ ba → 4096 ≤ bb →
      (ba + bs.length ≤ bb ∨ bb + bs.length ≤ ba) →
      (ba + bs.length ≤ 0x600000 ∨ 0x603000 ≤ ba) →
      (bb + bs.length ≤ 0x600000 ∨ 0x603000 ≤ bb) →
      hits Da ba → ¬ hits Db bb := by
    intro Da ba Db bb hDa hDb hne hba hbb hdisj hBssA hBssB ⟨ja, hja, ha⟩ ⟨jb, hjb, hb⟩
    unfold inFootprint at ha hb
    rcases ha with haD | haB <;> rcases hb with hbD | hbB
    · rcases hDa with rfl | rfl | rfl <;> rcases hDb with rfl | rfl | rfl <;> omega
    · rcases hDa with rfl | rfl | rfl <;> rcases hBssB with h | h <;> omega
    · rcases hDb with rfl | rfl | rfl <;> rcases hBssA with h | h <;> omega
    · rcases hdisj with h | h <;> omega
  have hBss1 := ht.disjBss (tripledB1 s) (by simp)
  have hBss2 := ht.disjBss (tripledB2 s) (by simp)
  have hBss3 := ht.disjBss (tripledB3 s) (by simp)
  by_cases h1 : hits 0x600000 ht.b1
  · left
    exact ⟨hsub2 (hpair (Or.inl rfl) (Or.inr (Or.inl rfl)) (by decide) ht.hiB1 ht.hiB2 ht.disj12 hBss1 hBss2 h1),
      hsub3 (hpair (Or.inl rfl) (Or.inr (Or.inr rfl)) (by decide) ht.hiB1 ht.hiB3 ht.disj13 hBss1 hBss3 h1)⟩
  · by_cases h2 : hits 0x601000 ht.b2
    · right; left
      exact ⟨hsub1 h1,
        hsub3 (hpair (Or.inr (Or.inl rfl)) (Or.inr (Or.inr rfl)) (by decide) ht.hiB2 ht.hiB3 ht.disj23 hBss2 hBss3 h2)⟩
    · right; right; exact ⟨hsub1 h1, hsub2 h2⟩

/-! ### Register/flag helpers and post-pass continuation theorems -/

private theorem verdictWf_of_tripled {bs : List Nat} {s : St} (ht : TripledState bs s) :
    VerdictRegsWf s := by
  intro r hr
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl
  · exact ⟨REJECT, by decide, ht.x19⟩
  · exact ⟨REJECT, by decide, ht.x20⟩
  · exact ⟨REJECT, by decide, ht.x21⟩

private theorem flipReg_verdictWf {s : St} (hwf : VerdictRegsWf s) (hl : s.regs.length = 31)
    (i bit : Nat) (hbit : bit < 64) : VerdictRegsWf (s.flipReg i bit) := by
  intro r hr
  have hlt31 : r < 31 := by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
    rcases hr with rfl | rfl | rfl <;> decide
  unfold St.flipReg
  split
  · exact hwf r hr
  · by_cases hir : i = r
    · subst hir
      obtain ⟨u, hu, hru⟩ := hwf i hr
      simp only [hru, Option.map_some]
      refine ⟨Nat.xor u (2 ^ bit), Nat.xor_lt_two_pow hu (Nat.pow_lt_pow_right (by decide) hbit), ?_⟩
      simp [St.reg, St.setReg, hl, hlt31]
    · rw [reg_setReg_ne hir]; exact hwf r hr

private theorem flipReg_len {s : St} (hl : s.regs.length = 31) (i bit : Nat) :
    (s.flipReg i bit).regs.length = 31 := by
  unfold St.flipReg; split
  · exact hl
  · simp [St.setReg, hl]

private theorem flipReg_pc {s : St} (i bit : Nat) : (s.flipReg i bit).pc = s.pc := by
  unfold St.flipReg; split <;> rfl

private theorem flipReg_mem {s : St} (i bit : Nat) : (s.flipReg i bit).mem = s.mem := by
  unfold St.flipReg; split <;> rfl

private theorem flipReg_reg_ne {s : St} {i r : Nat} (hne : i ≠ r) (bit : Nat) :
    (s.flipReg i bit).reg r = s.reg r := by
  unfold St.flipReg; split
  · rfl
  · exact reg_setReg_ne hne

private theorem nzcvFlips_props {s s2 : St} (h : s2 ∈ nzcvFlips s) :
    s2.pc = s.pc ∧ s2.regs = s.regs ∧ s2.mem = s.mem := by
  unfold AArch64Phys.nzcvFlips at h
  simp only [List.mem_append, List.mem_map] at h
  rcases h with ((⟨_, _, rfl⟩ | ⟨_, _, rfl⟩) | ⟨_, _, rfl⟩) | ⟨_, _, rfl⟩ <;> exact ⟨rfl, rfl, rfl⟩

private theorem wrZR_len' (s : St) (w rd v : Nat) : (s.wrZR w rd v).regs.length = s.regs.length := by
  unfold St.wrZR; split
  · rfl
  · simp [St.setReg]

private theorem wrSP_len' (s : St) (w rd v : Nat) : (s.wrSP w rd v).regs.length = s.regs.length := by
  unfold St.wrSP; split
  · rfl
  · simp [St.setReg]

private theorem wrZR_pc' (s : St) (w rd v : Nat) : (s.wrZR w rd v).pc = s.pc := by
  unfold St.wrZR; split <;> rfl

private theorem wrSP_pc' (s : St) (w rd v : Nat) : (s.wrSP w rd v).pc = s.pc := by
  unfold St.wrSP; split <;> rfl

private theorem wrZR_mem' (s : St) (w rd v : Nat) : (s.wrZR w rd v).mem = s.mem := by
  unfold St.wrZR; split <;> rfl

private theorem wrSP_mem' (s : St) (w rd v : Nat) : (s.wrSP w rd v).mem = s.mem := by
  unfold St.wrSP; split <;> rfl

private theorem wrZR_reg_ne'' (s : St) (w rd v r : Nat) (hne : rd ≠ r) :
    (s.wrZR w rd v).reg r = s.reg r := by
  unfold St.wrZR; split
  · rfl
  · exact reg_setReg_ne hne

private theorem wrSP_reg_ne'' (s : St) (w rd v r : Nat) (hne : rd ≠ r) :
    (s.wrSP w rd v).reg r = s.reg r := by
  unfold St.wrSP; split
  · rfl
  · exact reg_setReg_ne hne

private theorem verdictWf_wrZR' {s : St} {w rd v : Nat}
    (hwf : VerdictRegsWf s) (hl : s.regs.length = 31) (hw : wOk w = true) :
    VerdictRegsWf (s.wrZR w rd v) := by
  intro r hr
  unfold St.wrZR; split
  · exact hwf r hr
  · by_cases heq : rd = r
    · subst heq
      have hr31 : rd < 31 := by
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
        rcases hr with rfl | rfl | rfl <;> decide
      have hltW : v % pow2 w < AArch64Env.W := by
        unfold wOk at hw; simp only [Bool.or_eq_true, beq_iff_eq] at hw
        rcases hw with rfl | rfl
        · exact Nat.lt_trans (Nat.mod_lt _ (by decide)) (by decide)
        · unfold AArch64Env.W; rw [← pow2_64]; exact Nat.mod_lt _ (by decide)
      exact ⟨v % pow2 w, hltW, by simp [St.reg, St.setReg, hl, hr31]⟩
    · rw [reg_setReg_ne heq]; exact hwf r hr

private theorem verdictWf_wrSP' {s : St} {w rd v : Nat}
    (hwf : VerdictRegsWf s) (hl : s.regs.length = 31) (hw : wOk w = true) :
    VerdictRegsWf (s.wrSP w rd v) := by
  intro r hr
  unfold St.wrSP; split
  · exact hwf r hr
  · by_cases heq : rd = r
    · subst heq
      have hr31 : rd < 31 := by
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
        rcases hr with rfl | rfl | rfl <;> decide
      have hltW : v % pow2 w < AArch64Env.W := by
        unfold wOk at hw; simp only [Bool.or_eq_true, beq_iff_eq] at hw
        rcases hw with rfl | rfl
        · exact Nat.lt_trans (Nat.mod_lt _ (by decide)) (by decide)
        · unfold AArch64Env.W; rw [← pow2_64]; exact Nat.mod_lt _ (by decide)
      exact ⟨v % pow2 w, hltW, by simp [St.reg, St.setReg, hl, hr31]⟩
    · rw [reg_setReg_ne heq]; exact hwf r hr

private theorem run_im_none_ne_exit0 {s : St} (hnone : im s.pc = none) :
    ∀ n, run im n (.ok s) ≠ .exit 0
  | 0 => by simp [run]
  | _ + 1 => by simp [run, step, hnone, run_stuck]

private theorem run_after_p2_p3 {bs : List Nat} {s : St} (ht : TripledState bs s)
    (hinv : checkFlat bs ≠ some true) {u : St} {m : Nat}
    (hpc : u.pc = L_p2) (hl : u.regs.length = 31) (hwf : VerdictRegsWf u)
    (hsub2 : MemSub (passSubMem s.mem 0x601000 ht.b2 bs) u.mem)
    (hsub3 : MemSub (passSubMem s.mem 0x602000 ht.b3 bs) u.mem) :
    run im m (.ok u) ≠ .exit 0 := by
  intro h0
  have hs2 := passSubMem_spec ht.hD2_ptr ht.hD2_len ht.hBuf2 (ht.disjD2 0x601000 (by simp))
  have hs3 := passSubMem_spec ht.hD3_ptr ht.hD3_len ht.hBuf3 (ht.disjD3 0x602000 (by simp))
  have hD2p : rdMem u.mem 0x601000 .x8 = some ht.b2 := rdMem_memSub hsub2 hs2.1
  have hD2l : rdMem u.mem (0x601000 + 8) .x8 = some bs.length := rdMem_memSub hsub2 hs2.2.1
  have hB2 : Buf u.mem ht.b2 bs := fun j hj => hsub2 _ _ (hs2.2.2.1 j hj)
  have hD3p : rdMem u.mem 0x602000 .x8 = some ht.b3 := rdMem_memSub hsub3 hs3.1
  have hD3l : rdMem u.mem (0x602000 + 8) .x8 = some bs.length := rdMem_memSub hsub3 hs3.2.1
  have hB3 : Buf u.mem ht.b3 bs := fun j hj => hsub3 _ _ (hs3.2.2.1 j hj)
  obtain ⟨k2, u2, hr2, hf2, hpc2, h20_2, hl2, hm2⟩ :=
    p2_spec hD2p hD2l hB2 ht.bytes ht.hSm2 hl hpc rfl
  obtain ⟨k3, u3, hr3, hf3, hpc3, h21_3, hl3, _⟩ :=
    p3_spec hD3p hD3l hB3 ht.bytes ht.hSm3 hl2 hpc2 hm2
  have h20_3 : u3.reg 20 = some (checkFlatWord bs) := by rw [hf3.regs 20 (by decide)]; exact h20_2
  obtain ⟨a19, ha19, hu19⟩ := hwf 19 (by simp)
  have h19_3 : u3.reg 19 = some a19 := by
    rw [hf3.regs 19 (by decide), hf2.regs 19 (by decide)]; exact hu19
  have hreach : run im (k2 + k3) (.ok u) = .ok u3 := by rw [run_add, hr2, hr3]
  have hv_ne := checkFlatWord_ne_accept hinv
  have hne0 := vote_two_ne_accept hl3 hpc3 h19_3 h20_3 h21_3
    ha19 (checkFlatWord_lt_W bs) (checkFlatWord_lt_W bs) (Or.inr (Or.inr ⟨hv_ne, hv_ne⟩))
  by_cases hle : k2 + k3 ≤ m
  · rw [show m = (k2 + k3) + (m - (k2 + k3)) by omega, run_add, hreach] at h0
    exact hne0 (m - (k2 + k3)) h0
  · have h := congrArg (run im ((k2 + k3) - m)) h0
    rw [← run_add, show m + ((k2 + k3) - m) = k2 + k3 by omega, hreach, run_exit] at h; cases h

private theorem run_after_p3_with_19_ne_acc {bs : List Nat} {s : St} (ht : TripledState bs s)
    (hinv : checkFlat bs ≠ some true) {u : St} {m a19 : Nat}
    (hpc : u.pc = L_p3) (hl : u.regs.length = 31) (hwf : VerdictRegsWf u)
    (h19 : u.reg 19 = some a19) (ha19_lt : a19 < AArch64Env.W) (ha19_ne : a19 ≠ ACCEPT)
    (hsub3 : MemSub (passSubMem s.mem 0x602000 ht.b3 bs) u.mem) :
    run im m (.ok u) ≠ .exit 0 := by
  intro h0
  have hs3 := passSubMem_spec ht.hD3_ptr ht.hD3_len ht.hBuf3 (ht.disjD3 0x602000 (by simp))
  have hD3p : rdMem u.mem 0x602000 .x8 = some ht.b3 := rdMem_memSub hsub3 hs3.1
  have hD3l : rdMem u.mem (0x602000 + 8) .x8 = some bs.length := rdMem_memSub hsub3 hs3.2.1
  have hB3 : Buf u.mem ht.b3 bs := fun j hj => hsub3 _ _ (hs3.2.2.1 j hj)
  obtain ⟨k3, u3, hr3, hf3, hpc3, h21_3, hl3, _⟩ :=
    p3_spec hD3p hD3l hB3 ht.bytes ht.hSm3 hl hpc rfl
  have h19_3 : u3.reg 19 = some a19 := by rw [hf3.regs 19 (by decide)]; exact h19
  obtain ⟨a20, ha20, hu20⟩ := hwf 20 (by simp)
  have h20_3 : u3.reg 20 = some a20 := by rw [hf3.regs 20 (by decide)]; exact hu20
  have hne0 := vote_two_ne_accept hl3 hpc3 h19_3 h20_3 h21_3
    ha19_lt ha20 (checkFlatWord_lt_W bs) (Or.inr (Or.inl ⟨ha19_ne, checkFlatWord_ne_accept hinv⟩))
  by_cases hle : k3 ≤ m
  · rw [show m = k3 + (m - k3) by omega, run_add, hr3] at h0
    exact hne0 (m - k3) h0
  · have h := congrArg (run im (k3 - m)) h0
    rw [← run_add, show m + (k3 - m) = k3 by omega, hr3, run_exit] at h; cases h

private theorem run_after_p3_clean_with_19_20_ne_acc {u : St} {m a19 a20 : Nat}
    (hin : inPassOrNxt L_p3 L_p3_end L_vote u.pc ∨ u.pc = L_p3_end)
    (hl : u.regs.length = 31) (hwf : VerdictRegsWf u)
    (h19 : u.reg 19 = some a19) (ha19_lt : a19 < AArch64Env.W) (ha19_ne : a19 ≠ ACCEPT)
    (h20 : u.reg 20 = some a20) (ha20_lt : a20 < AArch64Env.W) (ha20_ne : a20 ≠ ACCEPT) :
    run im m (.ok u) ≠ .exit 0 := by
  rcases hin with hin | heq_end
  · intro h0
    obtain ⟨t, _, uv, _, hpcv, _, hlv, hwfv, hregv, hend⟩ :=
      pass_clean_escape (Or.inr (Or.inr ⟨rfl, rfl, rfl, rfl⟩)) m u hin hl hwf h0
    have h19v : uv.reg 19 = some a19 := (hregv 19 (by simp) (by decide)).trans h19
    have h20v : uv.reg 20 = some a20 := (hregv 20 (by simp) (by decide)).trans h20
    obtain ⟨a21, ha21, h21v⟩ := hwfv 21 (by simp)
    exact vote_two_ne_accept hlv hpcv h19v h20v h21v ha19_lt ha20_lt ha21
      (Or.inl ⟨ha19_ne, ha20_ne⟩) (m - t) hend
  · exact run_im_none_ne_exit0 (heq_end ▸ im_pad1_none (Nat.le_refl _) (by decide)) m

private theorem run_in_corePass_with_19_20_ne_acc {bs : List Nat} {s0 : St} (ht : TripledState bs s0)
    (hinv : checkFlat bs ≠ some true) {u : St} {m a19 a20 : Nat}
    (htgt : targetInPasses u.pc = true) (hl : u.regs.length = 31) (hwf : VerdictRegsWf u)
    (h19 : u.reg 19 = some a19) (ha19_lt : a19 < AArch64Env.W) (ha19_ne : a19 ≠ ACCEPT)
    (h20 : u.reg 20 = some a20) (ha20_lt : a20 < AArch64Env.W) (ha20_ne : a20 ≠ ACCEPT)
    (hsub2 : MemSub (passSubMem s0.mem 0x601000 ht.b2 bs) u.mem)
    (hsub3 : MemSub (passSubMem s0.mem 0x602000 ht.b3 bs) u.mem) :
    run im m (.ok u) ≠ .exit 0 := by
  unfold targetInPasses at htgt
  simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at htgt
  obtain ⟨⟨hlo1, hlt3⟩, hal1⟩ := htgt
  by_cases hp1 : u.pc < L_p2
  · intro h0
    obtain ⟨t, _, u2, _, hpc2, hm2, hl2, hwf2', _, hend⟩ :=
      pass_clean_escape (Or.inl ⟨rfl, rfl, rfl, rfl⟩) m u (Or.inl ⟨hlo1, hp1, hal1⟩) hl hwf h0
    exact run_after_p2_p3 ht hinv hpc2 hl2 hwf2' (hm2.symm ▸ hsub2) (hm2.symm ▸ hsub3) hend
  · by_cases hp2 : u.pc < L_p3
    · intro h0
      have hal2 : (u.pc - L_p2) % 4 = 0 := by
        have h12 : L_p1 + 2728 = L_p2 := by decide
        omega
      obtain ⟨t, _, u3, _, hpc3, hm3, hl3, hwf3', hreg3, hend⟩ :=
        pass_clean_escape (Or.inr (Or.inl ⟨rfl, rfl, rfl, rfl⟩)) m u (Or.inl ⟨by omega, hp2, hal2⟩) hl hwf h0
      exact run_after_p3_with_19_ne_acc ht hinv hpc3 hl3 hwf3'
        ((hreg3 19 (by simp) (by decide)).trans h19) ha19_lt ha19_ne (hm3.symm ▸ hsub3) hend
    · have hal3 : (u.pc - L_p3) % 4 = 0 := by
        have h13 : L_p1 + 5456 = L_p3 := by decide
        omega
      exact run_after_p3_clean_with_19_20_ne_acc (Or.inl (Or.inl ⟨by omega, hlt3, hal3⟩))
        hl hwf h19 ha19_lt ha19_ne h20 ha20_lt ha20_ne

/-! ### Synchronization after `cmp x14, #16` → `str xzr, [x14, #128]` at pass offset `16` -/

private theorem pass_off20_sync {lo hi nxt R Nrem : Nat}
    (h12 : (lo = L_p1 ∧ hi = L_p2 ∧ nxt = L_p2 ∧ R = 19) ∨
           (lo = L_p2 ∧ hi = L_p3 ∧ nxt = L_p3 ∧ R = 20))
    {m_k m' : Mem} (hsub : MemSub m_k m')
    {u1 s_end s' : St}
    (hu1_pc : u1.pc = lo + 20) (hu1_mem : u1.mem = m_k)
    (hu1_flc : u1.fl.c = some true)
    (hs'_pc : s'.pc = lo + 20) (hs'_regs : s'.regs = u1.regs) (hs'_sp : s'.sp = u1.sp)
    (hs'_k : s'.k = u1.k) (hs'_mem : s'.mem = m')
    (hl : u1.regs.length = 31) (hwf : VerdictRegsWf u1)
    (hrem : run im Nrem (.ok u1) = .ok s_end) (hend_pc : s_end.pc = nxt)
    (hstay : ∀ t < Nrem, ∃ u, run im t (.ok u1) = .ok u ∧ lo ≤ u.pc ∧ u.pc < hi ∧
      (u.pc - lo) % 4 = 0 ∧ u.regs.length = 31 ∧ VerdictRegsWf u) :
    (∃ k u_end, run im k (.ok s') = .ok u_end ∧ u_end.pc = nxt ∧ u_end.mem = m' ∧
      u_end.regs.length = 31 ∧ VerdictRegsWf u_end ∧
      (u_end = { s_end with mem := m' } ∨
        (u_end.reg R = some REJECT ∧ ∀ r, r ≠ R → u_end.reg r = u1.reg r))) ∨
    (∀ m, run im m (.ok s') ≠ .exit 0) := by
  have hpass : PassSpec lo hi nxt R := by
    rcases h12 with ⟨rfl, rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl, rfl⟩
    · exact Or.inl ⟨rfl, rfl, rfl, rfl⟩
    · exact Or.inr (Or.inl ⟨rfl, rfl, rfl, rfl⟩)
  have hloW : lo + 2728 < 2 ^ 64 := by
    rcases h12 with ⟨rfl, _, _, _⟩ | ⟨rfl, _, _, _⟩ <;> decide
  have hnxtW : nxt < 2 ^ 64 := by
    rcases h12 with ⟨_, _, rfl, _⟩ | ⟨_, _, rfl, _⟩ <;> decide
  have hnxt_ge : lo + 2724 ≤ nxt := by
    rcases h12 with ⟨rfl, _, rfl, _⟩ | ⟨rfl, _, rfl, _⟩ <;> decide
  have hR_lt : R < 31 := by
    rcases h12 with ⟨_, _, _, rfl⟩ | ⟨_, _, _, rfl⟩ <;> decide
  have hcode :
      im (lo + 20) = some (.bcond 3 (relOff (lo + 20) (lo + 2708)), 4) ∧
      im (lo + 24) = some (.ldr .w4 14 0 0, 4) ∧
      im (lo + 28) = some (.movWide 2 64 15 20563 0, 4) ∧
      im (lo + 32) = some (.movWide 3 64 15 12626 1, 4) ∧
      im (lo + 36) = some (.addSubReg true true 64 31 14 15 .lsl 0, 4) ∧
      CodeAt (verdictInsns (lo + 2708) R REJECT) ∧
      im (lo + 2724) = some (.b false (relOff (lo + 2724) nxt), 4) := by
    rcases h12 with ⟨rfl, rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl, rfl⟩
    · have pb := passBlocks_of_codeAt p1_codeAt
      have hH := pb.cHdr; simp only [CodeAt_cons] at hH
      have hRB := pb.cRejB; simp only [CodeAt_cons] at hRB
      exact ⟨hH.2.2.2.2.2.1, hH.2.2.2.2.2.2.1, hH.2.2.2.2.2.2.2.1, hH.2.2.2.2.2.2.2.2.1,
        hH.2.2.2.2.2.2.2.2.2.1, pb.cRej, hRB.1⟩
    · have pb := passBlocks_of_codeAt p2_codeAt
      have hH := pb.cHdr; simp only [CodeAt_cons] at hH
      have hRB := pb.cRejB; simp only [CodeAt_cons] at hRB
      exact ⟨hH.2.2.2.2.2.1, hH.2.2.2.2.2.2.1, hH.2.2.2.2.2.2.2.1, hH.2.2.2.2.2.2.2.2.1,
        hH.2.2.2.2.2.2.2.2.2.1, pb.cRej, hRB.1⟩
  obtain ⟨f20, f24, f28, f32, f36, hRej, hRejB⟩ := hcode
  cases hc : AArch64Phys.cond s'.fl 3 with
  | none =>
    right; intro m
    cases m with
    | zero => simp [run]
    | succ m' =>
      have hst : step im s' = .stuck := by
        simp [step, hs'_pc, f20, exec, lift, bind, Option.bind, hc]
      simp [run, hst, run_stuck]
  | some b =>
    cases b with
    | true =>
      left
      have hrej_pc : (lo + 20 + relOff (lo + 20) (lo + 2708)) % AArch64Env.W = lo + 2708 := by
        rcases h12 with ⟨rfl, _, _, _⟩ | ⟨rfl, _, _, _⟩ <;> decide
      have hst0 : step im s' = .ok { s' with pc := lo + 2708 } := by
        simp [step, hs'_pc, f20, exec, lift, bind, Option.bind, pure, hc, hrej_pc]
      obtain ⟨kv, u_end, hrv, hfv, hpcv, hRv, hlv⟩ :=
        verdict_spec hRej hRejB (by omega) hnxtW hnxt_ge (by decide) hR_lt (hs'_regs ▸ hl)
          (s := { s' with pc := lo + 2708 }) rfl
      have hwf_end : VerdictRegsWf u_end := by
        intro r hr
        by_cases hrR : r = R
        · subst hrR; exact ⟨REJECT, by decide, hRv⟩
        · obtain ⟨vr, hvr_lt, hvr⟩ := hwf r hr
          refine ⟨vr, hvr_lt, ?_⟩
          rw [hfv.regs r (by simp [hrR])]
          change s'.reg r = some vr; simp [St.reg, hs'_regs]; exact hvr
      refine ⟨kv + 1, u_end, by rw [show kv + 1 = 1 + kv by omega, run_add]; simp [run, hst0, hrv],
        hpcv, hfv.mem.trans hs'_mem, hlv, hwf_end, Or.inr ⟨hRv, fun r hne => ?_⟩⟩
      rw [hfv.regs r (by simp [hne])]
      change s'.reg r = u1.reg r; simp [St.reg, hs'_regs]
    | false =>
      left
      have hcu1 : AArch64Phys.cond u1.fl 3 = some false := by
        simp [AArch64Phys.cond, hu1_flc]
      have h24 : (lo + 20 + 4) % AArch64Env.W = lo + 24 := modW_small (by omega)
      have h28 : (lo + 24 + 4) % AArch64Env.W = lo + 28 := modW_small (by omega)
      have h32 : (lo + 28 + 4) % AArch64Env.W = lo + 32 := modW_small (by omega)
      have h36 : (lo + 32 + 4) % AArch64Env.W = lo + 36 := modW_small (by omega)
      have h40 : (lo + 36 + 4) % AArch64Env.W = lo + 40 := modW_small (by omega)
      -- Step 1 (`lo + 20`): both fall through to `lo + 24`
      have hs1_u : step im u1 = .ok { u1 with pc := lo + 24 } := by
        simp [step, hu1_pc, f20, exec, lift, bind, Option.bind, pure, hcu1, h24]
      have hs1_s : step im s' = .ok { s' with pc := lo + 24 } := by
        simp [step, hs'_pc, f20, exec, lift, bind, Option.bind, pure, hc, h24]
      have hN1 : 1 ≤ Nrem := by
        cases Nrem with
        | zero => simp [run] at hrem; subst hrem; omega
        | succ _ => omega
      have hrem1 : run im (Nrem - 1) (.ok { u1 with pc := lo + 24 }) = .ok s_end := by
        have := hrem; rw [show Nrem = 1 + (Nrem - 1) by omega, run_add] at this
        simpa [run, hs1_u] using this
      -- Step 2 (`lo + 24`): `ldr w14, [x0, #0]`
      have hN2 : 2 ≤ Nrem := by
        by_cases h1 : Nrem - 1 = 0
        · rw [h1] at hrem1; simp [run] at hrem1; subst hrem1
          have hpc_eq : lo + 24 = nxt := hend_pc; omega
        · omega
      cases hx0 : u1.rdSP 64 0 with
      | none =>
        have hx0' : ({ u1 with pc := lo + 24 }).rdSP 64 0 = none := hx0
        have hst : step im { u1 with pc := lo + 24 } = .stuck := by
          simp [step, f24, exec, lift, bind, Option.bind, hx0']
        rw [show Nrem - 1 = 1 + (Nrem - 2) by omega, run_add] at hrem1
        simp [run, hst, run_stuck] at hrem1
      | some base0 =>
        have hx0' : ({ u1 with pc := lo + 24, mem := m_k }).rdSP 64 0 = some base0 := hx0
        cases hrd0 : rdMem m_k (base0 % AArch64Env.W) .w4 with
        | none =>
          have hst : step im { u1 with pc := lo + 24 } = .stuck := by
            simp [step, f24, exec, lift, bind, Option.bind, hu1_mem, hx0', hrd0]
          rw [show Nrem - 1 = 1 + (Nrem - 2) by omega, run_add] at hrem1
          simp [run, hst, run_stuck] at hrem1
        | some w0 =>
          have hx0_s : ({ s' with pc := lo + 24, mem := m' }).rdSP 64 0 = some base0 := by
            simpa [St.rdSP, St.reg, hs'_regs] using hx0
          have hrd0' : rdMem m' (base0 % AArch64Env.W) .w4 = some w0 := rdMem_memSub hsub hrd0
          let u2 := ({ u1 with pc := lo + 28 }).wrZR 32 14 w0
          let s2 := ({ s' with pc := lo + 28 }).wrZR 32 14 w0
          have hs2_u : step im { u1 with pc := lo + 24 } = .ok u2 := by
            have : step im { u1 with pc := lo + 24, mem := m_k } = .ok ({ u1 with pc := lo + 28, mem := m_k }.wrZR 32 14 w0) := by
              simp [step, f24, exec, lift, bind, Option.bind, pure, hx0', hrd0, h28, MemSize.width]
            simpa [hu1_mem, u2] using this
          have hs2_s : step im { s' with pc := lo + 24 } = .ok s2 := by
            have : step im { s' with pc := lo + 24, mem := m' } = .ok ({ s' with pc := lo + 28, mem := m' }.wrZR 32 14 w0) := by
              simp [step, f24, exec, lift, bind, Option.bind, pure, hx0_s, hrd0', h28, MemSize.width]
            simpa [hs'_mem, s2] using this
          have hrem2 : run im (Nrem - 2) (.ok u2) = .ok s_end := by
            rw [show Nrem - 1 = 1 + (Nrem - 2) by omega, run_add] at hrem1
            simpa [run, hs2_u] using hrem1
          -- Step 3 (`lo + 28`): `movz x15, #20563`
          have hN3 : 3 ≤ Nrem := by
            by_cases h2 : Nrem - 2 = 0
            · rw [h2] at hrem2; simp [run] at hrem2; subst hrem2
              have : u2.pc = lo + 28 := by simp [u2]
              omega
            · omega
          let u3 := ({ u2 with pc := lo + 32 }).wrZR 64 15 (20563 * pow2 (0 * 16))
          let s3 := ({ s2 with pc := lo + 32 }).wrZR 64 15 (20563 * pow2 (0 * 16))
          have hu2_pc : u2.pc = lo + 28 := by simp [u2]
          have hs2_pc : s2.pc = lo + 28 := by simp [s2]
          have hs3_u : step im u2 = .ok u3 := by
            simp [step, hu2_pc, f28, exec, h32, u3]
          have hs3_s : step im s2 = .ok s3 := by
            simp [step, hs2_pc, f28, exec, h32, s3]
          have hrem3 : run im (Nrem - 3) (.ok u3) = .ok s_end := by
            rw [show Nrem - 2 = 1 + (Nrem - 3) by omega, run_add] at hrem2
            simpa [run, hs3_u] using hrem2
          -- Step 4 (`lo + 32`): `movk x15, #12626, lsl #16`
          have hN4 : 4 ≤ Nrem := by
            by_cases h3 : Nrem - 3 = 0
            · rw [h3] at hrem3; simp [run] at hrem3; subst hrem3
              have : u3.pc = lo + 32 := by simp [u3]
              omega
            · omega
          have hu3_pc : u3.pc = lo + 32 := by simp [u3]
          have hs3_pc : s3.pc = lo + 32 := by simp [s3]
          have hs3_regs : s3.regs = u3.regs := by simp [s3, u3, s2, u2, St.wrZR, St.setReg, hs'_regs]
          cases h15 : u3.rdZR 64 15 with
          | none =>
            have hst : step im u3 = .stuck := by
              simp [step, hu3_pc, f32, exec, lift, bind, Option.bind, h15]
            rw [show Nrem - 3 = 1 + (Nrem - 4) by omega, run_add] at hrem3
            simp [run, hst, run_stuck] at hrem3
          | some old15 =>
            have h15_s : s3.rdZR 64 15 = some old15 := by simpa [St.rdZR, St.reg, hs3_regs] using h15
            let val15 := old15 % pow2 (1 * 16) + 12626 * pow2 (1 * 16) + old15 / pow2 (1 * 16 + 16) * pow2 (1 * 16 + 16)
            let u4 := ({ u3 with pc := lo + 36 }).wrZR 64 15 val15
            let s4 := ({ s3 with pc := lo + 36 }).wrZR 64 15 val15
            have hs4_u : step im u3 = .ok u4 := by
              simp [step, hu3_pc, f32, exec, lift, bind, Option.bind, pure, h15, h36, u4, val15]
            have hs4_s : step im s3 = .ok s4 := by
              simp [step, hs3_pc, f32, exec, lift, bind, Option.bind, pure, h15_s, h36, s4, val15]
            have hrem4 : run im (Nrem - 4) (.ok u4) = .ok s_end := by
              rw [show Nrem - 3 = 1 + (Nrem - 4) by omega, run_add] at hrem3
              simpa [run, hs4_u] using hrem3
            -- Step 5 (`lo + 36`): `cmp x14, x15` overwrites `fl`, syncing `s5 = { u5 with mem := m' }`!
            have hN5 : 5 ≤ Nrem := by
              by_cases h4 : Nrem - 4 = 0
              · rw [h4] at hrem4; simp [run] at hrem4; subst hrem4
                have : u4.pc = lo + 36 := by simp [u4]
                omega
              · omega
            have hu4_pc : u4.pc = lo + 36 := by simp [u4]
            have hs4_pc : s4.pc = lo + 36 := by simp [s4]
            have hs4_regs : s4.regs = u4.regs := by simp [s4, u4, St.wrZR, St.setReg, hs3_regs]
            have hs4_sp : s4.sp = u4.sp := by simp [s4, u4, s3, u3, s2, u2, St.wrZR, St.setReg, hs'_sp]
            have hs4_k : s4.k = u4.k := by simp [s4, u4, s3, u3, s2, u2, St.wrZR, St.setReg, hs'_k]
            have hu4_mem : u4.mem = m_k := by simp [u4, u3, u2, hu1_mem]
            have hs4_mem : s4.mem = m' := by simp [s4, s3, s2, hs'_mem]
            cases hr14 : u4.rdZR 64 14 with
            | none =>
              have hst : step im u4 = .stuck := by
                simp [step, hu4_pc, f36, exec, lift, bind, Option.bind, hr14]
              rw [show Nrem - 4 = 1 + (Nrem - 5) by omega, run_add] at hrem4
              simp [run, hst, run_stuck] at hrem4
            | some a14 =>
              cases hr15 : u4.rdZR 64 15 with
              | none =>
                have hst : step im u4 = .stuck := by
                  simp [step, hu4_pc, f36, exec, lift, bind, Option.bind, hr14, hr15]
                rw [show Nrem - 4 = 1 + (Nrem - 5) by omega, run_add] at hrem4
                simp [run, hst, run_stuck] at hrem4
              | some b15 =>
                have hr14_s : s4.rdZR 64 14 = some a14 := by simpa [St.rdZR, St.reg, hs4_regs] using hr14
                have hr15_s : s4.rdZR 64 15 = some b15 := by simpa [St.rdZR, St.reg, hs4_regs] using hr15
                let flNew := (addSubOp true 64 a14 (shiftVal .lsl 64 b15 0)).2
                let u5 := { u4 with pc := lo + 40, fl := flNew }
                have hs5_u : step im u4 = .ok u5 := by
                  simp [step, hu4_pc, f36, exec, lift, bind, Option.bind, pure, hr14, hr15, h40, St.wrZR, u5, flNew]
                have hs5_s : step im s4 = .ok { u5 with mem := m' } := by
                  have hseq : { s4 with pc := lo + 40, fl := flNew } = { u5 with mem := m' } := by
                    show St.mk s4.regs s4.sp (lo + 40) flNew s4.mem s4.k = St.mk u4.regs u4.sp (lo + 40) flNew m' u4.k
                    rw [hs4_regs, hs4_sp, hs4_mem, hs4_k]
                  simpa [step, hs4_pc, f36, exec, lift, bind, Option.bind, pure, hr14_s, hr15_s, h40, St.wrZR, flNew] using
                    congrArg Out.ok hseq
                have hr5_u : run im 5 (.ok u1) = .ok u5 := by
                  change run im 4 (step im u1) = _; rw [hs1_u]
                  change run im 3 (step im _) = _; rw [hs2_u]
                  change run im 2 (step im _) = _; rw [hs3_u]
                  change run im 1 (step im _) = _; rw [hs4_u]
                  change run im 0 (step im _) = _; rw [hs5_u]; rfl
                have hr5_s : run im 5 (.ok s') = .ok { u5 with mem := m' } := by
                  change run im 4 (step im s') = _; rw [hs1_s]
                  change run im 3 (step im _) = _; rw [hs2_s]
                  change run im 2 (step im _) = _; rw [hs3_s]
                  change run im 1 (step im _) = _; rw [hs4_s]
                  change run im 0 (step im _) = _; rw [hs5_s]; rfl
                have hrem5 : run im (Nrem - 5) (.ok u5) = .ok s_end := by
                  rw [show Nrem - 4 = 1 + (Nrem - 5) by omega, run_add] at hrem4
                  simpa [run, hs5_u] using hrem4
                have hstay5 : ∀ t < Nrem - 5, ∃ u, run im t (.ok u5) = .ok u ∧ lo ≤ u.pc ∧ u.pc < hi ∧
                    (u.pc - lo) % 4 = 0 ∧ u.regs.length = 31 ∧ VerdictRegsWf u := by
                  intro t ht'
                  obtain ⟨u, hu, h1, h2, h3, h4, h5⟩ := hstay (5 + t) (by omega)
                  rw [run_add, hr5_u] at hu
                  exact ⟨u, hu, h1, h2, h3, h4, h5⟩
                have hl_u5 : u5.regs.length = 31 := by
                  simp [u5, u4, u3, u2, hl]
                have hwf_u5 : VerdictRegsWf u5 := by
                  have hwf2 : VerdictRegsWf u2 := verdictWf_wrZR' (s := { u1 with pc := lo + 28 }) hwf hl (by decide)
                  have hwf3 : VerdictRegsWf u3 := verdictWf_wrZR' (s := { u2 with pc := lo + 32 }) hwf2 (by simp [u2, hl]) (by decide)
                  exact verdictWf_wrZR' (s := { u3 with pc := lo + 36 }) hwf3 (by simp [u3, u2, hl]) (by decide)
                have ⟨_, hrun_tail⟩ := pass_run_memSub hpass hsub (Nrem - 5) u5 s_end hu4_mem hl_u5 hwf_u5 hrem5 hstay5
                have htot : run im Nrem (.ok s') = .ok { s_end with mem := m' } := by
                  rw [show Nrem = 5 + (Nrem - 5) by omega, run_add, hr5_s, hrun_tail]
                -- `s_end` has length 31 and VerdictRegsWf via last step (or Nrem ≥ 5 -> step Nrem - 1)
                have hlen_wf_end : s_end.regs.length = 31 ∧ VerdictRegsWf s_end := by
                  obtain ⟨ul, hul, hlo_l, hlt_l, hal_l, hl_l, hwf_l⟩ := hstay (Nrem - 1) (by omega)
                  have hlast : step im ul = .ok s_end := by
                    have := hrem; rw [show Nrem = (Nrem - 1) + 1 by omega, run_add, hul] at this
                    simpa [run] using this
                  cases him_l : im ul.pc with
                  | none => simp [step, him_l] at hlast
                  | some p =>
                    obtain ⟨ins_l, len_l⟩ := p
                    obtain ⟨rfl, hsafe_l⟩ := im_in_pass_safe hpass hlo_l hlt_l him_l
                    have hhiW : hi < AArch64Env.W := by
                      rcases h12 with ⟨_, rfl, _, _⟩ | ⟨_, rfl, _, _⟩ <;> decide
                    have ⟨_, hprops_l, _, _⟩ := exec_pass_props hsafe_l hlo_l hlt_l hhiW hal_l hl_l hwf_l
                    have ⟨_, hle, _, hwfe, _⟩ := hprops_l s_end (by simpa [step, him_l] using hlast)
                    exact ⟨hle, hwfe⟩
                exact ⟨Nrem, { s_end with mem := m' }, htot, hend_pc, rfl, hlen_wf_end.1, hlen_wf_end.2, Or.inl rfl⟩

/-! ### Decoded 1-bit fetch step classification over `[L_p1, L_p3_end)` -/

private theorem lift_ne_exit' : ∀ (o : Option St) (c : Nat), lift o ≠ .exit c
  | none, _ => by simp [lift]
  | some _, _ => by simp [lift]

private theorem exec_passFetch_ne_exit {pc origW altW : Nat} {ins' : Ins} {s : St}
    (halt : passFetchAltOk pc origW altW = true) (hdec : decodeWord altW = some ins') :
    ∀ c, exec ins' 4 s ≠ .exit c := by
  intro c hx
  unfold passFetchAltOk at halt; rw [hdec] at halt
  cases ins' with
  | svc _ => simp at halt
  | movWide kind _ _ _ _ =>
    simp only [exec] at hx
    rcases kind with _ | _ | _ | _ | _ <;> first | cases hx | exact lift_ne_exit' _ c hx
  | adr _ _ _ => simp [exec] at hx
  | b _ _ => simp [exec] at hx
  | addSubImm _ _ _ _ _ _ | addSubReg _ _ _ _ _ _ _ _ | logicReg _ _ _ _ _ _ _ _
  | bitfield _ _ _ _ _ _ | ccmpReg _ _ _ _ _ _ | csel _ _ _ _ _ _
  | ldr _ _ _ _ | str _ _ _ _ | ldur _ _ _ _ | stur _ _ _ _
  | ldrReg _ _ _ _ _ _ | strReg _ _ _ _ _ _ | bcond _ _ | cbz _ _ _ _
  | tbz _ _ _ _ | br _ _ | ret _ =>
    exact lift_ne_exit' _ c hx

/-- Unified step classification for any 1-bit fetch-perturbed instruction `ins'` accepted by
`passFetchAltOk` inside `[L_p1, L_p3_end)`. -/
private theorem exec_passFetch_step {origW altW a19 a20 : Nat} {ins' : Ins} {s s' : St}
    (_hlo : L_p1 ≤ s.pc) (hhi : s.pc < L_p3_end)
    (hl : s.regs.length = 31) (hwf : VerdictRegsWf s)
    (h19 : s.reg 19 = some a19) (ha19_lt : a19 < AArch64Env.W) (ha19_ne : a19 ≠ ACCEPT)
    (h20 : s.reg 20 = some a20) (ha20_lt : a20 < AArch64Env.W) (ha20_ne : a20 ≠ ACCEPT)
    (h20_p3 : L_p3 ≤ s.pc → a20 = REJECT ∨ a20 = OVERFLOW)
    (halt : passFetchAltOk s.pc origW altW = true) (hdec : decodeWord altW = some ins')
    (hx : exec ins' 4 s = .ok s') :
    s'.regs.length = 31 ∧ VerdictRegsWf s' ∧
    ( -- Case 1: memory-preserving fallthrough to `s.pc + 4`
      (s'.pc = s.pc + 4 ∧ s'.mem = s.mem ∧
        (L_p2 ≤ s.pc → ∃ b19 < AArch64Env.W, s'.reg 19 = some b19 ∧ b19 ≠ ACCEPT) ∧
        (L_p3 ≤ s.pc → ∃ b20 < AArch64Env.W, s'.reg 20 = some b20 ∧ b20 ≠ ACCEPT)) ∨
      -- Case 2: memory-preserving branch keeping all verdict registers unchanged
      (s'.mem = s.mem ∧ s'.reg 19 = s.reg 19 ∧ s'.reg 20 = s.reg 20 ∧ s'.reg 21 = s.reg 21 ∧
        (targetInPasses s'.pc = true ∨ targetImNone s'.pc = true ∨
          (L_p3 ≤ s.pc ∧ targetInTailK01248 s'.pc = true))) ∨
      -- Case 3: Pass 3 store (preserves all registers)
      (L_p3 ≤ s.pc ∧ s'.pc = s.pc + 4 ∧ s'.regs = s.regs) ∨
      -- Case 4: Pass 1/2 store matching either a clean load or a clean `cmp`
      (s.pc < L_p3 ∧ s'.pc = s.pc + 4 ∧ s'.regs = s.regs ∧ s'.sp = s.sp ∧ s'.k = s.k ∧ s'.fl = s.fl ∧
        ((∃ sz rt rn off rt0 base v,
            ((ins' = .str sz rt rn off ∧ decodeWord origW = some (.ldr sz rt0 rn off)) ∨
             (ins' = .stur sz rt rn off ∧ decodeWord origW = some (.ldur sz rt0 rn off))) ∧
            s.rdSP 64 rn = some base ∧ ¬ (rn == 31 && base % 16 != 0) = true ∧
            wrMem s.mem ((base + off) % AArch64Env.W) v sz = some s'.mem) ∨
         (∃ rt rn imm12 base v,
            ins' = .str .x8 rt rn (imm12 * 8) ∧
            decodeWord origW = some (.addSubImm true true 64 31 rn imm12) ∧
            imm12 ≤ 140 ∧ s.rdSP 64 rn = some base ∧
            wrMem s.mem ((base + imm12 * 8) % AArch64Env.W) v .x8 = some s'.mem ∧
            let off := s.pc - (if s.pc < L_p2 then L_p1 else L_p2)
            ((off = 16 ∧ rn = 14 ∧ imm12 = 16) ∨ (inR14 off = true ∧ rn = 14) ∨ (inR15 off = true ∧ rn = 15)))))) := by
  have hpcW : s.pc + 4 < AArch64Env.W := by unfold L_p3_end AArch64Env.W at *; omega
  have hmod4 : (s.pc + 4) % AArch64Env.W = s.pc + 4 := Nat.mod_eq_of_lt hpcW
  unfold passFetchAltOk at halt; rw [hdec] at halt
  dsimp only at halt
  have hregSafe19 : ∀ {rd : Nat},
      ((if s.pc < L_p2 then 1 else if s.pc < L_p3 then 2 else 3) == 1 ||
       (rd != 19 && ((if s.pc < L_p2 then 1 else if s.pc < L_p3 then 2 else 3) == 2 || rd != 20))) = true →
      L_p2 ≤ s.pc → rd ≠ 19 := by
    intro rd h hge
    have hnot1 : ¬ s.pc < L_p2 := by omega
    simp only [hnot1, ↓reduceIte, Bool.or_eq_true, Bool.and_eq_true, beq_iff_eq, bne_iff_ne] at h
    rcases h with h1 | ⟨hne19, _⟩
    · split at h1 <;> omega
    · exact hne19
  have hregSafe20 : ∀ {rd : Nat},
      ((if s.pc < L_p2 then 1 else if s.pc < L_p3 then 2 else 3) == 1 ||
       (rd != 19 && ((if s.pc < L_p2 then 1 else if s.pc < L_p3 then 2 else 3) == 2 || rd != 20))) = true →
      L_p3 ≤ s.pc → rd ≠ 20 := by
    intro rd h hge
    have hnot1 : ¬ s.pc < L_p2 := by unfold L_p2 L_p3 at *; omega
    have hnot2 : ¬ s.pc < L_p3 := by omega
    simp only [hnot1, hnot2, ↓reduceIte, Bool.or_eq_true, Bool.and_eq_true, beq_iff_eq, bne_iff_ne] at h
    rcases h with h1 | ⟨_, h2 | hne20⟩
    · omega
    · omega
    · exact hne20
  have hbr_unpack : ∀ {offB : Nat},
      (targetInPasses ((s.pc + offB) % AArch64Env.W) || targetImNone ((s.pc + offB) % AArch64Env.W) ||
        ((if s.pc < L_p2 then 1 else if s.pc < L_p3 then 2 else 3) == 3 &&
          targetInTailK01248 ((s.pc + offB) % AArch64Env.W))) = true →
      targetInPasses ((s.pc + offB) % AArch64Env.W) = true ∨
      targetImNone ((s.pc + offB) % AArch64Env.W) = true ∨
      (L_p3 ≤ s.pc ∧ targetInTailK01248 ((s.pc + offB) % AArch64Env.W) = true) := by
    intro offB hb
    simp only [Bool.or_eq_true, Bool.and_eq_true, beq_iff_eq] at hb
    rcases hb with (h1 | h2) | ⟨h3, h4⟩
    · exact Or.inl h1
    · exact Or.inr (Or.inl h2)
    · refine Or.inr (Or.inr ⟨?_, h4⟩)
      by_cases hge : L_p3 ≤ s.pc
      · exact hge
      · exfalso
        by_cases h1 : s.pc < L_p2
        · simp [h1] at h3
        · simp [h1, show s.pc < L_p3 by omega] at h3
  cases ins' with
  | svc _ | br _ _ | ret _ | strReg _ _ _ _ _ _ => cases halt
  | b link offB =>
    simp only [exec] at hx; cases hx
    cases link with
    | false =>
      exact ⟨hl, hwf, Or.inr (Or.inl ⟨rfl, rfl, rfl, rfl, hbr_unpack halt⟩)⟩
    | true =>
      refine ⟨by simp [St.setReg, hl], fun r hr => ?_, Or.inr (Or.inl ⟨rfl, ?_, ?_, ?_, hbr_unpack halt⟩)⟩
      · have hne30 : (30 : Nat) ≠ r := by
          simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
          rcases hr with rfl | rfl | rfl <;> decide
        obtain ⟨u, hu, hru⟩ := hwf r hr
        exact ⟨u, hu, (reg_setReg_ne hne30).trans hru⟩
      · exact reg_setReg_ne (by decide)
      · exact reg_setReg_ne (by decide)
      · exact reg_setReg_ne (by decide)
  | bcond cc offB =>
    simp only [exec, lift, bind, Option.bind, pure] at hx
    cases hc : cond s.fl cc <;> simp [hc] at hx
    cases ‹Bool› <;> simp only [Bool.false_eq_true, ↓reduceIte] at hx <;> cases hx
    · exact ⟨hl, hwf, Or.inl ⟨hmod4, rfl, fun _ => ⟨a19, ha19_lt, h19, ha19_ne⟩, fun _ => ⟨a20, ha20_lt, h20, ha20_ne⟩⟩⟩
    · exact ⟨hl, hwf, Or.inr (Or.inl ⟨rfl, rfl, rfl, rfl, hbr_unpack halt⟩)⟩
  | cbz nz w rt offB =>
    rw [Bool.and_eq_true] at halt
    simp only [exec, lift, bind, Option.bind, pure] at hx
    cases hr : s.rdZR w rt with
    | none => simp [hr] at hx
    | some v =>
      simp only [hr] at hx
      by_cases hbr : (if nz = true then v != 0 else v == 0) = true
      · simp only [hbr, ↓reduceIte] at hx; cases hx
        exact ⟨hl, hwf, Or.inr (Or.inl ⟨rfl, rfl, rfl, rfl, hbr_unpack halt.2⟩)⟩
      · rw [if_neg hbr] at hx; cases hx
        exact ⟨hl, hwf, Or.inl ⟨hmod4, rfl, fun _ => ⟨a19, ha19_lt, h19, ha19_ne⟩, fun _ => ⟨a20, ha20_lt, h20, ha20_ne⟩⟩⟩
  | tbz nz bit rt offB =>
    simp only [exec, lift, bind, Option.bind, pure] at hx
    cases hr : s.rdZR 64 rt with
    | none => simp [hr] at hx
    | some v =>
      simp only [hr] at hx
      by_cases hbr : (if nz = true then v / pow2 bit % 2 == 1 else !(v / pow2 bit % 2 == 1)) = true
      · simp only [hbr, ↓reduceIte] at hx; cases hx
        exact ⟨hl, hwf, Or.inr (Or.inl ⟨rfl, rfl, rfl, rfl, hbr_unpack halt⟩)⟩
      · rw [if_neg hbr] at hx; cases hx
        exact ⟨hl, hwf, Or.inl ⟨hmod4, rfl, fun _ => ⟨a19, ha19_lt, h19, ha19_ne⟩, fun _ => ⟨a20, ha20_lt, h20, ha20_ne⟩⟩⟩
  | movWide kind w rd imm16 hw =>
    rw [Bool.and_eq_true] at halt
    obtain ⟨hwok, hsafe_or⟩ := halt
    rw [Bool.or_eq_true] at hsafe_or
    have hfinish : ∀ (v : Nat), s' = ({ s with pc := (s.pc + 4) % AArch64Env.W }).wrZR w rd v →
        (L_p3 ≤ s.pc → rd = 20 → w = 64 → v % pow2 64 ≠ ACCEPT) →
        s'.regs.length = 31 ∧ VerdictRegsWf s' ∧
        (s'.pc = s.pc + 4 ∧ s'.mem = s.mem ∧
          (L_p2 ≤ s.pc → ∃ b19 < AArch64Env.W, s'.reg 19 = some b19 ∧ b19 ≠ ACCEPT) ∧
          (L_p3 ≤ s.pc → ∃ b20 < AArch64Env.W, s'.reg 20 = some b20 ∧ b20 ≠ ACCEPT)) := by
      intro v hseq hne_acc20
      subst hseq
      refine ⟨by rw [wrZR_len']; exact hl, verdictWf_wrZR' (s := { s with pc := (s.pc + 4) % AArch64Env.W }) hwf hl hwok,
        by rw [wrZR_pc']; exact hmod4, wrZR_mem' _ _ _ _, ?_, ?_⟩
      · intro hp2
        rcases hsafe_or with hsafe | hspec
        · exact ⟨a19, ha19_lt, (wrZR_reg_ne'' _ _ _ _ 19 (hregSafe19 hsafe hp2)).trans h19, ha19_ne⟩
        · simp only [Bool.and_eq_true, beq_iff_eq] at hspec
          have hrd20 : rd = 20 := hspec.1.1.1.1.2
          exact ⟨a19, ha19_lt, (wrZR_reg_ne'' _ _ _ _ 19 (by omega)).trans h19, ha19_ne⟩
      · intro hp3
        rcases hsafe_or with hsafe | hspec
        · exact ⟨a20, ha20_lt, (wrZR_reg_ne'' _ _ _ _ 20 (hregSafe20 hsafe hp3)).trans h20, ha20_ne⟩
        · simp only [Bool.and_eq_true, beq_iff_eq] at hspec
          have hrd20 : rd = 20 := hspec.1.1.1.1.2
          have hw64 : w = 64 := hspec.1.1.1.2
          subst hrd20; subst hw64
          refine ⟨v % pow2 64, by unfold AArch64Env.W; rw [← pow2_64]; exact Nat.mod_lt _ (by decide),
            by simp [St.wrZR, St.setReg, St.reg, hl], hne_acc20 hp3 rfl rfl⟩
    simp only [exec] at hx
    rcases kind with _ | _ | _ | _ | _
    · cases hx
      have hne0 : L_p3 ≤ s.pc → rd = 20 → w = 64 → (pow2 w - 1 - imm16 * pow2 (hw * 16)) % pow2 64 ≠ ACCEPT := by
        intro hp3 hrd20 hw64
        rcases hsafe_or with hsafe | hspec
        · exact absurd hrd20 (hregSafe20 hsafe hp3)
        · subst hw64
          simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hspec
          exact (movWide_x20_ne_accept (h20_p3 hp3) (Or.inl rfl) hspec.1.1.2 hspec.1.2).1 rfl
      have hf := hfinish _ rfl hne0
      exact ⟨hf.1, hf.2.1, Or.inl hf.2.2⟩
    · cases hx
    · cases hx
      have hne2 : L_p3 ≤ s.pc → rd = 20 → w = 64 → (imm16 * pow2 (hw * 16)) % pow2 64 ≠ ACCEPT := by
        intro hp3 hrd20 hw64
        rcases hsafe_or with hsafe | hspec
        · exact absurd hrd20 (hregSafe20 hsafe hp3)
        · simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hspec
          exact (movWide_x20_ne_accept (h20_p3 hp3) (Or.inr (Or.inl rfl)) hspec.1.1.2 hspec.1.2).2.1 rfl
      have hf := hfinish _ rfl hne2
      exact ⟨hf.1, hf.2.1, Or.inl hf.2.2⟩
    · simp only [lift, bind, Option.bind, pure] at hx
      cases h1 : s.rdZR w rd with
      | none => simp [h1] at hx
      | some old =>
        simp only [h1] at hx; cases hx
        have hne3 : L_p3 ≤ s.pc → rd = 20 → w = 64 →
            (old % pow2 (hw * 16) + imm16 * pow2 (hw * 16) +
              old / pow2 (hw * 16 + 16) * pow2 (hw * 16 + 16)) % pow2 64 ≠ ACCEPT := by
          intro hp3 hrd20 hw64
          rcases hsafe_or with hsafe | hspec
          · exact absurd hrd20 (hregSafe20 hsafe hp3)
          · subst hrd20; subst hw64
            simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hspec
            have hold : old = a20 := by
              rw [rdZR_of_reg (by decide) h20] at h1; cases h1
              exact Nat.mod_eq_of_lt (by rw [pow2_64]; unfold AArch64Env.W at ha20_lt; exact ha20_lt)
            subst hold
            exact (movWide_x20_ne_accept (h20_p3 hp3) (Or.inr (Or.inr rfl)) hspec.1.1.2 hspec.1.2).2.2 rfl
        have hf := hfinish _ rfl hne3
        exact ⟨hf.1, hf.2.1, Or.inl hf.2.2⟩
    · cases hx
  | adr isPage rd imm =>
    simp only [exec] at hx; cases hx
    rw [Bool.or_eq_true] at halt
    refine ⟨by rw [wrZR_len']; exact hl,
      verdictWf_wrZR' (s := { s with pc := (s.pc + 4) % AArch64Env.W }) hwf hl (by decide),
      Or.inl ⟨by rw [wrZR_pc']; exact hmod4, wrZR_mem' _ _ _ _, ?_, ?_⟩⟩
    · intro hp2
      rcases halt with hsafe | hneA
      · exact ⟨a19, ha19_lt, (wrZR_reg_ne'' _ _ _ _ 19 (hregSafe19 hsafe hp2)).trans h19, ha19_ne⟩
      · rw [bne_iff_ne] at hneA
        by_cases hrd19 : rd = 19
        · subst hrd19
          have hmodW : ((if isPage then s.pc / 4096 * 4096 else s.pc) + imm) % AArch64Env.W % pow2 64 =
              ((if isPage then s.pc / 4096 * 4096 else s.pc) + imm) % AArch64Env.W := by
            rw [pow2_64]; unfold AArch64Env.W; exact Nat.mod_eq_of_lt (Nat.mod_lt _ (by decide))
          exact ⟨_, Nat.mod_lt _ (by decide), by simp [St.wrZR, St.setReg, St.reg, hl, hmodW], hneA⟩
        · exact ⟨a19, ha19_lt, (wrZR_reg_ne'' _ _ _ _ 19 hrd19).trans h19, ha19_ne⟩
    · intro hp3
      rcases halt with hsafe | hneA
      · exact ⟨a20, ha20_lt, (wrZR_reg_ne'' _ _ _ _ 20 (hregSafe20 hsafe hp3)).trans h20, ha20_ne⟩
      · rw [bne_iff_ne] at hneA
        by_cases hrd20 : rd = 20
        · subst hrd20
          have hmodW : ((if isPage then s.pc / 4096 * 4096 else s.pc) + imm) % AArch64Env.W % pow2 64 =
              ((if isPage then s.pc / 4096 * 4096 else s.pc) + imm) % AArch64Env.W := by
            rw [pow2_64]; unfold AArch64Env.W; exact Nat.mod_eq_of_lt (Nat.mod_lt _ (by decide))
          exact ⟨_, Nat.mod_lt _ (by decide), by simp [St.wrZR, St.setReg, St.reg, hl, hmodW], hneA⟩
        · exact ⟨a20, ha20_lt, (wrZR_reg_ne'' _ _ _ _ 20 hrd20).trans h20, ha20_ne⟩
  | addSubImm isSub sf w rd rn imm =>
    rw [Bool.and_eq_true] at halt
    simp only [exec, lift, bind, Option.bind, pure] at hx
    cases h1 : s.rdSP w rn <;> simp [h1] at hx
    cases sf <;> simp only [Bool.false_eq_true, ↓reduceIte] at hx <;> cases hx
    · exact ⟨by rw [wrSP_len']; exact hl,
        verdictWf_wrSP' (s := { s with pc := (s.pc + 4) % AArch64Env.W }) hwf hl halt.1,
        Or.inl ⟨by rw [wrSP_pc']; exact hmod4, wrSP_mem' _ _ _ _,
          fun hp2 => ⟨a19, ha19_lt, (wrSP_reg_ne'' _ _ _ _ 19 (hregSafe19 halt.2 hp2)).trans h19, ha19_ne⟩,
          fun hp3 => ⟨a20, ha20_lt, (wrSP_reg_ne'' _ _ _ _ 20 (hregSafe20 halt.2 hp3)).trans h20, ha20_ne⟩⟩⟩
    · exact ⟨by rw [wrZR_len']; exact hl,
        verdictWf_wrZR' (s := { s with pc := (s.pc + 4) % AArch64Env.W, fl := _ }) hwf hl halt.1,
        Or.inl ⟨by rw [wrZR_pc']; exact hmod4, wrZR_mem' _ _ _ _,
          fun hp2 => ⟨a19, ha19_lt, (wrZR_reg_ne'' _ _ _ _ 19 (hregSafe19 halt.2 hp2)).trans h19, ha19_ne⟩,
          fun hp3 => ⟨a20, ha20_lt, (wrZR_reg_ne'' _ _ _ _ 20 (hregSafe20 halt.2 hp3)).trans h20, ha20_ne⟩⟩⟩
  | addSubReg isSub sf w rd rn rm sh amt =>
    rw [Bool.and_eq_true] at halt
    simp only [exec, lift, bind, Option.bind, pure] at hx
    cases h1 : s.rdZR w rn <;> cases h2 : s.rdZR w rm <;> simp [h1, h2] at hx
    cases sf <;> simp only [Bool.false_eq_true, ↓reduceIte] at hx <;> cases hx <;>
      exact ⟨by rw [wrZR_len']; exact hl,
        verdictWf_wrZR' hwf hl halt.1,
        Or.inl ⟨by rw [wrZR_pc']; exact hmod4, wrZR_mem' _ _ _ _,
          fun hp2 => ⟨a19, ha19_lt, (wrZR_reg_ne'' _ _ _ _ 19 (hregSafe19 halt.2 hp2)).trans h19, ha19_ne⟩,
          fun hp3 => ⟨a20, ha20_lt, (wrZR_reg_ne'' _ _ _ _ 20 (hregSafe20 halt.2 hp3)).trans h20, ha20_ne⟩⟩⟩
  | logicReg op inv w rd rn rm sh amt =>
    rw [Bool.and_eq_true] at halt
    simp only [exec, lift, bind, Option.bind, pure] at hx
    cases h1 : s.rdZR w rn <;> cases h2 : s.rdZR w rm <;> simp [h1, h2] at hx
    cases op <;> simp only [↓reduceIte] at hx <;> cases hx <;>
      exact ⟨by rw [wrZR_len']; exact hl,
        verdictWf_wrZR' hwf hl halt.1,
        Or.inl ⟨by rw [wrZR_pc']; exact hmod4, wrZR_mem' _ _ _ _,
          fun hp2 => ⟨a19, ha19_lt, (wrZR_reg_ne'' _ _ _ _ 19 (hregSafe19 halt.2 hp2)).trans h19, ha19_ne⟩,
          fun hp3 => ⟨a20, ha20_lt, (wrZR_reg_ne'' _ _ _ _ 20 (hregSafe20 halt.2 hp3)).trans h20, ha20_ne⟩⟩⟩
  | bitfield kind w rd rn immr imms =>
    rw [Bool.and_eq_true] at halt
    simp only [exec, lift, bind, Option.bind, pure] at hx
    cases h1 : s.rdZR w rn <;> simp [h1] at hx
    cases kind with
    | bfm =>
      cases h2 : s.rdZR w rd <;> simp [h2] at hx; cases hx
      exact ⟨by rw [wrZR_len']; exact hl,
        verdictWf_wrZR' hwf hl halt.1,
        Or.inl ⟨by rw [wrZR_pc']; exact hmod4, wrZR_mem' _ _ _ _,
          fun hp2 => ⟨a19, ha19_lt, (wrZR_reg_ne'' _ _ _ _ 19 (hregSafe19 halt.2 hp2)).trans h19, ha19_ne⟩,
          fun hp3 => ⟨a20, ha20_lt, (wrZR_reg_ne'' _ _ _ _ 20 (hregSafe20 halt.2 hp3)).trans h20, ha20_ne⟩⟩⟩
    | _ =>
      cases hx
      exact ⟨by rw [wrZR_len']; exact hl,
        verdictWf_wrZR' hwf hl halt.1,
        Or.inl ⟨by rw [wrZR_pc']; exact hmod4, wrZR_mem' _ _ _ _,
          fun hp2 => ⟨a19, ha19_lt, (wrZR_reg_ne'' _ _ _ _ 19 (hregSafe19 halt.2 hp2)).trans h19, ha19_ne⟩,
          fun hp3 => ⟨a20, ha20_lt, (wrZR_reg_ne'' _ _ _ _ 20 (hregSafe20 halt.2 hp3)).trans h20, ha20_ne⟩⟩⟩
  | csel op w rd rn rm cc =>
    rw [Bool.and_eq_true] at halt
    simp only [exec, lift, bind, Option.bind, pure] at hx
    cases h1 : s.rdZR w rn <;> cases h2 : s.rdZR w rm <;> cases h3 : cond s.fl cc <;> simp [h1, h2, h3] at hx
    cases hx
    exact ⟨by rw [wrZR_len']; exact hl,
      verdictWf_wrZR' hwf hl halt.1,
      Or.inl ⟨by rw [wrZR_pc']; exact hmod4, wrZR_mem' _ _ _ _,
        fun hp2 => ⟨a19, ha19_lt, (wrZR_reg_ne'' _ _ _ _ 19 (hregSafe19 halt.2 hp2)).trans h19, ha19_ne⟩,
        fun hp3 => ⟨a20, ha20_lt, (wrZR_reg_ne'' _ _ _ _ 20 (hregSafe20 halt.2 hp3)).trans h20, ha20_ne⟩⟩⟩
  | ccmpReg isSub w rn rm nzcv cc =>
    simp only [exec, lift, bind, Option.bind, pure] at hx
    cases h1 : s.rdZR w rn <;> cases h2 : s.rdZR w rm <;> cases h3 : cond s.fl cc <;> simp [h1, h2, h3] at hx
    cases hx
    exact ⟨hl, hwf, Or.inl ⟨hmod4, rfl, fun _ => ⟨a19, ha19_lt, h19, ha19_ne⟩, fun _ => ⟨a20, ha20_lt, h20, ha20_ne⟩⟩⟩
  | ldr sz rt rn off | ldur sz rt rn off =>
    have hwok : wOk sz.width = true := by cases sz <;> decide
    simp only [exec, lift, bind, Option.bind, pure] at hx
    cases h1 : s.rdSP 64 rn with
    | none => simp [h1] at hx
    | some v =>
      simp only [h1] at hx
      by_cases hsp : (rn == 31 && v % 16 != 0) = true
      · simp [hsp] at hx
      · simp only [if_neg hsp] at hx
        cases h2 : rdMem s.mem ((v + off) % AArch64Env.W) sz with
        | none => simp [h2] at hx
        | some x =>
          simp only [h2] at hx; cases hx
          exact ⟨by rw [wrZR_len']; exact hl,
            verdictWf_wrZR' hwf hl hwok,
            Or.inl ⟨by rw [wrZR_pc']; exact hmod4, wrZR_mem' _ _ _ _,
              fun hp2 => ⟨a19, ha19_lt, (wrZR_reg_ne'' _ _ _ _ 19 (hregSafe19 halt hp2)).trans h19, ha19_ne⟩,
              fun hp3 => ⟨a20, ha20_lt, (wrZR_reg_ne'' _ _ _ _ 20 (hregSafe20 halt hp3)).trans h20, ha20_ne⟩⟩⟩
  | ldrReg sz rt rn rm ext scaled =>
    have hwok : wOk sz.width = true := by cases sz <;> decide
    simp only [exec, lift, bind, Option.bind, pure] at hx
    cases h1 : s.rdSP 64 rn with
    | none => simp [h1] at hx
    | some v =>
      simp only [h1] at hx
      by_cases hsp : (rn == 31 && v % 16 != 0) = true
      · simp [hsp] at hx
      · simp only [if_neg hsp] at hx
        cases h2 : s.rdZR 64 rm with
        | none => simp [h2] at hx
        | some idxRaw =>
          simp only [h2] at hx
          generalize h3 : rdMem s.mem _ sz = om at hx
          cases om with
          | none => cases hx
          | some x =>
            cases hx
            exact ⟨by rw [wrZR_len']; exact hl,
              verdictWf_wrZR' hwf hl hwok,
              Or.inl ⟨by rw [wrZR_pc']; exact hmod4, wrZR_mem' _ _ _ _,
                fun hp2 => ⟨a19, ha19_lt, (wrZR_reg_ne'' _ _ _ _ 19 (hregSafe19 halt hp2)).trans h19, ha19_ne⟩,
                fun hp3 => ⟨a20, ha20_lt, (wrZR_reg_ne'' _ _ _ _ 20 (hregSafe20 halt hp3)).trans h20, ha20_ne⟩⟩⟩
  | str sz rt rn offS =>
    simp only [exec, lift, bind, Option.bind, pure] at hx
    cases hbase : s.rdSP 64 rn with
    | none => simp [hbase] at hx
    | some base =>
      simp only [hbase] at hx
      by_cases hsp : (rn == 31 && base % 16 != 0) = true
      · simp [hsp] at hx
      · simp only [if_neg hsp] at hx
        cases hv : s.rdZR sz.width rt with
        | none => simp [hv] at hx
        | some v =>
          simp only [hv] at hx
          cases hwr : wrMem s.mem ((base + offS) % AArch64Env.W) v sz with
          | none => simp [hwr] at hx
          | some m' =>
            simp only [hwr] at hx; cases hx
            by_cases hp3 : L_p3 ≤ s.pc
            · exact ⟨hl, hwf, Or.inr (Or.inr (Or.inl ⟨hp3, hmod4, rfl⟩))⟩
            · have hnot3 : ((if s.pc < L_p2 then 1 else if s.pc < L_p3 then 2 else 3) == 3) = false := by
                by_cases h1 : s.pc < L_p2
                · simp [h1]
                · simp [h1, show s.pc < L_p3 by omega]
              simp only [hnot3, Bool.false_eq_true, ↓reduceIte] at halt
              refine ⟨hl, hwf, Or.inr (Or.inr (Or.inr ⟨by omega, hmod4, rfl, rfl, rfl, rfl, ?_⟩))⟩
              split at halt
              · next sz0 rt0 rn0 off0 horig =>
                simp only [Bool.and_eq_true, beq_iff_eq] at halt
                obtain ⟨⟨rfl, rfl⟩, rfl⟩ := halt
                exact Or.inl ⟨sz, rt, rn, offS, rt0, base, v, Or.inl ⟨rfl, horig⟩, hbase, hsp, hwr⟩
              · next rn0 imm12 horig =>
                simp only [Bool.and_eq_true, Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq] at halt
                obtain ⟨⟨⟨⟨rfl, rfl⟩, rfl⟩, himm⟩, hcase⟩ := halt
                have hoff_eq : (s.pc - if s.pc < L_p2 then L_p1 else if s.pc < L_p3 then L_p2 else L_p3) =
                    (s.pc - if s.pc < L_p2 then L_p1 else L_p2) := by
                  by_cases h1 : s.pc < L_p2 <;> simp [h1, show s.pc < L_p3 by omega]
                rw [hoff_eq] at hcase
                exact Or.inr ⟨rt, rn, imm12, base, v, rfl, horig, himm, hbase, hwr, by
                  rcases hcase with (⟨⟨h1, h2⟩, h3⟩ | ⟨h1, h2⟩) | ⟨h1, h2⟩
                  · exact Or.inl ⟨h1, h2, h3⟩
                  · exact Or.inr (Or.inl ⟨h1, h2⟩)
                  · exact Or.inr (Or.inr ⟨h1, h2⟩)⟩
              · cases halt
  | stur sz rt rn offS =>
    simp only [exec, lift, bind, Option.bind, pure] at hx
    cases hbase : s.rdSP 64 rn with
    | none => simp [hbase] at hx
    | some base =>
      simp only [hbase] at hx
      by_cases hsp : (rn == 31 && base % 16 != 0) = true
      · simp [hsp] at hx
      · simp only [if_neg hsp] at hx
        cases hv : s.rdZR sz.width rt with
        | none => simp [hv] at hx
        | some v =>
          simp only [hv] at hx
          cases hwr : wrMem s.mem ((base + offS) % AArch64Env.W) v sz with
          | none => simp [hwr] at hx
          | some m' =>
            simp only [hwr] at hx; cases hx
            by_cases hp3 : L_p3 ≤ s.pc
            · exact ⟨hl, hwf, Or.inr (Or.inr (Or.inl ⟨hp3, hmod4, rfl⟩))⟩
            · have hnot3 : ((if s.pc < L_p2 then 1 else if s.pc < L_p3 then 2 else 3) == 3) = false := by
                by_cases h1 : s.pc < L_p2
                · simp [h1]
                · simp [h1, show s.pc < L_p3 by omega]
              simp only [hnot3, Bool.false_eq_true, ↓reduceIte] at halt
              refine ⟨hl, hwf, Or.inr (Or.inr (Or.inr ⟨by omega, hmod4, rfl, rfl, rfl, rfl, ?_⟩))⟩
              split at halt
              · next sz0 rt0 rn0 off0 horig =>
                simp only [Bool.and_eq_true, beq_iff_eq] at halt
                obtain ⟨⟨rfl, rfl⟩, rfl⟩ := halt
                exact Or.inl ⟨sz, rt, rn, offS, rt0, base, v, Or.inr ⟨rfl, horig⟩, hbase, hsp, hwr⟩
              · cases halt

/-! ### Unconditional 4-channel (`fetch`, `reg`, `mem`, `nzcv`) immunity from `TripledState` -/

private theorem step4_inPassOrNxt {lo hi pc : Nat} (hlo : lo ≤ pc) (hlt : pc < hi)
    (hal : (pc - lo) % 4 = 0) (hhi : (hi - lo) % 4 = 0) :
    inPassOrNxt lo hi hi (pc + 4) := by
  by_cases hlt' : pc + 4 < hi
  · exact Or.inl ⟨by omega, hlt', by omega⟩
  · exact Or.inr (by omega)

private theorem step4_inP3OrEnd {pc : Nat} (hlo : L_p3 ≤ pc) (hlt : pc < L_p3_end)
    (hal : (pc - L_p3) % 4 = 0) :
    inPassOrNxt L_p3 L_p3_end L_vote (pc + 4) ∨ pc + 4 = L_p3_end := by
  have hdiff : L_p3_end - L_p3 = 2728 := by decide
  by_cases hlt' : pc + 4 < L_p3_end
  · exact Or.inl (Or.inl ⟨by omega, hlt', by omega⟩)
  · exact Or.inr (by omega)

theorem tripled_full_oneFault_no_false_accept {bs : List Nat} {s0 : St}
    (ht : TripledState bs s0) (hinv : checkFlat bs ≠ some true) :
    ∀ n o', OneFaultRun alts im n (.ok s0) o' → o' ≠ .exit 0 := by
  let v := checkFlatWord bs
  have hv_ne : v ≠ ACCEPT := checkFlatWord_ne_accept hinv
  have hv_ro : v = REJECT ∨ v = OVERFLOW := checkFlatWord_rej_or_ovf hinv
  have hv_lt : v < AArch64Env.W := checkFlatWord_lt_W bs
  have hv_notAcc : valNotAcc v := by
    rcases hv_ro with h | h <;> (rw [h]; unfold valNotAcc; decide)
  let m1 := passSubMem s0.mem 0x600000 ht.b1 bs
  let m2 := passSubMem s0.mem 0x601000 ht.b2 bs
  let m3 := passSubMem s0.mem 0x602000 ht.b3 bs
  have hs1 := passSubMem_spec ht.hD1_ptr ht.hD1_len ht.hBuf1 (ht.disjD1 0x600000 (by simp))
  have hs2 := passSubMem_spec ht.hD2_ptr ht.hD2_len ht.hBuf2 (ht.disjD2 0x601000 (by simp))
  have hs3 := passSubMem_spec ht.hD3_ptr ht.hD3_len ht.hBuf3 (ht.disjD3 0x602000 (by simp))
  have hsub1_0 : MemSub m1 s0.mem := hs1.2.2.2 _ fun _ _ => rfl
  have hsub2_0 : MemSub m2 s0.mem := hs2.2.2.2 _ fun _ _ => rfl
  have hsub3_0 : MemSub m3 s0.mem := hs3.2.2.2 _ fun _ _ => rfl
  have hby_m1 : ∀ a x, m1.rd a = some x → x < 256 :=
    passSubMem_bytes_lt_256 ht.hD1_ptr ht.hD1_len ht.hBuf1 (ht.disjD1 0x600000 (by simp))
      (ht.descBytes 0x600000 (by simp)) ht.bytes
  have hby_m2 : ∀ a x, m2.rd a = some x → x < 256 :=
    passSubMem_bytes_lt_256 ht.hD2_ptr ht.hD2_len ht.hBuf2 (ht.disjD2 0x601000 (by simp))
      (ht.descBytes 0x601000 (by simp)) ht.bytes
  have hlen_lt_W : bs.length < AArch64Env.W := by
    have := ht.hSm1; unfold HMAX AArch64Env.W at *; omega
  have hdisj12 : ∀ a, inFootprint 0x600000 ht.b1 bs.length a → ¬ inFootprint 0x601000 ht.b2 bs.length a := by
    intro a h1 h2
    have := ht.disj12; have := ht.disjD1 0x601000 (by simp); have := ht.disjD2 0x600000 (by simp)
    dsimp only [TripledState.b1, TripledState.b2, inFootprint] at h1 h2; omega
  have hdisj13 : ∀ a, inFootprint 0x600000 ht.b1 bs.length a → ¬ inFootprint 0x602000 ht.b3 bs.length a := by
    intro a h1 h3
    have := ht.disj13; have := ht.disjD1 0x602000 (by simp); have := ht.disjD3 0x600000 (by simp)
    dsimp only [TripledState.b1, TripledState.b3, inFootprint] at h1 h3; omega
  have hdisj23 : ∀ a, inFootprint 0x601000 ht.b2 bs.length a → ¬ inFootprint 0x602000 ht.b3 bs.length a := by
    intro a h2 h3
    have := ht.disj23; have := ht.disjD2 0x602000 (by simp); have := ht.disjD3 0x601000 (by simp)
    dsimp only [TripledState.b2, TripledState.b3, inFootprint] at h2 h3; omega
  have hwf0 : VerdictRegsWf s0 := verdictWf_of_tripled ht
  -- Clean Pass 1 trajectory on minimal `m1` and on `s0.mem`
  obtain ⟨N1, s1m, hr1m, hf1m, hpc1m, h19_1m, hl1m, hm1m⟩ :=
    p1_spec hs1.1 hs1.2.1 hs1.2.2.1 ht.bytes ht.hSm1 ht.regsLen ht.pc (s := { s0 with mem := m1 }) rfl
  have hstay1 := pass_stay_strict (Or.inl ⟨rfl, rfl, rfl, rfl⟩) N1 { s0 with mem := m1 } s1m
    (Or.inl ⟨by simp only [ht.pc, L_tripled]; decide, by simp only [ht.pc, L_tripled]; decide,
      by simp only [ht.pc, L_tripled]; decide⟩) ht.regsLen hwf0 hr1m hpc1m
  have ⟨_, hr1_full⟩ := pass_run_memSub (Or.inl ⟨rfl, rfl, rfl, rfl⟩) hsub1_0 N1 { s0 with mem := m1 } s1m
    rfl ht.regsLen hwf0 hr1m hstay1
  have hnever1 := clean_pass_never_acc (Or.inl ⟨rfl, rfl, rfl, rfl⟩) (s := { s0 with mem := m1 })
    (show s0.pc = L_p1 from ht.pc) ht.x19 hr1m hpc1m h19_1m hv_ne hstay1
  let s1 := { s1m with mem := s0.mem }
  have hwf_s1 : VerdictRegsWf s1 := by
    intro r hr; simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
    rcases hr with rfl | rfl | rfl
    · exact ⟨v, hv_lt, h19_1m⟩
    · exact ⟨REJECT, by decide, by rw [show s1.reg 20 = s1m.reg 20 from rfl, hf1m.regs 20 (by decide)]; exact ht.x20⟩
    · exact ⟨REJECT, by decide, by rw [show s1.reg 21 = s1m.reg 21 from rfl, hf1m.regs 21 (by decide)]; exact ht.x21⟩
  have hscr1 : VoterScratchClean s1 := show VoterScratchClean s1m from voterScratchClean_of_frame hf1m (by decide) ht.scratch
  -- Clean Pass 2 trajectory on minimal `m2` and on `s0.mem`
  obtain ⟨N2, s2m, hr2m, hf2m, hpc2m, h20_2m, hl2m, hm2m⟩ :=
    p2_spec hs2.1 hs2.2.1 hs2.2.2.1 ht.bytes ht.hSm2 hl1m hpc1m (s := { s1 with mem := m2 }) rfl
  have hstay2 := pass_stay_strict (Or.inr (Or.inl ⟨rfl, rfl, rfl, rfl⟩)) N2 { s1 with mem := m2 } s2m
    (Or.inl ⟨by simp [s1, hpc1m], by simp [s1, hpc1m]; decide, by simp [s1, hpc1m]⟩) hl1m hwf_s1 hr2m hpc2m
  have ⟨_, hr2_full⟩ := pass_run_memSub (Or.inr (Or.inl ⟨rfl, rfl, rfl, rfl⟩)) hsub2_0 N2 { s1 with mem := m2 } s2m
    rfl hl1m hwf_s1 hr2m hstay2
  have h20_s1 : ({ s1 with mem := m2 }).reg 20 = some REJECT := by
    change s1m.reg 20 = some REJECT
    rw [hf1m.regs 20 (by decide)]; exact ht.x20
  have hnever2 := clean_pass_never_acc (Or.inr (Or.inl ⟨rfl, rfl, rfl, rfl⟩)) (s := { s1 with mem := m2 })
    hpc1m h20_s1 hr2m hpc2m h20_2m hv_ne hstay2
  let s2 := { s2m with mem := s0.mem }
  have h19_2 : s2.reg 19 = some v := by rw [show s2.reg 19 = s2m.reg 19 from rfl, hf2m.regs 19 (by decide)]; exact h19_1m
  have hwf2 : VerdictRegsWf s2 := by
    intro r hr; simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
    rcases hr with rfl | rfl | rfl
    · exact ⟨v, hv_lt, h19_2⟩
    · exact ⟨v, hv_lt, h20_2m⟩
    · refine ⟨REJECT, by decide, ?_⟩
      calc s2.reg 21 = s2m.reg 21 := rfl
        _ = ({ s1 with mem := m2 }).reg 21 := hf2m.regs 21 (by decide)
        _ = s1m.reg 21 := rfl
        _ = ({ s0 with mem := m1 }).reg 21 := hf1m.regs 21 (by decide)
        _ = some REJECT := ht.x21
  have hscr2 : VoterScratchClean s2 := show VoterScratchClean s2m from voterScratchClean_of_frame hf2m (by decide) hscr1
  -- Clean Pass 3 trajectory on minimal `m3` and on `s0.mem`
  obtain ⟨N3, s3m, hr3m, hf3m, hpc3m, h21_3m, hl3m, hm3m⟩ :=
    p3_spec hs3.1 hs3.2.1 hs3.2.2.1 ht.bytes ht.hSm3 hl2m hpc2m (s := { s2 with mem := m3 }) rfl
  have hstay3 := pass_stay_strict (Or.inr (Or.inr ⟨rfl, rfl, rfl, rfl⟩)) N3 { s2 with mem := m3 } s3m
    (Or.inl ⟨by simp [s2, hpc2m], by simp [s2, hpc2m]; decide, by simp [s2, hpc2m]⟩) hl2m hwf2 hr3m hpc3m
  have ⟨_, hr3_full⟩ := pass_run_memSub (Or.inr (Or.inr ⟨rfl, rfl, rfl, rfl⟩)) hsub3_0 N3 { s2 with mem := m3 } s3m
    rfl hl2m hwf2 hr3m hstay3
  let s3 := { s3m with mem := s0.mem }
  have h19_3 : s3.reg 19 = some v := by rw [show s3.reg 19 = s3m.reg 19 from rfl, hf3m.regs 19 (by decide)]; exact h19_2
  have h20_3 : s3.reg 20 = some v := by rw [show s3.reg 20 = s3m.reg 20 from rfl, hf3m.regs 20 (by decide)]; exact h20_2m
  have hscr3 : VoterScratchClean s3 := show VoterScratchClean s3m from voterScratchClean_of_frame hf3m (by decide) hscr2
  have hpre3 : PreFault L_vote v s3 := by
    have ⟨hsafe, hodd⟩ := checkFlatWord_safeInvalid hinv
    exact ⟨hsafe, hodd, hl3m, hscr3, 0, by simp [s3, hpc3m],
      Or.inl ⟨by decide, h19_3, h20_3, h21_3m, fun h => by omega⟩⟩
  -- Helper: finish Pass 1 after `s'` at `L_p2` with `reg 19 = some a19 ≠ ACCEPT` and `m_f` preserving `m2` or `m3`
  have hfinish_p1_at_p2 : ∀ {rem_p a19 : Nat} {u_p2 : St},
      u_p2.pc = L_p2 → u_p2.regs.length = 31 → VerdictRegsWf u_p2 →
      u_p2.reg 19 = some a19 → a19 < AArch64Env.W → a19 ≠ ACCEPT →
      (MemSub m3 u_p2.mem ∨ MemSub m2 u_p2.mem) →
      run im rem_p (.ok u_p2) ≠ .exit 0 := by
    intro rem_p a19 u_p2 hpc_p2 hl_p2 hwf_p2 h19_p2 ha19_lt ha19_ne hsub_or h0
    rcases hsub_or with hsub3' | hsub2'
    · obtain ⟨t, _, u3, _, hpc3, hm3, hl3, hwf3', hreg3, hend⟩ :=
        pass_clean_escape (Or.inr (Or.inl ⟨rfl, rfl, rfl, rfl⟩)) rem_p u_p2
          (Or.inl ⟨by rw [hpc_p2]; exact Nat.le_refl _, by rw [hpc_p2]; decide, by rw [hpc_p2]; decide⟩) hl_p2 hwf_p2 h0
      exact run_after_p3_with_19_ne_acc ht hinv hpc3 hl3 hwf3'
        ((hreg3 19 (by simp) (by decide)).trans h19_p2) ha19_lt ha19_ne (hm3.symm ▸ hsub3') hend
    · have hD2p : rdMem u_p2.mem 0x601000 .x8 = some ht.b2 := rdMem_memSub hsub2' hs2.1
      have hD2l : rdMem u_p2.mem (0x601000 + 8) .x8 = some bs.length := rdMem_memSub hsub2' hs2.2.1
      have hB2 : Buf u_p2.mem ht.b2 bs := fun k hk => hsub2' _ _ (hs2.2.2.1 k hk)
      obtain ⟨k2, u2, hr2', hf2', hpc2', h20_2', hl2', _⟩ :=
        p2_spec hD2p hD2l hB2 ht.bytes ht.hSm2 hl_p2 hpc_p2 rfl
      have h19_2' : u2.reg 19 = some a19 := by rw [hf2'.regs 19 (by decide)]; exact h19_p2
      have hwf2' : VerdictRegsWf u2 := by
        intro r hr; simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
        rcases hr with rfl | rfl | rfl
        · exact ⟨a19, ha19_lt, h19_2'⟩
        · exact ⟨v, hv_lt, h20_2'⟩
        · obtain ⟨c, hc, hrc⟩ := hwf_p2 21 (by simp); exact ⟨c, hc, by rw [hf2'.regs 21 (by decide)]; exact hrc⟩
      by_cases hle2 : k2 ≤ rem_p
      · rw [show rem_p = k2 + (rem_p - k2) by omega, run_add, hr2'] at h0
        exact run_after_p3_clean_with_19_20_ne_acc
          (Or.inl (Or.inl ⟨by simp [hpc2'], by simp [hpc2']; decide, by simp [hpc2']⟩))
          hl2' hwf2' h19_2' ha19_lt ha19_ne h20_2' hv_lt hv_ne h0
      · have h := congrArg (run im (k2 - rem_p)) h0
        rw [← run_add, show rem_p + (k2 - rem_p) = k2 by omega, hr2', run_exit] at h; cases h
  -- Main case split on `OneFaultRun`
  intro n o' ho
  cases ho with
  | of_clean =>
    have ⟨k, he⟩ := three_pass_ends ht
    intro h0
    have hex0 := run_exit_unique he h0
    unfold expectedExit at hex0
    cases hcf : checkFlat bs with
    | none => simp only [hcf] at hex0; omega
    | some ok =>
      cases ok with
      | false => simp only [hcf] at hex0; omega
      | true => exact hinv hcf
  | of_fault i _ si oi hi hri hst =>
    let rem := n - 1 - i
    -- Case 1: disturbance inside tail (`i ≥ N1 + N2 + N3`)
    by_cases htail : N1 + N2 + N3 ≤ i
    · have hr3_tot : run im (N1 + N2 + N3) (.ok s0) = .ok s3 := by
        rw [run_add, run_add, show { s0 with mem := s0.mem } = s0 from rfl] at *
        rw [hr1_full, hr2_full, hr3_full]
      have hri' : run im (i - (N1 + N2 + N3)) (.ok s3) = .ok si := by
        have h := hri
        rw [show i = (N1 + N2 + N3) + (i - (N1 + N2 + N3)) by omega, run_add, hr3_tot] at h
        exact h
      have htail_one : OneFaultRun alts im (n - (N1 + N2 + N3)) (.ok s3) (run im rem oi) := by
        have hfin := OneFaultRun.of_fault (alts := alts) (n - (N1 + N2 + N3)) (i - (N1 + N2 + N3)) (.ok s3) si oi
          (by omega) hri' hst
        rwa [show n - (N1 + N2 + N3) - 1 - (i - (N1 + N2 + N3)) = rem by omega] at hfin
      exact (tail_full_oneFault_no_false_accept hpre3).1 _ _ htail_one
    -- Case 2: disturbance during Pass 1 (`i < N1`)
    · by_cases hp1 : i < N1
      · obtain ⟨ui, hui, hlo_i, hlt_i, hal_i, hl_i, hwf_i⟩ := hstay1 i hp1
        have ⟨hm_ui, hri_eq⟩ := pass_run_memSub (Or.inl ⟨rfl, rfl, rfl, rfl⟩) hsub1_0 i { s0 with mem := m1 } ui
          rfl ht.regsLen hwf0 hui (fun t ht' => hstay1 t (by omega))
        rw [hri_eq] at hri; cases hri
        obtain ⟨_, a19_ui, ha19_ui_lt, h19_ui, hmod19_ui⟩ := hnever1 i hp1 ui hui
        have ha19_ui_ne : a19_ui ≠ ACCEPT := fun h => hmod19_ui (congrArg (· % 65536) h)
        have h20_ui : ui.reg 20 = some REJECT := by
          clear hst hri_eq
          have hinv20 : ∀ t ≤ i, ∀ u, run im t (.ok { s0 with mem := m1 }) = .ok u → u.reg 20 = some REJECT := by
            intro t
            induction t with
            | zero => intro _ u hu; simp [run] at hu; subst hu; exact ht.x20
            | succ t' ih =>
              intro ht' u hu
              obtain ⟨ut, hut, hlo_t, hlt_t, hal_t, hl_t, hwf_t⟩ := hstay1 t' (by omega)
              rw [run_add, hut] at hu; simp only [run] at hu
              cases him : im ut.pc with
              | none => simp [step, him] at hu
              | some p =>
                obtain ⟨ins, len⟩ := p
                obtain ⟨rfl, hsafe⟩ := im_in_pass_safe (Or.inl ⟨rfl, rfl, rfl, rfl⟩) hlo_t hlt_t him
                have hx : exec ins 4 ut = .ok u := by simpa [step, him] using hu
                have hr20 := ((exec_pass_props hsafe hlo_t hlt_t (by decide) hal_t hl_t hwf_t).2.1 u hx).2.2.2.2 20 (by simp) (by decide)
                exact hr20.trans (ih (by omega) ut hut)
          exact hinv20 i (Nat.le_refl _) ui hui
        have ⟨_, hstep_inv1⟩ := clean_pass_step_inv (Or.inl ⟨rfl, rfl, rfl, rfl, rfl⟩) (s := { s0 with mem := m1 })
          (show s0.pc = L_p1 from ht.pc) hby_m1 hs1.2.1 hlen_lt_W i ui hui (fun j hj => hstay1 j (by omega))
        -- Get step `i + 1` on `m1` and `s0.mem`
        have hr_split : run im (N1 - i) (.ok ui) = .ok s1m := by
          have := hr1m; rw [show N1 = i + (N1 - i) by omega, run_add, hui] at this; exact this
        have hpos : N1 - i = 1 + (N1 - (i + 1)) := by omega
        rw [hpos, run_add] at hr_split
        cases hstep1 : step im ui with
        | exit _ => simp [run, hstep1, run_exit] at hr_split
        | stuck => simp [run, hstep1, run_stuck] at hr_split
        | ok u1 =>
          simp only [run, hstep1] at hr_split
          cases him : im ui.pc with
          | none => simp [step, him] at hstep1
          | some p =>
            obtain ⟨ins, len⟩ := p
            obtain ⟨rfl, hsafe⟩ := im_in_pass_safe (Or.inl ⟨rfl, rfl, rfl, rfl⟩) hlo_i hlt_i him
            have hx : exec ins 4 ui = .ok u1 := by simpa [step, him] using hstep1
            have ⟨_, hprops, _, hmono⟩ := exec_pass_props hsafe hlo_i hlt_i (by decide) hal_i hl_i hwf_i
            obtain ⟨hm1, hl1, hin1, hwf1, _⟩ := hprops u1 hx
            have hm_u1 : u1.mem = m1 := hm1.trans hm_ui
            have hin1' : inPassOrNxt L_p1 L_p2 L_p2 u1.pc := by
              rcases hin1 with ⟨h1, h2, h3⟩ | h4
              · exact Or.inl ⟨by omega, h2, h3⟩
              · exact Or.inr h4
            have hstep_full : step im { ui with mem := s0.mem } = .ok { u1 with mem := s0.mem } := by
              simp [step, him, hmono s0.mem (hm_ui ▸ hsub1_0) u1 hx]
            have hstay1_rem : ∀ t < N1 - (i + 1), ∃ u, run im t (.ok u1) = .ok u ∧ L_p1 ≤ u.pc ∧ u.pc < L_p2 ∧
                (u.pc - L_p1) % 4 = 0 ∧ u.regs.length = 31 ∧ VerdictRegsWf u := by
              intro t ht'
              obtain ⟨u, hu, h1, h2, h3, h4, h5⟩ := hstay1 (i + 1 + t) (by omega)
              rw [show i + 1 + t = i + (1 + t) by omega, run_add, hui, run_add] at hu
              simp only [run, hstep1] at hu
              exact ⟨u, hu, h1, h2, h3, h4, h5⟩
            cases hst with
            | clean =>
              rw [hstep_full]; intro h0
              obtain ⟨t, _, u2, _, hpc2, hm2, hl2, hwf2', _, hend⟩ :=
                pass_clean_escape (Or.inl ⟨rfl, rfl, rfl, rfl⟩) rem { u1 with mem := s0.mem } hin1' hl1 hwf1 h0
              exact run_after_p2_p3 ht hinv hpc2 hl2 hwf2' ((show u2.mem = s0.mem from hm2).symm ▸ hsub2_0) ((show u2.mem = s0.mem from hm2).symm ▸ hsub3_0) hend
            | reg s1_f idx bit hs_eq _ hbit =>
              rw [hstep_full] at hs_eq; cases hs_eq; intro h0
              have hin_fl : inPassOrNxt L_p1 L_p2 L_p2 ({ u1 with mem := s0.mem }.flipReg idx bit).pc := by
                rw [flipReg_pc]; exact hin1'
              obtain ⟨t, _, u2, _, hpc2, hm2, hl2, hwf2', _, hend⟩ :=
                pass_clean_escape (Or.inl ⟨rfl, rfl, rfl, rfl⟩) rem ({ u1 with mem := s0.mem }.flipReg idx bit)
                  hin_fl (flipReg_len (s := { u1 with mem := s0.mem }) hl1 idx bit)
                  (flipReg_verdictWf (s := { u1 with mem := s0.mem }) hwf1 hl1 idx bit hbit) h0
              have hm2' : u2.mem = s0.mem := hm2.trans (flipReg_mem (s := { u1 with mem := s0.mem }) idx bit)
              exact run_after_p2_p3 ht hinv hpc2 hl2 hwf2' (hm2'.symm ▸ hsub2_0) (hm2'.symm ▸ hsub3_0) hend
            | nzcv s1_f s2_f hs_eq hnz =>
              rw [hstep_full] at hs_eq; cases hs_eq; intro h0
              obtain ⟨hpc_n, hregs_n, hmem_n⟩ := nzcvFlips_props hnz
              have hwf_n : VerdictRegsWf s2_f := fun r hr => by
                obtain ⟨u, hu, hru⟩ := hwf1 r hr; exact ⟨u, hu, by simp [St.reg, hregs_n]; exact hru⟩
              obtain ⟨t, _, u2, _, hpc2, hm2, hl2, hwf2', _, hend⟩ :=
                pass_clean_escape (Or.inl ⟨rfl, rfl, rfl, rfl⟩) rem s2_f
                  (by rw [hpc_n]; exact hin1') (by rw [hregs_n]; exact hl1) hwf_n h0
              have hm2' : u2.mem = s0.mem := hm2.trans hmem_n
              exact run_after_p2_p3 ht hinv hpc2 hl2 hwf2' (hm2'.symm ▸ hsub2_0) (hm2'.symm ▸ hsub3_0) hend
            | mem s1_f idx j bit hs_eq =>
              rw [hstep_full] at hs_eq; cases hs_eq; intro h0
              let m_f := (s0.flipMem idx j bit).mem
              have hmf_eq : ({ u1 with mem := s0.mem }.flipMem idx j bit) = { u1 with mem := m_f } := by
                unfold St.flipMem; rfl
              rw [hmf_eq] at h0
              rcases flipMem_preserves_two_subMems ht idx j bit with ⟨hsub2', hsub3'⟩ | ⟨hsub1', hsub3'⟩ | ⟨hsub1', hsub2'⟩
              · obtain ⟨t, _, u2, _, hpc2, hm2, hl2, hwf2', _, hend⟩ :=
                  pass_clean_escape (Or.inl ⟨rfl, rfl, rfl, rfl⟩) rem { u1 with mem := m_f } hin1' hl1 hwf1 h0
                exact run_after_p2_p3 ht hinv hpc2 hl2 hwf2' ((show u2.mem = m_f from hm2).symm ▸ hsub2') ((show u2.mem = m_f from hm2).symm ▸ hsub3') hend
              · have ⟨_, hr1'⟩ := pass_run_memSub (Or.inl ⟨rfl, rfl, rfl, rfl⟩) hsub1' (N1 - (i + 1)) u1 s1m hm_u1 hl1 hwf1 hr_split hstay1_rem
                by_cases hle : N1 - (i + 1) ≤ rem
                · change run im rem _ = .exit 0 at h0
                  rw [show rem = (N1 - (i + 1)) + (rem - (N1 - (i + 1))) by omega, run_add, hr1'] at h0
                  exact hfinish_p1_at_p2 (u_p2 := { s1m with mem := m_f }) hpc1m hl1m hwf_s1 h19_1m hv_lt hv_ne (Or.inl hsub3') h0
                · have h := congrArg (run im ((N1 - (i + 1)) - rem)) h0
                  rw [← run_add, show n - 1 - i + ((N1 - (i + 1)) - rem) = N1 - (i + 1) by omega, hr1', run_exit] at h; cases h
              · have ⟨_, hr1'⟩ := pass_run_memSub (Or.inl ⟨rfl, rfl, rfl, rfl⟩) hsub1' (N1 - (i + 1)) u1 s1m hm_u1 hl1 hwf1 hr_split hstay1_rem
                by_cases hle : N1 - (i + 1) ≤ rem
                · change run im rem _ = .exit 0 at h0
                  rw [show rem = (N1 - (i + 1)) + (rem - (N1 - (i + 1))) by omega, run_add, hr1'] at h0
                  exact hfinish_p1_at_p2 (u_p2 := { s1m with mem := m_f }) hpc1m hl1m hwf_s1 h19_1m hv_lt hv_ne (Or.inr hsub2') h0
                · have h := congrArg (run im ((N1 - (i + 1)) - rem)) h0
                  rw [← run_add, show n - 1 - i + ((N1 - (i + 1)) - rem) = N1 - (i + 1) by omega, hr1', run_exit] at h; cases h
            | fetch _ _ wAlt _ hw =>
              obtain ⟨wOrig, bit, _, him_w, hwAlt, halt⟩ :=
                alts_in_corePass_audit hlo_i (by unfold L_p2 L_p3_end at *; omega) hal_i hw
              rw [hwAlt]
              cases hdec : decodeWord (Nat.xor wOrig (2 ^ bit)) with
              | none => simp [execO, run_stuck]
              | some ins' =>
                simp only [execO, Option.map_some]
                cases hx' : exec ins' 4 { ui with mem := s0.mem } with
                | exit c => exact absurd hx' (exec_passFetch_ne_exit halt hdec c)
                | stuck => simp [run_stuck]
                | ok s' =>
                  have ⟨hl', hwf', hcase⟩ := exec_passFetch_step (s := { ui with mem := s0.mem })
                    hlo_i (Nat.lt_trans hlt_i (by decide : L_p2 < L_p3_end)) hl_i hwf_i
                    h19_ui ha19_ui_lt ha19_ui_ne h20_ui (by decide) (by decide)
                    (fun h => by
                      have h23 : L_p2 < L_p3 := by decide
                      omega) halt hdec hx'
                  dsimp only at hcase
                  rcases hcase with ⟨hpc', hm', _, _⟩ | ⟨hm', h19', h20', _, hbr⟩ | ⟨hp3, _, _⟩ | ⟨_, hpc', hregs', hsp', hk', hfl', hstr⟩
                  · intro h0
                    have hin_s' : inPassOrNxt L_p1 L_p2 L_p2 s'.pc := by
                      rw [hpc']; exact step4_inPassOrNxt hlo_i hlt_i hal_i (by decide)
                    obtain ⟨t, _, u2, _, hpc2, hm2, hl2, hwf2', _, hend⟩ :=
                      pass_clean_escape (Or.inl ⟨rfl, rfl, rfl, rfl⟩) rem s' hin_s' hl' hwf' h0
                    exact run_after_p2_p3 ht hinv hpc2 hl2 hwf2' ((hm2.trans hm').symm ▸ hsub2_0) ((hm2.trans hm').symm ▸ hsub3_0) hend
                  · rcases hbr with hin_p | hnone | ⟨hp3, _⟩
                    · exact run_in_corePass_with_19_20_ne_acc ht hinv hin_p hl' hwf'
                        (h19'.trans h19_ui) ha19_ui_lt ha19_ui_ne (h20'.trans h20_ui) (by decide) (by decide)
                        (hm'.symm ▸ hsub2_0) (hm'.symm ▸ hsub3_0)
                    · exact run_im_none_ne_exit0 (targetImNone_sound hnone) rem
                    · unfold L_p2 L_p3 at *; omega
                  · unfold L_p2 L_p3 at *; omega
                  · rcases hstr with ⟨sz, rt, rn, offS, rt0, base, val_s, hpair_ls, hbase, hsp, hwr⟩ |
                      ⟨rt, rn, imm12, base, val_s, _, horig_cmp, himm12, hbase, hwr, hcases_cmp⟩
                    · -- Clean `ldr`/`ldur` flipped to `str`/`stur` at same `(ea, sz)` -> `m2, m3` preserved!
                      have hbase_ui : ui.rdSP 64 rn = some base := hbase
                      have hrd_m1 : ∃ val0, rdMem m1 ((base + offS) % AArch64Env.W) sz = some val0 := by
                        rcases hpair_ls with ⟨_, horig⟩ | ⟨_, horig⟩
                        · rw [horig] at him_w; rw [him] at him_w; cases him_w
                          simp only [exec, lift, bind, Option.bind, pure, hbase_ui, if_neg hsp, hm_ui] at hx
                          cases hrd : rdMem m1 ((base + offS) % AArch64Env.W) sz with
                          | none => simp [hrd] at hx
                          | some v0 => exact ⟨v0, rfl⟩
                        · rw [horig] at him_w; rw [him] at him_w; cases him_w
                          simp only [exec, lift, bind, Option.bind, pure, hbase_ui, if_neg hsp, hm_ui] at hx
                          cases hrd : rdMem m1 ((base + offS) % AArch64Env.W) sz with
                          | none => simp [hrd] at hx
                          | some v0 => exact ⟨v0, rfl⟩
                      obtain ⟨val0, hrd0⟩ := hrd_m1
                      have hsub2' : MemSub m2 s'.mem := wrMem_passSubMem_preserves_other ht.memWf
                        ht.hD1_ptr ht.hD1_len ht.hBuf1 (ht.disjD1 0x600000 (by simp))
                        ht.hD2_ptr ht.hD2_len ht.hBuf2 (ht.disjD2 0x601000 (by simp)) hdisj12 hrd0 hwr
                      have hsub3' : MemSub m3 s'.mem := wrMem_passSubMem_preserves_other ht.memWf
                        ht.hD1_ptr ht.hD1_len ht.hBuf1 (ht.disjD1 0x600000 (by simp))
                        ht.hD3_ptr ht.hD3_len ht.hBuf3 (ht.disjD3 0x602000 (by simp)) hdisj13 hrd0 hwr
                      intro h0
                      have hin_s' : inPassOrNxt L_p1 L_p2 L_p2 s'.pc := by
                        rw [hpc']; by_cases hlt' : ui.pc + 4 < L_p2
                        · exact Or.inl ⟨by omega, hlt', by omega⟩
                        · exact Or.inr (by unfold L_p1 L_p2 at *; omega)
                      obtain ⟨t, _, u2, _, hpc2, hm2, hl2, hwf2', _, hend⟩ :=
                        pass_clean_escape (Or.inl ⟨rfl, rfl, rfl, rfl⟩) rem s' hin_s' hl' hwf' h0
                      exact run_after_p2_p3 ht hinv hpc2 hl2 hwf2' (hm2.symm ▸ hsub2') (hm2.symm ▸ hsub3') hend
                    · -- Clean `cmp` flipped to `str .x8`
                      simp only [hlt_i, ↓reduceIte] at hcases_cmp
                      rcases hcases_cmp with ⟨hoff16, rfl, rfl⟩ | ⟨hr14, rfl⟩ | ⟨hr15, rfl⟩
                      · -- `off = 16`: `cmp x14, #16` -> `str xzr, [x14, #128]`
                        have hpc16 : ui.pc = L_p1 + 16 := by omega
                        have h14_len : ui.reg 14 = some bs.length := hstep_inv1.2.1 (Or.inr hpc16)
                        have hbase_eq : base = bs.length := by
                          have hb' : ui.rdSP 64 14 = some base := hbase
                          simp only [St.rdSP, show (14 == 31) = false by decide, Bool.false_eq_true, ↓reduceIte,
                            h14_len, Option.map_some, Option.some.injEq] at hb'
                          rw [← hb', Nat.mod_eq_of_lt (by rw [pow2_64]; unfold AArch64Env.W at hlen_lt_W; exact hlen_lt_W)]
                        subst hbase_eq
                        have hea_eq : (bs.length + 16 * 8) % AArch64Env.W = bs.length + 128 := by
                          have := ht.hSm1; unfold HMAX AArch64Env.W at *; omega
                        rw [hea_eq] at hwr
                        obtain ⟨h16_le, htwo⟩ := wrMem_cmp16_preserves_two ht hwr
                        rcases htwo with ⟨hsub2', hsub3'⟩ | ⟨hsub1', hsub3'⟩ | ⟨hsub1', hsub2'⟩
                        · intro h0
                          have hin_s' : inPassOrNxt L_p1 L_p2 L_p2 s'.pc := by
                            rw [hpc', hpc16]; exact Or.inl ⟨by omega, by unfold L_p1 L_p2; decide, by decide⟩
                          obtain ⟨t, _, u2, _, hpc2, hm2, hl2, hwf2', _, hend⟩ :=
                            pass_clean_escape (Or.inl ⟨rfl, rfl, rfl, rfl⟩) rem s' hin_s' hl' hwf' h0
                          exact run_after_p2_p3 ht hinv hpc2 hl2 hwf2' (hm2.symm ▸ hsub2') (hm2.symm ▸ hsub3') hend
                        · -- `MemSub m1 s'.mem ∧ MemSub m3 s'.mem`: synchronize via `pass_off20_sync`
                          rw [horig_cmp] at him_w; rw [him] at him_w; cases him_w
                          have h14_sp : ui.rdSP 64 14 = some bs.length := by
                            simpa [St.rdSP, show (14 == 31) = false by decide, h14_len] using
                              Nat.mod_eq_of_lt (by rw [pow2_64]; unfold AArch64Env.W at hlen_lt_W; exact hlen_lt_W)
                          have hu1_eq : u1 = { ui with pc := L_p1 + 20, fl := (addSubOp true 64 bs.length 16).2 } := by
                            simp only [exec, lift, bind, Option.bind, pure, h14_sp,
                              show (16 : Nat) % pow2 64 = 16 by decide, ↓reduceIte,
                              hpc16, show (L_p1 + 16 + 4) % AArch64Env.W = L_p1 + 20 by decide, St.wrZR] at hx
                            cases hx; rfl
                          have hu1_flc : u1.fl.c = some true := by
                            rw [hu1_eq]; simp [addSubOp, h16_le]
                          rcases pass_off20_sync (Or.inl ⟨rfl, rfl, rfl, rfl⟩) hsub1'
                              (by rw [hu1_eq]) hm_u1 hu1_flc (by rw [hpc', hpc16])
                              (by rw [hregs', hu1_eq]) (by rw [hsp', hu1_eq]) (by rw [hk', hu1_eq]) rfl
                              hl1 hwf1 hr_split hpc1m hstay1_rem with
                            ⟨ks, u_end, hrs, hpc_e, hm_e, hl_e, hwf_e, hcases_e⟩ | hne_all
                          · intro h0
                            change run im rem _ = .exit 0 at h0
                            by_cases hle : ks ≤ rem
                            · rw [show rem = ks + (rem - ks) by omega, run_add, hrs] at h0
                              rcases hcases_e with rfl | ⟨h19_rej, _⟩
                              · exact hfinish_p1_at_p2 hpc_e hl_e hwf_e h19_1m hv_lt hv_ne (Or.inl hsub3') h0
                              · exact hfinish_p1_at_p2 hpc_e hl_e hwf_e h19_rej (by decide) (by decide) (Or.inl (hm_e.symm ▸ hsub3')) h0
                            · have h := congrArg (run im (ks - rem)) h0
                              rw [← run_add, show rem + (ks - rem) = ks by omega, hrs, run_exit] at h; cases h
                          · exact hne_all rem
                        · -- `MemSub m1 s'.mem ∧ MemSub m2 s'.mem`: synchronize via `pass_off20_sync`
                          rw [horig_cmp] at him_w; rw [him] at him_w; cases him_w
                          have h14_sp : ui.rdSP 64 14 = some bs.length := by
                            simpa [St.rdSP, show (14 == 31) = false by decide, h14_len] using
                              Nat.mod_eq_of_lt (by rw [pow2_64]; unfold AArch64Env.W at hlen_lt_W; exact hlen_lt_W)
                          have hu1_eq : u1 = { ui with pc := L_p1 + 20, fl := (addSubOp true 64 bs.length 16).2 } := by
                            simp only [exec, lift, bind, Option.bind, pure, h14_sp,
                              show (16 : Nat) % pow2 64 = 16 by decide, ↓reduceIte,
                              hpc16, show (L_p1 + 16 + 4) % AArch64Env.W = L_p1 + 20 by decide, St.wrZR] at hx
                            cases hx; rfl
                          have hu1_flc : u1.fl.c = some true := by
                            rw [hu1_eq]; simp [addSubOp, h16_le]
                          rcases pass_off20_sync (Or.inl ⟨rfl, rfl, rfl, rfl⟩) hsub1'
                              (by rw [hu1_eq]) hm_u1 hu1_flc (by rw [hpc', hpc16])
                              (by rw [hregs', hu1_eq]) (by rw [hsp', hu1_eq]) (by rw [hk', hu1_eq]) rfl
                              hl1 hwf1 hr_split hpc1m hstay1_rem with
                            ⟨ks, u_end, hrs, hpc_e, hm_e, hl_e, hwf_e, hcases_e⟩ | hne_all
                          · intro h0
                            change run im rem _ = .exit 0 at h0
                            by_cases hle : ks ≤ rem
                            · rw [show rem = ks + (rem - ks) by omega, run_add, hrs] at h0
                              rcases hcases_e with rfl | ⟨h19_rej, _⟩
                              · exact hfinish_p1_at_p2 hpc_e hl_e hwf_e h19_1m hv_lt hv_ne (Or.inr hsub2') h0
                              · exact hfinish_p1_at_p2 hpc_e hl_e hwf_e h19_rej (by decide) (by decide) (Or.inr (hm_e.symm ▸ hsub2')) h0
                            · have h := congrArg (run im (ks - rem)) h0
                              rw [← run_add, show rem + (ks - rem) = ks by omega, hrs, run_exit] at h; cases h
                          · exact hne_all rem
                      · -- `inR14`: `ea < 1376 < 4096` contradicts `wrMem_some_ge_4096`!
                        exfalso
                        obtain ⟨b, hb256, h14b⟩ := hstep_inv1.2.2.1 hr14
                        have hbase_eq : base = b := by
                          have hb' : ui.rdSP 64 14 = some base := hbase
                          simp only [St.rdSP, show (14 == 31) = false by decide, Bool.false_eq_true, ↓reduceIte,
                            h14b, Option.map_some, Option.some.injEq] at hb'
                          rw [← hb', Nat.mod_eq_of_lt (by rw [pow2_64]; omega)]
                        have hge := wrMem_some_ge_4096 ht.noLowMem hwr
                        rw [hbase_eq, Nat.mod_eq_of_lt (by unfold AArch64Env.W; omega)] at hge
                        omega
                      · -- `inR15`: `ea < 1376 < 4096` contradicts `wrMem_some_ge_4096`!
                        exfalso
                        obtain ⟨b, hb256, h15b⟩ := hstep_inv1.2.2.2 hr15
                        have hbase_eq : base = b := by
                          have hb' : ui.rdSP 64 15 = some base := hbase
                          simp only [St.rdSP, show (15 == 31) = false by decide, Bool.false_eq_true, ↓reduceIte,
                            h15b, Option.map_some, Option.some.injEq] at hb'
                          rw [← hb', Nat.mod_eq_of_lt (by rw [pow2_64]; omega)]
                        have hge := wrMem_some_ge_4096 ht.noLowMem hwr
                        rw [hbase_eq, Nat.mod_eq_of_lt (by unfold AArch64Env.W; omega)] at hge
                        omega
      -- Case 3: disturbance during Pass 2 (`N1 ≤ i < N1 + N2`)
      · by_cases hp2 : i < N1 + N2
        · let j2 := i - N1
          obtain ⟨ui, hui, hlo_i, hlt_i, hal_i, hl_i, hwf_i⟩ := hstay2 j2 (by omega)
          have ⟨hm_ui, hri_eq⟩ := pass_run_memSub (Or.inr (Or.inl ⟨rfl, rfl, rfl, rfl⟩)) hsub2_0 j2 { s1 with mem := m2 } ui
            rfl hl1m hwf_s1 hui (fun t ht' => hstay2 t (by omega))
          have hri2 : run im i (.ok s0) = .ok { ui with mem := s0.mem } := by
            rw [show i = N1 + j2 by omega, run_add, hr1_full, hri_eq]
          rw [hri2] at hri; cases hri
          obtain ⟨_, a20_ui, ha20_ui_lt, h20_ui, hmod20_ui⟩ := hnever2 j2 (by omega) ui hui
          have ha20_ui_ne : a20_ui ≠ ACCEPT := fun h => hmod20_ui (congrArg (· % 65536) h)
          have h19_ui : ui.reg 19 = some v := by
            clear hst hri2
            have hinv19 : ∀ t ≤ j2, ∀ u, run im t (.ok { s1 with mem := m2 }) = .ok u → u.reg 19 = some v := by
              intro t
              induction t with
              | zero => intro _ u hu; simp [run] at hu; subst hu; exact h19_1m
              | succ t' ih =>
                intro ht' u hu
                obtain ⟨ut, hut, hlo_t, hlt_t, hal_t, hl_t, hwf_t⟩ := hstay2 t' (by omega)
                rw [run_add, hut] at hu; simp only [run] at hu
                cases him : im ut.pc with
                | none => simp [step, him] at hu
                | some p =>
                  obtain ⟨ins, len⟩ := p
                  obtain ⟨rfl, hsafe⟩ := im_in_pass_safe (Or.inr (Or.inl ⟨rfl, rfl, rfl, rfl⟩)) hlo_t hlt_t him
                  have hx : exec ins 4 ut = .ok u := by simpa [step, him] using hu
                  have hr19 := ((exec_pass_props hsafe hlo_t hlt_t (by decide) hal_t hl_t hwf_t).2.1 u hx).2.2.2.2 19 (by simp) (by decide)
                  exact hr19.trans (ih (by omega) ut hut)
            exact hinv19 j2 (Nat.le_refl _) ui hui
          have ⟨_, hstep_inv2⟩ := clean_pass_step_inv (Or.inr ⟨rfl, rfl, rfl, rfl, rfl⟩) (s := { s1 with mem := m2 })
            hpc1m hby_m2 hs2.2.1 hlen_lt_W j2 ui hui (fun j hj => hstay2 j (by omega))
          have hr_split : run im (N2 - j2) (.ok ui) = .ok s2m := by
            have := hr2m; rw [show N2 = j2 + (N2 - j2) by omega, run_add, hui] at this; exact this
          have hpos : N2 - j2 = 1 + (N2 - (j2 + 1)) := by omega
          rw [hpos, run_add] at hr_split
          cases hstep2 : step im ui with
          | exit _ => simp [run, hstep2, run_exit] at hr_split
          | stuck => simp [run, hstep2, run_stuck] at hr_split
          | ok u1 =>
            simp only [run, hstep2] at hr_split
            cases him : im ui.pc with
            | none => simp [step, him] at hstep2
            | some p =>
              obtain ⟨ins, len⟩ := p
              obtain ⟨rfl, hsafe⟩ := im_in_pass_safe (Or.inr (Or.inl ⟨rfl, rfl, rfl, rfl⟩)) hlo_i hlt_i him
              have hx : exec ins 4 ui = .ok u1 := by simpa [step, him] using hstep2
              have ⟨_, hprops, _, hmono⟩ := exec_pass_props hsafe hlo_i hlt_i (by decide) hal_i hl_i hwf_i
              obtain ⟨hm1, hl1, hin1, hwf1', hreg1⟩ := hprops u1 hx
              have hm_u1 : u1.mem = m2 := hm1.trans hm_ui
              have hin1' : inPassOrNxt L_p2 L_p3 L_p3 u1.pc := by
                rcases hin1 with ⟨h1, h2, h3⟩ | h4
                · exact Or.inl ⟨by omega, h2, h3⟩
                · exact Or.inr h4
              have h19_u1 : u1.reg 19 = some v := (hreg1 19 (by simp) (by decide)).trans h19_ui
              have hstep_full : step im { ui with mem := s0.mem } = .ok { u1 with mem := s0.mem } := by
                simp [step, him, hmono s0.mem (hm_ui ▸ hsub2_0) u1 hx]
              have hstay2_rem : ∀ t < N2 - (j2 + 1), ∃ u, run im t (.ok u1) = .ok u ∧ L_p2 ≤ u.pc ∧ u.pc < L_p3 ∧
                  (u.pc - L_p2) % 4 = 0 ∧ u.regs.length = 31 ∧ VerdictRegsWf u := by
                intro t ht'
                obtain ⟨u, hu, h1, h2, h3, h4, h5⟩ := hstay2 (j2 + 1 + t) (by omega)
                rw [show j2 + 1 + t = j2 + (1 + t) by omega, run_add, hui, run_add] at hu
                simp only [run, hstep2] at hu
                exact ⟨u, hu, h1, h2, h3, h4, h5⟩
              cases hst with
              | clean =>
                rw [hstep_full]; intro h0
                obtain ⟨t, _, u3, _, hpc3, hm3, hl3, hwf3', hreg3, hend⟩ :=
                  pass_clean_escape (Or.inr (Or.inl ⟨rfl, rfl, rfl, rfl⟩)) rem { u1 with mem := s0.mem } hin1' hl1 hwf1' h0
                exact run_after_p3_with_19_ne_acc ht hinv hpc3 hl3 hwf3'
                  ((hreg3 19 (by simp) (by decide)).trans h19_u1) hv_lt hv_ne ((show u3.mem = s0.mem from hm3).symm ▸ hsub3_0) hend
              | nzcv s1_f s2_f hs_eq hnz =>
                rw [hstep_full] at hs_eq; cases hs_eq; intro h0
                obtain ⟨hpc_n, hregs_n, hmem_n⟩ := nzcvFlips_props hnz
                have hwf_n : VerdictRegsWf s2_f := fun r hr => by
                  obtain ⟨u, hu, hru⟩ := hwf1' r hr; exact ⟨u, hu, by simp [St.reg, hregs_n]; exact hru⟩
                have h19_n : s2_f.reg 19 = some v := by simp [St.reg, hregs_n]; exact h19_u1
                obtain ⟨t, _, u3, _, hpc3, hm3, hl3, hwf3', hreg3, hend⟩ :=
                  pass_clean_escape (Or.inr (Or.inl ⟨rfl, rfl, rfl, rfl⟩)) rem s2_f
                    (by rw [hpc_n]; exact hin1') (by rw [hregs_n]; exact hl1) hwf_n h0
                have hm3' : u3.mem = s0.mem := hm3.trans hmem_n
                exact run_after_p3_with_19_ne_acc ht hinv hpc3 hl3 hwf3'
                  ((hreg3 19 (by simp) (by decide)).trans h19_n) hv_lt hv_ne (hm3'.symm ▸ hsub3_0) hend
              | reg s1_f idx bit hs_eq _ hbit =>
                rw [hstep_full] at hs_eq; cases hs_eq; intro h0
                by_cases hidx19 : idx = 19
                · subst hidx19
                  have hfl_eq : ({ u1 with mem := s0.mem }.flipReg 19 bit) =
                      { u1 with mem := s0.mem }.setReg 19 (some (Nat.xor v (2 ^ bit))) := by
                    simp [St.flipReg, h19_u1]
                  rw [hfl_eq] at h0
                  have ⟨_, hrun2_full⟩ := pass_run_memSub (Or.inr (Or.inl ⟨rfl, rfl, rfl, rfl⟩)) hsub2_0
                    (N2 - (j2 + 1)) u1 s2m hm_u1 hl1 hwf1' hr_split hstay2_rem
                  have hstay2_full : ∀ t < N2 - (j2 + 1), ∃ u, run im t (.ok { u1 with mem := s0.mem }) = .ok u ∧
                      L_p2 ≤ u.pc ∧ u.pc < L_p3 ∧ (u.pc - L_p2) % 4 = 0 ∧ u.regs.length = 31 ∧ VerdictRegsWf u := by
                    intro t ht'
                    obtain ⟨u, hu, h1, h2, h3, h4, h5⟩ := hstay2_rem t ht'
                    have ⟨_, hu_f⟩ := pass_run_memSub (Or.inr (Or.inl ⟨rfl, rfl, rfl, rfl⟩)) hsub2_0 t u1 u hm_u1 hl1 hwf1' hu
                      (fun t' htt' => hstay2_rem t' (by omega))
                    exact ⟨{ u with mem := s0.mem }, hu_f, h1, h2, h3, h4, h5⟩
                  have hrun2_set := pass_run_unread_reg (Or.inr (Or.inl ⟨rfl, rfl, rfl, rfl⟩)) (by simp : (19 : Nat) ∈ [19, 20, 21])
                    (by decide) (some (Nat.xor v (2 ^ bit))) (N2 - (j2 + 1)) { u1 with mem := s0.mem } { s2m with mem := s0.mem }
                    hl1 hwf1' hrun2_full hstay2_full
                  by_cases hle : N2 - (j2 + 1) ≤ rem
                  · change run im rem _ = .exit 0 at h0
                    rw [show rem = (N2 - (j2 + 1)) + (rem - (N2 - (j2 + 1))) by omega, run_add, hrun2_set] at h0
                    obtain ⟨k3, u3, hr3', hf3', hpc3', h21_3', hl3', _⟩ :=
                      p3_spec ht.hD3_ptr ht.hD3_len ht.hBuf3 ht.bytes ht.hSm3 (by simp [hl2m]) (by simp [hpc2m])
                        (s := ({ s2m with mem := s0.mem }).setReg 19 (some (Nat.xor v (2 ^ bit)))) rfl
                    have h19_3' : u3.reg 19 = some (Nat.xor v (2 ^ bit)) := by
                      rw [hf3'.regs 19 (by decide)]; simp [hl2m]
                    have h20_3' : u3.reg 20 = some v := by
                      rw [hf3'.regs 20 (by decide), reg_setReg_ne (by decide)]; exact h20_2m
                    have hxor_lt : Nat.xor v (2 ^ bit) < AArch64Env.W :=
                      Nat.xor_lt_two_pow hv_lt (Nat.pow_lt_pow_right (by decide) hbit)
                    by_cases hle3 : k3 ≤ rem - (N2 - (j2 + 1))
                    · rw [show rem - (N2 - (j2 + 1)) = k3 + (rem - (N2 - (j2 + 1)) - k3) by omega, run_add, hr3'] at h0
                      exact vote_two_ne_accept hl3' hpc3' h19_3' h20_3' h21_3'
                        hxor_lt hv_lt hv_lt (Or.inr (Or.inr ⟨hv_ne, hv_ne⟩)) _ h0
                    · have h := congrArg (run im (k3 - (rem - (N2 - (j2 + 1))))) h0
                      rw [← run_add, show rem - (N2 - (j2 + 1)) + (k3 - (rem - (N2 - (j2 + 1)))) = k3 by omega, hr3', run_exit] at h; cases h
                  · have h := congrArg (run im ((N2 - (j2 + 1)) - rem)) h0
                    rw [← run_add, show n - 1 - i + ((N2 - (j2 + 1)) - rem) = N2 - (j2 + 1) by omega, hrun2_set, run_exit] at h; cases h
                · obtain ⟨t, _, u3, _, hpc3, hm3, hl3, hwf3', hreg3, hend⟩ :=
                    pass_clean_escape (Or.inr (Or.inl ⟨rfl, rfl, rfl, rfl⟩)) rem ({ u1 with mem := s0.mem }.flipReg idx bit)
                      (by rw [flipReg_pc]; exact hin1')
                      (flipReg_len (s := { u1 with mem := s0.mem }) hl1 idx bit)
                      (flipReg_verdictWf (s := { u1 with mem := s0.mem }) hwf1' hl1 idx bit hbit) h0
                  have h19_u3 : u3.reg 19 = some v :=
                    (hreg3 19 (by simp) (by decide)).trans ((flipReg_reg_ne (s := { u1 with mem := s0.mem }) hidx19 bit).trans h19_u1)
                  have hm3' : u3.mem = s0.mem := hm3.trans (flipReg_mem (s := { u1 with mem := s0.mem }) idx bit)
                  exact run_after_p3_with_19_ne_acc ht hinv hpc3 hl3 hwf3' h19_u3 hv_lt hv_ne (hm3'.symm ▸ hsub3_0) hend
              | mem s1_f idx j bit hs_eq =>
                rw [hstep_full] at hs_eq; cases hs_eq; intro h0
                let m_f := (s0.flipMem idx j bit).mem
                have hmf_eq : ({ u1 with mem := s0.mem }.flipMem idx j bit) = { u1 with mem := m_f } := by
                  unfold St.flipMem; rfl
                rw [hmf_eq] at h0
                rcases flipMem_preserves_two_subMems ht idx j bit with ⟨_, hsub3'⟩ | ⟨_, hsub3'⟩ | ⟨_, hsub2'⟩
                · obtain ⟨t, _, u3, _, hpc3, hm3, hl3, hwf3', hreg3, hend⟩ :=
                    pass_clean_escape (Or.inr (Or.inl ⟨rfl, rfl, rfl, rfl⟩)) rem { u1 with mem := m_f } hin1' hl1 hwf1' h0
                  exact run_after_p3_with_19_ne_acc ht hinv hpc3 hl3 hwf3'
                    ((hreg3 19 (by simp) (by decide)).trans h19_u1) hv_lt hv_ne ((show u3.mem = m_f from hm3).symm ▸ hsub3') hend
                · obtain ⟨t, _, u3, _, hpc3, hm3, hl3, hwf3', hreg3, hend⟩ :=
                    pass_clean_escape (Or.inr (Or.inl ⟨rfl, rfl, rfl, rfl⟩)) rem { u1 with mem := m_f } hin1' hl1 hwf1' h0
                  exact run_after_p3_with_19_ne_acc ht hinv hpc3 hl3 hwf3'
                    ((hreg3 19 (by simp) (by decide)).trans h19_u1) hv_lt hv_ne ((show u3.mem = m_f from hm3).symm ▸ hsub3') hend
                · have ⟨_, hr2'⟩ := pass_run_memSub (Or.inr (Or.inl ⟨rfl, rfl, rfl, rfl⟩)) hsub2' (N2 - (j2 + 1)) u1 s2m
                    hm_u1 hl1 hwf1' hr_split hstay2_rem
                  by_cases hle : N2 - (j2 + 1) ≤ rem
                  · change run im rem _ = .exit 0 at h0
                    rw [show rem = (N2 - (j2 + 1)) + (rem - (N2 - (j2 + 1))) by omega, run_add, hr2'] at h0
                    exact run_after_p3_clean_with_19_20_ne_acc (u := { s2m with mem := m_f })
                      (Or.inl (Or.inl ⟨by simp [hpc2m], by simp [hpc2m]; decide, by simp [hpc2m]⟩))
                      hl2m hwf2 h19_2 hv_lt hv_ne h20_2m hv_lt hv_ne h0
                  · have h := congrArg (run im ((N2 - (j2 + 1)) - rem)) h0
                    rw [← run_add, show n - 1 - i + ((N2 - (j2 + 1)) - rem) = N2 - (j2 + 1) by omega, hr2', run_exit] at h; cases h
              | fetch _ _ wAlt _ hw =>
                have hlo1_i : L_p1 ≤ ui.pc := Nat.le_trans (by decide : L_p1 ≤ L_p2) hlo_i
                have hlt3_i : ui.pc < L_p3_end := Nat.lt_trans hlt_i (by decide : L_p3 < L_p3_end)
                have hal1_i : (ui.pc - L_p1) % 4 = 0 := by
                  have h12 : L_p1 + 2728 = L_p2 := by decide
                  omega
                obtain ⟨wOrig, bit, _, him_w, hwAlt, halt⟩ :=
                  alts_in_corePass_audit (pc := ui.pc) hlo1_i hlt3_i hal1_i hw
                rw [hwAlt]
                cases hdec : decodeWord (Nat.xor wOrig (2 ^ bit)) with
                | none => simp [execO, run_stuck]
                | some ins' =>
                  simp only [execO, Option.map_some]
                  cases hx' : exec ins' 4 { ui with mem := s0.mem } with
                  | exit c => exact absurd hx' (exec_passFetch_ne_exit halt hdec c)
                  | stuck => simp [run_stuck]
                  | ok s' =>
                    have ⟨hl', hwf', hcase⟩ := exec_passFetch_step (s := { ui with mem := s0.mem })
                      hlo1_i hlt3_i hl_i hwf_i
                      h19_ui hv_lt hv_ne h20_ui ha20_ui_lt ha20_ui_ne
                      (fun (h : L_p3 ≤ ui.pc) => by omega) halt hdec hx'
                    dsimp only at hcase
                    rcases hcase with ⟨hpc', hm', h19_f, _⟩ | ⟨hm', h19', h20', _, hbr⟩ | ⟨hp3, _, _⟩ | ⟨_, hpc', hregs', hsp', hk', hfl', hstr⟩
                    · intro h0
                      obtain ⟨b19, hb19_lt, hb19, hb19_ne⟩ := h19_f hlo_i
                      have hin_s' : inPassOrNxt L_p2 L_p3 L_p3 s'.pc := by
                        rw [hpc']; exact step4_inPassOrNxt hlo_i hlt_i hal_i (by decide)
                      obtain ⟨t, _, u3, _, hpc3, hm3, hl3, hwf3', hreg3, hend⟩ :=
                        pass_clean_escape (Or.inr (Or.inl ⟨rfl, rfl, rfl, rfl⟩)) rem s' hin_s' hl' hwf' h0
                      exact run_after_p3_with_19_ne_acc ht hinv hpc3 hl3 hwf3'
                        ((hreg3 19 (by simp) (by decide)).trans hb19) hb19_lt hb19_ne ((hm3.trans hm').symm ▸ hsub3_0) hend
                    · rcases hbr with hin_p | hnone | ⟨hp3, _⟩
                      · exact run_in_corePass_with_19_20_ne_acc ht hinv hin_p hl' hwf'
                          (h19'.trans h19_ui) hv_lt hv_ne (h20'.trans h20_ui) ha20_ui_lt ha20_ui_ne
                          (hm'.symm ▸ hsub2_0) (hm'.symm ▸ hsub3_0)
                      · exact run_im_none_ne_exit0 (targetImNone_sound hnone) rem
                      · omega
                    · omega
                    · rcases hstr with ⟨sz, rt, rn, offS, rt0, base, val_s, hpair_ls, hbase, hsp, hwr⟩ |
                        ⟨rt, rn, imm12, base, val_s, _, horig_cmp, himm12, hbase, hwr, hcases_cmp⟩
                      · -- Clean `ldr`/`ldur` flipped to `str`/`stur` at same `(ea, sz)` -> `m3` preserved!
                        have hbase_ui : ui.rdSP 64 rn = some base := hbase
                        have hrd_m2 : ∃ val0, rdMem m2 ((base + offS) % AArch64Env.W) sz = some val0 := by
                          rcases hpair_ls with ⟨_, horig⟩ | ⟨_, horig⟩
                          · rw [horig] at him_w; rw [him] at him_w; cases him_w
                            simp only [exec, lift, bind, Option.bind, pure, hbase_ui, if_neg hsp, hm_ui] at hx
                            cases hrd : rdMem m2 ((base + offS) % AArch64Env.W) sz with
                            | none => simp [hrd] at hx
                            | some v0 => exact ⟨v0, rfl⟩
                          · rw [horig] at him_w; rw [him] at him_w; cases him_w
                            simp only [exec, lift, bind, Option.bind, pure, hbase_ui, if_neg hsp, hm_ui] at hx
                            cases hrd : rdMem m2 ((base + offS) % AArch64Env.W) sz with
                            | none => simp [hrd] at hx
                            | some v0 => exact ⟨v0, rfl⟩
                        obtain ⟨val0, hrd0⟩ := hrd_m2
                        have hsub3' : MemSub m3 s'.mem := wrMem_passSubMem_preserves_other ht.memWf
                          ht.hD2_ptr ht.hD2_len ht.hBuf2 (ht.disjD2 0x601000 (by simp))
                          ht.hD3_ptr ht.hD3_len ht.hBuf3 (ht.disjD3 0x602000 (by simp)) hdisj23 hrd0 hwr
                        intro h0
                        have hin_s' : inPassOrNxt L_p2 L_p3 L_p3 s'.pc := by
                          rw [hpc']; by_cases hlt' : ui.pc + 4 < L_p3
                          · exact Or.inl ⟨by omega, hlt', by omega⟩
                          · exact Or.inr (by unfold L_p2 L_p3 at *; omega)
                        obtain ⟨t, _, u3, _, hpc3, hm3, hl3, hwf3', hreg3, hend⟩ :=
                          pass_clean_escape (Or.inr (Or.inl ⟨rfl, rfl, rfl, rfl⟩)) rem s' hin_s' hl' hwf' h0
                        have h19_s' : s'.reg 19 = some v := by simp [St.reg, hregs']; exact h19_ui
                        exact run_after_p3_with_19_ne_acc ht hinv hpc3 hl3 hwf3'
                          ((hreg3 19 (by simp) (by decide)).trans h19_s') hv_lt hv_ne (hm3.symm ▸ hsub3') hend
                      · -- Clean `cmp` flipped to `str .x8`
                        have hnot_lt_p2 : ¬ ui.pc < L_p2 := by omega
                        simp only [hnot_lt_p2, ↓reduceIte] at hcases_cmp
                        rcases hcases_cmp with ⟨hoff16, rfl, rfl⟩ | ⟨hr14, rfl⟩ | ⟨hr15, rfl⟩
                        · -- `off = 16`: `cmp x14, #16` -> `str xzr, [x14, #128]`
                          have hpc16 : ui.pc = L_p2 + 16 := by omega
                          have h14_len : ui.reg 14 = some bs.length := hstep_inv2.2.1 (Or.inr hpc16)
                          have hbase_eq : base = bs.length := by
                            have hb' : ui.rdSP 64 14 = some base := hbase
                            simp only [St.rdSP, show (14 == 31) = false by decide, Bool.false_eq_true, ↓reduceIte,
                              h14_len, Option.map_some, Option.some.injEq] at hb'
                            rw [← hb', Nat.mod_eq_of_lt (by rw [pow2_64]; unfold AArch64Env.W at hlen_lt_W; exact hlen_lt_W)]
                          subst hbase_eq
                          have hea_eq : (bs.length + 16 * 8) % AArch64Env.W = bs.length + 128 := by
                            have := ht.hSm1; unfold HMAX AArch64Env.W at *; omega
                          rw [hea_eq] at hwr
                          obtain ⟨h16_le, htwo⟩ := wrMem_cmp16_preserves_two ht hwr
                          have h19_s' : s'.reg 19 = some v := by simp [St.reg, hregs']; exact h19_ui
                          rcases htwo with ⟨_, hsub3'⟩ | ⟨_, hsub3'⟩ | ⟨_, hsub2'⟩
                          · intro h0
                            have hin_s' : inPassOrNxt L_p2 L_p3 L_p3 s'.pc := by
                              rw [hpc', hpc16]; exact Or.inl ⟨by omega, by unfold L_p2 L_p3; decide, by decide⟩
                            obtain ⟨t, _, u3, _, hpc3, hm3, hl3, hwf3', hreg3, hend⟩ :=
                              pass_clean_escape (Or.inr (Or.inl ⟨rfl, rfl, rfl, rfl⟩)) rem s' hin_s' hl' hwf' h0
                            exact run_after_p3_with_19_ne_acc ht hinv hpc3 hl3 hwf3'
                              ((hreg3 19 (by simp) (by decide)).trans h19_s') hv_lt hv_ne (hm3.symm ▸ hsub3') hend
                          · intro h0
                            have hin_s' : inPassOrNxt L_p2 L_p3 L_p3 s'.pc := by
                              rw [hpc', hpc16]; exact Or.inl ⟨by omega, by unfold L_p2 L_p3; decide, by decide⟩
                            obtain ⟨t, _, u3, _, hpc3, hm3, hl3, hwf3', hreg3, hend⟩ :=
                              pass_clean_escape (Or.inr (Or.inl ⟨rfl, rfl, rfl, rfl⟩)) rem s' hin_s' hl' hwf' h0
                            exact run_after_p3_with_19_ne_acc ht hinv hpc3 hl3 hwf3'
                              ((hreg3 19 (by simp) (by decide)).trans h19_s') hv_lt hv_ne (hm3.symm ▸ hsub3') hend
                          · -- `MemSub m2 s'.mem`: synchronize via `pass_off20_sync` -> `L_p3` with `x19, x20 ≠ ACCEPT`
                            rw [horig_cmp] at him_w; rw [him] at him_w; cases him_w
                            have h14_sp : ui.rdSP 64 14 = some bs.length := by
                              simpa [St.rdSP, show (14 == 31) = false by decide, h14_len] using
                                Nat.mod_eq_of_lt (by rw [pow2_64]; unfold AArch64Env.W at hlen_lt_W; exact hlen_lt_W)
                            have hu1_eq : u1 = { ui with pc := L_p2 + 20, fl := (addSubOp true 64 bs.length 16).2 } := by
                              simp only [exec, lift, bind, Option.bind, pure, h14_sp,
                                show (16 : Nat) % pow2 64 = 16 by decide, ↓reduceIte,
                                hpc16, show (L_p2 + 16 + 4) % AArch64Env.W = L_p2 + 20 by decide, St.wrZR] at hx
                              cases hx; rfl
                            have hu1_flc : u1.fl.c = some true := by
                              rw [hu1_eq]; simp [addSubOp, h16_le]
                            rcases pass_off20_sync (Or.inr ⟨rfl, rfl, rfl, rfl⟩) hsub2'
                                (by rw [hu1_eq]) hm_u1 hu1_flc (by rw [hpc', hpc16])
                                (by rw [hregs', hu1_eq]) (by rw [hsp', hu1_eq]) (by rw [hk', hu1_eq]) rfl
                                hl1 hwf1' hr_split hpc2m hstay2_rem with
                              ⟨ks, u_end, hrs, hpc_e, _, hl_e, hwf_e, hcases_e⟩ | hne_all
                            · intro h0
                              change run im rem _ = .exit 0 at h0
                              by_cases hle : ks ≤ rem
                              · rw [show rem = ks + (rem - ks) by omega, run_add, hrs] at h0
                                have hin3 : inPassOrNxt L_p3 L_p3_end L_vote u_end.pc ∨ u_end.pc = L_p3_end :=
                                  Or.inl (Or.inl ⟨by rw [hpc_e]; exact Nat.le_refl _, by rw [hpc_e]; decide, by rw [hpc_e]; decide⟩)
                                rcases hcases_e with rfl | ⟨h20_rej, hreg_ne⟩
                                · exact run_after_p3_clean_with_19_20_ne_acc hin3 hl_e hwf_e
                                    h19_2 hv_lt hv_ne h20_2m hv_lt hv_ne h0
                                · exact run_after_p3_clean_with_19_20_ne_acc hin3 hl_e hwf_e
                                    ((hreg_ne 19 (by decide)).trans h19_u1) hv_lt hv_ne h20_rej (by decide) (by decide) h0
                              · have h := congrArg (run im (ks - rem)) h0
                                rw [← run_add, show rem + (ks - rem) = ks by omega, hrs, run_exit] at h; cases h
                            · exact hne_all rem
                        · exfalso
                          obtain ⟨b, hb256, h14b⟩ := hstep_inv2.2.2.1 hr14
                          have hbase_eq : base = b := by
                            have hb' : ui.rdSP 64 14 = some base := hbase
                            simp only [St.rdSP, show (14 == 31) = false by decide, Bool.false_eq_true, ↓reduceIte,
                              h14b, Option.map_some, Option.some.injEq] at hb'
                            rw [← hb', Nat.mod_eq_of_lt (by rw [pow2_64]; omega)]
                          have hge := wrMem_some_ge_4096 ht.noLowMem hwr
                          rw [hbase_eq, Nat.mod_eq_of_lt (by unfold AArch64Env.W; omega)] at hge
                          omega
                        · exfalso
                          obtain ⟨b, hb256, h15b⟩ := hstep_inv2.2.2.2 hr15
                          have hbase_eq : base = b := by
                            have hb' : ui.rdSP 64 15 = some base := hbase
                            simp only [St.rdSP, show (15 == 31) = false by decide, Bool.false_eq_true, ↓reduceIte,
                              h15b, Option.map_some, Option.some.injEq] at hb'
                            rw [← hb', Nat.mod_eq_of_lt (by rw [pow2_64]; omega)]
                          have hge := wrMem_some_ge_4096 ht.noLowMem hwr
                          rw [hbase_eq, Nat.mod_eq_of_lt (by unfold AArch64Env.W; omega)] at hge
                          omega
        -- Case 4: disturbance during Pass 3 (`N1 + N2 ≤ i < N1 + N2 + N3`)
        · let j3 := i - (N1 + N2)
          obtain ⟨ui, hui, hlo_i, hlt_i, hal_i, hl_i, hwf_i⟩ := hstay3 j3 (by omega)
          have ⟨hm_ui, hri_eq⟩ := pass_run_memSub (Or.inr (Or.inr ⟨rfl, rfl, rfl, rfl⟩)) hsub3_0 j3 { s2 with mem := m3 } ui
            rfl hl2m hwf2 hui (fun t ht' => hstay3 t (by omega))
          have hri3 : run im i (.ok s0) = .ok { ui with mem := s0.mem } := by
            rw [show i = N1 + N2 + j3 by omega, run_add, run_add, hr1_full, hr2_full, hri_eq]
          rw [hri3] at hri; cases hri
          have h1920_ui : ui.reg 19 = some v ∧ ui.reg 20 = some v := by
            clear hst hri3
            have hinv1920 : ∀ t ≤ j3, ∀ u, run im t (.ok { s2 with mem := m3 }) = .ok u → u.reg 19 = some v ∧ u.reg 20 = some v := by
              intro t
              induction t with
              | zero => intro _ u hu; simp [run] at hu; subst hu; exact ⟨h19_2, h20_2m⟩
              | succ t' ih =>
                intro ht' u hu
                obtain ⟨ut, hut, hlo_t, hlt_t, hal_t, hl_t, hwf_t⟩ := hstay3 t' (by omega)
                rw [run_add, hut] at hu; simp only [run] at hu
                cases him : im ut.pc with
                | none => simp [step, him] at hu
                | some p =>
                  obtain ⟨ins, len⟩ := p
                  obtain ⟨rfl, hsafe⟩ := im_in_pass_safe (Or.inr (Or.inr ⟨rfl, rfl, rfl, rfl⟩)) hlo_t hlt_t him
                  have hx : exec ins 4 ut = .ok u := by simpa [step, him] using hu
                  have hreg := ((exec_pass_props hsafe hlo_t hlt_t (by decide) hal_t hl_t hwf_t).2.1 u hx).2.2.2.2
                  have ⟨h19t, h20t⟩ := ih (by omega) ut hut
                  exact ⟨(hreg 19 (by simp) (by decide)).trans h19t, (hreg 20 (by simp) (by decide)).trans h20t⟩
            exact hinv1920 j3 (Nat.le_refl _) ui hui
          have hr_split : run im (N3 - j3) (.ok ui) = .ok s3m := by
            have := hr3m; rw [show N3 = j3 + (N3 - j3) by omega, run_add, hui] at this; exact this
          have hpos : N3 - j3 = 1 + (N3 - (j3 + 1)) := by omega
          rw [hpos, run_add] at hr_split
          cases hstep3 : step im ui with
          | exit _ => simp [run, hstep3, run_exit] at hr_split
          | stuck => simp [run, hstep3, run_stuck] at hr_split
          | ok u1 =>
            simp only [run, hstep3] at hr_split
            cases him : im ui.pc with
            | none => simp [step, him] at hstep3
            | some p =>
              obtain ⟨ins, len⟩ := p
              obtain ⟨rfl, hsafe⟩ := im_in_pass_safe (Or.inr (Or.inr ⟨rfl, rfl, rfl, rfl⟩)) hlo_i hlt_i him
              have hx : exec ins 4 ui = .ok u1 := by simpa [step, him] using hstep3
              have ⟨_, hprops, _, hmono⟩ := exec_pass_props hsafe hlo_i hlt_i (by decide) hal_i hl_i hwf_i
              obtain ⟨hm1, hl1, hin1, hwf1', hreg1⟩ := hprops u1 hx
              have hm_u1 : u1.mem = m3 := hm1.trans hm_ui
              have hin1' : inPassOrNxt L_p3 L_p3_end L_vote u1.pc := by
                rcases hin1 with ⟨h1, h2, h3⟩ | h4
                · exact Or.inl ⟨by omega, h2, h3⟩
                · exact Or.inr h4
              have h19_u1 : u1.reg 19 = some v := (hreg1 19 (by simp) (by decide)).trans h1920_ui.1
              have h20_u1 : u1.reg 20 = some v := (hreg1 20 (by simp) (by decide)).trans h1920_ui.2
              have hstep_full : step im { ui with mem := s0.mem } = .ok { u1 with mem := s0.mem } := by
                simp [step, him, hmono s0.mem (hm_ui ▸ hsub3_0) u1 hx]
              have hstay3_rem : ∀ t < N3 - (j3 + 1), ∃ u, run im t (.ok u1) = .ok u ∧ L_p3 ≤ u.pc ∧ u.pc < L_p3_end ∧
                  (u.pc - L_p3) % 4 = 0 ∧ u.regs.length = 31 ∧ VerdictRegsWf u := by
                intro t ht'
                obtain ⟨u, hu, h1, h2, h3, h4, h5⟩ := hstay3 (j3 + 1 + t) (by omega)
                rw [show j3 + 1 + t = j3 + (1 + t) by omega, run_add, hui, run_add] at hu
                simp only [run, hstep3] at hu
                exact ⟨u, hu, h1, h2, h3, h4, h5⟩
              cases hst with
              | clean =>
                rw [hstep_full]
                exact run_after_p3_clean_with_19_20_ne_acc (Or.inl hin1') hl1 hwf1' h19_u1 hv_lt hv_ne h20_u1 hv_lt hv_ne
              | mem s1_f idx j bit hs_eq =>
                rw [hstep_full] at hs_eq; cases hs_eq
                exact run_after_p3_clean_with_19_20_ne_acc (Or.inl (by unfold St.flipMem; exact hin1'))
                  (by unfold St.flipMem; exact hl1) (by unfold St.flipMem; exact hwf1')
                  (by unfold St.flipMem; exact h19_u1) hv_lt hv_ne
                  (by unfold St.flipMem; exact h20_u1) hv_lt hv_ne
              | nzcv s1_f s2_f hs_eq hnz =>
                rw [hstep_full] at hs_eq; cases hs_eq
                obtain ⟨hpc_n, hregs_n, _⟩ := nzcvFlips_props hnz
                have hwf_n : VerdictRegsWf s2_f := fun r hr => by
                  obtain ⟨u, hu, hru⟩ := hwf1' r hr; exact ⟨u, hu, by simp [St.reg, hregs_n]; exact hru⟩
                exact run_after_p3_clean_with_19_20_ne_acc (Or.inl (by rw [hpc_n]; exact hin1'))
                  (by rw [hregs_n]; exact hl1) hwf_n
                  (by simp [St.reg, hregs_n]; exact h19_u1) hv_lt hv_ne
                  (by simp [St.reg, hregs_n]; exact h20_u1) hv_lt hv_ne
              | reg s1_f idx bit hs_eq _ hbit =>
                rw [hstep_full] at hs_eq; cases hs_eq; intro h0
                by_cases h1920 : idx = 19 ∨ idx = 20
                · have ⟨_, hrun3_full⟩ := pass_run_memSub (Or.inr (Or.inr ⟨rfl, rfl, rfl, rfl⟩)) hsub3_0
                    (N3 - (j3 + 1)) u1 s3m hm_u1 hl1 hwf1' hr_split hstay3_rem
                  have hstay3_full : ∀ t < N3 - (j3 + 1), ∃ u, run im t (.ok { u1 with mem := s0.mem }) = .ok u ∧
                      L_p3 ≤ u.pc ∧ u.pc < L_p3_end ∧ (u.pc - L_p3) % 4 = 0 ∧ u.regs.length = 31 ∧ VerdictRegsWf u := by
                    intro t ht'
                    obtain ⟨u, hu, h1, h2, h3, h4, h5⟩ := hstay3_rem t ht'
                    have ⟨_, hu_f⟩ := pass_run_memSub (Or.inr (Or.inr ⟨rfl, rfl, rfl, rfl⟩)) hsub3_0 t u1 u hm_u1 hl1 hwf1' hu
                      (fun t' htt' => hstay3_rem t' (by omega))
                    exact ⟨{ u with mem := s0.mem }, hu_f, h1, h2, h3, h4, h5⟩
                  have hxor_lt : Nat.xor v (2 ^ bit) < AArch64Env.W :=
                    Nat.xor_lt_two_pow hv_lt (Nat.pow_lt_pow_right (by decide) hbit)
                  rcases h1920 with rfl | rfl
                  · have hfl_eq : ({ u1 with mem := s0.mem }.flipReg 19 bit) =
                        { u1 with mem := s0.mem }.setReg 19 (some (Nat.xor v (2 ^ bit))) := by
                      simp [St.flipReg, h19_u1]
                    rw [hfl_eq] at h0
                    have hrun3_set := pass_run_unread_reg (Or.inr (Or.inr ⟨rfl, rfl, rfl, rfl⟩)) (by simp : (19 : Nat) ∈ [19, 20, 21])
                      (by decide) (some (Nat.xor v (2 ^ bit))) (N3 - (j3 + 1)) { u1 with mem := s0.mem } { s3m with mem := s0.mem }
                      hl1 hwf1' hrun3_full hstay3_full
                    by_cases hle : N3 - (j3 + 1) ≤ rem
                    · change run im rem _ = .exit 0 at h0
                      rw [show rem = (N3 - (j3 + 1)) + (rem - (N3 - (j3 + 1))) by omega, run_add, hrun3_set] at h0
                      exact vote_two_ne_accept (by simp [hl3m]) (by simp [hpc3m])
                        (by simp [hl3m]) (by rw [reg_setReg_ne (by decide)]; exact h20_3)
                        (by rw [reg_setReg_ne (by decide)]; exact h21_3m)
                        hxor_lt hv_lt hv_lt (Or.inr (Or.inr ⟨hv_ne, hv_ne⟩)) _ h0
                    · have h := congrArg (run im ((N3 - (j3 + 1)) - rem)) h0
                      rw [← run_add, show n - 1 - i + ((N3 - (j3 + 1)) - rem) = N3 - (j3 + 1) by omega, hrun3_set, run_exit] at h; cases h
                  · have hfl_eq : ({ u1 with mem := s0.mem }.flipReg 20 bit) =
                        { u1 with mem := s0.mem }.setReg 20 (some (Nat.xor v (2 ^ bit))) := by
                      simp [St.flipReg, h20_u1]
                    rw [hfl_eq] at h0
                    have hrun3_set := pass_run_unread_reg (Or.inr (Or.inr ⟨rfl, rfl, rfl, rfl⟩)) (by simp : (20 : Nat) ∈ [19, 20, 21])
                      (by decide) (some (Nat.xor v (2 ^ bit))) (N3 - (j3 + 1)) { u1 with mem := s0.mem } { s3m with mem := s0.mem }
                      hl1 hwf1' hrun3_full hstay3_full
                    by_cases hle : N3 - (j3 + 1) ≤ rem
                    · change run im rem _ = .exit 0 at h0
                      rw [show rem = (N3 - (j3 + 1)) + (rem - (N3 - (j3 + 1))) by omega, run_add, hrun3_set] at h0
                      exact vote_two_ne_accept (by simp [hl3m]) (by simp [hpc3m])
                        (by rw [reg_setReg_ne (by decide)]; exact h19_3) (by simp [hl3m])
                        (by rw [reg_setReg_ne (by decide)]; exact h21_3m)
                        hv_lt hxor_lt hv_lt (Or.inr (Or.inl ⟨hv_ne, hv_ne⟩)) _ h0
                    · have h := congrArg (run im ((N3 - (j3 + 1)) - rem)) h0
                      rw [← run_add, show n - 1 - i + ((N3 - (j3 + 1)) - rem) = N3 - (j3 + 1) by omega, hrun3_set, run_exit] at h; cases h
                · have hne19 : idx ≠ 19 := fun h => h1920 (Or.inl h)
                  have hne20 : idx ≠ 20 := fun h => h1920 (Or.inr h)
                  exact run_after_p3_clean_with_19_20_ne_acc (Or.inl (by rw [flipReg_pc]; exact hin1'))
                    (flipReg_len (s := { u1 with mem := s0.mem }) hl1 idx bit)
                    (flipReg_verdictWf (s := { u1 with mem := s0.mem }) hwf1' hl1 idx bit hbit)
                    ((flipReg_reg_ne (s := { u1 with mem := s0.mem }) hne19 bit).trans h19_u1) hv_lt hv_ne
                    ((flipReg_reg_ne (s := { u1 with mem := s0.mem }) hne20 bit).trans h20_u1) hv_lt hv_ne h0
              | fetch _ _ wAlt _ hw =>
                have hlo1_i : L_p1 ≤ ui.pc := Nat.le_trans (by decide : L_p1 ≤ L_p3) hlo_i
                have hal1_i : (ui.pc - L_p1) % 4 = 0 := by
                  have h13 : L_p1 + 5456 = L_p3 := by decide
                  omega
                obtain ⟨wOrig, bit, _, _, hwAlt, halt⟩ :=
                  alts_in_corePass_audit (pc := ui.pc) hlo1_i hlt_i hal1_i hw
                rw [hwAlt]
                cases hdec : decodeWord (Nat.xor wOrig (2 ^ bit)) with
                | none => simp [execO, run_stuck]
                | some ins' =>
                  simp only [execO, Option.map_some]
                  cases hx' : exec ins' 4 { ui with mem := s0.mem } with
                  | exit c => exact absurd hx' (exec_passFetch_ne_exit halt hdec c)
                  | stuck => simp [run_stuck]
                  | ok s' =>
                    have ⟨hl', hwf', hcase⟩ := exec_passFetch_step (s := { ui with mem := s0.mem })
                      hlo1_i hlt_i hl_i hwf_i
                      h1920_ui.1 hv_lt hv_ne h1920_ui.2 hv_lt hv_ne
                      (fun _ => hv_ro) halt hdec hx'
                    dsimp only at hcase
                    rcases hcase with ⟨hpc', _, h19_f, h20_f⟩ | ⟨hm', h19', h20', _, hbr⟩ | ⟨_, hpc', hregs'⟩ | ⟨hlt3, _, _⟩
                    · obtain ⟨b19, hb19_lt, hb19, hb19_ne⟩ := h19_f (by unfold L_p2 L_p3 at *; omega)
                      obtain ⟨b20, hb20_lt, hb20, hb20_ne⟩ := h20_f hlo_i
                      have hin_s' : inPassOrNxt L_p3 L_p3_end L_vote s'.pc ∨ s'.pc = L_p3_end := by
                        rw [hpc']; exact step4_inP3OrEnd hlo_i hlt_i hal_i
                      exact run_after_p3_clean_with_19_20_ne_acc hin_s' hl' hwf' hb19 hb19_lt hb19_ne hb20 hb20_lt hb20_ne
                    · rcases hbr with hin_p | hnone | ⟨_, htail_k⟩
                      · exact run_in_corePass_with_19_20_ne_acc ht hinv hin_p hl' hwf'
                          (h19'.trans h1920_ui.1) hv_lt hv_ne (h20'.trans h1920_ui.2) hv_lt hv_ne
                          (hm'.symm ▸ hsub2_0) (hm'.symm ▸ hsub3_0)
                      · exact run_im_none_ne_exit0 (targetImNone_sound hnone) rem
                      · obtain ⟨c21, hc21_lt, h21'⟩ := hwf' 21 (by simp)
                        unfold targetInTailK01248 at htail_k
                        simp only [Bool.or_eq_true, beq_iff_eq] at htail_k
                        rcases htail_k with (((hk0 | hk1) | hk2) | hk4) | hk8
                        · exact tail_entry_k01248_ne_exit0 (Or.inl rfl) hk0 hl'
                            (h19'.trans h1920_ui.1) (h20'.trans h1920_ui.2) h21' hv_notAcc hc21_lt rem
                        · exact tail_entry_k01248_ne_exit0 (Or.inr (Or.inl rfl)) hk1 hl'
                            (h19'.trans h1920_ui.1) (h20'.trans h1920_ui.2) h21' hv_notAcc hc21_lt rem
                        · exact tail_entry_k01248_ne_exit0 (Or.inr (Or.inr (Or.inl rfl))) hk2 hl'
                            (h19'.trans h1920_ui.1) (h20'.trans h1920_ui.2) h21' hv_notAcc hc21_lt rem
                        · exact tail_entry_k01248_ne_exit0 (Or.inr (Or.inr (Or.inr (Or.inl rfl)))) hk4 hl'
                            (h19'.trans h1920_ui.1) (h20'.trans h1920_ui.2) h21' hv_notAcc hc21_lt rem
                        · exact tail_entry_k01248_ne_exit0 (Or.inr (Or.inr (Or.inr (Or.inr rfl)))) hk8 hl'
                            (h19'.trans h1920_ui.1) (h20'.trans h1920_ui.2) h21' hv_notAcc hc21_lt rem
                    · have hin_s' : inPassOrNxt L_p3 L_p3_end L_vote s'.pc ∨ s'.pc = L_p3_end := by
                        rw [hpc']; by_cases hlt' : ui.pc + 4 < L_p3_end
                        · exact Or.inl (Or.inl ⟨by omega, hlt', by omega⟩)
                        · exact Or.inr (by unfold L_p3 L_p3_end at *; omega)
                      exact run_after_p3_clean_with_19_20_ne_acc hin_s' hl' hwf'
                        (by simp [St.reg, hregs']; exact h1920_ui.1) hv_lt hv_ne
                        (by simp [St.reg, hregs']; exact h1920_ui.2) hv_lt hv_ne
                    · omega

/-- **Unconditional 4-Channel Single-Disturbance Immunity Theorem (`spr1_check_arm64`).**
For any invalid input `bs` (`∀ p, Sporos.decode bs = some p → Sporos.checkProg p ≠ true`)
and any post-triplication machine state `TripledState bs s0` (`s0.pc = L_tripled`):
no single disturbance on any of the 4 hardware channels (`fetch`, `reg`, `mem`, `nzcv`)
at any step across `[L_tripled, L_core_end)` (`P1`, `P2`, `P3`, and `[L_vote, L_core_end)`)
on the full `47,636`-byte ELF image `im` and Hamming-1 fetch alternatives `alts` can
produce `.exit 0` (both deterministic `OneFaultRun` and probabilistic `prunWithFaultAt`
for every `ch : Fin 4`). -/
theorem spr1_arm64_bin_no_single_fault_false_accept {bs : List Nat} {s0 : St}
    (ht : TripledState bs s0)
    (hne : ∀ p, Sporos.decode bs = some p → Sporos.checkProg p ≠ true) :
    (∀ n o', OneFaultRun alts im n (.ok s0) o' → o' ≠ .exit 0) ∧
    (∀ (r : Rates) (hr : r.ok) (ch : Fin 4) (fi n : Nat) (y : (Out × Bool) × Rat),
      y ∈ prunWithFaultAt alts r hr ch im (some fi) n (.ok s0) → 0 < y.2 → y.1.1 ≠ .exit 0) := by
  have hinv : checkFlat bs ≠ some true := fun h => by
    obtain ⟨p, hd, hok⟩ := checkFlat_sound h; exact hne p hd hok
  have h4 := tripled_full_oneFault_no_false_accept ht hinv
  exact ⟨h4, fun r hr ch fi n y hy hpos =>
    h4 n y.1.1 (prunWithFaultAt_some_oneFault n fi (.ok s0) hy hpos)⟩

end SporosCheckArm64Bin
