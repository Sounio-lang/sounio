/-
TMRBurst: contiguous `B`-byte cache-line burst fault model on `X86Env.Mem` and
`X86Int.St`, geometric single-burst arena isolation (`burst_hits_at_most_one_arena`),
zero single-burst silent data corruption (`sdc_single_burst_zero`), and
preservation of the quadratic `3 · q_burst²` TMR bound on shared 3-arena memory.

While `X86Int.memFlips` models independent single-bit upsets, physical DRAM/SRAM
and L1/L2/L3 cache-line upsets can corrupt an entire contiguous `B`-byte window
(`B = 64` bytes for an x86-64 cache line) with an arbitrary adversarial payload
`pat : List Nat` (`pat.length ≤ B`) at `[addr, addr + pat.length)`.

Main results:
  1. `burstFault`: instantiates `X86Int.Fault` for arbitrary families of
     contiguous `B`-byte memory bursts `(addr, pat)`.
  2. `burst_hits_at_most_one_arena`: if three TMR arenas `A0, A1, A2` have
     pairwise address separation `≥ B` (`Arena.sep B Ai Aj`), then for EVERY
     address `addr : Nat` and EVERY adversarial payload `pat : List Nat` with
     `pat.length ≤ B`, the burst overlaps at most 1 of the 3 arenas and leaves
     at least 2 arenas byte-for-byte untouched.
  3. `single_burst_vote_arena_exact`, `single_burst_vote_word_exact`, and
     `sdc_single_burst_zero`: any single `≤ B`-byte adversarial burst on
     `X86Env.Mem` preserves the exact majority vote `some g` across `A0, A1, A2`
     (`SDC_single_burst = 0`).
  4. `shared_mem_single_burst_err_zero` and `shared_mem_three_burst_draws_err_le`:
     on a SINGLE shared `X86Env.Mem` holding all three arenas `A0, A1, A2` with
     `pairwiseSep B A0 A1 A2`, a single burst draw has error mass `0`, and three
     sequential burst fault draws have majority-vote error probability at most
     `3 · q_burst²`.
  5. `burst_tri_err`, `burst_trun_err`, `burst_exposure_err`, and
     `burst_voted_step_phys_bound`: per-arena `TMR.Channel` instantiations
     preserving `3 · q_burst²`, `n · 3 · q_burst²`, `3 · ((k + 1) · q_burst)²`,
     and `3 · q_burst² + 9 · r.total` with the 38-byte physical machine-code
     voter (`TMRVoter.votedStep`), plus concrete `p7` rational witness and
     packed-arena (`< 64B`) negative control (`packed_arenas_single_burst_sdc`).
-/
import Bootstrap.PDist
import Bootstrap.TMR
import Bootstrap.TMRExposure
import Bootstrap.X86Env
import Bootstrap.X86Int
import Bootstrap.TMRVoter

namespace TMRBurst

open X86Env X86Int PDist TMR

/-- Standard x86-64 cache-line burst width in bytes (`B = 64`). -/
def cacheLineB : Nat := 64

/-! ### Contiguous `B`-byte adversarial burst operations on `X86Env.Mem` and `X86Int.St` -/

/-- Write byte `v % 256` at address `a` if `a` lies in a writable region of `m`;
otherwise leave `m` unchanged. -/
def burstWrByte (m : Mem) (a v : Nat) : Mem :=
  (m.wr a v).getD m

/-- Apply an adversarial byte payload `pat` across `[addr, addr + pat.length)`. -/
def applyBurstMem (m : Mem) (addr : Nat) : List Nat → Mem
  | [] => m
  | b :: bs => applyBurstMem (burstWrByte m addr b) (addr + 1) bs

/-- A contiguous burst fault specification: starting address `addr` and
adversarial byte payload `pat`. -/
structure Burst where
  addr : Nat
  pat : List Nat
deriving DecidableEq, Repr

/-- A burst is bounded by width `B` when its payload length is at most `B`. -/
def Burst.wf (B : Nat) (b : Burst) : Prop :=
  b.pat.length ≤ B

/-- Apply burst `b` to the memory of state `s : X86Int.St`. -/
def applyBurstSt (s : St) (b : Burst) : St :=
  { s with mem := applyBurstMem s.mem b.addr b.pat }

theorem burstWrByte_shape (m : Mem) (a v : Nat) :
    shape (burstWrByte m a v) = shape m := by
  unfold burstWrByte
  cases h : m.wr a v with
  | none => rfl
  | some m' => exact wr_shape h

theorem applyBurstMem_shape (m : Mem) (addr : Nat) (pat : List Nat) :
    shape (applyBurstMem m addr pat) = shape m := by
  induction pat generalizing m addr with
  | nil => rfl
  | cons b bs ih =>
    simp only [applyBurstMem]
    rw [ih, burstWrByte_shape]

theorem applyBurstMem_wf {m : Mem} (hw : m.wf) (addr : Nat) (pat : List Nat) :
    (applyBurstMem m addr pat).wf :=
  wf_of_shape (applyBurstMem_shape m addr pat) hw

theorem applyBurstMem_eq_of_wrN : ∀ (m : Mem) (addr : Nat) (pat : List Nat) {m' : Mem},
    m.wrN addr pat = some m' → applyBurstMem m addr pat = m'
  | m, _, [], m', h => by
    simp only [Mem.wrN, Option.some.injEq] at h
    exact h
  | m, addr, b :: bs, m', h => by
    simp only [Mem.wrN, bind, Option.bind] at h
    split at h
    · contradiction
    · rename_i m1 h1
      simp only [applyBurstMem, burstWrByte, h1, Option.getD_some]
      exact applyBurstMem_eq_of_wrN m1 (addr + 1) bs h

/-! ### Frame lemmas for `burstWrByte` and `applyBurstMem` -/

private theorem find_bounds : ∀ (m : Mem) (a k : Nat) {j : Nat} {r : Region},
    Mem.find m a k = some (j, r) → r.base ≤ a ∧ a < r.hi
  | [], _, _, _, _, h => by simp [Mem.find] at h
  | s :: rest, a, k, j, r, h => by
    simp only [Mem.find] at h
    split at h
    · rename_i hcond
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨_, rfl⟩ := h
      exact hcond
    · exact find_bounds rest a (k + 1) h

private theorem find_rd_set_ne : ∀ (m : Mem) (i : Nat) (r : Region) (idx val x k : Nat),
    m[i]? = some r → (r.base ≤ x → x - r.base ≠ idx) →
    (do let (_, s) ← Mem.find (m.set i { r with data := r.data.set idx val }) x k
        s.data[x - s.base]?) =
    (do let (_, s) ← Mem.find m x k
        s.data[x - s.base]?)
  | [], _, _, _, _, _, _, h, _ => by simp at h
  | s :: rest, 0, r, idx, val, x, k, h, hne => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    simp only [List.set, Mem.find, Region.hi, List.length_set]
    by_cases hc : s.base ≤ x ∧ x < s.base + s.data.length
    · simp only [hc, and_self, ↓reduceIte, bind, Option.bind]
      have hne' : idx ≠ x - s.base := (hne hc.1).symm
      rw [List.getElem?_set_ne hne']
    · simp only [hc, ↓reduceIte]
  | s :: rest, i + 1, r, idx, val, x, k, h, hne => by
    simp only [List.getElem?_cons_succ] at h
    simp only [List.set, Mem.find]
    by_cases hc : s.base ≤ x ∧ x < s.hi
    · simp only [hc, and_self, ↓reduceIte]
    · simp only [hc, ↓reduceIte]
      exact find_rd_set_ne rest i r idx val x (k + 1) h hne

/-- Writing byte `v` at `a` does not alter any read at `x ≠ a`. -/
theorem burstWrByte_rd_ne (m : Mem) (a v x : Nat) (hne : x ≠ a) :
    (burstWrByte m a v).rd x = m.rd x := by
  unfold burstWrByte
  cases hwr : m.wr a v with
  | none => rfl
  | some m' =>
    simp only [Option.getD_some]
    simp only [Mem.wr, bind, Option.bind] at hwr
    split at hwr
    · contradiction
    · rename_i p hp
      obtain ⟨i, r⟩ := p
      have hmi : m[i]? = some r := by
        have := (find_spec m a 0 hp).2
        simpa using this
      have hab : r.base ≤ a ∧ a < r.hi := find_bounds m a 0 hp
      dsimp only at hwr
      split at hwr
      · simp only [Option.some.injEq, pure] at hwr
        subst hwr
        unfold Mem.rd
        exact find_rd_set_ne m i r (a - r.base) (v % 256) x 0 hmi (by omega)
      · contradiction

/-- Applying payload `pat` at `[addr, addr + pat.length)` preserves `Mem.rd x`
for every address `x` outside `[addr, addr + pat.length)`. -/
theorem applyBurstMem_rd_outside (m : Mem) (addr : Nat) (pat : List Nat) (x : Nat)
    (hx : x < addr ∨ addr + pat.length ≤ x) :
    (applyBurstMem m addr pat).rd x = m.rd x := by
  induction pat generalizing m addr with
  | nil => rfl
  | cons b bs ih =>
    simp only [List.length_cons] at hx
    simp only [applyBurstMem]
    have hne : x ≠ addr := by omega
    have hx' : x < addr + 1 ∨ (addr + 1) + bs.length ≤ x := by omega
    rw [ih (burstWrByte m addr b) (addr + 1) hx', burstWrByte_rd_ne m addr b x hne]

/-- Multi-byte reads `Mem.rdN a n` depend only on `Mem.rd x` for `x ∈ [a, a + n)`. -/
theorem rdN_congr_of_rd_eq {m1 m2 : Mem} : ∀ (a n : Nat),
    (∀ x, a ≤ x → x < a + n → m1.rd x = m2.rd x) →
    m1.rdN a n = m2.rdN a n
  | _, 0, _ => rfl
  | a, n + 1, h => by
    have h0 : m1.rd a = m2.rd a := h a (by omega) (by omega)
    have hrest : m1.rdN (a + 1) n = m2.rdN (a + 1) n :=
      rdN_congr_of_rd_eq (a + 1) n (fun x h1 h2 => h x (by omega) (by omega))
    simp only [Mem.rdN, h0, hrest]

/-- If `[addr, addr + pat.length)` is disjoint from `[a, a + n)`, then
`applyBurstMem m addr pat` preserves `Mem.rdN a n`. -/
theorem applyBurstMem_rdN_disjoint (m : Mem) (addr : Nat) (pat : List Nat) (a n : Nat)
    (hdisj : addr + pat.length ≤ a ∨ a + n ≤ addr) :
    (applyBurstMem m addr pat).rdN a n = m.rdN a n := by
  refine rdN_congr_of_rd_eq a n ?_
  intro x hlo hhi
  exact applyBurstMem_rd_outside m addr pat x (by omega)

/-- If `[addr, addr + pat.length)` is disjoint from `[a, a + 8)`, then
`applyBurstMem m addr pat` preserves `X86Int.rd64 m a`. -/
theorem applyBurstMem_rd64_disjoint (m : Mem) (addr : Nat) (pat : List Nat) (a : Nat)
    (hdisj : addr + pat.length ≤ a ∨ a + 8 ≤ addr) :
    rd64 (applyBurstMem m addr pat) a = rd64 m a := by
  simp only [rd64, applyBurstMem_rdN_disjoint m addr pat a 8 hdisj]

/-! ### Instantiation of `X86Int.Fault` for `B`-byte burst faults -/

/-- Candidate states produced by applying each burst in `bursts s` to `s`. -/
def burstAlts (bursts : St → List Burst) (s : St) : List St :=
  (bursts s).map (applyBurstSt s)

/-- Contiguous burst fault model on `X86Int.St`, instantiating `X86Int.Fault`
for any state-indexed list of bursts `bursts : St → List Burst` at rate `p`. -/
def burstFault (bursts : St → List Burst) (p : Rat) (hp : Rate p) : Fault :=
  Fault.uniform (burstAlts bursts) p hp

/-- Single deterministic burst `(addr, pat)` applied with fault probability `p`. -/
def singleBurstFault (b : Burst) (p : Rat) (hp : Rate p) : Fault :=
  burstFault (fun _ => [b]) p hp

/-- Combined retired-state fault model: standard 3-site single-bit upsets
(`Rates.state`) followed by contiguous `B`-byte memory bursts (`burstFault`). -/
def stateWithBurst (r : Rates) (hr : r.ok)
    (bursts : St → List Burst) (qBurst : Rat) (hq : Rate qBurst) : Fault :=
  (r.state hr).seq (burstFault bursts qBurst hq)

/-- All bursts generated by placing any payload `pat ∈ pats` at any byte offset
of any mapped region in `s.mem`. -/
def memBursts (pats : List (List Nat)) (s : St) : List Burst :=
  s.mem.flatMap fun r =>
    (List.range r.data.length).flatMap fun j =>
      pats.map fun pat => ⟨r.base + j, pat⟩

theorem memBursts_wf {B : Nat} {pats : List (List Nat)}
    (hpats : ∀ pat ∈ pats, pat.length ≤ B) (s : St) :
    ∀ b ∈ memBursts pats s, b.pat.length ≤ B := by
  intro b hb
  simp only [memBursts, List.mem_flatMap, List.mem_range, List.mem_map] at hb
  obtain ⟨r, _, j, _, pat, hpat, rfl⟩ := hb
  exact hpats pat hpat

theorem mem_burstFault_hit (bursts : St → List Burst) (p : Rat) (hp : Rate p)
    (s : St) {y : (St × Bool) × Rat} (hy : y ∈ (burstFault bursts p hp).hit s) :
    (y.1.2 = false ∧ y.1.1 = s) ∨
    (y.1.2 = true ∧ ∃ b ∈ bursts s, y.1.1 = applyBurstSt s b) := by
  dsimp only [burstFault, Fault.uniform, burstAlts] at hy
  cases hbs : bursts s with
  | nil =>
    rw [hbs] at hy
    simp only [List.map_nil, unif, pureF, List.mem_cons, List.not_mem_nil, or_false] at hy
    subst hy
    exact Or.inl ⟨rfl, rfl⟩
  | cons b0 rest =>
    rw [hbs] at hy
    simp only [List.map_cons, unif, List.mem_cons, List.mem_map] at hy
    rcases hy with rfl | htail
    · exact Or.inl ⟨rfl, rfl⟩
    · rcases htail with ⟨_, ⟨rfl, rfl⟩⟩ | ⟨z, ⟨b, hb, rfl⟩, ⟨rfl, rfl⟩⟩
      · exact Or.inr ⟨rfl, b0, List.mem_cons_self .., rfl⟩
      · exact Or.inr ⟨rfl, b, List.mem_cons_of_mem _ hb, rfl⟩

/-! ### Geometric Single-Burst Isolation across 3 TMR Arenas -/

/-- A contiguous memory arena `[base, base + size)`. -/
structure Arena where
  base : Nat
  size : Nat
deriving DecidableEq, Repr

def Arena.hi (A : Arena) : Nat := A.base + A.size

/-- Two arenas `A1` and `A2` have address separation at least `B` bytes when
there is a gap of at least `B` bytes between the end of one and the start of the
other. -/
def Arena.sep (B : Nat) (A1 A2 : Arena) : Prop :=
  A1.hi + B ≤ A2.base ∨ A2.hi + B ≤ A1.base

/-- Pairwise address separation `≥ B` across three TMR arenas `A0, A1, A2`. -/
def pairwiseSep (B : Nat) (A0 A1 A2 : Arena) : Prop :=
  Arena.sep B A0 A1 ∧ Arena.sep B A0 A2 ∧ Arena.sep B A1 A2

/-- Burst `b` is address-disjoint from arena `A`. -/
def burstDisjoint (b : Burst) (A : Arena) : Prop :=
  b.addr + b.pat.length ≤ A.base ∨ A.hi ≤ b.addr

instance instDecidableBurstDisjoint (b : Burst) (A : Arena) : Decidable (burstDisjoint b A) := by
  unfold burstDisjoint; exact inferInstance

/-- Boolean indicator of whether `[b.addr, b.addr + b.pat.length)` overlaps `A`. -/
def burstHits (b : Burst) (A : Arena) : Bool :=
  decide (b.addr < A.hi ∧ A.base < b.addr + b.pat.length)

/-- Number of arenas among `(A0, A1, A2)` overlapped by burst `b`. -/
def burstHitCount (b : Burst) (A0 A1 A2 : Arena) : Nat :=
  (if burstHits b A0 then 1 else 0) +
  (if burstHits b A1 then 1 else 0) +
  (if burstHits b A2 then 1 else 0)

theorem burstHits_false_iff (b : Burst) (A : Arena) :
    burstHits b A = false ↔ burstDisjoint b A := by
  unfold burstHits burstDisjoint
  rw [decide_eq_false_iff_not]
  omega

/-- Key geometric separation lemma: if `A1` and `A2` are separated by at least
`B` bytes, no single burst of length `≤ B` can overlap both `A1` and `A2`. -/
theorem burst_cannot_hit_two {B : Nat} {A1 A2 : Arena} (hsep : Arena.sep B A1 A2)
    (b : Burst) (hb : b.pat.length ≤ B) :
    ¬ (burstHits b A1 = true ∧ burstHits b A2 = true) := by
  unfold burstHits Arena.sep at *
  simp only [decide_eq_true_eq]
  omega

/-- **Geometric Single-Burst Isolation (`burst_hits_at_most_one_arena`)**:
If three TMR arenas `A0, A1, A2` have pairwise separation `≥ B`, then for EVERY
burst `b = ⟨addr, pat⟩` with `pat.length ≤ B` (universally quantified over all
addresses and all adversarial payloads `pat : List Nat` of length `≤ B`):
  1. `b` hits at most 1 of the 3 arenas (`burstHitCount b A0 A1 A2 ≤ 1`);
  2. At least 2 of the 3 arenas are completely disjoint from `b`. -/
theorem burst_hits_at_most_one_arena {B : Nat} {A0 A1 A2 : Arena}
    (hsep : pairwiseSep B A0 A1 A2) (b : Burst) (hb : b.pat.length ≤ B) :
    burstHitCount b A0 A1 A2 ≤ 1 ∧
    ((burstDisjoint b A0 ∧ burstDisjoint b A1) ∨
     (burstDisjoint b A0 ∧ burstDisjoint b A2) ∨
     (burstDisjoint b A1 ∧ burstDisjoint b A2)) := by
  obtain ⟨h01, h02, h12⟩ := hsep
  have hn01 := burst_cannot_hit_two h01 b hb
  have hn02 := burst_cannot_hit_two h02 b hb
  have hn12 := burst_cannot_hit_two h12 b hb
  unfold burstHitCount
  cases h0 : burstHits b A0 <;> cases h1 : burstHits b A1 <;> cases h2 : burstHits b A2
  · refine ⟨by decide, Or.inl ⟨(burstHits_false_iff b A0).1 h0, (burstHits_false_iff b A1).1 h1⟩⟩
  · refine ⟨by decide, Or.inl ⟨(burstHits_false_iff b A0).1 h0, (burstHits_false_iff b A1).1 h1⟩⟩
  · refine ⟨by decide, Or.inr (Or.inl ⟨(burstHits_false_iff b A0).1 h0, (burstHits_false_iff b A2).1 h2⟩)⟩
  · exact False.elim (hn12 ⟨h1, h2⟩)
  · refine ⟨by decide, Or.inr (Or.inr ⟨(burstHits_false_iff b A1).1 h1, (burstHits_false_iff b A2).1 h2⟩)⟩
  · exact False.elim (hn02 ⟨h0, h2⟩)
  · exact False.elim (hn01 ⟨h0, h1⟩)
  · exact False.elim (hn01 ⟨h0, h1⟩)

/-- If burst `b` is disjoint from arena `A`, applying `b` leaves the entire
byte slice `m.rdN A.base A.size` untouched. -/
theorem applyBurstMem_arena_eq_of_disjoint (m : Mem) (b : Burst) (A : Arena)
    (hdisj : burstDisjoint b A) :
    (applyBurstMem m b.addr b.pat).rdN A.base A.size = m.rdN A.base A.size :=
  applyBurstMem_rdN_disjoint m b.addr b.pat A.base A.size hdisj

/-- Under pairwise separation `≥ B`, any single `≤ B`-byte adversarial burst
leaves at least 2 of the 3 arena byte slices untouched. -/
theorem burst_preserves_two_arenas {B : Nat} {A0 A1 A2 : Arena}
    (hsep : pairwiseSep B A0 A1 A2) (m : Mem) (b : Burst) (hb : b.pat.length ≤ B) :
    let m' := applyBurstMem m b.addr b.pat
    (m'.rdN A0.base A0.size = m.rdN A0.base A0.size ∧
     m'.rdN A1.base A1.size = m.rdN A1.base A1.size) ∨
    (m'.rdN A0.base A0.size = m.rdN A0.base A0.size ∧
     m'.rdN A2.base A2.size = m.rdN A2.base A2.size) ∨
    (m'.rdN A1.base A1.size = m.rdN A1.base A1.size ∧
     m'.rdN A2.base A2.size = m.rdN A2.base A2.size) := by
  dsimp only
  rcases (burst_hits_at_most_one_arena hsep b hb).2 with ⟨d0, d1⟩ | ⟨d0, d2⟩ | ⟨d1, d2⟩
  · exact Or.inl ⟨applyBurstMem_arena_eq_of_disjoint m b A0 d0,
                  applyBurstMem_arena_eq_of_disjoint m b A1 d1⟩
  · exact Or.inr (Or.inl ⟨applyBurstMem_arena_eq_of_disjoint m b A0 d0,
                          applyBurstMem_arena_eq_of_disjoint m b A2 d2⟩)
  · exact Or.inr (Or.inr ⟨applyBurstMem_arena_eq_of_disjoint m b A1 d1,
                          applyBurstMem_arena_eq_of_disjoint m b A2 d2⟩)

/-- Full-arena majority vote after any single `≤ B`-byte adversarial burst
recovers the exact clean arena contents `some g`. -/
theorem single_burst_vote_arena_exact {B : Nat} {A0 A1 A2 : Arena}
    (hsep : pairwiseSep B A0 A1 A2) (m : Mem) {g : Option (List Nat)}
    (h0 : m.rdN A0.base A0.size = g)
    (h1 : m.rdN A1.base A1.size = g)
    (h2 : m.rdN A2.base A2.size = g)
    (b : Burst) (hb : b.pat.length ≤ B) :
    let m' := applyBurstMem m b.addr b.pat
    TMR.vote (m'.rdN A0.base A0.size) (m'.rdN A1.base A1.size) (m'.rdN A2.base A2.size) = some g := by
  dsimp only
  rcases burst_preserves_two_arenas hsep m b hb with ⟨e0, e1⟩ | ⟨e0, e2⟩ | ⟨e1, e2⟩
  · exact TMR.vote_of_two (Or.inl ⟨e0.trans h0, e1.trans h1⟩)
  · exact TMR.vote_of_two (Or.inr (Or.inl ⟨e0.trans h0, e2.trans h2⟩))
  · exact TMR.vote_of_two (Or.inr (Or.inr ⟨e1.trans h1, e2.trans h2⟩))

/-- Read a 64-bit word at relative byte offset `off` inside arena `A` (defaulting
to `0` if unmapped). -/
def rdArenaWord (m : Mem) (A : Arena) (off : Nat) : Nat :=
  (rd64 m (A.base + off)).getD 0

theorem applyBurstMem_arenaWord_eq_of_disjoint (m : Mem) (b : Burst) (A : Arena)
    {off : Nat} (hoff : off + 8 ≤ A.size) (hdisj : burstDisjoint b A) :
    rdArenaWord (applyBurstMem m b.addr b.pat) A off = rdArenaWord m A off := by
  unfold rdArenaWord
  have hdisj8 : b.addr + b.pat.length ≤ A.base + off ∨ (A.base + off) + 8 ≤ b.addr := by
    unfold burstDisjoint Arena.hi at hdisj
    omega
  rw [applyBurstMem_rd64_disjoint m b.addr b.pat (A.base + off) hdisj8]

/-- Under pairwise separation `≥ B`, any single `≤ B`-byte adversarial burst
leaves at least 2 of the 3 replica words at offset `off` untouched. -/
theorem burst_preserves_two_words {B : Nat} {A0 A1 A2 : Arena}
    (hsep : pairwiseSep B A0 A1 A2)
    {off : Nat} (ho0 : off + 8 ≤ A0.size) (ho1 : off + 8 ≤ A1.size) (ho2 : off + 8 ≤ A2.size)
    (m : Mem) (b : Burst) (hb : b.pat.length ≤ B) :
    let m' := applyBurstMem m b.addr b.pat
    (rdArenaWord m' A0 off = rdArenaWord m A0 off ∧
     rdArenaWord m' A1 off = rdArenaWord m A1 off) ∨
    (rdArenaWord m' A0 off = rdArenaWord m A0 off ∧
     rdArenaWord m' A2 off = rdArenaWord m A2 off) ∨
    (rdArenaWord m' A1 off = rdArenaWord m A1 off ∧
     rdArenaWord m' A2 off = rdArenaWord m A2 off) := by
  dsimp only
  rcases (burst_hits_at_most_one_arena hsep b hb).2 with ⟨d0, d1⟩ | ⟨d0, d2⟩ | ⟨d1, d2⟩
  · exact Or.inl ⟨applyBurstMem_arenaWord_eq_of_disjoint m b A0 ho0 d0,
                  applyBurstMem_arenaWord_eq_of_disjoint m b A1 ho1 d1⟩
  · exact Or.inr (Or.inl ⟨applyBurstMem_arenaWord_eq_of_disjoint m b A0 ho0 d0,
                          applyBurstMem_arenaWord_eq_of_disjoint m b A2 ho2 d2⟩)
  · exact Or.inr (Or.inr ⟨applyBurstMem_arenaWord_eq_of_disjoint m b A1 ho1 d1,
                          applyBurstMem_arenaWord_eq_of_disjoint m b A2 ho2 d2⟩)

/-- Per-word majority vote across `A0, A1, A2` after ANY single `≤ B`-byte
adversarial burst `(addr, pat)` recovers the exact clean word `some g`. -/
theorem single_burst_vote_word_exact {B : Nat} {A0 A1 A2 : Arena}
    (hsep : pairwiseSep B A0 A1 A2)
    {off : Nat} (ho0 : off + 8 ≤ A0.size) (ho1 : off + 8 ≤ A1.size) (ho2 : off + 8 ≤ A2.size)
    (m : Mem) {g : Nat}
    (h0 : rdArenaWord m A0 off = g)
    (h1 : rdArenaWord m A1 off = g)
    (h2 : rdArenaWord m A2 off = g)
    (b : Burst) (hb : b.pat.length ≤ B) :
    let m' := applyBurstMem m b.addr b.pat
    TMR.vote (rdArenaWord m' A0 off) (rdArenaWord m' A1 off) (rdArenaWord m' A2 off) = some g := by
  dsimp only
  rcases burst_preserves_two_words hsep ho0 ho1 ho2 m b hb with ⟨e0, e1⟩ | ⟨e0, e2⟩ | ⟨e1, e2⟩
  · exact TMR.vote_of_two (Or.inl ⟨e0.trans h0, e1.trans h1⟩)
  · exact TMR.vote_of_two (Or.inr (Or.inl ⟨e0.trans h0, e2.trans h2⟩))
  · exact TMR.vote_of_two (Or.inr (Or.inr ⟨e1.trans h1, e2.trans h2⟩))

/-- **Zero Single-Burst Error / Zero Single-Burst SDC (`sdc_single_burst_zero`)**:
Quantifying universally over every address `addr : Nat` and every adversarial
payload `pat : List Nat` with `pat.length ≤ B`, the majority-vote error indicator
after applying `(addr, pat)` is identically `0`. -/
theorem sdc_single_burst_zero {B : Nat} {A0 A1 A2 : Arena}
    (hsep : pairwiseSep B A0 A1 A2)
    {off : Nat} (ho0 : off + 8 ≤ A0.size) (ho1 : off + 8 ≤ A1.size) (ho2 : off + 8 ≤ A2.size)
    (m : Mem) {g : Nat}
    (h0 : rdArenaWord m A0 off = g)
    (h1 : rdArenaWord m A1 off = g)
    (h2 : rdArenaWord m A2 off = g)
    (addr : Nat) (pat : List Nat) (hpat : pat.length ≤ B) :
    let m' := applyBurstMem m addr pat
    TMR.ι (decide (TMR.vote (rdArenaWord m' A0 off)
                            (rdArenaWord m' A1 off)
                            (rdArenaWord m' A2 off) ≠ some g)) = 0 := by
  dsimp only
  have hv := single_burst_vote_word_exact hsep ho0 ho1 ho2 m h0 h1 h2 ⟨addr, pat⟩ hpat
  dsimp only at hv
  rw [hv]
  simp [TMR.ι]

/-! ### Shared-Memory 1-Draw Zero Error and 3-Draw Quadratic `3 · q_burst²` Bound -/

/-- One burst-fault draw on shared 3-arena state `s`, followed by majority vote
of the three replica words at offset `off`. -/
def sharedVoteAfter1 (bursts : St → List Burst) (qBurst : Rat) (hq : Rate qBurst)
    (A0 A1 A2 : Arena) (off g : Nat) (s : St) : PDF (Option Nat) :=
  TMR.bindW ((burstFault bursts qBurst hq).hit s) fun x =>
    TMR.mark g (TMR.vote (rdArenaWord x.1.mem A0 off)
                         (rdArenaWord x.1.mem A1 off)
                         (rdArenaWord x.1.mem A2 off))

/-- Any outcome in `(burstFault bursts qBurst hq).hit s` (whether clean or faulty)
votes to `some g` across `A0, A1, A2` when `∀ b ∈ bursts s, b.pat.length ≤ B`. -/
theorem hit_burst_vote_word_exact {B : Nat} {A0 A1 A2 : Arena}
    (hsep : pairwiseSep B A0 A1 A2)
    {off : Nat} (ho0 : off + 8 ≤ A0.size) (ho1 : off + 8 ≤ A1.size) (ho2 : off + 8 ≤ A2.size)
    (bursts : St → List Burst) (hB : ∀ s, ∀ b ∈ bursts s, b.pat.length ≤ B)
    (qBurst : Rat) (hq : Rate qBurst) (s : St) {g : Nat}
    (h0 : rdArenaWord s.mem A0 off = g)
    (h1 : rdArenaWord s.mem A1 off = g)
    (h2 : rdArenaWord s.mem A2 off = g)
    {y : (St × Bool) × Rat} (hy : y ∈ (burstFault bursts qBurst hq).hit s) :
    TMR.vote (rdArenaWord y.1.1.mem A0 off)
             (rdArenaWord y.1.1.mem A1 off)
             (rdArenaWord y.1.1.mem A2 off) = some g := by
  rcases mem_burstFault_hit bursts qBurst hq s hy with ⟨_, hseq⟩ | ⟨_, b, hb, hseq⟩
  · rw [hseq, h0, h1, h2]
    exact TMR.vote_of_two (Or.inl ⟨rfl, rfl⟩)
  · rw [hseq]
    exact single_burst_vote_word_exact hsep ho0 ho1 ho2 s.mem h0 h1 h2 b (hB s b hb)

/-- After any single burst-fault draw on shared 3-arena memory, the voted error
probability `fm (sharedVoteAfter1 ...)` is identically `0`. -/
theorem shared_mem_single_burst_err_zero {B : Nat} {A0 A1 A2 : Arena}
    (hsep : pairwiseSep B A0 A1 A2)
    {off : Nat} (ho0 : off + 8 ≤ A0.size) (ho1 : off + 8 ≤ A1.size) (ho2 : off + 8 ≤ A2.size)
    (bursts : St → List Burst) (hB : ∀ s, ∀ b ∈ bursts s, b.pat.length ≤ B)
    (qBurst : Rat) (hq : Rate qBurst) (s : St) {g : Nat}
    (h0 : rdArenaWord s.mem A0 off = g)
    (h1 : rdArenaWord s.mem A1 off = g)
    (h2 : rdArenaWord s.mem A2 off = g) :
    fm (sharedVoteAfter1 bursts qBurst hq A0 A1 A2 off g s) = 0 := by
  have hm : ∀ v : Option Nat, fm (TMR.mark g v) = TMR.ι (decide (v ≠ some g)) := fun v => by
    show (if decide (v ≠ some g) = true then (1 : Rat) else 0) + 0 = _
    rw [Rat.add_zero]; rfl
  simp only [sharedVoteAfter1, TMR.fm_bindW, hm]
  have hD := (burstFault bursts qBurst hq).ok s
  have hle : TMR.sumW ((burstFault bursts qBurst hq).hit s)
      (fun x => TMR.ι (decide (TMR.vote (rdArenaWord x.1.mem A0 off)
                                        (rdArenaWord x.1.mem A1 off)
                                        (rdArenaWord x.1.mem A2 off) ≠ some g))) ≤
      TMR.sumW ((burstFault bursts qBurst hq).hit s) (fun _ => (0 : Rat)) := by
    refine TMR.sumW_le hD.1 fun x hx => ?_
    rw [hit_burst_vote_word_exact hsep ho0 ho1 ho2 bursts hB qBurst hq s h0 h1 h2 hx]
    simp [TMR.ι]
  rw [TMR.sumW_const, Rat.mul_zero] at hle
  have hge : 0 ≤ fm (sharedVoteAfter1 bursts qBurst hq A0 A1 A2 off g s) := by
    refine fm_nonneg (TMR.bindW_nonneg hD.1 fun a y hy => ?_)
    simp only [TMR.mark, List.mem_cons, List.not_mem_nil, or_false] at hy
    subst hy; show (0 : Rat) ≤ 1; decide
  simp only [sharedVoteAfter1, TMR.fm_bindW, hm] at hge
  exact Rat.le_antisymm hle hge

/-- Three sequential burst-fault draws on a SINGLE shared 3-arena state `s`,
followed by majority vote of the three replica words at offset `off`. -/
def sharedVoteAfter3 (bursts : St → List Burst) (qBurst : Rat) (hq : Rate qBurst)
    (A0 A1 A2 : Arena) (off g : Nat) (s : St) : PDF (Option Nat) :=
  let F := burstFault bursts qBurst hq
  TMR.bindW (F.hit s) fun x1 =>
    TMR.bindW (F.hit x1.1) fun x2 =>
      TMR.bindW (F.hit x2.1) fun x3 =>
        TMR.mark g (TMR.vote (rdArenaWord x3.1.mem A0 off)
                             (rdArenaWord x3.1.mem A1 off)
                             (rdArenaWord x3.1.mem A2 off))

private theorem sumW_affine_tag_le {α : Type} {D : PDF α} (hD : D.ok) {q : Rat}
    (hq : fm D ≤ q) (A : Rat) {B : Rat} (hB : 0 ≤ B) :
    TMR.sumW D (fun x => A + TMR.ι x.2 * B) ≤ A + q * B := by
  have hs : ffm D + fm D = 1 := by rw [Rat.add_comm, ← mass_split, hD.2]
  rw [TMR.sumW_tag D (fun t => A + TMR.ι t * B)]
  simp only [TMR.ι, Bool.false_eq_true, ↓reduceIte, Rat.zero_mul, Rat.add_zero, Rat.one_mul]
  have hid : ffm D * A + fm D * (A + B) = (ffm D + fm D) * A + fm D * B := by grind
  rw [hid, hs, Rat.one_mul]
  exact add_le_add' Rat.le_refl (Rat.mul_le_mul_of_nonneg_right hq hB)

private theorem p3_decomp3 (t1 t2 t3 : Bool) :
    TMR.P3 t1 t2 t3 = TMR.ι (t1 && t2) + TMR.ι t3 * (TMR.ι t1 + TMR.ι t2) := by
  cases t1 <;> cases t2 <;> cases t3 <;> simp [TMR.P3, TMR.ι] <;> decide +kernel

private theorem p3_decomp2 (t1 t2 : Bool) (q : Rat) :
    TMR.ι (t1 && t2) + q * (TMR.ι t1 + TMR.ι t2) =
      q * TMR.ι t1 + TMR.ι t2 * (TMR.ι t1 + q) := by
  cases t1 <;> cases t2 <;> simp [TMR.ι] <;> grind

private theorem p3_decomp1 (t1 : Bool) (q : Rat) :
    q * TMR.ι t1 + q * (TMR.ι t1 + q) = q * q + TMR.ι t1 * (2 * q) := by
  cases t1 <;> simp [TMR.ι] <;> grind

/-- **Shared-Memory Three-Burst Quadratic Bound (`shared_mem_three_burst_draws_err_le`)**:
On a single shared `X86Int.St` whose memory `s.mem` contains three arenas
`A0, A1, A2` with pairwise separation `≥ B`, three sequential burst-fault
opportunities (each applying any `≤ B`-byte adversarial burst from `bursts` with
probability `≤ qBurst`) preserve the exact quadratic `3 · qBurst²` TMR bound. -/
theorem shared_mem_three_burst_draws_err_le {B : Nat} {A0 A1 A2 : Arena}
    (hsep : pairwiseSep B A0 A1 A2)
    {off : Nat} (ho0 : off + 8 ≤ A0.size) (ho1 : off + 8 ≤ A1.size) (ho2 : off + 8 ≤ A2.size)
    (bursts : St → List Burst) (hB : ∀ s, ∀ b ∈ bursts s, b.pat.length ≤ B)
    (qBurst : Rat) (hq : Rate qBurst) (s : St) {g : Nat}
    (h0 : rdArenaWord s.mem A0 off = g)
    (h1 : rdArenaWord s.mem A1 off = g)
    (h2 : rdArenaWord s.mem A2 off = g) :
    (sharedVoteAfter3 bursts qBurst hq A0 A1 A2 off g s).ok ∧
    fm (sharedVoteAfter3 bursts qBurst hq A0 A1 A2 off g s) ≤ 3 * (qBurst * qBurst) := by
  let F := burstFault bursts qBurst hq
  refine ⟨?_, ?_⟩
  · refine ⟨?_, ?_⟩
    · refine TMR.bindW_nonneg (F.ok s).1 fun x1 =>
        TMR.bindW_nonneg (F.ok x1.1).1 fun x2 =>
          TMR.bindW_nonneg (F.ok x2.1).1 fun x3 y hy => ?_
      simp only [TMR.mark, List.mem_cons, List.not_mem_nil, or_false] at hy
      subst hy; show (0 : Rat) ≤ 1; decide
    · have h1 : ∀ v : Option Nat, mass (TMR.mark g v) = 1 := fun v => by
        show (1 : Rat) + 0 = 1; rw [Rat.add_zero]
      have h3 : ∀ s2 : St, mass (TMR.bindW (F.hit s2) fun x3 =>
          TMR.mark g (TMR.vote (rdArenaWord x3.1.mem A0 off)
                               (rdArenaWord x3.1.mem A1 off)
                               (rdArenaWord x3.1.mem A2 off))) = 1 := by
        intro s2
        rw [TMR.mass_bindW, TMR.sumW_congr _ (fun _ => h1 _), TMR.sumW_const, (F.ok s2).2, Rat.one_mul]
      have h2' : ∀ s1 : St, mass (TMR.bindW (F.hit s1) fun x2 =>
          TMR.bindW (F.hit x2.1) fun x3 =>
            TMR.mark g (TMR.vote (rdArenaWord x3.1.mem A0 off)
                                 (rdArenaWord x3.1.mem A1 off)
                                 (rdArenaWord x3.1.mem A2 off))) = 1 := by
        intro s1
        rw [TMR.mass_bindW, TMR.sumW_congr _ (fun x2 => h3 x2.1), TMR.sumW_const, (F.ok s1).2, Rat.one_mul]
      unfold sharedVoteAfter3
      rw [TMR.mass_bindW, TMR.sumW_congr _ (fun x1 => h2' x1.1), TMR.sumW_const, (F.ok s).2, Rat.one_mul]
  · have hm : ∀ v : Option Nat, fm (TMR.mark g v) = TMR.ι (decide (v ≠ some g)) := fun v => by
      show (if decide (v ≠ some g) = true then (1 : Rat) else 0) + 0 = _
      rw [Rat.add_zero]; rfl
    simp only [sharedVoteAfter3, TMR.fm_bindW, hm]
    have hpt : ∀ x1 ∈ F.hit s, ∀ x2 ∈ F.hit x1.1.1, ∀ x3 ∈ F.hit x2.1.1,
        TMR.ι (decide (TMR.vote (rdArenaWord x3.1.1.mem A0 off)
                                (rdArenaWord x3.1.1.mem A1 off)
                                (rdArenaWord x3.1.1.mem A2 off) ≠ some g)) ≤
          TMR.P3 x1.1.2 x2.1.2 x3.1.2 := by
      intro x1 hx1 x2 hx2 x3 hx3
      by_cases hv : TMR.vote (rdArenaWord x3.1.1.mem A0 off)
                             (rdArenaWord x3.1.1.mem A1 off)
                             (rdArenaWord x3.1.1.mem A2 off) = some g
      · rw [hv]
        have : TMR.ι (decide (some g ≠ some g)) = 0 := by simp [TMR.ι]
        rw [this]; exact TMR.P3_nonneg _ _ _
      · refine Rat.le_trans (TMR.ι_le_one _) ?_
        rcases TMR.two_or x1.1.2 x2.1.2 x3.1.2 with ⟨ht1, ht2⟩ | ⟨ht1, ht3⟩ | ⟨ht2, ht3⟩ | hp3
        · have hs1 : x1.1.1 = s := TMR.clean_val (F.clean s) hx1 ht1
          have hs2 : x2.1.1 = x1.1.1 := TMR.clean_val (F.clean x1.1.1) hx2 ht2
          rw [hs2, hs1] at hx3
          exact absurd (hit_burst_vote_word_exact hsep ho0 ho1 ho2 bursts hB qBurst hq s h0 h1 h2 hx3) hv
        · have hs1 : x1.1.1 = s := TMR.clean_val (F.clean s) hx1 ht1
          have hs3 : x3.1.1 = x2.1.1 := TMR.clean_val (F.clean x2.1.1) hx3 ht3
          rw [hs1] at hx2
          rw [hs3] at hv
          exact absurd (hit_burst_vote_word_exact hsep ho0 ho1 ho2 bursts hB qBurst hq s h0 h1 h2 hx2) hv
        · have hs2 : x2.1.1 = x1.1.1 := TMR.clean_val (F.clean x1.1.1) hx2 ht2
          have hs3 : x3.1.1 = x2.1.1 := TMR.clean_val (F.clean x2.1.1) hx3 ht3
          rw [hs3, hs2] at hv
          exact absurd (hit_burst_vote_word_exact hsep ho0 ho1 ho2 bursts hB qBurst hq s h0 h1 h2 hx1) hv
        · exact hp3
    have hstep1 :
        TMR.sumW (F.hit s) (fun x1 =>
          TMR.sumW (F.hit x1.1) (fun x2 =>
            TMR.sumW (F.hit x2.1) (fun x3 =>
              TMR.ι (decide (TMR.vote (rdArenaWord x3.1.mem A0 off)
                                      (rdArenaWord x3.1.mem A1 off)
                                      (rdArenaWord x3.1.mem A2 off) ≠ some g))))) ≤
        TMR.sumW (F.hit s) (fun x1 =>
          TMR.sumW (F.hit x1.1) (fun x2 =>
            TMR.sumW (F.hit x2.1) (fun x3 => TMR.P3 x1.2 x2.2 x3.2))) := by
      refine TMR.sumW_le (F.ok s).1 fun x1 hx1 =>
        TMR.sumW_le (F.ok x1.1.1).1 fun x2 hx2 =>
          TMR.sumW_le (F.ok x2.1.1).1 fun x3 hx3 => hpt x1 hx1 x2 hx2 x3 hx3
    refine Rat.le_trans hstep1 ?_
    have hL3 : ∀ (s2 : St) (t1 t2 : Bool),
        TMR.sumW (F.hit s2) (fun x3 => TMR.P3 t1 t2 x3.2) ≤
          TMR.ι (t1 && t2) + qBurst * (TMR.ι t1 + TMR.ι t2) := by
      intro s2 t1 t2
      rw [TMR.sumW_congr (F.hit s2) (fun x3 => p3_decomp3 t1 t2 x3.2)]
      exact sumW_affine_tag_le (F.ok s2) (F.le s2) (TMR.ι (t1 && t2))
        (nonneg_add (TMR.ι_nonneg t1) (TMR.ι_nonneg t2))
    have hL2 : ∀ (s1 : St) (t1 : Bool),
        TMR.sumW (F.hit s1) (fun x2 => TMR.sumW (F.hit x2.1) (fun x3 => TMR.P3 t1 x2.2 x3.2)) ≤
          qBurst * TMR.ι t1 + qBurst * (TMR.ι t1 + qBurst) := by
      intro s1 t1
      refine Rat.le_trans
        (TMR.sumW_le (F.ok s1).1
          (ψ := fun x2 => TMR.ι (t1 && x2.2) + qBurst * (TMR.ι t1 + TMR.ι x2.2))
          fun x2 _ => hL3 x2.1.1 t1 x2.1.2) ?_
      rw [TMR.sumW_congr (F.hit s1) (fun x2 => p3_decomp2 t1 x2.2 qBurst)]
      exact sumW_affine_tag_le (F.ok s1) (F.le s1) (qBurst * TMR.ι t1)
        (nonneg_add (TMR.ι_nonneg t1) hq.1)
    refine Rat.le_trans
      (TMR.sumW_le (F.ok s).1
        (ψ := fun x1 => qBurst * TMR.ι x1.2 + qBurst * (TMR.ι x1.2 + qBurst))
        fun x1 _ => hL2 x1.1.1 x1.1.2) ?_
    rw [TMR.sumW_congr (F.hit s) (fun x1 => p3_decomp1 x1.2 qBurst)]
    refine Rat.le_trans
      (sumW_affine_tag_le (F.ok s) (F.le s) (qBurst * qBurst) (Rat.mul_nonneg (by decide) hq.1)) ?_
    exact TMR.le_of_eq' (by show qBurst * qBurst + qBurst * (2 * qBurst) = 3 * (qBurst * qBurst); grind)

/-! ### Per-Arena `TMR.Channel` Instantiations under Burst Faults -/

/-- State-level projection channel for `X86Int.St` under `burstFault` on a
mapped memory word at `A.base + off`. -/
def burstMemWordProjChannel (bursts : St → List Burst) (qBurst : Rat) (hq : Rate qBurst)
    (s0 : St) (A : Arena) (off : Nat) (fn : Nat → Nat) : TMR.Channel Nat where
  f v :=
    let s := { s0 with mem := (wr64 s0.mem (A.base + off) (fn v)).getD s0.mem }
    (rd64 s.mem (A.base + off)).getD (fn v)
  step v :=
    let s := { s0 with mem := (wr64 s0.mem (A.base + off) (fn v)).getD s0.mem }
    relabel (fun s' => (rd64 s'.mem (A.base + off)).getD (fn v))
      ((burstFault bursts qBurst hq).hit s)
  q := qBurst
  q0 := hq.1
  ok _ := relabel_ok _ ((burstFault bursts qBurst hq).ok _)
  clean _ := by rw [faultFree_relabel, (burstFault bursts qBurst hq).clean]; rfl
  le _ := by rw [fm_relabel]; exact (burstFault bursts qBurst hq).le _

/-- Compute the 64-bit word at `wordAddr` after applying burst `b` to an 8-byte
writable region holding `(fn v) % W`. -/
def corruptWordByBurst (wordAddr val : Nat) (b : Burst) : Nat :=
  let m0 : Mem := [⟨wordAddr, bytes8 val, true⟩]
  (rd64 (applyBurstMem m0 b.addr b.pat) wordAddr).getD (val % W)

/-- Per-arena 64-bit word `TMR.Channel` under an arbitrary list of candidate
bursts `bursts : List Burst` at rate `qBurst`, with exact architectural transfer
function `fn`. -/
def burstWordChannel (fn : Nat → Nat) (wordAddr : Nat) (bursts : List Burst)
    (qBurst : Rat) (hq : Rate qBurst) : TMR.Channel Nat where
  f := fn
  step v := unif qBurst (fn v) (bursts.map (corruptWordByBurst wordAddr (fn v)))
  q := qBurst
  q0 := hq.1
  ok _ := unif_ok hq _ _
  clean _ := faultFree_unif _ _ _
  le _ := fm_unif_le hq _ _

private theorem leVal_lt_pow : ∀ (bs : List Nat),
    (∀ b ∈ bs, b < 256) → leVal bs < 256 ^ bs.length
  | [], _ => by simp [leVal]
  | b :: bs, h => by
    have hb : b < 256 := h b (List.mem_cons_self ..)
    have hbs : leVal bs < 256 ^ bs.length :=
      leVal_lt_pow bs (fun x hx => h x (List.mem_cons_of_mem _ hx))
    simp only [leVal, List.length_cons, Nat.pow_succ, Nat.mul_comm (256 ^ bs.length) 256]
    omega

private theorem rd_lt_256_of_all : ∀ (m : Mem) (a : Nat) {b : Nat},
    (∀ r ∈ m, ∀ x ∈ r.data, x < 256) → m.rd a = some b → b < 256 := by
  intro m a b hall hrd
  unfold Mem.rd at hrd
  simp only [bind, Option.bind] at hrd
  split at hrd
  · contradiction
  · rename_i p hp
    obtain ⟨i, r⟩ := p
    have hmi : m[i]? = some r := by
      have := (find_spec m a 0 hp).2
      simpa using this
    have hr_in : r ∈ m := List.mem_of_getElem? hmi
    have hb_in : b ∈ r.data := List.mem_of_getElem? hrd
    exact hall r hr_in b hb_in

private theorem burstWrByte_all_lt_256 {m : Mem}
    (hall : ∀ r ∈ m, ∀ x ∈ r.data, x < 256) (a v : Nat) :
    ∀ r ∈ burstWrByte m a v, ∀ x ∈ r.data, x < 256 := by
  unfold burstWrByte
  cases hwr : m.wr a v with
  | none => exact hall
  | some m' =>
    simp only [Option.getD_some]
    simp only [Mem.wr, bind, Option.bind] at hwr
    split at hwr
    · contradiction
    · rename_i p hp
      obtain ⟨i, r⟩ := p
      have hmi : m[i]? = some r := by
        have := (find_spec m a 0 hp).2
        simpa using this
      have hr_in : r ∈ m := List.mem_of_getElem? hmi
      dsimp only at hwr
      split at hwr
      · simp only [Option.some.injEq, pure] at hwr
        subst hwr
        intro s hs x hx
        rcases List.mem_or_eq_of_mem_set hs with hs_in | rfl
        · exact hall s hs_in x hx
        · rcases List.mem_or_eq_of_mem_set hx with hx_in | rfl
          · exact hall r hr_in x hx_in
          · omega
      · contradiction

private theorem applyBurstMem_all_lt_256 {m : Mem}
    (hall : ∀ r ∈ m, ∀ x ∈ r.data, x < 256) (addr : Nat) (pat : List Nat) :
    ∀ r ∈ applyBurstMem m addr pat, ∀ x ∈ r.data, x < 256 := by
  induction pat generalizing m addr with
  | nil => exact hall
  | cons b bs ih =>
    exact ih (burstWrByte_all_lt_256 hall addr b) (addr + 1)

private theorem rdN_all_lt_256 {m : Mem} (hall : ∀ r ∈ m, ∀ x ∈ r.data, x < 256) :
    ∀ (a n : Nat) {bs : List Nat}, m.rdN a n = some bs → bs.length = n ∧ ∀ b ∈ bs, b < 256
  | _, 0, bs, h => by
    simp only [Mem.rdN, Option.some.injEq] at h
    subst h
    exact ⟨rfl, by intro b hb; simp at hb⟩
  | a, n + 1, bs, h => by
    simp only [Mem.rdN, bind, Option.bind] at h
    split at h
    · contradiction
    · rename_i b0 hb0
      dsimp only at h
      split at h
      · contradiction
      · rename_i rest hrest
        simp only [Option.some.injEq, pure] at h
        subst h
        obtain ⟨hlen, hrest_lt⟩ := rdN_all_lt_256 hall (a + 1) n hrest
        refine ⟨by simp [hlen], ?_⟩
        intro b hb
        rcases List.mem_cons.1 hb with rfl | hmem
        · exact rd_lt_256_of_all m a hall hb0
        · exact hrest_lt b hmem

private theorem corruptWordByBurst_lt_W (wordAddr val : Nat) (b : Burst) :
    corruptWordByBurst wordAddr val b < W := by
  unfold corruptWordByBurst rd64
  dsimp only
  have hm0 : ∀ r ∈ ([⟨wordAddr, bytes8 val, true⟩] : Mem), ∀ x ∈ r.data, x < 256 := by
    intro r hr x hx
    simp only [List.mem_singleton] at hr
    subst hr
    simp only [bytes8, List.mem_map, List.mem_range] at hx
    obtain ⟨i, _, rfl⟩ := hx
    omega
  have hm1 := applyBurstMem_all_lt_256 hm0 b.addr b.pat
  cases hrd : (applyBurstMem [⟨wordAddr, bytes8 val, true⟩] b.addr b.pat).rdN wordAddr 8 with
  | none =>
    simp only [Option.map, Option.getD]
    have hWpos : 0 < W := by decide
    exact Nat.mod_lt val hWpos
  | some bs =>
    simp only [Option.map, Option.getD]
    obtain ⟨hlen, hlt⟩ := rdN_all_lt_256 hm1 wordAddr 8 hrd
    have hpow := leVal_lt_pow bs hlt
    rw [hlen] at hpow
    have hW : (256 : Nat) ^ 8 = W := by decide
    rwa [hW] at hpow

private theorem mem_unif_burst {β : Type} {p : Rat} {c : β} {xs : List β} {y : (β × Bool) × Rat}
    (hy : y ∈ unif p c xs) : y.1.1 = c ∨ y.1.1 ∈ xs := by
  cases xs with
  | nil =>
    simp only [unif, pureF, List.mem_cons, List.not_mem_nil, or_false] at hy
    subst hy; exact Or.inl rfl
  | cons x r =>
    simp only [unif, List.mem_cons, List.mem_map] at hy
    rcases hy with h | ⟨z, hz, h⟩
    · rw [h]; exact Or.inl rfl
    · rw [← h]; exact Or.inr (by simpa using hz)

theorem burstWordChannel_lt_W {fn : Nat → Nat} (hfn : ∀ v, v < W → fn v < W)
    (wordAddr : Nat) (bursts : List Burst) (qBurst : Rat) (hq : Rate qBurst)
    {v : Nat} (hv : v < W) :
    ∀ y ∈ (burstWordChannel fn wordAddr bursts qBurst hq).step v, y.1.1 < W := by
  intro y hy
  dsimp only [burstWordChannel] at hy
  rcases mem_unif_burst hy with heq | hmem
  · rw [heq]; exact hfn v hv
  · simp only [List.mem_map] at hmem
    obtain ⟨b, _, hcorr⟩ := hmem
    rw [← hcorr]
    exact corruptWordByBurst_lt_W wordAddr (fn v) b

/-- One voted step under `burstWordChannel` has error probability at most `3 · qBurst²`. -/
theorem burst_tri_err (fn : Nat → Nat) (wordAddr : Nat) (bursts : List Burst)
    (qBurst : Rat) (hq : Rate qBurst) (v : Nat) :
    fm (tri ((burstWordChannel fn wordAddr bursts qBurst hq).step v) (fn v)) ≤
      3 * (qBurst * qBurst) :=
  TMR.tri_err (burstWordChannel fn wordAddr bursts qBurst hq) v

/-- `n` voted steps under `burstWordChannel` have error probability at most
`n · 3 · qBurst²`. -/
theorem burst_trun_err (fn : Nat → Nat) (wordAddr : Nat) (bursts : List Burst)
    (qBurst : Rat) (hq : Rate qBurst) (n : Nat) (o : Option Nat) :
    fm (trun (burstWordChannel fn wordAddr bursts qBurst hq) n o) ≤
      (n : Rat) * (3 * (qBurst * qBurst)) :=
  TMR.trun_err (burstWordChannel fn wordAddr bursts qBurst hq) n o

/-- Decision-50 exposure bound under `B`-byte burst faults: a word left unvoted
for `k` burst opportunities and then written fails majority voting with
probability at most `3 · ((k + 1) · qBurst)²`. -/
theorem burst_exposure_err (fn : Nat → Nat) (wordAddr : Nat) (bursts : List Burst)
    (qBurst : Rat) (hq : Rate qBurst) (k : Nat) (v : Nat) :
    let C := burstWordChannel fn wordAddr bursts qBurst hq
    fm (tri (exposed C k v) (iter C.f (k + 1) v)) ≤
      3 * ((((k + 1 : Nat) : Rat) * qBurst) * (((k + 1 : Nat) : Rat) * qBurst)) :=
  TMR.exposure_err (burstWordChannel fn wordAddr bursts qBurst hq) k v

/-- End-to-end bound for a triplicated burst-corrupted memory word step followed
by the 9-step physical machine-code voter (`TMRVoter.votedStep`):
valid distribution, clean convergence to `TMRVoter.voterDone s (fn v)`, and
fault probability at most `3 · qBurst² + 9 · r.total`. -/
theorem burst_voted_step_phys_bound {fn : Nat → Nat} (hfn : ∀ v, v < W → fn v < W)
    (wordAddr : Nat) (bursts : List Burst) (qBurst : Rat) (hq : Rate qBurst)
    (r : Rates) (hr : r.ok) (s : St) {v : Nat} (hv : v < W) :
    let C := burstWordChannel fn wordAddr bursts qBurst hq
    (TMRVoter.votedStep C r hr v s).ok ∧
    Out.ok (TMRVoter.voterDone s (fn v)) ∈ faultFree (TMRVoter.votedStep C r hr v s) ∧
    (∀ u ∈ faultFree (TMRVoter.votedStep C r hr v s), u = .ok (TMRVoter.voterDone s (fn v))) ∧
    fm (TMRVoter.votedStep C r hr v s) ≤ 3 * (qBurst * qBurst) + 9 * r.total :=
  TMRVoter.voted_step_with_faulty_voter_bound
    (burstWordChannel fn wordAddr bursts qBurst hq) r hr v s
    (hfn v hv) (burstWordChannel_lt_W hfn wordAddr bursts qBurst hq hv)

/-! ### Concrete Decision-50 4096B Arenas, Guard-Page Arenas, and Negative Control -/

/-- Canonical Decision-50 page-disjoint 4096-byte writable arenas at
`0x100000`, `0x200000`, and `0x300000`. -/
def d50Arena0 : Arena := ⟨0x100000, 4096⟩
def d50Arena1 : Arena := ⟨0x200000, 4096⟩
def d50Arena2 : Arena := ⟨0x300000, 4096⟩

/-- Tight guard-page layout: three 4096-byte arenas at `0x100000`, `0x102000`,
and `0x104000`, separated by single 4096-byte `PROT_NONE` guard pages
(`[0x101000, 0x102000)` and `[0x103000, 0x104000)`). -/
def guardArena0 : Arena := ⟨0x100000, 4096⟩
def guardArena1 : Arena := ⟨0x102000, 4096⟩
def guardArena2 : Arena := ⟨0x104000, 4096⟩

theorem d50Arenas_sep64 : pairwiseSep cacheLineB d50Arena0 d50Arena1 d50Arena2 := by
  unfold pairwiseSep Arena.sep Arena.hi cacheLineB d50Arena0 d50Arena1 d50Arena2
  decide +kernel

theorem guardArenas_sep64 : pairwiseSep cacheLineB guardArena0 guardArena1 guardArena2 := by
  unfold pairwiseSep Arena.sep Arena.hi cacheLineB guardArena0 guardArena1 guardArena2
  decide +kernel

/-- For the canonical Decision-50 arenas (`0x100000`, `0x200000`, `0x300000`),
EVERY single `≤ 64`-byte adversarial burst `(addr, pat)` hits at most 1 arena
and causes zero single-burst SDC on any 64-bit word offset `off + 8 ≤ 4096`. -/
theorem d50_single_burst64_isolation {off : Nat} (hoff : off + 8 ≤ 4096)
    (m : Mem) {g : Nat}
    (h0 : rdArenaWord m d50Arena0 off = g)
    (h1 : rdArenaWord m d50Arena1 off = g)
    (h2 : rdArenaWord m d50Arena2 off = g)
    (addr : Nat) (pat : List Nat) (hpat : pat.length ≤ cacheLineB) :
    let m' := applyBurstMem m addr pat
    burstHitCount ⟨addr, pat⟩ d50Arena0 d50Arena1 d50Arena2 ≤ 1 ∧
    TMR.vote (rdArenaWord m' d50Arena0 off)
             (rdArenaWord m' d50Arena1 off)
             (rdArenaWord m' d50Arena2 off) = some g ∧
    TMR.ι (decide (TMR.vote (rdArenaWord m' d50Arena0 off)
                            (rdArenaWord m' d50Arena1 off)
                            (rdArenaWord m' d50Arena2 off) ≠ some g)) = 0 :=
  ⟨(burst_hits_at_most_one_arena d50Arenas_sep64 ⟨addr, pat⟩ hpat).1,
   single_burst_vote_word_exact d50Arenas_sep64 hoff hoff hoff m h0 h1 h2 ⟨addr, pat⟩ hpat,
   sdc_single_burst_zero d50Arenas_sep64 hoff hoff hoff m h0 h1 h2 addr pat hpat⟩

/-- Adversarial 64-byte all-zero payload (`0x00` × 64). -/
def pat64Zero : List Nat := List.replicate 64 0x00

/-- Adversarial 64-byte all-ones payload (`0xff` × 64). -/
def pat64Ones : List Nat := List.replicate 64 0xff

/-- Adversarial 64-byte alternating payload (`0xa5, 0x5a, ...`). -/
def pat64Alt : List Nat := (List.range 64).map fun i => if i % 2 == 0 then 0xa5 else 0x5a

/-- Concrete 3-arena test memory with 64-byte windows at `0x100000`, `0x200000`,
and `0x300000` initialized to golden word `0x0123456789abcdef` at offset `0`. -/
def goldenWord : Nat := 0x0123456789abcdef

def demoSeparatedMem : Mem :=
  let init64 := bytes8 goldenWord ++ List.replicate 56 0
  [ ⟨0x100000, init64, true⟩
  , ⟨0x200000, init64, true⟩
  , ⟨0x300000, init64, true⟩
  ]

/-- Concrete ground witness: applying 64-byte adversarial bursts (`all-0x00`,
`all-0xff`, `alternating`) directly onto `A0`, `A1`, or `A2` corrupts the
targeted replica while `TMR.vote` across the 3 arenas still returns
`some goldenWord`. -/
theorem demo_separated_single_burst64_vote_ok :
    (let m0 := applyBurstMem demoSeparatedMem 0x100000 pat64Zero
     rdArenaWord m0 d50Arena0 0 = 0 ∧
     TMR.vote (rdArenaWord m0 d50Arena0 0) (rdArenaWord m0 d50Arena1 0) (rdArenaWord m0 d50Arena2 0) =
       some goldenWord) ∧
    (let m1 := applyBurstMem demoSeparatedMem 0x200000 pat64Ones
     rdArenaWord m1 d50Arena1 0 = 0xffffffffffffffff ∧
     TMR.vote (rdArenaWord m1 d50Arena0 0) (rdArenaWord m1 d50Arena1 0) (rdArenaWord m1 d50Arena2 0) =
       some goldenWord) ∧
    (let m2 := applyBurstMem demoSeparatedMem 0x300000 pat64Alt
     TMR.vote (rdArenaWord m2 d50Arena0 0) (rdArenaWord m2 d50Arena1 0) (rdArenaWord m2 d50Arena2 0) =
       some goldenWord) := by
  decide +kernel

/-- **Negative Control (Packed Replicas `< 64B` Apart)**:
When two replica arenas `packedA0` (`[0x100000, 0x100008)`) and `packedA1`
(`[0x100020, 0x100028)`) are placed within the same 64-byte cache line
(24-byte gap `< 64`), pairwise 64-byte separation fails, and a SINGLE 64-byte
burst at `0x100000` (`pat64Ones`) hits BOTH `packedA0` and `packedA1`,
corrupting 2 of 3 replicas and flipping `TMR.vote` from `some goldenWord` to
`some 0xffffffffffffffff` (`SDC = 1`). -/
def packedA0 : Arena := ⟨0x100000, 8⟩
def packedA1 : Arena := ⟨0x100020, 8⟩
def packedA2 : Arena := ⟨0x200000, 8⟩

def demoPackedMem : Mem :=
  [ ⟨0x100000, bytes8 goldenWord ++ List.replicate 24 0 ++ bytes8 goldenWord ++ List.replicate 24 0, true⟩
  , ⟨0x200000, bytes8 goldenWord, true⟩
  ]

theorem packed_arenas_not_sep64 :
    ¬ Arena.sep cacheLineB packedA0 packedA1 := by
  unfold Arena.sep Arena.hi cacheLineB packedA0 packedA1
  decide +kernel

theorem packed_arenas_single_burst_sdc :
    let b : Burst := ⟨0x100000, pat64Ones⟩
    let m' := applyBurstMem demoPackedMem b.addr b.pat
    b.pat.length ≤ cacheLineB ∧
    burstHitCount b packedA0 packedA1 packedA2 = 2 ∧
    rdArenaWord demoPackedMem packedA0 0 = goldenWord ∧
    rdArenaWord demoPackedMem packedA1 0 = goldenWord ∧
    rdArenaWord demoPackedMem packedA2 0 = goldenWord ∧
    TMR.vote (rdArenaWord m' packedA0 0)
             (rdArenaWord m' packedA1 0)
             (rdArenaWord m' packedA2 0) = some 0xffffffffffffffff ∧
    TMR.vote (rdArenaWord m' packedA0 0)
             (rdArenaWord m' packedA1 0)
             (rdArenaWord m' packedA2 0) ≠ some goldenWord := by
  decide +kernel

/-! ### Concrete `PDist.p7` (`[5·10⁻⁸, 2·10⁻⁷]`) Rational Witness -/

private theorem inc_lt_W (v : Nat) (_ : v < W) : (v + 1) % W < W :=
  Nat.mod_lt _ (by decide)

/-- Concrete `p7` (`[5·10⁻⁸, 2·10⁻⁷]`) rational witness under 64-byte adversarial
cache-line bursts (`pat64Zero`, `pat64Ones`, `pat64Alt`):
  1. On a single shared 3-arena memory state `s` holding `d50Arena0`, `d50Arena1`,
     `d50Arena2`, one burst draw has `fm (sharedVoteAfter1 ...) = 0`, and three
     sequential burst draws have `fm (sharedVoteAfter3 ...) ≤ 3 / 25000000000000`.
  2. Under `burstWordChannel` + the 9-step physical machine-code voter
     (`TMRVoter.votedStep`), one voted step cleans non-vacuously to
     `TMRVoter.voterDone s ((v + 1) % W)` and faults with probability
     `≤ 180000003 / 25000000000000`. -/
theorem witness_burst_p7 (qBurst : Rat) (hq : PDist.p7.mem qBurst)
    (r : Rates) (hr : r.ok)
    (hf : PDist.p7.mem r.fetch) (hg : PDist.p7.mem r.reg)
    (hm : PDist.p7.mem r.mem) (hl : PDist.p7.mem r.flag)
    (s : St) {g v : Nat} (hv : v < W)
    (h0 : rdArenaWord s.mem d50Arena0 0 = g)
    (h1 : rdArenaWord s.mem d50Arena1 0 = g)
    (h2 : rdArenaWord s.mem d50Arena2 0 = g) :
    let hqRate := PDist.p7.rate PDist.p7_prob hq
    let bursts := memBursts [pat64Zero, pat64Ones, pat64Alt]
    let wBursts : List Burst :=
      [⟨0x100000, pat64Zero⟩, ⟨0x100000, pat64Ones⟩, ⟨0x100000, pat64Alt⟩]
    let C := burstWordChannel (fun x => (x + 1) % W) 0x100000 wBursts qBurst hqRate
    fm (sharedVoteAfter1 bursts qBurst hqRate d50Arena0 d50Arena1 d50Arena2 0 g s) = 0 ∧
    (sharedVoteAfter3 bursts qBurst hqRate d50Arena0 d50Arena1 d50Arena2 0 g s).ok ∧
    fm (sharedVoteAfter3 bursts qBurst hqRate d50Arena0 d50Arena1 d50Arena2 0 g s) ≤
      3 / 25000000000000 ∧
    (TMRVoter.votedStep C r hr v s).ok ∧
    Out.ok (TMRVoter.voterDone s ((v + 1) % W)) ∈ faultFree (TMRVoter.votedStep C r hr v s) ∧
    fm (TMRVoter.votedStep C r hr v s) ≤ 180000003 / 25000000000000 := by
  intro hqRate bursts wBursts C
  have hpats : ∀ pat ∈ [pat64Zero, pat64Ones, pat64Alt], pat.length ≤ cacheLineB := by
    intro pat hpat
    rcases List.mem_cons.1 hpat with rfl | hpat
    · decide
    · rcases List.mem_cons.1 hpat with rfl | hpat
      · decide
      · rcases List.mem_cons.1 hpat with rfl | hpat
        · decide
        · simp at hpat
  have hB : ∀ st, ∀ b ∈ bursts st, b.pat.length ≤ cacheLineB :=
    memBursts_wf hpats
  have h1z := shared_mem_single_burst_err_zero d50Arenas_sep64
    (by decide) (by decide) (by decide) bursts hB qBurst hqRate s h0 h1 h2
  have h3 := shared_mem_three_burst_draws_err_le d50Arenas_sep64
    (by decide) (by decide) (by decide) bursts hB qBurst hqRate s h0 h1 h2
  have hq_sq : qBurst * qBurst ≤ (1 / 5000000 : Rat) * (1 / 5000000) :=
    TMR.sq_le hqRate.1 hq.2
  have h3_rat : 3 * ((1 / 5000000 : Rat) * (1 / 5000000)) = 3 / 25000000000000 := by
    decide +kernel
  have h3_le : 3 * (qBurst * qBurst) ≤ 3 / 25000000000000 :=
    Rat.le_trans (Rat.mul_le_mul_of_nonneg_left hq_sq (by decide)) (h3_rat ▸ Rat.le_refl)
  have hphys := burst_voted_step_phys_bound inc_lt_W 0x100000 wBursts qBurst hqRate r hr s hv
  have htot : r.total ≤ 4 / 5000000 := by
    have hsum : r.fetch + (r.reg + (r.mem + r.flag)) ≤
        (1 / 5000000 : Rat) + (1 / 5000000 + (1 / 5000000 + 1 / 5000000)) :=
      add_le_add' hf.2 (add_le_add' hg.2 (add_le_add' hm.2 hl.2))
    have h4 : (1 / 5000000 : Rat) + (1 / 5000000 + (1 / 5000000 + 1 / 5000000)) = 4 / 5000000 := by
      decide +kernel
    exact Rat.le_trans hsum (h4 ▸ Rat.le_refl)
  have hphys_le : 3 * (qBurst * qBurst) + 9 * r.total ≤ 180000003 / 25000000000000 := by
    have h3q : 3 * (qBurst * qBurst) ≤ 3 * ((1 / 5000000 : Rat) * (1 / 5000000)) :=
      Rat.mul_le_mul_of_nonneg_left hq_sq (by decide)
    have h9 : 9 * r.total ≤ 9 * (4 / 5000000 : Rat) :=
      Rat.mul_le_mul_of_nonneg_left htot (by decide)
    exact Rat.le_trans (add_le_add' h3q h9) (TMRVoter.p7_step_bound_rat ▸ Rat.le_refl)
  exact ⟨h1z, h3.1, Rat.le_trans h3.2 h3_le,
         hphys.1, hphys.2.1, Rat.le_trans hphys.2.2.2 hphys_le⟩

end TMRBurst
