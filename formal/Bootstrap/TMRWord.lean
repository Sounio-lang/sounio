/-
TMRWord: the per-word fault rates that decision 50 needs, for the partial-vote
regime chosen on 2026-11-17 (registers, flags and rip voted every step; memory
words voted when written, and otherwise left exposed).

In that regime the bound for a memory word `w` left unwritten for `k` steps has
two per-step terms:
  - `q_own(w)`: an upset of `w` itself;
  - `q_stray(w)`: a fault elsewhere in the step (an address register, a fetch)
    that makes the faulty copy write `w`.

This module proves the first one exactly for the model's memory fault: under
`Fault.uniform memFlips p` (one single-bit upset, uniform over every bit of
every mapped byte), a given mapped byte is changed with probability `p / N`,
`N` the number of mapped bytes (`memFault_hits`). Of the `8·N` upsets, exactly
8 touch the byte (`countP_hits_memFlips`, `length_memFlips`).

Not here yet: `q_stray`, and the partial-vote run that combines them into
P(w wrong at its next write) ≤ 3·(k·(q_own + q_stray))².
-/
import Bootstrap.X86Int

open PDist X86Int X86Env

namespace TMRWord

theorem xor_pow_ne (v b : Nat) : Nat.xor v (2 ^ b) ≠ v := by
  intro h
  have h2 : Nat.xor (Nat.xor v (2 ^ b)) v = 0 := by rw [h]; exact Nat.xor_self v
  have h3 : Nat.xor (Nat.xor v (2 ^ b)) v = 2 ^ b := by
    show (v ^^^ 2 ^ b) ^^^ v = 2 ^ b
    rw [Nat.xor_comm v, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
  rw [h3] at h2
  exact absurd h2 (Nat.pos_iff_ne_zero.mp (Nat.two_pow_pos b))

def byteAt (s : St) (i j : Nat) : Option Nat := (s.mem[i]?).bind fun r => r.data[j]?

theorem byteAt_flipMem (s : St) (i' j' b i j : Nat) :
    byteAt (s.flipMem i' j' b) i j =
      if i' = i ∧ j' = j then (byteAt s i j).map (fun v => Nat.xor v (2 ^ b)) else byteAt s i j := by
  unfold byteAt St.flipMem
  simp only [List.getElem?_modify]
  by_cases hi : i' = i
  · subst hi
    cases h : s.mem[i']? with
    | none => simp
    | some r =>
      by_cases hj : j' = j
      · subst hj; simp
      · simp [hj]
  · simp [hi]

/-- Whether `t` differs from `s` at byte `(i, j)`. -/
def hits (s : St) (i j : Nat) (t : St) : Bool := decide (byteAt t i j ≠ byteAt s i j)

theorem hits_flipMem (s : St) (i' j' b i j : Nat) :
    hits s i j (s.flipMem i' j' b) = (decide (i' = i ∧ j' = j) && (byteAt s i j).isSome) := by
  unfold hits
  rw [byteAt_flipMem]
  by_cases h : i' = i ∧ j' = j
  · rw [if_pos h]
    cases hb : byteAt s i j with
    | none => simp [h]
    | some v => simp only [h, and_self, decide_true, Option.map_some, Option.isSome_some, Bool.and_true, ne_eq, Option.some.injEq]; exact decide_eq_true (xor_pow_ne v b)
  · rw [if_neg h]; simp [h]

theorem sum_range_ind (n i c : Nat) :
    ((List.range n).map fun k => if k = i then c else 0).sum = if i < n then c else 0 := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [List.range_succ, List.map_append, List.sum_append, ih]
    by_cases h1 : i < n
    · have : n ≠ i := by omega
      simp [h1, this, show i < n + 1 by omega]
    · by_cases h2 : i = n
      · subst h2; simp
      · have : n ≠ i := fun e => h2 e.symm
        simp [h1, this, show ¬ i < n + 1 by omega]

/-- Of all single-bit memory upsets, exactly 8 change byte `(i, j)` when it is
mapped, and none when it is not. -/
theorem countP_hits_memFlips (s : St) (i j : Nat) :
    (memFlips s).countP (hits s i j) = if (byteAt s i j).isSome then 8 else 0 := by
  unfold memFlips
  rw [List.countP_flatMap]
  have inner : ∀ i', (List.countP (hits s i j) ∘ fun i' =>
      (List.range ((s.mem[i']?.map fun r => r.data.length).getD 0)).flatMap fun j' =>
        (List.range 8).map fun b => s.flipMem i' j' b) i' =
      if i' = i then (if (byteAt s i j).isSome then 8 else 0) else 0 := by
    intro i'
    simp only [Function.comp, List.countP_flatMap]
    have hb : ∀ j', List.countP (hits s i j ∘ fun b => s.flipMem i' j' b) (List.range 8) =
        if j' = j then (if i' = i ∧ (byteAt s i j).isSome then 8 else 0) else 0 := by
      intro j'
      have : (hits s i j ∘ fun b => s.flipMem i' j' b) = fun _ => (decide (i' = i ∧ j' = j) && (byteAt s i j).isSome) := by
        funext b; simp [Function.comp, hits_flipMem]
      rw [this]
      by_cases h1 : i' = i <;> by_cases h2 : j' = j <;> by_cases h3 : (byteAt s i j).isSome <;> simp [h1, h2, h3]
    have hf : (List.countP (hits s i j) ∘ fun j' => (List.range 8).map fun b => s.flipMem i' j' b) =
        fun j' => if j' = j then (if i' = i ∧ (byteAt s i j).isSome then 8 else 0) else 0 := by
      funext j'; simp only [Function.comp, List.countP_map]; exact hb j'
    rw [hf, sum_range_ind]
    by_cases h1 : i' = i
    · subst h1
      by_cases h3 : (byteAt s i' j).isSome
      · have hj : j < (s.mem[i']?.map fun r => r.data.length).getD 0 := by
          unfold byteAt at h3
          cases hm : s.mem[i']? with
          | none => simp [hm] at h3
          | some r =>
            simp only [hm, Option.bind_some, Option.isSome_iff_exists, List.getElem?_eq_some_iff] at h3
            obtain ⟨_, hlt, _⟩ := h3
            simpa [hm] using hlt
        simp [hj, h3]
      · simp [h3]
    · simp [h1]
  rw [List.map_congr_left (fun i' _ => inner i'), sum_range_ind]
  by_cases h3 : (byteAt s i j).isSome
  · have hi : i < s.mem.length := by
      unfold byteAt at h3
      cases hm : s.mem[i]? with
      | none => simp [hm] at h3
      | some r => exact (List.getElem?_eq_some_iff.mp hm).1
    simp [hi, h3]
  · simp [h3]


/-- The weight of the outcomes whose value satisfies `P`. -/
def wBy {α : Type} (P : α → Bool) : PDF α → Rat
  | [] => 0
  | ((x, _), w) :: r => (if P x then w else 0) + wBy P r

theorem wBy_const {α : Type} (P : α → Bool) (l : List α) (w : Rat) (t : Bool) :
    wBy P (l.map fun x => ((x, t), w)) = (l.countP P : Rat) * w := by
  induction l with
  | nil => simp [wBy]
  | cons x r ih =>
    simp only [List.map_cons, wBy, ih, List.countP_cons]
    by_cases h : P x
    · simp only [h, if_true]; push_cast; grind
    · simp [h, Rat.zero_add]

theorem wBy_unif {α : Type} (P : α → Bool) {c : α} (hc : P c = false) (p : Rat) (xs : List α) :
    wBy P (unif p c xs) = (xs.countP P : Rat) * (p / xs.length) := by
  cases xs with
  | nil => simp [unif, pureF, wBy, hc, Rat.zero_add]
  | cons x r =>
    show (if P c then 1 - p else 0) + wBy P ((x :: r).map fun y => ((y, true), p / (x :: r).length)) = _
    rw [wBy_const, hc]; simp [Rat.zero_add]

/-- How many bytes are mapped. -/
def nBytes (s : St) : Nat := (s.mem.map fun r => r.data.length).sum

theorem sum_range_getElem? {α : Type} (f : Option α → Nat) :
    ∀ l : List α, ((List.range l.length).map fun i => f l[i]?).sum = (l.map fun x => f (some x)).sum := by
  intro l
  induction l with
  | nil => simp
  | cons x r ih =>
    rw [List.length_cons, List.range_succ_eq_map, List.map_cons, List.map_map]
    simp only [List.sum_cons, List.map_cons, List.getElem?_cons_zero]
    congr 1

theorem length_memFlips (s : St) : (memFlips s).length = 8 * nBytes s := by
  unfold memFlips nBytes
  rw [List.length_flatMap]
  have : ∀ i, (fun i => ((List.range ((s.mem[i]?.map fun r => r.data.length).getD 0)).flatMap
      fun j => (List.range 8).map fun b => s.flipMem i j b).length) i =
      8 * ((s.mem[i]?.map fun r => r.data.length).getD 0) := by
    intro i
    simp only [List.length_flatMap, List.length_map, List.length_range]
    generalize (s.mem[i]?.map fun r => r.data.length).getD 0 = n
    induction n with
    | zero => simp
    | succ n ih => rw [List.range_succ, List.map_append, List.sum_append, ih]; simp; omega
  rw [List.map_congr_left (fun i _ => this i)]
  have h := sum_range_getElem? (fun o => 8 * ((o.map fun r => r.data.length).getD 0)) s.mem
  simp only at h
  rw [h]
  simp only [Option.map_some, Option.getD_some]
  generalize s.mem = l
  induction l with
  | nil => simp
  | cons x r ih => simp only [List.map_cons, List.sum_cons, ih]; omega

/-- A uniform single-bit memory upset at rate `p` changes a given mapped byte
with probability exactly `p / N`, `N` the number of mapped bytes. -/
theorem memFault_hits (s : St) (i j : Nat) {p : Rat} (hp : Rate p) (h : (byteAt s i j).isSome) :
    wBy (hits s i j) ((Fault.uniform memFlips p hp).hit s) = p / nBytes s := by
  show wBy (hits s i j) (unif p s (memFlips s)) = _
  have hc : hits s i j s = false := by simp [hits]
  rw [wBy_unif _ hc, countP_hits_memFlips, if_pos h, length_memFlips]
  have hN : (nBytes s : Rat) ≠ 0 := by
    intro h0
    have : nBytes s = 0 := by exact_mod_cast h0
    have hlt : 0 < nBytes s := by
      unfold byteAt at h
      cases hm : s.mem[i]? with
      | none => simp [hm] at h
      | some r =>
        simp only [hm, Option.bind_some, Option.isSome_iff_exists, List.getElem?_eq_some_iff] at h
        obtain ⟨_, hlt, _⟩ := h
        obtain ⟨hi, hr⟩ := List.getElem?_eq_some_iff.mp hm
        unfold nBytes
        have hmem : r ∈ s.mem := hr ▸ List.getElem_mem hi
        have key : ∀ l : List Region, r ∈ l → r.data.length ≤ (l.map fun r => r.data.length).sum := by
          intro l hl
          induction l with
          | nil => simp at hl
          | cons y l ih =>
            simp only [List.map_cons, List.sum_cons]
            rcases List.mem_cons.mp hl with e | e
            · subst e; omega
            · have := ih e; omega
        have := key s.mem hmem
        omega
    omega
  push_cast
  grind

end TMRWord
