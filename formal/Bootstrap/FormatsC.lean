/-
The listing format of bootstrap root C (bootstrap/roots/c), as an inductive
grammar `BinC`, and a reference decoder `decC` proved equivalent to its grammar.

Format C:
  - each byte is a contiguous group of 8 binary digits ('0' = 48, '1' = 49),
    most significant bit first;
  - space (32), newline (10), tab (9), and ':' (58, comment to newline or EOF)
    are allowed ONLY between groups;
  - anything else (including '\r', digits '2'..'9', letters, '#', ';',
    separators or ':' inside a group, and incomplete trailing groups) is refused.
-/
namespace Bootstrap.FormatsC

/-- Value of a format-C binary digit: '0' -> 0, '1' -> 1. -/
def binVal (c : Nat) : Option Nat :=
  if c = 48 then some 0
  else if c = 49 then some 1
  else none

/-- Space, newline, tab. -/
def isSep (c : Nat) : Bool := c == 32 || c == 10 || c == 9

/-- `BinC L o`: the listing `L` describes the bytes `o` in format C. A byte is a
group of exactly 8 contiguous binary digits ('0' or '1'); separators and ':'
comments stand only between groups. -/
inductive BinC : List Nat → List Nat → Prop where
  | done : BinC [] []
  | sep {c l o} : isSep c = true → BinC l o → BinC (c :: l) o
  | comment {body l o} : 10 ∉ body → BinC l o → BinC (58 :: body ++ 10 :: l) o
  | commentEnd {body} : 10 ∉ body → BinC (58 :: body) []
  | group {c0 c1 c2 c3 c4 c5 c6 c7 v0 v1 v2 v3 v4 v5 v6 v7 l o} :
      binVal c0 = some v0 → binVal c1 = some v1 → binVal c2 = some v2 → binVal c3 = some v3 →
      binVal c4 = some v4 → binVal c5 = some v5 → binVal c6 = some v6 → binVal c7 = some v7 →
      BinC l o →
      BinC (c0 :: c1 :: c2 :: c3 :: c4 :: c5 :: c6 :: c7 :: l)
        ((v0 * 128 + v1 * 64 + v2 * 32 + v3 * 16 + v4 * 8 + v5 * 4 + v6 * 2 + v7) :: o)

/-- The reference decoder for format C, in `seed-c`'s own state: `com` (inside
a comment, x21), `n` bits read in the current group (0..7, x20), `a` the
accumulator (x19), `acc` bytes written (latest first). -/
def decC : List Nat → Bool → Nat → Nat → List Nat → Option (List Nat)
  | [], _, n, _, acc => if n = 0 then some acc.reverse else none
  | c :: l, true, n, a, acc => if c = 10 then decC l false n a acc else decC l true n a acc
  | c :: l, false, n, a, acc =>
    if c = 58 || isSep c then
      (if n ≠ 0 then none else if c = 58 then decC l true 0 a acc else decC l false 0 a acc)
    else match binVal c with
      | none => none
      | some v =>
        if n + 1 = 8 then decC l false 0 0 ((a * 2 + v) :: acc)
        else decC l false (n + 1) (a * 2 + v) acc

end Bootstrap.FormatsC

namespace Bootstrap.FormatsC

theorem binVal_not_sep {c v : Nat} (h : binVal c = some v) : isSep c = false ∧ c ≠ 58 := by
  unfold binVal at h; unfold isSep
  split at h
  · subst_vars; decide
  · split at h
    · subst_vars; decide
    · cases h

theorem binVal_le {c v : Nat} (h : binVal c = some v) : v ≤ 1 := by
  unfold binVal at h
  split at h
  · cases h; omega
  · split at h
    · cases h; omega
    · cases h

theorem isSep_ne_colon {c : Nat} (h : isSep c = true) : c ≠ 58 := by
  unfold isSep at h; simp at h; omega

theorem decC_skip (n a : Nat) (acc : List Nat) :
    ∀ (body rest : List Nat), 10 ∉ body → decC (body ++ rest) true n a acc = decC rest true n a acc := by
  intro body; induction body with
  | nil => intro rest _; rfl
  | cons b body ih =>
    intro rest h
    have hb : b ≠ 10 := fun e => h (e ▸ List.mem_cons_self ..)
    have hbody : 10 ∉ body := fun m => h (List.mem_cons_of_mem _ m)
    simp only [List.cons_append, decC, hb, if_false]
    exact ih rest hbody

theorem decC_digit {c v n a : Nat} {l acc : List Nat} (hv : binVal c = some v) :
    decC (c :: l) false n a acc =
      (if n + 1 = 8 then decC l false 0 0 ((a * 2 + v) :: acc)
       else decC l false (n + 1) (a * 2 + v) acc) := by
  have ⟨hs, h58⟩ := binVal_not_sep hv
  simp only [decC, hs, Bool.or_false, decide_eq_true_eq, h58, if_false, hv]

theorem decC_bit_step {c v n a : Nat} {l acc : List Nat} (hv : binVal c = some v) (hn : n + 1 < 8) :
    decC (c :: l) false n a acc = decC l false (n + 1) (a * 2 + v) acc := by
  rw [decC_digit hv]
  have hne : n + 1 ≠ 8 := by omega
  simp [hne]

theorem decC_bit_last {c v a : Nat} {l acc : List Nat} (hv : binVal c = some v) :
    decC (c :: l) false 7 a acc = decC l false 0 0 ((a * 2 + v) :: acc) := by
  rw [decC_digit hv]; simp

theorem decC_eof_mid {n a : Nat} {acc : List Nat} (hn : n ≠ 0) : decC [] false n a acc = none := by
  simp [decC, hn]

theorem decC_mid_nondigit {c n a : Nat} {l acc : List Nat} (hv : binVal c = none) (hn : n ≠ 0) :
    decC (c :: l) false n a acc = none := by
  simp only [decC]
  split
  · simp
  · simp [hv]

theorem decC_group {c0 c1 c2 c3 c4 c5 c6 c7 v0 v1 v2 v3 v4 v5 v6 v7 : Nat} {l acc : List Nat}
    (h0 : binVal c0 = some v0) (h1 : binVal c1 = some v1) (h2 : binVal c2 = some v2) (h3 : binVal c3 = some v3)
    (h4 : binVal c4 = some v4) (h5 : binVal c5 = some v5) (h6 : binVal c6 = some v6) (h7 : binVal c7 = some v7) :
    decC (c0 :: c1 :: c2 :: c3 :: c4 :: c5 :: c6 :: c7 :: l) false 0 0 acc =
      decC l false 0 0 ((v0 * 128 + v1 * 64 + v2 * 32 + v3 * 16 + v4 * 8 + v5 * 4 + v6 * 2 + v7) :: acc) := by
  rw [decC_bit_step h0 (by omega), decC_bit_step h1 (by omega), decC_bit_step h2 (by omega),
      decC_bit_step h3 (by omega), decC_bit_step h4 (by omega), decC_bit_step h5 (by omega),
      decC_bit_step h6 (by omega), decC_bit_last h7]
  have e : (((((((0 * 2 + v0) * 2 + v1) * 2 + v2) * 2 + v3) * 2 + v4) * 2 + v5) * 2 + v6) * 2 + v7
      = v0 * 128 + v1 * 64 + v2 * 32 + v3 * 16 + v4 * 8 + v5 * 4 + v6 * 2 + v7 := by omega
  rw [e]

theorem decC_complete {L o : List Nat} (h : BinC L o) :
    ∀ acc, decC L false 0 0 acc = some (acc.reverse ++ o) := by
  induction h with
  | done => intro acc; simp [decC]
  | sep hs _ ih =>
    intro acc
    simp only [decC, hs, Bool.or_true, if_true, isSep_ne_colon hs, ne_eq, not_true, if_false]
    exact ih acc
  | comment hb _ ih =>
    intro acc
    simp only [decC, Bool.true_or, decide_true, if_true, ne_eq, not_true, if_false, List.cons_append]
    rw [decC_skip _ _ _ _ _ hb]; simp only [decC, if_true]; exact ih acc
  | commentEnd hb =>
    intro acc
    simp only [decC, Bool.true_or, decide_true, if_true, ne_eq, not_true, if_false]
    have := decC_skip 0 0 acc _ [] hb
    rw [List.append_nil] at this; rw [this]; simp [decC]
  | group h0 h1 h2 h3 h4 h5 h6 h7 _ ih =>
    intro acc
    rw [decC_group h0 h1 h2 h3 h4 h5 h6 h7, ih]; simp

theorem decC_comment_cases :
    ∀ (L : List Nat) (n a : Nat) (acc bs : List Nat), decC L true n a acc = some bs →
      ∃ body, 10 ∉ body ∧
        ((∃ rest, L = body ++ 10 :: rest ∧ decC rest false n a acc = some bs) ∨
         (L = body ∧ n = 0 ∧ bs = acc.reverse)) := by
  intro L; induction L with
  | nil =>
    intro n a acc bs h
    simp only [decC] at h
    split at h
    · exact ⟨[], by simp, Or.inr ⟨rfl, by assumption, (Option.some.inj h).symm⟩⟩
    · cases h
  | cons c l ih =>
    intro n a acc bs h
    by_cases hc : c = 10
    · subst hc; simp only [decC, if_true] at h
      exact ⟨[], by simp, Or.inl ⟨l, rfl, h⟩⟩
    · simp only [decC, hc, if_false] at h
      obtain ⟨body, hb, hcase⟩ := ih n a acc bs h
      refine ⟨c :: body, ?_, ?_⟩
      · simp; exact ⟨fun e => hc e.symm, hb⟩
      · rcases hcase with ⟨rest, hl, hr⟩ | ⟨hl, hn, hbs⟩
        · exact Or.inl ⟨rest, by simp [hl], hr⟩
        · exact Or.inr ⟨by simp [hl], hn, hbs⟩

theorem decC_next_bit {n a : Nat} {L acc bs : List Nat} (hn1 : n ≠ 0) (hn7 : n + 1 < 8)
    (h : decC L false n a acc = some bs) :
    ∃ c v l, L = c :: l ∧ binVal c = some v ∧ decC l false (n + 1) (a * 2 + v) acc = some bs := by
  cases L with
  | nil => rw [decC_eof_mid hn1] at h; cases h
  | cons c l =>
    cases hv : binVal c with
    | none => rw [decC_mid_nondigit hv hn1] at h; cases h
    | some v =>
      rw [decC_bit_step hv hn7] at h
      exact ⟨c, v, l, rfl, hv, h⟩

theorem decC_last_bit {a : Nat} {L acc bs : List Nat} (h : decC L false 7 a acc = some bs) :
    ∃ c v l, L = c :: l ∧ binVal c = some v ∧ decC l false 0 0 ((a * 2 + v) :: acc) = some bs := by
  cases L with
  | nil => rw [decC_eof_mid (by omega)] at h; cases h
  | cons c l =>
    cases hv : binVal c with
    | none => rw [decC_mid_nondigit hv (by omega)] at h; cases h
    | some v =>
      rw [decC_bit_last hv] at h
      exact ⟨c, v, l, rfl, hv, h⟩

theorem decC_sound : ∀ (m : Nat) (L : List Nat), L.length ≤ m → ∀ (acc bs : List Nat),
    decC L false 0 0 acc = some bs → ∃ o, BinC L o ∧ bs = acc.reverse ++ o := by
  intro m; induction m with
  | zero =>
    intro L hl acc bs h
    have : L = [] := List.eq_nil_of_length_eq_zero (by omega)
    subst this; simp [decC] at h; exact ⟨[], BinC.done, by simp [h]⟩
  | succ m ih =>
    intro L hl acc bs h
    cases L with
    | nil => simp [decC] at h; exact ⟨[], BinC.done, by simp [h]⟩
    | cons c0 l0 =>
      simp only [List.length_cons] at hl
      by_cases h58 : c0 = 58
      · subst h58
        simp only [decC, Bool.true_or, decide_true, if_true, ne_eq, not_true, if_false] at h
        obtain ⟨body, hb, hcase⟩ := decC_comment_cases l0 0 0 acc bs h
        rcases hcase with ⟨rest, hl', hr⟩ | ⟨hl', _, hbs⟩
        · have hlen : rest.length ≤ m := by
            have := congrArg List.length hl'; simp at this; omega
          obtain ⟨o, ho, hbs⟩ := ih rest hlen acc bs hr
          exact ⟨o, by rw [hl']; exact BinC.comment hb ho, hbs⟩
        · subst hl'; exact ⟨[], BinC.commentEnd hb, by simp [hbs]⟩
      · by_cases hs : isSep c0 = true
        · simp only [decC, hs, Bool.or_true, if_true, ne_eq, not_true, if_false, h58] at h
          obtain ⟨o, ho, hbs⟩ := ih l0 (by omega) acc bs h
          exact ⟨o, BinC.sep hs ho, hbs⟩
        · cases hv0 : binVal c0 with
          | none =>
            simp only [decC, h58, decide_false, Bool.false_or, hs, Bool.false_eq_true, if_false, hv0] at h
            cases h
          | some v0 =>
            rw [decC_bit_step hv0 (by omega)] at h
            obtain ⟨c1, v1, l1, rfl, hv1, h⟩ := decC_next_bit (by omega) (by omega) h
            obtain ⟨c2, v2, l2, rfl, hv2, h⟩ := decC_next_bit (by omega) (by omega) h
            obtain ⟨c3, v3, l3, rfl, hv3, h⟩ := decC_next_bit (by omega) (by omega) h
            obtain ⟨c4, v4, l4, rfl, hv4, h⟩ := decC_next_bit (by omega) (by omega) h
            obtain ⟨c5, v5, l5, rfl, hv5, h⟩ := decC_next_bit (by omega) (by omega) h
            obtain ⟨c6, v6, l6, rfl, hv6, h⟩ := decC_next_bit (by omega) (by omega) h
            obtain ⟨c7, v7, l7, rfl, hv7, h⟩ := decC_last_bit h
            simp only [List.length_cons] at hl
            obtain ⟨o, ho, hbs⟩ := ih l7 (by omega) _ bs h
            refine ⟨_, BinC.group hv0 hv1 hv2 hv3 hv4 hv5 hv6 hv7 ho, ?_⟩
            rw [hbs]
            have e : (((((((0 * 2 + v0) * 2 + v1) * 2 + v2) * 2 + v3) * 2 + v4) * 2 + v5) * 2 + v6) * 2 + v7
                = v0 * 128 + v1 * 64 + v2 * 32 + v3 * 16 + v4 * 8 + v5 * 4 + v6 * 2 + v7 := by omega
            rw [e]; simp

theorem decC_iff (L o : List Nat) : decC L false 0 0 [] = some o ↔ BinC L o := by
  constructor
  · intro h; obtain ⟨o', ho, hbs⟩ := decC_sound L.length L (Nat.le_refl _) [] o h
    simp at hbs; subst hbs; exact ho
  · intro h; have := decC_complete h []; simpa using this

theorem binC_unique {L b1 b2 : List Nat} (h1 : BinC L b1) (h2 : BinC L b2) : b1 = b2 := by
  have e1 := (decC_iff L b1).2 h1; have e2 := (decC_iff L b2).2 h2
  rw [e1] at e2; exact Option.some.inj e2

end Bootstrap.FormatsC
