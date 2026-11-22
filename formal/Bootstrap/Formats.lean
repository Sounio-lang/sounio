/-
The two listing formats of the bootstrap roots, as inductive grammars, and a
reference decoder for each, proved equivalent to its grammar.

The grammars are the specification: `HexA L bs` says that the listing `L` (format
A, bootstrap/roots/a) describes exactly the bytes `bs`; `OctB L bs` likewise for
format B (bootstrap/roots/b). The decoders follow the seeds' own state (pending
nibble or digit count, inside a comment or not), so that the proof that a seed
computes its decoder can relate machine state to decoder state step by step.

Byte values: '0' 48 … '9' 57, 'a' 97 … 'f' 102, ' ' 32, '\n' 10, '\t' 9,
'#' 35, ';' 59.
-/
namespace X86Min.Formats

/-! ## Format A: hexadecimal pairs (seed-a) -/

/-- Value of a format-A digit: '0'-'9' and 'a'-'f' (lower case only). -/
def hexVal (c : Nat) : Option Nat :=
  if 48 ≤ c ∧ c ≤ 57 then some (c - 48)
  else if 97 ≤ c ∧ c ≤ 102 then some (c - 87) else none

/-- Space, newline, tab. -/
def isSep (c : Nat) : Bool := c == 32 || c == 10 || c == 9

/-- `HexA L p o`: reading `L` with pending high nibble `p` produces the bytes `o`.
Separators and comments may stand anywhere, also between the two digits of a
byte; a comment runs from '#' to the end of the line or of the input; the input
must end on a byte boundary. -/
inductive HexA : List Nat → Option Nat → List Nat → Prop where
  | done : HexA [] none []
  | sep {c l p o} : isSep c = true → HexA l p o → HexA (c :: l) p o
  | comment {body l p o} : 10 ∉ body → HexA l p o → HexA (35 :: body ++ 10 :: l) p o
  | commentEnd {body} : 10 ∉ body → HexA (35 :: body) none []
  | hi {c v l o} : hexVal c = some v → HexA l (some v) o → HexA (c :: l) none o
  | lo {c v h l o} : hexVal c = some v → HexA l none o → HexA (c :: l) (some h) ((h * 16 + v) :: o)

/-- The reference decoder, in the seed's own state: `com` (inside a comment, r14),
`p` (pending nibble, r12's sentinel form), `acc` (bytes written, latest first). -/
def decA : List Nat → Bool → Option Nat → List Nat → Option (List Nat)
  | [], _, none, acc => some acc.reverse
  | [], _, some _, _ => none
  | c :: l, true, p, acc => if c = 10 then decA l false p acc else decA l true p acc
  | c :: l, false, p, acc =>
    if c = 35 then decA l true p acc
    else if isSep c then decA l false p acc
    else match hexVal c, p with
      | some v, none => decA l false (some v) acc
      | some v, some h => decA l false none ((h * 16 + v) :: acc)
      | none, _ => none

/-! ## Format B: octal groups (seed-b) -/

/-- Value of an octal digit '0'-'7'. -/
def octVal (c : Nat) : Option Nat := if 48 ≤ c ∧ c ≤ 55 then some (c - 48) else none

/-- `OctB L o`: the listing `L` describes the bytes `o`. A byte is a group of exactly
three contiguous octal digits whose first digit is '0'-'3'; separators and ';'
comments stand only between groups. -/
inductive OctB : List Nat → List Nat → Prop where
  | done : OctB [] []
  | sep {c l o} : isSep c = true → OctB l o → OctB (c :: l) o
  | comment {body l o} : 10 ∉ body → OctB l o → OctB (59 :: body ++ 10 :: l) o
  | commentEnd {body} : 10 ∉ body → OctB (59 :: body) []
  | group {c1 c2 c3 v1 v2 v3 l o} :
      octVal c1 = some v1 → v1 ≤ 3 → octVal c2 = some v2 → octVal c3 = some v3 →
      OctB l o → OctB (c1 :: c2 :: c3 :: l) ((v1 * 64 + v2 * 8 + v3) :: o)

/-- The reference decoder, in the seed's own state: `com` (r14), `n` digits read in
the current group (r13), `a` the accumulator (r12), `acc` bytes written. -/
def decB : List Nat → Bool → Nat → Nat → List Nat → Option (List Nat)
  | [], _, n, _, acc => if n = 0 then some acc.reverse else none
  | c :: l, true, n, a, acc => if c = 10 then decB l false n a acc else decB l true n a acc
  | c :: l, false, n, a, acc =>
    if c = 59 || isSep c then
      (if n ≠ 0 then none else if c = 59 then decB l true 0 a acc else decB l false 0 a acc)
    else match octVal c with
      | none => none
      | some v =>
        if n = 0 && v > 3 then none
        else if n + 1 = 3 then decB l false 0 0 ((a * 8 + v) :: acc)
        else decB l false (n + 1) (a * 8 + v) acc

end X86Min.Formats

/-! ## Format A: the decoder is the grammar -/
namespace X86Min.Formats

theorem hexVal_not_sep {c v : Nat} (h : hexVal c = some v) : isSep c = false := by
  unfold hexVal at h; unfold isSep
  split at h
  · have : c ≠ 32 ∧ c ≠ 10 ∧ c ≠ 9 := by omega
    simp [this.1, this.2.1, this.2.2]
  · split at h
    · have : c ≠ 32 ∧ c ≠ 10 ∧ c ≠ 9 := by omega
      simp [this.1, this.2.1, this.2.2]
    · cases h

theorem hexVal_not_hash {c v : Nat} (h : hexVal c = some v) : c ≠ 35 := by
  unfold hexVal at h; split at h
  · omega
  · split at h
    · omega
    · cases h

theorem isSep_ne_hash {c : Nat} (h : isSep c = true) : c ≠ 35 := by
  unfold isSep at h; simp at h; omega

/-- Inside a comment, the decoder skips every byte that is not a newline. -/
theorem decA_skip (p : Option Nat) (acc : List Nat) :
    ∀ (body rest : List Nat), 10 ∉ body → decA (body ++ rest) true p acc = decA rest true p acc := by
  intro body; induction body with
  | nil => intro rest _; rfl
  | cons b body ih =>
    intro rest h
    have hb : b ≠ 10 := fun e => h (e ▸ List.mem_cons_self ..)
    have hbody : 10 ∉ body := fun m => h (List.mem_cons_of_mem _ m)
    simp only [List.cons_append, decA, hb, if_false]
    exact ih rest hbody

/-- Every derivation of the grammar is accepted by the decoder, with the same bytes. -/
theorem decA_complete {L : List Nat} {p : Option Nat} {o : List Nat} (h : HexA L p o) :
    ∀ acc, decA L false p acc = some (acc.reverse ++ o) := by
  induction h with
  | done => intro acc; simp [decA]
  | sep hs _ ih =>
    intro acc; simp only [decA, isSep_ne_hash hs, if_false, hs, if_true]; exact ih acc
  | comment hb _ ih =>
    intro acc
    simp only [decA, if_true, List.cons_append]
    rw [decA_skip _ _ _ _ hb]
    simp only [decA, if_true]; exact ih acc
  | commentEnd hb =>
    intro acc
    simp only [decA, if_true]
    have := decA_skip none acc _ [] hb
    rw [List.append_nil] at this; rw [this]; simp [decA]
  | hi hv _ ih =>
    intro acc
    simp only [decA, hexVal_not_hash hv, if_false, hexVal_not_sep hv, hv]
    simp; exact ih acc
  | lo hv _ ih =>
    intro acc
    simp only [decA, hexVal_not_hash hv, if_false, hexVal_not_sep hv, hv]
    simp only [Bool.false_eq_true, if_false]
    rw [ih]; simp

/-- What the decoder does inside a comment: it reaches a newline and goes on, or
reaches the end of the input. -/
theorem decA_comment_cases :
    ∀ (L : List Nat) (p : Option Nat) (acc bs : List Nat), decA L true p acc = some bs →
      ∃ body, 10 ∉ body ∧
        ((∃ rest, L = body ++ 10 :: rest ∧ decA rest false p acc = some bs) ∨
         (L = body ∧ p = none ∧ bs = acc.reverse)) := by
  intro L; induction L with
  | nil =>
    intro p acc bs h
    cases p with
    | none => simp [decA] at h; exact ⟨[], by simp, Or.inr ⟨rfl, rfl, h.symm⟩⟩
    | some _ => simp [decA] at h
  | cons c l ih =>
    intro p acc bs h
    by_cases hc : c = 10
    · subst hc; simp only [decA, if_true] at h
      exact ⟨[], by simp, Or.inl ⟨l, rfl, h⟩⟩
    · simp only [decA, hc, if_false] at h
      obtain ⟨body, hb, hcase⟩ := ih p acc bs h
      refine ⟨c :: body, ?_, ?_⟩
      · simp; exact ⟨fun e => hc e.symm, hb⟩
      · rcases hcase with ⟨rest, hl, hr⟩ | ⟨hl, hp, hbs⟩
        · exact Or.inl ⟨rest, by simp [hl], hr⟩
        · exact Or.inr ⟨by simp [hl], hp, hbs⟩

/-- Every listing the decoder accepts has a derivation, with the bytes it wrote. -/
theorem decA_sound : ∀ (n : Nat) (L : List Nat), L.length ≤ n → ∀ (p : Option Nat) (acc bs : List Nat),
    decA L false p acc = some bs → ∃ o, HexA L p o ∧ bs = acc.reverse ++ o := by
  intro n; induction n with
  | zero =>
    intro L hl p acc bs h
    have : L = [] := List.eq_nil_of_length_eq_zero (by omega)
    subst this
    cases p with
    | none => simp [decA] at h; exact ⟨[], HexA.done, by simp [h]⟩
    | some _ => simp [decA] at h
  | succ n ih =>
    intro L hl p acc bs h
    cases L with
    | nil =>
      cases p with
      | none => simp [decA] at h; exact ⟨[], HexA.done, by simp [h]⟩
      | some _ => simp [decA] at h
    | cons c l =>
      simp only [List.length_cons] at hl
      by_cases h35 : c = 35
      · subst h35; simp only [decA, if_true] at h
        obtain ⟨body, hb, hcase⟩ := decA_comment_cases l p acc bs h
        rcases hcase with ⟨rest, hl', hr⟩ | ⟨hl', hp, hbs⟩
        · have hlen : rest.length ≤ n := by
            have := congrArg List.length hl'; simp at this; omega
          obtain ⟨o, ho, hbs⟩ := ih rest hlen p acc bs hr
          exact ⟨o, by rw [hl']; exact HexA.comment hb ho, hbs⟩
        · subst hl' hp; exact ⟨[], HexA.commentEnd hb, by simp [hbs]⟩
      · simp only [decA, h35, if_false] at h
        by_cases hs : isSep c = true
        · simp only [hs, if_true] at h
          obtain ⟨o, ho, hbs⟩ := ih l (by omega) p acc bs h
          exact ⟨o, HexA.sep hs ho, hbs⟩
        · simp only [hs, if_false, Bool.false_eq_true] at h
          cases hv : hexVal c with
          | none => simp [hv] at h
          | some v =>
            cases p with
            | none =>
              simp only [hv] at h
              obtain ⟨o, ho, hbs⟩ := ih l (by omega) (some v) acc bs h
              exact ⟨o, HexA.hi hv ho, hbs⟩
            | some hh =>
              simp only [hv] at h
              obtain ⟨o, ho, hbs⟩ := ih l (by omega) none _ bs h
              exact ⟨_, HexA.lo hv ho, by rw [hbs]; simp⟩

/-- The decoder is the grammar: a listing describes `bs` exactly when the decoder
returns `bs`. -/
theorem decA_iff (L bs : List Nat) : decA L false none [] = some bs ↔ HexA L none bs := by
  constructor
  · intro h; obtain ⟨o, ho, hbs⟩ := decA_sound L.length L (Nat.le_refl _) none [] bs h
    simp at hbs; subst hbs; exact ho
  · intro h; have := decA_complete h []; simpa using this

/-- A listing describes at most one byte sequence. -/
theorem hexA_unique {L b1 b2 : List Nat} (h1 : HexA L none b1) (h2 : HexA L none b2) : b1 = b2 := by
  have e1 := (decA_iff L b1).2 h1; have e2 := (decA_iff L b2).2 h2
  rw [e1] at e2; exact Option.some.inj e2

end X86Min.Formats

/-! ## Format B: the decoder is the grammar -/
namespace X86Min.Formats

theorem octVal_not_sep {c v : Nat} (h : octVal c = some v) : isSep c = false ∧ c ≠ 59 := by
  unfold octVal at h; unfold isSep
  split at h
  · have : c ≠ 32 ∧ c ≠ 10 ∧ c ≠ 9 ∧ c ≠ 59 := by omega
    simp [this.1, this.2.1, this.2.2.1, this.2.2.2]
  · cases h

theorem octVal_lt {c v : Nat} (h : octVal c = some v) : v ≤ 7 := by
  unfold octVal at h; split at h
  · cases h; omega
  · cases h

theorem isSep_ne_semi {c : Nat} (h : isSep c = true) : c ≠ 59 := by
  unfold isSep at h; simp at h; omega

theorem decB_skip (n a : Nat) (acc : List Nat) :
    ∀ (body rest : List Nat), 10 ∉ body → decB (body ++ rest) true n a acc = decB rest true n a acc := by
  intro body; induction body with
  | nil => intro rest _; rfl
  | cons b body ih =>
    intro rest h
    have hb : b ≠ 10 := fun e => h (e ▸ List.mem_cons_self ..)
    have hbody : 10 ∉ body := fun m => h (List.mem_cons_of_mem _ m)
    simp only [List.cons_append, decB, hb, if_false]
    exact ih rest hbody

/-- One octal digit read by the decoder outside a comment. -/
theorem decB_digit {c v n a : Nat} {l acc : List Nat} (hv : octVal c = some v) :
    decB (c :: l) false n a acc =
      (if n = 0 && v > 3 then none
       else if n + 1 = 3 then decB l false 0 0 ((a * 8 + v) :: acc)
       else decB l false (n + 1) (a * 8 + v) acc) := by
  have ⟨hs, h59⟩ := octVal_not_sep hv
  simp only [decB, hs, Bool.or_false, decide_eq_true_eq, h59, if_false, hv]

theorem decB_first {c v a : Nat} {l acc : List Nat} (h : octVal c = some v) (hle : v ≤ 3) :
    decB (c :: l) false 0 a acc = decB l false 1 (a * 8 + v) acc := by
  rw [decB_digit h]; have hn : ¬ (v > 3) := by omega
  simp [hn]

theorem decB_second {c v a : Nat} {l acc : List Nat} (h : octVal c = some v) :
    decB (c :: l) false 1 a acc = decB l false 2 (a * 8 + v) acc := by
  rw [decB_digit h]; simp

theorem decB_third {c v a : Nat} {l acc : List Nat} (h : octVal c = some v) :
    decB (c :: l) false 2 a acc = decB l false 0 0 ((a * 8 + v) :: acc) := by
  rw [decB_digit h]; simp

theorem decB_eof_mid {n a : Nat} {acc : List Nat} (hn : n ≠ 0) : decB [] false n a acc = none := by
  simp [decB, hn]

/-- A whole group of three digits, outside a comment and at a group boundary. -/
theorem decB_group {c1 c2 c3 v1 v2 v3 : Nat} {l acc : List Nat}
    (h1 : octVal c1 = some v1) (hle : v1 ≤ 3) (h2 : octVal c2 = some v2) (h3 : octVal c3 = some v3) :
    decB (c1 :: c2 :: c3 :: l) false 0 0 acc = decB l false 0 0 ((v1 * 64 + v2 * 8 + v3) :: acc) := by
  rw [decB_first h1 hle, decB_second h2, decB_third h3]
  have e : (0 * 8 + v1) * 8 + v2 = v1 * 8 + v2 := by omega
  have e2 : (v1 * 8 + v2) * 8 + v3 = v1 * 64 + v2 * 8 + v3 := by omega
  rw [e, e2]

/-- Inside a group, anything but an octal digit is refused. -/
theorem decB_mid_nondigit {c n a : Nat} {l acc : List Nat} (hv : octVal c = none) (hn : n ≠ 0) :
    decB (c :: l) false n a acc = none := by
  simp only [decB]
  split
  · simp
  · simp [hv]

theorem decB_complete {L o : List Nat} (h : OctB L o) :
    ∀ acc, decB L false 0 0 acc = some (acc.reverse ++ o) := by
  induction h with
  | done => intro acc; simp [decB]
  | sep hs _ ih =>
    intro acc
    simp only [decB, hs, Bool.or_true, if_true, isSep_ne_semi hs, ne_eq, not_true, if_false]
    exact ih acc
  | comment hb _ ih =>
    intro acc
    simp only [decB, Bool.true_or, decide_true, if_true, ne_eq, not_true, if_false, List.cons_append]
    rw [decB_skip _ _ _ _ _ hb]; simp only [decB, if_true]; exact ih acc
  | commentEnd hb =>
    intro acc
    simp only [decB, Bool.true_or, decide_true, if_true, ne_eq, not_true, if_false]
    have := decB_skip 0 0 acc _ [] hb
    rw [List.append_nil] at this; rw [this]; simp [decB]
  | group h1 h1le h2 h3 _ ih =>
    intro acc
    rw [decB_group h1 h1le h2 h3, ih]; simp

theorem decB_comment_cases :
    ∀ (L : List Nat) (n a : Nat) (acc bs : List Nat), decB L true n a acc = some bs →
      ∃ body, 10 ∉ body ∧
        ((∃ rest, L = body ++ 10 :: rest ∧ decB rest false n a acc = some bs) ∨
         (L = body ∧ n = 0 ∧ bs = acc.reverse)) := by
  intro L; induction L with
  | nil =>
    intro n a acc bs h
    simp only [decB] at h
    split at h
    · exact ⟨[], by simp, Or.inr ⟨rfl, by assumption, (Option.some.inj h).symm⟩⟩
    · cases h
  | cons c l ih =>
    intro n a acc bs h
    by_cases hc : c = 10
    · subst hc; simp only [decB, if_true] at h
      exact ⟨[], by simp, Or.inl ⟨l, rfl, h⟩⟩
    · simp only [decB, hc, if_false] at h
      obtain ⟨body, hb, hcase⟩ := ih n a acc bs h
      refine ⟨c :: body, ?_, ?_⟩
      · simp; exact ⟨fun e => hc e.symm, hb⟩
      · rcases hcase with ⟨rest, hl, hr⟩ | ⟨hl, hn, hbs⟩
        · exact Or.inl ⟨rest, by simp [hl], hr⟩
        · exact Or.inr ⟨by simp [hl], hn, hbs⟩

/-- Outside a group and a comment, with a cleared accumulator: every accepted
listing has a derivation. -/
theorem decB_sound : ∀ (m : Nat) (L : List Nat), L.length ≤ m → ∀ (acc bs : List Nat),
    decB L false 0 0 acc = some bs → ∃ o, OctB L o ∧ bs = acc.reverse ++ o := by
  intro m; induction m with
  | zero =>
    intro L hl acc bs h
    have : L = [] := List.eq_nil_of_length_eq_zero (by omega)
    subst this; simp [decB] at h; exact ⟨[], OctB.done, by simp [h]⟩
  | succ m ih =>
    intro L hl acc bs h
    cases L with
    | nil => simp [decB] at h; exact ⟨[], OctB.done, by simp [h]⟩
    | cons c l =>
      simp only [List.length_cons] at hl
      by_cases h59 : c = 59
      · subst h59
        simp only [decB, Bool.true_or, decide_true, if_true, ne_eq, not_true, if_false] at h
        obtain ⟨body, hb, hcase⟩ := decB_comment_cases l 0 0 acc bs h
        rcases hcase with ⟨rest, hl', hr⟩ | ⟨hl', _, hbs⟩
        · have hlen : rest.length ≤ m := by
            have := congrArg List.length hl'; simp at this; omega
          obtain ⟨o, ho, hbs⟩ := ih rest hlen acc bs hr
          exact ⟨o, by rw [hl']; exact OctB.comment hb ho, hbs⟩
        · subst hl'; exact ⟨[], OctB.commentEnd hb, by simp [hbs]⟩
      · by_cases hs : isSep c = true
        · simp only [decB, hs, Bool.or_true, if_true, ne_eq, not_true, if_false, h59] at h
          obtain ⟨o, ho, hbs⟩ := ih l (by omega) acc bs h
          exact ⟨o, OctB.sep hs ho, hbs⟩
        · cases hv1 : octVal c with
          | none =>
            simp only [decB, h59, decide_false, Bool.false_or, hs, Bool.false_eq_true, if_false, hv1] at h
            cases h
          | some v1 =>
            by_cases hle : v1 > 3
            · rw [decB_digit hv1] at h; simp [hle] at h
            · rw [decB_first hv1 (by omega)] at h
              cases l with
              | nil => rw [decB_eof_mid (by decide)] at h; cases h
              | cons c2 l2 =>
                cases hv2 : octVal c2 with
                | none => rw [decB_mid_nondigit hv2 (by decide)] at h; cases h
                | some v2 =>
                  rw [decB_second hv2] at h
                  cases l2 with
                  | nil => rw [decB_eof_mid (by decide)] at h; cases h
                  | cons c3 l3 =>
                    cases hv3 : octVal c3 with
                    | none => rw [decB_mid_nondigit hv3 (by decide)] at h; cases h
                    | some v3 =>
                      rw [decB_third hv3] at h
                      simp only [List.length_cons] at hl
                      obtain ⟨o, ho, hbs⟩ := ih l3 (by omega) _ bs h
                      refine ⟨_, OctB.group hv1 (by omega) hv2 hv3 ho, ?_⟩
                      rw [hbs]
                      have e : (0 * 8 + v1) * 8 + v2 = v1 * 8 + v2 := by omega
                      have e2 : (v1 * 8 + v2) * 8 + v3 = v1 * 64 + v2 * 8 + v3 := by omega
                      rw [e, e2]; simp

theorem decB_iff (L bs : List Nat) : decB L false 0 0 [] = some bs ↔ OctB L bs := by
  constructor
  · intro h; obtain ⟨o, ho, hbs⟩ := decB_sound L.length L (Nat.le_refl _) [] bs h
    simp at hbs; subst hbs; exact ho
  · intro h; have := decB_complete h []; simpa using this

theorem octB_unique {L b1 b2 : List Nat} (h1 : OctB L b1) (h2 : OctB L b2) : b1 = b2 := by
  have e1 := (decB_iff L b1).2 h1; have e2 := (decB_iff L b2).2 h2
  rw [e1] at e2; exact Option.some.inj e2

end X86Min.Formats
