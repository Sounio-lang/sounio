/-!
# Sounio — All-Vocabulary Prompt-Level Conformal Argmax Certification

Mathlib-free Lean 4 kernel supporting Step 1d / Phase 10:
1. All-vocabulary global envelope soundness (`all_vocab_envelope_no_flip`,
   `all_vocab_margin_envelope_sound`): certifies against ALL `V` rows (zero tail loophole).
2. Margin-ordered soundness (`scaled_margin_cert_sound`): if normalized margin exceeds
   `q` and row overtaking distortion is bounded by `q`, the argmax cannot flip.
3. Decision-exact prompt flip-score equivalence (`prompt_no_false_cert_iff_flip_score_le`)
   and per-token corollary (`prompt_sound_implies_token_sound`): no token in prompt `P`
   receives a false certificate at threshold `q` if and only if the prompt-level flip
   score is `≤ q`.
-/

namespace Sounio.Conformal

/-- Every row `j < V` of `zb` lies inside the symmetric radius `r j` around `zq j`. -/
def EnvelopeSound (V : Nat) (zq zb r : Nat → Int) : Prop :=
  ∀ j, j < V → zq j - r j ≤ zb j ∧ zb j ≤ zq j + r j

/-- Global all-vocabulary certificate: `top` clears every competing row `j ≠ top` in `0..V-1`. -/
def GlobalCertified (V top : Nat) (zq r : Nat → Int) : Prop :=
  top < V ∧ ∀ j, j < V → j ≠ top → zq j + r j < zq top - r top

/-- 1) Sound all-vocabulary envelope guarantees strict argmax preservation over all `V` rows. -/
theorem all_vocab_envelope_no_flip
    {V top : Nat} {zq zb r : Nat → Int}
    (hs : EnvelopeSound V zq zb r)
    (hc : GlobalCertified V top zq r) :
    ∀ j, j < V → j ≠ top → zb j < zb top := by
  obtain ⟨htop, hclear⟩ := hc
  intro j hj hne
  have h1 := hs top htop
  have h2 := hs j hj
  have h3 := hclear j hj hne
  omega

/-- 2) Pairwise overtaking distortion condition across all `j ≠ top`: directly implies no flip
    without requiring two-sided control on tail rows individually. -/
theorem all_vocab_overtake_no_flip
    {V top : Nat} {zq zb r : Nat → Int}
    (htop : top < V)
    (ho : ∀ j, j < V → j ≠ top → (zq top - zb top) + (zb j - zq j) ≤ r top + r j)
    (hc : GlobalCertified V top zq r) :
    ∀ j, j < V → j ≠ top → zb j < zb top := by
  obtain ⟨_, hclear⟩ := hc
  have _ := htop
  intro j hj hne
  have hov := ho j hj hne
  have hcl := hclear j hj hne
  omega

/-- 3) Positive-scale linearized margin: with threshold `q ≥ 0` and positive scales `s`,
    `(zq top - zq j) > q * (s top + s j)` implies `GlobalCertified` for radius `q * s`. -/
theorem scaled_margin_cert_sound
    {V top : Nat} {zq zb s : Nat → Int} {q : Int}
    (htop : top < V)
    (hmargin : ∀ j, j < V → j ≠ top → q * (s top + s j) < zq top - zq j)
    (hs : ∀ j, j < V → zq j - q * s j ≤ zb j ∧ zb j ≤ zq j + q * s j) :
    ∀ j, j < V → j ≠ top → zb j < zb top := by
  apply all_vocab_envelope_no_flip (r := fun j => q * s j) hs
  refine ⟨htop, ?_⟩
  intro j hj hne
  have hm := hmargin j hj hne
  have hdist : q * (s top + s j) = q * s top + q * s j := Int.mul_add q (s top) (s j)
  dsimp only
  omega

/-- Prompt-level flip score over a list of token `(margin_score, flipped_flag)` pairs.
    `none` denotes zero flipped tokens in the prompt. -/
def promptFlipScore (P : List (Int × Bool)) : Option Int :=
  match P with
  | [] => none
  | (mu, err) :: rest =>
    match promptFlipScore rest, err with
    | none, false => none
    | none, true => some mu
    | some m, false => some m
    | some m, true => some (if m ≤ mu then mu else m)

/-- Predicate `promptFlipScore P ≤ q` (vacuously true when `none`, i.e., no token flipped). -/
def OptLe (x : Option Int) (q : Int) : Prop :=
  match x with
  | none => True
  | some m => m ≤ q

/-- Helper: `OptLe` upper bound characterization for `promptFlipScore`. -/
theorem prompt_flip_score_upper_iff (P : List (Int × Bool)) (q : Int) :
    OptLe (promptFlipScore P) q ↔ ∀ mu err, (mu, err) ∈ P → err = true → mu ≤ q := by
  induction P with
  | nil =>
    constructor
    · intro _ mu err hmem
      cases hmem
    · intro _
      trivial
  | cons hd tl ih =>
    rcases hd with ⟨mu0, err0⟩
    simp only [promptFlipScore]
    cases htail : promptFlipScore tl <;> cases err0 <;> simp only [OptLe]
    · constructor
      · intro _ mu err hmem herr
        rcases List.mem_cons.mp hmem with heq | htl
        · cases heq; cases herr
        · rw [htail] at ih
          exact ih.mp trivial mu err htl herr
      · intro _
        trivial
    · constructor
      · intro hmu0 mu err hmem herr
        rcases List.mem_cons.mp hmem with heq | htl
        · cases heq; exact hmu0
        · rw [htail] at ih
          exact ih.mp trivial mu err htl herr
      · intro hall
        exact hall mu0 true List.mem_cons_self rfl
    · rename_i m
      rw [htail] at ih
      simp only [OptLe] at ih
      constructor
      · intro hm mu err hmem herr
        rcases List.mem_cons.mp hmem with heq | htl
        · cases heq; cases herr
        · exact ih.mp hm mu err htl herr
      · intro hall
        apply ih.mpr
        intro mu err htl herr
        exact hall mu err (List.mem_cons_of_mem _ htl) herr
    · rename_i m
      rw [htail] at ih
      simp only [OptLe] at ih
      constructor
      · intro hmax mu err hmem herr
        have hle : m ≤ q ∧ mu0 ≤ q := by
          by_cases hcmp : m ≤ mu0
          · rw [if_pos hcmp] at hmax
            omega
          · rw [if_neg hcmp] at hmax
            omega
        rcases List.mem_cons.mp hmem with heq | htl
        · cases heq; exact hle.2
        · exact ih.mp hle.1 mu err htl herr
      · intro hall
        have hm0 : mu0 ≤ q := hall mu0 true List.mem_cons_self rfl
        have hm : m ≤ q := ih.mpr (fun mu err htl herr =>
          hall mu err (List.mem_cons_of_mem _ htl) herr)
        by_cases hcmp : m ≤ mu0
        · rw [if_pos hcmp]; exact hm0
        · rw [if_neg hcmp]; exact hm

/-- 4) Decision-exact equivalence: prompt `P` produces ZERO false certificates at threshold `q`
    (`q < mu → err = false` for all tokens in `P`) IF AND ONLY IF `promptFlipScore P ≤ q`. -/
theorem prompt_no_false_cert_iff_flip_score_le (P : List (Int × Bool)) (q : Int) :
    (∀ mu err, (mu, err) ∈ P → q < mu → err = false) ↔ OptLe (promptFlipScore P) q := by
  rw [prompt_flip_score_upper_iff]
  constructor
  · intro hno mu err hmem herr
    by_cases hgt : q < mu
    · have hf := hno mu err hmem hgt
      rw [herr] at hf
      cases hf
    · omega
  · intro hle mu err hmem hgt
    cases err
    · rfl
    · have hmu : mu ≤ q := hle mu true hmem rfl
      omega

/-- 5) Per-token soundness corollary from prompt-level conformal validity:
    whenever `promptFlipScore P ≤ q`, every token in `P` with `q < mu` has `err = false`. -/
theorem prompt_sound_implies_token_sound
    {P : List (Int × Bool)} {q : Int}
    (hsound : OptLe (promptFlipScore P) q)
    {mu : Int} {err : Bool} (hmem : (mu, err) ∈ P) (hcert : q < mu) :
    err = false :=
  (prompt_no_false_cert_iff_flip_score_le P q).mpr hsound mu err hmem hcert

/-- 6) Tightness witness: if `promptFlipScore P` exceeds `q`, at least one token in `P`
    has a false certificate at threshold `q`. -/
theorem prompt_flip_score_exceed_implies_false_cert
    {P : List (Int × Bool)} {q : Int}
    (hexceed : ¬ OptLe (promptFlipScore P) q) :
    ∃ mu err, (mu, err) ∈ P ∧ q < mu ∧ err = true := by
  have hnot := mt (prompt_no_false_cert_iff_flip_score_le P q).mp hexceed
  cases h : promptFlipScore P with
  | none =>
    rw [h] at hexceed
    exact False.elim (hexceed trivial)
  | some m =>
    have hm : q < m := by
      rw [h] at hexceed
      simp only [OptLe] at hexceed
      omega
    -- By induction find witness achieving m
    clear hnot hexceed
    induction P generalizing m with
    | nil => cases h
    | cons hd tl ih =>
      rcases hd with ⟨mu0, err0⟩
      simp only [promptFlipScore] at h
      cases htl : promptFlipScore tl <;> cases err0 <;> rw [htl] at h <;> dsimp only at h
      · cases h
      · cases h
        exact ⟨m, true, List.mem_cons_self, hm, rfl⟩
      · cases h
        obtain ⟨mu, err, hmem, hgt, he⟩ := ih m htl hm
        exact ⟨mu, err, List.mem_cons_of_mem _ hmem, hgt, he⟩
      · rename_i m_tl
        by_cases hcmp : m_tl ≤ mu0
        · rw [if_pos hcmp] at h; cases h
          exact ⟨m, true, List.mem_cons_self, hm, rfl⟩
        · rw [if_neg hcmp] at h; cases h
          obtain ⟨mu, err, hmem, hgt, he⟩ := ih m htl hm
          exact ⟨mu, err, List.mem_cons_of_mem _ hmem, hgt, he⟩

end Sounio.Conformal

#print axioms Sounio.Conformal.all_vocab_envelope_no_flip
#print axioms Sounio.Conformal.all_vocab_overtake_no_flip
#print axioms Sounio.Conformal.scaled_margin_cert_sound
#print axioms Sounio.Conformal.prompt_no_false_cert_iff_flip_score_le
#print axioms Sounio.Conformal.prompt_sound_implies_token_sound
#print axioms Sounio.Conformal.prompt_flip_score_exceed_implies_false_cert

/-!
## Epistemic Certificate-Gated KV-Scrubbing (Theorems 7–10)
-/

namespace Sounio.Conformal

def nth? {α : Type} : List α → Nat → Option α
  | [], _ => none
  | x :: _, 0 => some x
  | _ :: xs, k + 1 => nth? xs k

def replayRef {S Tok : Type} (stepRef : S → Tok → S × Tok) (s : S) : List Tok → S
  | [] => s
  | x :: xs => replayRef stepRef (stepRef s x).1 xs

theorem replayRef_append {S Tok : Type}
    (stepRef : S → Tok → S × Tok) (s : S) (xs ys : List Tok) :
    replayRef stepRef s (xs ++ ys) = replayRef stepRef (replayRef stepRef s xs) ys := by
  induction xs generalizing s with
  | nil => rfl
  | cons x xs ih => exact ih (stepRef s x).1

def rolloutRef {S Tok : Type} (stepRef : S → Tok → S × Tok) (s : S) (tok : Tok) : Nat → List Tok
  | 0 => []
  | n + 1 =>
    let res := stepRef s tok
    res.2 :: rolloutRef stepRef res.1 res.2 n

def rolloutScrub {S Tok : Type}
    (stepRef : S → Tok → S × Tok)
    (stepFast : S → Tok → S × Tok × Bool)
    (sCur sClean : S) (dirty : List Tok) (tok : Tok) : Nat → List (Tok × Bool)
  | 0 => []
  | n + 1 =>
    let f := stepFast sCur tok
    if f.2.2 then
      (f.2.1, true) :: rolloutScrub stepRef stepFast f.1 sClean (dirty ++ [tok]) f.2.1 n
    else
      let sScrub := replayRef stepRef sClean dirty
      let r := stepRef sScrub tok
      (r.2, false) :: rolloutScrub stepRef stepFast r.1 r.1 [] r.2 n

/-- 7) Non-Monotone Fallback Inversion Witness:
    Constructs finite `stepRef`, `stepFast` where pure `stepFast` agrees with pure
    `stepRef` on both steps `0` and `1`, yet taking single-step `stepRef` fallback on
    un-scrubbed `stepFast` KV state disagrees at step `1`. -/
theorem fallback_inversion_witness :
    ∃ (S Tok : Type) (stepRef stepFast : S → Tok → S × Tok) (s0 : S) (tok0 : Tok),
      let f0 := stepFast s0 tok0
      let r0 := stepRef s0 tok0
      f0.2 = r0.2 ∧
      (stepFast f0.1 f0.2).2 = (stepRef r0.1 r0.2).2 ∧
      (stepRef f0.1 f0.2).2 ≠ (stepRef r0.1 r0.2).2 := by
  refine ⟨Bool, Bool,
    (fun s tok => match s, tok with
      | false, false => (true, true)
      | false, true => (true, false)
      | true, _ => (true, true)),
    (fun s _ => match s with
      | false => (false, true)
      | true => (true, true)),
    false, false, ?_⟩
  decide

/-- 8) Scrub Elimination of Fallback First-Divergences:
    Under epistemic KV-scrubbing from last clean checkpoint `sClean`, at the minimal
    divergent index `k < n`, the step MUST have been certified (`Bool = true`); an
    abstention/fallback step can never be the first divergent token. -/
theorem scrub_no_fallback_first_divergence
    {S Tok : Type}
    (stepRef : S → Tok → S × Tok)
    (stepFast : S → Tok → S × Tok × Bool)
    (sRef sCur sClean : S) (dirty : List Tok) (tok : Tok) (n : Nat)
    (hinv : sRef = replayRef stepRef sClean dirty)
    (k : Nat) (hk : k < n)
    (hpref : ∀ j, j < k →
      (nth? (rolloutScrub stepRef stepFast sCur sClean dirty tok n) j).map Prod.fst =
      nth? (rolloutRef stepRef sRef tok n) j)
    (hdiv : (nth? (rolloutScrub stepRef stepFast sCur sClean dirty tok n) k).map Prod.fst ≠
            nth? (rolloutRef stepRef sRef tok n) k) :
    ∃ y, nth? (rolloutScrub stepRef stepFast sCur sClean dirty tok n) k = some (y, true) := by
  induction n generalizing sRef sCur sClean dirty tok k with
  | zero => omega
  | succ m ih =>
    cases hcert : (stepFast sCur tok).2.2 with
    | true =>
      cases k with
      | zero =>
        refine ⟨(stepFast sCur tok).2.1, ?_⟩
        simp only [rolloutScrub, hcert, ↓reduceIte, nth?]
      | succ k' =>
        have h0 := hpref 0 (Nat.zero_lt_succ k')
        simp only [rolloutScrub, rolloutRef, hcert, ↓reduceIte, nth?, Option.map] at h0
        have hy : (stepFast sCur tok).2.1 = (stepRef sRef tok).2 := Option.some.inj h0
        have hinv' : (stepRef sRef tok).1 = replayRef stepRef sClean (dirty ++ [tok]) := by
          rw [replayRef_append, ← hinv]
          rfl
        have hpref' : ∀ j, j < k' →
            (nth? (rolloutScrub stepRef stepFast (stepFast sCur tok).1 sClean (dirty ++ [tok])
              (stepRef sRef tok).2 m) j).map Prod.fst =
            nth? (rolloutRef stepRef (stepRef sRef tok).1 (stepRef sRef tok).2 m) j := by
          intro j hj
          have hj1 := hpref (j + 1) (Nat.succ_lt_succ hj)
          simp only [rolloutScrub, rolloutRef, hcert, ↓reduceIte, hy, nth?] at hj1
          exact hj1
        have hdiv' :
            (nth? (rolloutScrub stepRef stepFast (stepFast sCur tok).1 sClean (dirty ++ [tok])
              (stepRef sRef tok).2 m) k').map Prod.fst ≠
            nth? (rolloutRef stepRef (stepRef sRef tok).1 (stepRef sRef tok).2 m) k' := by
          simp only [rolloutScrub, rolloutRef, hcert, ↓reduceIte, hy, nth?] at hdiv
          exact hdiv
        obtain ⟨y, hyk⟩ := ih (stepRef sRef tok).1 (stepFast sCur tok).1 sClean (dirty ++ [tok])
          (stepRef sRef tok).2 hinv' k' (Nat.lt_of_succ_lt_succ hk) hpref' hdiv'
        refine ⟨y, ?_⟩
        simp only [rolloutScrub, hcert, ↓reduceIte, hy, nth?]
        exact hyk
    | false =>
      have hscrub : replayRef stepRef sClean dirty = sRef := hinv.symm
      cases k with
      | zero =>
        exfalso
        apply hdiv
        simp only [rolloutScrub, rolloutRef, hcert, hscrub, Bool.false_eq_true, ↓reduceIte, nth?,
          Option.map]
      | succ k' =>
        have hinv' : (stepRef sRef tok).1 = replayRef stepRef (stepRef sRef tok).1 [] := rfl
        have hpref' : ∀ j, j < k' →
            (nth? (rolloutScrub stepRef stepFast (stepRef sRef tok).1 (stepRef sRef tok).1 []
              (stepRef sRef tok).2 m) j).map Prod.fst =
            nth? (rolloutRef stepRef (stepRef sRef tok).1 (stepRef sRef tok).2 m) j := by
          intro j hj
          have hj1 := hpref (j + 1) (Nat.succ_lt_succ hj)
          simp only [rolloutScrub, rolloutRef, hcert, hscrub, Bool.false_eq_true, ↓reduceIte, nth?] at hj1
          exact hj1
        have hdiv' :
            (nth? (rolloutScrub stepRef stepFast (stepRef sRef tok).1 (stepRef sRef tok).1 []
              (stepRef sRef tok).2 m) k').map Prod.fst ≠
            nth? (rolloutRef stepRef (stepRef sRef tok).1 (stepRef sRef tok).2 m) k' := by
          simp only [rolloutScrub, rolloutRef, hcert, hscrub, Bool.false_eq_true, ↓reduceIte, nth?] at hdiv
          exact hdiv
        obtain ⟨y, hyk⟩ := ih (stepRef sRef tok).1 (stepRef sRef tok).1 (stepRef sRef tok).1 []
          (stepRef sRef tok).2 hinv' k' (Nat.lt_of_succ_lt_succ hk) hpref' hdiv'
        refine ⟨y, ?_⟩
        simp only [rolloutScrub, hcert, hscrub, Bool.false_eq_true, ↓reduceIte, nth?]
        exact hyk

def verifyChunk {S Tok : Type} [DecidableEq Tok]
    (stepRef : S → Tok → S × Tok)
    (s : S) (tok : Tok) : Nat → List Tok → Nat × S × Tok × List Tok
  | 0, _ => (0, s, tok, [])
  | rem + 1, [] =>
    let r := stepRef s tok
    (rem, r.1, r.2, [r.2])
  | rem + 1, c :: cs =>
    let r := stepRef s tok
    if c = r.2 then
      let sub := verifyChunk stepRef r.1 r.2 rem cs
      (sub.1, sub.2.1, sub.2.2.1, r.2 :: sub.2.2.2)
    else
      (rem, r.1, r.2, [r.2])

theorem verifyChunk_le {S Tok : Type} [DecidableEq Tok]
    (stepRef : S → Tok → S × Tok)
    (s : S) (tok : Tok) (rem : Nat) (cs : List Tok) :
    (verifyChunk stepRef s tok rem cs).1 ≤ rem := by
  induction rem generalizing s tok cs with
  | zero => simp [verifyChunk]
  | succ m ih =>
    cases cs with
    | nil => simp [verifyChunk]
    | cons c cs' =>
      by_cases heq : c = (stepRef s tok).2
      · simp only [verifyChunk, if_pos heq]
        exact Nat.le_trans (ih _ _ cs') (Nat.le_succ m)
      · simp only [verifyChunk, if_neg heq]
        omega

theorem verifyChunk_lt {S Tok : Type} [DecidableEq Tok]
    (stepRef : S → Tok → S × Tok)
    (s : S) (tok : Tok) (rem : Nat) (cs : List Tok) :
    (verifyChunk stepRef s tok (rem + 1) cs).1 < rem + 1 := by
  cases cs with
  | nil => simp [verifyChunk]
  | cons c cs' =>
    by_cases heq : c = (stepRef s tok).2
    · simp only [verifyChunk, if_pos heq]
      have hle := verifyChunk_le stepRef (stepRef s tok).1 (stepRef s tok).2 rem cs'
      omega
    · simp only [verifyChunk, if_neg heq]
      omega

theorem verifyChunk_exact {S Tok : Type} [DecidableEq Tok]
    (stepRef : S → Tok → S × Tok)
    (s : S) (tok : Tok) (rem : Nat) (cs : List Tok) :
    let res := verifyChunk stepRef s tok rem cs
    res.2.2.2 ++ rolloutRef stepRef res.2.1 res.2.2.1 res.1 = rolloutRef stepRef s tok rem := by
  induction rem generalizing s tok cs with
  | zero => rfl
  | succ m ih =>
    cases cs with
    | nil =>
      rfl
    | cons c cs' =>
      by_cases heq : c = (stepRef s tok).2
      · simp only [verifyChunk, if_pos heq, List.cons_append]
        have ih' := ih (stepRef s tok).1 (stepRef s tok).2 cs'
        dsimp only at ih' ⊢
        rw [ih']
        rfl
      · simp only [verifyChunk, if_neg heq, List.singleton_append]
        rfl

def exactScrubDecode {S Tok : Type} [DecidableEq Tok]
    (stepRef : S → Tok → S × Tok)
    (oracle : S → Tok → Nat → List Tok)
    (s : S) (tok : Tok) : Nat → List Tok
  | 0 => []
  | rem + 1 =>
    let cs := oracle s tok (rem + 1)
    let res := verifyChunk stepRef s tok (rem + 1) cs
    res.2.2.2 ++ exactScrubDecode stepRef oracle res.2.1 res.2.2.1 res.1
termination_by rem => rem
decreasing_by
  exact verifyChunk_lt stepRef s tok rem cs

/-- 9) Exactness of Window-Rollback + Tail-Scrubbing (`R4 q8_exact_scrub`):
    For ANY arbitrary draft oracle `oracle` (even adversarial), verifying proposed
    draft chunks via `verifyChunk` (with rollback to the first discrepancy and terminal
    verification before completion) produces the EXACT `rolloutRef` token sequence. -/
theorem rollback_plus_tail_scrub_exact
    {S Tok : Type} [DecidableEq Tok]
    (stepRef : S → Tok → S × Tok)
    (oracle : S → Tok → Nat → List Tok)
    (s : S) (tok : Tok) (rem : Nat) :
    exactScrubDecode stepRef oracle s tok rem = rolloutRef stepRef s tok rem := by
  induction rem using Nat.strongRecOn generalizing s tok with
  | ind n ih =>
    cases n with
    | zero =>
      rw [exactScrubDecode]
      rfl
    | succ m =>
      rw [exactScrubDecode]
      have hlt : (verifyChunk stepRef s tok (m + 1) (oracle s tok (m + 1))).1 < m + 1 :=
        verifyChunk_lt stepRef s tok m (oracle s tok (m + 1))
      rw [ih _ hlt]
      exact verifyChunk_exact stepRef s tok (m + 1) (oracle s tok (m + 1))

def supFailRank (bad : Nat → Bool) : Nat → Nat
  | 0 => 0
  | g + 1 => if bad (g + 1) then g + 1 else supFailRank bad g

theorem supFailRank_spec (bad : Nat → Bool) (G j : Nat) :
    supFailRank bad G < j → j ≤ G → bad j = false := by
  induction G with
  | zero =>
    intro h1 h2
    omega
  | succ m ih =>
    intro hlt hle
    by_cases hbad : bad (m + 1) = true
    · simp only [supFailRank, hbad, ↓reduceIte] at hlt
      omega
    · have hf : bad (m + 1) = false := by cases h : bad (m + 1) <;> simp_all
      simp only [supFailRank, hf] at hlt
      have hcases : j = m + 1 ∨ j ≤ m := by omega
      rcases hcases with rfl | hjm
      · exact hf
      · exact ih hlt hjm

/-- 10) Monotonicity-Free Supremum-Rank Sequence Conformal Soundness:
    Even when `bad` is non-monotone (Fallback Inversion), if exact fallback cap `G + 1`
    satisfies `bad (G + 1) = false` and `supFailRank bad G ≤ k`, then deploying
    `deployRank := if k < G then k + 1 else G + 1` satisfies `bad deployRank = false`. -/
theorem supremum_rank_conformal_no_mismatch
    (bad : Nat → Bool) (G k : Nat)
    (hcap : bad (G + 1) = false)
    (hle : supFailRank bad G ≤ k) :
    bad (if k < G then k + 1 else G + 1) = false := by
  split
  · next hlt =>
    exact supFailRank_spec bad G (k + 1) (by omega) (by omega)
  · exact hcap

end Sounio.Conformal

#print axioms Sounio.Conformal.fallback_inversion_witness
#print axioms Sounio.Conformal.scrub_no_fallback_first_divergence
#print axioms Sounio.Conformal.rollback_plus_tail_scrub_exact
#print axioms Sounio.Conformal.supremum_rank_conformal_no_mismatch

/-!
## Tile-Pruned All-Vocabulary Certification (Theorems 11–12)
-/

namespace Sounio.Conformal

/-- 11) Soundness of `O(1)` per-tile upper bound:
    if a tile upper-bounds `zq j ≤ z_max` and `r j ≤ r_max`, then `zq j + r j ≤ z_max + r_max`. -/
theorem tile_upper_bound_sound
    {zq_j z_max r_j r_max : Int}
    (hz : zq_j ≤ z_max)
    (hr : r_j ≤ r_max) :
    zq_j + r_j ≤ z_max + r_max := by
  omega

/-- 12) Exact equivalence between unpruned all-vocabulary certification (`GlobalCertified`)
    and tile-pruned certification: if every pruned row satisfies `zq j + r j < zq top - r top`
    (discharged in `O(1)` per tile via `tile_v1 + delta * tile_smax < lower`), checking only
    unpruned rows is necessary and sufficient for `GlobalCertified V top zq r`. -/
theorem tile_pruned_cert_iff_global_cert
    {V top : Nat} {zq r : Nat → Int} (pruned : Nat → Bool)
    (_htop : top < V)
    (_htop_unpruned : pruned top = false)
    (hprune_sound : ∀ j, j < V → pruned j = true → zq j + r j < zq top - r top) :
    GlobalCertified V top zq r ↔
      (top < V ∧ ∀ j, j < V → j ≠ top → pruned j = false → zq j + r j < zq top - r top) := by
  constructor
  · intro hc
    obtain ⟨ht, hclear⟩ := hc
    refine ⟨ht, ?_⟩
    intro j hj hne _
    exact hclear j hj hne
  · intro ⟨ht, hunpruned⟩
    refine ⟨ht, ?_⟩
    intro j hj hne
    cases hp : pruned j
    · exact hunpruned j hj hne hp
    · exact hprune_sound j hj hp

/-- 13) Jacobian-Scaled Tile-Pruned Conformal Argmax Soundness:
    Given non-negative threshold `q`, final-norm Jacobian factor `invRms`,
    offset `cAdd`, and layer proxy `linProxy`, setting `delta := q * (invRms * (cAdd + linProxy))`
    and per-row error bound `err r := delta * s r`, if all reference logits lie inside
    the interval `zq r ± err r`, every tile-pruned row is strictly dominated by `zq top - err top`,
    and every unpruned competitor has strictly positive conformal margin, then `top`
    strictly maximizes `zb` over the entire vocabulary `0..V-1`. -/
theorem jacobian_scaled_cert_sound
    {V top : Nat} {zq zb s : Nat → Int} {q invRms cAdd linProxy : Int}
    (pruned : Nat → Bool)
    (htop : top < V)
    (htop_unpruned : pruned top = false)
    (hs : ∀ r, r < V →
      let delta := q * (invRms * (cAdd + linProxy))
      let err := delta * s r
      zq r - err ≤ zb r ∧ zb r ≤ zq r + err)
    (hprune_sound : ∀ r, r < V → pruned r = true →
      let delta := q * (invRms * (cAdd + linProxy))
      zq r + delta * s r < zq top - delta * s top)
    (hunpruned_margin : ∀ r, r < V → r ≠ top → pruned r = false →
      let delta := q * (invRms * (cAdd + linProxy))
      0 < (zq top - delta * s top) - (zq r + delta * s r)) :
    ∀ r, r < V → r ≠ top → zb r < zb top := by
  let delta : Int := q * (invRms * (cAdd + linProxy))
  let err : Nat → Int := fun r => delta * s r
  have hglobal : GlobalCertified V top zq err := by
    apply (tile_pruned_cert_iff_global_cert (V := V) (top := top) (zq := zq) (r := err)
      pruned htop htop_unpruned hprune_sound).mpr
    refine ⟨htop, ?_⟩
    intro r hr hne hnp
    have hm := hunpruned_margin r hr hne hnp
    dsimp only [err, delta] at hm ⊢
    omega
  exact all_vocab_envelope_no_flip hs hglobal

end Sounio.Conformal

#print axioms Sounio.Conformal.tile_upper_bound_sound
#print axioms Sounio.Conformal.tile_pruned_cert_iff_global_cert
#print axioms Sounio.Conformal.jacobian_scaled_cert_sound
