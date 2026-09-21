/-!
Arithmetic code for the unsigned support recursion.
A word records successive ordinary/complemented clique doublings.
This file proves uniqueness, exact decoding, and exactly 2^depth distinct
integer codes at every fixed depth, using Lean core only.
It does NOT formalize the CD graph-to-recursion bridge, determinant/rank
theorem, or bibliographic novelty.
-/
namespace Sounio.ZDUnsignedCode

def parentOrder (depth : Nat) : Int := (2 : Int)^(depth+2) - 1

def parentCapacity (depth : Nat) : Int :=
  ((2 : Int)^(depth+2) - 1) * ((2 : Int)^(depth+1) - 1)

def step (depth : Nat) (complemented : Bool) (edges : Int) : Int :=
  3 * parentOrder depth +
    4 * (if complemented then parentCapacity depth - edges else edges)

/-- Head is the newest (outermost) operation; the empty code is K3. -/
def edgeCode : List Bool → Int
  | [] => 3
  | b :: bs => step bs.length b (edgeCode bs)

theorem pow_two_even (k : Nat) : (2 : Int)^(k+1) % 2 = 0 := by
  rw [Int.pow_succ]
  omega

theorem parentOrder_odd (depth : Nat) : parentOrder depth % 2 = 1 := by
  have h : (2 : Int)^(depth+2) % 2 = 0 := by
    simpa only [Nat.add_assoc] using pow_two_even (depth+1)
  unfold parentOrder
  omega

theorem parentCapacity_odd (depth : Nat) : parentCapacity depth % 2 = 1 := by
  have h1 : (2 : Int)^(depth+2) % 2 = 0 := by
    simpa only [Nat.add_assoc] using pow_two_even (depth+1)
  have h2 := pow_two_even depth
  have hl : ((2 : Int)^(depth+2) - 1) % 2 = 1 := by omega
  have hr : ((2 : Int)^(depth+1) - 1) % 2 = 1 := by omega
  unfold parentCapacity
  rw [Int.mul_emod, hl, hr]
  decide

/-- Twice the parent capacity is q(q-1), the simple-graph edge-capacity identity. -/
theorem parentCapacity_identity (depth : Nat) :
    2 * parentCapacity depth = parentOrder depth * (parentOrder depth - 1) := by
  unfold parentCapacity parentOrder
  rw [Int.mul_left_comm]
  congr 1
  have hp : (2 : Int)^(depth+2) = (2 : Int)^(depth+1) * 2 := by
    simpa only [Nat.add_assoc] using (Int.pow_succ (2 : Int) (depth+1))
  rw [hp]
  omega

theorem edgeCode_odd (bs : List Bool) : edgeCode bs % 2 = 1 := by
  cases bs with
  | nil => decide
  | cons b bs =>
    have h := parentOrder_odd bs.length
    cases b <;> simp only [edgeCode, step, Bool.false_eq_true, ↓reduceIte] <;> omega

theorem step_injective (depth : Nat) (b : Bool) (x y : Int)
    (h : step depth b x = step depth b y) : x = y := by
  cases b <;> simp only [step, Bool.false_eq_true, ↓reduceIte] at h <;> omega

theorem step_disjoint (depth : Nat) (x y : Int)
    (hx : x % 2 = 1) (hy : y % 2 = 1) :
    step depth false x ≠ step depth true y := by
  have hc := parentCapacity_odd depth
  simp only [step, Bool.false_eq_true, ↓reduceIte]
  omega

/-- Equal-depth words with the same code are identical. -/
theorem edgeCode_injective :
    ∀ (xs ys : List Bool), xs.length = ys.length →
      edgeCode xs = edgeCode ys → xs = ys := by
  intro xs
  induction xs with
  | nil =>
    intro ys hlen _
    cases ys with
    | nil => rfl
    | cons b bs => simp at hlen
  | cons a xs ih =>
    intro ys hlen he
    cases ys with
    | nil => simp at hlen
    | cons b ys =>
      have hlen' : xs.length = ys.length := by simpa using hlen
      have he' : step xs.length a (edgeCode xs) = step xs.length b (edgeCode ys) := by
        simpa only [edgeCode, ← hlen'] using he
      have hx := edgeCode_odd xs
      have hy := edgeCode_odd ys
      cases a <;> cases b
      · have h := step_injective xs.length false _ _ he'
        exact congrArg (List.cons false) (ih ys hlen' h)
      · exact False.elim (step_disjoint xs.length _ _ hx hy he')
      · exact False.elim (step_disjoint xs.length _ _ hy hx he'.symm)
      · have h := step_injective xs.length true _ _ he'
        exact congrArg (List.cons true) (ih ys hlen' h)

/-- The parity after removing the hub and twin contributions identifies the branch. -/
def outerBranch (depth : Nat) (edges : Int) : Bool :=
  ((edges - 3 * parentOrder depth) / 4) % 2 == 0

def undoStep (depth : Nat) (b : Bool) (edges : Int) : Int :=
  let residual := (edges - 3 * parentOrder depth) / 4
  if b then parentCapacity depth - residual else residual

theorem step_quotient (depth : Nat) (b : Bool) (x : Int) :
    (step depth b x - 3 * parentOrder depth) / 4 =
      (if b then parentCapacity depth - x else x) := by
  cases b <;> simp only [step, Bool.false_eq_true, ↓reduceIte] <;> omega

theorem outerBranch_step (depth : Nat) (b : Bool) (x : Int)
    (hx : x % 2 = 1) : outerBranch depth (step depth b x) = b := by
  have hc := parentCapacity_odd depth
  unfold outerBranch
  rw [step_quotient]
  cases b <;> simp only [Bool.false_eq_true, ↓reduceIte]
  · simp [hx]
  · have h : (parentCapacity depth - x) % 2 = 0 := by omega
    simp [h]

theorem undoStep_step (depth : Nat) (b : Bool) (x : Int) :
    undoStep depth b (step depth b x) = x := by
  unfold undoStep
  rw [step_quotient]
  cases b <;> simp only [Bool.false_eq_true, ↓reduceIte] <;> omega

/-- Total decoder; correctness is asserted on codes in the image at this depth. -/
def decode : Nat → Int → List Bool
  | 0, _ => []
  | depth+1, edges =>
      let b := outerBranch depth edges
      b :: decode depth (undoStep depth b edges)

/-- Exact left inverse for every depth; no finite enumeration is used. -/
theorem decode_edgeCode (bs : List Bool) : decode bs.length (edgeCode bs) = bs := by
  induction bs with
  | nil => rfl
  | cons b bs ih =>
    simp only [List.length_cons, edgeCode, decode]
    rw [outerBranch_step _ _ _ (edgeCode_odd bs), undoStep_step, ih]

/-- The decoder also supplies a second derivation of injectivity. -/
theorem edgeCode_injective_via_decode (xs ys : List Bool)
    (hlen : xs.length = ys.length) (he : edgeCode xs = edgeCode ys) : xs = ys := by
  calc
    xs = decode xs.length (edgeCode xs) := (decode_edgeCode xs).symm
    _ = decode ys.length (edgeCode ys) := by rw [hlen, he]
    _ = ys := decode_edgeCode ys

/-- All operation words at a fixed depth, with no quotienting. -/
def words : Nat → List (List Bool)
  | 0 => [[]]
  | depth+1 => (words depth).map (List.cons false) ++
      (words depth).map (List.cons true)

theorem mem_words_iff (depth : Nat) (bs : List Bool) :
    bs ∈ words depth ↔ bs.length = depth := by
  induction depth generalizing bs with
  | zero => simp [words]
  | succ depth ih =>
    cases bs with
    | nil => simp [words]
    | cons b bs => cases b <;> simp [words, List.mem_map, ih]

theorem words_length (depth : Nat) : (words depth).length = 2^depth := by
  induction depth with
  | zero => rfl
  | succ depth ih =>
    simp only [words, List.length_append, List.length_map, ih, Nat.pow_succ]
    omega

theorem words_nodup (depth : Nat) : (words depth).Nodup := by
  induction depth with
  | zero => simp [words]
  | succ depth ih =>
    have hm (b : Bool) : ((words depth).map (List.cons b)).Nodup :=
      List.Pairwise.map (List.cons b)
        (fun _ _ hne he => hne (List.cons.inj he).2) ih
    rw [words, List.nodup_append]
    refine ⟨hm false, hm true, ?_⟩
    intro a ha b hb he
    obtain ⟨a', _, rfl⟩ := List.mem_map.mp ha
    obtain ⟨b', _, rfl⟩ := List.mem_map.mp hb
    cases he

def codes (depth : Nat) : List Int := (words depth).map edgeCode

/-- This enumerates exactly all codes of words at this depth. -/
theorem mem_codes_iff (depth : Nat) (e : Int) :
    e ∈ codes depth ↔ ∃ bs, bs.length = depth ∧ edgeCode bs = e := by
  simp only [codes, List.mem_map, mem_words_iff]

theorem codes_length (depth : Nat) : (codes depth).length = 2^depth := by
  simp only [codes, List.length_map, words_length]

/-- Together with codes_length, this is an exact distinct-code count for every depth. -/
theorem codes_nodup (depth : Nat) : (codes depth).Nodup := by
  unfold codes List.Nodup
  rw [List.pairwise_map]
  apply List.Pairwise.imp_of_mem _ (words_nodup depth)
  intro a b ha hb hne he
  exact hne (edgeCode_injective a b
    ((mem_words_iff depth a).mp ha |>.trans ((mem_words_iff depth b).mp hb).symm) he)

theorem first_split : edgeCode [false] = 21 ∧ edgeCode [true] = 9 := by decide

theorem second_split :
    edgeCode [false,false] = 105 ∧ edgeCode [false,true] = 57 ∧
    edgeCode [true,false] = 21 ∧ edgeCode [true,true] = 69 := by decide

/-- Depth must be fixed: a valid code can recur at different depths. -/
theorem depth_required :
    edgeCode [false] = edgeCode [true,false] ∧
    ([false] : List Bool).length ≠ [true,false].length := by decide

end Sounio.ZDUnsignedCode
