# Blueprint: close `isZD ⟹ hasXorAnnih` ∀n (Target 1), and `offSeam ⟺ anti0` ∀n (Target 2)

All facts below are empirically verified on loHi at n=4..8 (Target 1) / n=4..7 (Target 2). Mathlib-free,
kernel axioms `[propext, Quot.sound]` (the only legit `native_decide` is the k=3 base already inside
`converse_holds`, inherited — no NEW `native_decide`). `cdSigma` = `SounioCDTowerSeam.cdSigma`.
Env: `cd /workspace/sounio-exact-algebra; export PATH="$HOME/.elan/bin:$PATH"; ulimit -v unlimited;
cd formal/lean4 && lake build SounioCDConverse`.

═══════════════════════════════════════════════════════════════════════════════════════════════════
# TARGET 1 — `isZD ⟹ hasXorAnnih` on loHi, ∀ bits≥4  (SHIP THIS FIRST; independent of Target 2)
═══════════════════════════════════════════════════════════════════════════════════════════════════

## (A) Reduction extraction `annih_forces` — the reverse of the existing `annih_of`
```
theorem annih_forces (bits l u a b : Nat) (s : Int)
    (hl : l < 2^bits) (hu : u < 2^bits) (ha : a < 2^bits) (hne : l ≠ u) (hab : a ≠ b)
    (hh : annih bits l u a b s = true) :
    b = a ^^^ (l ^^^ u) ∧
    cdSigma l a bits * cdSigma u a bits * cdSigma l b bits * cdSigma u b bits = 1
```
Proof. `unfold annih at hh; rw [List.all_eq_true] at hh` → `hh : ∀ k ∈ range (2^bits), (…4-indicator sum…) == 0`.
Helper: `hk : ∀ k, k < 2^bits → (that sum) = 0` via `hh k (List.mem_range.mpr …)` + `beq_iff_eq`.

**Step 1 — `b = a ^^^ (l⊕u)`.** By contradiction: assume `b ≠ a ^^^ (l⊕u)`. Then the four source
indices `l⊕a, l⊕b, u⊕a, u⊕b` are pairwise-distinct at `k = l⊕a`:
  - `l⊕a = l⊕b ⟺ a=b` (excluded by `hab`); `l⊕a = u⊕a ⟺ l=u` (excluded); `l⊕a = u⊕b ⟺ b = a⊕(l⊕u)`
    (excluded by assumption). [Prove each `≠` with the xor-cancel idiom: `intro h; apply …; <xor_assoc/self/zero>`.]
  So at `k := l⊕a` (`< 2^bits` since `l,a < 2^bits`, `Nat.xor_lt_two_pow`): term1 `= cdSigma l a` (the
  `l⊕a==k` guard is `rfl`-true), terms 2,3,4 are `0` (their guards are the three `≠`s). So `hk (l⊕a) …`
  gives `cdSigma l a + 0 + 0 + 0 = 0`, i.e. `cdSigma l a = 0` — contradiction with `cdSigma_pm`.
  → `b = a ^^^ (l⊕u)`. Introduce `hb : b = a ^^^ (l ^^^ u)` and `subst`/rewrite.

**Step 2 — the two sign equations, then `P=1`.** With `b = a⊕(l⊕u)` the indices fold:
`u⊕b = l⊕a` and `l⊕b = u⊕a` (xor algebra). Evaluate `hk` at `k₁ := l⊕a` and `k₂ := l⊕b`:
  - at `k₁=l⊕a` (=`u⊕b`): terms 1 (`cdSigma l a`) and 4 (`s·cdSigma u b`) fire, 2&3 don't (`l⊕b=u⊕a≠l⊕a`
    since `l≠u`; `u⊕a≠l⊕a`). → `e1 : cdSigma l a + s * cdSigma u b = 0`.
  - at `k₂=l⊕b` (=`u⊕a`): terms 2 (`s·cdSigma l b`) and 3 (`cdSigma u a`) fire. → `e2 : s*cdSigma l b + cdSigma u a = 0`.
  Now the four-sign product. First `have hs : s = 1 ∨ s = -1` from `e1` (`rcases cdSigma_pm bits l a`,
  `cdSigma_pm bits u b`; `rw` into `e1`; `omega`). Then
  `rcases hs <;> rcases cdSigma_pm bits l a <;> rcases cdSigma_pm bits u a <;> rcases cdSigma_pm bits l b
   <;> rcases cdSigma_pm bits u b <;> rw […] at e1 e2 ⊢ <;> first | (revert e1 e2; decide) | omega`.
  (Impossible sign combos are killed by `e1`/`e2` becoming `±2=0`; the rest make the goal a numeric `=1`.)

## New branch lemma `cdSigma_hi_pow` (mirror `cdSigma_hi_hi`, bLo=0 sub-branch)
```
theorem cdSigma_hi_pow (n uL : Nat) (huL : uL < 2^(n+1)) :
    cdSigma (2^(n+1) + uL) (2^(n+1)) (n+2) = -1
```
Proof mirrors `cdSigma_hi_hi` (lines ~356-375) up to the guard/hi/mod rewrites, then the def's both-high
branch is `if bLo==0 then -cdSigma 0 aLo (n+1) else …`; here `bLo = 2^(n+1) % 2^(n+1) = 0`, `aLo = uL`,
so it reduces to `-cdSigma 0 uL (n+1)`, closed by `cdSigma_zero_left (n+1) uL (by omega)` → `-(1) = -1`.
(Cross-checked by `#eval`: this branch value is `-1` for all `uL`.)

## (P3) on-seam ⟹ P(0) = −1  (the a=0 corner)
```
theorem P0_neg_of_onSeam (bits l u : Nat) (hb : 2 ≤ bits)
    (hl1 : 1 ≤ l) (hl : l < 2^(bits-1)) (hu1 : 2^(bits-1) ≤ u) (hu : u < 2^bits)
    (hon : offSeam bits l u = false) :
    cdSigma l 0 bits * cdSigma u 0 bits
      * cdSigma l (l^^^u) bits * cdSigma u (l^^^u) bits = -1
```
`cdSigma l 0 = cdSigma u 0 = 1` (`cdSigma_zero_right`). So reduce to `cdSigma l d * cdSigma u d = -1`,
`d = l⊕u`. Obtain `n` with `bits = n+2`, `top = 2^(n+1) = 2^(bits-1)`. Parse `hon` (`offSeam` def +
`Bool.not_eq_false`/`Bool.or_eq_true`): `u = top ∨ (l⊕u) = top`. Two subcases (mutually exclusive on
loHi, both give the SAME product −1; verified per-case σ values below):
  - **`u = top`** (subst `u = 2^(n+1)`): `d = l⊕2^(n+1) = 2^(n+1)+l` (`two_pow_xor_eq_add`+`Nat.xor_comm`).
    `cdSigma l d = cdSigma l (2^(n+1)+l) = cdSigma_lo_hi n l l … = cdSigma l l (n+1) = -1` (`cdSigma_diag`).
    `cdSigma u d = cdSigma (2^(n+1)) (2^(n+1)+l) = cdSigma_hi_hi n 0 l … = cdSigma l 0 (n+1) = 1`
    (note `2^(n+1) = 2^(n+1)+0`, `Nat.add_zero`; needs `1≤l`). Product `(-1)(1) = -1`.
  - **`(l⊕u) = top`** i.e. `d = 2^(n+1)`, and `u ≠ top` so use `u = 2^(n+1)+l` (from `d=top`,`l<top`):
    `cdSigma l d = cdSigma l (2^(n+1)) = cdSigma l (2^(n+1)+0) = cdSigma_lo_hi n 0 l … = cdSigma 0 l (n+1) = 1`.
    `cdSigma u d = cdSigma (2^(n+1)+l) (2^(n+1)) = cdSigma_hi_pow n l … = -1`. Product `(1)(-1) = -1`.
  (Per-case σ values are `#eval`-confirmed: u=top→(−1,1), d=top→(1,−1).)

## Assembly `hasXorAnnih_complete` (Target 1 deliverable)
```
theorem hasXorAnnih_complete (bits l u : Nat) (hb : 4 ≤ bits)
    (hl1 : 1 ≤ l) (hl : l < 2^(bits-1)) (hu1 : 2^(bits-1) ≤ u) (hu : u < 2^bits)
    (hzd : isZD bits l u = true) : hasXorAnnih bits l u = true
```
`hlt : l < 2^bits` and `hult : u < 2^bits` (from bounds, `2^(bits-1) ≤ 2^bits`). `hne : l ≠ u` (`l<top≤u`).
`unfold isZD at hzd; rw [List.any_eq_true] at hzd` twice → obtain `a, b`, `mem`s, `a<b` (`decide_eq_true`),
and `annih … a b 1 = true ∨ annih … a b (-1) = true`. In each disjunct get `s` with `annih … a b s = true`.
`annih_forces` → `hb' : b = a ^^^ (l⊕u)` and `hP : cdSigma l a * cdSigma u a * cdSigma l b * cdSigma u b = 1`.
Rewrite `hP` with `hb'` to `cdSigma l a * cdSigma u a * cdSigma l (a^^^(l⊕u)) * cdSigma u (a^^^(l⊕u)) = 1`
(= the `hasXorAnnih` winner test at `a`).
  - **`a ≥ 1`**: `a ≠ l⊕u` (else `b = a⊕d = 0 < a`, contra `a<b`). Inject `a` into `hasXorAnnih`:
    `refine List.any_eq_true.mpr ⟨a, List.mem_range.mpr (by omega), ?_⟩; rw [Bool.and_eq_true,
     Bool.and_eq_true, decide_eq_true_eq, …, beq_iff_eq]; exact ⟨⟨by omega, by omega/hane⟩, hP⟩`.
  - **`a = 0`**: then `b = l⊕u`. `hP` at `a=0` is exactly `P(0) = 1`. By `P0_neg_of_onSeam`'s
    contrapositive, `offSeam bits l u = true` (if it were false, `P(0)=−1≠1`). Then
    `exact converse_holds bits l u hb hl1 hl hu1 hu hoff`.  ← reuses the already-proved converse.

## Corollary generalizing `xorAnnih_eq_isZD_16` to ∀n
```
theorem xorAnnih_eq_isZD_all (bits : Nat) (hb : 4 ≤ bits) :
    (loHi bits).all (fun p => hasXorAnnih bits p.1 p.2 == isZD bits p.1 p.2) = true
```
`rw [List.all_eq_true]; intro p hp; obtain ⟨…⟩ := loHi_mem bits (by omega) p hp`. `beq` of two Bools:
case both directions — `hasXorAnnih ⟹ isZD` = `hasXorAnnih_sound` (needs `l,u<2^bits`, `l≠u`),
`isZD ⟹ hasXorAnnih` = `hasXorAnnih_complete`. Close with `Bool.eq_iff_iff`/`decide` on the two implications.

═══════════════════════════════════════════════════════════════════════════════════════════════════
# TARGET 2 — `offSeam ⟺ anti0` on loHi, ∀ bits≥4   (REUSES the converse via the Q=P bridge)
═══════════════════════════════════════════════════════════════════════════════════════════════════
`anti0 bits l u = (range 2^bits).all (fun c => cdSigma l (u⊕c) · cdSigma u c + cdSigma u (l⊕c) · cdSigma l c == 0)`.
Write `A(c) := cdSigma l (u⊕c)·cdSigma u c`, `B(c) := cdSigma u (l⊕c)·cdSigma l c` (each `±1`). The `anti0`
term is `A(c)+B(c) == 0`, which (both `±1`) holds iff `A(c)·B(c) = -1`. Define `Q(c) := A(c)·B(c)` and
`P(c) := cdSigma l c·cdSigma u c·cdSigma l (c⊕d)·cdSigma u (c⊕d)` (d=l⊕u, the converse winner product).

## THE UNLOCK — `Q(c) = P(c)` ∀c  (verified `#eval`, all pairs l,u≥1, n=4..6)
```
theorem anti0_QP (bits l u c : Nat) (hl0 : l ≠ 0) (hl : l < 2^bits) (hu : u < 2^bits) :
    cdSigma l (u ^^^ c) bits * cdSigma u c bits * cdSigma u (l ^^^ c) bits * cdSigma l c bits
  = cdSigma l c bits * cdSigma u c bits * cdSigma l (c ^^^ (l^^^u)) bits * cdSigma u (c ^^^ (l^^^u)) bits
```
Proof: two cocycle rewrites. `cdSigma_cocycle' bits l (l⊕u⊕c)` (or arrange args) gives
`cdSigma l (u⊕c) = - cdSigma l (c ⊕ (l⊕u))` [since `l ⊕ (l⊕u⊕c) = u⊕c` and `cdSigma l x · cdSigma l (l⊕x) = -1`
⟹ `cdSigma l (l⊕x) = - cdSigma l x`; pick `x` so `l⊕x = u⊕c` and `x = c⊕(l⊕u)`]. Likewise
`cdSigma u (l⊕c) = - cdSigma u (c⊕(l⊕u))`. The two `(-1)`s cancel. Needs `l,u ≠ 0` (cocycle `i≠0`).
(Regrouping: LHS = `[cdSigma l (u⊕c)·cdSigma l c]·[cdSigma u c·cdSigma u (l⊕c)]`; each cocycle rewrite
turns a cross-factor into the converse form.)

## Target: `theorem seam_eq_anti0 (bits l u) (hb:4≤bits) (loHi bounds) : anti0 bits l u = ! offSeam bits l u`
Bridge lemma first: `anti0 bits l u = true ↔ ∀ c < 2^bits, P(c) = -1`, via `anti0_QP` + `A+B=0 ↔ A·B=-1`
(the `A,B=±1` step: `rcases cdSigma_pm` on the four factors, `omega`/`decide`). Then two directions:

  - **`offSeam ⟹ ¬anti0`** (⟹ `anti0 = false`): `converse_holds` gives `hasXorAnnih = true`, i.e. a
    witness `a` with `a≥1, a≠d, P(a) = +1` (unfold `hasXorAnnih`, `any_eq_true`). Via the bridge, `P(a)=+1`
    contradicts `∀c P(c)=-1`, so `anti0 = false`. **Reuses `converse_holds` directly** — no new existence
    argument. (`Bool`: show `anti0 = false` by exhibiting `c=a` where the `all` predicate fails.)
  - **`¬offSeam ⟹ anti0`** (on-seam ⟹ `∀c P(c) = -1`): the on-seam loHi pairs are EXACTLY the converse
    edge pairs. `P` is orbit-constant (`P(c) = P(c⊕d)`; prove `P_symm` by swapping the two `fVal` factors),
    so it suffices to hit each orbit's low rep `a < top`:
      · **`u = top`** (pair `(l, 2^k)`, `k=bits-1`): `edge_m_eq_H k l a` gives `P(a)=-1` for `1≤a<top, a≠l`.
        Corners: `a = l` (orbit `{l, top+l}`) — small lemma `P(l)=-1` via `cdSigma_diag`+branches (verified:
        `σ(l,l)σ(top,l)σ(l,top)σ(top,top) = (-1)(-1)(1)(-1) = -1`); `a = 0` (orbit `{0,top}`) — `P0_neg_of_onSeam`.
      · **`d = top`** (pair `(l, top+l)`, `u_lo=l`): `edge_m_eq_H_plus_l k l a` gives `P(a)=-1` for `1≤a<top`
        (covers `a=l` too — no exclusion). `a=0` orbit — `P0_neg_of_onSeam`.
      Every `c` maps (via `P_symm`) to a low rep `c⊕d` or `c` `< top`, covered above.  (Both edge lemmas +
      `P0_neg_of_onSeam` already proved; this direction is assembly + orbit bookkeeping, not fresh σ-work.)
If Target 2 stalls, ship Target 1 alone and report the blocking piece (likely the on-seam orbit bookkeeping).

## Guardrails
- `#print axioms hasXorAnnih_complete` / `xorAnnih_eq_isZD_all` MUST be `[propext, Quot.sound]` + at most
  the inherited k=3 base anchor (from `converse_holds`). NO new `native_decide`, NO `sorry`, NO `axiom`.
- Avoid `simp [cdSigma]`, bare `simp at h` (leak `Classical.choice`). Use `rw`/`omega`/`eq_of_beq`/`decide`.
- Reuse: `annih_of` (line 117), `converse_holds`, `loHi_mem`, `hasXorAnnih_sound`, `cdSigma_pm`,
  `cdSigma_lo_hi/hi_hi/hi_lo/stable/diag/zero_left/zero_right`, `two_pow_xor_eq_add`, `orbit_low_to_high`,
  `xor_eq_zero_of`, `cdSigma_cocycle'`, `cdAntisym_all`.
