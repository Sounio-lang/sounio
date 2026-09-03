# Formal blueprint: closing the CD-tower converse (offSeam ⟹ hasXorAnnih ∀n)

All facts below are numerically verified (k ≤ 7). The recursive proof-checker in
`cd_tower_converse_counting.py`-adjacent exploration produces a valid non-exceptional
disagreeing witness for every pair via these rules; the induction is well-founded (strong
induction on `k`, each rule either terminates or recurses to `k-1`).

Notation (in `SounioCDConverse`, `cdSigma` = `SounioCDTowerSeam.cdSigma`):
- `fVal l m a k = cdSigma l a k * cdSigma m a k`.
- `P l m a k = fVal l m a k * fVal l m (a ^^^ (l^^^m)) k`  (constant on the orbit `{a, a⊕d}`, `d=l⊕m`).
- An orbit `{a,a⊕d}` is **non-exceptional** iff `a ∉ {0, d, l, m}` (⟹ `a⊕d ∉ {0,d,l,m}` too).

## Target theorem Q (strong induction on k)
```
theorem Q : ∀ k, 3 ≤ k → ∀ l m,
    1 ≤ l → l < 2^k → 1 ≤ m → m < 2^k → l ≠ m →
    ∃ a, a < 2^k ∧ a ≠ 0 ∧ a ≠ (l ^^^ m) ∧ a ≠ l ∧ a ≠ m ∧
         fVal l m a k * fVal l m (a ^^^ (l ^^^ m)) k = -1
```

## Base: k = 3 (octonions)  — `native_decide` (ONLY place a native axiom is allowed)
`Q 3` is a finite `∀ l m < 8, ∃ a < 8, …` — prove the whole statement at k=3 by `native_decide`
(it exhibits the witness). Every octonion orbit disagrees, so a non-exceptional one always exists.

## Step lemmas (all ∀n, clean `[propext, Quot.sound]`; reuse existing branch lemmas)
Let `H = 2^k` at the induction level `k+1` (so the pair lives in `A_{k+1}`, `H = 2^{(k+1)-1}`).

**L1 `fVal_high_stable` (both-high low-block stability):** for `l_lo, m_lo, a < 2^k`,
`fVal (2^k + l_lo) (2^k + m_lo) a (k+1) = fVal l_lo m_lo a k`.
Proof: case `a = 0` (both sides 1, via `cdSigma_zero_right`); `a ≠ 0` — `cdSigma_hi_lo` on each factor
gives `(-cdSigma l_lo a k)(-cdSigma m_lo a k) = fVal l_lo m_lo a k`.

**L2 `edge_m_eq_H` (pair `(l, 2^k)`):** for `1 ≤ l < 2^k`, `1 ≤ a < 2^k`, `a ≠ l`:
`P l (2^k) a (k+1) = -1`.
Proof chain (verified): low `a`: `fVal l (2^k) a (k+1) = cdSigma l a k * (-1)` (2nd factor `cdSigma (2^k) a (k+1)
= -cdSigma 0 a k = -1` via `cdSigma_hi_lo` uL=0 + `cdSigma_zero_left`) `= -cdSigma l a k` (1st via
`cdSigma_stable`). Partner `a⊕d = 2^k + (a⊕l)` (high). `fVal l (2^k) (2^k+(a⊕l)) (k+1)
= cdSigma (a⊕l) l k` (via `cdSigma_lo_hi` for the l-factor, `cdSigma_hi_hi` uL=0 for the 2^k-factor `= 1`).
So `P = (-cdSigma l a k)·cdSigma (a⊕l) l k = -cdSigma l a k · (-cdSigma l (a⊕l) k)` [`cdAntisym_all` on
`cdSigma (a⊕l) l = -cdSigma l (a⊕l)`, needs `a⊕l ≠ l` i.e. `a≠0`, and `≠0` i.e. `a≠l`] `= cdSigma l a k ·
cdSigma l (l⊕a) k = -1` [`cdSigma_cocycle` i=l≠0].

**L3 `edge_m_eq_H_plus_l` (pair `(l, 2^k+l)`):** for `1 ≤ l < 2^k`, `1 ≤ a < 2^k`:
`P l (2^k+l) a (k+1) = -1`.
Proof: low `a`: `fVal l (2^k+l) a (k+1) = cdSigma l a k · (-cdSigma l a k) = -1` [`cdSigma_stable` +
`cdSigma_hi_lo` uL=l]. Partner `a⊕d = 2^k + a` (d = `l⊕(2^k+l) = 2^k`). `fVal l (2^k+l) (2^k+a) (k+1)
= cdSigma a l k · cdSigma a l k = 1` [`cdSigma_lo_hi` + `cdSigma_hi_hi` uL=l,bL=a]. So `P = (-1)·1 = -1`.

**L4 `edge_l_eq_H` (pair `(2^k, 2^k+m_lo)`):** for `1 ≤ m_lo < 2^k`, `1 ≤ a < 2^k`, `a ≠ m_lo`:
`P (2^k) (2^k+m_lo) a (k+1) = -1`.
Proof: `d = m_lo` (low). low `a`: `fVal (2^k)(2^k+m_lo) a (k+1) = (-1)·(-cdSigma m_lo a k) = cdSigma m_lo a k`
[`cdSigma_hi_lo` uL=0 gives -1; uL=m_lo gives -cdSigma m_lo a k]. Partner `a⊕m_lo` (low). Same form:
`fVal … (a⊕m_lo) (k+1) = cdSigma m_lo (a⊕m_lo) k`. So `P = cdSigma m_lo a k · cdSigma m_lo (m_lo⊕a) k
= -1` [`cdSigma_cocycle` i=m_lo≠0]. (Need `a ≠ m_lo` so partner `a⊕m_lo ≠ 0`, keeping it a real orbit /
`hi_lo` applicable; and `a ≠ 0`.)

## Inductive step (k+1, given `Q k`), case split — EXACTLY mirror the checker; prove exhaustive
Pair `(l,m)`, `1 ≤ l < m < 2^(k+1)`, `H = 2^k`. Let `hi_l = l ≥ H`, `hi_m = m ≥ H`, `l_lo=l%H`, `m_lo=m%H`.
Since `l < m`, `hi_l → hi_m`. Cases:
1. **`m = H`** (edge): witness by L2. Exhibit explicit non-excep low `a`: `a := if l = 1 then 2 else 1`
   (both `< H` since `H ≥ 8`; `≠ 0`; `≠ l`; `< H ≤ m`; `≠ m`; and `≠ d = 2^k+l` since `a < H`). L2 gives P=-1.
2. **`m = H + l`** (edge, `hi_m`, `¬hi_l`, `m_lo = l`): witness by L3. Explicit `a := if l = 1 then 2 else 1`
   (non-excep: `≠0`; `≠ l`; `< H` so `≠ H = d`... d here is `2^k`; `≠ l, ≠ m=2^k+l`). L3 gives P=-1.
3. **`l = H`** (edge, both-high, `l_lo = 0`): witness by L4. Explicit `a := if m_lo = 1 then 2 else 1`
   (non-excep: `≠ 0, ≠ m_lo=d, ≠ H=l, ≠ 2^k+m_lo=m`). L4 gives P=-1.
4. **`¬hi_m`** (both-low, `m < H`): `(l,m)` valid at `k`; `Q k` gives witness `a` (`a<H`, `a∉{0,d,l,m}`,
   `P l m a k = -1`). Lift: `P l m a (k+1) = P l m a k` by `P_stable_low` (needs `a, a⊕d < H` — true).
   Same `a` works: non-excep unchanged (`{0,d,l,m}` identical), `a < H < 2^(k+1)`.
5. **`hi_m ∧ ¬hi_l ∧ m_lo ∉ {0,l}`** (mixed non-edge): witness `a := m_lo` by `mixed_witness_disagree`
   (already proved: `P l (2^k+m_lo) m_lo (k+1) = -1`). Non-excep: `m_lo ≠ 0` (given), `m_lo ≠ l` (given,
   this is the `m_lo∉{0,l}` case), `m_lo ≠ d`? `d = 2^k+(l⊕m_lo)`, `m_lo<H` so `≠ d`; `m_lo ≠ m = 2^k+m_lo`.
   (m=H is case 1, m=H+l is case 2, so here `m_lo∉{0,l}` and `hi_m,¬hi_l` — consistent.)
6. **`hi_m ∧ hi_l ∧ l ≠ H`** (both-high, `l_lo ≥ 1`): `l_lo, m_lo ∈ [1,H)`, `l_lo ≠ m_lo` (from `l≠m`).
   `Q k` on `(l_lo, m_lo)` gives witness `a` (`a<H`, `a∉{0, l_lo⊕m_lo, l_lo, m_lo}`, `P l_lo m_lo a k=-1`).
   Note `d = l⊕m = l_lo⊕m_lo` (top bits cancel). Lift: `P l m a (k+1) = P l_lo m_lo a k` by
   `fVal_high_stable` (L1) applied to both `fVal` factors (`a, a⊕d < H`). Non-excep at `k+1`:
   `a < H` ⟹ `a ≠ l, m` (both `≥ H`); `a ≠ 0, ≠ d` from the `k` non-exceptionality. Same `a` works.

**Exhaustiveness:** every pair falls in exactly one of 1–6. Prove by `omega`/`decide` on the bit
predicates: given `l < m < 2^(k+1)` and `1 ≤ l`, the split on (`m<H`? / `l≥H`? / `m_lo∈{0,l}`?) is total.
Cases 1,2,3 catch the seam-element pairs (`m=H`, `m=H+l`, `l=H`) BEFORE cases 5,6 so their side
conditions (`m_lo∉{0,l}` in 5; `l≠H` in 6) hold.

## Connection: `converse_holds` (the payoff)
```
theorem converse_holds (bits l u : Nat)
    (hl1 : 1 ≤ l) (hl : l < 2^(bits-1)) (hu1 : 2^(bits-1) ≤ u) (hu : u < 2^bits)
    (hoff : offSeam bits l u = true) :
    hasXorAnnih bits l u = true
```
Let `n := bits-1`, `u_lo := u - 2^n` (so `u = 2^n + u_lo`, `u_lo < 2^n`). offSeam ⟹ `u_lo ≠ 0 ∧ u_lo ≠ l`.
Then `(l, u_lo)` is distinct-nonzero in `A_n`; apply `Q n` (need `n ≥ 3`, i.e. `bits ≥ 4`) to get a
non-exceptional disagreeing witness `a` for `(l, u_lo)`: `a < 2^n`, `a ∉ {0, l⊕u_lo, l, u_lo}`,
`P l u_lo a n = -1`. This `a` satisfies `converse_recursion'`'s hypotheses (`1≤a`, `a<2^n`, `a≠l`,
`a≠u_lo`, `a≠l⊕u_lo`), so `P l u a bits = - P l u_lo a n = +1`. Finally `a` witnesses `hasXorAnnih`
(`a ≥ 1` from `a≠0`; `a ≠ l⊕u` since `a < 2^n ≤ l⊕u`; the four-sign product `= +1`). Provide the
`List.any` witness. (For `bits ≤ 3` the loHi/offSeam locus is empty or octonion — handle `bits<4` by the
hypotheses being unsatisfiable / a small `decide`.)

## Induction is ORDINARY, not strong
Cases 4 (both-low) and 6 (both-high) both step down to **exactly** the IH level (k-1 in blueprint
terms). So prove `∀ k, Qstmt (k+3)` by plain `Nat.rec` on the offset: base `Qstmt 3`, step
`Qstmt (k+3) → Qstmt (k+3+1)`. No `Nat.strongRecOn` needed. Width map: level `k+3 = W`, step level
`W+1 = (n+1)+1 = n+2` with `n = k+1`, so `H = 2^(n+1) = 2^(k+3)` matches the L1–L4 lemma idiom
(width `n+2`, `H = 2^(n+1)`).

## Existing exact signatures (build on these; `cdSigma` = `SounioCDTowerSeam.cdSigma`)
- `cdSigma_stable (n a b) (ha:a<2^(n+1)) (hb:b<2^(n+1)) : cdSigma a b (n+2) = cdSigma a b (n+1)`
- `cdSigma_hi_lo (n uL a) (huL:uL<2^(n+1)) (ha1:1≤a) (ha:a<2^(n+1)) : cdSigma (2^(n+1)+uL) a (n+2) = -cdSigma uL a (n+1)`
- `cdSigma_lo_hi (n bL a) (hbL:bL<2^(n+1)) (ha1:1≤a) (ha:a<2^(n+1)) : cdSigma a (2^(n+1)+bL) (n+2) = cdSigma bL a (n+1)`
- `cdSigma_hi_hi (n uL bL) (huL:uL<2^(n+1)) (hb1:1≤bL) (hbL:bL<2^(n+1)) : cdSigma (2^(n+1)+uL) (2^(n+1)+bL) (n+2) = cdSigma bL uL (n+1)`
- `cdSigma_zero_right (bits x) (1≤bits) : cdSigma x 0 bits = 1`  /  `cdSigma_zero_left (bits x) (1≤bits) : cdSigma 0 x bits = 1`
- `cdSigma_diag (k x) (1≤x) (x<2^k) : cdSigma x x k = -1`
- `cdAntisym_all (m) : ∀ x y, 1≤x → x<2^m → 1≤y → y<2^m → x≠y → cdSigma x y m = -cdSigma y x m`
- `cdSigma_pm (bits a b) : cdSigma a b bits = 1 ∨ cdSigma a b bits = -1`
- `two_pow_xor_eq_add (k z) (z<2^k) : 2^k ^^^ z = 2^k + z`  (note `l ^^^ 2^k = 2^k ^^^ l` via `Nat.xor_comm`)
- `orbit_low_to_high (k a dL) (a<2^k) (dL<2^k) : a ^^^ (2^k + dL) = 2^k + (a ^^^ dL)`
- `xor_eq_zero_of (a b) : a ^^^ b = 0 → a = b`  /  `Nat.xor_lt_two_pow`
- `converse_recursion' (n l uL a) (1≤l)(l<2^(n+1))(1≤uL)(uL<2^(n+1))(1≤a)(a<2^(n+1))(a≠l)(a≠uL)(a≠l^^^uL) : fVal l (2^(n+1)+uL) a (n+2) * fVal l (2^(n+1)+uL) (a^^^(l^^^(2^(n+1)+uL))) (n+2) = -(fVal l uL a (n+1) * fVal l uL (a^^^(l^^^uL)) (n+1))`
- base `oct_all_disagree` (native_decide), `mixed_witness_disagree` (case 5), `P_stable_low` (case 4)
- `hasXorAnnih (bits l u) : Bool := (range (2^bits)).any (fun a => a≥1 && a≠(l^^^u) && (cdSigma l a bits * cdSigma u a bits * cdSigma l (a^^^(l^^^u)) bits * cdSigma u (a^^^(l^^^u)) bits == 1))`
- `offSeam (bits l u) = ! (u == 2^(bits-1) || (l^^^u) == 2^(bits-1))`  (in `SounioCDTowerSeam`)

## FIRST add a bridged cocycle (TowerSeam sign):
```
theorem cdSigma_cocycle' (n i j : Nat) (hi : i < 2^n) (hj : j < 2^n) (hi0 : i ≠ 0) :
    cdSigma i j n * cdSigma i (i ^^^ j) n = -1 := by
  rw [← cdSigma_defeq n i j, ← cdSigma_defeq n i (i ^^^ j)]
  exact SounioCDCocycle.cdSigma_cocycle n i j hi hj hi0
```

## Mathlib-free tactic idioms USED IN THIS FILE (do not reach for `ring`/`simp_all`/`interval_cases`)
- Sign arithmetic: `rcases cdSigma_pm .. with h|h <;> rw [h] <;> decide` (see `mixed_witness_disagree`).
- Two signs tied by a hypothesis `hAB : A*B = -1`: `rcases cdSigma_pm .. with hA|hA <;> rcases
  cdSigma_pm .. with hB|hB <;> rw [hA, hB] at hAB ⊢ <;> first | decide | exact absurd hAB (by decide)`
  (the `A*B=+1` cases contradict `hAB`).
- `2^(n+1)` as `2^(n+1)+0` for the `uL=0` edge: `have := cdSigma_hi_lo n 0 a (Nat.two_pow_pos (n+1)) ha1 ha;
  rw [Nat.add_zero] at this` then `this : cdSigma (2^(n+1)) a (n+2) = -cdSigma 0 a (n+1)`; finish with
  `cdSigma_zero_left`. Likewise `cdSigma_hi_hi n 0 .. ` + `Nat.add_zero` + `cdSigma_zero_right` for the
  both-high `uL=0` factor.
- Guards: `eq_of_beq`, `if_pos`/`if_neg`, `Nat.mod_eq_of_lt`. **Avoid `simp [cdSigma]` and `simp at h`**
  — they leak `Classical.choice` (verify with `#print axioms`). Length/bit reasoning: `omega`.
- Base `Qstmt 3`: bridge from a decidable Bool (`native_decide`) over `List.range 8`, extract the
  witness with `List.any_eq_true` (core) → `∃ a, a ∈ range ∧ …`, then `List.mem_range`,
  `Bool.and_eq_true`, `beq_iff_eq`, `bne_iff_ne` to convert Bool conds ↔ the `∃ a` Prop conjunction.
  Consider defining `Qstmt` around a Bool witness `hasNEwitness (k l m) : Bool` so base = one `decide`
  and the witness extraction is uniform with the connection step.

## Guardrails
- `#print axioms converse_holds` MUST be `[propext, Quot.sound]` + ONLY the k=3 base's native anchor
  (`Lean.ofReduceBool` / `Lean.trustCompiler`-style). Any `sorry` / extra native axiom in the inductive
  step = the empirical fact was smuggled, not proved. Same check on `Q_all`.
- Mathlib-free; no `native_decide` outside the k=3 base. `ulimit -v unlimited` before every `lake build`
  (else `pthread_create` crashes); `export PATH="$HOME/.elan/bin:$PATH"`.
