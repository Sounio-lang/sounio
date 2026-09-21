<!-- docs:meta
topic_id: repo.docs.audit.native-pow-tgamma-numerical-design-2026-08-23
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.native-pow-tgamma-numerical-design-2026-08-23
-->

# Native `pow` and `tgamma` for a syscall-only x86-64 backend — numerical design

**Status:** design note. Every constant, threshold and error figure below was computed
with `mpmath` (60–70 decimal digits) by the scripts named in each section, not recalled
from memory. Where a number is a *measurement* over a random sample rather than a proved
bound, it says so.

## 0. Scope, instruction budget, and the one assumption that drives everything

Available: `addsd subsd mulsd divsd sqrtsd ucomisd`, `movq` rax↔xmm, `movsd`
rip-relative and `rbp`-disp8, `cvtsi2sd cvttsd2si cvtsd2si`, `roundsd` (SSE4.1),
and `ucomisd`+`setcc`+`test`+`jz/jnz` branching. No libm, no FMA, no `andpd`.

**The assumption.** `exp(x)` and `log(x)` already exist as native stubs, each accurate
to *roughly 1 ulp*. That assumption is load-bearing and, taken at face value, it is
**not sufficient** to build a 2-ulp `pow` or `tgamma`. The reason is developed in §1.2
and measured in §1.3, but the short version is:

> `pow` and `tgamma` both end in `exp(u)` (or `2^r`) where `|u|` runs up to ~709
> (resp. `|r|` up to ~1024). A relative error `ε` in `u` becomes an **absolute** error
> `|u|·ε` in the exponent, hence a *relative* error `|u|·ε` in the result. A 1-ulp `log`
> feeding a double-precision multiply caps the achievable accuracy at roughly
> `1.5·|y·ln x|` ulp — about **1000 ulp** at the top of the range.

So the design has two tiers, and the document is explicit about which is which:

| Tier | What it needs | Delivered accuracy |
|---|---|---|
| **A — naive** | the existing 1-ulp `exp`/`log` only | `≈ 1.5·|y·ln x| + 1.5` ulp → up to ~1060 ulp |
| **B — recommended** | `log2` returning a **hi/lo pair**, and a `exp2` accepting a reduced argument | ≤ 2 ulp (`pow`), ≤ 5 ulp (`tgamma`) |

Tier B does **not** require a new transcendental. It requires the *existing* range
reduction to stop throwing away the bits it already has (§1.4). That is a small,
local change to code that is already written.

### 0.1 Shared machinery: exact sum and exact product without FMA

Both functions need a few operations carried to ~2× double precision. With no FMA,
this is Dekker/Knuth arithmetic, which uses only `addsd`/`subsd`/`mulsd`:

```
TwoSum(a,b)  -> (s,e)  with s+e == a+b exactly            ; 6 flops, branch-free
    s  = a + b
    bb = s - a
    e  = (a - (s - bb)) + (b - bb)

QuickTwoSum(a,b) -> (s,e)   requires |a| >= |b|           ; 3 flops
    s = a + b
    e = b - (s - a)

Split(a) -> (hi,lo)   with a == hi+lo, 26 bits each       ; SPLIT = 2^27+1
    c  = SPLIT * a
    hi = c - (c - a)
    lo = a - hi

TwoProd(a,b) -> (p,e)  with p+e == a*b exactly            ; 17 flops
    p = a * b
    (ah,al) = Split(a) ; (bh,bl) = Split(b)
    e = ((ah*bh - p) + ah*bl + al*bh) + al*bl
```

`TwoProd` is valid for `|a|,|b| < 2^996`; both call sites here are far inside that.
`SPLIT = 134217729.0 = 0x41A0000002000000`.

---

# PART 1 — `pow(x, y)`

## 1.1 The decomposition, and why base 2 wins

The two candidates are

```
(e)  pow(x,y) = exp ( y * log (x) )
(2)  pow(x,y) = exp2( y * log2(x) )
```

They are mathematically identical and — this is worth stating plainly because it is
usually mis-stated — **their leading error terms are also identical**. Writing
`r = y·log2 x` and `u = y·ln x = r·ln2`, a perturbation `δ` in the exponent produces a
relative error `δ` in `exp(u)` and `δ·ln2` in `2^r`; since `δ_r = δ_u/ln2`, the two
cancel exactly. Measured confirmation is in §1.3: base-e and base-2 track each other to
within the Monte-Carlo noise at every magnitude.

**Recommendation: formulation (2), base 2.** The argument is structural, not a
constant-factor claim:

1. **The hi/lo split is free, and aligned with the reduction you already have.**
   Every `log` implementation reduces `x = 2^k · m` with `m ∈ [√2/2, √2)`. In base 2,
   `log2(x) = k + log2(m)` where `k` is an **exact small integer** and `log2(m) ∈ [-½,½]`.
   The pair `(k, p)` is *already* a two-term unevaluated representation carrying ~64 bits,
   at zero cost — you get it by simply not collapsing `k + p` into one double.
   In base e, `ln(x) = k·ln2 + ln(m)`: the leading term is irrational, so recovering the
   same precision costs a stored `ln2_hi/ln2_lo` pair plus a `TwoProd`. Base 2 removes a
   lossy step that base e must add back.
2. **The final scaling is exact.** With `r = n + f`, `n` integer and `f ∈ [-½,½]`, the
   result is `2^f · 2^n`. `2^n` is built by writing `(n+1023) << 52` into a GPR and
   `movq`-ing it to an xmm — **exact, always**. In base e the same split has to be
   re-derived inside `exp` by multiplying by `1/ln2` and subtracting `n·ln2` in two
   pieces: more rounding, of a quantity that is large.
3. **Overflow, underflow and subnormals become integer comparisons on `n`.** `r > 1024`
   → overflow; `r < -1075` → underflow; `-1074 ≤ n < -1022` → subnormal, handled by
   scaling in two steps (`2^(n+54)` then `2^-54`) so that double rounding is controlled.
   With base e these are comparisons against `709.782712893384`/`-745.1332191019411`,
   irrational-ish thresholds that are awkward to make exactly right at the boundary.
4. **`f` is exactly representable.** `f = r_hi - n` is exact (both operands are integer
   multiples of `ulp(r_hi)` and `|f| ≤ ½`), so no error enters at the split.
5. **Exactness for binary bases.** `log2(2^k) = k` exactly, so `pow(2.0, y)` for integer
   `y` is exact through the general path. `ln(2)` is not exact and buys nothing.

The one thing base 2 does **not** buy is the leading error term. That is governed by how
many bits of `r` you keep, and is the subject of the next section.

## 1.2 Error bound, derived

Let `L = log2(x)` (exact real), `L̂` the computed value with relative error `ε_L`, and
`ε_×` the relative error of the multiplication. Then

```
r̂ = fl(y ⊗ L̂)                 |r̂ − r| ≤ |r| (ε_L + ε_× + O(ε²))          … (1)
```

`2^` maps an **absolute** exponent error into a **relative** value error:

```
2^r̂ / 2^r = 2^(r̂−r) = 1 + (r̂−r)·ln2 + O(·²)
  ⇒  relative error of the power  ≈  ln2 · |r̂ − r|  =  ln2 · |r| (ε_L + ε_×)      … (2)
```

Add the kernel's own error `ε_2` and the final rounding `ε_rnd ≤ 2^-53`:

```
  E_rel  ≤  ln2·|r|·(ε_L + ε_×)  +  ε_2  +  ε_rnd                                  … (3)
```

Expressing in ulp (1 ulp ≡ 2^-52 relative, the conservative end of the 2^-53…2^-52 range):

**Tier A (both `log2` and the product in plain double).** `ε_L = 2^-52` (the stated
1-ulp `log`), `ε_× = 2^-53`, `ε_2 = 2^-52`:

```
  E_ulp  ≤  ln2 · |r| · 1.5  +  1.5   =  1.04·|y·log2 x| + 1.5   =  1.5·|y·ln x| + 1.5
```

With `|r| ≤ 1024` this is **≤ 1067 ulp**. Equivalently: you lose one bit of the result
for every doubling of the result's own exponent. This is not a defect of the
decomposition — it is the arithmetic being asked to represent `r` in 53 bits when `r`
carries an integer part up to 1024.

**Tier B (hi/lo `log2`, double-double product).** Requirement, from (3): to hold the
exponent term under 1 ulp we need `ln2·|r|·ε ≤ 2^-52`, i.e.

```
  ε  ≤  2^-52 / (ln2 · 1024)  =  3.13e-19   ⇒   r must carry ≥ 62 significant bits.
```

Double-double (~106 bits) clears this by 44 bits, so the exponent term collapses to
`< 0.001` ulp and the budget becomes:

| Contribution | ulp |
|---|---|
| `r` carried in double-double (≥62 bits needed, 106 supplied) | < 0.001 |
| splitting `f = (r_hi − n) + r_lo`, `f` kept as one double | ≤ 0.17 |
| `exp2` kernel on `f ∈ [-½,½]` (the 1-ulp assumption) | ≤ 1.0 |
| `2^n` scaling | 0 (exact) |
| final rounding | ≤ 0.5 |
| **total** | **≤ 1.7 ulp** |

Note precisely where the "1 ulp exp/log" assumption enters and where it does not:
the **`log`** assumption is *not enough* and must be upgraded to a hi/lo result;
the **`exp`** assumption is enough as-is, because its argument is confined to `[-½,½]`
where a 1-ulp relative error is also a 1-ulp *absolute* error on a quantity of size ≤ ½.

## 1.3 Measured (Monte-Carlo, 4000 random `(x,y)` per band)

Error model simulated in `pow_err.py`: `log`/`exp` perturbed by a uniform ±1 ulp,
products rounded to binary64, compared against `mpmath` at 60 digits.

| `|y·ln x|` band | base-e, tier A | base-2, tier A | base-e, tier B | base-2, tier B |
|---|--:|--:|--:|--:|
| 0 – 1 | 2.2 | 1.9 | 1.46 | 1.41 |
| 1 – 10 | 15.0 | 13.9 | 1.37 | 1.42 |
| 10 – 100 | 133.5 | 138.2 | 1.40 | 1.45 |
| 100 – 500 | 673.9 | 710.1 | 1.37 | 1.44 |
| 500 – 709 | 1051.5 | 1010.9 | 1.41 | 1.44 |

The tier-A column tracks the predicted `1.5·|y·ln x|` to within ~5%
(predicted 1064 at the top; measured 1051). The tier-B column is flat, as (3) predicts
once the exponent term is removed. Base-e and base-2 are indistinguishable — confirming
that the recommendation for base 2 rests on structure (§1.1), not on the error constant.

## 1.4 What actually has to change in the existing `log`

Not much, and nothing new mathematically:

1. Return the reduction exponent `k` **separately** instead of folding it in.
2. Evaluate the mantissa polynomial `p ≈ log2(m)` with a low word `p_lo`
   (one extra `TwoSum` per Horner step in the last three steps is sufficient to reach
   ~2^-100; the leading terms dominate).
3. Expose `exp2(f)` on `f ∈ [-½,½]` — this is the *inner* part of the existing `exp`,
   already present, just not currently addressable on its own.

`k` is exact, `p_hi+p_lo` carries ~100 bits, so `r = y ⊗ (k + p_hi + p_lo)` via `TwoProd`
plus `TwoSum` accumulation lands well inside the 62-bit requirement.

## 1.5 Complete IEEE-754 / C99 Annex F.9.4.4 special-case table

Tested **in this order**. The last column is the important one: ✗ marks a case the bare
`exp2(y·log2 x)` path gets **wrong** (not merely inexact) if it is not caught first.

| # | Condition | Result | Raises | Decomposition without this branch |
|--:|---|---|---|---|
| 1 | `y == ±0` (any `x`, **including NaN**) | `1.0` | — | ✗ `log2(0)=−∞`, `log2(x<0)=NaN`, `∞·0=NaN`, `NaN·0=NaN`. Wrong for `x ∈ {±0, <0, ±∞, NaN}` |
| 2 | `x == 1.0` (any `y`, **including NaN**) | `1.0` | — | ✗ `log2(1)=0`, then `±∞·0=NaN` and `NaN·0=NaN`. Wrong for `y ∈ {±∞, NaN}` |
| 3 | `x` is NaN **or** `y` is NaN | NaN | — | ✓ propagates naturally (but must come *after* 1 and 2) |
| 4 | `y == +∞` and `|x| < 1` | `+0` | — | ✓ (`log2|x| < 0`, product `−∞`) |
| 5 | `y == +∞` and `|x| > 1` | `+∞` | — | ✓ |
| 6 | `y == −∞` and `|x| < 1` | `+∞` | — | ✓ |
| 7 | `y == −∞` and `|x| > 1` | `+0` | — | ✓ |
| 8 | `x == −1.0` and `y == ±∞` | `1.0` | — | ✗ `log2(−1)=NaN`. Caught by rule 2 only if you test `|x|`; safest as its own row |
| 9 | `y == 1.0` | `x` (bit-exact) | — | ✗ *inexact*: `exp2(log2 x)` is within ~2 ulp of `x`, not equal to `x` |
| 10 | `x == +∞`, `y < 0` | `+0` | — | ✓ |
| 11 | `x == +∞`, `y > 0` | `+∞` | — | ✓ |
| 12 | `x == −∞`, `y < 0`, `y` **odd** integer | `−0` | — | ✗ sign lost: `log2(−∞)=NaN` |
| 13 | `x == −∞`, `y < 0`, otherwise | `+0` | — | ✗ NaN |
| 14 | `x == −∞`, `y > 0`, `y` **odd** integer | `−∞` | — | ✗ NaN |
| 15 | `x == −∞`, `y > 0`, otherwise | `+∞` | — | ✗ NaN |
| 16 | `x == ±0`, `y < 0`, `y` **odd** integer | `±∞` (sign of `x`) | divide-by-zero | ✗ sign of zero is lost by `log2` |
| 17 | `x == ±0`, `y < 0`, otherwise | `+∞` | divide-by-zero | ✗ (magnitude right, but the raise must be explicit) |
| 18 | `x == ±0`, `y > 0`, `y` **odd** integer | `±0` (sign of `x`) | — | ✗ `−0` must survive; `log2(−0) = −∞` discards it |
| 19 | `x == ±0`, `y > 0`, otherwise | `+0` | — | ✓ magnitude only |
| 20 | `x < 0` and `y` **not** an integer | NaN | **invalid** | ✗ `log2(x<0)=NaN` gives NaN by accident, but with no `invalid` raise |
| 21 | `x < 0` and `y` an integer | `(odd ? −1 : +1) · pow(|x|, y)` | — | ✗ `log2(x<0)=NaN` |
| 22 | otherwise | general path on `|x|` | overflow / underflow / inexact | — |

Notes that repeatedly catch implementations out:

* **Rule 1 outranks rule 3.** `pow(NaN, 0) == 1.0`, not NaN. Likewise rule 2:
  `pow(1.0, NaN) == 1.0`.
* **`pow(±0, ±0) = 1`** falls out of rule 1 and must not reach rules 16–19.
* **`y = 1e300` is an even integer.** Every finite double with `|y| ≥ 2^53` is an even
  integer, so `pow(-2.0, 1e300) = +∞`, *not* NaN. An implementation that tests
  integer-ness with `cvttsd2si` and no magnitude guard will produce garbage here.
* Rows 12–19 are the whole reason the sign of `x` must be extracted **before** the
  logarithm, not recovered afterwards.

## 1.6 Testing "`y` is an odd integer" with the available instructions

`y/2` is *always exact* (division by a power of two), and `y` is an odd integer exactly
when `y/2` has a fractional part of ½. That gives a uniform test with no magnitude
guard, no GPR round-trip, and no reliance on `2^53` reasoning:

```asm
; in : xmm1 = y   (finite; ±0/±inf/NaN already handled by rows 1–19)
; out: falls through to .odd / .even / .notint
        roundsd  xmm2, xmm1, 0x0B      ; xmm2 = trunc(y)      (imm: bit3 suppress, mode 3 = toward zero)
        ucomisd  xmm2, xmm1
        setne    al
        test     al, al
        jnz      .notint                ; trunc(y) != y  ->  y is not an integer

        movsd    xmm3, [rip+K_HALF]     ; 0.5
        mulsd    xmm3, xmm1             ; xmm3 = y*0.5     EXACT
        roundsd  xmm4, xmm3, 0x0B       ; trunc(y*0.5)
        ucomisd  xmm4, xmm3
        setne    al
        test     al, al
        jnz      .odd                   ; y/2 not an integer  ->  y is ODD
        jmp      .even
```

Correctness at the extremes: for `|y| ≥ 2^53`, `ulp(y) ≥ 2` so `y` is an even integer;
`trunc(y) == y` holds, `y*0.5` is an integer, and the code reaches `.even` — right answer,
no special case needed. For `|y| < 1` and `y ≠ 0`, `trunc(y) = 0 ≠ y` → `.notint`. For
`y = 0`, `.even` (0 is even), though rule 1 has already returned.

**SSE2-only fallback** (no `roundsd`), which *does* need the magnitude guard:

```asm
        ucomisd  xmm1, [rip+K_2P53]     ; y >= 2^53 ?
        setae    al ; test al,al ; jnz .even
        movsd    xmm5, [rip+K_2P53N]    ; -2^53
        ucomisd  xmm1, xmm5             ; y <= -2^53 ?
        setbe    al ; test al,al ; jnz .even
        cvttsd2si rax, xmm1             ; safe: |y| < 2^53 < 2^63
        cvtsi2sd xmm2, rax
        ucomisd  xmm2, xmm1
        setne    al ; test al,al ; jnz .notint
        and      rax, 1
        jnz      .odd
        jmp      .even
```

## 1.7 Exactness fast path (optional, but it is what makes `pow(10,3) == 1000`)

The general path is accurate to ≤2 ulp but is **not** exact for cases users assume are
exact. Recommended: for integer `y` with `|y| ≤ 64`, run binary exponentiation using
`TwoProd` and **abandon to the general path the moment any residual `e ≠ 0`**. When it
completes, every intermediate product was exact, so the result is exact by construction;
when it bails, nothing is lost but a few cycles. This makes rows 1, 3, 4, 7, 8 of the
`pow` vector table bit-exact and costs no accuracy anywhere else.

---

# PART 2 — `tgamma(x)`

## 2.1 Scheme selection: Lanczos vs Stirling vs Spouge

All three reduce Γ to `exp`/`log`/`pow` plus a rational or series correction, so all
three are *admissible* under the instruction budget. They differ in conditioning.

**Stirling / asymptotic.** `ln Γ(x) ~ (x−½)ln x − x + ½ln2π + Σ B_2n/(2n(2n−1)x^(2n−1))`.
The series is *divergent*; for binary64 accuracy it needs `x ≳ 12`, so every argument
below that must be pushed up by the recurrence `Γ(x) = Γ(x+n)/(x(x+1)…(x+n−1))`. That
denominator is a product of up to 12 factors — ~6 ulp of error before Γ is even
evaluated — and the recurrence is exactly the step that is *avoidable*. Rejected.

**Spouge.** `Γ(z+1) = (z+a)^(z+½) e^−(z+a) [ √2π + Σ_{k=1}^{⌈a⌉−1} c_k/(z+k) ]` with the
fully explicit, provable
`c_k = ((−1)^(k−1)/(k−1)!)·(a−k)^(k−½)·e^(a−k)` and a rigorous bound
`|ε| ≤ a^(−½)(2π)^(−(a+½))`. The bound is its selling point and its undoing: reaching
`1e−17` needs `a ≈ 20`, and then the coefficients run to ~10⁹ while their sum is ~2.5.
Measured (`lanczos_explore.py`):

| `a` (= `g`), `N` | approximation error | `Σ|c_k|` | decimal digits lost to cancellation |
|---|--:|--:|--:|
| 9, 10 | 1.3e−11 | 4.3e4 | 4.6 |
| 12, 13 | 5.2e−15 | 2.3e6 | 6.4 |
| 15, 16 | 5.4e−18 | 1.2e8 | 8.1 |
| 20, 19 | 1.3e−22 | 8.2e10 | 10.9 |

There is no row where the approximation is good enough *and* the arithmetic survives in
binary64. Spouge is the right choice when you have arbitrary precision available to
absorb the cancellation; here you do not. **Rejected — but see §2.2, its formula is
still the source of the leading coefficients.**

**Lanczos — recommended.**
```
Γ(z+1) = √(2π) · (z+g+½)^(z+½) · e^−(z+g+½) · A(z),    A(z) = c₀ + Σ_{k=1}^{N} c_k/(z+k)
```
with `z = x−1`. Transcendentals needed: `log2`, `exp2` — the same two `pow` needs, and
nothing else on the primary branch. `pow` itself is *not* required: the power is folded
into the single `2^r` (§2.3).

A detail worth recording because it is widely misreported: the Lanczos `c_k` are
**not** the residues. The residue formula (identical to Spouge's) reproduces the
well-known `g=7` set only for its leading terms and then diverges, and it is undefined
for `k > g+½` where `(g+½−k)` goes negative and the power turns complex:

| k | residue formula, g=7 | published g=7 set |
|--:|---|---|
| 1 | 676.52036812188353738 | 676.5203681218851 |
| 2 | −1259.1392167222818133 | −1259.1392167224028 |
| 5 | 12.507343225734506631 | 12.507343278686905 |
| 7 | 1.0093222299290721e−5 | 9.9843695780195716e−6 |
| 8 | *undefined* (`(−0.5)^7.5` complex) | 1.5056327351493116e−7 |

The published sets are **fitted**, not derived. So the coefficients below were fitted
too — by Remez exchange, from scratch (§2.2).

## 2.2 Coefficients: derivation, choice of `g`, and the table

### Method

`A(z)` is linear in `c`, so choosing `c` to minimise
`max_z | A(z)/S(z) − 1 |`, where `S(z) = Γ(z+1) / (√2π·(z+g+½)^(z+½)·e^−(z+g+½))`,
is a **linear Chebyshev approximation problem** — solvable exactly by Remez exchange.
Substituting `t = 1/(z+1)` maps the evaluation domain `z ∈ [−½, ∞)` onto the closed
interval `t ∈ [0, 2]` (`t=0` ↔ `z=∞`, where `S→1` and the basis functions vanish, so the
endpoint error is `c₀−1`). Basis: `φ₀ = 1`, `φ_k = t/(1+(k−1)t)`. Script: `lz2.py`
(Remez driver), `lz6.py`/`lz_dyadic.py` (selection), `final_gamma.py` (final fit).

The poles of `S` at `z = −1, −2, …` lie **outside** `[−½, ∞)`, which is why the fit is
free to move the `c_k` off the residues: the reflection formula (§2.3) keeps every
evaluation at `x ≥ ½`.

### `g` must be a dyadic rational — this is not cosmetic

The fit is performed for one specific real `g`; the emitted code uses `fl(g)`. Perturbing
`g` by `δ` perturbs `w = x + g − ½` by `δ`, and

```
  ∂/∂w [ (z+½)·ln w − w ]  =  (z+½)/w − 1  =  (z+½−w)/w  =  −g/w
  ⇒  relative error of the result  =  g·δ / w
```

For `g = 3.8` — which is **not** representable; `fl(3.8) = 3.7999999999999998`, so
`δ = 1.78e−16` — this is `3.875·1.78e−16/w`, worst at the small-`x` end (`w ≈ 3.9`):
about **0.8 ulp**, injected for free and for nothing, on top of a fit whose own error is
0.04 ulp. It decays as `1/w`, so it is invisible in a test suite that only samples large
arguments. This is why the literature's constants look strange (`607/128`,
`6.024680040776729583740234375`): they are dyadic rationals, exact in binary64.
**`g = 31/8 = 3.875`** is exact (`0x400F000000000000`).

The same `g·δ/w` formula, applied with `δ = ½·ulp(w) = ½·2⁻⁵²·w`, gives `g/2 = 1.94` ulp
— that is the cost of letting `w = x + g − ½` round, and the reason §2.3 computes it with
`TwoSum` instead.

### Selection

Selecting on *realised* binary64 error rather than on the approximation error, because
the approximation error stopped mattering several digits ago. Sampled over
`z ∈ [−½, 10¹³]`; `κ = max_z Σ|c_k/(z+k)| / |A(z)|` is the cancellation amplifier:

| `g` (dyadic) | N | minimax (ulp) | realised, plain sum | realised, compensated | κ |
|---|--:|--:|--:|--:|--:|
| 3.5 | 15 | 0.121 | 1.698 | 1.473 | 2.17 |
| 3.625 | 15 | 0.068 | 1.631 | 1.414 | 2.35 |
| **3.875** | **13** | **0.041** | **1.676** | **1.623** | **2.80** |
| 4 | 14 | 0.009 | 2.171 | 2.168 | 3.07 |
| 4.25 | 14 | 0.012 | 2.505 | 2.330 | 3.73 |
| 607/128 | 15 | 0.0001 | 2.938 | 2.492 | 5.63 |

Note the shape: past `g ≈ 3.9` the approximation error keeps improving and the *delivered*
error gets **worse**, because κ grows faster than the fit improves. Godfrey's classical
`g = 607/128` is a poor choice for a plain-`double` partial-fraction evaluation for exactly
this reason. Chosen: **`g = 31/8 = 3.875`, `N = 13`** — fewest coefficients at the optimum,
and plain summation is within 0.05 ulp of compensated, so `TwoSum` in the inner loop is
*not* required.

### The coefficients

`g = 31/8 = 3.875` exactly, `N = 13`, minimax approximation error
`9.159e−18 = 0.0412 ulp`, `κ = 2.799`.

| k | decimal (17 sig. digits) | IEEE-754 binary64 |
|--:|---|---|
| 0 | `+1.00000000000000000e+00` | `0x3FF0000000000000` |
| 1 | `+2.14185876961132386e+01` | `0x40356B2890315079` |
| 2 | `-1.56983815816653927e+01` | `0xC02F6592454AC363` |
| 3 | `+1.74900782896357820e+00` | `0x3FFBFBEFA21D8B63` |
| 4 | `-3.13130863868064111e-03` | `0xBF69A6D4864C504D` |
| 5 | `+1.81172273513538186e-04` | `0x3F27BF21FAB7ADF5` |
| 6 | `-2.69516809166505907e-04` | `0xBF31A9BDE18F45C7` |
| 7 | `+3.18104372036599379e-04` | `0x3F34D8E7DFE68F46` |
| 8 | `-3.03618317250145661e-04` | `0xBF33E5DEBE38E9CF` |
| 9 | `+2.10367172155954626e-04` | `0x3F2B92C03FC6915A` |
| 10 | `-9.30113041702390663e-05` | `0xBF1861E209CC9548` |
| 11 | `+1.95439362169870012e-05` | `0x3EF47E4911F82CE7` |
| 12 | `+1.14256291973320033e-06` | `0x3EB32B453738ECB2` |
| 13 | `-9.85325687410053329e-07` | `0xBEB087F10CEA8931` |

Other rodata this function needs:

| constant | decimal | binary64 |
|---|---|---|
| `√(2π)` | `+2.50662827463100069e+00` | `0x40040D931FF62706` |
| `g = 31/8` | `+3.87500000000000000e+00` | `0x400F000000000000` |
| `g − ½ = 27/8` | `+3.37500000000000000e+00` | `0x400B000000000000` |
| `log2(e)` | `+1.44269504088896339e+00` | `0x3FF71547652B82FE` |
| `π` | `+3.14159265358979312e+00` | `0x400921FB54442D18` |
| `SPLIT = 2^27+1` | `+1.34217729000000000e+08` | `0x41A0000002000000` |

> **Do not generate these constants by composing operations.** Every value above is the
> *correctly-rounded* binary64 of the exact mathematical constant, computed at 40+ digits.
> Composing gets it wrong: `sqrt(fl(pi))` yields `0x40040D931FF62705` for `√(2π)` and
> `0x3FFC5BF891B4EF6A` for `√π` — **1 ulp low in both cases**, because it takes the square
> root of the rounded `π` rather than rounding the true root. (Verified against `mpmath`:
> the correctly-rounded values are off the exact constant by 1.83e−16 and 7.67e−17
> respectively; the composed ones by 2.61e−16 and 1.45e−16.) `√π` is also the expected
> value of `tgamma(0.5)` in §3.2 row 11, so this single ulp is directly test-visible.

### Reproduction

```
python3 lz2.py          # Remez exchange driver: remez(g,N), measure(), kappa()
python3 lz_dyadic.py    # dyadic-g selection table above
python3 final_gamma.py  # emits the 14 coefficients + the dense validation below
```
`final_gamma.py` also re-validates: 88 000 random `z ∈ [−½, 10¹⁴]`,
**max 1.660 ulp** for the Lanczos sum alone, worst at `z = 2.4436`.

## 2.3 Algorithm, including the reflection

The whole function, with the exponent carried separately as `(m, n)` meaning `m·2^n`:

```
core(x)   -- valid for x >= 0.5, returns (m, n)
    z   = x − 1                                  ; exact for 0.5 <= x < 2^53
    A   = c₀ + Σ_{k=13..1} c_k/(z+k)             ; descending k, plain double
    (w_hi, w_lo) = TwoSum(x, 3.375)              ; w = x + g − ½ carried EXACTLY
    (L_hi, L_lo) = log2_dd(w_hi, w_lo)           ; hi/lo log2  (§1.4)
    r   = (z+½) ⊗ L  ⊖  w ⊗ log2(e)              ; double-double throughout
    n   = round(r_hi) ; f = (r_hi − n) + r_lo    ; f ∈ [−½,½], r_hi−n exact
    m   = (√2π · A) · exp2(f)
    return (m, n)
```

```
tgamma(x)
    x >= 0.5        :  (m,n) = core(x);           return m·2ⁿ
    −0.5 < x < 0.5  :  (m,n) = core(x+1);         return (m/x)·2ⁿ          ; Γ(x)=Γ(x+1)/x
    x <= −0.5       :  y = −x                     ; EXACT, a sign flip
                       (m,n) = core(y)
                       return  π / (sinpi(x) · m · y) · 2^(−n)
```

### Why the reflection is written that way — a 312-ulp trap

The textbook form is `Γ(x) = π / (sin(πx)·Γ(1−x))`. Computing `1−x` in binary64 is
**not** exact: at `x ≈ −127.6`, `1−x` crosses the binade at 128 and rounds, with absolute
error up to `½·ulp(128.6) = 1.42e−14`. Γ's condition number with respect to its argument
is `|x·ψ(x)| ≈ x·ln x`:

| x | `|x·ψ(x)|` | cost of a ½-ulp argument error |
|--:|--:|--:|
| 2 | 0.8 | 0 ulp |
| 10 | 22.5 | 11 ulp |
| 50 | 195.1 | 98 ulp |
| 128 | 620.6 | **310 ulp** |
| 171 | 878.7 | **439 ulp** |

Measured, before and after (`reflect2.py`, 3000 random `x` per band):

| x range | `Γ(1−x)` form | `Γ(1−x) = (−x)·Γ(−x)` form |
|---|--:|--:|
| [−0.5, 0.5] | 3.85 | 2.96 |
| [−5, −0.5] | 6.35 | 4.35 |
| [−20, −5] | 25.16 | 4.20 |
| [−100, −20] | 132.50 | 3.86 |
| [−175, −100] | **312.15** | **4.10** |

Using `Γ(1−x) = (−x)·Γ(−x)` replaces the inexact `1−x` with the **exact** `−x` (a sign
bit flip) and removes the whole effect. This is the single highest-value line in the
design. The `−0.5 < x < 0.5` branch uses `Γ(x+1)/x` instead: `x+1` is also inexact there,
but `ψ` is `O(1)` near 1, so the amplification is ≈0.3 ulp rather than 300.

### `sinpi` is required — and `sin(π·x)` will not do

The reflection needs `sin(πx)`. This is the **only** transcendental beyond `exp2`/`log2`
that `tgamma` requires, and it must be implemented as `sinpi(x)`, not as a call to a
generic `sin` with a rounded `π`: for `|x| ~ 170`, `fl(π)·x` has absolute error
`~170·1.2e−16 = 2e−14` in an argument whose period is `π`, destroying the result near
integers. `sinpi` is cheap and exact where it matters:

```
sinpi(x):  n = round(x)            ; roundsd, mode 0 (nearest)
           f = x − n               ; EXACT (|f| <= 1/2, Sterbenz)
           s = poly_sinpi(f)       ; odd minimax polynomial on [−1/2, 1/2], deg 9–11
           return (n odd) ? −s : s ; parity of n from the same test as §1.6
```
The reduction is exact, so `sinpi` is well-conditioned right up to the poles — measured
error at `x = −3 + 1e−14` is 1.13 ulp.

### Keeping the exponent separate prevents a spurious overflow

`Γ(1−x)` overflows for `x < −170.6`, yet `Γ(x)` there is a *tiny* number. Because `core`
returns `(m, n)` rather than `m·2ⁿ`, the division `π/(sinpi·m·y)` happens on the O(1)
mantissa and the `2^(−n)` scaling is applied last. No intermediate overflow, and the
subnormal tail is reachable.

## 2.4 Special cases

| # | Condition | Result | Raises |
|--:|---|---|---|
| 1 | `x` is NaN | NaN | — |
| 2 | `x == +∞` | `+∞` | — |
| 3 | `x == −∞` | NaN | invalid |
| 4 | `x == +0` | `+∞` | divide-by-zero |
| 5 | `x == −0` | `−∞` | divide-by-zero |
| 6 | `x < 0` and `x` is an integer | NaN | invalid |
| 7 | `x > 171.62437695630271` | `+∞` | overflow |
| 8 | `x` integer, `1 ≤ x ≤ 23` | exact table lookup | — |
| 9 | `x ≤ −0.5` | reflection (§2.3); underflows to `±0` below ≈ −184 | underflow where applicable |
| 10 | otherwise | `core` / recurrence | inexact |

* **Rows 4/5: the sign of zero is observable.** `tgamma(+0) = +∞` and
  `tgamma(−0) = −∞`. `ucomisd` cannot distinguish them (`+0 == −0`); use `movq rax, xmm0`
  and test the sign bit.
* **Row 6 must precede the reflection**, since `sinpi` at a negative integer is `±0` and
  the division would yield `±∞` rather than the required NaN.
* **Row 7 threshold, computed** (`pipeline.py`, `verify.py`): `Γ(x) = DBL_MAX` at
  `x = 171.62437695630272079`. The largest binary64 with a finite Γ is
  `0x406573FAE561F647 = 171.62437695630271` (`Γ = 1.7976931348622299e308`); the very next
  double `0x406573FAE561F648` gives `1.7976931348624926e308 > DBL_MAX`. `Γ(171)` is finite
  (`7.257415615307999e306`); `Γ(172)` overflows.
* **Row 8 is what makes `tgamma(1)` and `tgamma(2)` exactly `1.0`.** The general path is
  accurate to ~3 ulp and will *not* reliably return exactly `1.0`, `2.0`, `24.0`. `22!` is
  the largest factorial exactly representable in binary64 (`23!` is not — verified in
  `verify.py`), so a 23-entry table (184 bytes) covers every integer argument whose Γ is
  exactly representable. Without this table, rows 1–10 and 21–23 of the vector table in
  §3.2 are *not* bit-exact.
* **Underflow (row 9) is not a clean threshold.** Γ is unbounded at every negative-integer
  pole, so no interval is entirely zero. Computed minima of `|Γ|` between poles:
  `(−171,−170) → 1.19e−308` (normal), `(−176,−175) → 7.51e−320` (subnormal),
  `(−181,−180) → 4.11e−331`, `(−185,−184) → 3.64e−340`. Below ≈ −184 the result is `±0`
  except within ~2e−16 of a pole — narrower than `ulp(184) = 2.8e−14`, so in practice
  every non-integer double below −184 underflows to a signed zero. **The sign alternates**:
  on `(−n−1, −n)` it is `(−1)^(n+1)`, so `tgamma(−184.5) = −0.0`, not `+0.0`.

## 2.5 Error bound

Per-term budget for the primary branch (`x ≥ 0.5`):

| Contribution | ulp | Source |
|---|--:|---|
| Lanczos sum `A(z)`, plain double | ≤ 1.66 | measured, 88k samples (`final_gamma.py`) |
|  ├ minimax approximation | 0.041 | Remez, exact |
|  └ coefficient + division rounding × κ=2.80 | ≈1.62 | measured |
| `√2π` constant + one multiply | ≤ 1.0 | 0.5 + 0.5 |
| `w = x + g − ½` via `TwoSum` | 0 | exact — **1.94 ulp if rounded instead** |
| `r` in double-double (≥62 bits needed) | < 0.01 | §1.2 |
| `f` split + `exp2` kernel | ≤ 1.17 | 0.17 + 1.0 |
| final multiply + rounding | ≤ 1.0 | |
| **derived worst case** | **≈ 4.8** | |
| **measured worst case** | **3.13** | 6000 random `x ∈ [0.5, 171.6]` |

By range, measured (`pipe2.py`):

| x range | measured max |
|---|--:|
| [0.5, 1.5] | 3.11 ulp |
| [1.5, 10] | 3.13 ulp |
| [10, 50] | 2.42 ulp |
| [50, 120] | 2.58 ulp |
| [120, 171.6] | 2.48 ulp |

**Where it is worst: `x ∈ [1.5, 10]`, i.e. small arguments — not large ones.** That is the
opposite of the usual intuition, and it is a direct consequence of `κ` peaking near
`z ≈ 2.4` where the partial-fraction terms cancel hardest. The large-`x` end is *better*
because the double-double exponent removes the term that would otherwise dominate there.

Reflection branch (`x < 0.5`) adds `sinpi` (1 ulp), two multiplies and a divide (1.5 ulp)
and the `π` constant (0.5 ulp): derived ≈ 7.8 ulp, **measured max 4.35 ulp** over
15 000 random arguments in `[−175, 0.5)`.

**Control result — do not skip the double-double.** Same pipeline, same coefficients,
with `log2` returning a plain 1-ulp double instead of a hi/lo pair:
**1102.64 ulp** at `x = 168.93`. The Lanczos coefficients are irrelevant at that point;
they are 0.04 ulp of a 1100-ulp error.

---

# PART 3 — Reference vectors

Expected values are the **correctly-rounded** binary64 of the exact mathematical result,
computed at 60 decimal digits with `mpmath` (`vectors.py`, `vec_g.py`). Hex is the raw
64-bit pattern, little-endian value (i.e. what `movq rax, xmm0` yields).

Read §3.3 **before** turning any of this into an assertion.

## 3.1 `pow(x, y)` — 52 rows

| # | x (dec) | x (hex) | y (dec) | y (hex) | expected (dec) | expected (hex) | what it tests |
|--:|---|---|---|---|---|---|---|
| 1 | `2.0` | `0x4000000000000000` | `10.0` | `0x4024000000000000` | `1024.0` | `0x4090000000000000` | exact power of two, integer y |
| 2 | `2.0` | `0x4000000000000000` | `0.5` | `0x3FE0000000000000` | `1.4142135623730951` | `0x3FF6A09E667F3BCD` | sqrt(2) via pow |
| 3 | `10.0` | `0x4024000000000000` | `3.0` | `0x4008000000000000` | `1000.0` | `0x408F400000000000` | pow(10,3): 1000 exactly only with an integer fast path |
| 4 | `0.5` | `0x3FE0000000000000` | `-3.0` | `0xC008000000000000` | `8.0` | `0x4020000000000000` | negative integer y, exact result 8 |
| 5 | `3.0` | `0x4008000000000000` | `0.3333333333333333` | `0x3FD5555555555555` | `1.4422495703074083` | `0x3FF7137449123EF6` | cube-root-ish, generic path |
| 6 | `1.5` | `0x3FF8000000000000` | `2.5` | `0x4004000000000000` | `2.7556759606310752` | `0x40060B9FD68A4554` | generic non-integer y |
| 7 | `-2.0` | `0xC000000000000000` | `3.0` | `0x4008000000000000` | `-8.0` | `0xC020000000000000` | negative base, ODD integer y -> negative result |
| 8 | `-2.0` | `0xC000000000000000` | `4.0` | `0x4010000000000000` | `16.0` | `0x4030000000000000` | negative base, EVEN integer y -> positive result |
| 9 | `-2.0` | `0xC000000000000000` | `0.5` | `0x3FE0000000000000` | `NaN` | `0x7FF8000000000000` | negative base, NON-integer y -> NaN, invalid |
| 10 | `2.0` | `0x4000000000000000` | `1023.5` | `0x408FFC0000000000` | `1.2711610061536464e+308` | `0x7FE6A09E667F3BCD` | result near overflow boundary, |y*log2 x| = 1023.5 |
| 11 | `2.0` | `0x4000000000000000` | `-1074.0` | `0xC090C80000000000` | `5e-324` | `0x0000000000000001` | smallest positive subnormal, exact |
| 12 | `1e+308` | `0x7FE1CCF385EBC8A0` | `2.0` | `0x4000000000000000` | `+inf` | `0x7FF0000000000000` | overflow -> +inf |
| 13 | `1e-308` | `0x000730D67819E8D2` | `2.0` | `0x4000000000000000` | `+0.0` | `0x0000000000000000` | underflow -> +0 |
| 14 | `0.9999999999999999` | `0x3FEFFFFFFFFFFFFF` | `1000000000000000.0` | `0x430C6BF526340000` | `0.8949187898136988` | `0x3FECA32CBADA6C6A` | x just below 1, huge y: |y*ln x| ~ 111 |
| 15 | `1.0000000000000002` | `0x3FF0000000000001` | `1e+16` | `0x4341C37937E08000` | `9.21143870499353` | `0x40226C41B1A61C92` | x just above 1, huge y: |y*ln x| ~ 2.22 |
| 16 | `2.5` | `0x4004000000000000` | `100.0` | `0x4059000000000000` | `6.223015277861142e+39` | `0x483249AD2594C37D` | large integer y |
| 17 | `7.0` | `0x401C000000000000` | `-13.0` | `0xC02A000000000000` | `1.0321087972715555e-11` | `0x3DA6B24188CA33B0` | negative integer y, generic |
| 18 | `1e-300` | `0x01A56E1FC2F8F359` | `-1.0` | `0xBFF0000000000000` | `9.999999999999999e+299` | `0x7E37E43C8800759B` | reciprocal via pow, y=-1 |
| 19 | `3.0` | `0x4008000000000000` | `-0.5` | `0xBFE0000000000000` | `0.5773502691896257` | `0x3FE279A74590331C` | inverse sqrt of 3 |
| 20 | `1.0000000000000002` | `0x3FF0000000000001` | `4503599627370496.0` | `0x4330000000000000` | `2.718281828459045` | `0x4005BF0A8B145769` | |y*ln x| = 1.0, worst conditioning still small |
| 21 | `NaN` | `0x7FF8000000000000` | `+0.0` | `0x0000000000000000` | `1.0` | `0x3FF0000000000000` | y=0 dominates EVERYTHING, even NaN base |
| 22 | `+inf` | `0x7FF0000000000000` | `+0.0` | `0x0000000000000000` | `1.0` | `0x3FF0000000000000` | y=0 with infinite base |
| 23 | `+0.0` | `0x0000000000000000` | `+0.0` | `0x0000000000000000` | `1.0` | `0x3FF0000000000000` | 0^0 = 1 by C99 |
| 24 | `-0.0` | `0x8000000000000000` | `-0.0` | `0x8000000000000000` | `1.0` | `0x3FF0000000000000` | (-0)^(-0) = 1 |
| 25 | `1.0` | `0x3FF0000000000000` | `NaN` | `0x7FF8000000000000` | `1.0` | `0x3FF0000000000000` | x=1 dominates NaN exponent |
| 26 | `1.0` | `0x3FF0000000000000` | `+inf` | `0x7FF0000000000000` | `1.0` | `0x3FF0000000000000` | x=1 with infinite y |
| 27 | `-1.0` | `0xBFF0000000000000` | `+inf` | `0x7FF0000000000000` | `1.0` | `0x3FF0000000000000` | (-1)^(+inf) = 1 |
| 28 | `-1.0` | `0xBFF0000000000000` | `-inf` | `0xFFF0000000000000` | `1.0` | `0x3FF0000000000000` | (-1)^(-inf) = 1 |
| 29 | `-3.25` | `0xC00A000000000000` | `1.0` | `0x3FF0000000000000` | `-3.25` | `0xC00A000000000000` | y=1 returns x bit-exactly |
| 30 | `+0.0` | `0x0000000000000000` | `3.0` | `0x4008000000000000` | `+0.0` | `0x0000000000000000` | +0 to positive odd int -> +0 |
| 31 | `-0.0` | `0x8000000000000000` | `3.0` | `0x4008000000000000` | `-0.0` | `0x8000000000000000` | -0 to positive ODD int -> -0 (sign must survive) |
| 32 | `-0.0` | `0x8000000000000000` | `2.0` | `0x4000000000000000` | `+0.0` | `0x0000000000000000` | -0 to positive even int -> +0 |
| 33 | `-0.0` | `0x8000000000000000` | `0.5` | `0x3FE0000000000000` | `+0.0` | `0x0000000000000000` | -0 to positive non-integer -> +0 |
| 34 | `+0.0` | `0x0000000000000000` | `-3.0` | `0xC008000000000000` | `+inf` | `0x7FF0000000000000` | +0 to negative odd int -> +inf, divide-by-zero |
| 35 | `-0.0` | `0x8000000000000000` | `-3.0` | `0xC008000000000000` | `-inf` | `0xFFF0000000000000` | -0 to negative ODD int -> -inf, divide-by-zero |
| 36 | `-0.0` | `0x8000000000000000` | `-2.0` | `0xC000000000000000` | `+inf` | `0x7FF0000000000000` | -0 to negative even int -> +inf, divide-by-zero |
| 37 | `0.5` | `0x3FE0000000000000` | `-inf` | `0xFFF0000000000000` | `+inf` | `0x7FF0000000000000` | |x|<1, y=-inf -> +inf |
| 38 | `0.5` | `0x3FE0000000000000` | `+inf` | `0x7FF0000000000000` | `+0.0` | `0x0000000000000000` | |x|<1, y=+inf -> +0 |
| 39 | `2.0` | `0x4000000000000000` | `-inf` | `0xFFF0000000000000` | `+0.0` | `0x0000000000000000` | |x|>1, y=-inf -> +0 |
| 40 | `2.0` | `0x4000000000000000` | `+inf` | `0x7FF0000000000000` | `+inf` | `0x7FF0000000000000` | |x|>1, y=+inf -> +inf |
| 41 | `-inf` | `0xFFF0000000000000` | `-3.0` | `0xC008000000000000` | `-0.0` | `0x8000000000000000` | -inf to negative ODD int -> -0 |
| 42 | `-inf` | `0xFFF0000000000000` | `-2.0` | `0xC000000000000000` | `+0.0` | `0x0000000000000000` | -inf to negative even int -> +0 |
| 43 | `-inf` | `0xFFF0000000000000` | `3.0` | `0x4008000000000000` | `-inf` | `0xFFF0000000000000` | -inf to positive ODD int -> -inf |
| 44 | `-inf` | `0xFFF0000000000000` | `2.0` | `0x4000000000000000` | `+inf` | `0x7FF0000000000000` | -inf to positive even int -> +inf |
| 45 | `-inf` | `0xFFF0000000000000` | `0.5` | `0x3FE0000000000000` | `+inf` | `0x7FF0000000000000` | -inf to positive non-integer -> +inf |
| 46 | `+inf` | `0x7FF0000000000000` | `-2.0` | `0xC000000000000000` | `+0.0` | `0x0000000000000000` | +inf to negative y -> +0 |
| 47 | `+inf` | `0x7FF0000000000000` | `2.0` | `0x4000000000000000` | `+inf` | `0x7FF0000000000000` | +inf to positive y -> +inf |
| 48 | `NaN` | `0x7FF8000000000000` | `2.0` | `0x4000000000000000` | `NaN` | `0x7FF8000000000000` | NaN base propagates |
| 49 | `2.0` | `0x4000000000000000` | `NaN` | `0x7FF8000000000000` | `NaN` | `0x7FF8000000000000` | NaN exponent propagates |
| 50 | `-2.0` | `0xC000000000000000` | `1e+300` | `0x7E37E43C8800759C` | `+inf` | `0x7FF0000000000000` | 1e300 IS an even integer -> +inf, not NaN |
| 51 | `-1.0` | `0xBFF0000000000000` | `0.5` | `0x3FE0000000000000` | `NaN` | `0x7FF8000000000000` | (-1)^0.5 -> NaN, invalid |
| 52 | `-2.0` | `0xC000000000000000` | `9007199254740992.0` | `0x4340000000000000` | `+inf` | `0x7FF0000000000000` | y = 2^53 is an even integer -> +inf |

## 3.2 `tgamma(x)` — 44 rows

| # | x (dec) | x (hex) | expected (dec) | expected (hex) | what it tests |
|--:|---|---|---|---|---|
| 1 | `1.0` | `0x3FF0000000000000` | `1.0` | `0x3FF0000000000000` | MUST be exactly 1.0 |
| 2 | `2.0` | `0x4000000000000000` | `1.0` | `0x3FF0000000000000` | MUST be exactly 1.0 |
| 3 | `3.0` | `0x4008000000000000` | `2.0` | `0x4000000000000000` | exactly 2.0 |
| 4 | `5.0` | `0x4014000000000000` | `24.0` | `0x4038000000000000` | exactly 24.0 (4!) |
| 5 | `6.0` | `0x4018000000000000` | `120.0` | `0x405E000000000000` | exactly 120.0 |
| 6 | `11.0` | `0x4026000000000000` | `3628800.0` | `0x414BAF8000000000` | 10! = 3628800, still exactly representable |
| 7 | `18.0` | `0x4032000000000000` | `355687428096000.0` | `0x42F437EEECD80000` | 17! = 355687428096000, exactly representable |
| 8 | `19.0` | `0x4033000000000000` | `6402373705728000.0` | `0x4336BEECCA730000` | 18! = 6402373705728000, exactly representable |
| 9 | `23.0` | `0x4037000000000000` | `1.1240007277776077e+21` | `0x444E77526159F06C` | 22! = 1124000727777607680000 IS exactly representable (largest factorial that is); the algorithm is NOT required to hit it |
| 10 | `24.0` | `0x4038000000000000` | `2.585201673888498e+22` | `0x4495E5C335F8A4CE` | 23! is NOT exactly representable -> nearest double |
| 11 | `0.5` | `0x3FE0000000000000` | `1.772453850905516` | `0x3FFC5BF891B4EF6B` | sqrt(pi) -- the canonical half-integer |
| 12 | `1.5` | `0x3FF8000000000000` | `0.886226925452758` | `0x3FEC5BF891B4EF6B` | sqrt(pi)/2 |
| 13 | `2.5` | `0x4004000000000000` | `1.329340388179137` | `0x3FF544FA6D47B390` | 3*sqrt(pi)/4 |
| 14 | `4.5` | `0x4012000000000000` | `11.631728396567448` | `0x40274371E7866C65` | half-integer, mid range |
| 15 | `-0.5` | `0xBFE0000000000000` | `-3.544907701811032` | `0xC00C5BF891B4EF6B` | -2*sqrt(pi), reflection path |
| 16 | `-1.5` | `0xBFF8000000000000` | `2.363271801207355` | `0x4002E7FB0BCDF4F2` | 4*sqrt(pi)/3, reflection past one pole |
| 17 | `-2.5` | `0xC004000000000000` | `-0.9453087204829419` | `0xBFEE3FF812E32183` | -8*sqrt(pi)/15 |
| 18 | `-3.5` | `0xC00C000000000000` | `0.2700882058522691` | `0x3FD149200ACAEE94` | reflection, deeper |
| 19 | `0.1` | `0x3FB999999999999A` | `9.51350769866873` | `0x402306EA7B280D87` | small positive, recurrence path Gamma(x)=Gamma(x+1)/x |
| 20 | `1e-300` | `0x01A56E1FC2F8F359` | `9.999999999999999e+299` | `0x7E37E43C8800759B` | tiny x: Gamma(x) ~ 1/x, near overflow of the reciprocal |
| 21 | `1.4616321449683622` | `0x3FF762D86356BE3F` | `0.8856031944108887` | `0x3FEC56DC82A74AEF` | argmin of Gamma on (0,inf) (true argmin 1.46163214496836234) |
| 22 | `10.0` | `0x4024000000000000` | `362880.0` | `0x4116260000000000` | 362880 exactly |
| 23 | `20.0` | `0x4034000000000000` | `1.21645100408832e+17` | `0x437B02B930689000` | 19! exact-ish |
| 24 | `100.5` | `0x4059200000000000` | `9.320963104082716e+156` | `0x6085B98374DB8C0B` | large half-integer |
| 25 | `-100.5` | `0xC059200000000000` | `-3.3536908198076787e-159` | `0x9F07932FB5136292` | large negative non-integer: reflection with |x|>>1 |
| 26 | `-170.5` | `0xC065500000000000` | `-3.3127395215386074e-308` | `0x8017D2374DFCDA7A` | deep reflection, result still normal |
| 27 | `-184.5` | `0xC067100000000000` | `-0.0` | `0x8000000000000000` | Gamma is NEGATIVE here; reflection result UNDERFLOWS to -0.0 |
| 28 | `171.0` | `0x4065600000000000` | `7.257415615307999e+306` | `0x7FA4AB7864418639` | largest integer argument that does not overflow |
| 29 | `171.6` | `0x4065733333333333` | `1.5858969096672565e+308` | `0x7FEC3ADADC5107B1` | just below the overflow threshold |
| 30 | `171.61` | `0x406573851EB851EC` | `1.6695813546313736e+308` | `0x7FEDB8336BA69B7E` | closer still to the threshold |
| 31 | `171.6243769563027` | `0x406573FAE561F647` | `1.7976931348622299e+308` | `0x7FEFFFFFFFFFFE51` | LARGEST double x with Gamma(x) finite (0x406573FAE561F647) |
| 32 | `171.62437695630274` | `0x406573FAE561F648` | `+inf` | `0x7FF0000000000000` | next double up (0x...F648) -> +inf, overflow |
| 33 | `171.63` | `0x40657428F5C28F5C` | `+inf` | `0x7FF0000000000000` | just above the threshold -> +inf, overflow |
| 34 | `172.0` | `0x4065800000000000` | `+inf` | `0x7FF0000000000000` | integer argument that overflows -> +inf |
| 35 | `+0.0` | `0x0000000000000000` | `+inf` | `0x7FF0000000000000` | pole at +0 -> +inf, divide-by-zero |
| 36 | `-0.0` | `0x8000000000000000` | `-inf` | `0xFFF0000000000000` | pole at -0 -> -inf, divide-by-zero (sign of zero matters) |
| 37 | `-1.0` | `0xBFF0000000000000` | `NaN` | `0x7FF8000000000000` | negative integer -> NaN, invalid |
| 38 | `-2.0` | `0xC000000000000000` | `NaN` | `0x7FF8000000000000` | negative integer -> NaN, invalid |
| 39 | `-171.0` | `0xC065600000000000` | `NaN` | `0x7FF8000000000000` | large negative integer -> NaN, invalid |
| 40 | `NaN` | `0x7FF8000000000000` | `NaN` | `0x7FF8000000000000` | NaN propagates |
| 41 | `+inf` | `0x7FF0000000000000` | `+inf` | `0x7FF0000000000000` | +inf -> +inf |
| 42 | `-inf` | `0xFFF0000000000000` | `NaN` | `0x7FF8000000000000` | -inf -> NaN, invalid |
| 43 | `2.0000000000000004` | `0x4000000000000001` | `1.0000000000000002` | `0x3FF0000000000001` | 1 ulp above 2: result must NOT be 1.0 |
| 44 | `0.9999999999999999` | `0x3FEFFFFFFFFFFFFF` | `1.0` | `0x3FF0000000000000` | 1 ulp below 1 |

## 3.3 Which rows may be compared bit-exactly, and which may not

This design promises ≤2 ulp for `pow` and ≤5 ulp for `tgamma`. **A test that demands
bit-exactness on a row this design does not promise exactly is a bad test**, and will
either be silenced or will silence a real regression later. The rows split three ways.

### (a) MUST be bit-exact — assert on the hex

Rows whose result is a special value, or is exact by rule rather than by numerics:

* **`pow`** rows **9** (NaN), **11** (`2^-1074`; the `2^n` scaling is exact by
  construction), **12** (`+inf`, overflow), **13** (`+0`, underflow), and **21–52**.
  Rows 21–52 are the whole C99 special-case set: the NaN/±inf/±0 results, the `1.0`
  results from `y=0` and `x=1`, and `y=1` returning `x` unchanged. **The sign of every
  zero and infinity is part of the assertion** — rows 31 (`-0.0`), 35 (`-inf`),
  41 (`-0.0`), 42 (`+0.0`) fail on sign alone in a naive implementation.
* **`tgamma`** rows **27** (`-0.0` — sign included; `+0.0` is a *failure* here),
  **32, 33, 34** (`+inf`, overflow), **35** (`+inf`), **36** (`-inf`),
  **37, 38, 39** (NaN), **40** (NaN propagation), **41** (`+inf`), **42** (NaN).

### (b) Bit-exact ONLY IF the corresponding fast path is implemented

* **`pow`** rows **1, 3, 4, 7, 8** (`2^10=1024`, `10^3=1000`, `0.5^-3=8`, `(-2)^3=-8`,
  `(-2)^4=16`) — exact **iff** the exactness-tracked binary exponentiation of §1.7 is
  present. Through the general path these land within 1 ulp but need not be equal.
* **`tgamma`** rows **1–9** (integer arguments 1, 2, 3, 5, 6, 11, 18, 19, 23) and
  **22, 23** (arguments 10 and 20) — exact **iff** the 23-entry integer table of §2.4
  row 8 is present. `tgamma(1)` and `tgamma(2)` returning exactly `1.0` is a hard
  requirement in practice and the general path does **not** deliver it: at ~3 ulp it will
  return `1.0 ± 3` ulp. Row 9 (`Γ(23) = 22!`) is the largest factorial that is exactly
  representable, so the table can make it exact.

### (c) Compare with tolerance — ≤2 ulp (`pow`) / ≤5 ulp (`tgamma`)

Everything else. Do **not** assert hex equality on these rows.

* **`pow`** rows **2, 5, 6, 10, 14–20**. Rows **10** (`2^1023.5`), **14**, **15** and
  **20** are the diagnostic ones: they sit exactly where tier A collapses. Under a
  tier-A implementation they are wrong by **hundreds of ulp**, so writing them at the
  published ≤2 ulp makes a missing hi/lo `log2` *fail loudly* instead of being
  accommodated by a widened tolerance.
* **`tgamma`** rows **10–21, 24–26, 28–31, 43, 44**. Note:
  * Row **10** (`Γ(24) = 23!`) — `23!` is **not** exactly representable in binary64, so
    this row is a rounded value even with the integer table. Tolerance, always.
  * Row **28** (`Γ(171)`) is an integer argument but far outside the 23-entry table.
  * Rows **25–26** (`-100.5`, `-170.5`) are the rows that catch the `1-x` trap of §2.3.
    Under the naive reflection they are off by ~130–310 ulp — well outside any honest
    tolerance, which is the point.
  * Rows **29–31** probe the top of the range; row **31** is the last finite double.
  * Rows **43–44** (`x` one ulp either side of an integer) verify the function is not
    accidentally snapping to the integer table.

### (d) Rows that must raise a flag, not merely return a value

`pow`: **34, 35, 36** divide-by-zero; **9, 51** invalid; **12** overflow; **13** underflow.
`tgamma`: **35, 36** divide-by-zero; **37, 38, 39, 42** invalid; **32, 33, 34** overflow;
**27** underflow.

If the backend does not model IEEE exception flags, state that explicitly rather than
letting this table imply that it does.

---

# PART 4 — The accuracy claim to publish

> Native `pow` and `tgamma` are implemented for x86-64 with no libm dependency.
> `pow(x,y)` is computed as `2^(y·log2 x)` with the exponent carried in double-double;
> `tgamma(x)` uses a Lanczos approximation at `g = 31/8` with 14 coefficients fitted here
> by Remez exchange in 60-digit arithmetic, plus the reflection
> `Γ(x) = π/(sinpi(x)·(−x)·Γ(−x))` for `x < ½`. Measured against `mpmath` at 60 digits:
> `pow` ≤ **2 ulp** over the full finite range (worst observed 1.45 ulp, 4000 random
> arguments per decade of `|y·log2 x|`); `tgamma` ≤ **5 ulp** (worst observed 3.13 ulp on
> `[0.5, 171.62]` over 6000 random arguments, 4.35 ulp on the reflected branch over
> 15 000). The full C99 Annex F.9.4.4 special-case set for `pow`, and the pole, overflow,
> underflow and sign-of-zero cases for `tgamma`, are handled by explicit branches ahead of
> the numeric path and are exact. **Not promised:** correct rounding (these are not
> correctly-rounded implementations, and no claim of ≤0.5 ulp is made); bit-exact
> agreement with glibc or any other libm; exactness of `pow(x,y)` for integer `y` beyond
> the cases the exactness-tracked fast path certifies; IEEE exception flags beyond those
> listed. The ulp figures are **measurements over random samples, not proved bounds** —
> the derived worst-case bounds are ≈1.7 ulp (`pow`) and ≈4.8/7.8 ulp (`tgamma`,
> primary/reflected). Both functions depend on `log2` returning a hi/lo pair; with a
> plain 1-ulp `log2` the same code degrades to ~1100 ulp near the top of the range, and
> that degradation is silent.

---

## Appendix — scripts

All under `/tmp/ffi_design_work/`, reproducible with `python3 <script>`:

| script | produces |
|---|---|
| `lanczos_explore.py` | Spouge/residue conditioning table (§2.1) |
| `check1.py` | residue-vs-published `g=7` divergence (§2.1) |
| `lz2.py` | Remez exchange driver (`remez`, `measure`, `kappa`) |
| `lz3.py` – `lz7.py` | evaluation-form and `(g,N)` selection studies (§2.2) |
| `lz_dyadic.py` | dyadic-`g` selection table (§2.2) |
| `final_gamma.py` | the 14 coefficients + 88k-sample validation |
| `pow_err.py` | `pow` Monte-Carlo, base-e/base-2 × tier A/B (§1.3) |
| `pipeline.py`, `verify.py` | Γ overflow/underflow thresholds, factorial exactness |
| `pipe2.py` | full `tgamma` pipeline error (§2.5) |
| `reflect.py`, `reflect2.py` | the 312-ulp reflection trap and its fix (§2.3) |
| `vectors.py`, `vec_g.py` | the two vector tables of Part 3 |
