<!-- docs:meta
topic_id: repo.docs.research.nullity-histogram-law-spec-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.nullity-histogram-law-spec-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Nullity-histogram counting law — exact multiplicities of the canonical ZD nullity spectrum

**Date:** 2026-07-26  
**Status:** `HYPOTHESIS` → `EXECUTABLE` (C_GREEN reached 2026-07-26)  
**Parents:** `docs/research/routon_zd_spec_2026-07-26.md` (2-cycle criterion, nullity spectrum, the open question solved here), `scripts/research/routon_zd_contract.py` (sign table, exact scan)  
**Harness:** `scripts/research/nullity_histogram_law_contract.py`  
**Gate:** `scripts/ci/nullity_histogram_law_gate.sh`

---

## 1. What this is

The routon contract tabulated the level-7 nullity histogram of the canonical zero divisors,

```
{4:684, 8:504, 12:504, 16..44:336 each, 48:504, 52:504, 56:684, 60:870}
```

and explicitly disclaimed an explanation of the multiplicities ("Not an explanation of the odd-part distribution", §4 of the parent). This contract **solves that open question**: the multiplicities are given by an exact combinatorial law, derived from first principles (the 2-cycle criterion, the fiber structure, and the sign-table block recursion), verified exhaustively at levels 4–7, and **confirmed out of sample at level 8** against the independently tabulated L8 census (whose own spec still listed the multiplicity law as open).

**The law (terminal-level form).** At every level `b ≥ 4` the nullity histogram has exactly `b − 3` distinct multiplicities. For each **terminal level** `s ∈ {4, …, b}`, exactly `2^(b−s)` distinct nullity values occur with multiplicity

```
μ_s(b) = 2^(b−s+1) · c⁰(s−1),   where   c⁰(b) = 3·(2b−3)·2^(b−2) + 3
```

is the number of **invertible** canonical candidate pairs at level `b` (c⁰ = 21, 63, 171, 435, 1059 at b = 3..7). At level 7:

| s | values `2^(7−s)` | multiplicity `μ_s` | nullities |
|---|---|---|---|
| 4 | 8 | `2^4·c⁰(3) = 16·21 = 336` | 16, 20, 24, 28, 32, 36, 40, 44 |
| 5 | 4 | `2^3·c⁰(4) = 8·63 = 504` | 8, 12, 48, 52 |
| 6 | 2 | `2^2·c⁰(5) = 4·171 = 684` | 4, 56 |
| 7 | 1 | `2·c⁰(6) = 2·435 = 870` | 60 |

`8·336 + 4·504 + 2·684 + 1·870 = 6942 = Z(7)/2` ✓. The multiplicities are distinguished by 2-adic valuation: `v2(μ_s) = b − s + 1`, with odd part `c⁰(s−1)`.

**The law (per-pair form).** Let `N(m, b, t)` = number of canonical ZD index pairs at level `b`, born at level `m` (xor-label in `(2^(m−1), 2^m)`), with nullity `2^(b−m+2)·t`, `t` odd, `1 ≤ t ≤ 2^(m−3)−1`. Then

```
N(m, b, t) = 2^(b−m) · N(m, m, t),
N(m, m, t) = 2^(V+1) · c⁰(m_s − 1),
```

where `(V, m_s)` is the **2-adic descent** of `t`: iterate

```
(m, t) ↦ (m − v, (2^(m−3)−1−t)/2^v),   v = v2(2^(m−3)−1−t),
```

until `t = 2^(m−3)−1` (the maximal odd part); `V` is the accumulated valuation and `m_s = m − V` the terminal level. The descent terminates because `m` strictly decreases and at `m = 4` the only allowed value is `t = 1 = max`. Since `v2(nullity) = b − m + 2` determines `m`, and then `t`, the per-pair form and the terminal-level form are equivalent; the descent gives a bijection between allowed `(m, t)` and pairs `(m_s, λ)` with `λ` a composition of `V = m − m_s` (parts ≥ 1), which is why `2^(m−m_s−1)` values of `t` in class `m > m_s` share terminal `m_s` (and one, `t = max`, in class `m = m_s`).

Example (`m = 7, t = 5`, nullity 20 at L7): `max = 15`, `u = 10`, `v = 1` → `(6, 5)`; `u = 2`, `v = 1` → `(5, 1)`; `u = 2`, `v = 1` → `(4, 1) = max`: `V = 3`, `m_s = 4`, `N = 2^(7−7+3+1)·c⁰(3) = 16·21 = 336` ✓.

**Level-8 confirmation (out of sample).** The law's level-8 prediction — `μ_s(8)` for `s = 4..8`: 672×16, 1008×8, 1368×4, 1740×2, 2118×1 distinct nullity values, total `29886 = Z(8)/2` with `Z(8) = 59772` — matches the independently computed L8 census of `docs/research/l8_zd_census_benchmark_spec_2026-07-26.md` **exactly** (contract C5; that census is itself audited by full `GF(65521)` rank computations on all 64770 pair-signs, and this contract additionally re-verified the law against the raw exact L8 scan, `NULLITY_LAW_L8_EXACT=1`). The law thereby survives its first out-of-sample test, on data tabulated *before* the law was stated. It also explains the L8 spec's "extremes do not double" observation: the extreme multiplicities have terminal levels `s = b−1` and `s = b` (e.g. nullity 4 is `(m, t) = (b, 1)`, which descends one step to `m_s = b−1`), so they track `c⁰(b−2)` and `c⁰(b−1)` — 171→435 and 435→1059 from L7 to L8 — rather than doubling. **Level-9 prediction (next falsification target):** 1344×32, 2016×16, 2736×8, 3480×4, 4236×2, 4998×1 values, total `124542 = Z(9)/2`.

---

## 2. Derivation from first principles

Notation: `S(u, v)` = Cayley–Dickson sign of `e_u·e_v` (`cds()` in the harness), level `b`, `h = 2^(b−1)`, `n = 2^b`. For a pair `{i, j}`, `ℓ = i ⊕ j`, and `p(k) = S(i,k)S(j,k)S(i,k⊕ℓ)S(j,k⊕ℓ)`; by the exact 2-cycle criterion (parent contract §2), `nullity(L_{e_i+e_j}) = ½·#{k : p(k) = +1}`. The block recursion of `cds()` gives, with `x = (x0, x1)`, `x0 < h`, `x1 ∈ {0,1}` the top bit:

```
S((x0,0),(k0,0)) = S(x0,k0)          S((x0,0),(k0,1)) = S(k0,x0)
S((x0,1),(k0,0)) = −S(x0,k0) [k0≠0], +1 [k0=0]
S((x0,1),(k0,1)) = S(k0,x0) [k0≠0], −1 [k0=0]
```

### Lemma A (anticommutation and square signs)

For distinct `u, v ≥ 1`: `S(u,v) = −S(v,u)`; for `u ≥ 1`: `S(u,u) = −1`. *Proof:* induction on bits over the four high-bit cases of the block recursion; base `b = 1` is `S(1,1) = −1`. (Also what makes imaginary units imaginary.) ∎

### Lemma L1 (the ε-identity)

For every candidate pair `{i, j}`, `ℓ = i ⊕ j`:

```
ε(i,j) := S(i,ℓ)·S(j,ℓ) = −1.
```

*Proof.* Induction on bits. Write `j = i ⊕ ℓ` and split on the high bits of `i, ℓ`:

- `i, ℓ < h`: then `j < h`, both factors are level-`b` signs, induction hypothesis applies.
- `ℓ < h ≤ i`, `i = h+i0`, `j = h+j0`, `j0 = i0⊕ℓ`: both factors are of type `(x0,1)×(k0,0)`, so `ε = S(i0,ℓ)S(j0,ℓ)`; if `i0 = 0` or `i0 = ℓ` this is `(−1)(+1)` resp. `(+1)(−1)` by Lemma A, else the induction hypothesis applies to `(i0, ℓ)`.
- `i < h ≤ ℓ`, `ℓ = h+ℓ0`: `S(i,ℓ) = S(ℓ0,i)`, `S(j,ℓ) = S(ℓ0,j0)` (or `−1` if `ℓ0 = 0`); by Lemma A, `S(ℓ0,i)S(ℓ0,i⊕ℓ0) = S(i,ℓ0)S(i⊕ℓ0,ℓ0)`, again the induction hypothesis (edge cases `ℓ0 = 0`, `i = ℓ0` give `−1` directly).
- `h ≤ i, ℓ`: `j = i0⊕ℓ0 < h`; `S(i,ℓ) = S(ℓ0,i0)`, `S(j,ℓ) = S(ℓ0,j)`; reduce as in the previous case. ∎

Equivalently `(e_i e_ℓ)e_ℓ = −e_i` for distinct imaginary units — the basis-element instance of the right alternative law, which survives the whole tower. Verified exhaustively at `b = 3..7` (contract L1: 0 violations over 21+105+465+1953+8001 candidate pairs).

### Lemma L2 (left p-function = right p-function)

For `a = e_i + e_j`, right multiplication `R_a` has the same 2-cycle structure with `pR(k) = S(k,i)S(k,j)S(k⊕ℓ,i)S(k⊕ℓ,j)`. For `k ∉ {0, i, j, ℓ}` four applications of Lemma A give `pR(k) = pL(k)`; at each of the four exceptional points both sides evaluate to `−1`. Hence `pL ≡ pR` **pointwise**, and `nullity(L_a) = nullity(R_a)` for every candidate pair. Verified exhaustively (contract L2). ∎

### Lemma L4 (doubling doubles the kernel — embedded and high pairs)

At level `b`, splitting rows and columns at `h`:

- **Embedded pair** (`i, j < h`): `L_a = diag(L_a^(b−1), R_a^(b−1))` (block rows `k < h` close under left multiplication by `e_i, e_j`; rows `k ≥ h` give `e_i e_{h+k0} = S(k0,i)·(0, e_{k0⊕i})`, the right-multiplication block). Hence `nullity_b = ν_L + ν_R = 2ν` by L2.
- **High pair** (`{h+i0, h+j0}`): `L_a` is block-anti-diagonal with blocks `−L_a^(b−1)` and `R_a^(b−1)`, each modified by a sign flip of a single row (`k = 0` resp. `k0 = 0`); single-row sign flips preserve rank. Hence `nullity_b = ν_L + ν_R = 2ν` again.

Pairs with label `< h` never straddle `h` (if `i < h` and `ℓ < h` then `j = i⊕ℓ < h`), so every old-label pair at level `b` is embedded, high, or the always-invertible `{h, h+ℓ}` (inverse `conj(a)/2`, since `a·conj(a) = |a|²` throughout the tower). Verified exhaustively, including invertible pairs (`ν = 0 → 0`) (contract L4). ∎

### Lemma L3 (master native recursion)

A native pair at level `m` with label `ℓ = h + r` (`1 ≤ r ≤ h−1`) has the form `{i0, h + j0}` with `i0 ∈ [1, h) ∖ {r}`, `j0 = i0 ⊕ r` (the excluded `i0 = r` is the fundamental missing pair `{r, h}` of the defect diagonal). Then

```
nullity_m = h − 2·ν − 4,
```

where `ν` is the nullity of `{i0, j0}` at level `m−1` (0 if invertible).

*Proof.* Split `k = (k0, c)`, `c ∈ {0,1}`, and let `p̃(k0) = S(i0,k0)S(j0,k0)S(i0,k0⊕r)S(j0,k0⊕r)` be the level-`(m−1)` p-function of `{i0, j0}` (label `r`). From the block recursion, for `k0 ∉ E = {0, r, i0, j0}`:

```
p((k0, c)) = −p̃(k0)   (c = 0, 1),
```

(one explicit minus from the `(x0,1)` row of the block table; the anticommutations needed to align the arguments come in cancelling pairs by Lemma A). At the 8 exceptional points, direct evaluation gives:

| `k0` | `c = 0` | `c = 1` | generic `−p̃` |
|---|---|---|---|
| `0` | `ε` | `ε` | `−ε` |
| `r` | `ε` | `ε` | `−ε` |
| `i0` | `−1` | `−1` | `+1` |
| `j0` | `−1` | `−1` | `+1` |

using `p̃(0) = p̃(r) = ε` and `p̃(i0) = p̃(j0) = −1` (Lemma A). The correction to `#{k : p = 1}` is therefore `4ε − 4`, and with `P̃ = #{k0 : p̃ = 1} = 2ν`:

```
nullity_m = ½·[ 2(h − P̃) + 4ε − 4 ] = h − 2ν + 2ε − 2 = h − 2ν − 4   (L1: ε = −1). ∎
```

Verified exhaustively at `m = 4..7` (contract L3: 0 mismatches over 42+210+930+3906 = 5088 native pairs).

**Consistency checks.** Positivity: `ν ≤ 2^(m−2) − 4 < (h−4)/2`, so every native candidate pair is a ZD — the fiber-size and defect-diagonal laws of the parent contract. Odd part: if `{i0,j0}` is `m′`-born with nullity `2^(m−m′+1)·t′`, then `nullity_m = 4t` with `t = 2^(m−3) − 2^(m−m′)t′ − 1`, odd (both subtracted terms even), `≥ 1` (since `t′ ≤ 2^(m′−3)−1`), and `≤ 2^(m−3)−1` (with equality iff `ν = 0`). The **odd-part law of the parent contract (its C9 set statement) is thereby proved**, and its completeness clause reduces to positivity of the counting law below.

### Counting

Every candidate pair `P` at level `b−1` — ZD or invertible, nullity `ν` — yields **exactly two native ZD pairs at level `b`** (from `i0` and `j0` as the lower index), both of nullity `2^(b−1) − 4 − 2ν` (L3); and every ZD pair at level `b−1` yields **exactly two old-label pairs at level `b`** (embedded + high), both of nullity `2ν` (L4). With `Z_b(ν)` the level-`b` histogram and `c_b` the full candidate census (`c_b(0)` invertible, `c_b(ν) = Z_b(ν)` for `ν > 0`):

```
Z_b(ν) = 2·Z_{b−1}(ν/2)  +  2·c_{b−1}( (2^(b−1) − 4 − ν)/2 ).
```

Unrolling the second term: a pair born at `m` with odd part `t` descends from a pair at level `m−1` with nullity `ν′ = 2·(2^(m−3) − 1 − t)`; if `ν′ = 0` the line bottoms out in an invertible pair (count `2·c⁰(m−1)`), otherwise `ν′` is `m′`-born with `m′ = m − v2(2^(m−3)−1−t)` and odd part `t′ = (2^(m−3)−1−t)/2^v` — the descent map of §1 — contributing a factor `2^v` per step from doubling (first term). This gives the per-pair law `N(m,b,t) = 2^(b−m+V+1)·c⁰(m_s−1)`.

The invertible census itself is `c⁰(b) = C(2^b−1, 2) − Z(b)/2 = 3·(2b−3)·2^(b−2) + 3`, and the census satisfies `z_b = 2·z_{b−1} + (2^(b−1)−1)(2^(b−1)−2)` with `z_3 = 0` (`z_b = Z(b)/2`), i.e.

```
Z(b)/2 = Σ_{g=4}^{b} 2^(b−g)·(2^(g−1)−1)(2^(g−1)−2),
```

whose closed form is the parent's census law `Z(b) = 4^b − (3b−1)·2^b + 2^(b−1) − 4` (induction on `b`). The growth law, fiber-size law, odd-part law and histogram law are thus all consequences of Lemmas L1–L4.

Summing the per-pair law over the composition bijection (`#t` in class `m` with terminal `s` = number of compositions of `m − s` = `2^(m−s−1)` for `m > s`, and `1` for `m = s`) gives the terminal-level form: `Σ_{m=s}^{b} … = 2^(b−s)` nullity values share the multiplicity `μ_s = 2^(b−s+1)·c⁰(s−1)`, distinct since `v2(μ_s) = b − s + 1`.

---

## 3. Contract clauses

| Clause | Statement | Acceptance gate |
|---|---|---|
| **L1_EPS_IDENTITY** | `S(i,ℓ)S(j,ℓ) = −1` for every candidate pair, `b = 3..7`. | 0 violations over 10545 pairs. |
| **L2_LEFT_RIGHT_NULLITY** | `nullity(L_a) = nullity(R_a)` for every candidate pair, `b = 4..7` (exact 2-cycle counts, integer arithmetic). | 0 mismatches. |
| **L3_NATIVE_RECURSION** | `nullity_m = 2^(m−1) − 2ν − 4` for every native pair, `m = 4..7`. | 0 mismatches over 5088 native pairs. |
| **L4_DOUBLING** | Embedded and high pairs at level `b` have nullity `2ν` (incl. `ν = 0`), `b = 4..7`. | 0 mismatches. |
| **C1_INVERTIBLE_CENSUS** | `c⁰(b) = 3(2b−3)2^(b−2)+3` reproduces the invertible census 21, 63, 171, 435, 1059 at `b = 3..7`. | Exact match. |
| **C2_DESCENT_LAW** | `N(m,b,t) = 2^(b−m+V+1)·c⁰(m_s−1)` reproduces the full per-class and total nullity histograms at `b = 4..7`. | Exact match, all classes. |
| **C3_TERMINAL_STRUCTURE** | Multiplicity `μ_s = 3·2^(b−s+1)((2s−5)2^(s−3)+1)` attained by exactly `2^(b−s)` distinct nullity values, `s = 4..b`, `b = 4..7`. | Exact match. |
| **C4_L7_HEADLINE** | The level-7 histogram is exactly `{4:684, 8:504, 12:504, 16..44:336, 48:504, 52:504, 56:684, 60:870}`, total 6942, reproduced by the law. | Exact match. |
| **C5_L8_OUT_OF_SAMPLE** | The law's level-8 prediction (672×16, 1008×8, 1368×4, 1740×2, 2118×1 values, total 29886) matches the tabulated L8 census; optional `NULLITY_LAW_L8_EXACT=1` re-runs the raw exact L8 scan. | Exact match (both modes verified). |

---

## 4. What this is NOT

- **Confirmed at level 8, not yet beyond.** Lemmas L1–L4 are proved from the block recursion plus the 2-cycle criterion and verified exhaustively at `b ≤ 7`; the derivation itself is level-independent, and the law's L8 prediction is confirmed out of sample (C5). Level 9 (predicted histogram 1344×32, 2016×16, 2736×8, 3480×4, 4236×2, 4998×1 values, `Z(9)/2 = 124542`) is the next falsification target.
- **Not the full zero-divisor locus.** Only canonical 2-unit sums `e_i ± e_j` are counted, as in the parent contracts.
- **Not a G₂ or automorphism statement.**
- **Not a clinical claim.**

## 5. Reproduce

```bash
python3 scripts/research/nullity_histogram_law_contract.py
# expect: L1..L4, C1..C4 PASS, NULLITY_HISTOGRAM_LAW_VERDICT C_GREEN   (~1 s)

bash scripts/ci/nullity_histogram_law_gate.sh
# expect: NULLITY_HISTOGRAM_LAW_GATE_OK
```

Pure Python + NumPy, self-contained; reuses the sign table and exact scan of `scripts/research/routon_zd_contract.py` (no SVD anywhere — all clauses are exact integer arithmetic).

## 6. AI disclosure

Spec and harness drafted under human direction (2026-07-26). No clinical content. GAIDeT-ICMJE 2025.
