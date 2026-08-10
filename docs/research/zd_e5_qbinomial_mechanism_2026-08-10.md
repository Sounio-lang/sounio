<!-- docs:meta
topic_id: repo.docs.research.zd-e5-qbinomial-mechanism-2026-08-10
authority: research-active
audience: researchers
last_validated: 2026-08-10
-->

# E5 — where the q-binomial actually lives

Edge **E5** of the completeness pincer (`docs/research/zd_completeness_pincer_dag_2026-08-10.md`)
is obligation (iii) of §57.50: the base case of the unmasked deviation law

\[
s3(2^m) - s3(1) \;=\; 1728 \cdot [m,3]_2
\qquad (H = 2^{m+1}).
\]

Measured in the source note for `m = 3..7`. Reconfirmed here through **`m = 8`**
(`scripts/research/zd_e5_qbinomial_base_probe.py`).

This note records a **correction of seat** and a **proof-shaped reduction** of the
maximal-seam half. It does **not** claim E5 proved, and it does not touch the
load-bearing Lean tip (claimed by another lane).

---

## 1. The headline finding

| end | closed form | q-binomial? | status |
|---|---|---|---|
| **maximal seam** `W = 2^m` | \(s3 = H^3 - 12 H^2 + 28 H - 16\) | **none** | MEASURED `m=3..8`; forced by a matrix model (below) |
| **reference** `W = 1` | \(s3 = H^3 - 12 H^2 + 28 H - 16 - 1728\cdot[m,3]_2\) | **all of E5** | MEASURED `m=3..8` |
| **difference** | \(1728\cdot[m,3]_2 = \dfrac{9}{7}(H-2)(H-4)(H-8)\) | identity | algebraic, since \([m,3]_2 = (H-2)(H-4)(H-8)/1344\) |

**The Gaussian binomial of E5 sits in the g = 0 reference `W = 1`, not in the
maximal seam.**

This rewrites a slogan the lane has been using correctly for *other* objects and
incorrectly for E5:

- **Still true:** the maximal seam is where \([m-1,2]_2\) enters the *orthant pieces*
  `T1, T2, T3` and the quadratic form `Q` (192·[m−1,2]₂, 96·[m−1,2]₂, …). Same
  *location*, different *content*.
- **False for unmasked `s3`:** “E5’s q-binomial is a seam feature.” Absolute `s3` at
  the seam is a pure polynomial in `H`. The gap is the reference falling short of
  that polynomial by \(1728\cdot[m,3]_2\).

The two q-binomials remain non-proportional:

\[
[m,3]_2 = [m-1,2]_2 \cdot \frac{2^m - 1}{7}.
\]

---

## 2. Numbers (independent recompute)

| m | H | s3(2^m) | s3(1) | Δ | 1728·[m,3]₂ | poly seam |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 16 | 1456 | −272 | 1728 | 1728 | 1456 |
| 4 | 32 | 21360 | −4560 | 25920 | 25920 | 21360 |
| 5 | 64 | 214768 | −53072 | 267840 | 267840 | 214768 |
| 6 | 128 | 1904112 | −506448 | 2410560 | 2410560 | 1904112 |
| 7 | 256 | 15997936 | −4411472 | 20409408 | 20409408 | 15997936 |
| 8 | 512 | 131086320 | −36797520 | 167883840 | 167883840 | 131086320 |

`m = 8` is out of the source range `m = 3..7` and matches.

`cp2` at both ends equals `−(H−2)(H−6)` (E4a / Tier 109 class), as expected for
`g = 0`.

---

## 3. Matrix model at the maximal seam (why the polynomial is forced)

Write `M = P3(·,·, W=2^m, m)` as an `H × H` matrix on indices `{0,1,…,H−1}`, and
`V* = {1,…,H−1}`, `n = H−1`.

**Structural laws, verified `m = 3..7`:**

| # | law | lean foothold |
|---|---|---|
| S0 | `M_00 = +1` | `P3_zero_zero` |
| S1 | `M_ii = −1` for `i ≠ 0` | `P3_diag` |
| S2 | `M_0b · M_b0 = −1` for `b ≠ 0` | `P3_col0_eq_neg_row0` (Tier 112) |
| S3 | `B := M|_{V*×V*}` is symmetric, and `B = s sᵀ − 2 I` for some `s ∈ {±1}^n` | `P3_pow2_coherent` / empty two-graph on `V*` (Tier 65 covers triples avoiding the seam *vertex* in a related formulation; the block statement is the counting form needed here) |
| S4 | **alignment:** row-0 on `V*` equals `s` | **not yet isolated as a named theorem** |

Consequences of S3–S4:

- On `V*`, every oriented triple product is `+1` (empty two-graph / coboundary).
- After the diagonal switch `diag(1, s)` one obtains the *constant* matrix

\[
M' \;=\;
\begin{pmatrix}
1 & \mathbf{1}^{\mathsf T} \\
-\mathbf{1} & J - 2I
\end{pmatrix}.
\]

- For a general border vector `v ∈ {±1}^n` with `p = ∑ v_i`, block arithmetic gives

\[
\operatorname{tr}((M'_v)^3) \;=\; n^3 - 6n^2 + 7n + 1 - 3p^2.
\]

- Alignment S4 forces `v = 1`, hence `p = n`, hence

\[
\operatorname{tr}(M'^3) \;=\; n^3 - 9n^2 + 7n + 1
\;=\;
H^3 - 12 H^2 + 28 H - 16.
\]

Similarity by a diagonal `±1` matrix preserves the cubed trace, so this is `s3(2^m)`.

**S4 in multiplicative form** (useful for Lean): for `b, c ∈ V*`, `b ≠ c`,

\[
P3(0,b)\, P3(0,c) \;=\; P3(b,c)
\qquad\text{at } W = 2^m.
\]

That is the Gram/coboundary statement: the nonzero principal submatrix is the
rank-one form of row-0, corrected on the diagonal to `−1`.

---

## 4. What E5 still is

After this reduction, E5 is **exactly one** of the following (equivalent given the
seam form):

1. **Prove the seam polynomial** (S0–S4 + arithmetic) **and** the reference form
   `s3(1) = poly(H) − 1728 [m,3]_2`.
2. **Prove the seam polynomial** and a direct difference argument that never needs
   the absolute reference form.
3. **Prove the reference form alone** against the already-measured seam polynomial
   (weaker as a programme: the seam is the structured end).

The residual combinatorial content — the part that still deserves the name
“q-binomial mechanism” — is therefore:

> Why does the g = 0 reference `W = 1` fall short of the maximal-seam polynomial
> by exactly \(1728 \cdot [m,3]_2 = \frac{9}{7}(H-2)(H-4)(H-8)\)?

Classical reading: `[m,3]_2` counts 3-dimensional `𝔽₂`-subspaces of `𝔽₂^m`.
The factor `1728 = 2 · 864` matches §44’s identity that the base-level
`y = 0` graph has `864 · [j,3]_2` positive triangles — so E5’s gap is

\[
\Delta s3 \;=\; 2 \times \bigl(\text{positive-triangle count of the } y=0 \text{ base graph at } j = m\bigr),
\]

which is consistent with, but does not yet prove, a double-cover or two-orientation
count between the unmasked triple sum and that graph. §44 already showed that
chasing this equality *inside* the curvature chain is circular; the way out is
still the `σ`-recursion / structural matrix evaluation, not further rearrangement
of `K = −1 − Bp_T`.

---

## 5. Routes already closed (do not reopen)

From the v1 reduction arc, recorded so this lane does not re-spend them:

| route | outcome |
|---|---|
| edge-level bijection for the base residual | closed |
| triangle-level bijection | closed |
| high-branch twin of §18.1 | refuted |
| complementation third invariant | refuted |
| spectral fold for general seam box | `y = 0` only; general box unclassified (§57.2–57.3) |
| `W ↦ W ⊕ 2^m` conjugation collapsing the base family | **refuted** (entrywise agreement exactly `H²/2`) |
| “extract ε_T, then count” (§43.3) | circular (§44) |
| treat Q’s seam gap as E5 | **refuted**: `Q` gap is `96·[m−1,2]_2`, not proportional to E5 |

---

## 6. Relation to the live pincer (E4b / Q)

Tiers 110–116 reduced the transfer of `Q` and the T3 leg to the single obligation

\[
B \;=\; -Q + 8H - 12
\]

(with `quadSplit` a theorem). That obligation is **not** E5. Closing it gives
`Q(m+1) = 16H − 28` above the level where the label exists; the base level of a
label is still where absolute values (and E5) live.

Recommended parallelism, not collision:

- **E4b / `B` lane** (claimed elsewhere): finish the transfer.
- **E5 lane (this note):** prove S0–S4 ⇒ seam polynomial; then attack `s3(1)` or the
  difference with the subspace-counting reading in hand.

---

## 7. Lean targets (named, not written here)

The Lean tip `formal/lean4/SounioZDFiberAntisym.lean` is under another active claim.
Targets for a future tier, once the claim is free:

1. `P3_pow2_block_coboundary` — S3: on `V*`, `P3(a,b) = s_a s_b` off-diagonal at `W = 2^m`.
2. `P3_pow2_row0_align` — S4: `P3(0,b) = s_b` (same `s`).
3. `s3_maximal_seam` — `tr(M^3) = H^3 - 12 H^2 + 28 H - 16` at `W = 2^m`.
4. `s3_ref_gap` / `e5_base` — the residual.

Items 1–3 are finite sign algebra plus one arithmetic identity. Item 4 is the
open mathematics.

---

## 8. Evidence commands

```bash
.venv/bin/python scripts/research/zd_e5_qbinomial_base_probe.py
```

Expect: E5 exact on `m=3..8`, structural laws on `m=3..7`, exit code 0.

---

## 9. Claim discipline

- This note and the probes are claimed under lane `zd-e5-qbinomial-20260810`.
- Do **not** append these findings into `cd_tower_zd_fiber_v1_reduction_spec_2026-07-31.md`
  or the pincer DAG while those paths remain under another agent’s claim; merge by
  handoff when free.
- No novelty claim relative to external literature is made here. The internal
  correction (“E5’s q-binomial sits at `W = 1`”) is a lane-memory fix, not a paper
  headline. The paper headline remains spectral completeness (§58.1).

---

## 10. Residual attack — `s3(1)` by transfer + induction (2026-08-10, step 2)

Probe: `scripts/research/zd_e5_ref_gap_probe.py` (PASS).

### 10.1 Package

| tag | statement | status |
|---|---|---|
| **(T)** | `s3(m+1,1) = 8·s3(m,1) + 24·cp2(m,1) − 176 + 72·H` | MEASURED (E4b / obligation (i); not yet Lean) |
| **(C)** | `cp2(m,1) = −(H−2)(H−6)` | **PROVED** E4a / Tier 109 (`g = 0`) |
| **(B)** | `s3(3,1) = −272` | finite, brute-verified (`H = 16`) |
| **(S)** | `s3(m,2^m) = P(H) := H³ − 12H² + 28H − 16` | MEASURED + model (§3); not yet Lean |
| **(I)** | `F(m) := P(2^{m+1}) − 1728·[m,3]_2` satisfies (T)+(C) | **PROVED as arithmetic** |
| **(II)** | `F(3) = −272` | **PROVED** (one-line) |

**Corollary (arithmetic):** `(T)+(C)+(B)+(I)+(II)` ⇒ `s3(m,1) = F(m)` for all `m ≥ 3`.

**Corollary (E5):** that + `(S)` ⇒ `s3(2^m) − s3(1) = 1728·[m,3]_2` for all `m ≥ 3`.

So E5 is no longer an open shape: it is **four CD obligations** (of which one is already a
theorem) plus **finished arithmetic**.

### 10.2 The induction step, factored

With `H = 2^{m+1}` the step `F(m) ↦ F(m+1)` under (T)+(C) is equivalent to

\[
8\,P(H) - P(2H) + 24\bigl(-(H-2)(H-6)\bigr) + 72H - 176
\;=\;
1728\bigl(8[m,3]_2 - [m+1,3]_2\bigr).
\]

**Left side (poly residual).** Expand:

\[
8P(H)-P(2H) = -48H^2 + 168H - 112,
\]
\[
24\bigl(-(H-2)(H-6)\bigr) + 72H - 176 = -24H^2 + 264H - 464,
\]
\[
\text{sum} = -72H^2 + 432H - 576 = -72(H-2)(H-4).
\]

**Right side (Gaussian residual).** Using

\[
[m,3]_2 = \frac{(H-2)(H-4)(H-8)}{1344},\qquad
[m+1,3]_2 = \frac{(2H-2)(2H-4)(2H-8)}{1344} = \frac{8(H-1)(H-2)(H-4)}{1344},
\]

\[
8[m,3]_2 - [m+1,3]_2
= \frac{8(H-2)(H-4)}{1344}\bigl((H-8)-(H-1)\bigr)
= -\frac{(H-2)(H-4)}{24}
= -[m,2]_2.
\]

Hence

\[
1728\bigl(8[m,3]_2 - [m+1,3]_2\bigr)
= 1728\cdot\bigl(-[m,2]_2\bigr)
= -72(H-2)(H-4).
\]

**Equal.** The q-binomial’s recursive content in E5 is exactly the Gaussian Pascal fragment

\[
8\,[m,3]_2 - [m+1,3]_2 \;=\; -[m,2]_2,
\]

matched to the poly residual of `P` under the transfer’s eigenvalue-8 channel.

### 10.3 Why this is the mechanism (and what it is not)

- The factor `8` is the transfer eigenvalue for `s3` (homogenised within a fibre once
  `Δcp2 = 0`). It is **not** `2³` from orthant counting of a doubling of the *seam*
  label — the seam label changes with `m` and cannot ride the fixed-`W` transfer.
- The q-binomial enters because `F` subtracts `1728[m,3]_2` from a polynomial that
  *almost* obeys the inhomogeneous transfer; the defect is `−72(H−2)(H−4)`, which is
  proportional to `[m,2]_2`, and the Gaussian identity upgrades that to the `[m,3]_2`
  recurrence.
- This does **not** prove E5. It proves: *if* (T), (C), (B), (S) hold, *then* E5 holds
  for all `m ≥ 3` by induction. (C) is already a theorem. (B) is a 16³ enumeration.
  (S) and (T) are the remaining CD content.

### 10.4 Base case `(B)`, explicit

At `m = 3`, `H = 16`:

\[
P(16) = 4096 - 3072 + 448 - 16 = 1456,
\qquad 1728\cdot[3,3]_2 = 1728,
\qquad F(3) = 1456 - 1728 = -272.
\]

Direct `tr(M³)` on the `16×16` matrix `P3(·,·,W=1,m=3)` equals `−272` (brute and
`numpy` agree).

### 10.5 Bonus observation (not load-bearing)

On the whole `g = 0` class at fixed `m`, `s3` is **not** constant: the powers of two
`W = 2^j` interpolate between `F(m)` (for `j` small / `W < 8`) and `P(H)` (for
`j = m`). In particular `W ∈ {1,…,7}` all share the value `F(m)` at every `m` tested.
So the E5 reference can be taken as any low octonion label, not only `W = 1`.

### 10.6 Lean targets (updated)

| # | statement | depends on | status |
|---|---|---|---|
| L1 | `s3_maximal_seam`: (S) | S0–S4 of §3 | open (CD) |
| L2 | transfer of `s3` at fixed `W` (at least `W = 1`) | E4b orthant closed forms | open (CD) |
| L3 | `s3_ref_base`: `s3(3,1) = -272` | `decide` / small enumeration | finite; `F_base` / `F7_base` recorded |
| L4 | `e5_inductive_form`: `F` closed under (T)+(C) | pure `Int` arithmetic | **LANDED** |
| L5 | `e5_base`: assemble L1–L4 ⇒ E5 ∀ `m ≥ 3` | L1–L4 | open |

**L4 location:** `formal/lean4/SounioZDE5Inductive.lean` (own Lake target
`SounioZDE5Inductive`; does **not** import the tip — parallel-safe with the E4b claim).

Form used: the 7-cleared integer

\[
F_7(H) \;=\; 7\,P(H) - 9(H-2)(H-4)(H-8)
\]

(so \(F_7 = 7F\) whenever the product formula for \([m,3]_2\) is exact). Kernel-clean theorem:

\[
F_7(2H) \;=\; 8\,F_7(H) + 7\cdot\bigl(24(-(H-2)(H-6)) - 176 + 72H\bigr)
\qquad(\forall\, H\in\mathbb{Z}).
\]

Also: `poly_residual`, `gauss_residual_cleared`, `residuals_match`, tower specialisation
`e5_inductive_form_tower`. Axioms of the free-`H` theorems:
`[propext, Classical.choice, Quot.sound]`. Base constants use `native_decide`.

### 10.7 Evidence commands

```bash
.venv/bin/python scripts/research/zd_e5_qbinomial_base_probe.py   # seat + seam model
.venv/bin/python scripts/research/zd_e5_ref_gap_probe.py          # residual induction
# Lean L4 (toolchain: formal/lean4/lean-toolchain; elan provides `lake`)
(cd formal/lean4 && lake build SounioZDE5Inductive)
# or direct:
# lean formal/lean4/SounioZDE5Inductive.lean
```

### 10.8 Residual after L4 (2026-08-10, step 3)

E5 is now:

> **Prove (T) and (S); invoke (C), (B), L4.**

| obligation | role | seat |
|---|---|---|
| **(T)** | transfer of `s3` at `W = 1` | E4b orthants → absolute `s3` at fixed label |
| **(S)** | seam poly `s3(2^m) = P(H)` | S0–S4 of §3; alignment S4 is the open structural pin |
| **(C)** | `cp2` on g=0 | E4a / Tier 109 **PROVED** |
| **(B)** | `s3(3,1) = −272` | finite; `F_base` |
| **L4** | `F` closed under (T)+(C) | **PROVED** (`e5_inductive_form`) |

Recommended next attack on this lane (still avoiding the tip claim): deepen S0–S4 as a
**probe + named lemmas note**, or wait for the tip claim to free and land L1 as a tier there.
Do not reopen closed v1 routes (§5).
