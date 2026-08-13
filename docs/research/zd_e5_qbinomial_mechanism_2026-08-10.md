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

---

## 11. S4 closed form — the seam coboundary is a step function (2026-08-10, step 4)

Probe: `scripts/research/zd_e5_seam_s4_probe.py` (PASS, `m` through 9 on the cocycle column).

### 11.1 The vector `s`

At `W = 2^m`, `H = 2^{m+1}`, `V* = {1,…,H−1}`:

\[
s_b \;=\;
\begin{cases}
+1 & \text{if } 1 \le b \le W, \\
-1 & \text{if } W < b \le H-1.
\end{cases}
\qquad\bigl(\,=\, \mathbf{1}_{b\le W} - \mathbf{1}_{b>W}\,\bigr)
\]

| law | statement at `W = 2^m` | status |
|---|---|---|
| **S0** | `P3(0,0) = +1` | tip: `P3_zero_zero` |
| **S1** | `P3(b,b) = −1` for `b ≠ 0` | tip: `P3_diag` |
| **S2** | `P3(0,b)·P3(b,0) = −1` for `b ≠ 0` | tip: `P3_col0_eq_neg_row0` |
| **S4** | `P3(0,b) = s_b` for `b ∈ V*` | **CLOSED FORM** (below); Lean via tip reduce + one column |
| **S3** | `P3(b,c) = s_b s_c` for `b ≠ c` in `V*` | MEASURED `m=3..7`; = multiplicative S4 |

### 11.2 Reduction of S4 to one cocycle column

The tip already has (`P3_row0_reduce`, every label):

\[
P3(0,b) \;=\; -\,\sigma(W,b)\qquad\text{at level }m+1,\; b\neq 0.
\]

So S4 is exactly

\[
\sigma(2^m,\, b)_{m+1}
\;=\;
\begin{cases}
+1 & b=0 \text{ or } b > 2^m, \\
-1 & 1 \le b \le 2^m.
\end{cases}
\]

**Proof sketch (case split on the CD recursion; uses tip `R_ul` / `R_uu`).**
For `m ≥ 1` write `2^m = 0 + 2^m` as the seam lift of `0` at level `m+1 = (m−1)+2`:

| case | branch | value |
|---|---|---|
| `b = 0` | `R_ul 0 0` | `+1` |
| `0 < b < 2^m` | `R_ul 0 b` → `−σ(0,b) = −1` | `−1` |
| `b = 2^m` | `R_uu 0 0` | `−1` |
| `b = v + 2^m`, `v > 0` | `R_uu 0 v` → `σ(v,0) = +1` | `+1` |

That is the whole of S4.  Kernel-ready once written against the tip's `cdSigma` / `R_ul` / `R_uu`
(this lane does not append to the tip while it is claimed elsewhere).

### 11.3 S3 is the Gram form of the same `s`

With S4 in hand, S3 is the statement that the nonzero principal submatrix is the rank-one form
of row-0, corrected on the diagonal to `−1`:

\[
P3(b,c) \;=\; s_b\, s_c \qquad (b\neq c),\qquad P3(b,b) \;=\; -1.
\]

Equivalently (multiplicative S4): `P3(0,b)·P3(0,c) = P3(b,c)` for `b ≠ c` in `V*`.
Measured through `m = 7`.  Lean path: evaluate `P3_red` at `W = 2^m` under the same
half-split, or derive from empty two-graph / coboundary counting already in the tip's Tier 65
neighbourhood.  Not yet a theorem.

### 11.4 What (S) still is

Once S0–S4 are theorems, the diagonal switch `diag(1,s)` yields the constant matrix of §3 and
the arithmetic

\[
\operatorname{tr}(M^3) \;=\; H^3 - 12 H^2 + 28 H - 16
\]

is free `Int` (same shape as L4).  So **(S) = S3 as a theorem + one arithmetic block**.
S4 is no longer an unknown shape — it is a named cocycle column with a four-line proof.

### 11.5 Evidence

```bash
.venv/bin/python scripts/research/zd_e5_seam_s4_probe.py
.venv/bin/python scripts/research/zd_e5_qbinomial_base_probe.py
.venv/bin/python scripts/research/zd_e5_ref_gap_probe.py
(cd formal/lean4 && lake build SounioZDE5Inductive)   # L4 still green
```

### 11.6 Lean landing plan (when tip claim is free)

| lemma | content | depends on |
|---|---|---|
| `cdSigma_pow2_col` | column formula of §11.2 | `R_ul`, `R_uu`, `cdSig0` |
| `P3_pow2_row0` | S4: `P3(0,b)=s_b` at `W=2^m` | `P3_row0_reduce` + `cdSigma_pow2_col` |
| `P3_pow2_block_coboundary` | S3 off-diag | `P3_red` + half-split |
| `s3_maximal_seam` | (S) | S0–S4 + `tr` arithmetic |

Do **not** open a parallel copy of `cdSigma` on this lane while the tip owns the names; land as a
tier in `SounioZDFiberAntisym.lean` when free.

---

## 9. Reference-side anatomy, measured (kimi, 2026-08-13)

First measurements of the reference matrix `M = P3(·,·,1,m)` against the seam structure of
Tiers 134–136.  **Everything in this section is a finite measurement** (reproducible:
`scripts/research/zd_e5_reference_anatomy_probe.py`, same definitions as the Lean file),
NOT Lean-proved; the only in-kernel piece is the row-0 law (Tier 138).

**Row 0 (PROVED, Tier 138).** `s_x = (−1)^{popcount x}` except at the seam point `x = 1`,
where `s_1 = +1`. As a level recursion: lo half level-independent, hi half sign-flipped,
seam translate excepted.

**The defect set.** The coboundary `P3(a,b)·s_a·s_b = 1` FAILS at `W = 1` on exactly
`24·[m−1,2]₂` unordered pairs — measured at m=2,3,4 (0, 24, 168), so the formula is a
three-point fit, not a theorem. Reproduce: `python3 scripts/research/zd_e5_reference_anatomy_probe.py`. The
level-(m) defect set contains two embedded copies of the level-(m−1) one (lo half, and its
translate by `2^m`) plus a new mixed layer (120 pairs at m=4). An exact recursive rule for
the mixed layer is not yet isolated.

**The diagonal of M³.** At `W = 1`: `(M³)_00 = 32 − 10H` (m=3..9; ⚠ an earlier draft of
this section said `−H² + 2H` — that was WRONG, it holds only at m=2 by coincidence;
caught by the swarm's DIAGONAL lane).  The nonzero diagonal is NOT constant: at m=4 it
takes three values — `+48` on `{1, 16, 17} = {W, 2^m, 2^m⊕W}`, `−48` on `{8, 9, 24, 25}`
(the `2^(m−1)` labels and their translates), `−176` on the remaining 24. At m=3: two
values, `{1,8,9}` at `+16`, the rest at `−16`.  The full rule (2-adic valuation classes)
is in §10.

**Trace decomposition.** Write `M = P + E` with `P` the entry-law matrix (the matrix the
seam theorems would predict: `s_a s_b` off-diagonal, `−1` diagonal, `−s_a` column 0) and
`E` the defect correction. Then, measured exactly at m=2,3,4:

    tr(P³) = H³ − 12H² + 28H − 16   — the seam polynomial, from the entry laws ALONE
    Δs3    = 3·tr(EP²) + 3·tr(PE²) + tr(E³)
           = −2880 + 1152 + 0        (m=3, total −1728 = −1728·[3,3]₂)
           = −52416 + 40320 − 13824  (m=4, total −25920 = −1728·[4,3]₂)

So the whole deviation law is the interaction of the defect set with the predicted matrix;
the m=3 case has NO defect triangles (`tr(E³) = 0`), m=4 has plenty. The mixed terms do
NOT vanish — any proof of the base case must evaluate all three, or find the cancellation
between them. The per-term numbers do not split as clean multiples of `[m,3]₂` on their own.

---

## 10. The reference side, mapped by the swarm (kimi + 8 lanes, 2026-08-13)

Eight parallel measurement lanes, each verified at m=2..6 (some to m=9); raw reports in
`.tmp/e5_swarm/`.  **All statements below are measured exactly, not Lean-proved** — but
they are set-level exact (zero mismatches on every entry/pair/triple at every level
computed), and they turn E5's base case into a finite list of counting obligations.

### 10.1 The defect set has an exact rule (DEFECT-RULE + DEFECT-RECURSION lanes)

At `W = 1`, with `s_x = (−1)^popcount(x)` off the seam point (Tier 138), the coboundary
fails on exactly `24·[m−1,2]₂` unordered pairs, and membership is decidable by a
strip-depth parity: on reduced variables `A = a≫1, B = b≫1`, repeatedly delete the common
binary prefix plus first differing bit; **defect iff the number of deletions is even**
(equivalent computational and recursive forms verified).  Set-level recursion verified:
level m = level-(m−1) embedded + its translate by 2^m + a mixed layer that is a
block-of-4 blow-up of a complement-of-Q construction.  Structural identity on the mixed
block: `M(a, b′+2^m) = cd_sigma(b′⊕1, a, m)·cd_sigma(b′, a⊕1, m)` (argument order matters).

### 10.2 The defect matrix E is a pure coboundary flip with a normal form (DEFECT-ALGEBRA)

`E = M − P` satisfies `E(a,b) = −2·s_a·s_b` exactly on defect pairs (amplitude never 4),
`E = D(−2A₀)D` with `D = diag(s_x)`, rank_Q(E) = `2(2^{m−1}−1)`, GF(2) adjacency rank
`2^m−4`.  Each pair of distinct nonzero c-orbits carries exactly 8 edges, which
**structurally derives** the count `8·C(2^{m−1}−1,2) = 24·[m−1,2]₂`.  New sub-lemma
(verified k=2..6): `cd_sigma` is antisymmetric on distinct nonzero arguments.

### 10.3 The three deviation terms close SEPARATELY (DEVIATION-TERMS lane)

    tr(EP²) = −(H−4)(H−6)(H−8)
    tr(PE²) = +(H−4)(H−8)(H−12)
    tr(E³)  = −(9/7)(H−4)(H−8)(H−16)   [= −48·T3 with T3 the defect-triangle count]

verified m=2..6, with the polynomial identity `3tr(EP²) + 3tr(PE²) = −18(H−4)(H−8)` and
total `−(9/7)(H−2)(H−4)(H−8) = −1728·[m,3]₂` identically.  Per defect pair the
contributions are linear in H — the counting-proof target.

### 10.4 The defect graph and its triangles (TWO-GRAPH lane)

Triangle sign law (0 failures): `M(a,b)M(b,c)M(c,a) = Pprod·(−1)^k`, k = #defect edges.
The defect graph on `V* = {x : x mod 2^m ≥ 2}` (|V*| = H−4) is connected, regular of
degree `d = 2^m − 4`, and has `T3 = 288·[m−1,3]₂` defect triangles.  The deviation is
fully combinatorial: **`Δ = −36·|V*|·d − 48·T3`**, i.e. a degree term plus a
defect-triangle term.  (Refuted en route: "negative triples = triples containing a
defect edge"; the 864·[m,3]₂ of §44 is the ORDERED net switched-triple count,
`net = 144·[m,3]₂` unordered.)

### 10.5 The diagonal rule (DIAGONAL-RULE lane)

`K_a = (M³)_aa` at W=1 is governed by the 2-adic valuation of `a` (or `a−1` for odd a):
master formula `K_a = (8u² − 6Hu + 18H − 80)/3` with `u = H/2^j`, verified at m=2..9.
Generic class (a mod 8 ∈ {2..7}, 3H/4 labels): `−(H−8)(H−10)/3`.  `K_0 = 32 − 10H`
(corrects §9).  M is NOT symmetric at W=1 — diagonal computations need `Σ_b (M²)_ab·M_ba`.

### 10.6 The seam-vs-reference difference (DIFF-MATRIX lane)

`M_ref = ε·M_seam·ε + C` with `ε` an explicit sign vector (`ε_x = −1` on
`{x ∈ [2,2^m] : popcount odd} ∪ {x > 2^m : popcount even}`) and `C` supported exactly on
the defect entries.  Since the conjugated term preserves the trace, **the entire
deviation is attributable to the defect core C**.  No signed-permutation conjugation
between the two matrices exists for m ≥ 3 — the E5 deviation IS the obstruction.

### 10.7 The subspace reading (S3-RECURSION lane)

    s3(1,m)  = −384·[m,3]₂ + 48·[m,2]₂ − 32·[m,1]₂
    s3(2^m)  = 1344·[m,3]₂ + 48·[m,2]₂ − 32·[m,1]₂     [= poly(H)]

— the dim-1 and dim-2 subspace weights are IDENTICAL at reference and seam; **the entire
deviation is a dim-3-subspace weight shift of exactly −1728**.  Also: s3(1) satisfies
`s3(m+1) = 8·s3(m) − 24H² + 264H − 464`, and the deviation recursion
`D(m+1) = 8·D(m) + 1728·[m,2]₂` via q-Pascal — a provable-by-induction skeleton.

### 10.8 The proof program, now finite

1. Defect rule in Lean (Tier-family): the strip-depth parity or the level recursion
   (§10.1), by induction through the cd_sigma top-bit branches.
2. Degree regularity `d = 2^m − 4` on V* and the triangle count `T3 = 288·[m−1,3]₂`
   (counting from the defect rule).
3. `Δ = −36|V*|d − 48·T3` (two-graph reduction), or equivalently the three separate
   trace forms (§10.3).
4. `tr(P³) = poly(H)` (the predicted matrix's trace — currently only measured).
5. Assembly: s3(1) = poly(H) − 1728·[m,3]₂ = −384[m,3]₂ + 48[m,2]₂ − 32[m,1]₂.
