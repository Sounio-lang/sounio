<!-- docs:meta
topic_id: repo.docs.research.cd-tower-zd-fiber-l2-reduction-spec-2026-08-01
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.cd-tower-zd-fiber-l2-reduction-spec-2026-08-01
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# CD-tower ZD fibers — L2 reduced to a fiber-free statement about the τ-discrepancy of σ

**Date:** 2026-08-01
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `DIAMOND_IS_A_FINITE_STABLE_FAMILY__GAP_LOCUS_INCLUDED`
**Parents:** `cd_tower_zd_fiber_l2_switching_spec_2026-07-31.md`, `cd_tower_zd_fiber_l1_reduction_spec_2026-07-31.md`
**Harness:** `scripts/research/cd_tower_zd_fiber_l2_reduction_contract.py`

---

## 0. The result

**L2 is not proven.** It is replaced by a smaller statement, in the same move that worked for L1.

With the **τ-discrepancy of the cocycle**

```
g(x,y) = σ(τx, τy) · σ(x,y)
```

and `j = lsb(Y)`, `p_j(x)` = parity of the bits of `x` below `j`:

> **(♦)** For even-weight `Y`, and `a ≠ 0`, `b ≠ 0`, `b ≠ Y`:
> `Qgen'(Y,a,b) = −1` ⟹ `g(a,b) · g(a⊕Y, b⊕Y) = (−1)^{p_j(a) + p_j(b)}`

No fiber, no top bit, one level down — the same shape (★) has for L1. Measured at levels 5, 6, 7 (`N6`), zero violations in 212 712 checks — and the **reduction to
it is Lean-proven ∀n** (`l2_reduction`), so (♦) is the only measured link in L2's chain.

The previous rung had already removed the cohomology from L2 by writing λ in closed form. This
one removes the fiber.

---

## 1. The chain, verified link by link

| clause | link | result |
|---|---|---|
| `N4` | the fiber-level discrepancy **is** `g(a,b)·g(b⊕Y,a⊕Y)` at one level down — via the `R_ll`/`R_uu` branch reductions | **Lean-proven ∀n** (`l2_reduction`); the clause pins it |
| `N1` | `g` is **symmetric**, so the argument swap in `N4` disappears | 0 violations, 122 880 entries — **and already Lean-proven ∀n** |
| `N5` | the reduced resonance predicate is the proven Lean lemma `Qred_hi_ll`: `Qgen(Y+H,a,b,n+1) = −Qgen'(Y,a,b,n)` | 0 violations, 566 208 checks |
| `N6` | (♦) itself | 0 violations, levels 5,6,7 |

`N1` is not new work. `g(x,y) = g(y,x)` is equivalent to `chi(τx,τy) = chi(x,y)` for the
commutation sign `chi(x,y) = σ(x,y)σ(y,x)`, and that is **`chi_tau`, proven ∀n**. It earns its
keep at exactly one step: `R_uu` returns its arguments **swapped**, so the raw reduction gives
`g(b⊕Y, a⊕Y)`, and `gdisc_symm` is what turns that into `g(a⊕Y, b⊕Y)`.

**`N4` is proven too, not measured.** `l2_reduction` and `l2_reduction_symm` in
`formal/lean4/SounioZDFiberAntisym.lean` are kernel-checked ∀n: four branch reductions plus
`tau_seam`/`tau_xor`, with `b ⊕ Y ≠ 0` as `R_uu`'s branch condition — and it governs **both**
sides, since `τ(b⊕Y) = 0 ↔ b⊕Y = 0` by `tau_inj`. So **`L2 ⟸ (♦)` is a theorem** and (♦) is the
only measured link left. That is strictly better than where L1 stands, whose `K2`/`K3`/`K4` are
genuinely measured and unproven.

---

## 2. What carries the content, and what does not

**`N2` — `g` does not factor, and this is the whole point.** The obvious hope is that `g` is
itself a coboundary, `g(x,y) = μ(x)ν(y)`; then L2 would follow in one line with
`λ(a) = μ(a)μ(a⊕Y)`. The rectangle test refutes it in bulk (52 812 / 122 880). So the coboundary
in (♦) is **created by the pairing along `Y`**, not inherited from `g` — L2's analogue of `K5`.

The probe was worth running and it was worth running *first*: it is cheap, and had it come back
positive the rung would have been a one-liner. It also had to fail. If `g` factored, a λ would
exist for **odd-weight** `Y` too, contradicting the previous rung's triangle obstruction.

**`N3` — the F2-linear route is walled.** `R21` closed the ZD locality lemma by finding the
F2-linear identity behind it. `g` is F2-additive in each argument **only for `j ≤ 2`**; from
`j = 3` it fails in bulk. So that route does not generalise here.

**`N7` — the resonance hypothesis is essential, and this is the structural difference from (★).**
Unrestricted, (♦) **fails** (32 832 / 561 162) — and **every** failure is off resonance, none on
it. So L2 is genuinely a statement *on the resonance graph*, whereas (★) is an unrestricted
identity of the cocycle. That difference is the thing to plan around — and *not* because an induction cannot carry a
hypothesis. `star_step_low`/`star_step_hi` thread `hnd` through all four quadrants and
`star_forall` carries `Y % 2^j = 0` down the whole recursion. It is that (♦)'s hypothesis is
`Qgen'(Y,a,b) = −1`, so an induction must **re-establish that predicate at the reduced level in
each quadrant** — which means knowing how `Qgen'` reduces and whether the sign survives. `N5`
already shows one minus sign hiding in exactly that kind of reduction. Mapping it is where the
next attempt should start.

**`N5` — a pin that caught a real error.** The first draft of this rung wrote the reduced
resonance predicate **without the minus sign** in `Qred_hi_ll`. The failure locus then came out
saying every failure was *on* resonance, contradicting the cross-tabulation that says none is.
The contradiction is what surfaced the dropped sign. The clause now pins the Lean lemma to the
measured object, in the same discipline `K21` established for τ.

---

## 2b. How `Qgen'` reduces, and what that buys — the attack on (♦)

**The map, read off sixteen proven theorem statements** (not measured — `N11` parses the
`.lean`):

| | label low | label high |
|---|---|---|
| **sign** | `+1` | `−1` |
| `Qgen` becomes | `Qgen` | `Qgen'` |
| `Qgen'` becomes | `Qgen'` if `ll`/`uu`, `Qgen` if `lu`/`ul` | same rule |

> **The sign is `−1` exactly when the LABEL is high, and nothing else touches it.** Priming is
> governed separately: from `Q` by the label's half, from `Q'` by whether exactly one of `a, b`
> is upper.

Three consequences.

**(i) `N12` — the even-weight hypothesis IS the parity of the sign flips.** Descending the
resonance predicate from level `n` to level `j+2` flips sign once per level where `Y` is high,
so the accumulated sign is `(−1)^{popcount(Y ≫ (j+2))}`, whatever `a` and `b` do. With
`lsb(Y) = j`, `weight(Y) = 1 + bit_{j+1}(Y) + popcount(Y ≫ (j+2))`, so **even weight is exactly
the statement that the accumulated sign is `−(−1)^{bit_{j+1}(Y)}`**. L2's parity hypothesis is
not an extra condition bolted on; it is the descent's own bookkeeping.

**(ii) `N9` — (♦)'s conclusion is LEVEL-BOUNDED.** `G(Y,a,b) = g(a,b)·g(a⊕Y, b⊕Y)` is invariant
under dropping a level and truncating every argument:

```
G_n(Y, a, b)  =  G_{j+2}(Y mod 2^{j+2}, a mod 2^{j+2}, b mod 2^{j+2})
```

**Proven ∀n** as `G_descend` (0 violations in 1 138 688 checks besides, as a pin). It follows
from a *single* lemma — `gdisc` itself descends:

```lean
gdisc_descend : gdisc j x y (m+2) = gdisc j (x mod H) (y mod H) (m+1)
```

unconditionally, in all four quadrants. The degenerate branches never surface because `R_ul` and
`R_uu` guard on `v = 0` while the `τ` factor guards on `τv = 0` — the **same** condition, by
`tau_inj` — so the two guards fire together and their constants (`1·1` and `(−1)·(−1)`) both
multiply to `1`, which is exactly what `gdisc` is at a zero argument. That is why `N9` has no
degeneracy exceptions: they cancel in pairs.
So (♦) is not a statement about an object that grows with `n`. Its conclusion depends only on
the bottom `j+2` bits, and its target `(−1)^{p_j(a)+p_j(b)}` only on the bottom `j`. The
unbounded direction of (♦) is entirely in the *hypothesis*.

**(iii) `N10` — the hypothesis only does work at `j ≥ 3`.** At the bottom level the defect
`G·T` is **identically `+1` for `j ≤ 2`**: (♦) holds there with no hypothesis at all. From
`j = 3` it does not (192/1024, 1344/4096, 6720/16384 for `j = 3,4,5`, and the same count for
both `Y₀`). That boundary is independently where `N3` finds `g` stops being F2-bilinear — two
different probes landing on the same `j ≤ 2`.

**`N13` — two closed forms tried and refuted.** The defect is neither `−Q'(Y₀,a,b)` nor
`−Q(Y₀,a,b)`. Recorded so they are not retried.

---

## 2c. The attack on (♦): it is a **bounded family**, plus a gap locus

`N9` proved the *conclusion* is level-bounded. The natural next question is whether the
*hypothesis* is too. The sign and priming laws predict it should be:

```
Q'_n(Y,a,b) = (−1)^{popcount(Y ≫ (j+2))} · X_{j+2}(Y₀, a₀, b₀)
X = Q'  if popcount((a⊕b) ≫ (j+2)) is even,  else  Q
```

— the **sign** counts the levels where the *label* is high, the **priming** counts the levels
where exactly *one* of `a, b` is upper.

**Unrestricted this fails in bulk (828 192 / 2 318 336). On the CLEAN locus — no degeneracy at
any level of the descent — it holds exactly: 0 / 633 888.** So the hypothesis descends too, but
only where nothing degenerates on the way down. That is the `K17` phenomenon (★) already met: a
tuple non-degenerate at level `m+2` can reduce to a degenerate one at `m+1`.

**Consequence (`N15`): on the clean locus, (♦) has no reference to `n` at all.** With `ε` fixed
by the even-weight hypothesis (`N12`), it becomes, at level `j+2`:

> for `Y₀ ∈ {2^j, 3·2^j}`, `ε = −(−1)^{bit_{j+1}(Y₀)}`, and for **both** primings `X`:
> `X(Y₀,a₀,b₀) = −ε` ⟹ `D(Y₀,a₀,b₀) = +1`

Checked **exhaustively for `j = 1 … 7`** — every `(a₀,b₀)` at every level up to 9. Zero
violations. The two-parameter `∀n, ∀Y` statement has become a one-parameter family of finite
checks.

**And a quarter of it is already discharged (`N16`).** Of the four cases, `Y₀ = 2^j` with
priming `Q` is **empty at every `j`** — because `Qgen` at a single-bit label is identically `−1`
(`Qgen_pow2`, proven ∀n), so its hypothesis `X = +1` is unsatisfiable.

---

## 2d. The gap locus, closed — and a correction to §2c

§2c framed (♦) on the *clean* locus. That framing was **necessary but not sufficient**, and the
gap locus is what shows it.

First, the clean locus was undercounted. §2c used a blunt proxy — no degeneracy at *any* level.
Running the descent with the **sixteen lemmas' actual side conditions** recovers a lot of it
(9 480 → 24 060 clean at `n = 6`), and the descent law is exact on everything that reaches
bottom (0 violations). But ~57% of hypothesis-satisfying tuples still **block**.

So the right object is not the clean locus. It is the **reachable bottom set**:

```
REACH_j(Y₀) = { (a mod 2^{j+2}, b mod 2^{j+2}) : Q'_n(Y,a,b) = −1,  Y mod 2^{j+2} = Y₀ }
```

Because the **conclusion** *truncates to level `k` for every `k > j`* — `G_trunc`, **proven ∀n**
by iterating `G_descend`, with `xor_mod_two_pow` and `gdisc_trunc` — (♦) is **exactly**

> `REACH_j(Y₀) ⊆ { D = +1 }`

and that has no `n` in it **provided `REACH` stabilises**. It does:

| | `j=0` | `j=1` | `j=2` | `j=3` |
|---|---|---|---|---|
| `\|REACH\|`, `Y₀ = 2^j` | 16/16 | 40/64 | 88/256 | 184/1024 |
| `\|REACH\|`, `Y₀ = 3·2^j` | 16/16 | 64/64 | 160/256 | 352/1024 |

stable from `n = j+4` onward (`j ≤ 2` from `n = 6`, `j = 3` from `n = 7`, all unchanged at
`n = 8`), and **never containing a `D = −1` point**.

**Half of the stabilisation is a theorem** (`Reach_succ` / `Reach_mono`, ∀n, kernel-checked).
`Q'red_low_ll` reduces `Qgen'` at a low label with both arguments low **with no side conditions
at all** — the one unconditional lemma of the sixteen. So a level-`n` witness is *verbatim* a
level-`(n+1)` witness: nothing has to be re-established, and `REACH` can only grow. A monotone
family of subsets of the fixed finite square `[0,2^{j+2})²` has a limit, which is what makes
"`REACH_j(Y₀) ⊆ {D = +1}`" well-posed rather than secretly quantified over `n`.

**The `n = j+4` boundary, tested wider (`N19`).** It was fitted to `j ≤ 3`. It holds at `j = 4`
and `j = 5` too, and it is **sharp**: level `j+3` is strictly smaller, `j+4` attains, `j+5` is
unchanged — and the deficit at `j+3` is **exactly 4** in all six cases. The sizes have closed
forms matching every `j` measured (`j = 0…5`):

```
|REACH(2^j)|   = 24·2^j − 8          |REACH(3·2^j)| = 48·2^j − 32
```

**The cheap proof route is REFUTED (`N20`).** The obvious attempt — take a level-`(j+5)` witness
and truncate it to level `j+4` — fails for **half** of all witnesses (20 544/40 896 at `j = 2`;
45 120/90 048 at `j = 3`). So attainment is a **realizability** statement — the bottom pair is
carried by some *other* level-`(j+4)` witness — not a truncation statement. That is exactly why
`Reach_mono`, which works because a witness is *verbatim* reusable upward, has no downward
counterpart.

**The other half is not proven and is not cheap.** That the limit is *attained* at `n = j+4`
would need every bottom triple reachable at any level to be reachable at `j+4` — i.e. the
hypothesis to survive truncation, which is exactly what blocks (§2d). Measured only,
`j ≤ 3`, `n ≤ 8`. That closes the gap locus in the only sense
that matters here: (♦) — blocked tuples included — is a **finite, `n`-free statement per `j`**.

**The correction.** `N15`'s clean-locus predicate does **not** cover `REACH`: for `Y₀ = 2^j` there
are 8, 16, 32, 64 reachable points outside it at `j = 0,1,2,3`. The blocked tuples land strictly
outside the family §2c described. `REACH`, not that predicate, is the object.

---

## 3. Not claimed

- **L2 is not proven, and neither is (♦).** What *is* proven ∀n is the **reduction**
  (`l2_reduction`) and its symmetry ingredient (`gdisc_symm`/`chi_tau`). (♦) itself is measured
  at three levels.
- **(♦) is not proven.** What is proven of it: the conclusion is level-bounded (`G_descend`),
  the reduction into it is a theorem (`l2_reduction`), its symmetry ingredient is a theorem
  (`gdisc_symm`), and one of the four bounded cases is vacuous by `Qgen_pow2`. What is not:
  the finite family itself. The gap locus is no longer a separate obstacle — `REACH` absorbs it
  — and the step that makes `REACH` *the* content is now a theorem: `G_trunc` (∀n,
  kernel-checked) says (♦)'s conclusion at any level IS its value at level `k`, for every
  `k > j`, so nothing above bit `j+1` can matter. What remains measured: **that `REACH`'s limit is
  attained at `n = j+4`** (its monotonicity, hence the limit's existence, is now proven) and
  **`REACH_j ⊆ {D = +1}`** (checked per `j`, not for all `j`). Those two are the whole of what
  is left of L2.
- **`N14`'s clean/gap split is measured, not proven.** The one-step laws behind it are the
  sixteen proven reduction lemmas; what is measured is that their composition is exact on the
  clean locus.
- **`N12` explains the parity hypothesis; it does not discharge it.** The sign law it rests on
  is proven; the arithmetic is trivial; but (♦) itself still needs the descent of the *object*,
  where the priming alternates.
- **The reduction is weaker than L1's.** (★) dropped the fiber from *both* the hypothesis and
  the conclusion. (♦) drops it from the conclusion, and its hypothesis becomes `Qgen'(Y,a,b) = −1`
  — fiber-free, but a genuine hypothesis, which `N7` shows cannot be discarded.
- **Nothing here is claimed about odd-weight `Y`.** `N8` shows *this* identity fails there. The
  non-existence of any λ remains the previous rung's triangle obstruction: evidence, not a proof.
- **(c) is unchanged in status.** Its (★) leg is discharged in Lean
  (`SounioZDCollapse.Phi_preserves_adj_star`); its L2 leg is this. **(c) is still open.**
- **V1 is untouched.**

---

## 4. Reproduce

```sh
python3 scripts/research/cd_tower_zd_fiber_l2_reduction_contract.py
```

Nine clauses, `N0`–`N8`, single verdict token. Runs in well under a second.
