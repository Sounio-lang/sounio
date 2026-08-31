---
title: "Annihilation in affective composition: a partial concatenation structure with a singular zero"
status: working draft — axioms only, no theorem claimed
authority: canonical
created: 2026-08-31
---

# Annihilation in affective composition

**Working draft.** This document states an axiom system and identifies precisely what must be
proved. **No representation theorem is claimed here.** Section 6 lists what is conjecture.

## 0. What is new, and what is not

Not new, and cited on page one of any paper built on this:

- **Representation without associativity** is Narens & Luce (1976), *The algebra of measurement*,
  J. Pure Appl. Algebra **8**, 197–233 — *positive concatenation structures* (PCS), with a
  uniqueness theorem. Also Cohen & Narens (1979), JMP **20**, 193–232.
- **Scale-type classification via the automorphism group** is Luce & Narens (1983, 1985).
- **Singular points and absorbing elements** are Luce (1992), *Singular points in generalized
  concatenation structures*, Math. Social Sciences **24**, 79–103.
- **Hypercomplex algebra as the structure of mind** was claimed by Goertzel (1996), with the
  *opposite* premise: he argues consciousness is a **division** algebra (no zero divisors).
  This work is the explicit negation of that thesis and must engage it, not ignore it.

The claim here is narrower: **an empirical composition operation that admits annihilation** —
two non-null states whose composition yields no resultant — and the axioms under which such a
structure still admits a numerical representation.

## 1. The empirical structure

Let $X$ be a set of *affective episodes*, $\succsim$ a weak order on $X$ read as "at least as
disruptive as", and $\circ$ a **partial** binary operation read as *composition of episodes*
(temporal concatenation, or co-presentation).

Let $z \in X$ be a distinguished element: the **null resultant** — the outcome of a composition
that produces no movement. Empirically, $z$ is not "neutral affect"; it is the *absence of a
resultant* from a composition whose inputs were both non-null.

> **Annihilation.** $a \circ b = z$ with $a \neq z$ and $b \neq z$.

This is the object. It is *not* the cancellation of opponent-process theory (Solomon & Corbit,
1974) nor destructive interference in quantum-cognition models: both are cancellation **by sum**,
$x + (-x) = 0$, which exists in any group and is not annihilation. Annihilation requires a
**product** of non-invertible elements, which exists only outside division algebras. This
distinction belongs in the abstract.

## 2. Axioms

Adapted from Narens & Luce (1976, Def. 2.1). Axioms 1–3, 5, 7 are theirs unchanged. **The
modification is confined to Axioms 4 and 6.** Write $X^\ast = X \setminus \{z\}$.

For all $w, x, y, z' \in X$:

**A1 — Weak ordering.** $\succsim$ is connected and transitive.

**A2 — Nontriviality.** There exist $u, v \in X$ with $u \succ v$.

**A3 — Local definability.** If $x \circ y$ is defined, $x \succsim w$ and $y \succsim z'$, then
$w \circ z'$ is defined.

**A4″ — Monotonicity off the singular point.** For all $x, y, c \in X^\ast$:
(i) if $x \circ c$ and $y \circ c$ are defined **and both lie in $X^\ast$**, then
$x \succsim y \iff x \circ c \succsim y \circ c$;
(ii) if $c \circ x$ and $c \circ y$ are defined **and both lie in $X^\ast$**, then
$x \succsim y \iff c \circ x \succsim c \circ y$.

> ⚠️ **Corrected 2026-08-31.** The first draft of this axiom quantified only over the
> *arguments* being in $X^\ast$, and is **false**. Counterexample in the model of §3.1:
> $c = 0.5$, $x = 0.5$, $y = 0.4$ give $x \succ y$ but $x \circ c = z \prec 0.9 = y \circ c$.
> Monotonicity fails precisely when one composition lands on the annihilator, so the *results*
> must be restricted too. Found by attempting to build a model — which is what models are for.

> *This is the whole modification.* Narens & Luce require monotonicity on all of $X$. Strict
> monotonicity and an annihilator are **logically incompatible**: if $a \circ z = z = b \circ z$
> with $a \succ b$, monotonicity demands $a \circ z \succ b \circ z$. Restricting to $X^\ast$ is
> the minimal weakening. It is licensed by Luce (1992): $z$ is a **singular point**, fixed by
> every automorphism, and Luce & Narens (1994, n. 13) already allow monotonicity to need "some
> care … in dealing with extreme points, if such exist".

**A5 — Restricted solvability.** If $x \succ y$ then there is $u$ with $x \succ y \circ u$.

**A6′ — Positivity off the singular point, and absorption at it.**
(i) If $x \circ y$ is defined and $x \circ y \neq z$, then $x \circ y \succ x$ and $x \circ y \succ y$.
(ii) If $x \circ y = z$ then $z \circ x = z \circ y = z$ (**absorption**).

> Positivity is known to be dispensable (Narens & Luce 1976 §3, intensive structures; Luce &
> Narens 1985 Thm 5.1), so weakening it is cheap. What is *not* cheap is A4′, and the paper must
> say so.

**A7 — Archimedean.** There is $n \in \mathbb{I}^+$ such that either $nx$ is undefined or
$nx \succsim y$, where $1x = x$ and $nx = [(n-1)x] \circ x$.

**A8 — Annihilation is non-vacuous.** There exist $a, b \in X^\ast$ with $a \circ b = z$.

**A9 — Annihilation is thin.** For every $a, b \in X^\ast$ with $a \circ b = z$ and every
$c \in X^\ast$ with $c \neq b$, if $a \circ c$ is defined then $a \circ c \neq z$.

> A9 is the empirical content of a measurement made independently of this axiom system: in the
> sedenion model the zero-divisor set is a **measure-zero variety**, and any admixture along the
> aligned direction leaves it (verified in `tests/run-pass/psi_producao_vs_prereg.sio`). A9 is
> the order-theoretic shadow of that fact — annihilation is a **boundary**, not a region. It is
> also what keeps the structure out of the t-norm regime (Ling 1965; Mostert & Shields 1957),
> where annihilation fills an open region and **associativity becomes obligatory**.

## 3. Why partiality costs nothing

$\circ$ is partial in Narens & Luce from 1976 — A3 is native, not a concession. The domain may
be declared to exclude compositions that would annihilate, and the 1976/79 machinery then applies
verbatim on $X^\ast$. The annihilator enters afterwards as a Dedekind completion point (NL 1976
§7 gives the algebraic conditions: tightness, interval solvability, regularity). Precedent for
extensive measurement with restricted concatenation and maximal elements: Luce & Marley (1969).

## 4. The two-layer reading, and where it is licensed

If $X$ carries two descriptive layers — what the subject **reports** and what is **measured** —
then a norm-multiplicative algebra on each layer separately forbids annihilation *within* a
layer, by Hurwitz. Annihilation can then only occur **across the interface**.

This is a theorem about the algebra, not a finding about people: it is Hurwitz's theorem, and it
must be presented as such. Its interpretive content is exactly one sentence, and no more:
**failure of coupling is not a property of any layer; it is a property of the relation between
layers.**

Everything else previously drafted around this — that alignment "protects" coupling, that
interoceptive concordance is the mechanism — is corollary and projection. Alignment is not
orthogonality; that is geometry, not psychology.

## 5. The graded observable

The event $a \circ b = z$ has measure zero and is therefore not directly measurable. The
pre-registered dependent variable must be **graded**:

$$\delta(a,b) \;=\; \lVert a \rVert\,\lVert b \rVert \;-\; \lVert a \circ b \rVert$$

$\delta \ge 0$, and $\delta = 0$ exactly in the Hurwitz regime. Annihilation is the boundary case
$\lVert a \circ b \rVert = 0$, not an event to be detected.

⚠️ $\delta$ is defined on the numerical side. Stating it as the dependent variable **presupposes
the representation theorem that Section 6 says is unproved.** Until then $\delta$ is a modelling
commitment, not an observable — and the paper must not blur that.

## 6. What is NOT proved here

1. **The representation theorem.** Whether ⟨$X$, $\succsim$, $\circ$, $z$⟩ satisfying A1–A9
   admits a homomorphism into a normed algebra with zero divisors. **Open.** The obstruction to
   watch: A4′ removes monotonicity exactly where Cohen & Narens (1979, Thm 2.1) get automorphism
   rigidity, and that rigidity is the engine of *all* non-associative uniqueness. If the singular
   point breaks the fixed-point trichotomy, uniqueness fails.
2. **Uniqueness, and the anchor it will need.** Likely requires homogeneity under translations
   **off** the null set, in the manner of Luce (1992). Unproved.
3. **That the target algebra must be 16-dimensional.** Nothing above forces $\mathbb{S}$. The
   dimension is a separate argument and is not made here. Orbit counts (168, 336, 1848) are
   **symmetry-forced** and carry no empirical content; their only legitimate place is kernel
   validation.
4. **Any dynamical claim.** There is no equation of motion. Without a specified flow there is no
   rate, no regime, and no "time near the variety". All dynamical language is barred until a flow
   is written.
5. **Identifiability.** No published work decides between competing composition algebras from
   finite behavioural data. There is no ready defence against *"16 dimensions and a 14-dimensional
   automorphism group are unidentifiable"* — the first objection any JMP referee will raise. It
   must be answered in the paper, with counted degrees of freedom and a discriminating test.

## 7. What must be frozen before data

Not the states — the **operations**. A pre-registration that fixes $\psi$ but leaves the
operation→composition mapping free leaves all post-hoc freedom intact. The design must fix, a
priori: three operations, two parenthesisations, the mapping from empirical operation to algebraic
product, and the model-comparison procedure against the associative class with degrees of freedom
counted.

## 8. Open ground

> **Non-associative structures with weak monotonicity and a region of annihilation.**

No theorem and no counterexample published. Connects to Problems 9 and 11 of Luce & Narens
(1994). The present system deliberately stays *outside* it (A9 makes annihilation thin), because
inside it associativity appears to become obligatory. Whether that appearance is a theorem is
itself open.
