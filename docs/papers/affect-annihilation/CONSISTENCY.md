---
title: "Consistency of A1–A9, and why they are not yet enough"
status: result — model exhibited; and a gap identified
authority: canonical
created: 2026-08-31
---

# Consistency of the axiom system, and the gap it exposes

Two results, one positive and one negative. The negative one matters more.

## 1. A4′ was false. Corrected to A4″.

The first draft restricted monotonicity to arguments in $X^\ast$ but said nothing about the
*results*. That is inconsistent with A8 (annihilation is non-vacuous), and the counterexample is
immediate in the model below:

$$c = 0.5,\quad x = 0.5,\quad y = 0.4 \;\Longrightarrow\; x \succ y \ \text{ but } \
x \circ c = z \prec 0.9 = y \circ c.$$

Monotonicity fails exactly when one composition lands on the annihilator. **A4″ additionally
requires both results to lie in $X^\ast$.** Found by attempting to construct a model, not by
inspection — which is the argument for building models early.

## 2. A1–A9 are consistent. A model.

Let

$$X = (0,1) \cup \{z\}, \qquad z \succ r \ \text{ for all } r \in (0,1), \qquad
\succsim \ = \ \ge \ \text{ on } (0,1),$$

> ⚠️ **Corrected 2026-08-31.** This first said $z \prec r$ — $z$ at the *bottom* — while the
> model annihilates at the *ceiling*. Two of my own documents disagreed on where $z$ sits, and
> nothing in the axiom list settled it. A2′ now fixes $z$ as the strict **maximum**.

and define $\circ$ by, for $a, b \in (0,1)$:

$$a \circ b \;=\;
\begin{cases}
a + b & \text{if } a + b < 1\\[2pt]
z & \text{if } a + b = 1\\[2pt]
\text{undefined} & \text{if } a + b > 1
\end{cases}
\qquad\qquad z \circ x \;=\; x \circ z \;=\; z .$$

Verification, axiom by axiom:

| axiom | holds because |
|---|---|
| A1, A2′ order | $z$ is the strict **maximum**; $\ge$ orders $(0,1)$ |
| A2 nontriviality | $0.6 \succ 0.3$ |
| A3 local definability | $w \preceq x,\ z' \preceq y \Rightarrow w + z' \le x + y \le 1$; and any composition with $z$ is defined |
| **A4″** monotonicity | on $x+c<1,\ y+c<1$: $x \ge y \iff x+c \ge y+c$ |
| A5 restricted solvability | $x \succ y$: take $u \in (0, x-y)$, then $y \circ u = y+u < x$ |
| A6′(i) positivity off $z$ | $a+b > a$ and $a+b > b$, both summands positive |
| A6′(ii) absorption | by definition |
| A7 Archimedean | for $x>0$ there is $n$ with $nx > 1$, hence undefined |
| **A8** annihilation non-vacuous | $0.5 \circ 0.5 = z$, both $\ne z$ |
| **A9** annihilation is thin | for fixed $a$, only $b = 1-a$ annihilates |

**The system is consistent.** A8 and A9 are not vacuous, and they coexist with monotonicity once
A4″ is stated correctly.

Note the contrast with the nilpotent t-norms (Ling 1965; Mostert & Shields 1957): the
Łukasiewicz operation $\max(0, x+y-1)$ annihilates on an **open region** — every pair with
$x+y \le 1$. Here annihilation is confined to the **line** $x+y = 1$. That is exactly A9, and it
is what keeps the structure out of the regime where associativity becomes obligatory.

## 3. 🔴 The gap: A1–A9 do not force what the paper is about

**The witness above is associative wherever it is defined** — it is addition. Therefore:

> **A1–A9 are consistent, but they admit an associative model. They do not force
> non-associativity, and non-associativity is the entire empirical claim.**

The axiom system as stated is too weak. Annihilation and non-associativity are independent, and
we have only axiomatised the first. What is missing is an axiom that *forbids* associativity, and
it must be **observable** — a grouping effect:

**A10 — Grouping.** There exist $a, b, c \in X^\ast$ such that $(a \circ b) \circ c$ and
$a \circ (b \circ c)$ are both defined and $(a \circ b) \circ c \nsim a \circ (b \circ c)$.

This is the axiom that carries the empirical content, and it is the one the experiment must test.
Everything else is scaffolding.

## 3.1 🟢 RESOLVED — A1–A10 are consistent

The gap of §3 is closed. A model satisfying **all ten axioms**, including non-associativity.

**Construction.** Start from the unit form of Cohen & Narens (1979, Thm 3.3),
$a \circ b = b\,f(a/b)$, where monotonicity in both arguments is equivalent to *$f$ strictly
increasing and $f(t)/t$ strictly decreasing*, and positivity to *$f(t) > 1$ and $f(t) > t$*.
Take

$$f(t) \;=\; t + 1 + \varepsilon\,\frac{t}{1+t}, \qquad \varepsilon > 0 .$$

All three conditions hold **globally**: $f'(t) = 1 + \varepsilon/(1+t)^2 > 0$;
$(f(t)/t)' = -1/t^2 - \varepsilon/(1+t)^2 < 0$; and $f(t) > t+1 > \max(1,t)$. The induced
operation is

$$\boxed{\;a \circ b \;=\; a + b + \varepsilon\,\frac{ab}{a+b}\;}$$

on $X = (0,1) \cup \{z\}$, with the ceiling: the value if it is $< 1$; $z$ if it equals $1$;
undefined if $> 1$. And $z$ absorbing.

**Why each axiom holds** — analytically, not by grid:

| axiom | argument |
|---|---|
| A4″ monotonicity | $\partial_a(a\circ b) = 1 + \varepsilon b^2/(a+b)^2 > 0$, symmetrically in $b$ |
| A6′(i) positivity | $\varepsilon > 0 \Rightarrow a \circ b > a + b > a, b$ |
| A3 local definability | the domain $\{a \circ b \le 1\}$ is downward closed, by monotonicity |
| **A9 thinness** | $\circ$ strictly increasing in $b$ $\Rightarrow$ **at most one** $b$ per $a$ with $a \circ b = 1$ |
| **A8 non-vacuity** | $a \circ a = a(2 + \varepsilon/2) = 1$ at $a^\ast = 1/(2+\varepsilon/2) \in (0,1)$ |
| A7 Archimedean | $\circ$ exceeds addition, so $n a$ leaves $(0,1)$ |
| **A10 grouping** | exhibited below |

**The non-associativity witness** ($\varepsilon = 1$; $\circ$ is homogeneous of degree 1, so the
witness may be scaled into $(0,1)$ freely):

$$a = \tfrac19,\quad b = \tfrac29,\quad c = \tfrac39
\;\Longrightarrow\;
(a \circ b) \circ c = 0.924 \;\neq\; 0.895 = a \circ (b \circ c),$$

both defined, both below the ceiling. Verified in
`tests/run-pass/modelo_a1_a10.sio` (0 monotonicity violations, 0 positivity violations,
0 local-definability violations, at most **1** crossing of the ceiling per $a$, and
$a^\ast \circ a^\ast = 1.000$ exactly).

**Note on $\varepsilon$.** At $\varepsilon = 0$ the operation degenerates to addition — the
associative witness of §2. So $\varepsilon$ is exactly the *grouping* parameter, and A10 holds
iff $\varepsilon \neq 0$. That is a convenient handle: the empirical question becomes the
estimation of a single parameter whose null value is the associative model.

## 3.2 🔴 Three axiom defects found by adversarial reading — and repaired

An adversarial pass over the axiom list (2026-08-31) found three defects **before** any question
of representation. All three are now repaired in AXIOMS.md; recorded here because they change
what the earlier sections of this document claimed.

**A — Unrestricted local definability contradicted the witness.** With $z$ absorbing, $z \circ x$
is defined for every $x$; with $z$ the maximum, every $w \preceq z$, so the unrestricted axiom
forces $\circ$ **total**. The model has $0.9 \circ 0.9$ above the ceiling and undefined. The
axiom list contradicted the model it was supposed to admit. **Repair: A3′, stated for
$\circ^\ast$.**

**B — "The interior is a PCS" does not follow.** Countermodel: on $(0,1)$ with $a \circ b = a+b$,
pick an interior pair with $a_0 + b_0 < 1$ and **decree** $a_0 \circ b_0 := z$, keeping thinness.
All of A1–A9 as first listed still hold, yet $(a_0+\delta) \circ b_0 \in X^\ast$ while the
smaller pair annihilates — the domain of $\circ^\ast$ is not downward closed and Narens & Luce
does not apply. **This also holed the rigidity proposition**, which rests on the same claim.
**Repair: A9′, coherence.** It is *necessary*, by the countermodel — the obstruction it negates
is **interior annihilation**.

**C — The position of $z$ was never postulated**, and §2 above placed it at the bottom while the
model annihilates at the ceiling. **Repair: A2′, $z$ is the strict maximum.**

**The witness satisfies coherence**, and for a structural reason worth isolating: its annihilation
set is a **level curve of a strictly increasing function**. Verified in
`tests/run-pass/coerencia_da_testemunha.sio` — 3240 pairs tested by raising both arguments from an
annihilating pair, **0 violations**, and 0 violations of the strict growth that causes it.

## 3.3 The existence theorem, now statable

With the repaired system — call it **PCS$_z$**: $z$ strict maximum and absorbing, annihilation
reached and thin, NL1976 Axioms 2–7 stated for $\circ^\ast$, plus coherence — the existence
proof is **Narens & Luce (1976) Theorem 2.1 applied to $X^\ast$ verbatim**, plus a one-point
extension.

> **Conjecture (existence).** Every coherent PCS$_z$ has a representation in a numerical
> PCS$_z$ $\langle R \cup \{\zeta\}, \ge, \hat\otimes\rangle$, unique given the target
> (by triviality of the automorphism group, §6.2 of AXIOMS.md).

Two things this reframes:

1. **The target cannot satisfy NL1976 Def. 2.2.** A reached absorbing $\zeta$ violates positivity
   and strict monotonicity, exactly as $X$ does. The requirement *"the target is a PCS"* descends
   one level: it is required of the target's **interior**.
2. **The extension to $z$ is by FIAT, not by completion or continuity.** The Dedekind completion
   of NL1976 §7 **cannot** produce $z$: its cuts are proper by definition (no top is ever added),
   and Theorem 7.4(i) proves the completion **is a PCS** — and PCSs exclude absorbing elements by
   positivity. Continuity fails too: in the witness, $\lim_{a \to 1^-} a \circ b > 1$, so the
   continuous prolongation is *undefined*, not $z$. Calling it "continuity" would sell a
   canonicity that does not exist.

The remaining work is half a page of routine lemmas: well-definedness of $\hat\otimes$ on the
fibre of $z$ (needs coherence + $z$ strict maximum), descent of thinness to the target, and
absorption in the target.

## 4. What is now open

1. ~~Is A10 consistent with A1–A9?~~ **Resolved in §3.1: yes, model exhibited.**
2. **The representation theorem** (AXIOMS.md §6.1) is now the target, with the
   automorphism-coupling obstruction still unresolved — see AXIOMS.md §6.2 for the missing lemma
   (invariance of the annihilation boundary, $\alpha(g(a)) = g(\alpha(a))$, replacing Luce's
   solvability coupling).
3. **Does the model of §3.1 satisfy the boundary-conjugation equation?** It is the obvious first
   test case: its annihilation boundary is $b = g(a)$ with $a + b + \varepsilon ab/(a+b) = 1$,
   and $\circ$ is homogeneous of degree 1, so dilations are candidate automorphisms — but the
   ceiling breaks homogeneity. Whether any non-identity automorphism survives is checkable and
   not yet checked.

**No theorem is claimed in this document.** §2 is a verification; §3 is a gap; §4 is a programme.
