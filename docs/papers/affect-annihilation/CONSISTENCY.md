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

$$X = (0,1) \cup \{z\}, \qquad z \prec r \ \text{ for all } r \in (0,1), \qquad
\succsim \ = \ \ge \ \text{ on } (0,1),$$

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
| A1 weak order | $z$ is the minimum; $\ge$ orders $(0,1)$ |
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

## 4. What is now open

1. **Is A10 consistent with A1–A9?** A model satisfying all ten is required. The witness of §2
   cannot be extended: addition is associative. Whether thin annihilation and non-associativity
   can coexist under monotonicity **is not known to me, and I have not found it in the
   literature**. This is the first real theorem to attack.
2. If they cannot coexist, the project is over in its present form, and the honest report is that
   thin annihilation *forces* associativity — which would itself be a publishable negative
   result, and a sharp one.
3. If they can, the representation theorem of AXIOMS.md §6.1 becomes the next target, with the
   automorphism-rigidity obstruction (Cohen & Narens 1979, Thm 2.1) still unresolved.

**No theorem is claimed in this document.** §2 is a verification; §3 is a gap; §4 is a programme.
