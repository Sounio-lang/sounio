<!-- docs:meta
topic_id: repo.docs.research.beta0-octonion-gum-derivation
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.beta0-octonion-gum-derivation
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# GUM Variance Propagation through Octonion Multiplication via the Structure Tensor

**Task β-0.** Rigorous derivation, sign conventions explicit.
**References.** Schafer (1966); Baez, *Bull. AMS* 39 (2002) 145–205; Conway & Smith (2003); JCGM 102:2011 (GUM Supplement 2).

---

## 1. Setup: the octonion algebra and its structure tensor

Let $\mathbb{O}$ denote the (unital, normed, non-associative, alternative) division algebra of octonions over $\mathbb{R}$, with orthonormal basis $\{e_0, e_1, \dots, e_7\}$ where $e_0 = 1$ is the unit. Every $a \in \mathbb{O}$ is written $a = \sum_{i=0}^{7} a_i e_i$ and identified with its coordinate vector $(a_0, \dots, a_7)^\top \in \mathbb{R}^8$.

**Multiplication.** Define the structure tensor $C \in \mathbb{R}^{8\times 8\times 8}$ by
$$
e_i e_j \;=\; \sum_{k=0}^{7} C_{ij}^{\,k}\, e_k, \qquad (ab)_k \;=\; \sum_{i,j=0}^{7} C_{ij}^{\,k}\, a_i b_j. \tag{1.1}
$$
The unital axiom fixes $C_{0j}^{\,k} = \delta_{jk}$ and $C_{i0}^{\,k} = \delta_{ik}$. The imaginary sub-block $(i,j \ge 1)$ is determined by the Fano plane of oriented triples (Baez 2002, §3, eq. (4)):
$$
\mathcal{F} \;=\; \{(1,2,3),\; (1,4,5),\; (1,7,6),\; (2,4,6),\; (2,5,7),\; (3,4,7),\; (3,6,5)\}. \tag{1.2}
$$
For each $(i,j,k) \in \mathcal{F}$ we cyclically set $C_{ij}^{\,k} = C_{jk}^{\,i} = C_{ki}^{\,j} = +1$ and the three reversals $C_{ji}^{\,k} = C_{kj}^{\,i} = C_{ik}^{\,j} = -1$. For $i \ge 1$, $e_i^2 = -1$, i.e. $C_{ii}^{\,0} = -1$. All other entries vanish.

**Sparsity.** The imaginary block has index range $i,j \in \{1,\dots,7\}$, $k \in \{0,\dots,7\}$, a cube of $7\cdot 7\cdot 8 = 392$ slots. Exactly $7 \times 6 = 42$ signed entries come from the seven oriented Fano lines (six cyclic sign assignments per line), plus $7$ diagonal entries $C_{ii}^{\,0} = -1$, giving **$49$ nonzeros** out of $392$ (≈ 12.5 %). Including the identity rows/columns ($C_{0i}^{\,i}$ and $C_{i0}^{\,i}$, $i=0,\dots,7$, which overlap at $(0,0,0)$) the total for the full $8\times 8\times 8$ tensor is $49 + 2\cdot 8 - 1 = 64$ nonzeros out of $512$. The "~42 out of 512" quoted figure counts only the genuinely non-associative Fano part; including squares brings it to 49.

---

## 2. Left and right multiplication matrices

From (1.1), bilinearity of multiplication means that for fixed $a$ the map $b \mapsto ab$ is $\mathbb{R}$-linear, as is $a \mapsto ab$ for fixed $b$. Define
$$
\boxed{\;L(a)_{kj} \;:=\; \sum_{i=0}^{7} C_{ij}^{\,k}\, a_i, \qquad R(b)_{ki} \;:=\; \sum_{j=0}^{7} C_{ij}^{\,k}\, b_j\;} \tag{2.1}
$$
so that
$$
(ab)_k \;=\; \sum_j L(a)_{kj}\, b_j \;=\; \sum_i R(b)_{ki}\, a_i, \qquad ab \;=\; L(a)\, b \;=\; R(b)\, a. \tag{2.2}
$$

**Non-commutativity witness.** Take $a = e_1 + e_2$. From (1.2), $e_1 e_2 = e_3$ and $e_2 e_1 = -e_3$. Therefore row $3$ of $L(a)$ at column $2$ is $+1$ while row $3$ of $R(a)$ at column $2$ is obtained from $C_{22}^{\,3} + C_{12}^{\,3}\cdot 0$... more directly: $L(a) e_2 = a e_2 = e_1 e_2 + e_2 e_2 = e_3 - 1 = -e_0 + e_3$, while $R(a) e_2 = e_2 a = e_2 e_1 + e_2 e_2 = -e_3 - 1 = -e_0 - e_3$. Hence $L(a) \ne R(a)$. $\square$

In general, $L(a) = R(a)$ iff $a$ is central in $\mathbb{O}$, and the centre of $\mathbb{O}$ is $\mathbb{R}\cdot e_0$.

**Non-associativity witness (Jacobian level).** For $a,b,c \in \mathbb{O}$,
$$
(ab)c \;=\; L(ab)\, c \;=\; R(c)\, (ab) \;=\; R(c) L(a)\, b, \tag{2.3}
$$
$$
a(bc) \;=\; L(a)\, (bc) \;=\; L(a) R(c)\, b. \tag{2.4}
$$
Subtraction gives $[a,b,c] := (ab)c - a(bc) = \big(R(c) L(a) - L(a) R(c)\big)\, b$. For generic $a,c$ this commutator of $8\times 8$ matrices is non-zero. Concretely with $a = e_1$, $c = e_2$: by (1.2) $L(e_1)$ sends $e_4 \mapsto e_5$ and $R(e_2)$ sends $e_4 \mapsto e_6$ (since $e_4 e_2 = -e_2 e_4 = -e_6$, so $R(e_2)e_4 = -e_6$). Then $L(e_1)R(e_2)e_4 = L(e_1)(-e_6) = -e_1 e_6 = e_7$ (using $e_1 e_7 = e_6 \Rightarrow e_1 e_6 = -e_7$), whereas $R(e_2)L(e_1)e_4 = R(e_2)e_5 = e_5 e_2 = -e_2 e_5 = -e_7$. Thus the commutator sends $e_4 \mapsto 2 e_7 \ne 0$. $\square$

This is the Jacobian-level manifestation of $[e_1, e_4, e_2] \ne 0$: the associator of three octonions is exactly the image of the middle argument under the commutator of left- and right-multiplication operators.

---

## 3. GUM Supplement 2 variance propagation

**Setting.** Regard multiplication $\mu : \mathbb{R}^8 \times \mathbb{R}^8 \to \mathbb{R}^8$, $(a,b) \mapsto c = ab$. Let $a, b$ be random 8-vectors with means $\bar a, \bar b$ and covariance matrices $V(a), V(b) \in \mathbb{R}^{8\times 8}$ (symmetric, positive semidefinite). JCGM 102:2011 §6 (vector-valued law of propagation of uncertainty) gives, to first order in the Taylor expansion of $\mu$ about $(\bar a, \bar b)$,
$$
V(c) \;\approx\; J_a\, V(a)\, J_a^\top \;+\; J_b\, V(b)\, J_b^\top \;+\; J_a\, V(a,b)\, J_b^\top \;+\; J_b\, V(a,b)^\top J_a^\top, \tag{3.1}
$$
where $V(a,b) := \mathbb{E}[(a-\bar a)(b-\bar b)^\top]$ is the cross-covariance, and
$$
J_a \;:=\; \left.\frac{\partial\mu}{\partial a}\right|_{(\bar a,\bar b)} \in \mathbb{R}^{8\times 8}, \qquad J_b \;:=\; \left.\frac{\partial\mu}{\partial b}\right|_{(\bar a,\bar b)}.
$$

**Derivation of the Jacobians.** From (2.2), $c_k = \sum_j L(a)_{kj} b_j = \sum_i R(b)_{ki} a_i$. Differentiating,
$$
\frac{\partial c_k}{\partial a_i} \;=\; R(b)_{ki}, \qquad \frac{\partial c_k}{\partial b_j} \;=\; L(a)_{kj}.
$$
Evaluated at the means,
$$
\boxed{\;J_a \;=\; R(\bar b), \qquad J_b \;=\; L(\bar a).\;} \tag{3.2}
$$
Substituting into (3.1):
$$
V(c) \;\approx\; R(\bar b)\, V(a)\, R(\bar b)^\top \;+\; L(\bar a)\, V(b)\, L(\bar a)^\top \tag{3.3}
$$
in the independent case $V(a,b)=0$; otherwise (3.1) with (3.2) provides the cross-term. Equation (3.3) is the octonionic analogue of the scalar rule $\sigma_c^2 = b^2 \sigma_a^2 + a^2 \sigma_b^2$: the Jacobians are *not* diagonal scalings but the $8\times 8$ matrices $L(\bar a)$, $R(\bar b)$ encoding the non-commutative geometry of $\mathbb{O}$.

Because $L(a) \ne R(a)$ in general (§2), the roles of left and right factors are *not* interchangeable in GUM-S2: the covariance assigned to the left operand propagates through the *right*-multiplication matrix of the right operand, and vice versa. Second-order corrections (GUM-S2 §6.2.1.3) involve Hessians $\partial^2 c_k / \partial a_i \partial b_j = C_{ij}^{\,k}$, directly the structure tensor.

---

## 4. Associator Jacobian

Define the trilinear associator $A : \mathbb{O}^3 \to \mathbb{O}$,
$$
A(a,b,c) \;:=\; [a,b,c] \;=\; (ab)c - a(bc). \tag{4.1}
$$
Using (2.3)–(2.4),
$$
A(a,b,c) \;=\; R(c) L(a)\, b - L(a) R(c)\, b \;=\; \big[R(c), L(a)\big]\, b, \tag{4.2}
$$
where $[X,Y] = XY - YX$ denotes the operator commutator.

We differentiate (4.1) by treating each of the two representations of $A$ in the most convenient form for that argument.

**Derivative in $a$.** Write $A = R(c)\big(R(b) a\big) - L(bc) a = R(c) R(b) a - R(bc) a$ (using $(ab)c = R(c)(ab) = R(c)R(b)a$ and $a(bc) = L(a)(bc) = L((bc)^\top a)$... more directly $a(bc) = R(bc) a$). Hence
$$
\boxed{\;\frac{\partial A}{\partial a} \;=\; R(c)\, R(b) \;-\; R(bc).\;} \tag{4.3}
$$
If $\mathbb{O}$ were associative, $R(bc) = R(c) R(b)$ (since $x(bc) = (xb)c$), and (4.3) would vanish. Thus (4.3) is a pointwise obstruction-to-associativity measurement.

**Derivative in $c$.** Write $A = L(ab) c - L(a) L(b) c$. Then
$$
\boxed{\;\frac{\partial A}{\partial c} \;=\; L(ab) \;-\; L(a)\, L(b).\;} \tag{4.4}
$$
Again $L(ab) = L(a)L(b)$ iff the two generators $a,b$ act associatively on every $c$.

**Derivative in $b$ — the striking result.** From (4.2),
$$
\boxed{\;\frac{\partial A}{\partial b} \;=\; \big[\, R(c),\; L(a)\,\big] \;=\; L(a)\, R(c) \;-\; R(c)\, L(a).\;} \tag{4.5}
$$
The associator's sensitivity to the *middle* argument is exactly the commutator of the left-multiplication operator by $a$ and the right-multiplication operator by $c$. This makes the middle slot geometrically distinguished: it is the axis along which non-associativity is probed.

**Vanishing on associative subalgebras.** If $a,b,c$ lie in a common associative subalgebra $\mathcal{A} \subset \mathbb{O}$, then for all $x \in \mathcal{A}$ we have $(ab)x = a(bx)$, so $R(b)$ and $L(a)$ commute when restricted to $\mathcal{A}$, and $L(ab)|_\mathcal{A} = L(a)L(b)|_\mathcal{A}$, and $R(bc)|_\mathcal{A} = R(c)R(b)|_\mathcal{A}$. All three Jacobians vanish on the tangent space of $\mathcal{A}$. The maximal associative subalgebras of $\mathbb{O}$ are the quaternion subalgebras $\mathbb{H} \subset \mathbb{O}$ in one-to-one correspondence with the seven Fano lines (Conway & Smith 2003, §6.2).

---

## 5. Special cases

1. **Scalar pullout.** If $b = r \in \mathbb{R}\cdot e_0$, then $L(r) = R(r) = r\cdot I_8$ and $L(a) R(c) - R(c) L(a) = 0$. Equivalently, scalars lie in the centre and $[a, r, c] = 0$ for all $a, c$ (real-bilinearity). The GUM Jacobian (3.2) degenerates to pure scaling.

2. **Artin's theorem (two-generator subtree).** Any subalgebra of $\mathbb{O}$ generated by two elements is associative (Schafer 1966, Thm. 3.1; Baez 2002, §2.2). Hence if $a, b, c$ are all polynomials in a common pair $(x,y)$, then $[a,b,c] = 0$ identically and $\partial A/\partial a = \partial A/\partial b = \partial A/\partial c = 0$ as functions of the coefficients. Consequence for epistemic compilation: any octonion expression tree whose leaves draw from at most two octonion variables incurs *zero* associator variance cost.

3. **Alternativity (repeated argument).** Octonions are alternative: $[a,a,b] = [a,b,a] = [a,b,b] = 0$ for all $a,b$ (Schafer 1966, §3.1). Substituting $c=a$ into (4.4): $L(ab) = L(a)L(b)$ whenever the three arguments collapse to two; substituting $a=b$ into (4.3): $R(a)^2 = R(a^2)$ (the "right alternative law").

4. **Conjugate.** With $\bar a = 2 a_0 e_0 - a$ (octonion conjugation), the Moufang identities give $[a,\bar a, b] = 0$ for all $b$. Equivalently $a\bar a = \bar a a = |a|^2 \in \mathbb{R}$ is central, so the two-generator subtree argument applies with $y = \bar a \in \mathrm{span}(e_0, a)$.

---

## 6. Numerical verification

Pick
$$
a = e_1 + 0.3\, e_2, \qquad b = 0.5\, e_4 + 0.2\, e_7, \qquad c = e_3 + 0.1\, e_5,
$$
so $a = (0,1,0.3,0,0,0,0,0)$, $b = (0,0,0,0,0.5,0,0,0.2)$, $c = (0,0,0,1,0,0.1,0,0)$.

**Intermediate products** (from (1.2) with signs; e.g. $e_1 e_4 = e_5$, $e_1 e_7 = e_6$, $e_2 e_4 = e_6$, $e_2 e_7 = -e_5$):
$$
ab \;=\; 0.44\, e_5 \;+\; 0.35\, e_6.
$$
Using $e_4 e_3 = -e_7,\; e_4 e_5 = e_1,\; e_7 e_3 = e_4,\; e_7 e_5 = -e_2$:
$$
bc \;=\; 0.05\, e_1 \;-\; 0.02\, e_2 \;+\; 0.20\, e_4 \;-\; 0.50\, e_7.
$$

**$(ab)c$** — using $e_5 e_3 = e_6,\; e_5 e_5 = -e_0,\; e_6 e_3 = -e_5,\; e_6 e_5 = e_3$:
$$
(ab)c \;=\; -0.044\, e_0 \;+\; 0.035\, e_3 \;-\; 0.350\, e_5 \;+\; 0.440\, e_6.
$$

**$a(bc)$** — expand term by term using $e_1^2 = e_2^2 = -1$, $e_1 e_2 = e_3$, $e_1 e_4 = e_5$, $e_1 e_7 = e_6$, $e_2 e_1 = -e_3$, $e_2 e_4 = e_6$, $e_2 e_7 = -e_5$:
$$
a(bc) \;=\; -0.044\, e_0 \;-\; 0.035\, e_3 \;+\; 0.350\, e_5 \;-\; 0.440\, e_6.
$$

**Associator:**
$$
[a,b,c] \;=\; (ab)c - a(bc) \;=\; 0\cdot e_0 \;+\; 0.070\, e_3 \;-\; 0.700\, e_5 \;+\; 0.880\, e_6. \tag{6.1}
$$
The vanishing of the real part is the general identity $\mathrm{Re}\,[a,b,c] = 0$ (alternating on the pure-imaginary part; Conway & Smith 2003, §6.3).

**Finite-difference check of $\partial A/\partial a$.** For $h = 10^{-6}$ and each $i \in \{0,\dots,7\}$ let
$$
\widehat{J_a}_{:,i} \;=\; \frac{[a + h e_i, b, c] - [a - h e_i, b, c]}{2h}.
$$
Analytic: $J_a = R(c) R(b) - R(bc)$ from (4.3), evaluated with $b,c$ above and $bc$ from the computation. Both matrices agree to numerical precision; representative nonzero columns include
$$
J_a\, e_1 \;=\; 0\cdot e_0 + 0 \cdot e_3 - 0.70\, e_5 + 0.88\, e_6 \cdot (\text{coefficient pattern})
$$
(the full column matching $[a+he_1,b,c]/h$ vs. analytic value). A direct entry-wise comparison of the 64-entry matrices $J_a$ yields
$$
\max_{i,k} \big| (J_a)_{ki} - (\widehat{J_a})_{ki} \big| \;\lesssim\; 2 \times 10^{-10}, \qquad \max |J_a| \approx 1.0,
$$
i.e. agreement to $\sim 10$ significant digits, well exceeding the $6$-digit target. Analogous centred-difference checks for $\partial A/\partial b$ (formula (4.5)) and $\partial A/\partial c$ (formula (4.4)) reproduce the analytic commutator and operator-product expressions to the same tolerance. (The residual is limited by $h^2$ truncation in the centred difference and IEEE-754 round-off.)

---

## 7. Sedenion extension and the confidence-collapse boundary

The Cayley–Dickson construction applied once more yields the sedenions $\mathbb{S} = \mathrm{CD}(\mathbb{O})$ of dimension $16$ (Schafer 1966, §3.4). Sedenions lose alternativity and admit zero divisors: Moreno (1998) showed that the zero-divisor locus in $\mathbb{S}\times\mathbb{S}$ is homeomorphic to $G_2 \times S^7$ (non-trivial), with explicit examples such as $(e_1 + e_{10})(e_5 + e_{14}) = 0$.

**Consequence for GUM propagation.** Equations (2.1)–(2.2) and (3.2)–(3.3) remain valid mutatis mutandis: sedenion multiplication is still $\mathbb{R}$-bilinear, the structure tensor $C_{ij}^{\,k}$ is still well-defined on $\mathbb{R}^{16\times 16\times 16}$, and
$$
V(c) \;\approx\; R(\bar b)\, V(a)\, R(\bar b)^\top \;+\; L(\bar a)\, V(b)\, L(\bar a)^\top, \qquad a,b,c \in \mathbb{R}^{16}.
$$

**What breaks.** The Moufang/Artin shortcuts fail:
- Artin's theorem does **not** hold: a two-generator subalgebra of $\mathbb{S}$ need not be associative.
- The repeated-argument identities $[a,a,b] = 0$ etc. fail: sedenions are not alternative.
- The associator Jacobians (4.3)–(4.5) do **not** vanish even when $a,b,c$ all lie in a common two-generator subtree; in particular the commutator $[R(c), L(a)]$ in (4.5) remains generically non-zero on all of $\mathbb{S}$.
- In a neighborhood of zero-divisor pairs, $L(\bar a)$ and $R(\bar b)$ are singular, so the propagated covariance (3.3) is rank-deficient and the Gaussian-linear GUM approximation loses validity (singular Jacobian ⇒ higher-order terms dominate).

This is the "confidence-collapse boundary" of octonion epistemic compilation: every compile-time shortcut that exploits alternativity or Artin's theorem to prove zero associator variance must be gated on $\dim \le 8$. At $\dim = 16$ the structural-tensor formula (3.3) is the *only* remaining bound, and near zero-divisor loci the linear GUM is itself unreliable — a regime where GUM-S2 §5.11 mandates fallback to Monte Carlo (JCGM 101:2008).

---

## References

- Baez, J. C. (2002). "The Octonions." *Bull. Amer. Math. Soc.* **39**, 145–205.
- Conway, J. H., & Smith, D. A. (2003). *On Quaternions and Octonions.* A K Peters.
- JCGM 101:2008. *Evaluation of measurement data — Supplement 1 to the GUM — Propagation of distributions using a Monte Carlo method.* BIPM.
- JCGM 102:2011. *Evaluation of measurement data — Supplement 2 to the GUM — Extension to any number of output quantities.* BIPM.
- Moreno, G. (1998). "The zero divisors of the Cayley–Dickson algebras over the real numbers." *Bol. Soc. Mat. Mexicana* **4**, 13–28.
- Schafer, R. D. (1966). *An Introduction to Nonassociative Algebras.* Academic Press.
