#set document(
  title: "A Dual Pathway to 168: Anti-Associativity, Sign Cancellation, and the Primitive Zero-Divisor Bijection in Sedenions",
  author: ("Demetrios C. Agourakis", "Marli Gerenutti"),
  date: datetime(year: 2026, month: 4, day: 1),
)

#set page(margin: (x: 2.5cm, y: 2.5cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")

#align(center)[
  #text(15pt, weight: "bold")[A Dual Pathway to 168:\ Anti-Associativity, Sign Cancellation,\ and the Primitive Zero-Divisor Bijection in Sedenions]

  #v(1em)
  #text(12pt)[Demetrios C. Agourakis#super[\*] #h(1em) Marli Gerenutti]

  #v(0.5em)
  #text(9pt)[
    Biomaterials and Regenerative Medicine Post-Graduate Program,\
    Pontifícia Universidade Católica de São Paulo (PUC-SP), Sorocaba, SP, Brazil\
    Faculdade São Leopoldo Mandic, Campinas, SP, Brazil\
    D.C.A. ORCID: 0009-0001-8671-8878
  ]

  #v(0.5em)
  #text(9pt)[April 2026 — Preprint]

  #v(0.3em)
  #text(8pt)[\*Correspondence: demetrios\@agourakis.med.br]
]

#v(1em)

#block(width: 100%, inset: (x: 1cm, y: 0.7cm), stroke: 0.5pt)[
  #text(weight: "bold")[Abstract.]
  The number 168 = |PSL(2,7)| governs the combinatorics of non-associativity in the octonions and the zero-divisor geometry of the sedenions. We establish a _mechanism_ connecting these two manifestations. The collinearity--associativity equivalence in the Fano plane PG(2,2) shows that each of the 168 ordered non-collinear triples corresponds to an anti-associative basis configuration. Under Cayley--Dickson doubling $OO arrow.r SS$, each such configuration produces a primitive zero-divisor pair via a sign cancellation identity (the _Sign Cancellation Lemma_) that holds for all 168 admissible quadruples. We give an intrinsic characterisation of primitive zero divisors, prove that each has exactly 4 primitive partners (nullity 4), and establish a natural bijection $overline(Phi)$ between ordered non-collinear Fano triples and unordered primitive zero-divisor pairs — both sets having cardinality 168. The bijection is $"GL"(3,2)$-equivariant under the embedding $"GL"(3,2) tilde.equiv "PSL"(2,7) hookrightarrow G_2 = "Aut"(OO)$. The ordered pair space $cal(Z)(SS) = {(a,b) : norm(a) = norm(b) = 1, a b = 0}$ is isometric to the compact Lie group $G_2$ (Reggiani 2024); together with $"Der"(SS) = frak(g)_2$ (Wilmot 2025), this yields two independent pathways to 168: a _discrete_ path through Fano combinatorics ($42 times 4 = 168$) and a _continuous_ path through $"PSL"(2,7) subset G_2 tilde.equiv cal(Z)(SS)$. All results are verified by exhaustive computation.

  #v(0.3em)
  #text(weight: "bold")[Keywords:] sedenions, zero divisors, Fano plane, PSL(2,7), Cayley--Dickson, $G_2$, anti-associativity, sign cancellation

  #v(0.3em)
  #text(weight: "bold")[MSC 2020:] 17A75, 17D05, 22E10, 20B25, 51E20
]

#v(1em)

// ============================================================================
= Introduction
// ============================================================================

The sedenions $SS = OO plus.circle OO ell$ are the first Cayley--Dickson algebra with zero divisors: there exist non-zero $a, b in SS$ with $a b = 0$ #cite(<moreno1998>). In a companion paper #cite(<agourakis2026a>), we proved that the number of ordered triples of imaginary octonion basis elements with non-zero associator is exactly $168 = |"PSL"(2,7)|$, and observed that the primitive zero-divisor pair count in sedenions is $336 = 2 times 168$.

This paper establishes the _mechanism_ underlying that coincidence. The central results are:

+ *Sign Cancellation Lemma* (Lemma 2): For every admissible quadruple $(p,q,r,s)$ of octonion indices, a specific product of four structure constants equals $-1$. This identity, verified in all 168 cases, is the algebraic engine that converts anti-associativity into zero division under Cayley--Dickson doubling.

+ *Primitive Zero-Divisor Definition* (Definition 1): An intrinsic characterisation of the 84 projective classes of primitive zero divisors, independent of the construction recipe, via three conditions (P1--P3) on the Cayley--Dickson decomposition.

+ *The 168 Bijection* (Theorem 2): A natural $"GL"(3,2)$-equivariant bijection between ordered non-collinear Fano triples and unordered primitive zero-divisor pairs.

The result yields two independent pathways to the number 168:
- *Discrete path* (bottom-up): $7 times 6 = 42$ ordered pairs of distinct octonion indices $times$ 4 off-line partners $= 168$.
- *Continuous path* (top-down): $"PSL"(2,7) hookrightarrow G_2 tilde.equiv cal(Z)(SS)$, where $cal(Z)(SS) = {(a,b) in SS times SS : norm(a) = norm(b) = 1, a b = 0}$ is the ordered zero-divisor pair space, isometric to the compact exceptional Lie group $G_2$ #cite(<reggiani2024>).

*Notation.* Throughout, $OO$ denotes the real octonions with standard basis ${e_0, ..., e_7}$, and $SS = OO plus.circle OO ell$ the sedenions with basis ${e_0, ..., e_(15)}$. We write $sigma(i,j) in {+1, -1}$ for the structure constants: $e_i e_j = sigma(i,j) e_(i xor j)$ for distinct imaginary indices. The symbol $xor$ denotes bitwise XOR in $FF_2^3 without {0}$.


// ============================================================================
= The Fano Plane and Associativity
// ============================================================================

== Collinearity--associativity equivalence

The Fano plane $"PG"(2,2)$ has 7 points (the non-zero elements of $FF_2^3$) and 7 lines (each containing 3 collinear points). The octonion multiplication table is encoded by the oriented lines of $"PG"(2,2)$.

#block(inset: (left: 0em), stroke: (left: 2pt + black), outset: (left: 0.5em))[
  *Lemma 1* (Collinearity--associativity equivalence). _For distinct $p, q, r in {1, ..., 7}$, the associator $[e_p, e_q, e_r] = (e_p e_q)e_r - e_p(e_q e_r)$ satisfies:_

  _(i)_ $[e_p, e_q, e_r] = 0$ _if and only if_ ${p, q, r}$ _is collinear in_ $"PG"(2,2)$, _i.e.,_ ${e_p, e_q, e_r}$ _generates a quaternion subalgebra_ $HH subset OO$.

  _(ii) If_ ${p, q, r}$ _is non-collinear, then_ $[e_p, e_q, e_r] = 2(e_p e_q)e_r$. _Equivalently,_ $e_p(e_q e_r) = -(e_p e_q)e_r$: _the associator attains its maximal value (complete sign reversal)._
]

_Proof._ Part (i) is classical #cite(<baez2002>). For (ii): both $(e_p e_q)e_r$ and $e_p(e_q e_r)$ equal $plus.minus e_((p xor q) xor r)$ since products of basis elements are $plus.minus$ basis elements, and the $ZZ_2^3$ grading gives $(p xor q) xor r = p xor (q xor r)$ so both land on the same basis element. Non-collinearity forces $[e_p, e_q, e_r] eq.not 0$ by (i), so the two signs must differ: $(e_p e_q)e_r = -e_p(e_q e_r)$, giving $[e_p, e_q, e_r] = 2(e_p e_q)e_r$. $square$

*Remark.* The dichotomy is sharp: a triple of octonion basis elements either fully associates (collinear) or fully _anti_-associates (non-collinear), with no intermediate case.

== The 210 = 42 + 168 partition

Among the $7 times 6 times 5 = 210$ ordered triples of distinct imaginary octonion indices, 42 are collinear ($7$ lines $times$ $3! = 6$ orderings) and $168 = 210 - 42$ are non-collinear. As proved in #cite(<agourakis2026a>), the 168 non-collinear triples form a single orbit under the regular action of $"Aut"("PG"(2,2)) tilde.equiv "GL"(3, FF_2) tilde.equiv "PSL"(2,7)$.


// ============================================================================
= The Mechanism: Anti-Associativity Produces Zero Division
// ============================================================================

== Cayley--Dickson doubling

The sedenions are constructed by $SS = OO plus.circle OO ell$ with multiplication:

$ (a,b)(c,d) = (a c - overline(d)b, quad d a + b overline(c)), quad a,b,c,d in OO. $ <eq:cd>

Write $e_(p+8) = e_p ell$ for $p = 0, ..., 7$, so the sedenion basis is ${e_0, ..., e_(15)}$.

== Admissible quadruples

#block(inset: (left: 0em), stroke: (left: 2pt + black), outset: (left: 0.5em))[
  *Definition 1* (Admissible quadruple). _A quadruple $(p,q,r,s) in {1,...,7}^4$ is *admissible* if (i) $p,q,r,s$ are pairwise distinct, and (ii) $p xor q = r xor s$ in $FF_2^3$._
]

Given distinct $p, q in {1,...,7}$, the value $v = p xor q in {1,...,7}$ determines the third point on the Fano line through $p$ and $q$. The admissibility condition $r xor s = v$ with ${r,s} sect {p,q} = nothing$ selects pairs from the complementary set ${1,...,7} without {p,q}$. By exhaustive enumeration, there are exactly 168 admissible ordered quadruples.

== The Sign Cancellation Lemma

#block(inset: (left: 0em), stroke: (left: 2pt + black), outset: (left: 0.5em))[
  *Lemma 2* (Sign Cancellation Lemma). _Let $(p,q,r,s)$ be an admissible quadruple. Then:_

  $ sigma(p,r) dot sigma(s,q) dot sigma(s,p) dot sigma(q,r) = -1. $ <eq:scl>
]

_Proof._ The admissibility constraint forces ${p,r,s xor q}$ and ${s,q,p xor r}$ to be non-collinear triples in $"PG"(2,2)$ (collinearity would require $p xor r in {p, r}$, contradicting distinctness). By Lemma 1(ii), each non-collinear triple produces a sign reversal. The sign product @eq:scl encodes two such reversals composed around a Fano quadrangle, yielding net $-1$ from the global sign obstruction of non-alternativity.

Concretely: exhaustive enumeration over the standard multiplication table yields exactly 168 admissible ordered quadruples. In each case, the four structure constants are read off and their product is $-1$ without exception. The enumeration is implemented in the computational supplement. $square$

== Zero-divisor construction

Expanding the Cayley--Dickson product @eq:cd with $a = e_p$, $b = delta_a e_q$, $c = e_r$, $d = delta_b e_s$ (suppressing normalisation, $delta_a, delta_b in {+1,-1}$):

$ a c - overline(d)b &= sigma(p,r) e_(p xor r) + delta_a delta_b sigma(s,q) e_(s xor q), $ <eq:first>

$ d a + b overline(c) &= delta_b sigma(s,p) e_(s xor p) - delta_a sigma(q,r) e_(q xor r). $ <eq:second>

The admissibility condition $p xor q = r xor s$ ensures $p xor r = s xor q$ and $s xor p = q xor r$, collapsing each component to a single basis direction.

The first component @eq:first vanishes for the sign choice:

$ delta_b = -delta_a dot sigma(p,r) dot sigma(s,q). $ <eq:deltab>

Substituting into @eq:second, the second component vanishes if and only if @eq:scl holds. The Sign Cancellation Lemma guarantees this for every admissible quadruple.

#block(inset: (left: 0em), stroke: (left: 2pt + black), outset: (left: 0.5em))[
  *Theorem 1* (Anti-associativity $arrow.r$ zero division). _For any admissible quadruple $(p,q,r,s)$ and any $delta_a in {+1,-1}$, define:_

  $ a = (e_p + delta_a e_(q+8)) / sqrt(2), quad b = (e_r + delta_b e_(s+8)) / sqrt(2), $ <eq:pair>

  _where $delta_b$ is given by @eq:deltab. Then $a b = 0$ in $SS$, with $norm(a) = norm(b) = 1$._
]

_Proof._ Immediate from the component cancellation above: @eq:deltab kills the first component, and the SCL (@eq:scl) kills the second. The norms are $norm(a)^2 = 1/2 + 1/2 = 1$, similarly for $b$. $square$

== Worked example

Take the non-collinear triple $(p,q,r) = (1,2,4)$. Then $s = 1 xor 2 xor 4 = 7$, admissible since $p xor q = 3 = r xor s$. Set $delta_a = +1$.

From the Fano table ($e_1 e_4 = e_5$ on line ${1,4,5}$; $e_2 e_5 = e_7$ on line ${2,5,7}$):

$ sigma(1,4) = +1, quad sigma(7,2) = +1, quad sigma(7,1) = -1, quad sigma(2,4) = +1. $

SCL check: $(+1)(+1)(-1)(+1) = -1$. #h(0.3em) $checkmark$

Partner sign: $delta_b = -(+1)(+1)(+1) = -1$.

Zero-divisor pair: $a = (e_1 + e_(10))/sqrt(2)$, $b = (e_4 - e_(15))/sqrt(2)$.

Verification (raw CD): In CD form, $a = (e_1, e_2)/sqrt(2)$ and $b = (e_4, -e_7)/sqrt(2)$. First component: $e_1 e_4 - overline((-e_7)) e_2 = e_5 - e_7 e_2 = e_5 - e_5 = 0$. Second: $(-e_7)e_1 + e_2 overline(e_4) = -e_7 e_1 - e_2 e_4 = e_6 - e_6 = 0$. $checkmark$


// ============================================================================
= Primitive Zero Divisors
// ============================================================================

== Intrinsic characterisation

#block(inset: (left: 0em), stroke: (left: 2pt + black), outset: (left: 0.5em))[
  *Definition 2* (Primitive zero divisor). _An element $v in SS$ with $norm(v) = 1$ is a *primitive zero divisor* if, in the Cayley--Dickson decomposition $v = (alpha, beta)$ with $alpha, beta in OO$:_

  _(P1)_ $alpha in "Im"(OO) without {0}$ _(purely imaginary, non-zero);_

  _(P2)_ $beta in "Im"(OO) without {0}$ _(purely imaginary, non-zero);_

  _(P3)_ $hat(alpha) eq.not plus.minus hat(beta)$ _(distinct imaginary directions)._
]

*Remark.* (P3) excludes the _diagonal family_ ($hat(alpha) = plus.minus hat(beta)$, corresponding to $e_p plus.minus e_(p+8)$), while (P1)--(P2) exclude the _$e_8$-touching family_ (elements with a real octonion component). Both excluded families admit zero divisors #cite(<bissDuggerIsaksen2008>), but arise from degenerate configurations outside the Fano correspondence.

== Basis form

#block(inset: (left: 0em), stroke: (left: 2pt + black), outset: (left: 0.5em))[
  *Proposition 1* (Basis form). _$v in SS$ is a primitive zero divisor if and only if_

  $ v = (e_p plus.minus e_(q+8)) / sqrt(2), quad p, q in {1,...,7}, quad p eq.not q. $ <eq:basis>
]

_Proof._ (P1)--(P2) force $alpha = lambda e_p$ and $beta = mu e_q$ for $p,q in {1,...,7}$, $lambda, mu in RR without {0}$. Unit norm gives $lambda^2 + mu^2 = 1$. For $v$ to be a zero divisor, expanding the CD product with a primitive partner yields a $2 times 2$ system in the partner coefficients. By the SCL, the determinant simplifies to $(lambda^2 - mu^2) sigma(p,r) sigma(s,p)$, which vanishes iff $|lambda| = |mu| = 1/sqrt(2)$. Absorbing signs into $plus.minus$ gives @eq:basis; (P3) forces $p eq.not q$. Conversely, every element of form @eq:basis participates in at least one annihilating pair by Theorem 1. $square$

== Census

#block(inset: (left: 0em), stroke: (left: 2pt + black), outset: (left: 0.5em))[
  *Proposition 2* (Projective census). _The primitive zero divisors form exactly 84 projective classes ($v tilde -v$) and 168 signed elements._
]

_Proof._ From @eq:basis: $7 times 6 times 2 = 84$ elements. Since $-v = -(e_p plus.minus e_(q+8))/sqrt(2)$ is _not_ of the form @eq:basis (it would require $-e_p$ to be a basis element), each projective class intersects the set @eq:basis in exactly one element: 84 classes. Including negatives: $84 times 2 = 168$ signed elements. $square$

== Nullity

#block(inset: (left: 0em), stroke: (left: 2pt + black), outset: (left: 0.5em))[
  *Proposition 3* (Nullity 4). _Each primitive zero divisor has exactly 4 ordered primitive partners._
]

_Proof._ Fix $v = (e_p + delta_a e_(q+8))/sqrt(2)$. A primitive partner has the form $(e_r + delta_b e_(s+8))/sqrt(2)$ with $(p,q,r,s)$ admissible and ${r,s} sect {p,q} = nothing$. Exhaustive enumeration confirms exactly 4 ordered solutions $(r,s)$ for each $(p,q)$; the sign $delta_b$ is uniquely determined. $square$


// ============================================================================
= The 168 Bijection
// ============================================================================

Let $cal(P)$ denote the set of primitive zero divisors (Definition 2) and $cal(P)_+$ the 84 elements of the form @eq:basis with positive leading coefficient.

#block(inset: (left: 0em), stroke: (left: 2pt + black), outset: (left: 0.5em))[
  *Theorem 2* (The 168 bijection). _There is a natural bijection_

  $ overline(Phi): {"ordered non-collinear triples in" "PG"(2,2)} arrow.r^(tilde) {"unordered primitive zero-divisor pairs"} $

  _with both sets of cardinality 168. The bijection is $"GL"(3,2)$-equivariant._
]

_Proof._

_Construction._ Let $T = (p,q,r)$ be an ordered non-collinear triple. Define $s = p xor q xor r$. Then $r xor s = p xor q$ (admissibility), $s eq.not 0$ (non-collinearity), and $s in.not {p,q,r}$ (else $q xor r = 0$, contradicting $q eq.not r$). Set $delta_a = +1$, determine $delta_b$ by @eq:deltab. Define $Phi(T) = (a,b)$ as the ordered pair from @eq:pair.

_Injectivity of $Phi$._ If $(p,q) eq.not (p',q')$, then $a eq.not a'$. If $(p,q) = (p',q')$ but $r eq.not r'$, then $s eq.not s'$ and $b eq.not b'$. In either case $Phi(T) eq.not Phi(T')$.

_Cardinality._ By Proposition 2, $|cal(P)_+| = 84$. By Proposition 3, each has 4 partners in $cal(P)_+$, giving $84 times 4 = 336$ ordered pairs, hence $336 / 2 = 168$ unordered pairs.

_Bijectivity of $overline(Phi)$._ The induced map $overline(Phi)(T) = {Phi(T)}_("unord")$ sends ordered triples to unordered ZD pairs. By injectivity of $Phi$, distinct triples with the same $(p,q)$ yield different $b$, hence different unordered pairs. For triples with different $(p,q)$, exhaustive enumeration confirms that $overline(Phi)$ remains injective. Since $|"dom"| = 168 = |"cod"|$, $overline(Phi)$ is a bijection.

_Equivariance._ $"GL"(3,2)$ acts on $FF_2^3 without {0}$ by linear maps, extending to $SS$ via $g dot e_p = e_(g p)$ and $g dot e_(p+8) = e_(g p + 8)$. As a subgroup of $G_2 = "Aut"(OO)$, it preserves the CD product: $g dot (a b) = (g dot a)(g dot b)$. The sign choice @eq:deltab depends on structure constants $sigma$, which are preserved by automorphisms. Hence $Phi(g dot T) = g dot Phi(T)$. $square$

#block(inset: (left: 0em), stroke: (left: 2pt + black), outset: (left: 0.5em))[
  *Corollary 1.* _Primitive zero divisors in $SS$ are the exact algebraic complement of associativity:_

  $ underbrace(210, "all ordered triples") = underbrace(42, "collinear" \ "(associative)") + underbrace(168, "non-collinear" \ "(anti-associative)"). $
]


// ============================================================================
= The Dual Pathway
// ============================================================================

== The continuous path: $G_2 tilde.equiv cal(Z)(SS)$

Define the _ordered zero-divisor pair space_:

$ cal(Z)(SS) = {(a,b) in SS times SS : norm(a) = norm(b) = 1, a b = 0}. $

Moreno #cite(<moreno1998>) proved $cal(Z)(SS)$ is homeomorphic to the compact exceptional Lie group $G_2$. Reggiani #cite(<reggiani2024>) upgraded this to an _isometry_ when $G_2$ carries its naturally reductive left-invariant metric.

The continuous pathway to 168 is:

$ "PSL"(2,7) tilde.equiv "GL"(3,2) = "Aut"("PG"(2,2)) hookrightarrow G_2 = "Aut"(OO) tilde.equiv cal(Z)(SS). $

The embedding $"PSL"(2,7) hookrightarrow G_2$ is not maximal: Cohen and Wales #cite(<cohenWales1983>) classify all irreducible finite subgroups of $G_2(CC)$, showing that $"PSL"(2,7)$ (order 168) sits inside $"PGL"(2,7)$ (order 336) and $G_2(2)$ (order 12,096). Two non-conjugate embeddings exist #cite(<kingToumazetWybourne1999>). For the bijection Theorem 2, only the _existence_ of the embedding is needed, not maximality.

== The discrete path: $42 times 4 = 168$

The discrete pathway is the content of Sections 3--5:

$ 7 times 6 = 42 "ordered pairs" (p,q) arrow.r^("4 partners each") 168 "ordered ZD pairs" arrow.r^("pair quotient") 168 "unordered ZD pairs." $

The 42 pairs are indexed by the ordered pairs of distinct imaginary octonion indices. Each pair $(p,q)$ determines a Fano line; the 4 partners come from the 2 complementary Fano line pairs (each contributing 2 orderings). The structure decomposes into 7 XOR-fibers, each a bipartite graph on $6+6$ vertices with degree 4 and 24 edges.

== The correspondence diagram

The 168 bijection (Theorem 2) connects both pathways:

#align(center)[
  #table(
    columns: 3,
    align: (center, center, center),
    stroke: none,
    [*Discrete*], [$arrow.l.r^(overline(Phi))$], [*Continuous*],
    [168 non-collinear triples], [], [168 primitive ZD pairs],
    [$"GL"(3,2)$ acts regularly], [], [$"PSL"(2,7) subset G_2 tilde.equiv cal(Z)(SS)$],
    [Fano combinatorics], [], [Sedenion zero-divisor geometry],
  )
]

The bijection $overline(Phi)$ is the formal bridge: it is $"GL"(3,2)$-equivariant, so the discrete and continuous $"PSL"(2,7)$ actions are the _same_ action seen from different perspectives.

== The derivation firewall

$"Der"(SS) = frak(g)_2$ unconditionally. Wilmot #cite(<wilmot2025>) resolves the classical dispute (Schafer #cite(<schafer1954>) vs. Brown) on $"Aut"(SS)$: even if $"Aut"(SS) tilde.equiv G_2 times S_3$ (Brown's claim), the discrete factor $S_3$ contributes $"Lie"(S_3) = 0$, so $"Der"(SS) = "Lie"("Aut"(SS))_0 = frak(g)_2$ regardless. The Lie algebra of symmetries of the zero-divisor structure is always $frak(g)_2$, independent of the automorphism group controversy.

== Hurwitz connection

The number $168 = 84(g - 1)$ at genus $g = 3$ is the Hurwitz bound for the Klein quartic $x^3 y + y^3 z + z^3 x = 0$, whose automorphism group is $"PSL"(2,7)$ — the same group acting on both the Fano plane and the primitive zero-divisor pairs. The Klein quartic realises the Hurwitz bound _because_ its symmetry group is $"PSL"(2,7)$, which embeds in $G_2 tilde.equiv cal(Z)(SS)$. This is not an independent third pathway but a geometric confirmation of the $"PSL"(2,7)$ symmetry.


// ============================================================================
= Computational Verification
// ============================================================================

All results are verified by exhaustive computation.

#figure(
  table(
    columns: 3,
    align: (left, center, left),
    stroke: 0.5pt,
    [*Object*], [*Count*], [*Structure*],
    [Ordered triples of distinct $e_p, e_q, e_r$], [210], [$7 times 6 times 5$],
    [— collinear (associative)], [42], [$7$ lines $times 3!$],
    [— non-collinear (anti-associative)], [168], [ordered bases of $FF_2^3$],
    [Admissible quadruples (Definition 1)], [168], [$7$ XOR fibers $times 24$],
    [Primitive zero divisors (signed)], [168], [$7 times 6 times 2 times 2$],
    [Primitive zero divisors (projective)], [84], [$7 times 6 times 2$],
    [Ordered primitive ZD pairs], [336], [$84 times 4$],
    [Unordered primitive ZD pairs], [168], [$336 slash 2$],
    [SCL verifications (all $-1$)], [168], [0 failures],
  ),
  caption: [Census of primitive zero-divisor combinatorics. All counts verified exhaustively.],
) <table:census>

The census is implemented in two independent codebases:
- *Python* (`generate_sedenion_zero_divisor_geometry.py`): exact integer arithmetic, mirrors the compiler's Cayley--Dickson sign function. Reports 336 ordered, 168 unordered, 84 projective, 42 quartets, 7 XOR fibers — all matching @table:census.
- *Sounio* (`test_fractal_sedenion_e2e.sio`): 8 end-to-end tests including T7 (the 168 theorem as executable code), passing 8/8. The Fano characterisation $"Fano"(i,j,k) arrow.l.r i xor j xor k = 0$ is verified over all 210 ordered triples with zero mismatches.

Source code: #link("https://github.com/agourakis82/sounio")[github.com/agourakis82/sounio]. Computational artifacts: `artifacts/research/sedenion_zero_divisor_geometry.v1.json`.


// ============================================================================
= Discussion
// ============================================================================

The 168 bijection reveals that the emergence of zero divisors in the sedenions is not accidental but is governed by the same combinatorial structure — the Fano plane — that encodes octonion multiplication. The Sign Cancellation Lemma is the algebraic mechanism: anti-associativity at the octonion level creates sign patterns that, under Cayley--Dickson doubling, force exact cancellation in both components of the sedenion product.

The intrinsic characterisation of primitive zero divisors (Definition 2) resolves a circularity in the literature, where "primitive" was defined by the construction recipe rather than by geometric properties of the elements themselves. The conditions (P1)--(P3) are checkable without knowing the construction, and the basis form (Proposition 1) shows that every primitive ZD has a canonical representative in terms of two imaginary octonion units.

The dual pathway is not merely an observation about the number 168. It is a structural statement: the discrete combinatorics of the Fano plane and the continuous geometry of the $G_2$ Lie group are connected by a single $"GL"(3,2)$-equivariant bijection. The derivation firewall ($"Der"(SS) = frak(g)_2$ unconditionally) ensures that this connection is robust against the automorphism group controversy.

The remaining gap is a purely analytic proof that the induced map $overline(Phi)$ on unordered pairs is injective (currently established by enumeration). A closed-form sign-product identity beyond the SCL would eliminate the computational step.


// ============================================================================

#bibliography("168-dual-pathway-refs.yml", style: "springer-mathphys")
