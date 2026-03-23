#set document(
  title: "The 168 Quantum of Non-Associativity in the Cayley–Dickson Tower",
  author: ("Demetrios C. Agourakis",),
  date: datetime(year: 2026, month: 3, day: 23),
)

#set page(margin: (x: 2.5cm, y: 2.5cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")

#align(center)[
  #text(16pt, weight: "bold")[The 168 Quantum of Non-Associativity\ in the Cayley–Dickson Tower]

  #v(1em)
  #text(12pt)[Demetrios C. Agourakis]

  #v(0.5em)
  #text(9pt)[
    Biomaterials and Regenerative Medicine Post-Graduate Program,\
    Pontifícia Universidade Católica de São Paulo (PUC-SP), Sorocaba, SP, Brazil\
    ORCID: 0009-0001-8671-8878
  ]

  #v(0.5em)
  #text(9pt)[March 2026]

  #v(0.3em)
  #text(8pt)[Correspondence: demetrios\@agourakis.med.br]
]

#v(1em)

#block(width: 100%, inset: (x: 1cm, y: 0.7cm), stroke: 0.5pt)[
  #text(weight: "bold")[Abstract.]
  We study the number $T_k$ of ordered triples of imaginary basis elements with nonzero associator in the $2^k$-dimensional Cayley–Dickson algebra. It is classical that $T_3 = 168 = |"PSL"(2,7)|$ for the octonions. By exhaustive computation we establish $T_k$ for $k = 3, ..., 7$ (through dimension 128), and conjecture $T_k = 168 (P_k - 4 P_(k-1))$ where $P_k = (2^k - 1)(2^(k-1) - 1)(2^(k-2) - 1) slash 21$ counts Fano subplanes in $"PG"(k-1, 2)$. This formula is verified at five levels. We prove that the associator norm dichotomy $||[e_i, e_j, e_k]|| in {0, 2}$ holds for _all_ Cayley–Dickson algebras, as a consequence of the bivalence of the Cayley–Dickson sign function and the associativity of bitwise exclusive-or. We derive that $T_k slash (2^k - 1)^3 arrow.r 1 slash 2$ asymptotically: half of all basis triples are non-associative. The number of Fano subplanes grows by a factor of 7 per doubling, yielding a combinatorial scaling exponent $log_2 7 approx 2.807$.

  #v(0.3em)
  #text(weight: "bold")[Keywords:] Cayley–Dickson algebras, non-associativity, Fano plane, projective geometry, PSL(2,7)

  #v(0.3em)
  #text(weight: "bold")[MSC 2020:] 17A75, 20B25, 17D05, 51E20
]

#v(1em)

= Introduction

The Cayley–Dickson construction produces a tower of real algebras by iterated doubling:
$ RR arrow.r CC arrow.r HH arrow.r OO arrow.r SS arrow.r TT arrow.r dots.c $
At each step, a structural property is lost: commutativity at the quaternions $HH$, associativity at the octonions $OO$, and the division property at the sedenions $SS$ #cite(<baez2002>) #cite(<moreno1998>). In a companion paper #cite(<agourakis2026aaca>), we proved that the number of ordered triples $(i,j,k) in {1, ..., 7}^3$ of imaginary octonion basis elements with nonzero associator is exactly $T_3 = 168 = |"PSL"(2,7)|$, following from the regular action of $"Aut"("Fano")$ on ordered non-collinear triples.

A natural question arises: _does the 168-divisibility persist in higher Cayley–Dickson algebras?_ In the sedenions ($2^4 = 16$ dimensions, $15^3 = 3375$ basis triples) and beyond, new phenomena appear — zero divisors, non-alternativity, and a proliferation of Fano subplanes in the associated projective geometry $"PG"(k-1, 2)$. The nonzero associator count $T_k$ grows rapidly, but its relationship to 168 has not been studied.

In this paper we address this gap. By exhaustive computation through $k = 6$ (dimension 64, requiring classification of $63^3 = 250 thin 047$ triples), we establish that $T_k$ is a multiple of 168 at every level tested, and conjecture a closed formula relating $T_k$ to the number of Fano subplanes in $"PG"(k-1, 2)$. We derive the asymptotic fraction of non-associative triples and identify a combinatorial scaling exponent for the Fano subplane structure.

*Contributions.*
+ $T_k$ computed exhaustively for $k = 3, 4, 5, 6, 7$ (Theorem 1).
+ A conjectured closed formula $T_k = 168(P_k - 4 P_(k-1))$ verified at five levels (Conjecture 2).
+ Asymptotic analysis: $T_k slash N^3 arrow.r 1 slash 2$ (Proposition 3).
+ The binary norm property $||[e_i, e_j, e_k]|| in {0, 2}$ _proven_ for all $k$ (Theorem 4).
+ Fano subplane branching exponent $log_2 7 approx 2.807$ (Proposition 5).
+ A sign function method for computing $T_k$ without high-dimensional array operations (§4).

= Preliminaries

== Cayley–Dickson algebras

Given a real algebra $A$ with conjugation $overline(dot.c)$, the Cayley–Dickson doubling defines a new algebra $A' = A times A$ with multiplication:
$ (a, b)(c, d) = (a c - overline(d) b, quad d a + b overline(c)) $

Starting from $RR$, this yields $CC$, $HH$, $OO$, $SS$, and the $2^k$-dimensional algebra $A_k$ for each $k gt.eq 0$. The algebra $A_k$ has $2^k - 1$ imaginary basis elements $e_1, ..., e_(2^k - 1)$, with $e_0 = 1$ the identity. The associator of three elements is $[a, b, c] = (a b)c - a(b c)$.

== Fano subplanes in $"PG"(k-1, 2)$

The imaginary basis elements of $A_k$ are naturally indexed by nonzero vectors of $FF_2^k$, and the projective space $"PG"(k-1, 2)$ has $2^k - 1$ points. A *Fano subplane* is a subgeometry isomorphic to $"PG"(2, 2)$ — the Fano plane, with 7 points and 7 lines. The number of Fano subplanes in $"PG"(k-1, 2)$ is:
$ P_k = ((2^k - 1)(2^(k-1) - 1)(2^(k-2) - 1)) / 21 $ <eq:Pk>

with $P_2 = 0$, $P_3 = 1$, $P_4 = 15$, $P_5 = 155$, $P_6 = 1395$ #cite(<hirschfeld1998>).

= Results

== Exhaustive computation of $T_k$

We define $T_k$ as the number of ordered triples $(i, j, k)$ with $i, j, k in {1, ..., 2^k - 1}$ such that $[e_i, e_j, e_k] eq.not 0$ in $A_k$.

#block(inset: (left: 0em), stroke: (left: 2pt + black), outset: (left: 0.5em))[
  *Theorem 1.* _The nonzero basis associator counts for $k = 3, 4, 5, 6, 7$ are:_

  #figure(
    table(
      columns: 6,
      align: (center, center, center, center, center, center),
      stroke: 0.5pt,
      [$k$], [Dim], [$N = 2^k - 1$], [$N^3$], [$T_k$], [$T_k slash 168$],
      [3], [8], [7], [343], [168], [1],
      [4], [16], [15], [3 375], [1 848], [11],
      [5], [32], [31], [29 791], [15 960], [95],
      [6], [64], [63], [250 047], [130 200], [775],
      [7], [128], [127], [2 048 383], [1 046 808], [6 231],
    ),
  )

  _Each value is verified by exhaustive classification of all $N^3$ ordered triples._
]

For $k = 3$, the result is classical (168 = number of ordered non-collinear triples in the Fano plane #cite(<baez2002>)). For $k = 4$, the value $T_4 = 1848 = 11 times 168$ was reported in #cite(<agourakis2026aaca>). The values $T_5$, $T_6$, and $T_7$ are new.

For $k lt.eq 6$, each associator is computed via four Cayley–Dickson multiplications (recursively through $k$ levels of doubling); two independent runs produced identical results for $k = 6$. For $k = 7$, we use the sign function method (§3.3): each triple is classified by four evaluations of $sigma$ (integer recursion), avoiding high-dimensional array operations entirely.

== A conjectured closed formula

#block(inset: (left: 0em), stroke: (left: 2pt + black), outset: (left: 0.5em))[
  *Conjecture 2.* _For all $k gt.eq 3$:_
  $ T_k = 168(P_k - 4 P_(k-1)) $ <eq:Tk>
  _where_ $P_k$ _is given by_ @eq:Pk.
]

*Verification.* The conjecture is confirmed at all four computed levels:

#figure(
  table(
    columns: 5,
    align: (center, center, center, center, center),
    stroke: 0.5pt,
    [$k$], [$P_k$], [$P_(k-1)$], [$168(P_k - 4 P_(k-1))$], [Match],
    [3], [1], [0], [$168 times 1 = 168$], [✓],
    [4], [15], [1], [$168 times 11 = 1848$], [✓],
    [5], [155], [15], [$168 times 95 = 15 thin 960$], [✓],
    [6], [1395], [155], [$168 times 775 = 130 thin 200$], [✓],
    [7], [11 811], [1395], [$168 times 6231 = 1 thin 046 thin 808$], [✓],
  ),
)

The factor $P_k - 4 P_(k-1)$ admits a structural interpretation. Each Fano subplane in $"PG"(k-1, 2)$ supports a copy of the octonion multiplication table. In the Cayley–Dickson algebra $A_k$, these subplanes split into two classes: _octonionic_ subplanes (where the restricted algebra is alternative, contributing 168 nonzero associator triples each) and _quasi-octonionic_ subplanes (where alternativity fails, contributing fewer). The structure of this decomposition has been studied for sedenions by Cawagas #cite(<cawagas2004>). A proof of @eq:Tk for general $k$ would require establishing that quasi-octonionic subplanes contribute exactly 72 nonzero triples each and that their count is $7 P_(k-1)$. We leave this as an open problem.

== Asymptotic analysis

#block(inset: (left: 0em), stroke: (left: 2pt + black), outset: (left: 0.5em))[
  *Proposition 3.* _If Conjecture 2 holds, then $T_k slash (2^k - 1)^3 arrow.r 1 slash 2$ as $k arrow.r infinity$._

  _Proof._ For large $k$, $P_k approx 2^(3k - 3) slash 21$ (from @eq:Pk). Then:
  $ T_k approx 168 dot (2^(3k-3) slash 21 - 4 dot 2^(3(k-1)-3) slash 21) = 168 dot 2^(3k-6) dot (8 - 4) slash 21 = 168 dot 4 dot 2^(3k-6) slash 21 = 2^(3k-1) $

  Since $(2^k - 1)^3 approx 2^(3k)$, the ratio $T_k slash (2^k - 1)^3 arrow.r 2^(3k-1) slash 2^(3k) = 1 slash 2$. $square$
]

This means that asymptotically, _exactly half_ of all ordered basis triples are non-associative. The convergence is visible from the data: the fraction is $0.490$ at $k=3$, $0.548$ at $k=4$, $0.536$ at $k=5$, $0.521$ at $k=6$, and $0.511$ at $k=7$, approaching $0.500$.

== Norm dichotomy

#block(inset: (left: 0em), stroke: (left: 2pt + black), outset: (left: 0.5em))[
  *Theorem 4.* _For any $k gt.eq 1$ and any nonzero $i, j, m in ZZ_2^k$: $quad ||[e_i, e_j, e_m]|| in {0, 2}$._

  _Proof._ The Cayley–Dickson construction determines a sign function $sigma: (ZZ_2^k backslash {0})^2 arrow.r {+1, -1}$ such that $e_i dot e_j = sigma(i, j) dot e_(i xor j)$. Expanding the associator:

  $ (e_i e_j) e_m = sigma(i, j) dot sigma(i xor j, m) dot e_((i xor j) xor m) $
  $ e_i (e_j e_m) = sigma(j, m) dot sigma(i, j xor m) dot e_(i xor (j xor m)) $

  Since $xor$ is associative on $ZZ_2^k$, both parenthesisations yield the same basis element $e_(i xor j xor m)$. Let $alpha = sigma(i,j) dot sigma(i xor j, m)$ and $beta = sigma(j,m) dot sigma(i, j xor m)$. Since $sigma$ is bivalent, $alpha, beta in {+1, -1}$, so $alpha - beta in {-2, 0, +2}$. Therefore $[e_i, e_j, e_m] = (alpha - beta) dot e_(i xor j xor m)$ and $||[e_i, e_j, e_m]|| in {0, 2}$. $square$
]

*Remark.* The proof is dimension-independent: it uses only the bivalence of $sigma$ and the associativity of $xor$. No properties specific to the octonions (alternativity, norm multiplicativity) are required. The result applies to all Cayley–Dickson algebras, including those with zero divisors.

== Fano subplane branching exponent

#block(inset: (left: 0em), stroke: (left: 2pt + black), outset: (left: 0.5em))[
  *Proposition 5.* _If Conjecture 2 holds and the quasi-octonionic plane count is $Q_k = 7 P_(k-1)$, then the total number of Fano subplanes satisfies $P_k slash P_(k-1) arrow.r 2^3 = 8$ as $k arrow.r infinity$, and the number of non-associative triples per Fano subplane scales with branching factor $7$, yielding a combinatorial scaling exponent of $log_2 7 approx 2.807$._

  _Proof._ From $P_k approx 2^(3k-3) slash 21$, we have $P_k slash P_(k-1) approx 8$. Each doubling creates $P_k - P_(k-1) approx 7 P_(k-1)$ new quasi-octonionic planes (and the $P_(k-1)$ existing planes persist as octonionic). The branching factor for the non-octonionic structure is therefore 7, and the scaling exponent in the logarithmic base of the dimension ($2^k$) is $log_2 7 approx 2.807$. $square$
]

The codimension of the Fano subplane structure in the three-dimensional triple space is $3 - log_2 7 approx 0.193$. This means the non-associative structure, while filling asymptotically half the triple space by measure, is organized around a Fano subplane skeleton of sub-integer dimension.

= Computational methods

Two methods are used. For $k lt.eq 6$, associators are computed via full Cayley–Dickson multiplication: $(a,b)(c,d) = (a c - overline(d) b, d a + b overline(c))$, recursing through $k$ levels. For $k = 7$ (and potentially higher), we use a _sign function method_ that avoids high-dimensional arrays entirely.

== Sign function method

By Theorem 4, determining whether $[e_i, e_j, e_m] eq.not 0$ requires only checking whether $alpha eq.not beta$, where $alpha = sigma(i,j) dot sigma(i xor j, m)$ and $beta = sigma(j,m) dot sigma(i, j xor m)$. The sign function $sigma$ is computed by integer recursion through $k$ levels (one comparison and one XOR per level), with no floating-point arithmetic and no array allocation. This reduces the per-triple cost from $O(2^k)$ (full multiplication) to $O(k)$ (sign recursion), enabling computation at dimensions where full multiplication is impractical.

At $k = 7$ ($127^3 = 2 thin 048 thin 383$ triples), the sign function method classifies all triples using only integer operations. The method scales to arbitrary $k$ limited only by the $O(N^3) = O(2^(3k))$ enumeration cost.

Source code (including the recursive sign function implementation) is available at #link("https://github.com/agourakis82/sounio")[github.com/agourakis82/sounio], implemented in the Sounio programming language #cite(<sounio2026>).

= Open questions

+ *Prove Conjecture 2 for all $k gt.eq 3$.* The key step is establishing the octonionic/quasi-octonionic subplane decomposition: show that each Fano subplane in $A_k$ is either octonionic (contributing 168 nonzero associator triples) or quasi-octonionic (contributing 72), and that the quasi-octonionic count is $7 P_(k-1)$.

+ *Characterize the structure of the sign function.* Theorem 4 establishes that the norm dichotomy follows from the bivalence of $sigma$. Can the nonzero count $T_k$ be derived analytically from the properties of $sigma$, without exhaustive enumeration?

+ *Characterize the sub-class distribution.* For sedenions, the nonzero associator triples decompose by index range (oct-oct-oct, sed-sed-sed, cross) with each sub-count a multiple of 168 #cite(<agourakis2026aaca>). Does this hold for $k gt.eq 5$?

= Acknowledgments

The octonion base result ($T_3 = 168$) was established in joint work with M. Gerenutti #cite(<agourakis2026aaca>).

#bibliography("168-tower-refs.yml", style: "springer-mathphys")
