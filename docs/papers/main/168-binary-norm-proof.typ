#set document(
  title: "The Binary Norm Property for Cayley–Dickson Associators",
  author: ("Demetrios C. Agourakis",),
  date: datetime(year: 2026, month: 3, day: 23),
)

#set page(margin: (x: 2.5cm, y: 2.5cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")

#align(center)[
  #text(16pt, weight: "bold")[The Binary Norm Property\ for Cayley–Dickson Associators]

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
  We prove that for any Cayley–Dickson algebra of dimension $2^k$ ($k gt.eq 3$), the associator of three imaginary basis elements has norm either 0 or 2, with no intermediate values. The proof is elementary: it follows from the $ZZ_2^k$ grading of the Cayley–Dickson sign function and the associativity of bitwise exclusive-or. As a corollary, basis non-associativity in the entire Cayley–Dickson tower is a $ZZ_2$ parity phenomenon — a sign flip within a single basis direction. We verify the result computationally through dimension 128 ($k = 7$, classifying $127^3 = 2 thin 048 thin 383$ triples) and confirm the conjectured nonzero count $T_7 = 1 thin 046 thin 808 = 6231 times 168$ #cite(<agourakis2026tower>).

  #v(0.3em)
  #text(weight: "bold")[Keywords:] Cayley–Dickson algebras, associator, norm dichotomy, sign function

  #v(0.3em)
  #text(weight: "bold")[MSC 2020:] 17A75, 17D05
]

#v(1em)

= Introduction

In a companion paper #cite(<agourakis2026tower>), we conjectured that the number of ordered triples of imaginary basis elements with nonzero associator in the $2^k$-dimensional Cayley–Dickson algebra is $T_k = 168(P_k - 4P_(k-1))$, and observed computationally that all nonzero associator norms equal exactly 2 through dimension 64 ($k = 6$). The binary norm property ($||[e_i, e_j, e_k]|| in {0, 2}$) was listed as an open question.

In this note we prove the binary norm property for _all_ Cayley–Dickson algebras, using an elementary argument based on the $ZZ_2^k$ grading of the construction. The proof is dimension-independent and requires no case analysis.

= The Cayley–Dickson sign function

Let $A_k$ denote the Cayley–Dickson algebra of dimension $2^k$, with imaginary basis elements $e_1, ..., e_(2^k - 1)$ indexed by nonzero vectors of $ZZ_2^k$. The Cayley–Dickson construction determines a _sign function_ $sigma: (ZZ_2^k backslash {0})^2 arrow.r {+1, -1}$ such that:
$ e_i dot e_j = sigma(i, j) dot e_(i xor j) $ <eq:sigma>
for all nonzero $i, j in ZZ_2^k$, where $xor$ denotes bitwise exclusive-or #cite(<baez2002>). The sign function $sigma$ is defined recursively by the Cayley–Dickson doubling; its explicit form is not needed for the proof.

The key property is that $sigma(i, j) in {+1, -1}$ for all inputs — it is a _bivalent_ function. This follows from the fact that each level of the Cayley–Dickson doubling combines products with coefficients $plus.minus 1$ only.

= The theorem

#block(inset: (left: 0em), stroke: (left: 2pt + black), outset: (left: 0.5em))[
  *Theorem.* _For any $k gt.eq 1$ and any nonzero $i, j, m in ZZ_2^k$:_
  $ ||[e_i, e_j, e_m]|| in {0, 2} $
]

_Proof._ Expanding the associator using @eq:sigma:

$ (e_i e_j) e_m &= sigma(i, j) dot e_(i xor j) dot e_m = sigma(i, j) dot sigma(i xor j, m) dot e_((i xor j) xor m) $ <eq:left>

$ e_i (e_j e_m) &= sigma(j, m) dot e_i dot e_(j xor m) = sigma(j, m) dot sigma(i, j xor m) dot e_(i xor (j xor m)) $ <eq:right>

Since $xor$ is associative on $ZZ_2^k$: $(i xor j) xor m = i xor (j xor m)$. Both parenthesisations therefore yield a scalar multiple of the _same_ basis element $e_(i xor j xor m)$. Define:

$ alpha &= sigma(i, j) dot sigma(i xor j, m) quad & quad "(left parenthesisation sign)" \
  beta &= sigma(j, m) dot sigma(i, j xor m) quad & quad "(right parenthesisation sign)" $

Since $sigma$ is bivalent, $alpha$ and $beta$ are each products of two elements of ${+1, -1}$, hence $alpha, beta in {+1, -1}$.

The associator is:
$ [e_i, e_j, e_m] = (alpha - beta) dot e_(i xor j xor m) $

Since $alpha, beta in {+1, -1}$:
$ alpha - beta in {-2, 0, +2} $

Therefore $||[e_i, e_j, e_m]|| = |alpha - beta| dot ||e_(i xor j xor m)|| = |alpha - beta| dot 1 in {0, 2}$. $square$

*Remark 1.* The associator is nonzero if and only if $alpha eq.not beta$, i.e., the two parenthesisations assign _opposite signs_ to the same basis element. Non-associativity in the Cayley–Dickson tower is therefore a pure $ZZ_2$ parity phenomenon: a sign disagreement, not a directional deviation.

*Remark 2.* The proof uses only two properties: (i) the sign function $sigma$ is bivalent, and (ii) the index function is $xor$ (associative). Both hold by construction at every level of the Cayley–Dickson tower. No special properties of the octonions (alternativity, norm multiplicativity, division) are required; the result applies equally to sedenions, trigintaduonions, and all higher algebras.

= Computational verification and $T_7$

As a cross-check, we implemented the recursive Cayley–Dickson sign function in the Sounio programming language #cite(<sounio2026>) and verified the binary norm property exhaustively at dimensions 8, 16, 32, and 128. The sign function approach avoids constructing full Cayley–Dickson multiplication tables: each associator classification requires only four evaluations of $sigma$ (two for $alpha$, two for $beta$), each computed by integer recursion through $k$ levels. No floating-point arithmetic or high-dimensional arrays are needed.

At dimension 128 ($k = 7$), all $127^3 = 2 thin 048 thin 383$ ordered basis triples were classified. The nonzero count is:

$ T_7 = 1 thin 046 thin 808 = 6231 times 168 $

This is the fifth verification point for the conjectured formula $T_k = 168(P_k - 4 P_(k-1))$ #cite(<agourakis2026tower>):

#figure(
  table(
    columns: 5,
    align: (center, center, center, center, center),
    stroke: 0.5pt,
    [$k$], [Dim], [$T_k$], [$T_k slash 168$], [Method],
    [3], [8], [168], [1], [Theorem #cite(<agourakis2026aaca>)],
    [4], [16], [1 848], [11], [Exhaustive],
    [5], [32], [15 960], [95], [Exhaustive],
    [6], [64], [130 200], [775], [Exhaustive],
    [7], [128], [1 046 808], [6 231], [Sign function (this work)],
  ),
)

= Acknowledgments

The sign function approach was implemented in Sounio, a systems programming language for epistemic computing #cite(<sounio2026>). The $T_3 = 168$ base case was established in joint work with M. Gerenutti #cite(<agourakis2026aaca>).

#bibliography("168-binary-norm-refs.yml", style: "springer-mathphys")
