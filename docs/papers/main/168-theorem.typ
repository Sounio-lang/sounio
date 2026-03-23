#set document(
  title: "The 168 Count for Octonion Basis Associators and a Computational Cayley–Dickson Tower Extension",
  author: ("Demetrios C. Agourakis", "Marli Gerenutti"),
  date: datetime(year: 2026, month: 3, day: 22),
)

#set page(margin: (x: 2.5cm, y: 2.5cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")

#align(center)[
  #text(16pt, weight: "bold")[The 168 Count for Octonion Basis Associators\ and a Computational Cayley–Dickson Tower Extension]

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
  #text(9pt)[March 2026 — Revised 22 March 2026]

  #v(0.3em)
  #text(8pt)[\*Correspondence: demetrios\@agourakis.med.br]
]

#v(1em)

#block(width: 100%, inset: (x: 1cm, y: 0.7cm), stroke: 0.5pt)[
  #text(weight: "bold")[Abstract.]
  We prove that the number of ordered triples of imaginary octonion basis elements with nonzero associator is exactly 168 = |PSL(2,7)|. The result follows from the regular action of $"Aut"("Fano") tilde.equiv "PGL"(3, FF_2) tilde.equiv "PSL"(2,7)$ on ordered non-collinear triples. We observe that the associator norm on basis elements takes exactly the values ${0, 2}$ — a $ZZ_2$ dichotomy with no intermediate magnitudes — and prove this from the Cayley–Dickson $ZZ_2^k$ grading without case analysis. Exhaustive computation over the sedenion ($dim = 16$) and trigintaduonion ($dim = 32$) algebras reveals that all nonzero basis associator counts are multiples of 168: $1848 = 11 times 168$ for sedenions and $15 thin 960 = 95 times 168$ for trigintaduonions. The norm dichotomy extends to both higher dimensions. We conjecture a closed formula $T_k = 168(P_k - 4 P_(k-1))$, where $P_k$ counts Fano subplanes in $"PG"(k-1, 2)$, verified at $k = 3, 4, 5$. The primitive zero-divisor pair count $336 = 2 times 168$ in sedenions is shown to be a combinatorial artifact of the $ZZ_2^4$ upper-half structure, not a structural bijection.
  All enumerations are verified exhaustively in the Sounio programming language.

  #v(0.3em)
  #text(weight: "bold")[Keywords:] octonions, sedenions, trigintaduonions, associator, Fano plane, PSL(2,7), Cayley–Dickson, zero divisors, projective geometry

  #v(0.3em)
  #text(weight: "bold")[MSC 2020:] 17A75, 20B25, 17D05, 51E20
]

#v(1em)

= Introduction

The Cayley–Dickson construction produces a tower of algebras by iterated doubling:

$ RR arrow.r^(times 2) CC arrow.r^(times 2) HH arrow.r^(times 2) OO arrow.r^(times 2) SS arrow.r^(times 2) TT arrow.r^(times 2) dots.c $

At each step, a structural property is lost: commutativity at the quaternions $HH$, associativity at the octonions $OO$, and the division property at the sedenions $SS$ (which acquire zero divisors) #cite(<baez2002>) #cite(<moreno1998>). The octonions are the last normed division algebra (Hurwitz, 1898) and have found applications in string theory #cite(<green1987>), exceptional Lie groups #cite(<conway2003>), and quantum information #cite(<levay2008>).

The multiplication table of the octonions is encoded by the Fano plane — the unique projective plane of order 2, with 7 points and 7 lines. The automorphism group of the Fano plane is $"Aut"("Fano") tilde.equiv "PGL"(3, FF_2) = "GL"(3, FF_2) tilde.equiv "PSL"(2,7)$, a simple group of order 168 #cite(<dickson1899>). This group acts on the octonion basis elements by permuting them while preserving the multiplication structure.

While the role of $"PSL"(2,7)$ in octonion multiplication has been extensively studied #cite(<baez2002>) #cite(<conway2003>) #cite(<cerchiai2019>), we provide an explicit combinatorial treatment of the basis associator count and extend it computationally up the Cayley–Dickson tower:

#block(inset: (left: 2em, right: 2em))[
  _The number of ordered triples $(i,j,k) in {1,...,7}^3$ for which the octonion basis associator $[e_i, e_j, e_k] = (e_i e_j)e_k - e_i(e_j e_k)$ is nonzero is exactly 168. This count is a multiple of 168 in every higher Cayley–Dickson algebra tested._
]

We prove the octonionic result, establish a norm dichotomy for basis associators (with a proof avoiding case analysis), and report exhaustive computations extending the 168-divisibility pattern to sedenions ($dim = 16$) and trigintaduonions ($dim = 32$).

*Contributions.* (1) _Theorem 1_: the nonzero octonion basis associator count equals $|"PSL"(2,7)|$, with decomposition $343 = 133 + 42 + 168$. (2) _Lemma 2_: $||[e_i, e_j, e_k]|| in {0, 2}$ for all basis elements, with a proof via the $ZZ_2^k$ grading of the Cayley–Dickson construction. (3) _Computational observations_ (Sections 4–5): all nonzero associator counts in sedenions and trigintaduonions are multiples of 168; the norm dichotomy extends to $dim = 32$. (4) _Conjecture 5_: a closed formula $T_k = 168(P_k - 4 P_(k-1))$ for the nonzero associator count in terms of Fano subplane counts.

= The 168 Theorem

== Setup

Let ${e_0, e_1, ..., e_7}$ denote the standard octonion basis, where $e_0 = 1$ is the identity and $e_1, ..., e_7$ are imaginary units. The multiplication is determined by the Fano plane: for each oriented line $(i,j,k)$, we have $e_i e_j = e_k$ and $e_j e_i = -e_k$.

The *associator* of three octonions is:
$ [a, b, c] = (a b)c - a(b c) $

The octonions are _alternative_: $[a, a, b] = [a, b, b] = 0$ for all $a, b$. By the Artin theorem, any subalgebra generated by two elements is associative #cite(<baez2002>).

== Counting nonzero basis associators

We restrict to imaginary basis elements $e_1, ..., e_7$, excluding the real unit $e_0$ (for which all associators vanish trivially). Consider all $7^3 = 343$ ordered triples $(i,j,k) in {1,...,7}^3$. We use ordered triples because the associator is alternating: $[a,b,c] = -[b,a,c]$; the corresponding count of _unordered_ non-collinear triples is $168 slash 6 = 28$.

*Layer 1: Repeated indices (133 zeros).* If any two of ${i,j,k}$ coincide, the associator vanishes by alternativity ($[a,a,b] = 0$) or flexibility ($[a,b,a] = 0$, which holds in any alternative algebra #cite(<baez2002>)). By inclusion-exclusion over the conditions $i=j$, $j=k$, $i=k$:
$ 3(49) - 3(7) + 7 = 133 $

*Layer 2: Fano-line triples (42 zeros).* Among the $7 times 6 times 5 = 210$ triples with all-distinct indices, those where ${i,j,k}$ forms a line of the Fano plane generate a quaternion subalgebra, which is associative. There are 7 lines, each contributing $3! = 6$ ordered triples: $7 times 6 = 42$.

*Layer 3: Non-collinear triples (168 nonzero).* The remaining: $343 - 133 - 42 = 168$.

== The group-theoretic explanation

#block(inset: (left: 0em), stroke: (left: 2pt + black), outset: (left: 0.5em))[
  *Theorem 1.* _The number of nonzero basis associator triples equals $|"PSL"(2,7)| = 168$._

  _Proof._ The Fano plane $FF$ has 7 points and 7 lines. An ordered non-collinear triple is an ordered triple $(p,q,r)$ of distinct points not all on a line. The count is $7 times 6 times 5 - 7 times 6 = 168$.

  $"Aut"(FF) tilde.equiv "PGL"(3, FF_2) = "GL"(3, FF_2) tilde.equiv "PSL"(2,7)$ acts on ordered non-collinear triples. This action is *regular* (free and transitive):
  - _Transitive_: $"GL"(3, FF_2)$ acts transitively on ordered bases of $FF_2^3$. Non-collinear triples in $FF$ correspond to ordered bases of $FF_2^3$ under the identification $FF tilde.equiv PP(FF_2^3)$, since three projective points are non-collinear if and only if the corresponding vectors are linearly independent.
  - _Free_: The stabilizer of an ordered basis of $FF_2^3$ in $"GL"(3, FF_2)$ is trivial, since an invertible linear map is uniquely determined by its action on a basis.

  Therefore $|{"nonzero associator triples"}| = |"PSL"(2,7)| = 168$. $square$
]

= Basis Associator Norm Dichotomy

#block(inset: (left: 0em), stroke: (left: 2pt + black), outset: (left: 0.5em))[
  *Lemma 2.* _For all $i, j, k in {1, ..., 7}$: $quad ||[e_i, e_j, e_k]|| in {0, 2}$._

  _Proof._ For distinct imaginary basis elements $e_i, e_j$ ($i eq.not j$, both $gt.eq 1$), the Fano plane multiplication gives $e_i e_j = epsilon.alt_(i j) e_(f(i,j))$ where $epsilon.alt_(i j) in {+1, -1}$ and $f(i,j) in {1,...,7}$. Therefore:

  $ (e_i e_j) e_k = epsilon.alt_(i j) dot epsilon.alt_(f(i,j), k) dot e_(f(f(i,j),k)) = plus.minus e_p $
  $ e_i (e_j e_k) = epsilon.alt_(j k) dot epsilon.alt_(i, f(j,k)) dot e_(f(i,f(j,k))) = plus.minus e_q $

  for some $p, q in {0, 1, ..., 7}$ (where index 0 arises when both factors are equal, giving $e_m^2 = -e_0$, a real scalar). Each parenthesization yields $plus.minus$ a single basis element.

  We claim $p = q$, i.e., both parenthesizations land on the same basis element (possibly with different signs). This follows from the $ZZ_2^k$ grading of the Cayley–Dickson construction: there exists a labeling of the basis in which the product $e_i e_j$ has index $i xor j$ (bitwise exclusive-or over $ZZ_2^3$), up to sign. Since $xor$ is associative on $ZZ_2^3$:

  $ (i xor j) xor k = i xor (j xor k) $

  so both parenthesizations yield the same index. The assertion $p = q$ is a property of the abstract algebra $OO$, independent of the choice of labeling: all valid octonion multiplication tables define isomorphic algebras, so a labeling-independent property that holds in one labeling holds in all.

  Since $p = q$, the associator $[e_i, e_j, e_k] = (plus.minus 1 minus.plus 1) e_p$, which is either $0$ (signs match) or $plus.minus 2 e_p$ (signs oppose, giving $||plus.minus 2 e_p|| = 2$). $square$
]

*Remark.* Non-associativity for basis elements is a $ZZ_2$ parity phenomenon — a sign flip within a single basis direction, not a rotation to a different direction. The $ZZ_2^k$ grading argument for $p = q$ extends to all Cayley–Dickson algebras, since the index relation $f(i,j) = i xor j$ holds universally in the standard Cayley–Dickson construction.

= Computational Extension to Higher Dimensions

We extend the analysis computationally to the sedenions $SS$ ($dim = 16$) and trigintaduonions $TT$ ($dim = 32$), constructed by iterated Cayley–Dickson doubling.

#figure(
  table(
    columns: 5,
    align: (left, left, center, center, center),
    stroke: 0.5pt,
    [*Algebra*], [*Dim*], [*Total triples*], [*Nonzero*], [*Factor*],
    [Octonion $OO$], [8], [$7^3 = 343$], [168], [$1 times 168$],
    [Sedenion $SS$], [16], [$15^3 = 3375$], [1848], [$11 times 168$],
    [Trigintaduonion $TT$], [32], [$31^3 = 29 thin 791$], [$15 thin 960$], [$95 times 168$],
  ),
  caption: [Nonzero basis associator counts across the Cayley–Dickson tower (exhaustive computation). Every entry is a multiple of 168.],
) <table:tower>

*Observation 3.* _Every nonzero associator count in @table:tower is a multiple of 168. The norm dichotomy ($||[e_i, e_j, e_k]|| in {0, 2}$) extends to all sedenion and trigintaduonion basis elements._

These are computational observations verified by exhaustive enumeration over all $7^3 = 343$, $15^3 = 3375$, and $31^3 = 29 thin 791$ ordered basis triples in dimensions 8, 16, and 32 respectively.

== Sedenion sub-class decomposition

#figure(
  table(
    columns: 5,
    align: (left, left, center, center, center),
    stroke: 0.5pt,
    [*Sub-class*], [*Range*], [*Total*], [*Nonzero*], [*Factor*],
    [Oct-oct-oct], [$i,j,k in {1,...,7}$], [343], [168], [$1 times 168$],
    [Sed-sed-sed], [$i,j,k in {8,...,15}$], [512], [336], [$2 times 168$],
    [Cross (mixed)], [at least one from each half], [2520], [1344], [$8 times 168$],
    [*Total*], [$i,j,k in {1,...,15}$], [*3375*], [*1848*], [*$11 times 168$*],
  ),
  caption: [Sedenion sub-class decomposition. Every entry is a multiple of 168.],
) <table:sedenion>

The sed-sed count $336 = 2 times 168$ has a simple explanation: in the $ZZ_2^4$ representation, the upper-half indices ${8,...,15}$ all share bit 3. For any two upper-half indices $a, b$: bit 3 of $a xor b$ is $1 xor 1 = 0$, so $a xor b$ falls in the lower half. Therefore no line of $"PG"(3,2)$ (consisting of ${a, b, a xor b}$) lies entirely in the upper half, and all $8 times 7 times 6 = 336$ ordered triples of distinct upper-half elements are non-collinear — hence non-associating. This argument does not extend to zero-divisor structure; at $dim = 32$ the analogous upper-half count is $16 times 15 times 14 = 3360$, breaking the numerical coincidence (see §5.2).

= Primitive Zero-Divisor Enumeration

The sedenions are the first Cayley–Dickson algebra with zero divisors #cite(<moreno1998>) #cite(<cawagas2004>).

== Restricted enumeration

We enumerate zero-divisor pairs of the restricted form $z = e_i + s_1 e_j$, $w = e_k + s_2 e_l$ (two-term basis sums with $plus.minus$ signs) — the "primitive unit zero-divisors" in the terminology of de Marrais #cite(<demarrais2000>). The full set of sedenion zero-divisors is larger (including arbitrary linear combinations); our enumeration covers only this algebraically fundamental subclass.

Exhaustive search over all such pairs with $i < j$, $k < l$, $s_1, s_2 in {+1, -1}$, and $i, j, k, l in {0, ..., 15}$ yields exactly *336* ordered pairs (84 unordered pairs $times$ 4 sign choices). The first pair found is $(e_1 + e_(10))(e_3 + e_(14)) = 0$.

$ 336 = 2 times 168 = 2 times |"PSL"(2,7)| $

== Resolution of the $336 = 336$ coincidence

*Observation 4.* _The primitive zero-divisor pair count (336) equals the sed-sed nonzero associator count (336, @table:sedenion row 2). Both are $2 times |"PSL"(2,7)|$._

This numerical equality is a combinatorial artifact, not a structural bijection. The sed-sed count $336 = 8 times 7 times 6$ follows from the $ZZ_2^4$ upper-half structure (§4.2): every ordered triple of distinct upper-half elements is non-collinear. The zero-divisor count 336 has an independent origin in the structure constants of the Cayley–Dickson product. At dimension 32, the upper-half count becomes $16 times 15 times 14 = 3360$, while the zero-divisor structure scales differently — confirming that the $336 = 336$ equality is particular to the sedenions.

de Marrais #cite(<demarrais2000>) independently counted 168 "primitive unit zero-divisors" in sedenions, arranged in 42 "Assessors"; our 336 counts _ordered_ pairs ($336 = 2 times 168$), consistent with his enumeration.

= Computational Verification

All enumerations are verified by exhaustive computation in the Sounio programming language #cite(<sounio2026>), with native Cayley–Dickson algebra support:

- Octonion multiplication via Fano plane triples (1,2,4)(2,3,5)(3,4,6)(4,5,7)(5,6,1)(6,7,2)(7,1,3).
- Sedenion and trigintaduonion multiplication via iterated Cayley–Dickson: $(a,b)(c,d) = (a c - overline(d)b, d a + b overline(c))$.
- All $7^3 = 343$ octonion, $15^3 = 3375$ sedenion, and $31^3 = 29 thin 791$ trigintaduonion associator norms computed and classified.
- All 57,120 candidate zero-divisor pairs tested.
- A related result — path product norm invariance for octonion-labeled graphs (a consequence of norm multiplicativity applied to binary tree evaluations) — is machine-checked in Lean 4 with 0 `sorry` statements.

Source code: #link("https://github.com/agourakis82/sounio")[github.com/agourakis82/sounio].

= Discussion

== The orbit size 168 in the Cayley–Dickson tower

The number 168 appears at multiple points in the Cayley–Dickson tower: (1) as the count of nonzero octonion basis associators (Theorem 1); (2) as the denominator in Wilmot's formula $T_n = (2^n - 1)(2^n - 2)(2^n - 4) slash 168$ for counting Cayley–Dickson automorphisms #cite(<wilmot2025>); (3) as one-half the primitive sedenion zero-divisor pair count; (4) as the universal divisor of nonzero associator counts across the tower (@table:tower). These appearances confirm that $|"PSL"(2,7)| = 168$ serves as a basic orbit size in the combinatorics of the tower.

== A conjectured tower formula

Let $P_k$ denote the number of Fano subplanes in the projective geometry $"PG"(k-1, 2)$:

$ P_k = ((2^k - 1)(2^(k-1) - 1)(2^(k-2) - 1)) / 21 $

with $P_2 = 0$, $P_3 = 1$, $P_4 = 15$, $P_5 = 155$.

#block(inset: (left: 0em), stroke: (left: 2pt + black), outset: (left: 0.5em))[
  *Conjecture 5.* _For all $k gt.eq 3$, the number of ordered basis triples with nonzero associator in the $2^k$-dimensional Cayley–Dickson algebra is:_

  $ T_k = 168 (P_k - 4 P_(k-1)) $
]

*Verification.* The conjecture is confirmed computationally at three levels:
- $k=3$ (octonions): $T_3 = 168(1 - 0) = 168$ #h(1em) ✓ (Theorem 1)
- $k=4$ (sedenions): $T_4 = 168(15 - 4) = 1848$ #h(1em) ✓ (Observation 3)
- $k=5$ (trigintaduonions): $T_5 = 168(155 - 60) = 15 thin 960$ #h(1em) ✓ (new computation)

The formula predicts $T_6 = 168(1395 - 620) = 130 thin 200$ for the $64$-dimensional Cayley–Dickson algebra.

*Remark.* An analytical proof may follow from the decomposition of Fano subplanes in $"PG"(k-1, 2)$ into "octonionic" subplanes (preserving alternativity, contributing 168 nonzero triples each) and "quasi-octonionic" subplanes (where alternativity fails, contributing fewer), as studied by Cawagas #cite(<cawagas2004>) for the sedenion case. We leave the general proof as an open problem.

*Remark.* The factors $1, 11, 95$ are given by $P_k - 4 P_(k-1)$. In particular, the factor $11 = P_4 - 4 P_3 = 15 - 4$ in $1848 = 11 times 168$ has combinatorial rather than group-theoretic significance (note $11 divides.not 168 = 2^3 dot 3 dot 7$).

== Norm dichotomy

The $ZZ_2$ character of basis non-associativity (Lemma 2) — a sign flip, not a continuous rotation — implies that for basis elements, non-associativity is a discrete phenomenon. This extends computationally to sedenions and trigintaduonions (Observation 3), suggesting it is a general property of the Cayley–Dickson construction, derivable from the bilinearity of the structure constants, the $plus.minus 1$ character of the signs, and the $ZZ_2^k$ index grading.

== Open questions

The original open questions from the initial submission are now resolved (OQ1: $168 times$ divisibility confirmed at $dim = 32$; OQ2: $336 = 336$ explained as combinatorial artifact; OQ3: factor 11 identified via $P_k$ formula; OQ4: $p = q$ proved via $ZZ_2^k$ grading). The following questions remain:

1. Prove Conjecture 5 analytically for all $k gt.eq 3$. The key step would be establishing the octonionic/quasi-octonionic subplane decomposition and showing that quasi-octonionic planes contribute exactly 72 nonzero associator triples.
2. Does the norm dichotomy $||[e_i, e_j, e_k]|| in {0, 2}$ hold for _all_ Cayley–Dickson algebras ($k gt.eq 6$)? It is verified at $k = 3, 4, 5$.
3. Characterize the sub-class distribution (analogous to @table:sedenion) for trigintaduonions and higher, and determine whether the sub-class counts are individually multiples of 168.

#bibliography("168-refs.yml", style: "springer-mathphys")
