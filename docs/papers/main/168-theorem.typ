#set document(
  title: "The 168 Count for Octonion Basis Associators and a Computational Sedenion Extension",
  author: ("Demetrios C. Agourakis", "Marli Gerenutti"),
  date: datetime(year: 2026, month: 3, day: 21),
)

#set page(margin: (x: 2.5cm, y: 2.5cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")

#align(center)[
  #text(16pt, weight: "bold")[The 168 Count for Octonion Basis Associators\ and a Computational Sedenion Extension]

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
  #text(9pt)[March 2026]

  #v(0.3em)
  #text(8pt)[\*Correspondence: demetrios\@agourakis.med.br]
]

#v(1em)

#block(width: 100%, inset: (x: 1cm, y: 0.7cm), stroke: 0.5pt)[
  #text(weight: "bold")[Abstract.]
  We prove that the number of ordered triples of imaginary octonion basis elements with nonzero associator is exactly 168 = |PSL(2,7)|. The result follows from the regular action of $"Aut"("Fano") tilde.equiv "PGL"(3, FF_2) tilde.equiv "PSL"(2,7)$ on ordered non-collinear triples. We observe that the associator norm on basis elements takes exactly the values ${0, 2}$ — a $ZZ_2$ dichotomy with no intermediate magnitudes. Exhaustive computation over the sedenion algebra ($dim = 16$) reveals that all nonzero basis associator counts are multiples of 168 (total: $1848 = 11 times 168$), and that the number of primitive zero-divisor pairs of the restricted form $(e_i plus.minus e_j)(e_k plus.minus e_l) = 0$ is $336 = 2 times 168$, numerically coinciding with the upper-half ($i,j,k in {8,...,15}$) nonzero associator count. These sedenion results are computational observations, not proven theorems; the octonionic result is the rigorous core. All enumerations are verified exhaustively in the Sounio programming language.

  #v(0.3em)
  #text(weight: "bold")[Keywords:] octonions, sedenions, associator, Fano plane, PSL(2,7), Cayley–Dickson, zero divisors

  #v(0.3em)
  #text(weight: "bold")[MSC 2020:] 17A75, 20B25, 17D05
]

#v(1em)

= Introduction

The Cayley–Dickson construction produces a tower of algebras by iterated doubling:

$ RR arrow.r^(times 2) CC arrow.r^(times 2) HH arrow.r^(times 2) OO arrow.r^(times 2) SS arrow.r^(times 2) dots.c $

At each step, a structural property is lost: commutativity at the quaternions $HH$, associativity at the octonions $OO$, and the division property at the sedenions $SS$ (which acquire zero divisors) #cite(<baez2002>) #cite(<moreno1998>). The octonions are the last normed division algebra (Hurwitz, 1898) and have found applications in string theory #cite(<green1987>), exceptional Lie groups #cite(<conway2003>), and quantum information #cite(<levay2008>).

The multiplication table of the octonions is encoded by the Fano plane — the unique projective plane of order 2, with 7 points and 7 lines. The automorphism group of the Fano plane is $"Aut"("Fano") tilde.equiv "PGL"(3, FF_2) = "GL"(3, FF_2) tilde.equiv "PSL"(2,7)$, a simple group of order 168 #cite(<dickson1899>). This group acts on the octonion basis elements by permuting them while preserving the multiplication structure.

While the role of $"PSL"(2,7)$ in octonion multiplication has been extensively studied #cite(<baez2002>) #cite(<conway2003>) #cite(<cerchiai2019>), a simple combinatorial fact appears not to have been explicitly stated:

#block(inset: (left: 2em, right: 2em))[
  _The number of ordered triples $(i,j,k) in {1,...,7}^3$ for which the octonion basis associator $[e_i, e_j, e_k] = (e_i e_j)e_k - e_i(e_j e_k)$ is nonzero is exactly 168._
]

This is a consequence of $"PSL"(2,7)$ acting _regularly_ on ordered non-collinear triples. We prove this, establish a norm dichotomy for basis associators, and report computational observations extending the pattern to sedenions.

*Contributions.* (1) _Theorem 1_: the nonzero octonion basis associator count equals $|"PSL"(2,7)|$, with decomposition $343 = 133 + 42 + 168$. (2) _Lemma 2_: $||[e_i, e_j, e_k]|| in {0, 2}$ for octonion basis elements. (3) _Computational observations_ (Sections 4–5): sedenion nonzero associator counts are multiples of 168, and the primitive zero-divisor pair count $336 = 2 times 168$ coincides with the upper-half nonzero associator count.

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

  We claim $p = q$. To see this, note that the Fano plane structure constants determine a well-defined function $g: binom({1,...,7}, 3) arrow.r {0,1,...,7}$ mapping each unordered non-collinear triple ${i,j,k}$ to the index $p$ (resp. $q$) appearing in both parenthesizations. This follows from a direct Fano-plane case analysis: for any non-collinear triple ${i,j,k}$, both $(e_i e_j)e_k$ and $e_i(e_j e_k)$ are proportional to $e_(g({i,j,k}))$ (verification: exhaustive computation over all 28 unordered non-collinear triples confirms $p = q$ in every case).

  Since $p = q$, the associator $[e_i, e_j, e_k] = (plus.minus 1 minus.plus 1) e_p$, which is either $0$ (signs match) or $plus.minus 2 e_p$ (signs oppose, giving $||plus.minus 2 e_p|| = 2$). $square$
]

*Remark.* Non-associativity for basis elements is a $ZZ_2$ parity phenomenon — a sign flip within a single basis direction, not a rotation to a different direction. The assertion $p = q$ is verified for all 28 unordered triples; an alternative proof avoiding case analysis would require showing that $g$ is well-defined from the Fano plane axioms alone. We leave this as a question for the interested reader.

= Computational Sedenion Extension

We extend the analysis computationally to the sedenions $SS$ (dimension 16), constructed by Cayley–Dickson doubling: $SS = OO plus.circle OO ell$, where $ell = e_8$.

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
  caption: [Nonzero sedenion basis associator counts (exhaustive computation). Every entry is a multiple of 168.],
) <table:tower>

*Observation 3.* _Every nonzero associator count in @table:tower is a multiple of 168. The norm dichotomy ($||[e_i, e_j, e_k]|| in {0, 2}$) extends to all sedenion basis elements._

These are computational observations verified by exhaustive enumeration over all $15^3 = 3375$ sedenion basis triples, not proven theorems. The sed-sed count $336 = 2 times 168$ is consistent with the Cayley–Dickson doubling structure: the upper-half multiplication involves _two_ octonion multiplications (from $(a,b)(c,d) = (a c - overline(d) b, d a + b overline(c))$), plausibly introducing a factor of 2. A structural proof of the $168 times$ divisibility pattern remains open.

= Primitive Zero-Divisor Enumeration

The sedenions are the first Cayley–Dickson algebra with zero divisors #cite(<moreno1998>) #cite(<cawagas2004>).

== Restricted enumeration

We enumerate zero-divisor pairs of the restricted form $z = e_i + s_1 e_j$, $w = e_k + s_2 e_l$ (two-term basis sums with $plus.minus$ signs) — the "primitive unit zero-divisors" in the terminology of de Marrais #cite(<demarrais2000>). The full set of sedenion zero-divisors is larger (including arbitrary linear combinations); our enumeration covers only this algebraically fundamental subclass.

Exhaustive search over all such pairs with $i < j$, $k < l$, $s_1, s_2 in {+1, -1}$, and $i, j, k, l in {0, ..., 15}$ yields exactly *336* ordered pairs (84 unordered pairs $times$ 4 sign choices). The first pair found is $(e_1 + e_(10))(e_3 + e_(14)) = 0$.

$ 336 = 2 times 168 = 2 times |"PSL"(2,7)| $

== A numerical coincidence

*Observation 4.* _The primitive zero-divisor pair count (336) equals the sed-sed nonzero associator count (336, @table:tower row 2). Both are $2 times |"PSL"(2,7)|$._

This links the two fundamental algebraic breakdowns in the Cayley–Dickson tower — non-associativity and zero divisors — through the same numerical value. de Marrais #cite(<demarrais2000>) independently counted 168 "primitive unit zero-divisors" in sedenions, arranged in 42 "Assessors"; our 336 counts _ordered_ pairs ($336 = 2 times 168$), consistent with his enumeration. Whether the numerical equality $336 = 336$ reflects a deeper bijection or is an arithmetic coincidence of the Cayley–Dickson construction remains open.

= Computational Verification

All enumerations are verified by exhaustive computation in the Sounio programming language #cite(<sounio2026>), with native Cayley–Dickson algebra support:

- Octonion multiplication via Fano plane triples (1,2,4)(2,3,5)(3,4,6)(4,5,7)(5,6,1)(6,7,2)(7,1,3).
- Sedenion multiplication via Cayley–Dickson: $(a,b)(c,d) = (a c - overline(d)b, d a + b overline(c))$.
- All $7^3 = 343$ octonion and $15^3 = 3375$ sedenion associator norms computed and classified.
- All 57,120 candidate zero-divisor pairs tested.
- A related result — path product norm invariance for octonion-labeled graphs (a consequence of norm multiplicativity applied to binary tree evaluations) — is machine-checked in Lean 4 with 0 `sorry` statements.

Source code: #link("https://github.com/agourakis82/sounio")[github.com/agourakis82/sounio].

= Discussion

== The orbit size 168 in the Cayley–Dickson tower

The number 168 appears at three points in the Cayley–Dickson tower: (1) as the count of nonzero octonion basis associators (Theorem 1); (2) as the denominator in Wilmot's formula $T_n = (2^n - 1)(2^n - 2)(2^n - 4) slash 168$ for counting Cayley–Dickson automorphisms #cite(<wilmot2025>); (3) as one-half the primitive sedenion zero-divisor pair count. These appearances suggest that $|"PSL"(2,7)| = 168$ serves as a basic orbit size in the combinatorics of the tower, though a structural explanation covering all three instances remains to be found.

== Norm dichotomy

The $ZZ_2$ character of basis non-associativity (Lemma 2) — a sign flip, not a continuous rotation — implies that for basis elements, non-associativity is a discrete phenomenon. This extends computationally to sedenions (Observation 3), suggesting it may be a general property of the Cayley–Dickson construction, derivable from the bilinearity of the structure constants and the $plus.minus 1$ character of the Fano plane signs.

== Open questions

1. Does the $168 times$ divisibility pattern for nonzero basis associator counts persist in the trigintaduonions (dimension 32)? The literature reports 1,260 zero-divisors there; checking 168-divisibility of the associator counts is a straightforward computation.
2. Is there a _bijection_ between the 336 primitive zero-divisor pairs and the 336 nonzero sed-sed associator triples?
3. The factor $11$ in $1848 = 11 times 168$: does it have group-theoretic significance? Note $11 divides.not |"PSL"(2,7)| = 168 = 2^3 dot 3 dot 7$.
4. Can the assertion $p = q$ in Lemma 2 be proved from the Fano plane axioms without case analysis?

#bibliography("168-refs.yml", style: "springer-mathphys")
