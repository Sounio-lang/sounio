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

= Cohomological Reformulation

This section recasts the results above in the language of group cohomology and twisted group algebras, following Albuquerque & Majid #cite(<albuquerque-majid-1999>) and Bremner & Hentzel #cite(<bremner-hentzel-2017>). The reformulation does not introduce new arithmetic but exposes that the $ZZ_2^k$ grading argument used implicitly in §3 (Lemma 2) is in fact a statement about a 3-cocycle in $H^3((ZZ slash 2)^k, {plus.minus 1})$. The cohomological view yields (a) a closed-form alternative for $T_k$, (b) a structural decomposition of 3-dimensional subspaces of $(ZZ slash 2)^k$ that constrains any analytic proof of Conjecture 5, and (c) a categorical chassis $"Hyper"(G, F)$ unifying the entire Cayley–Dickson tower as a single parametric family.

== Octonions as a twisted group algebra

Albuquerque & Majid #cite(<albuquerque-majid-1999>) prove that the octonions are the group algebra $k[(ZZ slash 2)^3]$ twisted by a 2-cochain $F : (ZZ slash 2)^3 times (ZZ slash 2)^3 -> {plus.minus 1}$:

$ e_x dot e_y = F(x, y) dot e_(x + y) quad (#text("addition is XOR in") (ZZ slash 2)^3) $

Bremner & Hentzel #cite(<bremner-hentzel-2017>) give the explicit closed form via the identification $(ZZ slash 2)^3 tilde.equiv FF_8$ (additive) where $FF_8 = FF_2[alpha] slash (alpha^3 + alpha + 1)$:

$ F(x, y) = (-1)^(phi(x, y)), quad phi(x, y) = "tr"(y dot x^6) in FF_2 $

with $"tr": FF_8 -> FF_2$ the absolute trace $"tr"(z) = z + z^2 + z^4$ and $x^6 = x^(-1)$ for nonzero $x$ (since $x^7 = 1$ in $FF_8^times$).

The cochain $F$ is *not* a 2-cocycle — its failure to satisfy the cocycle condition is precisely the non-associativity. The associator is the coboundary $phi = diff F$ given by:

$ phi(x, y, z) = F(x, y) dot F(x + y, z) slash (F(x, y + z) dot F(y, z)) in {plus.minus 1} $

Although $F$ is not a 2-cocycle, $phi = diff F$ *is* a 3-cocycle (since $diff^2 = 0$ formally), and $phi$ represents a class in $H^3((ZZ slash 2)^3, {plus.minus 1})$ — the *Fano cocycle* $omega_("Fano")$.

== The 168 Theorem as cocycle support

Lemma 2 (§3) is now seen as the statement that the associator coboundary takes values in a sign group rather than continuous $U(1)$:

#block(inset: (left: 0em), stroke: (left: 2pt + black), outset: (left: 0.5em))[
  *Lemma 2$'$* (cohomological form). _The Cayley–Dickson cocycle $phi_k : ((ZZ slash 2)^k)^3 -> U(1)$ takes values in the discrete subgroup ${plus.minus 1} subset U(1)$ for all $k gt.eq 3$._
]

Theorem 1 is then the statement that the support of the Fano cocycle has cardinality equal to the order of its automorphism group:

#block(inset: (left: 0em), stroke: (left: 2pt + black), outset: (left: 0.5em))[
  *Theorem 1$'$* (cocycle support). _Let $omega_("Fano") in H^3((ZZ slash 2)^3, {plus.minus 1})$ denote the Fano 3-cocycle. Then $|"supp"(omega_("Fano"))| = |"Aut"(omega_("Fano"))| = |"PSL"(2, 7)| = 168$, where $"Aut"(omega_("Fano"))$ is the stabilizer in $"GL"(3, FF_2)$ acting on $((ZZ slash 2)^3)^3$._
]

The proof of §3.3 supplies this directly: the action is regular on ordered non-collinear triples (= ordered bases), and these are precisely the triples for which $phi(x, y, z) = -1$.

== Closed form and subspace decomposition

The conjectured formula $T_k = 168(P_k - 4 P_(k-1))$ admits an equivalent closed form:

$ T_k = 168 dot (2^(k-1) - 1)(2^(k-2) - 1)(2^(k-1) + 3) slash 21 $

The factor $(2^(k-1) + 3)$ reflects a *cohomological weight* specific to the Cayley–Dickson doubling, distinct from the purely combinatorial Fano-subplane count.

A natural cohomological interpretation of Conjecture 5 would be: $T_k = 168 dot n_("oct")(k)$, where $n_("oct")(k)$ counts 3-dimensional subspaces $V <= (ZZ slash 2)^k$ on which the cocycle restriction $phi_k|_V$ equals the Fano cocycle $omega_("Fano")$. We verified empirically that this naive interpretation requires refinement.

Enumerate the 15 three-dimensional subspaces of $(ZZ slash 2)^4$ as $V_c = ker(c)$ for nonzero $c in (ZZ slash 2)^4$. Counting nonzero basis associators per subspace reveals a *trichotomy*, not a dichotomy:

#figure(
  table(
    columns: 3,
    align: (left, center, center),
    stroke: 0.5pt,
    [*Subspace class*], [*Count per subspace*], [*Number of subspaces*],
    [Fully octonionic ($phi_k|_V = omega_("Fano")$)], [168], [8],
    [Mixed cocycle (intermediate class)], [96], [7],
    [Trivial ($phi_k|_V = 0$)], [0], [0],
  ),
  caption: [Empirical sub-decomposition of the 15 three-dimensional subspaces of $(ZZ slash 2)^4$, classified by associator count among ordered triples. Verified by exhaustive enumeration in Sounio (`examples/cocycle_subspace_168.sio`).],
) <table:subspace>

Naive sums: $8 dot 168 + 7 dot 96 = 2016 = T_4 + 168$. The overcount of $168 = 3 dot 56$ corresponds to exactly $56$ non-linearly-independent triples with nonzero sedenion associator (each lying in three subspaces $V_c$ via a shared 2-dimensional parent $W subset V_c$). Such triples cannot exist in alternative algebras; their presence in sedenions reflects the loss of alternativity and contributes a previously uncatalogued substructure to the basis associator landscape.

The same enumeration extended to the trigintaduonions ($k = 5$) reveals that the $k = 4$ trichotomy does *not* persist as a low-class structure — instead, seven distinct cocycle restriction classes appear among the $P_5 = 155$ three-dimensional subspaces:

#figure(
  table(
    columns: 2,
    align: (center, center),
    stroke: 0.5pt,
    [*Count per subspace*], [*Number of subspaces*],
    [180], [7],
    [168], [43],
    [108], [7],
    [96], [35],
    [92], [21],
    [88], [21],
    [76], [21],
    [*Total*], [*155*],
  ),
  caption: [Distribution of nonzero basis associator counts at $k = 5$, exhaustive over all $155$ three-dimensional subspaces of $(ZZ slash 2)^5$. Verified in Sounio (`examples/cocycle_subspace_k5.sio`).],
) <table:subspace-k5>

Salient features of @table:subspace-k5: the multiplicities $(7, 43, 7, 35, 21, 21, 21)$ are integers dividing or divisible by $7 = $ rank of $"PSL"(2,7)$ acting on Fano points, suggesting an underlying orbit structure; one class has count $180 > 168$, demonstrating that some 3-dimensional subspaces of trigintaduonions carry *more* nonzero basis associators than the pure octonion subalgebra (a phenomenon impossible in alternative algebras and absent at $k = 4$). Sum of LI nonzero counts across the seven classes: $7 dot 180 + 43 dot 168 + 7 dot 108 + 35 dot 96 + 21 dot 92 + 21 dot 88 + 21 dot 76 = 17976 = T_5 + 2016$, with the overcount $2016 = 12 dot 168 = 36 dot 56$ corresponding to a structured set of non-LI nonzero triples in trigintaduonions.

The same enumeration carried one further level up the tower to the chingons ($k = 6$, $dim = 64$) gives the third data point. Of the $P_6 = 1395$ three-dimensional subspaces of $(ZZ slash 2)^6$, the per-subspace counts span *sixteen* distinct values:

#figure(
  table(
    columns: 4,
    align: (center, center, center, center),
    stroke: 0.5pt,
    [*Count*], [*Multiplicity*], [*Count*], [*Multiplicity*],
    [188], [21], [96], [112],
    [184], [21], [94], [84],
    [180], [21], [92], [84],
    [168], [247], [90], [63],
    [108], [84], [88], [273],
    [104], [21], [84], [21],
    [100], [21], [76], [252],
    [98], [21], [72], [49],
  ),
  caption: [Distribution of nonzero basis associator counts at $k = 6$, exhaustive over all $1395$ three-dimensional subspaces of $(ZZ slash 2)^6$. Verified in Sounio (`examples/cocycle_subspace_k6.sio`).],
) <table:subspace-k6>

The class count grows superlinearly across the tower: $1$ class at $k = 3$, $2$ at $k = 4$, $7$ at $k = 5$, $16$ at $k = 6$. At $k = 6$ the spread of counts widens (from $72$ to $188$); three super-octonionic classes (counts $180, 184, 188$) emerge with multiplicity $21$ each. Most multiplicities at $k = 6$ are divisible by $7$ or $21$ (consistent with $"PSL"(2,7)$-orbit structure), with the notable exception of count $168$ at multiplicity $247 = 13 dot 19$, which suggests a contribution from a different orbit family. The sum across all sixteen classes is $149016 = T_6 + 18816$ with overcount $18816 = 112 dot 168$.

Pushed one further level up the tower to the routons ($k = 7$, $dim = 128$), the fourth data point is obtained by enumerating the $P_7 = 11811$ three-dimensional subspaces of $(ZZ slash 2)^7$ as canonical lex-minimal LI quadruples of dual functionals. The per-subspace counts span *twenty-three* distinct values:

#figure(
  table(
    columns: 4,
    align: (center, center, center, center),
    stroke: 0.5pt,
    [*Count*], [*Multiplicity*], [*Count*], [*Multiplicity*],
    [194], [21],   [100], [147],
    [190], [84],   [98],  [252],
    [188], [147],  [96],  [350],
    [186], [63],   [94],  [819],
    [184], [147],  [92],  [294],
    [180], [49],   [90],  [840],
    [168], [1535], [88],  [2499],
    [110], [21],   [86],  [252],
    [108], [784],  [84],  [147],
    [106], [84],   [76],  [2352],
    [104], [147],  [72],  [693],
    [102], [84],   [--],  [--],
  ),
  caption: [Distribution of nonzero basis associator counts at $k = 7$, exhaustive over all $11811$ three-dimensional subspaces of $(ZZ slash 2)^7$. Verified in Sounio (`examples/cocycle_subspace_k7.sio`); total $T_7 = 1046808 = 6231 dot 168$ matches the prediction of Conjecture 5.],
) <table:subspace-k7>

Three structural features survive the passage from $k = 6$ to $k = 7$. First, the formula $T_k = 168(P_k - 4 P_(k-1))$ predicted $T_7 = 168 dot 6231 = 1046808$ and the enumeration produces exactly that total, confirming the cohomological balance at one more level. Second, the super-octonionic family $(180, 184, 188)$ that emerged at $k = 6$ persists at $k = 7$ and *expands* to six counts above $168$: ${180, 184, 186, 188, 190, 194}$, all with $7$-divisible multiplicities (49, 147, 63, 147, 84, 21 — each of the form $7 dot n$ with $n in {3, 7, 9, 12, 21, 21}$). Third, the principal anomaly at count $168$ — multiplicity $247 = 13 dot 19$ at $k = 6$, not $7$-divisible — *grows* at $k = 7$ to multiplicity $1535 = 5 dot 307$, again not $7$-divisible. The anomaly orbit family therefore propagates up the tower with a level-specific arithmetic signature (each multiplicity is a product of two primes, neither equal to $7$), reinforcing the conjecture that it arises from a non-$"PSL"(2,7)$ orbit. The class-count chain ${1, 2, 7, 16, 23}$ at $k in {3, 4, 5, 6, 7}$ slows from the $k = 5 -> k = 6$ doubling to a $7$-class jump at $k = 6 -> k = 7$, suggesting a saturation regime tested at $k = 8$ below.

Pushed one further level — voudons at $k = 8$, $dim = 256$ — the fifth and most decisive data point requires a different enumeration: a dual-functional walk at $k = 8$ would scan $approx 127^5 slash 120 approx 1.4 dot 10^9$ raw LI quintuples and is infeasible. We switch to a *direct $3$-LI-generator basis* — canonical $v_1 < v_2 < v_3$ with $v_3 in.not "span"(v_1, v_2)$ and each generator the minimum of its remaining coset — which enumerates $P_8 = 97155 = binom(8,3)_2$ canonical triples in $approx 8$ million raw iterations. The resulting per-subspace counts span:

#figure(
  table(
    columns: 4,
    align: (center, center, center, center),
    stroke: 0.5pt,
    [*Count*], [*Multiplicity*], [*Count*], [*Multiplicity*],
    [194], [315],   [100], [735],
    [190], [1260],  [98],  [2310],
    [188], [735],   [96],  [1162],
    [186], [945],   [94],  [6405],
    [184], [735],   [92],  [1050],
    [180], [105],   [90],  [8190],
    [168], [10383], [88],  [20895],
    [110], [315],   [86],  [3780],
    [108], [6720],  [84],  [735],
    [106], [1260],  [76],  [20160],
    [104], [735],   [72],  [6965],
    [102], [1260],  [--],  [--],
  ),
  caption: [Distribution of nonzero basis associator counts at $k = 8$, exhaustive over all $97155$ three-dimensional subspaces of $(ZZ slash 2)^8$. Verified in Sounio (`examples/cocycle_subspace_k8.sio`); total $T_8 = 8385048 = 49911 dot 168$ matches the prediction of Conjecture 5. The class set (twenty-three count values) is *identical* to the class set at $k = 7$.],
) <table:subspace-k8>

The $k = 8$ data settles three open questions raised by the $k = 6 -> k = 7$ deceleration:

1. *Saturation is confirmed at $k = 7$.* The class-count chain ${1, 2, 7, 16, 23, 23}$ at $k in {3, 4, 5, 6, 7, 8}$ shows the distinct-count set stabilising at twenty-three. The set ${72, 76, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 168, 180, 184, 186, 188, 190, 194}$ is *bit-identical* between $k = 7$ and $k = 8$. Further Cayley–Dickson doublings change multiplicities but, evidently, do not produce new count values. The full classification problem is therefore reduced to enumerating $23$ orbit families in the limiting $"GL"(infinity, FF_2)$ action.

2. *Conjecture 5 holds at one more level.* Predicted $T_8 = 168 dot (P_8 - 4 P_7) = 168 dot (97155 - 47244) = 168 dot 49911 = 8385048$; measured total $= 8385048$ bit-exact. The closed form $T_k = 168 dot (2^(k-1) - 1)(2^(k-2) - 1)(2^(k-1) + 3) slash 21$ is now validated against five consecutive levels $k in {4, 5, 6, 7, 8}$.

3. *Every non-anomaly multiplicity at $k = 8$ is $7$-divisible.* The twenty-two count classes other than $168$ have multiplicities in ${105, 315, 735, 945, 1050, 1162, 1260, 2310, 3780, 6405, 6720, 6965, 8190, 20160, 20895}$, every entry of which factors through $7$ (e.g. $20895 = 3 dot 5 dot 7 dot 199$; $1162 = 2 dot 7 dot 83$; $6965 = 5 dot 7 dot 199$). The single count-$168$ multiplicity $10383 = 3 dot 3461$ — again a product of two primes, neither $7$ — extends the level-specific signature ($247 = 13 dot 19$ at $k = 6$, $1535 = 5 dot 307$ at $k = 7$, $10383 = 3 dot 3461$ at $k = 8$). The "two-prime non-$7$" structure of the count-$168$ anomaly orbit is robust under three levels of Cayley–Dickson doubling and is unlikely to be coincidental.

A sixth level ($k = 9$, $dim = 512$) was added to test whether the $k = 7 = k = 8$ plateau is a two-level coincidence or a genuine asymptotic regime. Direct $3$-LI-generator enumeration scans the $511^3 slash 6 approx 22$ million raw triples in approximately $11.5$ seconds wall clock and resolves all $P_9 = 788035 = binom(9,3)_2$ canonical subspaces:

#figure(
  table(
    columns: 4,
    align: (center, center, center, center),
    stroke: 0.5pt,
    [*Count*], [*Multiplicity*], [*Count*], [*Multiplicity*],
    [194], [3255],   [100], [3255],
    [190], [13020],  [98],  [19530],
    [188], [3255],   [96],  [4130],
    [186], [9765],   [94],  [48825],
    [184], [3255],   [92],  [3906],
    [180], [217],    [90],  [71610],
    [168], [75183],  [88],  [169911],
    [110], [3255],   [86],  [39060],
    [108], [55552],  [84],  [3255],
    [106], [13020],  [76],  [166656],
    [104], [3255],   [72],  [61845],
    [102], [13020],  [--],  [--],
  ),
  caption: [Distribution of nonzero basis associator counts at $k = 9$, exhaustive over all $788035$ three-dimensional subspaces of $(ZZ slash 2)^9$. Verified in Sounio (`examples/cocycle_subspace_k9.sio`); total $T_9 = 67101720 = 399415 dot 168$ matches the prediction of Conjecture 5. The class set (twenty-three count values) is *bit-identical* to the class sets at $k = 7$ and $k = 8$.],
) <table:subspace-k9>

Three more results survive the passage to $k = 9$:

- *Saturation is genuine, not a two-level coincidence.* The class-count chain ${1, 2, 7, 16, 23, 23, 23}$ at $k in {3, 4, 5, 6, 7, 8, 9}$ now spans *three* consecutive plateaus at twenty-three classes, with the same set ${72, 76, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 168, 180, 184, 186, 188, 190, 194}$ recovered bit-exact. The "no new count values appear past $k = 7$" conjecture is now strongly supported.

- *Conjecture 5 holds at six consecutive levels.* Predicted $T_9 = 168 dot (P_9 - 4 P_8) = 168 dot 399415 = 67101720$; measured: $67101720$ bit-exact. The formula is now validated at $k in {4, 5, 6, 7, 8, 9}$ without exception.

- *Anomaly multiplicity grows geometrically.* The count-$168$ multiplicities ${43, 247, 1535, 10383, 75183}$ at $k in {5, 6, 7, 8, 9}$ give ratios $5.74, 6.21, 6.76, 7.24$, monotone increasing and apparently converging from below toward $8 = 2^3$. The factorisations $43, 247 = 13 dot 19, 1535 = 5 dot 307, 10383 = 3 dot 3461, 75183 = 3 dot 25061$ continue the "single prime or two-prime non-$7$" signature (note $25061$ is prime, as is $3461$). The other twenty-two multiplicities at $k = 9$ are again all $7$-divisible, including the entire super-octonionic family ${180, 184, 186, 188, 190, 194}$ with $7$-divisible multiplicities ${217, 3255, 9765, 3255, 13020, 3255}$ where $217 = 7 dot 31$ is the smallest. The factor $31 = 2^5 - 1$ appears in many $k = 9$ multiplicities, suggesting a $"GL"(5, FF_2)$ subgroup structure inherited from the trigintaduonion level.

A seventh level ($k = 10$, $dim = 1024$) was added to test the four-level-plateau hypothesis. The direct $3$-LI-generator enumeration scans the $1023^3 slash 6 approx 178$ million raw triples in approximately $95$ seconds wall clock and resolves all $P_{10} = 6347715 = binom(10,3)_2$ canonical subspaces:

#figure(
  table(
    columns: 4,
    align: (center, center, center, center),
    stroke: 0.5pt,
    [*Count*], [*Multiplicity*], [*Count*], [*Multiplicity*],
    [194], [29295],   [100], [13671],
    [190], [117180],  [98],  [160146],
    [188], [13671],   [96],  [15442],
    [186], [87885],   [94],  [376929],
    [184], [13671],   [92],  [14994],
    [180], [441],     [90],  [597618],
    [168], [569327],  [88],  [1368423],
    [110], [29295],   [86],  [351540],
    [108], [451584],  [84],  [13671],
    [106], [117180],  [76],  [1354752],
    [104], [13671],   [72],  [520149],
    [102], [117180],  [--],  [--],
  ),
  caption: [Distribution of nonzero basis associator counts at $k = 10$, exhaustive over all $6347715$ three-dimensional subspaces of $(ZZ slash 2)^{10}$. Verified in Sounio (`examples/cocycle_subspace_k10.sio`); total $T_{10} = 536856600 = 3195575 dot 168$ matches the prediction of Conjecture 5. The class set (twenty-three count values) is *bit-identical* to the class sets at $k = 7, 8, 9$.],
) <table:subspace-k10>

The $k = 10$ data delivers a four-level saturation plateau and an unexpected refinement of the anomaly factorisation pattern:

- *Saturation across four levels.* The class-count chain extends to ${1, 2, 7, 16, 23, 23, 23, 23}$ at $k in {3, 4, 5, 6, 7, 8, 9, 10}$. The distinct-count set ${72, 76, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 168, 180, 184, 186, 188, 190, 194}$ is bit-identical across four consecutive Cayley–Dickson levels. The conjecture that the limiting distinct-count set is exactly twenty-three is now strongly supported.

- *Conjecture 5 holds at seven consecutive levels.* Predicted $T_{10} = 168 dot (P_{10} - 4 P_9) = 168 dot 3195575 = 536856600$; measured: $536856600$ bit-exact. The closed form is validated without exception at $k in {4, 5, 6, 7, 8, 9, 10}$.

- *Anomaly factorisation transitions from two primes to three.* The count-$168$ multiplicity at $k = 10$ is $569327 = 11 dot 73 dot 709$, with all three factors prime. This breaks the "two-prime" pattern that held at $k = 6, 7, 8, 9$ but preserves the structural feature — the multiplicity is still not divisible by $7$. The pattern is therefore refined from "two-prime non-$7$" to "non-$7$ with bounded prime count growing slowly with $k$". The multiplicity ratio $569327 slash 75183 approx 7.57$ extends the monotone-increasing sequence ${5.74, 6.21, 6.76, 7.24, 7.57}$, still approaching $8 = 2^3$ from below. Every other multiplicity at $k = 10$ remains $7$-divisible, including all six super-octonionic counts and the highly composite multiplicities like $1368423 = 3 dot 7 dot 65163$ at count $88$ or $1354752 = 2^9 dot 3^3 dot 7^2$ at count $76$.

== Implications

1. *Conjecture 5 is structurally richer than its compact form $T_k = 168(P_k - 4 P_(k-1))$ suggests.* The simple "subtract trivial subspaces" interpretation is empirically false: at $k = 5$ no 3-dimensional subspace carries the trivial cocycle, and the per-subspace counts span seven distinct values. The arithmetic identity $T_k = 168(P_k - 4 P_(k-1))$ is therefore a *cohomological balance*, with positive contributions from "super-octonionic" subspaces (e.g. count $180$ at $k = 5$) cancelling against partial-cocycle subspaces (counts below 168). A complete proof must enumerate the cocycle restriction classes and weight them with their LI nonzero contributions, recovering the closed form $T_k = 168 dot (2^(k-1) - 1)(2^(k-2) - 1)(2^(k-1) + 3) slash 21$.

2. *A unified parametric type emerges.* The construction $"Hyper"(G, F)$ — a finite abelian group $G$ together with a 2-cochain $F: G times G -> {plus.minus 1}$ — instantiates the entire Cayley–Dickson tower:

$ CC = "Hyper"((ZZ slash 2)^1, F_(CC)), quad HH = "Hyper"((ZZ slash 2)^2, F_(HH)), quad OO = "Hyper"((ZZ slash 2)^3, F_("Fano")) $
$ SS = "Hyper"((ZZ slash 2)^4, F_4), quad TT = "Hyper"((ZZ slash 2)^5, F_5), thick dots $

This is the natural categorical home for the algebras and organizes the Sounio compiler's hypercomplex type system around a single parametric primitive #cite(<sounio2026>).

3. *Zero divisors as cohomological obstruction.* The appearance of zero divisors at $k = 4$ is the obstruction to extending the Fano cocycle from $(ZZ slash 2)^3$ to $(ZZ slash 2)^4$ in a way compatible with multiplicative cancellation. The seven cocycle restriction classes at $k = 5$ provide the first nontrivial data on how this obstruction propagates through the tower.

#block(inset: (left: 0em), stroke: (left: 2pt + black), outset: (left: 0.5em))[
  *Revised Open Question 1.* _Classify the cocycle restriction classes that arise as $phi_k|_V$ for 3-dimensional subspaces $V <= (ZZ slash 2)^k$, with their multiplicities, and prove Conjecture 5 by weighting each class by its LI nonzero associator count._
]

@table:subspace, @table:subspace-k5, @table:subspace-k6, @table:subspace-k7, @table:subspace-k8, @table:subspace-k9, and @table:subspace-k10 supply seven empirical inputs for this classification. The multiplicities at $k = 4, 5, 6, 7, 8, 9, 10$ are predominantly multiples of $7$ and $21$, consistent with an orbit-counting derivation under the $"GL"(k, FF_2)$ action lifting the canonical $"PSL"(2,7)$-action on the Fano cocycle. *At $k = 8, 9, 10$ every non-anomaly multiplicity is $7$-divisible*; the only exception at each level is the count-$168$ class. The anomaly-multiplicity factorisations at $k = 6, 7, 8, 9, 10$ are $247 = 13 dot 19$, $1535 = 5 dot 307$, $10383 = 3 dot 3461$, $75183 = 3 dot 25061$, $569327 = 11 dot 73 dot 709$ — all non-$7$, with the count of prime factors transitioning from two (at $k = 6..9$) to three (at $k = 10$). The growth ratios ${5.74, 6.21, 6.76, 7.24, 7.57}$ are monotone increasing and converge from below toward $8 = 2^3$. The class-set saturation at $k = 7$ is now confirmed across four consecutive levels ($k = 7, 8, 9, 10$), reducing the classification target from an infinite family to a finite set of $23$ orbits: a complete proof must identify *two* orbit families — the $"PSL"(2,7)$-derived multiples of $7$, and the count-$168$ anomaly with its level-dependent non-$7$ signature.

== Computational verification of the cohomological construction

The cohomological reformulation is computationally validated in Sounio:

- `examples/phi_fano_cohomological.sio` — implements $phi_("Fano")$ via $"tr"(y dot x^6)$ over $FF_8$ from scratch, defines basis multiplication $e_x dot e_y = (-1)^(phi(x, y)) e_(x xor y)$, and verifies that the resulting algebra satisfies (i) 168 nonzero basis associators, (ii) norm dichotomy ${0, 2}$, (iii) alternativity (7/7 PASS).
- `examples/cocycle_subspace_168.sio` — enumerates the 15 three-dimensional subspaces of $(ZZ slash 2)^4$ via dual functionals, computes per-subspace associator counts using the Cayley–Dickson sedenion table, produces @table:subspace.
- `examples/cocycle_subspace_k5.sio` — extends to trigintaduonions; enumerates the 155 three-dimensional subspaces of $(ZZ slash 2)^5$ via canonical pairs of dual functionals, produces @table:subspace-k5 (2/2 PASS, $T_5 = 15960$ confirmed).
- `examples/cocycle_subspace_k6.sio` — extends to chingons (dim $64$); enumerates the 1395 three-dimensional subspaces of $(ZZ slash 2)^6$ via canonical lex-minimal LI triples of dual functionals, produces @table:subspace-k6 (2/2 PASS, $T_6 = 130200$ confirmed).
- `examples/cocycle_subspace_k7.sio` — extends to routons (dim $128$); enumerates the 11811 three-dimensional subspaces of $(ZZ slash 2)^7$ via canonical lex-minimal LI quadruples of dual functionals, with VLIST inner-loop optimization (precompute $V$'s seven nonzero elements before triple iteration), producing @table:subspace-k7 (3/3 PASS, $T_7 = 1046808 = 6231 dot 168$ confirmed).
- `examples/cocycle_subspace_k8.sio` — extends to voudons (dim $256$); enumerates the 97155 three-dimensional subspaces of $(ZZ slash 2)^8$. The dual-functional walk used at $k = 4..7$ is infeasible at $k = 8$ ($approx 1.4$ billion raw LI quintuples), so this case switches to direct 3-LI-generator enumeration (canonical $v_1 < v_2 < v_3$, $v_3 in.not "span"(v_1, v_2)$, each generator the minimum of its coset). Produces @table:subspace-k8 (3/3 PASS, $T_8 = 8385048 = 49911 dot 168$ confirmed) and establishes the class-set saturation at twenty-three count values from $k = 7$ onward.
- `examples/cocycle_subspace_k9.sio` — extends to $1024$-ions (dim $512$); enumerates the 788035 three-dimensional subspaces of $(ZZ slash 2)^9$ via the same direct $3$-generator enumeration as $k = 8$, with one additional Cayley–Dickson doubling step (six total: $OO -> SS -> TT -> "Chingons" -> "Routons" -> "Voudons" -> "1024-ions"$). Produces @table:subspace-k9 (3/3 PASS, $T_9 = 67101720 = 399415 dot 168$ confirmed) and confirms saturation at twenty-three count values across three consecutive levels ($k = 7, 8, 9$). Wall clock: $approx 11.5$ s on Linux x86-64 native.
- `examples/cocycle_subspace_k10.sio` — extends to $2048$-ions (dim $1024$); enumerates the 6,347,715 three-dimensional subspaces of $(ZZ slash 2)^{10}$ via the same direct $3$-generator enumeration as $k = 8, 9$, with one more Cayley–Dickson doubling step (seven total). Static memory: $1024^2 dot 2$ i64 $= 16$ MB BSS for the $k = 10$ multiplication table alone (plus $approx 5.34$ MB for all lower-CD tables). Produces @table:subspace-k10 (3/3 PASS, $T_{10} = 536856600 = 3195575 dot 168$ confirmed) and confirms saturation across four consecutive levels ($k = 7, 8, 9, 10$). Wall clock: $approx 95$ s on Linux x86-64 native; gate timeout 600 s.

#bibliography("168-refs.yml", style: "springer-mathphys")
