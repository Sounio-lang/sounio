#set document(
  title: "PG(k−1, 2) Geometry as a Mathematical Framework for Order-Dependent Processes in Biology",
  author: ("Demetrios C. Agourakis",),
  date: datetime(year: 2026, month: 3, day: 23),
)

#set page(margin: (x: 2.5cm, y: 2.5cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")

#align(center)[
  #text(16pt, weight: "bold")[$"PG"(k-1, 2)$ Geometry as a Mathematical Framework\ for Order-Dependent Processes in Biology:\ Drug Metabolism and the Genetic Code]

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
  We propose that finite projective geometries $"PG"(k-1, 2)$ provide a natural mathematical framework for analysing order-dependent processes in biology. Two applications are developed. First, we observe that the FDA's canonical set of seven major drug-metabolising CYP450 isoforms admits a bijection with the points of the Fano plane $"PG"(2, 2)$, under which the 343 ordered enzyme triples partition into 133 trivial, 42 associative (corresponding to genetically or functionally clustered enzyme families), and 168 non-associative triples. We derive testable predictions: drug–drug interactions mediated by enzyme triples on a Fano line should exhibit weaker order-dependence than those off a Fano line. Second, we embed the 64 codons of the standard genetic code into $ZZ_2^6$ via the biochemical binary encoding (purine/pyrimidine $times$ weak/strong hydrogen bond) and compute properties of the resulting $"PG"(5, 2)$ structure. Hamming distance in $ZZ_2^6$ correlates monotonically with amino acid hydrophobicity difference ($r = 0.199$, strictly monotonic across all six distance classes), and 56% of single-bit mutations preserve amino acid chemical class. The ambient Cayley–Dickson algebra of dimension 64 has $T_6 = 130 thin 200 = 775 times 168$ nonzero basis associator triples #cite(<agourakis2026tower>), confirming that 168 is the quantum of non-associativity at the dimension where the genetic code lives. Validation against clinical pharmacovigilance data is beyond the scope of this mathematical note and is left as empirical future work.

  #v(0.3em)
  #text(weight: "bold")[Keywords:] CYP450, drug–drug interactions, genetic code, projective geometry, Fano plane, non-associativity, Cayley–Dickson

  #v(0.3em)
  #text(weight: "bold")[MSC 2020:] 92C45, 92D20, 17A75, 51E20
]

#v(1em)

= Introduction

Many biological processes are order-dependent: the clinical outcome of a drug cascade depends on the sequence of administration #cite(<wicha2017>), the phenotypic effect of a mutation path depends on the order of substitutions #cite(<weinreich2006>), and the efficacy of combination therapies depends on scheduling #cite(<mokhtari2017>). Despite this, mathematical frameworks for classifying _which_ combinations are order-dependent and which are not remain underdeveloped.

In algebra, order-dependence is captured by the *associator* $[a, b, c] = (a b)c - a(b c)$. When the associator vanishes, the parenthesisation of a triple product does not matter; when it is nonzero, the order of operations is significant. The octonions $OO$ — the largest normed division algebra — are the first algebra in the Cayley–Dickson tower where the associator is generically nonzero. The count of nonzero basis associator triples is exactly $168 = |"PSL"(2,7)|$, governed by the Fano plane $"PG"(2, 2)$ #cite(<baez2002>) #cite(<agourakis2026aaca>).

In a companion paper #cite(<agourakis2026tower>), we showed that this 168-divisibility extends throughout the Cayley–Dickson tower, with $T_k = 168(P_k - 4 P_(k-1))$ verified at four levels through dimension 64.

In this paper, we ask: _do biological systems whose interaction structure involves 7 or $2^k - 1$ elements exhibit non-associative patterns consistent with the $"PG"(k-1, 2)$ geometry?_ We develop two applications — one at the scale of $"PG"(2, 2)$ (drug metabolism) and one at $"PG"(5, 2)$ (the genetic code) — and derive testable predictions from each.

= The CYP450 system in $"PG"(2, 2)$

== Background: CYP450 drug metabolism

The cytochrome P450 (CYP) superfamily is responsible for the oxidative metabolism of the majority of clinically used drugs #cite(<zanger2013>). Regulatory agencies identify seven principal isoforms for drug–drug interaction (DDI) assessment: CYP1A2, CYP2B6, CYP2C8, CYP2C9, CYP2C19, CYP2D6, and CYP3A4 #cite(<fda2020>).

Drug interactions mediated by these enzymes are frequently order-dependent. Mechanism-based inhibition (MBI) — irreversible inactivation of the enzyme — creates a time-arrow: pre-treatment with an MBI perpetrator before introduction of a victim substrate produces maximum inhibition from day 1, whereas the reverse order produces a gradual escalation over 3–5 days #cite(<zhou2005>). Quantitatively, 67% of pharmacodynamic DDIs in a large-scale analysis were found to be monodirectional, with distinct perpetrators and victims #cite(<wicha2017>).

== The bijection

Under the FDA's canonical seven-isoform simplification, the CYP450 system admits a bijection with the points of $"PG"(2, 2)$. We propose the following mapping, motivated by genetic clustering and substrate overlap:

#figure(
  table(
    columns: 4,
    align: (center, left, left, left),
    stroke: 0.5pt,
    [*Basis*], [*CYP isoform*], [*Chromosome*], [*Rationale*],
    [$e_1$], [CYP1A2], [15q24], [Isolated family; planar aromatic substrates],
    [$e_2$], [CYP2C9], [10q23], [CYP2C cluster; acidic/neutral substrates],
    [$e_3$], [CYP2C8], [10q23], [CYP2C cluster; shared substrates with CYP3A4],
    [$e_4$], [CYP2B6], [19q13], [CYP2 family; distinct from CYP2C],
    [$e_5$], [CYP2C19], [10q23], [CYP2C cluster; completes Fano line $(2,3,5)$],
    [$e_6$], [CYP2D6], [22q13], [Unique: non-inducible; basic nitrogen substrates],
    [$e_7$], [CYP3A4], [7q22], [Dominant enzyme ($approx$30% of drugs)],
  ),
  caption: [Proposed bijection between CYP450 isoforms and $"PG"(2,2)$ points.],
) <table:cyp>

The mapping is constrained by the requirement that the three enzymes of the CYP2C gene cluster on chromosome 10q23.33 — CYP2C8, CYP2C9, and CYP2C19 — form a Fano line (here, line $(2, 3, 5)$). These three enzymes share a genetic locus spanning $approx$500 kb, exhibit overlapping substrate selectivity, and show coordinated expression patterns #cite(<goldstein2001>). Within the Fano plane, additional lines acquire pharmacological interpretations:

#figure(
  table(
    columns: 3,
    align: (center, left, left),
    stroke: 0.5pt,
    [*Fano line*], [*CYP isoforms*], [*Pharmacological interpretation*],
    [$(2, 3, 5)$], [CYP2C9–CYP2C8–CYP2C19], [CYP2C gene cluster (chr 10)],
    [$(6, 7, 2)$], [CYP2D6–CYP3A4–CYP2C9], [The "big three" metabolisers ($approx$65% of drugs)],
    [$(4, 5, 7)$], [CYP2B6–CYP2C19–CYP3A4], [Prodrug activation triad],
    [$(5, 6, 1)$], [CYP2C19–CYP2D6–CYP1A2], [Polymorphic triad],
    [$(7, 1, 3)$], [CYP3A4–CYP1A2–CYP2C8], [Paclitaxel metabolism bridge],
    [$(1, 2, 4)$], [CYP1A2–CYP2C9–CYP2B6], [CYP1/2 family axis],
    [$(3, 4, 6)$], [CYP2C8–CYP2B6–CYP2D6], [CYP2 subfamily convergence],
  ),
  caption: [Fano lines and their pharmacological interpretations under the proposed mapping.],
) <table:lines>

== The 168 partition and testable predictions

The 343 ordered triples of CYP isoforms partition as $343 = 133 + 42 + 168$, where 133 involve repeated enzymes (trivially order-independent), 42 lie on Fano lines (predicted order-independent), and 168 are non-collinear (predicted order-dependent). This partition follows algebraically from the Fano plane structure and is independent of the biological mapping.

The biological content lies in the mapping itself: we predict that the 42 Fano-line triples correspond to enzyme combinations whose DDI outcomes are less sensitive to administration order (because they share a mechanistic family), while the 168 non-collinear triples correspond to combinations where order-dependence is clinically significant.

*Prediction 1.* _DDIs mediated by CYP enzyme triples on a Fano line (e.g., CYP2C9–CYP2C8–CYP2C19) should exhibit weaker order-dependence than DDIs mediated by non-collinear triples (e.g., CYP1A2–CYP2D6–CYP3A4), as measured by asymmetry of AUC ratios in perpetrator–victim pharmacokinetic studies._

*Prediction 2.* _Among the 168 non-collinear triples, those involving CYP2D6 (the only non-inducible major isoform) should show the strongest order-dependence, because the lack of induction-mediated recovery creates a longer-lasting time-arrow._

*Prediction 3.* _The binary property of the associator norm ($||[e_i, e_j, e_k]|| in {0, 2}$) predicts a bimodal distribution: DDI order-dependence should be either absent or fully present, with few intermediate cases._

Validation of these predictions requires analysis of pharmacovigilance databases (e.g., FDA FAERS) or dedicated pharmacokinetic studies with controlled sequencing, and is beyond the scope of this mathematical note.

= The genetic code in $"PG"(5, 2)$

== The biochemical binary encoding

Each nucleotide base has two binary biochemical properties:

#figure(
  table(
    columns: 4,
    align: (center, center, center, center),
    stroke: 0.5pt,
    [*Base*], [*Ring*], [*H-bond strength*], [$ZZ_2^2$ vector],
    [A], [Purine (0)], [Weak (0)], [(0, 0)],
    [G], [Purine (0)], [Strong (1)], [(0, 1)],
    [U], [Pyrimidine (1)], [Weak (0)], [(1, 0)],
    [C], [Pyrimidine (1)], [Strong (1)], [(1, 1)],
  ),
  caption: [Binary encoding of nucleotide bases.],
) <table:bases>

Each codon (three bases) maps to a vector in $ZZ_2^6 = ZZ_2^2 times ZZ_2^2 times ZZ_2^2$. The 63 nontrivial codons (excluding AAA, the zero vector) are the points of $"PG"(5, 2)$. This encoding is not arbitrary — it is determined by the biochemistry of the bases.

== Hamming distance and amino acid chemistry

The relationship between Hamming distance in $ZZ_2^6$ and amino acid chemical similarity was studied by Freeland and Hurst #cite(<freeland1998>) from an evolutionary optimality perspective. The present work recasts this relationship in the projective geometry $"PG"(5, 2)$ and connects it to the Cayley–Dickson tower via $T_6 = 168 times 775$ #cite(<agourakis2026tower>).

We compute the mean absolute difference in Kyte–Doolittle hydrophobicity #cite(<kyte1982>) between all pairs of nonzero codons, stratified by Hamming distance:

#figure(
  table(
    columns: 3,
    align: (center, center, center),
    stroke: 0.5pt,
    [*Hamming distance*], [*Mean $|Delta H|$*], [*$N$ pairs*],
    [1], [2.03], [186],
    [2], [3.02], [465],
    [3], [3.53], [620],
    [4], [3.78], [465],
    [5], [3.92], [186],
    [6], [4.00], [31],
  ),
  caption: [Mean hydrophobicity difference by Hamming distance in $ZZ_2^6$. The relationship is strictly monotonic.],
) <table:hamming>

The relationship is strictly monotonic: codons that are near in $ZZ_2^6$ encode amino acids with similar hydrophobicity. The Pearson correlation between Hamming distance and $|Delta H|$ is $r = 0.199$ (computed over all 1,953 nonzero codon pairs).

== Mutation robustness

Among 372 single-bit mutations (Hamming distance 1) from nonzero codons, 26.3% are synonymous (same amino acid) and 55.9% preserve amino acid chemical class (nonpolar, polar, positive, negative). The genetic code is robust to the perturbation structure defined by $ZZ_2^6$.

== Fano line structure

In contrast to the metric (Hamming distance) structure, Fano lines in $"PG"(5, 2)$ do not organise codons by amino acid identity or class. Among 651 Fano lines connecting nonzero codons, 64 (9.8%) have all three codons in the same chemical class, compared to a random baseline of 11.5%. The genetic code's degeneracy follows the _metric_ structure of $"PG"(5, 2)$, not its seven-point subplane structure.

== The ambient algebra

The 63 nonzero codons are the imaginary basis elements of the $2^6 = 64$-dimensional Cayley–Dickson algebra. In #cite(<agourakis2026tower>), we verified exhaustively that $T_6 = 130 thin 200 = 775 times 168$: among the $63^3 = 250 thin 047$ ordered basis triples, exactly 130,200 have nonzero associator, and all nonzero associator norms equal exactly 2. The genetic code lives in a space with 775 copies of the 168-quantum of non-associativity.

= The fractal branching structure

The Fano subplane measure in the Cayley–Dickson tower proliferates with branching factor 7 per doubling #cite(<agourakis2026tower>), yielding a combinatorial scaling exponent
$ D_("Fano") = log_2 7 approx 2.807 $ <eq:Dfano>
in three-dimensional triple space. The codimension $3 - D_("Fano") approx 0.193$ characterises the gap between the Fano subplane skeleton and the ambient triple space.

Biologically, this means that the non-associative structure at each scale of the Cayley–Dickson tower is organised by the same self-similar Fano architecture: $"PG"(2, 2)$ at the enzyme scale, $"PG"(5, 2)$ at the codon scale, with the 168-quantum preserved at every level and the biology adapting to the relevant geometric structure at each scale — combinatorial (Fano lines) for small systems, metric (Hamming distance) for large ones.

= Discussion

The framework presented here is a mathematical model, not a claim that biological systems are governed by non-associative algebra. The value of the model lies in its testable predictions and its capacity to organise known phenomena (DDI order-dependence, mutation robustness) within a unified geometric structure.

Several limitations should be noted. The CYP450 mapping requires the FDA's seven-isoform simplification; in reality, additional CYP enzymes (CYP2E1, CYP3A5, and others) contribute to drug metabolism, and the restriction to seven is a regulatory convention. The genetic code embedding depends on the specific binary encoding (@table:bases); other biochemically motivated encodings are possible and may yield different geometric properties. The ambient Cayley–Dickson algebra at dimension 64 is not directly involved in codon biology — it provides the algebraic context in which the projective geometry acquires its associator structure.

The strongest immediate application is the CYP450 model, where the seven-element constraint is externally validated (by regulatory practice), the order-dependence is clinically documented (67% monodirectional DDIs), and the predictions are testable against pharmacovigilance data. We encourage validation using the FDA Adverse Event Reporting System (FAERS) or controlled pharmacokinetic studies with varied drug sequencing.

#bibliography("168-biology-refs.yml", style: "springer-mathphys")
