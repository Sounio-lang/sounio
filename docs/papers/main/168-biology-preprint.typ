#set document(
  title: "PG(k−1, 2) Geometry as a Mathematical Framework for Order-Dependent Processes in Biology",
  author: ("Demetrios C. Agourakis", "Marli Gerenutti"),
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
  We propose that finite projective geometries $"PG"(k-1, 2)$ provide a natural mathematical framework for analysing order-dependent processes in biology. Two applications are developed. First, we observe that the FDA's canonical set of seven major drug-metabolising CYP450 isoforms admits a bijection with the points of the Fano plane $"PG"(2, 2)$, under which the 343 ordered enzyme triples partition into 133 trivial, 42 associative (corresponding to genetically or functionally clustered enzyme families), and 168 non-associative triples. We derive testable predictions: drug–drug interactions mediated by enzyme triples on a Fano line should exhibit weaker order-dependence than those off a Fano line. Second, we embed the 64 codons of the standard genetic code into $ZZ_2^6$ via the biochemical binary encoding (purine/pyrimidine $times$ weak/strong hydrogen bond) and compute properties of the resulting $"PG"(5, 2)$ structure. Hamming distance in $ZZ_2^6$ correlates monotonically with amino acid hydrophobicity difference ($r = 0.199$, strictly monotonic across all six distance classes under two stop-codon conventions, permutation $p = 0.015$), and 56% of single-bit mutations preserve amino acid chemical class (vs. 31% chance baseline). The ambient Cayley–Dickson algebra of dimension 64 has $T_6 = 130 thin 200 = 775 times 168$ nonzero basis associator triples #cite(<agourakis2026tower>), confirming that 168 is the quantum of non-associativity at the dimension where the genetic code lives. Validation against clinical pharmacovigilance data is beyond the scope of this mathematical note and is left as empirical future work.

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

The mapping is constrained by the requirement that the three enzymes of the CYP2C gene cluster on chromosome 10q23.33 — CYP2C8, CYP2C9, and CYP2C19 — form a Fano line (here, line $(2, 3, 5)$). These three enzymes share a genetic locus (the four-gene CYP2C18–2C19–2C9–2C8 cluster, spanning $approx$500 kb), exhibit overlapping substrate selectivity, and show coordinated expression patterns #cite(<goldstein2001>); all three loci lie in cytogenetic band 10q23.33 (NCBI Gene), as do the remaining assignments in @table:cyp (verified clause C4 of the validation contract, see @sec:verification). Within the Fano plane, additional lines acquire pharmacological interpretations:

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

*Gauge dependence of the bijection.* The single constraint "the CYP2C cluster forms a Fano line" does not uniquely determine the mapping. There are $1008 = 7 times 3! times 4!$ bijections satisfying it, and they form exactly $6$ equivalence classes under the automorphism group $"PSL"(2, 7)$ of the Fano plane (order 168). Crucially, only *one* line — the CYP2C line itself — is common to all six classes; the remaining six line assignments of @table:lines are therefore representative-dependent and should be read as properties of one gauge, not of the biology. Adding a second, independently motivated constraint — that the "big three" metabolisers (CYP3A4, CYP2D6, CYP2C9; together $approx$65% of marketed drugs #cite(<zanger2013>)) form a line — reduces the count to $336$ bijections in $2$ equivalence classes; @table:cyp is one representative satisfying both constraints. The two surviving classes agree on three lines — the CYP2C line, the big-three line, and the CYP1A2–CYP2B6–CYP2C9 line — and differ only in the remaining four line incidences (in the representative of @table:cyp, a swap of CYP2D6 with CYP3A4; equivalently, modulo automorphism, a swap of CYP1A2 with CYP2B6). Any line-level claim beyond the three shared lines therefore requires a third biological constraint to be well-defined. These counts are exact and are re-derived by the validation contract (clause C5, @sec:verification).

== The 168 partition and testable predictions

The 343 ordered triples of CYP isoforms partition as $343 = 133 + 42 + 168$, where 133 involve repeated enzymes (trivially order-independent), 42 lie on Fano lines (predicted order-independent), and 168 are non-collinear (predicted order-dependent). This partition follows algebraically from the Fano plane structure and is independent of the biological mapping. We verified it by exhaustive enumeration of all $7^3$ ordered triples and, independently, by direct multiplication in the octonion algebra: the 168 non-collinear triples are exactly the triples with nonzero basis associator, and every nonzero associator has norm 2 (clauses C1–C3 of the validation contract, @sec:verification). Clause C3 also certifies the design constraint of the DDI test: any two isoforms lie on a (unique) Fano line, so *pairs* cannot discriminate the geometry — only triples can.

The biological content lies in the mapping itself: we predict that the 42 Fano-line triples correspond to enzyme combinations whose DDI outcomes are less sensitive to administration order (because they share a mechanistic family), while the 168 non-collinear triples correspond to combinations where order-dependence is clinically significant.

*Important distinction: associativity vs. commutativity.* The 168 metric classifies _associativity_ — sensitivity to grouping: $(A dot B) dot C eq.not A dot (B dot C)$. This is distinct from _commutativity_ (sensitivity to ordering: $A dot B eq.not B dot A$). In the Cayley–Dickson algebras, distinct imaginary units are universally anti-commutative ($e_i e_j = -e_j e_i$ for $i eq.not j$), so commutativity does not discriminate between triples. The Fano classification is specifically about _grouping_, which corresponds clinically to _phased treatment design_: how interventions are grouped into treatment phases, rather than the temporal sequencing of individual prescriptions.

*Prediction 1 (phase-grouping dependence).* _Clinical drug cascades involving CYP enzyme triples on a Fano line should exhibit weaker sensitivity to phase grouping than non-collinear triples. That is, "(Drug A + Drug B) then Drug C" vs. "Drug A then (Drug B + Drug C)" should differ more when the three CYP enzymes are non-collinear (168 triples) than when they are collinear (42 triples)._

To make this prediction precise and testable, we fix an operational endpoint. For a drug triple $(A, B, C)$ with primary metabolising enzymes $(e_A, e_B, e_C)$, define the *phase-grouping sensitivity*
$ S(A, B, C) = |E_((A B) arrow.r C) - E_(A arrow.r (B C))|, $ <eq:sensitivity>
where $E$ is a standardised pharmacokinetic or clinical effect measure: the steady-state AUC ratio of the last-introduced (victim) drug relative to its monotherapy baseline, or, in outcome studies, a standardised ordinal clinical response. Two grouping regimens are compared: $(A B) arrow.r C$ (A and B co-initiated in phase 1, C added in phase 2) versus $A arrow.r (B C)$ (A alone in phase 1, B and C co-initiated in phase 2), with a washout or steady-state interval of at least five elimination half-lives between phases so that mechanism-based inhibition can express its time-arrow #cite(<zhou2005>).

The test statistic is the difference of means
$ Delta = overline(S)_("non-collinear") - overline(S)_("collinear") $
over the 28 non-collinear and 7 collinear unordered isoform triples (drug triples are selected so that each drug is a recognised substrate of exactly one of the three enzymes, from the FDA DDI table #cite(<fda2020>)). Significance is assessed by a permutation test over the 35 collinearity labels ($binom(35, 7) = 6,724,520$ relabellings, or $10^5$ Monte Carlo samples), one-sided in the predicted direction. _The prediction is supported iff $hat(Delta) > 0$ with permutation $p < 0.05$ and standardised effect size (Cohen's $d$) $>= 0.5$._ Suitable data sources are crossover pharmacokinetic studies with three-drug phasing and phased clinical trials (STAR\*D, STEP-BD, CATIE) re-analysed at the level of phase grouping rather than temporal order.

*Prediction 2 (bimodality).* _The binary property of the associator norm ($||[e_i, e_j, e_k]|| in {0, 2}$, proven for all Cayley–Dickson algebras #cite(<agourakis2026tower>)) predicts a bimodal distribution: phase-grouping sensitivity should be either absent or fully present, with few intermediate cases._ Operationally, across a panel of at least 35 measured triples the distribution of $S$ should reject unimodality (Hartigan's dip test $p < 0.05$, or bimodality coefficient $"BC" > 5/9$).

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

The relationship is strictly monotonic: codons that are near in $ZZ_2^6$ encode amino acids with similar hydrophobicity. The Pearson correlation between Hamming distance and $|Delta H|$ is $r = 0.199$ (computed over all 1,953 nonzero codon pairs; Spearman $rho = 0.208$). Two conventions and one significance statement must accompany these numbers:

- *Stop-codon convention.* @table:hamming assigns $H("stop") = 0$ when a pair involves one of the three stop codons (this is what reproduces the tabulated means). Excluding the 183 stop-involving pairs instead yields means $2.00, 3.05, 3.62, 3.93, 4.08, 4.15$ — still strictly monotonic, with $r = 0.218$ over the remaining 1,770 pairs. The qualitative claim is invariant to the stop convention.
- *Significance.* Under a permutation null (Kyte–Doolittle values permuted among the 21 residue categories, $10^4$ permutations), the observed correlation has $p = 0.015$: the hydrophobicity gradient across Hamming classes is a property of the code, not of the hydrophobicity scale's marginal distribution.
- *Encoding non-optimality (limitation).* The correlation is *not* a distinctive virtue of the biochemical encoding. Among the six encodings with $A = (0,0)$ fixed by the zero-codon convention, the biochemical one ranks only third, and plain nucleotide Hamming distance on codon strings yields a *higher* correlation ($r = 0.261$). The interest of the $ZZ_2^6$ embedding is structural — it is determined by base biochemistry and connects the code to the Cayley–Dickson tower — not statistical optimality for hydrophobicity. All values above are re-derived by the validation contract (clauses C6–C8, @sec:verification).

== Mutation robustness

Among 372 single-bit mutations (Hamming distance 1) from nonzero codons — the 378 directed single-bit flips minus the 6 that land on the excluded zero codon AAA — 26.3% are synonymous (same amino acid) and 55.9% preserve amino acid chemical class (nonpolar, polar, positive, negative). The genetic code is robust to the perturbation structure defined by $ZZ_2^6$. Against the chance baseline of 30.9% (probability that two random nonzero codons share a class), the 55.9% class preservation is overwhelming (binomial $p approx 2 times 10^(-23)$; clause C9). Note that single-bit flips are a subset of single-nucleotide substitutions: transitions and the transversions A$arrow.l.r$U, G$arrow.l.r$C flip one bit, while A$arrow.l.r$C and G$arrow.l.r$U flip two; the $ZZ_2^6$ perturbation structure therefore under-covers, rather than over-covers, the mutational spectrum.

== Fano line structure

In contrast to the metric (Hamming distance) structure, Fano lines in $"PG"(5, 2)$ do not organise codons by amino acid identity or class. Among 651 Fano lines connecting nonzero codons, 64 (9.8%) have all three codons in the same chemical class, compared to a random baseline of 11.5% — a deficit that is not statistically significant (binomial $p = 0.10$; clause C10). The genetic code's degeneracy follows the _metric_ structure of $"PG"(5, 2)$, not its seven-point subplane structure.

== Remark on the ambient algebra

The 63 nonzero codons coincide in number with the imaginary basis elements of the 64-dimensional Cayley–Dickson algebra, for which $T_6 = 130 thin 200 = 775 times 168$ #cite(<agourakis2026tower>). We note this dimensional coincidence without asserting a mechanistic connection: codons do not undergo hypercomplex multiplication, and the Cayley–Dickson algebra at dimension 64 provides a mathematical context rather than a biological mechanism. The biologically meaningful structure is the metric (Hamming distance in $ZZ_2^6$), not the ambient non-associative algebra.

= The fractal branching structure

The Fano subplane measure in the Cayley–Dickson tower proliferates with branching factor 7 per doubling #cite(<agourakis2026tower>), yielding a combinatorial scaling exponent
$ D_("Fano") = log_2 7 approx 2.807 $ <eq:Dfano>
in three-dimensional triple space. The codimension $3 - D_("Fano") approx 0.193$ characterises the gap between the Fano subplane skeleton and the ambient triple space.

Biologically, this means that the non-associative structure at each scale of the Cayley–Dickson tower is organised by the same self-similar Fano architecture: $"PG"(2, 2)$ at the enzyme scale, $"PG"(5, 2)$ at the codon scale, with the 168-quantum preserved at every level and the biology adapting to the relevant geometric structure at each scale — combinatorial (Fano lines) for small systems, metric (Hamming distance) for large ones.

= Falsification criteria

The biological interpretation is refuted — in whole or in part — by any of the following:

- *F1 (Prediction 1).* If, in an adequately powered phased-treatment dataset, collinear (Fano-line) triples show mean phase-grouping sensitivity $overline(S)$ *greater than or equal to* that of non-collinear triples (reversed direction, or $Delta approx 0$ with the 95% confidence interval excluding $|d| >= 0.3$), the CYP450–Fano correspondence is refuted at the level of its central empirical claim.
- *F2 (Prediction 2).* If the distribution of $S$ over $>= 35$ measured triples is consistent with unimodality (Hartigan's dip test $p > 0.05$ and $"BC" < 5/9$), the binary-norm prediction is refuted.
- *F3 (gauge dependence).* Only the CYP2C line is gauge-invariant (common to all six equivalence classes of the bijection). If a line-level pharmacological pattern *other* than the CYP2C line is claimed, it must be re-derived under each of the surviving gauge classes; a pattern that appears in some classes and not others is an artefact of the representative, and claiming it refutes itself. Conversely, if independent biological data fix a gauge different from @table:cyp, the line interpretations of @table:lines are superseded, not falsified — the partition $343 = 133 + 42 + 168$ and Prediction 1 survive any relabelling that keeps the CYP2C line.
- *F4 (genetic code, metric claim).* The Hamming–hydrophobicity gradient is refuted if it loses strict monotonicity under the stop-excluded convention or permutation significance ($p >= 0.05$). It currently survives both (monotonic under both conventions; $p = 0.015$).
- *F5 (genetic code, embedding claim).* The claim that the purine/pyrimidine $times$ H-bond encoding is *special* would be refuted by a systematic advantage of alternative encodings across chemical descriptors. On hydrophobicity alone it is already *not* optimal (nucleotide Hamming $r = 0.261 > 0.199$; clause C8); the paper therefore claims only that the encoding is biochemically determined and structurally connected to the Cayley–Dickson tower, and F5 stands as a standing challenge to any stronger reading.

= Computational verification <sec:verification>

Every quantitative statement in this paper is reproduced from first principles by a deterministic, network-free Python contract, `scripts/research/168_biology_validation_contract.py`, enforced in CI by `scripts/ci/168_biology_validation_gate.sh`. The contract comprises ten clauses: C1 (brute-force $343 = 133 + 42 + 168$), C2 (octonion multiplication table: exactly 168 nonzero basis associators, all of norm 2), C3 (Fano-plane design facts; pairs cannot discriminate), C4 (CYP locus audit against NCBI Gene cytogenetic bands), C5 (gauge analysis: 1008 bijections, 6 classes, 1 invariant line; 336 bijections, 2 classes with the big-three constraint), C6 (Hamming–hydrophobicity table under both stop conventions, $r = 0.199$ / $0.218$, $rho = 0.208$), C7 (permutation test, $p = 0.015$), C8 (encoding sweep; non-optimality), C9 (mutation robustness with baselines), C10 (Fano-line class-coherence null). The gate fails if any clause ceases to reproduce the published values.

= Discussion

The framework presented here is a mathematical model, not a claim that biological systems are governed by non-associative algebra. The value of the model lies in its testable predictions and its capacity to organise known phenomena (DDI order-dependence, mutation robustness) within a unified geometric structure.

Several limitations should be noted. The CYP450 mapping requires the FDA's seven-isoform simplification; in reality, additional CYP enzymes (CYP2E1, CYP3A5, and others) contribute to drug metabolism, and the restriction to seven is a regulatory convention. The mapping is additionally under-determined: of the seven line assignments in @table:lines, only the CYP2C line is invariant across the six equivalence classes of bijections compatible with the CYP2C-cluster constraint (see the gauge analysis above). The genetic code embedding depends on the specific binary encoding (@table:bases); other biochemically motivated encodings are possible, and the chosen one is provably *not* hydrophobicity-optimal (clause C8) — its role is structural, not statistical. The ambient Cayley–Dickson algebra at dimension 64 is not directly involved in codon biology — it provides the algebraic context in which the projective geometry acquires its associator structure.

The strongest immediate application is the CYP450 model, where the seven-element constraint is externally validated (by regulatory practice), the phase-grouping dependence is clinically relevant to phased treatment design, and the predictions are testable against clinical trial data.

== Preliminary analysis: FAERS temporal asymmetry (negative result)

As an initial probe, we queried the FDA Adverse Event Reporting System (FAERS) via the openFDA API for all $binom(7, 3) = 35$ unordered CYP isoform triples, measuring _temporal_ asymmetry (which drug was started first). Of 23 triples with temporal data (5 Fano, 18 non-Fano), the mean temporal asymmetry was 0.469 for Fano triples and 0.424 for non-Fano triples (difference $-0.044$, permutation $p = 0.61$). This is a null result.

*Critically, this test addressed the wrong algebraic property.* FAERS temporal asymmetry measures _commutativity_ (whether Drug A before Drug B differs from Drug B before Drug A). The Fano classification predicts _associativity_ (whether phase grouping matters). As noted above, commutativity does not discriminate between CYP triples in the Cayley–Dickson framework. The null result is therefore expected and does not test the actual prediction.

A valid test of Prediction 1 would require phased clinical trial data — studies where the same three drugs are administered in different _phase groupings_ (e.g., $(A+B)$ then $C$ vs. $A$ then $(B+C)$), not merely different temporal orderings. We report the FAERS analysis in the interest of transparency and to delineate what the model does and does not predict.

#bibliography("168-biology-refs.yml", style: "springer-mathphys")
