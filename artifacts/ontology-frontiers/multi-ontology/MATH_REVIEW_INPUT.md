# Round 13 math claims for review — EL+ role-aware closure on GO root cones + CL/UBERON

Context: rounds 9-12 of artifacts/ontology-frontiers built a role-aware
EL+ boolean closure (executable mirror of the Lean-verified 8-rule
saturation, crStep / closeSatF of formal/OntologyELPlusClosureComplete.lean)
and scaled it to the FULL GO go-plus ontology (38,245 classes, 92 roles)
via a bitmask reduction validated against the general set-based fixpoint.
Round 13 (this review) runs the SAME engine on: (a) the three top-level
GO root cones separately (GO:0008150 biological_process, GO:0005575
cellular_component, GO:0003674 molecular_function), cut from the round-12
full-GO extraction; (b) two additional OBO ontologies, CL (3,335
CL-namespace classes, 29 roles) and UBERON (14,975 UBERON-namespace
classes, 128 roles), extracted under the round-12 namespace-only policy
with role axioms from ro.owl.

## Claim 1 (cone partition + atomic decomposition)
The three descendant cones of GO:0008150 / GO:0005575 / GO:0003674 under
stated subClassOf PARTITION the 38,245 full-GO classes (pairwise
disjoint, union = all; no class has a stated parent outside its cone —
all measured). Therefore the per-cone ancestor-closure edge counts sum
EXACTLY to the round-12 full-GO atomic closure edge count:
298,203 + 23,943 + 73,793 = 395,939. (Asserted in the generator.)

## Claim 2 (conflict decomposition + independent recomputation)
conflict(c1,c2) := some atom ancestor of c1 is disjoint with some atom
ancestor of c2 (ordered pairs, c1 != c2). Because the cones partition the
classes, full-GO conflicts split into intra-cone (sum over cones =
29,770,678) plus cross-cone (763,044,078, 96.24%). The full total was
recomputed from the per-cone ancestor masks alone (lifted back to global
ids) with an INDEPENDENT grouped counter:
  since pm[c] = union of partner-bitmasks over the set bits of epm[c] is
  a FUNCTION of epm[c], classes group by endpoint mask v, and
  conf = sum_{v1} sum_{v2 : pm(v1) & v2} n(v1)*n(v2)
       - sum_{v : pm(v) & v} n(v)   (diagonal exclusion).
Result: 792,814,846 = exactly the round-12 full-GO value (which was
computed by the O(n_actors^2) pair scan). The grouped counter also agrees
with the pair scan on all 5 round-13 targets (asserted).

## Claim 3 (role-edge decomposition is NOT an identity — measured deficit)
The sum of per-cone atom-source role edges (1,632,115) is strictly less
than the round-12 full-GO value (2,135,207) because 3,603 stated
existential restrictions cross cones (1,860 BP / 813 CC / 930 MF, measured
by counting stated (c,r,f) with c in the cone and f outside). The deficit
503,092 is the closure attributable to those dropped cross-cone
restrictions; no exact identity is claimed for role edges. The only
cross-cone disjoint pairs are the three root pairs MFxBP, MFxCC, BPxCC
(measured: each cone reports exactly 2 half-cross pairs).

## Claim 4 (profile theorem still covers CL/UBERON extraction)
Rounds 11-12's profile theorem (no extracted conjunctions, no
superclass-side restrictions => roles add no atomic subsumptions or
conflicts; atom-row existential targets = role-edge fillers by the
stoR/RtoS bijection) is applied to CL and UBERON. Probe results: CL has 1
superclass-side restriction and 2 equivalentClass restrictions; UBERON
has 1 and 15 (all skipped by the syntactic extractor and counted).
Claim: superclass-side restrictions (ex r.F <= C) cannot change any
ATOM-level statistic: they only add role edges whose SOURCE is an
existential concept, and no completion rule produces an atom-atom
subsumption or an atom-sourced edge from them (RtoS maps edge (r,X,D) to
X <= ex r.D for any source X; if X is existential the conclusion is not
an atom row; no rule maps an existential-subsumer axiom to an
atom-subsumer one).

## Claim 5 (word-generalized endpoint masks in the driver)
The driver's conflict computation was generalized from a fixed 2-word
endpoint mask (<=128 endpoints, round 12) to NEPW words
(epm[c*NEPW+w], pm[c*NEPW+w]) because UBERON has 589 disjoint pairs with
822 distinct endpoints. Claim: the word-index math (word = k/64, bit =
k - (k/64)*64, k = ep_id-1 1-based) computes the same pm/epm semantics as
round 12, hence the same conflict relation; validated by driver == mirror
on all 5 targets (incl. GO cones with NEPW=2, matching round-12 shape,
and UBERON with NEPW=26).

## Measured results (driver == mirror on every number, ALL PASS)

| target | H | NR | atomic edges | role edges | conflicts | rounds |
|---|---|---|---|---|---|---|
| GO BP | 24,129 | 32 | 298,203 | 1,480,543 | 21,144,668 | 4 |
| GO CC | 4,075 | 7 | 23,943 | 105,887 | 8,621,578 | 4 |
| GO MF | 10,041 | 28 | 73,793 | 45,685 | 4,522 | 3 |
| CL | 3,335 | 29 | 37,926 | 146,188 | 1,071,098 | 5 |
| UBERON | 14,975 | 128 | 150,515 | 2,343,535 | 25,001,610 | 7 |

Ablations (role edges without the family / contribution): roleSub
dominates everywhere: BP 75%, CC 46%, MF 77%, CL 75%, UBERON 56%
(roleComp 36% in UBERON — its largest share anywhere so far; in the
round-11 GO slice roleComp dominated with 42% of 21,628 edges).

Question: are claims 1-5 correct as stated? In particular: is the
grouped-counter formula (Claim 2) a faithful count of the ordered
conflict pairs; is the Claim-4 argument that superclass-side restrictions
cannot affect atom-level statistics sound within the 8-rule system; and
is the Claim-3 deficit attribution (measured, not derived) framed
honestly?
