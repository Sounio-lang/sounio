# Math-review input — Round 15 ChEBI + PATO EL+ closure

Review the following claims for arithmetic correctness, scope honesty,
and whether the conclusions follow from the stated measurements.
Flag any overclaim relative to atom-level / namespace-only extraction.

## Measurements (Python sparse/bitmask mirror == Sounio driver)

### ChEBI (namespace-only CHEBI_, deprecated excluded)

- H=218253, NR=13, sub=307965, exsub=109108, disj=0, roleSub=10, roleComp=4
- atomic_edges=5297445
- role_edges (atom-source)=29846298
- conf=0
- fixpoint rounds=5
- ablation no-roleComp role_edges=22643261
- ablation no-roleSub role_edges=12049267
- super-side restrictions=0, equivalentClass restrictions=0

### PATO (namespace-only PATO_)

- H=1887, NR=12, sub=2227, exsub=37, disj=67, roleSub=5, roleComp=2
- atomic_edges=12433
- role_edges=4005
- conf=471806
- rounds=2
- no-roleComp=4005, no-roleSub=2287
- super-side restrictions=1 (skipped), equivalentClass restrictions=0
- declared classes before namespace cut: 7643

## Claims to review

**Claim 1 — Amplification arithmetic**

- ChEBI: 29846298 / 109108 = 273.546… ≈ **273.5×**
- PATO: 4005 / 37 = 108.243… ≈ **108.2×**

**Claim 2 — Ablation contributions**

Define contrib(X) = full − ablation_without_X.

ChEBI:

- roleComp contrib = 29846298 − 22643261 = **7203037** (7203037/29846298 ≈ **24.13%**)
- roleSub contrib = 29846298 − 12049267 = **17797031** (17797031/29846298 ≈ **59.63%** ≈ **60%**)
- Note: roleComp and roleSub contributions are **not** a partition of
  the full count (edges can depend on both families; 24%+60%=84% < 100%).

PATO:

- roleComp contrib = 4005 − 4005 = **0** (0%)
- roleSub contrib = 4005 − 2287 = **1718** (1718/4005 ≈ **42.90%** ≈ **43%**)

**Claim 3 — roleSub dominates**

On both targets, roleSub contribution ≥ roleComp contribution
(ChEBI 60% > 24%; PATO 43% > 0%). Same qualitative pattern as round-13
GO cones / CL / UBERON.

**Claim 4 — ChEBI conf=0 is structural**

With disj=0 in the extracted TBox, the atomic conflict count is
identically 0 (no disjointness endpoints). This is not evidence of a
broken conflict counter.

**Claim 5 — Profile / inertness honesty**

- ChEBI: 0 super-side and 0 equivalentClass restrictions → profile-theorem
  premise holds for the extracted TBox.
- PATO: 1 super-side restriction skipped; reported for honesty; numbers
  are exact for the extracted TBox, not full OWL semantics (same framing
  as round-13 CL/UBERON).

**Claim 6 — Capacity sizing**

Measured peaks: role arena 46234910 words, ancestor arena 13090818,
cells non-empty 1108632. Capacities ARENA=67108864, AARC=33554432,
CELLS=NRC*HC=32*220000=7040000 are strictly larger. H+1=218254 ≤ HC=220000;
NSUB=307965 ≤ SUBC=320000; NEX=109108 ≤ EXC=120000.

## Questions for the reviewer

1. Any arithmetic error in Claims 1–2?
2. Is the non-partition caveat for dual ablation contributions stated
   clearly enough?
3. Any overclaim about “largest” / completeness vs namespace-only scope?
4. Anything else that should be hardened before treating this as a
   science closeout receipt?
