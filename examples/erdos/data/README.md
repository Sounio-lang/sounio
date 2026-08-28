<!-- docs:meta
topic_id: repo.examples.erdos.data.readme
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#repo.examples.erdos.data.readme
-->

# de Grey / Parts unit-distance core data (provenance)

Edge lists and exact-coordinate files for small 5-chromatic unit-distance graphs,
used by `souc_sat.sio` (graph-from-file mode) to certify chi >= 5.

| file | graph | vertices | edges | format | source |
|---|---|---:|---:|---|---|
| `degrey_529.edge` | G529 (Heule) | 529 | 2670 | DIMACS `p edge N M` / `e u v` (1-based) | github.com/marijnheule/CNP-SAT edge/529.edge |
| `degrey_529.vtx`  | G529 (Heule) | 529 | -- | Mathematica points `{x,y}`, exact (Sqrt[3], Sqrt[11/3], Sqrt[5]) over Q(r3,r5,r11) | CNP-SAT vtx/529.vtx |
| `parts_510.edge`  | G510 | 510 | 2504 | DIMACS edge | CNP-SAT edge/510.edge |

`data/degrey/` (earlier session) holds the full 1581-vertex de Grey graph.

## Part A -- non-4-colourability (DONE)

`souc_sat` reads `degrey_529.edge`, builds the 4-colouring CNF (529 at-least-one +
2670*4 edge clauses) + one sound triangle-precolour unit triple, refutes it
(~327k conflicts), streams a ~72 MB DRAT proof, and `drat-trim` returns
`s VERIFIED`  ==>  chi(G529) >= 5. The 3 precolour units are satisfiability-
preserving (a real triangle WLOG takes colours 0,1,2), so augmented-UNSAT implies
the original CNF is UNSAT.

## Part B -- exact unit-distance realisation (TODO)

`degrey_529.vtx` has exact algebraic coordinates over Q(sqrt3,sqrt5,sqrt11).
Certifying every edge dist^2 = 1 exactly (degree-16 field kernel in
`degrey_fieldtower.sio`) connects the abstract graph to the plane ==> chi(R^2) >= 5.
Upstream uses Singular/Groebner (`check/*.singular`); the Sounio version is next.

## Reproduce Part A

    export SOUNIO_SOUC_BIN="$PWD/artifacts/self-hosted/souc-self-hosted-x86_64"
    $SOUNIO_SOUC_BIN examples/erdos/souc_sat.sio /tmp/souc_sat.elf
    cd /tmp && /tmp/souc_sat.elf 0 4 1 1 "$OLDPWD/examples/erdos/data/degrey_529.edge"
    drat-trim souc_sat_worker.cnf souc_sat_worker.drat   # => s VERIFIED

Args: `souc_sat <seed> <k> <lrb> <sb> <edgefile>` -- k=4 colours, LRB on,
SB=1 (triangle precolour; without it our CDCL does not close in 300 s).
