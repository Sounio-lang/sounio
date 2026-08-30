# Round 15 — EL+ role-aware closure at ChEBI scale (+ PATO)

> **These numbers cannot be reproduced from this repository, and the driver that
> produced them is no longer in it.**
>
> ChEBI's `versionIRI` names release 254; `obo/chebi/254/chebi.owl` returns 404,
> and the undated purl now serves a different, larger release (865,772,908
> bytes, sha256 `4557df5b6683…`). Measured 2026-08-24 and recorded in
> `fetch_downloads.sh`. So the committed `chebi_*` artifacts came from an older,
> unrecorded release that is no longer retrievable. PATO is unaffected — it is
> pinned at release 2025-05-14 with a sha256.
>
> `chebi_pato_elplus_driver.sio` was therefore dropped rather than carried as a
> gate entry that skips: a driver in the list that cannot run is a check
> reporting on nothing, and `ontology_multi_ontology_gate.sh` correctly exited 1
> on it. The results stay here, labelled, because a measurement whose inputs are
> gone is still a record — it is just not evidence anyone can re-derive.
>
> To make this line reproducible again, regenerate from the current purl and pin
> that sha; the published figures below will change.

**Date:** 2026-08-06 · **Lane:** `ontology-chebi-pato-scale`
· **Compiler:** `bin/souc` (Madaros v0.80.0), branch
`research/zd-fiber-antisymmetry-lemma-20260731`.

Round 13 closed five multi-ontology targets (GO BP/CC/MF cones, CL,
UBERON) under the self-validating packed format. Round 14 left the sparse
sorted-list engine in `real-data/scale/go_full_elplus_driver.sio`. This
round lifts that engine to the **largest chemical ontology in OBO** —
**ChEBI** (Chemical Entities of Biological Interest) — plus **PATO**
(Phenotype And Trait Ontology) as the small multi-target sanity leg.

## 1. Method

| piece | source |
|---|---|
| OWL extraction | `extract_tbox.parse_go` / `parse_ro` (rounds 11–12) |
| Namespace-only OBO cut | `gen_multi_data.extract_obo` (round 13) |
| Python mirror | sparse set reduce for H≥50k (`gen_chebi_data.sparse_reduce`); bitmask path for PATO |
| Ablations | no-roleComp / no-roleSub (same schedule as rounds 12–13) |
| Packed format | 13-int self-validating header (round 13) |
| Sounio engine | round-14 sparse sorted-list multi-target driver |

Policy (unchanged from rounds 12–13):

- namespace-only (`/CHEBI_`, `/PATO_`) for classes, parents, fillers, disj partners
- `owl:deprecated` excluded
- roles RO-closed from `ro.owl` (superproperties + composition targets, iterated)
- superclass-side and `equivalentClass` restrictions **probed and reported, not extracted** (profile theorem scope)

Generator: `gen_chebi_data.py`. Driver:
`chebi_pato_elplus_driver.sio` (compile-time capacities from CAPACITY
REPORT; refuses data if exceeded).

## 2. Results (Python mirror == Sounio driver, every number)

| target | H | NR | sub | exsub | disj | roleSub | roleComp | atomic edges | role edges (= ∃ targets) | conflicts | rounds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **ChEBI** | 218 253 | 13 | 307 965 | 109 108 | 0 | 10 | 4 | 5 297 445 | **29 846 298** | 0 | 5 |
| **PATO** | 1 887 | 12 | 2 227 | 37 | 67 | 5 | 2 | 12 433 | 4 005 | 471 806 | 2 |

Ablations (role edges without the named family; contribution in parentheses):

| target | no-roleComp | roleComp contrib | no-roleSub | roleSub contrib | dominant |
|---|---:|---:|---:|---:|---|
| ChEBI | 22 643 261 | 7 203 037 (**24%**) | 12 049 267 | 17 797 031 (**60%**) | roleSub |
| PATO | 4 005 | 0 (0%) | 2 287 | 1 718 (**43%**) | roleSub |

Amplification ratio, defined as **role_edges / stated exsub**
(conventional scale factor for atom-source existential blow-up; not a
derived invariant):

| target | amp | note |
|---|---:|---|
| ChEBI | **273.5×** | dense ancestor cones × RO roleSub; conf=0 because disj=0 |
| PATO | 108.2× | roleComp inert on this cut (0 contrib) |

Workspace peaks (Sounio, full ChEBI run):

- non-empty role×class cells: 1 108 632
- sparse role arena words: 46 234 910
- ancestor arena words: 13 090 818

Compile-time capacities used: `HC=220000`, `NRC=32`, `SUBC=320000`,
`EXC=120000`, `CELLS=7_040_000`, `ARENA=67_108_864`, `AARC=33_554_432`.

## 3. Extraction honesty

| target | declared classes | deprecated skipped | H (namespace-only) | super-side restr. | equivClass restr. |
|---|---:|---:|---:|---:|---:|
| ChEBI | 218 253 | 19 419 | 218 253 | **0** | **0** |
| PATO | 7 643 | 982 | 1 887 | **1** | 0 |

- **ChEBI** has 0 super-side and 0 equivalentClass restrictions on this
  cut (measurement-level inertness matching the round-11/12 profile
  premise on the *extracted* TBox; this is not a re-proof of that
  theorem). disj=0 so conflict count is trivially 0.
- **PATO** has 1 super-side restriction skipped (filler outside the
  extracted shape / inert under namespace-only — same honesty pattern as
  round-13 CL/UBERON). Declared 7 643 → H=1 887 is the namespace-only
  cut (external parents/fillers dropped), not a bug.
- Sparse mirror for ChEBI (H≥50k) replaces the O(H²/64) bitmask path
  that OOMs; PATO still runs bitmask + sparse cross-check in the
  generator path used for this emit.

## 4. Science takeaway

1. **Scale.** ChEBI is ~14× UBERON classes and ~13× UBERON role edges
   (29.8M vs 2.3M). The sparse multi-target Sounio engine closes it in
   one process with PATO, self-validating against the Python mirror.
2. **roleSub still dominates** at chemical scale (60% of ChEBI role
   edges; 43% of PATO). roleComp is material on ChEBI (24%) and inert on
   this PATO cut (0%).
3. **Amplification** stays in the same order as CL (306×) / UBERON
   (137×): ChEBI 274×, PATO 108× — hierarchy depth × RO roleSub, not a
   new regime.
4. **Zero conflicts on ChEBI** is structural (no disjointness axioms in
   the CHEBI-namespace cut), not a closure failure.

## 5. Bug fixed en route

The first ChEBI-capacity build hung after loading axioms: the
role-hierarchy transitive-closure triple loop was missing `d = d + 1`
(present in `go_roots` / `obo` drivers). Reflexive `rclos[r,r]=true`
made the inner loop infinite. Fixed; also residual `coff`/`poff` sizes
`170001` → `220001` to match `HC=220000`.

## 6. Reproduce

```bash
# from repo root
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib

# OWL inputs (symlinks to shared downloads ok)
ls artifacts/ontology-frontiers/multi-ontology/downloads/{chebi,pato,ro}.owl

# Python extract + mirror (writes packed files + CAPACITY REPORT)
python3 artifacts/ontology-frontiers/multi-ontology/gen_chebi_data.py

# Sounio: PATO then ChEBI; expects exact line ALL PASS
./bin/souc run artifacts/ontology-frontiers/multi-ontology/chebi_pato_elplus_driver.sio

# CI gate (includes this driver; raise timeout if needed)
ONTOLOGY_MULTI_RUN_TIMEOUT=900 bash scripts/ci/ontology_multi_ontology_gate.sh
```

## 7. Limitations (honest)

- Atom-level statistics only (same scope as rounds 12–14): no
  existential-source edges, no cells over the full interned universe.
- Namespace-only cut: external fillers (e.g. ChEBI→GO/UBERON) omitted.
- PATO’s single super-side restriction is reported, not closed.
- Static BSS for the ChEBI-capacity driver is ~1.5 GiB (large arrays);
  needs a machine with several GiB free RSS.
- Gate default timeout is raised for this driver (ChEBI full + 2
  ablations is slower than GO cones).

## 8. LLM-offload

Math claims (amplification arithmetic, ablation contributions, profile
inertness, capacity sizing) reviewed via
`bin/llm-offload -t math-review -p xai` on
`CHEBI_PATO_MATH_REVIEW_INPUT.md` → **no [FAIL]**; two wording
tightenables applied above (name amp as role_edges/exsub; keep
profile-premise language measurement-level). Receipt:
`CHEBI_PATO_math_review_out.txt`.
