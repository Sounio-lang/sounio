# Real-data validation of the verified ontology pipeline — OAEI 2016 Anatomy

**Date:** 2026-08-02 · **Lane:** `kimi-swarm--real-data-20260802` ·
**Compiler:** `bin/souc` (Madaros engine), branch
`research/zd-fiber-antisymmetry-lemma-20260731`.

The verified pipeline of the ontology-frontiers line (subsumption-closure
fixpoint → closure-expanded disjointness → derived alignment conflicts →
greedy epistemic repair, prototyped in `../el-grounding/el_conflict_demo.sio`
and proved in `formal/OntologyELReasoner.lean` /
`formal/OntologyAlignmentRepair.lean`) is here run **unchanged** on real
biomedical ontologies: the OAEI 2016 Anatomy track (adult mouse anatomy vs
NCI Thesaurus human anatomy). An independent python mirror of the same
algorithms is embedded in the generated data module as `expected_*()`
functions; the Sounio driver prints `ALL PASS` only if its internal sanity
checks hold **and** every number agrees with the mirror.

## 1. Downloads (exact, reproducible)

Official OAEI 2016 Anatomy dataset, downloaded 2026-08-02 (~18:17 UTC):

```
curl -sSL -o downloads/anatomy-dataset.zip \
  https://oaei.ontologymatching.org/2016/anatomy/anatomy-dataset.zip
```

| file | bytes | sha256 |
|---|---|---|
| `downloads/anatomy-dataset.zip` | 321,164 | `21aa4dcd457c5dec7a8c0c968f2c681f44f6f9b9091adc4f543adbad514ae141` |
| `downloads/mouse.owl` | 1,402,405 | `93756393d6306c8f332c884401aa447cc4b2557f0a6be3e0efd988a943cb68f8` |
| `downloads/human.owl` | 3,436,928 | `5c1ca432d9845f1abb36ecfce313df46d8c56d2fb4137d0590ec4e4a6a9b05bf` |
| `downloads/reference.rdf` | 329,284 | `b6a6b12f3e7a786e5b58c4898024f0f483426e9036bdc3939aa61b046dbc26c4` |

(zip extracted with `python3 -m zipfile -e` — no `unzip` on this machine).
The byte sizes of `mouse.owl`/`human.owl` coincide with the independent
GitHub mirror `ernestojimenezruiz/oaei-evaluation`
(`ontologies/anatomy/{mouse2012,human2012}.owl`), cross-verified via the
GitHub API on 2026-08-02. The official OAEI reference alignment
(`reference.rdf`, 1,516 `=` mappings) was obtained inside the same zip, so
**both** a lexical matcher and the reference are used (see §3).

## 2. TBox extraction (`extract_tbox.py`)

**Syntactic extraction** (no DL reasoning): RDF/XML parsed with stdlib
`xml.etree.ElementTree`; per ontology we take every declared named
`owl:Class` (with first `rdfs:label`), every `rdfs:subClassOf` edge between
two *declared named* classes, and every `owl:disjointWith` pair
(symmetrised). Anonymous restrictions (`rdf:nodeID`) are skipped and
counted.

```
$ python3 extract_tbox.py
mouse: 2744 classes, 2856 sub, 0 disj (skipped anon=1637, ext=0)
human: 3304 classes, 3761 sub, 17 disj (skipped anon=1662, ext=0)
```

Outputs: `classes.tsv` (6,048 classes, id↔IRI↔label) and `tbox.txt`
(6,617 sub axioms + 17 disjoint pairs). No cap at this stage.

## 3. Candidate alignment (`lexical_match.py`)

Simple lexical matcher (we **also** have the reference; the matcher is the
system under test):

- `normalize` = lowercase, non-`[a-z0-9]` → space, collapse, split;
- stopwords `{of,the,and,or,a,an,to,in,for,by}` removed (full token set if
  that empties it);
- **conf = 1.0** if normalized labels equal, else **Jaccard** on the
  content-token sets, `|T1∩T2| / |T1∪T2|`;
- candidates: conf ≥ 0.3, top-3 per mouse class (token inverted index).

```
$ python3 lexical_match.py
mouse classes: 2744, human classes: 3304
candidate mappings (conf>=0.3, top-3): 6638
mouse classes with >=1 candidate: 2541
reference mappings (=): 1516
candidates:  TP=1238 P=0.1865 R=0.8166
top-1 only:  TP=1158 P=0.4557 R=0.7639
```

(987 of the 6,638 candidates are exact-label matches with conf = 1.0;
candidate conf mean 0.5455.) Output: `mappings.tsv`.

## 4. Data module generation + python mirror (`gen_sounio_data.py`)

**Cap.** The Sounio closure is an O(H³) fixpoint over an H×H bool matrix,
so human classes are capped at `--human-cap 2000`. Selection is
**ancestor-closed** (every class referenced by a candidate mapping, plus
*all* its subsumption ancestors, plus disjointWith endpoints and their
ancestors), so the capped closure coincides with the full 3,304-class
closure on the referenced subgraph — verified by running the mirror on the
full TBox (identical 368 unordered derived conflicts; an earlier
referenced-only, non-ancestor-closed cap lost paths and yielded only 338
ordered pairs).

Selected: **H = 1,961** human classes (< 2,000 cap, no truncation),
2,266 sub axioms, all 17 disjoint pairs, all **M = 6,638** candidate
mappings (mouse entities 2,541, ids not remapped). Pairs are packed into
single i64s (`a*10000+b`; all ids < 10000) and confidences are exact i64
per-10000 (see §7).

Mirror results (identical iteration order to the driver; independently
cross-checked with a bitset Warshall implementation — same numbers):

- closure: **12,669 edges**, 4 fixpoint passes;
- closure-expanded disjointness `disjC`: **287,108 ordered entries**;
- derived conflicts (same mouse entity, disjoint-reachable human targets):
  **736 ordered pairs** = 368 unordered, across **217 mouse entities**;
  - reference breakdown (unordered): ref–ref 0, ref–nonref 74,
    nonref–nonref 294;
- greedy repair: **kept 6,392, dropped 246**;
  - dropped conf: min 0.3333, max 0.7500, mean 0.4099;
    kept conf mean 0.5507 (epistemically weak mappings removed);
  - **reference mappings dropped: only 3 of 1,238** (1,235 kept) — the
    repair preferentially removes non-reference (likely wrong) mappings;
  - top-5 dropped (lowest conf): ids 45, 46, 52, 56, 77, all conf 0.3333.

## 5. Sounio run (the verified pipeline on the real data)

```
$ ./bin/souc check artifacts/ontology-frontiers/real-data/real_repair_driver.sio
check: OK
$ ./bin/souc run artifacts/ontology-frontiers/real-data/real_repair_driver.sio
=== OAEI 2016 Anatomy: real-data pipeline validation ===
human classes (H):
1961
sub axioms:
2266
disjoint pairs:
17
candidate mappings (M):
6638
derived conflicts (ordered pairs):
736
kept:
6392
dropped:
246
top-5 dropped by confidence (lowest first; conf per-10000):
45
3333
46
3333
52
3333
56
3333
77
3333
ALL PASS
```

Wall time ~25 s total (compile + run; closure fixpoint dominates).
`ALL PASS` certifies, on the real data: closure reflexive on the diagonal;
derived conflict relation symmetric; derived-conflict count, kept/dropped
counts and the top-5 dropped **exactly equal to the independent python
mirror**; retained set conflict-free; every dropped mapping has a
maximality witness (a retained conflicting mapping of ≥ confidence).

## 6. Scale / N limits found

No bisection of N was needed for the arrays themselves: H = 1,961 with
H² = 3,847,521-cell bool matrices (×3) and M = 6,638 mappings compile and
run in ~25 s; the ancestor-closed cap made the full candidate set fit
under the 2,000-class budget, so nothing was truncated. The binding
limits found are elsewhere in the compiler (all verified by minimal
reproduction this session, Madaros v0.80.0):

1. **>682 statements per function are silently dropped.** A function with
   N array-assignment statements initialises exactly min(N, 682) elements
   (bisected: N=682 OK, N=683 loses the 683rd). Fix: `init_data()` calls
   32 chunk functions of ≤500 statements each.
2. **Module-level splat-initialised arrays contain garbage in their
   leading elements**: indices 0–2 of `bool` arrays read `true`, index 0
   of `i64`/`f64` arrays reads garbage, for sizes 8 … 3,847,521
   (reproduced in imported *and* main modules; arrays local to `main` are
   unaffected). Fully-assigned data arrays are safe (assignments overwrite
   the garbage); partially-written working matrices need an explicit
   fixup of cells 0–2, emitted first in `init_data()`.
3. **f64 array-element assignment inside a non-`main` function is a
   silent no-op** (`a[2] = 0.3333` in a called function leaves 0.0; works
   in `main`). Fix: confidences carried as exact i64 per-10000 (which
   also matches the per-mil style of `formal/OntologyClaimStatus.lean`).
4. **Multimodule thin-link fails beyond ~24k array-assignment statements
   in the imported module** (`multimodule native thin-link compilation
   failed`, rc=19; 17,842 links, 24,480 fails, content-dependent).
   Fix: pair-packing (`a*10000+b`) halves the statement count → 15,588.

These complement the pitfalls already documented in
`../compiler-repros/` (no `where` clauses; struct arrays / non-splat
arrays segfault).

## 7. Reproduction

```bash
cd artifacts/ontology-frontiers/real-data
# 1. data (or use the committed downloads/)
curl -sSL -o downloads/anatomy-dataset.zip \
  https://oaei.ontologymatching.org/2016/anatomy/anatomy-dataset.zip
cd downloads && python3 -m zipfile -e anatomy-dataset.zip . && cd ..
sha256sum downloads/*   # must match the table in §1
# 2. pipeline
python3 extract_tbox.py                    # -> classes.tsv, tbox.txt
python3 lexical_match.py                   # -> mappings.tsv (+ P/R)
python3 gen_sounio_data.py                 # -> tbox_data.sio (+ mirror)
# 3. verified pipeline on the real data
../../../bin/souc check real_repair_driver.sio   # check: OK
../../../bin/souc run   real_repair_driver.sio   # ALL PASS
```

## 8. Files

| file | role |
|---|---|
| `downloads/` | OAEI 2016 anatomy zip + extracted OWL/RDF (see §1) |
| `extract_tbox.py` | syntactic RDF/XML → TBox extractor (stdlib only) |
| `classes.tsv`, `tbox.txt` | extracted classes/axioms |
| `lexical_match.py` | Jaccard lexical matcher + P/R vs reference |
| `mappings.tsv` | 6,638 candidate mappings |
| `gen_sounio_data.py` | cap/selection, packing, python mirror, emitter |
| `tbox_data.sio` | generated data module (15.7k lines, ~546 KB) |
| `real_repair_driver.sio` | the verified pipeline driver → `ALL PASS` |

## 9. Limitations

- Extraction is **syntactic**: anonymous-restriction subsumptions
  (1,637/1,662 skipped) and any axioms not materialised in the RDF/XML are
  invisible; the mouse ontology contributes **no** disjointness axioms, so
  conflicts are derived only via the 17 human `disjointWith` pairs.
- The conflict rule is the one verified in the prototypes (same asserted
  entity, disjoint-reachable asserted classes); richer rules (e.g. via the
  mouse hierarchy) are out of scope.
- The lexical matcher is a deliberately simple baseline (top-1
  P=0.456/R=0.764); the pipeline numbers, not the matcher quality, are
  the result.
