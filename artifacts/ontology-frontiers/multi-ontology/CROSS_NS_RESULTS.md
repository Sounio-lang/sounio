# Round 16a — multi-namespace (open_fillers) vs ns_only

Policy `open_fillers`: primary-namespace classes plus any
parent/filler/disj partner of a primary subject, closed under
superclasses.  Mirrors are the same bitmask/sparse engines as
rounds 13–15.  Packed drivers remain ns_only receipts.

### pato

| metric | ns_only | open_fillers | Δ |
|---|---:|---:|---:|
| H | 1887 | 1888 | 1 |
| foreign_interned | 0 | 1 | 1 |
| sub | 2227 | 2228 | 1 |
| exsub | 37 | 37 | 0 |
| disj | 67 | 67 | 0 |
| NR | 12 | 12 | 0 |
| atomic_edges | 12433 | 14321 | 1888 |
| role_edges | 4005 | 4841 | 836 |
| conf | 471806 | 471806 | 0 |
| amp (edges/exsub) | 108.2× | 130.8× | +22.6 |
| foreign_filler dropped by ns_only | 0 | (recovered in open) | — |
| foreign_parent dropped by ns_only | 1 | (recovered in open) | — |
| super_side / equiv_restr (probed) | 1/0 | 1/0 | — |

### cl

| metric | ns_only | open_fillers | Δ |
|---|---:|---:|---:|
| H | 3335 | 6206 | 2871 |
| foreign_interned | 0 | 2871 | 2871 |
| sub | 4664 | 8964 | 4300 |
| exsub | 477 | 5697 | 5220 |
| disj | 35 | 120 | 85 |
| NR | 29 | 173 | 144 |
| atomic_edges | 37926 | 87490 | 49564 |
| role_edges | 146188 | 1204329 | 1058141 |
| conf | 1071098 | 7418464 | 6347366 |
| amp (edges/exsub) | 306.5× | 211.4× | -95.1 |
| foreign_filler dropped by ns_only | 3754 | (recovered in open) | — |
| foreign_parent dropped by ns_only | 216 | (recovered in open) | — |
| super_side / equiv_restr (probed) | 1/2 | 1/2 | — |

### uberon

| metric | ns_only | open_fillers | Δ |
|---|---:|---:|---:|
| H | 14975 | 16295 | 1320 |
| foreign_interned | 0 | 1320 | 1320 |
| sub | 19607 | 21371 | 1764 |
| exsub | 17080 | 19737 | 2657 |
| disj | 589 | 682 | 93 |
| NR | 128 | 202 | 74 |
| atomic_edges | 150515 | 229014 | 78499 |
| role_edges | 2343535 | 5190470 | 2846935 |
| conf | 25001610 | 44540444 | 19538834 |
| amp (edges/exsub) | 137.2× | 263.0× | +125.8 |
| foreign_filler dropped by ns_only | 2040 | (recovered in open) | — |
| foreign_parent dropped by ns_only | 11 | (recovered in open) | — |
| super_side / equiv_restr (probed) | 1/15 | 1/15 | — |

## Takeaway

- **pato**: 0 foreign fillers but **1 foreign parent** (the super-side / external
  parent honesty case). Interning it raises H 1887→1888 and, via denser
  ancestor rows, **+836 role edges** (4005→4841) with conf unchanged
  (disj endpoints unchanged, so conflict pairs are invariant here).
- **cl**: ns_only drops **3754 foreign-filler axioms** and **216 foreign-parent
  axioms** (axiom counts, not unique classes). Unique foreign classes interned
  under open_fillers: **2871** (H 3335→6206). exsub 477→5697 (+5220), role edges
  **146 188→1 204 329** (+1 058 141, ~8.2×). Amp falls 306.5×→211.4×: the
  *marginal* edges/exsub on the recovered slice is ≈202.7, below the ns_only
  average (not “denser restrictions” in the colloquial sense).
- **uberon**: drops **2040 foreign-filler axioms** / **11 foreign parents**;
  unique foreign interned **1320** (H 14975→16295). Role edges
  **2 343 535→5 190 470** (+2 846 935, ~2.2×). Amp *rises* 137.2×→263.0×
  (foreign lattice deepens role propagation).

If Δ role_edges ≫ 0, namespace-only **understates** the OWL TBox. Round-13/15
ns_only numbers remain exact for the *extracted* TBox; they are not
OWL-complete. Packed Sounio drivers stay on ns_only until an open_fillers
capacity resize is deliberate.

## GO full sparse reaffirmation (item 4)

`artifacts/ontology-frontiers/real-data/scale/go_full_elplus_driver.sio`
(round-14 sparse engine) re-run under Madaros 2026-08-06: **ALL PASS**
with the round-12 mirror numbers (H=38 245, role edges=2 135 207,
conf=792 814 846, rounds=4). Same sparse lineage as the ChEBI driver;
GO is not a new engine, it is the already-scaled target re-verified.
