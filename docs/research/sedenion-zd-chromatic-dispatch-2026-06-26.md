# DISPATCH — Track 3: Sedenion ZD-surgery chromatic generation

<!-- docs:meta
topic_id: repo.research.sedenion-zd-chromatic-dispatch
authority: research
audience: agents
last_validated: 2026-06-26
-->

**Branch:** `research/sedenion-zd-chromatic`
**Worktree:** `/workspace/sounio-zd-surgery`
**Scope:** independent — this track does NOT depend on Track 2 or on the Lean χ≥5 track
**Level 3 target:** novel algebraic method that generates unit-distance graphs with measurable chromatic structure, recovering known 5-chromatic graphs (G₅₂₉ / parts_510) from sedenion zero-divisor surgery

## Artefacts at HEAD

- `stdlib/hypercomplex_graph/erdos_unit_distance.sio` — ZD-surgery frame (right-multiplication by primitive ZD element v = e_lo ± e_hi). Documents the mechanism and the bipartite obstruction honestly.
- `stdlib/snn/sedenion_layer_core.sio`, `stdlib/snn/sedenion_layer_impl.sio` — live sedenion algebra in current stdlib.
- `formal/lean4/SounioZeroDivisorBridge.lean` — 84 valid primitives, 168 unordered ZD pairs (verified, no sorry).
- `formal/lean4/SounioCayleyDickson.lean` — cdSigma sign function (the multiplication table).
- `examples/erdos/data/degrey_529.{vtx,edge}` — Heule 529-vertex graph.
- `examples/erdos/data/parts_510.edge` — current world-record smallest 5-chromatic graph.
- `docs/research/erdos-168-chromatic-separation.md` — the six-step sedenion obstruction study. **Read before starting.**

## Honest mathematical frame (from existing artefacts)

Integer-coordinate unit-distance graphs are **always bipartite** (subgraph of Z^16 lattice, 2-coloured by parity). The non-trivial question Track 3 investigates: **can ZD surgery (right-multiply by primitive ZD element) break bipartite structure and force χ ≥ 3?** This is a method-novelty question, not a χ(R²)≥6 claim.

The 168-theorem / ZD-bridge structure is verified in Lean. The gap is computational: **does ZD-surgery on a concrete point set generate graphs with chromatic number > 2, and can it recover the structure of known 5-chromatic unit-distance graphs?**

## Blockers (typed per PARALLEL_BLOCKER_CONTRACT)

### BLOCKER-ZD-B1 — erdos_unit_distance.sio does not exercise full ZD-surgery pipeline
- **Class:** `compiler-semantics`
- **Severity:** B1 (lane-blocking)
- **Evidence:** E0 — file exists as a frame but needs verification it compiles and runs the surgery end-to-end.
- **Owner:** Track 3 agent
- **Acceptance gate:** `souc run` of a ZD-surgery probe produces a concrete conflict graph and measures its chromatic number (exhaustive k-colouring for small instances).
- **Next action:** compile erdos_unit_distance.sio; if broken, repair against current compiler.

### BLOCKER-ZD-B2 — no chromatic-number measurement harness for generated graphs
- **Class:** `harness-routing`
- **Severity:** B1 (lane-blocking)
- **Evidence:** E0 — no script exists that takes a surgery-generated graph and computes χ (exact for small, SAT-refutation for larger).
- **Owner:** Track 3 agent
- **Acceptance gate:** given a graph, the harness reports χ or a lower bound via SAT refutation (can borrow souc_sat from HEAD read-only, but does NOT modify it).
- **Next action:** build the measurement harness as a new file in this worktree.

## Phased plan

| Phase | Deliverable | Gate |
|---|---|---|
| **P0** | Compile + run erdos_unit_distance.sio; repair against current compiler if needed (close B1) | `souc run` produces a surgery-generated conflict graph |
| **P1** | Build chromatic measurement harness (close B2) | reports χ for known bipartite graph (=2) as sanity, then for surgery graphs |
| **P2** | Sweep all 84 primitives / 168 ZD pairs; measure chromatic structure of generated graphs | table: primitive → (graph size, edge count, χ) |
| **P3** | **The Level 3 question:** does any ZD-surgery configuration recover the edge-structure of a known 5-chromatic unit-distance graph (G₅₂₉, parts_510)? Or generate any χ≥3 graph? | measured result, honestly classified as recover / novel-χ≥3 / negative |
| **P4** | If P3 positive: document the structural correspondence. If negative: document why ZD surgery cannot escape bipartiteness (still a publishable structural result). | written analysis with `bin/llm-offload -t math-review` |

## Discipline

- This is edge-of-novelty work. Do NOT claim χ(R²)≥6. Claim exactly what is measured.
- Negative results are valid: "ZD surgery cannot escape bipartiteness for integer coords" is a structural theorem worth publishing.
- Math claims → `bin/llm-offload -t math-review -p xai` before commit.
- Do NOT edit files owned by Track 2 (see ownership table).

## File ownership (disjoint from Track 2)

| Owned by Track 3 | Owned by Track 2 (DO NOT TOUCH) |
|---|---|
| `stdlib/hypercomplex_graph/erdos_unit_distance.sio` | `examples/erdos/souc_sat.sio` |
| `examples/erdos/168_*.sio` | `examples/erdos/SOTA_LITERATURE_AND_PLAN_2026-05.md` |
| `examples/erdos/degrey_*.sio` | benchmark harness (Track 2) |
| surgery code + chromatic harness (new, TBD) | `examples/erdos/data/parts_510.edge` (Track 2 reads it too) |
