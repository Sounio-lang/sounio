# DISPATCH — Track 2: Solver SOTA-class on graph-colouring UNSAT

<!-- docs:meta
topic_id: repo.research.solver-sota-class-dispatch
authority: research
audience: agents
last_validated: 2026-06-26
-->

**Branch:** `research/solver-sota-class`
**Worktree:** `/workspace/sounio-solver-sota`
**Scope:** independent — this track does NOT depend on Track 3 or on the Lean χ≥5 track
**Level 3 target:** domain-competitive CDCL solver on the graph-colouring UNSAT family, with honest PAR-2 measurement vs kissat 4.0.4

## Artefacts at HEAD

- `examples/erdos/souc_sat.sio` — CDCL core (2-watched-literal, 1-UIP, LRB/VSIDS, LBD deletion, chrono-BT, streamed DRAT). SOTA doc (2026-05-29) claims G₅₂₉ refutation in ~31s with checked proof.
- `stdlib/theorem/smt.sio` — epistemic DPLL(T) heuristic engine (Thompson sampling, Beta-Bernoulli polarity). Separate artefact; do not confuse.
- `examples/erdos/SOTA_LITERATURE_AND_PLAN_2026-05.md` — calibrated literature review + lever plan (S1–S5). **Read before starting.**

## Blockers (typed per PARALLEL_BLOCKER_CONTRACT)

### BLOCKER-SOTA-B1 — souc_sat.sio does not compile on current HEAD
- **Class:** `compiler-semantics` / `gate-regression`
- **Severity:** B1 (lane-blocking)
- **Evidence:** E1 — reproduced: `SOUNIO_STDLIB_PATH=$(pwd)/stdlib ./bin/souc check examples/erdos/souc_sat.sio` → exit 1, 30 errors (E008, E001, E004 families). Compiler drift since last green run.
- **Owner:** Track 2 agent
- **Acceptance gate:** `souc check` exits 0; `souc run` refutes the G₅₂₉ 4-colouring SAT instance and exits UNSAT.
- **Next action:** diff souc_sat.sio against last-known-green commit (`git log --oneline -10 -- examples/erdos/souc_sat.sio`); identify which compiler change broke it.

### BLOCKER-SOTA-B3 — reference solver toolchain not installed
- **Class:** `platform-resource`
- **Severity:** B3 (evidence-blocking)
- **Evidence:** E0 — `which kissat cadical march sbva drat-trim` returns nothing.
- **Owner:** Track 2 agent (or escalate to operator for package install)
- **Acceptance gate:** kissat 4.0.4, CaDiCaL-SC2025, drat-trim, SBVA installed; versions recorded.
- **Next action:** build from source or request install. Without these, no honest PAR-2 — do NOT fabricate comparison numbers.

## Phased plan (from SOTA_LITERATURE_AND_PLAN, levers S1/S3/S5 first)

| Phase | Deliverable | Gate |
|---|---|---|
| **P0** | Repair souc_sat.sio to compile on HEAD (close B1) | `souc check` exit 0 + G₅₂ₒ refutation runs |
| **P1** | Install reference toolchain (close B3) | kissat/cadical/drat-trim/sbva present |
| **P2** | Honest PAR-2 benchmark: souc_sat vs kissat on graph-colouring UNSAT family (G₅₂₉, parts_510, SATLIB colouring) | published table, no fabrication |
| **P3** | SBVA as external preprocessor (S1): sbva → souc_cat → stitch DRAT → drat-trim validates | end-to-end DRAT proof accepted |
| **P4** | Arena clause layout + inline assign[] in propagate (S3): kill the 1.5B litval dominance | proof unchanged, drat-trim still passes, constant-factor speedup measured |

**Deliberately deferred (multi-week, do not start):** S2 in-solver inprocessing (BVE/vivification) — RAT-proof emission is the hard part and must not be rushed. S4 cube-and-conquer — needs cluster, separate dispatch.

## Discipline

- Every performance claim gated by drat-trim / cake_lpr. No fabrication.
- Heavy builds through `scripts/dev/souc-build-lock.sh`. Cheap `souc check` does not need lock.
- Math/bound claims → `bin/llm-offload -t math-review -p xai` before commit.
- Do NOT edit files owned by Track 3 (see ownership table below).

## File ownership (disjoint from Track 3)

| Owned by Track 2 | Owned by Track 3 (DO NOT TOUCH) |
|---|---|
| `examples/erdos/souc_sat.sio` | `stdlib/hypercomplex_graph/erdos_unit_distance.sio` |
| `examples/erdos/SOTA_LITERATURE_AND_PLAN_2026-05.md` | `examples/erdos/168_*.sio` |
| `examples/erdos/data/parts_510.edge` (read-only) | `examples/erdos/degrey_*.sio` |
| benchmark harness (new, TBD) | sedenion surgery code (new, TBD) |
