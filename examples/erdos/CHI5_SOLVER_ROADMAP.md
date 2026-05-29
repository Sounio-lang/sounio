# Roadmap: a Sounio SAT solver that surpasses kissat

> Status: planning. Author target — single author, the same one who wrote the
> self-hosted Sounio compiler in < 90 days. This document is deliberately ambitious
> *and* concrete: every phase ends in a runnable artefact and an external
> verification gate. No milestone is "claimed" until a machine check passes.

---

## 0. Honest framing — what "surpass kissat" means

kissat (Armin Biere) is ~20 years of CDCL research distilled into ~50k lines of
hand-tuned C and a multi-year SAT-Competition winning streak. Beating it **in the
general SAT-Competition aggregate** in one pass is not a realistic first target, and
saying otherwise would be drift (CLAUDE.md §6.1, §6.7).

So we define *three fronts* where Sounio can genuinely win, ordered by how decisively
"win" can be defended:

| Front | What "surpass" means | Why Sounio can win |
|---|---|---|
| **F1 — Verified performance** | The **fastest *formally-verified* SAT solver**: a competitive CDCL whose core is proven correct in Lean 4 (the repo already has `formal/lean4/`). kissat is *not* verified; verified solvers (e.g. versioned IsaSAT) are far slower. | Sounio has effects, refinement types, and a Lean pipeline. A solver that is *both* fast *and* end-to-end verified is a new point on the Pareto front — a CAV/POPL contribution, not a benchmark footnote. |
| **F2 — Domain dominance** | Beat kissat **wall-clock on the geometric / unit-distance / graph-colouring family** (de Grey, Heule cores, Ramsey-type, Hadwiger–Nelson). Specialisation beats generality on a fixed family. | We control the *encoding* and can fuse exact-arithmetic structure (symmetry, geometry) the way a black-box solver cannot. |
| **F3 — Proof-native throughput** | Emit + check proofs **faster and smaller** than the DRAT/LRAT toolchain kissat feeds (binary LRAT with hints, GPU-checked). | Sounio owns the whole stack (solver → proof → checker → Lean), so we can co-design the proof format and the checker. |

The de Grey χ ≥ 5 certificate is the **flagship demo** of F1+F2+F3 together: exact
rational geometry → CNF → UNSAT → machine-checked proof → Lean theorem. Closing it is
Phase 4; the solver work that gets us there is Phases 1–3.

---

## 1. Where we are (measured, this repo)

`examples/erdos/cdcl_fast.sio` is a from-scratch CDCL with:

- two-watched-literal propagation, integer VSIDS + decay, phase saving;
- inner/outer (MiniSat-style) restarts;
- **LBD clause deletion** with DRUP `d` records — externally **`drat-trim` verified**
  (K₇ proof with 1136 deletions; non-vacuity rejects a corrupted lemma);
- in-memory DRUP buffer + an independent native RUP checker.

Measured ceiling: on de Grey's real 1581-vtx 4-colouring it sustains ~2700
conflicts/s at **bounded memory** (40k–230k live clauses) for 8.6 M conflicts without
closing UNSAT; the in-memory proof buffer fills at ~7.5 M conflicts.

**Gap to kissat** is therefore *search quality* and *engineering*, not data or memory.
The missing pieces below are exactly the deltas between a textbook CDCL and a modern
one.

---

## 2. The modern-CDCL gap (Phase 1–2 backlog)

Ordered by expected impact on hard structured UNSAT (the de Grey regime):

### 2.1 Decision heuristics
- [ ] **VSIDS → EVSIDS** (exponential VSIDS with float-free fixed-point decay; we
      already use integer activities — switch to the rescaling EVSIDS scheme).
- [ ] **LRB (Learning-Rate Branching)** + **CHB** — the heuristics that beat VSIDS on
      structured instances (Liang et al., 2016). Maintain a per-variable learning-rate
      EMA; this is *the* single highest-leverage change for de Grey-type problems.
- [ ] **Mode switching** (CaDiCaL "stable vs focused"): alternate LRB/VSIDS phases.

### 2.2 Restarts & phases
- [ ] **Glucose-style LBD-based dynamic restarts** (restart when the recent-LBD EMA
      exceeds the global LBD average × K) instead of fixed inner/outer.
- [ ] **Reluctant doubling / Luby** as the stable-mode schedule.
- [ ] **Rephasing** with multiple strategies (saved / inverted / random / best-found).
- [ ] **Target phases / local search rephasing** (walk-based phase seeding).

### 2.3 Clause database
- [x] LBD clause deletion (done).
- [ ] **Tier system** (CaDiCaL): core (LBD ≤ 2) / tier-2 (LBD ≤ 6, used-recently) /
      local; age + activity based eviction with `used` bits.
- [ ] **Clause activity** bumping (alongside LBD) for the local tier.

### 2.4 Inprocessing (the big one)
- [ ] **On-the-fly self-subsumption** during conflict analysis (clause minimisation —
      *recursive* minimisation à la MiniSat, removes redundant learned literals;
      typically 30–50% shorter clauses → big speedup *and* smaller proofs).
- [ ] **Vivification** (asymmetric branching to strengthen clauses).
- [ ] **Bounded Variable Elimination (BVE)** + **subsumption** + **self-subsuming
      resolution** as a preprocessing and periodic inprocessing pass.
- [ ] **Equivalent-literal substitution** (SCC over the binary implication graph),
      **failed-literal probing**, **hyper-binary resolution**.
- [ ] Every inprocessing step must **emit its DRAT/LRAT lemmas + deletions** so the
      proof stays valid — this is non-trivial and is gated by Phase 3.

### 2.5 Engine
- [ ] **Blocking literals** in watch lists (cache the other watch's value to skip
      clause access) — measurable propagation speedup.
- [ ] **Chronological backtracking** (Nadel & Ryvchin 2018) for the cases where it
      helps.
- [ ] **Cache-friendly clause arena** (we already use flat arrays — formalise the
      arena layout, 32-bit clause refs, aligned literals).
- [ ] **Phase-saving + trail reuse** on restart (don't always cancel to 0 in focused
      mode).

**Validation rule for all of Phase 1–2.** Each feature lands behind the *same*
correctness gate: the pigeonhole + small-graph suite must stay `RUP:VERIFIED` *and*
`drat-trim s VERIFIED`, with the new inprocessing lemmas/deletions present. A feature
that can only be made to pass by weakening the proof is rejected (CLAUDE.md §6.6).

---

## 3. Proof infrastructure (Phase 3) — required before de Grey scale

The in-memory DRUP array is the current hard limit. de Grey-scale needs:

- [ ] **Streamed proof to disk** (write lemmas/deletions as produced; no global array).
- [ ] **Binary DRAT** output (compact) — and/or
- [ ] **LRAT** (Linear RAT with resolution hints) emitted directly. LRAT is *checkable
      in linear time* and is what modern verified toolchains use.
- [ ] **Native Sounio LRAT checker** (forward, hint-guided) — fast, no backward search.
- [ ] **Formally-verified LRAT checker in Lean** (the `cake_lpr` / `lrat-check` idea):
      a Sounio→Lean-extracted or Lean-proven checker whose soundness theorem is
      `checker accepts ⇒ formula UNSAT`. This is the keystone of Front F1.
- [ ] **GPU-checked proofs** (optional, Front F3): the repo has PTX codegen; RUP
      replay is embarrassingly parallel across independent lemma cones.

Gate: re-verify all Phase-1/2 certificates through the new LRAT path *and* drat-trim
*and* the Lean checker. Three independent checkers must agree.

---

## 4. The de Grey χ ≥ 5 flagship (Phase 4)

Two parallel attack lines; either suffices for the certificate.

### 4.1 Smaller 5-chromatic cores first
- [ ] Acquire / reconstruct the **minimised cores** (Heule's 510-vertex graph and the
      ~500–600-vertex family). Verify each core's unit-distance edges **exactly** in
      Sounio (we already do this for the 1581-vtx graph with `i64` fixed-point; upgrade
      to exact ℚ(√3,√5,√7,√11) integer tuples, the degree-16 kernel already in
      `degrey_fieldtower.sio`).
- [ ] Close the 4-colouring UNSAT of a core with the Phase-1/2 solver; emit LRAT;
      verify with drat-trim + Lean checker. **This is the first real χ ≥ 5 certificate.**

### 4.2 Symmetry-aware encoding (domain win, Front F2)
- [ ] de Grey's graph has a large rotation/reflection automorphism group. Add
      **symmetry-breaking predicates** (lex-leader on colour classes) at the CNF level —
      this is where owning the encoder beats a black-box solver.
- [ ] **Cube-and-conquer**: split on a few high-degree vertices' colours, solve cubes
      in parallel (Front F3, GPU/cluster), recombine proofs.

### 4.3 The end-to-end Lean theorem (the prize)
- [ ] Formalise in Lean 4: *"if `G` is a finite unit-distance graph in ℝ² (edges
      certified at exact distance 1 over the number field) and the 4-colouring CNF of
      `G` is UNSAT (per a verified LRAT proof), then χ(ℝ²) ≥ 5."* Compose:
      `geometry (exact) → graph → CNF (sound encoding lemma) → UNSAT (LRAT checker
      theorem) → χ ≥ 5`. End-to-end machine-checked, Sounio-native. **POPL/CAV-grade.**

---

## 5. Surpassing kissat — the competitive phase (Phase 5)

Once Phase 1–3 land, benchmark honestly:

- [ ] **Benchmark harness**: SATLIB + SAT-Competition `main`/`structured` tracks, plus
      our geometric family. Report PAR-2, solved-count, and proof sizes vs kissat 4.x
      and CaDiCaL, same hardware (the `cpu-ops` node; document exact CPU/RAM).
- [ ] **F2 first**: demonstrate wall-clock dominance on the geometric/colouring family
      (where specialised encoding + symmetry + structure-aware heuristics should win).
- [ ] **F1 next**: publish the verified-solver Pareto point (fastest *verified*).
- [ ] **F3**: smaller/faster proofs + GPU checking.
- [ ] Only then, if the data supports it, claim a head-to-head win on a defined track —
      with a re-runnable command, per CLAUDE.md §6.1.

### Sounio-native advantages to exploit (the "edge of novelty", CLAUDE.md §10)
- **Refinement types** to make solver invariants *checked by the type system*
  (e.g. "watched literal index < clause length", "trail level monotone") — fewer bugs,
  and the invariants double as Lean proof obligations.
- **Algebraic effects** to cleanly separate the pure search core (verifiable) from
  `IO`/`Alloc`/proof-emission effects — the verified core has *no* side effects.
- **Linear types** for the clause arena (no aliasing bugs, deterministic frees).
- **GPU effect** (`with GPU`) for parallel propagation / proof checking / portfolio.
- **`Knowledge[T]` / epistemic types** for a principled portfolio that tracks
  confidence across solver configs (a genuinely novel angle).

---

## 6. Phase plan & gates (summary)

| Phase | Deliverable | External gate |
|---|---|---|
| **P1** | LRB/CHB + clause minimisation + tiered DB in `cdcl_fast` | pigeonhole + small graphs `drat-trim s VERIFIED`, with minimised clauses |
| **P2** | Inprocessing (BVE, vivification, probing) with proof emission | same suite + new lemmas, 3-checker agreement |
| **P3** | Streamed LRAT + native + Lean LRAT checker | de Grey-scale proof checkable on disk; Lean checker soundness theorem |
| **P4** | First χ ≥ 5 certificate (small core) → full de Grey | drat-trim + Lean checker accept; geometry exact in number field |
| **P5** | Benchmark vs kissat/CaDiCaL; claim a defended front | re-runnable PAR-2 table on documented hardware |

**Effort estimate (single author, calibrated to the 90-day compiler):** P1 ≈ 2–3
weeks, P2 ≈ 3–4 weeks, P3 ≈ 3–4 weeks (Lean checker is the long pole), P4 ≈ 2–3 weeks
(cores) + open-ended (full de Grey), P5 ongoing. The verified-checker (P3) and LRB
(P1) are the highest-leverage items — do them first.

---

## 7. Risks & honesty guards

- **"Surpass" inflation.** Never claim a general kissat win from a domain or verified
  result. State the front explicitly and show the command (CLAUDE.md §6.1, §6.7).
- **Proof-weakening drift.** A feature that needs the proof relaxed to pass is a
  rejected feature, not a passed gate (§6.6).
- **Inprocessing ↔ proof coupling.** Every BVE/vivification/probing step must emit
  correct DRAT/LRAT; this is the subtlest correctness risk — gate each step in
  isolation before composing.
- **de Grey may stay open** under our solver for a long time; the smaller-core line
  (P4.1) is the de-risked path to a *real* χ ≥ 5 certificate.
- **Single arbiter dependence.** Keep ≥ 2 independent checkers (drat-trim + Lean LRAT)
  so no single checker bug can pass an unsound proof.

---

## 8. Immediate next action (when P1 starts)

1. Add **recursive clause minimisation** to `analyze` in `data/degrey/gen_solver.py`
   (biggest single win, smaller proofs, low risk) and re-run the gate.
2. Add **LRB** alongside VSIDS with mode switching.
3. Re-run de Grey on SLURM; measure conflicts-to-first-core-shrink and whether the
   `trail` starts trending toward a level-0 refutation.

Everything in this file is gated, re-runnable, and externally checkable. That is the
only way a solo solver beats a 20-year-old institution: not by claiming, by *proving*.
