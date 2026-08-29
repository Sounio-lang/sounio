<!-- docs:meta
topic_id: repo.examples.erdos.sota-literature-and-plan-2026-05
authority: research
audience: researchers
last_validated: 2026-05-29
validated_by: agent
-->

# SOTA literature review + continuation plan — Sounio SAT / χ(ℝ²)≥5 (2026-05-29)

**Purpose.** Calibrate against the *actual* state of the art (SAT Competition 2025, current
kissat/CaDiCaL, cube-and-conquer, SBVA, and Lean-internal verified SAT checking) and lay out an
honest, non-drift continuation plan for (A) the Sounio solver `souc_sat.sio` and (B) the
machine-checked χ(ℝ²)≥5 chain. No claim here exceeds what the cited source supports; where a goal
is hard or out of reach short-term, it is said plainly.

---

## 0. Calibration — what SOTA actually is (cited)

**SAT Competition 2025 (the relevant facts).**
- **Main-track / SAT-track winner: `AE-Kissat-MAB`** (Ding, Luo, Chu-Min Li et al.) — Kissat-MAB
  improved by an **LLM-driven configuration framework (AESAT, <$0.5 per training run)**; first
  LLM-improved solver to win a SAT competition. (satcompetition.github.io/2025; mis.u-picardie.fr/~cli)
- **UNSAT track winner — the one that matters for us: `CaDiCaL-SC2025`** (Biere, Faller, Fleury,
  Froleyks, Pollitt); `Kissat-VSA` 2nd. Our entire domain (refuting 4-colourings) is **UNSAT**.
- kissat is at **rel-4.0.4 (Oct 2025)**; "keep-it-simple bare-metal" C port of CaDiCaL with better
  data structures + inprocessing scheduling. (github.com/arminbiere/kissat)

**What CaDiCaL-SC2025 added to win UNSAT** (TU Wien proceedings, Codel-2025): they did *not* try to
beat Kissat by tuning; they **ported Kissat's inprocessing into CaDiCaL** and added LRAT proofs for
it — code grew 46→64 KLOC C++:
- **clausal congruence closure** (detect AND/XOR/ITE gates, merge equivalent outputs),
- **bounded variable addition (BVA / "factor")** — reverse-BVE via extended resolution,
- **ticks-based inprocessing scheduling**,
- **revisited vivification** (PoS'25: implicit prefix-tree/trie of clauses to reuse decisions and
  propagations across vivification candidates — Heule/Biere),
- **look-ahead lucky phases**, **clausal equivalence sweeping**, **semantic definition mining**,
  the last two via the embedded **Kitten** sub-solver.
- Hard engineering point they flag: **producing LRAT with antecedents for advanced inprocessing
  (esp. congruence closure) is the real difficulty**; Kissat sidesteps it by emitting only DRUP
  (compact, but much slower to *check*). A **new clausal proof calculus** was needed for BVA in the
  incremental setting (blocked clauses + a form of extended resolution).

**Inprocessing ranking (CaDiCaL thesis, Froleyks):** **BVE (bounded variable elimination) is
"arguably the most effective"** single inprocessing technique; then vivification, instantiation,
probing, ELS (equivalent-literal substitution), subsumption, BCE.

**Preprocessing — SBVA (Haberlandt, Green, Heule, SAT'23):** Structured Bounded Variable Addition.
`SBVA-CaDiCaL` was **overall winner of SAT Competition 2023** (1st overall, 1st SAT, 2nd UNSAT). It
re-encodes formulas by adding auxiliary variables via a connectivity (3-hop) tie-break heuristic,
**emits a DRAT proof of the transformation**, and is a *standalone preprocessor* composable with any
DRAT-producing solver. Open source (github.com/hgarrereyn/SBVA, meelgroup/SBVA).

**Parallel hard-UNSAT — cube-and-conquer (Heule/Kullmann/Biere; Paracooba; AlphaMapleSAT 2024):**
the dominant way to solve hard combinatorial UNSAT in parallel. A **lookahead** solver (`march`)
splits the instance into thousands–millions of **cubes**; **CDCL** conquers each in parallel; proofs
compose. **Paracooba** = distributed/elastic cube-and-conquer for clusters (reuses CaDiCaL parser).
**AlphaMapleSAT** (arXiv 2401.13770) replaces march's shallow lookahead with **MCTS-guided cubing**,
**1.6×–7.6× wall-clock speedup vs march** on Kochen–Specker / Ramsey / Murty–Simon on 128 cores.
march has been the cubing backbone for ~15 years.

**Chromatic-number-of-the-plane SOTA (Hadwiger–Nelson).**
- de Grey 2018: 1581-vertex 5-chromatic unit-distance graph (from a 20 425-vertex seed).
- Heule, "Trimming Graphs Using Clausal Proof Optimization" (arXiv 1907.00929): reduced to **529
  vertices / 2670 edges** (≈100 000 CPU-hours) — **this is exactly our `G₅₂₉`**.
- **Current smallest record: 510 vertices** (Heule; Marktoberdorf summer-school slides, Aug 2025) —
  **we already vendor `data/parts_510.edge`.** Heule's slide deck explicitly lists "ideally formally
  verified" as an open desideratum; validation to date = DRAT-trim + (separately) verified checkers
  in ACL2/Coq/Isabelle, with geometry checked apart from the SAT refutation.
- **No published *end-to-end* machine-checked χ(ℝ²)≥5** (geometry ⊕ SAT ⊕ reduction in one verified
  artifact). That is the open niche.

**Lean-internal verified SAT checking (decisive for our proof track).**
- **LeanSAT merged into Lean 4 core as `Std.Tactic.BVDecide`** (since nightly 2024-08-29): ships a
  **verified LRAT checker** + a verified AIG/CNF path. `bv_decide` bitblasts a Bool/BitVec goal,
  calls an external solver, and **checks the LRAT proof inside Lean by reflection**; `bv_check
  "f.lrat"` checks a cached proof with no solver present. Trust note: it uses **`ofReduceBool`**
  (the compiled checker), i.e. the same *native-compilation* trust family as `native_decide` — **no
  Mathlib required** (it is Std/core).
- **Template precedents:** Heule & Scheucher's **empty hexagon theorem** was established by SAT +
  verified checking, and **Subercaseaux et al. formalised the encoding in Lean and verified the
  geometry→SAT reduction** — structurally identical to our χ≥5 task. **PBLean** (arXiv 2602.08692,
  2026) does end-to-end-verified combinatorial results (Paley-graph independence numbers) via a
  reflection checker + `native_decide`, scaling to **~63 000 proof lines in ~3.5 min** — direct
  evidence our 36 MB LRAT is plausibly checkable in-kernel after trimming.

---

## 1. Honest positioning (no BS)

- **"Beat kissat in general" is not a near-term truth.** kissat/CaDiCaL are 15+ KLOC of decades of
  tuned inprocessing. Claiming general superiority would be drift. **What is defensible and still an
  overreach of the field:**
  1. **Domain win:** be *competitive-to-winning on the CNP / graph-colouring UNSAT family* (our
     `souc_sat` already refutes G₅₂₉ in ~31 s with a checked proof), and
  2. **Verification-integration win (the genuine niche):** a **fully self-hosted solver whose
     refutations are machine-checked end-to-end inside a theorem prover, with no Mathlib** — the
     *first Lean-internal* χ≥5 lower bound. kissat has nothing analogous; this is the edge-of-novelty
     contribution, not a me-too solver.
- Every performance claim below is gated by `drat-trim`/`cake_lpr`/Lean `s VERIFIED`; every benchmark
  must be run, never fabricated (kissat/CaDiCaL were **not installed** in the sandbox at E4c time).

---

## 2. Track A — solver `souc_sat.sio`

Current engine (measured): CDCL + 2-watched-literal + 1-UIP + LRB/VSIDS + LBD deletion + Glucose
LBD-EMA restarts + phase saving + recursive minimisation + chrono-BT + blocking literals + clique
pre-colour symmetry break + **streamed DRAT**; E4c profile on G₅₂₉: `litval` dominates at **1.50 B
calls** (~22/propagation), chrono fires 0×, blocking skips 54 %.

Gap vs SOTA = **inprocessing + cache-layout + parallel cubing**. Prioritised:

| # | Lever | SOTA basis | Expected payoff | Risk / proof coupling |
|---|-------|-----------|-----------------|------------------------|
| **S1** | **SBVA as external DRAT-composable preprocessor** in the pipeline (sbva → souc_sat → stitch DRAT) | SBVA, SAT'23 winner; standalone, DRAT-logged | Large on structured/combinatorial UNSAT incl. colouring; near-zero solver-code risk | Low: SBVA emits its own DRAT prefix; concatenate + drat-trim the whole. Verify end-to-end. |
| **S2** | **E3 inprocessing in-solver: BVE first, then vivification** | BVE "most effective" (CaDiCaL thesis); revisited vivification PoS'25 | The core general UNSAT lever | **High — the real work is DRAT/RAT proof emission for eliminated vars + strengthened clauses.** Multi-week. Soundness gated by drat-trim. Stage BVE (RUP+RAT) before vivification. |
| **S3** | **Arena clause layout + inline `assign[]` in propagate hot loop** | kissat arena / cache design; our own E4c profile | Constant-factor (the 1.5 B `litval`); no search change ⇒ identical proof | Medium-low: pure data-structure refactor; proof unchanged; drat-trim must still pass. |
| **S4** | **Cube-and-conquer on the cluster (384 cores / 3 GPU)** | Heule C&C; Paracooba (cluster); AlphaMapleSAT MCTS-cubing 1.6–7.6× | The scalable path to *open* de Grey fragments + pushing **510** and below | Medium: cubing tool (port a march-style lookahead or call `march_cu`) + per-cube souc_sat + **proof composition** (P2 staged). GPU only helps the MCTS-cubing/lookahead, not CDCL. |
| **S5** | **Honest benchmark vs kissat 4.0.4 / CaDiCaL-SC2025** | PAR-2 protocol (SATComp) | Calibration; publishable table | None except integrity: install both, run on graph-colouring + SATLIB, report PAR-2 + proof-size + check-time. No fabrication. |

**Recommended solver order:** S1 (cheap, big, low-risk) → S3 (constant-factor, proof-safe) →
S5 (get the real numbers) → S2/S4 (the deep multi-week levers). S2's value is real but its honest
cost is the RAT-proof emission, which must not be rushed (soundness).

**Deliberately not claimed:** Kitten-style semantic mining, congruence closure, target-phase
mining — high engineering cost, and their *proof logging* (LRAT-with-antecedents) is the hardest
part of all, per CaDiCaL-SC2025's own authors. Out of scope unless a specific instance demands them.

---

## 3. Track B — the χ(ℝ²)≥5 proof (the real overreach: first Lean-internal end-to-end, no Mathlib)

State: SAT leg = `cake_lpr` (external, machine-code-verified). Geometry leg = Lean `native_decide`
over ℚ(√3,√5,√7,√11). Reduction = Lean core, **zero axioms**. Geometry leg **fused** into the
reduction over the symbolic field-plane `QF×QF` (`SounioDeGreyChi5Concrete.lean`), depends only on
`[propext, native_decide.ax]`. Only `h_sat` (SAT leg) is still an external hypothesis.

| # | Step | SOTA basis | Why it's the niche | Risk |
|---|------|-----------|--------------------|------|
| **B1** | **Internalise the SAT leg in Lean** — use Lean-core `Std.Tactic.BVDecide` **verified LRAT checker** (reflection, `ofReduceBool` — **no Mathlib**) to prove our G₅₂₉ 4-colour CNF UNSAT *inside Lean*; formalise the colouring-CNF encoding and prove `VColourable ↔ CNF-SAT`; discharge `h_sat`. | bv_decide verified checker (Lean core); empty-hexagon (Subercaseaux/Heule) geometry→SAT-in-Lean template; PBLean reflection-checker scaling (63 k lines / 3.5 min) | Collapses to a **single end-to-end theorem with BOTH legs Lean-kernel-checked, no external cake_lpr, no Mathlib** — the first Lean-internal χ≥5 (field-plane). Heule lists this as "ideally" open. | Medium-high but **bounded & known-feasible**: (1) wire/trim 36 MB LRAT through the Std checker (PBLean shows it scales); (2) the encoding-equivalence `VColourable ↔ CNF` is finite propositional Lean (no Mathlib). |
| **B2** | **Finish the QF commutative-ring laws** (qmul assoc/distrib/unit/neg) toward "QF *is* the multiquadratic field" | our `SounioMultiquadRing.lean` groundwork (qadd/qmul comm done) | Makes the field-level statement self-contained; pre-req for any future ℝ lift | Medium: assoc needs 16×16×16 reindex without `ring`/BigOperators — likely more finite `native_decide` on indices or a small Mathlib-free sum lemma. |
| **B3** *(deferred / honest gap)* | ℝ embedding `QF↪ℝ` (b_mask↦√∏primes) lifting field-plane→Euclidean χ(ℝ²)≥5 | standard multiquadratic field theory | The *only* genuinely-missing analytic step | Out of scope without ℝ; needs an **in-tree Mathlib-free real-radical model** (or accepting Mathlib, which adds only the textbook part). Documented as standard, not a novelty gap. |

**Recommended proof order:** **B1 first** — it is the highest-novelty, lowest-conceptual-risk step
and removes the last external dependency (cake_lpr) using infrastructure that already exists in Lean
core. B2 in parallel (independent file). B3 stays deferred with the honest framing already in
`docs/research/multiquad-faithfulness-note.md`.

---

## 4. Sequencing, cluster allocation, win-conditions

1. **Immediate, parallel, low-risk:** S1 (SBVA preprocessor) ‖ S3 (arena/inline) ‖ B1 spike (does
   the Std LRAT checker accept our trimmed LRAT at all?). All three are independent and gated.
2. **Then:** S5 (install kissat 4.0.4 + CaDiCaL-SC2025; real PAR-2 table) — converts "we think we're
   competitive" into evidence.
3. **Deep, staged (multi-week, cluster):** S2 (BVE+vivification with RAT proofs) and S4
   (cube-and-conquer; march-style cubing + souc_sat conquer + proof composition) — aim the cluster at
   **510-vertex** verification and at *open* de Grey fragments. GPUs → MCTS/lookahead cubing only.
4. **Proof summit:** B1 → single Lean-internal end-to-end (field-plane) χ≥5; B2 ring laws; B3 noted.

**Win-conditions (honest):**
- *Solver:* `souc_sat` competitive with kissat/CaDiCaL **on the CNP/graph-colouring UNSAT family**
  (measured PAR-2), with end-to-end checked proofs — **not** a general SATComp claim.
- *Proof:* the **first χ≥5 lower bound checked entirely inside a theorem prover with no Mathlib**
  (field-plane), SAT + geometry + reduction in one artifact. This is the contribution kissat-class
  tools structurally cannot make, and it is the edge-of-novelty target, not drift-to-mean.

**Anti-drift guardrails:** no fabricated benchmarks; every proof gated by drat-trim/cake_lpr/Lean;
no general-superiority claims beyond measured families; RAT-proof-emitting inprocessing (S2) is not
shipped until its proofs check; the ℝ gap (B3) stays labelled standard/deferred, never hand-waved.

---

*Sources: SAT Competition 2025 results/proceedings (satcompetition.github.io/2025; TU Wien
Codel-2025); kissat rel-4.0.4 + CaDiCaL releases (github.com/arminbiere); Heule/Biere revisited
vivification (PoS'25); Froleyks, "Deep integration of SAT solving and model checking" (CaDiCaL 2.0
thesis); Haberlandt/Green/Heule SBVA (LIPIcs SAT'23); Heule/Kullmann/Biere cube-and-conquer;
Heisinger et al. Paracooba; AlphaMapleSAT (arXiv 2401.13770); Heule 1907.00929 (529-graph) +
Marktoberdorf 2025 slides (510-graph); Lean `Std.Tactic.BVDecide` / LeanSAT; Subercaseaux/Heule
empty-hexagon; PBLean (arXiv 2602.08692); cake_lpr (Tan/Heule/Myreen, STTT 2023).*
