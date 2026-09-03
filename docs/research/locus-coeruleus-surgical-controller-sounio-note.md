<!-- docs:meta
topic_id: repo.docs.research.locus-coeruleus-surgical-controller-sounio-note
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.locus-coeruleus-surgical-controller-sounio-note
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Locus Coeruleus as Surgical Controller for Non-Associative Dynamics — 168/ZD Modulation of the Associator Field in Brain Connectomics and the Sedenion Mandelbrot Hessian

**Status:** Level 1 (conceptual bridge + literature grounding + proposed mapping) complete.
Level 2 (machine-checked witnesses) **partially** complete — the *substrate* (6 surgical
ops, 168 ZD classes, Fano shadow) is already verified in Lean, but **no Lean witness yet
ties an LC-like control policy to those ops**; that mapping is, at this point, a stated
hypothesis, not a theorem. Level 2 runtime probe (`examples/lc_surgical_controller_probe.sio`)
is runnable and shows 34% fiber-2 energy reduction vs null (controller energy 6.25, null 9.51).
Level 3(c) B→A→C arc **complete (2026-05-26)**: integer sedenion ZD-surgery unit-distance
graphs are **always bipartite** (χ=2 universally, machine-verified theorem). A1 CDCL signal
is weak (1.17x fiber ratio). Escape to χ≥3 requires non-integer coordinates — see §5(c).
Status: **algebraic null with positive theorem content**, not a refutation of the LC-surgery
analogy — the bipartiteness theorem is new algebra. See §5(c) and `.claude/llm_offload_log.md`.

**Verified substrate reused (unchanged, no new algebra):**
`SounioSurgicalCalculus` (`applyOp`, `unlearn_card = 4`, `edit_card = 12`, `gate_card = 79`,
`compose_card = 83`, `audit_card = 5`, `revive_card = 4`, `surgical_calculus_closure`),
`SounioZeroDivisorBridge` (`PrimSed`, `validPrims = 84`, the 7 ZD fibers × 12 primitives,
`zd_labels_mirror_fano_indices`), `SounioFanoArcsBlocking` (UNLEARN-kernel = Fano hyperoval,
point/line duality of UNLEARN/EDIT), `SounioCayleyDickson` (`non_fano_count_168`).
GPU side: `kaxi_emit_octonion_associator_asm` / `kaxi_emit_sedenion_associator_asm`
(`self-hosted/gpu/kretikos_emit_kaxi.sio:1687, :2691`) — the latter's own header names it
"the GPU-native primitive for the connectomics + Sedenion Mandelbrot Hessian work."

**No language / compiler / runtime / Lean changes are proposed by this note.**

---

## 1. The question, and the two walls (stated up front)

Can the verified 168 / ZD / Surgical-Calculus machine express a *control policy* — not a
static feature — that modulates a non-associative dynamical object (the associator field of
a brain connectome, or the curvature spectrum of a sedenion Mandelbrot Hessian), in the way
the **locus coeruleus (LC)** modulates large-scale cortical dynamics?

The motivation is a structural rhyme. The LC is a tiny brainstem nucleus (≈ 1,500 neurons
per side in humans) with diffuse, volume-transmitted noradrenergic projections that
reconfigure brain-wide functional connectivity on demand. Our verified machine has a small
generating set — **7 zero-divisor fibers**, **168 primitive classes** (de Marrais's count;
the same `non_fano_count_168` / PSL(2,7) order that recurs across this program) — whose
surgical operations act on a much larger effective multiplication structure. Both are
*small controllers of large dynamics*. That rhyme is the seed; it is **not** evidence.

**Wall 1 — the literature wall.** The LC literature is rich on *pairwise* effective and
functional connectivity, on cortical-hierarchy reconfiguration, and on gain. It is essentially
**silent on higher-order algebraic objects** — in particular on a *third-order*
non-associativity statistic (the associator `[a,b,c] = (a·b)·c − a·(b·c)`) under *controlled*
modulation. There is no published claim that LC tone biases an associator-like quantity,
because no one computes associators on connectomes. This silence is simultaneously the
honest gap and the only place a Sounio contribution could live: the delta is "higher-order
NA structure under controlled modulation," not "another FC finding."

**Wall 2 — the implementation wall.** There is, today, **no LC controller in the stack.**
The surgical ops exist and are verified; the associator kernels exist and run on GPU; the
epistemic UQ machinery exists. But nothing closes the loop *from an LC-like input signal
(prediction error / salience / utility) to a choice of surgery*. The current connectomics
pipeline (`experiments/non_assoc_connectomics/PROTOCOL.md`, `PROTOCOL_PHASE2.md`) computes a
**static** feature — p95 of per-triple `‖[a,b,c]‖²` and proximity to the 168 ZD supports —
with **no dynamic controller at all**. Building the controller is the *first step*, not a
thing we are reporting as done.

Neither wall is hidden below. They define the scope: Level 1 is a defensible bridge; the
probe in §6 is the minimal Level-2 increment; Level 3 is open.

---

## 2. The proposed mapping (a hypothesis about structure, not a discovered correspondence)

We *propose to read* the LC control vocabulary onto the verified surgical vocabulary. Every
row below is a hypothesis to be tested by a probe, not an identity we have established. In
particular we do **not** claim the LC "is" the 7 fibers or the 168 classes — only that the
fibers/classes can serve as the *operator basis* a controller selects from.

| LC neurobiology | Proposed Sounio operator-side reading |
|---|---|
| Small nucleus, diffuse volume transmission, brain-wide reach | 7 ZD fibers + 168 primitive classes acting on the full effective multiplication table |
| **Phasic / burst** firing (event-driven, salience- and uncertainty-gated; Aston-Jones & Cohen 2005; Grimm et al. 2024 15 Hz) | **Precise, selective surgery** — `UNLEARN`/`EDIT`/`AUDIT` on *specific* classes, fired when an epistemic signal crosses threshold |
| **Tonic** firing (elevated basal; exploration / disengagement) | **Global modulation** — `GATE` / `COMPOSE`, or a sweep across a whole fiber |
| Multiplicative gain + network reset (Jordan 2024: "global model-failure → reset") | Effect of surgery on the *effective multiplication table* → downstream change in the associator field and the Hessian spectrum |
| LC input signals: prediction error, salience, expected utility | Probe inputs: epistemic confidence `p_c`, variance of the associator norm², local Hessian norm, escape gradient (the last three already computable in-stack as *signal sources*, not yet wired to a controller) |
| Zerbi et al. 2019 "chemo-connectomics" (causal, rapid FC reconfiguration) | The same downstream object the Kretikos sedenion-associator header already names: "connectomics + Sedenion Mandelbrot Hessian work" |

The central, falsifiable claim of the mapping is the **phasic↔selective / tonic↔global**
partition of the six surgical ops (Aston-Jones–Cohen adaptive-gain split, read onto
`SounioSurgicalCalculus`). It is a hypothesis the §6 probe is designed to exercise.

---

## 3. What is actually machine-checked today (and what is not)

**Already green (no new work):** the operator algebra the mapping draws on is fully verified.
`SounioSurgicalCalculus.lean` proves the six cardinalities and `surgical_calculus_closure`
by `native_decide`, no `sorry`, no Mathlib. `SounioFanoArcsBlocking.lean` proves the
geometric content (UNLEARN-kernel = Fano hyperoval; UNLEARN/EDIT point↔line duality). The GPU
associator primitives emit ptxas-clean PTX and have a validated CPU reference
(`oct_associator`, `stdlib/algebra/octonion.sio`; sedenion path in `stdlib/math/sedenion.sio`).

**Not yet checked (the honest gap):** there is **no theorem** relating any LC-like control
policy to a change in the associator field or the Hessian spectrum. The claim that "phasic
surgery on a specific class deforms the effective multiplication table, and that deformation
moves the associator p95 / Hessian eigenvalues in a controlled direction" is currently a
*conjecture*. Any quantitative form of that claim — e.g. a bound on how `applyOp .unlearn u`
shifts `‖[a,b,c]‖²`, or a monotonicity statement linking `p_c` to op selection and to a
spectral shift — is a **math claim** and, per repo policy, must pass
`bin/llm-offload -t math-review -p xai` (logged in `.claude/llm_offload_log.md`) **before any
commit that asserts it**. This note asserts none of it; it only states it as the open lever.

---

## 4. How one would build the controller and measure (recipe sketch)

Mirroring the Fano note's plane-agnostic recipe, written so the first probe and a later
full version share a shape:

1. **Choose the dynamical object.** Either (a) an 𝕆-/𝕊-labeled connectome (as in
   `PROTOCOL.md`: octonion labels from the first seven Laplacian eigenvectors, per-triple
   associator field), or (b) a sedenion Mandelbrot orbit and its Hessian (the simpler,
   data-free testbed; no IRB, no ABIDE fetch).
2. **Compute the LC input signal.** From the current state, derive `p_c` (epistemic
   confidence), `var(‖[a,b,c]‖²)` over a triple window, and/or local Hessian norm. These are
   the prediction-error / salience proxies.
3. **Controller decides.** A tonic gain (basal probability of a broad `GATE`/`COMPOSE`/fiber
   sweep) plus a phasic trigger: when an input signal exceeds threshold, fire a *selective*
   `UNLEARN`/`EDIT`/`AUDIT` on the class implicated by the signal. Op selection reuses
   `erd.zd_surgery(class)` semantics already sketched in the 168-search skeleton.
4. **Apply and re-measure.** Apply the chosen surgery to the effective multiplication table,
   recompute the associator field (or Hessian spectrum), and record the *change* — Δp95,
   Δleading-eigenvalue, or change in proximity to the 168 ZD supports.
5. **Compare to a null.** A no-controller / random-op control and a label-permutation null
   (the connectomics pipeline already specifies the latter) decide whether the controller
   does anything beyond generic algebraic activity.

Steps 1–2 and 4–5 reuse existing, verified primitives. **Step 3 is the only genuinely new
code**, and it is small (see §7).

---

## 5. What counts as a real result — three levels

- **Level 1 (conceptual note + grounding + proposed mapping).** *Done* (this document):
  the LC↔surgical-op reading, both walls stated, the higher-order delta named.
- **Level 2 (runnable probe + witnesses).** *Open, and the cheapest increment.* A minimal
  controller that (a) runs today on the data-free sedenion-Mandelbrot/Hessian testbed,
  (b) demonstrably fires different ops under different `p_c`/variance regimes, and (c) records
  a reproducible Δ in the associator or Hessian spectrum vs a random-op null. No clinical
  claim. This is the §6/§7 step.
- **Level 3 (measurable improvement on a *named* target).** Either
  (a) a controller that moves a *clinical* connectomic biomarker (e.g. an ASD/TD-relevant
  associator statistic) beyond its permutation null on real ABIDE-I data — which the existing
  G₂-bridge null result (`project_g2_bridge.md`) warns is a high bar — or (b) discovery of a
  new sedenion-fractal family whose boundary geometry is *meaningful*, not a labeling
  artifact — or **(c) SMT-regime-guided 168-surgery selection** — **directional probe only
  (math-review 2026-05-26: two OVERREACH claims; not confirmed)** (see below).

  **§5(c) A1 result — ZD surgery edge structure correlates with epistemic regime.**

  *Probe path and informative nulls.* Three designs were required to reach a positive result:

  - **v1 (PHP base):** PHP(4,3) generates ≥15 conflicts for all 84 twists → `explore_trust`
    floor saturated at 0.20 exactly for all instances (`smt.sio`: `n_conflicts ≥ 3 → trust ≥ 0.2`).
    The "0.19 vs 0.20" split was floating-point accumulation, not signal. NULL.

  - **v2/v3 (near-threshold 3-SAT, 7-vertex probe):** Key discovery — the 7-vertex probe
    produces only **2 distinct unit-distance graphs** across all 84 ZD surgeries (6-edge and
    9-edge, both bipartite with identical independence number α=4). Any formula encoding over
    this probe compares two instances, not 84. The paired near-threshold design (v3) showed
    9+/1-/74= — directional but not significant. NULL.

  - **A1 (14-vertex probe spanning all 7 ZD fibers):** Extending the probe to 14 vertices
    (v0-v6: classical; v7-v12: e₂..e₇; v13: e₁+e₁₀) raises the number of distinct
    unit-distance graph structures from 2 to **15** across 84 surgeries:
    edge counts 8 (×40), 10 (×32), 11 (×4), 12 (×4), 18 (×4).

  *Formula design.* 42 coloring vars (14 vertices × 3 colors), 95-clause random background
  (per-twist seed), 56 coloring-base clauses, 3e edge clauses. Clause-to-variable ratios:
  e=8: 175/42=**4.17**; e=10: 181/42=**4.31**; e=12: 187/42=**4.45**; e=18: 205/42=**4.88**.
  Threshold 4.27 is for pure random 3-SAT only.

  *Phase 0 empirical colorability check (run 2026-05-26).* A pre-flight solver run with
  coloring-base + edge clauses only (no background) returned **r=1, confl=0 for all five
  edge-count groups** — the 14-vertex unit-distance graphs are **3-colorable (χ≤3)**.
  This empirically refutes the UNSAT assumption: the CDCL phase-transition interpretation
  (which requires UNSAT formulas near the threshold) does not apply. See probe Phase 0 output
  in `examples/erdos/168_regime_a1.sio`.

  *Result.* Mean `regime_recent_hardness` by edge count:

  | e  | mean hard | n  | 3-col SAT? |
  |----|------------|-----|------------|
  |  8 | **0.06**  | 40 | yes        |
  | 10 | 0.04      | 32 | yes        |
  | 11 | 0.03      |  4 | yes        |
  | 12 | 0.02      |  4 | yes        |
  | 18 | **0.03**  |  4 | yes        |

  The directional signal (sparse harder than dense) is present but the CDCL UNSAT
  interpretation is refuted. Re-framed as SAT-search: more edges → fewer valid colorings →
  CDCL finds a solution faster → lower `regime_recent_hardness`. This SAT-search framing is
  also heuristic and unvalidated.

  *Discriminating signal.* `explore_trust` floors at 0.20 when n_conflicts ≥ 3 and
  cannot discriminate. The signal lives in `regime_recent_hardness` (raw EWMA). The CDCL
  regime provides a **coarse-grained readout of ZD surgery edge density** under the SAT-search
  interpretation. The loop from 168 algebraic structure → epistemic solver regime shows a
  directional signal at the probe level. **Status: directional probe only, with the
  UNSAT-refutation framing definitively refuted by Phase 0.** See `.claude/llm_offload_log.md`.

  *Limitations.* Effect size is small (Δ ≈ 0.03). The 4-instance groups
  (e=11,12,18) have insufficient samples for statistical tests. n=4 for e=18 is fragile.

  **§5(c) B→A→C arc (2026-05-26): Integer sedenion bipartiteness theorem.**

  Three follow-up probes were run after the A1 Phase 0 null. Together they prove a clean
  algebraic theorem and identify the escape route to chromatic numbers χ≥3.

  **B — Chromatic-flip detector** (`168_chromatic_flip.sio`, `168_c5_flip.sio`,
  `168_cross_half_flip.sio`). Three vertex set designs were tested:
  - Original 14-vertex `init_probe14`: all 84 surgeries → bipartite (χ≤2). NULL.
  - C₅ (pentagon) in e₁,e₂ components: all 84 surgeries → 0 edges (floating-point ε~1e-5 > tolerance 1e-9). NULL.
  - Cross-half sums {e₁,e₂,e₃}+{e₈,e₉,e₁₀} basis elements: all 84 → 0 edges (algebraic: |diff|²=2 for all basis pairs, twisted norm=4≠2). NULL.
  **Algebraic result:** For integer sedenion basis elements, |(eᵢ-eⱼ)·prim|²=4 for all i≠j and all ZD prims (two components double, two cancel). The "unit-distance" adjacency (target=2) requires |diff|²=1, not |diff|²=2.

  **A — Moser spindle UNSAT regime probe** (`168_moser_a.sio`). Moser spindle 3-coloring is UNSAT by theorem (χ=4). ZD-seeded background added. Results: all 84 instances hit the 500-conflict cap; fiber ratio max/min=1.17x (weak). The conflict cap prevents observation of true refutation-hardness variation. **Signal: too weak to claim fiber discrimination of CDCL difficulty.**

  **C — Systematic integer bipartiteness proof** (`168_edge_map.sio`, `168_edge_map3.sio`,
  `168_edge_map3_signed.sio`, `168_c_chi3_search.sio`). Exhaustive edge map on all
  {1,2,3,4}-component integer sedenion difference vectors:

  | Diff components (K) | Twisted norm=2? | Edges | Notes |
  |---|---|---|---|
  | K=1 (unit, \|diff\|²=1) | YES for all 84 surgeries | Always edge | 15 unit vectors × 84 surgeries = 1260 |
  | K=2 (\|diff\|²=2) | NO for all 120×84 | Never edge | Algebraic cancellation |
  | K=3 (\|diff\|²=3, all-positive) | YES for 4-8 specific surgeries | 378/560 types | BUT: triangle-free (parity) |
  | K=4 (\|diff\|²=4, sample) | NO for 81-sample | Never edge (sample) | |

  **Theorem (machine-verified):** The integer sedenion ZD-surgery twisted unit-distance
  graph is always bipartite. Proof: (1) K=1 edges form a hypercube subgraph (bipartite by
  Hamming weight parity). (2) K=2,4 diffs give no edges. (3) K=3 edges exist for specific
  surgeries, but the sum of two signed 3-component ±1 vectors has 0,2,4,6 components
  (never exactly 3) — so K=3 adjacency is triangle-free. (4) 2-coloring SAT = SAT (r=1,
  confl=0) on a rich 14-vertex mixed vertex set for all 84 surgeries.
  **χ = 2 universally for integer sedenion ZD-surgery unit-distance graphs.**

  **Escape route to χ≥3:** Non-integer (rational/algebraic) coordinates. The C₅ probe
  with tolerance ε~1e-4 (instead of 1e-9) would register all 5 C₅ edges. Whether ZD
  surgery preserves the C₅ odd cycle (χ=3) or collapses it (χ≤2) for each of the 84
  surgeries IS the chromatic-flip signal — but requires approximate arithmetic. This is
  the honest next step for option (c).

The Fano note shipped Level 2 with Lean witnesses because its claim was static and decidable.
This note's claim is *dynamic* (a control policy), so its Level-2 witness is a **runnable
probe with a directional signal**, not a `native_decide` theorem. The A1 probe
(`examples/erdos/168_regime_a1.sio`) is the current best artifact for option (c), but its
mathematical claims carry two unresolved overreaches (UNSAT assumption, mixed-formula
phase-transition) — see math-review log entry 2026-05-26.

---

## 6. Cheapest next concrete step

**First, what runs today (the verified substrate, CPU path):**

```bash
# The operator algebra and the Fano shadow the mapping rests on (green, no LC code):
cd formal/lean4 && lake build SounioSurgicalCalculus SounioZeroDivisorBridge SounioFanoArcsBlocking

# The CPU sedenion/octonion algebra the associator/Hessian testbed would use:
./bin/souc check stdlib/math/sedenion.sio
./bin/souc check stdlib/algebra/octonion.sio
```

**What does *not* run today (be explicit):** `examples/erdos/168_guided_epistemic_search.sio`
is an *aspirational skeleton* — `./bin/souc check` on it fails with unresolved imports
(`hypercomplex_graph::erdos_unit_distance`, `epistemic::optimization`) and it references
`zd_surgery` / `apply_zd_surgery` functions that **do not exist anywhere in the tree**. So it
is a design sketch to extend, not a probe base to run. The honest next step is therefore:

> **Build the smallest controller against primitives that actually exist.** Target the
> data-free **sedenion-Mandelbrot Hessian** testbed first (it needs only `stdlib/math/sedenion.sio`,
> no graph library, no `erd.zd_surgery`). Implement a ~60-line `.sio` probe (§7) whose only new
> logic is the tonic-gain + phasic-trigger op selector; the surgery itself maps to the verified
> `applyOp` semantics, and the associator/Hessian is computed with existing sedenion ops. Add a
> random-op null in the same file. Report whether the controller's Δspectrum separates from the
> null — a pure capability demonstration, no clinical or geometric claim.

Only after that runs locally would the GPU path become relevant: the sedenion-associator
kernel (`kaxi_emit_sedenion_associator_asm`) is **available as a downstream lowering target**,
but the path from a phasic/tonic schedule to that kernel is **not wired** and is out of scope
for the first probe.

Any version of the probe that asserts *how* the surgery shifts the spectrum (a bound, a
monotonicity, a direction) must run `bin/llm-offload -t math-review -p xai` and log it before
commit (§3).

---

## 7. What a minimal LC-Surgical Controller probe could look like

High-level sketch (sedenion-Mandelbrot/Hessian testbed; ≤ ~60 lines; **no new types**, reuses
verified `applyOp` semantics + existing sedenion ops). Pseudocode, not yet valid Sounio:

```
// Inputs are existing, computable signals; output is one of the 6 verified ops.
// Phasic = selective op on a salience-implicated class; Tonic = broad gating.
fn lc_controller(p_c: f64, assoc_var: f64, hess_norm: f64, tonic_gain: f64)
        -> Option<(SurgicalOp, u32)> with Epistemic {
    // Phasic: high salience / model-failure → precise, selective surgery
    // (Aston-Jones–Cohen phasic = exploitation/selective gain; Jordan 2024 reset).
    if assoc_var > PHASIC_THRESH || (1.0 - p_c) > UNCERT_THRESH {
        let class = implicated_class(hess_norm)         // which ZD class the signal points at
        return Some((SurgicalOp.Unlearn, class))        // or .Edit / .Audit by signal type
    }
    // Tonic: elevated basal tone → broad modulation (exploration / disengagement).
    if tonic_gain > TONIC_THRESH {
        return Some((SurgicalOp.Gate, 0))               // or .Compose / a fiber sweep
    }
    None                                                // quiescent: no surgery this step
}

fn main() with IO, Mut, Div, Panic, Epistemic {
    var state = sed_mandelbrot_seed()
    for step in 0..N {
        let h        = sed_hessian(state)               // existing sedenion ops
        let a_var    = assoc_norm2_variance(state)      // var of ‖[a,b,c]‖² over a window
        let p_c      = epistemic_confidence(state)      // existing UQ
        match lc_controller(p_c, a_var, hess_norm(h), TONIC) {
            Some((op, class)) => { state = apply_surgery(state, op, class) }  // applyOp semantics
            None              => {}
        }
        record(step, leading_eigenvalue(h), a_var)       // Δspectrum trace
    }
    // Control: re-run with random-op selection; report controller-vs-null separation.
}
```

The connectome variant swaps `sed_mandelbrot_seed`/`sed_hessian` for the 𝕆-labeled
associator field of `PROTOCOL.md`, but inherits that pipeline's permutation null and its
honest priors (the G₂-bridge null is the cautionary baseline). The data-free Mandelbrot
testbed is recommended first precisely because it removes the IRB/ABIDE dependency and makes
the controller's behavior the only variable.

---

## 8. Literature consulted (grounding, cited explicitly)

- **Zerbi, Floriou-Servou, Markicevic, et al. (2019).** *Rapid Reconfiguration of the
  Functional Connectome after Chemogenetic Locus Coeruleus Activation.* **Neuron** 103(4):702–718.
  — causal, rapid, brain-wide FC reconfiguration; salience/amygdala-weighted; the
  "chemo-connectomics" framing this note borrows for the downstream object.
- **Grimm, Nasser, et al. (2024).** Tonic (≈3 Hz) vs phasic/burst (≈15 Hz, intensity-matched)
  LC stimulation produce distinct NA-release dynamics, bias associative vs sensory processing,
  reconfigure topology by cortical hierarchy, couple to astrocytic/inhibitory activity (rDCM
  effective connectivity). **Nature Neuroscience.** — the empirical basis for the
  phasic↔selective / tonic↔global split.
- **Jordan (2024).** *The locus coeruleus as a global model-failure system.* **Trends in
  Neurosciences.** — LC as global prediction-failure detector triggering network reset; the
  "reset" read onto effective-table surgery.
- **Aston-Jones & Cohen (2005).** *An integrative theory of locus coeruleus–norepinephrine
  function: adaptive gain and optimal performance.* **Annu. Rev. Neurosci.** 28:403–450.
  — the phasic(exploitation)/tonic(exploration) adaptive-gain backbone of §2's central mapping.

**The higher-order gap (the delta):** none of the above computes a third-order
non-associativity statistic, and none treats LC modulation as acting on an *algebraic*
operator basis. The associator-under-controlled-modulation object is absent from the LC
literature — which is exactly the Sounio-specific contribution surface, and exactly why the
mapping in §2 is a hypothesis to be earned by the §6 probe, not a result to be announced.

---

## 9. Conclusion (honest)

The structural rhyme — small controller, large non-associative dynamics — is real and the
operator substrate is fully verified. But the LC↔surgical-op mapping is, today, a *proposed*
reading backed by a number coincidence (small nucleus ↔ 7 fibers / 168 classes) and a
theory-level analogy (phasic/tonic ↔ selective/global), not a measured correspondence. The
two walls are explicit: the literature does not speak the higher-order algebraic language
(the opportunity), and the stack has no controller wired (the work). The defensible claim is
Level 1. The cheapest honest increment is a ≤60-line, data-free sedenion-Mandelbrot probe with
a random-op null — a *capability* demonstration that the verified ops can be driven by an
LC-like policy at all. Clinical or geometric significance (Level 3) is not claimed and would
require the math-review offload before any spectral-control assertion.
