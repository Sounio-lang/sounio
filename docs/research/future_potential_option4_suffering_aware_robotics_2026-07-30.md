<!-- docs:meta
topic_id: repo.docs.research.future-potential-option4-suffering-aware-robotics-2026-07-30
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.future-potential-option4-suffering-aware-robotics-2026-07-30
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Future Potential — Option 4: Suffering-Aware Robotics (Mercyful Learning in Clinical Robotics)

**Date:** 2026-07-30
**Author:** evaluation subagent (no git commit; report to parent)
**Status:** `ASSESSMENT` — opinion document, not an executable contract
**Inputs read:** `docs/papers/mercyful_learning_paradigm_2026-07-26.md`,
`docs/research/mercyful_continuous_control_spec_2026-07-26.md` (V-rung),
`docs/research/suffering_aware_architecture_spec_2026-07-28.md` (SAN),
`docs/research/federated_san_spec_2026-07-30.md` (FED-SAN),
`docs/research/mercyful_clinical_integration_spec_2026-07-25.md` (PK-twin suffering fields)

> **Scope and honesty notes.**
> 1. This is an evaluation of a research direction, not new results. No new
>    experiments were run.
> 2. Live literature search was **unavailable** during this evaluation (web
>    search quota exhausted). Claims about the external field rely on
>    established knowledge (safe RL / constrained MDPs, control barrier
>    functions, ISO 10218 / ISO/TS 15066, IEC 60601/62304, ISO 14971,
>    rehabilitation-robotics practice) and are marked as such. Treat
>    field-state claims as ~training-cutoff knowledge, not a 2026 survey.
> 3. "Machine suffering" throughout means the program's standing
>    **operational computational/energetic-burden proxy** (metered FLOPs,
>    control energy, actuator wear). No claim of machine consciousness,
>    sentience, or phenomenology is made or needed.
> 4. This document evaluates **Option 4 only**. The cross-option ranking
>    ("which option has the most real future potential") requires the
>    sibling evaluations of Options 1–3; see §9.

---

## 1. What the direction is

Suffering-Aware Robotics transplants the Mercyful Learning objective —
minimize accumulated suffering subject to a hard anti-Goodhart target
constraint — from training loops and treatment scheduling onto **physical
clinical robots**:

```
J[u] = ∫₀^T [ s_patient(x(t)) + σ·‖u(t)‖² ] dt + μ · sup_t s_patient(x(t))
s.t.  ẋ = f(x,u),  x(0)=x₀,  x(T) ∈ TARGET (safety/efficacy, categorical)
```

where `x` now includes physical state (limb pose, tissue interaction force,
infusion rate), `s_patient` is a patient-discomfort/harm field, `σ‖u‖²`
charges control energy and actuator strain, and `μ·sup` keeps worst-moment
(Rawlsian) peak aversion. The program's distinguishing commitments carry
over: **constraints and gates, not penalties**; necessary vs gratuitous
suffering decomposition; a metered, auditable suffering ledger.

Candidate clinical-robotics hosts, ordered by distance to market:

1. **Rehabilitation / physiotherapy robots** (exoskeletons, end-effector
   gait trainers): pacing assistance to minimize pain/effort while reaching
   a therapy target — the direct physical analog of the validated
   exposure-therapy pacing result (`v*(x) ∝ √(s(x)/κ)`: slow where the
   field is high, fast where low).
2. **Drug-delivery / infusion robotics**: closed-loop titration with
   suffering fields from PK bands — the repo already computes such fields
   from Knightian vancomycin/tacrolimus twins; the missing piece is the
   actuator, not the math.
3. **Assistive care robots** (transfer, mobility, feeding): force/comfort-
   constrained interaction with a hard safety envelope.
4. **Surgical robotics**: suffering-aware trajectory and force control —
   highest impact, highest regulatory barrier, longest timeline.

## 2. Technical feasibility — score 6/10

**What transfers today (proven inside the program):**

- The objective functional, the peak-aversion term, the anti-Goodhart hard
  constraint, and the necessary/gratuitous decomposition are
  **formulated and executable** (V-rung, K-rung, SAN A1–A8). Nothing about
  them is specific to training loops; the V-rung's continuous-control
  formulation is already a robot-shaped problem (state trajectory, control
  energy, terminal constraint).
- The "mercyful pacing profile" (slow where the field is high) is a
  closed-form, real-time-computable control law — trivially within the
  compute budget of any clinical robot controller.
- The machine channel maps *better* onto robots than onto neural nets:
  control energy, torque ripple, and actuator temperature are directly
  metered physical quantities, not proxies. This is the one setting where
  the machine-suffering ledger becomes unambiguous.

**What does not transfer (the real gaps):**

- **The patient suffering field has no sensor.** In every validated rung,
  `s_patient` was synthetic or a declared normative structure (e.g., the
  5:1 harm weighting). On a robot, the field must come from physiology
  (HR/HRV, galvanic response, EMG, facial/vocal analysis) or patient
  report. Pain is subjective, delayed, individually calibrated, and
  non-stationary; there is no ground-truth discomfort sensor at any TRL.
  This is the single largest feasibility gap — the math is ready for an
  input that does not yet exist reliably.
- **Real-time certified control.** Hard-constraint enforcement on a
  physical safety-critical system needs formal guarantees (control barrier
  functions, reachability) under IEC 62304 / ISO 14971 development
  processes — a different engineering discipline from Python contracts.
- **The field is crowded.** Safe RL, constrained MDPs, CBF-based safety
  filters, and power-and-force-limited collaborative robotics (ISO/TS
  15066) already occupy "minimize harm subject to getting the job done."
  The Mercyful framing must demonstrate it adds something — the ledger,
  the peak term, the necessary/gratuitous split — beyond a re-labeled
  constrained MDP. This is a differentiation burden, not an impossibility.

**Verdict:** buildable as a research prototype on a rehab or infusion
platform within current technology; buildable as a certified clinical
device only with a multi-year regulatory program. The bottleneck is
sensing and certification, not algorithms.

## 3. Timeline — score 5/10

| Milestone | Estimate | Basis |
|---|---|---|
| Sim-to-bench prototype (rehab pacing or infusion titration on a lab robot, synthetic/physio-sim suffering field) | 1–2 years | V-rung math directly reusable; standard ROS/hardware work |
| Human-factors pilot (healthy volunteers, discomfort-proxy field, no therapy claim) | 2–4 years | IRB + usability studies; no device clearance needed for non-therapeutic pacing research |
| Clinical investigation (patients, therapeutic claim) | 5–8 years | IDE/early-feasibility study; suffering field validation against patient-reported outcomes |
| Cleared/CE-marked suffering-aware feature on a commercial platform | 7–12 years | 510(k)/De Novo or MDR pathway; predicate devices exist for force-limited interaction but not for "suffering-minimizing" control — likely De Novo, which is slower |
| Surgical-robotics variant | 10–20+ years | Autonomy in surgery is itself unsettled; adding a novel objective restarts the evidence clock |

Timeline risk is dominated by the **regulatory and evidence path**, which
is measured in years per stage and is largely independent of algorithm
quality.

## 4. Impact — score 6/10

If realized:

- **Rehabilitation robotics** (plausible, near-term): patient-specific,
  pain-pacing therapy that provably reaches the therapy target — addresses
  the real clinical problem of adherence and over/under-exertion. Impact:
  **high within the niche**, affecting a large and growing rehab market
  (aging populations, stroke care), but it improves existing devices rather
  than enabling a new capability.
- **Infusion/titration robotics**: a principled upgrade over today's
  threshold-based closed loops (e.g., insulin pumps, sedation systems
  already run closed-loop control). Impact: medium-high, incremental.
- **Surgical robotics**: transformative *if* it ever lands, but that
  conditional carries most of the timeline risk.
- **Conceptual contribution**: the two-channel ledger (patient discomfort
  + machine burden) and the necessary/gratuitous decomposition are a
  genuinely unusual frame in robotics, where energy/wear and patient harm
  are optimized in separate literatures. A unified, auditable suffering
  ledger for a robot's whole operating history would be a real
  contribution to safety cases and post-market surveillance.

Net: **high impact in specific niches, not broadly transformative.** The
honest ceiling is "a better objective and audit trail for a class of
clinical robots," not a new robot category.

## 5. Risks — score 6/10 (lower is better)

Ranked by severity:

1. **Proxy-Goodhart on the suffering field itself (high).** The paradigm's
   anti-Goodhart machinery guards the *target*; it does nothing for the
   *input*. A robot minimizing a physiological discomfort proxy can learn
   to suppress the proxy (e.g., damp the motion that perturbs the sensor)
   while harming the patient. The repo's own ethics spec treats harm
   weightings as declared normative inputs — defensible in simulation, a
   safety hazard when actuated. Mitigation exists (hard force/pressure
   envelopes independent of the learned field) but must be designed in
   from the start.
2. **Regulatory novelty (high).** "Suffering-minimizing control" has no
   predicate device. Expect a De Novo-class pathway, a demanded clinical
   evidence base for the objective function itself (not just the
   hardware), and reviewer skepticism toward the word "suffering" in a
   indications-for-use statement. The machine-suffering channel, however
   carefully scoped, invites misreading as a machine-welfare claim —
   rhetorically costly in a regulatory or IRB context.
3. **Pain measurement validity (high).** No validated, individual-calibrated,
   real-time discomfort estimator exists. The whole direction rests on
   this input; a 5-year program could still end with a field too noisy to
   certify.
4. **Differentiation failure (medium).** Reviewers may read the
   contribution as constrained optimal control with new vocabulary. The
   program's own honesty note (V-rung §1.1: "the upgrade is standard
   optimal control") applies double here.
5. **Sim-to-real and liability (medium).** Standard for clinical robotics
   but non-negotiable; suffering-aware control adds a new failure mode
   (miscalibrated mercy) to the hazard analysis.
6. **Program bandwidth (medium, internal).** The Mercyful line is deep in
   compiler/SAN/federated work; robotics requires hardware, a robotics
   lab, and clinical partners the repo does not currently show.

## 6. Is this a real research direction?

**Yes, with a qualification.** The underlying problem — discomfort- and
effort-aware assistance in rehabilitation and assistive robotics under
hard safety constraints — is a real, active research area with immediate
applications (rehab adherence, closed-loop drug delivery, collaborative
medical robots). It is not speculative in the way that, e.g., machine-
welfare-motivated objectives would be; the patient channel alone justifies
the work.

The qualification: the **Mercyful-specific delta** (two-channel ledger,
necessary/gratuitous split, architectural anti-Goodhart gate) is real but
modest, and the program's most distinctive term — the machine-suffering
channel — is the least clinically necessary and the most rhetorically
expensive. The direction is strongest when pitched as *constrained,
peak-averse optimal control with an auditable harm ledger*, and weakest
when pitched as an extension of the machine-suffering ethics into hardware.

**Immediate applications exist**: rehab pacing and infusion titration are
direct ports of already-validated rungs (exposure therapy V-rung; PK-twin
suffering fields). That is the strongest feasibility signal in this
evaluation — the first two years of work are largely re-expression of
proven math, not new science.

## 7. Scores

| Dimension | Score (1–10) | One-line justification |
|---|---|---|
| Technical feasibility | **6** | Math ready and ported in simulation; pain-field sensing and certified real-time control unproven |
| Timeline | **5** | Prototype 1–2 y; any clinical reality 5–12 y, surgery 10–20+ y |
| Impact | **6** | High in rehab/infusion niches; not a new capability class |
| Risk (lower = better) | **6** | Proxy-Goodhart, regulatory novelty, pain-measurement validity |
| **Overall future potential** | **5.5** | Real direction, real niches, heavy non-algorithmic drag |

## 8. Recommendation: **explore**

Not "pursue": the program should not open a hardware line now — it lacks
the robotics lab, clinical partner, and regulatory capacity, and the
decisive blocker (a validated patient-suffering field) is a sensing
problem that hardware spending does not solve.

Not "defer": the near-term work is cheap and mostly already done. The
highest-value next steps are simulation- and partnership-shaped:

1. **Paper/benchmark rung (0–12 months):** re-express the exposure-therapy
   V-rung as a rehab-robot pacing benchmark and the PK-twin rung as a
   closed-loop infusion benchmark, in an open physics simulator, against
   safe-RL/constrained-MDP baselines. This settles the differentiation
   question (risk #4) at near-zero cost.
2. **Suffering-field-from-physiology study (0–18 months):** the actual
   linchpin. Partner for an IRB-approved healthy-volunteer study mapping
   physiological signals to patient-reported discomfort; publish the
   field-estimation error bounds. Everything downstream is gated on this.
3. **Drop the machine channel from the clinical pitch.** Keep it in the
   math where it is honest and metered; keep it out of the IRB/FDA-facing
   framing.
4. **Re-evaluate at the benchmark gate:** if the Mercyful objective beats
   constrained-MDP baselines on the ledger metrics the program cares about
   (peak harm, gratuitous suffering, auditability), upgrade to "pursue"
   via a partnership with an existing rehab-robotics group rather than
   in-house hardware.

## 9. Cross-option ranking (as instructed)

The task asks which option has the most real future potential. This
document evaluated **Option 4 only**; Options 1–3 were assigned to sibling
evaluations whose outputs were not available to this agent (and no
`future_potential_*` documents existed in the repo at evaluation time). A
defensible cross-option ranking cannot be produced from one quadrant.
Within what this evaluation can support: Option 4 scores **5.5/10
overall, recommendation: explore** — real but niche-bound, with its
timeline dominated by regulation and pain-sensing rather than by the
Mercyful math, which is the part already proven.

## 10. Assumptions and leftovers

- Assumed "Option 4" = suffering-aware robotics applied to *clinical*
  robots (rehab, infusion, assistive, surgical), per the task statement.
- Assumed the filename slug `option4_suffering_aware_robotics` for
  `future_potential_{option}_2026-07-30.md`.
- No web literature search was possible (quota exhausted); field-state
  claims are from established knowledge and flagged in the scope note.
- **Not done (left for parent):** registration of this document's
  `topic_id` in `docs/governance/topic-registry.v1.json` — skipped to
  avoid parallel-edit collisions with sibling option evaluations.
- **Not done (by policy):** git commit — per instructions, all results
  are reported to the parent agent uncommitted.
- LLM-offload review (per `.claude/AGENT_OFFLOAD_POLICY.md`) was **not**
  run: this is an internal assessment document, not a math claim,
  clinical-pathway code change, or external-facing artifact.
