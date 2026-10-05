<!-- docs:meta
topic_id: repo.docs.research.future-potential-suffering-aware-quantum-2026-07-30
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.future-potential-suffering-aware-quantum-2026-07-30
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Future Potential Evaluation — Option 3: Suffering-Aware Quantum

**Date:** 2026-07-30
**Direction:** Mercyful Learning (suffering-minimization training paradigm,
`docs/papers/mercyful_learning_paradigm_2026-07-26.md`) applied to / realized on
quantum computing.
**Status:** Evaluation only. No implementation proposed or started.
**Verdict up front:** **Defer.** This is a real but early and unproven research
intersection with no immediate application, a 10–20 year horizon for the
meaningful version, and significant hype/credibility risk to the core paradigm.

---

## 1. What the direction would actually be

"Suffering-Aware Quantum" decomposes into three distinguishable claims, and the
evaluation differs sharply across them:

- **(A) Quantum hardware as a mercyful workload target.** Run mercyful learning
  (suffering-weighted objectives, distributional cost instead of expectation,
  machine-burden channel) on quantum machine-learning (QML) substrates:
  variational quantum circuits, quantum kernels, quantum annealers.
- **(B) Quantum resource consumption as the machine-burden channel.** The
  paradigm's "machine suffering" channel is already defined as an *operational
  computational-burden proxy* (energy, FLOPs, parameter norm, verification load).
  Quantum hardware makes this channel unusually concrete: circuit depth, shot
  counts, coherence-time budget, cryogenic energy, QEC overhead. A
  suffering-aware *scheduler* for quantum resources is mostly classical
  software wrapped around a quantum backend.
- **(C) Quantum simulation for suffering-relevant science.** Quantum chemistry /
  Hamiltonian simulation feeding the clinical applications of Mercyful Learning
  (e.g., drug–receptor or PK-relevant molecular properties for the vancomycin /
  chemo-sequencing lines).

Only (B) is buildable today, and it is barely "quantum" — it is resource-aware
scheduling that happens to target a quantum backend. (A) and (C) are the
versions that would justify the name, and both are blocked on hardware.

## 2. Current state of the field (assumptions and evidence base)

Assumptions, stated explicitly since this is a fast-moving field:

- Quantum ML in 2026 remains in the NISQ era. No demonstrated, replicated
  quantum advantage for a practical ML task; several early QML advantage claims
  were weakened by classical dequantization results. Barren plateaus and
  shot-noise scaling remain unresolved for variational approaches.
- Fault-tolerant quantum computing is roadmapped by major vendors for the
  ~2029–2035 window, but roadmaps in this field have historically slipped.
- Quantum chemistry advantage (claim C's substrate) is the most credible
  near-term advantage candidate, but "credible candidate" is not "delivered."
- The Mercyful Learning paradigm itself is at preprint stage with synthetic
  benchmarks and Lean formalization in progress; its classical instantiation is
  not yet validated on real clinical data (see
  `docs/research/mimic_iv_mercyful_validation_2026-07-26.md` and related).
- Repo state: no existing mercyful×quantum artifact. The only quantum-adjacent
  asset found is `benchmarks/qnn/mnist_epistemic.sio` (epistemic QNN toy). This
  direction is greenfield.

## 3. Dimension evaluations

### 3.1 Technical feasibility — score 4/10

- Claim (B) — quantum resource burden accounting: feasible now (8/10 in
  isolation), but it is classical scheduling software; calling it a quantum
  research direction is a stretch.
- Claim (A) — mercyful objectives on QML hardware: *implementable* today as a
  toy (a suffering-weighted loss on a 10–50 qubit variational circuit is a
  weekend project on a simulator), but there is no known mechanism by which the
  suffering-minimization objective benefits from quantum substrate, and no
  reason to expect one. Suffering-weighting is a change to the *loss geometry*;
  it is substrate-agnostic. Feasibility without advantage is not feasibility of
  a research direction.
- Claim (C) — quantum chemistry for mercyful clinical pipelines: blocked on
  fault-tolerant hardware; not feasible with near-future (≤5 yr) technology at
  clinically relevant system sizes.

The composite score reflects: the only feasible parts don't need quantum, and
the parts that need quantum aren't feasible.

### 3.2 Timeline — score 3/10

Scale: 10 = realizable within 1 year; 1 = 20+ years or never.

- (B) burden-aware quantum scheduling: 1–2 years (but low novelty, see above).
- (A) credible QML mercyful experiment with any advantage claim: 10+ years,
  contingent on fault tolerance *and* on someone finding a quantum-native reason
  for distributional cost objectives — neither is scheduled.
- (C) clinical-grade quantum chemistry feeding mercyful dosing: 10–20 years.

### 3.3 Impact — score 5/10 (highly conditional)

- If fault-tolerant QC arrives and QML advantage exists at all, a
  suffering-aware objective would inherit whatever advantage the substrate
  offers — but it would not *add* to it. The marginal impact of the mercyful
  layer on quantum is the same as its marginal impact on classical hardware,
  which is the (unproven, preprint-stage) core bet of the paradigm itself.
- Impact of (B) alone: low — a nice engineering contribution to quantum cloud
  cost/carbon accounting, a field that already exists (quantum resource
  estimation, energy-aware scheduling).
- Impact if everything works: medium-to-high for the clinical pipeline, but the
  quantum layer is replaceable in that pipeline by classical simulation in
  almost every scenario.

### 3.4 Risks — score 7/10 (higher = worse)

- **Hype-coupling risk (primary).** The Mercyful Learning paradigm's
  credibility rests on disciplined claim separation (proven / measured /
  conjectural) and a firm refusal of consciousness claims. Attaching the
  program to quantum computing — the field with the worst
  hype-to-delivered-ratio in modern computing — before the classical core is
  validated on real data risks tainting the whole program. "Suffering-aware
  quantum" reads, to an outside reviewer, like two speculative labels
  multiplied.
- **No-advantage risk.** Dequantization results keep showing that quantum ML
  "advantages" evaporate. The prior that a mercyful QML variant shows genuine
  advantage is low.
- **Resource risk.** Every hour spent here is an hour not spent on the
  classical validation path (MIMIC-IV, FAERS, vancomycin TDM) that actually
  gates the paradigm's acceptance.
- **Ethics-framing risk.** "Quantum suffering" invites misreading as a
  consciousness claim — exactly the misreading the paradigm paper's scope
  statement works to preclude.
- Mitigations exist (strict (B)-only scoping, explicit "no advantage claimed"
  framing), but they reduce the direction to classical scheduling.

### 3.5 Overall future potential — score 3/10

Weighted judgment, not an average: feasibility and timeline dominate because
the direction's only buildable form is not genuinely quantum, and its genuinely
quantum forms are unbuildable for a decade with no identified quantum-native
mechanism for the objective.

## 4. Score summary

| Dimension | Score (1–10) | Note |
|---|---|---|
| Technical feasibility | 4 | Feasible parts aren't quantum; quantum parts aren't feasible |
| Timeline | 3 | 10–20 yrs for meaningful versions; 1–2 yrs only for the classical-scheduling carve-out |
| Impact | 5 | Conditional on hardware + advantage that may never materialize |
| Risk | 7 | Hype-coupling, no-advantage prior, opportunity cost, ethics misreading |
| **Overall future potential** | **3** | Defer |

## 5. Recommendation: **defer**

Defer as a research direction. Two caveats:

1. **Cheap carve-out worth keeping on the radar:** if the program ever needs a
   quantum backend for another reason, claim (B) — treating quantum resource
   budgets (shots, depth, cryo-energy) as the machine-burden channel — is a
   natural, honest, near-zero-cost extension of the existing burden proxy. It
   should be a paragraph in a future scheduling paper, not a research
   direction.
2. **Revisit trigger:** re-evaluate only if (i) the classical Mercyful Learning
   core is validated on real data, *and* (ii) a replicated quantum ML advantage
   for a learning task structurally similar to suffering-weighted optimization
   is published. Both are currently absent.

## 6. Which option has the most real future potential

This evaluation covers Option 3 only; the sibling option evaluations were
delegated to other agents. Judged against the repo's existing evidence base,
Option 3 (Suffering-Aware Quantum) is almost certainly **not** the strongest
option: directions grounded in the classical paradigm (clinical validation,
suffering-aware architectures, federated SAN) have working code, gates, and
datasets in-repo today, while this option has none. Final ranking across
options is deferred to the parent's synthesis of all option reports.

---

*Evaluation performed independently by a subagent on 2026-07-30. No code
changed; no commits made. LLM-offload review not invoked: this is an internal
evaluation memo, not a math claim, clinical-pathway artifact, or
external-facing document (`.claude/AGENT_OFFLOAD_POLICY.md`).*
