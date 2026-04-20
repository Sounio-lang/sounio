# Research Roadmap — Q2/Q3 2026 Milestone Document

**Author**: Demetrios Chiuratto Agourakis (ORCID: 0009-0001-8671-8878)  
**Date**: 2026-04-20  
**Scope**: 2026-04-20 to 2026-09-22 (6 months); dissertation deadline **CRITICAL**  

## 1. Gantt-Style Month-by-Month Breakdown

| Month       | Dissertation Focus                          | Paper 3 (Connectomics)          | Paper 4 (Seizure) | Paper 5 (Compilation) | Cross-Cuts                  |
|-------------|---------------------------------------------|---------------------------------|-------------------|-----------------------|-----------------------------|
| **Apr 2026** (20th-start) | β⁵ inter-procedural variance impl (~500 LoC); rapamycin_epistemic_adaptive.sio unblock | HPC session + container build; Phase 1 n=10 run | Draft polish (intro/methods) | Abstract + intro draft | β⁵ prototype test; HPC ticket |
| **May 2026** | β⁵ complete (~500 LoC); ODE+GUM integration; 3-comp PBPK model stub | Phase 1 results; |d| check | Results section | Related work + β⁴ eval | Confidence gates prototype |
| **Jun 2026** | Full sim runs (Rapamycin PK); uncertainty budgets; sensitivity_of() validation | Phase 1 paper draft (if |d|>0.15) | Full draft | Methods + impl details | HPC Phase 2 queue (if gate pass) |
| **Jul 2026** | Compile-time gates; full epistemic PBPK validation; draft Ch1-3 | Phase 2 n=1034 kickoff (if gated) | Submit to Epilepsia | Revisions; submit POPL/PC | Advisor review loop 1 |
| **Aug 2026** | Draft Ch4-6 + results; revisions; uncertainty budget ISO compliance | Phase 2 interim results | Revisions (if R&R) | PLDI fallback prep | Submission buffers |
| **Sep 2026** (to 22nd) | Final polish; defense prep; submit | Phase 2 draft (Network Neuroscience) | - | - | Final reviews |

## 2. Per-Paper Status

| Paper | Status | Next Action | Target Deadline | Risk Level |
|-------|--------|-------------|-----------------|------------|
| **1. 168 Theorem** | Submitted (AACA 03-21, EJM 03-31); E2E Sounio verified | Monitor reviews; prep 1-page rebuttal if R&R | Passive (reviews ~Q4) | Low (no action needed) |
| **2. G₂ Bridge** | CLOSED (null results: Gram trivial, d=0.06, z=2 artifact) | None | N/A | None |
| **3. Non-Assoc Epistemic Connectomics** | Phase 1 (n=10): α-gate PASS (04-12); BLOCKED on HPC/container | 1. File K8s/Slurm ticket + build beagle-sounio img (by 04-30). 2. Run Phase 1; compute |d|. 3. If >0.15, queue Phase 2 n=1034 | Phase 1 complete: 05-15; Phase 2 draft: 09-15 (Network Neuroscience) | High (HPC access) |
| **4. Seizure Dynamics** | S-SSM gating: 5.6× reg on EEG (12 expts); artifacts ready | 1. Polish draft (intro/methods/results by 05-15). 2. Submit Epilepsia | Submit: 07-15 | Medium (journal fit) |
| **5. Epistemic Gradual Compilation** | β⁴ landed (variance/sensitivity builtins); Sounio design core | 1. Draft sections (abs/intro/related: 04-30; methods/impl: 06-15). 2. β⁵ tie-in evals. 3. Submit POPL (if ready) or PLDI | Submit: 08-01 (POPL deadline proxy) | High (β⁵ dependency) |

## 3. Dissertation: Month-by-Month Deliverables + Final Checklist

**Title**: GUM-Native Pharmacokinetic Simulation via Epistemic Gradual Compilation (Rapamycin PBPK)  
**Est. Remaining**: 1,500 LoC; sims + writing.  
**Definition of "Done" (Submission 2026-09-22)**:  
- Code: Full epistemic_adaptive.sio runnable; Bogacki-Shampine ODE+GUM; confidence gates; ISO uncertainty budgets (3-comp model validated on Rapamycin Cypher data).  
- Outputs: PK curves + epistemic envelopes (blood/brain/periph); sensitivity analysis; advisor-approved.  
- Doc: 80+ page PDF (Ch1-6 + appendices); LaTeX src; defense slides (20 min).  
- Metrics: variance_of() <5% on PK peaks; metrological step control passes 95% bootstraps.  
- Submitted: PDF + src to advisor/committee via institutional repo.

**Month-by-Month**:
- **Apr**: β⁵ variance barrier impl (500 LoC); unblock sim; test on toy ODE.
- **May**: β⁵ complete (500 LoC); 3-comp PBPK stub; initial Rapamycin sims (no gates).
- **Jun**: GUM propagation + metrological control; sensitivity_of() on full model.
- **Jul**: Compile-time gates; validation runs; draft Ch1-3 (intro/lit/Methods).
- **Aug**: Full results (Ch4-5); uncertainty budgets; revisions from advisor.
- **Sep (to 22)**: Ch6/discussion; appendices; final proofread; submit.

## 4. Cross-Cutting Risks

| Risk | Description | Mitigation | Owner/Action |
|------|-------------|------------|-------------|
| **β⁵ Inter-Procedural Variance** | Core blocker for diss/paper5; ~1k LoC untested | Prototype by 04-30; fallback: scalar variance approx (lose 20% fidelity) | Self: daily 4hr coding block |
| **HPC Access** | BeagleCockpit/K8s needed for Phase 3 connectomics | Ticket 04-22; alt: local GPU (n=10 only, delay Phase 2) | Self: submit today; weekly ping |
| **Reviewer Timelines** | Paper1 passive; paper4/5 submissions mid-year | Buffer 1mo post-deadline; dual-sub fallback (e.g., PLDI→arXiv) | Advisor: 07-01 sync |
| **Advisor Bandwidth** | Excited but shared | Biweekly 30min mtgs; share drafts early (Jun/Jul) | Self: schedule 05-01 |
| **Compute Overload** | GPU queue for Phase2/simulations | Prioritize diss sims; cap connectomics at 20% time | Self: job scripts with priorities |

**Overall Risk**: Medium-High (β⁵/HPC = 70% timeline impact).

## 5. Priority Stack (Time-Scarce Ordering)

1. **Dissertation β⁵** (4-6hr/day; blocks everything).
2. **Paper5 Draft** (2hr/day; leverages diss code).
3. **HPC Unblock → Connectomics Phase1** (1hr/day until running).
4. **Seizure Draft** (weekends; low-code).
5. **Reviews/Monitoring** (passive; 15min/day email check).

**Weekly Cadence**: Mon: plan; Fri: commit review + advisor ping if stalled. Track in GitHub Projects. Total est. 40hr/wk feasible.
