<!-- docs:meta
topic_id: repo.docs.dissertation.dossiers.archetype-coverage-gate
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.dossiers.archetype-coverage-gate
-->

# Archetype coverage gate — 10 new PBPK28 scenarios

Gate rule (EPIC SIO-515): mechanism coverage requires implementation
evidence in the canonical scenario path — input, clearance, or coupling
term — not the drug name, and not an unused library function. Verdicts:
COVERED (mechanism executes), PARTIAL (part executes, part named-only),
with every named-only item located. No scenario is called validated by
parity alone; computation vs external evaluation stay separate (SIO-562).

## 1. Verdict matrix

| Scenario | Claimed archetype | What executes in code | Verdict |
| --- | --- | --- | --- |
| lithium_oral | purely renal, no CYP | renal-only CL 1.3 L/h as cl_central; F 1.00, no first-pass; no hepatic term anywhere | COVERED |
| risperidone_oral | CYP2D6 mother plus metabolite | pheno-parametrized formation sink on liver, coupled 9-OH CN; LAI explicitly disclaimed (oral-only, no LAI claim) | COVERED as oral pair |
| aripiprazole_oral | CYP2D6 plus 3A4 parallel, long half-life | two sinks in one implicit liver solve; phenotype plus PM cross-check (132 vs 146 h); dehydro out of scope | COVERED as parent dual pathway |
| everolimus_des | DES stent controlled release | Higuchi rate KH over 2 sqrt(t); 80 percent at 30 days by construction of KH | COVERED |
| clozapine_oral | CYP1A2 plus smoking | CL 20 single sink (1A2 implicit); smoker x1.5 accessor EXISTS in drug file but scenario never calls it | PARTIAL: smoking unwired |
| olanzapine_oral | CYP1A2 plus smoking | CL 25 single sink; full smoker variant EXISTS in PBPK14 drug file but PBPK28 scenario runs NM only | PARTIAL: smoking unwired |
| quetiapine_oral | CYP3A4 (plus VFX DDI) | fm 0.89 accessor exists; CL 84 single sink, no inhibitor arm; Paulzen 2018 DDI explicitly out | PARTIAL: DDI named-only |
| tacrolimus_oral | CYP3A4/3A5, F percent | F 0.25 first-pass implemented; CL 28 single sink; no 3A5-genotype split | PARTIAL: 3A5 named-only |
| tirzepatide_sc_depot | SC depot plus dual GIP/GLP-1 | weekly SC first-order depot implemented; no TMDD/PD layer | PARTIAL: pharmacology named-only |
| vancomycin_iv | IV infusion plus renal TDM | zero-order 1 h infusion implemented; no CrCl-impairment variant | PARTIAL: renal scaling named-only |

## 2. Cross-cutting finding: implemented but unwired

Three mechanism variants exist as tested library code but never execute in
a canonical scenario: clozapine_smoker_cl_multiplier (x1.5),
olanzapine_pbpk_params_smoker (CL x2, PBPK14), quetiapine_fm_cyp3a4 (0.89).
Cheapest gate upgrades if the owner wants them wired: a smoker scenario
variant for clozapine/olanzapine, a ketoconazole inhibitor arm for
quetiapine. Kept declared-out here — single-dose NM scope stands — but
their exact locations are now on record so wiring is mechanical.

## 3. What this gate does not claim

No new numeric execution; classification from code inspection of the
merged scenarios. Wiring a variant later requires its own parity plus mass
residual before it counts. Dehydro-aripiprazole, LAI risperidone,
quetiapine-venlafaxine DDI, full-Vd quetiapine, and CrCl-stratified
vancomycin remain follow-ups, not blockers.

Versions: main 3d3da08066 (post 15-drug gate); R-R module still uncalled by
drug files (SIO-550); aripiprazole values from merged PR 3460.
