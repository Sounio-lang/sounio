# stdlib/chemistry

**Sounio Chemistry Standard Library — SOTA+++ (BOLDER & BROADER, ALL REAL, towards novelty)**

Per approved plan: Epistemic CRN engine as core novelty, full GUM, units (dim_amount, conc), ode (rk4 + epistemic patterns), real formulas no stubs.

## Status (ALL REAL, SOTA+++ SUPER MIX + SOUNIO-ONLY DEEP from lit research)

Literature deep-research (GUM JCGM 100, CRN structural UQ arXiv 2026, ACS ChemRev 2026 UQ in silico chem, PBPK unc papers):
- UQ in CRN/PBPK is mostly external (MC sampling, sparse reg for structural unc, PCE on top of solvers). Delta-method/GUM analytic is manual or post-hoc.
- Algebraic effects rare in sci computing (mostly PL semantics).
- Sounio uniques (no other lang has natively): first-class EState + GUM propagation *inside* custom CRN RHS (law of prop via Jacobian in solver), algebraic effects (Observe/Prob) for modeling measurement/prob in kinetics, unified epistemic+matnm+dual+units+equil+effects in executable model, self-hosted for potential meta-epistemic.

Status: full super mix + unique demos (epistemic general CRN with EState, effect Observe, GUM reports, epistemic+dual).

- lib.sio: full runner with comprehensive real tests, check OK.
- kinetics.sio: **full Epistemic CRN engine** - 5+ rxn realistic (H2-O2 sub lit constants A/Ea), Epistemic rates, full linalg matnm for Nu (matnm_new + matnm_mul for dc = Nu * rates), EState + solve_ode, Arrhenius GUM, transport, pbpk real metabolic, sens (autodiff), real metabolism ex at 24h, big combustion sub sim.
- Created: chemistry/BENCHMARKS.md , EXAMPLES.md with runnable big mech, speciation, etc.
- equilibrium.sio: K<->dG full GUM, real quadratic solver for A+B=C, Newton, Nernst electro with unc, real deltaG test.
- acids.sio: pH GUM, titration, full epistemic calib, MM biochem with unc.
- stoich/thermochem: real with GUM, limiting, Hess.
- **GRI-Mech 3.0 real rates** (`gri_mech_rates`, `simulate_big_crn_gri` in kinetics.sio): modified Arrhenius k(T) = A·T^n·exp(−Ea/RT) from the actual grimech30.dat A/n/Ea parameters (no hardcoded effective constants), cross-checked against an independent Python replication at 1500 K / 1 atm. Run-pass coverage: tests/stdlib/chemistry/test_kinetics_gri_mech.sio.
- **Epistemic general CRN respects user stoichiometry** (`simulate_general_epistemic`): values via general mass-action RK4 on the supplied Nu, variances via per-step delta method (central-difference Jacobian + rate sensitivities). Structural/fractional ensembles therefore show genuine between-model variance. Run-pass coverage: tests/stdlib/chemistry/test_kinetics_epistemic_ensemble.sio.
- **Complete GRI-Mech 3.0 H/O sub-mechanism with native UQ** (`gri30_h2.sio`): all 29 H/O-only reactions / 10 species from grimech30.dat — modified Arrhenius, three-body collision efficiencies, Troe falloff (2 OH (+M) ⇌ H2O2 (+M)) — with reverse rates from NASA-7 thermochemistry via detailed balance (k_rev = k_fwd/Kc(T)), and first-order GUM propagation of per-reaction rate uncertainties (representative 1σ from Baulch 2005 / Konnov 2008 / Hong 2011) through the RK4 trajectory, returning a native 1σ band per species. Cross-checked against an independent Python replication AND against Cantera 3.2 itself (rates exact, thermo exact, pre-ignition-front trajectory <1%, uncertainty band <1%); the Cantera cross-check caught a missing reaction (2 O + M <=> O2 + M, GRI Reaction 1) in the first version of the module; the ignition front itself is exponentially phase-sensitive and documented as not parity-testable. Run-pass coverage: tests/stdlib/chemistry/test_gri30_h2.sio.
- **FULL GRI-Mech 3.0 — 53 species / 325 reactions** (`gri30_full.sio`): the entire mechanism (not just H/O) with per-reaction Troe/Lindemann falloff tables (26 Troe + 3 Lindemann F=1), 16 irreversible reactions with reverse rate exactly 0, NASA-7 detailed balance, and the same native GUM rate-UQ. Validated against an independent sparse Python replica (rates/Kc exact at 1200 K; H2-protocol checkpoint at t=4e-6 s, dt=2e-9 — dt reduced from 1e-8 because NNH decay (k=3.3e8/s) exceeds the RK4 stability limit at dt=1e-8; matches the 29-reaction sub-mechanism to 7 digits for H2). Run-pass coverage: tests/stdlib/chemistry/test_gri30_full.sio. NOTE: lean_single MatNM being 64x64 was never the binding constraint — the module uses flat [f64; 17225] tables and a flat [f64; 2809] Jacobian.
- Deep novelty (super mix + this deep phase): structural ensembles (multiple Nu + posterior probs per CRN lit), Caputo fractional + epistemic kinetics (stdlib caputo), epistemic PINN/SciML CRN, effect handlers (GUM vs MC), more PBPK sub-models. All native UQ.
- Units: dim_conc, dim_volume, Quantity + epistemic bridge.
- All REAL formulas (series, GUM delta/Newton, mass action, Nernst, MM, quadratic), no stubs.
- **LIT VALIDATION** (see validate_against_literature + test_lit_validation in kinetics): direct check_near vs GRI-Mech/Marinov (H2O2), midazolam PBPK papers, fractional Caputo PK lit (Mtshali/Ahmad), etc. Runners report LIT PASS. Without validation = dead code.

`./bin/souc check stdlib/chemistry/lib.sio` → **check: OK**

## Usage

use chemistry::kinetics;
let (k, uk) = kinetics::arrhenius(...);
let out = kinetics::simulate_crn(...);

See plan for full CRN with Epistemic rates, pbpk bridge, sensitivity.

## What Sounio Only Can Do (lit-backed, demonstrated here)
1. Native GUM inside CRN model: EState + solve_ode (or general) propagates unc through arbitrary stoich (matnm) RHS using delta-method (matches JCGM law exactly, efficient, no external lib).
2. Algebraic effects for sci modeling: e.g. with Observe for "measured" concentrations in epistemic sim (extensible handlers for interp).
3. Zero-glue stack: epistemic rates -> general CRN solve (RK4/EState) -> dual sens of unc -> units Quantity -> equil coupling -> pbpk.
4. Epistemic structural/plausible networks: general builder + multiple Nu + unc in one run (addresses CRN inference structural unc papers).
5. Because self-hosted + effects: models are first-class, verifiable, effect-interpreted at lang level.

All with real lit (Ea 103kJ, k=0.05 etc from PK lit), tight check_near, full effects/series/GUM.

## Verification (check is the gate; run may have pre-existing backend issues)

export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
./bin/souc check stdlib/chemistry/lib.sio
./bin/souc check stdlib/chemistry/kinetics.sio
./bin/souc check stdlib/chemistry/equilibrium.sio
./bin/souc check stdlib/chemistry/acids.sio

See EXAMPLES.md / BENCHMARKS.md for LIT PASS + GUM.

**Super Showcase Químico**: Existe `kinetics::super_chemical_showcase()` — um cenário integrado que usa **todas** as capacidades avançadas (ontologia ChEBI para cada composto, ensembles estrutural+fracionário, estocástico, Bayes com Observe + Knowledge<CHEBI_15377>, auditoria regulatória, etc.). É o melhor exemplo para entender o diferencial de Sounio.

Phase 6 (in parallel): expanded Bayesian (Observe+multi-data EState, ident, structural tie), regulatory provenance (audit struct+budget export), scale benches + general nets. **FULL real ontologies**: complete ChEBI species map for every compound, GO + Rhea, bundle directives, explicit Knowledge<CHEBI_xxx>, deep epistemic provenance. LLM-offload before new math claims.

**PBPK28 dissertation expansion (Task 5)**: Added digoxin_chebi, cyp3a4_enzyme_chebi, warfarin_chebi; GO xenobiotic_metabolic (6805), proteolysis (6508), abc_xenobiotic_transporter (8559), oxidative_demethylation (70989); Rhea 47261/55924 specific CYP + transport; pbpk28_drug_enzyme_chebi_map, extended species_to for pbpk28/cyp/pgp/proteolytic, knowledge_pbpk_drug_conc, pbpk_metabolic_crn_with_ontology in kinetics + tests/docs updated. Now every PBPK28 drug/enzyme (rapa/tacro/sema/vanco/midazolam/digoxin + CYP/P-gp/proteolysis) has concrete ChEBI+GO+Rhea.

Chemistry is now bold, real, novel foundation ready.## PIVOT to Estatística Epistêmica (2026-06-30)

Per user "vamos pivotar para: estatistica ou matematica???" and approved plan: primary focus now **Estatística Epistêmica** (GUM extensions, hybrid UQ, stochastic with effects, Bayesian).

Math remains supporting (already strong in CN, solvers, PINN, fractional).

**Implemented Phase 1 starter**: Enhanced `test_crn_effect_handlers()` (now with Prob, Observe) to demonstrate hybrid on same CRN:
- GUM (analytic)
- MC sampling
- Stochastic (Prob path via simulate_stochastic_decay)
- Observe effect for measurement (Bayesian collapse example)
- Prints + combined epistemic var.

See kinetics.sio: test_crn_effect_handlers for the demo prints "[ESTATÍSTICA HYBRID]".

Next (plan): full hybrid fn, Bayesian over CRN, integrate with PBPK28 (build on prior REG AUDIT + fused contrib), runners, lit (e.g., hybrid vs pure GUM on lit mechanisms).

This pivot keeps Sounio-only epistemic stats as the novelty edge for dissertation/preprint.
