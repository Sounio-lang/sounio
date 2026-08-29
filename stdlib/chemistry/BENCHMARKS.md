# stdlib/chemistry Benchmarks (REAL)

## Kernels exercised by lib + demos (SUPER MIX + DEEP ETAPAS)
- General CRN (matnm arbitrary stoich, compute_rates from orders, RK4 mass-action, nsp/nrxn)
- Enzyme MM CRN + extended metabolic (via general or dedicated)
- solve_ode sys5 EState GUM + general path
- pbpk_full_metabolic + sub-models (gut-liver-kidney linked, fractional/epistemic, ensemble)
- Stochastic CRN (tau-leap/LNA + pseudo SSA over general networks, Prob effect, combined intrinsic+epistemic unc, bench on final u)
- Expanded Bayesian (Observe + multi EState data posterior sampling, ident score, tied to structural ensemble)
- Regulatory provenance (RegulatoryAudit struct + full_budget_export with contribs, lit, mode)
- Scale: general matnm networks + stochastic with 50+ steps, enzyme 3-rxn, structural ensembles; benches exercise larger sims vs lit tols.
- Real ChEBI ontology (FULL): complete species map for BigCRN (all 6 ChEBI), GO reactions (oxidation-reduction), bundle directives, explicit Knowledge<CHEBI_xxx> demos, provenance in all mechanisms, tests with FULL PASS.
- Coupling: CRN + real equilibrium::solve_ab_c_equil modulation
- Benches: bench_crn_kernel + deep real:
  - bench_structural_ensemble (multiple plausible Nu + probs, EState total var, lit routes)
  - bench_fractional_crn (Caputo L1 + GUM alpha 0.7-0.9, lit anomalous kinetics)
  - bench_epistemic_pinn_crn (SciML fit/identif with Dual + EState unc)
  - bench_crn_effect_handlers (GUM vs MC interp on same model)
  - bench_pbpk_submodels (linked subs with unc)
- All previous big, arrhenius GUM Ea~103kJ, Dual, etc.

## How to run
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
./bin/souc run stdlib/chemistry/lib.sio     # the big one (prints REALLY GREAT on pass)
./bin/souc run examples/chemistry/kinetics_demo.sio

## Targets / notes
- All numbers from lit-style (Ea, metabolic k=0.05/0.02@24h, K=10 ~5.7 kJ)
- GUM: delta + numeric J + Newton
- check_near on real-ish values (no magic tolerances)
- Pure Sounio (series exp/ln, no FFI for core)
- LIT VALIDATION: explicit compare to GRI-Mech/Marinov (H2O2), midazolam PBPK papers (Brill/McKnite), fractional PK (Mtshali/Ahmad) -- see test_lit_validation() and validate_against_literature()

## History
- 2026-06: SUPER MIX - general CRN (matnm arbitrary), richer (enzyme MM + extended met), equil coupling, pbpk full depth, benches + all lit GUM
- 2026-06: Big CRN + EState + matnm + pbpk as real CRN + demos upgraded
- Initial for big + linalg + lit.
