<!-- docs:meta
topic_id: repo.examples.-attic.readme
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#repo.examples.-attic.readme
-->

# examples/_attic — quarantined examples

Files here were moved out of `examples/` because they do not demonstrate what
their name says. They are kept, not deleted: each sits at
`examples/_attic/<original path>`, so `git mv` back restores it, and `git log --follow`
keeps its history. Nothing in this directory is part of [`examples/INDEX.md`](../INDEX.md).

Two groups:

- **stub** — the original program is commented out under an
  `Aspirational example preserved below` header, and the live code is a
  placeholder (`scale(Sample{3}, 7)` + checksum) that prints `example: <name>` and `36`.
  It runs, but demonstrates nothing. To restore one, uncomment the original and
  make it compile on `./bin/souc check`.
- **legacy** — pre-module example that fails on both engines (`./bin/souc run` and
  `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run`) with `E137` (use of undeclared
  variable), because it calls library functions it never imports, or uses syntax or
  intrinsics the compiler no longer has. Measured in the two-engine sweep of
  2026-10-05 and re-checked on `main` 4a8adec60 before the move.

| file | group | reason | moved |
|---|---|---|---|
| `alpha_geo_zero.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `alpha_sounio.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `alphageozero_final.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `alphageozero_selfplay.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `async_demo.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `autodiff.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `beta_epistemic.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `build_system_demo.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `cross_compilation.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `darwin_atlas/approx_metric.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `darwin_atlas/exact_symmetry.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `darwin_atlas/ffi.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `darwin_atlas/operators.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `darwin_atlas/quaternion.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `darwin_atlas_operators.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `darwin_compat_test.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `data_pipeline/main.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `day20_tooling.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `day21_build_system.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `debug_profile_demo.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `distributed_build.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `effects.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `epistemic_ml_demo.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `ffi_demo.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `ffi_exports.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `fidelity_demo.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `imo_benchmark_eval.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `imo_showcase.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `linalg_demo.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `macro_system_demo.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `monte_carlo/main.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `ode_demo.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `pbpk/caffeine_model.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `pbpk/caffeine_model_v2.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `pbpk/darwin_14comp.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `pbpk/darwin_full_ode.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `pbpk/darwin_full_pbpk.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `pbpk/darwin_output.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `pbpk/darwin_pbpk_14comp.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `pbpk/darwin_pbpk_units.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `pbpk/darwin_validation_1232.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `pbpk/mechanistic_ddi.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `pbpk/metformin_model.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `pbpk/metformin_model_v2.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `pbpk/metformin_simulation.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `pbpk/neural_ode_pbpk.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `pbpk/ode_solver.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `pbpk/pbpk_simple.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `pbpk/rbc_dynamics.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `pbpk_model/main.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `pharmacokinetics.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `pkpd.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `quantum_h2_vqe.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `quaternion_embeddings.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `scientific_computing_demo.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `scientific_computing_full.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `test_example.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `test_linalg.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `tile_matmul.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `uncertainty_aspirational.sio.disabled` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `unified_epistemic_science.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `units_simple.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `watch_mode_demo.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `wave1_bayesian_coin.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `wave1_ode_exponential_decay.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `wave1_projectile_motion.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `wave1_tensor_operations.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `wave1_uncertainty.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `wave2_autodiff_simple.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `wave2_gradient_descent.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `wave2_neural_ode.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `wave3_explainable_nn.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `wave3_hybrid_regression.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `wave3_kepler_discovery.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `wave3_pinn_heat.sio` | stub | original commented out; placeholder prints `36` | 2026-10-06 |
| `epistemic/autodiff_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `epistemic/budget_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `epistemic/causal_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `epistemic/combine_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `epistemic/core_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `epistemic/correlation_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `epistemic/coverage_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `epistemic/dual_check_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `epistemic/fusion_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `epistemic/gum_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `epistemic/interval_ieee_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `epistemic/invariants_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `epistemic/montecarlo_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `epistemic/multivariate_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `epistemic/policy_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `epistemic/proptest_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `epistemic/prov_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `epistemic/refutation_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `epistemic/roi_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `epistemic/slsa_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `epistemic/sobol_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `epistemic/stats_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `epistemic/traceability_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `fmri/atlas_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `fmri/connectivity_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `fmri/connectivity_epistemic_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `fmri/nifti_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `fmri/pipeline_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `fmri/preprocess_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `fusion/eeg_fmri_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `multi_feature_training_example.sio` | legacy | uses an undeclared `Quaternion` and pre-current syntax; E137 + E004/E019 and others on both engines | 2026-10-06 |
| `ontology/biomedical/go_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `ontology/biomedical/hpo_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `ontology/biomedical/loinc_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `ontology/namespaces_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `pbpk/covariate_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `pbpk/error_models_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `pbpk/population_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `pbpk/types_demo.sio` | legacy | calls library functions it never imports (pre-module demo); E137 on both engines | 2026-10-06 |
| `science/epistemic_adaptive.sio` | legacy | Gen-17 R15 intrinsics `read_conf`/`update_conf` no longer exist; E137 on both engines | 2026-10-06 |
| `units/basics.sio` | legacy | Gen-19 `_suffix` unit literals (`500.0_mg`) no longer resolve; E137 + E001/E004 on both engines | 2026-10-06 |
