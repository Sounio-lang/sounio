# Examples index

Generated 2026-10-06 at commit `6c842d2d3bfb` by `bash scripts/dev/gen_examples_index.sh`.
Do not edit by hand; re-run the script.

Each example with a `fn main` was run from the repository root on both engines, with a 120s timeout and stdin from `/dev/null`:

- **Madaros**: `./bin/souc run <file>` (the default engine, Madaros v0.80.0)
- **lean_single**: `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run <file>`

Files without `fn main` are libraries (modules other files import); they are listed, not run.
Quarantined examples (stubs, pre-module legacy syntax) are in [`_attic/`](_attic/README.md).

## Summary

| | count |
|---|---:|
| `.sio` files indexed | 980 |
| programs (have `fn main`) | 963 |
| libraries (no `fn main`) | 17 |
| programs passing on Madaros | 503 |
| programs passing on lean_single | 450 |
| programs passing on both | 378 |
| programs passing on neither | 388 |
| timeouts (Madaros / lean_single) | 29 / 68 |

Status: `pass` = exit code 0; `fail rc=N (Exxx)` = non-zero exit with the first diagnostic codes printed; `timeout` = killed after 120s; `crash (signal N)` = killed by a signal. Wall time includes compilation.

## Programs and libraries

| file | what it is | Madaros | lean_single | wall time (s) Madaros / lean_single |
|---|---|---|---|---|
| [`ablation_suite.sio`](ablation_suite.sio) | Ablation Suite: Isolating Associator Feature from Non-Associative Dynamics | crash (signal 54) | timeout | 66.0 / 120.0 |
| [`adding_problem_benchmark.sio`](adding_problem_benchmark.sio) | Adding Problem: O-SSM vs Diagonal — Classic RNN Benchmark | pass | fail rc=1 (E035) | 22.0 / 0.8 |
| [`advanced_glm_optimization.sio`](advanced_glm_optimization.sio) | Advanced GLM-4.7 Optimization Demo | fail rc=1 (E043) | fail rc=1 (E200 E035 E043) | 4.6 / 0.8 |
| [`ai_scientific_demo.sio`](ai_scientific_demo.sio) | AI Scientific Computing Demo - Sounio | fail rc=1 | fail rc=1 (E260 E035 E200) | 4.5 / 1.0 |
| [`algebra_demo.sio`](algebra_demo.sio) | algebra_demo.sio — How a mathematician uses Sounio | pass | pass | 5.4 / 0.8 |
| [`algo/graph_demo.sio`](algo/graph_demo.sio) | Graph Algorithms Demonstration | fail rc=1 | fail rc=1 (E035 E001 E006) | 5.0 / 1.0 |
| [`algo/sorting_demo.sio`](algo/sorting_demo.sio) | Sorting Algorithms Demonstration | fail rc=1 | fail rc=1 (E001 E035 E200) | 4.4 / 0.8 |
| [`algorithms/complex_native_demo.sio`](algorithms/complex_native_demo.sio) | COMPLEX DEMO: Quicksort + Sieve + GCD + Fibonacci + Collatz | pass | pass | 6.5 / 0.7 |
| [`algorithms/mandelbrot.sio`](algorithms/mandelbrot.sio) | Mandelbrot Set: ASCII render at 80×40 resolution | pass | fail rc=1 (E035) | 5.7 / 0.7 |
| [`algorithms/matrix_mul.sio`](algorithms/matrix_mul.sio) | Dense Matrix Multiplication: 64×64 matrices (262,144 multiply-add ops) | pass | pass | 4.9 / 0.7 |
| [`algorithms/quicksort.sio`](algorithms/quicksort.sio) | Quicksort: in-place sort of 10,000 integers | pass | pass | 5.5 / 0.9 |
| [`algorithms/radix_sort.sio`](algorithms/radix_sort.sio) | Radix Sort (LSD, base-256) on 100,000 integers | pass | pass | 6.1 / 0.8 |
| [`algorithms/sieve.sio`](algorithms/sieve.sio) | Sieve of Eratosthenes: find all primes up to 100,000 | pass | pass | 6.0 / 0.8 |
| [`alpha_hierarchical_reanalysis.sio`](alpha_hierarchical_reanalysis.sio) | Exp 3: Subject-stratified mixed-effects re-analysis of α | crash (signal 11) | pass | 6.2 / 0.8 |
| [`arithmetic.sio`](arithmetic.sio) | — | fail rc=3 | fail rc=3 | 5.4 / 0.9 |
| [`associativity_probe_benchmark.sio`](associativity_probe_benchmark.sio) | Associativity Probe: Direct Octonionic Composition — THE Definitive Test | pass | timeout | 31.7 / 120.0 |
| [`atomic_demo.sio`](atomic_demo.sio) | Demonstration of atomic operations for lock-free concurrency | fail rc=1 (E137) | fail rc=1 (E200) | 4.7 / 0.8 |
| [`autodiff/dual_demo.sio`](autodiff/dual_demo.sio) | Demo for stdlib/autodiff/dual.sio | fail rc=1 (E137) | fail rc=1 (E200) | 5.1 / 0.8 |
| [`autodiff/epistemic_dual_demo.sio`](autodiff/epistemic_dual_demo.sio) | Demo for stdlib/autodiff/epistemic_dual.sio | fail rc=1 (E137) | fail rc=1 (E200) | 4.3 / 0.9 |
| [`autodiff/grad_demo.sio`](autodiff/grad_demo.sio) | Demo for stdlib/autodiff/grad.sio | fail rc=1 (E137) | fail rc=1 (E200) | 4.2 / 0.9 |
| [`autodiff/tape_demo.sio`](autodiff/tape_demo.sio) | Reverse-Mode Automatic Differentiation Demo | fail rc=1 | fail rc=1 (E001 E006) | 4.5 / 1.0 |
| [`autodiff/tape_test.sio`](autodiff/tape_test.sio) | Test runner for reverse-mode AD (stdlib/autodiff/tape.sio) | fail rc=1 | fail rc=1 (E001 E006) | 4.8 / 0.9 |
| [`autodiff_neural.sio`](autodiff_neural.sio) | autodiff_neural.sio — Self-contained 2-layer neural net with manual backprop | pass | fail rc=1 (E035) | 5.8 / 1.0 |
| [`bayes/diagnostics_demo.sio`](bayes/diagnostics_demo.sio) | Demo for stdlib/bayes/diagnostics.sio | fail rc=1 | fail rc=1 (E200) | 4.4 / 0.6 |
| [`bayes/mcmc_demo.sio`](bayes/mcmc_demo.sio) | Demo for stdlib/bayes/mcmc.sio | fail rc=1 (E137) | fail rc=1 (E200 E035 E001) | 4.5 / 0.8 |
| [`bayes/prior_demo.sio`](bayes/prior_demo.sio) | Demo for stdlib/bayes/prior.sio | fail rc=1 | fail rc=1 (E200) | 4.3 / 0.8 |
| [`bayes/vi_demo.sio`](bayes/vi_demo.sio) | Demo for stdlib/bayes/vi.sio | fail rc=1 (E137) | fail rc=1 (E200 E035 E001) | 4.1 / 0.8 |
| [`bayesian_ssm.sio`](bayesian_ssm.sio) | Bayesian State-Space Model with Epistemic Uncertainty | pass | pass | 5.6 / 0.9 |
| [`binary_norm_proof.sio`](binary_norm_proof.sio) | binary_norm_proof.sio | pass | pass | 5.2 / 0.8 |
| [`bracket_3way_benchmark.sio`](bracket_3way_benchmark.sio) | Bracket Matching 3-Way: O-SSM vs S4-DIAG vs Naive-DIAG | pass | pass | 12.7 / 49.0 |
| [`bracket_l_sweep.sio`](bracket_l_sweep.sio) | Bracket L-Sweep: L={6,8,10,12,16,20} - convergence vs sequence length | pass | pass | 8.8 / 42.7 |
| [`bracket_matching_benchmark.sio`](bracket_matching_benchmark.sio) | Bracket Matching: Hierarchical Structure Recognition | pass | pass | 10.4 / 34.0 |
| [`brain_associator_demo.sio`](brain_associator_demo.sio) | brain_associator_demo.sio — Clinical Brain Network Analysis | pass | pass | 7.1 / 0.9 |
| [`brain_hessian_abide.sio`](brain_hessian_abide.sio) | Brain O-SSM Hessian Diagnosis on ABIDE | fail rc=1 (E011) | timeout | 3.8 / 120.0 |
| [`brain_orc_demo.sio`](brain_orc_demo.sio) | brain_orc_demo.sio — Brain ORC + Associator Field Pipeline | pass | pass | 7.5 / 1.0 |
| [`brain_ossm_abide.sio`](brain_ossm_abide.sio) | Brain O-SSM ABIDE Cross-Site CV | fail rc=1 (E011) | timeout | 5.9 / 120.0 |
| [`brain_ossm_classifier.sio`](brain_ossm_classifier.sio) | Brain Connectome O-SSM Classifier: ASD vs ADHD vs Control | pass | pass | 12.4 / 39.9 |
| [`bubble_sort.sio`](bubble_sort.sio) | Bubble sort using struct wrapper pattern (idiomatic Sounio) | pass | pass | 4.7 / 1.0 |
| [`categorical_bridge_bidir.sio`](categorical_bridge_bidir.sio) | CATEGORICAL BRIDGE — Part 2: Bidirectional Typing as Forward/Backward Propagation | pass | pass | 4.8 / 0.8 |
| [`categorical_bridge_dagger.sio`](categorical_bridge_dagger.sio) | CATEGORICAL BRIDGE — Part 3: Dagger Decomposition (84/84 Duality) | pass | pass | 5.0 / 0.8 |
| [`categorical_bridge_skew.sio`](categorical_bridge_skew.sio) | CATEGORICAL BRIDGE — Part 1: Skew Associator Classification | pass | pass | 5.4 / 0.9 |
| [`causal/core_demo.sio`](causal/core_demo.sio) | Demo for stdlib/causal/core.sio | fail rc=1 (E137 E019 E011) | fail rc=1 (E200) | 4.4 / 0.9 |
| [`causal/discovery_demo.sio`](causal/discovery_demo.sio) | Demo for stdlib/causal/discovery.sio | fail rc=1 (E137 E011) | fail rc=1 (E200) | 4.4 / 0.8 |
| [`causal/mod_demo.sio`](causal/mod_demo.sio) | Demo for stdlib/causal/mod.sio | pass | pass | 4.4 / 0.7 |
| [`causal/refutation_demo.sio`](causal/refutation_demo.sio) | Demo for stdlib/causal/refutation.sio | fail rc=1 (E137) | fail rc=1 (E001 E200) | 4.4 / 0.6 |
| [`causal/uplift_demo.sio`](causal/uplift_demo.sio) | Demo for stdlib/causal/uplift.sio | fail rc=1 (E137) | fail rc=1 (E200) | 4.2 / 0.6 |
| [`causal_model.sio`](causal_model.sio) | Causal Model Example - Sounio Language | fail rc=1 | pass | 4.0 / 0.7 |
| [`cayley_dickson_borromean_rerun.sio`](cayley_dickson_borromean_rerun.sio) | examples/cayley_dickson_borromean_rerun.sio | pass | fail rc=1 (E035) | 13.2 / 0.7 |
| [`cayley_dickson_hessian_tower.sio`](cayley_dickson_hessian_tower.sio) | cayley_dickson_hessian_tower.sio | pass | pass | 8.2 / 0.9 |
| [`cayley_dickson_lemon_g2_ffi.sio`](cayley_dickson_lemon_g2_ffi.sio) | examples/cayley_dickson_lemon_g2_ffi.sio | pass | pass | 6.9 / 2.4 |
| [`cayley_dickson_sedentree_realtree16_training.sio`](cayley_dickson_sedentree_realtree16_training.sio) | examples/cayley_dickson_sedentree_realtree16_training.sio | crash (signal 53) | crash (signal 11) | 16.1 / 1.0 |
| [`cd_l10_projective_measurement.sio`](cd_l10_projective_measurement.sio) | examples/cd_l10_projective_measurement.sio | fail rc=1 (E035 E008 E004) | pass | 4.2 / 44.3 |
| [`cd_l11_projective_measurement.sio`](cd_l11_projective_measurement.sio) | examples/cd_l11_projective_measurement.sio | fail rc=1 (E035 E008 E004) | timeout | 3.7 / 120.0 |
| [`cd_l9_projective_measurement.sio`](cd_l9_projective_measurement.sio) | examples/cd_l9_projective_measurement.sio | fail rc=1 (E035 E008 E004) | pass | 4.1 / 5.9 |
| [`cdylib_export.sio`](cdylib_export.sio) | Example: Exporting functions for use from C/Python/other languages | fail rc=1 | fail rc=1 (E200) | 3.6 / 0.6 |
| [`chemistry/band_sweep.sio`](chemistry/band_sweep.sio) | examples/chemistry/band_sweep.sio | pass | timeout | 29.0 / 120.0 |
| [`chemistry/full_probe.sio`](chemistry/full_probe.sio) | examples/chemistry/full_probe.sio | timeout | timeout | 120.0 / 120.0 |
| [`chemistry/gbs_oracle.sio`](chemistry/gbs_oracle.sio) | examples/chemistry/gbs_oracle.sio | fail rc=1 (E001) | timeout | 4.0 / 120.0 |
| [`chemistry/h2_adiabatic_shocktube_demo.sio`](chemistry/h2_adiabatic_shocktube_demo.sio) | examples/chemistry/h2_adiabatic_shocktube_demo.sio | timeout | timeout | 120.0 / 120.0 |
| [`chemistry/h2_flame_speed_demo.sio`](chemistry/h2_flame_speed_demo.sio) | examples/chemistry/h2_flame_speed_demo.sio | timeout | timeout | 120.0 / 120.0 |
| [`chemistry/h2_ignition_uq_demo.sio`](chemistry/h2_ignition_uq_demo.sio) | examples/chemistry/h2_ignition_uq_demo.sio | timeout | timeout | 120.0 / 120.0 |
| [`chemistry/h2_precision_probe.sio`](chemistry/h2_precision_probe.sio) | examples/chemistry/h2_precision_probe.sio | timeout | timeout | 120.0 / 120.0 |
| [`chemistry/h2_probe2.sio`](chemistry/h2_probe2.sio) | examples/chemistry/h2_probe2.sio | timeout | timeout | 120.0 / 120.0 |
| [`chemistry/h2_surface_reaction_order.sio`](chemistry/h2_surface_reaction_order.sio) | examples/chemistry/h2_surface_reaction_order.sio | pass | pass | 10.2 / 5.0 |
| [`chemistry/rep_adiabatic_bug.sio`](chemistry/rep_adiabatic_bug.sio) | examples/chemistry/rep_adiabatic_bug.sio | timeout | timeout | 120.0 / 120.0 |
| [`chemistry/rep_stagnation.sio`](chemistry/rep_stagnation.sio) | examples/chemistry/rep_stagnation.sio | fail rc=1 (E001) | timeout | 4.5 / 120.0 |
| [`chemistry/rep_traj_bug.sio`](chemistry/rep_traj_bug.sio) | examples/chemistry/rep_traj_bug.sio | fail rc=1 (E001) | timeout | 3.7 / 120.0 |
| [`chingon_projective_measurement.sio`](chingon_projective_measurement.sio) | examples/chingon_projective_measurement.sio | fail rc=1 (E035 E008 E004) | pass | 3.5 / 0.6 |
| [`clifford_benchmark.sio`](clifford_benchmark.sio) | clifford_benchmark.sio — Performance comparison of Cl(3,1) implementations | pass | fail rc=1 (E035) | 6.5 / 0.6 |
| [`clifford_demo.sio`](clifford_demo.sio) | clifford_demo.sio — Cl(3,1) spacetime algebra in Sounio | pass | fail rc=1 (E035) | 4.2 / 0.7 |
| [`clinical/ddi_elplus_demo.sio`](clinical/ddi_elplus_demo.sio) | examples/clinical/ddi_elplus_demo.sio | pass | pass | 5.8 / 1.0 |
| [`clinical/pathway_temporal_demo.sio`](clinical/pathway_temporal_demo.sio) | examples/clinical/pathway_temporal_demo.sio | pass | pass | 6.7 / 1.1 |
| [`clinical/pgx_elplus_demo.sio`](clinical/pgx_elplus_demo.sio) | examples/clinical/pgx_elplus_demo.sio | pass | pass | 6.3 / 1.1 |
| [`clinical_curvature_analysis.sio`](clinical_curvature_analysis.sio) | Clinical Curvature Analysis — Example Pipeline | library | library | — |
| [`clinical_trial_epistemic.sio`](clinical_trial_epistemic.sio) | Clinical Trial Epistemic Analysis Pipeline | pass | pass | 5.1 / 0.7 |
| [`closure_lean_test.sio`](closure_lean_test.sio) | closure_lean_test.sio — Sprint 234 Track A gate | fail rc=6 | fail rc=6 | 3.5 / 0.7 |
| [`cocycle_subspace_168.sio`](cocycle_subspace_168.sio) | Conjecture 5 Verification — Subspace Decomposition Validation | pass | pass | 4.0 / 0.5 |
| [`cocycle_subspace_k10.sio`](cocycle_subspace_k10.sio) | Cohomological subspace decomposition at k=10 (2048-ions, dim 1024) | timeout | timeout | 120.0 / 120.0 |
| [`cocycle_subspace_k5.sio`](cocycle_subspace_k5.sio) | Cohomological subspace decomposition at k=5 (trigintaduonions) | pass | pass | 5.1 / 0.6 |
| [`cocycle_subspace_k6.sio`](cocycle_subspace_k6.sio) | Cohomological subspace decomposition at k=6 (chingons, dim 64) | pass | pass | 5.0 / 1.6 |
| [`cocycle_subspace_k7.sio`](cocycle_subspace_k7.sio) | Cohomological subspace decomposition at k=7 (routons, dim 128) | pass | pass | 10.5 / 4.3 |
| [`cocycle_subspace_k8.sio`](cocycle_subspace_k8.sio) | Cohomological subspace decomposition at k=8 (voudons, dim 256) | pass | pass | 15.9 / 12.2 |
| [`cocycle_subspace_k9.sio`](cocycle_subspace_k9.sio) | Cohomological subspace decomposition at k=9 (1024-ions, dim 512) | pass | pass | 112.8 / 109.3 |
| [`cognitive_ossm/cognitive_ossm.sio`](cognitive_ossm/cognitive_ossm.sio) | Canonical cognitive O-SSM sketch for SWOW-EN trajectories. | fail rc=1 (E004 E007 E009) | pass | 3.7 / 0.6 |
| [`cognitive_ossm/export_results.sio`](cognitive_ossm/export_results.sio) | Export helper for bounded canonical Sounio O-SSM parity outputs. | fail rc=1 (E019) | fail rc=1 | 3.7 / 0.7 |
| [`cognitive_ossm/run_ossm_native_reference.sio`](cognitive_ossm/run_ossm_native_reference.sio) | Byte-level Sounio implementation of the CPC 2026 O-SSM reference recurrence. | fail rc=1 (E010) | fail rc=1 (E035) | 3.8 / 0.6 |
| [`cognitive_ossm/run_regimes.sio`](cognitive_ossm/run_regimes.sio) | Bounded canonical Sounio runner for cognitive O-SSM parity outputs. | fail rc=1 (E019 E004 E007) | fail rc=1 (E001 E035) | 3.4 / 0.8 |
| [`collections/bitset_demo.sio`](collections/bitset_demo.sio) | Bit Set Demonstration | fail rc=1 | fail rc=1 (E035 E200 E001) | 4.1 / 0.7 |
| [`collections/heap_demo.sio`](collections/heap_demo.sio) | Binary Heap (Priority Queue) Demonstration | fail rc=1 | fail rc=1 (E035 E001 E200) | 3.7 / 0.8 |
| [`compiler/check/checker_demo.sio`](compiler/check/checker_demo.sio) | Demo for stdlib/compiler/check/checker.sio | fail rc=1 | fail rc=1 (E200) | 3.5 / 0.7 |
| [`compiler/check/env_demo.sio`](compiler/check/env_demo.sio) | Demo for stdlib/compiler/check/env.sio | fail rc=1 | fail rc=1 (E200) | 3.3 / 0.6 |
| [`compiler/check/types_demo.sio`](compiler/check/types_demo.sio) | Demo for stdlib/compiler/check/types.sio | fail rc=1 | fail rc=1 (E200) | 3.4 / 0.6 |
| [`compiler/codegen/bytecode_demo.sio`](compiler/codegen/bytecode_demo.sio) | Demo for stdlib/compiler/codegen/bytecode.sio | fail rc=1 | fail rc=1 (E200) | 3.7 / 0.8 |
| [`compiler/codegen/vm_demo.sio`](compiler/codegen/vm_demo.sio) | Demo for stdlib/compiler/codegen/vm.sio | fail rc=1 | fail rc=1 (E200) | 3.9 / 0.7 |
| [`compiler/lexer/comparison_harness_demo.sio`](compiler/lexer/comparison_harness_demo.sio) | Demo for stdlib/compiler/lexer/comparison_harness.sio | fail rc=1 (E137) | fail rc=1 (E200) | 3.4 / 0.8 |
| [`compiler/lexer/mod_demo.sio`](compiler/lexer/mod_demo.sio) | Demo for stdlib/compiler/lexer/mod.sio | fail rc=1 | fail rc=1 (E200) | 3.8 / 0.6 |
| [`compiler/parser/expr_parse_demo.sio`](compiler/parser/expr_parse_demo.sio) | Demo for stdlib/compiler/parser/expr_parse.sio | pass | pass | 3.5 / 0.6 |
| [`compiler/parser/expr_simple_demo.sio`](compiler/parser/expr_simple_demo.sio) | Demo for stdlib/compiler/parser/expr_simple.sio | fail rc=1 | fail rc=1 (E200) | 3.6 / 0.5 |
| [`compiler/parser/mod_demo.sio`](compiler/parser/mod_demo.sio) | Demo for stdlib/compiler/parser/mod.sio | pass | pass | 3.8 / 0.7 |
| [`compiler/parser/stmt_parse_demo.sio`](compiler/parser/stmt_parse_demo.sio) | Demo for stdlib/compiler/parser/stmt_parse.sio | fail rc=1 | fail rc=1 (E200) | 3.7 / 0.7 |
| [`compiler/types/unify_demo.sio`](compiler/types/unify_demo.sio) | Demo for stdlib/compiler/types/unify.sio | pass | pass | 3.5 / 0.6 |
| [`computational_psychiatry_demo.sio`](computational_psychiatry_demo.sio) | examples/computational_psychiatry_demo.sio | fail rc=1 (E259) | pass | 3.6 / 0.6 |
| [`conjecture3_weak_values.sio`](conjecture3_weak_values.sio) | Conjecture 3 — Computational Proof | pass | pass | 4.5 / 0.8 |
| [`conjecture5_bridge_test.sio`](conjecture5_bridge_test.sio) | THE BRIDGE TEST: Does the Pythagorean decomposition match full PK? | pass | pass | 5.0 / 1.6 |
| [`conjecture5_pbpk_test.sio`](conjecture5_pbpk_test.sio) | CONJECTURE 5 TEST: Pythagorean Prediction via PBPK Simulation | pass | pass | 5.4 / 1.1 |
| [`connectivity/network_metrics_demo.sio`](connectivity/network_metrics_demo.sio) | Demo for stdlib/connectivity/network_metrics.sio | fail rc=1 | fail rc=1 (E200) | 3.4 / 0.8 |
| [`connectivity/phase_demo.sio`](connectivity/phase_demo.sio) | Demo for stdlib/connectivity/phase.sio | fail rc=1 | fail rc=1 (E200) | 4.2 / 0.7 |
| [`connectome_abide_associator.sio`](connectome_abide_associator.sio) | Connectome Associator Field: ASD vs TD Graph Structure | pass | fail rc=1 (E035) | 5.5 / 0.5 |
| [`contest_seizure_detection.sio`](contest_seizure_detection.sio) | Exp 6: Contest<lt;SeizureDetection> | pass | pass | 5.0 / 0.9 |
| [`conversational_ossm/agent_cli.sio`](conversational_ossm/agent_cli.sio) | — | pass | pass | 10.7 / 0.8 |
| [`conversational_ossm/associator_telemetry.sio`](conversational_ossm/associator_telemetry.sio) | Associator Telemetry Module — Structural Instability Monitor | fail rc=1 (E008) | fail rc=1 (E218 E200 E001) | 4.5 / 0.9 |
| [`conversational_ossm/bidirectional_ossm_v0.sio`](conversational_ossm/bidirectional_ossm_v0.sio) | Bidirectional O-SSM v0 — Late Fusion Conversational Engine | crash (signal 11) | fail rc=1 (E200 E001 E006) | 6.3 / 0.7 |
| [`conversational_ossm/o_ssm_conflict.sio`](conversational_ossm/o_ssm_conflict.sio) | — | library | library | — |
| [`conversational_ossm/o_ssm_core.sio`](conversational_ossm/o_ssm_core.sio) | — | library | library | — |
| [`conversational_ossm/o_ssm_memory.sio`](conversational_ossm/o_ssm_memory.sio) | — | library | library | — |
| [`conversational_ossm/o_ssm_router.sio`](conversational_ossm/o_ssm_router.sio) | — | library | library | — |
| [`conversational_ossm/test_syntax.sio`](conversational_ossm/test_syntax.sio) | — | pass | pass | 4.2 / 0.6 |
| [`copy_task_benchmark.sio`](copy_task_benchmark.sio) | Copy Task: O-SSM vs Diagonal — Order-Dependent Long-Range Benchmark | pass | fail rc=1 (E035) | 26.3 / 0.7 |
| [`cosine_theorem_proof.sio`](cosine_theorem_proof.sio) | THEOREM (Cosine Profile of the Arrow Field): | pass | pass | 4.9 / 0.7 |
| [`csv/mod_demo.sio`](csv/mod_demo.sio) | Demo for stdlib/csv/mod.sio | fail rc=1 (E137 E011) | fail rc=1 (E200) | 3.7 / 0.7 |
| [`csv_stats.sio`](csv_stats.sio) | Compute column statistics from inline data | pass | pass | 4.3 / 0.7 |
| [`cybernetic_demo.sio`](cybernetic_demo.sio) | examples/cybernetic_demo.sio | fail rc=1 (E259 E174) | fail rc=1 (E035) | 4.5 / 0.9 |
| [`cyp450_168_demo.sio`](cyp450_168_demo.sio) | cyp450_168_demo.sio | pass | pass | 4.5 / 0.5 |
| [`darwin_atlas/lib.sio`](darwin_atlas/lib.sio) | Darwin Atlas Kernels - Pipeline Demo (single-file, module-free) | fail rc=1 (E019) | fail rc=1 (E035 E218) | 4.2 / 0.7 |
| [`darwin_atlas/test_runner.sio`](darwin_atlas/test_runner.sio) | Test Runner for Darwin Atlas Kernels | pass | fail rc=1 (E035) | 4.0 / 0.7 |
| [`darwin_atlas/verify_knowledge.sio`](darwin_atlas/verify_knowledge.sio) | Atlas Epistemic Knowledge Verifier | library | library | — |
| [`darwin_pbpk/rodgers_rowland_demo.sio`](darwin_pbpk/rodgers_rowland_demo.sio) | Demo for stdlib/darwin_pbpk/core/rodgers_rowland.sio | fail rc=1 (E137) | fail rc=1 (E200 E035) | 4.5 / 0.7 |
| [`darwin_pbpk/simulation_demo.sio`](darwin_pbpk/simulation_demo.sio) | Demo for stdlib/darwin_pbpk/simulation.sio | fail rc=1 | fail rc=1 (E035 E200) | 3.8 / 0.7 |
| [`darwin_pbpk/tsit5_pbpk14_demo.sio`](darwin_pbpk/tsit5_pbpk14_demo.sio) | Demo for stdlib/darwin_pbpk/tsit5_pbpk14.sio | fail rc=1 (E137) | fail rc=1 (E200) | 4.3 / 0.6 |
| [`data/biodiversity_inequality_showcase.sio`](data/biodiversity_inequality_showcase.sio) | Biodiversity + inequality showcase: a per-site survey profile over data::bigframe, | crash (signal 11) | pass | 13.3 / 3.2 |
| [`data/clinical_lab_qc_showcase.sio`](data/clinical_lab_qc_showcase.sio) | Clinical-lab QC showcase: per-cohort robust profiling of a fixed-precision biomarker over | crash (signal 11) | pass | 11.5 / 3.4 |
| [`data/dataframe_groupby_workflow.sio`](data/dataframe_groupby_workflow.sio) | Example: a real group-by + join analytics workflow on CSV files, showing all three join kinds. | fail rc=1 (E035) | crash (signal 11) | 4.6 / 0.7 |
| [`data/dataframe_workflow.sio`](data/dataframe_workflow.sio) | Example: an end-to-end DataFrame workflow on a real CSV file. | pass | pass | 6.4 / 0.9 |
| [`data/frame_demo.sio`](data/frame_demo.sio) | Demo for stdlib/data/frame.sio | fail rc=1 (E137 E001) | fail rc=1 (E200 E001) | 4.6 / 0.8 |
| [`data/grouped_analytics_showcase.sio`](data/grouped_analytics_showcase.sio) | Grouped-analytics showcase: a per-sensor measurement-QC pipeline over data::bigframe, | crash (signal 4) | pass | 14.4 / 3.7 |
| [`data/io_demo.sio`](data/io_demo.sio) | Demo for stdlib/data/io.sio | fail rc=1 (E137 E011) | fail rc=1 (E200) | 3.7 / 0.7 |
| [`data/measurement_qc_workflow.sio`](data/measurement_qc_workflow.sio) | Capstone: an end-to-end measurement-QC workflow that composes the Sounio "measurement DataFrame" | pass | pass | 8.2 / 0.9 |
| [`data/mod_demo.sio`](data/mod_demo.sio) | Demo for stdlib/data/mod.sio | pass | pass | 4.1 / 0.6 |
| [`data/ops_demo.sio`](data/ops_demo.sio) | Demo for stdlib/data/ops.sio | fail rc=1 (E137 E001 E011) | fail rc=1 (E200 E001) | 3.7 / 0.5 |
| [`data/series_demo.sio`](data/series_demo.sio) | Demo for stdlib/data/series.sio | fail rc=1 (E137 E015) | fail rc=1 (E200) | 4.0 / 0.6 |
| [`deep_psychiatry_demo.sio`](deep_psychiatry_demo.sio) | examples/deep_psychiatry_demo.sio | fail rc=1 (E010 E259) | fail rc=1 (E006) | 4.1 / 0.6 |
| [`dissertation_168_polypharmacy.sio`](dissertation_168_polypharmacy.sio) | examples/dissertation_168_polypharmacy.sio | fail rc=1 (E259) | pass | 3.9 / 0.7 |
| [`dissertation_demo.sio`](dissertation_demo.sio) | Dissertation Demo: Rapamycin PBPK-14 with Epistemic Computing | pass | pass | 7.5 / 1.4 |
| [`dissertation_hybrid_demo.sio`](dissertation_hybrid_demo.sio) | examples/dissertation_hybrid_demo.sio | pass | pass | 6.0 / 0.6 |
| [`dissertation_interactive.sio`](dissertation_interactive.sio) | Interactive PBPK Demo — HTML with Canvas2D | pass | pass | 7.4 / 1.5 |
| [`dissertation_olanzapine_demo.sio`](dissertation_olanzapine_demo.sio) | examples/dissertation_olanzapine_demo.sio | pass | pass | 7.4 / 0.8 |
| [`dissertation_oral_pd_demo.sio`](dissertation_oral_pd_demo.sio) | Oral rapamycin + BBB + Hill-PD demo (dissertation extensions) | fail rc=1 (E259) | pass | 3.8 / 1.9 |
| [`dissertation_pbpk_rapamycin.sio`](dissertation_pbpk_rapamycin.sio) | Door 6: Dissertation PBPK Demo — Rapamycin with GUM Uncertainty | pass | pass | 5.3 / 0.8 |
| [`dissertation_pgx_compile_gate_demo.sio`](dissertation_pgx_compile_gate_demo.sio) | examples/dissertation_pgx_compile_gate_demo.sio | pass | pass | 9.2 / 1.0 |
| [`dissertation_plot.sio`](dissertation_plot.sio) | Dissertation Plot: Rapamycin PBPK-14 SVG output | pass | pass | 8.0 / 1.2 |
| [`dissertation_pop_pbpk_pd_demo.sio`](dissertation_pop_pbpk_pd_demo.sio) | examples/dissertation_pop_pbpk_pd_demo.sio | pass | pass | 8.6 / 1.0 |
| [`dissertation_scenario_gate_demo.sio`](dissertation_scenario_gate_demo.sio) | Scenario-level VoI-weighted confidence gate demo. | fail rc=1 (E137 E259) | pass | 5.1 / 48.9 |
| [`dissertation_steady_state_demo.sio`](dissertation_steady_state_demo.sio) | Multi-dose rapamycin steady-state demo. | fail rc=1 (E259 E008 E137) | pass | 4.6 / 5.4 |
| [`dissertation_steady_state_fullvd_demo.sio`](dissertation_steady_state_fullvd_demo.sio) | Multi-dose steady-state using the clinical full-Vd rapamycin model. | fail rc=1 (E259 E008 E137) | pass | 4.7 / 12.2 |
| [`dissertation_tirzepatide_demo.sio`](dissertation_tirzepatide_demo.sio) | examples/dissertation_tirzepatide_demo.sio | pass | pass | 5.5 / 0.9 |
| [`dissertation_vancomycin_demo.sio`](dissertation_vancomycin_demo.sio) | examples/dissertation_vancomycin_demo.sio | pass | pass | 5.3 / 0.6 |
| [`door_e_alpha_mechanism.sio`](door_e_alpha_mechanism.sio) | Door E: α mechanism — hidden state separation + associator ratio | fail rc=1 (E035) | fail rc=1 (E035) | 4.9 / 0.8 |
| [`door_f_assoc_preictal_chb02.sio`](door_f_assoc_preictal_chb02.sio) | Door F: Pre-ictal sweep — MSE vs Associator Norm | fail rc=1 | pass | 14.8 / 6.9 |
| [`door_f_assoc_preictal_chb03.sio`](door_f_assoc_preictal_chb03.sio) | Door F: Pre-ictal sweep — MSE vs Associator Norm | fail rc=1 | pass | 15.5 / 8.5 |
| [`door_f_assoc_preictal_chb05.sio`](door_f_assoc_preictal_chb05.sio) | Door F: Pre-ictal sweep — MSE vs Associator Norm | fail rc=1 | pass | 15.5 / 9.7 |
| [`door_f_assoc_preictal_chb06.sio`](door_f_assoc_preictal_chb06.sio) | Door F: Pre-ictal sweep — MSE vs Associator Norm | fail rc=1 | pass | 15.1 / 6.1 |
| [`door_f_assoc_preictal_chb10.sio`](door_f_assoc_preictal_chb10.sio) | Door F: Pre-ictal sweep — MSE vs Associator Norm | fail rc=1 | pass | 17.2 / 8.2 |
| [`drug_cascade_168_demo.sio`](drug_cascade_168_demo.sio) | drug_cascade_168_demo.sio | pass | pass | 5.0 / 0.7 |
| [`eeg_hessian_temporal.sio`](eeg_hessian_temporal.sio) | EEG O-SSM Hessian: Temporal Non-Associativity in Motor Imagery | fail rc=1 (E011) | timeout | 4.3 / 120.0 |
| [`effect_demo.sio`](effect_demo.sio) | — | pass | pass | 4.0 / 0.6 |
| [`effects/basic_handler_continuation.sio`](effects/basic_handler_continuation.sio) | Test basic effect handler with real continuations | pass | pass | 4.2 / 0.8 |
| [`effects/comprehensive_effects.sio`](effects/comprehensive_effects.sio) | Comprehensive end-to-end test for effect handlers with continuations | pass | pass | 4.6 / 0.8 |
| [`effects_simple.sio`](effects_simple.sio) | — | pass | pass | 4.2 / 0.6 |
| [`eisa_cancellation_kernel.sio`](eisa_cancellation_kernel.sio) | EISA E5 showcase: catastrophic-cancellation kernel through the full stack | pass | pass | 20.2 / 1.9 |
| [`ekan_ablation_suite.sio`](ekan_ablation_suite.sio) | E-KAN Ablation Suite — Depth, Noise, Knot Count | pass | fail rc=1 (E035) | 7.7 / 0.6 |
| [`ekan_advanced_ablation.sio`](ekan_advanced_ablation.sio) | E-KAN Advanced Ablation: Width, σ Threshold, Heteroscedastic, OOD | pass | pass | 8.5 / 11.9 |
| [`ekan_concrete_uci.sio`](ekan_concrete_uci.sio) | E-KAN on UCI Concrete Compressive Strength — Paper B Experiment E1 | fail rc=1 (E035) | fail rc=1 (E035) | 4.8 / 0.7 |
| [`ekan_energy_uci.sio`](ekan_energy_uci.sio) | E-KAN on UCI Energy Efficiency — Paper B Experiment E9 | fail rc=1 (E035) | fail rc=1 (E035) | 4.3 / 0.6 |
| [`ekan_final_experiments.sio`](ekan_final_experiments.sio) | E-KAN Final Experiments: Adversarial, Multi-Output, Feature Importance | pass | fail rc=1 (E035) | 8.0 / 0.9 |
| [`ekan_gum_vs_montecarlo.sio`](ekan_gum_vs_montecarlo.sio) | GUM vs Monte Carlo Validation — Epistemic KAN (4→6→2) | pass | pass | 7.5 / 4.5 |
| [`ekan_knowledge.sio`](ekan_knowledge.sio) | Epistemic KAN with Knowledge (EK) confidence tracking | pass | pass | 5.8 / 0.6 |
| [`ekan_regression_benchmark.sio`](ekan_regression_benchmark.sio) | E-KAN Regression Benchmark — sin(x) + noise with calibration analysis | fail rc=1 (E035) | fail rc=1 (E035) | 4.6 / 0.9 |
| [`ekan_regression_friedman.sio`](ekan_regression_friedman.sio) | E-KAN Regression Benchmark #2 — Friedman-1 Function (multivariate) | fail rc=1 (E035) | fail rc=1 (E035) | 4.6 / 0.8 |
| [`ekan_second_order_gum.sio`](ekan_second_order_gum.sio) | Second-Order GUM for E-KAN: Hessian Correction | fail rc=1 (E035) | fail rc=1 (E035) | 4.9 / 0.9 |
| [`ekan_uq_baselines.sio`](ekan_uq_baselines.sio) | UQ Baseline Comparisons for Paper B (E-KAN) | fail rc=1 (E035) | fail rc=1 (E035) | 5.6 / 0.7 |
| [`ekan_wine_uci.sio`](ekan_wine_uci.sio) | E-KAN on UCI Wine Quality Red — Paper B Experiment E2 | fail rc=1 (E035) | fail rc=1 (E035) | 5.2 / 0.9 |
| [`epistemic/affine_nonassoc_demo.sio`](epistemic/affine_nonassoc_demo.sio) | examples/epistemic/affine_nonassoc_demo.sio | pass | pass | 8.1 / 0.7 |
| [`epistemic/affine_nonassoc_n4_demo.sio`](epistemic/affine_nonassoc_n4_demo.sio) | examples/epistemic/affine_nonassoc_n4_demo.sio | pass | pass | 7.4 / 0.8 |
| [`epistemic/gum_measurement_chain.sio`](epistemic/gum_measurement_chain.sio) | GUM measurement chain using the stdlib epistemic::gum module. | pass | pass | 6.1 / 0.8 |
| [`epistemic/gum_to_csv.sio`](epistemic/gum_to_csv.sio) | GUM measurement -> byte-exact CSV row (Data & Science I/O, Trilha A). | pass | pass | 6.7 / 0.8 |
| [`epistemic/gum_to_json.sio`](epistemic/gum_to_json.sio) | GUM measurement -> byte-exact JSON object (Data & Science I/O, Trilha A). | pass | pass | 6.8 / 0.8 |
| [`epistemic/knowledge_units.sio`](epistemic/knowledge_units.sio) | Epistemic Knowledge Units — D3 Min-Propagation Semantics | pass | fail rc=1 (E035) | 6.4 / 0.8 |
| [`epistemic/ledger_demo.sio`](epistemic/ledger_demo.sio) | Demo for stdlib/epistemic/ledger.sio | fail rc=1 | fail rc=1 (E200) | 5.3 / 0.8 |
| [`epistemic/pce_complete_demo.sio`](epistemic/pce_complete_demo.sio) | Complete Polynomial Chaos Expansion (PCE) Demonstration | fail rc=1 | fail rc=1 (E001 E006 E035) | 4.8 / 1.0 |
| [`epistemic/pce_demo.sio`](epistemic/pce_demo.sio) | Demonstrates Polynomial Chaos Expansion (PCE) for uncertainty quantification | fail rc=1 | fail rc=1 (E001 E006 E035) | 4.6 / 0.8 |
| [`epistemic/pce_test.sio`](epistemic/pce_test.sio) | Simple PCE test to validate the implementation | fail rc=1 | fail rc=1 (E001 E006) | 4.9 / 1.1 |
| [`epistemic/pk_curve_gum_to_csv.sio`](epistemic/pk_curve_gum_to_csv.sio) | Multi-row PK concentration-time table with per-point GUM uncertainty -> | pass | pass | 6.8 / 0.9 |
| [`epistemic/pk_example.sio`](epistemic/pk_example.sio) | Pharmacokinetic Example with Rigorous Epistemic Semantics | pass | fail rc=1 (E035) | 6.3 / 1.0 |
| [`epistemic/prov_elplus_demo.sio`](epistemic/prov_elplus_demo.sio) | PROV-DM / SLSA provenance derivation closure — the PROV-CONSTRAINTS | pass | fail rc=1 (E035) | 7.1 / 0.8 |
| [`epistemic/rk4_correlated_uncertainty.sio`](epistemic/rk4_correlated_uncertainty.sio) | rk4_correlated_uncertainty.sio | pass | pass | 8.0 / 1.6 |
| [`epistemic/rupture_claims_verified.sio`](epistemic/rupture_claims_verified.sio) | Bound claims manifest — rung R1 of the self-falsifying compilation line. | pass | pass | 5.0 / 0.9 |
| [`epistemic/traceability_elplus_demo.sio`](epistemic/traceability_elplus_demo.sio) | Metrological traceability as EL+ role composition — the VIM3 "unbroken | pass | pass | 6.7 / 1.2 |
| [`epistemic_bmi.sio`](epistemic_bmi.sio) | Epistemic BMI example with compiler-native uncertainty propagation. | fail rc=1 (E236 E245 E230) | fail rc=1 | 4.9 / 0.9 |
| [`epistemic_classifier.sio`](epistemic_classifier.sio) | Epistemic MLP Classifier — Drug A vs Drug B from PK Features | fail rc=3 | fail rc=3 | 7.0 / 0.6 |
| [`epistemic_ddi_simulator.sio`](epistemic_ddi_simulator.sio) | Epistemic DDI (Drug-Drug Interaction) Simulator | pass | pass | 5.9 / 0.7 |
| [`epistemic_dempster_shafer.sio`](epistemic_dempster_shafer.sio) | Dempster-Shafer evidence combination with type-level guarantees | fail rc=1 | fail rc=1 (E200) | 5.2 / 1.0 |
| [`epistemic_fo_second_order/fo_pk_exposure_driver.sio`](epistemic_fo_second_order/fo_pk_exposure_driver.sio) | Scientific FO driver — oral steady-state exposure under Madaros FO GUM. | fail rc=1 (E245) | fail rc=1 (E200 E001) | 5.3 / 0.6 |
| [`epistemic_fo_second_order/fo_pk_exposure_import_driver.sio`](epistemic_fo_second_order/fo_pk_exposure_import_driver.sio) | Scientific FO driver — multi-mod path via stdlib epistemic::fo. | fail rc=1 (E245) | fail rc=1 (E200 E001) | 5.2 / 1.0 |
| [`epistemic_fo_second_order/fo_pk_import_auc_thalf_driver.sio`](epistemic_fo_second_order/fo_pk_import_auc_thalf_driver.sio) | R5 import ↔ method parity — multi-mod epistemic::fo vs Pk methods. | pass | fail rc=1 (E200) | 5.7 / 1.0 |
| [`epistemic_fo_second_order/fo_pk_import_auct_driver.sio`](epistemic_fo_second_order/fo_pk_import_auct_driver.sio) | R12b — multi-mod fo_auc_tau / fo_css_tau / fo_auc ↔ method. | pass | fail rc=1 (E200) | 5.3 / 0.8 |
| [`epistemic_fo_second_order/fo_pk_import_cmax_driver.sio`](epistemic_fo_second_order/fo_pk_import_cmax_driver.sio) | R7b — multi-mod fo_cmax / fo_cmin / fo_ptf freezes. | pass | pass | 6.0 / 0.7 |
| [`epistemic_fo_second_order/fo_pk_import_fss_driver.sio`](epistemic_fo_second_order/fo_pk_import_fss_driver.sio) | R8b — multi-mod fo_fss / fo_n90 ↔ method / peel. | pass | pass | 5.4 / 0.7 |
| [`epistemic_fo_second_order/fo_pk_import_ld_driver.sio`](epistemic_fo_second_order/fo_pk_import_ld_driver.sio) | R11b — multi-mod fo_ld / fo_fe freezes. | pass | pass | 5.4 / 0.9 |
| [`epistemic_fo_second_order/fo_pk_import_method_driver.sio`](epistemic_fo_second_order/fo_pk_import_method_driver.sio) | Import + method FO parity — multi-mod stdlib helpers vs Pk struct methods. | pass | fail rc=1 (E200) | 5.9 / 1.0 |
| [`epistemic_fo_second_order/fo_pk_import_mrt_driver.sio`](epistemic_fo_second_order/fo_pk_import_mrt_driver.sio) | R10b — multi-mod fo_mrt / fo_t90 ↔ method / peel. | pass | pass | 5.3 / 0.7 |
| [`epistemic_fo_second_order/fo_pk_import_ptr_driver.sio`](epistemic_fo_second_order/fo_pk_import_ptr_driver.sio) | R9b — multi-mod fo_ptr / fo_dof ↔ method / peel. | pass | pass | 5.6 / 0.8 |
| [`epistemic_fo_second_order/fo_pk_import_rac_driver.sio`](epistemic_fo_second_order/fo_pk_import_rac_driver.sio) | R6b — multi-mod fo_rac / fo_frac_rem ↔ method / peel. | pass | fail rc=1 (E200) | 6.2 / 0.9 |
| [`epistemic_fo_second_order/fo_pk_struct_auc_thalf_driver.sio`](epistemic_fo_second_order/fo_pk_struct_auc_thalf_driver.sio) | R5 — Oral AUC and elimination half-life FO (Pk methods + surfaces). | pass | fail rc=1 (E200) | 5.8 / 1.0 |
| [`epistemic_fo_second_order/fo_pk_struct_auct_driver.sio`](epistemic_fo_second_order/fo_pk_struct_auct_driver.sio) | R12 — Steady-state AUC over dosing interval AUC_τ = F·Dose/CL = Css·τ. | pass | fail rc=1 (E200) | 5.4 / 0.7 |
| [`epistemic_fo_second_order/fo_pk_struct_cmax_driver.sio`](epistemic_fo_second_order/fo_pk_struct_cmax_driver.sio) | R7 — Multi-dose peak / trough Css + peak–trough fluctuation FO. | pass | fail rc=1 (E200) | 5.4 / 0.9 |
| [`epistemic_fo_second_order/fo_pk_struct_fss_driver.sio`](epistemic_fo_second_order/fo_pk_struct_fss_driver.sio) | R8 — Fraction of steady state after n doses + doses to 90% SS. | pass | fail rc=1 (E200) | 5.1 / 0.8 |
| [`epistemic_fo_second_order/fo_pk_struct_ld_driver.sio`](epistemic_fo_second_order/fo_pk_struct_ld_driver.sio) | R11 — Loading dose LD = Dose·Rac + fraction eliminated per interval. | pass | fail rc=1 (E200) | 5.9 / 0.7 |
| [`epistemic_fo_second_order/fo_pk_struct_method_driver.sio`](epistemic_fo_second_order/fo_pk_struct_method_driver.sio) | Dissertation-shaped PK science driver — method FO stack after Madaros FO 42/42. | crash (signal 4) | fail rc=1 (E200) | 5.8 / 1.0 |
| [`epistemic_fo_second_order/fo_pk_struct_mrt_driver.sio`](epistemic_fo_second_order/fo_pk_struct_mrt_driver.sio) | R10 — Mean residence time + time to 90% steady state (hours). | pass | fail rc=1 (E200) | 4.5 / 0.8 |
| [`epistemic_fo_second_order/fo_pk_struct_multidose_driver.sio`](epistemic_fo_second_order/fo_pk_struct_multidose_driver.sio) | Multi-dose / dosing-interval FO series — Pk-method science companion. | crash (signal 4) | fail rc=1 (E200) | 5.0 / 0.7 |
| [`epistemic_fo_second_order/fo_pk_struct_ptr_driver.sio`](epistemic_fo_second_order/fo_pk_struct_ptr_driver.sio) | R9 — Peak–trough ratio + degree of fluctuation FO. | pass | fail rc=1 (E200) | 5.4 / 0.8 |
| [`epistemic_fo_second_order/fo_pk_struct_rac_driver.sio`](epistemic_fo_second_order/fo_pk_struct_rac_driver.sio) | R6 — Multi-dose accumulation ratio + residual fraction FO. | pass | fail rc=1 (E200) | 5.7 / 0.8 |
| [`epistemic_fo_second_order/fo_pk_struct_rho_tau_driver.sio`](epistemic_fo_second_order/fo_pk_struct_rho_tau_driver.sio) | Companion science driver: exposure ρ-sweep + Css with τ uncertainty. | pass | fail rc=1 (E200) | 4.1 / 0.7 |
| [`epistemic_gpu_pipeline.sio`](epistemic_gpu_pipeline.sio) | Example: Full Epistemic GPU Pipeline — GUM uncertainty through multi-backend GPU computation | fail rc=1 (E070) | fail rc=1 (E070) | 4.4 / 0.9 |
| [`epistemic_kan.sio`](epistemic_kan.sio) | Epistemic Kolmogorov-Arnold Network (E-KAN) | pass | pass | 6.3 / 0.8 |
| [`epistemic_kan_fixed_point.sio`](epistemic_kan_fixed_point.sio) | Fixed-point E-KAN witness. | pass | pass | 4.7 / 0.7 |
| [`epistemic_kan_trained.sio`](epistemic_kan_trained.sio) | Epistemic Kolmogorov-Arnold Network — Trained (E-KAN-T) | fail rc=1 | pass | 5.0 / 1.0 |
| [`epistemic_lm.sio`](epistemic_lm.sio) | First Epistemic Language Model | pass | pass | 10.4 / 16.9 |
| [`epistemic_lm_100k.sio`](epistemic_lm_100k.sio) | 100K-Parameter Epistemic Language Model — 8 Layers | pass | pass | 9.8 / 3.9 |
| [`epistemic_lm_32k.sio`](epistemic_lm_32k.sio) | 32K-Parameter Epistemic Language Model | pass | pass | 6.9 / 2.6 |
| [`epistemic_lm_backprop.sio`](epistemic_lm_backprop.sio) | Epistemic LM with Proper Backpropagation | pass | pass | 7.2 / 1.7 |
| [`epistemic_lm_fileio.sio`](epistemic_lm_fileio.sio) | Epistemic LM with File I/O — First Sounio Model Trained on Real Data | fail rc=1 | fail rc=1 | 10.4 / 11.1 |
| [`epistemic_lm_scaled.sio`](epistemic_lm_scaled.sio) | Scaled Epistemic Language Model — 2-Layer O-SSM, 16K+ Parameters | pass | pass | 12.6 / 16.1 |
| [`epistemic_mcts.sio`](epistemic_mcts.sio) | examples/epistemic_mcts.sio | pass | pass | 6.3 / 1.0 |
| [`epistemic_mcts_full.sio`](epistemic_mcts_full.sio) | examples/epistemic_mcts_full.sio | pass | pass | 7.1 / 0.8 |
| [`epistemic_preictal_workflow.sio`](epistemic_preictal_workflow.sio) | Epistemic Workflow Prototype | fail rc=1 | fail rc=1 (E224) | 6.5 / 0.9 |
| [`epistemic_propagation.sio`](epistemic_propagation.sio) | RSS (Root Sum of Squares) uncertainty propagation with SMT verification | fail rc=1 | pass | 5.7 / 0.9 |
| [`epistemic_quantum_vqe.sio`](epistemic_quantum_vqe.sio) | Epistemic Quantum VQE (1-qubit H₂ approximation) | pass | fail rc=1 (E035) | 7.4 / 0.7 |
| [`epistemic_refinements.sio`](epistemic_refinements.sio) | Epistemic refinement types with SMT-verified bounds | fail rc=1 | fail rc=1 (E200) | 5.8 / 0.8 |
| [`epistemic_smoke_native.sio`](epistemic_smoke_native.sio) | Sprint 131: Epistemic computing smoke test — native x86-64 ELF target | fail rc=1 (E245 E012) | pass | 5.9 / 0.8 |
| [`epistemic_transformer.sio`](epistemic_transformer.sio) | Epistemic Transformer Block (seq=4, d=16, 2 heads, d_k=8) | pass | pass | 9.2 / 0.9 |
| [`epistemic_viz_demo.sio`](epistemic_viz_demo.sio) | examples/epistemic_viz_demo.sio | fail rc=1 | pass | 5.8 / 3.3 |
| [`epistemic_witness_minimal.sio`](epistemic_witness_minimal.sio) | Epistemic witness: minimal program exercising Knowledge<lt;T> as a native type annotation. | pass | pass | 5.5 / 0.9 |
| [`equivalence_theory/lean_obligation_demo.sio`](equivalence_theory/lean_obligation_demo.sio) | Demo for `madaros emit-lean-obligations`. Each function below is a distinct | library | library | — |
| [`erdos/168_c5_flip.sio`](erdos/168_c5_flip.sio) | examples/erdos/168_c5_flip.sio | fail rc=1 (E137) | fail rc=1 (E200) | 6.4 / 1.6 |
| [`erdos/168_c5_flip_loose.sio`](erdos/168_c5_flip_loose.sio) | examples/erdos/168_c5_flip_loose.sio | fail rc=1 (E137) | fail rc=1 (E200) | 6.6 / 1.6 |
| [`erdos/168_c_chi3_search.sio`](erdos/168_c_chi3_search.sio) | examples/erdos/168_c_chi3_search.sio | fail rc=1 (E137) | fail rc=1 (E200 E035) | 7.3 / 1.7 |
| [`erdos/168_chromatic_flip.sio`](erdos/168_chromatic_flip.sio) | examples/erdos/168_chromatic_flip.sio | fail rc=1 (E137) | fail rc=1 (E200) | 6.2 / 1.7 |
| [`erdos/168_cross_half_flip.sio`](erdos/168_cross_half_flip.sio) | examples/erdos/168_cross_half_flip.sio | fail rc=1 (E137) | fail rc=1 (E200) | 6.3 / 1.6 |
| [`erdos/168_edge_map.sio`](erdos/168_edge_map.sio) | examples/erdos/168_edge_map.sio | fail rc=1 (E137) | fail rc=1 (E200) | 5.4 / 1.6 |
| [`erdos/168_edge_map3.sio`](erdos/168_edge_map3.sio) | examples/erdos/168_edge_map3.sio | fail rc=1 (E137) | fail rc=1 (E200) | 5.5 / 1.3 |
| [`erdos/168_edge_map3_signed.sio`](erdos/168_edge_map3_signed.sio) | examples/erdos/168_edge_map3_signed.sio | fail rc=1 (E137) | fail rc=1 (E200) | 5.3 / 0.8 |
| [`erdos/168_k4_full_check.sio`](erdos/168_k4_full_check.sio) | examples/erdos/168_k4_full_check.sio | fail rc=1 (E137) | fail rc=1 (E200) | 5.2 / 1.0 |
| [`erdos/168_k6_escape.sio`](erdos/168_k6_escape.sio) | examples/erdos/168_k6_escape.sio | fail rc=1 (E137) | fail rc=1 (E200) | 5.5 / 1.0 |
| [`erdos/168_kgraph_coloring_test.sio`](erdos/168_kgraph_coloring_test.sio) | examples/erdos/168_kgraph_coloring_test.sio | fail rc=1 (E137) | fail rc=1 (E200) | 5.6 / 0.9 |
| [`erdos/168_moser_a.sio`](erdos/168_moser_a.sio) | examples/erdos/168_moser_a.sio | pass | pass | 11.0 / 1.3 |
| [`erdos/168_orbit_chi.sio`](erdos/168_orbit_chi.sio) | examples/erdos/168_orbit_chi.sio | fail rc=1 (E137) | fail rc=1 (E035 E200) | 5.8 / 1.2 |
| [`erdos/168_orbit_chi6_proof.sio`](erdos/168_orbit_chi6_proof.sio) | examples/erdos/168_orbit_chi6_proof.sio | fail rc=1 (E137) | fail rc=1 (E035 E200) | 5.6 / 1.2 |
| [`erdos/168_orbit_exact_chi.sio`](erdos/168_orbit_exact_chi.sio) | examples/erdos/168_orbit_exact_chi.sio | fail rc=1 (E137) | fail rc=1 (E035 E200) | 5.3 / 1.0 |
| [`erdos/168_orbit_zd_pairs.sio`](erdos/168_orbit_zd_pairs.sio) | examples/erdos/168_orbit_zd_pairs.sio | fail rc=1 (E137) | fail rc=1 (E035 E200) | 5.8 / 1.3 |
| [`erdos/168_regime_a1.sio`](erdos/168_regime_a1.sio) | examples/erdos/168_regime_a1.sio | fail rc=1 (E137) | fail rc=1 (E200) | 5.7 / 1.3 |
| [`erdos/168_regime_chromatic_probe.sio`](erdos/168_regime_chromatic_probe.sio) | examples/erdos/168_regime_chromatic_probe.sio | pass | pass | 11.6 / 1.2 |
| [`erdos/168_zd_regime_a2.sio`](erdos/168_zd_regime_a2.sio) | examples/erdos/168_zd_regime_a2.sio | fail rc=1 (E137) | fail rc=1 (E200) | 5.9 / 1.2 |
| [`erdos/cdcl_fast.sio`](erdos/cdcl_fast.sio) | examples/erdos/cdcl_fast.sio (GENERATED gen_solver.py php) | pass | pass | 10.3 / 2.6 |
| [`erdos/cdcl_proof.sio`](erdos/cdcl_proof.sio) | examples/erdos/cdcl_proof.sio | timeout | timeout | 120.0 / 120.0 |
| [`erdos/degrey_chi5.sio`](erdos/degrey_chi5.sio) | examples/erdos/degrey_chi5.sio  (GENERATED by examples/erdos/data/degrey/gen_sio.py) | timeout | timeout | 120.0 / 120.0 |
| [`erdos/degrey_chi5_fast.sio`](erdos/degrey_chi5_fast.sio) | examples/erdos/degrey_chi5_fast.sio (GENERATED gen_solver.py degrey) | timeout | timeout | 120.0 / 123.0 |
| [`erdos/degrey_fieldtower.sio`](erdos/degrey_fieldtower.sio) | examples/erdos/degrey_fieldtower.sio | pass | pass | 5.5 / 1.0 |
| [`erdos/degrey_fragment_q3q11.sio`](erdos/degrey_fragment_q3q11.sio) | examples/erdos/degrey_fragment_q3q11.sio | pass | pass | 10.7 / 1.0 |
| [`erdos/degrey_geometry.sio`](erdos/degrey_geometry.sio) | examples/erdos/degrey_geometry.sio | pass | pass | 6.3 / 0.8 |
| [`erdos/degrey_q3q11_spindle.sio`](erdos/degrey_q3q11_spindle.sio) | examples/erdos/degrey_q3q11_spindle.sio | fail rc=1 (E035) | pass | 4.7 / 0.9 |
| [`erdos/dpll_scale_wall.sio`](erdos/dpll_scale_wall.sio) | examples/erdos/dpll_scale_wall.sio | pass | pass | 29.7 / 18.5 |
| [`erdos/erdos90_cubic_tower_base.sio`](erdos/erdos90_cubic_tower_base.sio) | examples/erdos/erdos90_cubic_tower_base.sio | pass | pass | 5.7 / 1.0 |
| [`erdos/erdos90_repcount_engine.sio`](erdos/erdos90_repcount_engine.sio) | examples/erdos/erdos90_repcount_engine.sio | pass | pass | 4.9 / 0.9 |
| [`erdos/moser_zd_probe.sio`](erdos/moser_zd_probe.sio) | examples/erdos/moser_zd_probe.sio | timeout | timeout | 120.0 / 120.0 |
| [`erdos/native_sat_scale_demo.sio`](erdos/native_sat_scale_demo.sio) | examples/erdos/native_sat_scale_demo.sio | pass | pass | 10.0 / 1.9 |
| [`erdos/nsat_smoke.sio`](erdos/nsat_smoke.sio) | examples/erdos/nsat_smoke.sio | fail rc=1 (E137) | fail rc=1 (E200) | 5.6 / 1.3 |
| [`erdos/reproducer_madaros_codegen_2026-06-16g.sio`](erdos/reproducer_madaros_codegen_2026-06-16g.sio) | — | pass | pass | 5.2 / 0.7 |
| [`erdos/sat_proof_kernel.sio`](erdos/sat_proof_kernel.sio) | examples/erdos/sat_proof_kernel.sio | pass | pass | 5.5 / 0.7 |
| [`erdos/souc_sat.sio`](erdos/souc_sat.sio) | examples/erdos/souc_sat.sio | pass | pass | 12.4 / 1.9 |
| [`erdos/spindle_proof_cert.sio`](erdos/spindle_proof_cert.sio) | examples/erdos/spindle_proof_cert.sio | pass | pass | 6.2 / 0.8 |
| [`exit42.sio`](exit42.sio) | — | fail rc=42 | fail rc=42 | 4.8 / 0.7 |
| [`fibonacci.sio`](fibonacci.sio) | Fibonacci sequence example | pass | pass | 4.9 / 0.8 |
| [`financial_risk_epistemic.sio`](financial_risk_epistemic.sio) | Financial Risk — Epistemic Uncertainty Pipeline | pass | pass | 6.9 / 0.9 |
| [`flow_viz/main.sio`](flow_viz/main.sio) | examples/flow_viz/main.sio — 2D vector field streamlines demo | timeout | timeout | 120.0 / 120.0 |
| [`fractal/curvature_demo.sio`](fractal/curvature_demo.sio) | Demo for stdlib/fractal/curvature.sio | fail rc=1 (E137 E015 E011) | fail rc=1 (E200) | 4.5 / 0.8 |
| [`fractal/dimension_demo.sio`](fractal/dimension_demo.sio) | Demo for stdlib/fractal/dimension.sio | fail rc=1 (E137 E011) | fail rc=1 (E200) | 4.5 / 0.8 |
| [`fractal/entropy_demo.sio`](fractal/entropy_demo.sio) | Demo for stdlib/fractal/entropy.sio | fail rc=1 (E137 E011) | fail rc=1 (E200) | 4.5 / 0.8 |
| [`fractal/gpu/box_counting_demo.sio`](fractal/gpu/box_counting_demo.sio) | Demo for stdlib/fractal/gpu/box_counting.sio | fail rc=1 | fail rc=1 (E200) | 3.7 / 0.7 |
| [`fractal/gpu/mod_demo.sio`](fractal/gpu/mod_demo.sio) | Demo for stdlib/fractal/gpu/mod.sio | fail rc=1 (E137) | fail rc=1 (E200) | 4.4 / 0.7 |
| [`fractal/kec_demo.sio`](fractal/kec_demo.sio) | Demo for stdlib/fractal/kec.sio | fail rc=1 (E137 E011 E015) | fail rc=1 (E200 E035) | 4.1 / 0.6 |
| [`fractal/lacunarity_demo.sio`](fractal/lacunarity_demo.sio) | Demo for stdlib/fractal/lacunarity.sio | fail rc=1 (E011 E015 E013) | fail rc=1 (E200) | 3.9 / 0.7 |
| [`fractal/mod_demo.sio`](fractal/mod_demo.sio) | Demo for stdlib/fractal/mod.sio | fail rc=1 | fail rc=1 (E200) | 4.2 / 0.6 |
| [`fractal/multifractal_demo.sio`](fractal/multifractal_demo.sio) | Demo for stdlib/fractal/multifractal.sio | fail rc=1 | fail rc=1 (E200) | 4.1 / 0.8 |
| [`fractal_g2_ossm.sio`](fractal_g2_ossm.sio) | Fractal-G2 O-SSM: CONFIRMATION RUN | pass | timeout | 55.8 / 120.0 |
| [`fractal_g2_ossm_v2.sio`](fractal_g2_ossm_v2.sio) | Fractal-G2 O-SSM v2: Full Jacobian + Analytical Associator Gradient | pass | timeout | 62.1 / 120.0 |
| [`fractal_g2_ossm_v3.sio`](fractal_g2_ossm_v3.sio) | Fractal-G2 O-SSM v3 — 10-seed probe + ListOps downstream | crash (signal 54) | timeout | 69.3 / 120.0 |
| [`g2_octonion_derivations.sio`](g2_octonion_derivations.sio) | examples/g2_octonion_derivations.sio | fail rc=1 | crash (signal 11) | 4.5 / 1.3 |
| [`g2_projection_preictal.sio`](g2_projection_preictal.sio) | Exp 1: G₂-invariant subspace projection across the pre-ictal window | fail rc=1 | fail rc=1 (E035) | 4.2 / 0.8 |
| [`gate_vs_zd_ablation.sio`](gate_vs_zd_ablation.sio) | Gate vs Zero-Divisor Ablation: Decomposing the S-SSM Advantage | pass | timeout | 45.3 / 120.0 |
| [`genetic_code_168_demo.sio`](genetic_code_168_demo.sio) | genetic_code_168_demo.sio | pass | fail rc=1 (E035) | 6.2 / 0.8 |
| [`genomics_fairness_demo.sio`](genomics_fairness_demo.sio) | Genomics Fairness — Epistemic Bias Quantification Pipeline | pass | pass | 6.5 / 0.7 |
| [`glm_optimization_demo.sio`](glm_optimization_demo.sio) | GLM-4.7 ML-Guided Optimization Demo | fail rc=1 | fail rc=1 (E200 E170 E035) | 4.2 / 0.8 |
| [`gpu.sio`](gpu.sio) | Public GPU-profile example for the checked artifact. | pass | pass | 4.4 / 0.8 |
| [`gpu/fft_demo.sio`](gpu/fft_demo.sio) | Demo for stdlib/gpu/fft.sio | fail rc=1 | fail rc=1 (E200) | 4.1 / 0.7 |
| [`gpu/mod_demo.sio`](gpu/mod_demo.sio) | Demo for stdlib/gpu/mod.sio | fail rc=1 | fail rc=1 | 4.7 / 0.9 |
| [`gpu/smooth_demo.sio`](gpu/smooth_demo.sio) | Demo for stdlib/gpu/smooth.sio | fail rc=1 | fail rc=1 (E200) | 4.1 / 0.8 |
| [`gpu/stats_demo.sio`](gpu/stats_demo.sio) | Demo for stdlib/gpu/stats.sio | fail rc=1 | fail rc=1 (E200) | 4.1 / 0.9 |
| [`gpu/vec_add.sio`](gpu/vec_add.sio) | examples/gpu/vec_add.sio — GPU vector addition kernel example (Door 2) | pass | pass | 4.2 / 0.6 |
| [`gpu_epistemic_showcase.sio`](gpu_epistemic_showcase.sio) | GPU epistemic showcase preflight. | pass | pass | 4.8 / 0.6 |
| [`gpu_hypercomplex.sio`](gpu_hypercomplex.sio) | Hypercomplex arithmetic demo used as GPU benchmark scaffold. | pass | pass | 5.5 / 0.6 |
| [`grand_challenges/euler_line.sio`](grand_challenges/euler_line.sio) | Grand Challenge: Euler Line Theorem | fail rc=1 (E019) | fail rc=1 | 4.3 / 0.5 |
| [`grand_challenges/toxic_peak.sio`](grand_challenges/toxic_peak.sio) | Grand Challenge: Toxic Peak Verification | fail rc=1 (E019) | fail rc=1 (E035) | 3.7 / 0.8 |
| [`grand_challenges/turbulence.sio`](grand_challenges/turbulence.sio) | Grand Challenge: Mass Conservation Verification | fail rc=1 (E019) | fail rc=1 | 4.3 / 0.8 |
| [`graph/coherence_demo.sio`](graph/coherence_demo.sio) | Demo for stdlib/graph/coherence.sio | fail rc=1 (E019 E137) | fail rc=1 (E200) | 4.3 / 0.8 |
| [`graph/curvature_demo.sio`](graph/curvature_demo.sio) | Demo for stdlib/graph/curvature.sio | fail rc=1 (E019 E137) | fail rc=1 (E200) | 4.0 / 0.8 |
| [`graph/entropy_demo.sio`](graph/entropy_demo.sio) | Demo for stdlib/graph/entropy.sio | fail rc=1 (E019 E137) | fail rc=1 (E200 E035) | 4.2 / 0.7 |
| [`graph/multi_module_demo.sio`](graph/multi_module_demo.sio) | Multi-Module Graph Analysis Demo | fail rc=1 (E019) | fail rc=1 (E218) | 4.2 / 0.6 |
| [`graphics/demos/phase1_showcase.sio`](graphics/demos/phase1_showcase.sio) | Phase 1 Showcase - Comprehensive demo of Sounio terminal graphics | pass | pass | 5.3 / 0.7 |
| [`graphics/demos/phase2_showcase.sio`](graphics/demos/phase2_showcase.sio) | Phase 2 Showcase - Comprehensive 2D plotting demonstration | pass | fail rc=1 (E001) | 5.6 / 0.9 |
| [`graphics/demos/phase3_showcase.sio`](graphics/demos/phase3_showcase.sio) | phase3_showcase.sio - Comprehensive Phase 3 (3D Graphics) Demo | fail rc=1 (E137) | fail rc=1 (E200) | 4.5 / 0.7 |
| [`graphics/demos/phase4_showcase.sio`](graphics/demos/phase4_showcase.sio) | phase4_showcase.sio - Interactive Features Demo | fail rc=1 | fail rc=1 (E200) | 4.0 / 0.8 |
| [`graphics/demos/phase5_showcase.sio`](graphics/demos/phase5_showcase.sio) | phase5_showcase.sio - Phase 5 (Advanced Animation) Showcase | fail rc=1 | fail rc=1 (E200) | 4.2 / 0.6 |
| [`graphics/demos/phase6_showcase.sio`](graphics/demos/phase6_showcase.sio) | Phase 6 Showcase - Scientific Charts Demonstration | fail rc=1 | fail rc=1 (E035) | 4.4 / 0.9 |
| [`graphics/demos/phase7_showcase.sio`](graphics/demos/phase7_showcase.sio) | phase7_showcase.sio - Network/Graph Visualization Demonstration | fail rc=1 | fail rc=1 (E035) | 4.5 / 0.8 |
| [`graphics/lib/00_canvas.sio`](graphics/lib/00_canvas.sio) | Canvas Abstraction - In-memory framebuffer for terminal graphics | pass | pass | 4.9 / 0.6 |
| [`graphics/lib/00_canvas_simple.sio`](graphics/lib/00_canvas_simple.sio) | Simplified canvas test to debug borrow checker | pass | fail rc=1 (E218) | 3.9 / 0.8 |
| [`graphics/lib/01_colors.sio`](graphics/lib/01_colors.sio) | Color Management - ANSI 16-color palette with semantic mappings | pass | pass | 4.4 / 0.7 |
| [`graphics/lib/02_drawing.sio`](graphics/lib/02_drawing.sio) | Drawing Primitives - Bresenham lines, circles, rectangles, and fills | pass | pass | 4.8 / 0.7 |
| [`graphics/lib/03_text.sio`](graphics/lib/03_text.sio) | Text Rendering - Arbitrary positioning with alignment and styling | pass | pass | 4.8 / 0.7 |
| [`graphics/lib/10_axes.sio`](graphics/lib/10_axes.sio) | Viewport and Axes - Coordinate transformation for 2D plotting | pass | pass | 5.4 / 0.8 |
| [`graphics/lib/11_scatter.sio`](graphics/lib/11_scatter.sio) | Scatter Plots - 2D point plotting with markers and colors | pass | fail rc=1 (E001) | 4.5 / 0.6 |
| [`graphics/lib/12_line_plot.sio`](graphics/lib/12_line_plot.sio) | Line Plots - Connected curves for time series and continuous functions | pass | fail rc=1 (E001) | 7.5 / 0.7 |
| [`graphics/lib/13_bar_chart.sio`](graphics/lib/13_bar_chart.sio) | Bar Charts - Vertical and horizontal bar visualization | pass | pass | 5.3 / 0.7 |
| [`graphics/lib/14_histogram.sio`](graphics/lib/14_histogram.sio) | Histograms - Distribution visualization with automatic binning | pass | fail rc=1 (E001) | 6.2 / 0.6 |
| [`graphics/lib/15_heatmap.sio`](graphics/lib/15_heatmap.sio) | Heatmaps - 2D intensity visualization with color mapping | pass | pass | 5.7 / 0.7 |
| [`graphics/lib/20_projection.sio`](graphics/lib/20_projection.sio) | 20_projection.sio - 3D to 2D projection and transformations | fail rc=1 | fail rc=1 (E035 E200 E006) | 3.7 / 0.8 |
| [`graphics/lib/21_wireframe.sio`](graphics/lib/21_wireframe.sio) | 21_wireframe.sio - 3D wireframe mesh rendering | fail rc=1 | fail rc=1 (E200) | 4.0 / 0.6 |
| [`graphics/lib/23_scatter3d.sio`](graphics/lib/23_scatter3d.sio) | 23_scatter3d.sio - 3D scatter plot visualization | fail rc=1 | fail rc=1 (E200) | 4.3 / 0.7 |
| [`graphics/lib/30_input.sio`](graphics/lib/30_input.sio) | 30_input.sio - Terminal input handling (mouse and keyboard) | fail rc=1 | fail rc=1 | 3.9 / 0.8 |
| [`graphics/lib/31_interactive_plot.sio`](graphics/lib/31_interactive_plot.sio) | 31_interactive_plot.sio - Interactive plotting with zoom and pan | fail rc=1 | fail rc=1 (E035 E200 E001) | 4.0 / 0.7 |
| [`graphics/lib/40_animation.sio`](graphics/lib/40_animation.sio) | 40_animation.sio - Animation framework with double buffering | pass | pass | 4.9 / 0.6 |
| [`graphics/lib/41_particle_system.sio`](graphics/lib/41_particle_system.sio) | 41_particle_system.sio - Particle physics simulation | fail rc=1 | fail rc=1 (E200 E035) | 3.8 / 0.8 |
| [`graphics/lib/42_fluid_sim.sio`](graphics/lib/42_fluid_sim.sio) | 42_fluid_sim.sio - Fluid dynamics simulation | fail rc=1 | fail rc=1 (E200) | 4.2 / 0.6 |
| [`graphics/lib/50_phase_diagram.sio`](graphics/lib/50_phase_diagram.sio) | 50_phase_diagram.sio - Phase diagram visualization for scientific computing | fail rc=1 | fail rc=1 (E035) | 4.3 / 0.6 |
| [`graphics/lib/51_bifurcation.sio`](graphics/lib/51_bifurcation.sio) | 51_bifurcation.sio - Bifurcation diagram visualization | fail rc=1 | fail rc=1 (E035) | 4.0 / 0.8 |
| [`graphics/lib/52_poincare.sio`](graphics/lib/52_poincare.sio) | 52_poincare.sio - Poincaré section visualization | fail rc=1 | fail rc=1 (E035) | 4.7 / 0.7 |
| [`graphics/lib/53_streamlines.sio`](graphics/lib/53_streamlines.sio) | 53_streamlines.sio - Streamline visualization for vector fields | fail rc=1 | fail rc=1 (E035) | 4.3 / 0.6 |
| [`graphics/lib/60_graph.sio`](graphics/lib/60_graph.sio) | 60_graph.sio - Graph data structure for network visualization | pass | pass | 5.7 / 0.8 |
| [`graphics/lib/61_force_directed.sio`](graphics/lib/61_force_directed.sio) | 61_force_directed.sio - Force-directed graph layout | pass | pass | 6.7 / 0.9 |
| [`graphics/lib/62_tree_layout.sio`](graphics/lib/62_tree_layout.sio) | 62_tree_layout.sio - Tree layout algorithms | pass | pass | 6.2 / 0.6 |
| [`graphics/lib/63_graph_render.sio`](graphics/lib/63_graph_render.sio) | 63_graph_render.sio - Graph rendering with animation | pass | pass | 6.1 / 0.8 |
| [`graphics/lib/test_ansi.sio`](graphics/lib/test_ansi.sio) | Test if ANSI codes work | pass | pass | 4.6 / 0.6 |
| [`gui_hello/main.sio`](gui_hello/main.sio) | examples/gui_hello/main.sio — Native GUI demo | timeout | timeout | 120.0 / 120.0 |
| [`hello.sio`](hello.sio) | First Sounio program | pass | pass | 3.8 / 0.8 |
| [`hex_encode.sio`](hex_encode.sio) | Hex encoding: bytes to hex string representation | pass | pass | 4.6 / 0.8 |
| [`hexbin_demo/main.sio`](hexbin_demo/main.sio) | examples/hexbin_demo/main.sio — Hex-bin density + ridgeline distribution demo | timeout | timeout | 120.0 / 120.0 |
| [`higher_order.sio`](higher_order.sio) | Sprint 229: Native Higher-Order Functions Demo | fail rc=75 | fail rc=75 | 4.5 / 0.5 |
| [`hot_reload/server.sio`](hot_reload/server.sio) | examples/hot_reload/server.sio | fail rc=1 (E035 E137) | fail rc=1 (E224) | 3.9 / 0.6 |
| [`hsi_sedenion_demo.sio`](hsi_sedenion_demo.sio) | Sounio Demo: Sedenion Neural Networks for HSI Tissue Classification | pass | fail rc=1 (E035) | 6.7 / 0.8 |
| [`hsi_tissue_classification.sio`](hsi_tissue_classification.sio) | Sounio Example: HSI Tissue Classification with Sedenion NNs | fail rc=1 (E009 E137 E016) | fail rc=1 (E001 E200) | 3.5 / 1.0 |
| [`hssm_4way_benchmark.sio`](hssm_4way_benchmark.sio) | H-SSM 4-Way Benchmark: O-SSM vs S4-DIAG vs Naive-DIAG vs H-SSM | pass | pass | 11.4 / 63.6 |
| [`hssm_bptt_ablation.sio`](hssm_bptt_ablation.sio) | H-SSM BPTT Ablation: O-SSM vs H-SSM with Full Backprop Through A | pass | pass | 23.0 / 79.5 |
| [`hssm_deep_listops.sio`](hssm_deep_listops.sio) | Deep ListOps: O-SSM vs S4-DIAG vs Naive-DIAG vs H-SSM at Depth-3 | pass | timeout | 30.8 / 120.0 |
| [`hssm_native_algebra.sio`](hssm_native_algebra.sio) | Native Algebra SSM Benchmark: True Octonion vs True Quaternion vs Diagonal | pass | pass | 14.6 / 100.8 |
| [`http/mod_demo.sio`](http/mod_demo.sio) | Demo for stdlib/http/mod.sio | fail rc=1 (E137 E011) | fail rc=1 (E200 E006 E001) | 3.5 / 0.6 |
| [`hydrogen/h2_verified_surface_rate.sio`](hydrogen/h2_verified_surface_rate.sio) | examples/hydrogen/h2_verified_surface_rate.sio | fail rc=1 (E004 E001) | pass | 3.2 / 0.8 |
| [`hydrogen/mhhc_batch_margins.sio`](hydrogen/mhhc_batch_margins.sio) | examples/hydrogen/mhhc_batch_margins.sio | pass | pass | 9.0 / 1.0 |
| [`hydrogen/mhhc_cascade.sio`](hydrogen/mhhc_cascade.sio) | examples/hydrogen/mhhc_cascade.sio | pass | timeout | 75.0 / 120.0 |
| [`hydrogen/mhhc_corner_pbox.sio`](hydrogen/mhhc_corner_pbox.sio) | examples/hydrogen/mhhc_corner_pbox.sio | timeout | timeout | 120.0 / 120.0 |
| [`hyperbolic_semantic_networks/_orc_helpers_block.sio`](hyperbolic_semantic_networks/_orc_helpers_block.sio) | cost: c00=0, c01=1, c10=1, c11=0, edge_dist=1 | library | library | — |
| [`hyperbolic_semantic_networks/affect_net_curv_data.sio`](hyperbolic_semantic_networks/affect_net_curv_data.sio) | GENERATED by scripts/research/esm_affect_network_fixture.py — Kossakowski ESM. DO NOT EDIT. | library | library | — |
| [`hyperbolic_semantic_networks/affect_net_curvature.sio`](hyperbolic_semantic_networks/affect_net_curvature.sio) | Temporal discrete-curvature of the individual affect NETWORK — real Kossakowski 2017 ESM (MDD). | fail rc=1 (E137) | fail rc=1 (E200) | 3.8 / 0.7 |
| [`hyperbolic_semantic_networks/affect_network_orc.sio`](hyperbolic_semantic_networks/affect_network_orc.sio) | Multi-Variable Temporal Affect-Network ORC — Kossakowski 2017 ESM Data | fail rc=1 (E230 E245 E004) | pass | 3.6 / 1.3 |
| [`hyperbolic_semantic_networks/affect_orc_exact.sio`](hyperbolic_semantic_networks/affect_orc_exact.sio) | Exact-OT Ollivier-Ricci curvature of the per-window affect network — the ORIGINAL instrument. | fail rc=1 (E137) | fail rc=1 (E200) | 4.0 / 0.8 |
| [`hyperbolic_semantic_networks/affect_temporal_orc_data.sio`](hyperbolic_semantic_networks/affect_temporal_orc_data.sio) | GENERATED (temporal mode) — Kossakowski ESM. W=50 step=50 nW=29 K=12. DO NOT EDIT. | library | library | — |
| [`hyperbolic_semantic_networks/affect_temporal_orc_main.sio`](hyperbolic_semantic_networks/affect_temporal_orc_main.sio) | TEMPORAL-TRANSITION full-network ORC — direct heir of affect_network_orc.sio (4 nodes -> all 12). | fail rc=1 (E137) | fail rc=1 (E200) | 3.4 / 0.7 |
| [`hyperbolic_semantic_networks/certified_ews_min_sample.sio`](hyperbolic_semantic_networks/certified_ews_min_sample.sio) | Certified Geometric EWS — Minimum Sample-Size Curve | pass | pass | 6.0 / 1.7 |
| [`hyperbolic_semantic_networks/esm_real_data_orc.sio`](hyperbolic_semantic_networks/esm_real_data_orc.sio) | Real-Data Epistemic ORC on ESM Affect Data (Kossakowski 2017) | pass | pass | 6.8 / 1.2 |
| [`hyperbolic_semantic_networks/openesm_curv.sio`](hyperbolic_semantic_networks/openesm_curv.sio) | PREREGISTERED multi-subject test: does affect-network curvature predict depression BEYOND density? | fail rc=1 (E137) | fail rc=1 (E200) | 3.9 / 0.9 |
| [`hyperbolic_semantic_networks/openesm_fisher_data.sio`](hyperbolic_semantic_networks/openesm_fisher_data.sio) | GENERATED openESM per-subject curvature fixture — fisher. DO NOT EDIT. | library | library | — |
| [`hyperbolic_semantic_networks/openesm_geschwind_data.sio`](hyperbolic_semantic_networks/openesm_geschwind_data.sio) | GENERATED openESM per-subject curvature fixture — geschwind. DO NOT EDIT. | library | library | — |
| [`hyperbolic_semantic_networks/openesm_kuczynski_data.sio`](hyperbolic_semantic_networks/openesm_kuczynski_data.sio) | GENERATED openESM per-subject curvature fixture — kuczynski. DO NOT EDIT. | library | library | — |
| [`hyperbolic_semantic_networks/openesm_stouffer.sio`](hyperbolic_semantic_networks/openesm_stouffer.sio) | Stouffer Z combination of the three preregistered one-sided permutation p-values (H1: Forman | pass | pass | 4.8 / 0.6 |
| [`image/canvas_field.sio`](image/canvas_field.sio) | examples/image/canvas_field.sio — the ergonomic native-plot path: | fail rc=1 | crash (signal 11) | 3.8 / 3.9 |
| [`image/fractal_512.sio`](image/fractal_512.sio) | examples/image/fractal_512.sio — a 512x512 inferno Mandelbrot rendered and | fail rc=1 (E137) | timeout | 4.3 / 120.0 |
| [`image/pk_uncertainty_field.sio`](image/pk_uncertainty_field.sio) | examples/image/pk_uncertainty_field.sio | fail rc=1 (E137) | crash (signal 11) | 4.0 / 3.4 |
| [`image/png_native.sio`](image/png_native.sio) | examples/image/png_native.sio — native PNG output, no FFI, no Python. | pass | crash (signal 11) | 10.1 / 1.7 |
| [`integrate/decay_report.sio`](integrate/decay_report.sio) | First-order elimination with GUM uncertainty, via stdlib integrate::epistemic_ode. | pass | pass | 4.8 / 0.7 |
| [`interop_serve.sio`](interop_serve.sio) | examples/interop_serve.sio — Standalone Sounio IPC server example. | pass | pass | 3.8 / 0.6 |
| [`interp_168_basis.sio`](interp_168_basis.sio) | AMI — The 168 Basis as Mech-Interp Substrate | pass | pass | 5.0 / 0.8 |
| [`interp_vs_sae.sio`](interp_vs_sae.sio) | AMI vs SAE — Baseline comparison | pass | pass | 4.4 / 0.8 |
| [`io/argparse_demo.sio`](io/argparse_demo.sio) | Demo for stdlib/io/argparse.sio | fail rc=1 (E019 E137 E010) | fail rc=1 (E200) | 3.7 / 0.6 |
| [`jordan_168_hunt.sio`](jordan_168_hunt.sio) | THE HUNT: Does 168 appear in J₃(O)? | crash (signal 54) | pass | 6.4 / 1.8 |
| [`jordan_albert_demo.sio`](jordan_albert_demo.sio) | Exceptional Jordan Algebra J₃(O) — the Albert algebra | pass | pass | 5.0 / 0.6 |
| [`kernel_epistemic_vec_add.sio`](kernel_epistemic_vec_add.sio) | Example: Epistemic GPU kernel for vector addition with uncertainty | fail rc=1 (E070) | fail rc=1 (E070) | 3.8 / 0.5 |
| [`kernel_epistemic_wmma_matmul.sio`](kernel_epistemic_wmma_matmul.sio) | Example: Epistemic WMMA Tensor Core Matmul — Knowledge<lt;f32> 16×16 | pass | pass | 3.9 / 0.5 |
| [`kernel_matmul.sio`](kernel_matmul.sio) | Example: GPU kernel for matrix multiplication | pass | pass | 3.8 / 0.6 |
| [`kernel_source_level.sio`](kernel_source_level.sio) | Kretikos GPU Compiler — bounded kernel-surface example | crash (signal 4) | pass | 4.2 / 0.6 |
| [`kernel_vec_add.sio`](kernel_vec_add.sio) | Example: GPU kernel for vector addition | pass | pass | 4.2 / 0.7 |
| [`kmc/c2x2_exact_anchors.sio`](kmc/c2x2_exact_anchors.sio) | examples/kmc/c2x2_exact_anchors.sio | pass | pass | 20.0 / 3.9 |
| [`kmc/cme_2d_exact.sio`](kmc/cme_2d_exact.sio) | examples/kmc/cme_2d_exact.sio | pass | pass | 13.4 / 2.7 |
| [`kmc/cme_dissociative.sio`](kmc/cme_dissociative.sio) | examples/kmc/cme_dissociative.sio | pass | pass | 5.5 / 0.6 |
| [`kmc/cme_dissociative_exact.sio`](kmc/cme_dissociative_exact.sio) | examples/kmc/cme_dissociative_exact.sio | pass | pass | 5.6 / 0.6 |
| [`kmc/cme_dissociative_scaling.sio`](kmc/cme_dissociative_scaling.sio) | examples/kmc/cme_dissociative_exact.sio | pass | pass | 8.4 / 1.6 |
| [`kmc/cme_exact_langmuir.sio`](kmc/cme_exact_langmuir.sio) | examples/kmc/cme_exact_langmuir.sio | pass | pass | 5.5 / 0.8 |
| [`kmc/cme_exact_limits.sio`](kmc/cme_exact_limits.sio) | examples/kmc/cme_exact_limits.sio | crash (signal 8) | crash (signal 8) | 5.2 / 0.9 |
| [`kmc/cme_lateral.sio`](kmc/cme_lateral.sio) | examples/kmc/cme_dissociative_exact.sio | pass | pass | 8.4 / 1.8 |
| [`kmc/cme_lateral_crt.sio`](kmc/cme_lateral_crt.sio) | examples/kmc/cme_lateral_crt.sio | pass | pass | 10.8 / 3.6 |
| [`kmc/cme_lateral_reachable_thermolimit.sio`](kmc/cme_lateral_reachable_thermolimit.sio) | examples/kmc/cme_lateral_reachable_thermolimit.sio | pass | pass | 6.4 / 1.9 |
| [`kmc/cme_lateral_thermolimit.sio`](kmc/cme_lateral_thermolimit.sio) | examples/kmc/cme_lateral_thermolimit.sio | pass | pass | 4.9 / 0.8 |
| [`kmc/cme_modular_solve.sio`](kmc/cme_modular_solve.sio) | examples/kmc/cme_modular_solve.sio | pass | pass | 4.8 / 0.8 |
| [`kmc/hot_atom_blind.sio`](kmc/hot_atom_blind.sio) | examples/kmc/hot_atom_blind.sio | pass | pass | 38.6 / 25.0 |
| [`kmc/hot_atom_diffraction.sio`](kmc/hot_atom_diffraction.sio) | examples/kmc/hot_atom_diffraction.sio | pass | pass | 74.9 / 44.4 |
| [`kmc/kmc_c2x2_coarsening.sio`](kmc/kmc_c2x2_coarsening.sio) | examples/kmc/kmc_c2x2_coarsening.sio | timeout | timeout | 120.0 / 120.0 |
| [`kmc/kmc_c2x2_growth.sio`](kmc/kmc_c2x2_growth.sio) | examples/kmc/kmc_c2x2_growth.sio | timeout | timeout | 120.0 / 120.0 |
| [`kmc/kmc_c2x2_order.sio`](kmc/kmc_c2x2_order.sio) | examples/kmc/kmc_c2x2_order.sio | timeout | timeout | 120.0 / 120.0 |
| [`kmc/kmc_c2x2_production.sio`](kmc/kmc_c2x2_production.sio) | examples/kmc/kmc_c2x2_production.sio | timeout | timeout | 120.0 / 120.0 |
| [`kmc/kmc_gillespie_1d.sio`](kmc/kmc_gillespie_1d.sio) | examples/kmc/kmc_gillespie_1d.sio | pass | pass | 19.3 / 22.2 |
| [`kmc/kmc_gillespie_2d.sio`](kmc/kmc_gillespie_2d.sio) | examples/kmc/kmc_gillespie_2d.sio | timeout | timeout | 120.0 / 120.0 |
| [`kmc/m9_pair_relaxation_kmc.sio`](kmc/m9_pair_relaxation_kmc.sio) | examples/kmc/m9_pair_relaxation_kmc.sio | timeout | timeout | 120.0 / 120.0 |
| [`kmc/uptake_diffraction.sio`](kmc/uptake_diffraction.sio) | examples/kmc/uptake_diffraction.sio | timeout | pass | 120.0 / 84.4 |
| [`kmc/uptake_pd100_calibration.sio`](kmc/uptake_pd100_calibration.sio) | examples/kmc/uptake_pd100_calibration.sio | pass | timeout | 54.6 / 120.0 |
| [`knowledge_associator.sio`](knowledge_associator.sio) | Exp 5: Knowledge<lt;Associator> | pass | pass | 5.9 / 0.6 |
| [`knowledge_native.sio`](knowledge_native.sio) | Knowledge — GUM-compliant epistemic computing (JCGM 100:2008) | fail rc=1 | fail rc=1 (E035) | 3.7 / 0.5 |
| [`kretikos/gpu_pid_population.sio`](kretikos/gpu_pid_population.sio) | — | pass | pass | 3.5 / 0.5 |
| [`kretikos/lower_epistemic_dual_output_f32.sio`](kretikos/lower_epistemic_dual_output_f32.sio) | K-AXI source-lowering witness. | pass | pass | 3.4 / 0.5 |
| [`kretikos/lower_fma_f32.sio`](kretikos/lower_fma_f32.sio) | K-AXI source-lowering witness. | pass | pass | 3.5 / 0.6 |
| [`kretikos/lower_fma_f64.sio`](kretikos/lower_fma_f64.sio) | K-AXI source-lowering witness — f64 affine multiply-add. | pass | pass | 4.1 / 0.5 |
| [`kretikos/lower_knowledge_dual_output_f32.sio`](kretikos/lower_knowledge_dual_output_f32.sio) | K-AXI Knowledge<lt;f32> source-lowering witness. | pass | pass | 3.9 / 0.5 |
| [`kretikos/lower_vec_add_f32.sio`](kretikos/lower_vec_add_f32.sio) | K-AXI source-lowering witness. | pass | pass | 4.0 / 0.5 |
| [`kretikos/lower_vec_add_f64.sio`](kretikos/lower_vec_add_f64.sio) | K-AXI source-lowering witness — f64 scalar vector add. | pass | pass | 3.6 / 0.6 |
| [`kretikos/lower_vec_div_f32.sio`](kretikos/lower_vec_div_f32.sio) | K-AXI source-lowering witness. | pass | pass | 3.3 / 0.6 |
| [`kretikos/lower_vec_div_f64.sio`](kretikos/lower_vec_div_f64.sio) | K-AXI source-lowering witness — f64 scalar vector div. | pass | pass | 4.3 / 0.5 |
| [`kretikos/lower_vec_mul_f32.sio`](kretikos/lower_vec_mul_f32.sio) | K-AXI source-lowering witness. | pass | pass | 3.9 / 0.7 |
| [`kretikos/lower_vec_mul_f64.sio`](kretikos/lower_vec_mul_f64.sio) | K-AXI source-lowering witness — f64 scalar vector mul. | pass | pass | 3.6 / 0.7 |
| [`kretikos/lower_vec_sub_f32.sio`](kretikos/lower_vec_sub_f32.sio) | K-AXI source-lowering witness. | pass | pass | 3.9 / 0.5 |
| [`kretikos/lower_vec_sub_f64.sio`](kretikos/lower_vec_sub_f64.sio) | K-AXI source-lowering witness — f64 scalar vector sub. | pass | pass | 3.5 / 0.7 |
| [`kretikos/real_epistemic_dual_output.sio`](kretikos/real_epistemic_dual_output.sio) | kretikos: profile=epistemic_dual_output_f32 | pass | pass | 3.5 / 0.5 |
| [`kretikos/real_epistemic_elementwise.sio`](kretikos/real_epistemic_elementwise.sio) | kretikos: profile=epistemic_elementwise_f32 | pass | pass | 3.4 / 0.6 |
| [`kretikos/real_fma_f32.sio`](kretikos/real_fma_f32.sio) | kretikos: profile=fma_f32 | pass | pass | 4.0 / 0.7 |
| [`kretikos/real_fma_f64.sio`](kretikos/real_fma_f64.sio) | kretikos: profile=fma_f64 | pass | pass | 3.4 / 0.5 |
| [`kretikos/real_store_u32_const.sio`](kretikos/real_store_u32_const.sio) | kretikos: profile=store_u32_const | pass | pass | 3.8 / 0.5 |
| [`kretikos/real_vec_add.sio`](kretikos/real_vec_add.sio) | kretikos: profile=vec_add_f32 | pass | pass | 3.5 / 0.7 |
| [`kretikos/real_vec_add_f64.sio`](kretikos/real_vec_add_f64.sio) | kretikos: profile=vec_add_f64 | pass | pass | 3.3 / 0.5 |
| [`kretikos/real_vec_div.sio`](kretikos/real_vec_div.sio) | kretikos: profile=vec_div_f32 | pass | pass | 4.0 / 0.6 |
| [`kretikos/real_vec_div_f64.sio`](kretikos/real_vec_div_f64.sio) | kretikos: profile=vec_div_f64 | pass | pass | 3.9 / 0.5 |
| [`kretikos/real_vec_mul.sio`](kretikos/real_vec_mul.sio) | kretikos: profile=vec_mul_f32 | pass | pass | 3.4 / 0.5 |
| [`kretikos/real_vec_mul_f64.sio`](kretikos/real_vec_mul_f64.sio) | kretikos: profile=vec_mul_f64 | pass | pass | 4.2 / 0.5 |
| [`kretikos/real_vec_sub.sio`](kretikos/real_vec_sub.sio) | kretikos: profile=vec_sub_f32 | pass | pass | 3.3 / 0.6 |
| [`kretikos/real_vec_sub_f64.sio`](kretikos/real_vec_sub_f64.sio) | kretikos: profile=vec_sub_f64 | pass | pass | 3.5 / 0.7 |
| [`lc_surgical_controller_probe.sio`](lc_surgical_controller_probe.sio) | examples/lc_surgical_controller_probe.sio | pass | pass | 4.4 / 0.7 |
| [`lean_mini_compiler.sio`](lean_mini_compiler.sio) | lean_mini_compiler.sio — Phase C2: 3-Stage Bootstrap | pass | pass | 3.6 / 0.7 |
| [`lean_utils_self_host.sio`](lean_utils_self_host.sio) | lean_utils_self_host.sio — Phase C1: Self-Hosting Gate | fail rc=63 | fail rc=63 | 4.3 / 0.6 |
| [`lethal_dose_sedenion.sio`](lethal_dose_sedenion.sio) | Epistemic Warfarin Dosing Decision | pass | fail rc=1 (E035) | 6.1 / 0.6 |
| [`linalg/matrix_demo.sio`](linalg/matrix_demo.sio) | Demo for stdlib/linalg/matrix.sio | fail rc=1 (E137 E015) | fail rc=1 (E200) | 4.4 / 0.7 |
| [`linalg/solve_report.sio`](linalg/solve_report.sio) | Solve a linear system A x = b with stdlib linalg::matnm and print the result. | pass | pass | 4.5 / 0.8 |
| [`linalg/vector_demo.sio`](linalg/vector_demo.sio) | Demo for stdlib/linalg/vector.sio | fail rc=1 (E137) | fail rc=1 (E200) | 3.5 / 0.6 |
| [`listops_3way_benchmark.sio`](listops_3way_benchmark.sio) | ListOps 3-Way: O-SSM vs S4-DIAG vs Naive-DIAG | pass | pass | 11.0 / 41.8 |
| [`listops_benchmark.sio`](listops_benchmark.sio) | ListOps Benchmark: Nested Expression Evaluation (LRA-style) | pass | pass | 10.4 / 31.4 |
| [`listops_fullbp_3way.sio`](listops_fullbp_3way.sio) | ListOps with Full BPTT 2-Phase: O-SSM vs S4-DIAG vs Naive-DIAG | pass | pass | 5.4 / 6.6 |
| [`listops_nesting_depth.sio`](listops_nesting_depth.sio) | ListOps Nesting Depth: Compare 0-nesting (flat) vs 1-nesting vs 2-nesting | pass | pass | 5.8 / 10.1 |
| [`lsp_demo.sio`](lsp_demo.sio) | LSP Demo for Sounio Language Server Protocol | fail rc=1 | fail rc=1 (E200 E035 E170) | 3.1 / 0.5 |
| [`medlang/ast_demo.sio`](medlang/ast_demo.sio) | Demo for stdlib/medlang/ast.sio | fail rc=1 | fail rc=1 (E200) | 3.0 / 0.5 |
| [`medlang/codegen_demo.sio`](medlang/codegen_demo.sio) | Demo for stdlib/medlang/codegen.sio | fail rc=1 | fail rc=1 (E200) | 3.2 / 0.6 |
| [`medlang/integrate_demo.sio`](medlang/integrate_demo.sio) | Demo for stdlib/medlang/integrate.sio | fail rc=1 | fail rc=1 (E200) | 3.0 / 0.5 |
| [`medlang/lexer_demo.sio`](medlang/lexer_demo.sio) | Demo for stdlib/medlang/lexer.sio | fail rc=1 | fail rc=1 (E200) | 4.0 / 0.5 |
| [`medlang/parser_demo.sio`](medlang/parser_demo.sio) | Demo for stdlib/medlang/parser.sio | fail rc=1 | fail rc=1 (E200) | 3.7 / 0.7 |
| [`medlang/parser_ext_demo.sio`](medlang/parser_ext_demo.sio) | Demo for stdlib/medlang/parser_ext.sio | fail rc=1 (E137) | fail rc=1 (E200) | 3.1 / 0.6 |
| [`medlang/population/estimation_demo.sio`](medlang/population/estimation_demo.sio) | Demo for stdlib/medlang/population/estimation.sio | fail rc=1 | fail rc=1 (E200) | 3.5 / 0.5 |
| [`medlang/population/model_demo.sio`](medlang/population/model_demo.sio) | Demo for stdlib/medlang/population/model.sio | fail rc=1 (E137 E011) | fail rc=1 (E200) | 4.1 / 0.5 |
| [`medlang/population/simulation_demo.sio`](medlang/population/simulation_demo.sio) | Demo for stdlib/medlang/population/simulation.sio | fail rc=1 | fail rc=1 (E200) | 3.4 / 0.5 |
| [`medlang/population/variability_demo.sio`](medlang/population/variability_demo.sio) | Demo for stdlib/medlang/population/variability.sio | fail rc=1 (E137 E011) | fail rc=1 (E200) | 3.7 / 0.7 |
| [`meta_self_editing.sio`](meta_self_editing.sio) | examples/meta_self_editing.sio | fail rc=1 (E035) | pass | 3.5 / 0.5 |
| [`minimal.sio`](minimal.sio) | — | fail rc=42 | fail rc=42 | 3.6 / 0.5 |
| [`ml/autodiff_demo.sio`](ml/autodiff_demo.sio) | Demo for stdlib/ml/autodiff.sio | fail rc=1 (E137) | fail rc=1 (E200) | 3.5 / 0.7 |
| [`mlp_concrete_ablation.sio`](mlp_concrete_ablation.sio) | MLP Ablation on UCI Concrete — Paper B Experiment E3 | fail rc=1 (E035) | fail rc=1 (E035) | 3.5 / 0.6 |
| [`mode_switch_benchmark.sio`](mode_switch_benchmark.sio) | Mode-Switch Benchmark: ZD as Structural State Discontinuity | pass | timeout | 13.5 / 120.0 |
| [`mol_viewer/main.sio`](mol_viewer/main.sio) | examples/mol_viewer/main.sio — Interactive 3D molecular viewer | timeout | timeout | 120.0 / 120.0 |
| [`monte_carlo_pi.sio`](monte_carlo_pi.sio) | Monte Carlo smoke test using a simple PRNG and integer fixed-point output. | pass | pass | 3.7 / 0.5 |
| [`morse_decoding_benchmark.sio`](morse_decoding_benchmark.sio) | Morse Code Decoding: O-SSM vs Diagonal SSM | pass | fail rc=1 (E035) | 10.7 / 0.5 |
| [`multihead_unit_oct_benchmark.sio`](multihead_unit_oct_benchmark.sio) | Multi-Head Unit-Octonion SSM with Associator-as-Feature | pass | timeout | 79.1 / 120.0 |
| [`native/array_basics.sio`](native/array_basics.sio) | — | pass | fail rc=1 (E200) | 4.1 / 0.5 |
| [`native/array_init_tail.sio`](native/array_init_tail.sio) | Smoke case for stage1-smoke regress-check of the step-4 array-init handler. | pass | pass | 3.3 / 0.5 |
| [`native/enum_match.sio`](native/enum_match.sio) | Enum variant construction + match dispatch in native-v2 driver. | fail rc=1 (E004 E009) | pass | 3.1 / 0.7 |
| [`native/extended_reg_driver.sio`](native/extended_reg_driver.sio) | Test extended register R8..R15 encoding through native_compile_driver | pass | pass | 3.5 / 0.5 |
| [`native/fib.sio`](native/fib.sio) | A2: Recursive fibonacci — tests recursive calls, if/else, comparison, argument passing. | pass | pass | 3.9 / 0.8 |
| [`native/fibonacci.sio`](native/fibonacci.sio) | A2: Recursive fibonacci — tests function calls, if/else, comparison, recursion. | pass | pass | 3.3 / 0.7 |
| [`native/float_arith.sio`](native/float_arith.sio) | Float arithmetic smoke test. | pass | pass | 3.5 / 0.5 |
| [`native/float_ops.sio`](native/float_ops.sio) | Float operations comprehensive test. | pass | pass | 3.6 / 0.7 |
| [`native/hello.sio`](native/hello.sio) | A1: Minimal native compilation smoke test. | pass | pass | 3.5 / 0.7 |
| [`native/hof_closure_literal.sio`](native/hof_closure_literal.sio) | — | pass | pass | 4.1 / 0.5 |
| [`native/hof_fn_ref_call.sio`](native/hof_fn_ref_call.sio) | — | pass | pass | 3.3 / 0.5 |
| [`native/hof_param_signature.sio`](native/hof_param_signature.sio) | M1.2 — HOF parameter signature smoke test. | fail rc=1 (E009) | fail rc=1 (E001) | 3.1 / 0.7 |
| [`native/hof_science_kernel.sio`](native/hof_science_kernel.sio) | — | pass | pass | 3.6 / 0.5 |
| [`native/logical_ops.sio`](native/logical_ops.sio) | — | fail rc=1 (E005) | fail rc=1 | 3.4 / 0.7 |
| [`native/mutvar.sio`](native/mutvar.sio) | Minimal mutable variable test — isolate assignment bug. | pass | pass | 3.7 / 0.7 |
| [`native/nested_field.sio`](native/nested_field.sio) | Nested struct field access (a.b.c chains) in native-v2 driver. | pass | pass | 3.3 / 0.8 |
| [`native/struct_basic.sio`](native/struct_basic.sio) | Struct literal construction and field access in native-v2 driver. | pass | pass | 4.1 / 0.6 |
| [`native/struct_mutation.sio`](native/struct_mutation.sio) | Struct field write (mutation) in native-v2 driver. | pass | pass | 3.7 / 0.8 |
| [`native/struct_param.sio`](native/struct_param.sio) | Struct as function parameter + field access in native-v2 driver. | pass | pass | 4.1 / 0.7 |
| [`native/struct_return.sio`](native/struct_return.sio) | Function returning a struct (multi-value via rax:rdx) in native-v2 driver. | pass | pass | 3.5 / 0.5 |
| [`native/user_global_basic.sio`](native/user_global_basic.sio) | M1.2 step D — canonical user-global test: a user program that declares | pass | pass | 4.0 / 0.5 |
| [`native/while_loop.sio`](native/while_loop.sio) | A3: While loop + mutable variables + assignment. | pass | pass | 3.9 / 0.5 |
| [`native_algebra_4way_benchmark.sio`](native_algebra_4way_benchmark.sio) | 4-Way Native Algebra Benchmark: O-SSM vs H-SSM vs S-SSM vs Naive | pass | timeout | 27.9 / 120.0 |
| [`native_calculus.sio`](native_calculus.sio) | Sprint 230: Functional Calculus - Native ELF Demo | fail rc=1 (E009) | fail rc=49 | 3.8 / 0.5 |
| [`native_epistemic_pk.sio`](native_epistemic_pk.sio) | Native Epistemic PK: Full 6-Observation Gauss-Newton | fail rc=63 | fail rc=63 | 4.8 / 0.7 |
| [`native_higher_order.sio`](native_higher_order.sio) | Sprint 229: Native Higher-Order Functions Demo | fail rc=75 | fail rc=75 | 3.8 / 0.5 |
| [`network/null_hypothesis_demo.sio`](network/null_hypothesis_demo.sio) | Null Hypothesis Testing for Network Analysis | fail rc=1 | fail rc=1 (E200 E006 E035) | 3.4 / 0.6 |
| [`neuroreceptor_pet/pet_2tcm_epistemic.sio`](neuroreceptor_pet/pet_2tcm_epistemic.sio) | examples/neuroreceptor_pet/pet_2tcm_epistemic.sio | pass | pass | 6.0 / 1.1 |
| [`neuroreceptor_pet/pet_2tcm_export.sio`](neuroreceptor_pet/pet_2tcm_export.sio) | examples/neuroreceptor_pet/pet_2tcm_export.sio | pass | pass | 5.4 / 0.8 |
| [`neuroreceptor_pet/pet_fit_montecarlo.sio`](neuroreceptor_pet/pet_fit_montecarlo.sio) | examples/neuroreceptor_pet/pet_fit_montecarlo.sio | pass | timeout | 26.7 / 120.0 |
| [`neuroreceptor_pet/pet_fit_validation.sio`](neuroreceptor_pet/pet_fit_validation.sio) | examples/neuroreceptor_pet/pet_fit_validation.sio | fail rc=1 | pass | 4.2 / 14.8 |
| [`neuroreceptor_pet/pet_lammertsma1996_analysis.sio`](neuroreceptor_pet/pet_lammertsma1996_analysis.sio) | examples/neuroreceptor_pet/pet_lammertsma1996_analysis.sio | pass | pass | 4.2 / 0.8 |
| [`neuroreceptor_pet/pet_srtm.sio`](neuroreceptor_pet/pet_srtm.sio) | examples/neuroreceptor_pet/pet_srtm.sio | pass | pass | 5.0 / 0.9 |
| [`neuroreceptor_pet/pet_tracer_variants.sio`](neuroreceptor_pet/pet_tracer_variants.sio) | examples/neuroreceptor_pet/pet_tracer_variants.sio | pass | pass | 5.4 / 3.8 |
| [`newton_root.sio`](newton_root.sio) | Sprint 230: Functional Calculus - Native ELF Demo | fail rc=50 | fail rc=50 | 3.7 / 0.6 |
| [`nn/activation_demo.sio`](nn/activation_demo.sio) | Demo for stdlib/nn/activation.sio | fail rc=1 (E137) | fail rc=1 (E200) | 3.7 / 0.7 |
| [`nn/autograd_demo.sio`](nn/autograd_demo.sio) | Demo for stdlib/nn/autograd.sio | fail rc=1 (E137 E015) | fail rc=1 (E200) | 5.4 / 1.7 |
| [`nn/dense2_demo.sio`](nn/dense2_demo.sio) | Demo for stdlib/nn/dense2.sio | fail rc=1 (E137) | fail rc=1 (E200) | 4.1 / 0.6 |
| [`nn/dense_demo.sio`](nn/dense_demo.sio) | Demo for stdlib/nn/dense.sio | fail rc=1 (E137) | fail rc=1 (E200) | 3.3 / 0.5 |
| [`nn/dense_layer_demo.sio`](nn/dense_layer_demo.sio) | Demo for stdlib/nn/dense_layer.sio | fail rc=1 (E137 E015) | fail rc=1 (E200) | 3.6 / 0.5 |
| [`nn/dense_quaternion_demo.sio`](nn/dense_quaternion_demo.sio) | Demo for stdlib/nn/dense_quaternion.sio | fail rc=1 (E137 E011) | fail rc=1 (E200) | 3.1 / 0.5 |
| [`nn/g2_equivariant_demo.sio`](nn/g2_equivariant_demo.sio) | Demo for stdlib/nn/g2_equivariant.sio | fail rc=1 (E137) | fail rc=1 (E200) | 3.0 / 0.6 |
| [`nn/mlp_classifier_demo.sio`](nn/mlp_classifier_demo.sio) | Demo for stdlib/nn/mlp_classifier.sio | fail rc=1 (E137) | fail rc=1 (E200) | 4.2 / 0.8 |
| [`nn/mlp_xor_demo.sio`](nn/mlp_xor_demo.sio) | Demo for stdlib/nn/mlp_xor.sio | fail rc=1 (E137 E015) | fail rc=1 (E200) | 3.1 / 0.5 |
| [`nn/optimizers_quaternion_demo.sio`](nn/optimizers_quaternion_demo.sio) | Demo for stdlib/nn/optimizers_quaternion.sio | fail rc=1 (E137) | fail rc=1 (E200) | 4.1 / 0.5 |
| [`nn/pbpk_example.sio`](nn/pbpk_example.sio) | stdlib/nn/pbpk_example.sio | fail rc=1 | fail rc=1 (E224) | 3.8 / 0.6 |
| [`nn/quaternion_demo.sio`](nn/quaternion_demo.sio) | Demo for stdlib/nn/quaternion.sio | fail rc=1 (E137) | fail rc=1 (E200) | 3.0 / 0.6 |
| [`nn/tensor_demo.sio`](nn/tensor_demo.sio) | Demo for stdlib/nn/tensor.sio | fail rc=1 (E137) | fail rc=1 (E200) | 3.7 / 0.5 |
| [`numerics/f128_is_f64_probe.sio`](numerics/f128_is_f64_probe.sio) | build 1e-20 by division so no literal-parsing path is involved | fail rc=1 (E004) | pass | 3.1 / 0.5 |
| [`oct_associator_ddi.sio`](oct_associator_ddi.sio) | Does octonion non-associativity predict clinical DDI order-dependence? | fail rc=1 (E001) | pass | 3.6 / 0.6 |
| [`oct_associator_ddi_drugbank.sio`](oct_associator_ddi_drugbank.sio) | DDI expanded replication: DrugBank multi-drug FAERS (faers_drugbank.csv) | fail rc=1 (E001 E175) | pass | 4.1 / 0.6 |
| [`oct_associator_direction.sio`](oct_associator_direction.sio) | 2b — Associator DIRECTION vs clinical order-asymmetry structure | fail rc=1 (E008) | pass | 3.5 / 0.6 |
| [`oct_associator_expanded.sio`](oct_associator_expanded.sio) | 2b-expanded — Associator DIRECTION on EXPANDED FAERS data (openFDA) structure | fail rc=1 (E008) | pass | 4.2 / 0.6 |
| [`oct_cohort_test.sio`](oct_cohort_test.sio) | Cohort-confounding test for the octonion-associator DDI finding | fail rc=1 (E008 E009) | fail rc=1 (E035) | 4.5 / 0.7 |
| [`oct_conjecture_test.sio`](oct_conjecture_test.sio) | oct_conjecture_test.sio — Testing three conjectures from first data | crash (signal 54) | pass | 9.6 / 1.9 |
| [`oct_connectome_demo.sio`](oct_connectome_demo.sio) | oct_connectome_demo.sio — First-ever computation of associator fields | pass | pass | 5.4 / 0.7 |
| [`oct_mediator_test.sio`](oct_mediator_test.sio) | Mechanistic validation — does the algebra's "4th mediator" show up in data? | fail rc=1 (E008) | fail rc=1 (E035) | 3.5 / 0.6 |
| [`oct_permutation_test.sio`](oct_permutation_test.sio) | Permutation test for η²(\|asymmetry\| \| m) = 0.833 on 12 filtered triples | fail rc=1 (E008) | fail rc=1 (E035) | 4.0 / 0.6 |
| [`oct_signed_test.sio`](oct_signed_test.sio) | Secondary analysis (pre-registered in cohort-test discussion): | fail rc=1 (E008) | fail rc=1 (E035) | 3.5 / 0.6 |
| [`octonion_168_associators.sio`](octonion_168_associators.sio) | The 168 Theorem: Octonion Basis Associators and PSL(2,7) | fail rc=1 (E137) | pass | 4.6 / 0.5 |
| [`octonion_albert_algebra.sio`](octonion_albert_algebra.sio) | The Albert Algebra: Exceptional Jordan Algebra J₃(O) | pass | fail rc=1 (E035) | 4.3 / 0.5 |
| [`octonion_associator_field.sio`](octonion_associator_field.sio) | Associator Field on a Graph — Bridge to Connectomics | pass | pass | 4.5 / 0.7 |
| [`octonion_catalan_branching.sio`](octonion_catalan_branching.sio) | Catalan Parenthesization Branching in Non-Associative Products | fail rc=1 (E056) | fail rc=1 (E035) | 4.2 / 0.8 |
| [`octonion_cross_product_7d.sio`](octonion_cross_product_7d.sio) | 7-Dimensional Cross Product from Octonions | pass | fail rc=1 (E035) | 4.6 / 0.7 |
| [`octonion_derivation_algebra.sio`](octonion_derivation_algebra.sio) | Der(O) = G₂: The Derivation Algebra of the Octonions | pass | fail rc=1 (E035) | 4.5 / 0.7 |
| [`octonion_example.sio`](octonion_example.sio) | Octonion Neural Network (OctNN) Example | fail rc=1 (E019 E007) | pass | 4.5 / 0.6 |
| [`octonion_g2_automorphisms.sio`](octonion_g2_automorphisms.sio) | G₂ Automorphisms of the Octonions | pass | pass | 4.6 / 0.6 |
| [`octonion_holonomy.sio`](octonion_holonomy.sio) | Octonion Holonomy: Curvature from Non-Associativity | pass | pass | 5.8 / 0.5 |
| [`octonion_magic_square.sio`](octonion_magic_square.sio) | Freudenthal-Tits Magic Square | pass | pass | 4.2 / 0.5 |
| [`octonion_malcev_algebra.sio`](octonion_malcev_algebra.sio) | Malcev Algebra: The Tangent Algebra of the Octonion Moufang Loop | pass | pass | 4.9 / 0.6 |
| [`octonion_nn_demo.sio`](octonion_nn_demo.sio) | Octonion Neural Network Demonstration | fail rc=1 (E019 E007) | pass | 3.9 / 0.8 |
| [`octonion_path_products.sio`](octonion_path_products.sio) | Octonion Path Products on Graphs — Norm Invariance & Non-Associativity | pass | pass | 4.0 / 0.6 |
| [`octonion_projective_plane.sio`](octonion_projective_plane.sio) | The Cayley Plane OP²: Octonion Projective Plane | pass | fail rc=1 (E035) | 4.5 / 0.7 |
| [`octonion_ssm.sio`](octonion_ssm.sio) | Octonion State Space Model (O-SSM) | pass | pass | 6.0 / 0.5 |
| [`octonion_ssm_extended.sio`](octonion_ssm_extended.sio) | Octonion State Space Model — Extended (O-SSM-X) | pass | pass | 7.5 / 0.6 |
| [`octonion_triality.sio`](octonion_triality.sio) | Triality: The Unique Symmetry of Spin(8) via Octonions | pass | fail rc=1 (E035) | 5.6 / 0.5 |
| [`octonionic_relativity.sio`](octonionic_relativity.sio) | octonionic_relativity.sio | crash (signal 11) | pass | 5.8 / 0.6 |
| [`ode/pbpk14_demo.sio`](ode/pbpk14_demo.sio) | Demo for stdlib/ode/pbpk14.sio | fail rc=1 (E015 E137) | fail rc=1 (E200) | 3.7 / 0.5 |
| [`ode/pbpk14_identifiability.sio`](ode/pbpk14_identifiability.sio) | PBPK-14 STRUCTURAL IDENTIFIABILITY from plasma observations | pass | pass | 6.3 / 0.7 |
| [`ode/pbpk14_rk4_demo.sio`](ode/pbpk14_rk4_demo.sio) | Demo for stdlib/ode/pbpk14_rk4.sio | fail rc=1 (E015 E137) | fail rc=1 (E200) | 3.3 / 0.6 |
| [`ode/pbpk14_stiff_backward_euler.sio`](ode/pbpk14_stiff_backward_euler.sio) | PBPK-14 model-form repair: a STIFF (implicit, L-stable) integrator | pass | pass | 7.2 / 1.2 |
| [`ode/pbpk3_stable_demo.sio`](ode/pbpk3_stable_demo.sio) | Demo for stdlib/ode/pbpk3_stable.sio | fail rc=1 (E015 E137) | fail rc=1 (E200) | 3.6 / 0.5 |
| [`ode/pbpk_debug_demo.sio`](ode/pbpk_debug_demo.sio) | Demo for stdlib/ode/pbpk_debug.sio | fail rc=1 | fail rc=1 (E200) | 2.9 / 0.5 |
| [`ode/pbpk_fast_demo.sio`](ode/pbpk_fast_demo.sio) | Demo for stdlib/ode/pbpk_fast.sio | fail rc=1 | fail rc=1 (E200) | 3.9 / 0.7 |
| [`ode/pbpk_minimal_demo.sio`](ode/pbpk_minimal_demo.sio) | Demo for stdlib/ode/pbpk_minimal.sio | fail rc=1 (E015 E137) | fail rc=1 (E200) | 3.2 / 0.5 |
| [`ode/pbpk_tiny_demo.sio`](ode/pbpk_tiny_demo.sio) | Demo for stdlib/ode/pbpk_tiny.sio | fail rc=1 | fail rc=1 (E200) | 3.0 / 0.5 |
| [`ode/pbpk_unrolled_demo.sio`](ode/pbpk_unrolled_demo.sio) | Demo for stdlib/ode/pbpk_unrolled.sio | fail rc=1 | fail rc=1 (E200) | 3.3 / 0.5 |
| [`ode/pbpk_working_demo.sio`](ode/pbpk_working_demo.sio) | Demo for stdlib/ode/pbpk_working.sio | fail rc=1 | fail rc=1 (E200) | 3.1 / 0.5 |
| [`ode/rk4_demo.sio`](ode/rk4_demo.sio) | Demo for stdlib/ode/rk4.sio | fail rc=1 (E137 E015) | fail rc=1 (E200) | 3.8 / 0.5 |
| [`ode/solver_demo.sio`](ode/solver_demo.sio) | Demo for stdlib/ode/solver.sio | fail rc=1 (E137 E015) | fail rc=1 (E200) | 3.9 / 0.6 |
| [`ode/tsit5_demo.sio`](ode/tsit5_demo.sio) | Demo for stdlib/ode/tsit5.sio | fail rc=1 (E137) | fail rc=1 (E200) | 4.0 / 0.5 |
| [`ode/tsit5_multicomp_demo.sio`](ode/tsit5_multicomp_demo.sio) | Demo for stdlib/ode/tsit5_multicomp.sio | fail rc=1 (E015 E137) | fail rc=1 (E200) | 3.3 / 0.6 |
| [`ofssm_benchmark.sio`](ofssm_benchmark.sio) | O-FSSM: Second-Order Optimization Over Non-Associative Landscapes | pass | pass | 17.8 / 89.1 |
| [`ofssm_trajectory_benchmark.sio`](ofssm_trajectory_benchmark.sio) | O-FSSM Trajectory Divergence with ANALYTICAL BPTT | pass | pass | 11.6 / 19.0 |
| [`onn_rotation_prediction.sio`](onn_rotation_prediction.sio) | 3D Rotation Prediction with Octonion Neural Networks | fail rc=1 (E007 E008 E009) | fail rc=1 (E035) | 3.7 / 0.6 |
| [`ontology/biomedical/mod_demo.sio`](ontology/biomedical/mod_demo.sio) | Demo for stdlib/ontology/biomedical/mod.sio | fail rc=1 | pass | 3.6 / 0.6 |
| [`ontology/biomedical/snomed_demo.sio`](ontology/biomedical/snomed_demo.sio) | Demo for stdlib/ontology/biomedical/snomed.sio | fail rc=1 | fail rc=1 (E200) | 3.9 / 0.5 |
| [`ontology/biomedical/snomed_elplus_adapter_demo.sio`](ontology/biomedical/snomed_elplus_adapter_demo.sio) | examples/ontology/biomedical/snomed_elplus_adapter_demo.sio | pass | fail rc=1 | 6.2 / 0.8 |
| [`ontology/biomedical/snomed_elplus_demo.sio`](ontology/biomedical/snomed_elplus_demo.sio) | examples/ontology/biomedical/snomed_elplus_demo.sio | pass | pass | 4.6 / 0.8 |
| [`ontology/cache_demo.sio`](ontology/cache_demo.sio) | Demo for stdlib/ontology/cache.sio | fail rc=1 | fail rc=1 (E200) | 3.6 / 0.7 |
| [`ontology/mod_demo.sio`](ontology/mod_demo.sio) | Demo for stdlib/ontology/mod.sio | fail rc=1 | pass | 3.6 / 0.6 |
| [`ontology/model_demo.sio`](ontology/model_demo.sio) | Demo for stdlib/ontology/model.sio | fail rc=1 | fail rc=1 (E200) | 3.4 / 0.7 |
| [`ontology/query_demo.sio`](ontology/query_demo.sio) | Demo for stdlib/ontology/query.sio | fail rc=1 | fail rc=1 (E200) | 3.6 / 0.6 |
| [`ontology/reasoner_demo.sio`](ontology/reasoner_demo.sio) | Demo for stdlib/ontology/reasoner.sio | fail rc=1 | fail rc=1 (E200) | 3.8 / 0.6 |
| [`ontology_elplus_closure_demo.sio`](ontology_elplus_closure_demo.sio) | Role-aware EL+ boolean closure demo — an executable mirror of the | pass | pass | 4.4 / 0.7 |
| [`ontology_pipeline_demo.sio`](ontology_pipeline_demo.sio) | End-to-end ontology pipeline demo — exercises the reusable stdlib | pass | pass | 5.5 / 1.0 |
| [`optimize/bfgs_demo.sio`](optimize/bfgs_demo.sio) | Demo for stdlib/optimize/bfgs.sio | fail rc=1 (E137) | fail rc=1 (E200 E035) | 4.1 / 0.6 |
| [`optimize/differential_evolution_demo.sio`](optimize/differential_evolution_demo.sio) | Demo for stdlib/optimize/differential_evolution.sio | fail rc=1 (E137) | fail rc=1 (E200 E035) | 3.7 / 0.5 |
| [`optimize/levenberg_marquardt_demo.sio`](optimize/levenberg_marquardt_demo.sio) | Demo for stdlib/optimize/levenberg_marquardt.sio | fail rc=1 (E137) | fail rc=1 (E035 E200) | 3.1 / 0.5 |
| [`optimize/mod_demo.sio`](optimize/mod_demo.sio) | Demo for stdlib/optimize/mod.sio | pass | pass | 3.4 / 0.5 |
| [`optimize/nelder_mead_demo.sio`](optimize/nelder_mead_demo.sio) | Demo for stdlib/optimize/nelder_mead.sio | fail rc=1 (E137) | fail rc=1 (E200 E035) | 3.9 / 0.6 |
| [`optimize/uncertainty_demo.sio`](optimize/uncertainty_demo.sio) | Demo for stdlib/optimize/uncertainty.sio | fail rc=1 | fail rc=1 (E200) | 3.6 / 0.7 |
| [`ossm_adam_sweep.sio`](ossm_adam_sweep.sio) | O-SSM Optimizer Sweep — SGD vs Momentum vs Adam vs Adam+Warmup | pass | pass | 11.3 / 43.1 |
| [`ossm_associator_attention.sio`](ossm_associator_attention.sio) | Non-Associative Attention: Associator-Weighted State Mixing | pass | pass | 5.5 / 1.9 |
| [`ossm_cayley_dickson_tower.sio`](ossm_cayley_dickson_tower.sio) | Cayley-Dickson Tower: R → C → H → O State Space Model Comparison | pass | pass | 7.7 / 10.8 |
| [`ossm_distinguishability.sio`](ossm_distinguishability.sio) | O-SSM State Distinguishability: Cross-Dim vs Diagonal | pass | pass | 4.8 / 0.5 |
| [`ossm_ekan_pipeline.sio`](ossm_ekan_pipeline.sio) | O-SSM → E-KAN Pipeline with Epistemic Knowledge tracking | pass | pass | 7.1 / 0.6 |
| [`ossm_fano_selective.sio`](ossm_fano_selective.sio) | Fano-Selective O-SSM: Structured Non-Associative Coupling | pass | pass | 5.1 / 1.9 |
| [`ossm_fullbp_sorting.sio`](ossm_fullbp_sorting.sio) | O-SSM with Full Backpropagation Through A Matrix | pass | pass | 13.1 / 16.2 |
| [`ossm_fullbp_v2.sio`](ossm_fullbp_v2.sio) | O-SSM Full-BPTT v2: Continuous Value Encoding | pass | pass | 10.9 / 33.1 |
| [`ossm_hessian_lm_mandelbrot.sio`](ossm_hessian_lm_mandelbrot.sio) | O-SSM with Levenberg-Marquardt Newton + Mandelbrot Curvature Regularizer | crash (signal 54) | timeout | 20.9 / 120.0 |
| [`ossm_hessian_newton.sio`](ossm_hessian_newton.sio) | O-SSM with Explicit Hessian Newton Step | crash (signal 54) | timeout | 35.1 / 120.0 |
| [`ossm_knowledge.sio`](ossm_knowledge.sio) | Octonion SSM with Epistemic Knowledge (EK) tracking | pass | pass | 5.8 / 0.6 |
| [`ossm_longrange_benchmark.sio`](ossm_longrange_benchmark.sio) | O-SSM Long-Range Sequence Order Benchmark | pass | pass | 10.5 / 30.4 |
| [`ossm_moufang_dynamics.sio`](ossm_moufang_dynamics.sio) | Moufang Loop Dynamics in O-SSM | pass | pass | 4.8 / 0.6 |
| [`ossm_multihead.sio`](ossm_multihead.sio) | Multi-Head O-SSM: 4 × 8-dim Octonion Heads, 1000+ Parameters | pass | pass | 17.7 / 40.4 |
| [`ossm_norm_composition.sio`](ossm_norm_composition.sio) | Composition Algebra Norm in O-SSM State Evolution | pass | pass | 4.8 / 0.6 |
| [`ossm_order_sensitivity.sio`](ossm_order_sensitivity.sio) | O-SSM Order Sensitivity: Where Cross-Dimensional Coupling Wins | pass | pass | 4.8 / 1.1 |
| [`ossm_scaled_comparison.sio`](ossm_scaled_comparison.sio) | Scaled O-SSM vs Diagonal SSM — 7-dim, 100 epochs, full gradient | pass | pass | 5.4 / 2.3 |
| [`ossm_scaled_lm.sio`](ossm_scaled_lm.sio) | Scaled O-SSM vs Q-SSM Comparison — 8-dim, 200 epochs, order-dependent task | pass | pass | 7.8 / 15.1 |
| [`ossm_selective.sio`](ossm_selective.sio) | Selective Octonion State Space Model (Selective O-SSM) | fail rc=1 (E137) | pass | 3.3 / 0.7 |
| [`ossm_stacked_2layer.sio`](ossm_stacked_2layer.sio) | Stacked 2-Layer Octonion State Space Model (O-SSM-2L) | fail rc=4 | fail rc=4 | 7.2 / 0.6 |
| [`ossm_vs_qssm_ablation.sio`](ossm_vs_qssm_ablation.sio) | O-SSM vs Q-SSM Ablation — Paper A Core Experiment | pass | pass | 7.2 / 1.0 |
| [`ossm_vs_qssm_fullbp.sio`](ossm_vs_qssm_fullbp.sio) | O-SSM vs Q-SSM — Cross-Dimensional vs Element-Wise on XOR | pass | pass | 4.5 / 0.7 |
| [`palindrome_benchmark.sio`](palindrome_benchmark.sio) | Palindrome Detection: Symmetry Recognition via Sequential Processing | pass | fail rc=1 (E035) | 6.3 / 0.7 |
| [`paper_validation.sio`](paper_validation.sio) | Paper Validation: Emax PD Sensitivity + PBPK Euler Convergence | pass | fail rc=1 (E035) | 5.5 / 0.6 |
| [`particle_physics/a_mu_cmd3_2pi_split.sio`](particle_physics/a_mu_cmd3_2pi_split.sio) | Citation witness. Sounio does not integrate σ(e⁺e⁻→π⁺π⁻). | pass | fail rc=1 (E200 E006 E001) | 5.4 / 1.4 |
| [`particle_physics/a_mu_gum_split.sio`](particle_physics/a_mu_gum_split.sio) | Citation witness. Sounio does not compute hadronic VP. It refuses to | pass | fail rc=1 (E200 E006 E001) | 6.9 / 1.5 |
| [`particle_physics/a_mu_hvp_ld_window.sio`](particle_physics/a_mu_hvp_ld_window.sio) | Citation witness. Sounio does not compute a Euclidean window. | pass | fail rc=1 (E200 E006 E001) | 5.2 / 1.4 |
| [`particle_physics/a_mu_hvp_sd_control.sio`](particle_physics/a_mu_hvp_sd_control.sio) | Citation witness. Sounio does not compute a Euclidean window. | pass | fail rc=1 (E200 E006 E001) | 5.4 / 1.2 |
| [`particle_physics/a_mu_hvp_window_ud.sio`](particle_physics/a_mu_hvp_window_ud.sio) | Citation witness. Sounio does not compute a Euclidean window. | pass | fail rc=1 (E200 E006 E001) | 4.8 / 1.2 |
| [`particle_physics/a_mu_wp25_dd_hvp_lo_absent.sio`](particle_physics/a_mu_wp25_dd_hvp_lo_absent.sio) | Table 5: estimates not provided. Sounio does not invent 6931/7132/7045 | pass | fail rc=1 (E200 E006 E001) | 5.5 / 1.0 |
| [`particle_physics/exp10_approx_effect_algebra.sio`](particle_physics/exp10_approx_effect_algebra.sio) | examples/particle_physics/exp10_approx_effect_algebra.sio | fail rc=1 (E259 E035) | pass | 4.4 / 0.8 |
| [`particle_physics/exp10_approx_physics.sio`](particle_physics/exp10_approx_physics.sio) | examples/particle_physics/exp10_approx_physics.sio | pass | fail rc=1 (E200 E006 E001) | 5.9 / 1.3 |
| [`particle_physics/exp11_scheme_approx_product.sio`](particle_physics/exp11_scheme_approx_product.sio) | examples/particle_physics/exp11_scheme_approx_product.sio | fail rc=1 (E259) | pass | 4.0 / 0.7 |
| [`particle_physics/exp123_madaros_core.sio`](particle_physics/exp123_madaros_core.sio) | examples/particle_physics/exp123_madaros_core.sio | pass | fail rc=1 (E200 E006 E001) | 8.0 / 1.6 |
| [`particle_physics/exp123_z_metrology_nonunitary_ew.sio`](particle_physics/exp123_z_metrology_nonunitary_ew.sio) | examples/particle_physics/exp123_z_metrology_nonunitary_ew.sio | pass | fail rc=1 (E200 E006 E001) | 10.0 / 1.9 |
| [`particle_physics/exp12_residual_closure.sio`](particle_physics/exp12_residual_closure.sio) | examples/particle_physics/exp12_residual_closure.sio | fail rc=1 (E259) | pass | 4.3 / 0.7 |
| [`particle_physics/exp13_amplitude_honesty.sio`](particle_physics/exp13_amplitude_honesty.sio) | examples/particle_physics/exp13_amplitude_honesty.sio | pass | pass | 6.4 / 1.0 |
| [`particle_physics/exp14_amp_to_xsec.sio`](particle_physics/exp14_amp_to_xsec.sio) | examples/particle_physics/exp14_amp_to_xsec.sio | pass | pass | 6.9 / 1.0 |
| [`particle_physics/exp15_w_amp_to_xsec.sio`](particle_physics/exp15_w_amp_to_xsec.sio) | examples/particle_physics/exp15_w_amp_to_xsec.sio | pass | pass | 5.6 / 0.8 |
| [`particle_physics/exp16_h_amp_to_xsec.sio`](particle_physics/exp16_h_amp_to_xsec.sio) | examples/particle_physics/exp16_h_amp_to_xsec.sio | pass | pass | 5.8 / 1.0 |
| [`particle_physics/exp17_zwh_amp_xsec_ledger.sio`](particle_physics/exp17_zwh_amp_xsec_ledger.sio) | examples/particle_physics/exp17_zwh_amp_xsec_ledger.sio | pass | pass | 8.7 / 1.0 |
| [`particle_physics/exp18_w_vertex_amp_to_xsec.sio`](particle_physics/exp18_w_vertex_amp_to_xsec.sio) | examples/particle_physics/exp18_w_vertex_amp_to_xsec.sio | pass | pass | 6.7 / 1.0 |
| [`particle_physics/exp19_h_yukawa_amp_to_xsec.sio`](particle_physics/exp19_h_yukawa_amp_to_xsec.sio) | examples/particle_physics/exp19_h_yukawa_amp_to_xsec.sio | pass | pass | 6.2 / 1.0 |
| [`particle_physics/exp4_unstable_spectrum.sio`](particle_physics/exp4_unstable_spectrum.sio) | examples/particle_physics/exp4_unstable_spectrum.sio | pass | pass | 6.8 / 0.9 |
| [`particle_physics/exp5_broken_structure_dual.sio`](particle_physics/exp5_broken_structure_dual.sio) | examples/particle_physics/exp5_broken_structure_dual.sio | pass | pass | 8.9 / 0.8 |
| [`particle_physics/exp6_universal_deficit_xi.sio`](particle_physics/exp6_universal_deficit_xi.sio) | examples/particle_physics/exp6_universal_deficit_xi.sio | pass | pass | 6.3 / 0.7 |
| [`particle_physics/exp7_gum_xi_tension_transfer.sio`](particle_physics/exp7_gum_xi_tension_transfer.sio) | examples/particle_physics/exp7_gum_xi_tension_transfer.sio | crash (signal 4) | fail rc=1 (E200 E006 E001) | 6.5 / 1.7 |
| [`particle_physics/exp8_deficit_collapse_failure.sio`](particle_physics/exp8_deficit_collapse_failure.sio) | examples/particle_physics/exp8_deficit_collapse_failure.sio | fail rc=1 (E259) | fail rc=1 (E200 E006 E001) | 3.7 / 1.8 |
| [`particle_physics/exp9_engine_joint_gum.sio`](particle_physics/exp9_engine_joint_gum.sio) | examples/particle_physics/exp9_engine_joint_gum.sio | pass | pass | 7.0 / 0.7 |
| [`pathion_projective_measurement.sio`](pathion_projective_measurement.sio) | examples/pathion_projective_measurement.sio | pass | pass | 4.0 / 0.6 |
| [`pbpk/darwin_calibrated_10drugs.sio`](pbpk/darwin_calibrated_10drugs.sio) | — | library | library | — |
| [`pbpk_dashboard/main.sio`](pbpk_dashboard/main.sio) | examples/pbpk_dashboard/main.sio — Interactive one-compartment PK dashboard | timeout | timeout | 120.0 / 120.0 |
| [`pbpk_gum_vs_montecarlo.sio`](pbpk_gum_vs_montecarlo.sio) | GUM vs Monte Carlo Validation — Rapamycin 3-Compartment PBPK | pass | pass | 6.0 / 7.2 |
| [`pbpk_simple.sio`](pbpk_simple.sio) | Simplified PBPK-style one-compartment demo with compiler-native epistemic types. | fail rc=1 (E245 E230 E012) | pass | 3.4 / 0.6 |
| [`pbpk_viz/rapamycin_anova_genotype.sio`](pbpk_viz/rapamycin_anova_genotype.sio) | examples/pbpk_viz/rapamycin_anova_genotype.sio | pass | crash (signal 11) | 77.2 / 4.7 |
| [`pbpk_viz/rapamycin_bland_altman.sio`](pbpk_viz/rapamycin_bland_altman.sio) | examples/pbpk_viz/rapamycin_bland_altman.sio | pass | crash (signal 11) | 97.9 / 2.7 |
| [`pbpk_viz/rapamycin_curve.sio`](pbpk_viz/rapamycin_curve.sio) | examples/pbpk_viz/rapamycin_curve.sio | pass | fail rc=1 (E001) | 74.7 / 1.8 |
| [`pbpk_viz/rapamycin_cyp3a5.sio`](pbpk_viz/rapamycin_cyp3a5.sio) | examples/pbpk_viz/rapamycin_cyp3a5.sio — rapamycin exposure by CYP3A5 genotype. | pass | crash (signal 11) | 76.2 / 2.4 |
| [`pbpk_viz/rapamycin_dose_linearity.sio`](pbpk_viz/rapamycin_dose_linearity.sio) | examples/pbpk_viz/rapamycin_dose_linearity.sio | pass | crash (signal 11) | 62.5 / 3.0 |
| [`pbpk_viz/rapamycin_qq_normal.sio`](pbpk_viz/rapamycin_qq_normal.sio) | examples/pbpk_viz/rapamycin_qq_normal.sio | pass | crash (signal 11) | 62.6 / 8.0 |
| [`pbpk_viz/rapamycin_therapeutic_window.sio`](pbpk_viz/rapamycin_therapeutic_window.sio) | examples/pbpk_viz/rapamycin_therapeutic_window.sio | pass | crash (signal 11) | 74.0 / 2.8 |
| [`pediatric_pbpk_demo.sio`](pediatric_pbpk_demo.sio) | examples/pediatric_pbpk_demo.sio | fail rc=1 (E259) | pass | 3.8 / 1.3 |
| [`pharmacokinetic_model.sio`](pharmacokinetic_model.sio) | Modelo Farmacocinético com Incerteza Epistêmica em Sounio | fail rc=1 | fail rc=1 (E218 E035 E170) | 3.4 / 0.8 |
| [`phi_fano_cohomological.sio`](phi_fano_cohomological.sio) | Cohomological reformulation of the 168 Theorem | pass | pass | 4.3 / 0.7 |
| [`phonon_live/main.sio`](phonon_live/main.sio) | examples/phonon_live/main.sio — Live hepatic phonon lattice simulation | fail rc=1 | fail rc=1 (E224) | 4.0 / 0.6 |
| [`physics/cl13_lorentz.sio`](physics/cl13_lorentz.sio) | Lorentz Rotors in Spacetime Algebra Cl(1,3) | pass | fail rc=1 | 5.0 / 0.8 |
| [`physics/j3o_mass_spectrum.sio`](physics/j3o_mass_spectrum.sio) | J3(O) diagonal vacuum -> mass spectrum + Koide bridge, native Sounio. | pass | pass | 3.8 / 0.8 |
| [`physics/jordan_j3o.sio`](physics/jordan_j3o.sio) | J3(O) — exceptional Jordan (Albert) algebra: Freudenthal cubic norm, native Sounio. | pass | pass | 5.7 / 0.8 |
| [`physics/octonion_action_r7.sio`](physics/octonion_action_r7.sio) | Sequential left multiplication on O ≅ R⁸. For pure imaginaries the | pass | pass | 5.9 / 0.7 |
| [`physics/octonion_mass_delta.sio`](physics/octonion_mass_delta.sio) | Octonion exceptional-Jordan fermion-mass delta-consistency + cross-sector held-out test. | pass | pass | 6.1 / 2.4 |
| [`physics/sedenion_artin.sio`](physics/sedenion_artin.sio) | Alternativity: [a,a,v] = (aa)v − a(av). Octonions satisfy it (Artin). | pass | pass | 6.5 / 0.6 |
| [`physics/sedenion_ker_basis.sio`](physics/sedenion_ker_basis.sio) | Scan of e_i ± e_j finds exactly these four; they span (rank 4). | pass | pass | 6.8 / 0.6 |
| [`physics/sedenion_ker_intersect.sio`](physics/sedenion_ker_intersect.sio) | The four pair-type generators of ker L_z also satisfy v*z = 0. | pass | pass | 8.4 / 0.6 |
| [`physics/sedenion_ker_lw.sio`](physics/sedenion_ker_lw.sio) | w is in ker L_z and not in ker L_w (w*w nsq 4). z is in ker L_w | pass | pass | 7.3 / 0.6 |
| [`physics/sedenion_ker_lw_basis.sio`](physics/sedenion_ker_lw_basis.sio) | Scan of e_i ± e_j finds exactly these four; they span (rank 4). | pass | pass | 6.8 / 0.7 |
| [`physics/sedenion_ker_lz.sio`](physics/sedenion_ker_lz.sio) | Rank-nullity of left multiplication on S ≅ R¹⁶. Canonical | pass | pass | 6.6 / 0.9 |
| [`physics/sedenion_moufang.sio`](physics/sedenion_moufang.sio) | Unital alternative iff Moufang, so Artin fail already implies this. | pass | pass | 6.2 / 0.6 |
| [`physics/sedenion_zd_action.sio`](physics/sedenion_zd_action.sio) | Sequential left multiplication on S ≅ R¹⁶. Canonical z=e3+e10, | pass | pass | 6.7 / 0.6 |
| [`physics/sedenion_zd_twosided.sio`](physics/sedenion_zd_twosided.sio) | z=e3+e10, w=e6−e15. Madaros: z*w = w*z = 0, but z*z and w*w have | pass | pass | 6.2 / 0.6 |
| [`physics/triality_check.sio`](physics/triality_check.sio) | Triality core checks, native Sounio (algebra::octonion). | pass | pass | 5.3 / 0.5 |
| [`pireus_aarchmrs_tbl_import.sio`](pireus_aarchmrs_tbl_import.sio) | First Sounio semantic-authority witness for the pinned open Arm AARCHMRS | fail rc=2 | fail rc=2 | 8.3 / 0.9 |
| [`pireus_apple_a64_tbl_lowering.sio`](pireus_apple_a64_tbl_lowering.sio) | First Sounio authority stream for the Apple A64 TBL XOR candidate. | fail rc=1 (E035 E008) | fail rc=2 | 5.4 / 2.9 |
| [`pireus_apple_cpu_dependency_latency_interface_feasibility.sio`](pireus_apple_cpu_dependency_latency_interface_feasibility.sio) | First Sounio authority stream for Apple CPU cycle-interface feasibility. | fail rc=1 (E258) | fail rc=2 | 3.1 / 4.5 |
| [`pireus_apple_cpu_dependency_latency_interface_material_ingestion.sio`](pireus_apple_cpu_dependency_latency_interface_material_ingestion.sio) | Sounio authority stream for an Apple CPU material envelope. | fail rc=1 (E232 E035 E008) | fail rc=2 | 6.4 / 5.0 |
| [`pireus_apple_cpu_dependency_latency_request.sio`](pireus_apple_cpu_dependency_latency_request.sio) | First Sounio authority stream for the value-free Apple CPU request binding. | fail rc=1 (E258) | fail rc=2 | 3.3 / 3.6 |
| [`pireus_apple_metal_family_import.sio`](pireus_apple_metal_family_import.sio) | First Sounio semantic-authority witness for Apple's pinned MTLGPUFamily | fail rc=1 (E035) | fail rc=2 | 3.7 / 0.9 |
| [`pireus_apple_metal_table_cells.sio`](pireus_apple_metal_table_cells.sio) | First Sounio witness for geometry beneath the pinned Apple Metal tables. | fail rc=1 (E008) | fail rc=2 | 4.3 / 1.7 |
| [`pireus_apple_metal_table_cells_negatives.sio`](pireus_apple_metal_table_cells_negatives.sio) | Fail-closed witnesses for the Pireus PDF table-cell projection. | fail rc=1 (E008) | pass | 4.3 / 1.7 |
| [`pireus_cubic_operator_forge.sio`](pireus_cubic_operator_forge.sio) | Frozen Sounio authority executable for Cubic Operator Forge v4. | fail rc=1 (E232) | timeout | 3.9 / 120.0 |
| [`pireus_dgx_ptx_shfl_lowering.sio`](pireus_dgx_ptx_shfl_lowering.sio) | First Sounio authority stream for the DGX PTX `shfl.sync.bfly` candidate. | fail rc=1 (E035 E008) | fail rc=2 | 5.0 / 2.9 |
| [`pireus_execution_engine_query.sio`](pireus_execution_engine_query.sio) | Sounio semantic-authority witness for multi-engine Pireus machines. | pass | pass | 5.4 / 0.9 |
| [`pireus_graph_identity_composition.sio`](pireus_graph_identity_composition.sio) | First Sounio semantic-authority executable for Pireus graph identity. | fail rc=1 (E259 E035) | fail rc=2 | 4.4 / 3.4 |
| [`pireus_intel_vpermpd_semantics.sio`](pireus_intel_vpermpd_semantics.sio) | Sounio semantic-authority extraction of Intel VPERMPD selector semantics. | fail rc=1 (E008 E035) | fail rc=2 | 4.3 / 1.4 |
| [`pireus_material_engine_admission.sio`](pireus_material_engine_admission.sio) | First Sounio authority stream for receipt-bound Pireus engine admission. | fail rc=1 (E232 E035 E008) | fail rc=2 | 5.1 / 2.9 |
| [`pireus_multiprobe_block_certification.sio`](pireus_multiprobe_block_certification.sio) | Matcher-free Sounio semantic-authority producer for the Pireus V14 | timeout | timeout | 120.0 / 120.0 |
| [`pireus_multiprobe_block_certification_frozen_replay.sio`](pireus_multiprobe_block_certification_frozen_replay.sio) | Frozen Sounio semantic-authority replay for the V14 plan. | timeout | timeout | 120.0 / 120.0 |
| [`pireus_multiprobe_block_reuse_admission.sio`](pireus_multiprobe_block_reuse_admission.sio) | Sounio fixture for reuse admission without block-work replay. | pass | pass | 5.9 / 0.7 |
| [`pireus_operator_autogenesis.sio`](pireus_operator_autogenesis.sio) | Matcher-free Sounio authority executable for Operator Autogenesis v9. | fail rc=1 (E035 E232) | pass | 4.9 / 4.5 |
| [`pireus_operator_discovery_engine.sio`](pireus_operator_discovery_engine.sio) | Matcher-free Sounio authority executable for Operator Discovery Engine v10. | fail rc=1 (E035 E232) | timeout | 5.3 / 120.0 |
| [`pireus_operator_genesis.sio`](pireus_operator_genesis.sio) | First Sounio-produced TwistedXor16 operator-genesis search transcript. | pass | pass | 8.7 / 1.1 |
| [`pireus_operator_genesis_bilinear.sio`](pireus_operator_genesis_bilinear.sio) | First Sounio-produced transcript for the full bilinear Operator Genesis | crash (signal 53) | pass | 101.5 / 77.6 |
| [`pireus_operator_genesis_gl4.sio`](pireus_operator_genesis_gl4.sio) | First Sounio-produced GL(4,2) and sign-gauge Operator Genesis transcript. | fail rc=1 | pass | 3.7 / 98.0 |
| [`pireus_operator_genome.sio`](pireus_operator_genome.sio) | First Sounio semantic-authority executable for Operator Genome v3. | fail rc=1 (E232) | pass | 4.0 / 88.2 |
| [`pireus_operator_lowering_forge.sio`](pireus_operator_lowering_forge.sio) | First matcher-free Sounio authority executable for Operator-Lowering Forge | fail rc=1 (E035 E232) | timeout | 4.5 / 120.0 |
| [`pireus_operator_morphogenesis.sio`](pireus_operator_morphogenesis.sio) | Matcher-free first Sounio authority executable for Operator Morphogenesis | fail rc=1 (E137 E232) | timeout | 4.7 / 120.0 |
| [`pireus_operator_novelty_feedback.sio`](pireus_operator_novelty_feedback.sio) | Sounio authority executable for Operator Novelty Feedback v7. Its first | fail rc=1 (E035 E232) | timeout | 5.4 / 120.0 |
| [`pireus_operator_novelty_frontier.sio`](pireus_operator_novelty_frontier.sio) | Matcher-free Sounio authority executable for the V11 novelty frontier. | timeout | timeout | 120.0 / 120.0 |
| [`pireus_operator_orbit_canonicalization.sio`](pireus_operator_orbit_canonicalization.sio) | Frozen Sounio semantic authority replay for Pireus V13. | fail rc=1 (E137 E232) | timeout | 4.4 / 120.0 |
| [`pireus_operator_seed_kernel.sio`](pireus_operator_seed_kernel.sio) | Frozen Sounio authority executable for Operator Seed Kernel v8. | fail rc=1 (E035 E232) | pass | 4.7 / 5.4 |
| [`pireus_ptx_prmt_import.sio`](pireus_ptx_prmt_import.sio) | First Sounio semantic-authority witness for the pinned NVIDIA PTX `prmt` | fail rc=1 (E035) | fail rc=2 | 3.5 / 0.9 |
| [`pireus_quotient_novelty_forge.sio`](pireus_quotient_novelty_forge.sio) | Frozen Sounio authority executable for Quotient Novelty Forge v5. | fail rc=1 (E232) | timeout | 5.0 / 120.0 |
| [`pireus_target_cost_observation.sio`](pireus_target_cost_observation.sio) | First Sounio authority stream for typed, value-free Pireus cost requests. | fail rc=1 (E232 E035 E008) | fail rc=2 | 5.4 / 2.6 |
| [`pireus_target_profile_query.sio`](pireus_target_profile_query.sio) | Sounio-authority witness for Pireus v0.1 target and material profiles. | pass | pass | 6.4 / 0.8 |
| [`pireus_u250_dual_card_admission.sio`](pireus_u250_dual_card_admission.sio) | First Sounio authority result for the dual AMD Alveo U250 fleet. | pass | pass | 4.8 / 0.5 |
| [`pireus_u250_execution_engine.sio`](pireus_u250_execution_engine.sio) | Sounio authority projection of the admitted U250 into the Pireus graph. | fail rc=2 | fail rc=2 | 10.0 / 0.8 |
| [`pireus_u250_fpga_kernel_artifact.sio`](pireus_u250_fpga_kernel_artifact.sio) | Sounio authority declaration of a U250 kernel-artifact blueprint. | fail rc=2 | fail rc=2 | 10.6 / 1.0 |
| [`pireus_u250_kernel_launch_receipt.sio`](pireus_u250_kernel_launch_receipt.sio) | Sounio admission of one material U250 kernel launch observation. | fail rc=2 | fail rc=2 | 14.8 / 1.2 |
| [`pireus_u250_material_ingestion.sio`](pireus_u250_material_ingestion.sio) | Sounio authority classification of one sealed U250 material observation. | fail rc=2 | fail rc=2 | 6.0 / 0.6 |
| [`pireus_u250_xclbin_material_receipt.sio`](pireus_u250_xclbin_material_receipt.sio) | Sounio admission of recovered U250 xclbin material. | fail rc=2 | fail rc=2 | 10.8 / 0.9 |
| [`pireus_vector_capability_query.sio`](pireus_vector_capability_query.sio) | First Sounio-executable Pireus witness. | pass | pass | 4.7 / 0.6 |
| [`pireus_xed_permute_import.sio`](pireus_xed_permute_import.sio) | Sounio semantic-authority witness for the first pinned Intel XED slice. | fail rc=1 (E259) | fail rc=2 | 3.9 / 1.1 |
| [`pireus_xor_convolution_operation.sio`](pireus_xor_convolution_operation.sio) | First Sounio semantic-authority executable for the Pireus operation DAG. | fail rc=1 (E035) | pass | 4.5 / 1.0 |
| [`pireus_xor_lowering_legality.sio`](pireus_xor_lowering_legality.sio) | First Sounio authority stream for complete XOR lowering legality. | fail rc=1 (E035 E008) | fail rc=2 | 5.2 / 2.1 |
| [`pireus_xor_material_matching.sio`](pireus_xor_material_matching.sio) | First Sounio executable for the Pireus XOR selector plan. | fail rc=1 (E035) | fail rc=2 | 4.4 / 1.4 |
| [`pireus_xor_selector_material_admission.sio`](pireus_xor_selector_material_admission.sio) | First Sounio authority stream for target-local material receipt admission. | fail rc=1 (E035 E008) | fail rc=2 | 4.9 / 2.6 |
| [`playground_provenance_gate.sio`](playground_provenance_gate.sio) | Playground provenance gate | pass | pass | 3.6 / 0.7 |
| [`playground_publication_gate.sio`](playground_publication_gate.sio) | Playground publication gate | pass | pass | 4.1 / 0.7 |
| [`playground_scientific_contract.sio`](playground_scientific_contract.sio) | Playground scientific contract | pass | pass | 4.6 / 0.7 |
| [`ppcr_neural_net_demo.sio`](ppcr_neural_net_demo.sio) | PPCR × ML Demo: Study-Block-Governed Neural Network Experiment | pass | pass | 4.8 / 0.9 |
| [`precision_dosing.sio`](precision_dosing.sio) | Precision Medicine Dosing with Epistemic Uncertainty | pass | pass | 4.3 / 0.7 |
| [`privacy/dp_laplace_mean.sio`](privacy/dp_laplace_mean.sio) | examples/privacy/dp_laplace_mean.sio — ε-DP Laplace mean with GUM uncertainty | fail rc=1 (E039) | fail rc=1 (E035) | 4.2 / 0.7 |
| [`prob/beta_demo.sio`](prob/beta_demo.sio) | Demo for stdlib/prob/beta.sio | fail rc=1 (E137) | fail rc=1 (E200) | 3.9 / 0.6 |
| [`prob/distribution_report.sio`](prob/distribution_report.sio) | Distribution report using stdlib prob::distributions. | pass | pass | 5.4 / 0.8 |
| [`prob/distributions_demo.sio`](prob/distributions_demo.sio) | Demo for stdlib/prob/distributions.sio | fail rc=1 | fail rc=1 (E200) | 3.7 / 0.5 |
| [`prob/inference_demo.sio`](prob/inference_demo.sio) | Demo for stdlib/prob/inference.sio | fail rc=1 (E015 E137) | fail rc=1 (E200) | 3.5 / 0.7 |
| [`prob/mcmc_demo.sio`](prob/mcmc_demo.sio) | Demo for stdlib/prob/mcmc.sio | fail rc=1 | fail rc=1 (E200) | 4.1 / 0.7 |
| [`prob/mod_demo.sio`](prob/mod_demo.sio) | Demo for stdlib/prob/mod.sio | fail rc=1 (E137) | fail rc=1 (E200) | 4.0 / 0.6 |
| [`prob/normal_demo.sio`](prob/normal_demo.sio) | Demo for stdlib/prob/normal.sio | fail rc=1 (E137) | fail rc=1 (E200) | 4.5 / 0.5 |
| [`prob/random_demo.sio`](prob/random_demo.sio) | Demo for stdlib/prob/random.sio | fail rc=1 | fail rc=1 (E200) | 3.6 / 0.7 |
| [`process_test.sio`](process_test.sio) | Test program for stdlib/os/process.sio | pass | pass | 4.0 / 0.8 |
| [`projects/bad_missing_import/src/main.sio`](projects/bad_missing_import/src/main.sio) | — | fail rc=1 | fail rc=1 (E224) | 4.2 / 0.5 |
| [`projects/hello_pkg/src/greet.sio`](projects/hello_pkg/src/greet.sio) | — | library | library | — |
| [`projects/hello_pkg/src/main.sio`](projects/hello_pkg/src/main.sio) | — | pass | pass | 3.6 / 0.6 |
| [`psmnist_benchmark.sio`](psmnist_benchmark.sio) | Permuted Sequential MNIST (psMNIST): The Hardest SSM Benchmark | pass | fail rc=1 (E035) | 5.1 / 0.7 |
| [`qat_training_example.sio`](qat_training_example.sio) | Quantization-Aware Training (QAT) Example in Sounio | fail rc=1 (E007 E019 E011) | fail rc=1 (E035 E218) | 4.1 / 0.6 |
| [`qnn/01_hello_quaternion.sio`](qnn/01_hello_quaternion.sio) | 01_hello_quaternion.sio | fail rc=1 (E007 E001) | pass | 4.0 / 0.6 |
| [`qnn/02_basic_linear.sio`](qnn/02_basic_linear.sio) | 02_basic_linear.sio | fail rc=1 (E007 E008 E001) | fail rc=1 (E001) | 3.9 / 0.7 |
| [`qnn_complete_demo.sio`](qnn_complete_demo.sio) | Complete Quaternionic Neural Network Demo | fail rc=1 | fail rc=1 (E224) | 3.4 / 0.7 |
| [`qnn_demo_simple.sio`](qnn_demo_simple.sio) | Simple QNN (Quaternionic Neural Network) Demonstration | fail rc=1 (E137) | fail rc=1 (E035 E200) | 4.0 / 0.7 |
| [`qnn_example.sio`](qnn_example.sio) | Quaternionic Neural Network (QNN) Example | fail rc=1 | fail rc=1 (E224) | 3.3 / 0.5 |
| [`qnn_intrinsic_test.sio`](qnn_intrinsic_test.sio) | Minimal test for QNN intrinsics (no stdlib imports) | fail rc=1 (E137) | fail rc=1 (E200) | 3.7 / 0.7 |
| [`qnn_minimal_test.sio`](qnn_minimal_test.sio) | Minimal QNN validation example | fail rc=1 (E137 E003) | fail rc=1 (E224) | 4.0 / 0.7 |
| [`qnn_mnist.sio`](qnn_mnist.sio) | Quaternionic Neural Network - MNIST Classification Demo | fail rc=1 (E137) | fail rc=1 (E224) | 3.7 / 0.7 |
| [`qnn_mnist_train.sio`](qnn_mnist_train.sio) | End-to-End MNIST QNN Training Demo | fail rc=1 (E137 E015 E259) | fail rc=1 (E200) | 3.2 / 0.6 |
| [`qnn_simple_quat.sio`](qnn_simple_quat.sio) | Test just the quat() constructor | fail rc=1 (E137) | fail rc=1 (E200) | 3.5 / 0.6 |
| [`qnn_simple_test.sio`](qnn_simple_test.sio) | Simplest possible QNN test | fail rc=1 (E137 E003) | fail rc=1 (E224) | 3.8 / 0.5 |
| [`qnn_stdlib_test.sio`](qnn_stdlib_test.sio) | Test QNN with stdlib imports | fail rc=1 (E137 E003) | fail rc=1 (E224) | 3.4 / 0.5 |
| [`qnn_test_basic.sio`](qnn_test_basic.sio) | Basic Quat type test without imports | fail rc=1 (E137 E015) | fail rc=1 (E200) | 4.1 / 0.7 |
| [`qnn_validation.sio`](qnn_validation.sio) | QNN Validation - Demonstrates Working Quaternionic Neural Network Intrinsics | fail rc=1 (E137) | fail rc=1 (E200) | 3.4 / 0.6 |
| [`quantum_ml_pipeline.sio`](quantum_ml_pipeline.sio) | Quantum ML Pipeline — Molecular Property Prediction | fail rc=1 (E175 E012 E259) | fail rc=1 (E200 E001) | 3.6 / 0.8 |
| [`quaternion_null_model.sio`](quaternion_null_model.sio) | Exp 2: Quaternion null model | fail rc=1 | pass | 3.8 / 0.7 |
| [`random/distributions_demo.sio`](random/distributions_demo.sio) | Demo for stdlib/random/distributions.sio | fail rc=1 (E137) | fail rc=1 (E200) | 3.1 / 0.5 |
| [`random/mod_demo.sio`](random/mod_demo.sio) | Demo for stdlib/random/mod.sio | pass | pass | 3.7 / 0.7 |
| [`random/rng_demo.sio`](random/rng_demo.sio) | Demo for stdlib/random/rng.sio | fail rc=1 (E137) | fail rc=1 (E200) | 3.8 / 0.7 |
| [`random/sampling_demo.sio`](random/sampling_demo.sio) | Demo for stdlib/random/sampling.sio | fail rc=1 (E137 E011) | fail rc=1 (E200) | 3.2 / 0.5 |
| [`real_sounio_capability_demo.sio`](real_sounio_capability_demo.sio) | examples/real_sounio_capability_demo.sio | pass | pass | 5.2 / 0.6 |
| [`real_sounio_native_knowledge_demo.sio`](real_sounio_native_knowledge_demo.sio) | examples/real_sounio_native_knowledge_demo.sio | fail rc=1 (E245 E230 E004) | pass | 4.1 / 0.5 |
| [`real_world/01_dose_uncertainty.sio`](real_world/01_dose_uncertainty.sio) | Real-World Example 1: Pharmaceutical Dose Calculation with Uncertainty | pass | fail rc=1 (E035) | 4.7 / 0.5 |
| [`real_world/02_pbpk_oral_absorption.sio`](real_world/02_pbpk_oral_absorption.sio) | Real-World Example 2: Two-Compartment PBPK Model | fail rc=1 (E004) | fail rc=1 (E035) | 3.5 / 0.6 |
| [`real_world/03_trial_sample_size.sio`](real_world/03_trial_sample_size.sio) | Real-World Example 3: Clinical Trial Sample Size with Bayesian Updating | pass | fail rc=1 (E035) | 4.1 / 0.6 |
| [`real_world/04_gum_measurement_chain.sio`](real_world/04_gum_measurement_chain.sio) | Real-World Example 4: GUM-Compliant Measurement Traceability | pass | fail rc=1 (E035) | 4.5 / 0.8 |
| [`real_world/04_gum_measurement_simple.sio`](real_world/04_gum_measurement_simple.sio) | Real-World Example 4: GUM-Compliant Measurement (Simplified) | pass | fail rc=1 (E035) | 4.7 / 0.6 |
| [`real_world/05_pkpd_data_analysis.sio`](real_world/05_pkpd_data_analysis.sio) | Real-World Example 5: PK/PD Data Analysis with Uncertainty (native-compatible) | fail rc=1 (E004) | fail rc=1 (E035) | 4.1 / 0.7 |
| [`real_world/06_climate_ensemble.sio`](real_world/06_climate_ensemble.sio) | Real-World Example 6: Climate Model Ensemble Uncertainty | pass | fail rc=1 (E035) | 5.3 / 0.6 |
| [`real_world/06_darwin_atlas_pipeline.sio`](real_world/06_darwin_atlas_pipeline.sio) | Real-World Example 6: Darwin Atlas Pipeline in Pure Sounio | fail rc=2 | fail rc=1 (E035) | 6.9 / 0.6 |
| [`real_world/07_sensor_fusion.sio`](real_world/07_sensor_fusion.sio) | Real-World Example 7: Bayesian Sensor Fusion (Kalman Filter) | pass | fail rc=1 (E035) | 3.8 / 0.5 |
| [`real_world/pbpk_native.sio`](real_world/pbpk_native.sio) | PBPK Drug Concentration Simulator — Native compile target for souc-v2 | fail rc=1 (E137) | fail rc=1 (E035) | 3.8 / 0.7 |
| [`render/causal_dag.sio`](render/causal_dag.sio) | Causal DAG rendering — graph visualization with do-calculus annotations | pass | fail rc=1 (E200) | 4.6 / 0.5 |
| [`render/causal_intervention_frontier.sio`](render/causal_intervention_frontier.sio) | Causal Intervention Frontier | pass | pass | 7.9 / 1.7 |
| [`render/coverage_crystal_atelier.sio`](render/coverage_crystal_atelier.sio) | Headless four-sample coverage-AA material study. | pass | pass | 6.9 / 2.3 |
| [`render/cube_wireframe.sio`](render/cube_wireframe.sio) | Wireframe cube with perspective projection as real PPM output. | pass | fail rc=1 (E200) | 5.1 / 0.8 |
| [`render/epistemic_field_atelier.sio`](render/epistemic_field_atelier.sio) | Synthetic epistemic field study rendered entirely by Sounio. | pass | pass | 9.6 / 3.4 |
| [`render/quaternion_rotation.sio`](render/quaternion_rotation.sio) | Quaternion rotation visualization — hypercomplex geometry rendering | pass | fail rc=1 (E200) | 5.0 / 0.6 |
| [`render/rapamycin_material_study.sio`](render/rapamycin_material_study.sio) | Headless molecular material study rendered entirely by Sounio. | pass | pass | 13.0 / 2.9 |
| [`render/triangle_basic.sio`](render/triangle_basic.sio) | Basic triangle rendering as real PPM output. | pass | pass | 4.4 / 0.7 |
| [`render/triangle_ppm.sio`](render/triangle_ppm.sio) | Software-rasterized triangle output as PPM P3 | pass | pass | 4.7 / 0.7 |
| [`render/uncertainty_field.sio`](render/uncertainty_field.sio) | Uncertainty field visualization — epistemic rendering showcase | pass | pass | 4.5 / 0.7 |
| [`render/uncertainty_ppm.sio`](render/uncertainty_ppm.sio) | Uncertainty field rendered as PPM — epistemic heatmap | pass | pass | 3.9 / 0.8 |
| [`render_compile_driver.sio`](render_compile_driver.sio) | Minimal native compile driver for render fixtures. | fail rc=1 (E137) | fail rc=1 (E224) | 3.3 / 0.7 |
| [`research/rna_cd_confirmatory/rna_cd_manifest.sio`](research/rna_cd_confirmatory/rna_cd_manifest.sio) | Fixture-scale canonical producer for the RNA Cayley-Dickson confirmatory lane. | fail rc=1 | fail rc=2 | 4.9 / 0.6 |
| [`retrocausal_interference.sio`](retrocausal_interference.sio) | THE RETROCAUSAL INTERFERENCE PATTERN | pass | pass | 4.0 / 0.6 |
| [`reverse_copy_benchmark.sio`](reverse_copy_benchmark.sio) | Reverse Copy: Reproduce Tokens in REVERSE Order | pass | fail rc=1 (E035) | 16.4 / 0.6 |
| [`routon_projective_measurement.sio`](routon_projective_measurement.sio) | examples/routon_projective_measurement.sio | fail rc=1 (E035 E008 E004) | pass | 3.8 / 0.6 |
| [`run_sedenion_benchmark.sio`](run_sedenion_benchmark.sio) | Sounio Example: Run Sedenion Benchmark with PAC Learning | fail rc=1 (E002 E003) | fail rc=1 (E200) | 4.2 / 0.6 |
| [`s4_baseline_benchmark.sio`](s4_baseline_benchmark.sio) | S4-Style Baseline: HiPPO-Initialized Diagonal SSM vs O-SSM | pass | pass | 8.0 / 27.0 |
| [`science/active_inference.sio`](science/active_inference.sio) | Minimal Friston Active Inference Agent | pass | fail rc=1 (E035) | 3.9 / 0.6 |
| [`science/bayesian_clinical_trial.sio`](science/bayesian_clinical_trial.sio) | Bayesian Sequential Clinical Trial -- Beta-Binomial Model | pass | pass | 5.0 / 1.8 |
| [`science/darwin_epistemic_pbpk.sio`](science/darwin_epistemic_pbpk.sio) | Darwin PBPK — Measured Parameter Uncertainty Bridge (Gen 13) | fail rc=1 (E035) | fail rc=1 (E035) | 3.8 / 0.6 |
| [`science/darwin_pop_confidence_budget.sio`](science/darwin_pop_confidence_budget.sio) | Darwin 2-Compartment PBPK — Population Confidence Budget (Gen 16) | fail rc=1 (E035) | fail rc=1 (E035) | 3.1 / 0.5 |
| [`science/entropy_information.sio`](science/entropy_information.sio) | Shannon Information Theory -- The Mathematics of Knowledge | pass | pass | 4.5 / 0.6 |
| [`science/epistemic_cascade.sio`](science/epistemic_cascade.sio) | Epistemic Uncertainty Cascade Through a Measurement Chain | pass | pass | 4.5 / 0.7 |
| [`science/epistemic_decay.sio`](science/epistemic_decay.sio) | Gen 17 D5: Temporal Confidence Decay | pass | fail rc=1 (E035) | 4.4 / 0.5 |
| [`science/epistemic_measured.sio`](science/epistemic_measured.sio) | Epistemic Measurement Bridge — Gen 12 Knowledge<lt;T> Demonstration | pass | pass | 4.1 / 0.5 |
| [`science/kalman_filter_tracking.sio`](science/kalman_filter_tracking.sio) | 1D Kalman Filter for State Estimation | pass | pass | 5.3 / 0.7 |
| [`science/lotka_volterra_ecosystem.sio`](science/lotka_volterra_ecosystem.sio) | Lotka-Volterra Predator-Prey Ecosystem Dynamics | pass | pass | 4.2 / 0.9 |
| [`science/markov_chain_monte_carlo.sio`](science/markov_chain_monte_carlo.sio) | Metropolis-Hastings MCMC for Bayesian Posterior Inference | pass | pass | 5.7 / 3.1 |
| [`science/maxwell_cl13.sio`](science/maxwell_cl13.sio) | Spacetime Algebra Cl(1,3) — Maxwell's Equations in Geometric Algebra | pass | pass | 4.4 / 0.7 |
| [`science/pbpk_2comp.sio`](science/pbpk_2comp.sio) | Two-Compartment PBPK Model — amounts formulation with RK4 Integration | pass | pass | 4.8 / 0.7 |
| [`science/pbpk_population.sio`](science/pbpk_population.sio) | Population PBPK Confidence Budget — 2-Compartment Model | pass | pass | 5.7 / 1.7 |
| [`science/pkpd_simulation.sio`](science/pkpd_simulation.sio) | One-Compartment PK Model with RK4 Integration | pass | pass | 4.6 / 2.2 |
| [`science/signal_analysis.sio`](science/signal_analysis.sio) | DFT-Based Signal Analysis | pass | pass | 4.9 / 0.7 |
| [`science/simpsons_paradox.sio`](science/simpsons_paradox.sio) | Simpson's Paradox Detection and Causal Correction | pass | pass | 4.2 / 0.6 |
| [`science/uncertainty_propagation.sio`](science/uncertainty_propagation.sio) | Measurement Uncertainty Propagation (GUM) | pass | pass | 4.2 / 0.5 |
| [`scientific_hello.sio`](scientific_hello.sio) | scientific_hello.sio — Sprint 235: print_f64 gate | pass | pass | 4.1 / 0.6 |
| [`scifar10_benchmark.sio`](scifar10_benchmark.sio) | Synthetic sCIFAR-10: O-SSM vs Diagonal SSM — 3072-Step Sequential | crash (signal 54) | fail rc=1 (E035) | 15.5 / 0.6 |
| [`search/mcts/core_demo.sio`](search/mcts/core_demo.sio) | Demo for stdlib/search/mcts/core.sio | fail rc=1 (E137) | fail rc=1 (E200) | 3.6 / 0.5 |
| [`search/mcts/examples/tictactoe_demo.sio`](search/mcts/examples/tictactoe_demo.sio) | Demo for stdlib/search/mcts/examples/tictactoe.sio | fail rc=1 (E019 E137) | fail rc=1 (E001 E200) | 3.3 / 0.7 |
| [`search/mcts/examples/uncertainty_demo_demo.sio`](search/mcts/examples/uncertainty_demo_demo.sio) | Demo for stdlib/search/mcts/examples/uncertainty_demo.sio | fail rc=1 (E137) | fail rc=1 (E200) | 3.3 / 0.6 |
| [`search/mcts/node_demo.sio`](search/mcts/node_demo.sio) | Demo for stdlib/search/mcts/node.sio | fail rc=1 (E137 E011) | fail rc=1 (E200 E006 E001) | 3.2 / 0.8 |
| [`search/mcts/policy_demo.sio`](search/mcts/policy_demo.sio) | Demo for stdlib/search/mcts/policy.sio | fail rc=1 (E137 E019) | fail rc=1 (E200 E001) | 3.2 / 0.7 |
| [`sedenion_168_verify.sio`](sedenion_168_verify.sio) | sedenion_168_verify.sio — Is 336 = 2 × 168 a coincidence? | crash (signal 54) | pass | 7.0 / 2.2 |
| [`sedenion_boundary.sio`](sedenion_boundary.sio) | Sedenion Boundary: Why O-SSM is the Deepest Possible | pass | fail rc=1 (E035) | 4.6 / 0.5 |
| [`sedenion_hessian_brain_demo.sio`](sedenion_hessian_brain_demo.sio) | examples/sedenion_hessian_brain_demo.sio | crash (signal 11) | crash (signal 11) | 9.5 / 1.3 |
| [`sedenion_projective_measurement.sio`](sedenion_projective_measurement.sio) | examples/sedenion_projective_measurement.sio | pass | pass | 6.1 / 0.8 |
| [`sedenion_ssm.sio`](sedenion_ssm.sio) | S-SSM: Sedenion State Space Model with Zero-Divisor Active Gating | pass | pass | 7.5 / 0.9 |
| [`sedenion_ssm_16ch_eeg.sio`](sedenion_ssm_16ch_eeg.sio) | Exp 13: S-SSM 16-channel spatial embedding | fail rc=3 | pass | 13.8 / 4.5 |
| [`sedenion_ssm_16freq.sio`](sedenion_ssm_16freq.sio) | S-SSM 16-frequency demux: the cleanest cross-coupling benchmark yet | pass | fail rc=1 (E035) | 8.3 / 0.8 |
| [`sedenion_ssm_ablation.sio`](sedenion_ssm_ablation.sio) | Door E — Fair-comparison ablation: is the win G₂-specific or just sedenion? | fail rc=1 (E035) | fail rc=1 (E035) | 3.7 / 0.9 |
| [`sedenion_ssm_alpha_why.sio`](sedenion_ssm_alpha_why.sio) | Why α=0.2?  Frequency sweep: does the optimum shift with signal period? | pass | fail rc=1 (E035) | 10.0 / 0.6 |
| [`sedenion_ssm_bptt.sio`](sedenion_ssm_bptt.sio) | S-SSM BPTT: gradient descent on α — basin of attraction + multi-basin map | pass | fail rc=1 (E035) | 10.0 / 0.9 |
| [`sedenion_ssm_deep.sio`](sedenion_ssm_deep.sio) | S-SSM Deep: Fine α-grid + Two-Tone Benchmark | pass | fail rc=1 (E035) | 9.6 / 0.9 |
| [`sedenion_ssm_edge_annihilation.sio`](sedenion_ssm_edge_annihilation.sio) | S-SSM Edge of Annihilation: Subspace-Selective Forgetting via ZD Proximity | pass | pass | 7.7 / 1.0 |
| [`sedenion_ssm_eeg_real.sio`](sedenion_ssm_eeg_real.sio) | S-SSM on REAL EEG: 16-channel motor-cortex strip, cross-hemisphere | pass | fail rc=1 (E035) | 8.2 / 0.6 |
| [`sedenion_ssm_free_a.sio`](sedenion_ssm_free_a.sio) | S-SSM Free A: BPTT on all 16 components of A — does A stay on G₂? | pass | fail rc=1 (E035) | 10.0 / 0.6 |
| [`sedenion_ssm_orbit.sio`](sedenion_ssm_orbit.sio) | Door F — Orbit universality: does α=0.2 hold for OTHER ZD pairs? | pass | fail rc=1 (E035) | 8.9 / 0.6 |
| [`sedenion_ssm_seizure_chb01.sio`](sedenion_ssm_seizure_chb01.sio) | Door I: S-SSM seizure detection — subject chb01 | fail rc=1 (E035) | fail rc=1 (E035) | 4.7 / 0.7 |
| [`sedenion_ssm_seizure_chb03.sio`](sedenion_ssm_seizure_chb03.sio) | Door I: S-SSM seizure detection — subject chb03 | fail rc=1 (E035) | fail rc=1 (E035) | 4.5 / 0.7 |
| [`sedenion_ssm_seizure_chb05.sio`](sedenion_ssm_seizure_chb05.sio) | Door I: S-SSM seizure detection — subject chb05 | fail rc=1 (E035) | fail rc=1 (E035) | 3.7 / 0.7 |
| [`sedenion_ssm_seizure_chb11.sio`](sedenion_ssm_seizure_chb11.sio) | Door D: S-SSM seizure detection — subject chb11 (blank-channel-cleaned) | fail rc=1 (E035) | fail rc=1 (E035) | 4.4 / 0.7 |
| [`sedenion_ssm_selective.sio`](sedenion_ssm_selective.sio) | Selective Sedenion SSM (S-SSM) with Zero-Divisor Annihilation | pass | pass | 6.1 / 0.9 |
| [`sedenion_ssm_spectrum.sio`](sedenion_ssm_spectrum.sio) | Door G — Spectral analysis of R_A : ℝ¹⁶ → ℝ¹⁶, R_A(h) = A · h | pass | pass | 6.4 / 1.0 |
| [`sedenion_ssm_stacked_2layer.sio`](sedenion_ssm_stacked_2layer.sio) | Stacked 2-Layer Selective Sedenion SSM (S-SSM-2L) | fail rc=2 | fail rc=2 | 6.9 / 1.9 |
| [`sedenion_ssm_train.sio`](sedenion_ssm_train.sio) | Trainable S-SSM: Sedenion State Space Model vs Real Diagonal SSM | pass | pass | 13.2 / 21.7 |
| [`sedenion_ssm_transfer.sio`](sedenion_ssm_transfer.sio) | Door H — Transfer function: is α=0.2 a linear resonance? | pass | pass | 6.7 / 1.0 |
| [`sedenion_unitarity_break.sio`](sedenion_unitarity_break.sio) | sedenion_unitarity_break.sio — Where Unitarity Dies | pass | pass | 6.0 / 1.1 |
| [`sedenion_wirtinger_residue.sio`](sedenion_wirtinger_residue.sio) | THEOREM: Wirtinger gradient through Cayley-Dickson multiplication is EXACT | pass | fail rc=1 (E035) | 6.0 / 0.7 |
| [`sedenion_wirtinger_theorem.sio`](sedenion_wirtinger_theorem.sio) | THEOREM (Wirtinger adjoint for Cayley-Dickson algebras) | fail rc=1 (E009 E008 E002) | fail rc=1 (E035) | 3.5 / 0.9 |
| [`sedenion_zero_div_hunt.sio`](sedenion_zero_div_hunt.sio) | sedenion_zero_div_hunt.sio — Systematic zero-divisor search | crash (signal 54) | pass | 7.5 / 5.1 |
| [`sedeniontrip_projective_measurement.sio`](sedeniontrip_projective_measurement.sio) | examples/sedeniontrip_projective_measurement.sio | fail rc=1 (E035 E008 E004) | pass | 3.5 / 1.1 |
| [`seizure_dynamics.sio`](seizure_dynamics.sio) | Seizure Microdinâmica: Hessian curvature over time | fail rc=1 (E011) | timeout | 3.5 / 120.0 |
| [`seizure_hessian_deep.sio`](seizure_hessian_deep.sio) | Epilepsy O-SSM Hessian: Deep Analysis of Non-Associative Curvature | fail rc=1 (E011) | timeout | 4.0 / 120.0 |
| [`seizure_hessian_ossm.sio`](seizure_hessian_ossm.sio) | Epilepsy O-SSM Hessian: Non-Associative Curvature in Seizure EEG | fail rc=1 (E011) | timeout | 4.2 / 120.0 |
| [`seizure_perpatient.sio`](seizure_perpatient.sio) | Per-Patient Seizure Hessian: Consistency + Effect Size + Fano + O/H-SSM | fail rc=1 (E011) | timeout | 3.6 / 120.0 |
| [`seizure_preprint.sio`](seizure_preprint.sio) | Seizure O-SSM Preprint Benchmark: Per-Patient + Cohen's d + Permutation + Fano | fail rc=1 (E011) | timeout | 4.1 / 120.0 |
| [`semantic_orc/dct_kec_compute.sio`](semantic_orc/dct_kec_compute.sio) | KEC-alpha COMPUTE + native cross-subject stats (Sounio does ALL the math). | fail rc=1 (E137) | fail rc=1 (E200) | 3.1 / 0.5 |
| [`semantic_orc/dct_kec_core.sio`](semantic_orc/dct_kec_core.sio) | KEC-alpha CORE — semantic-coherence-of-egocentric-distance, metric C. | pass | pass | 4.1 / 0.5 |
| [`semantic_orc/dct_kec_spectral.sio`](semantic_orc/dct_kec_spectral.sio) | ORIGINAL KEC (spectral half) on the DCT — E = von Neumann spectral entropy, C = lambda2. | fail rc=1 (E137) | fail rc=1 (E200) | 3.2 / 0.6 |
| [`semantic_orc/dct_kec_wave1_data.sio`](semantic_orc/dct_kec_wave1_data.sio) | GENERATED by scripts/research/dct_kec_fixture.py — wave 1. DO NOT EDIT. | library | library | — |
| [`semantic_orc/dct_kec_wave2_data.sio`](semantic_orc/dct_kec_wave2_data.sio) | GENERATED by scripts/research/dct_kec_fixture.py — wave 2. DO NOT EDIT. | library | library | — |
| [`semantic_orc/depression_epistemic_orc.sio`](semantic_orc/depression_epistemic_orc.sio) | depression_epistemic_orc.sio — EPISTEMIC LAYER over exact-OT semantic curvature | pass | pass | 10.5 / 0.8 |
| [`semantic_orc/depression_semantic_orc.sio`](semantic_orc/depression_semantic_orc.sio) | depression_semantic_orc.sio — Bootstrap CI over Published SWOW-EN κ Statistics | pass | pass | 4.1 / 0.8 |
| [`semantic_orc/depression_swow_orc.sio`](semantic_orc/depression_swow_orc.sio) | depression_swow_orc.sio — Depression Severity ORC + Octonion Associator Field | pass | pass | 5.1 / 0.8 |
| [`semantic_orc/sinkhorn_lse_orc.sio`](semantic_orc/sinkhorn_lse_orc.sio) | Semantic ORC Sinkhorn-LSE runtime gate. | pass | fail rc=1 (E035) | 6.3 / 0.6 |
| [`serialization/toml_demo.sio`](serialization/toml_demo.sio) | TOML Configuration Parser Demonstration | fail rc=1 | fail rc=1 (E200 E006 E001) | 3.3 / 0.8 |
| [`showcase/concurrent_pipeline.sio`](showcase/concurrent_pipeline.sio) | concurrent_pipeline.sio — Data processing pipeline | pass | crash (signal 11) | 4.0 / 0.7 |
| [`showcase/drug_dose_optimizer.sio`](showcase/drug_dose_optimizer.sio) | drug_dose_optimizer.sio — Pharmacokinetic modeling with uncertainty | pass | pass | 3.9 / 0.6 |
| [`showcase/effect_test_harness.sio`](showcase/effect_test_harness.sio) | effect_test_harness.sio — Algebraic effects for testable I/O | fail rc=1 | crash (signal 11) | 3.6 / 0.6 |
| [`showcase/genome_motif_scanner.sio`](showcase/genome_motif_scanner.sio) | genome_motif_scanner.sio — DNA sequence analysis with epistemic uncertainty | pass | crash (signal 11) | 4.5 / 0.6 |
| [`showcase/knowledge_graph_trainer.sio`](showcase/knowledge_graph_trainer.sio) | knowledge_graph_trainer.sio — Sedenion knowledge graph embedding | pass | crash (signal 11) | 4.4 / 0.6 |
| [`showcase/linear_file_server.sio`](showcase/linear_file_server.sio) | linear_file_server.sio — Linear types for resource safety | pass | pass | 3.9 / 0.6 |
| [`showcase/measurement_lab.sio`](showcase/measurement_lab.sio) | measurement_lab.sio — ISO GUM-compliant uncertainty propagation | pass | pass | 4.3 / 0.6 |
| [`showcase/ode_predator_prey.sio`](showcase/ode_predator_prey.sio) | ode_predator_prey.sio — Lotka-Volterra predator-prey dynamics | pass | pass | 4.8 / 0.6 |
| [`showcase/spectral_analyzer.sio`](showcase/spectral_analyzer.sio) | spectral_analyzer.sio — FFT-based signal analysis with epistemic confidence | pass | crash (signal 11) | 4.2 / 0.6 |
| [`showcase/type_safe_units.sio`](showcase/type_safe_units.sio) | type_safe_units.sio — Dimensional analysis at the type level | pass | pass | 4.7 / 0.5 |
| [`signal/epoch_demo.sio`](signal/epoch_demo.sio) | Demo for stdlib/signal/epoch.sio | fail rc=1 | fail rc=1 (E200) | 3.2 / 0.6 |
| [`signal/filter_demo.sio`](signal/filter_demo.sio) | Demo for stdlib/signal/filter.sio | fail rc=1 | fail rc=1 (E200) | 3.0 / 0.6 |
| [`signal/filter_report.sio`](signal/filter_report.sio) | Biosignal filtering demo with stdlib signal::filter. | pass | pass | 5.3 / 0.8 |
| [`signal/fractal_demo.sio`](signal/fractal_demo.sio) | Demo for stdlib/signal/fractal.sio | fail rc=1 | fail rc=1 (E200) | 3.0 / 0.6 |
| [`signal/spectral_demo.sio`](signal/spectral_demo.sio) | Demo for stdlib/signal/spectral.sio | fail rc=1 | fail rc=1 (E200) | 3.3 / 0.5 |
| [`signal/spectrum_report.sio`](signal/spectrum_report.sio) | Spectrum analysis with stdlib signal::fft: a two-tone signal over N=16 samples. | pass | pass | 4.7 / 0.5 |
| [`simple_test.sio`](simple_test.sio) | Simple test file | pass | pass | 3.1 / 0.6 |
| [`simulation/nbody.sio`](simulation/nbody.sio) | N-Body Gravitational Simulation (16 bodies, 2D, 1000 steps) | pass | pass | 4.8 / 1.5 |
| [`sleep_orbit_demo.sio`](sleep_orbit_demo.sio) | sleep_orbit_demo.sio — THE EXPERIMENT | pass | fail rc=1 | 5.5 / 0.6 |
| [`smnist_3way_benchmark.sio`](smnist_3way_benchmark.sio) | Sequential MNIST (Negative Control): O-SSM vs S4-DIAG vs Naive-DIAG | pass | pass | 5.1 / 5.5 |
| [`smnist_ossm_benchmark.sio`](smnist_ossm_benchmark.sio) | Sequential MNIST: O-SSM vs Diagonal SSM — Real Benchmark | pass | fail rc=1 (E035) | 4.5 / 0.5 |
| [`sorting_classification_benchmark.sio`](sorting_classification_benchmark.sio) | Sorting Classification: Pure Order Discrimination | pass | fail rc=1 (E035) | 7.7 / 0.5 |
| [`sorting_k_sweep.sio`](sorting_k_sweep.sio) | Sorting K-Sweep: O-SSM vs S4-DIAG vs Naive across K={3,5,8,10} | pass | pass | 8.2 / 24.5 |
| [`sorting_lr_ablation.sio`](sorting_lr_ablation.sio) | Sorting with LR Ablation: can S4-DIAG recover with 5x smaller learning rate? | pass | pass | 5.4 / 10.0 |
| [`sounio_science_flex/main.sio`](sounio_science_flex/main.sio) | examples/sounio_science_flex/main.sio | pass | pass | 8.1 / 1.4 |
| [`special/erf_report.sio`](special/erf_report.sio) | Error-function / normal-tail report using stdlib special::erf. | pass | pass | 5.4 / 0.6 |
| [`special/gamma_report.sio`](special/gamma_report.sio) | Gamma / log-gamma / digamma report using stdlib special::gamma. | pass | pass | 4.3 / 0.5 |
| [`src/autodiff/mod_demo.sio`](src/autodiff/mod_demo.sio) | Demo for stdlib/src/autodiff/mod.sio | fail rc=1 (E137) | fail rc=1 (E200) | 3.1 / 0.5 |
| [`src/linalg/matrix_demo.sio`](src/linalg/matrix_demo.sio) | Demo for stdlib/src/linalg/matrix.sio | fail rc=1 (E137 E015) | fail rc=1 (E200) | 3.1 / 0.5 |
| [`stats/box_plot.sio`](stats/box_plot.sio) | examples/stats/box_plot.sio | pass | crash (signal 11) | 57.7 / 2.4 |
| [`stats/descriptive_demo.sio`](stats/descriptive_demo.sio) | Demo for stdlib/stats/descriptive.sio | fail rc=1 | fail rc=1 (E200) | 3.0 / 0.5 |
| [`stats/effect_sizes_demo.sio`](stats/effect_sizes_demo.sio) | Effect Size Calculations Demo | fail rc=1 | fail rc=1 (E200 E006 E035) | 2.9 / 0.6 |
| [`stats/epistemic_suite_demo.sio`](stats/epistemic_suite_demo.sio) | examples/stats/epistemic_suite_demo.sio | pass | pass | 13.8 / 0.7 |
| [`stats/forest_plot.sio`](stats/forest_plot.sio) | examples/stats/forest_plot.sio | pass | crash (signal 11) | 60.0 / 2.5 |
| [`stats/full_analysis_report.sio`](stats/full_analysis_report.sio) | examples/stats/full_analysis_report.sio | pass | pass | 15.7 / 0.8 |
| [`stats/funnel_plot.sio`](stats/funnel_plot.sio) | examples/stats/funnel_plot.sio | pass | crash (signal 11) | 55.5 / 2.6 |
| [`stats/inferential_demo.sio`](stats/inferential_demo.sio) | Demo for stdlib/stats/inferential.sio | fail rc=1 | fail rc=1 (E200) | 3.0 / 0.5 |
| [`stats/multiple_comparisons_test.sio`](stats/multiple_comparisons_test.sio) | Validation for stats::multiple_comparisons (kept external: an in-module test | pass | pass | 3.5 / 0.5 |
| [`stats/multiple_testing_demo.sio`](stats/multiple_testing_demo.sio) | Demo for stdlib/stats/multiple_testing.sio | fail rc=1 | fail rc=1 (E200) | 3.0 / 0.5 |
| [`stats/permutation_test.sio`](stats/permutation_test.sio) | Validation for stats::permutation. Separated groups -> small p (<lt;0.05); | pass | pass | 3.7 / 0.5 |
| [`stats/resampling_demo.sio`](stats/resampling_demo.sio) | Demo for stdlib/stats/resampling.sio | fail rc=1 | fail rc=1 (E200) | 3.2 / 0.5 |
| [`stats/validation_demo.sio`](stats/validation_demo.sio) | Demo for stdlib/stats/validation.sio | fail rc=1 (E019 E137) | fail rc=1 (E001 E200) | 3.0 / 0.5 |
| [`structs.sio`](structs.sio) | — | pass | pass | 3.1 / 0.6 |
| [`survival_dashboard/main.sio`](survival_dashboard/main.sio) | examples/survival_dashboard/main.sio — Clinical statistics dashboard | crash (signal 54) | timeout | 45.3 / 120.0 |
| [`sync/atomic_demo.sio`](sync/atomic_demo.sio) | Atomic Operations Demonstration | fail rc=1 | fail rc=1 (E035 E200 E001) | 3.2 / 0.6 |
| [`t7_sign_function.sio`](t7_sign_function.sio) | T_7 via sign function — no arrays, no multiplication, pure algebra | pass | pass | 5.8 / 3.0 |
| [`test_array_len.sio`](test_array_len.sio) | Test array_len and array_ptr builtins | fail rc=1 (E137) | fail rc=1 (E200) | 3.2 / 0.5 |
| [`test_array_repeat.sio`](test_array_repeat.sio) | Test array repeat syntax [value; count] | pass | pass | 3.2 / 0.5 |
| [`test_autodiff_exp_log.sio`](test_autodiff_exp_log.sio) | Test autodiff with exp and log | pass | fail rc=1 (E035) | 4.1 / 0.5 |
| [`test_autodiff_hyp_complete.sio`](test_autodiff_hyp_complete.sio) | Test autodiff with remaining hyperbolic functions | pass | fail rc=1 (E035) | 4.3 / 0.6 |
| [`test_autodiff_inverse_hyp.sio`](test_autodiff_inverse_hyp.sio) | Test autodiff with inverse trig and hyperbolic functions | pass | fail rc=1 (E035) | 4.3 / 0.6 |
| [`test_autodiff_log_atan2.sio`](test_autodiff_log_atan2.sio) | Test autodiff with log2, log10, atan2 | pass | fail rc=1 (E035) | 4.3 / 0.6 |
| [`test_autodiff_polynomial.sio`](test_autodiff_polynomial.sio) | Test autodiff: derivative of polynomial f(x) = x^3 + 2x^2 - 5x + 3 | pass | fail rc=1 (E035) | 3.4 / 0.6 |
| [`test_autodiff_sqrt_pow.sio`](test_autodiff_sqrt_pow.sio) | Test autodiff with sqrt and pow | pass | fail rc=1 (E035) | 4.0 / 0.6 |
| [`test_autodiff_tan_atan_abs.sio`](test_autodiff_tan_atan_abs.sio) | Test autodiff with tan, atan, abs | pass | fail rc=1 (E035) | 4.6 / 0.6 |
| [`test_autodiff_tanh_only.sio`](test_autodiff_tanh_only.sio) | Test autodiff with tanh only | pass | fail rc=1 (E035) | 3.6 / 0.6 |
| [`test_autodiff_transcendental.sio`](test_autodiff_transcendental.sio) | Test autodiff with transcendental functions | pass | fail rc=1 (E035) | 3.7 / 0.5 |
| [`test_autodiff_x_squared.sio`](test_autodiff_x_squared.sio) | Test autodiff: compute derivative of x^2 | pass | fail rc=1 (E035) | 3.4 / 0.5 |
| [`test_grad.sio`](test_grad.sio) | Test the grad builtin function for autodiff | fail rc=1 (E137) | fail rc=1 (E200) | 3.0 / 0.5 |
| [`test_grad_comprehensive.sio`](test_grad_comprehensive.sio) | Comprehensive test of the grad builtin function | fail rc=1 (E137) | fail rc=1 (E200) | 3.1 / 0.5 |
| [`test_hessian_real.sio`](test_hessian_real.sio) | Real Hessian computation test | fail rc=1 (E137) | fail rc=1 (E200) | 3.0 / 0.5 |
| [`test_i64_bug_fix.sio`](test_i64_bug_fix.sio) | Test that 0_i64 is now correctly typed as i64 | fail rc=1 (E137) | fail rc=42 | 3.1 / 0.5 |
| [`test_import_qnn.sio`](test_import_qnn.sio) | Test if qnn module can be imported | fail rc=1 | fail rc=1 (E224) | 3.0 / 0.5 |
| [`test_jacobian_codegen.sio`](test_jacobian_codegen.sio) | Test that jacobian and hessian codegen works | fail rc=1 (E137) | fail rc=1 (E200) | 3.0 / 0.5 |
| [`test_jacobian_real.sio`](test_jacobian_real.sio) | Real Jacobian computation test | fail rc=1 (E137) | fail rc=1 (E200) | 3.0 / 0.5 |
| [`test_jacobian_simple.sio`](test_jacobian_simple.sio) | Simple test that jacobian and hessian builtins are recognized | fail rc=1 (E137) | fail rc=1 (E200) | 3.0 / 0.5 |
| [`test_knowledge_type.sio`](test_knowledge_type.sio) | Test Knowledge type syntax | fail rc=1 | fail rc=1 | 3.0 / 0.5 |
| [`test_one_qnn_intrinsic.sio`](test_one_qnn_intrinsic.sio) | Test a single QNN intrinsic | fail rc=1 (E137) | fail rc=1 (E200) | 3.0 / 0.5 |
| [`test_process.sio`](test_process.sio) | Test program for stdlib/os/process.sio | pass | fail rc=1 (E035 E218) | 3.1 / 0.5 |
| [`test_simple_struct.sio`](test_simple_struct.sio) | Test simple struct syntax | pass | pass | 3.1 / 0.5 |
| [`test_sort.sio`](test_sort.sio) | Test program for sorting algorithms | crash (signal 4) | pass | 4.0 / 0.5 |
| [`test_zstd.sio`](test_zstd.sio) | Test program for zstd compression module | fail rc=1 (E137 E011) | fail rc=1 (E200 E006 E001) | 3.0 / 0.6 |
| [`therapy_session.sio`](therapy_session.sio) | examples/therapy_session.sio | fail rc=1 (E259 E006 E137) | fail rc=1 (E200) | 3.0 / 0.5 |
| [`time/mod_demo.sio`](time/mod_demo.sio) | Demo for stdlib/time/mod.sio | fail rc=1 (E011 E015) | fail rc=1 (E200) | 3.0 / 0.5 |
| [`token_erase_benchmark.sio`](token_erase_benchmark.sio) | Token-Erase Benchmark: ZD as Hard-Reset Mechanism | pass | pass | 14.6 / 113.5 |
| [`triple_product_ossm_benchmark.sio`](triple_product_ossm_benchmark.sio) | Triple-Product O-SSM: The Architecture That ACTIVATES Non-Associativity | pass | pass | 14.9 / 75.8 |
| [`typed_literals.sio`](typed_literals.sio) | Typed constants via type annotations (literal suffixes like 42i32 are not supported by Madaros) | fail rc=52 | fail rc=1 (E200 E001) | 3.1 / 0.5 |
| [`uncertainty.sio`](uncertainty.sio) | GUM uncertainty propagation through arithmetic | pass | pass | 3.2 / 0.5 |
| [`unified_closure_demo.sio`](unified_closure_demo.sio) | Sprint 232: Unified Closure Type Theory -- Working Demo | fail rc=1 (E009) | pass | 3.1 / 0.5 |
| [`units/dimensional_report.sio`](units/dimensional_report.sio) | Dimensional-analysis report using stdlib units::lib. | pass | pass | 3.7 / 0.5 |
| [`units/pharma_dose.sio`](units/pharma_dose.sio) | Pharmacological dose calculation with dimensional analysis (Gen 19) | fail rc=1 (E050 E001 E035) | fail rc=1 (E035) | 3.1 / 0.5 |
| [`vancomycin_auc_affine.sio`](vancomycin_auc_affine.sio) | Twin of examples/vancomycin_auc_epistemic.sio. That example uses the builtin | pass | pass | 4.9 / 0.6 |
| [`vancomycin_auc_epistemic.sio`](vancomycin_auc_epistemic.sio) | Epistemic Vancomycin AUC-Guided TDM — GUM-Exact PK Chain | fail rc=1 (E004 E245 E230) | pass | 3.1 / 0.5 |
| [`virus_theorem.sio`](virus_theorem.sio) | virus_theorem.sio | pass | pass | 4.5 / 0.5 |
| [`visual/01_octonion_multiplication_table.sio`](visual/01_octonion_multiplication_table.sio) | Visual Example 1: Octonion Multiplication Table | fail rc=1 (E004 E007) | fail rc=1 (E035) | 3.1 / 0.6 |
| [`visual/02_fmri_octonion_activation.sio`](visual/02_fmri_octonion_activation.sio) | Visual Example 2: fMRI Brain Activation with Octonion Processing | fail rc=1 (E007 E004 E008) | fail rc=1 (E035 E001) | 3.1 / 0.5 |
| [`visual/03_octonion_network_efficiency.sio`](visual/03_octonion_network_efficiency.sio) | Visual Example 3: Octonion Neural Network Parameter Efficiency | fail rc=1 (E007 E004) | fail rc=1 (E035) | 3.2 / 0.5 |
| [`visual/04_fmri_color_heatmap.sio`](visual/04_fmri_color_heatmap.sio) | Visual Example 4: fMRI Brain Activation with TRUE COLOR | fail rc=1 (E004 E008 E007) | fail rc=1 (E035) | 3.1 / 0.5 |
| [`visual/05_color_demo.sio`](visual/05_color_demo.sio) | Visual Example 5: ANSI Color Demo - Terminal Colors in Sounio! | pass | pass | 3.3 / 0.5 |
| [`visual/06_octonion_color_table.sio`](visual/06_octonion_color_table.sio) | Visual Example 6: Octonion Multiplication Table - COLOR EDITION | pass | fail rc=1 (E035) | 4.8 / 0.5 |
| [`visual/07_epistemic_uncertainty_bars.sio`](visual/07_epistemic_uncertainty_bars.sio) | Visual Example 7: Epistemic Uncertainty Visualization | fail rc=1 (E004) | fail rc=1 (E035) | 3.0 / 0.5 |
| [`visual/08_climate_ensemble_color.sio`](visual/08_climate_ensemble_color.sio) | Visual Example 8: Climate Model Ensemble - Color Edition | fail rc=1 (E004) | fail rc=1 (E035) | 3.1 / 0.6 |
| [`visual/09_pkpd_color_curves.sio`](visual/09_pkpd_color_curves.sio) | Visual Example 9: PK/PD Concentration-Time Curves - Color Edition | fail rc=1 (E004) | fail rc=1 (E035) | 3.0 / 0.5 |
| [`visual/10_kalman_filter_color.sio`](visual/10_kalman_filter_color.sio) | Visual Example 10: Bayesian Sensor Fusion with Kalman Filter - Color Edition | fail rc=1 (E004) | fail rc=1 (E035) | 3.0 / 0.5 |
| [`visual/11_sir_epidemic_color.sio`](visual/11_sir_epidemic_color.sio) | Visual Example 11: SIR Epidemic Model - Color Edition | fail rc=1 (E004) | fail rc=1 (E035) | 3.0 / 0.5 |
| [`visual/12_animated_diffusion.sio`](visual/12_animated_diffusion.sio) | Visual Example 12: Animated Heat Diffusion - Frame-Based Animation | pass | fail rc=1 (E035) | 3.6 / 0.5 |
| [`viz_3d/main.sio`](viz_3d/main.sio) | examples/viz_3d/main.sio — Rotating 3D tetrahedron with Phong shading | crash (signal 54) | timeout | 27.3 / 120.0 |
| [`viz_hello/main.sio`](viz_hello/main.sio) | examples/viz_hello/main.sio — smallest Visual IR Canvas example | pass | pass | 12.2 / 1.6 |
| [`viz_html_export/main.sio`](viz_html_export/main.sio) | examples/viz_html_export/main.sio — static HTML/SVG export over Visual IR | pass | pass | 10.1 / 1.3 |
| [`viz_lab/main.sio`](viz_lab/main.sio) | examples/viz_lab/main.sio — native visual frontend smoke demo | fail rc=1 | pass | 14.0 / 1.7 |
| [`viz_lab_window/main.sio`](viz_lab_window/main.sio) | examples/viz_lab_window/main.sio - optional native Visual IR window demo | crash (signal 54) | crash (signal 11) | 22.4 / 1.7 |
| [`viz_pbpk/main.sio`](viz_pbpk/main.sio) | examples/viz_pbpk/main.sio — PK uncertainty visualization demo | timeout | timeout | 120.0 / 120.0 |
| [`viz_physchem_demo/main.sio`](viz_physchem_demo/main.sio) | examples/viz_physchem_demo/main.sio - Physico-chemistry Visual IR demo | pass | pass | 16.7 / 1.8 |
| [`viz_workbench/main.sio`](viz_workbench/main.sio) | examples/viz_workbench/main.sio - Sounio native visual workbench demo | pass | pass | 17.8 / 1.9 |
| [`wave1_hello_scientific.sio`](wave1_hello_scientific.sio) | Wave 1: Simple Hello Scientific Computing | fail rc=1 (E137) | fail rc=1 (E200) | 2.9 / 0.5 |
| [`wave2_causal_intervention.sio`](wave2_causal_intervention.sio) | Wave 2 Example: Causal Intervention with do-calculus | pass | pass | 3.0 / 0.5 |
| [`wave2_causal_simpson.sio`](wave2_causal_simpson.sio) | Wave 2 Example: Simpson's Paradox in Causal Inference | fail rc=1 (E010) | pass | 2.9 / 0.5 |
| [`wave3_symbolic_calculus.sio`](wave3_symbolic_calculus.sio) | Wave 3 Example: Symbolic Calculus | pass | pass | 3.1 / 0.5 |
| [`zd_antiwindup_pid.sio`](zd_antiwindup_pid.sio) | ZD Anti-Windup PID Controller | pass | fail rc=1 | 5.6 / 0.5 |
| [`zd_audit_witness.sio`](zd_audit_witness.sio) | ZD Audit Witness Emission (G9, Audited<lt;T>) | pass | pass | 3.3 / 0.5 |
| [`zd_bptt_ssm.sio`](zd_bptt_ssm.sio) | ZD-SSM BPTT: Backpropagation Through Sedenion Multiplication | pass | pass | 10.6 / 111.2 |
| [`zd_capability_removal.sio`](zd_capability_removal.sio) | ZD-based Exact Capability Removal (G7, AI Safety) | pass | pass | 5.7 / 0.6 |
| [`zd_capacity_constrained.sio`](zd_capacity_constrained.sio) | ZD Capacity-Constrained Forgetting Benchmark | pass | pass | 8.8 / 32.5 |
| [`zd_constructive_reset_proof.sio`](zd_constructive_reset_proof.sio) | ZD Constructive Reset Proof | pass | pass | 5.9 / 0.6 |
| [`zd_epistemic_collapse.sio`](zd_epistemic_collapse.sio) | ZD Epistemic Collapse: Hard Constraints vs Ghost Beliefs | pass | pass | 5.8 / 0.9 |
| [`zd_forgettable_training.sio`](zd_forgettable_training.sio) | examples/zd_forgettable_training.sio | fail rc=1 (E035) | pass | 3.1 / 0.5 |
| [`zd_machine_unlearning.sio`](zd_machine_unlearning.sio) | ZD-based Exact Machine Unlearning (G3, RTBF / GDPR Art. 17) | pass | pass | 5.2 / 0.5 |
| [`zd_model_composition.sio`](zd_model_composition.sio) | ZD-Orthogonal Model Composition (G8, Composable<lt;T>) | pass | pass | 3.6 / 0.5 |
| [`zd_model_editing_locality.sio`](zd_model_editing_locality.sio) | ZD-based Exact Model Editing with Locality Guarantee (G5, ROME ripple) | pass | pass | 5.0 / 0.5 |
| [`zd_ratio_sweep.sio`](zd_ratio_sweep.sio) | ZD-SSM vs H-SSM: accuracy as a function of TEMP/FRESH signal ratio | pass | timeout | 26.0 / 120.0 |
| [`zd_revivable_edit.sio`](zd_revivable_edit.sio) | Time-Bounded Reversible Surgery (G10, Revivable<lt;T>) | pass | pass | 3.2 / 0.5 |
| [`zd_vs_capacity_ablation.sio`](zd_vs_capacity_ablation.sio) | ZD vs Capacity: The Killer Experiment | pass | timeout | 20.7 / 120.0 |
