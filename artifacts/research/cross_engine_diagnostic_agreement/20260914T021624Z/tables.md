### Verdicts over 8232 files

| Verdict | Files |
|---|---:|
| AGREE_ACCEPT | 3113 |
| LEAN_ONLY | 2440 |
| AGREE_REJECT | 1921 |
| MADAROS_ONLY | 726 |
| TIMEOUT | 32 |

### Timeouts by exit codes (lean_rc / madaros_rc; 124 = timed out)

| lean_rc | madaros_rc | Files |
|---:|---:|---:|
| 124 | 124 | 21 |
| 124 | 1 | 5 |
| 124 | 0 | 5 |
| 0 | 124 | 1 |

### Disagreements by class

| Class | LEAN_ONLY | MADAROS_ONLY | Total |
|---|---:|---:|---:|
| HARNESS | 919 | 0 | 919 |
| INTENTIONAL | 749 | 36 | 785 |
| DEFECT-LEAN | 566 | 38 | 604 |
| DEFECT-MADAROS | 109 | 330 | 439 |
| MADAROS-GUARANTEE | 0 | 250 | 250 |
| UNRESOLVED | 37 | 67 | 104 |
| BOTH | 60 | 0 | 60 |
| SOURCE-BUG | 0 | 5 | 5 |

### Groups

| Verdict | Class | Group | Files | Example |
|---|---|---|---:|---|
| LEAN_ONLY | HARNESS | no main: lean check is a full compile | 919 | `artifacts/ontology-frontiers/compiler-repros/p5_qualified_leaf.sio` |
| LEAN_ONLY | INTENTIONAL | test declares requires: madaros (lean: E035 Mut demanded for function-local mutation) | 338 | `tests/run-pass/cardinality_at_most_two_tiny.sio` |
| LEAN_ONLY | DEFECT-LEAN | E035 Mut demanded for function-local mutation | 337 | `artifacts/ontology-frontiers/compiler-repros/p4_thinlink_control_leaf.sio` |
| LEAN_ONLY | INTENTIONAL | test declares requires: madaros (lean: re-export (pub use) not followed by lean) | 313 | `tests/run-pass/lorenz_i256_ball_fixed_bridge_imported.sio` |
| MADAROS_ONLY | MADAROS-GUARANTEE | private fn used across modules (source bug exposed) | 100 | `demos/hydrogen/uhs_brine_calcite.sio` |
| MADAROS_ONLY | MADAROS-GUARANTEE | missing effect on a caller of an effectful fn (Madaros E035; lean does not report) | 98 | `artifacts/zd-ssm/inference.sio` |
| LEAN_ONLY | DEFECT-LEAN | lean analyses code inside /* */ (stub files) | 70 | `examples/alpha_geo_zero.sio` |
| LEAN_ONLY | DEFECT-MADAROS | E035 caller-observable mutation or IO without the effect (documented Madaros gap) | 64 | `artifacts/ontology-frontiers/real-data/scale/dense_full_data.sio` |
| LEAN_ONLY | BOTH | E035 mixed: local-var sites (lean wrong) and caller-observable sites (Madaros Mut/IO gap) | 58 | `examples/lethal_dose_sedenion.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | method calls on non-struct receivers (E019) | 56 | `examples/epistemic/rk4_correlated_uncertainty.sio` |
| LEAN_ONLY | INTENTIONAL | Knowledge annotation surface (audit 2026-08-19) | 44 | `docs/audit/exactly_private_ta/witness_knowledge_epsilon.sio` |
| MADAROS_ONLY | UNRESOLVED | module root differs (lean also searches self-hosted/) | 40 | `explore_bench_clark_unit_test.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | parse failure with no diagnostic printed | 27 | `examples/causal_model.sio` |
| LEAN_ONLY | DEFECT-LEAN | comma inside generic type arguments counted as a parameter separator | 24 | `tests/gpu/oct_assoc_wmma_tile.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | builtin used by tracked code missing (append_file) | 23 | `stdlib/graphics/animate.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | run-pass test rejected by Madaros (E001) | 18 | `stdlib/units/astronomical.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | run-pass test rejected by Madaros (E009) | 16 | `tests/packages/package_import_science_witness.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | Knowledge op Knowledge declared unsupported (E245) | 16 | `examples/epistemic_smoke_native.sio` |
| LEAN_ONLY | DEFECT-LEAN | reference to a call result (&f()) | 16 | `examples/conversational_ossm/bidirectional_ossm_v0.sio` |
| MADAROS_ONLY | INTENTIONAL | documented reservations and fixtures (E249, E250) [documented known-failure] | 15 | `tests/compile-fail/f128_cast_from_f64_unimplemented.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | run-pass test rejected by Madaros (E137) | 15 | `tests/run-pass/async_join.sio` |
| LEAN_ONLY | INTENTIONAL | test declares requires: madaros (lean: E035 mixed: local-var sites (lean wrong) and caller-observable sites (Madaros Mut/IO gap)) | 15 | `tests/run-pass/lorenz_i256_bit_budget_tiny.sio` |
| LEAN_ONLY | DEFECT-LEAN | Madaros builtin absent from lean_single (second_order_mean, correlate) | 15 | `examples/epistemic_fo_second_order/fo_pk_import_auc_thalf_driver.sio` |
| MADAROS_ONLY | INTENTIONAL | documented reservations and fixtures (E249, E250) | 13 | `examples/numerics/f128_is_f64_probe.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | run-pass test rejected by Madaros (E008) | 12 | `examples/dissertation_steady_state_demo.sio` |
| LEAN_ONLY | UNRESOLVED | field privacy (lean enforces; Madaros has no check; docs silent) | 12 | `self-hosted/gpu/test_kaxi_differential.sio` |
| LEAN_ONLY | DEFECT-LEAN | run-pass test rejected by lean -- cause: contest expression not supported | 12 | `tests/frontend/chain_validated_param_contest.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | typed int literal in array repeat | 11 | `examples/data/dataframe_groupby_workflow.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | read_file result untyped (.as_bytes) | 11 | `examples/brain_hessian_abide.sio` |
| MADAROS_ONLY | UNRESOLVED | imported module does not parse in Madaros (not isolated) | 10 | `examples/image/canvas_field.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | run-pass test rejected by Madaros (E004) | 10 | `tests/run-pass/algebra_g2_null_model.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | compile-time division-by-zero check ignores reassignment and loop guard | 10 | `tests/stdlib/data/test_csv_f64.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | run-pass test rejected by Madaros (run_check_mode: AST closure incomplete nodes=N unresolved=N ) | 9 | `tests/stdlib/analysis/test_analysis.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | run-pass test rejected by Madaros (E037) | 9 | `tests/run-pass/d4_optimizer_integration.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | builtin used by tracked code missing (ln) | 9 | `examples/image/fractal_512.sio` |
| LEAN_ONLY | DEFECT-LEAN | struct name resolved by bare name across modules | 9 | `self-hosted/check/effects.sio` |
| MADAROS_ONLY | MADAROS-GUARANTEE | effectful fn ref passed as an effect-free fn type | 8 | `benchmarks/humaneval_sounio/028_trapezoidal_rule.sio` |
| MADAROS_ONLY | INTENTIONAL | Knowledge annotation surface (audit 2026-08-19) | 8 | `docs/audit/probes/knowledge-annotation-parser-coverage-2026-08-19/angle_literature.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | float-literal array into an f32 array | 8 | `examples/qnn/01_hello_quaternion.sio` |
| LEAN_ONLY | INTENTIONAL | test declares requires: madaros (lean: reference to a call result (&f())) | 8 | `tests/run-pass/a_mu_cmd3_2pi_split.sio` |
| LEAN_ONLY | INTENTIONAL | test declares requires: madaros (lean: Madaros builtin absent from lean_single (second_order_mean, correlate)) | 8 | `tests/run-pass/drift_gate_second_order_mean.sio` |
| LEAN_ONLY | DEFECT-MADAROS | named import of a function the module does not define (Madaros accepts) | 8 | `examples/particle_physics/exp7_gum_xi_tension_transfer.sio` |
| LEAN_ONLY | DEFECT-LEAN | reference to an indexed element as call argument | 8 | `examples/graphics/demos/phase2_showcase.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | run-pass test rejected by Madaros (E012) | 7 | `tests/run-pass/async_basic.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | imported user-declared effect not resolved (E246) | 7 | `tests/compiler/fleet_transaction_privacy/fleet_transaction_anchor_prefix_mismatch.sio` |
| MADAROS_ONLY | DEFECT-LEAN | negative probe accepted by lean (expected rejection by path or name) | 7 | `tests/frontend/parser_stability/invalid/bad_fn_signature.sio` |
| LEAN_ONLY | DEFECT-LEAN | a sqrt method on an imported type shadows the free sqrt(f64) | 7 | `stdlib/particle_physics/amplitude.sio` |
| MADAROS_ONLY | MADAROS-GUARANTEE | error in an imported module that lean_single does not report | 6 | `scripts/ci/fixtures/let_ann_multimodule_main.sio` |
| MADAROS_ONLY | DEFECT-LEAN | compile-fail test accepted by lean (Madaros: E001) [declared error-pattern matched] | 6 | `tests/compile-fail/annotation_diffprivate_mismatch.sio` |
| LEAN_ONLY | DEFECT-MADAROS | compile-fail test accepted by Madaros (lean: <no diagnostic line; last: typecheck: failed>) [declared error-pattern matched] | 6 | `tests/compile-fail/gtt_body_precision_unused_param_refused.sio` |
| LEAN_ONLY | DEFECT-LEAN | contest expression not supported | 6 | `tests/frontend/contest_ir_authority_instrumented.sio` |
| MADAROS_ONLY | UNRESOLVED | builtin print_char argument type (u8 vs i64), docs silent | 5 | `repro/exact_orc/en_edge_native.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | builtin used by tracked code missing (print_i64) | 5 | `examples/dissertation_scenario_gate_demo.sio` |
| LEAN_ONLY | INTENTIONAL | test declares requires: madaros (lean: re-export (pub use) not followed by lean) [documented known-failure] | 5 | `tests/run-pass/lorenz_i256_cover_child1_obligation_seed_imported.sio` |
| LEAN_ONLY | DEFECT-MADAROS | value returned from a unit fn (Madaros accepts) | 5 | `docs/audit/EPISTEMIC_MADAROS_SIGSEGV_2026-06-29/reference/epistemic_bmi_f64_println_probe.sio` |
| LEAN_ONLY | DEFECT-LEAN | type alias inside a tuple return type | 5 | `docs/audit/r2_7_pcg_state_unify/reference/core_alias_smoke.sio` |
| LEAN_ONLY | DEFECT-LEAN | import X::* not supported | 5 | `artifacts/ontology-frontiers/real-data/real_repair_driver.sio` |
| LEAN_ONLY | DEFECT-LEAN | data-carrying enum variant V::M { .. } | 5 | `docs/handoff/repros/d4_enum_match.sio` |
| MADAROS_ONLY | SOURCE-BUG | loop used as an identifier (reserved keyword) | 4 | `docs/audit/r3_0_transitive_alias_chain/reference/transitive_main.sio` |
| MADAROS_ONLY | MADAROS-GUARANTEE | compile-fail test declares requires: madaros (E040) [declared error-pattern matched] | 4 | `tests/compile-fail/clinical_d12_observation_cannot_be_discarded_d12.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | run-pass test rejected by Madaros (E036) | 4 | `tests/run-pass/epistemic_guarded_nested_measure_call.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | if-branch join (literal adoption or statement position) | 4 | `self-hosted/compiler/lean_single.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | Knowledge op Knowledge declared unsupported (E245) [documented known-failure] | 4 | `tests/known_failures/madaros_gum_fo_knowledge_ops.sio` |
| LEAN_ONLY | UNRESOLVED | module root differs (lean resolves imports under self-hosted/ or the parent directory) | 4 | `tests/compiler/exp_cos_import_collision/main_cos.sio` |
| LEAN_ONLY | DEFECT-MADAROS | compile-fail test accepted by Madaros (lean: E208) [declared error-pattern matched] | 4 | `tests/compile-fail/refinement_guard_and_escape.sio` |
| LEAN_ONLY | DEFECT-LEAN | re-export (pub use) not followed by lean | 4 | `tests/compiler/pub_use_reexport/missing_consumer.sio` |
| LEAN_ONLY | DEFECT-LEAN | loop { } not supported | 4 | `examples/render/causal_dag.sio` |
| LEAN_ONLY | DEFECT-LEAN | int-literal array as fn tail | 4 | `examples/graphics/lib/00_canvas_simple.sio` |
| MADAROS_ONLY | MADAROS-GUARANTEE | negative controls | 3 | `docs/audit/exactly_private_ta/witness_ep_two_args.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | run-pass test rejected by Madaros (E015) | 3 | `tests/run-pass/algebra_commutative_default.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | run-pass test rejected by Madaros (E011) | 3 | `tests/run-pass/async_channels.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | refinement returned as its base type | 3 | `docs/audit/repro/refinement_div_discharge/D2_gt5_implies_gt0.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | char literal as array-repeat element does not parse | 3 | `examples/epistemic_viz_demo.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | builtin used by tracked code missing (read_line) | 3 | `artifacts/lsp_bringup/sb6.sio` |
| MADAROS_ONLY | DEFECT-LEAN | compile-fail test accepted by lean (Madaros: parse error: expected token at line N expected=T actual=T) [declared error-pattern matched] | 3 | `tests/ui/parser/fn_no_name.sio` |
| LEAN_ONLY | INTENTIONAL | test declares requires: madaros (lean: private fn names collide across modules) | 3 | `tests/run-pass/gum_fo_analytic_transcendentals.sio` |
| LEAN_ONLY | DEFECT-MADAROS | compile-fail test accepted by Madaros (lean: E214) [declared error-pattern matched] | 3 | `tests/compile-fail/confidence_gate_reject.sio` |
| LEAN_ONLY | DEFECT-LEAN | tuple index > 1 | 3 | `experiments/faers_fano_order_asymmetry/faers_fano_order_asymmetry_v2.sio` |
| LEAN_ONLY | DEFECT-LEAN | run-pass test rejected by lean -- cause: qualified path call a::b::f() not supported | 3 | `tests/regression/test_simple.sio` |
| LEAN_ONLY | DEFECT-LEAN | array var declared without initializer | 3 | `artifacts/ontology-frontiers/compiler-repros/struct_array_segfault.sio` |
| MADAROS_ONLY | MADAROS-GUARANTEE | var declared inside a block, reused after it (block scope; source bug) | 2 | `examples/octonion_168_associators.sio` |
| MADAROS_ONLY | MADAROS-GUARANTEE | independence-assuming op over correlated uncertainty | 2 | `examples/hyperbolic_semantic_networks/affect_network_orc.sio` |
| MADAROS_ONLY | MADAROS-GUARANTEE | f64 result bound to f32 (source bug) | 2 | `examples/oct_associator_ddi.sio` |
| MADAROS_ONLY | MADAROS-GUARANTEE | f64 field returned as f32 (source bug) | 2 | `examples/oct_associator_direction.sio` |
| MADAROS_ONLY | MADAROS-GUARANTEE | compile-fail test declares requires: madaros (E252) [declared error-pattern matched] | 2 | `tests/compile-fail/e252_fano_selective_on_sedenion.sio` |
| MADAROS_ONLY | MADAROS-GUARANTEE | compile-fail test declares requires: madaros (E246) [declared error-pattern matched] | 2 | `tests/compile-fail/effect_ninth_slot_refused.sio` |
| MADAROS_ONLY | MADAROS-GUARANTEE | compile-fail test declares requires: madaros (E042) [declared error-pattern matched] | 2 | `tests/compile-fail/refinement_holds_at_implicit_tail.sio` |
| MADAROS_ONLY | MADAROS-GUARANTEE | compile-fail test declares requires: madaros (E010) [declared error-pattern matched] | 2 | `tests/compile-fail/epistemic_arity_must_refuse.sio` |
| MADAROS_ONLY | MADAROS-GUARANTEE | builtin Knowledge passed where KCoreKnowledge is declared (source bug) | 2 | `packages/epistemic-core/tests/test_gum_propagation.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | underscore literal suffix (documented 500_mg) not lexed | 2 | `examples/test_i64_bug_fix.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | str alias missing (&str) | 2 | `demos/hydrogen/site_screening.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | run-pass test rejected by Madaros (E221) | 2 | `tests/run-pass/epistemic_hessian_transcendentals.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | run-pass test rejected by Madaros (E038) | 2 | `tests/run-pass/proof_search_basic.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | run-pass test rejected by Madaros (E010) | 2 | `tests/math/test_zd_deep_dive.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | fn-return where clause does not parse | 2 | `artifacts/ontology-frontiers/compiler-repros/where_refinement_parse.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | compound unit f64<a/b> does not parse | 2 | `demo_unidades.sio` |
| MADAROS_ONLY | DEFECT-LEAN | unresolved pub use silently accepted by lean | 2 | `examples/ontology/biomedical/mod_demo.sio` |
| MADAROS_ONLY | DEFECT-LEAN | compile-fail test accepted by lean (Madaros: E247) [declared error-pattern matched] | 2 | `tests/audit/zd_mut_spine/exactly_private_locus_malformed.sio` |
| MADAROS_ONLY | DEFECT-LEAN | compile-fail test accepted by lean (Madaros: E063) [documented known-failure] [no declared error-pattern] | 2 | `tests/compile-fail/contest_model_arity_reject.sio` |
| MADAROS_ONLY | DEFECT-LEAN | compile-fail test accepted by lean (Madaros: E057) [declared error-pattern matched] | 2 | `tests/ui/type/modulo_by_zero_assign_const.sio` |
| MADAROS_ONLY | DEFECT-LEAN | compile-fail test accepted by lean (Madaros: E056) [declared error-pattern matched] | 2 | `tests/ui/type/division_by_zero_assign_const.sio` |
| LEAN_ONLY | UNRESOLVED | extern "C" system(cmd: string) called with a literal (lean E001; binding types differ) | 2 | `docs/audit/p0f_repros/system_string_binding_works.sio` |
| LEAN_ONLY | UNRESOLVED | borrows of disjoint fields of one struct (lean tracks per variable; docs silent) | 2 | `stdlib/crypto/hash.sio` |
| LEAN_ONLY | UNRESOLVED | Knowledge<T where {..}>: lean static check vs runtime-guard expansion | 2 | `demo_portas_rejeicao.sio` |
| LEAN_ONLY | INTENTIONAL | test declares requires: madaros (lean: wide integers (i128/i256/i512) unsupported) | 2 | `tests/run-pass/r1_i256_lorenz_peak.sio` |
| LEAN_ONLY | INTENTIONAL | test declares requires: madaros (lean: error: logical not requires bool operand at <main>:N) | 2 | `tests/run-pass/parser_spaced_bang_expression.sio` |
| LEAN_ONLY | DEFECT-MADAROS | compile-fail test accepted by Madaros (lean: error: match must be exhaustive at <main>:N) [declared error-pattern matched] | 2 | `tests/compile-fail/nested_match_outer_non_exhaustive.sio` |
| LEAN_ONLY | DEFECT-MADAROS | compile-fail test accepted by Madaros (lean: error: linear value not consumed at <main>:N) [declared error-pattern matched] | 2 | `tests/compile-fail/linear_early_return.sio` |
| LEAN_ONLY | DEFECT-MADAROS | compile-fail test accepted by Madaros (lean: E209) [declared error-pattern matched] | 2 | `tests/compile-fail/refinement_violation_probability.sio` |
| LEAN_ONLY | DEFECT-MADAROS | compile-fail test accepted by Madaros (lean: <no diagnostic line; last: typecheck: failed>) [no declared error-pattern] | 2 | `tests/compile-fail/gtt_loop_wrong_channel.sio` |
| LEAN_ONLY | DEFECT-LEAN | wide integers (i128/i256/i512) unsupported | 2 | `tests/typekind/i128/pass.sio` |
| LEAN_ONLY | DEFECT-LEAN | run-pass test rejected by lean (E200) | 2 | `tests/run-pass/parser_card_a_regression.sio` |
| LEAN_ONLY | DEFECT-LEAN | run-pass test rejected by lean (E200) [documented known-failure] | 2 | `tests/run-pass/handler_discharge.sio` |
| LEAN_ONLY | DEFECT-LEAN | qualified path call a::b::f() not supported | 2 | `stdlib/database/pure/engine.sio` |
| LEAN_ONLY | DEFECT-LEAN | int-literal list into [i8; N] | 2 | `stdlib/epistemic/audited.sio` |
| LEAN_ONLY | DEFECT-LEAN | int-literal array as call argument | 2 | `self-hosted/native/dwarf.sio` |
| LEAN_ONLY | DEFECT-LEAN | empty array literal [] for an unsized field or local | 2 | `stdlib/search/mcts/node.sio` |
| MADAROS_ONLY | UNRESOLVED | whole-module use a::b (unresolved in Madaros; lean accepts) | 1 | `stdlib/chemistry/lib.sio` |
| MADAROS_ONLY | UNRESOLVED | typed literal suffix 0i64 in stdlib/cybernetic/distinction.sio:303 (both engines reject it in isolation) | 1 | `tests/stdlib/cybernetic/test_distinction_stdlib.sio` |
| MADAROS_ONLY | UNRESOLVED | test marked //@ ignore | 1 | `tests/run-pass/g2_abide_sounio.sio` |
| MADAROS_ONLY | UNRESOLVED | open-range slice &b[..k] does not parse in Madaros; no harness contract | 1 | `tests/selfhost/gpu_runtime/test_gpu_runtime_dynamic_slice.sio` |
| MADAROS_ONLY | UNRESOLVED | module root differs (lean also searches self-hosted/) [documented known-failure] | 1 | `tests/run-pass/ptx_maxntid.sio` |
| MADAROS_ONLY | UNRESOLVED | explicit generic call f<T>(x) syntax (Madaros E137); turbofish ::<T> is the form other tests use | 1 | `tests/selfhost/native_runtime/generic_explicit_fn_id_42.sio` |
| MADAROS_ONLY | UNRESOLVED | explicit generic call f<T>(x) syntax (Madaros E137) | 1 | `tests/selfhost/native_runtime/generic_fn_id_i32_42.sio` |
| MADAROS_ONLY | UNRESOLVED | E038 exclusive borrow while other borrows active (Madaros borrow rule; docs silent) | 1 | `stdlib/theorem/search.sio` |
| MADAROS_ONLY | UNRESOLVED | E036 confidence bound not tight enough (not isolated) | 1 | `stdlib/research/erdos90_epistemic.sio` |
| MADAROS_ONLY | UNRESOLVED | E009 passing fn ref sobol_model_pbpk28 (not isolated) | 1 | `stdlib/darwin_pbpk/validation/pbpk28_sobol_pce.sio` |
| MADAROS_ONLY | UNRESOLVED | E009 on orc_load_csv(..) (definition not found) | 1 | `repro/exact_orc/en_edge_smt_cert.sio` |
| MADAROS_ONLY | UNRESOLVED | E004 on `result > 0.0` in stdlib/stats/inferential.sio:688 (types not isolated) | 1 | `tests/stdlib/stats/test_spearman_stdlib.sio` |
| MADAROS_ONLY | SOURCE-BUG | Rust format arguments in print | 1 | `examples/wave2_causal_simpson.sio` |
| MADAROS_ONLY | MADAROS-GUARANTEE | provenance-only Knowledge boundary mismatch lean_single does not report (the probe documents the silence) | 1 | `tests/audit/knowledge_provenance_boundary_silence_lean.sio` |
| MADAROS_ONLY | MADAROS-GUARANTEE | independence-assuming op over correlated uncertainty [documented known-failure] | 1 | `tests/run-pass/ns_unknown_absorb_trace.sio` |
| MADAROS_ONLY | MADAROS-GUARANTEE | i64 compared to an enum variant (source bug) | 1 | `examples/native/enum_match.sio` |
| MADAROS_ONLY | MADAROS-GUARANTEE | f64 * f32 without conversion (source bug) | 1 | `examples/cognitive_ossm/cognitive_ossm.sio` |
| MADAROS_ONLY | MADAROS-GUARANTEE | compile-fail test declares requires: madaros (error: at tests/compile-fail/incomplete_assignment_at_eof.si) [declared error-pattern matched] | 1 | `tests/compile-fail/incomplete_assignment_at_eof.sio` |
| MADAROS_ONLY | MADAROS-GUARANTEE | compile-fail test declares requires: madaros (E242) [declared error-pattern matched] | 1 | `tests/compile-fail/epistemic_index_must_refuse.sio` |
| MADAROS_ONLY | MADAROS-GUARANTEE | compile-fail test declares requires: madaros (E232) [declared error-pattern matched] | 1 | `tests/compile-fail/shared_array_initialiser_must_refuse.sio` |
| MADAROS_ONLY | MADAROS-GUARANTEE | compile-fail test declares requires: madaros (E221) [declared error-pattern matched] | 1 | `tests/compile-fail/native_math_without_backend_builtin.sio` |
| MADAROS_ONLY | MADAROS-GUARANTEE | compile-fail test declares requires: madaros (E178) [declared error-pattern matched] | 1 | `tests/compile-fail/ns_source_cap_overflow_loud.sio` |
| MADAROS_ONLY | MADAROS-GUARANTEE | compile-fail test declares requires: madaros (E050) [declared error-pattern matched] | 1 | `tests/compile-fail/ns_rem_result_is_opaque.sio` |
| MADAROS_ONLY | MADAROS-GUARANTEE | compile-fail test declares requires: madaros (E001) [declared error-pattern not in the rejecting engine's output] | 1 | `tests/compile-fail/array_repeat_i8_binding_type_mismatch.sio` |
| MADAROS_ONLY | MADAROS-GUARANTEE | builtin Knowledge passed where &KCoreKnowledge is declared (source bug) | 1 | `packages/epistemic-core/src/examples/basic_measurement.sio` |
| MADAROS_ONLY | MADAROS-GUARANTEE | CI fixture built to be rejected by Madaros | 1 | `scripts/ci/fixtures/madaros_intrinsic_knowledge_type/reject_nonknowledge.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | run-pass test rejected by Madaros -- cause: builtin used by tracked code missing (atan2) | 1 | `tests/run-pass/epistemic_hessian_two_arg.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | run-pass test rejected by Madaros (parse error: expected token at line N expected=T actual=T) | 1 | `tests/stdlib/runtime_regression/runtime_dynamic_slice.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | run-pass test rejected by Madaros (E178) [documented known-failure] | 1 | `tests/run-pass/ns_source_cap_unknown_trace.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | run-pass test rejected by Madaros (E042) | 1 | `tests/run-pass/refinement_nested_arithmetic.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | run-pass test rejected by Madaros (E014) | 1 | `tests/run-pass/slice_fat_pointers.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | run-pass test rejected by Madaros (E013) | 1 | `tests/run-pass/test_spectral.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | run-pass test rejected by Madaros (E006) | 1 | `tests/run-pass/if_let_pattern.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | run-pass test rejected by Madaros (E005) | 1 | `tests/run-pass/bitwise_not_bootstrap_regression.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | field store through *mut by auto-deref (repro of known gap) | 1 | `repro/nested_autoderef_field_store_min.sio` |
| MADAROS_ONLY | DEFECT-MADAROS | builtin used by tracked code missing (atan2) | 1 | `stdlib/graphics/pie.sio` |
| MADAROS_ONLY | DEFECT-LEAN | empty match accepted by lean (error-audit probe) | 1 | `tests/error_audit/18_match_no_patterns.sio` |
| MADAROS_ONLY | DEFECT-LEAN | compile-fail test accepted by lean -- cause: imported user-declared effect not resolved (E246) [declared error-pattern matched] | 1 | `tests/compiler/loom_continuity_privacy/loom_continuity_unsealed_admission_main.sio` |
| MADAROS_ONLY | DEFECT-LEAN | compile-fail test accepted by lean (Madaros: parse error: expected token at line N expected=T actual=T) [declared error-pattern not in the rejecting engine's output] | 1 | `tests/ui/parser/missing_type_annotation.sio` |
| MADAROS_ONLY | DEFECT-LEAN | compile-fail test accepted by lean (Madaros: error: at tests/ui/parser/invalid_expr.sio:6:11 - syntax app) [declared error-pattern not in the rejecting engine's output] | 1 | `tests/ui/parser/invalid_expr.sio` |
| MADAROS_ONLY | DEFECT-LEAN | compile-fail test accepted by lean (Madaros: E200) [declared error-pattern matched] | 1 | `tests/audit/zd_mut_spine/forgettable_param_nozd.sio` |
| MADAROS_ONLY | DEFECT-LEAN | compile-fail test accepted by lean (Madaros: E177) [declared error-pattern matched] | 1 | `tests/multimodule/visibility_enum_private_main.sio` |
| MADAROS_ONLY | DEFECT-LEAN | compile-fail test accepted by lean (Madaros: E137) [declared error-pattern not in the rejecting engine's output] | 1 | `tests/compile-fail/study_call_without_audit.sio` |
| MADAROS_ONLY | DEFECT-LEAN | compile-fail test accepted by lean (Madaros: E042) [declared error-pattern matched] | 1 | `tests/compile-fail/multitest_no_correction.sio` |
| MADAROS_ONLY | DEFECT-LEAN | compile-fail test accepted by lean (Madaros: E020) [declared error-pattern not in the rejecting engine's output] | 1 | `tests/compile-fail/parser_card_a_const_dim_name_checker_boundary.sio` |
| MADAROS_ONLY | DEFECT-LEAN | compile-fail test accepted by lean (Madaros: E018) [declared error-pattern matched] | 1 | `tests/ui/type/match_arm_mismatch.sio` |
| MADAROS_ONLY | DEFECT-LEAN | compile-fail test accepted by lean (Madaros: E010) [documented known-failure] [declared error-pattern matched] | 1 | `tests/compile-fail/turbofish_type_arg_arity.sio` |
| MADAROS_ONLY | DEFECT-LEAN | compile-fail test accepted by lean (Madaros: E009) [declared error-pattern matched] | 1 | `tests/ui/resolve/undefined_type.sio` |
| LEAN_ONLY | UNRESOLVED | unit-literal arithmetic `dose_rate * 24.0<h>` with `mg/L` params | 1 | `examples/pharmacokinetics.sio` |
| LEAN_ONLY | UNRESOLVED | module-level `let` string constant as a field initializer (lean mismatch, not isolated) | 1 | `stdlib/hardware/launch.sio` |
| LEAN_ONLY | UNRESOLVED | module search path (sibling module in parent dir) | 1 | `docs/audit/r2_4_pcg64_algorithmic_bug/reference/pcg32_attempt/pcg32_fingerprint_probe.sio` |
| LEAN_ONLY | UNRESOLVED | match exhaustiveness (lean enforces, Madaros has no check, docs silent) | 1 | `self-hosted/gpu/lower_to_ptx_wmma_lean.sio` |
| LEAN_ONLY | UNRESOLVED | lean borrow checker (exclusive borrow already active); Madaros has no equivalent check | 1 | `examples/ontology/biomedical/snomed_elplus_adapter_demo.sio` |
| LEAN_ONLY | UNRESOLVED | lean E200 on `S` in `var p: S = S {..}`; a minimal copy passes | 1 | `docs/audit/g1_wip/MODULAR_INTRA_SRET_minimal_repro_2026-06-02.sio` |
| LEAN_ONLY | UNRESOLVED | lean E001 reported at a blank line (not isolated) | 1 | `stdlib/compiler/parser/mod.sio` |
| LEAN_ONLY | UNRESOLVED | lean "missing field in struct literal" (not isolated) | 1 | `stdlib/qnn/conv.sio` |
| LEAN_ONLY | UNRESOLVED | lean "array index must be integer" (not isolated) | 1 | `stdlib/linalg/shaped.sio` |
| LEAN_ONLY | UNRESOLVED | lean "Mut borrow requires mutable binding" on `match &! self.left` | 1 | `examples/debug_profile_demo.sio` |
| LEAN_ONLY | UNRESOLVED | gate fixture: lean_single 256-byte string literal limit | 1 | `tests/compiler/imported_string_literal_overflow/lib.sio` |
| LEAN_ONLY | UNRESOLVED | gate fixture: lean_single 256-byte string literal limit (reached through import) | 1 | `tests/compiler/imported_string_literal_overflow/main.sio` |
| LEAN_ONLY | UNRESOLVED | field privacy (lean enforces; Madaros has no check; docs silent) [documented known-failure] | 1 | `tests/known_failures/zero_event_private_field_access_probe.sio` |
| LEAN_ONLY | UNRESOLVED | audit control; expected outcome for the lean leg not stated in the file | 1 | `tests/audit/knowledge_validity_boundary_control.sio` |
| LEAN_ONLY | UNRESOLVED | E035 sites not recognised | 1 | `stdlib/particle_physics/jet.sio` |
| LEAN_ONLY | INTENTIONAL | test declares requires: madaros (lean: wide integers (i128/i256/i512) unsupported) [documented known-failure] | 1 | `tests/run-pass/wide_i128_fn_abi_known_failure.sio` |
| LEAN_ONLY | INTENTIONAL | test declares requires: madaros (lean: int-literal list into [i8; N]) | 1 | `tests/run-pass/array_repeat_i8_binding.sio` |
| LEAN_ONLY | INTENTIONAL | test declares requires: madaros (lean: int-literal array as fn tail) [documented known-failure] | 1 | `tests/run-pass/global_string_lit_init.sio` |
| LEAN_ONLY | INTENTIONAL | test declares requires: madaros (lean: error: use of moved value at <main>:N) | 1 | `tests/run-pass/linear_nested_branches.sio` |
| LEAN_ONLY | INTENTIONAL | test declares requires: madaros (lean: error: unknown identifier `X` at <main>:N) | 1 | `tests/run-pass/ffi_pow_tgamma_vectors.sio` |
| LEAN_ONLY | INTENTIONAL | test declares requires: madaros (lean: error: linear value not consumed at <main>:N) | 1 | `tests/run-pass/linear_capture_closure_consumed.sio` |
| LEAN_ONLY | INTENTIONAL | test declares requires: madaros (lean: error: field initializer type does not match struct field at) | 1 | `tests/run-pass/generic_struct_compound_field_inference.sio` |
| LEAN_ONLY | INTENTIONAL | test declares requires: madaros (lean: contest expression not supported) | 1 | `tests/run-pass/keyword_on_is_identifier_capable.sio` |
| LEAN_ONLY | INTENTIONAL | test declares requires: madaros (lean: E200) | 1 | `tests/run-pass/gpu_public_launch_check.sio` |
| LEAN_ONLY | INTENTIONAL | test declares requires: madaros (lean: E001) | 1 | `tests/run-pass/field_identity_survives_declaration_order.sio` |
| LEAN_ONLY | INTENTIONAL | test declares requires: madaros (lean: <no diagnostic line; last: typecheck: failed>) | 1 | `tests/run-pass/drift_gate_hessian_of.sio` |
| LEAN_ONLY | DEFECT-MADAROS | compile-fail test accepted by Madaros (lean: type error line N: Knightian uncertainty (ε=⊥) cannot sat) [declared error-pattern matched] | 1 | `tests/compile-fail/covid_2020_knightian_refusal.sio` |
| LEAN_ONLY | DEFECT-MADAROS | compile-fail test accepted by Madaros (lean: lex error line N: invalid escape sequence) [declared error-pattern matched] | 1 | `tests/ui/lexer/invalid_escape.sio` |
| LEAN_ONLY | DEFECT-MADAROS | compile-fail test accepted by Madaros (lean: error: use of moved value at <main>:N) [declared error-pattern matched] | 1 | `tests/compile-fail/affine_double_use.sio` |
| LEAN_ONLY | DEFECT-MADAROS | compile-fail test accepted by Madaros (lean: error: unknown unit in f64<UnitExpr> annotation at <main>:N) [declared error-pattern matched] | 1 | `tests/compile-fail/unit_f64_unit_expr_unknown_reject.sio` |
| LEAN_ONLY | DEFECT-MADAROS | compile-fail test accepted by Madaros (lean: error: linear value consumed in loop body at <main>:N) [declared error-pattern matched] | 1 | `tests/compile-fail/linear_loop_consume.sio` |
| LEAN_ONLY | DEFECT-MADAROS | compile-fail test accepted by Madaros (lean: error: duplicate parameter name at <main>:N) [declared error-pattern matched] | 1 | `tests/ui/resolve/duplicate_param.sio` |
| LEAN_ONLY | DEFECT-MADAROS | compile-fail test accepted by Madaros (lean: error: affine value used more than once at <main>:N) [declared error-pattern matched] | 1 | `tests/ui/ownership/affine_copy.sio` |
| LEAN_ONLY | DEFECT-MADAROS | compile-fail test accepted by Madaros (lean: error: Mut borrow requires mutable binding at <main>:N) [declared error-pattern matched] | 1 | `tests/ui/ownership/mutable_borrow_immut.sio` |
| LEAN_ONLY | DEFECT-MADAROS | compile-fail test accepted by Madaros (lean: E213) [declared error-pattern matched] | 1 | `tests/compile-fail/tuple_destructure_arity_mismatch.sio` |
| LEAN_ONLY | DEFECT-MADAROS | compile-fail test accepted by Madaros (lean: E211) [declared error-pattern matched] | 1 | `tests/compile-fail/study_missing_hypothesis.sio` |
| LEAN_ONLY | DEFECT-MADAROS | compile-fail test accepted by Madaros (lean: E001) [declared error-pattern matched] | 1 | `tests/ui/type/void_assign.sio` |
| LEAN_ONLY | DEFECT-LEAN | run-pass test rejected by lean (error: unknown identifier `X` at <main>:N) | 1 | `tests/stdlib/epistemic/test_dp_value_iteration.sio` |
| LEAN_ONLY | DEFECT-LEAN | run-pass test rejected by lean (error: logical not requires bool operand at <main>:N) [documented known-failure] | 1 | `tests/run-pass/mercyful_exposure_therapy.sio` |
| LEAN_ONLY | DEFECT-LEAN | run-pass test rejected by lean (error: linear value not consumed at <main>:N) | 1 | `tests/run-pass/closure_linear.sio` |
| LEAN_ONLY | DEFECT-LEAN | run-pass test rejected by lean (E001) | 1 | `tests/run-pass/ffi_system_exec.sio` |
| LEAN_ONLY | DEFECT-LEAN | nested array type on a local | 1 | `examples/zd_antiwindup_pid.sio` |
| LEAN_ONLY | DEFECT-LEAN | narrow unsigned integers (u8/u16/u32) | 1 | `tests/known_failures/u32_u16_u8_width_wrap_known_failure.sio` |
| LEAN_ONLY | DEFECT-LEAN | narrow unsigned integers (u8/u16/u32) [documented known-failure] | 1 | `tests/known_failures/u32_u16_u8_cast_mask_known_failure.sio` |
| LEAN_ONLY | DEFECT-LEAN | let bound to a call lean cannot type (nested or forward call) | 1 | `examples/sleep_orbit_demo.sio` |
| LEAN_ONLY | DEFECT-LEAN | int-literal list into [i8; N] [documented known-failure] | 1 | `tests/run-pass/cpc2026_scientific_float_parser.sio` |
| LEAN_ONLY | DEFECT-LEAN | implicit borrow of an array argument into a & / &! parameter (intended per the file, #1510) | 1 | `tests/multimodule/madaros_imported_slice_ref_global_main.sio` |
| LEAN_ONLY | DEFECT-LEAN | generic struct CDExact<F> (struct-level generics) | 1 | `docs/handoff/spike_generic_struct_return.sio` |
| LEAN_ONLY | DEFECT-LEAN | effect handler syntax handle<E> { .. } not supported by lean | 1 | `tests/audit/handle_io.sio` |
| LEAN_ONLY | BOTH | handle<E> over an invented effect: lean lacks handler syntax; Madaros accepts the unknown effect (probe's own negative control) | 1 | `tests/audit/handle_unknown_effect.sio` |
| LEAN_ONLY | BOTH | compile-fail test accepted by Madaros; lean rejects via its own defect: lean_single crashed (segmentation fault) [declared error-pattern not in the rejecting engine's output] | 1 | `tests/compile-fail/door1_too_many_globals_1025.sio` |
