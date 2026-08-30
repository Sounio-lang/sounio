<!-- docs:meta
topic_id: repo.docs.audit.gate-mortality-candidates-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: codex-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.gate-mortality-candidates-2026-08-19
-->

# Workflow-unreferenced gate mortality candidates

Source snapshot: `origin/main@f9b314736421f6cff0ca02ffe02c6cb7def71a0a`.

This is a decision aid, not a deletion list. No script was deleted, wired, or executed in this lane. The four operational dispatch classes are rendered in English below. Age orders investigation but never decides mortality.

## Instrument and boundary

- Universe: top-level `scripts/ci/*.sh`, intentionally excluding nine test fixtures under `scripts/ci/fixtures/`.
- Total: **547**; directly named by `.github/`: **104**; not directly named: **443**.
- Positive control: `concept_status_gate.sh` is named at `.github/workflows/ci.yml:68`.
- Negative control: `exact_bitwise_rebracket_authority_gate.sh` has no `.github/` mention.
- The first recursive count was 556/104/452; it was rejected because it incorrectly admitted the nine fixture scripts.
- Last-touch dates are included only as ordering evidence.

## Classification counts

- **Clearly dead candidate:** 11
- **Superseded or covered by a running parent:** 45
- **Reconnectable:** 25
- **Undetermined:** 362
- **Total:** 443

## Clearly dead candidates

| gate | last touch | evidence |
|---|---|---|
| `scripts/ci/check_overnight_plan_big_health.sh` | 2026-05-07 | Every status and recovery command targets deleted root-level scripts: `overnight_plan_big_status.sh`, `stop_overnight_plan_big.sh`, and `start_overnight_plan_big.sh`. The gate cannot observe or repair the service it names. |
| `scripts/ci/check_traceability_generator.sh` | 2026-05-07 | Its default path depends on deleted `scripts/latest_diagnostic_run_dir.sh` and directs users to deleted `scripts/fast_gate.sh`; without an explicit run directory it cannot reach the live report generator. |
| `scripts/ci/ekan_native_blocker_probe.sh` | 2026-07-11 | The entire body runs `ekan_native_frontier_matrix.sh` and filters existing HIDDEN rows; it adds no independent assertion and is a point blocker probe. |
| `scripts/ci/ekan_native_input_width_probe.sh` | 2026-07-11 | The entire body runs `ekan_native_frontier_matrix.sh` and filters existing input-width rows; it adds no independent assertion and is a point probe. |
| `scripts/ci/ekan_native_readout_probe.sh` | 2026-07-11 | The entire body runs `ekan_native_frontier_matrix.sh` and filters existing readout rows; it adds no independent assertion and is a point probe. |
| `scripts/ci/ekan_native_width_probe.sh` | 2026-07-11 | The entire body runs `ekan_native_frontier_matrix.sh` and filters the existing TWO_HIDDEN row; it adds no independent assertion and is a point probe. |
| `scripts/ci/madaros_open_blockers_probe.sh` | 2026-06-22 | Its own usage text says it reproduces known-open blockers without promoting them into required-pass manifests; it is explicitly a diagnostic probe, not a contract. |
| `scripts/ci/m6_closure_v2_validation.sh` | 2026-04-01 | Its compile matrix targets deleted `m4_flat_transitive_probe.sio`, `ontology_min_input.sio`, and `ontology_witness_program_probe.sio`; the advertised closure validation cannot start. |
| `scripts/ci/ontology_hash_benchmark.sh` | 2026-05-25 | It patches `lean_single.sio` to call deleted `ontology_run_cli()`; the independent ontology audit reproduced the resulting compile failure. The script is a micro-benchmark, not a validator. |
| `scripts/ci/run_release_critical_pack.sh` | 2026-06-14 | The release pack still invokes a long deleted series of root-level `sprint*_gate.sh` scripts; it cannot complete as a release contract on the present tree. |
| `scripts/ci/verify_skip_build_compat.sh` | 2026-05-10 | Five of its seven compatibility targets use deleted pre-reorganisation paths, including the self-host independence, zero-fallback, bootstrap-seed, feature-matrix, and golden-snapshot scripts. |

## Superseded or covered by a running parent

These scripts are not direct workflow steps, but a command-level scan finds that a workflow-reachable parent invokes them. The parent is the successor/coverage evidence; deleting the child without changing the parent would break the live chain.

| gate | last touch | running parent or successor |
|---|---|---|
| `scripts/ci/check_check_sio_integration_window.sh` | 2026-05-02 | Invoked by `scripts/ci/claude_operational_contract_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/check_claude_plan_consistency.sh` | 2026-05-01 | Invoked by `scripts/ci/claude_operational_contract_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/check_parallel_blocker_contract.sh` | 2026-05-03 | Invoked by `scripts/ci/claude_operational_contract_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/check_prompt_execution_contract.sh` | 2026-05-01 | Invoked by `scripts/ci/check_claude_plan_consistency.sh`, `scripts/ci/claude_operational_contract_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/claim_ast_gate.sh` | 2026-08-18 | Invoked by `scripts/ci/sigpipe_hygiene_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/dissertation_pbpk_suite_gate.sh` | 2026-06-16 | Invoked by `scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/kaxi_ptx_capture.sh` | 2026-05-10 | Invoked by `scripts/ci/kaxi_ptx_golden_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/kaxi_ptx_golden_gate.sh` | 2026-05-10 | Invoked by `scripts/ci/kretikos_kaxi_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/known_failure_madaros_recheck.sh` | 2026-08-18 | Invoked by `scripts/ci/madaros_changed_tests_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/kretikos_kaxi_gate.sh` | 2026-05-09 | Invoked by `scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/kretikos_kaxi_iso_budget_gate.sh` | 2026-05-10 | Invoked by `scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/kretikos_kaxi_phase_j_gate.sh` | 2026-05-10 | Invoked by `scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/kretikos_kaxi_phase_y_gate.sh` | 2026-08-18 | Invoked by `scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh`, `scripts/ci/sigpipe_hygiene_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/lean_single_fixed_point_gate.sh` | 2026-08-08 | Invoked by `scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/madaros_blocker_contract_gate.sh` | 2026-06-22 | Invoked by `scripts/ci/claude_operational_contract_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/madaros_global_capacity_gate.sh` | 2026-08-05 | Invoked by `scripts/ci/madaros_current_source_f64_lowering_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/madaros_global_f64_scratch_gate.sh` | 2026-07-17 | Invoked by `scripts/ci/madaros_current_source_f64_lowering_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/madaros_imported_call_arity_13_gate.sh` | 2026-07-26 | Invoked by `scripts/ci/madaros_current_source_f64_lowering_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/madaros_imported_capacity_gate.sh` | 2026-08-09 | Invoked by `scripts/ci/madaros_current_source_f64_lowering_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/madaros_imported_deref_f64_array_gate.sh` | 2026-07-13 | Invoked by `scripts/ci/madaros_current_source_f64_lowering_gate.sh`, `scripts/ci/madaros_full_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/mli_s3_bit_identity_gate.sh` | 2026-08-18 | Invoked by `scripts/ci/sigpipe_hygiene_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh` | 2026-06-01 | Invoked by `scripts/ci/native_v2_frontend_convergence_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/native_v2_driver_self_compile_gate.sh` | 2026-05-10 | Invoked by `scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh`, `scripts/ci/native_v2_epistemic_science_spine_gate.sh`, `scripts/ci/native_v2_gum_primitives_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/native_v2_e2e_codegen_suite_gate.sh` | 2026-06-01 | Invoked by `scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/native_v2_epistemic_science_spine_gate.sh` | 2026-04-28 | Invoked by `scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh`, `scripts/ci/native_v2_f64_ladder_gate.sh`, `scripts/ci/native_v2_hof_closure_gate.sh`, `scripts/ci/native_v2_semantic_hardening_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/native_v2_f64_ladder_gate.sh` | 2026-04-28 | Invoked by `scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/native_v2_frontend_convergence_gate.sh` | 2026-08-18 | Invoked by `scripts/ci/native_v2_imported_core_abi_gate.sh`, `scripts/ci/native_v2_imported_hof_abi_gate.sh`, `scripts/ci/sigpipe_hygiene_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/native_v2_gum_primitives_gate.sh` | 2026-08-18 | Invoked by `scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/native_v2_hof_closure_gate.sh` | 2026-05-10 | Invoked by `scripts/ci/native_v2_imported_closure_boundary_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/native_v2_imported_body_lowering_gate.sh` | 2026-08-18 | Invoked by `scripts/ci/native_v2_imported_closure_boundary_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/native_v2_imported_captured_closure_boundary_gate.sh` | 2026-06-28 | Invoked by `scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/native_v2_imported_closure_boundary_gate.sh` | 2026-06-28 | Invoked by `scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh`, `scripts/ci/native_v2_imported_captured_closure_boundary_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/native_v2_imported_core_abi_gate.sh` | 2026-06-28 | Invoked by `scripts/ci/native_v2_struct_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/native_v2_imported_hof_abi_gate.sh` | 2026-06-28 | Invoked by `scripts/ci/native_v2_struct_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/native_v2_metal_algebra_gate.sh` | 2026-08-18 | Invoked by `scripts/ci/native_v2_struct_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/native_v2_nvidia_bare_metal_gate.sh` | 2026-04-29 | Invoked by `scripts/ci/native_v2_struct_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/native_v2_semantic_hardening_gate.sh` | 2026-04-30 | Invoked by `scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/native_v2_serious_track_gate.sh` | 2026-06-14 | Invoked by `scripts/ci/native_v2_driver_self_compile_gate.sh`, `scripts/ci/native_v2_struct_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/native_v2_struct_gate.sh` | 2026-05-25 | Invoked by `scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/package_import_science_gate.sh` | 2026-06-16 | Invoked by `scripts/ci/package_pbpk_gum_gate.sh`, `scripts/ci/sounio_package_support_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/sedenion_phi_injectivity_gate.sh` | 2026-08-18 | Invoked by `scripts/ci/sigpipe_hygiene_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/seed_receipt_provenance_gate.sh` | 2026-08-18 | Invoked by `scripts/ci/canonical_compiler_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/self_falsifying_compiler_gate.sh` | 2026-08-18 | Invoked by `scripts/ci/sigpipe_hygiene_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/semantic_orc_depression_orc_gate.sh` | 2026-08-18 | Invoked by `scripts/ci/sigpipe_hygiene_gate.sh` in the workflow-reachable command closure. |
| `scripts/ci/sounio_validation.sh` | 2026-08-18 | Invoked by `scripts/ci/sigpipe_hygiene_gate.sh` in the workflow-reachable command closure. |

## Reconnectable

These rows have a current measured pass or a specific small repair recorded by an independent audit. They are candidates to reconnect or repair, not candidates to delete.

| gate | last touch | evidence |
|---|---|---|
| `scripts/ci/168_biology_validation_gate.sh` | 2026-08-03 | `CI_GATE_BUDGET_2026-08-18.tsv`: measured=yes, rc=0, 5 s. |
| `scripts/ci/build_ontology_validation_souc.sh` | 2026-06-29 | Independent ontology audit PR #1970 on this SHA: rc=0; isolated validation-wrapper build. |
| `scripts/ci/compile_fail_harness_contract_gate.sh` | 2026-08-18 | `CI_GATE_BUDGET_2026-08-18.tsv`: measured=yes, rc=0, 1 s; live harness contract. |
| `scripts/ci/generated_ontology_gate.sh` | 2026-05-24 | Independent ontology audit PR #1970 on this SHA: rc=0 under `--check`; regeneration left the tree clean. |
| `scripts/ci/generated_ontology_manifest_gate.sh` | 2026-05-25 | Independent ontology audit PR #1970 on this SHA: rc=0; validates the nine-bundle manifest. |
| `scripts/ci/knowledge_context_phase2_ontology_gate.sh` | 2026-05-25 | Independent ontology audit PR #1970: the instrument stops at EACCES because it omits `chmod +x`; the witness remains relevant and repair is local. |
| `scripts/ci/kretikos_kaxi_phase_j_aggregate_gate.sh` | 2026-05-10 | `CI_GATE_BUDGET_2026-08-18.tsv`: measured=yes, rc=0, 3 s, 6 PASS / 0 FAIL. |
| `scripts/ci/madaros_ols_fixed_e2e_gate.sh` | 2026-08-06 | `CI_GATE_BUDGET_2026-08-18.tsv`: measured=yes, rc=0, 2 s. |
| `scripts/ci/madaros_print_f64_negative_gate.sh` | 2026-07-22 | `CI_GATE_BUDGET_2026-08-18.tsv`: measured=yes, rc=0, 2 s. |
| `scripts/ci/madaros_zero_provenance_failclosed_gate.sh` | 2026-08-05 | `CI_GATE_BUDGET_2026-08-18.tsv`: measured=yes, rc=0, 2 s. |
| `scripts/ci/mir_instr_capacity_coherence_gate.sh` | 2026-07-20 | `CI_GATE_BUDGET_2026-08-18.tsv`: measured=yes, rc=0, below 1 s; source-coherence contract. |
| `scripts/ci/octonion_probes_gate.sh` | 2026-08-18 | `CI_GATE_BUDGET_2026-08-18.tsv`: measured=yes, rc=0, 1 s. |
| `scripts/ci/ontology_bundle_directive_gate.sh` | 2026-05-25 | Independent ontology audit PR #1970: the required sibling refusal is live as E009; only the expected diagnostic string is stale. |
| `scripts/ci/ontology_bundle_directive_native_scan_gate.sh` | 2026-05-25 | Independent ontology audit PR #1970: call-site arity drift (2 versus 3 arguments) is a small, identified instrument repair. |
| `scripts/ci/ontology_cache_compile_gate.sh` | 2026-06-12 | Independent ontology audit PR #1970 on this SHA: rc=0; compile-and-run witness. |
| `scripts/ci/ontology_cache_frontend_composition_gate.sh` | 2026-05-25 | Independent ontology audit PR #1970: check passes; execution reaches the known multimodule native thin-link rc=12 rather than an obsolete target. |
| `scripts/ci/ontology_model_compile_gate.sh` | 2026-06-12 | Independent ontology audit PR #1970 on this SHA: rc=0; compile-and-run witness. |
| `scripts/ci/ontology_query_compile_gate.sh` | 2026-06-12 | Independent ontology audit PR #1970 on this SHA: rc=0; compile-and-run witness. |
| `scripts/ci/ontology_reasoner_compile_gate.sh` | 2026-06-12 | Independent ontology audit PR #1970 on this SHA: rc=0; compile-and-run witness. |
| `scripts/ci/ontology_typed_bridge_gate.sh` | 2026-06-12 | Independent ontology audit PR #1970 on this SHA: rc=0; positive bridge plus named E152 refusal. |
| `scripts/ci/ontology_unit_metadata_gate.sh` | 2026-05-25 | Independent ontology audit PR #1970: small repair identified; read the live Madaros unit registry in `self-hosted/check/units.sio`, not stale lean_single text. |
| `scripts/ci/run_pass_output_gate.sh` | 2026-08-18 | `CI_GATE_BUDGET_2026-08-18.tsv`: measured=yes, rc=0, 53 s; 219 outputs checked against the committed baseline. |
| `scripts/ci/self_falsifying_compilation_line_gate.sh` | 2026-08-03 | `CI_GATE_BUDGET_2026-08-18.tsv`: measured=yes, rc=0, 1 s; SUBSTRATE_LIVE receipt. |
| `scripts/ci/sounio_direct_driver_support_gate.sh` | 2026-06-27 | `CI_GATE_BUDGET_2026-08-18.tsv`: measured=yes, rc=0, 16 s, 24/24. |
| `scripts/ci/stdlib_source_byte_ceiling_gate.sh` | 2026-08-17 | `CI_GATE_BUDGET_2026-08-18.tsv`: measured=yes, rc=0, 4 s. |

## Undetermined

Read-only inspection did not establish a deleted target, exact duplicate, live successor, current pass, or bounded repair. Per dispatch, these are deliberately not guessed. Each needs execution or a narrower semantic comparison before a mortality decision.

| gate | last touch | evidence boundary |
|---|---|---|
| `scripts/ci/ade_wildgen_mckay_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/assert_no_rust_markers.sh` | 2026-04-05 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/associator_gum_variance_gate.sh` | 2026-07-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/bootstrap_chain_gate.sh` | 2026-05-10 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/brain_ossm_sigmoid_polarity_gate.sh` | 2026-07-11 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/cd_tower_nullity_histogram_law_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/cd_zd_graph_invariants_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/check_doc_snippets.sh` | 2026-05-10 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/check_feature_matrix.sh` | 2026-05-07 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/check_golden_snapshots.sh` | 2026-05-07 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/check_new_warnings.sh` | 2026-05-07 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/check_no_active_cargo_jobs.sh` | 2026-05-07 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/chingon_zd_gate.sh` | 2026-07-26 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/claim_native_gate.sh` | 2026-07-26 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/claim_root_preflight_gate.sh` | 2026-08-18 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/compiler_stage_contract_gate.sh` | 2026-06-16 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/cpc2026_yale_evidence_gate.sh` | 2026-07-11 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/deep_four_lane_gate.sh` | 2026-07-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/dissertation_dossier_gate.sh` | 2026-05-10 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/dissertation_pbpk28_parity_gate.sh` | 2026-08-17 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/dissertation_pbpk_hessian_gate.sh` | 2026-08-18 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/dyadic_non_reduction_gate.sh` | 2026-07-26 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/dyadic_relational_associator_gate.sh` | 2026-07-26 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/e_series_semantic_germ_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/effect_archaeology_gate.sh` | 2026-08-18 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/effects_handler_interaction_lean_gate.sh` | 2026-08-06 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/eisa_bridge_conformance_gate.sh` | 2026-07-10 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/eisa_bridge_conformance_gate_madaros.sh` | 2026-08-17 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/eisa_h_zd_reference_gate.sh` | 2026-07-12 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/ekan_native_core_gate.sh` | 2026-07-11 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/ekan_native_frontier_matrix.sh` | 2026-07-11 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/engine_parity_gate.sh` | 2026-08-18 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/epistemic_monotonicity_gate.sh` | 2026-05-09 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/epistemic_multidrug_gate.sh` | 2026-05-10 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/epistemic_pbpk_gate.sh` | 2026-05-10 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/epistemic_witness_gate.sh` | 2026-05-10 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/exact_bitwise_rebracket_authority_gate.sh` | 2026-07-17 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/exact_bitwise_rebracket_source_ir_gate.sh` | 2026-07-17 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/falsification_ledger_gate.sh` | 2026-07-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/federated_san_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/ffi_posix_builtin_gate.sh` | 2026-08-16 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/float_slot_capacity_coherence_gate.sh` | 2026-07-20 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_bytecode_fragment_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_css_surface_parity_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_emit_pure_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_engine_install_fragment_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_method_xfer_fragment_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_multimod_fragment_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_pk_import_auc_thalf_driver_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_pk_import_auct_driver_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_pk_import_cmax_driver_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_pk_import_fss_driver_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_pk_import_ld_driver_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_pk_import_method_driver_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_pk_import_mrt_driver_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_pk_import_ptr_driver_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_pk_import_rac_driver_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_pk_struct_auc_thalf_driver_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_pk_struct_auct_driver_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_pk_struct_cmax_driver_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_pk_struct_fss_driver_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_pk_struct_ld_driver_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_pk_struct_method_driver_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_pk_struct_mrt_driver_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_pk_struct_multidose_driver_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_pk_struct_ptr_driver_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_pk_struct_rac_driver_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_pk_struct_rho_tau_driver_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_registration_fragment_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_residual4_stack_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fo_surface_transfer_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/founder_intent_contract_gate.sh` | 2026-07-12 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/fpga_catastrophe_scan_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/functor_f_g2_covariance_gate.sh` | 2026-07-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/g1_expr_recursion_gate.sh` | 2026-06-01 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/g2_zd_fibers_gate.sh` | 2026-07-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/garden_to_claim_gate.sh` | 2026-07-26 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/gate_assert_instrument_selftest.sh` | 2026-08-18 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/gate_coverage_audit.sh` | 2026-05-10 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/global_aggregate_store_gate.sh` | 2026-08-06 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/gpu_knowledge_vecmat_evidence_gate.sh` | 2026-07-07 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/graphics_companion_gate.sh` | 2026-05-23 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/graphics_scaffold_gate.sh` | 2026-05-23 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/gri30_full_workspace_allocation_gate.sh` | 2026-08-12 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/handle_table_ceiling_gate.sh` | 2026-08-18 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/ir_instr_arena_gate.sh` | 2026-08-11 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/ir_module_arena_v2_soir_v5_bridge_gate.sh` | 2026-07-18 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/ir_nop_marker_gate.sh` | 2026-08-11 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/irfunction_instr_capacity_coherence_gate.sh` | 2026-08-11 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/journal_submission_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/k2_check_import_bridge_classifier.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/knowledge_context_composite_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/knowledge_context_numeric_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/knowledge_context_runtime_obligation_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/knowledge_context_static_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/knowledge_context_static_value_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/knowledge_context_unit_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/knowledge_runtime_guard_directive_native_scan_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/knowledge_runtime_guard_expansion_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/knowledge_runtime_guard_lowering_plan_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/knowledge_runtime_guard_native_lowering_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/kretikos_associator_emit_gate.sh` | 2026-05-08 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/kretikos_cross_backend_cuda_benchmark_gate.sh` | 2026-05-19 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/kretikos_cross_backend_cuda_runtime_gate.sh` | 2026-05-19 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/kretikos_cross_backend_semantic_gate.sh` | 2026-05-19 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/kretikos_cubin_evidence_gate.sh` | 2026-05-18 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/kretikos_cubin_evidence_matrix_gate.sh` | 2026-05-18 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/kretikos_f64_runtime_gate.sh` | 2026-05-09 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/kretikos_hpc_slurm_runtime_gate.sh` | 2026-05-09 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/kretikos_hpc_source_lowering_gate.sh` | 2026-05-09 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/kretikos_kaxi_fmad_invariance_gate.sh` | 2026-05-12 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/kretikos_kaxi_l4_launch_gate.sh` | 2026-05-10 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/kretikos_kaxi_lowering_gate.sh` | 2026-05-09 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/kretikos_kaxi_lse8_gate.sh` | 2026-05-11 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/kretikos_kaxi_phase_w2_gate.sh` | 2026-05-10 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/kretikos_kaxi_phase_w_gate.sh` | 2026-05-10 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/kretikos_kaxi_phase_x_gate.sh` | 2026-05-10 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/kretikos_kaxi_phase_z_assoc_gate.sh` | 2026-05-22 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/kretikos_kaxi_round_mode_gate.sh` | 2026-05-11 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/kretikos_kaxi_sinkhorn16_epistemic_slurm_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/kretikos_kaxi_sinkhorn16_gate.sh` | 2026-05-20 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/kretikos_kaxi_sinkhorn16_slurm_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/kretikos_ptx_kaxi_verify.sh` | 2026-05-07 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/kretikos_spirv_vulkan_storage_semantic_baseline_gate.sh` | 2026-05-20 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/kretikos_spirv_vulkan_storage_vec_add_gate.sh` | 2026-05-20 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/l8_zd_census_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/l9_nullity_histogram_law_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/l9_zd_census_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/lean_falsification_ledger_gate.sh` | 2026-07-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/lean_single_pub_use_reexport_gate.sh` | 2026-07-13 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/local_cuda_device_admission_gate.sh` | 2026-04-29 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_862_import_print_gate.sh` | 2026-08-06 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_contest_callsite_box_gate.sh` | 2026-07-17 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_contest_ir_authority_gate.sh` | 2026-07-17 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_correlated_eq_gate.sh` | 2026-08-17 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_d3_exclref_shipped_gate.sh` | 2026-08-06 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_d3_openslice_len_gate.sh` | 2026-08-06 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_d6_const_nonmain_gate.sh` | 2026-08-06 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_ep_gate_imported_gate.sh` | 2026-07-26 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_f128_f256_format_identity_gate.sh` | 2026-07-14 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_f128_f256_ladder_gate.sh` | 2026-08-17 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_f128_f256_numeric_payload_gate.sh` | 2026-07-14 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_f128_f256_numeric_wire_gate.sh` | 2026-07-15 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_f128_f256_v0c_wire_gate.sh` | 2026-08-17 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_f128_f256_v0d_softfloat_gate.sh` | 2026-08-17 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_field_if_i64_gate.sh` | 2026-07-26 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_fixed_array_call_boundary_alias_gate.sh` | 2026-07-14 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_global_array_ref_gate.sh` | 2026-07-21 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_global_string_init_gate.sh` | 2026-07-22 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_gum_cross_function_gate.sh` | 2026-07-28 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_gum_fo_trust_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_gum_semantic_suite_gate.sh` | 2026-07-28 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_high_arity_ref_gate.sh` | 2026-08-13 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_imported_array_byvalue_gate.sh` | 2026-07-21 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_imported_ep_var_preserve_gate.sh` | 2026-08-06 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_imported_f64_const_gate.sh` | 2026-07-21 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_imported_f64_mul_gate.sh` | 2026-07-26 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_intrinsic_knowledge_type_gate.sh` | 2026-08-18 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_launcher_exit_status_gate.sh` | 2026-08-17 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_local_known_extent_word_array_copy_gate.sh` | 2026-07-14 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_metal_gate.sh` | 2026-06-15 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_method_receiver_gate.sh` | 2026-07-28 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_monolithic_public_lower_call_gate.sh` | 2026-07-13 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_monolithic_public_lower_call_matrix.sh` | 2026-07-13 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_operational_contract_gate.sh` | 2026-06-20 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_print_dispatch_refusal_gate.sh` | 2026-08-18 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_propagate_monte_carlo_fnptr_failclosed_gate.sh` | 2026-08-06 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_ptx_gate.sh` | 2026-06-15 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_qd128_core_native_v2_gate.sh` | 2026-08-05 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_qd128_mul_native_v2_gate.sh` | 2026-08-05 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_sedenion_native_v2_gate.sh` | 2026-08-05 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_syscall6_gate.sh` | 2026-07-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_thinlink_bool_cmp_field_gate.sh` | 2026-08-05 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_trait_i64_cd_exact_gate.sh` | 2026-08-06 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_validation_import_gate.sh` | 2026-08-06 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_visibility_context_gate.sh` | 2026-07-13 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_wide_int_gate.sh` | 2026-06-14 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_xoshiro_imported_gate.sh` | 2026-08-06 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/madaros_zero_provenance_native_v2_gate.sh` | 2026-08-05 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/mercyful_chemo_sequencing_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/mercyful_clinical_sequencing_gate.sh` | 2026-07-26 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/mercyful_continuous_control_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/mercyful_expanded_ethics_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/mercyful_independent_tdm_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/mercyful_lean_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/mercyful_learned_field_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/mercyful_machine_channel_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/mercyful_mimic_iv_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/mercyful_mimic_iv_sensitivity_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/mercyful_mimic_iv_subgroup_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/mercyful_pontryagin_control_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/mercyful_preprint_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/mercyful_runtime_gate.sh` | 2026-07-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/mercyful_sounio_gate.sh` | 2026-07-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/moonshot_a_abide_claim_discipline_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/moonshot_a_abide_cohort_manifest_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/moonshot_a_abide_epistemic_cohort_slurm_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/moonshot_a_abide_epistemic_orc_slice_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/moonshot_a_abide_f32_cohort_analysis_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/moonshot_a_abide_transport_conditioned_orc_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/moonshot_a_phase_status_adopt_runtime_artifacts.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/moonshot_a_phase_status_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/moonshot_a_sinkhorn16_slurm_gate_common.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/moonshot_a_slurm_blocker_handoff_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/moonshot_a_transport_168_curvature_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/moonshot_a_transport_168_linearity_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/moonshot_a_transport_168_manifest_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/moonshot_a_transport_168_modulation_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/native_capacity_tiers_gate.sh` | 2026-08-05 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/native_v2_albert_gate.sh` | 2026-05-10 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/native_v2_algebra_law_gate.sh` | 2026-05-10 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/native_v2_array_gate.sh` | 2026-04-27 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/native_v2_associator_gate.sh` | 2026-04-28 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/native_v2_capturing_closure_unsupported_gate.sh` | 2026-06-27 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/native_v2_clifford_grade_gate.sh` | 2026-04-28 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/native_v2_dissertation_rapamycin_gate.sh` | 2026-05-10 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/native_v2_e2e_exit_code_gate.sh` | 2026-05-30 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/native_v2_enum_match_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/native_v2_epistemic_accel_spine_gate.sh` | 2026-07-17 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/native_v2_epistemic_gpu_runtime_parity_gate.sh` | 2026-05-09 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/native_v2_fractal_gate.sh` | 2026-04-28 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/native_v2_logical_gate.sh` | 2026-04-27 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/native_v2_muraki_pentagon_gate.sh` | 2026-04-28 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/native_v2_nested_field_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/native_v2_out_param_boundary_gate.sh` | 2026-05-02 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/native_v2_prebundle_gate.sh` | 2026-04-28 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/native_v2_q_deform_gate.sh` | 2026-04-28 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/native_v2_q_deform_octonion_gate.sh` | 2026-04-28 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/native_v2_sed_zd_gate.sh` | 2026-04-28 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/native_v2_struct_mutation_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/native_v2_struct_param_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/native_v2_struct_return_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/native_v2_wasserstein_orc_gate.sh` | 2026-05-10 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/no_false_float_axioms.sh` | 2026-06-30 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/nullity_histogram_law_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/nunwa_lean_gate.sh` | 2026-08-06 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/oopsla2027_paper_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/ordered_path_provenance_source_ir_gate.sh` | 2026-07-17 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/package_boundary_release_gate.sh` | 2026-07-17 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/paper168_cocycle_subspace_gate.sh` | 2026-05-11 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/paper_168_theorem_claims_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/parser_keyword_classification_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/parser_module_path_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/particle_broken_structure_dual_gate.sh` | 2026-07-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/particle_exp10_approx_algebra_gate.sh` | 2026-07-26 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/particle_exp11_scheme_approx_gate.sh` | 2026-07-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/particle_exp123_gate.sh` | 2026-07-26 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/particle_exp123_oracle_gate.sh` | 2026-07-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/particle_exp12_residual_closure_gate.sh` | 2026-07-26 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/particle_exp13_amplitude_honesty_gate.sh` | 2026-07-26 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/particle_exp17_zwh_ledger_gate.sh` | 2026-08-06 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/particle_exp6_universal_xi_gate.sh` | 2026-07-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/particle_exp7_gum_transfer_gate.sh` | 2026-07-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/particle_exp8_collapse_gate.sh` | 2026-07-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/particle_exp9_engine_joint_gate.sh` | 2026-07-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/particle_unstable_effect_gate.sh` | 2026-07-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/pediatric_pbpk_gate.sh` | 2026-07-27 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/physical_extraction_canonical_cutover_approval_gate.sh` | 2026-07-17 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/physical_extraction_canonical_cutover_execution_gate.sh` | 2026-07-17 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/physical_extraction_canonical_production_gap_gate.sh` | 2026-07-17 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/physical_extraction_canonical_production_mapping_decision_gate.sh` | 2026-07-18 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/physical_extraction_inventory_gate.sh` | 2026-07-17 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/physical_extraction_materialization_gate.sh` | 2026-07-17 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/physical_extraction_source_removal_authorization_gate.sh` | 2026-07-17 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/physical_extraction_source_removal_execution_gate.sh` | 2026-07-17 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/plan_big_gate.sh` | 2026-05-07 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/plan_big_strict_canary.sh` | 2026-05-07 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/poseidon_gate.sh` | 2026-05-07 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/project_spine_gate.sh` | 2026-07-01 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/proof_carrying_deployment_validity_revocable_authority_gate.sh` | 2026-07-26 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/proof_carrying_endogenous_observability_gate.sh` | 2026-07-26 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/proof_carrying_model_contest_gate.sh` | 2026-07-26 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/proof_carrying_path_conditioned_identification_gate.sh` | 2026-07-26 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/proof_carrying_policy_observation_associator_gate.sh` | 2026-07-26 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/proof_carrying_policy_state_feedback_gate.sh` | 2026-07-26 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/proof_carrying_rebracketing_protocol_gate.sh` | 2026-07-26 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/proof_carrying_reflexive_inquiry_gate.sh` | 2026-07-26 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/proof_carrying_shift_robust_risk_transport_gate.sh` | 2026-07-26 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/proof_carrying_statistical_coverage_empirical_binding_gate.sh` | 2026-07-26 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/r2_continuous_law_theorem_gate.sh` | 2026-07-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/real_language_runner_gate.sh` | 2026-06-30 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/refinement_gate.sh` | 2026-05-07 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/registry_attestation_spec_gate.sh` | 2026-07-17 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/release_check.sh` | 2026-04-05 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/reproduce_artifact.sh` | 2026-05-10 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/rna_cd_confirmatory_contract.sh` | 2026-08-15 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/routon_zd_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/sac_llm_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/san_imagenet_fpga_dl380_gate.sh` | 2026-08-04 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/san_real_patient_data_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/science_boundary_gate.sh` | 2026-07-16 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/self_falsifying_compilation_line_r11_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/self_falsifying_compilation_line_r17_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/self_falsifying_compilation_line_r20_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/self_falsifying_compilation_line_r26_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/self_falsifying_compilation_line_r2_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/self_falsifying_compilation_line_r4_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/self_falsifying_compilation_line_r5_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/selfhost_driver_output_gate.sh` | 2026-05-07 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/semantic_coordination_gate.sh` | 2026-07-12 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/semantic_knowledge_spine_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/semantic_orc_sinkhorn_lse_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/semantic_orc_swow16_fixture_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/semantic_orc_swow16_graph_degree_shuffle_fixture_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/semantic_orc_swow16_graph_edge_kaxi_pack_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/semantic_orc_swow16_graph_edge_multifixture_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/semantic_orc_swow16_graph_edge_tile_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/semantic_orc_swow16_graph_edge_tile_matrix_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/semantic_orc_swow16_graph_edge_tile_matrix_parameter_sweep_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/semantic_orc_swow16_graph_edge_tile_matrix_reducer_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/semantic_orc_swow16_kaxi_pack_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/semantic_orc_swow16_kaxi_runtime_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/semantic_orc_swow16_kaxi_slurm_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/semantic_orc_swow16_multisupport_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/semantic_orc_swow16_permutation_fixture_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/serious_language_claim_closure_gate.sh` | 2026-05-13 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/serious_language_conformance_gate.sh` | 2026-05-12 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/serious_language_spec_drift_gate.sh` | 2026-06-27 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/soir_v5_empty_reader_gate.sh` | 2026-07-18 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/soir_v6_bss_layout_gate.sh` | 2026-07-19 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/souc-native-wrapper.sh` | 2026-06-30 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/souc-seq-leansingle.sh` | 2026-06-23 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/souc_invoke_selftest.sh` | 2026-08-04 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/sounio_editor_tooling_support_gate.sh` | 2026-07-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/sounio_install_support_gate.sh` | 2026-06-26 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/sounio_release_production_readiness_gate.sh` | 2026-06-27 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/sounio_science_flex_gate.sh` | 2026-07-26 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/sounio_stdlib_surface_support_gate.sh` | 2026-06-27 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/stdlib_evolution_gate.sh` | 2026-05-10 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/stdlib_science_pipeline_gate.sh` | 2026-04-24 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/suffering_aware_architecture_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/suffering_aware_deep_architecture_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/suffering_aware_game_theory_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/suffering_aware_large_architecture_gate.sh` | 2026-08-04 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/suffering_aware_multi_agent_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/suffering_aware_multi_agent_scale_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/suffering_aware_multi_agent_sophisticated_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/track_a_nv2_parity_inventory.sh` | 2026-05-13 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/trigintaduonion_zd_gate.sh` | 2026-07-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/typekind_archaeology_c.sh` | 2026-08-18 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/typekind_archaeology_gate.sh` | 2026-08-18 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/ui_type_backlog_quality_gate.sh` | 2026-05-07 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/ui_type_deignore_audit.sh` | 2026-05-07 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/ui_type_deignore_batch_plan.sh` | 2026-05-07 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/unit_types_clinical_current_source_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/unit_types_derived_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/unit_types_phase1_gate.sh` | 2026-05-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/validate_skills_coverage.sh` | 2026-05-10 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/verify_compiler_pin.sh` | 2026-05-14 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/verify_lean_seed.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/windows_pe_smoke_gate.sh` | 2026-05-21 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/witness_based_compilation_paper_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/zd_deep_dive_gate.sh` | 2026-05-10 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/zd_fiber_spectra_witness_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/zd_fiber_spectra_witness_perturbed_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/zd_orbit_equivalence_gate.sh` | 2026-05-10 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/zd_qec_prediction_gate.sh` | 2026-08-03 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/zero_event_gate.sh` | 2026-07-11 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/zero_event_native_compile_privacy_gate.sh` | 2026-07-12 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/zero_event_native_v2_matrix.sh` | 2026-08-05 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/zero_provenance_claims_gate.sh` | 2026-07-25 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |
| `scripts/ci/zero_provenance_witness_gate.sh` | 2026-07-11 | No direct workflow mention; no content-backed death, successor, or current receipt established in this read-only pass. Execution or criterion comparison required. |

## Founder decision surface

Only the 11 **clearly dead candidates** have affirmative mortality evidence in this pass. The 45 parent-covered rows must not be deleted as if they were unreachable. The 25 reconnectable rows retain live or locally repairable contracts. The 362 undetermined rows are the remaining investigation queue, not an implied keep list.

For the reachability manifest owned by the parallel lane, this document should initially feed `obsoleto` only after the founder accepts an individual death row. The manifest owner requires a non-empty owner and reason; this lane intentionally does not write that TSV.
