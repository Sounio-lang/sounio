<!-- docs:meta
topic_id: repo.docs.audit.engine-divergence-corpus-2026-08-30
authority: repo_only
audience: users
last_validated: 2026-08-30
validated_by: claude-3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.engine-divergence-corpus-2026-08-30
-->

# 112 of 1854 run-pass tests are refused by the default engine

**Date:** 2026-08-30
**Method:** `souc check` on every file in `tests/run-pass/` (1854), under two
engines, comparing verdicts. Madaros was **built from `origin/main` source that
day** (`scripts/ci/build_modular_madaros.sh`); the other arm is the committed
bootstrap seed `bin/souc-lean-single-x86_64`.

## Result

| | count |
|---|---|
| both engines agree | 1742 |
| **Madaros refuses, seed accepts** | **112** |
| seed refuses, Madaros accepts | 724 — see "the invalid arm" |

The 1742 agreements are the internal control: this is not a broken environment.
94% of the corpus checks identically.

`bin/souc` routes to Madaros by default — its own header says so: *"DEFAULT
ENGINE: Madaros ... This is the compiler people invoke for normal use."* So these
112 are shipped tests that fail for anyone running `souc check` on them.

## None of them declares an engine dependency

Of the 112: **zero** carry `//@ requires`, one carries `//@ known-failure`, one
carries `//@ ignore`. The remaining 110 are ordinary `run-pass` tests that claim
to pass and do not, under the default compiler.

## What Madaros objects to

Top diagnostics across the 112:

    18  E137  use of undeclared variable
    13  E009  argument type does not match parameter
    10  E004
     9  E175
     9  E001  type mismatch
     7  E012  this type has no field named
     6  E008  return value does not match declared return type
     5  E035  effect not declared in function signature

Two worked examples:

    approx_propagation.sio
      madaros: error[E035] ... (missing: Approx, Mut, Div, Panic) -- required by `approx_iter`
      seed:    0 errors

    async_basic.sio
      madaros: error[E012] this type has no field named   (x2)
      seed:    0 errors

The direction matters. This is the **restrictive** class — Madaros refusing
programs the seed accepts, the same shape as the refinement-type divergence in
`ENGINE_DIVERGENCE_E221_REFINEMENTS_2026-08-30.md`, and the opposite of the E221
case where Madaros is the permissive one. Several look like Madaros enforcing a
rule the seed does not (E035 effect declarations) rather than Madaros being wrong.
Which side is correct is a per-file question this audit does not settle.

## The invalid arm, recorded so nobody re-derives it

The 724 files where the seed refuses and Madaros accepts are **not** reported as
divergences. Sampling one (`a_mu_cmd3_2pi_split.sio`), the seed's errors are
inside the standard library, not the test:

    error[E200]: undefined identifier `Epistemic` at stdlib/epistemic/knowledge.sio:196
    error[E006]: arity mismatch at line 196 expected 1 got 2

That arm measures the frozen seed's coverage of current stdlib, not anything about
the tests. It is listed here only so the next person does not spend the sweep
again.

## GAP CLOSED 2026-08-31 — CI's engine IS the seed, byte for byte

Built `souc-stage2` exactly as CI does: `scripts/ci/selfhost_host_gate.sh` with
`SOUNIO_SELFHOST_HOST_GATE_DIR` **and `SOUNIO_FORCE_SOURCE_BOOTSTRAP=1`**, the
variable the `native-selfhost-linux-x86_64` job sets. Gate passes,
`stage2_sha256=b4f9eed017644248…`.

    md5(souc-stage2 built from source)        e9b044ae1f8e0001a200a1757b858cbd
    md5(bin/souc-lean-single-x86_64, committed) e9b044ae1f8e0001a200a1757b858cbd

**Identical.** The engine CI calibrates against is the committed seed, so the
question "does stage2 side with the seed or with Madaros" is answered by identity:
it *is* the seed. Green CI on these 112 therefore says nothing whatever about the
compiler `bin/souc` invokes by default.

Confirmed against the real harness rather than by inference:
`run_sio_test_suite_v2.sh --filter-exact async_basic.sio` with
`SOUNIO_TEST_SOUC_BIN=<stage2>` reports **Pass: 1**, while `bin/souc check` on the
same file under Madaros reports `error[E012] this type has no field named`.

### Without the source-bootstrap flag the gate fails, and it is not a finding

Omitting `SOUNIO_FORCE_SOURCE_BOOTSTRAP=1` makes the gate exit 1 in 5 s with
`error: self-host fixed-point mismatch for x86_64-linux`, reproducibly, producing
a deterministic 2 557 444-byte binary that differs from the committed seed. That
is the flag missing, not the seed drifting. Recorded because the failure is
convincing enough to be mistaken for one.

### Three invocation paths, three behaviours

Do not compare across them; the binary is not the only variable.

    bin/souc check           (wrapper, ENGINE=lean_single)  -> OK
    SOUNIO_SOUC_BIN=<elf> check   (raw CLI, same bytes)     -> error[E221]: no main
    run_sio_test_suite_v2.sh (harness, SOUNIO_TEST_SOUC_BIN) -> Pass

The middle row is the same binary as the first and reports "no main" on a file
that *has* `fn main`. An earlier draft of this note read that as a third engine
disagreeing; it is one binary under three entry points. The harness is ground
truth for what CI does.

## What this audit still does NOT close

Which side is right, per file. Madaros refusing `approx_propagation.sio` with
"effect not declared (missing: Approx, Mut, Div, Panic)" may well be Madaros being
correct and the seed being lax. This audit establishes that the two disagree and
that CI cannot see it; it does not adjudicate 112 individual cases.

What is certain: 110 tests with no declared engine dependency do not pass under
the compiler `bin/souc` invokes by default, and no gate in this repository
notices.

## Affected files

    tests/run-pass/algebra_commutative_default.sio
    tests/run-pass/algebra_g2_null_model.sio
    tests/run-pass/algebra_observe_synthesis.sio
    tests/run-pass/algebra_properties_basic.sio
    tests/run-pass/approx_propagation.sio
    tests/run-pass/associator_variance_mc.sio
    tests/run-pass/async_basic.sio
    tests/run-pass/async_channels.sio
    tests/run-pass/async_join.sio
    tests/run-pass/async_sleep.sio
    tests/run-pass/async_spawn.sio
    tests/run-pass/async_spawn_syscall_pid.sio
    tests/run-pass/async_stress_channel.sio
    tests/run-pass/async_stress_cow.sio
    tests/run-pass/async_stress_forks.sio
    tests/run-pass/async_stress_nested.sio
    tests/run-pass/async_stress_slot_size.sio
    tests/run-pass/bdf_stiff.sio
    tests/run-pass/bitwise_not_bootstrap_regression.sio
    tests/run-pass/clinical_deployment_validity_revocable_authority_witness.sio
    tests/run-pass/closure_capture.sio
    tests/run-pass/closure_effect_infer.sio
    tests/run-pass/closure_effect_infer_auto.sio
    tests/run-pass/closure_generic_hof.sio
    tests/run-pass/closure_hof.sio
    tests/run-pass/connectome_laplacian_eigenvectors.sio
    tests/run-pass/covid_2020_kernel.sio
    tests/run-pass/cybernetic_proof.sio
    tests/run-pass/d4_optimizer_integration.sio
    tests/run-pass/darwin_venlafaxine_xr_pgx_smoke.sio
    tests/run-pass/epistemic_guarded_nested_measure_call.sio
    tests/run-pass/epistemic_hessian_transcendentals.sio
    tests/run-pass/epistemic_hessian_two_arg.sio
    tests/run-pass/epistemic_unspecified_epsilon.sio
    tests/run-pass/epsilon_comparison_valid.sio
    tests/run-pass/for_in_vec.sio
    tests/run-pass/g2_abide_sounio.sio
    tests/run-pass/g2_bridge_pipeline.sio
    tests/run-pass/g2_cohort_comparison.sio
    tests/run-pass/generic_arg_infer.sio
    tests/run-pass/gpu_sedenion_f3.sio
    tests/run-pass/graphics_epistemic_advanced_smoke.sio
    tests/run-pass/graphics_epistemic_view_smoke.sio
    tests/run-pass/graphics_gallery_smoke.sio
    tests/run-pass/graphics_png_export_smoke.sio
    tests/run-pass/graphics_ppm_export_smoke.sio
    tests/run-pass/graphics_smoke.sio
    tests/run-pass/graphics_svg_export_smoke.sio
    tests/run-pass/graphics_tile_smoke.sio
    tests/run-pass/graphics_tiled_ppm_export_smoke.sio
    tests/run-pass/hof_mut_struct_min.sio
    tests/run-pass/if_let_pattern.sio
    tests/run-pass/import_basic.sio
    tests/run-pass/import_basic_main.sio
    tests/run-pass/import_chain_b.sio
    tests/run-pass/import_chain_main.sio
    tests/run-pass/invariant_tests.sio
    tests/run-pass/knightian_syntax.sio
    tests/run-pass/knowledge_octonion_inner.sio
    tests/run-pass/knowledge_value_with_epistemic.sio
    tests/run-pass/match_or_patterns.sio
    tests/run-pass/match_patterns_complete.sio
    tests/run-pass/math_atan_quadrant_reduction.sio
    tests/run-pass/native_enum_basic.sio
    tests/run-pass/native_tokenizer.sio
    tests/run-pass/native_v2_branch_smoke.sio
    tests/run-pass/observe_pattern_ok.sio
    tests/run-pass/observe_with_effect.sio
    tests/run-pass/oct_minimal.sio
    tests/run-pass/octonion_associator_gum_validation.sio
    tests/run-pass/octonion_basic_demo.sio
    tests/run-pass/octonion_basic_ops_standalone.sio
    tests/run-pass/octonion_cayley_dickson.sio
    tests/run-pass/ode_generic_solver.sio
    tests/run-pass/ontology_axiom_conjunction.sio
    tests/run-pass/pkg_manifest_parse_e2e.sio
    tests/run-pass/proof_obligation_basic.sio
    tests/run-pass/proof_search_basic.sio
    tests/run-pass/ptx_maxntid.sio
    tests/run-pass/quadrature.sio
    tests/run-pass/refinement_float_bound.sio
    tests/run-pass/refinement_nested_arithmetic.sio
    tests/run-pass/refinement_satisfied.sio
    tests/run-pass/root_finding.sio
    tests/run-pass/scheduler_machine_reorder.sio
    tests/run-pass/second_order_proof.sio
    tests/run-pass/sedenion_embedding_basic.sio
    tests/run-pass/sensitivity_transcendental.sio
    tests/run-pass/seq_epistemic.sio
    tests/run-pass/seq_kaxi_fuse.sio
    tests/run-pass/seq_knowledge_uncertain.sio
    tests/run-pass/site_screening_selftest.sio
    tests/run-pass/slice_fat_pointers.sio
    tests/run-pass/study_with_audit.sio
    tests/run-pass/test_gp.sio
    tests/run-pass/test_large_import.sio
    tests/run-pass/test_spectral.sio
    tests/run-pass/tmp_mono_repro.sio
    tests/run-pass/trait_basic.sio
    tests/run-pass/trait_bounded_dispatch_multi_call.sio
    tests/run-pass/tuple_destructure_let.sio
    tests/run-pass/uhs_brine_calcite_selftest.sio
    tests/run-pass/unit_cast_compatible.sio
    tests/run-pass/unit_cast_time.sio
    tests/run-pass/unit_decl_keyword.sio
    tests/run-pass/unit_div_cancel.sio
    tests/run-pass/unit_energy_explicit_conversion.sio
    tests/run-pass/unit_same_add.sio
    tests/run-pass/unit_scalar_mul.sio
    tests/run-pass/unobserved_basic.sio
    tests/run-pass/wave_a_algebra_struct.sio
    tests/run-pass/while_for_struct_patterns.sio
