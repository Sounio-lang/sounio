<!-- docs:meta
topic_id: repo.docs.audit.token-ceiling-blocked-runpass-census-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.token-ceiling-blocked-runpass-census-2026-08-17
-->

# Token/source ceiling — blocked run-pass census

**Date:** 2026-08-17  
**Lane:** grok-cli3 / local-only (GitHub outage)  
**Depends on:** E229 refusal (`lane/grok-cli5/token-ceiling-refusal`, commit `cc636846d2`) — honest half  
**This note:** the other half — *which* run-pass tests stay dark, and **which of the three paths** each deserves

## Three paths (do not collapse them)

| Path | When it is correct | When it is wrong |
|------|--------------------|------------------|
| **A. Split the file** | Content is a legitimately large *product* of many independent items that should be separate compilation units | Splitting a single irreducible algorithm mid-proof |
| **B. Raise the 2 097 152 ceiling (with E229 retained)** | A *single reasonable entry* (one module, one concern) measures above the wall and cannot be split without lying about modularity | Raising so a generated megatable “passes” — only moves the next failure |
| **C. Delete/regenerate less** | The bulk is **generated repetition** (copy-paste certificates, version clones, fingerprint catalogs) that should not live as one source file | Deleting unique semantics |

**Rule kept:** never raise the number without a measured cause. E229 must remain at whatever bound ships.

## Instrument (validated before census)

### Positive control — the wall is real

| Module | Bytes | vs CAP 2 097 152 | Approx tokens (lexer-ish regex) |
|--------|------:|------------------:|--------------------------------:|
| `stdlib/theorem/portfolio.sio` | **2 109 065** | **OVER by 11 913** | ~345 576 (≪ token wall) |
| `stdlib/systems/lorenz_i256_cert.sio` | 2 095 899 | under by 1 253 | ~324 925 |

```text
# Measured on prebuilt Madaros (origin/main toolchain in this worktree):
./bin/souc check stdlib/theorem/portfolio.sio
# → parse error at line 50272:24  (silent mid-item clip — pre-E229)
# cumulative UTF-8 bytes first reach CAP exactly at line 50272
```

So the blocking wall for this corpus is the **source byte buffer (CURSOR_SOURCE / lex clip at 2 097 152 bytes)**, not the token-table slot count. Token pressure on these files is ~0.16–0.19 tok/byte — an order of magnitude below 2M tokens.

After E229 lands and Madaros is rebuilt from source, the same input must print **`error[E229]`** (source exceeds lexer byte buffer) instead of a mid-line parse error. That does **not** green the tests; it names the real limit.

### What “41” was

The E229 audit quoted a dispatch figure of **41** run-pass tests without re-counting.  
**This census re-measured on `origin/main` sources:**

| Query | Count |
|-------|------:|
| `tests/run-pass/**/*.sio` containing `use theorem::portfolio::*` | **169** |
| of which already `//@ known-failure` | **169** (all) |
| of which `*_imported.sio` | **169** (all) |
| of which `*_tiny.sio` with that import | **0** |

Tiny portfolio smokes **do not** import the megamodule (they bind scalar fingerprints locally) — **181** `solver_portfolio_*tiny*` files are outside this block class.

**Honest replacement for “41”:** **169** imported run-pass tests are blocked because their import closure cannot parse `stdlib/theorem/portfolio.sio` past the source-byte wall. Treat “41” as a stale underspecified figure, not a second set.

Machine list: `.scratch/token_ceiling_blocked_runpass.tsv` (session) / table below.

## Root module diagnosis — path **C then A**, not B

### `stdlib/theorem/portfolio.sio` (the blocker)

| Fact | Value |
|------|------:|
| Bytes | 2 109 065 |
| Lines | 50 536 |
| `pub fn` count | **2 160** |
| Long integer literals | ~10 916 |
| Dominant shape | Catalog of `portfolio_kind_*`, `known_solver_portfolio_manifest_*`, `portfolio_checker_lorenz_*`, per-version `solver_portfolio_vN_*` binders |
| Bytes past CAP | 11 913 (~264 lines after clip) |
| Clip site | Line **50272** mid-parameter list of a still-open item |

This is **not** one irreducible algorithm that needs a bigger buffer. It is a **grown certificate/fingerprint catalog** — generated/repetitive structure (hundreds of near-clone versioned binders).  

**Primary path: C (stop monolithic growth) + A (split by family into separate modules).**  
**Path B (raise ceiling) is the wrong permanent answer** here: +12 KiB would green today’s tip and fail again at the next portfolio version dump.

Suggested split axes (fn-name buckets, measured counts):

| Bucket (name heuristic) | ~pub fn |
|-------------------------|--------:|
| core / manifests / kinds / non-Lorenz | large remainder |
| `lorenz_i256_step1` … `step6` | ~80 each |
| `lorenz_i256_cover_child0/1` | ~85 / ~47 |
| `lorenz_i256_other` | ~122 |
| sat/lrat / smt / pb / lorenz_ball | small |

Each child module should stay **comfortably under** ~1.0–1.5 MiB with headroom. Re-export a thin `theorem::portfolio` façade only if needed — façade must not re-concatenate source into one lex unit.

### `stdlib/systems/lorenz_i256_cert.sio` (near-wall companion)

| Fact | Value |
|------|------:|
| Bytes | 2 095 899 (1 253 under CAP) |
| `pub fn` | ~700 |
| Shape | Same disease: step/cover certificate checkers, repeated guard patterns (`ok_mask`, `global_flowpipe_claim_mask`, …) |

Checking this file today **still fails** because its import closure pulls `theorem/portfolio.sio` (measured log: parse failure attributed to portfolio at line 50272). Even after portfolio is split, **lorenz_i256_cert is one version bump from the wall** — classify **A/C preemptively**, not B.

### Token wall vs byte wall

For this blocked set, raising **token** capacity alone would **not** help: portfolio is ~0.35M tokens. The live failure mode is **byte clip**. E229 must cover **both** walls (as grok-cli5’s fix does).

## Path choice for the 169 tests

Every row shares the same cause and the same choice:

| Field | Value |
|-------|--------|
| **Block reason** | Import of `stdlib/theorem/portfolio.sio`, which exceeds the **2 097 152-byte** lexer source buffer |
| **Path class** | **A+C** — split/regenerate the megamodule; do **not** raise the ceiling to green these |
| **Test file itself** | Small (typically few KB); not oversized; not the thing to bisect |
| **Already annotated** | All 169 carry `//@ known-failure` about imported portfolio / Madaros multimodule |

**Do not bisect the test files.** Bisect, if ever needed on a large *module*, must delete **complete items** with balanced `{}` `()` `[]` (dispatch rule). The correct surgery is on `portfolio.sio` boundaries between complete `pub fn` items / version families — not brace-unbalanced cuts.

### What would wrongly look like path B

“Raise CAP to 4 MiB so portfolio parses.”  
Measured: portfolio grows by whole version clones (`solver_portfolio_vN_*` triples). A higher CAP without modularity is a **scheduled regression**. Only consider B if, after split, a **single** coherent module still measures over CAP with a written memory budget — and **keep E229**.

## Full blocked list (169)

Format: `path` — family slug — portfolio version in filename (if any).

- `tests/run-pass/lorenz_i256_portfolio_v16_composed_imported.sio` — `lorenz_i256_portfolio_composed` v16
- `tests/run-pass/solver_portfolio_erdos_scope_v30_imported.sio` — `erdos_scope` v30
- `tests/run-pass/solver_portfolio_lorenz_ball_fixed_explicit_enclosure_v27_imported.sio` — `lorenz_ball_fixed_explicit_enclosure` v27
- `tests/run-pass/solver_portfolio_lorenz_ball_fixed_policy_chain_v24_imported.sio` — `lorenz_ball_fixed_policy_chain` v24
- `tests/run-pass/solver_portfolio_lorenz_ball_fixed_radius_budget_v19_imported.sio` — `lorenz_ball_fixed_radius_budget` v19
- `tests/run-pass/solver_portfolio_lorenz_ball_fixed_step_policy_guard_v23_imported.sio` — `lorenz_ball_fixed_step_policy_guard` v23
- `tests/run-pass/solver_portfolio_lorenz_ball_fixed_step_policy_margin_v22_imported.sio` — `lorenz_ball_fixed_step_policy_margin` v22
- `tests/run-pass/solver_portfolio_lorenz_ball_fixed_step_policy_v21_imported.sio` — `lorenz_ball_fixed_step_policy` v21
- `tests/run-pass/solver_portfolio_lorenz_ball_fixed_v17_imported.sio` — `lorenz_ball_fixed` v17
- `tests/run-pass/solver_portfolio_lorenz_i256_ball_fixed_bridge_v29_imported.sio` — `lorenz_i256_ball_fixed_bridge` v29
- `tests/run-pass/solver_portfolio_lorenz_i256_cover_child0_axis_arithmetic_bundle_v164_imported.sio` — `lorenz_i256_cover_child0_axis_arithmetic_bundle` v164
- `tests/run-pass/solver_portfolio_lorenz_i256_cover_child0_axis_arithmetic_readiness_v165_imported.sio` — `lorenz_i256_cover_child0_axis_arithmetic_readiness` v165
- `tests/run-pass/solver_portfolio_lorenz_i256_cover_child0_axis_witness_bundle_v160_imported.sio` — `lorenz_i256_cover_child0_axis_witness_bundle` v160
- `tests/run-pass/solver_portfolio_lorenz_i256_cover_child0_containment_obligation_v155_imported.sio` — `lorenz_i256_cover_child0_containment_obligation` v155
- `tests/run-pass/solver_portfolio_lorenz_i256_cover_child0_discharge_preflight_v167_imported.sio` — `lorenz_i256_cover_child0_discharge_preflight` v167
- `tests/run-pass/solver_portfolio_lorenz_i256_cover_child0_local_flowpipe_preflight_v152_imported.sio` — `lorenz_i256_cover_child0_local_flowpipe_preflight` v152
- `tests/run-pass/solver_portfolio_lorenz_i256_cover_child0_local_flowpipe_proof_skeleton_v153_imported.sio` — `lorenz_i256_cover_child0_local_flowpipe_proof_skeleton` v153
- `tests/run-pass/solver_portfolio_lorenz_i256_cover_child0_local_flowpipe_replay_executor_v154_imported.sio` — `lorenz_i256_cover_child0_local_flowpipe_replay_executor` v154
- `tests/run-pass/solver_portfolio_lorenz_i256_cover_child0_obligation_seed_v151_imported.sio` — `lorenz_i256_cover_child0_obligation_seed` v151
- `tests/run-pass/solver_portfolio_lorenz_i256_cover_child0_validation_core_v166_imported.sio` — `lorenz_i256_cover_child0_validation_core` v166
- `tests/run-pass/solver_portfolio_lorenz_i256_cover_child0_validation_guard_v156_imported.sio` — `lorenz_i256_cover_child0_validation_guard` v156
- `tests/run-pass/solver_portfolio_lorenz_i256_cover_child0_x_axis_arithmetic_validator_v161_imported.sio` — `lorenz_i256_cover_child0_x_axis_arithmetic_validator` v161
- `tests/run-pass/solver_portfolio_lorenz_i256_cover_child0_x_axis_validation_witness_v157_imported.sio` — `lorenz_i256_cover_child0_x_axis_validation_witness` v157
- `tests/run-pass/solver_portfolio_lorenz_i256_cover_child0_y_axis_arithmetic_validator_v162_imported.sio` — `lorenz_i256_cover_child0_y_axis_arithmetic_validator` v162
- `tests/run-pass/solver_portfolio_lorenz_i256_cover_child0_y_axis_validation_witness_v158_imported.sio` — `lorenz_i256_cover_child0_y_axis_validation_witness` v158
- `tests/run-pass/solver_portfolio_lorenz_i256_cover_child0_z_axis_arithmetic_validator_v163_imported.sio` — `lorenz_i256_cover_child0_z_axis_arithmetic_validator` v163
- `tests/run-pass/solver_portfolio_lorenz_i256_cover_child0_z_axis_validation_witness_v159_imported.sio` — `lorenz_i256_cover_child0_z_axis_validation_witness` v159
- `tests/run-pass/solver_portfolio_lorenz_i256_cover_child1_axis_witness_bundle_v177_imported.sio` — `lorenz_i256_cover_child1_axis_witness_bundle` v177
- `tests/run-pass/solver_portfolio_lorenz_i256_cover_child1_containment_obligation_v172_imported.sio` — `lorenz_i256_cover_child1_containment_obligation` v172
- `tests/run-pass/solver_portfolio_lorenz_i256_cover_child1_local_flowpipe_preflight_v169_imported.sio` — `lorenz_i256_cover_child1_local_flowpipe_preflight` v169
- `tests/run-pass/solver_portfolio_lorenz_i256_cover_child1_local_flowpipe_proof_skeleton_v170_imported.sio` — `lorenz_i256_cover_child1_local_flowpipe_proof_skeleton` v170
- `tests/run-pass/solver_portfolio_lorenz_i256_cover_child1_local_flowpipe_replay_executor_v171_imported.sio` — `lorenz_i256_cover_child1_local_flowpipe_replay_executor` v171
- `tests/run-pass/solver_portfolio_lorenz_i256_cover_child1_obligation_seed_v168_imported.sio` — `lorenz_i256_cover_child1_obligation_seed` v168
- `tests/run-pass/solver_portfolio_lorenz_i256_cover_child1_validation_guard_v173_imported.sio` — `lorenz_i256_cover_child1_validation_guard` v173
- `tests/run-pass/solver_portfolio_lorenz_i256_cover_child1_x_axis_validation_witness_v174_imported.sio` — `lorenz_i256_cover_child1_x_axis_validation_witness` v174
- `tests/run-pass/solver_portfolio_lorenz_i256_cover_child1_y_axis_validation_witness_v175_imported.sio` — `lorenz_i256_cover_child1_y_axis_validation_witness` v175
- `tests/run-pass/solver_portfolio_lorenz_i256_cover_child1_z_axis_validation_witness_v176_imported.sio` — `lorenz_i256_cover_child1_z_axis_validation_witness` v176
- `tests/run-pass/solver_portfolio_lorenz_i256_cover_refinement_ledger_v150_imported.sio` — `lorenz_i256_cover_refinement_ledger` v150
- `tests/run-pass/solver_portfolio_lorenz_i256_finite_cover_candidate_v149_imported.sio` — `lorenz_i256_finite_cover_candidate` v149
- `tests/run-pass/solver_portfolio_lorenz_i256_five_step_local_flowpipe_chain_v147_imported.sio` — `lorenz_i256_five_step_local_flowpipe_chain` v147
- `tests/run-pass/solver_portfolio_lorenz_i256_global_flowpipe_claim_preflight_v148_imported.sio` — `lorenz_i256_global_flowpipe_claim_preflight` v148
- `tests/run-pass/solver_portfolio_lorenz_i256_projection_certificate_envelope_v37_imported.sio` — `lorenz_i256_projection_certificate_envelope` v37
- `tests/run-pass/solver_portfolio_lorenz_i256_projection_dependency_dag_v38_imported.sio` — `lorenz_i256_projection_dependency_dag` v38
- `tests/run-pass/solver_portfolio_lorenz_i256_projection_directed_rounding_v35_imported.sio` — `lorenz_i256_projection_directed_rounding` v35
- `tests/run-pass/solver_portfolio_lorenz_i256_projection_div_witness_v34_imported.sio` — `lorenz_i256_projection_div_witness` v34
- `tests/run-pass/solver_portfolio_lorenz_i256_projection_inclusion_v26_imported.sio` — `lorenz_i256_projection_inclusion` v26
- `tests/run-pass/solver_portfolio_lorenz_i256_projection_interval_containment_v36_imported.sio` — `lorenz_i256_projection_interval_containment` v36
- `tests/run-pass/solver_portfolio_lorenz_i256_projection_margin_v33_imported.sio` — `lorenz_i256_projection_margin` v33
- `tests/run-pass/solver_portfolio_lorenz_i256_projection_roundoff_v32_imported.sio` — `lorenz_i256_projection_roundoff` v32
- `tests/run-pass/solver_portfolio_lorenz_i256_range_budget_v31_imported.sio` — `lorenz_i256_range_budget` v31
- `tests/run-pass/solver_portfolio_lorenz_i256_step1_step2_local_flowpipe_bridge_v143_imported.sio` — `lorenz_i256_step1_step2_local_flowpipe_bridge` v143
- `tests/run-pass/solver_portfolio_lorenz_i256_step1_taylor2_center_artifact_v131_imported.sio` — `lorenz_i256_step1_taylor2_center_artifact` v131
- `tests/run-pass/solver_portfolio_lorenz_i256_step1_taylor2_completed_candidate_bundle_v134_imported.sio` — `lorenz_i256_step1_taylor2_completed_candidate_bundle` v134
- `tests/run-pass/solver_portfolio_lorenz_i256_step1_taylor2_enclosure_validator_guard_v138_imported.sio` — `lorenz_i256_step1_taylor2_enclosure_validator_guard` v138
- `tests/run-pass/solver_portfolio_lorenz_i256_step1_taylor2_flowpipe_obligation_v141_imported.sio` — `lorenz_i256_step1_taylor2_flowpipe_obligation` v141
- `tests/run-pass/solver_portfolio_lorenz_i256_step1_taylor2_local_containment_obligation_v139_imported.sio` — `lorenz_i256_step1_taylor2_local_containment_obligation` v139
- `tests/run-pass/solver_portfolio_lorenz_i256_step1_taylor2_local_flowpipe_proof_v142_imported.sio` — `lorenz_i256_step1_taylor2_local_flowpipe_proof` v142
- `tests/run-pass/solver_portfolio_lorenz_i256_step1_taylor2_local_flowpipe_seed_v128_imported.sio` — `lorenz_i256_step1_taylor2_local_flowpipe_seed` v128
- `tests/run-pass/solver_portfolio_lorenz_i256_step1_taylor2_point_time_slab_containment_v129_imported.sio` — `lorenz_i256_step1_taylor2_point_time_slab_containment` v129
- `tests/run-pass/solver_portfolio_lorenz_i256_step1_taylor2_proof_trace_skeleton_v135_imported.sio` — `lorenz_i256_step1_taylor2_proof_trace_skeleton` v135
- `tests/run-pass/solver_portfolio_lorenz_i256_step1_taylor2_radius_artifact_v132_imported.sio` — `lorenz_i256_step1_taylor2_radius_artifact` v132
- `tests/run-pass/solver_portfolio_lorenz_i256_step1_taylor2_remainder_obligation_v133_imported.sio` — `lorenz_i256_step1_taylor2_remainder_obligation` v133
- `tests/run-pass/solver_portfolio_lorenz_i256_step1_taylor2_replay_executor_v137_imported.sio` — `lorenz_i256_step1_taylor2_replay_executor` v137
- `tests/run-pass/solver_portfolio_lorenz_i256_step1_taylor2_replay_preflight_v136_imported.sio` — `lorenz_i256_step1_taylor2_replay_preflight` v136
- `tests/run-pass/solver_portfolio_lorenz_i256_step1_taylor2_response_envelope_v130_imported.sio` — `lorenz_i256_step1_taylor2_response_envelope` v130
- `tests/run-pass/solver_portfolio_lorenz_i256_step1_taylor2_time_slab_containment_v140_imported.sio` — `lorenz_i256_step1_taylor2_time_slab_containment` v140
- `tests/run-pass/solver_portfolio_lorenz_i256_step2_step3_local_flowpipe_bridge_v144_imported.sio` — `lorenz_i256_step2_step3_local_flowpipe_bridge` v144
- `tests/run-pass/solver_portfolio_lorenz_i256_step2_taylor2_center_artifact_v116_imported.sio` — `lorenz_i256_step2_taylor2_center_artifact` v116
- `tests/run-pass/solver_portfolio_lorenz_i256_step2_taylor2_completed_candidate_bundle_v119_imported.sio` — `lorenz_i256_step2_taylor2_completed_candidate_bundle` v119
- `tests/run-pass/solver_portfolio_lorenz_i256_step2_taylor2_enclosure_validator_guard_v123_imported.sio` — `lorenz_i256_step2_taylor2_enclosure_validator_guard` v123
- `tests/run-pass/solver_portfolio_lorenz_i256_step2_taylor2_flowpipe_obligation_v126_imported.sio` — `lorenz_i256_step2_taylor2_flowpipe_obligation` v126
- `tests/run-pass/solver_portfolio_lorenz_i256_step2_taylor2_local_containment_obligation_v124_imported.sio` — `lorenz_i256_step2_taylor2_local_containment_obligation` v124
- `tests/run-pass/solver_portfolio_lorenz_i256_step2_taylor2_local_flowpipe_proof_v127_imported.sio` — `lorenz_i256_step2_taylor2_local_flowpipe_proof` v127
- `tests/run-pass/solver_portfolio_lorenz_i256_step2_taylor2_local_flowpipe_seed_v113_imported.sio` — `lorenz_i256_step2_taylor2_local_flowpipe_seed` v113
- `tests/run-pass/solver_portfolio_lorenz_i256_step2_taylor2_point_time_slab_containment_v114_imported.sio` — `lorenz_i256_step2_taylor2_point_time_slab_containment` v114
- `tests/run-pass/solver_portfolio_lorenz_i256_step2_taylor2_proof_trace_skeleton_v120_imported.sio` — `lorenz_i256_step2_taylor2_proof_trace_skeleton` v120
- `tests/run-pass/solver_portfolio_lorenz_i256_step2_taylor2_radius_artifact_v117_imported.sio` — `lorenz_i256_step2_taylor2_radius_artifact` v117
- `tests/run-pass/solver_portfolio_lorenz_i256_step2_taylor2_remainder_obligation_v118_imported.sio` — `lorenz_i256_step2_taylor2_remainder_obligation` v118
- `tests/run-pass/solver_portfolio_lorenz_i256_step2_taylor2_replay_executor_v122_imported.sio` — `lorenz_i256_step2_taylor2_replay_executor` v122
- `tests/run-pass/solver_portfolio_lorenz_i256_step2_taylor2_replay_preflight_v121_imported.sio` — `lorenz_i256_step2_taylor2_replay_preflight` v121
- `tests/run-pass/solver_portfolio_lorenz_i256_step2_taylor2_response_envelope_v115_imported.sio` — `lorenz_i256_step2_taylor2_response_envelope` v115
- `tests/run-pass/solver_portfolio_lorenz_i256_step2_taylor2_time_slab_containment_v125_imported.sio` — `lorenz_i256_step2_taylor2_time_slab_containment` v125
- `tests/run-pass/solver_portfolio_lorenz_i256_step3_step4_local_flowpipe_bridge_v145_imported.sio` — `lorenz_i256_step3_step4_local_flowpipe_bridge` v145
- `tests/run-pass/solver_portfolio_lorenz_i256_step3_taylor2_center_artifact_v101_imported.sio` — `lorenz_i256_step3_taylor2_center_artifact` v101
- `tests/run-pass/solver_portfolio_lorenz_i256_step3_taylor2_completed_candidate_bundle_v104_imported.sio` — `lorenz_i256_step3_taylor2_completed_candidate_bundle` v104
- `tests/run-pass/solver_portfolio_lorenz_i256_step3_taylor2_enclosure_validator_guard_v108_imported.sio` — `lorenz_i256_step3_taylor2_enclosure_validator_guard` v108
- `tests/run-pass/solver_portfolio_lorenz_i256_step3_taylor2_flowpipe_obligation_v111_imported.sio` — `lorenz_i256_step3_taylor2_flowpipe_obligation` v111
- `tests/run-pass/solver_portfolio_lorenz_i256_step3_taylor2_local_containment_obligation_v109_imported.sio` — `lorenz_i256_step3_taylor2_local_containment_obligation` v109
- `tests/run-pass/solver_portfolio_lorenz_i256_step3_taylor2_local_flowpipe_proof_v112_imported.sio` — `lorenz_i256_step3_taylor2_local_flowpipe_proof` v112
- `tests/run-pass/solver_portfolio_lorenz_i256_step3_taylor2_local_flowpipe_seed_v98_imported.sio` — `lorenz_i256_step3_taylor2_local_flowpipe_seed` v98
- `tests/run-pass/solver_portfolio_lorenz_i256_step3_taylor2_point_time_slab_containment_v99_imported.sio` — `lorenz_i256_step3_taylor2_point_time_slab_containment` v99
- `tests/run-pass/solver_portfolio_lorenz_i256_step3_taylor2_proof_trace_skeleton_v105_imported.sio` — `lorenz_i256_step3_taylor2_proof_trace_skeleton` v105
- `tests/run-pass/solver_portfolio_lorenz_i256_step3_taylor2_radius_artifact_v102_imported.sio` — `lorenz_i256_step3_taylor2_radius_artifact` v102
- `tests/run-pass/solver_portfolio_lorenz_i256_step3_taylor2_remainder_obligation_v103_imported.sio` — `lorenz_i256_step3_taylor2_remainder_obligation` v103
- `tests/run-pass/solver_portfolio_lorenz_i256_step3_taylor2_replay_executor_v107_imported.sio` — `lorenz_i256_step3_taylor2_replay_executor` v107
- `tests/run-pass/solver_portfolio_lorenz_i256_step3_taylor2_replay_preflight_v106_imported.sio` — `lorenz_i256_step3_taylor2_replay_preflight` v106
- `tests/run-pass/solver_portfolio_lorenz_i256_step3_taylor2_response_envelope_v100_imported.sio` — `lorenz_i256_step3_taylor2_response_envelope` v100
- `tests/run-pass/solver_portfolio_lorenz_i256_step3_taylor2_time_slab_containment_v110_imported.sio` — `lorenz_i256_step3_taylor2_time_slab_containment` v110
- `tests/run-pass/solver_portfolio_lorenz_i256_step4_step5_local_flowpipe_bridge_v146_imported.sio` — `lorenz_i256_step4_step5_local_flowpipe_bridge` v146
- `tests/run-pass/solver_portfolio_lorenz_i256_step4_taylor2_center_artifact_v86_imported.sio` — `lorenz_i256_step4_taylor2_center_artifact` v86
- `tests/run-pass/solver_portfolio_lorenz_i256_step4_taylor2_completed_candidate_bundle_v89_imported.sio` — `lorenz_i256_step4_taylor2_completed_candidate_bundle` v89
- `tests/run-pass/solver_portfolio_lorenz_i256_step4_taylor2_enclosure_validator_guard_v93_imported.sio` — `lorenz_i256_step4_taylor2_enclosure_validator_guard` v93
- `tests/run-pass/solver_portfolio_lorenz_i256_step4_taylor2_flowpipe_obligation_v96_imported.sio` — `lorenz_i256_step4_taylor2_flowpipe_obligation` v96
- `tests/run-pass/solver_portfolio_lorenz_i256_step4_taylor2_local_containment_obligation_v94_imported.sio` — `lorenz_i256_step4_taylor2_local_containment_obligation` v94
- `tests/run-pass/solver_portfolio_lorenz_i256_step4_taylor2_local_flowpipe_proof_v97_imported.sio` — `lorenz_i256_step4_taylor2_local_flowpipe_proof` v97
- `tests/run-pass/solver_portfolio_lorenz_i256_step4_taylor2_local_flowpipe_seed_v83_imported.sio` — `lorenz_i256_step4_taylor2_local_flowpipe_seed` v83
- `tests/run-pass/solver_portfolio_lorenz_i256_step4_taylor2_point_time_slab_containment_v84_imported.sio` — `lorenz_i256_step4_taylor2_point_time_slab_containment` v84
- `tests/run-pass/solver_portfolio_lorenz_i256_step4_taylor2_proof_trace_skeleton_v90_imported.sio` — `lorenz_i256_step4_taylor2_proof_trace_skeleton` v90
- `tests/run-pass/solver_portfolio_lorenz_i256_step4_taylor2_radius_artifact_v87_imported.sio` — `lorenz_i256_step4_taylor2_radius_artifact` v87
- `tests/run-pass/solver_portfolio_lorenz_i256_step4_taylor2_remainder_obligation_v88_imported.sio` — `lorenz_i256_step4_taylor2_remainder_obligation` v88
- `tests/run-pass/solver_portfolio_lorenz_i256_step4_taylor2_replay_executor_v92_imported.sio` — `lorenz_i256_step4_taylor2_replay_executor` v92
- `tests/run-pass/solver_portfolio_lorenz_i256_step4_taylor2_replay_preflight_v91_imported.sio` — `lorenz_i256_step4_taylor2_replay_preflight` v91
- `tests/run-pass/solver_portfolio_lorenz_i256_step4_taylor2_response_envelope_v85_imported.sio` — `lorenz_i256_step4_taylor2_response_envelope` v85
- `tests/run-pass/solver_portfolio_lorenz_i256_step4_taylor2_time_slab_containment_v95_imported.sio` — `lorenz_i256_step4_taylor2_time_slab_containment` v95
- `tests/run-pass/solver_portfolio_lorenz_i256_step5_taylor2_center_artifact_v71_imported.sio` — `lorenz_i256_step5_taylor2_center_artifact` v71
- `tests/run-pass/solver_portfolio_lorenz_i256_step5_taylor2_completed_candidate_bundle_v74_imported.sio` — `lorenz_i256_step5_taylor2_completed_candidate_bundle` v74
- `tests/run-pass/solver_portfolio_lorenz_i256_step5_taylor2_enclosure_validator_guard_v78_imported.sio` — `lorenz_i256_step5_taylor2_enclosure_validator_guard` v78
- `tests/run-pass/solver_portfolio_lorenz_i256_step5_taylor2_flowpipe_obligation_v81_imported.sio` — `lorenz_i256_step5_taylor2_flowpipe_obligation` v81
- `tests/run-pass/solver_portfolio_lorenz_i256_step5_taylor2_local_containment_obligation_v79_imported.sio` — `lorenz_i256_step5_taylor2_local_containment_obligation` v79
- `tests/run-pass/solver_portfolio_lorenz_i256_step5_taylor2_local_flowpipe_proof_v82_imported.sio` — `lorenz_i256_step5_taylor2_local_flowpipe_proof` v82
- `tests/run-pass/solver_portfolio_lorenz_i256_step5_taylor2_local_flowpipe_seed_v68_imported.sio` — `lorenz_i256_step5_taylor2_local_flowpipe_seed` v68
- `tests/run-pass/solver_portfolio_lorenz_i256_step5_taylor2_point_time_slab_containment_v69_imported.sio` — `lorenz_i256_step5_taylor2_point_time_slab_containment` v69
- `tests/run-pass/solver_portfolio_lorenz_i256_step5_taylor2_proof_trace_skeleton_v75_imported.sio` — `lorenz_i256_step5_taylor2_proof_trace_skeleton` v75
- `tests/run-pass/solver_portfolio_lorenz_i256_step5_taylor2_radius_artifact_v72_imported.sio` — `lorenz_i256_step5_taylor2_radius_artifact` v72
- `tests/run-pass/solver_portfolio_lorenz_i256_step5_taylor2_remainder_obligation_v73_imported.sio` — `lorenz_i256_step5_taylor2_remainder_obligation` v73
- `tests/run-pass/solver_portfolio_lorenz_i256_step5_taylor2_replay_executor_v77_imported.sio` — `lorenz_i256_step5_taylor2_replay_executor` v77
- `tests/run-pass/solver_portfolio_lorenz_i256_step5_taylor2_replay_preflight_v76_imported.sio` — `lorenz_i256_step5_taylor2_replay_preflight` v76
- `tests/run-pass/solver_portfolio_lorenz_i256_step5_taylor2_response_envelope_v70_imported.sio` — `lorenz_i256_step5_taylor2_response_envelope` v70
- `tests/run-pass/solver_portfolio_lorenz_i256_step5_taylor2_time_slab_containment_v80_imported.sio` — `lorenz_i256_step5_taylor2_time_slab_containment` v80
- `tests/run-pass/solver_portfolio_lorenz_i256_step6_center_artifact_v50_imported.sio` — `lorenz_i256_step6_center_artifact` v50
- `tests/run-pass/solver_portfolio_lorenz_i256_step6_completed_candidate_bundle_v55_imported.sio` — `lorenz_i256_step6_completed_candidate_bundle` v55
- `tests/run-pass/solver_portfolio_lorenz_i256_step6_enclosure_candidate_bundle_v53_imported.sio` — `lorenz_i256_step6_enclosure_candidate_bundle` v53
- `tests/run-pass/solver_portfolio_lorenz_i256_step6_enclosure_projection_artifact_v52_imported.sio` — `lorenz_i256_step6_enclosure_projection_artifact` v52
- `tests/run-pass/solver_portfolio_lorenz_i256_step6_enclosure_validator_guard_v59_imported.sio` — `lorenz_i256_step6_enclosure_validator_guard` v59
- `tests/run-pass/solver_portfolio_lorenz_i256_step6_proof_trace_skeleton_v56_imported.sio` — `lorenz_i256_step6_proof_trace_skeleton` v56
- `tests/run-pass/solver_portfolio_lorenz_i256_step6_radius_artifact_v51_imported.sio` — `lorenz_i256_step6_radius_artifact` v51
- `tests/run-pass/solver_portfolio_lorenz_i256_step6_remainder_obligation_v54_imported.sio` — `lorenz_i256_step6_remainder_obligation` v54
- `tests/run-pass/solver_portfolio_lorenz_i256_step6_repaired_projection_inclusion_validator_v61_imported.sio` — `lorenz_i256_step6_repaired_projection_inclusion_validator` v61
- `tests/run-pass/solver_portfolio_lorenz_i256_step6_replay_executor_v58_imported.sio` — `lorenz_i256_step6_replay_executor` v58
- `tests/run-pass/solver_portfolio_lorenz_i256_step6_replay_preflight_v57_imported.sio` — `lorenz_i256_step6_replay_preflight` v57
- `tests/run-pass/solver_portfolio_lorenz_i256_step6_taylor2_flowpipe_link_preflight_v62_imported.sio` — `lorenz_i256_step6_taylor2_flowpipe_link_preflight` v62
- `tests/run-pass/solver_portfolio_lorenz_i256_step6_taylor2_flowpipe_obligation_v65_imported.sio` — `lorenz_i256_step6_taylor2_flowpipe_obligation` v65
- `tests/run-pass/solver_portfolio_lorenz_i256_step6_taylor2_local_containment_obligation_v63_imported.sio` — `lorenz_i256_step6_taylor2_local_containment_obligation` v63
- `tests/run-pass/solver_portfolio_lorenz_i256_step6_taylor2_local_flowpipe_proof_v66_imported.sio` — `lorenz_i256_step6_taylor2_local_flowpipe_proof` v66
- `tests/run-pass/solver_portfolio_lorenz_i256_step6_taylor2_time_slab_containment_v64_imported.sio` — `lorenz_i256_step6_taylor2_time_slab_containment` v64
- `tests/run-pass/solver_portfolio_lorenz_i256_step6_z_margin_repair_v60_imported.sio` — `lorenz_i256_step6_z_margin_repair` v60
- `tests/run-pass/solver_portfolio_lorenz_i256_taylor2_adaptive_step_decision_v46_imported.sio` — `lorenz_i256_taylor2_adaptive_step_decision` v46
- `tests/run-pass/solver_portfolio_lorenz_i256_taylor2_remainder_v44_imported.sio` — `lorenz_i256_taylor2_remainder` v44
- `tests/run-pass/solver_portfolio_lorenz_i256_taylor2_response_envelope_v49_imported.sio` — `lorenz_i256_taylor2_response_envelope` v49
- `tests/run-pass/solver_portfolio_lorenz_i256_taylor2_step_policy_v45_imported.sio` — `lorenz_i256_taylor2_step_policy` v45
- `tests/run-pass/solver_portfolio_lorenz_i256_taylor2_step_request_v48_imported.sio` — `lorenz_i256_taylor2_step_request` v48
- `tests/run-pass/solver_portfolio_lorenz_i256_taylor2_step_schedule_v47_imported.sio` — `lorenz_i256_taylor2_step_schedule` v47
- `tests/run-pass/solver_portfolio_lorenz_i256_taylor_ball_bridge_v43_imported.sio` — `lorenz_i256_taylor_ball_bridge` v43
- `tests/run-pass/solver_portfolio_lorenz_i256_trajectory5_step6_local_flowpipe_bridge_v67_imported.sio` — `lorenz_i256_trajectory5_step6_local_flowpipe_bridge` v67
- `tests/run-pass/solver_portfolio_lorenz_i256_trajectory_v16_imported.sio` — `lorenz_i256_trajectory` v16
- `tests/run-pass/solver_portfolio_lorenz_wide_precision_ladder_v20_imported.sio` — `lorenz_wide_precision_ladder` v20
- `tests/run-pass/solver_portfolio_lrat_deletion_lifecycle_v28_imported.sio` — `lrat_deletion_lifecycle` v28
- `tests/run-pass/solver_portfolio_pb_kernel_trace_v41_imported.sio` — `pb_kernel_trace` v41
- `tests/run-pass/solver_portfolio_proof_trace_interop_v40_imported.sio` — `proof_trace_interop` v40
- `tests/run-pass/solver_portfolio_sat_frat_elaboration_v25_imported.sio` — `sat_frat_elaboration` v25
- `tests/run-pass/solver_portfolio_smt_alethe_micro_reconstruction_v42_imported.sio` — `smt_alethe_micro_reconstruction` v42
- `tests/run-pass/solver_portfolio_smt_external_proof_v18_imported.sio` — `smt_external_proof` v18
- `tests/run-pass/solver_portfolio_sota_alignment_v39_imported.sio` — `sota_alignment` v39
- `tests/run-pass/solver_portfolio_v16_acceptance_from_counts_imported.sio` — `v16_acceptance_from_counts` v16
- `tests/run-pass/solver_portfolio_v16_acceptance_from_entries_imported.sio` — `v16_acceptance_from_entries` v16
- `tests/run-pass/solver_portfolio_v16_acceptance_receipt_imported.sio` — `v16_acceptance_receipt` v16
- `tests/run-pass/solver_portfolio_v16_audit_receipt_imported.sio` — `v16_audit_receipt` v16
- `tests/run-pass/solver_portfolio_v16_checker_family_coverage_imported.sio` — `v16_checker_family_coverage` v16
- `tests/run-pass/solver_portfolio_v16_coverage_imported.sio` — `v16_coverage` v16

## Not in this blocked set (important negatives)

| Set | Why not blocked by this wall |
|-----|------------------------------|
| `solver_portfolio_*_tiny.sio` (~181) | No `use theorem::portfolio::*`; local scalars only |
| Ordinary `tests/run-pass` math/PBPK files | Max run-pass source ~53 KiB; none near CAP |
| Token-table flood (2M commas) | Separate E229 witness; not these tests |

## Recommended sequence (implementation later — out of census scope)

1. **Land E229** (fail-closed) so clip cannot present as innocent parse errors.  
2. **Split `portfolio.sio`** by family/version into modules each ≪ CAP; fix imports in the 169 tests (or a single façade that does not re-lex a concat blob).  
3. **Pre-split `lorenz_i256_cert.sio`** the same way before it crosses CAP.  
4. **Gate:** `souc check stdlib/theorem/portfolio.sio` (and children) must pass; optional CI max-bytes per stdlib file (e.g. fail if any unit > 1.5 MiB) so the catalog cannot silently re-monolith.  
5. **Only then** revisit CAP with peak measurements on `main.sio` multi-module loads + memory — E229 stays.

## Bottom line

| Question | Answer |
|----------|--------|
| What blocks the tests? | **`stdlib/theorem/portfolio.sio` over the 2 097 152-**byte** source wall** (clip at line 50272) |
| Token table full? | **No** for this corpus (~0.35M tokens) |
| How many run-pass? | **169** imported portfolio tests (dispatch “41” was stale/under-count) |
| Path A split? | **Yes — primary** |
| Path B raise ceiling? | **No** as the fix for these 169 |
| Path C generated bulk? | **Yes — primary cause of size** |
| Bisect the tests? | **No** — fix the module; if bisecting modules, complete items only |

E229 makes the failure honest. **Splitting the generated portfolio catalog** makes the tests runnable. Raising 2 097 152 alone would only postpone the next lie.


## Update 2026-08-17 — split landed (same lane)

`stdlib/theorem/portfolio.sio` was split into a thin façade + `portfolio_*.sio` parts (path **A+C**). Measured: façade and sample of 30/30 former importers `check: OK`; `lorenz_i256_cert.sio` `check: OK`. Gate: `scripts/ci/stdlib_source_byte_ceiling_gate.sh`. Ceiling **not** raised. See `PORTFOLIO_SPLIT_SOURCE_CEILING_2026-08-17.md`.

## Update 2026-08-17 — lorenz cert pre-split (same lane)

`stdlib/systems/lorenz_i256_cert.sio` (2 095 899 B, 1 253 under CAP) was split the same way: thin façade + `lorenz_i256_cert_{core,step1..step6,trajectory5,cover_child0,cover_child1,cover_refinement}.sio`. Largest part ~345 KB. Sequence item 3 is done. Ceiling still not raised.
