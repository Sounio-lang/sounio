<!-- docs:meta
topic_id: repo.docs.compiler.sio60-import-fatal-classification
authority: repo_only
audience: contributors
last_validated: 2026-11-03
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.sio60-import-fatal-classification
-->

# SIO-60: classification of newly-refused corpus files (lean_single import-fatal)

## Shipped scope (probe-only)

The SIO-60 fix as shipped makes exactly one new check import-fatal on
`lean_single`: a module-level `let`/`const` bool-vs-scalar mismatch
(`let x: i64 = true`) reports E001 via a direct `TYPECHECK_FAILED` set,
in main and imported modules alike (the card acceptance probe). The
change is purely additive (77 insertions, 0 deletions in
`self-hosted/compiler/lean_single.sio`); every other diagnostic keeps its
previous import-tolerance, because full fatality breaks the self-hosted
tree itself (`self-hosted/check/check.sio` relies on i64-as-pointer
punning that a strict import checker refuses — a language-design question
for follow-up, not this card).

## Method (full-fatality census, for follow-up)

A broader build (additionally removing the `FN_EFFECTS & 2048`
early-returns in `tc_mark_failed`/`note_limit_error`) was censused to
classify what full import fatality would refuse. 2583 `tests/run-pass`
files compiled with the committed ELF vs the broad ELF; 1716
committed-accepted files re-compiled: **34 unique newly-refused files**.
Full-suite shards added 31 `tests/stdlib/` files (same classes).

Load-bearing evidence: **all 34 print at least one `error` diagnostic under
the committed compiler with exit 0** (verified per file; 1–54 diagnostics
each). Full fatality introduces zero new diagnostics — it only converts
already printed import diagnostics from tolerated to fatal.

Madaros oracle (`bin/madaros check` over the 34): 29 accept / 5 refuse.

## Verdicts by stdlib root cause

| # | stdlib root | diagnostic (lean) | files | Madaros | verdict |
|---|---|---|---|---|---|
| 1 | `systems/ball_fixed.sio:76,921,926` — calls to `solver_portfolio_v17/v21_*_fingerprint`, never defined in the import closure | E200 undefined identifier | 10 lorenz_ball files | accepts (lax) | **corpus defect** |
| 2 | `os/process.sio:84` — `[]` tail vs declared `[string; 0]` | E218 tail mismatch | 9 (all pre-tagged `requires: madaros`) | accepts | **corpus defect** |
| 3 | `physics/sr.sio:46-47` — `v / C`, `C != 0.0` with module-level `let C: f64` resolved as int (const-scan ignores the annotation) | numeric mismatch | physics_entropy_legacy, physics_sr_compton | accepts | **corpus defect** (relies on const typing lean_single never provided) |
| 4 | `physics/em.sio:296-297` — same int/float const pattern | numeric mismatch | physics_lorentz_legacy | accepts | **corpus defect** (same class as 3) |
| 5 | `compiler/effects/infer.sio` — call argument type vs declared parameter | E001 call-arg | effects_sandbox_integration | refuses (E035, test fns) | **corpus defect** (both engines refuse) |
| 6 | `stats/multiple_testing.sio` (`benjamini_hochberg`, `test_bonferroni`, `test_bh`) — argument type vs parameter | E001 call-arg | spearman_holm_bca | refuses (E009, same root) | **corpus defect** (both engines agree on the root) |
| 7 | `nn/optimizer.sio:173`, `tensor/tape.sio:253` — struct literal missing field | missing field | d4_optimizer_integration | accepts | **corpus defect** |
| 8 | `chemistry/equilibrium.sio:61,66` — return type vs signature + tail mismatch | return-type / E218 | uhs_brine_calcite_selftest, site_screening_selftest | accepts | **corpus defect** |
| 9 | `snn/sedenion_similarity.sio:149` — undefined `len` | E200 | sedenion_embedding_basic | refuses (E004, test-body asserts) | **corpus defect** (both sides defective) |
| 10 | `epistemic/sign.sio:44-45` — `second_order_mean` never defined anywhere (only a comment mentions it, in `epistemic/fo.sio:7`) | E200 | knowledge_sign (pre-tagged) | accepts | **corpus defect** |
| 11 | `epistemic/pce.sio:332` — `nodes.7` flagged out of bounds, but `gauss_legendre_nodes()` returns an 8-tuple (`.0`–`.7` all in bounds); fires identically with the module as main | tuple index OOB | pbpk28_consumer_mass_balance | refuses (E259, unrelated) | **checker false positive** (code is correct; needs checker follow-up) |
| 12 | `graphics/surface.sio:68` — `x >= 0 && x < out.width` (all `i64` by inspection) flagged only when `graphics::svg` is co-imported; a minimal `surface`-only import passes; the committed compiler prints the same diagnostic with exit 0 | ordered-comparison mismatch | graphics_smoke | refuses (fails to parse `svg.sio`) | **checker quirk** (import-context-only type confusion; needs checker follow-up) |

Totals (run-pass census): 32 corpus-defect files, 1 checker false positive,
1 checker quirk. Zero regressions: no file is refused for a diagnostic the
pre-fix compiler did not already print.

## Extension: tests/stdlib failures (full-suite shards)

The run-pass census missed `tests/stdlib/`, where the same newly-fatal
classes refuse 31 more files (all untagged, all passing Madaros check and
green on the Madaros job today). Same verdict — corpus defects, diagnostics
pre-existing under the committed compiler (verified per file: committed
prints the diagnostic with exit 0):

| stdlib root | files |
|---|---|
| `os/process.sio:84` (E218) | test_os |
| `logic/pure/types.sio:59` (E200 `q2`) | test_logic |
| `random/rng.sio:106` (E218) | test_rng |
| `random/distributions.sio:378` (return-type) | test_distributions_e2e |
| `nn/mlp_xor.sio:300` (unknown field) | test_mlp_xor_e2e |
| `graph/algorithms.sio:76,124` (not indexable) | test_algorithms, test_algorithms_e2e |
| `snn/sedenion_layer_core.sio:378` (E001) | test_snn_e2e |
| `graph/entropy.sio:521` (Seq.push receiver) | test_entropy_e2e |
| `quantum/born_shot_law.sio:792,798` (linear-use) | test_qpu_chsh, test_born_shot_law |
| `quantum/qasm_tape.sio:774` (linear-use) | test_qasm_tape |
| `tensor/tape.sio:253` (missing field) | test_tensor_autograd_d2, test_tensor_autograd_d2_hardening, test_nn_primitives_d3, test_caputo_l1_tape |
| `nn/optimizer.sio:173` (missing field) | test_pinn_caputo_residual_d6 |
| `chemistry/equilibrium.sio:61,66` (return-type/E218) | test_kinetics_epistemic_ensemble, test_chem_network_ode_sys5, test_kin_sigma_small_variance, test_kinetics_gri_mech, test_kinetics_core, test_lib_surface, test_pbpk_ontology, test_pbpk_clearance_ode (all retagged); test_equilibrium_acids (manifest entry — the file is website-pinned, so no header edit; `tests/known_failures/hardened_diagnostics_full_suite.txt`) |
| `mem/arena.sio:44,108` (E001) | test_arena_e2e |
| `physics/em.sio:296-297` (numeric) | test_em_e2e |
| `physics/sr.sio:46-47` (numeric) | test_thermo_sr_e2e |
| `particle_physics` (E001 call-arg) | test_particle_physics_core |

Grand totals: 63 corpus-defect files, 1 checker false positive, 1 checker
quirk — 65 files, zero regressions.

## The 204 E035 files (explicitly out of scope)

The broad (pre-narrowing) build newly refused 216 files: 204 E035, 9 E218,
2 E200. The 204 E035 are genuine missing-effect annotations (e.g.
`darwin_pbpk/cumulants.sio:285-288` stores through `out.d1[0]` with only
`with Div`; the committed compiler already emits E035 there). Fatalizing them
would break ~200 currently-passing tests, so E035 stays import-tolerant in
this change. Fixing the stdlib effect annotations is follow-up work; each of
the 204 is a corpus defect of the same kind as rows 1–10 above.

## The 121 vs 34 delta

The card's 121 predates the current corpus and a narrower fatality scope
(E001/E218/E200 fatal; E035 tolerated). Re-running the same methodology on
current `origin/main` yields 34. The classification above covers all 34; the
remaining ~80 historical files are inside the 204 tolerated E035s.

## Suite handling

None needed: the shipped probe-only change fires on no corpus file (the
new bool check hits zero of 2583 run-pass files; the full suite passes
unchanged). The retag/known-failure/manifest options explored during the
full-fatality census were all reverted. When full fatality lands as
follow-up, this classification gives the per-file handling (retag vs
known-failure) for all 65 files.

## Follow-ups (not in this change)

1. Fix the stdlib defects in rows 1–10 (12 root sites) and shrink the
   `requires: madaros` / `known-failure` tags file by file.
2. Fix the tuple-index false positive (row 11) and the svg co-import type
   confusion (row 12); both reproduce without SIO-60.
3. Fatalize E035 in imports once the ~204 missing-effect annotations are
   repaired (the 216-file broad census is kept in the task notes).
