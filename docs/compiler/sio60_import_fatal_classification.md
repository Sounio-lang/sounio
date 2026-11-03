<!-- docs:meta
topic_id: repo.docs.compiler.sio60-import-fatal-classification
authority: repo_only
audience: contributors
last_validated: 2026-11-03
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.sio60-import-fatal-classification
-->

# SIO-60: classification of newly-refused corpus files (lean_single import-fatal)

## Method

The SIO-60 fix makes hard type errors in imported modules fatal on
`lean_single` (ADR-020 option (a)): `tc_mark_failed` no longer early-returns
for `FN_EFFECTS & 2048` (imported) functions, and a new module-level
`let`-scalar check reports E001 for bool/numeric mismatches in `let`/`const`
declarations. E035 effect violations stay import-tolerant (see below).

Census: 2583 `tests/run-pass` files compiled with the committed ELF vs the
fixed ELF. 1716 committed-accepted files re-compiled with the fixed ELF:
**34 unique newly-refused files** (39 refusal rows, 5 duplicates from the
census list).

Load-bearing evidence: **all 34 print at least one `error` diagnostic under
the committed compiler with exit 0** (verified per file; 1–54 diagnostics
each). SIO-60 introduces zero new diagnostics — it only converts already
printed import diagnostics from tolerated to fatal — plus the new
module-level bool E001, which fires on no corpus file (all 4 E001 mentions in
the recount are call-argument/tail classes, verified genuine below).

Madaros oracle (`bin/madaros check`): 29 accept / 5 refuse.

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

Totals: 32 corpus-defect files, 1 checker false positive, 1 checker quirk.
Zero regressions: no file is refused for a diagnostic the pre-fix compiler did
not already print.

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

## Suite handling (keeps CI green)

- 16 files retagged `//@ requires: madaros` (they run and pass on Madaros
  today; lean_single refusal documented per file with its stdlib root).
- 13 files already carried `requires: madaros` — untouched.
- 5 files fail on both engines → `//@ known-failure`:
  `pbpk28_consumer_mass_balance` (pinned: xfail-expect tuple index out of
  bounds, xfail-exit 1; Madaros skips it via `requires: lean_single`),
  `effects_sandbox_integration`, `spearman_holm_bca`,
  `sedenion_embedding_basic`, `graphics_smoke` (unpinned: per-engine
  failure modes differ; any future pass raises the standard XPAS notice).

## Follow-ups (not in this change)

1. Fix the stdlib defects in rows 1–10 (12 root sites) and shrink the
   `requires: madaros` / `known-failure` tags file by file.
2. Fix the tuple-index false positive (row 11) and the svg co-import type
   confusion (row 12); both reproduce without SIO-60.
3. Fatalize E035 in imports once the ~204 missing-effect annotations are
   repaired (the 216-file broad census is kept in the task notes).
