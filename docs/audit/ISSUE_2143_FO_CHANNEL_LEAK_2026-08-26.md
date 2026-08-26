<!-- docs:meta
topic_id: repo.docs.audit.issue-2143-fo-channel-leak-2026-08-26
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.issue-2143-fo-channel-leak-2026-08-26
-->

# Issue #2143 — FO channel-table per-function reset (measurement + fix + evidence)

**One line:** the FO channel table (`fo_nchan`, `fo_sigma2`, `fo_bind_count`; and the
write-only `FO_SIGMA_REG` mirror) had its per-function reset stranded in the dead
`begin_function_body_lowering`; the `*mut` port never carried it to the live
`lowerer_lower_fn_item_mut`. The leak is **real but currently masked** — no reproduced
miscompile — so this change is the **defensive sibling of #2145** plus removal of the dead
function. Verified **0 regressions** across 168 FO tests.

## Measurement (self-hosted/ir/lower.sio, base origin/main 25a6cf9c8b)

- `begin_function_body_lowering` (was line 6223) is the pre-`*mut`-refactor by-value
  original of the per-function setup — **0 callers**. Live path: `lowerer_lower_fn_item_mut`
  (4343), one long-lived `&! Lowerer` iterated per function by `lowerer_lower_program_items_mut`.
- Exhaustive assignment audit: `fo_nchan` is set to `0` only inside the dead function and
  grows at the seed; `fo_bind_count` the same. The live entry never touches the FO channel
  table. So these are zeroed once per **module** at construction (the four `lowerer_new*`
  constructors) and accumulate across functions.
- The live channel map is the field **`fo_sigma2`** (read at `fo_seed_from_variance`'s reuse
  check `fo_sigma2[k] == variance_reg`). **`FO_SIGMA_REG` is write-only** — written at the
  seed, never read — a vestigial mirror. Neither is read by the inter-procedural transfer
  (`fo_register_pure_fn_transfer` uses none), so the table is per-function-body-local and
  resetting it is safe.
- The live entry already carries the #2145 correlation reset (`FO_COV_RHO`, 4360) with a
  comment that itself names this gap: "Everything it [the dead fn] resets — fo_nchan,
  fo_sigma2, FO_SIGMA_REG, fo_expr_sens — is therefore never reset by it."

## Why it is masked (the empirical finding)

Three repro strategies against a source-built buggy compiler (Madaros v0.80.0) **all
produced correct results** — no miscompile:

1. **Differential** — `f2` alone vs `f1`-then-`f2`, `f2 = Var(x·y)`, x=2±0.1, y=3±0.1.
   Correct = 0.13. Both gave `f2_var=0.130000` (and `f1_var=407.160000`, also correct).
2. **Saturation** — 18 distinct-prologue FO functions before a target; target still 0.13.
3. **Correlation-adjacent** — covered by #2145 already resetting `FO_COV_RHO` per function.

The leak stays inert because (a) #2145 already resets the one FO global that *manifested*
(`FO_COV_RHO` — its author measured `Var(c+d)=0.16` instead of `0.10`, the previous
function's number entire); (b) `fo_expr_sens` and `FO_HOFF` are cleared **per seed**, so a
stale channel carries no sensitivity and is skipped in the variance/Hessian emit; and (c)
the reuse check keys on vreg-number equality, which is self-consistent in the reusing
function's own IR. So `fo_nchan`/`fo_sigma2`/`fo_bind_count` leaking inflates the channel
count and stales the map, but nothing downstream reads it into a wrong number **today**.

**Honest verdict:** the resets *matter* structurally (they are a genuine cross-function
leak the port dropped) but do **not** currently cause a miscompile I could reproduce. This
is a latent hazard in actively-developed FO code, not a live bug.

## The fix (sibling of #2145)

In `lowerer_lower_fn_item_mut`, beside the #2145 `FO_COV_RHO` reset, added the channel-table
reset:

```
(*lo).fo_nchan = 0
(*lo).fo_bind_count = 0
var fo_sri: i64 = 0
while fo_sri < 32 {
    (*lo).fo_sigma2[fo_sri as usize] = -1
    FO_SIGMA_REG[fo_sri as usize] = -1
    fo_sri = fo_sri + 1
}
```

and **removed the dead `begin_function_body_lowering`** (44 lines, 0 callers) — its presence
is exactly what let the reset rot through the `*mut` port. Updated the #2145 comment (which
referenced it) accordingly. `fo_expr_sens`/`fo_pending_sens` are already per-seed-cleared, so
not re-added here.

## Verification (buggy vs fixed, both source-built)

- Light type-check of the modified `lower.sio`: `check: OK`, 0 errors (large stack).
- Repros on the **fixed** compiler: identical correct results (`f2_var=0.130000`,
  `f1_var=407.160000`, `ftarget_var=0.130000`) — no behaviour change on the correct path.
- **FO regression, 168 tests** (`tests/run-pass/*{fo,gum,hessian,autodiff,beta10,epistemic,pbpk,variance,knowledge,correlat}*`),
  fixed vs buggy: **PASS 138 / FAIL 30 on BOTH; 0 regressions.** Every one of the 30 fails on
  the buggy (main) compiler too — they are pre-existing (madaros FO-across-call/import
  features still under development), not caused by this change. The inter-proc FO tests among
  them fail on both, confirming the reset is not what fails them; the 138 that pass do so
  identically, confirming the reset is safe.

## Status

Fix applied on branch `fable/issue2143-fo-leak` from `origin/main` 25a6cf9c8b, with codex's
confirmation of base and approach. Buggy/fixed codegen receipt preserved above. Framed
honestly as defensive + dead-code removal (no reproduced miscompile), 0 regressions.

## Regression differential (deterministic build-level, 296 tests)

Execution differentials were abandoned as unreliable: the suite is flaky — the *same*
buggy binary run 3× on `linear_match_binding_consumed`, `gpu_hlir_vec4_lane_plan_leaf`,
`knowledge_array` gives self-inconsistent output (intermittent timeout/no-output under
parallel load), so a single-run buggy-vs-fixed comparison measures flakiness, not codegen.

Compilation, by contrast, is deterministic (verified: same binary builds byte-identical
ELF twice). So the authoritative differential compares the **emitted ELF** buggy vs fixed
over 296 tests — 169 FO-seeding (the complete set the fix can touch) + 127 non-FO control:

| result | count |
|---|---|
| **ELF byte-identical** (no codegen change) | **251** |
| **ELF differs** (AFF 0 / CTL 0) | **0** |
| NOBUILD on BOTH binaries (pre-existing) | 45 |
| NOBUILD asymmetric (fix-caused) | **0** |

**DIFF count = 0.** The fix produces byte-identical emitted code for every one of the 251
buildable tests — including all FO-affected ones — and introduces zero new build failures
(the 45 NOBUILDs fail identically on origin/main's buggy binary too). This is the strongest
no-regression proof and independently reconfirms the MASKED finding: the reset changes only
internal Lowerer state that no current input lowers into different machine code.
