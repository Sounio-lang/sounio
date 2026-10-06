<!-- docs:meta
topic_id: repo.docs.audit.rebracket-assoc-gates-census-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: glm-cli1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.rebracket-assoc-gates-census-2026-08-19
-->

# Rebracket/associativity gates — run, classify, map (2026-08-19)

**Lane:** glm-cli1 / `rebracket-gates-census-20260819`
**Question:** of the ten `scripts/ci/` gates whose names carry rebracket/assoc, which run in CI, which pass on today's main, and which have rotted.
**Head measured:** `f9b3147364` (origin/main, 2026-08-19). Each gate executed individually, `ulimit -s 524288`, compiler-path environment scrubbed; rc recorded per gate, no aggregation.

## Instrument validation (before believing any zero)

Wiring probe `git grep -c "<basename>" origin/main -- .github/`:
- positive controls FIRE: `sedenion_associator_1848_gate` → 1 hit in `ci.yml`; `concept_status_gate` → 1 hit (entered `ci.yml` 2026-08-19).
- the nine others return zero `.github/` references.

**Result: 1 of 10 gates is wired. Nine are not — including `exact_bitwise_rebracket_authority_gate.sh`, the gate the concept `docs/internal/concepts/rebracketing-authority.md` (SOUNIO-REBRACKETING-AUTHORITY, `last_validated` 2026-03-07) names as its authority, and `proof_carrying_rebracketing_protocol_gate.sh`.** The concept's status line claims its protocols are "executable under strict hash-bound gates"; for the authority pair that claim is currently not exercised by any CI path.

## Per-gate census

| gate | wired | rc | classification | cause (one line) |
|---|---|---|---|---|
| associator_gum_variance | no | 0 | GREEN, non-vacuous | re-runs the experiment live (`souc run`), receipt verdict PASS |
| dyadic_relational_associator | no | 0 | GREEN | passes on today's main |
| exact_bitwise_rebracket_authority | no | 1 | **ROTTEN** (anchor drift) | greps production anchor `opt_cleanup_module_inplace(&! merged_module)`; call sites now read `(&! (*module_box))` — function and protocol exist, anchor spelling stale |
| exact_bitwise_rebracket_source_ir | no | 1 | **ROTTEN** (anchor drift; silent death) | anchor `(*module).functions[fi as usize] = opt_cleanup_function_with_algebras_and_audit` no longer matches after the IR-SoA landing (#1717); gate dies silently (`set -e` + empty command substitution) instead of failing loudly |
| kretikos_associator_emit | no | 1 | **ROTTEN** (dangling import) | driver `self-hosted/gpu/kretikos_emit_kaxi.sio:34` does `use gpu::erdos90_hc_smoke_emit::*`; that file is absent from origin/main (never merged — snapshot commit `b8828063d6` is off-main ancestry) |
| kretikos_kaxi_phase_z_assoc | no | 0 | GREEN (disclosed deferral) | CPU scaffold passes; TC-4 GPU PTX parity SKIPPED — kernel not built, printed in the gate's own output |
| native_v2_associator | no | 1 | **ROTTEN** (same dangling file) | delegates to the science-spine gate, which dies silently inside `native_v2_driver_self_compile_gate` on `fatal: path 'self-hosted/gpu/erdos90_hc_smoke_emit.sio' does not exist in 'origin/main'` |
| proof_carrying_policy_observation_associator | no | 0 | GREEN | passes on today's main |
| proof_carrying_rebracketing_protocol | no | 0 | GREEN | passes on today's main |
| sedenion_associator_1848 | **yes** | 0 | GREEN + wired | the only CI-reachable gate of the ten, passing |

6 green / 4 red / 0 real-defect. The partial red is itself evidence the instrument ran real work.

## Founder-thesis check (the alert case)

**No gate accuses an effective regrouping on a non-associative type.** Nothing in the four reds reports the compiler rewriting `(a·b)·c` to `a·(b·c)` without authority — all four are anchor/file rot in the gates' own bindings to the tree. No immediate-violation alert is warranted.

The structural finding is different and worse in a quiet way: the two gates that bind the rebracketing-authority claim to the production compiler path (authority + source_ir) are simultaneously (a) unwired and (b) rotted since the IR-SoA rewrite reshaped the code they hash-bind to. Since that landing, the authority protocol's presence at the new call shapes has been re-derived by no one and checked by nothing. A future silent regrouping would not be caught by this pair today.

## Proposed re-connection order (nothing wired — main is red for an unrelated reason, the stale known-failure tag owned by another lane; wiring a red gate would compound it)

1. **Re-derive, then re-anchor, then wire the authority pair.** Re-deriving means confirming the authority protocol survives the post-SoA call shapes (`&! (*module_box)`, arena accessors) — not patching the regexes and walking away. Until then the concept's "executable under strict hash-bound gates" status line overstates.
2. **Resolve the `erdos90_hc_smoke_emit.sio` dangling reference** (land the file or drop the kaxi driver's import — an owner decision; it blocks two gates at once), then wire `kretikos_associator_emit` and `native_v2_associator`.
3. **Wire the four as-is greens** (gum_variance, dyadic, policy_observation, protocol) once main's unrelated red clears.
4. `kretikos_kaxi_phase_z_assoc`: wire with its disclosed GPU deferral, or defer wiring until the PTX kernel exists.

## Map: what the ten cover collectively

- **Authority of rebracketing** (who may rewrite brackets, under which obligation): authority + source_ir — both rotted, both unwired. The thesis-bearing core is the least protected.
- **Proof-carrying protocol/observation**: two greens, unwired.
- **Associator numerics**: sedenion 1848 (green, wired — the only live protection); native_v2 octonion Fano/norm (rotted).
- **Uncertainty over associators**: GUM variance experiment (green, re-runs live, unwired).
- **Dyadic relational associator**: green, unwired.
- **Emission paths** (kretikos K-AXI): emit gate rotted; phase-Z CPU scaffold green with GPU leg deferred.

Coverage on paper is broad; in execution today the non-associativity surface is guarded by exactly one live gate (sedenion 1848) plus four passing-but-unreachable ones, and the authority spine is guarded by none.

## Receipts

- Per-gate logs: `/tmp/rb_<gate-name>.log` (session-local); rcs: 0,0,1,1,1,0,1,0,0,0 in the table order above.
- Anchor evidence: `module_frontend.sio:6502,6588` (`opt_cleanup_module_inplace(&! (*module_box))`); `opt_cleanup.sio` retains `opt_cleanup_function_with_algebras_and_audit` at `:8958,:8961,:9926` in SoA shapes.
- Dangling import: `kretikos_emit_kaxi.sio:34`; manual driver compile fails `error: unreadable import: self-hosted/gpu/erdos90_hc_smoke_emit.sio`; `git cat-file -e origin/main:<path>` → absent; snapshot commit `b8828063d6` not an ancestor of main.
- Silent-death mechanics: source_ir dies at `cleanup_call_line=$(rg -n … | cut -d: -f1)` with empty match under `set -euo pipefail`; spine dies inside the self-compile gate whose output is redirected to `$OUT_DIR/logs/native_v2_driver_self_compile_gate.log`.

**Semantic declaration:** this document is measurement only. No gate was wired, unwired, modified, or reverted; no source file was changed. All statements above were measured on 2026-08-19 at head `f9b3147364` with the wiring probe validated in both directions before any zero was believed.
