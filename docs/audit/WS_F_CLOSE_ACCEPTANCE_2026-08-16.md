<!-- docs:meta
topic_id: repo.docs.audit.ws-f-close-acceptance-2026-08-16
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.ws-f-close-acceptance-2026-08-16
-->

# WS-F Close Acceptance Criteria (2026-08-16)

**Owner**: grok-cli4 (WS-F close-prep lane)  
**Blocker lifted by**: codex-1 (Madaros E137 import-visibility fix in multi-module emitter, `tools/eisa/eisa_bridge_emit.sio` + `stdlib/eisa/*` imports).  
**Gate**: `scripts/ci/eisa_bridge_conformance_gate_madaros.sh` (updated with crisp post-fix criteria).  
**C3 boundary confirmation**: No overlap. WS-C PR1 (codex-2 `ws-c-pr2-staging` claim) targets `self-hosted/enir/` MIR join/lower + frontier-only additions that do not exist on main. No modification to `tools/eisa/` or `stdlib/eisa/` (explicitly owned by WS-F). Confirmed via `bin/sounio-coord brief` and bus claims (no ownership-conflict blocker). Collision risk zero; PR2 staging manifest respects frontier-only rule.

## Pre-measured Golden Baseline (lean_single, captured 2026-08-16)
- Command: `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tools/eisa/eisa_bridge_emit.sio`
- Full set: **31** `.eisax.elf` files in `artifacts/eisa/` (30 programs + `golden-mul-tampered`).
- Byte-level baseline ready for post-fix `diff` / `sha256sum` verification. Current SHA256 manifest (run `scripts/ci/eisa_bridge_conformance_gate_madaros.sh` post-fix to re-verify):

```bash
# Expected post-E137 (example; regenerate on close)
sha256sum artifacts/eisa/*.eisax.elf | sort
# All must match pre-fix lean_single baseline or produce documented divergence receipt.
```

## Crisp Pass Criteria (post-E137 world)
The gate now enforces **exact** closure conditions (no silent rc=12 acceptance):

1. **Madaros emitter success**: Default `./bin/souc run tools/eisa/eisa_bridge_emit.sio` returns rc=0 (no E137, no visibility preflight failure, full IR/HIR/SIR lowering, 31 ELFs emitted). `madaros_runtime_status=PASS`.
2. **EVM vs Bridge parity**: All 30 program ELFs + tampered produce stdout matching `eisa_evm_run.sio` reference (`diff -u` clean) **or** explicit documented divergence with receipt (currently none).
3. **Byte-identical goldens**: `sha256sum` of all `artifacts/eisa/*.eisax.elf` matches pre-measured lean_single baseline (or explicit "divergence-receipt" marker for intentional Madaros changes).
4. **Tamper-sensitivity & anti-vacuity**: 
   - Tampered golden-mul produces **different** receipt from original.
   - All ELFs contain expected version prefix (`v=1 prog=`, `v=2 prog=`, `v=3 prog=`) but **no** mantissa digit runs (`m[0-9]{8,}`) from receipts (proves computation, not baking).
5. **Tolerated BLOCKED list**: Empty post-P0-F. Any rc=12 must be explicitly in `tolerated_blocked=( )` array with justification (currently none).
6. **No source changes to EISA**: `tools/eisa/` and `stdlib/eisa/` untouched (C3 compliance).

**Closure command** (post-fix):
```bash
./scripts/ci/eisa_bridge_conformance_gate_madaros.sh
# Expect: [eisa-bridge-madaros] PASS: ... madaros_runtime=PASS ... parity=PASS
```

**Acceptance gate**: Gate rc=0 + "PASS: EISA bridge conformance under Madaros" + all 31 goldens byte-identical + no new E137. Update `KNOWN_LIMITATIONS.md` only if residual documented BLOCKED remains.

**Evidence**: Full baseline captured under lean_single. Gate updated with criteria above. Bus confirmation sent. Ready for instant close the moment codex-1 lifts E137 blocker.

Last revised: 2026-08-16 (this lane).  
See also: `scripts/ci/eisa_h_zd_reference_gate.sh` (template), `docs/compiler/KNOWN_LIMITATIONS.md` (D4/E137 class), `MADAROS_FOCUS_PLAN_2026-08-16.md` §WS-F.
