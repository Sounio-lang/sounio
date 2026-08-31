<!-- docs:meta
topic_id: repo.docs.audit.pin-count-unwired-honesty-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.pin-count-unwired-honesty-2026-08-18
-->

# `pin_count` is UNWIRED — make absence detectable

**Date:** 2026-08-18  
**Lane:** grok-cli2 / `pin-count-honesty`  
**Context:** rc=182 wall family closed; external probes had been reading `runtime_context.pin_count` as a “live” proxy. On the active Madaros backend that field is **never incremented** after trampoline init — a silent zero of the same honesty class as GUM `var=0.000000` (#1792).

**Not in scope:** implementing pinning / reclamation (grok-cli1 `handle-reclaim` owns `gc.sio`). This change only stops advertising and encoding “zero” as if measured.

---

## 1. Classification

| Question | Answer |
|---|---|
| Vestige of a removed mechanism? | **Partial.** Sprint 90 sketched pinning (`gc.sio` model, `codegen.sio` slow-path pin/live probe before empty-frame reset). The **live** emitter is `codegen_x86_linux.sio`, which fail-closes at 182 **without** consulting pin and **never writes** context `pin_count` after init. |
| Backend where it is written? | **Only init to a constant.** No `add`/`inc` of `runtime_context_field_pin_count` exists under `self-hosted/native/` on either codegen file. Per-**entry** `pin_count` fields are zeroed on alloc (honest “this handle is not pinned”). Context-level count stays stale. |
| Dead surface? | **Dead as a live metric; live as a lie.** Contract JSON still said `pin_registry_ready: true` and `ffi_pinning_model: true`. |

---

## 2. Fix chosen: detectably absent (not “wire pin”)

Wiring real pins is reclaim-adjacent and blocked on death detection (grok-cli1). Honesty first:

1. **`runtime_context_pin_count_unwired() -> -1`** — trampoline inits use this sentinel instead of `0`.
2. **`pin_registry_ready: false`** and **`ffi_pinning_model: false`** in both codegen emitters’ contract JSON.
3. **`scripts/ci/pin_registry_honesty_gate.sh`** — same pattern as `precise_stack_maps_honesty_gate.sh`; wired into Contracts CI.
4. Omega shadow gate + frozen artifacts updated to expect `false`.

A reader that sees `pin_count = -1` or `pin_registry_ready: false` cannot confuse “unmeasured” with “zero live pins.”

---

## 3. Non-claims

- Does not reclaim handles.
- Does not change the 182 fail-closed path.
- Does not edit `gc.sio` (reclaim lane).
- Does not assert that `-1` is a permanent ABI — when pinning ships, init becomes `0` and the honesty gate flips in the **same** commit as the writer.

---

## 4. Document control

| Date | Change |
|---|---|
| 2026-08-18 | Classify pin_count UNWIRED; sentinel + contract honesty + CI gate. |
