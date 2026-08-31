<!-- docs:meta
topic_id: repo.docs.audit.runtime-gc-capability-honesty-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.runtime-gc-capability-honesty-2026-08-18
-->

# P1 after the unwritten-fields census — pin_registry + GC capability honesty

**Date:** 2026-08-18  
**Lane:** grok-cli2 / `runtime-honesty-p1`  
**Follows:** `#1830` (pin_count UNWIRED), `#1834` (census).

## What changed

| Surface | Before | After |
|---|---|---|
| `runtime_context.pin_registry` init | `0` (plausible null) | **`-1` UNWIRED** (`runtime_context_pin_registry_unwired`) |
| `tracing_gc` contract / layout default | `true` | **`false`** |
| `gc_mark_compact_model` | `true` | **`false`** |
| `gc_precise_descriptor_scanning` | `true` | **`false`** |
| `gc_handle_relocation_model` | `true` | **`false`** |
| `gc_runtime_retry_active` | `runtime_metadata_active` | **`false`** (x86_linux 182 is fail-closed) |
| `gc_current_frame_root_scan` | `runtime_metadata_active` | **`false`** |

## Why

Census class B/E + contract lies: readable zeros / true flags where the live Madaros path does not run a collector, does not call `empty_frame_reset`, and does not populate a pin registry. Same honesty class as `#1792` / `#1830`.

## Gates

- Existing `pin_registry_honesty_gate.sh` (pin_registry_ready / ffi_pinning)
- New `gc_capability_honesty_gate.sh` — blocks re-advertising the GC model flags as true under `self-hosted/native/`
- Both wired in Contracts CI

## Non-claims

Does not implement reclaim, pin writers, or Windows IAT fills (P2 in the census). Does not edit `gc.sio`.
