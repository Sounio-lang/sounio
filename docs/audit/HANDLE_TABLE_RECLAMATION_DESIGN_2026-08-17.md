<!-- docs:meta
topic_id: repo.docs.audit.handle-table-reclamation-design-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: grok-cli1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.handle-table-reclamation-design-2026-08-17
-->

# Handle reclamation — how do you know a handle died?

**Owner (implementation + design):** grok-cli1 · lane `handle-reclaim`  
**Write scope claimed:** `self-hosted/native/gc.sio`, `stdlib/runtime/**` (when created), this doc  
**Do not edit without bus message to grok-cli1:** those paths  
**Out of scope for this lane:** `self-hosted/ir/lower.sio`, unrelated codegen drive-bys  

**Verdict:** reclamation is **expensive**, not cheap. The hard question is not “free-list yes/no”; it is **death detection**. Until that is solved honestly, any reclaim path is the #555/#651 class (wipe live state). **Do not implement reclaim yet.**

---

## Shared facts (measured by fleet — do not rediscover)

| Fact | Source |
|---|---|
| pbpk_suite: 24 → 19 fails; ten println-SEGV139 → **5 PASS + 5 rc=182**; rc=182 family **2 → 7** (largest) | grok-cli2 re-measure |
| #1799 turned compile-time death into pass **or** handle wall | same |
| Indexing is `handle_id * 48` (entry stride); **no mask / wrap** on live path | codex-2 refute (`codegen_x86_linux.sio` 6368–6374, 7235–7243, …) |
| Full table → slow path → **`emit_exit(182)`** deliberately | 6395–6398, 7258–7259 |
| Capacity is **2²² = 4 194 304** (`19095d6658` / `gc.sio`), not 2²⁰; bump **moves the wall** | `gc.sio:44–65` |
| **d2_gum** exceeds **3 000 001** handles in one run | fleet measure |
| #555 raised 4096→2²⁰ and left reclaim undone; SoA SEGV = table full + **crude GC reset under live handles** | historical |

---

## The whole question

> **How do you know a handle died?**

A free-list, a generation bump, or a mark-sweep pass are all **mechanisms after death is known**. The previous failure mode skipped death detection and reset the bump heap + `handle_count := 0` while handles were still reachable → wrong values / SEGV.

### What “dead” means here

A handle id `h ∈ 1 .. capacity−1` is dead iff **no live root** still holds `h` (or a pointer derived through a managed object that still contains `h` in a pointer-bitmap field).

Roots in this runtime:

1. Stack / spill slots holding handle ids (or raw ptrs above the id threshold).
2. Callee-saved registers (not fully modelled for GC today).
3. Fields of other managed objects (descriptor `pointer_bitmap`).
4. Anything the “pin registry” was meant to cover (fields exist; **no end-to-end pin/unpin protocol** on the hot path).

If you cannot enumerate (1)–(3) at a safepoint, you **cannot** free.

---

## Three classical answers — and what this backend has

### A. Tracked roots (precise stack maps / safepoint bitmaps)

**Idea:** at every call/loop safepoint, a bitmap says which slots hold handles; mark from those + heap pointer fields; unmarked handles are dead.

**What Madaros has today:**

```text
stack_map_root_temp_counts[i] = temp_count   // COUNT, not bitmap
precise_stack_maps = false                   // honesty gate forbids true
```

Live probe used by the *legacy* reset path (`native_v2_emit_current_frame_live_probe`):

```text
for each stack/spill slot:
  load; test; if nonzero → “frame live”
```

That is **not** “which slots are handles”. Any nonzero i64 (loop index, f64 bits, handle id, raw ptr) counts as live. It both:

- **false-positive live** (blocks reset when only scalars are nonzero), and historically
- **false-negative live** for boxed value state the probe did not see in the walked slots/registers (#651 under-detection narrative),

which is why empty-frame reset was **unwired** on the live `codegen_x86_linux` path in favour of **exit 182**.

**Cost of doing A properly:** weeks-class. Must derive roots from **regalloc liveness**, emit bitmaps (or typed slot tags), walk descriptor bitmaps, flip honesty gate in the **same** commit. Without A, mark-sweep reclaim is unsafe.

### B. Reference counting

**Idea:** each handle entry has a refcount (the `pin_count` field is the natural slot). Inc on copy, dec on drop; free when zero.

**What Madaros has today:**

- Per-entry `pin_count` field, **zeroed at alloc**, never incremented on MIR copy.
- Process-level `pin_count` / `pin_registry` fields, not a working pin API.
- Language has `Drop` in traits / type-erasure **scaffolding**, not native-v2 insert-drop-at-end-of-scope for managed handles.

**Cost of doing B properly:** medium–large *after* IR last-use / copy tracking is trustworthy:

- every MIR move/copy of a handle-shaped temp must inc/dec,
- aggregate field stores that store handle ids must participate,
- cycles in object graphs need a backup (or forbid cycles for RC-managed types).

**Harder than a “normal” RC runtime:** Sounio already mixes **handle ids and raw unboxed pointers** in the same word-sized slots (`native_v2_handle_raw_ptr_threshold`). A blind “nonzero = handle” RC is wrong; you need a **type/tag discipline at every store**, which the current stack maps do not provide.

### C. Generations / epochs

**Idea:** bump a generation; free handles from older generations when no root can point into them (nursery collection).

**What Madaros has today:**

- `collector_epoch` fields and empty-frame reset that **bumps epoch and zeros handle_count** — a degenerate “generation” that assumes **no live roots at all**.

That is exactly the failed design: **generation without root set = wipe**.

A real generational scheme still needs **A** (roots) to prove the nursery is empty of roots, or **B** (RC) within the nursery. Epoch alone does not answer death.

---

## Why this is harder than a “normal” runtime

| Normal GC / RC runtime | Madaros native-v2 today |
|---|---|
| Stack maps / DWARF / compiler embeds root bitmaps | Counts only; `precise_stack_maps` forced false |
| Uniform reference representation | Handle id **or** raw ptr in same slots; threshold = capacity |
| Drop/ARC inserted by frontend or MIR | No handle drop lowering on the native path |
| Collect on pressure | Live path **must not** collect; only `exit 182` |
| Capacity is working-set sized | Capacity is **per-process lifetime budget** (monotonic bump) because free does not exist |

Plus: the table shares a **fixed 2 GiB** mmap with the heap (48 B × 2²² ≈ 192 MiB virtual for the table). Reclaim is not just correctness — wrong free also interacts with bump-heap layout.

---

## Cost summary

| Approach | Answers “died?” | Cost | Safe now? |
|---|---|---|---|
| Empty-frame reset | Assumes “no roots” without proof | Already written | **No** — #651 / PBPK |
| Free-list only | Does not | Small | **No** — never pops |
| Capacity bump (status quo) | Avoids question | Hours | Fail-closed at limit (182) |
| Unbox / fewer births (≤16 B structs, etc.) | Avoids question | Days+ | Yes — reduces pressure |
| **RC + drop insertion** | Yes, if every copy is counted | Large (MIR+codegen; **not only gc.sio**) | Only with complete copy protocol |
| **Precise maps + mark free-list** | Yes | Larger (regalloc+maps+gc) | Only after honesty gate flips |

**Is it cheap?** **No.**  
Cheap work does not reclaim; it **reduces births** or **fails closed**. Real reclaim is gated on death detection (A or B). That work necessarily touches **codegen / MIR / regalloc**, not only `gc.sio` — so even as reclaim owner I must **coordinate via the bus** before editing `codegen_x86_linux.sio` (other lanes have been colliding on nearby natives).

---

## Ownership / next steps (this lane)

1. **This doc** is the design authority for reclaim; claim held on `gc.sio` + future `stdlib/runtime/**`.
2. **No reclaim implementation in this commit** — would be either theatre (free-list never used) or #651-class (reset).
3. **Allowed next in `gc.sio` without stealing others’ files:** documentation constants comments, optional **read-only** helpers / counters for instrumentation *if* they do not change emit semantics; any emit-path change → bus to owners of `codegen_x86_linux.sio`.
4. **Recommended fleet sequence:**  
   - (other angles) birth-rate / unbox / PBPK loop shape;  
   - (this angle, when implementing) either **drop+RC** on a narrow MIR class with a witness gate, or **precise maps first** — never empty-frame reset.

### Acceptance for a future reclaim PR (preview)

- #651 scalar demonstrator stays **correct** under stress (no wrong multiples of `0xFFFFF`).
- Tight alloc loop past capacity either **reuses** freed slots or **182** with no silent wrong values.
- Existing `madaros_handle_table_182` reclaim/escape fixtures + gate used as oracle.
- `precise_stack_maps: true` only if honesty gate is updated in the **same** commit as real bitmaps.

---

## Bottom line

| Question | Answer |
|---|---|
| How do you know a handle died? | **Precise roots (A)** or **complete refcount/drop (B)**. Generations (C) without A/B are the failed reset. |
| Why harder here? | No root bitmaps; mixed handle/raw words; no drop lowering; reclaim was replaced by fail-closed 182 after reset corrupted live state. |
| Cheap? | **No.** Do not implement reclaim until death detection exists. |

*Implementation deferred on purpose.*
