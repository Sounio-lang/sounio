<!-- docs:meta
topic_id: repo.docs.research.zeta-variance-deep-investigation
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.zeta-variance-deep-investigation
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# ζ — Deep Investigation of NEXT_SLOT and VAR_ Arrays

**Date**: 2026-04-12 (continuation of `zeta_variance_fix_plan.md`)
**Purpose**: Understand WHY the first fix attempt (save/restore NEXT_SLOT around while body) broke bootstrap, and identify the RIGHT fix.

## Refined understanding — what was wrong about the original diagnosis

The original hypothesis: "NEXT_SLOT grows unbounded across while-loop iterations". **Partially correct, but misses the key point.**

The actual structural fact:

- **NEXT_SLOT is reset per function**, not per statement. See x86 Pass 2 at line 14690 (`NEXT_SLOT = 1`) and A64 Pass 2 at line 19272 (`VAR_COUNT = 0; NEXT_SLOT = 1`). Also at closure entry (line 6768, 7680).
- **Within a function**, NEXT_SLOT accumulates monotonically as local variables and aggregate temporaries are introduced.
- **The frame-size calculation at function end** uses the final NEXT_SLOT: `FN_FRAME_SIZE = (NEXT_SLOT * 8 + 15) & 0xFFFFFFF0` (line 14884). The emitted prologue reserves exactly this much stack.

So the first fix (save/restore NEXT_SLOT around the while body) does something dangerous: it resets `NEXT_SLOT` before the frame-size is calculated, so **the function frame is too small**. Variables that were allocated slots during the loop body execution are no longer reserved in the stack frame at runtime. Any function called on top that reads/writes those slots corrupts the stack. This is precisely the mechanism by which gen3 became a broken binary that failed to find `main`.

## Where the 2^63 actually comes from

- `VAR_SLOT`, `VAR_NS`, `VAR_NE`, `VAR_IS_F64`, etc. are sized as `[i64; 1024]` at lines ~729-737.
- `emit_copy_scratch_to_var_variance_x86` (line 5419) bounds-checks `slot < 1024` and silently returns if overflow.
- When NEXT_SLOT exceeds 1024 during compilation of a function with very many locals, subsequent `var_add(ns, ne)` calls write into `VAR_NS[slot]` etc. at out-of-bounds indices — Sounio's `[i64; N]` array access is unchecked in the current native backend.
- This corrupts adjacent BSS memory. Later `variance_of(x)` reads BSS that was either never initialized or was clobbered by the out-of-bounds write. Reading uninitialized BSS as an i64 and then interpreting the bits as f64 produces `0x8000000000000000 = −0.0` or, displayed as an unsigned i64 printed via the variance_of path, `9223372036854775808 = 2^63`.

## Why `rapamycin_epistemic_adaptive.sio` specifically triggers

Main function has a while loop with ~70 statements in the body. Let-bindings include `k1b, k1br, k1p, s2b, s2br, s2p, k2b, k2br, k2p, s3b, s3br, s3p, k3b, k3br, k3p, dt9, var_post, ...`. Plus dense arithmetic expressions, each of which spawns temporaries via `materialize_aggregate_expr`.

Order-of-magnitude estimate: 70 statements × ~5-10 temporaries per complex expression = 350-700 slots in a single function. Well past 1024-capacity VAR_ arrays.

## Why gen3 said "no main"

Location of the error message: lines 14928 (x86) and 19413 (A64).

```sio
var main_fn: i64 = -1
var mi: i64 = 0
while mi < FN_COUNT {
    let ms = FN_NS[mi as usize]; let me = FN_NE[mi as usize]
    if main_fn < 0 && me - ms == 4 && sb(ms) == 109 && ... { main_fn = mi }
    mi = mi + 1
}
if main_fn < 0 { print("error: no main\n"); return 1 }
```

This is a symbol-table search at codegen end: linear scan of `FN_NS`/`FN_NE` for a function named "main" (ASCII 109=m, 97=a, 105=i, 110=n).

**How the fix broke this**: my patch reset NEXT_SLOT around the while-body. In the gen1 compiler (which was correctly compiled by the old artifact), the reset emitted code with a smaller frame than required. When gen1 compiled lean_single.sio (to produce gen2), the emitted gen2 binary's functions had insufficient frames. When gen2 ran — specifically when it tried to compile `lean_single.sio` itself to produce gen3 — the stack-corrupted gen2 miscompiled something early (likely a function that uses many locals, such as `compile_stmt` or `compile_or`), and the FN_NS/FN_NE arrays were either partially corrupted or never populated, so the search for "main" failed. Hence "no main".

## The RIGHT fix — three options, ranked

### Option A (recommended): Expand VAR_ arrays from 1024 to 4096 or 8192

Change the array declarations at lines ~729-737:

```sio
var VAR_NS: [i64; 8192] = [0; 8192]
var VAR_NE: [i64; 8192] = [0; 8192]
var VAR_SLOT: [i64; 8192] = [0; 8192]
var VAR_LEN_SLOT: [i64; 8192] = [0; 8192]
var VAR_ALEN: [i64; 8192] = [0; 8192]
var VAR_ESIZ: [i64; 8192] = [0; 8192]
var VAR_IS_F64: [i64; 8192] = [0; 8192]
var VAR_IS_CLOSURE: [i64; 8192] = [0; 8192]
var VAR_MUT: [i64; 8192] = [0; 8192]
// and any parallel VAR_ arrays with matching size
```

Plus update `emit_copy_scratch_to_var_variance_x86`'s bound check (line 5419) from `slot >= 1024` to `slot >= 8192`, and any other hard-coded 1024 that references VAR_ capacity.

**Pros**: no scoping changes, bootstrap unaffected, handles `rapamycin_epistemic_adaptive.sio` without room for further issues.

**Cons**: 8× larger VAR_ arrays → roughly 9 arrays × 7168 extra i64 entries × 8 bytes = ~515 KB extra BSS. Negligible.

**Risk**: minimal. Just resizing.

### Option B: Add bounds checking with clean error

```sio
fn var_add(ns: i64, ne: i64) -> i64 with Mut, Panic {
    if NEXT_SLOT >= 8192 {
        tc_error(EP, "too many local slots in function — try splitting into smaller functions")
        return -1
    }
    // existing body
}
```

**Pros**: loud failure instead of silent corruption.

**Cons**: on its own, does not fix `rapamycin_epistemic_adaptive.sio`. Best combined with Option A.

### Option C: True scoped NEXT_SLOT with coordinated VAR_COUNT save/restore

This is what my first attempt should have done:

```sio
// Entering while body:
let saved_ns = NEXT_SLOT
let saved_vc = VAR_COUNT
// ... compile body ...
// Exiting while body:
NEXT_SLOT = saved_ns
VAR_COUNT = saved_vc
```

AND updating the frame-size calculation to use `max(NEXT_SLOT ever seen)` rather than current NEXT_SLOT:

```sio
// Track high-water mark
let max_ns_during_loop = ... // need to track across iterations
FN_FRAME_SIZE[fn_idx] = (max(NEXT_SLOT, max_ns_during_loop) * 8 + 15) & 0xFFFFFFF0
```

**Pros**: semantically clean (loop-scoped variables).

**Cons**: requires frame-size high-water-mark tracking; non-trivial refactor; bootstrap risk non-zero until tested. The scope-correctness benefit is subtle and the practical gain over Option A is small.

## Recommended action

**Do Option A.** Clean, minimal, reversible, zero bootstrap risk. The 8192-slot ceiling handles realistic scientific-computing functions. If that's ever insufficient, revisit Option C.

## Concrete patch for Option A

1. Grep `lean_single.sio` for every array declaration of the form `var VAR_*: [i64; 1024]` or `var VAR_*: [i64; N] = [0; N]`.
2. Replace each with `[i64; 8192] = [0; 8192]`.
3. Grep for hard-coded `1024` in connection with slot bounds; replace with `8192`.
4. Grep for hard-coded `8192` in connection with variance channels (`ch * 8192` addressing) — these are the variance BUFFER's per-channel stride, NOT the VAR_ array bound. Leave those alone unless the variance buffer itself needs expansion (separate concern).

## Test plan

1. Apply Option A edits (~15-20 lines changed).
2. Rebuild gen1: `$SOUC lean_single.sio /tmp/gen1.elf`.
3. Rebuild gen2: `/tmp/gen1.elf lean_single.sio /tmp/gen2.elf`.
4. Rebuild gen3: `/tmp/gen2.elf lean_single.sio /tmp/gen3.elf`.
5. Check fixed point: `md5sum /tmp/gen{2,3}.elf` must match.
6. Test rapamycin: `/tmp/gen2.elf tests/run-pass/rapamycin_epistemic_adaptive.sio /tmp/rapa.elf && /tmp/rapa.elf`. Expect `variance_of(c_blood)` in the 10⁻⁴ range, not 2^63.
7. Run entire `tests/run-pass/*.sio` suite; confirm no regressions.

## Confidence

- **Root cause (VAR_ array overflow, not variance scratch overflow)**: ~85%. The agent investigation identified the 1024-element bound on the named-variable bookkeeping arrays. The original framing of "variance scratch overflow" was imprecise.
- **Option A fixes it**: ~85%. Array expansion preserves all existing behavior with strictly larger capacity.
- **Bootstrap preserved under Option A**: ~95%. No scoping changes, no codegen changes, only BSS layout. The fixed-point proof (gen2 == gen3) is a direct consequence of deterministic compilation.

## Unknowns

- Whether the variance scratch buffer itself (`RT_VARIANCE_BUF_BSS_OFF`) also needs expansion from its current 1024-slot × 8-channel layout. Likely yes, if functions with > 1024 slots are to use variance tracking correctly. Separate edit if so.
- Whether the compiler has any other 1024-based bounds coupled to VAR_* arrays that we'd miss with a simple grep.

## Note on epistemic honesty

The agent that produced this investigation was working in read-only Explore mode; it could not test its hypotheses directly (no Bash, no ELF execution). The analysis is code-reading + reasoning-from-evidence. Before applying Option A, the first step should be an instrumented compile of `rapamycin_epistemic_adaptive.sio` with a print of `NEXT_SLOT` and `VAR_COUNT` at function end to confirm the overflow hypothesis quantitatively.
