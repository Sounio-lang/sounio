# Self-host lower/codegen wall — diagnosis (2026-06-08)

Branch `claude/kw-demote-module` @ `fa37e61c5`. All heavy runs on SLURM `cpu-ops`
(bounded `--mem`), never the workspace pod (see the 2026-06-08 OOM-crash incident).
Harness: `submit.sh` + `fetch.sh` (this dir). Job 2356 result:
`results/lower-oom-20260608T223642/`.

## What the gen-N self-host attempt actually does (job 2356, bounded 16 GB)

`gen-N --native-compile self-hosted/compiler/main.sio` (gen-N = `bin/souc` mini_native
compiling main.sio, then that binary self-compiling):

```
compact modular IR path → "Merged IR: 64 functions" → imported_simple_ir_missing_main
  → "falling back to full IR path"
  → RUN_RC=139 (SIGSEGV), PEAK_HWM=7.9 GB, FNINSTR_LINES=0
```

### CONFIRMED (empirical)
1. **It is a SIGSEGV (139), NOT an OOM (137).** Peak RSS 7.9 GB — far under the 16 GB
   cap and under available memory. The earlier "OOM/137 crashed k8s" was a *pod-level*
   symptom (memory pressure + k8s liveness probe under an unbounded direct run), not
   gen-N's actual failure mode.
2. **It dies before completing the first body function.** No `lower_ordered: fn_begin`
   line appears (`SOUNIO_LOWER_ORDERED_TRACE=1` was set; the run.log tail ends at "falling
   back to full IR path" with nothing after) and `FNINSTR_LINES=0`. Body-function lowering
   never started — death is in the front-half: parse → resolve → typecheck →
   epistemic-convert → **preseed**, or inside the very first body function.
3. **The compact path is a confirmed dead-end** (64-entry pattern table; `main` isn't a
   "simple" fn → `missing_main`; unfixable by bumping the 64 cap).

### NOT the cause (revised — my pre-run hypotheses were about phases never reached)
- **Not** the `flush_current_func` 96 MB/function churn, **not** the `emit` 68 KB/instr
  churn. Those are in body-lowering, which never runs. They are real and would bite
  later, but they are downstream of this segfault.
- **Not** OOM. **Not** the fixed-array caps overflowing cleanly (`find_or_add_fn_id`
  *is* guarded: returns -1 past `IR_MAX_FUNCS`).

## Mechanism (high confidence — source + a bug the code documents itself)

`IrModule` is **~96 MB by value**: `functions: [IrFunction; 1400]`, each `IrFunction`
~68 KB via inline `instrs: [IrInstr; 256]` (`ir/ir.sio:654,702,1922`). The gen-N native
backend has a **documented large-boxed-struct miscompile, `c634b38f`**: auto-dereferencing
`self.module.<field>` (instead of explicit `(*self.module).<field>`) **materializes the
entire module by value** and, per the in-code comment at `lower.sio:3325`, *"materializes
the whole [IrFunction;1400] module and faults idx=1400."* The worker stack is only
**8 MB** (`ulimit -s`), so any stack materialization of a 96 MB temp is instantly fatal;
larger temps spill to BSS/heap (the ~7.9 GB churn). The lowerer has **dozens** of such
auto-deref read sites (`lower.sio:1897–2092+`); only some are hand-patched with explicit
`(*box)` single-derefs. **Hypothesis (the pinpoint job settles it):** main.sio's module has
**1418 functions > `IR_MAX_FUNCS=1400`**, reaching the exact cap boundary the comment warns of
— but `find_or_add_fn_id` *is* guarded (returns -1, no fault), so 1418>1400 triggers the crash
only if a *materialized* by-value copy is indexed OOB at idx 1400. Stated as hypothesis, not
established.

## Pinpoint refined (jobs 2357, 2358 + read_env test) — crash is at the FULL-IR ENTRY

Two more bounded jobs added env-gated (`SOUNIO_STAGE_TRACE=1`) markers across the whole
front-half — a **function-entry canary** at line 1 of `load_multimodule_ir_traced`, then
`main_parsed` / `deps_done` / `resolve_begin/done` / `m0_typecheck_begin/done` /
`preseed_begin/done` + a preseed per-fn counter. Both jobs: **identical RUN_RC=139, 7.9 GB,
and NOT ONE marker fired** — last output still "falling back to full IR path".

Ruled out the obvious confounders:
- **`read_env` works in gen.elf** — re-running the persisted binary with
  `SOUNIO_TARGET_OVERRIDE=x86_64-linux` flips the dispatch line `source=fallback` →
  `source=env_override`. So the env-gating is fine.
- **`print` is effectively unbuffered** — the early lines (Madares/dispatch/compact/"falling
  back") appear reliably and in order, well under any 4 KB buffer; a self-hosted SIGSEGV would
  lose buffered output, so an unreached canary means **genuinely unreached**.
- **Not a stack-frame overflow** — `Program`≈176 B, so `[Program;64]` + two `[string;64]`
  locals ≈ ~15 KB frame, nowhere near the 8 MB stack.

⟹ **The SIGSEGV fires at the very ENTRY of the full-IR path** — in `load_multimodule_ir`
(wrapper) / the prologue of `load_multimodule_ir_traced`, **before its first statement runs.**
And **~7.9 GB is already resident at that point**, accumulated by the preceding **compact
path** — which scans all source to build a mere **64-entry** table (itself pathological:
7.9 GB for 64 entries). The compact path is a dead-end we want to remove anyway, and its huge
residual heap may be implicated in the entry fault.

### Still open (needs ONE rebuild to settle)
- Wrapper vs prologue vs trace-construction (use *unconditional* prints, not env-gated, in
  `load_multimodule_ir` and at `_traced` entry — removes any last doubt).
- Whether disabling the compact path (go straight to full-IR from low memory) lets the
  full-IR entry survive — a small source edit; also the right long-term move (compact is a
  dead-end). This is the highest-value next experiment.

## Fix direction — proportionate first, refactor as fallback

This codebase's *own established remedy* for `c634b38f` is a **local explicit-`(*box)`
single-deref patch** at the offending site (already applied at `lower.sio:207/347/607/3325…`).
So the proportionate sequence is:

1. **Pinpoint** the first faulting site (the stage-marker job below).
2. **Patch it locally** the same way — replace the auto-deref `self.module.<field>` with the
   explicit `(*self.module).<field>` pattern (or the bound-local `var m = lo.module; (*m)…`
   pattern used in `find_or_add_fn_id`). Re-run.
3. This likely just advances to the **next wall** — probably the body-lowering churn already
   characterized (`flush_current_func` 96 MB/fn, `emit` 68 KB/instr). Repeat: measure, patch.

**Fallback (only if patching becomes whack-a-mole across the dozens of sites at 1897–2092):**
the strategic fix is that **`IrModule` is too large to be a value type** — heap-indirect /
variable-length the big arrays (`functions` out of `[;1400]`, `instrs` out of `[;256]` →
pool/handle storage) so the structs are small handles, removing the whole hazard class at once
(fault + churn + the 1400 cap). Budget this only if the local patches don't converge.

Note: a naive `IR_MAX_FUNCS` bump to fit 1418 is **not** a fix — it enlarges the by-value
struct (+68 KB/fn), worsening the materialization.

## UPDATE — job 2362 (2026-06-09): compact-disable + unconditional entry prints

The decisive 1-rebuild experiment. Two changes in `/tmp/kw-demote`:
1. **Compact modular IR path DISABLED** in `module_native_driver.sio:1137` (the `else`
   branch no longer calls `load_multimodule_imported_simple_ir_global`; the full-IR path is
   entered directly). Verified independent: the full-IR recursive lowerer re-derives
   everything from `main_path` and never reads the compact `imported_simple_ir_global_*`
   accessors (only the compact ELF writer does).
2. **Three unconditional prints** (not env-gated): `FULLIR-A` (driver call-site before
   `load_multimodule_ir`), `FULLIR-B` (`load_multimodule_ir` wrapper body),
   `FULLIR-C` (`load_multimodule_ir_traced` body, first statement).

Result (`results/lower-oom-20260609T010823/`): **RUN_RC=139 (SIGSEGV), PEAK_HWM 5.6 GB,
FNINSTR=0.** All three prints fired (A, B, C). Run.log ends right after `FULLIR-C`.

### What this settles (two prior conclusions overturned, mechanism confirmed)
- **"Crash at entry, before `_traced`'s first statement" (jobs 2357/2358) is OVERTURNED.**
  That was an **env-gating artifact** — proven as an **in-run control**, not inference: in
  job 2362 the env-gated `STAGE` markers were *still empty* while the unconditional `FULLIR-C`
  at the same code location *did* fire. Same run, same location, only the env-gating differs ⟹
  `SOUNIO_STAGE_TRACE` is broken and `_traced`'s body **does** execute. The fault is in **real
  work**, not entry/prologue.
- **The compact residue is NOT the trigger.** Disabling compact dropped peak 7.9 GB → 5.6 GB
  (the ~2.3 GB delta = the removed compact residue) yet it **still SIGSEGVs**. Memory is not
  the cause (139 not 137; 5.6 GB ≪ 16 GB cap). Confirms the documented `c634b38f` by-value
  materialization — independent of resident heap.
- **Fault LOCATION localized to `module_frontend_lower_imported_source_recursive`**
  (`module_frontend.sio:2625`), the statement immediately after `FULLIR-C`
  (`_traced:3673`: `var merged_imported = ...rec(...)`), and before any "Merged IR:" prints.
  FNINSTR=0 ⟹ still pre-body-lowering. Confirmed by-value `IrModule` plumbing in this fn:
  2634 `module_frontend_lower_source_file_summary(...) -> IrModule` (by value),
  2642 recursive return by value, 2644 `ir_merge_modules(dst: IrModule, src: IrModule) -> IrModule`
  (two by value in, one out). All three confirmed full-`IrModule`-by-value via signature grep.

- **MECHANISM is NOT yet pinned — and the earlier "96 MB stack temp = instant SIGSEGV"
  framing is RETRACTED (advisor-corrected, contradicted by this run's own RSS).** Peak 5.6 GB
  is ~2.5 GB *above* the 3.05 GB bss — a substantial lowering working set accumulated before
  death. A single fatal 96 MB stack temp on recursion call #1 would die at ~3.15 GB, not after
  +2.5 GB; and per this harness's own notes big temps **spill to heap, not the 8 MB stack**.
  ⟹ the likelier proximate trigger is the **documented OOB-index-at-1400**: main.sio alone =
  **1418 fns > `IR_MAX_FUNCS=1400`** (`ir/ir.sio:11`), so `ir_merge_modules` (2644) writes the
  merged `functions` array past the `[IrFunction;1400]` bound. Stack-materialization vs
  OOB-1400 is **unresolved** — the next job settles it.

### Next steps (concrete)
- **Discriminating pinpoint job (cheap, 1 rebuild):** at each recursion, print **depth/iteration
  AND the running merged `fn_count`** (not just fn-entry/after-2634/per-iter markers). This settles
  both (a) first-call-on-main vs deep recursion, and (b) whether the crash coincides with `fn_count`
  crossing 1400. That single measurement tells you whether the fix is **materialization-side**
  (`&!out`/heap-indirect) or **cap-side** (variable-length `functions` array).
- **Fix direction depends on which mechanism the pinpoint job confirms — do NOT pre-commit:**
  - If **OOB-1400**: the `&!out` SRET pattern does **NOT** fix it (a `[IrFunction;1400]` array still
    can't hold 1418). A naive `IR_MAX_FUNCS` raise may now be a valid **cheap interim** — the old
    "bump = worse fault" reasoning assumed stack materialization, which this run refutes (memory is
    not the constraint: 5.6/16 GB).
  - If **materialization**: the in-place `&!out` SRET pattern (already used by
    `load_multimodule_ir_traced_into`) is the proportionate fix.
  - **Durable fix covers BOTH:** heap-indirect / variable-length `IrModule` (pool the `functions`/
    `instrs` arrays → small handle) removes the fault + the body-lowering churn + the 1400-cap at once.
- The teed-up `load_multimodule_ir` → `_into` switch at `module_native_driver.sio:1157` fixes only
  the wrapper's by-value RETURN, not the recursive lowerer's internal sites — lower priority.
- **Operator note (separately committable):** disabling the compact path is a legitimate standalone
  cleanup (confirmed dead-end) independent of the by-value/OOB fix.

## UPDATE 4 — job 2365 (2026-06-09): FIX APPLIED + VERIFIED — recursion wall CLEARED, advanced to the next wall

Fix (clean, in `/tmp/kw-demote`; diagnostic instrumentation reverted):
- **`module_frontend_lower_imported_source_recursive` rewritten from recursion → iterative BFS +
  visited-set (dedup)**, mirroring the proven `module_frontend_import_typecheck_main` traversal:
  Phase 1 BFS-collects the distinct transitively-imported module paths (256-bounded, `visited_contains`),
  Phase 2 lowers each distinct module once and merges sequentially. Signature kept → both callers
  (`load_multimodule_ir_traced` 3680, `_traced_into` 3913) inherit the fix; the self-recursion is gone.
  Correctness: merge order is irrelevant because `ir_module_resolve_named_calls` resolves cross-module
  calls **by name** (verified, `module_frontend.sio:969` `ir_name_eq`), not by position; dedup leaves
  exactly one definition per name (more deterministic than the old duplicate-laden module).
- **Companion cleanup:** disabled the dead-end compact modular-IR path in `module_native_driver.sio`
  so the full-IR path is entered directly (separately committable).

Result (`results/lower-oom-20260609T021101/`, default 8 MB stack — the real test):
```
module_native_driver: compact modular IR table path disabled; using full IR path
Merged IR: 1400 functions          ← REACHED (prior runs crashed in SECONDS at depth 10)
```
- ✅ **Recursion wall CLEARED — the traversal/merge code ran to completion and returned.** "Merged
  IR: 1400" only prints *after* the rewritten function returns, under the original 8 MB stack, where
  every prior run died in seconds. The depth-~10 stack overflow (job 2363) and the re-visit OOM
  (job 2364) are **both gone**.
- ⚠️ **BUT on self-host the merge did NOTHING — `1400` is main.sio TRUNCATED, not a real merge.**
  Phase 2 lowers `file_paths[0]`=main first; `summary(main)` caps at 1400 (main alone = 1418 fns);
  so `merged.fn_count`=1400, and every subsequent `ir_merge_modules(merged, part)` hits its own
  `merged.fn_count < IR_MAX_FUNCS` guard → `1400 < 1400` is false → copies **zero** functions. The
  merged module is main's first 1400 fns; **no imported module contributed anything.** So this run
  exercised the traversal (no crash) but NOT the cross-module merge.
- ⟹ **`IR_MAX_FUNCS=1400` is the BINDING next constraint, not an orthogonal aside.** The post-merge
  grind (~21 min, peak 13.4 GB, under the 16 GB cap — did NOT OOM; killed by the 25-min TIMEOUT, not
  a crash) ran on a module missing ~5200 of ~6642 fns. `ir_module_resolve_named_calls` cannot resolve
  cross-module calls to functions that aren't present, so **this path cannot yield a working compiler
  no matter the walltime.** Lifting the cap (or heap-indirect `IrModule` that dissolves it) is the
  precondition for anything downstream to matter — and is also what finally lets the merge do real work.

### Verification status & honest caveats
- Proven: the **stack/OOM recursion crash is fixed** and the traversal runs to completion. NOT proven:
  the whole compiler self-hosts (no gen2==gen3 gate reached; post-merge timed out producing no binary).
- **Cross-module merge correctness is UNVERIFIED.** The order-independence-via-by-name-resolution
  property was validated by *reading* `ir_module_resolve_named_calls`, but never stress-tested at
  runtime — because the cap meant nothing cross-module survived to be merged.
- **No-regression for NORMAL programs is UNVERIFIED and is the real precondition for commit.** This
  change is on the path for *every* multi-module native compile. For normal programs (main < 1400 fns)
  the BFS *does* merge deps — a bug there would regress real compiles, and the self-host run (where the
  cap masks the merge) would not have caught it. Risk is low (mirrors the proven typecheck BFS), but
  the multi-module no-regression gate — NOT the self-host grind — is what must pass before landing.
- **NOT committed/pushed** (awaiting request). Edits isolated: iterative-lowering fix
  (`module_frontend.sio`) vs compact-disable cleanup (`module_native_driver.sio`).

### Next (corrected ordering)
1. **Lift `IR_MAX_FUNCS` (or heap-indirect `IrModule`)** — the binding blocker; until then the merge
   is inert on self-host and the downstream wall is on an unusable, incomplete module. Do NOT spend a
   longer-walltime re-run characterizing slow-vs-OOM on a main-only module — low value.
2. Once the cap is lifted and the merge does real work: stress-test cross-module merge correctness, and
   add the **name→fn_id index** to `ir_module_resolve_named_calls` (O(n²·m) → O(n·m)) if it's the sink.
3. Before any commit: the **multi-module no-regression gate** on normal programs.

## UPDATE 3 — job 2364 (2026-06-09): `ulimit -s unlimited` discriminator — STACK OVERFLOW **CONFIRMED**, + a second OOM problem revealed

Re-ran the persisted harness with `ulimit -s unlimited` before gen.elf (confirmed achieved:
`env.txt` → `ulimit_s_before_run=unlimited`). **The crash MOVED, decisively:**

| | 8 MB stack (job 2363) | unlimited stack (job 2364) |
|---|---|---|
| RC | **139 (SIGSEGV)** | **137 (OOM / OUT_OF_MEMORY)** |
| peak | 5.6 GB | **15 GB** (hit the 16 GB cap) |
| max recursion depth | 9 (died entering 10) | **17** (the depth-16 guard line) |
| nodes traced | 70 (partial) | 355+ (partial, OOM'd) |

- **⟹ STACK OVERFLOW CONFIRMED.** Raising the stack limit pushed the crash from depth ~10 to
  depth 17 — exactly the discriminator's "moves deeper → stack overflow" outcome. **bss-spill
  aliasing is REFUTED** (it would have stayed at ~10 regardless of stack). My pre-test local
  frame-budget analysis *favored* aliasing and was **wrong** — the empirical test overrides it;
  the per-frame stack cost is genuinely large (~0.8 MB/frame: the IrModule-handling locals/SRET
  scratch in `module_frontend_lower_imported_source_recursive`), even though the static slot
  count didn't pinpoint which temp. Lesson: ran the test instead of trusting the static estimate.
- **⟹ SECOND face of the SAME bug, unmasked once the stack stops biting: it OOMs.** With the depth
  wall gone the recursion reaches the depth-16 guard, visiting ~355+ nodes and climbing to 15 GB.
  This is **not a second independent bug** — it's the same unbounded import-lowering recursion seen
  without the stack mask: one subsystem to rework.
  - ⚠️ **"exponential" is NOT established — the data shows re-visiting, magnitude unconfirmed.** The
    nodes/level histogram is **flat** (7, 11, 13, 19, 24, 28, 26, 28, 26, …), not geometric
    (2,4,8,16,…). Flat ~28/level over 17 levels ≈ re-traversal of a *smaller* distinct-module set by
    ~5–17×, meaningful but not astronomical — and the trace is **OOM-truncated**, so the tail is
    unknown. The recursion has no visited-set/dedup (line 2642 recurses every import path), which is
    *consistent with* re-visiting, but this run printed depth+`fn_count`, **not paths**, so
    re-visitation is inferred, not measured.

### Fix — direction (not a firm ranking; the OOM magnitude is unconfirmed)
- **Make the import-lowering traversal iterative** (explicit worklist instead of deep recursion) —
  this removes the confirmed stack-overflow directly.
- The traversal **appears to re-visit modules**, so a **visited-set (dedup)** should help and is the
  natural companion — but whether dedup *alone* suffices (vs is even necessary) is not yet measured.
- **Heap-indirect / variable-length `IrModule`** (small handle) is a complementary option: it shrinks
  frames AND cuts per-node memory, so with only ~355 nodes it **might also bring the OOM under the
  cap on its own** — plausible, not established. It also covers the downstream body-lowering churn +
  the 1400 cap. **Cap raise: useless** (`fn_count`=4).
- **Optional cheap check for whoever implements the fix** (one-liner, no new job needed mid-diagnosis):
  print the module path per `REC-ENTER` and count distinct-vs-total — substantiates "dedup helps" and
  separates re-traversal from genuinely-many-modules, pinning whether dedup is sufficient.

## UPDATE 2 — job 2363 (2026-06-09): discriminating pinpoint — DEPTH-driven crash at recursion depth ~10 (NOT size/cap; stack-overflow vs bss-aliasing resolved by UPDATE 3 = stack overflow)

Instrumented `module_frontend_lower_imported_source_recursive` (`module_frontend.sio:2625`) with
unconditional `PIN` prints: entry depth, summary-return `fn_count`, and pre/post-merge running
`fn_count` at every recursion node. Result (`results/lower-oom-20260609T013115/`): **RUN_RC=139,
peak 5.6 GB, FNINSTR=0** — identical crash, now fully resolved:

- **Crash at recursion DEPTH ~10, NOT at any function-count boundary.** Max depth reached = 9;
  `REC-ENTER d=10` never prints (count 0) — the SIGSEGV fires the **first time** the DFS tries to
  descend from depth 9 → 10. The DFS depth trace climbs `…7 8 9 8 8 9` and dies entering 10.
- **`fn_count` is TINY at the crash (4–15), nowhere near 1400.** The modules at the crash depth are
  4-fn and 15-fn. ⟹ **NOT the OOB-1400 cap-side hypothesis, and NOT large-single-module
  materialization.** Both prior guesses are refuted by direct measurement.
- **RSS is flat (~5.6 GB, ~3 GB of it the fixed bss); it does not climb with depth** ⟹ the 96 MB
  `IrModule` value-locals (`var merged`, `let dep`) are bss-spilled (fixed region), not heap-grown.
- **Mechanism = depth-driven, within the recursion — but stack-overflow vs bss-spill-aliasing is
  NOT yet separated (one cheap `ulimit -s` job settles it).** What's proven: the crash scales with
  recursion DEPTH, not module size. The two candidate sub-mechanisms:
  - **bss-spill ALIASING corruption (favored by local frame-budget analysis).** Backend constants
    (`lean_single.sio`): `local_bss_spill_bytes()=512 KB`, locals ≥512 KB spill to a **fixed
    per-site BSS address** (`bss_alloc_aligned`, compile-time-assigned). The 96 MB `IrModule` locals
    `var merged`/`let dep` spill; a **recursive** function reuses the **same fixed addresses at
    every depth** → child clobbers parent's IrModule → a corrupted count/pointer eventually faults.
    Explains the prologue fault (by depth ~10 the aliased fixed slot holds a bad SRET pointer, so
    the depth-10 call faults setting up its return destination *before* `REC-ENTER` prints) and the
    sharp deterministic threshold. **Per-frame STACK is only ~KB** (the `imports` local =
    `ImportPathList`=`[string;256]`+count ≈ 2 KB since `string`=1 slot/8 B via `arr_storage_slots`;
    everything ≥512 KB is bss-spilled, not stacked) — so a few-KB frame would need ~800 frames to
    exhaust 8 MB, **not 10**, which is why pure stack-overflow looks unlikely.
  - **stack overflow (favored by the sharp frame-count threshold + prologue fault).** Uniform frames
    dying at a fixed frame-count is the textbook signature — but it REQUIRES ~0.8 MB of *stack* per
    frame that the budget above cannot name. Tension unresolved by static analysis.
  - **Discriminator (decisive, cheap):** re-run with a larger `ulimit -s`. Crash moves deeper →
    stack overflow. Crash stays at depth ~10 → aliasing/data corruption.
  - The **companion** `module_frontend_count_imported_source_recursive` (2602) — same recursion +
    depth-16 guard but **i64-only locals (no IrModule, no bss-spill)** — survives. Consistent with
    BOTH sub-mechanisms (small frames AND no aliased big-locals), so it doesn't break the tie.
- The recursion also has **no visited-set / dedup** (line 2642 recurses on every import path) → the
  same modules are re-lowered along every path (exponential re-work); the depth-driven overflow just
  bites first. Pathological, but secondary to the overflow.

### Fix — re-scoped by this result
- **Cap raise (`IR_MAX_FUNCS`): USELESS here** — `fn_count` is 4, not 1400. (It may still matter for
  the later body-lowering wall, but it is not this crash.)
- **`&!out` SRET accumulator: its sufficiency DEPENDS on the sub-mechanism.** Under **aliasing**,
  passing `merged` by pointer (caller-owned buffer, distinct per active call) removes the aliased
  fixed-slot reuse → likely a **full fix** regardless of frame size. Under **stack overflow**, it only
  shrinks the frame and may just push the crash deeper. So the ulimit discriminator also decides the
  `&!out` ranking.
- **Robust fixes that work under EITHER sub-mechanism (recommend now):** (a) **convert the recursion to
  an iterative worklist** (explicit heap stack/queue) — sequential iteration never aliases fixed slots
  AND removes call-stack depth; add a **visited-set** to also kill the exponential re-lowering; (b)
  **heap-indirect / variable-length `IrModule`** (small handle) → tiny, distinct frames → no aliasing,
  no depth overflow, and it also covers the downstream body-lowering churn + the 1400 cap. Both are
  safe to pursue before the ulimit job; the iterative-worklist + dedup rewrite is the most direct match.

### Open / next
- Print the module **path** at each `REC-ENTER` (one more trivial rebuild, optional) to confirm
  whether the depth-10 chain is a legitimately deep acyclic DAG or a re-traversed cycle — informs
  whether dedup alone shortens the chain or only the longest-acyclic-path/iteration fix does.
- The stack-overflow conclusion is depth-reproducible-by-construction but came from a single PIN run;
  a no-PIN re-run isn't needed (job 2362 already crashed identically). The discriminator is solid.

## UPDATE 5 — independent re-verification (2026-06-09, read-only, no jobs)

Re-confirmed UPDATE 4's conclusion against the source in the `feat/typological-expansion`
workspace — **no builds, no SLURM, no edits** (the heavy self-host build is SLURM-gated per the
SUPREME DIRECTIVE, and the UPDATE-4 iterative-BFS merge fix lives in `/tmp/kw-demote`, NOT this
tree):
- **Resolver is O(n²·m) as stated.** `ir_module_resolve_named_calls` (`module_frontend.sio:929`)
  has the inner `ji` linear name-scan (`938–944`, `ir_name_eq`) nested inside the `fi`×`ii` loops →
  a full fn-table scan per unresolved `IrCall`. A `name→fn_id` index would make it O(n·m).
- **Cap confirmed binding.** `IR_MAX_FUNCS = 1400` (`ir/ir.sio:11`); `IrModule.functions: [IrFunction;1400]`
  (`ir.sio:1898`) inline, `IrFunction.instrs: [IrInstr;128]` (`ir.sio:681`) inline ⟹ `IrModule` is a
  large value type. Whole program ≈ 6642 fns ≫ 1400, so the merge stays inert until the cap is lifted.

⟹ **Conclusion stands: the binding next blocker is `IR_MAX_FUNCS=1400`.** Lifting it (or heap-indirect
`IrModule`) + the *conditional* resolver index ("if it's still the sink", measurable only after the cap
is lifted on a real merged module) is the identified next workstream. It is **SLURM-gated**, depends on
the un-landed `/tmp/kw-demote` merge fix, and **awaits operator request**. Note (per UPDATE 4): the
"cheap interim" cap-bump does NOT extrapolate for free — `[IrFunction;6642]` ≈ ~450 MB by value and
`ir_merge_modules` churns two-in/one-out per call (peak was already 13.4 GB on the *truncated* module),
so even the interim is a SLURM experiment, not a safe local edit. **Nothing further to run here.**

## UPDATE 6 — job 2366 (2026-06-09): cap lift 1400->8192 + resolver index — BUILT, but OOM + ambiguous no-regression

Implemented in `/tmp/kw-demote` (committed): `c8b1843b0` = the UPDATE-4 iterative-BFS merge
fix + compact-disable (instrumentation stripped); `fdeedf9c1` = IR_MAX_FUNCS 1400->8192 across
all 5 coupled fixed arrays (ir.sio constant/field/init + normalize.sio ×2) + an
open-addressing name->fn_id index in `ir_module_resolve_named_calls` (16384 slots, load <=0.5,
lowest-index first-match-wins preserved). Harness: `submit-capfix.sh` (build + `--native-compile`
no-regression on v1..v7 + self-host experiment). Results `results/capfix-20260609T084618/`.

- ✅ **BUILD_RC=0** — gen.elf (89 MB, bss=3.0 GB) built from the edited source under gen N-1.
  The resolver index + cap=8192 PARSE + TYPECHECK cleanly; the old souc-mc-check.elf "+1 parse
  error" was a false alarm (that elf predates current `&!`/deref syntax). The build.log's
  match-exhaustive / E001 lines are pre-existing non-fatal warnings (BUILD_RC still 0).
- 🔴 **Self-host OOM (137): peak 16.7 GB > 16 GB cap.** Died immediately after "compact path
  disabled; using full IR path", BEFORE any "Merged IR:" line. With the merge no longer
  truncating at 1400 and IrModule ~5.7x larger at cap 8192, the full-IR merge blows the 16 GB
  ceiling. ⟹ **the naive cap bump is NOT sufficient on its own — this is the predicted evidence
  that heap-indirect / variable-length `IrModule` is the mandatory durable fix** (advisor framed
  an OOM here as a RESULT, not a failed run). Memory, not the cap value, is now the wall.
- ⚠️ **No-regression 0/7 — DISCRIMINATED on the worker (binaries preserved under
  `/tmp/capfix-...-2366/`, light toy compiles, no SLURM build): the 0/7 is NOT my regression.**
  - **My BFS merge fix WORKS.** A minimal FLAT 2-module program (`use lib::{add5}` + sibling
    `lib.sio`) compiled with gen.elf → `Merged IR: 2 functions`. The v-witnesses' `Merged IR: 1`
    is a SUBDIR-import path-resolution issue (`use helper::util::*` → `helper/util.sio`),
    orthogonal to the cap/resolver change.
  - **The real reason for 0/7: the full-IR `--native-compile` path writes NO ELF even on
    success.** Both the flat (`Merged IR: 2`) and v-witness compiles printed `Native compilation
    successful: output=out.elf`, returned rc=0, and wrote NEITHER out.elf NOR a.out. The writer is
    `compile_native_x86_linux_to_file` → `compile_native_finalize_and_write_ref`
    (`codegen_x86_linux.sio:9107/9126`) — code my two commits NEVER touch (they touch only
    ir.sio/normalize.sio/module_frontend.sio/module_native_driver.sio). So this is a PRE-EXISTING
    full-IR-path codegen/writer gap (consistent with UPDATE 4 job 2365: "post-merge timed out
    producing no binary"). The historically-working multimodule native paths were the COMPACT path
    and `--native-v2-compile` (source-concat bridge) — both bypassed here. ⟹ my no-regression
    INPUTS/PATH were invalid for measuring my change; 0/7 says nothing about my code's correctness.
  - The attempted `bin/souc` baseline was VOID: `bin/souc` is **mini_native** (usage
    `mini_native <src> <out>`), has no `--native-compile` flag at all. The only
    `--native-compile`-capable binary is the gen.elf built from main.sio.

- 🔴 **Self-host OOM (137): peak 16.7 GB.** Corrected framing (advisor): it died INSIDE
  `load_multimodule_ir`'s BFS lower+merge, BEFORE "Merged IR:" and far before body-lowering — so
  this does NOT prove body-lowering needs heap-indirection. It implicates **by-value `IrModule`
  churn**, now 5.7× larger at cap 8192: `ir_merge_modules` does `var merged = dst` and
  `lower_source_file_summary` returns `IrModule` by value, ~once per module across ~6642 fns. ⟹
  the **evidence-indicated next step is the deferred in-place `&!` merge** (caller-owned
  accumulator, no per-merge full copy), NOT (yet) full heap-indirect `IrModule`. Heap-indirect is
  the heavier fallback that may still be needed for the body-lowering wall this run never reached.
  Do NOT record cap + in-place-merge as proven-insufficient — the experiment didn't isolate it.

⟹ **Status: NOT landable; cap+index are a NECESSARY but not sufficient step.** Two distinct
remaining walls now isolated: (1) **by-value merge-churn OOM** → in-place `&!` merge (cheap, next);
(2) **full-IR `--native-compile` writes no ELF** → a pre-existing codegen/writer gap, orthogonal to
the cap/merge work and a prerequisite for ANY working self-host via this path. Commits
`c8b1843b0`/`fdeedf9c1` stay local (build-clean, merge-correct on flat imports, but the path can't
yet emit a binary).

## UPDATE 7 — jobs 2367/2368 (2026-06-09): baseline-confirmed — multi-module --native-compile emit is PRE-EXISTING broken; cap+resolver cause ZERO regression

Two follow-up jobs settled the attribution UPDATE 6 left open.

- **Job 2367** (compact path RE-ENABLED — reverted the harmful disable, commit `1f5282655`):
  no-regression STILL 0/8, including the new guaranteed-simple flat 2-module witness. The flat
  compile log shows BOTH emit paths fail: compact → `Native compilation failed:
  imported_simple_ir_emit_failed` (rc=1) → falls through → full-IR → `successful: output=out.elf`
  but writes NO file. So re-enabling compact did not help; the compact-disable was NOT the cause.
- **Job 2368** (BASELINE `fa37e61c5`, pre-ALL-my-commits, no-regression only): **identical 0/8,
  byte-for-byte same failure mode** (`imported_simple_ir_emit_failed` → full-IR false-success,
  no file). BUILD_RC=0.

⟹ **DEFINITIVE: `gen.elf --native-compile` has NEVER emitted an ELF for multi-module programs —
both the compact writer (`native_driver_write_imported_simple_ir_elf` → `imported_simple_ir_emit_failed`)
and the full-IR finalizer (`compile_native_finalize_and_write_ref`, returns 0 but writes nothing)
are PRE-EXISTING broken, in codegen code 0-lines-diff from baseline.** The cap lift (1400->8192),
resolver index, and iterative BFS merge cause **zero regression** (baseline reproduces 0/8 exactly)
and are build-clean + merge-correct (flat import -> `Merged IR: 2`). The historically-working
multi-module native paths are `--native-v2-compile` (source-concat bridge) and `bin/souc`
mini_native single-module — NOT this full-IR `--native-compile` path.

### The self-host wall is deeper than the cap — at least THREE orthogonal blockers
1. **Multi-module ELF EMISSION is broken on both `--native-compile` paths** (pre-existing,
   baseline-proven). This is a prerequisite for ANY working self-host via `--native-compile`, and
   it is independent of the cap. Largest unknown; untouched by this work.
2. **Self-host main.sio merge OOMs at cap 8192** (16.7 GB, in `load_multimodule_ir` BFS lower+merge,
   by-value `IrModule` churn) — addressable by the deferred in-place `&!` merge, but MOOT until (1).
3. The cap itself (1400) — NECESSARY (else the merge silently truncates) but, as now shown, far
   from sufficient.

### Landing status
- Commits in `/tmp/kw-demote` (local, unpushed): `c8b1843b0` (BFS merge) + `fdeedf9c1`
  (cap+resolver) + `1f5282655` (compact re-enable = back to baseline behavior). All build-clean,
  no-regression-confirmed (baseline reproduces the same 0/8). They implement the goal's stated
  action (lift cap + index resolver) correctly and safely, but do NOT unblock self-host on their
  own — blockers (1) and (2) remain. NOT a landable "self-host works" claim; a safe, correct
  prerequisite step with the next walls now precisely isolated.

## UPDATE 8 — root-cause of the multi-module --native-compile emit bug (2026-06-09, worker toy compiles, no build)

Localized the pre-existing emit failure via a differential on the preserved job-2367 gen.elf:
- ✅ `--native-v2-compile` flat 2-module → **out.elf EXIT=42** (works; source-concat
  `bridge_combine_multimodule_source`, does NOT call `load_multimodule_ir`).
- ❌ `--native-compile` SINGLE-module (`fn main(){42}`) → **SIGSEGV, 3.3 GB core dump**
  (≈ the bss; a materialization fault) on the streaming path.
- ❌ `--native-compile` MULTI-module → "Merged IR: 2" then `Native compilation successful` (rc=0)
  but writes NO file — the writer (`compile_native_x86_linux_to_file` → `..._finalize_and_write_ref`,
  which prints "Native binary size"/"Written to") is NEVER reached, contradicting the source ⟹
  gen.elf is MISCOMPILED at this site.
- ❌ `--emit-obj` MULTI-module → "Merged IR: 2" then nothing (no success, no panic — crashed).

**Common factor of the broken paths: they call `load_multimodule_ir(main_path)` and consume the
returned `MultiModuleIrResult` BY VALUE.** That struct embeds an `IrModule` by value (~48 MB at
cap 1400, ~280 MB at 8192). **The bug is the documented c634b38f SRET hazard** — module_frontend.sio:3988
spells it out verbatim: the by-value return "triggers the frame-sensitive SRET epilogue `rep movsq`
/ dropped-sret-pointer crash (c634b38f)". PRE-EXISTING (baseline cap-1400 also 0/8); orthogonal to
my cap change (which only enlarges the already-faulting struct).

**The remedy already exists but is UNWIRED dead code:** `load_multimodule_ir_into` /
`load_multimodule_ir_traced_into` (module_frontend.sio:3999/4253) write the result through an
`&!out` pointer via in-place field writes (no by-value return). BUT: (a) NO caller uses them —
`compile_multimodule_native_advanced` (driver:1147) and `run_emit_obj_mode` still call the by-value
`load_multimodule_ir`; (b) `compiler_preflight_ir_load_into` named in the comment does not exist;
(c) every `MultiModuleIrResult` constructor (`empty_multimodule_ir_result`→`multimodule_ir_error`→
`ir_empty_module`) ALSO returns the ~280 MB struct by value, and uninitialized struct locals are an
error (E001) — so wiring `_into` needs a **bss-global backing store** for the out buffer (per the
3988 comment's "BSS-global MultiModuleIrResult backing store"), not a stack local.

⟹ Two fix routes: **(A)** finish wiring the existing `_into` variant via a module-level bss-global
`MultiModuleIrResult` (localized; must verify the global's one-time init doesn't itself hit the SRET);
**(B)** the structural **heap-indirect `IrModule`** (small handle) that dissolves SRET +
materialization + the 1400 cap + the merge-churn OOM at once (heavier, but the durable end of this
whole bug family). Approach decision pending.

## UPDATE 9 — jobs 2369/2370: the emit bug is a WRITER-INTERNAL MISCOMPILE, not the by-value load

Wired the SRET-safe `load_multimodule_ir_into` into the driver via a bss-global
`MultiModuleIrResult` (commit `ec55931ff`). Flat 2-module STILL 0/8, byte-identical. An
instrumented build (job 2370, unconditional `DBG-*` markers) localized it decisively:

```
DBG-DRIVER-INTO ok=1 fn_count=2        <- _into fix WORKS: global populated correctly
DBG-DRIVER-CALLWRITER
DBG-WRITER-ENTER fn_count=2             <- writer entered with correct module
DBG-FINALIZE-ENTER                     <- compile_native_finalize_and_write_ref entered
Native compilation successful          <- returns 0, but NO "Native binary size"/"Written to", NO file
```

⟹ **The by-value `MultiModuleIrResult` return was NOT the active cause** (the writer was reached
fine even before; my `_into` fix addressed a documented-but-not-firing hazard). **The real bug is a
MISCOMPILE inside `compile_native_finalize_and_write_ref`** (`codegen_x86_linux.sio:8956`): it is
straight-line code — enter → build ELF bytes → `let elf_size=NATIVE_ELF_OFF` → `if elf_size<=0 return 1`
→ `if write_file(...)<0 return 1` → **unconditional** `print("Native binary size…")` ×3 → `0`. Job 2370
proves it ENTERS (DBG-FINALIZE-ENTER) and RETURNS 0, yet none of the empty-elf / write-fail / success
prints fire and no file is written. In correct codegen, reaching `return 0` is impossible without
executing the unconditional prints right above it ⟹ **gen.elf miscompiles this function's body/tail**
(the ~130-line ELF-byte-construction region between entry and the write). This writer
(`compile_native_x86_linux_to_file` → `..._finalize_and_write_ref`) is used ONLY by the full-IR
multi-module `--native-compile` path — never exercised end-to-end before, never confirmed working —
so the miscompile was latent. Same FAMILY as the broader self-host miscompiles (the compiler
mis-lowering itself), localized to this function.

**Status:** the `_into` fix is a legit latent-SRET-hazard hardening (no regression) but does NOT fix
the emit bug; the active cause is a codegen miscompile of the writer, an open-ended backend-debugging
task (each bisection iteration = one SLURM build). Diagnostic `DBG-*` prints are in the `/tmp/kw-demote`
worktree (uncommitted, in module_native_driver.sio + codegen_x86_linux.sio).

## UPDATE 10 — jobs 2372/2373: EMIT BUG FIXED (writer miscompile); next layer = full-IR codegen SIGBUS

Bisected the writer miscompile and fixed it.
- Job 2372 (markers bracketing finalize): `DBG-M-HDRDONE off=232` fired, `DBG-M-COPYDONE` did
  NOT — execution skipped the section-copy loops + the entire tail (write_file + prints) yet
  returned 0. Localized to the copy region inside the oversized
  `compile_native_finalize_and_write_ref`.
- Refuted alternatives on bin/souc directly (fast, no SLURM): an 800-statement function compiles
  fully (no instr cap), and `(*nc).code.bytes[ci]` in isolation works (construct is fine). ⟹
  frame/register-pressure MISCOMPILE specific to the large function, in bin/souc (gen N-1).
- **FIX (commit `d731cc3ce`): extract the 3 section-copy loops into a small standalone helper
  `nc_finalize_copy_sections`.** Job 2373 VERIFIED: the writer now runs and emits a well-formed
  4706-byte ELF (`Native binary size: 4706 / Written to out.elf`) — was: no file. **The EMIT bug
  is fixed.**

⟹ **Next layer exposed: the emitted ELF SIGBUSes (exit 135, signal 7) — PRE-EXISTING, baseline-confirmed.**
**Attribution — RIGOROUS (job 2374): `fa37e61c5` + ONLY the emit fix (cap=1400, NO resolver, NO
iterative merge) → IDENTICAL exit 135, same 4706-byte ELF.** ⟹ the SIGBUS is genuinely PRE-EXISTING;
my cap/resolver/merge/`_into` are INNOCENT (`NativeCompiler` has no `IR_MAX_FUNCS`-sized arrays, so
the cap bump doesn't grow its frames; and the no-call `main(){42}` faults, excluding the resolver).

### Entry-offset investigation (jobs 2375/2376) — entry is CORRECT; it was a red herring
Drilled the SIGBUS at the user's request. Instrumented the codegen layout (writer dump):
`pretramp=70, off0=0, off1=35, off2=-1` ⟹ 2 functions (main@0, second@35), 70 bytes total, entry
trampoline at offset **70 = 0x46**. So `entry_offset = 0x46` correctly points at the `_start`
trampoline. **My earlier "entry 2 bytes before _start" was a hand-disassembly MISCOUNT (main is 35
bytes, not 37) — RETRACTED.** The real fault is the **unpatched call** (`e8 00000000`, rel32=0 — a
call relocation never applied by the full-IR path) and, pervasively, **bin/souc large-struct
miscompiles all over this codegen path**: the SAME finalize-fragility class struck the
instrumentation itself — a 13-statement debug block re-truncated `finalize`, and in the small writer
the `nc.entry_offset`/`nc.code.len` field reads (large-local-struct materialization) silently
dropped from the print while `nc.fn_offsets[i]` printed. ⟹ the full-IR `--native-compile` codegen is
**pervasively miscompiled by the gen N-1 bootstrap** (unpatched relocations + large-`NativeCompiler`
field-access faults), not a single contained slip. This is a LARGE workstream (hardening every
large-struct access in the codegen against the c634b38f class, or fixing the bootstrap's codegen),
NOT a one-line entry fix. The production multi-module path is `--native-v2-compile` (which works).

### Net status of the cap/emit work (all local, unpushed, branch `claude/kw-demote-module`)
- `c8b1843b0` iterative BFS merge; `fdeedf9c1` cap 8192 + resolver index; `1f5282655` compact
  re-enable; `a6a0dc8dc` `_into` (latent SRET hardening, NOT the emit fix); `d731cc3ce` finalize
  copy-loop extraction (THE emit fix). All build-clean.
- Multi-module `--native-compile`: was "no ELF, false success" → now "emits a well-formed ELF that
  SIGBUSes". Real progress (writer fixed); end-to-end runnable binary still blocked on the full-IR
  codegen-correctness bug above (+ the self-host merge OOM for main.sio specifically).

## Reproduce / continue
- `bash slurm-jobs/selfhost-lower-oom/submit.sh` (builds gen-N + bounded instrumented run)
- `bash slurm-jobs/selfhost-lower-oom/fetch.sh <RUN_ID>` (pulls results from worker pod)
- Instrumentation in the worktree (`/tmp/kw-demote`): lower.sio FNINSTR print +
  the FULLIR-A/B/C prints + the compact-disable edit (job 2362). All diagnostic-only;
  revert/replace with the real `&!out` fix before anything lands.
