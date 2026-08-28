<!-- docs:meta
topic_id: repo.docs.audit.modular-compiler-stack-clash-2026-05-29
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.modular-compiler-stack-clash-2026-05-29
-->

# Modular compiler end-to-end: root-cause analysis (2026-05-29 / 30)

> **⚠️ SUPERSEDED 2026-06-10** — the crash state described below no longer
> reproduces on the current modular binary. Re-measured in
> [`MODULAR_COMPILER_AUDIT_2026-06-10.md`](MODULAR_COMPILER_AUDIT_2026-06-10.md):
> `artifacts/self-hosted/souc-mc-check.elf` (built 2026-06-07) passes `--check`
> on `hello.sio`, `fn main() { 1 }` and `let x = 1` with rc=0 under an 8 MB
> stack. Evidence is binary-level (the fixing commit was not attributed).
> The root-cause history below remains valid as forensic record. For the
> current state of the full-IR `--native-compile` path (still broken), see
> `slurm-jobs/selfhost-lower-oom/DIAGNOSIS.md`.

## ⚠️ 2026-05-30 SESSION F — CORRECTION: crash NOT fixed. Patch SAFE but does not close the goal.

**RETRACTION of an error made earlier this same session:** I briefly wrote "TYPECHECK CRASH FIXED"
off ONE fluky `rc=0` run. Careful re-measure: `mc.elf --check tests/run-pass/hello.sio` SIGSEGVs
**30/30 (100%), ASLR on AND off (setarch -R 10/10)**. 0/30 print "preflight succeeded". The env.count
crash is **NOT fixed**; the modular compiler does NOT run E2E. (The lone rc=0/bytes=5044 was a
measurement artifact — 3 different modes all reported identical 5044 bytes, impossible if real.)

**What IS verified-real (the safe, genuine increment):**
- Bootstrap fixed point: stage2==stage3==**`bded845eec9e7e5ec7ccaf19f4f97bf1`** (patched self-hosts).
  Build: `/workspace/mc-build/bootstrap.sh`. **bin/souc takes POSITIONAL `<src> <out>`, NOT `-o`.**
- **503/503 run-pass IDENTICAL to baseline bin/souc** → the patch (tuple-slot cache recovery +
  nested-match local `match_ends` x86+a64 + prologue stack probe) has ZERO regressions. SAFE to commit.
- Patch's codegen fixes work IN ISOLATION (nested-match returning Big struct → 165 ✓).
- main.sio → 84 MB mc.elf, 0 errors.

**The crash (still the real blocker, unchanged class):** all driver modes that run the checker SIGSEGV
100% — `--check`, `--probe-frontend`, `--probe-load-ir`, `--native-compile`; default mode cleanly fails
"IR preflight failed" (rc 1). Signature: on hello.sio (no bitwise ops, no types) the checker emits the
ENTIRE error catalog (E048/E005/E012/E014/E013/E019/E017/E006…) all "at 64..71", then an infinite
`print_type_name` catalog spam (`Knowledge<?>Model<i8i32…fn#-1`, 4 MB) → stack-overflow SIGSEGV. Both
are downstream of a **garbage loop-counter / corrupted Checker-env** — the SAME bug class as documented
below, NOT eliminated by the landed patch. **Conclusion: the targeted-codegen-patch hypothesis for this
crash is FALSIFIED.** The credible remaining fix is the by-value→`*mut Checker` conversion of the check
phase (or at least the hot value-thread spine: check_expr/check_stmt/check_block/check_call_expr/
check_opt_expr/check_expr_depth), which eliminates the by-value Checker copies entirely.
Gotchas: Sounio stdout buffers→**0 bytes on crash**; gdb/strace ABSENT; /tmp FLAKY (use
/workspace/mc-build). NOT committed.

---

## ✅ 2026-05-30 — ROOT CAUSE SOLVED (superseded by SESSION F above)

The crash is a **single codegen bug in `lean_single.sio`: synthetic-tuple slot-count
encoding overflows for tuple elements ≥ 1000 slots.** It is NOT a stack clash and NOT a
by-value-copy corruption — those earlier theses chased the symptom.

**Mechanism (proven by ptrace `/tmp/segchain2`,`segwatch`,`segsrc` + capstone, ASLR off, and exact arithmetic):**
- Tuple types are encoded as a synthetic hash `tcount*1000000 + first_nslots*1000 + total_nslots`
  (`lean_single.sio:23496`, also 4641/9458). Each field assumes **< 1000 slots**.
- The check phase returns `(Checker, TypeEntry)` from ~413 by-value methods. `Checker` is
  **0x51ff = 20991 slots**, `TypeEntry` 34; total 21025. The packed low part is
  `20991*1000 + 21025 = 21012025`. `ret_agg_nslots` (`lean_single.sio:3170`) decodes
  `ty_hash % 1000` = **25**. So a 167 KB tuple return is copied as **25 qwords (200 bytes)**
  to the sret slot; `sret[200B..167KB]` is left uninitialised (stale stack = descending
  pointers spaced 0x1000 from a prior frame's `[Struct;N]` fill).
- The caller (`c = ann_pair.0`) then copies the full 0x51ff qwords, propagating the garbage.
  `env.count` (offset 0x400 = qword 128 > 25) lands in the garbage → `TypeEnv::lookup`
  (`env.sio:79`) reads `bindings[garbage]` → SIGSEGV `mov rax,[rcx+rdx*8]` at `lookup+0x99`.
- **Why only main.sio:** `lean_single.sio` contains **zero `Checker`**; its tuples are all
  < 1000 slots, so it never trips the overflow. `bin/souc` HAS the bug (it is lean_single's codegen).
- The `.0`/`.1` field-access offset (`lean_single.sio:13792`) and a latent **scalar-element
  type bug** (`elem_slots==1` path forced bool/enum → i64, breaking `tuple.1 && x` for
  `(BigStruct, bool)` like `gpu_fusion_append_ops`) are the same overflow surfacing elsewhere.

**Fix (landed in `lean_single.sio`, mirrors the existing `tuple_destructure_from_ptr_x86`
cache-override):** added `tup_first_slots_true` / `tup_total_slots_true` helpers that recover
the true slot counts from the element-type cache (`TUP_CACHE_*`, registered at every encode
site); used them in `ret_agg_nslots` and the `.0`/`.1` field-access path; raised the `.1`
offset cap 8192 → 2e9; made the scalar-element path take its type from the cache.

**Verification:** isolated repros `/tmp/big3.sio` (return-trunc), `/tmp/big4.sio`
(`.0`+`.1` aggregate offset), `/tmp/big5.sio` (`(BigStruct,bool)` `.1` in `&&`) all FAIL on
`bin/souc` and PASS on the fixed compiler; small-tuple regression `/tmp/reg.sio` unchanged.
Blocker #1 "stack clash" was a **misdiagnosis** of this same crash — the prologue page-probe is
not required once the truncation is fixed (confirm via the end-to-end run).

**Known follow-up (not blocking hello):** the overflowed hash is also the `TUP_CACHE` key
(16384-capped) → two distinct ≥1000-slot tuples can collide and get wrong element types.

---

## ⭐ STATUS (read first) — 1 blocker FIXED, 1 root UNRESOLVED  *(SUPERSEDED — see top)*
`self-hosted/compiler/main.sio` **type-checks (0 real errors) and emits an 84 MB ELF**, but that ELF
SIGSEGVs when run. NOT yet working end-to-end.

**Honest correction (do not over-claim):** an earlier draft asserted "one convergent root: by-value
Checker copying." That is NOT proven and is contradicted by the evidence below — every observed
Checker copy is correct-size, and a correct-size copy from a clean source cannot corrupt. The
copy-thesis is retracted as the stated root. Two candidate roots for blocker #2 were tested and BOTH
are insufficient (see #2). The originating WRITE is not yet caught.

Two blockers:
1. **Blocker #1 — stack clash:** each by-value `Checker` temp is a multi-MB stack frame; 76
   functions get 4–65 MB frames; lean_single emits no stack-probe, so the prologue `sub rsp,<huge>`
   skips the guard page → SIGSEGV. FIXED (verified) by an x86 prologue page-probe + large `ulimit -s`.
2. **Blocker #2 — env.count corruption (HARDWARE-WATCHPOINT EVIDENCE):** with the clash fixed,
   `--check` still SIGSEGVs in `TypeEnv::lookup` because `env.count` (offset 0x400=1024 in the env,
   which is Checker field 0) is read as garbage → garbage loop index → `bindings[garbage]`.
   - Staged asserts: `env.count` correct after init AND after collect (both `*mut`).
   - **Hardware watchpoint** (`/tmp/segtrace4`, DR0/DR7 via PTRACE_POKEUSER, ASLR off) on the HEAP
     `checker_addr+0x400`: it is written to **0 exactly twice and STAYS 0** through the crash —
     **no garbage is ever written to the heap env.count.** So the garbage lives **only in the
     by-value STACK COPIES** of the Checker made during the check phase — NOT a heap mutation.
   - `sizeof(Checker)` ≈ **164 KB** (copy uses `rep movsq` ecx=0x51ff=20991 qwords), NOT 7.6 MB;
     the 7.6 MB `check_expr` frame is ~46 by-value Checker copies stacked across its match arms.
   - The bridge copy at `checker_check_expr_mut` (`(*c).check_expr(e)`, check.sio:1126) is CORRECT
     (164 KB incl. count@1024 = 0). So the corruption is **inside `check_expr`'s own value-threading**
     of the by-value Checker (the 344 `self: Checker` methods), in a stack copy.
   - DISCRIMINATING TESTS RUN (both candidate roots INSUFFICIENT):
     - by-value-copy thesis: RETRACTED — all observed copies are correct-size (0x51ff / 0xc2); 4
       minimal repros pass. A correct copy from a clean source can't corrupt → garbage enters via a
       **WRITE/return**, not a copy. The copy at `check_expr+0x684` merely *propagates* garbage whose
       source (`depth_pair.0` from `check_expr_depth`'s `(Checker,bool)` return) is already dirty.
     - `(*ref).arr[i]=v` write bug: confirmed real but rebuilding `main.sio` with the full 128-site
       auto-deref rewrite (m6) gives the **identical** crash at 0x919648 (garbage RDX) → NOT the root.
   - SOURCE-LEVEL BISECT (distinct exit codes: panic=1, div0=136, natural SIGSEGV=139):
     - `c.expr_depth = c.expr_depth + 1` does NOT corrupt count (the INCR div0=136 never fired).
     - `self` is **already garbage at `check_expr`'s ENTRY** (entry assert → exit 1), BEFORE any
       copy in check_expr/check_expr_depth. → **CONFIRMED: garbage enters via an upstream WRITE, NOT
       a copy** (the advisor's thesis; the by-value-copy thesis is fully retracted). The write is in
       a by-value check function in the chain *before* check_expr's body executes
       (check_block/check_stmts/check_stmt, or a prior match-arm `c = c.check_xxx(e)`).
   - FURTHER BISECT (2026-05-30, distinct exit codes panic=1/div0=136/segv=139): localized along
     `check_stmt`(clean) → `check_expr_stmt` → `check_expr`#1(call, clean entry) → `check_call_expr`
     (clean entry) → `c.check_opt_expr(e.left)` (c clean right before) → `check_expr`#2 (callee
     `println` ident) → `env.lookup` → CRASH. check_block/check_stmt entries clean; `expr_depth++`
     does not corrupt; the multitest/hypothesis section is clean for `println`.
   - ⚠️ CORRECTED CHARACTERIZATION: the corruption is **layout-sensitive STALE/UNINITIALIZED stack
     data, NOT a deterministic write.** Proof: an identical `self.env.count > 100000` assert at
     check_expr entry FIRED in one build (m13) but did NOT fire in another (m17) that merely added a
     second assert — i.e. the count value at the same point DIFFERS between builds. Adding asserts
     perturbs the stack layout and moves/hides the garbage. So **source-level assert bisecting is
     UNRELIABLE here**, and the bug is a Checker **copy/return path that leaves `env.count`
     (offset 0x400) UNINITIALIZED** (stale stack), which only crashes when the stale value is
     out-of-range. This also explains why all minimal repros pass (stale stack happens to be benign).
   - REVISED FIX outlook: because it's a MISSING init (not a corrupt-from-clean copy), eliminating
     the by-value Checker copies (the `*mut` refactor) WOULD plausibly fix it — back on the table,
     but still large. A targeted codegen fix needs the specific copy/return that omits count.
   - FRAME-ZEROING TEST (2026-05-30): changed the prologue probe to ZERO each function's frame
     ([rsp,rbp) in 8-byte stores) — a defensive zero-init that would make uninitialized reads = 0
     (which is the CORRECT value for hello.sio, where env.count should be 0). Rebuilt the probe
     compiler + main.sio (m18). Result: STILL SIGSEGVs at the same lookup. → count is NOT merely
     left uninitialized; it is **actively WRITTEN by a copy with a wrong-source value** (the garbage
     is a heap binding-pointer, `0x7fffbac9…008`). Since the copy overwrites the zeroed slot with
     that value, zeroing can't help.
   - REFINED ROOT: a by-value copy in the value-thread reads `count` (dest offset 0x400 = pointer
     layout) from a SOURCE where a **binding-pointer sits at offset 0x400** — i.e. the source has
     `[TypeBinding;128]` laid out so a binding lands at 1024 (inline-ish / shifted), while the
     copy/dest use pointer-layout (count@1024). A **layout/offset disagreement** for the struct-array
     field across two copy participants. (All sites I disassembled — type_env_new, lookup, the bridge
     and check_call_expr copies — use pointer-layout 0xc2/0x51ff; the inline/shifted source is
     produced somewhere up the value-thread, not yet found.)
   - RELIABLE RUNTIME TRACE (2026-05-30, segtrace5b = int3 sw-breakpoint reading [rbp+off] / one deref,
     ASLR off, NO source change so layout-stable): walked the self pointer through the check phase:
     - check_fn_item / check_stmts / check_stmt: self.count = 0 (CLEAN).
     - check_expr#1: incoming self.count = GARBAGE (a heap binding-pointer 0x7fffbac8d008).
     - self POINTER VALUE changes: check_stmts self_ptr=0x7fffffb979d0 (count 0) → check_expr#1
       self_ptr=0x7fffff5a1ca0 (count garbage). So a COPY in between produced the garbage Checker.
     - The chain check_stmt→check_expr_stmt→check_opt_expr→check_expr makes THREE full-size copies
       (`rep movsq` ecx=0x51ff): check_stmt+0x44e, check_expr_stmt+0x057, check_opt_expr+0x096 — each
       `src=rsi=[rbp-0x10]` (its self_ptr), `dst=[rbp-0x29xxx]`. Disassembly shows each is a faithful
       full copy from its self_ptr; check_stmt's self_ptr points to a count=0 Checker.
   - ⚠️ STATIC/RUNTIME CONTRADICTION (limit reached): statically, a full `rep movsq` from a clean
     source (count=0 @0x400) MUST yield a clean dest, yet the runtime dest has a binding-pointer at
     0x400. Three chained clean full-copies cannot produce garbage by static reasoning — so the real
     mechanism is a runtime/codegen subtlety not visible in the disassembly (candidates: rcx/rsi
     clobbered at runtime, rep movsq miscount, the layout-sensitive stale value being read mid-copy,
     or a copy whose dest is later overwritten by an overlapping deeper frame). Resolving this needs
     an INTERACTIVE DEBUGGER (gdb/rr) to single-step the actual `rep movsq` and inspect rsi/rcx/the
     copied bytes — NOT available in this env (gdb/strace/valgrind absent). ptrace+capstone is exhausted.
   - PATHS TO FIX: (a) get gdb/rr onto the box and single-step the check_stmt+0x44e copy; (b) the
     `*mut Checker` refactor of the check phase (eliminates all these by-value copies — definite fix,
     large); (c) audit lean_single's by-value-large-struct ARGUMENT-passing codegen (the `rep movsq`
     emitter for struct args) for the case that triggers here.
   - (superseded) NEXT (layout-STABLE tooling only — asserts AND frame-zeroing perturb/мask): build segtrace5 =
     int3 software-breakpoint (read rbp) + DR0/DR7 hardware watchpoint, and WALK the copy chain up
     from the crash `self.env.count` (its source = `*[rbp-0x2af90]+0x400` in check_expr, etc.) one
     level per cycle until the ORIGIN copy that first writes a binding-pointer to a count slot from a
     clean ancestor; that copy's RIP + the two structs' layouts pinpoint the disagreement. OR audit
     `struct_like_nslots`/`arr_storage_slots`/`st_field_offset` + `fill_repeat_struct_array_slots_x86`
     for an inconsistent `[StructType;N]` sizing. Then the fix is a small codegen consistency fix.

### FIX PLAN
- **NEXT STEP (confirm mechanism FIRST):** set a hardware watchpoint on the transient stack-temp
  `self.env.count` inside `check_expr` (or single-step the value-thread) to catch the EXACT copy/
  write that first puts garbage there. Only then is the root proven. The minimal repros (Tests
  D/F/G/H) all pass, so the precise trigger is unknown — do not commit a fix before catching it.
- **DO NOT** start the 344-method `*mut` refactor — it is expensive, was justified by the now-retracted
  copy thesis, and may not fix anything. Off the table until a root is PROVEN by a passing end-to-end run.
- **Low-confidence candidate (only if the watchpoint proves by-value copies are inherent):** convert the 344
  by-value `self: Checker` check methods to `*mut Checker` in place, like the collect phase
  (`checker_collect_items_mut`, `checker_init_in_place`). Would eliminate by-value copies → would fix
  BOTH blockers and remove the giant frames. Only touches `self-hosted/check/*.sio`;
  `lean_single.sio` untouched → md5 stable. Effort: large/mechanical, multi-session. This is ONE
  candidate; if the watchpoint reveals a specific copy/write codegen bug, a small targeted codegen
  fix in lean_single may be cheaper and fix it for all programs.
- **Alt/secondary:** the prologue stack-probe (verified) is a correct general codegen fix in
  lean_single for the residual large-frame class (bootstrap-md5 flips; verify gen2==gen3 + suite).
- Separate confirmed bug (off the #2 path, reverted): `(*ref).arrayfield[i] = v` (explicit-deref
  array-field WRITE) is silently miscompiled; auto-deref `ref.arrayfield[i] = v` works. 128 sites
  in the modular tree. Fix as a scoped codegen handler for `(*p).field[i]=v` (lean_single uses 0
  such sites → md5-stable) OR rewrite sites to auto-deref.

---

# Appendix: investigation log (stack-clash first, then blocker #2)

## Status
- `self-hosted/compiler/main.sio` (modular entrypoint) **type-checks with 0 real errors**
  (only the known `parser/parser.sio:124` phantom pair) and **emits an 84 MB ELF** via
  `bin/souc` (`ac08e3b8…`, the current closures-arc fixed point).
- That ELF: `--version-json` ✓, `--check <file>` ✓ (frontend + typecheck only).
- That ELF: **every IR-lowering / native path SIGSEGVs** — default `compile()`,
  `--native-compile`, `--ir-dump`, `--probe-load-ir`, `--probe-frontend` (after typecheck).

## Root cause — stack overflow / stack-clash (NOT a logic bug)
ptrace evidence (no gdb/strace in env; built `/tmp/segtrace2.c`):
- Fault = `push rax` writing 8 bytes below RSP; RSP ~6 MB **below** the bottom of the
  mapped `[stack]` VMA → single-step descent skips the kernel's 1 MB `stack_guard_gap`.
- RBP-chain backtrace (identical for `--probe-load-ir` and `--ir-dump`) ends in two
  enormous adjacent frames:
  - **fn#7445 = 8,063,776 B (7.69 MB)** — caller
  - **fn#7484 = 6,056,352 B (5.78 MB)** — callee (the one whose prologue clashes)
  - nested = ~13.5 MB, exceeding the ~8–12 MB stack.

## Why it compiles but crashes
- `lean_single.sio:884 max_frame_bytes() = 4 MB`, checked at codegen epilogue
  (`frame_sz = (NEXT_SLOT*8+15)&~15`; sites ~24325 / 30509).
- But `tc_stack_frame_too_large` (`lean_single.sio:3257`) is only a **warning**, not an
  error: *"Not a hard error — large-struct value-semantics are valid but risky at runtime."*
  → the oversized frame is emitted anyway.
- The compile log has **76 functions with 4–65 MB frames** (32 > 8 MB; worst fn#3316 = 65 MB).
- lean_single emits **no stack-probe** in the prologue, so any frame larger than the
  guard gap clashes regardless of `ulimit -s` (verified: `ulimit -s unlimited` does not help).
- `bin/souc`/lean_single never *runs* these multimodule-driver frames, so it is unaffected.

## Why `--check` survives
`run_check_mode` → `preflight_multimodule_frontend` (frontend+typecheck, stops before IR).
The IR-load paths (`load_multimodule_ir*`, `compiler_preflight_ir_load`) nest fn#7445→fn#7484.

## Source of the giant frames
`local_bss_spill_bytes() = 512 KB` spills only *individual arrays ≥512 KB* to BSS.
By-value aggregates (`Program`, `IrModule`, large structs/arrays passed/returned by value)
and many sub-512 KB locals accumulate into multi-MB frames that are never spilled.

## Candidate fixes (to decide)
- A. Codegen: emit stack-probe touch-loop in prologue for frames > guard gap (+raise runtime
     stack). General; fixes all 76. Flips bootstrap md5.
- B. Codegen: spill more aggressively to BSS (lower threshold / spill by-value temporaries).
     Risk: BSS spill is non-reentrant — breaks recursion.
- C. Source: reduce frames on the compile path (move by-value locals to module-level BSS / Box).
     Bootstrap-safe; laborious; must cover the on-path functions.
- D. Make `tc_stack_frame_too_large` a hard error to surface offenders, then fix them.

## FIX PROGRESS (2026-05-29 session)

### Blocker #1 — stack clash: FIXED (verified) via prologue page-probe
- Added an x86 prologue page-probe (`mov rax,rbp; .L: sub rax,0x1000; cmp rax,rsp; jbe .E;
  or byte[rax],0; jmp .L`) right after `sub rsp,<frame>` at lean_single.sio:24144.
  19 bytes; no-op for frames <4 KB; touches each page top-down so the kernel grows the
  stack incrementally (no guard-skip). Build with a large `ulimit -s` (tested 1 GB).
- VERIFIED: probe-built compiler emits correct code (hello.sio compiles+runs); probe moves
  the `--probe-load-ir` crash from `collect_fn_def` (clash) onward to the typecheck phase.
- Probe is INNOCENT of the second crash: non-probe build crashes at the same logical spot.
- NOTE: only the regular-fn prologue (24144) needs it; the closure prologue (9186) does NOT
  (giant frames are all regular fns). Throwaway probe compiler: `/tmp/lean_probe2.elf`;
  probe-built modular ELF: `/tmp/m4.elf`.

### Blocker #2 — UPDATE: root narrowed to a struct-layout inconsistency in the full Checker
Disassembly (capstone) of the probe-built modular ELF:
- `TypeEnv::lookup` reads `self.count` at **offset 0x400 (1024)** = `mov rax,[self+0x400]`, then
  `self.bindings[i]` as `[self + i*8]` (8-byte/pointer stride). i.e. lookup assumes
  `bindings: [TypeBinding;128]` is a **128-pointer array** (`fill_repeat_struct_array_slots_x86`
  heap-allocates struct-array elements → pointer arrays). `type_env_new` builds `bindings` the
  same way (8-byte slots). So lookup vs type_env_new AGREE.
- Yet `[self+0x400]` holds garbage → confirmed by a `count` guard at lookup top turning the
  SIGSEGV (139) into a controlled exit (1).
- `env` is the FIRST Checker field (offset 0); `Checker` is `heap_alloc(8388608)` (8 MB) then
  `checker_init_in_place` writes all ~52 fields via `(*raw).field = …`. Bumping the alloc to
  64 MB did NOT help → not an allocation overflow.
- Therefore the most likely root: **`sizeof(TypeEnv)` (the `env` field) is mis-computed in the
  Checker layout**, so the NEXT field write (`(*raw).borrows = …`, offset = sizeof(env)) lands at
  an offset < 1032 and OVERWRITES `env.count` at byte 1024. Equivalent class: any struct-array
  field sized inconsistently between `st_field_offset`/`struct_like_nslots` (layout) and
  `fill_repeat_struct_array_slots_x86` (storage). Manifests only with the full 52-field Checker —
  EVERY minimal repro passes (Tests D/F/G/H: large env field, value-thread copy, `*mut` raw-ptr
  large-field write, `(*c).env.lookup()` by-value pass — all correct).
- DISASSEMBLY VERDICT (capstone on m4.elf, all THREE static components AGREE — pointer-sizing,
  TypeEnv = 0xc2 = 194 qwords = 1552 bytes, count at offset 0x400 = 1024):
  - `checker_init_in_place`: `env` written via `rep movsq` 0xc2 to c+0 (so c+1024 = count = 0);
    next field `borrows` at c+0x610 = 1552 — NO overlap; all 52 field offsets monotonic & correct.
  - caller `checker_lower_named_type_mut`: `(*c).env` extracted via `rep movsq` 0xc2 (1552 B) to a
    stack temp before the by-value `lookup` call — same size.
  - `lookup`: reads `self.count` at `[self+0x400]`, `bindings[i]` as `[self+i*8]` — same layout.
  => `env.count` IS 0 right after init; it becomes garbage **at runtime between init and the
     lookup**. So blocker #2 is NOT a static layout bug — it is a RUNTIME corruption of the
     env.count slot (mutation-path write to a wrong offset, or heap-aliasing of the heap-allocated
     `bindings` pointers) that only manifests with the full Checker exercised by real typecheck.
- PRECISE NEXT STEP (needs a memory watchpoint — gdb/lldb absent here; install one or add a
  software watch): watch the 8 bytes at `checker_ptr + 0x400` (env.count) from just after
  `checker_init_in_place` through the first `lookup`; the writing instruction is the culprit.
  Prime suspects: the env-mutation write-back path (`(*c).env = c.env.bind(...)` / push_scope) and
  any `_mut` field write whose offset is computed wrong only in the 52-field struct. Tooling
  ready: capstone disasm + FNMAP map + segtrace/2/3 ptrace harnesses.

### Blocker #2 — typecheck garbage-pointer crash: OPEN (genuine bug, not probe-induced)
- After the clash is fixed, `--check`/`--probe-load-ir` crash in **`TypeEnv::lookup`**
  (`self-hosted/check/env.sio:79`), faulting instruction `mov rax,[rcx+rdx*8]; mov rax,[rax]`
  with a GARBAGE base — i.e. `self` (a large `TypeEnv`, a field of the huge `Checker`)
  arrives as a bad pointer.
- rbp-chain backtrace: `dep_normalize_step` (recursive) → `dep_add_subst` →
  `dep_check_single_constraint` (7.6 MB frame) → `TypeEnv::lookup` → fault.
- Present in BOTH probe and non-probe builds (probe did not move this crash) → NOT the clash.
- si_addr varies run-to-run (uninitialized/garbage pointer); currently deterministically faults.
- NOT reproducible in isolation: minimal repros of (a) nested array-of-struct ctor
  (`[empty_trait_method();16]`), (b) `Big.env.lookup()` large-struct field method call — both
  PASS. So it is a context-specific interaction, likely large-struct value-semantics in the
  modular checker, or a bad/uninitialized env passed on the dependent-type path.
- Offset→name mapping: code starts at file off 0x1000; crash FN_OFF 0x918648 = `lookup`.

## Tooling (reusable)
- `/tmp/segtrace.c` (regs+code@RIP), `/tmp/segtrace2.c` (RBP-chain + frame sizes) — ptrace, no gdb needed.
- `/tmp/lean_probe2_src.sio` — lean_single + 24144 probe; `/tmp/lean_dump_src.sio` — +FN_OFF/name dump.
- FN_OFF→name map: run dump compiler, parse `FNMAP@<off>` / `=<name>` pairs (print_int splits lines).
- Oversized-frame list: parse compile log for "stack frame too large" (warning prints fn# + bytes).
- Crash-fn mapping: `(RIP-0x400000) - 0x1000` = FN_OFF; bisect into FNMAP.
