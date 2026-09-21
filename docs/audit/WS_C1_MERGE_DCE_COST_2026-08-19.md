<!-- docs:meta
topic_id: repo.docs.audit.ws-c1-merge-dce-cost-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.ws-c1-merge-dce-cost-2026-08-19
-->

# WS-C1 — cost of running reachability-DCE on the merge path

**Status:** measured. Not implemented this turn — and the prune is
already on the merge path.
**Claims-forbidden:** “Madaros is fixed-point-verified.” It is not.
`CLAUDE.md` forbids the phrase. The seed `lean_single.sio` has a
byte-identical gen2==gen3 pair. Madaros does not. The ladder that
would measure that pair is `scripts/ci/madaros_fixed_point_gate.sh`;
its recorded rung is still `check`.

## Semantic declaration

Three different numbers are not one number.

| name | what it counts | today |
|---|---|---|
| declared | `fn` lines in `main.sio`'s import closure | **10940** across **120** nodes |
| marked | source-level name hashes reachable from `main`, plus (after #1935) every callee of every impl/trait method body | **7044** on the shipped ELF |
| slotted | what lowering will actually emit: surviving `ItemFn` + every method of every kept `ItemImpl` + BSS globals (`fn_def: None`) | **1892 + 7221 = 9113** |

The 5997 in the gate header is a name-based reachability census of
top-level declarations. It is not the slot census. Impl methods are
never deleted (`spec_dce_filter_items` keeps every non-`ItemFn`) and
the dep-body pass lowers every method of a kept impl. BSS globals
are `ItemFn` with `fn_def: None` and are also kept. 5997 would fit
under 8191. 9113 does not.

“Put the prune on the merge path” is not a new function. It is a
question about a pass that has been wired since `146f5b039f`
(2026-08-05): `spec_dce_mark_across_programs` then
`spec_dce_filter_with_global_marks` at both lower-array read sites.
`spec_dce_unreachable_item_fns` remains the specialized-collapse
entry. The ordinary multi-module path — the path `main.sio` takes —
uses the global-marks pair.

The gate header is stale on that point. The code is not.

## What was measured this turn

Shipped ELF `bin/madaros-linux-x86_64` at `3d1f143e7a` (2026-08-17
07:09Z), SHA-256 `437bdd8f96a205906d53ca50a2a29ccf5f03a71c2e98e020b54d01351a0bff44`.
That blob is **not** an ancestor of the 16384 raise (`6a55b3bc60`,
2026-08-18) and **not** an ancestor of #1935 (`6f23dfe1da`,
2026-08-19). It is the compiler the fleet actually runs.

Source under test: `origin/main` `6f23dfe1da` (post-#1935), staged
at `/orangefs/training/ws-c1-merge-dce-20260819T0308Z`, host
`cpuops-t560-proxmox`, 2026-08-19T03:10:28Z.

```
imported_compile: loaded 120 modules
imported_compile: typecheck ok
lower_array: dce marks=7044
IR slot census: globals 1892 + functions 7221 = 9113 (max 8191, over by 922)
IR lowering failed during merge: too many lowered items:
    combined globals and functions exceed shared IR module capacity (max 8191 slots)
```

`madaros check self-hosted/compiler/main.sio` → rc=0, `verdict=0`,
zero `error[E…]`. Rung `check` holds. Rung `gen2` does not. No ELF.

Declaration census, same tree, `SOUNIO_IR_CAPACITY_PROBE_REPORT_ONLY=1`:

```
closure_nodes    120
fn_declarations  10940
ir_max_funcs     16384     ← source
headroom         5444
```

A static walk of the same 120 nodes: 9798 column-0 `fn` lines, 1142
indented `fn` lines (impl-method shaped), 69 `impl` headers.

Source `IR_MAX_FUNCS` is 16384. The shipped ELF still refuses at
8191. The raise exists; it has not been compiled into the compiler
that is asked to compile itself.

## The prune is already running. This is what it cost.

`146f5b039f` claimed “the merge capacity wall is gone.” The wall
moved. DCE dropped ~3700 dead top-level functions (10940 declared →
7221 slotted). The survivors plus 1892 BSS globals still overflow
the cap baked into the shipped ELF.

Cost that was paid to put it there, and that any second wiring
would pay again or double:

1. **Marks live in globals.** Filtering by
   `(*programs)[k].items = filtered` is a nested field-in-array
   store through `&!`. lean_single drops that store. The marks are
   computed once; the caller filters at its two *read* sites into
   an ordinary local.
2. **The preflight counts the filtered list.** Counting parsed
   items while lowering filters would refuse a program that fits.
   `madaros_imported_capacity_gate.sh` had to grow reachable
   padding for the same reason: dead locals no longer occupy slots.
3. **Refusal, not silent delete.** `SPEC_DCE_MAX = 8192` (50% of
   16384 slots). If the mark set saturates or the fixed-point
   exceeds 64 rounds, `spec_dce_mark_across_programs` returns -1
   and both filters hand their input back unchanged. A saturated
   table used to delete every unmarked live function
   (`madaros_dce_reachability_gate.sh`, 600-link chain → 512 marks
   → exit 8 instead of 99).
4. **`fn_def: None` is kept.** Those slots are BSS globals. Dropping
   them shifts every later global. Measured 2026-08-06:
   `var G: [i64; 4]` + `print_int(G[0])` printed nothing.
5. **`SOUNIO_DISABLE_MM_DCE=1`** is the A/B. Two from-source
   Madaros ELFs cannot be diffed cheaply. The knob is how #1935
   proved the hole was the pass.

Do not wire it a second time. A second call would either no-op
(`SPEC_DCE_G_READY` already 1) or re-scan and change the mark set
under a filter that already ran.

## Does #1935 make this safe for the first time?

**For the shape it named: yes.** For self-compile: not yet.

#1935 closed a contradiction, not a missing call.

- The filter never deletes an `ItemImpl`.
- The dep-body pass lowers every method of a kept impl.
- `spec_call_expr_callee_name` still returns `empty_name()` for any
  multi-segment path (`Foo::bar`, `Epistemic::measured`).
- Before #1935 the scanner only entered an impl method body when
  the *bare* method name was marked. An associated call marks
  nothing. The wrapper lowered; its free callee had been deleted;
  native codegen emitted `ud2`; SIGILL rc=132.
- The same 2-file program with `SOUNIO_DISABLE_MM_DCE=1` printed
  `42.000000`, rc=0. The pass was the only variable.

Fix: scan impl and trait method bodies **unconditionally**.
Whatever is always lowered must always contribute its callees.
That is mark policy matching retention policy. It is not a fix of
`spec_call_expr_callee_name`.

So the merge-path DCE was **already running and was unsafe** on
assoc→free until 2026-08-19. #1935 is the first time that path is
safe *for that shape*. It is not the first time the path exists.

### What #1935 changes on the self-compile live set

#1935 does not add slots for impl methods — those 1142 were
already in the 7221. It **adds top-level `ItemFn`s** that are
reachable only from impl method bodies, including unused ones.

Upper bound: every declared `fn` survives (10940) + BSS 1892 =
12832. That is under source `IR_MAX_FUNCS - 1 = 16383` and over
the shipped ELF's 8191.

Mark-set risk: shipped ELF reports **7044** marks against
`SPEC_DCE_MAX = 8192`. Headroom **1148**. If unconditional impl
scans insert more than 1148 new hashes, the pass **refuses**,
pruning stops, and the slot census jumps toward 10940+1892. On
the shipped 8191-cap ELF that is a worse overflow. On a 16384
from-source ELF it still fits.

This turn did **not** rebuild Madaros from post-#1935 source.
The 7044/7221 numbers are the pre-#1935 pass running against
post-#1935 trees. A from-source gen1 (seed compiles `main.sio`,
~40 min, does not exercise this DCE) then that gen1 compiling
`main.sio` (exercises #1935) is the measurement that would name
the new mark count. It is the next turn, not this one.

### What can still break after #1935

1. **Lowering-synthesised calls.** Source-level reachability cannot
   see them. Measured 2026-08-06: a 5-fn program with a
   struct-typed global, 9 merged → 7 after DCE, and
   `print_int(G.len)` silently did not execute. The mark set was
   complete for the AST and wrong for the program. The disable
   knob exists because of this, not as decoration.
2. **Multi-segment names still do not mark.** #1935 works around
   the empty callee name by scanning every impl body. A future
   retained body that is *not* an impl/trait method and is only
   reached by `A::b(...)` is the same hole.
3. **Hash 0 is rejected.** `ast_name_hash` returning 0 cannot be
   inserted. That name is invisible to the pass.
4. **Raising `IR_MAX_FUNCS` again** is the answer the gate told
   us not to give. It was given anyway, coordinated, on
   2026-08-18 (`6a55b3bc60`, 8192 → 16384). That cleared the
   *source* slot wall. The shipped ELF was not refreshed. The
   next wall on the from-source 16384 build was
   `IR_MAX_INSTRS = 16384` (`run_compiler_main_self_tests` needed
   33829). `ddcded1284` split that function the same day. Whether
   gen2 now completes on a from-source post-split ELF is
   unmeasured here. The recorded expect is still `check`.

## What this turn is not

- Not a patch to `module_frontend.sio`. The call is there.
- Not a raise of `IR_MAX_FUNCS`.
- Not a refresh of `bin/madaros-linux-x86_64`.
- Not a claim that rung `gen2` is green.
- Not a claim that Madaros compiles itself, let alone to a
  fixed point.

## What a later turn may do

1. Rebuild gen1 from this source (seed → `main.sio`). That ELF
   contains #1935 and 16384. It does not prove #1935.
2. Compile `main.sio` with that gen1, `SOUNIO_MM_SPEC_TRACE=1`,
   and read `dce marks=` and the slot census. If marks ≥ 8192
   the pass refused. If slots ≤ 16383 the slot wall is gone
   *for that ELF*.
3. If gen2 is an ELF that answers `--version` as Madaros, the
   ratchet in `madaros_fixed_point_gate.sh` must be raised from
   `check` to `gen2` (or `run`). Leaving it at `check` after
   progress is how ground is lost silently; raising it without
   the ELF is how a green gate lies.
4. Do not “add” `spec_dce_mark_across_programs` a second time.
5. A follow-on that makes `spec_call_expr_callee_name` return
   the last segment of a multi-segment path is a different
   dispatch. #1935 is the retention-aligned workaround, not
   that dispatch.

## AI disclosure

Measurement and this note by AI agent (grok-cli1) under human
direction, 2026-08-19. GAIDeT-ICMJE 2025. No from-source Madaros
rebuild this turn.
