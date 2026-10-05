<!-- docs:meta
topic_id: repo.docs.audit.soir-blocked-drafts-triage-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.soir-blocked-drafts-triage-2026-08-17
-->

# Triage — the five [BLOCKED] SOIR drafts (#869, #870, #881, #883, #885)

**Date:** 2026-08-17 · **Audited tree:** `0b0c5cdd5b` (= `origin/main`, "fix(docs):
registry sync preserves real provenance … (#1752)") · **Author:** glm-cli1
(Wave 3 sweep, brief in the lane `.scratch/W3_SOIR_BLOCKED.md`).

**Method.** The five drafts were opened 2026-07-14 and tagged `[BLOCKED]` against
a tree that has since taken ~2,000 main-side commits, including the IR arena and
SoA landings that rewrote `self-hosted/ir/` underneath them. This audit answers,
per draft, from the code at `0b0c5cdd5b` rather than from PR text: what the
recorded blocker was, whether it still exists, and whether the change is still
wanted. Serializer facts were verified by checking the commit out clean and
running its own entrypoint (`bin/souc check self-hosted/ir/serialize.sio` with
`SOUC_BIN`/`SOUNIO_STDLIB_PATH` unset). **Nothing was closed or merged**; all
recommendations are for the founder.

---

## 0. Ground truth that reframes all five

**`self-hosted/ir/serialize.sio` — the file every draft modifies — does not
typecheck on main, sits outside the Madaros build closure, and has no CI gate.**

`bin/souc check` at `0b0c5cdd5b` (byte-identical to `453b2e6e2f` for every file
involved): `verdict=1`, **29 errors** —

| error | site | verified cause |
|---|---|---|
| 25× E137 `use of undeclared variable … name len` | `deserialize_ir_epistemic_section` — `if p < len` guards at serialize.sio:4044–4148+ | the signature is `(buf, pos)`, no `len` parameter. Signature and guards landed together in `c694966317` (2026-03-07, "complete epistemic core bootstrap lane"): **the errors are as old as the function — five months, unnoticed** |
| E046 `struct literal has wrong number of fields` | `deserialize_ir_function` (serialize.sio:655) | builds a 16-field literal with `instrs:`; `IrFunction` now has 18 fields — `region:` + `float_reg_bits`, no `instrs` (ir.sio:1724–1770) |
| E002 `expected [IrFunction; 8192], found [IrFunction; 1024]` | `deserialize_ir_module` (serialize.sio:4491) | `IR_MAX_FUNCS` went 2048→8192 in #1672 (2026-08-09); the 1024-element local stayed |
| E012 `no field name on type IrInstr` | `serialize_ir_instr` | `IrInstr.name` was interned to `name_id` (517af688a7); the codec was not updated |
| E175 `function is private in its defining module` | `read_type_tag` → `check/types::ty_clifford` | `fn ty_clifford` at check/types.sio:995 is not `pub` |

Why nothing catches this: `main.sio` no longer imports `module_loader`
(the only mention is a comment, compiler/main.sio:5); `module_loader.sio` is the
**sole** importer of `ir::serialize`; no `use module_loader` exists anywhere in
`self-hosted/compiler/`; and the only SOIR-referencing gates
(`soir_v5_empty_reader_gate.sh`, `soir_v6_bss_layout_gate.sh`,
`ir_module_arena_v2_soir_v5_bridge_gate.sh`) exercise the *shadow verticals*
(`soir_reader/writer`, `soir_v6_*`, `arena_v2_*`), never `ir/serialize.sio`.
The checker also warns ~13.75 MB stack frames for `serialize_ir_module_into` /
`deserialize_ir_module` / `ir_empty_module` — the wide-`IrModule` problem in
miniature, which open draft #1729 (B3 BSS pool) is now taking off-struct.

Framing corrections against the wave brief: the arena/SoA landed via **#1650**
(variable-size region arena, merged 2026-08-09) and **#1717** ("replay, not
merge" of `probe/ir-soa-phase0`, commit `042c29be53`, merged 2026-08-12) — #1649
was the *issue* (closed by #1726) and #1695 was a merge attempt closed unmerged.
WS-B's `scripts/ci/soir_roundtrip_gate.sh` is dispatched (MADAROS_FOCUS_PLAN
2026-08-16, wave 1, codex-2) but does not exist at `0b0c5cdd5b`.

---

## 1. #870 — fix(ir): make SOIR capacity handling fail closed [BLOCKED]

**Original blocker (quoted from the PR):**

> `BLK-20260713-IR-T17-WIDE-MODULE-ARENA` — "Observed: 64 MiB compile rc139;
> unlimited compile rc0; executable immediate rc139 … Next-Action: provide
> runtime-safe storage/addressing for the current ~1.12 GB IrModule layout,
> then rerun T17."
>
> `BLK-20260713-NATIVE-V2-HEAP-ALLOC-RUNTIME` — "Observed: isolated
> current-source heap_alloc(8) probe compiles rc0 and runs rc139 … Next-Action:
> repair or classify native-v2 allocator initialization before using heap
> storage as the wide-module witness."

**Does it still exist on `0b0c5cdd5b`? Both blockers are gone.**

- *Heap:* **#876 "Fix native-v2 heap allocation builtins" merged
  2026-07-14T02:54Z — 2 h 17 min after this draft opened (00:37Z)** — with the
  same symptom and root cause ("`heap_alloc(8)` compiled successfully but
  crashed at runtime"; the backend did not classify heap builtins), plus a
  permanent witness (`tests/run-pass/native_v2_heap_alloc_builtin_roundtrip.sio`,
  `//@ requires: madaros`). Heap has been in routine use since: heap-backed
  columnar DataFrame (#1154, merged 07-18), compiler heap-accounting fixes
  (91d376689d, 00075a3060).
- *Wide module:* dissolved by design. `IrFunction.instrs` is now
  `region: IrInstrRegion` (ir.sio:1726–1730, comment naming #1649: "used to be
  `[IrInstr; 4096]` inline … every IrModule ~2 GB whether it uses it or not");
  instructions live in global SoA columns (`IR_A_OP…`, `[i64; 1048576]`,
  ir.sio:813+). A by-value `IrModule` measures ~13.75 MB (the checker's own
  warning), and #1729 moves even `functions` off-struct into BSS pools.

**Still wanted?** The *goal* is more valid than ever — the live codec still
fails **open**: bad magic/version silently returns `ir_empty_module()`
(serialize.sio:4453); wire `fn_count`/`string_count` drive unclamped loop
writes against fixed local arrays; `bss_size` decodes hardcoded to `0`
(serialize.sio:671) — exactly the silent-corruption class this PR said to
reject; there is no envelope preflight. But the *patch* is written against a
v4/v5 wire layout and a struct that no longer exist, in a file that no longer
compiles and is outside every build/check surface.

**Recommendation: CLOSE (superseded).** Blockers resolved by #876 and the
#1649/#1650/#1717 arena landings; there is nothing left to rebase onto. Carry
the kernel forward: the fail-closed requirements (silent-empty returns,
unclamped wire counts, BSS-bearing rejection, envelope preflight) belong in
WS-B's codec-repair brief.

## 2. #869 — DefId provenance partial #854 — SOIR blocked

**Original blocker (quoted):**

> `BLK-20260713-IR-SERIALIZE-CAPACITY-DRIFT` (B2) — "Observed: base
> `IrFunction.instrs` has capacity 2048 while `ir_empty_function` still
> initializes 1024; `deserialize_ir_module` similarly allocates 1024 functions
> for a 2048-function `IrModule`. Madaros reports E016 … Acceptance gate: T17
> v5 provenance roundtrip plus genuine v4 downgrade both complete with `rc=0`.
> Next action: align the 1024/2048 capacities."

**Does it still exist on `0b0c5cdd5b`? The named drift was fixed
independently; the substance never landed.** **#1171 "align IrFunction
instruction capacity" (merged 2026-07-19)** performed the literal next-action
plus a coherence gate (`irfunction_instr_capacity_coherence_gate.sh`) — and its
Slurm A/B honestly showed the drift was not even causal for the then-current
readback failure. The arena then removed `instrs` as an array entirely. The
drift *class* escalated: the 1024-vs-8192 mismatch in `deserialize_ir_module`
is now a hard E002 compile error (§0). Provenance: `SOIR_VERSION` is still 4
(serialize.sio:48) and `IrFunction` has no `defining_module_id`;
`check/check.sio`'s signature-level `defining_module_id` (commits 64651510e8,
b709201a8b, bbc3fb3a5b) serves E175/E177 *diagnostics*, not wire provenance.

**Still wanted?** Parent #854 is open, but the 2026-08-16 focus plan scopes
SOIR as "verify existing, add one gate" (WS-B) with no provenance workstream;
serialization investment is going to WS-C/WS-D. The draft's paired-float-marker
mechanics are also obsolete (`float_reg_bits` replaced instruction-stream
marker smuggling, #1669).

**Recommendation: CLOSE (superseded — by #1171, the arena/SoA restructuring,
and the E175 diagnostic lane).** If provenance is wanted when #854 resumes,
re-derive it against the post-arena struct and the post-WS-B codec; rebasing
this stack (its base #867 is itself an untouched draft) is not a path.

## 3. #881 — feat(ir): add explicit heap module bridge [BLOCKED]

**Original blocker (quoted):**

> `BLK-20260714-IR-HEAP-SERIALIZE-CLOSURE` (B1, evidence #879) — "seed still
> segfaults at `build_modular_madaros.sh:115` while emitting the newly
> activated full `ir::serialize` closure." Acceptance required splitting a
> minimal SOIR core, completing a source-fresh build, and passing
> `scripts/ci/ir_module_heap_bridge_gate.sh` with `IR_MODULE_HEAP_BRIDGE_PASS`.

**Does it still exist on `0b0c5cdd5b`? No.** #879 was **closed
2026-07-14T04:42Z**, twenty seconds after #883 opened — resolved locally by
that split lane, which was never merged. More fundamentally, the bridge's
reason to exist — dodging the ~1.12 GB inline `IrModule` — was eliminated by
the arena. No `IrModuleHeapBridge`/heap-bridge symbol exists on main. The live
successors are different shapes: `arena_v2_shadow.sio` (a 2-slot scalar
ModuleId/generation proof-of-concept that "owns no legacy module aggregate and
is not imported by the default compiler", landed via #1140 on 07-18) and #1729's
`*mut [IrFunction; 8192]` BSS pool. Side conditions: #877 (checker rejected
valid linear consume-and-return threading) was closed 2026-08-10; #878 (unknown
wire tags decode as `IrNop`) remains open.

**Recommendation: CLOSE (superseded — by the #1649 arena, the arena-v2 shadow
lane #1140, and #1729).**

## 4. #883 — refactor(ir): extract bounded SOIR core [BLOCKED]

**Original blocker (quoted):**

> "Durable blocker: #882, `BLK-20260714-IR-HEAP-POINTER-GRAPH-MATERIALIZATION`
> (`B1`, `compiler-semantics`, `E3`) … The zeroed reservation contains a null
> slot. This falsifies the bridge's prior assumption that zeroed storage
> already contains a live inline `IrFunction`; `IrFunction.instrs` needs the
> same compiler-owned graph materialization treatment." The split's purpose was
> shrinking the heap bridge's dependency closure "427770 source bytes → 316416".

**Does it still exist on `0b0c5cdd5b`?** Issue #882 is still open, but it
describes a bridge that does not exist on main, and the split's only consumer
was that bridge. Meanwhile the *pattern* it pioneered — small bounded SOIR
verticals instead of the 4.8K-line facade — became the repo's standard and
landed repeatedly: `soir_reader/writer` (bounded empty-v5, f891fbd137, 07-18)
and `soir_v6_reader/writer` (BSS layout, 1fb1471ca3, 07-19), each with its own
gate. The general facade was never split and is now the uncompilable file of §0.

**Recommendation: CLOSE (superseded — by the bounded-vertical pattern, which
landed; the facade's repair belongs to WS-B and will follow the same vertical
pattern, not this core/facade split).**

## 5. #885 — fix(ir): materialize bounded heap graphs [BLOCKED]

**Original blocker (quoted):**

> Root cause: "The raw zeroed `IrModule` reservation does not contain valid
> aggregate objects under pointer/handle lowering. Its live aggregate slots are
> null until canonical values are assigned." Gate status at the time:
> `IR_MODULE_HEAP_BRIDGE_FAIL reason=semantic_assertion` /
> `internal_self_test_rc_1`; plus the separate repeated-use lifecycle blocker
> #884.

**Does it still exist on `0b0c5cdd5b`?** #882 and #884 are both still open as
issues, but their subject — the heap bridge — is absent from main. The hazard
*class* was defused by construction elsewhere: the arena quarantines zeroed
handles by design (region slot 0 is the quarantine, ir.sio:1141–1147: "a zeroed
handle cannot address real data"); aggregate binding was changed to value-copy
(#1480, closing #1475); and #1741 (merged 2026-08-15, "allocate the arena
region for hand-built IrFunctions, 38+1 sites") performed the
canonical-constructor materialization for exactly the hand-built-function case
this draft worried about.

**Recommendation: CLOSE (superseded — by arena quarantine + #1480 + #1741).**
Flag #882/#884 for re-triage in the same sweep: open issues about a design that
no longer exists on main.

---

## 6. Verified timeline

| date | event |
|---|---|
| 07-13/14 | stack opened #867→#869→#870→(#880)→#881→#883→#885, all `[BLOCKED]` |
| 07-14 02:54 | **#876 merges** — heap_alloc rc139 fixed on main, 2 h 17 min after #870 opened |
| 07-14 04:41–05:09 | #883, #885 open; **#879 closed** (resolved locally, unmerged) |
| 07-18 | #1140 shadow Arena v2 SOIR v5 vertical; #1154 heap-backed DataFrame; `soir_reader/writer` |
| 07-19 | **#1171 merges** (capacity alignment + coherence gate); `soir_v6_*` |
| 08-09→12 | #1650 arena; #1672 (`IR_MAX_FUNCS` 8192); name interning; **#1717 SoA replay** — serialize.sio gets a one-line write-path fix, the read path is left to rot |
| 08-10 | ten sibling `[DO NOT MERGE]` drafts bulk-closed (#887, #893, #910, #946, #947, #956, #960, #961, #965, #973) — these five were not in that sweep |
| 08-15 | #1741 (regions for hand-built IrFunctions); #1164/#1174 shadow verticals merged |
| 08-16 | focus plan; WS-B census dispatched (codex-2); #1729 BSS-pool draft active |
| 08-17 | this audit; `453b2e6e2f → 0b0c5cdd5b` touched nothing under `self-hosted/ir/` or `self-hosted/compiler/` |

## 7. What carries forward

All five recommendations are **CLOSE**, each superseded by work that already
landed. Two items outlive the drafts:

1. **WS-B will hit §0 cold.** The dispatched census + roundtrip gate author
   (codex-2) should receive the 29-error inventory; "verify existing SOIR"
   currently means "resurrect `ir/serialize.sio` first." The repair should
   follow the repo's landed vertical pattern (`soir_reader`-style bounded
   modules), not any of the five drafts' designs.
2. **#870's fail-closed intent survives its branch.** The live codec silently
   returns empty modules on bad magic, trusts wire counts, and zeroes
   `bss_size` on decode. Those requirements should be written into WS-B's
   codec-repair acceptance, and #882/#884 re-triaged as stale.
