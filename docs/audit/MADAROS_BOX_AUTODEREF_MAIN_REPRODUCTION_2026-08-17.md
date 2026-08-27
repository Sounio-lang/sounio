<!-- docs:meta
topic_id: repo.docs.audit.madaros-box-autoderef-main-reproduction-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: empryo-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-box-autoderef-main-reproduction-2026-08-17
-->

# Madaros Box Auto-Deref — Main Reproduction, and a Sixth Cause Refuted

Date: 2026-08-17

Status: **reproduced on `origin/main` from a source build; sixth cause
refuted.** The Box read failure on current main is NOT a regression from the
E219 empty-stub trap (`dc58713a7e`). The pre-E219 build fails the same
witness too (rc=139), and the native-v2 IR trace is byte-identical across the
two bases. E219 moved the crash from the field read (SIGSEGV, fabricated 0)
to the `Box::new` call itself (SIGILL, honest `ud2`) — it changed the symptom,
not the defect.

## Instrument

- Worktree: `/workspace/.wt/empryo-1`, branch
  `lane/empryo-1/box-autoderef-20260817`, detached from `origin/main` at
  `9079afbac119` (zero dirty paths at start).
- Source-built Madaros SHA-256:
  `2dcaf1594465e67f07858f4e178409b8c642fec2e02a000f095e8f7cf505745e`
- Build command (never the prebuilt `bin/souc`):

  ```sh
  ulimit -s 524288
  SOUC_BIN= SOUNIO_SOUC_BIN= \
    bash scripts/ci/build_modular_madaros.sh \
    /tmp/empryo-1-box/madaros-baseline
  ```

- Witness: `tests/run-pass/box_all_read_forms.sio` (landed on the open branch
  `codex/box-autoderef-gen-fixedpoint-20260817`, commit `8a74735bc5`; used
  here verbatim from that branch).
- Positive control: the same witness compiled by the source-tracking
  lean_single seed prints `BOXMATRIX OK`, rc=0.

## Measurements on main

| Subject | Result |
|---|---|
| Source-built main Madaros, `check` on witness | rc=0 (`check: OK`) |
| Same, compile witness | rc=0 |
| Same, run witness ELF | **rc=132 (SIGILL)** at `ud2`, PC inside the bodyless `Box_new` stub |
| Minimal witness `let b = Box::new(42); let _ = b; println(7)` | **rc=132** — no field read at all |
| lean_single seed, same witness | `BOXMATRIX OK`, rc=0 |
| Pre-E219 build (base `c5754c0c84`, SHA-256 `785aad0e…`, built by codex-1) on the same witness | **rc=139 (SIGSEGV)** |
| NV2 IR trace of the witness, main vs pre-E219 base | **byte-identical** (`diff` of all `NV2_IR` lines: empty) |
| `tests/run-pass/door1_box_new_array_65536.sio` under source-built main Madaros | compiles, rc=132 |
| `recursive_struct_boxed`, `epistemic_pbox_selftest`, `explicit_deref_field` under the same build | rc=0 — the failure is specific to `Box::new` call sites |

Controls that stay green on the same build: `println(42)` (rc=0) and a plain
struct field read `let e = mk(); println(e.tag)` (rc=0, prints 9). The
instrument therefore fails only on the Box path, and is shown capable of
passing.

## The sixth cause, refuted

**REFUTED:** "The Box read failure on current main is caused by the E219
empty-stub trap (`dc58713a7e`, 2026-08-17) — i.e. it is a new regression."

1. The pre-E219 build (base `c5754c0c84`, which predates `dc58713a7e` —
   verified with `git merge-base --is-ancestor`) fails the identical witness
   with rc=139. The defect exists without E219.
2. The native-v2 IR emitted for the witness is byte-identical between the two
   bases: `field_get field_idx=52` for `b.tag` auto, `field_idx=51` for
   `b.span` auto, explicit forms carrying the payload read
   (`field_get 0` then the layout index). E219 changed nothing in lowering.
3. What E219 did change is the backend's treatment of the bodyless `Box_new`
   symbol: previously prologue+ret (a fabricated return, rax=0, which then
   resolved through handle slot 0 to object_base 0 and faulted at the first
   field read — the July symptom), now `ud2` (the crash at the call itself —
   the August symptom). An honest trap replaced a silent fabrication. Neither
   is the root defect.

## The closed mechanism on main

Two stacked defects; the first masks the second.

**Defect 1 (call site, observable today).** `Box::new(x)` arrives at the
call-expression path already mangled: `expr_to_callee_name_ref` joins the
two-segment path into `Box_new` (7 bytes). `ir_name_is_box`
(`self-hosted/ir/ir.sio:3631`) requires the 3-byte `Box`, so the test at
`self-hosted/ir/lower.sio:14867` never matches and `lower_box_new`
(`lower.sio:13875`) is dead code on main. The call falls through to the
generic named-call path; no `Box_new` body exists; the backend emits an
empty stub — `ud2` since E219. Every `Box::new` program on main dies at the
first `Box::new` call, rc=132, before any Box read executes. Measured: 1367
`Box::new` sites in `self-hosted/` alone (per `fd0fb03f3b`).

The repair exists: `fd0fb03f3b` reads the callee PATH's first segment before
mangling. It is on the PR #1527 branch
(`origin/madaros/self-parse-visibility-box-w44-20260727`), not on main —
verified with `git merge-base --is-ancestor fd0fb03f3b origin/main` → false.

**Defect 2 (auto-deref, latent behind defect 1).** Once `Box::new` allocates
again, the auto-deref miscompile returns: for a Box-typed local or param,
`lookup_local_struct_type` returns the literal `Box`; no struct by that name
exists in the layout table, so `field_idx_from_name` falls back to the
first-byte hash. The IR trace proves it at byte level: `tag` → 116 % 64 =
**52**, `span` → 115 % 64 = **51**, where the `Ex` layout indices are
`kind=0, span=1, tag=2`; and the auto forms are missing the payload read
(`field_get(base, 0)`) that every explicit `(*b).field` form carries. Scalar
fields read garbage with exit 0; struct fields fault. This is the defect
behind the 174 auto-deref sites on Box-typed names in the compiler's own
source (162 in `ir/lower.sio`, 6 in `parser/exprs.sio`, measured in
`7ba8a5e0bf`) and therefore the blocker for gen1 == gen2.

The repair exists and was measured: `6aee037e15` (same PR #1527 branch) keeps
`struct_types` holding `Box` (twelve readers, two `ir_name_is_box` tests),
records the inner-T layout index in a side table read through the locals Box
directly (a by-value `self` read at the field-access site is what killed the
first four attempts), emits `field_get(base, 0)` before the field offset, and
resolves the index against T. Five of six read forms repaired; the sixth
assertion is a `&T` case that fails identically on the pre-fix compiler.

## What this means for the fix lane

1. Defect 1 must land first (PR #1527 or its Box quarter); until then no
   Box auto-deref measurement on main exercises defect 2 — every program
   traps at `Box::new`.
2. Defect 2's repair lives in `self-hosted/ir/lower.sio`, which carries an
   ACTIVE claim by grok-cli5 (Seq work) at the time of writing; coordination
   request sent on the bus (`msg-1787005234-2187520-12936`). The port is
   self-contained (+145 lines) and touches param-bind and field-access
   lowering only.
3. A gate for the repair already exists on the fix branch
   (`scripts/ci/madaros_box_deref_gate.sh`, `fa0360aa48`), runs off the
   shared current-source ELF, and asserts values at non-zero field offsets —
   the four earlier attempts were invalidated by testing offset 0 only.

## Refutation ledger, now six

1. `is_ref` as the carrier — the backend drops it (`7ba8a5e0bf`).
2. A `[Name; 4096]` size cliff — `[i64; 4096]` breaks identically.
3. A struct-valued if-expression store — statement stores break identically.
4. A by-value `self` method reached through `&! Lowerer` — a free function
   taking the table by reference breaks identically.
5. `label_id` dropped at the IR→MIR boundary — the broken instructions carry
   `label_id=0`; the wrong index is chosen in lowering
   (`MADAROS_BOX_AUTODEREF_BACKEND_REFUTATION_2026-08-17.md`).
6. **The E219 empty-stub trap as cause on main — refuted here: pre-E219
   rc=139 on the same witness, byte-identical IR trace.**

## Resolution (same lane, later the same day)

The repair landed on this branch (`lane/empryo-1/box-autoderef-20260817`,
commits `08efd067b4`, `33eb2a314c`, `b553d1c2bc`, `eab77517b3`, rebased onto
the refreshed branch head `e9f0010bd7`): the three Box fixes from the PR
#1527 branch ported onto current main, plus three port-time repairs main's
drift made necessary —

1. `field_is_pointer_for_struct` read `StructFieldEntry.name`, which main
   interned to `name_id`; it now reads `ir_name_at(name_id)`. Without this,
   `let bi = hb.inner` segfaulted during lowering.
2. The pointer class arm (`lower_type_expr_is_pointer_like` → 4) appended
   after main's new bool arm at all six `is_float` construction sites.
3. `box_inner_layout` reset to −1 at both local-bind sites. The locals stack
   is reused across functions (count reset, arrays not), so a stale layout
   from an earlier function's Box param leaked into an unrelated local at the
   same slot — `let after_box_call = mk()` then `after_box_call.tag` lowered
   through a phantom `field_get(0)` and SIGSEGVd. This is the no-field
   control the witness was built around.

Measured on the rebased source build, SHA-256
`807ef05dfa2b456a82a701acda6e5b1dd2815a51c8e3fccf47d73be5e6520a3e`:

- `box_all_read_forms.sio` — **BOXMATRIX OK, rc=0** (pre-port: rc=132).
- param/local scalar and struct reads, auto and explicit: 9/9/111/111.
- `hb.inner.tag` field chain: 9; inline control `hi.inner.tag`: 9.
- `takes_box(Box::new(mk()))` then `mk().tag`: 7 then 9 — the no-field
  control passes.
- `door1_box_new_array_65536` PASS; `explicit_deref_field` PASS;
  `deref_indexed_elem_field_store` ALL PASS; `imported_deref_f64_array`
  PASS; `arm64_nested_deref_store` PASS; `recursive_struct_boxed` and
  `epistemic_pbox_selftest` check OK.

Residual, named rather than hidden: `(*hb.inner).tag` (explicit deref of a
field-chain Box) still SIGSEGVs at runtime, and binding a Box field into a
local reads 0. Both are outside the witness matrix; the 174 auto-deref sites
that block gen1 == gen2 are ident-base reads, which now all resolve against
T.

Both residuals closed in `6ab8e0f2a1` (same branch, source build SHA-256
`fdb8dd633782ba5937b9b9fb66b5986412eb6e46b51b511f6ac6b8b3122b491c`): a
Box-only field predicate (`is_float == 4` AND `named_type_name_id == Box`,
so a `&T` field keeps raw semantics), an explicit-deref route for field
chains, a bind site narrowed to Box fields, and a read site that treats a
Box-tagged local with layout −1 as "deref, then resolve by name" — the
route the explicit `(*p).f` form already takes. The witness grew 12 → 14
checks and prints `BOXMATRIX OK`; `let bi = hb.inner; bi.tag` reads 9 (was
0) and `(*hb.inner).tag` reads 9 (was SIGSEGV). The Box surface measured in
this audit is now closed: every read form — param, local, field chain,
auto and explicit — resolves against T.

## Fixed-point ladder — attribution, not progress

The Box repair was dispatched because it "blocks gen1 == gen2". Measured
against the fixed-point ladder (`scripts/ci/madaros_fixed_point_gate.sh`),
it neither advances nor regresses that ladder — and saying so precisely is
the point.

Both the pre-fix build (SHA-256 `2dcaf159…`) and the post-repair build
(SHA-256 `ac82ead7…`) reach rung **check** (the recorded rung) and fail the
same way at rung **gen2**:

    IR slot census: globals 1891 + functions 7206 = 9097 (max 8191, over by 906)
    IR lowering failed during merge: too many lowered items: combined globals
    and functions exceed shared IR module capacity (max 8191 slots)

The wall is identical byte-for-byte across the two builds. So the Box
auto-deref miscompile was never what stopped Madaros compiling itself — the
merge-capacity overflow was, and is. The Box repair removes a real
miscompile that corrupts any `Box<T>` read (including the 174 ident-base
sites in the compiler's own source), but the fixed-point ladder's next rung
is gated by a capacity overflow that cross-module DCE already runs against
and still loses.

Sharpened by an A/B on the post-repair build (2026-08-18, `ac82ead7…`):

    SOUNIO_MM_SPEC_TRACE=1   dce marks=7029
                             census 1891 + 7206 = 9097 (over by 906)
    SOUNIO_DISABLE_MM_DCE=1  census 1891 + 10929 = 12820 (over by 4629)

Cross-module DCE (`spec_dce_mark_across_programs`, wired onto the ordinary
path by `146f5b039f`, hardened by `541536f777` and `731aee7b6f`) IS running
and removes ~3700 dead functions. It is not that the pruning is missing — it
is that the surviving 7206 live functions plus 1891 BSS globals still total
9097 slots against a cap of `IR_MAX_FUNCS - 1 = 8191`. The gate's header
records a name-based reachability census of 5997 live declarations, which
would fit; the lowering's own count is higher because it counts every
surviving `ItemFn` slot (impl methods included) and every BSS global, not
just top-level reachable names. Closing rung gen2 therefore needs either a
larger IR slot budget or a tighter live-set, not the Box fix.

What this means: do not expect gen2 to turn green from the Box fix alone.
The recorded rung stays `check`; the ladder is unchanged, and that is the
correct, bounded result.

Attempted the documented lever and found why it is not self-contained
(2026-08-18, same lane). Raising `IR_MAX_FUNCS` 8192 → 16384 requires, in
the same change:

1. `ir_region_table_capacity()` (`self-hosted/ir/ir.sio:1141`, currently
   8448) MUST exceed `IR_MAX_FUNCS` — its own comment records the bisect
   ("8189 live slots compile, 8190 and 8191 do not"). At 16384 the region
   table is the binding constraint and the contract check refuses.
2. The coupled array literals: `IrModule.functions` (`[IrFunction; 8192]`),
   `normalize.sio`'s two `[IrFunction; 8192]` sites, the four backends'
   `fn_offsets: [i64; 8192]` (`codegen_x86_linux`, `elf`, `elf_bulk`,
   `reloc`), and `SPEC_DCE_SLOTS`/`SPEC_DCE_MAX` plus the `[i64; 16384]`
   mark arrays in `specializer.sio`.
3. The fixture `tests/multimodule/ir_capacity/` raised past the old ceiling,
   and the literal sweep the probe gate's header demands.

A partial raise (constant + arrays, without the region table) was applied
and reverted in this worktree; nothing shipped. The lever is real but it is
a coordinated multi-file change with its own risk profile, not a one-line
constant bump. That is the next task, named and scoped, and it is separate
from the Box repair this audit is about.

The lever was then PROVEN by the minimal coordinated slice (2026-08-18,
branch `lane/empryo-1/gen2-raise-measure-20260818`, source build SHA-256
`da172b4f…`). All eight coupled files were free of claims at the time
(grok-cli2's `lower.sio` claim expired between 17:55Z and 17:58Z; fable-1's
`codegen_x86_linux.sio` claim had cleared). The slice raised `IR_MAX_FUNCS`
8192 → 16384 together with every array indexed by the IrModule.functions
slot — `IrModule.functions`, `normalize.sio`'s two `[IrFunction; …]` sites,
`lower.sio`'s `elem_kinds`, `fn_offsets` in `codegen_x86_linux`/`elf`/
`elf_bulk`/`reloc`/`frame`, `elf.sio`'s `name_offsets` — and the region
table (`IR_REGION_*` arrays + `ir_region_table_capacity()` 8448 → 16640,
keeping the +256 margin). The specializer needed no change: reachable marks
are 7029, already under `SPEC_DCE_MAX=8192`.

Result, measured on the from-source raise build:

    rung check   rc=0 errors=0          (gen1 typechecks main.sio)
    rung gen2    rc=139 — but the slot-census wall is GONE. The failure is
                 now a DIFFERENT wall: `run_compiler_main_self_tests` needs
                 33829 IR instructions vs IR_MAX_INSTRS=16384, and the
                 refusal path then segfaults.

No regressions: `box_all_read_forms.sio` still prints `BOXMATRIX OK` rc=0,
and the DCE reachability gate passes all three arms (keeps 602 live, drops
300/303 dead, carries 6000 functions to a correct binary).

SAFE STOP POINT, per the execution order: the coordinated `IR_MAX_FUNCS`
raise is self-consistent and regression-free, and it is where this lane
stops. The next wall is per-function `IR_MAX_INSTRS`, which is NOT a simple
bump — `ir/dce.sio:31` caps liveness at `DCE_MAX_INSTRS=8192`, and the
`irfunction_instr_capacity_coherence_gate.sh` header is explicit that a
truncated liveness analysis is a WRONG analysis (a use past the cap is never
recorded, so its definition looks dead and the sweep deletes live code,
silently, at rc=0). Raising `IR_MAX_INSTRS` without raising `DCE_MAX_INSTRS`
and every per-instruction context that stops at its own cap converts an
honest refusal into silent miscompilation. That is a separate, larger task.

The `IR_MAX_INSTRS` wall was then cleared WITHOUT raising any capacity
(2026-08-18, branch `lane/empryo-1/normalize-byref-20260818`, source build
SHA-256 `7ffc6c82…`). The compiler's own error message prescribed the fix:
"split it into smaller functions". `run_compiler_main_self_tests`
(`self-hosted/compiler/main.sio`, 5839-line body, 1163 independent test
blocks) lowered to 33829 IR instructions — past the 16384 cap. Splitting it
into ten part-functions (`compiler_main_self_tests_part_01..10`, ~116 blocks
each) keeps every part far under the IR cap AND under the DCE/const-prop
8192 caps, so every analysis pass stays complete on every part — no refusal,
no truncation, and no capacity constant touched. `DCE_MAX_INSTRS`,
`CP_MAX_INSTRS`, and every lateral array stay exactly where they are.

Measured on the from-source split build:

    fixed-point gate   GATE_RC=0, reached `check` as recorded.
                       The 33829/IR_MAX_INSTRS wall is GONE (0 occurrences in
                       the gen2 log). gen2 now progresses past typecheck into
                       lowering before hitting a deeper, different SIGSEGV —
                       a separate wall, not this one.
    box_all_read_forms BOXMATRIX OK rc=0 — no regression
    dce_reachability   all three arms pass
    typecheck main.sio 80 errors before == 80 after (all pre-existing
                       ontology E175); the split adds ZERO

This is the safe stop point for the capacity question: the instruction wall
did not need a raise, it needed the function split. Raising `IR_MAX_INSTRS`
remains the wrong lever here, for the liveness-capacity reason above.
