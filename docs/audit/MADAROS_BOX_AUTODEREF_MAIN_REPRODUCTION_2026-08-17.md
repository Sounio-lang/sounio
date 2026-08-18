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
