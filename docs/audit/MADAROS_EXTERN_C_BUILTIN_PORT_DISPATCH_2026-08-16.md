<!-- docs:meta
topic_id: repo.docs.audit.madaros-extern-c-builtin-port-dispatch-2026-08-16
authority: repo_only
audience: users
last_validated: 2026-08-16
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-extern-c-builtin-port-dispatch-2026-08-16
-->

# Track A — make `extern "C"` genuine under default Madaros via the builtin registry — dispatch

**Date:** 2026-08-16
**Engine:** Madaros v0.80.0 (default `bin/souc`), modular pipeline (`self-hosted/compiler/main.sio` + `self-hosted/{parser,check,ir,native}/`)
**Parent:** `docs/audit/EXTERN_C_FFI_SILENT_NOOP_DISPATCH_2026-08-13.md` — Track A, explicitly left open by the Track B commit (`e1109c4773`: "Track A (porting the fix to Madaros) remains open"). Both repros re-verified live under default `bin/souc` on 2026-08-15/16: `getpid()` fabricates `0`; `system()` prints rc=0 without executing.
**Owner:** lane `glm-cli1` (this dispatch is executed by that lane in three sequenced steps, below)
**Status:** OPEN — executing. `self-hosted/` edits are confined to `self-hosted/native/codegen_x86_linux.sio` + `self-hosted/parser/{items,mod}.sio`; `lean_single.sio` and `main.sio`/`lower.sio` are untouched (claimed or contested by other lanes).

## Why this dispatch

`bin/souc` routes to Madaros by default, so every `KNOWN_LIMITATIONS.md`-documented "unblocked" consumer (`stdlib/os/`, `stdlib/mem/`, `stdlib/sync/mutex.sio`) is, on the default engine, either failing to type-check (dropped extern declarations) or silently receiving fabricated zeros. This dispatch localises both failure loci, rejects one repair route with evidence, selects another, and records the acceptance gate.

## The two loci (both verified in source)

### M1 — parser: a brace-form `extern "C"` block keeps only its FIRST declaration

`self-hosted/parser/items.sio:4191-4247`, `parse_extern_fn_item`. The brace-form loop **parses every** declaration but retains only the first into `first_*` locals (`:4228-4233`); the comment at `:4188-4190` admits this ("conservatively retain only the first parsed declaration"). Every later name in the block vanishes from the AST, so each of its call sites fails with `E137 use of undeclared variable`. `parse_export_fn_item` (`:4278-4285`) delegates here and inherits the defect.

Baseline measured on the default engine (spans mapped to source text at the call sites):

| file | decls in block | retained | E137 sites |
|---|---|---|---|
| `stdlib/os/process.sio` | `getpid, getppid, exit, abort` | `getpid` | `getppid`@990..997, `abort`@1855..1860 |
| `stdlib/sync/mutex.sio` | `malloc, free, pthread_mutex_{init,lock,trylock,unlock,destroy}` | `malloc` | all 5 pthreads @984..1667, `free`@1679..1683 |
| `stdlib/mem/arena.sio` | `malloc, free` | `malloc` | `free`@3214..3218 (+3 `null_mut` E137s — separate, pre-existing) |
| `stdlib/mem/pool.sio` | `malloc, free` | `malloc` | `free`@5144..5148 (+5 `null_mut` E137s — separate, pre-existing) |
| `stdlib/mem/arc.sio` | — | — | clean (rc=0) |

The `null_mut` errors are **not** M1 (not extern names) and are expected to survive the fix; recorded so the post-fix re-run is judged honestly.

### M2 — lowering: a bodyless extern `FnDef` never takes the extern call path; the call becomes a plain call to an empty body returning 0

A bodyless extern declaration gets `compile_strategy = lowerer_compute_strategy_from_ast_ref(...)` (`self-hosted/ir/lower.sio:1399`), and that function (`lower.sio:1753-1783`) keys only on effects and return type — never on bodylessness — so it returns `IR_STRATEGY_STANDARD`, never `6`/`IR_STRATEGY_EXTERN`. The call-site guard `if callee_strat_ext == 6` (`lower.sio:16970-16974`) therefore never fires; the call lowers as a plain `ir_call` into an empty body → returns 0. This is the fabricated-`0` mechanism.

## Rejected route: dynamic linking via `extern_relocs` (dead apparatus)

`lower_call_extern` (`self-hosted/native/lower_ir.sio:983-1004`) emits `call rel32` placeholders and records `ExternReloc` entries (`:995-998`), but `extern_relocs` has **zero consumers** in the tree (only the declaration at `self-hosted/native/frame.sio:328` and the two population lines). Wiring it up means building real ELF dynamic linking (`.dynamic`/`.dynsym`/DT_NEEDED, PLT/GOT) into the native emitter — a large, risky surface for no additional capability over the builtin route below. Rejected.

## Step 0 — empirical result (route refined, 2026-08-16)

The dispatch's caveat demanded the by-name chain be verified before any emitter was written. It was, with three probes under default `bin/souc`:

1. `extern "C" { fn sqrt(x: f64) -> f64 }` then `sqrt(4.0)` → **exactly 0.0** (fabricated), while the same call with **no declaration** → 2.0. 
2. `extern "C" { fn str_len(s: string) -> i64 }` then `str_len("abcd")` → **broken**, while undeclared `str_len` works everywhere.
3. Undeclared `getpid()` and `zzz_nosuchfn()` both fail check with E137 — the checker resolves undeclared names only through its runtime-builtin table (`checker_collect_runtime_builtins_inplace`, `self-hosted/check/check.sio:3302`, whose own comment documents the same failure class from 2026-07-13).

Conclusion: a bodyless extern `FnDef` **lowers as a real function whose body returns 0, and that fn shadows the by-name builtin resolution** — for names already in the registry, not just absent ones. Adding registry ids alone would change nothing. The route is therefore refined to the lean_single `strip_extern_blocks` concept, ported to the Madaros parser (the "Chosen route" machinery below remains the backend half):

- **Parser** (`items.sio`): every extern "C" declaration — not just the first — is rewritten into an ordinary Sounio wrapper fn preserving the declared signature, whose body forwards to a `ffi_<name>` intrinsic (distinct name ⇒ no shadow). Extra declarations in a brace block go through a pending buffer drained by both item loops in `mod.sio` (fixing M1 in the same change). An extern whose intrinsic has no checker+registry entry now fails at the wrapper's inner call with a clear E137 — the "fail with a clear, documented error" the parent dispatch Track B asked for.
- **Checker** (`check.sio:3302` table): bind `ffi_getpid`/`ffi_getppid` (PR-A1); `ffi_system` is added only together with its emitter (PR-A2), so an unimplemented `system()` stays a check-time error rather than a runtime zero.
- **Registry + emitters** (`codegen_x86_linux.sio`): ids 31/32 (getpid/getppid, syscalls 39/110) — this is where the original "add ids" route applied, and it is necessary, just not sufficient.



`ir_module_ensure_builtin_call_targets` (`self-hosted/compiler/module_frontend.sio:2154`, invoked at `:2264` and `:6134`) scans every `IrCall`/`IrCallSret` **by callee name**, asks `native_v2_builtin_id_for_name` (`self-hosted/native/codegen_x86_linux.sio:6575`), and if the target is missing appends an empty stub fn and rebinds the call site (`:2173-2189`). `native_v2_builtin_id_for_func_ref` (`:1102`) then hands any `instr_count == 0` function to `native_v2_emit_builtin_by_id_into` (`:6627`), which emits a hand-coded body (idiom: `emit_builtin_heap_alloc_into` `:3445`, syscall form `:3482-3483`).

The registry already carries `heap_alloc`/`heap_free` (ids 23/24) and the float intrinsics (`sqrt` 14, `exp` 16, `log` 17, `sin` 18, `cos` 19, `tan` 27, `pow` 28, `asin` 29, `acos` 30). **`getpid`, `getppid`, `system` are absent** — that is the entire Track A gap for the documented allowlist. Adding them needs only `codegen_x86_linux.sio`: name recognisers (shape of `name_is_heap_alloc` `:985`), ids (31/32 free), emitters, dispatch arms.

**Caveat recorded before implementation (per the plan discipline):** the chain IrCall-by-name → ensure_builtin → empty-stub rebind → builtin-by-func-ref is read from source and strongly supported (`heap_alloc`, `read_file` etc. demonstrably reach codegen this way), but the *bodyless-extern-decl* case specifically has not been observed end-to-end. **Step 0 of the fix is empirical**: compile a `getpid` probe with a freshly built Madaros and confirm the mechanism engages — if the call still returns 0 with no emitter added but the name registered, stop and re-localise rather than force.

`self-hosted/native/codegen.sio` carries a twin registry (`name_is_*` `:905+`, dispatch `:3995`/`:4162`); `module_frontend.sio:15` imports from `native::codegen_x86_linux`, making that the live path. The twin is left untouched and the decision recorded here; a divergent twin is worse than an untouched one.

## Sequenced execution (this lane)

1. **PR-A1** — the full refined route for `getpid`/`getppid`: parser wrapper rewrite (fixes M1 as a side effect — every declaration of a brace block becomes an item), checker entries `ffi_getpid`/`ffi_getppid`, registry ids 31/32 + emitters (`mov rax,39; syscall; ret` / `mov rax,110; syscall; ret`) in `codegen_x86_linux.sio`. Acceptance: `tests/run-pass/ffi_integer_return.sio` — **currently RED on the default suite** (`getpid()==0`) — turns green; new `tests/run-pass/ffi_getppid_return.sio` and `tests/run-pass/ffi_extern_block_multi_decl.sio` (two declarations in one block — the M1 shape) pass; the four stdlib baselines above re-run.
2. **PR-A2** — `system()` builtin: checker entry `ffi_system` + id 33 + emitter fork(57)/execve(59, `"/bin/sh"`,`"-c"`,cmd)/wait4(61), `rdi`=cmd on entry, string literals via the data-reloc mechanism `native_v2_persist_builtin_emit_into` (`:4089`) documents. Then remove `ffi_system_exec.sio`'s `//@ ignore` (the test runs under the default engine) and add a Madaros arm to `scripts/ci/ffi_extern_c_gate.sh`. Halt-partial (allowlist without `system`) is an acceptable outcome if literal placement proves disproportionate — recorded now, per "halt is a deliverable".
3. **PR-A3** — folded into PR-A1 by the step-0 refinement (the wrapper rewrite emits every declaration; there is no longer a separate parser change).

## Acceptance gate (the parent dispatch's list, for this track)

1. Parent Repro 1 and Repro 2 pass under **default** `bin/souc` (`system("touch …")` creates the file; `getpid()` nonzero).
2. `ffi_integer_return.sio`, `ffi_getppid_return.sio`, `ffi_extern_block_multi_decl.sio` green under the default suite; `ffi_system_exec.sio`'s `//@ ignore` removed (engine-forcing no longer needed) and green.
3. The four stdlib baselines above re-run: the extern-name E137s are gone; remaining errors reported honestly (expected: the `null_mut` family).
4. `make madaros-full-gate` PASS (the only valid Madaros proof gate), plus `scripts/ci/madaros_operational_contract_gate.sh`.

## Impact if unaddressed

Every default-engine user of `extern "C"` — the several `stdlib/` modules `KNOWN_LIMITATIONS.md:174` credits as unblocked — keeps either failing type-check (multi-decl blocks) or silently receiving fabricated zeros (single decls). `KNOWN_LIMITATIONS.md:174`'s "fixed" claim is false on the default engine until this lands; that paragraph is corrected in the close-out.

## AI disclosure

Localisation (source-read, cited line ranges), route analysis, baseline capture, and implementation by AI agent (Claude) under human direction, 2026-08-15/16, on Madaros v0.80.0 default engine and the source-built lean_single fixed point. All repros re-runnable with `unset SOUC_BIN SOUNIO_STDLIB_PATH` from the worktree root. `self-hosted/compiler/lean_single.sio`, `main.sio`, and `ir/lower.sio` were not modified. GAIDeT-ICMJE 2025.
