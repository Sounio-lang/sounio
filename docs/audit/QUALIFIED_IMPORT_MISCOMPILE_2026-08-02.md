<!-- docs:meta
topic_id: repo.docs.audit.qualified-import-miscompile-2026-08-02
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.qualified-import-miscompile-2026-08-02
-->

# AUDIT: qualified-import (`use m; m::f(...)`) miscompile — 2026-08-02

- Author: kimi-swarm, lane `miscompile-hunt-20260802` (ontology-frontiers round 7)
- Branch: `research/zd-fiber-antisymmetry-lemma-20260731`
- Compiler: `bin/souc` → Madaros v0.80.0 (self-hosted modular compiler)
- Scope: read-only analysis. **No compiler source was edited.** The fix below is
  delivered as an UNAPPLIED candidate diff:
  `artifacts/ontology-frontiers/compiler-repros/qualified_import_fix_candidate.diff`
- Coordination: `bin/sounio-coord brief` showed ACTIVE claims by other lanes on
  `self-hosted/check/*`, `self-hosted/ir/lower.sio`, `self-hosted/compiler/*`
  (lanes `issue901-authority-current-20260802`,
  `issue901-associator-integration2-20260802`). All findings here are derived
  from reading only; the candidate patch is left for those lanes / the human
  author to apply.

## Symptom

With the whole-module (qualified) import form

```
use p5_qualified_leaf;
fn main() -> i32 with IO, Mut {
    var x: [i64; 2] = [0; 2]
    p5_qualified_leaf::bump(&!x)     // &! mutation through imported fn
    println(x[0])                    // expected 1
    println(p5_qualified_leaf::add(40, 2))  // scalar imported call, expected 42
    ...
}
```

the compiled program misbehaves in three independently observable ways:

1. **`&!` mutations are silently lost** — `bump(&!x)` returns without any
   effect; `x[0]` prints `0`. No diagnostic from `check` or `build`.
2. **Array fills through `&!` are lost** — a 4-store helper leaves the array
   all-zero.
3. **A scalar cross-module call segfaults** — when the qualified call's result
   is consumed (`println(m::add(40, 2))`), the emitted ELF dies with SIGSEGV
   (exit 139) at run time.

The named-import form `use p5_qualified_leaf::{bump, add, fill7}` compiles the
identical program correctly (`1 / 42 / 7 7 7 7`).

## Minimal repro

Checked into `artifacts/ontology-frontiers/compiler-repros/`:

- `p5_qualified_leaf.sio` — leaf module: `bump(&![i64;2])`, `add(i64,i64)->i64`,
  `fill7(&![i64;4])`.
- `p5_qualified_main.sio` — qualified-import main (faulting).
- `p5_named_control_main.sio` — named-import control (works).

Commands and observed output (2026-08-02, this worktree):

```
$ ./bin/souc run artifacts/ontology-frontiers/compiler-repros/p5_named_control_main.sio
1
42
7
7
7
7

$ ./bin/souc run artifacts/ontology-frontiers/compiler-repros/p5_qualified_main.sio
Compilation successful!
0
/workspace/sounio/bin/madaros: line 634: ... Segmentation fault      "$out" "$@"
(exit 139)
```

Isolated single-behavior variants (not checked in, reproducible in /tmp):

| Program shape (all qualified-import)            | Observed        |
|-------------------------------------------------|-----------------|
| only `m::bump(&!x)` then `println(x[0])`        | prints `0` (mutation lost), exit 0 |
| only `println(m::add(40, 2))`                   | SIGSEGV, exit 139 |
| only `m::fill7(&!y)` then print all 4 elems     | `0 0 0 0` (all stores lost) |

## Root cause (bisected compiler location)

**Primary site: `self-hosted/ir/lower.sio:15698-15717`,
`Lowerer.expr_to_callee_name_ref`.**

For an `ExprCall` whose callee is an `ExprPath` (`m::f`), the callee name is
computed as:

```
let first = ir_path_first_name(e.path)   // "m"
let last  = ir_path_last_name(e.path)    // "f"
if first != last { return ir_mangle_method_name(first, last) }  // "m_f"
```

`ir_mangle_method_name` (`self-hosted/ir/ir.sio:2250`) produces
`<first>_<last>` — the **Type::method** convention, as the code comment says.
But a *module-qualified* call `m::f` under `use m;` is not a method: imported
functions are merged into the IR module under their **plain** name `f` (this is
why the named-import form works — `lowerer_find_or_add_fn_id_mut` finds the
existing `f`).

**Amplifier: `self-hosted/ir/lower.sio:16928`,
`lowerer_find_or_add_fn_id_mut(&! lo1, callee_name)`.** Because no function
named `m_f` exists, this helper silently **creates a body-less stub** named
`m_f` and the `IrCall` is emitted against it. No diagnostic is issued.

**Merge keeps the dangling target: `self-hosted/compiler/module_frontend.sio:1689-1712`
(`ir_module_resolve_call_target_fields`) and 1716-1744
(`ir_module_resolve_all_call_targets`).** Post-merge rebinding resolves calls
by `instr.name`; `m_f` matches only the body-less stub, so the call stays bound
to the stub into codegen. A call to a body-less function explains both failure
modes: the callee does nothing (lost `&!` mutations — symptoms 1/2), and when
the call's return slot/control transfer is actually needed the emitted code
jumps into a non-body (SIGSEGV — symptom 3).

### Empirical confirmation (no compiler edits)

Prediction of the theory: the qualified call `m::f(...)` is wired to a function
literally named `m_f`. Both probes ran against the unmodified compiler:

- Defining `fn p5_qualified_leaf_bump(a: &![i64; 2]) { a[0] = 99 }` **in main**
  makes `p5_qualified_leaf::bump(&!x)` print **99** — the qualified call bound
  to the function named `p5_qualified_leaf_bump`.
- Defining `fn p5_qualified_leaf_add(a: i64, b: i64) -> i64 { return 777 }`
  makes `println(p5_qualified_leaf::add(40, 2))` print **777**.
- Defining a plain `fn add(...) -> i64 { return 222 }` in main does **not**
  capture the call — the program still segfaults, confirming the call is not
  resolved to the plain name `add`.

### Related site (same bug shape, different pipeline)

`self-hosted/hlir/lower.sio:2545-2546` takes only `hlir_ast_path_head_to_string`
(the FIRST segment, i.e. the module name) for `ExprPath` callees — also wrong
for `m::f`, worse (drops the function name entirely). The Madaros multimodule
pipeline uses `ir::lower`, not `hlir::lower`, so this site is not the cause of
the observed failures but should be fixed in the same pass.

## Candidate fix (UNAPPLIED)

`artifacts/ontology-frontiers/compiler-repros/qualified_import_fix_candidate.diff`
— one hunk in `self-hosted/ir/lower.sio` (`expr_to_callee_name_ref`):

- Compute `mangled = ir_mangle_method_name(first, last)` as today.
- **Only keep the mangled name if a function with that name exists**
  (`lowerer_lookup_fn_id_by_name_ref(&self, mangled) >= 0`) — the real
  `Type::method` case; impl methods are preseeded under mangled names before
  body lowering (`self-hosted/ir/lower.sio:1989, 2785, 2909, 3060`), so the
  lookup is populated at call-lowering time.
- Otherwise return `last` — the module-qualified case, matching how imported
  functions are actually registered.

Trade-off, stated honestly: for a program that has BOTH a module `m` exporting
`f` AND an impl method on a type `m` named `f`, the impl method wins (same as
today's behavior for genuine impls); that program is ambiguous anyway and
should be rejected by resolve. The diff deliberately does not touch
`lowerer_find_or_add_fn_id_mut` or the merge pass; a follow-up hardening worth
considering is a diagnostic when `ir_module_resolve_all_call_targets` leaves a
call bound to a body-less (`instr_count == 0`) stub — today that condition is
always silent and is the common failure amplifier for this class of bug.

## Verification status

- All symptoms reproduced on the unmodified compiler (commands above).
- Root cause confirmed by the three probes above (mangled-name binding,
  plain-name non-binding).
- The candidate diff is **not applied and not compile-tested** (compiler
  sources are under active claims by other lanes); it is a reviewed,
  mechanism-matched proposal for the owning lane to apply and validate.
