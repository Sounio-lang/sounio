<!-- docs:meta
topic_id: repo.docs.audit.g1-wip.enum-collector-readpath-blocker-2026-06-01
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.g1-wip.enum-collector-readpath-blocker-2026-06-01
-->

# Enum collector — body FIXED, wiring gated on a downstream codegen bug (2026-06-01)

## TL;DR

- **The enum-collector *body* crash is fixed.** `checker_collect_enum_def_inplace` +
  the variant collectors now run to completion on every enum shape (fieldless,
  fielded, multi-variant) — verified by stage-marker tracing reaching the final
  `(*c).enums.add(info)` with `rc=0`.
- **The `ItemEnum` arm stays DISABLED** in `checker_collect_item_inplace`. Wiring it
  is net-negative on today's binary: **0 corpus wins, 1 regression**.
- The regression is a **separate, pre-existing `bin/souc` codegen bug** in the
  by-value enum-READ path, not a collector defect. gdb-confirmed return-address smash.
- Enabling enum collection is therefore **gated on FIX #2** (migrate the enum-read off
  by-value `self`-threading onto the `*mut` spine).

## What landed (the bankable fix)

`checker_collect_variant_list_mut` previously returned a `MutVariantListResult`, which
carries a `VariantInfoList` (containing a `VariantInfo`, ~1 KB) **by value**. Returning
that aggregate by SRET out of a `*mut` fn miscompiled under `bin/souc` and SIGSEGV'd
right after the deepest variant built its result (localized via stage-marker tracing:
the trace reached `VL_NONE` then died before the return).

Fix: the variant collectors now return the small `MutVariantsResult`
(`{Option<Box<VariantInfoList>>, count}` = pointer + int). Each node is built into a
local then boxed (`let node = …; Some(Box::new(node))`), mirroring the struct path's
known-good `Box::new(local)`. No large aggregate is ever returned by value.
`MutVariantListResult` is now unused and was removed.

This corrects **dormant** code (the collector body is only reached once `ItemEnum` is
wired), so FIX #2 starts from "collector works" instead of "collector crashes."

## The downstream blocker (why ItemEnum stays disabled)

With the collector enabled, collection completes (`DBG_ENUM_COLLECTED_OK` prints), but
programs that *use* an enum as a value path-expr crash in the type-check phase. The
single corpus regression is `examples/native/enum_match.sio` (`Color::Green` etc.);
the synthetic `let x = E::A` reproduces it too.

### Localization (stage-marker tracing + gdb)

The crash is in `check_path_expr` (`self-hosted/check/check.sio`), the
`if enum_idx >= 0` branch — i.e. the path taken **only when the enum table is
non-empty**. At baseline (empty table) this branch is never taken; `find` always
misses and the program falls through to the lenient `env.lookup` path (rc=0). That is
why the bug was dormant until collection populated the table.

- `EnumTable.find` works: it prints `count=1`, hits at i=0 — the collected table is
  intact through the by-value bridge (no corruption of the heap output).
- The crash is **independent of the return value**: returning `ty_unknown()` instead of
  `ty_named(first_name)` from that branch still crashes.
- The crash is **independent of stack size**: rc=139 at `ulimit -s` 64 MB, 256 MB,
  and 1 GB — so it is NOT stack overflow.

### gdb fault classification

```
Program received signal SIGSEGV
rip = 0x7fffeac50008   (a STACK address, ~0x1000 below rbp = 0x7fffeac51008)
=> 0x7fffeac50008:  add %al,(%rax)        # executing zeroed stack bytes
```

`rip` jumped into the stack region and is executing garbage → a **corrupted saved
return address**. A miscompiled large-struct copy wrote out of bounds and clobbered the
return address; the `ret` jumped into stack data. `check_path_expr` returns
`(Checker, TypeEntry)` — the `Checker` is ~8 MB — so the SRET return of `self` out of
the enum-found branch is the large-struct move that smashes the stack.

This is the **same family** as the documented if/while / largestruct `bin/souc`
miscompile (large-struct VALUE MOVEMENT + control flow; sharpest historical trigger
`if 1{}`). The branch-correlation (then-branch crashes, else-branch identical-shape
return is fine) is the *signature* of a control-flow-dependent codegen bug: the
if-true and if-false epilogues generate different code and one is buggy.

### Source-level dodges that FAILED (do not retry these)

1. **if-chain dispatch** in `checker_collect_item_inplace` (the `0afad182c` match-dodge):
   fixed the *dispatch* (struct/control unaffected) but the crash is in the collector
   body / read path, not the dispatch.
2. **single tail return** in `check_path_expr` (collapse the multiple `(self, …)`
   return sites into one): still crashes — it is not a multi-return-site artifact.
3. **`ty_unknown()` return** from the enum-found branch: still crashes — the fault is
   not the returned value.

All three were verified by full rebuild + run. The fault is in `bin/souc`'s codegen of
the 8 MB `self` return under that control flow, not in any source shape reachable by
restructuring `check_path_expr`.

## FIX #2 (the actual unblock)

Migrate the enum-read path off by-value `self`-threading. `check_path_expr` is a pure
read (it never mutates `self` — it only reads `self.enums` and returns `self`
unchanged). Returning the 8 MB `Checker` by value is what trips the miscompile.

Options, in order of preference:
1. **`*mut` migration**: give the enum-read path a `*mut Checker` receiver that returns
   only the `TypeEntry` (no `self` move), matching the move-codegen spine. This is the
   principled FIX #2 and removes the large-struct return entirely.
2. **bin/souc codegen fix**: pin and fix the large-struct-move + control-flow miscompile
   in `lean_single.sio`. HIGH RISK — requires re-bootstrap, and a prior unpinned
   threshold-guess edit already failed. Do NOT attempt without a pinned line.

### Forward note (scope FIX #2 on correctness, not corpus count)

The sweep showed **0 crash→pass wins** because the modular `--check` is lenient on
enums (it passes `let x = E::A`, `match`, and even a duplicate `enum E` at rc=0). So
FIX #2's reader will not flip crash→pass either; its payoff is **catching real enum
type errors**, not corpus count. Scope FIX #2 success on correctness.

## Verification provenance

- Builds are deterministic: a from-source rebuild of the committed tree reproduced the
  committed `.dbg/mc.elf` md5 `33d9ee39…` exactly. Build time ≈ 155 s.
- Collector-completes proof: stage markers reached `DBG_ENUM_COLLECTED_OK` (rc=0) on
  `enum_decl_only`, fielded-variant, and `match` enum programs (binary `mc_enum_fix`).
- Regression measurement (847 `examples/`): enabling collection = exactly 1 transition
  `0→139` (`enum_match.sio`), 0 pass→fail, 0 other crashes (binary `mc_enum_fix` vs the
  md5-verified baseline).
- gdb fault classification on `mc_diag` (enum-found→`ty_unknown` variant): rc=139 at
  all stack sizes; `rip` in the stack region (return-address smash).

The committed change keeps `ItemEnum` disabled, so the shipped binary is net-neutral
vs baseline — re-verified by full sweep + `g1_expr_recursion_gate` on the marker-free
build.

## FIX #2 attempt (2026-06-01): in-place ExprPath handler — necessary but NOT sufficient

Tried the advisor-recommended `*mut` migration of the enum-read: a `*mut` ExprPath leaf
handler that returns only the `TypeEntry` (no 8MB `self` move), wired into the `*mut`
expr dispatch, with `ItemEnum` re-enabled. Built + swept.

**Result: top-level path-exprs fixed, nested ones still crash.**
- `let x = E::A` (synthetic `probe_construct`) → **rc=0** (was crash). The `*mut` spine
  dispatches this ExprPath directly to the in-place handler. Fixed.
- `examples/native/enum_match.sio` (`if c == Color::Red`) → **still rc=139**. Its
  `Color::Red` is nested inside a binary `==` inside an `if`. Both `ExprIf` and
  `ExprBinary` are non-leaf → they bridge to by-value `check_expr`, which checks the
  nested `Color::Red` via the **by-value** `check_path_expr` (the source-undodgeable one)
  — NOT the `*mut` handler. So the in-place handler never sees nested path-exprs.
- Full 847 sweep with the handler + `ItemEnum` on: still **exactly 1 regression**
  (`enum_match` 0→139), **0 wins**.

**Conclusion — FIX #2 is spine-completion, not a bounded fix.** Clearing the nested case
requires `*mut` handlers for `ExprIf` AND `ExprBinary`; `ExprBinary` transitively pulls
in op-typing / units / knowledge checking, each of which threads `self` by value and
bridges again. The terminal state is "migrate the entire expression checker off by-value
`self`-threading" — i.e. complete the half-converted `*mut` spine. With 0 structural
corpus wins (the checker is lenient on enums), this is unbounded work for near-zero
corpus payoff, and must be done as its own focused session (per the rule: do NOT chain a
half-converted-spine migration on low context — it bricks mc.elf).

## Spine-completion session (2026-06-01): Build A LANDED (80 crash rescues); ItemEnum now gated on a SEMANTIC blocker

Did the spine-completion as its own session. Outcome split cleanly:

**Build A — LANDED (`cb92d66a9`).** Added `*mut` leaf handlers for `ExprPath` and
`ExprBinary` (split-and-bridge: check operands via the `*mut` spine, bridge the op-typing
tail `check_binary_with_operand_types` — a verbatim extraction of `check_binary_expr`).
With `ItemEnum` OFF, full 847 sweep vs baseline: **0 regressions** (every previously
completing program byte-identical → operand-routing faithful) and **80 crashes RESCUED**
(481→401 rc=139): 3 now type-check clean (incl. `struct_param.sio`, the FIX#1
struct-as-param SIGSEGV limit), 77 now run to completion and report errors gracefully.
So routing operand-checking off the by-value frame path heals frame-disease crashes
corpus-wide — a real, large win independent of enums. g1 gate PASS.

**ItemEnum enablement — DISCARDED; now gated on a SEMANTIC (not codegen) blocker.**
Enabling `ItemEnum` on top of Build A changed **exactly one** corpus program: `enum_match`
(the only one with a nested enum comparison). It still crashed (its `Color::Red` is nested
in `if c == Color::Red` → bridges to by-value `check_if_expr` → by-value `check_path_expr`
→ crash; ExprIf not migrated). The GO/NO-GO probe `enum E{A,B} fn f(c:i64){ let x = c==E::A }`
(top-level binary, no `if`) → **rc=1, NO crash** — proving the `*mut` binary path itself
works; the bridged tail ran and typed `i64 == E::A` as a mismatch (E004).

But that rc=1 is a **divergence from canonical, not a correct rejection**: canonical
`bin/souc` COMPILES both `enum_match.sio` and the probe successfully (`i64 == fieldless-enum`
is valid in real Sounio — C-style int-like variants). The modular checker rejects it
because `check_path_expr` types `Color::Red` as `ty_named("Color")` — it uses only the HEAD
path segment (`checker_copy_string_list_to_name` copies `seg.head` only), so the whole path
is typed as the enum, not an int. The lenient empty-table baseline masked this (uncollected
enum → `Color::Red` typed as unknown → no mismatch).

⇒ **Migrating ExprIf would only convert `enum_match` from crash (139) to a divergent type-
error (1)** — trading a crash for a wrong answer, while opening the broad ExprIf regression
surface. Reaching the canonical-correct rc=0 needs a SEMANTIC fix to enum-path typing
(fieldless variants int-like, or relax enum/int compare), which is unscoped, unmeasured,
and has its own corpus-wide blast radius. That is a TYPING project, not codegen — a fresh
scoping problem (how should fieldless enums type? what does canonical do for enum/int
compare across the corpus?). **ItemEnum stays disabled; full enable = ExprIf migration +
enum-path/compare semantics matching canonical, as a separate session.**

The in-place `check_path_expr` handler IS now landed (in Build A, `checker_check_path_expr_inplace`);
the FIRST step below is therefore done. Remaining for full enable: the semantic typing fix
+ `ExprIf`/transitive `*mut` handlers. The in-place handler is preserved verbatim below:

```sounio
// wire into checker_check_expr_inplace's if-chain, before the `else` bridge:
//   } else if e.kind == ExprKind::ExprPath {
//       result = checker_check_path_expr_inplace(c, e)
//   } else { result = checker_check_expr_mut(c, e) }

// Faithful in-place transcription of by-value check_path_expr (a PURE READ: returns
// `self` unchanged). Returns only the TypeEntry — no 8MB self SRET — so it dodges the
// read-path return-address smash. Verified: fixes `let x = E::A` (rc=0).
fn checker_check_path_expr_inplace(c: *mut Checker, e: Expr) -> TypeEntry with Mut, Panic, Div, Alloc, IO {
    let first_name = checker_copy_string_list_to_name(e.path.segments)
    let enum_idx = (*c).enums.find(first_name)
    if enum_idx >= 0 {
        ty_named(first_name)
    } else {
        let variant_enum_idx = (*c).enums.find_variant_enum(first_name)
        if variant_enum_idx >= 0 {
            let ei = (*c).enums.get(variant_enum_idx)
            ty_named(ei.name)
        } else {
            (*c).env.lookup(first_name)
        }
    }
}
```
