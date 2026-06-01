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
