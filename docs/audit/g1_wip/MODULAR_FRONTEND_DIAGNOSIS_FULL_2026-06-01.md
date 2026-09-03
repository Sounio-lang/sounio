<!-- docs:meta
topic_id: repo.docs.audit.g1-wip.modular-frontend-diagnosis-full-2026-06-01
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.g1-wip.modular-frontend-diagnosis-full-2026-06-01
-->

All four keystones confirmed:
1. `check.sio:2278` — `_ => {}` skip with the documented CORRECTNESS GAP comment (2272-2277). Confirmed verbatim.
2. `collect_struct_def_inplace` does NOT exist — the only grep hit is the comment at 2277 referencing it as "pending." Fix = write new fn, not wire existing.
3. `check.sio:1146` — `(*c).check_expr(e)` by-value bridge in `checker_check_expr_mut`. Confirmed.
4. `check.sio:12717` — `ExprIdent` resolves via `c.env.lookup(e.name)` only. Confirmed.

The spine is sound. Writing the report now.

---

# Modular Compiler Diagnosis — Next-Session Report (2026-06-01)

Branch `modular/native-v2-e2e-gate` / worktree `/workspace/sounio-move-codegen`. All cites are into `self-hosted/check/check.sio` unless prefixed otherwise.

**One meta-cause dominates.** Nearly every failure is the same disease: the `*mut "_inplace"` checker spine is *incompletely wired*. It does one of three things on the un-migrated paths — (a) **skips work** (struct/enum/impl collection never runs), (b) **miscompiles on a by-value Checker copy** (the rc=139 crashes), or (c) **drops/misreads Checker state** across the by-value bridge (effects, return type). The migration traded the universal empty-`main` crash away for these per-feature gaps.

---

## 1. Crash Triage

The CRASH TRIAGE input bucket was `[]`, but **three rc=139 SIGSEGVs live inside the FEATURE PROBES** and must be surfaced here.

| Bucket | Probes | Count | rc |
|---|---|---|---|
| **`else`-arm by-value recursion** | `cf_ifelse.sio` (if WITHOUT else = rc=0; adding `else` = rc=139, deterministic 3/3) | 1 | 139 |
| **match-arm-with-statement by-value recursion** | `e2_match.sio` (pure-expr arm `=> {1}` = rc=0; arm containing `print(1);` = rc=139) | 1 | 139 |
| **bare-`return <expr>` large-frame instability** | `cf_ret_lit.sio` / `cf_ret_ident.sio` (`return 5;`, `return n;` — also emit spurious E008) | 1 (class) | 139 |

**Dominant root cause (single):** un-migrated expression kinds fall off the `*mut` spine back onto the **by-value Checker bridge** at `check.sio:1146` (`(*c).check_expr(e)` in `checker_check_expr_mut`). That copies the whole multi-MB Checker by value. The two concrete recursion sites:
- `check_if_expr` `Some(else_e)` arm → `c.check_expr(*else_e)` at **check.sio:15987** (the `None`/no-else arm returns `(c, ty_unit())` with no recursion → safe, explaining if-alone passing).
- `check_match_arm` → `c.check_expr(arm.body)` at **check.sio:16041** (pure-expr arms survive; a statement in the arm block re-enters the by-value `check_stmt`/`check_expr` chain).

Both routed through `checker_check_expr_inplace` (2527-2528) → `checker_check_expr_mut` (1146).

**⚠️ FORMATTER-RECURSION SURVIVOR — REGRESSION FLAG.** Memory `project_g1_codegen_largestruct_mut_2026-06-01` documents this exact crash class as the large-struct value-move miscompile (gdb-confirmed 634-deep `print_type_name` self-recursion on a corrupted cyclic `TypeEntry`). The struct-collection refactor *traded that crash away* by skipping collection. **if/else and match were never migrated**, so the same frame-disease **survives** on those by-value bridges. These rc=139s are NOT new crashes — they are the documented frame-disease persisting on the un-converted spine. Migrating ExprIf/ExprMatch onto the `*mut` spine is the fix (see ranked #2).

---

## 2. Feature Coverage Table

| Category | Status | Severity | Root cause (file:line) |
|---|---|---|---|
| **let / var / assign / compound / shadow** | ✅ WORKS (5/6; `let mut` = by-design reject) | works | dispatch check.sio:12427-12430 correct; `let mut` parser gap parser/stmts.sio:101-106 (by-design) |
| **structs (all usage)** | ❌ E015 on every literal | blocks-basic | `_ => {}` skips ItemStruct — check.sio:2278 (collector `checker_collect_item_inplace` @2267) |
| **enums — C-style decl + `E::A`** | ✅ WORKS | — | collect_enum_def 11286 + check_path_expr 12765 |
| **enums — match with statement arm** | 💥 CRASH rc=139 | blocks-basic | by-value bridge check.sio:1146 via check_match_arm 16041 |
| **enums — tuple-variant decl `Some(i64)`** | ❌ UNSUPPORTED (parse) | blocks-basic | no tuple branch in parse_enum_item — parser/items.sio:559-580 |
| **enums — struct-payload construct `E::V{..}`** | ❌ SPURIOUS E015 | blocks-basic | check_struct_lit only `self.structs.find` — check.sio:16266 (no enums fallback) |
| **arrays — literal / index / `.len()`** | ✅ WORKS | — | check_index_expr 13307; len_method_supported 8499 |
| **arrays — `.len` (bare field)** | ❌ UNSUPPORTED (by-design syntax) | — | no TyArray arm in check_field_access; use `.len()` |
| **arrays — `&a[0..2]` slice** | ❌ SPURIOUS E014 | blocks-advanced | dispatch guard tests wrong nesting — check.sio:13140 (orphans handler 13067) |
| **arrays — array-in-struct** | ❌ E015 (not array-specific) | blocks-advanced | struct collection gap check.sio:2278 |
| **control — `if` (no else)** | ✅ WORKS | — | None arm check.sio:15997 |
| **control — `if/else`** | 💥 CRASH rc=139 | blocks-basic | else-arm by-value recursion check.sio:15987 via 1146 |
| **control — while / for-in / match-expr** | ✅ WORKS (bodies genuinely checked) | — | check_while/for/match wired |
| **control — `loop` + `break`** | ❌ UNSUPPORTED (parse / binary skew) | blocks-basic | source OK (parser/exprs.sio:790,860); mc.elf parse fails — build skew |
| **control — explicit `return <expr>;`** | ❌ SPURIOUS E008 | blocks-basic | cached current_return_type reads `()` — check.sio:2489-2492 (set @2387) |
| **methods/impl — all (self, assoc fn, call)** | ❌ UNSUPPORTED (parse / binary skew) | blocks-basic | source OK (parser/items.sio:414-427, 267); mc.elf rejects `self`/impl skeleton — build skew |
| **types — `&x` shared, `&!x` mutable, prims, Option, Box** | ✅ WORKS | — | builtins independent of user-struct path |
| **types — `&mut`** | ❌ UNSUPPORTED (by-design; use `&!`) | — | lean_single.sio:3745 help text |
| **types — generic struct use `Foo<T>`** | ❌ SPURIOUS E015 (not generics-specific) | blocks-basic | struct collection gap check.sio:2278; check_generic_struct_lit 16272 unreachable |
| **types — `type X = Y` alias** | ❌ UNSUPPORTED (parse / binary skew) | blocks-basic | source OK (parser/items.sio:777); mc.elf parse fails — build skew; + collect_type_alias 9917 unwired |
| **effects — effect-row parse, `with IO,Mut`** | ✅ WORKS | — | parse_effect_list types.sio:565 |
| **effects — Knowledge<T> type, `.epsilon`, `.unwrap("r")`, `measure()`, `acknowledge()`** | ✅ WORKS | — | checker_lower_knowledge_type_mut 930; check_measure_expr 15533 |
| **effects — `.value` under `with Epistemic`** | ❌ SPURIOUS E170 | blocks-basic | current_effects empty at gate — check.sio:13198 (dropped across by-value bridge 1146) |
| **effects — `Knowledge(..)` constructor** | ❌ UNSUPPORTED (silent rc=0 laundering) | blocks-basic | no ctor handler; ExprIdent→env.lookup only → ty_unknown — check.sio:12717 / falls to 13643 |
| **effects — effect/arg-count/arg-type discipline on free-fn calls** | ❌ UNSUPPORTED (silent rc=0, DEAD) | blocks-basic | check_callee_effects only in TyFn branch 13634; free-fn callee never TyFn — check.sio:12717 |

---

## 3. Ranked Next-Fix List (highest leverage first)

**FIX #0 (do BEFORE touching the parser) — Rebuild mc.elf from current source and re-probe.**
Four "unsupported" rows — `loop`+`break`, `self` receiver, `impl` skeleton, `type X = Y` — are attributed to **mc.elf diverging from correct source** (the source paths exist and are wired: parser/exprs.sio:790/860, parser/items.sio:414-427/267/777; lexer tables present). Editing correct source would be wasted work. One rebuild may clear all four rows. *Cheap, must precede any parser edits.*

**FIX #1 — Implement `collect_struct_def_inplace` (+ enum/impl/type-alias siblings) and wire into `checker_collect_item_inplace`.**
`check.sio:2267` (replace the `_ => {}` at **2278**). Model on the working by-value `collect_struct_def` (**11225**, which does `c.structs = c.structs.add(info)` at **11247**) and `collect_type_alias` (**9917**), rewritten to mutate through `c: *mut Checker` instead of returning a copied Checker (that copy is exactly the 8MB SRET frame that was skipped).
*Unblocks the single largest set:* structs (all 6 probes), array-in-struct, generic struct use, struct-payload enum construct, methods-impl end-to-end, type-alias checker leg. **Keystone — confirmed `collect_struct_def_inplace` does not yet exist.**

**FIX #2 — Migrate ExprIf and ExprMatch onto the `*mut` spine.**
`checker_check_expr_inplace` dispatch at **check.sio:2527-2528**; today they fall through to the by-value bridge **1146** → recurse by value at `check_if_expr` else-arm **15987** and `check_match_arm` **16041**. Add inline `*mut` handlers that recurse via `checker_check_expr_mut`/`checker_check_expr_inplace` instead of `c.check_expr(...)`.
*Kills both rc=139 survivors* (`if/else`, statement-in-match-arm) — the formatter-recursion regression.

**FIX #3 — Resolve free-function callees through `fn_sigs`, not just `env.lookup`.**
`ExprIdent` arm at **check.sio:12717** (`c.env.lookup(e.name)` only). When lookup yields unknown, fall back to `fn_sigs.get(name)` and produce a `TyFn` so the call lands in the real `check_call_expr` path instead of the `is_error_or_unknown` bypass at **13643**.
*Revives an entire safety class at once:* `check_callee_effects` (**13634**), arg-count, and arg-type checking — all currently silently DEAD. Also surfaces the Knowledge-ctor laundering (prereq for #5).

**FIX #4 — Fix explicit-return type read (spurious E008).**
`checker_check_return_expr_inplace` reads cached `(*c).current_return_type` at **check.sio:2489-2492**, which reads as `()` though set at **2387** from `sig.return_type`. Either re-read fresh from `fn_sigs.get()` like the implicit-tail path does at **2400**, or fix the `*mut` store of the large-`TypeEntry` value at 2387. (Dev comment 2377-2383 flags this exact path unvalidated.)
*Unblocks every helper fn using `return <expr>;`.*

**FIX #5 — Add a `Knowledge(..)` constructor handler in `check_call_expr`.**
`check.sio:13494` (depends on #3 so the callee no longer degrades to unknown). Produce `TyKnowledge` from `Knowledge(value, epsilon)`. Removes the silent rc=0 laundering where `let k: i64 = Knowledge(0.5,0.1)` is wrongly accepted.

**FIX #6 — Fix slice-borrow dispatch guard (spurious E014).**
`check_unary_expr` guard at **check.sio:13140** currently tests `expr_is_half_open_range(e.left)` (true only when `&`-operand is itself an ExprRange — impossible). Change to "`e.left` is an ExprIndex whose `.right` is a half-open range." The handler `check_slice_borrow_expr` (**13067**) already reads `idx_expr.right` correctly; it is merely orphaned.

**FIX #7 — Add `current_effects` survival across the by-value bridge (spurious E170).**
`.value` gate at **check.sio:13198** reads empty `current_effects` even with `with Epistemic`. Most-likely mechanism: the by-value materialization of `*c` into `check_expr`'s `self` at **1146** drops the fixed-size `[i64;8] current_effects` array. Verify whether `checker_extract_effects_mut` landed id 8 into `sig.effects` (effects.sio:204 / check.sio:2101) before assuming the drop is in the copy. (Largely subsumed once `.value`/field-access moves onto the `*mut` spine, à la #2.)

**FIX #8 — Add tuple-variant declaration to `parse_enum_item`.**
`parser/items.sio:559-580` parses only struct-style `Variant { f: T }`; add a `(` branch for `Variant(Type)`. The pattern parser already supports tuple `Some(x)` (parser/patterns.sio:265-295), so this closes the decl/use asymmetry.

---

## 4. Spurious-Error vs Unsupported vs By-Design

### A. SPURIOUS-ERROR — valid program WRONGLY REJECTED (real bugs; these mislead users into thinking their correct code is wrong)
- **All struct literal use** → E015 (`P { x: 5 }`, nested, arg, return, array-in-struct, generic) — check.sio:2278 collection skip. *(Fix #1)*
- **Generic struct use** `Foo<i64>` → E015 (same collection gap, not generics-specific). *(Fix #1)*
- **Struct-payload enum construct** `E::V{..}` → E015 (no enums fallback in check_struct_lit) — check.sio:16266. *(Fix #1/#8 area)*
- **`if/else`** → rc=139 crash on valid program — check.sio:15987. *(Fix #2)*
- **match with a statement in an arm** → rc=139 crash — check.sio:16041. *(Fix #2)*
- **explicit `return <expr>;`** → E008 (declared `-> i64` read as `()`) — check.sio:2489-2492. *(Fix #4)*
- **`.value` under `with Epistemic`** → E170 — check.sio:13198. *(Fix #7)*
- **`&a[0..2]` slice borrow** → E014 — check.sio:13140 guard. *(Fix #6)*

### B. UNSUPPORTED — not implemented / dead (no error fires, OR fails before reaching the checker)
- **`Knowledge(..)` constructor** — silent rc=0 laundering, no TyKnowledge produced — check.sio:12717/13643. *(Fix #5)*
- **Effect + arg-count + arg-type discipline on free-fn calls** — entirely non-firing, every free-fn call unchecked — check.sio:12717/13634. *(Fix #3)*
- **tuple-variant decl** `Some(i64)` — parse error, never reaches checker — parser/items.sio:559-580. *(Fix #8)*
- **`loop` + `break`** — parse fail in mc.elf (source is correct → likely build skew). *(Fix #0 first)*
- **methods/impl** (`self` receiver, impl skeleton, assoc fn) — parse fail in mc.elf (source correct → likely build skew). *(Fix #0 first)*
- **`type X = Y` alias** — parse fail in mc.elf (source correct → build skew) + latent collect_type_alias 9917 unwired. *(Fix #0 first, then #1)*

### C. BY-DESIGN — NOT A BUG. Do NOT "fix" these; keep them OUT of the ranked list.
- **`let mut x`** — Sounio uses `var` for mutable. Toolchain-wide explicit rejection (E003 help text, lsp quick-fix server.sio:3634, lean_single.sio:3736/18238, test_bootstrap.sio:70). Only wart: confusing generic parse-error message vs the legacy friendly one (cosmetic).
- **`&mut T`** — Sounio mutable ref is `&!` (AmpBang), which works. `&mut` rejected with help "change `&mut T` to `&! T`" (lean_single.sio:3745).
- **bare `.len`** — array length is the `.len()` *method* (works). Bare field form is not a Sounio construct.
- **bare `x.unwrap()`** — Knowledge.unwrap is reason-carrying by design; requires one string-literal arg. `unwrap("reason")` works; bare form correctly rejected E172.
- **`with { }` block expression** — not in the grammar; only the fn-signature effect row is supported. Out-of-scope extension.

---

**Bottom line for the next session:** run FIX #0 (rebuild + re-probe) to potentially clear 4 unsupported rows for free, then land FIX #1 (`collect_*_inplace`) which alone unblocks structs/generics/enums-payload/methods-impl/array-in-struct, then FIX #2 to kill the two surviving frame-disease crashes. Those three account for the majority of "blocks-basic" failures. #3 is the highest-value correctness fix (an entire effect/arg safety class is silently off).
