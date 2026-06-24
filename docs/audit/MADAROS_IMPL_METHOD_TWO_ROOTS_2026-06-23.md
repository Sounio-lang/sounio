<!-- docs:meta
topic_id: repo.docs.audit.madaros-impl-method-two-roots-2026-06-23
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-impl-method-two-roots-2026-06-23
-->

# Madaros impl/method codegen — triage: two roots, both large-aggregate-by-value (2026-06-23)

*Investigation off `main` (madaros-fix = main + struct-field fix #413 + Codex machine_ir fix).*
*Status:* TRIAGED (not fixed). Enum construction is already fixed; method/`impl` has two roots.

## Headline

`madaros compile`/`--check` of a program with an `impl` method **crashes the compiler
(exit 139)** at compile time. Enum construction + `match` now **works** (was a separate
hole, fixed by the enum-variant registration fix `62f2b3a28`). So the next root is
**method/`impl`**, and it is **two distinct bugs**, both instances of the systemic madaros
theme: **large aggregates passed/returned by value**.

## Decisive isolation (control = `ulimit -s unlimited`)

| Program | default stack | `ulimit -s unlimited` | Phase |
|---|---|---|---|
| `impl C {}` (empty) | SIGSEGV | **compiles (0)** | lowering |
| `impl C { fn get(&self)->i64 {…} }` | SIGSEGV | **still SIGSEGV** | `--check` (frontend) |
| `impl C { fn make()->i64 {7} }` (static, no self) | SIGSEGV | **still SIGSEGV** | `--check` |
| free `fn make()->i64 {7}` (no impl) | — | compiles + runs (7) | — |
| `--check impl_empty` | 0 (OK) | 0 (OK) | — |

So: the empty impl crash is a **stack overflow** (ulimit fixes it); adding **any** method
(even a static one that ignores `self`) makes `--check` itself crash **even under ulimit**
— a genuine miscompile, in the **checker**, specific to methods-in-impl.

## ROOT 1 — impl-block lowering STACK OVERFLOW — **ALREADY MITIGATED IN PRODUCTION**

**Correction (empirically resolved): ROOT 1 is not a live production blocker and needs no
code fix.** `bin/madaros` (the production launcher) already sets **`ulimit -s unlimited`**
(line 58), and through the wrapper an empty/any-impl program **compiles (exit 0)**. The
exit-139 in the isolation table above was an **artifact of invoking the raw ELF
`artifacts/self-hosted/madaros-*` directly**, bypassing the wrapper. Methods still crash
*through the wrapper* (139) — that is ROOT 2, below, not stack.

This is **systemic, not one function** (my earlier "contained seed-by-value" hypothesis was
wrong: the `*_seed_owned` / `*_owned` by-value families it named are **dead code**; the live
preseed path `ir_fast_summary_preseed_impl_methods`/`_items` is already `&! IrModule` +
iterative). A direct scan of the madaros binary finds **115 functions with stack frames
> 4 MB** (several > 8 MB, up to ~19 MB), from pervasive by-value large-aggregate locals/
params (e.g. only 4 are full `var module = *lo.module` copies; the rest are tables, `Checker`,
`IrFunction` arrays, etc.). The `ulimit -s unlimited` launcher is the correct, standard
mitigation for all of them at once.

**Actionable residue (not blocking):**
1. Any path that invokes the **raw** madaros ELF without the wrapper (a future
   Madaros-based test harness pointing `SOUNIO_TEST_SOUC_BIN` at the raw binary) must set
   `ulimit -s unlimited` (or invoke via `bin/madaros`). This is a harness-config item for the
   Madaros-official migration, **not a compiler change**.
2. *Optional robustness:* slim the worst-offender frames (convert `var module = *lo.module`
   copy-modify-writeback to single-level box-pointer stores `var m=lo.module; (*m).f[i]=…`,
   the enum/struct-fix idiom) so the compiler doesn't *depend* on unlimited stack. Latent
   fragility, not a live bug.

## ROOT 2 — checker miscompile on impl methods (`Checker` by value)

`--check` of an impl **with a method** crashes even under ulimit → a by-value-aggregate
**miscompile** (not stack overflow), in the checker's method-collection path:
`check_impl_item` (`check.sio:16516`), `collect_impl_methods` (`16293`), `collect_impl_method`
(`16303`) — all `self: Checker -> Checker` returning the **large `Checker` struct by value**.
The general functional `Checker -> Checker` style works for free fns / structs / enums; it
crashes **only** for methods-in-impl, so a specific by-value store/return in that path is
hit by the same miscompile class as the struct-layout and `machine_ir` bugs. The existing
`checker_check_impl_item_bridge` (`check.sio:3215`, "the by-value `check_impl_item` bridge")
shows this was already a known trouble spot, isolated but not fully de-by-valued.

**Fix (needs localisation):** convert the impl-method collection to the `*mut Checker`
in-place pattern already used at the top-level bridge, eliminating the by-value `Checker`
returns inside the per-method loop. The exact miscompiled store should be pinned the same
way the struct bug was (trace + observe), as it may be a single two-level/by-value store.

## Both are needed for method calls in production

Production runs without `ulimit`. A method-calling program hits ROOT 2 first (checker) then
ROOT 1 (lowering); both must be fixed. They are the **same systemic root** as the two
already-fixed bugs — large aggregates by value — suggesting the path to Madaros-official is
a **systematic sweep**: convert hot large-aggregate-by-value paths (`IrFastSummaryOwnedSeed`,
`Checker`, `IrModule`, normalize.sio `[IrFunction; N]`) to by-reference / in-place, and add a
loud compile-time guard for the field-index hash fallback so future drops fail loudly.

## Recommendation

ROOT 1 first (contained, mechanical, unblocks the empty-impl/lowering overflow), then ROOT 2
(localise the checker by-value store, then de-by-value the method-collection loop). Each is a
separate PR with the repros above as the gate. Enum construction needs no work.

## AI disclosure
Triage by AI agent (Claude) under human direction; every claim backed by the re-runnable
`madaros compile`/`--check` commands and `ulimit` control above.
