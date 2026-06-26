<!-- docs:meta
topic_id: repo.docs.audit.madaros-selfhost-typeenv-sret-2026-06-25
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-selfhost-typeenv-sret-2026-06-25
-->

# Madaros self-host typecheck blocker — `TypeEnv` by-value SRET mega-copy (2026-06-25)

*Working diagnosis for the dominant remaining self-host typecheck blocker (E137→E017/E035
cascade, prior session ~1,958 errors). Branch `fix/checker-fnsig-table-heap-indirect`,
clean tip `090767311`. Builds on the FnSigTable/Struct/Enum heap-indirection landed on this
branch. Adversarial about its own claim — the build reconfirmation gates it.*

## 1. Symptom (inherited, being reconfirmed on a clean build)

Madaros checking `self-hosted/compiler/main.sio` emits a large error count dominated by
**E137** ("cannot be called"), with **E017** and **E035** cascading from it: an unresolved
local lowers to a `ty_error` callee, which then reports "cannot be called". Prior session's
empirical narrowing across nine fix-attempts: **scope-0 bindings work; scope-1+ bindings are
lost.** Mechanism was not cracked.

## 2. The structural asymmetry

`Checker.env: TypeEnv` is a **by-value field** (`check.sio:174`), unlike `structs`/`enums`/
`fn_sigs` which this branch made `*mut` (heap-indirect, shared). But the spine is already
`*mut Checker` (178 fns vs 2 by-value), so the *sharing* problem the heap-indirection solved
for the tables does **not** apply to env — there is one `Checker`, hence one `env`.

The real cost is different. `TypeEnv` is enormous:

```
TypeEnv { bindings: [TypeBinding; 4096], count, scope_starts:[i64;64], scope_depth }
TypeBinding { name: Name{buf:[i8;128],len}, ty: TypeEntry(large), is_mutable, scope_depth }
```

`TypeBinding` ≳ 136 B (name) + sizeof(TypeEntry) + 16 ≈ several hundred bytes; ×4096 ⇒
**`TypeEnv` is multiple MB**, embedded by value in `Checker`.

## 3. The suspected miscompile site

`bind`/`push_scope`/`pop_scope` are `impl TypeEnv` methods with signature `(self) -> TypeEnv`
(`env.sio:60-90`): they take the multi-MB `TypeEnv` **by value** and **return it by value**
(SRET). `push_scope` is the cleanest suspect:

```sounio
fn push_scope(self) -> TypeEnv with Mut, Panic, Div {
    var env = self                          // COPY all 4096 bindings (multi-MB, SRET arg)
    if env.scope_depth < 64 {
        env.scope_starts[env.scope_depth] = env.count
        env.scope_depth = env.scope_depth + 1
    }
    env                                     // RETURN by value (SRET)
}
```

Every block / function-body entry calls `(*c).env = (*c).env.push_scope()`, i.e. a full
multi-MB by-value `TypeEnv` copy. **By-value large-struct copy/return (SRET) is the systemic
Madaros miscompile class** recorded across many prior sessions. If that copy drops bindings
above some array offset K, then entering the *first* nested scope corrupts the outer env —
**scope-0 works, scope-1+ loses bindings**, exactly the observed signature. The lost binding's
`lookup` returns `ty_error()` (`env.sio:100`) → callee `ty_error` → **E137** → E017/E035.

Why the frontend still typechecks small programs: small `count` ⇒ the miscompiled copy stays
below offset K ⇒ no loss. main.sio's huge functions push `count` past K.

## 4. Candidate fix — in-place `*mut TypeEnv` mutators (no data-model change)

The in-place primitive already exists: `type_env_bind_mut(env: *mut TypeEnv, …)` (`env.sio:46`).
Mirror it for scope ops and replace the by-value method calls:

1. Add `type_env_push_scope_mut(env: *mut TypeEnv)` / `type_env_pop_scope_mut(env: *mut TypeEnv)`
   — mutate `scope_starts`/`scope_depth`/`count` **in place**, no copy.
2. Rewrite the 16 `*mut`-spine sites `(*c).env = (*c).env.bind|push_scope|pop_scope(…)` →
   `type_env_{bind,push_scope,pop_scope}_mut(&!(*c).env, …)`. Kills the mega-copy entirely.
3. The 33 by-value-spine sites (`c.env = c.env.…`, in `self: Checker -> Checker` methods that
   copy the whole `Checker` anyway) are a **separate, larger** exposure — type-alias / policy /
   model decl handlers, rarely hit when checking the compiler. Convert opportunistically or
   leave as a follow-up; they are not the high-frequency path.

`env` stays a by-value field (one per Checker) — no `*mut TypeEnv` field, no sharing-semantics
change, scope discipline identical (push records `scope_start`, pop truncates `count`).

## 5. Call-site inventory (`self-hosted/check/check.sio`)

| Form | Count | Spine |
|---|---:|---|
| `(*c).env = (*c).env.{bind,push,pop}` | 16 | `*mut` (high-frequency — fix first) |
| `c.env = c.env.{bind,push,pop}` | 33 | by-value `impl Checker` (decls — defer) |
| method totals | 38 bind / 12 push / 12 pop | |

`cur_env` snapshots at 6410 / 16828 verified harmless (snapshot immediately precedes use).

## 6. Reconfirmation gate (in progress)

Building madaros on clean worktree off `090767311`, then `madaros check
self-hosted/compiler/main.sio`. Must confirm: (a) error count + E137/E017/E035 dominance,
(b) witnesses are lost **locals** (not free fns), (c) loss correlates with env array-offset /
scope-depth — the discriminating test between this SRET hypothesis and a scope-logic bug.

## 7. ⛔ SRET-mega-copy mechanism REFUTED (2026-06-25, direct seed repro)

Tested the §3 hypothesis directly against the seed `bin/souc-lean-single-x86_64` (the exact
compiler that builds madaros) with three by-value copy repros of increasing fidelity:

| Repro | Element | Array | Total copy | Scope churn | Result |
|---|---|---|---:|---|---|
| `/tmp/sret_repro.sio` | `i64` | 4096 | 32 KB | 1 bump | **lost=0** |
| `/tmp/env_repro.sio` | nested `[i8;128]`+i64s | 4096 (count 1000) | ~145 KB | 1 push+bind | **lost=0** |
| `/tmp/env_repro2.sio` | ~408 B (2×`[i8;128]`+`[i64;16]`) | 4096 (count 3500) | **~1.4 MB** | 10× push/bind/pop + final push/bind | **lost=0** |

All three: by-value `push_scope/pop_scope/bind`-shaped `(self: Env) -> Env` copies preserve
**every** binding faithfully across the SRET return, at sizes matching the real `TypeEnv`
(`TypeBinding` ≈ 408 B × 4096 ≈ 1.67 MB). **The seed does not miscompile the large by-value
`TypeEnv` copy.** §3's mechanism is wrong; the §4 in-place-mutator fix would NOT clear the
blocker. (In-place mutators may still be worth it as a *perf* win — 1.4 MB copied per binding
— but that is not the correctness blocker.)

**Consequence:** the inherited "scope-1+ lost, scope-0 works" framing is not explained by a
copy miscompile, and may itself be a mischaracterisation. Do not re-chase SRET. Await the real
`madaros check main.sio` witnesses (§6) before forming the next hypothesis — the witnesses
(exact unresolved identifiers + whether they are locals vs free fns vs imports) redirect this.

## 8. ⛔ Undersized-allocation mechanism also DISMISSED (2026-06-25, static)

Second candidate: a `heap_alloc(N) as *mut Checker` where `N < sizeof(Checker)` would let
high-offset in-place binding writes scribble past the buffer (loss grows with `count` ⇒
correlates with scope depth). Checked every `as *mut Checker` site:

- Production multi-module import-check path (`check_modules_verdict_boot4` etc.) routes **all**
  `*mut Checker` allocations through the single `check_mod_alloc_checker()` (`mod.sio:62`) =
  `heap_alloc(134217728)` = **128 MB**, with an explicit maintainer comment that it is kept
  "ahead of the struct size so in-place checker paths cannot scribble past a stale literal
  allocation." 14 call sites, all via this allocator.
- Only non-production undersized sites: `ontology_validation_debug_driver.sio` (256 KB) and
  probes — not on the self-host check path. Tests use 8 MB.

`sizeof(Checker)` is well under 128 MB (largest field `env` ≈ 1.67 MB; the big tables were
moved to `*mut` on this branch). So the allocation is **not** undersized on the production
path. Dismissed. (Still verify the exact `sizeof(Checker)` once measured, but 128 MB has wide
margin.)

**Both leading static hypotheses are now eliminated by evidence.** The next move is strictly
gated on the real witnesses — no further static guessing.

## 9. ⚠️ RECONCILIATION — the real target is `feat/madaros-bump-arena`, not this branch

This branch (`fix/checker-fnsig-table-heap-indirect`, `090767311`) is the **table-caps lane**;
its `madaros check main.sio` **SIGSEGVs (stack overflow in `check_modules_verdict_boot4`)** —
it does NOT reach the error-emitting state. The summary's "~1,958 errors / scope blocker"
session is a **different worktree**: `/workspace/sounio-arena`, branch `feat/madaros-bump-arena`,
HEAD `0f5b5b225`. Its 7 commits already include the fixes I was independently re-deriving:

- `d89a33369` **`*mut` in-place impl collect+check — eliminate ~8MB Checker SRET** (§3/§7's fix,
  already landed; my repros independently confirmed the copy was never the bug).
- `191b33987` **resolve functions via fn_sigs (not env) — frees env for locals, cuts E137 92%**.
- `87b6c3359` ref struct_idx two-byte encoding; `0f5b5b225` register missing runtime intrinsics
  in imported typecheck (the 7th fix, −125 errors).

## 10. ⛔ fn_sig cap/overflow ALSO refuted (arena source, static)

Candidate (advisor-suggested): arena routes resolution through `fn_sigs` but with a `[FnSig;512]`
cap < ~10,000 functions ⇒ high-index functions overflow ⇒ E137. **Refuted by reading arena
source:** `FnSigTable` is **chunked-growable** — `FnSigChunk{entries:[FnSig;512], next:*mut
FnSigChunk}`; 512 is the *chunk* size, the table grows via linked chunks to 1,048,576.
`fn_sig_table_find` correctly walks **all** chunks (`while gi<count` over `cur->next`);
`fn_sig_table_add` appends new chunks. So functions do not overflow and are found at any index.
`StructTable[;4096]` > ~1,620 structs (ok). Open: `EnumTable[;256]` is a **flat** cap — check
enum count; and imported-symbol **seeding completeness** (last fix was exactly that).

## 11. Eliminated so far (do not re-chase) / live candidates

ELIMINATED: SRET mega-copy (repro + already fixed by `d89a33369`); undersized Checker alloc
(128 MB deliberate); fn_sig cap/traversal (chunked-growable, correct).
LIVE (need witnesses): (a) incomplete imported-symbol seeding into fn_sigs; (c) genuine lost
locals in specific binders (match/if-let/while-let/closure/for). Building arena madaros to dump
`madaros check main.sio` witnesses → classify each unresolved name. **No fix until witnesses
classify.**

ELIMINATED (added): flat `EnumTable[;256]` overflow — main.sio has only ~130 enums (< 256),
`enum_table_find` linear over count; no overflow.

**Discriminator:** witnesses that are IMPORTED functions ⇒ candidate (a), fix = complete the
imported-symbol seeding (direct, no workflow). Witnesses that are genuine LOCALS ⇒ candidate (c),
fix = fan out read-only traces on the specific binder paths (the warranted workflow).

## 12. ✅ ROOT CAUSE FOUND — `TypeEnv` 4096 cap silently drops globals+locals (2026-06-25, arena build)

Built arena madaros (`make build-madaros`, off the lock-deadlock) and ran `madaros check
self-hosted/compiler/main.sio`: **exit 1, 1985 errors** (E137 785, E017 486, E035 150, …) —
no crash. Real witnesses recovered by mapping the error byte-offsets to module-0 (main.sio):
`eligible`, `after_len`, `after`, `func` — all **local variables used in a function's tail
expression** (e.g. `let eligible = check.0 … !eligible`), each paired with an E008 return-type
mismatch on the enclosing fn (the cascade). E137 = "use of undeclared variable".

**Mechanism (proven in source, quantitatively supported):**
- `TypeEnv.bindings: [TypeBinding; 4096]` (`env.sio:31`) — fixed cap.
- `checker_collect_global_item_inplace` (`check.sio`) binds each top-level global into scope-0
  env behind `if benv_idx < 4096 { … }` — **silently drops** past 4096 (no diagnostic).
- Top-level `let`/`var` parse as global items (`parser/items.sio: parse_global_let_item/var`),
  collected in **pass-1 over all 117 modules into ONE shared checker/env** with **no env reset
  between modules** (`check_modules_verdict_boot4`, confirmed no reset in `mod.sio`).
- The self-hosted bundle has **4774 column-0 globals > 4096**. Once cumulative scope-0 count
  crosses 4096 during collection, later globals — and in pass-2 the **locals** of functions in
  the late-collected modules — fail to bind ⇒ `lookup` returns `ty_error` ⇒ E137 ⇒ E017/E035/
  E008 cascade. `4774 − 4096 = 678 ≈ 785`; failures cluster in the **last ~13 modules** while
  module-0 (main.sio, collected first) is nearly clean — exactly the predicted signature. The
  inherited "scope-1+ lost / scope-0 works" was really "**low-collection-index seeds, high-index
  overflows the 4096 env cap**" — a CAP overflow, NOT a scope-logic bug.

This is the same silent-truncation class as the FnSig/Struct/Enum cap work. The per-module
frontend is immune because each module gets a fresh env (≤ its own globals, never 4774).

**Fix direction (caps-round pattern):** raise the `TypeEnv` capacity above the global count
with margin, make overflow a LOUD diagnostic (never silent drop), and — because a larger inline
`[TypeBinding;4096]` array enlarges every by-value `TypeEnv` copy in `bind/push_scope/pop_scope`
— either convert env mutation to in-place `*mut` (the existing `type_env_bind_mut`) or make
`TypeEnv` chunked-growable like `FnSigTable`. Empirical confirmation = the rebuild: raising the
cap should collapse E137. Verify no regression + that overflow can no longer be silent.

## 13. ⛔ CAP / LEAK / FILL ALL REFUTED by instrumented build (2026-06-25)

Instrumented `check_modules_verdict_boot4` to print `env.count` per module across both passes,
and `checker_collect_global_item_inplace`'s drop. Rebuilt arena, re-ran. Findings:
- **`DIAG_DROP_GLOBAL = 0`** — no global is ever dropped at collect time.
- **`env.count` climbs 98→3789 during pass-1** (globals), then is **FLAT at 3789 through all of
  pass-2** (functions push/pop cleanly). So: **no scope leak** (pass-2 flat) and **max count
  3789 < 4096** — the cap is **never reached**.
- The witness fn `compiler_main_test_tco_eligibility_alloc` has only **4 locals**
  (`func,info,check,eligible`) → binds at count ~3793 < 4096. **Not dropped by the cap.**

So §12's cap/fill mechanism is **wrong**: bindings are never dropped. E137 fires (`check.sio:6193`
`if !(*c).env.has_binding(e.name)`) because `eligible` is **genuinely absent from env at lookup**
despite `let eligible = …` earlier in the same block, with no pop between (stmt list is linear,
`checker_check_stmts_inplace:3760`). The only difference from the working per-module path is the
**high baseline (3789 globals in the shared env)**. Direct seed repros proved faithful at this
scale: 1.4 MB by-value `TypeEnv` copy, AND `let snap=(*c).env; (*c).env=snap.bind(...)` at count
3789 then lookup → **found** (`/tmp/envcopy_repro.sio`, exit 0). So it is **not** a generic
copy/cap/leak miscompile.

**Open, narrowed:** the binding for specific tail-locals (RHS = tuple-index `.0`, nested field
`.code.len`, or a call) is present-then-absent **only** at high baseline in the `*mut` all-modules
spine. Build #4 instruments the real `let`-bind (`DIAG_BOUND … has=`) and the has_binding failure
(`DIAG_MISS …`) gated to module 0 to settle: was `eligible` bound (`has=1`) and later vanished,
or did `bind` silently not take (`has=0`)? That observation localizes the true mechanism — which
is NOT any of: SRET copy, cap overflow, scope leak, global fill, high-count env-copy (all refuted).

## 14. ✅ ROOT CAUSE CONFIRMED — by-value `TypeEntry` param miscompile corrupts `env.count`

Bisected across builds #4–#7 (instrumentation gated to module-0 lets). The witness
`let eligible = check.0` in `compiler_main_test_tco_eligibility_alloc`:
- dispatches correctly as `StmtKind::StmtLet` (kind=0), enters `checker_check_let_stmt_inplace`;
- markers LM1/LM2/LM3 all fire → control reaches the bind region;
- with the bind moved first (build #7), `DIAG_BOUND` shows **`cnt=3793 has=1`** (eligible IS
  bound), but the later `!eligible` lookup `DIAG_MISS` shows **`cnt=3792`** — `env.count` has
  **rolled back by exactly 1**, dropping eligible's binding (index 3792 ≥ count 3792).

The rollback happens during **`checker_check_refinement_literal_inplace(c, s.span, bind_ty)`**,
which is the only call between the (now-faithful) bind and the lookup that takes a **264-byte
`TypeEntry` by value**. The seed (lean_single) **miscompiles that large by-value struct param**
when `bind_ty` originates from a field-access RHS (`check.0`, `nc.code.len`), and the bad copy
clobbers the caller's adjacent `(*c).env.count` by 1. Call-RHS lets (`func`/`info`/`check`) have
simpler `bind_ty` that doesn't trip it. The reorder (build #7) didn't help — it moved the bind
ahead of the corruption but the corruption still un-bound it (count 3793→3792). This is the same
**by-value large-struct miscompile class** named across prior sessions, here corrupting env state
rather than the return value.

**FIX attempts (builds #8–#9) — corruption is SYSTEMIC, not a single call:**
- `checker_check_refinement_literal_inplace` → take `refinement_id: i64` (build #8): no effect on
  E137 — that call runs *before* the bind in the original order, so it couldn't cause the
  post-bind rollback. (Kept anyway; behaviour-preserving and removes one by-value `TypeEntry`.)
- `checker_is_linear_type_inplace` → take `&TypeEntry` (build #9): **did stop the in-handler
  rollback** — `DIAG_END` count now 3793 (was rolling back inside the handler) — but the single
  `-1` **relocated** to during the `!eligible` tail check: MISS still `cnt=3792`. The unary
  checker's MISS is at the operand lookup (`check.sio:4499`), i.e. count is *already* 3792 on
  entry — the drop happens while passing `!eligible` as by-value `Expr`/`Stmt`.

**Conclusion: this is a systemic seed (lean_single) codegen miscompile** — a by-value large-struct
parameter copy (`TypeEntry`/`Expr`/`Stmt`, 264 B+) in the `*mut` checker spine writes a stale/
mis-addressed value into the adjacent `(*c).env.count`, dropping it by 1, **only when the baseline
is high (~3792, the 117-module global accumulation)**. Per-module checking is immune because it
never reaches that count. Fixing one by-value call site relocates the `-1` to the next — there are
hundreds of such copies, so call-site whack-a-mole will not converge. The `is_linear` `&TypeEntry`
fix is a genuine partial improvement and is sound to keep.

**Real fix candidates (strategic, not piecemeal):** (a) eliminate the let handler's
`let cur_env = (*c).env` 1.5 MB stack copy via in-place `type_env_bind_mut(&!(*c).env, …)` — tests
whether the giant stack object is the layout trigger; (b) prevent the env baseline from reaching
~3792 (the trigger) — e.g. resolve globals via a side table rather than accumulating them all in
scope-0 env across 117 modules; (c) fix the seed by-value large-struct copy codegen (root, but the
frozen bootstrap seed). The `is_linear` fix + this diagnosis are the durable deliverable; the
strategic fix is multi-session.

## 15. ✅✅ ACTUAL ROOT CAUSE — parser line-start `!` absorption (NOT codegen, NOT count)

The §12–14 "high-count / multi-module / systemic codegen" framing was **WRONG** — built on two
unverified inherited claims ("per-module works", "is_linear relocated the −1"). A 6-line
single-file `madaros check` (no `use`, tiny count) **reproduces instantly**:
```
fn h() -> (bool, i64) { (true, 5) }
fn t() -> bool { let a = h()  let b = a.0  !b }   // E137: b undeclared
```
Variant matrix nails the invariant — it is **not** count, multi-module, or the RHS kind:
| program | E137 |
|---|---|
| `let b=a.0` ⏎ `!b` | **1** |
| `let b=a.0` ⏎ `b` (no `!`) | 0 |
| `let b=true` ⏎ `!b` | **1** |
| `let b=g()` ⏎ `!b` | **1** |
| `let b=a.0` ⏎ `let z=1` ⏎ `!b` | 0 |

**Root:** `tk_precedence(TokenKind::Bang) == 13` (a postfix precedence for `x!`). In the
expression-continuation loop (`parser/exprs.sio`), a line-starting `!b` after a complete
expression is consumed as a high-precedence continuation → `let b = X` becomes `let b = X ! b`,
referencing `b` inside its own initializer → E137 undeclared. Identical class to the already-fixed
line-start `*`/`&`/`&!` deref bug (`1edd39d7d`); `!` and `~` were simply never added to the
newline-tight break. Explains all three witnesses: `!eligible`, `!ocp_reg_flows_to_observation(…)`,
and (for `after_len`) the same when a tail begins with a tight prefix op.

**FIX (build #10):** in `parser/exprs.sio`, mirror the Star/Amp/AmpBang break for `Bang`/`Tilde`:
a line-starting tight `!`/`~` terminates the previous expression (begins a new statement). A real
postfix `x!` has the newline *after* the operand, so `had_newline_before` is false → unaffected.
One-line-class parser fix; expected to collapse the E137 cascade. The `is_linear`/`refinement_id`
by-ref changes are independent soundness/perf improvements and are retained.

**Lesson:** the discriminating test (minimal repro, verify inherited assumptions) belonged at the
top, not after 9 rebuilds chasing a phantom codegen corruption.

## 16. ✅ LANDED FIX + residual (final state, arena worktree)

The line-start-operator absorption was a **whole class**, not just `!`. All in `parser/exprs.sio`,
each a one-line newline guard mirroring the pre-existing LParen guard (line 57):
1. **`!`/`~`** (Bang/Tilde, prec 13): line-start tight `!b`/`~x` begins a new statement (infix loop).
2. **method-call `(`** in `parse_dot_expr_box` (~1307): `nc.code.len ⏎ (after_len-…)` was glued into
   `nc.code.len(after_len-…)`. Added `&& !p.had_newline_before()`.
3. **index `[`** in both postfix loops (line 63 infix loop, ~1133 second loop): `x ⏎ [a,b]` was glued
   into `x[a,b]`; also fixes array-literal tails `compute() ⏎ [x,x,x]`.

**Result:** `madaros check self-hosted/compiler/main.sio` **E137 785 → 588, total 1985 → ~1614**
(−371). **main.sio (module 0) is now 100% E137-clean.** Minimal repros all pass. Diff is parser-only
(check.sio/mod.sio reverted to `0f5b5b225`; the refinement_id/is_linear by-ref detour changes were
dropped — not needed once the real cause was found). Atomic, reviewable.

**Residual 588 E137 = a SECOND, distinct bug** (not parser): **tuple-destructuring `let (a,b) = …`
does not bind its pattern variables.** Concentrated in 4 GPU SPIR-V modules (`gpu/epistemic_spirv.sio`
=260, `gpu/spirv_lower.sio`=142, `gpu/spirv.sio`=123, +`gpu/spirv_*`=53), reproducible **standalone**
(`epistemic_spirv.sio` alone = 260 E137). Minimal repro: `fn pr()->(i64,i64){(1,2)} fn t()->i64{let
(a,b)=pr()  a+b}` → both `a` and `b` E137-undeclared (inserting a statement between does NOT fix it,
so it is NOT the line-start class). `Stmt` has only `name: Name`; tuple-lets are parser-desugared
(memory: `tptup_desugar`/`TPTUP_DESUGAR_OUT`), and the AST shows separate `let a`/`let b` (kind 0) —
yet neither binds at check time. The desugar→`*mut`-check interaction is the next investigation;
`checker_bind_tuple_pattern_list_idx_inplace` (check.sio:6350) already binds tuple patterns for
`match` and is the likely reuse target. **This is the clean next dispatch — separate from the landed
parser fix.**

## AI disclosure
Diagnosis by AI agent (Claude) under human direction; every claim re-runnable against the
cited source lines and the pending build reconfirmation.
