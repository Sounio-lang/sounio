<!-- docs:meta
topic_id: repo.docs.audit.g1-wip.if-while-cond-segv-diag-2026-06-01
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.g1-wip.if-while-cond-segv-diag-2026-06-01
-->

## ✅ RESOLVED 2026-06-01 — root cause OVERTURNED, crash FIXED (no re-bootstrap)

**The entire "multi-struct-arg corruption of `got`" thesis below is FALSIFIED.**
Fresh clean-gdb on the rebuilt `.dbg/mc.elf` (ASLR-on, no instrumentation) showed
BOTH args clean at `checker_report_mismatch_inplace` entry: `expected.kind=2`
(bool), `got.kind=0` (i64) — a *correct* mismatch. The crash is in
`print_type_name` (check.sio:4667): given a clean `bool`, it recurses to overflow.

**Real root cause: `bin/souc` miscompiles `match` arms with BARE (unqualified)
enum-variant patterns** — it drops discriminant dispatch entirely, so EVERY arm
executes in sequence. `print_type_name`'s `match ty.kind { TyI8 => … }` used bare
`TyX` patterns ⇒ all leaf arms print unconditionally AND the `TyModel`/
`TyModelFamily` arms (which recurse via `type_entry_list_get`) fire for ANY input
⇒ unbounded self-recursion ⇒ 12.7 MB stack overflow. Minimal repro (in this doc's
session, compiled by `bin/souc`):
- BARE `match k { A0 => …, A1 => … }`, `name(K::A1)` → prints `a0.a1.` (ALL arms).
- QUALIFIED `match k { K::A0 => …, K::A1 => … }`, `name(K::A1)` → `a1.` (correct).
- Recursion analog: bare `Node => { …; show(K::Leaf) }` → infinite; qualified → fine.

**FIX (conforms to the codebase's own convention — every other check/ file already
qualifies; check.sio was the lone all-bare holdout):** qualified all **84** bare
`Ty[A-Z]… =>` arms in check.sio to `TypeKind::Ty…` (verified all 84 ∈ TypeKind;
TypeExprKind `Type…` arms left untouched). Rebuilt `.dbg/mc.elf` with the EXISTING
`bin/souc` (md5 `e35ef063…` UNCHANGED — **no lean_single edit, no re-bootstrap,
zero brick risk**). New mc.elf md5 `3e456577…`.

**VALIDATED:** all 6 previously-SIGSEGV cases (`if 1{}`, `if 1==1{}`, `while 1==1{}`,
`if 1<2{}`, `let b=1<2;if b{}`, `f()`) now return cleanly — no rc=139. `if 1{}` →
clean `E006 expected bool / found i64` (print_type_name prints correct names).
Controls still rc=0. **`scripts/ci/g1_expr_recursion_gate.sh` PASSES** (VmStk
peak=132 kB). `f()` now rc=0.

**UPDATE — `1==1`→`()` ALSO FIXED (same bare-pattern bug):** it was NOT a separate
"op-type check disabled" gap as first guessed — `binary_result_type` lives in
**compat.sio** (not check.sio) and used bare `OpAdd =>`/`OpEq =>` BinaryOp patterns →
all-arms-fire → returned `()` for comparisons. Qualified compat.sio's 32 bare arms
(BinaryOp/UnaryOp/AssignOp) → rebuilt → `if 1==1{}`,`if 1<2{}`,`while 1==1{}` now
**rc=0** (valid). Verified no-regression vs `mc_g2c.elf` baseline (only the if/while
comparison cases changed, all 139→0/1; struct/enum/let/array error identically =
pre-existing modular gaps). mc.elf md5 `aeee8a2c…`. G1 gate still GREEN.

**REMAINING (the bare-pattern bug is COMPILER-WIDE, not just check.sio):** still-bare
enum-variant matches in resolve/resolve.sio (~125, affects --check), resolve/imports.sio
(~22), printer/print_ast.sio (~141, output-only), printer/buffer_print.sio (~122,
output-only). All qualifiable via the verified 257-entry variant→enum map. See
[[project_g1_crash_fixed_bare_patterns]]. (epistemic/dependent "bare" arms are ALL-CAPS
const-pattern matches, a DIFFERENT category, left alone.)

**DECISION FORK for the user (the OLD "codegen root fix vs workaround" framing is
dead — it rested on the falsified premise):**
- (A) Qualify the remaining bare-pattern families source-wide (per-enum, mechanical,
  reversible, no re-bootstrap) — i.e. finish conforming check.sio to convention.
- (B) Address bare patterns at `bin/souc` level — but the right fix may be to
  *reject* unresolved bare variants (a checker error) rather than *accept* them;
  re-bootstrap = brick-risk + now low-value. Likely not worth it.

Everything below is the (now-superseded) diagnostic trail that LED to this fix.
─────────────────────────────────────────────────────────────────────────────

# gdb root-cause: `if/while <binary-cond>` crash under the banked *mut patch

**Session 2026-06-01 (fresh). Branch `modular/move-codegen`, worktree
`/workspace/sounio-move-codegen`. State: banked `g1_mut_expr_spine_wip.patch`
APPLIED + 3 (ineffective) "hoist field-match→local" edits. Built `.dbg/mc.elf`
(85MB). `bin/souc` UNTOUCHED. gdb 15.1 installed (`sudo apt-get install gdb`).**

## The crash, fully localized with gdb

`fn main(){ if 1<2 {} }` --check → SIGSEGV. gdb shows:
- Faulting insn `orb $0x0,(%rax)` = a **stack-clash PROBE** in a function prologue
  (`sub $0x4df0,%rsp` then probe loop). It probes BELOW the stack base
  `0x7fffff3ca000` ⇒ **stack overflow**.
- `ulimit -s = 12500` (12.2 MB). rbp sits ~12.2 MB below the stack top ⇒ the WHOLE
  stack is consumed. (The earlier "460 KB VmStk" was a SAMPLING ARTIFACT — my
  coarse probe loop missed the fast growth.)
- rbp-chain walk: **652 frames, 634 with the IDENTICAL return address 0x4c3daa0**,
  each frame exactly **19,976 bytes**. ⇒ **direct unbounded self-recursion** of one
  function F. Disasm of the call site: `0x4c3da9b: call 0x4c3cbd9`, and F's entry
  IS `0x4c3cbd9` (`push rbp; mov rsp,rbp; sub $0x4df0,rsp; <probe>`). F calls
  itself. 634 × ~20 KB ≈ 12.7 MB → overflow.

## What F is, and why it recurses

Under gdb (ASLR off → different layout) the program does NOT segv at the same
spot; it prints `error[E006] conditions must be of type bool` and then an
**unbounded `…Model<…Model<…` type dump**. `F` is the **type-name formatter**: it
recurses through `TypeEntry.inner: Option<Box<TypeEntry>>` (types.sio:126) and has
**no depth guard**. The `cond_ty` it is asked to print is a **corrupted, CYCLIC
TypeEntry** (its `inner` forms a cycle), so the formatter recurses forever.

## The chain (root → crash)

1. `if`/`while` type the condition: `let cond_ty = checker_check_opt_expr_inplace(c, e.left)`.
2. For a BINARY cond (`1<2`, `1+2`), the *mut binary checker returns a **CORRUPTED
   TypeEntry** (should be clean `bool`). For a LEAF cond (`true`, an ident bound to
   a literal) the type is clean → no crash.
3. corrupted cond_ty ≠ bool ⇒ E006 mismatch fires ⇒ the formatter prints the type
   ⇒ recurses on the cyclic `inner` ⇒ 634-deep self-recursion ⇒ stack overflow.

## Why this is a bin/souc CODEGEN miscompile (not source, not "more spine")

- In SOURCE, `checker_check_binary_expr_inplace` for int<int returns
  `binary_result_type(OpLt,i64,i64)` = a clean `bool` (no `inner`). The
  **by-value** compiler (bin/souc itself) type-checks `if 1<2{}` with zero errors.
  So the source logic is correct; the *mut build corrupts the returned struct.
- The corruption is in the **SRET (large-struct) return of `TypeEntry` through the
  *mut expr spine** — `binary_result_type → binary_inplace → expr_inplace
  (var result = …) → opt_expr_inplace → caller`. `ident_inplace`'s TypeEntry return
  (a copy of an env entry) is NOT corrupted; `binary_inplace`'s CONSTRUCTED-then-
  SRET-returned TypeEntry IS — and `binary_inplace` has a HUGE frame (it reserves
  stack for by-value `*c` copies at the hyper/knowledge branches, lines 55/80).
- Proof the corruption rides with the binary RESULT TYPE, not the cond position:
  `let b=1<2; if b {}` (leaf-ident cond) ALSO crashes — `b`'s STORED type is
  already corrupt; `let x=true; if x {}` works.

## Behavioral evidence table (capped 8 MB stack)
| program | rc | |
|---|---|---|
| `1+2`, `1+2*3`, `1<2`, `let b=1<2`, `{1<2}` | 0 | binary type computed but never CONSUMED/printed |
| `if true{}`, `while true{}`, `let x=true; if x{}`, `if x{}` | 0 | LEAF cond → clean type |
| `if 1<2{}`, `if 1+2{}`, `if (1<2){}`, `while 1==1{}` | 139 | binary cond → corrupt type → formatter recursion |
| `let b=1<2; if b{}`, `let b:bool=1<2; if b{}`, `let x=1; if x<2{}` | 139 | corrupt binary type later consumed |

## Implication for strategy
This is **NOT incomplete spine conversion** — the *mut binary checker IS
converted; bin/souc MISCOMPILES it. Converting more *mut functions does NOT fix
this. Same family as bug#1 (`0afad182c`, match-on-field miscompile in *mut fns).

## Fix options
1. **bin/souc codegen fix** (the deprecated "lever", now PRECISELY localized to
   SRET/large-struct `TypeEntry` return out of a *mut fn). Real fix; requires
   editing `lean_single.sio` + re-bootstrap (highest-risk op).
2. **Source workaround in the *mut spine**: route the binary result type through a
   Checker field (`checker_store_from_value` already exists) instead of the
   miscompiled SRET return; read it back in the dispatcher. Testable in-worktree,
   `bin/souc` untouched. Cheapest to try next.
3. **Depth-guard the type formatter F** — stops the crash but the E006 is SPURIOUS
   (1<2 IS bool); BAND-AID, wrong type-check result. Not a real fix.

## gdb root-cause (CONFIRMED, refined)
- The recursing fn F = type-name formatter at `0x4c3cbd9`. It calls helper
  `G` (`0xcb4c73`, an inner/next-type extractor) which returns a `TypeEntry`
  (34 slots = `rep movsq ecx=0x22` = 272 B), then recurses F on G's result.
  Never terminates because the type is a self-perpetuating kind=21 (`TyModel`).
- The `cond_ty` (the "got" type in the E006 "must be bool" mismatch) is a
  CORRUPTED `TypeEntry`: kind=21 (Model) with a cyclic inner, instead of the
  real type. Confirmed for BOTH `if 1<2{}` (real type bool) AND `if 1 {}`
  (real type i64) — same recursion, same kind=2→kind=21 pattern.
- **Frame-size is NOT the trigger:** `if 1 {}` corrupts cond_ty without ever
  calling `binary_inplace` (literal `1` is inlined in expr_inplace). So the
  earlier "huge binary_inplace frame" hypothesis is FALSIFIED. The corruption
  is in basic large-struct (`TypeEntry`, 34 slots) value movement through the
  *mut spine (SRET-return capture into a local, and/or by-value-arg passing to
  `checker_report_mismatch_inplace`).
- ident conds work (`if x{}`, x bound to a literal) because bool conds never
  enter the mismatch/format path; the corruption only surfaces when cond_ty is
  consumed by the type formatter.

## Codegen sites read (lean_single.sio, bin/souc source)
- `emit_sret_destination_x86` (1707): <512KB returns → `[rbp-off]` stack slot
  from NEXT_SLOT; ≥512KB → fixed BSS addr (TypeEntry is 272B → stack-slot path).
- `stabilize_return_agg_x86` (7195): copies result into `[r12]` (CURRENT_SRET_SLOT>0)
  else into fixed BSS `RET_AGG_BSS_OFF` (7204) — a shared dest = reentrancy hazard.
- prologue (24407-24421): SRET fns `push r12; mov r12, rdi`; non-SRET fns do NOT
  save r12 → a non-SRET callee using r12 as scratch would clobber an SRET
  caller's dest ptr. (Unconfirmed; grep can't clear raw-byte r12 uses.)

## SHARPEST trigger + localization (for the fix)
- Minimal trigger: `fn main(){ if 1 {} }`. cond_ty=i64 is CLEAN at `is_bool_type`
  (check.sio:2732) — confirmed (i64≠bool fires the mismatch correctly). It is
  CORRUPTED only after being passed as the **5th argument** (`got: TypeEntry`,
  by value) to `checker_report_mismatch_inplace` (check.sio:2001), which then
  does `print_type_name(got)` (check.sio:2024). `print_type_name(expected)`
  (line 2022, also a by-value TypeEntry) prints FINE → the formatter is OK; the
  `got` arg is already garbage on arrival.
- Arg signature of the corrupting call: `(c:*mut Checker, span:Span, code:i64,
  expected:TypeEntry, got:TypeEntry)` — THREE struct-by-value args. Calling
  convention (lean_single emit_setup_call_args_shift_x86:1686) passes each arg as
  ONE slot = a POINTER to a materialized copy (materialize_aggregate_expr_x86:7034,
  struct-like branch 7052-7066 → emit_bulk_copy_to_slots_x86 into NEXT_SLOT temp).
  Suspect: materialization of multiple struct args clobbers/aliases the temp slot
  or pointer of the later arg (`got`).
- SECOND manifestation: `if 1<2{}` corrupts cond_ty earlier (at the binary SRET
  return, before is_bool_type). Likely the SAME codegen family (large-struct value
  movement in *mut fns): SRET return (stabilize_return_agg_x86:7195) + struct-arg
  materialization (7034).
- bin/souc itself type-checks `if 1<2{}` with 0 errors (by-value path), so the
  defect is specific to how bin/souc compiles the *mut spine's large-struct moves.

## DO NOT edit lean_single.sio on a hypothesis
The previous session already burned a guess-and-rebootstrap (the 512KB→64KB
threshold change in materialize_aggregate) — it FAILED because it targeted frame
size, not the slot/pointer corruption. Pin the EXACT miscompiled instruction
(minimal repro disasm OR mc.elf arg-materialization bisection) BEFORE editing.

## PINNED to: multi-struct-argument passing corrupts the LATER struct arg
Source instrumentation (print kind, then reverted) on the real mc.elf:
- `if true{}`: cond_ty.kind=2 (TyBool) — CORRECT, no mismatch, no crash.
- `if 1{}`:   cond_ty.kind=0 (TyI64) — **CORRECT at if_expr assignment**. So cond_ty
  is NOT corrupted at the SRET return; it is correct entering the mismatch path.
- `if 1<2{}`: cond_ty.kind=3 (TyUnit) — WRONG (should be TyBool=2); a SEPARATE
  binary-result mistyping bug, but not the crash cause.
- At `checker_report_mismatch_inplace(c, span, code, expected:TE, got:TE)` ENTRY:
  `expected.kind`=2 (TyBool, CORRECT — the 4th struct arg survives), but reading
  `got.kind` (5th struct arg) **CRASHED** (wild pointer) WITH the probe; WITHOUT
  the probe `got` is a readable-but-corrupt TyError/TyModel (kind 21/24) cyclic.

**Conclusion:** the crash is **multi-struct-by-value-argument passing to a *mut fn
miscompiled — the LATER struct arg (`got`, 5th) gets a wild pointer / corrupt
value while the earlier struct arg (`expected`, 4th) survives.** cond_ty is correct
until the call; the call corrupts it. Manifestation (wild ptr vs corrupt value) is
LAYOUT-SENSITIVE — adding any statement shifts it (so source instrumentation is
itself Heisenberg-unreliable; synthetic 3-struct-arg repro did NOT trigger).

Calling convention (lean_single): args evaluated L→R, each struct materialized via
materialize_aggregate_expr_x86:7034 → emit_bulk_copy_to_slots_x86:7008 (leaves
rax=&copy, CORRECT on paper) → `push rax`; then emit_setup_call_args_shift_x86:1686
loads first 6 stack slots into rdi/rsi/rdx/rcx/r8. SOURCE LOOKS CORRECT → bug is a
subtle emitted-code interaction (NEXT_SLOT temp aliasing / frame-size patch /
register clobber) when ≥2 large struct args are materialized for one call.

## NEXT: pin the exact line (needs emitted-code gdb, fresh context)
On a CLEAN-rebuilt mc.elf: break at `checker_report_mismatch_inplace` entry (find
via the formatter F's caller return addr), dump rcx(=&expected) and r8(=&got) and
[rcx]/[r8]. If r8 is a non-stack/wild addr → 5th-arg POINTER setup is wrong
(emit_setup_call_args_shift / push ordering). If r8 valid but [r8] garbage → the
materialized COPY is clobbered (NEXT_SLOT temp aliasing in materialize_aggregate).
That single bit picks the exact lean_single fix. THEN edit + gen2==gen3 + full gate.

## (older) NEXT note: pin the exact codegen defect
Minimal repro compiled directly by `bin/souc` (fast loop + tractable disasm):
a *mut-receiver fn returning a 34-slot struct with `Option<Box<Self>> inner`,
captured into a local by a caller that then consumes it. If it reproduces the
kind/inner corruption, disassemble to find the miscompiled large-struct move,
fix in lean_single.sio, re-bootstrap (gen2==gen3 + full gate). If it does NOT
reproduce (layout-sensitive), bisect on the real mc.elf path instead.

## State / safety
- gdb at default stack is SAFE here (NOT recursion-to-OOM in the 100GB sense — it's
  a bounded 12.2 MB overflow; rc=139 fast). The OOM warning applies only to
  `ulimit -s unlimited`.
- Tree: patch + 3 hoist edits (the hoists are harmless but did NOT fix anything;
  candidates for revert). Nothing committed. `bin/souc` md5 unchanged.
- Refines [[project_modular_B_repro_verdict]] (layout-sensitive) and
  [[project_g1_crash_recursion_vmstk]] (recursion confirmed, now NAMED via gdb).
