<!-- docs:meta
topic_id: repo.docs.audit.madaros-sret-root-synthesis-2026-06-20
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-sret-root-synthesis-2026-06-20
-->

# Madaros front-half crashes — SRET root, split into two distinct causes (2026-06-20)

Investigation of the "SRET/by-value-struct miscompile root" behind the native_v2 crashes
(Box, enum, method, impl). Measured against prebuilt `bin/madaros-linux-x86_64` @ main
`659492156`. **Decisive control: `ulimit -s unlimited`.**

| Construct | default 8 MB stack | `ulimit -s unlimited` | Faulting instr | Cause |
|---|---|---|---|---|
| `impl C { }` (empty) | SIGSEGV | **compiles (rc 5)** | `0x54e5993` stack-probe | **stack overflow** |
| method def+call (`c.get()`) | SIGSEGV | still SIGSEGV | `0x5d9f82b` `mov 0x0(%rdx)` | null-deref miscompile |
| enum ctor (`E::A`) | SIGSEGV | still SIGSEGV | `0x3ebe4f2` `mov 0x0(%rdx)` | null-deref miscompile |
| `Box::new(7)` | E137 / SIGSEGV | rc 1 (clean error) | — | context-dependent |

## Root 1 — SRET-cascade STACK OVERFLOW (impl blocks)
The function processing `impl` items has prologue:
```
54e5978: push %rbp ; mov %rsp,%rbp ; push %r12
54e597e: sub $0x19b6c40,%rsp          # 0x19b6c40 = 26,962,496 bytes ≈ 25.7 MB frame
54e5985..96: stack-probe loop; orb $0x0,(%rax) faults past the 8 MB guard page
```
A ~25.7 MB stack frame — from accumulated by-value `Lowerer`/module SRET slots in the
impl-processing path in **`self-hosted/ir/lower.sio`** (the summary-preseed +
`lowerer_lower_impl_methods_mut` family iterate items returning big structs by value).
`ulimit -s unlimited` makes the empty-impl program compile and run — confirming pure stack
overflow, no logic error.

**Relation to `fix/native-codegen-sret-regression` (`5b42e985b`):** that branch eliminated
the *same class* of 20+ MB frames, but only in **codegen** files (`lower_ir.sio`,
`codegen.sio`, `codegen_x86_linux.sio`, `encode.sio`) — it did **not** touch `lower.sio`.
So the impl-processing frames in `lower.sio` are a **separate instance of the same
SRET-cascade pattern** that the branch's `&!`-mutable-ref technique should be applied to.
Fix shape: convert the by-value `(Lowerer)` / module returns in the impl summary/lowering
functions to `&! Lowerer` void-return (the proven `var c = (*nc); …; (*nc) = c` pattern).

## Root 2 — by-value-struct NULL-DEREF miscompile (method, enum)
method and enum crash **even with unlimited stack**, at `mov 0x0(%rdx),%rax` with `rdx`
null — a genuine miscompile (a struct copied from a null pointer / a malformed list whose
`data[0]` is null though `len>0`). Distinct from Root 1 (more stack does not help).
Hypotheses tried and **falsified**: (a) "by-value `Lowerer` return is miscompiled" —
contradicted because `LowerExprArgsResult` embeds a `Lowerer` and works in the free-call
path; (b) "method name resolved bare vs registered mangled → phantom" — contradicted because
a same-named free function shadow still crashes, and `impl`-method-declared-but-not-called
crashes too. The exact null source is not yet named (stripped binary; `read_env` broken in
the prebuilt so `SOUNIO_LOWER_*_TRACE` diagnostics are dead). Needs a symbol-ful build or a
working trace to localize precisely.

## Net
- The four census "holes" + this dig resolve into: **2 genuine independent lowering gaps**
  (int-println, for-loop — FIXED), **1 SRET-cascade stack overflow** in `lower.sio`'s impl
  path (Root 1 — `ulimit` workaround; proper fix = `&!` refactor, not yet applied to
  `lower.sio`), and **1 by-value-struct null-deref miscompile** (Root 2 — method/enum/Box).
- **Quick win available now:** running madaros with `ulimit -s unlimited` unblocks impl
  blocks (Root 1) without any rebuild. It does NOT fix Root 2.
- The keystone for "Madaros 100%" / fixed point is fixing **both** roots, since the
  compiler's own source uses `impl` blocks and methods pervasively.
