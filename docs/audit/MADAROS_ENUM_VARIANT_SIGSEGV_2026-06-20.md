<!-- docs:meta
topic_id: repo.docs.audit.madaros-enum-variant-sigsegv-2026-06-20
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-enum-variant-sigsegv-2026-06-20
-->

# Madaros: enum-variant construction SIGSEGV — converges on the Box::new crash (2026-06-20)

## Finding (overturns the census)
The census listed "enum match → silent no-emit (rc 0)". That was a **measurement artifact**:
the harness used `local cr=$?`, which captures the `local` builtin's exit (always 0), not the
build's. The real behaviour:

```
enum E { A, B }
fn main() -> i64 { let e = E::A   5 }      # build -> EXIT 139 (SIGSEGV), no marker files
```

`build` crashes **inside `bridge_lower_single_module_box`** (front-half IR lowering) — before
the first progress marker `/tmp/nv2_m1_after_lower` is written.

## It is the Box::new crash, reached from the enum path
Core-dump forensics (`/tmp/core.*`, madaros static non-PIE base `0x400000`):

| | enum `E::A` | `Box::new(7)` |
|---|---|---|
| Faulting RIP | `0x3ebe4f2` | `0x3ebe4f2` |
| Instruction | `mov 0x0(%rdx),%rax` | `mov 0x0(%rdx),%rax` |
| Regs | `rax=1 rbx=0 rdx=1` | `rax=1 rbx=0 rdx=1` |
| Shared frame in chain | `0xf81a40` | `0xf81a40` |

`0x3ebe4f2` is a loop: `mov (%rcx,%rdx,8),%rax ; mov %rax,%rdx ; mov 0x0(%rdx),%rax` —
load `data[index]`, then deref `element[index]`; crashes on `index 0` with `len>0` but
`data[0]` null. `0xf81a40` copies a ≥0x28-byte struct field-by-field right after a `call`
that returns a struct — the **by-value large-struct return** machinery (in the lowerer,
methods return `(Lowerer, i64)`; the `Lowerer` is a huge struct holding internal lists).

The two crashes differ only in the *caller* above `0xf81a40` (Box: `0x786f42`; enum:
`0x3f22fb1 → 0xf82173 → 0xf822ce`). The crash site and the by-value-struct mechanism are
identical. This is the documented lean_single nested-store / SRET by-value-struct miscompile
(`feedback_lean_single_miscompilations`, `project_madaros_boxnew_sigsegv_2026-06-19`).

Enum *declaration* alone is fine (`enum E{A,B} fn main()->i64{5}` runs); only *use* of a
variant (`E::A`, an `ExprPath` lowered at `lower.sio:6939`) triggers it.

## Conclusion
Enum-variant construction is **not an independent codegen hole** and has **no clean
enum-specific source fix** — it is gated by the same by-value-struct-return / nested-store
miscompile as `Box::new` (Codex's lane). 

**Recommendation:** re-test enum construction + match **after** the Box::new fix rebuild.
- If that fix targets the root (lean_single nested-store / by-value-struct return), enum is
  very likely fixed at the same time.
- If it is a Box-call-site-only workaround, enum will still crash and needs the same
  build-in-a-local workaround applied at the enum-lowering value spine.

Do **not** patch enum independently now: it would be a codegen change at the exact crash
site Codex is editing, not an enum-logic fix.

## Census correction
`MADAROS_NATIVE_V2_CODEGEN_CENSUS_2026-06-19.md` "enum match: silent no-emit" should read
**"SIGSEGV@compile (= Box::new crash family)"**. The "method call: no-emit" row is suspect for
the same harness reason and must be re-measured with a correct exit-code capture.
