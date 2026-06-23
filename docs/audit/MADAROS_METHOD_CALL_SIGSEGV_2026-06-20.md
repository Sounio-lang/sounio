<!-- docs:meta
topic_id: repo.docs.audit.madaros-method-call-sigsegv-2026-06-20
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-method-call-sigsegv-2026-06-20
-->

# Madaros: method-call SIGSEGV — by-value-struct-return miscompile (2026-06-20)

## Finding (census correction)
Census said "method call → no-emit (rc 0)" — another `local cr=$?` harness artifact.
Measured correctly:

```
struct C { v: i64 }
impl C { fn get(self: &C) -> i64 { self.v } }
fn main() -> i64 { let c = C { v: 11 }   c.get() }     # build -> EXIT 139 (SIGSEGV)
```

Crashes inside `bridge_lower_single_module_box` (front-half), before the `m1` marker.
Plain struct construction + field access (`C{v:11}; c.v`) **works** (rc 11) — it is the
method *call* that crashes.

## Distinct site, same root class as Box/enum
| | method `c.get()` | Box::new / enum |
|---|---|---|
| Faulting RIP | `0x5d9f82b` | `0x3ebe4f2` |
| Instruction | `mov 0x0(%rdx),%rax`, `rdx = [-0x18(rbp)] = 0` | `mov 0x0(%rdx),%rax`, list `data[0]` null |
| Regs | `rax=rbx=rdx=0` (null deref) | `rax=1 rbx=0 rdx=1` |
| Mechanism | copy big struct field-by-field from a **null** pointer | iterate list, deref null element[0] |

Both are **by-value large-struct return** miscompiles (the lowerer's methods return
`(Lowerer, i64)`; helpers return `LowerExprArgsResult` etc. — all big structs copied by
value). `lower_method_call_expr_ref` (`lower.sio:6589`) is logically clean — it lowers the
base, lowers args, `prepend_arg`, `find_or_add_fn_id`, `ir_lower_prepend_validated_arg`,
emits the call. One of those nested big-struct-by-value returns is miscompiled so its SRET
pointer is null, and the caller copies fields from null → crash.

Not universal: `while`/`if` lowering also return `(Lowerer, i64)` by value and **work**, so
this is specific struct shapes / call sites, not every by-value return.

## Conclusion — corrected census
The four census "holes" resolve into **two genuine lowering gaps (fixed)** and **two
instances of a systemic by-value-large-struct-return miscompile**:

| Hole | Real nature | Status |
|---|---|---|
| integer `println` | missing int→string dispatch in lowering | ✅ FIXED `fix/madaros-print-int-dispatch` |
| `for` range loop | missing `ExprForIn` lowering case | ✅ FIXED `fix/madaros-for-loop-lowering` |
| enum-variant construction | SIGSEGV `0x3ebe4f2` (by-value-struct, = Box) | 🔗 gated by Box/SRET fix |
| method call | SIGSEGV `0x5d9f82b` (by-value-struct, distinct site) | 🔗 gated by SRET/by-value fix |

**The real "Madaros 100%" blocker is the systemic by-value-large-struct-return / SRET
nested-store miscompile** (lean_single), which crashes madaros at multiple lowering sites
(Box::new, enum construction, method calls). The frame-fix / SRET-regression work
(`fix/native-codegen-sret-regression`, `5b42e985b`) and Codex's Box::new lane target this
family. Once that root lands, enum + method-call should be re-tested together — they are
**not** independent source fixes and should not be patched at these crash sites
independently (it would collide with the SRET/codegen work).
