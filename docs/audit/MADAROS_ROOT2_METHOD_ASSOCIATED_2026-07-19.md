<!-- docs:meta
topic_id: repo.docs.audit.madaros-root2-method-associated-2026-07-19
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-root2-method-associated-2026-07-19
-->

# Madaros Root 2 — method / associated-function fix (partial) — 2026-07-19

## Claim

> Under **default Madaros**, same-module **`&self` method calls** and **`Type::method`
> associated functions** (including multi-module import of associated constructors)
> compile and run correctly. Multi-module **instance** method calls remain residual.

## Root causes (two)

| Bug | Symptom | Fix |
|---|---|---|
| A. Path callee name | `E::of` lowered as bare `"E"` → body-less fn → **runtime SEGV** | `expr_to_callee_name_ref`: multi-segment Path → `ir_mangle_method_name(first, last)` = `E_of` |
| B. `&self` value pass | `x.get()` with `self: &E` passed **value** not pointer → **runtime SEGV** | Track `IrFunction.first_param_is_ref`; auto-`OpRef` on non-ref receivers |

Same-module method+associated (including get-after-add) now **PASS**.

| Bug | Fix |
|---|---|
| Path `E::of` → bare `"E"` | mangle `Type_method` |
| `&self` value pass | `first_param_is_ref` + auto-`OpRef` |
| `let s = a.add(&b)` lost type | bind struct type on **ExprMethodCall** lets |

**Residual:** multi-module instance method (`x.get()` across `use`) still compile-SEGV
at seed lower. Free-function / associated multi-module paths work.

## Evidence (rebuilt Madaros)

```
R1 free multi-module          OK
R2 E::of multi-module         OK  (was runtime SEGV)
R5 &self method same-module   OK  (was runtime SEGV)
R6 associated same-module     OK  (was runtime SEGV)
get after a.add(&b)           OK  (was compile SEGV)
Epistemic::measured import    OK
multi-module e.val()          still compile SEGV (residual)
```

## Gate

```bash
bash scripts/madaros_root2_method_gate.sh
# → MADAROS_ROOT2_METHOD_GATE_OK
```

Requires current-source Madaros (`make build-madaros` / CI modular build).

## claims_not_made

- Multi-module instance method call (`x.method()` across `use`)  
- Enum ctor path / full Root 2 census closed  

## Priority next

Multi-module method call at seed lower (import-side `E_get` resolution /
`first_param_is_ref` on merged stubs).
