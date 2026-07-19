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

Same-module method+associated now **PASS**. Residuals:

- Multi-module `x.method()` still **compile SEGV** at `lower_array: seed_begin`
- Same-module chain `let s = a.add(&b); s.get()` still **compile SEGV** (method on
  result of method); free helper `e_val(&s)` works

Free-function API remains the robust cross-module path.

## Evidence (rebuilt Madaros)

```
R1 free multi-module          OK
R2 E::of multi-module         OK  (was runtime SEGV)
R5 &self method same-module   OK  (was runtime SEGV)
R6 associated same-module     OK  (was runtime SEGV)
Epistemic::measured import    OK
of+get+add (no get-after-add) OK
multi-module e.val()          still compile SEGV (residual)
get after method-return       still compile SEGV (residual)
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
- Lean_single-only paths as sole witness  

## Priority next

Multi-module method call at seed lower (import-side `E_get` resolution /
`first_param_is_ref` on merged stubs).
