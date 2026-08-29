<!-- docs:meta
topic_id: repo.docs.audit.madaros-root2-multimodule-method-2026-07-19
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-root2-multimodule-method-2026-07-19
-->

# Madaros Root 2 multi-module instance methods — 2026-07-19

## Claim

> Under **default Madaros**, instance methods on types defined in **imported**
> modules (`x.get()`, `e.add(&f)`, `print(e.val())`) compile and run correctly.

## Root causes fixed (this PR)

| Bug | Symptom | Fix |
|---|---|---|
| External preseed skipped `impl` methods | Seed module invented body-less `"get"`; compile SEGV / wrong ABI | `lowerer_preseed_external_struct_items_mut` also preseeds mangled `Type_method` signatures (`first_param_is_ref`, `returns_float`, `return_struct_name`) |
| `expr_result_is_float_ref` ignored method calls | `print(e.val())` treated f64 as char* → runtime SEGV | Classify method results via mangled fn `returns_float` |

Depends on #1219 (path mangling, auto-OpRef, method-let struct bind).

## Gate

```bash
bash scripts/madaros_root2_multimodule_method_gate.sh
# → MADAROS_ROOT2_MULTIMODULE_METHOD_GATE_OK
```

Requires current-source Madaros.

## Measured

```
10.000000
14.000000
KNOW_METHOD_MULTI_OK
ROOT2_MULTIMODULE_METHOD_OK
```

## claims_not_made

- Enum ctor paths / every Root 2 census item  
- lean_single-only as sole evidence  
