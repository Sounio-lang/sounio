<!-- docs:meta
topic_id: repo.docs.audit.madaros-root2-method-chain-2026-07-21
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-root2-method-chain-2026-07-21
-->

# Madaros Root 2 — inline method-chain residual closeout — 2026-07-21

## Claim

> Under **default Madaros**, **inline method chains** on both same-module types and
> **imported** science-path types compile and run correctly:
>
> - `E::of(3.0).get()`
> - `a.add(&b).get()` / `a.add(&b).add(&c).get()`
> - multi-module `Epistemic::certain(1.0).val()`, `a.add(&b).val()`,
>   `a.add(&b).add(&c).val()`, `print(a.add(&b).std())`

Stepwise bind (`let t = a.add(&b); t.get()`) was already green after #1219 / #1227.
Inline chains were the remaining Root-2 method residual measured post-#1392.

## Root cause

`lower_method_recv_type` only typed **Ident** and **Index** receivers. A receiver
that was itself an `ExprMethodCall` or `ExprCall` returned the empty name, so the
outer call mangled to bare `"get"` / `"add"` / `"val"`. The seed lower then
invented a body-less function and **SEGV'd at `lower_array: seed_begin`**.

## Fix

In `self-hosted/ir/lower.sio` → `lower_method_recv_type`:

| Receiver kind | Resolution |
|---|---|
| `ExprMethodCall` | Recurse for inner recv type → mangle → `return_struct_name_for_fn_id`; fallback to inner recv type (Self-returning methods) |
| `ExprCall` | `expr_call_return_struct_name_ref` (associated `Type::method` constructors) |
| `ExprIdent` / `ExprIndex` | unchanged |

Does **not** touch Wave13 paths (`module_frontend.sio`, `parser/items.sio`).

## Gate

```bash
bash scripts/madaros_root2_method_gate.sh
# → MADAROS_ROOT2_METHOD_GATE_OK

bash scripts/madaros_root2_multimodule_method_gate.sh
# → MADAROS_ROOT2_MULTIMODULE_METHOD_GATE_OK
```

Requires current-source Madaros (chain fixtures fail on pre-fix stock ELFs).

## Fixtures

| File | Sentinel |
|---|---|
| `tests/run-pass/madaros_root2_method_chain.sio` | `ROOT2_METHOD_CHAIN_OK` |
| `tests/run-pass/madaros_root2_multimodule_method_chain.sio` | `ROOT2_MULTIMODULE_METHOD_CHAIN_OK` |

## Measured (current-source Madaros)

```
ROOT2_METHOD_CHAIN_OK
ROOT2_MULTIMODULE_METHOD_CHAIN_OK
MADAROS_ROOT2_METHOD_GATE_OK
MADAROS_ROOT2_MULTIMODULE_METHOD_GATE_OK
```

## claims_not_made

- Full Root-2 census closed  
- Enum ctor path  
- Arbitrary-depth method-chain census beyond the fixtures above  
- Stock prebuilt ELF without rebuild (chain residual needs the lower fix in the running Madaros)
