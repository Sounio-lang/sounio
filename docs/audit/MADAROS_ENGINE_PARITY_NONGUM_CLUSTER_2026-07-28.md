<!-- docs:meta
topic_id: repo.docs.audit.madaros-engine-parity-nongum-cluster-2026-07-28
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-engine-parity-nongum-cluster-2026-07-28
-->

# Engine parity harvest — non-GUM cluster (2026-07-28)

**Branch tip:** FO cross-fn + field-name float unanimity  
**Binary:** `artifacts/self-hosted/madaros-value-name`

## Probe summary (FO-fixed Madaros vs lean_single)

### Madaros **FAIL** → fixed this wave

| Witness | Pre-fix | Root cause | Post-fix |
|---|---|---|---|
| `method_receiver_correct.sio` | FAIL (value bits = f64 1.0/2.0/12.0) | `field_is_float_by_name_simple("value")` first-match on Knowledge.value | **PASS** |

### Madaros-ahead (lean wrong / crash)

| Witness | Madaros | lean | Notes |
|---|---|---|---|
| `global_array_element_list_*` | LIST_OK / correct elems | wrong/repeated / exit 2 | Madaros element-list init |
| `let_var_binding_name.sio` | LET_VAR_BINDING_OK | FAIL | Madaros correct |
| most `imported_f64_*` | *_OK | *_OK | SEM_AGREE (print spacing) |

### Still open Madaros residuals

| Witness | Class | Notes |
|---|---|---|
| `ffi_integer_return.sio` | M_FAIL (pid=0) | lean injects getpid stub via `strip_extern_blocks`; Madaros modular path has no extern "C" stubs yet |
| `closure_escape.sio` / `closure_returned.sio` | M_CRASH empty | escaping closure env not heap-lifted on Madaros native path |
| `correlated_eq_identity.sio` | M_FAIL T1/T3/T5 | provenance identity FO incomplete on Madaros (independent path OK) |

### Print-only / SEM_AGREE

`basic_math`, `approx_basic`, `native_struct_*`, `mc_struct_*`, `global_scalar_*`, most closures that only print `0`.

## Fix: unanimous field-name float/int heuristics

**File:** `self-hosted/ir/lower.sio`

`field_is_float_by_name_simple` / `field_is_int_by_name_simple` previously returned on the **first** layout match. Knowledge is preseeded with `value: f64` (is_float=1), so any later `struct Counter { value: i64 }` inherited float ops: `c.value + 1` became f64 add and `println` printed IEEE bit patterns.

**Change:** only return true when **all** structs that share the field name agree (float-only or int-only). Ambiguous shared names fall through to `field_is_float_for_base_ref` / `field_is_int_for_base_ref` (typed base).

### Evidence

```text
# pre-fix
value=4607182418800017408   # 1.0 as i64 bits
method_receiver_correct: FAIL

# post-fix
value=1
method_receiver_correct: PASS
```

Knowledge FO path unchanged: gum_cross_function still PASS (sum=5, scaled=16); full GUM semantic suite 10/10.

## Gates

- `scripts/ci/madaros_method_receiver_gate.sh`
- existing `madaros_gum_cross_function_gate.sh` / `madaros_gum_semantic_suite_gate.sh`

## Next non-GUM targets

1. Madaros `extern "C"` integer stubs (getpid) — port lean strip_extern path  
2. Escaping closure heap env  
3. `eq_prob_correlated` identity (shared provenance V=0)
