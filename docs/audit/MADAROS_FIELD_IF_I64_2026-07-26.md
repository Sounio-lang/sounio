# Madaros multimodule: i64 field load OK on return, wrong on if/sub

**Date:** 2026-07-26  
**Status:** residual open (stdlib workarounds live; native fix blocked by active claims on `self-hosted/native/**` + `self-hosted/ir/lower.sio`)  
**Witness:** `tests/multimodule/madaros_field_if_i64_{leaf,main}.sio`  
**Gate:** `scripts/ci/madaros_field_if_i64_gate.sh`

## Measured matrix (current-source Madaros, 2026-07-26)

| Pattern | Result |
|---|---|
| `return e.confidence` | **846** OK |
| `let c = e.confidence; return c` | **846** OK |
| `e.confidence + 0` | **846** OK |
| `e.confidence - m` | **garbage** (pointer-scale i64, e.g. −4.57e18) |
| `if e.confidence >= m { 1 } else { 0 }` | **0** wrong |
| `let c = e.confidence; if c >= m` | **0** wrong |
| `ge(e.confidence, m)` call-arg | **1** OK |
| `ge(ret_conf(e), m)` | **1** OK |

Interpretation: the **load itself is not always wrong** — return and `+ 0` see 846.
Using the field value as the **condition of a branch** or as the **LHS of subtraction**
sees a wrong/pointer-like value. Passing the same field through a **call argument**
re-materialises a clean i64.

## Product impact (already mitigated)

| Surface | Mitigation |
|---|---|
| `ep_gate` / `ep_require_conf` / `ep_is_credible` | `ep_i64_ge(field, k)` (#1478) |
| `pb_is_credible` / `ck_is_credible` | same pattern (#1492) |
| EXP123 confidence gates 111/113 | closed via ep_gate |

## Suspected fix surface (for owner of native lane)

Priority order for forensics:

1. **Native branch condition materialisation** for SSA values produced by `IrFieldGet` of i64  
   (`self-hosted/native/codegen_x86_linux.sio` — active claim elsewhere)
2. **Imported multimodule finalize / restore** of field-get results used as cmp operands  
   (`self-hosted/compiler/module_frontend.sio` finalize path)
3. **IR lower of `if` with field-loaded condition**  
   (`self-hosted/ir/lower.sio` — active FO claim elsewhere)

Acceptance for close:

```text
MADAROS_FIELD_IF_I64_FIXED
# requires: gate_field=1 gate_let=1 sub=46 via_arg=1 ret=846
```

## Reproduction

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
export MADAROS_RAW_BIN=artifacts/self-hosted/madaros   # or bin/madaros-linux-x86_64
bash scripts/ci/madaros_field_if_i64_gate.sh
# expect: PASS: RESIDUAL documented  until native fix
```

## Coordination note

2026-07-26: `self-hosted/native/codegen_x86_linux.sio` held by claude co-own lane;
`self-hosted/ir/lower.sio` held by fo-transcendental. This audit + witness ship
without touching those files. Handoff ready for the native owner.
