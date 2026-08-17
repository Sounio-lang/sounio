<!-- docs:meta
topic_id: repo.docs.audit.madaros-field-if-i64-2026-07-26
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-field-if-i64-2026-07-26
-->

# Madaros multimodule: i64 field load OK on return, wrong on if/sub

**Date:** 2026-07-26  
**Status:** **CLOSED** (#1511 lower.sio + workaround drop)  
**Witness:** `tests/multimodule/madaros_field_if_i64_{leaf,main}.sio`  
**Gate:** `scripts/ci/madaros_field_if_i64_gate.sh` → `MADAROS_FIELD_IF_I64_FIXED`

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

## ROOT CAUSE (confirmed)

`self-hosted/ir/lower.sio` — `ir_register_knowledge_layout`:

```text
fields[2] confidence  is_float: 1   // WRONG — confidence is i64
// should be is_float: 3  (integer marker)
```

`field_is_float_by_name_simple("confidence")` walks **all** layouts and returns
true if **any** field named confidence has `is_float==1`. So MiniEp.confidence
and Epistemic.confidence FieldGets get `IR_FLOAT_REG_MARKER_FLAG` even when the
real type is i64. Then `OpGe`/`OpSub` take the float path in codegen.

**Fix:** `is_float: 3` for confidence (landed in this lane's lower.sio).

## Smoking gun (evidence)

Measured `sub_conf(&e, 800)` under Madaros:

```
printed i64 = -4573123946618028032
bit pattern = 0xc089000000000000
as f64      = -800.0   exactly
```

So the binop path is doing **float** `0.0 - 800.0` (or equivalent), not integer
`846 - 800`. That fits:

1. field value of `confidence` is seen as **0.0** on the binop path, and  
2. `OpSub` / `OpGe` take the **float** arm of `nc_emit_core_binop`.

Integer `return e.confidence` still prints **846** — so not every use of the
field is broken; return vs binop diverge.

### Where to look first

| Site | File (approx) | Why |
|---|---|---|
| `IrFieldGet` emit + float mark | `codegen_x86_linux.sio` ~7272–7286 | marks dst float via `nc_core_field_is_float` / imm_flags |
| float vs int binop | same file ~6788–6868 | `OpGe`/`OpSub` float override when reg typed float |
| `IrBranchTrue` | same file ~7577–7582 | `test rax,rax` — needs clean 0/1 from int cmp |
| ref vs handle FieldGet | `label_id == 1` branch at FieldGet | `&MiniEp` must use ref-field load |

### Hypotheses

**H1 (preferred):** `confidence` (i64 after two f64 fields) is wrongly
float-marked after FieldGet; binops reinterpret / zero; return path still
prints a prior integer materialisation of 846.

**H2:** multimodule FieldGet on `&T` misses `label_id=1`, handle-resolve yields 0
for some consumers; return uses another path.

### Acceptance

```text
MADAROS_FIELD_IF_I64_FIXED
# requires: gate_field=1 gate_let=1 sub=46 via_arg=1 ret=846
bash scripts/ci/madaros_field_if_i64_gate.sh
```

## Reproduction

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
export MADAROS_RAW_BIN=artifacts/self-hosted/madaros
bash scripts/ci/madaros_field_if_i64_gate.sh
# residual: PASS: RESIDUAL documented
# fixed:    PASS: FIXED
```

## Coordination

Product workarounds already on main (`ep_i64_ge`, `pb_i64_ge`, `ck_i64_ge`).
Do **not** drop them until FIXED is green. Founder asked Grok to help
Claude-1/2 — this audit is the intake package; native patch is theirs (or
shared if they free the claim).
