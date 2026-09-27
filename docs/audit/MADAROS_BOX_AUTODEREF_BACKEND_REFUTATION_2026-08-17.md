<!-- docs:meta
topic_id: repo.docs.audit.madaros-box-autoderef-backend-refutation-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-box-autoderef-backend-refutation-2026-08-17
-->

# Madaros Box Auto-Deref Backend Refutation

Date: 2026-08-17

Status: **fifth cause refuted**. Propagating `IrFieldGet.label_id` through
Machine IR cannot repair the `Box<T>` auto-deref miscompile.

## Instrument

- Source base: `c5754c0c847aea3a90e6e6a4e8a430c9f965e7b6`
- Source-built Madaros SHA-256:
  `785aad0ec3a1c02aafc27ec077eec8bca991d13b4a284d04600fec49dd219609`
- Build command:

  ```sh
  ulimit -s 524288
  env -u SOUC_BIN -u SOUNIO_SOUC_BIN \
    bash scripts/ci/build_modular_madaros.sh \
    /tmp/box-autoderef-evidence/madaros-baseline
  ```

- Witness: `tests/run-pass/box_all_read_forms.sio`
- Baseline result: check `rc=0`; normal execution `rc=139`; native-v2
  execution `rc=139`.

The native-v2 IR trace was captured with:

```sh
SOUNIO_NV2_IR_TRACE=1 \
  /tmp/box-autoderef-evidence/madaros-baseline \
  --native-v2-compile tests/run-pass/box_all_read_forms.sio \
  /tmp/box-autoderef-evidence/box-matrix-trace.elf
```

## Observation

The IR shapes are already different before Machine IR:

| Source expression | IR emitted for the function body |
|---|---|
| `b.tag`, `b: Box<Ex>` | `field_get src=b field_idx=52` |
| `(*b).tag` | `field_get src=b field_idx=0`; `field_get field_idx=2` |
| `let s = b.span; s.a` | `field_get src=b field_idx=51`; `field_get field_idx=0` |
| `let s = (*b).span; s.a` | `field_get src=b field_idx=0`; `field_get field_idx=1`; `field_get field_idx=0` |

The auto forms are missing the Box payload read at index 0 and use fallback
indices 52 and 51 instead of `Ex.tag=2` and `Ex.span=1`. The explicit forms
contain the required extra read and the correct `Ex` layout indices.

## Why `label_id` Is Not Causal

`self-hosted/ir/lower.sio` marks a binding as `is_ref` only when its type is
`TypeReference` or `TypeRefMut`. A `Box<Ex>` parameter is `TypeNamed`, so its
`is_ref` slot is zero. Field access assigns `instr.label_id = 1` only when that
slot is set. Consequently, the broken `b.tag` instruction does not carry the
reference marker that Machine IR is accused of dropping.

The transport loss is real: `self-hosted/native/machine_ir.sio` converts
`IrFieldGet` into `MIR_OP_PSEUDO_FIELD_GET` using only `dst`, `src1`, and
`field_idx`. Its legalization then performs exactly one field load with that
index. The direct native backend likewise loads `[base + field_idx * 8]` and
does not interpret `label_id` for a field read.

But preserving a value that is absent cannot add the missing payload read, and
it cannot transform fallback index 52 into layout index 2. The miscompile is
therefore established in expression lowering, before either backend.

## Verdict

**REFUTED:** "The Box auto-deref miscompile is caused by `label_id` being
dropped at the IR-to-MIR boundary."

The backend omission may be a separate contract defect for genuine `&T` field
access, but it is not the driver of this `Box<T>` failure. The eventual repair
must retain the inner `T` layout at binding time and emit an explicit
`field_get(base, 0)` before applying the field index resolved against `T`.

The run-pass witness also contains the no-field control that invalidated four
earlier implementations: pass a `Box<Ex>` to a function, then construct an
unrelated `Ex`. Any metadata implementation that copies the full lowerer while
reading its side table will fail that control even if the five read forms are
repaired.
