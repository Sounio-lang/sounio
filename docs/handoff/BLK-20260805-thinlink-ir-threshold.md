<!-- docs:meta
topic_id: repo.docs.handoff.blk-20260805-thinlink-ir-threshold
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.blk-20260805-thinlink-ir-threshold
-->

# Blocker: BLK-20260805-thinlink-ir-threshold

```text
Blocker-ID: BLK-20260805-thinlink-ir-threshold
Status: closed (2026-09-13) — KL-12 bool-cmp-in-field; zero-provenance remains separate
Severity: B2
Class: compiler-native / thin-link / struct-field-lowering
Owner: cursor--p0-thinlink-threshold-20260805
Lane: p0-thinlink-threshold-20260805
Worktree: /workspace/wt-kl12
Branch: feat/kl12-thinlink
Root-Cause: `ensure_struct_literal_layout_ref` overwrote declared bool
  (`is_float=3`) slots with `is_float=1` when the initializer was an f64
  comparison (`lower_struct_field_float_kind_ref` treated float-operand
  cmps as float fields). Later `field_get` stamped
  `IR_FLOAT_REG_MARKER_FLAG` and `p.a && p.b` became a float binop that
  native-v2 refused (rc=12). Fix: comparisons/logic yield kind 0; do not
  overwrite declared integer/bool or pointer layout classes.
Acceptance-Gate: scripts/ci/madaros_thinlink_bool_cmp_field_gate.sh (expects PASS)
Evidence-Level: E3
LLM-Offload: not-required
Residual: none for bool-cmp-in-field. Zero-provenance fat CU remains under
  KL-12 / BLK-20260805-p0b-zero-provenance — do not cite “~41 fn ceiling”.
Next-Action: none for this BLK.
```

## Evidence ladder (2026-08-05, main @ 4fd0c48985)

| Case | Outcome (pre-fix) | Outcome (post KL-12) |
|---|---|---|
| `Pair { a: 2.0 > 0.0, b: 3.0 > 0.0 }` | Madaros `rc=12`; lean_single PASS | Madaros PASS; lean_single PASS |
| Same with `let a = 2.0 > 0.0; let b = …; Pair { a: a, b: b }` | Madaros PASS | Madaros PASS |
| Single `Wrap { flag: 2.0 > 0.0 }` | Madaros PASS | Madaros PASS |
| Two **i64** comparisons in bool fields | Madaros PASS | Madaros PASS |
| Two f64 `==` in bool fields | Madaros `rc=12` | Madaros PASS |
| Compact zero-provenance + 8 pad fns (`final_fn_count` 49) | Madaros PASS | Madaros PASS |
| Fat `ZeroWitness` with `sed_norm_sq(…) > 0.0` in bool fields | Madaros `rc=12` (same shape family) | closed with bool-cmp fix |

```bash
bash scripts/ci/madaros_thinlink_bool_cmp_field_gate.sh
./bin/souc run tests/run-pass/thinlink_bool_cmp_field.sio
# expect: BOOL_CMP_FIELD PASS
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/run-pass/thinlink_bool_cmp_field.sio
# expect: BOOL_CMP_FIELD PASS
```

## Relation to BLK-20260805-p0b-zero-provenance

The combined `eisa::core_v2`+sedenion probe (~111 fn) remains a **separate**
multi-module scale residual. The earlier “41→49 fn” reading of the fat
sedenion+`ZeroWitness` witness was a **misattribution**: that CU failed because
it embedded live f64 comparisons into bool struct fields, not because it
crossed an IR function-count ceiling. The bool-cmp shape is closed; the fat
combined import is not.
