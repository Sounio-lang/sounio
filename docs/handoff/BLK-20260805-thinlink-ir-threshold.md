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
Status: classified (2026-08-05) — shape residual, not fn-count ceiling
Severity: B2
Class: compiler-native / thin-link / struct-field-lowering
Owner: cursor--p0-thinlink-threshold-20260805
Lane: p0-thinlink-threshold-20260805
Worktree: /tmp/sounio-thinlink-threshold-20260805
Branch: research/thinlink-ir-threshold-20260805
Root-Cause: Stock Madaros native emit fails closed (thin-link rc=12, no
  segfault) when a struct literal initializes **two or more** `bool` fields
  from **f64 comparison expressions** in field position
  (`Pair { a: 2.0 > 0.0, b: 3.0 > 0.0 }`). Same CU with comparisons prebound
  to locals is green. lean_single oracle PASS. IR `final_fn_count` is **not**
  the discriminator: pad-to-49 i64 helpers on the compact zero-provenance
  smoke still emit; this 3-fn probe fails.
Acceptance-Gate: scripts/ci/madaros_thinlink_bool_cmp_field_gate.sh
Evidence-Level: E3
LLM-Offload: not-required (compiler residual classification; no new math claim)
Residual: compiler fix for f64-cmp → bool field init under native emit
Next-Action: compiler lane may reopen; stdlib/tests should prefer precomp
  locals for multi-bool struct literals until fixed. Do not cite “~41 fn
  ceiling” as the zero-provenance thin-link cause.
```

## Evidence ladder (2026-08-05, main @ 4fd0c48985)

| Case | Outcome |
|---|---|
| `Pair { a: 2.0 > 0.0, b: 3.0 > 0.0 }` | Madaros `rc=12`; lean_single PASS |
| Same with `let a = 2.0 > 0.0; let b = …; Pair { a: a, b: b }` | Madaros PASS |
| Single `Wrap { flag: 2.0 > 0.0 }` | Madaros PASS |
| Two **i64** comparisons in bool fields | Madaros PASS |
| Two f64 `==` in bool fields | Madaros `rc=12` |
| Compact zero-provenance + 8 pad fns (`final_fn_count` 49) | Madaros PASS |
| Fat `ZeroWitness` with `sed_norm_sq(…) > 0.0` in bool fields | Madaros `rc=12` (same shape family) |

```bash
bash scripts/ci/madaros_thinlink_bool_cmp_field_gate.sh
./bin/souc run tests/known_failures/thinlink_bool_cmp_field_probe.sio
# expect: Failed to write native binary … rc=12
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/known_failures/thinlink_bool_cmp_field_probe.sio
# expect: BOOL_CMP_FIELD PASS
```

## Relation to BLK-20260805-p0b-zero-provenance

The combined `eisa::core_v2`+sedenion probe (~111 fn) remains a **separate**
multi-module scale residual. The earlier “41→49 fn” reading of the fat
sedenion+`ZeroWitness` witness was a **misattribution**: that CU failed because
it embedded live f64 comparisons into bool struct fields, not because it
crossed an IR function-count ceiling.
