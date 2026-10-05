<!-- docs:meta
topic_id: repo.docs.handoff.blk-20260805-p0b-zero-provenance
authority: repo_only
audience: users
last_validated: 2026-09-13
validated_by: cursor-kl12
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.blk-20260805-p0b-zero-provenance
-->

# Blocker: BLK-20260805-p0b-zero-provenance

```text
Blocker-ID: BLK-20260805-p0b-zero-provenance
Status: closed (2026-09-13) — KL-12; same root cause as bool-cmp-in-field
Severity: B2
Class: compiler-native / multimodule-thin-link
Owner: cursor--p0b-zero-prov-20260805
Lane: p0b-zero-provenance-madaros-20260805
Worktree: /workspace/wt-kl12
Branch: feat/kl12-zeroprov
Root-Cause: Combined sedenion + eisa::core_v2 CU embeds f64 comparisons into
  ZeroWitness bool fields (`sed_norm_sq(…) > 0.0`, `x != 0.0`). That stamped
  declared bool slots as float (`is_float=1`), so native-v2 refused `main`
  (NV2_IR unsupported fn=main / thin-link rc=12). Fixed in KL-12 layout
  honesty (`lower_struct_field_float_kind_ref` + no overwrite of is_float=3/4).
Acceptance-Gate: scripts/ci/madaros_zero_provenance_failclosed_gate.sh (expects PASS)
Evidence-Level: E3
LLM-Offload: not-required
Residual: none for this combined import.
Next-Action: none. Compact smoke remains a smaller CU, not a substitute claim.
```

## Evidence commands (post-fix)

```bash
# Madaros combined — expect ZERO_PROVENANCE PASS
./bin/souc run tests/run-pass/zero_provenance_native_v2_combined.sio

# lean_single oracle — expect ZERO_PROVENANCE PASS
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/run-pass/zero_provenance_native_v2_combined.sio

bash scripts/ci/madaros_zero_provenance_failclosed_gate.sh
```

## Context

The compact Madaros-green smoke
`tests/run-pass/zero_provenance_native_v2_smoke.sio` (~41 fn, no
`eisa::core_v2`) remains a distinct, smaller CU. The earlier “~111 fn /
multimodule scale” reading of this BLK was a **misattribution**: the fail
locus was `main` / `ZeroWitness` bool fields from f64 cmps — the same shape
as `BLK-20260805-thinlink-ir-threshold`.
