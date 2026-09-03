<!-- docs:meta
topic_id: repo.docs.handoff.blk-20260804-p0b-qd128-native-v2
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.blk-20260804-p0b-qd128-native-v2
-->

# Blocker: BLK-20260804-p0b-qd128-native-v2

```text
Blocker-ID: BLK-20260804-p0b-qd128-native-v2
Status: closed (2026-08-04)
Severity: B1 (was)
Class: compiler-native / stdlib-reshape
Owner: cursor--p0b3-qd-mul-native-v2-20260804
Lane: p0b3-qd-mul-native-v2-20260804
Worktree: /tmp/sounio-p0b3-qd-mul-20260804
Branch: research/p0b3-qd-mul-native-v2-20260804
Closed-By: stdlib-only reshape of qd_nine_two_sum / qd_nine_one_sum to take [f64; 9]
Root-Cause: Madaros native-v2 rejects IrCall with 9+ scalar f64 args from *imported*
  module function bodies (same-file / main-unit calls and ≤8 float scalars succeed).
Acceptance-Gate: scripts/ci/madaros_qd128_mul_native_v2_gate.sh
  + scripts/ci/zero_event_native_v2_matrix.sh (qd128 green; combined still fail-closed on stock Madaros)
Evidence-Level: E3
LLM-Offload: not-required (mechanical ABI reshape; HLB arithmetic unchanged)
Residual: combined sedenion+eisa zero-provenance thin-link `rc=12` on stock Madaros (lean_single green)
Next-Action: none for qd_mul; zero-provenance multi-import remains a separate BLK/lane
```

## Context

`math::qd128_core` constructors closed earlier. Full `math::qd128` including
`qd_mul` now compiles and runs under default Madaros after packing the nine-term
helpers into a single array argument (same pattern family as sedenion’s
array-ref CD helpers for high-arity float calls). Combined sedenion+eisa
zero-provenance remains fail-closed under stock Madaros thin-link (`rc=12`) as of
shepherd-merge onto `origin/main` 2026-08-05; lean_single still prints
`ZERO_PROVENANCE PASS`.
