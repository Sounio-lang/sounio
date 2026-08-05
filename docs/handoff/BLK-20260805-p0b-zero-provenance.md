<!-- docs:meta
topic_id: repo.docs.handoff.blk-20260805-p0b-zero-provenance
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.blk-20260805-p0b-zero-provenance
-->

# Blocker: BLK-20260805-p0b-zero-provenance

```text
Blocker-ID: BLK-20260805-p0b-zero-provenance
Status: waived-E3 (2026-08-05) — fail-closed classified; not Madaros-green
Severity: B2
Class: compiler-native / multimodule-thin-link
Owner: cursor--p0b-zero-prov-20260805
Lane: p0b-zero-provenance-madaros-20260805
Worktree: /tmp/sounio-p0b-zero-prov-20260805
Branch: research/p0b-zero-provenance-madaros-20260805
Root-Cause: Combined import of algebra::sedenion + eisa::core_v2 (transitive
  dd64/qd128) yields ~5 modules / ~111 lowered fns; stock Madaros native emit
  fails closed with thin-link rc=12 after successful typecheck+lower. No segfault.
  Alone: sedenion smoke ~41 fn green; eisa-only ~65 fn green; lean_single combined PASS.
Acceptance-Gate: scripts/ci/madaros_zero_provenance_failclosed_gate.sh
  + scripts/ci/zero_event_native_v2_matrix.sh (combined expect fail-closed)
Evidence-Level: E3
LLM-Offload: not-required (compiler residual classification; no new math claim)
Residual: thin-link scale for this specific multi-algebra import union
Next-Action: compiler lane may reopen as thin-link/D3-scale work; do not promote
  to run-pass or TRUSTWORTHY under Madaros until rc=0 + ZERO_PROVENANCE PASS
```

## Evidence commands (2026-08-05, main @ 86b6a79e56)

```bash
# Madaros combined — expect rc=1, marker Failed to write native binary, no segfault
./bin/souc run tests/known_failures/zero_provenance_native_v2_probe.sio

# lean_single oracle — expect ZERO_PROVENANCE PASS
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/known_failures/zero_provenance_native_v2_probe.sio

# Alone controls
bash scripts/ci/madaros_sedenion_native_v2_gate.sh
# eisa-only minimal: ereg2_exact/eadd2/esub2 → EISA_ONLY PASS under Madaros
```

## Honest claim boundary

Do **not** claim combined sedenion+eisa zero-provenance under default Madaros.
Do claim: fail-closed without crash; semantic oracle under lean_single; component imports green.
