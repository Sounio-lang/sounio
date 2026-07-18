<!-- docs:meta
topic_id: repo.docs.handoff.d6-multimodule-runtime-blocker-2026-07-18
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.d6-multimodule-runtime-blocker-2026-07-18
-->

# D6 Multimodule Runtime Blocker Handoff

The D6 scientific lane is intentionally narrowed to a compiler-checked reusable
kernel, an executed scalar native witness, and an independent exhaustive
oracle. This blocker belongs to the separate Madaros multimodule runtime lane;
it does not convert the scalar witness into canonical imported runtime evidence.

```text
Blocker-ID: BLK-20260718-D6-MULTIMODULE-RUNTIME
Status: proposed
Severity: B1
Class: compiler-semantics
Owner: codex-2
Lane: Madaros multimodule imported runtime
Worktree: /tmp/sounio-psychiatric-mainline-20260717
Branch: codex/psychiatric-mainline-d0-d2-20260717
Files-Owned: none; diagnosis and handoff only until codex-2 declares an exact compiler write set
Files-Read-Only: stdlib/epistemic/proof_carrying_policy_observation_associator.sio; tests/run-pass/clinical_proof_carrying_policy_observation_associator_witness.sio
Do-Not-Touch: bin/souc; scripts/lib/resolve_souc.sh; scripts/run_sio_test_suite.sh from the D6 scientific lane
Repro: bin/souc run tests/run-pass/clinical_proof_carrying_policy_observation_associator_witness.sio
Observed: compact modular IR emission fails, full-IR thin-link returns rc=12, and no executable reaches main
Expected: imported multimodule compilation writes an ELF, executes main, and returns zero
Acceptance-Gate: bin/souc run tests/run-pass/clinical_proof_carrying_policy_observation_associator_witness.sio
Evidence-Level: E2
Evidence: /tmp/d6-multimodule-runtime-20260718.log
Fallback-Path: full IR was attempted automatically and also failed; the scalar witness is separate evidence, not a canonical runtime fallback
Legacy-Kept: yes; imported kernel and scalar witness are both retained with distinct evidence roles
LLM-Offload: not-required for diagnosis; required according to policy before any math-bearing compiler fix
Next-Action: codex-2 reproduces the command, declares its exact compiler write set, and isolates the thin-link failure without editing D6 scientific files
```

The native scalar run also reports backend dispatch
`source=fallback fallback=unresolved_default_x86_64_linux`. The D6 gate proves
Madaros engine selection without engine fallback; it does not prove an
unqualified zero-fallback backend path.
