<!-- docs:meta
topic_id: repo.docs.handoff.d9-current-main-d4-ast-closure-regression-2026-07-19
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.d9-current-main-d4-ast-closure-regression-2026-07-19
-->

# D9 Current-Main D4 AST Closure Regression Handoff

Date: 2026-07-19

This is an integration blocker discovered while recursively validating D8-D0
from the D9 worktree after merging current `origin/main`. D9 does not own or
modify the compiler, resolver, D4 kernel, or D4 gate.

```text
Blocker-ID: BLK-20260719-D9-D4-CURRENT-MAIN-AST-CLOSURE
Status: classified
Severity: B2
Class: gate-regression
Owner: codex-2
Lane: D9 post-D8 current-main integration
Worktree: /tmp/sounio-psychiatric-d9-20260719
Branch: codex/psychiatric-d9-statistical-binding-20260719
Files-Owned: none assigned by D9; compiler owner chooses the repair surface
Files-Read-Only: stdlib/epistemic/proof_carrying_endogenous_observability.sio; scripts/ci/proof_carrying_endogenous_observability_gate.sh; bin/madaros-linux-x86_64
Do-Not-Touch: bin/souc; bin/madaros; bin/madaros-linux-x86_64; scripts/lib/resolve_souc.sh; self-hosted compiler sources from the D9 lane
Repro: run current-main bin/souc against the D8 copy of stdlib/epistemic/proof_carrying_endogenous_observability.sio
Observed: current-main Madaros reports raw AST closure does not contain the root source, raw AST parser reported failure, nodes=0, saturated=false
Expected: check: OK, as produced by the D8 head compiler over the byte-identical D4 source
Acceptance-Gate: bin/souc check stdlib/epistemic/proof_carrying_endogenous_observability.sio && bash scripts/ci/proof_carrying_path_conditioned_identification_gate.sh
Evidence-Level: E3
Evidence: this handoff plus the exact hashes and commands below
Fallback-Path: none; the D8 binary is a comparison control, not an accepted fallback
Legacy-Kept: yes; D0-D8 sources and all legacy ontology paths remain unchanged
LLM-Offload: not-required
Next-Action: codex-2 should compare AST closure-report generation between the two Madaros binaries and restore the D4 direct check on current main without weakening the closure gate
```

## Exact Comparison

The D4 source is byte-identical in both worktrees:

```text
5e4f0cdc7643b21f99d890d8318e9eba870a82a909db619bd4d45f90621d6336
stdlib/epistemic/proof_carrying_endogenous_observability.sio
```

D8 head `8e5cab45f` uses:

```text
fa3bcbcb5f72c6d3d851f97521f60dfd6a277ea7faff984d31a10ca377335c2e
bin/madaros-linux-x86_64
```

and returns:

```text
check: OK
```

The initial integration reproduction merged `origin/main` parent `d05f8069c`
into D8 as temporary commit `e69fe9726`. That compiler uses:

```text
99f6d955d6a07286f3c9653012bccb6ff8b50fbe87853b33121fc0e9a51979ef
bin/madaros-linux-x86_64
```

and returns:

```text
closure parser incomplete: raw AST closure does not contain the root source
closure parser incomplete: raw AST parser reported failure
run_check_mode: AST closure incomplete nodes=0 unresolved=0 saturated=false
```

The temporary merge was later removed from the stacked D9 history. A fresh
detached retest at live `origin/main` `37a5d9f0130d1e5c14f9e7bb108b7e9471941ec8`
used current-main `bin/souc` against the same absolute D8 source path. Its
Madaros binary SHA-256 is:

```text
11e7730f01f5382f1f8a5afc3599d7069b3d917f6972e6e47ffb57aa6bf4421e
bin/madaros-linux-x86_64
```

and it reproduces the same `nodes=0`, `unresolved=0`, `saturated=false`
AST-closure failure. Thus the blocker remains current; it is not inferred from
the removed merge or from a stale remote-tracking ref.

The recursive failure chain is D4 -> D5 -> D6 -> D7 -> D8. The focused D9 gate
itself passes on the current-main binary. This blocker therefore governs
retargeting and integration to current `main`; it does not justify changing D9
semantics or silently selecting the older compiler.
