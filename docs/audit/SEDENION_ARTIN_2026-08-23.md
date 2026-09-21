<!-- docs:meta
topic_id: repo.docs.audit.sedenion-artin-2026-08-23
authority: repo_only
audience: users
last_validated: 2026-08-23
validated_by: grok-cli2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.sedenion-artin-2026-08-23
-->

# Sedenion Artin probe — alternativity fails; octonion embedding holds

```text
Semantic-Lane-ID: sedenion-artin-20260823
Owner: grok-cli2
Concept-IDs: SOUNIO-NONASSOCIATIVE-ORDER
Intent-Preserved: parenthesization remains explicit; Artin is an
  algebraic identity of O, not a physical ontology; this is not a
  consciousness allegory
Transformation: name [a,a,v] as the same two left-multiplication
  protocols with a=b; pin a=e1+e10, v=e4 as a Madaros witness with
  norm² 4; keep the octonion embedding as the vanishing control
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: Sounio computes [e1+e10, e1+e10, e4] with nsq 4;
  the octonion Artin pair embedded as (o,0) still has nsq 0
Claims-Forbidden: new physics; sedenion consciousness; ZD solves g-2;
  Sounio proved a classification of alternative subalgebras;
  Madaros is fixed-point-verified
Assumptions: Convention X; Artin for O is [a,a,v]=0 (already a
  control in octonion_action); the witness was found by enumerating
  a=ei+e_{8+k}, v=ej under Madaros, not by citation
Write-Set: stdlib/algebra/sedenion_action.sio,
  examples/physics/sedenion_artin.sio,
  tests/run-pass/sedenion_artin.sio, this file
Read-Set: stdlib/algebra/octonion_action.sio,
  docs/audit/SEDENION_ZD_ACTION_2026-08-23.md
Positive-Witness: souc run prints ARTIN_SED_NSQ 4 and
  ARTIN_OCT_EMBED_NSQ 0
Negative-Witness: octonion embedding Artin remains 0; g-2 is not
  this lane
Acceptance-Gate: tests/run-pass/sedenion_artin.sio exit 0 under Madaros
Integration-Target: origin/main
Authoritative-Only-If: the nsqs are produced by Madaros running Sounio
```

## What this is

Octonion action already used Artin as a vanishing control.
Sedenions are not alternative. The same protocol must not vanish
identically. Witness (Madaros scan): \(a=e_1+e_{10}\), \(v=e_4\),
norm² of \([a,a,v]\) is 4. The first \(e_1+e_8\) candidate was
Artin-zero; it is not used.

## Receipts (2026-08-23)

Control ELF `/workspace/sounio/bin/madaros-linux-x86_64` (100902241 B).

```
ARTIN_SED_NSQ 4.000000
ARTIN_OCT_EMBED_NSQ 0.000000
VERDICT_SED_ARTIN_FAILS 1
VERDICT_OCT_EMBED_ARTIN_HOLDS 1
```

Example and test **rc=0**.

LLM-offload math-review (`/tmp/llm-offload-yC5iSj`):

- xAI grok-4.3: three `[OK]` (octonion Artin, witness nsq 4, k=4 break).
- Z.AI: weekly quota exhausted until 2026-08-25.
- Outcome: **PASS_SINGLE_PROVIDER_DEGRADED**.
