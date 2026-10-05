<!-- docs:meta
topic_id: repo.docs.audit.sedenion-zd-action-2026-08-23
authority: repo_only
audience: users
last_validated: 2026-08-23
validated_by: grok-cli2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.sedenion-zd-action-2026-08-23
-->

# Sedenion left action — canonical zero-divisor annihilation

```text
Semantic-Lane-ID: sedenion-zd-action-20260823
Owner: grok-cli2
Concept-IDs: SOUNIO-NONASSOCIATIVE-ORDER
Intent-Preserved: parenthesization remains explicit where the algebra is
  nonassociative; a zero divisor is not a physical ontology; this is not
  a consciousness allegory
Transformation: expose L_z(w)=0 for the canonical pair as an action
  protocol; composed (zw)v vanishes for every v; sequential z*(w*v)
  need not; octonion embedding (o,0) cannot annihilate
Types-Changed: none
Effects-Changed: none (sed_mul is Mut-only; no new ZD effect)
IR-Changed: none
Claims-Introduced: Sounio computes z*w=0 with ||z||²=||w||²=2 on the
  canonical pair; composed action on e1 vanishes; sequential does not;
  embedded e1*e2 has norm² 1
Claims-Forbidden: new physics; sedenion consciousness; ZD solves g-2;
  Sounio proved a new theorem beyond Cayley-Dickson; NonAssoc is Mut;
  Madaros is fixed-point-verified
Assumptions: Convention X in algebra::sedenion; canonical pair
  z=e3+e10, w=e6−e15 as documented there; octonion subalgebra is
  the first half via sed_from_pair
Write-Set: stdlib/algebra/sedenion_action.sio,
  examples/physics/sedenion_zd_action.sio,
  tests/run-pass/sedenion_zd_action.sio, this file
Read-Set: stdlib/algebra/sedenion.sio,
  stdlib/algebra/octonion_action.sio,
  docs/audit/OCTONION_ACTION_R7_2026-08-23.md
Positive-Witness: souc run prints ZD_PRODUCT_NSQ 0, sequential > 0,
  OCT_EMBED_PRODUCT_NSQ 1
Negative-Witness: octonion embedding of e1,e2 does not annihilate;
  g-2 pins are not this lane
Acceptance-Gate: tests/run-pass/sedenion_zd_action.sio exit 0 under
  Madaros; example exit 0
Integration-Target: origin/main
Authoritative-Only-If: the nsqs are produced by Madaros running Sounio
```

## What was already there

`sed_mul` and `sed_zd_z` / `sed_zd_w` already exist. Inline tests in
`sedenion.sio` check the product. `octonion_action` is the two-protocol
story for O. Conversational OSSM reads ZD *proximity*; this lane is
not that.

## What this lane does

It names two protocols on a vector in S ≅ R¹⁶:

1. compose then act: \(L_{ab}(v)=(ab)v\)
2. act then act: \((L_a\circ L_b)(v)=a(bv)\)

On the canonical pair, (1) annihilates because \(zw=0\). (2) is the
falsifier: if it also vanishes for the probe \(v=e_1\), the
“composed-only annihilation” claim is dead. The octonion embedding is
the division-algebra control.

## Receipts (2026-08-23)

Control ELF `/workspace/sounio/bin/madaros-linux-x86_64` (100902241 B).

Madaros `souc check` verdict=0. Example and test **rc=0**.

```
ZD_Z_NSQ 2.000000
ZD_W_NSQ 2.000000
ZD_PRODUCT_NSQ 0.000000
ZD_COMPOSED_NSQ 0.000000
ZD_SEQUENTIAL_NSQ 8.000000
OCT_EMBED_PRODUCT_NSQ 1.000000
OCT_EMBED_COMPOSED_NSQ 1.000000
OCT_EMBED_SEQUENTIAL_NSQ 1.000000
OCT_EMBED_DISCREPANCY_NSQ 4.000000
VERDICT_ZD_ANNIHILATES_COMPOSED 1
VERDICT_ZD_SEQUENTIAL_SURVIVES 1
VERDICT_OCT_EMBED_DIVIDES 1
VERDICT_OCT_EMBED_ASSOCIATOR 1
```

Composed annihilation is \(zw=0\), so \((zw)e_1=0\). Sequential
\(z*(w*e_1)\) has norm² 8. The octonion embedding of \((e_1,e_2,e_4)\)
keeps the associator: discrepancy norm² 4, both protocols length 1.

LLM-offload math-review (`/tmp/llm-offload-cOpBlg`):

- xAI grok-4.3: five `[OK]` (canonical pair, composed annihilation,
  sequential witness, octonion embedding, associator 4).
- Z.AI: weekly quota exhausted until 2026-08-25.
- Outcome: **PASS_SINGLE_PROVIDER_DEGRADED**.
