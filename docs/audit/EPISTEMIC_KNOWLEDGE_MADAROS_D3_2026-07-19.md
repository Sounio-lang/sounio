<!-- docs:meta
topic_id: repo.docs.audit.epistemic-knowledge-madaros-d3-2026-07-19
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.epistemic-knowledge-madaros-d3-2026-07-19
-->

# Epistemic Knowledge under default Madaros (D3 partial) — 2026-07-19

## Claim

> The flagship **`Epistemic` free-function API** and **method form**
> (`Epistemic::measured` / `e.val()` / `e.add`) import and run correctly under
> **default Madaros multi-module** (no lean_single pin). Method residual closed
> (Wave9; Root 2 multi-module methods + free/method parity gate).

## Root cause (re-diagnosed)

| Observation | Result |
|---|---|
| Tiny free-function module import | OK |
| `impl Epistemic { … }` methods **defined, never called** | OK |
| Any **method call site** (`Epistemic::measured`, `e.val()`) | historically **SEGV** in `lower_array: seed_begin`; **now green** |
| Free-function call sites (`ep_measured`, `ep_val`) | OK under multi-module |
| Free vs method numeric parity | OK under multi-module |

Historical cause was Madaros **method-call lowering** (Root 2 in
`MADAROS_SRET_ROOT_SYNTHESIS_2026-06-20`), closed by multi-module method
preseed + related Root 2 work (`MADAROS_ROOT2_MULTIMODULE_METHOD_2026-07-19`).

Name clash: do **not** export a free function named `measure` — collides with the
language `Knowledge<T>` builtin (`found &Knowledge<f64>`).

## What landed

### 1. Free-function surface (`stdlib/epistemic/knowledge.sio`)

`ep_new`, `ep_measured`, `ep_certain`, `ep_val`, `ep_variance`, `ep_confidence`,
`ep_std`, `ep_add`/`sub`/`mul`/`div`, `ep_scale`/`shift`/`square`/`sqrt_ep`,
`ep_gate`, `ep_merge`, …

Thin `impl Epistemic` wrappers call the free functions; method call sites under
Madaros multi-module are **TRUSTWORTHY** (Wave9 residual closeout).

Selftests use **free functions only**; avoid `print(i64)` under Madaros.

### 2. Gates

```bash
bash scripts/epistemic_knowledge_madaros_e2e_gate.sh
# → EPISTEMIC_KNOWLEDGE_MADAROS_E2E_GATE_OK

bash scripts/epistemic_trust_gate.sh
# Section A includes knowledge free + method API

bash scripts/madaros_knowledge_method_residual_gate.sh
# → MADAROS_KNOWLEDGE_METHOD_RESIDUAL_GATE_OK
```

### 3. Residual (closed / remaining)

| Path | Status |
|---|---|
| Free-function import | **TRUSTWORTHY** under Madaros |
| Method-call import | **TRUSTWORTHY** under Madaros (Wave9 closeout) |
| Free vs method parity | **TRUSTWORTHY** — `knowledge_method_parity.sio` |
| `order_spread` / `propagate` free leaves | **TRUSTWORTHY** (later promotions) |
| Language `Knowledge<T>` / `measure` | separate from stdlib `Epistemic` |
| gum k95 f64→i64 cast | still corrupted under native import (trust map §B) |

## Measured (Madaros v0.80.0)

```
─── 32/32 PASS ───
ALL PASS
KNOW_IMPORT v=10 sum=14 prod=40 merge=4.827586
KNOWLEDGE_MADAROS_IMPORT_E2E_OK
```

## claims_not_made

- Language generic `Knowledge<T>`  
- Full Root 2 census closed (enum ctor paths, etc.)  
- gum k95 f64→i64 cast fixed  
- Bedside / numpy  

## Priority next (compiler)

Other Madaros native-import defects (D1 gum k95 cast, residual census items).
Method form residual is closed — keep `madaros_knowledge_method_residual_gate.sh` green.
