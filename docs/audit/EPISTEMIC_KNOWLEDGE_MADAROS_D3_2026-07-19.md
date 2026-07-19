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

> The flagship **`Epistemic` free-function API** imports and runs correctly under
> **default Madaros multi-module** (no lean_single pin). Method-call form remains
> blocked by Madaros Root 2.

## Root cause (re-diagnosed)

| Observation | Result |
|---|---|
| Tiny free-function module import | OK |
| `impl Epistemic { … }` methods **defined, never called** | OK |
| Any **method call site** (`Epistemic::measured`, `e.val()`) | **SEGV** in `lower_array: seed_begin` |
| `ulimit -s unlimited` | does **not** help (Root 2, not Root 1 stack) |
| Free-function call sites (`ep_measured`, `ep_val`) | OK under multi-module |

This is **not** “module has use deps” (knowledge is self-contained). It is Madaros
**method-call lowering** (Root 2 in `MADAROS_SRET_ROOT_SYNTHESIS_2026-06-20`).

Name clash: do **not** export a free function named `measure` — collides with the
language `Knowledge<T>` builtin (`found &Knowledge<f64>`).

## What landed

### 1. Free-function surface (`stdlib/epistemic/knowledge.sio`)

`ep_new`, `ep_measured`, `ep_certain`, `ep_val`, `ep_variance`, `ep_confidence`,
`ep_std`, `ep_add`/`sub`/`mul`/`div`, `ep_scale`/`shift`/`square`/`sqrt_ep`,
`ep_gate`, `ep_merge`, …

Thin `impl Epistemic` wrappers remain for lean_single / source-compat (definitions
only; call sites under Madaros still SEGV).

Selftests use **free functions only**; avoid `print(i64)` under Madaros.

### 2. Gates

```bash
bash scripts/epistemic_knowledge_madaros_e2e_gate.sh
# → EPISTEMIC_KNOWLEDGE_MADAROS_E2E_GATE_OK

bash scripts/epistemic_trust_gate.sh
# Section A includes knowledge free API
```

### 3. Residual

| Path | Status |
|---|---|
| Free-function import | **TRUSTWORTHY** under Madaros |
| Method-call import | **blocked** (Root 2) — `witness_import_knowledge_method` |
| `order_spread_exact` / `propagate` | still blocked (depend on further work) |
| Language `Knowledge<T>` / `measure` | separate from stdlib `Epistemic` |

## Measured (Madaros v0.80.0)

```
─── 32/32 PASS ───
ALL PASS
KNOW_IMPORT v=10 sum=14 prod=40 merge=4.827586
KNOWLEDGE_MADAROS_IMPORT_E2E_OK
```

## claims_not_made

- Method-call form under Madaros  
- Full Root 2 compiler fix  
- `propagate` / `order_spread_exact` native import  
- Language generic `Knowledge<T>`  
- Bedside / numpy  

## Priority next (compiler)

Root 2 method-call null-deref in `self-hosted/ir/lower.sio` (and related).
Then re-enable method form and promote `propagate`.
