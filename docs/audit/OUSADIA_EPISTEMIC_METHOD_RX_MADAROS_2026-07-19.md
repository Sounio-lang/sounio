<!-- docs:meta
topic_id: repo.docs.audit.ousadia-epistemic-method-rx-madaros-2026-07-19
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.ousadia-epistemic-method-rx-madaros-2026-07-19
-->

# Ousadia: Epistemic methods → dose decision under default Madaros — 2026-07-19

## The claim (no alibi)

> Under **default Madaros multi-module import**, the flagship type is used **as methods**:
> `Epistemic::measured`, `.val()`, `.std()`, `.add()`, `.is_credible()` — and those
> results **drive ADMIT/ADJUST/REFUSE** on a vancomycin AUC/MIC scaffold.
> Over-confident prescribe paths remain compile-fail witnesses.

This is the vertical that was impossible until Root 2 multi-module (#1227).

## Measured (Madaros v0.80.0, no lean_single)

```
OUSADIA_RX ep_q=532.97 ep_std=79.95 typeA_u95=99.25 k95_table=2.776 decide=ADJUST
OUSADIA_RX ep_q_ren=2006.88 decide=REFUSE
OUSADIA_EPISTEMIC_METHOD_RX_CHAIN_OK
K95 k95=2.776 dof=4  (gum alone)
```

## Gate

```bash
bash scripts/ousadia_epistemic_method_rx_gate.sh
# → OUSADIA_EPISTEMIC_METHOD_RX_GATE_OK
```

## Residual (honest)

| Item | Status |
|---|---|
| Dual `use epistemic::gum` + `use epistemic::knowledge` | Madaros **E175** private preflight — Type-A U95 uses JCGM t₄=2.776 table; gum k95 smoked alone |
| Bedside / NONMEM / numpy | claims_not_made |

## Stack this sits on

1. GUM D1 k95 (#1198)
2. Knowledge free API (#1203)
3. Root 2 same-module methods (#1219)
4. Root 2 multi-module methods (#1227)
5. **This ousadia vertical** — methods → decision
