# Associator GUM variance — Experiment 2

Numerical check of first-order GUM variance for  
`A = ‖(a·b)·c − a·(b·c)‖²` on the octonions (β-thread).

| | |
|---|---|
| Protocol | [`PROTOCOL.md`](PROTOCOL.md) |
| Design | [`docs/superpowers/specs/2026-07-25-associator-gum-variance-design.md`](../../docs/superpowers/specs/2026-07-25-associator-gum-variance-design.md) |
| Analysis | [`docs/research/variance_of_associator.md`](../../docs/research/variance_of_associator.md) |
| Program | [`associator_gum_variance.sio`](associator_gum_variance.sio) |
| Receipt | [`results/associator_gum_variance/receipt.v1.json`](../../results/associator_gum_variance/receipt.v1.json) |

```bash
bash experiments/associator_gum_variance/run_and_receipt.sh
bash scripts/ci/associator_gum_variance_gate.sh
```

**Primary Fano result (σ=0.05):** truth FO `0.16`, FD `0.16`, MC ~`0.158`, stepwise blind `0.08` (ratio 2×).  
Quaternion: `A=0`, FD var ~0. Experiment requires `PASS`.
