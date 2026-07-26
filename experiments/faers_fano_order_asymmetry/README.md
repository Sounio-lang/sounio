# FAERS Fano order-asymmetry (deeper track)

## Root cause of bad labels

`scripts/fetch_faers_drugbank.py` used a **disjoint** Fano plane (alternate CD table),
not `stdlib/medical/cyp450_fano.sio`. Fixed via `scripts/lib/cyp_fano_canon.py`.
`data/faers_drugbank.csv` relabeled (14 flips).

## Runs

| Version | What | Verdict |
|---|---|---|
| v2 | geometry-correct means + bootstrap | INCONCLUSIVE (diff +0.024) |
| **v3** | + weighted + **perm p one-sided** (B=5000) | **INCONCLUSIVE** (p≈0.37) |

```bash
./bin/souc run experiments/faers_fano_order_asymmetry/faers_fano_order_asymmetry_v3.sio
```

Not clinical. Proxy = start-order asymmetry only.
